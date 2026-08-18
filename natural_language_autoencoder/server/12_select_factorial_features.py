#!/usr/bin/env python3
"""Freeze B6+B4 feature selection before any AV/AR inference.

Selection uses only document-level SAE statistics from the factorial corpus's
``train`` split.  The held-out ``test`` split is evaluated and recorded but
never gates inclusion.  This prevents selecting features after reading NLA
explanations or round-trip scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from pilot_common import JumpReLUSAE


LEGACY_FEATURES = [1491, 5389, 239, 276]
LEGACY_EXCLUDE = [
    161,
    166,
    443,
    490,
    565,
    5389,
    1491,
    7508,
    2190,
    11642,
    276,
    239,
    4524,
    10497,
    8461,
    5146,
    717,
    13957,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return round(float(value), 8)
    if isinstance(value, np.integer):
        return int(value)
    return value


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Vectorized AUROC for [documents, features] arrays."""
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUROC requires positive and negative documents")
    greater = positive[:, None, :] > negative[None, :, :]
    equal = positive[:, None, :] == negative[None, :, :]
    return (greater.mean(axis=(0, 1)) + 0.5 * equal.mean(axis=(0, 1)))


def label_masks(
    label: str, doc_domains: np.ndarray, doc_languages: np.ndarray
) -> np.ndarray:
    axis, value = label.split(":", 1)
    if axis == "domain":
        return doc_domains == value
    if axis == "language":
        return doc_languages == value
    raise ValueError(f"unknown label axis {axis}")


def metrics_for_split(
    scores: np.ndarray,
    fires: np.ndarray,
    split_mask: np.ndarray,
    positive_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    positive = split_mask & positive_mask
    negative = split_mask & ~positive_mask
    pos_scores = scores[positive]
    neg_scores = scores[negative]
    pos_fires = fires[positive]
    neg_fires = fires[negative]

    pos_mean = pos_scores.mean(axis=0)
    neg_mean = neg_scores.mean(axis=0)
    pooled_std = scores[split_mask].std(axis=0)
    pos_sum = pos_scores.sum(axis=0)
    dominance = pos_scores.max(axis=0) / np.maximum(pos_sum, 1e-12)
    pos_support = pos_fires.sum(axis=0)
    neg_support = neg_fires.sum(axis=0)
    support_precision = pos_support / np.maximum(pos_support + neg_support, 1)
    return {
        "auc": binary_auc(pos_scores, neg_scores),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "effect": (pos_mean - neg_mean) / np.maximum(pooled_std, 1e-6),
        "raw_difference": pos_mean - neg_mean,
        "pos_support": pos_support,
        "neg_support": neg_support,
        "support_precision": support_precision,
        "dominance": dominance,
        "n_positive_docs": np.full(scores.shape[1], positive.sum()),
        "n_negative_docs": np.full(scores.shape[1], negative.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sae", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vectors-out", required=True, type=Path)
    parser.add_argument("--stats-out", required=True, type=Path)
    parser.add_argument("--max-semantic", type=int, default=24)
    parser.add_argument("--n-structural", type=int, default=8)
    parser.add_argument("--n-active-control", type=int, default=8)
    parser.add_argument("--n-gaussian", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    table = pq.read_table(args.activations)
    required_columns = {
        "activation_vector",
        "token",
        "token_id",
        "position",
        "doc_id",
        "prompt_id",
        "axis_domain",
        "axis_language",
        "split",
        "topic",
        "prompt",
    }
    missing = required_columns - set(table.column_names)
    if missing:
        raise KeyError(f"activation parquet missing columns {sorted(missing)}")

    x = np.asarray(
        table.column("activation_vector").combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError(f"invalid activation matrix shape={x.shape}")
    row_meta = {
        column: table.column(column).to_pylist()
        for column in required_columns
        if column != "activation_vector"
    }
    row_doc_ids = np.asarray(row_meta["doc_id"], dtype=np.int64)
    unique_doc_ids = np.unique(row_doc_ids)
    if not np.array_equal(unique_doc_ids, np.arange(len(unique_doc_ids))):
        raise ValueError("doc_id values must be contiguous from zero")

    doc_metadata = []
    for doc_id in unique_doc_ids:
        rows = np.flatnonzero(row_doc_ids == doc_id)
        values = {}
        for key in (
            "prompt_id",
            "axis_domain",
            "axis_language",
            "split",
            "topic",
            "prompt",
        ):
            unique_values = {row_meta[key][index] for index in rows}
            if len(unique_values) != 1:
                raise ValueError(f"doc {doc_id} has inconsistent {key}")
            values[key] = next(iter(unique_values))
        values["doc_id"] = int(doc_id)
        values["n_rows"] = int(len(rows))
        doc_metadata.append(values)

    doc_domains = np.asarray([row["axis_domain"] for row in doc_metadata])
    doc_languages = np.asarray([row["axis_language"] for row in doc_metadata])
    doc_splits = np.asarray([row["split"] for row in doc_metadata])
    train_docs = doc_splits == "train"
    test_docs = doc_splits == "test"
    if train_docs.sum() != test_docs.sum():
        raise ValueError("factorial design requires balanced train/test documents")

    # Freeze a document-balanced mean direction using discovery documents only.
    document_means = np.stack(
        [x[row_doc_ids == doc_id].mean(axis=0) for doc_id in unique_doc_ids]
    )
    m_hat = document_means[train_docs].mean(axis=0).astype(np.float64)
    m_hat /= np.linalg.norm(m_hat)
    target_norm = float(
        np.mean(
            [
                np.linalg.norm(x[row_doc_ids == doc_id], axis=1).mean()
                for doc_id in unique_doc_ids[train_docs]
            ]
        )
    )

    sae = JumpReLUSAE(str(args.sae), device="cuda")
    _, activations = sae(torch.from_numpy(x))
    activations = activations.float().cpu()
    width = int(activations.shape[1])

    # Document is the statistical unit: mean of the top three token activations.
    doc_scores = np.empty((len(unique_doc_ids), width), dtype=np.float32)
    doc_fires = np.empty((len(unique_doc_ids), width), dtype=bool)
    for doc_id in unique_doc_ids:
        rows = np.flatnonzero(row_doc_ids == doc_id)
        values = activations[rows]
        k = min(3, len(rows))
        doc_scores[doc_id] = torch.topk(values, k=k, dim=0).values.mean(0).numpy()
        doc_fires[doc_id] = (values > 0).any(dim=0).numpy()
    doc_scores_log = np.log1p(doc_scores)

    w_dec = sae.w_dec.float().cpu().numpy()
    decoder_norms = np.linalg.norm(w_dec, axis=1)
    decoder_unit = w_dec / np.maximum(decoder_norms[:, None], 1e-12)
    mean_alignment = decoder_unit @ m_hat
    projected_ratio = np.sqrt(np.maximum(0.0, 1.0 - mean_alignment**2))
    centered_directions = (
        decoder_unit - np.outer(mean_alignment, m_hat)
    ) / np.maximum(projected_ratio[:, None], 1e-12)

    domains = sorted(set(doc_domains))
    languages = sorted(set(doc_languages))
    labels = [f"domain:{value}" for value in domains] + [
        f"language:{value}" for value in languages
    ]
    train_metrics = {}
    test_metrics = {}
    for label in labels:
        positive = label_masks(label, doc_domains, doc_languages)
        train_metrics[label] = metrics_for_split(
            doc_scores_log, doc_fires, train_docs, positive
        )
        test_metrics[label] = metrics_for_split(
            doc_scores_log, doc_fires, test_docs, positive
        )

    excluded = set(LEGACY_EXCLUDE)
    selected_feature_ids: list[int] = []
    selected_centered: list[np.ndarray] = []
    selected_records: list[dict[str, Any]] = []

    def composite(metric: dict[str, np.ndarray]) -> np.ndarray:
        return (
            np.maximum(metric["auc"] - 0.5, 0.0)
            * np.maximum(metric["raw_difference"], 0.0)
            * (0.5 + 0.5 * metric["support_precision"])
        )

    rankings = {}
    quotas = {}
    for label in labels:
        metric = train_metrics[label]
        scores = composite(metric)
        rankings[label] = np.argsort(-scores, kind="stable")
        quotas[label] = 4 if label.startswith("domain:") else 3

    def tier_for(feature: int, label: str) -> str | None:
        metric = train_metrics[label]
        n_positive = int(metric["n_positive_docs"][feature])
        strict_support = 2 if label.startswith("domain:") else 3
        relaxed_support = 2
        if (
            metric["auc"][feature] >= 0.80
            and metric["pos_support"][feature] >= strict_support
            and metric["raw_difference"][feature] > 0
            and metric["dominance"][feature] <= 0.75
        ):
            return "strict"
        if (
            metric["auc"][feature] >= 0.70
            and metric["pos_support"][feature] >= min(
                relaxed_support, n_positive
            )
            and metric["raw_difference"][feature] > 0
            and metric["dominance"][feature] <= 0.85
        ):
            return "exploratory"
        return None

    def direction_is_duplicate(feature: int) -> bool:
        if not selected_centered:
            return False
        similarities = np.abs(
            np.stack(selected_centered) @ centered_directions[feature]
        )
        return bool(np.max(similarities) > 0.80)

    # Round-robin across labels prevents the first axis from exhausting slots.
    cursors = {label: 0 for label in labels}
    label_counts = {label: 0 for label in labels}
    while len(selected_feature_ids) < args.max_semantic:
        progressed = False
        for label in labels:
            if len(selected_feature_ids) >= args.max_semantic:
                break
            if label_counts[label] >= quotas[label]:
                continue
            ranking = rankings[label]
            chosen = None
            while cursors[label] < len(ranking):
                feature = int(ranking[cursors[label]])
                cursors[label] += 1
                if (
                    feature in excluded
                    or feature in selected_feature_ids
                    or projected_ratio[feature] < 0.20
                ):
                    continue
                tier = tier_for(feature, label)
                if tier is None or direction_is_duplicate(feature):
                    continue
                chosen = (feature, tier)
                break
            if chosen is None:
                continue
            feature, tier = chosen
            selected_feature_ids.append(feature)
            selected_centered.append(centered_directions[feature])
            label_counts[label] += 1
            selected_records.append(
                {
                    "group": "semantic_new",
                    "feature": feature,
                    "label": label,
                    "selection_tier": tier,
                }
            )
            progressed = True
        if not progressed:
            break

    strict_count = sum(
        record["selection_tier"] == "strict" for record in selected_records
    )

    used = set(selected_feature_ids) | excluded
    max_train_auc = np.max(
        np.stack([train_metrics[label]["auc"] for label in labels]), axis=0
    )
    train_doc_support = doc_fires[train_docs].sum(axis=0)
    train_mean_score = doc_scores_log[train_docs].mean(axis=0)

    # Structural/common controls: fire in nearly every discovery document but
    # are not strongly selective for any factorial label.
    structural_pool = np.flatnonzero(
        (train_doc_support >= max(1, train_docs.sum() - 2))
        & (max_train_auc <= 0.70)
        & (projected_ratio >= 0.20)
    )
    structural_pool = structural_pool[
        np.argsort(-train_mean_score[structural_pool], kind="stable")
    ]
    structural_ids = []
    for feature in structural_pool:
        feature = int(feature)
        if feature in used:
            continue
        structural_ids.append(feature)
        used.add(feature)
        if len(structural_ids) == args.n_structural:
            break

    # Active, nonselective controls matched greedily to semantic features on
    # firing support, magnitude, mean alignment, and decoder norm.
    active_pool = np.flatnonzero(
        (train_doc_support >= 1)
        & (max_train_auc <= 0.70)
        & (projected_ratio >= 0.20)
    )
    active_pool = np.asarray(
        [int(feature) for feature in active_pool if int(feature) not in used]
    )
    if len(active_pool) < args.n_active_control:
        raise RuntimeError("not enough active nonselective controls")

    variables = np.stack(
        [
            train_doc_support.astype(np.float64),
            train_mean_score.astype(np.float64),
            np.abs(mean_alignment),
            np.log1p(decoder_norms),
        ],
        axis=1,
    )
    scale = np.std(variables[active_pool], axis=0)
    scale[scale < 1e-8] = 1.0
    semantic_targets = selected_feature_ids[: args.n_active_control]
    active_ids = []
    active_matches = {}
    for target in semantic_targets:
        remaining = np.asarray(
            [feature for feature in active_pool if int(feature) not in used]
        )
        distances = np.sum(
            ((variables[remaining] - variables[target]) / scale) ** 2, axis=1
        )
        chosen = int(remaining[int(np.argmin(distances))])
        active_ids.append(chosen)
        active_matches[str(chosen)] = {
            "matched_semantic_feature": int(target),
            "standardized_squared_distance": float(np.min(distances)),
        }
        used.add(chosen)

    rng = np.random.default_rng(args.seed)
    gaussian = rng.standard_normal(
        (args.n_gaussian, x.shape[1])
    ).astype(np.float32)

    def top_contexts(feature: int, limit: int = 8) -> list[dict[str, Any]]:
        values = activations[:, feature].numpy()
        order = np.argsort(-values, kind="stable")
        contexts = []
        for row_index in order:
            value = float(values[row_index])
            if value <= 0:
                break
            contexts.append(
                {
                    "row_index": int(row_index),
                    "doc_id": int(row_doc_ids[row_index]),
                    "prompt_id": row_meta["prompt_id"][row_index],
                    "axis_domain": row_meta["axis_domain"][row_index],
                    "axis_language": row_meta["axis_language"][row_index],
                    "split": row_meta["split"][row_index],
                    "position": int(row_meta["position"][row_index]),
                    "token": row_meta["token"][row_index],
                    "activation": value,
                }
            )
            if len(contexts) == limit:
                break
        return contexts

    def feature_record(
        feature: int,
        group: str,
        label: str | None,
        selection_tier: str | None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "group": group,
            "feature": int(feature),
            "label": label,
            "selection_tier": selection_tier,
            "decoder_norm": float(decoder_norms[feature]),
            "mean_alignment": float(mean_alignment[feature]),
            "projected_norm_ratio": float(projected_ratio[feature]),
            "train_doc_support": int(train_doc_support[feature]),
            "top_contexts": top_contexts(feature),
        }
        if label is not None:
            train = train_metrics[label]
            test = test_metrics[label]
            for prefix, metric in (("train", train), ("test", test)):
                record[prefix] = {
                    key: float(metric[key][feature])
                    for key in (
                        "auc",
                        "pos_mean",
                        "neg_mean",
                        "effect",
                        "raw_difference",
                        "pos_support",
                        "neg_support",
                        "support_precision",
                        "dominance",
                        "n_positive_docs",
                        "n_negative_docs",
                    )
                }
        positive_test_rows = np.flatnonzero(
            np.asarray(row_meta["split"]) == "test"
        )
        if label is not None:
            doc_positive = label_masks(label, doc_domains, doc_languages)
            positive_test_rows = np.asarray(
                [
                    index
                    for index in positive_test_rows
                    if doc_positive[row_doc_ids[index]]
                ],
                dtype=np.int64,
            )
        row_values = activations[positive_test_rows, feature].numpy()
        high_local = int(np.argmax(row_values))
        low_local = int(np.argmin(row_values))
        record["carrier_high"] = {
            "row_index": int(positive_test_rows[high_local]),
            "activation": float(row_values[high_local]),
        }
        record["carrier_low"] = {
            "row_index": int(positive_test_rows[low_local]),
            "activation": float(row_values[low_local]),
        }
        return record

    records = []
    for selection in selected_records:
        records.append(
            feature_record(
                selection["feature"],
                selection["group"],
                selection["label"],
                selection["selection_tier"],
            )
        )
    for feature in LEGACY_FEATURES:
        records.append(
            feature_record(feature, "semantic_legacy", None, "legacy_replication")
        )
    for feature in structural_ids:
        records.append(feature_record(feature, "structural", None, "control"))
    for feature in active_ids:
        record = feature_record(
            feature, "active_nonselective", None, "matched_control"
        )
        record["match"] = active_matches[str(feature)]
        records.append(record)
    for index in range(args.n_gaussian):
        direction = gaussian[index].astype(np.float64)
        direction /= np.linalg.norm(direction)
        alignment = float(direction @ m_hat)
        records.append(
            {
                "group": "gaussian",
                "feature": -1 - index,
                "label": None,
                "selection_tier": "control",
                "decoder_norm": float(np.linalg.norm(gaussian[index])),
                "mean_alignment": alignment,
                "projected_norm_ratio": math.sqrt(max(0.0, 1 - alignment**2)),
                "train_doc_support": 0,
                "top_contexts": [],
            }
        )

    direction_ids = []
    direction_groups = []
    direction_labels = []
    directions = []
    for record in records:
        feature = int(record["feature"])
        direction_ids.append(feature)
        direction_groups.append(record["group"])
        direction_labels.append(record["label"] or "")
        directions.append(
            gaussian[-1 - feature] if feature < 0 else w_dec[feature]
        )

    # Top candidate tables are sufficient for human audit; dense per-feature
    # metrics are saved losslessly in stats-out.
    top_candidates = {}
    for label in labels:
        candidates = []
        metric = train_metrics[label]
        for feature in rankings[label][:200]:
            feature = int(feature)
            candidates.append(
                {
                    "feature": feature,
                    "score": float(composite(metric)[feature]),
                    "tier": tier_for(feature, label),
                    "excluded_legacy": feature in excluded,
                    "projected_norm_ratio": float(projected_ratio[feature]),
                    "train_auc": float(metric["auc"][feature]),
                    "test_auc": float(test_metrics[label]["auc"][feature]),
                    "train_pos_support": int(metric["pos_support"][feature]),
                    "test_pos_support": int(
                        test_metrics[label]["pos_support"][feature]
                    ),
                }
            )
        top_candidates[label] = candidates

    payload = {
        "schema_version": 1,
        "experiment": "B6+B4 factorial semantic feature selection",
        "status": "selection_frozen_before_AV_AR",
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": sha256_file(args.activations),
            "sae": str(args.sae),
            "sae_params_sha256": sha256_file(args.sae / "params.safetensors"),
            "n_rows": int(len(x)),
            "d_model": int(x.shape[1]),
            "width": width,
            "n_documents": int(len(unique_doc_ids)),
        },
        "protocol": {
            "statistical_unit": "document",
            "document_score": "mean of top-3 token JumpReLU activations",
            "selection_split": "train only",
            "heldout_split": "test; recorded but never used as a gate",
            "centering": (
                "unit direction of the mean of per-document mean activations "
                "over train documents only"
            ),
            "strict_gate": (
                "AUROC>=0.80, positive document support>=2 domain or >=3 "
                "language, positive effect, dominance<=0.75"
            ),
            "exploratory_gate": (
                "AUROC>=0.70, positive support>=2, positive effect, "
                "dominance<=0.85"
            ),
            "direction_dedup": "abs centered decoder cosine <= 0.80",
            "projected_norm_gate": "||P_perp(w)||/||w|| >= 0.20",
            "legacy_feature_ids_excluded_from_new_selection": LEGACY_EXCLUDE,
            "seed": args.seed,
        },
        "dataset": doc_metadata,
        "summary": {
            "labels": labels,
            "requested_new_semantic": args.max_semantic,
            "selected_new_semantic": len(selected_feature_ids),
            "strict_new_semantic": strict_count,
            "exploratory_new_semantic": len(selected_feature_ids) - strict_count,
            "label_counts": label_counts,
            "structural_controls": len(structural_ids),
            "active_nonselective_controls": len(active_ids),
            "gaussian_controls": args.n_gaussian,
            "total_directions": len(records),
            "target_norm": target_norm,
        },
        "selected_directions": records,
        "top_candidates_by_label": top_candidates,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(to_builtin(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.vectors_out,
        x=x,
        row_doc_ids=row_doc_ids,
        m_hat=m_hat.astype(np.float32),
        target_norm=np.asarray(target_norm, dtype=np.float32),
        direction_ids=np.asarray(direction_ids, dtype=np.int64),
        direction_groups=np.asarray(direction_groups),
        direction_labels=np.asarray(direction_labels),
        directions=np.stack(directions).astype(np.float32),
        selected_feature_activations=activations[
            :, [feature for feature in direction_ids if feature >= 0]
        ].numpy(),
    )
    np.savez_compressed(
        args.stats_out,
        feature_ids=np.arange(width, dtype=np.int64),
        labels=np.asarray(labels),
        doc_ids=unique_doc_ids,
        doc_domains=doc_domains,
        doc_languages=doc_languages,
        doc_splits=doc_splits,
        doc_scores=doc_scores,
        doc_fires=doc_fires,
        train_auc=np.stack(
            [train_metrics[label]["auc"] for label in labels]
        ).astype(np.float32),
        test_auc=np.stack(
            [test_metrics[label]["auc"] for label in labels]
        ).astype(np.float32),
        train_effect=np.stack(
            [train_metrics[label]["effect"] for label in labels]
        ).astype(np.float32),
        test_effect=np.stack(
            [test_metrics[label]["effect"] for label in labels]
        ).astype(np.float32),
        train_pos_support=np.stack(
            [train_metrics[label]["pos_support"] for label in labels]
        ).astype(np.int16),
        test_pos_support=np.stack(
            [test_metrics[label]["pos_support"] for label in labels]
        ).astype(np.int16),
        mean_alignment=mean_alignment.astype(np.float32),
        projected_ratio=projected_ratio.astype(np.float32),
        decoder_norms=decoder_norms.astype(np.float32),
    )
    print("FACTORIAL_SELECTION_COMPLETE")
    print(
        json.dumps(
            {
                "new_semantic": len(selected_feature_ids),
                "strict": strict_count,
                "exploratory": len(selected_feature_ids) - strict_count,
                "label_counts": label_counts,
                "structural": len(structural_ids),
                "active_control": len(active_ids),
                "gaussian": args.n_gaussian,
                "total": len(records),
            }
        )
    )
    print(f"wrote -> {args.out} + {args.vectors_out} + {args.stats_out}")


if __name__ == "__main__":
    main()
