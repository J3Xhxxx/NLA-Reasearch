#!/usr/bin/env python3
"""Freeze discovery-only SAE features for the C1 confirmatory experiment.

This selector intentionally cannot accept held-out activations.  It validates
the 24 x 4 discovery design, computes document-balanced top-3 JumpReLU
statistics in chunks, resolves and verifies the frozen prior-feature denylist,
and writes a selection asset before any AV, AR, or held-out computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch

from pilot_common import JumpReLUSAE


N_CONCEPTS = 24
DOCS_PER_CONCEPT = 4
TOP_K = 3
MAX_PER_CONCEPT = 4
MIN_SELECTED_FEATURES = 60
MIN_POPULATED_CONCEPTS = 18
MIN_COMPLETE_HARD_NEGATIVE_PAIRS = 9
MAX_CENTERED_ABS_COSINE = 0.80
MIN_PROJECTED_RATIO = 0.20
REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path]
    # Locally the repository contains ``server/``; on AutoDL the contents of
    # that folder are deployed directly to ``.../nla_compare``.
    if path.parts and path.parts[0] == "server":
        candidates.append(Path(__file__).resolve().parent / Path(*path.parts[1:]))
    existing = list(
        dict.fromkeys(
            candidate.resolve()
            for candidate in candidates
            if candidate.exists()
        )
    )
    if len(existing) != 1:
        raise FileNotFoundError(
            f"cannot resolve exactly one frozen source for {path}: {candidates}"
        )
    return existing[0]


def feature_id(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{location} must be an integer feature ID")
    if value < 0:
        raise ValueError(f"{location} must be nonnegative")
    return value


def resolve_denylist(path: Path) -> tuple[set[int], dict[str, Any]]:
    """Resolve prior JSON candidates plus the manifest's frozen legacy IDs."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = int(manifest["expected_unique_resolved_ids"])
    resolved: set[int] = set()
    source_audit = []
    saw_prior_json = False
    saw_legacy_source = False

    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("denylist must declare exactly two frozen sources")
    for source in sources:
        source_path = resolve_repo_path(source["path"])
        actual_sha = sha256_file(source_path)
        expected_sha = str(source["sha256"]).lower()
        if actual_sha != expected_sha:
            raise ValueError(
                f"denylist source SHA mismatch for {source_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        before = len(resolved)
        if "json_paths" in source:
            expected_paths = {
                "selected_directions[*].feature",
                "top_candidates_by_label.*[*].feature",
            }
            if set(source["json_paths"]) != expected_paths:
                raise ValueError("unexpected prior-selection JSON paths")
            prior = json.loads(source_path.read_text(encoding="utf-8"))
            for index, row in enumerate(prior["selected_directions"]):
                value = row.get("feature")
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    resolved.add(value)
                elif not (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value < 0
                ):
                    raise TypeError(
                        "invalid feature at "
                        f"selected_directions[{index}].feature"
                    )
            for label, rows in prior["top_candidates_by_label"].items():
                if not isinstance(rows, list):
                    raise TypeError(f"top candidate table {label} is not a list")
                for index, row in enumerate(rows):
                    resolved.add(
                        feature_id(
                            row.get("feature"),
                            f"top_candidates_by_label.{label}[{index}].feature",
                        )
                    )
            saw_prior_json = True
        elif source.get("field") == "LEGACY_EXCLUDE":
            legacy = manifest.get("legacy_exclude")
            if not isinstance(legacy, list) or not legacy:
                raise ValueError("denylist legacy_exclude must be a nonempty list")
            for index, value in enumerate(legacy):
                resolved.add(feature_id(value, f"legacy_exclude[{index}]"))
            saw_legacy_source = True
        else:
            raise ValueError(f"unsupported denylist source declaration: {source}")
        source_audit.append(
            {
                "path": str(source_path),
                "sha256": actual_sha,
                "new_unique_ids_after_union": len(resolved) - before,
            }
        )

    if not saw_prior_json or not saw_legacy_source:
        raise ValueError("denylist did not resolve both prior JSON and legacy IDs")
    if len(resolved) != expected:
        raise ValueError(
            f"resolved denylist count mismatch: expected {expected}, "
            f"got {len(resolved)}"
        )
    return resolved, {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "expected_unique_ids": expected,
        "resolved_unique_ids": len(resolved),
        "sources": source_audit,
    }


def read_design(
    activation_path: Path, concept_spec: dict[str, Any]
) -> tuple[dict[str, list[Any]], np.ndarray, list[dict[str, Any]], list[str]]:
    required = {
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
        "prompt_sha256",
    }
    parquet = pq.ParquetFile(activation_path)
    columns = set(parquet.schema_arrow.names)
    missing = required - columns
    if missing:
        raise KeyError(f"activation parquet missing columns {sorted(missing)}")
    metadata_table = pq.read_table(activation_path, columns=sorted(required))
    row_meta = {
        name: metadata_table.column(name).combine_chunks().to_pylist()
        for name in required
    }
    n_rows = len(row_meta["doc_id"])
    if n_rows != parquet.metadata.num_rows or n_rows == 0:
        raise ValueError("invalid activation metadata row count")

    row_doc_ids = np.asarray(row_meta["doc_id"], dtype=np.int64)
    unique_doc_ids = np.unique(row_doc_ids)
    expected_docs = N_CONCEPTS * DOCS_PER_CONCEPT
    if not np.array_equal(unique_doc_ids, np.arange(expected_docs)):
        raise ValueError(
            f"discovery doc_id must be contiguous 0..{expected_docs - 1}"
        )
    if set(row_meta["split"]) != {"train"}:
        raise ValueError(
            "selector accepts discovery-only parquet with every split='train'"
        )
    if set(row_meta["axis_language"]) != {"en"}:
        raise ValueError("C1 confirmatory discovery parquet must be English-only")
    positions = np.asarray(row_meta["position"], dtype=np.int64)
    if np.any(positions < 50):
        raise ValueError("discovery activations include positions below 50")

    concepts = concept_spec.get("concepts")
    if not isinstance(concepts, list) or len(concepts) != N_CONCEPTS:
        raise ValueError(f"concept spec must contain exactly {N_CONCEPTS} concepts")
    labels = [str(concept["id"]) for concept in concepts]
    if len(set(labels)) != N_CONCEPTS:
        raise ValueError("concept IDs must be unique")
    concept_by_id = {str(concept["id"]): concept for concept in concepts}
    if set(row_meta["topic"]) != set(labels):
        raise ValueError("activation topics do not exactly match frozen concepts")

    doc_metadata: list[dict[str, Any]] = []
    topic_counts = {label: 0 for label in labels}
    seen_prompt_ids: set[str] = set()
    for doc_id in unique_doc_ids:
        indices = np.flatnonzero(row_doc_ids == doc_id)
        if len(indices) < TOP_K:
            raise ValueError(f"doc {doc_id} has fewer than {TOP_K} eligible tokens")
        values: dict[str, Any] = {}
        for key in (
            "prompt_id",
            "axis_domain",
            "axis_language",
            "split",
            "topic",
            "prompt",
            "prompt_sha256",
        ):
            unique_values = {row_meta[key][index] for index in indices}
            if len(unique_values) != 1:
                raise ValueError(f"doc {doc_id} has inconsistent {key}")
            values[key] = next(iter(unique_values))

        prompt_id = str(values["prompt_id"])
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"prompt_id {prompt_id} appears in multiple documents")
        seen_prompt_ids.add(prompt_id)
        prompt = str(values["prompt"])
        actual_prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if values["prompt_sha256"] != actual_prompt_sha:
            raise ValueError(f"prompt SHA mismatch in doc {doc_id}")
        doc_positions = np.sort(positions[indices])
        if not np.array_equal(
            doc_positions, np.arange(50, int(doc_positions[-1]) + 1)
        ):
            raise ValueError(
                f"doc {doc_id} does not contain every eligible position >=50"
            )

        label = str(values["topic"])
        concept = concept_by_id[label]
        if values["axis_domain"] != concept["superdomain"]:
            raise ValueError(f"doc {doc_id} superdomain disagrees with concept spec")
        topic_counts[label] += 1
        doc_metadata.append(
            {
                "doc_id": int(doc_id),
                "prompt_id": prompt_id,
                "topic": label,
                "superdomain": str(values["axis_domain"]),
                "axis_language": "en",
                "split": "train",
                "prompt": prompt,
                "prompt_sha256": actual_prompt_sha,
                "n_rows": int(len(indices)),
                "first_position": int(doc_positions[0]),
                "last_position": int(doc_positions[-1]),
            }
        )
    wrong_counts = {
        label: count
        for label, count in topic_counts.items()
        if count != DOCS_PER_CONCEPT
    }
    if wrong_counts:
        raise ValueError(
            f"each concept requires {DOCS_PER_CONCEPT} discovery docs: "
            f"{wrong_counts}"
        )
    return row_meta, row_doc_ids, doc_metadata, labels


def iter_activation_chunks(
    path: Path, batch_size: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["activation_vector", "doc_id"]
    ):
        x = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
        doc_ids = np.asarray(batch.column(1).to_pylist(), dtype=np.int64)
        if x.ndim != 2 or len(x) != len(doc_ids):
            raise ValueError(f"invalid activation chunk shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("activation parquet contains non-finite values")
        yield x, doc_ids


@torch.inference_mode()
def full_sae_activations(sae: JumpReLUSAE, x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(x)).to(
        sae.device, sae.dtype
    )
    pre = tensor @ sae.w_enc + sae.b_enc
    activations = torch.relu(pre) * (pre > sae.threshold)
    result = activations.float().cpu().numpy()
    del tensor, pre, activations
    return result


@torch.inference_mode()
def selected_sae_activations(
    sae: JumpReLUSAE, x: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    if len(selected) == 0:
        return np.empty((len(x), 0), dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(x)).to(
        sae.device, sae.dtype
    )
    selected_gpu = torch.as_tensor(selected, device=sae.device, dtype=torch.long)
    w_enc = sae.w_enc.index_select(1, selected_gpu)
    b_enc = sae.b_enc.index_select(0, selected_gpu)
    threshold = sae.threshold.index_select(0, selected_gpu)
    pre = tensor @ w_enc + b_enc
    activations = torch.relu(pre) * (pre > threshold)
    result = activations.float().cpu().numpy()
    del tensor, selected_gpu, w_enc, b_enc, threshold, pre, activations
    return result


def merge_top_k(store: np.ndarray, doc_id: int, values: np.ndarray) -> None:
    if values.shape[1] != store.shape[2]:
        raise ValueError("feature width changed while merging document scores")
    if values.shape[0] > TOP_K:
        split = values.shape[0] - TOP_K
        local = np.partition(values, split, axis=0)[split:]
    else:
        local = values
    merged = np.concatenate((store[doc_id], local), axis=0)
    split = merged.shape[0] - TOP_K
    store[doc_id] = np.partition(merged, split, axis=0)[split:]


def first_pass_statistics(
    path: Path,
    sae: JumpReLUSAE,
    row_doc_ids: np.ndarray,
    n_documents: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    width = int(sae.width)
    d_model = int(sae.d_model)
    top_values = np.zeros(
        (n_documents, TOP_K, width), dtype=np.float32
    )
    doc_vector_sums = np.zeros((n_documents, d_model), dtype=np.float64)
    doc_norm_sums = np.zeros(n_documents, dtype=np.float64)
    doc_row_counts = np.zeros(n_documents, dtype=np.int64)
    x_chunks: list[np.ndarray] = []
    observed_doc_chunks: list[np.ndarray] = []

    processed = 0
    for chunk_index, (x, doc_ids) in enumerate(
        iter_activation_chunks(path, batch_size), start=1
    ):
        if x.shape[1] != d_model:
            raise ValueError(
                f"activation d_model={x.shape[1]} differs from SAE d_model={d_model}"
            )
        if np.any((doc_ids < 0) | (doc_ids >= n_documents)):
            raise ValueError("chunk contains out-of-range doc_id")
        acts = full_sae_activations(sae, x)
        for doc_id in np.unique(doc_ids):
            mask = doc_ids == doc_id
            merge_top_k(top_values, int(doc_id), acts[mask])
            doc_vector_sums[doc_id] += x[mask].sum(axis=0, dtype=np.float64)
            doc_norm_sums[doc_id] += np.linalg.norm(
                x[mask].astype(np.float64), axis=1
            ).sum()
            doc_row_counts[doc_id] += int(mask.sum())
        x_chunks.append(x)
        observed_doc_chunks.append(doc_ids)
        processed += len(x)
        print(f"[selection pass 1] chunk={chunk_index} rows={processed}")

    observed_doc_ids = np.concatenate(observed_doc_chunks)
    if not np.array_equal(observed_doc_ids, row_doc_ids):
        raise ValueError("chunked doc_id order differs from parquet metadata")
    if np.any(doc_row_counts < TOP_K):
        raise ValueError("at least one document lacks three activation rows")
    document_means = doc_vector_sums / doc_row_counts[:, None]
    mean_direction = document_means.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_direction))
    if not np.isfinite(mean_norm) or mean_norm <= 0:
        raise ValueError("invalid discovery mean direction")
    m_hat = mean_direction / mean_norm
    target_norm = float(np.mean(doc_norm_sums / doc_row_counts))
    if not np.isfinite(target_norm) or target_norm <= 0:
        raise ValueError("invalid discovery target norm")

    doc_scores = top_values.mean(axis=1)
    doc_fires = np.any(top_values > 0, axis=1)
    x_all = np.concatenate(x_chunks)
    if len(x_all) != len(row_doc_ids):
        raise ValueError("activation row count changed during first pass")
    return doc_scores, doc_fires, m_hat, target_norm, x_all


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUROC requires positive and negative documents")
    greater = positive[:, None, :] > negative[None, :, :]
    equal = positive[:, None, :] == negative[None, :, :]
    return greater.mean(axis=(0, 1)) + 0.5 * equal.mean(axis=(0, 1))


def label_metrics(
    scores: np.ndarray, fires: np.ndarray, positive: np.ndarray
) -> dict[str, np.ndarray]:
    negative = ~positive
    pos_scores = scores[positive]
    neg_scores = scores[negative]
    pos_fires = fires[positive]
    neg_fires = fires[negative]
    pos_mean = pos_scores.mean(axis=0)
    neg_mean = neg_scores.mean(axis=0)
    pooled_std = scores.std(axis=0)
    pos_sum = pos_scores.sum(axis=0)
    pos_support = pos_fires.sum(axis=0)
    neg_support = neg_fires.sum(axis=0)
    support_precision = pos_support / np.maximum(
        pos_support + neg_support, 1
    )
    return {
        "auc": binary_auc(pos_scores, neg_scores),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "effect": (pos_mean - neg_mean) / np.maximum(pooled_std, 1e-6),
        "raw_difference": pos_mean - neg_mean,
        "pos_support": pos_support,
        "neg_support": neg_support,
        "support_precision": support_precision,
        "dominance": pos_scores.max(axis=0) / np.maximum(pos_sum, 1e-12),
        "n_positive_docs": np.full(scores.shape[1], int(positive.sum())),
        "n_negative_docs": np.full(scores.shape[1], int(negative.sum())),
    }


def composite(metric: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.maximum(metric["auc"] - 0.5, 0.0)
        * np.maximum(metric["raw_difference"], 0.0)
        * (0.5 + 0.5 * metric["support_precision"])
    )


def second_pass_selected(
    path: Path,
    sae: JumpReLUSAE,
    selected: np.ndarray,
    row_doc_ids: np.ndarray,
    n_documents: int,
    expected_doc_scores: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    observed_doc_chunks: list[np.ndarray] = []
    selected_top = np.zeros(
        (n_documents, TOP_K, len(selected)), dtype=np.float32
    )
    processed = 0
    for chunk_index, (x, doc_ids) in enumerate(
        iter_activation_chunks(path, batch_size), start=1
    ):
        acts = selected_sae_activations(sae, x, selected)
        if len(selected):
            for doc_id in np.unique(doc_ids):
                merge_top_k(selected_top, int(doc_id), acts[doc_ids == doc_id])
        chunks.append(acts)
        observed_doc_chunks.append(doc_ids)
        processed += len(x)
        print(f"[selection pass 2] chunk={chunk_index} rows={processed}")
    observed_doc_ids = np.concatenate(observed_doc_chunks)
    if not np.array_equal(observed_doc_ids, row_doc_ids):
        raise ValueError("second-pass doc_id order differs from metadata")
    result = np.concatenate(chunks)
    if len(selected):
        observed_scores = selected_top.mean(axis=1)
        expected = expected_doc_scores[:, selected]
        if not np.allclose(observed_scores, expected, rtol=1e-2, atol=1e-1):
            largest = float(np.max(np.abs(observed_scores - expected)))
            raise ValueError(
                "selected-only second pass disagrees with selection pass; "
                f"largest absolute document-score difference={largest}"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sae", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument(
        "--spec", "--concept-spec", dest="spec", required=True, type=Path
    )
    parser.add_argument("--denylist", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vectors-out", required=True, type=Path)
    parser.add_argument("--stats-out", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-candidates", type=int, default=200)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.top_candidates <= 0:
        raise ValueError("--top-candidates must be positive")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("documents_per_concept") != {"train": 4, "test": 2}:
        raise ValueError("concept spec discovery design is not frozen 4+2")
    concepts = spec["concepts"]
    concept_by_id = {str(concept["id"]): concept for concept in concepts}
    if len(concept_by_id) != N_CONCEPTS:
        raise ValueError("concept IDs must be unique")
    reciprocal_pairs: set[tuple[str, str]] = set()
    for concept in concepts:
        label = str(concept["id"])
        negative_label = str(concept["hard_negative_id"])
        negative = concept_by_id.get(negative_label)
        if (
            negative is None
            or str(negative["hard_negative_id"]) != label
            or negative["superdomain"] != concept["superdomain"]
        ):
            raise ValueError(
                f"hard negative mapping is not reciprocal within domain: {label}"
            )
        reciprocal_pairs.add(tuple(sorted((label, negative_label))))
    if len(reciprocal_pairs) != N_CONCEPTS // 2:
        raise ValueError("frozen design must contain 12 reciprocal concept pairs")
    row_meta, row_doc_ids, dataset, labels = read_design(
        args.activations, spec
    )
    doc_topics = np.asarray([row["topic"] for row in dataset])

    denied, deny_audit = resolve_denylist(args.denylist)
    activation_sha = sha256_file(args.activations)
    spec_sha = sha256_file(args.spec)
    sae_params = args.sae / "params.safetensors"
    sae_sha = sha256_file(sae_params)

    sae = JumpReLUSAE(str(args.sae), device="cuda", dtype=torch.float32)
    width = int(sae.width)
    if any(value >= width for value in denied):
        raise ValueError("denylist contains feature outside SAE width")

    doc_scores, doc_fires, m_hat, target_norm, x = first_pass_statistics(
        args.activations,
        sae,
        row_doc_ids,
        len(dataset),
        args.chunk_size,
    )
    doc_scores_log = np.log1p(doc_scores)

    train_metrics: dict[str, dict[str, np.ndarray]] = {}
    composite_by_label = []
    for label in labels:
        metric = label_metrics(doc_scores, doc_fires, doc_topics == label)
        train_metrics[label] = metric
        composite_by_label.append(composite(metric))
    composite_matrix = np.stack(composite_by_label)
    best_label_index = np.argmax(composite_matrix, axis=0)
    best_composite = composite_matrix[
        best_label_index, np.arange(width, dtype=np.int64)
    ]

    w_dec = sae.w_dec.float().cpu().numpy()
    decoder_norms = np.linalg.norm(w_dec.astype(np.float64), axis=1)
    m_hat32 = m_hat.astype(np.float32)
    decoder_dot_mean = w_dec @ m_hat32
    mean_alignment = decoder_dot_mean / np.maximum(decoder_norms, 1e-12)
    projected_ratio = np.sqrt(
        np.maximum(0.0, 1.0 - np.square(mean_alignment))
    )
    projected_ratio[decoder_norms <= 1e-12] = 0.0

    denied_mask = np.zeros(width, dtype=bool)
    denied_mask[np.asarray(sorted(denied), dtype=np.int64)] = True

    def gate_tier(feature: int, label: str) -> str | None:
        metric = train_metrics[label]
        common = (
            metric["pos_support"][feature] >= 3
            and metric["raw_difference"][feature] > 0
            and projected_ratio[feature] >= MIN_PROJECTED_RATIO
            and decoder_norms[feature] > 1e-12
        )
        if (
            common
            and metric["auc"][feature] >= 0.85
            and metric["dominance"][feature] <= 0.70
        ):
            return "strict"
        if (
            common
            and metric["auc"][feature] >= 0.75
            and metric["dominance"][feature] <= 0.85
        ):
            return "relaxed"
        return None

    rankings: dict[str, np.ndarray] = {}
    feature_ids = np.arange(width, dtype=np.int64)
    for label_index, label in enumerate(labels):
        assigned = feature_ids[best_label_index == label_index]
        rankings[label] = assigned[
            np.lexsort((assigned, -composite_matrix[label_index, assigned]))
        ]

    selected_ids: list[int] = []
    selected_set: set[int] = set()
    selected_centered: list[np.ndarray] = []
    selected_tiers: list[str] = []
    selected_labels: list[str] = []
    label_counts = {label: 0 for label in labels}

    def centered_direction(feature: int) -> np.ndarray:
        direction = w_dec[feature].astype(np.float64)
        direction /= decoder_norms[feature]
        direction -= float(direction @ m_hat) * m_hat
        norm = float(np.linalg.norm(direction))
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError(f"feature {feature} has invalid centered direction")
        return direction / norm

    def duplicate(direction: np.ndarray) -> bool:
        if not selected_centered:
            return False
        similarities = np.abs(np.stack(selected_centered) @ direction)
        return bool(np.max(similarities) > MAX_CENTERED_ABS_COSINE)

    # Exhaust the strict tier round-robin before considering any relaxed row.
    for requested_tier in ("strict", "relaxed"):
        cursors = {label: 0 for label in labels}
        while True:
            progressed = False
            for label in labels:
                if label_counts[label] >= MAX_PER_CONCEPT:
                    continue
                ranking = rankings[label]
                chosen: tuple[int, np.ndarray] | None = None
                while cursors[label] < len(ranking):
                    feature = int(ranking[cursors[label]])
                    cursors[label] += 1
                    if feature in denied or feature in selected_set:
                        continue
                    if gate_tier(feature, label) != requested_tier:
                        continue
                    direction = centered_direction(feature)
                    if duplicate(direction):
                        continue
                    chosen = feature, direction
                    break
                if chosen is None:
                    continue
                feature, direction = chosen
                selected_ids.append(feature)
                selected_set.add(feature)
                selected_centered.append(direction)
                selected_tiers.append(requested_tier)
                selected_labels.append(label)
                label_counts[label] += 1
                progressed = True
            if not progressed:
                break

    selected = np.asarray(selected_ids, dtype=np.int64)
    populated_concepts = sum(count > 0 for count in label_counts.values())
    complete_hard_negative_pairs = sum(
        label_counts[left] > 0 and label_counts[right] > 0
        for left, right in reciprocal_pairs
    )
    gate_pass = (
        len(selected) >= MIN_SELECTED_FEATURES
        and populated_concepts >= MIN_POPULATED_CONCEPTS
        and complete_hard_negative_pairs >= MIN_COMPLETE_HARD_NEGATIVE_PAIRS
    )
    status = (
        "selection_frozen_before_AV_AR"
        if gate_pass
        else "selection_failed_stop_no_AV_AR"
    )

    selected_token_activations = second_pass_selected(
        args.activations,
        sae,
        selected,
        row_doc_ids,
        len(dataset),
        doc_scores,
        args.chunk_size,
    )

    metric_keys = (
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
    records = []
    for feature, label, tier in zip(
        selected_ids, selected_labels, selected_tiers
    ):
        concept = concept_by_id[label]
        metric = train_metrics[label]
        records.append(
            {
                "group": "semantic_new",
                "feature": feature,
                "label": label,
                "concept_title": concept["title"],
                "concept_summary": concept["summary"],
                "superdomain": concept["superdomain"],
                "hard_negative_id": concept["hard_negative_id"],
                "selection_tier": tier,
                "composite_score": float(
                    composite_matrix[labels.index(label), feature]
                ),
                "decoder_norm": float(decoder_norms[feature]),
                "mean_alignment": float(mean_alignment[feature]),
                "projected_norm_ratio": float(projected_ratio[feature]),
                "train": {
                    key: (
                        int(metric[key][feature])
                        if key
                        in {
                            "pos_support",
                            "neg_support",
                            "n_positive_docs",
                            "n_negative_docs",
                        }
                        else float(metric[key][feature])
                    )
                    for key in metric_keys
                },
            }
        )

    top_candidates: dict[str, list[dict[str, Any]]] = {}
    for label_index, label in enumerate(labels):
        rows = []
        for feature_value in rankings[label][: args.top_candidates]:
            feature = int(feature_value)
            metric = train_metrics[label]
            rows.append(
                {
                    "feature": feature,
                    "composite_score": float(
                        composite_matrix[label_index, feature]
                    ),
                    "best_label": labels[int(best_label_index[feature])],
                    "eligible_tier": gate_tier(feature, label),
                    "excluded_prior": feature in denied,
                    "selected": feature in selected_set,
                    "projected_norm_ratio": float(projected_ratio[feature]),
                    "train_auc": float(metric["auc"][feature]),
                    "train_raw_difference": float(
                        metric["raw_difference"][feature]
                    ),
                    "train_pos_support": int(
                        metric["pos_support"][feature]
                    ),
                    "train_support_precision": float(
                        metric["support_precision"][feature]
                    ),
                    "train_dominance": float(
                        metric["dominance"][feature]
                    ),
                }
            )
        top_candidates[label] = rows

    strict_count = selected_tiers.count("strict")
    relaxed_count = selected_tiers.count("relaxed")
    payload = {
        "schema_version": 1,
        "experiment": "C1 confirmatory discovery-only SAE feature selection",
        "status": status,
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": activation_sha,
            "concept_spec": str(args.spec),
            "concept_spec_sha256": spec_sha,
            "denylist": deny_audit,
            "sae": str(args.sae),
            "sae_params_sha256": sae_sha,
            "n_rows": int(len(row_doc_ids)),
            "d_model": int(x.shape[1]),
            "width": width,
            "n_documents": len(dataset),
        },
        "protocol": {
            "statistical_unit": "discovery document",
            "document_score": (
                "mean of top-3 token JumpReLU activations"
            ),
            "selection_data": (
                "96 discovery documents only; held-out parquet is not read"
            ),
            "best_label_assignment": (
                "argmax composite; ties use frozen concept order"
            ),
            "composite": (
                "max(AUROC-0.5,0) * max(raw_mean_difference,0) * "
                "(0.5 + 0.5*support_precision)"
            ),
            "strict_gate": (
                "AUROC>=0.85, positive support>=3/4, raw difference>0, "
                "dominance<=0.70, projected ratio>=0.20"
            ),
            "relaxed_gate": (
                "AUROC>=0.75, positive support>=3/4, raw difference>0, "
                "dominance<=0.85, projected ratio>=0.20"
            ),
            "selection_order": (
                "strict tier deterministic round-robin, then relaxed tier; "
                "max four per concept; feature-ID breaks score ties"
            ),
            "direction_dedup": (
                "global absolute centered decoder cosine must be <=0.80"
            ),
            "centering": (
                "unit direction of the equal-document-weight discovery "
                "residual mean"
            ),
            "stop_gate": (
                "proceed only with >=60 selected features and >=18 populated "
                "concept clusters and >=9 reciprocal hard-negative pairs with "
                "at least one selected feature on both sides"
            ),
            "selected_activation_materialization": (
                "independent second chunked pass after feature IDs freeze"
            ),
        },
        "dataset": dataset,
        "summary": {
            "labels": labels,
            "requested_max_features": N_CONCEPTS * MAX_PER_CONCEPT,
            "selected_features": len(selected_ids),
            "strict_features": strict_count,
            "relaxed_features": relaxed_count,
            "populated_concepts": populated_concepts,
            "complete_hard_negative_pairs": complete_hard_negative_pairs,
            "label_counts": label_counts,
            "gate_pass": gate_pass,
            "minimum_selected_features": MIN_SELECTED_FEATURES,
            "minimum_populated_concepts": MIN_POPULATED_CONCEPTS,
            "minimum_complete_hard_negative_pairs": (
                MIN_COMPLETE_HARD_NEGATIVE_PAIRS
            ),
            "target_norm": target_norm,
        },
        "selected_directions": records,
        "top_candidates_by_label": top_candidates,
    }

    for output in (args.out, args.vectors_out, args.stats_out):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            to_builtin(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.vectors_out,
        x=x.astype(np.float32, copy=False),
        row_doc_ids=row_doc_ids,
        m_hat=m_hat.astype(np.float32),
        target_norm=np.asarray(target_norm, dtype=np.float32),
        direction_ids=selected,
        direction_groups=np.asarray(
            ["semantic_new"] * len(selected_ids), dtype="U16"
        ),
        direction_labels=np.asarray(selected_labels, dtype="U64"),
        selection_tiers=np.asarray(selected_tiers, dtype="U16"),
        directions=w_dec[selected].astype(np.float32, copy=False),
        selected_feature_activations=selected_token_activations,
        activations_sha256=np.asarray(activation_sha),
        concept_spec_sha256=np.asarray(spec_sha),
        denylist_sha256=np.asarray(deny_audit["manifest_sha256"]),
        sae_params_sha256=np.asarray(sae_sha),
    )
    np.savez_compressed(
        args.stats_out,
        feature_ids=feature_ids,
        labels=np.asarray(labels, dtype="U64"),
        doc_ids=np.arange(len(dataset), dtype=np.int64),
        doc_topics=doc_topics.astype("U64"),
        doc_prompt_ids=np.asarray(
            [row["prompt_id"] for row in dataset], dtype="U128"
        ),
        doc_scores=doc_scores,
        doc_scores_log1p=doc_scores_log.astype(np.float32),
        doc_fires=doc_fires,
        train_auc=np.stack(
            [train_metrics[label]["auc"] for label in labels]
        ).astype(np.float32),
        train_effect=np.stack(
            [train_metrics[label]["effect"] for label in labels]
        ).astype(np.float32),
        train_raw_difference=np.stack(
            [train_metrics[label]["raw_difference"] for label in labels]
        ).astype(np.float32),
        train_pos_support=np.stack(
            [train_metrics[label]["pos_support"] for label in labels]
        ).astype(np.int16),
        train_neg_support=np.stack(
            [train_metrics[label]["neg_support"] for label in labels]
        ).astype(np.int16),
        train_support_precision=np.stack(
            [train_metrics[label]["support_precision"] for label in labels]
        ).astype(np.float32),
        train_dominance=np.stack(
            [train_metrics[label]["dominance"] for label in labels]
        ).astype(np.float32),
        composite_scores=composite_matrix.astype(np.float32),
        best_label_index=best_label_index.astype(np.int16),
        best_composite=best_composite.astype(np.float32),
        denied_prior=denied_mask,
        decoder_norms=decoder_norms.astype(np.float32),
        mean_alignment=mean_alignment.astype(np.float32),
        projected_ratio=projected_ratio.astype(np.float32),
        selected_feature_ids=selected,
        activations_sha256=np.asarray(activation_sha),
        concept_spec_sha256=np.asarray(spec_sha),
        denylist_sha256=np.asarray(deny_audit["manifest_sha256"]),
        sae_params_sha256=np.asarray(sae_sha),
    )
    print("C1_CONFIRMATORY_SELECTION_COMPLETE")
    print(
        json.dumps(
            {
                "status": status,
                "selected": len(selected_ids),
                "strict": strict_count,
                "relaxed": relaxed_count,
                "populated_concepts": populated_concepts,
                "complete_hard_negative_pairs": complete_hard_negative_pairs,
                "gate_pass": gate_pass,
                "label_counts": label_counts,
            },
            sort_keys=True,
        )
    )
    print(f"wrote -> {args.out} + {args.vectors_out} + {args.stats_out}")


if __name__ == "__main__":
    main()
