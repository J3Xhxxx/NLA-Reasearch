#!/usr/bin/env python3
"""Freeze the C1 external-validity protocol-pilot benchmark.

This script is CPU-only and must run before any C1 AR scoring.  It deliberately
separates:

* historically frozen coarse axes (four domains and three languages);
* researcher-authored, train-context-only feature references;
* controlled negatives and exact generic controls;
* existing AV greedy explanations, which are candidates rather than truth.

The 24 semantic features were already inspected during B6+B4.  Therefore this
asset is a protocol pilot, never a confirmatory human-label benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


AXIS_REFERENCE = {
    "domain:biology": "This feature is associated with biology-related explanatory text.",
    "domain:geography": "This feature is associated with geography-related explanatory text.",
    "domain:history": "This feature is associated with history-related explanatory text.",
    "domain:software": "This feature is associated with software-related explanatory text.",
    "language:en": "This feature is associated with text written in English.",
    "language:es": "This feature is associated with text written in Spanish.",
    "language:zh": "This feature is associated with text written in Chinese.",
}

AXIS_PARAPHRASE = {
    "domain:biology": "This direction represents explanatory prose about biology.",
    "domain:geography": "This direction represents explanatory prose about geography.",
    "domain:history": "This direction represents explanatory prose about history.",
    "domain:software": "This direction represents explanatory prose about software.",
    "language:en": "This direction represents English-language text.",
    "language:es": "This direction represents Spanish-language text.",
    "language:zh": "This direction represents Chinese-language text.",
}

AXIS_DERANGEMENT = {
    "domain:biology": "domain:geography",
    "domain:geography": "domain:history",
    "domain:history": "domain:software",
    "domain:software": "domain:biology",
    "language:en": "language:es",
    "language:es": "language:zh",
    "language:zh": "language:en",
}

GENERIC_FALLBACK = [
    "The passage uses a structured informational style and continues the current topic.",
    "This is a coherent piece of explanatory prose with ordinary grammatical structure.",
    "The context establishes a subject and prepares a likely continuation of the discussion.",
    "The text contains semantic and syntactic information typical of a written document.",
    "A descriptive answer is being developed in a clear and organized format.",
    "The final token fits a locally predictable continuation in the surrounding sentence.",
    "The activation reflects general language structure, topical context, and discourse form.",
    "This appears to be an informative response that elaborates on previously introduced material.",
]

AXIS_ONLY_EVIDENCE = {8347, 969, 7176}
REPEATED_LEXEME_EVIDENCE = {
    6809,
    14470,
    10000,
    293,
    115,
    992,
    3800,
    507,
    485,
    918,
    15207,
    15066,
    13887,
    15234,
    1394,
    4581,
    2725,
    935,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_id(feature: int, kind: str, ordinal: int = 0) -> str:
    payload = f"C1pilot|f{feature}|{kind}|{ordinal}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def compact_context(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": int(row["row_index"]),
        "doc_id": int(row["doc_id"]),
        "prompt_id": str(row["prompt_id"]),
        "axis_domain": str(row["axis_domain"]),
        "axis_language": str(row["axis_language"]),
        "split": str(row["split"]),
        "topic": str(row["topic"]),
        "position": int(row["position"]),
        "token": str(row["token"]),
        "activation": float(row["activation"]),
        "prompt": str(row["prompt"]),
    }


def unique_doc_top_contexts(
    metadata: list[dict[str, Any]],
    activations: np.ndarray,
    split: str,
    limit: int,
) -> list[dict[str, Any]]:
    indices = [
        index
        for index, row in enumerate(metadata)
        if row["split"] == split and float(activations[index]) > 0.0
    ]
    indices.sort(key=lambda index: (-float(activations[index]), index))
    seen_docs: set[int] = set()
    output: list[dict[str, Any]] = []
    for index in indices:
        doc_id = int(metadata[index]["doc_id"])
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        row = dict(metadata[index])
        row["activation"] = float(activations[index])
        output.append(compact_context(row))
        if len(output) >= limit:
            break
    return output


def axis_contexts(
    dataset: list[dict[str, Any]],
    label: str,
    positive: bool,
    limit: int,
) -> list[dict[str, Any]]:
    axis, value = label.split(":", 1)
    field = "axis_domain" if axis == "domain" else "axis_language"
    rows = [
        row
        for row in dataset
        if row["split"] == "test" and ((row[field] == value) == positive)
    ]
    rows.sort(key=lambda row: (row["axis_domain"], row["axis_language"], row["doc_id"]))
    if not positive:
        # Deterministic matched negatives: retain a balanced spread rather than
        # letting the first domain/language dominate.
        stride = max(1, len(rows) // max(1, limit))
        rows = rows[::stride]
    return [
        {
            "doc_id": int(row["doc_id"]),
            "prompt_id": str(row["prompt_id"]),
            "axis_domain": str(row["axis_domain"]),
            "axis_language": str(row["axis_language"]),
            "topic": str(row["topic"]),
            "prompt": str(row["prompt"]),
        }
        for row in rows[:limit]
    ]


def static_candidate(
    feature: int,
    kind: str,
    text: str,
    expected_validity: str,
    ordinal: int = 0,
) -> dict[str, Any]:
    # The old B6 score was computed from the exact AV explanation string.
    # Preserve its internal whitespace so C1 can hard-check bit-for-bit protocol
    # equivalence at the text boundary; normalize researcher-authored controls.
    stripped = str(text).strip()
    normalized = stripped if kind == "nla_original" else " ".join(stripped.split())
    if not normalized:
        raise ValueError(f"empty candidate f{feature} {kind}")
    return {
        "candidate_id": candidate_id(feature, kind, ordinal),
        "kind": kind,
        "ordinal": ordinal,
        "text": normalized,
        "expected_validity": expected_validity,
        "generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-train-contexts", type=int, default=4)
    parser.add_argument("--max-axis-contexts", type=int, default=4)
    args = parser.parse_args()

    paths = [
        args.selection,
        args.result,
        args.vectors,
        args.activations,
        args.labels,
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    label_asset = json.loads(args.labels.read_text(encoding="utf-8"))
    if selection["status"] != "selection_frozen_before_AV_AR":
        raise ValueError("selection is not the frozen B6+B4 asset")
    if result["status"] != "complete":
        raise ValueError("B6+B4 result is not complete")

    semantic_records = [
        row for row in selection["selected_directions"] if row["group"] == "semantic_new"
    ]
    if len(semantic_records) != 24:
        raise ValueError(f"expected 24 semantic_new directions, got {len(semantic_records)}")
    features = [int(row["feature"]) for row in semantic_records]
    if len(set(features)) != len(features):
        raise ValueError("duplicate semantic feature")

    authored = {int(row["feature"]): row for row in label_asset["records"]}
    if set(authored) != set(features):
        raise ValueError(
            f"label feature mismatch missing={sorted(set(features)-set(authored))} "
            f"extra={sorted(set(authored)-set(features))}"
        )

    greedy_rows = {
        int(row["feature"]): row
        for row in result["scored_generation_rows"]
        if row.get("probe_type") == "direction"
        and row.get("group") == "semantic_new"
        and int(row.get("sign", 0)) == 1
        and int(row.get("sample_index", -1)) == 0
    }
    if set(greedy_rows) != set(features):
        raise ValueError("missing or duplicate greedy +w_dec rows")

    generic_texts = list(result["generic_control"].get("texts", GENERIC_FALLBACK))
    if generic_texts != GENERIC_FALLBACK:
        raise ValueError("generic controls drifted from the frozen B6+B4 protocol")

    table = pq.read_table(
        args.activations,
        columns=[
            "doc_id",
            "prompt_id",
            "axis_domain",
            "axis_language",
            "split",
            "topic",
            "prompt",
            "position",
            "token",
        ],
    )
    metadata = table.to_pylist()
    for index, row in enumerate(metadata):
        row["row_index"] = index

    with np.load(args.vectors, allow_pickle=False) as archive:
        row_doc_ids = np.asarray(archive["row_doc_ids"], dtype=np.int64)
        direction_ids = np.asarray(archive["direction_ids"], dtype=np.int64)
        direction_groups = np.asarray(archive["direction_groups"])
        activations = np.asarray(archive["selected_feature_activations"], dtype=np.float32)
        directions = np.asarray(archive["directions"], dtype=np.float32)
        m_hat = np.asarray(archive["m_hat"], dtype=np.float32)
    if len(metadata) != len(row_doc_ids) or activations.shape[0] != len(metadata):
        raise ValueError("activation/metadata row mismatch")
    if not np.array_equal(row_doc_ids, np.array([row["doc_id"] for row in metadata])):
        raise ValueError("row_doc_ids disagree with parquet metadata")
    semantic_indices = [
        index for index, group in enumerate(direction_groups.tolist()) if group == "semantic_new"
    ]
    if len(semantic_indices) != 24:
        raise ValueError("direction group mismatch")
    if activations.shape[1] < 24:
        raise ValueError("selected_feature_activations lacks semantic columns")
    if directions.shape != (45, 3840) or m_hat.shape != (3840,):
        raise ValueError("unexpected vector shapes")
    if not np.isfinite(directions).all() or not np.isfinite(m_hat).all():
        raise ValueError("non-finite vector asset")

    groups: dict[str, list[int]] = {}
    for row in semantic_records:
        groups.setdefault(str(row["label"]), []).append(int(row["feature"]))
    sibling = {}
    for label, group_features in groups.items():
        for index, feature in enumerate(group_features):
            sibling[feature] = group_features[(index + 1) % len(group_features)]

    output_records = []
    for semantic_position, record in enumerate(semantic_records):
        feature = int(record["feature"])
        label = str(record["label"])
        if label not in AXIS_REFERENCE:
            raise ValueError(f"unknown label {label}")
        if int(direction_ids[semantic_indices[semantic_position]]) != feature:
            raise ValueError(f"direction order mismatch at f{feature}")
        feature_acts = activations[:, semantic_position]
        train_contexts = unique_doc_top_contexts(
            metadata, feature_acts, "train", args.max_train_contexts
        )
        test_activation_contexts = unique_doc_top_contexts(
            metadata, feature_acts, "test", args.max_train_contexts
        )
        if not train_contexts:
            raise ValueError(f"f{feature} has no positive train contexts")

        label_row = authored[feature]
        sibling_feature = sibling[feature]
        candidates = [
            static_candidate(
                feature, "axis_reference", AXIS_REFERENCE[label], "coarse_positive"
            ),
            static_candidate(
                feature, "axis_paraphrase", AXIS_PARAPHRASE[label], "coarse_positive"
            ),
            static_candidate(
                feature,
                "axis_hard_negative",
                AXIS_REFERENCE[AXIS_DERANGEMENT[label]],
                "controlled_negative",
            ),
            static_candidate(
                feature, "train_reference", label_row["reference"], "train_supported"
            ),
            static_candidate(
                feature,
                "train_reference_paraphrase",
                label_row["reference_paraphrase"],
                "train_supported",
            ),
            static_candidate(
                feature,
                "train_hard_negative",
                label_row["hard_negative"],
                "controlled_negative",
            ),
            static_candidate(
                feature,
                "sibling_mismatch",
                authored[sibling_feature]["reference"],
                "same_axis_negative",
            ),
            static_candidate(
                feature,
                "nla_original",
                greedy_rows[feature]["explanation"],
                "unknown",
            ),
        ]
        candidates.extend(
            static_candidate(feature, "generic", text, "generic_negative", ordinal)
            for ordinal, text in enumerate(generic_texts)
        )
        candidate_ids = [row["candidate_id"] for row in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate id f{feature}")

        output_records.append(
            {
                "feature": feature,
                "direction_index": int(semantic_indices[semantic_position]),
                "label": label,
                "selection_tier": record["selection_tier"],
                "sibling_feature": sibling_feature,
                "train_metrics": record["train"],
                "test_metrics": record["test"],
                "b6_nla_original_target_cos_centered": float(
                    greedy_rows[feature]["target_cos_centered"]
                ),
                "heldout_valid_by_b6_rule": bool(
                    float(record["test"]["auc"]) >= 0.75
                    and float(record["test"]["effect"]) > 0
                    and float(record["test"]["pos_support"]) >= 2
                ),
                "reference_evidence_level": (
                    "axis_only"
                    if feature in AXIS_ONLY_EVIDENCE
                    else (
                        "repeated_train_lexeme"
                        if feature in REPEATED_LEXEME_EVIDENCE
                        else "train_document_topic"
                    )
                ),
                "train_contexts": train_contexts,
                "test_activation_contexts": test_activation_contexts,
                "judge_positive_contexts": axis_contexts(
                    selection["dataset"], label, True, args.max_axis_contexts
                ),
                "judge_negative_contexts": axis_contexts(
                    selection["dataset"], label, False, args.max_axis_contexts
                ),
                "static_candidates": candidates,
                "generation_requests": [
                    {
                        "kind": "base_autointerp",
                        "candidate_id": candidate_id(feature, "base_autointerp"),
                    },
                    {
                        "kind": "nla_paraphrase",
                        "candidate_id": candidate_id(feature, "nla_paraphrase"),
                    },
                ],
            }
        )

    all_candidate_ids = [
        candidate["candidate_id"]
        for record in output_records
        for candidate in record["static_candidates"]
    ] + [
        request["candidate_id"]
        for record in output_records
        for request in record["generation_requests"]
    ]
    if len(all_candidate_ids) != len(set(all_candidate_ids)):
        raise ValueError("candidate ids are not globally unique")

    payload = {
        "schema_version": 1,
        "experiment": "C1 external-validity protocol pilot",
        "status": "benchmark_frozen_before_C1_AR",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "n_features": 24,
            "statistical_unit": "feature",
            "confirmatory": False,
            "reason": (
                "B6 features and AV outputs were previously inspected; references are "
                "researcher-authored and not independent human ground truth."
            ),
        },
        "inputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in paths
        },
        "protocol": {
            "primary": (
                "feature-level delta q_AR(axis_reference) - "
                "q_AR(axis_hard_negative); this is the only historically frozen "
                "label contrast"
            ),
            "exploratory_context_contrast": (
                "feature-level delta q_AR(train_reference) - "
                "q_AR(train_hard_negative); references were authored after B6 "
                "inspection and are not independent ground truth"
            ),
            "projection": (
                "project frozen B6 m_hat out of AR reconstruction and decoder "
                "direction, then L2 normalize"
            ),
            "retrieval_warning": (
                "seven coarse axis labels repeat across features; axis retrieval and "
                "feature retrieval are reported separately"
            ),
            "heldout_valid_rule": "test AUC>=0.75, effect>0, positive support>=2",
            "generic_controls": "exact eight frozen B6+B4 texts",
            "judge": (
                "base Gemma blind text-context judge; provisional external model "
                "signal, not human ground truth"
            ),
        },
        "records": output_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("C1_PILOT_BENCHMARK_FROZEN")
    print(
        json.dumps(
            {
                "features": len(output_records),
                "static_candidates": sum(
                    len(row["static_candidates"]) for row in output_records
                ),
                "generated_candidates": sum(
                    len(row["generation_requests"]) for row in output_records
                ),
                "out": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
