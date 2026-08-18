#!/usr/bin/env python3
"""Analyze C1-confirmatory with reciprocal pairs as independent units.

For each feature, the frozen effect is the mean centered AR score of its three
correct references minus the mean score of the three template-matched
reciprocal hard negatives.  Feature effects are first averaged within concept.
The primary independent unit is then a *complete reciprocal pair*: the mean of
the two concept-cluster effects, requiring selected features on both sides.

The primary p-value enumerates all 2^n joint pair-level sign assignments
(n <= 12 by construction).  Its uncertainty interval is the preregistered
20,000-resample pair bootstrap.  Feature- and concept-level results are
descriptive and cannot rescue a failed pair-level primary endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BENCHMARK_STATUS = "benchmark_frozen_before_C1_AV_AR_and_heldout"
RESULT_STATUS = "complete_ready_for_preregistered_cluster_analysis"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260731
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} is not a lowercase SHA256 digest")
    return result


def generated_text_binding_sha256(row: dict[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "candidate_id": str(row["candidate_id"]),
                "feature": int(row["feature"]),
                "kind": str(row["kind"]),
                "ordinal": int(row["ordinal"]),
                "candidate_concept_id": row.get("candidate_concept_id"),
                "text_sha256": str(row["text_sha256"]),
            }
        )
    )


def expected_candidate_mapping(
    benchmark_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for record in benchmark_records:
        feature = int(record["feature"])
        concept_id = str(record["concept_id"])
        superdomain = str(record["superdomain"])
        for candidate in record.get("static_candidates", []):
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in expected:
                raise ValueError(f"duplicate benchmark candidate ID {candidate_id}")
            expected[candidate_id] = {
                "feature": feature,
                "concept_id": concept_id,
                "superdomain": superdomain,
                "kind": str(candidate["kind"]),
                "ordinal": int(candidate["ordinal"]),
                "assigned_concept_id": str(candidate["assigned_concept_id"]),
                "candidate_concept_id": str(
                    candidate["candidate_concept_id"]
                ),
                "candidate_superdomain": str(
                    candidate["candidate_superdomain"]
                ),
                "text": str(candidate["text"]),
                "text_sha256": require_sha256(
                    candidate.get("text_sha256"),
                    f"benchmark candidate {candidate_id}.text_sha256",
                ),
                "generated": False,
            }
        for request in record.get("generation_requests", []):
            candidate_id = str(request["candidate_id"])
            if candidate_id in expected:
                raise ValueError(f"duplicate benchmark candidate ID {candidate_id}")
            expected[candidate_id] = {
                "feature": feature,
                "concept_id": concept_id,
                "superdomain": superdomain,
                "kind": str(request["kind"]),
                "ordinal": 0,
                "assigned_concept_id": concept_id,
                "candidate_concept_id": None,
                "candidate_superdomain": None,
                "generated": True,
            }
    return expected


def verify_candidate_mapping(
    expected: dict[str, dict[str, Any]],
    scored: list[dict[str, Any]],
) -> None:
    scored_by_id: dict[str, dict[str, Any]] = {}
    for row in scored:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id in scored_by_id:
            raise ValueError(f"result contains duplicate candidate ID {candidate_id}")
        scored_by_id[candidate_id] = row
    if set(scored_by_id) != set(expected):
        raise ValueError(
            "result candidate set differs from frozen benchmark "
            f"(missing={len(set(expected)-set(scored_by_id))}, "
            f"extra={len(set(scored_by_id)-set(expected))})"
        )
    common_fields = (
        "feature",
        "concept_id",
        "superdomain",
        "kind",
        "ordinal",
        "assigned_concept_id",
        "candidate_concept_id",
        "candidate_superdomain",
        "generated",
    )
    for candidate_id, expected_row in expected.items():
        row = scored_by_id[candidate_id]
        for field in common_fields:
            actual = row.get(field)
            frozen = expected_row[field]
            if field in {"feature", "ordinal"}:
                actual = int(actual)
            if actual != frozen:
                raise ValueError(
                    f"candidate {candidate_id} {field} mapping drift: "
                    f"{actual!r} != {frozen!r}"
                )
        text = str(row.get("text", ""))
        if not text or text != normalize_text(text):
            raise ValueError(
                f"candidate {candidate_id} text is empty or not normalized"
            )
        text_sha = require_sha256(
            row.get("text_sha256"), f"candidate {candidate_id}.text_sha256"
        )
        if text_sha != sha256_text(normalize_text(text)):
            raise ValueError(f"candidate {candidate_id} text SHA drift")
        if expected_row["generated"]:
            expected_key = (
                f"av:f{expected_row['feature']}"
                if expected_row["kind"] == "nla_av"
                else f"base:f{expected_row['feature']}"
            )
            if str(row.get("generation_checkpoint_key")) != expected_key:
                raise ValueError(
                    f"candidate {candidate_id} checkpoint-key mapping drift"
                )
            binding = require_sha256(
                row.get("generated_text_binding_sha256"),
                f"candidate {candidate_id}.generated_text_binding_sha256",
            )
            if binding != generated_text_binding_sha256(row):
                raise ValueError(
                    f"candidate {candidate_id} generated-text binding drift"
                )
            require_sha256(
                row.get("generation_input_sha256"),
                f"candidate {candidate_id}.generation_input_sha256",
            )
        else:
            if (
                text != expected_row["text"]
                or text_sha != expected_row["text_sha256"]
            ):
                raise ValueError(
                    f"static candidate {candidate_id} text mapping drift"
                )


def verify_feature_metadata_mapping(
    benchmark_records: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    expected_order = [int(row["feature"]) for row in benchmark_records]
    actual_order = [int(row.get("feature", -1)) for row in result_rows]
    if actual_order != expected_order or len(actual_order) != len(set(actual_order)):
        raise ValueError(
            "feature_metadata feature order/uniqueness differs from benchmark"
        )
    output: dict[int, dict[str, Any]] = {}
    frozen_fields = (
        "feature",
        "direction_index",
        "concept_id",
        "superdomain",
        "hard_negative_id",
        "selection_tier",
        "train_metrics",
        "discovery_contexts",
    )
    for benchmark_row, result_row in zip(benchmark_records, result_rows):
        feature = int(benchmark_row["feature"])
        expected = {
            "feature": feature,
            "direction_index": int(benchmark_row["direction_index"]),
            "concept_id": str(benchmark_row["concept_id"]),
            "superdomain": str(benchmark_row["superdomain"]),
            "hard_negative_id": str(benchmark_row["hard_negative_id"]),
            "selection_tier": str(benchmark_row["selection_tier"]),
            "train_metrics": benchmark_row["train_metrics"],
            "discovery_contexts": benchmark_row["discovery_contexts"],
        }
        actual = {field: result_row.get(field) for field in frozen_fields}
        actual["feature"] = int(actual["feature"])
        actual["direction_index"] = int(actual["direction_index"])
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError(
                f"feature_metadata frozen-field mapping drift for f{feature}"
            )
        metrics = result_row.get("heldout_metrics")
        contexts = result_row.get("heldout_contexts")
        if not isinstance(metrics, dict) or not isinstance(contexts, list):
            raise ValueError(f"feature_metadata held-out fields missing for f{feature}")
        if any(str(row.get("split")) != "test" for row in contexts):
            raise ValueError(f"non-test held-out context in feature_metadata f{feature}")
        output[feature] = result_row
    return output


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def percentile_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def exact_joint_pair_signflip(values: np.ndarray) -> dict[str, Any]:
    """One-sided exact random-sign test of the equal-pair mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= len(values) <= 12:
        raise ValueError("exact pair signflip requires 1..12 pair effects")
    if not np.isfinite(values).all():
        raise ValueError("pair effects must be finite")
    observed = float(np.mean(values))
    magnitudes = np.abs(values)
    assignments = 1 << len(values)
    states = np.arange(assignments, dtype=np.uint16)[:, None]
    bits = (states >> np.arange(len(values), dtype=np.uint16)) & 1
    signs = bits.astype(np.float64) * 2.0 - 1.0
    null_means = (signs @ magnitudes) / len(values)
    extreme = int(np.sum(null_means >= observed - 1e-15))
    return {
        "method": "exact joint reciprocal-pair random-sign test",
        "alternative": "equal-pair mean > 0",
        "n_pairs": int(len(values)),
        "assignments": int(assignments),
        "extreme_assignments": extreme,
        "p_one_sided": float(extreme / assignments),
        "observed_equal_pair_mean": observed,
    }


def pair_bootstrap(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_RESAMPLES, len(values)),
    )
    estimates = np.mean(values[indices], axis=1)
    return {
        "method": "percentile bootstrap over reciprocal pairs",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "ci_95": percentile_interval(estimates),
    }


def exact_binomial_sign_test(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    nonzero = values[values != 0.0]
    positives = int(np.sum(nonzero > 0.0))
    trials = int(len(nonzero))
    if trials == 0:
        return {
            "method": "exact one-sided binomial sign test",
            "positive": 0,
            "nonzero": 0,
            "p_one_sided": None,
        }
    tail = sum(math.comb(trials, k) for k in range(positives, trials + 1))
    return {
        "method": "exact one-sided binomial sign test",
        "positive": positives,
        "nonzero": trials,
        "p_one_sided": float(tail / (2**trials)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    rx, ry = rankdata(x), rankdata(y)
    if float(np.std(rx)) <= 0.0 or float(np.std(ry)) <= 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive": int(np.sum(array > 0.0)),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def summarize_candidate_kind(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    q = [float(row["target_cos_centered"]) for row in rows]
    feature_ranks = [float(row["feature_retrieval_rank"]) for row in rows]
    concept_ranks = [float(row["concept_retrieval_rank"]) for row in rows]
    return {
        "n": len(rows),
        "target_q": distribution(q),
        "feature_retrieval_rank_median": float(np.median(feature_ranks)),
        "feature_retrieval_top5_fraction": float(
            np.mean([rank <= 5.0 for rank in feature_ranks])
        ),
        "concept_retrieval_rank_median": float(np.median(concept_ranks)),
        "concept_retrieval_top1_fraction": float(
            np.mean([rank == 1.0 for rank in concept_ranks])
        ),
    }


def feature_effects(
    benchmark_records: list[dict[str, Any]],
    scored_by_feature: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in benchmark_records:
        feature = int(record["feature"])
        rows = scored_by_feature[feature]
        correct = sorted(
            (row for row in rows if row["kind"] == "correct_reference"),
            key=lambda row: int(row["ordinal"]),
        )
        hard_negative = sorted(
            (row for row in rows if row["kind"] == "hard_negative_reference"),
            key=lambda row: int(row["ordinal"]),
        )
        if (
            len(correct) != 3
            or len(hard_negative) != 3
            or [int(row["ordinal"]) for row in correct] != [0, 1, 2]
            or [int(row["ordinal"]) for row in hard_negative] != [0, 1, 2]
        ):
            raise ValueError(
                f"feature {feature} lacks the frozen 3x3 matched references"
            )
        concept_id = str(record["concept_id"])
        negative_id = str(record["hard_negative_id"])
        if any(str(row["candidate_concept_id"]) != concept_id for row in correct):
            raise ValueError(f"correct reference identity drift for f{feature}")
        if any(
            str(row["candidate_concept_id"]) != negative_id
            for row in hard_negative
        ):
            raise ValueError(f"hard-negative identity drift for f{feature}")
        correct_q = [float(row["target_cos_centered"]) for row in correct]
        negative_q = [
            float(row["target_cos_centered"]) for row in hard_negative
        ]
        template_deltas = [
            correct_q[index] - negative_q[index] for index in range(3)
        ]
        all_wrong = [
            float(row["target_cos_centered"])
            for row in rows
            if row["kind"]
            in {
                "hard_negative_reference",
                "other_within_superdomain_reference",
            }
        ]
        output.append(
            {
                "feature": feature,
                "concept_id": concept_id,
                "superdomain": str(record["superdomain"]),
                "hard_negative_id": negative_id,
                "selection_tier": str(record["selection_tier"]),
                "correct_q": correct_q,
                "hard_negative_q": negative_q,
                "template_deltas": template_deltas,
                "delta": float(np.mean(correct_q) - np.mean(negative_q)),
                "correct_vs_all_within_superdomain_wrong_delta": (
                    float(np.mean(correct_q) - np.mean(all_wrong))
                    if all_wrong
                    else None
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if benchmark.get("status") != BENCHMARK_STATUS:
        raise ValueError("benchmark status is not frozen")
    if result.get("status") != RESULT_STATUS:
        raise ValueError("result is not complete")
    benchmark_sha = sha256_file(args.benchmark)
    recorded_sha = result.get("inputs", {}).get("benchmark", {}).get("sha256")
    if recorded_sha != benchmark_sha:
        raise ValueError("result was not produced from the supplied benchmark")

    benchmark_records = benchmark.get("records", [])
    feature_ids = [int(row["feature"]) for row in benchmark_records]
    if len(feature_ids) < 60 or len(feature_ids) != len(set(feature_ids)):
        raise ValueError("invalid benchmark feature cohort")
    if result.get("scope") != benchmark.get("scope"):
        raise ValueError("result scope differs from frozen benchmark")
    if result.get("protocol") != benchmark.get("protocol"):
        raise ValueError("result protocol differs from frozen benchmark")
    expected_candidates = expected_candidate_mapping(benchmark_records)
    expected_candidate_ids = set(expected_candidates)
    scored = result.get("scored_candidates", [])
    if not isinstance(scored, list):
        raise ValueError("result scored_candidates must be a list")
    scored_ids = [str(row["candidate_id"]) for row in scored]
    verify_candidate_mapping(expected_candidates, scored)
    for row in scored:
        for key in (
            "target_cos_centered",
            "feature_retrieval_rank",
            "concept_retrieval_rank",
        ):
            if not np.isfinite(float(row[key])):
                raise ValueError(f"non-finite {key} for {row['candidate_id']}")
        if any("judge" in key.lower() for key in row):
            raise ValueError("automatic judge field unexpectedly present")
    scored_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        scored_by_feature[int(row["feature"])].append(row)
    if set(scored_by_feature) != set(feature_ids):
        raise ValueError("result feature set differs from benchmark")

    features = feature_effects(benchmark_records, scored_by_feature)
    feature_delta_by_id = {
        int(row["feature"]): float(row["delta"]) for row in features
    }
    concepts: list[dict[str, Any]] = []
    features_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        features_by_concept[str(row["concept_id"])].append(row)
    for concept_id, rows in features_by_concept.items():
        domains = {str(row["superdomain"]) for row in rows}
        negatives = {str(row["hard_negative_id"]) for row in rows}
        if len(domains) != 1 or len(negatives) != 1:
            raise ValueError(f"inconsistent metadata within concept {concept_id}")
        deltas = [float(row["delta"]) for row in rows]
        concepts.append(
            {
                "concept_id": concept_id,
                "superdomain": next(iter(domains)),
                "hard_negative_id": next(iter(negatives)),
                "n_features": len(rows),
                "feature_ids": [int(row["feature"]) for row in rows],
                "effect": float(np.mean(deltas)),
                "feature_delta_median": float(np.median(deltas)),
            }
        )
    concept_by_id = {str(row["concept_id"]): row for row in concepts}
    concept_values = np.asarray(
        [float(row["effect"]) for row in concepts],
        dtype=np.float64,
    )
    concept_cluster_sign_test = exact_binomial_sign_test(concept_values)
    frozen_pairs = benchmark.get("scope", {}).get(
        "complete_reciprocal_pairs", []
    )
    if not isinstance(frozen_pairs, list) or len(frozen_pairs) < 9:
        raise ValueError("benchmark lacks at least nine frozen complete pairs")
    pairs: list[dict[str, Any]] = []
    seen_concepts: set[str] = set()
    for pair_index, pair in enumerate(frozen_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"invalid frozen pair at index {pair_index}")
        left_id, right_id = str(pair[0]), str(pair[1])
        if left_id not in concept_by_id or right_id not in concept_by_id:
            raise ValueError(f"frozen pair is no longer complete: {pair}")
        left, right = concept_by_id[left_id], concept_by_id[right_id]
        if (
            str(left["hard_negative_id"]) != right_id
            or str(right["hard_negative_id"]) != left_id
            or str(left["superdomain"]) != str(right["superdomain"])
        ):
            raise ValueError(f"pair reciprocity drifted: {pair}")
        if left_id in seen_concepts or right_id in seen_concepts:
            raise ValueError(f"concept appears in more than one primary pair: {pair}")
        seen_concepts.update((left_id, right_id))
        pairs.append(
            {
                "pair_id": f"{left_id}__{right_id}",
                "concept_ids": [left_id, right_id],
                "superdomain": str(left["superdomain"]),
                "concept_effects": [
                    float(left["effect"]),
                    float(right["effect"]),
                ],
                "concept_feature_counts": [
                    int(left["n_features"]),
                    int(right["n_features"]),
                ],
                "effect": float(
                    0.5 * (float(left["effect"]) + float(right["effect"]))
                ),
            }
        )
    pair_values = np.asarray([row["effect"] for row in pairs], dtype=np.float64)
    if not 9 <= len(pair_values) <= 12:
        raise ValueError(
            f"primary requires 9..12 complete pairs, got {len(pair_values)}"
        )
    exact_test = exact_joint_pair_signflip(pair_values)
    bootstrap = pair_bootstrap(pair_values)
    primary_positive = bool(
        exact_test["p_one_sided"] < 0.05 and float(np.mean(pair_values)) > 0.0
    )

    metadata_rows = result.get("feature_metadata")
    if not isinstance(metadata_rows, list):
        raise ValueError("result feature_metadata must be a list")
    metadata_by_feature = verify_feature_metadata_mapping(
        benchmark_records, metadata_rows
    )
    heldout_fields = ("auc", "effect", "raw_difference", "pos_support")
    heldout_associations = {}
    feature_delta_ordered = [feature_delta_by_id[feature] for feature in feature_ids]
    for field in heldout_fields:
        values = [
            float(metadata_by_feature[feature]["heldout_metrics"][field])
            for feature in feature_ids
        ]
        heldout_associations[f"spearman_delta_vs_heldout_{field}"] = spearman(
            feature_delta_ordered, values
        )

    tiers: dict[str, list[float]] = defaultdict(list)
    within_domain_wrong: list[float] = []
    for row in features:
        tiers[str(row["selection_tier"])].append(float(row["delta"]))
        if row["correct_vs_all_within_superdomain_wrong_delta"] is not None:
            within_domain_wrong.append(
                float(row["correct_vs_all_within_superdomain_wrong_delta"])
            )
    candidate_kinds = sorted({str(row["kind"]) for row in scored})
    candidate_kind_summaries = {
        kind: summarize_candidate_kind(
            [row for row in scored if str(row["kind"]) == kind]
        )
        for kind in candidate_kinds
    }
    leave_one_superdomain_out = {}
    for superdomain in sorted({str(row["superdomain"]) for row in pairs}):
        retained = [
            float(row["effect"])
            for row in pairs
            if str(row["superdomain"]) != superdomain
        ]
        leave_one_superdomain_out[superdomain] = {
            "n_pairs": len(retained),
            "equal_pair_mean": mean_or_none(retained),
        }
    leave_one_pair_out = {
        str(row["pair_id"]): mean_or_none(
            [
                float(other["effect"])
                for other in pairs
                if other["pair_id"] != row["pair_id"]
            ]
        )
        for row in pairs
    }
    primary_pair_by_concept: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        for concept_id in pair["concept_ids"]:
            if concept_id in primary_pair_by_concept:
                raise ValueError(
                    f"concept {concept_id} appears in multiple primary pairs"
                )
            primary_pair_by_concept[str(concept_id)] = pair
    all_concept_effects = {
        str(row["concept_id"]): float(row["effect"]) for row in concepts
    }
    leave_one_concept_out: dict[str, dict[str, Any]] = {}
    for concept_id in all_concept_effects:
        affected_pair = primary_pair_by_concept.get(concept_id)
        if affected_pair is None:
            retained_pairs = [float(row["effect"]) for row in pairs]
            reciprocal_id = None
            removed_pair_id = None
            removed_concepts: list[str] = []
            consequence = (
                "This selected concept is not in a complete primary pair, so "
                "dropping it leaves the preregistered pair estimand unchanged."
            )
        else:
            retained_pairs = [
                float(row["effect"])
                for row in pairs
                if row["pair_id"] != affected_pair["pair_id"]
            ]
            reciprocal_id = next(
                str(value)
                for value in affected_pair["concept_ids"]
                if str(value) != concept_id
            )
            removed_pair_id = str(affected_pair["pair_id"])
            removed_concepts = [
                str(value) for value in affected_pair["concept_ids"]
            ]
            consequence = (
                "Because reciprocal hard-negative concepts form one primary "
                "independent unit, leaving out this concept invalidates and "
                "removes its entire pair, including the reciprocal concept."
            )
        retained_array = np.asarray(retained_pairs, dtype=np.float64)
        sensitivity_test = exact_joint_pair_signflip(retained_array)
        retained_mean = float(np.mean(retained_array))
        unpaired_concepts = [
            effect
            for other_id, effect in all_concept_effects.items()
            if other_id != concept_id
        ]
        leave_one_concept_out[concept_id] = {
            "omitted_concept_id": concept_id,
            "reciprocal_concept_id": reciprocal_id,
            "removed_primary_pair_id": removed_pair_id,
            "removed_concept_ids_from_primary_pair_estimand": removed_concepts,
            "paired_consequence": consequence,
            "retained_primary_pairs": len(retained_pairs),
            "equal_pair_mean": retained_mean,
            "exact_joint_pair_signflip": sensitivity_test,
            "would_meet_primary_sign_and_p_threshold": bool(
                retained_mean > 0.0
                and float(sensitivity_test["p_one_sided"]) < 0.05
            ),
            "unpaired_equal_concept_mean_after_dropping_only_this_concept": (
                mean_or_none(unpaired_concepts)
            ),
        }

    analysis = {
        "schema_version": 1,
        "experiment": "C1 confirmatory synthetic cohort v1",
        "status": "analysis_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "benchmark": {
                "path": str(args.benchmark),
                "sha256": benchmark_sha,
            },
            "result": {
                "path": str(args.result),
                "sha256": sha256_file(args.result),
            },
        },
        "primary": {
            "independent_unit": "complete reciprocal hard-negative concept pair",
            "construction": (
                "feature 3-template delta -> equal-feature concept mean -> "
                "mean of the two concept effects in each reciprocal pair"
            ),
            "gate": {
                "required_complete_pairs": 9,
                "observed_complete_pairs": len(pairs),
                "passed": len(pairs) >= 9,
            },
            "n_pairs": len(pairs),
            "equal_pair_mean": float(np.mean(pair_values)),
            "pair_median": float(np.median(pair_values)),
            "positive_pairs": int(np.sum(pair_values > 0.0)),
            "positive_pair_fraction": float(np.mean(pair_values > 0.0)),
            "bootstrap": bootstrap,
            "exact_joint_pair_signflip": exact_test,
            "robustness_pair_sign_test": exact_binomial_sign_test(pair_values),
            "alpha": 0.05,
            "primary_positive": primary_positive,
            "pairs": pairs,
        },
        "descriptive": {
            "feature_effects": features,
            "feature_effect_distribution": distribution(
                [float(row["delta"]) for row in features]
            ),
            "feature_effect_by_selection_tier": {
                tier: distribution(values) for tier, values in sorted(tiers.items())
            },
            "concept_effects": concepts,
            "concept_effect_distribution": distribution(
                concept_values
            ),
            "concept_cluster_exact_sign_test": concept_cluster_sign_test,
            "correct_vs_all_within_superdomain_wrong": distribution(
                within_domain_wrong
            ),
            "heldout_associations": heldout_associations,
            "candidate_kind_summaries": candidate_kind_summaries,
            "leave_one_superdomain_out": leave_one_superdomain_out,
            "leave_one_pair_out": leave_one_pair_out,
            "leave_one_concept_out": {
                "status": (
                    "descriptive sensitivity only; cannot change the frozen "
                    "full-cohort primary decision"
                ),
                "paired_design_consequence": (
                    "For any concept in a complete reciprocal pair, a valid "
                    "primary-estimand leave-one-concept-out calculation removes "
                    "the whole pair. Therefore the paired equal-pair result is "
                    "identical for the two concepts in that pair and is the "
                    "corresponding leave-one-pair-out sensitivity."
                ),
                "results": leave_one_concept_out,
            },
        },
        "qa": {
            "n_features": len(features),
            "n_concepts": len(concepts),
            "n_complete_pairs": len(pairs),
            "expected_candidate_ids": len(expected_candidate_ids),
            "scored_candidate_ids": len(scored_ids),
            "all_scores_finite": True,
            "automatic_judge_excluded": True,
            "candidate_mapping_exactly_verified": True,
            "feature_metadata_mapping_exactly_verified": True,
            "primary_does_not_treat_features_or_concepts_as_independent": True,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            to_builtin(analysis),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    ci = bootstrap["ci_95"]
    verdict = (
        "PRIMARY POSITIVE"
        if primary_positive
        else "PRIMARY NOT POSITIVE"
    )
    pair_lines = "\n".join(
        f"| `{row['pair_id']}` | {row['superdomain']} | "
        f"{row['concept_effects'][0]:.5f} | {row['concept_effects'][1]:.5f} | "
        f"{row['effect']:.5f} |"
        for row in pairs
    )
    loco_pair_means = [
        float(row["equal_pair_mean"])
        for row in leave_one_concept_out.values()
    ]
    concept_sign_p = concept_cluster_sign_test["p_one_sided"]
    concept_sign_p_text = (
        "undefined (all concept-cluster effects are zero)"
        if concept_sign_p is None
        else f"{concept_sign_p:.8f}"
    )
    markdown = f"""# C1 confirmatory analysis

## Primary result

**{verdict}.** The independent unit is the complete reciprocal hard-negative
concept pair (`n={len(pairs)}`), not a feature or an individual concept.

- Equal-pair mean effect: `{float(np.mean(pair_values)):.6f}`
- Pair median: `{float(np.median(pair_values)):.6f}`
- Positive pairs: `{int(np.sum(pair_values > 0))}/{len(pairs)}`
- Pair-bootstrap 95% percentile interval: `[{ci[0]:.6f}, {ci[1]:.6f}]`
- Exact joint-pair one-sided signflip: `p={exact_test['p_one_sided']:.8f}`
  (`{exact_test['extreme_assignments']}/{exact_test['assignments']}` assignments)

The effect is computed in three frozen steps: matched three-template
feature-level delta, equal-feature concept mean, then the mean of both concept
effects within each reciprocal pair. The decision threshold is `alpha=0.05`.

## Pair effects

| Pair | Superdomain | Side A | Side B | Pair effect |
|---|---|---:|---:|---:|
{pair_lines}

## Descriptive checks

- Features: `{len(features)}`; concepts with selected features: `{len(concepts)}`.
- Feature mean/median delta:
  `{analysis['descriptive']['feature_effect_distribution']['mean']:.6f}` /
  `{analysis['descriptive']['feature_effect_distribution']['median']:.6f}`.
- Concept mean/median effect:
  `{analysis['descriptive']['concept_effect_distribution']['mean']:.6f}` /
  `{analysis['descriptive']['concept_effect_distribution']['median']:.6f}`.
- Preregistered concept-cluster exact one-sided sign test:
  `{concept_cluster_sign_test['positive']}/{concept_cluster_sign_test['nonzero']}`
  positive nonzero effects, `p={concept_sign_p_text}` (descriptive robustness;
  pair-level inference remains primary).
- NLA AV and base autointerpretation comparisons, held-out associations,
  within-superdomain negatives, and leave-one-out sensitivities are descriptive
  only; they cannot rescue a failed pair-level primary endpoint.
- Preregistered leave-one-concept-out paired sensitivity range:
  `[{min(loco_pair_means):.6f}, {max(loco_pair_means):.6f}]`. For a concept in
  a complete reciprocal pair, omitting that concept necessarily removes the
  entire pair (including its reciprocal concept); the two concepts therefore
  share the corresponding leave-one-pair-out result.
- The Gemma-family automatic judge is absent by design. Human specificity and
  correctness claims remain pending blinded ratings from at least three raters.
"""
    args.out_md.write_text(markdown, encoding="utf-8", newline="\n")
    print("C1_CONFIRMATORY_ANALYSIS_COMPLETE")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "pairs": len(pairs),
                "equal_pair_mean": float(np.mean(pair_values)),
                "ci_95": ci,
                "exact_p": exact_test["p_one_sided"],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
