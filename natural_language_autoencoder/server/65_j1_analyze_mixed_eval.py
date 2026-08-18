#!/usr/bin/env python3
"""Full prespecified statistical analysis of the mixed-labeler J1 Terra run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _load_base() -> Any:
    path = Path(__file__).with_name("59_j1_discovery_evaluate.py")
    spec = importlib.util.spec_from_file_location("j1_eval_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator metric base: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
REPS = 20_000
SEED = 20260806
EXPECTED_JOB_SHA256 = (
    "9fd8628a46155a98e0670a79a947cadf69014dddb4aecce02ad0cf281eb599cb"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "0dd98407e73e0dc286b5345be8498f67ad2c4b1ff74291127472947dd3bee96d"
)
EXPECTED_RESULT_SHA256 = (
    "893e59583d69f25979fc4c2324e47b55ffe829c891a0c9d555cc21f74c51b9b8"
)
EXPECTED_MIXED_LABEL_SHA256 = (
    "2ca779f8ffb89d93531fef31beb12a5d81b0185d18d7d02e6450c296ce562b8b"
)
EXPECTED_ANALYSIS_PLAN_SHA256 = (
    "772042a159188d7777f944b59f152b2484cae719cfbfd72190921d91f1aa7147"
)
CONTRASTS = {
    "ASSISTED_vs_SAE": ("NLA_ASSISTED", "SAE_CONTEXT"),
    "ASSISTED_vs_MISMATCHED": ("NLA_ASSISTED", "NLA_MISMATCHED"),
    "CONTRASTIVE_vs_SAE": ("NLA_CONTRASTIVE", "SAE_CONTEXT"),
    "CONTRASTIVE_vs_MISMATCHED": (
        "NLA_CONTRASTIVE",
        "NLA_MISMATCHED",
    ),
}
EXPECTED_PAIR_COUNTS = {
    "ASSISTED_vs_SAE": {"luna": 31, "fable": 12, "mixed": 2},
    "ASSISTED_vs_MISMATCHED": {"luna": 30, "fable": 11, "mixed": 4},
    "CONTRASTIVE_vs_SAE": {"luna": 30, "fable": 11, "mixed": 4},
    "CONTRASTIVE_vs_MISMATCHED": {"luna": 31, "fable": 12, "mixed": 2},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_exact(path: Path, expected: str) -> str:
    actual = BASE.verify_sidecar(path, required=True)
    if actual != expected:
        raise ValueError(f"unexpected SHA-256 for {path}: {actual} != {expected}")
    return actual


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def tail_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    negatives = [row for row in rows if int(row["truth"]) == 0]
    positives = [row for row in rows if int(row["truth"]) == 1]
    if not negatives or not positives:
        raise ValueError("tail diagnostics require positives and negatives")
    neg_p = [float(row["probability"]) for row in negatives]
    pos_p = [float(row["probability"]) for row in positives]
    return {
        "negative_n": len(negatives),
        "negative_mean_probability": statistics.fmean(neg_p),
        "negative_p95_probability": quantile(neg_p, 0.95),
        "negative_rate_p_ge_0_5": sum(p >= 0.5 for p in neg_p) / len(neg_p),
        "negative_rate_p_ge_0_8": sum(p >= 0.8 for p in neg_p) / len(neg_p),
        "negative_abstain_rate": sum(bool(row["abstain"]) for row in negatives)
        / len(negatives),
        "positive_n": len(positives),
        "positive_mean_probability": statistics.fmean(pos_p),
        "positive_p05_probability": quantile(pos_p, 0.05),
        "positive_rate_p_le_0_5": sum(p <= 0.5 for p in pos_p) / len(pos_p),
        "positive_rate_p_le_0_2": sum(p <= 0.2 for p in pos_p) / len(pos_p),
        "positive_abstain_rate": sum(bool(row["abstain"]) for row in positives)
        / len(positives),
        "quantile_method": "linear_interpolation_(n-1)*p",
    }


def metric_bundle(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    bundle = BASE.metric_bundle(rows)
    bundle["tail"] = tail_diagnostics(rows)
    return bundle


def group_scores(
    scores: Sequence[dict[str, Any]]
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in scores:
        grouped.setdefault((int(row["feature"]), str(row["arm"])), []).append(row)
    return grouped


def rows_for(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    features: Sequence[int],
    arm: str,
) -> list[dict[str, Any]]:
    return [
        row
        for feature in features
        for row in grouped.get((int(feature), arm), [])
    ]


def raw_contrast(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    features: Sequence[int],
    left: str,
    right: str,
) -> dict[str, Any]:
    left_rows = rows_for(grouped, features, left)
    right_rows = rows_for(grouped, features, right)
    left_metrics = metric_bundle(left_rows)
    right_metrics = metric_bundle(right_rows)
    metric_names = {
        "micro_average_precision": "micro_pooled_average_precision",
        "macro_average_precision": "macro_mean_feature_average_precision",
        "mean_pairwise_accuracy": "mean_pairwise_accuracy",
        "brier": "brier",
        "coverage": "non_abstain_coverage",
    }
    deltas = {
        output: float(left_metrics[source]) - float(right_metrics[source])
        for output, source in metric_names.items()
    }
    for name in (
        "negative_mean_probability",
        "negative_p95_probability",
        "negative_rate_p_ge_0_5",
        "negative_rate_p_ge_0_8",
        "positive_mean_probability",
        "positive_p05_probability",
        "positive_rate_p_le_0_5",
        "positive_rate_p_le_0_2",
        "negative_abstain_rate",
        "positive_abstain_rate",
    ):
        deltas[f"tail.{name}"] = (
            float(left_metrics["tail"][name]) - float(right_metrics["tail"][name])
        )
    return {
        "left": left,
        "right": right,
        "n_features": len(features),
        "features": list(features),
        "left_metrics": left_metrics,
        "right_metrics": right_metrics,
        "delta_left_minus_right": deltas,
    }


def _bootstrap_counts(n_features: int, reps: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    counts = np.zeros((reps, n_features), dtype=np.int16)
    for rep in range(reps):
        for _ in range(n_features):
            counts[rep, rng.randrange(n_features)] += 1
    return counts


def _weighted_micro_ap(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    features: Sequence[int],
    arm: str,
    counts: np.ndarray,
) -> np.ndarray:
    feature_index = {feature: index for index, feature in enumerate(features)}
    unique_rows = rows_for(grouped, features, arm)
    ordered = sorted(
        unique_rows,
        key=lambda row: (-float(row["probability"]), str(row["context_id"])),
    )
    rank = np.zeros(counts.shape[0], dtype=np.int32)
    hits = np.zeros(counts.shape[0], dtype=np.int32)
    ap_sum = np.zeros(counts.shape[0], dtype=np.float64)
    for row in ordered:
        weight = counts[:, feature_index[int(row["feature"])]].astype(np.int32)
        if int(row["truth"]) == 1:
            max_weight = int(weight.max(initial=0))
            for duplicate_index in range(1, max_weight + 1):
                mask = weight >= duplicate_index
                ap_sum[mask] += (
                    hits[mask] + duplicate_index
                ) / (rank[mask] + duplicate_index)
            hits += weight
        rank += weight
    positives = 4 * len(features)
    if not np.all(hits == positives):
        raise ValueError(f"bootstrap AP positive count mismatch for {arm}")
    return ap_sum / positives


def _feature_metric_arrays(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    features: Sequence[int],
    arm: str,
) -> dict[str, np.ndarray]:
    arrays = {
        "macro_average_precision": [],
        "mean_pairwise_accuracy": [],
        "brier": [],
        "coverage": [],
    }
    for feature in features:
        rows = list(grouped[(feature, arm)])
        arrays["macro_average_precision"].append(BASE.average_precision(rows))
        arrays["mean_pairwise_accuracy"].append(BASE.pairwise_accuracy(rows))
        arrays["brier"].append(BASE.brier(rows))
        arrays["coverage"].append(BASE.coverage(rows))
    return {name: np.asarray(values, dtype=np.float64) for name, values in arrays.items()}


def summarize_bootstrap(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.5, method="linear")),
        "ci95": {
            "lower": float(np.quantile(values, 0.025, method="linear")),
            "upper": float(np.quantile(values, 0.975, method="linear")),
        },
        "probability_delta_gt_0": float(np.mean(values > 0)),
        "probability_delta_lt_0": float(np.mean(values < 0)),
    }


def bootstrap_contrast(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    features: Sequence[int],
    left: str,
    right: str,
    *,
    reps: int = REPS,
    seed: int = SEED,
) -> dict[str, Any]:
    features = list(features)
    if not features:
        raise ValueError("cannot bootstrap an empty feature subset")
    counts = _bootstrap_counts(len(features), reps, seed)
    left_ap = _weighted_micro_ap(grouped, features, left, counts)
    right_ap = _weighted_micro_ap(grouped, features, right, counts)
    values: dict[str, np.ndarray] = {
        "micro_average_precision": left_ap - right_ap
    }
    left_arrays = _feature_metric_arrays(grouped, features, left)
    right_arrays = _feature_metric_arrays(grouped, features, right)
    for name in left_arrays:
        per_feature_delta = left_arrays[name] - right_arrays[name]
        values[name] = (counts @ per_feature_delta) / len(features)
    return {
        "seed": seed,
        "reps": reps,
        "cluster": "feature",
        "sampler": "python_random_Random_randrange_to_feature_counts",
        "quantile_method": "numpy_linear",
        "metrics": {
            name: summarize_bootstrap(metric_values)
            for name, metric_values in values.items()
        },
    }


def analyze_contrast_subset(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    features: Sequence[int],
    left: str,
    right: str,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    features = sorted(int(feature) for feature in features)
    result = raw_contrast(grouped, features, left, right)
    result["feature_cluster_bootstrap"] = (
        bootstrap_contrast(grouped, features, left, right)
        if bootstrap
        else None
    )
    return result


def collapse_flags(comparison: Mapping[str, Any]) -> dict[str, Any]:
    delta = comparison["delta_left_minus_right"]
    flags = {
        "brier_degradation_gt_0_05": float(delta["brier"]) > 0.05,
        "coverage_loss_gt_0_10": float(delta["coverage"]) < -0.10,
        "negative_mean_probability_increase_gt_0_10": (
            float(delta["tail.negative_mean_probability"]) > 0.10
        ),
        "negative_rate_p_ge_0_8_increase_gt_0_10": (
            float(delta["tail.negative_rate_p_ge_0_8"]) > 0.10
        ),
    }
    return {"flag": any(flags.values()), "components": flags}


def validate_result(
    result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]]]:
    if result.get("status") != "EXPLORATORY_BLINDED_EVAL_MIXED_COMPLETE":
        raise ValueError(f"Terra result is not complete: {result.get('status')}")
    if result.get("failures"):
        raise ValueError("Terra result retains failures")
    scores = result.get("scores")
    if not isinstance(scores, list) or len(scores) != 1_800:
        raise ValueError("Terra result must contain exactly 1,800 scores")
    grouped = group_scores(scores)
    if len(grouped) != 225:
        raise ValueError("Terra result must contain 225 feature-arm groups")
    for key, rows in grouped.items():
        if len(rows) != 8:
            raise ValueError(f"group {key} does not contain eight scores")
        if sum(int(row["truth"]) for row in rows) != 4:
            raise ValueError(f"group {key} is not 4-positive/4-negative")
        label_fields = {
            (
                row.get("labeler"),
                row.get("label_batch_id"),
                row.get("label_agent_task"),
                row.get("label_model"),
                row.get("label_transport"),
            )
            for row in rows
        }
        if len(label_fields) != 1:
            raise ValueError(f"group {key} has inconsistent label provenance")
    features = sorted({int(row["feature"]) for row in scores})
    if len(features) != 45:
        raise ValueError("Terra scores do not cover 45 features")
    for feature in features:
        if {arm for f, arm in grouped if f == feature} != set(BASE.ARMS):
            raise ValueError(f"feature {feature} lacks an arm")
    return [dict(row) for row in scores], grouped


def label_assignment(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]]
) -> dict[tuple[int, str], str]:
    return {
        key: str(rows[0]["labeler"])
        for key, rows in grouped.items()
    }


def complete_labeler_subsets(
    features: Sequence[int],
    assignment: Mapping[tuple[int, str], str],
) -> dict[str, list[int]]:
    result = {"luna_only": [], "fable_only": [], "mixed": []}
    for feature in features:
        values = {assignment[(feature, arm)] for arm in BASE.ARMS}
        if values == {"luna"}:
            result["luna_only"].append(feature)
        elif values == {"fable"}:
            result["fable_only"].append(feature)
        else:
            result["mixed"].append(feature)
    if (
        len(result["luna_only"]) != 28
        or len(result["fable_only"]) != 9
        or len(result["mixed"]) != 8
    ):
        raise ValueError(f"complete-labeler subset counts changed: {result}")
    return result


def pair_subsets(
    features: Sequence[int],
    assignment: Mapping[tuple[int, str], str],
    left: str,
    right: str,
) -> dict[str, list[int]]:
    result = {"luna": [], "fable": [], "mixed": []}
    for feature in features:
        pair = (assignment[(feature, left)], assignment[(feature, right)])
        if pair == ("luna", "luna"):
            result["luna"].append(feature)
        elif pair == ("fable", "fable"):
            result["fable"].append(feature)
        else:
            result["mixed"].append(feature)
    return result


def per_feature_contrast(
    grouped: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    features: Sequence[int],
    left: str,
    right: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        comparison = raw_contrast(grouped, [feature], left, right)
        rows.append(
            {
                "feature": feature,
                "delta_left_minus_right": comparison["delta_left_minus_right"],
            }
        )
    return rows


def hypothesis_generation_diagnostics(
    mixed_labels: Mapping[str, Any]
) -> dict[str, Any]:
    case_to_arm: dict[str, tuple[int, str]] = {}
    for job in mixed_labels.get("jobs", []):
        cmap = job.get("condition_map", {}) if isinstance(job, dict) else {}
        for case_id, mapped in cmap.items() if isinstance(cmap, dict) else []:
            if isinstance(mapped, dict):
                case_to_arm[str(case_id)] = (
                    int(mapped["feature"]),
                    str(mapped["condition"]),
                )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in mixed_labels.get("rows", []):
        for case in row.get("cases", []) if isinstance(row, dict) else []:
            case_id = str(case.get("case_id"))
            if case_id in seen or case_id not in case_to_arm:
                continue
            seen.add(case_id)
            _, arm = case_to_arm[case_id]
            groups.setdefault((arm, str(case.get("labeler"))), []).append(case)
    result: dict[str, Any] = {}
    for (arm, labeler), cases in sorted(groups.items()):
        result.setdefault(arm, {})[labeler] = {
            "n": len(cases),
            "non_abstain_rate": sum(not bool(case["abstain"]) for case in cases)
            / len(cases),
            "mean_confidence": statistics.fmean(
                float(case["confidence"]) for case in cases
            ),
        }
    return result


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.6f}"
    return str(value)


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# J1 mixed-labeler Terra analysis",
        "",
        f"Status: **{analysis['status']}** — exploratory discovery only.",
        "",
        "## ITT metrics",
        "",
        "| Arm | micro AP | macro AP | pairwise | Brier | coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in BASE.ARMS:
        row = analysis["complete_itt"]["by_arm"][arm]
        lines.append(
            f"| {arm} | {_fmt(row['micro_pooled_average_precision'])} | "
            f"{_fmt(row['macro_mean_feature_average_precision'])} | "
            f"{_fmt(row['mean_pairwise_accuracy'])} | {_fmt(row['brier'])} | "
            f"{_fmt(row['non_abstain_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision-relevant contrasts",
            "",
            "| Contrast | ITT Δmicro AP [95% bootstrap CI] | Luna-Luna Δ | Fable-Fable Δ | sign flag |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, row in analysis["contrasts"].items():
        full = row["complete_itt"]
        full_delta = full["delta_left_minus_right"]["micro_average_precision"]
        ci = full["feature_cluster_bootstrap"]["metrics"][
            "micro_average_precision"
        ]["ci95"]
        luna = row["same_labeler"]["luna"]["delta_left_minus_right"][
            "micro_average_precision"
        ]
        fable = row["same_labeler"]["fable"]["delta_left_minus_right"][
            "micro_average_precision"
        ]
        lines.append(
            f"| {name} | {_fmt(full_delta)} [{_fmt(ci['lower'])}, {_fmt(ci['upper'])}] | "
            f"{_fmt(luna)} | {_fmt(fable)} | {row['sign_difference_flag']} |"
        )
    decision = analysis["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['verdict']}**",
            "",
            decision["rationale"],
            "",
            "This is a design decision, not a confirmatory scientific claim.",
            "",
            "## Key limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis["limitations"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    results = Path(__file__).resolve().parents[1] / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job", type=Path, default=results / "j1_blinded_eval_job_mixed_v2.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=results / "j1_blinded_eval_checkpoint_mixed_v2.jsonl",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=results / "j1_blinded_eval_result_mixed_v2.json",
    )
    parser.add_argument(
        "--mixed-labels",
        type=Path,
        default=results / "j1_discovery_labels_mixed_result_v3.json",
    )
    parser.add_argument(
        "--label-jobs",
        type=Path,
        default=results / "j1_discovery_labels_jobs_v1.json",
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=results / "J1_DISCOVERY_MIXED_ANALYSIS_PLAN_2026-08-06.md",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=results / "j1_blinded_eval_analysis_mixed_v2.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=results / "J1_BLINDED_EVAL_ANALYSIS_MIXED_v2.md",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    job_sha = verify_exact(args.job, EXPECTED_JOB_SHA256)
    checkpoint_sha = verify_exact(args.checkpoint, EXPECTED_CHECKPOINT_SHA256)
    result_sha = verify_exact(args.result, EXPECTED_RESULT_SHA256)
    mixed_label_sha = verify_exact(
        args.mixed_labels, EXPECTED_MIXED_LABEL_SHA256
    )
    analysis_plan_sha = verify_exact(
        args.analysis_plan, EXPECTED_ANALYSIS_PLAN_SHA256
    )
    result = BASE.load_json(args.result)
    mixed_labels = BASE.load_json(args.mixed_labels)
    label_jobs = BASE.load_json(args.label_jobs)
    if not all(
        isinstance(item, dict) for item in (result, mixed_labels, label_jobs)
    ):
        raise ValueError("analysis JSON inputs must be objects")
    scores, grouped = validate_result(result)
    features = sorted({int(row["feature"]) for row in scores})
    assignment = label_assignment(grouped)
    complete_subsets = complete_labeler_subsets(features, assignment)
    if complete_subsets["mixed"] != [
        2096,
        2700,
        3176,
        3441,
        15742,
        15793,
        16016,
        16059,
    ]:
        raise ValueError("mixed-feature IDs differ from the frozen design audit")
    by_arm = {
        arm: metric_bundle(rows_for(grouped, features, arm)) for arm in BASE.ARMS
    }
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in BASE.STRATA:
        stratum_features = sorted(
            {
                int(row["feature"])
                for row in scores
                if row["stratum"] == stratum
            }
        )
        if len(stratum_features) != 15:
            raise ValueError(f"stratum {stratum} does not contain 15 features")
        by_stratum[stratum] = {
            "features": stratum_features,
            "by_arm": {
                arm: metric_bundle(rows_for(grouped, stratum_features, arm))
                for arm in BASE.ARMS
            },
        }
    by_labeler_and_arm: dict[str, dict[str, Any]] = {}
    for labeler in ("fable", "luna"):
        by_labeler_and_arm[labeler] = {}
        for arm in BASE.ARMS:
            rows = [
                row
                for row in scores
                if row["arm"] == arm and row["labeler"] == labeler
            ]
            by_labeler_and_arm[labeler][arm] = {
                "feature_count": len({int(row["feature"]) for row in rows}),
                "metrics": metric_bundle(rows),
            }
    contrasts: dict[str, Any] = {}
    for name, (left, right) in CONTRASTS.items():
        subsets = pair_subsets(features, assignment, left, right)
        observed_counts = {key: len(value) for key, value in subsets.items()}
        if observed_counts != EXPECTED_PAIR_COUNTS[name]:
            raise ValueError(
                f"pair counts for {name} changed: {observed_counts}"
            )
        full = analyze_contrast_subset(grouped, features, left, right)
        luna = analyze_contrast_subset(
            grouped, subsets["luna"], left, right
        )
        fable = analyze_contrast_subset(
            grouped, subsets["fable"], left, right
        )
        mixed_point = per_feature_contrast(
            grouped, subsets["mixed"], left, right
        )
        full_delta = full["delta_left_minus_right"][
            "micro_average_precision"
        ]
        luna_delta = luna["delta_left_minus_right"][
            "micro_average_precision"
        ]
        fable_delta = fable["delta_left_minus_right"][
            "micro_average_precision"
        ]
        contrasts[name] = {
            "left": left,
            "right": right,
            "pair_category_counts": observed_counts,
            "pair_category_features": subsets,
            "complete_itt": full,
            "same_labeler": {"luna": luna, "fable": fable},
            "mixed_pairs_per_feature": mixed_point,
            "sign_difference_flag": (
                (full_delta > 0) != (luna_delta > 0)
                or (full_delta > 0) != (fable_delta > 0)
            ),
            "signs": {
                "complete_itt": 1 if full_delta > 0 else -1 if full_delta < 0 else 0,
                "luna_luna": 1 if luna_delta > 0 else -1 if luna_delta < 0 else 0,
                "fable_fable": 1 if fable_delta > 0 else -1 if fable_delta < 0 else 0,
            },
            "collapse_complete_itt": collapse_flags(full),
            "collapse_luna_luna": collapse_flags(luna),
        }
    complete_subset_analyses: dict[str, Any] = {}
    for subset_name, subset_features in (
        (
            "homogeneous_all_arms",
            sorted(
                complete_subsets["luna_only"] + complete_subsets["fable_only"]
            ),
        ),
        ("luna_only_all_arms", complete_subsets["luna_only"]),
        ("fable_only_all_arms", complete_subsets["fable_only"]),
    ):
        complete_subset_analyses[subset_name] = {
            "n_features": len(subset_features),
            "features": subset_features,
            "by_arm": {
                arm: metric_bundle(rows_for(grouped, subset_features, arm))
                for arm in BASE.ARMS
            },
            "contrasts": {
                name: analyze_contrast_subset(
                    grouped, subset_features, left, right
                )
                for name, (left, right) in CONTRASTS.items()
            },
        }
    stratum_deltas: dict[str, dict[str, float]] = {}
    for candidate in ("NLA_ASSISTED", "NLA_CONTRASTIVE"):
        stratum_deltas[candidate] = {}
        for stratum in BASE.STRATA:
            metrics = by_stratum[stratum]["by_arm"]
            for comparator in ("SAE_CONTEXT", "NLA_MISMATCHED"):
                stratum_deltas[candidate][f"{stratum}__vs__{comparator}"] = (
                    metrics[candidate]["micro_pooled_average_precision"]
                    - metrics[comparator]["micro_pooled_average_precision"]
                )
    candidate_decisions: dict[str, Any] = {}
    for candidate, prefix in (
        ("NLA_ASSISTED", "ASSISTED"),
        ("NLA_CONTRASTIVE", "CONTRASTIVE"),
    ):
        vs_sae = contrasts[f"{prefix}_vs_SAE"]
        vs_mismatch = contrasts[f"{prefix}_vs_MISMATCHED"]
        full_positive = (
            vs_sae["complete_itt"]["delta_left_minus_right"][
                "micro_average_precision"
            ]
            > 0
            and vs_mismatch["complete_itt"]["delta_left_minus_right"][
                "micro_average_precision"
            ]
            > 0
        )
        luna_positive = (
            vs_sae["same_labeler"]["luna"]["delta_left_minus_right"][
                "micro_average_precision"
            ]
            > 0
            and vs_mismatch["same_labeler"]["luna"]["delta_left_minus_right"][
                "micro_average_precision"
            ]
            > 0
        )
        fable_vs_sae = vs_sae["same_labeler"]["fable"][
            "delta_left_minus_right"
        ]["micro_average_precision"]
        fable_vs_mismatch = vs_mismatch["same_labeler"]["fable"][
            "delta_left_minus_right"
        ]["micro_average_precision"]
        fable_not_reverse_both = not (
            fable_vs_sae <= 0 and fable_vs_mismatch <= 0
        )
        favorable_strata = []
        for stratum in BASE.STRATA:
            if (
                stratum_deltas[candidate][f"{stratum}__vs__SAE_CONTEXT"] > 0
                and stratum_deltas[candidate][
                    f"{stratum}__vs__NLA_MISMATCHED"
                ]
                > 0
            ):
                favorable_strata.append(stratum)
        collapse = any(
            item["flag"]
            for item in (
                vs_sae["collapse_complete_itt"],
                vs_mismatch["collapse_complete_itt"],
                vs_sae["collapse_luna_luna"],
                vs_mismatch["collapse_luna_luna"],
            )
        )
        labeler_only_vs_sae = (
            vs_sae["same_labeler"]["luna"]["delta_left_minus_right"][
                "micro_average_precision"
            ]
            > 0
            and fable_vs_sae <= 0
        )
        directional_gate = (
            full_positive
            and luna_positive
            and fable_not_reverse_both
            and len(favorable_strata) >= 2
            and not collapse
        )
        immediate_launch = directional_gate and not labeler_only_vs_sae
        candidate_decisions[candidate] = {
            "complete_itt_beats_both": full_positive,
            "luna_luna_beats_both": luna_positive,
            "fable_fable_does_not_reverse_both": fable_not_reverse_both,
            "fable_fable_delta_vs_sae": fable_vs_sae,
            "fable_fable_delta_vs_mismatched": fable_vs_mismatch,
            "favorable_strata": favorable_strata,
            "favorable_in_at_least_two_strata": len(favorable_strata) >= 2,
            "obvious_collapse": collapse,
            "benefit_vs_sae_appears_only_in_luna_labels": labeler_only_vs_sae,
            "frozen_directional_gate": directional_gate,
            "immediate_fresh_confirmatory_launch": immediate_launch,
        }
    if any(
        row["immediate_fresh_confirmatory_launch"]
        for row in candidate_decisions.values()
    ):
        verdict = "PROCEED_TO_FRESH_CONFIRMATORY_J1"
        rationale = (
            "At least one assisted arm passes the frozen directional and "
            "mixed-labeler robustness requirements without a labeler-only gain."
        )
    elif any(
        row["frozen_directional_gate"] for row in candidate_decisions.values()
    ):
        verdict = "REDESIGN_REPLICATE_BEFORE_CONFIRMATORY"
        rationale = (
            "The complete ITT and Luna-Luna directions are favorable, but the "
            "gain over SAE_CONTEXT reverses under Fable-Fable labels. Because "
            "Luna labels are evaluated by the same OpenAI/Codex model family, "
            "the present signal is labeler-dependent and does not justify an "
            "immediate expensive fresh confirmatory run."
        )
    else:
        any_full = any(
            row["complete_itt_beats_both"] for row in candidate_decisions.values()
        )
        verdict = (
            "REDESIGN_REPLICATE_BEFORE_CONFIRMATORY"
            if any_full
            else "DEPRIORITIZE_J1"
        )
        rationale = (
            "The frozen robustness requirements are not jointly satisfied."
        )
    byte_budget = BASE._byte_budget_flags(label_jobs, mixed_labels)
    analysis = {
        "schema_version": 2,
        "experiment": "J1 mixed-labeler Terra full statistical analysis",
        "status": "EXPLORATORY_ANALYSIS_MIXED_COMPLETE",
        "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "inputs": {
            "eval_job_sha256": job_sha,
            "eval_checkpoint_sha256": checkpoint_sha,
            "eval_result_sha256": result_sha,
            "mixed_label_result_sha256": mixed_label_sha,
            "mixed_analysis_plan_sha256": analysis_plan_sha,
            "analysis_script_sha256": sha256_file(Path(__file__)),
        },
        "integrity": {
            "features": len(features),
            "arms": len(BASE.ARMS),
            "scores": len(scores),
            "feature_arm_groups": len(grouped),
            "failures": 0,
            "truth_per_group": "4 positive / 4 exact-zero hard negative",
            "labeler_scores_by_arm": {
                arm: {
                    labeler: sum(
                        1
                        for row in scores
                        if row["arm"] == arm and row["labeler"] == labeler
                    )
                    for labeler in ("fable", "luna")
                }
                for arm in BASE.ARMS
            },
        },
        "complete_itt": {
            "features": features,
            "by_arm": by_arm,
            "by_stratum": by_stratum,
            "stratum_micro_ap_deltas": stratum_deltas,
        },
        "by_labeler_and_arm_unpaired_descriptive": by_labeler_and_arm,
        "complete_labeler_subsets": complete_subsets,
        "contrasts": contrasts,
        "complete_subset_robustness": complete_subset_analyses,
        "hypothesis_generation_diagnostics": hypothesis_generation_diagnostics(
            mixed_labels
        ),
        "input_byte_budget_audit": byte_budget,
        "decision": {
            "verdict": verdict,
            "candidate_arms": candidate_decisions,
            "rationale": rationale,
            "next_required_action": (
                "Before a fresh confirmatory J1, replicate all five arms with "
                "one fixed heterogeneous non-OpenAI interpreter (or human "
                "labels), retain Terra blinded evaluation, and add the planned "
                "capacity-matched strong baseline."
                if verdict == "REDESIGN_REPLICATE_BEFORE_CONFIRMATORY"
                else "Follow the frozen fresh-corpus confirmatory design."
            ),
        },
        "limitations": [
            "Discovery-only reused N3 cohort; no confirmatory NLA-assisted-SAE claim is permitted.",
            "Thirteen batches use Fable and 32 use Luna; labeler is collinear with batch order.",
            "Terra and Luna are both OpenAI/Codex-family models, so Luna-label performance may include family-specific communication.",
            "The SAE_CONTEXT baseline is not token/capacity matched; byte-budget equality is not established.",
            "Only one model, layer, SAE family, and AV-format-eligible feature cohort are evaluated.",
            "Mismatched NLA is a harmful-content control, not necessarily a strong neutral autointerp baseline.",
            "Bootstrap intervals are exploratory percentile intervals, not preregistered significance gates.",
        ],
        "bootstrap_contract": {
            "reps": REPS,
            "seed": SEED,
            "cluster": "feature",
            "shared_resamples_within_each_contrast": True,
        },
    }
    output_sha = BASE.write_immutable(args.out, analysis)
    BASE.write_sidecar(Path(str(args.out) + ".sha256"), output_sha)
    markdown = render_markdown(analysis).encode("utf-8")
    if args.out_md.exists():
        if args.out_md.read_bytes() != markdown:
            raise RuntimeError(f"refusing to overwrite analysis markdown: {args.out_md}")
    else:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        with args.out_md.open("xb") as handle:
            handle.write(markdown)
            handle.flush()
            os.fsync(handle.fileno())
    BASE.write_sidecar(
        Path(str(args.out_md) + ".sha256"), hashlib.sha256(markdown).hexdigest()
    )
    print(
        f"analysis={args.out} sha256={output_sha} verdict={verdict}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"65_j1_analyze_mixed_eval: FAIL CLOSED: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
