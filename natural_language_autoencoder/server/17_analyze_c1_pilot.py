#!/usr/bin/env python3
"""Analyze the completed C1 protocol pilot.

All uncertainty resamples the 24 features as clusters.  The only primary
contrast uses the seven historically frozen coarse axes.  Context-derived
labels, base-model judgments, retrieval, private-code tests, and the B6
heldout-valid subgroup are exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Replace non-standard JSON NaN/Infinity values with null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def percentile_ci(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(finite, [2.5, 97.5])]


def bootstrap_vector(
    values: np.ndarray,
    rng: np.random.Generator,
    reps: int,
    statistic: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    point = float(statistic(values))
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    samples = statistic(values[indices])
    return point, percentile_ci(np.asarray(samples))


def mean_stat(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=-1)


def median_stat(x: np.ndarray) -> np.ndarray:
    return np.median(x, axis=-1)


def win_stat(x: np.ndarray) -> np.ndarray:
    return np.mean(x > 0, axis=-1)


def exact_sign_p_one_sided(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    nonzero = values[np.abs(values) > 1e-12]
    wins = int(np.sum(nonzero > 0))
    n = len(nonzero)
    p = (
        sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)
        if n
        else 1.0
    )
    return {
        "wins": wins,
        "losses": int(n - wins),
        "ties": int(len(values) - n),
        "n_nonzero": n,
        "p_one_sided": float(p),
    }


def exact_signflip_mean_p_one_sided(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.0
    observed = float(np.mean(values))
    distribution = np.array([0.0], dtype=np.float64)
    for value in values:
        distribution = np.concatenate((distribution + value, distribution - value))
    distribution /= len(values)
    return float(np.mean(distribution >= observed - 1e-15))


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return float("nan")
    rx = rankdata(x[valid])
    ry = rankdata(y[valid])
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    y_true = y_true[valid]
    scores = scores[valid]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize_delta(
    values: np.ndarray, rng: np.random.Generator, reps: int
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    mean, mean_ci = bootstrap_vector(values, rng, reps, mean_stat)
    median, median_ci = bootstrap_vector(values, rng, reps, median_stat)
    win_rate, win_ci = bootstrap_vector(values, rng, reps, win_stat)
    return {
        "n_features": len(values),
        "mean": mean,
        "mean_bootstrap_95": mean_ci,
        "median": median,
        "median_bootstrap_95": median_ci,
        "positive_fraction": win_rate,
        "positive_fraction_bootstrap_95": win_ci,
        "sign_test": exact_sign_p_one_sided(values),
        "exact_signflip_mean_p_one_sided": exact_signflip_mean_p_one_sided(values),
    }


def summarize_kind(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q = np.array([row["target_cos_centered"] for row in rows], dtype=np.float64)
    axis_rank = np.array([row["axis_retrieval_rank"] for row in rows], dtype=np.float64)
    feature_rank = np.array(
        [row["feature_retrieval_rank"] for row in rows], dtype=np.float64
    )
    judge = np.array(
        [
            float(row["blind_base_judge_score"])
            if row["blind_base_judge_score"] is not None
            else np.nan
            for row in rows
        ]
    )
    unsupported = np.array(
        [
            float(row["blind_base_judge_unsupported"])
            if row["blind_base_judge_unsupported"] is not None
            else np.nan
            for row in rows
        ]
    )
    judge_valid = np.isfinite(judge)
    unsupported_valid = np.isfinite(unsupported)
    return {
        "n_rows": len(rows),
        "n_features": len({row["feature"] for row in rows}),
        "target_cos_mean": float(np.mean(q)),
        "target_cos_median": float(np.median(q)),
        "target_cos_positive_fraction": float(np.mean(q > 0)),
        "axis_top1": float(np.mean(axis_rank == 1)),
        "axis_mrr": float(np.mean(1.0 / axis_rank)),
        "feature_top1": float(np.mean(feature_rank == 1)),
        "feature_top5": float(np.mean(feature_rank <= 5)),
        "feature_mrr": float(np.mean(1.0 / feature_rank)),
        "judge_n_valid": int(np.sum(judge_valid)),
        "judge_score_mean": (
            float(np.mean(judge[judge_valid])) if np.any(judge_valid) else None
        ),
        "judge_score_median": (
            float(np.median(judge[judge_valid])) if np.any(judge_valid) else None
        ),
        "judge_score_ge2_fraction": (
            float(np.mean(judge[judge_valid] >= 2)) if np.any(judge_valid) else None
        ),
        "judge_unsupported_fraction": (
            float(np.mean(unsupported[unsupported_valid]))
            if np.any(unsupported_valid)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result["status"] != "complete":
        raise ValueError("C1 pilot result is incomplete")
    rows = result["scored_candidates"]
    features = [int(row["feature"]) for row in result["feature_metadata"]]
    if len(features) != 24 or len(set(features)) != 24:
        raise ValueError("expected 24 unique feature clusters")
    metadata = {int(row["feature"]): row for row in result["feature_metadata"]}

    by_feature_kind: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_ids = set()
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate {candidate_id}")
        candidate_ids.add(candidate_id)
        feature = int(row["feature"])
        if feature not in metadata:
            raise ValueError(f"unknown feature {feature}")
        if not math.isfinite(float(row["target_cos_centered"])):
            raise ValueError("non-finite target score")
        by_feature_kind[(feature, row["kind"])].append(row)
        by_kind[row["kind"]].append(row)

    singleton_kinds = [
        "axis_reference",
        "axis_paraphrase",
        "axis_hard_negative",
        "train_reference",
        "train_reference_paraphrase",
        "train_hard_negative",
        "sibling_mismatch",
        "nla_original",
        "base_autointerp",
        "nla_paraphrase",
    ]
    for feature in features:
        for kind in singleton_kinds:
            if len(by_feature_kind[(feature, kind)]) != 1:
                raise ValueError(f"expected one {kind} for f{feature}")
        if len(by_feature_kind[(feature, "generic")]) != 8:
            raise ValueError(f"expected eight generic controls for f{feature}")

    def value(feature: int, kind: str, field: str = "target_cos_centered") -> float:
        raw = by_feature_kind[(feature, kind)][0][field]
        return float(raw) if raw is not None else float("nan")

    rng = np.random.default_rng(args.seed)
    axis_delta = np.array(
        [
            value(feature, "axis_reference")
            - value(feature, "axis_hard_negative")
            for feature in features
        ]
    )
    context_delta = np.array(
        [
            value(feature, "train_reference")
            - value(feature, "train_hard_negative")
            for feature in features
        ]
    )
    sibling_delta = np.array(
        [
            value(feature, "train_reference")
            - value(feature, "sibling_mismatch")
            for feature in features
        ]
    )
    axis_paraphrase_change = np.array(
        [
            value(feature, "axis_paraphrase")
            - value(feature, "axis_reference")
            for feature in features
        ]
    )
    context_paraphrase_change = np.array(
        [
            value(feature, "train_reference_paraphrase")
            - value(feature, "train_reference")
            for feature in features
        ]
    )
    nla_paraphrase_change = np.array(
        [
            value(feature, "nla_paraphrase") - value(feature, "nla_original")
            for feature in features
        ]
    )
    private_code_interaction = -nla_paraphrase_change + context_paraphrase_change

    kind_summary = {
        kind: summarize_kind(kind_rows) for kind, kind_rows in sorted(by_kind.items())
    }
    # Generic is first summarized row-wise above and then correctly aggregated
    # to one mean score per feature for comparisons.
    generic_feature_mean = np.array(
        [
            np.mean(
                [
                    float(row["target_cos_centered"])
                    for row in by_feature_kind[(feature, "generic")]
                ]
            )
            for feature in features
        ]
    )
    axis_vs_generic = np.array(
        [
            value(feature, "axis_reference") - generic_feature_mean[index]
            for index, feature in enumerate(features)
        ]
    )
    context_vs_generic = np.array(
        [
            value(feature, "train_reference") - generic_feature_mean[index]
            for index, feature in enumerate(features)
        ]
    )

    primary = summarize_delta(axis_delta, rng, args.bootstrap)
    exploratory_deltas = {
        "train_reference_minus_train_hard_negative": summarize_delta(
            context_delta, rng, args.bootstrap
        ),
        "train_reference_minus_sibling_mismatch": summarize_delta(
            sibling_delta, rng, args.bootstrap
        ),
        "axis_reference_minus_generic_feature_mean": summarize_delta(
            axis_vs_generic, rng, args.bootstrap
        ),
        "train_reference_minus_generic_feature_mean": summarize_delta(
            context_vs_generic, rng, args.bootstrap
        ),
        "axis_paraphrase_minus_axis_reference": summarize_delta(
            axis_paraphrase_change, rng, args.bootstrap
        ),
        "train_paraphrase_minus_train_reference": summarize_delta(
            context_paraphrase_change, rng, args.bootstrap
        ),
        "nla_paraphrase_minus_nla_original": summarize_delta(
            nla_paraphrase_change, rng, args.bootstrap
        ),
        "private_code_interaction": summarize_delta(
            private_code_interaction, rng, args.bootstrap
        ),
    }

    feature_rows = []
    for index, feature in enumerate(features):
        feature_rows.append(
            {
                "feature": feature,
                "label": metadata[feature]["label"],
                "heldout_valid": metadata[feature]["heldout_valid_by_b6_rule"],
                "test_auc": float(metadata[feature]["test_metrics"]["auc"]),
                "reference_evidence_level": metadata[feature][
                    "reference_evidence_level"
                ],
                "axis_delta": float(axis_delta[index]),
                "context_delta": float(context_delta[index]),
                "sibling_delta": float(sibling_delta[index]),
                "axis_paraphrase_change": float(axis_paraphrase_change[index]),
                "context_paraphrase_change": float(
                    context_paraphrase_change[index]
                ),
                "nla_paraphrase_change": float(nla_paraphrase_change[index]),
                "private_code_interaction": float(private_code_interaction[index]),
                "generic_target_cos_mean": float(generic_feature_mean[index]),
                "axis_reference_q": value(feature, "axis_reference"),
                "train_reference_q": value(feature, "train_reference"),
                "nla_original_q": value(feature, "nla_original"),
                "nla_paraphrase_q": value(feature, "nla_paraphrase"),
                "base_autointerp_q": value(feature, "base_autointerp"),
                "axis_reference_judge": value(
                    feature, "axis_reference", "blind_base_judge_score"
                ),
                "train_reference_judge": value(
                    feature, "train_reference", "blind_base_judge_score"
                ),
                "nla_original_judge": value(
                    feature, "nla_original", "blind_base_judge_score"
                ),
                "base_autointerp_judge": value(
                    feature, "base_autointerp", "blind_base_judge_score"
                ),
            }
        )

    label_summary = {}
    labels = list(dict.fromkeys(row["label"] for row in feature_rows))
    for label in labels:
        cohort = [row for row in feature_rows if row["label"] == label]
        label_summary[label] = {
            "n_features": len(cohort),
            "axis_delta_mean": float(np.mean([row["axis_delta"] for row in cohort])),
            "axis_delta_positive_fraction": float(
                np.mean([row["axis_delta"] > 0 for row in cohort])
            ),
            "context_delta_mean": float(
                np.mean([row["context_delta"] for row in cohort])
            ),
        }
    leave_one_label_out = {}
    for label in labels:
        cohort = [row["axis_delta"] for row in feature_rows if row["label"] != label]
        leave_one_label_out[label] = float(np.mean(cohort))

    heldout_valid = np.array([row["heldout_valid"] for row in feature_rows], dtype=bool)
    subgroup = {
        "n_heldout_valid": int(np.sum(heldout_valid)),
        "axis_delta_heldout_valid": summarize_delta(
            axis_delta[heldout_valid], rng, args.bootstrap
        ),
        "axis_delta_heldout_invalid": summarize_delta(
            axis_delta[~heldout_valid], rng, args.bootstrap
        ),
        "context_delta_heldout_valid": summarize_delta(
            context_delta[heldout_valid], rng, args.bootstrap
        ),
        "context_delta_heldout_invalid": summarize_delta(
            context_delta[~heldout_valid], rng, args.bootstrap
        ),
    }

    test_auc = np.array([row["test_auc"] for row in feature_rows])
    external_associations = {}
    for kind in (
        "axis_reference",
        "train_reference",
        "nla_original",
        "nla_paraphrase",
        "base_autointerp",
    ):
        q = np.array([value(feature, kind) for feature in features])
        judge = np.array(
            [
                value(feature, kind, "blind_base_judge_score")
                for feature in features
            ]
        )
        external_associations[kind] = {
            "q_vs_test_auc_spearman": spearman(q, test_auc),
            "q_vs_blind_base_judge_spearman": spearman(q, judge),
            "q_auc_for_heldout_valid": auc_score(heldout_valid.astype(int), q),
            "q_auc_for_judge_score_ge2": auc_score((judge >= 2).astype(int), q),
        }

    pooled_q = np.array([float(row["target_cos_centered"]) for row in rows])
    pooled_judge = np.array(
        [
            float(row["blind_base_judge_score"])
            if row["blind_base_judge_score"] is not None
            else np.nan
            for row in rows
        ]
    )
    pooled_feature = np.array([int(row["feature"]) for row in rows])
    centered_q = pooled_q.copy()
    centered_judge = pooled_judge.copy()
    for feature in features:
        mask = pooled_feature == feature
        centered_q[mask] -= np.nanmean(pooled_q[mask])
        centered_judge[mask] -= np.nanmean(pooled_judge[mask])
    provisional_judge = {
        "feature_centered_q_vs_judge_spearman": spearman(
            centered_q, centered_judge
        ),
        "pooled_q_auc_for_judge_score_ge2": auc_score(
            (pooled_judge >= 2).astype(int), pooled_q
        ),
        "warning": (
            "candidate rows are nested in features and the judge is Gemma-family; "
            "these pooled metrics are descriptive, not independent human validity"
        ),
    }

    analysis = {
        "schema_version": 1,
        "experiment": "C1 external-validity protocol pilot analysis",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.result), "sha256": sha256_file(args.result)},
        "scope": result["scope"],
        "primary_axis_reference_minus_axis_hard_negative": primary,
        "exploratory_deltas": exploratory_deltas,
        "kind_summary": kind_summary,
        "label_summary": label_summary,
        "leave_one_label_out_axis_delta_mean": leave_one_label_out,
        "heldout_valid_descriptive": subgroup,
        "external_associations": external_associations,
        "provisional_blind_base_judge": provisional_judge,
        "undefined_metric_notes": {
            "external_associations.axis_reference.q_auc_for_judge_score_ge2": (
                "undefined because all 24 axis-reference judge scores are >=2; "
                "the binary target has no negative class"
            )
        },
        "feature_rows": feature_rows,
        "interpretation_limits": [
            "The 24 B6 features and AV outputs were inspected before this pilot.",
            "Only seven coarse axis labels were historically frozen; fine-grained references were authored later.",
            "The base judge and base autointerpreter are provisional model signals, not blind human ground truth.",
            "Features are nested in seven labels and come from one model, layer, SAE setting, and synthetic prompt design.",
            "The heldout-valid subgroup is post-selection descriptive and never replaces ITT n=24.",
            "AR discrimination or retrieval does not establish monosemanticity, causal validity, or behavioral steering.",
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    analysis = json_safe(analysis)
    args.out_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    def fmt_ci(summary: dict[str, Any], key: str) -> str:
        ci = summary[f"{key}_bootstrap_95"]
        return f"{summary[key]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"

    lines = [
        "# C1 External-Validity Protocol Pilot",
        "",
        "## Scope",
        "",
        "- ITT unit: 24 previously inspected B6 semantic features.",
        "- Only the seven coarse domain/language axes were historically frozen.",
        "- Fine-grained references and the Gemma context judge are exploratory, not human ground truth.",
        "",
        "## Primary frozen-axis contrast",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Mean delta | {fmt_ci(primary, 'mean')} |",
        f"| Median delta | {fmt_ci(primary, 'median')} |",
        f"| Positive fraction | {fmt_ci(primary, 'positive_fraction')} |",
        (
            f"| Exact sign test | {primary['sign_test']['wins']}/"
            f"{primary['sign_test']['n_nonzero']} wins, "
            f"p={primary['sign_test']['p_one_sided']:.6g} |"
        ),
        (
            f"| Exact sign-flip test on mean | "
            f"p={primary['exact_signflip_mean_p_one_sided']:.6g} |"
        ),
        "",
        "The delta is `q_AR(axis reference) - q_AR(fixed deranged axis)` for each feature.",
        "",
        "## Candidate-source summaries",
        "",
        "| Kind | n | q median | axis Top-1 | feature Top-1 | judge>=2 | unsupported |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    preferred_order = [
        "axis_reference",
        "axis_paraphrase",
        "axis_hard_negative",
        "train_reference",
        "train_reference_paraphrase",
        "train_hard_negative",
        "sibling_mismatch",
        "base_autointerp",
        "nla_original",
        "nla_paraphrase",
        "generic",
    ]
    for kind in preferred_order:
        summary = kind_summary[kind]
        lines.append(
            f"| {kind} | {summary['n_rows']} | {summary['target_cos_median']:.4f} | "
            f"{summary['axis_top1']:.1%} | {summary['feature_top1']:.1%} | "
            f"{summary['judge_score_ge2_fraction']:.1%} | "
            f"{summary['judge_unsupported_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Exploratory contrasts",
            "",
            "| Contrast | Median [95% bootstrap] | Positive fraction | sign p |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, summary in exploratory_deltas.items():
        ci = summary["median_bootstrap_95"]
        lines.append(
            f"| {name} | {summary['median']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{summary['positive_fraction']:.1%} | "
            f"{summary['sign_test']['p_one_sided']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## External-association diagnostics",
            "",
            "| Candidate | q vs test AUC ρ | heldout-valid AUC | q vs blind judge ρ |",
            "|---|---:|---:|---:|",
        ]
    )
    for kind, summary in external_associations.items():
        lines.append(
            f"| {kind} | {summary['q_vs_test_auc_spearman']:.3f} | "
            f"{summary['q_auc_for_heldout_valid']:.3f} | "
            f"{summary['q_vs_blind_base_judge_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            *[f"- {item}" for item in analysis["interpretation_limits"]],
            "",
        ]
    )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print("C1_PILOT_ANALYSIS_COMPLETE")
    print(
        json.dumps(
            {
                "primary_median": primary["median"],
                "primary_wins": primary["sign_test"]["wins"],
                "primary_sign_p": primary["sign_test"]["p_one_sided"],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
