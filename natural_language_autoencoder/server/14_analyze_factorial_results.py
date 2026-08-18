#!/usr/bin/env python3
"""Post-hoc reporting for the completed B6+B4 factorial batch.

This script does not alter the frozen cohort or rescore with a model.  It
adds the reporting strata needed for honest interpretation:

* end-to-end held-out selection yield, split by domain vs language;
* intention-to-test versus heldout-label-selective direction readability;
* per-axis results and frozen active-control pairs;
* carrier AR-difference norms, so a high cosine on a near-zero difference is
  not mistaken for a strong intervention effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        value = float(value)
        return round(value, 8) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def heldout_valid(record: dict[str, Any]) -> bool:
    test = record.get("test")
    return bool(
        record["group"] == "semantic_new"
        and test
        and float(test["auc"]) >= 0.75
        and float(test["raw_difference"]) > 0
        and float(test["pos_support"]) >= 2
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_features": 0}
    signed_ranks = [
        rank
        for row in rows
        for rank in (row["plus_signed_rank"], row["minus_signed_rank"])
    ]
    feature_ranks = [
        rank
        for row in rows
        for rank in (row["plus_feature_rank"], row["minus_feature_rank"])
    ]
    output = {"n_features": len(rows)}
    for field in (
        "q_plus",
        "r_minus",
        "polarity",
        "difference_direction_cos_centered",
    ):
        values = np.asarray([float(row[field]) for row in rows])
        output[f"{field}_mean"] = float(values.mean())
        output[f"{field}_median"] = float(np.median(values))
        output[f"{field}_positive_fraction"] = float(np.mean(values > 0))
    output.update(
        {
            "sign_accuracy": float(
                np.mean([bool(row["sign_correct"]) for row in rows])
            ),
            "signed_retrieval_top1": float(
                np.mean(np.asarray(signed_ranks) <= 1)
            ),
            "signed_retrieval_top5": float(
                np.mean(np.asarray(signed_ranks) <= 5)
            ),
            "signed_retrieval_mrr": float(
                np.mean(1.0 / np.asarray(signed_ranks))
            ),
            "feature_retrieval_top1": float(
                np.mean(np.asarray(feature_ranks) <= 1)
            ),
            "feature_retrieval_top5": float(
                np.mean(np.asarray(feature_ranks) <= 5)
            ),
            "feature_retrieval_mrr": float(
                np.mean(1.0 / np.asarray(feature_ranks))
            ),
        }
    )
    return output


def rank_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = rank_average_ties(np.asarray(x, dtype=np.float64))
    ry = rank_average_ties(np.asarray(y, dtype=np.float64))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result["status"] != "complete":
        raise ValueError("result is not complete")
    records = selection["selected_directions"]
    record_by_feature = {
        int(record["feature"]): record for record in records
    }
    greedy = [
        row
        for row in result["polarity_rows"]
        if int(row["sample_index"]) == 0
    ]
    greedy_features = [int(row["feature"]) for row in greedy]
    if len(greedy_features) != len(set(greedy_features)):
        raise ValueError("duplicate greedy polarity row for a feature")
    unknown_greedy = set(greedy_features) - set(record_by_feature)
    if unknown_greedy:
        raise ValueError(
            f"result contains features absent from selection: "
            f"{sorted(unknown_greedy)}"
        )
    greedy_by_feature = {
        int(row["feature"]): row for row in greedy
    }

    semantic_records = [
        record for record in records if record["group"] == "semantic_new"
    ]
    selection_yield = {}
    strata = {
        "all": semantic_records,
        "domain": [
            record
            for record in semantic_records
            if record["label"].startswith("domain:")
        ],
        "language": [
            record
            for record in semantic_records
            if record["label"].startswith("language:")
        ],
    }
    for name, cohort in strata.items():
        valid = [record for record in cohort if heldout_valid(record)]
        selection_yield[name] = {
            "selected": len(cohort),
            "heldout_valid": len(valid),
            "yield": len(valid) / len(cohort) if cohort else float("nan"),
        }
    for label in sorted({record["label"] for record in semantic_records}):
        cohort = [
            record for record in semantic_records if record["label"] == label
        ]
        valid = [record for record in cohort if heldout_valid(record)]
        selection_yield[label] = {
            "selected": len(cohort),
            "heldout_valid": len(valid),
            "yield": len(valid) / len(cohort),
        }

    # A debug-direction-limit result is an intentional subset of the frozen
    # selection.  Selection-yield remains a property of the full manifest,
    # while direction summaries only use features actually present in result.
    observed_semantic_records = [
        record
        for record in semantic_records
        if int(record["feature"]) in greedy_by_feature
    ]
    observed_strata = {
        "domain": [
            record
            for record in observed_semantic_records
            if record["label"].startswith("domain:")
        ],
        "language": [
            record
            for record in observed_semantic_records
            if record["label"].startswith("language:")
        ],
    }
    cohorts = {
        "semantic_new_itt": [
            greedy_by_feature[int(record["feature"])]
            for record in observed_semantic_records
        ],
        "semantic_new_heldout_valid": [
            greedy_by_feature[int(record["feature"])]
            for record in observed_semantic_records
            if heldout_valid(record)
        ],
        "domain_itt": [
            greedy_by_feature[int(record["feature"])]
            for record in observed_strata["domain"]
        ],
        "domain_heldout_valid": [
            greedy_by_feature[int(record["feature"])]
            for record in observed_strata["domain"]
            if heldout_valid(record)
        ],
        "language_itt": [
            greedy_by_feature[int(record["feature"])]
            for record in observed_strata["language"]
        ],
        "language_heldout_valid": [
            greedy_by_feature[int(record["feature"])]
            for record in observed_strata["language"]
            if heldout_valid(record)
        ],
    }
    for group in (
        "semantic_legacy",
        "structural",
        "active_nonselective",
        "gaussian",
    ):
        cohorts[group] = [row for row in greedy if row["group"] == group]
    for label in sorted({record["label"] for record in semantic_records}):
        cohorts[label] = [
            greedy_by_feature[int(record["feature"])]
            for record in observed_semantic_records
            if record["label"] == label
        ]

    summaries = {name: summarize(rows) for name, rows in cohorts.items()}

    stochastic_by_feature = {
        int(row["feature"]): row
        for row in result.get("stochastic_by_feature", [])
    }
    if len(stochastic_by_feature) != len(
        result.get("stochastic_by_feature", [])
    ):
        raise ValueError("duplicate stochastic summary for a feature")

    def summarize_stochastic(
        direction_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pairs = [
            (row, stochastic_by_feature[int(row["feature"])])
            for row in direction_rows
            if int(row["feature"]) in stochastic_by_feature
        ]
        rows = [stochastic for _, stochastic in pairs]
        if not rows:
            return {"n_features": 0}
        output = {"n_features": len(rows)}
        for field in (
            "q_plus_mean",
            "q_plus_std",
            "polarity_mean",
            "polarity_std",
        ):
            values = np.asarray([float(row[field]) for row in rows])
            output[f"{field}_mean"] = float(values.mean())
            output[f"{field}_median"] = float(np.median(values))
        sign_consistency = np.asarray(
            [float(row["sign_consistency"]) for row in rows]
        )
        output["sign_consistency_mean"] = float(sign_consistency.mean())
        output["sign_consistency_median"] = float(
            np.median(sign_consistency)
        )
        output["greedy_q_plus_vs_stochastic_mean_spearman"] = spearman(
            [float(direction["q_plus"]) for direction, _ in pairs],
            [float(stochastic["q_plus_mean"]) for _, stochastic in pairs],
        )
        output[
            "greedy_polarity_vs_stochastic_mean_spearman"
        ] = spearman(
            [float(direction["polarity"]) for direction, _ in pairs],
            [float(stochastic["polarity_mean"]) for _, stochastic in pairs],
        )
        return output

    stochastic_summaries = {
        name: summarize_stochastic(rows)
        for name, rows in cohorts.items()
    }

    # Only eight controls have a frozen approximate match.  Report those
    # pairs explicitly instead of pretending all controls are paired.
    frozen_active_records = [
        record
        for record in records
        if record["group"] == "active_nonselective"
    ]
    active_pairs = []
    for record in frozen_active_records:
        control_feature = int(record["feature"])
        semantic_feature = int(record["match"]["matched_semantic_feature"])
        if (
            semantic_feature not in greedy_by_feature
            or control_feature not in greedy_by_feature
        ):
            continue
        semantic_row = greedy_by_feature[semantic_feature]
        control_row = greedy_by_feature[control_feature]
        active_pairs.append(
            {
                "semantic_feature": semantic_feature,
                "control_feature": control_feature,
                "matching_distance": float(
                    record["match"]["standardized_squared_distance"]
                ),
                "q_plus_difference": float(
                    semantic_row["q_plus"] - control_row["q_plus"]
                ),
                "polarity_difference": float(
                    semantic_row["polarity"] - control_row["polarity"]
                ),
            }
        )
    if active_pairs:
        active_pair_summary = {
            "n_frozen_pairs": len(frozen_active_records),
            "n_pairs": len(active_pairs),
            "median_matching_distance": float(
                np.median([row["matching_distance"] for row in active_pairs])
            ),
            "median_q_plus_difference": float(
                np.median([row["q_plus_difference"] for row in active_pairs])
            ),
            "q_plus_difference_positive_fraction": float(
                np.mean([row["q_plus_difference"] > 0 for row in active_pairs])
            ),
            "median_polarity_difference": float(
                np.median([row["polarity_difference"] for row in active_pairs])
            ),
            "rows": active_pairs,
        }
    else:
        active_pair_summary = {
            "n_frozen_pairs": len(frozen_active_records),
            "n_pairs": 0,
            "median_matching_distance": None,
            "median_q_plus_difference": None,
            "q_plus_difference_positive_fraction": None,
            "median_polarity_difference": None,
            "rows": [],
        }

    heldout_auc = [
        float(record["test"]["auc"]) for record in observed_semantic_records
    ]
    q_plus = [
        float(greedy_by_feature[int(record["feature"])]["q_plus"])
        for record in observed_semantic_records
    ]

    with np.load(args.vectors, allow_pickle=False) as archive:
        keys = np.asarray(archive["keys"]).astype(str)
        reconstructions = np.asarray(
            archive["reconstruction_vectors"], dtype=np.float64
        )
        m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
    m_hat /= np.linalg.norm(m_hat)
    if len(keys) != len(set(keys)):
        raise ValueError("vectors archive contains duplicate job keys")
    key_to_index = {key: index for index, key in enumerate(keys)}

    carrier_rows = []
    for effect in result["carrier_effects"]:
        feature = int(effect["feature"])
        record = record_by_feature[feature]
        high_row = int(record["carrier_high"]["row_index"])
        low_row = int(record["carrier_low"]["row_index"])
        key_map = {
            "high_baseline": f"carrier:f{feature}:high_baseline:row{high_row}",
            "high_amplify": f"carrier:f{feature}:high_amplify:row{high_row}",
            "high_ablate": f"carrier:f{feature}:high_ablate:row{high_row}",
            "low_baseline": f"carrier:f{feature}:low_baseline:row{low_row}",
            "low_insert": f"carrier:f{feature}:low_insert:row{low_row}",
        }
        rec = {
            name: reconstructions[key_to_index[key]]
            for name, key in key_map.items()
        }
        differences = {
            "amplify_minus_high": rec["high_amplify"] - rec["high_baseline"],
            "high_minus_ablate": rec["high_baseline"] - rec["high_ablate"],
            "insert_minus_low": rec["low_insert"] - rec["low_baseline"],
        }
        carrier_row = {
            **effect,
            "heldout_valid": heldout_valid(record),
            "intervention_is_noop": float(effect["carrier_activation"]) == 0.0,
        }
        for name, difference in differences.items():
            projected = difference - (difference @ m_hat) * m_hat
            carrier_row[f"{name}_raw_norm"] = float(
                np.linalg.norm(difference)
            )
            carrier_row[f"{name}_projected_norm"] = float(
                np.linalg.norm(projected)
            )
        carrier_rows.append(carrier_row)

    def summarize_carriers(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n_features": 0}
        output = {
            "n_features": len(rows),
            "noop_interventions": sum(
                bool(row["intervention_is_noop"]) for row in rows
            ),
        }
        for effect_name in (
            "amplify_minus_high",
            "high_minus_ablate",
            "insert_minus_low",
        ):
            for suffix in ("cos_centered", "projected_norm"):
                field = f"{effect_name}_{suffix}"
                values = np.asarray([float(row[field]) for row in rows])
                output[f"{field}_mean"] = float(values.mean())
                output[f"{field}_median"] = float(np.median(values))
                if suffix == "cos_centered":
                    output[f"{field}_positive_fraction"] = float(
                        np.mean(values > 0)
                    )
        return output

    carrier_summaries = {
        "semantic_new_itt": summarize_carriers(
            [row for row in carrier_rows if row["group"] == "semantic_new"]
        ),
        "semantic_new_nonzero": summarize_carriers(
            [
                row
                for row in carrier_rows
                if row["group"] == "semantic_new"
                and not row["intervention_is_noop"]
            ]
        ),
        "semantic_new_heldout_valid": summarize_carriers(
            [
                row
                for row in carrier_rows
                if row["group"] == "semantic_new"
                and row["heldout_valid"]
                and not row["intervention_is_noop"]
            ]
        ),
        "semantic_legacy": summarize_carriers(
            [
                row
                for row in carrier_rows
                if row["group"] == "semantic_legacy"
                and not row["intervention_is_noop"]
            ]
        ),
    }

    output = {
        "schema_version": 1,
        "experiment": "B6+B4 factorial post-hoc stratified report",
        "result_scope": {
            "selection_directions": len(records),
            "observed_greedy_directions": len(greedy_by_feature),
            "observed_semantic_new": len(observed_semantic_records),
            "is_partial_result": len(greedy_by_feature) != len(records),
        },
        "selection_yield": selection_yield,
        "direction_summary": summaries,
        "frozen_primary_summary": {
            name: result["summary_by_cohort_greedy"][name]
            for name in (
                "semantic_new_intention_to_test",
                "semantic_new_heldout_valid",
                "active_nonselective",
                "gaussian",
            )
        },
        "stochastic_summary": stochastic_summaries,
        "generic_control": result["generic_control"],
        "heldout_auc_vs_q_plus_spearman": spearman(heldout_auc, q_plus),
        "active_control_frozen_pairs": active_pair_summary,
        "carrier_summary": carrier_summaries,
        "carrier_rows_with_difference_norms": carrier_rows,
        "interpretation_limits": [
            "domain positives within a split are translations of one topic, not independent topics",
            "language strata include script, tokenization, and length effects",
            "heldout-valid means label-selective, not human-validated monosemantic",
            "structural control has n=1 and active controls cover only eight approximate pairs",
            "isolated -w_dec is an OOD signed-axis test, not a semantic antifeature",
            "carrier tests measure verbalization/reconstruction sensitivity, not downstream model behavior",
            "same-family AV-to-AR round-trip is internal communication, not external label fidelity",
        ],
    }
    args.out.write_text(
        json.dumps(
            to_builtin(output),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# B6+B4 Factorial — Stratified Result",
        "",
        "## Protocol and completion",
        "",
        f"- Directions: {result['inputs']['n_directions']}; jobs: "
        f"{result['inputs']['n_jobs']} "
        f"({result['inputs']['n_direction_jobs']} direction + "
        f"{result['inputs']['n_carrier_jobs']} carrier).",
        f"- AV generation: "
        f"{result['generation']['av_elapsed_seconds_this_invocation']/60:.1f} "
        f"minutes; AR reconstruction: "
        f"{result['generation']['critic_elapsed_seconds']:.1f} seconds.",
        f"- Explanation tag success: "
        f"{result['generation']['explanation_tag_success_fraction']:.1%}.",
        "- Greedy generation is the primary result; four temperature-0.7 "
        "draws estimate within-feature generation stability.",
        "",
        "## Selection transfer",
        "",
        "| Stratum | Selected | Heldout-valid | Yield |",
        "|---|---:|---:|---:|",
    ]
    for name in ("all", "domain", "language"):
        row = selection_yield[name]
        lines.append(
            f"| {name} | {row['selected']} | {row['heldout_valid']} | "
            f"{row['yield']:.1%} |"
        )

    def format_summary_value(
        row: dict[str, Any], key: str, spec: str
    ) -> str:
        value = row.get(key)
        if value is None or not np.isfinite(float(value)):
            return "—"
        return format(float(value), spec)

    primary_itt = result["summary_by_cohort_greedy"][
        "semantic_new_intention_to_test"
    ]
    primary_valid = result["summary_by_cohort_greedy"][
        "semantic_new_heldout_valid"
    ]
    lines.extend(
        [
            "",
            "Primary frozen-cohort intervals:",
            "",
            f"- Semantic ITT q+ median "
            f"{primary_itt['q_plus_median']:.3f} "
            f"(bootstrap 95% "
            f"[{primary_itt['q_plus_median_bootstrap_95'][0]:.3f}, "
            f"{primary_itt['q_plus_median_bootstrap_95'][1]:.3f}]); "
            f"polarity median {primary_itt['polarity_median']:.3f} "
            f"([{primary_itt['polarity_median_bootstrap_95'][0]:.3f}, "
            f"{primary_itt['polarity_median_bootstrap_95'][1]:.3f}]).",
            f"- Heldout-valid q+ median "
            f"{primary_valid['q_plus_median']:.3f} "
            f"([{primary_valid['q_plus_median_bootstrap_95'][0]:.3f}, "
            f"{primary_valid['q_plus_median_bootstrap_95'][1]:.3f}]); "
            f"polarity median {primary_valid['polarity_median']:.3f} "
            f"([{primary_valid['polarity_median_bootstrap_95'][0]:.3f}, "
            f"{primary_valid['polarity_median_bootstrap_95'][1]:.3f}]).",
            f"- Fixed generic texts: mean absolute centered cosine "
            f"{result['generic_control']['mean_abs_centered_cos_all_directions']:.3f} "
            f"across all directions and "
            f"{result['generic_control']['mean_abs_centered_cos_by_group']['semantic_new']:.3f} "
            f"for semantic_new.",
            f"- Heldout AUC vs greedy q+ Spearman: "
            f"{output['heldout_auc_vs_q_plus_spearman']:.3f}.",
            "",
            "## Stochastic stability",
            "",
            "| Cohort | n | median mean q+ | median q+ SD | "
            "median mean polarity | mean sign consistency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "semantic_new_itt",
        "semantic_new_heldout_valid",
        "domain_itt",
        "domain_heldout_valid",
        "language_itt",
        "language_heldout_valid",
        "active_nonselective",
        "gaussian",
    ):
        row = stochastic_summaries[name]
        lines.append(
            f"| {name} | {row['n_features']} | "
            f"{format_summary_value(row, 'q_plus_mean_median', '.3f')} | "
            f"{format_summary_value(row, 'q_plus_std_median', '.3f')} | "
            f"{format_summary_value(row, 'polarity_mean_median', '.3f')} | "
            f"{format_summary_value(row, 'sign_consistency_mean', '.1%')} |"
        )
    lines.extend(
        [
            "",
            "- Greedy-vs-stochastic-mean q+ Spearman: "
            f"ITT {stochastic_summaries['semantic_new_itt']['greedy_q_plus_vs_stochastic_mean_spearman']:.3f}; "
            f"heldout-valid "
            f"{stochastic_summaries['semantic_new_heldout_valid']['greedy_q_plus_vs_stochastic_mean_spearman']:.3f}; "
            f"domain-valid "
            f"{stochastic_summaries['domain_heldout_valid']['greedy_q_plus_vs_stochastic_mean_spearman']:.3f}; "
            f"language-valid "
            f"{stochastic_summaries['language_heldout_valid']['greedy_q_plus_vs_stochastic_mean_spearman']:.3f}.",
        ]
    )
    pair = active_pair_summary
    lines.extend(
        [
            "",
            "## Frozen active-control pairs",
            "",
            f"- Analyzable pairs: {pair['n_pairs']}/"
            f"{pair['n_frozen_pairs']}; median matching distance "
            f"{format_summary_value(pair, 'median_matching_distance', '.3f')}.",
            f"- Median semantic-minus-control q+ difference: "
            f"{format_summary_value(pair, 'median_q_plus_difference', '.3f')} "
            f"({format_summary_value(pair, 'q_plus_difference_positive_fraction', '.1%')} "
            f"positive); polarity difference: "
            f"{format_summary_value(pair, 'median_polarity_difference', '.3f')}.",
            "",
            "## Carrier-conditioned AV/AR readout",
            "",
            "| Cohort | n | no-op | amplify median cos / norm | "
            "ablate median cos / norm | insert median cos / norm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "semantic_new_itt",
        "semantic_new_nonzero",
        "semantic_new_heldout_valid",
        "semantic_legacy",
    ):
        row = carrier_summaries[name]
        lines.append(
            f"| {name} | {row['n_features']} | "
            f"{row.get('noop_interventions', 0)} | "
            f"{format_summary_value(row, 'amplify_minus_high_cos_centered_median', '.3f')} / "
            f"{format_summary_value(row, 'amplify_minus_high_projected_norm_median', '.0f')} | "
            f"{format_summary_value(row, 'high_minus_ablate_cos_centered_median', '.3f')} / "
            f"{format_summary_value(row, 'high_minus_ablate_projected_norm_median', '.0f')} | "
            f"{format_summary_value(row, 'insert_minus_low_cos_centered_median', '.3f')} / "
            f"{format_summary_value(row, 'insert_minus_low_projected_norm_median', '.0f')} |"
        )
    lines.extend(
        [
            "",
            "## Greedy isolated-direction results",
            "",
            "| Cohort | n | q+ median | r− median | polarity median | sign acc. | signed Top-1 | feature Top-1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for name in (
        "semantic_new_itt",
        "semantic_new_heldout_valid",
        "domain_itt",
        "domain_heldout_valid",
        "language_itt",
        "language_heldout_valid",
        "active_nonselective",
        "gaussian",
    ):
        row = summaries[name]
        lines.append(
            f"| {name} | {row['n_features']} | "
            f"{format_summary_value(row, 'q_plus_median', '.3f')} | "
            f"{format_summary_value(row, 'r_minus_median', '.3f')} | "
            f"{format_summary_value(row, 'polarity_median', '.3f')} | "
            f"{format_summary_value(row, 'sign_accuracy', '.1%')} | "
            f"{format_summary_value(row, 'signed_retrieval_top1', '.1%')} | "
            f"{format_summary_value(row, 'feature_retrieval_top1', '.1%')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            *[
                f"- {item}"
                for item in output["interpretation_limits"]
            ],
            "",
        ]
    )
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")
    print("B6B4_STRATIFIED_ANALYSIS_COMPLETE")
    print(f"wrote -> {args.out} + {args.markdown_out}")


if __name__ == "__main__":
    main()
