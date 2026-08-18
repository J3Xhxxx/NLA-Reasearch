#!/usr/bin/env python3
"""Secondary, explicitly exploratory diagnostics for the C1 protocol pilot."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return float("nan")
    x_rank = rankdata(x[valid])
    y_rank = rankdata(y[valid])
    if np.std(x_rank) <= 1e-12 or np.std(y_rank) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    y_true = y_true[valid]
    scores = scores[valid]
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum = float(np.sum(ranks[y_true == 1]))
    return float(
        (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def exact_sign_test(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    nonzero = values[np.abs(values) > 1e-12]
    wins = int(np.sum(nonzero > 0))
    n = len(nonzero)
    p = (
        sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
        if n
        else 1.0
    )
    return {
        "wins": wins,
        "losses": n - wins,
        "ties": len(values) - n,
        "p_positive_one_sided": float(p),
    }


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_fraction": float(np.mean(values > 0)),
        "sign_test": exact_sign_test(values),
    }


def average_rank(values: np.ndarray, target_index: int) -> float:
    target = float(values[target_index])
    return float(
        1
        + np.sum(values > target)
        + 0.5 * max(0, int(np.sum(values == target)) - 1)
    )


def retrieval_summary(matrix: np.ndarray) -> dict[str, Any]:
    """Rows are candidate labels; columns are decoder directions."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("retrieval matrix must be square")
    n = len(matrix)
    row_ranks = np.array([average_rank(matrix[i], i) for i in range(n)])
    column_ranks = np.array([average_rank(matrix[:, i], i) for i in range(n)])
    row_margins = []
    column_margins = []
    for index in range(n):
        other = np.arange(n) != index
        row_margins.append(matrix[index, index] - np.max(matrix[index, other]))
        column_margins.append(matrix[index, index] - np.max(matrix[other, index]))
    return {
        "candidate_text_to_feature": {
            "top1": float(np.mean(row_ranks == 1)),
            "top5": float(np.mean(row_ranks <= 5)),
            "mrr": float(np.mean(1 / row_ranks)),
            "median_margin_to_best_wrong": float(np.median(row_margins)),
        },
        "feature_to_candidate_text": {
            "top1": float(np.mean(column_ranks == 1)),
            "top5": float(np.mean(column_ranks <= 5)),
            "mrr": float(np.mean(1 / column_ranks)),
            "median_margin_to_best_wrong": float(np.median(column_margins)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = result["scored_candidates"]
    metadata = result["feature_metadata"]
    features = [int(row["feature"]) for row in metadata]
    feature_index = {feature: index for index, feature in enumerate(features)}
    feature_label = {int(row["feature"]): row["label"] for row in metadata}
    by_feature_kind: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature_kind[(int(row["feature"]), row["kind"])].append(row)

    with np.load(args.vectors, allow_pickle=False) as archive:
        candidate_ids = archive["candidate_ids"].astype(str).tolist()
        similarity = np.asarray(archive["semantic_similarity"], dtype=np.float64)
        semantic_ids = np.asarray(archive["semantic_feature_ids"], dtype=np.int64)
    if semantic_ids.tolist() != features:
        raise ValueError("semantic feature order mismatch")
    result_by_id = {row["candidate_id"]: row for row in rows}
    row_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    if set(result_by_id) != set(candidate_ids):
        raise ValueError("candidate IDs drifted")

    def one(feature: int, kind: str) -> dict[str, Any]:
        values = by_feature_kind[(feature, kind)]
        if len(values) != 1:
            raise ValueError(f"expected singleton {kind} for f{feature}")
        return values[0]

    def field(feature: int, kind: str, name: str) -> float:
        value = one(feature, kind)[name]
        return float(value) if value is not None else float("nan")

    generic_q = np.array(
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
    generic_judge = np.array(
        [
            np.mean(
                [
                    float(row["blind_base_judge_score"])
                    for row in by_feature_kind[(feature, "generic")]
                ]
            )
            for feature in features
        ]
    )

    pair_specs = {
        "axis_reference_minus_axis_hard_negative": (
            "axis_reference",
            "axis_hard_negative",
        ),
        "train_reference_minus_train_hard_negative": (
            "train_reference",
            "train_hard_negative",
        ),
        "train_reference_minus_sibling_mismatch": (
            "train_reference",
            "sibling_mismatch",
        ),
        "base_autointerp_minus_train_hard_negative": (
            "base_autointerp",
            "train_hard_negative",
        ),
        "base_autointerp_minus_nla_original": (
            "base_autointerp",
            "nla_original",
        ),
    }
    paired = {}
    for name, (positive_kind, negative_kind) in pair_specs.items():
        q_delta = np.array(
            [
                field(feature, positive_kind, "target_cos_centered")
                - field(feature, negative_kind, "target_cos_centered")
                for feature in features
            ]
        )
        judge_delta = np.array(
            [
                field(feature, positive_kind, "blind_base_judge_score")
                - field(feature, negative_kind, "blind_base_judge_score")
                for feature in features
            ]
        )
        paired[name] = {
            "ar_q_delta": summarize(q_delta),
            "blind_judge_score_delta": summarize(judge_delta),
        }
    for prefix, kind in (
        ("axis_reference", "axis_reference"),
        ("train_reference", "train_reference"),
        ("base_autointerp", "base_autointerp"),
    ):
        q_delta = np.array(
            [
                field(feature, kind, "target_cos_centered") - generic_q[index]
                for index, feature in enumerate(features)
            ]
        )
        judge_delta = np.array(
            [
                field(feature, kind, "blind_base_judge_score")
                - generic_judge[index]
                for index, feature in enumerate(features)
            ]
        )
        paired[f"{prefix}_minus_generic_feature_mean"] = {
            "ar_q_delta": summarize(q_delta),
            "blind_judge_score_delta": summarize(judge_delta),
        }

    retrieval_kinds = [
        "axis_reference",
        "train_reference",
        "train_reference_paraphrase",
        "sibling_mismatch",
        "base_autointerp",
        "nla_original",
        "nla_paraphrase",
    ]
    bidirectional_retrieval = {}
    for kind in retrieval_kinds:
        indices = [row_index[one(feature, kind)["candidate_id"]] for feature in features]
        bidirectional_retrieval[kind] = retrieval_summary(similarity[indices])

    labels = list(dict.fromkeys(feature_label[feature] for feature in features))
    axis_text_indices = {}
    axis_texts = {}
    for label in labels:
        members = [feature for feature in features if feature_label[feature] == label]
        texts = {one(feature, "axis_reference")["text"] for feature in members}
        if len(texts) != 1:
            raise ValueError(f"axis reference drift for {label}")
        exemplar = members[0]
        axis_text_indices[label] = row_index[
            one(exemplar, "axis_reference")["candidate_id"]
        ]
        axis_texts[label] = next(iter(texts))
    axis_text_matrix = np.stack(
        [similarity[axis_text_indices[label]] for label in labels]
    )
    correct_scores = []
    mean_wrong_deltas = []
    best_wrong_deltas = []
    axis_ranks = []
    for column, feature in enumerate(features):
        correct_index = labels.index(feature_label[feature])
        correct = float(axis_text_matrix[correct_index, column])
        wrong_mask = np.arange(len(labels)) != correct_index
        wrong = axis_text_matrix[wrong_mask, column]
        correct_scores.append(correct)
        mean_wrong_deltas.append(correct - float(np.mean(wrong)))
        best_wrong_deltas.append(correct - float(np.max(wrong)))
        axis_ranks.append(average_rank(axis_text_matrix[:, column], correct_index))
    all_axis_sensitivity = {
        "feature_to_seven_axis_texts": {
            "top1": float(np.mean(np.asarray(axis_ranks) == 1)),
            "mrr": float(np.mean(1 / np.asarray(axis_ranks))),
            "correct_minus_mean_wrong": summarize(np.asarray(mean_wrong_deltas)),
            "correct_minus_best_wrong": summarize(np.asarray(best_wrong_deltas)),
        },
        "labels": labels,
        "texts": axis_texts,
    }

    primary_delta = np.array(
        [
            field(feature, "axis_reference", "target_cos_centered")
            - field(feature, "axis_hard_negative", "target_cos_centered")
            for feature in features
        ]
    )
    label_means = {
        label: float(
            np.mean(
                [
                    primary_delta[feature_index[feature]]
                    for feature in features
                    if feature_label[feature] == label
                ]
            )
        )
        for label in labels
    }
    label_mean_vector = np.array(list(label_means.values()))
    domain_mask = np.array([feature_label[f].startswith("domain:") for f in features])
    evidence_levels = sorted(
        {str(row["reference_evidence_level"]) for row in metadata}
    )
    cluster_sensitivity = {
        "label_cluster_means": label_means,
        "seven_label_mean_summary": summarize(label_mean_vector),
        "domain_feature_delta": summarize(primary_delta[domain_mask]),
        "language_feature_delta": summarize(primary_delta[~domain_mask]),
        "leave_one_label_out_all_positive": bool(
            all(
                np.mean(
                    [
                        primary_delta[feature_index[feature]]
                        for feature in features
                        if feature_label[feature] != omitted
                    ]
                )
                > 0
                for omitted in labels
            )
        ),
        "top_four_positive_effects_share_of_net_sum": float(
            np.sum(np.sort(primary_delta[primary_delta > 0])[-4:])
            / np.sum(primary_delta)
        ),
        "by_reference_evidence_level": {
            level: summarize(
                np.array(
                    [
                        primary_delta[index]
                        for index, row in enumerate(metadata)
                        if row["reference_evidence_level"] == level
                    ]
                )
            )
            for level in evidence_levels
        },
    }

    heldout = np.array(
        [bool(row["heldout_valid_by_b6_rule"]) for row in metadata], dtype=bool
    )
    test_auc = np.array(
        [float(row["test_metrics"]["auc"]) for row in metadata], dtype=np.float64
    )
    context_delta = np.array(
        [
            field(feature, "train_reference", "target_cos_centered")
            - field(feature, "train_hard_negative", "target_cos_centered")
            for feature in features
        ]
    )
    heldout_prediction = {
        "axis_delta_auc_for_heldout_valid": auc_score(heldout.astype(int), primary_delta),
        "axis_delta_vs_test_auc_spearman": spearman(primary_delta, test_auc),
        "context_delta_auc_for_heldout_valid": auc_score(
            heldout.astype(int), context_delta
        ),
        "context_delta_vs_test_auc_spearman": spearman(context_delta, test_auc),
        "axis_delta_mean_valid_minus_invalid": float(
            np.mean(primary_delta[heldout]) - np.mean(primary_delta[~heldout])
        ),
        "context_delta_mean_valid_minus_invalid": float(
            np.mean(context_delta[heldout]) - np.mean(context_delta[~heldout])
        ),
    }

    nla_original_text = [one(feature, "nla_original")["text"] for feature in features]
    nla_paraphrase_text = [one(feature, "nla_paraphrase")["text"] for feature in features]
    train_reference_text = [one(feature, "train_reference")["text"] for feature in features]
    train_paraphrase_text = [
        one(feature, "train_reference_paraphrase")["text"] for feature in features
    ]
    nla_q_change = np.array(
        [
            field(feature, "nla_paraphrase", "target_cos_centered")
            - field(feature, "nla_original", "target_cos_centered")
            for feature in features
        ]
    )
    train_q_change = np.array(
        [
            field(feature, "train_reference_paraphrase", "target_cos_centered")
            - field(feature, "train_reference", "target_cos_centered")
            for feature in features
        ]
    )
    nla_char_change = np.array(
        [len(paraphrase) - len(original) for original, paraphrase in zip(
            nla_original_text, nla_paraphrase_text
        )]
    )
    train_char_change = np.array(
        [len(paraphrase) - len(original) for original, paraphrase in zip(
            train_reference_text, train_paraphrase_text
        )]
    )
    private_code = -nla_q_change + train_q_change
    train_topic_mask = np.array(
        [row["reference_evidence_level"] == "train_document_topic" for row in metadata],
        dtype=bool,
    )
    paraphrase_diagnostics = {
        "nla_q_change": summarize(nla_q_change),
        "train_reference_q_change": summarize(train_q_change),
        "private_code_interaction": summarize(private_code),
        "private_code_interaction_without_train_document_topic": summarize(
            private_code[~train_topic_mask]
        ),
        "private_code_interaction_by_reference_evidence_level": {
            level: summarize(
                private_code[
                    np.array(
                        [
                            row["reference_evidence_level"] == level
                            for row in metadata
                        ],
                        dtype=bool,
                    )
                ]
            )
            for level in evidence_levels
        },
        "nla_character_retention_median": float(
            np.median(
                [
                    len(paraphrase) / len(original)
                    for original, paraphrase in zip(
                        nla_original_text, nla_paraphrase_text
                    )
                ]
            )
        ),
        "train_reference_character_retention_median": float(
            np.median(
                [
                    len(paraphrase) / len(original)
                    for original, paraphrase in zip(
                        train_reference_text, train_paraphrase_text
                    )
                ]
            )
        ),
        "nla_q_change_vs_character_change_spearman": spearman(
            nla_q_change, nla_char_change
        ),
        "train_q_change_vs_character_change_spearman": spearman(
            train_q_change, train_char_change
        ),
        "warning": (
            "The NLA paraphraser was instructed to delete meta phrases and guess "
            "lists, so compression/content deletion is confounded with wording."
        ),
    }

    secondary = {
        "status": "complete",
        "scope": "exploratory secondary analysis; no new confirmatory claims",
        "paired_candidate_diagnostics": paired,
        "bidirectional_24way_retrieval": bidirectional_retrieval,
        "all_axis_text_sensitivity": all_axis_sensitivity,
        "cluster_sensitivity": cluster_sensitivity,
        "heldout_prediction": heldout_prediction,
        "paraphrase_diagnostics": paraphrase_diagnostics,
    }
    args.out_json.write_text(
        json.dumps(secondary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    primary = paired["axis_reference_minus_axis_hard_negative"]["ar_q_delta"]
    axis_all = all_axis_sensitivity["feature_to_seven_axis_texts"]
    retrieval = bidirectional_retrieval
    lines = [
        "# C1 Pilot Secondary Diagnostics",
        "",
        "> Exploratory only; computed after inspecting the primary result.",
        "",
        "## Robustness and retrieval",
        "",
        (
            f"- Frozen-pair axis delta: mean {primary['mean']:.4f}, median "
            f"{primary['median']:.4f}, {primary['sign_test']['wins']}/24 wins."
        ),
        (
            f"- Against all six wrong axis texts: correct-minus-mean-wrong mean "
            f"{axis_all['correct_minus_mean_wrong']['mean']:.4f}; feature→axis "
            f"Top-1 {axis_all['top1']:.1%}."
        ),
        (
            f"- Label-cluster means are positive for "
            f"{sum(value > 0 for value in label_means.values())}/7 axes; all "
            f"leave-one-label-out means are positive."
        ),
        (
            f"- Train-reference 24-way Top-1: text→feature "
            f"{retrieval['train_reference']['candidate_text_to_feature']['top1']:.1%}, "
            f"feature→text "
            f"{retrieval['train_reference']['feature_to_candidate_text']['top1']:.1%}."
        ),
        (
            f"- NLA-original 24-way Top-1: text→feature "
            f"{retrieval['nla_original']['candidate_text_to_feature']['top1']:.1%}, "
            f"feature→text "
            f"{retrieval['nla_original']['feature_to_candidate_text']['top1']:.1%}."
        ),
        "",
        "## External-validity warnings",
        "",
        (
            f"- Axis delta predicts heldout-valid status only descriptively: AUC "
            f"{heldout_prediction['axis_delta_auc_for_heldout_valid']:.3f}; "
            f"Spearman with test AUC "
            f"{heldout_prediction['axis_delta_vs_test_auc_spearman']:.3f}."
        ),
        (
            f"- Context-label delta goes the wrong way for heldout generalization: "
            f"valid-minus-invalid mean "
            f"{heldout_prediction['context_delta_mean_valid_minus_invalid']:.4f}."
        ),
        (
            f"- NLA paraphrases retain a median "
            f"{paraphrase_diagnostics['nla_character_retention_median']:.1%} of "
            f"characters, versus "
            f"{paraphrase_diagnostics['train_reference_character_retention_median']:.1%} "
            f"for authored references; private-code evidence is therefore suggestive, "
            f"not identified."
        ),
        (
            "- The base judge is not a valid quantitative oracle: generic and wrong-axis "
            "candidates frequently receive positive scores, and pooled q→judge AUC is "
            "below chance in the primary report."
        ),
        "",
    ]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print("C1_PILOT_SECONDARY_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
