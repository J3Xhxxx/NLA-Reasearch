#!/usr/bin/env python3
"""Compute the frozen N6 H6-A/H6-B endpoints and decision labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from n6_common import (
    IDENTITY_KL_TOL,
    canonical_sha256,
    require_binding_preregistration,
    sanitize_kl,
    sha256_file,
    verify_code_manifest,
    verify_sha256_sidecar,
    write_new_json,
    write_new_text,
)


N_BOOT = 50_000
N_ANALYSIS = 400
BOOT_SEED = 20260803
QUANTILE_METHOD = "linear"
TOP_K = (1, 5, 10, 50)
N5_GATE_V2_SHA256 = (
    "036477f21fb550b317978a880df0a708dcf42a5201d301ca0757fade3baea059"
)
CONDITIONS = (
    "identity",
    "orig",
    "p3_true",
    "p3_cross_matched",
    "p3_candidate_strip",
    "p3_anchor_strip",
    "p3_all_quote_strip",
    "p12",
    "sae_big",
    "zero",
)


def validate_variant_contract(variants: dict[str, Any]) -> dict[str, Any]:
    if (
        variants.get("status")
        != "COMPLETE_FROZEN_BEFORE_AR_CANDIDATE_MASS_OR_CAUSAL_OUTCOME"
    ):
        raise ValueError("variants are not frozen before any outcome stage")
    rows = variants.get("rows")
    if not isinstance(rows, list) or len(rows) != N_ANALYSIS:
        raise ValueError(f"N6 analysis requires exactly {N_ANALYSIS} variants")
    uids = [str(row.get("row_uid")) for row in rows]
    groups = [str(row.get("content_group_id")) for row in rows]
    docs = [str(row.get("doc_id")) for row in rows]
    if not (
        len(set(uids))
        == len(set(groups))
        == len(set(docs))
        == N_ANALYSIS
    ):
        raise ValueError(
            "N6 requires 400 unique row_uid/content_group_id/doc_id values"
        )
    donor_audit = variants.get("donor_assignment")
    if (
        not isinstance(donor_audit, dict)
        or donor_audit.get("hard_block") != ["source", "candidate_count"]
    ):
        raise ValueError("variants do not declare the frozen hard donor block")
    by_uid = {str(row["row_uid"]): row for row in rows}
    recipient_by_cell: dict[tuple[str, int], set[str]] = defaultdict(set)
    donor_by_cell: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        uid = str(row["row_uid"])
        donor_uid = str(row.get("donor_row_uid"))
        donor = by_uid.get(donor_uid)
        if donor is None or donor_uid == uid:
            raise ValueError(f"{uid} has an invalid donor")
        if (
            str(row["content_group_id"])
            == str(donor["content_group_id"])
            or str(row["doc_id"]) == str(donor["doc_id"])
        ):
            raise ValueError(f"{uid} donor shares a forbidden identity")
        cell = (str(row["source"]), int(row["candidate_count"]))
        donor_cell = (
            str(donor["source"]),
            int(donor["candidate_count"]),
        )
        if cell != donor_cell:
            raise ValueError(f"{uid} donor violates source/count hard match")
        if row.get("cross_candidates") != donor.get("true_candidates"):
            raise ValueError(f"{uid} cross candidates differ from donor")
        true_normalized = set(row.get("true_candidates_normalized", []))
        cross_normalized = set(row.get("cross_candidates_normalized", []))
        if (
            not true_normalized
            or not cross_normalized
            or true_normalized & cross_normalized
        ):
            raise ValueError(f"{uid} donor shares normalized candidates")
        recipient_by_cell[cell].add(uid)
        donor_by_cell[cell].add(donor_uid)
    if recipient_by_cell != donor_by_cell:
        raise ValueError("donor mapping is not one-to-one within every hard cell")
    return {
        "hard_source_and_candidate_count_match": True,
        "one_to_one_derangement_within_cells": True,
        "recipient_content_document_distinct": True,
        "no_shared_normalized_candidate": True,
    }


def interval(values: np.ndarray, confidence: float = 0.95) -> list[float]:
    alpha = 1.0 - confidence
    return [
        float(
            np.quantile(
                values, alpha / 2.0, method=QUANTILE_METHOD
            )
        ),
        float(
            np.quantile(
                values, 1.0 - alpha / 2.0, method=QUANTILE_METHOD
            )
        ),
    ]


def lower_bound(values: np.ndarray, confidence: float = 0.95) -> float:
    return float(
        np.quantile(values, 1.0 - confidence, method=QUANTILE_METHOD)
    )


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} is non-finite")
    return number


def validate_inputs(
    *,
    plan_path: Path,
    variants_path: Path,
    recon_json_path: Path,
    recon_npz_path: Path,
    causal_path: Path,
    n5_gate_path: Path,
    model_manifest_path: Path,
    prereg_path: Path,
    code_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    prereg_sha = require_binding_preregistration(prereg_path)
    code_manifest_sha = verify_code_manifest(code_manifest_path, __file__)
    hashes = {
        "prereg_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": verify_sha256_sidecar(model_manifest_path),
        "plan_sha256": verify_sha256_sidecar(plan_path),
        "variants_sha256": verify_sha256_sidecar(variants_path),
        "recon_json_sha256": verify_sha256_sidecar(recon_json_path),
        "recon_npz_sha256": verify_sha256_sidecar(recon_npz_path),
        "causal_sha256": verify_sha256_sidecar(causal_path),
        "n5_gate_sha256": verify_sha256_sidecar(n5_gate_path),
    }
    if hashes["n5_gate_sha256"] != N5_GATE_V2_SHA256:
        raise ValueError("analysis requires the frozen N5 gate-v2 artifact")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    recon = json.loads(recon_json_path.read_text(encoding="utf-8"))
    causal = json.loads(causal_path.read_text(encoding="utf-8"))
    validate_variant_contract(variants)
    if plan.get("preregistration_sha256") != prereg_sha:
        raise ValueError("plan/preregistration hash mismatch")
    for key in ("code_manifest_sha256", "model_manifest_sha256"):
        if plan.get("inputs", {}).get(key) != hashes[key]:
            raise ValueError(f"plan/{key} mismatch")
    for key in (
        "plan_sha256",
        "prereg_sha256",
        "code_manifest_sha256",
        "model_manifest_sha256",
    ):
        if variants.get("inputs", {}).get(key) != hashes[key]:
            raise ValueError(f"variants/{key} mismatch")
    for key in (
        "plan_sha256",
        "variants_sha256",
        "prereg_sha256",
        "code_manifest_sha256",
        "model_manifest_sha256",
    ):
        if recon.get("inputs", {}).get(key) != hashes[key]:
            raise ValueError(f"reconstruction/{key} mismatch")
    if recon.get("outputs", {}).get("vecs_sha256") != hashes["recon_npz_sha256"]:
        raise ValueError("reconstruction JSON does not bind reconstruction NPZ")
    for key in (
        "plan_sha256",
        "variants_sha256",
        "recon_json_sha256",
        "recon_npz_sha256",
        "prereg_sha256",
        "code_manifest_sha256",
        "model_manifest_sha256",
    ):
        if causal.get("inputs", {}).get(key) != hashes[key]:
            raise ValueError(f"causal/{key} mismatch")
    if causal.get("status") != "complete":
        raise ValueError("causal artifact is incomplete")
    return plan, variants, recon, causal, hashes


def extract_rows(
    variants: dict[str, Any],
    recon: dict[str, Any],
    causal: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    float,
]:
    variant_rows = variants.get("rows")
    recon_rows = recon.get("rows")
    causal_rows = causal.get("rows")
    if not all(isinstance(rows, list) for rows in (variant_rows, recon_rows, causal_rows)):
        raise ValueError("variants, reconstruction, and causal artifacts require rows")
    assert isinstance(variant_rows, list)
    assert isinstance(recon_rows, list)
    assert isinstance(causal_rows, list)
    n = len(variant_rows)
    if n != N_ANALYSIS or len(recon_rows) != n or len(causal_rows) != n:
        raise ValueError("analysis artifact row counts differ")
    uids = [str(row.get("row_uid")) for row in variant_rows]
    if len(set(uids)) != n:
        raise ValueError("variant row_uids are not unique")
    if [str(row.get("row_uid")) for row in recon_rows] != uids:
        raise ValueError("reconstruction row order differs from variants")
    if [str(row.get("row_uid")) for row in causal_rows] != uids:
        raise ValueError("causal row order differs from variants")
    groups = [str(row.get("content_group_id")) for row in variant_rows]
    if len(set(groups)) != n:
        raise ValueError("analysis requires one unique content group per row")

    clamped: list[dict[str, Any]] = []
    values = {
        condition: np.empty(n, dtype=np.float64) for condition in CONDITIONS
    }
    a_meanmass = np.empty(n, dtype=np.float64)
    a_setmass = np.empty(n, dtype=np.float64)
    identity16_max = 0.0
    for index, (variant, row) in enumerate(zip(variant_rows, causal_rows)):
        uid = uids[index]
        for field in ("source", "candidate_count", "doc_id"):
            if str(row.get(field)) != str(variant.get(field)):
                raise ValueError(f"{uid} causal/variant mismatch at {field}")
        results = row.get("results")
        if not isinstance(results, dict) or set(results) != set(CONDITIONS):
            raise ValueError(f"{uid} causal conditions differ from frozen contract")
        for condition in CONDITIONS:
            result = results[condition]
            if int(result.get("n_positions", -1)) != 16:
                raise ValueError(f"{uid} {condition} does not report 16 positions")
            values[condition][index] = sanitize_kl(
                result.get("kl_at_pos"),
                row_uid=uid,
                condition=condition,
                field="kl_at_pos",
                clamped=clamped,
            )
            value16 = sanitize_kl(
                result.get("kl_mean_first16"),
                row_uid=uid,
                condition=condition,
                field="kl_mean_first16",
                clamped=clamped,
            )
            if condition == "identity":
                identity16_max = max(identity16_max, abs(value16))
            _finite_float(
                result.get("ce_first16"),
                f"{uid}.{condition}.ce_first16",
            )
        alignment = row.get("candidate_alignment")
        if not isinstance(alignment, dict):
            raise ValueError(f"{uid} lacks candidate alignment")
        if float(alignment.get("epsilon", -1)) != 1e-12:
            raise ValueError(f"{uid} candidate epsilon drift")
        n_true = int(alignment.get("n_unique_true", 0))
        n_cross = int(alignment.get("n_unique_cross", 0))
        p_true = _finite_float(
            alignment.get("p_true_setmass"), f"{uid}.p_true_setmass"
        )
        p_cross = _finite_float(
            alignment.get("p_cross_setmass"), f"{uid}.p_cross_setmass"
        )
        candidate_count = int(variant["candidate_count"])
        if not (
            1 <= n_true <= candidate_count
            and 1 <= n_cross <= candidate_count
            and 0.0 <= p_true <= 1.0
            and 0.0 <= p_cross <= 1.0
        ):
            raise ValueError(f"{uid} has invalid candidate masses/counts")
        a_meanmass[index] = np.log(p_true / n_true + 1e-12) - np.log(
            p_cross / n_cross + 1e-12
        )
        a_setmass[index] = np.log(p_true + 1e-12) - np.log(
            p_cross + 1e-12
        )
        if abs(a_meanmass[index] - float(alignment["a_meanmass_row"])) > 1e-12:
            raise ValueError(f"{uid} reported A_meanmass differs from raw masses")
        if abs(a_setmass[index] - float(alignment["a_setmass_row"])) > 1e-12:
            raise ValueError(f"{uid} reported A_setmass differs from raw masses")
    identity_max = float(np.max(np.abs(values["identity"])))
    if max(identity_max, identity16_max) > IDENTITY_KL_TOL:
        raise ValueError(
            "identity KL exceeds tolerance: "
            f"position={identity_max}, first16={identity16_max}, "
            f"tolerance={IDENTITY_KL_TOL}"
        )
    if float(values["zero"].sum()) <= 1e-12:
        raise ValueError("sum(KL_zero) is nonpositive")
    return (
        variant_rows,
        values,
        a_meanmass,
        a_setmass,
        clamped,
        identity16_max,
    )


def bootstrap_endpoints(
    values: dict[str, np.ndarray],
    a_meanmass: np.ndarray,
    a_setmass: np.ndarray,
) -> tuple[dict[str, np.ndarray], str]:
    n = len(a_meanmass)
    matrix_names = [*CONDITIONS, "a_meanmass", "a_setmass"]
    matrix = np.column_stack(
        [values[name] for name in CONDITIONS] + [a_meanmass, a_setmass]
    )
    column = {name: index for index, name in enumerate(matrix_names)}
    output = {
        name: np.empty(N_BOOT, dtype=np.float64)
        for name in (
            "g_specific",
            "g_content",
            "m_majority",
            "g_candidate_anchor",
            "raw_cross_minus_true",
            "a_meanmass",
            "a_setmass",
            "t_p3",
            *[f"r_{condition}" for condition in CONDITIONS],
        )
    }
    rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
    digest = hashlib.sha256()
    batch_size = 1000
    for start in range(0, N_BOOT, batch_size):
        stop = min(start + batch_size, N_BOOT)
        draws = rng.integers(
            0, n, size=(stop - start, n), dtype=np.int32
        )
        digest.update(draws.tobytes(order="C"))
        sums = matrix[draws].sum(axis=1, dtype=np.float64)
        zero = sums[:, column["zero"]]
        if np.any(zero <= 1e-12):
            raise ValueError("a bootstrap resample has nonpositive sum(KL_zero)")
        cross_minus_true = (
            sums[:, column["p3_cross_matched"]]
            - sums[:, column["p3_true"]]
        )
        content_minus_true = (
            sums[:, column["p3_candidate_strip"]]
            - sums[:, column["p3_true"]]
        )
        output["g_specific"][start:stop] = cross_minus_true / zero
        output["g_content"][start:stop] = content_minus_true / zero
        output["m_majority"][start:stop] = (
            cross_minus_true - 0.5 * content_minus_true
        ) / zero
        output["g_candidate_anchor"][start:stop] = (
            sums[:, column["p3_candidate_strip"]]
            - sums[:, column["p3_anchor_strip"]]
        ) / zero
        output["raw_cross_minus_true"][start:stop] = (
            cross_minus_true / n
        )
        output["a_meanmass"][start:stop] = (
            sums[:, column["a_meanmass"]] / n
        )
        output["a_setmass"][start:stop] = (
            sums[:, column["a_setmass"]] / n
        )
        for condition in CONDITIONS:
            output[f"r_{condition}"][start:stop] = (
                1.0 - sums[:, column[condition]] / zero
            )
        r_orig = output["r_orig"][start:stop]
        r_true = output["r_p3_true"][start:stop]
        output["t_p3"][start:stop] = np.divide(
            r_true,
            r_orig,
            out=np.full_like(r_true, np.nan),
            where=r_orig > 0.0,
        )
    return output, digest.hexdigest()


def point_endpoints(
    values: dict[str, np.ndarray],
    a_meanmass: np.ndarray,
    a_setmass: np.ndarray,
) -> dict[str, Any]:
    zero = float(values["zero"].sum(dtype=np.float64))
    cross_minus_true = values["p3_cross_matched"] - values["p3_true"]
    content_minus_true = values["p3_candidate_strip"] - values["p3_true"]
    recoveries = {
        condition: float(
            1.0 - values[condition].sum(dtype=np.float64) / zero
        )
        for condition in CONDITIONS
    }
    r_orig = recoveries["orig"]
    return {
        "sum_kl_zero": zero,
        "g_specific": float(cross_minus_true.sum() / zero),
        "g_content": float(content_minus_true.sum() / zero),
        "m_majority": float(
            (
                cross_minus_true.sum()
                - 0.5 * content_minus_true.sum()
            )
            / zero
        ),
        "g_candidate_anchor": float(
            (
                values["p3_candidate_strip"]
                - values["p3_anchor_strip"]
            ).sum()
            / zero
        ),
        "raw_cross_minus_true": float(cross_minus_true.mean()),
        "a_meanmass": float(a_meanmass.mean()),
        "a_setmass": float(a_setmass.mean()),
        "recoveries": recoveries,
        "t_p3": (
            float(recoveries["p3_true"] / r_orig)
            if r_orig > 0.0
            else None
        ),
    }


def endpoint_report(
    point: dict[str, Any],
    boot: dict[str, np.ndarray],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in (
        "g_specific",
        "g_content",
        "m_majority",
        "g_candidate_anchor",
        "raw_cross_minus_true",
        "a_meanmass",
        "a_setmass",
    ):
        report[name] = {
            "point": float(point[name]),
            "ci95_two_sided": interval(boot[name]),
        }
    recoveries = {}
    for condition, value in point["recoveries"].items():
        recoveries[condition] = {
            "point": float(value),
            "ci95_two_sided": interval(boot[f"r_{condition}"]),
        }
    t_finite = bool(np.isfinite(boot["t_p3"]).all())
    report["recoveries"] = recoveries
    report["t_p3"] = {
        "point": point["t_p3"],
        "testable": bool(point["t_p3"] is not None and t_finite),
        "one_sided_95_lower": (
            lower_bound(boot["t_p3"])
            if point["t_p3"] is not None and t_finite
            else None
        ),
        "ci95_two_sided": (
            interval(boot["t_p3"])
            if point["t_p3"] is not None and t_finite
            else None
        ),
        "bootstrap_nonpositive_r_orig_count": int(
            np.sum(boot["r_orig"] <= 0.0)
        ),
    }
    return report


def decisions(endpoints: dict[str, Any]) -> dict[str, Any]:
    t = endpoints["t_p3"]
    h6a_gates = {
        "g_specific_two_sided_95_lower_gt_0": (
            endpoints["g_specific"]["ci95_two_sided"][0] > 0.0
        ),
        "g_content_two_sided_95_lower_gt_0": (
            endpoints["g_content"]["ci95_two_sided"][0] > 0.0
        ),
        "t_p3_one_sided_95_lower_gt_0.90": bool(
            t["testable"]
            and t["one_sided_95_lower"] is not None
            and t["one_sided_95_lower"] > 0.90
        ),
    }
    h6a_pass = all(h6a_gates.values())
    h6b_gate = endpoints["a_meanmass"]["ci95_two_sided"][0] > 0.0
    majority_gate = endpoints["m_majority"]["ci95_two_sided"][0] > 0.0
    anchor_gate = (
        endpoints["g_candidate_anchor"]["ci95_two_sided"][0] > 0.0
    )
    headline = h6a_pass and h6b_gate
    return {
        "h6a": {
            "label": (
                "SAMPLE-SPECIFIC CHANNEL CONFIRMED"
                if h6a_pass
                else "NO SAMPLE-SPECIFIC CHANNEL CLAIM"
            ),
            "pass": h6a_pass,
            "gates": h6a_gates,
            "failed_gates": [
                name for name, passed in h6a_gates.items() if not passed
            ],
            "m_majority_is_not_a_gate": True,
        },
        "h6b": {
            "label": (
                "PREDICTIVE ALIGNMENT CONFIRMED"
                if h6b_gate
                else "NO PREDICTIVE ALIGNMENT CLAIM"
            ),
            "pass": h6b_gate,
            "gate": {
                "a_meanmass_two_sided_95_lower_gt_0": h6b_gate,
            },
            "a_setmass_is_secondary": True,
        },
        "majority_of_candidate_benefit": {
            "label": (
                "MAJORITY-OF-CANDIDATE-BENEFIT SUPPORTED"
                if majority_gate
                else "NO MAJORITY-OF-CANDIDATE-BENEFIT CLAIM"
            ),
            "pass": majority_gate,
            "gate": "M_majority two-sided 95% CI lower bound > 0",
            "confirmatory_h6a_gate": False,
        },
        "candidate_anchor_secondary": {
            "label": (
                "CANDIDATE DOMINANCE SUPPORTED"
                if anchor_gate
                else "NO CANDIDATE DOMINANCE CLAIM"
            ),
            "pass": anchor_gate,
        },
        "headline": {
            "label": (
                "SAMPLE-SPECIFIC NATURAL-LANGUAGE PREDICTIVE-STATE CODE CONFIRMED"
                if headline
                else "NO SAMPLE-SPECIFIC NATURAL-LANGUAGE PREDICTIVE-STATE CODE CLAIM"
            ),
            "pass": headline,
            "requires_h6a_and_h6b": True,
        },
    }


def subgroup_point(
    indices: np.ndarray,
    values: dict[str, np.ndarray],
    a_meanmass: np.ndarray,
) -> dict[str, Any]:
    if len(indices) == 0:
        raise ValueError("empty subgroup")
    zero = float(values["zero"][indices].sum())
    if zero <= 1e-12:
        return {"n": int(len(indices)), "status": "zero denominator"}
    cross_true = (
        values["p3_cross_matched"][indices]
        - values["p3_true"][indices]
    )
    content_true = (
        values["p3_candidate_strip"][indices]
        - values["p3_true"][indices]
    )
    return {
        "n": int(len(indices)),
        "g_specific": float(cross_true.sum() / zero),
        "g_content": float(content_true.sum() / zero),
        "a_meanmass": float(a_meanmass[indices].mean()),
    }


def diagnostics(
    variant_rows: list[dict[str, Any]],
    recon: dict[str, Any],
    causal: dict[str, Any],
    values: dict[str, np.ndarray],
    a_meanmass: np.ndarray,
) -> dict[str, Any]:
    sources: dict[str, list[int]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(variant_rows):
        sources[str(row["source"])].append(index)
        counts[str(int(row["candidate_count"]))].append(index)
    causal_rows = causal["rows"]
    hit_rates = {}
    for k in TOP_K:
        hit_rates[str(k)] = {
            side: float(
                np.mean(
                    [
                        bool(
                            row["candidate_alignment"]["hit_at_k"][str(k)][
                                side
                            ]
                        )
                        for row in causal_rows
                    ]
                )
            )
            for side in ("true", "cross")
        }
    recon_rows = recon["rows"]
    centered_cosine = {}
    for condition in (
        "orig",
        "p3_true",
        "p3_cross_matched",
        "p3_candidate_strip",
        "p3_anchor_strip",
        "p3_all_quote_strip",
        "p12",
        "sae_big",
    ):
        observed = [
            float(row["scores"][condition]["cos_c"]) for row in recon_rows
        ]
        centered_cosine[condition] = {
            "mean": float(np.mean(observed)),
            "median": float(np.median(observed)),
        }
    p3_lengths = np.asarray(
        [int(row["p3_token_length_true"]) for row in variant_rows],
        dtype=np.float64,
    )
    candidate_lengths = np.asarray(
        [
            int(length)
            for row in variant_rows
            for length in row["true_candidate_token_lengths"]
        ],
        dtype=np.float64,
    )
    total_candidate_lengths = np.asarray(
        [
            int(row["true_total_candidate_token_length"])
            for row in variant_rows
        ],
        dtype=np.float64,
    )

    def length_summary(values_: np.ndarray) -> dict[str, float]:
        return {
            "min": float(values_.min()),
            "median": float(np.median(values_)),
            "mean": float(values_.mean()),
            "max": float(values_.max()),
        }

    lexical_overlap: list[float] = []
    context_contains_target: list[bool] = []
    for row in variant_rows:
        true_words = set(
            re.findall(
                r"\w+",
                " ".join(str(value) for value in row["true_candidates"]),
                flags=re.UNICODE,
            )
        )
        cross_words = set(
            re.findall(
                r"\w+",
                " ".join(str(value) for value in row["cross_candidates"]),
                flags=re.UNICODE,
            )
        )
        union = true_words | cross_words
        lexical_overlap.append(
            len(true_words & cross_words) / len(union) if union else 0.0
        )
        target_normalized = str(row["target_token_normalized"])
        context_normalized = str(row["context_anchor_normalized"])
        context_contains_target.append(
            bool(
                target_normalized
                and target_normalized.casefold()
                in context_normalized.casefold()
            )
        )
    all_indices = np.arange(len(variant_rows), dtype=np.int64)
    leave_one_source_out = {}
    for source, source_indices in sorted(sources.items()):
        excluded = set(source_indices)
        retained = np.asarray(
            [index for index in all_indices if int(index) not in excluded],
            dtype=np.int64,
        )
        leave_one_source_out[source] = subgroup_point(
            retained, values, a_meanmass
        )
    return {
        "by_source": {
            key: subgroup_point(
                np.asarray(indices, dtype=np.int64), values, a_meanmass
            )
            for key, indices in sorted(sources.items())
        },
        "by_candidate_count": {
            key: subgroup_point(
                np.asarray(indices, dtype=np.int64), values, a_meanmass
            )
            for key, indices in sorted(counts.items())
        },
        "leave_one_source_out": leave_one_source_out,
        "token_lengths": {
            "p3_true": length_summary(p3_lengths),
            "individual_true_candidate": length_summary(candidate_lengths),
            "total_true_candidates_per_row": length_summary(
                total_candidate_lengths
            ),
        },
        "anchor_status": {
            "target_anchor_gate_pass_count": int(
                sum(
                    bool(row["target_anchor_gate_pass"])
                    for row in variant_rows
                )
            ),
            "target_anchor_gate_fail_count": int(
                sum(
                    not bool(row["target_anchor_gate_pass"])
                    for row in variant_rows
                )
            ),
            "context_anchor_contains_normalized_target_count": int(
                sum(context_contains_target)
            ),
            "context_anchor_is_diagnostic_only": True,
        },
        "true_cross_lexical_overlap": {
            "definition": (
                "Jaccard overlap of Unicode-word sets across concatenated "
                "true versus matched-cross candidate text"
            ),
            "min": float(np.min(lexical_overlap)),
            "median": float(np.median(lexical_overlap)),
            "mean": float(np.mean(lexical_overlap)),
            "max": float(np.max(lexical_overlap)),
        },
        "candidate_hit_at_k": hit_rates,
        "observed_next_token_membership": {
            "true_rate": float(
                np.mean(
                    [
                        row["candidate_alignment"][
                            "observed_next_token_in_true_canonical_set"
                        ]
                        for row in causal_rows
                    ]
                )
            ),
            "cross_rate": float(
                np.mean(
                    [
                        row["candidate_alignment"][
                            "observed_next_token_in_cross_canonical_set"
                        ]
                        for row in causal_rows
                    ]
                )
            ),
        },
        "centered_cosine_by_reconstruction": centered_cosine,
        "tails": {
            "cross_worse_than_true_by_gt_1_nat": int(
                np.sum(
                    values["p3_cross_matched"]
                    - values["p3_true"]
                    > 1.0
                )
            ),
            "true_worse_than_sae_big_by_gt_1_nat": int(
                np.sum(values["p3_true"] - values["sae_big"] > 1.0)
            ),
        },
    }


def analyze(
    plan: dict[str, Any],
    variants: dict[str, Any],
    recon: dict[str, Any],
    causal: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    (
        rows,
        values,
        a_meanmass,
        a_setmass,
        clamped,
        identity16_max,
    ) = extract_rows(variants, recon, causal)
    point = point_endpoints(values, a_meanmass, a_setmass)
    boot, draw_hash = bootstrap_endpoints(values, a_meanmass, a_setmass)
    endpoints = endpoint_report(point, boot)
    decision = decisions(endpoints)
    return {
        "schema_version": 1,
        "experiment": "N6 preregistered candidate-channel analysis",
        "status": "complete",
        "inputs": {
            **hashes,
            "analysis_script_sha256": sha256_file(__file__),
        },
        "cohort": {
            "n_rows": len(rows),
            "n_content_groups": len(
                {str(row["content_group_id"]) for row in rows}
            ),
            "by_source": dict(
                Counter(str(row["source"]) for row in rows)
            ),
            "selection": variants.get("selection"),
            "plan_status": plan.get("status"),
            "donor_hard_constraint_qa": validate_variant_contract(variants),
        },
        "formula_audit": {
            "g_specific": "sum(KL_cross-KL_true)/sum(KL_zero)",
            "g_content": "sum(KL_candidate_strip-KL_true)/sum(KL_zero)",
            "m_majority": (
                "[sum(KL_cross-KL_true)-0.5*"
                "sum(KL_candidate_strip-KL_true)]/sum(KL_zero)"
            ),
            "g_candidate_anchor": (
                "sum(KL_candidate_strip-KL_anchor_strip)/sum(KL_zero)"
            ),
            "recovery": "R_s=1-sum(KL_s)/sum(KL_zero)",
            "retention": "T_p3=R_p3_true/R_orig",
            "a_meanmass": (
                "mean(log(Ptrue/n_unique_true+1e-12)-"
                "log(Pcross/n_unique_cross+1e-12))"
            ),
            "a_setmass_secondary": (
                "mean(log(Ptrue+1e-12)-log(Pcross+1e-12))"
            ),
            "rowwise_kl_zero_division": False,
        },
        "bootstrap": {
            "n_resamples": N_BOOT,
            "seed": BOOT_SEED,
            "ordinary_content_group_bootstrap": True,
            "draws_shared_across_every_endpoint": True,
            "draw_indices_sha256": draw_hash,
            "quantile_method": QUANTILE_METHOD,
        },
        "numerical_qa": {
            "sum_kl_zero": point["sum_kl_zero"],
            "negative_kl_values_clamped": clamped,
            "identity_kl_at_pos_abs_max": float(
                np.max(np.abs(values["identity"]))
            ),
            "identity_kl16_abs_max": identity16_max,
            "identity_kl_tolerance": IDENTITY_KL_TOL,
        },
        "endpoints": endpoints,
        "decisions": decision,
        "mandatory_descriptives": diagnostics(
            rows, recon, causal, values, a_meanmass
        ),
    }


def render_markdown(output: dict[str, Any]) -> str:
    decisions_block = output["decisions"]
    endpoints = output["endpoints"]
    lines = [
        "# N6+ confirmatory analysis",
        "",
        f"- H6-A: **{decisions_block['h6a']['label']}**",
        f"- H6-B: **{decisions_block['h6b']['label']}**",
        f"- Headline: **{decisions_block['headline']['label']}**",
        (
            "- Majority secondary: **"
            f"{decisions_block['majority_of_candidate_benefit']['label']}**"
        ),
        "",
        "## Confirmatory endpoints",
        "",
    ]
    for name in ("g_specific", "g_content", "a_meanmass"):
        item = endpoints[name]
        lines.append(
            f"- `{name}` = {item['point']:.8g}; "
            f"two-sided 95% CI {item['ci95_two_sided']}"
        )
    t = endpoints["t_p3"]
    lines.append(
        f"- `t_p3` = {t['point']}; one-sided 95% lower "
        f"{t['one_sided_95_lower']}"
    )
    lines.extend(
        [
            "",
            "H6-A uses G_specific, G_content, and T_p3 only. "
            "M_majority is a named secondary endpoint. "
            "H6-B uses A_meanmass; A_setmass is mandatory secondary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--recon-json", required=True, type=Path)
    parser.add_argument("--recon-npz", required=True, type=Path)
    parser.add_argument("--causal", required=True, type=Path)
    parser.add_argument("--n5-gate", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    plan, variants, recon, causal, hashes = validate_inputs(
        plan_path=args.plan,
        variants_path=args.variants,
        recon_json_path=args.recon_json,
        recon_npz_path=args.recon_npz,
        causal_path=args.causal,
        n5_gate_path=args.n5_gate,
        model_manifest_path=args.model_manifest,
        prereg_path=args.prereg,
        code_manifest_path=args.code_manifest,
    )
    output = analyze(plan, variants, recon, causal, hashes)
    output_sha = write_new_json(args.out, output)
    markdown_sha = write_new_text(args.markdown, render_markdown(output))
    print(
        f"N6_ANALYSIS_COMPLETE rows={output['cohort']['n_rows']} "
        f"h6a={output['decisions']['h6a']['pass']} "
        f"h6b={output['decisions']['h6b']['pass']} "
        f"json={output_sha} markdown={markdown_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
