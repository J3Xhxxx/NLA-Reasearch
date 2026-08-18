#!/usr/bin/env python3
"""Independently audit N6 endpoints and every formal Stage-55 decision.

This module deliberately does not import the Stage-55 analysis module.  It
starts again from the frozen 400-row variants, reconstruction, and causal
artifacts; sanitizes raw KL values; regenerates the shared 50,000 PCG64
bootstrap draws; and checks the published Stage-55 endpoint tree to 1e-12.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ROWS = 400
N_BOOT = 50_000
BOOT_SEED = 20260803
QUANTILE_METHOD = "linear"
NEGATIVE_KL_TOL = -1e-7
IDENTITY_KL_TOL = 1e-5
COMPARISON_ATOL = 1e-12
EPSILON = 1e-12
TOP_K = (1, 5, 10, 50)
N5_GATE_V2_SHA256 = (
    "036477f21fb550b317978a880df0a708dcf42a5201d301ca0757fade3baea059"
)
EXPECTED_DRAW_INDICES_SHA256 = (
    "478a7789dfcac82b7b2c3663a60da82147e6014942fd29291e4c0a8a3688297e"
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
FORMAL_DECISION_NAMES = (
    "h6a",
    "h6b",
    "majority_of_candidate_benefit",
    "candidate_anchor_secondary",
    "headline",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"SHA-256 sidecar is missing: {sidecar}")
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(f"sidecar must contain exactly one line: {sidecar}")
    fields = lines[0].split()
    if len(fields) != 2:
        raise ValueError(f"malformed SHA-256 sidecar: {sidecar}")
    declared, name = fields
    if declared.lower() != digest or name != path.name:
        raise ValueError(f"SHA-256 sidecar mismatch: {sidecar}")
    return digest


def verify_binding_preregistration(path: Path) -> str:
    if ".DRAFT" in path.name.upper():
        raise ValueError("independent audit refuses a draft preregistration")
    prefix = "\n".join(path.read_text(encoding="utf-8").splitlines()[:20])
    if re.search(r"(?im)^status\s*:\s*.*\bdraft\b", prefix):
        raise ValueError("independent audit refuses DRAFT preregistration status")
    return verify_sidecar(path)


def verify_own_code_manifest(path: Path) -> str:
    manifest_sha = verify_sidecar(path)
    matches: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"malformed code manifest line {line_number}")
        if Path(fields[1].strip().lstrip("*")).name == Path(__file__).name:
            matches.append(fields[0].lower())
    if len(matches) != 1 or matches[0] != sha256_file(Path(__file__)):
        raise ValueError("independent audit differs from frozen code manifest")
    return manifest_sha


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def write_new_json(path: Path, value: Any) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} is non-finite")
    return number


def sanitize_kl(
    value: Any,
    *,
    row_uid: str,
    condition: str,
    field: str,
    clamped: list[dict[str, Any]],
) -> float:
    number = finite_float(value, f"{row_uid}.{condition}.{field}")
    if number < NEGATIVE_KL_TOL:
        raise ValueError(
            f"{row_uid} {condition}.{field}={number} is below "
            f"{NEGATIVE_KL_TOL}"
        )
    if number < 0.0:
        clamped.append(
            {
                "row_uid": row_uid,
                "condition": condition,
                "field": field,
                "raw_value": number,
            }
        )
        return 0.0
    return number


def require_rows(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        found = len(rows) if isinstance(rows, list) else None
        raise ValueError(f"{label} requires 400 rows, found {found}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must all be JSON objects")
    uids = [str(row.get("row_uid", "")) for row in rows]
    if any(not uid for uid in uids) or len(set(uids)) != EXPECTED_ROWS:
        raise ValueError(f"{label} requires 400 unique nonempty row_uids")
    return rows


def validate_npz_alignment(
    recon_npz_path: Path | None,
    uids: list[str],
    groups: list[str],
    recon: dict[str, Any],
) -> dict[str, Any]:
    if recon_npz_path is None:
        return {"checked": False}
    digest = sha256_file(recon_npz_path)
    declared = recon.get("outputs", {}).get("vecs_sha256")
    if declared is not None and str(declared) != digest:
        raise ValueError("reconstruction JSON does not bind supplied NPZ")
    with np.load(recon_npz_path, allow_pickle=False) as archive:
        if "row_uids" not in archive.files:
            raise ValueError("reconstruction NPZ lacks row_uids")
        npz_uids = [
            str(value) for value in np.asarray(archive["row_uids"]).tolist()
        ]
        if npz_uids != uids:
            raise ValueError("reconstruction NPZ row_uids are misaligned")
        groups_checked = "content_group_ids" in archive.files
        if groups_checked:
            npz_groups = [
                str(value)
                for value in np.asarray(archive["content_group_ids"]).tolist()
            ]
            if npz_groups != groups:
                raise ValueError(
                    "reconstruction NPZ content_group_ids are misaligned"
                )
    return {
        "checked": True,
        "sha256": digest,
        "row_uids_exact": True,
        "content_group_ids_exact": groups_checked,
    }


def extract_raw_estimands(
    variants: dict[str, Any],
    recon: dict[str, Any],
    causal: dict[str, Any],
    recon_npz_path: Path | None = None,
) -> dict[str, Any]:
    variant_rows = require_rows(variants, "variants")
    recon_rows = require_rows(recon, "reconstruction")
    causal_rows = require_rows(causal, "causal")
    if recon.get("status") not in ("COMPLETE", "complete"):
        raise ValueError("reconstruction artifact is incomplete")
    if causal.get("status") != "complete":
        raise ValueError("causal artifact is incomplete")

    uids = [str(row["row_uid"]) for row in variant_rows]
    groups = [str(row.get("content_group_id", "")) for row in variant_rows]
    if any(not group for group in groups) or len(set(groups)) != EXPECTED_ROWS:
        raise ValueError("variants require one unique content group per row")
    if [str(row["row_uid"]) for row in recon_rows] != uids:
        raise ValueError("reconstruction row order differs from variants")
    if [str(row["row_uid"]) for row in causal_rows] != uids:
        raise ValueError("causal row order differs from variants")
    if [str(row.get("content_group_id", "")) for row in recon_rows] != groups:
        raise ValueError("reconstruction content groups differ from variants")
    if [str(row.get("content_group_id", "")) for row in causal_rows] != groups:
        raise ValueError("causal content groups differ from variants")
    npz_qa = validate_npz_alignment(recon_npz_path, uids, groups, recon)

    values = {
        condition: np.empty(EXPECTED_ROWS, dtype=np.float64)
        for condition in CONDITIONS
    }
    values16 = {
        condition: np.empty(EXPECTED_ROWS, dtype=np.float64)
        for condition in CONDITIONS
    }
    a_meanmass = np.empty(EXPECTED_ROWS, dtype=np.float64)
    a_setmass = np.empty(EXPECTED_ROWS, dtype=np.float64)
    clamped: list[dict[str, Any]] = []
    for index, (variant, recon_row, causal_row) in enumerate(
        zip(variant_rows, recon_rows, causal_rows)
    ):
        uid = uids[index]
        for field in ("content_group_id", "source", "candidate_count", "doc_id"):
            expected = str(variant.get(field))
            if str(recon_row.get(field)) != expected:
                raise ValueError(f"{uid} reconstruction mismatch at {field}")
            if str(causal_row.get(field)) != expected:
                raise ValueError(f"{uid} causal mismatch at {field}")
        results = causal_row.get("results")
        if not isinstance(results, dict) or set(results) != set(CONDITIONS):
            raise ValueError(f"{uid} causal conditions differ from contract")
        for condition in CONDITIONS:
            result = results[condition]
            if not isinstance(result, dict):
                raise ValueError(f"{uid}.{condition} result is not an object")
            if int(result.get("n_positions", -1)) != 16:
                raise ValueError(f"{uid}.{condition} does not report 16 positions")
            values[condition][index] = sanitize_kl(
                result.get("kl_at_pos"),
                row_uid=uid,
                condition=condition,
                field="kl_at_pos",
                clamped=clamped,
            )
            values16[condition][index] = sanitize_kl(
                result.get("kl_mean_first16"),
                row_uid=uid,
                condition=condition,
                field="kl_mean_first16",
                clamped=clamped,
            )
            finite_float(
                result.get("ce_first16"),
                f"{uid}.{condition}.ce_first16",
            )

        alignment = causal_row.get("candidate_alignment")
        if not isinstance(alignment, dict):
            raise ValueError(f"{uid} lacks candidate_alignment")
        if finite_float(alignment.get("epsilon"), f"{uid}.epsilon") != EPSILON:
            raise ValueError(f"{uid} candidate-mass epsilon drift")
        n_true = int(alignment.get("n_unique_true", 0))
        n_cross = int(alignment.get("n_unique_cross", 0))
        p_true = finite_float(
            alignment.get("p_true_setmass"), f"{uid}.p_true_setmass"
        )
        p_cross = finite_float(
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
        a_meanmass[index] = np.log(p_true / n_true + EPSILON) - np.log(
            p_cross / n_cross + EPSILON
        )
        a_setmass[index] = np.log(p_true + EPSILON) - np.log(
            p_cross + EPSILON
        )
        if (
            abs(
                a_meanmass[index]
                - finite_float(
                    alignment.get("a_meanmass_row"),
                    f"{uid}.a_meanmass_row",
                )
            )
            > COMPARISON_ATOL
        ):
            raise ValueError(f"{uid} A_meanmass differs from raw masses")
        if (
            abs(
                a_setmass[index]
                - finite_float(
                    alignment.get("a_setmass_row"),
                    f"{uid}.a_setmass_row",
                )
            )
            > COMPARISON_ATOL
        ):
            raise ValueError(f"{uid} A_setmass differs from raw masses")

    identity_max = float(np.max(np.abs(values["identity"])))
    identity16_max = float(np.max(np.abs(values16["identity"])))
    if max(identity_max, identity16_max) > IDENTITY_KL_TOL:
        raise ValueError(
            "identity KL exceeds tolerance: "
            f"position={identity_max}, first16={identity16_max}"
        )
    zero_sum = float(values["zero"].sum(dtype=np.float64))
    if zero_sum <= EPSILON:
        raise ValueError("sum(KL_zero) is nonpositive")
    return {
        "rows": variant_rows,
        "uids": uids,
        "groups": groups,
        "values": values,
        "values16": values16,
        "a_meanmass": a_meanmass,
        "a_setmass": a_setmass,
        "clamped": clamped,
        "identity_max": identity_max,
        "identity16_max": identity16_max,
        "zero_sum": zero_sum,
        "npz_qa": npz_qa,
    }


def interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025, method=QUANTILE_METHOD)),
        float(np.quantile(values, 0.975, method=QUANTILE_METHOD)),
    ]


def lower_bound(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.05, method=QUANTILE_METHOD))


def shared_bootstrap(
    values: dict[str, np.ndarray],
    a_meanmass: np.ndarray,
    a_setmass: np.ndarray,
) -> tuple[dict[str, np.ndarray], str]:
    matrix_names = [*CONDITIONS, "a_meanmass", "a_setmass"]
    matrix = np.column_stack(
        [values[name] for name in CONDITIONS] + [a_meanmass, a_setmass]
    )
    column = {name: index for index, name in enumerate(matrix_names)}
    names = (
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
    output = {
        name: np.empty(N_BOOT, dtype=np.float64) for name in names
    }
    rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
    digest = hashlib.sha256()
    for start in range(0, N_BOOT, 1000):
        stop = min(start + 1000, N_BOOT)
        draws = rng.integers(
            0,
            EXPECTED_ROWS,
            size=(stop - start, EXPECTED_ROWS),
            dtype=np.int32,
        )
        digest.update(draws.tobytes(order="C"))
        sums = matrix[draws].sum(axis=1, dtype=np.float64)
        zero = sums[:, column["zero"]]
        if np.any(zero <= EPSILON):
            raise ValueError("bootstrap resample has nonpositive sum(KL_zero)")
        cross_true = (
            sums[:, column["p3_cross_matched"]]
            - sums[:, column["p3_true"]]
        )
        content_true = (
            sums[:, column["p3_candidate_strip"]]
            - sums[:, column["p3_true"]]
        )
        output["g_specific"][start:stop] = cross_true / zero
        output["g_content"][start:stop] = content_true / zero
        output["m_majority"][start:stop] = (
            cross_true - 0.5 * content_true
        ) / zero
        output["g_candidate_anchor"][start:stop] = (
            sums[:, column["p3_candidate_strip"]]
            - sums[:, column["p3_anchor_strip"]]
        ) / zero
        output["raw_cross_minus_true"][start:stop] = (
            cross_true / EXPECTED_ROWS
        )
        output["a_meanmass"][start:stop] = (
            sums[:, column["a_meanmass"]] / EXPECTED_ROWS
        )
        output["a_setmass"][start:stop] = (
            sums[:, column["a_setmass"]] / EXPECTED_ROWS
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


def point_estimates(
    values: dict[str, np.ndarray],
    a_meanmass: np.ndarray,
    a_setmass: np.ndarray,
) -> dict[str, Any]:
    zero = float(values["zero"].sum(dtype=np.float64))
    cross_true = values["p3_cross_matched"] - values["p3_true"]
    content_true = values["p3_candidate_strip"] - values["p3_true"]
    recoveries = {
        condition: float(
            1.0 - values[condition].sum(dtype=np.float64) / zero
        )
        for condition in CONDITIONS
    }
    return {
        "sum_kl_zero": zero,
        "g_specific": float(cross_true.sum(dtype=np.float64) / zero),
        "g_content": float(content_true.sum(dtype=np.float64) / zero),
        "m_majority": float(
            (
                cross_true.sum(dtype=np.float64)
                - 0.5 * content_true.sum(dtype=np.float64)
            )
            / zero
        ),
        "g_candidate_anchor": float(
            (
                values["p3_candidate_strip"]
                - values["p3_anchor_strip"]
            ).sum(dtype=np.float64)
            / zero
        ),
        "raw_cross_minus_true": float(cross_true.mean()),
        "a_meanmass": float(a_meanmass.mean()),
        "a_setmass": float(a_setmass.mean()),
        "recoveries": recoveries,
        "t_p3": (
            float(recoveries["p3_true"] / recoveries["orig"])
            if recoveries["orig"] > 0.0
            else None
        ),
    }


def endpoint_report(
    point: dict[str, Any], boot: dict[str, np.ndarray]
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
    report["recoveries"] = {
        condition: {
            "point": float(value),
            "ci95_two_sided": interval(boot[f"r_{condition}"]),
        }
        for condition, value in point["recoveries"].items()
    }
    finite_t = bool(np.isfinite(boot["t_p3"]).all())
    testable = bool(point["t_p3"] is not None and finite_t)
    report["t_p3"] = {
        "point": point["t_p3"],
        "testable": testable,
        "one_sided_95_lower": (
            lower_bound(boot["t_p3"]) if testable else None
        ),
        "ci95_two_sided": interval(boot["t_p3"]) if testable else None,
        "bootstrap_nonpositive_r_orig_count": int(
            np.sum(boot["r_orig"] <= 0.0)
        ),
    }
    return report


def formal_decisions(endpoints: dict[str, Any]) -> dict[str, Any]:
    t_p3 = endpoints["t_p3"]
    h6a_gates = {
        "g_specific_two_sided_95_lower_gt_0": (
            endpoints["g_specific"]["ci95_two_sided"][0] > 0.0
        ),
        "g_content_two_sided_95_lower_gt_0": (
            endpoints["g_content"]["ci95_two_sided"][0] > 0.0
        ),
        "t_p3_one_sided_95_lower_gt_0.90": bool(
            t_p3["testable"]
            and t_p3["one_sided_95_lower"] is not None
            and t_p3["one_sided_95_lower"] > 0.90
        ),
    }
    h6a_pass = all(h6a_gates.values())
    h6b_pass = endpoints["a_meanmass"]["ci95_two_sided"][0] > 0.0
    majority_pass = (
        endpoints["m_majority"]["ci95_two_sided"][0] > 0.0
    )
    anchor_pass = (
        endpoints["g_candidate_anchor"]["ci95_two_sided"][0] > 0.0
    )
    headline_pass = h6a_pass and h6b_pass
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
                if h6b_pass
                else "NO PREDICTIVE ALIGNMENT CLAIM"
            ),
            "pass": h6b_pass,
            "gate": {
                "a_meanmass_two_sided_95_lower_gt_0": h6b_pass,
            },
            "a_setmass_is_secondary": True,
        },
        "majority_of_candidate_benefit": {
            "label": (
                "MAJORITY-OF-CANDIDATE-BENEFIT SUPPORTED"
                if majority_pass
                else "NO MAJORITY-OF-CANDIDATE-BENEFIT CLAIM"
            ),
            "pass": majority_pass,
            "gate": "M_majority two-sided 95% CI lower bound > 0",
            "confirmatory_h6a_gate": False,
        },
        "candidate_anchor_secondary": {
            "label": (
                "CANDIDATE DOMINANCE SUPPORTED"
                if anchor_pass
                else "NO CANDIDATE DOMINANCE CLAIM"
            ),
            "pass": anchor_pass,
        },
        "headline": {
            "label": (
                "SAMPLE-SPECIFIC NATURAL-LANGUAGE PREDICTIVE-STATE CODE CONFIRMED"
                if headline_pass
                else "NO SAMPLE-SPECIFIC NATURAL-LANGUAGE PREDICTIVE-STATE CODE CLAIM"
            ),
            "pass": headline_pass,
            "requires_h6a_and_h6b": True,
        },
    }


def compare_tree(
    recomputed: Any,
    published: Any,
    *,
    path: str,
    numeric_checks: list[dict[str, Any]],
) -> None:
    if isinstance(recomputed, bool) or recomputed is None:
        if published != recomputed:
            raise ValueError(f"Stage-55 mismatch at {path}")
        return
    if isinstance(recomputed, (int, float, np.integer, np.floating)):
        if isinstance(published, bool):
            raise ValueError(f"Stage-55 type mismatch at {path}")
        observed = finite_float(published, f"published.{path}")
        expected = float(recomputed)
        error = abs(observed - expected)
        numeric_checks.append(
            {
                "path": path,
                "recomputed": expected,
                "stage55": observed,
                "absolute_error": error,
                "within_1e-12": error <= COMPARISON_ATOL,
            }
        )
        if error > COMPARISON_ATOL:
            raise ValueError(
                f"Stage-55 numerical mismatch at {path}: {error} > 1e-12"
            )
        return
    if isinstance(recomputed, str):
        if published != recomputed:
            raise ValueError(f"Stage-55 text mismatch at {path}")
        return
    if isinstance(recomputed, list):
        if not isinstance(published, list) or len(published) != len(recomputed):
            raise ValueError(f"Stage-55 list mismatch at {path}")
        for index, (expected, observed) in enumerate(
            zip(recomputed, published)
        ):
            compare_tree(
                expected,
                observed,
                path=f"{path}[{index}]",
                numeric_checks=numeric_checks,
            )
        return
    if isinstance(recomputed, dict):
        if not isinstance(published, dict) or set(published) != set(recomputed):
            raise ValueError(f"Stage-55 object-key mismatch at {path}")
        for key in recomputed:
            compare_tree(
                recomputed[key],
                published[key],
                path=f"{path}.{key}",
                numeric_checks=numeric_checks,
            )
        return
    raise TypeError(f"unsupported comparison value at {path}")


def compare_stage55(
    analysis: dict[str, Any],
    endpoints: dict[str, Any],
    decisions: dict[str, Any],
    draw_hash: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    if analysis.get("status") != "complete":
        raise ValueError("Stage-55 analysis artifact is incomplete")
    if int(analysis.get("cohort", {}).get("n_rows", -1)) != EXPECTED_ROWS:
        raise ValueError("Stage-55 analysis does not report 400 rows")
    published_bootstrap = analysis.get("bootstrap")
    if not isinstance(published_bootstrap, dict):
        raise ValueError("Stage-55 analysis lacks bootstrap metadata")
    frozen_bootstrap = {
        "n_resamples": N_BOOT,
        "seed": BOOT_SEED,
        "ordinary_content_group_bootstrap": True,
        "draws_shared_across_every_endpoint": True,
        "quantile_method": QUANTILE_METHOD,
    }
    for key, expected in frozen_bootstrap.items():
        if published_bootstrap.get(key) != expected:
            raise ValueError(f"Stage-55 bootstrap contract mismatch at {key}")
    published_hash = str(published_bootstrap.get("draw_indices_sha256", ""))
    if published_hash != draw_hash:
        raise ValueError(
            "Stage-55 bootstrap draw-byte SHA-256 differs from independent "
            "PCG64 regeneration"
        )

    endpoint_checks: list[dict[str, Any]] = []
    compare_tree(
        endpoints,
        analysis.get("endpoints"),
        path="endpoints",
        numeric_checks=endpoint_checks,
    )
    decision_numeric_checks: list[dict[str, Any]] = []
    compare_tree(
        decisions,
        analysis.get("decisions"),
        path="decisions",
        numeric_checks=decision_numeric_checks,
    )
    label_checks = {
        name: {
            "recomputed": decisions[name]["label"],
            "stage55": analysis["decisions"][name]["label"],
            "exact": (
                decisions[name]["label"]
                == analysis["decisions"][name]["label"]
            ),
        }
        for name in FORMAL_DECISION_NAMES
    }

    numerical = analysis.get("numerical_qa")
    if not isinstance(numerical, dict):
        raise ValueError("Stage-55 analysis lacks numerical_qa")
    numerical_checks: list[dict[str, Any]] = []
    compare_tree(
        raw["zero_sum"],
        numerical.get("sum_kl_zero"),
        path="numerical_qa.sum_kl_zero",
        numeric_checks=numerical_checks,
    )
    compare_tree(
        raw["identity_max"],
        numerical.get("identity_kl_at_pos_abs_max"),
        path="numerical_qa.identity_kl_at_pos_abs_max",
        numeric_checks=numerical_checks,
    )
    if numerical.get("negative_kl_values_clamped") != raw["clamped"]:
        raise ValueError("Stage-55 negative-KL clamp audit differs")
    return {
        "absolute_tolerance": COMPARISON_ATOL,
        "endpoint_numeric_leaf_checks": endpoint_checks,
        "endpoint_numeric_leaf_count": len(endpoint_checks),
        "all_endpoint_leaves_within_1e-12": True,
        "formal_decisions_exact": True,
        "formal_label_checks": label_checks,
        "bootstrap_draw_bytes": {
            "generator": "numpy.random.Generator(PCG64(20260803))",
            "dtype": "int32",
            "order": "C",
            "recomputed_sha256": draw_hash,
            "stage55_sha256": published_hash,
            "exact": True,
        },
        "numerical_qa_checks": numerical_checks,
        "negative_kl_clamp_records_exact": True,
        "all_checks_pass": True,
    }


def recompute(
    variants: dict[str, Any],
    recon: dict[str, Any],
    causal: dict[str, Any],
    *,
    recon_npz_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    raw = extract_raw_estimands(
        variants, recon, causal, recon_npz_path=recon_npz_path
    )
    point = point_estimates(
        raw["values"], raw["a_meanmass"], raw["a_setmass"]
    )
    boot, draw_hash = shared_bootstrap(
        raw["values"], raw["a_meanmass"], raw["a_setmass"]
    )
    endpoints = endpoint_report(point, boot)
    decisions = formal_decisions(endpoints)
    return endpoints, decisions, draw_hash, raw


def audit_artifacts(
    *,
    variants_path: Path,
    recon_json_path: Path,
    recon_npz_path: Path,
    causal_path: Path,
    analysis_path: Path,
) -> dict[str, Any]:
    hashes = {
        "variants_sha256": verify_sidecar(variants_path),
        "recon_json_sha256": verify_sidecar(recon_json_path),
        "recon_npz_sha256": verify_sidecar(recon_npz_path),
        "causal_sha256": verify_sidecar(causal_path),
        "analysis_sha256": verify_sidecar(analysis_path),
    }
    variants = read_object(variants_path)
    recon = read_object(recon_json_path)
    causal = read_object(causal_path)
    analysis = read_object(analysis_path)
    for name, expected in (
        ("variants_sha256", hashes["variants_sha256"]),
        ("recon_json_sha256", hashes["recon_json_sha256"]),
        ("recon_npz_sha256", hashes["recon_npz_sha256"]),
        ("causal_sha256", hashes["causal_sha256"]),
    ):
        declared = analysis.get("inputs", {}).get(name)
        if declared != expected:
            raise ValueError(f"Stage-55 input binding mismatch at {name}")

    endpoints, decisions, draw_hash, raw = recompute(
        variants,
        recon,
        causal,
        recon_npz_path=recon_npz_path,
    )
    comparison = compare_stage55(
        analysis, endpoints, decisions, draw_hash, raw
    )
    rows = raw["rows"]
    return {
        "schema_version": 1,
        "experiment": "N6 independent raw-artifact audit",
        "status": "complete",
        "inputs": {
            **hashes,
            "audit_script_sha256": sha256_file(Path(__file__)),
        },
        "cohort": {
            "n_rows": EXPECTED_ROWS,
            "n_content_groups": EXPECTED_ROWS,
            "row_order_exact_across_variants_recon_causal": True,
            "by_source": dict(
                Counter(str(row["source"]) for row in rows)
            ),
            "reconstruction_npz_alignment": raw["npz_qa"],
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
            "t_p3": "R_p3_true/R_orig",
            "a_meanmass": (
                "mean(log(Ptrue/ntrue+1e-12)-"
                "log(Pcross/ncross+1e-12))"
            ),
            "a_setmass": (
                "mean(log(Ptrue+1e-12)-log(Pcross+1e-12))"
            ),
            "rowwise_kl_zero_division": False,
        },
        "bootstrap": {
            "n_resamples": N_BOOT,
            "seed": BOOT_SEED,
            "bit_generator": "PCG64",
            "ordinary_content_group_bootstrap": True,
            "draws_shared_across_every_endpoint": True,
            "draw_indices_sha256": draw_hash,
            "quantile_method": QUANTILE_METHOD,
        },
        "numerical_qa": {
            "sum_kl_zero": raw["zero_sum"],
            "negative_kl_values_clamped": raw["clamped"],
            "identity_kl_at_pos_abs_max": raw["identity_max"],
            "identity_kl16_abs_max": raw["identity16_max"],
            "identity_kl_tolerance": IDENTITY_KL_TOL,
        },
        "endpoints": endpoints,
        "decisions": decisions,
        "stage55_comparison": comparison,
    }


def synthetic_payloads(
    *,
    positive: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    variant_rows: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    if positive:
        kl = {
            "identity": 0.0,
            "orig": 2.0,
            "p3_true": 1.0,
            "p3_cross_matched": 3.0,
            "p3_candidate_strip": 4.0,
            "p3_anchor_strip": 2.0,
            "p3_all_quote_strip": 5.0,
            "p12": 3.0,
            "sae_big": 4.0,
            "zero": 10.0,
        }
        p_true, p_cross = 0.4, 0.1
    else:
        kl = {
            "identity": 0.0,
            "orig": 2.0,
            "p3_true": 3.0,
            "p3_cross_matched": 2.0,
            "p3_candidate_strip": 2.5,
            "p3_anchor_strip": 3.0,
            "p3_all_quote_strip": 4.0,
            "p12": 3.0,
            "sae_big": 4.0,
            "zero": 10.0,
        }
        p_true, p_cross = 0.1, 0.4
    a_mean = float(
        np.log(p_true / 2 + EPSILON)
        - np.log(p_cross / 2 + EPSILON)
    )
    a_set = float(np.log(p_true + EPSILON) - np.log(p_cross + EPSILON))
    for index in range(EXPECTED_ROWS):
        common = {
            "row_uid": f"synthetic-{index:04d}",
            "content_group_id": f"group-{index:04d}",
            "source": f"source-{index % 13:02d}",
            "candidate_count": 4 + index % 3,
            "doc_id": index,
        }
        variant_rows.append(dict(common))
        recon_rows.append(dict(common))
        results = {
            condition: {
                "kl_at_pos": value,
                "kl_mean_first16": (
                    -5e-8
                    if index == 0 and condition == "identity"
                    else value
                ),
                "ce_first16": 1.0 + value,
                "n_positions": 16,
            }
            for condition, value in kl.items()
        }
        causal_rows.append(
            {
                **common,
                "results": results,
                "candidate_alignment": {
                    "epsilon": EPSILON,
                    "n_unique_true": 2,
                    "n_unique_cross": 2,
                    "p_true_setmass": p_true,
                    "p_cross_setmass": p_cross,
                    "a_meanmass_row": a_mean,
                    "a_setmass_row": a_set,
                },
            }
        )
    return (
        {
            "status": "ANALYSIS_COHORT_FROZEN",
            "rows": variant_rows,
        },
        {"status": "COMPLETE", "rows": recon_rows},
        {"status": "complete", "rows": causal_rows},
    )


def synthetic_stage55(
    endpoints: dict[str, Any],
    decisions: dict[str, Any],
    draw_hash: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "complete",
        "cohort": {"n_rows": EXPECTED_ROWS},
        "bootstrap": {
            "n_resamples": N_BOOT,
            "seed": BOOT_SEED,
            "ordinary_content_group_bootstrap": True,
            "draws_shared_across_every_endpoint": True,
            "draw_indices_sha256": draw_hash,
            "quantile_method": QUANTILE_METHOD,
        },
        "numerical_qa": {
            "sum_kl_zero": raw["zero_sum"],
            "negative_kl_values_clamped": raw["clamped"],
            "identity_kl_at_pos_abs_max": raw["identity_max"],
        },
        "endpoints": copy.deepcopy(endpoints),
        "decisions": copy.deepcopy(decisions),
    }


def run_self_test() -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    hashes: list[str] = []
    for name, positive in (("positive", True), ("negative", False)):
        variants, recon, causal = synthetic_payloads(positive=positive)
        endpoints, decisions, draw_hash, raw = recompute(
            variants, recon, causal
        )
        published = synthetic_stage55(
            endpoints, decisions, draw_hash, raw
        )
        comparison = compare_stage55(
            published, endpoints, decisions, draw_hash, raw
        )
        expected_pass = positive
        checks = {
            "h6a_expected": decisions["h6a"]["pass"] is expected_pass,
            "h6b_expected": decisions["h6b"]["pass"] is expected_pass,
            "headline_expected": (
                decisions["headline"]["pass"] is expected_pass
            ),
            "majority_expected": (
                decisions["majority_of_candidate_benefit"]["pass"]
                is expected_pass
            ),
            "candidate_anchor_expected": (
                decisions["candidate_anchor_secondary"]["pass"]
                is expected_pass
            ),
            "stage55_tree_match": comparison["all_checks_pass"],
            "negative_roundoff_was_clamped": len(raw["clamped"]) == 1,
        }
        if not all(checks.values()):
            raise AssertionError(f"{name} synthetic checks failed: {checks}")
        hashes.append(draw_hash)
        scenarios[name] = {
            "checks": checks,
            "labels": {
                decision: decisions[decision]["label"]
                for decision in FORMAL_DECISION_NAMES
            },
            "selected_points": {
                endpoint: endpoints[endpoint]["point"]
                for endpoint in (
                    "g_specific",
                    "g_content",
                    "m_majority",
                    "g_candidate_anchor",
                    "a_meanmass",
                    "a_setmass",
                )
            },
            "t_p3": endpoints["t_p3"]["point"],
        }
    if hashes[0] != hashes[1]:
        raise AssertionError("shared PCG64 draw-byte hash changed by scenario")
    if hashes[0] != EXPECTED_DRAW_INDICES_SHA256:
        raise AssertionError(
            "PCG64 draw-byte hash differs from the frozen 400x50,000 contract"
        )
    rejected = False
    try:
        sanitize_kl(
            -2e-7,
            row_uid="synthetic",
            condition="zero",
            field="kl_at_pos",
            clamped=[],
        )
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("KL value below frozen tolerance was not rejected")
    return {
        "status": "PASS",
        "n_rows": EXPECTED_ROWS,
        "n_resamples": N_BOOT,
        "seed": BOOT_SEED,
        "bit_generator": "PCG64",
        "draw_indices_sha256": hashes[0],
        "below_tolerance_negative_kl_rejected": True,
        "scenarios": scenarios,
    }


def validate_frozen_cli(args: argparse.Namespace) -> None:
    expected = {
        "selection_seed": BOOT_SEED,
        "analysis_target": EXPECTED_ROWS,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "quantile_method": QUANTILE_METHOD,
        "negative_kl_tol": NEGATIVE_KL_TOL,
        "identity_kl_tol": IDENTITY_KL_TOL,
        "top_k": ",".join(str(value) for value in TOP_K),
    }
    for name, value in expected.items():
        observed = getattr(args, name)
        if observed is not None and observed != value:
            raise ValueError(
                f"--{name.replace('_', '-')}={observed!r} differs from "
                f"frozen value {value!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", "--variants-donor", dest="variants", type=Path)
    parser.add_argument("--recon-json", type=Path)
    parser.add_argument("--recon-npz", type=Path)
    parser.add_argument(
        "--causal", "--causal-candidate-mass", dest="causal", type=Path
    )
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")

    # Accepted for the frozen runner contract and independently validated where
    # numerical values are protocol constants.
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--n5-gate", type=Path)
    parser.add_argument("--parser-config", type=Path)
    parser.add_argument("--donor-cost-config", type=Path)
    parser.add_argument("--model-manifest", "--manifest", dest="model_manifest", type=Path)
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--prereg", type=Path)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--analysis-target", type=int)
    parser.add_argument("--min-cell-size", type=int)
    parser.add_argument("--top-k")
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--quantile-method")
    parser.add_argument("--negative-kl-tol", type=float)
    parser.add_argument("--identity-kl-tol", type=float)
    args = parser.parse_args()
    validate_frozen_cli(args)

    if args.self_test:
        artifact_args = (
            args.variants,
            args.recon_json,
            args.recon_npz,
            args.causal,
            args.analysis,
            args.out,
        )
        if any(path is not None for path in artifact_args):
            parser.error("--self-test cannot be combined with artifact paths")
        output = run_self_test()
        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        return

    required = {
        "--variants": args.variants,
        "--recon-json": args.recon_json,
        "--recon-npz": args.recon_npz,
        "--causal": args.causal,
        "--analysis": args.analysis,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    provenance_required = {
        "--plan": args.plan,
        "--n5-gate": args.n5_gate,
        "--model-manifest": args.model_manifest,
        "--code-manifest": args.code_manifest,
        "--prereg": args.prereg,
    }
    provenance_missing = [
        name for name, value in provenance_required.items() if value is None
    ]
    if provenance_missing:
        parser.error(
            "artifact audit requires provenance arguments: "
            + ", ".join(provenance_missing)
        )
    assert args.variants is not None
    assert args.recon_json is not None
    assert args.recon_npz is not None
    assert args.causal is not None
    assert args.analysis is not None
    assert args.out is not None
    assert args.plan is not None
    assert args.n5_gate is not None
    assert args.model_manifest is not None
    assert args.code_manifest is not None
    assert args.prereg is not None
    provenance_hashes = {
        "plan_sha256": verify_sidecar(args.plan),
        "n5_gate_sha256": verify_sidecar(args.n5_gate),
        "model_manifest_sha256": verify_sidecar(args.model_manifest),
        "code_manifest_sha256": verify_own_code_manifest(
            args.code_manifest
        ),
        "prereg_sha256": verify_binding_preregistration(args.prereg),
    }
    if provenance_hashes["n5_gate_sha256"] != N5_GATE_V2_SHA256:
        raise ValueError("independent audit requires frozen N5 gate-v2")
    analysis_payload = read_object(args.analysis)
    for name, expected in provenance_hashes.items():
        if analysis_payload.get("inputs", {}).get(name) != expected:
            raise ValueError(
                f"Stage-55 provenance mismatch at {name}: "
                f"{analysis_payload.get('inputs', {}).get(name)!r} != {expected!r}"
            )
    output = audit_artifacts(
        variants_path=args.variants,
        recon_json_path=args.recon_json,
        recon_npz_path=args.recon_npz,
        causal_path=args.causal,
        analysis_path=args.analysis,
    )
    output["inputs"].update(provenance_hashes)
    digest = write_new_json(args.out, output)
    print(
        "N6_INDEPENDENT_AUDIT_COMPLETE "
        f"rows={output['cohort']['n_rows']} "
        f"h6a={output['decisions']['h6a']['pass']} "
        f"h6b={output['decisions']['h6b']['pass']} "
        f"sha256={digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
