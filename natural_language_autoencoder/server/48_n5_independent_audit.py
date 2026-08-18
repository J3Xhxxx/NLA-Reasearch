#!/usr/bin/env python3
"""Independently recompute the frozen N5 held-out confirmatory endpoints.

This script intentionally does not import ``47_n5_analyze.py``.  It consumes
only the frozen gate, held-out reconstruction JSON/NPZ pair, and held-out
causal rows.  Routes are recomputed from the NPZ vectors and frozen gate; the
published per-row routes and causal summaries are never used as estimands.

Frozen formulas (N5 preregistration v1, retained verbatim by v2):

* H5-A ``G = sum(KL_big - KL_hybrid) / sum(KL_zero)``.
* H5-A ``Delta_raw = mean(KL_big - KL_hybrid)``.
* Catastrophic regret is strictly ``KL_hybrid - KL_big > 1.0`` nat, with a
  one-sided 95% exact Clopper-Pearson upper bound.
* H5-B ``G_p3_p12 = sum(KL_p12 - KL_p3) / sum(KL_zero)``.
* ``R_s = 1 - sum(KL_s) / sum(KL_zero)`` and ``T = R_p3 / R_orig``.
* Every bootstrap uses the same 50,000 content-group resamples, stratified by
  Pile versus XNLI, with NumPy Generator seed 20260730.  The gate is never
  retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ROWS = 400
EXPECTED_CORPUS_COUNTS = Counter({"pile": 300, "xnli": 100})
N_BOOT = 50_000
SEED = 20260730
NEGATIVE_KL_TOL = -1e-7
IDENTITY_KL_TOL = 1e-5
CAT_REGRET_NAT = 1.0


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uid_sha256(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def scalar_string(archive: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in archive.files:
        return None
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"NPZ {key} must be a scalar, found {value.shape}")
    return str(value.item())


def row_uid(row: dict[str, Any]) -> str:
    uid = row.get("row_uid")
    if uid is None or not str(uid):
        raise ValueError("every row must contain a nonempty row_uid")
    return str(uid)


def require_heldout_rows(payload: dict[str, Any], label: str) -> list[dict]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        found = len(rows) if isinstance(rows, list) else None
        raise ValueError(f"{label} requires 400 rows, found {found}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be JSON objects")
    uids = [row_uid(row) for row in rows]
    if len(set(uids)) != EXPECTED_ROWS:
        raise ValueError(f"{label} row_uid values are not unique")
    declared_splits = {
        str(row["split"]) for row in rows if row.get("split") is not None
    }
    if declared_splits and declared_splits != {"heldout"}:
        raise ValueError(f"{label} has unexpected splits {declared_splits}")
    groups = [
        str(row["content_group_id"])
        for row in rows
        if row.get("content_group_id") is not None
    ]
    if groups and (
        len(groups) != EXPECTED_ROWS or len(set(groups)) != EXPECTED_ROWS
    ):
        raise ValueError(f"{label} is not one row per content group")
    return rows


def centered_cosine(
    prediction: np.ndarray, target: np.ndarray, mean_direction: np.ndarray
) -> np.ndarray:
    """Absolute NLA router score, recomputed in float64 from raw arrays."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    direction = np.asarray(mean_direction, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            f"prediction/target shapes differ: {prediction.shape}, {target.shape}"
        )
    if direction.shape != (target.shape[1],):
        raise ValueError(
            f"mean direction shape {direction.shape} != {(target.shape[1],)}"
        )
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        raise ValueError(f"frozen discovery mean direction is not unit: {norm}")
    pred_c = prediction - np.outer(prediction @ direction, direction)
    target_c = target - np.outer(target @ direction, direction)
    denominator = np.linalg.norm(pred_c, axis=1) * np.linalg.norm(
        target_c, axis=1
    )
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 1e-12):
        bad = np.flatnonzero(
            (~np.isfinite(denominator)) | (denominator <= 1e-12)
        )[:10].tolist()
        raise ValueError(f"centered-cosine denominator failed at rows {bad}")
    answer = np.sum(pred_c * target_c, axis=1) / denominator
    if not np.isfinite(answer).all():
        raise ValueError("centered cosine contains non-finite values")
    return answer


def sanitize_kl(
    value: Any,
    uid: str,
    condition: str,
    field: str = "kl_at_pos",
) -> tuple[float, bool]:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{uid} {condition}.{field} is non-finite")
    if number < NEGATIVE_KL_TOL:
        raise ValueError(
            f"{uid} {condition}.{field}={number} is below -1e-7"
        )
    if number < 0:
        return 0.0, True
    return number, False


def binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return P[X <= successes], X~Binomial(trials, probability)."""
    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        + k * log_p
        + (trials - k) * log_q
        for k in range(successes + 1)
    ]
    maximum = max(terms)
    return math.exp(maximum) * sum(math.exp(term - maximum) for term in terms)


def clopper_pearson_upper(
    successes: int, trials: int, confidence: float = 0.95
) -> float:
    """Exact one-sided binomial upper bound via monotone CDF inversion."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial successes/trials")
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(120):
        middle = (low + high) / 2.0
        if binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def stratified_bootstrap_sums(
    values: np.ndarray,
    strata: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    chunk_size: int = 500,
) -> np.ndarray:
    """Return bootstrap sums for every column using one frozen draw set.

    Candidate indices for an entire stratum are generated in one call before
    moving to the next sorted stratum.  This deliberately matches the frozen
    seed/stratum draw ordering while the statistic implementation is otherwise
    independent of script 47.
    """
    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(strata, dtype=object)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("bootstrap values/strata have incompatible shapes")
    if not np.isfinite(matrix).all():
        raise ValueError("bootstrap values contain non-finite entries")
    rng = np.random.default_rng(seed)
    output = np.zeros((n_boot, matrix.shape[1]), dtype=np.float64)
    for name in sorted(set(labels.tolist())):
        positions = np.flatnonzero(labels == name)
        if not len(positions):
            raise ValueError(f"empty bootstrap stratum {name!r}")
        local = rng.integers(
            0,
            len(positions),
            size=(n_boot, len(positions)),
            dtype=np.int32,
        )
        stratum_values = matrix[positions]
        for start in range(0, n_boot, chunk_size):
            stop = min(start + chunk_size, n_boot)
            # Sum one endpoint at a time.  Besides keeping the operation
            # transparent, this fixes floating-point reduction order to the
            # natural row order for every endpoint.
            draws = local[start:stop]
            for column in range(matrix.shape[1]):
                output[start:stop, column] += stratum_values[
                    :, column
                ][draws].sum(axis=1)
        del local
    return output


def interval(values: np.ndarray, level: float) -> list[float]:
    alpha = (1.0 - level) / 2.0
    return [
        float(np.quantile(values, alpha, method="linear")),
        float(np.quantile(values, 1.0 - alpha, method="linear")),
    ]


def lower_bound(values: np.ndarray, confidence: float = 0.95) -> float:
    return float(
        np.quantile(values, 1.0 - confidence, method="linear")
    )


def normalize_corpus(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "pile":
        return "pile"
    if text == "xnli":
        return "xnli"
    raise ValueError(f"unexpected held-out corpus label {value!r}")


def audit_artifacts(
    gate_path: Path,
    recon_json_path: Path,
    recon_npz_path: Path,
    causal_path: Path,
) -> dict[str, Any]:
    gate_sha = sha256_file(gate_path)
    recon_json_sha = sha256_file(recon_json_path)
    recon_npz_sha = sha256_file(recon_npz_path)
    causal_sha = sha256_file(causal_path)

    gate_payload = read_object(gate_path)
    recon = read_object(recon_json_path)
    causal = read_object(causal_path)
    gate = gate_payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("gate JSON lacks a top-level gate object")
    if gate_payload.get("status") != "complete":
        raise ValueError("gate artifact is not complete")
    if gate_payload.get("split") not in (None, "discovery"):
        raise ValueError("gate was not frozen on discovery")
    feasible = bool(gate.get("feasible"))
    expected_gate_status = "FEASIBLE" if feasible else "GATE TRAINING FAILURE"
    if gate.get("status") != expected_gate_status:
        raise ValueError("gate status/feasible fields disagree")
    if gate.get("score_name") != "absolute_nla_centered_cosine":
        raise ValueError("gate uses a non-preregistered score")

    recon_rows = require_heldout_rows(recon, "held-out reconstruction")
    causal_rows = require_heldout_rows(causal, "held-out causal")
    if recon.get("status") != "COMPLETE":
        raise ValueError("held-out reconstruction is not a full COMPLETE run")
    if causal.get("status") != "complete":
        raise ValueError("held-out causal artifact is not complete")
    if recon.get("split") not in (None, "heldout"):
        raise ValueError("reconstruction is not heldout")
    if causal.get("split") not in (None, "heldout"):
        raise ValueError("causal artifact is not heldout")

    declared_npz = recon.get("outputs", {}).get("vecs_sha256")
    if declared_npz is not None and str(declared_npz) != recon_npz_sha:
        raise ValueError("reconstruction JSON does not bind the supplied NPZ")
    recon_gate = recon.get("inputs", {}).get("gate_sha256")
    if recon_gate is not None and str(recon_gate) != gate_sha:
        raise ValueError("reconstruction JSON binds a different gate")
    causal_gate = causal.get("inputs", {}).get("gate_sha256")
    if causal_gate is not None and str(causal_gate) != gate_sha:
        raise ValueError("causal JSON binds a different gate")
    causal_recon = causal.get("inputs", {}).get("recon_sha256")
    if causal_recon is not None and str(causal_recon) != recon_npz_sha:
        raise ValueError("causal JSON binds a different reconstruction NPZ")

    recon_uids = [row_uid(row) for row in recon_rows]
    with np.load(recon_npz_path, allow_pickle=False) as archive:
        required = {"x", "pred_orig", "m_hat", "row_uids"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"held-out NPZ lacks keys {missing}")
        x = np.asarray(archive["x"], dtype=np.float64)
        pred_orig = np.asarray(archive["pred_orig"], dtype=np.float64)
        stored_m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
        npz_uids = [
            str(value) for value in np.asarray(archive["row_uids"]).tolist()
        ]
        npz_q = (
            np.asarray(archive["q_router"], dtype=np.float64)
            if "q_router" in archive.files
            else None
        )
        npz_route = (
            np.asarray(archive["route_nla"], dtype=bool)
            if "route_nla" in archive.files
            else None
        )
        channel_vector_keys = {
            "pred_p3_only",
            "pred_p12",
            "pred_quote_strip_p3",
        }
        npz_channel_active = channel_vector_keys <= set(archive.files)
        npz_channel_partial = bool(
            channel_vector_keys & set(archive.files)
        ) and not npz_channel_active
        embedded_gate_sha = scalar_string(archive, "gate_sha256")
    if x.shape[0] != EXPECTED_ROWS or x.ndim != 2:
        raise ValueError(f"unexpected held-out x shape {x.shape}")
    if pred_orig.shape != x.shape:
        raise ValueError("pred_orig and x shapes differ")
    if npz_uids != recon_uids:
        raise ValueError("NPZ row_uids do not match reconstruction JSON order")
    if npz_channel_partial:
        raise ValueError("held-out NPZ has a partial paragraph channel")
    if embedded_gate_sha is not None and embedded_gate_sha != gate_sha:
        raise ValueError("held-out NPZ embeds a different gate hash")

    frozen_m_hat = np.asarray(
        gate.get("discovery_mean_direction"), dtype=np.float64
    )
    if stored_m_hat.shape != frozen_m_hat.shape or not np.array_equal(
        stored_m_hat, frozen_m_hat
    ):
        max_error = (
            float(np.max(np.abs(stored_m_hat - frozen_m_hat)))
            if stored_m_hat.shape == frozen_m_hat.shape
            else None
        )
        raise ValueError(
            "held-out NPZ m_hat differs from frozen gate direction "
            f"(max_abs={max_error})"
        )
    q = centered_cosine(pred_orig, x, frozen_m_hat)
    if feasible:
        threshold = float(gate["threshold"])
        tie_cutoff = str(gate["tie_hash_cutoff_inclusive"])
        if not np.isfinite(threshold):
            raise ValueError("frozen threshold is non-finite")
        if (
            len(tie_cutoff) != 64
            or any(character not in "0123456789abcdef" for character in tie_cutoff)
        ):
            raise ValueError("malformed frozen row-UID tie hash")
        uid_hashes = np.asarray(
            [uid_sha256(uid) for uid in recon_uids], dtype="<U64"
        )
        route_nla = (q > threshold) | (
            (q == threshold) & (uid_hashes <= tie_cutoff)
        )
    else:
        threshold = None
        tie_cutoff = None
        route_nla = np.zeros(EXPECTED_ROWS, dtype=bool)

    q_errors = [
        abs(float(row["q"]) - float(q[index]))
        for index, row in enumerate(recon_rows)
        if row.get("q") is not None
    ]
    if len(q_errors) != EXPECTED_ROWS or max(q_errors) > 5e-7:
        raise ValueError(
            "reconstruction JSON q does not match independently recomputed q"
        )
    json_routes = [
        row.get("route_nla") for row in recon_rows
    ]
    if not all(isinstance(value, bool) for value in json_routes):
        raise ValueError("reconstruction JSON omits boolean route_nla values")
    if not np.array_equal(np.asarray(json_routes, dtype=bool), route_nla):
        raise ValueError("published reconstruction routes fail frozen gate")
    expected_destinations = [
        "nla" if selected else "sae_big" for selected in route_nla
    ]
    if [
        str(row.get("routed_to")) for row in recon_rows
    ] != expected_destinations:
        raise ValueError("published routed_to values fail frozen gate")
    if npz_q is not None and (
        npz_q.shape != q.shape
        or not np.allclose(npz_q, q, rtol=1e-7, atol=5e-7)
    ):
        raise ValueError("NPZ q_router differs from raw-vector score")
    if npz_route is not None and (
        npz_route.shape != route_nla.shape
        or not np.array_equal(npz_route, route_nla)
    ):
        raise ValueError("NPZ route_nla differs from frozen gate")

    causal_by_uid = {row_uid(row): row for row in causal_rows}
    if set(causal_by_uid) != set(recon_uids):
        raise ValueError("reconstruction and causal row UID sets differ")
    aligned_causal = [causal_by_uid[uid] for uid in recon_uids]
    condition_sets = [set(row.get("results", {})) for row in aligned_causal]
    if any(current != condition_sets[0] for current in condition_sets[1:]):
        raise ValueError("causal rows have inconsistent condition sets")
    available = condition_sets[0]
    required_causal = {"identity", "orig", "sae_big", "zero"}
    if not required_causal <= available:
        raise ValueError(
            f"causal rows lack {sorted(required_causal - available)}"
        )

    kl_by_condition: dict[str, np.ndarray] = {}
    kl16_by_condition: dict[str, np.ndarray] = {}
    negative_clamped: list[dict[str, Any]] = []
    for condition in sorted(available):
        values = np.empty(EXPECTED_ROWS, dtype=np.float64)
        values16 = np.empty(EXPECTED_ROWS, dtype=np.float64)
        for index, (uid, row) in enumerate(zip(recon_uids, aligned_causal)):
            result = row["results"].get(condition)
            if (
                not isinstance(result, dict)
                or "kl_at_pos" not in result
                or "kl_mean_first16" not in result
            ):
                raise ValueError(
                    f"{uid} lacks {condition} KL-at-position/KL16"
                )
            values[index], clamped = sanitize_kl(
                result["kl_at_pos"], uid, condition
            )
            if clamped:
                negative_clamped.append(
                    {
                        "row_uid": uid,
                        "condition": condition,
                        "field": "kl_at_pos",
                        "raw_value": float(result["kl_at_pos"]),
                    }
                )
            values16[index], clamped16 = sanitize_kl(
                result["kl_mean_first16"],
                uid,
                condition,
                "kl_mean_first16",
            )
            if clamped16:
                negative_clamped.append(
                    {
                        "row_uid": uid,
                        "condition": condition,
                        "field": "kl_mean_first16",
                        "raw_value": float(result["kl_mean_first16"]),
                    }
                )
        kl_by_condition[condition] = values
        kl16_by_condition[condition] = values16
    identity_max = float(np.max(np.abs(kl_by_condition["identity"])))
    identity16_max = float(
        np.max(np.abs(kl16_by_condition["identity"]))
    )
    if max(identity_max, identity16_max) > IDENTITY_KL_TOL:
        raise ValueError(
            "identity KL exceeds 1e-5 "
            f"(position={identity_max}, first16={identity16_max})"
        )

    corpus_values = []
    for recon_row, causal_row in zip(recon_rows, aligned_causal):
        corpus = causal_row.get("corpus", recon_row.get("corpus"))
        corpus_values.append(normalize_corpus(corpus))
    strata = np.asarray(corpus_values, dtype=object)
    corpus_counts = Counter(strata.tolist())
    if corpus_counts != EXPECTED_CORPUS_COUNTS:
        raise ValueError(
            f"held-out corpus counts are {dict(corpus_counts)}, "
            f"expected {dict(EXPECTED_CORPUS_COUNTS)}"
        )

    zero = kl_by_condition["zero"]
    sae_big = kl_by_condition["sae_big"]
    orig = kl_by_condition["orig"]
    zero_sum = float(zero.sum(dtype=np.float64))
    if zero_sum <= 1e-6:
        raise ValueError(f"held-out sum(KL_zero)={zero_sum} is invalid")
    hybrid = np.where(route_nla, orig, sae_big)
    gain = sae_big - hybrid

    channel_conditions = {"p3_only", "p12", "quote_strip_p3"}
    discovery_channel_active = bool(
        gate.get("discovery_channel_active", False)
    )
    causal_channel_active = channel_conditions <= available
    causal_channel_partial = bool(channel_conditions & available) and not (
        causal_channel_active
    )
    if causal_channel_partial:
        raise ValueError("held-out causal JSON has a partial paragraph channel")
    if causal_channel_active != npz_channel_active:
        raise ValueError("NPZ and causal paragraph-channel states disagree")
    h5b_available = (
        discovery_channel_active and causal_channel_active and npz_channel_active
    )

    column_names = ["gain", "zero", "route"]
    columns = [gain, zero, route_nla.astype(np.float64)]
    if h5b_available:
        for name in ("orig", "p3_only", "p12", "quote_strip_p3"):
            column_names.append(name)
            columns.append(kl_by_condition[name])
    bootstrap_sums = stratified_bootstrap_sums(
        np.column_stack(columns), strata
    )
    boot = {
        name: bootstrap_sums[:, index]
        for index, name in enumerate(column_names)
    }
    if np.any(boot["zero"] <= 1e-12):
        raise ValueError("a bootstrap resample has nonpositive sum(KL_zero)")

    boot_g = boot["gain"] / boot["zero"]
    boot_delta = boot["gain"] / EXPECTED_ROWS
    boot_coverage = boot["route"] / EXPECTED_ROWS
    g_point = float(gain.sum(dtype=np.float64) / zero_sum)
    delta_point = float(gain.mean())
    coverage = float(route_nla.mean())
    catastrophes = (hybrid - sae_big) > CAT_REGRET_NAT
    catastrophe_count = int(catastrophes.sum())
    catastrophe_upper = clopper_pearson_upper(
        catastrophe_count, EXPECTED_ROWS, 0.95
    )
    g_ci90 = interval(boot_g, 0.90)
    g_lower95 = lower_bound(boot_g, 0.95)
    coverage_lower95 = lower_bound(boot_coverage, 0.95)
    feasible_pass = feasible
    coverage_pass = coverage_lower95 > 0.10
    superiority_pass = g_lower95 > 0.0
    tail_pass = catastrophe_upper < 0.03
    parity_pass = g_ci90[0] >= -0.01 and g_ci90[1] <= 0.01
    decision = (
        "SELECTIVE IMPROVEMENT"
        if feasible_pass and coverage_pass and superiority_pass and tail_pass
        else "SAFE SELECTIVE PARITY"
        if (
            feasible_pass
            and coverage_pass
            and not superiority_pass
            and tail_pass
            and parity_pass
        )
        else "NO SELECTIVE CLAIM"
    )

    h5a = {
        "decision": decision,
        "gate_feasible": feasible,
        "threshold": threshold,
        "tie_hash_cutoff_inclusive": tie_cutoff,
        "n_route_nla": int(route_nla.sum()),
        "g": g_point,
        "g_one_sided_95_lower": g_lower95,
        "g_ci90_stratified_bootstrap": g_ci90,
        "g_ci95_stratified_bootstrap": interval(boot_g, 0.95),
        "delta_raw": delta_point,
        "delta_raw_ci95_stratified_bootstrap": interval(
            boot_delta, 0.95
        ),
        "coverage": coverage,
        "coverage_one_sided_95_lower": coverage_lower95,
        "coverage_ci95_stratified_bootstrap": interval(
            boot_coverage, 0.95
        ),
        "catastrophic_regret_count": catastrophe_count,
        "catastrophic_regret_rate": catastrophe_count / EXPECTED_ROWS,
        "catastrophic_regret_one_sided_95_cp_upper": catastrophe_upper,
        "gates": {
            "feasible_discovery_gate": feasible_pass,
            "coverage_lower_gt_0.10": coverage_pass,
            "g_lower_gt_0": superiority_pass,
            "catastrophe_cp_upper_lt_0.03": tail_pass,
            "g_ci90_inside_parity_margin": parity_pass,
        },
    }

    if not discovery_channel_active:
        h5b: dict[str, Any] = {
            "status": "ABORTED_ON_DISCOVERY",
            "replicated": False,
        }
    elif not h5b_available:
        h5b = {
            "status": "ABORTED_MISSING_HELDOUT_CHANNEL",
            "replicated": False,
        }
    else:
        gp_numerator = (
            kl_by_condition["p12"] - kl_by_condition["p3_only"]
        )
        boot_gp = (boot["p12"] - boot["p3_only"]) / boot["zero"]
        gp_ci95 = interval(boot_gp, 0.95)
        recoveries: dict[str, dict[str, Any]] = {}
        boot_recoveries: dict[str, np.ndarray] = {}
        for name in ("orig", "p3_only", "p12", "quote_strip_p3"):
            point = float(1.0 - kl_by_condition[name].sum() / zero_sum)
            boot_value = 1.0 - boot[name] / boot["zero"]
            boot_recoveries[name] = boot_value
            recoveries[name] = {
                "point": point,
                "ci95_stratified_bootstrap": interval(boot_value, 0.95),
            }
        r_orig = recoveries["orig"]["point"]
        r_p3 = recoveries["p3_only"]["point"]
        retention = (
            float(r_p3 / r_orig) if abs(float(r_orig)) > 1e-12 else None
        )
        retention_testable = bool(
            retention is not None
            and np.isfinite(retention)
            and r_orig > 0.0
            and np.all(boot_recoveries["orig"] > 0.0)
        )
        if retention_testable:
            boot_t = (
                boot_recoveries["p3_only"] / boot_recoveries["orig"]
            )
            retention_lower = lower_bound(boot_t, 0.95)
            retention_ci95 = interval(boot_t, 0.95)
        else:
            retention_lower = None
            retention_ci95 = None
        gp_pass = gp_ci95[0] > 0.0
        retention_pass = bool(
            retention_testable
            and retention_lower is not None
            and retention_lower > 0.90
        )
        replicated = gp_pass and retention_pass
        h5b = {
            "status": (
                "CHANNEL REPLICATED"
                if replicated
                else "CHANNEL NOT REPLICATED"
                if retention_testable
                else "NOT TESTABLE"
            ),
            "replicated": replicated,
            "g_p3_p12": float(gp_numerator.sum() / zero_sum),
            "g_p3_p12_ci95_stratified_bootstrap": gp_ci95,
            "raw_p12_minus_p3_mean": float(gp_numerator.mean()),
            "raw_p12_minus_p3_ci95_stratified_bootstrap": interval(
                (boot["p12"] - boot["p3_only"]) / EXPECTED_ROWS,
                0.95,
            ),
            "r": recoveries,
            "retention_t": retention,
            "retention_testable": retention_testable,
            "retention_one_sided_95_lower": retention_lower,
            "retention_ci95_stratified_bootstrap": retention_ci95,
            "bootstrap_r_orig_min": float(
                boot_recoveries["orig"].min()
            ),
            "sign_count_p12_minus_p3": {
                "positive": int((gp_numerator > 0).sum()),
                "negative": int((gp_numerator < 0).sum()),
                "zero": int((gp_numerator == 0).sum()),
            },
            "gates": {
                "g_p3_p12_ci95_lower_gt_0": gp_pass,
                "retention_one_sided_95_lower_gt_0.90": retention_pass,
            },
        }

    return {
        "schema_version": 1,
        "experiment": "N5 independent held-out raw-artifact audit",
        "status": "complete",
        "formula_audit": {
            "h5a_g": "sum(KL_big-KL_hybrid)/sum(KL_zero)",
            "h5a_delta_raw": "mean(KL_big-KL_hybrid)",
            "hybrid": "orig when frozen gate routes NLA, else SAE-big",
            "catastrophe": "strictly KL_hybrid-KL_big > 1.0 nat",
            "catastrophe_interval": (
                "one-sided 95% exact Clopper-Pearson"
            ),
            "h5b_g": "sum(KL_p12-KL_p3)/sum(KL_zero)",
            "h5b_recovery": "R_s=1-sum(KL_s)/sum(KL_zero)",
            "h5b_retention": "T=R_p3/R_orig",
            "bootstrap": (
                "50,000 shared content-group resamples stratified by "
                "Pile/XNLI; seed 20260730; gate never retrained"
            ),
        },
        "inputs": {
            "gate": str(gate_path),
            "gate_sha256": gate_sha,
            "recon_json": str(recon_json_path),
            "recon_json_sha256": recon_json_sha,
            "recon_npz": str(recon_npz_path),
            "recon_npz_sha256": recon_npz_sha,
            "causal": str(causal_path),
            "causal_sha256": causal_sha,
        },
        "cohort": {
            "n_rows": EXPECTED_ROWS,
            "corpus_counts": dict(corpus_counts),
            "row_uids_unique": True,
            "content_group_itt": True,
        },
        "router_qa": {
            "score_recomputed_from": "NPZ pred_orig, x, and gate m_hat",
            "reported_q_max_abs_error": float(max(q_errors)),
            "reported_routes_exact": True,
            "npz_routes_exact": npz_route is not None,
        },
        "numerical_qa": {
            "sum_kl_zero": zero_sum,
            "negative_kl_values_clamped_to_zero": negative_clamped,
            "identity_kl_at_pos_abs_max": identity_max,
            "identity_kl16_abs_max": identity16_max,
        },
        "bootstrap": {
            "n_resamples": N_BOOT,
            "seed": SEED,
            "strata": ["pile", "xnli"],
            "draws_shared_across_endpoints": True,
            "quantile_method": "linear",
        },
        "h5a": h5a,
        "h5b": h5b,
    }


def run_synthetic_smoke() -> dict[str, Any]:
    """Exercise the complete file parser and all H5-A/H5-B formulas."""
    n = EXPECTED_ROWS
    uids = [f"synthetic-heldout-{index:04d}" for index in range(n)]
    route_expected = np.asarray([index % 5 == 0 for index in range(n)])
    q_expected = np.where(route_expected, 0.8, 0.2)
    x = np.tile(np.asarray([1.0, 1.0, 0.0]), (n, 1))
    pred = np.column_stack(
        [
            np.ones(n),
            q_expected,
            np.sqrt(1.0 - q_expected**2),
        ]
    )
    m_hat = np.asarray([1.0, 0.0, 0.0])
    gate_payload = {
        "status": "complete",
        "split": "discovery",
        "gate": {
            "status": "FEASIBLE",
            "feasible": True,
            "score_name": "absolute_nla_centered_cosine",
            "discovery_mean_direction": m_hat.tolist(),
            "threshold": 0.5,
            "tie_hash_cutoff_inclusive": "f" * 64,
            "discovery_channel_active": True,
        },
    }
    with tempfile.TemporaryDirectory(prefix="n5-independent-smoke-") as tmp:
        directory = Path(tmp)
        gate_path = directory / "gate.json"
        npz_path = directory / "heldout.npz"
        recon_path = directory / "heldout_recon.json"
        causal_path = directory / "heldout_causal.json"
        gate_path.write_text(
            json.dumps(gate_payload, indent=2), encoding="utf-8"
        )
        gate_sha = sha256_file(gate_path)
        np.savez_compressed(
            npz_path,
            x=x.astype(np.float32),
            pred_orig=pred.astype(np.float32),
            pred_p3_only=pred.astype(np.float32),
            pred_p12=pred.astype(np.float32),
            pred_quote_strip_p3=pred.astype(np.float32),
            m_hat=m_hat.astype(np.float64),
            row_uids=np.asarray(uids),
            q_router=q_expected.astype(np.float64),
            route_nla=route_expected,
            gate_sha256=np.asarray(gate_sha),
        )
        npz_sha = sha256_file(npz_path)
        recon_rows = []
        causal_rows = []
        for index, uid in enumerate(uids):
            corpus = "pile" if index < 300 else "xnli"
            common = {
                "row_uid": uid,
                "content_group_id": f"group-{index:04d}",
                "split": "heldout",
                "corpus": corpus,
            }
            recon_rows.append(
                {
                    **common,
                    "q": float(q_expected[index]),
                    "route_nla": bool(route_expected[index]),
                    "routed_to": (
                        "nla" if route_expected[index] else "sae_big"
                    ),
                }
            )
            orig = 3.0 if route_expected[index] else 5.0
            results = {
                "identity": {
                    "kl_at_pos": 0.0,
                    "kl_mean_first16": 0.0,
                },
                "orig": {
                    "kl_at_pos": orig,
                    "kl_mean_first16": orig,
                },
                "sae_big": {
                    "kl_at_pos": 4.0,
                    "kl_mean_first16": 4.0,
                },
                "zero": {
                    "kl_at_pos": 10.0,
                    "kl_mean_first16": 10.0,
                },
                "p3_only": {
                    "kl_at_pos": 2.0,
                    "kl_mean_first16": 2.0,
                },
                "p12": {
                    "kl_at_pos": 3.0,
                    "kl_mean_first16": 3.0,
                },
                "quote_strip_p3": {
                    "kl_at_pos": 2.5,
                    "kl_mean_first16": 2.5,
                },
            }
            causal_rows.append({**common, "results": results})
        recon_path.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "split": "heldout",
                    "inputs": {"gate_sha256": gate_sha},
                    "outputs": {"vecs_sha256": npz_sha},
                    "rows": recon_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        causal_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "split": "heldout",
                    "inputs": {
                        "gate_sha256": gate_sha,
                        "recon_sha256": npz_sha,
                    },
                    "rows": causal_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result = audit_artifacts(
            gate_path, recon_path, npz_path, causal_path
        )

    expected_cp_zero = 1.0 - 0.05 ** (1.0 / EXPECTED_ROWS)
    checks = {
        "route_count_80": result["h5a"]["n_route_nla"] == 80,
        "g_exact_0.02": abs(result["h5a"]["g"] - 0.02) < 1e-12,
        "delta_exact_0.2": (
            abs(result["h5a"]["delta_raw"] - 0.2) < 1e-12
        ),
        "coverage_exact_0.2": (
            abs(result["h5a"]["coverage"] - 0.2) < 1e-12
        ),
        "cp_zero_matches_closed_form": (
            abs(
                result["h5a"][
                    "catastrophic_regret_one_sided_95_cp_upper"
                ]
                - expected_cp_zero
            )
            < 1e-12
        ),
        "g_p3_p12_exact_0.1": (
            abs(result["h5b"]["g_p3_p12"] - 0.1) < 1e-12
        ),
        "r_orig_exact_0.54": (
            abs(result["h5b"]["r"]["orig"]["point"] - 0.54) < 1e-12
        ),
        "r_p3_exact_0.8": (
            abs(result["h5b"]["r"]["p3_only"]["point"] - 0.8) < 1e-12
        ),
        "t_exact_40_over_27": (
            abs(result["h5b"]["retention_t"] - (40.0 / 27.0)) < 1e-12
        ),
        "all_intervals_contain_points": all(
            bounds[0] - 1e-12 <= point <= bounds[1] + 1e-12
            for point, bounds in (
                (
                    result["h5a"]["g"],
                    result["h5a"]["g_ci95_stratified_bootstrap"],
                ),
                (
                    result["h5a"]["delta_raw"],
                    result["h5a"][
                        "delta_raw_ci95_stratified_bootstrap"
                    ],
                ),
                (
                    result["h5b"]["g_p3_p12"],
                    result["h5b"][
                        "g_p3_p12_ci95_stratified_bootstrap"
                    ],
                ),
                (
                    result["h5b"]["retention_t"],
                    result["h5b"][
                        "retention_ci95_stratified_bootstrap"
                    ],
                ),
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"synthetic smoke failed: {checks}")
    return {
        "status": "PASS",
        "n_boot": N_BOOT,
        "seed": SEED,
        "checks": checks,
        "selected_results": {
            "h5a_g": result["h5a"]["g"],
            "h5a_delta_raw": result["h5a"]["delta_raw"],
            "h5a_coverage": result["h5a"]["coverage"],
            "h5a_catastrophe_cp_upper": result["h5a"][
                "catastrophic_regret_one_sided_95_cp_upper"
            ],
            "h5b_g_p3_p12": result["h5b"]["g_p3_p12"],
            "h5b_r_orig": result["h5b"]["r"]["orig"]["point"],
            "h5b_r_p3": result["h5b"]["r"]["p3_only"]["point"],
            "h5b_t": result["h5b"]["retention_t"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--recon-json", type=Path)
    parser.add_argument("--recon-npz", type=Path)
    parser.add_argument("--causal", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="optional JSON output; stdout is always written",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a 400-row, 50k-bootstrap end-to-end synthetic smoke",
    )
    args = parser.parse_args()

    if args.self_test:
        if any(
            path is not None
            for path in (
                args.gate,
                args.recon_json,
                args.recon_npz,
                args.causal,
                args.out,
            )
        ):
            parser.error("--self-test cannot be combined with artifact paths")
        output = run_synthetic_smoke()
    else:
        required = {
            "--gate": args.gate,
            "--recon-json": args.recon_json,
            "--recon-npz": args.recon_npz,
            "--causal": args.causal,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
        assert args.gate is not None
        assert args.recon_json is not None
        assert args.recon_npz is not None
        assert args.causal is not None
        output = audit_artifacts(
            args.gate,
            args.recon_json,
            args.recon_npz,
            args.causal,
        )

    rendered = json.dumps(
        output, ensure_ascii=False, indent=2, allow_nan=False
    )
    print(rendered, flush=True)
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
