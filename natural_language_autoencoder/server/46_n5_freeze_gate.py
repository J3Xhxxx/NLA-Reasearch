#!/usr/bin/env python3
"""Freeze the preregistered N5 discovery-only NLA/SAE-big gate.

This script is deliberately discovery-only.  It validates the reconstruction
JSON/NPZ pair, recomputes the sole allowed centered-cosine score from the
stored discovery mean direction, aligns causal rows by frozen row UID, applies
the finite routing-fraction grid, and writes an immutable gate artifact for
the held-out stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ROWS = 200
EXPECTED_PREREG_SHA256 = (
    "63dc31b4f9607e54ac15f1c364fcae2ee903f228fe0afb4d388c6dad1a6f9103"
)
EXPECTED_MODEL_MANIFEST_SHA256 = (
    "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735"
)
FRACTIONS = tuple(round(x, 2) for x in np.arange(0.20, 0.801, 0.05))
CAT_REGRET_NAT = 1.0
MAX_CAT_RATE = 0.03
OBJECTIVE_TIE_TOL = 1e-6


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def uid_hash(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("mean direction is non-finite or zero norm")
    return vector / norm


def centered_cosine(
    prediction: np.ndarray, target: np.ndarray, mean_direction: np.ndarray
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean_direction = np.asarray(mean_direction, dtype=np.float64)
    pred_centered = prediction - np.outer(
        prediction @ mean_direction, mean_direction
    )
    target_centered = target - np.outer(
        target @ mean_direction, mean_direction
    )
    denominator = np.linalg.norm(pred_centered, axis=1) * np.linalg.norm(
        target_centered, axis=1
    )
    if np.any(denominator <= 1e-12) or not np.isfinite(denominator).all():
        bad = np.flatnonzero(
            (denominator <= 1e-12) | ~np.isfinite(denominator)
        )[:10].tolist()
        raise ValueError(f"centered-cosine denominator failed at rows {bad}")
    scores = np.sum(pred_centered * target_centered, axis=1) / denominator
    if not np.isfinite(scores).all():
        raise ValueError("centered-cosine scores contain non-finite values")
    return scores


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_sha256_sidecar(path: Path, observed_sha256: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"frozen SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise ValueError(f"malformed frozen SHA-256 sidecar: {sidecar}")
    declared_sha256, declared_name = fields
    if declared_sha256.lower() != observed_sha256.lower():
        raise ValueError(
            f"{path} SHA-256 differs from its frozen sidecar: "
            f"{observed_sha256} != {declared_sha256}"
        )
    if declared_name != path.name:
        raise ValueError(
            f"{sidecar} names {declared_name!r}, expected {path.name!r}"
        )


def require_npz_scalar_string(
    archive: np.lib.npyio.NpzFile, key: str, expected: str
) -> None:
    if key not in archive.files:
        raise ValueError(f"discovery NPZ lacks embedded provenance key {key}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(
            f"discovery NPZ provenance key {key} must be scalar, "
            f"found shape {value.shape}"
        )
    observed = str(value.item())
    if observed != expected:
        raise ValueError(
            f"discovery NPZ embedded {key}={observed!r}, "
            f"expected {expected!r}"
        )


def frozen_plan_rows(plan: dict, split: str, expected: int) -> list[dict]:
    if plan.get("status") != "frozen_before_base_model_load":
        raise ValueError("cohort plan is not frozen before model load")
    if plan.get("preregistration_sha256") != EXPECTED_PREREG_SHA256:
        raise ValueError("cohort plan/preregistration hash mismatch")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 600:
        raise ValueError("cohort plan must contain exactly 600 rows")
    selected = [row for row in rows if row.get("split") == split]
    if len(selected) != expected:
        raise ValueError(
            f"cohort plan {split} requires {expected} rows, found {len(selected)}"
        )
    return selected


def parse_weight_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(
                f"malformed weight manifest at {path}:{line_number}"
            )
        entries[fields[1].strip()] = fields[0].lower()
    if len(entries) != 25:
        raise ValueError("full model manifest must contain 25 unique files")
    return entries


def validate_actual_model_bindings(
    recon: dict,
    causal: dict,
    manifest_entries: dict[str, str],
    manifest_sha256: str,
) -> None:
    require_input_hash(
        recon,
        "model_manifest_sha256",
        manifest_sha256,
        "reconstruction",
    )
    require_input_hash(
        causal,
        "model_manifest_sha256",
        manifest_sha256,
        "causal",
    )
    model_manifests = recon.get("inputs", {}).get("model_manifests")
    if not isinstance(model_manifests, dict):
        raise ValueError("reconstruction omits actual model-file manifests")
    reconstructed_files: dict[str, str] = {}
    for label, item in model_manifests.items():
        if not isinstance(item, dict) or not isinstance(item.get("files"), dict):
            raise ValueError(f"malformed reconstruction model manifest {label}")
        root = str(item.get("root", "")).rstrip("/")
        for relative, digest in item["files"].items():
            reconstructed_files[f"{root}/{relative}"] = str(digest).lower()
    expected_reconstruction = {
        path: digest
        for path, digest in manifest_entries.items()
        if "/gemma-3-12b-it/" not in path
    }
    if any(
        reconstructed_files.get(path) != digest
        for path, digest in expected_reconstruction.items()
    ):
        raise ValueError(
            "reconstruction actual AV/AR/SAE files differ from the frozen "
            "25-file manifest"
        )
    expected_base = {
        path: digest
        for path, digest in manifest_entries.items()
        if "/gemma-3-12b-it/" in path
    }
    causal_base = causal.get("inputs", {}).get(
        "verified_base_files_from_full_manifest"
    )
    if causal_base != expected_base:
        raise ValueError(
            "causal actual base files differ from the frozen 25-file manifest"
        )


def require_rows(payload: dict, label: str, expected: int) -> list[dict]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != expected:
        found = len(rows) if isinstance(rows, list) else None
        raise ValueError(f"{label} requires {expected} rows, found {found}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be JSON objects")
    return rows


def row_uid(row: dict) -> str:
    value = row.get("row_uid")
    if value is None or not str(value):
        raise ValueError("every N5 row must carry a nonempty frozen row_uid")
    return str(value)


def validate_split(rows: list[dict], label: str, split: str) -> None:
    declared = {str(row.get("split")) for row in rows if "split" in row}
    if declared and declared != {split}:
        raise ValueError(f"{label} has unexpected splits: {sorted(declared)}")


def validate_unique_groups(rows: list[dict], label: str) -> None:
    uids = [row_uid(row) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError(f"{label} row_uid values are not unique")
    groups = [
        str(row.get("content_group_id"))
        for row in rows
        if row.get("content_group_id") is not None
    ]
    if groups and len(groups) != len(rows):
        raise ValueError(f"{label} has partial content_group_id provenance")
    if groups and len(set(groups)) != len(groups):
        raise ValueError(
            f"{label} violates one-row-per-independent-content-group"
        )


def require_input_hash(
    payload: dict, field: str, expected: str, label: str
) -> None:
    declared = payload.get("inputs", {}).get(field)
    if declared is None:
        raise ValueError(f"{label} does not declare required input hash {field}")
    if str(declared) != expected:
        raise ValueError(
            f"{label} declares {field}={declared}, expected {expected}"
        )


def validate_rows_against_plan(
    artifact_rows: list[dict], plan_rows: list[dict], label: str
) -> None:
    by_uid = {row_uid(row): row for row in artifact_rows}
    expected_uids = [str(row["row_uid"]) for row in plan_rows]
    if set(by_uid) != set(expected_uids):
        raise ValueError(f"{label} row UIDs differ from the frozen cohort plan")
    for planned in plan_rows:
        uid = str(planned["row_uid"])
        observed = by_uid[uid]
        for field in (
            "content_group_id",
            "split",
            "doc_id",
            "position",
            "corpus",
            "source",
            "lang",
        ):
            if str(observed.get(field)) != str(planned.get(field)):
                raise ValueError(
                    f"{label} {uid} differs from plan at {field}: "
                    f"{observed.get(field)!r} != {planned.get(field)!r}"
                )


def sanitize_kl(value: Any, uid: str, condition: str) -> tuple[float, bool]:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{uid} {condition} KL is non-finite")
    if number < -1e-7:
        raise ValueError(f"{uid} {condition} KL={number} is below -1e-7")
    if number < 0:
        return 0.0, True
    return number, False


def make_candidate(
    fraction: float,
    scores: np.ndarray,
    uid_hashes: list[str],
    kl_orig: np.ndarray,
    kl_big: np.ndarray,
    kl_zero: np.ndarray,
) -> tuple[dict, np.ndarray]:
    n_rows = len(scores)
    requested = fraction * n_rows
    target_count = int(round(requested))
    if abs(target_count - requested) > 1e-9:
        raise ValueError(
            f"fraction {fraction} does not produce an integer discovery count"
        )
    order = sorted(
        range(n_rows), key=lambda i: (-float(scores[i]), uid_hashes[i])
    )
    route = np.zeros(n_rows, dtype=bool)
    route[order[:target_count]] = True
    threshold = float(scores[order[target_count - 1]])
    boundary = np.flatnonzero(scores == threshold)
    boundary_selected = sorted(uid_hashes[i] for i in boundary if route[i])
    boundary_all = sorted(uid_hashes[i] for i in boundary)
    if not boundary_selected:
        raise RuntimeError("top-q cutoff selected no boundary row")
    tie_cutoff = boundary_selected[-1]
    reproduced = (scores > threshold) | (
        (scores == threshold)
        & np.asarray([h <= tie_cutoff for h in uid_hashes], dtype=bool)
    )
    if not np.array_equal(route, reproduced):
        raise RuntimeError("frozen threshold/tie rule does not reproduce top-q")

    hybrid = np.where(route, kl_orig, kl_big)
    regret = hybrid - kl_big
    raw_gain = float(np.sum(kl_big - hybrid, dtype=np.float64))
    zero_sum = float(np.sum(kl_zero, dtype=np.float64))
    if zero_sum <= 1e-12:
        raise ValueError("discovery sum(KL_zero) is not positive")
    normalized_gain = raw_gain / zero_sum
    catastrophe = regret > CAT_REGRET_NAT
    catastrophe_count = int(catastrophe.sum())
    catastrophe_rate = catastrophe_count / n_rows
    feasible = bool(raw_gain > 0 and catastrophe_rate <= MAX_CAT_RATE)
    candidate = {
        "routing_fraction": fraction,
        "nla_count": target_count,
        "threshold": threshold,
        "boundary_n_rows": int(len(boundary)),
        "boundary_selected_n_rows": int(len(boundary_selected)),
        "boundary_selected_uid_sha256": boundary_selected,
        "boundary_all_uid_sha256": boundary_all,
        "tie_hash_cutoff_inclusive": tie_cutoff,
        "raw_gain_sum_kl_big_minus_hybrid": raw_gain,
        "sum_kl_zero": zero_sum,
        "normalized_gain": normalized_gain,
        "catastrophic_regret_count": catastrophe_count,
        "catastrophic_regret_rate": catastrophe_rate,
        "feasible": feasible,
        "feasibility": {
            "raw_gain_strictly_positive": bool(raw_gain > 0),
            "catastrophic_regret_rate_le_0.03": bool(
                catastrophe_rate <= MAX_CAT_RATE
            ),
        },
    }
    return candidate, route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon-json", required=True, type=Path)
    parser.add_argument("--recon-npz", required=True, type=Path)
    parser.add_argument("--causal", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.out.exists():
        raise ValueError(f"refusing to overwrite frozen gate {args.out}")

    recon_sha = sha256_file(args.recon_json)
    npz_sha = sha256_file(args.recon_npz)
    causal_sha = sha256_file(args.causal)
    plan_sha = sha256_file(args.plan)
    manifest_sha = sha256_file(args.manifest)
    prereg_sha = sha256_file(args.prereg)
    script_sha = sha256_file(__file__)
    if prereg_sha != EXPECTED_PREREG_SHA256:
        raise ValueError(
            f"unexpected preregistration SHA-256 {prereg_sha}"
        )
    if manifest_sha != EXPECTED_MODEL_MANIFEST_SHA256:
        raise ValueError(
            f"unexpected full model manifest SHA-256 {manifest_sha}"
        )
    verify_sha256_sidecar(args.recon_json, recon_sha)
    verify_sha256_sidecar(args.recon_npz, npz_sha)
    verify_sha256_sidecar(args.causal, causal_sha)
    verify_sha256_sidecar(args.plan, plan_sha)
    verify_sha256_sidecar(args.manifest, manifest_sha)
    manifest_entries = parse_weight_manifest(args.manifest)

    recon = read_json(args.recon_json)
    causal = read_json(args.causal)
    plan = read_json(args.plan)
    plan_rows = frozen_plan_rows(plan, "discovery", EXPECTED_ROWS)
    recon_rows = require_rows(recon, "discovery reconstruction", EXPECTED_ROWS)
    causal_rows = require_rows(causal, "discovery causal", EXPECTED_ROWS)
    validate_split(recon_rows, "discovery reconstruction", "discovery")
    validate_split(causal_rows, "discovery causal", "discovery")
    validate_unique_groups(recon_rows, "discovery reconstruction")
    validate_unique_groups(causal_rows, "discovery causal")
    validate_rows_against_plan(
        recon_rows, plan_rows, "discovery reconstruction"
    )
    validate_rows_against_plan(causal_rows, plan_rows, "discovery causal")

    require_input_hash(recon, "prereg_sha256", prereg_sha, "reconstruction")
    require_input_hash(causal, "prereg_sha256", prereg_sha, "causal")
    require_input_hash(causal, "recon_sha256", npz_sha, "causal")
    require_input_hash(causal, "gate_sha256", "", "causal")
    require_input_hash(recon, "plan_sha256", plan_sha, "reconstruction")
    if recon.get("outputs", {}).get("vecs_sha256") != npz_sha:
        raise ValueError(
            "discovery reconstruction JSON does not bind the supplied NPZ"
        )
    validate_actual_model_bindings(
        recon, causal, manifest_entries, manifest_sha
    )
    recon_activation_sha = recon.get("inputs", {}).get("activations_sha256")
    causal_activation_sha = causal.get("inputs", {}).get("activations_sha256")
    if not recon_activation_sha or not causal_activation_sha:
        raise ValueError(
            "discovery reconstruction and causal artifacts must both bind "
            "the activation parquet"
        )
    if str(recon_activation_sha) != str(causal_activation_sha):
        raise ValueError(
            "discovery reconstruction/causal activation hashes differ"
        )
    if str(causal.get("split", "discovery")) != "discovery":
        raise ValueError("causal artifact is not the discovery split")
    if causal.get("status") not in (None, "complete"):
        raise ValueError(f"causal artifact status is {causal.get('status')!r}")
    causal_qa = causal.get("qa", {})
    if not bool(causal_qa.get("provenance_all_bit_exact", False)):
        raise ValueError("discovery causal clean activations are not bit-exact")
    identity_max = float(causal_qa.get("identity_kl_at_pos_abs_max", np.inf))
    identity16_max = float(causal_qa.get("identity_kl16_abs_max", np.inf))
    if max(identity_max, identity16_max) > 1e-5:
        raise ValueError(
            "discovery causal identity KL exceeds 1e-5: "
            f"pos={identity_max}, kl16={identity16_max}"
        )

    with np.load(args.recon_npz, allow_pickle=False) as archive:
        required = {
            "x",
            "pred_orig",
            "m_hat",
            "row_uids",
            "activations_sha256",
            "plan_sha256",
            "model_manifest_sha256",
            "prereg_sha256",
            "gate_sha256",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"discovery NPZ lacks required keys: {missing}")
        require_npz_scalar_string(
            archive, "activations_sha256", str(recon_activation_sha)
        )
        require_npz_scalar_string(archive, "plan_sha256", plan_sha)
        require_npz_scalar_string(
            archive, "model_manifest_sha256", manifest_sha
        )
        require_npz_scalar_string(archive, "prereg_sha256", prereg_sha)
        require_npz_scalar_string(archive, "gate_sha256", "")
        x = np.asarray(archive["x"], dtype=np.float32)
        pred_orig = np.asarray(archive["pred_orig"], dtype=np.float32)
        stored_m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
        npz_uids = [
            str(value) for value in np.asarray(archive["row_uids"]).tolist()
        ]
        channel_keys = {
            "pred_p3_only",
            "pred_p12",
            "pred_quote_strip_p3",
        }
        channel_keys_present = sorted(channel_keys & set(archive.files))
        if channel_keys_present and len(channel_keys_present) != len(channel_keys):
            raise ValueError(
                "discovery NPZ contains only a partial paragraph channel: "
                f"{channel_keys_present}"
            )
    if x.ndim != 2 or x.shape[0] != EXPECTED_ROWS:
        raise ValueError(f"unexpected discovery x shape {x.shape}")
    if pred_orig.shape != x.shape:
        raise ValueError(
            f"pred_orig shape {pred_orig.shape} does not match x {x.shape}"
        )
    if stored_m_hat.shape != (x.shape[1],):
        raise ValueError(f"unexpected m_hat shape {stored_m_hat.shape}")
    if not np.isfinite(x).all() or not np.isfinite(pred_orig).all():
        raise ValueError("reconstruction vectors contain non-finite values")

    recon_uids = [row_uid(row) for row in recon_rows]
    if npz_uids != recon_uids:
        raise ValueError("NPZ row_uids do not match reconstruction JSON order")
    computed_m_hat = unit(x.mean(axis=0, dtype=np.float64))
    if abs(float(np.linalg.norm(stored_m_hat)) - 1.0) > 1e-6:
        raise ValueError("stored discovery m_hat is not unit norm")
    m_hat_error = float(np.max(np.abs(stored_m_hat - computed_m_hat)))
    if m_hat_error > 1e-6:
        raise ValueError(
            f"stored m_hat does not match discovery x mean: {m_hat_error}"
        )
    scores = centered_cosine(pred_orig, x, stored_m_hat)
    reported_scores = []
    for row in recon_rows:
        try:
            reported_scores.append(float(row["scores"]["orig"]["cos_c"]))
        except (KeyError, TypeError, ValueError):
            reported_scores = []
            break
    score_max_error = None
    if reported_scores:
        reported = np.asarray(reported_scores, dtype=float)
        score_max_error = float(np.max(np.abs(reported - scores)))
        if not np.allclose(reported, scores, rtol=1e-7, atol=5e-7):
            raise ValueError(
                "reconstruction JSON centered cosine does not match NPZ/frozen "
                f"m_hat; max_abs={score_max_error}"
            )

    causal_by_uid = {row_uid(row): row for row in causal_rows}
    if set(causal_by_uid) != set(recon_uids):
        raise ValueError("discovery reconstruction/causal row UID sets differ")
    kl_orig = np.empty(EXPECTED_ROWS, dtype=float)
    kl_big = np.empty(EXPECTED_ROWS, dtype=float)
    kl_zero = np.empty(EXPECTED_ROWS, dtype=float)
    negative_clamped: list[dict] = []
    for i, uid in enumerate(recon_uids):
        row = causal_by_uid[uid]
        results = row.get("results", {})
        for condition in ("orig", "sae_big", "zero"):
            if condition not in results or "kl_at_pos" not in results[condition]:
                raise ValueError(f"{uid} lacks causal {condition}.kl_at_pos")
        for condition, destination in (
            ("orig", kl_orig),
            ("sae_big", kl_big),
            ("zero", kl_zero),
        ):
            value, clamped = sanitize_kl(
                results[condition]["kl_at_pos"], uid, condition
            )
            destination[i] = value
            if clamped:
                negative_clamped.append(
                    {
                        "row_uid": uid,
                        "condition": condition,
                        "raw_value": float(results[condition]["kl_at_pos"]),
                    }
                )
    zero_sum = float(kl_zero.sum(dtype=np.float64))
    if zero_sum <= 1e-6:
        raise ValueError(f"discovery sum(KL_zero)={zero_sum} is invalid")

    uid_hashes = [uid_hash(uid) for uid in recon_uids]
    candidates: list[dict] = []
    candidate_routes: list[np.ndarray] = []
    for fraction in FRACTIONS:
        candidate, route = make_candidate(
            fraction,
            scores,
            uid_hashes,
            kl_orig,
            kl_big,
            kl_zero,
        )
        candidates.append(candidate)
        candidate_routes.append(route)

    best_index: int | None = None
    for index, candidate in enumerate(candidates):
        if not candidate["feasible"]:
            continue
        if best_index is None:
            best_index = index
            continue
        best = candidates[best_index]
        difference = candidate["normalized_gain"] - best["normalized_gain"]
        if difference > OBJECTIVE_TIE_TOL or (
            abs(difference) <= OBJECTIVE_TIE_TOL
            and candidate["routing_fraction"] < best["routing_fraction"]
        ):
            best_index = index

    feasible = best_index is not None
    selected = candidates[best_index] if feasible else None
    route = (
        candidate_routes[best_index]
        if feasible
        else np.zeros(EXPECTED_ROWS, dtype=bool)
    )
    declared_channel = recon.get("channel")
    if isinstance(declared_channel, dict) and "status" in declared_channel:
        discovery_channel_status = str(declared_channel["status"])
        channel_status_source = "reconstruction_json"
    else:
        discovery_channel_status = (
            "ACTIVE"
            if len(channel_keys_present) == len(channel_keys)
            else "ABORTED_FEWER_THAN_3_PARAGRAPHS"
        )
        channel_status_source = "inferred_from_npz_keys"
    if discovery_channel_status not in {
        "ACTIVE",
        "ABORTED_FEWER_THAN_3_PARAGRAPHS",
    }:
        raise ValueError(
            f"unexpected discovery channel status {discovery_channel_status!r}"
        )
    channel_npz_active = len(channel_keys_present) == len(channel_keys)
    if (discovery_channel_status == "ACTIVE") != channel_npz_active:
        raise ValueError(
            "discovery channel status contradicts paragraph arrays in NPZ"
        )
    causal_condition_sets = [
        set(row.get("results", {})) for row in causal_rows
    ]
    if any(value != causal_condition_sets[0] for value in causal_condition_sets):
        raise ValueError("discovery causal rows have inconsistent conditions")
    causal_channel = {"p3_only", "p12", "quote_strip_p3"}
    causal_channel_active = causal_channel <= causal_condition_sets[0]
    causal_channel_partial = bool(
        causal_channel & causal_condition_sets[0]
    ) and not causal_channel_active
    if causal_channel_partial:
        raise ValueError("discovery causal artifact has a partial channel")
    if causal_channel_active != channel_npz_active:
        raise ValueError(
            "discovery causal conditions contradict reconstruction channel state"
        )

    gate = {
        "status": "FEASIBLE" if feasible else "GATE TRAINING FAILURE",
        "feasible": feasible,
        "score_name": "absolute_nla_centered_cosine",
        "score_formula": "cos(perp_m_D(pred_orig), perp_m_D(x))",
        "discovery_mean_direction": stored_m_hat.tolist(),
        "discovery_mean_direction_dtype": "float64 JSON numbers",
        "discovery_mean_direction_sha256": hashlib.sha256(
            np.asarray(stored_m_hat, dtype="<f8").tobytes()
        ).hexdigest(),
        "routing_fraction": (
            selected["routing_fraction"] if selected is not None else 0.0
        ),
        "nla_count_discovery": int(route.sum()),
        "threshold": selected["threshold"] if selected is not None else None,
        "tie_hash_cutoff_inclusive": (
            selected["tie_hash_cutoff_inclusive"]
            if selected is not None
            else None
        ),
        "boundary_selected_uid_sha256": (
            selected["boundary_selected_uid_sha256"]
            if selected is not None
            else []
        ),
        "row_uid_hash_rule": "sha256(UTF-8 row_uid), lowercase hexadecimal",
        "heldout_threshold_rule": (
            "route NLA iff score > threshold, or score == threshold and "
            "sha256(row_uid) <= tie_hash_cutoff_inclusive"
            if feasible
            else "always route SAE-big"
        ),
        "catastrophe_definition": (
            "KL_hybrid_at_pos - KL_sae_big_at_pos > 1.0 nat"
        ),
        "catastrophe_threshold_nat": CAT_REGRET_NAT,
        "maximum_discovery_catastrophe_rate": MAX_CAT_RATE,
        "candidate_routing_fraction_grid": list(FRACTIONS),
        "objective": (
            "maximize sum(KL_big-KL_hybrid)/sum(KL_zero) among feasible "
            "candidates; differences within 1e-6 choose smaller fraction"
        ),
        "objective_tie_tolerance": OBJECTIVE_TIE_TOL,
        "discovery_channel_status": discovery_channel_status,
        "discovery_channel_active": discovery_channel_status == "ACTIVE",
        "discovery_channel_status_source": channel_status_source,
    }
    gate["gate_contract_sha256"] = canonical_hash(gate)

    assignments = []
    for i, uid in enumerate(recon_uids):
        assignments.append(
            {
                "row_uid": uid,
                "row_uid_sha256": uid_hashes[i],
                "q": float(scores[i]),
                "route_nla": bool(route[i]),
                "kl_orig": float(kl_orig[i]),
                "kl_sae_big": float(kl_big[i]),
                "kl_zero": float(kl_zero[i]),
                "kl_big_minus_hybrid": float(
                    kl_big[i] - (kl_orig[i] if route[i] else kl_big[i])
                ),
                "catastrophic_regret": bool(
                    (kl_orig[i] if route[i] else kl_big[i]) - kl_big[i]
                    > CAT_REGRET_NAT
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "experiment": "N5 discovery-only selective hybrid gate freeze",
        "status": "complete",
        "split": "discovery",
        "inputs": {
            "recon_json": str(args.recon_json),
            "recon_json_sha256": recon_sha,
            "recon_npz": str(args.recon_npz),
            "recon_npz_sha256": npz_sha,
            "causal": str(args.causal),
            "causal_sha256": causal_sha,
            "activations_sha256": str(recon_activation_sha),
            "plan": str(args.plan),
            "plan_sha256": plan_sha,
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_sha,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha,
            "script_sha256": script_sha,
        },
        "protocol": {
            "discovery_only": True,
            "n_rows": EXPECTED_ROWS,
            "independent_unit": "content group; exactly one row per group",
            "heldout_used": False,
            "rowwise_kl_zero_division": False,
        },
        "qa": {
            "n_rows": EXPECTED_ROWS,
            "n_unique_row_uids": len(set(recon_uids)),
            "n_unique_content_groups": len(
                {
                    str(row.get("content_group_id", row_uid(row)))
                    for row in causal_rows
                }
            ),
            "by_corpus": dict(
                Counter(str(row.get("corpus")) for row in causal_rows)
            ),
            "stored_m_hat_norm": float(np.linalg.norm(stored_m_hat)),
            "stored_m_hat_max_abs_vs_x_mean": m_hat_error,
            "reported_score_max_abs_error": score_max_error,
            "sum_kl_zero": zero_sum,
            "negative_kl_clamped": negative_clamped,
            "activation_sha256_crossref": str(recon_activation_sha),
            "required_upstream_hash_crossrefs": True,
            "rows_match_frozen_plan": True,
        },
        "gate": gate,
        "candidates": candidates,
        "discovery_assignments": assignments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    output_sha = sha256_file(args.out)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{output_sha}  {args.out.name}\n", encoding="utf-8"
    )
    print(
        "N5_GATE_FROZEN "
        f"status={gate['status']} q={gate['routing_fraction']} "
        f"threshold={gate['threshold']} sha256={output_sha} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
