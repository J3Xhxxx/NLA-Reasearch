#!/usr/bin/env python3
"""Analyze the frozen N5 held-out selective hybrid and paragraph channel.

Only held-out rows enter inference.  The discovery gate and mean direction are
read as immutable inputs, never re-fit.  Aggregate causal recovery uses a
ratio of sums (never row-wise division by KL_zero), and rare catastrophic
regret uses an exact one-sided Clopper-Pearson upper bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ROWS = 400
EXPECTED_PREREG_SHA256 = (
    "63dc31b4f9607e54ac15f1c364fcae2ee903f228fe0afb4d388c6dad1a6f9103"
)
EXPECTED_MODEL_MANIFEST_SHA256 = (
    "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735"
)
N_BOOT = 50_000
SEED = 20260730
NEGATIVE_KL_TOL = -1e-7
IDENTITY_KL_TOL = 1e-5
CAT_REGRET_NAT = 1.0
TAIL_MARGIN = 0.03
PARITY_MARGIN = 0.01


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uid_hash(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


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
            f"{path} SHA-256 differs from its sidecar: "
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
        raise ValueError(f"held-out NPZ lacks embedded provenance key {key}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(
            f"held-out NPZ provenance key {key} must be scalar, "
            f"found shape {value.shape}"
        )
    observed = str(value.item())
    if observed != expected:
        raise ValueError(
            f"held-out NPZ embedded {key}={observed!r}, "
            f"expected {expected!r}"
        )


def parse_weight_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
        rows.append({"sha256": fields[0].lower(), "path": fields[1].strip()})
    if len(rows) != 25 or len({row["path"] for row in rows}) != 25:
        raise ValueError("full model manifest must contain 25 unique files")
    return rows


def validate_actual_model_bindings(
    recon: dict,
    causal: dict,
    manifest_rows: list[dict[str, str]],
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
    expected = {row["path"]: row["sha256"] for row in manifest_rows}
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
        for path, digest in expected.items()
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
        for path, digest in expected.items()
        if "/gemma-3-12b-it/" in path
    }
    causal_base = causal.get("inputs", {}).get(
        "verified_base_files_from_full_manifest"
    )
    if causal_base != expected_base:
        raise ValueError(
            "causal actual base files differ from the frozen 25-file manifest"
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


def require_rows(payload: dict, label: str) -> list[dict]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        found = len(rows) if isinstance(rows, list) else None
        raise ValueError(f"{label} requires {EXPECTED_ROWS} rows, found {found}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be JSON objects")
    return rows


def row_uid(row: dict) -> str:
    value = row.get("row_uid")
    if value is None or not str(value):
        raise ValueError("every held-out row must carry frozen row_uid")
    return str(value)


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
    values = np.sum(pred_centered * target_centered, axis=1) / denominator
    if not np.isfinite(values).all():
        raise ValueError("centered cosine contains non-finite values")
    return values


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


def sanitize_kl(
    value: Any,
    uid: str,
    condition: str,
    field: str,
    clamped: list[dict],
) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{uid} {condition}.{field} is non-finite")
    if number < NEGATIVE_KL_TOL:
        raise ValueError(
            f"{uid} {condition}.{field}={number} is below -1e-7"
        )
    if number < 0:
        clamped.append(
            {
                "row_uid": uid,
                "condition": condition,
                "field": field,
                "raw_value": number,
            }
        )
        return 0.0
    return number


def binomial_cdf(k: int, n: int, probability: float) -> float:
    """Stable P[X <= k] for X~Binomial(n,p), without scipy."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(j + 1)
        - math.lgamma(n - j + 1)
        + j * log_p
        + (n - j) * log_q
        for j in range(k + 1)
    ]
    maximum = max(logs)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)


def clopper_pearson_upper(
    successes: int, trials: int, confidence: float = 0.95
) -> float:
    """One-sided exact binomial upper confidence bound."""
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        # CDF decreases monotonically with p.
        if binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


class StratifiedBootstrap:
    """Frozen stratified content-group resamples reused by every endpoint."""

    def __init__(
        self,
        strata: np.ndarray,
        n_boot: int = N_BOOT,
        seed: int = SEED,
    ) -> None:
        self.strata = np.asarray(strata, dtype=object)
        self.n_boot = int(n_boot)
        self.n_rows = len(self.strata)
        rng = np.random.default_rng(seed)
        self.draws: list[tuple[np.ndarray, np.ndarray]] = []
        for name in sorted(set(self.strata.tolist())):
            positions = np.flatnonzero(self.strata == name)
            if len(positions) == 0:
                raise ValueError(f"empty bootstrap stratum {name!r}")
            local = rng.integers(
                0,
                len(positions),
                size=(self.n_boot, len(positions)),
                dtype=np.int32,
            )
            self.draws.append((positions, local))

    def sums(self, values: np.ndarray, chunk_size: int = 500) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_rows,) or not np.isfinite(values).all():
            raise ValueError("bootstrap input has invalid shape or values")
        output = np.empty(self.n_boot, dtype=np.float64)
        for start in range(0, self.n_boot, chunk_size):
            stop = min(start + chunk_size, self.n_boot)
            total = np.zeros(stop - start, dtype=np.float64)
            for positions, local in self.draws:
                total += values[positions[local[start:stop]]].sum(axis=1)
            output[start:stop] = total
        return output

    def means(self, values: np.ndarray) -> np.ndarray:
        return self.sums(values) / self.n_rows

    def ratios(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        num = self.sums(numerator)
        den = self.sums(denominator)
        if np.any(den <= 1e-12):
            raise ValueError("bootstrap ratio has a nonpositive denominator")
        return num / den


def interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    alpha = (1.0 - level) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def lower_bound(values: np.ndarray, confidence: float = 0.95) -> float:
    return float(np.quantile(values, 1.0 - confidence))


def raw_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "sum": float(values.sum(dtype=np.float64)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "min": float(values.min()),
        "n_positive": int((values > 0).sum()),
        "n_negative": int((values < 0).sum()),
        "n_zero": int((values == 0).sum()),
    }


def subgroup_descriptives(
    fields: dict[str, np.ndarray],
    conditions: dict[str, dict[str, np.ndarray]],
    route_nla: np.ndarray,
    clean_ce: np.ndarray,
) -> dict:
    output: dict[str, dict] = {}
    zero = conditions["zero"]["kl_at_pos"]
    hybrid = conditions["hybrid"]["kl_at_pos"]
    big = conditions["sae_big"]["kl_at_pos"]
    for field in ("corpus", "source", "lang"):
        values = np.asarray(fields[field], dtype=object)
        output[field] = {}
        for name in sorted(set(values.tolist())):
            selected = values == name
            zero_sum = float(zero[selected].sum())
            zero16_sum = float(
                conditions["zero"]["kl_mean_first16"][selected].sum()
            )
            condition_metrics = {}
            for condition, metrics in conditions.items():
                kl = metrics["kl_at_pos"][selected]
                kl16 = metrics["kl_mean_first16"][selected]
                ce16 = metrics["ce_first16"][selected]
                condition_metrics[condition] = {
                    "kl_at_pos": raw_summary(kl),
                    "ratio_of_sums_recovered_at_pos": (
                        float(1.0 - kl.sum() / zero_sum)
                        if zero_sum > 1e-12
                        else None
                    ),
                    "kl_mean_first16": raw_summary(kl16),
                    "ratio_of_sums_recovered_first16": (
                        float(1.0 - kl16.sum() / zero16_sum)
                        if zero16_sum > 1e-12
                        else None
                    ),
                    "ce_first16": raw_summary(ce16),
                    "ce_delta_from_clean_first16": raw_summary(
                        ce16 - clean_ce[selected]
                    ),
                }
            output[field][str(name)] = {
                "n": int(selected.sum()),
                "coverage": float(route_nla[selected].mean()),
                "hybrid_gain": (
                    float((big[selected] - hybrid[selected]).sum() / zero_sum)
                    if zero_sum > 1e-12
                    else None
                ),
                "catastrophic_regret_rate": float(
                    ((hybrid[selected] - big[selected]) > CAT_REGRET_NAT).mean()
                ),
                "conditions": condition_metrics,
                "paired_sae_big_minus_hybrid": {
                    "kl_at_pos": raw_summary(
                        conditions["sae_big"]["kl_at_pos"][selected]
                        - conditions["hybrid"]["kl_at_pos"][selected]
                    ),
                    "kl_mean_first16": raw_summary(
                        conditions["sae_big"]["kl_mean_first16"][selected]
                        - conditions["hybrid"]["kl_mean_first16"][selected]
                    ),
                    "ce_first16": raw_summary(
                        conditions["sae_big"]["ce_first16"][selected]
                        - conditions["hybrid"]["ce_first16"][selected]
                    ),
                },
            }
    return output


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--recon-json", required=True, type=Path)
    parser.add_argument("--recon-npz", required=True, type=Path)
    parser.add_argument("--causal", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    if args.out.exists() or args.markdown.exists():
        raise ValueError("refusing to overwrite a frozen N5 analysis artifact")

    gate_sha = sha256_file(args.gate)
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
    verify_sha256_sidecar(args.gate, gate_sha)
    verify_sha256_sidecar(args.recon_json, recon_sha)
    verify_sha256_sidecar(args.recon_npz, npz_sha)
    verify_sha256_sidecar(args.causal, causal_sha)
    verify_sha256_sidecar(args.plan, plan_sha)
    verify_sha256_sidecar(args.manifest, manifest_sha)
    model_weight_files = parse_weight_manifest(args.manifest)

    gate_payload = read_json(args.gate)
    recon = read_json(args.recon_json)
    causal = read_json(args.causal)
    plan = read_json(args.plan)
    plan_rows = frozen_plan_rows(plan, "heldout", EXPECTED_ROWS)
    gate = gate_payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("gate artifact lacks a gate object")
    if gate_payload.get("split") != "discovery":
        raise ValueError("gate artifact was not frozen on discovery")
    gate_inputs = gate_payload.get("inputs", {})
    if gate_inputs.get("prereg_sha256") != prereg_sha:
        raise ValueError("gate/preregistration hash mismatch")
    if gate_inputs.get("manifest_sha256") != manifest_sha:
        raise ValueError("gate/manifest hash mismatch")
    if gate_inputs.get("plan_sha256") != plan_sha:
        raise ValueError("gate/cohort-plan hash mismatch")
    gate_feasible = bool(gate.get("feasible"))
    if gate.get("status") not in {"FEASIBLE", "GATE TRAINING FAILURE"}:
        raise ValueError(f"unexpected gate status {gate.get('status')!r}")
    if gate_feasible != (gate.get("status") == "FEASIBLE"):
        raise ValueError("gate feasible/status fields disagree")
    if gate.get("score_name") != "absolute_nla_centered_cosine":
        raise ValueError("gate uses a forbidden score family")

    recon_rows = require_rows(recon, "held-out reconstruction")
    causal_rows = require_rows(causal, "held-out causal")
    for label, rows in (
        ("held-out reconstruction", recon_rows),
        ("held-out causal", causal_rows),
    ):
        declared = {str(row.get("split")) for row in rows if "split" in row}
        if declared and declared != {"heldout"}:
            raise ValueError(f"{label} has unexpected splits {sorted(declared)}")
        uids = [row_uid(row) for row in rows]
        if len(set(uids)) != EXPECTED_ROWS:
            raise ValueError(f"{label} row_uid values are not unique")
        groups = [
            str(row.get("content_group_id"))
            for row in rows
            if row.get("content_group_id") is not None
        ]
        if groups and (
            len(groups) != EXPECTED_ROWS or len(set(groups)) != EXPECTED_ROWS
        ):
            raise ValueError(f"{label} violates one-row-per-content-group")
    validate_rows_against_plan(
        recon_rows, plan_rows, "held-out reconstruction"
    )
    validate_rows_against_plan(causal_rows, plan_rows, "held-out causal")
    if str(causal.get("split", "heldout")) != "heldout":
        raise ValueError("causal result is not heldout")
    if causal.get("status") not in (None, "complete"):
        raise ValueError(f"causal result status is {causal.get('status')!r}")

    require_input_hash(recon, "prereg_sha256", prereg_sha, "reconstruction")
    require_input_hash(causal, "prereg_sha256", prereg_sha, "causal")
    require_input_hash(causal, "recon_sha256", npz_sha, "causal")
    require_input_hash(causal, "gate_sha256", gate_sha, "causal")
    require_input_hash(recon, "plan_sha256", plan_sha, "reconstruction")
    require_input_hash(recon, "gate_sha256", gate_sha, "reconstruction")
    if recon.get("outputs", {}).get("vecs_sha256") != npz_sha:
        raise ValueError(
            "held-out reconstruction JSON does not bind the supplied NPZ"
        )
    validate_actual_model_bindings(
        recon, causal, model_weight_files, manifest_sha
    )
    recon_activation_sha = recon.get("inputs", {}).get("activations_sha256")
    causal_activation_sha = causal.get("inputs", {}).get("activations_sha256")
    if not recon_activation_sha or not causal_activation_sha:
        raise ValueError(
            "held-out reconstruction and causal artifacts must both bind "
            "the activation parquet"
        )
    if recon_activation_sha != causal_activation_sha:
        raise ValueError("held-out reconstruction/causal activation hashes differ")

    key_map = {
        "orig": "pred_orig",
        "sae_small": "recon_sae_small",
        "sae_big": "recon_sae_big",
        "p3_only": "pred_p3_only",
        "p12": "pred_p12",
        "quote_strip_p3": "pred_quote_strip_p3",
    }
    with np.load(args.recon_npz, allow_pickle=False) as archive:
        required = {
            "x",
            "pred_orig",
            "recon_sae_small",
            "recon_sae_big",
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
            raise ValueError(f"held-out NPZ lacks required keys: {missing}")
        require_npz_scalar_string(
            archive, "activations_sha256", str(recon_activation_sha)
        )
        require_npz_scalar_string(archive, "plan_sha256", plan_sha)
        require_npz_scalar_string(
            archive, "model_manifest_sha256", manifest_sha
        )
        require_npz_scalar_string(archive, "prereg_sha256", prereg_sha)
        require_npz_scalar_string(archive, "gate_sha256", gate_sha)
        x = np.asarray(archive["x"], dtype=np.float32)
        stored_m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
        npz_uids = [
            str(value) for value in np.asarray(archive["row_uids"]).tolist()
        ]
        vectors = {
            condition: np.asarray(archive[key], dtype=np.float32)
            for condition, key in key_map.items()
            if key in archive.files
        }
    channel_names = {"p3_only", "p12", "quote_strip_p3"}
    vector_channel_present = channel_names <= set(vectors)
    vector_channel_partial = bool(channel_names & set(vectors)) and not (
        vector_channel_present
    )
    if vector_channel_partial:
        raise ValueError("held-out NPZ contains a partial paragraph channel")
    declared_channel = recon.get("channel")
    if not isinstance(declared_channel, dict) or "status" not in declared_channel:
        raise ValueError("held-out reconstruction omits frozen channel status")
    heldout_channel_status = str(declared_channel["status"])
    if heldout_channel_status not in {
        "ACTIVE",
        "ABORTED_FEWER_THAN_3_PARAGRAPHS",
        "ABORTED_ON_DISCOVERY",
    }:
        raise ValueError(
            f"unexpected held-out channel status {heldout_channel_status!r}"
        )
    if (heldout_channel_status == "ACTIVE") != vector_channel_present:
        raise ValueError(
            "held-out channel status contradicts paragraph arrays in NPZ"
        )
    if x.ndim != 2 or x.shape[0] != EXPECTED_ROWS:
        raise ValueError(f"unexpected held-out x shape {x.shape}")
    if stored_m_hat.shape != (x.shape[1],):
        raise ValueError(f"unexpected held-out m_hat shape {stored_m_hat.shape}")
    if not np.isfinite(x).all():
        raise ValueError("held-out x contains non-finite values")
    for condition, value in vectors.items():
        if value.shape != x.shape or not np.isfinite(value).all():
            raise ValueError(f"invalid held-out vector array for {condition}")

    recon_uids = [row_uid(row) for row in recon_rows]
    if npz_uids != recon_uids:
        raise ValueError("NPZ row_uids do not match reconstruction JSON order")
    frozen_m_hat = np.asarray(
        gate.get("discovery_mean_direction"), dtype=np.float64
    )
    if frozen_m_hat.shape != (x.shape[1],):
        raise ValueError("gate discovery mean direction has the wrong shape")
    if abs(float(np.linalg.norm(frozen_m_hat)) - 1.0) > 1e-6:
        raise ValueError("gate discovery mean direction is not unit norm")
    m_hat_error = float(np.max(np.abs(stored_m_hat - frozen_m_hat)))
    if m_hat_error > 1e-12:
        raise ValueError(
            "held-out NPZ did not use the exact frozen discovery m_hat: "
            f"max_abs={m_hat_error}"
        )
    m_hat_hash = hashlib.sha256(
        np.asarray(frozen_m_hat, dtype="<f8").tobytes()
    ).hexdigest()
    if m_hat_hash != gate.get("discovery_mean_direction_sha256"):
        raise ValueError("gate mean-direction hash does not match its values")

    score_by_condition: dict[str, np.ndarray] = {}
    score_max_error: dict[str, float] = {}
    for condition, prediction in vectors.items():
        values = centered_cosine(prediction, x, frozen_m_hat)
        score_by_condition[condition] = values
        reported = []
        for row in recon_rows:
            try:
                reported.append(float(row["scores"][condition]["cos_c"]))
            except (KeyError, TypeError, ValueError):
                reported = []
                break
        if reported:
            error = float(
                np.max(np.abs(values - np.asarray(reported, dtype=float)))
            )
            score_max_error[condition] = error
            if not np.allclose(
                values, np.asarray(reported), rtol=1e-7, atol=5e-7
            ):
                raise ValueError(
                    f"reconstruction JSON/NPZ {condition}.cos_c mismatch: "
                    f"{error}"
                )
    q_values = score_by_condition["orig"]

    if gate_feasible:
        threshold = float(gate["threshold"])
        tie_cutoff = str(gate["tie_hash_cutoff_inclusive"])
        route_nla = (q_values > threshold) | (
            (q_values == threshold)
            & np.asarray(
                [uid_hash(uid) <= tie_cutoff for uid in recon_uids], dtype=bool
            )
        )
    else:
        threshold = None
        tie_cutoff = None
        route_nla = np.zeros(EXPECTED_ROWS, dtype=bool)

    reported_route_checked = 0
    reported_q_max_abs_error = 0.0
    for index, row in enumerate(recon_rows):
        if "q" not in row:
            raise ValueError(
                f"held-out reconstruction omits frozen q for {row_uid(row)}"
            )
        q_error = abs(float(row["q"]) - float(q_values[index]))
        reported_q_max_abs_error = max(reported_q_max_abs_error, q_error)
        if not np.isclose(
            float(row["q"]), float(q_values[index]), rtol=1e-7, atol=5e-7
        ):
            raise ValueError(
                f"held-out q mismatch for {row_uid(row)}: max_abs={q_error}"
            )
        reported_route = None
        for key in ("route_nla", "gate_route_nla"):
            if key in row:
                if not isinstance(row[key], bool):
                    raise ValueError(
                        f"held-out {key} is not boolean for {row_uid(row)}"
                    )
                reported_route = row[key]
                break
        if reported_route is None and isinstance(row.get("gate"), dict):
            if "route_nla" in row["gate"]:
                if not isinstance(row["gate"]["route_nla"], bool):
                    raise ValueError(
                        f"held-out nested route_nla is not boolean for "
                        f"{row_uid(row)}"
                    )
                reported_route = row["gate"]["route_nla"]
        if reported_route is None:
            raise ValueError(
                f"held-out reconstruction omits route_nla for {row_uid(row)}"
            )
        reported_route_checked += 1
        if reported_route != bool(route_nla[index]):
            raise ValueError(
                f"held-out gate assignment mismatch for {row_uid(row)}"
            )
        expected_routed_to = "nla" if route_nla[index] else "sae_big"
        if row.get("routed_to") != expected_routed_to:
            raise ValueError(
                f"held-out routed_to mismatch for {row_uid(row)}: "
                f"{row.get('routed_to')!r} != {expected_routed_to!r}"
            )
    if reported_route_checked != EXPECTED_ROWS:
        raise ValueError("not every held-out route assignment was verified")

    causal_by_uid = {row_uid(row): row for row in causal_rows}
    if set(causal_by_uid) != set(recon_uids):
        raise ValueError("held-out reconstruction/causal UID sets differ")
    aligned_causal = [causal_by_uid[uid] for uid in recon_uids]
    mandatory = {
        "identity",
        "orig",
        "sae_small",
        "sae_big",
        "zero",
    }
    available = set(aligned_causal[0].get("results", {}))
    for row in aligned_causal[1:]:
        if set(row.get("results", {})) != available:
            raise ValueError("held-out causal rows have inconsistent conditions")
    if not mandatory <= available:
        raise ValueError(
            f"held-out causal results lack conditions {sorted(mandatory-available)}"
        )
    causal_channel_present = channel_names <= available
    causal_channel_partial = bool(channel_names & available) and not (
        causal_channel_present
    )
    if causal_channel_partial:
        raise ValueError("held-out causal artifact has a partial channel")
    if causal_channel_present != vector_channel_present:
        raise ValueError(
            "held-out causal conditions contradict reconstruction channel state"
        )
    if (
        not bool(gate.get("discovery_channel_active", False))
        and vector_channel_present
    ):
        raise ValueError(
            "held-out paragraph variants exist after the discovery-global abort"
        )

    negative_clamped: list[dict] = []
    condition_arrays: dict[str, dict[str, np.ndarray]] = {}
    for condition in sorted(available):
        kl_pos = np.empty(EXPECTED_ROWS, dtype=float)
        kl16 = np.empty(EXPECTED_ROWS, dtype=float)
        ce16 = np.empty(EXPECTED_ROWS, dtype=float)
        for i, (uid, row) in enumerate(zip(recon_uids, aligned_causal)):
            result = row["results"][condition]
            kl_pos[i] = sanitize_kl(
                result["kl_at_pos"],
                uid,
                condition,
                "kl_at_pos",
                negative_clamped,
            )
            kl16[i] = sanitize_kl(
                result["kl_mean_first16"],
                uid,
                condition,
                "kl_mean_first16",
                negative_clamped,
            )
            ce16[i] = float(result["ce_first16"])
            if not np.isfinite(ce16[i]):
                raise ValueError(f"{uid} {condition}.ce_first16 is non-finite")
        condition_arrays[condition] = {
            "kl_at_pos": kl_pos,
            "kl_mean_first16": kl16,
            "ce_first16": ce16,
        }

    clean_ce = np.asarray(
        [float(row["ce_clean_first16"]) for row in aligned_causal], dtype=float
    )
    if not np.isfinite(clean_ce).all():
        raise ValueError("clean CE contains non-finite values")
    for field in ("kl_at_pos", "kl_mean_first16", "ce_first16"):
        condition_arrays.setdefault("hybrid", {})[field] = np.where(
            route_nla,
            condition_arrays["orig"][field],
            condition_arrays["sae_big"][field],
        )
    oracle_uses_nla = (
        condition_arrays["orig"]["kl_at_pos"]
        < condition_arrays["sae_big"]["kl_at_pos"]
    )
    for field in ("kl_at_pos", "kl_mean_first16", "ce_first16"):
        condition_arrays.setdefault("oracle_nla_sae_big", {})[field] = np.where(
            oracle_uses_nla,
            condition_arrays["orig"][field],
            condition_arrays["sae_big"][field],
        )

    identity_max = float(condition_arrays["identity"]["kl_at_pos"].max())
    identity16_max = float(
        condition_arrays["identity"]["kl_mean_first16"].max()
    )
    if max(identity_max, identity16_max) > IDENTITY_KL_TOL:
        raise ValueError(
            f"identity KL QA failed: pos={identity_max}, kl16={identity16_max}"
        )
    if not bool(causal.get("qa", {}).get("provenance_all_bit_exact", False)):
        raise ValueError("causal artifact does not report bit-exact provenance")

    fields = {
        name: np.asarray(
            [str(row.get(name)) for row in aligned_causal], dtype=object
        )
        for name in ("corpus", "source", "lang")
    }
    strata = np.asarray(
        [
            "pile" if str(value).lower() == "pile" else "xnli"
            if str(value).lower() == "xnli" else str(value).lower()
            for value in fields["corpus"]
        ],
        dtype=object,
    )
    corpus_counts = Counter(strata.tolist())
    if corpus_counts != Counter({"pile": 300, "xnli": 100}):
        raise ValueError(
            "held-out corpus quotas differ from preregistration: "
            f"{dict(corpus_counts)}"
        )
    bootstrap = StratifiedBootstrap(strata)

    zero = condition_arrays["zero"]["kl_at_pos"]
    zero_sum = float(zero.sum(dtype=np.float64))
    if zero_sum <= 1e-6:
        raise ValueError(f"held-out sum(KL_zero)={zero_sum} is invalid")
    big = condition_arrays["sae_big"]["kl_at_pos"]
    hybrid = condition_arrays["hybrid"]["kl_at_pos"]
    gain_numerator = big - hybrid
    g_point = float(gain_numerator.sum(dtype=np.float64) / zero_sum)
    delta_raw = float(gain_numerator.mean())
    boot_gain_sum = bootstrap.sums(gain_numerator)
    boot_zero_sum = bootstrap.sums(zero)
    if np.any(boot_zero_sum <= 1e-12):
        raise ValueError("held-out bootstrap generated zero KL denominator")
    boot_g = boot_gain_sum / boot_zero_sum
    boot_raw = boot_gain_sum / EXPECTED_ROWS
    boot_coverage = bootstrap.means(route_nla.astype(float))

    g_lower95 = lower_bound(boot_g)
    g_ci90 = interval(boot_g, 0.90)
    g_ci95 = interval(boot_g, 0.95)
    raw_ci95 = interval(boot_raw, 0.95)
    coverage = float(route_nla.mean())
    coverage_lower95 = lower_bound(boot_coverage)

    catastrophic_hybrid = (hybrid - big) > CAT_REGRET_NAT
    catastrophic_nla = (
        condition_arrays["orig"]["kl_at_pos"] - big
    ) > CAT_REGRET_NAT
    catastrophic_count = int(catastrophic_hybrid.sum())
    catastrophic_rate = catastrophic_count / EXPECTED_ROWS
    catastrophic_upper = clopper_pearson_upper(
        catastrophic_count, EXPECTED_ROWS, 0.95
    )
    tail_reduction = catastrophic_nla.astype(float) - catastrophic_hybrid
    boot_tail_reduction = bootstrap.means(tail_reduction)

    feasible_pass = gate_feasible
    coverage_pass = coverage_lower95 > 0.10
    superiority_pass = g_lower95 > 0
    tail_pass = catastrophic_upper < TAIL_MARGIN
    parity_pass = (
        g_ci90[0] >= -PARITY_MARGIN and g_ci90[1] <= PARITY_MARGIN
    )
    failed_gates: list[str] = []
    if not feasible_pass:
        failed_gates.append("discovery gate infeasible")
    if not coverage_pass:
        failed_gates.append("coverage one-sided 95% lower bound <= .10")
    if not superiority_pass:
        failed_gates.append("G one-sided 95% lower bound <= 0")
    if not tail_pass:
        failed_gates.append(
            "catastrophic-regret exact one-sided 95% upper bound >= .03"
        )
    selective_improvement = (
        feasible_pass and coverage_pass and superiority_pass and tail_pass
    )
    safe_parity = (
        feasible_pass
        and coverage_pass
        and not superiority_pass
        and tail_pass
        and parity_pass
    )
    failed_decision_gates = list(failed_gates)
    if not selective_improvement and not safe_parity and not parity_pass:
        failed_decision_gates.append(
            "G two-sided 90% CI is not wholly inside [-.01,+.01]"
        )
    decision_h5a = (
        "SELECTIVE IMPROVEMENT"
        if selective_improvement
        else "SAFE SELECTIVE PARITY"
        if safe_parity
        else "NO SELECTIVE CLAIM"
    )

    condition_summary = {}
    zero16 = condition_arrays["zero"]["kl_mean_first16"]
    zero16_sum = float(zero16.sum(dtype=np.float64))
    for condition, metrics in condition_arrays.items():
        kl = metrics["kl_at_pos"]
        kl16 = metrics["kl_mean_first16"]
        ce = metrics["ce_first16"]
        condition_summary[condition] = {
            "kl_at_pos": raw_summary(kl),
            "ratio_of_sums_recovered_at_pos": float(1.0 - kl.sum() / zero_sum),
            "kl_mean_first16": raw_summary(kl16),
            "ratio_of_sums_recovered_first16": (
                float(1.0 - kl16.sum() / zero16_sum)
                if zero16_sum > 1e-12
                else None
            ),
            "ce_first16": raw_summary(ce),
            "ce_delta_from_clean_first16": raw_summary(ce - clean_ce),
        }

    h5b_required = {"p3_only", "p12", "quote_strip_p3"}
    discovery_channel_active = bool(
        gate.get("discovery_channel_active", False)
    )
    heldout_channel_present = h5b_required <= available and h5b_required <= set(
        vectors
    )
    h5b: dict[str, Any]
    if not discovery_channel_active:
        h5b = {
            "status": "ABORTED",
            "replicated": False,
            "reason": (
                "discovery contained an explanation with fewer than three "
                "paragraphs; global H5-B abort was frozen before heldout"
            ),
        }
    elif not heldout_channel_present:
        h5b = {
            "status": "ABORTED",
            "replicated": False,
            "reason": (
                "heldout paragraph channel was aborted or required "
                "reconstruction/causal conditions are absent"
            ),
        }
    else:
        retrieval_diagnostics = {}
        for condition in sorted(h5b_required):
            value = (
                recon.get("summary", {})
                .get(condition, {})
                .get("retrieval")
            )
            if not isinstance(value, dict) or not value:
                raise ValueError(
                    f"mandatory retrieval diagnostic is missing for {condition}"
                )
            retrieval_diagnostics[condition] = value
        p3 = condition_arrays["p3_only"]["kl_at_pos"]
        p12 = condition_arrays["p12"]["kl_at_pos"]
        orig = condition_arrays["orig"]["kl_at_pos"]
        gp_num = p12 - p3
        gp_point = float(gp_num.sum() / zero_sum)
        boot_gp = bootstrap.sums(gp_num) / boot_zero_sum
        gp_ci95 = interval(boot_gp, 0.95)
        r_orig = float(1.0 - orig.sum() / zero_sum)
        r_p3 = float(1.0 - p3.sum() / zero_sum)
        retention = r_p3 / r_orig if abs(r_orig) > 1e-12 else None
        boot_r_orig = 1.0 - bootstrap.sums(orig) / boot_zero_sum
        boot_r_p3 = 1.0 - bootstrap.sums(p3) / boot_zero_sum
        retention_testable = bool(
            retention is not None
            and np.isfinite(retention)
            and r_orig > 0
            and np.all(boot_r_orig > 0)
        )
        if retention_testable:
            boot_retention = boot_r_p3 / boot_r_orig
            retention_lower = lower_bound(boot_retention)
            retention_ci95 = interval(boot_retention, 0.95)
        else:
            boot_retention = None
            retention_lower = None
            retention_ci95 = None
        gp_pass = gp_ci95[0] > 0
        retention_pass = bool(
            retention_testable and retention_lower is not None
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
            "g_p3_p12": gp_point,
            "g_p3_p12_ci95_stratified_bootstrap": gp_ci95,
            "raw_kl_p12_minus_p3": {
                **raw_summary(gp_num),
                "ci95_stratified_bootstrap": interval(
                    bootstrap.means(gp_num), 0.95
                ),
            },
            "r_orig": r_orig,
            "r_p3": r_p3,
            "retention_t": retention,
            "retention_testable": retention_testable,
            "retention_denominator_rule": (
                "R_orig point estimate and all 50,000 bootstrap R_orig values "
                "must be strictly positive"
            ),
            "bootstrap_r_orig_min": float(boot_r_orig.min()),
            "retention_one_sided_95_lower": retention_lower,
            "retention_ci95_stratified_bootstrap": retention_ci95,
            "gates": {
                "g_p3_p12_ci95_lower_gt_0": gp_pass,
                "retention_one_sided_95_lower_gt_0.90": retention_pass,
            },
            "sign_count_p12_minus_p3": {
                "positive": int((gp_num > 0).sum()),
                "negative": int((gp_num < 0).sum()),
                "zero": int((gp_num == 0).sum()),
            },
            "diagnostics": {
                "quote_strip_p3": condition_summary["quote_strip_p3"],
                "kl16_p12_minus_p3": raw_summary(
                    condition_arrays["p12"]["kl_mean_first16"]
                    - condition_arrays["p3_only"]["kl_mean_first16"]
                ),
                "ce16_p12_minus_p3": raw_summary(
                    condition_arrays["p12"]["ce_first16"]
                    - condition_arrays["p3_only"]["ce_first16"]
                ),
                "centered_cosine": {
                    condition: raw_summary(score_by_condition[condition])
                    for condition in h5b_required
                },
                "retrieval": {
                    condition: retrieval_diagnostics[condition]
                    for condition in sorted(h5b_required)
                },
            },
        }

    contrasts = {
        "sae_big_minus_hybrid_kl_at_pos": {
            **raw_summary(gain_numerator),
            "ci95_stratified_bootstrap": raw_ci95,
        },
        "sae_big_minus_hybrid_kl16": raw_summary(
            condition_arrays["sae_big"]["kl_mean_first16"]
            - condition_arrays["hybrid"]["kl_mean_first16"]
        ),
        "sae_big_minus_hybrid_ce16": raw_summary(
            condition_arrays["sae_big"]["ce_first16"]
            - condition_arrays["hybrid"]["ce_first16"]
        ),
        "always_nla_tail_minus_hybrid_tail": {
            "point_difference": float(tail_reduction.mean()),
            "ci95_stratified_bootstrap": interval(
                boot_tail_reduction, 0.95
            ),
            "n_always_nla_catastrophes": int(catastrophic_nla.sum()),
            "n_hybrid_catastrophes": catastrophic_count,
        },
    }

    output = {
        "schema_version": 1,
        "experiment": "N5 preregistered held-out selective hybrid analysis",
        "status": "complete",
        "inputs": {
            "gate": str(args.gate),
            "gate_sha256": gate_sha,
            "recon_json": str(args.recon_json),
            "recon_json_sha256": recon_sha,
            "recon_npz": str(args.recon_npz),
            "recon_npz_sha256": npz_sha,
            "causal": str(args.causal),
            "causal_sha256": causal_sha,
            "plan": str(args.plan),
            "plan_sha256": plan_sha,
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_sha,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha,
            "script_sha256": script_sha,
        },
        "model_weight_manifest_entries": model_weight_files,
        "artifact_and_code_hashes": {
            "analysis_script_sha256": script_sha,
            "gate_script_sha256": gate_inputs.get("script_sha256"),
            "reconstruction_script_sha256": recon.get("inputs", {}).get(
                "script_sha256"
            ),
            "causal_script_sha256": causal.get("inputs", {}).get(
                "script_sha256"
            ),
            "activation_parquet_sha256": recon_activation_sha,
            "cohort_plan_sha256": plan_sha,
            "gate_sha256": gate_sha,
            "reconstruction_json_sha256": recon_sha,
            "reconstruction_npz_sha256": npz_sha,
            "causal_json_sha256": causal_sha,
            "full_model_manifest_sha256": manifest_sha,
            "preregistration_sha256": prereg_sha,
        },
        "analysis_protocol": {
            "split": "heldout only",
            "independent_unit": "content group; exactly one row per group",
            "itt": f"all {EXPECTED_ROWS} frozen held-out rows",
            "bootstrap": (
                f"{N_BOOT} content-group resamples stratified pile/xnli; "
                f"seed={SEED}; discovery gate never retrained"
            ),
            "primary": (
                "sum(KL_big-KL_hybrid)/sum(KL_zero); no row-wise division"
            ),
            "catastrophe": "KL_hybrid-KL_big > 1.0 nat",
            "tail_interval": "one-sided 95% exact Clopper-Pearson",
        },
        "qa": {
            "n_rows_itt": EXPECTED_ROWS,
            "n_unique_row_uids": len(set(recon_uids)),
            "n_unique_content_groups": len(
                {
                    str(row.get("content_group_id", row_uid(row)))
                    for row in aligned_causal
                }
            ),
            "by_corpus": dict(corpus_counts),
            "provenance_all_bit_exact": True,
            "identity_kl_at_pos_max": identity_max,
            "identity_kl16_max": identity16_max,
            "negative_kl_clamped": negative_clamped,
            "sum_kl_zero": zero_sum,
            "sum_kl_zero_first16": zero16_sum,
            "frozen_m_hat_max_abs_vs_npz": m_hat_error,
            "score_json_npz_max_abs_error": score_max_error,
            "reported_routes_checked": reported_route_checked,
            "reported_q_max_abs_error": reported_q_max_abs_error,
            "required_upstream_hash_crossrefs": True,
            "rows_match_frozen_plan": True,
            "qa_failures": [],
            "causal_n_forwards": causal.get("n_forwards"),
            "causal_elapsed_seconds": causal.get("elapsed_seconds"),
            "reconstruction_elapsed_seconds": recon.get("elapsed_seconds"),
            "reconstruction_forward_counts": recon.get("forward_counts"),
        },
        "gate_frozen": {
            "status": gate.get("status"),
            "feasible": gate_feasible,
            "score_name": gate.get("score_name"),
            "routing_fraction_discovery": gate.get("routing_fraction"),
            "threshold": threshold,
            "tie_hash_cutoff_inclusive": tie_cutoff,
            "gate_contract_sha256": gate.get("gate_contract_sha256"),
            "heldout_nla_count": int(route_nla.sum()),
            "heldout_assignments": [
                {
                    "row_uid": uid,
                    "q": float(q_values[i]),
                    "route_nla": bool(route_nla[i]),
                }
                for i, uid in enumerate(recon_uids)
            ],
        },
        "h5a_selective_hybrid": {
            "decision": decision_h5a,
            "selective_improvement": selective_improvement,
            "safe_selective_parity": safe_parity,
            "g": g_point,
            "g_one_sided_95_lower": g_lower95,
            "g_ci90_stratified_bootstrap": g_ci90,
            "g_ci95_stratified_bootstrap": g_ci95,
            "delta_raw": delta_raw,
            "delta_raw_ci95_stratified_bootstrap": raw_ci95,
            "coverage": coverage,
            "coverage_one_sided_95_lower": coverage_lower95,
            "catastrophic_regret": {
                "count": catastrophic_count,
                "n": EXPECTED_ROWS,
                "rate": catastrophic_rate,
                "one_sided_95_exact_upper": catastrophic_upper,
            },
            "gates": {
                "feasible_discovery_gate": feasible_pass,
                "coverage_lower_gt_0.10": coverage_pass,
                "g_lower_gt_0": superiority_pass,
                "catastrophe_exact_upper_lt_0.03": tail_pass,
                "g_ci90_inside_pm_0.01": parity_pass,
            },
            "failed_primary_gates": failed_gates,
            "failed_decision_gates": failed_decision_gates,
        },
        "h5b_paragraph_channel": h5b,
        "condition_summary": condition_summary,
        "contrasts": contrasts,
        "subgroups_descriptive_only": subgroup_descriptives(
            fields, condition_arrays, route_nla, clean_ce
        ),
        "oracle_note": (
            "oracle_nla_sae_big is a per-row upper bound selected by primary KL "
            "and never trains or alters the frozen gate"
        ),
        "analysis_elapsed_seconds": round(time.time() - started, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    output_sha = sha256_file(args.out)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{output_sha}  {args.out.name}\n", encoding="utf-8"
    )

    h5b_status = h5b["status"]
    markdown = [
        "# N5 held-out selective hybrid — results",
        "",
        f"- ITT cohort: **{EXPECTED_ROWS} independent content groups** "
        f"(Pile 300 / XNLI 100).",
        f"- Frozen gate: **{gate.get('status')}**; held-out NLA coverage "
        f"**{coverage:.3f}** (one-sided 95% lower {coverage_lower95:.3f}).",
        f"- H5-A: **{decision_h5a}**.",
        f"- H5-B: **{h5b_status}**.",
        "",
        "## H5-A",
        "",
        f"- `G = {g_point:+.6f}`; one-sided 95% lower "
        f"`{g_lower95:+.6f}`; 90% CI "
        f"`[{g_ci90[0]:+.6f}, {g_ci90[1]:+.6f}]`.",
        f"- `Delta_raw = {delta_raw:+.6f}` nat; 95% CI "
        f"`[{raw_ci95[0]:+.6f}, {raw_ci95[1]:+.6f}]`.",
        f"- Catastrophic regret: **{catastrophic_count}/{EXPECTED_ROWS}** "
        f"({catastrophic_rate:.3%}); one-sided exact 95% upper "
        f"**{catastrophic_upper:.3%}**.",
        f"- Failed gates: "
        f"{', '.join(failed_decision_gates) if failed_decision_gates else 'none'}.",
        "",
        "## H5-B",
        "",
    ]
    if "g_p3_p12" in h5b:
        markdown += [
            f"- `G_p3_p12 = {h5b['g_p3_p12']:+.6f}`; 95% CI "
            f"`[{h5b['g_p3_p12_ci95_stratified_bootstrap'][0]:+.6f}, "
            f"{h5b['g_p3_p12_ci95_stratified_bootstrap'][1]:+.6f}]`.",
            f"- `T = R_p3/R_orig = {h5b['retention_t']}`; one-sided 95% "
            f"lower `{h5b['retention_one_sided_95_lower']}`.",
        ]
    else:
        markdown.append(f"- {h5b.get('reason', 'not testable')}")
    markdown += [
        "",
        "## QA",
        "",
        f"- Frozen mean-direction match max abs: `{m_hat_error:.3g}`.",
        f"- Identity KL maxima: position `{identity_max:.3g}`, "
        f"KL16 `{identity16_max:.3g}`.",
        f"- Negative numerical KL values clamped under the frozen rule: "
        f"**{len(negative_clamped)}**.",
        f"- Result SHA-256: `{output_sha}`.",
    ]
    args.markdown.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown), flush=True)
    print(
        f"N5_ANALYSIS_COMPLETE sha256={output_sha} -> "
        f"{args.out} + {args.markdown}",
        flush=True,
    )


if __name__ == "__main__":
    main()
