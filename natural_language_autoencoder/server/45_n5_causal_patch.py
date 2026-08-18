#!/usr/bin/env python3
"""N5 causal patching on a frozen discovery or held-out content-group split.

Every row uses the exact frozen full sequence and batch size one.  A clean
forward both verifies the layer-32 activation bit-for-bit and supplies the
reference logits.  Every substitute then receives its own independent forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM


EXPECTED_PREREG_SHA256 = (
    "63dc31b4f9607e54ac15f1c364fcae2ee903f228fe0afb4d388c6dad1a6f9103"
)
EXPECTED_MODEL_MANIFEST_SHA256 = (
    "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735"
)
ACTIVATION_METADATA_PREFIX = "nla.n5_activation_extraction."


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_npz_scalar_string(
    archive: np.lib.npyio.NpzFile, key: str, expected: str
) -> None:
    if key not in archive.files:
        raise ValueError(f"reconstruction archive lacks provenance key {key}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(
            f"reconstruction provenance key {key} must be scalar, "
            f"found shape {value.shape}"
        )
    observed = str(value.item())
    if observed != expected:
        raise ValueError(
            f"reconstruction embedded {key}={observed!r}, expected {expected!r}"
        )


def validate_gate_file(
    path: Path,
    prereg_sha256: str,
    plan_sha256: str,
    model_manifest_sha256: str,
) -> dict[str, Any]:
    gate_sha256 = sha256_file(path)
    verify_sha256_sidecar(path, gate_sha256)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("split") != "discovery":
        raise ValueError("gate artifact is not a complete discovery freeze")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("gate artifact lacks top-level inputs object")
    required_hashes = {
        "prereg_sha256": prereg_sha256,
        "plan_sha256": plan_sha256,
        "manifest_sha256": model_manifest_sha256,
    }
    for name, expected in required_hashes.items():
        if inputs.get(name) != expected:
            raise ValueError(
                f"gate {name} mismatch: {inputs.get(name)!r} != {expected!r}"
            )
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("gate artifact lacks top-level gate object")
    gate_for_hash = dict(gate)
    declared_contract_sha256 = gate_for_hash.pop("gate_contract_sha256", None)
    if declared_contract_sha256 != canonical_hash(gate_for_hash):
        raise ValueError("gate contract hash does not match the gate object")
    if gate.get("score_name") != "absolute_nla_centered_cosine":
        raise ValueError(f"forbidden gate score: {gate.get('score_name')!r}")
    feasible = gate.get("feasible")
    if not isinstance(feasible, bool):
        raise ValueError("gate feasible flag must be boolean")
    expected_status = "FEASIBLE" if feasible else "GATE TRAINING FAILURE"
    if gate.get("status") != expected_status:
        raise ValueError(
            f"gate status {gate.get('status')!r} contradicts feasible={feasible}"
        )
    discovery_channel_active = gate.get("discovery_channel_active")
    if not isinstance(discovery_channel_active, bool):
        raise ValueError("gate discovery_channel_active must be boolean")
    return {
        "sha256": gate_sha256,
        "discovery_channel_active": discovery_channel_active,
    }


def resolve_layers(model: torch.nn.Module):
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
    ):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def weight_manifest(model_dir: Path) -> dict[str, str]:
    for name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        path = model_dir / name
        if path.exists():
            return {"path": name, "sha256": sha256_file(path)}
    raise ValueError(f"no weight manifest under {model_dir}")


def verify_base_files_from_full_manifest(
    manifest_path: Path, base_model: Path
) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(
                f"malformed full model manifest at line {line_number}"
            )
        entries[fields[1].strip()] = fields[0].lower()
    base_prefix = str(base_model).rstrip("/") + "/"
    base_entries = {
        path: digest
        for path, digest in entries.items()
        if path.startswith(base_prefix)
    }
    if len(base_entries) != 7:
        raise ValueError(
            f"full manifest must bind 7 base files, found {len(base_entries)}"
        )
    observed: dict[str, str] = {}
    for path, expected in sorted(base_entries.items()):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"base-model file hash mismatch: {path}: {actual} != {expected}"
            )
        observed[path] = actual
    return observed


def load_split(path: Path, split: str) -> dict[str, Any]:
    table = pq.read_table(path)
    schema_metadata = {
        key.decode("utf-8", errors="strict"): value.decode(
            "utf-8", errors="strict"
        )
        for key, value in (table.schema.metadata or {}).items()
    }
    required = {
        "activation_vector",
        "row_uid",
        "content_group_id",
        "split",
        "doc_id",
        "position",
        "input_ids",
        "token",
        "token_id",
        "corpus",
        "source",
        "lang",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"activation parquet missing columns: {missing}")
    splits = np.asarray(table["split"].to_pylist(), dtype=object)
    indices = np.flatnonzero(splits == split)
    if len(indices) == 0:
        raise ValueError(f"activation parquet contains no {split!r} rows")
    part = table.take(indices)

    def values(name: str, default=None):
        if name not in part.column_names:
            return [default] * part.num_rows
        return part[name].to_pylist()

    vectors = np.asarray(
        part["activation_vector"].combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    data = {
        "x": vectors,
        "row_uid": [str(x) for x in values("row_uid")],
        "content_group_id": [str(x) for x in values("content_group_id")],
        "split": [str(x) for x in values("split")],
        "doc_id": np.asarray(values("doc_id"), dtype=np.int64),
        "orig_index": np.asarray(values("orig_index", -1), dtype=np.int64),
        "passage_id": np.asarray(values("passage_id", -1), dtype=np.int64),
        "position": np.asarray(values("position"), dtype=np.int64),
        "input_ids": [
            np.asarray(x, dtype=np.int64) for x in values("input_ids")
        ],
        "token": values("token"),
        "token_id": values("token_id"),
        "corpus": values("corpus"),
        "source": values("source"),
        "lang": values("lang"),
        "activation_metadata": schema_metadata,
    }
    if len(set(data["row_uid"])) != len(indices):
        raise ValueError("row_uid is not unique within split")
    if len(set(data["content_group_id"])) != len(indices):
        raise ValueError("N5 requires one row per independent content group")
    if not np.isfinite(vectors).all():
        raise ValueError("frozen activations contain non-finite values")
    return data


def resolve_key(archive, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in archive.files:
            return candidate
    raise KeyError(
        f"reconstruction archive lacks {label}; tried {candidates}, "
        f"available={archive.files}"
    )


def load_recon(
    path: Path,
    data: dict[str, Any],
    expected_provenance: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        required_identity = {"x", "row_uids", "doc_ids", "positions"}
        missing = sorted(required_identity - set(archive.files))
        if missing:
            raise ValueError(f"reconstruction archive lacks identity keys: {missing}")
        for key, expected in expected_provenance.items():
            require_npz_scalar_string(archive, key, expected)
        archive_x = np.asarray(archive["x"], dtype=np.float32)
        row_uids = [str(x) for x in np.asarray(archive["row_uids"]).tolist()]
        doc_ids = np.asarray(archive["doc_ids"], dtype=np.int64)
        positions = np.asarray(archive["positions"], dtype=np.int64)
        if archive_x.shape != data["x"].shape:
            raise ValueError(
                f"reconstruction x shape {archive_x.shape} != {data['x'].shape}"
            )
        if not np.array_equal(archive_x, data["x"]):
            error = float(np.max(np.abs(archive_x - data["x"])))
            raise ValueError(f"reconstruction x not bit-exact (max_abs={error})")
        if row_uids != data["row_uid"]:
            raise ValueError("reconstruction row_uids do not match activation split")
        if not np.array_equal(doc_ids, data["doc_id"]):
            raise ValueError("reconstruction doc_ids do not match activation split")
        if not np.array_equal(positions, data["position"]):
            raise ValueError("reconstruction positions do not match activation split")

        key_map = {
            "orig": resolve_key(archive, ("pred_orig", "pred_nla_orig"), "orig"),
            "sae_small": resolve_key(
                archive, ("recon_sae_small",), "SAE-small"
            ),
            "sae_big": resolve_key(archive, ("recon_sae_big",), "SAE-big"),
        }
        optional = {
            "p3_only": ("pred_p3_only",),
            "p12": ("pred_p12",),
            "quote_strip_p3": ("pred_quote_strip_p3",),
        }
        for condition, candidates in optional.items():
            for candidate in candidates:
                if candidate in archive.files:
                    key_map[condition] = candidate
                    break
        present = set(key_map)
        channel_present = {"p3_only", "p12", "quote_strip_p3"} <= present
        channel_partial = bool(
            present & {"p3_only", "p12", "quote_strip_p3"}
        ) and not channel_present
        if channel_partial:
            raise ValueError("partial paragraph-channel reconstruction archive")

        reconstructed = {
            name: np.asarray(archive[key], dtype=np.float32)
            for name, key in key_map.items()
        }
        for name, vectors in reconstructed.items():
            if vectors.shape != data["x"].shape or not np.isfinite(vectors).all():
                raise ValueError(f"invalid {name} reconstruction array")
    return reconstructed, {
        "npz_key_map": key_map,
        "channel_present": channel_present,
        "embedded_provenance_verified": dict(expected_provenance),
    }


def norm_match(source: np.ndarray, target: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source_norm = np.linalg.norm(source.astype(np.float64), axis=1)
    target_norm = np.linalg.norm(target.astype(np.float64), axis=1)
    if not np.isfinite(source).all() or not np.isfinite(source_norm).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(source_norm <= 1e-12):
        bad = np.flatnonzero(source_norm <= 1e-12)[:10].tolist()
        raise ValueError(f"{name} contains zero-norm rows: {bad}")
    return (source * (target_norm / source_norm)[:, None]).astype(np.float32)


@torch.inference_mode()
def clean_forward(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    ids: torch.Tensor,
    position: int,
    horizon: int,
) -> tuple[torch.Tensor, np.ndarray]:
    if ids.shape[0] != 1:
        raise ValueError("clean forward requires batch size one")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["x"] = hidden[0, position, :].detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    try:
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
    finally:
        handle.remove()
    window = output.logits[0, position : position + horizon].float().clone()
    observed = captured["x"].numpy().astype(np.float32, copy=False)
    del output
    return window, observed


@torch.inference_mode()
def patched_forward(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    ids: torch.Tensor,
    position: int,
    vector: np.ndarray,
    horizon: int,
) -> torch.Tensor:
    patch = torch.as_tensor(vector, dtype=torch.float32, device=ids.device)

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        hidden[0, position, :] = patch.to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    handle = layer.register_forward_hook(hook)
    try:
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
    finally:
        handle.remove()
    window = output.logits[0, position : position + horizon].float().clone()
    del output, patch
    return window


def clean_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ce = -log_probs.gather(1, targets[:, None]).mean()
    return {"log_probs": log_probs, "probs": probs, "ce": float(ce)}


def score(
    clean: dict[str, Any],
    patched_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    patched_log_probs = torch.log_softmax(patched_logits, dim=-1)
    kl = (
        clean["probs"] * (clean["log_probs"] - patched_log_probs)
    ).sum(dim=-1)
    ce = -patched_log_probs.gather(1, targets[:, None]).mean()
    return {
        "kl_at_pos": float(kl[0]),
        "kl_mean_first16": float(kl.mean()),
        "ce_first16": float(ce),
        "n_positions": int(len(kl)),
    }


def load_checkpoint(
    path: Path, expected_contract: str, expected_uids: set[str]
) -> tuple[dict[str, dict], int]:
    rows: dict[str, dict] = {}
    forwards = 0
    if not path.exists():
        return rows, forwards
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("contract_sha256") != expected_contract:
            raise ValueError(f"checkpoint contract mismatch at line {line_number}")
        row = record["row"]
        uid = str(row["row_uid"])
        if uid not in expected_uids:
            raise ValueError(f"checkpoint has unexpected row_uid {uid}")
        if uid in rows:
            raise ValueError(f"duplicate checkpoint row_uid {uid}")
        rows[uid] = row
        forwards += int(record["n_forwards"])
    return rows, forwards


def append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def summarize(rows: list[dict], conditions: list[str]) -> dict:
    output = {}
    clean_ce = np.asarray([row["ce_clean_first16"] for row in rows], float)
    zero_pos = np.asarray(
        [row["results"]["zero"]["kl_at_pos"] for row in rows], float
    )
    zero_16 = np.asarray(
        [row["results"]["zero"]["kl_mean_first16"] for row in rows], float
    )
    for condition in conditions:
        k = np.asarray(
            [row["results"][condition]["kl_at_pos"] for row in rows], float
        )
        k16 = np.asarray(
            [row["results"][condition]["kl_mean_first16"] for row in rows],
            float,
        )
        ce = np.asarray(
            [row["results"][condition]["ce_first16"] for row in rows], float
        )
        output[condition] = {
            "n": len(rows),
            "kl_at_pos_mean": float(k.mean()),
            "kl_at_pos_median": float(np.median(k)),
            "kl_at_pos_p95": float(np.percentile(k, 95)),
            "kl_at_pos_max": float(k.max()),
            "kl16_mean": float(k16.mean()),
            "ce16_mean": float(ce.mean()),
            "ce16_delta_from_clean_mean": float((ce - clean_ce).mean()),
            "ratio_of_sums_recovered_at_pos": (
                float(1.0 - k.sum() / zero_pos.sum())
                if zero_pos.sum() > 1e-12
                else None
            ),
            "ratio_of_sums_recovered_kl16": (
                float(1.0 - k16.sum() / zero_16.sum())
                if zero_16.sum() > 1e-12
                else None
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--recon", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--split", choices=("discovery", "heldout"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layer-index", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--identity-kl-tol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.horizon != 16:
        raise ValueError("N5 preregistration freezes horizon=16")
    if args.layer_index != 32:
        raise ValueError("N5 preregistration freezes layer_index=32")
    if args.dtype != "bfloat16":
        raise ValueError("N5 preregistration freezes dtype=bfloat16")
    if abs(args.identity_kl_tol - 1e-5) > 1e-15:
        raise ValueError("N5 preregistration freezes identity_kl_tol=1e-5")
    if args.split == "discovery" and args.gate is not None:
        raise ValueError("discovery causal patching must precede and cannot read a gate")
    if args.split == "heldout" and args.gate is None:
        raise ValueError("heldout causal patching requires the frozen --gate artifact")
    if not hasattr(torch, args.dtype):
        raise ValueError(f"torch has no dtype {args.dtype!r}")
    if args.out.exists():
        raise ValueError(f"refusing to overwrite frozen result {args.out}")

    started = time.time()
    activation_sha = sha256_file(args.activations)
    recon_sha = sha256_file(args.recon)
    plan_sha = sha256_file(args.plan)
    model_manifest_sha = sha256_file(args.model_manifest)
    prereg_sha = sha256_file(args.prereg)
    script_sha = sha256_file(__file__)
    if prereg_sha != EXPECTED_PREREG_SHA256:
        raise ValueError(f"unexpected preregistration SHA-256 {prereg_sha}")
    if model_manifest_sha != EXPECTED_MODEL_MANIFEST_SHA256:
        raise ValueError(
            f"unexpected full model manifest SHA-256 {model_manifest_sha}"
        )
    verify_sha256_sidecar(args.activations, activation_sha)
    verify_sha256_sidecar(args.recon, recon_sha)
    verify_sha256_sidecar(args.plan, plan_sha)
    verify_sha256_sidecar(args.model_manifest, model_manifest_sha)
    verify_sha256_sidecar(args.prereg, prereg_sha)
    gate_info = None
    if args.gate is not None:
        gate_info = validate_gate_file(
            args.gate,
            prereg_sha,
            plan_sha,
            model_manifest_sha,
        )
    gate_sha = "" if gate_info is None else str(gate_info["sha256"])
    verified_base_files = verify_base_files_from_full_manifest(
        args.model_manifest, args.base_model
    )
    data = load_split(args.activations, args.split)
    extraction_metadata = data["activation_metadata"]
    required_activation_metadata = {
        f"{ACTIVATION_METADATA_PREFIX}plan_sha256": plan_sha,
        f"{ACTIVATION_METADATA_PREFIX}preregistration_sha256": prereg_sha,
        f"{ACTIVATION_METADATA_PREFIX}model_manifest_sha256": (
            model_manifest_sha
        ),
        f"{ACTIVATION_METADATA_PREFIX}split": args.split,
        f"{ACTIVATION_METADATA_PREFIX}layer_index": "32",
        f"{ACTIVATION_METADATA_PREFIX}dtype": "bfloat16",
        f"{ACTIVATION_METADATA_PREFIX}batch_size": "1",
        f"{ACTIVATION_METADATA_PREFIX}full_frozen_sequence": "true",
    }
    for key, expected in required_activation_metadata.items():
        if extraction_metadata.get(key) != expected:
            raise ValueError(
                f"activation metadata {key}={extraction_metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_base_model_load":
        raise ValueError("cohort plan is not frozen before model load")
    if plan.get("preregistration_sha256") != prereg_sha:
        raise ValueError("cohort plan/preregistration hash mismatch")
    all_plan_rows = plan.get("rows")
    if not isinstance(all_plan_rows, list) or len(all_plan_rows) != 600:
        raise ValueError("cohort plan must contain exactly 600 rows")
    plan_rows = [
        row for row in all_plan_rows if row.get("split") == args.split
    ]
    expected_n = 200 if args.split == "discovery" else 400
    if len(plan_rows) != expected_n:
        raise ValueError(
            f"cohort plan {args.split} requires {expected_n} rows"
        )
    if [str(row["row_uid"]) for row in plan_rows] != data["row_uid"]:
        raise ValueError(
            "activation row order differs from the frozen cohort plan"
        )
    for index, planned in enumerate(plan_rows):
        for field in (
            "content_group_id",
            "doc_id",
            "position",
            "corpus",
            "source",
            "lang",
        ):
            observed = (
                data[field][index]
                if field in data
                else None
            )
            if str(observed) != str(planned.get(field)):
                raise ValueError(
                    f"activation row {index} differs from plan at {field}"
                )
    expected_npz_provenance = {
        "activations_sha256": activation_sha,
        "plan_sha256": plan_sha,
        "model_manifest_sha256": model_manifest_sha,
        "prereg_sha256": prereg_sha,
        "gate_sha256": gate_sha,
    }
    reconstructed, recon_qa = load_recon(
        args.recon,
        data,
        expected_npz_provenance,
    )
    if (
        gate_info is not None
        and not bool(gate_info["discovery_channel_active"])
        and bool(recon_qa["channel_present"])
    ):
        raise ValueError(
            "heldout reconstruction contains a paragraph channel after the "
            "frozen discovery gate globally aborted H5-B"
        )
    n, width = data["x"].shape
    if n != expected_n:
        raise ValueError(
            f"N5 {args.split} requires {expected_n} rows, found {n}"
        )

    source_vectors = {
        "identity": data["x"],
        **reconstructed,
    }
    substitutes = {
        name: norm_match(vectors, data["x"], name)
        for name, vectors in source_vectors.items()
    }
    substitutes["zero"] = np.zeros_like(data["x"])
    conditions = ["identity", "orig", "sae_small", "sae_big"]
    if recon_qa["channel_present"]:
        conditions += ["p3_only", "p12", "quote_strip_p3"]
    conditions += ["zero"]

    for index, (position, ids) in enumerate(
        zip(data["position"], data["input_ids"])
    ):
        if position < 0 or position + args.horizon >= len(ids):
            raise ValueError(
                f"row {index} lacks frozen horizon: p={position}, len={len(ids)}"
            )

    config = args.base_model / "config.json"
    if not config.exists():
        raise ValueError(f"missing base config {config}")
    weights = weight_manifest(args.base_model)
    contract = {
        "activation_sha256": activation_sha,
        "recon_sha256": recon_sha,
        "plan_sha256": plan_sha,
        "model_manifest_sha256": model_manifest_sha,
        "prereg_sha256": prereg_sha,
        "gate_sha256": gate_sha,
        "script_sha256": script_sha,
        "base_config_sha256": sha256_file(config),
        "base_weight_manifest": weights,
        "split": args.split,
        "layer_index": args.layer_index,
        "horizon": args.horizon,
        "dtype": args.dtype,
        "conditions": conditions,
        "row_uids_sha256": canonical_hash(data["row_uid"]),
    }
    contract_sha = canonical_hash(contract)
    checkpoint_rows, forward_count = load_checkpoint(
        args.checkpoint, contract_sha, set(data["row_uid"])
    )

    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device
    layers = resolve_layers(model)
    if not 0 <= args.layer_index < len(layers):
        raise ValueError("layer index outside model")
    layer = layers[args.layer_index]
    model_load_seconds = time.time() - load_started

    output_rows: list[dict | None] = [
        checkpoint_rows.get(uid) for uid in data["row_uid"]
    ]
    provenance_max = max(
        [
            float(row["provenance"]["max_abs"])
            for row in checkpoint_rows.values()
        ],
        default=0.0,
    )
    provenance_exact = all(
        bool(row["provenance"]["exact"])
        for row in checkpoint_rows.values()
    )
    forward_started = time.time()
    for index, uid in enumerate(data["row_uid"]):
        if output_rows[index] is not None:
            print(
                f"[{args.split} {index + 1}/{n}] {uid} checkpoint",
                flush=True,
            )
            continue
        ids_np = data["input_ids"][index]
        position = int(data["position"][index])
        ids = torch.as_tensor(ids_np[None, :], dtype=torch.long, device=device)
        clean_logits, observed = clean_forward(
            model, layer, ids, position, args.horizon
        )
        forward_count += 1
        frozen = data["x"][index]
        difference = np.abs(observed - frozen)
        max_abs = float(difference.max())
        exact = bool(np.array_equal(observed, frozen))
        provenance_max = max(provenance_max, max_abs)
        provenance_exact = provenance_exact and exact
        if not exact:
            raise RuntimeError(
                f"row {uid} clean activation differs from frozen x: "
                f"max_abs={max_abs}"
            )

        targets = ids[0, position + 1 : position + 1 + args.horizon]
        clean = clean_metrics(clean_logits, targets)
        results: dict[str, dict] = {}
        for condition in conditions:
            patched_logits = patched_forward(
                model,
                layer,
                ids,
                position,
                substitutes[condition][index],
                args.horizon,
            )
            forward_count += 1
            results[condition] = score(clean, patched_logits, targets)
            del patched_logits

        row = {
            "idx_split": index,
            "row_uid": uid,
            "content_group_id": data["content_group_id"][index],
            "split": args.split,
            "doc_id": int(data["doc_id"][index]),
            "orig_index": int(data["orig_index"][index]),
            "passage_id": int(data["passage_id"][index]),
            "position": position,
            "token": data["token"][index],
            "token_id": int(data["token_id"][index]),
            "corpus": data["corpus"][index],
            "source": data["source"][index],
            "lang": data["lang"][index],
            "seq_len": int(len(ids_np)),
            "x_norm": float(np.linalg.norm(frozen)),
            "provenance": {"exact": exact, "max_abs": max_abs},
            "ce_clean_first16": float(clean["ce"]),
            "results": results,
        }
        output_rows[index] = row
        append_checkpoint(
            args.checkpoint,
            {
                "contract_sha256": contract_sha,
                "n_forwards": 1 + len(conditions),
                "row": row,
            },
        )
        elapsed = time.time() - forward_started
        completed = sum(row is not None for row in output_rows)
        eta = elapsed / max(completed - len(checkpoint_rows), 1) * (n - completed)
        print(
            f"[{args.split} {completed}/{n}] {uid} "
            f"orig={results['orig']['kl_at_pos']:.4g} "
            f"big={results['sae_big']['kl_at_pos']:.4g} "
            f"zero={results['zero']['kl_at_pos']:.4g} "
            f"eta={eta / 60:.1f}m",
            flush=True,
        )
        del ids, clean_logits, clean, results

    rows = [row for row in output_rows if row is not None]
    if len(rows) != n:
        raise RuntimeError(f"produced {len(rows)}/{n} rows")
    summary = summarize(rows, conditions)
    identity_abs_max = max(
        abs(float(row["results"]["identity"]["kl_at_pos"])) for row in rows
    )
    identity16_abs_max = max(
        abs(float(row["results"]["identity"]["kl_mean_first16"]))
        for row in rows
    )
    if max(identity_abs_max, identity16_abs_max) > args.identity_kl_tol:
        raise RuntimeError(
            "identity KL QA failed: "
            f"pos={identity_abs_max}, kl16={identity16_abs_max}"
        )

    payload = {
        "schema_version": 1,
        "experiment": "N5 frozen full-sequence causal patch",
        "status": "complete",
        "split": args.split,
        "protocol": {
            "layer_index": args.layer_index,
            "horizon": args.horizon,
            "batch_size": 1,
            "full_sequence": True,
            "norm_matching": True,
            "kl_direction": "KL(clean || patched)",
            "conditions": conditions,
        },
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": activation_sha,
            "recon": str(args.recon),
            "recon_sha256": recon_sha,
            "plan": str(args.plan),
            "plan_sha256": plan_sha,
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha,
            "gate": None if args.gate is None else str(args.gate),
            "gate_sha256": gate_sha,
            "base_model": str(args.base_model),
            "base_config_sha256": contract["base_config_sha256"],
            "base_weight_manifest": weights,
            "verified_base_files_from_full_manifest": verified_base_files,
            "activation_metadata": extraction_metadata,
            "script_sha256": script_sha,
            "contract_sha256": contract_sha,
            **recon_qa,
        },
        "cohort": {
            "n_rows": n,
            "n_content_groups": len(set(data["content_group_id"])),
            "by_corpus": {
                corpus: sum(x == corpus for x in data["corpus"])
                for corpus in sorted(set(data["corpus"]))
            },
        },
        "qa": {
            "provenance_all_bit_exact": provenance_exact,
            "provenance_max_abs": provenance_max,
            "identity_kl_at_pos_abs_max": identity_abs_max,
            "identity_kl16_abs_max": identity16_abs_max,
            "identity_kl_tolerance": args.identity_kl_tol,
        },
        "summary": summary,
        "rows": rows,
        "n_forwards": forward_count,
        "model_load_seconds": model_load_seconds,
        "forward_seconds": time.time() - forward_started,
        "elapsed_seconds": time.time() - started,
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
        f"N5_CAUSAL_COMPLETE split={args.split} rows={n} "
        f"forwards={forward_count} elapsed={payload['elapsed_seconds']:.1f}s "
        f"sha256={output_sha} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
