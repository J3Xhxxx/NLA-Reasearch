#!/usr/bin/env python3
"""N5 split-wise NLA/SAE reconstruction with a frozen held-out router.

This stage is deliberately separate from N4's frozen implementation.  It:

* consumes one split of the frozen 600-row N5 activation parquet;
* generates only the original AV explanation, greedily, with an append-only
  checkpoint and freezes every explanation before loading the AR;
* enables the three H5-B variants only when every relevant explanation has at
  least three blank-line-separated paragraphs;
* reconstructs NLA-orig, SAE-small and SAE-big for H5-A, plus the H5-B
  variants when the channel audit passes;
* derives ``m_hat`` only from discovery activations;
* requires held-out to read the already-frozen discovery mean and threshold
  from ``--gate``.  It never estimates any held-out center or threshold.

The NPZ is the vector source of truth.  The JSON contains row identities,
scores, routing decisions, hashes, and an explicit JSON/NPZ semantic contract.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from pilot_common import AVLocal, JumpReLUSAE, NLACritic


EXPECTED_SPLIT_COUNTS = {"discovery": 200, "heldout": 400}
EXPECTED_PREREG_SHA256 = (
    "63dc31b4f9607e54ac15f1c364fcae2ee903f228fe0afb4d388c6dad1a6f9103"
)
EXPECTED_MODEL_MANIFEST_SHA256 = (
    "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735"
)
ACTIVATION_METADATA_PREFIX = "nla.n5_activation_extraction."
QUOTE_RE = re.compile(r'"[^"]*"')
QUOTE_PLACEHOLDER = '"[...]"'
CHANNEL_ACTIVE = "ACTIVE"
CHANNEL_ABORTED = "ABORTED_FEWER_THAN_3_PARAGRAPHS"


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


def parse_frozen_model_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(
                f"malformed frozen model manifest at line {line_number}"
            )
        entries[fields[1].strip()] = fields[0].lower()
    if len(entries) != 25:
        raise ValueError("frozen model manifest must contain 25 unique files")
    return entries


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_frozen_json(path: Path, value: Any) -> str:
    """Create a canonical JSON artifact once; require exact identity on resume."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen artifact differs on resume: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def write_new_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def model_file_manifest(root: Path) -> dict[str, Any]:
    """Hash the actual model weights plus small configuration/tokenizer files."""
    if not root.is_dir():
        raise ValueError(f"model directory does not exist: {root}")
    files: set[Path] = set()
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "nla_meta.yaml",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
        "value_head.safetensors",
        "params.safetensors",
    ):
        candidate = root / name
        if candidate.is_file():
            files.add(candidate)

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for relative in set(index.get("weight_map", {}).values()):
            shard = root / relative
            if not shard.is_file():
                raise ValueError(f"weight index references missing shard: {shard}")
            files.add(shard)

    weight_files = [
        path
        for path in files
        if path.suffix in {".safetensors", ".bin"}
    ]
    if not weight_files:
        raise ValueError(f"no model weight files found under {root}")
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
    }
    return {
        "root": str(root),
        "files": hashes,
        "manifest_sha256": canonical_sha256(hashes),
    }


def validate_actual_reconstruction_models(
    actual: dict[str, dict[str, Any]], frozen: dict[str, str]
) -> None:
    flattened: dict[str, str] = {}
    for label, item in actual.items():
        root = str(item["root"]).rstrip("/")
        files = item.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"actual model manifest {label} lacks files")
        for relative, digest in files.items():
            flattened[f"{root}/{relative}"] = str(digest).lower()
    expected = {
        path: digest
        for path, digest in frozen.items()
        if "/gemma-3-12b-it/" not in path
    }
    missing_or_wrong = {
        path: {"expected": digest, "actual": flattened.get(path)}
        for path, digest in expected.items()
        if flattened.get(path) != digest
    }
    if missing_or_wrong:
        raise ValueError(
            "actual AV/AR/SAE model files differ from the frozen 25-file "
            f"manifest: {missing_or_wrong}"
        )


def paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def build_channel_variants(explanations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in explanations:
        parts = paragraphs(str(row["explanation"]))
        if len(parts) < 3:
            raise ValueError("channel variants requested after paragraph audit failed")
        p3 = parts[-1]
        p12 = "\n\n".join(parts[:-1])
        quote_strip = "\n\n".join(
            [*parts[:-1], QUOTE_RE.sub(QUOTE_PLACEHOLDER, p3)]
        )
        rows.append(
            {
                "idx": int(row["idx"]),
                "global_idx": int(row["global_idx"]),
                "row_uid": str(row["row_uid"]),
                "variants": {
                    "p3_only": p3,
                    "p12": p12,
                    "quote_strip_p3": quote_strip,
                },
            }
        )
    return rows


def table_metadata(table) -> dict[str, str]:
    metadata = table.schema.metadata or {}
    return {
        key.decode("utf-8", errors="replace"): value.decode(
            "utf-8", errors="replace"
        )
        for key, value in metadata.items()
    }


def find_manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    for path in (
        ("rows",),
        ("positions",),
        ("selected_rows",),
        ("cohort", "rows"),
        ("manifest", "rows"),
    ):
        value: Any = manifest
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
    return None


def nested_value(mapping: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def validate_plan(
    plan_path: Path,
    prereg_sha256: str,
    activation_row_uids: list[str],
    split: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    manifest = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha256 = sha256_file(plan_path)
    declared_prereg = nested_value(
        manifest,
        (
            ("prereg_sha256",),
            ("preregistration_sha256",),
            ("inputs", "prereg_sha256"),
            ("inputs", "preregistration_sha256"),
        ),
    )
    if declared_prereg is None:
        raise ValueError("cohort manifest does not bind a preregistration SHA-256")
    if str(declared_prereg) != prereg_sha256:
        raise ValueError(
            "cohort manifest preregistration hash mismatch: "
            f"{declared_prereg} != {prereg_sha256}"
        )

    all_rows = find_manifest_rows(manifest)
    if all_rows is None or len(all_rows) != 600:
        raise ValueError("cohort plan must contain exactly 600 inspectable rows")
    rows = [row for row in all_rows if row.get("split") == split]
    expected = EXPECTED_SPLIT_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(
            f"cohort plan {split} requires {expected} rows, found {len(rows)}"
        )
    manifest_uids = [str(row.get("row_uid")) for row in rows]
    if any(uid in {"", "None"} for uid in manifest_uids):
        raise ValueError("cohort manifest row lacks row_uid")
    if manifest_uids != activation_row_uids:
        raise ValueError("activation parquet row_uids do not match manifest order")
    closure = {
        "plan_total_row_count": len(all_rows),
        "plan_split_row_count": len(rows),
        "plan_row_uids_match_activation_in_order": True,
        "plan_row_uid_sequence_sha256": canonical_sha256(manifest_uids),
        "plan_prereg_sha256_matches": True,
    }
    return manifest, plan_sha256, closure


def load_activation_subset(
    path: Path, split: str, limit: int
) -> tuple[np.ndarray, dict[str, list[Any]], np.ndarray, dict[str, Any]]:
    table = pq.read_table(path)
    required = {
        "row_uid",
        "content_group_id",
        "split",
        "doc_id",
        "orig_index",
        "passage_id",
        "activation_vector",
        "position",
        "input_ids",
        "token",
        "token_id",
        "corpus",
        "source",
        "lang",
        "seq_len",
        "context_tail",
        "continuation",
        "norm",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"activation parquet lacks required columns: {missing}")

    all_splits = [str(value) for value in table.column("split").to_pylist()]
    all_uids = [str(value) for value in table.column("row_uid").to_pylist()]
    all_groups = [
        str(value) for value in table.column("content_group_id").to_pylist()
    ]
    if len(all_uids) != len(set(all_uids)):
        raise ValueError("activation parquet row_uid values are not unique")
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("activation parquet content_group_id values are not unique")
    expected = EXPECTED_SPLIT_COUNTS[split]
    if not limit:
        if table.num_rows != expected:
            raise ValueError(
                f"N5 {split} activation parquet must have {expected} rows, "
                f"found {table.num_rows}"
            )
        counts = Counter(all_splits)
        if counts != Counter({split: expected}):
            raise ValueError(
                f"N5 activation split differs from invocation: {dict(counts)}"
            )
    elif limit < 0:
        raise ValueError("--limit cannot be negative")

    global_indices = np.asarray(
        [index for index, value in enumerate(all_splits) if value == split],
        dtype=np.int64,
    )
    if not limit and len(global_indices) != expected:
        raise ValueError(f"{split} must contain {expected} rows")
    if limit:
        global_indices = global_indices[:limit]
    if len(global_indices) == 0:
        raise ValueError(f"activation parquet contains no selected rows for {split}")

    vectors_all = np.asarray(
        table.column("activation_vector").combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    x = vectors_all[global_indices]
    if x.ndim != 2 or x.shape[1] != 3840:
        raise ValueError(f"expected activation shape [n,3840], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("activation vectors contain non-finite values")

    metadata_columns = sorted(required - {"activation_vector"})
    meta = {
        name: [table.column(name)[int(index)].as_py() for index in global_indices]
        for name in metadata_columns
    }
    for local_index, (position, ids, seq_len, token_id) in enumerate(
        zip(meta["position"], meta["input_ids"], meta["seq_len"], meta["token_id"])
    ):
        position = int(position)
        if int(seq_len) != len(ids):
            raise ValueError(f"row {local_index} seq_len does not match input_ids")
        if not 0 <= position < len(ids):
            raise ValueError(f"row {local_index} has out-of-range position")
        if int(ids[position]) != int(token_id):
            raise ValueError(f"row {local_index} token_id does not match input_ids")
        if position + 16 >= len(ids):
            raise ValueError(f"row {local_index} lacks the frozen 16-token horizon")

    qa = {
        "activation_table_metadata": table_metadata(table),
        "activation_total_rows": table.num_rows,
        "activation_split_counts": dict(Counter(all_splits)),
        "all_row_uids": all_uids,
        "selected_row_uid_sequence_sha256": canonical_sha256(
            [str(uid) for uid in meta["row_uid"]]
        ),
        "selected_content_groups_unique": (
            len(meta["content_group_id"]) == len(set(meta["content_group_id"]))
        ),
    }
    return x, meta, global_indices, qa


def load_generation_checkpoint(
    path: Path,
    contract_sha256: str,
    row_uids: list[str],
    global_indices: np.ndarray,
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("contract_sha256") != contract_sha256:
            raise ValueError(f"checkpoint contract mismatch at line {line_number}")
        index = int(row["idx"])
        if not 0 <= index < len(row_uids):
            raise ValueError(f"checkpoint idx out of range at line {line_number}")
        if index in completed:
            raise ValueError(f"duplicate checkpoint idx {index}")
        if str(row.get("row_uid")) != row_uids[index]:
            raise ValueError(f"checkpoint row_uid mismatch at idx {index}")
        if int(row.get("global_idx")) != int(global_indices[index]):
            raise ValueError(f"checkpoint global_idx mismatch at idx {index}")
        completed[index] = row
    return completed


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize a zero/non-finite mean direction")
    return vector / norm


def centered_cosine(
    prediction: np.ndarray, target: np.ndarray, mean_direction: np.ndarray
) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64)
    m = np.asarray(mean_direction, dtype=np.float64)
    pc = p - np.outer(p @ m, m)
    xc = x - np.outer(x @ m, m)
    denominator = np.linalg.norm(pc, axis=1) * np.linalg.norm(xc, axis=1)
    return np.sum(pc * xc, axis=1) / np.maximum(denominator, 1e-30)


def raw_cosine(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64)
    denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(x, axis=1)
    return np.sum(p * x, axis=1) / np.maximum(denominator, 1e-30)


def retrieval(
    prediction: np.ndarray, target: np.ndarray, mean_direction: np.ndarray
) -> dict[str, float]:
    p = np.asarray(prediction, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64)
    m = np.asarray(mean_direction, dtype=np.float64)
    p = p - np.outer(p @ m, m)
    x = x - np.outer(x @ m, m)
    p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-30)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    similarity = p @ x.T
    diagonal = np.diag(similarity)
    ranks = 1 + (similarity > diagonal[:, None] + 1e-12).sum(axis=1)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def prediction_scores(
    prediction: np.ndarray, target: np.ndarray, mean_direction: np.ndarray
) -> dict[str, np.ndarray]:
    errors = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    target_norm = np.linalg.norm(np.asarray(target, np.float64), axis=1)
    prediction_norm = np.linalg.norm(np.asarray(prediction, np.float64), axis=1)
    return {
        "cos_c": centered_cosine(prediction, target, mean_direction),
        "cos_raw": raw_cosine(prediction, target),
        "l2_error": np.linalg.norm(errors, axis=1),
        "pred_norm": prediction_norm,
        "target_norm": target_norm,
        "norm_ratio": prediction_norm / np.maximum(target_norm, 1e-30),
    }


def validate_gate(
    path: Path,
    width: int,
    prereg_sha256: str,
    plan_sha256: str,
    model_manifest_sha256: str,
) -> dict[str, Any]:
    gate_sha256 = sha256_file(path)
    verify_sha256_sidecar(path, gate_sha256)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("split") != "discovery":
        raise ValueError("gate artifact is not a complete discovery freeze")
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("gate artifact lacks top-level gate object")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("gate artifact lacks top-level inputs object")

    required_hashes = {
        "prereg_sha256": prereg_sha256,
        "plan_sha256": plan_sha256,
        "manifest_sha256": model_manifest_sha256,
    }
    for name, expected in required_hashes.items():
        actual = inputs.get(name)
        if actual != expected:
            raise ValueError(
                f"gate {name} mismatch: expected {expected}, found {actual}"
            )

    direction = np.asarray(gate.get("discovery_mean_direction"), dtype=np.float64)
    if direction.shape != (width,) or not np.isfinite(direction).all():
        raise ValueError("gate discovery_mean_direction has invalid shape/values")
    direction_norm = float(np.linalg.norm(direction))
    if abs(direction_norm - 1.0) > 1e-5:
        raise ValueError(
            f"gate discovery_mean_direction is not unit norm: {direction_norm}"
        )

    feasible = bool(gate.get("feasible"))
    status = str(gate.get("status"))
    if feasible and status != "FEASIBLE":
        raise ValueError("feasible gate must have status FEASIBLE")
    if not feasible and status != "GATE TRAINING FAILURE":
        raise ValueError("infeasible gate must have status GATE TRAINING FAILURE")
    threshold = gate.get("threshold")
    tie_cutoff = gate.get("tie_hash_cutoff_inclusive")
    if feasible:
        if threshold is None or not math.isfinite(float(threshold)):
            raise ValueError("feasible gate lacks a finite threshold")
        if not isinstance(tie_cutoff, str) or len(tie_cutoff) != 64:
            raise ValueError("feasible gate lacks a SHA-256 tie cutoff")
    elif threshold is not None:
        raise ValueError("failed gate must have threshold=null")
    elif tie_cutoff is not None:
        raise ValueError("failed gate must have tie cutoff=null")

    discovery_channel_active = gate.get("discovery_channel_active")
    discovery_channel_status = gate.get("discovery_channel_status")
    if not isinstance(discovery_channel_active, bool):
        raise ValueError("gate must freeze discovery_channel_active as a boolean")
    expected_channel_status = (
        CHANNEL_ACTIVE if discovery_channel_active else CHANNEL_ABORTED
    )
    if discovery_channel_status != expected_channel_status:
        raise ValueError("gate discovery channel status/boolean are inconsistent")

    score_name = gate.get("score_name")
    if score_name != "absolute_nla_centered_cosine":
        raise ValueError(f"forbidden gate score: {score_name!r}")
    frozen_rule = str(gate.get("heldout_threshold_rule"))
    if feasible and "sha256(row_uid)" not in frozen_rule:
        raise ValueError("gate does not freeze the required row-UID tie rule")
    gate_for_hash = dict(gate)
    declared_contract_hash = gate_for_hash.pop("gate_contract_sha256", None)
    if declared_contract_hash != canonical_sha256(gate_for_hash):
        raise ValueError("gate contract hash does not match the gate object")

    return {
        "payload": payload,
        "gate": gate,
        "sha256": gate_sha256,
        "m_hat": direction,
        "feasible": feasible,
        "threshold": None if threshold is None else float(threshold),
        "tie_hash_cutoff_inclusive": tie_cutoff,
        "heldout_rule": frozen_rule,
        "discovery_channel_active": discovery_channel_active,
        "discovery_channel_status": discovery_channel_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--av", required=True, type=Path)
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--sae-small", required=True, type=Path)
    parser.add_argument("--sae-big", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("discovery", "heldout"))
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--explanations-out", required=True, type=Path)
    parser.add_argument("--variants-out", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vecs-out", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    if args.max_new_tokens != 200:
        raise ValueError("N5 preregistration fixes --max-new-tokens to 200")
    if args.split == "discovery" and args.gate is not None:
        raise ValueError("discovery reconstruction must precede and cannot read a gate")
    if args.split == "heldout" and args.gate is None:
        raise ValueError("heldout reconstruction requires the frozen --gate artifact")
    for target in (args.out, args.vecs_out):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite completed output: {target}")

    activations_sha256 = sha256_file(args.activations)
    verify_sha256_sidecar(args.activations, activations_sha256)
    plan_sha256 = sha256_file(args.plan)
    verify_sha256_sidecar(args.plan, plan_sha256)
    prereg_sha256 = sha256_file(args.prereg)
    verify_sha256_sidecar(args.prereg, prereg_sha256)
    if prereg_sha256 != EXPECTED_PREREG_SHA256:
        raise ValueError(
            "unexpected N5 preregistration: "
            f"{prereg_sha256} != {EXPECTED_PREREG_SHA256}"
        )
    model_manifest_sha256 = sha256_file(args.model_manifest)
    verify_sha256_sidecar(args.model_manifest, model_manifest_sha256)
    if model_manifest_sha256 != EXPECTED_MODEL_MANIFEST_SHA256:
        raise ValueError(
            "unexpected N5 model manifest: "
            f"{model_manifest_sha256} != {EXPECTED_MODEL_MANIFEST_SHA256}"
        )
    frozen_model_manifest = parse_frozen_model_manifest(args.model_manifest)
    script_sha256 = sha256_file(__file__)
    pilot_common_sha256 = sha256_file(Path(__file__).with_name("pilot_common.py"))
    x, meta, global_indices, activation_qa = load_activation_subset(
        args.activations, args.split, args.limit
    )
    _, validated_plan_sha256, plan_closure = validate_plan(
        args.plan,
        prereg_sha256,
        activation_qa["all_row_uids"],
        args.split,
    )
    if validated_plan_sha256 != plan_sha256:
        raise RuntimeError("cohort plan changed while it was being validated")
    expected_activation_metadata = {
        "plan_sha256": plan_sha256,
        "preregistration_sha256": prereg_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "split": args.split,
        "layer_index": "32",
        "dtype": "bfloat16",
        "batch_size": "1",
        "full_frozen_sequence": "true",
    }
    observed_activation_metadata = activation_qa["activation_table_metadata"]
    for key, expected in expected_activation_metadata.items():
        full_key = f"{ACTIVATION_METADATA_PREFIX}{key}"
        observed = observed_activation_metadata.get(full_key)
        if observed != expected:
            raise ValueError(
                f"activation metadata {full_key} mismatch: "
                f"{observed!r} != {expected!r}"
            )
    n, width = x.shape
    row_uids = [str(value) for value in meta["row_uid"]]

    gate_info = None
    if args.split == "heldout":
        assert args.gate is not None
        gate_info = validate_gate(
            args.gate,
            width,
            prereg_sha256,
            plan_sha256,
            model_manifest_sha256,
        )

    print("[hash] binding run to complete model weight manifests", flush=True)
    model_manifests = {
        "av": model_file_manifest(args.av),
        "ar": model_file_manifest(args.ar),
        "sae_small": model_file_manifest(args.sae_small),
        "sae_big": model_file_manifest(args.sae_big),
    }
    validate_actual_reconstruction_models(
        model_manifests, frozen_model_manifest
    )
    contract = {
        "experiment": "N5 reconstruction",
        "split": args.split,
        "activations_sha256": activations_sha256,
        "plan_sha256": plan_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "prereg_sha256": prereg_sha256,
        "script_sha256": script_sha256,
        "pilot_common_sha256": pilot_common_sha256,
        "actual_model_manifest_sha256": {
            name: value["manifest_sha256"]
            for name, value in model_manifests.items()
        },
        "gate_sha256": None if gate_info is None else gate_info["sha256"],
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "row_uid_sequence_sha256": canonical_sha256(row_uids),
    }
    contract_sha256 = canonical_sha256(contract)

    completed = load_generation_checkpoint(
        args.checkpoint, contract_sha256, row_uids, global_indices
    )
    missing = [index for index in range(n) if index not in completed]
    print(
        f"[input] split={args.split} rows={n} checkpoint={len(completed)} "
        f"missing={len(missing)} activation={activations_sha256}",
        flush=True,
    )

    if missing:
        av = AVLocal(args.av, device="cuda")
        for ordinal, index in enumerate(missing, 1):
            explanation = av.generate(
                x[index], temperature=0.0, max_new_tokens=args.max_new_tokens
            )
            parts = paragraphs(explanation)
            row = {
                "contract_sha256": contract_sha256,
                "idx": int(index),
                "global_idx": int(global_indices[index]),
                "row_uid": row_uids[index],
                "content_group_id": str(meta["content_group_id"][index]),
                "doc_id": int(meta["doc_id"][index]),
                "position": int(meta["position"][index]),
                "token": str(meta["token"][index]),
                "explanation": explanation,
                "paragraph_count": len(parts),
            }
            append_checkpoint(args.checkpoint, row)
            completed[index] = row
            print(
                f"[AV {len(completed):>3}/{n}] split={args.split} "
                f"uid={row_uids[index][:14]} paragraphs={len(parts)} "
                f"chars={len(explanation)}",
                flush=True,
            )
        del av
        gc.collect()
        torch.cuda.empty_cache()

    explanations = [
        {
            key: completed[index][key]
            for key in (
                "idx",
                "global_idx",
                "row_uid",
                "content_group_id",
                "doc_id",
                "position",
                "token",
                "explanation",
                "paragraph_count",
            )
        }
        for index in range(n)
    ]
    paragraph_counts = [int(row["paragraph_count"]) for row in explanations]
    offending = [
        str(row["row_uid"])
        for row in explanations
        if int(row["paragraph_count"]) < 3
    ]
    local_paragraph_pass = not offending
    discovery_channel_active = (
        True if gate_info is None else gate_info["discovery_channel_active"]
    )
    channel_active = local_paragraph_pass and discovery_channel_active
    channel_status = CHANNEL_ACTIVE if channel_active else CHANNEL_ABORTED
    abort_origin = None
    if not discovery_channel_active:
        abort_origin = "discovery"
    elif not local_paragraph_pass:
        abort_origin = args.split

    explanation_payload = {
        "schema_version": 1,
        "experiment": "N5 frozen AV original explanations",
        "split": args.split,
        "status": "COMPLETE_FROZEN_BEFORE_AR",
        "inputs": {
            "activations_sha256": activations_sha256,
            "plan_sha256": plan_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "prereg_sha256": prereg_sha256,
            "script_sha256": script_sha256,
            "contract_sha256": contract_sha256,
            "gate_sha256": None if gate_info is None else gate_info["sha256"],
        },
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "paragraph_audit": {
            "all_at_least_three": local_paragraph_pass,
            "distribution": dict(sorted(Counter(paragraph_counts).items())),
            "offending_row_uids": offending,
        },
        "rows": explanations,
    }
    explanations_sha256 = write_frozen_json(
        args.explanations_out, explanation_payload
    )

    variant_rows = build_channel_variants(explanations) if channel_active else []
    variant_payload = {
        "schema_version": 1,
        "experiment": "N5 frozen H5-B paragraph variants",
        "split": args.split,
        "status": channel_status,
        "abort_origin": abort_origin,
        "explanations_sha256": explanations_sha256,
        "protocol": {
            "p3_only": "final blank-line-separated paragraph",
            "p12": "all paragraphs before the final paragraph",
            "quote_strip_p3": (
                f"replace {QUOTE_RE.pattern!r} matches in p3 with "
                f"{QUOTE_PLACEHOLDER!r}"
            ),
        },
        "offending_row_uids": offending,
        "rows": variant_rows,
    }
    variants_sha256 = write_frozen_json(args.variants_out, variant_payload)
    print(
        f"[freeze] explanations={explanations_sha256} "
        f"variants={variants_sha256} channel={channel_status}",
        flush=True,
    )

    # AR starts only after both explanation and variant artifacts are frozen.
    critic = NLACritic(args.ar, device="cuda")
    ar_cache: dict[str, np.ndarray] = {}

    def reconstruct_text(text: str) -> np.ndarray:
        if text not in ar_cache:
            vector = critic.reconstruct(text).numpy().astype(np.float32)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError("AR produced an invalid reconstruction vector")
            ar_cache[text] = vector
            if len(ar_cache) % 100 == 0:
                print(f"[AR] {len(ar_cache)} unique texts", flush=True)
        return ar_cache[text]

    predictions: dict[str, np.ndarray] = {
        "orig": np.stack(
            [reconstruct_text(str(row["explanation"])) for row in explanations]
        )
    }
    if channel_active:
        for name in ("p3_only", "p12", "quote_strip_p3"):
            predictions[name] = np.stack(
                [
                    reconstruct_text(str(row["variants"][name]))
                    for row in variant_rows
                ]
            )
            print(f"[AR] {name} complete", flush=True)
    del critic
    gc.collect()
    torch.cuda.empty_cache()

    xt = torch.from_numpy(x)
    sae_small = JumpReLUSAE(str(args.sae_small), device="cuda")
    recon_small_t, acts_small = sae_small(xt)
    recon_small = recon_small_t.float().cpu().numpy()
    l0_small = (acts_small > 0).sum(1).cpu().numpy().astype(np.int64)
    del sae_small, recon_small_t, acts_small
    gc.collect()
    torch.cuda.empty_cache()

    sae_big = JumpReLUSAE(str(args.sae_big), device="cuda")
    recon_big_t, acts_big = sae_big(xt)
    recon_big = recon_big_t.float().cpu().numpy()
    l0_big = (acts_big > 0).sum(1).cpu().numpy().astype(np.int64)
    del sae_big, recon_big_t, acts_big, xt
    gc.collect()
    torch.cuda.empty_cache()

    predictions["sae_small"] = recon_small.astype(np.float32)
    predictions["sae_big"] = recon_big.astype(np.float32)
    for name, prediction in predictions.items():
        if prediction.shape != x.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{name} reconstruction has invalid shape/values")

    if args.split == "discovery":
        m_hat = unit(x.mean(axis=0, dtype=np.float64))
        centering_source = "unit mean of discovery x only"
    else:
        assert gate_info is not None
        m_hat = gate_info["m_hat"]
        centering_source = "frozen gate.discovery_mean_direction; no heldout fit"

    scores = {
        name: prediction_scores(prediction, x, m_hat)
        for name, prediction in predictions.items()
    }
    q_router = scores["orig"]["cos_c"].astype(np.float64)
    route_nla = np.zeros(n, dtype=bool)
    routed_to = np.full(n, "unassigned_discovery", dtype="<U24")
    if args.split == "heldout":
        assert gate_info is not None
        if gate_info["feasible"]:
            threshold = float(gate_info["threshold"])
            tie_cutoff = str(gate_info["tie_hash_cutoff_inclusive"])
            row_uid_hashes = np.asarray(
                [
                    hashlib.sha256(uid.encode("utf-8")).hexdigest()
                    for uid in row_uids
                ],
                dtype="<U64",
            )
            route_nla = (q_router > threshold) | (
                (q_router == threshold) & (row_uid_hashes <= tie_cutoff)
            )
        routed_to = np.where(route_nla, "nla", "sae_big")

    summary: dict[str, Any] = {}
    for name, prediction in predictions.items():
        item_scores = scores[name]
        summary[name] = {
            "n": n,
            "mean_cos_c": float(item_scores["cos_c"].mean()),
            "median_cos_c": float(np.median(item_scores["cos_c"])),
            "mean_cos_raw": float(item_scores["cos_raw"].mean()),
            "mean_l2_error": float(item_scores["l2_error"].mean()),
            "retrieval": retrieval(prediction, x, m_hat),
        }
    summary["sae_small"]["mean_l0"] = float(l0_small.mean())
    summary["sae_big"]["mean_l0"] = float(l0_big.mean())

    rows_out = []
    metadata_names = (
        "row_uid",
        "content_group_id",
        "split",
        "doc_id",
        "orig_index",
        "passage_id",
        "position",
        "token",
        "token_id",
        "corpus",
        "source",
        "lang",
        "seq_len",
        "context_tail",
        "continuation",
        "norm",
    )
    for index in range(n):
        row = {
            "idx": index,
            "global_idx": int(global_indices[index]),
            **{name: meta[name][index] for name in metadata_names},
            "paragraph_count": paragraph_counts[index],
            "q": float(q_router[index]),
            "q_router": float(q_router[index]),
            "route_nla": bool(route_nla[index]),
            "routed_to": str(routed_to[index]),
            "scores": {
                name: {
                    field: float(values[field][index])
                    for field in (
                        "cos_c",
                        "cos_raw",
                        "l2_error",
                        "pred_norm",
                        "target_norm",
                        "norm_ratio",
                    )
                }
                for name, values in scores.items()
            },
        }
        rows_out.append(row)

    npz_payload: dict[str, np.ndarray] = {
        "x": x.astype(np.float32),
        "m_hat": m_hat.astype(np.float64),
        "doc_ids": np.asarray(meta["doc_id"], dtype=np.int64),
        "positions": np.asarray(meta["position"], dtype=np.int64),
        "global_indices": global_indices.astype(np.int64),
        "row_uids": np.asarray(row_uids, dtype=np.str_),
        "content_group_ids": np.asarray(
            meta["content_group_id"], dtype=np.str_
        ),
        "splits": np.asarray([args.split] * n, dtype=np.str_),
        "pred_orig": predictions["orig"].astype(np.float32),
        "recon_sae_small": recon_small.astype(np.float32),
        "recon_sae_big": recon_big.astype(np.float32),
        "l0_sae_small": l0_small,
        "l0_sae_big": l0_big,
        "q_router": q_router.astype(np.float64),
        "route_nla": route_nla,
        "routed_to": routed_to,
        "activations_sha256": np.asarray(activations_sha256),
        "plan_sha256": np.asarray(plan_sha256),
        "model_manifest_sha256": np.asarray(model_manifest_sha256),
        "prereg_sha256": np.asarray(prereg_sha256),
        "explanations_sha256": np.asarray(explanations_sha256),
        "variants_sha256": np.asarray(variants_sha256),
        "gate_sha256": np.asarray(
            "" if gate_info is None else gate_info["sha256"]
        ),
    }
    if channel_active:
        npz_payload.update(
            {
                "pred_p3_only": predictions["p3_only"].astype(np.float32),
                "pred_p12": predictions["p12"].astype(np.float32),
                "pred_quote_strip_p3": predictions[
                    "quote_strip_p3"
                ].astype(np.float32),
            }
        )

    args.vecs_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = args.vecs_out.with_name(args.vecs_out.name + ".partial.npz")
    if temporary_npz.exists():
        raise FileExistsError(f"stale partial NPZ requires audit: {temporary_npz}")
    np.savez_compressed(temporary_npz, **npz_payload)
    os.replace(temporary_npz, args.vecs_out)
    vecs_sha256 = sha256_file(args.vecs_out)
    args.vecs_out.with_suffix(args.vecs_out.suffix + ".sha256").write_text(
        f"{vecs_sha256}  {args.vecs_out.name}\n", encoding="utf-8"
    )

    output = {
        "schema_version": 1,
        "experiment": "N5 split-wise NLA/SAE reconstruction",
        "status": "COMPLETE" if not args.limit else "SMOKE_LIMITED",
        "split": args.split,
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": activations_sha256,
            "plan": str(args.plan),
            "plan_sha256": plan_sha256,
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha256,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha256,
            "script_sha256": script_sha256,
            "pilot_common_sha256": pilot_common_sha256,
            "contract_sha256": contract_sha256,
            "gate": None if args.gate is None else str(args.gate),
            "gate_sha256": None if gate_info is None else gate_info["sha256"],
            "model_manifests": model_manifests,
        },
        "cohort": {
            "n_rows": n,
            "expected_n_rows": EXPECTED_SPLIT_COUNTS[args.split],
            "limit": args.limit,
            "n_content_groups": len(set(meta["content_group_id"])),
            "by_corpus": dict(Counter(str(value) for value in meta["corpus"])),
            "by_source": dict(Counter(str(value) for value in meta["source"])),
            "by_language": dict(Counter(str(value) for value in meta["lang"])),
            "row_uid_sequence_sha256": canonical_sha256(row_uids),
        },
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint_scope": "append-only AV original explanations",
            "checkpoint_rows_loaded": len(completed) - len(missing),
        },
        "forward_counts": {
            "av_generate_total_artifact": n,
            "av_generate_this_invocation": len(missing),
            "ar_reconstruct_unique_texts": len(ar_cache),
            "sae_small_batched_forwards": 1,
            "sae_big_batched_forwards": 1,
            "total_reconstruction_model_calls_artifact": (
                n + len(ar_cache) + 2
            ),
        },
        "channel": {
            "status": channel_status,
            "active": channel_active,
            "abort_origin": abort_origin,
            "local_all_at_least_three_paragraphs": local_paragraph_pass,
            "discovery_channel_active": discovery_channel_active,
            "offending_row_uids": offending,
            "paragraph_count_distribution": dict(
                sorted(Counter(paragraph_counts).items())
            ),
            "npz_channel_keys_present": (
                ["pred_p3_only", "pred_p12", "pred_quote_strip_p3"]
                if channel_active
                else []
            ),
        },
        "centering": {
            "score": "cos(P_perp_m(prediction), P_perp_m(x))",
            "source": centering_source,
            "m_hat_norm": float(np.linalg.norm(m_hat.astype(np.float64))),
            "heldout_center_refit": False,
        },
        "routing": {
            "score_name": "absolute_nla_centered_cosine",
            "q_equals_scores_orig_cos_c": True,
            "gate_status": None if gate_info is None else gate_info["gate"]["status"],
            "gate_feasible": None if gate_info is None else gate_info["feasible"],
            "routing_fraction_frozen": (
                None
                if gate_info is None
                else gate_info["gate"].get("routing_fraction")
            ),
            "threshold": None if gate_info is None else gate_info["threshold"],
            "tie_hash_cutoff_inclusive": (
                None
                if gate_info is None
                else gate_info["tie_hash_cutoff_inclusive"]
            ),
            "heldout_threshold_rule": (
                None if gate_info is None else gate_info["heldout_rule"]
            ),
            "n_routed_nla": int(route_nla.sum()),
            "n_routed_sae_big": int(n - route_nla.sum()),
        },
        "summary": summary,
        "rows": rows_out,
        "qa": {
            **plan_closure,
            "activation_selected_row_uid_sequence_sha256": activation_qa[
                "selected_row_uid_sequence_sha256"
            ],
            "content_groups_unique": activation_qa[
                "selected_content_groups_unique"
            ],
            "all_vectors_finite": True,
            "q_matches_orig_cos_c_max_abs": float(
                np.max(np.abs(q_router - scores["orig"]["cos_c"]))
            ),
            "heldout_uses_frozen_discovery_center": args.split == "heldout",
        },
        "outputs": {
            "explanations": str(args.explanations_out),
            "explanations_sha256": explanations_sha256,
            "variants": str(args.variants_out),
            "variants_sha256": variants_sha256,
            "vecs": str(args.vecs_out),
            "vecs_sha256": vecs_sha256,
        },
        "npz_semantic_contract": {
            "row_axis": (
                "JSON rows, NPZ x/predictions/doc_ids/positions/row_uids share "
                "the exact split-parquet order"
            ),
            "row_uids_key": "row_uids",
            "target_key": "x",
            "prediction_keys": {
                "orig": "pred_orig",
                "sae_small": "recon_sae_small",
                "sae_big": "recon_sae_big",
                **(
                    {
                        "p3_only": "pred_p3_only",
                        "p12": "pred_p12",
                        "quote_strip_p3": "pred_quote_strip_p3",
                    }
                    if channel_active
                    else {}
                ),
            },
            "m_hat_key": "m_hat",
            "q_key": "q_router",
            "route_key": "route_nla",
            "npz_keys": sorted(npz_payload),
            "npz_shape_x": list(x.shape),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "n_unique_ar_texts": len(ar_cache),
    }
    output_sha256 = write_new_json(args.out, output)
    print("\n=== N5 reconstruction summary ===")
    print(
        f"split={args.split} rows={n} channel={channel_status} "
        f"route_nla={int(route_nla.sum())}"
    )
    for name in predictions:
        values = summary[name]
        print(
            f"{name:<18} cos_c={values['mean_cos_c']:+.5f} "
            f"top1={values['retrieval']['top1']:.3f}"
        )
    print(
        f"N5_RECON_COMPLETE split={args.split} json={output_sha256} "
        f"npz={vecs_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
