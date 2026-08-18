#!/usr/bin/env python3
"""Shared mechanical contracts for the unfrozen N6+ implementation.

This module contains only artifact/provenance helpers and descriptive geometry.
It does not define cohort quotas, donor policy, confirmatory endpoints, or
decision thresholds.  Those values must come from a binding preregistration
and the frozen artifacts that descend from it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NEGATIVE_KL_TOL = -1e-7
IDENTITY_KL_TOL = 1e-5
N5_MODEL_MANIFEST_ENTRIES = 25


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact_hash(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> str:
    """Require a readable legacy artifact to match one immutable known hash."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required {label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: {observed} != {expected_sha256.lower()}"
        )
    return observed


def shahex(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def verify_sha256_sidecar(path: str | Path) -> str:
    """Verify the standard ``HASH  basename`` sidecar and return HASH."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    observed = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise ValueError(f"malformed SHA-256 sidecar: {sidecar}")
    declared, declared_name = fields
    if declared.lower() != observed:
        raise ValueError(
            f"{path} differs from sidecar: observed={observed}, declared={declared}"
        )
    if declared_name != path.name:
        raise ValueError(
            f"{sidecar} names {declared_name!r}, expected {path.name!r}"
        )
    return observed


def require_binding_preregistration(path: str | Path) -> str:
    """Reject the current draft and require a verified binding sidecar."""
    path = Path(path)
    if ".DRAFT" in path.name.upper():
        raise ValueError(
            f"N6 preregistration is still a draft and cannot bind a run: {path}"
        )
    text = path.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:20]).upper()
    if "STATUS:" in head and "DRAFT" in head:
        raise ValueError(
            f"N6 preregistration declares DRAFT status and cannot bind a run: {path}"
        )
    return verify_sha256_sidecar(path)


def write_new_json(path: str | Path, value: Any) -> str:
    """Write one immutable pretty JSON artifact and its SHA-256 sidecar."""
    path = Path(path)
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


def write_new_text(path: str | Path, text: str) -> str:
    """Write one immutable UTF-8 text artifact and standard sidecar."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def write_or_verify_frozen_json(path: str | Path, value: Any) -> str:
    """Create a frozen JSON once or require byte identity on checkpoint resume."""
    path = Path(path)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen artifact differs on resume: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n"
    if sidecar.exists():
        if sidecar.read_text(encoding="utf-8") != sidecar_payload:
            raise ValueError(f"frozen sidecar differs on resume: {sidecar}")
    else:
        sidecar.write_text(sidecar_payload, encoding="utf-8")
    return digest


def parse_weight_manifest(path: str | Path) -> dict[str, str]:
    path = Path(path)
    rows: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"malformed model manifest at {path}:{line_number}")
        digest = fields[0].lower()
        filename = fields[1].strip()
        if filename in rows:
            raise ValueError(f"duplicate model manifest path: {filename}")
        rows[filename] = digest
    if len(rows) != N5_MODEL_MANIFEST_ENTRIES:
        raise ValueError(
            f"N5 combined model manifest requires "
            f"{N5_MODEL_MANIFEST_ENTRIES} entries, found {len(rows)}"
        )
    return rows


def verify_code_manifest(
    manifest_path: str | Path,
    script_path: str | Path,
    *,
    extra_paths: Iterable[str | Path] = (),
) -> str:
    """Bind a stage and every imported experiment helper to frozen code.

    ``n6_common.py`` is checked automatically because every N6 Python stage
    imports it.  Callers must pass any additional experiment helpers they
    import (for example ``pilot_common.py`` and ``nla_inference.py``).
    Matching by basename deliberately permits a manifest made on the remote
    host to be verified against the corresponding local checkout, while the
    exactly-once rule prevents an ambiguous basename from being accepted.
    """
    manifest_path = Path(manifest_path)
    manifest_sha = verify_sha256_sidecar(manifest_path)
    script_path = Path(script_path)
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(
                f"malformed code manifest at {manifest_path}:{line_number}"
            )
        declared_path = fields[1].strip().lstrip("*")
        entries.append((fields[0].lower(), declared_path))

    required_paths: list[Path] = [script_path, Path(__file__)]
    required_paths.extend(Path(value) for value in extra_paths)
    seen_required: set[str] = set()
    for required_path in required_paths:
        resolved = str(required_path.resolve())
        if resolved in seen_required:
            continue
        seen_required.add(resolved)
        if not required_path.is_file():
            raise ValueError(f"required code dependency is missing: {required_path}")
        candidates = [
            (digest, declared_path)
            for digest, declared_path in entries
            if Path(declared_path).name == required_path.name
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"code manifest must name {required_path.name} exactly once; "
                f"found {len(candidates)}"
            )
        observed = sha256_file(required_path)
        if candidates[0][0] != observed:
            raise ValueError(
                f"{required_path.name} differs from frozen code manifest: "
                f"{observed} != {candidates[0][0]}"
            )
    return manifest_sha


def model_file_manifest(root: str | Path) -> dict[str, Any]:
    """Hash model weights plus the small identity/configuration files."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"model directory does not exist: {root}")
    files: set[Path] = set()
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
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
        index = read_json_object(index_path)
        for relative in set(index.get("weight_map", {}).values()):
            shard = root / str(relative)
            if not shard.is_file():
                raise ValueError(f"weight index references missing shard: {shard}")
            files.add(shard)
    if not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError(f"no model weights found under {root}")
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
    }
    return {
        "root": str(root),
        "files": hashes,
        "manifest_sha256": canonical_sha256(hashes),
    }


def validate_model_subset(
    actual: dict[str, dict[str, Any]],
    frozen: dict[str, str],
) -> None:
    flattened: dict[str, str] = {}
    for label, item in actual.items():
        root = str(item["root"]).rstrip("/")
        for relative, digest in item["files"].items():
            flattened[f"{root}/{relative}"] = str(digest).lower()
    expected = {
        path: digest
        for path, digest in frozen.items()
        if any(path.startswith(str(item["root"]).rstrip("/") + "/") for item in actual.values())
    }
    if not expected:
        raise ValueError("model roots do not occur in the frozen N5 manifest")
    wrong = {
        path: {"expected": digest, "actual": flattened.get(path)}
        for path, digest in expected.items()
        if flattened.get(path) != digest
    }
    if wrong:
        raise ValueError(f"actual model files differ from N5 manifest: {wrong}")


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize a zero/non-finite vector")
    return vector / norm


def centered_cosine(
    prediction: np.ndarray,
    target: np.ndarray,
    mean_direction: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    direction = unit(mean_direction)
    pred_centered = prediction - np.outer(prediction @ direction, direction)
    target_centered = target - np.outer(target @ direction, direction)
    denominator = np.linalg.norm(pred_centered, axis=1) * np.linalg.norm(
        target_centered, axis=1
    )
    if np.any(denominator <= 1e-12) or not np.isfinite(denominator).all():
        raise ValueError("centered-cosine denominator is zero or non-finite")
    answer = np.sum(pred_centered * target_centered, axis=1) / denominator
    if not np.isfinite(answer).all():
        raise ValueError("centered cosine is non-finite")
    return answer


def raw_cosine(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    if np.any(denominator <= 1e-12):
        raise ValueError("raw-cosine denominator is zero")
    return np.sum(prediction * target, axis=1) / denominator


def retrieval(
    prediction: np.ndarray,
    target: np.ndarray,
    mean_direction: np.ndarray,
) -> dict[str, float]:
    prediction = np.array(prediction, dtype=np.float64, copy=True)
    target = np.array(target, dtype=np.float64, copy=True)
    direction = unit(mean_direction)
    prediction -= np.outer(prediction @ direction, direction)
    target -= np.outer(target @ direction, direction)
    prediction /= np.linalg.norm(prediction, axis=1, keepdims=True)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    similarity = prediction @ target.T
    diagonal = np.diag(similarity)
    ranks = 1 + (similarity > diagonal[:, None] + 1e-12).sum(axis=1)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def prediction_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    mean_direction: np.ndarray,
) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    errors = prediction - target
    pred_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    return {
        "cos_c": centered_cosine(prediction, target, mean_direction),
        "cos_raw": raw_cosine(prediction, target),
        "l2_error": np.linalg.norm(errors, axis=1),
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "norm_ratio": pred_norm / np.maximum(target_norm, 1e-30),
    }


def norm_match(source: np.ndarray, target: np.ndarray, name: str) -> np.ndarray:
    """Exact N5 row-wise norm match for every nonzero substitute."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.shape != target.shape:
        raise ValueError(f"{name} shape {source.shape} != target {target.shape}")
    source_norm = np.linalg.norm(source.astype(np.float64), axis=1)
    target_norm = np.linalg.norm(target.astype(np.float64), axis=1)
    if not np.isfinite(source).all() or not np.isfinite(source_norm).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(source_norm <= 1e-12):
        bad = np.flatnonzero(source_norm <= 1e-12)[:10].tolist()
        raise ValueError(f"{name} contains zero-norm rows: {bad}")
    return (source * (target_norm / source_norm)[:, None]).astype(np.float32)


def sanitize_kl(
    value: Any,
    *,
    row_uid: str,
    condition: str,
    field: str,
    clamped: list[dict[str, Any]],
) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{row_uid} {condition}.{field} is non-finite")
    if number < NEGATIVE_KL_TOL:
        raise ValueError(
            f"{row_uid} {condition}.{field}={number} is below "
            f"{NEGATIVE_KL_TOL}"
        )
    if number < 0:
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


def scalar_npz_string(
    archive: np.lib.npyio.NpzFile,
    key: str,
    expected: str | None = None,
) -> str:
    if key not in archive.files:
        raise ValueError(f"NPZ lacks provenance key {key}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"NPZ {key} must be scalar, found {value.shape}")
    observed = str(value.item())
    if expected is not None and observed != expected:
        raise ValueError(f"NPZ {key}={observed!r}, expected {expected!r}")
    return observed


def require_unique(values: Iterable[Any], label: str) -> list[Any]:
    materialized = list(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} values are not unique")
    return materialized
