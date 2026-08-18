#!/usr/bin/env python3
"""Independent, CPU-only audit of the J1 discovery (stage 57) artifacts.

The stage-57 runner performs GPU extraction and AV generation.  This module
does not import that runner and never loads a model on CUDA.  It starts from
the frozen JSON/NPZ/checkpoint/result artifacts, reloads only the SAE decoder
on CPU, and verifies the structural and numerical contracts that can be
recomputed after a remote run.  It deliberately makes no semantic or
confirmatory claim about the generated explanations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


SEED = 20260806
N_STRATA = 3
PER_STRATUM = 15
N_FEATURES = 45
N_DISCOVERY = 4
N_HELDOUT = 4
N_SELECTED_CONTEXTS = N_FEATURES * (N_DISCOVERY + N_HELDOUT)
N_AV_PLANS = N_FEATURES * N_DISCOVERY * 2
STRATA = ("source_concentrated", "source_distributed", "language_selective")
ACTIVATION_RTOL = 2.5e-2
ACTIVATION_ATOL = 1.0
FLOAT32_RTOL = 2.0e-5
FLOAT32_ATOL = 2.0e-5
QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)

DEFAULT_RESULTS = Path("/root/autodl-tmp/results")
DEFAULT_PROTOCOL = DEFAULT_RESULTS / "J1_DISCOVERY_PILOT_PROTOCOL_2026-08-06.md"
DEFAULT_FREEZE = DEFAULT_RESULTS / "j1_discovery_freeze_v1.json"
DEFAULT_VECTORS = DEFAULT_RESULTS / "j1_discovery_vectors_v1.npz"
DEFAULT_CHECKPOINT = DEFAULT_RESULTS / "j1_discovery_av_checkpoint_v1.jsonl"
DEFAULT_RESULT = DEFAULT_RESULTS / "j1_discovery_result_v1.json"
DEFAULT_SAE_PARAMS = (
    Path("/root/autodl-tmp/models/gemma-scope-2-12b-it")
    / "resid_post_all/layer_32_width_16k_l0_small/params.safetensors"
)


class CheckFailure(ValueError):
    """A checked contract failed; the caller records it and keeps auditing."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def array_sha256(value: np.ndarray) -> str:
    """Stage-57's vector hash: contiguous little-endian float32 bytes."""
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    little = array.astype("<f4", copy=False)
    return sha256_bytes(little.tobytes(order="C"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise CheckFailure(f"{label} is missing or is not a regular file: {path}")


def verify_sidecar(path: Path, sidecar: Path | None = None) -> str:
    """Verify either ``digest`` or ``digest  filename`` sidecar spelling."""
    _require_file(path, "artifact")
    digest = sha256_file(path)
    sidecar_path = sidecar or Path(str(path) + ".sha256")
    _require_file(sidecar_path, "SHA-256 sidecar")
    lines = [line.strip() for line in sidecar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise CheckFailure(f"sidecar must contain exactly one non-empty line: {sidecar_path}")
    fields = lines[0].split()
    if len(fields) not in (1, 2) or fields[0].lower() != digest:
        raise CheckFailure(f"SHA-256 sidecar mismatch: {sidecar_path}")
    if len(fields) == 2 and fields[1] != path.name:
        raise CheckFailure(f"sidecar filename mismatch: {sidecar_path}")
    return digest


def read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact JSON decoder differs by Python
        raise CheckFailure(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckFailure(f"{label} must contain one JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require_file(path, label)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise CheckFailure(f"invalid JSONL {label}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise CheckFailure(f"JSONL {label}:{line_number} is not an object")
            rows.append(value)
    return rows


def _as_int(value: Any, label: str) -> int:
    try:
        # bool is an int subclass but never a valid artifact index.
        if isinstance(value, bool):
            raise ValueError
        return int(value)
    except Exception as exc:
        raise CheckFailure(f"{label} is not an integer: {value!r}") from exc


def _as_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise CheckFailure(f"{label} is not numeric: {value!r}") from exc
    if not np.isfinite(number):
        raise CheckFailure(f"{label} is non-finite")
    return number


def _close(actual: float | np.ndarray, expected: float | np.ndarray, *,
           rtol: float = FLOAT32_RTOL, atol: float = FLOAT32_ATOL) -> bool:
    return bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))


def _json_close(actual: Any, expected: Any, *, atol: float = 1e-12) -> bool:
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=atol))
    except Exception:
        return actual == expected


class Audit:
    """Small check ledger that records failures instead of short-circuiting."""

    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, Any]] = []

    def run(self, name: str, function: Callable[[], Any]) -> Any | None:
        try:
            details = function()
            if details is None:
                details = {}
            if isinstance(details, dict):
                recorded = details
            elif isinstance(details, (list, tuple)):
                # Some successful checks return rich in-memory values for
                # subsequent recomputation (notably AV plans containing NumPy
                # vectors).  Record only their cardinality in the JSON audit;
                # the original value is still returned to the caller.
                recorded = {"count": len(details)}
            else:
                recorded = {"value": details.item() if isinstance(details, np.generic) else details}
            self.checks[name] = {"pass": True, **recorded}
            return details
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            self.checks[name] = {"pass": False, "error": message}
            self.errors.append({"check": name, "error": message, "type": type(exc).__name__})
            return None


def _load_sae_wdec(path: Path) -> np.ndarray:
    """Load only ``w_dec`` on CPU; importing safetensors is intentionally lazy."""
    if path.is_dir():
        path = path / "params.safetensors"
    _require_file(path, "SAE params.safetensors")
    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as exc:  # pragma: no cover - remote environment supplies it
        raise CheckFailure(f"safetensors is unavailable: {exc}") from exc
    try:
        tensors = load_file(str(path), device="cpu")
        if "w_dec" not in tensors:
            raise CheckFailure("SAE params.safetensors lacks w_dec")
        tensor = tensors["w_dec"]
        # ``detach``/``cpu`` also handles a torch tensor loaded in a non-default dtype.
        value = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    except CheckFailure:
        raise
    except Exception as exc:
        raise CheckFailure(f"could not load CPU SAE w_dec: {exc}") from exc
    if value.ndim != 2 or not np.isfinite(value).all():
        raise CheckFailure(f"SAE w_dec must be finite rank-2, got {value.shape}")
    return np.ascontiguousarray(value, dtype=np.float32)


def _write_immutable(path: Path, payload: bytes) -> str:
    """Create an output once; an existing byte-identical output is resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical audit output: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return sha256_bytes(payload)


def _write_immutable_sidecar(path: Path, digest: str, target: Path) -> None:
    payload = f"{digest}  {target.name}\n".encode("ascii")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical audit sidecar: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def _artifact_descriptor(path: Path, digest: str | None = None) -> dict[str, Any]:
    return {"path": str(path), "sha256": digest or sha256_file(path), "bytes": path.stat().st_size}


def _check_protocol(protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    # The protocol is Markdown; these checks avoid silently auditing another
    # experiment while not attempting to judge its prose.
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    if "j1" not in lowered or "discovery" not in lowered:
        raise CheckFailure("protocol does not identify J1 discovery")
    if "status: **exploratory / discovery only**" not in lowered:
        raise CheckFailure("protocol status is not exploratory/discovery-only")
    if "confirmatory" not in lowered:
        raise CheckFailure("protocol lacks explicit confirmatory boundary")
    return {"status": "exploratory_discovery_only", "bytes": path.stat().st_size}


def _check_freeze_status(freeze: dict[str, Any]) -> dict[str, Any]:
    if freeze.get("experiment") != "J1 NLA-to-SAE exploratory discovery pilot":
        raise CheckFailure("freeze experiment binding is not J1 discovery")
    if freeze.get("status") != "EXPLORATORY_DISCOVERY_FROZEN_BEFORE_AV":
        raise CheckFailure(f"unexpected freeze status {freeze.get('status')!r}")
    if freeze.get("confirmatory") is not False:
        raise CheckFailure("freeze confirmatory must be false")
    if freeze.get("claim_scope") != "discovery_only_no_confirmatory_inference":
        raise CheckFailure("freeze claim_scope is not discovery-only")
    if freeze.get("schema_version") != 1:
        raise CheckFailure("unsupported freeze schema_version")
    selection = freeze.get("selection")
    if not isinstance(selection, dict):
        raise CheckFailure("freeze selection metadata is missing")
    if _as_int(selection.get("seed"), "selection.seed") != SEED:
        raise CheckFailure("selection seed differs from protocol")
    if tuple(selection.get("strata", ())) != STRATA:
        raise CheckFailure("selection strata differ from protocol")
    for key, expected in (("per_stratum", PER_STRATUM), ("n_features", N_FEATURES),
                          ("contexts_per_feature", 8), ("discovery_per_feature", 4),
                          ("heldout_positive_per_feature", 4)):
        if _as_int(selection.get(key), f"selection.{key}") != expected:
            raise CheckFailure(f"selection.{key} differs: expected {expected}")
    if selection.get("document_disjoint_within_feature") is not True:
        raise CheckFailure("document-disjoint selection contract is not true")
    return {"status": freeze["status"], "seed": SEED}


def _flatten_freeze_contexts(freeze: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    features = freeze.get("features")
    if not isinstance(features, list) or len(features) != N_FEATURES:
        raise CheckFailure(f"freeze must contain exactly {N_FEATURES} features")
    selected: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    feature_ids: list[int] = []
    seen_features: set[int] = set()
    seen_indices: set[int] = set()
    for feature_row_number, feature_row in enumerate(features):
        if not isinstance(feature_row, dict):
            raise CheckFailure(f"feature row {feature_row_number} is not an object")
        feature = _as_int(feature_row.get("feature"), f"features[{feature_row_number}].feature")
        if feature in seen_features:
            raise CheckFailure(f"duplicate feature {feature}")
        seen_features.add(feature)
        feature_ids.append(feature)
        if str(feature_row.get("stratum")) not in STRATA:
            raise CheckFailure(f"feature {feature} has unexpected stratum")
        discovery = feature_row.get("discovery")
        heldout = feature_row.get("heldout_positive")
        if not isinstance(discovery, list) or len(discovery) != N_DISCOVERY:
            raise CheckFailure(f"feature {feature} must have four discovery contexts")
        if not isinstance(heldout, list) or len(heldout) != N_HELDOUT:
            raise CheckFailure(f"feature {feature} must have four held-out positives")
        rows = discovery + heldout
        docs = []
        for local_index, context in enumerate(rows):
            if not isinstance(context, dict):
                raise CheckFailure(f"feature {feature} context {local_index} is not an object")
            role = "discovery" if local_index < N_DISCOVERY else "heldout_positive"
            if context.get("role") != role:
                raise CheckFailure(f"feature {feature} context {local_index} role mismatch")
            if _as_int(context.get("feature"), "context.feature") != feature:
                raise CheckFailure(f"feature {feature} context feature mismatch")
            if str(context.get("stratum")) != str(feature_row["stratum"]):
                raise CheckFailure(f"feature {feature} context stratum mismatch")
            context_index = _as_int(context.get("context_index"), "context.context_index")
            if context_index in seen_indices:
                raise CheckFailure(f"duplicate selected context_index {context_index}")
            seen_indices.add(context_index)
            doc_id = _as_int(context.get("doc_id"), "context.doc_id")
            position = _as_int(context.get("position"), "context.position")
            if position < 0:
                raise CheckFailure(f"feature {feature} context position is negative")
            docs.append(doc_id)
            for key in ("source", "lang", "corpus", "raw_text_sha256", "context_text"):
                if key not in context:
                    raise CheckFailure(f"feature {feature} context lacks {key}")
            if not isinstance(context.get("context_text"), str) or not context["context_text"].strip():
                raise CheckFailure(f"feature {feature} context_text is empty")
            if not _is_sha256(context.get("raw_text_sha256")):
                raise CheckFailure(f"feature {feature} raw_text_sha256 is malformed")
            if "expected_activation" not in context:
                raise CheckFailure(f"feature {feature} context lacks expected_activation")
            expected = _as_float(context["expected_activation"], "context.expected_activation")
            if expected <= 0.0:
                raise CheckFailure(f"feature {feature} expected activation is not positive")
            # Stage 57 freezes the independently recomputed activation and its
            # errors into every selected context.  Require those leaves rather
            # than silently accepting a freeze that only records expectations.
            for key in ("actual_activation", "activation_absolute_error", "activation_relative_error"):
                if key not in context:
                    raise CheckFailure(f"feature {feature} context lacks {key}")
                _as_float(context[key], f"context.{key}")
            selected.append(context)
            if role == "heldout_positive":
                negative = context.get("hard_negative")
                if not isinstance(negative, dict):
                    raise CheckFailure(f"feature {feature} positive lacks hard_negative")
                negatives.append({"feature_row": feature_row, "positive": context, "negative": negative})
        if len(set(docs)) != N_DISCOVERY + N_HELDOUT:
            raise CheckFailure(f"feature {feature} discovery/positive documents overlap")
    counts = {stratum: sum(str(row.get("stratum")) == stratum for row in features) for stratum in STRATA}
    if counts != {stratum: PER_STRATUM for stratum in STRATA}:
        raise CheckFailure(f"feature stratum counts are {counts}, expected 15 each")
    if seen_indices != set(range(N_SELECTED_CONTEXTS)):
        raise CheckFailure("selected context_index values are not exactly 0..359")
    return selected, negatives, feature_ids


def _check_background_metadata(freeze: dict[str, Any], negative_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = freeze.get("background_pool")
    if not isinstance(metadata, dict):
        raise CheckFailure("freeze lacks background_pool metadata")
    for key in ("total", "by_group", "groups", "sha256", "seed", "requested_per_group", "max_per_doc", "min_per_group"):
        if key not in metadata:
            raise CheckFailure(f"background_pool lacks {key}")
    total = _as_int(metadata["total"], "background_pool.total")
    by_group = metadata["by_group"]
    groups = metadata["groups"]
    if total < 0 or not isinstance(by_group, dict) or not isinstance(groups, dict):
        raise CheckFailure("background_pool total/by_group/groups malformed")
    if not _is_sha256(metadata.get("sha256")):
        raise CheckFailure("background_pool.sha256 is malformed")
    if sum(_as_int(value, f"background_pool.by_group[{key}]") for key, value in by_group.items()) != total:
        raise CheckFailure("background_pool.by_group counts do not sum to total")
    if set(by_group) != set(groups):
        raise CheckFailure("background_pool by_group/groups keys differ")
    for group_key, realized_value in by_group.items():
        realized = _as_int(realized_value, f"background_pool.by_group[{group_key}]")
        detail = groups[group_key]
        if not isinstance(detail, dict):
            raise CheckFailure(f"background group {group_key} metadata is not an object")
        if _as_int(detail.get("realized"), f"background group {group_key}.realized") != realized:
            raise CheckFailure(f"background group {group_key} realized count differs")
        if _as_int(detail.get("requested"), f"background group {group_key}.requested") != 32:
            raise CheckFailure(f"background group {group_key} requested count differs")
        if _as_int(detail.get("max_per_doc"), f"background group {group_key}.max_per_doc") != 2:
            raise CheckFailure(f"background group {group_key} max_per_doc differs")
        if _as_int(detail.get("min_required"), f"background group {group_key}.min_required") != 4 or realized < 4:
            raise CheckFailure(f"background group {group_key} violates minimum")
    embedded_by_group: dict[str, int] = {}
    for pair in negative_pairs:
        negative = pair["negative"]
        key = "|".join((str(negative.get("source")), str(negative.get("lang")), str(negative.get("corpus"))))
        embedded_by_group[key] = embedded_by_group.get(key, 0) + 1
    for key, count in embedded_by_group.items():
        if key not in by_group or count > _as_int(by_group[key], f"background_pool.by_group[{key}]"):
            raise CheckFailure(f"embedded hard negatives exceed background group count for {key}")
    # The complete candidate stable-row list is intentionally not in the
    # freeze, so its hash cannot be regenerated without the corpus/tokenizer.
    return {
        "total": total,
        "groups": len(by_group),
        "by_group": {str(key): _as_int(value, f"background_pool.by_group[{key}]") for key, value in by_group.items()},
        "embedded_hard_negative_counts": embedded_by_group,
        "stable_rows_sha256_format": True,
        "stable_rows_hash_recomputed": False,
    }


def _load_vectors(path: Path, w_dec: np.ndarray, freeze: dict[str, Any]) -> dict[str, np.ndarray]:
    _require_file(path, "vectors NPZ")
    required = {
        "residuals", "sae_acts", "wdec_controls", "sae_ablated_residuals",
        "contrastive_activation", "contrastive_norms", "contrastive_cosine",
        "context_indices", "feature_ids",
    }
    try:
        archive = np.load(path, allow_pickle=False)
        with archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise CheckFailure(f"vectors NPZ lacks {missing}")
            vectors = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except CheckFailure:
        raise
    except Exception as exc:
        raise CheckFailure(f"could not read vectors NPZ: {exc}") from exc
    n_rows = vectors["residuals"].shape[0] if vectors["residuals"].ndim >= 1 else -1
    width, d_model = int(w_dec.shape[0]), int(w_dec.shape[1])
    expected_shapes = {
        "residuals": (n_rows, d_model),
        "sae_acts": (n_rows, width),
        "wdec_controls": (N_FEATURES, N_DISCOVERY, d_model),
        "sae_ablated_residuals": (N_FEATURES, N_DISCOVERY, d_model),
        "contrastive_activation": (N_FEATURES, N_DISCOVERY),
        "contrastive_norms": (N_FEATURES, N_DISCOVERY, 2),
        "contrastive_cosine": (N_FEATURES, N_DISCOVERY),
        "context_indices": (n_rows,),
        "feature_ids": (N_FEATURES,),
    }
    for key, shape in expected_shapes.items():
        if tuple(vectors[key].shape) != tuple(shape):
            raise CheckFailure(f"vectors[{key}] shape {vectors[key].shape} != {shape}")
    if n_rows < N_SELECTED_CONTEXTS:
        raise CheckFailure(f"vectors have only {n_rows} rows; need at least 360")
    if not np.array_equal(vectors["context_indices"], np.arange(n_rows, dtype=vectors["context_indices"].dtype)):
        raise CheckFailure("vectors.context_indices are not sequential")
    if not np.issubdtype(vectors["context_indices"].dtype, np.integer) or not np.issubdtype(vectors["feature_ids"].dtype, np.integer):
        raise CheckFailure("vectors context_indices/feature_ids must be integer arrays")
    for key, value in vectors.items():
        if key in ("context_indices", "feature_ids"):
            continue
        if not np.isfinite(value).all():
            raise CheckFailure(f"vectors[{key}] contains non-finite values")
    freeze_ids = [_as_int(row.get("feature"), "freeze.feature") for row in freeze.get("features", [])]
    feature_ids = [_as_int(value, "vectors.feature_ids") for value in vectors["feature_ids"].tolist()]
    if len(set(feature_ids)) != N_FEATURES or sorted(feature_ids) != sorted(freeze_ids):
        raise CheckFailure("vectors.feature_ids do not exactly match freeze feature IDs")
    if n_rows != N_SELECTED_CONTEXTS + _as_int(freeze.get("background_pool", {}).get("total"), "background_pool.total"):
        raise CheckFailure("vectors row count does not equal 360 selected plus background total")
    return vectors


def _check_activation_metrics(
    freeze: dict[str, Any],
    selected: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
) -> dict[str, Any]:
    acts = vectors["sae_acts"]
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    sign_mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    for context in selected:
        ci = _as_int(context.get("context_index"), "context.context_index")
        feature = _as_int(context.get("feature"), "context.feature")
        expected = _as_float(context.get("expected_activation"), "context.expected_activation")
        actual = float(acts[ci, feature])
        if actual <= 0.0:
            raise CheckFailure(f"feature {feature} context {ci} actual activation is not positive")
        if (expected > 0.0) != (actual > 0.0):
            sign_mismatches += 1
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), 1e-12)
        abs_errors.append(absolute)
        rel_errors.append(relative)
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        if not np.isclose(actual, expected, rtol=ACTIVATION_RTOL, atol=ACTIVATION_ATOL):
            raise CheckFailure(f"activation mismatch at context {ci}, feature {feature}: {actual} vs {expected}")
        for key, recomputed in (("actual_activation", actual), ("activation_absolute_error", absolute), ("activation_relative_error", relative)):
            if key in context and not _json_close(context[key], recomputed, atol=1e-12):
                raise CheckFailure(f"freeze context {ci} {key} does not match vectors")
    if sign_mismatches:
        raise CheckFailure(f"activation firing sign mismatches={sign_mismatches}")
    quant_abs = {str(q): float(np.quantile(abs_errors, q)) for q in QUANTILES}
    quant_rel = {str(q): float(np.quantile(rel_errors, q)) for q in QUANTILES}
    if max_rel > ACTIVATION_RTOL:
        raise CheckFailure(f"maximum relative activation error {max_rel} exceeds {ACTIVATION_RTOL}")
    stored = freeze.get("checks", {}).get("activation_verification")
    if not isinstance(stored, dict):
        raise CheckFailure("freeze checks.activation_verification is missing")
    expected_scalars = {
        "n_contexts": len(selected),
        "rtol": ACTIVATION_RTOL,
        "atol": ACTIVATION_ATOL,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "firing_sign_mismatches": 0,
    }
    for key, expected_value in expected_scalars.items():
        if key not in stored or not _json_close(stored[key], expected_value, atol=1e-12):
            raise CheckFailure(f"freeze activation_verification.{key} does not match recomputation")
    for key, expected_map in (("absolute_error_quantiles", quant_abs), ("relative_error_quantiles", quant_rel)):
        value = stored.get(key)
        if not isinstance(value, dict):
            raise CheckFailure(f"freeze activation_verification.{key} is missing")
        for q, expected_value in expected_map.items():
            if q not in value or not _json_close(value[q], expected_value, atol=1e-12):
                raise CheckFailure(f"freeze activation_verification.{key}[{q}] does not match")
    if freeze.get("checks", {}).get("activation_verification_errors") != []:
        raise CheckFailure("freeze contains activation_verification_errors")
    return {
        "n_contexts": len(selected),
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "absolute_error_quantiles": quant_abs,
        "relative_error_quantiles": quant_rel,
        "firing_sign_mismatches": sign_mismatches,
        "rtol": ACTIVATION_RTOL,
        "atol": ACTIVATION_ATOL,
    }


def _check_negatives(
    negative_pairs: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
) -> dict[str, Any]:
    residuals = vectors["residuals"]
    acts = vectors["sae_acts"]
    n_rows = residuals.shape[0]
    seen_by_feature: dict[int, set[tuple[int, int]]] = {}
    seen_candidate_indices: dict[int, set[int]] = {}
    tier_counts = {0: 0, 1: 0, 2: 0}
    candidate_pool_counts = {"selected_context": 0, "background": 0}
    for pair in negative_pairs:
        feature_row = pair["feature_row"]
        positive = pair["positive"]
        negative = pair["negative"]
        feature = _as_int(feature_row.get("feature"), "negative.feature")
        if _as_int(negative.get("feature"), "negative.feature") != feature:
            raise CheckFailure(f"feature {feature} negative feature mismatch")
        positive_ci = _as_int(positive.get("context_index"), "positive.context_index")
        if _as_int(negative.get("target_positive_context_index"), "negative.target_positive_context_index") != positive_ci:
            raise CheckFailure(f"feature {feature} negative target positive index mismatch")
        candidate_ci = _as_int(negative.get("candidate_context_index"), "negative.candidate_context_index")
        # Stage 57 searches the complete extracted candidate set: the 360
        # selected contexts plus the deterministic A2 background extension.
        # A hard negative is therefore allowed to be another feature's
        # selected context; it is not required to come from the background.
        if not 0 <= candidate_ci < n_rows:
            raise CheckFailure(f"feature {feature} candidate context_index {candidate_ci} is out of range")
        candidate_pool_counts["selected_context" if candidate_ci < N_SELECTED_CONTEXTS else "background"] += 1
        candidate_doc = _as_int(negative.get("doc_id"), "negative.doc_id")
        candidate_position = _as_int(negative.get("position"), "negative.position")
        physical = (candidate_doc, candidate_position)
        seen = seen_by_feature.setdefault(feature, set())
        if physical in seen:
            raise CheckFailure(f"feature {feature} reuses physical negative position {physical}")
        seen.add(physical)
        if candidate_ci in seen_candidate_indices.setdefault(feature, set()):
            raise CheckFailure(f"feature {feature} reuses candidate context_index {candidate_ci}")
        seen_candidate_indices[feature].add(candidate_ci)
        selected_docs = {
            _as_int(context.get("doc_id"), "context.doc_id")
            for context in (feature_row.get("discovery", []) + feature_row.get("heldout_positive", []))
        }
        if candidate_doc in selected_docs:
            raise CheckFailure(f"feature {feature} negative reuses a selected document")
        required_fields = ("token_id", "token", "before", "after", "context_text", "raw_text_sha256", "seq_len", "source", "lang", "corpus", "target_activation", "residual_norm", "norm_log_distance", "position_distance", "preference_tier")
        missing = [key for key in required_fields if key not in negative]
        if missing:
            raise CheckFailure(f"feature {feature} negative lacks {missing}")
        # Tokenizer-decoded whitespace tokens are valid real-text positions
        # (12 such selected-context negatives occur in this frozen cohort).
        # Their surrounding rendered context must still be nonblank.
        for key in ("token", "before", "after"):
            if not isinstance(negative[key], str):
                raise CheckFailure(f"feature {feature} negative {key} is not text")
        if not isinstance(negative["context_text"], str) or not negative["context_text"].strip():
            raise CheckFailure(f"feature {feature} negative context_text is empty")
        if not _is_sha256(negative.get("raw_text_sha256")):
            raise CheckFailure(f"feature {feature} negative raw_text_sha256 is malformed")
        if _as_int(negative.get("seq_len"), "negative.seq_len") <= 0:
            raise CheckFailure(f"feature {feature} negative seq_len is non-positive")
        candidate_activation = float(acts[candidate_ci, feature])
        stored_activation = _as_float(negative.get("target_activation"), "negative.target_activation")
        if stored_activation != candidate_activation or candidate_activation != 0.0:
            raise CheckFailure(f"feature {feature} negative activation is not exact vectors zero ({stored_activation} vs {candidate_activation})")
        positive_source = str(positive.get("source"))
        positive_lang = str(positive.get("lang"))
        positive_corpus = str(positive.get("corpus"))
        source = str(negative.get("source"))
        lang = str(negative.get("lang"))
        corpus = str(negative.get("corpus"))
        if (source, lang) == (positive_source, positive_lang):
            tier = 0
        elif source == positive_source:
            tier = 1
        elif (corpus, lang) == (positive_corpus, positive_lang):
            tier = 2
        else:
            raise CheckFailure(f"feature {feature} negative does not satisfy strict tier 0/1/2")
        if _as_int(negative.get("preference_tier"), "negative.preference_tier") != tier:
            raise CheckFailure(f"feature {feature} negative preference tier is incorrect")
        positive_position = _as_int(positive.get("position"), "positive.position")
        positive_norm = max(float(np.linalg.norm(residuals[positive_ci])), 1e-12)
        candidate_norm = float(np.linalg.norm(residuals[candidate_ci]))
        if not np.isfinite(positive_norm) or not np.isfinite(candidate_norm):
            raise CheckFailure(f"feature {feature} negative residual norms are non-finite")
        stored_norm = _as_float(negative.get("residual_norm"), "negative.residual_norm")
        if not _close(stored_norm, candidate_norm, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL):
            raise CheckFailure(f"feature {feature} negative residual_norm mismatch")
        norm_distance = abs(float(np.log(max(candidate_norm, 1e-12)) - np.log(positive_norm)))
        position_distance = abs(candidate_position - positive_position) / max(1.0, float(max(candidate_position, positive_position)))
        if not _json_close(negative.get("norm_log_distance"), norm_distance, atol=2e-6):
            raise CheckFailure(f"feature {feature} norm_log_distance mismatch")
        if not _json_close(negative.get("position_distance"), position_distance, atol=2e-12):
            raise CheckFailure(f"feature {feature} position_distance mismatch")
        tier_counts[tier] += 1
    if len(negative_pairs) != N_FEATURES * N_HELDOUT:
        raise CheckFailure(f"negative count {len(negative_pairs)} != 180")
    if any(len(values) != N_HELDOUT for values in seen_by_feature.values()) or set(seen_by_feature) != {
        _as_int(row.get("feature"), "freeze.feature") for row in [pair["feature_row"] for pair in negative_pairs]
    }:
        raise CheckFailure("negative counts are not four per feature")
    return {
        "n_negatives": len(negative_pairs),
        "tier_counts": {str(key): value for key, value in tier_counts.items()},
        "candidate_pool_counts": candidate_pool_counts,
        "features": len(seen_by_feature),
    }


def _check_ablation(
    freeze: dict[str, Any],
    vectors: dict[str, np.ndarray],
    w_dec: np.ndarray,
    feature_ids: list[int],
) -> dict[str, Any]:
    if freeze.get("include_contrastive_ablation") is not True:
        raise CheckFailure("freeze does not enable the required contrastive ablation arm")
    if freeze.get("include_wdec_control") is not False:
        raise CheckFailure("w_dec control rows are not allowed in the required 360-row audit")
    feature_to_vector_index = {feature: index for index, feature in enumerate(feature_ids)}
    max_abs = 0.0
    max_rel = 0.0
    max_norm_abs = 0.0
    max_cos_abs = 0.0
    features = freeze.get("features", [])
    for feature_row in features:
        feature = _as_int(feature_row.get("feature"), "freeze.feature")
        fi = feature_to_vector_index[feature]
        discovery = feature_row.get("discovery", [])
        for di, context in enumerate(discovery):
            ci = _as_int(context.get("context_index"), "discovery.context_index")
            raw = np.asarray(vectors["residuals"][ci], dtype=np.float32)
            stored_activation = np.float32(vectors["contrastive_activation"][fi, di])
            sae_activation = np.float32(vectors["sae_acts"][ci, feature])
            if stored_activation.tobytes() != sae_activation.tobytes():
                raise CheckFailure(f"feature {feature} discovery {di} stored ablation activation differs from sae_acts")
            decoder = np.asarray(w_dec[feature], dtype=np.float32)
            expected_ablated = np.asarray(raw - stored_activation * decoder, dtype=np.float32)
            actual_ablated = np.asarray(vectors["sae_ablated_residuals"][fi, di], dtype=np.float32)
            abs_error = float(np.max(np.abs(actual_ablated - expected_ablated)))
            rel_error = float(abs_error / max(float(np.max(np.abs(expected_ablated))), 1e-12))
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
            if not _close(actual_ablated, expected_ablated):
                raise CheckFailure(f"feature {feature} discovery {di} SAE-ablation vector mismatch (max_abs={abs_error})")
            expected_norms = np.asarray([np.linalg.norm(raw), np.linalg.norm(actual_ablated)], dtype=np.float32)
            stored_norms = np.asarray(vectors["contrastive_norms"][fi, di], dtype=np.float32)
            norm_error = float(np.max(np.abs(stored_norms - expected_norms)))
            max_norm_abs = max(max_norm_abs, norm_error)
            if not _close(stored_norms, expected_norms):
                raise CheckFailure(f"feature {feature} discovery {di} stored norms mismatch")
            denominator = max(float(np.linalg.norm(raw)) * float(np.linalg.norm(actual_ablated)), 1e-12)
            expected_cosine = np.float32(float(np.dot(raw, actual_ablated) / denominator))
            stored_cosine = np.float32(vectors["contrastive_cosine"][fi, di])
            cosine_error = abs(float(stored_cosine) - float(expected_cosine))
            max_cos_abs = max(max_cos_abs, cosine_error)
            if not _close(stored_cosine, expected_cosine):
                raise CheckFailure(f"feature {feature} discovery {di} stored cosine mismatch")
    return {"pairs": N_AV_PLANS, "max_ablated_abs_error": max_abs, "max_ablated_rel_error": max_rel, "max_norm_abs_error": max_norm_abs, "max_cosine_abs_error": max_cos_abs}


def _make_plans(freeze: dict[str, Any], vectors: dict[str, np.ndarray], feature_ids: list[int]) -> list[dict[str, Any]]:
    feature_to_vector_index = {feature: index for index, feature in enumerate(feature_ids)}
    plans: list[dict[str, Any]] = []
    for feature_row in freeze.get("features", []):
        feature = _as_int(feature_row.get("feature"), "freeze.feature")
        fi = feature_to_vector_index[feature]
        for di, context in enumerate(feature_row.get("discovery", [])):
            ci = _as_int(context.get("context_index"), "discovery.context_index")
            raw = np.asarray(vectors["residuals"][ci], dtype=np.float32)
            plans.append({
                "feature": feature, "doc_id": _as_int(context.get("doc_id"), "discovery.doc_id"),
                "position": _as_int(context.get("position"), "discovery.position"),
                "role": "discovery", "arm": "NLA_RAW", "control": False,
                "discovery_index": di, "context_text": str(context.get("context_text")),
                "vector_sha256": array_sha256(raw), "vector": raw,
            })
            ablated = np.asarray(vectors["sae_ablated_residuals"][fi, di], dtype=np.float32)
            plans.append({
                "feature": feature, "doc_id": _as_int(context.get("doc_id"), "discovery.doc_id"),
                "position": _as_int(context.get("position"), "discovery.position"),
                "role": "sae_feature_ablated", "arm": "NLA_CONTRASTIVE", "control": False,
                "discovery_index": di, "context_text": str(context.get("context_text")),
                "vector_sha256": array_sha256(ablated), "vector": ablated,
            })
    if len(plans) != N_AV_PLANS:
        raise CheckFailure(f"constructed {len(plans)} AV plans, expected {N_AV_PLANS}")
    return plans


def _check_bindings_and_av(
    freeze: dict[str, Any],
    result: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    freeze_sha: str,
    vectors_sha: str,
    script_sha: str,
) -> dict[str, Any]:
    if "smoke" in str(result.get("status", "")).lower() or "smoke" in str(result.get("run_mode", "")).lower():
        raise CheckFailure("result is a smoke artifact")
    if result.get("status") != "EXPLORATORY_DISCOVERY_AV_COMPLETE":
        raise CheckFailure(f"unexpected result status {result.get('status')!r}")
    if result.get("confirmatory") is not False or result.get("claim_scope") != "discovery_only_no_confirmatory_inference":
        raise CheckFailure("result confirmatory/claim_scope boundary is invalid")
    if result.get("schema_version") != 1:
        raise CheckFailure("unsupported result schema_version")
    if result.get("freeze_sha256") != freeze_sha or result.get("script_sha256") != script_sha:
        raise CheckFailure("result freeze/script binding mismatch")
    if freeze.get("script_sha256") != script_sha:
        raise CheckFailure("freeze script binding mismatch")
    if freeze.get("vectors", {}).get("sha256") != vectors_sha:
        raise CheckFailure("freeze vectors SHA-256 does not match NPZ")
    if result.get("models", {}).get("av_manifest_sha256") is None or not _is_sha256(result.get("models", {}).get("av_manifest_sha256")):
        raise CheckFailure("result model AV manifest hash is missing/malformed")
    if not isinstance(result.get("models"), dict):
        raise CheckFailure("result models binding is missing")
    if len(checkpoint_rows) != N_AV_PLANS:
        raise CheckFailure(f"checkpoint has {len(checkpoint_rows)} rows; expected {N_AV_PLANS}")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != N_AV_PLANS:
        raise CheckFailure(f"result has {len(rows) if isinstance(rows, list) else None} rows; expected {N_AV_PLANS}")
    checkpoint_by_idx: dict[int, dict[str, Any]] = {}
    result_by_idx: dict[int, dict[str, Any]] = {}
    expected_contract = {
        "experiment": "J1 exploratory discovery AV generation",
        "freeze_sha256": freeze_sha,
        "vectors_sha256": vectors_sha,
        "av_manifest_sha256": result["models"]["av_manifest_sha256"],
        "script_sha256": script_sha,
        "pilot_common_sha256": result["models"].get("pilot_common_sha256"),
        "temperature": 0.0,
        "max_new_tokens": 200,
        "include_wdec_control": False,
        "include_contrastive_ablation": True,
        "run_mode": "full_discovery",
        "planned_jobs": N_AV_PLANS,
        "vector_sha_sequence": canonical_sha256([plan["vector_sha256"] for plan in plans]),
    }
    if expected_contract["pilot_common_sha256"] is None:
        raise CheckFailure("result model pilot_common_sha256 binding is missing")
    contract_sha = canonical_sha256(expected_contract)
    if result.get("contract_sha256") != contract_sha:
        raise CheckFailure("result contract_sha256 does not match independently recomputed contract")
    if not isinstance(result.get("generation"), dict) or result["generation"].get("checkpoint_rows") != N_AV_PLANS:
        raise CheckFailure("result generation checkpoint_rows is not 360")
    for source_name, source_rows, destination in (("checkpoint", checkpoint_rows, checkpoint_by_idx), ("result", rows, result_by_idx)):
        for row_number, row in enumerate(source_rows):
            idx = _as_int(row.get("idx"), f"{source_name}[{row_number}].idx")
            if idx in destination or not 0 <= idx < N_AV_PLANS:
                raise CheckFailure(f"{source_name} idx is duplicate/out of range: {idx}")
            destination[idx] = row
    if set(checkpoint_by_idx) != set(range(N_AV_PLANS)) or set(result_by_idx) != set(range(N_AV_PLANS)):
        raise CheckFailure("checkpoint/result indices are not exactly 0..359")
    for idx, plan in enumerate(plans):
        checkpoint = checkpoint_by_idx[idx]
        result_row = result_by_idx[idx]
        if checkpoint.get("contract_sha256") != contract_sha:
            raise CheckFailure(f"checkpoint row {idx} has wrong contract (smoke or mixed run)")
        for row_name, row in (("checkpoint", checkpoint), ("result", result_row)):
            for key in ("feature", "doc_id", "position", "role", "arm", "discovery_index", "control", "vector_sha256"):
                expected = plan[key]
                if str(row.get(key)) != str(expected):
                    raise CheckFailure(f"{row_name} row {idx} {key} mismatch")
            if row.get("context_text") != plan["context_text"] or not isinstance(row.get("context_text"), str) or not row["context_text"].strip():
                raise CheckFailure(f"{row_name} row {idx} context_text is empty/mismatched")
            if row.get("vector_sha256") != plan["vector_sha256"]:
                raise CheckFailure(f"{row_name} row {idx} vector hash mismatch")
        explanation = checkpoint.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise CheckFailure(f"checkpoint row {idx} explanation is empty")
        if result_row.get("explanation") != explanation:
            raise CheckFailure(f"checkpoint/result explanation mismatch at row {idx}")
        explanation_sha = sha256_bytes(explanation.encode("utf-8"))
        if checkpoint.get("explanation_utf8_sha256") != explanation_sha:
            raise CheckFailure(f"checkpoint row {idx} explanation hash mismatch")
        if checkpoint.get("raw_text_sha256") is not None and not _is_sha256(checkpoint.get("raw_text_sha256")):
            raise CheckFailure(f"checkpoint row {idx} raw_text_sha256 malformed")
        if str(result_row.get("raw_text_ref")) != str(plan["doc_id"]):
            raise CheckFailure(f"result row {idx} raw_text_ref mismatch")
    raw_texts = result.get("raw_texts")
    if not isinstance(raw_texts, dict):
        raise CheckFailure("result raw_texts metadata is missing")
    for idx, plan in enumerate(plans):
        key = str(plan["doc_id"])
        entry = raw_texts.get(key)
        if not isinstance(entry, dict) or not _is_sha256(entry.get("sha256")):
            raise CheckFailure(f"result raw_texts lacks hash for doc {key}")
        if not isinstance(entry.get("context_windows"), list) or plan["context_text"] not in entry["context_windows"]:
            raise CheckFailure(f"result raw_texts lacks context window for doc {key}")
        checkpoint = checkpoint_by_idx[idx]
        if checkpoint.get("raw_text_sha256") != entry.get("sha256"):
            raise CheckFailure(f"checkpoint raw_text_sha256 differs from result raw_texts for doc {key}")
    if result.get("models", {}).get("pilot_common_sha256") and not _is_sha256(result["models"]["pilot_common_sha256"]):
        raise CheckFailure("result pilot_common_sha256 is malformed")
    return {"n_plans": len(plans), "n_checkpoint_rows": len(checkpoint_rows), "n_result_rows": len(rows), "contract_sha256": contract_sha}


def _check_model_bindings(freeze: dict[str, Any], result: dict[str, Any], script_path: Path) -> dict[str, Any]:
    freeze_models = freeze.get("models")
    result_models = result.get("models")
    if not isinstance(freeze_models, dict) or not isinstance(result_models, dict):
        raise CheckFailure("freeze/result models metadata is missing")
    for key in ("base_model_manifest_sha256", "sae_manifest_sha256"):
        if not _is_sha256(freeze_models.get(key)):
            raise CheckFailure(f"freeze models.{key} missing/malformed")
        if result_models.get(key) != freeze_models.get(key):
            raise CheckFailure(f"result models.{key} differs from freeze")
    for key in ("av_manifest_sha256", "pilot_common_sha256"):
        if not _is_sha256(result_models.get(key)):
            raise CheckFailure(f"result models.{key} missing/malformed")
    pilot_common_path = script_path.with_name("pilot_common.py")
    pilot_common_checked = False
    if pilot_common_path.is_file():
        pilot_common_checked = True
        if sha256_file(pilot_common_path) != result_models["pilot_common_sha256"]:
            raise CheckFailure("pilot_common.py hash differs from result model binding")
    return {"freeze_base_sae_bindings": True, "av_binding": result_models["av_manifest_sha256"], "pilot_common_checked": pilot_common_checked}


def _markdown_report(payload: dict[str, Any]) -> bytes:
    lines = [
        "# J1 independent audit",
        "",
        f"Status: **{payload.get('status', 'UNKNOWN')}**",
        "",
        "This is a structural/numeric artifact audit only. It makes no semantic judgment and no confirmatory claim.",
        "",
        "## Checks",
        "",
        "| Check | Result | Details |",
        "|---|---|---|",
    ]
    for name, value in payload.get("checks", {}).items():
        result = "PASS" if value.get("pass") else "FAIL"
        details = value.get("error", "")
        if not details:
            compact = {key: item for key, item in value.items() if key != "pass"}
            details = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        details = str(details).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{name}` | {result} | {details} |")
    lines.extend(["", "## Counts", "", "```json", json.dumps(payload.get("counts", {}), ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    if payload.get("errors"):
        lines.extend(["## Errors", "", "```json", json.dumps(payload["errors"], ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    lines.extend(["## Limitations", ""])
    for limitation in payload.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", type=Path, default=None)
    parser.add_argument("--script57", "--script", dest="script57", type=Path, default=Path(__file__).with_name("57_j1_discovery_pilot_gpu.py"))
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", type=Path, default=None)
    parser.add_argument("--vectors", "--vectors-npz", dest="vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--checkpoint", "--av-checkpoint", dest="checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--result", "--full-result", dest="result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--result-sha256", type=Path, default=None)
    parser.add_argument("--sae-params", "--params", "--sae", "--sae-dir", dest="sae_params", type=Path, default=DEFAULT_SAE_PARAMS)
    parser.add_argument("--out", "--out-audit", "--out-json", dest="out", type=Path, default=DEFAULT_RESULTS / "j1_independent_audit_v1.json")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-sha256", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol_sha256 is None:
        args.protocol_sha256 = Path(str(args.protocol) + ".sha256")
    if args.freeze_sha256 is None:
        args.freeze_sha256 = Path(str(args.freeze) + ".sha256")
    if args.result_sha256 is None:
        args.result_sha256 = Path(str(args.result) + ".sha256")
    if args.out_md is None:
        args.out_md = args.out.with_suffix(".md")
    if args.out_sha256 is None:
        args.out_sha256 = Path(str(args.out) + ".sha256")

    audit = Audit()
    paths = {"protocol": args.protocol, "script57": args.script57, "freeze": args.freeze, "vectors": args.vectors, "checkpoint": args.checkpoint, "result": args.result, "sae_params": args.sae_params}
    digests: dict[str, str] = {}
    for name, path in paths.items():
        audit.run(f"input_file.{name}", lambda path=path, name=name: (_require_file(path, name), {"path": str(path), "bytes": path.stat().st_size})[1])
    for name, path in (("protocol", args.protocol), ("freeze", args.freeze), ("result", args.result)):
        sidecar = {"protocol": args.protocol_sha256, "freeze": args.freeze_sha256, "result": args.result_sha256}[name]
        value = audit.run(f"sidecar.{name}", lambda path=path, sidecar=sidecar: {"sha256": verify_sidecar(path, sidecar)})
        if isinstance(value, dict) and isinstance(value.get("sha256"), str):
            digests[name] = value["sha256"]
    script_sha_value = audit.run("script57.hash", lambda: {"sha256": sha256_file(args.script57) if args.script57.is_file() else (_require_file(args.script57, "script57"), "")[1]})
    script_sha = script_sha_value.get("sha256") if isinstance(script_sha_value, dict) else None

    # The protocol is Markdown, not JSON.  Keep its text path for the binding
    # check and parse only the two JSON artifacts here.
    protocol: dict[str, Any] = {}
    freeze_value = audit.run("freeze.parse", lambda: {"value": read_json(args.freeze, "freeze")})
    result_value = audit.run("result.parse", lambda: {"value": read_json(args.result, "result")})
    checkpoint_value = audit.run("checkpoint.parse", lambda: {"rows": read_jsonl(args.checkpoint, "AV checkpoint")})
    freeze = freeze_value.get("value", {}) if isinstance(freeze_value, dict) else {}
    result = result_value.get("value", {}) if isinstance(result_value, dict) else {}
    checkpoint_rows = checkpoint_value.get("rows", []) if isinstance(checkpoint_value, dict) else []
    selected: list[dict[str, Any]] = []
    negative_pairs: list[dict[str, Any]] = []
    freeze_feature_ids: list[int] = []
    vectors: dict[str, np.ndarray] = {}
    w_dec = np.empty((0, 0), dtype=np.float32)
    plans: list[dict[str, Any]] = []

    audit.run("protocol.binding", lambda: _check_protocol(protocol, args.protocol))
    audit.run("freeze.status_and_selection", lambda: _check_freeze_status(freeze))
    flattened = audit.run("freeze.features_contexts", lambda: _flatten_freeze_contexts(freeze))
    if isinstance(flattened, tuple):
        selected, negative_pairs, freeze_feature_ids = flattened
    audit.run("freeze.background_metadata", lambda: _check_background_metadata(freeze, negative_pairs))
    wdec_value = audit.run("sae.w_dec_cpu", lambda: {"shape": list(_load_sae_wdec(args.sae_params).shape)})
    if isinstance(wdec_value, dict):
        try:
            w_dec = _load_sae_wdec(args.sae_params)
        except Exception:
            w_dec = np.empty((0, 0), dtype=np.float32)
    if w_dec.size:
        vectors_value = audit.run("vectors.schema_finite_alignment", lambda: {"keys": sorted(_load_vectors(args.vectors, w_dec, freeze))})
        try:
            vectors = _load_vectors(args.vectors, w_dec, freeze)
        except Exception:
            vectors = {}
    audit.run("activation.metrics", lambda: _check_activation_metrics(freeze, selected, vectors) if vectors else (_ for _ in ()).throw(CheckFailure("vectors unavailable")))
    if vectors and w_dec.size:
        audit.run("hard_negatives", lambda: _check_negatives(negative_pairs, vectors))
        audit.run("sae_ablation.recompute", lambda: _check_ablation(freeze, vectors, w_dec, [int(x) for x in vectors["feature_ids"].tolist()]))
        made_plans = audit.run("av.plan_vectors", lambda: _make_plans(freeze, vectors, [int(x) for x in vectors["feature_ids"].tolist()]))
        if isinstance(made_plans, list):
            plans = made_plans
    audit.run("model.bindings", lambda: _check_model_bindings(freeze, result, args.script57))
    if plans and script_sha:
        audit.run("av.checkpoint_result_bindings", lambda: _check_bindings_and_av(freeze, result, checkpoint_rows, plans, digests.get("freeze", ""), sha256_file(args.vectors) if args.vectors.is_file() else "", script_sha))
    else:
        audit.run("av.checkpoint_result_bindings", lambda: (_ for _ in ()).throw(CheckFailure("AV plans or script hash unavailable")))

    counts = {
        "features": len(freeze_feature_ids),
        "strata": N_STRATA,
        "discovery_contexts": sum(1 for row in selected if row.get("role") == "discovery"),
        "heldout_positive_contexts": sum(1 for row in selected if row.get("role") == "heldout_positive"),
        "hard_negatives": len(negative_pairs),
        "vector_rows": int(vectors.get("residuals", np.empty((0,))).shape[0]) if vectors else 0,
        "av_plans": len(plans),
        "checkpoint_rows": len(checkpoint_rows),
        "result_rows": len(result.get("rows", [])) if isinstance(result.get("rows"), list) else 0,
    }
    max_errors: dict[str, Any] = {}
    for check_name, detail in audit.checks.items():
        if not isinstance(detail, dict):
            continue
        for key, value in detail.items():
            if "error" in key and isinstance(value, (int, float)) and np.isfinite(value):
                max_errors[f"{check_name}.{key}"] = value
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "J1 independent artifact audit",
        "status": "PASS" if not audit.errors else "FAIL",
        "claim_scope": "audit_only_no_semantic_judgment_no_confirmatory_claim",
        "inputs": {name: _artifact_descriptor(path, digests.get(name)) if path.is_file() else {"path": str(path), "missing": True} for name, path in paths.items()},
        "sidecars": {"protocol": str(args.protocol_sha256), "freeze": str(args.freeze_sha256), "result": str(args.result_sha256)},
        "checks": audit.checks,
        "max_errors": max_errors,
        "counts": counts,
        "errors": audit.errors,
        "limitations": [
            "This audit does not prove model-weight provenance, tokenizer provenance, or the semantic truth of any explanation.",
            "The complete unselected background candidate stable-row list is not embedded in the freeze; its declared hash and counts are checked structurally, but the full stable-row hash is not regenerated here.",
            "No GPU, SSH session, AV model, interpreter, or confirmatory endpoint is used.",
        ],
    }
    try:
        json_payload = pretty_json_bytes(payload)
        _write_immutable(args.out, json_payload)
        _write_immutable(args.out_md, _markdown_report(payload))
        _write_immutable_sidecar(args.out_sha256, sha256_bytes(json_payload), args.out)
    except Exception as exc:
        print(f"J1 independent audit output error: {exc}", file=sys.stderr)
        return 2
    print(f"J1 independent audit status={payload['status']} checks={len(audit.checks)} errors={len(audit.errors)} out={args.out}")
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
