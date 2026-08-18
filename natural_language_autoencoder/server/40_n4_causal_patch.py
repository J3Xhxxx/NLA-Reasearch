#!/usr/bin/env python3
"""N4: causal patch evaluation on the frozen, real-text N3 cohort.

This runner deliberately consumes the ``input_ids`` stored in
``acts_L32_n3_v1.parquet``.  It never re-tokenizes text and never applies a chat
template.  Rows from the same document (at most two in the frozen cohort) are
grouped only for checkpointing and document metadata.  Every intervention is
evaluated in its own batch-size-one full-sequence forward so the frozen
activation, clean reference, and identity control use the same CUDA shape.

Expected reconstruction archive keys:

  x, pred_orig, recon_sae_small, recon_sae_big,
  pred_p3_only, pred_p12, and pred_quote_strip_p3 (or quote_strip_p3)

Every non-zero substitute is rescaled to the norm of its target activation.
The primary endpoints are KL(clean || patched) at the patched position, mean KL
over the first 16 affected logits, and cross-entropy on the corresponding 16
next tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM


EXPECTED_N3_ACTIVATION_SHA256 = (
    "eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66"
)


class _StopForward(Exception):
    """Stop immediately after the provenance hook captures layer 32."""


def resolve_layers(model: torch.nn.Module):
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
    ):
        obj = model
        for attribute in path:
            obj = getattr(obj, attribute, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_manifest(path: Path) -> dict[str, str]:
    """Bind the run contract to the local weight manifest (or single weight file)."""
    for name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        candidate = path / name
        if candidate.exists():
            return {"path": name, "sha256": sha256_file(candidate)}
    raise ValueError(f"no model weight manifest or weight file found under {path}")


def contract_digest(contract: dict[str, Any]) -> str:
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_checkpoint(
    path: Path, expected_contract: str, n_rows: int
) -> tuple[dict[int, dict[str, Any]], int]:
    rows: dict[int, dict[str, Any]] = {}
    forwards = 0
    if not path.exists():
        return rows, forwards
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("contract_sha256") != expected_contract:
            raise ValueError(f"checkpoint contract mismatch at line {line_no}")
        forwards += int(record.get("n_forwards", 0))
        for row in record["rows"]:
            index = int(row["idx"])
            if not 0 <= index < n_rows:
                raise ValueError(f"checkpoint row out of range: {index}")
            if index in rows:
                raise ValueError(f"duplicate checkpoint row: {index}")
            rows[index] = row
    return rows, forwards


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def parquet_column(table, name: str, default: Any = None) -> list[Any]:
    if name not in table.column_names:
        return [default] * table.num_rows
    return table.column(name).to_pylist()


def load_activations(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    required = {"activation_vector", "position", "doc_id", "input_ids"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"activation parquet is missing columns: {missing}")

    vectors = np.asarray(
        table.column("activation_vector").combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    input_ids = [
        np.asarray(ids, dtype=np.int64) for ids in table.column("input_ids").to_pylist()
    ]
    return {
        "x": vectors,
        "position": np.asarray(table.column("position").to_pylist(), dtype=np.int64),
        "doc_id": np.asarray(table.column("doc_id").to_pylist(), dtype=np.int64),
        "input_ids": input_ids,
        "token": parquet_column(table, "token"),
        "token_id": parquet_column(table, "token_id"),
        "corpus": parquet_column(table, "corpus"),
        "source": parquet_column(table, "source"),
        "lang": parquet_column(table, "lang"),
    }


def resolve_npz_key(archive, logical_name: str, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in archive.files:
            return candidate
    raise KeyError(
        f"reconstruction archive lacks {logical_name}; tried {list(candidates)}, "
        f"available keys are {archive.files}"
    )


def build_other_indices(
    doc_ids: np.ndarray,
    sources: list[Any],
    corpora: list[Any],
    seed: int,
) -> np.ndarray:
    """Choose a deterministic different-document activation, source-matched first."""
    n = len(doc_ids)
    all_indices = np.arange(n)
    result = np.empty(n, dtype=np.int64)
    for index in range(n):
        tiers = [
            [
                j
                for j in all_indices
                if doc_ids[j] != doc_ids[index] and sources[j] == sources[index]
            ],
            [
                j
                for j in all_indices
                if doc_ids[j] != doc_ids[index] and corpora[j] == corpora[index]
            ],
            [j for j in all_indices if doc_ids[j] != doc_ids[index]],
        ]
        candidates = next((tier for tier in tiers if tier), None)
        if candidates is None:
            raise ValueError("other-activation control needs at least two documents")
        result[index] = candidates[(index + seed) % len(candidates)]
    return result


def norm_match(source: np.ndarray, targets: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target_norm = np.linalg.norm(targets.astype(np.float64), axis=1)
    source_norm = np.linalg.norm(source.astype(np.float64), axis=1)
    if np.any(~np.isfinite(source)) or np.any(~np.isfinite(source_norm)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(source_norm <= 1e-12):
        bad = np.flatnonzero(source_norm <= 1e-12)[:10].tolist()
        raise ValueError(f"{name} has zero-norm rows: {bad}")
    scale = target_norm / source_norm
    return (source * scale[:, None]).astype(np.float32)


@torch.inference_mode()
def capture_provenance(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    ids: torch.Tensor,
    positions: list[int],
) -> np.ndarray:
    """Capture with batch size one, matching N3's extraction batch shape."""
    captured: dict[str, torch.Tensor] = {}
    capture_positions = torch.as_tensor(
        positions, dtype=torch.long, device=ids.device
    )

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["rows"] = (
            hidden[0, capture_positions, :].detach().float().cpu()
        )
        raise _StopForward

    handle = layer.register_forward_hook(hook)
    try:
        try:
            model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
            )
        except _StopForward:
            pass
    finally:
        handle.remove()
    if "rows" not in captured:
        raise RuntimeError("layer hook did not fire during provenance capture")
    return captured["rows"].numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def clean_forward(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    ids: torch.Tensor,
    positions: list[int],
    horizon: int,
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Evaluate the exact frozen batch-size-one input and capture target states."""
    if ids.shape[0] != 1:
        raise ValueError("clean provenance requires batch size one")
    patch_positions = torch.as_tensor(
        positions, dtype=torch.long, device=ids.device
    )
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["rows"] = (
            hidden[0, patch_positions, :].detach().float().cpu()
        )

    handle = layer.register_forward_hook(hook)
    try:
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
    finally:
        handle.remove()

    logits = output.logits
    windows = [
        logits[0, position : position + horizon].float()
        for position in positions
    ]
    result = [window.clone() for window in windows]
    clean_hidden = captured["rows"].numpy().astype(np.float32, copy=False)
    del output, logits, windows, patch_positions
    return result, clean_hidden


@torch.inference_mode()
def patched_forward(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    ids: torch.Tensor,
    position: int,
    vector: np.ndarray,
    horizon: int,
) -> torch.Tensor:
    """Patch one position in one full, frozen batch-size-one input."""
    if ids.shape[0] != 1:
        raise ValueError("patched evaluation requires batch size one")
    patch_vector = torch.as_tensor(vector, dtype=torch.float32, device=ids.device)

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        hidden[0, position, :] = patch_vector.to(hidden.dtype)
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

    logits = output.logits
    window = logits[0, position : position + horizon].float().clone()
    del output, logits, patch_vector
    return window


def prepare_clean_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor | float]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ce = -log_probs.gather(1, targets[:, None]).mean()
    return {"log_probs": log_probs, "probs": probs, "ce": float(ce)}


def score_window(
    clean: dict[str, torch.Tensor | float],
    patched_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    patched_log_probs = torch.log_softmax(patched_logits, dim=-1)
    clean_log_probs = clean["log_probs"]
    clean_probs = clean["probs"]
    assert isinstance(clean_log_probs, torch.Tensor)
    assert isinstance(clean_probs, torch.Tensor)
    kl = (clean_probs * (clean_log_probs - patched_log_probs)).sum(dim=-1)
    patched_ce = -patched_log_probs.gather(1, targets[:, None]).mean()
    return {
        "kl_at_pos": float(kl[0]),
        "kl_mean_first16": float(kl.mean()),
        "ce_first16": float(patched_ce),
        "n_positions": int(len(kl)),
    }


def summarize(rows: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    clean_ce = np.asarray([row["ce_clean_first16"] for row in rows], dtype=float)
    zero_kl_pos = np.asarray(
        [row["results"]["zero"]["kl_at_pos"] for row in rows], dtype=float
    )
    zero_kl_window = np.asarray(
        [row["results"]["zero"]["kl_mean_first16"] for row in rows], dtype=float
    )
    summary: dict[str, Any] = {}
    for condition in conditions:
        kl_pos = np.asarray(
            [row["results"][condition]["kl_at_pos"] for row in rows], dtype=float
        )
        kl_window = np.asarray(
            [row["results"][condition]["kl_mean_first16"] for row in rows],
            dtype=float,
        )
        ce = np.asarray(
            [row["results"][condition]["ce_first16"] for row in rows], dtype=float
        )
        recovered_pos = 1.0 - kl_pos / np.maximum(zero_kl_pos, 1e-6)
        recovered_window = 1.0 - kl_window / np.maximum(zero_kl_window, 1e-6)
        summary[condition] = {
            "n": len(rows),
            "kl_at_pos_mean": float(kl_pos.mean()),
            "kl_at_pos_median": float(np.median(kl_pos)),
            "kl_at_pos_max": float(kl_pos.max()),
            "kl_mean_first16_mean": float(kl_window.mean()),
            "kl_mean_first16_median": float(np.median(kl_window)),
            "ce_first16_mean": float(ce.mean()),
            "ce_first16_delta_from_clean_mean": float((ce - clean_ce).mean()),
            "kl_recovered_at_pos_mean": float(recovered_pos.mean()),
            "kl_recovered_at_pos_median": float(np.median(recovered_pos)),
            "kl_recovered_first16_mean": float(recovered_window.mean()),
            "kl_recovered_first16_median": float(np.median(recovered_window)),
            "kl_recovered_at_pos_ratio_of_sums": (
                float(1.0 - kl_pos.sum() / zero_kl_pos.sum())
                if zero_kl_pos.sum() > 1e-12
                else None
            ),
            "kl_recovered_first16_ratio_of_sums": (
                float(1.0 - kl_window.sum() / zero_kl_window.sum())
                if zero_kl_window.sum() > 1e-12
                else None
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--recon", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layer-index", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--provenance-atol",
        type=float,
        default=0.0,
        help="Maximum captured-vs-frozen absolute error. Default 0 requires bit-exact provenance.",
    )
    parser.add_argument(
        "--identity-kl-tol",
        type=float,
        default=1e-5,
        help="Maximum allowed identity KL at the patched position.",
    )
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if not hasattr(torch, args.dtype):
        raise ValueError(f"torch has no dtype named {args.dtype!r}")

    started = time.time()
    activation_sha256 = sha256_file(args.activations)
    recon_sha256 = sha256_file(args.recon)
    prereg_sha256 = sha256_file(args.prereg)
    script_sha256 = sha256_file(__file__)
    if activation_sha256 != EXPECTED_N3_ACTIVATION_SHA256:
        raise ValueError(
            "activation parquet is not the preregistered N3 cohort: "
            f"expected {EXPECTED_N3_ACTIVATION_SHA256}, got {activation_sha256}"
        )
    data = load_activations(args.activations)
    full_x = data["x"]
    full_n, width = full_x.shape

    with np.load(args.recon, allow_pickle=False) as archive:
        key_map = {
            "orig": resolve_npz_key(archive, "orig", ("pred_orig",)),
            "sae_small": resolve_npz_key(
                archive, "sae_small", ("recon_sae_small",)
            ),
            "sae_big": resolve_npz_key(archive, "sae_big", ("recon_sae_big",)),
            "p3_only": resolve_npz_key(
                archive, "p3_only", ("pred_p3_only", "p3_only")
            ),
            "p12": resolve_npz_key(archive, "p12", ("pred_p12", "p12")),
            "quote_strip_p3": resolve_npz_key(
                archive,
                "quote_strip_p3",
                ("pred_quote_strip_p3", "quote_strip_p3"),
            ),
        }
        archive_x = np.asarray(archive["x"], dtype=np.float32)
        if archive_x.ndim != 2 or archive_x.shape[1] != width:
            raise ValueError(
                f"NPZ x shape {archive_x.shape} is incompatible with d_model={width}"
            )
        archive_n = len(archive_x)
        if archive_n > full_n:
            raise ValueError(
                f"NPZ has {archive_n} rows but the frozen parquet has only {full_n}"
            )
        n = min(args.limit, full_n) if args.limit else full_n
        if n == 0:
            raise ValueError("activation cohort is empty")
        if archive_n < n:
            raise ValueError(
                f"NPZ has only {archive_n} rows, fewer than requested evaluation n={n}"
            )
        if not args.limit and archive_n != full_n:
            raise ValueError(
                f"full run requires {full_n} NPZ rows, found {archive_n}; "
                "pass --limit only for an explicitly limited smoke archive"
            )
        x_max_abs = float(np.max(np.abs(archive_x - full_x[:archive_n])))
        if not np.array_equal(archive_x, full_x[:archive_n]):
            raise ValueError(
                f"NPZ x is not bit-exact with the frozen parquet (max_abs={x_max_abs})"
            )
        if "doc_ids" not in archive.files or "positions" not in archive.files:
            raise ValueError("NPZ lacks doc_ids/positions row-identity arrays")
        archive_doc_ids = np.asarray(archive["doc_ids"], dtype=np.int64)
        archive_positions = np.asarray(archive["positions"], dtype=np.int64)
        if not np.array_equal(archive_doc_ids, data["doc_id"][:archive_n]):
            raise ValueError("NPZ doc_ids are not aligned to the frozen parquet")
        if not np.array_equal(archive_positions, data["position"][:archive_n]):
            raise ValueError("NPZ positions are not aligned to the frozen parquet")
        reconstructed = {
            condition: np.asarray(archive[key], dtype=np.float32)
            for condition, key in key_map.items()
        }

    for condition, vectors in reconstructed.items():
        if vectors.shape != archive_x.shape:
            raise ValueError(
                f"{condition} shape {vectors.shape} != NPZ x shape {archive_x.shape}"
            )

    full_doc_ids = data["doc_id"]
    other_indices = build_other_indices(
        full_doc_ids, data["source"], data["corpus"], args.seed
    )
    rng = np.random.default_rng(args.seed)
    gaussian = rng.standard_normal((n, width)).astype(np.float32)
    dataset_mean = np.repeat(full_x.mean(axis=0, keepdims=True), n, axis=0)

    source_substitutes: dict[str, np.ndarray] = {
        "identity": full_x[:n],
        **{name: vectors[:n] for name, vectors in reconstructed.items()},
        "dataset_mean": dataset_mean,
        "other_activation": full_x[other_indices[:n]],
        "gaussian": gaussian,
    }
    substitutes = {
        name: norm_match(vectors, full_x[:n], name)
        for name, vectors in source_substitutes.items()
    }
    substitutes["zero"] = np.zeros_like(full_x[:n])
    conditions = [
        "identity",
        "orig",
        "sae_small",
        "sae_big",
        "p3_only",
        "p12",
        "quote_strip_p3",
        "dataset_mean",
        "other_activation",
        "gaussian",
        "zero",
    ]

    positions = data["position"][:n]
    doc_ids = data["doc_id"][:n]
    input_ids = data["input_ids"][:n]
    for index, (position, ids) in enumerate(zip(positions, input_ids)):
        if position < 0 or position + args.horizon >= len(ids):
            raise ValueError(
                f"row {index} cannot support {args.horizon} next-token targets: "
                f"position={position}, seq_len={len(ids)}"
            )

    rows_by_doc: dict[int, list[int]] = defaultdict(list)
    canonical_ids: dict[int, np.ndarray] = {}
    for index, (doc_id, ids) in enumerate(zip(doc_ids, input_ids)):
        document = int(doc_id)
        rows_by_doc[document].append(index)
        if document in canonical_ids and not np.array_equal(canonical_ids[document], ids):
            raise ValueError(f"doc_id {document} has inconsistent frozen input_ids")
        canonical_ids[document] = ids
    too_many = {doc: len(rows) for doc, rows in rows_by_doc.items() if len(rows) > 2}
    if too_many:
        raise ValueError(f"N3 invariant violated: more than two rows per document: {too_many}")

    base_config = args.base_model / "config.json"
    if not base_config.exists():
        raise ValueError(f"base model config is missing: {base_config}")
    base_weight_manifest = weight_manifest(args.base_model)
    contract = {
        "activations_sha256": activation_sha256,
        "recon_sha256": recon_sha256,
        "prereg_sha256": prereg_sha256,
        "script_sha256": script_sha256,
        "base_config_sha256": sha256_file(base_config),
        "base_weight_manifest": base_weight_manifest,
        "layer_index": args.layer_index,
        "horizon": args.horizon,
        "dtype": args.dtype,
        "limit": args.limit,
        "seed": args.seed,
        "provenance_atol": args.provenance_atol,
        "identity_kl_tol": args.identity_kl_tol,
        "conditions": conditions,
    }
    run_contract_sha256 = contract_digest(contract)
    checkpoint_rows, checkpoint_forwards = load_checkpoint(
        args.checkpoint, run_contract_sha256, n
    )
    for doc_id, indices in rows_by_doc.items():
        present = [index in checkpoint_rows for index in indices]
        if any(present) and not all(present):
            raise ValueError(f"partial checkpoint document {doc_id}: {present}")

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
        raise ValueError(
            f"layer {args.layer_index} outside model with {len(layers)} layers"
        )
    layer = layers[args.layer_index]
    model_load_seconds = time.time() - load_started

    output_rows: list[dict[str, Any] | None] = [
        checkpoint_rows.get(i) for i in range(n)
    ]
    saved_provenance = [
        row["provenance"] for row in checkpoint_rows.values()
    ]
    provenance_max_abs = max(
        [float(p["max_abs"]) for p in saved_provenance], default=0.0
    )
    provenance_min_cos = min(
        [float(p["cosine"]) for p in saved_provenance], default=1.0
    )
    provenance_all_exact = all(
        bool(p["exact"]) for p in saved_provenance
    )
    clean_eval_all_exact = all(
        bool(p.get("clean_eval_exact_vs_frozen", False))
        for p in saved_provenance
    )
    clean_eval_max_abs = max(
        [
            float(p.get("clean_eval_max_abs_vs_frozen", float("inf")))
            for p in saved_provenance
        ],
        default=0.0,
    )
    forward_count = checkpoint_forwards
    document_items = list(rows_by_doc.items())
    forward_started = time.time()

    for document_ordinal, (doc_id, indices) in enumerate(document_items, start=1):
        if all(output_rows[index] is not None for index in indices):
            print(
                f"[doc {document_ordinal}/{len(document_items)}] id={doc_id} "
                f"rows={len(indices)} checkpoint",
                flush=True,
            )
            continue
        ids_np_full = canonical_ids[doc_id]
        document_positions = [int(positions[index]) for index in indices]
        # Preserve the exact extraction-time sequence shape.  Although a
        # causal mask makes the suffix mathematically irrelevant, changing
        # sequence/batch shape selected a different bf16 CUDA kernel in smoke
        # QA and moved the captured L32 state.
        ids_np = ids_np_full
        evaluated_len = len(ids_np)
        ids = torch.as_tensor(ids_np[None, :], dtype=torch.long, device=device)

        # One exact batch-size-one clean forward supplies both provenance and
        # clean logits.  Each intervention below receives its own independent
        # batch-size-one forward.
        clean_windows, clean_identity = clean_forward(
            model, layer, ids, document_positions, args.horizon
        )
        forward_count += 1
        captured = clean_identity

        clean_metrics = []
        provenance_rows = []
        for local_index, row_index in enumerate(indices):
            frozen = full_x[row_index]
            observed = captured[local_index]
            difference = np.abs(observed - frozen)
            max_abs = float(difference.max())
            exact = bool(np.array_equal(observed, frozen))
            cosine = float(
                observed.astype(np.float64) @ frozen.astype(np.float64)
                / (
                    np.linalg.norm(observed.astype(np.float64))
                    * np.linalg.norm(frozen.astype(np.float64))
                    + 1e-30
                )
            )
            provenance_max_abs = max(provenance_max_abs, max_abs)
            provenance_min_cos = min(provenance_min_cos, cosine)
            provenance_all_exact = provenance_all_exact and exact
            if max_abs > args.provenance_atol:
                raise RuntimeError(
                    f"row {row_index} provenance failed: max_abs={max_abs} "
                    f"> atol={args.provenance_atol}, cos={cosine}"
                )
            clean_observed = clean_identity[local_index]
            clean_difference = np.abs(clean_observed - frozen)
            clean_max_abs = float(clean_difference.max())
            clean_exact = bool(np.array_equal(clean_observed, frozen))
            clean_cosine = float(
                clean_observed.astype(np.float64) @ frozen.astype(np.float64)
                / (
                    np.linalg.norm(clean_observed.astype(np.float64))
                    * np.linalg.norm(frozen.astype(np.float64))
                    + 1e-30
                )
            )
            if not clean_exact:
                raise RuntimeError(
                    f"row {row_index} cropped/repeated clean state differs from "
                    f"the frozen activation: max_abs={clean_max_abs}, "
                    f"cos={clean_cosine}; disable the optimization before launch"
                )
            clean_eval_all_exact = clean_eval_all_exact and clean_exact
            clean_eval_max_abs = max(clean_eval_max_abs, clean_max_abs)
            provenance_rows.append(
                {
                    "exact": exact,
                    "max_abs": max_abs,
                    "cosine": cosine,
                    "clean_eval_exact_vs_frozen": clean_exact,
                    "clean_eval_max_abs_vs_frozen": clean_max_abs,
                    "clean_eval_cosine_vs_frozen": clean_cosine,
                }
            )

            position = int(positions[row_index])
            targets = ids[0, position + 1 : position + 1 + args.horizon]
            if len(targets) != args.horizon:
                raise RuntimeError(f"row {row_index} target window is truncated")
            clean_metrics.append(
                {
                    **prepare_clean_metrics(clean_windows[local_index], targets),
                    "targets": targets,
                }
            )

        condition_scores: dict[str, list[dict[str, float | int]]] = {}
        for condition in conditions:
            condition_scores[condition] = []
            for local_index, row_index in enumerate(indices):
                patched_window = patched_forward(
                    model,
                    layer,
                    ids,
                    int(positions[row_index]),
                    substitutes[condition][row_index],
                    args.horizon,
                )
                forward_count += 1
                condition_scores[condition].append(
                    score_window(
                        clean_metrics[local_index],
                        patched_window,
                        clean_metrics[local_index]["targets"],
                    )
                )
                del patched_window

        for local_index, row_index in enumerate(indices):
            zero_at_pos = float(
                condition_scores["zero"][local_index]["kl_at_pos"]
            )
            zero_first16 = float(
                condition_scores["zero"][local_index]["kl_mean_first16"]
            )
            for condition in conditions:
                result = condition_scores[condition][local_index]
                result["kl_recovered_at_pos_vs_zero"] = float(
                    1.0
                    - float(result["kl_at_pos"]) / max(zero_at_pos, 1e-6)
                )
                result["kl_recovered_first16_vs_zero"] = float(
                    1.0
                    - float(result["kl_mean_first16"])
                    / max(zero_first16, 1e-6)
                )
            output_rows[row_index] = {
                "idx": int(row_index),
                "doc_id": int(doc_ids[row_index]),
                "position": int(positions[row_index]),
                "token": data["token"][row_index],
                "token_id": data["token_id"][row_index],
                "corpus": data["corpus"][row_index],
                "source": data["source"][row_index],
                "lang": data["lang"][row_index],
                "seq_len_frozen": int(len(ids_np_full)),
                "seq_len_evaluated": int(len(ids_np)),
                "x_norm": float(np.linalg.norm(full_x[row_index])),
                "other_activation_idx": int(other_indices[row_index]),
                "provenance": provenance_rows[local_index],
                "ce_clean_first16": float(clean_metrics[local_index]["ce"]),
                "results": {
                    condition: condition_scores[condition][local_index]
                    for condition in conditions
                },
            }

        append_checkpoint(
            args.checkpoint,
            {
                "contract_sha256": run_contract_sha256,
                "doc_id": int(doc_id),
                "n_forwards": 1 + len(conditions) * len(indices),
                "rows": [output_rows[index] for index in indices],
            },
        )
        elapsed = time.time() - forward_started
        rate = elapsed / document_ordinal
        eta = rate * (len(document_items) - document_ordinal)
        orig_kl = condition_scores["orig"][0]["kl_at_pos"]
        sae_big_kl = condition_scores["sae_big"][0]["kl_at_pos"]
        print(
            f"[doc {document_ordinal}/{len(document_items)}] id={doc_id} "
            f"rows={len(indices)} orig_kl={orig_kl:.4g} "
            f"sae_big_kl={sae_big_kl:.4g} eta={eta/60:.1f}m",
            flush=True,
        )
        del clean_windows, clean_identity, clean_metrics, condition_scores, ids

    rows = [row for row in output_rows if row is not None]
    if len(rows) != n:
        raise RuntimeError(f"only produced {len(rows)}/{n} rows")
    summary = summarize(rows, conditions)
    identity_max = summary["identity"]["kl_at_pos_max"]
    if identity_max > args.identity_kl_tol:
        raise RuntimeError(
            f"identity patch KL QA failed: max={identity_max} "
            f"> tolerance={args.identity_kl_tol}"
        )

    payload = {
        "schema_version": 1,
        "experiment": "N4 causal patch on frozen N3 real-text activations",
        "protocol": {
            "layer_index": args.layer_index,
            "horizon": args.horizon,
            "input_contract": "consume frozen input_ids; no retokenization or chat template",
            "suffix_truncation": "none; preserve extraction-time sequence shape",
            "batching": (
                "batch size one; each row and condition receives an independent "
                "full-sequence patched forward"
            ),
            "norm_matching": (
                "all analytical non-zero substitutes rescaled to target norm; "
                "identity patches the frozen activation; the clean evaluation "
                "state is separately required to match it bit-exactly"
            ),
            "kl_direction": "KL(clean || patched)",
            "ce_targets": "the first 16 frozen next tokens after the patched position",
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": activation_sha256,
            "expected_activations_sha256": EXPECTED_N3_ACTIVATION_SHA256,
            "recon": str(args.recon),
            "recon_sha256": recon_sha256,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha256,
            "base_model": str(args.base_model),
            "base_config_sha256": contract["base_config_sha256"],
            "base_weight_manifest": base_weight_manifest,
            "script_sha256": script_sha256,
            "run_contract_sha256": run_contract_sha256,
            "npz_key_map": key_map,
            "npz_x_max_abs_vs_parquet": x_max_abs,
            "npz_doc_ids_match_parquet": True,
            "npz_positions_match_parquet": True,
        },
        "cohort": {
            "n_rows": n,
            "n_rows_full": full_n,
            "n_rows_in_recon_archive": archive_n,
            "n_documents": len(rows_by_doc),
            "limit": args.limit,
            "d_model": width,
            "checkpoint_rows_loaded": len(checkpoint_rows),
        },
        "qa": {
            "provenance_all_bit_exact": provenance_all_exact,
            "provenance_max_abs": provenance_max_abs,
            "provenance_min_cosine": provenance_min_cos,
            "provenance_atol": args.provenance_atol,
            "clean_eval_all_bit_exact_vs_frozen": clean_eval_all_exact,
            "clean_eval_max_abs_vs_frozen": clean_eval_max_abs,
            "identity_kl_at_pos_max": identity_max,
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
        json.dumps(
            payload, ensure_ascii=False, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    output_sha256 = sha256_file(args.out)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{output_sha256}  {args.out.name}\n", encoding="utf-8"
    )

    print("\n=== N4 summary ===")
    print(json.dumps(payload["qa"], ensure_ascii=False))
    for condition in conditions:
        values = summary[condition]
        print(
            f"{condition:<18} KL@pos={values['kl_at_pos_mean']:.5f} "
            f"KL16={values['kl_mean_first16_mean']:.5f} "
            f"CE16={values['ce_first16_mean']:.5f}"
        )
    print(
        f"N4_CAUSAL_PATCH_COMPLETE rows={n} docs={len(rows_by_doc)} "
        f"forwards={forward_count} elapsed={payload['elapsed_seconds']:.1f}s "
        f"sha256={output_sha256} -> {args.out}"
    )


if __name__ == "__main__":
    main()
