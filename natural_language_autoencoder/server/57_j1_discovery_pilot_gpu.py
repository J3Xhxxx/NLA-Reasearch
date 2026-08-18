#!/usr/bin/env python3
"""J1 exploratory/discovery NLA -> SAE pilot (GPU).

This runner is deliberately *not* confirmatory.  It consumes the frozen N3
candidate cohort and feature statistics, freezes a stratified 15 x 3 feature
selection with four document-disjoint discovery contexts and four held-out
positive contexts per feature, then generates provisional NLA snippets from
the discovery residuals.  No AR, judge, causal endpoint, or shutdown action
is present here.

The two model phases are intentionally sequential:

1. Gemma-3-12B-IT + GemmaScope small SAE are loaded on CUDA to reproduce N3
   residuals and to construct the hard-negative pool.  They are completely
   unloaded before phase 2.
2. ``pilot_common.AVLocal`` is loaded on CUDA for greedy temperature-zero AV
   generation.  Explanations are append/fsync checkpointed and resumable.

Every artifact is bound to input/model/script hashes.  A pre-existing freeze,
checkpoint, or result may only be reused if its exact bytes/contracts match;
the script never silently overwrites an artifact from another run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot_common import AVLocal, JumpReLUSAE


SEED = 20260806
CHUNK_LEN = 512
LAYER_INDEX = 32
N_STRATA = 3
PER_STRATUM = 15
N_FEATURES = N_STRATA * PER_STRATUM
N_DISCOVERY = 4
N_HELDOUT = 4
N_CONTEXTS = N_DISCOVERY + N_HELDOUT
DEFAULT_RESULTS = Path("/root/autodl-tmp/results")
STRATA = ("source_concentrated", "source_distributed", "language_selective")


class _StopForward(Exception):
    """Raised by the layer-32 hook for the exact N3 early exit."""


def resolve_layers(model: torch.nn.Module):
    """Resolve Gemma decoder layers across the supported model wrappers."""
    for path in (("model", "layers"), ("language_model", "model", "layers"),
                 ("model", "language_model", "layers")):
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_or_verify_json(path: Path, value: dict[str, Any]) -> str:
    """Write immutable JSON, or require byte-identical exact resume semantics."""
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_bytes()
        if current != data:
            raise RuntimeError(f"refusing to overwrite non-identical frozen artifact: {path}")
    else:
        # Exclusive creation prevents two launchers from racing into a silent
        # replacement.  There is no destructive recovery path here.
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return sha256_bytes(data)


def write_or_verify_sidecar(path: Path, digest: str) -> None:
    data = (digest + "\n").encode("ascii")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"sha256 sidecar mismatch; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_or_verify_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write an NPZ once and require equal arrays on resume.

    ``np.savez`` stores the current ZIP timestamp, so byte equality is not a
    useful resume test.  Existing arrays are compared by key, shape, dtype and
    values; the existing file's bytes remain the immutable hash target.
    """
    import io

    payload = io.BytesIO()
    # ``np.savez`` preserves insertion order, and all keys below are fixed.
    np.savez(payload, **arrays)
    data = payload.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays):
                raise RuntimeError(f"existing vectors keys differ: {path}")
            for key, expected in arrays.items():
                actual = np.asarray(loaded[key])
                expected_arr = np.asarray(expected)
                if actual.shape != expected_arr.shape or actual.dtype != expected_arr.dtype or not np.array_equal(actual, expected_arr):
                    raise RuntimeError(f"refusing to overwrite non-identical vectors: {path} key={key}")
        return sha256_file(path)
    else:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return sha256_bytes(data)


def path_manifest_sha256(path: Path) -> str:
    """Hash a file or a directory's relative file names and contents."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries: list[dict[str, Any]] = []
    for child in sorted((p for p in path.rglob("*") if p.is_file()),
                        key=lambda p: p.relative_to(path).as_posix()):
        entries.append({"path": child.relative_to(path).as_posix(),
                        "sha256": sha256_file(child), "bytes": child.stat().st_size})
    return canonical_sha256({"root": str(path), "files": entries})


def array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return sha256_bytes(arr.astype("<f4", copy=False).tobytes(order="C"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def load_corpus(path: Path) -> dict[int, dict[str, Any]]:
    docs: dict[int, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        for key in ("doc_id", "text", "source", "lang", "corpus"):
            if key not in row:
                raise ValueError(f"corpus row lacks {key!r}")
        doc_id = int(row["doc_id"])
        if doc_id in docs:
            raise ValueError(f"duplicate corpus doc_id={doc_id}")
        if not isinstance(row["text"], str):
            raise ValueError(f"corpus doc_id={doc_id} text is not a string")
        docs[doc_id] = {
            "doc_id": doc_id,
            "text": row["text"],
            "source": str(row["source"]),
            "lang": str(row["lang"]),
            "corpus": str(row["corpus"]),
            "orig_index": row.get("orig_index"),
            "text_sha256": str(row.get("text_sha256") or sha256_bytes(row["text"].encode("utf-8"))),
        }
    if not docs:
        raise ValueError("empty N3 corpus")
    return docs


def load_stats(path: Path) -> dict[str, np.ndarray]:
    required = ("small_top_val", "small_top_meta")
    with np.load(path, allow_pickle=False) as loaded:
        missing = [key for key in required if key not in loaded.files]
        if missing:
            raise ValueError(f"N3 stats missing {missing}")
        stats = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    values = stats["small_top_val"]
    meta = stats["small_top_meta"]
    if values.ndim != 2 or values.shape[1] < N_CONTEXTS:
        raise ValueError(f"small_top_val must be [features, >=8], got {values.shape}")
    if meta.ndim != 3 or meta.shape[:2] != values.shape or meta.shape[2] < 3:
        raise ValueError(f"small_top_meta shape mismatch: {meta.shape} / {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(meta).all():
        raise ValueError("N3 top arrays contain non-finite values")
    return stats


def load_cohort(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list) or len(features) != 120:
        raise ValueError(f"N3 cohort must contain exactly 120 features, got {len(features) if isinstance(features, list) else None}")
    seen: set[int] = set()
    by_stratum = {s: 0 for s in STRATA}
    out: list[dict[str, Any]] = []
    for item in features:
        if not isinstance(item, dict) or "feature" not in item or "stratum" not in item:
            raise ValueError("malformed N3 feature cohort row")
        feature = int(item["feature"])
        stratum = str(item["stratum"])
        if feature in seen:
            raise ValueError(f"duplicate N3 feature {feature}")
        if stratum not in by_stratum:
            raise ValueError(f"unexpected N3 stratum {stratum!r}")
        seen.add(feature)
        by_stratum[stratum] += 1
        out.append(dict(item, feature=feature, stratum=stratum))
    if by_stratum != {s: 40 for s in STRATA}:
        raise ValueError(f"N3 cohort stratum counts must be 40 each, got {by_stratum}")
    return out


def select_features(cohort: list[dict[str, Any]], stats: dict[str, np.ndarray],
                    seed: int = SEED) -> list[dict[str, Any]]:
    if seed != SEED:
        raise ValueError(f"J1 selection seed is fixed at {SEED}")
    meta = stats["small_top_meta"]
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for stratum in STRATA:
        candidates = [row for row in cohort if row["stratum"] == stratum]
        eligible = [row for row in candidates
                    if len({int(x) for x in meta[int(row["feature"]), :, 0]}) >= N_CONTEXTS]
        if len(eligible) < PER_STRATUM:
            raise ValueError(f"stratum {stratum} has only {len(eligible)} eligible features; need {PER_STRATUM}")
        # Sorting before sampling makes the result independent of source JSON
        # order while the fixed RNG makes it reproducible across launches.
        chosen = rng.sample(sorted(eligible, key=lambda row: int(row["feature"])), PER_STRATUM)
        selected.extend(sorted((dict(row) for row in chosen), key=lambda row: int(row["feature"])))
    if len(selected) != N_FEATURES or len({int(row["feature"]) for row in selected}) != N_FEATURES:
        raise AssertionError("J1 feature selection did not produce exactly 45 unique features")
    return selected


def build_contexts(selected: list[dict[str, Any]], stats: dict[str, np.ndarray],
                   docs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    top_vals = stats["small_top_val"]
    top_meta = stats["small_top_meta"]
    contexts: list[dict[str, Any]] = []
    for feature_row in selected:
        feature = int(feature_row["feature"])
        seen_docs: set[int] = set()
        picked: list[dict[str, Any]] = []
        for rank in range(top_meta.shape[1]):
            doc_id = int(top_meta[feature, rank, 0])
            position = int(top_meta[feature, rank, 1])
            token_id = int(top_meta[feature, rank, 2])
            value = float(top_vals[feature, rank])
            if doc_id not in docs or doc_id in seen_docs:
                continue
            if position < 0 or not np.isfinite(value) or value <= 0:
                continue
            seen_docs.add(doc_id)
            picked.append({
                "feature": feature,
                "stratum": str(feature_row["stratum"]),
                "rank": rank,
                "doc_id": doc_id,
                "position": position,
                "token_id": token_id,
                "expected_activation": value,
                "source": docs[doc_id]["source"],
                "lang": docs[doc_id]["lang"],
                "corpus": docs[doc_id]["corpus"],
                "raw_text_sha256": docs[doc_id]["text_sha256"],
            })
            if len(picked) == N_CONTEXTS:
                break
        if len(picked) != N_CONTEXTS or len({int(x["doc_id"]) for x in picked}) != N_CONTEXTS:
            raise ValueError(f"feature {feature} lacks 8 distinct valid top contexts")
        for i, row in enumerate(picked):
            row["role"] = "discovery" if i < N_DISCOVERY else "heldout_positive"
            row["context_index"] = len(contexts)
            row["within_feature_index"] = i
            contexts.append(row)
    if len(contexts) != N_FEATURES * N_CONTEXTS:
        raise AssertionError("J1 context count mismatch")
    return contexts


def _background_group_seed(group: tuple[str, str, str], seed: int = SEED) -> int:
    """Derive an auditable deterministic seed for one corpus/source/lang group."""
    if seed != SEED:
        raise ValueError(f"background seed is fixed at {SEED}")
    return int(canonical_sha256({"seed": seed, "group": list(group)})[:16], 16)


def build_background_candidates(*, base_model: str, contexts: list[dict[str, Any]],
                                docs: dict[int, dict[str, Any]],
                                seed: int = SEED, requested_per_group: int = 32,
                                max_per_doc: int = 2, min_per_group: int = 4
                                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic raw-token background candidates before model forward.

    Background rows are deliberately not selected features.  They are only a
    candidate pool for strict hard-negative matching.  We take at most two
    positions per document and make a first pass over distinct documents
    before a second position pass, so a large document cannot crowd out the
    group.  A group may realize fewer than 32 rows, but fewer than four is a
    protocol failure; the union of all groups supplies the tier-2 fallback.
    """
    if seed != SEED:
        raise ValueError(f"background seed is fixed at {SEED}")
    if requested_per_group != 32 or max_per_doc != 2 or min_per_group != 4:
        raise ValueError("J1 background construction constants are fixed (32/2/4)")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    special_ids = {int(x) for x in getattr(tok, "all_special_ids", [])}
    existing_positions = {(int(row["doc_id"]), int(row["position"])) for row in contexts
                          if row.get("role") in ("discovery", "heldout_positive")}
    heldout_groups = sorted({(str(row["source"]), str(row["lang"]), str(row["corpus"]))
                             for row in contexts if row.get("role") == "heldout_positive"})
    docs_by_group: dict[tuple[str, str, str], list[int]] = {}
    for doc_id, doc in docs.items():
        group = (str(doc["source"]), str(doc["lang"]), str(doc["corpus"]))
        if group in heldout_groups:
            docs_by_group.setdefault(group, []).append(int(doc_id))

    rows: list[dict[str, Any]] = []
    by_group: dict[str, int] = {}
    group_details: dict[str, Any] = {}
    for group in heldout_groups:
        group_key = "|".join(group)
        group_rng = random.Random(_background_group_seed(group, seed))
        doc_order = sorted(docs_by_group.get(group, []))
        group_rng.shuffle(doc_order)
        per_doc: list[tuple[int, list[tuple[int, int]]]] = []
        for doc_id in doc_order:
            ids = _tokenize_full(tok, docs[doc_id]["text"])
            valid: list[tuple[int, int]] = []
            # >=16 prefix and >=6 suffix are measured in full-document token
            # positions; all positions are raw/no-chat and special/blank tokens
            # are excluded.
            for position in range(16, max(16, len(ids) - 6)):
                if position + 6 >= len(ids) or (doc_id, position) in existing_positions:
                    continue
                token_id = int(ids[position])
                if token_id in special_ids:
                    continue
                token = tok.decode([token_id], skip_special_tokens=False)
                if not token or not token.strip():
                    continue
                valid.append((position, token_id))
            group_rng.shuffle(valid)
            if valid:
                per_doc.append((doc_id, valid[:max_per_doc]))
                if len(per_doc) >= requested_per_group:
                    # The first pass will already provide 32 distinct docs;
                    # avoid tokenizing the remainder of a large corpus group.
                    break
        picks: list[tuple[int, int, int]] = []
        # First one position per unique document, then an optional second.
        for pass_index in range(max_per_doc):
            for doc_id, valid in per_doc:
                if len(picks) >= requested_per_group:
                    break
                if pass_index < len(valid):
                    position, token_id = valid[pass_index]
                    picks.append((doc_id, position, token_id))
            if len(picks) >= requested_per_group:
                break
        if len(picks) < min_per_group:
            raise ValueError(f"background group {group_key} realized {len(picks)} < {min_per_group}")
        start_index = len(contexts) + len(rows)
        for local_index, (doc_id, position, token_id) in enumerate(picks):
            rows.append({
                "feature": None, "origin_feature": None,
                "role": "background_negative_candidate",
                "background_group": group_key,
                "context_index": start_index + local_index,
                "within_group_index": local_index,
                "doc_id": int(doc_id), "position": int(position),
                "token_id": int(token_id),
                "source": group[0], "lang": group[1], "corpus": group[2],
                "expected_activation": None,
                "raw_text_sha256": docs[doc_id]["text_sha256"],
            })
        by_group[group_key] = len(picks)
        group_details[group_key] = {
            "source": group[0], "lang": group[1], "corpus": group[2],
            "seed": _background_group_seed(group, seed),
            "requested": requested_per_group, "realized": len(picks),
            "n_docs_with_candidates": len(per_doc),
            "max_per_doc": max_per_doc, "min_required": min_per_group,
        }
    stable_rows = [{key: row[key] for key in
                    ("background_group", "context_index", "doc_id", "position", "token_id",
                     "source", "lang", "corpus", "role")}
                   for row in rows]
    metadata = {
        "construction": "same base tokenizer, raw add_special_tokens=True, no chat; >=16 prefix, >=6 suffix, non-special/non-blank; one pass per doc then second; max 2/doc",
        "seed": seed, "requested_per_group": requested_per_group,
        "max_per_doc": max_per_doc, "min_per_group": min_per_group,
        "total": len(rows), "by_group": by_group, "groups": group_details,
        "sha256": canonical_sha256(stable_rows),
    }
    return rows, metadata


def _tokenize_full(tok: Any, text: str) -> list[int]:
    # This is intentionally the same call as N3: full document, raw text,
    # add_special_tokens=True, with no chat template and no truncation.
    encoded = tok(text, add_special_tokens=True)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _context_text(tok: Any, ids: list[int], position: int) -> tuple[str, str, str, str]:
    token = tok.decode([ids[position]], skip_special_tokens=False)
    before = tok.decode(ids[max(0, position - 24):position], skip_special_tokens=False)
    after = tok.decode(ids[position + 1:position + 7], skip_special_tokens=False)
    return before + token + after, token, before, after


class ExtractionVerificationError(RuntimeError):
    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__(f"N3 extraction verification failed with {len(errors)} error(s)")
        self.errors = errors


def extract_candidate_vectors(*, contexts: list[dict[str, Any]], docs: dict[int, dict[str, Any]],
                               base_model: str, sae_dir: str, layer_index: int = LAYER_INDEX,
                               batch_size: int = 8, rtol: float = 2.5e-2,
                               atol: float = 1.0, include_contrastive_ablation: bool = True
                               ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Reproduce N3's raw 512-token chunk extraction and encode every target."""
    if layer_index != LAYER_INDEX:
        raise ValueError(f"J1 requires frozen N3 layer {LAYER_INDEX}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map={"": "cuda"},
        trust_remote_code=True).eval()
    device = torch.device("cuda")
    layers = resolve_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer {layer_index} not available (n={len(layers)})")
    sae = JumpReLUSAE(sae_dir, device="cuda", dtype=torch.float32)
    if sae.width != 16384:
        raise ValueError(f"J1 requires width-16k small SAE, got {sae.width}")

    # Tokenize each selected document once, then deduplicate the 512-token
    # chunks.  The target position always remains the full-document position.
    token_cache: dict[int, list[int]] = {}
    work: dict[tuple[int, int], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for row in contexts:
        doc_id = int(row["doc_id"])
        if doc_id not in token_cache:
            token_cache[doc_id] = _tokenize_full(tok, docs[doc_id]["text"])
        ids = token_cache[doc_id]
        position = int(row["position"])
        if position >= len(ids):
            errors.append({"kind": "position_out_of_range", "feature": row.get("feature"),
                           "doc_id": doc_id, "position": position, "token_count": len(ids)})
            continue
        actual_token_id = int(ids[position])
        if actual_token_id != int(row["token_id"]):
            errors.append({"kind": "token_id_mismatch", "feature": row.get("feature"),
                           "doc_id": doc_id, "position": position,
                           "stored": int(row["token_id"]), "actual": actual_token_id})
            continue
        chunk_start = (position // CHUNK_LEN) * CHUNK_LEN
        piece = ids[chunk_start:chunk_start + CHUNK_LEN]
        local = position - chunk_start
        if not piece or local < 0 or local >= len(piece) or len(piece) > CHUNK_LEN:
            errors.append({"kind": "invalid_chunk_position", "feature": row.get("feature"),
                           "doc_id": doc_id, "position": position,
                           "chunk_start": chunk_start, "chunk_len": len(piece)})
            continue
        key = (doc_id, chunk_start)
        item = work.setdefault(key, {"doc_id": doc_id, "chunk_start": chunk_start,
                                     "ids": piece, "targets": []})
        item["targets"].append((int(row["context_index"]), local))
        context_text, token, before, after = _context_text(tok, ids, position)
        row["chunk_start"] = chunk_start
        row["chunk_position"] = local
        row["token"] = token
        row["before"] = before
        row["after"] = after
        row["context_text"] = context_text
        row["seq_len"] = len(ids)

    if errors:
        # Do not load/continue into AV after a token/position mismatch.
        raise ExtractionVerificationError(errors)
    if not work:
        raise ValueError("no valid N3 chunks to extract")

    residuals = np.empty((len(contexts), sae.d_model), dtype=np.float32)
    acts = np.empty((len(contexts), sae.width), dtype=np.float32)
    grab: dict[str, torch.Tensor] = {}
    hook_calls = 0

    def hook(_module: Any, _inputs: Any, output: Any):
        nonlocal hook_calls
        grab["hidden"] = output[0] if isinstance(output, tuple) else output
        hook_calls += 1
        raise _StopForward

    handle = layers[layer_index].register_forward_hook(hook)
    work_values = list(work.values())
    try:
        for start in range(0, len(work_values), batch_size):
            batch = work_values[start:start + batch_size]
            max_len = max(len(item["ids"]) for item in batch)
            pad_id = tok.pad_token_id
            if pad_id is None:
                pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for i, item in enumerate(batch):
                seq = torch.tensor(item["ids"], dtype=torch.long)
                input_ids[i, :len(seq)] = seq
                attention[i, :len(seq)] = 1
            input_ids = input_ids.to(device)
            attention = attention.to(device)
            with torch.inference_mode():
                try:
                    model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                except _StopForward:
                    pass
                if "hidden" not in grab:
                    raise RuntimeError("layer-32 hook did not fire")
                hidden = grab.pop("hidden").float()
                if hidden.ndim != 3 or hidden.shape[0] != len(batch):
                    raise RuntimeError(f"unexpected layer-32 hidden shape {tuple(hidden.shape)}")
                flat = hidden.reshape(-1, hidden.shape[-1])
                _, encoded = sae(flat)
                for i, item in enumerate(batch):
                    for context_index, local in item["targets"]:
                        flat_index = i * max_len + local
                        residual = flat[flat_index].detach().float().cpu().numpy()
                        activation = encoded[flat_index].detach().float().cpu().numpy()
                        residuals[context_index] = residual
                        acts[context_index] = activation
            del input_ids, attention, hidden, flat, encoded
    finally:
        handle.remove()

    expected_batches = (len(work_values) + batch_size - 1) // batch_size
    if hook_calls != expected_batches:
        raise RuntimeError(f"layer-32 hook call count {hook_calls} != batches {expected_batches}")
    if not np.isfinite(residuals).all() or not np.isfinite(acts).all():
        raise RuntimeError("N3 residual/SAE arrays contain non-finite values")

    # Verify every selected feature's target activation against the frozen N3
    # top value.  Errors are collected before failing closed so the operator
    # has a useful audit record rather than an unexplained assertion.
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    verification_rows = [row for row in contexts
                         if row.get("role") in ("discovery", "heldout_positive")
                         and row.get("expected_activation") is not None]
    for row in verification_rows:
        index = int(row["context_index"])
        feature = int(row["feature"])
        actual = float(acts[index, feature])
        expected = float(row["expected_activation"])
        absolute_error = abs(actual - expected)
        relative_error = absolute_error / max(abs(expected), 1e-12)
        row["actual_activation"] = actual
        row["activation_absolute_error"] = absolute_error
        row["activation_relative_error"] = relative_error
        row["residual_norm"] = float(np.linalg.norm(residuals[index]))
        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)
        if not np.isclose(actual, expected, rtol=rtol, atol=atol):
            errors.append({"kind": "activation_mismatch", "context_index": index,
                           "feature": feature, "doc_id": int(row["doc_id"]),
                           "position": int(row["position"]), "expected": expected,
                           "actual": actual, "abs_error": absolute_error,
                           "rel_error": relative_error,
                           "rtol": rtol, "atol": atol})
    if errors:
        raise ExtractionVerificationError(errors)

    feature_ids = sorted({int(row["feature"]) for row in verification_rows})
    # Keep decoder controls on CPU before the SAE is unloaded.  A decoder
    # direction is scaled to the mean discovery residual norm and marked as
    # off-manifold; it is never mixed with ordinary discovery vectors.
    control = np.empty((len(feature_ids), N_DISCOVERY, sae.d_model), dtype=np.float32)
    ablated = np.empty((len(feature_ids), N_DISCOVERY, sae.d_model), dtype=np.float32)
    ablation_activation = np.empty((len(feature_ids), N_DISCOVERY), dtype=np.float32)
    ablation_norm = np.empty((len(feature_ids), N_DISCOVERY, 2), dtype=np.float32)
    ablation_cosine = np.empty((len(feature_ids), N_DISCOVERY), dtype=np.float32)
    for fi, feature in enumerate(feature_ids):
        dvec = sae.w_dec[feature].detach().float().cpu().numpy()
        discovery_rows = [row for row in verification_rows
                          if int(row["feature"]) == feature and row["role"] == "discovery"]
        discovery_norms = [float(np.linalg.norm(residuals[int(row["context_index"])]))
                           for row in discovery_rows]
        scale = float(np.mean(discovery_norms)) / max(float(np.linalg.norm(dvec)), 1e-12)
        control[fi, :, :] = dvec[None, :] * scale
        for di, row in enumerate(discovery_rows):
            ci = int(row["context_index"])
            af = float(acts[ci, feature])
            x = residuals[ci]
            x_minus = x - af * dvec
            ablation_activation[fi, di] = af
            ablated[fi, di] = x_minus
            ablation_norm[fi, di, 0] = float(np.linalg.norm(x))
            ablation_norm[fi, di, 1] = float(np.linalg.norm(x_minus))
            denom = max(float(np.linalg.norm(x)) * float(np.linalg.norm(x_minus)), 1e-12)
            ablation_cosine[fi, di] = float(np.dot(x, x_minus) / denom)
    if not np.isfinite(control).all() or not np.isfinite(ablated).all() or not np.isfinite(ablation_activation).all() or not np.isfinite(ablation_norm).all() or not np.isfinite(ablation_cosine).all():
        raise RuntimeError("w_dec/control or SAE-ablation vectors contain non-finite values")
    metadata = {
        "base_model_manifest_sha256": path_manifest_sha256(Path(base_model)),
        "sae_manifest_sha256": path_manifest_sha256(Path(sae_dir)),
        "hook_calls": str(hook_calls),
        "expected_batches": str(expected_batches),
        "n_chunks": str(len(work_values)),
        "activation_verification": {
            "n_contexts": len(verification_rows),
            "rtol": float(rtol),
            "atol": float(atol),
            "max_absolute_error": float(np.max(absolute_errors)),
            "max_relative_error": float(np.max(relative_errors)),
            "absolute_error_quantiles": {
                str(q): float(np.quantile(absolute_errors, q))
                for q in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "relative_error_quantiles": {
                str(q): float(np.quantile(relative_errors, q))
                for q in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "firing_sign_mismatches": int(sum(
                (float(row["expected_activation"]) > 0)
                != (float(row["actual_activation"]) > 0)
                for row in verification_rows
            )),
        },
    }
    vectors = {"residuals": residuals, "sae_acts": acts, "wdec_controls": control,
               "sae_ablated_residuals": ablated,
               "contrastive_activation": ablation_activation,
               "contrastive_norms": ablation_norm,
               "contrastive_cosine": ablation_cosine,
               "context_indices": np.arange(len(contexts), dtype=np.int64),
               "feature_ids": np.asarray(feature_ids, dtype=np.int64)}
    if not include_contrastive_ablation:
        # Keep the arrays in the freeze schema (zero rows are not useful as an
        # AV arm, but retaining a fixed schema makes accidental arm mixing
        # impossible).  The default protocol always enables this arm.
        vectors["sae_ablated_residuals"] = np.zeros_like(ablated)
        vectors["contrastive_activation"] = np.zeros_like(ablation_activation)
        vectors["contrastive_norms"] = np.zeros_like(ablation_norm)
        vectors["contrastive_cosine"] = np.zeros_like(ablation_cosine)
    return contexts, vectors, metadata


def assign_hard_negatives(*, contexts: list[dict[str, Any]], residuals: np.ndarray,
                          acts: np.ndarray) -> list[dict[str, Any]]:
    """Assign one exact-zero, non-reused candidate to every held-out positive.

    Matching is strictly within the three preregistered preference tiers:
    same source+language, same source, or same corpus+language.  A fourth
    "anything" tier would silently weaken the hard-negative contrast and is
    therefore not allowed.  Within a tier the candidate is closest to the
    positive in log residual norm and normalized position, with stable integer
    tie breakers.
    """
    if residuals.ndim != 2 or acts.ndim != 2 or residuals.shape[0] != len(contexts) or acts.shape[0] != len(contexts):
        raise ValueError("candidate residual/SAE arrays do not align with contexts")
    by_feature: dict[int, list[dict[str, Any]]] = {}
    for row in contexts:
        if row.get("role") in ("discovery", "heldout_positive"):
            by_feature.setdefault(int(row["feature"]), []).append(row)
    norms = np.linalg.norm(residuals, axis=1)
    negatives: list[dict[str, Any]] = []
    for feature in sorted(by_feature):
        feature_contexts = by_feature[feature]
        excluded = {int(row["doc_id"]) for row in feature_contexts}
        used_candidate_positions: set[tuple[int, int]] = set()
        positives = sorted((row for row in feature_contexts if row["role"] == "heldout_positive"),
                          key=lambda row: int(row["within_feature_index"]))
        for positive in positives:
            positive_index = int(positive["context_index"])
            positive_norm = max(float(norms[positive_index]), 1e-12)
            positive_position = int(positive["position"])
            candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            for candidate in contexts:
                ci = int(candidate["context_index"])
                if int(candidate["doc_id"]) in excluded:
                    continue
                candidate_position = (int(candidate["doc_id"]), int(candidate["position"]))
                if candidate_position in used_candidate_positions:
                    continue
                if float(acts[ci, feature]) != 0.0:
                    continue
                if not np.isfinite(norms[ci]) or not np.isfinite(residuals[ci]).all():
                    continue
                if (candidate["source"], candidate["lang"]) == (positive["source"], positive["lang"]):
                    tier = 0
                elif candidate["source"] == positive["source"]:
                    tier = 1
                elif (candidate["corpus"], candidate["lang"]) == (positive["corpus"], positive["lang"]):
                    tier = 2
                else:
                    continue
                norm_delta = abs(float(np.log(max(float(norms[ci]), 1e-12)) - np.log(positive_norm)))
                position_delta = abs(int(candidate["position"]) - positive_position) / max(
                    1.0, float(max(int(candidate["position"]), positive_position)))
                key = (tier, norm_delta + position_delta, norm_delta, position_delta,
                       int(candidate["position"]), int(candidate["doc_id"]), ci)
                candidates.append((key, candidate))
            if not candidates:
                raise ValueError(f"fewer than four unique strict-tier exact-zero hard negatives for feature {feature}; "
                                 f"heldout doc={positive['doc_id']} position={positive['position']}")
            _, chosen = min(candidates, key=lambda item: item[0])
            ci = int(chosen["context_index"])
            used_candidate_positions.add((int(chosen["doc_id"]), int(chosen["position"])))
            chosen_tier = (0 if (chosen["source"], chosen["lang"]) == (positive["source"], positive["lang"])
                           else 1 if chosen["source"] == positive["source"] else 2)
            negatives.append({
                "feature": feature,
                "target_positive_context_index": positive_index,
                "candidate_context_index": ci,
                "doc_id": int(chosen["doc_id"]),
                "position": int(chosen["position"]),
                "token_id": int(chosen["token_id"]),
                "token": chosen.get("token", ""),
                "before": chosen.get("before", ""),
                "after": chosen.get("after", ""),
                "context_text": chosen.get("context_text", ""),
                "raw_text_sha256": chosen.get("raw_text_sha256", ""),
                "seq_len": int(chosen.get("seq_len", 0)),
                "source": chosen["source"], "lang": chosen["lang"], "corpus": chosen["corpus"],
                "target_activation": float(acts[ci, feature]),
                "residual_norm": float(norms[ci]),
                "norm_log_distance": float(abs(np.log(max(float(norms[ci]), 1e-12)) - np.log(positive_norm))),
                "position_distance": float(abs(int(chosen["position"]) - positive_position) /
                                            max(1.0, float(max(int(chosen["position"]), positive_position)))),
                "preference_tier": int(chosen_tier),
            })
    if len(negatives) != N_FEATURES * N_HELDOUT:
        raise AssertionError("J1 hard-negative count mismatch")
    if any(float(row["target_activation"]) != 0.0 for row in negatives):
        raise AssertionError("hard-negative target activation is not exactly zero")
    return negatives


def _freeze_payload(*, selected: list[dict[str, Any]], contexts: list[dict[str, Any]],
                    negatives: list[dict[str, Any]], input_hashes: dict[str, str],
                    model_hashes: dict[str, str], vectors_sha256: str,
                    vector_shapes: dict[str, list[int]], script_sha256: str,
                    activation_verification: dict[str, Any],
                    background_meta: dict[str, Any],
                    include_wdec_control: bool,
                    include_contrastive_ablation: bool) -> dict[str, Any]:
    neg_by_positive = {int(row["target_positive_context_index"]): row for row in negatives}
    feature_rows: list[dict[str, Any]] = []
    for feature_row in selected:
        feature = int(feature_row["feature"])
        feature_contexts = [row for row in contexts
                            if row.get("role") in ("discovery", "heldout_positive")
                            and int(row["feature"]) == feature]
        discovery = [dict(row) for row in feature_contexts if row["role"] == "discovery"]
        heldout = [dict(row) for row in feature_contexts if row["role"] == "heldout_positive"]
        feature_rows.append({
            "feature": feature,
            "stratum": str(feature_row["stratum"]),
            "discovery": discovery,
            "heldout_positive": [dict(row, hard_negative=neg_by_positive[int(row["context_index"])])
                                  for row in heldout],
        })
    payload = {
        "schema_version": 1,
        "experiment": "J1 NLA-to-SAE exploratory discovery pilot",
        "status": "EXPLORATORY_DISCOVERY_FROZEN_BEFORE_AV",
        "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "itt": {"n_features": N_FEATURES, "n_strata": N_STRATA,
                 "per_stratum": PER_STRATUM, "recorded": 45},
        "selection": {"seed": SEED, "strata": list(STRATA),
                       "per_stratum": PER_STRATUM, "n_features": N_FEATURES,
                       "eligibility": "small_top_meta has >=8 unique doc_ids",
                       "contexts_per_feature": N_CONTEXTS,
                       "discovery_per_feature": N_DISCOVERY,
                       "heldout_positive_per_feature": N_HELDOUT,
                       "document_disjoint_within_feature": True},
        "background_pool": background_meta,
        "inputs": input_hashes,
        "models": model_hashes,
        "script_sha256": script_sha256,
        "include_wdec_control": bool(include_wdec_control),
        "include_contrastive_ablation": bool(include_contrastive_ablation),
        "vectors": {"path_role": "frozen_before_AV", "sha256": vectors_sha256,
                     "shapes": vector_shapes},
        "features": feature_rows,
        "checks": {"n_selected": N_FEATURES,
                   "n_discovery_contexts": N_FEATURES * N_DISCOVERY,
                   "n_heldout_contexts": N_FEATURES * N_HELDOUT,
                   "n_hard_negatives": N_FEATURES * N_HELDOUT,
                   "activation_verification_errors": [],
                   "activation_verification": activation_verification},
    }
    payload["freeze_contract_sha256"] = canonical_sha256({
        "selection": payload["selection"], "inputs": payload["inputs"],
        "background_pool": payload["background_pool"],
        "models": payload["models"], "script_sha256": script_sha256,
        "vectors_sha256": vectors_sha256, "include_wdec_control": bool(include_wdec_control),
        "include_contrastive_ablation": bool(include_contrastive_ablation),
    })
    return payload


def validate_freeze_semantics(freeze: dict[str, Any]) -> None:
    """Fail closed if an immutable freeze is structurally incomplete."""
    background = freeze.get("background_pool")
    if not isinstance(background, dict) or "total" not in background or "by_group" not in background or "sha256" not in background:
        raise ValueError("freeze lacks background-pool construction metadata")
    if int(background.get("total", -1)) < 0:
        raise ValueError("background-pool total is invalid")
    features = freeze.get("features")
    if not isinstance(features, list) or len(features) != N_FEATURES:
        raise ValueError("freeze must contain exactly 45 feature rows")
    seen_features: set[int] = set()
    seen_context_indices: set[int] = set()
    n_negatives = 0
    for feature_row in features:
        feature = int(feature_row.get("feature", -1))
        if feature in seen_features:
            raise ValueError(f"duplicate freeze feature {feature}")
        seen_features.add(feature)
        discovery = feature_row.get("discovery")
        heldout = feature_row.get("heldout_positive")
        if not isinstance(discovery, list) or not isinstance(heldout, list) or len(discovery) != N_DISCOVERY or len(heldout) != N_HELDOUT:
            raise ValueError(f"feature {feature} does not have 4+4 contexts")
        all_contexts = discovery + heldout
        docs_for_feature = [int(row["doc_id"]) for row in all_contexts]
        if len(set(docs_for_feature)) != N_CONTEXTS:
            raise ValueError(f"feature {feature} discovery/heldout docs overlap")
        negative_positions: set[tuple[int, int]] = set()
        for context in all_contexts:
            ci = int(context["context_index"])
            if ci in seen_context_indices:
                raise ValueError(f"duplicate freeze context_index {ci}")
            seen_context_indices.add(ci)
        for positive in heldout:
            negative = positive.get("hard_negative")
            if not isinstance(negative, dict):
                raise ValueError(f"feature {feature} heldout positive lacks hard negative")
            if int(negative["doc_id"]) in set(docs_for_feature):
                raise ValueError(f"feature {feature} hard negative reuses a feature document")
            negative_position = (int(negative["doc_id"]), int(negative["position"]))
            if negative_position in negative_positions:
                raise ValueError(f"feature {feature} hard negative physical position is reused")
            negative_positions.add(negative_position)
            if float(negative.get("target_activation", 1.0)) != 0.0:
                raise ValueError(f"feature {feature} hard negative is not exact-zero")
            if int(negative.get("preference_tier", -1)) not in (0, 1, 2):
                raise ValueError(f"feature {feature} hard negative has invalid preference tier")
            n_negatives += 1
    if n_negatives != N_FEATURES * N_HELDOUT or len(seen_context_indices) != N_FEATURES * N_CONTEXTS:
        raise ValueError("freeze context/negative totals are inconsistent")


def _smoke_path(path: Path, jobs: int) -> Path:
    return path.with_name(f"{path.stem}.smoke{jobs}{path.suffix}")


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path, *, contract_sha256: str, plans: list[dict[str, Any]],
                    limit: int) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid AV checkpoint line {line_number}: {exc}") from exc
            if row.get("contract_sha256") != contract_sha256:
                raise ValueError(f"AV checkpoint contract mismatch at line {line_number}")
            idx = int(row.get("idx", -1))
            if not 0 <= idx < limit:
                raise ValueError(f"AV checkpoint idx out of range at line {line_number}: {idx}")
            if idx in completed:
                raise ValueError(f"duplicate AV checkpoint idx {idx}")
            plan = plans[idx]
            for key in ("feature", "doc_id", "position", "role", "arm", "vector_sha256"):
                if str(row.get(key)) != str(plan.get(key)):
                    raise ValueError(f"AV checkpoint row {idx} mismatch in {key}")
            completed[idx] = row
    return completed


def build_av_plans(*, freeze: dict[str, Any], vectors: dict[str, np.ndarray],
                   docs: dict[int, dict[str, Any]], include_wdec_control: bool,
                   include_contrastive_ablation: bool) -> list[dict[str, Any]]:
    residuals = vectors["residuals"]
    controls = vectors["wdec_controls"]
    ablated = vectors["sae_ablated_residuals"]
    feature_ids = [int(x) for x in np.asarray(vectors["feature_ids"]).tolist()]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("frozen feature_ids contain duplicates")
    feature_to_vector_row = {feature: index for index, feature in enumerate(feature_ids)}
    if set(feature_to_vector_row) != {int(row["feature"]) for row in freeze["features"]}:
        raise ValueError("frozen feature_ids do not exactly match freeze feature rows")
    plans: list[dict[str, Any]] = []
    for feature_index, feature_row in enumerate(freeze["features"]):
        feature = int(feature_row["feature"])
        vector_feature_index = feature_to_vector_row[feature]
        discovery = feature_row["discovery"]
        for di, context in enumerate(discovery):
            vector = residuals[int(context["context_index"])] if "context_index" in context else residuals[feature_index * N_CONTEXTS + di]
            context_text = str(context.get("context_text", ""))
            plans.append({
                "feature": feature, "doc_id": int(context["doc_id"]),
                "position": int(context["position"]), "role": "discovery",
                "arm": "NLA_RAW", "discovery_index": di, "control": False,
                "vector": np.asarray(vector, dtype=np.float32),
                "vector_sha256": array_sha256(vector), "context_text": context_text,
                "raw_text": docs[int(context["doc_id"])]["text"],
            })
            if include_contrastive_ablation:
                ablated_vector = ablated[vector_feature_index, di]
                plans.append({
                    "feature": feature, "doc_id": int(context["doc_id"]),
                    "position": int(context["position"]),
                    "role": "sae_feature_ablated", "arm": "NLA_CONTRASTIVE",
                    "discovery_index": di, "control": False,
                    "vector": np.asarray(ablated_vector, dtype=np.float32),
                    "vector_sha256": array_sha256(ablated_vector),
                    "context_text": context_text,
                    "raw_text": docs[int(context["doc_id"])]["text"],
                    "ablation_activation": float(vectors["contrastive_activation"][vector_feature_index, di]),
                    "ablation_norm_x": float(vectors["contrastive_norms"][vector_feature_index, di, 0]),
                    "ablation_norm_x_minus": float(vectors["contrastive_norms"][vector_feature_index, di, 1]),
                    "ablation_cosine": float(vectors["contrastive_cosine"][vector_feature_index, di]),
                })
        if include_wdec_control:
            for di, context in enumerate(discovery):
                vector = controls[vector_feature_index, di]
                plans.append({
                    "feature": feature, "doc_id": int(context["doc_id"]),
                    "position": int(context["position"]), "role": "wdec_control",
                    "arm": "WDEC_OFF_MANIFOLD_CONTROL", "discovery_index": di, "control": True,
                    "vector": np.asarray(vector, dtype=np.float32),
                    "vector_sha256": array_sha256(vector),
                    "context_text": str(context.get("context_text", "")),
                    "raw_text": docs[int(context["doc_id"])]["text"],
                })
    expected = N_FEATURES * N_DISCOVERY * ((2 if include_contrastive_ablation else 1) + (1 if include_wdec_control else 0))
    if len(plans) != expected:
        raise AssertionError(f"AV plan count {len(plans)} != {expected}")
    if not all(np.isfinite(row["vector"]).all() for row in plans):
        raise ValueError("AV plan vectors contain non-finite values")
    for idx, row in enumerate(plans):
        row["idx"] = idx
    return plans


def _result_payload(*, freeze_sha256: str, contract_sha256: str, plans: list[dict[str, Any]],
                    completed: dict[int, dict[str, Any]], mode: str,
                    model_hashes: dict[str, str], script_sha256: str,
                    raw_texts: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx in sorted(completed):
        source = completed[idx]
        plan = plans[idx]
        rows.append({
            "idx": idx, "feature": int(plan["feature"]), "role": plan["role"],
            "control": bool(plan["control"]), "doc_id": int(plan["doc_id"]),
            "position": int(plan["position"]), "discovery_index": int(plan["discovery_index"]),
            "vector_sha256": plan["vector_sha256"], "context_text": plan["context_text"],
        "raw_text_ref": str(plan["doc_id"]), "arm": plan["arm"],
        "explanation": str(source["explanation"]),
        **({key: plan[key] for key in ("ablation_activation", "ablation_norm_x",
                                       "ablation_norm_x_minus", "ablation_cosine")
            if key in plan}),
        })
    return {
        "schema_version": 1,
        "experiment": "J1 NLA-to-SAE exploratory discovery pilot",
        "status": mode,
        "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "itt": {"n_features": N_FEATURES, "recorded": 45},
        "freeze_sha256": freeze_sha256,
        "contract_sha256": contract_sha256,
        "models": model_hashes,
        "script_sha256": script_sha256,
        "generation": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 200,
                        "no_ar": True, "no_judge": True, "checkpoint_rows": len(rows)},
        "raw_texts": raw_texts,
        "rows": rows,
        "checks": {"n_rows": len(rows), "n_features": N_FEATURES,
                   "feature_count_in_rows": len({int(row["feature"]) for row in rows})},
    }


def _unload_cuda(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_RESULTS / "n3_candidate_cohort_v1.json")
    parser.add_argument("--stats", type=Path, default=DEFAULT_RESULTS / "n3_feature_stats_v1.npz")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_RESULTS / "n3_corpus_v1.jsonl")
    parser.add_argument("--protocol", type=Path,
                        default=DEFAULT_RESULTS / "J1_DISCOVERY_PILOT_PROTOCOL_2026-08-06.md")
    parser.add_argument("--base-model", default="/root/autodl-tmp/models/gemma-3-12b-it")
    parser.add_argument("--sae", "--sae-dir", dest="sae", default="/root/autodl-tmp/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small")
    parser.add_argument("--av", default="/root/autodl-tmp/models/nla-gemma3-12b-L32-av")
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--activation-rtol", type=float, default=2.5e-2)
    parser.add_argument("--activation-atol", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-wdec-control", action="store_true", default=False)
    parser.add_argument("--contrastive-ablation", "--include-contrastive-ablation",
                        dest="include_contrastive_ablation", action="store_true",
                        default=True,
                        help="generate paired x/x_minus SAE-ablation AV rows (default: on)")
    parser.add_argument("--no-contrastive-ablation", dest="include_contrastive_ablation",
                        action="store_false", help="disable the paired NLA_CONTRASTIVE arm")
    parser.add_argument("--stop-after-av-jobs", type=int, default=0,
                        help="positive value writes an isolated smoke-only result/checkpoint")
    parser.add_argument("--out-freeze", type=Path, default=DEFAULT_RESULTS / "j1_discovery_freeze_v1.json")
    parser.add_argument("--out-freeze-sha256", type=Path, default=None)
    parser.add_argument("--out-checkpoint", type=Path, default=DEFAULT_RESULTS / "j1_discovery_av_checkpoint_v1.jsonl")
    parser.add_argument("--out-result", type=Path, default=DEFAULT_RESULTS / "j1_discovery_result_v1.json")
    parser.add_argument("--out-vectors", type=Path, default=DEFAULT_RESULTS / "j1_discovery_vectors_v1.npz")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed != SEED:
        raise ValueError(f"seed is fixed at {SEED}")
    if args.layer_index != LAYER_INDEX:
        raise ValueError(f"layer-index is fixed at {LAYER_INDEX}")
    if args.temperature != 0.0:
        raise ValueError("J1 AV generation is fixed greedy temperature=0")
    if args.max_new_tokens != 200:
        raise ValueError("J1 max-new-tokens is fixed at 200")
    if args.activation_rtol != 2.5e-2 or args.activation_atol != 1.0:
        raise ValueError("J1 engineering amendment A1 fixes activation tolerance at rtol=0.025, atol=1.0")
    if args.stop_after_av_jobs < 0:
        raise ValueError("stop-after-av-jobs cannot be negative")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for path in (args.cohort, args.stats, args.corpus, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.out_freeze_sha256 is None:
        args.out_freeze_sha256 = Path(str(args.out_freeze) + ".sha256")

    # Input hashes are calculated before model work and are embedded in the
    # immutable freeze.  Inputs are never opened for writing.
    input_hashes = {"cohort_sha256": sha256_file(args.cohort),
                    "stats_sha256": sha256_file(args.stats),
                    "corpus_sha256": sha256_file(args.corpus),
                    "protocol_sha256": sha256_file(args.protocol)}
    script_sha = sha256_file(Path(__file__))
    pilot_common_path = Path(__file__).with_name("pilot_common.py")
    pilot_common_sha = sha256_file(pilot_common_path)
    cohort = load_cohort(args.cohort)
    stats = load_stats(args.stats)
    docs = load_corpus(args.corpus)
    selected = select_features(cohort, stats, args.seed)
    contexts = build_contexts(selected, stats, docs)

    # A freeze is immutable.  If already present, verify its sidecar and
    # vector hash then reuse it without loading the base model again.
    vectors_sha: str
    vectors: dict[str, np.ndarray]
    model_hashes: dict[str, str]
    if args.out_freeze.exists():
        freeze = json.loads(args.out_freeze.read_text(encoding="utf-8"))
        freeze_sha = sha256_file(args.out_freeze)
        sidecar = args.out_freeze_sha256.read_text(encoding="ascii").strip()
        if sidecar != freeze_sha:
            raise ValueError("freeze JSON/sha256 sidecar mismatch")
        if freeze.get("status") != "EXPLORATORY_DISCOVERY_FROZEN_BEFORE_AV" or freeze.get("confirmatory"):
            raise ValueError("existing freeze is not an exploratory pre-AV freeze")
        if freeze.get("script_sha256") != script_sha:
            raise ValueError("existing freeze was created by a different script revision")
        if freeze.get("inputs") != input_hashes:
            raise ValueError("existing freeze input hashes differ")
        if int(freeze.get("selection", {}).get("seed", -1)) != SEED:
            raise ValueError("existing freeze seed differs")
        if bool(freeze.get("include_wdec_control")) != bool(args.include_wdec_control):
            raise ValueError("existing freeze control setting differs")
        if bool(freeze.get("include_contrastive_ablation", True)) != bool(args.include_contrastive_ablation):
            raise ValueError("existing freeze contrastive-ablation setting differs")
        if not args.out_vectors.is_file():
            raise FileNotFoundError(freeze.get("vectors", {}).get("path_role", "vectors"))
        vectors_sha = sha256_file(args.out_vectors)
        if vectors_sha != freeze.get("vectors", {}).get("sha256"):
            raise ValueError("existing freeze/vector hash mismatch")
        with np.load(args.out_vectors, allow_pickle=False) as loaded:
            vectors = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
        model_hashes = dict(freeze.get("models", {}))
        background_meta = dict(freeze.get("background_pool", {}))
    else:
        background_rows, background_meta = build_background_candidates(
            base_model=str(args.base_model), contexts=contexts, docs=docs, seed=args.seed)
        contexts.extend(background_rows)
        try:
            contexts, extracted, extraction_meta = extract_candidate_vectors(
                contexts=contexts, docs=docs, base_model=str(args.base_model), sae_dir=str(args.sae),
                layer_index=args.layer_index, batch_size=args.batch_size,
                rtol=args.activation_rtol, atol=args.activation_atol,
                include_contrastive_ablation=args.include_contrastive_ablation)
        except ExtractionVerificationError as exc:
            error_path = args.out_freeze.with_name(args.out_freeze.stem + ".extraction_errors.json")
            error_payload = {
                "schema_version": 1,
                "experiment": "J1 exploratory discovery extraction verification",
                "status": "FAILED_CLOSED_BEFORE_AV",
                "inputs": input_hashes, "script_sha256": script_sha,
                "errors": exc.errors,
            }
            write_or_verify_json(error_path, error_payload)
            raise
        negatives = assign_hard_negatives(contexts=contexts,
                                          residuals=extracted["residuals"], acts=extracted["sae_acts"])
        # Add a stable explicit context index to the freeze records.  It is
        # needed to bind vectors without relying on JSON list ordering.
        for row in contexts:
            row["context_index"] = int(row["context_index"])
        vectors = extracted
        vectors_sha = write_or_verify_npz(args.out_vectors, vectors)
        model_hashes = {
            "base_model_name": "Gemma-3-12B-IT",
            "base_model_path": str(args.base_model),
            "base_model_manifest_sha256": extraction_meta["base_model_manifest_sha256"],
            "sae_name": "GemmaScope small SAE (layer 32 width 16k l0_small)",
            "sae_path": str(args.sae),
            "sae_manifest_sha256": extraction_meta["sae_manifest_sha256"],
            "av_name": "NLA Gemma-3-12B L32 AV",
            "av_path": str(args.av),
        }
        vector_shapes = {key: list(np.asarray(value).shape) for key, value in vectors.items()}
        freeze_payload = _freeze_payload(selected=selected, contexts=contexts,
                                         negatives=negatives, input_hashes=input_hashes,
                                         model_hashes=model_hashes, vectors_sha256=vectors_sha,
                                         vector_shapes=vector_shapes, script_sha256=script_sha,
                                         activation_verification=extraction_meta["activation_verification"],
                                         background_meta=background_meta,
                                         include_wdec_control=args.include_wdec_control,
                                         include_contrastive_ablation=args.include_contrastive_ablation)
        freeze_sha = write_or_verify_json(args.out_freeze, freeze_payload)
        write_or_verify_sidecar(args.out_freeze_sha256, freeze_sha)
        # The freeze is complete before any AV construction is even attempted.
        freeze = freeze_payload
        _unload_cuda()

    validate_freeze_semantics(freeze)
    # On a resumed freeze, load the vectors only after validating all arrays.
    required_vectors = {"residuals", "sae_acts", "wdec_controls", "feature_ids",
                        "sae_ablated_residuals", "contrastive_activation",
                        "contrastive_norms", "contrastive_cosine"}
    if not required_vectors.issubset(vectors):
        raise ValueError(f"vectors missing {sorted(required_vectors - set(vectors))}")
    if not all(np.isfinite(np.asarray(v)).all() for v in vectors.values()):
        raise ValueError("frozen vectors contain non-finite values")
    if len(freeze.get("features", [])) != N_FEATURES:
        raise ValueError("freeze does not contain exactly 45 features")

    # AV model hashes are resolved only after base+SAE have been unloaded.  A
    # resumed freeze has no live base model at this point either.
    av_manifest_sha = path_manifest_sha256(Path(args.av))
    model_hashes = dict(model_hashes)
    model_hashes.setdefault("base_model_name", "Gemma-3-12B-IT")
    model_hashes.setdefault("sae_name", "GemmaScope small SAE (layer 32 width 16k l0_small)")
    model_hashes["av_name"] = "NLA Gemma-3-12B L32 AV"
    model_hashes["av_path"] = str(args.av)
    model_hashes["av_manifest_sha256"] = av_manifest_sha
    model_hashes["pilot_common_sha256"] = pilot_common_sha
    plans = build_av_plans(freeze=freeze, vectors=vectors, docs=docs,
                           include_wdec_control=args.include_wdec_control,
                           include_contrastive_ablation=args.include_contrastive_ablation)
    limit = len(plans) if args.stop_after_av_jobs == 0 else min(args.stop_after_av_jobs, len(plans))
    smoke = args.stop_after_av_jobs > 0
    checkpoint_path = _smoke_path(args.out_checkpoint, limit) if smoke else args.out_checkpoint
    result_path = _smoke_path(args.out_result, limit) if smoke else args.out_result
    mode = "SMOKE_ONLY_STOP_AFTER_AV_JOBS" if smoke else "EXPLORATORY_DISCOVERY_AV_COMPLETE"
    contract = {
        "experiment": "J1 exploratory discovery AV generation",
        "freeze_sha256": freeze_sha,
        "vectors_sha256": vectors_sha,
        "av_manifest_sha256": av_manifest_sha,
        "script_sha256": script_sha,
        "pilot_common_sha256": pilot_common_sha,
        "temperature": 0.0, "max_new_tokens": 200,
        "include_wdec_control": bool(args.include_wdec_control),
        "include_contrastive_ablation": bool(args.include_contrastive_ablation),
        "run_mode": "smoke_only" if smoke else "full_discovery",
        "planned_jobs": limit,
        "vector_sha_sequence": canonical_sha256([row["vector_sha256"] for row in plans[:limit]]),
    }
    contract_sha = canonical_sha256(contract)
    completed = load_checkpoint(checkpoint_path, contract_sha256=contract_sha,
                                plans=plans, limit=limit)
    missing = [idx for idx in range(limit) if idx not in completed]
    print(f"[J1 {'SMOKE' if smoke else 'DISCOVERY'}] features={N_FEATURES} "
          f"jobs={limit} checkpoint={len(completed)} missing={len(missing)}", flush=True)
    if missing:
        if not Path(args.av).exists():
            raise FileNotFoundError(args.av)
        av = AVLocal(str(args.av), device="cuda", dtype=torch.bfloat16)
        try:
            for idx in missing:
                plan = plans[idx]
                explanation = av.generate(plan["vector"], temperature=0.0, max_new_tokens=200)
                if not isinstance(explanation, str) or not explanation.strip():
                    raise RuntimeError(f"empty AV explanation for job {idx}")
                checkpoint_row = {
                    "contract_sha256": contract_sha, "idx": idx,
                    "feature": int(plan["feature"]), "role": plan["role"],
                    "arm": plan["arm"], "control": bool(plan["control"]),
                    "doc_id": int(plan["doc_id"]),
                    "position": int(plan["position"]),
                    "discovery_index": int(plan["discovery_index"]),
                    "vector_sha256": plan["vector_sha256"],
                    "context_text": plan["context_text"],
                    "raw_text_sha256": sha256_bytes(plan["raw_text"].encode("utf-8")),
                    "explanation": explanation,
                    "explanation_utf8_sha256": sha256_bytes(explanation.encode("utf-8")),
                }
                append_checkpoint(checkpoint_path, checkpoint_row)
                completed[idx] = checkpoint_row
                print(f"[J1 AV {len(completed)}/{limit}] feature={plan['feature']} "
                      f"role={plan['role']} doc={plan['doc_id']} pos={plan['position']}", flush=True)
        finally:
            _unload_cuda(av)
    if len(completed) != limit:
        raise RuntimeError("AV checkpoint is incomplete; result is not written")

    raw_texts: dict[str, Any] = {}
    for plan in plans[:limit]:
        doc_key = str(int(plan["doc_id"]))
        entry = raw_texts.setdefault(doc_key, {
            "sha256": sha256_bytes(plan["raw_text"].encode("utf-8")),
            "context_windows": [],
        })
        if plan["context_text"] not in entry["context_windows"]:
            entry["context_windows"].append(plan["context_text"])
    result = _result_payload(freeze_sha256=freeze_sha, contract_sha256=contract_sha,
                             plans=plans, completed=completed, mode=mode,
                             model_hashes=model_hashes, script_sha256=script_sha,
                             raw_texts=raw_texts)
    result_sha = write_or_verify_json(result_path, result)
    if not smoke:
        write_or_verify_sidecar(Path(str(result_path) + ".sha256"), result_sha)
    print(f"J1 {'SMOKE' if smoke else 'DISCOVERY'} RESULT status={mode} "
          f"rows={len(completed)} freeze_sha256={freeze_sha} result_sha256={result_sha}", flush=True)


if __name__ == "__main__":
    main()
