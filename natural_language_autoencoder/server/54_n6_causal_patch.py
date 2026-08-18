#!/usr/bin/env python3
"""Run the frozen N6 causal patches and clean candidate-mass diagnostics.

All ten substitutes are read from the frozen reconstruction archive.  Every
nonzero substitute, including identity, receives the exact N5 row-wise norm
match; the zero control remains exactly zero.  The clean forward is also used
to verify the frozen layer-32 activation and to score the preregistered
candidate first-token sets before any causal result exists.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from n6_common import (
    IDENTITY_KL_TOL,
    canonical_sha256,
    model_file_manifest,
    norm_match,
    parse_weight_manifest,
    require_binding_preregistration,
    scalar_npz_string,
    sha256_file,
    validate_model_subset,
    verify_code_manifest,
    verify_sha256_sidecar,
    write_new_json,
)


LAYER_INDEX = 32
HORIZON = 16
DTYPE = "bfloat16"
EPSILON = 1e-12
TOP_K = (1, 5, 10, 50)
ACTIVATION_METADATA_PREFIX = "nla.n6_activation_extraction."
TEXT_CONDITIONS = (
    "orig",
    "p3_true",
    "p3_cross_matched",
    "p3_candidate_strip",
    "p3_anchor_strip",
    "p3_all_quote_strip",
    "p12",
)
CONDITIONS = (
    "identity",
    *TEXT_CONDITIONS,
    "sae_big",
    "zero",
)
NPZ_KEYS = {
    "orig": "pred_orig",
    "p3_true": "pred_p3_true",
    "p3_cross_matched": "pred_p3_cross_matched",
    "p3_candidate_strip": "pred_p3_candidate_strip",
    "p3_anchor_strip": "pred_p3_anchor_strip",
    "p3_all_quote_strip": "pred_p3_all_quote_strip",
    "p12": "pred_p12",
    "sae_big": "recon_sae_big",
}


def resolve_layers(model: torch.nn.Module):
    for attributes in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
    ):
        current = model
        for attribute in attributes:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None:
            return current
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def table_metadata(table) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="strict"): value.decode(
            "utf-8", errors="strict"
        )
        for key, value in (table.schema.metadata or {}).items()
    }


def load_activation_rows(
    path: Path,
    *,
    plan: dict[str, Any],
    plan_sha: str,
    prereg_sha: str,
    code_manifest_sha: str,
    model_manifest_sha: str,
) -> dict[str, Any]:
    table = pq.read_table(path)
    required = {
        "activation_vector",
        "row_uid",
        "content_group_id",
        "doc_id",
        "position",
        "input_ids",
        "token",
        "token_id",
        "source",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"activation parquet lacks {missing}")
    plan_rows = plan.get("rows")
    if not isinstance(plan_rows, list) or table.num_rows != len(plan_rows):
        raise ValueError("activation/provisional-plan row count mismatch")
    values = {
        name: table[name].to_pylist()
        for name in required
        if name != "activation_vector"
    }
    values["x"] = np.asarray(
        table["activation_vector"].combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    if values["x"].ndim != 2 or not np.isfinite(values["x"]).all():
        raise ValueError(f"invalid activation matrix {values['x'].shape}")
    expected_uids = [str(row["row_uid"]) for row in plan_rows]
    if [str(value) for value in values["row_uid"]] != expected_uids:
        raise ValueError("activation order differs from provisional plan")
    metadata = table_metadata(table)
    expected_metadata = {
        "plan_sha256": plan_sha,
        "preregistration_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": model_manifest_sha,
        "layer_index": str(LAYER_INDEX),
        "dtype": DTYPE,
        "batch_size": "1",
        "full_frozen_sequence": "true",
    }
    for key, expected in expected_metadata.items():
        full_key = f"{ACTIVATION_METADATA_PREFIX}{key}"
        if metadata.get(full_key) != expected:
            raise ValueError(
                f"activation metadata {key}={metadata.get(full_key)!r}, "
                f"expected {expected!r}"
            )
    values["metadata"] = metadata
    return values


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    output: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text")
            if text is None:
                text = item.get("content")
        else:
            text = None
        if not isinstance(text, str) or not text:
            raise ValueError(f"{label}[{index}] lacks nonempty text")
        output.append(text)
    return output


def candidate_sets(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Read the two exact candidate-set fields frozen by stage 52."""
    true = _string_list(
        row.get("true_candidates"),
        f"{row.get('row_uid')}.true_candidates",
    )
    cross = _string_list(
        row.get("cross_candidates"),
        f"{row.get('row_uid')}.cross_candidates",
    )
    expected = int(row.get("candidate_count", -1))
    if len(true) != expected or len(cross) != expected:
        raise ValueError(
            f"{row.get('row_uid')} candidate counts "
            f"{len(true)}/{len(cross)} != {expected}"
        )
    return true, cross


def begins_whitespace_or_punctuation(text: str) -> bool:
    if not text:
        return False
    return text[0].isspace() or unicodedata.category(text[0]).startswith("P")


def first_token_ids(tokenizer, candidates: list[str]) -> dict[str, Any]:
    canonical_strings = [
        value
        if begins_whitespace_or_punctuation(value)
        else " " + value
        for value in candidates
    ]
    canonical_sequences = [
        [int(token) for token in tokenizer.encode(value, add_special_tokens=False)]
        for value in canonical_strings
    ]
    raw_sequences = [
        [int(token) for token in tokenizer.encode(value, add_special_tokens=False)]
        for value in candidates
    ]
    if any(not sequence for sequence in canonical_sequences + raw_sequences):
        raise ValueError("a candidate tokenized to an empty sequence")
    canonical_first = [sequence[0] for sequence in canonical_sequences]
    raw_first = [sequence[0] for sequence in raw_sequences]
    return {
        "candidate_texts": candidates,
        "canonical_strings": canonical_strings,
        "canonical_token_ids": canonical_sequences,
        "raw_token_ids": raw_sequences,
        "canonical_first_token_ids": canonical_first,
        "raw_first_token_ids": raw_first,
        "canonical_unique_first_token_ids": sorted(set(canonical_first)),
        "raw_unique_first_token_ids": sorted(set(raw_first)),
    }


@torch.inference_mode()
def clean_forward(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    ids: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, np.ndarray]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["x"] = hidden[0, position].detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    try:
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
    finally:
        handle.remove()
    logits = output.logits[0, position : position + HORIZON].float().clone()
    observed = captured["x"].numpy().astype(np.float32, copy=False)
    del output
    return logits, observed


@torch.inference_mode()
def patched_forward(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    ids: torch.Tensor,
    position: int,
    vector: np.ndarray,
) -> torch.Tensor:
    patch = torch.as_tensor(vector, dtype=torch.float32, device=ids.device)

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        hidden[0, position] = patch.to(hidden.dtype)
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
    logits = output.logits[0, position : position + HORIZON].float().clone()
    del output, patch
    return logits


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


def candidate_alignment(
    clean_probs: torch.Tensor,
    tokenizer,
    true_candidates: list[str],
    cross_candidates: list[str],
    observed_next_token_id: int,
) -> dict[str, Any]:
    true = first_token_ids(tokenizer, true_candidates)
    cross = first_token_ids(tokenizer, cross_candidates)
    probabilities = clean_probs[0].detach().float().cpu().numpy()
    true_unique = true["canonical_unique_first_token_ids"]
    cross_unique = cross["canonical_unique_first_token_ids"]
    true_mass = float(probabilities[true_unique].sum(dtype=np.float64))
    cross_mass = float(probabilities[cross_unique].sum(dtype=np.float64))
    n_true = len(true_unique)
    n_cross = len(cross_unique)
    top_values, top_ids = torch.topk(clean_probs[0], k=max(TOP_K))
    ranked_ids = [int(value) for value in top_ids.detach().cpu().tolist()]
    ranked_probs = [
        float(value) for value in top_values.detach().float().cpu().tolist()
    ]
    true_set = set(true_unique)
    cross_set = set(cross_unique)
    return {
        "canonical_rule": (
            "prepend one ASCII space unless candidate begins with Unicode "
            "whitespace or Unicode punctuation"
        ),
        "epsilon": EPSILON,
        "true": true,
        "cross": cross,
        "p_true_setmass": true_mass,
        "p_cross_setmass": cross_mass,
        "n_unique_true": n_true,
        "n_unique_cross": n_cross,
        "observed_next_token_id": int(observed_next_token_id),
        "observed_next_token_in_true_canonical_set": bool(
            observed_next_token_id in true_set
        ),
        "observed_next_token_in_cross_canonical_set": bool(
            observed_next_token_id in cross_set
        ),
        "p_true_meanmass": true_mass / n_true,
        "p_cross_meanmass": cross_mass / n_cross,
        "a_setmass_row": float(
            np.log(true_mass + EPSILON) - np.log(cross_mass + EPSILON)
        ),
        "a_meanmass_row": float(
            np.log(true_mass / n_true + EPSILON)
            - np.log(cross_mass / n_cross + EPSILON)
        ),
        "top_token_ids": ranked_ids,
        "top_token_probabilities": ranked_probs,
        "hit_at_k": {
            str(k): {
                "true": bool(true_set.intersection(ranked_ids[:k])),
                "cross": bool(cross_set.intersection(ranked_ids[:k])),
            }
            for k in TOP_K
        },
    }


def load_checkpoint(
    path: Path,
    contract_sha: str,
    expected_uids: set[str],
) -> tuple[dict[str, dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    forwards = 0
    if not path.exists():
        return rows, forwards
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("contract_sha256") != contract_sha:
            raise ValueError(f"checkpoint contract mismatch at line {line_number}")
        row = record.get("row")
        uid = str(row.get("row_uid")) if isinstance(row, dict) else ""
        if uid not in expected_uids or uid in rows:
            raise ValueError(f"invalid checkpoint row {uid!r}")
        rows[uid] = row
        forwards += int(record.get("n_forwards", 0))
    return rows, forwards


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    zero = np.asarray(
        [row["results"]["zero"]["kl_at_pos"] for row in rows], dtype=np.float64
    )
    denominator = float(zero.sum())
    if denominator <= 1e-12:
        raise ValueError(f"sum(KL_zero)={denominator} is invalid")
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        values = np.asarray(
            [row["results"][condition]["kl_at_pos"] for row in rows],
            dtype=np.float64,
        )
        values16 = np.asarray(
            [row["results"][condition]["kl_mean_first16"] for row in rows],
            dtype=np.float64,
        )
        conditions[condition] = {
            "kl_at_pos_mean": float(values.mean()),
            "kl_at_pos_median": float(np.median(values)),
            "kl_at_pos_max": float(values.max()),
            "kl16_mean": float(values16.mean()),
            "ratio_of_sums_recovery": float(1.0 - values.sum() / denominator),
        }
    align = [row["candidate_alignment"] for row in rows]
    return {
        "sum_kl_zero": denominator,
        "conditions": conditions,
        "candidate_alignment": {
            "a_meanmass_mean": float(
                np.mean([value["a_meanmass_row"] for value in align])
            ),
            "a_setmass_mean": float(
                np.mean([value["a_setmass_row"] for value in align])
            ),
            "mean_p_true_setmass": float(
                np.mean([value["p_true_setmass"] for value in align])
            ),
            "mean_p_cross_setmass": float(
                np.mean([value["p_cross_setmass"] for value in align])
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--recon-json", required=True, type=Path)
    parser.add_argument("--recon-npz", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite frozen output {args.out}")
    started = time.time()
    prereg_sha = require_binding_preregistration(args.prereg)
    code_manifest_sha = verify_code_manifest(args.code_manifest, __file__)
    model_manifest_sha = verify_sha256_sidecar(args.model_manifest)
    plan_sha = verify_sha256_sidecar(args.plan)
    activation_sha = verify_sha256_sidecar(args.activations)
    variants_sha = verify_sha256_sidecar(args.variants)
    recon_json_sha = verify_sha256_sidecar(args.recon_json)
    recon_npz_sha = verify_sha256_sidecar(args.recon_npz)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    variants = json.loads(args.variants.read_text(encoding="utf-8"))
    recon_json = json.loads(args.recon_json.read_text(encoding="utf-8"))
    if (
        variants.get("status")
        != "COMPLETE_FROZEN_BEFORE_AR_CANDIDATE_MASS_OR_CAUSAL_OUTCOME"
    ):
        raise ValueError("variant artifact is not frozen before outcomes")
    if plan.get("preregistration_sha256") != prereg_sha:
        raise ValueError("plan/preregistration hash mismatch")
    plan_inputs = plan.get("inputs", {})
    if plan_inputs.get("code_manifest_sha256") != code_manifest_sha:
        raise ValueError("plan/code-manifest hash mismatch")
    if plan_inputs.get("model_manifest_sha256") != model_manifest_sha:
        raise ValueError("plan/model-manifest hash mismatch")
    variant_inputs = variants.get("inputs", {})
    for key, expected in {
        "plan_sha256": plan_sha,
        "prereg_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": model_manifest_sha,
    }.items():
        if variant_inputs.get(key) != expected:
            raise ValueError(f"variants {key} mismatch")
    if recon_json.get("outputs", {}).get("vecs_sha256") != recon_npz_sha:
        raise ValueError("reconstruction JSON/NPZ hash mismatch")
    recon_inputs = recon_json.get("inputs", {})
    for key, expected in {
        "activations_sha256": activation_sha,
        "plan_sha256": plan_sha,
        "variants_sha256": variants_sha,
        "prereg_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": model_manifest_sha,
    }.items():
        if recon_inputs.get(key) != expected:
            raise ValueError(f"reconstruction {key} mismatch")

    activation = load_activation_rows(
        args.activations,
        plan=plan,
        plan_sha=plan_sha,
        prereg_sha=prereg_sha,
        code_manifest_sha=code_manifest_sha,
        model_manifest_sha=model_manifest_sha,
    )
    variant_rows = variants.get("rows")
    if not isinstance(variant_rows, list) or not variant_rows:
        raise ValueError("variant artifact has no analysis rows")
    uids = [str(row["row_uid"]) for row in variant_rows]
    if len(set(uids)) != len(uids):
        raise ValueError("variant row_uids are not unique")
    provisional_by_uid = {
        str(uid): index for index, uid in enumerate(activation["row_uid"])
    }
    if any(uid not in provisional_by_uid for uid in uids):
        raise ValueError("variant row is absent from activations")
    selected = np.asarray(
        [provisional_by_uid[uid] for uid in uids], dtype=np.int64
    )
    selected_x = activation["x"][selected]

    with np.load(args.recon_npz, allow_pickle=False) as archive:
        for key, expected in {
            "activations_sha256": activation_sha,
            "plan_sha256": plan_sha,
            "variants_sha256": variants_sha,
            "prereg_sha256": prereg_sha,
            "code_manifest_sha256": code_manifest_sha,
            "model_manifest_sha256": model_manifest_sha,
            "center_reference": "n5_gate_v2",
        }.items():
            scalar_npz_string(archive, key, expected)
        npz_uids = [str(value) for value in np.asarray(archive["row_uids"]).tolist()]
        if npz_uids != uids:
            raise ValueError("reconstruction row order differs from variants")
        archived_x = np.asarray(archive["x"], dtype=np.float32)
        if not np.array_equal(archived_x, selected_x):
            raise ValueError("reconstruction x differs from selected activations")
        source_vectors = {
            condition: np.asarray(archive[key], dtype=np.float32)
            for condition, key in NPZ_KEYS.items()
        }
    for condition, vectors in source_vectors.items():
        if vectors.shape != selected_x.shape or not np.isfinite(vectors).all():
            raise ValueError(f"invalid reconstruction array for {condition}")
    substitutes = {
        "identity": norm_match(selected_x, selected_x, "identity"),
        **{
            condition: norm_match(vectors, selected_x, condition)
            for condition, vectors in source_vectors.items()
        },
        "zero": np.zeros_like(selected_x),
    }
    if tuple(substitutes) != CONDITIONS:
        raise AssertionError(f"condition order drift: {tuple(substitutes)}")

    actual_base_manifest = model_file_manifest(args.base_model)
    frozen_model_manifest = parse_weight_manifest(args.model_manifest)
    validate_model_subset({"base": actual_base_manifest}, frozen_model_manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True
    )
    config_path = args.base_model / "config.json"
    contract = {
        "activations_sha256": activation_sha,
        "plan_sha256": plan_sha,
        "variants_sha256": variants_sha,
        "recon_json_sha256": recon_json_sha,
        "recon_npz_sha256": recon_npz_sha,
        "prereg_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": model_manifest_sha,
        "script_sha256": sha256_file(__file__),
        "base_config_sha256": sha256_file(config_path),
        "conditions": list(CONDITIONS),
        "row_uids_sha256": canonical_sha256(uids),
        "layer_index": LAYER_INDEX,
        "horizon": HORIZON,
        "dtype": DTYPE,
        "epsilon": EPSILON,
        "top_k": list(TOP_K),
    }
    contract_sha = canonical_sha256(contract)
    checkpoint_rows, forward_count = load_checkpoint(
        args.checkpoint, contract_sha, set(uids)
    )

    model_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    device = next(model.parameters()).device
    layers = resolve_layers(model)
    if not 0 <= LAYER_INDEX < len(layers):
        raise ValueError("layer 32 is outside the base model")
    layer = layers[LAYER_INDEX]
    model_load_seconds = time.time() - model_started

    output_rows: list[dict[str, Any] | None] = [
        checkpoint_rows.get(uid) for uid in uids
    ]
    provenance_max = max(
        (
            float(row["provenance"]["max_abs"])
            for row in checkpoint_rows.values()
        ),
        default=0.0,
    )
    forward_started = time.time()
    for index, (uid, variant_row) in enumerate(zip(uids, variant_rows)):
        if output_rows[index] is not None:
            print(f"[N6 causal {index + 1}/{len(uids)}] checkpoint {uid}", flush=True)
            continue
        provisional_index = int(selected[index])
        ids_np = np.asarray(
            activation["input_ids"][provisional_index], dtype=np.int64
        )
        position = int(activation["position"][provisional_index])
        if position < 0 or position + HORIZON >= len(ids_np):
            raise ValueError(f"{uid} lacks the frozen 16-token causal horizon")
        ids = torch.as_tensor(ids_np[None], dtype=torch.long, device=device)
        clean_logits, observed = clean_forward(model, layer, ids, position)
        forward_count += 1
        difference = np.abs(observed - selected_x[index])
        max_abs = float(difference.max())
        exact = bool(np.array_equal(observed, selected_x[index]))
        provenance_max = max(provenance_max, max_abs)
        if not exact:
            raise RuntimeError(
                f"{uid} clean L32 activation differs from frozen x: {max_abs}"
            )
        targets = ids[0, position + 1 : position + 1 + HORIZON]
        clean = clean_metrics(clean_logits, targets)
        true_candidates, cross_candidates = candidate_sets(variant_row)
        alignment = candidate_alignment(
            clean["probs"],
            tokenizer,
            true_candidates,
            cross_candidates,
            int(targets[0].item()),
        )
        if alignment["true"]["canonical_first_token_ids"] != [
            int(value)
            for value in variant_row["true_canonical_first_token_ids"]
        ]:
            raise ValueError(f"{uid} true canonical token IDs drifted after freeze")
        if alignment["cross"]["canonical_first_token_ids"] != [
            int(value)
            for value in variant_row["cross_canonical_first_token_ids"]
        ]:
            raise ValueError(f"{uid} cross canonical token IDs drifted after freeze")
        results: dict[str, dict[str, float | int]] = {}
        for condition in CONDITIONS:
            patched_logits = patched_forward(
                model,
                layer,
                ids,
                position,
                substitutes[condition][index],
            )
            forward_count += 1
            results[condition] = score(clean, patched_logits, targets)
            del patched_logits
        row = {
            "idx_analysis": index,
            "row_uid": uid,
            "content_group_id": str(variant_row["content_group_id"]),
            "doc_id": int(variant_row["doc_id"]),
            "source": str(variant_row["source"]),
            "candidate_count": int(variant_row["candidate_count"]),
            "position": position,
            "seq_len": int(len(ids_np)),
            "token": str(activation["token"][provisional_index]),
            "token_id": int(activation["token_id"][provisional_index]),
            "provenance": {"exact": exact, "max_abs": max_abs},
            "ce_clean_first16": float(clean["ce"]),
            "candidate_alignment": alignment,
            "results": results,
        }
        output_rows[index] = row
        append_checkpoint(
            args.checkpoint,
            {
                "contract_sha256": contract_sha,
                "n_forwards": 1 + len(CONDITIONS),
                "row": row,
            },
        )
        completed = sum(value is not None for value in output_rows)
        elapsed = time.time() - forward_started
        eta = elapsed / max(completed - len(checkpoint_rows), 1) * (
            len(uids) - completed
        )
        print(
            f"[N6 causal {completed}/{len(uids)}] {uid} "
            f"true={results['p3_true']['kl_at_pos']:.4g} "
            f"cross={results['p3_cross_matched']['kl_at_pos']:.4g} "
            f"Amean={alignment['a_meanmass_row']:.4g} eta={eta / 60:.1f}m",
            flush=True,
        )
        del ids, clean_logits, clean, results

    rows = [row for row in output_rows if row is not None]
    if len(rows) != len(uids):
        raise RuntimeError(f"produced {len(rows)}/{len(uids)} rows")
    identity_max = max(
        abs(float(row["results"]["identity"]["kl_at_pos"])) for row in rows
    )
    identity16_max = max(
        abs(float(row["results"]["identity"]["kl_mean_first16"])) for row in rows
    )
    if max(identity_max, identity16_max) > IDENTITY_KL_TOL:
        raise RuntimeError(
            "identity KL QA failed: "
            f"position={identity_max}, first16={identity16_max}"
        )
    payload = {
        "schema_version": 1,
        "experiment": "N6 frozen candidate-channel causal patch",
        "status": "complete",
        "protocol": {
            "layer_index": LAYER_INDEX,
            "horizon": HORIZON,
            "batch_size": 1,
            "full_sequence": True,
            "dtype": DTYPE,
            "conditions": list(CONDITIONS),
            "norm_matching": (
                "exact N5 row-wise norm matching for every nonzero substitute; "
                "zero remains zero"
            ),
            "kl_direction": "KL(clean || patched)",
            "candidate_mass_epsilon": EPSILON,
            "candidate_top_k": list(TOP_K),
            "confirmatory_alignment": "A_meanmass",
            "secondary_alignment": "A_setmass",
        },
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": activation_sha,
            "plan": str(args.plan),
            "plan_sha256": plan_sha,
            "variants": str(args.variants),
            "variants_sha256": variants_sha,
            "recon_json": str(args.recon_json),
            "recon_json_sha256": recon_json_sha,
            "recon_npz": str(args.recon_npz),
            "recon_npz_sha256": recon_npz_sha,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha,
            "code_manifest": str(args.code_manifest),
            "code_manifest_sha256": code_manifest_sha,
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha,
            "actual_base_model_manifest": actual_base_manifest,
            "activation_metadata": activation["metadata"],
            "contract_sha256": contract_sha,
            "script_sha256": sha256_file(__file__),
        },
        "cohort": {
            "n_rows": len(rows),
            "n_content_groups": len(
                {str(row["content_group_id"]) for row in rows}
            ),
            "by_source": {
                source: sum(row["source"] == source for row in rows)
                for source in sorted({row["source"] for row in rows})
            },
        },
        "qa": {
            "provenance_all_bit_exact": True,
            "provenance_max_abs": provenance_max,
            "identity_kl_at_pos_abs_max": identity_max,
            "identity_kl16_abs_max": identity16_max,
            "identity_kl_tolerance": IDENTITY_KL_TOL,
        },
        "summary": summarize(rows),
        "rows": rows,
        "n_forwards": forward_count,
        "model_load_seconds": model_load_seconds,
        "forward_seconds": time.time() - forward_started,
        "elapsed_seconds": time.time() - started,
    }
    output_sha = write_new_json(args.out, payload)
    print(
        f"N6_CAUSAL_COMPLETE rows={len(rows)} forwards={forward_count} "
        f"sha256={output_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
