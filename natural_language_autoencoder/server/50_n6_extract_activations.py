#!/usr/bin/env python3
"""Extract L32 ``resid_post`` for every frozen N6 provisional row.

No tokenization, filtering, resampling, or analysis-set selection occurs here.
The full input IDs and row order come from the immutable provisional plan.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from n6_common import (
    canonical_sha256,
    model_file_manifest,
    parse_weight_manifest,
    require_binding_preregistration,
    require_unique,
    sha256_file,
    shahex,
    validate_model_subset,
    verify_code_manifest,
    verify_sha256_sidecar,
)


METADATA_PREFIX = "nla.n6_activation_extraction."


class _StopForward(Exception):
    """Terminate immediately after the frozen residual layer fires."""


def tokenizer_file_hashes(model_dir: Path) -> dict[str, str]:
    names = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
        "config.json",
    }
    files = [
        path
        for path in model_dir.iterdir()
        if path.is_file() and (path.name in names or path.name.startswith("tokenizer."))
    ]
    if not files:
        raise ValueError(f"no tokenizer/config identity files under {model_dir}")
    return {path.name: sha256_file(path) for path in sorted(files)}


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


def validate_plan(
    plan: dict[str, Any],
    *,
    prereg_sha: str,
    model_manifest_sha: str,
    code_manifest_sha: str,
) -> list[dict[str, Any]]:
    if plan.get("schema_version") != 1:
        raise ValueError(f"unsupported N6 plan schema: {plan.get('schema_version')}")
    if plan.get("status") != "frozen_before_n6_model_output":
        raise ValueError(f"N6 provisional plan is not frozen: {plan.get('status')}")
    if plan.get("preregistration_sha256") != prereg_sha:
        raise ValueError("N6 plan/preregistration hash mismatch")
    if (
        plan.get("inputs", {}).get("model_manifest_sha256")
        != model_manifest_sha
    ):
        raise ValueError("N6 plan/model-manifest hash mismatch")
    if plan.get("inputs", {}).get("code_manifest_sha256") != code_manifest_sha:
        raise ValueError("N6 plan/code-manifest hash mismatch")
    rows = plan.get("rows")
    expected = int(plan.get("checks", {}).get("n_rows", -1))
    if not isinstance(rows, list) or len(rows) != expected or expected < 1:
        raise ValueError(
            f"N6 plan rows/check count mismatch: rows={len(rows) if isinstance(rows, list) else None}, "
            f"expected={expected}"
        )
    require_unique((str(row.get("row_uid")) for row in rows), "plan row_uid")
    require_unique(
        (str(row.get("content_group_id")) for row in rows), "plan content_group_id"
    )
    require_unique((int(row.get("doc_id")) for row in rows), "plan doc_id")
    required = {
        "row_uid",
        "content_group_id",
        "doc_id",
        "orig_index",
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
        "text_sha256",
        "input_ids_sha256",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"plan row {index} lacks {sorted(missing)}")
        ids = [int(value) for value in row["input_ids"]]
        position = int(row["position"])
        if len(ids) != int(row["seq_len"]):
            raise ValueError(f"plan row {index} seq_len mismatch")
        if not 64 <= position <= 480 or position + 16 >= len(ids):
            raise ValueError(f"plan row {index} violates frozen causal window")
        if int(row["token_id"]) != ids[position]:
            raise ValueError(f"plan row {index} token_id mismatch")
        if str(row["input_ids_sha256"]) != shahex(*ids):
            raise ValueError(f"plan row {index} input_ids_sha256 mismatch")
        if row["norm"] != "none":
            raise ValueError(f"plan row {index} has unexpected norm")
    return rows


def attach_metadata(table: pa.Table, values: dict[str, Any]) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    for key, value in values.items():
        metadata[f"{METADATA_PREFIX}{key}".encode("ascii")] = str(value).encode(
            "ascii"
        )
    return table.replace_schema_metadata(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--layer-index", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.layer_index != 32 or args.dtype != "bfloat16":
        raise ValueError("N6 contract requires layer-index=32 and dtype=bfloat16")
    output_paths = (
        args.out,
        args.out.with_suffix(args.out.suffix + ".sha256"),
        args.out.with_suffix(args.out.suffix + ".json"),
        args.out.with_suffix(args.out.suffix + ".json.sha256"),
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite a frozen N6 extraction")

    started = time.time()
    prereg_sha = require_binding_preregistration(args.prereg)
    code_manifest_sha = verify_code_manifest(args.code_manifest, __file__)
    plan_sha = verify_sha256_sidecar(args.plan)
    model_manifest_sha = verify_sha256_sidecar(args.model_manifest)
    frozen_model_manifest = parse_weight_manifest(args.model_manifest)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = validate_plan(
        plan,
        prereg_sha=prereg_sha,
        model_manifest_sha=model_manifest_sha,
        code_manifest_sha=code_manifest_sha,
    )
    current_tokenizer_hashes = tokenizer_file_hashes(args.base_model)
    if current_tokenizer_hashes != plan.get("inputs", {}).get(
        "tokenizer_file_sha256"
    ):
        raise ValueError("base tokenizer/config differs from provisional plan")
    config_path = args.base_model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"base-model config is missing: {config_path}")
    base_config_sha = sha256_file(config_path)
    actual_base_manifest = model_file_manifest(args.base_model)
    validate_model_subset({"base_model": actual_base_manifest}, frozen_model_manifest)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    device = next(model.parameters()).device
    layers = resolve_layers(model)
    if not 0 <= args.layer_index < len(layers):
        raise ValueError("layer index is outside the model")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output
        raise _StopForward

    handle = layers[args.layer_index].register_forward_hook(hook)
    vectors: list[np.ndarray] = []
    try:
        for ordinal, row in enumerate(rows, 1):
            ids = torch.tensor(
                [[int(value) for value in row["input_ids"]]],
                dtype=torch.long,
                device=device,
            )
            captured.clear()
            with torch.inference_mode():
                try:
                    model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                    )
                except _StopForward:
                    pass
            hidden = captured.pop("hidden", None)
            if hidden is None:
                raise RuntimeError(f"L32 hook did not fire for {row['row_uid']}")
            vector = (
                hidden[0, int(row["position"])]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise RuntimeError(f"invalid activation for {row['row_uid']}")
            vectors.append(vector)
            if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(rows):
                print(
                    f"[N6 activation {ordinal}/{len(rows)}] "
                    f"uid={row['row_uid'][:14]} source={row['source']}",
                    flush=True,
                )
    finally:
        handle.remove()

    x = np.stack(vectors).astype(np.float32)
    columns: dict[str, Any] = {
        "activation_vector": pa.array(
            [value.tolist() for value in x], type=pa.list_(pa.float32())
        ),
        "row_uid": pa.array([str(row["row_uid"]) for row in rows], pa.string()),
        "content_group_id": pa.array(
            [str(row["content_group_id"]) for row in rows], pa.string()
        ),
        "split": pa.array(["provisional"] * len(rows), pa.string()),
        "doc_id": pa.array([int(row["doc_id"]) for row in rows], pa.int64()),
        "orig_index": pa.array(
            [int(row["orig_index"]) for row in rows], pa.int64()
        ),
        "passage_id": pa.array([-1] * len(rows), pa.int64()),
        "position": pa.array(
            [int(row["position"]) for row in rows], pa.int32()
        ),
        "input_ids": pa.array(
            [[int(value) for value in row["input_ids"]] for row in rows],
            type=pa.list_(pa.int32()),
        ),
        "token": pa.array([str(row["token"]) for row in rows], pa.string()),
        "token_id": pa.array(
            [int(row["token_id"]) for row in rows], pa.int32()
        ),
        "corpus": pa.array(["pile"] * len(rows), pa.string()),
        "source": pa.array([str(row["source"]) for row in rows], pa.string()),
        "lang": pa.array(["en"] * len(rows), pa.string()),
        "seq_len": pa.array([int(row["seq_len"]) for row in rows], pa.int32()),
        "context_tail": pa.array(
            [str(row["context_tail"]) for row in rows], pa.string()
        ),
        "continuation": pa.array(
            [str(row["continuation"]) for row in rows], pa.string()
        ),
        "norm": pa.array(["none"] * len(rows), pa.string()),
        "text_sha256": pa.array(
            [str(row["text_sha256"]) for row in rows], pa.string()
        ),
        "input_ids_sha256": pa.array(
            [str(row["input_ids_sha256"]) for row in rows], pa.string()
        ),
    }
    script_sha = sha256_file(__file__)
    table = attach_metadata(
        pa.table(columns),
        {
            "schema_version": 1,
            "plan_sha256": plan_sha,
            "preregistration_sha256": prereg_sha,
            "script_sha256": script_sha,
            "code_manifest_sha256": code_manifest_sha,
            "model_manifest_sha256": model_manifest_sha,
            "layer_index": args.layer_index,
            "dtype": args.dtype,
            "batch_size": 1,
            "full_frozen_sequence": "true",
            "n_rows": len(rows),
            "base_config_sha256": base_config_sha,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary extraction: {temporary}")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, args.out)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_sha = sha256_file(args.out)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{output_sha}  {args.out.name}\n", encoding="utf-8"
    )
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    report = {
        "schema_version": 1,
        "experiment": "N6 frozen-plan L32 activation extraction",
        "status": "complete",
        "n_rows": len(rows),
        "inputs": {
            "plan": str(args.plan),
            "plan_sha256": plan_sha,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha,
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha,
            "code_manifest": str(args.code_manifest),
            "code_manifest_sha256": code_manifest_sha,
            "base_model": str(args.base_model),
            "base_config_sha256": base_config_sha,
            "actual_base_model_manifest": actual_base_manifest,
            "tokenizer_file_sha256": current_tokenizer_hashes,
            "script_sha256": script_sha,
        },
        "protocol": {
            "layer_index": args.layer_index,
            "dtype": args.dtype,
            "batch_size": 1,
            "full_frozen_sequence": True,
            "retokenized": False,
            "resampled": False,
            "analysis_eligibility_applied": False,
        },
        "checks": {
            "row_uid_sequence_sha256": canonical_sha256(
                [row["row_uid"] for row in rows]
            ),
            "activation_width": int(x.shape[1]),
            "all_finite": bool(np.isfinite(x).all()),
            "norm_min": float(norms.min()),
            "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()),
        },
        "outputs": {
            "parquet": str(args.out),
            "parquet_sha256": output_sha,
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "forward_count": len(rows),
    }
    report_path = args.out.with_suffix(args.out.suffix + ".json")
    report_payload = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    report_path.write_text(report_payload, encoding="utf-8")
    report_sha = sha256_file(report_path)
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(
        f"{report_sha}  {report_path.name}\n", encoding="utf-8"
    )
    print(
        f"N6_ACTIVATION_EXTRACTION_COMPLETE rows={len(rows)} "
        f"width={x.shape[1]} sha256={output_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
