#!/usr/bin/env python3
"""Extract N5 layer-32 activations from an immutable tokenizer-only plan.

This script performs no tokenization, sampling, filtering, or split assignment.
It accepts exactly one split per invocation so discovery can be completed and
the gate frozen before any held-out GPU work starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM


EXPECTED_PREREG_SHA256 = (
    "63dc31b4f9607e54ac15f1c364fcae2ee903f228fe0afb4d388c6dad1a6f9103"
)
EXPECTED_MODEL_MANIFEST_SHA256 = (
    "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735"
)
EXTRACTION_METADATA_PREFIX = "nla.n5_activation_extraction."


class _StopForward(Exception):
    """Terminate a forward immediately after the frozen residual layer."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_sha256_sidecar(path: Path, observed_sha256: str) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise SystemExit(f"frozen SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise SystemExit(f"malformed frozen SHA-256 sidecar: {sidecar}")
    declared_sha256, declared_name = fields
    if declared_sha256.lower() != observed_sha256.lower():
        raise SystemExit(
            "frozen plan/sidecar SHA-256 mismatch: "
            f"plan={observed_sha256}, sidecar={declared_sha256}"
        )
    if declared_name != path.name:
        raise SystemExit(
            f"frozen sidecar names {declared_name!r}, expected {path.name!r}"
        )
    return sidecar


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
        raise SystemExit(f"no tokenizer/config identity files found under {model_dir}")
    return {path.name: sha256_file(path) for path in sorted(files)}


def resolve_layers(model: torch.nn.Module):
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
    ):
        current = model
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None:
            return current
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def validate_plan(plan: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if plan.get("schema_version") != 2:
        raise SystemExit(f"unsupported N5 plan schema: {plan.get('schema_version')}")
    if plan.get("status") != "frozen_before_base_model_load":
        raise SystemExit(f"N5 plan is not frozen: {plan.get('status')}")
    if plan.get("preregistration_sha256") != EXPECTED_PREREG_SHA256:
        raise SystemExit("N5 plan preregistration hash mismatch")
    checks = plan.get("checks", {})
    mandatory_checks = {
        "n_rows": 600,
        "n_content_groups": 600,
        "n_row_uids": 600,
        "n_plan_doc_ids": 600,
        "n4_prefix_shingle_overlap_count": 0,
        "n5_internal_shingle_overlap_count": 0,
        "template_or_blank_token_count": 0,
    }
    for key, expected in mandatory_checks.items():
        if checks.get(key) != expected:
            raise SystemExit(
                f"N5 plan QA {key!r}: expected {expected!r}, got {checks.get(key)!r}"
            )
    required_v2_checks = {
        "xnli_raw_row_to_unit_mapping_matches_frozen": True,
        "xnli_selected_candidate_groups_unique": True,
        "xnli_all_candidate_unit_identities_unique": True,
        "xnli_selected_unit_identity_count": 600,
        "xnli_selected_unit_identities_unique": True,
        "xnli_unit_embargo_disjoint": True,
        "xnli_candidate_group_split_disjoint": True,
        "xnli_unit_split_disjoint": True,
    }
    for key, expected in required_v2_checks.items():
        if checks.get(key) != expected:
            raise SystemExit(
                f"N5-v2 plan QA {key!r}: expected {expected!r}, "
                f"got {checks.get(key)!r}"
            )
    if not checks.get("all_input_ids_frozen"):
        raise SystemExit("N5 plan does not freeze all input_ids")

    all_rows = plan.get("rows", [])
    if len(all_rows) != 600:
        raise SystemExit(f"N5 plan rows expected 600, got {len(all_rows)}")
    if Counter(str(row.get("split")) for row in all_rows) != Counter(
        {"discovery": 200, "heldout": 400}
    ):
        raise SystemExit("N5 plan does not contain the frozen 200/400 split")
    for identity_field in ("content_group_id", "row_uid", "doc_id"):
        values = [str(row.get(identity_field)) for row in all_rows]
        if len(set(values)) != 600:
            raise SystemExit(
                f"N5 plan repeats or omits cross-split {identity_field}"
            )
    rows = [row for row in all_rows if row.get("split") == split]
    expected_n = 200 if split == "discovery" else 400
    if len(rows) != expected_n:
        raise SystemExit(
            f"N5 {split} expected {expected_n} rows, observed {len(rows)}"
        )
    if len({row["content_group_id"] for row in rows}) != len(rows):
        raise SystemExit(f"N5 {split} repeats a content_group_id")
    if len({row["row_uid"] for row in rows}) != len(rows):
        raise SystemExit(f"N5 {split} repeats a row_uid")
    if len({int(row["doc_id"]) for row in rows}) != len(rows):
        raise SystemExit(f"N5 {split} repeats a plan-local doc_id")

    required = {
        "row_uid",
        "content_group_id",
        "split",
        "doc_id",
        "orig_index",
        "passage_id",
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
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise SystemExit(f"plan row {index} missing fields: {sorted(missing)}")
        input_ids = [int(token) for token in row["input_ids"]]
        position = int(row["position"])
        if len(input_ids) != int(row["seq_len"]) or not 0 <= position < len(input_ids):
            raise SystemExit(f"invalid frozen sequence/position at row {index}")
        if int(row["token_id"]) != input_ids[position]:
            raise SystemExit(f"frozen token_id mismatch at row {index}")
        if position < 64 or position > 480 or position + 16 >= len(input_ids):
            raise SystemExit(f"frozen causal-window invariant failed at row {index}")
        if row["norm"] != "none":
            raise SystemExit(f"unexpected frozen norm at row {index}: {row['norm']}")
    return rows


def attach_metadata(
    table: pa.Table,
    *,
    plan_sha256: str,
    prereg_sha256: str,
    script_sha256: str,
    split: str,
    layer_index: int,
    dtype: str,
    base_config_sha256: str,
    model_manifest_sha256: str,
) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    values = {
        "schema_version": "1",
        "plan_sha256": plan_sha256,
        "preregistration_sha256": prereg_sha256,
        "script_sha256": script_sha256,
        "split": split,
        "layer_index": str(layer_index),
        "dtype": dtype,
        "batch_size": "1",
        "full_frozen_sequence": "true",
        "base_config_sha256": base_config_sha256,
        "model_manifest_sha256": model_manifest_sha256,
    }
    metadata.update(
        {
            f"{EXTRACTION_METADATA_PREFIX}{key}".encode("ascii"): value.encode(
                "ascii"
            )
            for key, value in values.items()
        }
    )
    return table.replace_schema_metadata(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument(
        "--split", required=True, choices=("discovery", "heldout")
    )
    parser.add_argument("--layer-index", type=int, default=32)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sidecar = args.out.with_suffix(args.out.suffix + ".sha256")
    report_path = args.out.with_suffix(args.out.suffix + ".json")
    report_sidecar = report_path.with_suffix(report_path.suffix + ".sha256")
    existing_outputs = [
        path for path in (args.out, sidecar, report_path, report_sidecar) if path.exists()
    ]
    if existing_outputs:
        raise SystemExit(
            "refusing to overwrite an existing frozen extraction artifact: "
            + ", ".join(map(str, existing_outputs))
        )
    plan_sha256 = sha256_file(args.plan)
    plan_sidecar = verify_sha256_sidecar(args.plan, plan_sha256)
    model_manifest_sha256 = sha256_file(args.model_manifest)
    if model_manifest_sha256 != EXPECTED_MODEL_MANIFEST_SHA256:
        raise SystemExit(
            "full model-weight manifest SHA-256 mismatch: "
            f"expected {EXPECTED_MODEL_MANIFEST_SHA256}, "
            f"observed {model_manifest_sha256}"
        )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = validate_plan(plan, args.split)

    current_tokenizer_hashes = tokenizer_file_hashes(args.base_model)
    planned_tokenizer_hashes = (
        plan.get("inputs", {}).get("tokenizer_file_sha256", {})
    )
    if current_tokenizer_hashes != planned_tokenizer_hashes:
        raise SystemExit(
            "base-model tokenizer/config identity differs from the frozen N5 plan"
        )
    base_config = args.base_model / "config.json"
    if not base_config.exists():
        raise SystemExit(f"base model config missing: {base_config}")
    base_config_sha256 = sha256_file(base_config)

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device
    layers = resolve_layers(model)
    if not 0 <= args.layer_index < len(layers):
        raise SystemExit(
            f"layer {args.layer_index} outside model with {len(layers)} layers"
        )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden
        raise _StopForward

    handle = layers[args.layer_index].register_forward_hook(hook)
    vectors: list[np.ndarray] = []
    try:
        for ordinal, row in enumerate(rows, 1):
            input_ids = torch.tensor(
                [[int(token) for token in row["input_ids"]]],
                dtype=torch.long,
                device=device,
            )
            captured.clear()
            with torch.inference_mode():
                try:
                    model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                    )
                except _StopForward:
                    pass
            if "hidden" not in captured:
                raise RuntimeError(
                    f"layer-{args.layer_index} hook did not fire for {row['row_uid']}"
                )
            hidden = captured.pop("hidden")
            position = int(row["position"])
            vector = (
                hidden[0, position].detach().float().cpu().numpy().astype(np.float32)
            )
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise RuntimeError(f"invalid activation for {row['row_uid']}")
            vectors.append(vector)
            if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(rows):
                print(
                    f"[{args.split} {ordinal:>3}/{len(rows)}] "
                    f"uid={row['row_uid'][:12]} src={row['source']} "
                    f"pos={position} seq={len(row['input_ids'])}",
                    flush=True,
                )
    finally:
        handle.remove()

    activation_array = np.stack(vectors).astype(np.float32)
    columns: dict[str, Any] = {
        "activation_vector": pa.array(
            [vector.tolist() for vector in activation_array],
            type=pa.list_(pa.float32()),
        ),
        "row_uid": pa.array([str(row["row_uid"]) for row in rows], pa.string()),
        "content_group_id": pa.array(
            [str(row["content_group_id"]) for row in rows], pa.string()
        ),
        "split": pa.array([str(row["split"]) for row in rows], pa.string()),
        "doc_id": pa.array([int(row["doc_id"]) for row in rows], pa.int64()),
        "orig_index": pa.array(
            [int(row.get("orig_index", -1)) for row in rows], pa.int64()
        ),
        "passage_id": pa.array(
            [int(row.get("passage_id", -1)) for row in rows], pa.int64()
        ),
        "position": pa.array(
            [int(row["position"]) for row in rows], pa.int32()
        ),
        "input_ids": pa.array(
            [[int(token) for token in row["input_ids"]] for row in rows],
            type=pa.list_(pa.int32()),
        ),
        "token": pa.array([str(row["token"]) for row in rows], pa.string()),
        "token_id": pa.array(
            [int(row["token_id"]) for row in rows], pa.int32()
        ),
        "corpus": pa.array([str(row["corpus"]) for row in rows], pa.string()),
        "source": pa.array([str(row["source"]) for row in rows], pa.string()),
        "lang": pa.array([str(row["lang"]) for row in rows], pa.string()),
        "seq_len": pa.array([int(row["seq_len"]) for row in rows], pa.int32()),
        "context_tail": pa.array(
            [str(row["context_tail"]) for row in rows], pa.string()
        ),
        "continuation": pa.array(
            [str(row["continuation"]) for row in rows], pa.string()
        ),
        "norm": pa.array([str(row["norm"]) for row in rows], pa.string()),
        "text_sha256": pa.array(
            [str(row["text_sha256"]) for row in rows], pa.string()
        ),
        "input_ids_sha256": pa.array(
            [str(row["input_ids_sha256"]) for row in rows], pa.string()
        ),
    }
    table = pa.table(columns)
    script_sha256 = sha256_file(__file__)
    table = attach_metadata(
        table,
        plan_sha256=plan_sha256,
        prereg_sha256=plan["preregistration_sha256"],
        script_sha256=script_sha256,
        split=args.split,
        layer_index=args.layer_index,
        dtype=args.dtype,
        base_config_sha256=base_config_sha256,
        model_manifest_sha256=model_manifest_sha256,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise SystemExit(f"temporary output path unexpectedly exists: {temporary}")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, args.out)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_sha256 = sha256_file(args.out)
    sidecar.write_text(
        f"{output_sha256}  {args.out.name}\n", encoding="utf-8"
    )

    norms = np.linalg.norm(activation_array.astype(np.float64), axis=1)
    report = {
        "schema_version": 1,
        "experiment": "N5 frozen-plan layer-32 activation extraction",
        "status": "complete",
        "split": args.split,
        "n_rows": len(rows),
        "n_content_groups": len({row["content_group_id"] for row in rows}),
        "inputs": {
            "plan": str(args.plan),
            "plan_sha256": plan_sha256,
            "plan_sha256_sidecar": str(plan_sidecar),
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha256,
            "preregistration_sha256": plan["preregistration_sha256"],
            "base_model": str(args.base_model),
            "base_config_sha256": base_config_sha256,
            "tokenizer_file_sha256": current_tokenizer_hashes,
            "script_sha256": script_sha256,
        },
        "protocol": {
            "layer_index": args.layer_index,
            "dtype": args.dtype,
            "batch_size": 1,
            "full_frozen_sequence": True,
            "early_exit_after_layer": args.layer_index,
            "retokenized": False,
            "resampled": False,
        },
        "checks": {
            "row_uid_unique": len({row["row_uid"] for row in rows}) == len(rows),
            "content_group_id_unique": (
                len({row["content_group_id"] for row in rows}) == len(rows)
            ),
            "all_finite": bool(np.isfinite(activation_array).all()),
            "activation_width": int(activation_array.shape[1]),
            "norm_mean": float(norms.mean()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
        },
        "outputs": {
            "parquet": str(args.out),
            "parquet_sha256": output_sha256,
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "forward_count": len(rows),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_sha256 = sha256_file(report_path)
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(
        f"{report_sha256}  {report_path.name}\n", encoding="utf-8"
    )

    print("N5_ACTIVATION_EXTRACTION_COMPLETE")
    print(
        f"split={args.split} rows={len(rows)} width={activation_array.shape[1]} "
        f"mean_norm={norms.mean():.3f} elapsed={report['elapsed_seconds']:.1f}s"
    )
    print(f"parquet_sha256={output_sha256}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
