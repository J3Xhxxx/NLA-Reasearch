#!/usr/bin/env python3
"""Generate only provisional N6 AV explanations and freeze them before AR/SAE.

Generation is greedy and resumable through an append-only, contract-bound
checkpoint.  This stage never loads the AR, either SAE, or the base model.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from n6_common import (
    canonical_sha256,
    model_file_manifest,
    parse_weight_manifest,
    require_binding_preregistration,
    require_unique,
    sha256_file,
    validate_model_subset,
    verify_code_manifest,
    verify_sha256_sidecar,
    write_or_verify_frozen_json,
)
from pilot_common import AVLocal


ACTIVATION_METADATA_PREFIX = "nla.n6_activation_extraction."
MAX_NEW_TOKENS = 200


def paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", text) if part.strip()])


def table_metadata(table) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="strict"): value.decode(
            "utf-8", errors="strict"
        )
        for key, value in (table.schema.metadata or {}).items()
    }


def load_inputs(
    *,
    activation_path: Path,
    plan: dict[str, Any],
    plan_sha: str,
    prereg_sha: str,
    model_manifest_sha: str,
    code_manifest_sha: str,
) -> tuple[np.ndarray, dict[str, list[Any]], dict[str, Any]]:
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("N6 provisional plan has no rows")
    expected = int(plan.get("checks", {}).get("n_rows", -1))
    if len(rows) != expected:
        raise ValueError("N6 plan row count/check mismatch")
    if plan.get("preregistration_sha256") != prereg_sha:
        raise ValueError("N6 plan/preregistration hash mismatch")
    if plan.get("inputs", {}).get("model_manifest_sha256") != model_manifest_sha:
        raise ValueError("N6 plan/model-manifest hash mismatch")
    if plan.get("inputs", {}).get("code_manifest_sha256") != code_manifest_sha:
        raise ValueError("N6 plan/code-manifest hash mismatch")

    table = pq.read_table(activation_path)
    required = {
        "activation_vector",
        "row_uid",
        "content_group_id",
        "doc_id",
        "orig_index",
        "position",
        "token",
        "token_id",
        "source",
        "input_ids",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"activation parquet lacks {missing}")
    if table.num_rows != expected:
        raise ValueError(
            f"activation rows {table.num_rows} != provisional plan {expected}"
        )
    metadata = table_metadata(table)
    expected_metadata = {
        "plan_sha256": plan_sha,
        "preregistration_sha256": prereg_sha,
        "model_manifest_sha256": model_manifest_sha,
        "code_manifest_sha256": code_manifest_sha,
        "layer_index": "32",
        "dtype": "bfloat16",
        "batch_size": "1",
        "full_frozen_sequence": "true",
        "n_rows": str(expected),
    }
    for key, value in expected_metadata.items():
        full_key = f"{ACTIVATION_METADATA_PREFIX}{key}"
        if metadata.get(full_key) != value:
            raise ValueError(
                f"activation metadata {full_key}={metadata.get(full_key)!r}, "
                f"expected {value!r}"
            )

    meta = {
        name: table[name].to_pylist()
        for name in required
        if name != "activation_vector"
    }
    row_uids = [str(value) for value in meta["row_uid"]]
    plan_uids = [str(row["row_uid"]) for row in rows]
    if row_uids != plan_uids:
        raise ValueError("activation row order differs from provisional plan")
    require_unique(row_uids, "activation row_uid")
    require_unique(
        (str(value) for value in meta["content_group_id"]),
        "activation content_group_id",
    )
    for index, planned in enumerate(rows):
        for field in (
            "content_group_id",
            "doc_id",
            "orig_index",
            "position",
            "token",
            "token_id",
            "source",
        ):
            if str(meta[field][index]) != str(planned[field]):
                raise ValueError(f"activation/plan mismatch at row {index}.{field}")
        if [int(value) for value in meta["input_ids"][index]] != [
            int(value) for value in planned["input_ids"]
        ]:
            raise ValueError(f"activation/plan input_ids mismatch at row {index}")
    vectors = np.asarray(
        table["activation_vector"].combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != expected:
        raise ValueError(f"invalid activation shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("activation vectors contain non-finite values")
    return vectors, meta, metadata


def load_checkpoint(
    path: Path,
    *,
    contract_sha: str,
    row_uids: list[str],
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("contract_sha256") != contract_sha:
            raise ValueError(f"checkpoint contract mismatch at line {line_number}")
        index = int(row["idx"])
        if not 0 <= index < len(row_uids):
            raise ValueError(f"checkpoint idx out of range at line {line_number}")
        if index in completed:
            raise ValueError(f"duplicate checkpoint idx {index}")
        if str(row.get("row_uid")) != row_uids[index]:
            raise ValueError(f"checkpoint row_uid mismatch at line {line_number}")
        completed[index] = row
    return completed


def append_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    if args.max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError(f"N6 AV generation fixes max-new-tokens={MAX_NEW_TOKENS}")
    started = time.time()
    prereg_sha = require_binding_preregistration(args.prereg)
    code_manifest_sha = verify_code_manifest(
        args.code_manifest,
        __file__,
        extra_paths=(
            Path(__file__).with_name("pilot_common.py"),
            Path(os.environ.get("NLA_REPO", "/root/autodl-tmp/nla_repo"))
            / "nla_inference.py",
        ),
    )
    activation_sha = verify_sha256_sidecar(args.activations)
    plan_sha = verify_sha256_sidecar(args.plan)
    model_manifest_sha = verify_sha256_sidecar(args.model_manifest)
    frozen_model_manifest = parse_weight_manifest(args.model_manifest)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    x, meta, activation_metadata = load_inputs(
        activation_path=args.activations,
        plan=plan,
        plan_sha=plan_sha,
        prereg_sha=prereg_sha,
        model_manifest_sha=model_manifest_sha,
        code_manifest_sha=code_manifest_sha,
    )
    av_manifest = model_file_manifest(args.av)
    validate_model_subset({"av": av_manifest}, frozen_model_manifest)
    script_sha = sha256_file(__file__)
    row_uids = [str(value) for value in meta["row_uid"]]
    contract = {
        "experiment": "N6 provisional AV generation",
        "activations_sha256": activation_sha,
        "plan_sha256": plan_sha,
        "prereg_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": model_manifest_sha,
        "actual_av_manifest_sha256": av_manifest["manifest_sha256"],
        "script_sha256": script_sha,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "row_uid_sequence_sha256": canonical_sha256(row_uids),
    }
    contract_sha = canonical_sha256(contract)
    completed = load_checkpoint(
        args.checkpoint,
        contract_sha=contract_sha,
        row_uids=row_uids,
    )
    missing = [index for index in range(len(row_uids)) if index not in completed]
    print(
        f"[N6 AV] rows={len(row_uids)} checkpoint={len(completed)} "
        f"missing={len(missing)}",
        flush=True,
    )
    if missing:
        av = AVLocal(args.av, device="cuda")
        try:
            for index in missing:
                explanation = av.generate(
                    x[index],
                    temperature=0.0,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                row = {
                    "contract_sha256": contract_sha,
                    "idx": index,
                    "row_uid": row_uids[index],
                    "content_group_id": str(meta["content_group_id"][index]),
                    "doc_id": int(meta["doc_id"][index]),
                    "orig_index": int(meta["orig_index"][index]),
                    "position": int(meta["position"][index]),
                    "token": str(meta["token"][index]),
                    "token_id": int(meta["token_id"][index]),
                    "source": str(meta["source"][index]),
                    "explanation": explanation,
                    "paragraph_count": paragraph_count(explanation),
                    "explanation_utf8_sha256": sha256_file_bytes(
                        explanation.encode("utf-8")
                    ),
                }
                append_checkpoint(args.checkpoint, row)
                completed[index] = row
                print(
                    f"[N6 AV {len(completed)}/{len(row_uids)}] "
                    f"uid={row_uids[index][:14]} "
                    f"paragraphs={row['paragraph_count']}",
                    flush=True,
                )
        finally:
            del av
            gc.collect()
            torch.cuda.empty_cache()

    rows = []
    for index in range(len(row_uids)):
        source = completed[index]
        rows.append(
            {
                key: source[key]
                for key in (
                    "idx",
                    "row_uid",
                    "content_group_id",
                    "doc_id",
                    "orig_index",
                    "position",
                    "token",
                    "token_id",
                    "source",
                    "explanation",
                    "paragraph_count",
                    "explanation_utf8_sha256",
                )
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "N6 provisional frozen AV explanations",
        "status": "COMPLETE_FROZEN_BEFORE_TEXT_ELIGIBILITY_AND_AR",
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": activation_sha,
            "activation_metadata": activation_metadata,
            "plan": str(args.plan),
            "plan_sha256": plan_sha,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha,
            "code_manifest": str(args.code_manifest),
            "code_manifest_sha256": code_manifest_sha,
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha,
            "script_sha256": script_sha,
            "actual_av_manifest": av_manifest,
            "contract_sha256": contract_sha,
        },
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "checkpoint": str(args.checkpoint),
            "checkpoint_scope": "append-only provisional AV explanations",
        },
        "checks": {
            "n_rows": len(rows),
            "n_unique_row_uids": len(set(row_uids)),
            "row_uid_sequence_sha256": canonical_sha256(row_uids),
            "paragraph_count_distribution": {
                str(value): sum(row["paragraph_count"] == value for row in rows)
                for value in sorted({row["paragraph_count"] for row in rows})
            },
        },
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
        "av_calls_total_artifact": len(rows),
        "av_calls_this_invocation": len(missing),
    }
    output_sha = write_or_verify_frozen_json(args.out, payload)
    print(
        f"N6_AV_EXPLANATIONS_FROZEN rows={len(rows)} sha256={output_sha}",
        flush=True,
    )


def sha256_file_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
