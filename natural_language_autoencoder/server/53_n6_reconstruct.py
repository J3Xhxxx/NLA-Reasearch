#!/usr/bin/env python3
"""Reconstruct frozen N6 variants with AR and SAE-big only.

The AV is never loaded.  Every text is read from the frozen variant artifact,
and centered geometry is bound to the already-frozen N5 gate-v2 direction.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from n6_common import (
    canonical_sha256,
    model_file_manifest,
    parse_weight_manifest,
    prediction_scores,
    require_binding_preregistration,
    retrieval,
    scalar_npz_string,
    sha256_file,
    unit,
    validate_model_subset,
    verify_code_manifest,
    verify_sha256_sidecar,
    write_new_json,
)
from pilot_common import JumpReLUSAE, NLACritic


TEXT_CONDITIONS = (
    "orig",
    "p3_true",
    "p3_cross_matched",
    "p3_candidate_strip",
    "p3_anchor_strip",
    "p3_all_quote_strip",
    "p12",
)
PREDICTION_KEYS = {
    condition: f"pred_{condition}" for condition in TEXT_CONDITIONS
}
ACTIVATION_METADATA_PREFIX = "nla.n6_activation_extraction."
N5_GATE_V2_SHA256 = (
    "036477f21fb550b317978a880df0a708dcf42a5201d301ca0757fade3baea059"
)


def table_metadata(table) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="strict"): value.decode(
            "utf-8", errors="strict"
        )
        for key, value in (table.schema.metadata or {}).items()
    }


def load_activation_table(
    path: Path,
    *,
    plan: dict[str, Any],
    plan_sha: str,
    prereg_sha: str,
    code_manifest_sha: str,
    model_manifest_sha: str,
) -> tuple[np.ndarray, dict[str, list[Any]], dict[str, str]]:
    table = pq.read_table(path)
    required = {
        "activation_vector",
        "row_uid",
        "content_group_id",
        "doc_id",
        "position",
        "source",
        "token",
        "token_id",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"activation parquet lacks {missing}")
    rows = plan.get("rows")
    if not isinstance(rows, list) or table.num_rows != len(rows):
        raise ValueError("activation/provisional-plan row count mismatch")
    meta = {
        name: table[name].to_pylist()
        for name in required
        if name != "activation_vector"
    }
    uids = [str(value) for value in meta["row_uid"]]
    if uids != [str(row["row_uid"]) for row in rows]:
        raise ValueError("activation row order differs from provisional plan")
    x = np.asarray(
        table["activation_vector"].combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"invalid provisional activation matrix {x.shape}")
    metadata = table_metadata(table)
    required_metadata = {
        "plan_sha256": plan_sha,
        "preregistration_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": model_manifest_sha,
        "layer_index": "32",
        "dtype": "bfloat16",
        "batch_size": "1",
        "full_frozen_sequence": "true",
    }
    for key, expected in required_metadata.items():
        observed = metadata.get(f"{ACTIVATION_METADATA_PREFIX}{key}")
        if observed != expected:
            raise ValueError(
                f"activation metadata {key}={observed!r}, expected {expected!r}"
            )
    return x, meta, metadata


def load_variants(
    path: Path,
    *,
    plan_sha: str,
    prereg_sha: str,
    code_manifest_sha: str,
    model_manifest_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status")
        != "COMPLETE_FROZEN_BEFORE_AR_CANDIDATE_MASS_OR_CAUSAL_OUTCOME"
    ):
        raise ValueError(f"variant artifact is not frozen: {payload.get('status')}")
    inputs = payload.get("inputs", {})
    required_hashes = {
        "plan_sha256": plan_sha,
        "prereg_sha256": prereg_sha,
        "code_manifest_sha256": code_manifest_sha,
        "model_manifest_sha256": model_manifest_sha,
    }
    for key, expected in required_hashes.items():
        if inputs.get(key) != expected:
            raise ValueError(
                f"variant artifact {key}={inputs.get(key)!r}, expected {expected!r}"
            )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("variant artifact contains no analysis rows")
    expected = payload.get("selection", {}).get("n_selected", len(rows))
    if int(expected) != len(rows):
        raise ValueError("variant selected-row count mismatch")
    required = {
        "row_uid",
        "content_group_id",
        "doc_id",
        "source",
        "candidate_count",
        "variants",
        "variant_sha256",
        "donor_row_uid",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"variant row {index} lacks {sorted(missing)}")
        variants = row["variants"]
        if set(variants) != set(TEXT_CONDITIONS):
            raise ValueError(
                f"variant row {index} has condition keys {sorted(variants)}"
            )
        for condition, text in variants.items():
            digest = sha256_text(str(text))
            if row["variant_sha256"].get(condition) != digest:
                raise ValueError(
                    f"variant row {index}.{condition} hash mismatch"
                )
    uids = [str(row["row_uid"]) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("variant row_uids are not unique")
    return payload, rows


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def center_direction(
    *,
    provisional_x: np.ndarray,
    n5_gate: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    gate_sha = verify_sha256_sidecar(n5_gate)
    if gate_sha != N5_GATE_V2_SHA256:
        raise ValueError(
            "N6 centered geometry requires frozen n5_gate_v2.json: "
            f"{gate_sha} != {N5_GATE_V2_SHA256}"
        )
    payload = json.loads(n5_gate.read_text(encoding="utf-8"))
    direction = np.asarray(
        payload.get("gate", {}).get("discovery_mean_direction"),
        dtype=np.float64,
    )
    if direction.shape != (provisional_x.shape[1],):
        raise ValueError("N5 gate mean direction has the wrong width")
    direction = unit(direction)
    return direction, {
        "reference": "frozen N5 gate-v2 discovery mean direction",
        "n_rows": None,
        "external_artifact": str(n5_gate),
        "external_artifact_sha256": gate_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--sae-big", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--n5-gate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vecs-out", required=True, type=Path)
    args = parser.parse_args()

    if args.out.exists() or args.vecs_out.exists():
        raise FileExistsError("refusing to overwrite N6 reconstruction outputs")
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
    variants_sha = verify_sha256_sidecar(args.variants)
    model_manifest_sha = verify_sha256_sidecar(args.model_manifest)
    frozen_model_manifest = parse_weight_manifest(args.model_manifest)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("preregistration_sha256") != prereg_sha:
        raise ValueError("plan/preregistration hash mismatch")
    if plan.get("inputs", {}).get("code_manifest_sha256") != code_manifest_sha:
        raise ValueError("plan/code-manifest hash mismatch")
    if plan.get("inputs", {}).get("model_manifest_sha256") != model_manifest_sha:
        raise ValueError("plan/model-manifest hash mismatch")
    provisional_x, activation_meta, activation_metadata = load_activation_table(
        args.activations,
        plan=plan,
        plan_sha=plan_sha,
        prereg_sha=prereg_sha,
        code_manifest_sha=code_manifest_sha,
        model_manifest_sha=model_manifest_sha,
    )
    variant_payload, variant_rows = load_variants(
        args.variants,
        plan_sha=plan_sha,
        prereg_sha=prereg_sha,
        code_manifest_sha=code_manifest_sha,
        model_manifest_sha=model_manifest_sha,
    )
    provisional_uids = [
        str(value) for value in activation_meta["row_uid"]
    ]
    index_by_uid = {uid: index for index, uid in enumerate(provisional_uids)}
    selected_uids = [str(row["row_uid"]) for row in variant_rows]
    if any(uid not in index_by_uid for uid in selected_uids):
        raise ValueError("variant row is absent from provisional activations")
    selected_indices = np.asarray(
        [index_by_uid[uid] for uid in selected_uids], dtype=np.int64
    )
    x = provisional_x[selected_indices]
    for local, row in enumerate(variant_rows):
        global_index = int(selected_indices[local])
        for field in ("content_group_id", "doc_id", "source"):
            if str(row[field]) != str(activation_meta[field][global_index]):
                raise ValueError(f"variant/activation mismatch at {local}.{field}")
    m_hat, center_report = center_direction(
        provisional_x=provisional_x,
        n5_gate=args.n5_gate,
    )

    actual_manifests = {
        "ar": model_file_manifest(args.ar),
        "sae_big": model_file_manifest(args.sae_big),
    }
    validate_model_subset(actual_manifests, frozen_model_manifest)
    script_sha = sha256_file(__file__)

    critic = NLACritic(args.ar, device="cuda")
    cache: dict[str, np.ndarray] = {}

    def reconstruct(text: str) -> np.ndarray:
        if text not in cache:
            vector = critic.reconstruct(text).numpy().astype(np.float32)
            if vector.shape != (x.shape[1],) or not np.isfinite(vector).all():
                raise ValueError("AR produced an invalid vector")
            cache[text] = vector
            if len(cache) % 100 == 0:
                print(f"[N6 AR] {len(cache)} unique texts", flush=True)
        return cache[text]

    predictions: dict[str, np.ndarray] = {}
    try:
        for condition in TEXT_CONDITIONS:
            predictions[condition] = np.stack(
                [
                    reconstruct(str(row["variants"][condition]))
                    for row in variant_rows
                ]
            ).astype(np.float32)
            print(f"[N6 AR] {condition} complete", flush=True)
    finally:
        del critic
        gc.collect()
        torch.cuda.empty_cache()

    xt = torch.from_numpy(x)
    sae = JumpReLUSAE(str(args.sae_big), device="cuda")
    recon_t, acts = sae(xt)
    recon_sae_big = recon_t.float().cpu().numpy().astype(np.float32)
    l0_sae_big = (acts > 0).sum(1).cpu().numpy().astype(np.int64)
    del sae, recon_t, acts, xt
    gc.collect()
    torch.cuda.empty_cache()
    predictions["sae_big"] = recon_sae_big
    for condition, prediction in predictions.items():
        if prediction.shape != x.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{condition} has invalid shape/values")

    scores = {
        condition: prediction_scores(prediction, x, m_hat)
        for condition, prediction in predictions.items()
    }
    summary = {}
    for condition, values in scores.items():
        summary[condition] = {
            "n": len(x),
            "mean_cos_c": float(values["cos_c"].mean()),
            "median_cos_c": float(np.median(values["cos_c"])),
            "mean_cos_raw": float(values["cos_raw"].mean()),
            "mean_l2_error": float(values["l2_error"].mean()),
            "retrieval": retrieval(predictions[condition], x, m_hat),
        }
    summary["sae_big"]["mean_l0"] = float(l0_sae_big.mean())

    rows_out = []
    for index, row in enumerate(variant_rows):
        rows_out.append(
            {
                "idx": index,
                "provisional_index": int(selected_indices[index]),
                "row_uid": str(row["row_uid"]),
                "content_group_id": str(row["content_group_id"]),
                "doc_id": int(row["doc_id"]),
                "source": str(row["source"]),
                "candidate_count": int(row["candidate_count"]),
                "donor_row_uid": str(row["donor_row_uid"]),
                "scores": {
                    condition: {
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
                    for condition, values in scores.items()
                },
            }
        )

    npz_payload: dict[str, np.ndarray] = {
        "x": x.astype(np.float32),
        "m_hat": m_hat.astype(np.float64),
        "selected_provisional_indices": selected_indices,
        "row_uids": np.asarray(selected_uids, dtype=np.str_),
        "content_group_ids": np.asarray(
            [str(row["content_group_id"]) for row in variant_rows], dtype=np.str_
        ),
        "doc_ids": np.asarray(
            [int(row["doc_id"]) for row in variant_rows], dtype=np.int64
        ),
        "sources": np.asarray(
            [str(row["source"]) for row in variant_rows], dtype=np.str_
        ),
        "candidate_counts": np.asarray(
            [int(row["candidate_count"]) for row in variant_rows], dtype=np.int64
        ),
        "recon_sae_big": recon_sae_big,
        "l0_sae_big": l0_sae_big,
        "activations_sha256": np.asarray(activation_sha),
        "plan_sha256": np.asarray(plan_sha),
        "variants_sha256": np.asarray(variants_sha),
        "prereg_sha256": np.asarray(prereg_sha),
        "code_manifest_sha256": np.asarray(code_manifest_sha),
        "model_manifest_sha256": np.asarray(model_manifest_sha),
        "center_reference": np.asarray("n5_gate_v2"),
    }
    for condition in TEXT_CONDITIONS:
        npz_payload[PREDICTION_KEYS[condition]] = predictions[condition]
    args.vecs_out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.vecs_out.with_name(args.vecs_out.name + ".partial.npz")
    if partial.exists():
        raise FileExistsError(f"stale partial NPZ requires audit: {partial}")
    np.savez_compressed(partial, **npz_payload)
    os.replace(partial, args.vecs_out)
    vecs_sha = sha256_file(args.vecs_out)
    args.vecs_out.with_suffix(args.vecs_out.suffix + ".sha256").write_text(
        f"{vecs_sha}  {args.vecs_out.name}\n", encoding="utf-8"
    )

    output = {
        "schema_version": 1,
        "experiment": "N6 frozen variant AR plus SAE-big reconstruction",
        "status": "COMPLETE",
        "inputs": {
            "activations": str(args.activations),
            "activations_sha256": activation_sha,
            "activation_metadata": activation_metadata,
            "plan": str(args.plan),
            "plan_sha256": plan_sha,
            "variants": str(args.variants),
            "variants_sha256": variants_sha,
            "prereg": str(args.prereg),
            "prereg_sha256": prereg_sha,
            "code_manifest": str(args.code_manifest),
            "code_manifest_sha256": code_manifest_sha,
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha,
            "actual_model_manifests": actual_manifests,
            "script_sha256": script_sha,
        },
        "cohort": {
            "n_rows": len(variant_rows),
            "n_content_groups": len(
                {str(row["content_group_id"]) for row in variant_rows}
            ),
            "by_source": dict(
                Counter(str(row["source"]) for row in variant_rows)
            ),
            "by_candidate_count": dict(
                Counter(str(row["candidate_count"]) for row in variant_rows)
            ),
            "row_uid_sequence_sha256": canonical_sha256(selected_uids),
        },
        "protocol": {
            "text_conditions": list(TEXT_CONDITIONS),
            "loaded_models": ["AR", "SAE-big"],
            "av_loaded": False,
            "sae_small_loaded": False,
            "centered_geometry": center_report,
            "norm_matching_applied_here": False,
            "causal_norm_matching_contract": (
                "54_n6_causal_patch.py norm-matches every nonzero substitute "
                "row-wise to x; zero remains exactly zero"
            ),
        },
        "summary": summary,
        "rows": rows_out,
        "outputs": {
            "vecs": str(args.vecs_out),
            "vecs_sha256": vecs_sha,
        },
        "npz_semantic_contract": {
            "row_axis": "variant rows and every NPZ row array share exact order",
            "target_key": "x",
            "prediction_keys": {
                **PREDICTION_KEYS,
                "sae_big": "recon_sae_big",
            },
            "npz_keys": sorted(npz_payload),
        },
        "forward_counts": {
            "ar_unique_texts": len(cache),
            "sae_big_batched_forwards": 1,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_sha = write_new_json(args.out, output)
    print(
        f"N6_RECON_COMPLETE rows={len(variant_rows)} "
        f"json={output_sha} npz={vecs_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
