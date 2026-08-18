#!/usr/bin/env python3
"""J2-P0 causal patch for SAE->AV->AR serial reconstructions.

Consumes the create-once output of 66_j2_sae_projection_loop.py and evaluates
identity, the two new language-loop reconstructions, both reverse-order
SAE(NLA(x)) comparators, and zero. Existing direct-NLA and native-SAE causal
scores remain bound to the frozen N4 artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM


EXPECTED_ACTIVATIONS = (
    "eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66"
)
EXPECTED_PROTOCOL = (
    "a41b7d89893a270218bf79e226c3e3d7a8726f71ca1fe6d41f40b583616a700f"
)
EXPECTED_MODEL_MANIFEST = (
    "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735"
)
CONDITIONS = (
    "identity",
    "small_loop",
    "big_loop",
    "direct_small",
    "direct_big",
    "zero",
)
BASE_MODEL_ROOT = "/root/autodl-tmp/models/gemma-3-12b-it"


def load_n4_module():
    path = Path(__file__).with_name("40_n4_causal_patch.py")
    spec = importlib.util.spec_from_file_location("n4_causal_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load N4 causal helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def write_frozen(path: Path, payload: Any) -> str:
    encoded = canonical_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise SystemExit(f"refusing to overwrite non-identical output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def expected_recon_contract(inputs: dict[str, Any]) -> str:
    payload = {
        "experiment": "J2-P0 SAE projection language loop",
        "conditions": ["sae_small", "sae_big"],
        "generation": {
            "temperature": 0.0,
            "max_new_tokens": 200,
            "ordering": "sae_small_0..199_then_sae_big_0..199",
        },
        "inputs": inputs,
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def verify_base_model_files(
    manifest_path: Path, base_model_root: Path
) -> dict[str, str]:
    if str(base_model_root.resolve()) != BASE_MODEL_ROOT:
        raise SystemExit(
            f"base model root mismatch: {base_model_root.resolve()} "
            f"!= {BASE_MODEL_ROOT}"
        )
    verified: dict[str, str] = {}
    prefix = BASE_MODEL_ROOT + "/"
    for line_no, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SystemExit(f"invalid model manifest line {line_no}")
        declared_path = parts[1].strip()
        if not declared_path.startswith(prefix):
            continue
        relative = declared_path[len(prefix) :]
        actual_path = base_model_root / relative
        if not actual_path.is_file():
            raise SystemExit(f"missing base model file: {actual_path}")
        actual = sha256_file(actual_path)
        if actual != parts[0].lower():
            raise SystemExit(f"base model hash mismatch: {actual_path}")
        verified[relative] = actual
    if not verified:
        raise SystemExit("model manifest has no base-model files")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--j2-vectors", required=True, type=Path)
    parser.add_argument("--j2-result", required=True, type=Path)
    parser.add_argument("--j2-explanations", required=True, type=Path)
    parser.add_argument("--j2-av-checkpoint", required=True, type=Path)
    parser.add_argument("--j2-recon-script", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layer-index", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--provenance-atol", type=float, default=0.0)
    parser.add_argument("--identity-kl-tol", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    frozen_cli = {
        "layer_index": 32,
        "horizon": 16,
        "dtype": "bfloat16",
        "provenance_atol": 0.0,
        "identity_kl_tol": 1e-5,
    }
    for key, expected in frozen_cli.items():
        if getattr(args, key) != expected:
            raise SystemExit(
                f"frozen protocol requires --{key.replace('_', '-')}={expected}"
            )

    n4, n4_path = load_n4_module()
    hashes = {
        "activations": sha256_file(args.activations),
        "j2_vectors": sha256_file(args.j2_vectors),
        "j2_result": sha256_file(args.j2_result),
        "j2_explanations": sha256_file(args.j2_explanations),
        "j2_av_checkpoint": sha256_file(args.j2_av_checkpoint),
        "j2_recon_script": sha256_file(args.j2_recon_script),
        "protocol": sha256_file(args.protocol),
        "model_manifest": sha256_file(args.model_manifest),
        "script": sha256_file(__file__),
        "n4_helpers": sha256_file(n4_path),
    }
    if hashes["activations"] != EXPECTED_ACTIVATIONS:
        raise SystemExit("activation parquet hash mismatch")
    if hashes["protocol"] != EXPECTED_PROTOCOL:
        raise SystemExit("J2 protocol hash mismatch")
    if hashes["model_manifest"] != EXPECTED_MODEL_MANIFEST:
        raise SystemExit("model manifest hash mismatch")

    j2_result = json.loads(args.j2_result.read_text(encoding="utf-8"))
    if j2_result.get("status") != "EXPLORATORY_RECON_COMPLETE":
        raise SystemExit("J2 reconstruction result is not complete")
    if j2_result.get("confirmatory") is not False:
        raise SystemExit("J2 reconstruction claim scope is invalid")
    if j2_result.get("cohort", {}).get("n") != 200 or len(
        j2_result.get("rows", [])
    ) != 200:
        raise SystemExit("J2 reconstruction row count mismatch")
    recon_inputs = j2_result.get("inputs")
    if not isinstance(recon_inputs, dict):
        raise SystemExit("J2 reconstruction inputs are missing")
    static_expected = {
        "activations": EXPECTED_ACTIVATIONS,
        "protocol": EXPECTED_PROTOCOL,
        "model_manifest": EXPECTED_MODEL_MANIFEST,
        "n4_vectors": "e9d89713dc64381a52f05224d6522abb0ec547777a8c6a7f08b841a72a339967",
        "n4_explanations": "b656ded845c8fd122e4dcb1391ba5d81e1a903f80a69c30575bf26910e200942",
        "pilot_common": "69fb1b40d60d075c615acdaa23acf4f85c17b5b4cf02e2cc18113c4e14ecf63a",
        "script": hashes["j2_recon_script"],
    }
    for key, expected in static_expected.items():
        if recon_inputs.get(key) != expected:
            raise SystemExit(f"J2 reconstruction input mismatch: {key}")
    expected_model_roots = {
        "av": "/root/autodl-tmp/models/nla-gemma3-12b-L32-av",
        "ar": "/root/autodl-tmp/models/nla-gemma3-12b-L32-ar",
        "sae_small": (
            "/root/autodl-tmp/models/gemma-scope-2-12b-it/"
            "resid_post_all/layer_32_width_16k_l0_small"
        ),
        "sae_big": (
            "/root/autodl-tmp/models/gemma-scope-2-12b-it/"
            "resid_post_all/layer_32_width_16k_l0_big"
        ),
    }
    if recon_inputs.get("model_roots") != expected_model_roots:
        raise SystemExit("J2 reconstruction model roots mismatch")
    if set(recon_inputs.get("model_files", {})) != set(expected_model_roots):
        raise SystemExit("J2 reconstruction model-file audit is incomplete")
    if j2_result.get("contract_sha256") != expected_recon_contract(
        recon_inputs
    ):
        raise SystemExit("J2 reconstruction contract digest mismatch")
    if (
        j2_result.get("outputs", {}).get("vectors_sha256")
        != hashes["j2_vectors"]
    ):
        raise SystemExit("J2 result does not bind the supplied vector archive")
    if (
        j2_result.get("outputs", {}).get("explanations_sha256")
        != hashes["j2_explanations"]
    ):
        raise SystemExit("J2 result does not bind the supplied explanations")
    if (
        j2_result.get("outputs", {}).get("checkpoint_sha256")
        != hashes["j2_av_checkpoint"]
    ):
        raise SystemExit("J2 result does not bind the supplied AV checkpoint")

    data = n4.load_activations(args.activations)
    x = data["x"]
    if x.shape != (200, 3840):
        raise SystemExit(f"expected frozen (200,3840) cohort, got {x.shape}")
    result_rows = j2_result.get("rows", [])
    if (
        len({int(row.get("idx", -1)) for row in result_rows}) != 200
        or sorted(int(row.get("idx", -1)) for row in result_rows)
        != list(range(200))
    ):
        raise SystemExit("J2 reconstruction idx coverage is invalid")
    for row in result_rows:
        idx = int(row["idx"])
        if (
            int(row["doc_id"]) != int(data["doc_id"][idx])
            or int(row["position"]) != int(data["position"][idx])
            or row["token"] != data["token"][idx]
        ):
            raise SystemExit(f"J2 reconstruction metadata mismatch at idx {idx}")
    with np.load(args.j2_vectors, allow_pickle=False) as archive:
        required = {
            "doc_ids",
            "positions",
            "pred_sae_small_av_ar",
            "pred_sae_big_av_ar",
            "recon_sae_small_of_direct_nla",
            "recon_sae_big_of_direct_nla",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise SystemExit(f"J2 vector archive missing keys: {missing}")
        if not np.array_equal(
            np.asarray(archive["doc_ids"], dtype=np.int64), data["doc_id"]
        ):
            raise SystemExit("J2 doc_ids do not match the frozen cohort")
        if not np.array_equal(
            np.asarray(archive["positions"], dtype=np.int64), data["position"]
        ):
            raise SystemExit("J2 positions do not match the frozen cohort")
        small_loop = np.asarray(
            archive["pred_sae_small_av_ar"], dtype=np.float32
        )
        big_loop = np.asarray(
            archive["pred_sae_big_av_ar"], dtype=np.float32
        )
        direct_small = np.asarray(
            archive["recon_sae_small_of_direct_nla"], dtype=np.float32
        )
        direct_big = np.asarray(
            archive["recon_sae_big_of_direct_nla"], dtype=np.float32
        )
    for name, vectors in (
        ("small_loop", small_loop),
        ("big_loop", big_loop),
        ("direct_small", direct_small),
        ("direct_big", direct_big),
    ):
        if vectors.shape != x.shape or not np.all(np.isfinite(vectors)):
            raise SystemExit(f"{name} has invalid shape or non-finite values")

    substitutes = {
        "identity": x,
        "small_loop": n4.norm_match(small_loop, x, "small_loop"),
        "big_loop": n4.norm_match(big_loop, x, "big_loop"),
        "direct_small": n4.norm_match(
            direct_small, x, "direct_small"
        ),
        "direct_big": n4.norm_match(direct_big, x, "direct_big"),
        "zero": np.zeros_like(x),
    }
    base_config = args.base_model / "config.json"
    if not base_config.exists():
        raise SystemExit(f"base model config missing: {base_config}")
    contract = {
        "experiment": "J2-P0 SAE projection language-loop causal patch",
        "hashes": hashes,
        "base_config_sha256": sha256_file(base_config),
        "base_weight_manifest": n4.weight_manifest(args.base_model),
        "base_model_files": verify_base_model_files(
            args.model_manifest, args.base_model
        ),
        "layer_index": args.layer_index,
        "horizon": args.horizon,
        "dtype": args.dtype,
        "provenance_atol": args.provenance_atol,
        "identity_kl_tol": args.identity_kl_tol,
        "conditions": list(CONDITIONS),
        "input_contract": "frozen input_ids; batch-size-one; no retokenization",
        "norm_contract": "nonzero substitutes rescaled to target x norm",
    }
    run_contract = n4.contract_digest(contract)
    checkpoint_rows, checkpoint_forwards = n4.load_checkpoint(
        args.checkpoint, run_contract, len(x)
    )
    print(
        f"[J2 causal] contract={run_contract} "
        f"checkpoint={len(checkpoint_rows)}/200 dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        return

    positions = data["position"]
    doc_ids = data["doc_id"]
    input_ids = data["input_ids"]
    for idx, (position, ids) in enumerate(zip(positions, input_ids)):
        if position < 0 or position + args.horizon >= len(ids):
            raise SystemExit(f"row {idx} has truncated causal horizon")

    rows_by_doc: dict[int, list[int]] = defaultdict(list)
    canonical_ids: dict[int, np.ndarray] = {}
    for idx, (doc_id, ids) in enumerate(zip(doc_ids, input_ids)):
        document = int(doc_id)
        rows_by_doc[document].append(idx)
        if document in canonical_ids and not np.array_equal(
            canonical_ids[document], ids
        ):
            raise SystemExit(f"doc {document} has inconsistent frozen input_ids")
        canonical_ids[document] = ids
    for doc_id, indices in rows_by_doc.items():
        present = [idx in checkpoint_rows for idx in indices]
        if any(present) and not all(present):
            raise SystemExit(f"partial checkpoint document {doc_id}: {present}")

    if args.out.exists():
        if len(checkpoint_rows) != 200:
            raise SystemExit(
                "causal output exists but checkpoint is not complete"
            )
        checkpoint_sidecar = args.checkpoint.with_suffix(
            args.checkpoint.suffix + ".sha256"
        )
        if not checkpoint_sidecar.exists():
            raise SystemExit("causal output exists without checkpoint sidecar")
        checkpoint_digest = sha256_file(args.checkpoint)
        if (
            checkpoint_sidecar.read_text(encoding="utf-8").split()[0]
            != checkpoint_digest
        ):
            raise SystemExit("causal checkpoint sidecar mismatch")
        sidecar = args.out.with_suffix(args.out.suffix + ".sha256")
        if not sidecar.exists():
            raise SystemExit("causal output exists without sidecar")
        if sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(args.out):
            raise SystemExit("causal output sidecar mismatch")
        completed = json.loads(args.out.read_text(encoding="utf-8"))
        if (
            completed.get("status") != "EXPLORATORY_CAUSAL_COMPLETE"
            or completed.get("inputs", {}).get("run_contract_sha256")
            != run_contract
            or completed.get("inputs", {}).get("checkpoint_sha256")
            != checkpoint_digest
            or len(completed.get("rows", [])) != 200
            or completed.get("protocol", {}).get("conditions")
            != list(CONDITIONS)
        ):
            raise SystemExit("completed causal output contract/schema mismatch")
        print(f"J2_CAUSAL_ALREADY_COMPLETE sha256={sha256_file(args.out)}")
        return

    if not hasattr(torch, args.dtype):
        raise SystemExit(f"invalid torch dtype: {args.dtype}")
    started = time.time()
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device
    layers = n4.resolve_layers(model)
    layer = layers[args.layer_index]
    model_load_seconds = time.time() - load_started

    output_rows: list[dict[str, Any] | None] = [
        checkpoint_rows.get(idx) for idx in range(len(x))
    ]
    forward_count = checkpoint_forwards
    saved_provenance = [
        row["provenance"] for row in checkpoint_rows.values()
    ]
    provenance_max_abs = max(
        [float(item["max_abs"]) for item in saved_provenance], default=0.0
    )
    provenance_min_cos = min(
        [float(item["cosine"]) for item in saved_provenance], default=1.0
    )
    provenance_all_exact = all(
        bool(item["exact"]) for item in saved_provenance
    )
    forward_started = time.time()

    documents = list(rows_by_doc.items())
    for ordinal, (doc_id, indices) in enumerate(documents, 1):
        if all(output_rows[idx] is not None for idx in indices):
            print(
                f"[doc {ordinal}/{len(documents)}] id={doc_id} checkpoint",
                flush=True,
            )
            continue
        ids_np = canonical_ids[doc_id]
        ids = torch.as_tensor(ids_np[None, :], dtype=torch.long, device=device)
        document_positions = [int(positions[idx]) for idx in indices]
        clean_windows, clean_hidden = n4.clean_forward(
            model, layer, ids, document_positions, args.horizon
        )
        forward_count += 1

        clean_metrics: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        for local_idx, row_idx in enumerate(indices):
            observed = clean_hidden[local_idx]
            frozen = x[row_idx]
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
            if max_abs > args.provenance_atol:
                raise RuntimeError(
                    f"row {row_idx} provenance max_abs={max_abs} "
                    f"> {args.provenance_atol}"
                )
            provenance_max_abs = max(provenance_max_abs, max_abs)
            provenance_min_cos = min(provenance_min_cos, cosine)
            provenance_all_exact = provenance_all_exact and exact
            provenance.append(
                {"exact": exact, "max_abs": max_abs, "cosine": cosine}
            )
            position = int(positions[row_idx])
            targets = ids[0, position + 1 : position + 1 + args.horizon]
            clean_metric = n4.prepare_clean_metrics(
                clean_windows[local_idx], targets
            )
            clean_metric["targets"] = targets
            clean_metrics.append(clean_metric)

        condition_scores: dict[str, list[dict[str, Any]]] = {
            condition: [] for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            for local_idx, row_idx in enumerate(indices):
                patched = n4.patched_forward(
                    model,
                    layer,
                    ids,
                    int(positions[row_idx]),
                    substitutes[condition][row_idx],
                    args.horizon,
                )
                forward_count += 1
                condition_scores[condition].append(
                    n4.score_window(
                        clean_metrics[local_idx],
                        patched,
                        clean_metrics[local_idx]["targets"],
                    )
                )
                del patched

        for local_idx, row_idx in enumerate(indices):
            zero_pos = max(
                float(condition_scores["zero"][local_idx]["kl_at_pos"]), 1e-6
            )
            zero_16 = max(
                float(
                    condition_scores["zero"][local_idx]["kl_mean_first16"]
                ),
                1e-6,
            )
            for condition in CONDITIONS:
                score = condition_scores[condition][local_idx]
                score["kl_recovered_at_pos_vs_zero"] = float(
                    1.0 - float(score["kl_at_pos"]) / zero_pos
                )
                score["kl_recovered_first16_vs_zero"] = float(
                    1.0 - float(score["kl_mean_first16"]) / zero_16
                )
            output_rows[row_idx] = {
                "idx": row_idx,
                "doc_id": int(doc_ids[row_idx]),
                "position": int(positions[row_idx]),
                "token": data["token"][row_idx],
                "token_id": data["token_id"][row_idx],
                "corpus": data["corpus"][row_idx],
                "source": data["source"][row_idx],
                "lang": data["lang"][row_idx],
                "seq_len_evaluated": len(ids_np),
                "x_norm": float(np.linalg.norm(x[row_idx])),
                "provenance": provenance[local_idx],
                "ce_clean_first16": float(clean_metrics[local_idx]["ce"]),
                "results": {
                    condition: condition_scores[condition][local_idx]
                    for condition in CONDITIONS
                },
            }

        n4.append_checkpoint(
            args.checkpoint,
            {
                "contract_sha256": run_contract,
                "doc_id": int(doc_id),
                "n_forwards": 1 + len(CONDITIONS) * len(indices),
                "rows": [output_rows[idx] for idx in indices],
            },
        )
        eta = (
            (time.time() - forward_started)
            / ordinal
            * (len(documents) - ordinal)
        )
        print(
            f"[doc {ordinal}/{len(documents)}] id={doc_id} "
            f"rows={len(indices)} eta={eta / 60:.1f}m",
            flush=True,
        )
        del ids, clean_windows, clean_hidden, clean_metrics, condition_scores

    rows = [row for row in output_rows if row is not None]
    if len(rows) != 200:
        raise RuntimeError(f"causal output incomplete: {len(rows)}/200")
    summary = n4.summarize(rows, list(CONDITIONS))
    identity_max = float(summary["identity"]["kl_at_pos_max"])
    if identity_max > args.identity_kl_tol:
        raise RuntimeError(
            f"identity KL max {identity_max} > {args.identity_kl_tol}"
        )
    checkpoint_digest = sha256_file(args.checkpoint)
    args.checkpoint.with_suffix(args.checkpoint.suffix + ".sha256").write_text(
        f"{checkpoint_digest}  {args.checkpoint.name}\n", encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "experiment": "J2-P0 SAE projection language-loop causal patch",
        "status": "EXPLORATORY_CAUSAL_COMPLETE",
        "confirmatory": False,
        "claim_scope": "discovery_only_no_composite_superiority_claim",
        "protocol": {
            "layer_index": args.layer_index,
            "horizon": args.horizon,
            "kl_direction": "KL(clean || patched)",
            "norm_contract": "nonzero substitutes rescaled to target x norm",
            "conditions": list(CONDITIONS),
        },
        "inputs": {
            **hashes,
            "run_contract_sha256": run_contract,
            "checkpoint_sha256": checkpoint_digest,
        },
        "qa": {
            "rows": len(rows),
            "documents": len(rows_by_doc),
            "provenance_all_exact": provenance_all_exact,
            "provenance_max_abs": provenance_max_abs,
            "provenance_min_cosine": provenance_min_cos,
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
    digest = write_frozen(args.out, payload)
    print(
        f"J2_CAUSAL_COMPLETE sha256={digest} forwards={forward_count} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
