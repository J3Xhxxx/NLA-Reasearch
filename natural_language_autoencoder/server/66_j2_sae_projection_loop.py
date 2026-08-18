#!/usr/bin/env python3
"""J2-P0: real activation -> SAE reconstruction -> AV -> AR.

This is an exploratory serial-composition audit on the frozen N4 200-row
real-content cohort.  It also measures the SAE fixed point
E(x) versus E(D(E(x))).  The script is append-only/resumable for AV text
generation and create-once for frozen JSON/NPZ outputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from pilot_common import AVLocal, JumpReLUSAE, NLACritic


EXPECTED = {
    "activations": "eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66",
    "n4_vectors": "e9d89713dc64381a52f05224d6522abb0ec547777a8c6a7f08b841a72a339967",
    "n4_explanations": "b656ded845c8fd122e4dcb1391ba5d81e1a903f80a69c30575bf26910e200942",
    "model_manifest": "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735",
    "pilot_common": "69fb1b40d60d075c615acdaa23acf4f85c17b5b4cf02e2cc18113c4e14ecf63a",
    "protocol": "a41b7d89893a270218bf79e226c3e3d7a8726f71ca1fe6d41f40b583616a700f",
}
CONDITIONS = ("sae_small", "sae_big")
VECTOR_KEYS = {
    "sae_small": "recon_sae_small",
    "sae_big": "recon_sae_big",
}
TOP_K = 20
FROZEN_RECON_MAX_ABS_TOL = 1e-5
FROZEN_RECON_MAX_REL_L2_TOL = 1e-6
MODEL_ROOTS = {
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


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(map(str, a.shape)).encode("ascii"))
    digest.update(b"|")
    digest.update(a.tobytes(order="C"))
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def write_frozen_json(path: Path, payload: Any) -> str:
    encoded = canonical_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise SystemExit(f"refusing to overwrite non-identical frozen file: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return write_sidecar(path)


def validate_hash(label: str, path: str | Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{label} hash mismatch: {actual} != {expected}")
    return actual


def validate_model_files(
    manifest_path: Path,
    model_roots: dict[str, str],
) -> dict[str, Any]:
    declared: dict[str, str] = {}
    for line_no, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SystemExit(f"invalid model manifest line {line_no}")
        declared[parts[1].strip()] = parts[0].lower()

    result: dict[str, Any] = {}
    for label, root_string in model_roots.items():
        canonical_root = MODEL_ROOTS[label].rstrip("/")
        entries = {
            path: digest
            for path, digest in declared.items()
            if path == canonical_root or path.startswith(canonical_root + "/")
        }
        if not entries:
            raise SystemExit(f"model manifest has no entries for {label}")
        verified: dict[str, str] = {}
        for declared_path, expected_digest in sorted(entries.items()):
            relative = declared_path[len(canonical_root) :].lstrip("/")
            actual_path = Path(root_string) / relative
            if not actual_path.is_file():
                raise SystemExit(f"missing {label} model file: {actual_path}")
            actual_digest = sha256_file(actual_path)
            if actual_digest != expected_digest:
                raise SystemExit(
                    f"{label} model hash mismatch for {actual_path}: "
                    f"{actual_digest} != {expected_digest}"
                )
            verified[relative] = actual_digest
        manifest_digest = hashlib.sha256(
            json.dumps(
                verified,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result[label] = {
            "root": root_string,
            "files": verified,
            "verified_manifest_sha256": manifest_digest,
        }
    return result


def load_activations(path: Path) -> tuple[np.ndarray, dict[str, list[Any]]]:
    table = pq.read_table(path)
    required = (
        "activation_vector",
        "token",
        "token_id",
        "position",
        "doc_id",
        "corpus",
        "source",
        "lang",
        "seq_len",
        "context_tail",
        "continuation",
    )
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise SystemExit(f"activation parquet missing columns: {missing}")
    vectors = np.asarray(
        table["activation_vector"].combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    metadata = {
        name: table[name].to_pylist()
        for name in required
        if name != "activation_vector"
    }
    return vectors, metadata


def contract_digest(inputs: dict[str, Any]) -> str:
    payload = {
        "experiment": "J2-P0 SAE projection language loop",
        "conditions": list(CONDITIONS),
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


def load_checkpoint(
    path: Path,
    expected_contract: str,
    vectors: dict[str, np.ndarray],
    metadata: dict[str, list[Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("contract_sha256") != expected_contract:
            raise SystemExit(f"checkpoint contract mismatch at line {line_no}")
        condition = str(record["condition"])
        idx = int(record["idx"])
        if condition not in vectors or not 0 <= idx < len(vectors[condition]):
            raise SystemExit(f"checkpoint key out of range at line {line_no}")
        key = (condition, idx)
        if key in records:
            raise SystemExit(f"duplicate checkpoint key at line {line_no}: {key}")
        actual_vector_sha = sha256_array(vectors[condition][idx])
        if record.get("vector_sha256") != actual_vector_sha:
            raise SystemExit(f"checkpoint vector hash mismatch at line {line_no}")
        if int(record.get("doc_id", -1)) != int(metadata["doc_id"][idx]):
            raise SystemExit(f"checkpoint doc_id mismatch at line {line_no}")
        if int(record.get("position", -1)) != int(metadata["position"][idx]):
            raise SystemExit(f"checkpoint position mismatch at line {line_no}")
        if record.get("token") != metadata["token"][idx]:
            raise SystemExit(f"checkpoint token mismatch at line {line_no}")
        explanation = record.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise SystemExit(f"empty checkpoint explanation at line {line_no}")
        explanation_sha = hashlib.sha256(
            explanation.encode("utf-8")
        ).hexdigest()
        if record.get("explanation_utf8_sha256") != explanation_sha:
            raise SystemExit(
                f"checkpoint explanation hash mismatch at line {line_no}"
            )
        records[key] = record
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1e-12)


def top_contribution_indices(
    acts: np.ndarray,
    decoder_norms: np.ndarray,
    k: int = TOP_K,
) -> np.ndarray:
    if not np.all(np.isfinite(acts)) or not np.all(
        np.isfinite(decoder_norms)
    ):
        raise SystemExit("non-finite SAE activations or decoder norms")
    contribution = acts * decoder_norms[None, :]
    indices = np.argpartition(-contribution, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(contribution, indices, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1)


def sae_fixed_point(
    sae_path: str,
    x: np.ndarray,
    frozen_reconstruction: np.ndarray,
    direct_reconstruction: np.ndarray,
    loop_reconstruction: np.ndarray,
    condition: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    sae = JumpReLUSAE(sae_path, device="cuda")
    with torch.inference_mode():
        reconstruction_now_t, acts_x_t = sae(torch.from_numpy(x))
        reconstruction_now = reconstruction_now_t.float().cpu().numpy()
        difference = reconstruction_now - frozen_reconstruction
        max_abs_error = float(np.abs(difference).max())
        relative_error = safe_div(
            np.linalg.norm(difference, axis=1),
            np.linalg.norm(frozen_reconstruction, axis=1),
        )
        max_rel_error = float(relative_error.max())
        if (
            max_abs_error > FROZEN_RECON_MAX_ABS_TOL
            or max_rel_error > FROZEN_RECON_MAX_REL_L2_TOL
        ):
            raise SystemExit(
                f"{condition} current SAE reconstruction drift exceeds "
                f"frozen tolerances (max_abs={max_abs_error}, "
                f"max_rel={max_rel_error})"
            )
        reconstruction_2_t, acts_sae_t = sae(
            torch.from_numpy(frozen_reconstruction)
        )
        reconstruction_direct_sae_t, acts_direct_t = sae(
            torch.from_numpy(direct_reconstruction)
        )
        reconstruction_loop_sae_t, acts_loop_t = sae(
            torch.from_numpy(loop_reconstruction)
        )
    reconstruction_2 = reconstruction_2_t.float().cpu().numpy()
    reconstruction_direct_sae = (
        reconstruction_direct_sae_t.float().cpu().numpy()
    )
    reconstruction_loop_sae = reconstruction_loop_sae_t.float().cpu().numpy()
    acts_x = acts_x_t.float().cpu().numpy()
    acts_sae = acts_sae_t.float().cpu().numpy()
    acts_direct = acts_direct_t.float().cpu().numpy()
    acts_loop = acts_loop_t.float().cpu().numpy()
    decoder_norms = sae.w_dec.float().norm(dim=1).cpu().numpy()
    del (
        reconstruction_now_t,
        reconstruction_2_t,
        reconstruction_direct_sae_t,
        reconstruction_loop_sae_t,
        acts_x_t,
        acts_sae_t,
        acts_direct_t,
        acts_loop_t,
        sae,
    )
    gc.collect()
    torch.cuda.empty_cache()
    for label, array in (
        ("reconstruction_2", reconstruction_2),
        ("reconstruction_direct_sae", reconstruction_direct_sae),
        ("reconstruction_loop_sae", reconstruction_loop_sae),
        ("acts_x", acts_x),
        ("acts_sae", acts_sae),
        ("acts_direct", acts_direct),
        ("acts_loop", acts_loop),
        ("decoder_norms", decoder_norms),
    ):
        if not np.all(np.isfinite(array)):
            raise SystemExit(f"{condition} {label} contains non-finite values")

    support_x = acts_x > 0
    support_sae = acts_sae > 0
    support_direct = acts_direct > 0
    support_loop = acts_loop > 0
    intersection = np.logical_and(support_x, support_sae).sum(axis=1)
    union = np.logical_or(support_x, support_sae).sum(axis=1)
    l0_x = support_x.sum(axis=1)
    l0_sae = support_sae.sum(axis=1)
    births = np.logical_and(~support_x, support_sae)
    deaths = np.logical_and(support_x, ~support_sae)
    weighted_cos = safe_div(
        (acts_x * acts_sae).sum(axis=1),
        np.linalg.norm(acts_x, axis=1) * np.linalg.norm(acts_sae, axis=1),
    )
    loop_first_intersection = np.logical_and(
        support_x, support_loop
    ).sum(axis=1)
    loop_first_union = np.logical_or(support_x, support_loop).sum(axis=1)
    loop_second_intersection = np.logical_and(
        support_sae, support_loop
    ).sum(axis=1)
    loop_second_union = np.logical_or(
        support_sae, support_loop
    ).sum(axis=1)
    loop_first_weighted_cos = safe_div(
        (acts_x * acts_loop).sum(axis=1),
        np.linalg.norm(acts_x, axis=1) * np.linalg.norm(acts_loop, axis=1),
    )
    loop_second_weighted_cos = safe_div(
        (acts_sae * acts_loop).sum(axis=1),
        np.linalg.norm(acts_sae, axis=1) * np.linalg.norm(acts_loop, axis=1),
    )
    direct_first_intersection = np.logical_and(
        support_x, support_direct
    ).sum(axis=1)
    direct_first_union = np.logical_or(
        support_x, support_direct
    ).sum(axis=1)
    direct_first_weighted_cos = safe_div(
        (acts_x * acts_direct).sum(axis=1),
        np.linalg.norm(acts_x, axis=1)
        * np.linalg.norm(acts_direct, axis=1),
    )
    loop_direct_intersection = np.logical_and(
        support_loop, support_direct
    ).sum(axis=1)
    loop_direct_union = np.logical_or(
        support_loop, support_direct
    ).sum(axis=1)
    loop_direct_weighted_cos = safe_div(
        (acts_loop * acts_direct).sum(axis=1),
        np.linalg.norm(acts_loop, axis=1)
        * np.linalg.norm(acts_direct, axis=1),
    )
    mass_x = acts_x.sum(axis=1)
    mass_sae = acts_sae.sum(axis=1)
    birth_mass = (acts_sae * births).sum(axis=1)
    top_x = top_contribution_indices(acts_x, decoder_norms)
    top_sae = top_contribution_indices(acts_sae, decoder_norms)
    top_direct = top_contribution_indices(acts_direct, decoder_norms)
    top_loop = top_contribution_indices(acts_loop, decoder_norms)
    l0_direct = support_direct.sum(axis=1)
    l0_loop = support_loop.sum(axis=1)
    direct_births = np.logical_and(~support_x, support_direct)
    direct_deaths = np.logical_and(support_x, ~support_direct)
    direct_mass = acts_direct.sum(axis=1)
    direct_birth_mass = (acts_direct * direct_births).sum(axis=1)
    loop_births = np.logical_and(~support_x, support_loop)
    loop_deaths = np.logical_and(support_x, ~support_loop)
    loop_mass = acts_loop.sum(axis=1)
    loop_birth_mass = (acts_loop * loop_births).sum(axis=1)

    rows: list[dict[str, Any]] = []
    second_to_first_cosine = safe_div(
        (reconstruction_2 * frozen_reconstruction).sum(axis=1),
        np.linalg.norm(reconstruction_2, axis=1)
        * np.linalg.norm(frozen_reconstruction, axis=1),
    )
    second_to_x_cosine = safe_div(
        (reconstruction_2 * x).sum(axis=1),
        np.linalg.norm(reconstruction_2, axis=1)
        * np.linalg.norm(x, axis=1),
    )
    for idx in range(len(x)):
        top_overlap = len(set(top_x[idx].tolist()) & set(top_sae[idx].tolist()))
        loop_top_first_overlap = len(
            set(top_x[idx].tolist()) & set(top_loop[idx].tolist())
        )
        loop_top_second_overlap = len(
            set(top_sae[idx].tolist()) & set(top_loop[idx].tolist())
        )
        direct_top_first_overlap = len(
            set(top_x[idx].tolist()) & set(top_direct[idx].tolist())
        )
        loop_top_direct_overlap = len(
            set(top_loop[idx].tolist()) & set(top_direct[idx].tolist())
        )
        rows.append(
            {
                "idx": idx,
                "support_jaccard": float(safe_div(intersection[idx], union[idx])),
                "support_precision_second_vs_first": float(
                    safe_div(intersection[idx], l0_sae[idx])
                ),
                "support_recall_second_vs_first": float(
                    safe_div(intersection[idx], l0_x[idx])
                ),
                "weighted_code_cosine": float(weighted_cos[idx]),
                "l0_first": int(l0_x[idx]),
                "l0_second": int(l0_sae[idx]),
                "l0_change": int(l0_sae[idx] - l0_x[idx]),
                "births": int(births[idx].sum()),
                "deaths": int(deaths[idx].sum()),
                "activation_mass_ratio": float(safe_div(mass_sae[idx], mass_x[idx])),
                "birth_mass_ratio_second": float(
                    safe_div(birth_mass[idx], mass_sae[idx])
                ),
                "top20_overlap": int(top_overlap),
                "top20_first": [int(v) for v in top_x[idx]],
                "top20_second": [int(v) for v in top_sae[idx]],
                "reconstruction_2_raw_cosine_to_first": float(
                    second_to_first_cosine[idx]
                ),
                "reconstruction_2_raw_cosine_to_x": float(
                    second_to_x_cosine[idx]
                ),
                "loop_support_jaccard_vs_first": float(
                    safe_div(
                        loop_first_intersection[idx],
                        loop_first_union[idx],
                    )
                ),
                "loop_support_jaccard_vs_second": float(
                    safe_div(
                        loop_second_intersection[idx],
                        loop_second_union[idx],
                    )
                ),
                "loop_support_precision_vs_first": float(
                    safe_div(loop_first_intersection[idx], l0_loop[idx])
                ),
                "loop_support_recall_vs_first": float(
                    safe_div(loop_first_intersection[idx], l0_x[idx])
                ),
                "loop_weighted_code_cosine_vs_first": float(
                    loop_first_weighted_cos[idx]
                ),
                "loop_weighted_code_cosine_vs_second": float(
                    loop_second_weighted_cos[idx]
                ),
                "l0_loop": int(l0_loop[idx]),
                "loop_births_vs_first": int(loop_births[idx].sum()),
                "loop_deaths_vs_first": int(loop_deaths[idx].sum()),
                "loop_activation_mass_ratio_vs_first": float(
                    safe_div(loop_mass[idx], mass_x[idx])
                ),
                "loop_birth_mass_ratio": float(
                    safe_div(loop_birth_mass[idx], loop_mass[idx])
                ),
                "loop_top20_overlap_vs_first": int(loop_top_first_overlap),
                "loop_top20_overlap_vs_second": int(loop_top_second_overlap),
                "top20_loop": [int(v) for v in top_loop[idx]],
                "direct_support_jaccard_vs_first": float(
                    safe_div(
                        direct_first_intersection[idx],
                        direct_first_union[idx],
                    )
                ),
                "direct_support_precision_vs_first": float(
                    safe_div(direct_first_intersection[idx], l0_direct[idx])
                ),
                "direct_support_recall_vs_first": float(
                    safe_div(direct_first_intersection[idx], l0_x[idx])
                ),
                "direct_weighted_code_cosine_vs_first": float(
                    direct_first_weighted_cos[idx]
                ),
                "direct_l0": int(l0_direct[idx]),
                "direct_births_vs_first": int(direct_births[idx].sum()),
                "direct_deaths_vs_first": int(direct_deaths[idx].sum()),
                "direct_activation_mass_ratio_vs_first": float(
                    safe_div(direct_mass[idx], mass_x[idx])
                ),
                "direct_birth_mass_ratio": float(
                    safe_div(direct_birth_mass[idx], direct_mass[idx])
                ),
                "direct_top20_overlap_vs_first": int(
                    direct_top_first_overlap
                ),
                "top20_direct": [int(v) for v in top_direct[idx]],
                "loop_support_jaccard_vs_direct": float(
                    safe_div(
                        loop_direct_intersection[idx],
                        loop_direct_union[idx],
                    )
                ),
                "loop_weighted_code_cosine_vs_direct": float(
                    loop_direct_weighted_cos[idx]
                ),
                "loop_top20_overlap_vs_direct": int(
                    loop_top_direct_overlap
                ),
            }
        )

    summary = {
        "condition": condition,
        "n": len(rows),
        "frozen_reconstruction_recheck": {
            "max_absolute_error": max_abs_error,
            "max_relative_l2_error": float(relative_error.max()),
            "max_absolute_tolerance": FROZEN_RECON_MAX_ABS_TOL,
            "max_relative_l2_tolerance": FROZEN_RECON_MAX_REL_L2_TOL,
            "second_encode_input": "frozen_N4_reconstruction",
        },
        "mean_support_jaccard": float(
            np.mean([row["support_jaccard"] for row in rows])
        ),
        "mean_weighted_code_cosine": float(weighted_cos.mean()),
        "mean_l0_first": float(l0_x.mean()),
        "mean_l0_second": float(l0_sae.mean()),
        "mean_births": float(births.sum(axis=1).mean()),
        "mean_deaths": float(deaths.sum(axis=1).mean()),
        "mean_birth_mass_ratio_second": float(
            safe_div(birth_mass, mass_sae).mean()
        ),
        "mean_top20_overlap": float(
            np.mean([row["top20_overlap"] for row in rows])
        ),
        "mean_reconstruction_2_raw_cosine_to_first": float(
            second_to_first_cosine.mean()
        ),
        "mean_reconstruction_2_raw_cosine_to_x": float(
            second_to_x_cosine.mean()
        ),
        "mean_loop_support_jaccard_vs_first": float(
            safe_div(loop_first_intersection, loop_first_union).mean()
        ),
        "mean_loop_support_jaccard_vs_second": float(
            safe_div(loop_second_intersection, loop_second_union).mean()
        ),
        "mean_loop_weighted_code_cosine_vs_first": float(
            loop_first_weighted_cos.mean()
        ),
        "mean_loop_weighted_code_cosine_vs_second": float(
            loop_second_weighted_cos.mean()
        ),
        "mean_l0_loop": float(l0_loop.mean()),
        "mean_loop_births_vs_first": float(loop_births.sum(axis=1).mean()),
        "mean_loop_deaths_vs_first": float(loop_deaths.sum(axis=1).mean()),
        "mean_loop_birth_mass_ratio": float(
            safe_div(loop_birth_mass, loop_mass).mean()
        ),
        "mean_loop_top20_overlap_vs_first": float(
            np.mean(
                [row["loop_top20_overlap_vs_first"] for row in rows]
            )
        ),
        "mean_loop_top20_overlap_vs_second": float(
            np.mean(
                [row["loop_top20_overlap_vs_second"] for row in rows]
            )
        ),
        "mean_direct_support_jaccard_vs_first": float(
            safe_div(direct_first_intersection, direct_first_union).mean()
        ),
        "mean_direct_weighted_code_cosine_vs_first": float(
            direct_first_weighted_cos.mean()
        ),
        "mean_direct_l0": float(l0_direct.mean()),
        "mean_direct_births_vs_first": float(
            direct_births.sum(axis=1).mean()
        ),
        "mean_direct_deaths_vs_first": float(
            direct_deaths.sum(axis=1).mean()
        ),
        "mean_direct_birth_mass_ratio": float(
            safe_div(direct_birth_mass, direct_mass).mean()
        ),
        "mean_direct_top20_overlap_vs_first": float(
            np.mean(
                [row["direct_top20_overlap_vs_first"] for row in rows]
            )
        ),
        "mean_loop_support_jaccard_vs_direct": float(
            safe_div(loop_direct_intersection, loop_direct_union).mean()
        ),
        "mean_loop_weighted_code_cosine_vs_direct": float(
            loop_direct_weighted_cos.mean()
        ),
        "mean_loop_top20_overlap_vs_direct": float(
            np.mean(
                [row["loop_top20_overlap_vs_direct"] for row in rows]
            )
        ),
    }
    return (
        reconstruction_2.astype(np.float32),
        reconstruction_direct_sae.astype(np.float32),
        reconstruction_loop_sae.astype(np.float32),
        rows,
        summary,
    )


def save_npz_frozen(path: Path, arrays: dict[str, np.ndarray]) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise SystemExit(
                    f"existing vector archive key mismatch: {archive.files}"
                )
            for key, expected in arrays.items():
                observed = archive[key]
                if (
                    observed.shape != expected.shape
                    or observed.dtype != expected.dtype
                    or not np.array_equal(observed, expected)
                ):
                    raise SystemExit(
                        f"existing vector archive differs at key {key}"
                    )
        return write_sidecar(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise SystemExit(f"stale temporary archive requires audit: {temporary}")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return write_sidecar(path)


def completed_output_is_valid(
    result_path: Path,
    vectors_path: Path,
    explanations_path: Path,
    checkpoint_path: Path,
    expected_contract: str,
) -> bool:
    if not result_path.exists():
        return False
    required_paths = (
        result_path,
        vectors_path,
        explanations_path,
        checkpoint_path,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"completed result has missing artifacts: {missing}")
    for path in required_paths:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists():
            raise SystemExit(f"missing sidecar for completed output: {path}")
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"completed output sidecar mismatch: {path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "EXPLORATORY_RECON_COMPLETE":
        raise SystemExit("completed J2 result has wrong status")
    if result.get("contract_sha256") != expected_contract:
        raise SystemExit("completed J2 result contract mismatch")
    if result.get("cohort", {}).get("n") != 200 or len(
        result.get("rows", [])
    ) != 200:
        raise SystemExit("completed J2 result has wrong row count")
    if result.get("outputs", {}).get("vectors_sha256") != sha256_file(
        vectors_path
    ):
        raise SystemExit("completed J2 result/vector binding mismatch")
    if result.get("outputs", {}).get("explanations_sha256") != sha256_file(
        explanations_path
    ):
        raise SystemExit("completed J2 result/explanation binding mismatch")
    if result.get("outputs", {}).get("checkpoint_sha256") != sha256_file(
        checkpoint_path
    ):
        raise SystemExit("completed J2 result/checkpoint binding mismatch")

    explanations = json.loads(explanations_path.read_text(encoding="utf-8"))
    if explanations.get("status") != "FROZEN_BEFORE_AR_OR_FIXED_POINT":
        raise SystemExit("completed explanation artifact has wrong status")
    if explanations.get("contract_sha256") != expected_contract:
        raise SystemExit("completed explanation contract mismatch")
    if len(explanations.get("rows", [])) != 200:
        raise SystemExit("completed explanation row count mismatch")

    checkpoint_keys: set[tuple[str, int]] = set()
    for line_no, line in enumerate(
        checkpoint_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("contract_sha256") != expected_contract:
            raise SystemExit(
                f"completed checkpoint contract mismatch at line {line_no}"
            )
        key = (str(record.get("condition")), int(record.get("idx", -1)))
        if key in checkpoint_keys:
            raise SystemExit(f"duplicate completed checkpoint key: {key}")
        checkpoint_keys.add(key)
    expected_keys = {
        (condition, idx)
        for condition in CONDITIONS
        for idx in range(200)
    }
    if checkpoint_keys != expected_keys:
        raise SystemExit("completed checkpoint does not cover 400 frozen pairs")

    with np.load(vectors_path, allow_pickle=False) as archive:
        expected_shapes = {
            "doc_ids": (200,),
            "positions": (200,),
            "pred_sae_small_av_ar": (200, 3840),
            "pred_sae_big_av_ar": (200, 3840),
            "recon_sae_small_2": (200, 3840),
            "recon_sae_big_2": (200, 3840),
            "recon_sae_small_of_direct_nla": (200, 3840),
            "recon_sae_big_of_direct_nla": (200, 3840),
            "recon_sae_small_of_loop": (200, 3840),
            "recon_sae_big_of_loop": (200, 3840),
        }
        for key, shape in expected_shapes.items():
            if key not in archive.files or archive[key].shape != shape:
                raise SystemExit(
                    f"completed vector archive key/shape mismatch: {key}"
                )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av", required=True)
    parser.add_argument("--ar", required=True)
    parser.add_argument("--sae-small", required=True)
    parser.add_argument("--sae-big", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--n4-vectors", required=True)
    parser.add_argument("--n4-explanations", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--pilot-common", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--explanations-out", required=True)
    parser.add_argument("--vectors-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens != 200:
        raise SystemExit("frozen protocol requires max_new_tokens=200")

    paths = {
        "activations": Path(args.activations),
        "n4_vectors": Path(args.n4_vectors),
        "n4_explanations": Path(args.n4_explanations),
        "model_manifest": Path(args.model_manifest),
        "pilot_common": Path(args.pilot_common),
        "protocol": Path(args.protocol),
    }
    bound_hashes = {
        label: validate_hash(label, path, EXPECTED[label])
        for label, path in paths.items()
    }
    model_arguments = {
        "av": str(Path(args.av).resolve()),
        "ar": str(Path(args.ar).resolve()),
        "sae_small": str(Path(args.sae_small).resolve()),
        "sae_big": str(Path(args.sae_big).resolve()),
    }
    for label, expected_root in MODEL_ROOTS.items():
        if model_arguments[label] != str(Path(expected_root)):
            raise SystemExit(
                f"{label} model root mismatch: "
                f"{model_arguments[label]} != {expected_root}"
            )
    bound_hashes["model_roots"] = model_arguments
    bound_hashes["model_files"] = validate_model_files(
        paths["model_manifest"], model_arguments
    )
    bound_hashes["script"] = sha256_file(__file__)
    contract_sha = contract_digest(bound_hashes)

    result_path = Path(args.out)
    vectors_out_path = Path(args.vectors_out)
    explanations_out_path = Path(args.explanations_out)
    checkpoint_path = Path(args.checkpoint)
    if completed_output_is_valid(
        result_path,
        vectors_out_path,
        explanations_out_path,
        checkpoint_path,
        contract_sha,
    ):
        print(
            f"J2_P0_ALREADY_COMPLETE result={sha256_file(result_path)} "
            f"vectors={sha256_file(vectors_out_path)}"
        )
        return

    x, metadata = load_activations(paths["activations"])
    if x.shape != (200, 3840):
        raise SystemExit(f"expected activation shape (200,3840), got {x.shape}")
    if len(set(zip(metadata["doc_id"], metadata["position"]))) != 200:
        raise SystemExit("duplicate (doc_id, position) in frozen cohort")

    with np.load(paths["n4_vectors"]) as archive:
        required_keys = {
            "x",
            "doc_ids",
            "positions",
            "recon_sae_small",
            "recon_sae_big",
            "pred_orig",
        }
        missing = sorted(required_keys - set(archive.files))
        if missing:
            raise SystemExit(f"N4 vector archive missing keys: {missing}")
        n4_x = archive["x"].astype(np.float32)
        direct_reconstruction = archive["pred_orig"].astype(np.float32)
        condition_vectors = {
            condition: archive[VECTOR_KEYS[condition]].astype(np.float32)
            for condition in CONDITIONS
        }
        archive_doc_ids = np.asarray(archive["doc_ids"], dtype=np.int64)
        archive_positions = np.asarray(archive["positions"], dtype=np.int64)
    if not np.array_equal(x, n4_x):
        max_abs = float(np.abs(x - n4_x).max())
        raise SystemExit(f"parquet x differs from N4 vectors (max_abs={max_abs})")
    if not np.array_equal(
        archive_doc_ids, np.asarray(metadata["doc_id"], dtype=np.int64)
    ):
        raise SystemExit("N4 vector doc_ids differ from activation parquet")
    if not np.array_equal(
        archive_positions, np.asarray(metadata["position"], dtype=np.int64)
    ):
        raise SystemExit("N4 vector positions differ from activation parquet")
    for label, array in (
        ("n4_x", n4_x),
        ("direct_reconstruction", direct_reconstruction),
        *condition_vectors.items(),
    ):
        if array.shape != (200, 3840) or not np.all(np.isfinite(array)):
            raise SystemExit(f"{label} has invalid shape or non-finite values")

    old_explanations = json.loads(
        paths["n4_explanations"].read_text(encoding="utf-8")
    )
    if old_explanations.get("status") != "complete_frozen_before_AR":
        raise SystemExit("N4 explanation artifact has unexpected status")
    if (
        old_explanations.get("activation_sha256")
        != EXPECTED["activations"]
    ):
        raise SystemExit("N4 explanation activation binding mismatch")
    if len(old_explanations.get("rows", [])) != 200:
        raise SystemExit("N4 explanation row count mismatch")
    if len({int(row["idx"]) for row in old_explanations["rows"]}) != 200:
        raise SystemExit("duplicate N4 explanation idx")
    direct_by_idx = {
        int(row["idx"]): row["explanation"]
        for row in old_explanations["rows"]
    }
    if sorted(direct_by_idx) != list(range(200)):
        raise SystemExit("N4 direct explanations do not cover idx 0..199 exactly")
    for row in old_explanations["rows"]:
        idx = int(row["idx"])
        if (
            int(row["doc_id"]) != int(metadata["doc_id"][idx])
            or int(row["position"]) != int(metadata["position"][idx])
            or row["token"] != metadata["token"][idx]
        ):
            raise SystemExit(f"N4 explanation metadata mismatch at idx {idx}")

    done = load_checkpoint(
        checkpoint_path, contract_sha, condition_vectors, metadata
    )
    missing_pairs = [
        (condition, idx)
        for condition in CONDITIONS
        for idx in range(200)
        if (condition, idx) not in done
    ]
    print(
        f"[J2-P0] contract={contract_sha} complete={len(done)}/400 "
        f"missing={len(missing_pairs)} dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        return

    started = time.time()
    if missing_pairs:
        av = AVLocal(args.av, device="cuda")
        for ordinal, (condition, idx) in enumerate(missing_pairs, 1):
            vector = condition_vectors[condition][idx]
            explanation = av.generate(
                vector,
                temperature=0.0,
                max_new_tokens=args.max_new_tokens,
            )
            record = {
                "contract_sha256": contract_sha,
                "condition": condition,
                "idx": idx,
                "doc_id": int(metadata["doc_id"][idx]),
                "position": int(metadata["position"][idx]),
                "token": metadata["token"][idx],
                "vector_sha256": sha256_array(vector),
                "explanation": explanation,
                "explanation_utf8_sha256": hashlib.sha256(
                    explanation.encode("utf-8")
                ).hexdigest(),
            }
            append_checkpoint(checkpoint_path, record)
            done[(condition, idx)] = record
            print(
                f"[AV {len(done):>3}/400] {condition} idx={idx} "
                f"chars={len(explanation)} {explanation[:60]!r}",
                flush=True,
            )
        del av
        gc.collect()
        torch.cuda.empty_cache()
    if len(done) != 400:
        raise SystemExit(f"AV checkpoint incomplete after generation: {len(done)}/400")
    checkpoint_sha = write_sidecar(checkpoint_path)

    explanation_rows = []
    for idx in range(200):
        explanation_rows.append(
            {
                "idx": idx,
                "doc_id": int(metadata["doc_id"][idx]),
                "position": int(metadata["position"][idx]),
                "token": metadata["token"][idx],
                "corpus": metadata["corpus"][idx],
                "source": metadata["source"][idx],
                "lang": metadata["lang"][idx],
                "context_tail": metadata["context_tail"][idx],
                "continuation": metadata["continuation"][idx],
                "direct_n4": direct_by_idx[idx],
                "sae_small": done[("sae_small", idx)]["explanation"],
                "sae_big": done[("sae_big", idx)]["explanation"],
            }
        )
    explanation_payload = {
        "schema_version": 1,
        "experiment": "J2-P0 SAE projection language loop",
        "status": "FROZEN_BEFORE_AR_OR_FIXED_POINT",
        "confirmatory": False,
        "contract_sha256": contract_sha,
        "checkpoint_sha256": checkpoint_sha,
        "inputs": bound_hashes,
        "generation": {
            "conditions": list(CONDITIONS),
            "temperature": 0.0,
            "max_new_tokens": 200,
            "n_per_condition": 200,
        },
        "rows": explanation_rows,
    }
    explanations_sha = write_frozen_json(
        explanations_out_path, explanation_payload
    )
    print(f"[freeze] explanations={explanations_sha}", flush=True)

    critic = NLACritic(args.ar, device="cuda")
    loop_reconstructions: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        reconstructed = []
        for idx in range(200):
            reconstructed.append(
                critic.reconstruct(done[(condition, idx)]["explanation"])
                .numpy()
                .astype(np.float32)
            )
            if (idx + 1) % 25 == 0:
                print(f"[AR] {condition} {idx + 1}/200", flush=True)
        loop_reconstructions[condition] = np.stack(reconstructed)
    del critic
    gc.collect()
    torch.cuda.empty_cache()

    fixed_point_rows: dict[str, list[dict[str, Any]]] = {}
    fixed_point_summary: dict[str, dict[str, Any]] = {}
    second_reconstructions: dict[str, np.ndarray] = {}
    direct_sae_reconstructions: dict[str, np.ndarray] = {}
    loop_sae_reconstructions: dict[str, np.ndarray] = {}
    for condition, sae_path in (
        ("sae_small", args.sae_small),
        ("sae_big", args.sae_big),
    ):
        (
            reconstruction_2,
            reconstruction_direct_sae,
            reconstruction_loop_sae,
            rows,
            summary,
        ) = sae_fixed_point(
            sae_path,
            x,
            condition_vectors[condition],
            direct_reconstruction,
            loop_reconstructions[condition],
            condition,
        )
        second_reconstructions[condition] = reconstruction_2
        direct_sae_reconstructions[condition] = reconstruction_direct_sae
        loop_sae_reconstructions[condition] = reconstruction_loop_sae
        fixed_point_rows[condition] = rows
        fixed_point_summary[condition] = summary
        print(
            f"[SAE fixed point] {condition} "
            f"J={summary['mean_support_jaccard']:.4f} "
            f"code_cos={summary['mean_weighted_code_cosine']:.4f}",
            flush=True,
        )

    arrays = {
        "doc_ids": np.asarray(metadata["doc_id"], dtype=np.int64),
        "positions": np.asarray(metadata["position"], dtype=np.int64),
        "pred_sae_small_av_ar": loop_reconstructions["sae_small"],
        "pred_sae_big_av_ar": loop_reconstructions["sae_big"],
        "recon_sae_small_2": second_reconstructions["sae_small"],
        "recon_sae_big_2": second_reconstructions["sae_big"],
        "recon_sae_small_of_direct_nla": direct_sae_reconstructions[
            "sae_small"
        ],
        "recon_sae_big_of_direct_nla": direct_sae_reconstructions[
            "sae_big"
        ],
        "recon_sae_small_of_loop": loop_sae_reconstructions["sae_small"],
        "recon_sae_big_of_loop": loop_sae_reconstructions["sae_big"],
    }
    vectors_sha = save_npz_frozen(vectors_out_path, arrays)

    result_rows = []
    for idx in range(200):
        result_rows.append(
            {
                "idx": idx,
                "doc_id": int(metadata["doc_id"][idx]),
                "position": int(metadata["position"][idx]),
                "token": metadata["token"][idx],
                "corpus": metadata["corpus"][idx],
                "source": metadata["source"][idx],
                "lang": metadata["lang"][idx],
                "vector_sha256": {
                    "x": sha256_array(x[idx]),
                    "nla_direct": sha256_array(direct_reconstruction[idx]),
                    "sae_small": sha256_array(condition_vectors["sae_small"][idx]),
                    "sae_big": sha256_array(condition_vectors["sae_big"][idx]),
                    "small_loop": sha256_array(
                        loop_reconstructions["sae_small"][idx]
                    ),
                    "big_loop": sha256_array(
                        loop_reconstructions["sae_big"][idx]
                    ),
                    "sae_small_2": sha256_array(
                        second_reconstructions["sae_small"][idx]
                    ),
                    "sae_big_2": sha256_array(
                        second_reconstructions["sae_big"][idx]
                    ),
                    "sae_small_of_direct_nla": sha256_array(
                        direct_sae_reconstructions["sae_small"][idx]
                    ),
                    "sae_big_of_direct_nla": sha256_array(
                        direct_sae_reconstructions["sae_big"][idx]
                    ),
                    "sae_small_of_loop": sha256_array(
                        loop_sae_reconstructions["sae_small"][idx]
                    ),
                    "sae_big_of_loop": sha256_array(
                        loop_sae_reconstructions["sae_big"][idx]
                    ),
                },
                "fixed_point": {
                    "sae_small": fixed_point_rows["sae_small"][idx],
                    "sae_big": fixed_point_rows["sae_big"][idx],
                },
            }
        )

    result_payload = {
        "schema_version": 1,
        "experiment": "J2-P0 SAE projection language loop",
        "status": "EXPLORATORY_RECON_COMPLETE",
        "confirmatory": False,
        "claim_scope": (
            "discovery_only_no_SAE_grounded_NLA_or_composite_superiority_claim"
        ),
        "contract_sha256": contract_sha,
        "inputs": bound_hashes,
        "outputs": {
            "checkpoint_sha256": checkpoint_sha,
            "explanations_sha256": explanations_sha,
            "vectors_sha256": vectors_sha,
        },
        "cohort": {
            "n": 200,
            "documents": len(set(metadata["doc_id"])),
            "conditions": list(CONDITIONS),
        },
        "fixed_point_summary": fixed_point_summary,
        "rows": result_rows,
        "elapsed_seconds": time.time() - started,
    }
    result_sha = write_frozen_json(result_path, result_payload)
    print(
        f"J2_P0_RECON_COMPLETE result={result_sha} vectors={vectors_sha} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
