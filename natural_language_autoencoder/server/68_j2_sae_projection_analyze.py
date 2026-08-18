#!/usr/bin/env python3
"""CPU analysis for J2-P0 SAE projection -> AV -> AR.

Produces full aggregate/paired statistics and a metric-only frozen case
shortlist.  It deliberately does not copy explanation text into the shortlist;
human-readable cases are rendered only after this ranking artifact is frozen.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np


EXPECTED = {
    "activations": "eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66",
    "n4_vectors": "e9d89713dc64381a52f05224d6522abb0ec547777a8c6a7f08b841a72a339967",
    "n4_explanations": "b656ded845c8fd122e4dcb1391ba5d81e1a903f80a69c30575bf26910e200942",
    "n4_causal": "8dd532f65d8c9c153f04ba433cc6f160798598fbbcbee388c15fb4a75a366233",
    "n4_analysis": "3c8a4d87d7289ac6c41b58e2bbdd6955585db46eaaa5306822d9d802259943cc",
    "model_manifest": "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735",
    "pilot_common": "69fb1b40d60d075c615acdaa23acf4f85c17b5b4cf02e2cc18113c4e14ecf63a",
    "protocol": "a41b7d89893a270218bf79e226c3e3d7a8726f71ca1fe6d41f40b583616a700f",
}
CONDITIONS = ("sae_small", "sae_big")
LOOP_KEYS = {
    "sae_small": "pred_sae_small_av_ar",
    "sae_big": "pred_sae_big_av_ar",
}
SECOND_KEYS = {
    "sae_small": "recon_sae_small_2",
    "sae_big": "recon_sae_big_2",
}
LOOP_SAE_KEYS = {
    "sae_small": "recon_sae_small_of_loop",
    "sae_big": "recon_sae_big_of_loop",
}
DIRECT_SAE_KEYS = {
    "sae_small": "recon_sae_small_of_direct_nla",
    "sae_big": "recon_sae_big_of_direct_nla",
}
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
QUOTE_RE = re.compile(r'"([^"]*)"')
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260806
CONTROL_RECHECK_MAX_ABS_TOL = 1e-6


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


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def perpendicular(array: np.ndarray, mean_hat: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    return values - np.outer(values @ mean_hat, mean_hat)


def row_cosine(
    left: np.ndarray,
    right: np.ndarray,
    mean_hat: np.ndarray | None = None,
) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if mean_hat is not None:
        a = perpendicular(a, mean_hat)
        b = perpendicular(b, mean_hat)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.maximum(denominator, 1e-12)


def lodo_cosine(
    left: np.ndarray,
    right: np.ndarray,
    doc_ids: np.ndarray,
) -> np.ndarray:
    result = np.empty(len(right), dtype=np.float64)
    total = right.sum(axis=0, dtype=np.float64)
    for document in np.unique(doc_ids):
        test = doc_ids == document
        train_n = int((~test).sum())
        mean_hat = unit(
            (total - right[test].sum(axis=0, dtype=np.float64)) / train_n
        )
        result[test] = row_cosine(left[test], right[test], mean_hat)
    return result


def optimal_centered_fve(
    prediction: np.ndarray,
    target: np.ndarray,
    mean_hat: np.ndarray,
) -> dict[str, float]:
    pred = perpendicular(prediction, mean_hat)
    truth = perpendicular(target, mean_hat)
    denominator = float((pred * pred).sum())
    scale = float((pred * truth).sum() / max(denominator, 1e-12))
    residual = truth - scale * pred
    denominator_truth = max(float((truth * truth).sum()), 1e-12)
    fve = 1.0 - float((residual * residual).sum()) / denominator_truth
    fixed_scale_fve = 1.0 - float(((truth - pred) ** 2).sum()) / denominator_truth
    return {
        "optimal_scale": scale,
        "centered_fve": fve,
        "fixed_scale_centered_fve": fixed_scale_fve,
    }


def vector_metric_bundle(
    prediction: np.ndarray,
    target: np.ndarray,
    mean_hat: np.ndarray,
    doc_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    centered = row_cosine(prediction, target, mean_hat)
    target_mean_hat = unit(target.mean(axis=0, dtype=np.float64))
    target_mean_centered = row_cosine(
        prediction, target, target_mean_hat
    )
    raw = row_cosine(prediction, target)
    lodo = lodo_cosine(prediction, target, doc_ids)
    norm_ratio = np.linalg.norm(prediction, axis=1) / np.maximum(
        np.linalg.norm(target, axis=1), 1e-12
    )
    aggregate = {
        "n": len(target),
        "centered_cosine_mean": float(centered.mean()),
        "centered_cosine_median": float(np.median(centered)),
        "centered_cosine_q05": float(np.quantile(centered, 0.05)),
        "target_mean_centered_cosine_mean": float(
            target_mean_centered.mean()
        ),
        "lodo_centered_cosine_mean": float(lodo.mean()),
        "raw_cosine_mean": float(raw.mean()),
        "norm_ratio_mean": float(norm_ratio.mean()),
        **optimal_centered_fve(prediction, target, mean_hat),
    }
    rows = {
        "centered_cosine": centered,
        "raw_cosine": raw,
        "lodo_centered_cosine": lodo,
        "norm_ratio": norm_ratio,
    }
    return aggregate, rows


def normalized_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def text_diagnostics(left: str, right: str) -> dict[str, float | int]:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    left_set, right_set = set(left_tokens), set(right_tokens)
    union = left_set | right_set
    token_jaccard = len(left_set & right_set) / max(len(union), 1)
    sequence_ratio = difflib.SequenceMatcher(
        a=left_tokens, b=right_tokens, autojunk=False
    ).ratio()
    left_quotes = normalized_tokens(" ".join(QUOTE_RE.findall(left)))
    right_quotes = normalized_tokens(" ".join(QUOTE_RE.findall(right)))
    quote_union = set(left_quotes) | set(right_quotes)
    quote_overlap = len(set(left_quotes) & set(right_quotes)) / max(
        len(quote_union), 1
    )
    return {
        "token_jaccard": float(token_jaccard),
        "sequence_ratio": float(sequence_ratio),
        "left_words": len(left_tokens),
        "right_words": len(right_tokens),
        "left_chars": len(left),
        "right_chars": len(right),
        "left_quote_spans": len(QUOTE_RE.findall(left)),
        "right_quote_spans": len(QUOTE_RE.findall(right)),
        "quoted_token_jaccard": float(quote_overlap),
    }


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def document_cluster_bootstrap(
    values: np.ndarray,
    doc_ids: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, Any]:
    documents = np.unique(doc_ids)
    by_document = {
        int(document): np.flatnonzero(doc_ids == document)
        for document in documents
    }
    rng = random.Random(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for rep in range(BOOTSTRAP_REPS):
        sampled = [
            int(documents[rng.randrange(len(documents))])
            for _ in range(len(documents))
        ]
        indices = np.concatenate([by_document[document] for document in sampled])
        draws[rep] = float(statistic(values[indices]))
    return {
        "cluster": "document",
        "documents": int(len(documents)),
        "reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "point": float(statistic(values)),
        "ci95": {
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)),
        },
        "probability_gt_0": float(np.mean(draws > 0)),
        "probability_lt_0": float(np.mean(draws < 0)),
    }


def causal_arrays(
    rows: list[dict[str, Any]], condition: str
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(
            [row["results"][condition][key] for row in rows], dtype=np.float64
        )
        for key in ("kl_at_pos", "kl_mean_first16", "ce_first16")
    }


def causal_summary(
    values: dict[str, np.ndarray],
    zero: dict[str, np.ndarray],
    clean_ce: np.ndarray,
) -> dict[str, Any]:
    result = {key: summarize_values(array) for key, array in values.items()}
    zero_pos = float(zero["kl_at_pos"].sum())
    zero_16 = float(zero["kl_mean_first16"].sum())
    if zero_pos <= 1e-12 or zero_16 <= 1e-12:
        raise SystemExit("zero-control denominator is non-positive")
    zero_ce_excess = float((zero["ce_first16"] - clean_ce).sum())
    if zero_ce_excess <= 1e-12:
        raise SystemExit("zero-control CE-excess denominator is non-positive")
    result["recovery_ratio_of_sums"] = {
        "kl_at_pos": float(1.0 - values["kl_at_pos"].sum() / zero_pos),
        "kl_mean_first16": float(
            1.0 - values["kl_mean_first16"].sum() / zero_16
        ),
        "ce_excess_first16": float(
            1.0
            - (values["ce_first16"] - clean_ce).sum()
            / zero_ce_excess
        ),
    }
    return result


def top_indices(
    values: np.ndarray,
    count: int = 3,
    largest: bool = True,
    eligible: np.ndarray | None = None,
) -> list[int]:
    candidates = np.arange(len(values))
    if eligible is not None:
        candidates = candidates[eligible]
    ordered = sorted(
        candidates.tolist(),
        key=lambda idx: ((-values[idx]) if largest else values[idx], idx),
    )
    return [int(idx) for idx in ordered[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--n4-vectors", required=True, type=Path)
    parser.add_argument("--n4-explanations", required=True, type=Path)
    parser.add_argument("--n4-causal", required=True, type=Path)
    parser.add_argument("--n4-analysis", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--pilot-common", required=True, type=Path)
    parser.add_argument("--j2-explanations", required=True, type=Path)
    parser.add_argument("--j2-av-checkpoint", required=True, type=Path)
    parser.add_argument("--j2-vectors", required=True, type=Path)
    parser.add_argument("--j2-result", required=True, type=Path)
    parser.add_argument("--j2-causal", required=True, type=Path)
    parser.add_argument("--j2-causal-checkpoint", required=True, type=Path)
    parser.add_argument("--recon-script", required=True, type=Path)
    parser.add_argument("--causal-script", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shortlist-out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    input_hashes = {
        "activations": sha256_file(args.activations),
        "n4_vectors": sha256_file(args.n4_vectors),
        "n4_explanations": sha256_file(args.n4_explanations),
        "n4_causal": sha256_file(args.n4_causal),
        "n4_analysis": sha256_file(args.n4_analysis),
        "model_manifest": sha256_file(args.model_manifest),
        "pilot_common": sha256_file(args.pilot_common),
        "j2_explanations": sha256_file(args.j2_explanations),
        "j2_av_checkpoint": sha256_file(args.j2_av_checkpoint),
        "j2_vectors": sha256_file(args.j2_vectors),
        "j2_result": sha256_file(args.j2_result),
        "j2_causal": sha256_file(args.j2_causal),
        "j2_causal_checkpoint": sha256_file(args.j2_causal_checkpoint),
        "recon_script": sha256_file(args.recon_script),
        "causal_script": sha256_file(args.causal_script),
        "protocol": sha256_file(args.protocol),
        "script": sha256_file(__file__),
    }
    for label, expected in EXPECTED.items():
        if input_hashes[label] != expected:
            raise SystemExit(f"{label} hash mismatch")

    j2_result = json.loads(args.j2_result.read_text(encoding="utf-8"))
    j2_explanations = json.loads(
        args.j2_explanations.read_text(encoding="utf-8")
    )
    n4_explanations = json.loads(
        args.n4_explanations.read_text(encoding="utf-8")
    )
    n4_causal = json.loads(args.n4_causal.read_text(encoding="utf-8"))
    j2_causal = json.loads(args.j2_causal.read_text(encoding="utf-8"))
    if j2_result.get("status") != "EXPLORATORY_RECON_COMPLETE":
        raise SystemExit("J2 reconstruction artifact is not complete")
    recon_inputs = j2_result.get("inputs")
    if not isinstance(recon_inputs, dict):
        raise SystemExit("J2 reconstruction inputs missing")
    recon_expected = {
        "activations": input_hashes["activations"],
        "n4_vectors": input_hashes["n4_vectors"],
        "n4_explanations": input_hashes["n4_explanations"],
        "model_manifest": input_hashes["model_manifest"],
        "pilot_common": input_hashes["pilot_common"],
        "protocol": input_hashes["protocol"],
        "script": input_hashes["recon_script"],
    }
    for key, expected in recon_expected.items():
        if recon_inputs.get(key) != expected:
            raise SystemExit(f"J2 reconstruction input mismatch: {key}")
    if j2_result.get("contract_sha256") != expected_recon_contract(
        recon_inputs
    ):
        raise SystemExit("J2 reconstruction contract digest mismatch")
    if j2_result.get("outputs", {}).get("vectors_sha256") != input_hashes[
        "j2_vectors"
    ]:
        raise SystemExit("J2 result/vector binding mismatch")
    if (
        j2_result.get("outputs", {}).get("explanations_sha256")
        != input_hashes["j2_explanations"]
    ):
        raise SystemExit("J2 result/explanation binding mismatch")
    if (
        j2_result.get("outputs", {}).get("checkpoint_sha256")
        != input_hashes["j2_av_checkpoint"]
    ):
        raise SystemExit("J2 result/AV-checkpoint binding mismatch")
    if j2_causal.get("status") != "EXPLORATORY_CAUSAL_COMPLETE":
        raise SystemExit("J2 causal artifact is not complete")
    causal_inputs = j2_causal.get("inputs", {})
    causal_expected = {
        "activations": input_hashes["activations"],
        "j2_vectors": input_hashes["j2_vectors"],
        "j2_result": input_hashes["j2_result"],
        "j2_explanations": input_hashes["j2_explanations"],
        "j2_av_checkpoint": input_hashes["j2_av_checkpoint"],
        "j2_recon_script": input_hashes["recon_script"],
        "protocol": input_hashes["protocol"],
        "model_manifest": input_hashes["model_manifest"],
        "script": input_hashes["causal_script"],
        "checkpoint_sha256": input_hashes["j2_causal_checkpoint"],
    }
    for key, expected in causal_expected.items():
        if causal_inputs.get(key) != expected:
            raise SystemExit(f"J2 causal input mismatch: {key}")

    with np.load(args.n4_vectors, allow_pickle=False) as archive:
        required_n4 = {
            "x",
            "doc_ids",
            "positions",
            "pred_orig",
            "recon_sae_small",
            "recon_sae_big",
        }
        missing = sorted(required_n4 - set(archive.files))
        if missing:
            raise SystemExit(f"N4 vector archive missing keys: {missing}")
        x = np.asarray(archive["x"], dtype=np.float32)
        doc_ids = np.asarray(archive["doc_ids"], dtype=np.int64)
        positions = np.asarray(archive["positions"], dtype=np.int64)
        direct = np.asarray(archive["pred_orig"], dtype=np.float32)
        native = {
            condition: np.asarray(
                archive[f"recon_{condition}"], dtype=np.float32
            )
            for condition in CONDITIONS
        }
    with np.load(args.j2_vectors, allow_pickle=False) as archive:
        required_j2 = {
            "doc_ids",
            "positions",
            *LOOP_KEYS.values(),
            *SECOND_KEYS.values(),
            *LOOP_SAE_KEYS.values(),
            *DIRECT_SAE_KEYS.values(),
        }
        missing = sorted(required_j2 - set(archive.files))
        if missing:
            raise SystemExit(f"J2 vector archive missing keys: {missing}")
        if not np.array_equal(
            np.asarray(archive["doc_ids"], dtype=np.int64), doc_ids
        ):
            raise SystemExit("J2/N4 doc_id mismatch")
        if not np.array_equal(
            np.asarray(archive["positions"], dtype=np.int64), positions
        ):
            raise SystemExit("J2/N4 position mismatch")
        loops = {
            condition: np.asarray(archive[LOOP_KEYS[condition]], dtype=np.float32)
            for condition in CONDITIONS
        }
        second = {
            condition: np.asarray(
                archive[SECOND_KEYS[condition]], dtype=np.float32
            )
            for condition in CONDITIONS
        }
        loop_sae = {
            condition: np.asarray(
                archive[LOOP_SAE_KEYS[condition]], dtype=np.float32
            )
            for condition in CONDITIONS
        }
        direct_sae = {
            condition: np.asarray(
                archive[DIRECT_SAE_KEYS[condition]], dtype=np.float32
            )
            for condition in CONDITIONS
        }
    if x.shape != (200, 3840):
        raise SystemExit(f"unexpected cohort shape: {x.shape}")
    if doc_ids.shape != (200,) or positions.shape != (200,):
        raise SystemExit("N4 metadata arrays have invalid shape")
    vector_sets = {
        "x": x,
        "direct": direct,
        **{f"native_{key}": value for key, value in native.items()},
        **{f"loop_{key}": value for key, value in loops.items()},
        **{f"second_{key}": value for key, value in second.items()},
        **{f"loop_sae_{key}": value for key, value in loop_sae.items()},
        **{f"direct_sae_{key}": value for key, value in direct_sae.items()},
    }
    for label, array in vector_sets.items():
        if array.shape != (200, 3840) or not np.all(np.isfinite(array)):
            raise SystemExit(f"{label} has invalid shape or non-finite values")
    mean_hat = unit(x.mean(axis=0, dtype=np.float64))

    direct_metrics, direct_rows = vector_metric_bundle(
        direct, x, mean_hat, doc_ids
    )
    aggregate_vectors: dict[str, Any] = {
        "nla_direct_to_x": direct_metrics
    }
    row_vectors: dict[str, dict[str, np.ndarray]] = {
        "nla_direct_to_x": direct_rows
    }
    for condition in CONDITIONS:
        for name, prediction, target in (
            (f"{condition}_to_x", native[condition], x),
            (f"{condition}_2_to_{condition}", second[condition], native[condition]),
            (f"{condition}_2_to_x", second[condition], x),
            (f"{condition}_loop_to_{condition}", loops[condition], native[condition]),
            (f"{condition}_loop_to_x", loops[condition], x),
            (
                f"{condition}_direct_sae_to_x",
                direct_sae[condition],
                x,
            ),
            (
                f"{condition}_direct_sae_to_nla_direct",
                direct_sae[condition],
                direct,
            ),
            (f"{condition}_loop_sae_to_x", loop_sae[condition], x),
            (
                f"{condition}_loop_sae_to_loop",
                loop_sae[condition],
                loops[condition],
            ),
        ):
            aggregate, rows = vector_metric_bundle(
                prediction, target, mean_hat, doc_ids
            )
            aggregate_vectors[name] = aggregate
            row_vectors[name] = rows

    geometry_paired_contrasts: dict[str, Any] = {}
    for condition in CONDITIONS:
        contrast_pairs = {
            f"{condition}_loop_to_x_minus_nla_direct_to_x": (
                f"{condition}_loop_to_x",
                "nla_direct_to_x",
            ),
            f"{condition}_loop_to_x_minus_{condition}_to_x": (
                f"{condition}_loop_to_x",
                f"{condition}_to_x",
            ),
            f"{condition}_loop_to_{condition}_minus_nla_direct_to_x": (
                f"{condition}_loop_to_{condition}",
                "nla_direct_to_x",
            ),
            f"{condition}_direct_sae_to_x_minus_nla_direct_to_x": (
                f"{condition}_direct_sae_to_x",
                "nla_direct_to_x",
            ),
        }
        for contrast_name, (left_key, right_key) in contrast_pairs.items():
            geometry_paired_contrasts[contrast_name] = {}
            for metric in (
                "centered_cosine",
                "lodo_centered_cosine",
                "raw_cosine",
            ):
                delta = (
                    row_vectors[left_key][metric]
                    - row_vectors[right_key][metric]
                )
                geometry_paired_contrasts[contrast_name][metric] = {
                    "summary": summarize_values(delta),
                    "document_cluster_bootstrap_mean": (
                        document_cluster_bootstrap(delta, doc_ids)
                    ),
                }

    new_explanation_rows = j2_explanations.get("rows", [])
    old_explanation_rows = n4_explanations.get("rows", [])
    if len(new_explanation_rows) != 200 or len(old_explanation_rows) != 200:
        raise SystemExit("explanation artifacts must each contain 200 rows")
    new_indices = [int(row["idx"]) for row in new_explanation_rows]
    old_indices = [int(row["idx"]) for row in old_explanation_rows]
    if (
        len(set(new_indices)) != 200
        or sorted(new_indices) != list(range(200))
        or len(set(old_indices)) != 200
        or sorted(old_indices) != list(range(200))
    ):
        raise SystemExit("explanation idx coverage/uniqueness failure")
    new_explanations_by_idx = {
        int(row["idx"]): row for row in new_explanation_rows
    }
    old_explanations_by_idx = {
        int(row["idx"]): row["explanation"]
        for row in old_explanation_rows
    }
    old_meta_by_idx = {
        int(row["idx"]): row for row in old_explanation_rows
    }
    for idx in range(200):
        new_row = new_explanations_by_idx[idx]
        old_row = old_meta_by_idx[idx]
        if (
            int(new_row["doc_id"]) != int(doc_ids[idx])
            or int(new_row["position"]) != int(positions[idx])
            or int(old_row["doc_id"]) != int(doc_ids[idx])
            or int(old_row["position"]) != int(positions[idx])
            or new_row["token"] != old_row["token"]
        ):
            raise SystemExit(f"explanation metadata mismatch at idx {idx}")
    text_rows: dict[str, list[dict[str, float | int]]] = {
        condition: [] for condition in CONDITIONS
    }
    for idx in range(200):
        direct_text = old_explanations_by_idx[idx]
        for condition in CONDITIONS:
            text_rows[condition].append(
                text_diagnostics(
                    direct_text,
                    new_explanations_by_idx[idx][condition],
                )
            )
    text_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        text_summary[condition] = {
            key: summarize_values(
                np.asarray([row[key] for row in text_rows[condition]], dtype=float)
            )
            for key in text_rows[condition][0]
        }

    n4_rows = sorted(n4_causal["rows"], key=lambda row: int(row["idx"]))
    j2_rows = sorted(j2_causal["rows"], key=lambda row: int(row["idx"]))
    if [row["idx"] for row in n4_rows] != list(range(200)):
        raise SystemExit("N4 causal rows are not aligned")
    if [row["idx"] for row in j2_rows] != list(range(200)):
        raise SystemExit("J2 causal rows are not aligned")
    for idx, (old_row, new_row) in enumerate(zip(n4_rows, j2_rows)):
        if (
            int(old_row["doc_id"]) != int(doc_ids[idx])
            or int(new_row["doc_id"]) != int(doc_ids[idx])
            or int(old_row["position"]) != int(positions[idx])
            or int(new_row["position"]) != int(positions[idx])
            or old_row.get("token_id") != new_row.get("token_id")
            or old_row.get("token") != new_row.get("token")
            or old_row.get("token") != new_explanations_by_idx[idx]["token"]
        ):
            raise SystemExit(f"causal metadata mismatch at idx {idx}")
    old_zero = causal_arrays(n4_rows, "zero")
    new_zero = causal_arrays(j2_rows, "zero")
    old_identity = causal_arrays(n4_rows, "identity")
    new_identity = causal_arrays(j2_rows, "identity")
    old_clean_ce = np.asarray(
        [float(row["ce_clean_first16"]) for row in n4_rows],
        dtype=np.float64,
    )
    new_clean_ce = np.asarray(
        [float(row["ce_clean_first16"]) for row in j2_rows],
        dtype=np.float64,
    )
    control_recheck = {
        "zero": {
            key: float(np.max(np.abs(old_zero[key] - new_zero[key])))
            for key in old_zero
        },
        "identity": {
            key: float(
                np.max(np.abs(old_identity[key] - new_identity[key]))
            )
            for key in old_identity
        },
        "clean_ce_first16": float(
            np.max(np.abs(old_clean_ce - new_clean_ce))
        ),
    }
    control_recheck_overall = max(
        *control_recheck["zero"].values(),
        *control_recheck["identity"].values(),
        control_recheck["clean_ce_first16"],
    )
    if control_recheck_overall > CONTROL_RECHECK_MAX_ABS_TOL:
        raise SystemExit(
            "new/old causal-control reproduction gate failed: "
            f"{control_recheck_overall} > "
            f"{CONTROL_RECHECK_MAX_ABS_TOL}"
        )
    zero_recheck = {
        key: float(np.max(np.abs(old_zero[key] - new_zero[key])))
        for key in old_zero
    }

    causal: dict[str, Any] = {
        "control_recheck_max_abs": control_recheck,
        "control_recheck_overall_max_abs": control_recheck_overall,
        "control_recheck_max_abs_tolerance": (
            CONTROL_RECHECK_MAX_ABS_TOL
        ),
    }
    causal_row_values: dict[str, dict[str, np.ndarray]] = {}
    causal_row_values["identity"] = old_identity
    causal_row_values["nla_direct"] = causal_arrays(n4_rows, "orig")
    causal_row_values["sae_small"] = causal_arrays(n4_rows, "sae_small")
    causal_row_values["sae_big"] = causal_arrays(n4_rows, "sae_big")
    causal_row_values["small_loop"] = causal_arrays(j2_rows, "small_loop")
    causal_row_values["big_loop"] = causal_arrays(j2_rows, "big_loop")
    causal_row_values["direct_small"] = causal_arrays(
        j2_rows, "direct_small"
    )
    causal_row_values["direct_big"] = causal_arrays(j2_rows, "direct_big")
    causal_row_values["zero"] = old_zero
    for name, values in causal_row_values.items():
        causal[name] = causal_summary(values, old_zero, old_clean_ce)

    causal["paired_contrasts"] = {}
    for condition, loop_name, direct_name in (
        ("sae_small", "small_loop", "direct_small"),
        ("sae_big", "big_loop", "direct_big"),
    ):
        contrast_pairs = [
            (loop_name, "nla_direct"),
            (loop_name, condition),
            (loop_name, direct_name),
            (direct_name, "nla_direct"),
            (direct_name, condition),
        ]
        for left_name, comparator in contrast_pairs:
            contrast_name = f"{left_name}_minus_{comparator}"
            causal["paired_contrasts"][contrast_name] = {}
            for metric in ("kl_at_pos", "kl_mean_first16", "ce_first16"):
                delta = (
                    causal_row_values[left_name][metric]
                    - causal_row_values[comparator][metric]
                )
                causal["paired_contrasts"][contrast_name][metric] = {
                    "summary": summarize_values(delta),
                    "document_cluster_bootstrap_mean": document_cluster_bootstrap(
                        delta, doc_ids
                    ),
                }

    result_rows = j2_result.get("rows", [])
    result_indices = [int(row["idx"]) for row in result_rows]
    if (
        len(result_rows) != 200
        or len(set(result_indices)) != 200
        or sorted(result_indices) != list(range(200))
    ):
        raise SystemExit("J2 result idx coverage/uniqueness failure")
    result_rows_by_idx = {int(row["idx"]): row for row in result_rows}
    for idx, row in result_rows_by_idx.items():
        if (
            int(row["doc_id"]) != int(doc_ids[idx])
            or int(row["position"]) != int(positions[idx])
            or row["token"] != new_explanations_by_idx[idx]["token"]
        ):
            raise SystemExit(f"J2 result metadata mismatch at idx {idx}")

    fixed_point_statistics: dict[str, Any] = {}
    grounding_code_contrasts: dict[str, Any] = {}
    primary_fixed_fields = {
        "support_jaccard",
        "weighted_code_cosine",
        "l0_change",
        "birth_mass_ratio_second",
        "reconstruction_2_raw_cosine_to_first",
        "loop_support_jaccard_vs_first",
        "loop_weighted_code_cosine_vs_first",
        "loop_birth_mass_ratio",
        "direct_support_jaccard_vs_first",
        "direct_weighted_code_cosine_vs_first",
        "direct_birth_mass_ratio",
        "loop_support_jaccard_vs_direct",
        "loop_weighted_code_cosine_vs_direct",
    }
    for condition in CONDITIONS:
        fixed_rows = [
            result_rows_by_idx[idx]["fixed_point"][condition]
            for idx in range(200)
        ]
        numeric_fields = sorted(
            key
            for key, value in fixed_rows[0].items()
            if key != "idx"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        fixed_point_statistics[condition] = {}
        for field in numeric_fields:
            values = np.asarray(
                [float(row[field]) for row in fixed_rows],
                dtype=np.float64,
            )
            record: dict[str, Any] = {
                "summary": summarize_values(values)
            }
            if field in primary_fixed_fields:
                record["document_cluster_bootstrap_mean"] = (
                    document_cluster_bootstrap(values, doc_ids)
                )
            fixed_point_statistics[condition][field] = record

        code_pairs = {
            "loop_minus_direct_weighted_code_cosine_vs_first": (
                "loop_weighted_code_cosine_vs_first",
                "direct_weighted_code_cosine_vs_first",
            ),
            "loop_minus_direct_support_jaccard_vs_first": (
                "loop_support_jaccard_vs_first",
                "direct_support_jaccard_vs_first",
            ),
            "loop_minus_direct_top20_overlap_vs_first": (
                "loop_top20_overlap_vs_first",
                "direct_top20_overlap_vs_first",
            ),
            "loop_minus_direct_birth_mass_ratio": (
                "loop_birth_mass_ratio",
                "direct_birth_mass_ratio",
            ),
        }
        grounding_code_contrasts[condition] = {}
        for name, (left_field, right_field) in code_pairs.items():
            delta = np.asarray(
                [
                    float(row[left_field]) - float(row[right_field])
                    for row in fixed_rows
                ],
                dtype=np.float64,
            )
            grounding_code_contrasts[condition][name] = {
                "summary": summarize_values(delta),
                "document_cluster_bootstrap_mean": (
                    document_cluster_bootstrap(delta, doc_ids)
                ),
            }

    shortlist_categories: dict[str, list[dict[str, Any]]] = {}
    for condition, loop_name in (
        ("sae_small", "small_loop"),
        ("sae_big", "big_loop"),
    ):
        native_centered = row_vectors[f"{condition}_to_x"]["centered_cosine"]
        native_raw = row_vectors[f"{condition}_to_x"]["raw_cosine"]
        loop_target = row_vectors[
            f"{condition}_loop_to_{condition}"
        ]["centered_cosine"]
        support_jaccard = np.asarray(
            [
                result_rows_by_idx[idx]["fixed_point"][condition][
                    "support_jaccard"
                ]
                for idx in range(200)
            ],
            dtype=float,
        )
        birth_mass = np.asarray(
            [
                result_rows_by_idx[idx]["fixed_point"][condition][
                    "birth_mass_ratio_second"
                ]
                for idx in range(200)
            ],
            dtype=float,
        )
        loop_birth_mass = np.asarray(
            [
                result_rows_by_idx[idx]["fixed_point"][condition][
                    "loop_birth_mass_ratio"
                ]
                for idx in range(200)
            ],
            dtype=float,
        )
        loop_code_cosine = np.asarray(
            [
                result_rows_by_idx[idx]["fixed_point"][condition][
                    "loop_weighted_code_cosine_vs_first"
                ]
                for idx in range(200)
            ],
            dtype=float,
        )
        direct_code_cosine = np.asarray(
            [
                result_rows_by_idx[idx]["fixed_point"][condition][
                    "direct_weighted_code_cosine_vs_first"
                ]
                for idx in range(200)
            ],
            dtype=float,
        )
        grounding_code_gain = loop_code_cosine - direct_code_cosine
        token_jaccard = np.asarray(
            [row["token_jaccard"] for row in text_rows[condition]], dtype=float
        )
        loop_kl = causal_row_values[loop_name]["kl_at_pos"]
        direct_kl = causal_row_values["nla_direct"]["kl_at_pos"]
        native_kl = causal_row_values[condition]["kl_at_pos"]
        high_centered = native_centered >= np.median(native_centered)
        high_raw = native_raw >= np.median(native_raw)

        category_indices = {
            f"{condition}:high_fidelity_high_code_churn": top_indices(
                support_jaccard, largest=False, eligible=high_centered
            ),
            f"{condition}:language_loop_rescue": top_indices(
                loop_kl - direct_kl, largest=False
            ),
            f"{condition}:language_loop_catastrophe": top_indices(
                loop_kl - native_kl, largest=True
            ),
            f"{condition}:tiny_geometry_large_text_change": top_indices(
                token_jaccard, largest=False, eligible=high_raw
            ),
            f"{condition}:worst_sae_manifold_roundtrip": top_indices(
                loop_target, largest=False
            ),
            f"{condition}:fixed_point_leakage": top_indices(
                birth_mass, largest=True
            ),
            f"{condition}:language_code_leakage": top_indices(
                loop_birth_mass, largest=True
            ),
            f"{condition}:sae_grounding_code_rescue": top_indices(
                grounding_code_gain, largest=True
            ),
            f"{condition}:sae_grounding_code_catastrophe": top_indices(
                grounding_code_gain, largest=False
            ),
        }
        for category, indices in category_indices.items():
            shortlist_categories[category] = []
            for rank, idx in enumerate(indices, 1):
                shortlist_categories[category].append(
                    {
                        "rank": rank,
                        "idx": idx,
                        "doc_id": int(doc_ids[idx]),
                        "condition": condition,
                        "native_centered_cosine_to_x": float(
                            native_centered[idx]
                        ),
                        "native_raw_cosine_to_x": float(native_raw[idx]),
                        "loop_centered_cosine_to_sae": float(loop_target[idx]),
                        "support_jaccard": float(support_jaccard[idx]),
                        "birth_mass_ratio_second": float(birth_mass[idx]),
                        "loop_birth_mass_ratio": float(loop_birth_mass[idx]),
                        "loop_code_cosine_vs_first": float(
                            loop_code_cosine[idx]
                        ),
                        "direct_code_cosine_vs_first": float(
                            direct_code_cosine[idx]
                        ),
                        "grounding_code_cosine_gain": float(
                            grounding_code_gain[idx]
                        ),
                        "text_token_jaccard": float(token_jaccard[idx]),
                        "kl_loop": float(loop_kl[idx]),
                        "kl_nla_direct": float(direct_kl[idx]),
                        "kl_sae_native": float(native_kl[idx]),
                        "kl_loop_minus_direct": float(
                            loop_kl[idx] - direct_kl[idx]
                        ),
                        "kl_loop_minus_sae": float(
                            loop_kl[idx] - native_kl[idx]
                        ),
                    }
                )

    shortlist_payload = {
        "schema_version": 1,
        "experiment": "J2-P0 SAE projection language-loop case shortlist",
        "status": "FROZEN_BEFORE_HUMAN_CASE_READING",
        "confirmatory": False,
        "selection_contract": (
            "nine protocol-defined categories x two SAE operating points x top3; "
            "overlap allowed; ties broken by idx"
        ),
        "inputs": input_hashes,
        "categories": shortlist_categories,
        "unique_indices": sorted(
            {
                case["idx"]
                for cases in shortlist_categories.values()
                for case in cases
            }
        ),
    }
    shortlist_sha = write_frozen(args.shortlist_out, shortlist_payload)

    analysis_payload = {
        "schema_version": 1,
        "experiment": "J2-P0 SAE projection language loop",
        "status": "EXPLORATORY_ANALYSIS_COMPLETE",
        "confirmatory": False,
        "claim_scope": "mechanism_audit_and_hypothesis_generation_only",
        "inputs": input_hashes,
        "integrity": {
            "n": 200,
            "documents": int(len(np.unique(doc_ids))),
            "conditions": list(CONDITIONS),
            "case_shortlist_sha256": shortlist_sha,
            "zero_recheck_max_abs": zero_recheck,
            "control_recheck_overall_max_abs": (
                control_recheck_overall
            ),
            "control_recheck_max_abs_tolerance": (
                CONTROL_RECHECK_MAX_ABS_TOL
            ),
        },
        "vector_geometry": aggregate_vectors,
        "geometry_paired_contrasts": geometry_paired_contrasts,
        "fixed_point_generation_summary": j2_result[
            "fixed_point_summary"
        ],
        "fixed_point_statistics": fixed_point_statistics,
        "grounding_code_contrasts": grounding_code_contrasts,
        "text_diagnostics": text_summary,
        "causal": causal,
        "case_shortlist": {
            "sha256": shortlist_sha,
            "categories": len(shortlist_categories),
            "unique_indices": len(shortlist_payload["unique_indices"]),
        },
        "limitations": [
            "reuses the N4 cohort and is exploratory",
            "single target model, layer, NLA pair, and SAE family",
            "text diagnostics are not proposition-level human fidelity",
            "SAE projection and language bottleneck are serial, so losses are non-additive",
            "case studies are hypothesis-generating and cannot replace aggregate statistics",
        ],
    }
    analysis_sha = write_frozen(args.out, analysis_payload)

    lines = [
        "# J2-P0 SAE projection → language loop",
        "",
        "Status: **EXPLORATORY_ANALYSIS_COMPLETE**",
        "",
        "## Vector geometry",
        "",
        "| Condition | target | centered cos | LODO cos | centered FVE |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for suffix, target_label in (
            ("to_x", "x"),
            (f"loop_to_{condition}", condition),
            ("loop_to_x", "x"),
        ):
            key = f"{condition}_{suffix}"
            metric = aggregate_vectors[key]
            lines.append(
                f"| `{key}` | `{target_label}` | "
                f"{metric['centered_cosine_mean']:.6f} | "
                f"{metric['lodo_centered_cosine_mean']:.6f} | "
                f"{metric['centered_fve']:.6f} |"
            )
    lines += [
        "",
        "## Causal KL at patched position",
        "",
        "| Condition | mean KL | median KL | aggregate recovery |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "nla_direct",
        "sae_small",
        "small_loop",
        "direct_small",
        "sae_big",
        "big_loop",
        "direct_big",
    ):
        item = causal[name]
        lines.append(
            f"| `{name}` | {item['kl_at_pos']['mean']:.6f} | "
            f"{item['kl_at_pos']['median']:.6f} | "
            f"{item['recovery_ratio_of_sums']['kl_at_pos']:.6f} |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "This reused-cohort result is a mechanism audit, not confirmatory evidence "
        "that SAE-grounded NLA is superior.",
        "",
        f"Frozen metric-only case shortlist SHA-256: `{shortlist_sha}`.",
    ]
    markdown_payload = "\n".join(lines) + "\n"
    if args.markdown.exists():
        if args.markdown.read_text(encoding="utf-8") != markdown_payload:
            raise SystemExit("refusing to overwrite non-identical markdown")
    else:
        args.markdown.write_text(markdown_payload, encoding="utf-8")
    markdown_sha = sha256_file(args.markdown)
    args.markdown.with_suffix(args.markdown.suffix + ".sha256").write_text(
        f"{markdown_sha}  {args.markdown.name}\n", encoding="utf-8"
    )
    print(
        f"J2_ANALYSIS_COMPLETE analysis={analysis_sha} "
        f"shortlist={shortlist_sha} markdown={markdown_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
