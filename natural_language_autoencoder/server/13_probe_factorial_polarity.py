#!/usr/bin/env python3
"""B6+B4 GPU batch: semantic readability, polarity, and carrier interventions.

The feature cohort must already be frozen by 12_select_factorial_features.py.
This script never selects or drops features based on AV/AR output.

Workload:
  1. isolated directions: every selected/control direction, both signs,
     one greedy plus four seeded stochastic AV generations;
  2. natural carriers: for every semantic direction, greedy explanations of
     a held-out high-activation carrier before amplification/ablation and a
     held-out low-activation carrier before/after insertion;
  3. AR reconstruction of every explanation exactly once, centered scoring,
     signed retrieval, and feature-level summaries.

AV rows are appended to JSONL with fsync and stable keys, so an interrupted
run can resume without repeating completed generations.  No shutdown command
is issued by this script or its runner.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot_common import AVLocal, EXPLANATION_RE, NLACritic


GENERIC_TEXTS = [
    "The passage uses a structured informational style and continues the current topic.",
    "This is a coherent piece of explanatory prose with ordinary grammatical structure.",
    "The context establishes a subject and prepares a likely continuation of the discussion.",
    "The text contains semantic and syntactic information typical of a written document.",
    "A descriptive answer is being developed in a clear and organized format.",
    "The final token fits a locally predictable continuation in the surrounding sentence.",
    "The activation reflects general language structure, topical context, and discourse form.",
    "This appears to be an informative response that elaborates on previously introduced material.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def project_rows(x: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - np.outer(x @ m_hat, m_hat)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm row cannot be normalized")
    return x / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def centered_cosine(
    a: np.ndarray, b: np.ndarray, m_hat: np.ndarray
) -> float:
    a_p = a - (a @ m_hat) * m_hat
    b_p = b - (b @ m_hat) * m_hat
    return cosine(a_p, b_p)


def token_jaccard(a: str, b: str) -> float:
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def percentile_interval(
    values: list[float],
    rng: np.random.Generator,
    n_bootstrap: int,
    statistic: str = "median",
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True)
    if statistic == "median":
        estimates = np.median(samples, axis=1)
    elif statistic == "mean":
        estimates = np.mean(samples, axis=1)
    else:
        raise ValueError(statistic)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def binomial_sign_p_one_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return float("nan")
    return float(
        sum(math.comb(trials, k) for k in range(successes, trials + 1))
        / (2**trials)
    )


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return round(float(value), 8)
    if isinstance(value, np.integer):
        return int(value)
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["key"]
        if key in rows:
            raise ValueError(f"duplicate checkpoint key {key} at line {line_number}")
        rows[key] = row
    return rows


@torch.inference_mode()
def generate_with_raw(
    av: AVLocal,
    vector: np.ndarray,
    *,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, str, bool]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    embeds_np, _ = av.client._build_embeds(
        torch.as_tensor(np.asarray(vector, np.float32)), None
    )
    inputs = torch.from_numpy(embeds_np)[None].to(
        av.device, av.model.dtype
    )
    attention = torch.ones(
        inputs.shape[:2], dtype=torch.long, device=av.device
    )
    kwargs = {
        "inputs_embeds": inputs,
        "attention_mask": attention,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": av.tok.eos_token_id,
    }
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature)
    else:
        kwargs.update(do_sample=False)
    output = av.model.generate(**kwargs)
    raw = av.tok.decode(output[0], skip_special_tokens=False)
    match = EXPLANATION_RE.search(raw)
    explanation = match.group(1).strip() if match else raw
    return raw, explanation, bool(match)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--av", required=True, type=Path)
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vectors-out", required=True, type=Path)
    parser.add_argument("--samples-per-sign", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and print the job plan without loading AV/AR.",
    )
    parser.add_argument(
        "--stop-after-av-jobs",
        type=int,
        default=0,
        help=(
            "For a resumable smoke test, generate only this many pending AV "
            "rows and exit before AR. Zero runs the complete experiment."
        ),
    )
    parser.add_argument(
        "--debug-direction-limit",
        type=int,
        default=0,
        help="Debug only: restrict the run to the first N frozen directions.",
    )
    args = parser.parse_args()

    if args.samples_per_sign < 1:
        raise ValueError("--samples-per-sign must be positive")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    records = selection["selected_directions"]
    if selection["status"] != "selection_frozen_before_AV_AR":
        raise ValueError("selection manifest is not frozen")

    with np.load(args.vectors, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float32)
        row_doc_ids = np.asarray(archive["row_doc_ids"], dtype=np.int64)
        m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
        target_norm = float(archive["target_norm"])
        direction_ids = np.asarray(archive["direction_ids"], dtype=np.int64)
        direction_groups = np.asarray(archive["direction_groups"]).astype(str)
        direction_labels = np.asarray(archive["direction_labels"]).astype(str)
        directions = np.asarray(archive["directions"], dtype=np.float32)

    if len(records) != len(direction_ids):
        raise ValueError("selection JSON and direction NPZ length mismatch")
    for index, record in enumerate(records):
        if int(record["feature"]) != int(direction_ids[index]):
            raise ValueError(f"direction order mismatch at {index}")
    if args.debug_direction_limit > 0:
        limit = min(args.debug_direction_limit, len(records))
        records = records[:limit]
        direction_ids = direction_ids[:limit]
        direction_groups = direction_groups[:limit]
        direction_labels = direction_labels[:limit]
        directions = directions[:limit]
    m_hat /= np.linalg.norm(m_hat)
    direction_norms = np.linalg.norm(directions, axis=1)
    direction_unit = directions / np.maximum(direction_norms[:, None], 1e-12)
    centered_direction_unit = normalize_rows(
        project_rows(direction_unit, m_hat)
    )

    jobs: list[dict[str, Any]] = []
    job_inputs: dict[str, np.ndarray] = {}

    for direction_index, record in enumerate(records):
        pure = direction_unit[direction_index] * target_norm
        for sign in (1, -1):
            for sample_index in range(args.samples_per_sign):
                temperature = 0.0 if sample_index == 0 else args.temperature
                key = (
                    f"direction:f{int(record['feature'])}:s{sign:+d}:"
                    f"sample{sample_index}"
                )
                job = {
                    "key": key,
                    "probe_type": "direction",
                    "direction_index": direction_index,
                    "feature": int(record["feature"]),
                    "group": record["group"],
                    "label": record.get("label"),
                    "selection_tier": record.get("selection_tier"),
                    "sign": sign,
                    "sample_index": sample_index,
                    "temperature": temperature,
                    "seed": stable_seed(args.seed, key),
                }
                jobs.append(job)
                job_inputs[key] = (sign * pure).astype(np.float32)

    # On-manifold-ish carrier tests are restricted to semantic cohorts.
    semantic_records = [
        (index, record)
        for index, record in enumerate(records)
        if record["group"] in {"semantic_new", "semantic_legacy"}
    ]
    for direction_index, record in semantic_records:
        feature = int(record["feature"])
        if feature < 0:
            continue
        coefficient = float(record["carrier_high"]["activation"])
        raw_direction = directions[direction_index]
        high_row = int(record["carrier_high"]["row_index"])
        low_row = int(record["carrier_low"]["row_index"])
        carrier_specs = [
            ("high_baseline", high_row, x[high_row]),
            (
                "high_amplify",
                high_row,
                x[high_row] + coefficient * raw_direction,
            ),
            (
                "high_ablate",
                high_row,
                x[high_row] - coefficient * raw_direction,
            ),
            ("low_baseline", low_row, x[low_row]),
            (
                "low_insert",
                low_row,
                x[low_row] + coefficient * raw_direction,
            ),
        ]
        for condition, row_index, vector in carrier_specs:
            key = f"carrier:f{feature}:{condition}:row{row_index}"
            jobs.append(
                {
                    "key": key,
                    "probe_type": "carrier",
                    "direction_index": direction_index,
                    "feature": feature,
                    "group": record["group"],
                    "label": record.get("label"),
                    "selection_tier": record.get("selection_tier"),
                    "carrier_condition": condition,
                    "carrier_row_index": row_index,
                    "carrier_doc_id": int(row_doc_ids[row_index]),
                    "intervention_coefficient": coefficient,
                    "sample_index": 0,
                    "temperature": 0.0,
                    "seed": stable_seed(args.seed, key),
                }
            )
            job_inputs[key] = np.asarray(vector, dtype=np.float32)

    keys = [job["key"] for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("constructed duplicate job keys")

    checkpoint_rows = load_checkpoint(args.checkpoint)
    unknown = set(checkpoint_rows) - set(keys)
    if unknown:
        raise ValueError(
            f"checkpoint contains keys not in current protocol: {sorted(unknown)[:5]}"
        )

    pending = [job for job in jobs if job["key"] not in checkpoint_rows]
    print(
        f"[plan] total_jobs={len(jobs)} complete={len(checkpoint_rows)} "
        f"pending={len(pending)} directions={len(records)} "
        f"semantic_carriers={len(semantic_records)}"
    )
    if args.plan_only:
        print("B6B4_PLAN_VALID")
        return

    pending_to_run = (
        pending[: args.stop_after_av_jobs]
        if args.stop_after_av_jobs > 0
        else pending
    )

    av_started = time.time()
    if pending_to_run:
        av = AVLocal(str(args.av), device="cuda")
        for ordinal, job in enumerate(pending_to_run, start=1):
            started = time.time()
            raw, explanation, tag_ok = generate_with_raw(
                av,
                job_inputs[job["key"]],
                temperature=float(job["temperature"]),
                max_new_tokens=args.max_new_tokens,
                seed=int(job["seed"]),
            )
            row = {
                **job,
                "raw_completion": raw,
                "explanation": explanation,
                "explanation_tag_ok": tag_ok,
                "generation_seconds": time.time() - started,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(args.checkpoint, row)
            checkpoint_rows[job["key"]] = row
            if (
                ordinal == 1
                or ordinal % 10 == 0
                or ordinal == len(pending_to_run)
            ):
                elapsed = time.time() - av_started
                rate = elapsed / ordinal
                remaining = rate * (len(pending_to_run) - ordinal)
                print(
                    f"[AV {ordinal}/{len(pending_to_run)}] key={job['key']} "
                    f"tag={tag_ok} {row['generation_seconds']:.1f}s "
                    f"eta={remaining/60:.1f}m"
                )
        del av
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    av_elapsed = time.time() - av_started
    if args.stop_after_av_jobs > 0 and len(pending_to_run) < len(pending):
        print(
            f"B6B4_SMOKE_STOP generated={len(pending_to_run)} "
            f"remaining={len(pending) - len(pending_to_run)}"
        )
        return

    ordered_rows = [checkpoint_rows[job["key"]] for job in jobs]

    critic_started = time.time()
    critic = NLACritic(str(args.ar), device="cuda")
    reconstruction_cache: dict[str, np.ndarray] = {}
    all_texts = [row["explanation"] for row in ordered_rows] + GENERIC_TEXTS
    for index, text in enumerate(dict.fromkeys(all_texts), start=1):
        reconstruction_cache[text] = critic.reconstruct(text).numpy()
        if index == 1 or index % 25 == 0 or index == len(set(all_texts)):
            print(f"[AR {index}/{len(set(all_texts))}]")
    critic_elapsed = time.time() - critic_started
    del critic
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    input_matrix = np.stack([job_inputs[row["key"]] for row in ordered_rows])
    reconstruction_matrix = np.stack(
        [reconstruction_cache[row["explanation"]] for row in ordered_rows]
    )
    reconstruction_centered = normalize_rows(
        project_rows(reconstruction_matrix, m_hat)
    )
    similarity = reconstruction_centered @ centered_direction_unit.T
    signed_similarity = np.concatenate([similarity, -similarity], axis=1)
    abs_similarity = np.abs(similarity)

    scored_rows = []
    direction_job_indices = []
    carrier_job_indices = []
    for row_index, row in enumerate(ordered_rows):
        scored = dict(row)
        scored["input_norm"] = float(np.linalg.norm(input_matrix[row_index]))
        scored["reconstruction_norm"] = float(
            np.linalg.norm(reconstruction_matrix[row_index])
        )
        if row["probe_type"] == "direction":
            direction_job_indices.append(row_index)
            direction_index = int(row["direction_index"])
            sign = int(row["sign"])
            axis_cos = float(similarity[row_index, direction_index])
            signed_target_index = (
                direction_index
                if sign > 0
                else len(records) + direction_index
            )
            signed_order = np.argsort(
                -signed_similarity[row_index], kind="stable"
            )
            feature_order = np.argsort(
                -abs_similarity[row_index], kind="stable"
            )
            scored.update(
                {
                    "axis_cos_centered": axis_cos,
                    "target_cos_centered": sign * axis_cos,
                    "target_cos_raw": cosine(
                        reconstruction_matrix[row_index],
                        sign * direction_unit[direction_index],
                    ),
                    "signed_retrieval_rank": int(
                        np.flatnonzero(signed_order == signed_target_index)[0]
                    )
                    + 1,
                    "feature_retrieval_rank": int(
                        np.flatnonzero(feature_order == direction_index)[0]
                    )
                    + 1,
                    "signed_top5_indices": [
                        int(value) for value in signed_order[:5]
                    ],
                    "feature_top5_direction_indices": [
                        int(value) for value in feature_order[:5]
                    ],
                }
            )
        else:
            carrier_job_indices.append(row_index)
        scored_rows.append(scored)

    # Pair + and - direction probes by direction and sample index.
    row_by_direction_key = {
        (
            int(row["direction_index"]),
            int(row["sign"]),
            int(row["sample_index"]),
        ): (index, row)
        for index, row in enumerate(scored_rows)
        if row["probe_type"] == "direction"
    }
    polarity_rows = []
    for direction_index, record in enumerate(records):
        for sample_index in range(args.samples_per_sign):
            plus_index, plus = row_by_direction_key[
                (direction_index, 1, sample_index)
            ]
            minus_index, minus = row_by_direction_key[
                (direction_index, -1, sample_index)
            ]
            difference = (
                reconstruction_matrix[plus_index]
                - reconstruction_matrix[minus_index]
            )
            polarity_rows.append(
                {
                    "direction_index": direction_index,
                    "feature": int(record["feature"]),
                    "group": record["group"],
                    "label": record.get("label"),
                    "selection_tier": record.get("selection_tier"),
                    "sample_index": sample_index,
                    "q_plus": plus["axis_cos_centered"],
                    "q_minus_axis": minus["axis_cos_centered"],
                    "r_minus": -minus["axis_cos_centered"],
                    "polarity": 0.5
                    * (
                        plus["axis_cos_centered"]
                        - minus["axis_cos_centered"]
                    ),
                    "difference_direction_cos_centered": centered_cosine(
                        difference, directions[direction_index], m_hat
                    ),
                    "sign_correct": bool(
                        plus["axis_cos_centered"] > 0
                        and minus["axis_cos_centered"] < 0
                    ),
                    "explanations_exactly_equal": bool(
                        plus["explanation"] == minus["explanation"]
                    ),
                    "explanation_token_jaccard": token_jaccard(
                        plus["explanation"], minus["explanation"]
                    ),
                    "plus_signed_rank": plus["signed_retrieval_rank"],
                    "minus_signed_rank": minus["signed_retrieval_rank"],
                    "plus_feature_rank": plus["feature_retrieval_rank"],
                    "minus_feature_rank": minus["feature_retrieval_rank"],
                }
            )

    # Carrier effects use AR reconstruction differences, not surface text alone.
    carrier_lookup = {
        (int(row["direction_index"]), row["carrier_condition"]): (index, row)
        for index, row in enumerate(scored_rows)
        if row["probe_type"] == "carrier"
    }
    carrier_effects = []
    for direction_index, record in semantic_records:
        high_base_index, high_base = carrier_lookup[
            (direction_index, "high_baseline")
        ]
        amplify_index, amplify = carrier_lookup[
            (direction_index, "high_amplify")
        ]
        ablate_index, ablate = carrier_lookup[
            (direction_index, "high_ablate")
        ]
        low_base_index, low_base = carrier_lookup[
            (direction_index, "low_baseline")
        ]
        insert_index, insert = carrier_lookup[
            (direction_index, "low_insert")
        ]
        effects = {
            "amplify_minus_high": (
                reconstruction_matrix[amplify_index]
                - reconstruction_matrix[high_base_index]
            ),
            "high_minus_ablate": (
                reconstruction_matrix[high_base_index]
                - reconstruction_matrix[ablate_index]
            ),
            "insert_minus_low": (
                reconstruction_matrix[insert_index]
                - reconstruction_matrix[low_base_index]
            ),
        }
        carrier_effects.append(
            {
                "direction_index": direction_index,
                "feature": int(record["feature"]),
                "group": record["group"],
                "label": record.get("label"),
                "selection_tier": record.get("selection_tier"),
                "heldout_auc": (
                    record.get("test", {}).get("auc")
                    if record["group"] == "semantic_new"
                    else None
                ),
                "heldout_effect": (
                    record.get("test", {}).get("raw_difference")
                    if record["group"] == "semantic_new"
                    else None
                ),
                "carrier_activation": float(
                    record["carrier_high"]["activation"]
                ),
                **{
                    f"{name}_cos_centered": centered_cosine(
                        delta, directions[direction_index], m_hat
                    )
                    for name, delta in effects.items()
                },
                "amplify_text_jaccard": token_jaccard(
                    amplify["explanation"], high_base["explanation"]
                ),
                "ablate_text_jaccard": token_jaccard(
                    ablate["explanation"], high_base["explanation"]
                ),
                "insert_text_jaccard": token_jaccard(
                    insert["explanation"], low_base["explanation"]
                ),
                "amplify_text_changed": (
                    amplify["explanation"] != high_base["explanation"]
                ),
                "ablate_text_changed": (
                    ablate["explanation"] != high_base["explanation"]
                ),
                "insert_text_changed": (
                    insert["explanation"] != low_base["explanation"]
                ),
            }
        )

    generic_reconstructions = np.stack(
        [reconstruction_cache[text] for text in GENERIC_TEXTS]
    )
    generic_centered = normalize_rows(
        project_rows(generic_reconstructions, m_hat)
    )
    generic_similarity = generic_centered @ centered_direction_unit.T

    # Direction-level summaries use greedy sample 0; generations are not
    # treated as independent statistical units.
    greedy_polarity = [
        row for row in polarity_rows if row["sample_index"] == 0
    ]
    record_by_feature = {
        int(record["feature"]): record for record in records
    }

    def heldout_valid(row: dict[str, Any]) -> bool:
        record = record_by_feature[int(row["feature"])]
        if record["group"] != "semantic_new":
            return False
        test = record["test"]
        return (
            float(test["auc"]) >= 0.75
            and float(test["raw_difference"]) > 0
            and float(test["pos_support"]) >= 2
        )

    cohorts: dict[str, list[dict[str, Any]]] = {
        "semantic_new_intention_to_test": [
            row for row in greedy_polarity if row["group"] == "semantic_new"
        ],
        "semantic_new_heldout_valid": [
            row for row in greedy_polarity if heldout_valid(row)
        ],
        "semantic_legacy": [
            row for row in greedy_polarity if row["group"] == "semantic_legacy"
        ],
        "structural": [
            row for row in greedy_polarity if row["group"] == "structural"
        ],
        "active_nonselective": [
            row
            for row in greedy_polarity
            if row["group"] == "active_nonselective"
        ],
        "gaussian": [
            row for row in greedy_polarity if row["group"] == "gaussian"
        ],
    }
    bootstrap_rng = np.random.default_rng(args.seed + 900_000)
    cohort_summaries = {}
    for name, cohort in cohorts.items():
        if not cohort:
            cohort_summaries[name] = {"n": 0}
            continue
        q_plus = [float(row["q_plus"]) for row in cohort]
        r_minus = [float(row["r_minus"]) for row in cohort]
        polarity = [float(row["polarity"]) for row in cohort]
        sign_successes = sum(bool(row["sign_correct"]) for row in cohort)
        signed_ranks = [
            rank
            for row in cohort
            for rank in (row["plus_signed_rank"], row["minus_signed_rank"])
        ]
        feature_ranks = [
            rank
            for row in cohort
            for rank in (row["plus_feature_rank"], row["minus_feature_rank"])
        ]
        cohort_summaries[name] = {
            "n_features": len(cohort),
            "q_plus_mean": float(np.mean(q_plus)),
            "q_plus_median": float(np.median(q_plus)),
            "q_plus_median_bootstrap_95": percentile_interval(
                q_plus, bootstrap_rng, args.bootstrap
            ),
            "r_minus_mean": float(np.mean(r_minus)),
            "r_minus_median": float(np.median(r_minus)),
            "polarity_mean": float(np.mean(polarity)),
            "polarity_median": float(np.median(polarity)),
            "polarity_median_bootstrap_95": percentile_interval(
                polarity, bootstrap_rng, args.bootstrap
            ),
            "polarity_positive_fraction": float(
                np.mean(np.asarray(polarity) > 0)
            ),
            "polarity_positive_sign_test_p_one_sided": (
                binomial_sign_p_one_sided(
                    sum(value > 0 for value in polarity), len(polarity)
                )
            ),
            "sign_accuracy": sign_successes / len(cohort),
            "signed_retrieval_top1": float(
                np.mean(np.asarray(signed_ranks) <= 1)
            ),
            "signed_retrieval_top5": float(
                np.mean(np.asarray(signed_ranks) <= 5)
            ),
            "signed_retrieval_mrr": float(
                np.mean(1.0 / np.asarray(signed_ranks))
            ),
            "feature_retrieval_top1": float(
                np.mean(np.asarray(feature_ranks) <= 1)
            ),
            "feature_retrieval_top5": float(
                np.mean(np.asarray(feature_ranks) <= 5)
            ),
            "feature_retrieval_mrr": float(
                np.mean(1.0 / np.asarray(feature_ranks))
            ),
            "exact_plus_minus_text_match_fraction": float(
                np.mean(
                    [
                        row["explanations_exactly_equal"]
                        for row in cohort
                    ]
                )
            ),
            "mean_plus_minus_token_jaccard": float(
                np.mean(
                    [row["explanation_token_jaccard"] for row in cohort]
                )
            ),
        }

    carrier_cohorts = {
        "semantic_new_intention_to_test": [
            row for row in carrier_effects if row["group"] == "semantic_new"
        ],
        "semantic_new_heldout_valid": [
            row
            for row in carrier_effects
            if row["group"] == "semantic_new"
            and row["heldout_auc"] is not None
            and float(row["heldout_auc"]) >= 0.75
            and float(row["heldout_effect"]) > 0
            and record_by_feature[int(row["feature"])]["test"]["pos_support"]
            >= 2
        ],
        "semantic_legacy": [
            row for row in carrier_effects if row["group"] == "semantic_legacy"
        ],
    }
    carrier_summaries = {}
    effect_names = (
        "amplify_minus_high_cos_centered",
        "high_minus_ablate_cos_centered",
        "insert_minus_low_cos_centered",
    )
    for name, cohort in carrier_cohorts.items():
        if not cohort:
            carrier_summaries[name] = {"n": 0}
            continue
        carrier_summaries[name] = {"n_features": len(cohort)}
        for effect_name in effect_names:
            values = [float(row[effect_name]) for row in cohort]
            carrier_summaries[name][f"{effect_name}_mean"] = float(
                np.mean(values)
            )
            carrier_summaries[name][f"{effect_name}_median"] = float(
                np.median(values)
            )
            carrier_summaries[name][
                f"{effect_name}_positive_fraction"
            ] = float(np.mean(np.asarray(values) > 0))

    stochastic_by_feature = []
    grouped = defaultdict(list)
    for row in polarity_rows:
        if row["sample_index"] > 0:
            grouped[int(row["feature"])].append(row)
    for feature, rows in grouped.items():
        stochastic_by_feature.append(
            {
                "feature": feature,
                "group": rows[0]["group"],
                "label": rows[0].get("label"),
                "n_samples": len(rows),
                "q_plus_mean": float(np.mean([row["q_plus"] for row in rows])),
                "q_plus_std": float(np.std([row["q_plus"] for row in rows])),
                "polarity_mean": float(
                    np.mean([row["polarity"] for row in rows])
                ),
                "polarity_std": float(
                    np.std([row["polarity"] for row in rows])
                ),
                "sign_consistency": float(
                    np.mean([row["sign_correct"] for row in rows])
                ),
            }
        )

    output = {
        "schema_version": 1,
        "experiment": "B6+B4 factorial semantic readability and polarity",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "selection": str(args.selection),
            "selection_sha256": sha256_file(args.selection),
            "vectors": str(args.vectors),
            "vectors_sha256": sha256_file(args.vectors),
            "av": str(args.av),
            "ar": str(args.ar),
            "n_directions": len(records),
            "n_jobs": len(jobs),
            "n_direction_jobs": len(direction_job_indices),
            "n_carrier_jobs": len(carrier_job_indices),
        },
        "generation": {
            "samples_per_sign": args.samples_per_sign,
            "greedy_sample_index": 0,
            "stochastic_temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "av_elapsed_seconds_this_invocation": av_elapsed,
            "critic_elapsed_seconds": critic_elapsed,
            "explanation_tag_success_fraction": float(
                np.mean(
                    [row["explanation_tag_ok"] for row in ordered_rows]
                )
            ),
            "mean_generation_seconds": float(
                np.mean([row["generation_seconds"] for row in ordered_rows])
            ),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        },
        "protocol_notes": {
            "centering": (
                "project frozen train-document-balanced mean direction out "
                "of prediction and target before cosine"
            ),
            "primary_unit": "feature; repeated generations are uncertainty only",
            "negative_direction_scope": (
                "-w_dec is an isolated off-manifold signed-axis test, not a "
                "semantic antifeature claim"
            ),
            "heldout_valid_gate": (
                "test AUROC>=0.75, positive heldout effect, positive support>=2"
            ),
            "signed_candidate_count": 2 * len(records),
            "feature_candidate_count": len(records),
        },
        "summary_by_cohort_greedy": cohort_summaries,
        "carrier_summary": carrier_summaries,
        "generic_control": {
            "texts": GENERIC_TEXTS,
            "mean_abs_centered_cos_all_directions": float(
                np.mean(np.abs(generic_similarity))
            ),
            "max_abs_centered_cos_all_directions": float(
                np.max(np.abs(generic_similarity))
            ),
            "mean_abs_centered_cos_by_group": {
                group: float(
                    np.mean(
                        np.abs(
                            generic_similarity[
                                :, direction_groups == group
                            ]
                        )
                    )
                )
                for group in sorted(set(direction_groups))
            },
        },
        "polarity_rows": polarity_rows,
        "carrier_effects": carrier_effects,
        "stochastic_by_feature": stochastic_by_feature,
        "scored_generation_rows": scored_rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_builtin(output), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    np.savez_compressed(
        args.vectors_out,
        keys=np.asarray([row["key"] for row in ordered_rows]),
        probe_types=np.asarray([row["probe_type"] for row in ordered_rows]),
        input_vectors=input_matrix.astype(np.float32),
        reconstruction_vectors=reconstruction_matrix.astype(np.float32),
        direction_ids=direction_ids,
        direction_groups=direction_groups,
        direction_labels=direction_labels,
        directions=directions,
        m_hat=m_hat.astype(np.float32),
        direction_similarity_matrix=similarity.astype(np.float32),
        generic_reconstruction_vectors=generic_reconstructions.astype(np.float32),
        generic_direction_similarity=generic_similarity.astype(np.float32),
    )

    print("B6B4_FACTORIAL_COMPLETE")
    print(
        json.dumps(
            {
                name: {
                    key: value
                    for key, value in summary.items()
                    if key
                    in {
                        "n_features",
                        "q_plus_median",
                        "r_minus_median",
                        "polarity_median",
                        "sign_accuracy",
                        "signed_retrieval_top1",
                        "feature_retrieval_top1",
                    }
                }
                for name, summary in cohort_summaries.items()
            }
        )
    )
    print(f"wrote -> {args.out} + {args.vectors_out}")


if __name__ == "__main__":
    main()
