#!/usr/bin/env python3
"""Run the C1 protocol pilot on the frozen B6+B4 feature pool.

Stages:
1. A plain Gemma-3-12B-IT model, not AV/AR, produces a context-only
   autointerpretation and a blind semantic paraphrase of each existing NLA
   explanation.
2. The same plain base model provides a provisional blind text-context score
   for every candidate.  This is an independent-model signal, not human truth.
3. NLA AR reconstructs every unique candidate text exactly once.
4. All scores are computed in the frozen B6 centered space.

Every generation is append-checkpointed.  This script never shuts down the
machine and never selects features based on C1 outcomes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot_common import NLACritic


LABEL_RE = re.compile(r"<label>\s*(.*?)\s*</label>", re.I | re.S)
PARAPHRASE_RE = re.compile(r"<paraphrase>\s*(.*?)\s*</paraphrase>", re.I | re.S)
SCORE_RE = re.compile(r"\bSCORE\s*=\s*([0-3])\b", re.I)
UNSUPPORTED_RE = re.compile(r"\bUNSUPPORTED\s*=\s*([01])\b", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["key"]
        if key in output:
            raise ValueError(f"duplicate checkpoint key {key} line {line_number}")
        output[key] = row
    return output


def clean_generated(text: str) -> str:
    text = text.replace("<end_of_turn>", " ").strip()
    return " ".join(text.split())


def extract_tag(text: str, regex: re.Pattern[str]) -> str:
    match = regex.search(text)
    if match:
        return clean_generated(match.group(1))
    cleaned = clean_generated(text)
    for prefix in ("LABEL:", "PARAPHRASE:", "Label:", "Paraphrase:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def format_contexts(contexts: list[dict[str, Any]], include_activation: bool) -> str:
    blocks = []
    for index, row in enumerate(contexts, 1):
        header = (
            f"[Context {index}; language={row['axis_language']}; "
            f"topic={row['topic']}"
        )
        if include_activation:
            header += (
                f"; active_token={row['token']!r}; "
                f"activation={float(row['activation']):.3f}"
            )
        header += "]"
        blocks.append(f"{header}\n{row['prompt']}")
    return "\n\n".join(blocks)


def autointerp_prompt(record: dict[str, Any]) -> str:
    contexts = format_contexts(record["train_contexts"], include_activation=True)
    return f"""You are labeling one sparse-autoencoder feature from training evidence.
The contexts below are the only evidence you may use.  The indicated token is
where this feature was highly active.  Write one concise English label of at
most 24 words.  Describe only recurring evidence; do not invent named entities,
products, code, or causal claims.  Do not mention feature IDs or activation
values.

{contexts}

Return exactly:
<label>your concise label</label>"""


def paraphrase_prompt(nla_text: str) -> str:
    return f"""Rewrite the following explanation using substantially different
wording while preserving its literal semantic claims, named entities, and
topic.  Do not see or infer any hidden feature label.  Do not correct the
explanation or move it toward a more plausible topic.  Remove only meta phrases
such as "Article structure", "The phrase", "Final token", and lists of guesses.
Use one or two sentences.

EXPLANATION:
{nla_text}

Return exactly:
<paraphrase>your semantic paraphrase</paraphrase>"""


def judge_prompt(record: dict[str, Any], candidate_text: str, candidate_id: str) -> str:
    positive = format_contexts(record["judge_positive_contexts"], include_activation=False)
    negative = format_contexts(record["judge_negative_contexts"], include_activation=False)
    return f"""Act as a blind evaluator of a proposed SAE feature label.  You do
not know the label source or feature identity.  Determine whether the candidate
selectively describes the POSITIVE held-out contexts better than the NEGATIVE
held-out contexts.

Scoring:
3 = specific, well-supported match to the positive set and discriminates negatives
2 = correct broad axis but incomplete or overly broad
1 = weak, generic, ambiguous, or only incidentally related
0 = contradicted, wrong language/domain, or unrelated

UNSUPPORTED=1 if the candidate asserts concrete entities, technologies, events,
or mechanisms not supported by the positive contexts; otherwise 0.

Blind candidate {candidate_id}:
{candidate_text}

POSITIVE HELD-OUT CONTEXTS:
{positive}

NEGATIVE HELD-OUT CONTEXTS:
{negative}

Return exactly two lines:
SCORE=<0|1|2|3>
UNSUPPORTED=<0|1>"""


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    tokenized = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(tokenized):
        tokenized = tokenized["input_ids"]
    input_ids = tokenized.to(next(model.parameters()).device)
    attention_mask = torch.ones_like(input_ids)
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(
        output[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    return text, elapsed


def project_rows(x: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m_hat = np.asarray(m_hat, dtype=np.float64)
    return x - np.outer(x @ m_hat, m_hat)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero projected row")
    return x / norms


def average_rank(scores: np.ndarray, target_index: int) -> float:
    target = float(scores[target_index])
    greater = int(np.sum(scores > target))
    equal_other = int(np.sum(scores == target)) - 1
    return 1.0 + greater + 0.5 * max(0, equal_other)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vectors-out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--stop-after-base-jobs",
        type=int,
        default=0,
        help="Smoke-test only: stop after this many pending base generation jobs.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark_sha256 = sha256_file(args.benchmark)
    if benchmark["status"] != "benchmark_frozen_before_C1_AR":
        raise ValueError("benchmark is not frozen")
    records = benchmark["records"]
    if len(records) != 24:
        raise ValueError("expected 24 feature records")

    with np.load(args.vectors, allow_pickle=False) as archive:
        direction_ids = np.asarray(archive["direction_ids"], dtype=np.int64)
        direction_groups = np.asarray(archive["direction_groups"])
        direction_labels = np.asarray(archive["direction_labels"])
        directions = np.asarray(archive["directions"], dtype=np.float32)
        m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
    semantic_indices = np.array(
        [
            index
            for index, group in enumerate(direction_groups.tolist())
            if group == "semantic_new"
        ],
        dtype=np.int64,
    )
    if len(semantic_indices) != 24:
        raise ValueError("semantic direction count drift")
    semantic_directions = directions[semantic_indices]
    semantic_ids = direction_ids[semantic_indices]
    semantic_labels = direction_labels[semantic_indices].tolist()
    if semantic_ids.tolist() != [int(row["feature"]) for row in records]:
        raise ValueError("benchmark/vector feature order mismatch")
    m_norm = np.linalg.norm(m_hat)
    if not np.isfinite(m_norm) or abs(m_norm - 1.0) > 1e-4:
        raise ValueError(f"m_hat norm {m_norm}")
    direction_centered = normalize_rows(project_rows(semantic_directions, m_hat))

    checkpoint_rows = load_checkpoint(args.checkpoint)
    candidate_generation_specs = []
    for record in records:
        feature = int(record["feature"])
        nla_original = next(
            row["text"]
            for row in record["static_candidates"]
            if row["kind"] == "nla_original"
        )
        for request in record["generation_requests"]:
            kind = request["kind"]
            key = f"gen:f{feature}:{kind}"
            prompt = (
                autointerp_prompt(record)
                if kind == "base_autointerp"
                else paraphrase_prompt(nla_original)
            )
            candidate_generation_specs.append(
                {
                    "key": key,
                    "feature": feature,
                    "kind": kind,
                    "candidate_id": request["candidate_id"],
                    "prompt": prompt,
                    "input_sha256": sha256_text(prompt),
                    "max_new_tokens": 64 if kind == "base_autointerp" else 180,
                }
            )

    for spec in candidate_generation_specs:
        existing = checkpoint_rows.get(spec["key"])
        if (
            existing is not None
            and existing.get("input_sha256") is not None
            and existing["input_sha256"] != spec["input_sha256"]
        ):
            raise ValueError(
                f"checkpoint input hash mismatch for {spec['key']}; "
                "use a fresh checkpoint"
            )
    pending_generation = [
        row for row in candidate_generation_specs if row["key"] not in checkpoint_rows
    ]
    print(
        f"[plan] feature_generations={len(candidate_generation_specs)} "
        f"pending={len(pending_generation)} checkpoint={len(checkpoint_rows)}"
    )

    base_model = None
    tokenizer = None
    base_started = time.perf_counter()
    generation_limit = (
        args.stop_after_base_jobs
        if args.stop_after_base_jobs > 0
        else len(pending_generation)
    )
    if pending_generation:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.base), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(args.base),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        for ordinal, spec in enumerate(pending_generation[:generation_limit], 1):
            raw, elapsed = generate(
                base_model,
                tokenizer,
                spec["prompt"],
                int(spec["max_new_tokens"]),
            )
            text = extract_tag(
                raw, LABEL_RE if spec["kind"] == "base_autointerp" else PARAPHRASE_RE
            )
            if not text:
                raise ValueError(f"empty generated candidate {spec['key']}")
            row = {
                "key": spec["key"],
                "job_type": "candidate_generation",
                "feature": spec["feature"],
                "kind": spec["kind"],
                "candidate_id": spec["candidate_id"],
                "input_sha256": spec["input_sha256"],
                "benchmark_sha256": benchmark_sha256,
                "raw": raw,
                "text": text,
                "seconds": elapsed,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(args.checkpoint, row)
            checkpoint_rows[spec["key"]] = row
            print(
                f"[base candidate {ordinal}/{min(len(pending_generation), generation_limit)}] "
                f"{spec['key']} {elapsed:.2f}s"
            )

    if (
        args.stop_after_base_jobs > 0
        and generation_limit < len(pending_generation)
    ):
        print(
            f"C1_PILOT_SMOKE_STOP completed={generation_limit} "
            f"remaining={len(pending_generation)-generation_limit}"
        )
        return

    all_candidates: list[dict[str, Any]] = []
    record_by_feature = {int(row["feature"]): row for row in records}
    for record in records:
        feature = int(record["feature"])
        for candidate in record["static_candidates"]:
            all_candidates.append({**candidate, "feature": feature, "label": record["label"]})
        for request in record["generation_requests"]:
            generated = checkpoint_rows[f"gen:f{feature}:{request['kind']}"]
            all_candidates.append(
                {
                    "candidate_id": request["candidate_id"],
                    "feature": feature,
                    "label": record["label"],
                    "kind": request["kind"],
                    "ordinal": 0,
                    "text": generated["text"],
                    "expected_validity": "unknown",
                    "generated": True,
                }
            )
    if len({row["candidate_id"] for row in all_candidates}) != len(all_candidates):
        raise ValueError("duplicate assembled candidate id")

    judge_prompts = {}
    pending_judges = []
    for candidate in all_candidates:
        record = record_by_feature[int(candidate["feature"])]
        prompt = judge_prompt(record, candidate["text"], candidate["candidate_id"])
        judge_prompts[candidate["candidate_id"]] = prompt
        key = f"judge:{candidate['candidate_id']}"
        existing = checkpoint_rows.get(key)
        expected_hash = sha256_text(prompt)
        if (
            existing is not None
            and existing.get("input_sha256") is not None
            and existing["input_sha256"] != expected_hash
        ):
            raise ValueError(
                f"checkpoint input hash mismatch for {key}; use a fresh checkpoint"
            )
        if existing is None:
            pending_judges.append(candidate)
    print(
        f"[plan] candidates={len(all_candidates)} judge_pending={len(pending_judges)}"
    )
    if pending_judges and base_model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.base), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(args.base),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
    for ordinal, candidate in enumerate(pending_judges, 1):
        record = record_by_feature[int(candidate["feature"])]
        prompt = judge_prompts[candidate["candidate_id"]]
        raw, elapsed = generate(base_model, tokenizer, prompt, 16)
        score_match = SCORE_RE.search(raw)
        unsupported_match = UNSUPPORTED_RE.search(raw)
        if score_match is None or unsupported_match is None:
            retry_prompt = (
                prompt
                + "\nYour prior format was invalid. Output only SCORE=n and "
                "UNSUPPORTED=n with no other text."
            )
            retry_raw, retry_elapsed = generate(base_model, tokenizer, retry_prompt, 12)
            raw = raw + "\n[RETRY]\n" + retry_raw
            elapsed += retry_elapsed
            score_match = SCORE_RE.search(retry_raw)
            unsupported_match = UNSUPPORTED_RE.search(retry_raw)
        row = {
            "key": f"judge:{candidate['candidate_id']}",
            "job_type": "blind_base_context_judge",
            "candidate_id": candidate["candidate_id"],
            "feature": int(candidate["feature"]),
            "kind": candidate["kind"],
            "input_sha256": sha256_text(prompt),
            "benchmark_sha256": benchmark_sha256,
            "raw": raw,
            "score": int(score_match.group(1)) if score_match else None,
            "unsupported": (
                int(unsupported_match.group(1)) if unsupported_match else None
            ),
            "parse_ok": bool(score_match and unsupported_match),
            "seconds": elapsed,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(args.checkpoint, row)
        checkpoint_rows[row["key"]] = row
        if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(pending_judges):
            print(
                f"[base judge {ordinal}/{len(pending_judges)}] "
                f"parse={row['parse_ok']} {elapsed:.2f}s"
            )

    if base_model is not None:
        del base_model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    base_seconds = time.perf_counter() - base_started

    unique_texts = list(dict.fromkeys(row["text"] for row in all_candidates))
    ar_started = time.perf_counter()
    critic = NLACritic(str(args.ar), device="cuda")
    reconstruction_cache = {}
    for index, text in enumerate(unique_texts, 1):
        reconstruction_cache[text] = critic.reconstruct(text).numpy()
        if index == 1 or index % 50 == 0 or index == len(unique_texts):
            print(f"[AR {index}/{len(unique_texts)}]")
    torch.cuda.synchronize()
    ar_seconds = time.perf_counter() - ar_started
    del critic
    gc.collect()
    torch.cuda.empty_cache()

    reconstruction_matrix = np.stack(
        [reconstruction_cache[row["text"]] for row in all_candidates]
    ).astype(np.float32)
    if not np.isfinite(reconstruction_matrix).all():
        raise ValueError("non-finite AR reconstruction")
    reconstruction_centered = normalize_rows(
        project_rows(reconstruction_matrix, m_hat)
    )
    similarity = reconstruction_centered @ direction_centered.T

    label_order = list(dict.fromkeys(semantic_labels))
    label_direction_indices = {
        label: np.array(
            [index for index, value in enumerate(semantic_labels) if value == label],
            dtype=np.int64,
        )
        for label in label_order
    }
    scored_rows = []
    max_nla_recompute_error = 0.0
    original_nla_scores = {
        int(row["feature"]): float(row["b6_nla_original_target_cos_centered"])
        for row in records
    }
    for row_index, candidate in enumerate(all_candidates):
        feature = int(candidate["feature"])
        target_index = int(np.where(semantic_ids == feature)[0][0])
        scores = similarity[row_index]
        target_score = float(scores[target_index])
        label = str(candidate["label"])
        axis_scores = np.array(
            [
                float(np.mean(scores[label_direction_indices[value]]))
                for value in label_order
            ]
        )
        axis_index = label_order.index(label)
        other_axis_mask = np.array(
            [value != label for value in semantic_labels], dtype=bool
        )
        judge = checkpoint_rows[f"judge:{candidate['candidate_id']}"]
        scored = {
            **candidate,
            "target_cos_centered": target_score,
            "feature_retrieval_rank": average_rank(scores, target_index),
            "feature_retrieval_top5": bool(average_rank(scores, target_index) <= 5),
            "axis_mean_cos_centered": float(axis_scores[axis_index]),
            "axis_retrieval_rank": average_rank(axis_scores, axis_index),
            "axis_retrieval_top1": bool(average_rank(axis_scores, axis_index) == 1),
            "target_beats_all_different_axis_directions": bool(
                target_score > float(np.max(scores[other_axis_mask]))
            ),
            "blind_base_judge_score": judge["score"],
            "blind_base_judge_unsupported": judge["unsupported"],
            "blind_base_judge_parse_ok": judge["parse_ok"],
        }
        scored_rows.append(scored)
        if candidate["kind"] == "nla_original":
            max_nla_recompute_error = max(
                max_nla_recompute_error,
                abs(target_score - original_nla_scores[feature]),
            )
    if max_nla_recompute_error > 1e-5:
        raise ValueError(
            f"NLA original AR recomputation drift {max_nla_recompute_error}"
        )

    output = {
        "schema_version": 1,
        "experiment": "C1 external-validity protocol pilot",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": benchmark["scope"],
        "inputs": {
            "benchmark": {
                "path": str(args.benchmark),
                "sha256": benchmark_sha256,
            },
            "vectors": {"path": str(args.vectors), "sha256": sha256_file(args.vectors)},
            "base": str(args.base),
            "ar": str(args.ar),
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256_file(args.checkpoint),
            },
        },
        "protocol": benchmark["protocol"],
        "runtime": {
            "base_seconds_this_invocation": base_seconds,
            "ar_seconds": ar_seconds,
            "n_candidates": len(all_candidates),
            "n_unique_texts": len(unique_texts),
            "n_checkpoint_rows": len(checkpoint_rows),
            "judge_parse_ok": int(
                sum(
                    checkpoint_rows[f"judge:{row['candidate_id']}"]["parse_ok"]
                    for row in all_candidates
                )
            ),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "qa": {
            "max_abs_nla_original_recompute_error": max_nla_recompute_error,
            "all_reconstructions_finite": True,
            "feature_order": semantic_ids.tolist(),
            "label_order": label_order,
        },
        "feature_metadata": [
            {
                key: record[key]
                for key in (
                    "feature",
                    "label",
                    "selection_tier",
                    "sibling_feature",
                    "train_metrics",
                    "test_metrics",
                    "b6_nla_original_target_cos_centered",
                    "heldout_valid_by_b6_rule",
                    "reference_evidence_level",
                )
            }
            for record in records
        ],
        "scored_candidates": scored_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.vectors_out,
        candidate_ids=np.array(
            [row["candidate_id"] for row in all_candidates], dtype="U16"
        ),
        feature_ids=np.array([row["feature"] for row in all_candidates], dtype=np.int64),
        kinds=np.array([row["kind"] for row in all_candidates], dtype="U32"),
        reconstruction_vectors=reconstruction_matrix,
        semantic_feature_ids=semantic_ids,
        semantic_directions=semantic_directions,
        semantic_similarity=similarity.astype(np.float32),
        m_hat=m_hat.astype(np.float32),
    )
    print("C1_PILOT_COMPLETE")
    print(
        json.dumps(
            {
                "candidates": len(all_candidates),
                "unique_texts": len(unique_texts),
                "judge_parse_ok": output["runtime"]["judge_parse_ok"],
                "nla_recompute_error": max_nla_recompute_error,
                "base_seconds": base_seconds,
                "ar_seconds": ar_seconds,
                "out": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
