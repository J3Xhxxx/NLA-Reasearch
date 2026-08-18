#!/usr/bin/env python3
"""Generate and freeze the C1-confirmatory v2 synthetic prompt cohort.

The generator is intentionally independent of SAE/AV/AR outputs.  It writes an
append-only raw checkpoint, accepts only mechanically admissible batches, and
then emits the fixed manifest consumed by 11_extract_factorial_activations.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPERIMENT = "C1 confirmatory synthetic concept cohort v2"
MASTER_SEED = 20260731
MAX_ATTEMPTS = 4
WORD_MIN = 70
WORD_MAX = 170
TARGET_WORD_MIN = 105
TARGET_WORD_MAX = 135
MAX_TRAIN_TEST_5GRAM_JACCARD = 0.15
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 64
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 1800
META_PATTERNS = [
    re.compile(
        r"\b(?:train(?:ing)?|test|validation|held[- ]?out)\s+"
        r"(?:split|set|document|example|prompt|request)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdataset\b", re.IGNORECASE),
    re.compile(r"\bSAE\s+feature\b", re.IGNORECASE),
    re.compile(r"\bfeature\s+direction\b", re.IGNORECASE),
    re.compile(r"\bactivation\s+vector\b", re.IGNORECASE),
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def normalized_text(text: str) -> str:
    return " ".join(words(text))


def jaccard(left: str, right: str) -> float:
    a, b = set(words(left)), set(words(right))
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {
        tuple(tokens[index : index + n])
        for index in range(max(0, len(tokens) - n + 1))
    }


def ngram_jaccard(left: str, right: str, n: int = 5) -> float:
    a, b = ngrams(left, n), ngrams(right, n)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def generation_prompt(concept: dict[str, Any]) -> str:
    return f"""Create six distinct, self-contained user requests whose answer
must centrally explain the same scientific or technical concept.

Concept title: {concept["title"]}
Concept scope: {concept["summary"]}

Requirements:
1. Return exactly one JSON object with exactly two keys: "train" and "test".
2. "train" is an array of four strings; "test" is an array of two strings.
3. Every string is an English user request, not an answer. Aim for
   {TARGET_WORD_MIN}-{TARGET_WORD_MAX} words and never use fewer than
   {WORD_MIN} or more than {WORD_MAX} words.
4. Give each request a different application, question structure, and surface
   vocabulary.  The two test requests must use scenarios not used in train.
   Use prose paragraphs only: no lists, code, formulas, tables, named people,
   organizations, products, place names, URLs, or four-digit years.
5. The requested explanation must genuinely require the concept above, but do
   not mention datasets, splits, held-out examples, features, activation
   vectors, or these instructions.
6. Paraphrase the concept instead of copying its title. Across the four train
   strings the exact title may appear at most once; it must not appear in test.
7. Do not number the strings and do not add markdown or commentary outside JSON.

Write the JSON now."""


def extract_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no complete JSON object")
    if text[:start].strip() or text[end + 1 :].strip():
        raise ValueError("commentary outside JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("top-level output is not an object")
    return value


def validate_batch(
    value: dict[str, Any], concept: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if set(value) != {"train", "test"}:
        errors.append(f"keys={sorted(value)}")
    train = value.get("train", [])
    test = value.get("test", [])
    if not isinstance(train, list) or len(train) != 4:
        errors.append("train_must_have_4")
    if not isinstance(test, list) or len(test) != 2:
        errors.append("test_must_have_2")
    documents = (
        list(train) + list(test)
        if isinstance(train, list) and isinstance(test, list)
        else []
    )
    if any(not isinstance(item, str) for item in documents):
        errors.append("all_documents_must_be_strings")
        documents = [item for item in documents if isinstance(item, str)]

    word_counts = [len(words(item)) for item in documents]
    for index, count in enumerate(word_counts):
        if not WORD_MIN <= count <= WORD_MAX:
            errors.append(f"word_count_{index}={count}")
    normalized = [normalized_text(item) for item in documents]
    if len(set(normalized)) != len(normalized):
        errors.append("duplicate_documents")

    forbidden_literals = {
        concept["id"].lower(),
        concept["superdomain"].lower(),
    }
    for index, item in enumerate(documents):
        lowered = item.lower()
        for literal in forbidden_literals:
            if literal in lowered:
                errors.append(f"forbidden_literal_{index}={literal}")
        if any(pattern.search(item) for pattern in META_PATTERNS):
            errors.append(f"meta_language_{index}")

    similarities = [
        {
            "left": left,
            "right": right,
            "unigram_jaccard": jaccard(
                documents[left], documents[right]
            ),
            "word_5gram_jaccard": ngram_jaccard(
                documents[left], documents[right], 5
            ),
        }
        for left, right in combinations(range(len(documents)), 2)
    ]
    train_test_similarities = [
        row
        for row in similarities
        if (row["left"] < 4) != (row["right"] < 4)
    ]
    for row in train_test_similarities:
        if row["word_5gram_jaccard"] >= MAX_TRAIN_TEST_5GRAM_JACCARD:
            errors.append(
                f"train_test_5gram_jaccard_{row['left']}_{row['right']}="
                f"{row['word_5gram_jaccard']:.4f}"
            )

    title = normalized_text(concept["title"])
    title_occurrences = [
        title in normalized_text(item) for item in documents
    ]
    if sum(title_occurrences[:4]) > 1:
        errors.append(
            f"exact_title_train_occurrences={sum(title_occurrences[:4])}"
        )
    if any(title_occurrences[4:]):
        errors.append(
            f"exact_title_test_occurrences={sum(title_occurrences[4:])}"
        )
    for index, item in enumerate(documents):
        if re.search(r"https?://|www\.", item, flags=re.IGNORECASE):
            errors.append(f"url_{index}")
        if re.search(r"\b\d{4}\b", item):
            errors.append(f"four_digit_year_{index}")
        if (
            "```" in item
            or re.search(r"(?m)^\s*[-*#]\s+", item)
            or re.search(r"(?m)^\s*\d+[.)]\s+", item)
            or re.search(r"(?m)^\s*\|.+\|\s*$", item)
            or re.search(r"\$\$|\\begin\{|\\\[|\\\]", item)
        ):
            errors.append(f"non_prose_format_{index}")
    diagnostics = {
        "word_counts": word_counts,
        "exact_title_occurrences": title_occurrences,
        "max_pairwise_unigram_jaccard": (
            max(
                (row["unigram_jaccard"] for row in similarities),
                default=0.0,
            )
        ),
        "max_train_test_word_5gram_jaccard": (
            max(
                (
                    row["word_5gram_jaccard"]
                    for row in train_test_similarities
                ),
                default=0.0,
            )
        ),
        "pairwise_jaccard": similarities,
    }
    return errors, diagnostics


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"checkpoint line {line_number} is not valid JSON"
            ) from exc
    return rows


def assess_raw(
    raw: str, concept: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    try:
        parsed = extract_object(raw)
        errors, diagnostics = validate_batch(parsed, concept)
        return parsed, errors, diagnostics
    except (json.JSONDecodeError, ValueError) as exc:
        return (
            None,
            [f"parse_error={type(exc).__name__}:{exc}"],
            {},
        )


def verify_frozen_file(
    freeze: dict[str, Any], role: str, path: Path
) -> None:
    rows = [
        row for row in freeze["files"] if row.get("role") == role
    ]
    if len(rows) != 1:
        raise ValueError(f"stage0 freeze must contain one role={role}")
    actual = sha256_file(path)
    if actual != rows[0]["sha256"]:
        raise ValueError(
            f"stage0 hash mismatch role={role}: "
            f"expected={rows[0]['sha256']} actual={actual}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--stage0-freeze", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--out-discovery-manifest", required=True, type=Path)
    parser.add_argument("--out-heldout-manifest", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    stage0_freeze = json.loads(
        args.stage0_freeze.read_text(encoding="utf-8")
    )
    verify_frozen_file(stage0_freeze, "concept_spec", args.spec)
    verify_frozen_file(
        stage0_freeze, "preregistration", args.preregistration
    )
    verify_frozen_file(
        stage0_freeze, "corpus_generator", Path(__file__).resolve()
    )
    if spec["experiment"] != EXPERIMENT:
        raise ValueError("spec experiment differs from frozen generator")
    if spec["corpus_generation"]["max_attempts"] != MAX_ATTEMPTS:
        raise ValueError("spec max_attempts differs from frozen generator")
    if spec["corpus_generation"]["required_prompt_words"] != [
        WORD_MIN,
        WORD_MAX,
    ]:
        raise ValueError("spec required word range differs from frozen generator")
    if spec["corpus_generation"]["target_prompt_words"] != [
        TARGET_WORD_MIN,
        TARGET_WORD_MAX,
    ]:
        raise ValueError("spec target word range differs from frozen generator")
    if spec["corpus_generation"]["seed"] != MASTER_SEED:
        raise ValueError("spec master seed differs from frozen generator")
    if spec["corpus_generation"]["temperature"] != TEMPERATURE:
        raise ValueError("spec temperature differs from frozen generator")
    if spec["corpus_generation"]["top_p"] != TOP_P:
        raise ValueError("spec top_p differs from frozen generator")
    if spec["corpus_generation"]["top_k"] != TOP_K:
        raise ValueError("spec top_k differs from frozen generator")
    if (
        spec["corpus_generation"]["repetition_penalty"]
        != REPETITION_PENALTY
    ):
        raise ValueError(
            "spec repetition_penalty differs from frozen generator"
        )
    if spec["corpus_generation"]["max_new_tokens"] != MAX_NEW_TOKENS:
        raise ValueError("spec max_new_tokens differs from frozen generator")
    if spec["documents_per_concept"] != {"train": 4, "test": 2}:
        raise ValueError("spec document counts differ from frozen generator")
    concepts = spec["concepts"]
    if len(concepts) != 24:
        raise ValueError("frozen cohort requires exactly 24 concepts")
    concept_by_id = {concept["id"]: concept for concept in concepts}
    if len(concept_by_id) != len(concepts):
        raise ValueError("concept IDs must be unique")
    for concept in concepts:
        negative = concept_by_id.get(concept["hard_negative_id"])
        if (
            negative is None
            or negative["hard_negative_id"] != concept["id"]
            or negative["superdomain"] != concept["superdomain"]
        ):
            raise ValueError(
                f"hard negative must be reciprocal and within superdomain: "
                f"{concept['id']}"
            )

    prompt_hashes = {
        concept["id"]: sha256_bytes(
            generation_prompt(concept).encode("utf-8")
        )
        for concept in concepts
    }
    checkpoint_rows = load_checkpoint(args.checkpoint)
    accepted: dict[str, dict[str, Any]] = {}
    attempted: dict[str, set[int]] = {}
    checkpoint_by_concept: dict[str, list[dict[str, Any]]] = {}
    for row in checkpoint_rows:
        concept_id = row.get("concept_id")
        if concept_id not in prompt_hashes:
            raise ValueError(f"checkpoint has unknown concept {concept_id}")
        if row.get("prompt_sha256") != prompt_hashes[concept_id]:
            raise ValueError(f"checkpoint prompt hash mismatch for {concept_id}")
        concept_index = next(
            index
            for index, concept in enumerate(concepts)
            if concept["id"] == concept_id
        )
        attempt = int(row.get("attempt", -1))
        expected_seed = MASTER_SEED + 100 * concept_index + attempt
        if (
            row.get("schema_version") != 1
            or row.get("concept_index") != concept_index
            or attempt not in range(MAX_ATTEMPTS)
            or row.get("seed") != expected_seed
            or not isinstance(row.get("seconds"), (int, float))
            or not math.isfinite(float(row["seconds"]))
            or float(row["seconds"]) < 0
        ):
            raise ValueError(
                f"checkpoint deterministic metadata mismatch for "
                f"{concept_id} attempt={attempt}"
            )
        if attempt in attempted.setdefault(concept_id, set()):
            raise ValueError(
                f"duplicate checkpoint attempt {concept_id}:{attempt}"
            )
        raw = row.get("raw")
        if (
            not isinstance(raw, str)
            or row.get("raw_sha256")
            != sha256_bytes(raw.encode("utf-8"))
        ):
            raise ValueError(
                f"checkpoint raw hash mismatch for {concept_id}:{attempt}"
            )
        parsed, errors, diagnostics = assess_raw(
            raw, concept_by_id[concept_id]
        )
        if (
            canonical_json(row.get("parsed")) != canonical_json(parsed)
            or row.get("errors") != errors
            or canonical_json(row.get("diagnostics"))
            != canonical_json(diagnostics)
            or bool(row.get("admissible")) != (not errors)
        ):
            raise ValueError(
                f"checkpoint validation mismatch for "
                f"{concept_id}:{attempt}"
            )
        attempted[concept_id].add(attempt)
        checkpoint_by_concept.setdefault(concept_id, []).append(row)
        if row.get("admissible") and concept_id not in accepted:
            accepted[concept_id] = row
    for concept_id, rows in checkpoint_by_concept.items():
        attempts = [int(row["attempt"]) for row in rows]
        if attempts != list(range(len(attempts))):
            raise ValueError(
                f"checkpoint attempts are not contiguous for {concept_id}: "
                f"{attempts}"
            )
        first_admissible = next(
            (
                index
                for index, row in enumerate(rows)
                if row["admissible"]
            ),
            None,
        )
        if (
            first_admissible is not None
            and first_admissible != len(rows) - 1
        ):
            raise ValueError(
                f"checkpoint continues after acceptance for {concept_id}"
            )

    pending = [concept for concept in concepts if concept["id"] not in accepted]
    print(
        f"[plan] concepts={len(concepts)} resumed={len(accepted)} "
        f"pending={len(pending)}"
    )
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        device = next(model.parameters()).device

        for concept_index, concept in enumerate(concepts):
            concept_id = concept["id"]
            if concept_id in accepted:
                continue
            prompt = generation_prompt(concept)
            for attempt in range(MAX_ATTEMPTS):
                if attempt in attempted.get(concept_id, set()):
                    continue
                seed = MASTER_SEED + 100 * concept_index + attempt
                set_seed(seed)
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if not torch.is_tensor(inputs):
                    inputs = inputs["input_ids"]
                inputs = inputs.to(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=inputs,
                        attention_mask=torch.ones_like(inputs),
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        repetition_penalty=REPETITION_PENALTY,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                raw = tokenizer.decode(
                    output[0, inputs.shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                parsed, errors, diagnostics = assess_raw(raw, concept)
                row = {
                    "schema_version": 1,
                    "concept_id": concept_id,
                    "concept_index": concept_index,
                    "attempt": attempt,
                    "seed": seed,
                    "prompt_sha256": prompt_hashes[concept_id],
                    "raw_sha256": sha256_bytes(raw.encode("utf-8")),
                    "seconds": seconds,
                    "admissible": not errors,
                    "errors": errors,
                    "diagnostics": diagnostics,
                    "parsed": parsed,
                    "raw": raw,
                }
                append_jsonl(args.checkpoint, row)
                attempted.setdefault(concept_id, set()).add(attempt)
                print(
                    f"[{concept_index + 1:02d}/24 {concept_id} "
                    f"attempt={attempt}] admissible={not errors} "
                    f"seconds={seconds:.1f} errors={errors[:2]}"
                )
                if not errors:
                    accepted[concept_id] = row
                    break
            if concept_id not in accepted:
                raise RuntimeError(
                    f"{concept_id} has no admissible batch after "
                    f"{MAX_ATTEMPTS} fixed attempts"
                )

    manifest: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for concept in concepts:
        row = accepted[concept["id"]]
        parsed = row["parsed"]
        errors, diagnostics = validate_batch(parsed, concept)
        if errors:
            raise ValueError(
                f"accepted checkpoint row became invalid for "
                f"{concept['id']}: {errors}"
            )
        quality.append(
            {
                "concept_id": concept["id"],
                "attempt": int(row["attempt"]),
                "seed": int(row["seed"]),
                **diagnostics,
            }
        )
        for split in ("train", "test"):
            for ordinal, text in enumerate(parsed[split]):
                manifest.append(
                    {
                        "id": (
                            f"c1c_{concept['id']}_{split}_{ordinal:02d}"
                        ),
                        "axis_domain": concept["superdomain"],
                        "axis_language": spec["language"],
                        "split": split,
                        "topic": concept["id"],
                        "text": text.strip(),
                        "concept_title": concept["title"],
                        "concept_summary": concept["summary"],
                        "hard_negative_id": concept["hard_negative_id"],
                        "generation_attempt": int(row["attempt"]),
                        "generation_seed": int(row["seed"]),
                    }
                )

    if len(manifest) != 144:
        raise ValueError(f"expected 144 manifest rows, got {len(manifest)}")
    ids = [row["id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest IDs are not unique")
    def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
            newline="\n",
        )

    discovery_manifest = [
        row for row in manifest if row["split"] == "train"
    ]
    heldout_manifest = [
        row for row in manifest if row["split"] == "test"
    ]
    write_manifest(args.out_manifest, manifest)
    write_manifest(args.out_discovery_manifest, discovery_manifest)
    write_manifest(args.out_heldout_manifest, heldout_manifest)
    report = {
        "schema_version": 1,
        "experiment": spec["experiment"],
        "status": "synthetic_corpus_frozen_before_activation_extraction",
        "inputs": {
            "base_model": str(args.base_model),
            "base_model_files": {
                filename: sha256_file(args.base_model / filename)
                for filename in (
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
                if (args.base_model / filename).exists()
            },
            "spec": str(args.spec),
            "spec_sha256": sha256_file(args.spec),
            "preregistration": str(args.preregistration),
            "preregistration_sha256": sha256_file(
                args.preregistration
            ),
            "stage0_freeze": str(args.stage0_freeze),
            "stage0_freeze_sha256": sha256_file(args.stage0_freeze),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        },
        "protocol": {
            "master_seed": MASTER_SEED,
            "max_attempts": MAX_ATTEMPTS,
            "attempt_seed": "master + 100 * concept_index + attempt",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "max_new_tokens": MAX_NEW_TOKENS,
            "word_range_inclusive": [WORD_MIN, WORD_MAX],
            "target_word_range": [TARGET_WORD_MIN, TARGET_WORD_MAX],
            "max_train_test_word_5gram_jaccard": (
                MAX_TRAIN_TEST_5GRAM_JACCARD
            ),
        },
        "counts": {
            "concepts": len(concepts),
            "superdomains": len({row["superdomain"] for row in concepts}),
            "documents": len(manifest),
            "train_documents": sum(row["split"] == "train" for row in manifest),
            "test_documents": sum(row["split"] == "test" for row in manifest),
        },
        "quality": quality,
    }
    report["outputs"] = {
        "manifest": str(args.out_manifest),
        "manifest_sha256": sha256_file(args.out_manifest),
        "discovery_manifest": str(args.out_discovery_manifest),
        "discovery_manifest_sha256": sha256_file(
            args.out_discovery_manifest
        ),
        "heldout_manifest": str(args.out_heldout_manifest),
        "heldout_manifest_sha256": sha256_file(args.out_heldout_manifest),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("C1_CONFIRMATORY_CORPUS_COMPLETE")
    print(
        f"wrote {len(manifest)} documents -> {args.out_manifest} "
        f"sha256={report['outputs']['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()

