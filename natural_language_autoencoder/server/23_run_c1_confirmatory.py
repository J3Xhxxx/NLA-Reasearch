#!/usr/bin/env python3
"""Run C1-confirmatory held-out scoring, generated candidates, and NLA AR.

The input benchmark must already be frozen.  Only then does this stage encode
held-out residual activations for the already selected feature IDs.  It also
generates one greedy NLA AV explanation and one discovery-context-only plain
base-model autointerpretation per feature, reconstructs every unique candidate
text exactly once with NLA AR, and scores all pairings in the discovery-centered
space.  No automatic semantic judge is used.

Generation rows are append-checkpointed with hashes of their complete inputs.
The script emits both a source-hidden randomized human-rating packet and a
separate key that should be withheld until ratings are frozen.  It never issues
shutdown, poweroff, instance-stop, or auto-release commands.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot_common import AVLocal, EXPLANATION_RE, NLACritic


BENCHMARK_STATUS = "benchmark_frozen_before_C1_AV_AR_and_heldout"
MODEL_FREEZE_STATUS = "models_frozen_before_C1_AV_AR"
CORPUS_STATUS = "synthetic_corpus_frozen_before_activation_extraction"
LABEL_RE = re.compile(r"<label>\s*(.*?)\s*</label>", re.I | re.S)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def input_hash(value: dict[str, Any]) -> str:
    return sha256_text(canonical_json(value))


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def static_candidate_id(
    feature: int,
    kind: str,
    ordinal: int,
    candidate_concept_id: str,
    text_sha256: str,
) -> str:
    payload = {
        "experiment": "C1-confirmatory-v1",
        "feature": int(feature),
        "kind": str(kind),
        "ordinal": int(ordinal),
        "candidate_concept_id": str(candidate_concept_id),
        "text_sha256": require_sha256(
            text_sha256, "static candidate text_sha256"
        ),
    }
    return sha256_text(canonical_json(payload))[:20]


def generated_candidate_id(feature: int, kind: str) -> str:
    payload = {
        "experiment": "C1-confirmatory-v1",
        "feature": int(feature),
        "kind": str(kind),
        "ordinal": 0,
        "candidate_concept_id": "__generated__",
    }
    return sha256_text(canonical_json(payload))[:20]


def validate_frozen_candidate_mapping(records: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    allowed_static = {
        "correct_reference",
        "hard_negative_reference",
        "other_within_superdomain_reference",
    }
    for record in records:
        feature = int(record["feature"])
        concept_id = str(record["concept_id"])
        for candidate in record.get("static_candidates", []):
            kind = str(candidate.get("kind"))
            if kind not in allowed_static or candidate.get("generated") is not False:
                raise ValueError(f"invalid frozen static candidate kind for f{feature}")
            text = str(candidate.get("text", ""))
            if text != normalize_text(text) or not text:
                raise ValueError(
                    f"frozen candidate text is not normalized for f{feature}"
                )
            text_sha = require_sha256(
                candidate.get("text_sha256"),
                f"frozen candidate f{feature} text_sha256",
            )
            if text_sha != sha256_text(normalize_text(text)):
                raise ValueError(f"frozen candidate text SHA drift for f{feature}")
            candidate_concept = str(candidate.get("candidate_concept_id", ""))
            expected_id = static_candidate_id(
                feature,
                kind,
                int(candidate.get("ordinal", -1)),
                candidate_concept,
                text_sha,
            )
            if str(candidate.get("candidate_id")) != expected_id:
                raise ValueError(f"frozen static candidate ID drift for f{feature}")
            if str(candidate.get("assigned_concept_id")) != concept_id:
                raise ValueError(
                    f"frozen static candidate assigned concept drift for f{feature}"
                )
            if expected_id in ids:
                raise ValueError(f"duplicate frozen candidate ID {expected_id}")
            ids.add(expected_id)
        requests = record.get("generation_requests")
        if not isinstance(requests, list) or {
            str(row.get("kind")) for row in requests
        } != {"nla_av", "base_autointerp"}:
            raise ValueError(f"invalid frozen generation requests for f{feature}")
        for request in requests:
            kind = str(request["kind"])
            expected_id = generated_candidate_id(feature, kind)
            if str(request.get("candidate_id")) != expected_id:
                raise ValueError(f"frozen generated request ID drift for f{feature}")
            if expected_id in ids:
                raise ValueError(f"duplicate frozen candidate ID {expected_id}")
            ids.add(expected_id)


def require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} is not a lowercase SHA256 digest")
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def npz_scalar_text(archive: Any, key: str) -> str:
    if key not in archive.files:
        raise ValueError(f"vectors asset is missing embedded {key}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"vectors embedded {key} is not scalar")
    item = value.reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def relevant_model_files(directory: Path, role: str) -> set[str]:
    """Return the exact files which the frozen model manifest must cover."""
    if role == "sae":
        required = {"params.safetensors", "config.json"}
        missing = [name for name in sorted(required) if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"SAE directory is missing required files {missing}: {directory}"
            )
        return required

    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = [name for name in sorted(required) if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{role} model directory is missing identity files {missing}: "
            f"{directory}"
        )
    names = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.safetensors")
        if path.is_file()
    }
    if not names:
        raise ValueError(f"{role} model has no safetensor weights: {directory}")
    names.update(required)
    for optional in ("generation_config.json", "nla_meta.yaml"):
        if (directory / optional).is_file():
            names.add(optional)
    return names


def verify_model_freeze(
    freeze_path: Path,
    actual_directories: dict[str, Path],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Verify every model byte used by this run against the frozen manifest."""
    freeze = read_json(freeze_path, "model freeze")
    if freeze.get("schema_version") != 1:
        raise ValueError("model freeze schema_version must be 1")
    if freeze.get("status") != MODEL_FREEZE_STATUS:
        raise ValueError(
            f"model freeze status must be {MODEL_FREEZE_STATUS!r}"
        )
    models = freeze.get("models")
    if not isinstance(models, dict) or set(models) != {"base", "av", "ar", "sae"}:
        raise ValueError("model freeze must contain exactly base/av/ar/sae")

    verified: dict[str, dict[str, Any]] = {}
    for role in ("base", "av", "ar", "sae"):
        entry = models.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"model freeze entry {role} is not an object")
        actual_directory = actual_directories[role].resolve()
        frozen_directory = Path(str(entry.get("path", ""))).resolve()
        if actual_directory != frozen_directory:
            raise ValueError(
                f"{role} path differs from model freeze: "
                f"actual={actual_directory} frozen={frozen_directory}"
            )
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"model freeze {role}.files is empty")
        names = [str(row.get("name", "")) for row in files if isinstance(row, dict)]
        if len(names) != len(files) or names != sorted(names) or len(names) != len(
            set(names)
        ):
            raise ValueError(
                f"model freeze {role}.files must have unique sorted names"
            )
        relevant = relevant_model_files(actual_directory, role)
        frozen_names = set(names)
        if role == "sae":
            if not relevant.issubset(frozen_names):
                raise ValueError(
                    f"model freeze SAE lacks {sorted(relevant-frozen_names)}"
                )
        elif frozen_names != relevant:
            raise ValueError(
                f"model freeze {role} coverage differs from actual relevant files "
                f"(missing={sorted(relevant-frozen_names)}, "
                f"extra={sorted(frozen_names-relevant)})"
            )

        canonical_files: list[dict[str, Any]] = []
        for row, name in zip(files, names):
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"unsafe frozen model filename {role}:{name!r}")
            expected_bytes = int(row.get("bytes", -1))
            expected_sha = require_sha256(
                row.get("sha256"), f"model freeze {role}:{name}"
            )
            path = actual_directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_bytes = int(path.stat().st_size)
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"model size mismatch {role}:{name}: "
                    f"{actual_bytes} != {expected_bytes}"
                )
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"model SHA256 mismatch {role}:{name}: "
                    f"{actual_sha} != {expected_sha}"
                )
            canonical_files.append(
                {"name": name, "bytes": expected_bytes, "sha256": expected_sha}
            )
        fingerprint = sha256_text(canonical_json(canonical_files))
        recorded_fingerprint = require_sha256(
            entry.get("fingerprint_sha256"),
            f"model freeze {role}.fingerprint_sha256",
        )
        if fingerprint != recorded_fingerprint:
            raise ValueError(
                f"model freeze fingerprint mismatch for {role}: "
                f"{fingerprint} != {recorded_fingerprint}"
            )
        verified[role] = {
            "path": str(actual_directory),
            "fingerprint_sha256": fingerprint,
            "n_verified_files": len(canonical_files),
            "files": canonical_files,
        }
    return verified, sha256_file(freeze_path)


def validate_benchmark_input_file(
    benchmark: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    entry = benchmark.get("inputs", {}).get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"benchmark lacks frozen inputs.{key}")
    path = Path(str(entry.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = require_sha256(
        entry.get("sha256"), f"benchmark inputs.{key}.sha256"
    )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"benchmark input {key} SHA256 mismatch: {actual_sha} != {expected_sha}"
        )
    return path, entry


def verify_heldout_provenance(
    provenance_path: Path,
    heldout_path: Path,
    benchmark: dict[str, Any],
    base_model_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    """Bind the embargoed held-out parquet to the frozen 48-document corpus."""
    provenance = read_json(provenance_path, "held-out provenance")
    if provenance.get("schema_version") != 1:
        raise ValueError("held-out provenance schema_version must be 1")
    if provenance.get("status") != "PASS":
        raise ValueError("held-out provenance status must be PASS")
    parameters = provenance.get("parameters", {})
    counts = provenance.get("counts", {})
    hashes = provenance.get("hashes", {})
    if (
        parameters.get("expected_split") != "test"
        or int(parameters.get("expected_documents", -1)) != 48
        or int(parameters.get("layer_index", -1)) != 32
        or int(parameters.get("min_position", -1)) != 50
        or int(parameters.get("max_per_prompt", -1)) != 0
        or parameters.get("dtype") != "bfloat16"
        or parameters.get("parquet_extraction_metadata_schema_version") != 1
        or parameters.get("verified_against_parquet_schema_metadata") is not True
    ):
        raise ValueError("held-out provenance parameters differ from protocol")
    if (
        int(counts.get("manifest_documents", -1)) != 48
        or int(counts.get("parquet_documents", -1)) != 48
        or int(counts.get("activation_rows", 0)) <= 0
        or int(counts.get("d_model", -1)) <= 0
    ):
        raise ValueError("held-out provenance has invalid corpus/parquet counts")
    activation_sha = sha256_file(heldout_path)
    heldout_extractor_sha = require_sha256(
        hashes.get("extractor_sha256"),
        "held-out provenance extractor_sha256",
    )
    discovery_extractor_sha = require_sha256(
        benchmark.get("inputs", {})
        .get("discovery_provenance", {})
        .get("extractor_sha256"),
        "benchmark discovery_provenance.extractor_sha256",
    )
    if heldout_extractor_sha != discovery_extractor_sha:
        raise ValueError(
            "held-out and discovery activations were produced by different "
            "extractor bytes"
        )
    if activation_sha != require_sha256(
        hashes.get("activations_sha256"),
        "held-out provenance activations_sha256",
    ):
        raise ValueError("held-out parquet SHA256 differs from provenance")
    manifest_sha = require_sha256(
        hashes.get("manifest_sha256"),
        "held-out provenance manifest_sha256",
    )
    if require_sha256(
        hashes.get("base_model_identity_sha256"),
        "held-out provenance base_model_identity_sha256",
    ) != base_model_fingerprint:
        raise ValueError(
            "held-out extraction base-model identity differs from model freeze"
        )

    documents = provenance.get("documents")
    if not isinstance(documents, list) or len(documents) != 48:
        raise ValueError("held-out provenance must contain 48 document records")
    doc_ids: set[int] = set()
    prompt_ids: set[str] = set()
    document_activation_rows = 0
    for index, row in enumerate(documents):
        if not isinstance(row, dict) or str(row.get("split")) != "test":
            raise ValueError(f"held-out provenance document {index} is not test")
        doc_id = int(row.get("doc_id", -1))
        prompt_id = str(row.get("prompt_id", ""))
        require_sha256(
            row.get("prompt_sha256"),
            f"held-out provenance document {index}.prompt_sha256",
        )
        if doc_id in doc_ids or not prompt_id or prompt_id in prompt_ids:
            raise ValueError("held-out provenance document IDs are not unique")
        if int(row.get("activation_rows", 0)) <= 0:
            raise ValueError(f"held-out provenance document {index} has no rows")
        if (
            int(row.get("first_position", -1)) < 50
            or int(row.get("last_position", -1))
            < int(row.get("first_position", -1))
            or not str(row.get("topic", ""))
            or str(row.get("axis_language", "")) != "en"
        ):
            raise ValueError(
                f"held-out provenance document {index} metadata is invalid"
            )
        document_activation_rows += int(row["activation_rows"])
        doc_ids.add(doc_id)
        prompt_ids.add(prompt_id)
    if document_activation_rows != int(counts["activation_rows"]):
        raise ValueError(
            "held-out provenance document row counts do not sum to total"
        )

    corpus_report_path, corpus_input = validate_benchmark_input_file(
        benchmark, "corpus_report"
    )
    corpus_report = read_json(corpus_report_path, "frozen corpus report")
    if corpus_report.get("status") != CORPUS_STATUS:
        raise ValueError("frozen corpus report status is invalid")
    corpus_counts = corpus_report.get("counts", {})
    if (
        int(corpus_counts.get("documents", -1)) != 144
        or int(corpus_counts.get("train_documents", -1)) != 96
        or int(corpus_counts.get("test_documents", -1)) != 48
    ):
        raise ValueError("frozen corpus report document counts are invalid")
    corpus_outputs = corpus_report.get("outputs", {})
    report_heldout_manifest_sha = require_sha256(
        corpus_outputs.get("heldout_manifest_sha256"),
        "corpus report heldout_manifest_sha256",
    )
    if report_heldout_manifest_sha != manifest_sha:
        raise ValueError(
            "held-out provenance manifest is not the frozen corpus held-out split"
        )
    if require_sha256(
        corpus_input.get("heldout_manifest_sha256"),
        "benchmark corpus_report.heldout_manifest_sha256",
    ) != manifest_sha:
        raise ValueError("benchmark and provenance held-out manifest hashes differ")

    audit = provenance.get("manual_audit")
    benchmark_audit = benchmark.get("inputs", {}).get("manual_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("status") != "PASS"
        or not isinstance(benchmark_audit, dict)
        or benchmark_audit.get("status") != "PASS"
    ):
        raise ValueError("held-out provenance/manual-audit PASS binding is absent")
    audit_manifest_sha = require_sha256(
        audit.get("manifest_sha256"),
        "held-out provenance manual_audit.manifest_sha256",
    )
    if audit_manifest_sha != manifest_sha:
        raise ValueError("manual audit is not bound to the frozen held-out split")
    _, audit_input = validate_benchmark_input_file(benchmark, "manual_audit")
    if require_sha256(
        hashes.get("manual_audit_sha256"),
        "held-out provenance manual_audit_sha256",
    ) != require_sha256(
        audit_input.get("sha256"), "benchmark manual_audit.sha256"
    ):
        raise ValueError("held-out provenance uses a different manual audit")
    if require_sha256(
        audit_input.get("heldout_manifest_sha256"),
        "benchmark manual_audit.heldout_manifest_sha256",
    ) != manifest_sha:
        raise ValueError("benchmark manual audit is not bound to held-out manifest")

    return {
        "status": "PASS",
        "path": str(provenance_path),
        "sha256": sha256_file(provenance_path),
        "heldout_activations_sha256": activation_sha,
        "heldout_manifest_sha256": manifest_sha,
        "corpus_report_sha256": require_sha256(
            corpus_input.get("sha256"), "benchmark corpus_report SHA256"
        ),
        "documents": 48,
        "activation_rows": int(counts["activation_rows"]),
        "d_model": int(counts["d_model"]),
        "extractor_sha256": heldout_extractor_sha,
        "base_model_fingerprint_sha256": base_model_fingerprint,
    }, activation_sha


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("attempted to serialize a non-finite float")
        return result
    if isinstance(value, np.integer):
        return int(value)
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                to_builtin(row),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["key"])
        if key in rows:
            raise ValueError(f"duplicate checkpoint key {key} line {line_number}")
        if not isinstance(row.get("input_sha256"), str):
            raise ValueError(f"checkpoint row {key} has no input_sha256")
        rows[key] = row
    return rows


def clean_generated(text: str) -> str:
    return " ".join(str(text).replace("<end_of_turn>", " ").strip().split())


def extract_label(text: str) -> tuple[str, bool]:
    match = LABEL_RE.search(text)
    if match:
        return clean_generated(match.group(1)), True
    cleaned = clean_generated(text)
    for prefix in ("LABEL:", "Label:", "label:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned, False


def format_discovery_contexts(contexts: list[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(contexts, start=1):
        activation = float(row.get("document_score", row.get("activation", 0.0)))
        blocks.append(
            f"[Context {index}; relative activation evidence={activation:.3f}]\n"
            f"{row['prompt']}"
        )
    return "\n\n".join(blocks)


def autointerp_prompt(record: dict[str, Any]) -> str:
    contexts = format_discovery_contexts(record["discovery_contexts"])
    return f"""You are labeling one sparse-autoencoder feature using only the
discovery contexts in which it is highly active.  Write one concise English
description of at most 30 words.  State the recurring semantic concept and
avoid unsupported named entities, product names, feature IDs, activation
values, causal claims, or guesses presented as alternatives.

{contexts}

Return exactly:
<label>one concise description</label>"""


@torch.inference_mode()
def base_generate(
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
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    raw = tokenizer.decode(
        output[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    return raw, seconds


@torch.inference_mode()
def av_generate(
    av: AVLocal,
    vector: np.ndarray,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, str, bool, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    embeddings, _ = av.client._build_embeds(
        torch.as_tensor(np.asarray(vector, np.float32)), None
    )
    inputs = torch.from_numpy(embeddings)[None].to(av.device, av.model.dtype)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=av.device)
    started = time.perf_counter()
    output = av.model.generate(
        inputs_embeds=inputs,
        attention_mask=attention,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=av.tok.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    raw = av.tok.decode(output[0], skip_special_tokens=False)
    match = EXPLANATION_RE.search(raw)
    explanation = clean_generated(match.group(1) if match else raw)
    return raw, explanation, bool(match), seconds


def project_rows(values: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    m_hat = np.asarray(m_hat, dtype=np.float64)
    return values - np.outer(values @ m_hat, m_hat)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("cannot normalize non-finite or zero projected row")
    return values / norms


def average_rank(scores: np.ndarray, target_index: int) -> float:
    target = float(scores[target_index])
    greater = int(np.sum(scores > target))
    tied_others = int(np.sum(scores == target)) - 1
    return 1.0 + greater + 0.5 * max(0, tied_others)


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUROC requires both positive and negative documents")
    greater = positive[:, None] > negative[None, :]
    equal = positive[:, None] == negative[None, :]
    return float(greater.mean() + 0.5 * equal.mean())


def document_rows(table: Any) -> tuple[np.ndarray, dict[str, list[Any]]]:
    required = {
        "activation_vector",
        "token",
        "position",
        "doc_id",
        "prompt_id",
        "axis_domain",
        "axis_language",
        "split",
        "topic",
        "prompt",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"held-out parquet missing columns {sorted(missing)}")
    x = np.asarray(
        table.column("activation_vector").combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    metadata = {
        column: table.column(column).to_pylist()
        for column in required
        if column != "activation_vector"
    }
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"invalid held-out activation matrix shape={x.shape}")
    if set(str(value) for value in metadata["split"]) != {"test"}:
        raise ValueError("held-out parquet must contain only split='test'")
    return x, metadata


def encode_selected_features(
    x: np.ndarray,
    feature_ids: np.ndarray,
    sae_dir: Path,
    batch_size: int,
) -> np.ndarray:
    params_path = sae_dir / "params.safetensors"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    params = load_file(str(params_path))
    width = int(params["w_enc"].shape[1])
    if np.any(feature_ids < 0) or np.any(feature_ids >= width):
        raise ValueError("selected feature ID falls outside SAE width")
    selected = torch.as_tensor(feature_ids, dtype=torch.long)
    w_enc = params["w_enc"].index_select(1, selected).to(
        "cuda", dtype=torch.float32
    )
    b_enc = params["b_enc"].index_select(0, selected).to(
        "cuda", dtype=torch.float32
    )
    threshold = params["threshold"].index_select(0, selected).to(
        "cuda", dtype=torch.float32
    )
    del params, selected
    output = np.empty((len(x), len(feature_ids)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            batch = torch.from_numpy(x[start:stop]).to("cuda", dtype=torch.float32)
            pre = batch @ w_enc + b_enc
            acts = torch.relu(pre) * (pre > threshold)
            output[start:stop] = acts.cpu().numpy()
            del batch, pre, acts
    del w_enc, b_enc, threshold
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return output


def heldout_contexts_and_metrics(
    heldout_path: Path,
    sae_dir: Path,
    records: list[dict[str, Any]],
    max_contexts: int,
    batch_size: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    table = pq.read_table(heldout_path)
    x, meta = document_rows(table)
    feature_ids = np.asarray([int(row["feature"]) for row in records], dtype=np.int64)
    acts = encode_selected_features(x, feature_ids, sae_dir, batch_size)
    row_doc_ids = np.asarray(meta["doc_id"], dtype=np.int64)
    doc_ids = np.unique(row_doc_ids)
    doc_topics: list[str] = []
    doc_scores = np.zeros((len(doc_ids), len(records)), dtype=np.float64)
    doc_fires = np.zeros_like(doc_scores, dtype=bool)
    doc_index_by_id: dict[int, int] = {}
    for doc_index, doc_id in enumerate(doc_ids.tolist()):
        doc_index_by_id[int(doc_id)] = doc_index
        indices = np.flatnonzero(row_doc_ids == doc_id)
        topics = {str(meta["topic"][index]) for index in indices}
        if len(topics) != 1:
            raise ValueError(f"held-out doc {doc_id} has inconsistent topic")
        doc_topics.append(next(iter(topics)))
        doc_acts = acts[indices]
        take = min(3, len(indices))
        partitioned = np.partition(doc_acts, len(indices) - take, axis=0)
        doc_scores[doc_index] = partitioned[-take:].mean(axis=0)
        doc_fires[doc_index] = doc_acts.max(axis=0) > 0.0

    output: dict[int, dict[str, Any]] = {}
    for feature_position, record in enumerate(records):
        feature = int(record["feature"])
        concept_id = str(record["concept_id"])
        positive = np.asarray(
            [topic == concept_id for topic in doc_topics],
            dtype=bool,
        )
        if int(positive.sum()) != 2:
            raise ValueError(
                f"expected two held-out documents for {concept_id}, "
                f"got {int(positive.sum())}"
            )
        scores = doc_scores[:, feature_position]
        fires = doc_fires[:, feature_position]
        pos_scores = scores[positive]
        neg_scores = scores[~positive]
        pos_fires = fires[positive]
        neg_fires = fires[~positive]
        standard_deviation = float(np.std(scores))
        pos_sum = float(np.sum(pos_scores))
        metrics = {
            "auc": binary_auc(pos_scores, neg_scores),
            "pos_mean": float(np.mean(pos_scores)),
            "neg_mean": float(np.mean(neg_scores)),
            "effect": float(
                (np.mean(pos_scores) - np.mean(neg_scores))
                / max(standard_deviation, 1e-6)
            ),
            "raw_difference": float(np.mean(pos_scores) - np.mean(neg_scores)),
            "pos_support": int(pos_fires.sum()),
            "neg_support": int(neg_fires.sum()),
            "support_precision": float(
                pos_fires.sum() / max(1, pos_fires.sum() + neg_fires.sum())
            ),
            "dominance": float(np.max(pos_scores) / max(pos_sum, 1e-12)),
            "n_positive_docs": int(positive.sum()),
            "n_negative_docs": int((~positive).sum()),
        }
        ordered_rows = np.argsort(-acts[:, feature_position], kind="stable")
        seen_docs: set[int] = set()
        contexts: list[dict[str, Any]] = []
        for row_index in ordered_rows.tolist():
            doc_id = int(row_doc_ids[row_index])
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            contexts.append(
                {
                    "doc_id": doc_id,
                    "prompt_id": str(meta["prompt_id"][row_index]),
                    "split": "test",
                    "topic": str(meta["topic"][row_index]),
                    "axis_domain": str(meta["axis_domain"][row_index]),
                    "axis_language": str(meta["axis_language"][row_index]),
                    "position": int(meta["position"][row_index]),
                    "token": str(meta["token"][row_index]),
                    "activation": float(acts[row_index, feature_position]),
                    "document_score": float(
                        doc_scores[doc_index_by_id[doc_id], feature_position]
                    ),
                    "prompt": str(meta["prompt"][row_index]).strip(),
                }
            )
            if len(contexts) >= max_contexts:
                break
        output[feature] = {
            "heldout_metrics": metrics,
            "heldout_contexts": contexts,
        }
    audit = {
        "rows": int(len(x)),
        "dimensions": int(x.shape[1]),
        "documents": int(len(doc_ids)),
        "features": int(len(records)),
        "all_feature_activations_finite": bool(np.isfinite(acts).all()),
    }
    del x, acts
    gc.collect()
    return output, audit


def generated_specs(
    records: list[dict[str, Any]],
    directions: np.ndarray,
    target_norm: float,
    benchmark_sha256: str,
    seed: int,
    base_model_fingerprint: str,
    av_model_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    jobs: list[dict[str, Any]] = []
    av_inputs: dict[int, np.ndarray] = {}
    for direction_index, record in enumerate(records):
        feature = int(record["feature"])
        direction = np.asarray(directions[direction_index], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"feature {feature} has zero/non-finite decoder row")
        vector = (direction / norm * target_norm).astype(np.float32)
        av_inputs[feature] = vector
        base_request = next(
            row
            for row in record["generation_requests"]
            if row["kind"] == "base_autointerp"
        )
        base_prompt = autointerp_prompt(record)
        base_payload = {
            "benchmark_sha256": benchmark_sha256,
            "feature": feature,
            "candidate_id": base_request["candidate_id"],
            "kind": "base_autointerp",
            "model_fingerprint_sha256": base_model_fingerprint,
            "prompt": base_prompt,
            "max_new_tokens": 72,
            "greedy": True,
        }
        jobs.append(
            {
                "key": f"base:f{feature}",
                "feature": feature,
                "kind": "base_autointerp",
                "candidate_id": base_request["candidate_id"],
                "prompt": base_prompt,
                "max_new_tokens": 72,
                "model_fingerprint_sha256": base_model_fingerprint,
                "input_sha256": input_hash(base_payload),
            }
        )
        av_request = next(
            row
            for row in record["generation_requests"]
            if row["kind"] == "nla_av"
        )
        vector_sha = sha256_bytes(np.ascontiguousarray(vector).tobytes())
        av_payload = {
            "benchmark_sha256": benchmark_sha256,
            "feature": feature,
            "candidate_id": av_request["candidate_id"],
            "kind": "nla_av",
            "model_fingerprint_sha256": av_model_fingerprint,
            "vector_sha256": vector_sha,
            "target_norm": float(target_norm),
            "max_new_tokens": 200,
            "temperature": 0.0,
            "seed": int(seed),
        }
        jobs.append(
            {
                "key": f"av:f{feature}",
                "feature": feature,
                "kind": "nla_av",
                "candidate_id": av_request["candidate_id"],
                "vector_sha256": vector_sha,
                "max_new_tokens": 200,
                "seed": int(seed),
                "model_fingerprint_sha256": av_model_fingerprint,
                "input_sha256": input_hash(av_payload),
            }
        )
    return jobs, av_inputs


def assemble_candidates(
    records: list[dict[str, Any]],
    checkpoint: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        feature = int(record["feature"])
        for candidate in record["static_candidates"]:
            output.append(
                {
                    **candidate,
                    "feature": feature,
                    "concept_id": str(record["concept_id"]),
                    "superdomain": str(record["superdomain"]),
                }
            )
        for request in record["generation_requests"]:
            key = (
                f"av:f{feature}"
                if request["kind"] == "nla_av"
                else f"base:f{feature}"
            )
            generated = checkpoint[key]
            text = str(generated["text"])
            if not text or text != normalize_text(text):
                raise ValueError(
                    f"generated checkpoint text is not normalized for {key}"
                )
            text_sha = sha256_text(text)
            binding = {
                "candidate_id": str(request["candidate_id"]),
                "feature": feature,
                "kind": str(request["kind"]),
                "ordinal": 0,
                "candidate_concept_id": None,
                "text_sha256": text_sha,
            }
            output.append(
                {
                    "candidate_id": request["candidate_id"],
                    "feature": feature,
                    "concept_id": str(record["concept_id"]),
                    "superdomain": str(record["superdomain"]),
                    "kind": request["kind"],
                    "ordinal": 0,
                    "assigned_concept_id": str(record["concept_id"]),
                    "candidate_concept_id": None,
                    "candidate_superdomain": None,
                    "text": text,
                    "text_sha256": text_sha,
                    "generated_text_binding_sha256": input_hash(binding),
                    "generation_checkpoint_key": key,
                    "generation_input_sha256": str(generated["input_sha256"]),
                    "generated": True,
                }
            )
    ids = [str(row["candidate_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("assembled candidate IDs are not globally unique")
    if any(not str(row["text"]).strip() for row in output):
        raise ValueError("empty assembled candidate")
    return output


def write_rating_assets(
    packet_path: Path,
    key_path: Path,
    records: list[dict[str, Any]],
    heldout: dict[int, dict[str, Any]],
    candidates: list[dict[str, Any]],
    benchmark_sha256: str,
    result_sha256: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    candidates_by_feature: dict[int, list[dict[str, Any]]] = {}
    allowed = {
        "correct_reference",
        "hard_negative_reference",
        "nla_av",
        "base_autointerp",
    }
    for candidate in candidates:
        if candidate["kind"] in allowed:
            candidates_by_feature.setdefault(int(candidate["feature"]), []).append(
                candidate
            )
    item_order = rng.permutation(len(records)).tolist()
    packet_items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []

    def blind_context(row: dict[str, Any]) -> dict[str, Any]:
        output = {"text": str(row["prompt"])}
        if "token" in row:
            output["active_token"] = str(row["token"])
        return output

    for display_index, record_index in enumerate(item_order, start=1):
        record = records[record_index]
        feature = int(record["feature"])
        item_id = f"c1c-item-{display_index:03d}"
        feature_candidates = candidates_by_feature[feature]
        if len(feature_candidates) != 8:
            raise ValueError(
                f"rating packet expected 8 candidates for f{feature}, "
                f"got {len(feature_candidates)}"
            )
        candidate_order = rng.permutation(len(feature_candidates)).tolist()
        blind_candidates = []
        key_candidates = []
        for candidate_position, candidate_index in enumerate(
            candidate_order, start=1
        ):
            candidate = feature_candidates[candidate_index]
            blind_id = f"{item_id}-candidate-{candidate_position:02d}"
            blind_candidates.append(
                {
                    "blind_candidate_id": blind_id,
                    "text": str(candidate["text"]),
                    "ratings": {
                        "correctness_0_to_3": None,
                        "specificity_0_to_3": None,
                        "unsupported_assertions_0_to_3": None,
                    },
                }
            )
            key_candidates.append(
                {
                    "blind_candidate_id": blind_id,
                    "candidate_id": str(candidate["candidate_id"]),
                    "kind": str(candidate["kind"]),
                    "ordinal": int(candidate["ordinal"]),
                    "candidate_concept_id": candidate.get(
                        "candidate_concept_id"
                    ),
                }
            )
        packet_items.append(
            {
                "item_id": item_id,
                "discovery_activation_contexts": [
                    blind_context(row) for row in record["discovery_contexts"]
                ],
                "heldout_activation_contexts": [
                    blind_context(row)
                    for row in heldout[feature]["heldout_contexts"]
                ],
                "candidates": blind_candidates,
            }
        )
        key_items.append(
            {
                "item_id": item_id,
                "feature": feature,
                "concept_id": str(record["concept_id"]),
                "superdomain": str(record["superdomain"]),
                "candidates": key_candidates,
            }
        )
    packet = {
        "schema_version": 1,
        "experiment": "C1 confirmatory blinded human rating packet",
        "status": "sources_hidden_ready_for_independent_rating",
        "seed": int(seed),
        "source_hashes": {
            "benchmark_sha256": benchmark_sha256,
            "result_sha256": result_sha256,
        },
        "instructions": (
            "Using only the activation contexts, independently rate every "
            "candidate for correctness, specificity, and unsupported assertions "
            "on the displayed 0-3 fields. Do not inspect the separate key. "
            "At least three independent raters are required."
        ),
        "items": packet_items,
    }
    key = {
        "schema_version": 1,
        "experiment": "C1 confirmatory rating source key",
        "status": "WITHHOLD_UNTIL_RATINGS_ARE_FROZEN",
        "seed": int(seed),
        "source_hashes": packet["source_hashes"],
        "items": key_items,
    }
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    key_path.write_text(
        json.dumps(key, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--heldout-activations", required=True, type=Path)
    parser.add_argument("--heldout-provenance", required=True, type=Path)
    parser.add_argument("--model-freeze", required=True, type=Path)
    parser.add_argument("--sae", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--av", required=True, type=Path)
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vectors-out", required=True, type=Path)
    parser.add_argument("--rating-packet-out", required=True, type=Path)
    parser.add_argument("--rating-key-out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--rating-seed", type=int, default=20260732)
    parser.add_argument("--heldout-contexts", type=int, default=4)
    parser.add_argument("--sae-batch-size", type=int, default=512)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    for path in (
        args.benchmark,
        args.vectors,
        args.heldout_activations,
        args.heldout_provenance,
        args.model_freeze,
        args.sae / "params.safetensors",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.heldout_contexts < 1 or args.sae_batch_size < 1:
        raise ValueError("context count and SAE batch size must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    benchmark = read_json(args.benchmark, "benchmark")
    if benchmark.get("status") != BENCHMARK_STATUS:
        raise ValueError("benchmark is not frozen at the preregistered boundary")
    records = benchmark.get("records", [])
    if len(records) < 60:
        raise ValueError("benchmark contains fewer than 60 selected features")
    if int(
        benchmark.get("scope", {}).get("n_complete_reciprocal_pairs", 0)
    ) < 9:
        raise ValueError("benchmark contains fewer than nine complete pairs")
    benchmark_sha = sha256_file(args.benchmark)
    features = np.asarray([int(row["feature"]) for row in records], dtype=np.int64)
    if len(set(features.tolist())) != len(features):
        raise ValueError("benchmark feature IDs are not unique")
    if [int(row.get("direction_index", -1)) for row in records] != list(
        range(len(records))
    ):
        raise ValueError("benchmark direction_index order is not contiguous")
    validate_frozen_candidate_mapping(records)

    benchmark_inputs = benchmark.get("inputs", {})
    vector_input = benchmark_inputs.get("vectors")
    selection_input = benchmark_inputs.get("selection")
    if not isinstance(vector_input, dict) or not isinstance(selection_input, dict):
        raise ValueError("benchmark lacks frozen selection/vector inputs")
    vectors_sha = sha256_file(args.vectors)
    if vectors_sha != require_sha256(
        vector_input.get("sha256"), "benchmark vectors.sha256"
    ):
        raise ValueError("runtime vectors file differs from frozen benchmark")
    selection_sae_sha = require_sha256(
        selection_input.get("sae_params_sha256"),
        "benchmark selection.sae_params_sha256",
    )
    vector_sae_sha = require_sha256(
        vector_input.get("sae_params_sha256"),
        "benchmark vectors.sae_params_sha256",
    )
    if selection_sae_sha != vector_sae_sha:
        raise ValueError("benchmark selection/vector SAE identities differ")

    verified_models, model_freeze_sha = verify_model_freeze(
        args.model_freeze,
        {
            "base": args.base,
            "av": args.av,
            "ar": args.ar,
            "sae": args.sae,
        },
    )
    actual_sae_sha = sha256_file(args.sae / "params.safetensors")
    if actual_sae_sha != selection_sae_sha:
        raise ValueError("runtime SAE params differ from frozen benchmark")
    frozen_sae_params = next(
        (
            row["sha256"]
            for row in verified_models["sae"]["files"]
            if row["name"] == "params.safetensors"
        ),
        None,
    )
    if frozen_sae_params != actual_sae_sha:
        raise ValueError("runtime SAE params differ from model freeze")
    discovery_model_sha = require_sha256(
        benchmark_inputs.get("discovery_provenance", {}).get(
            "base_model_identity_sha256"
        ),
        "benchmark discovery_provenance.base_model_identity_sha256",
    )
    if discovery_model_sha != verified_models["base"]["fingerprint_sha256"]:
        raise ValueError(
            "discovery extraction base model differs from runtime model freeze"
        )
    heldout_provenance, heldout_activations_sha = verify_heldout_provenance(
        args.heldout_provenance,
        args.heldout_activations,
        benchmark,
        verified_models["base"]["fingerprint_sha256"],
    )

    with np.load(args.vectors, allow_pickle=False) as archive:
        direction_ids = np.asarray(archive["direction_ids"], dtype=np.int64)
        directions = np.asarray(archive["directions"], dtype=np.float32)
        m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
        target_norm = float(np.asarray(archive["target_norm"]).reshape(()))
        embedded_sae_sha = require_sha256(
            npz_scalar_text(archive, "sae_params_sha256"),
            "vectors embedded sae_params_sha256",
        )
        embedded_activation_sha = require_sha256(
            npz_scalar_text(archive, "activations_sha256"),
            "vectors embedded activations_sha256",
        )
        embedded_spec_sha = require_sha256(
            npz_scalar_text(archive, "concept_spec_sha256"),
            "vectors embedded concept_spec_sha256",
        )
        embedded_denylist_sha = require_sha256(
            npz_scalar_text(archive, "denylist_sha256"),
            "vectors embedded denylist_sha256",
        )
    expected_embedded = {
        "sae_params_sha256": vector_sae_sha,
        "activations_sha256": require_sha256(
            vector_input.get("activations_sha256"),
            "benchmark vectors.activations_sha256",
        ),
        "concept_spec_sha256": require_sha256(
            vector_input.get("concept_spec_sha256"),
            "benchmark vectors.concept_spec_sha256",
        ),
        "denylist_sha256": require_sha256(
            vector_input.get("denylist_sha256"),
            "benchmark vectors.denylist_sha256",
        ),
    }
    actual_embedded = {
        "sae_params_sha256": embedded_sae_sha,
        "activations_sha256": embedded_activation_sha,
        "concept_spec_sha256": embedded_spec_sha,
        "denylist_sha256": embedded_denylist_sha,
    }
    if actual_embedded != expected_embedded:
        raise ValueError("runtime vector embedded hashes differ from benchmark")
    selection_embedded = {
        key: require_sha256(
            selection_input.get(key), f"benchmark selection.{key}"
        )
        for key in expected_embedded
    }
    if selection_embedded != expected_embedded:
        raise ValueError("benchmark selection/vector embedded bindings differ")
    if not np.array_equal(direction_ids, features):
        raise ValueError("benchmark/vector feature order mismatch")
    if directions.ndim != 2 or directions.shape[0] != len(records):
        raise ValueError("directions have an unexpected shape")
    if int(heldout_provenance["d_model"]) != int(directions.shape[1]):
        raise ValueError(
            "held-out provenance residual dimension differs from decoder directions"
        )
    if not np.isfinite(directions).all():
        raise ValueError("non-finite decoder direction")
    m_norm = float(np.linalg.norm(m_hat))
    if not np.isfinite(m_norm) or abs(m_norm - 1.0) > 1e-4:
        raise ValueError(f"m_hat is not unit norm: {m_norm}")
    direction_centered = normalize_rows(project_rows(directions, m_hat))

    jobs, av_inputs = generated_specs(
        records,
        directions,
        target_norm,
        benchmark_sha,
        args.seed,
        verified_models["base"]["fingerprint_sha256"],
        verified_models["av"]["fingerprint_sha256"],
    )
    checkpoint = load_checkpoint(args.checkpoint)
    expected = {row["key"]: row for row in jobs}
    unknown = set(checkpoint) - set(expected)
    if unknown:
        raise ValueError(
            f"checkpoint contains non-protocol keys: {sorted(unknown)[:5]}"
        )
    for key, row in checkpoint.items():
        if row["input_sha256"] != expected[key]["input_sha256"]:
            raise ValueError(
                f"checkpoint input hash mismatch for {key}; use a fresh checkpoint"
            )
        if row.get("benchmark_sha256") != benchmark_sha:
            raise ValueError(f"checkpoint benchmark hash mismatch for {key}")
        if (
            row.get("model_fingerprint_sha256")
            != expected[key]["model_fingerprint_sha256"]
        ):
            raise ValueError(f"checkpoint model fingerprint mismatch for {key}")
    pending_base = [
        row
        for row in jobs
        if row["kind"] == "base_autointerp" and row["key"] not in checkpoint
    ]
    pending_av = [
        row
        for row in jobs
        if row["kind"] == "nla_av" and row["key"] not in checkpoint
    ]
    print(
        f"[plan] features={len(records)} base_pending={len(pending_base)} "
        f"av_pending={len(pending_av)} checkpoint={len(checkpoint)} "
        f"reference_pairings={sum(len(row['static_candidates']) for row in records)}"
    )
    if args.plan_only:
        print("C1_CONFIRMATORY_PLAN_VALID")
        return

    total_started = time.perf_counter()
    heldout_started = time.perf_counter()
    heldout, heldout_audit = heldout_contexts_and_metrics(
        args.heldout_activations,
        args.sae,
        records,
        args.heldout_contexts,
        args.sae_batch_size,
    )
    heldout_seconds = time.perf_counter() - heldout_started
    if (
        int(heldout_audit["documents"]) != 48
        or int(heldout_audit["rows"])
        != int(heldout_provenance["activation_rows"])
        or int(heldout_audit["dimensions"]) != int(heldout_provenance["d_model"])
    ):
        raise ValueError(
            "held-out parquet contents differ from verified provenance counts"
        )
    print(
        f"[heldout] rows={heldout_audit['rows']} "
        f"documents={heldout_audit['documents']} seconds={heldout_seconds:.1f}"
    )

    base_started = time.perf_counter()
    if pending_base:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.base), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(args.base),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        for ordinal, spec in enumerate(pending_base, start=1):
            raw, seconds = base_generate(
                base_model,
                tokenizer,
                spec["prompt"],
                int(spec["max_new_tokens"]),
            )
            text, tag_ok = extract_label(raw)
            if not text:
                raise ValueError(f"empty base autointerpretation {spec['key']}")
            row = {
                "key": spec["key"],
                "job_type": "base_autointerp",
                "feature": int(spec["feature"]),
                "candidate_id": spec["candidate_id"],
                "input_sha256": spec["input_sha256"],
                "benchmark_sha256": benchmark_sha,
                "model_fingerprint_sha256": spec[
                    "model_fingerprint_sha256"
                ],
                "raw_completion": raw,
                "text": text,
                "label_tag_ok": tag_ok,
                "generation_seconds": seconds,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(args.checkpoint, row)
            checkpoint[spec["key"]] = row
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(pending_base):
                print(
                    f"[base {ordinal}/{len(pending_base)}] "
                    f"{spec['key']} tag={tag_ok} seconds={seconds:.1f}"
                )
        del base_model, tokenizer
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    base_seconds = time.perf_counter() - base_started

    av_started = time.perf_counter()
    if pending_av:
        av = AVLocal(str(args.av), device="cuda")
        for ordinal, spec in enumerate(pending_av, start=1):
            raw, text, tag_ok, seconds = av_generate(
                av,
                av_inputs[int(spec["feature"])],
                int(spec["max_new_tokens"]),
                int(spec["seed"]),
            )
            if not text:
                raise ValueError(f"empty NLA AV explanation {spec['key']}")
            row = {
                "key": spec["key"],
                "job_type": "nla_av_greedy",
                "feature": int(spec["feature"]),
                "candidate_id": spec["candidate_id"],
                "input_sha256": spec["input_sha256"],
                "benchmark_sha256": benchmark_sha,
                "model_fingerprint_sha256": spec[
                    "model_fingerprint_sha256"
                ],
                "vector_sha256": spec["vector_sha256"],
                "raw_completion": raw,
                "text": text,
                "explanation_tag_ok": tag_ok,
                "generation_seconds": seconds,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(args.checkpoint, row)
            checkpoint[spec["key"]] = row
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(pending_av):
                print(
                    f"[AV {ordinal}/{len(pending_av)}] "
                    f"{spec['key']} tag={tag_ok} seconds={seconds:.1f}"
                )
        del av
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    av_seconds = time.perf_counter() - av_started

    candidates = assemble_candidates(records, checkpoint)
    unique_texts = list(dict.fromkeys(str(row["text"]) for row in candidates))
    unique_text_hashes = [sha256_text(normalize_text(text)) for text in unique_texts]
    ar_input_sha = input_hash(
        {
            "benchmark_sha256": benchmark_sha,
            "vectors_sha256": vectors_sha,
            "ar_model_fingerprint_sha256": verified_models["ar"][
                "fingerprint_sha256"
            ],
            "unique_normalized_text_sha256": unique_text_hashes,
            "directions_sha256": sha256_bytes(
                np.ascontiguousarray(directions).tobytes()
            ),
            "m_hat_sha256": sha256_bytes(np.ascontiguousarray(m_hat).tobytes()),
            "centering": "discovery_m_hat_orthogonal_projection_then_l2",
        }
    )
    ar_started = time.perf_counter()
    critic = NLACritic(str(args.ar), device="cuda")
    reconstruction_cache: dict[str, np.ndarray] = {}
    for index, text in enumerate(unique_texts, start=1):
        vector = np.asarray(critic.reconstruct(text).numpy(), dtype=np.float32)
        if vector.shape != (directions.shape[1],) or not np.isfinite(vector).all():
            raise ValueError(f"invalid AR reconstruction at unique text {index}")
        reconstruction_cache[text] = vector
        if index == 1 or index % 25 == 0 or index == len(unique_texts):
            print(f"[AR {index}/{len(unique_texts)}]")
    torch.cuda.synchronize()
    ar_seconds = time.perf_counter() - ar_started
    del critic
    gc.collect()
    torch.cuda.empty_cache()

    unique_reconstructions = np.stack(
        [reconstruction_cache[text] for text in unique_texts]
    ).astype(np.float32)
    unique_centered = normalize_rows(
        project_rows(unique_reconstructions, m_hat)
    )
    unique_similarity = unique_centered @ direction_centered.T
    unique_index = {text: index for index, text in enumerate(unique_texts)}
    feature_index = {int(feature): index for index, feature in enumerate(features)}
    labels = [str(row["concept_id"]) for row in records]
    concept_order = list(dict.fromkeys(labels))
    concept_direction_indices = {
        concept_id: np.asarray(
            [index for index, label in enumerate(labels) if label == concept_id],
            dtype=np.int64,
        )
        for concept_id in concept_order
    }
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        feature = int(candidate["feature"])
        target_index = feature_index[feature]
        scores = unique_similarity[unique_index[str(candidate["text"])]]
        target_score = float(scores[target_index])
        concept_id = str(candidate["concept_id"])
        concept_scores = np.asarray(
            [
                float(np.mean(scores[concept_direction_indices[label]]))
                for label in concept_order
            ],
            dtype=np.float64,
        )
        concept_index = concept_order.index(concept_id)
        scored.append(
            {
                **candidate,
                "target_cos_centered": target_score,
                "feature_retrieval_rank": average_rank(scores, target_index),
                "feature_retrieval_top5": bool(
                    average_rank(scores, target_index) <= 5.0
                ),
                "assigned_concept_mean_cos_centered": float(
                    concept_scores[concept_index]
                ),
                "concept_retrieval_rank": average_rank(
                    concept_scores, concept_index
                ),
                "concept_retrieval_top1": bool(
                    average_rank(concept_scores, concept_index) == 1.0
                ),
            }
        )
    if any(
        not np.isfinite(float(row["target_cos_centered"])) for row in scored
    ):
        raise ValueError("non-finite centered q score")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.vectors_out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "experiment": "C1 confirmatory synthetic cohort v1",
        "status": "complete_ready_for_preregistered_cluster_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": benchmark["scope"],
        "inputs": {
            "benchmark": {
                "path": str(args.benchmark),
                "sha256": benchmark_sha,
            },
            "vectors": {
                "path": str(args.vectors),
                "sha256": vectors_sha,
                **actual_embedded,
            },
            "heldout_activations": {
                "path": str(args.heldout_activations),
                "sha256": heldout_activations_sha,
            },
            "heldout_provenance": heldout_provenance,
            "model_freeze": {
                "path": str(args.model_freeze),
                "sha256": model_freeze_sha,
                "status": MODEL_FREEZE_STATUS,
            },
            "sae_params": {
                "path": str(args.sae / "params.safetensors"),
                "sha256": actual_sae_sha,
                "model_fingerprint_sha256": verified_models["sae"][
                    "fingerprint_sha256"
                ],
            },
            "base": {
                **verified_models["base"],
            },
            "av": {
                **verified_models["av"],
            },
            "ar": {
                **verified_models["ar"],
                "reconstruction_input_sha256": ar_input_sha,
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256_file(args.checkpoint),
            },
        },
        "protocol": benchmark["protocol"],
        "runtime": {
            "heldout_sae_seconds": heldout_seconds,
            "base_seconds_this_invocation": base_seconds,
            "av_seconds_this_invocation": av_seconds,
            "ar_seconds": ar_seconds,
            "total_seconds_this_invocation": time.perf_counter() - total_started,
            "n_checkpoint_rows": len(checkpoint),
            "n_candidate_pairings": len(candidates),
            "n_unique_texts_reconstructed": len(unique_texts),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
            "available_disk_bytes": int(
                shutil.disk_usage(args.out.parent).free
            ),
        },
        "qa": {
            "heldout": heldout_audit,
            "feature_order": features.tolist(),
            "concept_order": concept_order,
            "all_reconstructions_finite": True,
            "all_scores_finite": True,
            "no_automatic_semantic_judge": True,
            "benchmark_candidate_mapping_verified": True,
            "heldout_provenance_verified": True,
            "model_freeze_verified": True,
        },
        "feature_metadata": [
            {
                "feature": int(record["feature"]),
                "direction_index": int(record["direction_index"]),
                "concept_id": str(record["concept_id"]),
                "superdomain": str(record["superdomain"]),
                "hard_negative_id": str(record["hard_negative_id"]),
                "selection_tier": str(record["selection_tier"]),
                "train_metrics": record["train_metrics"],
                "heldout_metrics": heldout[int(record["feature"])][
                    "heldout_metrics"
                ],
                "discovery_contexts": record["discovery_contexts"],
                "heldout_contexts": heldout[int(record["feature"])][
                    "heldout_contexts"
                ],
            }
            for record in records
        ],
        "scored_candidates": scored,
    }
    args.out.write_text(
        json.dumps(
            to_builtin(output),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    np.savez_compressed(
        args.vectors_out,
        unique_text_sha256=np.asarray(
            unique_text_hashes, dtype="U64"
        ),
        reconstruction_vectors=unique_reconstructions,
        semantic_similarity=unique_similarity.astype(np.float32),
        semantic_feature_ids=features,
        semantic_directions=directions,
        m_hat=m_hat.astype(np.float32),
        ar_model_fingerprint_sha256=np.asarray(
            verified_models["ar"]["fingerprint_sha256"]
        ),
        ar_reconstruction_input_sha256=np.asarray(ar_input_sha),
        benchmark_sha256=np.asarray(benchmark_sha),
    )
    result_sha = sha256_file(args.out)
    write_rating_assets(
        args.rating_packet_out,
        args.rating_key_out,
        records,
        heldout,
        candidates,
        benchmark_sha,
        result_sha,
        args.rating_seed,
    )
    print("C1_CONFIRMATORY_RUN_COMPLETE")
    print(
        canonical_json(
            {
                "features": len(records),
                "candidate_pairings": len(candidates),
                "unique_texts": len(unique_texts),
                "checkpoint_rows": len(checkpoint),
                "out": str(args.out),
                "out_sha256": result_sha,
                "rating_packet": str(args.rating_packet_out),
                "rating_packet_sha256": sha256_file(args.rating_packet_out),
                "rating_key": str(args.rating_key_out),
                "rating_key_sha256": sha256_file(args.rating_key_out),
            }
        )
    )


if __name__ == "__main__":
    main()
