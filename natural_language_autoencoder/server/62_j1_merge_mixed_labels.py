"""Validate Luna labels and assemble the mixed-labeler J1 artifact.

The first thirteen successful Fable rows are read from the immutable v1
checkpoint.  Batches 13--44 are read from one immutable JSON artifact per
fresh Luna worker.  The merger is intentionally a validator and copier: it
does not call a model, compute scientific metrics, or inspect held-out truth.
Only after all 45 batches and all 225 opaque cases pass the fail-closed
checks are versioned mixed checkpoint/result files written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


N_BATCHES = 45
N_CASES = 5
FABLE_BATCHES = tuple(range(13))
LUNA_BATCHES = tuple(range(13, N_BATCHES))
ALL_BATCHES = tuple(range(N_BATCHES))
EXPECTED_JOBS_SHA256 = (
    "411c67acd230018c60d50194d51c70f3cde847c7f85b38713456450b609f4aad"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "439f8af0c438e8a79a13af5bb6dd463932ca1d56e225912da3dba9edd8cd12e4"
)
EXPECTED_PROTOCOL_SHA256 = (
    "b8653f78accc76b8a3b61a460c812df73cf76ab0111bc202da6bdc1d030081e3"
)
EXPECTED_ANALYSIS_PLAN_SHA256 = (
    "772042a159188d7777f944b59f152b2484cae719cfbfd72190921d91f1aa7147"
)
EXPECTED_EXECUTION_AMENDMENT_SHA256 = (
    "7c7babec80ba123f640a82ca1b4b5d51648d28f3abd3c6a03314092880314114"
)
EXPECTED_LUNA_MODEL = "gpt-5.6-luna"
EXPECTED_LUNA_ROLE = "luna_worker"
EXPECTED_LUNA_REASONING = "max"
CASE_ID_RE = re.compile(r"^case_[a-z0-9]{18}$")
CLI_AGENT_TASKS = {
    batch: f"codex-cli://j1/luna_batch_{batch:02d}" for batch in range(27, 45)
}


def _load_prepare_module():
    path = Path(__file__).with_name("61_j1_prepare_luna_batches.py")
    spec = importlib.util.spec_from_file_location("j1_prepare_luna_batches", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling preparation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = _load_prepare_module()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                   allow_nan=False)
        + "\n"
    ).encode("utf-8")


def immutable_json(path: Path, value: Any) -> str:
    data = pretty_bytes(value)
    return immutable_bytes(path, data)


def immutable_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return sha256_bytes(data)


def immutable_sidecar(path: Path, digest: str) -> None:
    data = (digest + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        tokens = path.read_text(encoding="ascii").strip().split()
        if not tokens or tokens[0].lower() != digest:
            raise RuntimeError(f"sha256 sidecar mismatch: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def verify_sidecar(path: Path, *, expected: str | None = None) -> str:
    digest = sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing immutable SHA-256 sidecar: {sidecar}")
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    if not tokens or tokens[0].lower() != digest:
        declared = tokens[0] if tokens else "<empty>"
        raise ValueError(f"SHA-256 sidecar mismatch for {path}: {declared} != {digest}")
    if expected is not None and digest != expected:
        raise ValueError(f"unexpected SHA-256 for {path}: {digest} != {expected}")
    return digest


def validate_execution_amendment(path: Path) -> str:
    digest = verify_sidecar(path, expected=EXPECTED_EXECUTION_AMENDMENT_SHA256)
    text = path.read_text(encoding="utf-8")
    if "`codex-cli://j1/luna_batch_XX`" not in text:
        raise ValueError("CLI fallback amendment omits the frozen worker-ID contract")
    return digest


def expected_agent_task(batch: int, index_entry: Mapping[str, Any]) -> str:
    if batch in CLI_AGENT_TASKS:
        return CLI_AGENT_TASKS[batch]
    task = index_entry.get("expected_agent_task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"public index lacks expected agent task for batch {batch}")
    return task


def _finite(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _validate_luna_case(case: Any, expected_id: str, where: str) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"{where} is not an object")
    required = {"case_id", "hypothesis", "positive_cues", "exclusion_cues",
                "abstain", "confidence"}
    if set(case) != required:
        raise ValueError(f"{where} has missing or extra fields")
    if case.get("case_id") != expected_id:
        raise ValueError(f"{where} case_id mismatch")
    if (not isinstance(case.get("hypothesis"), str)
            or not case["hypothesis"].strip()
            or len(case["hypothesis"].split()) > 32):
        raise ValueError(f"{where} hypothesis must contain at most 32 words")
    for key in ("positive_cues", "exclusion_cues"):
        if not isinstance(case.get(key), list) or not all(isinstance(x, str) for x in case[key]):
            raise ValueError(f"{where} {key} must be an array of strings")
    if type(case.get("abstain")) is not bool:
        raise ValueError(f"{where} abstain must be boolean")
    if not _finite(case.get("confidence")) or not 0 <= float(case["confidence"]) <= 1:
        raise ValueError(f"{where} confidence must be in [0,1]")


def _safe_relative_name(name: Any, expected: str) -> str:
    if not isinstance(name, str) or name != expected or Path(name).name != name:
        raise ValueError(f"unexpected relative artifact filename {name!r}; expected {expected!r}")
    return name


def _validate_public_index(index_path: Path, jobs: Mapping[int, Mapping[str, Any]],
                           jobs_sha: str, checkpoint_sha: str, protocol_sha: str) -> dict[int, dict[str, Any]]:
    index_sha = verify_sidecar(index_path)
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("public Luna index must be an object")
    if int(index.get("schema_version", -1)) < 1:
        raise ValueError("public Luna index has invalid schema_version")
    if not str(index.get("status", "")).startswith("J1_LUNA_PUBLIC_BATCH_INDEX_FROZEN"):
        raise ValueError("public Luna index is not frozen")
    if index.get("jobs_sha256") != jobs_sha or index.get("checkpoint_sha256") != checkpoint_sha \
            or index.get("protocol_sha256") != protocol_sha:
        raise ValueError("public Luna index parent SHA binding mismatch")
    batches = index.get("batches")
    if not isinstance(batches, list) or len(batches) != len(LUNA_BATCHES):
        raise ValueError("public Luna index must contain exactly 32 batches")
    if {int(x) for x in index.get("batch_ids", [])} != set(LUNA_BATCHES):
        raise ValueError("public Luna index batch_ids are not exactly 13..44")
    by_batch: dict[int, dict[str, Any]] = {}
    for entry in batches:
        if not isinstance(entry, dict):
            raise ValueError("public Luna index batch entry is not an object")
        try:
            batch = int(entry["batch_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("public Luna index batch_id is malformed") from exc
        if batch not in LUNA_BATCHES or batch in by_batch:
            raise ValueError(f"public Luna index has duplicate/out-of-range batch {batch}")
        prompt_name = _safe_relative_name(entry.get("prompt_filename"), f"batch_{batch:02d}_prompt.json")
        artifact_path = index_path.parent / prompt_name
        artifact_sha = verify_sidecar(artifact_path)
        if entry.get("artifact_sha256") != artifact_sha:
            raise ValueError(f"public batch {batch} artifact SHA mismatch in index")
        expected_public_sha = entry.get("expected_public_prompt_sha256", artifact_sha)
        if expected_public_sha != artifact_sha:
            raise ValueError(f"public batch {batch} expected public prompt SHA mismatch")
        artifact = read_json(artifact_path)
        if not isinstance(artifact, dict):
            raise ValueError(f"public batch {batch} artifact is not an object")
        PREP._validate_public_payload(artifact, f"public batch {batch}")
        if artifact.get("batch_id") != batch or artifact.get("jobs_sha256") != jobs_sha \
                or artifact.get("checkpoint_sha256") != checkpoint_sha \
                or artifact.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"public batch {batch} parent/batch binding mismatch")
        if artifact.get("prompt_sha256") != sha256_bytes(str(artifact.get("prompt", "")).encode("utf-8")):
            raise ValueError(f"public batch {batch} embedded prompt SHA mismatch")
        cases = artifact.get("cases")
        if not isinstance(cases, list) or len(cases) != N_CASES:
            raise ValueError(f"public batch {batch} must contain five cases")
        expected_ids = {str(case["case_id"]) for case in jobs[batch]["cases"]}
        got_ids: set[str] = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"case_id", "evidence"}:
                raise ValueError(f"public batch {batch} case shape is not public")
            cid = case["case_id"]
            if not isinstance(cid, str) or not CASE_ID_RE.fullmatch(cid) or cid in got_ids:
                raise ValueError(f"public batch {batch} case IDs malformed/duplicated")
            got_ids.add(cid)
        if got_ids != expected_ids or set(entry.get("case_ids", [])) != expected_ids:
            raise ValueError(f"public batch {batch} case IDs differ from frozen job")
        output_name = _safe_relative_name(entry.get("expected_output_filename"), f"batch_{batch:02d}_output.json")
        if artifact.get("luna_output_filename") != output_name:
            raise ValueError(f"public batch {batch} output filename binding mismatch")
        if entry.get("expected_role") != EXPECTED_LUNA_ROLE or entry.get("expected_model") != EXPECTED_LUNA_MODEL \
                or entry.get("expected_reasoning") != EXPECTED_LUNA_REASONING:
            raise ValueError(f"public batch {batch} Luna provenance contract mismatch")
        by_batch[batch] = {
            "batch_id": batch,
            "artifact_path": artifact_path,
            "artifact_sha256": artifact_sha,
            "embedded_prompt_sha256": artifact["prompt_sha256"],
            "case_ids": sorted(expected_ids),
            "output_filename": output_name,
            "expected_agent_task": entry.get("expected_agent_task"),
        }
    if set(by_batch) != set(LUNA_BATCHES):
        raise ValueError("public Luna index omits a batch")
    return by_batch


def _validate_luna_output(path: Path, entry: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    artifact_sha = verify_sidecar(path)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Luna output {path.name} is not an object")
    allowed = {
        "schema_version", "status", "batch_id", "cases", "agent_task", "role",
        "model", "reasoning", "public_prompt_sha256", "prompt_sha256",
        "output_artifact_sha256", "labeler", "provenance",
    }
    if any(key not in allowed for key in payload):
        raise ValueError(f"Luna output {path.name} contains unknown/private top-level fields")
    batch = int(payload.get("batch_id", -1))
    if batch != int(entry["batch_id"]):
        raise ValueError(f"Luna output {path.name} batch_id mismatch")
    if payload.get("status") not in (None, "ok", "success"):
        raise ValueError(f"Luna output {path.name} status is not successful")
    agent_task = payload.get("agent_task")
    if not isinstance(agent_task, str) or not agent_task.strip():
        raise ValueError(f"Luna output {path.name} lacks canonical agent_task")
    expected_task = entry.get("expected_agent_task")
    if isinstance(expected_task, str) and expected_task and agent_task != expected_task:
        raise ValueError(f"Luna output {path.name} agent_task differs from public assignment")
    if payload.get("role") != EXPECTED_LUNA_ROLE or payload.get("model") != EXPECTED_LUNA_MODEL \
            or payload.get("reasoning") != EXPECTED_LUNA_REASONING:
        raise ValueError(f"Luna output {path.name} role/model/reasoning mismatch")
    if payload.get("labeler") not in (None, "luna"):
        raise ValueError(f"Luna output {path.name} labeler must be luna when present")
    if payload.get("public_prompt_sha256") != entry["artifact_sha256"]:
        raise ValueError(f"Luna output {path.name} public_prompt_sha256 must bind to public artifact file SHA")
    if "prompt_sha256" in payload and payload["prompt_sha256"] != entry["embedded_prompt_sha256"]:
        raise ValueError(f"Luna output {path.name} embedded prompt SHA mismatch")
    if "output_artifact_sha256" in payload and payload["output_artifact_sha256"] != artifact_sha:
        raise ValueError(f"Luna output {path.name} output_artifact_sha256 mismatch")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != N_CASES:
        raise ValueError(f"Luna output {path.name} must contain exactly five cases")
    expected_ids = set(entry["case_ids"])
    got_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError(f"Luna output {path.name} has malformed case")
        cid = case["case_id"]
        if not CASE_ID_RE.fullmatch(cid) or cid in got_ids or cid not in expected_ids:
            raise ValueError(f"Luna output {path.name} case IDs mismatch")
        got_ids.add(cid)
        _validate_luna_case(case, cid, f"Luna batch {entry['batch_id']} case {cid}")
    if got_ids != expected_ids:
        raise ValueError(f"Luna output {path.name} omitted a case")
    provenance = payload.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError(f"Luna output {path.name} provenance is not an object")
        for key in ("agent_task", "role", "model", "reasoning"):
            if provenance.get(key) != payload[key]:
                raise ValueError(f"Luna output {path.name} nested provenance mismatch: {key}")
    return payload, artifact_sha


def _fable_provenance(batch: int, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "agent_task": f"/root/fable_batch_{batch:02d}",
        "role": "fable_labeler",
        "model": "claude-fable-5",
        "reasoning": "low",
        "labeler": "fable",
        "source_raw_output_sha256": sha256_bytes(str(row["raw_cli_json"]).encode("utf-8")),
    }


def _luna_provenance(batch: int, payload: Mapping[str, Any], artifact_sha: str) -> dict[str, Any]:
    provenance = {
        "agent_task": payload["agent_task"],
        "role": payload["role"],
        "model": payload["model"],
        "reasoning": payload["reasoning"],
        "labeler": "luna",
        "source_output_artifact_sha256": artifact_sha,
        "batch_id": batch,
    }
    source = payload.get("provenance")
    if isinstance(source, dict):
        provenance["source_worker_provenance"] = dict(source)
    return provenance


def _merge_case(case: Mapping[str, Any], batch: int, provenance: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(case)
    merged["batch_id"] = batch
    merged["labeler"] = provenance["labeler"]
    merged["agent_task"] = provenance["agent_task"]
    merged["role"] = provenance["role"]
    merged["model"] = provenance["model"]
    merged["reasoning"] = provenance["reasoning"]
    merged["provenance"] = dict(provenance)
    return merged


def merge_mixed_labels(jobs_path: Path, checkpoint_path: Path, protocol_path: Path,
                       public_index_path: Path, luna_outputs_dir: Path,
                       out_checkpoint: Path, out_result: Path,
                       analysis_plan_path: Path | None = None,
                       execution_amendment_path: Path | None = None) -> dict[str, Any]:
    jobs_sha = PREP.verify_sidecar(jobs_path, expected=EXPECTED_JOBS_SHA256)
    checkpoint_sha = PREP.sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected v1 checkpoint snapshot SHA: {checkpoint_sha}")
    protocol_sha = PREP._validate_protocol(protocol_path)
    if execution_amendment_path is None:
        raise ValueError("CLI fallback execution amendment is required")
    execution_amendment_sha = validate_execution_amendment(
        execution_amendment_path
    )
    jobs_doc = PREP.read_json(jobs_path)
    jobs, _ = PREP._validate_jobs(jobs_doc)
    checkpoint_rows = PREP.read_jsonl(checkpoint_path)
    fable_rows = PREP._validate_checkpoint(checkpoint_rows, jobs)
    index_entries = _validate_public_index(public_index_path, jobs, jobs_sha,
                                           checkpoint_sha, protocol_sha)
    if not luna_outputs_dir.is_dir():
        raise FileNotFoundError(luna_outputs_dir)
    expected_output_names = {entry["output_filename"] for entry in index_entries.values()}
    actual_output_names = {path.name for path in luna_outputs_dir.glob("batch_*_output.json")}
    if actual_output_names != expected_output_names:
        missing = sorted(expected_output_names - actual_output_names)
        extra = sorted(actual_output_names - expected_output_names)
        raise ValueError(f"Luna output set mismatch; missing={missing}, extra={extra}")

    luna_payloads: dict[int, tuple[dict[str, Any], str]] = {}
    seen_tasks: set[str] = set()
    for batch in LUNA_BATCHES:
        entry = dict(index_entries[batch])
        entry["expected_agent_task"] = expected_agent_task(batch, entry)
        path = luna_outputs_dir / entry["output_filename"]
        payload, artifact_sha = _validate_luna_output(path, entry)
        task = payload["agent_task"]
        if task in seen_tasks:
            raise ValueError(f"Luna agent_task reused for batch {batch}: {task}")
        seen_tasks.add(task)
        luna_payloads[batch] = (payload, artifact_sha)
    if len(luna_payloads) != len(LUNA_BATCHES):
        raise ValueError("not all Luna batches validated")

    labeler_by_batch = {str(batch): ("fable" if batch in FABLE_BATCHES else "luna")
                        for batch in ALL_BATCHES}
    protocol_deviation = {
        "kind": "MIXED_LABELER_EXPLORATORY_TRIAGE",
        "single_interpreter": False,
        "fable_batches": list(FABLE_BATCHES),
        "luna_batches": list(LUNA_BATCHES),
        "fable_model": "claude-fable-5",
        "luna_model": EXPECTED_LUNA_MODEL,
        "luna_role": EXPECTED_LUNA_ROLE,
        "luna_reasoning": EXPECTED_LUNA_REASONING,
        "repeated_fable_batches": [],
        "fable_successes_reused": True,
        "fable_requests_issued_during_completion": False,
        "statement": "Batches 0..12 are retained Fable successes; fresh Luna workers label 13..44.",
        "cli_fallback_amendment_sha256": execution_amendment_sha,
        "luna_cli_batches": list(range(27, 45)),
        "unused_nested_spawn_attempt": "agent_thread_limit_reached_before_outputs",
    }

    merged_rows: list[dict[str, Any]] = []
    job_summaries: list[dict[str, Any]] = []
    for batch in ALL_BATCHES:
        job = jobs[batch]
        expected_ids = {str(case["case_id"]) for case in job["cases"]}
        if batch in FABLE_BATCHES:
            source_row = fable_rows[batch]
            provenance = _fable_provenance(batch, source_row)
            source_cases = source_row["cases"]
            source_sha = provenance["source_raw_output_sha256"]
        else:
            source_payload, source_sha = luna_payloads[batch]
            provenance = _luna_provenance(batch, source_payload, source_sha)
            source_cases = source_payload["cases"]
        if {str(case["case_id"]) for case in source_cases} != expected_ids:
            raise ValueError(f"merged batch {batch} case IDs do not cover frozen job")
        merged_cases = [_merge_case(case, batch, provenance) for case in source_cases]
        merged_row = {
            "batch_id": batch,
            "status": "ok",
            "jobs_sha256": jobs_sha,
            "freeze_sha256": jobs_doc.get("freeze_sha256"),
            "input_result_sha256": jobs_doc.get("input_result_sha256"),
            "label_protocol_sha256": jobs_doc.get("label_protocol_sha256"),
            "protocol_sha256": jobs_doc.get("protocol_sha256"),
            "luna_completion_protocol_sha256": protocol_sha,
            "labeler": provenance["labeler"],
            "agent_task": provenance["agent_task"],
            "role": provenance["role"],
            "model": provenance["model"],
            "reasoning": provenance["reasoning"],
            "prompt_sha256": job["prompt_sha256"],
            "input_prompt_sha256": job["prompt_sha256"],
            "public_prompt_sha256": (job["prompt_sha256"] if batch in FABLE_BATCHES
                                      else index_entries[batch]["artifact_sha256"]),
            "source_output_artifact_sha256": source_sha,
            "cases": merged_cases,
            "provenance": provenance,
        }
        merged_rows.append(merged_row)
        summary = {
            "batch_id": batch,
            "status": "ok",
            "labeler": provenance["labeler"],
            "labeler_by_batch": provenance["labeler"],
            "case_ids": sorted(expected_ids),
            "prompt_sha256": job["prompt_sha256"],
            "input_prompt_sha256": job["prompt_sha256"],
            "source_output_artifact_sha256": source_sha,
            "agent_task": provenance["agent_task"],
            "role": provenance["role"],
            "model": provenance["model"],
            "reasoning": provenance["reasoning"],
            "protocol_deviation": protocol_deviation,
        }
        job_summaries.append(summary)
    if len(merged_rows) != N_BATCHES or len(job_summaries) != N_BATCHES:
        raise ValueError("merged artifact must contain exactly 45 batches")
    if len({str(case["case_id"]) for row in merged_rows for case in row["cases"]}) != N_BATCHES * N_CASES:
        raise ValueError("merged artifact must contain exactly 225 globally unique cases")

    header = {
        "kind": "header",
        "schema_version": 2,
        "status": "J1_DISCOVERY_LABEL_CHECKPOINT_MIXED_V2",
        "jobs_sha256": jobs_sha,
        # Preserve the upstream bindings consumed by the existing evaluator;
        # these are provenance hashes, not private prompt content.
        "freeze_sha256": jobs_doc.get("freeze_sha256"),
        "input_result_sha256": jobs_doc.get("input_result_sha256"),
        "label_protocol_sha256": jobs_doc.get("label_protocol_sha256"),
        "source_checkpoint_sha256": checkpoint_sha,
        "protocol_sha256": jobs_doc.get("protocol_sha256"),
        "luna_completion_protocol_sha256": protocol_sha,
        "cli_fallback_amendment_sha256": execution_amendment_sha,
        "labeler_by_batch": labeler_by_batch,
        "protocol_deviation": protocol_deviation,
        "historical_checkpoint_errors_retained_in": "j1_discovery_labels_checkpoint_v1.jsonl",
        "scientific_rows_complete": True,
    }
    checkpoint_lines = [canonical_bytes(header)] + [canonical_bytes(row) for row in merged_rows]
    mixed_checkpoint_sha = immutable_bytes(out_checkpoint, b"".join(checkpoint_lines))
    immutable_sidecar(Path(str(out_checkpoint) + ".sha256"), mixed_checkpoint_sha)

    result = {
        "schema_version": 2,
        "status": "EXPLORATORY_LABEL_RESULT_MIXED_COMPLETE",
        "jobs_sha256": jobs_sha,
        "freeze_sha256": jobs_doc.get("freeze_sha256"),
        "input_result_sha256": jobs_doc.get("input_result_sha256"),
        "label_protocol_sha256": jobs_doc.get("label_protocol_sha256"),
        "source_checkpoint_sha256": checkpoint_sha,
        "mixed_checkpoint_sha256": mixed_checkpoint_sha,
        "protocol_sha256": jobs_doc.get("protocol_sha256"),
        "luna_completion_protocol_sha256": protocol_sha,
        "cli_fallback_amendment_sha256": execution_amendment_sha,
        "labeler_by_batch": labeler_by_batch,
        "protocol_deviation": protocol_deviation,
        "historical_checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha,
            "errors_not_scientific_rows": True,
        },
        "jobs": jobs_doc["jobs"],
        "job_summaries": job_summaries,
        "rows": merged_rows,
        "case_count": N_BATCHES * N_CASES,
        "batch_count": N_BATCHES,
        "labeler_counts": {"fable": len(FABLE_BATCHES), "luna": len(LUNA_BATCHES)},
        "luna_output_artifact_sha256_by_batch": {
            str(batch): luna_payloads[batch][1] for batch in LUNA_BATCHES
        },
        "public_index_sha256": verify_sidecar(public_index_path),
        "prepare_script_sha256": sha256_file(Path(__file__).with_name("61_j1_prepare_luna_batches.py")),
        "merge_script_sha256": sha256_file(Path(__file__)),
    }
    if analysis_plan_path is not None and analysis_plan_path.is_file():
        result["mixed_analysis_plan_sha256"] = verify_sidecar(analysis_plan_path,
                                                               expected=EXPECTED_ANALYSIS_PLAN_SHA256)
    result_sha = immutable_json(out_result, result)
    immutable_sidecar(Path(str(out_result) + ".sha256"), result_sha)
    return {
        "status": result["status"],
        "batch_count": N_BATCHES,
        "case_count": N_BATCHES * N_CASES,
        "mixed_checkpoint": str(out_checkpoint),
        "mixed_checkpoint_sha256": mixed_checkpoint_sha,
        "mixed_result": str(out_result),
        "mixed_result_sha256": result_sha,
    }


def _self_test() -> None:
    case = {
        "case_id": "case_abcdefghijklmnopqr",
        "hypothesis": "brief hypothesis",
        "positive_cues": ["cue"],
        "exclusion_cues": [],
        "abstain": False,
        "confidence": 0.5,
    }
    _validate_luna_case(case, case["case_id"], "self-test")
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--jobs", type=Path, default=root / "j1_discovery_labels_jobs_v1.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "j1_discovery_labels_checkpoint_v1.jsonl")
    parser.add_argument("--protocol", type=Path,
                        default=root / "J1_DISCOVERY_LUNA_COMPLETION_PROTOCOL_2026-08-06.md")
    parser.add_argument("--public-index", type=Path,
                        default=root / "j1_luna_public_batches_v2" / "index.json")
    parser.add_argument("--luna-outputs", "--outputs-dir", dest="luna_outputs", type=Path,
                        default=root / "j1_luna_outputs_v2")
    parser.add_argument("--out-checkpoint", type=Path,
                        default=root / "j1_discovery_labels_checkpoint_mixed_v2.jsonl")
    parser.add_argument("--out-result", type=Path,
                        default=root / "j1_discovery_labels_mixed_result_v2.json")
    parser.add_argument("--analysis-plan", type=Path,
                        default=root / "J1_DISCOVERY_MIXED_ANALYSIS_PLAN_2026-08-06.md")
    parser.add_argument(
        "--execution-amendment",
        type=Path,
        default=root / "J1_LUNA_CLI_FALLBACK_AMENDMENT_2026-08-06.md",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="validate all source/Luna artifacts without writing mixed results")
    parser.add_argument("--self-test", action="store_true", help="run stdlib validation self-tests")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        print("62_j1_merge_mixed_labels self-test: PASS")
        if args.dry_run:
            return 0
    # A dry-run still validates all Luna files; write functions are called only
    # after the complete 225-case contract passes.
    if args.dry_run:
        jobs_sha = PREP.verify_sidecar(args.jobs, expected=EXPECTED_JOBS_SHA256)
        checkpoint_sha = PREP.sha256_file(args.checkpoint)
        if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("unexpected v1 checkpoint snapshot SHA")
        protocol_sha = PREP._validate_protocol(args.protocol)
        execution_amendment_sha = validate_execution_amendment(
            args.execution_amendment
        )
        jobs, _ = PREP._validate_jobs(PREP.read_json(args.jobs))
        PREP._validate_checkpoint(PREP.read_jsonl(args.checkpoint), jobs)
        _validate_public_index(args.public_index, jobs, jobs_sha, checkpoint_sha, protocol_sha)
        # Reuse the full merge path with temporary output names only for
        # validation would violate immutable-write expectations; validate Luna
        # files directly instead.
        index_entries = _validate_public_index(args.public_index, jobs, jobs_sha,
                                               checkpoint_sha, protocol_sha)
        names = {entry["output_filename"] for entry in index_entries.values()}
        if {path.name for path in args.luna_outputs.glob("batch_*_output.json")} != names:
            raise ValueError("Luna output set is incomplete")
        for batch, entry in index_entries.items():
            entry = dict(entry)
            entry["expected_agent_task"] = expected_agent_task(batch, entry)
            _validate_luna_output(
                args.luna_outputs / entry["output_filename"], entry
            )
        print(json.dumps({"status": "VALID", "batch_count": N_BATCHES,
                          "case_count": N_BATCHES * N_CASES,
                          "jobs_sha256": jobs_sha,
                          "checkpoint_sha256": checkpoint_sha,
                          "protocol_sha256": protocol_sha,
                          "cli_fallback_amendment_sha256":
                              execution_amendment_sha}, sort_keys=True))
        return 0
    report = merge_mixed_labels(args.jobs, args.checkpoint, args.protocol,
                                args.public_index, args.luna_outputs,
                                args.out_checkpoint, args.out_result,
                                args.analysis_plan,
                                args.execution_amendment)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"62_j1_merge_mixed_labels: FAIL CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
