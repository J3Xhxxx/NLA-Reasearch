"""Freeze the public Luna prompts for the unfinished J1 label batches.

This is deliberately a small, standard-library-only boundary between the
private Fable job freeze and fresh Luna workers.  It reads the immutable
cross-feature jobs and the append-only Fable checkpoint, verifies that only
the first thirteen batches have successful Fable labels, and writes public
prompt envelopes for batches ``13..44``.  The envelopes contain opaque case
IDs and evidence only; the private condition map never crosses this boundary.

All writes are immutable and have conventional ``.sha256`` sidecars.  A
different pre-existing byte sequence is an error rather than an overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
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
CONDITIONS = (
    "SAE_CONTEXT",
    "NLA_ASSISTED",
    "NLA_CONTRASTIVE",
    "NLA_MISMATCHED",
    "NLA_ONLY",
)
EXPECTED_JOBS_SHA256 = (
    "411c67acd230018c60d50194d51c70f3cde847c7f85b38713456450b609f4aad"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "439f8af0c438e8a79a13af5bb6dd463932ca1d56e225912da3dba9edd8cd12e4"
)
EXPECTED_PROTOCOL_SHA256 = (
    "b8653f78accc76b8a3b61a460c812df73cf76ab0111bc202da6bdc1d030081e3"
)
EXPECTED_FABLE_MODEL = "claude-fable-5"
EXPECTED_PROTOCOL_PARENT_HASHES = (
    EXPECTED_JOBS_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    "8f7690f8b12842b32ce5cb32af7ee941b2ce2f71fcc0270768cf0f84edcb50d3",
    "d93d99e3c84b07a6f76b3b4549bb16fbe520f9c7aa59579f98e57bb2a85749a4",
    "638cb1454c4e248cd56e1148f88bc91b0840797ff7ad787f1ef095bd920777cf",
    "75fe0caf64a2e598cd7509ca13ab83c8fc09bcfe43f28c5107e6d5b9097e0da3",
    "132697447b3169dabab373fec2d0647f396424e611d080515f6ae85118ce37f9",
)
CASE_ID_RE = re.compile(r"^case_[a-z0-9]{18}$")
FORBIDDEN_PUBLIC_KEYS = {
    "condition_map",
    "feature",
    "feature_id",
    "arm",
    "condition",
    "condition_id",
    "heldout",
    "held_out",
    "truth",
    "activation",
    "hard_negative",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                   allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def immutable_json(path: Path, value: Any) -> str:
    data = pretty_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_bytes()
        if current != data:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return sha256_bytes(data)


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
    # A digest-only sidecar is accepted by all pipeline readers.  Existing
    # filename-compatible sidecars are deliberately not rewritten.
    data = (digest + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            tokens = path.read_text(encoding="ascii").strip().split()
            if not tokens or tokens[0].lower() != digest:
                raise RuntimeError(f"sha256 sidecar mismatch: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def verify_sidecar(path: Path, *, expected: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"checkpoint row is not an object at line {line_no}")
        rows.append(row)
    return rows


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _batch_id(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be an integer") from exc
    if isinstance(value, float) and result != value:
        raise ValueError(f"{where} must be an integer")
    return result


def _validate_case(case: Any, expected_id: str, where: str) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"{where} is not an object")
    required = {"case_id", "hypothesis", "positive_cues", "exclusion_cues",
                "abstain", "confidence"}
    if set(case) != required:
        raise ValueError(f"{where} has missing or extra fields")
    if case.get("case_id") != expected_id:
        raise ValueError(f"{where} case_id mismatch")
    hypothesis = case.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis.split()) > 32:
        raise ValueError(f"{where} hypothesis must contain at most 32 words")
    for key in ("positive_cues", "exclusion_cues"):
        value = case.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{where} {key} must be an array of strings")
    if type(case.get("abstain")) is not bool:
        raise ValueError(f"{where} abstain must be boolean")
    if not _finite(case.get("confidence")) or not 0 <= float(case["confidence"]) <= 1:
        raise ValueError(f"{where} confidence must be in [0,1]")


def _validate_public_payload(value: Any, where: str) -> None:
    """Reject metadata keys that could deblind a public worker."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{where} contains forbidden private key {key!r}")
            _validate_public_payload(item, where)
    elif isinstance(value, list):
        for item in value:
            _validate_public_payload(item, where)


def _validate_jobs(jobs_doc: Any) -> tuple[dict[int, dict[str, Any]], set[str]]:
    if not isinstance(jobs_doc, dict):
        raise ValueError("frozen jobs must be an object")
    if jobs_doc.get("schema_version") != 1:
        raise ValueError("frozen jobs schema_version must be 1")
    if jobs_doc.get("status") != "EXPLORATORY_LABEL_JOBS_FROZEN":
        raise ValueError("frozen jobs status is not immutable")
    if jobs_doc.get("tools_disabled") is not True or jobs_doc.get("substitution_before_outcome") is not False:
        raise ValueError("frozen jobs invocation provenance is unsafe")
    if jobs_doc.get("conditions") != list(CONDITIONS):
        raise ValueError("frozen jobs condition schedule mismatch")
    jobs = jobs_doc.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != N_BATCHES:
        raise ValueError("frozen jobs must contain exactly 45 jobs")
    by_batch: dict[int, dict[str, Any]] = {}
    all_ids: set[str] = set()
    feature_conditions: dict[int, set[str]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("frozen job is not an object")
        batch = _batch_id(job.get("batch_id"), "job batch_id")
        if batch in by_batch or batch not in range(N_BATCHES):
            raise ValueError(f"invalid or duplicate frozen batch_id {batch}")
        cases = job.get("cases")
        cmap = job.get("condition_map")
        if not isinstance(cases, list) or len(cases) != N_CASES or not isinstance(cmap, dict):
            raise ValueError(f"batch {batch} must contain five cases and a condition_map")
        if not isinstance(job.get("prompt"), str) or not job["prompt"]:
            raise ValueError(f"batch {batch} prompt is missing")
        prompt_sha = job.get("prompt_sha256")
        if not isinstance(prompt_sha, str) or sha256_bytes(job["prompt"].encode("utf-8")) != prompt_sha:
            raise ValueError(f"batch {batch} prompt SHA mismatch")
        seen_batch_ids: set[str] = set()
        seen_features: set[int] = set()
        seen_conditions: set[str] = set()
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
                raise ValueError(f"batch {batch} has malformed opaque case")
            cid = case["case_id"]
            if not CASE_ID_RE.fullmatch(cid):
                raise ValueError(f"batch {batch} has non-opaque case ID {cid!r}")
            if cid in seen_batch_ids or cid in all_ids:
                raise ValueError(f"duplicate opaque case ID {cid}")
            seen_batch_ids.add(cid)
            if set(case) != {"case_id", "evidence"} or not isinstance(case["evidence"], list):
                raise ValueError(f"batch {batch} case {cid} public shape mismatch")
            _validate_public_payload(case["evidence"], f"batch {batch} case {cid}")
            mapped = cmap.get(cid)
            if not isinstance(mapped, dict):
                raise ValueError(f"batch {batch} condition_map omits {cid}")
            try:
                feature = int(mapped["feature"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"batch {batch} private feature mapping malformed") from exc
            condition = str(mapped.get("condition", ""))
            if condition not in CONDITIONS:
                raise ValueError(f"batch {batch} private condition mapping malformed")
            if feature in seen_features or condition in seen_conditions:
                raise ValueError(f"batch {batch} reuses a feature or condition")
            seen_features.add(feature)
            seen_conditions.add(condition)
            all_ids.add(cid)
            feature_conditions.setdefault(feature, set()).add(condition)
        if set(cmap) != seen_batch_ids or seen_conditions != set(CONDITIONS):
            raise ValueError(f"batch {batch} condition_map/cases mismatch")
        by_batch[batch] = job
    if set(by_batch) != set(range(N_BATCHES)) or len(all_ids) != N_BATCHES * N_CASES:
        raise ValueError("frozen jobs do not cover 45 batches and 225 unique cases")
    if any(conditions != set(CONDITIONS) for conditions in feature_conditions.values()):
        raise ValueError("each private feature must have exactly five conditions")
    return by_batch, all_ids


def _parse_cli_cases(raw_cli_json: Any) -> Any:
    if not isinstance(raw_cli_json, str):
        raise ValueError("successful Fable row raw_cli_json is not text")
    try:
        envelope = json.loads(raw_cli_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"successful Fable raw_cli_json is not JSON: {exc}") from exc
    payload = envelope
    if isinstance(envelope, dict) and isinstance(envelope.get("structured_output"), dict):
        payload = envelope["structured_output"]
    elif isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
        try:
            payload = json.loads(envelope["result"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"successful Fable result string is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"cases"} or not isinstance(payload["cases"], list):
        raise ValueError("successful Fable payload lacks exactly a cases array")
    return payload


def _validate_fable_row(row: Mapping[str, Any], job: Mapping[str, Any], batch: int) -> None:
    if row.get("status") != "ok":
        raise ValueError(f"batch {batch} is not successful")
    expected_prompt_sha = job["prompt_sha256"]
    if row.get("prompt_sha256") != expected_prompt_sha or row.get("input_prompt_sha256") != expected_prompt_sha:
        raise ValueError(f"batch {batch} successful row prompt SHA mismatch")
    if row.get("requested_model") != "fable" or row.get("effort") != "low":
        raise ValueError(f"batch {batch} successful row is not the frozen Fable invocation")
    if row.get("tools_disabled") is not True or row.get("substitution_before_outcome") is not False:
        raise ValueError(f"batch {batch} successful row has unsafe invocation provenance")
    resolved = row.get("resolved_model_names")
    if not isinstance(resolved, list) or EXPECTED_FABLE_MODEL not in resolved:
        raise ValueError(f"batch {batch} successful row is not claude-fable-5")
    ids = [str(case["case_id"]) for case in job["cases"]]
    expected_ids = set(ids)
    cases = row.get("cases")
    if not isinstance(cases, list) or len(cases) != N_CASES:
        raise ValueError(f"batch {batch} successful row does not contain five cases")
    got_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError(f"batch {batch} successful row has malformed case")
        cid = case["case_id"]
        if cid in got_ids or cid not in expected_ids:
            raise ValueError(f"batch {batch} successful row case IDs mismatch")
        got_ids.add(cid)
        _validate_case(case, cid, f"Fable batch {batch} case {cid}")
    if got_ids != expected_ids:
        raise ValueError(f"batch {batch} successful row omitted a case")
    parsed = row.get("parsed_structured_result")
    payload = _parse_cli_cases(row.get("raw_cli_json"))
    if not isinstance(parsed, dict) or set(parsed) != {"cases"} or parsed != payload:
        raise ValueError(f"batch {batch} parsed_structured_result disagrees with raw output")
    if payload["cases"] != cases:
        raise ValueError(f"batch {batch} cases disagree with raw output")
    attempts = row.get("attempts")
    raw_sha = sha256_bytes(str(row["raw_cli_json"]).encode("utf-8"))
    if not isinstance(attempts, list) or not any(
            isinstance(attempt, dict) and attempt.get("ok") is True
            and attempt.get("stdout_sha256") == raw_sha for attempt in attempts):
        raise ValueError(f"batch {batch} has no successful attempt matching raw output SHA")


def _validate_checkpoint(rows: Sequence[Mapping[str, Any]], jobs: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    successes: dict[int, dict[str, Any]] = {}
    for line_no, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"checkpoint line {line_no} is not an object")
        batch = _batch_id(row.get("batch_id"), f"checkpoint line {line_no} batch_id")
        if batch not in jobs:
            raise ValueError(f"checkpoint line {line_no} references unknown batch {batch}")
        status = row.get("status")
        if status == "error":
            if not isinstance(row.get("error"), str) or not row["error"]:
                raise ValueError(f"checkpoint error row {batch} lacks error text")
            if batch in successes:
                raise ValueError(f"checkpoint error row follows successful batch {batch}")
            continue
        if status != "ok":
            raise ValueError(f"checkpoint line {line_no} has invalid status {status!r}")
        if batch in successes:
            raise ValueError(f"checkpoint has duplicate successful row for batch {batch}")
        if batch not in FABLE_BATCHES:
            raise ValueError(f"checkpoint has a successful row for Luna batch {batch}")
        _validate_fable_row(row, jobs[batch], batch)
        successes[batch] = dict(row)
    if set(successes) != set(FABLE_BATCHES):
        missing = sorted(set(FABLE_BATCHES) - set(successes))
        raise ValueError(f"checkpoint must have exactly one successful Fable row for batches 0..12; missing {missing}")
    return successes


def _validate_protocol(path: Path) -> str:
    digest = verify_sidecar(path, expected=EXPECTED_PROTOCOL_SHA256)
    text = path.read_text(encoding="utf-8")
    for parent_hash in EXPECTED_PROTOCOL_PARENT_HASHES:
        if parent_hash not in text:
            raise ValueError(f"Luna protocol omits bound parent hash {parent_hash}")
    if "batches `0..12`" not in text or "batches `13..44`" not in text:
        raise ValueError("Luna protocol does not state the frozen Fable/Luna batch split")
    return digest


def _build_public_artifact(job: Mapping[str, Any], batch: int, jobs_sha: str,
                           checkpoint_sha: str, protocol_sha: str) -> dict[str, Any]:
    # Reconstruct the public case objects instead of copying the private job.
    cases = [{"case_id": str(case["case_id"]), "evidence": case["evidence"]}
             for case in job["cases"]]
    for case in cases:
        _validate_public_payload(case, f"public batch {batch}")
    prompt = str(job["prompt"])
    artifact = {
        "schema_version": 2,
        "status": "J1_LUNA_PUBLIC_BATCH_FROZEN_V2",
        "batch_id": batch,
        "cases": cases,
        "case_ids": [case["case_id"] for case in cases],
        "prompt": prompt,
        # This is the embedded prompt-string hash.  Luna workers bind to the
        # immutable envelope file hash published by the index below, avoiding
        # a self-referential field inside this JSON artifact.
        "prompt_sha256": str(job["prompt_sha256"]),
        "jobs_sha256": jobs_sha,
        "checkpoint_sha256": checkpoint_sha,
        "protocol_sha256": protocol_sha,
        "luna_output_filename": f"batch_{batch:02d}_output.json",
        "luna_output_contract": {
            "required_top_level_fields": [
                "batch_id", "cases", "agent_task", "role", "model",
                "reasoning", "public_prompt_sha256",
            ],
            "public_prompt_sha256_semantics": "SHA-256 of immutable batch_XX_prompt.json file (index artifact_sha256)",
            "embedded_prompt_sha256_field": "prompt_sha256",
            "output_artifact_sha256_semantics": "SHA-256 of exact batch_XX_output.json bytes, recorded by its .sha256 sidecar",
            "role": "luna_worker",
            "model": "gpt-5.6-luna",
            "reasoning": "max",
            "case_schema": [
                "case_id", "hypothesis", "positive_cues", "exclusion_cues",
                "abstain", "confidence",
            ],
            "hypothesis_max_words": 32,
        },
    }
    # The public envelope itself is checked recursively.  Private keys are
    # intentionally absent even though the source job retains them.
    _validate_public_payload(artifact, f"public batch {batch}")
    return artifact


def build_public_batches(jobs_path: Path, checkpoint_path: Path, protocol_path: Path,
                         out_dir: Path, out_index: Path) -> dict[str, Any]:
    jobs_sha = verify_sidecar(jobs_path, expected=EXPECTED_JOBS_SHA256)
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected checkpoint snapshot SHA: {checkpoint_sha}")
    protocol_sha = _validate_protocol(protocol_path)
    jobs_doc = read_json(jobs_path)
    jobs, _ = _validate_jobs(jobs_doc)
    rows = read_jsonl(checkpoint_path)
    _validate_checkpoint(rows, jobs)

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for batch in LUNA_BATCHES:
        artifact = _build_public_artifact(jobs[batch], batch, jobs_sha,
                                          checkpoint_sha, protocol_sha)
        path = out_dir / f"batch_{batch:02d}_prompt.json"
        digest = immutable_json(path, artifact)
        immutable_sidecar(Path(str(path) + ".sha256"), digest)
        entries.append({
            "batch_id": batch,
            "prompt_filename": path.name,
            "prompt_sha256": artifact["prompt_sha256"],
            "artifact_sha256": digest,
            "expected_public_prompt_sha256": digest,
            "case_ids": list(artifact["case_ids"]),
            "expected_output_filename": artifact["luna_output_filename"],
            "expected_agent_task": f"/root/luna_batch_{batch:02d}",
            "expected_role": "luna_worker",
            "expected_model": "gpt-5.6-luna",
            "expected_reasoning": "max",
        })
    index = {
        "schema_version": 2,
        "status": "J1_LUNA_PUBLIC_BATCH_INDEX_FROZEN_V2",
        "batch_count": len(entries),
        "batch_ids": list(LUNA_BATCHES),
        "jobs_sha256": jobs_sha,
        "checkpoint_sha256": checkpoint_sha,
        "protocol_sha256": protocol_sha,
        "public_prompt_dir": out_dir.name,
        "batches": entries,
        "output_contract": {
            "filename_pattern": "batch_XX_output.json",
            "required_top_level_fields": [
                "batch_id", "cases", "agent_task", "role", "model",
                "reasoning", "public_prompt_sha256",
            ],
            "public_prompt_sha256_semantics": "SHA-256 of immutable batch_XX_prompt.json file (batches[].artifact_sha256)",
            "embedded_prompt_sha256_field": "prompt_sha256",
            "output_artifact_sha256_semantics": "SHA-256 of exact batch_XX_output.json bytes, recorded by its .sha256 sidecar",
            "role": "luna_worker",
            "model": "gpt-5.6-luna",
            "reasoning": "max",
            "one_fresh_worker_per_batch": True,
            "case_count": 5,
            "hypothesis_max_words": 32,
        },
    }
    index_digest = immutable_json(out_index, index)
    immutable_sidecar(Path(str(out_index) + ".sha256"), index_digest)
    return {"index": str(out_index), "index_sha256": index_digest,
            "batch_count": len(entries), "batch_ids": list(LUNA_BATCHES)}


def _self_test() -> None:
    case = {
        "case_id": "case_abcdefghijklmnopqr",
        "hypothesis": "short hypothesis",
        "positive_cues": ["cue"],
        "exclusion_cues": [],
        "abstain": False,
        "confidence": 0.5,
    }
    _validate_case(case, case["case_id"], "self-test")
    try:
        _validate_public_payload({"condition_map": {}}, "self-test")
    except ValueError:
        pass
    else:
        raise AssertionError("private public-payload key was accepted")
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--jobs", type=Path, default=root / "j1_discovery_labels_jobs_v1.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "j1_discovery_labels_checkpoint_v1.jsonl")
    parser.add_argument("--protocol", type=Path,
                        default=root / "J1_DISCOVERY_LUNA_COMPLETION_PROTOCOL_2026-08-06.md")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path,
                        default=root / "j1_luna_public_batches_v2")
    parser.add_argument("--out-index", type=Path, default=None,
                        help="index path (default: <out-dir>/index.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate all inputs without writing public artifacts")
    parser.add_argument("--self-test", action="store_true", help="run stdlib validation self-tests")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        print("61_j1_prepare_luna_batches self-test: PASS")
        if args.dry_run:
            return 0
    index = args.out_index or (args.out_dir / "index.json")
    if args.dry_run:
        # Validate without creating output directories or files.
        jobs_sha = verify_sidecar(args.jobs, expected=EXPECTED_JOBS_SHA256)
        checkpoint_sha = sha256_file(args.checkpoint)
        if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(f"unexpected checkpoint snapshot SHA: {checkpoint_sha}")
        protocol_sha = _validate_protocol(args.protocol)
        jobs, _ = _validate_jobs(read_json(args.jobs))
        _validate_checkpoint(read_jsonl(args.checkpoint), jobs)
        print(json.dumps({"status": "VALID", "jobs_sha256": jobs_sha,
                          "checkpoint_sha256": checkpoint_sha,
                          "protocol_sha256": protocol_sha,
                          "public_batch_ids": list(LUNA_BATCHES)},
                         sort_keys=True))
        return 0
    report = build_public_batches(args.jobs, args.checkpoint, args.protocol,
                                  args.out_dir, index)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"61_j1_prepare_luna_batches: FAIL CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
