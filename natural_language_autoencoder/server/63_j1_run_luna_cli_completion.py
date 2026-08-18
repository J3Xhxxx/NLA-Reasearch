#!/usr/bin/env python3
"""Complete J1 label batches 27..44 with isolated Luna Max CLI workers.

This runner is the fail-closed implementation of
``J1_LUNA_CLI_FALLBACK_AMENDMENT_2026-08-06.md``.  It never reads the private
condition map.  One fresh, ephemeral Codex CLI process receives exactly one
already-frozen public batch.  Parsed cases are copied into the existing Luna
output directory only after schema, ID, isolation, and provenance checks pass.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping


MODEL = "gpt-5.6-luna"
REASONING = "max"
ROLE = "luna_worker"
CLI_VERSION = "codex-cli 0.146.1"
CLI_BATCHES = tuple(range(27, 45))
MAX_ATTEMPTS = 2
EXPECTED_INDEX_SHA256 = (
    "748cf63b772f73d2f8652bc4f0c6434079ce72d2399b74719f1eba935ea759cb"
)
EXPECTED_SCHEMA_SHA256 = (
    "1812dfd39f9bc89a27d2cb04e28921f3a3d443b78b8baef2eda5b2783e797134"
)
EXPECTED_AMENDMENT_SHA256 = (
    "7c7babec80ba123f640a82ca1b4b5d51648d28f3abd3c6a03314092880314114"
)
CASE_ID_RE = re.compile(r"^case_[a-z0-9]{18}$")
ALLOWED_EVENT_TYPES = {"thread.started", "turn.started", "turn.completed"}
ALLOWED_ITEM_TYPES = {"agent_message", "reasoning"}


def _load_prepare_module() -> Any:
    path = Path(__file__).with_name("61_j1_prepare_luna_batches.py")
    spec = importlib.util.spec_from_file_location("j1_prepare_luna_batches", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load public-boundary module: {path}")
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


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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


def immutable_json(path: Path, value: Any) -> str:
    return immutable_bytes(path, pretty_bytes(value))


def immutable_sidecar(path: Path, digest: str) -> None:
    data = (digest + "\n").encode("ascii")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"sidecar differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def verify_sidecar(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing sidecar: {sidecar}")
    declared = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    if declared != digest:
        raise ValueError(f"sidecar mismatch for {path}: {declared} != {digest}")
    if expected is not None and digest != expected:
        raise ValueError(f"unexpected SHA-256 for {path}: {digest} != {expected}")
    return digest


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def worker_id(batch: int) -> str:
    return f"codex-cli://j1/luna_batch_{batch:02d}"


def _validate_case(case: Any, expected_ids: set[str], where: str) -> str:
    if not isinstance(case, dict):
        raise ValueError(f"{where} is not an object")
    required = {
        "case_id",
        "hypothesis",
        "positive_cues",
        "exclusion_cues",
        "abstain",
        "confidence",
    }
    if set(case) != required:
        raise ValueError(f"{where} has missing/extra fields")
    case_id = case.get("case_id")
    if (
        not isinstance(case_id, str)
        or not CASE_ID_RE.fullmatch(case_id)
        or case_id not in expected_ids
    ):
        raise ValueError(f"{where} has an unexpected case_id")
    hypothesis = case.get("hypothesis")
    if (
        not isinstance(hypothesis, str)
        or not hypothesis.strip()
        or len(hypothesis.split()) > 32
    ):
        raise ValueError(f"{where} hypothesis must be nonempty and <=32 words")
    for key in ("positive_cues", "exclusion_cues"):
        cues = case.get(key)
        if not isinstance(cues, list) or not all(
            isinstance(item, str) and item.strip() for item in cues
        ):
            raise ValueError(f"{where} {key} must contain nonempty strings")
    if type(case.get("abstain")) is not bool:
        raise ValueError(f"{where} abstain is not boolean")
    confidence = case.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"{where} confidence is outside [0,1]")
    return case_id


def validate_cases(payload: Any, expected_ids: set[str], where: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"cases"}:
        raise ValueError(f"{where} must contain only a cases array")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError(f"{where} must contain exactly five cases")
    seen = {_validate_case(case, expected_ids, f"{where} case") for case in cases}
    if seen != expected_ids or len(seen) != 5:
        raise ValueError(f"{where} case IDs are incomplete or duplicated")
    return [dict(case) for case in cases]


def resolve_codex(requested: str) -> dict[str, Any]:
    is_windows = sys.platform.startswith("win")
    candidates = ["codex.exe", "codex.cmd", "codex"] if (
        is_windows and requested == "codex"
    ) else [requested]
    resolved: str | None = None
    for candidate in candidates:
        found = shutil.which(candidate)
        if found and Path(found).suffix.lower() != ".ps1":
            resolved = str(Path(found).resolve())
            break
        if is_windows and not Path(candidate).is_absolute():
            for directory in os.environ.get("PATH", "").split(os.pathsep):
                literal = Path(directory) / candidate
                if literal.is_file() and literal.suffix.lower() != ".ps1":
                    resolved = str(literal.resolve())
                    break
        if resolved:
            break
    if resolved is None:
        raise FileNotFoundError(f"cannot resolve {requested!r}")
    if is_windows and Path(resolved).suffix.lower() in {".cmd", ".bat"}:
        comspec = os.environ.get("COMSPEC") or shutil.which("cmd.exe") or "cmd.exe"
        prefix = [comspec, "/d", "/s", "/c", resolved]
        kind = "cmd_wrapper_shell_false"
    else:
        prefix = [resolved]
        kind = "native_executable"
    return {"requested": requested, "resolved": resolved, "prefix": prefix, "kind": kind}


def cli_version(command: Mapping[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command["prefix"]) + ["--version"],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    value = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or value != CLI_VERSION:
        raise ValueError(
            f"Codex CLI version mismatch: rc={completed.returncode} value={value!r}"
        )
    return {
        "version": value,
        "resolved": command["resolved"],
        "launch_kind": command["kind"],
    }


def validate_public_inputs(
    index_path: Path, schema_path: Path, amendment_path: Path
) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    index_sha = verify_sidecar(index_path, EXPECTED_INDEX_SHA256)
    schema_sha = verify_sidecar(schema_path, EXPECTED_SCHEMA_SHA256)
    amendment_sha = verify_sidecar(amendment_path, EXPECTED_AMENDMENT_SHA256)
    amendment = amendment_path.read_text(encoding="utf-8")
    for batch in CLI_BATCHES:
        if worker_id(batch) not in amendment and (
            "`codex-cli://j1/luna_batch_XX`" not in amendment
        ):
            raise ValueError(f"CLI amendment omits worker contract for batch {batch}")
    index = read_json(index_path)
    if not isinstance(index, dict) or not str(index.get("status", "")).startswith(
        "J1_LUNA_PUBLIC_BATCH_INDEX_FROZEN"
    ):
        raise ValueError("public index status is not frozen")
    entries: dict[int, dict[str, Any]] = {}
    for row in index.get("batches", []):
        if not isinstance(row, dict) or "batch_id" not in row:
            raise ValueError("malformed public index row")
        batch = int(row["batch_id"])
        if batch not in CLI_BATCHES:
            continue
        if batch in entries:
            raise ValueError(f"duplicate public index batch {batch}")
        artifact_path = index_path.parent / str(row["prompt_filename"])
        artifact_sha = verify_sidecar(artifact_path)
        if artifact_sha != row.get("artifact_sha256"):
            raise ValueError(f"public artifact SHA mismatch batch {batch}")
        artifact = read_json(artifact_path)
        if not isinstance(artifact, dict):
            raise ValueError(f"public batch {batch} is not an object")
        PREP._validate_public_payload(artifact, f"CLI public batch {batch}")
        if int(artifact.get("batch_id", -1)) != batch:
            raise ValueError(f"public batch ID mismatch {batch}")
        prompt = artifact.get("prompt")
        if (
            not isinstance(prompt, str)
            or sha256_bytes(prompt.encode("utf-8")) != artifact.get("prompt_sha256")
        ):
            raise ValueError(f"public prompt hash mismatch batch {batch}")
        cases = artifact.get("cases")
        if not isinstance(cases, list) or len(cases) != 5:
            raise ValueError(f"public batch {batch} must contain five cases")
        expected_ids = {
            str(case["case_id"])
            for case in cases
            if isinstance(case, dict) and isinstance(case.get("case_id"), str)
        }
        if len(expected_ids) != 5 or expected_ids != set(artifact.get("case_ids", [])):
            raise ValueError(f"public case IDs malformed batch {batch}")
        entries[batch] = {
            "batch": batch,
            "artifact_path": artifact_path,
            "artifact_sha256": artifact_sha,
            "prompt_sha256": artifact["prompt_sha256"],
            "prompt": prompt,
            "cases": cases,
            "case_ids": expected_ids,
        }
    if set(entries) != set(CLI_BATCHES):
        raise ValueError("public index does not cover CLI batches 27..44")
    return entries, {
        "index_sha256": index_sha,
        "schema_sha256": schema_sha,
        "amendment_sha256": amendment_sha,
    }


def build_model_prompt(entry: Mapping[str, Any]) -> str:
    public_payload = {
        "batch_id": entry["batch"],
        "cases": entry["cases"],
        "instruction": entry["prompt"],
    }
    return (
        "You are a fresh isolated Luna Max execution worker. This is one "
        "exploratory public labeling batch. Do not use tools, browse, read "
        "files, execute commands, or infer hidden feature IDs, arms, truth, "
        "activations, held-out data, or grouping. Use only PUBLIC_BATCH. "
        "Return exactly one JSON object matching the supplied schema, with "
        "all five opaque case IDs exactly once. Hypotheses must be nonempty "
        "and at most 32 whitespace-separated words. No prose outside JSON.\n"
        "PUBLIC_BATCH="
        + json.dumps(public_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def parse_events(stdout: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"non-JSON CLI event line {line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"CLI event line {line_number} is not an object")
        events.append(event)
        event_type = str(event.get("type", ""))
        if event_type.startswith("item."):
            item = event.get("item")
            item_type = str(item.get("type", "")) if isinstance(item, dict) else ""
            if item_type not in ALLOWED_ITEM_TYPES:
                forbidden.append(
                    {"line": line_number, "event_type": event_type, "item_type": item_type}
                )
        elif event_type not in ALLOWED_EVENT_TYPES:
            forbidden.append(
                {"line": line_number, "event_type": event_type, "item_type": None}
            )
    if not events:
        raise ValueError("CLI emitted no JSON events")
    return events, forbidden


def next_attempt_number(raw_dir: Path, batch: int) -> int:
    values: list[int] = []
    pattern = re.compile(rf"^batch_{batch:02d}_attempt_(\d+)\.json$")
    if raw_dir.is_dir():
        for path in raw_dir.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                values.append(int(match.group(1)))
    return max(values, default=0) + 1


def run_batch(
    entry: Mapping[str, Any],
    *,
    command: Mapping[str, Any],
    schema_path: Path,
    outputs_dir: Path,
    raw_dir: Path,
    manifest_sha: str,
    script_sha: str,
    cli_meta: Mapping[str, Any],
    timeout: float,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    batch = int(entry["batch"])
    output_path = outputs_dir / f"batch_{batch:02d}_output.json"
    if output_path.exists():
        output_sha = verify_sidecar(output_path)
        existing = read_json(output_path)
        cases = validate_cases(
            {"cases": existing.get("cases")} if isinstance(existing, dict) else existing,
            set(entry["case_ids"]),
            f"existing batch {batch}",
        )
        if (
            existing.get("batch_id") != batch
            or existing.get("agent_task") != worker_id(batch)
            or existing.get("role") != ROLE
            or existing.get("model") != MODEL
            or existing.get("reasoning") != REASONING
            or existing.get("public_prompt_sha256") != entry["artifact_sha256"]
        ):
            raise ValueError(f"existing CLI output provenance mismatch batch {batch}")
        return {
            "batch_id": batch,
            "status": "existing_valid",
            "output_sha256": output_sha,
            "case_count": len(cases),
        }

    prompt = build_model_prompt(entry)
    prompt_sha = sha256_bytes(prompt.encode("utf-8"))
    attempt_start = next_attempt_number(raw_dir, batch)
    attempt_summaries: list[dict[str, Any]] = []
    for offset in range(MAX_ATTEMPTS):
        attempt = attempt_start + offset
        with tempfile.TemporaryDirectory(prefix=f"j1_luna_b{batch:02d}_empty_") as temp_name:
            temp_root = Path(temp_name)
            final_path = temp_root / "final.json"
            argv = list(command["prefix"]) + [
                "exec",
                "-",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "-C",
                str(temp_root),
                "-m",
                MODEL,
                "-c",
                'model_reasoning_effort="max"',
                "-c",
                'service_tier="fast"',
                "--output-schema",
                str(schema_path),
                "--json",
                "-o",
                str(final_path),
            ]
            try:
                completed = subprocess.run(
                    argv,
                    input=prompt,
                    shell=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
                raw_final = (
                    final_path.read_text(encoding="utf-8", errors="strict")
                    if final_path.is_file()
                    else ""
                )
                events: list[dict[str, Any]] = []
                forbidden: list[dict[str, Any]] = []
                event_error: str | None = None
                try:
                    events, forbidden = parse_events(completed.stdout or "")
                except Exception as exc:
                    event_error = repr(exc)
                raw_record = {
                    "schema_version": 1,
                    "batch_id": batch,
                    "attempt": attempt,
                    "model": MODEL,
                    "reasoning": REASONING,
                    "agent_task": worker_id(batch),
                    "manifest_sha256": manifest_sha,
                    "script_sha256": script_sha,
                    "public_prompt_sha256": entry["artifact_sha256"],
                    "embedded_prompt_sha256": entry["prompt_sha256"],
                    "model_input_sha256": prompt_sha,
                    "argv": argv,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout or "",
                    "stderr": completed.stderr or "",
                    "raw_final_json": raw_final,
                    "raw_final_json_sha256": sha256_bytes(raw_final.encode("utf-8")),
                    "events_parsed": len(events),
                    "forbidden_tool_events": forbidden,
                    "event_error": event_error,
                    "cli": dict(cli_meta),
                }
            except Exception as exc:
                raw_record = {
                    "schema_version": 1,
                    "batch_id": batch,
                    "attempt": attempt,
                    "model": MODEL,
                    "reasoning": REASONING,
                    "agent_task": worker_id(batch),
                    "manifest_sha256": manifest_sha,
                    "script_sha256": script_sha,
                    "public_prompt_sha256": entry["artifact_sha256"],
                    "embedded_prompt_sha256": entry["prompt_sha256"],
                    "model_input_sha256": prompt_sha,
                    "argv": argv,
                    "exception": repr(exc),
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "raw_final_json": "",
                    "raw_final_json_sha256": sha256_bytes(b""),
                    "events_parsed": 0,
                    "forbidden_tool_events": [],
                    "event_error": None,
                    "cli": dict(cli_meta),
                }
            raw_path = raw_dir / f"batch_{batch:02d}_attempt_{attempt:02d}.json"
            raw_sha = immutable_json(raw_path, raw_record)
            immutable_sidecar(Path(str(raw_path) + ".sha256"), raw_sha)
            attempt_summaries.append(
                {
                    "attempt": attempt,
                    "raw_artifact": raw_path.name,
                    "raw_artifact_sha256": raw_sha,
                    "returncode": raw_record.get("returncode"),
                }
            )
            failure: str | None = None
            if raw_record.get("returncode") != 0:
                failure = f"Codex return code {raw_record.get('returncode')}"
            elif raw_record.get("event_error"):
                failure = f"event parse failure: {raw_record['event_error']}"
            elif raw_record.get("forbidden_tool_events"):
                failure = "tool/non-message item event detected"
            else:
                try:
                    parsed = json.loads(str(raw_record["raw_final_json"]))
                    cases = validate_cases(
                        parsed, set(entry["case_ids"]), f"batch {batch} attempt {attempt}"
                    )
                except Exception as exc:
                    failure = f"structured output invalid: {exc}"
            if failure is not None:
                attempt_summaries[-1]["failure"] = failure
                with print_lock:
                    print(
                        f"batch={batch} attempt={attempt} FAIL {failure}",
                        flush=True,
                    )
                continue
            provenance = {
                "agent_task": worker_id(batch),
                "role": ROLE,
                "model": MODEL,
                "reasoning": REASONING,
                "transport": "codex_cli_ephemeral",
                "codex_cli_version": CLI_VERSION,
                "manifest_sha256": manifest_sha,
                "script_sha256": script_sha,
                "raw_attempt_artifact": raw_path.name,
                "raw_attempt_artifact_sha256": raw_sha,
                "model_input_sha256": prompt_sha,
                "tool_calls_detected": False,
            }
            output = {
                "schema_version": 1,
                "status": "ok",
                "batch_id": batch,
                "agent_task": worker_id(batch),
                "role": ROLE,
                "model": MODEL,
                "reasoning": REASONING,
                "labeler": "luna",
                "public_prompt_sha256": entry["artifact_sha256"],
                "prompt_sha256": entry["prompt_sha256"],
                "cases": cases,
                "provenance": provenance,
            }
            output_sha = immutable_json(output_path, output)
            immutable_sidecar(Path(str(output_path) + ".sha256"), output_sha)
            with print_lock:
                print(
                    f"batch={batch} OK output_sha256={output_sha}",
                    flush=True,
                )
            return {
                "batch_id": batch,
                "status": "ok",
                "output_sha256": output_sha,
                "raw_attempt_artifact_sha256": raw_sha,
                "attempts": attempt_summaries,
            }
    raise RuntimeError(
        f"batch {batch} exhausted {MAX_ATTEMPTS} fail-closed attempts: "
        f"{attempt_summaries}"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-index",
        type=Path,
        default=results / "j1_luna_public_batches_v2" / "index.json",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=results / "j1_luna_cli_output_schema_v1.json",
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=results / "J1_LUNA_CLI_FALLBACK_AMENDMENT_2026-08-06.md",
    )
    parser.add_argument(
        "--outputs-dir", type=Path, default=results / "j1_luna_outputs_v2"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=results / "j1_luna_cli_raw_v1"
    )
    parser.add_argument(
        "--manifest", type=Path, default=results / "j1_luna_cli_manifest_v1.json"
    )
    parser.add_argument(
        "--completion",
        type=Path,
        default=results / "j1_luna_cli_completion_v1.json",
    )
    parser.add_argument(
        "--codex-command",
        default="codex.cmd" if sys.platform.startswith("win") else "codex",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1 or args.concurrency > 3 or args.timeout <= 0:
        raise ValueError("concurrency must be 1..3 and timeout must be positive")
    entries, hashes = validate_public_inputs(
        args.public_index, args.schema, args.amendment
    )
    command = resolve_codex(args.codex_command)
    cli_meta = cli_version(command)
    script_sha = sha256_file(Path(__file__))
    manifest = {
        "schema_version": 1,
        "status": "J1_LUNA_CLI_COMPLETION_MANIFEST_FROZEN",
        "claim_scope": "exploratory_labels_only_no_scientific_analysis",
        "batches": list(CLI_BATCHES),
        "model": MODEL,
        "reasoning": REASONING,
        "role": ROLE,
        "codex_cli": cli_meta,
        "script_sha256": script_sha,
        **hashes,
        "contract": {
            "one_fresh_ephemeral_process_per_batch": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "sandbox": "read-only",
            "working_directory": "fresh_empty_temporary_directory",
            "max_attempts_parse_only": MAX_ATTEMPTS,
            "no_tool_events_allowed": True,
            "concurrency": args.concurrency,
            "timeout_seconds": args.timeout,
            "agent_task_by_batch": {
                str(batch): worker_id(batch) for batch in CLI_BATCHES
            },
            "public_prompt_artifact_sha256_by_batch": {
                str(batch): entries[batch]["artifact_sha256"]
                for batch in CLI_BATCHES
            },
        },
    }
    manifest_sha = immutable_json(args.manifest, manifest)
    immutable_sidecar(Path(str(args.manifest) + ".sha256"), manifest_sha)
    print(
        f"manifest={args.manifest} sha256={manifest_sha} batches=27..44",
        flush=True,
    )
    if args.dry_run:
        print("dry-run: no Luna model calls or label outputs", flush=True)
        return 0
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    print_lock = threading.Lock()
    results: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        future_by_batch = {
            executor.submit(
                run_batch,
                entries[batch],
                command=command,
                schema_path=args.schema.resolve(),
                outputs_dir=args.outputs_dir,
                raw_dir=args.raw_dir,
                manifest_sha=manifest_sha,
                script_sha=script_sha,
                cli_meta=cli_meta,
                timeout=args.timeout,
                print_lock=print_lock,
            ): batch
            for batch in CLI_BATCHES
        }
        for future in concurrent.futures.as_completed(future_by_batch):
            batch = future_by_batch[future]
            try:
                results[batch] = future.result()
            except Exception as exc:
                failures[batch] = repr(exc)
                with print_lock:
                    print(f"batch={batch} TERMINAL_FAIL {exc!r}", flush=True)
    if failures or set(results) != set(CLI_BATCHES):
        raise RuntimeError(
            f"CLI completion fail-closed; completed={sorted(results)} "
            f"failures={failures}"
        )
    completion = {
        "schema_version": 1,
        "status": "J1_LUNA_CLI_COMPLETION_COMPLETE",
        "claim_scope": "exploratory_labels_only_no_scientific_analysis",
        "manifest_sha256": manifest_sha,
        "script_sha256": script_sha,
        "batch_count": len(results),
        "case_count": len(results) * 5,
        "batches": {str(batch): results[batch] for batch in sorted(results)},
    }
    completion_sha = immutable_json(args.completion, completion)
    immutable_sidecar(Path(str(args.completion) + ".sha256"), completion_sha)
    print(
        f"COMPLETE batches={len(results)} cases={len(results) * 5} "
        f"completion_sha256={completion_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"63_j1_run_luna_cli_completion: FAIL CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
