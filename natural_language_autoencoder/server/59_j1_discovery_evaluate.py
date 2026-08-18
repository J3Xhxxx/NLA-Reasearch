#!/usr/bin/env python3
"""Blinded held-out evaluator for the exploratory J1 discovery pilot.

This is a CPU-only, standard-library runner for the artefacts produced by
``57_j1_discovery_pilot_gpu.py`` and the label-job runner (58).  It deliberately
does not import a model package.  The only model invocation is an explicit,
read-only Codex CLI subprocess, one request per feature, after an immutable
blinded request file has been written.

The evaluator is a measurement instrument.  The binary endpoint is read only
from the frozen SAE activations and embedded hard-negative records; no value
returned by the evaluator is used as truth.  Any malformed input, parser
omission, incomplete request, or evaluator failure leaves a retained failure
record and produces ``NO_COMPLETE_ANALYSIS`` rather than deleting an
unfavourable row.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SEED = 20260806
BOOTSTRAP_REPS = 20_000
N_FEATURES = 45
N_CONTEXTS = 8
N_HELDOUT_POSITIVE = 4
N_HARD_NEGATIVE = 4
ARMS = (
    "SAE_CONTEXT",
    "NLA_ASSISTED",
    "NLA_CONTRASTIVE",
    "NLA_MISMATCHED",
    "NLA_ONLY",
)
STRATA = ("source_concentrated", "source_distributed", "language_selective")
EVALUATOR_MODEL = "gpt-5.6-terra"
MAX_EVIDENCE_CODE = 120
CODEX_CONFIG_OVERRIDES = ("-c", 'model_reasoning_effort="xhigh"',
                          "-c", 'service_tier="fast"')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_sha(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_immutable(path: Path, value: Any) -> str:
    """Write an immutable JSON artifact, refusing to replace different bytes."""
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


def write_sidecar(path: Path, digest: str) -> None:
    data = (digest + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"sha256 sidecar mismatch: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def verify_sidecar(path: Path, *, required: bool = True) -> str:
    """Verify a conventional ``.sha256`` sidecar and return the file digest."""
    actual = sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.exists():
        if required:
            raise ValueError(f"missing immutable SHA-256 sidecar: {sidecar}")
        return actual
    declared = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    if declared != actual:
        raise ValueError(f"SHA-256 sidecar mismatch for {path}: {declared} != {actual}")
    return actual


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_no}")
            rows.append(row)
    return rows


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _as_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:  # bool is intentionally not accepted as an int
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and value != result:
        raise ValueError(f"{name} must be an integer")
    return result


def _as_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _status_is_exploratory(payload: dict[str, Any], what: str,
                           accepted_fragments: Sequence[str]) -> None:
    if payload.get("confirmatory") is True:
        raise ValueError(f"{what} is marked confirmatory")
    status = str(payload.get("status", ""))
    if not status or "EXPLORATORY" not in status.upper():
        raise ValueError(f"{what} does not have an exploratory status: {status!r}")
    upper = status.upper()
    if not any(fragment in upper for fragment in accepted_fragments):
        raise ValueError(f"{what} status is not complete/frozen: {status!r}")
    if "SMOKE" in upper or "PARTIAL" in upper:
        raise ValueError(f"{what} is a smoke/partial artifact: {status!r}")


def _nested_hash(payload: dict[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return value.lower()
    for container_name in ("inputs", "bindings", "contracts", "provenance"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            for name in names:
                value = container.get(name)
                if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                    return value.lower()
    return None


def _activation(context: dict[str, Any]) -> float:
    # ``expected_activation`` is the N3 top value retained by 57; actual SAE
    # activation aliases are accepted when a later freeze records them.
    for key in ("actual_activation", "sae_activation", "target_activation",
                "activation", "expected_activation"):
        if key in context and context[key] is not None:
            return _as_finite(context[key], f"context[{key}]")
    raise ValueError("held-out positive has no activation retained in freeze")


def _physical(row: dict[str, Any]) -> tuple[int, int]:
    return (_as_int(row.get("doc_id"), "doc_id"), _as_int(row.get("position"), "position"))


def _context_text(row: dict[str, Any]) -> str:
    # Freeze rows normally carry context_text/before/token/after.  Text is
    # intentionally reconstructed without source, role, arm, or activation.
    for key in ("context_text", "text", "window"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    before = str(row.get("before", ""))
    token = str(row.get("token", row.get("marked_token", "")))
    after = str(row.get("after", ""))
    text = (before + " " + token + " " + after).strip()
    if not text:
        raise ValueError(f"context {_physical(row)} has no text/window")
    return text


def _marked_text(row: dict[str, Any]) -> str:
    text = _context_text(row)
    token = str(row.get("token", row.get("marked_token", ""))).strip()
    # If 57 retained before/after, mark the token explicitly.  For an already
    # marked context_text, avoid trying to identify the token by substring.
    if token and "<TARGET_TOKEN>" not in text:
        before = str(row.get("before", "")).strip()
        after = str(row.get("after", "")).strip()
        if before or after:
            return f"{before} <TARGET_TOKEN> {token} </TARGET_TOKEN> {after}".strip()
    marked = text.replace("<TARGET>", "<TARGET_TOKEN>")
    if "<TARGET_TOKEN>" not in marked:
        # A valid 57 freeze normally carries before/token/after.  Preserve a
        # visible marker even for a legacy text-only row rather than silently
        # asking the evaluator about an unmarked position.
        marked = "<TARGET_TOKEN> " + marked + " </TARGET_TOKEN>"
    return marked


def validate_freeze(freeze: dict[str, Any], freeze_sha: str) -> tuple[list[dict[str, Any]], dict[int, str], list[str]]:
    """Validate the 45x(4 positive + 4 hard-negative) truth contract."""
    _status_is_exploratory(freeze, "J1 freeze", ("FROZEN",))
    features = freeze.get("features")
    if not isinstance(features, list) or len(features) != N_FEATURES:
        raise ValueError(f"J1 freeze must contain exactly {N_FEATURES} features")
    strata_count = {s: 0 for s in STRATA}
    seen_features: set[int] = set()
    normalized: list[dict[str, Any]] = []
    stratum_by_feature: dict[int, str] = {}
    risks: list[str] = []
    global_negatives: dict[tuple[int, int], int] = {}
    seen_context_ids: set[Any] = set()
    for feature_row in features:
        if not isinstance(feature_row, dict):
            raise ValueError("malformed J1 feature row")
        feature = _as_int(feature_row.get("feature"), "feature")
        if feature in seen_features:
            raise ValueError(f"duplicate feature {feature}")
        seen_features.add(feature)
        stratum = str(feature_row.get("stratum", ""))
        if stratum not in strata_count:
            raise ValueError(f"unexpected feature stratum {stratum!r}")
        strata_count[stratum] += 1
        stratum_by_feature[feature] = stratum
        discovery = feature_row.get("discovery")
        heldout = feature_row.get("heldout_positive")
        if not isinstance(discovery, list) or len(discovery) != 4:
            raise ValueError(f"feature {feature} must have exactly four discovery contexts")
        if not isinstance(heldout, list) or len(heldout) != 4:
            raise ValueError(f"feature {feature} must have exactly four held-out positives")
        all_contexts = discovery + heldout
        physicals = [_physical(row) for row in all_contexts]
        if len(set(physicals)) != 8:
            raise ValueError(f"feature {feature} has duplicate physical context positions")
        discovery_docs = {_as_int(row.get("doc_id"), "doc_id") for row in discovery}
        heldout_docs = {_as_int(row.get("doc_id"), "doc_id") for row in heldout}
        if discovery_docs & heldout_docs:
            raise ValueError(f"feature {feature} discovery/held-out documents overlap")
        negative_positions: set[tuple[int, int]] = set()
        # The external evaluator sees the held-out set only: four positive
        # contexts and their four embedded hard negatives.  Discovery rows are
        # validated above but are never part of a held-out score.
        contexts: list[dict[str, Any]] = []
        for index, row in enumerate(discovery):
            if not isinstance(row, dict):
                raise ValueError(f"feature {feature} malformed discovery row")
            if "context_index" in row:
                ci = row["context_index"]
                if ci in seen_context_ids:
                    raise ValueError(f"duplicate context_index {ci}")
                seen_context_ids.add(ci)
            # Discovery contexts are intentionally omitted from the evaluator
            # condition map; they are used only to enforce document-disjointness.
        for index, positive in enumerate(heldout):
            if not isinstance(positive, dict):
                raise ValueError(f"feature {feature} malformed held-out row")
            positive_activation = _activation(positive)
            if not positive_activation > 0.0:
                raise ValueError(f"feature {feature} held-out activation is not >0")
            negative = positive.get("hard_negative")
            if not isinstance(negative, dict):
                raise ValueError(f"feature {feature} held-out row lacks embedded hard_negative")
            negative_activation = _as_finite(negative.get("target_activation"), "target_activation")
            if negative_activation != 0.0:
                raise ValueError(f"feature {feature} hard negative target_activation is not exactly zero")
            neg_physical = _physical(negative)
            if neg_physical in negative_positions:
                raise ValueError(f"feature {feature} reuses a physical hard negative")
            negative_positions.add(neg_physical)
            global_negatives[neg_physical] = global_negatives.get(neg_physical, 0) + 1
            if neg_physical[0] in discovery_docs or neg_physical[0] in heldout_docs:
                raise ValueError(f"feature {feature} hard negative reuses a discovery/positive document")
            if _as_int(negative.get("preference_tier", -1), "preference_tier") not in (0, 1, 2):
                raise ValueError(f"feature {feature} hard negative has invalid preference tier")
            contexts.append({"feature": feature, "stratum": stratum, "role": "heldout_positive",
                             "index": index, "row": dict(positive), "truth": 1,
                             "marked_text": _marked_text(positive)})
            contexts.append({"feature": feature, "stratum": stratum, "role": "hard_negative",
                             "index": index, "row": dict(negative), "truth": 0,
                             "marked_text": _marked_text(negative)})
        normalized.append({"feature": feature, "stratum": stratum,
                           "discovery": discovery, "heldout": heldout,
                           "contexts": contexts})
    if strata_count != {s: 15 for s in STRATA}:
        raise ValueError(f"J1 freeze must contain 15 features per stratum, got {strata_count}")
    if global_negatives:
        repeated = sum(1 for count in global_negatives.values() if count > 1)
        if repeated:
            risks.append(f"{repeated} physical hard-negative positions are reused across features; allowed per-feature, report as risk")
    return normalized, stratum_by_feature, risks


def validate_av(av: dict[str, Any], freeze_sha: str,
                features: Sequence[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    _status_is_exploratory(av, "full AV result", ("COMPLETE",))
    bound = _nested_hash(av, ("freeze_sha256", "j1_freeze_sha256"))
    if bound != freeze_sha:
        raise ValueError(f"AV result freeze SHA binding mismatch: {bound} != {freeze_sha}")
    rows = av.get("rows")
    if not isinstance(rows, list):
        raise ValueError("AV result lacks rows")
    feature_ids = {int(row["feature"]) for row in features}
    expected: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("AV result row is not an object")
        if "feature" not in row or "discovery_index" not in row:
            continue
        feature = _as_int(row.get("feature"), "AV feature")
        if feature not in feature_ids:
            raise ValueError(f"AV row references feature outside freeze: {feature}")
        di = _as_int(row.get("discovery_index"), "AV discovery_index")
        if di not in range(4):
            raise ValueError("AV discovery_index outside 0..3")
        arm = str(row.get("arm", ""))
        if arm not in ("NLA_RAW", "NLA_CONTRASTIVE"):
            continue
        explanation = row.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"empty AV explanation feature={feature} arm={arm} di={di}")
        key = (feature, di, arm)
        if key in expected:
            raise ValueError(f"duplicate AV row {key}")
        expected[key] = dict(row)
    for feature_row in features:
        feature = int(feature_row["feature"])
        for di in range(4):
            for arm in ("NLA_RAW", "NLA_CONTRASTIVE"):
                key = (feature, di, arm)
                if key not in expected:
                    raise ValueError(f"full AV result omitted {key}")
                row = expected[key]
                ctx = feature_row["discovery"][di]
                if _as_int(row.get("doc_id"), "AV doc_id") != _as_int(ctx.get("doc_id"), "freeze doc_id"):
                    raise ValueError(f"AV row {key} does not bind discovery document")
                if _as_int(row.get("position"), "AV position") != _as_int(ctx.get("position"), "freeze position"):
                    raise ValueError(f"AV row {key} does not bind discovery position")
    return expected


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _build_arm_map(payload: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {arm: arm for arm in ARMS}
    for container_name in ("arm_map", "condition_map", "conditions", "arms"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            for opaque, canonical in container.items():
                if isinstance(canonical, dict):
                    canonical = _first(canonical, "arm", "name", "label")
                if str(canonical) in ARMS:
                    mapping[str(opaque)] = str(canonical)
        elif isinstance(container, list):
            for item in container:
                if not isinstance(item, dict):
                    continue
                canonical = _first(item, "arm", "condition", "name", "label", "canonical_arm")
                opaque = _first(item, "arm_id", "condition_id", "id", "case_arm")
                if str(canonical) in ARMS and opaque is not None:
                    mapping[str(opaque)] = str(canonical)
    return mapping


def _task_feature(row: dict[str, Any]) -> int | None:
    value = _first(row, "feature", "feature_id", "target_feature")
    if value is None:
        return None
    try:
        return _as_int(value, "label feature")
    except ValueError:
        return None


def _task_arm(row: dict[str, Any], arm_map: dict[str, str]) -> str | None:
    value = _first(row, "arm", "arm_name", "condition", "arm_id", "condition_id")
    if value is None:
        return None
    return arm_map.get(str(value), str(value) if str(value) in ARMS else None)


def _hypothesis(row: dict[str, Any]) -> str | None:
    for key in ("hypothesis", "hypothesis_text", "feature_hypothesis", "label", "description"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _task_id(row: dict[str, Any]) -> str | None:
    value = _first(row, "case_id", "hypothesis_id", "task_id", "id")
    return str(value) if value is not None else None


def extract_label_tasks(payload: dict[str, Any], *, what: str) -> list[dict[str, Any]]:
    arm_map = _build_arm_map(payload)
    features: set[int] = set()
    tasks: list[dict[str, Any]] = []
    for row in _walk_dicts(payload):
        feature = _task_feature(row)
        arm = _task_arm(row, arm_map)
        hypothesis = _hypothesis(row)
        if feature is None or arm is None or hypothesis is None:
            continue
        # A result row may have a nested ``hard_negative`` etc.; only rows
        # declaring an arm and hypothesis are candidate label jobs.
        tasks.append({"feature": feature, "arm": arm, "hypothesis": hypothesis,
                      "case_id": _task_id(row), "row": dict(row), "arm_map": arm_map})
        features.add(feature)
    if not tasks:
        raise ValueError(f"{what} has no parseable feature/arm/hypothesis tasks")
    # Deduplicate exact object traversal (e.g. a top-level rows list appears in
    # both a convenience field and a nested field) only if all fields agree.
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for task in tasks:
        key = (task["feature"], task["arm"])
        old = by_key.get(key)
        if old is None:
            by_key[key] = task
        elif old["hypothesis"] != task["hypothesis"] or old.get("case_id") != task.get("case_id"):
            raise ValueError(f"{what} has conflicting duplicate task {key}")
    expected_features = set(range(0))  # populated/checked by caller
    return list(by_key.values())


def _validate_label_status(payload: dict[str, Any], what: str) -> None:
    _status_is_exploratory(payload, what, ("FROZEN", "COMPLETE", "RESULT"))
    if "PARSE_FAILED" in str(payload.get("status", "")).upper():
        raise ValueError(f"{what} has parser failures")


def _canonical_label_contract(label_freeze: dict[str, Any], label_result: dict[str, Any],
                              checkpoint_rows: Sequence[dict[str, Any]],
                              expected_features: set[int]) -> dict[tuple[int, str], dict[str, Any]] | None:
    """Read 58's cross-feature batch schema and flatten its 225 cases.

    Each immutable batch has five cases from five different features and five
    different conditions.  The private condition map is the only deblinding
    source; result/checkpoint rows are keyed by ``batch_id`` and contain five
    parsed opaque cases.  This function returns one normalized hypothesis for
    every ``(feature, condition)`` pair without exposing the map to prompts.
    """
    jobs = label_freeze.get("jobs")
    summaries = label_result.get("job_summaries")
    result_rows = label_result.get("rows")
    if not isinstance(jobs, list) or not isinstance(summaries, list) or not isinstance(result_rows, list):
        return None
    if len(jobs) != N_FEATURES or len(summaries) != N_FEATURES or len(result_rows) != N_FEATURES:
        raise ValueError("cross-feature label schema must contain exactly 45 batches/summaries/rows")
    jobs_by_batch: dict[int, dict[str, Any]] = {}
    case_map: dict[str, dict[str, Any]] = {}
    expected: dict[tuple[int, str], dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("label job entry is not an object")
        batch_id = _as_int(job.get("batch_id"), "label batch_id")
        if batch_id in jobs_by_batch:
            raise ValueError(f"duplicate label batch_id {batch_id}")
        jobs_by_batch[batch_id] = job
        cases = job.get("cases")
        cmap = job.get("condition_map")
        if not isinstance(cases, list) or len(cases) != len(ARMS) or not isinstance(cmap, dict):
            raise ValueError(f"batch {batch_id} must have five cases and condition_map")
        seen_cases: set[str] = set()
        seen_features: set[int] = set()
        seen_conditions: set[str] = set()
        for case in cases:
            if not isinstance(case, dict) or case.get("case_id") is None:
                raise ValueError(f"batch {batch_id} malformed opaque case")
            case_id = str(case["case_id"])
            if case_id in seen_cases:
                raise ValueError(f"batch {batch_id} duplicate case_id {case_id}")
            seen_cases.add(case_id)
            mapped = cmap.get(case_id)
            if not isinstance(mapped, dict):
                raise ValueError(f"batch {batch_id} condition_map omits {case_id}")
            feature_value = mapped.get("feature")
            feature = _as_int(feature_value, "label case feature")
            if feature not in expected_features or feature in seen_features:
                raise ValueError(f"batch {batch_id} has duplicate/out-of-contract feature {feature}")
            condition = _first(mapped, "condition", "arm", "condition_name", "canonical_arm")
            if condition is None:
                condition = _build_arm_map(label_freeze).get(str(_first(mapped, "condition_id", "arm_id", "id")))
            condition = str(condition) if condition is not None else ""
            condition = _build_arm_map(label_freeze).get(condition, condition)
            if condition not in ARMS or condition in seen_conditions:
                raise ValueError(f"batch {batch_id} condition_map has invalid/non-unique condition {condition!r}")
            seen_features.add(feature)
            seen_conditions.add(condition)
            if case_id in case_map:
                raise ValueError(f"duplicate global opaque case_id {case_id}")
            case_map[case_id] = {"feature": feature, "condition": condition,
                                 "batch_id": batch_id, "condition_map": dict(mapped),
                                 "job": job}
        # Every cross-feature batch must contain five distinct frozen features;
        # the exact feature subset differs by batch and is checked globally.
        if len(seen_features) != len(ARMS):
            raise ValueError(f"batch {batch_id} does not contain five distinct features")
        if seen_conditions != set(ARMS):
            raise ValueError(f"batch {batch_id} does not cover all five conditions")
    if set(jobs_by_batch) != set(range(N_FEATURES)) or len(case_map) != N_FEATURES * len(ARMS):
        raise ValueError("cross-feature label jobs must contain batch IDs 0..44 and exactly 225 unique cases")
    feature_condition_cases: dict[int, set[str]] = {feature: set() for feature in expected_features}
    for case_id, mapped in case_map.items():
        feature = int(mapped["feature"])
        condition = str(mapped["condition"])
        key = (feature, condition)
        if key in expected:
            raise ValueError(f"duplicate feature/condition case {key}")
        feature_condition_cases[feature].add(condition)
        expected[key] = {"feature": feature, "arm": condition,
                         "case_id": case_id, "hypothesis": None,
                         "condition_map": dict(mapped["condition_map"]),
                         "batch_id": int(mapped["batch_id"]), "job": mapped["job"]}
    for feature, conditions in feature_condition_cases.items():
        if conditions != set(ARMS):
            raise ValueError(f"feature {feature} does not have exactly five conditions: {sorted(conditions)}")
    if len(expected) != N_FEATURES * len(ARMS):
        raise ValueError("cross-feature label jobs do not cover 45x5")
    summaries_by_batch: dict[int, dict[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            raise ValueError("label job summary is not an object")
        batch_id = _as_int(summary.get("batch_id"), "label summary batch_id")
        if batch_id in summaries_by_batch:
            raise ValueError(f"duplicate label job summary batch {batch_id}")
        summaries_by_batch[batch_id] = summary
        job = jobs_by_batch.get(batch_id)
        if job is None:
            raise ValueError(f"label summary references unknown batch {batch_id}")
        if summary.get("prompt_sha256") not in (None, job.get("prompt_sha256")):
            raise ValueError(f"label summary prompt SHA mismatch batch {batch_id}")
        if summary.get("input_prompt_sha256") not in (None, job.get("prompt_sha256")):
            raise ValueError(f"label summary input prompt SHA mismatch batch {batch_id}")
        summary_cases = summary.get("case_ids")
        if isinstance(summary_cases, list) and {str(x) for x in summary_cases} != {
                cid for cid, mapped in case_map.items() if int(mapped["batch_id"]) == batch_id}:
            raise ValueError(f"label summary case_ids mismatch batch {batch_id}")
    if set(summaries_by_batch) != set(jobs_by_batch):
        raise ValueError("label job summaries omit one or more cross-feature batches")
    parsed_by_batch: dict[int, dict[str, Any]] = {}
    for result_row in result_rows:
        if not isinstance(result_row, dict):
            raise ValueError("label result row is not an object")
        batch_id = _as_int(result_row.get("batch_id"), "label result batch_id")
        if batch_id in parsed_by_batch:
            raise ValueError(f"duplicate label result batch {batch_id}")
        parsed_by_batch[batch_id] = result_row
        expected_job = jobs_by_batch.get(batch_id)
        if expected_job is None:
            raise ValueError(f"label result references unknown batch {batch_id}")
        expected_prompt_sha = expected_job.get("prompt_sha256")
        if result_row.get("input_prompt_sha256") not in (None, expected_prompt_sha) or result_row.get("prompt_sha256") not in (None, expected_prompt_sha):
            raise ValueError(f"label result prompt SHA mismatch batch {batch_id}")
        if str(result_row.get("status", "")).lower() != "ok":
            raise ValueError(f"label result parser/status failure batch {batch_id}")
        cases = result_row.get("cases")
        if not isinstance(cases, list) or len(cases) != len(ARMS):
            raise ValueError(f"label result batch {batch_id} does not have five parsed cases")
        seen_cases: set[str] = set()
        for case in cases:
            if not isinstance(case, dict) or case.get("case_id") is None:
                raise ValueError(f"label result batch {batch_id} malformed parsed case")
            case_id = str(case["case_id"])
            if case_id in seen_cases:
                raise ValueError(f"label result batch {batch_id} duplicate case {case_id}")
            seen_cases.add(case_id)
            mapped = case_map.get(case_id)
            if mapped is None or int(mapped["batch_id"]) != batch_id:
                raise ValueError(f"label result batch {batch_id} case outside frozen job: {case_id}")
            feature = int(mapped["feature"])
            condition = str(mapped["condition"])
            hypothesis = _hypothesis(case)
            if hypothesis is None:
                raise ValueError(f"label result batch {batch_id} case {case_id} omits hypothesis")
            if len(hypothesis.split()) > 32:
                raise ValueError(f"label result hypothesis exceeds 32 words feature={feature} case={case_id}")
            abstain = _first(case, "abstain", "abstention")
            if type(abstain) is not bool:
                raise ValueError(f"label result abstain is not boolean feature={feature} case={case_id}")
            confidence = _first(case, "confidence", "probability")
            if confidence is not None:
                confidence_value = _as_finite(confidence, "label confidence")
                if not 0.0 <= confidence_value <= 1.0:
                    raise ValueError("label confidence outside [0,1]")
            key = (feature, condition)
            expected[key]["hypothesis"] = hypothesis
            expected[key]["result_row"] = dict(case)
        expected_case_ids = {cid for cid, mapped in case_map.items() if int(mapped["batch_id"]) == batch_id}
        if seen_cases != expected_case_ids:
            raise ValueError(f"label result omissions for batch {batch_id}")
    if set(parsed_by_batch) != set(jobs_by_batch):
        raise ValueError("label result rows omit one or more cross-feature batches")
    # Checkpoint is an append-only audit of the same 45 batch rows.
    checkpoint_by_batch: dict[int, dict[str, Any]] = {}
    checkpoint_errors: dict[int, int] = {}
    for row in checkpoint_rows:
        if not isinstance(row, dict):
            raise ValueError("label checkpoint row is not an object")
        if row.get("kind") in ("header", "contract") or row.get("checkpoint_header"):
            continue
        if "batch_id" not in row:
            raise ValueError("label checkpoint row omits batch_id")
        batch_id = _as_int(row["batch_id"], "label checkpoint batch_id")
        if batch_id not in jobs_by_batch:
            raise ValueError(f"label checkpoint batch outside immutable jobs: {batch_id}")
        status = str(row.get("status", "")).lower()
        if status == "error":
            if batch_id in checkpoint_by_batch:
                raise ValueError(f"label checkpoint error after successful batch {batch_id}")
            if not isinstance(row.get("error"), str) or not row.get("error"):
                raise ValueError(f"label checkpoint error row lacks error text batch {batch_id}")
            checkpoint_errors[batch_id] = checkpoint_errors.get(batch_id, 0) + 1
            continue
        if status != "ok":
            raise ValueError(f"label checkpoint parser/status failure batch {batch_id}")
        if batch_id in checkpoint_by_batch:
            raise ValueError(f"duplicate successful label checkpoint batch {batch_id}")
        checkpoint_by_batch[batch_id] = row
        expected_prompt_sha = jobs_by_batch[batch_id].get("prompt_sha256")
        if row.get("input_prompt_sha256") not in (None, expected_prompt_sha) or row.get("prompt_sha256") not in (None, expected_prompt_sha):
            raise ValueError(f"label checkpoint prompt SHA mismatch batch {batch_id}")
        case_ids = {str(case.get("case_id")) for case in row.get("cases", []) if isinstance(case, dict)}
        expected_case_ids = {cid for cid, mapped in case_map.items() if int(mapped["batch_id"]) == batch_id}
        if case_ids != expected_case_ids:
            raise ValueError(f"label checkpoint omissions batch {batch_id}")
    if set(checkpoint_by_batch) != set(jobs_by_batch):
        raise ValueError("label checkpoint omits one or more cross-feature batch rows")
    if any(task["hypothesis"] is None for task in expected.values()):
        raise ValueError("label result omitted one or more hypotheses")
    return expected


def validate_labels(label_freeze: dict[str, Any], label_result: dict[str, Any],
                    checkpoint_rows: Sequence[dict[str, Any]], freeze_sha: str,
                    av_sha: str, feature_rows: Sequence[dict[str, Any]],
                    label_freeze_sha: str | None = None) -> dict[tuple[int, str], dict[str, Any]]:
    _validate_label_status(label_freeze, "label-job freeze")
    _validate_label_status(label_result, "label-job result")
    expected_features = {int(row["feature"]) for row in feature_rows}
    if len(expected_features) != N_FEATURES:
        raise ValueError("internal feature count is not 45")
    for payload, what in ((label_freeze, "label-job freeze"), (label_result, "label-job result")):
        bound_freeze = _nested_hash(payload, ("freeze_sha256", "j1_freeze_sha256"))
        if bound_freeze != freeze_sha:
            raise ValueError(f"{what} J1 freeze SHA binding mismatch: {bound_freeze} != {freeze_sha}")
        bound_av = _nested_hash(payload, ("av_result_sha256", "av_sha256", "j1_av_result_sha256", "input_result_sha256"))
        if bound_av != av_sha:
            raise ValueError(f"{what} AV SHA binding mismatch: {bound_av} != {av_sha}")
    # The label addendum/protocol hashes are not evaluator inputs, but when
    # present they must agree across immutable jobs, result, and checkpoints.
    label_protocol_values = [payload.get("label_protocol_sha256") for payload in (label_freeze, label_result)
                             if isinstance(payload.get("label_protocol_sha256"), str)]
    protocol_values = [payload.get("protocol_sha256") for payload in (label_freeze, label_result)
                       if isinstance(payload.get("protocol_sha256"), str)]
    if label_protocol_values and len(set(label_protocol_values)) != 1:
        raise ValueError("label protocol SHA bindings disagree")
    if protocol_values and len(set(protocol_values)) != 1:
        raise ValueError("upstream protocol SHA bindings disagree")
    expected_label_protocol = label_protocol_values[0] if label_protocol_values else None
    expected_protocol = protocol_values[0] if protocol_values else None
    label_rows_for_binding = list(label_result.get("rows", [])) if isinstance(label_result.get("rows"), list) else []
    for row in list(checkpoint_rows) + label_rows_for_binding:
        if not isinstance(row, dict):
            continue
        row_freeze_sha = _nested_hash(row, ("freeze_sha256", "j1_freeze_sha256"))
        if row_freeze_sha is not None and row_freeze_sha != freeze_sha:
            raise ValueError("label row/checkpoint J1 freeze SHA binding mismatch")
        row_input_sha = _nested_hash(row, ("input_result_sha256", "av_result_sha256", "av_sha256"))
        if row_input_sha is not None and row_input_sha != av_sha:
            raise ValueError("label row/checkpoint AV/result SHA binding mismatch")
        if label_freeze_sha is not None:
            row_jobs_sha = _nested_hash(row, ("jobs_sha256", "label_job_freeze_sha256", "label_jobs_sha256"))
            if row_jobs_sha is not None and row_jobs_sha != label_freeze_sha:
                raise ValueError("label row/checkpoint jobs SHA binding mismatch")
    for row in checkpoint_rows:
        if not isinstance(row, dict) or row.get("kind") in ("header", "contract") or row.get("checkpoint_header"):
            continue
        if expected_label_protocol is not None and row.get("label_protocol_sha256") != expected_label_protocol:
            raise ValueError("label checkpoint label-protocol SHA binding mismatch")
        if expected_protocol is not None and row.get("protocol_sha256") != expected_protocol:
            raise ValueError("label checkpoint upstream-protocol SHA binding mismatch")
    if label_freeze_sha is not None:
        result_jobs_sha = _nested_hash(label_result, ("jobs_sha256", "label_job_freeze_sha256", "label_jobs_sha256"))
        if result_jobs_sha != label_freeze_sha:
            raise ValueError(f"label-job result freeze SHA binding mismatch: {result_jobs_sha} != {label_freeze_sha}")
        for row in checkpoint_rows:
            if not isinstance(row, dict) or row.get("kind") in ("header", "contract") or row.get("checkpoint_header"):
                continue
            checkpoint_jobs_sha = _nested_hash(row, ("jobs_sha256", "label_job_freeze_sha256", "label_jobs_sha256"))
            if checkpoint_jobs_sha is not None and checkpoint_jobs_sha != label_freeze_sha:
                raise ValueError("label checkpoint label-job freeze SHA binding mismatch")
    canonical = _canonical_label_contract(label_freeze, label_result, checkpoint_rows, expected_features)
    if canonical is not None:
        return canonical
    expected_tasks = extract_label_tasks(label_freeze, what="label-job freeze")
    expected: dict[tuple[int, str], dict[str, Any]] = {}
    for task in expected_tasks:
        key = (int(task["feature"]), str(task["arm"]))
        if key[0] not in expected_features or key[1] not in ARMS:
            raise ValueError(f"label-job freeze has out-of-contract task {key}")
        expected[key] = task
    if set(f for f, _ in expected) != expected_features:
        raise ValueError("label-job freeze does not cover all 45 features")
    for feature in expected_features:
        arms = {arm for f, arm in expected if f == feature}
        if arms != set(ARMS):
            raise ValueError(f"feature {feature} label-job freeze arms differ: {sorted(arms)}")
    if len(expected) != N_FEATURES * len(ARMS):
        raise ValueError(f"label-job freeze must contain exactly {N_FEATURES * len(ARMS)} tasks")

    result_tasks = extract_label_tasks(label_result, what="label-job result")
    result_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for task in result_tasks:
        key = (int(task["feature"]), str(task["arm"]))
        if key in result_by_key:
            raise ValueError(f"duplicate parsed label result task {key}")
        row = task["row"]
        for error_key in ("parse_error", "parser_error", "error"):
            if row.get(error_key):
                raise ValueError(f"label parser failure for {key}: {row[error_key]}")
        result_by_key[key] = task
    if set(result_by_key) != set(expected):
        missing = sorted(set(expected) - set(result_by_key))
        extra = sorted(set(result_by_key) - set(expected))
        raise ValueError(f"label result parse omissions/mismatches missing={missing[:5]} extra={extra[:5]}")
    for key, task in expected.items():
        result_task = result_by_key[key]
        # A case ID is a useful exact binding when present, but older 58
        # outputs did not expose it in every row; hypothesis must always bind.
        if result_task["hypothesis"] != task["hypothesis"]:
            raise ValueError(f"label result hypothesis differs from frozen task {key}")
        row = result_task["row"]
        parse_status = str(_first(row, "parse_status", "status", "result_status") or "complete").upper()
        if any(token in parse_status for token in ("FAIL", "ERROR", "OMIT", "MISSING")):
            raise ValueError(f"label parser status is not complete for {key}: {parse_status}")
        abst = _first(row, "abstain", "abstention")
        if abst is not None and type(abst) is not bool:
            raise ValueError(f"label abstention is not boolean for {key}")
    # The label checkpoint is an append-only audit.  It must contain every
    # frozen task, with no malformed/failed parse row; this prevents silently
    # replacing a missing parser row with a favourable final result.
    checkpoint_tasks: dict[tuple[int, str], dict[str, Any]] = {}
    for row in checkpoint_rows:
        feature = _task_feature(row)
        arm = _task_arm(row, _build_arm_map(label_freeze))
        if feature is None or arm is None:
            # Metadata/checkpoint headers are allowed only if explicitly marked.
            if row.get("kind") in ("header", "contract") or row.get("checkpoint_header"):
                continue
            raise ValueError("label checkpoint row has no parseable feature/arm")
        key = (feature, arm)
        if key in checkpoint_tasks:
            raise ValueError(f"duplicate label checkpoint task {key}")
        if key not in expected:
            raise ValueError(f"label checkpoint task outside frozen job {key}")
        for error_key in ("parse_error", "parser_error", "error"):
            if row.get(error_key):
                raise ValueError(f"label checkpoint parser failure for {key}")
        checkpoint_tasks[key] = row
    if set(checkpoint_tasks) != set(expected):
        missing = sorted(set(expected) - set(checkpoint_tasks))
        raise ValueError(f"label checkpoint omissions: {missing[:5]}")
    return result_by_key


def _opaque_id(rng: random.Random, prefix: str, used: set[str]) -> str:
    while True:
        candidate = prefix + "_" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(14))
        if candidate not in used:
            used.add(candidate)
            return candidate


def build_eval_job(feature_rows: Sequence[dict[str, Any]], labels: dict[tuple[int, str], dict[str, Any]],
                   freeze_sha: str, av_sha: str, label_freeze_sha: str,
                   *, script_sha: str) -> tuple[dict[str, Any], dict[int, str]]:
    rng = random.Random(SEED)
    used_context_ids: set[str] = set()
    used_case_ids: set[str] = set()
    public_features: list[dict[str, Any]] = []
    private_conditions: list[dict[str, Any]] = []
    prompts: dict[int, str] = {}
    for feature_index, feature_row in enumerate(sorted(feature_rows, key=lambda row: int(row["feature"]))):
        feature = int(feature_row["feature"])
        # The public order is randomised; condition_map privately binds each
        # opaque id to feature/arm/truth and is never interpolated in prompts.
        context_specs: list[dict[str, Any]] = []
        for item in feature_row["contexts"]:
            context_id = _opaque_id(rng, "ctx", used_context_ids)
            context_specs.append({"context_id": context_id,
                                  "marked_context": item["marked_text"]})
            private_conditions.append({"context_id": context_id, "feature": feature,
                                       "truth": int(item["truth"]), "role": item["role"],
                                       "doc_id": int(item["row"]["doc_id"]),
                                       "position": int(item["row"]["position"]),
                                       "stratum": feature_row["stratum"]})
        # A freeze's context order is discovery, held-out positive, negative;
        # randomise only the prompt/public order to avoid positional cues.
        rng.shuffle(context_specs)
        hypothesis_specs: list[dict[str, Any]] = []
        private_hypotheses: list[dict[str, Any]] = []
        for arm in ARMS:
            task = labels[(feature, arm)]
            case_id = _opaque_id(rng, "case", used_case_ids)
            hypothesis = str(task["hypothesis"])
            hypothesis_specs.append({"case_id": case_id, "hypothesis": hypothesis})
            private_hypotheses.append({"case_id": case_id, "feature": feature,
                                       "arm": arm, "hypothesis": hypothesis,
                                       "source_case_id": task.get("case_id")})
        rng.shuffle(hypothesis_specs)
        public = {"feature_key": f"feature_batch_{feature_index + 1:02d}",
                  "hypotheses": hypothesis_specs, "contexts": context_specs}
        public_features.append(public)
        private_conditions.append({"feature": feature, "feature_key": public["feature_key"],
                                   "hypotheses": private_hypotheses})
        prompts[feature] = build_prompt(public)
    job = {
        "schema_version": 1,
        "experiment": "J1 exploratory blinded held-out evaluator",
        "status": "EXPLORATORY_BLINDED_EVAL_JOB_FROZEN",
        "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "seed": SEED,
        "evaluator_model": EVALUATOR_MODEL,
        "inputs": {"freeze_sha256": freeze_sha, "av_result_sha256": av_sha,
                   "label_job_freeze_sha256": label_freeze_sha, "script_sha256": script_sha},
        "contract": {"n_features": N_FEATURES, "contexts_per_feature": N_CONTEXTS,
                     "hypotheses_per_feature": len(ARMS), "scores_per_feature": N_CONTEXTS * len(ARMS),
                     "arms": list(ARMS), "truth_source": "freeze_only_actual_SAE_activation_and_embedded_hard_negative",
                     "condition_map_not_in_prompt": True},
        "public_batches": public_features,
        # This map is deliberately stored after the public batches.  The
        # evaluator receives only build_prompt(public_batches[i]); it never
        # receives this private object or source/activation metadata.
        "condition_map_private": {"contexts": [x for x in private_conditions if "context_id" in x],
                                   "hypotheses": [x for x in private_conditions if "hypotheses" in x]},
        "prompts": {str(feature): {"sha256": sha256_bytes(prompt.encode("utf-8")),
                                    "bytes": len(prompt.encode("utf-8"))}
                    for feature, prompt in sorted(prompts.items())},
    }
    # The output schema is written separately in a temp directory; storing a
    # hash makes accidental request/schema drift visible on resume.
    job["prompt_contract_sha256"] = canonical_sha(job["prompts"])
    return job, prompts


def build_prompt(batch: dict[str, Any]) -> str:
    return (
        "You are an external measurement instrument in an exploratory blinded evaluation.\n"
        "Do not use tools, read files, inspect the repository, browse, or execute commands.\n"
        "The opaque IDs carry no semantic information. Do not infer source, language, arm, "
        "activation, truth, or document role; none is provided.\n"
        "For every hypothesis x context pair below, estimate the probability in [0,1] "
        "that the target SAE feature fires at the marked token. Return exactly one score "
        "for every pair (5 hypotheses x 8 contexts = 40 scores), with no omissions or duplicates.\n"
        "Also return supported (boolean), abstain (boolean), and a concise evidence_code "
        "(short string, no prose essay). Do not return any other interpretation.\n\n"
        "JSON output must have this shape: {\"scores\":[{\"case_id\":string,"
        "\"context_id\":string,\"probability\":number,\"supported\":boolean,"
        "\"abstain\":boolean,\"evidence_code\":string}]}\n\n"
        + json.dumps(batch, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["scores"],
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["case_id", "context_id", "probability", "supported", "abstain", "evidence_code"],
                "additionalProperties": False,
                "properties": {
                    "case_id": {"type": "string"},
                    "context_id": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                    "supported": {"type": "boolean"},
                    "abstain": {"type": "boolean"},
                    "evidence_code": {"type": "string", "maxLength": MAX_EVIDENCE_CODE},
                },
            },
        },
    },
}


def validate_scores(raw: Any, batch: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("scores"), list):
        raise ValueError("evaluator output must be an object with scores array")
    expected_cases = {str(x["case_id"]) for x in batch["hypotheses"]}
    expected_contexts = {str(x["context_id"]) for x in batch["contexts"]}
    expected_pairs = {(case, context) for case in expected_cases for context in expected_contexts}
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in raw["scores"]:
        if not isinstance(row, dict):
            raise ValueError("evaluator score is not an object")
        required = ("case_id", "context_id", "probability", "supported", "abstain", "evidence_code")
        if any(key not in row for key in required):
            raise ValueError("evaluator score omits required field")
        case_id, context_id = str(row["case_id"]), str(row["context_id"])
        pair = (case_id, context_id)
        if pair not in expected_pairs:
            raise ValueError(f"evaluator score references unknown case/context: {pair}")
        if pair in seen:
            raise ValueError(f"duplicate evaluator score: {pair}")
        seen.add(pair)
        probability = _as_finite(row["probability"], "probability")
        if probability < 0.0 or probability > 1.0:
            raise ValueError("evaluator probability outside [0,1]")
        _as_bool(row["supported"], "supported")
        _as_bool(row["abstain"], "abstain")
        code = row["evidence_code"]
        if not isinstance(code, str) or not code.strip() or len(code.strip()) > MAX_EVIDENCE_CODE or "\n" in code:
            raise ValueError("evidence_code is missing or not concise")
        out.append({"case_id": case_id, "context_id": context_id, "probability": probability,
                    "supported": row["supported"], "abstain": row["abstain"],
                    "evidence_code": code.strip()})
    if seen != expected_pairs:
        missing = sorted(expected_pairs - seen)
        raise ValueError(f"evaluator score completeness failure: {len(missing)} missing")
    return out


def resolve_codex_command(requested: str) -> dict[str, Any]:
    """Resolve one executable path without invoking a shell.

    Windows npm installs commonly put a non-executable ``codex.ps1`` before
    the real executable shim.  For the default command, prefer the native
    ``codex.exe`` application, then an explicit ``codex.cmd`` shim.  A caller
    that explicitly names ``codex.cmd`` is honoured through ``cmd.exe`` with
    ``shell=False`` and argument-vector construction (no command-string
    interpolation).  Unix keeps the requested ``codex`` executable.
    """
    requested = str(requested or "codex")
    is_windows = sys.platform.startswith("win")
    explicit_path = any(separator in requested for separator in ("/", "\\")) or Path(requested).suffix.lower() in {".exe", ".cmd", ".bat", ".ps1"}
    candidates: list[str] = []
    if is_windows and not explicit_path and requested.lower() == "codex":
        # shutil.which honours PATHEXT but may return the npm .CMD shim first;
        # explicitly ask for the native app before considering that shim.
        candidates.extend(["codex.exe", "codex.cmd", "codex"])
    else:
        candidates.append(requested)
    def which_exact(candidate: str) -> str | None:
        found = shutil.which(candidate)
        if found:
            return found
        # Some Windows PATH/PATHEXT combinations make shutil.which skip a
        # literal .cmd entry.  An explicit PATH scan still returns the exact
        # file and does not invoke PowerShell or a shell parser.
        if is_windows and not Path(candidate).is_absolute():
            for directory in os.environ.get("PATH", "").split(os.pathsep):
                if not directory:
                    continue
                literal = Path(directory) / candidate
                if literal.is_file():
                    return str(literal)
        return None

    resolved: str | None = None
    for candidate in candidates:
        path = which_exact(candidate)
        if path:
            suffix = Path(path).suffix.lower()
            if is_windows and suffix == ".ps1":
                continue
            resolved = str(Path(path).resolve())
            break
    if resolved is None:
        raise FileNotFoundError(f"unable to resolve Codex executable {requested!r}")
    suffix = Path(resolved).suffix.lower()
    if is_windows and suffix in {".cmd", ".bat"}:
        # Passing the command interpreter itself as argv[0] keeps shell=False;
        # args are appended as separate argv items and never shell-joined by
        # this runner.  Native codex.exe remains the default Windows choice.
        comspec = os.environ.get("COMSPEC") or shutil.which("cmd.exe") or "cmd.exe"
        argv_prefix = [comspec, "/d", "/s", "/c", resolved]
        launch_kind = "cmd_wrapper_shell_false"
    else:
        argv_prefix = [resolved]
        launch_kind = "native_executable"
    return {"requested": requested, "resolved": resolved,
            "argv_prefix": argv_prefix, "launch_kind": launch_kind,
            "platform": sys.platform}


def _run_version(resolved_command: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    argv = list(resolved_command["argv_prefix"]) + ["--version"]
    try:
        completed = subprocess.run(argv, shell=False, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace",
                                   timeout=timeout, check=False)
        return {"command": argv, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:  # retained in metadata; feature calls will fail too
        return {"command": argv, "error": repr(exc)}


def _call_codex(resolved_command: dict[str, Any], prompt: str, schema: dict[str, Any],
                *, timeout: float, max_attempts: int) -> dict[str, Any]:
    version: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory(prefix="j1_eval_empty_") as temp_name:
            temp_root = Path(temp_name)
            schema_path = temp_root / "output_schema.json"
            output_path = temp_root / "last_message.json"
            schema_path.write_bytes(pretty_bytes(schema))
            args = list(resolved_command["argv_prefix"]) + ["exec", "-m", EVALUATOR_MODEL, "-s", "read-only",
                    *CODEX_CONFIG_OVERRIDES,
                    "--ephemeral", "--skip-git-repo-check", "-C", str(temp_root),
                    "--output-schema", str(schema_path), "-o", str(output_path), "-"]
            request = {"args": args, "model": EVALUATOR_MODEL, "attempt": attempt,
                       "prompt_sha256": sha256_bytes(prompt.encode("utf-8"))}
            try:
                completed = subprocess.run(args, input=prompt, text=True, shell=False,
                                           encoding="utf-8", errors="replace",
                                           capture_output=True, timeout=timeout, check=False)
                raw_file = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
                raw_stdout = completed.stdout or ""
                raw_stderr = completed.stderr or ""
                raw_text = raw_file or raw_stdout
                record = {**request, "returncode": completed.returncode,
                          "raw_final_json": raw_file, "raw_stdout": raw_stdout,
                          "raw_stderr": raw_stderr}
                attempts.append(record)
                if completed.returncode != 0:
                    record["failure"] = f"codex returncode {completed.returncode}"
                    continue
                try:
                    parsed = json.loads(raw_text)
                except Exception as exc:
                    record["failure"] = f"invalid evaluator JSON: {exc}"
                    continue
                return {"ok": True, "parsed": parsed, "attempts": attempts,
                        "args": args, "model": EVALUATOR_MODEL, "version": version}
            except Exception as exc:
                attempts.append({**request, "failure": repr(exc), "raw_final_json": "",
                                 "raw_stdout": "", "raw_stderr": ""})
    return {"ok": False, "attempts": attempts, "args": attempts[-1].get("args") if attempts else [],
            "model": EVALUATOR_MODEL, "version": version,
            "failure": "bounded evaluator retries exhausted"}


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    data = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def _load_eval_checkpoint(path: Path, job_sha: str) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    latest: dict[int, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    if not path.exists():
        return latest, history
    for row in load_jsonl(path):
        if row.get("job_sha256") != job_sha:
            raise ValueError("evaluator checkpoint job SHA mismatch")
        if "feature" not in row or "prompt_sha256" not in row:
            raise ValueError("evaluator checkpoint row lacks feature/prompt binding")
        feature = _as_int(row["feature"], "checkpoint feature")
        latest[feature] = row
        history.append(row)
    return latest, history


def evaluate_all(job: dict[str, Any], prompts: dict[int, str], job_sha: str,
                 checkpoint_path: Path, *, resolved_command: dict[str, Any], concurrency: int,
                 retries: int, timeout: float, dry_run: bool) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    latest, history = _load_eval_checkpoint(checkpoint_path, job_sha)
    expected_features = []
    feature_key_to_id: dict[str, int] = {}
    for index, batch in enumerate(job["public_batches"]):
        key = str(batch["feature_key"])
        private = next(x for x in job["condition_map_private"]["hypotheses"] if x["feature_key"] == key)
        feature = int(private["feature"])
        expected_features.append(feature)
        feature_key_to_id[key] = feature
        prompt_hash = sha256_bytes(prompts[feature].encode("utf-8"))
        if job["prompts"][str(feature)]["sha256"] != prompt_hash:
            raise ValueError(f"prompt hash drift for feature {feature}")
        old = latest.get(feature)
        if old is not None and old.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"evaluator checkpoint prompt hash mismatch for feature {feature}")
    expected_set = set(expected_features)
    if set(latest) - expected_set:
        raise ValueError("evaluator checkpoint contains feature outside immutable job")
    if dry_run:
        return latest, history, {"status": "DRY_RUN_FREEZE_ONLY", "version": None,
                                "command": None, "model": EVALUATOR_MODEL}
    version = _run_version(resolved_command)
    # The Windows Store native executable can be ACL-denied even though it is
    # visible on PATH.  If the default command was requested, fall back once to
    # the npm ``codex.cmd`` shim; the chosen resolved argv is then used for both
    # this version check and every evaluator call.
    if (version.get("returncode") != 0 or version.get("error")) and str(resolved_command.get("requested", "")).lower() == "codex":
        fallback = resolve_codex_command("codex.cmd")
        fallback_version = _run_version(fallback)
        if fallback_version.get("returncode") == 0 and not fallback_version.get("error"):
            resolved_command = fallback
            version = {"initial_attempt": version, "chosen": fallback_version}
        else:
            version = {"initial_attempt": version, "fallback_attempt": fallback_version}
    lock = threading.Lock()
    pending = [feature for feature in expected_features
               if not (latest.get(feature, {}).get("ok") is True and latest.get(feature, {}).get("scores"))]

    def one(feature: int) -> tuple[int, dict[str, Any]]:
        prompt = prompts[feature]
        private_batch = next(x for x in job["condition_map_private"]["hypotheses"]
                             if int(x["feature"]) == feature)
        batch = next(batch for batch in job["public_batches"]
                     if str(batch["feature_key"]) == str(private_batch["feature_key"]))
        result = _call_codex(resolved_command, prompt, OUTPUT_SCHEMA,
                             timeout=timeout, max_attempts=max(1, retries + 1))
        row: dict[str, Any] = {"job_sha256": job_sha, "feature": feature,
                               "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                               "ok": False, "model": EVALUATOR_MODEL,
                               "version": version, "attempts": result.get("attempts", []),
                               "args": result.get("args", []),
                               "request_model": EVALUATOR_MODEL}
        if result.get("ok"):
            try:
                scores = validate_scores(result["parsed"], batch)
                row["ok"] = True
                row["scores"] = scores
                row["raw_final_json"] = result["attempts"][-1].get("raw_final_json", "")
            except Exception as exc:
                row["failure"] = f"schema/completeness validation: {exc}"
        else:
            row["failure"] = result.get("failure", "evaluator call failed")
        _append_jsonl(checkpoint_path, row, lock)
        return feature, row

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for feature, row in pool.map(one, pending):
                latest[feature] = row
                history.append(row)
    metadata = {"status": "CALLS_COMPLETE", "version": version,
                "codex_version": (version.get("chosen", version).get("stdout", "").strip()
                                   if isinstance(version, dict) else ""),
                "command": list(resolved_command["argv_prefix"]) + ["exec", "-m", EVALUATOR_MODEL, "-s", "read-only",
                             *CODEX_CONFIG_OVERRIDES,
                             "--ephemeral", "--skip-git-repo-check", "-C", "<empty-temp-root>",
                             "--output-schema", "<schemafile>", "-o", "<lastmessagefile>", "-"],
                "model": EVALUATOR_MODEL, "resolved_command": resolved_command,
                "concurrency": concurrency,
                "retries": retries, "pending_at_start": pending}
    return latest, history, metadata


def _truth_lookup(job: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[str, tuple[int, str]]]:
    truth: dict[str, int] = {}
    context_data: dict[str, dict[str, Any]] = {}
    case_map: dict[str, tuple[int, str]] = {}
    for item in job["condition_map_private"]["contexts"]:
        cid = str(item["context_id"])
        truth[cid] = int(item["truth"])
        context_data[cid] = item
    for batch in job["public_batches"]:
        key = str(batch["feature_key"])
        private = next(x for x in job["condition_map_private"]["hypotheses"] if x["feature_key"] == key)
        for hyp in private["hypotheses"]:
            case_map[str(hyp["case_id"])] = (int(private["feature"]), str(hyp["arm"]))
    return truth, context_data, case_map


def _scores_records(latest: dict[int, dict[str, Any]], job: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth, context_data, case_map = _truth_lookup(job)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for feature, row in sorted(latest.items()):
        if row.get("ok") is not True:
            failures.append({"feature": feature, "failure": row.get("failure", "unknown"),
                             "attempts": row.get("attempts", []), "raw_final_json": row.get("raw_final_json", "")})
            continue
        for score in row.get("scores", []):
            case_id = str(score["case_id"])
            context_id = str(score["context_id"])
            if case_id not in case_map or context_id not in truth:
                raise ValueError("checkpoint score references unknown private condition id")
            score_feature, arm = case_map[case_id]
            if score_feature != feature:
                raise ValueError("checkpoint case belongs to a different feature")
            scores.append({**score, "feature": feature, "arm": arm,
                           "context_id": context_id, "truth": truth[context_id],
                           "stratum": context_data[context_id]["stratum"],
                           "role": context_data[context_id]["role"]})
    return scores, failures


def average_precision(rows: Sequence[dict[str, Any]]) -> float:
    positives = sum(int(row["truth"]) for row in rows)
    if positives <= 0:
        return float("nan")
    # Stable opaque-ID tie break is part of the exact AP contract.
    ordered = sorted(rows, key=lambda row: (-float(row["probability"]), str(row["context_id"])))
    hits = 0
    total = 0.0
    for rank, row in enumerate(ordered, 1):
        if int(row["truth"]) == 1:
            hits += 1
            total += hits / rank
    return total / positives


def pairwise_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    positives = [row for row in rows if int(row["truth"]) == 1]
    negatives = [row for row in rows if int(row["truth"]) == 0]
    if len(positives) != 4 or len(negatives) != 4:
        raise ValueError("pairwise accuracy requires four positives and four negatives")
    total = 0.0
    for positive in positives:
        for negative in negatives:
            p, n = float(positive["probability"]), float(negative["probability"])
            total += 1.0 if p > n else 0.5 if p == n else 0.0
    return total / 16.0


def brier(rows: Sequence[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return sum((float(row["probability"]) - int(row["truth"])) ** 2 for row in rows) / len(rows)


def macro_average_precision(rows: Sequence[dict[str, Any]]) -> float:
    """Unweighted mean of exact AP over feature clusters."""
    by_feature: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_feature.setdefault(row.get("feature"), []).append(row)
    values = [average_precision(group) for group in by_feature.values() if group]
    return statistics.fmean(values) if values else float("nan")


def coverage(rows: Sequence[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if not bool(row["abstain"])) / len(rows)


def calibration(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for row in rows:
        p = float(row["probability"])
        index = 4 if p == 1.0 else min(4, int(p * 5.0))
        bins[index].append(row)
    result: list[dict[str, Any]] = []
    for index, bucket in enumerate(bins):
        result.append({"bin": index, "lower": index / 5.0, "upper": (index + 1) / 5.0,
                       "n": len(bucket),
                       "mean_probability": sum(float(x["probability"]) for x in bucket) / len(bucket) if bucket else None,
                       "empirical_rate": sum(int(x["truth"]) for x in bucket) / len(bucket) if bucket else None,
                       "abs_gap": abs(sum(float(x["probability"]) for x in bucket) / len(bucket) -
                                      sum(int(x["truth"]) for x in bucket) / len(bucket)) if bucket else None})
    return result


def metric_bundle(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) == 8:
        pairwise = pairwise_accuracy(rows)
    else:
        # The protocol reports a per-feature 4x4 pairwise accuracy, then the
        # unweighted feature mean for pooled/stratified ITT summaries.  A
        # pooled 180-row pairwise calculation would overweight cross-feature
        # comparisons and is not the preregistered endpoint.
        by_feature: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            by_feature.setdefault(row.get("feature"), []).append(row)
        per_feature = [pairwise_accuracy(group) for group in by_feature.values() if len(group) == 8]
        pairwise = statistics.fmean(per_feature) if per_feature else float("nan")
    micro_ap = average_precision(rows)
    macro_ap = macro_average_precision(rows)
    return {"n": len(rows), "positive": sum(int(x["truth"]) for x in rows),
            # ``average_precision`` remains the protocol's pooled/micro AP;
            # explicit names prevent confusion with the feature-macro AP.
            "average_precision": micro_ap,
            "micro_pooled_average_precision": micro_ap,
            "macro_mean_feature_average_precision": macro_ap,
            "mean_pairwise_accuracy": pairwise if rows else float("nan"),
            "brier": brier(rows), "non_abstain_coverage": coverage(rows),
            "calibration_5bin": calibration(rows)}


def _group_scores(scores: Sequence[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in scores:
        grouped.setdefault((int(row["feature"]), str(row["arm"])), []).append(row)
    return grouped


def _pooled(rows_by_feature: dict[int, list[dict[str, Any]]], sampled_features: Sequence[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in sampled_features:
        out.extend(rows_by_feature.get(feature, []))
    return out


def _bootstrap_deltas(scores: Sequence[dict[str, Any]], features: Sequence[int],
                      arms: Sequence[str], *, reps: int = BOOTSTRAP_REPS,
                      seed: int = SEED) -> dict[str, Any]:
    by_arm_feature: dict[str, dict[int, list[dict[str, Any]]]] = {arm: {} for arm in arms}
    for row in scores:
        arm = str(row["arm"])
        if arm in by_arm_feature:
            by_arm_feature[arm].setdefault(int(row["feature"]), []).append(row)
    baseline = "SAE_CONTEXT"
    if baseline not in by_arm_feature:
        raise ValueError("bootstrap lacks SAE_CONTEXT baseline")
    rng = random.Random(seed)
    observations: dict[str, dict[str, list[float]]] = {
        arm: {metric: [] for metric in ("average_precision", "macro_average_precision",
                                       "mean_pairwise_accuracy", "brier", "coverage")}
        for arm in arms if arm != baseline
    }
    feature_list = list(features)
    n = len(feature_list)
    for _ in range(reps):
        sampled = [feature_list[rng.randrange(n)] for _ in range(n)]
        base_rows = _pooled(by_arm_feature[baseline], sampled)
        base_ap = average_precision(base_rows)
        base_macro_ap = statistics.fmean(average_precision(by_arm_feature[baseline][feature]) for feature in sampled)
        base_pair = sum(pairwise_accuracy(by_arm_feature[baseline][feature]) for feature in sampled) / n
        base_brier = brier(base_rows)
        base_cov = coverage(base_rows)
        for arm in observations:
            arm_rows = _pooled(by_arm_feature[arm], sampled)
            observations[arm]["average_precision"].append(average_precision(arm_rows) - base_ap)
            arm_macro_ap = statistics.fmean(average_precision(by_arm_feature[arm][feature]) for feature in sampled)
            observations[arm]["macro_average_precision"].append(arm_macro_ap - base_macro_ap)
            observations[arm]["mean_pairwise_accuracy"].append(
                sum(pairwise_accuracy(by_arm_feature[arm][feature]) for feature in sampled) / n - base_pair)
            observations[arm]["brier"].append(brier(arm_rows) - base_brier)
            observations[arm]["coverage"].append(coverage(arm_rows) - base_cov)
    output: dict[str, Any] = {"seed": seed, "reps": reps, "cluster": "feature",
                              "baseline": baseline,
                              "ap_endpoints": {"average_precision": "pooled_micro_AP",
                                                "macro_average_precision": "mean_per_feature_AP"},
                              "metrics": {}}
    for arm, metrics in observations.items():
        output["metrics"][arm] = {}
        for metric, values in metrics.items():
            ordered = sorted(values)
            def quantile(probability: float) -> float:
                if not ordered:
                    return float("nan")
                pos = (len(ordered) - 1) * probability
                lo = int(math.floor(pos))
                hi = int(math.ceil(pos))
                if lo == hi:
                    return ordered[lo]
                return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
            output["metrics"][arm][metric] = {
                "mean": statistics.fmean(values),
                "median": quantile(0.5),
                "percentile95": quantile(0.95),
                "percentile2_5": quantile(0.025),
                "percentile97_5": quantile(0.975),
                "ci95": {"lower": quantile(0.025), "upper": quantile(0.975)},
            }
    return output


def _raw_comparison(scores: Sequence[dict[str, Any]], left: str, right: str,
                    features: Sequence[int]) -> dict[str, Any]:
    grouped = _group_scores(scores)
    left_rows = [r for f in features for r in grouped.get((f, left), [])]
    right_rows = [r for f in features for r in grouped.get((f, right), [])]
    left_pair = statistics.fmean(pairwise_accuracy(grouped[(f, left)]) for f in features)
    right_pair = statistics.fmean(pairwise_accuracy(grouped[(f, right)]) for f in features)
    values = {"average_precision": average_precision(left_rows) - average_precision(right_rows),
              "macro_average_precision": macro_average_precision(left_rows) - macro_average_precision(right_rows),
              "mean_pairwise_accuracy": left_pair - right_pair,
              "brier": brier(left_rows) - brier(right_rows),
              "coverage": coverage(left_rows) - coverage(right_rows)}
    return {"left": left, "right": right, "delta_left_minus_right": values,
            "left_metrics": metric_bundle(left_rows), "right_metrics": metric_bundle(right_rows)}


def _byte_budget_flags(label_freeze: dict[str, Any], label_result: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, list[float]] = {arm: [] for arm in ARMS}
    rows = list(_walk_dicts(label_freeze)) + list(_walk_dicts(label_result))
    for row in rows:
        arm = _task_arm(row, _build_arm_map(label_freeze))
        # Canonical 58 stores the per-arm byte budget in a mapping on each
        # feature job/summary rather than repeating an arm field per scalar.
        budget_map = row.get("arm_input_utf8_bytes")
        if isinstance(budget_map, dict):
            for budget_arm, budget_value in budget_map.items():
                canonical_arm = _build_arm_map(label_freeze).get(str(budget_arm), str(budget_arm))
                if canonical_arm in values and isinstance(budget_value, (int, float)) and not isinstance(budget_value, bool) and math.isfinite(float(budget_value)):
                    values[canonical_arm].append(float(budget_value))
        if arm not in values:
            continue
        value = _first(row, "input_bytes", "prompt_bytes", "input_utf8_bytes", "prompt_utf8_bytes", "bytes")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values[arm].append(float(value))
    summary = {arm: {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None,
                     "mean": statistics.fmean(v) if v else None} for arm, v in values.items()}
    known = [summary[arm]["mean"] for arm in ARMS if summary[arm]["mean"] is not None]
    differs = bool(known) and (max(known) - min(known) > 0.0)
    return {"status": "DIFFER" if differs else "UNKNOWN_NO_BYTE_COUNTS" if not known else "EQUAL_RECORDED",
            "differs": differs, "by_arm": summary,
            "risk": differs or not known}


def _dependence_flag(label_result: dict[str, Any], evaluator_model: str,
                     label_freeze: dict[str, Any] | None = None) -> dict[str, Any]:
    models: list[str] = []
    for payload in (label_freeze or {}, label_result):
        for key in ("model", "label_model", "generator_model", "interpreter_model"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in models:
                models.append(value.strip())
    evaluator = evaluator_model.lower()
    same = any(evaluator in model.lower() or model.lower() in evaluator for model in models)
    def family(name: str) -> str:
        lower = name.lower()
        if any(token in lower for token in ("fable", "claude", "anthropic")):
            return "anthropic/fable"
        if any(token in lower for token in ("terra", "gpt", "openai", "codex")):
            return "openai/codex"
        return "unknown"
    evaluator_family = family(evaluator_model)
    label_families = sorted({family(model) for model in models})
    different_family = bool(label_families and evaluator_family not in label_families and "unknown" not in label_families)
    return {"label_models": models, "evaluator_model": evaluator_model,
            "label_model_families": label_families, "evaluator_model_family": evaluator_family,
            "same_model_detected": same, "different_model_family": different_family,
            "status": "DEPENDENT" if same else "DIFFERENT_MODEL_FAMILY" if different_family else "UNKNOWN_OR_DIFFERENT"}


def _repeated_context_flag(label_freeze: dict[str, Any]) -> dict[str, Any]:
    by_feature_arm: dict[tuple[int, str], set[tuple[Any, Any]]] = {}
    canonical_signatures: dict[int, dict[str, set[str]]] = {}
    for row in _walk_dicts(label_freeze):
        feature = _task_feature(row)
        arm = _task_arm(row, _build_arm_map(label_freeze))
        if feature is None or arm is None:
            continue
        contexts = row.get("contexts", row.get("context_rows"))
        if isinstance(contexts, list):
            physical = set()
            for item in contexts:
                if isinstance(item, dict) and "doc_id" in item and "position" in item:
                    physical.add(_physical(item))
            if physical:
                by_feature_arm[(feature, arm)] = physical
    # Canonical 58 condition_payloads retain the same four discovery context
    # strings for SAE_CONTEXT/assisted/contrastive/mismatched.  Physical IDs
    # are intentionally not exposed to the label generator, so use a stable
    # text signature as an independent repetition check.
    for job in label_freeze.get("jobs", []) if isinstance(label_freeze.get("jobs"), list) else []:
        if not isinstance(job, dict):
            continue
        cmap = job.get("condition_map", {})
        signatures: dict[str, set[str]] = {}
        for case in job.get("cases", []) if isinstance(job.get("cases"), list) else []:
            if not isinstance(case, dict):
                continue
            cid = str(case.get("case_id"))
            mapped = cmap.get(cid) if isinstance(cmap, dict) else None
            condition = str(mapped.get("condition")) if isinstance(mapped, dict) else ""
            feature_value = mapped.get("feature") if isinstance(mapped, dict) else None
            if feature_value is None:
                feature_value = job.get("feature")
            if feature_value is None:
                continue
            feature = _as_int(feature_value, "label job feature")
            payload = case.get("condition_payload", case.get("evidence"))
            if condition not in ARMS or not isinstance(payload, list):
                continue
            contexts = {str(item.get("context")) for item in payload if isinstance(item, dict) and item.get("context") is not None}
            if contexts:
                signatures[condition] = contexts
                canonical_signatures.setdefault(feature, {})[condition] = contexts
        for condition, signature in signatures.items():
            by_feature_arm.setdefault((feature, condition), signature)
    repeated = 0
    repeated_features: set[int] = set()
    for feature in {feature for feature, _ in by_feature_arm}:
        sets = [by_feature_arm[(feature, arm)] for arm in ARMS if (feature, arm) in by_feature_arm]
        if len(sets) >= 2 and all(item == sets[0] for item in sets[1:]):
            repeated += 1
            repeated_features.add(feature)
    # In the cross-feature 58 design, a feature's four context-bearing arms
    # are deliberately distributed across different batches.  Count that
    # reuse even when snippets make their serialized payload signatures differ.
    for feature, signatures in canonical_signatures.items():
        context_sets = [value for condition, value in signatures.items()
                        if condition != "NLA_ONLY" and len(value) >= 4]
        if len(context_sets) >= 2 and feature not in repeated_features:
            repeated += 1
            repeated_features.add(feature)
    return {"features_with_repeated_context_sets": repeated,
            "flag": repeated > 0,
            "risk": "The four discovery contexts are intentionally reused across arms; this is not independent context replication."}


def analyze(scores: Sequence[dict[str, Any]], feature_rows: Sequence[dict[str, Any]],
            label_freeze: dict[str, Any], label_result: dict[str, Any],
            failures: Sequence[dict[str, Any]], *, evaluator_metadata: dict[str, Any],
            freeze_risks: Sequence[str]) -> tuple[dict[str, Any], str]:
    feature_ids = sorted(int(row["feature"]) for row in feature_rows)
    grouped = _group_scores(scores)
    if len(scores) != N_FEATURES * len(ARMS) * N_CONTEXTS:
        raise ValueError("analysis requires all 45x5x8 score rows")
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [r for feature in feature_ids for r in grouped[(feature, arm)]]
        by_arm[arm] = metric_bundle(rows)
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        ids = [int(row["feature"]) for row in feature_rows if row["stratum"] == stratum]
        by_stratum[stratum] = {}
        for arm in ARMS:
            rows = [r for feature in ids for r in grouped[(feature, arm)]]
            by_stratum[stratum][arm] = metric_bundle(rows)
    bootstrap = _bootstrap_deltas(scores, feature_ids,
                                  ("SAE_CONTEXT", "NLA_ASSISTED", "NLA_CONTRASTIVE", "NLA_MISMATCHED", "NLA_ONLY"))
    direct = {
        "NLA_ASSISTED_vs_NLA_MISMATCHED": _raw_comparison(
            scores, "NLA_ASSISTED", "NLA_MISMATCHED", feature_ids),
        "NLA_CONTRASTIVE_vs_NLA_MISMATCHED": _raw_comparison(
            scores, "NLA_CONTRASTIVE", "NLA_MISMATCHED", feature_ids),
    }
    # Stratum-driving flag: direction disagreement, or one stratum contributing
    # more than twice the absolute ITT effect, is reported as a risk, never a
    # reason to delete that stratum.
    stratum_flags: dict[str, Any] = {}
    for arm in ("NLA_ASSISTED", "NLA_CONTRASTIVE", "NLA_MISMATCHED", "NLA_ONLY"):
        overall = by_arm[arm]["average_precision"] - by_arm["SAE_CONTEXT"]["average_precision"]
        deltas = {s: by_stratum[s][arm]["average_precision"] - by_stratum[s]["SAE_CONTEXT"]["average_precision"] for s in STRATA}
        disagree = any((d > 0) != (overall > 0) for d in deltas.values() if d != 0 and overall != 0)
        dominant = any(abs(d) > 2.0 * abs(overall) for d in deltas.values()) if overall != 0 else any(d != 0 for d in deltas.values())
        stratum_flags[arm] = {"itt_ap_delta": overall, "stratum_ap_deltas": deltas,
                              "direction_disagreement": disagree, "dominant_stratum": dominant,
                              "flag": disagree or dominant}
    byte_flags = _byte_budget_flags(label_freeze, label_result)
    dependence = _dependence_flag(label_result, str(evaluator_metadata.get("model", EVALUATOR_MODEL)), label_freeze)
    repeated = _repeated_context_flag(label_freeze)
    risks = list(freeze_risks) + [
        "Discovery only: no confirmatory inference, p-value fishing, or gate promotion is permitted.",
        "Actual SAE activation in the immutable freeze is truth; evaluator probability/support/abstain is a measurement instrument.",
        "All 45 features and all five arms are ITT; evaluator failures would make the analysis incomplete rather than drop rows.",
    ]
    if byte_flags["risk"]:
        risks.append("Input-token/byte budget equality is not established across label arms.")
    if dependence["same_model_detected"]:
        risks.append("Label generator and evaluator appear to use the same model family; dependence limits interpretation.")
    elif not dependence.get("different_model_family"):
        risks.append("Label-generator/evaluator model-family separation is not fully established from provenance fields.")
    if repeated["flag"]:
        risks.append("The same four discovery contexts are repeated across arms by design.")
    if any(item["flag"] for item in stratum_flags.values()):
        risks.append("At least one stratum directionally disagrees with or dominates an ITT AP effect.")
    analysis = {
        "schema_version": 1,
        "experiment": "J1 exploratory blinded held-out evaluator analysis",
        "status": "EXPLORATORY_ANALYSIS_COMPLETE" if not failures else "NO_COMPLETE_ANALYSIS",
        "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "truth": {"source": "j1_discovery_freeze_only", "positive_rule": "actual SAE activation > 0",
                  "negative_rule": "embedded hard_negative target_activation == 0"},
        "itt": {"n_features": N_FEATURES, "arms": list(ARMS), "n_scores": len(scores),
                "n_failures": len(failures), "raw_itt_retained": True},
        "by_arm": by_arm, "by_stratum": by_stratum,
        "feature_cluster_bootstrap": bootstrap,
        "direct_assisted_vs_mismatched_and_contrastive_vs_mismatched": direct,
        "flags": {"input_byte_budgets": byte_flags, "evaluator_model_dependence": dependence,
                  "repeated_four_row_contexts": repeated, "stratum_drives_result": stratum_flags},
        "evaluator": evaluator_metadata,
        "failures": list(failures), "risks": risks,
    }
    md = render_analysis_md(analysis)
    return analysis, md


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.6f}"
    return str(value)


def render_analysis_md(analysis: dict[str, Any]) -> str:
    lines = ["# J1 exploratory blinded held-out evaluator analysis", "",
             f"Status: **{analysis['status']}** (discovery only; no confirmatory inference).", "",
             "Truth is the actual SAE activation retained in the immutable freeze; evaluator outputs are a measurement instrument, never truth.", "",
             "## ITT metrics", "", "AP is shown both as pooled/micro AP and unweighted macro mean feature AP; pairwise is the mean per-feature 4x4 endpoint.", "", "| Arm | micro AP | macro AP | pairwise 4x4 | Brier | coverage | n |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, row in analysis.get("by_arm", {}).items():
        lines.append(f"| {arm} | {_fmt(row.get('micro_pooled_average_precision', row.get('average_precision')))} | {_fmt(row.get('macro_mean_feature_average_precision'))} | {_fmt(row.get('mean_pairwise_accuracy'))} | {_fmt(row.get('brier'))} | {_fmt(row.get('non_abstain_coverage'))} | {row.get('n')} |")
    lines += ["", "## Feature-cluster bootstrap", "", "20,000 paired feature-cluster resamples, seed 20260806; deltas are arm minus SAE_CONTEXT (negative Brier is favourable).", ""]
    for arm, metrics in analysis.get("feature_cluster_bootstrap", {}).get("metrics", {}).items():
        lines.append(f"### {arm}")
        for metric, summary in metrics.items():
            lines.append(f"- {metric}: mean {_fmt(summary['mean'])}, median {_fmt(summary['median'])}, percentile95 {_fmt(summary['percentile95'])}, 95% CI [{_fmt(summary['percentile2_5'])}, {_fmt(summary['percentile97_5'])}]")
    lines += ["", "## Required risk flags", ""]
    flags = analysis.get("flags", {})
    lines.append(f"- Input byte budgets: **{flags.get('input_byte_budgets', {}).get('status')}**.")
    lines.append(f"- Evaluator/model dependence: **{flags.get('evaluator_model_dependence', {}).get('status')}**.")
    lines.append(f"- Repeated four-row contexts: **{flags.get('repeated_four_row_contexts', {}).get('flag')}**.")
    lines.append(f"- Stratum effects: `{json.dumps(flags.get('stratum_drives_result', {}), sort_keys=True)}`.")
    lines += ["", "## Risks and limitations", ""]
    lines.extend(f"- {risk}" for risk in analysis.get("risks", []))
    if analysis.get("failures"):
        lines += ["", "## Retained evaluator failures", ""]
        lines.extend(f"- Feature {row.get('feature')}: {row.get('failure')}" for row in analysis["failures"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--freeze", type=Path, default=result_dir / "j1_discovery_freeze_v1.json")
    parser.add_argument("--av-result", type=Path, default=result_dir / "j1_discovery_result_v1.json")
    parser.add_argument("--label-job-freeze", type=Path, default=result_dir / "j1_discovery_labels_jobs_v1.json")
    parser.add_argument("--label-job-result", type=Path, default=result_dir / "j1_discovery_labels_v1.json")
    parser.add_argument("--label-job-checkpoint", type=Path, default=result_dir / "j1_discovery_labels_checkpoint_v1.jsonl")
    parser.add_argument("--out-job", type=Path, default=result_dir / "j1_blinded_eval_job_v1.json")
    parser.add_argument("--out-job-sha256", type=Path, default=None)
    parser.add_argument("--out-checkpoint", type=Path, default=result_dir / "j1_blinded_eval_checkpoint_v1.jsonl")
    parser.add_argument("--out-result", type=Path, default=result_dir / "j1_blinded_eval_result_v1.json")
    parser.add_argument("--out-analysis", type=Path, default=result_dir / "j1_blinded_eval_analysis_v1.json")
    parser.add_argument("--out-analysis-md", type=Path, default=result_dir / "J1_BLINDED_EVAL_ANALYSIS_v1.md")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true", help="freeze and hash the blinded job only; never invoke Codex")
    parser.add_argument("--allow-missing-sidecars", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.concurrency < 1 or args.retries < 0 or args.timeout <= 0:
        raise ValueError("concurrency >=1, retries >=0, timeout >0 required")
    for path in (args.freeze, args.av_result, args.label_job_freeze,
                 args.label_job_result, args.label_job_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    required_sidecars = not args.allow_missing_sidecars
    freeze_sha = verify_sidecar(args.freeze, required=required_sidecars)
    av_sha = verify_sidecar(args.av_result, required=required_sidecars)
    label_freeze_sha = verify_sidecar(args.label_job_freeze, required=required_sidecars)
    label_result_sha = verify_sidecar(args.label_job_result, required=required_sidecars)
    freeze = load_json(args.freeze)
    av = load_json(args.av_result)
    label_freeze = load_json(args.label_job_freeze)
    label_result = load_json(args.label_job_result)
    checkpoint_rows = load_jsonl(args.label_job_checkpoint)
    if not isinstance(freeze, dict) or not isinstance(av, dict) or not isinstance(label_freeze, dict) or not isinstance(label_result, dict):
        raise ValueError("all JSON inputs must be objects")
    feature_rows, stratum_by_feature, freeze_risks = validate_freeze(freeze, freeze_sha)
    validate_av(av, freeze_sha, feature_rows)
    labels = validate_labels(label_freeze, label_result, checkpoint_rows,
                             freeze_sha, av_sha, feature_rows,
                             label_freeze_sha=label_freeze_sha)
    script_sha = sha256_file(Path(__file__))
    job, prompts = build_eval_job(feature_rows, labels, freeze_sha, av_sha,
                                  label_freeze_sha, script_sha=script_sha)
    job_sha = write_immutable(args.out_job, job)
    job_sha_path = args.out_job_sha256 or Path(str(args.out_job) + ".sha256")
    write_sidecar(job_sha_path, job_sha)
    print(f"J1 blinded eval job frozen: {args.out_job} sha256={job_sha}", flush=True)
    if args.dry_run:
        print("--dry-run: no Codex evaluator call, checkpoint, result, or analysis written", flush=True)
        return
    if args.out_result.exists():
        existing_result_sha = verify_sidecar(args.out_result, required=required_sidecars)
        existing_result = load_json(args.out_result)
        if not isinstance(existing_result, dict) or existing_result.get("job_sha256") != job_sha:
            raise ValueError("existing evaluator result does not bind the immutable eval job; refusing resume")
        if existing_result.get("status") != "EXPLORATORY_BLINDED_EVAL_COMPLETE":
            raise RuntimeError("existing evaluator result is incomplete; refusing to append a different final result")
        print(f"Existing complete evaluator result verified ({existing_result_sha}); no evaluator calls needed", flush=True)
        return
    resolved_command = resolve_codex_command(args.codex_command)
    latest, history, evaluator_metadata = evaluate_all(
        job, prompts, job_sha, args.out_checkpoint, resolved_command=resolved_command,
        concurrency=args.concurrency, retries=args.retries, timeout=args.timeout, dry_run=False)
    scores, failures = _scores_records(latest, job)
    result_status = "EXPLORATORY_BLINDED_EVAL_COMPLETE" if not failures and len(scores) == N_FEATURES * len(ARMS) * N_CONTEXTS else "NO_COMPLETE_ANALYSIS"
    result = {
        "schema_version": 1, "experiment": "J1 exploratory blinded held-out evaluator",
        "status": result_status, "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "seed": SEED, "job_sha256": job_sha, "freeze_sha256": freeze_sha,
        "av_result_sha256": av_sha, "label_job_freeze_sha256": label_freeze_sha,
        "label_job_result_sha256": label_result_sha,
        "evaluator": evaluator_metadata,
        "checkpoint": {"path": str(args.out_checkpoint), "history_rows": len(history)},
        "call_records": {str(feature): row for feature, row in sorted(latest.items())},
        "scores": scores, "failures": failures,
        "checks": {"expected_features": N_FEATURES, "expected_arms": len(ARMS),
                    "expected_scores": N_FEATURES * len(ARMS) * N_CONTEXTS,
                    "recorded_scores": len(scores), "recorded_failures": len(failures)},
        "raw_final_json_retained_in_checkpoint": True,
    }
    result_sha = write_immutable(args.out_result, result)
    write_sidecar(Path(str(args.out_result) + ".sha256"), result_sha)
    if result_status == "EXPLORATORY_BLINDED_EVAL_COMPLETE":
        analysis, analysis_md = analyze(scores, feature_rows, label_freeze, label_result,
                                        failures, evaluator_metadata=evaluator_metadata,
                                        freeze_risks=freeze_risks)
    else:
        analysis = {
            "schema_version": 1, "experiment": "J1 exploratory blinded held-out evaluator analysis",
            "status": "NO_COMPLETE_ANALYSIS", "confirmatory": False,
            "claim_scope": "discovery_only_no_confirmatory_inference",
            "truth": {"source": "j1_discovery_freeze_only"}, "failures": failures,
            "risks": ["Evaluator failure/incompleteness prevents a complete analysis; no rows were deleted.",
                      "Actual SAE activation is truth; evaluator outputs are a measurement instrument."],
        }
        analysis_md = render_analysis_md(analysis)
    analysis_sha = write_immutable(args.out_analysis, analysis)
    write_sidecar(Path(str(args.out_analysis) + ".sha256"), analysis_sha)
    md_bytes = analysis_md.encode("utf-8")
    args.out_analysis_md.parent.mkdir(parents=True, exist_ok=True)
    if args.out_analysis_md.exists():
        if args.out_analysis_md.read_bytes() != md_bytes:
            raise RuntimeError(f"refusing to overwrite immutable analysis markdown: {args.out_analysis_md}")
    else:
        with args.out_analysis_md.open("xb") as handle:
            handle.write(md_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    write_sidecar(Path(str(args.out_analysis_md) + ".sha256"), sha256_bytes(md_bytes))
    print(f"J1 evaluator result status={result_status} scores={len(scores)} failures={len(failures)} result_sha256={result_sha}", flush=True)
    print(f"J1 independent analysis status={analysis['status']} analysis_sha256={analysis_sha}", flush=True)


if __name__ == "__main__":
    main()
