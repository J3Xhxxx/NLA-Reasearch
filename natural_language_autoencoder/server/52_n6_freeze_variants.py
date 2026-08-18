#!/usr/bin/env python3
"""Freeze the N6 analysis cohort, matched donors, and byte-audited text variants.

This is a CPU/text-only stage.  It deliberately performs no AR reconstruction,
base-model forward pass, causal scoring, or outcome analysis.  The final
preregistration, plan, AV explanations, and code manifest must already be
frozen with valid SHA-256 sidecars.

Selection is deterministic:

1. Parse the final paragraph using ASCII double-quote bytes.
2. Apply the frozen text/tokenizer-only eligibility gates.
3. Form hard cells ``(source, candidate_count)`` and discard cells smaller than
   ``--min-cell-size``.
4. Seed ``--cell-seed-quota`` rows from every retained cell.
5. Fill to ``--target`` by deterministic source round-robin.
6. Within every selected hard cell, compute a deterministic minimum-cost
   one-to-one derangement with no normalized candidate shared by a pair.

Only the delayed tokenizer import is non-stdlib.  ``--self-test`` is stdlib-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
PLACEHOLDER = "[...]"
VARIANT_KEYS = (
    "orig",
    "p3_true",
    "p3_cross_matched",
    "p3_candidate_strip",
    "p3_anchor_strip",
    "p3_all_quote_strip",
    "p12",
)
COST_COMPONENTS = (
    "unique_canonical_first_token_id_count_abs_diff",
    "total_candidate_token_length_abs_diff",
    "sorted_per_candidate_token_length_l1",
    "p3_token_length_abs_diff",
    "sha256_tie_rank",
)
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
BLANK_LINE_BYTES_RE = re.compile(rb"(?:\r?\n)[ \t\f\v]*(?:\r?\n)")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_file_hashes(model_dir: Path) -> dict[str, str]:
    """Replicate the stage-49 tokenizer/config identity selection exactly."""
    names = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
        "config.json",
    }
    files = [
        path
        for path in model_dir.iterdir()
        if path.is_file() and (path.name in names or path.name.startswith("tokenizer."))
    ]
    if not files:
        raise ValueError(f"no tokenizer/config identity files under {model_dir}")
    return {path.name: sha256_file(path) for path in sorted(files)}


def verify_tokenizer_identity(
    model_dir: Path, plan: dict[str, Any]
) -> dict[str, str]:
    expected_raw = plan.get("inputs", {}).get("tokenizer_file_sha256")
    if not isinstance(expected_raw, dict) or not expected_raw:
        raise ValueError("plan does not bind tokenizer_file_sha256")
    expected = {
        str(name): require_hex64(digest, f"plan tokenizer hash {name}")
        for name, digest in expected_raw.items()
    }
    actual = tokenizer_file_hashes(model_dir)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(
            name
            for name in set(actual) & set(expected)
            if actual[name] != expected[name]
        )
        raise ValueError(
            "base-model tokenizer/config identity differs from the frozen plan: "
            f"missing={missing} extra={extra} wrong={wrong}"
        )
    return actual


def framed_sha256(seed: int, domain: str, *parts: object) -> str:
    """Hash an unambiguous, versioned sequence of UTF-8 fields."""
    digest = hashlib.sha256()
    for value in ("n6-freeze-v1", str(seed), domain, *map(str, parts)):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def require_hex64(value: Any, label: str) -> str:
    result = str(value)
    if not HEX64_RE.fullmatch(result):
        raise ValueError(f"{label} is not a SHA-256 hex digest: {result!r}")
    return result.lower()


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def verify_sidecar(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise ValueError(f"{label} SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise ValueError(f"malformed {label} SHA-256 sidecar: {sidecar}")
    declared, declared_name = fields
    declared = require_hex64(declared, f"{label} sidecar digest")
    if declared != observed:
        raise ValueError(
            f"{label} differs from sidecar: observed={observed} declared={declared}"
        )
    if declared_name != path.name:
        raise ValueError(
            f"{label} sidecar names {declared_name!r}, expected {path.name!r}"
        )
    return observed


def write_frozen_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different frozen output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    expected_sidecar = f"{digest}  {path.name}\n"
    sidecar = sidecar_path(path)
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise FileExistsError(f"refusing to replace different output sidecar: {sidecar}")
    if not sidecar.exists():
        sidecar.write_text(expected_sidecar, encoding="utf-8")
    return digest


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def nested_hash(payload: dict[str, Any], label: str, *paths: tuple[str, ...]) -> str:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return require_hex64(value, f"{label} {'.'.join(path)}")
    rendered = ", ".join(".".join(path) for path in paths)
    raise ValueError(f"{label} does not bind any of: {rendered}")


def reject_draft_prereg(path: Path) -> None:
    if ".draft" in path.name.casefold():
        raise ValueError(f"refusing a .DRAFT preregistration: {path}")
    prefix = path.read_text(encoding="utf-8")[:2048]
    if re.search(r"(?im)^status\s*:\s*.*\bdraft\b", prefix):
        raise ValueError("refusing a preregistration whose Status is DRAFT")


def _manifest_entries_from_json(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        if "path" in value and ("sha256" in value or "hash" in value):
            digest = value.get("sha256", value.get("hash"))
            if isinstance(value["path"], str) and isinstance(digest, str):
                found.append((value["path"], digest))
        for key, item in value.items():
            if isinstance(key, str) and isinstance(item, str) and HEX64_RE.fullmatch(item):
                found.append((key, item))
            elif isinstance(item, (dict, list)):
                found.extend(_manifest_entries_from_json(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_manifest_entries_from_json(item))
    return found


def code_manifest_entries(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: list[tuple[str, str]] = []
    try:
        entries.extend(_manifest_entries_from_json(json.loads(text)))
    except json.JSONDecodeError:
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.strip().split(None, 1)
            if len(fields) != 2 or not HEX64_RE.fullmatch(fields[0]):
                raise ValueError(f"malformed code manifest line {line_number}")
            entries.append((fields[1].lstrip("*").strip(), fields[0]))
    unique: dict[tuple[str, str], None] = {}
    for name, digest in entries:
        unique[(name.replace("\\", "/"), require_hex64(digest, "code digest"))] = None
    if not unique:
        raise ValueError("code manifest has no file SHA-256 entries")
    return list(unique)


def verify_current_script_in_manifest(path: Path) -> tuple[str, str]:
    manifest_sha256 = verify_sidecar(path, "code manifest")
    script = Path(__file__).resolve()
    script_sha256 = sha256_file(script)
    normalized_suffix = f"server/{script.name}".casefold()
    matches = [
        (name, digest)
        for name, digest in code_manifest_entries(path)
        if name.casefold().endswith(normalized_suffix)
        or Path(name).name.casefold() == script.name.casefold()
    ]
    if not matches:
        raise ValueError(f"code manifest does not list {script.name}")
    wrong = [(name, digest) for name, digest in matches if digest != script_sha256]
    if wrong:
        raise ValueError(
            f"code manifest hash for {script.name} differs from current script: {wrong}"
        )
    return manifest_sha256, script_sha256


@dataclass(frozen=True)
class QuoteSpan:
    open_quote: int
    content_start: int
    content_end: int
    close_quote: int


def ascii_quote_spans(data: bytes) -> list[QuoteSpan]:
    positions = [index for index, value in enumerate(data) if value == 0x22]
    if len(positions) % 2:
        raise ValueError("unbalanced ASCII double-quote byte")
    return [
        QuoteSpan(left, left + 1, right, right)
        for left, right in zip(positions[0::2], positions[1::2])
    ]


def paragraph_bytes(text: str) -> list[bytes]:
    raw = text.encode("utf-8")
    return [part.strip() for part in BLANK_LINE_BYTES_RE.split(raw) if part.strip()]


def quote_interiors(data: bytes, spans: Sequence[QuoteSpan]) -> list[bytes]:
    return [data[span.content_start : span.content_end] for span in spans]


def replace_quote_interiors(
    base: bytes, spans: Sequence[QuoteSpan], replacements: Sequence[bytes]
) -> bytes:
    if len(spans) != len(replacements):
        raise ValueError("replacement count differs from quote-span count")
    chunks: list[bytes] = []
    cursor = 0
    for span, replacement in zip(spans, replacements):
        if b'"' in replacement:
            raise ValueError("a quote-interior replacement contains an ASCII quote")
        chunks.append(base[cursor : span.content_start])
        chunks.append(replacement)
        cursor = span.content_end
    chunks.append(base[cursor:])
    return b"".join(chunks)


def outside_quote_segments(data: bytes, spans: Sequence[QuoteSpan]) -> list[bytes]:
    segments: list[bytes] = []
    cursor = 0
    for span in spans:
        segments.append(data[cursor : span.content_start])
        cursor = span.content_end
    segments.append(data[cursor:])
    return segments


def assert_only_quote_interiors_changed(base: bytes, transformed: bytes) -> None:
    base_spans = ascii_quote_spans(base)
    transformed_spans = ascii_quote_spans(transformed)
    if len(base_spans) != len(transformed_spans):
        raise AssertionError("quote count changed during variant construction")
    if outside_quote_segments(base, base_spans) != outside_quote_segments(
        transformed, transformed_spans
    ):
        raise AssertionError("a byte outside a quote interior changed")


def normalize_candidate(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalize_target_anchor(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in folded if not character.isspace())


def canonical_candidate_for_first_token(text: str) -> str:
    if not text:
        return text
    first = text[0]
    if first.isspace() or unicodedata.category(first).startswith("P"):
        return text
    return " " + text


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text, add_special_tokens=False)
    else:
        result = tokenizer(text, add_special_tokens=False)["input_ids"]
    if result and isinstance(result[0], list):
        if len(result) != 1:
            raise ValueError("tokenizer unexpectedly returned a batch")
        result = result[0]
    return [int(value) for value in result]


def first_token_id(tokenizer: Any, candidate: str) -> int:
    ids = tokenizer_ids(tokenizer, canonical_candidate_for_first_token(candidate))
    if not ids:
        raise ValueError("candidate has empty canonical first-token encoding")
    return ids[0]


def unique_in_order(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(values))


def row_uid(row: dict[str, Any]) -> str:
    value = row.get("row_uid")
    if not isinstance(value, str) or not value:
        raise ValueError("row has no non-empty row_uid")
    return value


def target_token_from_rows(
    explanation_row: dict[str, Any], plan_row: dict[str, Any]
) -> str:
    keys = ("target_token", "decoded_target_token", "token")
    candidates: list[tuple[str, str]] = []
    for prefix, row in (("explanation", explanation_row), ("plan", plan_row)):
        for key in keys:
            if key in row:
                candidates.append((f"{prefix}.{key}", str(row[key])))
                break
    if not candidates:
        raise ValueError(f"{row_uid(explanation_row)} has no decoded target token")
    if len({value for _, value in candidates}) != 1:
        raise ValueError(
            f"{row_uid(explanation_row)} target-token mismatch: {candidates}"
        )
    return candidates[0][1]


def validate_plan_identity(
    explanation_row: dict[str, Any], plan_row: dict[str, Any]
) -> None:
    for key in ("row_uid", "content_group_id", "doc_id", "source", "idx"):
        if key in explanation_row and key in plan_row:
            if str(explanation_row[key]) != str(plan_row[key]):
                raise ValueError(
                    f"{row_uid(explanation_row)} plan/explanation {key} mismatch"
                )


def parse_eligible_row(
    row: dict[str, Any],
    plan_row: dict[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    uid = row_uid(row)
    validate_plan_identity(row, plan_row)
    if not isinstance(row.get("source"), str) or not row["source"]:
        raise ValueError(f"{uid} has no source")
    if not isinstance(row.get("explanation"), str):
        raise ValueError(f"{uid} has no explanation string")
    target_token = target_token_from_rows(row, plan_row)
    parts = paragraph_bytes(row["explanation"])
    if "paragraph_count" in row and int(row["paragraph_count"]) != len(parts):
        raise ValueError(f"{uid} upstream paragraph_count differs from byte parser")
    reasons: list[str] = []
    if len(parts) < 3:
        return None, ["fewer_than_three_blank_line_paragraphs"]
    p3 = parts[-1]
    p12 = b"\n\n".join(parts[:-1])
    try:
        spans = ascii_quote_spans(p3)
    except ValueError:
        return None, ["unbalanced_ascii_double_quote"]
    if not 6 <= len(spans) <= 8:
        return None, ["ascii_quote_span_count_not_6_to_8"]
    interiors = [value.decode("utf-8") for value in quote_interiors(p3, spans)]
    target_anchor = interiors[0]
    context_anchor = interiors[1]
    true_candidates = interiors[2:]
    normalized = [normalize_candidate(value) for value in true_candidates]
    if any(not value for value in normalized):
        reasons.append("empty_normalized_candidate")
    normalized_target = normalize_target_anchor(target_token)
    normalized_anchor = normalize_target_anchor(target_anchor)
    if not normalized_target:
        reasons.append("empty_normalized_target_token")
    elif normalized_target not in normalized_anchor:
        reasons.append("target_anchor_substring_gate_failed")
    first_ids: list[int] = []
    candidate_lengths: list[int] = []
    if not reasons:
        try:
            first_ids = [first_token_id(tokenizer, value) for value in true_candidates]
            candidate_lengths = [
                len(tokenizer_ids(tokenizer, value)) for value in true_candidates
            ]
            if any(length == 0 for length in candidate_lengths):
                reasons.append("empty_candidate_token_encoding")
        except (TypeError, ValueError):
            reasons.append("empty_canonical_first_token_encoding")
    if reasons:
        return None, sorted(set(reasons))
    return {
        "input": row,
        "plan": plan_row,
        "idx": int(row["idx"]),
        "row_uid": uid,
        "content_group_id": str(row["content_group_id"]),
        "doc_id": row["doc_id"],
        "source": str(row["source"]),
        "paragraph_count": len(parts),
        "target_token": target_token,
        "target_token_normalized": normalized_target,
        "target_anchor": target_anchor,
        "target_anchor_normalized": normalized_anchor,
        "target_anchor_gate_pass": True,
        "context_anchor": context_anchor,
        "context_anchor_normalized": normalize_candidate(context_anchor),
        "candidate_count": len(true_candidates),
        "true_candidates": true_candidates,
        "true_candidates_normalized": normalized,
        "true_canonical_first_token_ids": first_ids,
        "true_unique_canonical_first_token_ids": unique_in_order(first_ids),
        "true_candidate_token_lengths": candidate_lengths,
        "true_total_candidate_token_length": sum(candidate_lengths),
        "p3_token_length_true": len(tokenizer_ids(tokenizer, p3.decode("utf-8"))),
        "_p3": p3,
        "_p12": p12,
        "_spans": spans,
    }, []


def selection_rank(row: dict[str, Any], seed: int, domain: str) -> str:
    return framed_sha256(
        seed, domain, row["source"], row["candidate_count"], row["row_uid"]
    )


def select_analysis_rows(
    eligible: Sequence[dict[str, Any]],
    *,
    seed: int,
    target: int,
    cell_seed_quota: int,
    min_cell_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target <= 0:
        raise ValueError("--target must be positive")
    if cell_seed_quota < 2:
        raise ValueError("--cell-seed-quota must be at least 2 for derangement")
    if min_cell_size < cell_seed_quota:
        raise ValueError("--min-cell-size must be >= --cell-seed-quota")
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        cells[(row["source"], row["candidate_count"])].append(row)
    retained = {
        cell: sorted(rows, key=lambda row: selection_rank(row, seed, "cell-row"))
        for cell, rows in cells.items()
        if len(rows) >= min_cell_size
    }
    discarded = {
        f"{source}|{count}": [row["row_uid"] for row in rows]
        for (source, count), rows in sorted(cells.items())
        if len(rows) < min_cell_size
    }
    if not retained:
        raise ValueError("no eligibility cell survives --min-cell-size")
    cell_order = sorted(
        retained,
        key=lambda cell: framed_sha256(seed, "cell-order", cell[0], cell[1]),
    )
    selected: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    remaining_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cell_order:
        rows = retained[cell]
        for row in rows[:cell_seed_quota]:
            row["_selection_phase"] = "cell_seed"
            row["_selection_rank_sha256"] = selection_rank(row, seed, "cell-row")
            selected.append(row)
            selected_uids.add(row["row_uid"])
        remaining_by_source[cell[0]].extend(rows[cell_seed_quota:])
    if len(selected) > target:
        raise ValueError(
            f"target {target} is below mandatory cell seed count {len(selected)}"
        )
    capacity = sum(len(rows) for rows in retained.values())
    if target > capacity:
        raise ValueError(
            f"target {target} exceeds retained eligible capacity {capacity}"
        )
    source_order = sorted(
        remaining_by_source,
        key=lambda source: framed_sha256(seed, "source-round-robin", source),
    )
    queues: dict[str, deque[dict[str, Any]]] = {}
    for source in source_order:
        rows = sorted(
            remaining_by_source[source],
            key=lambda row: selection_rank(row, seed, "source-fill"),
        )
        queues[source] = deque(rows)
    fill_by_source: Counter[str] = Counter()
    round_number = 0
    while len(selected) < target:
        round_number += 1
        progressed = False
        for source in source_order:
            if len(selected) >= target:
                break
            if queues[source]:
                row = queues[source].popleft()
                if row["row_uid"] in selected_uids:
                    raise AssertionError("selection emitted a row twice")
                row["_selection_phase"] = "source_round_robin"
                row["_selection_round"] = round_number
                row["_selection_rank_sha256"] = selection_rank(
                    row, seed, "source-fill"
                )
                selected.append(row)
                selected_uids.add(row["row_uid"])
                fill_by_source[source] += 1
                progressed = True
        if not progressed:
            raise AssertionError("round-robin exhausted before target")
    selected_by_cell = Counter(
        f"{row['source']}|{row['candidate_count']}" for row in selected
    )
    selected_by_source = Counter(row["source"] for row in selected)
    audit = {
        "seed": seed,
        "target": target,
        "cell_seed_quota": cell_seed_quota,
        "min_cell_size": min_cell_size,
        "hard_cell_key": ["source", "candidate_count"],
        "eligible_cell_sizes": {
            f"{source}|{count}": len(rows)
            for (source, count), rows in sorted(cells.items())
        },
        "discarded_cells": discarded,
        "retained_cell_order": [
            {"source": source, "candidate_count": count}
            for source, count in cell_order
        ],
        "source_round_robin_order": source_order,
        "round_robin_fill_by_source": dict(sorted(fill_by_source.items())),
        "selected_by_cell": dict(sorted(selected_by_cell.items())),
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "selected_row_uid_sequence_sha256": canonical_sha256(
            [row["row_uid"] for row in selected]
        ),
    }
    return selected, audit


def donor_edge_allowed(recipient: dict[str, Any], donor: dict[str, Any]) -> bool:
    if recipient["row_uid"] == donor["row_uid"]:
        return False
    if recipient["content_group_id"] == donor["content_group_id"]:
        return False
    if str(recipient["doc_id"]) == str(donor["doc_id"]):
        return False
    return not (
        set(recipient["true_candidates_normalized"])
        & set(donor["true_candidates_normalized"])
    )


def hungarian(cost: Sequence[Sequence[int]]) -> list[int]:
    """Deterministic O(n^3) minimum-cost assignment for an integer square matrix."""
    n = len(cost)
    if n == 0 or any(len(row) != n for row in cost):
        raise ValueError("Hungarian input must be a non-empty square matrix")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    infinity = max(max(row) for row in cost) * (n + 2) + 1
    for i in range(1, n + 1):
        p[0] = i
        minv = [infinity] * (n + 1)
        used = [False] * (n + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = infinity
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta or (minv[j] == delta and j < j1):
                    delta = minv[j]
                    j1 = j
            if j1 == 0:
                raise ValueError("assignment has no augmenting path")
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def match_block(
    rows: Sequence[dict[str, Any]], seed: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("donor block has fewer than two selected rows")
    source = rows[0]["source"]
    candidate_count = rows[0]["candidate_count"]
    if any(
        row["source"] != source or row["candidate_count"] != candidate_count
        for row in rows
    ):
        raise ValueError("match_block received mixed hard blocks")
    n = len(rows)
    tie_hash: dict[tuple[int, int], str] = {}
    for i, recipient in enumerate(rows):
        for j, donor in enumerate(rows):
            if donor_edge_allowed(recipient, donor):
                tie_hash[(i, j)] = framed_sha256(
                    seed,
                    "donor-tie",
                    source,
                    candidate_count,
                    recipient["row_uid"],
                    donor["row_uid"],
                )
    if not tie_hash:
        raise ValueError(f"no allowed donor edges in block {(source, candidate_count)}")
    hashes = sorted(tie_hash.values())
    if len(hashes) != len(set(hashes)):
        raise ValueError("SHA-256 tie-rank collision")
    tie_rank_by_hash = {value: rank for rank, value in enumerate(hashes)}
    raw_components: dict[tuple[int, int], tuple[int, ...]] = {}
    for (i, j), digest in tie_hash.items():
        recipient, donor = rows[i], rows[j]
        raw_components[(i, j)] = (
            abs(
                len(recipient["true_unique_canonical_first_token_ids"])
                - len(donor["true_unique_canonical_first_token_ids"])
            ),
            abs(
                recipient["true_total_candidate_token_length"]
                - donor["true_total_candidate_token_length"]
            ),
            sum(
                abs(left - right)
                for left, right in zip(
                    sorted(recipient["true_candidate_token_lengths"]),
                    sorted(donor["true_candidate_token_lengths"]),
                )
            ),
            abs(
                recipient["p3_token_length_true"] - donor["p3_token_length_true"]
            ),
            tie_rank_by_hash[digest],
        )
    edge_max = [
        max(components[index] for components in raw_components.values())
        for index in range(len(COST_COMPONENTS))
    ]
    assignment_sum_bounds = [n * value for value in edge_max]
    weights = [0] * len(COST_COMPONENTS)
    lower_bound = 0
    for index in range(len(COST_COMPONENTS) - 1, -1, -1):
        weights[index] = lower_bound + 1
        lower_bound += assignment_sum_bounds[index] * weights[index]
    scaled = {
        edge: sum(value * weight for value, weight in zip(components, weights))
        for edge, components in raw_components.items()
    }
    max_feasible_total = n * max(scaled.values())
    forbidden_cost = max_feasible_total + 1
    matrix = [
        [scaled.get((i, j), forbidden_cost) for j in range(n)] for i in range(n)
    ]
    assignment = hungarian(matrix)
    if any((i, j) not in raw_components for i, j in enumerate(assignment)):
        raise ValueError(
            f"no full donor derangement for hard block {(source, candidate_count)}"
        )
    if len(set(assignment)) != n:
        raise AssertionError("Hungarian assignment is not one-to-one")
    result: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    component_sums = [0] * len(COST_COMPONENTS)
    for i, j in enumerate(assignment):
        recipient, donor = rows[i], rows[j]
        components = raw_components[(i, j)]
        for index, value in enumerate(components):
            component_sums[index] += value
        record = {
            "recipient_row_uid": recipient["row_uid"],
            "donor_row_uid": donor["row_uid"],
            "components": dict(zip(COST_COMPONENTS, components)),
            "component_vector": list(components),
            "scaled_cost": scaled[(i, j)],
            "tie_sha256": tie_hash[(i, j)],
        }
        result[recipient["row_uid"]] = {"donor": donor, "cost": record}
        records.append(record)
    audit = {
        "source": source,
        "candidate_count": candidate_count,
        "n_rows": n,
        "algorithm": (
            "deterministic integer Hungarian minimum-cost assignment; hard "
            "forbidden self/content-group/document/shared-normalized-candidate edges"
        ),
        "component_order": list(COST_COMPONENTS),
        "edge_component_maxima": dict(zip(COST_COMPONENTS, edge_max)),
        "assignment_sum_upper_bounds": dict(
            zip(COST_COMPONENTS, assignment_sum_bounds)
        ),
        "lexicographic_integer_weights": dict(zip(COST_COMPONENTS, weights)),
        "forbidden_edge_cost": forbidden_cost,
        "component_sums": dict(zip(COST_COMPONENTS, component_sums)),
        "total_scaled_cost": sum(record["scaled_cost"] for record in records),
        "assignments": records,
        "assignments_sha256": canonical_sha256(records),
    }
    return result, audit


def assign_donors(
    selected: Sequence[dict[str, Any]], seed: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    blocks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        blocks[(row["source"], row["candidate_count"])].append(row)
    donor_map: dict[str, dict[str, Any]] = {}
    block_audits: list[dict[str, Any]] = []
    for block in sorted(
        blocks,
        key=lambda value: framed_sha256(seed, "donor-block", value[0], value[1]),
    ):
        mapping, audit = match_block(blocks[block], seed)
        donor_map.update(mapping)
        block_audits.append(audit)
    if set(donor_map) != {row["row_uid"] for row in selected}:
        raise AssertionError("donor map does not cover every selected row")
    ordered_pairs = [
        [row["row_uid"], donor_map[row["row_uid"]]["donor"]["row_uid"]]
        for row in selected
    ]
    return donor_map, {
        "hard_block": ["source", "candidate_count"],
        "component_order": list(COST_COMPONENTS),
        "blocks": block_audits,
        "donor_pair_sequence_sha256": canonical_sha256(ordered_pairs),
    }


def build_variant_row(
    analysis_idx: int,
    recipient: dict[str, Any],
    donor_entry: dict[str, Any],
) -> dict[str, Any]:
    donor = donor_entry["donor"]
    p3: bytes = recipient["_p3"]
    spans: list[QuoteSpan] = recipient["_spans"]
    interiors = quote_interiors(p3, spans)
    donor_candidates = [value.encode("utf-8") for value in donor["true_candidates"]]
    cross_replacements = [*interiors[:2], *donor_candidates]
    candidate_strip_replacements = [
        *interiors[:2],
        *([PLACEHOLDER.encode("ascii")] * recipient["candidate_count"]),
    ]
    anchor_strip_replacements = [
        PLACEHOLDER.encode("ascii"),
        PLACEHOLDER.encode("ascii"),
        *interiors[2:],
    ]
    all_strip_replacements = [PLACEHOLDER.encode("ascii")] * len(spans)
    p3_cross = replace_quote_interiors(p3, spans, cross_replacements)
    p3_candidate_strip = replace_quote_interiors(
        p3, spans, candidate_strip_replacements
    )
    p3_anchor_strip = replace_quote_interiors(p3, spans, anchor_strip_replacements)
    p3_all_quote_strip = replace_quote_interiors(
        p3, spans, all_strip_replacements
    )
    for transformed in (
        p3_cross,
        p3_candidate_strip,
        p3_anchor_strip,
        p3_all_quote_strip,
    ):
        assert_only_quote_interiors_changed(p3, transformed)
    variants = {
        "orig": str(recipient["input"]["explanation"]),
        "p3_true": p3.decode("utf-8"),
        "p3_cross_matched": p3_cross.decode("utf-8"),
        "p3_candidate_strip": p3_candidate_strip.decode("utf-8"),
        "p3_anchor_strip": p3_anchor_strip.decode("utf-8"),
        "p3_all_quote_strip": p3_all_quote_strip.decode("utf-8"),
        "p12": recipient["_p12"].decode("utf-8"),
    }
    if tuple(variants) != VARIANT_KEYS:
        raise AssertionError("variant key/order drift")
    cross_ids = donor["true_canonical_first_token_ids"]
    output = {
        "analysis_idx": analysis_idx,
        "idx": recipient["idx"],
        "row_uid": recipient["row_uid"],
        "content_group_id": recipient["content_group_id"],
        "doc_id": recipient["doc_id"],
        "source": recipient["source"],
        "paragraph_count": recipient["paragraph_count"],
        "target_token": recipient["target_token"],
        "target_token_normalized": recipient["target_token_normalized"],
        "target_anchor": recipient["target_anchor"],
        "target_anchor_normalized": recipient["target_anchor_normalized"],
        "target_anchor_gate_pass": True,
        "context_anchor": recipient["context_anchor"],
        "context_anchor_normalized": recipient["context_anchor_normalized"],
        "candidate_count": recipient["candidate_count"],
        "true_candidates": recipient["true_candidates"],
        "true_candidates_normalized": recipient["true_candidates_normalized"],
        "true_canonical_first_token_ids": recipient[
            "true_canonical_first_token_ids"
        ],
        "true_unique_canonical_first_token_ids": recipient[
            "true_unique_canonical_first_token_ids"
        ],
        "true_candidate_token_lengths": recipient[
            "true_candidate_token_lengths"
        ],
        "true_total_candidate_token_length": recipient[
            "true_total_candidate_token_length"
        ],
        "p3_token_length_true": recipient["p3_token_length_true"],
        "donor_idx": donor["idx"],
        "donor_row_uid": donor["row_uid"],
        "donor_content_group_id": donor["content_group_id"],
        "donor_doc_id": donor["doc_id"],
        "cross_candidates": donor["true_candidates"],
        "cross_candidates_normalized": donor["true_candidates_normalized"],
        "cross_canonical_first_token_ids": cross_ids,
        "cross_unique_canonical_first_token_ids": unique_in_order(cross_ids),
        "cross_candidate_token_lengths": donor["true_candidate_token_lengths"],
        "cross_total_candidate_token_length": donor[
            "true_total_candidate_token_length"
        ],
        "donor_p3_token_length": donor["p3_token_length_true"],
        "donor_cost_components": donor_entry["cost"]["components"],
        "donor_cost_component_vector": donor_entry["cost"]["component_vector"],
        "donor_cost_scaled": donor_entry["cost"]["scaled_cost"],
        "donor_tie_sha256": donor_entry["cost"]["tie_sha256"],
        "selection_phase": recipient["_selection_phase"],
        "selection_round": recipient.get("_selection_round"),
        "selection_rank_sha256": recipient["_selection_rank_sha256"],
        "variants": variants,
        "variant_sha256": {
            key: sha256_text(value) for key, value in variants.items()
        },
        "byte_identity_audit": {
            "only_quote_interiors_changed": True,
            "ascii_quote_span_count": len(spans),
            "recipient_noncandidate_bytes_sha256": canonical_sha256(
                [
                    value.hex()
                    for value in outside_quote_segments(p3, spans)
                ]
            ),
        },
    }
    return output


def rows_from_payload(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label}.rows must be a list of objects")
    return rows


def plan_row_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows_from_payload(payload, "plan"):
        uid = row_uid(row)
        if uid in result:
            raise ValueError(f"duplicate plan row_uid: {uid}")
        result[uid] = row
    return result


def validate_explanations_complete(
    plan: dict[str, Any], explanations: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_status = "COMPLETE_FROZEN_BEFORE_TEXT_ELIGIBILITY_AND_AR"
    if explanations.get("status") != expected_status:
        raise ValueError(
            "AV explanations are not complete before text eligibility: "
            f"{explanations.get('status')!r}"
        )
    plan_rows = rows_from_payload(plan, "plan")
    explanation_rows = rows_from_payload(explanations, "AV explanations")
    plan_uids = [row_uid(row) for row in plan_rows]
    explanation_uids = [row_uid(row) for row in explanation_rows]
    if len(set(plan_uids)) != len(plan_uids):
        raise ValueError("plan row_uid values are not unique")
    if explanation_uids != plan_uids:
        raise ValueError(
            "AV explanations row_uid sequence/coverage differs from the frozen plan"
        )
    checks = explanations.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("AV explanations lack checks")
    expected_checks = {
        "n_rows": len(explanation_rows),
        "n_unique_row_uids": len(set(explanation_uids)),
        "row_uid_sequence_sha256": canonical_sha256(explanation_uids),
    }
    for key, expected in expected_checks.items():
        observed = checks.get(key)
        if observed != expected:
            raise ValueError(
                f"AV explanations checks.{key}={observed!r}, expected {expected!r}"
            )
    plan_checks = plan.get("checks")
    if not isinstance(plan_checks, dict):
        raise ValueError("plan lacks checks")
    if int(plan_checks.get("n_rows", -1)) != len(plan_rows):
        raise ValueError("plan checks.n_rows differs from plan rows")
    if int(plan_checks.get("n_row_uids", -1)) != len(set(plan_uids)):
        raise ValueError("plan checks.n_row_uids differs from plan rows")
    if plan_checks.get("row_uid_sequence_sha256") != canonical_sha256(plan_uids):
        raise ValueError("plan checks.row_uid_sequence_sha256 differs from plan rows")
    return plan_rows, explanation_rows


def load_tokenizer(base_model: Path) -> Any:
    if not base_model.exists():
        raise FileNotFoundError(f"--base-model must be a local path: {base_model}")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required only for a real freeze run"
        ) from error
    return AutoTokenizer.from_pretrained(
        str(base_model),
        local_files_only=True,
        trust_remote_code=False,
    )


def run(args: argparse.Namespace) -> str:
    prereg = Path(args.prereg)
    reject_draft_prereg(prereg)
    prereg_sha256 = verify_sidecar(prereg, "final preregistration")
    plan_path = Path(args.plan)
    explanations_path = Path(args.explanations)
    plan_sha256 = verify_sidecar(plan_path, "plan")
    explanations_sha256 = verify_sidecar(explanations_path, "AV explanations")
    code_manifest_sha256, script_sha256 = verify_current_script_in_manifest(
        Path(args.code_manifest)
    )
    plan = load_json_object(plan_path, "plan")
    explanations = load_json_object(explanations_path, "AV explanations")
    plan_prereg = nested_hash(
        plan,
        "plan",
        ("inputs", "prereg_sha256"),
        ("inputs", "preregistration_sha256"),
        ("prereg_sha256",),
        ("preregistration_sha256",),
    )
    if plan_prereg != prereg_sha256:
        raise ValueError("plan preregistration hash mismatch")
    explanation_prereg = nested_hash(
        explanations,
        "AV explanations",
        ("inputs", "prereg_sha256"),
        ("inputs", "preregistration_sha256"),
    )
    explanation_plan = nested_hash(
        explanations, "AV explanations", ("inputs", "plan_sha256")
    )
    if explanation_prereg != prereg_sha256:
        raise ValueError("AV explanations preregistration hash mismatch")
    if explanation_plan != plan_sha256:
        raise ValueError("AV explanations plan hash mismatch")
    model_manifest_sha256 = nested_hash(
        explanations, "AV explanations", ("inputs", "model_manifest_sha256")
    )
    activations_sha256 = nested_hash(
        explanations, "AV explanations", ("inputs", "activations_sha256")
    )
    upstream_code_manifest = explanations.get("inputs", {}).get(
        "code_manifest_sha256"
    )
    if upstream_code_manifest is not None:
        if require_hex64(
            upstream_code_manifest, "AV explanations code_manifest_sha256"
        ) != code_manifest_sha256:
            raise ValueError("AV explanations code-manifest hash mismatch")
    _, input_rows = validate_explanations_complete(plan, explanations)
    tokenizer_hashes = verify_tokenizer_identity(Path(args.base_model), plan)
    tokenizer = load_tokenizer(Path(args.base_model))
    plan_rows = plan_row_map(plan)
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in input_rows:
        uid = row_uid(row)
        if uid in seen:
            raise ValueError(f"duplicate explanation row_uid: {uid}")
        seen.add(uid)
        if uid not in plan_rows:
            raise ValueError(f"explanation row not present in plan: {uid}")
        parsed, reasons = parse_eligible_row(row, plan_rows[uid], tokenizer)
        if parsed is None:
            rejected.append(
                {
                    "row_uid": uid,
                    "idx": row.get("idx"),
                    "source": row.get("source"),
                    "reasons": reasons,
                }
            )
        else:
            eligible.append(parsed)
    selected, selection_audit = select_analysis_rows(
        eligible,
        seed=args.seed,
        target=args.target,
        cell_seed_quota=args.cell_seed_quota,
        min_cell_size=args.min_cell_size,
    )
    donor_map, donor_audit = assign_donors(selected, args.seed)
    output_rows = [
        build_variant_row(index, row, donor_map[row["row_uid"]])
        for index, row in enumerate(selected)
    ]
    reason_counts = Counter(
        reason for row in rejected for reason in row["reasons"]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "N6 frozen eligible cohort, hard-source donors, and variants",
        "status": "COMPLETE_FROZEN_BEFORE_AR_CANDIDATE_MASS_OR_CAUSAL_OUTCOME",
        "inputs": {
            "prereg": str(prereg),
            "prereg_sha256": prereg_sha256,
            "plan": str(plan_path),
            "plan_sha256": plan_sha256,
            "explanations": str(explanations_path),
            "explanations_sha256": explanations_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "activations_sha256": activations_sha256,
            "code_manifest": str(args.code_manifest),
            "code_manifest_sha256": code_manifest_sha256,
            "script_sha256": script_sha256,
            "base_model": str(args.base_model),
            "tokenizer_file_sha256": tokenizer_hashes,
        },
        "parser_contract": {
            "paragraph_separator_bytes_regex": BLANK_LINE_BYTES_RE.pattern.decode(
                "ascii"
            ),
            "paragraph_trim": "bytes.strip()",
            "p3": "last non-empty blank-line-separated paragraph",
            "p12": "all prior stripped paragraphs joined by two LF bytes",
            "quote_parser": (
                "pair consecutive ASCII 0x22 bytes; reject an odd count; "
                "eligible iff 6..8 spans"
            ),
            "candidate_normalization": (
                "Unicode NFKC then Python str.split whitespace collapse with "
                "single ASCII spaces"
            ),
            "target_anchor_normalization": (
                "Unicode NFKC, casefold, remove every character for which "
                "str.isspace() is true"
            ),
            "target_anchor_gate": (
                "normalized target is non-empty and is a substring of normalized "
                "quoted span 1"
            ),
            "context_anchor": "quoted span 2; diagnostic only",
            "canonical_first_token": (
                "prepend one ASCII space unless raw candidate begins with Unicode "
                "whitespace or a Unicode P* punctuation code point; tokenize "
                "without special tokens"
            ),
            "candidate_token_length": (
                "tokenize the exact raw quote interior without special tokens"
            ),
        },
        "eligibility": {
            "n_input": len(input_rows),
            "n_eligible_before_cell_filter": len(eligible),
            "n_text_ineligible": len(rejected),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            "rejected_rows": rejected,
        },
        "selection": selection_audit,
        "donor_assignment": donor_audit,
        "variant_contract": {
            "placeholder_interior_bytes": PLACEHOLDER,
            "variant_keys": list(VARIANT_KEYS),
            "only_ascii_quote_interiors_may_change": True,
        },
        "rows": output_rows,
    }
    output_sha256 = write_frozen_json(Path(args.out), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": len(output_rows),
                "eligible": len(eligible),
                "rejected": len(rejected),
                "output": str(args.out),
                "sha256": output_sha256,
            },
            sort_keys=True,
        )
    )
    return output_sha256


class _SelfTestTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("self-test tokenizer must not receive special tokens")
        words = text.split()
        return [
            int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)
            for word in words
        ]


def _self_test_input(
    index: int,
    source: str,
    *,
    token: str = "Token",
    candidates: Sequence[str] = ("alpha", "beta", "gamma", "delta"),
    anchor: str = "To k en",
) -> tuple[dict[str, Any], dict[str, Any]]:
    quoted = [anchor, "local context", *candidates]
    p3 = "Prediction " + " / ".join(f'"{value}"' for value in quoted) + "."
    explanation = f"Paragraph one.\n\nParagraph two.\n\n{p3}"
    uid = f"row-{source}-{index}"
    row = {
        "idx": index,
        "row_uid": uid,
        "content_group_id": f"group-{source}-{index}",
        "doc_id": f"doc-{source}-{index}",
        "source": source,
        "explanation": explanation,
        "paragraph_count": 3,
    }
    plan = {
        "idx": index,
        "row_uid": uid,
        "content_group_id": row["content_group_id"],
        "doc_id": row["doc_id"],
        "source": source,
        "token": token,
    }
    return row, plan


def self_test() -> None:
    tokenizer = _SelfTestTokenizer()
    row, plan = _self_test_input(0, "A")
    parsed, reasons = parse_eligible_row(row, plan, tokenizer)
    assert parsed is not None and not reasons
    assert parsed["candidate_count"] == 4
    assert len(parsed["_spans"]) == 6
    bad_row, bad_plan = _self_test_input(1, "A", token="absent")
    bad, bad_reasons = parse_eligible_row(bad_row, bad_plan, tokenizer)
    assert bad is None and "target_anchor_substring_gate_failed" in bad_reasons

    # Stage-51 completeness is exact: a self-consistent subset still cannot pass.
    closure_plan_rows = [plan, bad_plan]
    closure_uids = [row_uid(value) for value in closure_plan_rows]
    closure_plan = {
        "inputs": {},
        "checks": {
            "n_rows": len(closure_plan_rows),
            "n_row_uids": len(set(closure_uids)),
            "row_uid_sequence_sha256": canonical_sha256(closure_uids),
        },
        "rows": closure_plan_rows,
    }
    closure_explanation_rows = [row, bad_row]
    closure_explanations = {
        "status": "COMPLETE_FROZEN_BEFORE_TEXT_ELIGIBILITY_AND_AR",
        "checks": {
            "n_rows": len(closure_explanation_rows),
            "n_unique_row_uids": len(
                {row_uid(value) for value in closure_explanation_rows}
            ),
            "row_uid_sequence_sha256": canonical_sha256(
                [row_uid(value) for value in closure_explanation_rows]
            ),
        },
        "rows": closure_explanation_rows,
    }
    validate_explanations_complete(closure_plan, closure_explanations)
    subset = json.loads(json.dumps(closure_explanations))
    subset["rows"] = subset["rows"][:1]
    subset_uids = [row_uid(value) for value in subset["rows"]]
    subset["checks"] = {
        "n_rows": 1,
        "n_unique_row_uids": 1,
        "row_uid_sequence_sha256": canonical_sha256(subset_uids),
    }
    try:
        validate_explanations_complete(closure_plan, subset)
    except ValueError as error:
        assert "sequence/coverage" in str(error)
    else:
        raise AssertionError("self-consistent explanation subset unexpectedly passed")

    # The tokenizer/config helper uses exactly the stage-49 identity file set.
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "tokenizer.json").write_text("tok", encoding="utf-8")
        (model_dir / "tokenizer.extra").write_text("extra", encoding="utf-8")
        (model_dir / "config.json").write_text("cfg", encoding="utf-8")
        (model_dir / "weights.bin").write_text("ignored", encoding="utf-8")
        expected_hashes = tokenizer_file_hashes(model_dir)
        assert set(expected_hashes) == {
            "config.json",
            "tokenizer.extra",
            "tokenizer.json",
        }
        identity_plan = {"inputs": {"tokenizer_file_sha256": expected_hashes}}
        assert verify_tokenizer_identity(model_dir, identity_plan) == expected_hashes
        wrong_identity = json.loads(json.dumps(identity_plan))
        wrong_identity["inputs"]["tokenizer_file_sha256"]["config.json"] = "0" * 64
        try:
            verify_tokenizer_identity(model_dir, wrong_identity)
        except ValueError as error:
            assert "wrong=['config.json']" in str(error)
        else:
            raise AssertionError("wrong tokenizer/config hash unexpectedly passed")

    # Deterministic cell seeding followed by balanced source round-robin.
    eligible: list[dict[str, Any]] = []
    for source in ("A", "B"):
        for index in range(5):
            test_row, test_plan = _self_test_input(
                10 * (ord(source) - 64) + index,
                source,
                candidates=(
                    f"{source}{index}alpha",
                    f"{source}{index}beta",
                    f"{source}{index}gamma",
                    f"{source}{index}delta",
                ),
            )
            item, item_reasons = parse_eligible_row(test_row, test_plan, tokenizer)
            assert item is not None and not item_reasons
            eligible.append(item)
    selected, selection = select_analysis_rows(
        eligible, seed=7, target=8, cell_seed_quota=2, min_cell_size=2
    )
    assert len(selected) == 8
    assert selection["round_robin_fill_by_source"] == {"A": 2, "B": 2}
    assert selection["selected_by_source"] == {"A": 4, "B": 4}

    donor_map, donor_audit = assign_donors(selected, seed=7)
    assert donor_audit["blocks"]
    assert len({entry["donor"]["row_uid"] for entry in donor_map.values()}) == 8
    for recipient in selected:
        donor = donor_map[recipient["row_uid"]]["donor"]
        assert donor["row_uid"] != recipient["row_uid"]
        assert donor["source"] == recipient["source"]
        assert not (
            set(donor["true_candidates_normalized"])
            & set(recipient["true_candidates_normalized"])
        )
    output_row = build_variant_row(0, selected[0], donor_map[selected[0]["row_uid"]])
    assert tuple(output_row["variants"]) == VARIANT_KEYS
    base = output_row["variants"]["p3_true"].encode("utf-8")
    for key in (
        "p3_cross_matched",
        "p3_candidate_strip",
        "p3_anchor_strip",
        "p3_all_quote_strip",
    ):
        assert_only_quote_interiors_changed(
            base, output_row["variants"][key].encode("utf-8")
        )

    # A normalized candidate shared by both members forbids the only derangement.
    left_row, left_plan = _self_test_input(
        90, "Z", candidates=("shared", "l2", "l3", "l4")
    )
    right_row, right_plan = _self_test_input(
        91, "Z", candidates=("shared", "r2", "r3", "r4")
    )
    left, _ = parse_eligible_row(left_row, left_plan, tokenizer)
    right, _ = parse_eligible_row(right_row, right_plan, tokenizer)
    assert left is not None and right is not None
    assert not donor_edge_allowed(left, right)
    try:
        match_block([left, right], seed=7)
    except ValueError as error:
        assert "allowed donor edges" in str(error) or "derangement" in str(error)
    else:
        raise AssertionError("shared-candidate-only block unexpectedly matched")
    print("SELF_TEST_OK")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prereg", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--explanations", "--input", dest="explanations", type=Path)
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--target", type=int)
    parser.add_argument(
        "--cell-seed-quota", "--quota", dest="cell_seed_quota", type=int, default=2
    )
    parser.add_argument("--min-cell-size", "--min-cell", dest="min_cell_size", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            name
            for name in (
                "prereg",
                "plan",
                "explanations",
                "code_manifest",
                "base_model",
                "out",
                "target",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("real freeze run requires: " + ", ".join(f"--{x.replace('_', '-')}" for x in missing))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
