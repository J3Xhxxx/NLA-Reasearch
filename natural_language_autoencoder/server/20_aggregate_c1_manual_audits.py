#!/usr/bin/env python3
"""Conservatively aggregate two independent C1 corpus reviews.

The reviewers may use different internal layouts for their complete decision
matrices.  This aggregator preserves both matrices verbatim, verifies their
shared frozen inputs and declared coverage, and applies the preregistered rule:
both reviews must be PASS; any failure or disagreement makes the aggregate
FAIL.  It never reads model activations or endpoint results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_COUNTS = {
    "combined": 144,
    "discovery": 96,
    "heldout": 48,
    "concepts": 24,
    "pairs": 12,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def jsonl_count(path: Path) -> int:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path} has empty or duplicate IDs")
    return len(rows)


def reviewer_id(review: dict[str, Any], fallback: str) -> str:
    value = review.get("reviewer")
    if isinstance(value, dict):
        value = value.get("reviewer_id")
    if not value:
        reviewers = review.get("reviewers")
        if isinstance(reviewers, list) and reviewers:
            first = reviewers[0]
            value = (
                first.get("reviewer_id")
                if isinstance(first, dict)
                else first
            )
    return str(value or fallback)


def validate_review(
    path: Path,
    review: dict[str, Any],
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    for field, expected in expected_hashes.items():
        if str(review.get(field, "")).lower() != expected:
            raise ValueError(
                f"{path} {field} differs from frozen input: "
                f"{review.get(field)!r} != {expected}"
            )
    if len(review.get("document_decisions", [])) != REQUIRED_COUNTS["concepts"]:
        raise ValueError(f"{path} must cover 24 concept document batches")
    if len(review.get("concept_batch_decisions", [])) != REQUIRED_COUNTS["concepts"]:
        raise ValueError(f"{path} must cover 24 concept batches")
    if len(review.get("pair_decisions", [])) != REQUIRED_COUNTS["pairs"]:
        raise ValueError(f"{path} must cover 12 reciprocal pairs")
    status = str(review.get("status", "")).upper()
    if status not in {"PASS", "FAIL"}:
        raise ValueError(f"{path} has invalid status {status!r}")
    failures = review.get("failure_reasons")
    if not isinstance(failures, list):
        raise ValueError(f"{path} failure_reasons must be a list")
    if (status == "PASS") != (len(failures) == 0):
        raise ValueError(
            f"{path} status and failure_reasons disagree "
            f"(status={status}, failures={len(failures)})"
        )
    return {
        "reviewer_id": reviewer_id(review, path.stem),
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status": status,
        "failure_count": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-a", required=True, type=Path)
    parser.add_argument("--review-b", required=True, type=Path)
    parser.add_argument("--rubric", required=True, type=Path)
    parser.add_argument("--combined-manifest", required=True, type=Path)
    parser.add_argument("--discovery-manifest", required=True, type=Path)
    parser.add_argument("--heldout-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    for path in (
        args.review_a,
        args.review_b,
        args.rubric,
        args.combined_manifest,
        args.discovery_manifest,
        args.heldout_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.review_a.resolve() == args.review_b.resolve():
        raise ValueError("the two independent reviews must be different files")

    observed_counts = {
        "combined": jsonl_count(args.combined_manifest),
        "discovery": jsonl_count(args.discovery_manifest),
        "heldout": jsonl_count(args.heldout_manifest),
    }
    for key, expected in REQUIRED_COUNTS.items():
        if key in observed_counts and observed_counts[key] != expected:
            raise ValueError(
                f"{key} manifest count {observed_counts[key]} != {expected}"
            )

    expected_hashes = {
        "rubric_sha256": sha256_file(args.rubric),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "discovery_manifest_sha256": sha256_file(args.discovery_manifest),
        "heldout_manifest_sha256": sha256_file(args.heldout_manifest),
    }
    review_a = read_json(args.review_a)
    review_b = read_json(args.review_b)
    audit_a = validate_review(args.review_a, review_a, expected_hashes)
    audit_b = validate_review(args.review_b, review_b, expected_hashes)
    if audit_a["reviewer_id"] == audit_b["reviewer_id"]:
        raise ValueError("independent reviews declare the same reviewer ID")

    status = (
        "PASS"
        if audit_a["status"] == audit_b["status"] == "PASS"
        else "FAIL"
    )
    aggregate = {
        "schema_version": 1,
        "experiment": "C1 confirmatory synthetic concept cohort v2",
        "status": status,
        **expected_hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_rule": (
            "Conservative conjunction: both complete independent reviews "
            "must PASS; any reviewer failure or disagreement is aggregate FAIL."
        ),
        "counts": {
            "concepts": REQUIRED_COUNTS["concepts"],
            "combined_documents": observed_counts["combined"],
            "discovery_documents": observed_counts["discovery"],
            "heldout_documents": observed_counts["heldout"],
            "reciprocal_pairs": REQUIRED_COUNTS["pairs"],
        },
        "reviewers": [audit_a, audit_b],
        "document_decisions": [
            {
                "reviewer_id": audit_a["reviewer_id"],
                "decisions": review_a["document_decisions"],
            },
            {
                "reviewer_id": audit_b["reviewer_id"],
                "decisions": review_b["document_decisions"],
            },
        ],
        "concept_batch_decisions": [
            {
                "reviewer_id": audit_a["reviewer_id"],
                "decisions": review_a["concept_batch_decisions"],
            },
            {
                "reviewer_id": audit_b["reviewer_id"],
                "decisions": review_b["concept_batch_decisions"],
            },
        ],
        "pair_decisions": [
            {
                "reviewer_id": audit_a["reviewer_id"],
                "decisions": review_a["pair_decisions"],
            },
            {
                "reviewer_id": audit_b["reviewer_id"],
                "decisions": review_b["pair_decisions"],
            },
        ],
        "failure_reasons": [
            {
                "reviewer_id": audit_a["reviewer_id"],
                "items": review_a["failure_reasons"],
            },
            {
                "reviewer_id": audit_b["reviewer_id"],
                "items": review_b["failure_reasons"],
            },
        ],
        "attestation": (
            "Both source reviews declare complete 24-concept, 144-document, "
            "12-pair inspection without activation or endpoint access. Their "
            "full decision matrices are preserved verbatim above. Because at "
            "least one source review failed, v2 is stopped before activation "
            "extraction and no generated document may be edited or selectively "
            "regenerated."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"C1_MANUAL_AUDIT_{status} out={args.out} "
        f"sha256={sha256_file(args.out)}"
    )


if __name__ == "__main__":
    main()
