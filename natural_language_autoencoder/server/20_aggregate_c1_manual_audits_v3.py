#!/usr/bin/env python3
"""Deterministically aggregate two independently locked C1 v3 reviews."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_COVERAGE = {
    "documents": 144,
    "document_checks": 1296,
    "concept_batches": 24,
    "concept_batch_checks": 120,
    "reciprocal_pairs": 12,
    "pair_checks": 36,
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


def manifest_ids(path: Path) -> list[str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [
        str(row.get("id", "")) if isinstance(row, dict) else ""
        for row in rows
    ]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path} has empty or duplicate document IDs")
    return ids


def reviewer_id(review: dict[str, Any], label: str) -> str:
    reviewer = review.get("reviewer")
    if isinstance(reviewer, dict):
        reviewer = reviewer.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError(f"{label} lacks reviewer.reviewer_id")
    return reviewer.strip()


def normalize_coverage(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}.coverage must be an object")
    observed = {
        key: int(value.get(key, -1)) for key in EXPECTED_COVERAGE
    }
    if observed != EXPECTED_COVERAGE:
        raise ValueError(
            f"{label} coverage differs from frozen addendum: {observed}"
        )
    return observed


def validate_review(
    path: Path,
    review: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    label = str(path)
    if review.get("schema_version") != 1:
        raise ValueError(f"{label} schema_version must be 1")
    for field, expected in hashes.items():
        if str(review.get(field, "")).lower() != expected:
            raise ValueError(
                f"{label} {field} differs from frozen input"
            )
    status = str(review.get("status", "")).upper()
    if status not in {"PASS", "FAIL"}:
        raise ValueError(f"{label} has invalid status {status!r}")
    failures = review.get("failure_reasons")
    if not isinstance(failures, list):
        raise ValueError(f"{label}.failure_reasons must be a list")
    if (status == "PASS") != (len(failures) == 0):
        raise ValueError(
            f"{label} status disagrees with {len(failures)} failures"
        )
    if len(review.get("document_decisions", [])) != 24:
        raise ValueError(f"{label} must contain 24 document batches")
    if len(review.get("concept_batch_decisions", [])) != 24:
        raise ValueError(f"{label} must contain 24 concept batches")
    if len(review.get("pair_decisions", [])) != 12:
        raise ValueError(f"{label} must contain 12 reciprocal pairs")
    coverage = normalize_coverage(review.get("coverage"), label)
    return {
        "reviewer_id": reviewer_id(review, label),
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status": status,
        "failure_count": len(failures),
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-a", required=True, type=Path)
    parser.add_argument("--review-b", required=True, type=Path)
    parser.add_argument("--base-rubric", required=True, type=Path)
    parser.add_argument("--addendum", required=True, type=Path)
    parser.add_argument("--scenario-anchors", required=True, type=Path)
    parser.add_argument("--combined-manifest", required=True, type=Path)
    parser.add_argument("--discovery-manifest", required=True, type=Path)
    parser.add_argument("--heldout-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    inputs = (
        args.review_a,
        args.review_b,
        args.base_rubric,
        args.addendum,
        args.scenario_anchors,
        args.combined_manifest,
        args.discovery_manifest,
        args.heldout_manifest,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.review_a.resolve() == args.review_b.resolve():
        raise ValueError("independent reviews must be distinct files")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {args.out}")

    combined_ids = manifest_ids(args.combined_manifest)
    discovery_ids = manifest_ids(args.discovery_manifest)
    heldout_ids = manifest_ids(args.heldout_manifest)
    if (
        len(combined_ids) != 144
        or len(discovery_ids) != 96
        or len(heldout_ids) != 48
        or set(discovery_ids) & set(heldout_ids)
        or set(combined_ids) != set(discovery_ids) | set(heldout_ids)
    ):
        raise ValueError("v3 manifest split identity is not 144=96+48")

    hashes = {
        "base_rubric_sha256": sha256_file(args.base_rubric),
        "addendum_sha256": sha256_file(args.addendum),
        "scenario_anchors_sha256": sha256_file(args.scenario_anchors),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "discovery_manifest_sha256": sha256_file(
            args.discovery_manifest
        ),
        "heldout_manifest_sha256": sha256_file(args.heldout_manifest),
    }
    addendum = read_json(args.addendum)
    if (
        addendum.get("status") != "frozen_before_v3r2_generation"
        or addendum.get("inherits", {}).get("sha256")
        != hashes["base_rubric_sha256"]
        or addendum.get("scenario_anchors", {}).get("sha256")
        != hashes["scenario_anchors_sha256"]
        or addendum.get("required_output", {}).get("coverage")
        != EXPECTED_COVERAGE
    ):
        raise ValueError("audit addendum differs from supplied frozen assets")

    review_a = read_json(args.review_a)
    review_b = read_json(args.review_b)
    audit_a = validate_review(args.review_a, review_a, hashes)
    audit_b = validate_review(args.review_b, review_b, hashes)
    if audit_a["reviewer_id"] == audit_b["reviewer_id"]:
        raise ValueError("independent reviews declare the same reviewer ID")

    status = (
        "PASS"
        if audit_a["status"] == audit_b["status"] == "PASS"
        else "FAIL"
    )
    payload = {
        "schema_version": 1,
        "experiment": "C1 confirmatory synthetic concept cohort v3",
        "status": status,
        "rubric_sha256": hashes["addendum_sha256"],
        **hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_rule": (
            "Deterministic conservative conjunction after two independent "
            "complete review files are hash-locked. Both must PASS."
        ),
        "coverage": EXPECTED_COVERAGE,
        "counts": {
            "concepts": 24,
            "combined_documents": 144,
            "discovery_documents": 96,
            "heldout_documents": 48,
            "reciprocal_pairs": 12,
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
            "Both source reviews declare complete independent inspection "
            "against the frozen base rubric and v3 addendum without access to "
            "the other review, activations, feature IDs, AV/AR outputs, "
            "held-out metrics, or endpoints. Full decision matrices are "
            "preserved verbatim. The aggregate status cannot be manually "
            "reconciled or upgraded."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"C1_V3_MANUAL_AUDIT_{status} out={args.out} "
        f"sha256={sha256_file(args.out)}"
    )


if __name__ == "__main__":
    main()
