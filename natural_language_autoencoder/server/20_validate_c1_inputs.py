#!/usr/bin/env python3
"""Validate frozen C1-confirmatory inputs without running any experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--denylist", required=True, type=Path)
    parser.add_argument("--prior-selection", required=True, type=Path)
    parser.add_argument("--legacy-selector", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    denylist = json.loads(args.denylist.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_selection.read_text(encoding="utf-8"))
    concepts = spec["concepts"]
    concept_by_id = {row["id"]: row for row in concepts}
    errors: list[str] = []
    if len(concepts) != 24 or len(concept_by_id) != 24:
        errors.append("expected_24_unique_concepts")
    groups = Counter(row["superdomain"] for row in concepts)
    if len(groups) != 6 or set(groups.values()) != {4}:
        errors.append(f"superdomain_balance={dict(groups)}")
    prior_overlap = sorted(
        set(concept_by_id) & set(spec["excluded_prior_topics"])
    )
    if prior_overlap:
        errors.append(f"prior_topic_overlap={prior_overlap}")
    for concept in concepts:
        negative = concept_by_id.get(concept["hard_negative_id"])
        if negative is None:
            errors.append(f"missing_negative={concept['id']}")
        elif (
            negative["hard_negative_id"] != concept["id"]
            or negative["superdomain"] != concept["superdomain"]
        ):
            errors.append(f"nonreciprocal_negative={concept['id']}")

    source_by_name = {
        Path(row["path"]).name: row for row in denylist["sources"]
    }
    expected_prior_sha = source_by_name[args.prior_selection.name]["sha256"]
    expected_selector_sha = source_by_name[args.legacy_selector.name]["sha256"]
    if sha256_file(args.prior_selection) != expected_prior_sha:
        errors.append("prior_selection_sha256_mismatch")
    if sha256_file(args.legacy_selector) != expected_selector_sha:
        errors.append("legacy_selector_sha256_mismatch")
    denied = {
        int(row["feature"])
        for row in prior["selected_directions"]
        if int(row["feature"]) >= 0
    }
    for rows in prior["top_candidates_by_label"].values():
        denied.update(
            int(row["feature"])
            for row in rows
            if int(row["feature"]) >= 0
        )
    denied.update(int(value) for value in denylist["legacy_exclude"])
    if len(denied) != denylist["expected_unique_resolved_ids"]:
        errors.append(
            f"denylist_count={len(denied)} expected="
            f"{denylist['expected_unique_resolved_ids']}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    length_checks = []
    for concept in concepts:
        negative = concept_by_id[concept["hard_negative_id"]]
        for template_index, template in enumerate(
            spec["reference_templates"]
        ):
            correct = template.format(**concept)
            foil = template.format(**negative)
            correct_tokens = len(
                tokenizer(
                    correct, add_special_tokens=False
                )["input_ids"]
            )
            foil_tokens = len(
                tokenizer(foil, add_special_tokens=False)["input_ids"]
            )
            ratio = correct_tokens / max(1, foil_tokens)
            length_checks.append(
                {
                    "concept_id": concept["id"],
                    "hard_negative_id": negative["id"],
                    "template_index": template_index,
                    "correct_tokens": correct_tokens,
                    "hard_negative_tokens": foil_tokens,
                    "ratio": ratio,
                }
            )
            if not 0.85 <= ratio <= 1.15:
                errors.append(
                    f"candidate_token_ratio={concept['id']}:"
                    f"{template_index}:{ratio:.4f}"
                )

    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "counts": {
            "concepts": len(concepts),
            "superdomains": len(groups),
            "reciprocal_pairs": len(concepts) // 2,
            "resolved_denied_features": len(denied),
            "reference_length_checks": len(length_checks),
        },
        "hashes": {
            "spec": sha256_file(args.spec),
            "denylist": sha256_file(args.denylist),
            "prior_selection": sha256_file(args.prior_selection),
            "legacy_selector": sha256_file(args.legacy_selector),
            "preregistration": sha256_file(args.preregistration),
        },
        "reference_length_checks": length_checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
