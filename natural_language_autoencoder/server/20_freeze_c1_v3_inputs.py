#!/usr/bin/env python3
"""Freeze all C1 v3 corpus-development inputs before text generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAILED_V3_ANCHORS_SHA256 = (
    "061903133b748ccffe2f85f697c5e6a7d53fd631e63fb1c0302ae42fbd59e6d5"
)
FAILED_V3_ANCHOR_AUDIT_SHA256 = (
    "5dcc267ad9f6bf9765bbb7c6ab3965498c275c7df3165780f5f1d79301c2a396"
)


PLANNED_OUTPUTS = {
    "append_only_checkpoint": (
        "/root/autodl-tmp/results/"
        "c1_confirmatory_corpus_checkpoint_v3.jsonl"
    ),
    "corpus_report": (
        "/root/autodl-tmp/results/c1_confirmatory_corpus_report_v3.json"
    ),
    "combined_manifest": (
        "/root/autodl-tmp/activations/c1_confirmatory_all_v3.jsonl"
    ),
    "discovery_manifest": (
        "/root/autodl-tmp/activations/"
        "c1_confirmatory_discovery_v3.jsonl"
    ),
    "heldout_manifest": (
        "/root/autodl-tmp/activations/c1_confirmatory_heldout_v3.jsonl"
    ),
    "job_log": (
        "/root/autodl-tmp/results/c1_confirmatory_corpus_v3.log"
    ),
    "gpu_monitor": (
        "/root/autodl-tmp/results/c1_confirmatory_corpus_gpu_v3.csv"
    ),
    "discovery_activation_parquet": (
        "/root/autodl-tmp/activations/"
        "acts_L32_c1_confirmatory_discovery_v3.parquet"
    ),
    "heldout_activation_parquet": (
        "/root/autodl-tmp/activations/"
        "acts_L32_c1_confirmatory_heldout_v3.parquet"
    ),
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


def file_record(role: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept-spec", required=True, type=Path)
    parser.add_argument("--denylist", required=True, type=Path)
    parser.add_argument("--base-preregistration", required=True, type=Path)
    parser.add_argument("--v2-amendment", required=True, type=Path)
    parser.add_argument("--v2-stage0-freeze", required=True, type=Path)
    parser.add_argument("--scenario-anchors", required=True, type=Path)
    parser.add_argument(
        "--preregistration-amendment", required=True, type=Path
    )
    parser.add_argument("--base-rubric", required=True, type=Path)
    parser.add_argument("--audit-addendum", required=True, type=Path)
    parser.add_argument("--audit-aggregator", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--anchor-audit", required=True, type=Path)
    parser.add_argument("--v2-aggregate-audit", required=True, type=Path)
    parser.add_argument("--v2-failure-record", required=True, type=Path)
    parser.add_argument("--model-freeze", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "concept_spec": args.concept_spec,
        "denylist_resolution_spec": args.denylist,
        "base_preregistration": args.base_preregistration,
        "v2_preregistration_amendment": args.v2_amendment,
        "v2_stage0_freeze": args.v2_stage0_freeze,
        "scenario_anchors": args.scenario_anchors,
        "preregistration_amendment": args.preregistration_amendment,
        "base_audit_rubric": args.base_rubric,
        "audit_addendum": args.audit_addendum,
        "audit_aggregator": args.audit_aggregator,
        "corpus_generator": args.generator,
        "corpus_runner": args.runner,
        "scenario_anchor_audit": args.anchor_audit,
        "v2_aggregate_audit": args.v2_aggregate_audit,
        "v2_failure_record": args.v2_failure_record,
        "model_freeze": args.model_freeze,
        "stage0_freezer": Path(__file__).resolve(),
    }
    files = [
        file_record(role, path) for role, path in inputs.items()
    ]
    by_role = {row["role"]: row for row in files}

    anchors = read_json(args.scenario_anchors)
    addendum = read_json(args.audit_addendum)
    anchor_audit = read_json(args.anchor_audit)
    v2_audit = read_json(args.v2_aggregate_audit)
    model_freeze = read_json(args.model_freeze)
    denylist = read_json(args.denylist)
    v2_stage0 = read_json(args.v2_stage0_freeze)

    if (
        anchors.get("schema_version") != 1
        or anchors.get("status") != "frozen_before_v3r2_generation"
        or len(anchors.get("concepts", [])) != 24
        or len(anchors.get("discourse_slots", [])) != 6
    ):
        raise ValueError("scenario anchors are not the frozen 24x6 v3r2 asset")
    anchor_count = sum(
        len(row.get("train", [])) + len(row.get("test", []))
        for row in anchors["concepts"]
    )
    if anchor_count != 144:
        raise ValueError(f"scenario-anchor count is {anchor_count}, not 144")
    anchor_sources = anchors.get("sources", {})
    anchor_revision = anchors.get("revision", {})
    if (
        anchor_sources.get("scenario_anchors_v3", {}).get("sha256")
        != FAILED_V3_ANCHORS_SHA256
        or anchor_sources.get("scenario_anchor_audit_v3", {}).get("sha256")
        != FAILED_V3_ANCHOR_AUDIT_SHA256
        or str(
            anchor_sources.get("scenario_anchor_audit_v3", {}).get(
                "status", ""
            )
        ).upper()
        != "FAIL"
        or anchor_revision.get("pre_text_design_iteration") is not True
        or anchor_revision.get("changed_anchor_count") != 33
        or anchor_revision.get("unchanged_anchor_count") != 111
        or "If an independent conservative audit of v3r2 fails"
        not in str(anchor_revision.get("stop_rule", ""))
    ):
        raise ValueError(
            "v3r2 anchors do not preserve the failed draft/audit binding "
            "and final-iteration stop rule"
        )

    if (
        addendum.get("schema_version") != 1
        or addendum.get("status") != "frozen_before_v3r2_generation"
        or addendum.get("inherits", {}).get("sha256")
        != by_role["base_audit_rubric"]["sha256"]
        or addendum.get("scenario_anchors", {}).get("sha256")
        != by_role["scenario_anchors"]["sha256"]
    ):
        raise ValueError("v3r2 audit addendum does not bind rubric and anchors")
    addendum_revision = addendum.get("revision", {})
    if (
        addendum_revision.get("failed_anchor_draft_sha256")
        != FAILED_V3_ANCHORS_SHA256
        or addendum_revision.get("failed_anchor_audit_sha256")
        != FAILED_V3_ANCHOR_AUDIT_SHA256
        or addendum_revision.get("generated_v3_text_existed_before_revision")
        is not False
        or addendum_revision.get("further_anchor_iteration_if_v3r2_fails")
        is not False
        or addendum_revision.get(
            "all_document_batch_and_pair_checks_unchanged"
        )
        is not True
    ):
        raise ValueError("v3r2 audit addendum revision contract changed")
    if (
        str(anchor_audit.get("status", "")).upper() != "PASS"
        or anchor_audit.get("scenario_anchors_sha256")
        != by_role["scenario_anchors"]["sha256"]
    ):
        raise ValueError("independent scenario-anchor audit did not PASS")
    if str(v2_audit.get("status", "")).upper() != "FAIL":
        raise ValueError("v2 aggregate audit must preserve the disclosed FAIL")
    if model_freeze.get("status") != "models_frozen_before_C1_AV_AR":
        raise ValueError("model freeze status is invalid")
    if "base" not in model_freeze.get("models", {}):
        raise ValueError("model freeze lacks the base model")
    if int(denylist.get("expected_unique_resolved_ids", -1)) != 1282:
        raise ValueError("denylist no longer freezes 1282 prior feature IDs")
    if (
        v2_stage0.get("stage") != "pre_generation"
        or v2_stage0.get("files") is None
    ):
        raise ValueError("v2 stage0 freeze is invalid")

    amendment_text = args.preregistration_amendment.read_text(
        encoding="utf-8"
    )
    normalized_amendment = " ".join(amendment_text.split())
    required_phrases = (
        "first complete six-document batch that passes",
        "adaptive corpus redevelopment",
        "independently hash-locked",
        "no difficulty-based reassignment",
    )
    missing_phrases = [
        phrase
        for phrase in required_phrases
        if phrase not in normalized_amendment
    ]
    if missing_phrases:
        raise ValueError(
            f"v3 amendment lacks required safeguards: {missing_phrases}"
        )

    planned = []
    for role, value in PLANNED_OUTPUTS.items():
        path = Path(value)
        exists = path.exists()
        planned.append(
            {"role": role, "path": value, "exists_at_freeze": exists}
        )
    present = [row["path"] for row in planned if row["exists_at_freeze"]]
    if present:
        raise FileExistsError(
            f"v3 planned outputs already exist at freeze: {present}"
        )
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite freeze: {args.out}")

    payload = {
        "schema_version": 3,
        "experiment": "C1 confirmatory synthetic concept cohort v3",
        "stage": "pre_generation",
        "status": (
            "frozen_after_disclosed_v2_semantic_failure_before_v3_generation"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adaptation": {
            "type": "fully disclosed pre-activation adaptive corpus redevelopment",
            "v2_aggregate_status": "FAIL",
            "v2_aggregate_sha256": by_role["v2_aggregate_audit"]["sha256"],
            "all_144_v3_documents_are_new": True,
            "no_v2_document_is_reused_or_selectively_regenerated": True,
            "no_activation_sae_av_ar_heldout_metric_or_endpoint_seen": True,
            "unchanged_downstream_scientific_design": True,
        },
        "files": files,
        "generation_protocol": {
            "seed": 20260801,
            "attempt_seed_formula": (
                "seed + 100 * concept_index + attempt"
            ),
            "max_attempts": 4,
            "attempt_selection": (
                "ascending attempts; retain first mechanically admissible "
                "complete batch immediately; never generate or compare later "
                "attempts after acceptance"
            ),
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "repetition_penalty": 1.0,
            "max_new_tokens": 1800,
            "accepted_word_range_inclusive": [80, 150],
            "target_word_range": [95, 120],
            "max_train_test_word_5gram_jaccard_exclusive": 0.15,
            "documents_per_concept": {"train": 4, "test": 2},
            "concepts": 24,
            "anchors": 144,
        },
        "planned_outputs": planned,
        "freshness": {
            "all_v3_planned_outputs_absent_at_freeze": True,
            "all_v3_text_must_be_generated_after_this_file": True,
            "scenario_anchors_semantically_audited_before_generation": True,
        },
        "embargo": {
            "semantic_reviewers_may_view_both_text_splits": True,
            "downstream_discovery_allowed_only_after_two_review_hashes_and_aggregate_PASS": True,
            "selector_allowed_inputs": [
                "c1_confirmatory_discovery_v3.jsonl"
            ],
            "heldout_prohibited_until_feature_and_candidate_benchmark_freeze": True,
        },
        "model_identity": {
            "model_freeze_sha256": by_role["model_freeze"]["sha256"],
            "base_fingerprint_sha256": model_freeze["models"]["base"][
                "fingerprint_sha256"
            ],
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "operations": {
            "server_must_remain_on": True,
            "no_shutdown_logic": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"C1_V3_STAGE0_FROZEN out={args.out} "
        f"sha256={sha256_file(args.out)}"
    )


if __name__ == "__main__":
    main()
