#!/usr/bin/env python3
"""Generate and freeze the fresh C1-confirmatory v3 corpus.

This program is deliberately blind to activations and to all v2 generated
text.  For each frozen concept it generates a complete six-document batch,
tests only preregistered mechanical admissibility, and immediately retains the
first passing attempt.  The append-only, hash-chained checkpoint is the source
of truth for safe restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any


EXPERIMENT = "C1 confirmatory synthetic concept cohort v3"
SPEC_EXPERIMENT = "C1 confirmatory synthetic concept cohort v2"
MASTER_SEED = 20260801
MAX_ATTEMPTS = 4
WORD_MIN = 80
WORD_MAX = 150
TARGET_WORD_MIN = 95
TARGET_WORD_MAX = 120
MAX_TRAIN_TEST_5GRAM_JACCARD = 0.15
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 64
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 1800
ZERO_HASH = "0" * 64
V3R2_ANCHOR_EXPERIMENT = (
    "C1 confirmatory synthetic concept cohort v3r2 scenario anchors"
)
V3R2_ANCHOR_STATUS = "frozen_before_v3r2_generation"
FAILED_V3_ANCHORS_SHA256 = (
    "061903133b748ccffe2f85f697c5e6a7d53fd631e63fb1c0302ae42fbd59e6d5"
)
FAILED_V3_ANCHOR_AUDIT_SHA256 = (
    "5dcc267ad9f6bf9765bbb7c6ab3965498c275c7df3165780f5f1d79301c2a396"
)
V3R2_CHANGED_ANCHOR_IDS = (
    "automatic_memory_reclamation_test_01",
    "dynamic_memory_allocation_test_01",
    "cryptographic_authentication_test_01",
    "error_detecting_codes_test_01",
    "protein_quality_control_test_00",
    "protein_quality_control_test_01",
    "membrane_vesicle_trafficking_test_00",
    "membrane_vesicle_trafficking_test_01",
    "microbial_quorum_sensing_test_00",
    "microbial_cross_feeding_test_00",
    "fault_rupture_mechanics_train_00",
    "fault_rupture_mechanics_train_01",
    "fault_rupture_mechanics_train_02",
    "fault_rupture_mechanics_train_03",
    "fault_rupture_mechanics_test_00",
    "fault_rupture_mechanics_test_01",
    "slope_failure_mechanics_train_00",
    "slope_failure_mechanics_train_01",
    "slope_failure_mechanics_train_02",
    "slope_failure_mechanics_train_03",
    "slope_failure_mechanics_test_00",
    "slope_failure_mechanics_test_01",
    "groundwater_contaminant_transport_test_01",
    "coastal_saltwater_intrusion_test_01",
    "census_classification_test_01",
    "cadastral_land_taxation_test_01",
    "quarantine_regimes_test_01",
    "lexical_semantic_ambiguity_test_00",
    "phonological_assimilation_test_00",
    "morphological_agreement_test_01",
    "feedback_control_stability_test_01",
    "dynamical_state_estimation_test_01",
    "fatigue_crack_growth_test_01",
)

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(
    r"(?:https?://|www\.|(?:[A-Za-z0-9-]+\.)+"
    r"(?:com|org|net|edu|gov|io|ai)\b)",
    re.IGNORECASE,
)
META_PATTERNS = (
    re.compile(
        r"\b(?:train(?:ing)?|test|validation|held[- ]?out)\s+"
        r"(?:split|set|document|example|prompt|request|item)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:dataset|corpus|benchmark)\b", re.IGNORECASE),
    re.compile(r"\b(?:scenario\s+anchor|anchor\s+text|anchor\s+id)\b", re.I),
    re.compile(r"\b(?:slot\s+id|discourse\s+role)\b", re.IGNORECASE),
    re.compile(r"\b(?:target|negative)\s+concept\b", re.IGNORECASE),
    re.compile(r"\b(?:these|the above|given)\s+instructions?\b", re.I),
    re.compile(r"\b(?:JSON|markdown)\b", re.IGNORECASE),
    re.compile(r"\bSAE(?:\s+feature)?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:activation\s+(?:vector|feature|direction)|"
        r"feature\s+direction)\b",
        re.IGNORECASE,
    ),
)
ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "because",
        "but",
        "can",
        "could",
        "does",
        "for",
        "from",
        "how",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "these",
        "this",
        "to",
        "under",
        "what",
        "when",
        "which",
        "while",
        "why",
        "with",
        "would",
    }
)
DISCOURSE_SLOTS = (
    (
        "train_00_causal_mechanism",
        "train",
        0,
        "causal_mechanism",
        "Present one observed phenomenon and request its causal mechanism.",
    ),
    (
        "train_01_contrastive_diagnosis",
        "train",
        1,
        "contrastive_diagnosis",
        "Contrast two conditions and request an explanation of the difference.",
    ),
    (
        "train_02_design_tradeoff",
        "train",
        2,
        "design_tradeoff",
        "Pose a prospective design decision and request relevant tradeoffs or safeguards.",
    ),
    (
        "train_03_failure_analysis",
        "train",
        3,
        "failure_analysis",
        "Describe a concrete failure pattern and request a mechanism-focused diagnosis.",
    ),
    (
        "test_00_counterfactual_prediction",
        "test",
        0,
        "counterfactual_prediction",
        "Introduce a new setting and request a prediction under one changed condition.",
    ),
    (
        "test_01_evidence_interpretation",
        "test",
        1,
        "evidence_interpretation",
        "Provide a generic observation pattern and request interpretation of the underlying mechanism.",
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def read_json_object(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def normalized_text(text: str) -> str:
    return " ".join(words(text))


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    haystack = f" {normalized_text(text)} "
    needle = normalized_text(phrase)
    return bool(needle) and f" {needle} " in haystack


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {
        tuple(tokens[index : index + n])
        for index in range(max(0, len(tokens) - n + 1))
    }


def ngram_jaccard(left: str, right: str, n: int = 5) -> float:
    left_set = ngrams(left, n)
    right_set = ngrams(right, n)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def unigram_jaccard(left: str, right: str) -> float:
    left_set = set(words(left))
    right_set = set(words(right))
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def expected_slots() -> list[dict[str, Any]]:
    return [
        {
            "slot_id": slot_id,
            "split": split,
            "ordinal": ordinal,
            "role": role,
            "instruction": instruction,
        }
        for slot_id, split, ordinal, role, instruction in DISCOURSE_SLOTS
    ]


def validate_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.get("schema_version") != 2:
        raise ValueError("concept spec schema_version must be 2")
    if spec.get("experiment") != SPEC_EXPERIMENT:
        raise ValueError("unexpected concept spec experiment")
    if spec.get("language") != "en":
        raise ValueError("concept spec must be English-only")
    if spec.get("documents_per_concept") != {"train": 4, "test": 2}:
        raise ValueError("concept spec must freeze four train and two test")
    concepts = spec.get("concepts")
    if not isinstance(concepts, list) or len(concepts) != 24:
        raise ValueError("concept spec must contain exactly 24 concepts")
    required = {
        "id",
        "superdomain",
        "title",
        "summary",
        "hard_negative_id",
    }
    for index, concept in enumerate(concepts):
        if not isinstance(concept, dict) or set(concept) != required:
            raise ValueError(f"invalid concept schema at index {index}")
        for key in required:
            if not isinstance(concept[key], str) or not concept[key].strip():
                raise ValueError(f"empty concept field {index}:{key}")
    by_id = {concept["id"]: concept for concept in concepts}
    if len(by_id) != len(concepts):
        raise ValueError("concept IDs must be unique")
    if len({concept["title"] for concept in concepts}) != len(concepts):
        raise ValueError("concept titles must be unique")
    if len({concept["superdomain"] for concept in concepts}) != 6:
        raise ValueError("concept spec must contain six superdomains")
    domain_counts: dict[str, int] = {}
    for concept in concepts:
        domain_counts[concept["superdomain"]] = (
            domain_counts.get(concept["superdomain"], 0) + 1
        )
        negative = by_id.get(concept["hard_negative_id"])
        if (
            negative is None
            or negative["hard_negative_id"] != concept["id"]
            or negative["superdomain"] != concept["superdomain"]
            or negative["id"] == concept["id"]
        ):
            raise ValueError(
                "hard negative is not reciprocal and within-superdomain: "
                f"{concept['id']}"
            )
    if set(domain_counts.values()) != {4}:
        raise ValueError("each superdomain must contain four concepts")
    return concepts


def validate_anchors(
    anchors: dict[str, Any],
    concepts: list[dict[str, Any]],
    spec_sha256: str,
    base_rubric_sha256: str,
) -> dict[str, dict[str, Any]]:
    expected_top = {
        "schema_version",
        "experiment",
        "status",
        "purpose",
        "sources",
        "generation_contract",
        "discourse_slots",
        "concepts",
        "validation_attestation",
        "revision",
    }
    if set(anchors) != expected_top:
        raise ValueError("scenario-anchor top-level schema changed")
    if (
        anchors["schema_version"] != 1
        or anchors["experiment"] != V3R2_ANCHOR_EXPERIMENT
        or anchors["status"] != V3R2_ANCHOR_STATUS
        or anchors["purpose"]
        != (
            "Freeze the final pre-text design iteration after the "
            "conservative v3 scenario-anchor audit failed, while leaving "
            "every final prose request ungenerated."
        )
    ):
        raise ValueError("scenario-anchor identity/status mismatch")
    sources = anchors["sources"]
    expected_source_ids = {
        "review_A",
        "review_B",
        "manual_audit_rubric",
        "concept_spec_v2",
        "scenario_anchors_v3",
        "scenario_anchor_audit_v3",
    }
    if (
        not isinstance(sources, dict)
        or set(sources) != expected_source_ids
    ):
        raise ValueError("scenario-anchor sources changed")
    for source_id, source in sources.items():
        expected_keys = (
            {"path", "sha256", "status"}
            if source_id == "scenario_anchor_audit_v3"
            else {"path", "sha256"}
        )
        if (
            not isinstance(source, dict)
            or set(source) != expected_keys
            or not isinstance(source["path"], str)
            or not SHA256_RE.fullmatch(str(source["sha256"]))
        ):
            raise ValueError(f"invalid anchor source: {source_id}")
    expected_source_paths = {
        "review_A": "results/c1_confirmatory_manual_review_A_v1.json",
        "review_B": "results/c1_confirmatory_manual_review_B_v1.json",
        "manual_audit_rubric": (
            "server/c1_confirmatory_manual_audit_rubric_v1.json"
        ),
        "concept_spec_v2": "server/c1_confirmatory_concepts_v2.json",
        "scenario_anchors_v3": (
            "server/c1_confirmatory_scenario_anchors_v3.json"
        ),
        "scenario_anchor_audit_v3": (
            "results/c1_confirmatory_scenario_anchor_audit_v3.json"
        ),
    }
    expected_source_hashes = {
        "review_A": (
            "06ca5fa5eec08e6ed10f73e5905bbb3bb1e23dfb441a2c6e1b2ce5c19c84a8d3"
        ),
        "review_B": (
            "78a3708341a5efdc5d620ec1bdc2f9207ebaed3488e7d2f1b6f7b95b5818619f"
        ),
        "manual_audit_rubric": base_rubric_sha256,
        "concept_spec_v2": spec_sha256,
        "scenario_anchors_v3": FAILED_V3_ANCHORS_SHA256,
        "scenario_anchor_audit_v3": FAILED_V3_ANCHOR_AUDIT_SHA256,
    }
    for source_id, expected_path in expected_source_paths.items():
        if (
            sources[source_id]["path"] != expected_path
            or sources[source_id]["sha256"]
            != expected_source_hashes[source_id]
        ):
            raise ValueError(f"anchor source binding changed: {source_id}")
    if sources["concept_spec_v2"]["sha256"] != spec_sha256:
        raise ValueError("scenario anchors do not bind the supplied spec")
    if sources["manual_audit_rubric"]["sha256"] != base_rubric_sha256:
        raise ValueError("scenario anchors do not bind the supplied rubric")
    if (
        sources["scenario_anchors_v3"]["sha256"]
        != FAILED_V3_ANCHORS_SHA256
        or sources["scenario_anchor_audit_v3"]["sha256"]
        != FAILED_V3_ANCHOR_AUDIT_SHA256
        or sources["scenario_anchor_audit_v3"]["status"] != "FAIL"
    ):
        raise ValueError(
            "v3r2 anchors do not bind the failed v3 draft and audit"
        )

    contract = anchors["generation_contract"]
    expected_contract = {
        "anchors_are_not_final_text": True,
        "generate_one_prose_user_request_per_anchor": True,
        "preserve_anchor_scenario_and_intent": True,
        "do_not_copy_anchor_as_a_list_or_heading": True,
        "do_not_add_named_people_organizations_products_specific_places_urls_or_four_digit_years": True,
        "do_not_make_the_reciprocal_hard_negative_a_required_or_co_equal_target": True,
        "do_not_reuse_a_scenario_within_a_concept": True,
        "do_not_turn_heldout_anchors_into_continuations_or_paraphrases_of_train_anchors": True,
        "pair_style_rule": (
            "For every reciprocal pair, both concepts use the same ordered "
            "discourse slots, split assignments, and matched framing "
            "specificity; content remains concept-specific."
        ),
        "pre_text_design_iteration": True,
        "iteration_label": "v3r2_final_pre_text_design_iteration",
        "stop_if_v3r2_anchor_audit_fails": True,
    }
    if canonical_json(contract) != canonical_json(expected_contract):
        raise ValueError("scenario-anchor generation contract changed")
    if anchors["discourse_slots"] != expected_slots():
        raise ValueError("frozen discourse-slot schedule changed")

    anchor_concepts = anchors["concepts"]
    if not isinstance(anchor_concepts, list) or len(anchor_concepts) != 24:
        raise ValueError("scenario anchors must contain 24 concepts")
    by_id: dict[str, dict[str, Any]] = {}
    anchor_ids: set[str] = set()
    scenario_texts: set[str] = set()
    for index, (concept, row) in enumerate(zip(concepts, anchor_concepts)):
        if not isinstance(row, dict) or set(row) != {
            "id",
            "hard_negative_id",
            "train",
            "test",
        }:
            raise ValueError(f"invalid anchor concept schema at {index}")
        if (
            row["id"] != concept["id"]
            or row["hard_negative_id"] != concept["hard_negative_id"]
        ):
            raise ValueError(f"anchor/spec ordering mismatch at {index}")
        by_id[row["id"]] = row
        for split, expected_count in (("train", 4), ("test", 2)):
            items = row[split]
            if not isinstance(items, list) or len(items) != expected_count:
                raise ValueError(f"invalid anchor count for {row['id']}:{split}")
            for ordinal, item in enumerate(items):
                slot = expected_slots()[ordinal if split == "train" else 4 + ordinal]
                if not isinstance(item, dict) or set(item) != {
                    "anchor_id",
                    "slot_id",
                    "role",
                    "scenario_anchor",
                }:
                    raise ValueError(
                        f"invalid anchor item schema {row['id']}:{split}:{ordinal}"
                    )
                expected_anchor_id = f"{row['id']}_{split}_{ordinal:02d}"
                if (
                    item["anchor_id"] != expected_anchor_id
                    or item["slot_id"] != slot["slot_id"]
                    or item["role"] != slot["role"]
                ):
                    raise ValueError(
                        f"anchor slot mismatch {row['id']}:{split}:{ordinal}"
                    )
                scenario = item["scenario_anchor"]
                if (
                    not isinstance(scenario, str)
                    or not scenario.strip()
                    or scenario != scenario.strip()
                    or "\n" in scenario
                ):
                    raise ValueError(f"invalid scenario text: {expected_anchor_id}")
                if item["anchor_id"] in anchor_ids:
                    raise ValueError(f"duplicate anchor ID: {item['anchor_id']}")
                normalized = normalized_text(scenario)
                if normalized in scenario_texts:
                    raise ValueError(f"duplicate scenario: {item['anchor_id']}")
                anchor_ids.add(item["anchor_id"])
                scenario_texts.add(normalized)
    if len(by_id) != 24 or len(anchor_ids) != 144:
        raise ValueError("scenario-anchor coverage is not 24 x 6")
    attestation = anchors["validation_attestation"]
    expected_attestation = {
        "concepts": 24,
        "reciprocal_pairs": 12,
        "anchors": 144,
        "train_anchors": 96,
        "test_anchors": 48,
        "exactly_four_train_and_two_test_per_concept": True,
        "anchor_ids_unique": True,
        "scenario_texts_unique": True,
        "all_concept_ids_match_v2_spec": True,
        "all_hard_negative_mappings_match_v2_spec_and_are_reciprocal": True,
        "paired_slot_distributions_identical": True,
        "all_six_scenarios_substantively_distinct_within_each_concept": True,
        "heldout_scenarios_do_not_duplicate_or_continue_train_scenarios": True,
        "no_named_person_organization_product_specific_place_url_or_four_digit_year": True,
        "no_anchor_requires_the_reciprocal_hard_negative_or_both_concepts_as_co_equal_targets": True,
        "review_A_failure_frameworks_avoided": True,
        "review_B_failure_frameworks_avoided": True,
        "final_prompt_text_generated": False,
        "v3_anchor_audit_status": "FAIL",
        "v3r2_changed_anchor_count": 33,
        "v3r2_unchanged_anchor_count": 111,
        "v3r2_mechanical_self_check": "PASS",
        "v3r2_per_concept_semantic_self_check": "PASS",
        "final_pre_text_design_iteration": True,
    }
    if canonical_json(attestation) != canonical_json(expected_attestation):
        raise ValueError("scenario-anchor validation attestation changed")
    expected_revision = {
        "basis": "Conservative v3 scenario-anchor audit FAIL",
        "pre_text_design_iteration": True,
        "scope": (
            "Only audit-implicated heldout anchors were replaced for "
            "within-concept overlap, except that all twelve "
            "fault-versus-slope anchors named by the pair-level style "
            "failure were reframed symmetrically."
        ),
        "changed_anchor_count": 33,
        "unchanged_anchor_count": 111,
        "changed_anchor_ids": list(V3R2_CHANGED_ANCHOR_IDS),
        "stop_rule": (
            "If an independent conservative audit of v3r2 fails, stop "
            "scenario-anchor iteration instead of generating another "
            "content revision."
        ),
    }
    if canonical_json(anchors["revision"]) != canonical_json(
        expected_revision
    ):
        raise ValueError("scenario-anchor v3r2 revision record changed")
    if (
        len(set(V3R2_CHANGED_ANCHOR_IDS)) != 33
        or not set(V3R2_CHANGED_ANCHOR_IDS).issubset(anchor_ids)
    ):
        raise ValueError("v3r2 changed-anchor IDs do not match the cohort")
    return by_id


def validate_addendum(
    addendum: dict[str, Any],
    addendum_path: Path,
    base_rubric_path: Path,
    anchors_path: Path,
    anchors: dict[str, Any],
) -> None:
    expected_top = {
        "schema_version",
        "experiment",
        "status",
        "purpose",
        "inherits",
        "scenario_anchors",
        "review_blinding",
        "inherited_document_check_order",
        "additional_document_checks",
        "inherited_concept_batch_check_order",
        "additional_concept_batch_checks",
        "reciprocal_pair_checks",
        "decision_rule",
        "independence_protocol",
        "required_output",
        "revision",
    }
    if (
        set(addendum) != expected_top
        or addendum.get("schema_version") != 1
        or addendum.get("experiment") != EXPERIMENT
        or addendum.get("status") != V3R2_ANCHOR_STATUS
        or addendum.get("purpose")
        != (
            "Apply the already frozen v3 semantic checks to the final "
            "v3r2 scenario-anchor asset before any generated v3 text exists."
        )
    ):
        raise ValueError("v3 audit addendum identity/status mismatch")
    inherits = addendum.get("inherits", {})
    if (
        set(inherits) != {
            "path",
            "sha256",
            "all_inherited_checks_remain_mandatory",
        }
        or inherits.get("path")
        != "server/c1_confirmatory_manual_audit_rubric_v1.json"
        or inherits.get("sha256") != sha256_file(base_rubric_path)
        or inherits.get("all_inherited_checks_remain_mandatory") is not True
    ):
        raise ValueError("audit addendum does not bind the base rubric")
    anchor_binding = addendum.get("scenario_anchors", {})
    if (
        set(anchor_binding) != {"path", "sha256", "status"}
        or anchor_binding.get("path")
        != "server/c1_confirmatory_scenario_anchors_v3r2.json"
        or anchor_binding.get("sha256") != sha256_file(anchors_path)
        or anchor_binding.get("status") != anchors["status"]
    ):
        raise ValueError("audit addendum does not bind the scenario anchors")
    expected_revision = {
        "type": "final pre-text anchor-design iteration",
        "failed_anchor_draft_sha256": FAILED_V3_ANCHORS_SHA256,
        "failed_anchor_audit_sha256": FAILED_V3_ANCHOR_AUDIT_SHA256,
        "generated_v3_text_existed_before_revision": False,
        "further_anchor_iteration_if_v3r2_fails": False,
        "all_document_batch_and_pair_checks_unchanged": True,
    }
    if canonical_json(addendum["revision"]) != canonical_json(
        expected_revision
    ):
        raise ValueError("v3r2 audit-addendum revision record changed")
    if (
        addendum["revision"]["failed_anchor_draft_sha256"]
        != anchors["sources"]["scenario_anchors_v3"]["sha256"]
        or addendum["revision"]["failed_anchor_audit_sha256"]
        != anchors["sources"]["scenario_anchor_audit_v3"]["sha256"]
    ):
        raise ValueError(
            "v3r2 addendum and anchors disagree on failed-design provenance"
        )
    expected_document_checks = [
        "scenario_anchor_adherence",
        "no_anchor_copy_or_meta_discussion",
        "no_exact_target_or_hard_negative_title",
        "single_english_prose_question",
    ]
    expected_batch_checks = [
        "discourse_schedule_adherence",
        "anchor_set_coverage",
    ]
    expected_pair_checks = [
        "reciprocal_mapping_integrity",
        "pair_hard_negative_separation",
        "reciprocal_pair_style_balance",
    ]
    for field, expected in (
        ("additional_document_checks", expected_document_checks),
        ("additional_concept_batch_checks", expected_batch_checks),
        ("reciprocal_pair_checks", expected_pair_checks),
    ):
        rows = addendum.get(field)
        if (
            not isinstance(rows, list)
            or [row.get("id") for row in rows] != expected
            or any(not isinstance(row.get("pass_rule"), str) for row in rows)
        ):
            raise ValueError(f"audit addendum check order changed: {field}")
    required = addendum.get("required_output", {})
    if required.get("coverage") != {
        "documents": 144,
        "document_checks": 1296,
        "concept_batches": 24,
        "concept_batch_checks": 120,
        "reciprocal_pairs": 12,
        "pair_checks": 36,
    }:
        raise ValueError("audit addendum coverage changed")
    if not addendum_path.is_file():
        raise FileNotFoundError(addendum_path)


def verify_frozen_role(
    stage0: dict[str, Any], role: str, path: Path
) -> None:
    files = stage0.get("files")
    if not isinstance(files, list):
        raise ValueError("stage0 files must be an array")
    matches = [row for row in files if row.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"stage0 must contain exactly one role={role}")
    row = matches[0]
    actual_hash = sha256_file(path)
    actual_bytes = path.stat().st_size
    if row.get("sha256") != actual_hash:
        raise ValueError(
            f"stage0 hash mismatch role={role}: "
            f"expected={row.get('sha256')} actual={actual_hash}"
        )
    if row.get("bytes") != actual_bytes:
        raise ValueError(f"stage0 byte-count mismatch role={role}")


def validate_stage0(
    stage0: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if (
        stage0.get("schema_version") != 3
        or stage0.get("experiment") != EXPERIMENT
        or stage0.get("stage") != "pre_generation"
        or "frozen" not in str(stage0.get("status", ""))
    ):
        raise ValueError("invalid C1 v3 stage0 identity/status")
    for role, path in (
        ("concept_spec", args.spec),
        ("scenario_anchors", args.anchors),
        ("preregistration_amendment", args.preregistration),
        ("audit_addendum", args.rubric),
        ("corpus_generator", Path(__file__).resolve()),
    ):
        verify_frozen_role(stage0, role, path)
    protocol = stage0.get("generation_protocol", {})
    expected_protocol = {
        "seed": MASTER_SEED,
        "attempt_seed_formula": "seed + 100 * concept_index + attempt",
        "max_attempts": MAX_ATTEMPTS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "accepted_word_range_inclusive": [WORD_MIN, WORD_MAX],
        "target_word_range": [TARGET_WORD_MIN, TARGET_WORD_MAX],
        "max_train_test_word_5gram_jaccard_exclusive": (
            MAX_TRAIN_TEST_5GRAM_JACCARD
        ),
        "documents_per_concept": {"train": 4, "test": 2},
        "concepts": 24,
        "anchors": 144,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"stage0 generation protocol mismatch: {key}")
    selection = str(protocol.get("attempt_selection", ""))
    if (
        "first mechanically admissible" not in selection
        or "never generate or compare later attempts" not in selection
    ):
        raise ValueError("stage0 does not freeze first-pass retention")
    planned_by_role = {
        row.get("role"): row
        for row in stage0.get("planned_outputs", [])
        if isinstance(row, dict)
    }
    planned = {
        "append_only_checkpoint": args.checkpoint,
        "corpus_report": args.out_report,
        "combined_manifest": args.out_manifest,
        "discovery_manifest": args.out_discovery_manifest,
        "heldout_manifest": args.out_heldout_manifest,
    }
    for role, path in planned.items():
        row = planned_by_role.get(role)
        if (
            row is None
            or os.path.normpath(str(row.get("path")))
            != os.path.normpath(str(path))
            or row.get("exists_at_freeze") is not False
        ):
            raise ValueError(f"stage0 planned output mismatch: {role}")
    freshness = stage0.get("freshness", {})
    if (
        freshness.get("all_v3_planned_outputs_absent_at_freeze") is not True
        or freshness.get("all_v3_text_must_be_generated_after_this_file")
        is not True
    ):
        raise ValueError("stage0 freshness attestation was weakened")
    operations = stage0.get("operations", {})
    if (
        operations.get("server_must_remain_on") is not True
        or operations.get("no_shutdown_logic") is not True
    ):
        raise ValueError("stage0 server-safety contract changed")


def anchor_batch_sha256(anchor_row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(anchor_row).encode("utf-8"))


def generation_prompt(
    concept: dict[str, Any],
    hard_negative: dict[str, Any],
    anchor_row: dict[str, Any],
) -> str:
    slot_lines: list[str] = []
    for split in ("train", "test"):
        for ordinal, anchor in enumerate(anchor_row[split]):
            slot = expected_slots()[ordinal if split == "train" else 4 + ordinal]
            slot_lines.append(
                f'- Output {split}[{ordinal}] | role "{slot["role"]}" | '
                f'common role instruction: {slot["instruction"]} | '
                f'scenario constraint: {anchor["scenario_anchor"]}'
            )
    schedule = "\n".join(slot_lines)
    return f"""Create one complete batch of six distinct English user requests.

TARGET CONCEPT (required explanatory scope)
Title, forbidden verbatim in output: {concept["title"]}
Scope: {concept["summary"]}

RECIPROCAL HARD NEGATIVE (explicit exclusion)
Title, forbidden verbatim in output: {hard_negative["title"]}
Scope to exclude as a required or co-equal goal: {hard_negative["summary"]}
The target alone must be sufficient for each requested explanation. Contextual adjacency is allowed only when it does not make the excluded scope necessary.

FROZEN OUTPUT SCHEDULE
{schedule}

Return exactly one JSON object with exactly two keys. "train" must be an array of four strings in the listed order, and "test" must be an array of two strings in the listed order. Every string must:
- realize its assigned scenario and discourse role once, while paraphrasing rather than copying the scenario constraint;
- be one self-contained English prose paragraph functioning as one user question, with exactly one question mark at the end;
- contain 80 to 150 words inclusive and aim for 95 to 120 words;
- omit both exact forbidden titles, the word Earth, URLs, four-digit years, named people, organizations, products, and specific places;
- contain no code, list, formula, table, heading, answer, dataset/split/anchor/slot/role language, or discussion of these instructions.

Use substantially different wording and framing for all six requests. Held-out scenarios must not duplicate, continue, or closely paraphrase any training scenario. Do not number strings and do not add markdown, code fences, or commentary outside the JSON object."""


def extract_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text != raw:
        raise ValueError("whitespace outside JSON object")
    value = strict_json_loads(text)
    if not isinstance(value, dict):
        raise ValueError("top-level generation output is not an object")
    return value


def _document_format_errors(text: str, index: int) -> list[str]:
    errors: list[str] = []
    if text != text.strip():
        errors.append(f"outer_whitespace_{index}")
    if "\n" in text or "\r" in text:
        errors.append(f"multiple_paragraphs_{index}")
    if text.count("?") != 1 or not text.endswith("?"):
        errors.append(f"not_one_terminal_question_{index}")
    if URL_RE.search(text):
        errors.append(f"url_{index}")
    if re.search(r"(?<!\d)\d{4}(?!\d)", text):
        errors.append(f"four_digit_year_{index}")
    if re.search(r"\bEarth\b", text, re.IGNORECASE):
        errors.append(f"earth_literal_{index}")
    if any(pattern.search(text) for pattern in META_PATTERNS):
        errors.append(f"meta_language_{index}")
    if (
        "```" in text
        or "`" in text
        or re.search(r"(?m)^\s*(?:[-*+#]|\d+[.)])\s+", text)
        or re.search(r"(?m)^\s*\|.*\|\s*$", text)
        or re.search(r"\$\$|\\begin\{|\\end\{|\\\[|\\\]", text)
        or re.search(r"[{}[\]|]", text)
        or re.search(r"(?:^|\s)(?:def|class|function|SELECT)\s+", text)
        or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=[^=]", text)
    ):
        errors.append(f"code_list_formula_table_{index}")
    if re.match(
        r"^\s*(?:question|heading|scenario|request|prompt)\s*:",
        text,
        re.IGNORECASE,
    ):
        errors.append(f"heading_{index}")
    if re.search(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b", text):
        errors.append(f"possible_named_entity_{index}")
    if re.search(
        r"\b[A-Z][A-Za-z&.-]*\s+"
        r"(?:University|Institute|Corporation|Company|Agency|Laboratory)\b",
        text,
    ):
        errors.append(f"possible_named_organization_{index}")
    non_ascii_letters = [
        char for char in text if char.isalpha() and not char.isascii()
    ]
    if non_ascii_letters:
        errors.append(f"non_ascii_alphabetic_{index}")
    token_set = set(words(text))
    if len(token_set & ENGLISH_FUNCTION_WORDS) < 4:
        errors.append(f"english_function_word_check_{index}")
    return errors


def validate_batch(
    value: dict[str, Any],
    concept: dict[str, Any],
    hard_negative: dict[str, Any],
    anchor_row: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if set(value) != {"train", "test"}:
        errors.append(f"top_level_keys={sorted(value)}")
    train = value.get("train")
    test = value.get("test")
    if not isinstance(train, list) or len(train) != 4:
        errors.append("train_must_have_exactly_4")
    if not isinstance(test, list) or len(test) != 2:
        errors.append("test_must_have_exactly_2")
    structurally_complete = (
        isinstance(train, list)
        and len(train) == 4
        and isinstance(test, list)
        and len(test) == 2
    )
    documents: list[Any] = list(train or []) + list(test or [])
    if any(not isinstance(item, str) for item in documents):
        errors.append("all_documents_must_be_strings")
    if not structurally_complete or any(
        not isinstance(item, str) for item in documents
    ):
        return errors, {
            "structurally_complete": False,
            "word_counts": [],
            "target_word_range_hits": [],
            "pairwise_similarity": [],
        }

    text_documents: list[str] = documents
    word_counts = [len(words(text)) for text in text_documents]
    target_hits = [
        TARGET_WORD_MIN <= count <= TARGET_WORD_MAX for count in word_counts
    ]
    normalized_documents = [normalized_text(text) for text in text_documents]
    for index, (text, count) in enumerate(zip(text_documents, word_counts)):
        if not WORD_MIN <= count <= WORD_MAX:
            errors.append(f"word_count_{index}={count}")
        errors.extend(_document_format_errors(text, index))
        if contains_normalized_phrase(text, concept["title"]):
            errors.append(f"exact_target_title_{index}")
        if contains_normalized_phrase(text, hard_negative["title"]):
            errors.append(f"exact_hard_negative_title_{index}")
        for literal_name, literal in (
            ("concept_id", concept["id"]),
            ("hard_negative_id", hard_negative["id"]),
            ("superdomain_id", concept["superdomain"]),
        ):
            if literal.lower() in text.lower():
                errors.append(f"{literal_name}_{index}")
        split = "train" if index < 4 else "test"
        ordinal = index if index < 4 else index - 4
        anchor = anchor_row[split][ordinal]["scenario_anchor"]
        anchor_normalized = normalized_text(anchor)
        if (
            anchor_normalized
            and anchor_normalized in normalized_documents[index]
        ):
            errors.append(f"verbatim_anchor_copy_{index}")
        if ngram_jaccard(text, anchor, 5) >= 0.70:
            errors.append(f"near_anchor_copy_{index}")
    if len(set(normalized_documents)) != 6:
        errors.append("duplicate_documents")

    similarities = []
    for left, right in combinations(range(6), 2):
        row = {
            "left": left,
            "right": right,
            "unigram_jaccard": unigram_jaccard(
                text_documents[left], text_documents[right]
            ),
            "word_5gram_jaccard": ngram_jaccard(
                text_documents[left], text_documents[right], 5
            ),
        }
        similarities.append(row)
        if (
            (left < 4) != (right < 4)
            and row["word_5gram_jaccard"]
            >= MAX_TRAIN_TEST_5GRAM_JACCARD
        ):
            errors.append(
                f"train_test_5gram_jaccard_{left}_{right}="
                f"{row['word_5gram_jaccard']:.6f}"
            )
    anchor_similarities = []
    for index, text in enumerate(text_documents):
        split = "train" if index < 4 else "test"
        ordinal = index if index < 4 else index - 4
        anchor = anchor_row[split][ordinal]
        anchor_similarities.append(
            {
                "document_index": index,
                "anchor_id": anchor["anchor_id"],
                "word_5gram_jaccard": ngram_jaccard(
                    text, anchor["scenario_anchor"], 5
                ),
            }
        )
    train_test = [
        row
        for row in similarities
        if (row["left"] < 4) != (row["right"] < 4)
    ]
    diagnostics = {
        "structurally_complete": True,
        "word_counts": word_counts,
        "target_word_range_hits": target_hits,
        "target_word_range_hit_count": sum(target_hits),
        "max_pairwise_unigram_jaccard": max(
            row["unigram_jaccard"] for row in similarities
        ),
        "max_train_test_word_5gram_jaccard": max(
            row["word_5gram_jaccard"] for row in train_test
        ),
        "pairwise_similarity": similarities,
        "anchor_similarity": anchor_similarities,
    }
    return errors, diagnostics


def assess_raw(
    raw: str,
    concept: dict[str, Any],
    hard_negative: dict[str, Any],
    anchor_row: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    try:
        parsed = extract_object(raw)
        errors, diagnostics = validate_batch(
            parsed, concept, hard_negative, anchor_row
        )
        return parsed, errors, diagnostics
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        return None, [f"parse_error={type(exc).__name__}:{exc}"], {}


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(row) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(
            "checkpoint has an incomplete final record; preserve it and "
            "recover from the last verified newline before rerunning"
        )
    rows: list[dict[str, Any]] = []
    for line_number, encoded in enumerate(raw.splitlines(), start=1):
        if not encoded:
            raise ValueError(f"blank checkpoint record at line {line_number}")
        try:
            text = encoded.decode("utf-8")
            value = strict_json_loads(text)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid checkpoint record at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"checkpoint record {line_number} is not an object"
            )
        if canonical_json(value) != text:
            raise ValueError(
                f"checkpoint record {line_number} is not canonical JSON"
            )
        rows.append(value)
    return rows


def verify_checkpoint(
    rows: list[dict[str, Any]],
    binding_sha256: str,
    concepts: list[dict[str, Any]],
    concept_by_id: dict[str, dict[str, Any]],
    anchor_by_id: dict[str, dict[str, Any]],
    prompts: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, int], str]:
    expected_keys = {
        "schema_version",
        "experiment",
        "binding_sha256",
        "checkpoint_index",
        "previous_row_sha256",
        "row_sha256",
        "concept_id",
        "concept_index",
        "attempt",
        "seed",
        "prompt_sha256",
        "anchor_batch_sha256",
        "raw_sha256",
        "seconds",
        "admissible",
        "errors",
        "diagnostics",
        "parsed",
        "raw",
    }
    accepted: dict[str, dict[str, Any]] = {}
    next_attempt: dict[str, int] = {}
    previous_hash = ZERO_HASH
    concept_cursor = 0
    attempt_cursor = 0
    for checkpoint_index, row in enumerate(rows):
        if set(row) != expected_keys:
            raise ValueError(
                f"checkpoint schema mismatch at row {checkpoint_index}"
            )
        if (
            row["schema_version"] != 2
            or row["experiment"] != EXPERIMENT
            or row["binding_sha256"] != binding_sha256
            or row["checkpoint_index"] != checkpoint_index
            or row["previous_row_sha256"] != previous_hash
        ):
            raise ValueError(
                f"checkpoint chain metadata mismatch at row {checkpoint_index}"
            )
        without_hash = {
            key: value for key, value in row.items() if key != "row_sha256"
        }
        calculated_hash = sha256_bytes(
            canonical_json(without_hash).encode("utf-8")
        )
        if row["row_sha256"] != calculated_hash:
            raise ValueError(
                f"checkpoint row hash mismatch at row {checkpoint_index}"
            )
        previous_hash = calculated_hash
        if concept_cursor >= len(concepts):
            raise ValueError("checkpoint contains rows after all acceptances")
        concept = concepts[concept_cursor]
        concept_id = concept["id"]
        attempt = row["attempt"]
        expected_seed = MASTER_SEED + 100 * concept_cursor + attempt_cursor
        if (
            type(row["checkpoint_index"]) is not int
            or type(row["concept_index"]) is not int
            or type(attempt) is not int
            or row["concept_id"] != concept_id
            or row["concept_index"] != concept_cursor
            or attempt != attempt_cursor
            or attempt not in range(MAX_ATTEMPTS)
            or row["seed"] != expected_seed
            or row["prompt_sha256"]
            != sha256_bytes(prompts[concept_id].encode("utf-8"))
            or row["anchor_batch_sha256"]
            != anchor_batch_sha256(anchor_by_id[concept_id])
        ):
            raise ValueError(
                "checkpoint deterministic ordering mismatch at "
                f"row {checkpoint_index}"
            )
        seconds = row["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or seconds < 0
        ):
            raise ValueError(f"invalid checkpoint duration at {checkpoint_index}")
        raw = row["raw"]
        if (
            not isinstance(raw, str)
            or row["raw_sha256"]
            != sha256_bytes(raw.encode("utf-8"))
        ):
            raise ValueError(f"checkpoint raw hash mismatch at {checkpoint_index}")
        negative = concept_by_id[concept["hard_negative_id"]]
        parsed, errors, diagnostics = assess_raw(
            raw, concept, negative, anchor_by_id[concept_id]
        )
        if (
            canonical_json(row["parsed"]) != canonical_json(parsed)
            or row["errors"] != errors
            or canonical_json(row["diagnostics"])
            != canonical_json(diagnostics)
            or row["admissible"] is not (not errors)
        ):
            raise ValueError(
                f"checkpoint revalidation mismatch at row {checkpoint_index}"
            )
        if row["admissible"]:
            accepted[concept_id] = row
            next_attempt[concept_id] = attempt_cursor + 1
            concept_cursor += 1
            attempt_cursor = 0
        else:
            attempt_cursor += 1
            next_attempt[concept_id] = attempt_cursor
            if attempt_cursor == MAX_ATTEMPTS and checkpoint_index != len(rows) - 1:
                raise ValueError(
                    "checkpoint continues after an exhausted concept"
                )
    if concept_cursor < len(concepts):
        next_attempt.setdefault(concepts[concept_cursor]["id"], attempt_cursor)
    return accepted, next_attempt, previous_hash


def set_seed(seed: int, torch_module: Any, numpy_module: Any) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed % (2**32))
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def write_create_or_identical(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise FileExistsError(
                f"refusing to overwrite conflicting frozen output: {path}"
            )
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(canonical_json(row) + "\n" for row in rows)
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument(
        "--rubric",
        required=True,
        type=Path,
        help="Frozen v3 manual-audit addendum (not the base rubric).",
    )
    parser.add_argument("--base-rubric", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--stage0-freeze", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--out-discovery-manifest", required=True, type=Path)
    parser.add_argument("--out-heldout-manifest", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    args = parser.parse_args()

    required_inputs = (
        args.base_model,
        args.spec,
        args.anchors,
        args.rubric,
        args.base_rubric,
        args.preregistration,
        args.stage0_freeze,
    )
    for path in required_inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.base_model.is_dir():
        raise ValueError("--base-model must be a local model directory")
    output_paths = (
        args.checkpoint,
        args.out_manifest,
        args.out_discovery_manifest,
        args.out_heldout_manifest,
        args.out_report,
    )
    if len({str(path.resolve()) for path in output_paths}) != len(output_paths):
        raise ValueError("checkpoint and output paths must be distinct")

    spec = read_json_object(args.spec)
    anchors = read_json_object(args.anchors)
    addendum = read_json_object(args.rubric)
    read_json_object(args.base_rubric)
    stage0 = read_json_object(args.stage0_freeze)
    concepts = validate_spec(spec)
    concept_by_id = {concept["id"]: concept for concept in concepts}
    anchor_by_id = validate_anchors(
        anchors,
        concepts,
        sha256_file(args.spec),
        sha256_file(args.base_rubric),
    )
    validate_addendum(
        addendum,
        args.rubric,
        args.base_rubric,
        args.anchors,
        anchors,
    )
    validate_stage0(stage0, args)

    prompts: dict[str, str] = {}
    for concept in concepts:
        prompts[concept["id"]] = generation_prompt(
            concept,
            concept_by_id[concept["hard_negative_id"]],
            anchor_by_id[concept["id"]],
        )
    generator_path = Path(__file__).resolve()
    input_hashes = {
        "concept_spec_sha256": sha256_file(args.spec),
        "scenario_anchors_sha256": sha256_file(args.anchors),
        "preregistration_amendment_sha256": sha256_file(args.preregistration),
        "audit_addendum_sha256": sha256_file(args.rubric),
        "base_audit_rubric_sha256": sha256_file(args.base_rubric),
        "stage0_freeze_sha256": sha256_file(args.stage0_freeze),
        "corpus_generator_sha256": sha256_file(generator_path),
    }
    binding = {
        "experiment": EXPERIMENT,
        "inputs": input_hashes,
        "protocol": {
            "master_seed": MASTER_SEED,
            "max_attempts": MAX_ATTEMPTS,
            "attempt_seed": "master + 100 * concept_index + attempt",
            "retention": (
                "retain first mechanically admissible complete batch "
                "immediately; never generate or compare later attempts"
            ),
            "word_range_inclusive": [WORD_MIN, WORD_MAX],
            "target_word_range": [TARGET_WORD_MIN, TARGET_WORD_MAX],
            "max_train_test_word_5gram_jaccard_exclusive": (
                MAX_TRAIN_TEST_5GRAM_JACCARD
            ),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    }
    binding_sha256 = sha256_bytes(canonical_json(binding).encode("utf-8"))
    checkpoint_rows = load_checkpoint(args.checkpoint)
    accepted, next_attempt, previous_hash = verify_checkpoint(
        checkpoint_rows,
        binding_sha256,
        concepts,
        concept_by_id,
        anchor_by_id,
        prompts,
    )
    if len(accepted) < len(concepts):
        current = concepts[len(accepted)]
        if next_attempt.get(current["id"], 0) >= MAX_ATTEMPTS:
            raise RuntimeError(
                f"{current['id']} exhausted all {MAX_ATTEMPTS} fixed "
                "attempts without a mechanically admissible batch; stopping"
            )
    print(
        f"[plan] concepts=24 accepted={len(accepted)} "
        f"pending={24 - len(accepted)} checkpoint_rows={len(checkpoint_rows)}",
        flush=True,
    )

    if len(accepted) < len(concepts):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for v3 corpus generation")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        device = next(model.parameters()).device
        for concept_index, concept in enumerate(concepts):
            concept_id = concept["id"]
            if concept_id in accepted:
                continue
            if concept_index != len(accepted):
                raise ValueError("checkpoint is not a valid concept-order prefix")
            start_attempt = next_attempt.get(concept_id, 0)
            negative = concept_by_id[concept["hard_negative_id"]]
            anchor_row = anchor_by_id[concept_id]
            prompt = prompts[concept_id]
            for attempt in range(start_attempt, MAX_ATTEMPTS):
                seed = MASTER_SEED + 100 * concept_index + attempt
                set_seed(seed, torch, np)
                templated = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(templated, dict):
                    input_ids = templated["input_ids"]
                    attention_mask = templated.get(
                        "attention_mask", torch.ones_like(input_ids)
                    )
                else:
                    input_ids = templated
                    attention_mask = torch.ones_like(input_ids)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        repetition_penalty=REPETITION_PENALTY,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=(
                            tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None
                            else tokenizer.eos_token_id
                        ),
                    )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                raw = tokenizer.decode(
                    output[0, input_ids.shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                parsed, errors, diagnostics = assess_raw(
                    raw, concept, negative, anchor_row
                )
                row_without_hash = {
                    "schema_version": 2,
                    "experiment": EXPERIMENT,
                    "binding_sha256": binding_sha256,
                    "checkpoint_index": len(checkpoint_rows),
                    "previous_row_sha256": previous_hash,
                    "concept_id": concept_id,
                    "concept_index": concept_index,
                    "attempt": attempt,
                    "seed": seed,
                    "prompt_sha256": sha256_bytes(
                        prompt.encode("utf-8")
                    ),
                    "anchor_batch_sha256": anchor_batch_sha256(anchor_row),
                    "raw_sha256": sha256_bytes(raw.encode("utf-8")),
                    "seconds": seconds,
                    "admissible": not errors,
                    "errors": errors,
                    "diagnostics": diagnostics,
                    "parsed": parsed,
                    "raw": raw,
                }
                row_hash = sha256_bytes(
                    canonical_json(row_without_hash).encode("utf-8")
                )
                row = {**row_without_hash, "row_sha256": row_hash}
                append_checkpoint(args.checkpoint, row)
                checkpoint_rows.append(row)
                previous_hash = row_hash
                print(
                    f"[{concept_index + 1:02d}/24 {concept_id} "
                    f"attempt={attempt}] admissible={not errors} "
                    f"seconds={seconds:.1f} errors={errors[:3]}",
                    flush=True,
                )
                if not errors:
                    accepted[concept_id] = row
                    break
            if concept_id not in accepted:
                raise RuntimeError(
                    f"{concept_id} has no mechanically admissible complete "
                    f"batch after {MAX_ATTEMPTS} fixed attempts; stopping"
                )
        del model
        torch.cuda.empty_cache()

    if len(accepted) != 24:
        raise RuntimeError("generation ended without 24 accepted batches")
    manifest: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for concept in concepts:
        concept_id = concept["id"]
        row = accepted[concept_id]
        parsed = row["parsed"]
        negative = concept_by_id[concept["hard_negative_id"]]
        errors, diagnostics = validate_batch(
            parsed, concept, negative, anchor_by_id[concept_id]
        )
        if errors:
            raise ValueError(
                f"accepted batch failed final validation: "
                f"{concept_id}: {errors}"
            )
        quality.append(
            {
                "concept_id": concept_id,
                "attempt": row["attempt"],
                "seed": row["seed"],
                "checkpoint_index": row["checkpoint_index"],
                **diagnostics,
            }
        )
        for split in ("train", "test"):
            for ordinal, text in enumerate(parsed[split]):
                anchor = anchor_by_id[concept_id][split][ordinal]
                manifest.append(
                    {
                        "id": f"c1c3_{concept_id}_{split}_{ordinal:02d}",
                        "axis_domain": concept["superdomain"],
                        "axis_language": "en",
                        "split": split,
                        "topic": concept_id,
                        "text": text,
                        "concept_title": concept["title"],
                        "concept_summary": concept["summary"],
                        "hard_negative_id": concept["hard_negative_id"],
                        "anchor_id": anchor["anchor_id"],
                        "slot_id": anchor["slot_id"],
                        "discourse_role": anchor["role"],
                        "generation_attempt": row["attempt"],
                        "generation_seed": row["seed"],
                    }
                )
    discovery = [row for row in manifest if row["split"] == "train"]
    heldout = [row for row in manifest if row["split"] == "test"]
    if (
        len(manifest) != 144
        or len(discovery) != 96
        or len(heldout) != 48
        or len({row["id"] for row in manifest}) != 144
        or len({row["anchor_id"] for row in manifest}) != 144
    ):
        raise ValueError("final manifest coverage or uniqueness mismatch")
    manifest_data = jsonl_bytes(manifest)
    discovery_data = jsonl_bytes(discovery)
    heldout_data = jsonl_bytes(heldout)
    write_create_or_identical(args.out_manifest, manifest_data)
    write_create_or_identical(
        args.out_discovery_manifest, discovery_data
    )
    write_create_or_identical(args.out_heldout_manifest, heldout_data)
    outputs = {
        "combined_manifest": {
            "path": str(args.out_manifest),
            "sha256": sha256_bytes(manifest_data),
            "rows": 144,
        },
        "discovery_manifest": {
            "path": str(args.out_discovery_manifest),
            "sha256": sha256_bytes(discovery_data),
            "rows": 96,
        },
        "heldout_manifest": {
            "path": str(args.out_heldout_manifest),
            "sha256": sha256_bytes(heldout_data),
            "rows": 48,
        },
        "manifest": str(args.out_manifest),
        "manifest_sha256": sha256_bytes(manifest_data),
        "discovery_manifest": str(args.out_discovery_manifest),
        "discovery_manifest_sha256": sha256_bytes(discovery_data),
        "heldout_manifest": str(args.out_heldout_manifest),
        "heldout_manifest_sha256": sha256_bytes(heldout_data),
    }
    report = {
        "schema_version": 2,
        "experiment": EXPERIMENT,
        "status": "synthetic_corpus_frozen_before_activation_extraction",
        "binding_sha256": binding_sha256,
        "inputs": {
            **input_hashes,
            "spec_sha256": input_hashes["concept_spec_sha256"],
            "concept_spec": str(args.spec),
            "scenario_anchors": str(args.anchors),
            "preregistration_amendment": str(args.preregistration),
            "audit_addendum": str(args.rubric),
            "base_audit_rubric": str(args.base_rubric),
            "stage0_freeze": str(args.stage0_freeze),
            "corpus_generator": str(generator_path),
            "base_model": str(args.base_model),
            "base_model_files": {
                name: sha256_file(args.base_model / name)
                for name in (
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
                if (args.base_model / name).is_file()
            },
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_final_row_sha256": previous_hash,
        },
        "protocol": binding["protocol"],
        "counts": {
            "concepts": 24,
            "reciprocal_pairs": 12,
            "superdomains": 6,
            "documents": 144,
            "train_documents": 96,
            "test_documents": 48,
            "checkpoint_attempts": len(checkpoint_rows),
        },
        "quality": quality,
        "outputs": outputs,
        "attestation": {
            "fresh_v3_generation_only": True,
            "v2_generated_text_not_read_or_reused": True,
            "whole_batch_attempt_unit": True,
            "first_mechanically_admissible_batch_retained_immediately": True,
            "no_later_attempt_generated_or_compared_after_acceptance": True,
            "no_semantic_ranking_or_selection": True,
            "manual_semantic_audit_required_before_activation_extraction": True,
            "server_power_state_changed": False,
        },
    }
    report_data = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    write_create_or_identical(args.out_report, report_data)
    print("C1_CONFIRMATORY_CORPUS_V3_COMPLETE", flush=True)
    print(
        f"manifest_sha256={outputs['combined_manifest']['sha256']} "
        f"report_sha256={sha256_file(args.out_report)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
