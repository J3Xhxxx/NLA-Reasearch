#!/usr/bin/env python3
"""Freeze the C1-confirmatory reference benchmark before held-out/AV/AR work.

This stage consumes only the discovery-only feature selection asset.  It
freezes the three matched positive references, their three reciprocal
within-superdomain hard negatives, all other within-superdomain references
used by secondary analyses, and discovery activation contexts for the two
generated candidate sources.  It must run before held-out activations are read
and before either NLA AV or AR is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PASS_STATUS = "selection_frozen_before_AV_AR"
BENCHMARK_STATUS = "benchmark_frozen_before_C1_AV_AR_and_heldout"
CORPUS_STATUS = "synthetic_corpus_frozen_before_activation_extraction"
AUDIT_STATUS = "PASS"
PROVENANCE_STATUS = "PASS"


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
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    invalid_character = any(
        character not in "0123456789abcdef" for character in digest
    )
    if len(digest) != 64 or invalid_character:
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return digest


def scalar_text(archive: Any, name: str) -> str:
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"vectors array {name!r} must contain exactly one value")
    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def normalized_text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def candidate_id(
    feature: int,
    kind: str,
    ordinal: int,
    candidate_concept_id: str,
    text_sha256: str,
) -> str:
    text_digest = require_sha256(text_sha256, "candidate text_sha256")
    payload = canonical_json(
        {
            "experiment": "C1-confirmatory-v1",
            "feature": int(feature),
            "kind": kind,
            "ordinal": int(ordinal),
            "candidate_concept_id": candidate_concept_id,
            "text_sha256": text_digest,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def generated_candidate_id(feature: int, kind: str) -> str:
    # This is a generation-request identifier, frozen before generated text
    # exists.  The runtime stage binds the resulting normalized text separately.
    payload = canonical_json(
        {
            "experiment": "C1-confirmatory-v1",
            "feature": int(feature),
            "kind": kind,
            "ordinal": 0,
            "candidate_concept_id": "__generated__",
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_spec(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    concepts = spec.get("concepts")
    templates = spec.get("reference_templates")
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("spec has no concepts")
    if not isinstance(templates, list) or len(templates) != 3:
        raise ValueError("the frozen spec must contain exactly three templates")
    required = {"id", "superdomain", "title", "summary", "hard_negative_id"}
    for index, concept in enumerate(concepts):
        missing = required - set(concept)
        if missing:
            raise ValueError(f"concept {index} missing fields {sorted(missing)}")
    by_id = {str(row["id"]): row for row in concepts}
    if len(by_id) != len(concepts):
        raise ValueError("concept IDs are not unique")
    for concept in concepts:
        negative = by_id.get(str(concept["hard_negative_id"]))
        if (
            negative is None
            or str(negative["hard_negative_id"]) != str(concept["id"])
            or str(negative["superdomain"]) != str(concept["superdomain"])
        ):
            raise ValueError(
                "hard negatives must be reciprocal and within-superdomain: "
                f"{concept['id']}"
            )
    return spec, concepts


def selected_records(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("status") != PASS_STATUS:
        raise ValueError(
            f"selection status must be {PASS_STATUS!r}, got "
            f"{selection.get('status')!r}"
        )
    rows = [
        row
        for row in selection.get("selected_directions", [])
        if row.get("group") == "semantic_new" and int(row.get("feature", -1)) >= 0
    ]
    if not rows:
        raise ValueError("selection contains no semantic_new rows")
    features = [int(row["feature"]) for row in rows]
    if len(features) != len(set(features)):
        raise ValueError("selected feature IDs are not unique")
    labels = [str(row.get("label", "")) for row in rows]
    if any(not label for label in labels):
        raise ValueError("every selected feature must have a concept label")
    if len(rows) < 60 or len(set(labels)) < 18:
        raise ValueError(
            "selection does not meet the preregistered gate "
            f"(features={len(rows)}, concepts={len(set(labels))})"
        )
    return rows


def document_metadata(selection: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = selection.get("dataset")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection is missing its discovery dataset metadata")
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        doc_id = int(row["doc_id"])
        if doc_id in output:
            raise ValueError(f"duplicate discovery doc_id {doc_id}")
        split = str(row.get("split", "train"))
        if split != "train":
            raise ValueError("selection dataset must be discovery/train only")
        output[doc_id] = row
    return output


def compact_context(
    metadata: dict[str, Any],
    *,
    activation: float,
    document_score: float,
) -> dict[str, Any]:
    prompt = metadata.get("prompt", metadata.get("text"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"document {metadata.get('doc_id')} has no prompt text")
    return {
        "doc_id": int(metadata["doc_id"]),
        "prompt_id": str(metadata.get("prompt_id", metadata.get("id", ""))),
        "split": "train",
        "topic": str(metadata.get("topic", "")),
        "axis_domain": str(
            metadata.get("axis_domain", metadata.get("superdomain", ""))
        ),
        "axis_language": str(metadata.get("axis_language", "en")),
        "activation": float(activation),
        "document_score": float(document_score),
        "prompt": prompt.strip(),
    }


def discovery_contexts(
    feature: int,
    feature_column: int,
    activations: np.ndarray,
    row_doc_ids: np.ndarray,
    metadata_by_doc: dict[int, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    values = np.asarray(activations[:, feature_column], dtype=np.float64)
    if values.shape != row_doc_ids.shape:
        raise ValueError("activation rows and row_doc_ids disagree")
    scored: list[tuple[float, float, int]] = []
    for doc_id in sorted(set(row_doc_ids.tolist())):
        rows = values[row_doc_ids == doc_id]
        take = min(3, len(rows))
        top = np.partition(rows, len(rows) - take)[-take:]
        score = float(np.mean(top))
        maximum = float(np.max(rows))
        scored.append((score, maximum, int(doc_id)))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    output = []
    for score, maximum, doc_id in scored[:limit]:
        if doc_id not in metadata_by_doc:
            raise ValueError(f"row_doc_ids contains unknown doc_id {doc_id}")
        output.append(
            compact_context(
                metadata_by_doc[doc_id],
                activation=maximum,
                document_score=score,
            )
        )
    if not output or output[0]["document_score"] <= 0:
        raise ValueError(f"feature {feature} has no positive discovery context")
    return output


def reference_candidate(
    feature: int,
    assigned_concept_id: str,
    candidate_concept: dict[str, Any],
    template: str,
    ordinal: int,
    kind: str,
) -> dict[str, Any]:
    text = normalize_text(
        template.format(
            title=str(candidate_concept["title"]),
            summary=str(candidate_concept["summary"]),
        )
    )
    if not text or "{" in text or "}" in text:
        raise ValueError(
            f"invalid populated reference {candidate_concept['id']} template {ordinal}"
        )
    concept_id = str(candidate_concept["id"])
    text_sha256 = normalized_text_sha256(text)
    return {
        "candidate_id": candidate_id(
            feature,
            kind,
            ordinal,
            concept_id,
            text_sha256,
        ),
        "kind": kind,
        "ordinal": int(ordinal),
        "assigned_concept_id": assigned_concept_id,
        "candidate_concept_id": concept_id,
        "candidate_superdomain": str(candidate_concept["superdomain"]),
        "text": text,
        "text_sha256": text_sha256,
        "generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--corpus-report", required=True, type=Path)
    parser.add_argument("--manual-audit", required=True, type=Path)
    parser.add_argument("--discovery-provenance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-discovery-contexts", type=int, default=4)
    args = parser.parse_args()

    if args.max_discovery_contexts < 1:
        raise ValueError("--max-discovery-contexts must be positive")
    input_paths = (
        args.selection,
        args.vectors,
        args.spec,
        args.corpus_report,
        args.manual_audit,
        args.discovery_provenance,
    )
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    input_hashes = {path: sha256_file(path) for path in input_paths}
    spec, concepts = load_spec(args.spec)
    concept_by_id = {str(row["id"]): row for row in concepts}
    concept_order = [str(row["id"]) for row in concepts]
    templates = [str(value) for value in spec["reference_templates"]]
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    corpus_report = json.loads(args.corpus_report.read_text(encoding="utf-8"))
    manual_audit = json.loads(args.manual_audit.read_text(encoding="utf-8"))
    discovery_provenance = json.loads(
        args.discovery_provenance.read_text(encoding="utf-8")
    )
    for label, value in (
        ("selection", selection),
        ("corpus report", corpus_report),
        ("manual audit", manual_audit),
        ("discovery provenance", discovery_provenance),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{label} must contain a JSON object")

    records = selected_records(selection)
    metadata_by_doc = document_metadata(selection)

    if corpus_report.get("status") != CORPUS_STATUS:
        raise ValueError(
            f"corpus report status must be {CORPUS_STATUS!r}, got "
            f"{corpus_report.get('status')!r}"
        )
    corpus_outputs = corpus_report.get("outputs")
    if not isinstance(corpus_outputs, dict):
        raise ValueError("corpus report is missing outputs")
    corpus_manifest_sha = require_sha256(
        corpus_outputs.get("manifest_sha256"),
        "corpus report combined manifest_sha256",
    )
    corpus_discovery_sha = require_sha256(
        corpus_outputs.get("discovery_manifest_sha256"),
        "corpus report discovery_manifest_sha256",
    )
    corpus_heldout_sha = require_sha256(
        corpus_outputs.get("heldout_manifest_sha256"),
        "corpus report heldout_manifest_sha256",
    )
    corpus_inputs = corpus_report.get("inputs")
    if not isinstance(corpus_inputs, dict):
        raise ValueError("corpus report is missing inputs")
    if (
        require_sha256(
            corpus_inputs.get("spec_sha256"),
            "corpus report concept-spec SHA256",
        )
        != input_hashes[args.spec]
    ):
        raise ValueError(
            "corpus report concept-spec SHA does not match the supplied spec"
        )

    if str(manual_audit.get("status", "")).upper() != AUDIT_STATUS:
        raise ValueError(
            f"manual audit status must be {AUDIT_STATUS!r}, got "
            f"{manual_audit.get('status')!r}"
        )
    audit_manifest_sha = require_sha256(
        manual_audit.get("combined_manifest_sha256"),
        "manual audit combined_manifest_sha256",
    )
    audit_discovery_sha = require_sha256(
        manual_audit.get("discovery_manifest_sha256"),
        "manual audit discovery_manifest_sha256",
    )
    audit_heldout_sha = require_sha256(
        manual_audit.get("heldout_manifest_sha256"),
        "manual audit heldout_manifest_sha256",
    )
    audit_rubric_sha = require_sha256(
        manual_audit.get("rubric_sha256"),
        "manual audit rubric_sha256",
    )
    if (
        audit_manifest_sha != corpus_manifest_sha
        or audit_discovery_sha != corpus_discovery_sha
        or audit_heldout_sha != corpus_heldout_sha
    ):
        raise ValueError(
            "manual audit manifest hashes do not match the frozen corpus report"
        )

    if str(discovery_provenance.get("status", "")).upper() != PROVENANCE_STATUS:
        raise ValueError(
            f"discovery provenance status must be {PROVENANCE_STATUS!r}, got "
            f"{discovery_provenance.get('status')!r}"
        )
    provenance_parameters = discovery_provenance.get("parameters")
    provenance_hashes = discovery_provenance.get("hashes")
    provenance_audit = discovery_provenance.get("manual_audit")
    provenance_counts = discovery_provenance.get("counts")
    if not isinstance(provenance_parameters, dict):
        raise ValueError("discovery provenance is missing parameters")
    if not isinstance(provenance_hashes, dict):
        raise ValueError("discovery provenance is missing hashes")
    if not isinstance(provenance_audit, dict):
        raise ValueError("discovery provenance is missing manual_audit")
    if not isinstance(provenance_counts, dict):
        raise ValueError("discovery provenance is missing counts")
    expected_provenance_parameters = {
        "expected_split": "train",
        "expected_documents": 96,
        "layer_index": 32,
        "min_position": 50,
        "max_per_prompt": 0,
        "dtype": "bfloat16",
    }
    observed_provenance_parameters = {
        key: provenance_parameters.get(key)
        for key in expected_provenance_parameters
    }
    if observed_provenance_parameters != expected_provenance_parameters:
        raise ValueError(
            "discovery provenance parameters differ from protocol: "
            f"{observed_provenance_parameters!r} != "
            f"{expected_provenance_parameters!r}"
        )
    if (
        provenance_parameters.get(
            "parquet_extraction_metadata_schema_version"
        )
        != 1
        or provenance_parameters.get(
            "verified_against_parquet_schema_metadata"
        )
        is not True
    ):
        raise ValueError(
            "discovery provenance did not verify extraction settings against "
            "Parquet schema metadata"
        )
    provenance_manifest_sha = require_sha256(
        provenance_hashes.get("manifest_sha256"),
        "discovery provenance manifest_sha256",
    )
    provenance_activations_sha = require_sha256(
        provenance_hashes.get("activations_sha256"),
        "discovery provenance activations_sha256",
    )
    provenance_audit_sha = require_sha256(
        provenance_hashes.get("manual_audit_sha256"),
        "discovery provenance manual_audit_sha256",
    )
    provenance_model_sha = require_sha256(
        provenance_hashes.get("base_model_identity_sha256"),
        "discovery provenance base_model_identity_sha256",
    )
    provenance_extractor_sha = require_sha256(
        provenance_hashes.get("extractor_sha256"),
        "discovery provenance extractor_sha256",
    )
    if provenance_manifest_sha != corpus_discovery_sha:
        raise ValueError(
            "discovery provenance manifest SHA does not match the corpus "
            "report discovery output"
        )
    if provenance_audit_sha != input_hashes[args.manual_audit]:
        raise ValueError(
            "discovery provenance manual-audit SHA does not match the "
            "supplied manual-audit file"
        )
    if str(provenance_audit.get("status", "")).upper() != AUDIT_STATUS:
        raise ValueError("discovery provenance embeds a non-PASS manual audit")
    if (
        require_sha256(
            provenance_audit.get("manifest_sha256"),
            "discovery provenance embedded manual-audit manifest_sha256",
        )
        != corpus_discovery_sha
    ):
        raise ValueError(
            "discovery provenance embedded manual-audit manifest SHA does "
            "not match the discovery corpus"
        )

    selection_inputs = selection.get("inputs")
    if not isinstance(selection_inputs, dict):
        raise ValueError("selection is missing inputs")
    selection_activation_sha = require_sha256(
        selection_inputs.get("activations_sha256"),
        "selection activations_sha256",
    )
    selection_spec_sha = require_sha256(
        selection_inputs.get("concept_spec_sha256"),
        "selection concept_spec_sha256",
    )
    selection_denylist = selection_inputs.get("denylist")
    if not isinstance(selection_denylist, dict):
        raise ValueError("selection inputs are missing denylist audit")
    selection_denylist_sha = require_sha256(
        selection_denylist.get("manifest_sha256"),
        "selection denylist manifest_sha256",
    )
    selection_sae_sha = require_sha256(
        selection_inputs.get("sae_params_sha256"),
        "selection SAE params SHA256",
    )
    if selection_activation_sha != provenance_activations_sha:
        raise ValueError(
            "selection activation SHA does not match discovery provenance"
        )
    if selection_spec_sha != input_hashes[args.spec]:
        raise ValueError(
            "selection concept-spec SHA does not match the supplied spec"
        )

    with np.load(args.vectors, allow_pickle=False) as archive:
        required_arrays = {
            "row_doc_ids",
            "m_hat",
            "target_norm",
            "direction_ids",
            "directions",
            "selected_feature_activations",
            "activations_sha256",
            "concept_spec_sha256",
            "denylist_sha256",
            "sae_params_sha256",
        }
        missing = required_arrays - set(archive.files)
        if missing:
            raise ValueError(f"vectors asset missing arrays {sorted(missing)}")
        row_doc_ids = np.asarray(archive["row_doc_ids"], dtype=np.int64)
        m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
        target_norm = float(np.asarray(archive["target_norm"]).reshape(()))
        direction_ids = np.asarray(archive["direction_ids"], dtype=np.int64)
        directions = np.asarray(archive["directions"], dtype=np.float32)
        activations = np.asarray(
            archive["selected_feature_activations"], dtype=np.float32
        )
        vector_embedded_hashes = {
            "activations_sha256": require_sha256(
                scalar_text(archive, "activations_sha256"),
                "vectors embedded activations_sha256",
            ),
            "concept_spec_sha256": require_sha256(
                scalar_text(archive, "concept_spec_sha256"),
                "vectors embedded concept_spec_sha256",
            ),
            "denylist_sha256": require_sha256(
                scalar_text(archive, "denylist_sha256"),
                "vectors embedded denylist_sha256",
            ),
            "sae_params_sha256": require_sha256(
                scalar_text(archive, "sae_params_sha256"),
                "vectors embedded sae_params_sha256",
            ),
        }
    expected_embedded_hashes = {
        "activations_sha256": selection_activation_sha,
        "concept_spec_sha256": selection_spec_sha,
        "denylist_sha256": selection_denylist_sha,
        "sae_params_sha256": selection_sae_sha,
    }
    if vector_embedded_hashes != expected_embedded_hashes:
        mismatches = {
            key: {
                "selection": expected_embedded_hashes[key],
                "vectors": vector_embedded_hashes[key],
            }
            for key in expected_embedded_hashes
            if vector_embedded_hashes[key] != expected_embedded_hashes[key]
        }
        raise ValueError(
            f"vector embedded hashes do not match selection inputs: {mismatches}"
        )

    expected_documents = 96
    if len(metadata_by_doc) != expected_documents:
        raise ValueError(
            "selection discovery dataset must contain exactly 96 documents"
        )
    for key in ("manifest_documents", "parquet_documents"):
        if int(provenance_counts.get(key, -1)) != expected_documents:
            raise ValueError(
                f"discovery provenance {key} does not match selection dataset "
                f"({provenance_counts.get(key)!r} != {expected_documents})"
            )
    feature_ids = np.asarray([int(row["feature"]) for row in records], dtype=np.int64)
    if not np.array_equal(direction_ids, feature_ids):
        raise ValueError("selection records and vector direction_ids differ in order")
    if directions.shape[0] != len(records) or directions.ndim != 2:
        raise ValueError("unexpected directions shape")
    if activations.shape != (len(row_doc_ids), len(records)):
        raise ValueError(
            "selected_feature_activations must have one column per selected feature"
        )
    if set(row_doc_ids.tolist()) != set(metadata_by_doc):
        raise ValueError("row_doc_ids and discovery dataset doc IDs differ")
    if not np.isfinite(directions).all() or not np.isfinite(activations).all():
        raise ValueError("non-finite vector or activation")
    m_norm = float(np.linalg.norm(m_hat))
    if not np.isfinite(m_norm) or abs(m_norm - 1.0) > 1e-4:
        raise ValueError(f"m_hat must be unit norm, got {m_norm}")
    if not np.isfinite(target_norm) or target_norm <= 0:
        raise ValueError(f"invalid target_norm {target_norm}")

    output_records: list[dict[str, Any]] = []
    for direction_index, selected in enumerate(records):
        feature = int(selected["feature"])
        concept_id = str(selected["label"])
        if concept_id not in concept_by_id:
            raise ValueError(f"selected feature {feature} has unknown label {concept_id}")
        concept = concept_by_id[concept_id]
        if (
            selected.get("superdomain") is not None
            and str(selected["superdomain"]) != str(concept["superdomain"])
        ):
            raise ValueError(f"superdomain drift for feature {feature}")
        negative_id = str(concept["hard_negative_id"])
        domain_concepts = [
            row
            for row in concepts
            if str(row["superdomain"]) == str(concept["superdomain"])
        ]
        static: list[dict[str, Any]] = []
        for ordinal, template in enumerate(templates):
            static.append(
                reference_candidate(
                    feature,
                    concept_id,
                    concept,
                    template,
                    ordinal,
                    "correct_reference",
                )
            )
            static.append(
                reference_candidate(
                    feature,
                    concept_id,
                    concept_by_id[negative_id],
                    template,
                    ordinal,
                    "hard_negative_reference",
                )
            )
        for other in domain_concepts:
            other_id = str(other["id"])
            if other_id in {concept_id, negative_id}:
                continue
            for ordinal, template in enumerate(templates):
                static.append(
                    reference_candidate(
                        feature,
                        concept_id,
                        other,
                        template,
                        ordinal,
                        "other_within_superdomain_reference",
                    )
                )
        primary_correct = [
            row for row in static if row["kind"] == "correct_reference"
        ]
        primary_negative = [
            row for row in static if row["kind"] == "hard_negative_reference"
        ]
        if (
            len(primary_correct) != 3
            or len(primary_negative) != 3
            or [row["ordinal"] for row in primary_correct] != [0, 1, 2]
            or [row["ordinal"] for row in primary_negative] != [0, 1, 2]
        ):
            raise AssertionError("primary matched reference construction drifted")
        output_records.append(
            {
                "feature": feature,
                "direction_index": direction_index,
                "concept_id": concept_id,
                "superdomain": str(concept["superdomain"]),
                "hard_negative_id": negative_id,
                "selection_tier": str(selected["selection_tier"]),
                "train_metrics": selected["train"],
                "discovery_contexts": discovery_contexts(
                    feature,
                    direction_index,
                    activations,
                    row_doc_ids,
                    metadata_by_doc,
                    args.max_discovery_contexts,
                ),
                "static_candidates": static,
                "generation_requests": [
                    {
                        "kind": "nla_av",
                        "candidate_id": generated_candidate_id(feature, "nla_av"),
                    },
                    {
                        "kind": "base_autointerp",
                        "candidate_id": generated_candidate_id(
                            feature, "base_autointerp"
                        ),
                    },
                ],
            }
        )

    all_ids = [
        candidate["candidate_id"]
        for record in output_records
        for candidate in record["static_candidates"]
    ] + [
        request["candidate_id"]
        for record in output_records
        for request in record["generation_requests"]
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("candidate IDs are not globally unique")
    for record in output_records:
        for candidate in record["static_candidates"]:
            normalized = normalize_text(candidate["text"])
            text_sha = normalized_text_sha256(normalized)
            if candidate["text"] != normalized:
                raise ValueError(
                    f"static candidate {candidate['candidate_id']} is not normalized"
                )
            if candidate.get("text_sha256") != text_sha:
                raise ValueError(
                    f"static candidate {candidate['candidate_id']} text SHA drifted"
                )
            expected_id = candidate_id(
                int(record["feature"]),
                str(candidate["kind"]),
                int(candidate["ordinal"]),
                str(candidate["candidate_concept_id"]),
                text_sha,
            )
            if candidate["candidate_id"] != expected_id:
                raise ValueError(
                    f"static candidate ID does not bind its text: "
                    f"{candidate['candidate_id']}"
                )
    selected_concepts = {row["concept_id"] for row in output_records}
    counts = {
        concept_id: sum(row["concept_id"] == concept_id for row in output_records)
        for concept_id in concept_order
        if concept_id in selected_concepts
    }
    if any(count > 4 for count in counts.values()):
        raise ValueError(f"per-concept selection quota exceeded: {counts}")
    reciprocal_pairs: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for concept in concepts:
        pair = tuple(
            sorted(
                (
                    str(concept["id"]),
                    str(concept["hard_negative_id"]),
                )
            )
        )
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            reciprocal_pairs.append(list(pair))
    complete_pairs = [
        pair
        for pair in reciprocal_pairs
        if pair[0] in selected_concepts and pair[1] in selected_concepts
    ]
    if len(complete_pairs) < 9:
        raise ValueError(
            "selection fails the revised reciprocal-pair analysis gate: "
            f"complete_pairs={len(complete_pairs)} < 9"
        )

    payload = {
        "schema_version": 1,
        "experiment": "C1 confirmatory synthetic cohort v1",
        "status": BENCHMARK_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "confirmatory": True,
            "n_features": len(output_records),
            "n_concepts": len(selected_concepts),
            "n_superdomains": len(
                {row["superdomain"] for row in output_records}
            ),
            "statistical_unit": "reciprocal hard-negative concept pair",
            "feature_counts_by_concept": counts,
            "complete_reciprocal_pairs": complete_pairs,
            "n_complete_reciprocal_pairs": len(complete_pairs),
        },
        "inputs": {
            "selection": {
                "path": str(args.selection),
                "sha256": input_hashes[args.selection],
                "status": str(selection["status"]),
                "activations_sha256": selection_activation_sha,
                "concept_spec_sha256": selection_spec_sha,
                "denylist_sha256": selection_denylist_sha,
                "sae_params_sha256": selection_sae_sha,
            },
            "vectors": {
                "path": str(args.vectors),
                "sha256": input_hashes[args.vectors],
                "activations_sha256": vector_embedded_hashes[
                    "activations_sha256"
                ],
                "concept_spec_sha256": vector_embedded_hashes[
                    "concept_spec_sha256"
                ],
                "denylist_sha256": vector_embedded_hashes[
                    "denylist_sha256"
                ],
                "sae_params_sha256": vector_embedded_hashes[
                    "sae_params_sha256"
                ],
            },
            "spec": {
                "path": str(args.spec),
                "sha256": input_hashes[args.spec],
            },
            "corpus_report": {
                "path": str(args.corpus_report),
                "sha256": input_hashes[args.corpus_report],
                "status": str(corpus_report["status"]),
                "manifest_sha256": corpus_manifest_sha,
                "discovery_manifest_sha256": corpus_discovery_sha,
                "heldout_manifest_sha256": corpus_heldout_sha,
            },
            "manual_audit": {
                "path": str(args.manual_audit),
                "sha256": input_hashes[args.manual_audit],
                "status": AUDIT_STATUS,
                "rubric_sha256": audit_rubric_sha,
                "combined_manifest_sha256": audit_manifest_sha,
                "discovery_manifest_sha256": audit_discovery_sha,
                "heldout_manifest_sha256": audit_heldout_sha,
            },
            "discovery_provenance": {
                "path": str(args.discovery_provenance),
                "sha256": input_hashes[args.discovery_provenance],
                "status": PROVENANCE_STATUS,
                "extraction_parameters": expected_provenance_parameters,
                "manifest_sha256": provenance_manifest_sha,
                "activations_sha256": provenance_activations_sha,
                "extractor_sha256": provenance_extractor_sha,
                "manual_audit_sha256": provenance_audit_sha,
                "base_model_identity_sha256": provenance_model_sha,
            },
        },
        "protocol": {
            "reference_templates": templates,
            "correct_references_per_feature": 3,
            "hard_negative_references_per_feature": 3,
            "hard_negative_mapping": "frozen reciprocal within-superdomain pair",
            "primary_feature_effect": (
                "mean centered q over three correct references minus mean "
                "centered q over the three matched reciprocal hard negatives"
            ),
            "primary_cluster_effect": (
                "mean feature effect within concept; concepts equally weighted"
            ),
            "primary_pair_effect": (
                "mean of the two concept-cluster effects in each complete "
                "reciprocal hard-negative pair; only pairs with selected "
                "features on both sides are included"
            ),
            "primary_inference": (
                "exact joint pair-level random-sign test over complete pairs; "
                "20,000 pair bootstrap resamples for the percentile interval"
            ),
            "pair_analysis_gate": "at least 9 complete reciprocal pairs",
            "centering": (
                "project AR reconstruction and SAE decoder row orthogonally "
                "to discovery-only m_hat, then L2 normalize"
            ),
            "heldout_boundary": (
                "this benchmark contains no held-out activation, metric, or "
                "context; held-out data may be read only after this file is frozen"
            ),
            "generated_secondary_candidates": [
                "greedy NLA AV explanation of target_norm-scaled decoder direction",
                "plain base-model autointerpretation from discovery contexts",
            ],
            "static_candidate_id_binding": (
                "first 20 hex characters of SHA-256 over canonical JSON "
                "including the SHA-256 of normalized candidate text"
            ),
            "generated_candidate_id_scope": (
                "pre-generation request identifier; generated normalized-text "
                "binding is finalized and validated by the runtime stage"
            ),
            "excluded_endpoint": "Gemma-family automatic judge",
        },
        "records": output_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("C1_CONFIRMATORY_BENCHMARK_FROZEN")
    print(
        canonical_json(
            {
                "features": len(output_records),
                "concepts": len(selected_concepts),
                "static_candidate_pairings": sum(
                    len(row["static_candidates"]) for row in output_records
                ),
                "generated_candidates": 2 * len(output_records),
                "out": str(args.out),
                "sha256": sha256_file(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
