#!/usr/bin/env python3
"""Construct a Pile-only N6 provisional cohort without running a model.

The current N6 preregistration is deliberately rejected while it remains a
``.DRAFT``.  The corrected pre-freeze audit fixes seed 20260803, caps each
source at 40 without requiring every source to fill the cap, applies global
N4/N5/N6 20-word-shingle exclusion, and requires at least 480 provisional rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from n6_common import (
    canonical_sha256,
    parse_weight_manifest,
    require_binding_preregistration,
    require_exact_hash,
    sha256_file,
    shahex,
    verify_code_manifest,
    verify_sha256_sidecar,
    write_new_json,
)


PILE_SOURCES = (
    "ArXiv",
    "DM Mathematics",
    "FreeLaw",
    "Github",
    "HackerNews",
    "NIH ExPorter",
    "OpenWebText2",
    "Pile-CC",
    "PubMed Abstracts",
    "PubMed Central",
    "StackExchange",
    "USPTO Backgrounds",
    "Wikipedia (en)",
)
SEQ_LEN = 512
MIN_POSITION = 64
MAX_POSITION = 480
MIN_CONTINUATION = 16
CONTENT_FRAC = 0.75
SHINGLE_WORDS = 20
SELECTION_SEED = 20260803
SOURCE_CAP = 40
MIN_PROVISIONAL_TOTAL = 480
EXPECTED_N3_CORPUS_SHA256 = (
    "d40069ab51c294ecbe3e76845d1f2f4dff1bb66a6061c5b6b4c612f7d0ff8816"
)
EXPECTED_N3_MANIFEST_SHA256 = (
    "500d5b88b78c8bc06ff7965c0dffcc25cb5b0e9f50bfa8ec1ae009f9312d6046"
)
EXPECTED_PILE_PARQUET_SHA256 = (
    "a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31"
)
EXPECTED_N4_ACTIVATIONS_SHA256 = (
    "eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66"
)
EXPECTED_N5_PLAN_SHA256 = (
    "6e7394476c4769bcb3334bbc82ca078fc778e4f006d9d600dac3882983cafb4c"
)
EXPECTED_N5_MODEL_MANIFEST_SHA256 = (
    "4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735"
)
WORD_RE = re.compile(r"\w", re.UNICODE)
WORDS_RE = re.compile(r"\w+", re.UNICODE)


def tokenizer_file_hashes(model_dir: Path) -> dict[str, str]:
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def normalized_words(text: str) -> list[str]:
    return WORDS_RE.findall(unicodedata.normalize("NFKC", text).lower())


def word_shingles(words: list[str]) -> set[bytes]:
    if len(words) < SHINGLE_WORDS:
        return set()
    return {
        hashlib.sha256(
            "\x1f".join(words[index : index + SHINGLE_WORDS]).encode("utf-8")
        ).digest()
        for index in range(len(words) - SHINGLE_WORDS + 1)
    }


def shingle_digest(shingles: set[bytes]) -> str:
    digest = hashlib.sha256()
    for value in sorted(shingles):
        digest.update(value)
    return digest.hexdigest()


def decoded_prefix(tokenizer, input_ids: list[int]) -> str:
    return tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def make_content_predicate(tokenizer) -> Callable[[int], bool]:
    special = set(int(value) for value in tokenizer.all_special_ids)
    cache: dict[int, bool] = {}

    def content(token_id: int) -> bool:
        token_id = int(token_id)
        if token_id in cache:
            return cache[token_id]
        if token_id in special:
            answer = False
        else:
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            stripped = text.strip()
            answer = bool(
                stripped
                and not (stripped.startswith("<") and stripped.endswith(">"))
                and WORD_RE.search(text)
            )
        cache[token_id] = answer
        return answer

    return content


def eligible_positions(
    input_ids: list[int], content: Callable[[int], bool]
) -> list[int]:
    high = min(MAX_POSITION, len(input_ids) - MIN_CONTINUATION - 1)
    if high < MIN_POSITION:
        return []
    result = []
    for position in range(MIN_POSITION, high + 1):
        if not content(input_ids[position]):
            continue
        continuation = input_ids[
            position + 1 : position + 1 + MIN_CONTINUATION
        ]
        if len(continuation) != MIN_CONTINUATION:
            continue
        fraction = sum(content(token) for token in continuation) / MIN_CONTINUATION
        if fraction >= CONTENT_FRAC:
            result.append(position)
    return result


def build_candidate(
    *,
    tokenizer,
    content: Callable[[int], bool],
    text: str,
    source: str,
    orig_index: int,
    content_group_id: str,
    selection_seed: int,
) -> tuple[dict[str, Any], set[bytes]] | None:
    input_ids = [
        int(value)
        for value in tokenizer(text, add_special_tokens=True)["input_ids"][:SEQ_LEN]
    ]
    positions = eligible_positions(input_ids, content)
    if not positions:
        return None
    position = min(
        positions,
        key=lambda value: shahex(
            selection_seed, "position", source, content_group_id, value
        ),
    )
    prefix = decoded_prefix(tokenizer, input_ids)
    words = normalized_words(prefix)
    shingles = word_shingles(words)
    token_id = int(input_ids[position])
    row = {
        "content_group_id": content_group_id,
        "corpus": "pile",
        "source": source,
        "lang": "en",
        "orig_index": int(orig_index),
        "passage_id": -1,
        "position": int(position),
        "position_rank_sha256": shahex(
            selection_seed, "position", source, content_group_id, position
        ),
        "candidate_rank_sha256": shahex(
            selection_seed, "candidate", source, content_group_id, position
        ),
        "token": tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "token_id": token_id,
        "seq_len": len(input_ids),
        "input_ids": input_ids,
        "input_ids_sha256": shahex(*input_ids),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prefix_text_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "prefix_normalized_words_sha256": shahex(*words),
        "prefix_word_count": len(words),
        "prefix_shingle_count": len(shingles),
        "prefix_shingle_set_sha256": shingle_digest(shingles),
        "context_tail": tokenizer.decode(
            input_ids[max(0, position - 24) : position + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "continuation": tokenizer.decode(
            input_ids[position + 1 : position + 1 + MIN_CONTINUATION],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "norm": "none",
    }
    return row, shingles


def load_n4_embargo(
    path: Path,
    corpus_by_doc_id: dict[int, dict[str, Any]],
    tokenizer,
) -> tuple[set[int], set[bytes], dict[str, Any]]:
    table = pq.read_table(path, columns=["doc_id", "input_ids"])
    doc_ids = [int(value) for value in table["doc_id"].to_pylist()]
    sequences = table["input_ids"].to_pylist()
    by_doc: dict[int, list[int]] = {}
    for doc_id, sequence in zip(doc_ids, sequences):
        ids = [int(token) for token in sequence]
        if doc_id in by_doc and by_doc[doc_id] != ids:
            raise ValueError(f"N4 doc_id {doc_id} has inconsistent input_ids")
        by_doc[doc_id] = ids
    missing = sorted(set(by_doc) - set(corpus_by_doc_id))
    if missing:
        raise ValueError(f"N4 doc_ids missing from corpus: {missing[:10]}")
    pile_ids = {
        doc_id
        for doc_id in by_doc
        if str(corpus_by_doc_id[doc_id].get("corpus")).lower() == "pile"
    }
    shingles: set[bytes] = set()
    for doc_id in pile_ids:
        shingles.update(
            word_shingles(normalized_words(decoded_prefix(tokenizer, by_doc[doc_id])))
        )
    return pile_ids, shingles, {
        "n_rows": len(doc_ids),
        "n_document_ids": len(by_doc),
        "n_pile_document_ids": len(pile_ids),
        "n_prefix_shingles": len(shingles),
        "prefix_shingle_union_sha256": shingle_digest(shingles),
    }


def load_n5_embargo(
    path: Path,
    tokenizer,
) -> tuple[set[int], set[str], set[bytes], dict[str, Any], str]:
    plan_sha = verify_sha256_sidecar(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("N5 plan lacks inspectable rows")
    pile_rows = [
        row for row in rows if str(row.get("corpus", "")).lower() == "pile"
    ]
    orig_indices = {int(row["orig_index"]) for row in pile_rows}
    content_groups = {str(row["content_group_id"]) for row in pile_rows}
    if len(orig_indices) != len(pile_rows) or len(content_groups) != len(pile_rows):
        raise ValueError("N5 Pile rows repeat document/content identity")
    shingles: set[bytes] = set()
    for row in pile_rows:
        ids = [int(value) for value in row["input_ids"]]
        shingles.update(
            word_shingles(normalized_words(decoded_prefix(tokenizer, ids)))
        )
    return orig_indices, content_groups, shingles, {
        "n_plan_rows": len(rows),
        "n_pile_rows": len(pile_rows),
        "n_pile_orig_indices": len(orig_indices),
        "n_pile_content_groups": len(content_groups),
        "n_prefix_shingles": len(shingles),
        "prefix_shingle_union_sha256": shingle_digest(shingles),
    }, plan_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("/root/autodl-tmp/results/n3_corpus_v1.jsonl"),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("/root/autodl-tmp/results/n3_corpus_v1.json"),
    )
    parser.add_argument(
        "--n4-activations",
        type=Path,
        default=Path("/root/autodl-tmp/activations/acts_L32_n3_v1.parquet"),
    )
    parser.add_argument(
        "--n5-plan",
        type=Path,
        default=Path("/root/autodl-tmp/results/n5_cohort_plan_v2.json"),
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/root/autodl-tmp/models/gemma-3-12b-it"),
    )
    parser.add_argument("--pile-parquet", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prereg_sha = require_binding_preregistration(args.prereg)
    code_manifest_sha = verify_code_manifest(args.code_manifest, __file__)
    corpus_sha = require_exact_hash(
        args.corpus, EXPECTED_N3_CORPUS_SHA256, "frozen N3 corpus JSONL"
    )
    manifest_sha = require_exact_hash(
        args.corpus_manifest,
        EXPECTED_N3_MANIFEST_SHA256,
        "frozen N3 corpus manifest",
    )
    n4_sha = require_exact_hash(
        args.n4_activations,
        EXPECTED_N4_ACTIVATIONS_SHA256,
        "frozen N4 activation parquet",
    )
    pile_sha = require_exact_hash(
        args.pile_parquet,
        EXPECTED_PILE_PARQUET_SHA256,
        "frozen Pile parquet",
    )
    model_manifest_sha = verify_sha256_sidecar(args.model_manifest)
    if model_manifest_sha != EXPECTED_N5_MODEL_MANIFEST_SHA256:
        raise ValueError(
            "N6 requires the frozen N5 combined model manifest: "
            f"{model_manifest_sha} != {EXPECTED_N5_MODEL_MANIFEST_SHA256}"
        )
    parse_weight_manifest(args.model_manifest)

    corpus_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    declared_corpus_sha = corpus_manifest.get("outputs", {}).get("jsonl_sha256")
    if declared_corpus_sha != corpus_sha:
        raise ValueError("N3 corpus manifest does not bind the supplied JSONL")
    declared_pile_sha = (
        corpus_manifest.get("provenance", {}).get("pile", {}).get("local_sha256")
    )
    if declared_pile_sha != pile_sha:
        raise ValueError("N3 corpus manifest does not bind the supplied Pile parquet")

    corpus_rows = load_jsonl(args.corpus)
    corpus_by_doc_id = {int(row["doc_id"]): row for row in corpus_rows}
    if len(corpus_by_doc_id) != len(corpus_rows):
        raise ValueError("N3 corpus doc_id values are not unique")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True
    )
    tokenizer_hashes = tokenizer_file_hashes(args.base_model)
    content = make_content_predicate(tokenizer)

    n4_doc_ids, n4_shingles, n4_report = load_n4_embargo(
        args.n4_activations, corpus_by_doc_id, tokenizer
    )
    (
        n5_orig_indices,
        n5_content_groups,
        n5_shingles,
        n5_report,
        n5_plan_sha,
    ) = load_n5_embargo(args.n5_plan, tokenizer)
    if n5_plan_sha != EXPECTED_N5_PLAN_SHA256:
        raise ValueError(
            "N6 requires the frozen N5 cohort plan: "
            f"{n5_plan_sha} != {EXPECTED_N5_PLAN_SHA256}"
        )
    embargo_shingles = n4_shingles | n5_shingles

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus_rows:
        if str(row.get("corpus", "")).lower() == "pile":
            source = str(row.get("source"))
            if source in PILE_SOURCES:
                by_source[source].append(row)
    if set(by_source) != set(PILE_SOURCES):
        raise ValueError(
            f"Pile source drift: missing={sorted(set(PILE_SOURCES)-set(by_source))}"
        )

    eligible: dict[str, list[tuple[dict[str, Any], set[bytes]]]] = {}
    diagnostics: dict[str, Any] = {}
    for source in PILE_SOURCES:
        skipped: Counter[str] = Counter()
        candidates: list[tuple[dict[str, Any], set[bytes]]] = []
        documents = sorted(
            by_source[source],
            key=lambda row: shahex(
                SELECTION_SEED,
                "pile-document",
                source,
                row["orig_index"],
                row["text_sha256"],
            ),
        )
        for document in documents:
            doc_id = int(document["doc_id"])
            orig_index = int(document["orig_index"])
            group = f"pile:{pile_sha}:{orig_index}"
            if doc_id in n4_doc_ids:
                skipped["n4_document_identity"] += 1
                continue
            if orig_index in n5_orig_indices or group in n5_content_groups:
                skipped["n5_document_or_content_identity"] += 1
                continue
            built = build_candidate(
                tokenizer=tokenizer,
                content=content,
                text=str(document["text"]),
                source=source,
                orig_index=orig_index,
                content_group_id=group,
                selection_seed=SELECTION_SEED,
            )
            if built is None:
                skipped["content_token_position_ineligible"] += 1
                continue
            candidate, shingles = built
            if shingles & embargo_shingles:
                skipped["n4_or_n5_prefix_shingle_embargo"] += 1
                continue
            candidates.append((candidate, shingles))
        candidates.sort(
            key=lambda item: (
                item[0]["candidate_rank_sha256"],
                item[0]["content_group_id"],
            )
        )
        eligible[source] = candidates
        diagnostics[source] = {
            "n_source_documents": len(documents),
            "n_eligible_after_n4_n5_embargo": len(candidates),
            "source_cap": SOURCE_CAP,
            "skipped": dict(sorted(skipped.items())),
        }

    all_candidates = [
        item for source in PILE_SOURCES for item in eligible[source]
    ]
    all_candidates.sort(
        key=lambda item: (
            item[0]["candidate_rank_sha256"],
            item[0]["source"],
            item[0]["content_group_id"],
        )
    )
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    accepted_n6_shingles: set[bytes] = set()
    for row, shingles in all_candidates:
        source = str(row["source"])
        if selected_counts[source] >= SOURCE_CAP:
            diagnostics[source]["skipped"]["source_cap_reached"] = (
                diagnostics[source]["skipped"].get("source_cap_reached", 0) + 1
            )
            continue
        if shingles & accepted_n6_shingles:
            diagnostics[source]["skipped"]["n6_internal_prefix_shingle_overlap"] = (
                diagnostics[source]["skipped"].get(
                    "n6_internal_prefix_shingle_overlap", 0
                )
                + 1
            )
            continue
        selected.append(row)
        selected_counts[source] += 1
        accepted_n6_shingles.update(shingles)
    if len(selected) < MIN_PROVISIONAL_TOTAL:
        availability = {
            source: {
                "selected": selected_counts[source],
                "eligible_before_n6_internal_shingles": len(eligible[source]),
                "cap": SOURCE_CAP,
            }
            for source in PILE_SOURCES
        }
        raise SystemExit(
            f"N6 provisional total {len(selected)} is below frozen minimum "
            f"{MIN_PROVISIONAL_TOTAL}; no plan was written. "
            + json.dumps(availability, ensure_ascii=False, sort_keys=True)
        )
    for index, row in enumerate(selected):
        row["provisional_index"] = index
        row["doc_id"] = index
        row["row_uid"] = shahex(
            SELECTION_SEED,
            "n6-provisional-row",
            row["source"],
            row["content_group_id"],
            row["position"],
        )
        row["split"] = "provisional"
    expected = len(selected)
    for field in ("row_uid", "content_group_id", "doc_id", "orig_index"):
        if len({str(row[field]) for row in selected}) != expected:
            raise ValueError(f"provisional rows repeat {field}")
    source_counts = Counter(str(row["source"]) for row in selected)
    if any(count > SOURCE_CAP for count in source_counts.values()):
        raise ValueError("provisional source cap was exceeded")
    for source in PILE_SOURCES:
        diagnostics[source]["selected_after_global_n6_shingle_filter"] = (
            source_counts[source]
        )

    plan = {
        "schema_version": 1,
        "experiment": "N6+ Pile-only provisional cohort",
        "status": "frozen_before_n6_model_output",
        "preregistration_sha256": prereg_sha,
        "selection_script_sha256": sha256_file(__file__),
        "selection_seed": SELECTION_SEED,
        "inputs": {
            "n3_corpus": str(args.corpus),
            "n3_corpus_sha256": corpus_sha,
            "n3_corpus_manifest": str(args.corpus_manifest),
            "n3_corpus_manifest_sha256": manifest_sha,
            "pile_parquet": str(args.pile_parquet),
            "pile_parquet_sha256": pile_sha,
            "n4_activations": str(args.n4_activations),
            "n4_activations_sha256": n4_sha,
            "n5_plan": str(args.n5_plan),
            "n5_plan_sha256": n5_plan_sha,
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": model_manifest_sha,
            "code_manifest": str(args.code_manifest),
            "code_manifest_sha256": code_manifest_sha,
            "base_model_tokenizer_root": str(args.base_model),
            "tokenizer_file_sha256": tokenizer_hashes,
        },
        "protocol": {
            "pile_sources": list(PILE_SOURCES),
            "source_cap": SOURCE_CAP,
            "source_cap_may_be_unfilled": True,
            "minimum_provisional_total": MIN_PROVISIONAL_TOTAL,
            "provisional_total": expected,
            "quota_status": "corrected-pre-freeze-audit-contract",
            "one_position_per_content_group": True,
            "seq_len": SEQ_LEN,
            "min_position": MIN_POSITION,
            "max_position": MAX_POSITION,
            "min_continuation": MIN_CONTINUATION,
            "continuation_content_fraction": CONTENT_FRAC,
            "position_order": (
                "SHA256(selection_seed, 'position', source, "
                "content_group_id, position)"
            ),
            "candidate_order": (
                "SHA256(selection_seed, 'candidate', source, "
                "content_group_id, position)"
            ),
            "shingle_words": SHINGLE_WORDS,
            "shingle_normalization": "Unicode NFKC, lower, Python Unicode \\w+",
            "n6_internal_shingle_filter": True,
            "global_candidate_order_then_source_cap": True,
        },
        "embargo": {
            "n4": n4_report,
            "n5": n5_report,
            "combined_prefix_shingle_count": len(embargo_shingles),
            "combined_prefix_shingle_union_sha256": shingle_digest(
                embargo_shingles
            ),
        },
        "availability": diagnostics,
        "checks": {
            "n_rows": expected,
            "n_content_groups": expected,
            "n_row_uids": expected,
            "n_orig_indices": expected,
            "source_counts": dict(sorted(source_counts.items())),
            "n4_n5_document_content_overlap": 0,
            "n4_n5_prefix_shingle_overlap": 0,
            "n6_internal_prefix_shingle_overlap": 0,
            "n_selected_prefix_shingles": len(accepted_n6_shingles),
            "selected_prefix_shingle_union_sha256": shingle_digest(
                accepted_n6_shingles
            ),
            "all_input_ids_frozen": True,
            "template_or_blank_token_count": sum(
                not content(int(row["token_id"])) for row in selected
            ),
            "row_uid_sequence_sha256": canonical_sha256(
                [row["row_uid"] for row in selected]
            ),
        },
        "rows": selected,
    }
    if plan["checks"]["template_or_blank_token_count"]:
        raise ValueError("selected cohort contains a non-content target token")
    output_sha = write_new_json(args.out, plan)
    print(
        f"N6_PROVISIONAL_PLAN_FROZEN rows={expected} "
        f"source_cap={SOURCE_CAP} seed={SELECTION_SEED} sha256={output_sha}"
    )


if __name__ == "__main__":
    main()
