#!/usr/bin/env python3
"""Freeze the N5 discovery/held-out position plan without loading a base model.

This is deliberately a tokenizer-only stage.  It verifies every preregistered
source hash, reconstructs the N4 content embargo, selects one eligible position
per independent content group, and freezes the complete input_ids used by all
later GPU stages.

The output is a single immutable JSON plan.  Discovery and held-out extraction
must consume that plan; they may not resample or retokenize documents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
from transformers import AutoTokenizer


EXPECTED_N3_CORPUS_SHA256 = (
    "d40069ab51c294ecbe3e76845d1f2f4dff1bb66a6061c5b6b4c612f7d0ff8816"
)
EXPECTED_PILE_PARQUET_SHA256 = (
    "a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31"
)
EXPECTED_XNLI_PARQUET_SHA256 = (
    "c5e6263b0872a3914c9bc165bfe3883e433aa2066c3fa3b9d142829a9b122518"
)
EXPECTED_N4_ACTIVATIONS_SHA256 = (
    "eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66"
)
EXPECTED_PREREG_SHA256 = (
    "110c952805e5c8d469815c9f60a6fcf4520537452c60c6305e1ea699bd3b82b0"
)

PILE_REPO = "NeelNanda/pile-10k"
PILE_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"
XNLI_REPO = "facebook/xnli"
XNLI_FILE = "all_languages/validation-00000-of-00001.parquet"

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
XNLI_LANGS = ("en", "es", "zh", "de", "fr", "ru", "ar", "hi", "tr", "vi")
SCRIPT_OF = {
    "en": "Latn",
    "es": "Latn",
    "de": "Latn",
    "fr": "Latn",
    "tr": "Latn",
    "vi": "Latn",
    "zh": "Hans",
    "ru": "Cyrl",
    "ar": "Arab",
    "hi": "Deva",
}

SEED = 20260730
SEQ_LEN = 512
MIN_POSITION = 64
MAX_POSITION = 480
MIN_CONTINUATION = 16
CONTENT_FRAC = 0.75
SHINGLE_WORDS = 20
XNLI_SENTENCES_PER_PASSAGE = 8
XNLI_FIRST_PASSAGE = 124
XNLI_LAST_PASSAGE = 273

WORD_RE = re.compile(r"\w", re.UNICODE)
WORDS_RE = re.compile(r"\w+", re.UNICODE)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def shahex(*parts: Any) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")


def write_frozen(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"frozen plan already exists with different bytes: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != sidecar_payload:
        raise SystemExit(f"hash sidecar differs: {sidecar}")
    if not sidecar.exists():
        sidecar.write_text(sidecar_payload, encoding="utf-8")
    return digest


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise SystemExit(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}: {path}"
        )
    return observed


def resolve_dataset_file(
    explicit: Path | None, repo: str, filename: str, expected: str, label: str
) -> Path:
    if explicit is None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo, filename, repo_type="dataset"))
    else:
        path = explicit
    require_hash(path, expected, label)
    return path


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
        raise SystemExit(f"no tokenizer/config identity files found under {model_dir}")
    return {path.name: sha256_file(path) for path in sorted(files)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return WORDS_RE.findall(normalized)


def word_shingles(words: list[str]) -> set[bytes]:
    if len(words) < SHINGLE_WORDS:
        return set()
    result = set()
    for index in range(len(words) - SHINGLE_WORDS + 1):
        phrase = "\x1f".join(words[index : index + SHINGLE_WORDS])
        result.add(hashlib.sha256(phrase.encode("utf-8")).digest())
    return result


def digest_shingle_set(shingles: set[bytes]) -> str:
    digest = hashlib.sha256()
    for item in sorted(shingles):
        digest.update(item)
    return digest.hexdigest()


def decoded_prefix(tokenizer, input_ids: list[int]) -> str:
    return tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def make_content_predicate(tokenizer) -> Callable[[int], bool]:
    special = set(tokenizer.all_special_ids)
    cache: dict[int, bool] = {}

    def content(token_id: int) -> bool:
        token_id = int(token_id)
        if token_id in cache:
            return cache[token_id]
        if token_id in special:
            result = False
        else:
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            stripped = text.strip()
            result = bool(
                stripped
                and not (stripped.startswith("<") and stripped.endswith(">"))
                and WORD_RE.search(text)
            )
        cache[token_id] = result
        return result

    return content


def eligible_positions(
    input_ids: list[int], content: Callable[[int], bool]
) -> list[int]:
    high = min(MAX_POSITION, len(input_ids) - MIN_CONTINUATION - 1)
    if high < MIN_POSITION:
        return []
    output = []
    for position in range(MIN_POSITION, high + 1):
        if not content(input_ids[position]):
            continue
        continuation = input_ids[
            position + 1 : position + 1 + MIN_CONTINUATION
        ]
        if len(continuation) != MIN_CONTINUATION:
            continue
        fraction = sum(content(token) for token in continuation) / len(continuation)
        if fraction >= CONTENT_FRAC:
            output.append(position)
    return output


def build_candidate(
    *,
    tokenizer,
    content: Callable[[int], bool],
    text: str,
    corpus: str,
    source: str,
    lang: str,
    content_group_id: str,
    orig_index: int,
    passage_id: int,
) -> tuple[dict[str, Any], set[bytes]] | None:
    # Match N3/N4 exactly: add the model's ordinary raw-text special tokens and
    # then take a literal prefix.  Tokenizer-side truncation can have different
    # end-token behavior for some tokenizer implementations.
    input_ids = [
        int(token)
        for token in tokenizer(text, add_special_tokens=True)["input_ids"][:SEQ_LEN]
    ]
    positions = eligible_positions(input_ids, content)
    if not positions:
        return None
    position = min(
        positions,
        key=lambda value: shahex(
            SEED, "position", source, content_group_id, value
        ),
    )
    prefix = decoded_prefix(tokenizer, input_ids)
    words = normalized_words(prefix)
    shingles = word_shingles(words)
    token_id = int(input_ids[position])
    row = {
        "content_group_id": content_group_id,
        "corpus": corpus,
        "source": source,
        "lang": lang,
        "orig_index": int(orig_index),
        "passage_id": int(passage_id),
        "position": int(position),
        "position_rank_sha256": shahex(
            SEED, "position", source, content_group_id, position
        ),
        "candidate_rank_sha256": shahex(
            SEED, "candidate", source, content_group_id, position
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
        "prefix_shingle_set_sha256": digest_shingle_set(shingles),
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


def equal_sha_quotas(total: int, split: str) -> tuple[dict[str, int], list[str]]:
    base, remainder = divmod(total, len(PILE_SOURCES))
    order = sorted(
        PILE_SOURCES,
        key=lambda source: shahex(SEED, "quota", split, source),
    )
    quotas = {
        source: base + int(source in set(order[:remainder]))
        for source in PILE_SOURCES
    }
    return quotas, order


def load_n4_embargo(
    *,
    n4_path: Path,
    corpus_by_id: dict[int, dict[str, Any]],
    tokenizer,
) -> tuple[set[int], set[int], set[bytes], dict[str, Any]]:
    table = pq.read_table(n4_path, columns=["doc_id", "input_ids"])
    doc_ids = [int(value) for value in table.column("doc_id").to_pylist()]
    sequences = table.column("input_ids").to_pylist()
    if len(doc_ids) != 200:
        raise SystemExit(f"N4 embargo expected 200 rows, observed {len(doc_ids)}")

    by_doc: dict[int, list[int]] = {}
    for doc_id, sequence in zip(doc_ids, sequences):
        sequence = [int(token) for token in sequence]
        if doc_id in by_doc and by_doc[doc_id] != sequence:
            raise SystemExit(f"N4 doc_id {doc_id} has inconsistent input_ids")
        by_doc[doc_id] = sequence
    if len(by_doc) != 101:
        raise SystemExit(f"N4 embargo expected 101 doc_ids, observed {len(by_doc)}")
    missing = sorted(set(by_doc) - set(corpus_by_id))
    if missing:
        raise SystemExit(f"N4 doc_ids missing from frozen N3 corpus: {missing[:20]}")

    pile_doc_ids = {
        doc_id for doc_id in by_doc if corpus_by_id[doc_id]["corpus"] == "pile"
    }
    xnli_passages = {
        int(corpus_by_id[doc_id]["passage_id"])
        for doc_id in by_doc
        if corpus_by_id[doc_id]["corpus"] == "xnli"
    }
    if len(pile_doc_ids) != 73 or len(by_doc) - len(pile_doc_ids) != 28:
        raise SystemExit(
            "N4 corpus composition drift: expected 73 Pile and 28 XNLI doc_ids"
        )
    if len(xnli_passages) != 27:
        raise SystemExit(
            f"N4 expected 27 independent XNLI passages, observed {len(xnli_passages)}"
        )

    prefix_shingles: set[bytes] = set()
    per_doc_counts = {}
    for doc_id, input_ids in by_doc.items():
        words = normalized_words(decoded_prefix(tokenizer, input_ids))
        shingles = word_shingles(words)
        prefix_shingles.update(shingles)
        per_doc_counts[str(doc_id)] = len(shingles)

    diagnostics = {
        "n_rows": len(doc_ids),
        "n_doc_ids": len(by_doc),
        "n_independent_content_groups": len(pile_doc_ids) + len(xnli_passages),
        "n_pile_doc_ids": len(pile_doc_ids),
        "n_xnli_doc_ids": len(by_doc) - len(pile_doc_ids),
        "n_xnli_passage_ids": len(xnli_passages),
        "xnli_passage_ids": sorted(xnli_passages),
        "n_unique_prefix_shingles": len(prefix_shingles),
        "prefix_shingle_union_sha256": digest_shingle_set(prefix_shingles),
        "per_doc_prefix_shingle_counts": per_doc_counts,
    }
    return pile_doc_ids, xnli_passages, prefix_shingles, diagnostics


def match_xnli_slots(
    candidates: dict[int, dict[str, tuple[dict[str, Any], set[bytes]]]]
) -> dict[int, tuple[str, str, int]]:
    """Deterministic perfect matching from passage IDs to split/language slots."""
    slots: list[tuple[str, str, int]] = []
    for language in XNLI_LANGS:
        slots.extend(("discovery", language, index) for index in range(5))
        slots.extend(("heldout", language, index) for index in range(10))
    if len(slots) != 150:
        raise AssertionError("XNLI slot construction drift")

    adjacency: dict[int, list[tuple[str, str, int]]] = {}
    for passage_id, by_language in candidates.items():
        edges = [slot for slot in slots if slot[1] in by_language]
        adjacency[passage_id] = sorted(
            edges,
            key=lambda slot: shahex(
                SEED,
                "xnli-edge",
                by_language[slot[1]][0]["candidate_rank_sha256"],
                slot[0],
                slot[1],
                slot[2],
            ),
        )

    matched_slot: dict[tuple[str, str, int], int] = {}

    def augment(passage_id: int, seen: set[tuple[str, str, int]]) -> bool:
        for slot in adjacency[passage_id]:
            if slot in seen:
                continue
            seen.add(slot)
            previous = matched_slot.get(slot)
            if previous is None or augment(previous, seen):
                matched_slot[slot] = passage_id
                return True
        return False

    passage_order = sorted(
        candidates, key=lambda passage: shahex(SEED, "xnli-passage", passage)
    )
    for passage_id in passage_order:
        if not augment(passage_id, set()):
            raise SystemExit(
                "XNLI eligibility cannot satisfy the frozen split/language quotas; "
                f"first unmatched passage_id={passage_id}"
            )
    if len(matched_slot) != 150:
        raise SystemExit(f"XNLI matching incomplete: {len(matched_slot)}/150 slots")
    passage_to_slot = {
        passage_id: slot for slot, passage_id in matched_slot.items()
    }
    if len(passage_to_slot) != 150:
        raise SystemExit("XNLI matching assigned a passage more than once")
    return passage_to_slot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "--base-model",
        type=Path,
        default=Path("/root/autodl-tmp/models/gemma-3-12b-it"),
    )
    parser.add_argument("--pile-parquet", type=Path)
    parser.add_argument("--xnli-parquet", type=Path)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path(
            "/root/autodl-tmp/results/n5_selective_hybrid_preregistration_v1.md"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/root/autodl-tmp/results/n5_cohort_plan_v1.json"),
    )
    args = parser.parse_args()

    require_hash(args.corpus, EXPECTED_N3_CORPUS_SHA256, "N3 corpus JSONL")
    require_hash(
        args.n4_activations, EXPECTED_N4_ACTIVATIONS_SHA256, "N4 activation cohort"
    )
    prereg_sha256 = require_hash(
        args.prereg, EXPECTED_PREREG_SHA256, "N5 preregistration"
    )
    pile_path = resolve_dataset_file(
        args.pile_parquet,
        PILE_REPO,
        PILE_FILE,
        EXPECTED_PILE_PARQUET_SHA256,
        "Pile parquet",
    )
    xnli_path = resolve_dataset_file(
        args.xnli_parquet,
        XNLI_REPO,
        XNLI_FILE,
        EXPECTED_XNLI_PARQUET_SHA256,
        "XNLI validation parquet",
    )

    corpus_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("outputs", {}).get("jsonl_sha256")
        != EXPECTED_N3_CORPUS_SHA256
    ):
        raise SystemExit("N3 corpus manifest does not bind the frozen JSONL")
    provenance = corpus_manifest.get("provenance", {})
    if (
        provenance.get("pile", {}).get("local_sha256")
        != EXPECTED_PILE_PARQUET_SHA256
        or provenance.get("xnli", {}).get("local_sha256")
        != EXPECTED_XNLI_PARQUET_SHA256
    ):
        raise SystemExit("N3 corpus manifest source provenance differs from prereg")

    corpus_rows = load_jsonl(args.corpus)
    if len(corpus_rows) != 10404:
        raise SystemExit(f"frozen N3 corpus expected 10404 rows, got {len(corpus_rows)}")
    corpus_by_id = {int(row["doc_id"]): row for row in corpus_rows}
    if len(corpus_by_id) != len(corpus_rows):
        raise SystemExit("N3 corpus doc_id values are not unique")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    tokenizer_hashes = tokenizer_file_hashes(args.base_model)
    content = make_content_predicate(tokenizer)

    (
        n4_pile_doc_ids,
        n4_xnli_passages,
        n4_prefix_shingles,
        n4_diagnostics,
    ) = load_n4_embargo(
        n4_path=args.n4_activations,
        corpus_by_id=corpus_by_id,
        tokenizer=tokenizer,
    )
    requested_xnli_passages = set(
        range(XNLI_FIRST_PASSAGE, XNLI_LAST_PASSAGE + 1)
    )
    if requested_xnli_passages & n4_xnli_passages:
        raise SystemExit("requested N5 XNLI passages overlap the N4 passage embargo")
    if len(requested_xnli_passages) != 150:
        raise AssertionError("N5 XNLI passage range must contain exactly 150 groups")

    discovery_quotas, discovery_quota_order = equal_sha_quotas(150, "discovery")
    heldout_quotas, heldout_quota_order = equal_sha_quotas(300, "heldout")
    if sorted(discovery_quotas.values()) != [11] * 6 + [12] * 7:
        raise AssertionError("discovery largest-remainder quota drift")
    if sorted(heldout_quotas.values()) != [23] * 12 + [24]:
        raise AssertionError("held-out largest-remainder quota drift")

    pile_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus_rows:
        if row["corpus"] == "pile" and row["source"] in PILE_SOURCES:
            pile_by_source[row["source"]].append(row)
    if set(pile_by_source) != set(PILE_SOURCES):
        raise SystemExit(
            f"Pile source set drift: missing={sorted(set(PILE_SOURCES)-set(pile_by_source))}"
        )

    selected: list[tuple[dict[str, Any], set[bytes]]] = []
    selected_group_ids: set[str] = set()
    accepted_shingles: set[bytes] = set()
    pile_diagnostics: dict[str, Any] = {}
    pile_candidates: list[tuple[dict[str, Any], set[bytes]]] = []
    pile_skipped: dict[str, Counter] = {}
    source_selection_order = sorted(
        PILE_SOURCES, key=lambda source: shahex(SEED, "source-selection", source)
    )

    for source in source_selection_order:
        skipped = Counter()
        pile_skipped[source] = skipped
        source_docs = sorted(
            pile_by_source[source],
            key=lambda row: shahex(
                SEED,
                "pile-document",
                source,
                row["orig_index"],
                row["text_sha256"],
            ),
        )
        for document in source_docs:
            doc_id = int(document["doc_id"])
            if doc_id in n4_pile_doc_ids:
                skipped["n4_document_embargo"] += 1
                continue
            content_group_id = (
                f"pile:{EXPECTED_PILE_PARQUET_SHA256}:{int(document['orig_index'])}"
            )
            built = build_candidate(
                tokenizer=tokenizer,
                content=content,
                text=document["text"],
                corpus="pile",
                source=source,
                lang=document["lang"],
                content_group_id=content_group_id,
                orig_index=int(document["orig_index"]),
                passage_id=-1,
            )
            if built is None:
                skipped["position_ineligible"] += 1
                continue
            row, shingles = built
            if shingles & n4_prefix_shingles:
                skipped["n4_prefix_shingle_embargo"] += 1
                continue
            pile_candidates.append((row, shingles))
        pile_diagnostics[source] = {
            "n_source_documents": len(source_docs),
            "n_eligible_after_n4_embargo": sum(
                item[0]["source"] == source for item in pile_candidates
            ),
            "selected": {},
            "skipped": dict(skipped),
        }

    # Quotas are per source, but duplicate conflicts can cross sources.  Resolve
    # them in one global candidate order, exactly the SHA order named in the
    # preregistration, rather than allowing source iteration order to decide.
    pile_candidates.sort(key=lambda item: item[0]["candidate_rank_sha256"])
    selected_counts: Counter = Counter()
    for split, quotas in (
        ("discovery", discovery_quotas),
        ("heldout", heldout_quotas),
    ):
        for row, shingles in pile_candidates:
            source = row["source"]
            if selected_counts[(split, source)] >= quotas[source]:
                continue
            group = row["content_group_id"]
            if group in selected_group_ids:
                continue
            if shingles & accepted_shingles:
                pile_skipped[source]["n5_internal_shingle_conflict"] += 1
                continue
            selected_row = dict(row)
            selected_row["split"] = split
            selected.append((selected_row, shingles))
            selected_group_ids.add(group)
            accepted_shingles.update(shingles)
            selected_counts[(split, source)] += 1
        shortfalls = {
            source: quotas[source] - selected_counts[(split, source)]
            for source in PILE_SOURCES
            if selected_counts[(split, source)] != quotas[source]
        }
        if shortfalls:
            raise SystemExit(
                f"Pile cannot fill frozen {split} quotas after global SHA/shingle "
                f"selection: shortfalls={shortfalls}"
            )
    for source in PILE_SOURCES:
        pile_diagnostics[source]["selected"] = {
            "discovery": selected_counts[("discovery", source)],
            "heldout": selected_counts[("heldout", source)],
        }
        pile_diagnostics[source]["skipped"] = dict(pile_skipped[source])

    # The unused XNLI validation passages are read from the original frozen file.
    premise_table = pq.read_table(xnli_path, columns=["premise"])
    premises = premise_table.column("premise").to_pylist()
    if len(premises) != 2490:
        raise SystemExit(f"XNLI validation expected 2490 premises, got {len(premises)}")
    if not premises or any(language not in premises[0] for language in XNLI_LANGS):
        raise SystemExit("XNLI premise struct lacks one or more frozen languages")

    xnli_candidates: dict[
        int, dict[str, tuple[dict[str, Any], set[bytes]]]
    ] = {}
    xnli_skipped = Counter()
    for passage_id in range(XNLI_FIRST_PASSAGE, XNLI_LAST_PASSAGE + 1):
        start = passage_id * XNLI_SENTENCES_PER_PASSAGE
        stop = start + XNLI_SENTENCES_PER_PASSAGE
        by_language: dict[str, tuple[dict[str, Any], set[bytes]]] = {}
        for language in XNLI_LANGS:
            sentences = [premises[index][language] for index in range(start, stop)]
            if any(not isinstance(sentence, str) or not sentence for sentence in sentences):
                raise SystemExit(
                    f"XNLI passage {passage_id}/{language} contains a missing sentence"
                )
            text = " ".join(sentences)
            group = (
                f"xnli:{EXPECTED_XNLI_PARQUET_SHA256}:validation:"
                f"passage:{passage_id}"
            )
            built = build_candidate(
                tokenizer=tokenizer,
                content=content,
                text=text,
                corpus="xnli",
                source=f"xnli:{language}",
                lang=language,
                content_group_id=group,
                orig_index=passage_id,
                passage_id=passage_id,
            )
            if built is None:
                xnli_skipped[f"ineligible:{language}"] += 1
                continue
            row, shingles = built
            if shingles & n4_prefix_shingles:
                xnli_skipped[f"n4_shingle:{language}"] += 1
                continue
            if shingles & accepted_shingles:
                xnli_skipped[f"pile_shingle:{language}"] += 1
                continue
            by_language[language] = (row, shingles)
        if not by_language:
            raise SystemExit(
                f"XNLI passage_id={passage_id} has no eligible, nonembargoed language"
            )
        xnli_candidates[passage_id] = by_language

    passage_to_slot = match_xnli_slots(xnli_candidates)
    xnli_selected: list[tuple[dict[str, Any], set[bytes]]] = []
    for passage_id, slot in passage_to_slot.items():
        split, language, slot_index = slot
        row, shingles = xnli_candidates[passage_id][language]
        selected_row = dict(row)
        selected_row["split"] = split
        selected_row["xnli_slot_index"] = int(slot_index)
        xnli_selected.append((selected_row, shingles))

    # No silent repair is allowed if the deterministic matching happens to
    # select translation text that duplicates another selected prefix.
    for row, shingles in sorted(
        xnli_selected, key=lambda item: item[0]["candidate_rank_sha256"]
    ):
        if row["content_group_id"] in selected_group_ids:
            raise SystemExit(f"duplicate N5 content group: {row['content_group_id']}")
        if shingles & accepted_shingles:
            raise SystemExit(
                "deterministic XNLI assignment creates an N5 internal 20-word "
                f"shingle conflict: passage_id={row['passage_id']} lang={row['lang']}"
            )
        selected.append((row, shingles))
        selected_group_ids.add(row["content_group_id"])
        accepted_shingles.update(shingles)

    # Stable final row ordering and UIDs.  The integer doc_id is plan-local and
    # never used as the scientific grouping key.
    selected.sort(
        key=lambda item: (
            0 if item[0]["split"] == "discovery" else 1,
            item[0]["candidate_rank_sha256"],
        )
    )
    rows: list[dict[str, Any]] = []
    all_row_shingles: list[set[bytes]] = []
    for doc_id, (row, shingles) in enumerate(selected):
        frozen = dict(row)
        frozen["doc_id"] = int(doc_id)
        frozen["row_uid"] = shahex(
            SEED,
            "row",
            frozen["split"],
            frozen["source"],
            frozen["content_group_id"],
            frozen["position"],
        )
        rows.append(frozen)
        all_row_shingles.append(shingles)

    split_corpus = Counter((row["split"], row["corpus"]) for row in rows)
    expected_split_corpus = {
        ("discovery", "pile"): 150,
        ("discovery", "xnli"): 50,
        ("heldout", "pile"): 300,
        ("heldout", "xnli"): 100,
    }
    if dict(split_corpus) != expected_split_corpus:
        raise SystemExit(
            f"frozen split/corpus count mismatch: {dict(split_corpus)}"
        )
    if len(rows) != 600 or len(selected_group_ids) != 600:
        raise SystemExit(
            f"N5 expected 600 rows/groups, got {len(rows)}/{len(selected_group_ids)}"
        )
    if len({row["row_uid"] for row in rows}) != 600:
        raise SystemExit("N5 row_uid collision")
    if len({row["doc_id"] for row in rows}) != 600:
        raise SystemExit("N5 plan-local doc_id collision")

    pile_counts = Counter(
        (row["split"], row["source"]) for row in rows if row["corpus"] == "pile"
    )
    for source in PILE_SOURCES:
        if pile_counts[("discovery", source)] != discovery_quotas[source]:
            raise SystemExit(f"discovery Pile quota mismatch for {source}")
        if pile_counts[("heldout", source)] != heldout_quotas[source]:
            raise SystemExit(f"held-out Pile quota mismatch for {source}")
    xnli_counts = Counter(
        (row["split"], row["lang"]) for row in rows if row["corpus"] == "xnli"
    )
    for language in XNLI_LANGS:
        if xnli_counts[("discovery", language)] != 5:
            raise SystemExit(f"XNLI discovery quota mismatch for {language}")
        if xnli_counts[("heldout", language)] != 10:
            raise SystemExit(f"XNLI held-out quota mismatch for {language}")

    selected_passages = [
        row["passage_id"] for row in rows if row["corpus"] == "xnli"
    ]
    if set(selected_passages) != requested_xnli_passages:
        raise SystemExit("N5 XNLI selected passage range drift")
    if len(selected_passages) != len(set(selected_passages)):
        raise SystemExit("an XNLI passage was selected in more than one language")

    # Recompute pairwise shingle disjointness without trusting greedy state.
    seen_shingles: set[bytes] = set()
    for row, shingles in zip(rows, all_row_shingles):
        overlap = shingles & seen_shingles
        if overlap:
            raise SystemExit(
                f"final N5 internal shingle QA failed at {row['content_group_id']}"
            )
        if shingles & n4_prefix_shingles:
            raise SystemExit(
                f"final N4 prefix embargo QA failed at {row['content_group_id']}"
            )
        seen_shingles.update(shingles)

    positions = [int(row["position"]) for row in rows]
    plan = {
        "schema_version": 1,
        "experiment": "N5 selective hybrid frozen content-token cohort",
        "status": "frozen_before_base_model_load",
        "preregistration_sha256": prereg_sha256,
        "selection_script_sha256": sha256_file(__file__),
        "seed": SEED,
        "inputs": {
            "n3_corpus": str(args.corpus),
            "n3_corpus_sha256": EXPECTED_N3_CORPUS_SHA256,
            "n3_corpus_manifest": str(args.corpus_manifest),
            "n3_corpus_manifest_sha256": sha256_file(args.corpus_manifest),
            "pile_parquet": str(pile_path),
            "pile_parquet_sha256": EXPECTED_PILE_PARQUET_SHA256,
            "xnli_validation_parquet": str(xnli_path),
            "xnli_validation_parquet_sha256": EXPECTED_XNLI_PARQUET_SHA256,
            "n4_activations": str(args.n4_activations),
            "n4_activations_sha256": EXPECTED_N4_ACTIVATIONS_SHA256,
            "base_model_tokenizer_root": str(args.base_model),
            "tokenizer_file_sha256": tokenizer_hashes,
        },
        "protocol": {
            "one_position_per_content_group": True,
            "seq_len": SEQ_LEN,
            "min_position": MIN_POSITION,
            "max_position": MAX_POSITION,
            "min_continuation": MIN_CONTINUATION,
            "continuation_content_fraction": CONTENT_FRAC,
            "unicode_normalization": "NFKC",
            "case_normalization": "lower",
            "word_regex": WORDS_RE.pattern,
            "shingle_words": SHINGLE_WORDS,
            "shingle_rule": (
                "no selected prefix shares any normalized contiguous 20-word "
                "shingle with an N4 prefix or another selected N5 prefix"
            ),
            "pile_sources": list(PILE_SOURCES),
            "pile_source_selection_order": source_selection_order,
            "pile_discovery_quotas": discovery_quotas,
            "pile_heldout_quotas": heldout_quotas,
            "pile_discovery_quota_sha_order": discovery_quota_order,
            "pile_heldout_quota_sha_order": heldout_quota_order,
            "xnli_languages": list(XNLI_LANGS),
            "xnli_passage_range_inclusive": [
                XNLI_FIRST_PASSAGE,
                XNLI_LAST_PASSAGE,
            ],
            "xnli_sentences_per_passage": XNLI_SENTENCES_PER_PASSAGE,
            "xnli_quota_per_language": {"discovery": 5, "heldout": 10},
            "candidate_order": (
                "SHA256(seed, 'candidate', source, content_group_id, position)"
            ),
            "position_order": (
                "SHA256(seed, 'position', source, content_group_id, position)"
            ),
        },
        "embargo": n4_diagnostics,
        "diagnostics": {
            "pile": pile_diagnostics,
            "xnli_skipped": dict(xnli_skipped),
            "n_selected_prefix_shingles": len(seen_shingles),
            "selected_prefix_shingle_union_sha256": digest_shingle_set(
                seen_shingles
            ),
        },
        "checks": {
            "n_rows": len(rows),
            "n_content_groups": len(selected_group_ids),
            "n_row_uids": len({row["row_uid"] for row in rows}),
            "n_plan_doc_ids": len({row["doc_id"] for row in rows}),
            "split_corpus_counts": {
                f"{split}:{corpus}": count
                for (split, corpus), count in sorted(split_corpus.items())
            },
            "pile_source_counts": {
                f"{split}:{source}": count
                for (split, source), count in sorted(pile_counts.items())
            },
            "xnli_language_counts": {
                f"{split}:{language}": count
                for (split, language), count in sorted(xnli_counts.items())
            },
            "xnli_passages_unique": len(set(selected_passages)) == 150,
            "xnli_passage_embargo_disjoint": not (
                set(selected_passages) & n4_xnli_passages
            ),
            "n4_prefix_shingle_overlap_count": 0,
            "n5_internal_shingle_overlap_count": 0,
            "template_or_blank_token_count": sum(
                not content(row["token_id"]) for row in rows
            ),
            "min_position_realized": min(positions),
            "max_position_realized": max(positions),
            "all_input_ids_frozen": all(bool(row["input_ids"]) for row in rows),
        },
        "rows": rows,
    }
    if plan["checks"]["template_or_blank_token_count"] != 0:
        raise SystemExit("final plan contains a non-content target token")
    digest = write_frozen(args.out, plan)
    print("N5_COHORT_PLAN_FROZEN")
    print(
        f"rows=600 discovery=200 heldout=400 groups=600 "
        f"positions=[{min(positions)},{max(positions)}]"
    )
    print(f"plan_sha256={digest}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
