#!/usr/bin/env python3
"""Diagnose the tokenizer-only XNLI passage-151/vi shingle collision."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


PILE = Path(
    "/root/autodl-tmp/hf/hub/datasets--NeelNanda--pile-10k/blobs/"
    "a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31"
)
XNLI = Path(
    "/root/autodl-tmp/hf/hub/datasets--facebook--xnli/blobs/"
    "c5e6263b0872a3914c9bc165bfe3883e433aa2066c3fa3b9d142829a9b122518"
)
MODEL = Path("/root/autodl-tmp/models/gemma-3-12b-it")
LANGS = ("en", "es", "zh", "de", "fr", "ru", "ar", "hi", "tr", "vi")
WORDS = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    return WORDS.findall(unicodedata.normalize("NFKC", text).lower())


def shingles(text: str) -> dict[tuple[str, ...], int]:
    values = words(text)
    return {
        tuple(values[index : index + 20]): index
        for index in range(max(0, len(values) - 19))
    }


def main() -> None:
    premises = pq.read_table(XNLI, columns=["premise"])["premise"].to_pylist()
    start = 151 * 8
    target_text = " ".join(premises[index]["vi"] for index in range(start, start + 8))
    target = shingles(target_text)
    print(f"target_shingles={len(target)}")

    for passage_id in range(124, 274):
        for language in LANGS:
            if passage_id == 151 and language == "vi":
                continue
            begin = passage_id * 8
            text = " ".join(
                premises[index][language] for index in range(begin, begin + 8)
            )
            overlap = set(target) & set(shingles(text))
            if overlap:
                phrase = " ".join(sorted(overlap)[0])
                print(
                    "XNLI_MATCH "
                    f"passage_id={passage_id} lang={language} "
                    f"n={len(overlap)} phrase={phrase!r}"
                )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    table = pq.read_table(PILE, columns=["text", "meta"])
    texts = table["text"].to_pylist()
    metas = table["meta"].to_pylist()
    for orig_index, (text, meta) in enumerate(zip(texts, metas)):
        input_ids = tokenizer(text, add_special_tokens=True)["input_ids"][:512]
        prefix = tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        overlap = set(target) & set(shingles(prefix))
        if overlap:
            phrase = " ".join(sorted(overlap)[0])
            print(
                "PILE_MATCH "
                f"orig_index={orig_index} source={meta.get('pile_set_name')} "
                f"n={len(overlap)} phrase={phrase!r}"
            )


if __name__ == "__main__":
    main()
