#!/usr/bin/env python3
"""N3 step 1 — build and FREEZE a real-text corpus base.

Why: every feature-level claim in this project so far rests on synthetic prompts
(24 hand-written documents for B6+B4, 5 short instruct prompts for E1-E7). That
caused two documented failures: the "held-out activation gate" was really a
"fires in 12 synthetic documents" gate (POSSBILITY F14.3), and the E1-E7 cohort
silently landed 13/40 positions on chat-template tokens (F13). Real text with
real source labels fixes the premise of both.

Two corpora, chosen for complementary reasons:

  pile   NeelNanda/pile-10k -- the standard interp smoke corpus. Ships a
         `meta.pile_set_name` label per document (Pile-CC, Github, ArXiv,
         StackExchange, PubMed, ...), which gives us REAL cross-source
         selectivity instead of our own axis labels.

  xnli   facebook/xnli `all_languages` validation -- 2490 premises, each present
         in all 15 languages. Because it is PARALLEL, content is held constant
         across languages, so a feature that fires only on one language cannot
         be explained away by topic. B6+B4's language features (88.9% "pass"
         rate, test AUC saturated at 1.0) never had this control.
         (First choice was openlanguagedata/flores_plus, but it is a gated repo
         and returns 403 on the mirror; XNLI is ungated and equally parallel.)

No `datasets` dependency: we hf_hub_download explicit paths and parse with
pyarrow / json, the same pattern that worked for the SAE weights.

Output is frozen before any model touches it: n3_corpus_v1.jsonl (documents)
plus n3_corpus_v1.json (manifest with per-file sha256 and per-source counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

PILE_REPO = "NeelNanda/pile-10k"
PILE_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"
XNLI_REPO = "facebook/xnli"
XNLI_FILE = "all_languages/validation-00000-of-00001.parquet"

# 10 of XNLI's 15 languages: the three B6+B4 used (en/es/zh) plus seven spanning
# five scripts, so "language feature" claims can be tested beyond Latin script.
XNLI_LANGS = ["en", "es", "zh", "de", "fr", "ru", "ar", "hi", "tr", "vi"]
SCRIPT_OF = {"en": "Latn", "es": "Latn", "de": "Latn", "fr": "Latn", "tr": "Latn",
             "vi": "Latn", "zh": "Hans", "ru": "Cyrl", "ar": "Arab", "hi": "Deva"}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pile(n_docs: int, min_chars: int) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(PILE_REPO, PILE_FILE, repo_type="dataset")
    table = pq.read_table(path)
    cols = table.column_names
    texts = table.column("text").to_pylist()
    metas = table.column("meta").to_pylist() if "meta" in cols else [None] * len(texts)

    docs: list[dict] = []
    for i, (text, meta) in enumerate(zip(texts, metas)):
        if len(docs) >= n_docs:
            break
        if not text or len(text) < min_chars:
            continue
        source = "unknown"
        if isinstance(meta, dict):
            source = meta.get("pile_set_name") or meta.get("set_name") or "unknown"
        docs.append({
            "corpus": "pile",
            "source": str(source),
            "lang": "eng_Latn" if source not in ("Github",) else "code",
            "orig_index": i,
            "text": text,
        })
    return docs, {"repo": PILE_REPO, "file": PILE_FILE, "local_sha256": sha256_file(path),
                  "columns": cols, "n_available": len(texts)}


def load_xnli(sents_per_passage: int, max_passages: int) -> tuple[list[dict], dict]:
    """Group consecutive parallel premises into passages, one set per language.

    Grouping matters: a single XNLI premise is ~20-40 tokens, too short for a
    layer-32 residual to carry much context. Grouping by a fixed premise index
    range keeps passages parallel ACROSS languages (passage k is the same content
    in every language), which is the whole point of using a parallel corpus.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(XNLI_REPO, XNLI_FILE, repo_type="dataset")
    premises = pq.read_table(path).column("premise").to_pylist()
    missing = [l for l in XNLI_LANGS if l not in (premises[0] or {})]
    if missing:
        raise SystemExit(f"XNLI premise struct lacks languages: {missing}")

    docs: list[dict] = []
    n_pass = min(max_passages, len(premises) // sents_per_passage)
    for lang in XNLI_LANGS:
        for k in range(n_pass):
            chunk = [premises[i][lang] for i in
                     range(k * sents_per_passage, (k + 1) * sents_per_passage)]
            docs.append({
                "corpus": "xnli",
                "source": f"xnli:{lang}",
                "lang": lang,
                "script": SCRIPT_OF[lang],
                "orig_index": k,
                "passage_id": k,           # same passage_id == same content, other language
                "sent_range": [k * sents_per_passage, (k + 1) * sents_per_passage],
                "text": " ".join(chunk),
            })
    return docs, {"repo": XNLI_REPO, "file": XNLI_FILE, "local_sha256": sha256_file(path),
                  "langs": XNLI_LANGS, "n_premises": len(premises),
                  "sents_per_passage": sents_per_passage, "passages_per_language": n_pass}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pile-docs", type=int, default=10000)
    ap.add_argument("--pile-min-chars", type=int, default=400)
    ap.add_argument("--xnli-sents-per-passage", type=int, default=8)
    ap.add_argument("--xnli-max-passages", type=int, default=124)
    ap.add_argument("--out-dir", default="/root/autodl-tmp/results")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] pile-10k ...", flush=True)
    pile_docs, pile_prov = load_pile(args.pile_docs, args.pile_min_chars)
    print(f"      {len(pile_docs)} docs, sources={Counter(d['source'] for d in pile_docs).most_common()}",
          flush=True)

    print("[2/3] xnli all_languages validation ...", flush=True)
    xnli_docs, xnli_prov = load_xnli(args.xnli_sents_per_passage, args.xnli_max_passages)
    print(f"      {len(xnli_docs)} passages over {len(XNLI_LANGS)} languages", flush=True)

    docs = pile_docs + xnli_docs
    for i, d in enumerate(docs):
        d["doc_id"] = i
        d["n_chars"] = len(d["text"])
        d["text_sha256"] = sha256_text(d["text"])

    jsonl_path = out_dir / f"n3_corpus_{args.tag}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Parallelism self-check: every xnli passage_id must appear once per language.
    per_pid = Counter(d["passage_id"] for d in xnli_docs)
    parallel_ok = bool(per_pid) and set(per_pid.values()) == {len(XNLI_LANGS)}

    manifest = {
        "schema_version": 1,
        "experiment": "N3 real-corpus base",
        "status": "corpus_frozen_before_any_model_forward",
        "purpose": [
            "replace the synthetic 'held-out activation gate' with real cross-source evidence (F14.3)",
            "provide a content-token cohort to replace the template-contaminated E1-E7 cohort (F13)",
            "supply covariates for the new main question 'what predicts readability' (F14.1)",
        ],
        "provenance": {"pile": pile_prov, "xnli": xnli_prov},
        "args": vars(args),
        "counts": {
            "n_docs": len(docs),
            "n_pile": len(pile_docs),
            "n_xnli": len(xnli_docs),
            "by_source": dict(Counter(d["source"] for d in docs)),
            "total_chars": sum(d["n_chars"] for d in docs),
        },
        "checks": {
            "xnli_parallel_complete": parallel_ok,
            "xnli_languages_per_passage": (max(per_pid.values()) if per_pid else 0),
            "unique_text_sha256": len({d["text_sha256"] for d in docs}) == len(docs),
        },
        "outputs": {"jsonl": str(jsonl_path), "jsonl_sha256": sha256_file(jsonl_path)},
    }
    man_path = out_dir / f"n3_corpus_{args.tag}.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[3/3] frozen.", flush=True)
    print(f"      docs        : {len(docs)} ({len(pile_docs)} pile + {len(xnli_docs)} xnli)")
    print(f"      chars       : {manifest['counts']['total_chars']:,}")
    print(f"      parallel ok : {parallel_ok}")
    print(f"      unique docs : {manifest['checks']['unique_text_sha256']}")
    print(f"      -> {jsonl_path}")
    print(f"      -> {man_path}")


if __name__ == "__main__":
    main()
