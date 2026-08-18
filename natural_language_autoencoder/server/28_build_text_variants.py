#!/usr/bin/env python3
"""N1 stage 1 (local, no GPU): build and FREEZE the C7/B3 text variants.

Decomposes the NLA explanation channel into separable components so that AR
rescoring can attribute the reconstruction score to each of them:

  orig             the original AV explanation (control; must reproduce 0.8593)
  para_tp          third-party (non-Gemma) semantically equivalent rewrite,
                   quotes verbatim, length matched  -> C7 circularity test
  entity_swap      minimal edit, only named referents replaced by wrong ones
                   of the same category                -> B3 faithfulness test
  p1_only          paragraph 1 only: genre/format claim
  p2_only          paragraph 2 only: verbatim quotation + topic claim
  p3_only          paragraph 3 only: final-token continuation prediction
  p12              paragraphs 1+2, continuation prediction dropped
  quote_strip_p2   verbatim source quotation blanked out
  quote_strip_p3   candidate-continuation strings blanked out
  quote_strip_all  every quoted span blanked out
  word_shuffle     seeded permutation of all words: destroys syntax and
                   discourse, keeps the bag of words -> is AR a topic detector?

Everything except `para_tp` is deterministic, so the whole condition set is
reproducible from this file plus the two frozen inputs.

    python 28_build_text_variants.py --results ../results \
        --out ../results/c7b3_variants_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PARA_SUBSET = list(range(0, 40, 2))  # pre-specified: every even row, 4 per doc
SHUFFLE_SEED = 20260730
QUOTE_RE = re.compile(r'"[^"]*"')
BLANK = '"[...]"'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_paraphrases(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8")
    out: dict[int, str] = {}
    cur: int | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^### idx (\d+)\s*$", line)
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur = int(m.group(1))
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts


def apply_swaps(text: str, pairs: list[list[str]]) -> tuple[str, int]:
    n = 0
    for src, dst in pairs:
        if src in text:
            n += text.count(src)
            text = text.replace(src, dst)
    return text, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--server", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = Path(args.results)
    srv = Path(args.server)
    rows = json.loads((res / "nla_results.json").read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 40, len(rows)

    para_path = srv / "c7_paraphrases_opus_v1.txt"
    swap_path = srv / "b3_entity_swap_map_v1.json"
    paras = parse_paraphrases(para_path)
    swap_map = json.loads(swap_path.read_text(encoding="utf-8"))["per_document"]

    missing = sorted(set(PARA_SUBSET) - set(paras))
    extra = sorted(set(paras) - set(PARA_SUBSET))
    if missing or extra:
        raise SystemExit(f"paraphrase subset mismatch: missing={missing} extra={extra}")

    out_rows = []
    diag = {"para_len_ratio": [], "para_shared_5gram": [], "swap_counts": [],
            "n_paragraphs": []}

    for r in rows:
        idx = r["idx"]
        orig = r["explanation"]
        ps = paragraphs(orig)
        diag["n_paragraphs"].append(len(ps))
        p1 = ps[0]
        p3 = ps[-1] if len(ps) >= 2 else ""
        p2 = "\n\n".join(ps[1:-1]) if len(ps) >= 3 else (ps[1] if len(ps) == 2 else "")

        variants: dict[str, str] = {
            "orig": orig,
            "p1_only": p1,
            "p2_only": p2,
            "p3_only": p3,
            "p12": "\n\n".join([x for x in (p1, p2) if x]),
            "quote_strip_p2": "\n\n".join(
                [x for x in (p1, QUOTE_RE.sub(BLANK, p2), p3) if x]
            ),
            "quote_strip_p3": "\n\n".join(
                [x for x in (p1, p2, QUOTE_RE.sub(BLANK, p3)) if x]
            ),
            "quote_strip_all": QUOTE_RE.sub(BLANK, orig),
        }

        rng = np.random.default_rng(SHUFFLE_SEED + idx)
        words = orig.split()
        variants["word_shuffle"] = " ".join(
            [words[i] for i in rng.permutation(len(words))]
        )

        swapped, n_sw = apply_swaps(orig, swap_map[str(r["doc_id"])])
        variants["entity_swap"] = swapped
        diag["swap_counts"].append(n_sw)
        if n_sw == 0:
            raise SystemExit(f"idx {idx}: entity swap map matched nothing")

        if idx in paras:
            pv = paras[idx]
            variants["para_tp"] = pv
            diag["para_len_ratio"].append(round(len(pv) / len(orig), 3))
            # R3 check: no shared 5-gram outside quoted spans
            # Blanking quotes collapses the candidate lists into "or or or ...",
            # so n-grams made only of function words are check artifacts.
            stop = {"or", "the", "a", "an", "of", "and", "to", "in", "for",
                    "is", "it", "as", "at", "by", "then", "likely", "then"}

            def unquoted_5grams(t: str) -> set[str]:
                t = QUOTE_RE.sub(" ", t).lower()
                w = re.findall(r"[a-z']+", t)
                grams = set()
                for i in range(max(0, len(w) - 4)):
                    g = w[i : i + 5]
                    if any(tok not in stop for tok in g):
                        grams.add(" ".join(g))
                return grams

            shared = unquoted_5grams(orig) & unquoted_5grams(pv)
            diag["para_shared_5gram"].append(sorted(shared))

        out_rows.append(
            {
                "idx": idx,
                "doc_id": r["doc_id"],
                "position": r["position"],
                "token": r["token"],
                "nla_cos_raw_reported": r["nla_cos"],
                "entity_swap_replacements": n_sw,
                "variants": variants,
            }
        )

    generic = [
        "The passage uses a structured informational style and continues the current topic.",
        "This is a coherent piece of explanatory prose with ordinary grammatical structure.",
        "The context establishes a subject and prepares a likely continuation of the discussion.",
        "The text contains semantic and syntactic information typical of a written document.",
        "A descriptive answer is being developed in a clear and organized format.",
        "The final token fits a locally predictable continuation in the surrounding sentence.",
        "The activation reflects general language structure, topical context, and discourse form.",
        "This appears to be an informative response that elaborates on previously introduced material.",
    ]

    payload = {
        "schema_version": 1,
        "experiment": "N1 / C7+B3 text-channel decomposition (frozen variants)",
        "frozen_inputs": {
            "c7_paraphrases_opus_v1.txt": sha256(para_path),
            "b3_entity_swap_map_v1.json": sha256(swap_path),
            "nla_results.json": sha256(res / "nla_results.json"),
        },
        "protocol": {
            "paraphrase_subset": PARA_SUBSET,
            "paraphrase_author": "Claude Opus 5 (non-Gemma family), blind to all AR scores",
            "shuffle_seed": SHUFFLE_SEED,
            "quote_blank_token": BLANK,
            "scoring": "AR reconstruct(text) then cosine against x_i with the E1-E7 "
            "mean direction m_hat projected out of BOTH sides; raw cos also reported",
            "primary_endpoints": [
                "para_tp vs orig  (C7: does third-party equivalent wording retain the score?)",
                "entity_swap vs orig  (B3: does referent identity matter at all?)",
                "word_shuffle vs orig  (is AR a bag-of-words topic detector?)",
                "p3_only / p12  (is the score carried by the next-token prediction?)",
                "quote_strip_p2  (is the score carried by the verbatim quotation?)",
            ],
        },
        "generic_fixed_texts": generic,
        "diagnostics": {
            "n_paragraphs_distribution": {
                str(k): int(sum(1 for v in diag["n_paragraphs"] if v == k))
                for k in sorted(set(diag["n_paragraphs"]))
            },
            "entity_swap_replacements_total": int(sum(diag["swap_counts"])),
            "entity_swap_replacements_min": int(min(diag["swap_counts"])),
            "entity_swap_rows_with_one_replacement": [
                r["idx"] for r in out_rows if r["entity_swap_replacements"] == 1
            ],
            "paraphrase_rows_meeting_R3": int(
                sum(1 for g in diag["para_shared_5gram"] if not g)
            ),
            "paraphrase_length_ratio_min": min(diag["para_len_ratio"]),
            "paraphrase_length_ratio_max": max(diag["para_len_ratio"]),
            "paraphrase_rows_with_shared_unquoted_5gram": {
                str(PARA_SUBSET[i]): g
                for i, g in enumerate(diag["para_shared_5gram"])
                if g
            },
        },
        "rows": out_rows,
    }

    outp = Path(args.out)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256(outp)
    (outp.with_suffix(".sha256")).write_text(f"{digest}  {outp.name}\n", encoding="utf-8")

    n_texts = sum(len(r["variants"]) for r in out_rows)
    print(json.dumps(payload["diagnostics"], ensure_ascii=False, indent=2))
    print(f"\nvariant texts: {n_texts} (+{len(generic)} generic) -> {outp}")
    print(f"frozen sha256: {digest}")


if __name__ == "__main__":
    main()
