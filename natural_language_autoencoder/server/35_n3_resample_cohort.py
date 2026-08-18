#!/usr/bin/env python3
"""N3 step 3 — resample an E1-E7-style activation cohort, without F13's defect.

The original cohort (activations/acts_L32.parquet) is unusable as a clean base:
`--min-position 50` silently never applied to the 5 short chat-formatted prompts,
the code fell back to `range(len(seq)//2, len(seq))`, and 13 of 40 positions
landed on chat-template or whitespace tokens (POSSBILITY F13). Every headline
number in the project (F1, F6, F8, F11, F12) inherits that cohort.

This script builds a replacement with the defect made impossible rather than
merely avoided:

  * raw text from the frozen N3 corpus, no chat template at all
  * a position is eligible only if it is a CONTENT token (not special, not
    whitespace-only, not punctuation-only) -- verified by decoding it
  * `--min-position` is a hard filter; documents that cannot satisfy it are
    skipped, never back-filled with a fallback range (the exact bug in 02)
  * a position must be followed by >= --min-continuation tokens that are
    themselves >= --content-frac content, so causal-patch CE windows are
    interpretable (F13's second casualty)
  * stratified across sources/languages, deterministic given --seed

Self-checks are printed and stored: template fraction must be exactly 0, and the
realised minimum position must be >= the requested one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORD_RE = re.compile(r"\w", re.UNICODE)


class _StopForward(Exception):
    pass


def resolve_layers(model: torch.nn.Module):
    for path in (("model", "layers"), ("language_model", "model", "layers"),
                 ("model", "language_model", "layers")):
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="/root/autodl-tmp/results/n3_corpus_v1.jsonl")
    ap.add_argument("--base-model", default="/root/autodl-tmp/models/gemma-3-12b-it")
    ap.add_argument("--layer-index", type=int, default=32)
    ap.add_argument("--n-target", type=int, default=200)
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--min-position", type=int, default=64)
    ap.add_argument("--max-position", type=int, default=480)
    ap.add_argument("--min-continuation", type=int, default=16)
    ap.add_argument("--content-frac", type=float, default=0.75)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out", default="/root/autodl-tmp/activations/acts_L32_n3_v1.parquet")
    ap.add_argument("--out-json", default="/root/autodl-tmp/results/n3_cohort_v1.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = [json.loads(l) for l in open(args.corpus, encoding="utf-8") if l.strip()]

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    special = set(tok.all_special_ids)

    def is_content(token_id: int) -> bool:
        if token_id in special:
            return False
        s = tok.decode([token_id], skip_special_tokens=False)
        if not s.strip():
            return False
        if s.strip().startswith("<") and s.strip().endswith(">"):
            return False
        return bool(WORD_RE.search(s))

    content_cache: dict[int, bool] = {}

    def content(tid: int) -> bool:
        if tid not in content_cache:
            content_cache[tid] = is_content(tid)
        return content_cache[tid]

    # --- stratified document order: round-robin over sources ---
    by_source: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_source[d["source"]].append(d)
    for lst in by_source.values():
        rng.shuffle(lst)
    order: list[dict] = []
    srcs = sorted(by_source)
    i = 0
    while any(by_source[s] for s in srcs):
        s = srcs[i % len(srcs)]
        if by_source[s]:
            order.append(by_source[s].pop())
        i += 1

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()
    device = next(model.parameters()).device
    layers = resolve_layers(model)

    grab: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        grab["h"] = (out[0] if isinstance(out, tuple) else out)
        raise _StopForward

    handle = layers[args.layer_index].register_forward_hook(hook)

    rows: list[dict] = []
    skipped = Counter()
    try:
        for d in order:
            if len(rows) >= args.n_target:
                break
            ids = tok(d["text"], add_special_tokens=True)["input_ids"][:args.seq_len]
            hi = min(args.max_position, len(ids) - args.min_continuation - 1)
            if hi <= args.min_position:
                skipped["too_short"] += 1
                continue

            cand = []
            for p in range(args.min_position, hi + 1):
                if not content(ids[p]):
                    continue
                win = ids[p + 1:p + 1 + args.min_continuation]
                if len(win) < args.min_continuation:
                    continue
                if sum(content(t) for t in win) / len(win) < args.content_frac:
                    continue
                cand.append(p)
            if not cand:
                skipped["no_eligible_position"] += 1
                continue

            rng.shuffle(cand)
            picked = sorted(cand[:args.per_doc])
            inp = torch.tensor([ids], device=device)
            with torch.inference_mode():
                try:
                    model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
                except _StopForward:
                    pass
                hidden = grab.pop("h")[0].float().cpu()

            for p in picked:
                if len(rows) >= args.n_target:
                    break
                rows.append({
                    "activation_vector": hidden[p].numpy().astype(np.float32),
                    "token": tok.decode([ids[p]], skip_special_tokens=False),
                    "token_id": int(ids[p]),
                    "position": int(p),
                    "doc_id": int(d["doc_id"]),
                    "corpus": d["corpus"],
                    "source": d["source"],
                    "lang": d["lang"],
                    "seq_len": int(len(ids)),
                    "context_tail": tok.decode(ids[max(0, p - 24):p + 1],
                                               skip_special_tokens=False),
                    "continuation": tok.decode(ids[p + 1:p + 1 + args.min_continuation],
                                               skip_special_tokens=False),
                    "input_ids": [int(t) for t in ids],
                    "norm": "none",
                })
            print(f"[{len(rows):>4}/{args.n_target}] doc={d['doc_id']} src={d['source']} "
                  f"seq={len(ids)} picked={picked}", flush=True)
    finally:
        handle.remove()

    arr = np.stack([r["activation_vector"] for r in rows])
    table = pa.table({
        "activation_vector": pa.array([list(map(float, v)) for v in arr],
                                      type=pa.list_(pa.float32())),
        **{k: [r[k] for r in rows] for k in
           ("token", "token_id", "position", "doc_id", "corpus", "source", "lang",
            "seq_len", "context_tail", "continuation", "norm")},
        "input_ids": pa.array([r["input_ids"] for r in rows], type=pa.list_(pa.int32())),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)

    # --- self-checks: the whole point of this script ---
    n_template = sum(1 for r in rows if not content(r["token_id"]))
    positions = np.array([r["position"] for r in rows])
    norms = np.linalg.norm(arr, axis=1)
    checks = {
        "n_rows": len(rows),
        "template_or_blank_tokens": int(n_template),
        "template_fraction": float(n_template / max(len(rows), 1)),
        "min_position_realised": int(positions.min()),
        "min_position_requested": args.min_position,
        "min_position_honoured": bool(positions.min() >= args.min_position),
        "position_percentiles": {p: float(np.percentile(positions, int(p)))
                                 for p in ("5", "25", "50", "75", "95")},
        "n_docs_used": len({r["doc_id"] for r in rows}),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_corpus": dict(Counter(r["corpus"] for r in rows)),
        "skipped": dict(skipped),
        "norm_mean": float(norms.mean()), "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
    out = {"schema_version": 1, "experiment": "N3 cohort resample (E1-E7 replacement)",
           "fixes": ["F13 template-token contamination", "F13 CE-window contamination"],
           "args": vars(args), "corpus_sha256": sha256_file(args.corpus),
           "checks": checks,
           "outputs": {"parquet": args.out, "parquet_sha256": sha256_file(args.out)}}
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    print("\n=== cohort self-check ===")
    for k in ("n_rows", "template_fraction", "min_position_realised",
              "min_position_honoured", "n_docs_used", "norm_mean"):
        print(f"  {k:28s} {checks[k]}")
    print(f"  positions p5/50/95        {checks['position_percentiles']['5']:.0f} / "
          f"{checks['position_percentiles']['50']:.0f} / "
          f"{checks['position_percentiles']['95']:.0f}")
    print(f"  by_corpus                 {checks['by_corpus']}")
    print(f"  sample tokens             {[r['token'] for r in rows[:12]]}")
    assert checks["template_fraction"] == 0.0, "content-token filter failed"
    assert checks["min_position_honoured"], "min-position filter failed"
    print(f"-> {args.out}\n-> {args.out_json}")


if __name__ == "__main__":
    main()
