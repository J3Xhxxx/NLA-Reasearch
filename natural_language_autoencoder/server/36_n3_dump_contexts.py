#!/usr/bin/env python3
"""N3 step 4 — decode real max-activating contexts, and freeze a larger cohort.

CPU only, seconds to run. Two jobs:

1. For every direction B6+B4 froze (45 of them), print what the feature ACTUALLY
   fires on in real text. This matters because the synthetic selection's
   "top_contexts" for e.g. feature 8347 labelled `domain:biology` were tokens
   like " las", " por", "." inside Spanish biology prompts -- consistent with the
   surface audit finding 9/24 labels clearly wrong. Real contexts let a later
   blind audit judge labels against evidence rather than against our own prompts.

2. Freeze a candidate cohort of ~120 features selected by REAL-corpus criteria,
   stratified into three pre-registered strata (source-concentrated,
   source-distributed, language-selective on the parallel half). This is what
   lets a future C1 run at n>=100 without us writing any corpus, and the strata
   are defined before anyone looks at an AV explanation.

Language selectivity is computed on the XNLI half only, where content is held
constant across the 10 languages -- so a high score cannot be topic leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)) if len(p) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default="/root/autodl-tmp/results/n3_feature_stats_v1.npz")
    ap.add_argument("--corpus", default="/root/autodl-tmp/results/n3_corpus_v1.jsonl")
    ap.add_argument("--selection",
                    default="/root/autodl-tmp/results/b6b4_factorial_selection.json")
    ap.add_argument("--base-model", default="/root/autodl-tmp/models/gemma-3-12b-it")
    ap.add_argument("--sae", default="small", choices=["small", "big"])
    ap.add_argument("--window-before", type=int, default=24)
    ap.add_argument("--window-after", type=int, default=6)
    ap.add_argument("--contexts-per-feature", type=int, default=8)
    ap.add_argument("--per-stratum", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out-contexts", default="/root/autodl-tmp/results/n3_contexts_v1.json")
    ap.add_argument("--out-cohort",
                    default="/root/autodl-tmp/results/n3_candidate_cohort_v1.json")
    args = ap.parse_args()

    st = np.load(args.stats, allow_pickle=True)
    pre = args.sae + "_"
    groups = [str(g) for g in st["groups"]]
    fire = st[pre + "fire"]                    # [F]
    act_sum = st[pre + "act_sum"]
    act_max = st[pre + "act_max"]
    gfire = st[pre + "group_fire"]             # [G, F]
    gtok = st[pre + "group_tokens"]            # [G]
    top_val = st[pre + "top_val"]              # [F, K]
    top_meta = st[pre + "top_meta"]            # [F, K, 3]
    n_tokens = int(st[pre + "n_tokens"])
    width = fire.shape[0]

    docs = {}
    for line in open(args.corpus, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            docs[d["doc_id"]] = d

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # cache re-tokenized documents lazily (positions in top_meta are indices into
    # the full document token stream, exactly as 34 built them)
    tok_cache: dict[int, list[int]] = {}

    def doc_ids(doc_id: int) -> list[int]:
        if doc_id not in tok_cache:
            tok_cache[doc_id] = tok(docs[doc_id]["text"], add_special_tokens=True)["input_ids"]
        return tok_cache[doc_id]

    def render(doc_id: int, pos: int) -> dict:
        ids = doc_ids(doc_id)
        lo = max(0, pos - args.window_before)
        hi = min(len(ids), pos + 1 + args.window_after)
        d = docs[doc_id]
        return {
            "doc_id": int(doc_id), "position": int(pos),
            "source": d["source"], "lang": d["lang"], "corpus": d["corpus"],
            "token": tok.decode(ids[pos:pos + 1], skip_special_tokens=False)
            if pos < len(ids) else "<OOR>",
            "before": tok.decode(ids[lo:pos], skip_special_tokens=False),
            "after": tok.decode(ids[pos + 1:hi], skip_special_tokens=False),
        }

    xnli_gi = [i for i, g in enumerate(groups) if g.startswith("xnli:")]
    pile_gi = [i for i, g in enumerate(groups) if not g.startswith("xnli:")]
    gfreq = gfire / np.maximum(gtok[:, None], 1)          # [G, F] per-group rate

    def feature_row(j: int) -> dict:
        f = float(fire[j])
        src_share = gfire[:, j] / max(f, 1.0)
        lang_rate = gfreq[xnli_gi, j]
        lang_fire = gfire[xnli_gi, j]
        top_g = int(np.argmax(gfire[:, j])) if f > 0 else -1
        return {
            "feature": int(j),
            "n_fire": int(f),
            "freq": f / max(n_tokens, 1),
            "mean_act_when_fire": float(act_sum[j] / f) if f else 0.0,
            "max_act": float(act_max[j]),
            "n_sources_fired": int((gfire[:, j] > 0).sum()),
            "top_source": groups[top_g] if top_g >= 0 else None,
            "top_source_share": float(src_share.max()) if f else 0.0,
            "source_entropy_norm": entropy(gfire[:, j]),
            "pile_fire": int(gfire[pile_gi, j].sum()),
            "xnli_fire": int(lang_fire.sum()),
            # parallel-corpus language selectivity: content is identical across
            # these 10 groups, so this cannot be explained by topic
            "lang_top": (groups[xnli_gi[int(np.argmax(lang_rate))]].split(":")[1]
                         if lang_fire.sum() > 0 else None),
            "lang_purity": (float(lang_fire.max() / lang_fire.sum())
                            if lang_fire.sum() > 0 else 0.0),
            "lang_rates": {groups[g].split(":")[1]: float(gfreq[g, j]) for g in xnli_gi},
            "n_langs_fired": int((lang_fire > 0).sum()),
        }

    # ---------- 1. the 45 frozen B6+B4 directions ----------
    sel = json.load(open(args.selection, encoding="utf-8"))
    frozen = []
    for d in sel["selected_directions"]:
        j = int(d["feature"])
        if j < 0:
            frozen.append({"group": d["group"], "feature": j, "label": d.get("label"),
                           "note": "gaussian control, not an SAE feature"})
            continue
        row = feature_row(j)
        row.update({"group": d["group"], "label": d.get("label"),
                    "synthetic_mean_alignment": d.get("mean_alignment"),
                    "synthetic_train_doc_support": d.get("train_doc_support")})
        k = min(args.contexts_per_feature, top_val.shape[1])
        ctxs = []
        for r in range(k):
            v = float(top_val[j, r])
            if v <= 0:
                continue
            doc_id, pos, _tid = (int(x) for x in top_meta[j, r])
            if doc_id in docs:
                c = render(doc_id, pos)
                c["activation"] = v
                ctxs.append(c)
        row["real_top_contexts"] = ctxs
        frozen.append(row)

    out_ctx = {
        "schema_version": 1,
        "experiment": "N3 real max-activating contexts for the B6+B4 frozen directions",
        "sae": args.sae, "n_tokens": n_tokens, "width": width,
        "stats_sha256": sha256_file(args.stats),
        "corpus_sha256": sha256_file(args.corpus),
        "args": vars(args),
        "directions": frozen,
    }
    Path(args.out_contexts).write_text(json.dumps(out_ctx, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    # ---------- 2. frozen candidate cohort from real-corpus criteria ----------
    freq = fire / max(n_tokens, 1)
    alive = fire > 0
    top_share = np.where(alive, gfire.max(0) / np.maximum(fire, 1), 0.0)
    n_src = (gfire > 0).sum(0)
    lang_fire_all = gfire[xnli_gi].sum(0)
    lang_purity_all = np.where(lang_fire_all > 0,
                               gfire[xnli_gi].max(0) / np.maximum(lang_fire_all, 1), 0.0)

    strata = {
        "source_concentrated": np.where(
            (fire >= 30) & (freq >= 1e-5) & (freq <= 1e-2) & (top_share >= 0.8))[0],
        "source_distributed": np.where(
            (fire >= 30) & (freq >= 1e-4) & (n_src >= 5) & (top_share <= 0.4))[0],
        "language_selective": np.where(
            (lang_fire_all >= 30) & (lang_purity_all >= 0.7))[0],
    }
    rng = random.Random(args.seed)
    cohort, used = [], set()
    for name, pool in strata.items():
        cand = [int(j) for j in pool if int(j) not in used]
        rng.shuffle(cand)
        picked = sorted(cand[:args.per_stratum])
        used.update(picked)
        for j in picked:
            row = feature_row(j)
            row["stratum"] = name
            cohort.append(row)

    out_cohort = {
        "schema_version": 1,
        "experiment": "N3 candidate feature cohort (real-corpus criteria)",
        "status": "frozen_before_any_AV_generation",
        "why": "B6+B4 could only reach n=24 because the cohort had to fit 24 "
               "hand-written documents; these criteria are computable on any real "
               "corpus, so n scales without us authoring text.",
        "sae": args.sae, "n_tokens": n_tokens,
        "stats_sha256": sha256_file(args.stats),
        "criteria": {
            "source_concentrated": "n_fire>=30, 1e-5<=freq<=1e-2, top_source_share>=0.8",
            "source_distributed": "n_fire>=30, freq>=1e-4, n_sources>=5, top_source_share<=0.4",
            "language_selective": "xnli_fire>=30, lang_purity>=0.7 (parallel corpus)",
        },
        "pool_sizes": {k: int(len(v)) for k, v in strata.items()},
        "args": vars(args),
        "features": cohort,
    }
    Path(args.out_cohort).write_text(json.dumps(out_cohort, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    print(f"alive features        : {int(alive.sum())}/{width}")
    print(f"pool sizes            : {out_cohort['pool_sizes']}")
    print(f"cohort frozen         : {len(cohort)} features")
    print(f"-> {args.out_contexts}")
    print(f"-> {args.out_cohort}")


if __name__ == "__main__":
    main()
