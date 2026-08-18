#!/usr/bin/env python3
"""Pilot 1 — can the NLA verbalizer read SAE decoder directions?

For a set of SAE features we feed the w_dec[j] direction (scaled to the mean
raw activation norm; the AV normalizes internally so only direction matters)
into the AV and score the description with the AR against that same direction.

Three groups, so off-manifold degradation is measurable:
  top   — features most frequently active on our 40 activations (in-domain,
          each comes with an empirical "label": the tokens/contexts firing it)
  rand  — random dictionary features (likely never active on our prompts)
  gauss — random Gaussian directions (fully unstructured control)

Because w_dec directions carry no shared-mean component, the round-trip cos
here is NOT inflated the way full-activation cos is.

    python 06_pilot_wdec.py --av ... --ar ... --sae ... \
        --activations acts_L32.parquet --out wdec_pilot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot_common import AVLocal, JumpReLUSAE, NLACritic, load_acts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--n-top", type=int, default=12)
    ap.add_argument("--n-rand", type=int, default=6)
    ap.add_argument("--n-gauss", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    sae = JumpReLUSAE(args.sae, device="cuda")
    vecs, meta = load_acts(args.activations)
    X = torch.from_numpy(vecs)
    _, acts = sae(X)                     # [N, F]
    acts = acts.cpu()
    target_norm = float(np.linalg.norm(vecs, axis=1).mean())

    # top features by firing frequency (ties -> higher mean value)
    fire = (acts > 0)
    freq = fire.sum(0)                   # [F]
    mean_val = torch.where(freq > 0, acts.sum(0) / freq.clamp_min(1), torch.zeros(()))
    order = np.lexsort((-mean_val.numpy(), -freq.numpy()))
    top_idx = [int(j) for j in order[:args.n_top]]

    never = np.flatnonzero(freq.numpy() == 0)
    rand_idx = [int(j) for j in rng.choice(never, size=args.n_rand, replace=False)]

    def contexts(j):
        rows = torch.nonzero(fire[:, j]).flatten().tolist()
        return [{"doc_id": meta["doc_id"][i], "position": meta["position"][i],
                 "token": meta["token"][i], "value": round(float(acts[i, j]), 2)}
                for i in rows]

    probes = []
    for j in top_idx:
        d = sae.w_dec[j].float().cpu().numpy()
        probes.append({"group": "top", "feature": j, "dir": d, "contexts": contexts(j)})
    for j in rand_idx:
        d = sae.w_dec[j].float().cpu().numpy()
        probes.append({"group": "rand", "feature": j, "dir": d, "contexts": []})
    for k in range(args.n_gauss):
        probes.append({"group": "gauss", "feature": -1 - k,
                       "dir": rng.standard_normal(sae.d_model).astype(np.float32),
                       "contexts": []})

    del acts, X
    torch.cuda.empty_cache()

    av = AVLocal(args.av, device="cuda")
    critic = NLACritic(args.ar, device="cuda")

    rows = []
    for p in probes:
        v = p["dir"] / np.linalg.norm(p["dir"]) * target_norm
        text = av.generate(v, max_new_tokens=args.max_new_tokens)
        mse, cos = critic.score(text, v)
        rows.append({
            "group": p["group"], "feature": p["feature"],
            "n_fire": len(p["contexts"]), "contexts": p["contexts"],
            "explanation": text,
            "cos": round(float(cos), 4), "mse_nrm": round(float(mse), 4),
        })
        print(f"[{p['group']}:{p['feature']}] fire={len(p['contexts'])} "
              f"cos={cos:.3f}  {text[:70]!r}")

    def grp(g):
        c = [r["cos"] for r in rows if r["group"] == g]
        return {"n": len(c), "mean_cos": round(float(np.mean(c)), 4),
                "min": round(float(np.min(c)), 4), "max": round(float(np.max(c)), 4)}

    summary = {g: grp(g) for g in ("top", "rand", "gauss")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"method": "wdec_pilot", "target_norm": round(target_norm, 1),
                    "summary": summary, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWDEC summary: {json.dumps(summary)}\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
