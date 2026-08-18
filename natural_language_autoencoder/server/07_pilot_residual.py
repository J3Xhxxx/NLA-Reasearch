#!/usr/bin/env python3
"""Pilot 2 — is the SAE's "dark matter" (reconstruction residual) readable?

For each of the 40 activations x: residual r = x - SAE(x)  (~1/3 of variance,
FVE=0.61). We verbalize r with the AV and score three round-trips with the AR:

  cos_rr : describe(r) vs r      — is the residual itself describable?
  cos_fr : describe(x) vs r      — does the FULL-vector explanation already
                                   cover the residual? (from nla_results.json)
  cos_rx : describe(r) vs x      — how much of the full vector the residual
                                   description carries.

If cos_rr >> cos_fr the residual contains distinct, describable content the
SAE dictionary missed — i.e. the dark matter is auditable with NLA.
Residuals are near-orthogonal to the shared mean direction, so these cos
values are honest (not inflated like full-vector cos).

    python 07_pilot_residual.py --av ... --ar ... --sae ... \
        --activations acts_L32.parquet --nla nla_results.json --out resid_pilot.json
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
    ap.add_argument("--nla", required=True, help="nla_results.json (full-vector explanations)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sae = JumpReLUSAE(args.sae, device="cuda")
    vecs, meta = load_acts(args.activations)
    X = torch.from_numpy(vecs)
    recon, _ = sae(X)
    resid = (X.float() - recon.cpu().float()).numpy()
    del recon
    torch.cuda.empty_cache()

    full = {r["idx"]: r["explanation"]
            for r in json.loads(Path(args.nla).read_text(encoding="utf-8"))["rows"]}

    n = len(vecs) if args.limit <= 0 else min(args.limit, len(vecs))
    av = AVLocal(args.av, device="cuda")
    critic = NLACritic(args.ar, device="cuda")

    rows = []
    for i in range(n):
        x, r = vecs[i], resid[i]
        rtext = av.generate(r, max_new_tokens=args.max_new_tokens)
        _, cos_rr = critic.score(rtext, r)
        _, cos_fr = critic.score(full[i], r)
        _, cos_rx = critic.score(rtext, x)
        rows.append({
            "idx": i, "doc_id": meta["doc_id"][i], "position": meta["position"][i],
            "token": meta["token"][i],
            "resid_frac": round(float(np.linalg.norm(r) / np.linalg.norm(x)), 4),
            "resid_explanation": rtext, "full_explanation": full[i],
            "cos_rr": round(float(cos_rr), 4),
            "cos_fr": round(float(cos_fr), 4),
            "cos_rx": round(float(cos_rx), 4),
        })
        print(f"[{i+1}/{n}] frac={rows[-1]['resid_frac']:.3f} "
              f"rr={cos_rr:.3f} fr={cos_fr:.3f} rx={cos_rx:.3f}  {rtext[:60]!r}")

    def m(k):
        return round(float(np.mean([r[k] for r in rows])), 4)

    summary = {"n": n, "mean_resid_frac": m("resid_frac"),
               "mean_cos_rr": m("cos_rr"), "mean_cos_fr": m("cos_fr"),
               "mean_cos_rx": m("cos_rx")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"method": "residual_pilot", "summary": summary, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESID summary: {summary}\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
