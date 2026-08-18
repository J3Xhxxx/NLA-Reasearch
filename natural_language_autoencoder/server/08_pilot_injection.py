#!/usr/bin/env python3
"""Pilot 3 — residual signal-injection sensitivity: can NLA FIND a hidden signal?

Threat model for SAE auditing: information the dictionary misses lives in the
residual r = x - SAE(x). Pilot 2 showed raw residuals are NLA-opaque (cos~0).
Here we plant a KNOWN signal in the residual at controlled strength and ask
whether the NLA verbalizer detects it:

    r_alpha = r + alpha * ||r|| * w_hat_j      (w_hat_j = unit w_dec direction)

For each generated description we score with the AR:
    cos_sig : description vs the pure injected direction  -> signal recovery
    cos_res : description vs the raw residual             -> residual content

Sweep alpha in {0, 0.25, 0.5, 1, 2}. alpha=1 means the planted signal is as
large as the whole residual (~12% of activation norm); alpha=0.25 ~ 3%.
Features are the 4 strongest round-trippers from pilot 1 (sign-blindness means
|cos_sig| is the detection statistic). alpha=0 rows give the per-residual
false-positive floor. Directions carry no shared-mean component, so scores
are honest.

    python 08_pilot_injection.py --av ... --ar ... --sae ... \
        --activations acts_L32.parquet --out injection_pilot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot_common import AVLocal, JumpReLUSAE, NLACritic, load_acts

FEATURES = [166, 443, 239, 490]   # strongest |round-trip cos| in wdec_pilot
RESID_ROWS = [0, 10, 20, 30]      # spread over the 5 docs
ALPHAS = [0.25, 0.5, 1.0, 2.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sae = JumpReLUSAE(args.sae, device="cuda")
    vecs, meta = load_acts(args.activations)
    X = torch.from_numpy(vecs)
    recon, _ = sae(X)
    resid = (X.float() - recon.cpu().float()).numpy()
    dirs = {j: (sae.w_dec[j] / sae.w_dec[j].norm()).float().cpu().numpy()
            for j in FEATURES}
    target_norm = float(np.linalg.norm(vecs, axis=1).mean())
    del recon, X
    torch.cuda.empty_cache()

    av = AVLocal(args.av, device="cuda")
    critic = NLACritic(args.ar, device="cuda")

    rows = []

    def probe(i, j, alpha, v):
        text = av.generate(v, max_new_tokens=args.max_new_tokens)
        _, cos_sig = critic.score(text, dirs[j] * target_norm)
        _, cos_res = critic.score(text, resid[i])
        rec = {"resid_idx": i, "token": meta["token"][i], "feature": j,
               "alpha": alpha,
               "overlap": round(float(dirs[j] @ (resid[i] / np.linalg.norm(resid[i]))), 4),
               "explanation": text,
               "cos_sig": round(float(cos_sig), 4),
               "cos_res": round(float(cos_res), 4)}
        rows.append(rec)
        print(f"[r{i} f{j} a={alpha}] sig={cos_sig:+.3f} res={cos_res:+.3f}  "
              f"{text[:60]!r}")

    for i in RESID_ROWS:
        r = resid[i]
        rn = float(np.linalg.norm(r))
        # alpha=0: one generation, scored against every feature direction
        text0 = av.generate(r, max_new_tokens=args.max_new_tokens)
        _, cos_res0 = critic.score(text0, r)
        for j in FEATURES:
            _, cs = critic.score(text0, dirs[j] * target_norm)
            rows.append({"resid_idx": i, "token": meta["token"][i], "feature": j,
                         "alpha": 0.0,
                         "overlap": round(float(dirs[j] @ (r / rn)), 4),
                         "explanation": text0,
                         "cos_sig": round(float(cs), 4),
                         "cos_res": round(float(cos_res0), 4)})
        print(f"[r{i} a=0] res={cos_res0:+.3f}  {text0[:60]!r}")
        for j in FEATURES:
            for a in ALPHAS:
                probe(i, j, a, r + a * rn * dirs[j])

    # summary: detection curve, |cos_sig| by alpha
    curve = {}
    for a in [0.0] + ALPHAS:
        sel = [abs(r["cos_sig"]) for r in rows if r["alpha"] == a]
        curve[str(a)] = {"n": len(sel),
                         "mean_abs_cos_sig": round(float(np.mean(sel)), 4),
                         "max": round(float(np.max(sel)), 4),
                         "frac_gt_0.5": round(float(np.mean([s > 0.5 for s in sel])), 3)}
    summary = {"features": FEATURES, "resid_rows": RESID_ROWS,
               "target_norm": round(target_norm, 1), "detection_curve": curve}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"method": "injection_pilot", "summary": summary, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nINJECTION curve: {json.dumps(curve)}\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
