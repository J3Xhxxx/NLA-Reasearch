#!/usr/bin/env python3
"""Rescore ALL saved explanations in mean-direction-centered space.

Motivation: injection_pilot's alpha=0 floor hit |cos|~0.92 — the 4 test
features' w_dec directions are strongly aligned with the dataset mean
activation direction, and AR outputs for ANY generic text also land near
that mean direction. Every uncentered score against those features is
therefore confounded.

Fix: project the dataset mean direction m_hat out of BOTH the AR
reconstruction and the target before cosine:

    a_perp = a - (a . m_hat) m_hat ;  cos_c = cos(pred_perp, target_perp)

This is scale-free (AR predictions live at mse_scale, not raw-activation
scale, so affine centering is ill-defined; direction projection is not).

Rescores: nla_results (full explanations vs x), resid_pilot (residual
explanations vs r and vs x), wdec_pilot (all 22 direction probes, gauss
dirs regenerated from the same seed), injection_pilot (all 80 rows).
Also recomputes the SAE small/big reconstructions' centered cos directly
from vectors, and dumps every AR reconstruction + originals to an .npz so
retrieval-style evals can run locally later without a GPU.

    python 09_rescore_centered.py --ar ... --sae-small ... --sae-big ... \
        --activations acts_L32.parquet --results /root/autodl-tmp/results \
        --out centered_rescore.json --vecs-out recon_vectors.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot_common import JumpReLUSAE, NLACritic, load_acts

FEATURES = [166, 443, 239, 490]   # must match 08_pilot_injection.py


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--sae-small", required=True)
    ap.add_argument("--sae-big", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vecs-out", required=True)
    args = ap.parse_args()

    res = Path(args.results)
    vecs, _ = load_acts(args.activations)          # [40, d]
    X = torch.from_numpy(vecs).float()
    m_hat = X.mean(0)
    m_hat = (m_hat / m_hat.norm()).numpy()

    def perp(a):
        a = np.asarray(a, np.float32)
        return a - (a @ m_hat) * m_hat

    def cos(a, b):
        a, b = perp(a), perp(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    sae_s = JumpReLUSAE(args.sae_small, device="cuda")
    sae_b = JumpReLUSAE(args.sae_big, device="cuda")
    recon_s, acts_s = sae_s(X)
    recon_b, _ = sae_b(X)
    recon_s = recon_s.cpu().float().numpy()
    recon_b = recon_b.cpu().float().numpy()
    resid = vecs - recon_s

    # regenerate 06's probe directions (same seed/sequence)
    fire = (acts_s.cpu() > 0)
    freq = fire.sum(0).numpy()
    rng = np.random.default_rng(0)
    never = np.flatnonzero(freq == 0)
    rand_idx = [int(j) for j in rng.choice(never, size=6, replace=False)]
    gauss = [rng.standard_normal(sae_s.d_model).astype(np.float32) for _ in range(4)]

    def w(j):
        return sae_s.w_dec[j].float().cpu().numpy()

    def probe_dir(feature):
        if feature >= 0:
            return w(feature)
        return gauss[-1 - feature]      # feature -1..-4 -> gauss[0..3]

    critic = NLACritic(args.ar, device="cuda")
    _cache: dict[str, np.ndarray] = {}

    def rec(text):
        if text not in _cache:
            _cache[text] = critic.reconstruct(text).numpy()
        return _cache[text]

    out = {"m_hat_note": "all cos_c project dataset mean direction out of both sides"}

    # confound quantification: feature alignment with mean direction
    out["feature_mean_alignment"] = {
        str(j): round(float((w(j) / np.linalg.norm(w(j))) @ m_hat), 4) for j in FEATURES}

    # 1) head-to-head centered
    nla = json.loads((res / "nla_results.json").read_text(encoding="utf-8"))["rows"]
    pred_full = np.stack([rec(r["explanation"]) for r in nla])
    nla_c = [round(cos(pred_full[i], vecs[i]), 4) for i in range(len(nla))]
    sae_s_c = [round(cos(recon_s[i], vecs[i]), 4) for i in range(len(nla))]
    sae_b_c = [round(cos(recon_b[i], vecs[i]), 4) for i in range(len(nla))]
    out["head_to_head_centered"] = {
        "nla_mean": round(float(np.mean(nla_c)), 4),
        "sae_small_mean": round(float(np.mean(sae_s_c)), 4),
        "sae_big_mean": round(float(np.mean(sae_b_c)), 4),
        "nla": nla_c, "sae_small": sae_s_c, "sae_big": sae_b_c}

    # 2) generic-text baseline, centered (resid explanations vs full vectors)
    rp = json.loads((res / "resid_pilot.json").read_text(encoding="utf-8"))["rows"]
    pred_resid = np.stack([rec(r["resid_explanation"]) for r in rp])
    out["resid_centered"] = {
        "cos_rx_c(generic floor)": round(float(np.mean(
            [cos(pred_resid[i], vecs[i]) for i in range(len(rp))])), 4),
        "cos_rr_c": round(float(np.mean(
            [cos(pred_resid[i], resid[i]) for i in range(len(rp))])), 4),
        "resid_mean_alignment": round(float(np.mean(
            [(resid[i] / np.linalg.norm(resid[i])) @ m_hat for i in range(len(rp))])), 4)}

    # 3) wdec probes centered
    wd = json.loads((res / "wdec_pilot.json").read_text(encoding="utf-8"))["rows"]
    for r in wd:
        r["cos_c"] = round(cos(rec(r["explanation"]), probe_dir(r["feature"])), 4)
    out["wdec_centered"] = {
        g: {"mean_abs_cos_c": round(float(np.mean(
                [abs(r["cos_c"]) for r in wd if r["group"] == g])), 4),
            "rows": {str(r["feature"]): r["cos_c"] for r in wd if r["group"] == g}}
        for g in ("top", "rand", "gauss")}

    # 4) injection detection curve, centered
    ij = json.loads((res / "injection_pilot.json").read_text(encoding="utf-8"))["rows"]
    for r in ij:
        r["cos_sig_c"] = round(cos(rec(r["explanation"]), probe_dir(r["feature"])), 4)
    curve = {}
    for a in sorted({r["alpha"] for r in ij}):
        sel = [abs(r["cos_sig_c"]) for r in ij if r["alpha"] == a]
        curve[str(a)] = {"n": len(sel),
                         "mean_abs_cos_sig_c": round(float(np.mean(sel)), 4),
                         "max": round(float(np.max(sel)), 4),
                         "frac_gt_0.3": round(float(np.mean([s > 0.3 for s in sel])), 3)}
    out["injection_centered_curve"] = curve
    out["injection_rows"] = [
        {k: r[k] for k in ("resid_idx", "feature", "alpha", "cos_sig", "cos_sig_c")}
        for r in ij]

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    np.savez_compressed(
        args.vecs_out, x=vecs, m_hat=m_hat, pred_full=pred_full,
        pred_resid=pred_resid, recon_sae_small=recon_s, recon_sae_big=recon_b,
        resid=resid, feature_dirs=np.stack([w(j) for j in FEATURES]),
        feature_ids=np.array(FEATURES))
    print("HEAD2HEAD_C:", json.dumps(out["head_to_head_centered"]["nla_mean"]),
          json.dumps(out["head_to_head_centered"]["sae_small_mean"]),
          json.dumps(out["head_to_head_centered"]["sae_big_mean"]))
    print("RESID_C:", json.dumps(out["resid_centered"]))
    print("WDEC_C:", json.dumps({g: out["wdec_centered"][g]["mean_abs_cos_c"]
                                 for g in ("top", "rand", "gauss")}))
    print("CURVE_C:", json.dumps(curve))
    print(f"wrote -> {args.out} + {args.vecs_out}")


if __name__ == "__main__":
    main()
