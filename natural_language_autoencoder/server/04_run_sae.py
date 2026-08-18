#!/usr/bin/env python3
"""SAE line: reconstruct each activation with a gemma-scope-2 JumpReLU SAE.

Self-contained — loads params.safetensors directly and applies the gemma-scope
JumpReLU formula. No sae-lens dependency (avoids the torch/transformers version
churn that pulling sae-lens can cause).

  encode:  pre = x @ w_enc + b_enc ; acts = relu(pre) * (pre > threshold)
  decode:  x_hat = acts @ w_dec + b_dec

Two metric families are reported so the comparison is fair AND complete:
  * direction (head-to-head with NLA): normalize x and x_hat to the AR's
    mse_scale (√d), then cos and mse_nrm = 2(1-cos). Directly comparable to the
    NLA numbers, which are direction-only.
  * native SAE quality: raw MSE, FVE (fraction of variance explained over the
    dataset), and L0 (mean active features). NLA has no analogue for these —
    they are the SAE's home turf.

    python 04_run_sae.py \
        --sae /root/autodl-tmp/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small \
        --activations /root/autodl-tmp/activations/acts_L32.parquet \
        --out /root/autodl-tmp/results/sae_results.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file


class JumpReLUSAE:
    def __init__(self, sae_dir: str, device: str = "cuda", dtype=torch.float32):
        p = load_file(str(Path(sae_dir) / "params.safetensors"))
        self.w_enc = p["w_enc"].to(device, dtype)      # [d, F]
        self.b_enc = p["b_enc"].to(device, dtype)      # [F]
        self.w_dec = p["w_dec"].to(device, dtype)      # [F, d]
        self.b_dec = p["b_dec"].to(device, dtype)      # [d]
        self.threshold = p["threshold"].to(device, dtype)  # [F]
        self.d_model = self.w_enc.shape[0]
        self.width = self.w_enc.shape[1]
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor):
        x = x.to(self.device, self.dtype)
        pre = x @ self.w_enc + self.b_enc
        acts = torch.relu(pre) * (pre > self.threshold)
        recon = acts @ self.w_dec + self.b_dec
        return recon, acts


def load_acts(path):
    t = pq.read_table(path)
    flat = t.column("activation_vector").combine_chunks()
    vecs = np.array(flat.to_pylist(), dtype=np.float32)
    meta = {c: t.column(c).to_pylist() for c in ("token", "position", "doc_id")}
    return vecs, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sae", required=True, help="Dir with params.safetensors + config.json.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mse-scale", type=float, default=0.0,
                    help="L2 scale for the direction metric. Default 0 => √d (matches AR).")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sae = JumpReLUSAE(args.sae, device=args.device)
    cfg = {}
    cfg_path = Path(args.sae) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    vecs, meta = load_acts(args.activations)
    if args.limit > 0:
        vecs = vecs[:args.limit]
        meta = {k: v[:args.limit] for k, v in meta.items()}
    assert vecs.shape[1] == sae.d_model, f"act d={vecs.shape[1]} != SAE d={sae.d_model}"

    scale = args.mse_scale or math.sqrt(sae.d_model)
    X = torch.from_numpy(vecs)
    recon, acts = sae(X)
    recon = recon.float().cpu()
    Xc = X.float()

    rows = []
    for i in range(len(vecs)):
        v, vh = Xc[i], recon[i]
        vn = v / v.norm().clamp_min(1e-12) * scale
        vhn = vh / vh.norm().clamp_min(1e-12) * scale
        cos = float((vn @ vhn) / (vn.norm() * vhn.norm()))
        mse_nrm = float(((vn - vhn) ** 2).mean())   # = 2(1-cos)
        rows.append({
            "idx": i,
            "doc_id": meta["doc_id"][i],
            "position": meta["position"][i],
            "token": meta["token"][i],
            "raw_norm": round(float(v.norm()), 3),
            "sae_cos": round(cos, 4),
            "sae_mse_nrm": round(mse_nrm, 4),
            "sae_l0": int((acts[i] > 0).sum().item()),
            "sae_mse_raw": round(float(((v - vh) ** 2).mean()), 4),
        })

    # Dataset-level FVE (the standard SAE reconstruction metric).
    resid = (Xc - recon)
    fve = 1.0 - (resid.pow(2).sum() / (Xc - Xc.mean(0, keepdim=True)).pow(2).sum()).item()
    cos = np.array([r["sae_cos"] for r in rows])
    summary = {
        "n": len(rows),
        "sae_dir": str(args.sae),
        "width": sae.width,
        "config_l0": cfg.get("l0"),
        "mean_cos": round(float(cos.mean()), 4),
        "median_cos": round(float(np.median(cos)), 4),
        "mean_mse_nrm": round(float(np.array([r["sae_mse_nrm"] for r in rows]).mean()), 4),
        "mean_l0": round(float(np.array([r["sae_l0"] for r in rows]).mean()), 1),
        "fve": round(fve, 4),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"method": "sae", "summary": summary, "rows": rows},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SAE summary: {summary}\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
