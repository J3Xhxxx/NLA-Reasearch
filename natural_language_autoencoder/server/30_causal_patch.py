#!/usr/bin/env python3
"""N2 / E10: causal patch-in of every reconstruction back into the base model.

This is the field-standard interpretability metric that the project has never
run: instead of asking how close a reconstruction is in cosine, patch it into
`model.layers[32]`'s output at the exact token position it came from, continue
the forward pass, and measure how much of the model's own computation survives.

Metrics per (row, substitute):
  kl_at_pos      KL(clean || patched) for the next-token distribution at the
                 patched position
  kl_mean_after  mean KL over all positions from the patched one to the end
  ce_after       cross entropy of the actual continuation tokens after the patch
  loss_recovered 1 - (ce_patched - ce_clean) / (ce_ablate - ce_clean), with both
                 zero ablation and dataset-mean ablation as the denominator

Substitutes are norm-matched to ||x_i|| by default (AR predictions live at
mse_scale, not raw activation scale, so an unmatched comparison would only
measure scale). The three `*_as_is` variants keep native scale to expose that.

    python 30_causal_patch.py --base-model ... --activations acts_L32.parquet \
        --recon results/recon_vectors.npz --out results/causal_patch_v1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER = 32


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


@torch.inference_mode()
def forward_logits(model, ids, patch=None):
    """patch = (position, vector[d]) applied to layers[LAYER] output."""
    layers = resolve_layers(model)
    handle = None
    if patch is not None:
        pos, vec = patch

        def hook(_m, _i, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            h = h.clone()
            h[:, pos, :] = vec.to(h.dtype).to(h.device)
            return (h,) + tuple(out[1:]) if is_tuple else h

        handle = layers[LAYER].register_forward_hook(hook)
    try:
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return o.logits[0].float()  # [seq, vocab]


@torch.inference_mode()
def capture_resid(model, ids):
    layers = resolve_layers(model)
    grab = {}

    def hook(_m, _i, out):
        grab["h"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()

    h = layers[LAYER].register_forward_hook(hook)
    try:
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        h.remove()
    return grab["h"][0]


def kl_rows(clean_logits, patched_logits, start):
    lp_c = torch.log_softmax(clean_logits[start:], dim=-1)
    lp_p = torch.log_softmax(patched_logits[start:], dim=-1)
    p_c = lp_c.exp()
    kl = (p_c * (lp_c - lp_p)).sum(-1)  # [n_pos]
    return kl


def ce_after(logits, ids, start):
    """CE of predicting the real token at t+1, for t in [start, L-2]."""
    seq = ids[0]
    if start >= seq.shape[0] - 1:
        return float("nan")
    lp = torch.log_softmax(logits[start:-1], dim=-1)
    tgt = seq[start + 1 :].to(lp.device)
    return float(-lp.gather(1, tgt[:, None]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    t = pq.read_table(args.activations)
    flat = t.column("activation_vector").combine_chunks()
    X = np.array(flat.to_pylist(), dtype=np.float32)
    positions = t.column("position").to_pylist()
    doc_ids = t.column("doc_id").to_pylist()
    prompts = t.column("prompt").to_pylist()
    tokens = t.column("token").to_pylist()
    n = len(X)
    docs = {}
    for d, p in zip(doc_ids, prompts):
        docs[int(d)] = p

    z = np.load(args.recon)
    m_vec = X.mean(0)  # affine dataset mean, for mean-ablation
    subs_src = {
        "identity": X,
        "nla": z["pred_full"],
        "sae_small": z["recon_sae_small"],
        "sae_big": z["recon_sae_big"],
        "resid_text": z["pred_resid"],
        "dataset_mean": np.repeat(m_vec[None], n, 0),
        "other_activation": np.stack([X[(i + 8) % n] for i in range(n)]),
    }
    rng = np.random.default_rng(20260730)
    subs_src["gaussian"] = rng.standard_normal(X.shape).astype(np.float32)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=getattr(torch, args.dtype),
        device_map="cuda", trust_remote_code=True).eval()
    device = next(model.parameters()).device
    t0 = time.time()

    ids_by_doc, clean_by_doc, resid_by_doc = {}, {}, {}
    for d, prompt in sorted(docs.items()):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(device)
        ids_by_doc[d] = ids
        resid_by_doc[d] = capture_resid(model, ids)
        clean_by_doc[d] = forward_logits(model, ids).cpu()
        print(f"[doc {d}] seq={ids.shape[1]} clean forward done "
              f"({time.time()-t0:.0f}s)", flush=True)

    # provenance: captured activation must match the stored one
    prov = []
    for i in range(n):
        h = resid_by_doc[int(doc_ids[i])][int(positions[i])].numpy()
        c = float(h @ X[i] / (np.linalg.norm(h) * np.linalg.norm(X[i]) + 1e-12))
        prov.append(round(c, 6))
    print(f"[provenance] min cos(captured, stored) = {min(prov):.6f}", flush=True)

    rows = []
    n_fwd = 0
    for i in range(n):
        d = int(doc_ids[i])
        pos = int(positions[i])
        ids = ids_by_doc[d]
        clean = clean_by_doc[d]
        xn = float(np.linalg.norm(X[i]))
        ce_c = ce_after(clean, ids, pos)

        rec: dict[str, dict] = {}
        plan = [(k, v[i], True) for k, v in subs_src.items()]
        plan += [(f"{k}_as_is", subs_src[k][i], False)
                 for k in ("nla", "sae_small", "sae_big")]
        plan += [("zero", np.zeros_like(X[i]), False)]

        for name, vec, match_norm in plan:
            v = np.asarray(vec, np.float32)
            nv = float(np.linalg.norm(v))
            if match_norm and nv > 0:
                v = v * (xn / nv)
            vt = torch.from_numpy(v)
            patched = forward_logits(model, ids, patch=(pos, vt)).cpu()
            n_fwd += 1
            kl = kl_rows(clean, patched, pos)
            rec[name] = {
                "kl_at_pos": round(float(kl[0]), 5),
                "kl_mean_after": round(float(kl.mean()), 5),
                "ce_after": round(ce_after(patched, ids, pos), 5),
                "norm_ratio": round(nv / xn, 4),
            }

        ce_zero = rec["zero"]["ce_after"]
        ce_mean = rec["dataset_mean"]["ce_after"]
        for name, r in rec.items():
            for tag, base in (("zero", ce_zero), ("mean", ce_mean)):
                den = base - ce_c
                r[f"loss_recovered_vs_{tag}"] = (
                    round(1.0 - (r["ce_after"] - ce_c) / den, 4)
                    if den and abs(den) > 1e-9 else None
                )
        rows.append({
            "idx": i, "doc_id": d, "position": pos, "token": tokens[i],
            "provenance_cos": prov[i], "ce_clean": round(ce_c, 5),
            "results": rec,
        })
        print(f"[row {i:>2}] pos={pos:>3} nla_kl={rec['nla']['kl_at_pos']:.3f} "
              f"sae_s_kl={rec['sae_small']['kl_at_pos']:.3f} "
              f"zero_kl={rec['zero']['kl_at_pos']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    names = list(rows[0]["results"].keys())
    summary = {}
    for name in names:
        get = lambda f: np.array(  # noqa: E731
            [r["results"][name][f] for r in rows if r["results"][name][f] is not None],
            dtype=float)
        summary[name] = {
            "kl_at_pos_mean": round(float(get("kl_at_pos").mean()), 5),
            "kl_at_pos_median": round(float(np.median(get("kl_at_pos"))), 5),
            "kl_mean_after_mean": round(float(get("kl_mean_after").mean()), 5),
            "ce_after_mean": round(float(get("ce_after").mean()), 5),
            "loss_recovered_vs_zero_mean": round(
                float(get("loss_recovered_vs_zero").mean()), 4),
            "loss_recovered_vs_mean_mean": round(
                float(get("loss_recovered_vs_mean").mean()), 4),
            "norm_ratio_mean": round(float(get("norm_ratio").mean()), 4),
        }

    # does centered cosine predict the causal metric?
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))

    m_hat = m_vec / np.linalg.norm(m_vec)

    def perp(a):
        return a - np.outer(a @ m_hat, m_hat)

    assoc = {}
    for name, key in (("nla", "pred_full"), ("sae_small", "recon_sae_small"),
                      ("sae_big", "recon_sae_big"), ("resid_text", "pred_resid")):
        P, T = perp(z[key].astype(np.float64)), perp(X.astype(np.float64))
        cc = np.sum(P * T, 1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1))
        kl = np.array([r["results"][name]["kl_at_pos"] for r in rows])
        lr = np.array([r["results"][name]["loss_recovered_vs_zero"] for r in rows],
                      dtype=float)
        assoc[name] = {
            "mean_centered_cos": round(float(cc.mean()), 4),
            "spearman_centered_cos_vs_kl_at_pos": round(spearman(cc, kl), 4),
            "spearman_centered_cos_vs_loss_recovered": round(spearman(cc, lr), 4),
        }
    pooled_cc, pooled_kl = [], []
    for name, key in (("nla", "pred_full"), ("sae_small", "recon_sae_small"),
                      ("sae_big", "recon_sae_big"), ("resid_text", "pred_resid")):
        P, T = perp(z[key].astype(np.float64)), perp(X.astype(np.float64))
        pooled_cc += list(np.sum(P * T, 1) /
                          (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1)))
        pooled_kl += [r["results"][name]["kl_at_pos"] for r in rows]
    assoc["pooled_all_methods"] = {
        "n": len(pooled_cc),
        "spearman_centered_cos_vs_kl_at_pos": round(
            spearman(np.array(pooled_cc), np.array(pooled_kl)), 4),
    }

    out = {
        "schema_version": 1,
        "experiment": "N2 / E10 causal patch-in (KL and loss recovered) at L32",
        "protocol": {
            "layer": LAYER,
            "patch": "replace layers[32] output at the source token position, "
                     "then continue the forward pass",
            "norm_matching": "substitutes rescaled to ||x_i|| except *_as_is and zero",
            "ce_window": "actual continuation tokens from the patched position to the end",
            "loss_recovered": "1 - (ce_patched - ce_clean)/(ce_ablate - ce_clean)",
            "dtype": args.dtype,
        },
        "provenance": {
            "min_cos_captured_vs_stored": min(prov),
            "mean_cos_captured_vs_stored": round(float(np.mean(prov)), 6),
        },
        "summary": summary,
        "cos_vs_causal_association": assoc,
        "rows": rows,
        "n_forwards": n_fwd,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("\nPROVENANCE:", json.dumps(out["provenance"]))
    for k, v in summary.items():
        print(f"{k:<18} kl@pos={v['kl_at_pos_mean']:8.4f} "
              f"klafter={v['kl_mean_after_mean']:8.4f} "
              f"LR_zero={v['loss_recovered_vs_zero_mean']:+.3f} "
              f"LR_mean={v['loss_recovered_vs_mean_mean']:+.3f}")
    print("ASSOC:", json.dumps(assoc))
    print(f"CAUSAL_PATCH_COMPLETE -> {args.out}")


if __name__ == "__main__":
    main()
