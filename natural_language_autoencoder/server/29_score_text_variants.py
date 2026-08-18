#!/usr/bin/env python3
"""N1 stage 2 (GPU, AR only -- no AV generation): score the frozen C7/B3 variants.

Reads the frozen `c7b3_variants_v1.json`, verifies its SHA-256 against the
sidecar, reconstructs every variant text with the AR critic and scores it
against its own activation in the mean-direction-centered space used by all
E5+ results. Also scores the 8 fixed generic texts against every activation,
which supplies the fixed generic floor that POSSBILITY F2 flagged as never
having been measured directly for the E1-E7 cohort.

    python 29_score_text_variants.py --ar ... --activations acts_L32.parquet \
        --variants results/c7b3_variants_v1.json \
        --out results/c7b3_scores_v1.json --vecs-out results/c7b3_recon_v1.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pilot_common import NLACritic, load_acts


def boot_ci(v: np.ndarray, stat=np.mean, n_boot: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    s = stat(v[idx], axis=1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def sign_test_p(v: np.ndarray) -> float:
    from math import comb

    n = int(np.sum(v != 0))
    k = int(np.sum(v > 0))
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return float(min(1.0, 2 * tail / 2**n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vecs-out", required=True)
    args = ap.parse_args()

    vp = Path(args.variants)
    digest = hashlib.sha256(vp.read_bytes()).hexdigest()
    side = vp.with_suffix(".sha256")
    if side.exists():
        expect = side.read_text(encoding="utf-8").split()[0]
        if expect != digest:
            raise SystemExit(f"variant file hash mismatch: {digest} != {expect}")
    frozen = json.loads(vp.read_text(encoding="utf-8"))
    print(f"[freeze] variants sha256={digest} verified", flush=True)

    vecs, meta = load_acts(args.activations)
    X = torch.from_numpy(vecs).float()
    m_hat = X.mean(0)
    m_hat = (m_hat / m_hat.norm()).numpy()

    def perp(a):
        a = np.asarray(a, np.float32)
        if a.ndim == 1:
            return a - (a @ m_hat) * m_hat
        return a - np.outer(a @ m_hat, m_hat)

    def cos_c(a, b):
        a, b = perp(a), perp(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def cos_raw(a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    critic = NLACritic(args.ar, device="cuda")
    cache: dict[str, np.ndarray] = {}

    def rec(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = critic.reconstruct(text).numpy()
        return cache[text]

    t0 = time.time()
    generic_recon = np.stack([rec(t) for t in frozen["generic_fixed_texts"]])
    print(f"[generic] {len(generic_recon)} texts in {time.time()-t0:.1f}s", flush=True)

    variant_names: list[str] = []
    for r in frozen["rows"]:
        for k in r["variants"]:
            if k not in variant_names:
                variant_names.append(k)

    rows_out = []
    recon_store: dict[str, list] = {k: [] for k in variant_names}
    recon_idx: dict[str, list] = {k: [] for k in variant_names}

    done = 0
    for r in frozen["rows"]:
        i = r["idx"]
        scores = {}
        for name, text in r["variants"].items():
            p = rec(text)
            scores[name] = {
                "cos_c": round(cos_c(p, vecs[i]), 4),
                "cos_raw": round(cos_raw(p, vecs[i]), 4),
                "chars": len(text),
            }
            recon_store[name].append(p.astype(np.float32))
            recon_idx[name].append(i)
            done += 1
        g = [cos_c(generic_recon[j], vecs[i]) for j in range(len(generic_recon))]
        rows_out.append(
            {
                "idx": i,
                "doc_id": r["doc_id"],
                "token": r["token"],
                "generic_fixed_cos_c_mean": round(float(np.mean(g)), 4),
                "generic_fixed_cos_c_max": round(float(np.max(g)), 4),
                "scores": scores,
            }
        )
        print(
            f"[row {i:>2}] {done} texts, {time.time()-t0:.0f}s, "
            f"orig={scores['orig']['cos_c']:.3f}",
            flush=True,
        )

    # ---------------- summaries ----------------
    def col(name: str, field: str = "cos_c") -> np.ndarray:
        return np.array(
            [r["scores"][name][field] for r in rows_out if name in r["scores"]]
        )

    orig_all = col("orig")
    summary = {}
    for name in variant_names:
        v = col(name)
        idxs = [r["idx"] for r in rows_out if name in r["scores"]]
        o = np.array([r["scores"]["orig"]["cos_c"] for r in rows_out if name in r["scores"]])
        d = v - o
        summary[name] = {
            "n": len(v),
            "mean_cos_c": round(float(np.mean(v)), 4),
            "median_cos_c": round(float(np.median(v)), 4),
            "mean_cos_raw": round(float(np.mean(col(name, "cos_raw"))), 4),
            "paired_delta_vs_orig_mean": round(float(np.mean(d)), 4),
            "paired_delta_vs_orig_median": round(float(np.median(d)), 4),
            "paired_delta_ci95": [round(x, 4) for x in boot_ci(d)],
            "frac_delta_negative": round(float(np.mean(d < 0)), 3),
            "sign_test_p_two_sided": round(sign_test_p(d), 6),
            "retention_of_orig_mean": round(
                float(np.mean(v) / np.mean(o)) if np.mean(o) else float("nan"), 4
            ),
            "rows": {str(k): float(x) for k, x in zip(idxs, v)},
        }

    gen_mean = np.array([r["generic_fixed_cos_c_mean"] for r in rows_out])
    summary["__generic_fixed__"] = {
        "n_texts": len(generic_recon),
        "mean_cos_c_over_40_targets": round(float(np.mean(gen_mean)), 4),
        "max_single_pair": round(
            float(np.max([r["generic_fixed_cos_c_max"] for r in rows_out])), 4
        ),
        "note": "fixed generic-text floor for the E1-E7 cohort, measured directly",
    }

    # centered 40-way retrieval for every variant (free, no extra GPU)
    Xc = perp(vecs)
    Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    retrieval = {}
    for name in variant_names:
        if len(recon_store[name]) != 40:
            continue
        P = perp(np.stack(recon_store[name]))
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        S = P @ Xc.T
        diag = np.diag(S)
        rank = (S >= diag[:, None]).sum(axis=1)
        retrieval[name] = {
            "top1": round(float(np.mean(rank == 1)), 4),
            "top5": round(float(np.mean(rank <= 5)), 4),
            "mrr": round(float(np.mean(1.0 / rank)), 4),
        }

    out = {
        "schema_version": 1,
        "experiment": "N1 / C7+B3 AR rescoring of frozen text variants",
        "variants_sha256": digest,
        "protocol": frozen["protocol"],
        "verification": {
            "orig_mean_cos_c": round(float(np.mean(orig_all)), 4),
            "orig_mean_cos_c_expected_from_E5": 0.8593,
            "orig_mean_cos_raw": round(float(np.mean(col("orig", "cos_raw"))), 4),
        },
        "summary_by_variant": summary,
        "retrieval_by_variant": retrieval,
        "rows": rows_out,
        "elapsed_seconds": round(time.time() - t0, 1),
        "n_unique_texts_scored": len(cache),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.vecs_out,
        m_hat=m_hat,
        x=vecs,
        generic_recon=generic_recon,
        **{f"recon_{k}": np.stack(v) for k, v in recon_store.items()},
        **{f"idx_{k}": np.array(recon_idx[k]) for k in recon_store},
    )
    print("\nVERIFY:", json.dumps(out["verification"]))
    for k, v in summary.items():
        if k.startswith("__"):
            continue
        print(
            f"{k:<16} n={v['n']:>2} mean_cos_c={v['mean_cos_c']:+.4f} "
            f"delta={v['paired_delta_vs_orig_mean']:+.4f} "
            f"neg={v['frac_delta_negative']:.2f} ret={v['retention_of_orig_mean']:.3f}"
        )
    print("GENERIC:", json.dumps(summary["__generic_fixed__"]))
    print("RETRIEVAL:", json.dumps(retrieval))
    print(f"C7B3_SCORING_COMPLETE -> {args.out}")


if __name__ == "__main__":
    main()
