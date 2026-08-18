#!/usr/bin/env python3
"""Local zero-GPU re-verification of E1-E7 / B2 headline numbers.

Adds two baselines that the existing reports do not contain:
  1. inter-sample centered cosine between distinct real activations
     (the "predict some other activation" floor for F1),
  2. scale-normalized retrieval margin (d-prime style) for B2, to test whether
     the reported raw-margin advantage survives per-method scale normalization.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parent.parent / "results"


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def perp(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    return a - np.outer(a @ m, m)


def rowcos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(unit(a) * unit(b), axis=1)


def main() -> None:
    z = np.load(RES / "recon_vectors.npz")
    out: dict = {"keys": sorted(z.files)}
    x = z["x"].astype(np.float64)
    m = z["m_hat"].astype(np.float64)
    m = m / np.linalg.norm(m)
    xc = perp(x, m)

    preds = {
        "nla": z["pred_full"],
        "sae_small": z["recon_sae_small"],
        "sae_big": z["recon_sae_big"],
        "resid_text": z["pred_resid"],
    }

    out["head_to_head_centered"] = {
        k: float(np.mean(rowcos(perp(v.astype(np.float64), m), xc)))
        for k, v in preds.items()
    }
    out["head_to_head_raw"] = {
        k: float(np.mean(rowcos(v.astype(np.float64), x))) for k, v in preds.items()
    }

    # ---- baseline 1: distinct real activations against each other ----
    Xc = unit(xc)
    S_xx = Xc @ Xc.T
    off = ~np.eye(40, dtype=bool)
    doc = np.arange(40) // 8
    same_doc = (doc[:, None] == doc[None, :]) & off
    diff_doc = (doc[:, None] != doc[None, :])
    out["activation_activation_centered_cos"] = {
        "mean_all_offdiag": float(S_xx[off].mean()),
        "median_all_offdiag": float(np.median(S_xx[off])),
        "p95_all_offdiag": float(np.percentile(S_xx[off], 95)),
        "mean_same_document": float(S_xx[same_doc].mean()),
        "mean_diff_document": float(S_xx[diff_doc].mean()),
        "mean_best_other_activation": float(
            np.mean([S_xx[i][off[i]].max() for i in range(40)])
        ),
    }
    # nearest-other-activation predictor: how well does the single best OTHER
    # real activation predict x_i? (an oracle non-parametric baseline)
    out["oracle_nearest_other_activation_cos"] = out[
        "activation_activation_centered_cos"
    ]["mean_best_other_activation"]

    # ---- baseline 2: B2 margins, raw vs normalized ----
    b2 = {}
    for k, v in preds.items():
        P = unit(perp(v.astype(np.float64), m))
        S = P @ Xc.T  # 40 x 40, row = prediction, col = target
        diagv = np.diag(S).copy()
        Soff = S.copy()
        np.fill_diagonal(Soff, -np.inf)
        best_other = Soff.max(axis=1)
        Sfin = S.copy()
        np.fill_diagonal(Sfin, np.nan)
        mu = np.nanmean(Sfin, axis=1)
        sd = np.nanstd(Sfin, axis=1, ddof=1)
        rank = (S >= diagv[:, None]).sum(axis=1)  # 1 = correct top-1
        b2[k] = {
            "top1": float(np.mean(rank == 1)),
            "mrr": float(np.mean(1.0 / rank)),
            "raw_margin_mean": float(np.mean(diagv - best_other)),
            "z_margin_mean": float(np.mean((diagv - mu) / sd)),
            "z_margin_median": float(np.median((diagv - mu) / sd)),
            "z_best_other_mean": float(np.mean((best_other - mu) / sd)),
            "offdiag_sd_mean": float(np.mean(sd)),
            "diag_mean": float(np.mean(diagv)),
            "best_other_mean": float(np.mean(best_other)),
        }
    out["b2_recheck"] = b2
    out["b2_paired_z_margin_delta"] = {
        f"nla_minus_{k}": float(
            np.mean(
                (
                    (lambda A: (np.diag(A) - np.nanmean(np.where(np.eye(40, dtype=bool), np.nan, A), axis=1)) / np.nanstd(np.where(np.eye(40, dtype=bool), np.nan, A), axis=1, ddof=1))(
                        unit(perp(preds["nla"].astype(np.float64), m)) @ Xc.T
                    )
                )
                - (
                    (lambda A: (np.diag(A) - np.nanmean(np.where(np.eye(40, dtype=bool), np.nan, A), axis=1)) / np.nanstd(np.where(np.eye(40, dtype=bool), np.nan, A), axis=1, ddof=1))(
                        unit(perp(preds[k].astype(np.float64), m)) @ Xc.T
                    )
                )
            )
        )
        for k in ("sae_small", "sae_big")
    }

    # ---- centered FVE-style score with a single optimal global scale ----
    fve = {}
    for k, v in preds.items():
        P = perp(v.astype(np.float64), m)
        s = float(np.sum(P * xc) / np.sum(P * P))
        resid = xc - s * P
        fve[k] = {
            "opt_scale": s,
            "fve_centered": float(1 - np.sum(resid**2) / np.sum(xc**2)),
        }
    # mean-predictor / zero-predictor is 0 by construction after centering
    fve["oracle_nearest_other_activation"] = {
        "fve_centered": float(
            np.mean(
                [
                    1
                    - np.min(
                        [
                            np.sum(
                                (
                                    xc[i]
                                    - (np.dot(xc[i], xc[j]) / np.dot(xc[j], xc[j])) * xc[j]
                                )
                                ** 2
                            )
                            / np.sum(xc[i] ** 2)
                            for j in range(40)
                            if j != i
                        ]
                    )
                    for i in range(40)
                ]
            )
        )
    }
    out["centered_fve_optimal_scale"] = fve

    (RES / "local_recheck_opus.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2)[:6000])


if __name__ == "__main__":
    main()
