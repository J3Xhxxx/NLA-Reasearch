#!/usr/bin/env python3
"""Stratified re-analysis of the F9 headline "heldout AUC vs q+ rho = -0.015".

The pooled correlation mixes the domain and language strata, which have opposite
selection yields (6/15 vs 8/9) and opposite conditional q+ levels. This script
recomputes the association within stratum and reports the bimodality of q+.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parent.parent / "results"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom else float("nan")


def find_auc(node: dict) -> float | None:
    for key in ("test_auroc", "auroc", "auc", "test_auc"):
        if key in node and isinstance(node[key], (int, float)):
            return float(node[key])
    return None


def main() -> None:
    sel = json.load(open(RES / "b6b4_factorial_selection.json", encoding="utf-8"))
    res = json.load(open(RES / "b6b4_factorial_result.json", encoding="utf-8"))

    rows = [
        r
        for r in res["polarity_rows"]
        if r["sample_index"] == 0 and r["group"] == "semantic_new"
    ]
    q = {int(r["feature"]): float(r["q_plus"]) for r in rows}
    lab = {int(r["feature"]): r["label"] for r in rows}

    recs = []
    for d in sel["selected_directions"]:
        f = int(d["feature"])
        if f not in q:
            continue
        test = d.get("test") or {}
        train = d.get("train") or {}
        recs.append(
            {
                "feature": f,
                "label": lab[f],
                "stratum": lab[f].split(":")[0],
                "q_plus": q[f],
                "test_auroc": find_auc(test),
                "train_auroc": find_auc(train),
                "test_keys": sorted(test.keys())[:12],
            }
        )

    out: dict = {"n": len(recs), "example_test_keys": recs[0]["test_keys"] if recs else []}
    have = [r for r in recs if r["test_auroc"] is not None]
    out["n_with_test_auroc"] = len(have)

    if have:
        qa = np.array([r["q_plus"] for r in have])
        aa = np.array([r["test_auroc"] for r in have])
        st = np.array([r["stratum"] for r in have])
        out["pooled_spearman_q_plus_vs_test_auroc"] = spearman(qa, aa)
        for s in sorted(set(st)):
            mask = st == s
            out[f"spearman_within_{s}"] = {
                "n": int(mask.sum()),
                "rho": spearman(qa[mask], aa[mask]),
                "median_q_plus": float(np.median(qa[mask])),
                "median_test_auroc": float(np.median(aa[mask])),
            }

    qall = np.array([r["q_plus"] for r in recs])
    out["q_plus_distribution"] = {
        "sorted": [round(float(v), 4) for v in np.sort(qall)[::-1]],
        "median": float(np.median(qall)),
        "n_above_0.3": int(np.sum(qall > 0.3)),
        "n_above_0.4": int(np.sum(qall > 0.4)),
        "n_below_0.15": int(np.sum(qall < 0.15)),
        "gap_between_10th_and_11th_largest": float(
            np.sort(qall)[::-1][9] - np.sort(qall)[::-1][10]
        ),
    }
    out["rows"] = sorted(recs, key=lambda r: -r["q_plus"])

    (RES / "local_recheck_stratified_opus.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    for r in out["rows"]:
        print(
            f"{r['feature']:>6} {r['stratum']:<9} {r['label']:<18} q+={r['q_plus']:+.4f} "
            f"testAUROC={r['test_auroc']}"
        )


if __name__ == "__main__":
    main()
