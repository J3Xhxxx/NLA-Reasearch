#!/usr/bin/env python3
"""N3 step 5 — LOCAL analysis. Zero GPU.

Answers the two questions F14 left open, using real-corpus evidence instead of
the 24 synthetic documents:

  Q1 (F14.2)  Was the "held-out activation gate" measuring the feature or the
              corpus? 8 of 24 features were dead on the synthetic test split
              (pos_mean = neg_mean = 0, AUC = 0.5 by tie convention) and those 8
              included the three MOST readable directions. If they fire happily
              on real text, the gate was measuring our prompts.

  Q2 (F14.1)  What predicts readability? q+ is bimodal (10 features >= 0.362,
              13 < 0.15). Correlate q+ against real-corpus covariates:
              frequency, source concentration, source entropy, activation
              magnitude, parallel-corpus language purity, and the decoder
              direction's alignment with the dataset mean.

Everything is reported with bootstrap CIs and n, because n=24 supports point
estimates only weakly -- the deliverable here is a properly-powered cohort
(n3_candidate_cohort_v1.json), not a significance claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d else float("nan")


def boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 7) -> list[float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        if np.unique(x[idx]).size < 3 or np.unique(y[idx]).size < 3:
            continue
        vals.append(spearman(x[idx], y[idx]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    if vals.size < 100:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/n3_analysis.json")
    args = ap.parse_args()
    R = Path(args.results)

    ctx = json.load(open(R / "n3_contexts_v1.json", encoding="utf-8"))
    cohort = json.load(open(R / "n3_candidate_cohort_v1.json", encoding="utf-8"))
    recheck = json.load(open(R / "local_recheck_b6b4_opus.json", encoding="utf-8"))
    strat = json.load(open(R / "local_recheck_stratified_opus.json", encoding="utf-8"))
    sel = json.load(open(R / "b6b4_factorial_selection.json", encoding="utf-8"))

    q_plus = {int(r["feature"]): float(r["q_plus"]) for r in recheck["semantic_new_table"]}
    floors = {int(r["feature"]): float(r["generic_floor_signed"])
              for r in recheck["semantic_new_table"]}
    auc = {int(r["feature"]): (None if r["test_auroc"] is None else float(r["test_auroc"]))
           for r in strat["rows"]}
    stratum = {int(r["feature"]): r["stratum"] for r in strat["rows"]}
    align = {int(d["feature"]): d.get("mean_alignment")
             for d in sel["selected_directions"] if int(d["feature"]) >= 0}

    real = {int(d["feature"]): d for d in ctx["directions"]
            if int(d.get("feature", -1)) >= 0 and "n_fire" in d}

    # ---------- Q1: was the synthetic gate measuring the corpus? ----------
    rows = []
    for f, q in sorted(q_plus.items(), key=lambda kv: -kv[1]):
        r = real.get(f, {})
        rows.append({
            "feature": f,
            "label": next((d.get("label") for d in ctx["directions"]
                           if d.get("feature") == f), None),
            "stratum": stratum.get(f),
            "q_plus": q,
            "generic_floor_signed": floors.get(f),
            "synthetic_test_auroc": auc.get(f),
            "synthetic_test_dead": (auc.get(f) is not None
                                    and abs(auc[f] - 0.5) < 1e-9),
            "real_n_fire": r.get("n_fire"),
            "real_freq": r.get("freq"),
            "real_max_act": r.get("max_act"),
            "real_n_sources": r.get("n_sources_fired"),
            "real_top_source": r.get("top_source"),
            "real_top_source_share": r.get("top_source_share"),
            "real_source_entropy": r.get("source_entropy_norm"),
            "real_lang_purity": r.get("lang_purity"),
            "real_lang_top": r.get("lang_top"),
            "real_n_langs": r.get("n_langs_fired"),
            "mean_alignment": align.get(f),
        })

    dead_syn = [r for r in rows if r["synthetic_test_dead"]]
    q1 = {
        "n_features": len(rows),
        "n_synthetic_test_dead": len(dead_syn),
        "synthetic_test_dead_features": [r["feature"] for r in dead_syn],
        "of_those_alive_on_real_text": sum(1 for r in dead_syn if (r["real_n_fire"] or 0) > 0),
        "their_real_freqs": {r["feature"]: r["real_freq"] for r in dead_syn},
        "their_q_plus": {r["feature"]: r["q_plus"] for r in dead_syn},
        "n_dead_on_real_text": sum(1 for r in rows if (r["real_n_fire"] or 0) == 0),
        "dead_on_real_features": [r["feature"] for r in rows if (r["real_n_fire"] or 0) == 0],
        "interpretation_note": (
            "If most synthetic-test-dead features fire on real text, the B6+B4 "
            "gate ranked our 24 documents, not the features -- and F14.3's "
            "conclusion (gate order is wrong) is confirmed on real evidence."
        ),
    }

    # ---------- Q2: what predicts readability? ----------
    q = np.array([r["q_plus"] for r in rows], float)
    covs = {
        "log10_real_freq": np.array([math.log10(r["real_freq"]) if (r["real_freq"] or 0) > 0
                                     else np.nan for r in rows]),
        "real_max_act": np.array([r["real_max_act"] if r["real_max_act"] is not None
                                  else np.nan for r in rows], float),
        "real_n_sources": np.array([r["real_n_sources"] if r["real_n_sources"] is not None
                                    else np.nan for r in rows], float),
        "real_top_source_share": np.array([r["real_top_source_share"]
                                           if r["real_top_source_share"] is not None
                                           else np.nan for r in rows], float),
        "real_source_entropy": np.array([r["real_source_entropy"]
                                         if r["real_source_entropy"] is not None
                                         else np.nan for r in rows], float),
        "real_lang_purity": np.array([r["real_lang_purity"]
                                      if r["real_lang_purity"] is not None
                                      else np.nan for r in rows], float),
        "abs_mean_alignment": np.array([abs(r["mean_alignment"])
                                        if r["mean_alignment"] is not None
                                        else np.nan for r in rows], float),
        "synthetic_test_auroc": np.array([r["synthetic_test_auroc"]
                                          if r["synthetic_test_auroc"] is not None
                                          else np.nan for r in rows], float),
    }
    q2 = {"n": int(len(rows)), "predictors": {}}
    for name, v in covs.items():
        ok = np.isfinite(v) & np.isfinite(q)
        q2["predictors"][name] = {
            "n": int(ok.sum()),
            "spearman_vs_q_plus": spearman(q, v),
            "ci95": boot_ci(q, v),
        }
    # readable vs not, using the 0.117 gap found in F14.1
    hi = q >= 0.362
    q2["bimodal_split"] = {
        "threshold": 0.362,
        "n_high": int(hi.sum()), "n_low": int((~hi).sum()),
        "medians": {name: {"high": float(np.nanmedian(v[hi])),
                           "low": float(np.nanmedian(v[~hi]))}
                    for name, v in covs.items()},
    }

    # ---------- cohort readiness ----------
    per_stratum: dict[str, list[dict]] = {}
    for f in cohort["features"]:
        per_stratum.setdefault(f["stratum"], []).append(f)
    q3 = {
        "pool_sizes": cohort["pool_sizes"],
        "frozen": {k: len(v) for k, v in per_stratum.items()},
        "total_frozen": len(cohort["features"]),
        "median_freq_by_stratum": {k: float(np.median([x["freq"] for x in v]))
                                   for k, v in per_stratum.items()},
        "median_lang_purity_by_stratum": {k: float(np.median([x["lang_purity"] for x in v]))
                                          for k, v in per_stratum.items()},
        "n_tokens_evidence": cohort["n_tokens"],
    }

    out = {
        "schema_version": 1,
        "experiment": "N3 analysis: real-corpus evidence vs synthetic gate, and readability predictors",
        "inputs": {
            "n3_contexts_v1.json": ctx.get("stats_sha256"),
            "n_tokens": ctx["n_tokens"], "sae": ctx["sae"],
        },
        "q1_was_the_gate_measuring_the_corpus": q1,
        "q2_what_predicts_readability": q2,
        "q3_cohort_readiness": q3,
        "per_feature": rows,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Q1: synthetic gate vs real text ===")
    print(f"  synthetic-test-dead features : {q1['n_synthetic_test_dead']}/{q1['n_features']}")
    print(f"  of those, alive on real text : {q1['of_those_alive_on_real_text']}")
    print(f"  dead even on real text       : {q1['n_dead_on_real_text']} {q1['dead_on_real_features']}")
    print("\n=== Q2: predictors of q+ (n=%d) ===" % q2["n"])
    for name, d in sorted(q2["predictors"].items(),
                          key=lambda kv: -abs(kv[1]["spearman_vs_q_plus"] or 0)):
        ci = d["ci95"]
        print(f"  {name:26s} rho={d['spearman_vs_q_plus']:+.3f}  "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}]  n={d['n']}")
    print("\n=== Q3: cohort ===")
    print(f"  pools  {q3['pool_sizes']}")
    print(f"  frozen {q3['frozen']} (total {q3['total_frozen']})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
