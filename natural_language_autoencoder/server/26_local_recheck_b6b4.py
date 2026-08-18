#!/usr/bin/env python3
"""Local zero-GPU re-verification of the B6+B4 (F9) primary numbers.

The stored assets contain `generic_direction_similarity` (8 generic texts x 45
directions), so the per-direction generic floor can be subtracted from q+ for
each feature instead of only comparing group means. That matched contrast is
what the F9 claim ("weak but systematic readable direction signal") needs.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parent.parent / "results"


def median_ci(vals: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    meds = np.median(vals[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def sign_test_p(vals: np.ndarray) -> float:
    from math import comb

    n = int(np.sum(vals != 0))
    k = int(np.sum(vals > 0))
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def main() -> None:
    z = np.load(RES / "b6b4_factorial_recon_vectors.npz", allow_pickle=True)
    res = json.load(open(RES / "b6b4_factorial_result.json", encoding="utf-8"))

    gen = z["generic_direction_similarity"]  # 8 x 45, centered cos
    groups = z["direction_groups"]
    labels = z["direction_labels"]
    dids = z["direction_ids"]

    rows = [r for r in res["polarity_rows"] if r["sample_index"] == 0]
    by_group: dict[str, list] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)

    out: dict = {}

    # per-direction generic floor, signed and absolute
    floor_signed = gen.mean(axis=0)
    floor_abs = np.abs(gen).mean(axis=0)
    floor_max = np.abs(gen).max(axis=0)

    summary = {}
    for g, rs in by_group.items():
        qp = np.array([r["q_plus"] for r in rs])
        rm = np.array([r["r_minus"] for r in rs])
        di = np.array([r["direction_index"] for r in rs])
        adj_signed = qp - floor_signed[di]
        adj_abs = qp - floor_abs[di]
        adj_max = qp - floor_max[di]
        summary[g] = {
            "n": len(rs),
            "q_plus_median": float(np.median(qp)),
            "r_minus_median": float(np.median(rm)),
            "generic_floor_signed_median": float(np.median(floor_signed[di])),
            "generic_floor_abs_median": float(np.median(floor_abs[di])),
            "q_plus_minus_generic_signed_median": float(np.median(adj_signed)),
            "q_plus_minus_generic_signed_ci": median_ci(adj_signed),
            "q_plus_minus_generic_signed_pos_frac": float(np.mean(adj_signed > 0)),
            "q_plus_minus_generic_abs_median": float(np.median(adj_abs)),
            "q_plus_minus_generic_abs_pos_frac": float(np.mean(adj_abs > 0)),
            "q_plus_gt_worstcase_generic_frac": float(np.mean(adj_max > 0)),
            "sign_test_p_adj_signed": sign_test_p(adj_signed)
            if len(rs) > 1
            else None,
        }
    out["group_summary_generic_adjusted"] = summary

    # does the per-direction generic floor predict q+? (residual mean-type confound)
    sem = by_group["semantic_new"]
    qp = np.array([r["q_plus"] for r in sem])
    di = np.array([r["direction_index"] for r in sem])

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))

    out["confound_checks_semantic_new"] = {
        "spearman_q_plus_vs_generic_floor_abs": spearman(qp, floor_abs[di]),
        "spearman_q_plus_vs_generic_floor_signed": spearman(qp, floor_signed[di]),
    }

    # how far are the 45 directions from mutually orthogonal? (retrieval difficulty)
    D = z["directions"].astype(np.float64)
    m = z["m_hat"].astype(np.float64)
    m = m / np.linalg.norm(m)
    Dc = D - np.outer(D @ m, m)
    Dc /= np.linalg.norm(Dc, axis=1, keepdims=True)
    G = Dc @ Dc.T
    offd = G[~np.eye(45, dtype=bool)]
    sem_idx = np.where(groups == "semantic_new")[0]
    Gs = G[np.ix_(sem_idx, sem_idx)]
    out["direction_geometry"] = {
        "mean_abs_offdiag_cos_all45": float(np.abs(offd).mean()),
        "max_abs_offdiag_cos_all45": float(np.abs(offd).max()),
        "mean_abs_offdiag_cos_semantic_new": float(
            np.abs(Gs[~np.eye(len(sem_idx), dtype=bool)]).mean()
        ),
        "mean_abs_alignment_with_m_hat": float(np.abs(D @ m / np.linalg.norm(D, axis=1)).mean()),
        "max_abs_alignment_with_m_hat": float(np.abs(D @ m / np.linalg.norm(D, axis=1)).max()),
    }

    # per-feature table for the semantic_new cohort
    tbl = []
    for r in sem:
        i = r["direction_index"]
        tbl.append(
            {
                "feature": r["feature"],
                "label": r["label"],
                "q_plus": round(r["q_plus"], 4),
                "generic_floor_signed": round(float(floor_signed[i]), 4),
                "generic_floor_abs": round(float(floor_abs[i]), 4),
                "q_plus_minus_floor_signed": round(
                    r["q_plus"] - float(floor_signed[i]), 4
                ),
                "beats_worstcase_generic": bool(r["q_plus"] > float(floor_max[i])),
                "plus_feature_rank": r["plus_feature_rank"],
            }
        )
    tbl.sort(key=lambda d: -d["q_plus"])
    out["semantic_new_table"] = tbl

    (RES / "local_recheck_b6b4_opus.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "semantic_new_table"}, indent=2))
    print("\nTOP/BOTTOM semantic_new rows:")
    for r in tbl[:8] + tbl[-4:]:
        print(r)


if __name__ == "__main__":
    main()
