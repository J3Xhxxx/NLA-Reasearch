#!/usr/bin/env python3
"""N4 document-clustered analysis for the preregistered real-content run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np


SEED = 20260730
N_BOOT = 20000


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def perp(a: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a - float(a @ m_hat) * m_hat
    return a - np.outer(a @ m_hat, m_hat)


def row_cos(p: np.ndarray, x: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    pp, xx = perp(p, m_hat), perp(x, m_hat)
    den = np.linalg.norm(pp, axis=1) * np.linalg.norm(xx, axis=1)
    return np.sum(pp * xx, axis=1) / np.maximum(den, 1e-12)


def row_cos_raw(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(x, axis=1)
    return np.sum(p * x, axis=1) / np.maximum(den, 1e-12)


def lodo_cos(
    p: np.ndarray, x: np.ndarray, docs: np.ndarray, m_hat_unused=None
) -> np.ndarray:
    del m_hat_unused
    out = np.empty(len(x), dtype=np.float64)
    total = x.sum(0, dtype=np.float64)
    for document in np.unique(docs):
        test = docs == document
        train_n = int((~test).sum())
        if train_n == 0:
            raise ValueError("leave-one-document-out mean has no training rows")
        mean_direction = unit(
            (total - x[test].sum(0, dtype=np.float64)) / train_n
        )
        out[test] = row_cos(p[test], x[test], mean_direction)
    return out


def validate_reconstruction_pair(recon: dict, vecs_path: Path) -> dict:
    """Prove that the reconstruction JSON scores came from this exact NPZ."""
    rows = recon["rows"]
    key_map = {
        "orig": "pred_orig",
        "p1_only": "pred_p1_only",
        "p2_only": "pred_p2_only",
        "p3_only": "pred_p3_only",
        "p12": "pred_p12",
        "quote_strip_p2": "pred_quote_strip_p2",
        "quote_strip_p3": "pred_quote_strip_p3",
        "quote_strip_all": "pred_quote_strip_all",
        "word_shuffle": "pred_word_shuffle",
        "sae_small": "recon_sae_small",
        "sae_big": "recon_sae_big",
    }
    max_error = {"cos_c": 0.0, "cos_c_lodo": 0.0, "cos_raw": 0.0}
    with np.load(vecs_path, allow_pickle=False) as archive:
        required = {
            "x", "m_hat", "doc_ids", "positions", "generic_recon", *key_map.values()
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"reconstruction NPZ lacks keys: {missing}")
        x = np.asarray(archive["x"], dtype=np.float32)
        docs = np.asarray(archive["doc_ids"], dtype=np.int64)
        positions = np.asarray(archive["positions"], dtype=np.int64)
        if x.shape != (len(rows), 3840):
            raise ValueError(f"unexpected reconstruction x shape: {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("reconstruction NPZ x contains non-finite values")
        if not np.array_equal(docs, [int(row["doc_id"]) for row in rows]):
            raise ValueError("reconstruction NPZ doc_ids do not match JSON rows")
        if not np.array_equal(
            positions, [int(row["position"]) for row in rows]
        ):
            raise ValueError("reconstruction NPZ positions do not match JSON rows")

        m_hat = unit(x.mean(0, dtype=np.float64))
        stored_m_hat = np.asarray(archive["m_hat"], dtype=np.float32)
        m_hat_max_abs = float(np.max(np.abs(stored_m_hat - m_hat)))
        if m_hat_max_abs > 1e-6:
            raise ValueError(
                f"stored mean direction differs from x-derived value: {m_hat_max_abs}"
            )

        for name, key in key_map.items():
            pred = np.asarray(archive[key], dtype=np.float32)
            if pred.shape != x.shape or not np.isfinite(pred).all():
                raise ValueError(f"{key} has invalid shape or non-finite values")
            recomputed = {
                "cos_c": row_cos(pred, x, m_hat),
                "cos_c_lodo": lodo_cos(pred, x, docs),
                "cos_raw": row_cos_raw(pred, x),
            }
            for field, values in recomputed.items():
                reported = np.asarray(
                    [float(row["scores"][name][field]) for row in rows]
                )
                error = float(np.max(np.abs(values - reported)))
                max_error[field] = max(max_error[field], error)
                if not np.allclose(
                    values, reported, rtol=1e-7, atol=5e-7
                ):
                    raise ValueError(
                        f"reconstruction JSON/NPZ mismatch for {name}.{field}: "
                        f"max_abs={error}"
                    )

        generic = np.asarray(archive["generic_recon"], dtype=np.float32)
        if generic.ndim != 2 or generic.shape[1] != x.shape[1]:
            raise ValueError(f"invalid generic reconstruction shape: {generic.shape}")
        generic_by_target = np.stack(
            [
                row_cos(np.repeat(vector[None], len(x), axis=0), x, m_hat)
                for vector in generic
            ]
        ).mean(0)
        reported_generic = np.asarray(
            [float(row["generic_floor_cos_c"]) for row in rows]
        )
        generic_error = float(
            np.max(np.abs(generic_by_target - reported_generic))
        )
        if not np.allclose(
            generic_by_target, reported_generic, rtol=1e-7, atol=5e-7
        ):
            raise ValueError(
                "reconstruction JSON/NPZ generic-floor mismatch: "
                f"max_abs={generic_error}"
            )
    return {
        "json_npz_semantic_match": True,
        "score_max_abs_error": max_error,
        "generic_floor_max_abs_error": generic_error,
        "m_hat_max_abs_error": m_hat_max_abs,
    }


def average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else None


def document_values(
    rows: list[dict], value: Callable[[dict], float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_doc: dict[int, list[float]] = defaultdict(list)
    corpus: dict[int, str] = {}
    for row in rows:
        d = int(row["doc_id"])
        by_doc[d].append(float(value(row)))
        corpus.setdefault(d, str(row["corpus"]))
        if corpus[d] != str(row["corpus"]):
            raise ValueError(f"doc {d} appears in multiple corpora")
    docs = np.array(sorted(by_doc), dtype=np.int64)
    vals = np.array([np.mean(by_doc[int(d)]) for d in docs], dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("document statistic contains non-finite values")
    strata = np.array([corpus[int(d)] for d in docs])
    return docs, vals, strata


def stratified_boot_mean(
    vals: np.ndarray,
    strata: np.ndarray,
    level: float = 0.95,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> list[float]:
    vals = np.asarray(vals, float)
    strata = np.asarray(strata)
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=float)
    for s in sorted(set(strata.tolist())):
        v = vals[strata == s]
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        boot += v[idx].sum(1) / len(vals)
    alpha = (1.0 - level) / 2.0
    return [
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1.0 - alpha)),
    ]


def doc_summary(rows, value, seed_offset: int = 0) -> dict:
    docs, vals, strata = document_values(rows, value)
    return {
        "n_docs": len(docs),
        "mean_equal_doc": float(vals.mean()),
        "median_doc": float(np.median(vals)),
        "ci95_stratified_document_bootstrap": stratified_boot_mean(
            vals, strata, seed=SEED + seed_offset
        ),
        "n_docs_positive": int((vals > 0).sum()),
        "n_docs_negative": int((vals < 0).sum()),
        "per_document": {str(d): float(v) for d, v in zip(docs, vals)},
    }


def recovered(row: dict, condition: str, field: str = "kl_at_pos") -> float:
    zero = float(row["results"]["zero"][field])
    val = float(row["results"][condition][field])
    return 1.0 - val / max(zero, 1e-6)


def subset_summary(rows: list[dict], condition: str) -> dict:
    out = {}
    for field in ("corpus", "source", "lang"):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(field))].append(r)
        out[field] = {}
        for group, rr in sorted(groups.items()):
            n_docs = len({int(r["doc_id"]) for r in rr})
            if n_docs < 3:
                continue
            _, vals, _ = document_values(rr, lambda r: recovered(r, condition))
            out[field][group] = {
                "n_rows": len(rr),
                "n_docs": n_docs,
                "kl_recovered_mean_equal_doc": float(vals.mean()),
            }
    return out


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def fmt_num(value, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True, type=Path)
    ap.add_argument("--vecs", required=True, type=Path)
    ap.add_argument("--causal", required=True, type=Path)
    ap.add_argument("--prereg", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--markdown", required=True, type=Path)
    args = ap.parse_args()

    recon = json.loads(args.recon.read_text(encoding="utf-8"))
    causal = json.loads(args.causal.read_text(encoding="utf-8"))
    vecs_sha256 = sha256_file(args.vecs)
    prereg_sha256 = sha256_file(args.prereg)
    if causal["inputs"]["recon_sha256"] != vecs_sha256:
        raise ValueError("causal result does not reference the supplied reconstruction NPZ")
    if causal["inputs"]["prereg_sha256"] != prereg_sha256:
        raise ValueError("causal result preregistration hash mismatch")
    if recon["inputs"]["prereg_sha256"] != prereg_sha256:
        raise ValueError("reconstruction result preregistration hash mismatch")
    if (
        causal["inputs"]["activations_sha256"]
        != recon["inputs"]["activations_sha256"]
    ):
        raise ValueError("causal/reconstruction activation hashes differ")
    pair_qa = validate_reconstruction_pair(recon, args.vecs)
    rrows = recon["rows"]
    crows = causal["rows"]
    if len(rrows) != 200 or len(crows) != 200:
        raise ValueError(f"full N4 requires 200+200 rows, got {len(rrows)}+{len(crows)}")
    if [r["idx"] for r in rrows] != [r["idx"] for r in crows]:
        raise ValueError("reconstruction and causal row ordering differs")
    for a, b in zip(rrows, crows):
        if (a["doc_id"], a["position"]) != (b["doc_id"], b["position"]):
            raise ValueError(f"row provenance mismatch at idx={a['idx']}")

    # ---------- H1: cosine-side channel localization ----------
    def rscore(row, name, field="cos_c"):
        return float(row["scores"][name][field])

    h1_condition = {}
    for name in ("orig", "p3_only", "p12", "p1_only", "p2_only",
                 "quote_strip_p3", "quote_strip_p2", "quote_strip_all",
                 "word_shuffle", "sae_small", "sae_big"):
        h1_condition[name] = doc_summary(
            rrows, lambda r, n=name: rscore(r, n), seed_offset=len(h1_condition)
        )
        if name in recon["summary"]:
            h1_condition[name]["retrieval"] = recon["summary"][name]["retrieval"]

    floor = doc_summary(
        rrows, lambda r: float(r["generic_floor_cos_c"]), seed_offset=20
    )
    orig_mean = h1_condition["orig"]["mean_equal_doc"]
    floor_mean = floor["mean_equal_doc"]
    h1_testable = bool(orig_mean > floor_mean)
    shares = {
        name: (
            (v["mean_equal_doc"] - floor_mean) / (orig_mean - floor_mean)
            if h1_testable else None
        )
        for name, v in h1_condition.items()
        if name not in ("sae_small", "sae_big")
    }
    p3_minus_p12 = doc_summary(
        rrows,
        lambda r: rscore(r, "p3_only") - rscore(r, "p12"),
        seed_offset=21,
    )
    h1_pass = bool(
        h1_testable
        and shares["p3_only"] is not None
        and shares["p3_only"] >= 0.80
        and shares["p12"] is not None
        and shares["p12"] <= 0.50
        and p3_minus_p12["ci95_stratified_document_bootstrap"][0] > 0
    )

    lodo_p3_minus_p12 = doc_summary(
        rrows,
        lambda r: rscore(r, "p3_only", "cos_c_lodo")
        - rscore(r, "p12", "cos_c_lodo"),
        seed_offset=22,
    )

    # ---------- H2/H3: causal KL ----------
    conditions = list(crows[0]["results"])
    causal_condition = {}
    for i, name in enumerate(conditions):
        causal_condition[name] = {
            "kl_at_pos": doc_summary(
                crows,
                lambda r, n=name: float(r["results"][n]["kl_at_pos"]),
                seed_offset=30 + i,
            ),
            "kl_recovered_at_pos": doc_summary(
                crows, lambda r, n=name: recovered(r, n), seed_offset=50 + i
            ),
            "kl_recovered_first16": doc_summary(
                crows,
                lambda r, n=name: recovered(r, n, "kl_mean_first16"),
                seed_offset=70 + i,
            ),
            "kl_mean_first16": doc_summary(
                crows,
                lambda r, n=name: float(
                    r["results"][n]["kl_mean_first16"]
                ),
                seed_offset=80 + i,
            ),
            "ce_first16": doc_summary(
                crows,
                lambda r, n=name: float(r["results"][n]["ce_first16"]),
                seed_offset=90 + i,
            ),
            "ce_delta_from_clean_first16": doc_summary(
                crows,
                lambda r, n=name: float(r["results"][n]["ce_first16"])
                - float(r["ce_clean_first16"]),
                seed_offset=120 + i,
            ),
        }

    orig_vs_big = doc_summary(
        crows,
        lambda r: recovered(r, "orig") - recovered(r, "sae_big"),
        seed_offset=100,
    )
    docs, vals, strata = document_values(
        crows, lambda r: recovered(r, "orig") - recovered(r, "sae_big")
    )
    orig_vs_big["ci90_stratified_document_bootstrap"] = stratified_boot_mean(
        vals, strata, level=0.90, seed=SEED + 101
    )
    orig_vs_small = doc_summary(
        crows,
        lambda r: recovered(r, "orig") - recovered(r, "sae_small"),
        seed_offset=102,
    )
    raw_kl_orig_vs_big = doc_summary(
        crows,
        lambda r: float(r["results"]["sae_big"]["kl_at_pos"])
        - float(r["results"]["orig"]["kl_at_pos"]),
        seed_offset=103,
    )
    raw_kl_orig_vs_small = doc_summary(
        crows,
        lambda r: float(r["results"]["sae_small"]["kl_at_pos"])
        - float(r["results"]["orig"]["kl_at_pos"]),
        seed_offset=104,
    )
    raw_kl16_orig_vs_big = doc_summary(
        crows,
        lambda r: float(r["results"]["sae_big"]["kl_mean_first16"])
        - float(r["results"]["orig"]["kl_mean_first16"]),
        seed_offset=105,
    )
    raw_kl16_orig_vs_small = doc_summary(
        crows,
        lambda r: float(r["results"]["sae_small"]["kl_mean_first16"])
        - float(r["results"]["orig"]["kl_mean_first16"]),
        seed_offset=106,
    )
    ce16_orig_vs_big = doc_summary(
        crows,
        lambda r: float(r["results"]["sae_big"]["ce_first16"])
        - float(r["results"]["orig"]["ce_first16"]),
        seed_offset=107,
    )
    ce16_orig_vs_small = doc_summary(
        crows,
        lambda r: float(r["results"]["sae_small"]["ce_first16"])
        - float(r["results"]["orig"]["ce_first16"]),
        seed_offset=108,
    )
    ci90 = orig_vs_big["ci90_stratified_document_bootstrap"]
    h2_equiv_big = bool(ci90[0] > -0.05 and ci90[1] < 0.05)
    h2_superior_small = bool(
        orig_vs_small["ci95_stratified_document_bootstrap"][0] > 0
    )
    h2_pass = h2_equiv_big and h2_superior_small

    p3_vs_p12 = doc_summary(
        crows,
        lambda r: recovered(r, "p3_only") - recovered(r, "p12"),
        seed_offset=110,
    )
    r_orig = causal_condition["orig"]["kl_recovered_at_pos"]["mean_equal_doc"]
    r_p3 = causal_condition["p3_only"]["kl_recovered_at_pos"]["mean_equal_doc"]
    h3_testable = bool(r_orig > 0)
    p3_retention = r_p3 / r_orig if h3_testable else None
    h3_pass = bool(
        h3_testable
        and p3_vs_p12["ci95_stratified_document_bootstrap"][0] > 0
        and r_p3 >= 0.75 * r_orig
    )

    # Cosine-vs-causal associations are diagnostic only and use average ties.
    causal_by_idx = {int(r["idx"]): r for r in crows}
    associations = {}
    for name in ("orig", "sae_small", "sae_big", "p3_only", "p12",
                 "quote_strip_p3"):
        cos = np.array([rscore(r, name) for r in rrows])
        kl = np.array(
            [causal_by_idx[int(r["idx"])]["results"][name]["kl_at_pos"]
             for r in rrows]
        )
        associations[name] = {
            "spearman_centered_cos_vs_kl": spearman(cos, kl),
            "spearman_centered_cos_vs_kl_recovered": spearman(
                cos,
                [recovered(causal_by_idx[int(r["idx"])], name) for r in rrows],
            ),
        }

    ce_clean = np.array([float(r["ce_clean_first16"]) for r in crows])
    qa = {
        **causal["qa"],
        **pair_qa,
        "clean_ce_first16_mean": float(ce_clean.mean()),
        "clean_ce_first16_median": float(np.median(ce_clean)),
        "clean_ce_first16_p95": float(np.percentile(ce_clean, 95)),
        "n_rows_zero_kl_below_1e-6": int(sum(
            float(r["results"]["zero"]["kl_at_pos"]) < 1e-6 for r in crows
        )),
        "n_rows": len(crows),
        "n_docs": len({int(r["doc_id"]) for r in crows}),
        "by_corpus_rows": dict(Counter(str(r["corpus"]) for r in crows)),
    }

    out = {
        "schema_version": 1,
        "experiment": "N4 preregistered real-content clustered analysis",
        "inputs": {
            "recon_sha256": sha256_file(args.recon),
            "vecs_sha256": vecs_sha256,
            "causal_sha256": sha256_file(args.causal),
            "prereg_sha256": prereg_sha256,
            "script_sha256": sha256_file(__file__),
        },
        "analysis_protocol": {
            "independent_unit": "document; row metrics averaged within document",
            "bootstrap": (
                f"{N_BOOT} resamples, stratified by pile/xnli, seed={SEED}"
            ),
            "spearman": "average ranks for ties",
            "itt": "all 200 rows / 101 documents",
        },
        "qa": qa,
        "h1_channel_localization": {
            "pass": h1_pass,
            "testable": h1_testable,
            "generic_floor": floor,
            "condition": h1_condition,
            "share_above_generic_floor": shares,
            "p3_minus_p12": p3_minus_p12,
            "lodo_p3_minus_p12": lodo_p3_minus_p12,
            "gates": {
                "orig_above_generic_floor": h1_testable,
                "p3_share_ge_0.80": (
                    shares["p3_only"] is not None and shares["p3_only"] >= 0.80
                ),
                "p12_share_le_0.50": (
                    shares["p12"] is not None and shares["p12"] <= 0.50
                ),
                "p3_minus_p12_ci95_above_zero": (
                    p3_minus_p12["ci95_stratified_document_bootstrap"][0] > 0
                ),
            },
        },
        "h2_causal_ranking": {
            "pass": h2_pass,
            "condition": causal_condition,
            "orig_minus_sae_big_recovered": orig_vs_big,
            "orig_minus_sae_small_recovered": orig_vs_small,
            "raw_kl_sae_big_minus_orig": raw_kl_orig_vs_big,
            "raw_kl_sae_small_minus_orig": raw_kl_orig_vs_small,
            "raw_kl16_sae_big_minus_orig": raw_kl16_orig_vs_big,
            "raw_kl16_sae_small_minus_orig": raw_kl16_orig_vs_small,
            "ce16_sae_big_minus_orig": ce16_orig_vs_big,
            "ce16_sae_small_minus_orig": ce16_orig_vs_small,
            "gates": {
                "equivalent_to_sae_big_margin_0.05": h2_equiv_big,
                "superior_to_sae_small_ci95_above_zero": h2_superior_small,
            },
        },
        "h3_causal_channel": {
            "pass": h3_pass,
            "testable": h3_testable,
            "p3_minus_p12_recovered": p3_vs_p12,
            "p3_retention_of_orig_recovered": p3_retention,
            "gates": {
                "orig_recovered_positive": h3_testable,
                "p3_minus_p12_ci95_above_zero": (
                    p3_vs_p12["ci95_stratified_document_bootstrap"][0] > 0
                ),
                "p3_retention_ge_0.75": (
                    p3_retention is not None and p3_retention >= 0.75
                ),
            },
        },
        "associations_diagnostic": associations,
        "subgroups_descriptive": {
            name: subset_summary(crows, name)
            for name in ("orig", "sae_small", "sae_big", "p3_only", "p12")
        },
        "decision": (
            "H1+H2 survive: proceed to corrected held-out 100+ feature benchmark"
            if h1_pass and h2_pass
            else "H1 survives, H2 fails: keep channel mechanism; drop causal parity"
            if h1_pass
            else "H2 survives, H1 fails: keep causal parity; old paragraph mechanism fails"
            if h2_pass
            else "H1+H2 fail: restrict F11/F12 to the original contaminated pilot"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    out_sha = sha256_file(args.out)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{out_sha}  {args.out.name}\n", encoding="utf-8"
    )

    h1 = out["h1_channel_localization"]
    h2 = out["h2_causal_ranking"]
    h3 = out["h3_causal_channel"]
    md = [
        "# N4 real-content replication — results",
        "",
        f"- Cohort: **{qa['n_rows']} rows / {qa['n_docs']} documents**, all content tokens.",
        f"- Provenance bit-exact: **{qa['provenance_all_bit_exact']}**; "
        f"identity max KL: **{qa['identity_kl_at_pos_max']:.3g}**.",
        f"- Reconstruction JSON/NPZ semantic closure: "
        f"**{qa['json_npz_semantic_match']}**.",
        f"- H1 channel localization: **{'PASS' if h1_pass else 'FAIL'}**.",
        f"- H2 causal ranking: **{'PASS' if h2_pass else 'FAIL'}**.",
        f"- H3 causal channel: **{'PASS' if h3_pass else 'FAIL'}**.",
        "",
        "## H1 — centered-cosine channel localization",
        "",
        "| Condition | Equal-doc mean cos | Share above generic | Retrieval Top-1 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("orig", "p3_only", "p12", "quote_strip_p3",
                 "p1_only", "p2_only", "word_shuffle"):
        v = h1_condition[name]
        md.append(
            f"| {name} | {v['mean_equal_doc']:+.4f} | {fmt_num(shares[name])} | "
            f"{v['retrieval']['top1']:.3f} |"
        )
    md += [
        "",
        f"`p3_only - p12` = {p3_minus_p12['mean_equal_doc']:+.4f}, "
        f"95% doc-bootstrap CI {fmt_ci(p3_minus_p12['ci95_stratified_document_bootstrap'])}.",
        "",
        "## H2 — causal reconstruction",
        "",
        "| Condition | KL recovered @ pos | KL @ pos | KL16 | CE16 delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("orig", "sae_small", "sae_big", "p3_only", "p12",
                 "quote_strip_p3", "dataset_mean", "zero"):
        v = causal_condition[name]
        md.append(
            f"| {name} | {v['kl_recovered_at_pos']['mean_equal_doc']:+.4f} | "
            f"{v['kl_at_pos']['mean_equal_doc']:.4f} | "
            f"{v['kl_mean_first16']['mean_equal_doc']:.4f} | "
            f"{v['ce_delta_from_clean_first16']['mean_equal_doc']:+.4f} |"
        )
    md += [
        "",
        f"NLA−SAE-big recovered = {orig_vs_big['mean_equal_doc']:+.4f}; "
        f"90% CI {fmt_ci(orig_vs_big['ci90_stratified_document_bootstrap'])}; "
        f"equivalence ±0.05: **{h2_equiv_big}**.",
        "",
        f"NLA−SAE-small recovered = {orig_vs_small['mean_equal_doc']:+.4f}; "
        f"95% CI {fmt_ci(orig_vs_small['ci95_stratified_document_bootstrap'])}; "
        f"superiority: **{h2_superior_small}**.",
        "",
        "## H3 — causal paragraph mechanism",
        "",
        f"`p3_only - p12` recovered = {p3_vs_p12['mean_equal_doc']:+.4f}, "
        f"95% CI {fmt_ci(p3_vs_p12['ci95_stratified_document_bootstrap'])}.",
        f"`p3_only/orig` recovered retention = **{fmt_num(p3_retention)}**.",
        "",
        "## Decision",
        "",
        out["decision"],
        "",
        f"Result SHA-256: `{out_sha}`",
    ]
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"N4_ANALYSIS_COMPLETE -> {args.out} ({out_sha}) + {args.markdown}")


if __name__ == "__main__":
    main()
