#!/usr/bin/env python3
"""Local analysis of N1 (C7/B3 variants) and N2 (causal patch).

Two things the runners deliberately left to a separate pass:
  * loss_recovered was averaged as a per-row ratio, whose denominator
    (ce_ablate - ce_clean) is near zero on some rows and produces values
    outside [0,1]. Here it is recomputed pooled and by median, which is the
    standard way the SAE literature reports it.
  * paired inference for the text-variant contrasts, plus the joint reading of
    the two experiments (does the text channel that carries the cosine also
    carry the causal effect?).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parent.parent / "results"


def boot(v, stat=np.mean, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    s = stat(np.asarray(v)[idx], axis=1)
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return round(float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb))), 4)


def main() -> None:
    out: dict = {}

    # ---------------- N1 ----------------
    s = json.load(open(RES / "c7b3_scores_v1.json", encoding="utf-8"))
    rows = s["rows"]
    names = list(s["summary_by_variant"].keys())
    names = [n for n in names if not n.startswith("__")]

    def col(name, field="cos_c"):
        return np.array([r["scores"][name][field] for r in rows if name in r["scores"]])

    def orig_matched(name):
        return np.array([r["scores"]["orig"]["cos_c"] for r in rows if name in r["scores"]])

    n1 = {}
    for name in names:
        v, o = col(name), orig_matched(name)
        d = v - o
        n1[name] = {
            "n": len(v),
            "mean_cos_c": round(float(v.mean()), 4),
            "mean_cos_c_ci": boot(v),
            "share_of_orig": round(float(v.mean() / o.mean()), 4),
            "paired_delta_mean": round(float(d.mean()), 4),
            "paired_delta_ci": boot(d),
            "n_negative": int((d < 0).sum()),
            "retrieval_top1": s["retrieval_by_variant"].get(name, {}).get("top1"),
        }
    out["n1_variant_summary"] = n1
    out["n1_generic_fixed_floor"] = s["summary_by_variant"]["__generic_fixed__"]
    out["n1_verification"] = s["verification"]

    # channel decomposition relative to the true generic floor (~0)
    floor = float(s["summary_by_variant"]["__generic_fixed__"]["mean_cos_c_over_40_targets"])
    base = n1["orig"]["mean_cos_c"] - floor
    out["n1_channel_share_above_generic_floor"] = {
        k: round((n1[k]["mean_cos_c"] - floor) / base, 3) for k in names
    }

    # ---------------- N2 ----------------
    c = json.load(open(RES / "causal_patch_v1.json", encoding="utf-8"))
    crows = c["rows"]
    subs = list(crows[0]["results"].keys())
    ce_clean = np.array([r["ce_clean"] for r in crows])

    # E1-E7 position selection fell back to "back half" because the prompts are
    # 28-38 tokens long and --min-position was 50, so part of the cohort sits on
    # chat-template boundary tokens. Stratify on that.
    def is_template(tokstr: str) -> bool:
        t = tokstr.strip()
        return (t.startswith("<") and t.endswith(">")) or t == ""

    tmpl = np.array([is_template(r["token"]) for r in crows])
    out["cohort_token_composition"] = {
        "n_template_or_whitespace": int(tmpl.sum()),
        "n_content": int((~tmpl).sum()),
        "template_tokens": sorted({r["token"] for r in crows if is_template(r["token"])}),
        "note": "02_extract_activations.py used --min-position 50 but the chat-"
                "templated prompts are only 28-38 tokens, so the documented "
                "invariant never applied and the back-half fallback selected "
                "positions, including template boundary tokens.",
    }

    def ce(name):
        return np.array([r["results"][name]["ce_after"] for r in crows])

    def kl(name, field="kl_at_pos"):
        return np.array([r["results"][name][field] for r in crows])

    ce_zero, ce_mean = ce("zero"), ce("dataset_mean")
    # rows whose patched position is the last token have no continuation window
    ok = np.isfinite(ce_clean) & np.isfinite(ce_zero) & np.isfinite(ce_mean)
    out["n2_ce_window_rows_used"] = [int(ok.sum()), int(len(ok))]
    n2 = {}
    for name in subs:
        cp = ce(name)
        k = ok & np.isfinite(cp)
        pooled_zero = 1 - (cp[k].mean() - ce_clean[k].mean()) / (
            ce_zero[k].mean() - ce_clean[k].mean())
        pooled_mean = 1 - (cp[k].mean() - ce_clean[k].mean()) / (
            ce_mean[k].mean() - ce_clean[k].mean())
        den = ce_zero[k] - ce_clean[k]
        safe = np.abs(den) > 1e-6
        per_row = 1 - (cp[k][safe] - ce_clean[k][safe]) / den[safe]
        n2[name] = {
            "n_ce_rows": int(k.sum()),
            "kl_at_pos_mean": round(float(kl(name).mean()), 4),
            "kl_at_pos_median": round(float(np.median(kl(name))), 4),
            "kl_at_pos_ci": boot(kl(name)),
            "kl_mean_after_mean": round(float(kl(name, "kl_mean_after").mean()), 4),
            "ce_after_mean": round(float(cp[k].mean()), 4),
            "loss_recovered_pooled_vs_zero": round(float(pooled_zero), 4),
            "loss_recovered_pooled_vs_mean": round(float(pooled_mean), 4),
            "loss_recovered_median_vs_zero": round(float(np.median(per_row)), 4),
        }
    out["n2_summary_robust"] = n2
    out["n2_ce_clean_mean"] = round(float(np.nanmean(ce_clean)), 4)
    out["n2_provenance"] = c["provenance"]
    out["n2_ce_endpoint_invalid"] = {
        "clean_ce_mean_nats": round(float(np.nanmean(ce_clean)), 3),
        "uniform_ce_nats": 12.48,
        "reason": "the continuation window is dominated by chat-template boundary "
                  "tokens (<end_of_turn>, <start_of_turn>, model) whose clean CE is "
                  "18-53 nats, i.e. far above uniform; see 32_diag_ce_window.py. "
                  "loss_recovered built on this window is not interpretable and is "
                  "superseded by the KL-recovered fraction below.",
    }

    # KL-recovered fraction: 1 at identity, 0 at zero ablation, well defined
    klz = kl("zero")
    klr = {}
    for name in subs:
        f = 1.0 - kl(name) / np.maximum(klz, 1e-6)
        klr[name] = {
            "kl_recovered_mean": round(float(f.mean()), 4),
            "kl_recovered_median": round(float(np.median(f)), 4),
            "kl_recovered_ci": boot(f),
            "kl_recovered_mean_content_tokens": round(float(f[~tmpl].mean()), 4),
            "kl_recovered_mean_template_tokens": round(float(f[tmpl].mean()), 4),
            "kl_at_pos_mean_content": round(float(kl(name)[~tmpl].mean()), 4),
            "kl_at_pos_mean_template": round(float(kl(name)[tmpl].mean()), 4),
        }
    out["n2_kl_recovered"] = klr

    # paired NLA vs SAE on the causal metric
    pair = {}
    for opp in ("sae_small", "sae_big", "resid_text", "dataset_mean"):
        d = kl(opp) - kl("nla")           # positive = NLA better (lower KL)
        dl = ce(opp) - ce("nla")          # positive = NLA better (lower CE)
        kok = np.isfinite(dl)
        pair[f"nla_vs_{opp}"] = {
            "kl_delta_mean": round(float(d.mean()), 4),
            "kl_delta_median": round(float(np.median(d)), 4),
            "kl_delta_ci": boot(d),
            "n_rows_nla_better_kl": int((d > 0).sum()),
            "ce_delta_mean": round(float(dl[kok].mean()), 4),
            "ce_delta_ci": boot(dl[kok]),
            "n_rows_nla_better_ce": f"{int((dl[kok] > 0).sum())}/{int(kok.sum())}",
        }
        # document-clustered sign test (5 docs = the real independent units)
        docs = np.array([r["doc_id"] for r in crows])
        per_doc = [float(d[docs == g].mean()) for g in sorted(set(docs.tolist()))]
        pair[f"nla_vs_{opp}"]["per_document_kl_delta"] = [round(x, 4) for x in per_doc]
        pair[f"nla_vs_{opp}"]["n_docs_nla_better"] = int(sum(1 for x in per_doc if x > 0))
    out["n2_paired"] = pair

    # ---------------- joint reading ----------------
    z = np.load(RES / "recon_vectors.npz")
    m = z["m_hat"].astype(np.float64); m /= np.linalg.norm(m)
    X = z["x"].astype(np.float64)

    def perp(a):
        a = np.asarray(a, np.float64)
        return a - np.outer(a @ m, m) if a.ndim == 2 else a - (a @ m) * m

    Xc = perp(X)
    joint = {}
    for name, key in (("nla", "pred_full"), ("sae_small", "recon_sae_small"),
                      ("sae_big", "recon_sae_big"), ("resid_text", "pred_resid")):
        P = perp(z[key].astype(np.float64))
        cc = np.sum(P * Xc, 1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Xc, axis=1))
        joint[name] = {
            "mean_centered_cos": round(float(cc.mean()), 4),
            "within_method_spearman_cos_vs_kl": spearman(cc, kl(name)),
        }
    cc_all, kl_all = [], []
    for name, key in (("nla", "pred_full"), ("sae_small", "recon_sae_small"),
                      ("sae_big", "recon_sae_big"), ("resid_text", "pred_resid")):
        P = perp(z[key].astype(np.float64))
        cc_all += list(np.sum(P * Xc, 1) /
                       (np.linalg.norm(P, axis=1) * np.linalg.norm(Xc, axis=1)))
        kl_all += list(kl(name))
    joint["pooled_across_methods"] = {
        "n": len(cc_all), "spearman_cos_vs_kl": spearman(cc_all, kl_all)}
    out["joint_cos_vs_causal"] = joint

    # does the text channel that carries cosine also carry the causal effect?
    # (uses the N1 recon vectors: patching was only run for the 4 main methods,
    #  so this is the cosine-side statement only, flagged as such)
    out["notes"] = [
        "loss_recovered_pooled_* is the primary form; the per-row ratio in "
        "causal_patch_v1.json summary is unstable when ce_ablate ~ ce_clean.",
        "N2 patched only the 4 reconstruction sources plus controls; the N1 text "
        "variants were not patched, so channel attribution is cosine-side only.",
        "5 documents are the independent units for N2 clustering.",
    ]

    (RES / "n1n2_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== N1 verification:", json.dumps(out["n1_verification"]))
    print("== N1 fixed generic floor:", json.dumps(out["n1_generic_fixed_floor"]))
    print("\n{:<17}{:>9}{:>9}{:>10}{:>8}{:>8}".format(
        "variant", "cos_c", "share", "delta", "neg", "top1"))
    for k, v in n1.items():
        print("{:<17}{:>9.4f}{:>9.3f}{:>10.4f}{:>8}{:>8}".format(
            k, v["mean_cos_c"], v["share_of_orig"], v["paired_delta_mean"],
            f'{v["n_negative"]}/{v["n"]}', str(v["retrieval_top1"])))
    print("\nchannel share above generic floor:",
          json.dumps(out["n1_channel_share_above_generic_floor"]))
    print("\n{:<18}{:>10}{:>10}{:>10}{:>10}".format(
        "substitute", "KL@pos", "KLmed", "LR_zero", "LR_mean"))
    for k, v in n2.items():
        print("{:<18}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format(
            k, v["kl_at_pos_mean"], v["kl_at_pos_median"],
            v["loss_recovered_pooled_vs_zero"], v["loss_recovered_pooled_vs_mean"]))
    print("\ncohort composition:", json.dumps(
        {k: v for k, v in out["cohort_token_composition"].items() if k != "note"}))
    print("\n{:<18}{:>10}{:>10}{:>12}{:>12}".format(
        "substitute", "KLrec", "KLrecMed", "KLrec_cont", "KLrec_tmpl"))
    for k, v in klr.items():
        print("{:<18}{:>10.4f}{:>10.4f}{:>12.4f}{:>12.4f}".format(
            k, v["kl_recovered_mean"], v["kl_recovered_median"],
            v["kl_recovered_mean_content_tokens"],
            v["kl_recovered_mean_template_tokens"]))
    print("\nN2 paired (KL, positive = NLA better):")
    for k, v in pair.items():
        print(f"  {k:<24} mean={v['kl_delta_mean']:+7.3f} "
              f"median={v['kl_delta_median']:+7.3f} ci={v['kl_delta_ci']} "
              f"rows={v['n_rows_nla_better_kl']}/40 docs={v['n_docs_nla_better']}/5")
    print("\njoint:", json.dumps(joint, indent=2))


if __name__ == "__main__":
    main()
