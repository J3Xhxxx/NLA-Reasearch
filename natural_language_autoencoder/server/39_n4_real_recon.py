#!/usr/bin/env python3
"""N4 stage 1: resumable AV generation, frozen channel ablations, AR/SAE recon.

The input is the clean 200-position N3 cohort.  The ordering is intentional:

  1. generate AV explanations with an append-only per-row checkpoint;
  2. freeze all deterministic text variants and their SHA-256;
  3. only then load the AR and score/reconstruct those texts;
  4. reconstruct the same activations with the two frozen SAEs.

This prevents an AR score from influencing how a text ablation is written.
The output NPZ is consumed by 40_n4_causal_patch.py.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from pilot_common import AVLocal, JumpReLUSAE, NLACritic


QUOTE_RE = re.compile(r'"[^"]*"')
BLANK = '"[...]"'
SHUFFLE_SEED = 20260730
GENERIC_TEXTS = [
    "The passage uses a structured informational style and continues the current topic.",
    "This is a coherent piece of explanatory prose with ordinary grammatical structure.",
    "The context establishes a subject and prepares a likely continuation of the discussion.",
    "The text contains semantic and syntactic information typical of a written document.",
    "A descriptive answer is being developed in a clear and organized format.",
    "The final token fits a locally predictable continuation in the surrounding sentence.",
    "The activation reflects general language structure, topical context, and discourse form.",
    "This appears to be an informative response that elaborates on previously introduced material.",
]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_bytes(obj) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")


def write_frozen(path: str | Path, obj) -> str:
    """Create once; on resume require byte-for-byte identity."""
    p = Path(path)
    payload = canonical_bytes(obj)
    if p.exists():
        old = p.read_bytes()
        if old != payload:
            raise SystemExit(f"frozen file differs on resume: {p}")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    p.with_suffix(p.suffix + ".sha256").write_text(
        f"{digest}  {p.name}\n", encoding="utf-8"
    )
    return digest


def load_activations(path: str | Path):
    t = pq.read_table(path)
    flat = t.column("activation_vector").combine_chunks()
    x = np.asarray(flat.to_pylist(), dtype=np.float32)
    wanted = (
        "token", "token_id", "position", "doc_id", "corpus", "source", "lang",
        "seq_len", "context_tail", "continuation",
    )
    meta = {c: t.column(c).to_pylist() for c in wanted}
    return x, meta


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def build_variants(explanations: list[dict], activation_sha: str) -> dict:
    rows = []
    n_para = []
    for r in explanations:
        i = int(r["idx"])
        orig = r["explanation"]
        ps = paragraphs(orig)
        n_para.append(len(ps))
        if len(ps) < 3:
            raise SystemExit(
                f"AUDIT STOP before AR scoring: idx={i} has {len(ps)} paragraphs"
            )
        p1, p3 = ps[0], ps[-1]
        p2 = "\n\n".join(ps[1:-1])
        rng = np.random.default_rng(SHUFFLE_SEED + i)
        words = orig.split()
        variants = {
            "orig": orig,
            "p1_only": p1,
            "p2_only": p2,
            "p3_only": p3,
            "p12": "\n\n".join((p1, p2)),
            "quote_strip_p2": "\n\n".join((p1, QUOTE_RE.sub(BLANK, p2), p3)),
            "quote_strip_p3": "\n\n".join((p1, p2, QUOTE_RE.sub(BLANK, p3))),
            "quote_strip_all": QUOTE_RE.sub(BLANK, orig),
            "word_shuffle": " ".join(words[j] for j in rng.permutation(len(words))),
        }
        rows.append(
            {
                "idx": i,
                "doc_id": r["doc_id"],
                "position": r["position"],
                "token": r["token"],
                "variants": variants,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "N4 real-content frozen channel ablations",
        "status": "frozen_before_any_AR_score",
        "activation_sha256": activation_sha,
        "protocol": {
            "paragraph_rule": (
                "split on blank lines; p1=first, p3=last, p2=all middle; "
                "fewer than 3 paragraphs aborts before AR scoring"
            ),
            "shuffle_seed": SHUFFLE_SEED,
            "quote_regex": QUOTE_RE.pattern,
            "generic_texts_frozen": True,
        },
        "diagnostics": {
            "n_rows": len(rows),
            "paragraph_count_distribution": dict(sorted(Counter(n_para).items())),
        },
        "generic_fixed_texts": GENERIC_TEXTS,
        "rows": rows,
    }


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


def lodo_cos(p: np.ndarray, x: np.ndarray, docs: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float64)
    total = x.sum(0, dtype=np.float64)
    for d in np.unique(docs):
        test = docs == d
        train_n = int((~test).sum())
        if train_n == 0:
            raise ValueError("leave-one-document-out mean has no training rows")
        m = unit((total - x[test].sum(0, dtype=np.float64)) / train_n)
        out[test] = row_cos(p[test], x[test], m)
    return out


def retrieval(p: np.ndarray, x: np.ndarray, m_hat: np.ndarray) -> dict:
    pp, xx = perp(p, m_hat), perp(x, m_hat)
    pp /= np.maximum(np.linalg.norm(pp, axis=1, keepdims=True), 1e-12)
    xx /= np.maximum(np.linalg.norm(xx, axis=1, keepdims=True), 1e-12)
    sim = pp @ xx.T
    diag = np.diag(sim)
    ranks = 1 + (sim > diag[:, None] + 1e-12).sum(1)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def load_generation_checkpoint(
    path: str | Path, activation_sha: str, n: int
) -> dict[int, dict]:
    p = Path(path)
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("activation_sha256") != activation_sha:
            raise SystemExit(f"checkpoint input hash mismatch at line {line_no}")
        i = int(r["idx"])
        if i in out:
            raise SystemExit(f"duplicate checkpoint idx {i}")
        if not 0 <= i < n:
            raise SystemExit(f"checkpoint idx out of range: {i}")
        out[i] = r
    return out


def append_checkpoint(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--av", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--sae-small", required=True)
    ap.add_argument("--sae-big", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--explanations-out", required=True)
    ap.add_argument("--variants-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vecs-out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    activation_sha = sha256_file(args.activations)
    prereg_sha = sha256_file(args.prereg)
    x, meta = load_activations(args.activations)
    if args.limit:
        x = x[: args.limit]
        meta = {k: v[: args.limit] for k, v in meta.items()}
    n = len(x)
    if not args.limit and n != 200:
        raise SystemExit(f"expected frozen 200-row cohort, got {n}")
    if len(set(zip(meta["doc_id"], meta["position"]))) != n:
        raise SystemExit("duplicate (doc_id, position) rows")

    # ---------------- AV: resumable, append-only ----------------
    done = load_generation_checkpoint(args.checkpoint, activation_sha, n)
    missing = [i for i in range(n) if i not in done]
    print(
        f"[input] n={n} docs={len(set(meta['doc_id']))} "
        f"sha256={activation_sha} checkpoint={len(done)} missing={len(missing)}",
        flush=True,
    )
    if missing:
        av = AVLocal(args.av, device="cuda")
        for k, i in enumerate(missing, 1):
            text = av.generate(
                x[i], temperature=0.0, max_new_tokens=args.max_new_tokens
            )
            r = {
                "activation_sha256": activation_sha,
                "idx": i,
                "doc_id": int(meta["doc_id"][i]),
                "position": int(meta["position"][i]),
                "token": meta["token"][i],
                "explanation": text,
            }
            append_checkpoint(args.checkpoint, r)
            done[i] = r
            print(
                f"[AV {len(done):>3}/{n}] doc={r['doc_id']} pos={r['position']} "
                f"chars={len(text)} {text[:65]!r}",
                flush=True,
            )
        del av
        gc.collect()
        torch.cuda.empty_cache()

    explanations = [
        {k: done[i][k] for k in ("idx", "doc_id", "position", "token", "explanation")}
        for i in range(n)
    ]
    explanation_payload = {
        "schema_version": 1,
        "experiment": "N4 real-content AV explanations",
        "status": "complete_frozen_before_AR",
        "activation_sha256": activation_sha,
        "prereg_sha256": prereg_sha,
        "generation": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "n": n,
        },
        "rows": explanations,
    }
    explanations_sha = write_frozen(args.explanations_out, explanation_payload)
    variants_payload = build_variants(explanations, activation_sha)
    variants_payload["explanations_sha256"] = explanations_sha
    variants_sha = write_frozen(args.variants_out, variants_payload)
    print(
        f"[freeze] explanations={explanations_sha} variants={variants_sha} "
        f"paragraphs={variants_payload['diagnostics']['paragraph_count_distribution']}",
        flush=True,
    )

    # ---------------- AR: starts only after variants are frozen ----------------
    critic = NLACritic(args.ar, device="cuda")
    cache: dict[str, np.ndarray] = {}

    def rec(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = critic.reconstruct(text).numpy().astype(np.float32)
            if len(cache) % 50 == 0:
                print(f"[AR] {len(cache)} unique texts", flush=True)
        return cache[text]

    generic_recon = np.stack([rec(t) for t in GENERIC_TEXTS])
    names = list(variants_payload["rows"][0]["variants"])
    pred: dict[str, np.ndarray] = {}
    for name in names:
        pred[name] = np.stack(
            [rec(r["variants"][name]) for r in variants_payload["rows"]]
        )
        print(f"[AR] variant {name} complete", flush=True)
    del critic
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- SAE native reconstructions ----------------
    xt = torch.from_numpy(x)
    sae_s = JumpReLUSAE(args.sae_small, device="cuda")
    recon_s, acts_s = sae_s(xt)
    recon_s = recon_s.float().cpu().numpy()
    l0_s = (acts_s > 0).sum(1).cpu().numpy()
    del sae_s, acts_s
    gc.collect()
    torch.cuda.empty_cache()

    sae_b = JumpReLUSAE(args.sae_big, device="cuda")
    recon_b, acts_b = sae_b(xt)
    recon_b = recon_b.float().cpu().numpy()
    l0_b = (acts_b > 0).sum(1).cpu().numpy()
    del sae_b, acts_b
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- centered/raw scoring ----------------
    docs = np.asarray(meta["doc_id"], dtype=np.int64)
    m_hat = unit(x.mean(0, dtype=np.float64))
    scores: dict[str, dict[str, np.ndarray]] = {}
    all_preds = {**pred, "sae_small": recon_s, "sae_big": recon_b}
    for name, p in all_preds.items():
        scores[name] = {
            "cos_c": row_cos(p, x, m_hat),
            "cos_c_lodo": lodo_cos(p, x, docs),
            "cos_raw": row_cos_raw(p, x),
        }

    # Every generic text is scored against every target.  The per-target mean
    # is used by the preregistered channel-share calculation.
    gen_mat = np.stack([row_cos(np.repeat(g[None], n, 0), x, m_hat)
                        for g in generic_recon])
    generic_by_target = gen_mat.mean(0)
    generic_floor = float(generic_by_target.mean())
    orig_mean = float(scores["orig"]["cos_c"].mean())

    summary = {}
    for name, p in all_preds.items():
        c = scores[name]["cos_c"]
        cl = scores[name]["cos_c_lodo"]
        raw = scores[name]["cos_raw"]
        item = {
            "n": n,
            "mean_cos_c": float(c.mean()),
            "median_cos_c": float(np.median(c)),
            "mean_cos_c_lodo": float(cl.mean()),
            "mean_cos_raw": float(raw.mean()),
            "retrieval": retrieval(p, x, m_hat),
        }
        if name in pred:
            item["share_above_generic_floor"] = float(
                (c.mean() - generic_floor)
                / max(orig_mean - generic_floor, 1e-12)
            )
            item["paired_delta_vs_orig_mean"] = float(
                (c - scores["orig"]["cos_c"]).mean()
            )
        summary[name] = item

    x_center = x - x.mean(0, keepdims=True)
    summary["sae_small"]["native_fve"] = float(
        1.0 - np.square(x - recon_s).sum() / np.square(x_center).sum()
    )
    summary["sae_small"]["mean_l0"] = float(l0_s.mean())
    summary["sae_big"]["native_fve"] = float(
        1.0 - np.square(x - recon_b).sum() / np.square(x_center).sum()
    )
    summary["sae_big"]["mean_l0"] = float(l0_b.mean())

    rows_out = []
    for i in range(n):
        rows_out.append(
            {
                "idx": i,
                **{k: meta[k][i] for k in (
                    "doc_id", "position", "token", "corpus", "source", "lang",
                    "context_tail", "continuation",
                )},
                "generic_floor_cos_c": float(generic_by_target[i]),
                "scores": {
                    name: {field: float(values[field][i]) for field in values}
                    for name, values in scores.items()
                },
            }
        )

    out = {
        "schema_version": 1,
        "experiment": "N4 real-content AR/SAE reconstruction and channel ablation",
        "status": "complete",
        "inputs": {
            "activations_sha256": activation_sha,
            "prereg_sha256": prereg_sha,
            "explanations_sha256": explanations_sha,
            "variants_sha256": variants_sha,
            "script_sha256": sha256_file(__file__),
        },
        "cohort": {
            "n_rows": n,
            "n_docs": len(set(meta["doc_id"])),
            "n_template_or_blank": 0,
            "by_corpus": dict(Counter(meta["corpus"])),
        },
        "centering": {
            "primary": "full-cohort unit mean direction projected from both sides",
            "sensitivity": "leave-one-document-out unit mean direction",
        },
        "generic_floor": {
            "n_texts": len(GENERIC_TEXTS),
            "mean_cos_c": generic_floor,
            "max_pair_cos_c": float(gen_mat.max()),
        },
        "summary": summary,
        "rows": rows_out,
        "elapsed_seconds": round(time.time() - t0, 1),
        "n_unique_ar_texts": len(cache),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_sha = sha256_file(outp)
    outp.with_suffix(outp.suffix + ".sha256").write_text(
        f"{out_sha}  {outp.name}\n", encoding="utf-8"
    )

    npz_payload = {
        "x": x.astype(np.float32),
        "m_hat": m_hat.astype(np.float32),
        "doc_ids": docs,
        "positions": np.asarray(meta["position"], dtype=np.int64),
        "generic_recon": generic_recon.astype(np.float32),
        "recon_sae_small": recon_s.astype(np.float32),
        "recon_sae_big": recon_b.astype(np.float32),
    }
    for name, p in pred.items():
        npz_payload[f"pred_{name}"] = p.astype(np.float32)
    np.savez_compressed(args.vecs_out, **npz_payload)
    vec_sha = sha256_file(args.vecs_out)
    Path(args.vecs_out).with_suffix(
        Path(args.vecs_out).suffix + ".sha256"
    ).write_text(f"{vec_sha}  {Path(args.vecs_out).name}\n", encoding="utf-8")

    print("\n=== N4 reconstruction summary ===")
    print(f"generic floor: {generic_floor:+.4f}")
    for name in ("orig", "p3_only", "p12", "quote_strip_p3",
                 "sae_small", "sae_big"):
        s = summary[name]
        share = s.get("share_above_generic_floor")
        print(
            f"{name:<16} cos_c={s['mean_cos_c']:+.4f} "
            f"lodo={s['mean_cos_c_lodo']:+.4f} "
            f"share={share if share is not None else float('nan'):.3f} "
            f"top1={s['retrieval']['top1']:.3f}"
        )
    print(
        f"N4_RECON_COMPLETE -> {args.out} ({out_sha}) + "
        f"{args.vecs_out} ({vec_sha})",
        flush=True,
    )


if __name__ == "__main__":
    main()
