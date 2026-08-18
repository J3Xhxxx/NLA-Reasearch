#!/usr/bin/env python3
"""N3 step 2 — real-corpus activation statistics for all 16k SAE features.

Streams the frozen N3 corpus through base Gemma-3-12B-IT, captures layer-32
resid_post, encodes with BOTH gemma-scope SAEs, and accumulates per-feature:

  * firing frequency, mean/max/var of activation (global and PER SOURCE)
  * top-K activating contexts (doc_id, position, token) for real max-act labels
  * per-language firing on the PARALLEL flores passages, i.e. language
    selectivity with content held constant

Design notes that matter for cost and correctness:

  - Early exit: a forward hook on layer 32 raises _StopForward, so we never run
    layers 33-47. Saves ~1/3 of the compute; the hook must fire on every batch
    and the script asserts it did.
  - Raw text, no chat template. E1-E7 applied a chat template to 5 short prompts
    and 13/40 sampled positions landed on template tokens (F13). SAE feature
    statistics should describe natural text, so no template is applied here.
  - Right padding + attention mask; pad positions are excluded from every
    statistic. Causal attention means real tokens are unaffected by later pads.
  - Checkpoints every --ckpt-every batches so a dropped SSH session or a
    container restart never costs more than a few minutes of GPU.

Primary SAE is l0_small because that is the one B6+B4 selected its 24 semantic
directions from (b6b4_factorial_selection.json -> inputs.sae); l0_big is
computed in the same pass since the encode is negligible next to the 12B forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot_common import JumpReLUSAE


class _StopForward(Exception):
    """Raised inside the layer-32 hook to skip layers 33-47."""


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


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Accum:
    """Per-feature running statistics for one SAE."""

    def __init__(self, width: int, n_groups: int, topk: int, device: str):
        self.width, self.topk, self.device = width, topk, device
        self.fire = torch.zeros(width, dtype=torch.float64, device=device)
        self.act_sum = torch.zeros(width, dtype=torch.float64, device=device)
        self.act_sq = torch.zeros(width, dtype=torch.float64, device=device)
        self.act_max = torch.zeros(width, dtype=torch.float32, device=device)
        self.group_fire = torch.zeros(n_groups, width, dtype=torch.float64, device=device)
        self.group_act_sum = torch.zeros(n_groups, width, dtype=torch.float64, device=device)
        self.group_tokens = torch.zeros(n_groups, dtype=torch.float64, device=device)
        self.top_val = torch.full((width, topk), -1.0, dtype=torch.float32, device=device)
        self.top_meta = torch.zeros(width, topk, 3, dtype=torch.int64, device=device)
        self.n_groups = n_groups
        self.n_tokens = 0

    @torch.inference_mode()
    def update(self, acts: torch.Tensor, group_idx: torch.Tensor, meta: torch.Tensor):
        """acts [T, F] (pad already removed); group_idx [T]; meta [T, 3]."""
        fired = acts > 0
        self.fire += fired.sum(0).double()
        self.act_sum += acts.sum(0).double()
        self.act_sq += (acts.double() ** 2).sum(0)
        self.act_max = torch.maximum(self.act_max, acts.max(0).values)
        self.group_fire.index_add_(0, group_idx, fired.double())
        self.group_act_sum.index_add_(0, group_idx, acts.double())
        self.group_tokens += torch.bincount(group_idx, minlength=self.n_groups).double()
        self.n_tokens += acts.shape[0]

        k = min(self.topk, acts.shape[0])
        vals, idx = torch.topk(acts, k=k, dim=0)          # [k, F]
        cand_val = torch.cat([self.top_val, vals.T], dim=1)               # [F, K+k]
        cand_meta = torch.cat([self.top_meta, meta[idx].permute(1, 0, 2)], dim=1)
        best = torch.topk(cand_val, k=self.topk, dim=1)
        self.top_val = best.values
        self.top_meta = torch.gather(cand_meta, 1, best.indices.unsqueeze(-1).expand(-1, -1, 3))

    def state(self) -> dict:
        return {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in {
            "fire": self.fire, "act_sum": self.act_sum, "act_sq": self.act_sq,
            "act_max": self.act_max, "group_fire": self.group_fire,
            "group_act_sum": self.group_act_sum, "group_tokens": self.group_tokens,
            "top_val": self.top_val, "top_meta": self.top_meta,
            "n_tokens": self.n_tokens,
        }.items()}

    def load(self, st: dict):
        self.fire = torch.as_tensor(st["fire"], device=self.device)
        self.act_sum = torch.as_tensor(st["act_sum"], device=self.device)
        self.act_sq = torch.as_tensor(st["act_sq"], device=self.device)
        self.act_max = torch.as_tensor(st["act_max"], device=self.device)
        self.group_fire = torch.as_tensor(st["group_fire"], device=self.device)
        self.group_act_sum = torch.as_tensor(st["group_act_sum"], device=self.device)
        self.group_tokens = torch.as_tensor(st["group_tokens"], device=self.device)
        self.top_val = torch.as_tensor(st["top_val"], device=self.device)
        self.top_meta = torch.as_tensor(st["top_meta"], device=self.device)
        self.n_tokens = int(st["n_tokens"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="/root/autodl-tmp/results/n3_corpus_v1.jsonl")
    ap.add_argument("--base-model", default="/root/autodl-tmp/models/gemma-3-12b-it")
    ap.add_argument("--sae-root",
                    default="/root/autodl-tmp/models/gemma-scope-2-12b-it/resid_post_all")
    ap.add_argument("--layer-index", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-doc-tokens", type=int, default=64)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = no global cap. Applied AFTER the per-corpus caps.")
    ap.add_argument("--max-pile-tokens", type=int, default=8_000_000,
                    help="Cap the Pile half. The parallel corpus is never capped: "
                         "dropping some languages of a passage would destroy the "
                         "content-held-constant control it exists for.")
    ap.add_argument("--shuffle-seed", type=int, default=20260730)
    ap.add_argument("--ckpt-every", type=int, default=40)
    ap.add_argument("--out-prefix", default="/root/autodl-tmp/results/n3_feature_stats_v1")
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(args.corpus, encoding="utf-8") if l.strip()]
    groups = sorted({d["source"] for d in docs})
    gidx = {g: i for i, g in enumerate(groups)}
    print(f"corpus: {len(docs)} docs, {len(groups)} sources", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()
    device = next(model.parameters()).device
    layers = resolve_layers(model)
    assert 0 <= args.layer_index < len(layers), f"layer {args.layer_index}/{len(layers)}"
    print(f"model: {len(layers)} layers, early exit after {args.layer_index}", flush=True)

    saes = {}
    for name, sub in (("small", "layer_32_width_16k_l0_small"),
                      ("big", "layer_32_width_16k_l0_big")):
        d = os.path.join(args.sae_root, sub)
        saes[name] = (JumpReLUSAE(d), d, sha256_file(os.path.join(d, "params.safetensors")))
    width = saes["small"][0].width
    print(f"saes: {list(saes)} width={width}", flush=True)

    accs = {n: Accum(width, len(groups), args.topk, str(device)) for n in saes}

    # --- tokenize into fixed-length chunks ---
    # Per-corpus caps: the Pile half is capped for cost, the parallel half is
    # always kept whole. Then one deterministic shuffle so that any prefix (i.e.
    # any interrupted run) is still a proportional sample of both corpora.
    by_corpus: dict[str, list[tuple[int, int, list[int]]]] = {}
    for d in docs:
        ids = tok(d["text"], add_special_tokens=True)["input_ids"]
        if len(ids) < args.min_doc_tokens:
            continue
        for ci in range(0, len(ids), args.seq_len):
            piece = ids[ci:ci + args.seq_len]
            if len(piece) < args.min_doc_tokens:
                break
            by_corpus.setdefault(d["corpus"], []).append((d["doc_id"], ci, piece))

    caps = {"pile": args.max_pile_tokens}
    chunks: list[tuple[int, int, list[int]]] = []
    for corpus, lst in sorted(by_corpus.items()):
        cap = caps.get(corpus, 0)
        avail = sum(len(c[2]) for c in lst)
        if cap and avail > cap:
            keep, acc_t = [], 0
            for c in lst:
                if acc_t + len(c[2]) > cap:
                    break
                keep.append(c)
                acc_t += len(c[2])
            lst = keep
        print(f"  {corpus:8s} {len(lst):6d} seqs  {sum(len(c[2]) for c in lst):>12,} tok"
              f"  (available {avail:,})", flush=True)
        chunks.extend(lst)

    import random as _random
    _random.Random(args.shuffle_seed).shuffle(chunks)
    total_tokens = sum(len(c[2]) for c in chunks)
    if args.max_tokens and total_tokens > args.max_tokens:
        keep, acc_t = [], 0
        for c in chunks:
            if acc_t + len(c[2]) > args.max_tokens:
                break
            keep.append(c)
            acc_t += len(c[2])
        chunks, total_tokens = keep, acc_t
    print(f"chunks: {len(chunks)} seqs, {total_tokens:,} tokens", flush=True)

    doc_group = {d["doc_id"]: gidx[d["source"]] for d in docs}
    ckpt_path = Path(str(args.out_prefix) + ".partial.npz")
    start_batch = 0
    n_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    if ckpt_path.exists():
        st = np.load(ckpt_path, allow_pickle=True)
        start_batch = int(st["batch_done"])
        for n in accs:
            accs[n].load({k[len(n) + 1:]: st[k] for k in st.files if k.startswith(n + "_")})
        print(f"resuming from batch {start_batch}/{n_batches}", flush=True)

    hook_fired = {"n": 0}
    grab: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        grab["h"] = (out[0] if isinstance(out, tuple) else out)
        hook_fired["n"] += 1
        raise _StopForward

    handle = layers[args.layer_index].register_forward_hook(hook)
    t0 = time.time()
    tokens_done = 0
    try:
        for b in range(start_batch, n_batches):
            batch = chunks[b * args.batch_size:(b + 1) * args.batch_size]
            if not batch:
                break
            L = max(len(c[2]) for c in batch)
            pad = tok.pad_token_id or 0
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, (_, _, piece) in enumerate(batch):
                ids[i, :len(piece)] = torch.tensor(piece)
                mask[i, :len(piece)] = 1
            ids, mask = ids.to(device), mask.to(device)

            with torch.inference_mode():
                try:
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
                except _StopForward:
                    pass
                hidden = grab.pop("h").float()          # [B, L, d]

                sel = mask.bool().reshape(-1)
                flat = hidden.reshape(-1, hidden.shape[-1])[sel]        # [T, d]
                meta_rows, group_rows = [], []
                for i, (doc_id, ci, piece) in enumerate(batch):
                    g = doc_group[doc_id]
                    for p, t in enumerate(piece):
                        meta_rows.append((doc_id, ci + p, t))
                        group_rows.append(g)
                meta = torch.tensor(meta_rows, dtype=torch.int64, device=device)
                group_idx = torch.tensor(group_rows, dtype=torch.int64, device=device)
                assert meta.shape[0] == flat.shape[0], (meta.shape, flat.shape)

                for name, (sae, _, _) in saes.items():
                    _, acts = sae(flat)
                    accs[name].update(acts, group_idx, meta)
                    del acts

            tokens_done += int(sel.sum())
            if (b + 1) % args.ckpt_every == 0 or b + 1 == n_batches:
                payload = {"batch_done": b + 1}
                for n, a in accs.items():
                    for k, v in a.state().items():
                        payload[f"{n}_{k}"] = v
                np.savez(ckpt_path, **payload)
                el = time.time() - t0
                rate = tokens_done / max(el, 1e-6)
                eta = (total_tokens - tokens_done) / max(rate, 1e-6) / 60
                print(f"[{b+1}/{n_batches}] {tokens_done:,} tok  "
                      f"{rate:,.0f} tok/s  eta {eta:.1f} min", flush=True)
    finally:
        handle.remove()

    assert hook_fired["n"] > 0, "layer-32 hook never fired -- early exit broken"
    print(f"hook fired {hook_fired['n']} times", flush=True)

    # --- final summary ---
    out = {"schema_version": 1, "experiment": "N3 real-corpus feature statistics",
           "args": vars(args), "sources": groups,
           "corpus_sha256": sha256_file(args.corpus),
           "saes": {n: {"dir": d, "params_sha256": s} for n, (_, d, s) in saes.items()},
           "totals": {"n_seqs": len(chunks), "n_tokens": int(tokens_done),
                      "hook_calls": hook_fired["n"]}}

    npz: dict[str, np.ndarray] = {}
    for name, a in accs.items():
        st = a.state()
        for k, v in st.items():
            npz[f"{name}_{k}"] = np.asarray(v)
        freq = st["fire"] / max(st["n_tokens"], 1)
        alive = int((st["fire"] > 0).sum())
        out.setdefault("per_sae", {})[name] = {
            "n_tokens": int(st["n_tokens"]),
            "alive_features": alive,
            "alive_fraction": alive / a.width,
            "median_freq_alive": float(np.median(freq[freq > 0])) if alive else 0.0,
            "mean_l0": float(st["fire"].sum() / max(st["n_tokens"], 1)),
        }
    npz["groups"] = np.array(groups)
    np.savez(str(args.out_prefix) + ".npz", **npz)
    Path(str(args.out_prefix) + ".json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink()

    print("\n=== done ===")
    for n, s in out["per_sae"].items():
        print(f"{n:6s} tokens={s['n_tokens']:,} alive={s['alive_features']}/{width} "
              f"({s['alive_fraction']:.1%}) mean_L0={s['mean_l0']:.1f}")
    print(f"-> {args.out_prefix}.npz / .json")


if __name__ == "__main__":
    main()
