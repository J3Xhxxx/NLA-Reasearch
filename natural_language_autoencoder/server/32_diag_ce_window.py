#!/usr/bin/env python3
"""Diagnose the CE continuation window used by 30_causal_patch.py.

Why this exists: the CE-based `loss_recovered` in causal_patch_v1.json came out
uninterpretable (negative for good reconstructions, >1 for bad ones). This
script prints, for every patched position, which tokens the window actually
scores and what the CLEAN model's per-token CE on them is.

Result (recorded in results/n1n2.log): the window is dominated by chat-template
boundary tokens (<end_of_turn>, <start_of_turn>, "model"), whose clean CE is
36-53 nats because the base model cannot predict them at all. Mean clean CE
over the window is ~21 nats, above the uniform-distribution bound of 12.48.
Any zero/mean-ablation normalization over such a window is meaningless, so
F12 reports KL and KL-recovered only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

RES = os.environ.get("RES", "/root/autodl-tmp/results")
MODELS = os.environ.get("MODELS", "/root/autodl-tmp/models")
BASE = os.path.join(MODELS, "gemma-3-12b-it")
CE_WINDOW = 8


def main() -> None:
    patch = json.load(open(os.path.join(RES, "causal_patch_v1.json"), encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(BASE)

    n_template = 0
    n_total = 0
    ce_all: list[float] = []

    for row in patch["rows"]:
        ids = row["input_ids"]
        pos = row["position"]
        window = ids[pos + 1 : pos + 1 + CE_WINDOW]
        texts = [tok.decode([t]) for t in window]
        ce = row.get("clean_ce_per_token")

        print(f"\n--- doc={row['doc']} pos={pos} len={len(ids)} ---")
        print(f"patched token: {tok.decode([ids[pos]])!r}")
        for i, (t, txt) in enumerate(zip(window, texts)):
            c = ce[i] if ce and i < len(ce) else float("nan")
            is_tpl = txt.strip() in ("", "<end_of_turn>", "<start_of_turn>", "model", "user")
            n_template += int(is_tpl)
            n_total += 1
            if not np.isnan(c):
                ce_all.append(c)
            print(f"  +{i+1} id={t:<7} {txt!r:<20} clean_ce={c:8.3f}  {'TEMPLATE' if is_tpl else ''}")

    print("\n=== summary ===")
    print(f"window tokens: {n_total}, template/blank: {n_template} ({n_template/n_total:.1%})")
    if ce_all:
        print(f"clean CE over window: mean={np.mean(ce_all):.3f} median={np.median(ce_all):.3f} nats")
    print(f"uniform-distribution bound: {np.log(tok.vocab_size):.3f} nats")
    print("=> CE-based loss_recovered is not interpretable on this window; report KL only.")


if __name__ == "__main__":
    sys.exit(main())
