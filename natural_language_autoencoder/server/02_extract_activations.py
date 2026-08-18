#!/usr/bin/env python3
"""Extract layer-32 residual activations from base Gemma-3-12B-IT.

These raw vectors are the SHARED input for both comparison lines:
  - NLA line  (03_run_nla.py):  vector -> AV text -> AR -> reconstruct
  - SAE line  (04_run_sae.py):  vector -> SAE encode -> decode

Extraction point is `model.layers[32]` block OUTPUT == resid_post of layer 32.
This is the exact point the NLA L32 checkpoints were trained on AND the point
gemma-scope-2 layer_32 resid_post SAE expects (its config says
hf_hook_point = "model.layers.32.output"). So both sides see identical vectors.

Invariants (from repo CLAUDE.md): vectors are stored RAW (norm="none");
normalization happens downstream. Positions < --min-position decode to noise
(not enough left-context) and are skipped by default.

    python 02_extract_activations.py \
        --base-model /root/autodl-tmp/models/gemma-3-12b-it \
        --prompts prompts.txt \
        --out /root/autodl-tmp/activations/acts_L32.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "Explain why the Eiffel Tower is one of the most famous landmarks in the world, and describe what visitors typically experience when they go to the top.",
    "Describe how photosynthesis works in plants, step by step, including the role of sunlight, water, and carbon dioxide.",
    "Tell the story of how the printing press changed European society in the 15th and 16th centuries.",
    "Compare and contrast cats and dogs as household pets, covering temperament, care requirements, and companionship.",
    "Walk through how a bill becomes a law in a typical parliamentary democracy, from proposal to final approval.",
]


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


@torch.inference_mode()
def capture(model, input_ids, layer_index):
    layers = resolve_layers(model)
    assert 0 <= layer_index < len(layers), f"layer {layer_index} / {len(layers)}"
    grab = {}

    def hook(_m, _i, out):
        grab["h"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()

    h = layers[layer_index].register_forward_hook(hook)
    try:
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    finally:
        h.remove()
    return grab["h"][0]  # [seq, d]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--prompts", default=None, help="Text file, one prompt per line. Default: built-in set.")
    ap.add_argument("--layer-index", type=int, default=32)
    ap.add_argument("--min-position", type=int, default=50,
                    help="Skip positions with too little left-context (repo invariant=50).")
    ap.add_argument("--max-per-prompt", type=int, default=8,
                    help="Cap stored positions per prompt (evenly spaced past min-position).")
    ap.add_argument("--no-chat", action="store_true", help="Do not apply chat template.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = (Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if args.prompts else DEFAULT_PROMPTS)
    prompts = [p for p in prompts if p.strip()]

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=getattr(torch, args.dtype),
        device_map="cuda", trust_remote_code=True).eval()
    device = next(model.parameters()).device

    vecs, toks, positions, doc_ids, doc_texts = [], [], [], [], []
    for doc_id, prompt in enumerate(prompts):
        if args.no_chat:
            ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        else:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids):  # transformers >=5 returns BatchEncoding
                ids = ids["input_ids"]
        ids = ids.to(device)
        hidden = capture(model, ids, args.layer_index)  # [seq, d]
        seq = ids[0].tolist()
        cand = [p for p in range(len(seq)) if p >= args.min_position]
        if not cand:  # short prompt: fall back to the back half
            cand = list(range(len(seq) // 2, len(seq)))
        if args.max_per_prompt > 0 and len(cand) > args.max_per_prompt:
            idx = np.linspace(0, len(cand) - 1, args.max_per_prompt).round().astype(int)
            cand = [cand[i] for i in sorted(set(idx))]
        for p in cand:
            vecs.append(hidden[p].numpy().astype(np.float32))
            toks.append(tok.decode([seq[p]], skip_special_tokens=False))
            positions.append(int(p))
            doc_ids.append(int(doc_id))
            doc_texts.append(prompt)
        print(f"[doc {doc_id}] seq={len(seq)} kept={len(cand)} positions")

    arr = np.stack(vecs)  # [N, d]
    table = pa.table({
        "activation_vector": pa.array(list(arr), type=pa.list_(pa.float32())),
        "token": toks,
        "position": positions,
        "doc_id": doc_ids,
        "prompt": doc_texts,
        "norm": ["none"] * len(vecs),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"\nwrote {len(vecs)} activation rows (d={arr.shape[1]}) -> {args.out}")
    print(f"norm stats: mean L2={np.linalg.norm(arr,axis=1).mean():.1f}  "
          f"min={np.linalg.norm(arr,axis=1).min():.1f}  max={np.linalg.norm(arr,axis=1).max():.1f}")


if __name__ == "__main__":
    main()
