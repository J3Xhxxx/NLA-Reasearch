#!/usr/bin/env python3
"""NLA line: for each activation vector, AV verbalizes it, AR reconstructs it.

    vector --AV--> explanation text --AR--> predicted vector
    score = cos(original, predicted),  mse_nrm = 2(1-cos)   [direction only]

Two AV backends:
  * LOCAL (default): run the AV with plain transformers `generate(inputs_embeds=)`.
    No SGLang, no torch upgrade — runs on the existing torch 2.5.1. Slower per
    vector but perfect for a comparison over a handful/hundreds of vectors.
  * SGLANG (--sglang-url http://localhost:30000): POST input_embeds to a running
    AV server (see launch_av_server.sh). Use this only when you need throughput
    for scanning large dictionaries. Requires the sglang install.

Both backends reuse NLAClient._build_embeds from the repo's nla_inference.py, so
the injection math / √d embed-scale / sidecar contract are byte-identical to the
trained recipe. The AR (NLACritic) is identical in both. Writes per-row JSON for
05_compare.py.

    python 03_run_nla.py \
        --av  /root/autodl-tmp/models/nla-gemma3-12b-L32-av \
        --ar  /root/autodl-tmp/models/nla-gemma3-12b-L32-ar \
        --activations /root/autodl-tmp/activations/acts_L32.parquet \
        --out /root/autodl-tmp/results/nla_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

# Reuse the repo's single-file inference client (validated injection math).
REPO = Path(os.environ.get("NLA_REPO", "/root/autodl-tmp/nla_repo"))
sys.path.insert(0, str(REPO))
from nla_inference import NLAClient, NLACritic, EXPLANATION_RE  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


class AVLocal:
    """AV inference with plain transformers — no SGLang server needed.

    NLAClient builds the (validated) injected embedding sequence; we hand it to
    the SAME checkpoint loaded as a full model via generate(inputs_embeds=...).
    """

    def __init__(self, av_dir, device="cuda", dtype=torch.bfloat16):
        # NLAClient loads only the embedding table + sidecar; the httpx client it
        # opens is never used here (we never call its .generate / SGLang path).
        self.client = NLAClient(av_dir, sglang_url="http://localhost:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            av_dir, torch_dtype=dtype, device_map=device).eval()
        self.device = device
        self.tok = self.client.tokenizer

    @torch.inference_mode()
    def generate(self, v, *, temperature=0.0, max_new_tokens=200):
        embeds_np, _ = self.client._build_embeds(torch.as_tensor(np.asarray(v, np.float32)), None)
        inp = torch.from_numpy(embeds_np)[None].to(self.device, self.model.dtype)
        attn = torch.ones(inp.shape[:2], dtype=torch.long, device=self.device)
        kw = dict(inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new_tokens,
                  pad_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        out = self.model.generate(**kw)
        text = self.tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(text)
        return m.group(1).strip() if m else text


def load_acts(path):
    t = pq.read_table(path)
    flat = t.column("activation_vector").combine_chunks()
    vecs = np.array(flat.to_pylist(), dtype=np.float32)
    meta = {c: t.column(c).to_pylist() for c in ("token", "position", "doc_id")}
    return vecs, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--sglang-url", default=None,
                    help="If set, use the SGLang AV server instead of local transformers.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ar-device", default="cuda")
    ap.add_argument("--av-temperature", type=float, default=0.0)
    ap.add_argument("--av-max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="Only first N vectors (0=all).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vecs, meta = load_acts(args.activations)
    if args.limit > 0:
        vecs = vecs[:args.limit]
        meta = {k: v[:args.limit] for k, v in meta.items()}

    if args.sglang_url:
        av = NLAClient(args.av, sglang_url=args.sglang_url)
        gen = lambda v: av.generate(v, temperature=args.av_temperature,
                                    max_new_tokens=args.av_max_new_tokens)
    else:
        av = AVLocal(args.av, device=args.device)
        gen = lambda v: av.generate(v, temperature=args.av_temperature,
                                    max_new_tokens=args.av_max_new_tokens)

    critic = NLACritic(args.ar, device=args.ar_device)

    rows = []
    for i, v in enumerate(vecs):
        text = gen(v)
        mse, cos = critic.score(text, v)
        rows.append({
            "idx": i, "doc_id": meta["doc_id"][i], "position": meta["position"][i],
            "token": meta["token"][i], "raw_norm": round(float(np.linalg.norm(v)), 3),
            "explanation": text,
            "nla_mse_nrm": round(float(mse), 4), "nla_cos": round(float(cos), 4),
        })
        print(f"[{i+1}/{len(vecs)}] pos={meta['position'][i]:>4} "
              f"cos={cos:.3f} mse_nrm={mse:.3f}  {text[:70]!r}")

    cos = np.array([r["nla_cos"] for r in rows])
    summary = {
        "n": len(rows),
        "backend": "sglang" if args.sglang_url else "local",
        "mean_cos": round(float(cos.mean()), 4),
        "median_cos": round(float(np.median(cos)), 4),
        "mean_mse_nrm": round(float(np.array([r["nla_mse_nrm"] for r in rows]).mean()), 4),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"method": "nla", "summary": summary, "rows": rows},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nNLA summary: {summary}\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
