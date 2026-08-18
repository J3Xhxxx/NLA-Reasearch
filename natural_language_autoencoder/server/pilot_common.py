#!/usr/bin/env python3
"""Shared pieces for the two NLA<->SAE pilot experiments (06 / 07).

AVLocal is copied verbatim from 03_run_nla.py (same injection math via
NLAClient._build_embeds); JumpReLUSAE / load_acts from 04_run_sae.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

REPO = Path(os.environ.get("NLA_REPO", "/root/autodl-tmp/nla_repo"))
sys.path.insert(0, str(REPO))
from nla_inference import NLAClient, NLACritic, EXPLANATION_RE  # noqa: E402,F401
from transformers import AutoModelForCausalLM  # noqa: E402


class AVLocal:
    """AV inference with plain transformers — no SGLang server needed."""

    def __init__(self, av_dir, device="cuda", dtype=torch.bfloat16):
        self.client = NLAClient(av_dir, sglang_url="http://localhost:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            av_dir, torch_dtype=dtype, device_map=device).eval()
        self.device = device
        self.tok = self.client.tokenizer

    @torch.inference_mode()
    def generate(self, v, *, temperature=0.0, max_new_tokens=200):
        embeds_np, _ = self.client._build_embeds(
            torch.as_tensor(np.asarray(v, np.float32)), None)
        inp = torch.from_numpy(embeds_np)[None].to(self.device, self.model.dtype)
        attn = torch.ones(inp.shape[:2], dtype=torch.long, device=self.device)
        kw = dict(inputs_embeds=inp, attention_mask=attn,
                  max_new_tokens=max_new_tokens,
                  pad_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        out = self.model.generate(**kw)
        text = self.tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(text)
        return m.group(1).strip() if m else text


class JumpReLUSAE:
    def __init__(self, sae_dir: str, device: str = "cuda", dtype=torch.float32):
        p = load_file(str(Path(sae_dir) / "params.safetensors"))
        self.w_enc = p["w_enc"].to(device, dtype)      # [d, F]
        self.b_enc = p["b_enc"].to(device, dtype)      # [F]
        self.w_dec = p["w_dec"].to(device, dtype)      # [F, d]
        self.b_dec = p["b_dec"].to(device, dtype)      # [d]
        self.threshold = p["threshold"].to(device, dtype)  # [F]
        self.d_model = self.w_enc.shape[0]
        self.width = self.w_enc.shape[1]
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor):
        x = x.to(self.device, self.dtype)
        pre = x @ self.w_enc + self.b_enc
        acts = torch.relu(pre) * (pre > self.threshold)
        recon = acts @ self.w_dec + self.b_dec
        return recon, acts


def load_acts(path):
    t = pq.read_table(path)
    flat = t.column("activation_vector").combine_chunks()
    vecs = np.array(flat.to_pylist(), dtype=np.float32)
    meta = {c: t.column(c).to_pylist() for c in ("token", "position", "doc_id")}
    return vecs, meta
