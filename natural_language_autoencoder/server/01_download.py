#!/usr/bin/env python3
"""Download exactly the weights the NLA-vs-SAE comparison needs — nothing more.

The gemma-scope-2-12b-it repo is ~8TB; we pull ONE SAE from it (layer 32
resid_post, width 16k). The base Gemma-3-12B-IT is gated and the hf-mirror does
NOT bypass the gate (verified: HTTP 403), so it needs a real HF token from an
account that has accepted the Gemma license. The two NLA checkpoints are open.

Targets (use --only to pick a subset):
    av    kitft/nla-gemma3-12b-L32-av     23.6 GB  open  (mirror)
    ar    kitft/nla-gemma3-12b-L32-ar     16.9 GB  open  (mirror)
    sae   google/gemma-scope-2-12b-it      ~1 GB   open  (mirror, 2 files x2 L0)
    base  google/gemma-3-12b-it           24.4 GB  GATED (real hub + token)

Examples:
    # open models via the fast mirror (run `source /etc/network_turbo` first)
    python 01_download.py --only av ar sae

    # gated base — needs a token; uses the REAL hub (mirror 403s on gated)
    HF_TOKEN=hf_xxx python 01_download.py --only base

    # everything
    HF_TOKEN=hf_xxx python 01_download.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = Path(os.environ.get("NLA_MODELS", "/root/autodl-tmp/models"))
MIRROR = "https://hf-mirror.com"
REAL = "https://huggingface.co"

# SAE: only layer-32 resid_post width-16k, both L0 (small≈20, big = denser).
# We skip examples.safetensors (feature activations, not needed to reconstruct).
SAE_PATTERNS = [
    "resid_post_all/layer_32_width_16k_l0_small/config.json",
    "resid_post_all/layer_32_width_16k_l0_small/params.safetensors",
    "resid_post_all/layer_32_width_16k_l0_big/config.json",
    "resid_post_all/layer_32_width_16k_l0_big/params.safetensors",
]

# repo_id, local_subdir, gated?, allow_patterns
JOBS = {
    "av":   ("kitft/nla-gemma3-12b-L32-av", "nla-gemma3-12b-L32-av", False, None),
    "ar":   ("kitft/nla-gemma3-12b-L32-ar", "nla-gemma3-12b-L32-ar", False, None),
    "sae":  ("google/gemma-scope-2-12b-it", "gemma-scope-2-12b-it",   False, SAE_PATTERNS),
    "base": ("google/gemma-3-12b-it",       "gemma-3-12b-it",         True,  None),
}


def fetch(key: str, token: str | None) -> None:
    repo_id, sub, gated, patterns = JOBS[key]
    dest = MODELS / sub
    # Mirror bypasses gating for OPEN repos and is fast in CN; but it 403s on
    # gated repos, so the gated base must go through the real hub with a token.
    endpoint = REAL if gated else MIRROR
    os.environ["HF_ENDPOINT"] = endpoint
    if gated and not token:
        raise SystemExit(
            f"[{key}] {repo_id} is gated. Set HF_TOKEN (from an account that "
            f"accepted the Gemma license at {REAL}/{repo_id})."
        )
    print(f"[{key}] {repo_id}  endpoint={endpoint}  -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=patterns,
        token=token if gated else None,
        max_workers=8,
    )
    print(f"[{key}] done -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(JOBS),
                    help="Subset to download. Default: all four.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    keys = args.only or list(JOBS)
    MODELS.mkdir(parents=True, exist_ok=True)
    for k in keys:
        fetch(k, token)
    print("\nAll requested downloads complete.")
    print("Disk check:  df -h /root/autodl-tmp")


if __name__ == "__main__":
    main()
