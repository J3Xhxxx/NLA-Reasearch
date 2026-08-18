#!/usr/bin/env bash
# Launch the SGLang server that hosts the AV (activation verbalizer) model.
# Keep this running in its own terminal; roundtrip.py / nla_inference.py talk to it.
#
# RTX 5090 (32GB) notes:
#   - Qwen2.5-7B in bf16 is ~15GB.
#   - --disable-radix-cache is REQUIRED for input_embeds requests (see docs/inference.md).
#   - NLA prompts are short, so a small --context-length saves KV cache.
#   - MEM_FRACTION controls how much of the 32GB the server reserves:
#       * AV-only (smoke test / nla_inference.py): 0.85 is fine (default).
#       * Round-trip on ONE card (demo/roundtrip.py loads the AR ~10.5GB in the
#         SAME process): set MEM_FRACTION=0.55 so the AR fits alongside the
#         server, e.g.  MEM_FRACTION=0.55 bash demo/launch_av_server.sh
set -euo pipefail

AV_DIR="${AV_DIR:-/root/autodl-tmp/models/nla-qwen-av}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"

exec python -m sglang.launch_server \
    --model-path "$AV_DIR" \
    --port "$PORT" \
    --disable-radix-cache \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LENGTH" \
    --trust-remote-code
