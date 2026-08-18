#!/usr/bin/env bash
# Environment setup for the NLA-vs-SAE comparison line on the AutoDL A800 box.
#
# Safe to run in 无卡模式 (no-GPU mode): only installs packages, downloads
# nothing GPU-bound. Re-runnable.
#
#   bash 00_setup_env.sh          # core deps + sglang
#   bash 00_setup_env.sh core     # core deps only (skip the heavy sglang install)
#
# Design:
#   - Models + HF cache live on the DATA disk (/root/autodl-tmp), never the
#     30GB system disk. We set HF_HOME there and persist it to ~/.bashrc.
#   - Core deps (transformers/safetensors/...) cover activation extraction,
#     the AR (NLACritic) and the whole SAE line — no sglang needed for those.
#   - sglang is ONLY needed to serve the AV (activation verbalizer). Installed
#     separately so a sglang hiccup doesn't block the rest.
set -euo pipefail

PY=/root/miniconda3/bin/python
PIP="$PY -m pip"
ROOT=/root/autodl-tmp

echo "=== Persist HF cache + endpoint to ~/.bashrc (data disk) ==="
mkdir -p "$ROOT/hf" "$ROOT/models" "$ROOT/activations" "$ROOT/results"
grep -q 'HF_HOME=/root/autodl-tmp/hf' ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# --- NLA/SAE experiment: keep HF downloads on the data disk ---
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
EOF
export HF_HOME=$ROOT/hf
export HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub

echo "=== Core deps (extraction + AR + SAE line) ==="
$PIP install -q -U \
    "transformers>=4.50" accelerate safetensors \
    httpx orjson pyyaml numpy pyarrow "huggingface_hub>=0.25"

echo "=== Versions ==="
$PY - <<'PY'
import importlib
for m in ("torch","transformers","safetensors","numpy","pyarrow","httpx","orjson","yaml","huggingface_hub"):
    try:
        mod=importlib.import_module(m); print(f"  {m:16s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:16s} MISSING ({e})")
import torch
print("  torch.cuda.is_available():", torch.cuda.is_available(), "(False is expected in 无卡模式)")
PY

# transformers must know Gemma-3. Quick check (no weights downloaded).
echo "=== Gemma-3 architecture support check ==="
$PY - <<'PY'
try:
    from transformers import Gemma3ForConditionalGeneration  # noqa
    print("  Gemma3ForConditionalGeneration: OK")
except Exception as e:
    print("  WARNING: transformers may be too old for Gemma-3:", e)
PY

if [ "${1:-all}" = "core" ]; then
    echo "=== Done (core only). Run without 'core' to install sglang for the AV server. ==="
    exit 0
fi

echo "=== sglang (AV serving only) — this is the heavy one ==="
echo "    NOTE: sglang may pull a newer torch/flashinfer. That is fine."
$PIP install -q "sglang[all]>=0.5.6" || {
    echo "!! sglang install failed. The SAE line + AR still work without it."
    echo "!! Retry later with: $PIP install 'sglang[all]>=0.5.6'"
    exit 0
}
$PY - <<'PY'
import sglang, torch
print("  sglang", sglang.__version__, "| torch", torch.__version__)
PY
echo "=== Setup complete. ==="
