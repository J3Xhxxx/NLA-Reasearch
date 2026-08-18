#!/bin/bash
# Autonomous runner for pilot 3 (residual signal injection). No hard time
# limit (user preference). Current keep-alive policy: record status and exit;
# never shut down the instance.
exec > /root/autodl-tmp/injection.log 2>&1
set -x
PY=/root/miniconda3/bin/python
CMP=/root/autodl-tmp/nla_compare
MODELS=/root/autodl-tmp/models
cd "$CMP"
export HF_HOME=/root/autodl-tmp/hf

finish() {
  echo "=== INJECTION EXIT (status=$1) at $(date) ==="
  sync
  exit "$1"
}

$PY 08_pilot_injection.py \
    --av "$MODELS/nla-gemma3-12b-L32-av" \
    --ar "$MODELS/nla-gemma3-12b-L32-ar" \
    --sae "$MODELS/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small" \
    --activations /root/autodl-tmp/activations/acts_L32.parquet \
    --out /root/autodl-tmp/results/injection_pilot.json || finish 8

echo "=== INJECTION_COMPLETE ==="
finish 0
