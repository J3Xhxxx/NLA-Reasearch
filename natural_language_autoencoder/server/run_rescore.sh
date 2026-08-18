#!/bin/bash
# Autonomous runner for 09 (centered rescoring). Current keep-alive policy:
# completion/failure records status and exits without shutting down the instance.
exec > /root/autodl-tmp/rescore.log 2>&1
set -x
PY=/root/miniconda3/bin/python
CMP=/root/autodl-tmp/nla_compare
MODELS=/root/autodl-tmp/models
cd "$CMP"
export HF_HOME=/root/autodl-tmp/hf

finish() {
  echo "=== RESCORE EXIT (status=$1) at $(date) ==="
  sync
  exit "$1"
}

$PY 09_rescore_centered.py \
    --ar "$MODELS/nla-gemma3-12b-L32-ar" \
    --sae-small "$MODELS/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small" \
    --sae-big   "$MODELS/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_big" \
    --activations /root/autodl-tmp/activations/acts_L32.parquet \
    --results /root/autodl-tmp/results \
    --out /root/autodl-tmp/results/centered_rescore.json \
    --vecs-out /root/autodl-tmp/results/recon_vectors.npz || finish 9

echo "=== RESCORE_COMPLETE ==="
finish 0
