#!/bin/bash
# Autonomous pilot runner: 06 (w_dec verbalization) -> 07 (residual audit).
# Current keep-alive policy: record status and exit; never shut down the instance.
# Results land in /root/autodl-tmp/results/, log pilot.log.
exec > /root/autodl-tmp/pilot.log 2>&1
set -x
PY=/root/miniconda3/bin/python
CMP=/root/autodl-tmp/nla_compare
MODELS=/root/autodl-tmp/models
ACTS=/root/autodl-tmp/activations/acts_L32.parquet
RES=/root/autodl-tmp/results
SAE=$MODELS/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small
cd "$CMP"
export HF_HOME=/root/autodl-tmp/hf

finish() {
  echo "=== PILOTS EXIT (status=$1) at $(date) ==="
  sync
  exit "$1"
}

$PY 06_pilot_wdec.py \
    --av "$MODELS/nla-gemma3-12b-L32-av" \
    --ar "$MODELS/nla-gemma3-12b-L32-ar" \
    --sae "$SAE" \
    --activations "$ACTS" \
    --out "$RES/wdec_pilot.json" || finish 6

$PY 07_pilot_residual.py \
    --av "$MODELS/nla-gemma3-12b-L32-av" \
    --ar "$MODELS/nla-gemma3-12b-L32-ar" \
    --sae "$SAE" \
    --activations "$ACTS" \
    --nla "$RES/nla_results.json" \
    --out "$RES/resid_pilot.json" || finish 7

echo "=== PILOTS_COMPLETE ==="
finish 0
