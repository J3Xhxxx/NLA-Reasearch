#!/bin/bash
# Autonomous runner for N1 (C7+B3 AR rescoring of frozen text variants) and
# N2 (E10 causal patch-in). No shutdown chain, no hard timeout: per the current
# user instruction the instance stays up. Completion/failure only exits.
exec > /root/autodl-tmp/n1n2.log 2>&1
set -x
PY=/root/miniconda3/bin/python
CMP=/root/autodl-tmp/nla_compare
MODELS=/root/autodl-tmp/models
RES=/root/autodl-tmp/results
ACTS=/root/autodl-tmp/activations/acts_L32.parquet
cd "$CMP" || exit 90
export HF_HOME=/root/autodl-tmp/hf
export NLA_REPO=/root/autodl-tmp/nla_repo
export TOKENIZERS_PARALLELISM=false

echo "N1N2_START $(date -u)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

$PY 29_score_text_variants.py \
  --ar "$MODELS/nla-gemma3-12b-L32-ar" \
  --activations "$ACTS" \
  --variants "$RES/c7b3_variants_v1.json" \
  --out "$RES/c7b3_scores_v1.json" \
  --vecs-out "$RES/c7b3_recon_v1.npz"
s1=$?
echo "N1_EXIT status=$s1 $(date -u)"

$PY 30_causal_patch.py \
  --base-model "$MODELS/gemma-3-12b-it" \
  --activations "$ACTS" \
  --recon "$RES/recon_vectors.npz" \
  --out "$RES/causal_patch_v1.json"
s2=$?
echo "N2_EXIT status=$s2 $(date -u)"

echo "N1N2_COMPLETE $(date -u) n1=$s1 n2=$s2"
exit 0
