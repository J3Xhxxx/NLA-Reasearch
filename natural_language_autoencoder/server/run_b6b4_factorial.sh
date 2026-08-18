#!/bin/bash
# Autonomous B6+B4 batch.  User explicitly requested that the server remain on,
# so this runner records status and exits without any shutdown path.
exec > /root/autodl-tmp/b6b4_factorial.log 2>&1
set -x

PY=/root/miniconda3/bin/python
CMP=/root/autodl-tmp/nla_compare
MODELS=/root/autodl-tmp/models
RESULTS=/root/autodl-tmp/results

cd "$CMP" || exit 90
export HF_HOME=/root/autodl-tmp/hf
export PYTHONUNBUFFERED=1

"$PY" 13_probe_factorial_polarity.py \
  --av "$MODELS/nla-gemma3-12b-L32-av" \
  --ar "$MODELS/nla-gemma3-12b-L32-ar" \
  --selection "$RESULTS/b6b4_factorial_selection.json" \
  --vectors "$RESULTS/b6b4_factorial_vectors.npz" \
  --checkpoint "$RESULTS/b6b4_factorial_av_rows.jsonl" \
  --out "$RESULTS/b6b4_factorial_result.json" \
  --vectors-out "$RESULTS/b6b4_factorial_recon_vectors.npz" \
  --samples-per-sign 5 \
  --temperature 0.7 \
  --max-new-tokens 200 \
  --seed 20260726 \
  --bootstrap 20000
status=$?

echo "=== B6B4_FACTORIAL_EXIT (status=$status) at $(date -Is) ==="
if test "$status" -eq 0; then
  echo "=== B6B4_FACTORIAL_COMPLETE ==="
fi
sync
exit "$status"
