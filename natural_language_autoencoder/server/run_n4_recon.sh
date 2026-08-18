#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE="$ROOT/nla_compare"
RESULTS="$ROOT/results"
PY=/root/miniconda3/bin/python
LOG="$RESULTS/n4_recon.log"

cd "$CODE"
export NLA_REPO="$ROOT/nla_repo"
export PYTHONUNBUFFERED=1

expected_acts=eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66
actual_acts="$(sha256sum "$ROOT/activations/acts_L32_n3_v1.parquet" | awk '{print $1}')"
if [[ "$actual_acts" != "$expected_acts" ]]; then
  echo "activation hash mismatch: $actual_acts != $expected_acts" >&2
  exit 2
fi

finish() {
  status=$?
  printf '[%s] N4_RECON_EXIT status=%s\n' "$(date --iso-8601=seconds)" "$status"
  exit "$status"
}
trap finish EXIT

printf '[%s] N4_RECON_START\n' "$(date --iso-8601=seconds)"
sha256sum \
  39_n4_real_recon.py \
  "$RESULTS/n4_real_content_preregistration_v1.md" \
  "$ROOT/activations/acts_L32_n3_v1.parquet"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

"$PY" -B 39_n4_real_recon.py \
  --av "$ROOT/models/nla-gemma3-12b-L32-av" \
  --ar "$ROOT/models/nla-gemma3-12b-L32-ar" \
  --sae-small "$ROOT/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small" \
  --sae-big "$ROOT/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_big" \
  --activations "$ROOT/activations/acts_L32_n3_v1.parquet" \
  --prereg "$RESULTS/n4_real_content_preregistration_v1.md" \
  --checkpoint "$RESULTS/n4_av_checkpoint_v1.jsonl" \
  --explanations-out "$RESULTS/n4_explanations_v1.json" \
  --variants-out "$RESULTS/n4_variants_v1.json" \
  --out "$RESULTS/n4_recon_analysis_v1.json" \
  --vecs-out "$RESULTS/n4_recon_vectors_v1.npz"

printf '[%s] N4_RECON_COMPLETE\n' "$(date --iso-8601=seconds)"
