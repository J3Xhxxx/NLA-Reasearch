#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE="$ROOT/nla_compare"
RESULTS="$ROOT/results"
PY=/root/miniconda3/bin/python

cd "$CODE"
export PYTHONUNBUFFERED=1

expected_acts=eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66
expected_prereg=44bd48998f6436616347bb2d74d9b8a569a85a3639c314ce9711a252c8075f1c
actual_acts="$(sha256sum "$ROOT/activations/acts_L32_n3_v1.parquet" | awk '{print $1}')"
actual_prereg="$(sha256sum "$RESULTS/n4_real_content_preregistration_v1.md" | awk '{print $1}')"
[[ "$actual_acts" == "$expected_acts" ]] || {
  echo "activation hash mismatch: $actual_acts != $expected_acts" >&2
  exit 2
}
[[ "$actual_prereg" == "$expected_prereg" ]] || {
  echo "prereg hash mismatch: $actual_prereg != $expected_prereg" >&2
  exit 2
}

for target in \
  "$RESULTS/n4_causal_patch_v1.json" \
  "$RESULTS/n4_analysis_v1.json" \
  "$RESULTS/n4_analysis_v1.md"; do
  [[ ! -e "$target" ]] || {
    echo "refusing to overwrite frozen output: $target" >&2
    exit 3
  }
done

finish() {
  status=$?
  printf '[%s] N4_CAUSAL_PIPELINE_EXIT status=%s\n' \
    "$(date --iso-8601=seconds)" "$status"
  exit "$status"
}
trap finish EXIT

printf '[%s] N4_CAUSAL_PIPELINE_START\n' "$(date --iso-8601=seconds)"
sha256sum \
  40_n4_causal_patch.py \
  41_n4_analyze.py \
  "$RESULTS/n4_real_content_preregistration_v1.md" \
  "$RESULTS/n4_recon_vectors_v1.npz" \
  "$RESULTS/n4_recon_analysis_v1.json"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

"$PY" -B 40_n4_causal_patch.py \
  --base-model "$ROOT/models/gemma-3-12b-it" \
  --activations "$ROOT/activations/acts_L32_n3_v1.parquet" \
  --recon "$RESULTS/n4_recon_vectors_v1.npz" \
  --prereg "$RESULTS/n4_real_content_preregistration_v1.md" \
  --checkpoint "$RESULTS/n4_causal_checkpoint_v1.jsonl" \
  --out "$RESULTS/n4_causal_patch_v1.json"

"$PY" -B 41_n4_analyze.py \
  --recon "$RESULTS/n4_recon_analysis_v1.json" \
  --vecs "$RESULTS/n4_recon_vectors_v1.npz" \
  --causal "$RESULTS/n4_causal_patch_v1.json" \
  --prereg "$RESULTS/n4_real_content_preregistration_v1.md" \
  --out "$RESULTS/n4_analysis_v1.json" \
  --markdown "$RESULTS/n4_analysis_v1.md"

sha256sum \
  "$RESULTS/n4_causal_patch_v1.json" \
  "$RESULTS/n4_analysis_v1.json" \
  "$RESULTS/n4_analysis_v1.md"
printf '[%s] N4_CAUSAL_PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)"
