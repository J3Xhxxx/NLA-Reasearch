#!/usr/bin/env bash
set -uo pipefail

root=/root/autodl-tmp
code=$root/nla_compare
results=$root/results
activations=$root/activations
log=$results/n5_heldout_extract_v2.log
status=$results/n5_heldout_extract_v2.exit

mkdir -p "$activations"
rm -f "$status"
cd "$code" || exit 97
if /root/miniconda3/bin/python -B 43_n5_extract_activations.py \
    --plan "$results/n5_cohort_plan_v2.json" \
    --base-model "$root/models/gemma-3-12b-it" \
    --model-manifest "$results/n5_model_weights_v1.sha256" \
    --split heldout \
    --layer-index 32 \
    --dtype bfloat16 \
    --out "$activations/acts_L32_n5_heldout_v2.parquet" \
    >"$log" 2>&1; then
    printf '0\n' >"$status"
else
    exit_code=$?
    printf '%s\n' "$exit_code" >"$status"
fi
