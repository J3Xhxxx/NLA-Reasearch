#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE=/root/autodl-tmp/nla_compare
RESULTS=/root/autodl-tmp/results
ACTIVATIONS=/root/autodl-tmp/activations
MODELS=/root/autodl-tmp/models

mkdir -p "$RESULTS"
cd "$CODE"

if [[ ! -f "$RESULTS/c1_pilot_benchmark.json" ]]; then
  "$PYTHON" 15_build_c1_pilot.py \
    --selection "$RESULTS/b6b4_factorial_selection.json" \
    --result "$RESULTS/b6b4_factorial_result.json" \
    --vectors "$RESULTS/b6b4_factorial_vectors.npz" \
    --activations "$ACTIVATIONS/acts_L32_factorial_v1.parquet" \
    --labels "$CODE/c1_pilot_labels.json" \
    --out "$RESULTS/c1_pilot_benchmark.json"
else
  printf '%s\n' C1_PILOT_REUSING_FROZEN_BENCHMARK
fi

"$PYTHON" 16_run_c1_pilot.py \
  --benchmark "$RESULTS/c1_pilot_benchmark.json" \
  --vectors "$RESULTS/b6b4_factorial_vectors.npz" \
  --base "$MODELS/gemma-3-12b-it" \
  --ar "$MODELS/nla-gemma3-12b-L32-ar" \
  --checkpoint "$RESULTS/c1_pilot_checkpoint_v2.jsonl" \
  --out "$RESULTS/c1_pilot_result.json" \
  --vectors-out "$RESULTS/c1_pilot_recon_vectors.npz"

"$PYTHON" 17_analyze_c1_pilot.py \
  --result "$RESULTS/c1_pilot_result.json" \
  --out-json "$RESULTS/c1_pilot_analysis.json" \
  --out-md "$RESULTS/c1_pilot_analysis.md"

printf '%s\n' C1_PILOT_PIPELINE_COMPLETE
