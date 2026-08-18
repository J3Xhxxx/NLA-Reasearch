#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE=/root/autodl-tmp/nla_compare
RESULTS=/root/autodl-tmp/results
ACTIVATIONS=/root/autodl-tmp/activations
MODELS=/root/autodl-tmp/models
GPU_CSV="$RESULTS/c1_confirmatory_corpus_gpu.csv"

mkdir -p "$RESULTS" "$ACTIVATIONS"
cd "$CODE"

printf '%s\n' \
  'timestamp,memory_used_mib,utilization_gpu_pct,power_draw_w,temperature_c' \
  > "$GPU_CSV"
nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  -l 2 >> "$GPU_CSV" &
MONITOR_PID=$!

stop_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap stop_monitor EXIT

START_EPOCH=$(date +%s)
"$PYTHON" 20_generate_c1_confirmatory_corpus.py \
  --base-model "$MODELS/gemma-3-12b-it" \
  --spec "$CODE/c1_confirmatory_concepts_v1.json" \
  --preregistration "$CODE/c1_confirmatory_preregistration_v1.md" \
  --stage0-freeze "$CODE/c1_confirmatory_stage0_freeze_v1.json" \
  --checkpoint "$RESULTS/c1_confirmatory_corpus_checkpoint_v1.jsonl" \
  --out-manifest "$ACTIVATIONS/c1_confirmatory_all_v1.jsonl" \
  --out-discovery-manifest \
    "$ACTIVATIONS/c1_confirmatory_discovery_v1.jsonl" \
  --out-heldout-manifest \
    "$ACTIVATIONS/c1_confirmatory_heldout_v1.jsonl" \
  --out-report "$RESULTS/c1_confirmatory_corpus_report_v1.json"
END_EPOCH=$(date +%s)

printf 'wall_seconds=%s\n' "$((END_EPOCH - START_EPOCH))"
printf '%s\n' C1_CONFIRMATORY_CORPUS_JOB_COMPLETE
