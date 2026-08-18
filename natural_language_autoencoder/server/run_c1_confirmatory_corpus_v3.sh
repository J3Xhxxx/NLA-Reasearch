#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON=/root/miniconda3/bin/python
CODE=/root/autodl-tmp/nla_compare
RESULTS=/root/autodl-tmp/results
ACTIVATIONS=/root/autodl-tmp/activations
MODELS=/root/autodl-tmp/models
RUN_LOG="$RESULTS/c1_confirmatory_corpus_v3.log"
GPU_CSV="$RESULTS/c1_confirmatory_corpus_gpu_v3.csv"

mkdir -p "$RESULTS" "$ACTIVATIONS"
cd "$CODE"

exec > >(tee -a "$RUN_LOG") 2>&1

if [[ ! -s "$GPU_CSV" ]]; then
  printf '%s\n' \
    'timestamp,memory_used_mib,utilization_gpu_pct,power_draw_w,temperature_c' \
    > "$GPU_CSV"
fi

nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  -l 2 >> "$GPU_CSV" &
MONITOR_PID=$!

stop_monitor() {
  if kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap stop_monitor EXIT INT TERM

START_EPOCH=$(date +%s)
printf 'C1_CONFIRMATORY_CORPUS_V3_JOB_START epoch=%s\n' "$START_EPOCH"

"$PYTHON" 20_generate_c1_confirmatory_corpus_v3.py \
  --base-model "$MODELS/gemma-3-12b-it" \
  --spec "$CODE/c1_confirmatory_concepts_v2.json" \
  --anchors "$CODE/c1_confirmatory_scenario_anchors_v3r2.json" \
  --rubric "$CODE/c1_confirmatory_manual_audit_addendum_v3r2.json" \
  --base-rubric "$CODE/c1_confirmatory_manual_audit_rubric_v1.json" \
  --preregistration \
    "$RESULTS/c1_confirmatory_preregistration_v3_amendment.md" \
  --stage0-freeze "$RESULTS/c1_confirmatory_stage0_freeze_v3.json" \
  --checkpoint "$RESULTS/c1_confirmatory_corpus_checkpoint_v3.jsonl" \
  --out-manifest "$ACTIVATIONS/c1_confirmatory_all_v3.jsonl" \
  --out-discovery-manifest \
    "$ACTIVATIONS/c1_confirmatory_discovery_v3.jsonl" \
  --out-heldout-manifest \
    "$ACTIVATIONS/c1_confirmatory_heldout_v3.jsonl" \
  --out-report "$RESULTS/c1_confirmatory_corpus_report_v3.json"

END_EPOCH=$(date +%s)
printf 'wall_seconds=%s\n' "$((END_EPOCH - START_EPOCH))"
printf '%s\n' C1_CONFIRMATORY_CORPUS_V3_JOB_COMPLETE
printf '%s\n' SERVER_LEFT_RUNNING
