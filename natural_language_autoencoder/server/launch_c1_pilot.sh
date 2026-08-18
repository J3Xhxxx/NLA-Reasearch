#!/usr/bin/env bash
set -uo pipefail

CODE=/root/autodl-tmp/nla_compare
RESULTS=/root/autodl-tmp/results
LOG="$RESULTS/c1_pilot.log"
GPU_LOG="$RESULTS/c1_pilot_gpu.csv"

cd "$CODE"
printf '%s\n' \
  "timestamp_server_local, utilization_gpu_percent, memory_used_mib, power_draw_w, temperature_c" \
  > "$GPU_LOG"

{
  printf 'C1_PILOT_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  time -p bash "$CODE/run_c1_pilot.sh"
  status=$?
  printf 'C1_PILOT_EXIT_CODE=%s\n' "$status"
  printf 'C1_PILOT_FINISHED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$status"
} > "$LOG" 2>&1 &
pipeline_pid=$!

while kill -0 "$pipeline_pid" 2>/dev/null; do
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader,nounits >> "$GPU_LOG" 2>/dev/null || true
  sleep 5
done

wait "$pipeline_pid"
