#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE=/root/autodl-tmp/nla_compare
RESULTS=/root/autodl-tmp/results
ACTIVATIONS=/root/autodl-tmp/activations
MODELS=/root/autodl-tmp/models

STAGE1="$RESULTS/c1_confirmatory_stage1_freeze_v2.json"
AUDIT="$RESULTS/c1_confirmatory_manual_audit_v1.json"
DISCOVERY_MANIFEST="$ACTIVATIONS/c1_confirmatory_discovery_v2.jsonl"
DISCOVERY_ACTIVATIONS="$ACTIVATIONS/acts_L32_c1_confirmatory_discovery_v2.parquet"
DISCOVERY_PROVENANCE="$RESULTS/c1_confirmatory_discovery_provenance_v2.json"
SELECTION="$RESULTS/c1_confirmatory_selection_v2.json"
VECTORS="$RESULTS/c1_confirmatory_vectors_v2.npz"
STATS="$RESULTS/c1_confirmatory_train_stats_v2.npz"
GPU_CSV="$RESULTS/c1_confirmatory_discovery_selection_gpu_v2.csv"

BASE="$MODELS/gemma-3-12b-it"
SAE="$MODELS/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small"

for required in \
  "$STAGE1" \
  "$AUDIT" \
  "$DISCOVERY_MANIFEST" \
  "$CODE/11_extract_factorial_activations.py" \
  "$CODE/21_validate_c1_activation_provenance.py" \
  "$CODE/21_select_c1_confirmatory_features.py" \
  "$CODE/c1_confirmatory_concepts_v2.json" \
  "$CODE/c1_confirmatory_denylist_v1.json"
do
  if [[ ! -f "$required" ]]; then
    printf 'missing required input: %s\n' "$required" >&2
    exit 1
  fi
done

for output in \
  "$DISCOVERY_ACTIVATIONS" \
  "$DISCOVERY_PROVENANCE" \
  "$SELECTION" \
  "$VECTORS" \
  "$STATS"
do
  if [[ -e "$output" ]]; then
    printf 'refusing to overwrite frozen output: %s\n' "$output" >&2
    exit 1
  fi
done

"$PYTHON" - "$STAGE1" "$AUDIT" <<'PY'
import json
import pathlib
import sys

stage1 = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if stage1.get("status") != "frozen_before_discovery_activation_extraction":
    raise SystemExit("invalid Stage-1 freeze status")
if str(audit.get("status", "")).upper() != "PASS":
    raise SystemExit("manual semantic audit did not PASS")
PY

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
cd "$CODE"

"$PYTHON" 11_extract_factorial_activations.py \
  --base-model "$BASE" \
  --manifest "$DISCOVERY_MANIFEST" \
  --layer-index 32 \
  --min-position 50 \
  --max-per-prompt 0 \
  --dtype bfloat16 \
  --out "$DISCOVERY_ACTIVATIONS"

"$PYTHON" 21_validate_c1_activation_provenance.py \
  --manifest "$DISCOVERY_MANIFEST" \
  --activations "$DISCOVERY_ACTIVATIONS" \
  --base-model "$BASE" \
  --extractor "$CODE/11_extract_factorial_activations.py" \
  --manual-audit "$AUDIT" \
  --expected-split train \
  --expected-documents 96 \
  --layer-index 32 \
  --min-position 50 \
  --max-per-prompt 0 \
  --dtype bfloat16 \
  --out "$DISCOVERY_PROVENANCE"

"$PYTHON" 21_select_c1_confirmatory_features.py \
  --sae "$SAE" \
  --activations "$DISCOVERY_ACTIVATIONS" \
  --spec "$CODE/c1_confirmatory_concepts_v2.json" \
  --denylist "$CODE/c1_confirmatory_denylist_v1.json" \
  --out "$SELECTION" \
  --vectors-out "$VECTORS" \
  --stats-out "$STATS"

"$PYTHON" - "$SELECTION" <<'PY'
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = selection.get("summary", {})
print(
    "C1_CONFIRMATORY_SELECTION_STATUS "
    f"status={selection.get('status')} "
    f"features={summary.get('selected_features')} "
    f"concepts={summary.get('populated_concepts')} "
    f"complete_pairs={summary.get('complete_hard_negative_pairs')}"
)
PY

END_EPOCH=$(date +%s)
printf 'wall_seconds=%s\n' "$((END_EPOCH - START_EPOCH))"
printf '%s\n' C1_CONFIRMATORY_DISCOVERY_SELECTION_V2_JOB_COMPLETE
