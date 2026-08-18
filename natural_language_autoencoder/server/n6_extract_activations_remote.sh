#!/usr/bin/env bash
# Stage 50: extract layer-32 activations for every provisional row.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

stage=50_extract_activations
log_path=${N6_STAGE50_LOG:-"$N6_RESULTS_ROOT/n6_50_extract_activations_v1.log"}
status_path=${N6_STAGE50_STATUS:-"$N6_RESULTS_ROOT/n6_50_extract_activations_v1.exit"}
plan_path=${N6_STAGE49_OUT:-"$N6_RESULTS_ROOT/n6_provisional_cohort_v1.json"}
output_path=${N6_STAGE50_OUT:-"$N6_ACTIVATIONS_ROOT/acts_L32_n6_provisional_v1.parquet"}
report_path="${output_path}.json"

stage_main() {
    n6_preflight_common || return 1
    n6_verify_stage_inputs "$plan_path" || return 1
    n6_require_new_outputs "$output_path" "$report_path" || return 1
    mkdir -p "$N6_ACTIVATIONS_ROOT" || return 1
    cd "$N6_CODE_ROOT" || return 97
    "$N6_PYTHON" -B "$N6_CODE_ROOT/50_n6_extract_activations.py" \
        --plan "$plan_path" \
        --base-model "$N6_BASE_MODEL" \
        --model-manifest "$N6_MODEL_MANIFEST" \
        --code-manifest "$N6_CODE_MANIFEST" \
        --prereg "$N6_PREREG" \
        --layer-index 32 \
        --dtype bfloat16 \
        --out "$output_path" || return $?
    n6_verify_sidecar "$output_path" "stage 50 activation parquet" || return 1
    n6_verify_sidecar "$report_path" "stage 50 extraction report"
}

if n6_run_stage "$stage" "$log_path" "$status_path" stage_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
