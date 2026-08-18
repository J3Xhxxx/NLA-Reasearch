#!/usr/bin/env bash
# Stage 52: run the text-only parser, select the analysis cohort, and freeze donors.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

stage=52_freeze_variants_donor
log_path=${N6_STAGE52_LOG:-"$N6_RESULTS_ROOT/n6_52_freeze_variants_donor_v1.log"}
status_path=${N6_STAGE52_STATUS:-"$N6_RESULTS_ROOT/n6_52_freeze_variants_donor_v1.exit"}
plan_path=${N6_STAGE49_OUT:-"$N6_RESULTS_ROOT/n6_provisional_cohort_v1.json"}
explanations_path=${N6_STAGE51_OUT:-"$N6_RESULTS_ROOT/n6_av_explanations_v1.json"}
output_path=${N6_STAGE52_OUT:-"$N6_RESULTS_ROOT/n6_variants_donor_v1.json"}

stage_main() {
    n6_preflight_common || return 1
    n6_verify_stage_inputs \
        "$plan_path" \
        "$explanations_path" || return 1
    n6_require_new_outputs "$output_path" || return 1
    cd "$N6_CODE_ROOT" || return 97
    "$N6_PYTHON" -B "$N6_CODE_ROOT/52_n6_freeze_variants.py" \
        --prereg "$N6_PREREG" \
        --plan "$plan_path" \
        --explanations "$explanations_path" \
        --code-manifest "$N6_CODE_MANIFEST" \
        --base-model "$N6_BASE_MODEL" \
        --seed 20260803 \
        --target 400 \
        --cell-seed-quota 2 \
        --min-cell-size 2 \
        --out "$output_path" || return $?
    n6_verify_sidecar "$output_path" "stage 52 frozen variants and donor map"
}

if n6_run_stage "$stage" "$log_path" "$status_path" stage_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
