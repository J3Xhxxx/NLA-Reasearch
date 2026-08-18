#!/usr/bin/env bash
# Stage 51: generate and freeze AV explanations only; no parser, AR, SAE, or outcome.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

stage=51_generate_av
log_path=${N6_STAGE51_LOG:-"$N6_RESULTS_ROOT/n6_51_generate_av_v1.log"}
status_path=${N6_STAGE51_STATUS:-"$N6_RESULTS_ROOT/n6_51_generate_av_v1.exit"}
plan_path=${N6_STAGE49_OUT:-"$N6_RESULTS_ROOT/n6_provisional_cohort_v1.json"}
activations_path=${N6_STAGE50_OUT:-"$N6_ACTIVATIONS_ROOT/acts_L32_n6_provisional_v1.parquet"}
checkpoint_path=${N6_STAGE51_CHECKPOINT:-"$N6_RESULTS_ROOT/n6_av_checkpoint_v1.jsonl"}
output_path=${N6_STAGE51_OUT:-"$N6_RESULTS_ROOT/n6_av_explanations_v1.json"}

stage_main() {
    n6_preflight_common || return 1
    n6_verify_stage_inputs "$plan_path" "$activations_path" || return 1
    n6_require_new_outputs "$output_path" || return 1
    cd "$N6_CODE_ROOT" || return 97
    "$N6_PYTHON" -B "$N6_CODE_ROOT/51_n6_generate_av.py" \
        --av "$N6_AV_MODEL" \
        --activations "$activations_path" \
        --plan "$plan_path" \
        --model-manifest "$N6_MODEL_MANIFEST" \
        --code-manifest "$N6_CODE_MANIFEST" \
        --prereg "$N6_PREREG" \
        --checkpoint "$checkpoint_path" \
        --max-new-tokens 200 \
        --out "$output_path" || return $?
    n6_verify_sidecar "$output_path" "stage 51 frozen explanations"
}

if n6_run_stage "$stage" "$log_path" "$status_path" stage_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
