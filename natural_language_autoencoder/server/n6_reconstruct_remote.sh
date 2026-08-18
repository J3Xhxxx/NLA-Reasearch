#!/usr/bin/env bash
# Stage 53: reconstruct frozen text variants with AR and the frozen SAE-big comparator.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

stage=53_reconstruct
log_path=${N6_STAGE53_LOG:-"$N6_RESULTS_ROOT/n6_53_reconstruct_v1.log"}
status_path=${N6_STAGE53_STATUS:-"$N6_RESULTS_ROOT/n6_53_reconstruct_v1.exit"}
plan_path=${N6_STAGE49_OUT:-"$N6_RESULTS_ROOT/n6_provisional_cohort_v1.json"}
activations_path=${N6_STAGE50_OUT:-"$N6_ACTIVATIONS_ROOT/acts_L32_n6_provisional_v1.parquet"}
variants_path=${N6_STAGE52_OUT:-"$N6_RESULTS_ROOT/n6_variants_donor_v1.json"}
n5_gate_path=${N6_N5_GATE:-"$N6_RESULTS_ROOT/n5_gate_v2.json"}
output_path=${N6_STAGE53_OUT:-"$N6_RESULTS_ROOT/n6_recon_v1.json"}
vectors_path=${N6_STAGE53_VECS_OUT:-"$N6_RESULTS_ROOT/n6_recon_vectors_v1.npz"}

stage_main() {
    n6_preflight_common || return 1
    n6_verify_stage_inputs \
        "$plan_path" \
        "$activations_path" \
        "$variants_path" \
        "$n5_gate_path" || return 1
    n6_require_new_outputs "$output_path" "$vectors_path" || return 1
    cd "$N6_CODE_ROOT" || return 97
    "$N6_PYTHON" -B "$N6_CODE_ROOT/53_n6_reconstruct.py" \
        --ar "$N6_AR_MODEL" \
        --sae-big "$N6_SAE_BIG_MODEL" \
        --activations "$activations_path" \
        --plan "$plan_path" \
        --variants "$variants_path" \
        --model-manifest "$N6_MODEL_MANIFEST" \
        --code-manifest "$N6_CODE_MANIFEST" \
        --prereg "$N6_PREREG" \
        --n5-gate "$n5_gate_path" \
        --out "$output_path" \
        --vecs-out "$vectors_path" || return $?
    n6_verify_sidecar "$output_path" "stage 53 reconstruction JSON" || return 1
    n6_verify_sidecar "$vectors_path" "stage 53 reconstruction NPZ"
}

if n6_run_stage "$stage" "$log_path" "$status_path" stage_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
