#!/usr/bin/env bash
# Stage 54: causal patch every frozen condition and score clean candidate mass.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

stage=54_causal_candidate_mass
log_path=${N6_STAGE54_LOG:-"$N6_RESULTS_ROOT/n6_54_causal_candidate_mass_v1.log"}
status_path=${N6_STAGE54_STATUS:-"$N6_RESULTS_ROOT/n6_54_causal_candidate_mass_v1.exit"}
plan_path=${N6_STAGE49_OUT:-"$N6_RESULTS_ROOT/n6_provisional_cohort_v1.json"}
activations_path=${N6_STAGE50_OUT:-"$N6_ACTIVATIONS_ROOT/acts_L32_n6_provisional_v1.parquet"}
variants_path=${N6_STAGE52_OUT:-"$N6_RESULTS_ROOT/n6_variants_donor_v1.json"}
recon_path=${N6_STAGE53_OUT:-"$N6_RESULTS_ROOT/n6_recon_v1.json"}
vectors_path=${N6_STAGE53_VECS_OUT:-"$N6_RESULTS_ROOT/n6_recon_vectors_v1.npz"}
checkpoint_path=${N6_STAGE54_CHECKPOINT:-"$N6_RESULTS_ROOT/n6_causal_candidate_mass_checkpoint_v1.jsonl"}
output_path=${N6_STAGE54_OUT:-"$N6_RESULTS_ROOT/n6_causal_candidate_mass_v1.json"}

stage_main() {
    n6_preflight_common || return 1
    n6_verify_stage_inputs \
        "$plan_path" \
        "$activations_path" \
        "$variants_path" \
        "$recon_path" \
        "$vectors_path" || return 1
    n6_require_new_outputs "$output_path" || return 1
    cd "$N6_CODE_ROOT" || return 97
    "$N6_PYTHON" -B "$N6_CODE_ROOT/54_n6_causal_patch.py" \
        --base-model "$N6_BASE_MODEL" \
        --activations "$activations_path" \
        --plan "$plan_path" \
        --variants "$variants_path" \
        --recon-json "$recon_path" \
        --recon-npz "$vectors_path" \
        --model-manifest "$N6_MODEL_MANIFEST" \
        --code-manifest "$N6_CODE_MANIFEST" \
        --prereg "$N6_PREREG" \
        --checkpoint "$checkpoint_path" \
        --out "$output_path" || return $?
    n6_verify_sidecar "$output_path" "stage 54 causal and candidate-mass output"
}

if n6_run_stage "$stage" "$log_path" "$status_path" stage_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
