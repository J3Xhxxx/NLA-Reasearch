#!/usr/bin/env bash
# Stage 56: independently recompute N6 endpoints from frozen raw artifacts.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

stage=56_independent_audit
log_path=${N6_STAGE56_LOG:-"$N6_RESULTS_ROOT/n6_56_independent_audit_v1.log"}
status_path=${N6_STAGE56_STATUS:-"$N6_RESULTS_ROOT/n6_56_independent_audit_v1.exit"}
plan_path=${N6_STAGE49_OUT:-"$N6_RESULTS_ROOT/n6_provisional_cohort_v1.json"}
variants_path=${N6_STAGE52_OUT:-"$N6_RESULTS_ROOT/n6_variants_donor_v1.json"}
recon_path=${N6_STAGE53_OUT:-"$N6_RESULTS_ROOT/n6_recon_v1.json"}
vectors_path=${N6_STAGE53_VECS_OUT:-"$N6_RESULTS_ROOT/n6_recon_vectors_v1.npz"}
causal_path=${N6_STAGE54_OUT:-"$N6_RESULTS_ROOT/n6_causal_candidate_mass_v1.json"}
analysis_path=${N6_STAGE55_OUT:-"$N6_RESULTS_ROOT/n6_analysis_v1.json"}
n5_gate_path=${N6_N5_GATE:-"$N6_RESULTS_ROOT/n5_gate_v2.json"}
output_path=${N6_STAGE56_OUT:-"$N6_RESULTS_ROOT/n6_independent_audit_v1.json"}

stage_main() {
    n6_preflight_common || return 1
    n6_verify_stage_inputs \
        "$plan_path" \
        "$variants_path" \
        "$recon_path" \
        "$vectors_path" \
        "$causal_path" \
        "$analysis_path" \
        "$n5_gate_path" || return 1
    n6_require_new_outputs "$output_path" || return 1
    cd "$N6_CODE_ROOT" || return 97
    "$N6_PYTHON" -B "$N6_CODE_ROOT/56_n6_independent_audit.py" \
        --variants "$variants_path" \
        --recon-json "$recon_path" \
        --recon-npz "$vectors_path" \
        --causal "$causal_path" \
        --analysis "$analysis_path" \
        --plan "$plan_path" \
        --n5-gate "$n5_gate_path" \
        --model-manifest "$N6_MODEL_MANIFEST" \
        --code-manifest "$N6_CODE_MANIFEST" \
        --prereg "$N6_PREREG" \
        --out "$output_path" || return $?
    n6_verify_sidecar "$output_path" "stage 56 independent audit"
}

if n6_run_stage "$stage" "$log_path" "$status_path" stage_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
