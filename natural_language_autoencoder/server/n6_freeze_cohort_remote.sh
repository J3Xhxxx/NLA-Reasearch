#!/usr/bin/env bash
# Stage 49: freeze the provisional, tokenizer-only N6 cohort.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

stage=49_freeze_cohort
log_path=${N6_STAGE49_LOG:-"$N6_RESULTS_ROOT/n6_49_freeze_cohort_v1.log"}
status_path=${N6_STAGE49_STATUS:-"$N6_RESULTS_ROOT/n6_49_freeze_cohort_v1.exit"}
output_path=${N6_STAGE49_OUT:-"$N6_RESULTS_ROOT/n6_provisional_cohort_v1.json"}

stage_main() {
    n6_require_env \
        N6_SOURCE_CORPUS \
        N6_SOURCE_CORPUS_MANIFEST \
        N6_N4_ACTIVATIONS \
        N6_N5_COHORT_PLAN \
        N6_PILE_PARQUET || return 1
    n6_preflight_common || return 1
    n6_verify_exact_sha256 \
        "$N6_SOURCE_CORPUS" \
        d40069ab51c294ecbe3e76845d1f2f4dff1bb66a6061c5b6b4c612f7d0ff8816 \
        "frozen N3 corpus JSONL" || return 1
    n6_verify_exact_sha256 \
        "$N6_SOURCE_CORPUS_MANIFEST" \
        500d5b88b78c8bc06ff7965c0dffcc25cb5b0e9f50bfa8ec1ae009f9312d6046 \
        "frozen N3 corpus manifest" || return 1
    n6_verify_exact_sha256 \
        "$N6_N4_ACTIVATIONS" \
        eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66 \
        "frozen N4 activation cohort" || return 1
    n6_verify_exact_sha256 \
        "$N6_PILE_PARQUET" \
        a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31 \
        "frozen Pile parquet" || return 1
    n6_verify_stage_inputs "$N6_N5_COHORT_PLAN" || return 1
    n6_require_new_outputs "$output_path" || return 1
    cd "$N6_CODE_ROOT" || return 97
    "$N6_PYTHON" -B "$N6_CODE_ROOT/49_n6_freeze_cohort.py" \
        --corpus "$N6_SOURCE_CORPUS" \
        --corpus-manifest "$N6_SOURCE_CORPUS_MANIFEST" \
        --n4-activations "$N6_N4_ACTIVATIONS" \
        --n5-plan "$N6_N5_COHORT_PLAN" \
        --pile-parquet "$N6_PILE_PARQUET" \
        --base-model "$N6_BASE_MODEL" \
        --model-manifest "$N6_MODEL_MANIFEST" \
        --code-manifest "$N6_CODE_MANIFEST" \
        --prereg "$N6_PREREG" \
        --out "$output_path" || return $?
    n6_verify_sidecar "$output_path" "stage 49 output"
}

if n6_run_stage "$stage" "$log_path" "$status_path" stage_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
