#!/usr/bin/env bash
set -uo pipefail

root=/root/autodl-tmp
code=$root/nla_compare
results=$root/results
log=$results/n5_resume_heldout_and_shutdown.log
status=$results/n5_resume_heldout_and_shutdown.exit

rm -f "$status"

stamp() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >>"$log"
}

run_stage() {
    local stage=$1
    local runner=$2
    local stage_status=$3

    stamp "START stage=$stage runner=$runner"
    rm -f "$stage_status"
    "$runner"

    if [ ! -f "$stage_status" ]; then
        stamp "FAILED stage=$stage reason=missing_status"
        return 98
    fi

    local exit_code
    exit_code=$(tr -d '[:space:]' <"$stage_status")
    if [ "$exit_code" != "0" ]; then
        stamp "FAILED stage=$stage exit=$exit_code"
        return "$exit_code"
    fi
    stamp "COMPLETE stage=$stage"
}

finalize() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [ "$exit_code" -eq 0 ]; then
        stamp "PIPELINE_COMPLETE"
    else
        stamp "PIPELINE_FAILED exit=$exit_code"
    fi
    printf '%s\n' "$exit_code" >"$status"
    sync
    stamp "POWER_OFF exit=$exit_code"
    /usr/bin/systemctl poweroff
    exit "$exit_code"
}
trap finalize EXIT INT TERM

cd "$code"
stamp "SUPERVISOR_START checkpoint_rows=$(wc -l <"$results/n5_heldout_av_checkpoint_v2.jsonl")"

# Held-out activations and the discovery-frozen gate already passed their stages.
# Reconstruction resumes from its append-only, contract-checked AV checkpoint.
run_stage \
    heldout_reconstruct \
    "$code/n5_heldout_reconstruct_remote.sh" \
    "$results/n5_heldout_reconstruct_v2.exit" || exit $?
run_stage \
    heldout_causal \
    "$code/n5_heldout_causal_remote.sh" \
    "$results/n5_heldout_causal_v2.exit" || exit $?
run_stage \
    analyze \
    "$code/n5_analyze_remote.sh" \
    "$results/n5_analyze_v2.exit" || exit $?
