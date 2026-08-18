#!/usr/bin/env bash
set -uo pipefail

root=/root/autodl-tmp
code=$root/nla_compare
results=$root/results
pipeline_log=$results/n5_pipeline_after_recon_v2.log
pipeline_status=$results/n5_pipeline_after_recon_v2.exit

rm -f "$pipeline_status"

stamp() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >>"$pipeline_log"
}

wait_for_status() {
    local stage=$1
    local path=$2
    while [ ! -f "$path" ]; do
        sleep 10
    done
    local exit_code
    exit_code=$(tr -d '[:space:]' <"$path")
    if [ "$exit_code" != "0" ]; then
        stamp "FAILED stage=$stage exit=$exit_code"
        printf 'failed:%s:%s\n' "$stage" "$exit_code" >"$pipeline_status"
        exit 1
    fi
    stamp "COMPLETE stage=$stage"
}

run_stage() {
    local stage=$1
    local runner=$2
    local status_path=$3
    stamp "START stage=$stage runner=$runner"
    "$runner"
    wait_for_status "$stage" "$status_path"
}

stamp "SUPERVISOR_START"
wait_for_status \
    discovery_reconstruct \
    "$results/n5_discovery_reconstruct_v2.exit"
run_stage \
    discovery_causal \
    "$code/n5_discovery_causal_remote.sh" \
    "$results/n5_discovery_causal_v2.exit"
run_stage \
    freeze_gate \
    "$code/n5_freeze_gate_remote.sh" \
    "$results/n5_freeze_gate_v2.exit"
run_stage \
    heldout_reconstruct \
    "$code/n5_heldout_reconstruct_remote.sh" \
    "$results/n5_heldout_reconstruct_v2.exit"
run_stage \
    heldout_causal \
    "$code/n5_heldout_causal_remote.sh" \
    "$results/n5_heldout_causal_v2.exit"
run_stage \
    analyze \
    "$code/n5_analyze_remote.sh" \
    "$results/n5_analyze_v2.exit"
stamp "PIPELINE_COMPLETE"
printf '0\n' >"$pipeline_status"
