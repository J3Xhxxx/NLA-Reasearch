#!/usr/bin/env bash
# Sequential N6 supervisor template. It always powers the server off.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=n6_stage_common.sh
source "$script_dir/n6_stage_common.sh"

supervisor_log=${N6_SUPERVISOR_LOG:-"$N6_RESULTS_ROOT/n6_supervisor_v1.log"}
supervisor_status=${N6_SUPERVISOR_STATUS:-"$N6_RESULTS_ROOT/n6_supervisor_v1.exit"}
resource_report=${N6_RESOURCE_REPORT:-"$N6_RESULTS_ROOT/n6_resource_report_v1.txt"}
pull_ready_file=${N6_PULL_READY_FILE:-"$N6_RESULTS_ROOT/n6_pull_ready_v1.txt"}
pull_ack_file=${N6_PULL_ACK_FILE:-"$N6_RESULTS_ROOT/n6_pull_ack_v1.txt"}
pull_timeout=${N6_PULL_ACK_TIMEOUT_SECONDS:-300}
pull_poll=${N6_PULL_ACK_POLL_SECONDS:-5}
N6_PREPARED_PULL_TOKEN=

supervisor_stamp() {
    printf '%s %s\n' "$(n6_timestamp)" "$*" >>"$supervisor_log"
}

stage_status_path() {
    case "$1" in
        49) printf '%s\n' "${N6_STAGE49_STATUS:-"$N6_RESULTS_ROOT/n6_49_freeze_cohort_v1.exit"}" ;;
        50) printf '%s\n' "${N6_STAGE50_STATUS:-"$N6_RESULTS_ROOT/n6_50_extract_activations_v1.exit"}" ;;
        51) printf '%s\n' "${N6_STAGE51_STATUS:-"$N6_RESULTS_ROOT/n6_51_generate_av_v1.exit"}" ;;
        52) printf '%s\n' "${N6_STAGE52_STATUS:-"$N6_RESULTS_ROOT/n6_52_freeze_variants_donor_v1.exit"}" ;;
        53) printf '%s\n' "${N6_STAGE53_STATUS:-"$N6_RESULTS_ROOT/n6_53_reconstruct_v1.exit"}" ;;
        54) printf '%s\n' "${N6_STAGE54_STATUS:-"$N6_RESULTS_ROOT/n6_54_causal_candidate_mass_v1.exit"}" ;;
        55) printf '%s\n' "${N6_STAGE55_STATUS:-"$N6_RESULTS_ROOT/n6_55_analyze_v1.exit"}" ;;
        56) printf '%s\n' "${N6_STAGE56_STATUS:-"$N6_RESULTS_ROOT/n6_56_independent_audit_v1.exit"}" ;;
        *) return 64 ;;
    esac
}

n6_supervisor_run_stage() {
    local number=$1
    local name=$2
    local runner=$3
    local status_path runner_exit stage_exit
    status_path=$(stage_status_path "$number") || return $?

    if [[ ! -f "$runner" ]]; then
        supervisor_stamp "FAILED stage=$number:$name reason=missing_runner path=$runner"
        return 97
    fi
    rm -f -- "$status_path"
    supervisor_stamp "START stage=$number:$name runner=$runner"
    if bash "$runner"; then
        runner_exit=0
    else
        runner_exit=$?
    fi
    if [[ ! -f "$status_path" ]]; then
        supervisor_stamp "FAILED stage=$number:$name reason=missing_status runner_exit=$runner_exit"
        return 98
    fi
    stage_exit=$(tr -d '[:space:]' <"$status_path")
    if ! [[ "$stage_exit" =~ ^(0|[1-9][0-9]*)$ ]] || ((stage_exit > 255)); then
        supervisor_stamp "FAILED stage=$number:$name reason=malformed_status value=${stage_exit@Q}"
        return 99
    fi
    if ((runner_exit != stage_exit)); then
        supervisor_stamp \
            "FAILED stage=$number:$name reason=status_disagrees runner_exit=$runner_exit status_exit=$stage_exit"
        return 100
    fi
    if ((stage_exit != 0)); then
        supervisor_stamp "FAILED stage=$number:$name exit=$stage_exit"
        return "$stage_exit"
    fi
    supervisor_stamp "COMPLETE stage=$number:$name"
}

n6_capture_resource_report() {
    local pipeline_exit=$1
    local temporary="${resource_report}.tmp.$$"
    mkdir -p "$(dirname "$resource_report")" || return 1
    {
        printf 'captured_at=%s\n' "$(n6_timestamp)"
        printf 'pipeline_exit=%s\n' "$pipeline_exit"
        printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf unavailable)"
        printf 'prereg=%s\n' "${N6_PREREG-UNSET}"
        printf 'code_manifest=%s\n' "${N6_CODE_MANIFEST-UNSET}"
        printf 'model_manifest=%s\n' "$N6_MODEL_MANIFEST"
        printf '\n[df]\n'
        df -h "$N6_ROOT" 2>&1 || true
        printf '\n[memory]\n'
        free -h 2>&1 || true
        printf '\n[gpu]\n'
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi \
                --query-gpu=name,uuid,memory.total,memory.used,utilization.gpu \
                --format=csv,noheader 2>&1 || true
        else
            printf 'nvidia-smi unavailable\n'
        fi
    } >"$temporary" || return 1
    mv -f -- "$temporary" "$resource_report" || return 1
    n6_write_sidecar "$resource_report"
}

n6_prepare_pull_ready() {
    local prereg_digest code_manifest_digest model_manifest_digest
    local analysis_path audit_path analysis_digest audit_digest report_digest
    local token temporary
    analysis_path=${N6_STAGE55_OUT:-"$N6_RESULTS_ROOT/n6_analysis_v1.json"}
    audit_path=${N6_STAGE56_OUT:-"$N6_RESULTS_ROOT/n6_independent_audit_v1.json"}
    n6_verify_sidecar "$analysis_path" "pull-ready analysis" || return 1
    n6_verify_sidecar "$audit_path" "pull-ready independent audit" || return 1
    n6_verify_sidecar "$resource_report" "pull-ready resource report" || return 1
    prereg_digest=$(sha256sum -- "$N6_PREREG")
    prereg_digest=${prereg_digest%% *}
    code_manifest_digest=$(sha256sum -- "$N6_CODE_MANIFEST")
    code_manifest_digest=${code_manifest_digest%% *}
    model_manifest_digest=$(sha256sum -- "$N6_MODEL_MANIFEST")
    model_manifest_digest=${model_manifest_digest%% *}
    analysis_digest=$(sha256sum -- "$analysis_path")
    analysis_digest=${analysis_digest%% *}
    audit_digest=$(sha256sum -- "$audit_path")
    audit_digest=${audit_digest%% *}
    report_digest=$(sha256sum -- "$resource_report")
    report_digest=${report_digest%% *}
    token=$(
        printf '%s\037%s\037%s\037%s\037%s\037%s\037%s\n' \
            "$prereg_digest" \
            "$code_manifest_digest" \
            "$model_manifest_digest" \
            "$analysis_digest" \
            "$audit_digest" \
            "$report_digest" \
            "$(date '+%s').$$" |
            sha256sum
    ) || return 1
    token=${token%% *}
    temporary="${pull_ready_file}.tmp.$$"
    mkdir -p "$(dirname "$pull_ready_file")" || return 1
    {
        printf 'ack_token=%s\n' "$token"
        printf 'ready_at=%s\n' "$(n6_timestamp)"
        printf 'analysis=%s\n' "$analysis_path"
        printf 'analysis_sha256=%s\n' "$analysis_digest"
        printf 'independent_audit=%s\n' "$audit_path"
        printf 'independent_audit_sha256=%s\n' "$audit_digest"
        printf 'resource_report=%s\n' "$resource_report"
        printf 'resource_report_sha256=%s\n' "$report_digest"
        printf 'ack_file=%s\n' "$pull_ack_file"
        printf 'ack_format=exact ack_token value followed by newline\n'
        printf 'ack_deadline_seconds=%s\n' "$pull_timeout"
    } >"$temporary" || return 1
    mv -f -- "$temporary" "$pull_ready_file" || return 1
    n6_write_sidecar "$pull_ready_file" || return 1
    N6_PREPARED_PULL_TOKEN=$token
}

n6_wait_for_pull_ack() {
    local token=$1
    local deadline ack_value remaining sleep_for
    if ((pull_timeout == 0)); then
        supervisor_stamp "PULL_ACK_SKIPPED timeout=0 ready=$pull_ready_file"
        return 0
    fi
    deadline=$((SECONDS + pull_timeout))
    supervisor_stamp \
        "PULL_READY ready=$pull_ready_file ack=$pull_ack_file timeout=${pull_timeout}s"
    while ((SECONDS < deadline)); do
        if [[ -f "$pull_ack_file" ]]; then
            ack_value=$(tr -d '[:space:]' <"$pull_ack_file" 2>/dev/null || true)
            if [[ "$ack_value" == "$token" ]]; then
                supervisor_stamp "PULL_ACKNOWLEDGED ack=$pull_ack_file"
                return 0
            fi
        fi
        remaining=$((deadline - SECONDS))
        sleep_for=$pull_poll
        if ((sleep_for > remaining)); then
            sleep_for=$remaining
        fi
        if ((sleep_for > 0)); then
            sleep "$sleep_for"
        fi
    done
    supervisor_stamp "PULL_ACK_TIMEOUT seconds=$pull_timeout; powering off anyway"
    return 0
}

n6_finalize() {
    local original_exit=$?
    local final_exit=$original_exit
    local pull_token
    trap - EXIT
    trap '' INT TERM

    if ! n6_capture_resource_report "$original_exit" >>"$supervisor_log" 2>&1; then
        supervisor_stamp "RESOURCE_REPORT_FAILED path=$resource_report"
        if ((final_exit == 0)); then
            final_exit=95
        fi
    else
        supervisor_stamp "RESOURCE_REPORT_COMPLETE path=$resource_report"
    fi

    if ((final_exit == 0)); then
        if n6_prepare_pull_ready >>"$supervisor_log" 2>&1; then
            pull_token=$N6_PREPARED_PULL_TOKEN
            n6_atomic_write_line "$supervisor_status" "$final_exit" || final_exit=94
            if ((final_exit == 0)); then
                n6_wait_for_pull_ack "$pull_token"
            fi
        else
            final_exit=96
            supervisor_stamp "PULL_READY_FAILED path=$pull_ready_file"
        fi
    fi

    n6_atomic_write_line "$supervisor_status" "$final_exit" || true
    if ((final_exit == 0)); then
        supervisor_stamp "SUPERVISOR_COMPLETE exit=0"
    else
        supervisor_stamp "SUPERVISOR_FAILED exit=$final_exit original_exit=$original_exit"
    fi
    supervisor_stamp "POWER_OFF exit=$final_exit"
    sync; /usr/bin/shutdown -h now
    exit "$final_exit"
}

supervisor_main() {
    if ! [[ "$pull_timeout" =~ ^(0|[1-9][0-9]*)$ ]] || ((pull_timeout > 300)); then
        supervisor_stamp "INVALID pull timeout: $pull_timeout (allowed 0..300)"
        return 64
    fi
    if ! [[ "$pull_poll" =~ ^[1-9][0-9]*$ ]] || ((pull_poll > 60)); then
        supervisor_stamp "INVALID pull polling interval: $pull_poll (allowed 1..60)"
        return 64
    fi
    n6_preflight_common >>"$supervisor_log" 2>&1 || return $?
    n6_supervisor_run_stage \
        49 freeze_cohort "$N6_CODE_ROOT/n6_freeze_cohort_remote.sh" || return $?
    n6_supervisor_run_stage \
        50 extract_activations "$N6_CODE_ROOT/n6_extract_activations_remote.sh" || return $?
    n6_supervisor_run_stage \
        51 generate_av "$N6_CODE_ROOT/n6_generate_av_remote.sh" || return $?
    n6_supervisor_run_stage \
        52 freeze_variants_donor "$N6_CODE_ROOT/n6_freeze_variants_donor_remote.sh" || return $?
    n6_supervisor_run_stage \
        53 reconstruct "$N6_CODE_ROOT/n6_reconstruct_remote.sh" || return $?
    n6_supervisor_run_stage \
        54 causal_candidate_mass "$N6_CODE_ROOT/n6_causal_candidate_mass_remote.sh" || return $?
    n6_supervisor_run_stage \
        55 analyze "$N6_CODE_ROOT/n6_analyze_remote.sh" || return $?
    n6_supervisor_run_stage \
        56 independent_audit "$N6_CODE_ROOT/n6_independent_audit_remote.sh" || return $?
}

trap n6_finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$N6_RESULTS_ROOT"
rm -f -- "$supervisor_status"
: >"$supervisor_log"
supervisor_stamp "SUPERVISOR_START"
if supervisor_main; then
    exit 0
else
    exit_code=$?
    exit "$exit_code"
fi
