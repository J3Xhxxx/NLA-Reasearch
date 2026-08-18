#!/usr/bin/env bash
# Shared, fail-closed helpers for the N6 stage runners.
#
# This file is a template dependency and must be sourced, not executed.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'n6_stage_common.sh must be sourced by an N6 runner\n' >&2
    exit 64
fi

N6_ROOT=${N6_ROOT:-/root/autodl-tmp}
N6_CODE_ROOT=${N6_CODE_ROOT:-"$N6_ROOT/nla_compare"}
N6_RESULTS_ROOT=${N6_RESULTS_ROOT:-"$N6_ROOT/results"}
N6_ACTIVATIONS_ROOT=${N6_ACTIVATIONS_ROOT:-"$N6_ROOT/activations"}
N6_BASE_MODEL=${N6_BASE_MODEL:-"$N6_ROOT/models/gemma-3-12b-it"}
N6_AV_MODEL=${N6_AV_MODEL:-"$N6_ROOT/models/nla-gemma3-12b-L32-av"}
N6_AR_MODEL=${N6_AR_MODEL:-"$N6_ROOT/models/nla-gemma3-12b-L32-ar"}
N6_SAE_BIG_MODEL=${N6_SAE_BIG_MODEL:-"$N6_ROOT/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_big"}
N6_MODEL_MANIFEST=${N6_MODEL_MANIFEST:-"$N6_RESULTS_ROOT/n5_model_weights_v1.sha256"}
N6_PYTHON=/root/miniconda3/bin/python

n6_timestamp() {
    date '+%F %T %Z'
}

n6_log() {
    printf '%s %s\n' "$(n6_timestamp)" "$*"
}

n6_error() {
    printf 'N6_PRECONDITION_FAILED: %s\n' "$*" >&2
    return 1
}

n6_require_env() {
    local name
    for name in "$@"; do
        if ! [[ -v "$name" ]] || [[ -z "${!name}" ]]; then
            n6_error "required environment variable is unset or empty: $name"
            return 1
        fi
    done
}

n6_require_positive_integer() {
    local name=$1
    n6_require_env "$name" || return 1
    if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        n6_error "$name must be a positive integer, found ${!name@Q}"
        return 1
    fi
}

n6_require_nonnegative_integer() {
    local name=$1
    n6_require_env "$name" || return 1
    if ! [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        n6_error "$name must be a nonnegative integer, found ${!name@Q}"
        return 1
    fi
}

n6_require_nonnegative_number() {
    local name=$1
    n6_require_env "$name" || return 1
    if ! [[ "${!name}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?$ ]]; then
        n6_error "$name must be a nonnegative decimal number, found ${!name@Q}"
        return 1
    fi
}

n6_require_boolean() {
    local name=$1
    n6_require_env "$name" || return 1
    if [[ "${!name}" != "true" && "${!name}" != "false" ]]; then
        n6_error "$name must be exactly true or false, found ${!name@Q}"
        return 1
    fi
}

n6_require_positive_integer_csv() {
    local name=$1
    n6_require_env "$name" || return 1
    if ! [[ "${!name}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
        n6_error "$name must be comma-separated positive integers, found ${!name@Q}"
        return 1
    fi
}

n6_require_file() {
    local path=$1
    local label=${2:-input}
    if [[ ! -f "$path" || ! -r "$path" ]]; then
        n6_error "$label is not a readable regular file: $path"
        return 1
    fi
}

n6_verify_exact_sha256() {
    local path=$1
    local expected=$2
    local label=${3:-legacy input}
    local actual
    n6_require_file "$path" "$label" || return 1
    if ! [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]]; then
        n6_error "$label has a malformed frozen expected SHA-256"
        return 1
    fi
    actual=$(sha256sum -- "$path") || return 1
    actual=${actual%% *}
    if [[ "${actual,,}" != "${expected,,}" ]]; then
        n6_error "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
        return 1
    fi
    n6_log "VERIFIED exact hash label=${label@Q} sha256=$actual path=$path"
}

n6_absolute_file() {
    local path=$1
    local directory
    directory=$(cd "$(dirname "$path")" && pwd -P) || return 1
    printf '%s/%s\n' "$directory" "$(basename "$path")"
}

n6_verify_sidecar() {
    local path=$1
    local label=${2:-input}
    local sidecar="${path}.sha256"
    local -a lines=()
    local declared declared_name extra actual

    n6_require_file "$path" "$label" || return 1
    n6_require_file "$sidecar" "$label SHA-256 sidecar" || return 1
    mapfile -t lines <"$sidecar"
    if [[ ${#lines[@]} -ne 1 ]]; then
        n6_error "$label sidecar must contain exactly one line: $sidecar"
        return 1
    fi
    read -r declared declared_name extra <<<"${lines[0]}"
    if [[ -z "$declared" || -z "$declared_name" || -n "${extra:-}" ]]; then
        n6_error "$label sidecar must be '<sha256>  <basename>': $sidecar"
        return 1
    fi
    if ! [[ "$declared" =~ ^[0-9a-fA-F]{64}$ ]]; then
        n6_error "$label sidecar has a malformed SHA-256: $sidecar"
        return 1
    fi
    if [[ "$declared_name" != "$(basename "$path")" ]]; then
        n6_error "$label sidecar names ${declared_name@Q}, expected $(basename "$path")"
        return 1
    fi
    actual=$(sha256sum -- "$path") || return 1
    actual=${actual%% *}
    if [[ "${actual,,}" != "${declared,,}" ]]; then
        n6_error "$label SHA-256 mismatch: declared=$declared actual=$actual path=$path"
        return 1
    fi
    n6_log "VERIFIED sidecar label=${label@Q} sha256=$actual path=$path"
}

n6_verify_manifest() {
    local manifest=$1
    local label=$2
    local check_root=${3:-}
    local absolute_manifest

    n6_verify_sidecar "$manifest" "$label" || return 1
    absolute_manifest=$(n6_absolute_file "$manifest") || return 1
    if [[ -n "$check_root" ]]; then
        if [[ ! -d "$check_root" ]]; then
            n6_error "$label check root is not a directory: $check_root"
            return 1
        fi
        (
            cd "$check_root" &&
                sha256sum --strict --check "$absolute_manifest"
        ) || {
            n6_error "$label entry verification failed: $manifest"
            return 1
        }
    else
        sha256sum --strict --check "$absolute_manifest" || {
            n6_error "$label entry verification failed: $manifest"
            return 1
        }
    fi
    n6_log "VERIFIED manifest entries label=${label@Q} path=$manifest"
}

n6_verify_code_manifest() {
    local manifest=$1
    local line_number=0
    local digest name resolved actual

    n6_verify_sidecar "$manifest" "frozen N6 code manifest" || return 1
    while read -r digest name; do
        line_number=$((line_number + 1))
        name=${name#\*}
        if ! [[ "$digest" =~ ^[0-9a-fA-F]{64}$ ]] ||
            ! [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
            n6_error "malformed N6 code manifest line $line_number"
            return 1
        fi
        if [[ "$name" == "nla_inference.py" ]]; then
            resolved="${NLA_REPO:-/root/autodl-tmp/nla_repo}/nla_inference.py"
        else
            resolved="$N6_CODE_ROOT/$name"
        fi
        if [[ ! -f "$resolved" ]]; then
            n6_error "frozen N6 code entry is missing: $resolved"
            return 1
        fi
        actual=$(sha256sum -- "$resolved") || return 1
        actual=${actual%% *}
        if [[ "${actual,,}" != "${digest,,}" ]]; then
            n6_error \
                "frozen N6 code entry differs: name=$name expected=$digest actual=$actual"
            return 1
        fi
    done <"$manifest"
    if ((line_number == 0)); then
        n6_error "frozen N6 code manifest is empty"
        return 1
    fi
    n6_log "VERIFIED frozen N6 code manifest entries=$line_number path=$manifest"
}

n6_preflight_common() {
    n6_require_env N6_PREREG N6_CODE_MANIFEST || return 1
    if [[ "$N6_PREREG" == *".DRAFT"* ]]; then
        n6_error "refusing a draft preregistration: $N6_PREREG"
        return 1
    fi
    if [[ ! -x "$N6_PYTHON" ]]; then
        n6_error "required Python is not executable: $N6_PYTHON"
        return 1
    fi
    command -v sha256sum >/dev/null 2>&1 || {
        n6_error "sha256sum is unavailable"
        return 1
    }
    n6_verify_sidecar "$N6_PREREG" "binding N6 preregistration" || return 1
    n6_verify_code_manifest "$N6_CODE_MANIFEST" || return 1
    n6_verify_manifest \
        "$N6_MODEL_MANIFEST" \
        "frozen N5 combined model manifest" || return 1
}

n6_verify_stage_inputs() {
    local path
    for path in "$@"; do
        n6_verify_sidecar "$path" "stage input" || return 1
    done
}

n6_require_new_outputs() {
    local path
    for path in "$@"; do
        if [[ -e "$path" || -e "${path}.sha256" ]]; then
            n6_error "refusing to overwrite a frozen output or sidecar: $path"
            return 1
        fi
    done
}

n6_atomic_write_line() {
    local path=$1
    local value=$2
    local temporary="${path}.tmp.$$"
    mkdir -p "$(dirname "$path")" || return 1
    printf '%s\n' "$value" >"$temporary" || return 1
    mv -f -- "$temporary" "$path"
}

n6_write_sidecar() {
    local path=$1
    local digest temporary
    n6_require_file "$path" "sidecar target" || return 1
    digest=$(sha256sum -- "$path") || return 1
    digest=${digest%% *}
    temporary="${path}.sha256.tmp.$$"
    printf '%s  %s\n' "$digest" "$(basename "$path")" >"$temporary" || return 1
    mv -f -- "$temporary" "${path}.sha256"
}

n6_run_stage() {
    local stage=$1
    local log_path=$2
    local status_path=$3
    shift 3
    local status_temporary="${status_path}.tmp.$$"
    local exit_code=99

    mkdir -p "$(dirname "$log_path")" "$(dirname "$status_path")" || return 1
    rm -f -- "$status_path" "$status_temporary"
    {
        n6_log "STAGE_START stage=$stage"
        if "$@"; then
            exit_code=0
            n6_log "STAGE_COMPLETE stage=$stage"
        else
            exit_code=$?
            n6_log "STAGE_FAILED stage=$stage exit=$exit_code"
        fi
    } >"$log_path" 2>&1
    printf '%s\n' "$exit_code" >"$status_temporary" || return 1
    mv -f -- "$status_temporary" "$status_path" || return 1
    return "$exit_code"
}
