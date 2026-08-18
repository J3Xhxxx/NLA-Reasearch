#!/usr/bin/env bash
# Frozen J2-P0 supervisor. It always requests host shutdown after a bounded
# pull-ack window, on success or failure.
set -euo pipefail

ROOT=/root/autodl-tmp
CODE="$ROOT/nla_compare"
RESULTS="$ROOT/results"
PY=/root/miniconda3/bin/python
LOG="$RESULTS/j2_sae_projection_supervisor_v1.log"
STATUS="$RESULTS/j2_sae_projection_supervisor_v1.exit"
RESOURCE="$RESULTS/j2_sae_projection_resource_v1.txt"
READY="$RESULTS/j2_sae_projection_pull_ready_v1.txt"
ACK="$RESULTS/j2_sae_projection_pull_ack_v1.txt"
PULL_WAIT_SECONDS=300

PROTOCOL="$RESULTS/J2_SAE_PROJECTION_LANGUAGE_LOOP_PROTOCOL_2026-08-06.md"
SCRIPT66="$CODE/66_j2_sae_projection_loop.py"
SCRIPT67="$CODE/67_j2_sae_projection_causal.py"
SCRIPT68="$CODE/68_j2_sae_projection_analyze.py"
SCRIPT69="$CODE/69_j2_render_case_bundle.py"

AV_CHECKPOINT="$RESULTS/j2_sae_projection_av_checkpoint_v1.jsonl"
EXPLANATIONS="$RESULTS/j2_sae_projection_explanations_v1.json"
VECTORS="$RESULTS/j2_sae_projection_vectors_v1.npz"
RECON="$RESULTS/j2_sae_projection_recon_v1.json"
CAUSAL_CHECKPOINT="$RESULTS/j2_sae_projection_causal_checkpoint_v1.jsonl"
CAUSAL="$RESULTS/j2_sae_projection_causal_v1.json"
ANALYSIS="$RESULTS/j2_sae_projection_analysis_v1.json"
SHORTLIST="$RESULTS/j2_sae_projection_case_shortlist_v1.json"
ANALYSIS_MD="$RESULTS/J2_SAE_PROJECTION_ANALYSIS_V1.md"
CASE_BUNDLE="$RESULTS/j2_sae_projection_case_bundle_v1.json"
CASE_BUNDLE_MD="$RESULTS/J2_SAE_PROJECTION_CASE_BUNDLE_V1.md"

timestamp() {
    date --iso-8601=seconds
}

check_hash() {
    local expected=$1
    local path=$2
    local actual
    [[ -f "$path" ]] || {
        printf 'missing frozen input: %s\n' "$path" >&2
        return 1
    }
    actual=$(sha256sum -- "$path")
    actual=${actual%% *}
    [[ "$actual" == "$expected" ]] || {
        printf 'hash mismatch: %s actual=%s expected=%s\n' \
            "$path" "$actual" "$expected" >&2
        return 1
    }
}

write_sidecar() {
    local path=$1
    local digest
    digest=$(sha256sum -- "$path")
    digest=${digest%% *}
    printf '%s  %s\n' "$digest" "$(basename "$path")" >"${path}.sha256"
}

finalize() {
    local pipeline_exit=$?
    local token deadline ack_value path
    trap - EXIT INT TERM
    set +e
    printf '%s\n' "$pipeline_exit" >"$STATUS"
    {
        printf 'captured_at=%s\n' "$(timestamp)"
        printf 'pipeline_exit=%s\n' "$pipeline_exit"
        printf '\n[df]\n'
        df -h "$ROOT"
        printf '\n[memory]\n'
        free -h
        printf '\n[gpu]\n'
        nvidia-smi \
            --query-gpu=name,uuid,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader
        printf '\n[artifacts]\n'
        for path in \
            "$AV_CHECKPOINT" "$EXPLANATIONS" "$VECTORS" "$RECON" \
            "$CAUSAL_CHECKPOINT" "$CAUSAL" "$ANALYSIS" "$SHORTLIST" \
            "$ANALYSIS_MD" "$CASE_BUNDLE" "$CASE_BUNDLE_MD"; do
            if [[ -f "$path" ]]; then
                sha256sum -- "$path"
            else
                printf 'MISSING  %s\n' "$path"
            fi
        done
    } >"$RESOURCE" 2>&1
    write_sidecar "$STATUS"
    write_sidecar "$RESOURCE"
    token=$(
        printf '%s\037%s\037%s\n' \
            "$pipeline_exit" "$(timestamp)" "$$" | sha256sum
    )
    token=${token%% *}
    {
        printf 'ack_token=%s\n' "$token"
        printf 'pipeline_exit=%s\n' "$pipeline_exit"
        printf 'ready_at=%s\n' "$(timestamp)"
        printf 'ack_file=%s\n' "$ACK"
        printf 'ack_deadline_seconds=%s\n' "$PULL_WAIT_SECONDS"
    } >"$READY"
    write_sidecar "$READY"
    sync

    deadline=$((SECONDS + PULL_WAIT_SECONDS))
    while ((SECONDS < deadline)); do
        if [[ -f "$ACK" ]]; then
            ack_value=$(tr -d '[:space:]' <"$ACK")
            if [[ "$ack_value" == "$token" ]]; then
                break
            fi
        fi
        sleep 5
    done
    printf '[%s] J2_POWER_OFF pipeline_exit=%s\n' \
        "$(timestamp)" "$pipeline_exit"
    sync
    /usr/bin/shutdown -h now
    exit "$pipeline_exit"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$RESULTS"
rm -f -- "$STATUS" "${STATUS}.sha256" "$RESOURCE" "${RESOURCE}.sha256" \
    "$READY" "${READY}.sha256" "$ACK"
cd "$CODE"
export NLA_REPO="$ROOT/nla_repo"
export PYTHONUNBUFFERED=1

printf '[%s] J2_START\n' "$(timestamp)"
check_hash a41b7d89893a270218bf79e226c3e3d7a8726f71ca1fe6d41f40b583616a700f "$PROTOCOL"
check_hash 9e042a46ee469046b75bc9c634fabec3aeecbdf7dd9d72f1ab9190c0177c559f "$SCRIPT66"
check_hash 27880e7b2a4b87d9367237043dfa8062a7d3ca9a890db82182e24b288174b77b "$SCRIPT67"
check_hash d2c5f3809f66b7eb92a2e17a9b89210f51a03aa387d050e6fb6bd2c8fa24355a "$SCRIPT68"
check_hash 64a3795d6b2a31834c21634144cd1f291f1214c8a413d53e1c5f3f501da944ef "$SCRIPT69"
check_hash eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66 "$ROOT/activations/acts_L32_n3_v1.parquet"
check_hash e9d89713dc64381a52f05224d6522abb0ec547777a8c6a7f08b841a72a339967 "$RESULTS/n4_recon_vectors_v1.npz"
check_hash b656ded845c8fd122e4dcb1391ba5d81e1a903f80a69c30575bf26910e200942 "$RESULTS/n4_explanations_v1.json"
check_hash 8dd532f65d8c9c153f04ba433cc6f160798598fbbcbee388c15fb4a75a366233 "$RESULTS/n4_causal_patch_v1.json"
check_hash 3c8a4d87d7289ac6c41b58e2bbdd6955585db46eaaa5306822d9d802259943cc "$RESULTS/n4_analysis_v1.json"
check_hash 4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735 "$RESULTS/n5_model_weights_v1.sha256"
check_hash 69fb1b40d60d075c615acdaa23acf4f85c17b5b4cf02e2cc18113c4e14ecf63a "$CODE/pilot_common.py"

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader

"$PY" -B "$SCRIPT66" \
    --av "$ROOT/models/nla-gemma3-12b-L32-av" \
    --ar "$ROOT/models/nla-gemma3-12b-L32-ar" \
    --sae-small "$ROOT/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small" \
    --sae-big "$ROOT/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_big" \
    --activations "$ROOT/activations/acts_L32_n3_v1.parquet" \
    --n4-vectors "$RESULTS/n4_recon_vectors_v1.npz" \
    --n4-explanations "$RESULTS/n4_explanations_v1.json" \
    --model-manifest "$RESULTS/n5_model_weights_v1.sha256" \
    --pilot-common "$CODE/pilot_common.py" \
    --protocol "$PROTOCOL" \
    --checkpoint "$AV_CHECKPOINT" \
    --explanations-out "$EXPLANATIONS" \
    --vectors-out "$VECTORS" \
    --out "$RECON"

"$PY" -B "$SCRIPT67" \
    --base-model "$ROOT/models/gemma-3-12b-it" \
    --activations "$ROOT/activations/acts_L32_n3_v1.parquet" \
    --j2-vectors "$VECTORS" \
    --j2-result "$RECON" \
    --j2-explanations "$EXPLANATIONS" \
    --j2-av-checkpoint "$AV_CHECKPOINT" \
    --j2-recon-script "$SCRIPT66" \
    --protocol "$PROTOCOL" \
    --model-manifest "$RESULTS/n5_model_weights_v1.sha256" \
    --checkpoint "$CAUSAL_CHECKPOINT" \
    --out "$CAUSAL"

"$PY" -B "$SCRIPT68" \
    --activations "$ROOT/activations/acts_L32_n3_v1.parquet" \
    --n4-vectors "$RESULTS/n4_recon_vectors_v1.npz" \
    --n4-explanations "$RESULTS/n4_explanations_v1.json" \
    --n4-causal "$RESULTS/n4_causal_patch_v1.json" \
    --n4-analysis "$RESULTS/n4_analysis_v1.json" \
    --model-manifest "$RESULTS/n5_model_weights_v1.sha256" \
    --pilot-common "$CODE/pilot_common.py" \
    --j2-explanations "$EXPLANATIONS" \
    --j2-av-checkpoint "$AV_CHECKPOINT" \
    --j2-vectors "$VECTORS" \
    --j2-result "$RECON" \
    --j2-causal "$CAUSAL" \
    --j2-causal-checkpoint "$CAUSAL_CHECKPOINT" \
    --recon-script "$SCRIPT66" \
    --causal-script "$SCRIPT67" \
    --protocol "$PROTOCOL" \
    --out "$ANALYSIS" \
    --shortlist-out "$SHORTLIST" \
    --markdown "$ANALYSIS_MD"

"$PY" -B "$SCRIPT69" \
    --shortlist "$SHORTLIST" \
    --analysis "$ANALYSIS" \
    --protocol "$PROTOCOL" \
    --j2-explanations "$EXPLANATIONS" \
    --j2-result "$RECON" \
    --n4-causal "$RESULTS/n4_causal_patch_v1.json" \
    --j2-causal "$CAUSAL" \
    --out "$CASE_BUNDLE" \
    --markdown "$CASE_BUNDLE_MD"

printf '[%s] J2_COMPLETE\n' "$(timestamp)"
