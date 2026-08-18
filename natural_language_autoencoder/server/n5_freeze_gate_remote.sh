#!/usr/bin/env bash
set -uo pipefail

root=/root/autodl-tmp
code=$root/nla_compare
results=$root/results
log=$results/n5_freeze_gate_v2.log
status=$results/n5_freeze_gate_v2.exit

rm -f "$status"
cd "$code" || exit 97
if /root/miniconda3/bin/python -B 46_n5_freeze_gate.py \
    --recon-json "$results/n5_discovery_recon_v2.json" \
    --recon-npz "$results/n5_discovery_recon_vectors_v2.npz" \
    --causal "$results/n5_discovery_causal_v2.json" \
    --plan "$results/n5_cohort_plan_v2.json" \
    --manifest "$results/n5_model_weights_v1.sha256" \
    --prereg "$results/n5_selective_hybrid_preregistration_v2.md" \
    --out "$results/n5_gate_v2.json" \
    >"$log" 2>&1; then
    printf '0\n' >"$status"
else
    exit_code=$?
    printf '%s\n' "$exit_code" >"$status"
fi
