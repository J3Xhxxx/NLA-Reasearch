#!/usr/bin/env bash
set -uo pipefail

root=/root/autodl-tmp
code=$root/nla_compare
results=$root/results
activations=$root/activations
log=$results/n5_heldout_causal_v2.log
status=$results/n5_heldout_causal_v2.exit

rm -f "$status"
cd "$code" || exit 97
if /root/miniconda3/bin/python -B 45_n5_causal_patch.py \
    --base-model "$root/models/gemma-3-12b-it" \
    --activations "$activations/acts_L32_n5_heldout_v2.parquet" \
    --recon "$results/n5_heldout_recon_vectors_v2.npz" \
    --plan "$results/n5_cohort_plan_v2.json" \
    --model-manifest "$results/n5_model_weights_v1.sha256" \
    --prereg "$results/n5_selective_hybrid_preregistration_v2.md" \
    --gate "$results/n5_gate_v2.json" \
    --split heldout \
    --checkpoint "$results/n5_heldout_causal_checkpoint_v2.jsonl" \
    --out "$results/n5_heldout_causal_v2.json" \
    --layer-index 32 \
    --horizon 16 \
    --dtype bfloat16 \
    --identity-kl-tol 1e-5 \
    >"$log" 2>&1; then
    printf '0\n' >"$status"
else
    exit_code=$?
    printf '%s\n' "$exit_code" >"$status"
fi
