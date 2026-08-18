#!/usr/bin/env bash
set -uo pipefail

root=/root/autodl-tmp
code=$root/nla_compare
results=$root/results
activations=$root/activations
log=$results/n5_discovery_reconstruct_v2.log
status=$results/n5_discovery_reconstruct_v2.exit

rm -f "$status"
cd "$code" || exit 97
if /root/miniconda3/bin/python -B 44_n5_reconstruct.py \
    --av "$root/models/nla-gemma3-12b-L32-av" \
    --ar "$root/models/nla-gemma3-12b-L32-ar" \
    --sae-small "$root/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small" \
    --sae-big "$root/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_big" \
    --activations "$activations/acts_L32_n5_discovery_v2.parquet" \
    --plan "$results/n5_cohort_plan_v2.json" \
    --model-manifest "$results/n5_model_weights_v1.sha256" \
    --prereg "$results/n5_selective_hybrid_preregistration_v2.md" \
    --split discovery \
    --checkpoint "$results/n5_discovery_av_checkpoint_v2.jsonl" \
    --explanations-out "$results/n5_discovery_explanations_v2.json" \
    --variants-out "$results/n5_discovery_variants_v2.json" \
    --out "$results/n5_discovery_recon_v2.json" \
    --vecs-out "$results/n5_discovery_recon_vectors_v2.npz" \
    >"$log" 2>&1; then
    printf '0\n' >"$status"
else
    exit_code=$?
    printf '%s\n' "$exit_code" >"$status"
fi
