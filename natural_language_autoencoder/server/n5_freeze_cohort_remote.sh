#!/usr/bin/env bash
set -uo pipefail

root=/root/autodl-tmp
code=$root/nla_compare
results=$root/results
log=$results/n5_v2_cohort_freeze.log
status=$results/n5_v2_cohort_freeze.exit

rm -f "$status"
cd "$code" || exit 97
if /root/miniconda3/bin/python -B 42_n5_freeze_cohort.py \
    --pile-parquet "$root/hf/hub/datasets--NeelNanda--pile-10k/blobs/a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31" \
    --xnli-parquet "$root/hf/hub/datasets--facebook--xnli/blobs/c5e6263b0872a3914c9bc165bfe3883e433aa2066c3fa3b9d142829a9b122518" \
    --prereg "$results/n5_selective_hybrid_preregistration_v2.md" \
    --out "$results/n5_cohort_plan_v2.json" \
    >"$log" 2>&1; then
    printf '0\n' >"$status"
else
    code=$?
    printf '%s\n' "$code" >"$status"
fi
