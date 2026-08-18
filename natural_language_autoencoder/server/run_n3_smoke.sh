#!/bin/bash
# N3 smoke test: tiny corpus, tiny token budget. Validates the early-exit hook,
# both SAE encodes, the top-K bookkeeping, the checkpoint/resume path and the
# cohort self-checks BEFORE spending an hour of A800 on the full run.
exec > /root/autodl-tmp/n3_smoke.log 2>&1
set -x

PY=/root/miniconda3/bin/python
CMP=/root/autodl-tmp/nla_compare
RES=/root/autodl-tmp/results
MODELS=/root/autodl-tmp/models

cd "$CMP"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf/hub
export PYTHONUNBUFFERED=1
export NLA_REPO=/root/autodl-tmp/nla_repo

echo "SMOKE_START $(date -u)"

$PY 33_n3_build_corpus.py --pile-docs 40 --xnli-max-passages 3 \
    --out-dir "$RES" --tag smoke
echo "SMOKE_CORPUS_EXIT status=$?"

$PY 34_n3_feature_stats.py \
    --corpus "$RES/n3_corpus_smoke.jsonl" \
    --base-model "$MODELS/gemma-3-12b-it" \
    --sae-root "$MODELS/gemma-scope-2-12b-it/resid_post_all" \
    --batch-size 4 --seq-len 256 --topk 8 --ckpt-every 2 --max-tokens 12000 \
    --out-prefix "$RES/n3_feature_stats_smoke"
echo "SMOKE_STATS_EXIT status=$?"

$PY 35_n3_resample_cohort.py \
    --corpus "$RES/n3_corpus_smoke.jsonl" \
    --base-model "$MODELS/gemma-3-12b-it" \
    --n-target 8 --per-doc 2 --min-position 64 --min-continuation 16 \
    --out /root/autodl-tmp/activations/acts_L32_n3_smoke.parquet \
    --out-json "$RES/n3_cohort_smoke.json"
echo "SMOKE_COHORT_EXIT status=$?"

echo "SMOKE_COMPLETE $(date -u)"
