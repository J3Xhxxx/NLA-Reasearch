#!/bin/bash
# N3 runner: build real corpus -> 16k feature statistics -> resampled cohort.
# No shutdown path (user keeps the box up). Resumable: 34 checkpoints every
# --ckpt-every batches, so a dropped session costs minutes, not the run.
exec > /root/autodl-tmp/n3.log 2>&1
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

echo "N3_START $(date -u)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# --- step 1: corpus (cheap, no GPU; skip if already frozen) ---
if [ ! -f "$RES/n3_corpus_v1.json" ]; then
  $PY 33_n3_build_corpus.py --out-dir "$RES" --tag v1
  echo "N3_CORPUS_EXIT status=$? $(date -u)"
else
  echo "N3_CORPUS_SKIP already frozen"
fi

# --- step 2: feature statistics over the whole corpus ---
$PY 34_n3_feature_stats.py \
    --corpus "$RES/n3_corpus_v1.jsonl" \
    --base-model "$MODELS/gemma-3-12b-it" \
    --sae-root "$MODELS/gemma-scope-2-12b-it/resid_post_all" \
    --batch-size 16 --seq-len 512 --topk 16 --ckpt-every 40 \
    --max-pile-tokens 8000000 \
    --out-prefix "$RES/n3_feature_stats_v1"
S2=$?
echo "N3_STATS_EXIT status=$S2 $(date -u)"

# --- step 3: clean cohort (fixes F13) ---
$PY 35_n3_resample_cohort.py \
    --corpus "$RES/n3_corpus_v1.jsonl" \
    --base-model "$MODELS/gemma-3-12b-it" \
    --n-target 200 --min-position 64 --min-continuation 16 \
    --out /root/autodl-tmp/activations/acts_L32_n3_v1.parquet \
    --out-json "$RES/n3_cohort_v1.json"
S3=$?
echo "N3_COHORT_EXIT status=$S3 $(date -u)"

echo "N3_COMPLETE $(date -u) stats=$S2 cohort=$S3"
