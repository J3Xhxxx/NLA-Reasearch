#!/bin/bash
# Autonomous NLA-vs-SAE pipeline. Waits for downloads, runs extraction ->
# NLA -> SAE(x2) -> compare, records status, and exits. Current keep-alive
# policy forbids automatic shutdown. All output -> /root/autodl-tmp/pipeline.log.
exec > /root/autodl-tmp/pipeline.log 2>&1
set -x
PY=/root/miniconda3/bin/python
CMP=/root/autodl-tmp/nla_compare
MODELS=/root/autodl-tmp/models
ACTS=/root/autodl-tmp/activations/acts_L32.parquet
RES=/root/autodl-tmp/results
cd "$CMP"
export HF_HOME=/root/autodl-tmp/hf

finish() {
  echo "=== PIPELINE EXIT (status=$1) at $(date) ==="
  sync
  exit "$1"
}

# 1. wait for the download script to finish (base model)
while pgrep -f '[r]un_downloads' >/dev/null; do sleep 60; done

# verify base is complete: every shard in the index must exist
$PY - <<'EOF' || { echo 'BASE INCOMPLETE - exiting; complete downloads before retry'; finish 1; }
import json, sys
from pathlib import Path
d = Path('/root/autodl-tmp/models/gemma-3-12b-it')
idx = d / 'model.safetensors.index.json'
if not idx.exists():
    sys.exit('no safetensors index')
shards = set(json.loads(idx.read_text())['weight_map'].values())
missing = [s for s in shards if not (d / s).exists()]
sys.exit(f'missing shards: {missing}' if missing else 0)
EOF

mkdir -p "$RES" "$(dirname "$ACTS")"

# 2. extract layer-32 activations from base (shared by both lines)
$PY 02_extract_activations.py \
    --base-model "$MODELS/gemma-3-12b-it" \
    --out "$ACTS" || finish 2

# 3. NLA line: vector -> AV text -> AR reconstruction
$PY 03_run_nla.py \
    --av  "$MODELS/nla-gemma3-12b-L32-av" \
    --ar  "$MODELS/nla-gemma3-12b-L32-ar" \
    --activations "$ACTS" \
    --out "$RES/nla_results.json" || finish 3

# 4. SAE line, both L0 variants
$PY 04_run_sae.py \
    --sae "$MODELS/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small" \
    --activations "$ACTS" \
    --out "$RES/sae_results.json" || finish 4
$PY 04_run_sae.py \
    --sae "$MODELS/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_big" \
    --activations "$ACTS" \
    --out "$RES/sae_results_big.json"   # big variant is bonus; no abort on fail

# 5. merge -> comparison.json + comparison.md
$PY 05_compare.py \
    --nla "$RES/nla_results.json" \
    --sae "$RES/sae_results.json" \
    --out "$RES/comparison" || finish 5

echo "=== PIPELINE_COMPLETE ==="
ls -la "$RES"
finish 0
