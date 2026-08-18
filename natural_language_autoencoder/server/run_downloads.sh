#!/bin/bash
# Resume/download the gated base model via hf-mirror WITH token.
# (hf-mirror serves gated repos when a valid HF token is attached — the earlier
#  403 was an unauthenticated probe. Mirror is ~20MB/s vs <1MB/s via turbo.)
# av/ar/sae are already complete; base resumes from partial files.
cd /root/autodl-tmp/nla_compare
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=$(cat /root/.hf_token)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

for i in 1 2 3 4 5; do
  /root/miniconda3/bin/python -u - <<'PYEOF' && break
import os
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-3-12b-it",
                  local_dir="/root/autodl-tmp/models/gemma-3-12b-it",
                  token=os.environ["HF_TOKEN"], max_workers=8)
print("[base] done")
PYEOF
  echo "[retry $i] base failed, sleeping 30s"; sleep 30
done
echo "=== ALL DOWNLOADS FINISHED ==="
df -h /root/autodl-tmp | tail -1
du -sh /root/autodl-tmp/models/*
