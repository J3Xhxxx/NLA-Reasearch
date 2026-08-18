#!/bin/bash
# flores_plus turned out to be a GATED repo (403 on the mirror). Find an ungated
# parallel multilingual corpus: we need content held constant across languages,
# otherwise "language feature" claims stay confounded with topic (the exact gap
# in B6+B4, where language features "passed" at 88.9% with saturated AUC).
# Probe by actually downloading one small file, not just listing.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf/hub
PY=/root/miniconda3/bin/python

$PY - <<'EOF'
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
CANDS = [
    "facebook/xnli",                 # genuinely parallel: same premises in 15 langs
    "Muennighoff/flores200",
    "gsarti/flores_101",
    "Helsinki-NLP/opus-100",
    "sentence-transformers/parallel-sentences-tatoeba",
    "wikimedia/wikipedia",           # not parallel, real-text fallback
]
for repo in CANDS:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        print(f"\n### {repo}: LIST FAILED {type(e).__name__}: {str(e)[:100]}")
        continue
    data = [f for f in files if f.endswith((".parquet", ".jsonl", ".json", ".tsv", ".gz"))]
    print(f"\n### {repo}: {len(files)} files, {len(data)} data")
    for f in sorted(data)[:10]:
        print("   ", f)
    if not data:
        continue
    # smallest-looking candidate first, to test gating cheaply
    probe = sorted(data, key=len)[0]
    try:
        p = hf_hub_download(repo, probe, repo_type="dataset")
        import os
        print(f"    DOWNLOAD OK  {probe}  ({os.path.getsize(p)/1e6:.1f} MB)")
    except Exception as e:
        print(f"    DOWNLOAD FAILED {type(e).__name__}: {str(e)[:140]}")
EOF
