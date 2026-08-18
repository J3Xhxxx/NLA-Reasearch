#!/bin/bash
# N3 preflight 2: are the candidate real corpora reachable as plain parquet on
# the mirror? We deliberately avoid `datasets` (not installed) and use
# hf_hub_download on explicit paths, the same pattern that worked for the SAE.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf/hub
PY=/root/miniconda3/bin/python

$PY - <<'EOF'
from huggingface_hub import HfApi

api = HfApi()
CANDIDATES = [
    "NeelNanda/pile-10k",          # diverse English incl. code/papers/forums, has meta.pile_set_name
    "openlanguagedata/flores_plus",# parallel multilingual (content held constant)
    "facebook/flores",
    "bigcode/the-stack-smol",      # code fallback
    "stas/openwebtext-10k",        # web fallback
]
for repo in CANDIDATES:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        print(f"\n### {repo}: UNREACHABLE {type(e).__name__}: {str(e)[:120]}")
        continue
    data = [f for f in files if f.endswith((".parquet", ".arrow", ".json", ".jsonl", ".py"))]
    print(f"\n### {repo}: {len(files)} files, {len(data)} data-ish")
    for f in sorted(data)[:14]:
        print("   ", f)
    if len(data) > 14:
        print(f"    ... (+{len(data)-14} more)")
EOF
