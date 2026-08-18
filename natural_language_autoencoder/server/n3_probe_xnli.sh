#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf/hub
PY=/root/miniconda3/bin/python

$PY - <<'EOF'
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

p = hf_hub_download("facebook/xnli", "all_languages/validation-00000-of-00001.parquet",
                    repo_type="dataset")
t = pq.read_table(p)
print("rows:", t.num_rows)
print("schema:", t.schema)
row = t.slice(0, 1).to_pylist()[0]
for k, v in row.items():
    s = repr(v)
    print(f"\n--- {k} ---\n{s[:600]}")
EOF
