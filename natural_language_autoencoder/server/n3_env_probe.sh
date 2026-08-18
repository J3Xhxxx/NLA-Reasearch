#!/bin/bash
# N3 preflight: what is already installed / configured on the box.
PY=/root/miniconda3/bin/python
echo "=== versions ==="
$PY - <<'EOF'
import importlib
for m in ("torch", "transformers", "pyarrow", "safetensors", "datasets", "huggingface_hub"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:18s} MISSING ({type(e).__name__})")
EOF

echo "=== hf env (from prior runners) ==="
grep -hiE 'HF_ENDPOINT|HF_HOME|HF_TOKEN|HUGGING' /root/.bashrc /root/autodl-tmp/nla_compare/*.sh 2>/dev/null | sort -u

echo "=== models ==="
ls -1 /root/autodl-tmp/models/
echo "--- sae dirs ---"
find /root/autodl-tmp/models -maxdepth 3 -name 'params.safetensors' | sed 's|/root/autodl-tmp/models/||'

echo "=== disk ==="
df -h /root/autodl-tmp / | tail -3

echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "=== hf cache datasets already present ==="
ls -1 /root/autodl-tmp/hf/datasets 2>/dev/null | head -20 || echo none

echo "=== mirror reachability ==="
curl -s -o /dev/null -w 'hf-mirror %{http_code} %{time_total}s\n' --max-time 20 https://hf-mirror.com/api/models/google/gemma-3-12b-it
