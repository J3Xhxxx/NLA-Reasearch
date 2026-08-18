#!/usr/bin/env bash
echo "=== pip config (domestic mirror?) ==="
pip config list 2>/dev/null; cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null | grep -iE "index-url|host" | head

echo; echo "=== gemma base gating bypass test (mirror, no token) ==="
code=$(curl -s -o /tmp/gemma_cfg.json -w "%{http_code}" "https://hf-mirror.com/google/gemma-3-12b-it/resolve/main/config.json")
echo "gemma-3-12b-it config.json -> HTTP $code  bytes=$(wc -c </tmp/gemma_cfg.json 2>/dev/null)"
head -c 200 /tmp/gemma_cfg.json; echo

echo; echo "=== SAE config.json (resid_post layer32 width16k l0_small) ==="
curl -s "https://hf-mirror.com/google/gemma-scope-2-12b-it/resolve/main/resid_post_all/layer_32_width_16k_l0_small/config.json"
echo

echo "=== SAE safetensors tensor keys ==="
source /etc/network_turbo 2>/dev/null
python - <<'PY'
import urllib.request, json, struct
url="https://hf-mirror.com/google/gemma-scope-2-12b-it/resolve/main/resid_post_all/layer_32_width_16k_l0_small/params.safetensors"
req=urllib.request.Request(url, headers={"User-Agent":"python-requests/2.31","Range":"bytes=0-8"})
n=struct.unpack("<Q", urllib.request.urlopen(req,timeout=30).read(8))[0]
req=urllib.request.Request(url, headers={"User-Agent":"python-requests/2.31","Range":f"bytes=8-{8+n-1}"})
hdr=json.loads(urllib.request.urlopen(req,timeout=30).read(n).decode())
for k,v in hdr.items():
    if k!="__metadata__": print(f"   {k:12s} {v['dtype']:8s} {v['shape']}")
PY
