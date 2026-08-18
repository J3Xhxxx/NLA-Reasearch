#!/usr/bin/env python3
import json, urllib.request, re
def get(mid):
    u = f"https://hf-mirror.com/api/models/{mid}"
    req = urllib.request.Request(u, headers={"User-Agent": "python-requests/2.31"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        print(f"### {mid}  ERROR {e}")
        return []
    sib = [s["rfilename"] for s in d.get("siblings", [])]
    print(f"### {mid}  gated={d.get('gated')}  files={len(sib)}")
    return sib

for m in ["kitft/nla-gemma3-12b-L32-av", "kitft/nla-gemma3-12b-L32-ar"]:
    s = get(m)
    for f in s:
        if any(k in f.lower() for k in ("meta", "config.json", "value_head", ".yaml")):
            print("   ", f)

s = get("google/gemma-scope-2-12b-it")
print("  total files:", len(s))
layers = sorted({int(m.group(1)) for f in s for m in [re.search(r"layer[_/]?(\d+)", f)] if m})
print("  layers seen (first/last):", layers[:5], "...", layers[-3:])
print("  sample paths:")
for f in s[:10]:
    print("   ", f)
print("  layer-32 matches:")
n = 0
for f in s:
    if re.search(r"(layer[_/]?32(?!\d)|/32/|_32_|l32)", f):
        print("   ", f); n += 1
        if n > 30: print("    ..."); break
