#!/usr/bin/env python3
"""Merge the NLA and SAE result files into one head-to-head comparison.

The fair, apples-to-apples axis is DIRECTION fidelity (cos / mse_nrm=2(1-cos)),
because the NLA round-trip is direction-only by construction. The SAE also gets
credit for its native strengths (FVE, L0) which the NLA simply doesn't produce,
and the NLA gets its unique output — a natural-language explanation per vector.

Run either or both lines first, then:
    python 05_compare.py \
        --nla /root/autodl-tmp/results/nla_results.json \
        --sae /root/autodl-tmp/results/sae_results.json \
        --out /root/autodl-tmp/results/comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else None


def key(r):
    return (r["doc_id"], r["position"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nla", default=None)
    ap.add_argument("--sae", default=None)
    ap.add_argument("--out", default="/root/autodl-tmp/results/comparison")
    args = ap.parse_args()

    nla, sae = load(args.nla), load(args.sae)
    if not nla and not sae:
        raise SystemExit("nothing to compare: pass --nla and/or --sae (files must exist)")

    nmap = {key(r): r for r in nla["rows"]} if nla else {}
    smap = {key(r): r for r in sae["rows"]} if sae else {}
    keys = sorted(set(nmap) | set(smap))

    merged = []
    for k in keys:
        n, s = nmap.get(k), smap.get(k)
        merged.append({
            "doc_id": k[0], "position": k[1],
            "token": (n or s).get("token"),
            "raw_norm": (n or s).get("raw_norm"),
            "nla_cos": n["nla_cos"] if n else None,
            "nla_mse_nrm": n["nla_mse_nrm"] if n else None,
            "sae_cos": s["sae_cos"] if s else None,
            "sae_mse_nrm": s["sae_mse_nrm"] if s else None,
            "sae_l0": s["sae_l0"] if s else None,
            "explanation": n["explanation"] if n else None,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"nla_summary": nla["summary"] if nla else None,
                    "sae_summary": sae["summary"] if sae else None,
                    "rows": merged}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown summary + per-token table.
    lines = ["# NLA vs SAE — Gemma-3-12B-IT, layer 32 (resid_post)\n"]
    if nla:
        lines.append(f"- **NLA**  n={nla['summary']['n']}  mean cos="
                     f"{nla['summary']['mean_cos']}  mean mse_nrm={nla['summary']['mean_mse_nrm']}")
    if sae:
        s = sae["summary"]
        lines.append(f"- **SAE**  n={s['n']}  mean cos={s['mean_cos']}  "
                     f"mean mse_nrm={s['mean_mse_nrm']}  FVE={s['fve']}  "
                     f"mean L0={s['mean_l0']}  width={s['width']}")
    lines.append("\n_Direction fidelity (cos, mse_nrm=2(1-cos)) is the fair head-to-head; "
                 "FVE/L0 are SAE-only; the explanation column is NLA-only._\n")
    lines.append("| doc | pos | token | NLA cos | SAE cos | NLA mse_nrm | SAE mse_nrm | SAE L0 |")
    lines.append("|----:|----:|:------|--------:|--------:|------------:|------------:|-------:|")
    for r in merged:
        lines.append(f"| {r['doc_id']} | {r['position']} | `{(r['token'] or '').strip()[:12]}` "
                     f"| {r['nla_cos']} | {r['sae_cos']} | {r['nla_mse_nrm']} "
                     f"| {r['sae_mse_nrm']} | {r['sae_l0']} |")
    if nla:
        lines.append("\n## Sample NLA explanations\n")
        for r in merged[:8]:
            if r["explanation"]:
                lines.append(f"- doc{r['doc_id']} pos{r['position']} `{(r['token'] or '').strip()}` "
                             f"(cos {r['nla_cos']}): {r['explanation']}")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:8]))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
