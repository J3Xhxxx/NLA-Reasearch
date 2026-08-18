#!/usr/bin/env python3
"""Readable dump of the N3 per-feature table plus real max-activating contexts.

Prints, for each of the 24 B6+B4 semantic directions: its readability score q+,
its synthetic held-out AUC (and whether that AUC was the tie-convention 0.5 that
F14.2 flagged), and what the feature actually fires on in 8.2M tokens of real
text. Intended for eyeballing whether the original labels survive real evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to GBK
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--contexts", type=int, default=3)
    ap.add_argument("--only-top", type=int, default=0,
                    help="if >0, show contexts only for the N highest-q+ features")
    args = ap.parse_args()
    R = Path(args.results)

    a = json.load(open(R / "n3_analysis.json", encoding="utf-8"))
    ctx = json.load(open(R / "n3_contexts_v1.json", encoding="utf-8"))
    by_feat = {d.get("feature"): d for d in ctx["directions"]}

    rows = a["per_feature"]
    hdr = (f"{'feat':>5} {'label':<20} {'q+':>7} {'synAUC':>6} {'dead':>4} "
           f"{'real_freq':>10} {'nsrc':>4} {'top_source':<20} {'shr':>5} {'lpur':>5} lang")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        auc = r["synthetic_test_auroc"]
        print(f"{r['feature']:>5} {str(r['label'])[:20]:<20} {r['q_plus']:+7.3f} "
              f"{(f'{auc:.2f}' if auc is not None else '-'):>6} "
              f"{('DEAD' if r['synthetic_test_dead'] else ''):>4} "
              f"{r['real_freq']:10.2e} {r['real_n_sources']:>4} "
              f"{str(r['real_top_source'])[:20]:<20} {r['real_top_source_share']:5.2f} "
              f"{r['real_lang_purity']:5.2f} {r['real_lang_top']}")

    q1 = a["q1_was_the_gate_measuring_the_corpus"]
    print(f"\nsynthetic-test-dead: {q1['n_synthetic_test_dead']}/{q1['n_features']}; "
          f"alive on real text: {q1['of_those_alive_on_real_text']}; "
          f"dead on real text: {q1['n_dead_on_real_text']}")

    show = rows if args.only_top <= 0 else rows[:args.only_top]
    print("\n=== real max-activating contexts ===")
    for r in show:
        d = by_feat.get(r["feature"], {})
        print(f"\n--- f{r['feature']}  label={r['label']}  q+={r['q_plus']:+.3f}  "
              f"freq={r['real_freq']:.2e}  top_source={r['real_top_source']} ---")
        for c in (d.get("real_top_contexts") or [])[:args.contexts]:
            before = c["before"].replace("\n", " ")[-90:]
            after = c["after"].replace("\n", " ")[:30]
            print(f"  [{c['activation']:8.0f}] {c['source']:<18} "
                  f"...{before}>>{c['token']}<<{after}...")


if __name__ == "__main__":
    main()
