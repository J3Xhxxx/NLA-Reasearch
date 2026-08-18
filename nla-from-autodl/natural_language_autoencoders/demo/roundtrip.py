"""Full NLA round-trip demo: AV (vector -> text) + AR (text -> vector, scored).

For each activation vector in a parquet:
  1. AV  (activation verbalizer)  : ask the SGLang-served AV model to describe it.
  2. AR  (activation reconstructor): reconstruct a vector from that description,
     score it against the original  ->  (mse_nrm, cos).

Reuses NLAClient + NLACritic from the repo's standalone nla_inference.py.

Prereqs:
  - The AV SGLang server must already be running (see README), e.g.:
      python -m sglang.launch_server --model-path <AV_DIR> \
          --port 30000 --disable-radix-cache --mem-fraction-static 0.85 --trust-remote-code
  - A parquet with an `activation_vector` column (see demo/make_parquet.py).

Usage:
  python demo/roundtrip.py \
      --av  /root/autodl-tmp/models/nla-qwen-av \
      --ar  /root/autodl-tmp/models/nla-qwen-ar \
      --parquet /root/autodl-tmp/demo.parquet \
      --sglang-url http://localhost:30000 \
      --skip-first 10 --temperature 0 --max-new-tokens 200 \
      --out /root/autodl-tmp/demo_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Make `import nla_inference` work no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nla_inference import NLAClient, NLACritic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av", required=True, help="AV (activation verbalizer) checkpoint dir.")
    ap.add_argument("--ar", required=True, help="AR (activation reconstructor) checkpoint dir.")
    ap.add_argument("--parquet", required=True, help="Parquet with activation_vector column.")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--device", default="cuda", help="Device for the AR model.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (reproducible).")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--skip-first", type=int, default=10,
                    help="Skip the first N positions (early-sequence decodes are noisy).")
    ap.add_argument("--n", type=int, default=0,
                    help="Process at most N rows after skipping (0 = all).")
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    # ---- read parquet ----------------------------------------------------
    pf = pq.read_table(args.parquet)
    cols = pf.column_names
    flat = pf.column("activation_vector").to_pylist()
    vecs = [np.asarray(v, dtype=np.float32) for v in flat]
    token_strs = pf.column("token_str").to_pylist() if "token_str" in cols else [""] * len(vecs)
    raw_norms = pf.column("raw_norm").to_pylist() if "raw_norm" in cols else [float(np.linalg.norm(v)) for v in vecs]

    idxs = list(range(args.skip_first, len(vecs)))
    if args.n > 0:
        idxs = idxs[: args.n]

    # ---- load models -----------------------------------------------------
    client = NLAClient(args.av, sglang_url=args.sglang_url)
    critic = NLACritic(args.ar, device=args.device)

    results = []
    print("\n" + "=" * 90)
    print(f"NLA round-trip — {len(idxs)} positions  (temp={args.temperature})")
    print("=" * 90)

    for i in idxs:
        v = vecs[i]
        explanation = client.generate(
            v, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        )
        mse, cos = critic.score(explanation, v)
        results.append({
            "position": i,
            "token": token_strs[i],
            "raw_norm": round(float(raw_norms[i]), 1),
            "mse_nrm": round(mse, 3),
            "cos": round(cos, 3),
            "explanation": explanation,
        })
        print(f"\n[{i:>3}] token={token_strs[i]!r}  ||v||={raw_norms[i]:.1f}  "
              f"mse_nrm={mse:.3f}  cos={cos:.3f}")
        print("    " + explanation.replace("\n", "\n    "))

    if results:
        coss = [r["cos"] for r in results]
        mses = [r["mse_nrm"] for r in results]
        print("\n" + "=" * 90)
        print(f"SUMMARY  n={len(results)}  mean cos={np.mean(coss):.3f}  "
              f"mean mse_nrm={np.mean(mses):.3f}  "
              f"(cos>=0.7 ~ good, ~0.0 ~ orthogonal/random)")
        print("=" * 90)

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n[roundtrip] wrote {len(results)} results -> {args.out}")


if __name__ == "__main__":
    main()
