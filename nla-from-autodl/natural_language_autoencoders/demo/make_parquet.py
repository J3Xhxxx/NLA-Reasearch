"""Extract layer-K residual-stream activations from a base model and write a
parquet that nla_inference.py / roundtrip.py can consume.

Replicates nla/datagen/extractors.py::HFExtractor exactly:
  layer_index=K  ->  output of decoder block K  ==  HF hidden_states[K+1]
(The repo top-level README quick-start uses hidden_states[20]; that is off by
one. The authoritative datagen convention is hidden_states[K+1], i.e. [21] for
Qwen layer 20. We follow the datagen convention here.)

Output columns:
  activation_vector : list[float]  (length d_model) -- the only column the
                      NLA inference code requires.
  token_id, token_str, position, raw_norm : diagnostics for readability.

Usage:
  python demo/make_parquet.py \
      --base-model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --layer-index 20 \
      --text "The Eiffel Tower, completed in 1889, stands on the Champ de Mars in Paris." \
      --chat \
      --out /root/autodl-tmp/demo.parquet
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True,
                    help="HF dir / id of the base model the NLA was trained on "
                         "(Qwen/Qwen2.5-7B-Instruct).")
    ap.add_argument("--layer-index", type=int, default=20,
                    help="Decoder block index K (Qwen NLA uses 20).")
    ap.add_argument("--text", required=True,
                    help="The text whose per-token activations you want to verbalize.")
    ap.add_argument("--chat", action="store_true",
                    help="Wrap --text as a user turn via the chat template "
                         "(matches the worked example). Omit for a raw corpus string.")
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=args.device,
        trust_remote_code=True,
    ).eval()

    if args.chat:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.text}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
    else:
        ids = tok(args.text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    ids = ids.to(model.device)

    layers = model.model.layers
    assert 0 <= args.layer_index < len(layers), (
        f"layer_index={args.layer_index} out of range (model has {len(layers)} layers)"
    )

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().clone()

    handle = layers[args.layer_index].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()

    assert "h" in captured, "forward hook did not fire"
    hs = captured["h"][0].float().cpu()  # [seq, d_model]
    token_ids = ids[0].cpu().tolist()
    token_strs = [tok.decode([t]) for t in token_ids]
    norms = hs.norm(dim=-1).tolist()

    table = pa.table({
        "activation_vector": [row.tolist() for row in hs],
        "token_id": token_ids,
        "token_str": token_strs,
        "position": list(range(len(token_ids))),
        "raw_norm": norms,
    })
    pq.write_table(table, args.out)
    print(f"[make_parquet] wrote {len(token_ids)} rows  d_model={hs.shape[1]}  "
          f"layer_index={args.layer_index} (= hidden_states[{args.layer_index + 1}])  "
          f"-> {args.out}")
    print(f"[make_parquet] tokens: {token_strs}")


if __name__ == "__main__":
    main()
