#!/usr/bin/env python3
"""Extract balanced factorial-corpus activations for B6+B4.

Input is JSONL with ``id``, ``axis_domain``, ``axis_language``, ``split``,
``topic``, and ``text``.  All eligible positions are retained by default,
and no short-prompt fallback is allowed: NLA's minimum-context invariant is
part of the experiment rather than silently relaxed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXTRACTION_METADATA_PREFIX = "nla.activation_extraction."


def attach_extraction_metadata(
    table: pa.Table,
    *,
    layer_index: int,
    min_position: int,
    max_per_prompt: int,
    dtype: str,
) -> pa.Table:
    """Bind the extractor settings to the Parquet schema itself."""
    metadata = dict(table.schema.metadata or {})
    values = {
        "schema_version": "1",
        "layer_index": str(layer_index),
        "min_position": str(min_position),
        "max_per_prompt": str(max_per_prompt),
        "dtype": dtype,
    }
    metadata.update(
        {
            f"{EXTRACTION_METADATA_PREFIX}{key}".encode("ascii"): value.encode(
                "ascii"
            )
            for key, value in values.items()
        }
    )
    return table.replace_schema_metadata(metadata)


def resolve_layers(model: torch.nn.Module):
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
    ):
        obj = model
        for attribute in path:
            obj = getattr(obj, attribute, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


@torch.inference_mode()
def capture(model, input_ids, layer_index):
    layers = resolve_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer {layer_index} outside model with {len(layers)} layers")
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().float().cpu()

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
    finally:
        handle.remove()
    return captured["hidden"][0]


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {
            "id",
            "axis_domain",
            "axis_language",
            "split",
            "topic",
            "text",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest line {line_number} missing {sorted(missing)}")
        if row["split"] not in {"train", "test"}:
            raise ValueError(f"invalid split on line {line_number}: {row['split']}")
        rows.append(row)
    if not rows:
        raise ValueError("empty prompt manifest")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt ids must be unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--layer-index", type=int, default=32)
    parser.add_argument("--min-position", type=int, default=50)
    parser.add_argument(
        "--max-per-prompt",
        type=int,
        default=0,
        help="0 keeps every position at or after min-position.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prompts = load_manifest(args.manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device

    vectors = []
    output = {
        "token": [],
        "token_id": [],
        "position": [],
        "doc_id": [],
        "prompt_id": [],
        "axis_domain": [],
        "axis_language": [],
        "split": [],
        "topic": [],
        "prompt": [],
        "prompt_sha256": [],
    }

    for doc_id, row in enumerate(prompts):
        tokenized = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not torch.is_tensor(tokenized):
            tokenized = tokenized["input_ids"]
        input_ids = tokenized.to(device)
        hidden = capture(model, input_ids, args.layer_index)
        sequence = input_ids[0].tolist()
        candidates = list(range(args.min_position, len(sequence)))
        if not candidates:
            raise ValueError(
                f"{row['id']} has only {len(sequence)} tokens; "
                f"requires > min_position={args.min_position}"
            )
        if args.max_per_prompt > 0 and len(candidates) > args.max_per_prompt:
            sampled = np.linspace(
                0, len(candidates) - 1, args.max_per_prompt
            ).round().astype(int)
            candidates = [candidates[index] for index in sorted(set(sampled))]

        prompt_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        for position in candidates:
            vectors.append(hidden[position].numpy().astype(np.float32))
            output["token"].append(
                tokenizer.decode([sequence[position]], skip_special_tokens=False)
            )
            output["token_id"].append(int(sequence[position]))
            output["position"].append(int(position))
            output["doc_id"].append(int(doc_id))
            output["prompt_id"].append(row["id"])
            output["axis_domain"].append(row["axis_domain"])
            output["axis_language"].append(row["axis_language"])
            output["split"].append(row["split"])
            output["topic"].append(row["topic"])
            output["prompt"].append(row["text"])
            output["prompt_sha256"].append(prompt_hash)
        print(
            f"[{doc_id:02d} {row['id']}] seq={len(sequence)} "
            f"kept={len(candidates)}"
        )

    array = np.stack(vectors)
    table = pa.table(
        {
            "activation_vector": pa.array(
                list(array), type=pa.list_(pa.float32())
            ),
            **output,
            "norm": ["none"] * len(array),
        }
    )
    table = attach_extraction_metadata(
        table,
        layer_index=args.layer_index,
        min_position=args.min_position,
        max_per_prompt=args.max_per_prompt,
        dtype=args.dtype,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    norms = np.linalg.norm(array, axis=1)
    print("FACTORIAL_EXTRACTION_COMPLETE")
    print(
        f"wrote {len(array)} rows d={array.shape[1]} "
        f"mean_norm={norms.mean():.1f} range=[{norms.min():.1f},{norms.max():.1f}] "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
