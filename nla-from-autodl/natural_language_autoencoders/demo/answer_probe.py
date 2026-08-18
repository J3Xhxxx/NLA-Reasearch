"""Generate an answer, then ask the AV to verbalize selected token activations.

This is the "model sees a prompt, answers, and NLA explains selected tokens" demo:

  1. Load the base instruct model and generate an answer for --prompt.
  2. Run one full forward pass over prompt + answer and capture layer-K residuals.
  3. For selected prompt/question/answer tokens, send those vectors to the AV.
  4. Optionally score each AV explanation with the AR checkpoint.

Prereqs:
  - The AV SGLang server must already be running, e.g.:
      AV_DIR=/root/autodl-tmp/models/nla-qwen-av MEM_FRACTION=0.6 bash demo/launch_av_server.sh

Example:
  python demo/answer_probe.py \
      --base-model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --av /root/autodl-tmp/models/nla-qwen-av \
      --ar /root/autodl-tmp/models/nla-qwen-ar \
      --prompt "Explain why the Eiffel Tower is famous in three sentences." \
      --layer-index 20 \
      --target all \
      --positions all \
      --quiet \
      --answer-temperature 0 \
      --av-temperature 0 \
      --out /root/autodl-tmp/answer_probe_full_context.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `import nla_inference` work no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nla_inference import NLAClient, NLACritic  # noqa: E402


def parse_positions(spec: str | None, total: int) -> list[int]:
    """Parse comma/range token specs such as "0,2,5-8"; default = all."""
    if spec is None or spec.strip().lower() in {"", "all", "*"}:
        return list(range(total))

    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))

    bad = sorted(i for i in out if i < 0 or i >= total)
    if bad:
        raise ValueError(f"token positions out of range 0..{total - 1}: {bad}")
    return sorted(out)


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int]:
    """Return [start, end) for needle in haystack, preferring the last match."""
    if not needle:
        raise ValueError("empty token subsequence")
    matches = []
    width = len(needle)
    for start in range(0, len(haystack) - width + 1):
        if haystack[start:start + width] == needle:
            matches.append(start)
    if not matches:
        raise ValueError(
            "could not locate the raw user prompt tokens inside the formatted "
            "prompt. Try --no-chat or inspect the tokenizer chat template."
        )
    start = matches[-1]
    return start, start + width


def classify_section(pos: int, *, prompt_len: int, question_span: tuple[int, int]) -> str:
    q_start, q_end = question_span
    if q_start <= pos < q_end:
        return "question"
    if pos < prompt_len:
        return "prompt_template"
    return "answer"


def selected_positions(
    token_ids: list[int],
    *,
    positions: str | None,
    start: int,
    limit: int,
    include_special: bool,
    special_ids: set[int],
) -> list[int]:
    available = [
        i for i, tid in enumerate(token_ids)
        if include_special or tid not in special_ids
    ]
    selected = [i for i in parse_positions(positions, len(token_ids)) if i in available]
    selected = [i for i in selected if i >= start]
    if limit > 0:
        selected = selected[:limit]
    return selected


def resolve_decoder_layers(model: torch.nn.Module):
    """Return the transformer block list for common HF causal-LM layouts."""
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        f"could not find decoder layers on {type(model).__name__}; "
        "add this architecture to resolve_decoder_layers()."
    )


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def build_prompt_ids_and_question_span(
    tokenizer,
    prompt: str,
    *,
    system: str | None,
    chat: bool,
) -> tuple[torch.Tensor, tuple[int, int]]:
    question_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if chat:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_list = ids[0].tolist()
        return ids, find_subsequence(prompt_list, question_ids)

    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    prompt_list = ids[0].tolist()
    try:
        span = find_subsequence(prompt_list, question_ids)
    except ValueError:
        span = (0, len(prompt_list))
    return ids, span


@torch.inference_mode()
def generate_answer(model, tokenizer, input_ids: torch.Tensor, args) -> tuple[list[int], str]:
    gen_kwargs = {
        "max_new_tokens": args.max_answer_tokens,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.answer_temperature and args.answer_temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": args.answer_temperature,
            "top_p": args.top_p,
        })
    else:
        gen_kwargs["do_sample"] = False

    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    seq = generated.sequences[0].detach().cpu().tolist()
    answer_ids = seq[input_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return answer_ids, answer_text


@torch.inference_mode()
def capture_layer_activations(
    model,
    full_ids: torch.Tensor,
    *,
    layer_index: int,
) -> torch.Tensor:
    layers = resolve_decoder_layers(model)
    assert 0 <= layer_index < len(layers), (
        f"layer_index={layer_index} out of range (model has {len(layers)} layers)"
    )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().float().cpu()

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        attention_mask = torch.ones_like(full_ids)
        model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    if "h" not in captured:
        raise RuntimeError("forward hook did not fire")
    return captured["h"][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True,
                    help="HF dir / id for the model that should generate the answer.")
    ap.add_argument("--av", required=True, help="AV checkpoint dir served by SGLang.")
    ap.add_argument("--ar", default=None,
                    help="Optional AR checkpoint dir. If set, score each AV explanation.")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--prompt", required=True, help="User prompt for the base model.")
    ap.add_argument("--system", default=None, help="Optional system message when using chat template.")
    ap.add_argument("--no-chat", action="store_true",
                    help="Do not apply the tokenizer chat template; treat --prompt as raw text.")
    ap.add_argument("--layer-index", type=int, default=20,
                    help="Decoder block index K (Qwen NLA uses 20).")
    ap.add_argument("--target", choices=["answer", "question", "both", "prompt", "all"], default="answer",
                    help="'prompt' is the full formatted prompt; 'all' is formatted prompt + answer.")
    ap.add_argument("--positions", default=None,
                    help="Token positions within each selected target, e.g. '0,3,5-9'. Default: all.")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip this many target tokens before applying --n/--positions.")
    ap.add_argument("--n", type=int, default=0,
                    help="Explain at most N selected target tokens (0 = no cap).")
    ap.add_argument("--include-special", action="store_true",
                    help="Include special tokens. Implied by --target prompt/all.")
    ap.add_argument("--device-map", default="cuda",
                    help="Passed to from_pretrained(device_map=...). Use 'auto' if preferred.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--answer-temperature", type=float, default=0.0,
                    help="Temperature for the base model answer. 0 = greedy.")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-answer-tokens", type=int, default=128)
    ap.add_argument("--av-temperature", type=float, default=0.0,
                    help="Temperature for AV explanation generation. 0 = greedy.")
    ap.add_argument("--av-max-new-tokens", type=int, default=200)
    ap.add_argument("--ar-device", default="cuda", help="Device for the optional AR model.")
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    ap.add_argument("--quiet", action="store_true",
                    help="Do not print the long answer/AV explanations; write them to --out only.")
    args = ap.parse_args()

    if args.quiet and not args.out:
        raise SystemExit("--quiet requires --out so results are not lost.")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    device = first_parameter_device(model)
    prompt_ids, question_span = build_prompt_ids_and_question_span(
        tokenizer,
        args.prompt,
        system=args.system,
        chat=not args.no_chat,
    )
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    answer_ids, answer_text = generate_answer(model, tokenizer, prompt_ids, args)
    if not answer_ids:
        raise RuntimeError("base model generated no answer tokens")

    full_ids = torch.tensor(
        prompt_ids[0].detach().cpu().tolist() + answer_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    hidden = capture_layer_activations(
        model,
        full_ids,
        layer_index=args.layer_index,
    )

    quiet_stream = open(os.devnull, "w", encoding="utf-8") if args.quiet else None
    stdout_cm = contextlib.redirect_stdout(quiet_stream) if quiet_stream else contextlib.nullcontext()
    with stdout_cm:
        client = NLAClient(args.av, sglang_url=args.sglang_url)
        critic = NLACritic(args.ar, device=args.ar_device) if args.ar else None

    if not args.quiet:
        print("\n" + "=" * 100)
        print("BASE MODEL ANSWER")
        print("=" * 100)
        print(answer_text.strip() or tokenizer.decode(answer_ids, skip_special_tokens=False))

    results = []
    special_ids = set(tokenizer.all_special_ids or [])
    full_token_ids = full_ids[0].detach().cpu().tolist()

    targets: list[tuple[str, int, list[int]]] = []
    if args.target in ("question", "both"):
        q_start, q_end = question_span
        targets.append(("question", q_start, full_token_ids[q_start:q_end]))
    if args.target == "prompt":
        targets.append(("prompt", 0, full_token_ids[:prompt_len]))
    if args.target in ("answer", "both"):
        targets.append(("answer", prompt_len, answer_ids))
    if args.target == "all":
        targets.append(("all", 0, full_token_ids))

    for target_name, absolute_start, token_ids in targets:
        token_strs = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
        include_special = args.include_special or target_name in ("prompt", "all")
        selected = selected_positions(
            token_ids,
            positions=args.positions,
            start=args.start,
            limit=args.n,
            include_special=include_special,
            special_ids=special_ids,
        )
        if not selected:
            raise RuntimeError(
                f"no {target_name} tokens selected; adjust --positions/--start/--include-special."
            )

        if not args.quiet:
            print("\n" + "=" * 100)
            print(
                f"AV EXPLANATIONS ({target_name.upper()})  layer={args.layer_index}  "
                f"selected={len(selected)}/{len(token_ids)} tokens"
            )
            print("=" * 100)

        for pos in selected:
            absolute_pos = absolute_start + pos
            vector = hidden[absolute_pos].numpy().astype(np.float32)
            with stdout_cm:
                explanation = client.generate(
                    vector,
                    temperature=args.av_temperature,
                    max_new_tokens=args.av_max_new_tokens,
                )
            row = {
                "target": target_name,
                "section": classify_section(
                    absolute_pos,
                    prompt_len=prompt_len,
                    question_span=question_span,
                ),
                "target_position": pos,
                "absolute_position": absolute_pos,
                "token_id": int(token_ids[pos]),
                "token": token_strs[pos],
                "token_repr": repr(token_strs[pos]),
                "raw_norm": round(float(np.linalg.norm(vector)), 3),
                "explanation": explanation,
            }
            if target_name == "answer":
                row["answer_position"] = pos
            else:
                row["question_position"] = pos

            score = ""
            if critic is not None:
                with stdout_cm:
                    mse, cos = critic.score(explanation, vector)
                row["mse_nrm"] = round(float(mse), 3)
                row["cos"] = round(float(cos), 3)
                score = f"  mse_nrm={mse:.3f}  cos={cos:.3f}"
            results.append(row)

            if not args.quiet:
                print(f"\n[{pos:>3}] token={token_strs[pos]!r}  ||v||={row['raw_norm']:.1f}{score}")
                print("    " + explanation.replace("\n", "\n    "))

    payload = {
        "prompt": args.prompt,
        "answer": answer_text,
        "layer_index": args.layer_index,
        "prompt_token_count": prompt_len,
        "question_span": list(question_span),
        "answer_token_count": len(answer_ids),
        "target": args.target,
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[answer_probe] wrote {len(results)} rows -> {args.out}")
    elif quiet_stream:
        raise SystemExit("internal error: --quiet was used without --out")

    if quiet_stream:
        quiet_stream.close()


if __name__ == "__main__":
    main()
