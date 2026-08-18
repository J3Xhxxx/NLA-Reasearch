#!/usr/bin/env python3
"""Single real-activation probe: mask unquoted AV scaffolding, then run AR."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot_common import NLACritic


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {result}")
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    return finite(a64 @ b64 / max(denominator, 1e-30))


def project(a: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    direction = np.asarray(m_hat, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-30)
    return arr - float(arr @ direction) * direction


def mask_unquoted_letters(text: str) -> tuple[str, dict[str, Any]]:
    masked: list[str] = []
    in_quote = False
    replaced_positions: list[int] = []
    quoted_letters = 0
    quote_count = 0
    for index, character in enumerate(text):
        if character == '"':
            in_quote = not in_quote
            quote_count += 1
            masked.append(character)
        elif character.isalpha() and not in_quote:
            masked.append("X")
            replaced_positions.append(index)
        else:
            masked.append(character)
            if character.isalpha() and in_quote:
                quoted_letters += 1
    if in_quote or quote_count % 2:
        raise ValueError("unbalanced ASCII double quotes")
    output = "".join(masked)
    if len(output) != len(text):
        raise AssertionError("mask changed character length")
    for index, (before, after) in enumerate(zip(text, output)):
        if index in replaced_positions:
            if not before.isalpha() or after != "X":
                raise AssertionError("invalid replaced character")
        elif before != after:
            raise AssertionError("mask altered a protected character")
    return output, {
        "character_length": len(text),
        "letters_replaced_by_X": len(replaced_positions),
        "letters_preserved_inside_quotes": quoted_letters,
        "ascii_double_quote_count": quote_count,
        "changed_character_fraction": len(replaced_positions) / max(len(text), 1),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score(prediction: np.ndarray, target: np.ndarray, m_hat: np.ndarray) -> dict[str, float]:
    raw_cosine = cosine(prediction, target)
    return {
        "direction_cosine": raw_cosine,
        "direction_mse": finite(2.0 * (1.0 - raw_cosine)),
        "centered_cosine": cosine(project(prediction, m_hat), project(target, m_hat)),
        "reconstruction_l2_norm": finite(np.linalg.norm(prediction.astype(np.float64))),
        "raw_relative_l2_error": finite(
            np.linalg.norm(prediction.astype(np.float64) - target.astype(np.float64))
            / max(float(np.linalg.norm(target.astype(np.float64))), 1e-30)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--idx", type=int, default=75)
    parser.add_argument("--out-prefix", required=True, type=Path)
    args = parser.parse_args()

    out_json = args.out_prefix.with_suffix(".json")
    out_npz = args.out_prefix.with_suffix(".npz")
    out_md = args.out_prefix.with_suffix(".md")
    out_sha = args.out_prefix.with_suffix(".sha256")
    outputs = (out_json, out_npz, out_md, out_sha)
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "refusing to overwrite: "
            + ", ".join(str(path) for path in outputs if path.exists())
        )
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    checkpoint_lines = args.checkpoint.read_text(encoding="utf-8").splitlines()
    checkpoint_rows = [json.loads(line) for line in checkpoint_lines if line.strip()]
    by_idx = {int(row["idx"]): row for row in checkpoint_rows}
    if len(by_idx) != len(checkpoint_rows) or args.idx not in by_idx:
        raise ValueError("checkpoint idx binding failure")
    av_row = by_idx[args.idx]
    original = str(av_row["explanation"])
    masked, mask_stats = mask_unquoted_letters(original)

    analysis_payload = load_json(args.analysis)
    analysis_rows = analysis_payload["rows"]
    if int(analysis_rows[args.idx]["idx"]) != args.idx:
        raise ValueError("analysis row ordering mismatch")
    meta = analysis_rows[args.idx]
    for field in ("doc_id", "position", "token"):
        if meta[field] != av_row[field]:
            raise ValueError(f"checkpoint/analysis mismatch: {field}")

    with np.load(args.vectors, allow_pickle=False) as frozen:
        x = np.asarray(frozen["x"][args.idx], dtype=np.float32)
        m_hat = np.asarray(frozen["m_hat"], dtype=np.float32)
        frozen_original_reconstruction = np.asarray(
            frozen["pred_orig"][args.idx], dtype=np.float32
        )
    if x.shape != (3840,) or m_hat.shape != x.shape:
        raise ValueError("unexpected activation shape")
    activation_sha = sha256_bytes(np.ascontiguousarray(x).tobytes())
    cohort_activation_sha = str(av_row["activation_sha256"])
    if {str(row["activation_sha256"]) for row in checkpoint_rows} != {
        cohort_activation_sha
    }:
        raise ValueError("checkpoint rows disagree on source activation hash")
    if (
        str(analysis_payload["inputs"]["activations_sha256"])
        != cohort_activation_sha
    ):
        raise ValueError("checkpoint/analysis source activation hash mismatch")

    critic = NLACritic(str(args.ar), device="cuda")
    explanation_token_counts = {
        "original": len(
            critic.tokenizer(original, add_special_tokens=False)["input_ids"]
        ),
        "structure_masked": len(
            critic.tokenizer(masked, add_special_tokens=False)["input_ids"]
        ),
    }
    full_prompt_token_counts = {
        "original": len(
            critic.tokenizer(
                critic.template.format(explanation=original),
                add_special_tokens=True,
            )["input_ids"]
        ),
        "structure_masked": len(
            critic.tokenizer(
                critic.template.format(explanation=masked),
                add_special_tokens=True,
            )["input_ids"]
        ),
    }
    original_reconstruction = critic.reconstruct(original).numpy().astype(np.float32)
    masked_reconstruction = critic.reconstruct(masked).numpy().astype(np.float32)
    mse_scale = finite(critic.mse_scale)
    del critic
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for name, vector in {
        "original": original_reconstruction,
        "masked": masked_reconstruction,
    }.items():
        if vector.shape != x.shape or not np.isfinite(vector).all():
            raise ValueError(f"invalid {name} reconstruction")

    original_scores = score(original_reconstruction, x, m_hat)
    masked_scores = score(masked_reconstruction, x, m_hat)
    deltas = {
        key: finite(masked_scores[key] - original_scores[key])
        for key in original_scores
    }
    reconstruction_comparison = {
        "raw_cosine_original_vs_masked": cosine(
            original_reconstruction, masked_reconstruction
        ),
        "centered_cosine_original_vs_masked": cosine(
            project(original_reconstruction, m_hat),
            project(masked_reconstruction, m_hat),
        ),
        "current_original_vs_frozen_original_raw_cosine": cosine(
            original_reconstruction, frozen_original_reconstruction
        ),
        "current_original_vs_frozen_original_max_abs_diff": finite(
            np.max(np.abs(original_reconstruction - frozen_original_reconstruction))
        ),
    }

    np.savez_compressed(
        out_npz,
        target_activation=x,
        reference_mean_direction=m_hat,
        ar_original=original_reconstruction,
        ar_structure_masked=masked_reconstruction,
        ar_original_frozen_n4=frozen_original_reconstruction,
    )

    artifact_inputs = {
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint)},
        "vectors": {"path": str(args.vectors), "sha256": sha256_file(args.vectors)},
        "analysis": {"path": str(args.analysis), "sha256": sha256_file(args.analysis)},
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "single_case_unquoted_structure_mask_ar_probe",
        "status": "EXPLORATORY_SINGLE_CASE_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "idx": args.idx,
            "doc_id": int(meta["doc_id"]),
            "position": int(meta["position"]),
            "token": str(meta["token"]),
            "corpus": str(meta["corpus"]),
            "source": str(meta["source"]),
            "lang": str(meta["lang"]),
            "context_tail": str(meta["context_tail"]),
            "continuation": str(meta["continuation"]),
            "row_activation_sha256_float32_bytes": activation_sha,
            "source_activation_cohort_sha256": cohort_activation_sha,
        },
        "mask_contract": {
            "definition": (
                "replace every Unicode alphabetic character outside ASCII "
                "double-quoted spans with one ASCII X; preserve all other characters"
            ),
            **mask_stats,
            "original_utf8_sha256": sha256_bytes(original.encode("utf-8")),
            "masked_utf8_sha256": sha256_bytes(masked.encode("utf-8")),
            "quoted_content_preserved": True,
            "character_length_preserved": len(original) == len(masked),
        },
        "texts": {"av_original": original, "av_structure_masked": masked},
        "token_counts": {
            "explanation_only": explanation_token_counts,
            "full_ar_prompt": full_prompt_token_counts,
            "warning": "equal character count does not imply equal tokenizer count",
        },
        "ar": {
            "model_path": str(args.ar),
            "mse_scale": mse_scale,
            "score_definition": "direction MSE = 2 * (1 - direction cosine)",
            "original": original_scores,
            "structure_masked": masked_scores,
            "masked_minus_original": deltas,
            "reconstruction_comparison": reconstruction_comparison,
        },
        "inputs": artifact_inputs,
        "scope_limits": [
            "selected single case; no population inference",
            "X masking changes AR tokenization and creates OOD text",
            "AR round-trip fidelity is not proposition-level human faithfulness",
            "quoted lexical evidence and candidates are intentionally preserved",
        ],
        "runtime_seconds": finite(time.perf_counter() - started),
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    def value(condition: str, metric: str) -> float:
        return payload["ar"][condition][metric]

    lines = [
        "# 真实 activation：AV 结构语言 X-mask → AR 单案例报告",
        "",
        f"> 冻结样本：N4/J2 `idx={args.idx}`，`{meta['source']}`，doc `{meta['doc_id']}`，position `{meta['position']}`。  ",
        "> 性质：**探索性单案例**；样本在查看本次 masked-AR 输出之前固定。",
        "",
        "## 1. 真实语料",
        "",
        "### 目标位置之前的上下文",
        "",
        "```text",
        str(meta["context_tail"]),
        "```",
        "",
        "### 真实后续文本",
        "",
        "```text",
        str(meta["continuation"]),
        "```",
        "",
        "## 2. 冻结的 AV 自然语言解释（原版）",
        "",
        "```text",
        original,
        "```",
        "",
        "## 3. 结构语言替换为 X 后",
        "",
        "规则：双引号外的每个 Unicode 字母逐个替换为一个 `X`；引号内内容、空格、标点、数字、段落和字符总长度保持不变。",
        "",
        "```text",
        masked,
        "```",
        "",
        "### 遮罩审计",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 总字符数（两版相同） | {mask_stats['character_length']} |",
        f"| 替换成 X 的字母数 | {mask_stats['letters_replaced_by_X']} |",
        f"| 引号内保留的字母数 | {mask_stats['letters_preserved_inside_quotes']} |",
        f"| 原版 explanation tokens | {explanation_token_counts['original']} |",
        f"| masked explanation tokens | {explanation_token_counts['structure_masked']} |",
        f"| 原版完整 AR prompt tokens | {full_prompt_token_counts['original']} |",
        f"| masked 完整 AR prompt tokens | {full_prompt_token_counts['structure_masked']} |",
        "",
        "## 4. AR 重建得分",
        "",
        "| 指标 | 原始 AV | X-mask AV | masked − original |",
        "|---|---:|---:|---:|",
        f"| Direction cosine ↑ | **{value('original', 'direction_cosine'):.6f}** | **{value('structure_masked', 'direction_cosine'):.6f}** | {deltas['direction_cosine']:+.6f} |",
        f"| Direction MSE ↓ | **{value('original', 'direction_mse'):.6f}** | **{value('structure_masked', 'direction_mse'):.6f}** | {deltas['direction_mse']:+.6f} |",
        f"| Centered cosine ↑ | {value('original', 'centered_cosine'):.6f} | {value('structure_masked', 'centered_cosine'):.6f} | {deltas['centered_cosine']:+.6f} |",
        f"| Raw relative L2 error ↓ | {value('original', 'raw_relative_l2_error'):.6f} | {value('structure_masked', 'raw_relative_l2_error'):.6f} | {deltas['raw_relative_l2_error']:+.6f} |",
        "",
        "AR 重建向量彼此的相似度：",
        "",
        f"- raw cosine：`{reconstruction_comparison['raw_cosine_original_vs_masked']:.6f}`",
        f"- centered cosine：`{reconstruction_comparison['centered_cosine_original_vs_masked']:.6f}`",
        f"- 本次原版 AR 与冻结 N4 原版 AR cosine：`{reconstruction_comparison['current_original_vs_frozen_original_raw_cosine']:.9f}`",
        "",
        "## 5. 解释边界",
        "",
        "- 这个差值是该样本的局部结构遮罩效应，不是总体平均效应。",
        "- 等字符数不等于等 token 数；连续 `X` 会改变 AR tokenizer 分词，因此结果同时包含结构语义移除与 OOD/tokenization 扰动。",
        "- 引号内候选和内容词被刻意保留，所以测试的是 unquoted scaffold 的增量贡献，不是移除全部文本信息。",
        "- AR 分数衡量内部 activation round-trip，不等价于人类命题级忠实度。",
        "",
        "## 6. 文件绑定",
        "",
        f"- row activation SHA-256（x[75] float32 bytes）：`{activation_sha}`",
        f"- source activation cohort SHA-256：`{cohort_activation_sha}`",
        f"- original text SHA-256：`{payload['mask_contract']['original_utf8_sha256']}`",
        f"- masked text SHA-256：`{payload['mask_contract']['masked_utf8_sha256']}`",
        f"- protocol SHA-256：`{artifact_inputs['protocol']['sha256']}`",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_sha.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in (out_npz, out_json, out_md)
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "original_cos": original_scores["direction_cosine"],
        "masked_cos": masked_scores["direction_cosine"],
        "delta_cos": deltas["direction_cosine"],
        "original_mse": original_scores["direction_mse"],
        "masked_mse": masked_scores["direction_mse"],
        "tokens": payload["token_counts"],
    }, ensure_ascii=False, indent=2), flush=True)
    print("STRUCTURE_MASK_AR_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
