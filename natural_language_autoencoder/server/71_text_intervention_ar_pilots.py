#!/usr/bin/env python3
"""Run two frozen text interventions against one frozen real activation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot_common import NLACritic


QUOTE_RE = re.compile(r'"[^"\n]*"')


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
        raise ValueError(f"non-finite result: {result}")
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


def scores(prediction: np.ndarray, target: np.ndarray, m_hat: np.ndarray) -> dict[str, float]:
    raw_cosine = cosine(prediction, target)
    return {
        "direction_cosine": raw_cosine,
        "direction_mse": finite(2.0 * (1.0 - raw_cosine)),
        "centered_cosine": cosine(project(prediction, m_hat), project(target, m_hat)),
        "raw_relative_l2_error": finite(
            np.linalg.norm(prediction.astype(np.float64) - target.astype(np.float64))
            / max(float(np.linalg.norm(target.astype(np.float64))), 1e-30)
        ),
        "reconstruction_l2_norm": finite(np.linalg.norm(prediction.astype(np.float64))),
    }


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {int(row["idx"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate checkpoint idx")
    return result


def paragraphs(payload: dict[str, Any], key: str) -> list[str]:
    value = payload[key]
    if key in ("pilot_1", "pilot_2"):
        value = value["paragraphs"]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must contain exactly three paragraphs")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{key} contains an invalid paragraph")
    return value


def write_sidecar(paths: list[Path], out: Path) -> None:
    out.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in paths) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--paraphrase-prefix", required=True, type=Path)
    parser.add_argument("--partial-x-prefix", required=True, type=Path)
    args = parser.parse_args()

    all_outputs: list[Path] = []
    for prefix in (args.paraphrase_prefix, args.partial_x_prefix):
        all_outputs.extend(
            [
                prefix.with_suffix(".json"),
                prefix.with_suffix(".npz"),
                prefix.with_suffix(".md"),
                prefix.with_suffix(".sha256"),
            ]
        )
    if any(path.exists() for path in all_outputs):
        raise FileExistsError(
            "refusing to overwrite: "
            + ", ".join(str(path) for path in all_outputs if path.exists())
        )
    for path in all_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    contract = json.loads(args.inputs.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_AR":
        raise ValueError("input contract is not frozen")
    idx = int(contract["sample"]["idx"])
    original_paragraphs = paragraphs(contract, "original_paragraphs")
    paraphrase_paragraphs = paragraphs(contract, "pilot_1")
    partial_x_paragraphs = paragraphs(contract, "pilot_2")
    original = "\n\n".join(original_paragraphs)
    paraphrase = "\n\n".join(paraphrase_paragraphs)
    partial_x = "\n\n".join(partial_x_paragraphs)

    checkpoint = load_checkpoint(args.checkpoint)
    if idx not in checkpoint or checkpoint[idx]["explanation"] != original:
        raise ValueError("contract original does not match frozen AV checkpoint")
    av_row = checkpoint[idx]
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    meta = analysis["rows"][idx]
    if int(meta["idx"]) != idx:
        raise ValueError("analysis row ordering mismatch")
    for field in ("doc_id", "position", "token"):
        if meta[field] != av_row[field] or meta[field] != contract["sample"][field]:
            raise ValueError(f"sample metadata mismatch: {field}")

    original_quotes_by_paragraph = [QUOTE_RE.findall(item) for item in original_paragraphs]
    paraphrase_quotes_by_paragraph = [QUOTE_RE.findall(item) for item in paraphrase_paragraphs]
    partial_x_quotes_by_paragraph = [QUOTE_RE.findall(item) for item in partial_x_paragraphs]
    if paraphrase_quotes_by_paragraph != original_quotes_by_paragraph:
        raise ValueError("paraphrase did not preserve quoted spans paragraph-by-paragraph")
    if partial_x_quotes_by_paragraph != original_quotes_by_paragraph:
        raise ValueError("partial-X did not preserve quoted spans paragraph-by-paragraph")
    if any(a == b for a, b in zip(original_paragraphs, paraphrase_paragraphs)):
        raise ValueError("each paraphrase paragraph must differ from its source")

    with np.load(args.vectors, allow_pickle=False) as frozen:
        target = np.asarray(frozen["x"][idx], dtype=np.float32)
        m_hat = np.asarray(frozen["m_hat"], dtype=np.float32)
        frozen_original = np.asarray(frozen["pred_orig"][idx], dtype=np.float32)
    if target.shape != (3840,) or m_hat.shape != target.shape:
        raise ValueError("unexpected activation shape")

    critic = NLACritic(str(args.ar), device="cuda")
    texts = {
        "original": original,
        "paragraph_independent_paraphrase": paraphrase,
        "user_supplied_partial_x_mask": partial_x,
    }
    token_counts = {
        name: {
            "explanation_only": len(
                critic.tokenizer(text, add_special_tokens=False)["input_ids"]
            ),
            "full_ar_prompt": len(
                critic.tokenizer(
                    critic.template.format(explanation=text),
                    add_special_tokens=True,
                )["input_ids"]
            ),
        }
        for name, text in texts.items()
    }
    text_statistics = {
        name: {
            "characters": len(text),
            "whitespace_delimited_words": len(text.split()),
            "unicode_letters": sum(character.isalpha() for character in text),
            "paragraphs": len(text.split("\n\n")),
            "quoted_spans": len(QUOTE_RE.findall(text)),
        }
        for name, text in texts.items()
    }
    reconstructions = {
        name: critic.reconstruct(text).numpy().astype(np.float32)
        for name, text in texts.items()
    }
    mse_scale = finite(critic.mse_scale)
    del critic
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for name, vector in reconstructions.items():
        if vector.shape != target.shape or not np.isfinite(vector).all():
            raise ValueError(f"invalid AR reconstruction: {name}")

    original_scores = scores(reconstructions["original"], target, m_hat)
    input_hash = sha256_file(args.inputs)
    shared = {
        "schema_version": 1,
        "status": "EXPLORATORY_SINGLE_CASE_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "idx": idx,
            "doc_id": int(meta["doc_id"]),
            "position": int(meta["position"]),
            "token": str(meta["token"]),
            "corpus": str(meta["corpus"]),
            "source": str(meta["source"]),
            "lang": str(meta["lang"]),
            "context_tail": str(meta["context_tail"]),
            "continuation": str(meta["continuation"]),
            "row_activation_sha256_float32_bytes": sha256_bytes(
                np.ascontiguousarray(target).tobytes()
            ),
            "source_activation_cohort_sha256": str(av_row["activation_sha256"]),
        },
        "original_av": original,
        "original_scores": original_scores,
        "model": {"ar_path": str(args.ar), "mse_scale": mse_scale},
        "input_contract": {"path": str(args.inputs), "sha256": input_hash},
        "script_sha256": sha256_file(Path(__file__)),
        "original_current_vs_frozen_n4": {
            "exact_float32_equal": bool(
                np.array_equal(reconstructions["original"], frozen_original)
            ),
            "raw_cosine": cosine(reconstructions["original"], frozen_original),
            "max_abs_diff": finite(
                np.max(np.abs(reconstructions["original"] - frozen_original))
            ),
        },
    }

    variants = [
        {
            "condition": "paragraph_independent_paraphrase",
            "prefix": args.paraphrase_prefix,
            "title": "逐段独立同义改写 → AR 单案例报告",
            "text": paraphrase,
            "paragraphs": paraphrase_paragraphs,
            "construction": contract["pilot_1"]["construction"],
            "scope_limits": [
                "single selected case; no population inference",
                "quoted spans are preserved, but paraphrase tokenization and wording change",
                "semantic equivalence is a controlled authoring judgment, not a human panel verdict",
                "AR round-trip fidelity is not proposition-level human faithfulness",
            ],
        },
        {
            "condition": "user_supplied_partial_x_mask",
            "prefix": args.partial_x_prefix,
            "title": "用户指定 partial-X 文本 → AR 单案例报告",
            "text": partial_x,
            "paragraphs": partial_x_paragraphs,
            "construction": contract["pilot_2"]["construction"],
            "scope_limits": [
                "single selected case; no population inference",
                "the user-supplied X spans alter semantics and tokenizer behavior",
                "unmasked unquoted semantic language and all quoted spans remain",
                "AR round-trip fidelity is not proposition-level human faithfulness",
            ],
        },
    ]

    for variant in variants:
        condition = variant["condition"]
        prefix: Path = variant["prefix"]
        variant_scores = scores(reconstructions[condition], target, m_hat)
        deltas = {
            key: finite(variant_scores[key] - original_scores[key])
            for key in original_scores
        }
        comparison = {
            "raw_cosine_original_vs_variant": cosine(
                reconstructions["original"], reconstructions[condition]
            ),
            "centered_cosine_original_vs_variant": cosine(
                project(reconstructions["original"], m_hat),
                project(reconstructions[condition], m_hat),
            ),
        }
        payload = {
            **shared,
            "experiment": condition + "_ar_probe",
            "intervention": {
                "condition": condition,
                "construction": variant["construction"],
                "text": variant["text"],
                "paragraphs": variant["paragraphs"],
                "utf8_sha256": sha256_bytes(variant["text"].encode("utf-8")),
                "quotes_preserved_paragraph_by_paragraph": True,
            },
            "token_counts": {
                "original": token_counts["original"],
                "variant": token_counts[condition],
            },
            "text_statistics": {
                "original": text_statistics["original"],
                "variant": text_statistics[condition],
                "variant_minus_original_characters": (
                    text_statistics[condition]["characters"]
                    - text_statistics["original"]["characters"]
                ),
            },
            "variant_scores": variant_scores,
            "variant_minus_original": deltas,
            "reconstruction_comparison": comparison,
            "scope_limits": variant["scope_limits"],
            "runtime_seconds_shared": finite(time.perf_counter() - started),
        }
        out_json = prefix.with_suffix(".json")
        out_npz = prefix.with_suffix(".npz")
        out_md = prefix.with_suffix(".md")
        out_sha = prefix.with_suffix(".sha256")
        np.savez_compressed(
            out_npz,
            target_activation=target,
            reference_mean_direction=m_hat,
            ar_original=reconstructions["original"],
            ar_variant=reconstructions[condition],
            ar_original_frozen_n4=frozen_original,
        )
        out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        lines = [
            f"# {variant['title']}",
            "",
            f"> 冻结样本：N4/J2 `idx={idx}`，`{meta['source']}`，doc `{meta['doc_id']}`，position `{meta['position']}`。  ",
            "> 性质：**探索性单案例**；intervention 文本在本次 AR 评分前冻结。",
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
            "## 2. 冻结的原始 AV 自然语言解释",
            "",
            "```text",
            original,
            "```",
            "",
            "## 3. 本次 intervention 文本",
            "",
            f"构造规则：{variant['construction']}",
            "",
            "```text",
            variant["text"],
            "```",
            "",
        ]
        if condition == "paragraph_independent_paraphrase":
            lines.extend(["### 逐段对应", ""])
            for number, (source, rewritten) in enumerate(
                zip(original_paragraphs, variant["paragraphs"]), start=1
            ):
                lines.extend(
                    [
                        f"#### 第 {number} 段",
                        "",
                        "原段：",
                        "",
                        f"> {source}",
                        "",
                        "独立同义改写：",
                        "",
                        f"> {rewritten}",
                        "",
                    ]
                )
        lines.extend(
            [
                "## 4. 文本与 Tokenizer 审计",
                "",
                "| 文本 | 字符 | whitespace words | explanation tokens | 完整 AR prompt tokens |",
                "|---|---:|---:|---:|---:|",
                f"| 原始 AV | {text_statistics['original']['characters']} | {text_statistics['original']['whitespace_delimited_words']} | {token_counts['original']['explanation_only']} | {token_counts['original']['full_ar_prompt']} |",
                f"| intervention | {text_statistics[condition]['characters']} | {text_statistics[condition]['whitespace_delimited_words']} | {token_counts[condition]['explanation_only']} | {token_counts[condition]['full_ar_prompt']} |",
                "",
                f"字符数变化：`{text_statistics[condition]['characters'] - text_statistics['original']['characters']:+d}`。两版均为 3 段、9 个顺序完全相同的双引号 span。",
                "",
                "## 5. AR 重建得分",
                "",
                "| 指标 | 原始 AV | intervention | intervention − original |",
                "|---|---:|---:|---:|",
                f"| Direction cosine ↑ | **{original_scores['direction_cosine']:.6f}** | **{variant_scores['direction_cosine']:.6f}** | {deltas['direction_cosine']:+.6f} |",
                f"| Direction MSE ↓ | **{original_scores['direction_mse']:.6f}** | **{variant_scores['direction_mse']:.6f}** | {deltas['direction_mse']:+.6f} |",
                f"| Centered cosine ↑ | {original_scores['centered_cosine']:.6f} | {variant_scores['centered_cosine']:.6f} | {deltas['centered_cosine']:+.6f} |",
                f"| Raw relative L2 error ↓ | {original_scores['raw_relative_l2_error']:.6f} | {variant_scores['raw_relative_l2_error']:.6f} | {deltas['raw_relative_l2_error']:+.6f} |",
                "",
                "原始与 intervention 的 AR reconstruction：",
                "",
                f"- raw cosine：`{comparison['raw_cosine_original_vs_variant']:.6f}`",
                f"- centered cosine：`{comparison['centered_cosine_original_vs_variant']:.6f}`",
                f"- 本次原始 AR 与冻结 N4 原始 AR 完全相同：`{shared['original_current_vs_frozen_n4']['exact_float32_equal']}`",
                "",
                "## 6. 解释边界",
                "",
                *[f"- {item}" for item in variant["scope_limits"]],
                "",
                "## 7. 文件绑定",
                "",
                f"- input contract SHA-256：`{input_hash}`",
                f"- intervention text SHA-256：`{payload['intervention']['utf8_sha256']}`",
                f"- row activation SHA-256：`{shared['sample']['row_activation_sha256_float32_bytes']}`",
                f"- runner script SHA-256：`{shared['script_sha256']}`",
                "",
            ]
        )
        out_md.write_text("\n".join(lines), encoding="utf-8")
        write_sidecar([out_npz, out_json, out_md], out_sha)
        print(
            json.dumps(
                {
                    "condition": condition,
                    "cos": variant_scores["direction_cosine"],
                    "delta_cos": deltas["direction_cosine"],
                    "centered_cos": variant_scores["centered_cosine"],
                    "delta_centered_cos": deltas["centered_cosine"],
                    "tokens": token_counts[condition],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    print("TEXT_INTERVENTION_AR_PILOTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
