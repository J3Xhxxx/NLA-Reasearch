#!/usr/bin/env python3
"""Generate one deterministic Gaussian L32 activation and run AV -> AR.

This is an exploratory/OOD probe, not a semantic-faithfulness experiment.
The input is an isotropic Gaussian whose expected L2 norm matches a frozen
real-activation reference cohort.  NLAClient subsequently normalizes every
non-zero input to the AV checkpoint's learned injection scale.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot_common import AVLocal, NLACritic


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    return float(a64 @ b64 / max(denom, 1e-30))


def project_mean_direction(a: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    direction = np.asarray(m_hat, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-30)
    if arr.ndim == 1:
        return arr - float(arr @ direction) * direction
    return arr - np.outer(arr @ direction, direction)


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite result: {result}")
    return result


def percentile_rank(values: np.ndarray, value: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(100.0 * np.mean(values <= value))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--av", required=True, type=Path)
    parser.add_argument("--ar", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    out_json = args.out_prefix.with_suffix(".json")
    out_npz = args.out_prefix.with_suffix(".npz")
    out_md = args.out_prefix.with_suffix(".md")
    out_sha = args.out_prefix.with_suffix(".sha256")
    outputs = (out_json, out_npz, out_md, out_sha)
    if any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    reference_sha = sha256_file(args.reference)
    with np.load(args.reference, allow_pickle=False) as frozen:
        if "x" not in frozen.files or "m_hat" not in frozen.files:
            raise ValueError("reference NPZ must contain x and m_hat")
        real_x = np.asarray(frozen["x"], dtype=np.float32)
        m_hat = np.asarray(frozen["m_hat"], dtype=np.float32)
    if real_x.ndim != 2 or not np.isfinite(real_x).all():
        raise ValueError("invalid real-activation reference matrix")
    if m_hat.shape != (real_x.shape[1],) or not np.isfinite(m_hat).all():
        raise ValueError("invalid reference mean direction")

    dimension = int(real_x.shape[1])
    real_norms = np.linalg.norm(real_x.astype(np.float64), axis=1)
    target_expected_norm = float(real_norms.mean())
    # If each coordinate is N(0, sigma^2), E||g|| is approximately sigma*sqrt(d).
    # At d=3840 the approximation error is negligible for this diagnostic probe.
    sigma = target_expected_norm / math.sqrt(dimension)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    standard_normal = rng.standard_normal(dimension).astype(np.float32)
    activation = (standard_normal * np.float32(sigma)).astype(np.float32)
    if activation.shape != (dimension,) or not np.isfinite(activation).all():
        raise ValueError("Gaussian activation is invalid")
    activation_sha = sha256_bytes(np.ascontiguousarray(activation).tobytes())

    print(
        f"[probe] seed={args.seed} d={dimension} sigma={sigma:.6f} "
        f"norm={np.linalg.norm(activation):.3f}",
        flush=True,
    )

    av_started = time.perf_counter()
    av = AVLocal(str(args.av), device="cuda")
    injection_scale = finite_float(av.client.cfg.injection_scale)
    explanation = av.generate(
        activation,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
    )
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("AV returned an empty explanation")
    explanation = explanation.strip()
    av_seconds = time.perf_counter() - av_started
    del av
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[probe] AV complete in {av_seconds:.1f}s", flush=True)

    ar_started = time.perf_counter()
    critic = NLACritic(str(args.ar), device="cuda")
    reconstruction = critic.reconstruct(explanation).numpy().astype(np.float32)
    mse_scale = finite_float(critic.mse_scale)
    if reconstruction.shape != activation.shape or not np.isfinite(reconstruction).all():
        raise ValueError("AR returned an invalid reconstruction")
    ar_seconds = time.perf_counter() - ar_started
    del critic
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[probe] AR complete in {ar_seconds:.1f}s", flush=True)

    activation_norm = finite_float(np.linalg.norm(activation.astype(np.float64)))
    reconstruction_norm = finite_float(
        np.linalg.norm(reconstruction.astype(np.float64))
    )
    injected_activation = (
        activation.astype(np.float64) / activation_norm * injection_scale
    ).astype(np.float32)
    raw_cos = cosine(reconstruction, activation)
    direction_mse = finite_float(2.0 * (1.0 - raw_cos))
    centered_cos = cosine(
        project_mean_direction(reconstruction, m_hat),
        project_mean_direction(activation, m_hat),
    )
    relative_l2 = finite_float(
        np.linalg.norm(
            reconstruction.astype(np.float64) - activation.astype(np.float64)
        )
        / activation_norm
    )

    real_raw_cos = np.asarray(
        [cosine(row, activation) for row in real_x], dtype=np.float64
    )
    activation_c = project_mean_direction(activation, m_hat)
    real_c = project_mean_direction(real_x, m_hat)
    real_centered_cos = np.asarray(
        [cosine(row, activation_c) for row in real_c], dtype=np.float64
    )
    nearest_raw_idx = int(np.argmax(real_raw_cos))
    nearest_centered_idx = int(np.argmax(real_centered_cos))

    explanation_sha = sha256_bytes(explanation.encode("utf-8"))
    reconstruction_sha = sha256_bytes(
        np.ascontiguousarray(reconstruction).tobytes()
    )
    np.savez_compressed(
        out_npz,
        input_activation=activation,
        standard_normal_draw=standard_normal,
        av_injected_activation=injected_activation,
        ar_reconstruction=reconstruction,
        reference_mean_direction=m_hat,
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "single_gaussian_activation_av_ar_probe",
        "scope": "exploratory OOD control; not semantic or proposition-level fidelity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "seed": int(args.seed),
            "bit_generator": "NumPy PCG64",
            "dimension": dimension,
            "distribution": "independent N(0, sigma^2) coordinates",
            "sigma": sigma,
            "sigma_rule": "mean_reference_l2_norm / sqrt(dimension)",
            "reference_mean_l2_norm": target_expected_norm,
            "input_activation_sha256_float32_bytes": activation_sha,
        },
        "models": {
            "av_path": str(args.av),
            "ar_path": str(args.ar),
            "av_injection_scale": injection_scale,
            "ar_mse_scale": mse_scale,
            "av_temperature": 0.0,
            "av_max_new_tokens": int(args.max_new_tokens),
        },
        "reference": {
            "path": str(args.reference),
            "sha256": reference_sha,
            "n": int(real_x.shape[0]),
            "dimension": dimension,
            "norm_mean": finite_float(real_norms.mean()),
            "norm_median": finite_float(np.median(real_norms)),
            "norm_std": finite_float(real_norms.std()),
        },
        "input_diagnostics": {
            "shape": [dimension],
            "dtype": "float32",
            "mean": finite_float(activation.mean(dtype=np.float64)),
            "std": finite_float(activation.std(dtype=np.float64)),
            "minimum": finite_float(activation.min()),
            "maximum": finite_float(activation.max()),
            "l2_norm": activation_norm,
            "reference_norm_percentile": percentile_rank(real_norms, activation_norm),
            "av_effective_injected_l2_norm": finite_float(
                np.linalg.norm(injected_activation.astype(np.float64))
            ),
            "nearest_real_raw_cosine": finite_float(real_raw_cos[nearest_raw_idx]),
            "nearest_real_raw_index": nearest_raw_idx,
            "mean_real_raw_cosine": finite_float(real_raw_cos.mean()),
            "nearest_real_centered_cosine": finite_float(
                real_centered_cos[nearest_centered_idx]
            ),
            "nearest_real_centered_index": nearest_centered_idx,
            "mean_real_centered_cosine": finite_float(real_centered_cos.mean()),
            "interpretation": (
                "Norm-matched isotropic Gaussian; covariance/mean structure is not "
                "matched, so this remains an out-of-distribution random direction."
            ),
        },
        "av": {
            "explanation": explanation,
            "explanation_utf8_sha256": explanation_sha,
            "seconds": finite_float(av_seconds),
        },
        "ar": {
            "reconstruction_sha256_float32_bytes": reconstruction_sha,
            "reconstruction_l2_norm": reconstruction_norm,
            "direction_cosine": raw_cos,
            "direction_mse": direction_mse,
            "direction_mse_definition": "2 * (1 - direction_cosine)",
            "centered_cosine_using_frozen_n4_mean_direction": centered_cos,
            "raw_relative_l2_error": relative_l2,
            "seconds": finite_float(ar_seconds),
            "interpretation": (
                "These scores measure AV-to-AR direction reconstruction only; "
                "they do not validate the truth of the AV text."
            ),
        },
        "artifacts": {
            "npz": str(out_npz),
            "json": str(out_json),
            "markdown": str(out_md),
        },
        "runtime_seconds": finite_float(time.perf_counter() - started),
    }
    write_json(out_json, payload)

    preview = activation[:24]
    lines = [
        "# 单个高斯 activation 的 AV → AR 结果",
        "",
        f"> 生成时间（UTC）：`{payload['created_at_utc']}`  ",
        "> 性质：**探索性 OOD 随机对照**；不能把 AV 文本当成该随机向量的“真实语义”。",
        "",
        "## 输入张量",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| shape / dtype | `{activation.shape}` / `float32` |",
        f"| 随机种子 | `{args.seed}`（NumPy PCG64） |",
        f"| 分布 | 每维独立 `N(0, {sigma:.6f}²)` |",
        f"| L2 norm | {activation_norm:.6f} |",
        f"| 冻结真实 activation 平均 norm | {target_expected_norm:.6f} |",
        f"| AV 内部实际注入 norm | {injection_scale:.6f} |",
        f"| 相对真实 cohort 的 norm percentile | {payload['input_diagnostics']['reference_norm_percentile']:.1f}% |",
        f"| 最近真实向量 raw cosine | {payload['input_diagnostics']['nearest_real_raw_cosine']:.6f} |",
        f"| 最近真实向量 centered cosine | {payload['input_diagnostics']['nearest_real_centered_cosine']:.6f} |",
        "",
        "前 24 个元素（完整 3840 维张量保存在 NPZ 中）：",
        "",
        "```text",
        np.array2string(preview, precision=5, separator=", ", max_line_width=110),
        "```",
        "",
        "读取方式：",
        "",
        "```python",
        "import numpy as np, torch",
        f"x = np.load({out_npz.name!r})['input_activation']",
        "tensor = torch.from_numpy(x)  # shape: [3840]",
        "```",
        "",
        "## AV 原文",
        "",
        "```text",
        explanation,
        "```",
        "",
        "## AR 得分",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
        f"| Direction cosine | **{fmt(raw_cos)}** | AV 文本经 AR 重建后，与输入方向的余弦 |",
        f"| Direction MSE | **{fmt(direction_mse)}** | 官方口径 `2(1-cos)`；0 最好、2 约为正交 |",
        f"| Centered cosine | {fmt(centered_cos)} | 投影掉冻结 N4 均值方向后的辅助诊断 |",
        f"| Raw relative L2 error | {fmt(relative_l2)} | 未做方向归一化的补充量，不是官方主分 |",
        f"| AR reconstruction norm | {fmt(reconstruction_norm)} | AR 输出向量的原始 L2 norm |",
        "",
        "## 如何解释",
        "",
        "- 这是维度合法、可直接输入 AV 的高斯张量；AV 会把任何非零输入重标到 checkpoint 的注入尺度。",
        "- 它只匹配真实 activation 的平均范数，不匹配均值、协方差或模型流形，因此仍是强 OOD 对照。",
        "- AV 即使输出流畅、具体的文本，也不代表随机向量客观携带那些命题；AR 分数只衡量同一 NLA 系统的 round-trip 方向重建。",
        "",
        "## 文件绑定",
        "",
        f"- 输入 tensor SHA-256（float32 bytes）：`{activation_sha}`",
        f"- AV 文本 SHA-256（UTF-8）：`{explanation_sha}`",
        f"- AR tensor SHA-256（float32 bytes）：`{reconstruction_sha}`",
        f"- 真实 reference SHA-256：`{reference_sha}`",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    hashes = []
    for path in (out_npz, out_json, out_md):
        hashes.append(f"{sha256_file(path)}  {path.name}")
    out_sha.write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(f"[probe] wrote {out_npz}, {out_json}, {out_md}, {out_sha}", flush=True)
    print("GAUSSIAN_AV_AR_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
