#!/usr/bin/env python3
"""Independent structural and numerical validation for the C1 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_and_normalize(rows: np.ndarray, mean_direction: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    mean_direction = np.asarray(mean_direction, dtype=np.float64)
    mean_direction /= np.linalg.norm(mean_direction)
    projected = rows - np.outer(rows @ mean_direction, mean_direction)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero projected row")
    return projected / norms


def average_rank(scores: np.ndarray, target_index: int) -> float:
    target = float(scores[target_index])
    return float(
        1
        + np.sum(scores > target)
        + 0.5 * max(0, int(np.sum(scores == target)) - 1)
    )


def parse_gpu_log(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.open(encoding="utf-8")))
    if len(rows) < 2:
        raise ValueError("GPU log has no samples")
    parsed = []
    for row in rows[1:]:
        timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        parsed.append((timestamp, *[float(value.strip()) for value in row[1:]]))
    elapsed = np.array(
        [(row[0] - parsed[0][0]).total_seconds() for row in parsed],
        dtype=np.float64,
    )
    values = np.array([row[1:] for row in parsed], dtype=np.float64)
    energy_wh = float(np.trapezoid(values[:, 2], elapsed) / 3600)
    return {
        "n_samples": len(parsed),
        "sampled_span_seconds": float(elapsed[-1]),
        "gpu_utilization_mean_percent": float(np.mean(values[:, 0])),
        "gpu_utilization_peak_percent": float(np.max(values[:, 0])),
        "memory_peak_mib": float(np.max(values[:, 1])),
        "power_mean_w": float(np.mean(values[:, 2])),
        "power_peak_w": float(np.max(values[:, 2])),
        "temperature_peak_c": float(np.max(values[:, 3])),
        "sampled_energy_wh": energy_wh,
        "timestamp_note": "nvidia-smi timestamps are server-local Asia/Shanghai",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--b6-result", required=True, type=Path)
    parser.add_argument("--gpu-log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    b6_result = json.loads(args.b6_result.read_text(encoding="utf-8"))
    if benchmark["status"] != "benchmark_frozen_before_C1_AR":
        raise ValueError("benchmark status mismatch")
    if result["status"] != "complete":
        raise ValueError("result status mismatch")
    if result["inputs"]["benchmark"]["sha256"] != sha256_file(args.benchmark):
        raise ValueError("result does not reference the supplied benchmark")

    result_rows = result["scored_candidates"]
    result_ids = [row["candidate_id"] for row in result_rows]
    if len(result_rows) != 432 or len(set(result_ids)) != 432:
        raise ValueError("expected 432 unique result rows")

    checkpoint_rows = [
        json.loads(line)
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint_keys = [row["key"] for row in checkpoint_rows]
    checkpoint_types = Counter(row["job_type"] for row in checkpoint_rows)
    if len(checkpoint_rows) != 480 or len(set(checkpoint_keys)) != 480:
        raise ValueError("expected 480 unique checkpoint rows")
    if checkpoint_types != {
        "candidate_generation": 48,
        "blind_base_context_judge": 432,
    }:
        raise ValueError(f"checkpoint job counts drifted: {checkpoint_types}")
    judge_rows = [
        row for row in checkpoint_rows if row["job_type"] == "blind_base_context_judge"
    ]
    if not all(row["parse_ok"] for row in judge_rows):
        raise ValueError("judge parse failure")

    with np.load(args.vectors, allow_pickle=False) as archive:
        vector_ids = archive["candidate_ids"].astype(str).tolist()
        vector_features = np.asarray(archive["feature_ids"], dtype=np.int64)
        vector_kinds = archive["kinds"].astype(str).tolist()
        reconstructions = np.asarray(
            archive["reconstruction_vectors"], dtype=np.float64
        )
        semantic_ids = np.asarray(archive["semantic_feature_ids"], dtype=np.int64)
        semantic_directions = np.asarray(
            archive["semantic_directions"], dtype=np.float64
        )
        saved_similarity = np.asarray(
            archive["semantic_similarity"], dtype=np.float64
        )
        mean_direction = np.asarray(archive["m_hat"], dtype=np.float64)

    if vector_ids != result_ids:
        raise ValueError("result/vector candidate order mismatch")
    if vector_features.tolist() != [int(row["feature"]) for row in result_rows]:
        raise ValueError("result/vector feature order mismatch")
    if vector_kinds != [row["kind"] for row in result_rows]:
        raise ValueError("result/vector kind order mismatch")
    expected_shapes = {
        "reconstructions": [432, 3840],
        "semantic_directions": [24, 3840],
        "similarity": [432, 24],
    }
    actual_shapes = {
        "reconstructions": list(reconstructions.shape),
        "semantic_directions": list(semantic_directions.shape),
        "similarity": list(saved_similarity.shape),
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"vector shape mismatch: {actual_shapes}")
    if not all(
        np.isfinite(array).all()
        for array in (reconstructions, semantic_directions, saved_similarity)
    ):
        raise ValueError("non-finite vector artifact")

    recomputed_similarity = project_and_normalize(
        reconstructions, mean_direction
    ) @ project_and_normalize(semantic_directions, mean_direction).T
    similarity_error = float(np.max(np.abs(recomputed_similarity - saved_similarity)))

    feature_to_index = {
        int(feature): index for index, feature in enumerate(semantic_ids)
    }
    target_score_errors = []
    feature_rank_errors = []
    for row_index, row in enumerate(result_rows):
        target_index = feature_to_index[int(row["feature"])]
        scores = recomputed_similarity[row_index]
        target_score_errors.append(
            abs(float(row["target_cos_centered"]) - float(scores[target_index]))
        )
        feature_rank_errors.append(
            abs(
                float(row["feature_retrieval_rank"])
                - average_rank(scores, target_index)
            )
        )

    old_nla = {
        int(row["feature"]): row["explanation"]
        for row in b6_result["scored_generation_rows"]
        if row.get("probe_type") == "direction"
        and row.get("group") == "semantic_new"
        and int(row.get("sign", 0)) == 1
        and int(row.get("sample_index", -1)) == 0
    }
    new_nla = {
        int(row["feature"]): row["text"]
        for row in result_rows
        if row["kind"] == "nla_original"
    }
    exact_nla_text_match = old_nla == new_nla
    if not exact_nla_text_match:
        raise ValueError("C1 no longer uses exact B6 NLA strings")

    qa = {
        "status": "all_checks_passed",
        "counts": {
            "result_rows": len(result_rows),
            "checkpoint_rows": len(checkpoint_rows),
            "candidate_generation_rows": checkpoint_types["candidate_generation"],
            "judge_rows": checkpoint_types["blind_base_context_judge"],
            "judge_parse_ok": sum(bool(row["parse_ok"]) for row in judge_rows),
            "unique_reconstructed_texts": int(result["runtime"]["n_unique_texts"]),
        },
        "vector_shapes": actual_shapes,
        "numerical": {
            "max_abs_saved_similarity_recompute_error": similarity_error,
            "max_abs_target_score_recompute_error": float(max(target_score_errors)),
            "max_abs_feature_rank_recompute_error": float(max(feature_rank_errors)),
            "max_abs_b6_nla_original_score_error": float(
                result["qa"]["max_abs_nla_original_recompute_error"]
            ),
            "all_vectors_finite": True,
            "exact_b6_nla_text_match": exact_nla_text_match,
        },
        "runtime": result["runtime"],
        "gpu_monitor": parse_gpu_log(args.gpu_log),
        "sha256": {
            path.name: sha256_file(path)
            for path in (
                args.benchmark,
                args.result,
                args.vectors,
                args.checkpoint,
                args.b6_result,
                args.gpu_log,
            )
        },
    }
    if max(
        similarity_error,
        max(target_score_errors),
        max(feature_rank_errors),
        qa["numerical"]["max_abs_b6_nla_original_score_error"],
    ) > 2e-6:
        raise ValueError("numerical recomputation exceeded tolerance")
    args.out.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C1_PILOT_VALIDATION_OK")
    print(json.dumps(qa["numerical"]))


if __name__ == "__main__":
    main()
