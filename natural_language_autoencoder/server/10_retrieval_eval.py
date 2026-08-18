#!/usr/bin/env python3
"""B2: discriminative retrieval evaluation for saved NLA/SAE reconstructions.

The input ``recon_vectors.npz`` already contains all vectors needed for this
evaluation, so no model or GPU is required.  Every vector is projected away
from the saved dataset-mean direction before cosine similarities are
computed.

For each reconstruction, the primary task is 40-way instance retrieval:
rank all original activations and ask whether the paired activation is
retrieved at Top-1/Top-5.  We additionally report:

* MRR and the paired-score margin over the best non-paired candidate;
* document/topic retrieval and within-document instance retrieval, so a
  method cannot look good merely by recognizing one of the five prompts;
* random-bijection permutation nulls;
* paired NLA-vs-SAE sign-flip tests, both per query and clustered by document.

Example:
    python 10_retrieval_eval.py \
        --vectors /root/autodl-tmp/results/recon_vectors.npz \
        --metadata /root/autodl-tmp/results/nla_results.json \
        --out /root/autodl-tmp/results/retrieval_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


METHOD_KEYS = {
    "nla": "pred_full",
    "sae_small": "recon_sae_small",
    "sae_big": "recon_sae_big",
    "residual_text_control": "pred_resid",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_and_normalize(x: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    """Project the mean direction out row-wise, then L2-normalize."""
    x = np.asarray(x, dtype=np.float64)
    projected = x - np.outer(x @ m_hat, m_hat)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms[:, 0] <= 1e-12).tolist()
        raise ValueError(f"zero-norm vectors after projection at rows {bad}")
    return projected / norms


def rank_matrix(similarity: np.ndarray) -> np.ndarray:
    """Return one-based candidate ranks for every query/candidate pair."""
    order = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(similarity.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, similarity.shape[1] + 1)
    return ranks


def metric_vectors(
    similarity: np.ndarray,
    ranks: np.ndarray,
    matched_candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-query metrics for an arbitrary one-to-one matching."""
    n = similarity.shape[0]
    rows = np.arange(n)
    matched_ranks = ranks[rows, matched_candidates]
    matched_scores = similarity[rows, matched_candidates]

    top_order = np.argsort(-similarity, axis=1, kind="stable")
    row_best = top_order[:, 0]
    row_second = top_order[:, 1]
    best_nonmatch = np.where(row_best == matched_candidates, row_second, row_best)
    margins = matched_scores - similarity[rows, best_nonmatch]

    return {
        "top1": (matched_ranks <= 1).astype(np.float64),
        "top5": (matched_ranks <= min(5, n)).astype(np.float64),
        "reciprocal_rank": 1.0 / matched_ranks,
        "margin": margins,
    }


def summarize_metric_vectors(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "top1": float(vectors["top1"].mean()),
        "top5": float(vectors["top5"].mean()),
        "mrr": float(vectors["reciprocal_rank"].mean()),
        "mean_margin": float(vectors["margin"].mean()),
        "median_margin": float(np.median(vectors["margin"])),
    }


def permutation_null(
    similarity: np.ndarray,
    ranks: np.ndarray,
    observed: dict[str, float],
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, dict[str, float]]:
    """Random-bijection null for strict retrieval metrics."""
    n = similarity.shape[0]
    rows = np.arange(n)
    top_order = np.argsort(-similarity, axis=1, kind="stable")
    row_best = top_order[:, 0]
    row_second = top_order[:, 1]
    names = ("top1", "top5", "mrr", "mean_margin")
    samples = {name: np.empty(n_permutations, dtype=np.float64) for name in names}

    for b in range(n_permutations):
        matching = rng.permutation(n)
        matched_ranks = ranks[rows, matching]
        matched_scores = similarity[rows, matching]
        best_nonmatch = np.where(
            row_best == matching, row_second, row_best
        )
        samples["top1"][b] = np.mean(matched_ranks <= 1)
        samples["top5"][b] = np.mean(matched_ranks <= min(5, n))
        samples["mrr"][b] = np.mean(1.0 / matched_ranks)
        samples["mean_margin"][b] = np.mean(
            matched_scores - similarity[rows, best_nonmatch]
        )

    result: dict[str, dict[str, float]] = {}
    for name in names:
        null = samples[name]
        result[name] = {
            "null_mean": float(null.mean()),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "p_one_sided_ge": float(
                (1 + np.count_nonzero(null >= observed[name]))
                / (n_permutations + 1)
            ),
        }
    return result


def within_document_metrics(
    similarity: np.ndarray,
    doc_ids: np.ndarray,
) -> tuple[dict[str, float], list[int], list[float]]:
    n = similarity.shape[0]
    truth_ranks: list[int] = []
    reciprocal_ranks: list[float] = []
    for query in range(n):
        candidates = np.flatnonzero(doc_ids == doc_ids[query])
        ordered = candidates[
            np.argsort(-similarity[query, candidates], kind="stable")
        ]
        rank = int(np.flatnonzero(ordered == query)[0]) + 1
        truth_ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
    ranks_array = np.asarray(truth_ranks)
    rr_array = np.asarray(reciprocal_ranks)
    summary = {
        "candidate_count_per_query": float(
            np.mean([np.count_nonzero(doc_ids == doc_ids[i]) for i in range(n)])
        ),
        "top1": float(np.mean(ranks_array == 1)),
        "mrr": float(rr_array.mean()),
    }
    return summary, truth_ranks, reciprocal_ranks


def within_document_permutation_null(
    similarity: np.ndarray,
    doc_ids: np.ndarray,
    observed: dict[str, float],
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, dict[str, float]]:
    """Shuffle pairings independently inside each document."""
    groups = [np.flatnonzero(doc_ids == doc) for doc in np.unique(doc_ids)]
    n = similarity.shape[0]
    samples_top1 = np.empty(n_permutations, dtype=np.float64)
    samples_mrr = np.empty(n_permutations, dtype=np.float64)

    local_rank_lookup = np.empty((n, n), dtype=np.int32)
    local_rank_lookup.fill(-1)
    for group in groups:
        for query in group:
            ordered = group[
                np.argsort(-similarity[query, group], kind="stable")
            ]
            local_rank_lookup[query, ordered] = np.arange(1, len(group) + 1)

    rows = np.arange(n)
    for b in range(n_permutations):
        matching = np.empty(n, dtype=np.int64)
        for group in groups:
            matching[group] = rng.permutation(group)
        matched_ranks = local_rank_lookup[rows, matching]
        samples_top1[b] = np.mean(matched_ranks == 1)
        samples_mrr[b] = np.mean(1.0 / matched_ranks)

    result = {}
    for name, null in (("top1", samples_top1), ("mrr", samples_mrr)):
        result[name] = {
            "null_mean": float(null.mean()),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "p_one_sided_ge": float(
                (1 + np.count_nonzero(null >= observed[name]))
                / (n_permutations + 1)
            ),
        }
    return result


def signflip_p_value(
    differences: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    """One-sided paired randomization test for mean(difference) > 0."""
    differences = np.asarray(differences, dtype=np.float64)
    observed = float(differences.mean())
    if np.allclose(differences, 0.0):
        return 1.0
    samples = np.empty(n_permutations, dtype=np.float64)
    for b in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        samples[b] = float((differences * signs).mean())
    return float(
        (1 + np.count_nonzero(samples >= observed)) / (n_permutations + 1)
    )


def exact_cluster_signflip_p_value(cluster_differences: np.ndarray) -> float:
    """Exact one-sided sign-flip test over the (five) document clusters."""
    cluster_differences = np.asarray(cluster_differences, dtype=np.float64)
    observed = float(cluster_differences.mean())
    k = len(cluster_differences)
    null = []
    for mask in range(1 << k):
        signs = np.asarray(
            [1.0 if (mask >> bit) & 1 else -1.0 for bit in range(k)]
        )
        null.append(float((cluster_differences * signs).mean()))
    null_array = np.asarray(null)
    return float(np.mean(null_array >= observed - 1e-15))


def paired_comparison(
    nla_vectors: dict[str, np.ndarray],
    other_vectors: dict[str, np.ndarray],
    doc_ids: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, dict[str, float]]:
    mapping = {
        "top1": "top1",
        "top5": "top5",
        "mrr": "reciprocal_rank",
        "mean_margin": "margin",
    }
    result = {}
    for output_name, vector_name in mapping.items():
        differences = nla_vectors[vector_name] - other_vectors[vector_name]
        cluster_differences = np.asarray(
            [
                differences[doc_ids == doc].mean()
                for doc in np.unique(doc_ids)
            ]
        )
        result[output_name] = {
            "nla_minus_comparator": float(differences.mean()),
            "query_signflip_p_one_sided": signflip_p_value(
                differences, rng, n_permutations
            ),
            "document_cluster_signflip_p_one_sided": (
                exact_cluster_signflip_p_value(cluster_differences)
            ),
            "document_cluster_differences": cluster_differences.tolist(),
        }
    return result


def load_metadata(path: Path | None, n: int) -> list[dict[str, Any]]:
    if path is None:
        return [
            {"idx": i, "doc_id": i, "position": None, "token": None}
            for i in range(n)
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if len(rows) != n:
        raise ValueError(f"metadata has {len(rows)} rows, expected {n}")
    rows = sorted(rows, key=lambda row: int(row["idx"]))
    if [int(row["idx"]) for row in rows] != list(range(n)):
        raise ValueError("metadata idx values must be exactly 0..n-1")
    return [
        {
            "idx": int(row["idx"]),
            "doc_id": int(row["doc_id"]),
            "position": int(row["position"]),
            "token": row["token"],
        }
        for row in rows
    ]


def round_floats(value: Any, digits: int = 8) -> Any:
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--permutations", type=int, default=50_000)
    args = parser.parse_args()

    if args.permutations < 1_000:
        raise ValueError("--permutations must be at least 1000")

    with np.load(args.vectors, allow_pickle=False) as archive:
        missing = {"x", "m_hat", *METHOD_KEYS.values()} - set(archive.files)
        if missing:
            raise KeyError(f"missing arrays in {args.vectors}: {sorted(missing)}")
        x = np.asarray(archive["x"], dtype=np.float64)
        m_hat = np.asarray(archive["m_hat"], dtype=np.float64)
        predictions = {
            name: np.asarray(archive[key], dtype=np.float64)
            for name, key in METHOD_KEYS.items()
        }

    if x.ndim != 2:
        raise ValueError(f"x must be rank 2, got shape {x.shape}")
    n, d_model = x.shape
    if m_hat.shape != (d_model,):
        raise ValueError(f"m_hat has shape {m_hat.shape}, expected {(d_model,)}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if not np.all(np.isfinite(m_hat)):
        raise ValueError("m_hat contains non-finite values")
    m_norm = float(np.linalg.norm(m_hat))
    if not math.isclose(m_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"m_hat must be unit norm, got {m_norm}")
    m_hat = m_hat / m_norm
    expected_m_hat = x.mean(axis=0)
    expected_m_hat /= np.linalg.norm(expected_m_hat)
    mean_direction_cos = float(expected_m_hat @ m_hat)
    if mean_direction_cos < 1.0 - 1e-5:
        raise ValueError(
            "saved m_hat does not match the normalized mean direction of x: "
            f"cos={mean_direction_cos}"
        )
    for name, prediction in predictions.items():
        if prediction.shape != x.shape:
            raise ValueError(
                f"{name} has shape {prediction.shape}, expected {x.shape}"
            )
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"{name} contains non-finite values")

    metadata = load_metadata(args.metadata, n)
    doc_ids = np.asarray([row["doc_id"] for row in metadata], dtype=np.int64)
    truth = np.arange(n)
    centered_targets = project_and_normalize(x, m_hat)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "B2 centered 40-way activation retrieval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "vectors_path": str(args.vectors),
            "vectors_sha256": sha256_file(args.vectors),
            "metadata_path": str(args.metadata) if args.metadata else None,
            "metadata_sha256": (
                sha256_file(args.metadata) if args.metadata else None
            ),
            "n_queries": n,
            "d_model": d_model,
            "n_documents": int(len(np.unique(doc_ids))),
            "saved_vs_recomputed_mean_direction_cos": mean_direction_cos,
        },
        "protocol": {
            "projection": (
                "project saved dataset mean direction m_hat out of both "
                "prediction and target, then row-wise L2 normalize"
            ),
            "primary_task": (
                "for reconstruction i, rank all x_j and retrieve paired x_i"
            ),
            "margin": "cos(pred_i,x_i) - max_{j!=i} cos(pred_i,x_j)",
            "permutation_null": (
                "random one-to-one query-target bijections; within-document "
                "null independently permutes targets inside each document"
            ),
            "seed": args.seed,
            "n_permutations": args.permutations,
        },
        "analytic_random_baselines": {
            "strict_top1": 1.0 / n,
            "strict_top5": min(5, n) / n,
            "strict_mrr": sum(1.0 / rank for rank in range(1, n + 1)) / n,
            "same_document_top1": float(
                np.mean(
                    [
                        np.count_nonzero(doc_ids == doc_ids[i]) / n
                        for i in range(n)
                    ]
                )
            ),
            "within_document_top1": float(
                np.mean(
                    [
                        1.0 / np.count_nonzero(doc_ids == doc_ids[i])
                        for i in range(n)
                    ]
                )
            ),
        },
        "methods": {},
        "paired_comparisons": {},
    }

    metric_vectors_by_method: dict[str, dict[str, np.ndarray]] = {}
    within_rr_by_method: dict[str, np.ndarray] = {}

    for method_index, (name, prediction) in enumerate(predictions.items()):
        centered_prediction = project_and_normalize(prediction, m_hat)
        similarity = centered_prediction @ centered_targets.T
        ranks = rank_matrix(similarity)
        vectors = metric_vectors(similarity, ranks, truth)
        summary = summarize_metric_vectors(vectors)
        within_summary, within_ranks, within_rr = within_document_metrics(
            similarity, doc_ids
        )
        top_indices = np.argsort(-similarity, axis=1, kind="stable")
        best_indices = top_indices[:, 0]
        best_nonmatch_indices = np.where(
            best_indices == truth, top_indices[:, 1], best_indices
        )

        method_rng = np.random.default_rng(args.seed + method_index * 10_000)
        null = permutation_null(
            similarity,
            ranks,
            summary,
            method_rng,
            args.permutations,
        )
        within_null = within_document_permutation_null(
            similarity,
            doc_ids,
            within_summary,
            method_rng,
            args.permutations,
        )

        rows = []
        for i in range(n):
            rows.append(
                {
                    **metadata[i],
                    "paired_rank": int(ranks[i, i]),
                    "paired_similarity": float(similarity[i, i]),
                    "best_index": int(best_indices[i]),
                    "best_similarity": float(similarity[i, best_indices[i]]),
                    "best_nonmatch_index": int(best_nonmatch_indices[i]),
                    "best_nonmatch_similarity": float(
                        similarity[i, best_nonmatch_indices[i]]
                    ),
                    "margin": float(vectors["margin"][i]),
                    "top5_indices": [
                        int(index) for index in top_indices[i, :5]
                    ],
                    "top1_same_document": bool(
                        doc_ids[best_indices[i]] == doc_ids[i]
                    ),
                    "within_document_rank": int(within_ranks[i]),
                }
            )

        method_payload = {
            "summary": {
                **summary,
                "same_document_top1": float(
                    np.mean(doc_ids[best_indices] == doc_ids)
                ),
                "within_document_top1": within_summary["top1"],
                "within_document_mrr": within_summary["mrr"],
                "mean_paired_similarity": float(np.mean(np.diag(similarity))),
                "mean_best_nonmatch_similarity": float(
                    np.mean(similarity[np.arange(n), best_nonmatch_indices])
                ),
                "positive_margin_fraction": float(
                    np.mean(vectors["margin"] > 0)
                ),
            },
            "permutation_null": null,
            "within_document_permutation_null": within_null,
            "rows": rows,
            "similarity_matrix": similarity.tolist(),
        }
        payload["methods"][name] = method_payload
        metric_vectors_by_method[name] = vectors
        within_rr_by_method[name] = np.asarray(within_rr)

    nla_vectors = metric_vectors_by_method["nla"]
    for comparison_index, comparator in enumerate(
        ("sae_small", "sae_big", "residual_text_control")
    ):
        comparison_rng = np.random.default_rng(
            args.seed + 100_000 + comparison_index * 10_000
        )
        comparison = paired_comparison(
            nla_vectors,
            metric_vectors_by_method[comparator],
            doc_ids,
            comparison_rng,
            args.permutations,
        )
        within_diff = (
            within_rr_by_method["nla"] - within_rr_by_method[comparator]
        )
        within_cluster_diff = np.asarray(
            [
                within_diff[doc_ids == doc].mean()
                for doc in np.unique(doc_ids)
            ]
        )
        comparison["within_document_mrr"] = {
            "nla_minus_comparator": float(within_diff.mean()),
            "query_signflip_p_one_sided": signflip_p_value(
                within_diff, comparison_rng, args.permutations
            ),
            "document_cluster_signflip_p_one_sided": (
                exact_cluster_signflip_p_value(within_cluster_diff)
            ),
            "document_cluster_differences": within_cluster_diff.tolist(),
        }
        payload["paired_comparisons"][f"nla_vs_{comparator}"] = comparison

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(round_floats(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("B2_RETRIEVAL_COMPLETE")
    for name, result in payload["methods"].items():
        summary = result["summary"]
        print(
            f"{name}: top1={summary['top1']:.3f} "
            f"top5={summary['top5']:.3f} mrr={summary['mrr']:.3f} "
            f"margin={summary['mean_margin']:.4f} "
            f"within_doc_top1={summary['within_document_top1']:.3f}"
        )
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
