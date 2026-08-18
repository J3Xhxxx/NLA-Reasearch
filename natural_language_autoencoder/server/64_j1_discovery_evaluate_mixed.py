#!/usr/bin/env python3
"""Run the frozen 1,800-score Terra evaluation for mixed-labeler J1.

The public evaluator prompt remains the original 5-hypothesis by 8-context
blind contract.  Labeler/batch/worker provenance is added only to the private
condition map and deblinded raw score rows, never to Terra's prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _load_base() -> Any:
    path = Path(__file__).with_name("59_j1_discovery_evaluate.py")
    spec = importlib.util.spec_from_file_location("j1_eval_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen evaluator base: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
EXPECTED_MIXED_RESULT_SHA256 = (
    "2ca779f8ffb89d93531fef31beb12a5d81b0185d18d7d02e6450c296ce562b8b"
)
EXPECTED_MIXED_CHECKPOINT_SHA256 = (
    "6bccf09a08ebe69dfb263c3cc38a6bf71c06bb36d88ff6db64f8ab334737d311"
)
EXPECTED_ANALYSIS_PLAN_SHA256 = (
    "772042a159188d7777f944b59f152b2484cae719cfbfd72190921d91f1aa7147"
)
EXPECTED_CLI_AMENDMENT_SHA256 = (
    "7c7babec80ba123f640a82ca1b4b5d51648d28f3abd3c6a03314092880314114"
)
EXPECTED_LUNA_COMPLETION_SHA256 = (
    "b8653f78accc76b8a3b61a460c812df73cf76ab0111bc202da6bdc1d030081e3"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_exact(path: Path, expected: str) -> str:
    actual = BASE.verify_sidecar(path, required=True)
    if actual != expected:
        raise ValueError(f"unexpected SHA-256 for {path}: {actual} != {expected}")
    return actual


def _label_provenance(task: Mapping[str, Any]) -> dict[str, Any]:
    row = task.get("result_row")
    if not isinstance(row, dict):
        raise ValueError("mixed label task lacks parsed result row")
    batch = int(task.get("batch_id", row.get("batch_id", -1)))
    labeler = row.get("labeler")
    expected_labeler = "fable" if batch < 13 else "luna"
    if labeler != expected_labeler:
        raise ValueError(
            f"labeler/batch mismatch batch={batch}: {labeler!r} != {expected_labeler!r}"
        )
    agent_task = row.get("agent_task")
    role = row.get("role")
    model = row.get("model")
    reasoning = row.get("reasoning")
    if not all(isinstance(x, str) and x for x in (agent_task, role, model, reasoning)):
        raise ValueError(f"incomplete label provenance for batch {batch}")
    provenance = row.get("provenance")
    source = dict(provenance) if isinstance(provenance, dict) else {}
    transport: str
    if batch < 13:
        transport = "claude_code_fable"
    elif batch < 27:
        transport = "collaboration_luna_worker"
    else:
        transport = "codex_cli_ephemeral"
        source_worker = source.get("source_worker_provenance")
        if isinstance(source_worker, dict):
            observed = source_worker.get("transport")
            if observed not in (None, transport):
                raise ValueError(f"unexpected CLI transport batch {batch}: {observed}")
    return {
        "labeler": labeler,
        "label_batch_id": batch,
        "label_agent_task": agent_task,
        "label_role": role,
        "label_model": model,
        "label_reasoning": reasoning,
        "label_transport": transport,
        "label_provenance": source,
    }


def validate_labeler_balance(
    labels: Mapping[tuple[int, str], Mapping[str, Any]]
) -> dict[str, Any]:
    counts = {
        arm: {"fable": 0, "luna": 0}
        for arm in BASE.ARMS
    }
    task_by_batch: dict[int, str] = {}
    batch_by_task: dict[str, int] = {}
    for (feature, arm), task in labels.items():
        if arm not in counts:
            raise ValueError(f"unexpected arm {arm}")
        provenance = _label_provenance(task)
        counts[arm][provenance["labeler"]] += 1
        task_id = provenance["label_agent_task"]
        batch = provenance["label_batch_id"]
        if batch in task_by_batch and task_by_batch[batch] != task_id:
            raise ValueError(f"batch {batch} has multiple agent tasks")
        if task_id in batch_by_task and batch_by_task[task_id] != batch:
            raise ValueError(f"agent task is reused across batches: {task_id}")
        task_by_batch[batch] = task_id
        batch_by_task[task_id] = batch
    expected = {arm: {"fable": 13, "luna": 32} for arm in BASE.ARMS}
    if counts != expected:
        raise ValueError(f"mixed labeler arm counts differ: {counts}")
    return {"by_arm": counts, "total": {"fable": 65, "luna": 160}}


def build_mixed_eval_job(
    feature_rows: Sequence[dict[str, Any]],
    labels: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    freeze_sha: str,
    av_sha: str,
    label_jobs_sha: str,
    label_result_sha: str,
    label_checkpoint_sha: str,
    analysis_plan_sha: str,
    cli_amendment_sha: str,
    script_sha: str,
) -> tuple[dict[str, Any], dict[int, str]]:
    rng = random.Random(BASE.SEED)
    used_context_ids: set[str] = set()
    used_case_ids: set[str] = set()
    public_features: list[dict[str, Any]] = []
    private_contexts: list[dict[str, Any]] = []
    private_hypothesis_batches: list[dict[str, Any]] = []
    prompts: dict[int, str] = {}
    for feature_index, feature_row in enumerate(
        sorted(feature_rows, key=lambda row: int(row["feature"]))
    ):
        feature = int(feature_row["feature"])
        context_specs: list[dict[str, Any]] = []
        for item in feature_row["contexts"]:
            context_id = BASE._opaque_id(rng, "ctx", used_context_ids)
            context_specs.append(
                {"context_id": context_id, "marked_context": item["marked_text"]}
            )
            private_contexts.append(
                {
                    "context_id": context_id,
                    "feature": feature,
                    "truth": int(item["truth"]),
                    "role": item["role"],
                    "doc_id": int(item["row"]["doc_id"]),
                    "position": int(item["row"]["position"]),
                    "stratum": feature_row["stratum"],
                }
            )
        rng.shuffle(context_specs)
        hypothesis_specs: list[dict[str, Any]] = []
        private_hypotheses: list[dict[str, Any]] = []
        for arm in BASE.ARMS:
            task = labels[(feature, arm)]
            case_id = BASE._opaque_id(rng, "case", used_case_ids)
            hypothesis = str(task["hypothesis"])
            hypothesis_specs.append({"case_id": case_id, "hypothesis": hypothesis})
            private_hypotheses.append(
                {
                    "case_id": case_id,
                    "feature": feature,
                    "arm": arm,
                    "hypothesis": hypothesis,
                    "source_case_id": task.get("case_id"),
                    **_label_provenance(task),
                }
            )
        rng.shuffle(hypothesis_specs)
        public = {
            "feature_key": f"feature_batch_{feature_index + 1:02d}",
            "hypotheses": hypothesis_specs,
            "contexts": context_specs,
        }
        public_features.append(public)
        private_hypothesis_batches.append(
            {
                "feature": feature,
                "feature_key": public["feature_key"],
                "hypotheses": private_hypotheses,
            }
        )
        prompts[feature] = BASE.build_prompt(public)
    job = {
        "schema_version": 2,
        "experiment": "J1 exploratory mixed-labeler blinded held-out evaluator",
        "status": "EXPLORATORY_BLINDED_EVAL_JOB_MIXED_FROZEN",
        "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "seed": BASE.SEED,
        "evaluator_model": BASE.EVALUATOR_MODEL,
        "inputs": {
            "freeze_sha256": freeze_sha,
            "av_result_sha256": av_sha,
            "label_job_freeze_sha256": label_jobs_sha,
            "label_job_result_sha256": label_result_sha,
            "label_job_checkpoint_sha256": label_checkpoint_sha,
            "mixed_analysis_plan_sha256": analysis_plan_sha,
            "cli_fallback_amendment_sha256": cli_amendment_sha,
            "luna_completion_protocol_sha256": EXPECTED_LUNA_COMPLETION_SHA256,
            "script_sha256": script_sha,
            "base_evaluator_script_sha256": sha256_file(
                Path(__file__).with_name("59_j1_discovery_evaluate.py")
            ),
        },
        "contract": {
            "n_features": BASE.N_FEATURES,
            "contexts_per_feature": BASE.N_CONTEXTS,
            "hypotheses_per_feature": len(BASE.ARMS),
            "scores_per_feature": BASE.N_CONTEXTS * len(BASE.ARMS),
            "total_scores": BASE.N_FEATURES * len(BASE.ARMS) * BASE.N_CONTEXTS,
            "arms": list(BASE.ARMS),
            "truth_source": (
                "freeze_only_actual_SAE_activation_and_embedded_hard_negative"
            ),
            "condition_map_not_in_prompt": True,
            "labeler_provenance_not_in_prompt": True,
            "labeler_provenance_on_every_deblinded_score": True,
        },
        "public_batches": public_features,
        "condition_map_private": {
            "contexts": private_contexts,
            "hypotheses": private_hypothesis_batches,
        },
        "prompts": {
            str(feature): {
                "sha256": BASE.sha256_bytes(prompt.encode("utf-8")),
                "bytes": len(prompt.encode("utf-8")),
            }
            for feature, prompt in sorted(prompts.items())
        },
    }
    job["prompt_contract_sha256"] = BASE.canonical_sha(job["prompts"])
    return job, prompts


def score_records(
    latest: Mapping[int, Mapping[str, Any]], job: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth: dict[str, int] = {}
    contexts: dict[str, dict[str, Any]] = {}
    hypotheses: dict[str, dict[str, Any]] = {}
    for item in job["condition_map_private"]["contexts"]:
        context_id = str(item["context_id"])
        truth[context_id] = int(item["truth"])
        contexts[context_id] = dict(item)
    for batch in job["condition_map_private"]["hypotheses"]:
        for item in batch["hypotheses"]:
            hypotheses[str(item["case_id"])] = dict(item)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for feature, row in sorted(latest.items()):
        if row.get("ok") is not True:
            failures.append(
                {
                    "feature": feature,
                    "failure": row.get("failure", "unknown"),
                    "attempts": row.get("attempts", []),
                    "raw_final_json": row.get("raw_final_json", ""),
                }
            )
            continue
        for score in row.get("scores", []):
            case_id = str(score["case_id"])
            context_id = str(score["context_id"])
            if case_id not in hypotheses or context_id not in truth:
                raise ValueError("Terra checkpoint references unknown blinded ID")
            hypothesis = hypotheses[case_id]
            if int(hypothesis["feature"]) != feature:
                raise ValueError("Terra score hypothesis belongs to another feature")
            context = contexts[context_id]
            scores.append(
                {
                    **score,
                    "feature": feature,
                    "arm": hypothesis["arm"],
                    "context_id": context_id,
                    "truth": truth[context_id],
                    "stratum": context["stratum"],
                    "role": context["role"],
                    "labeler": hypothesis["labeler"],
                    "label_batch_id": hypothesis["label_batch_id"],
                    "label_agent_task": hypothesis["label_agent_task"],
                    "label_role": hypothesis["label_role"],
                    "label_model": hypothesis["label_model"],
                    "label_reasoning": hypothesis["label_reasoning"],
                    "label_transport": hypothesis["label_transport"],
                    "source_case_id": hypothesis.get("source_case_id"),
                }
            )
    return scores, failures


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    results = Path(__file__).resolve().parents[1] / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", type=Path, default=results / "j1_discovery_freeze_v1.json"
    )
    parser.add_argument(
        "--av-result", type=Path, default=results / "j1_discovery_result_v1.json"
    )
    parser.add_argument(
        "--label-jobs",
        type=Path,
        default=results / "j1_discovery_labels_jobs_v1.json",
    )
    parser.add_argument(
        "--label-result",
        type=Path,
        default=results / "j1_discovery_labels_mixed_result_v3.json",
    )
    parser.add_argument(
        "--label-checkpoint",
        type=Path,
        default=results / "j1_discovery_labels_checkpoint_mixed_v3.jsonl",
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=results / "J1_DISCOVERY_MIXED_ANALYSIS_PLAN_2026-08-06.md",
    )
    parser.add_argument(
        "--cli-amendment",
        type=Path,
        default=results / "J1_LUNA_CLI_FALLBACK_AMENDMENT_2026-08-06.md",
    )
    parser.add_argument(
        "--out-job", type=Path, default=results / "j1_blinded_eval_job_mixed_v2.json"
    )
    parser.add_argument(
        "--out-checkpoint",
        type=Path,
        default=results / "j1_blinded_eval_checkpoint_mixed_v2.jsonl",
    )
    parser.add_argument(
        "--out-result",
        type=Path,
        default=results / "j1_blinded_eval_result_mixed_v2.json",
    )
    parser.add_argument("--codex-command", default="codex.cmd")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1 or args.retries < 0 or args.timeout <= 0:
        raise ValueError("concurrency >=1, retries >=0, timeout >0 required")
    freeze_sha = BASE.verify_sidecar(args.freeze, required=True)
    av_sha = BASE.verify_sidecar(args.av_result, required=True)
    jobs_sha = BASE.verify_sidecar(args.label_jobs, required=True)
    result_sha = verify_exact(args.label_result, EXPECTED_MIXED_RESULT_SHA256)
    checkpoint_sha = verify_exact(
        args.label_checkpoint, EXPECTED_MIXED_CHECKPOINT_SHA256
    )
    analysis_plan_sha = verify_exact(
        args.analysis_plan, EXPECTED_ANALYSIS_PLAN_SHA256
    )
    cli_amendment_sha = verify_exact(
        args.cli_amendment, EXPECTED_CLI_AMENDMENT_SHA256
    )
    freeze = BASE.load_json(args.freeze)
    av = BASE.load_json(args.av_result)
    label_jobs = BASE.load_json(args.label_jobs)
    label_result = BASE.load_json(args.label_result)
    checkpoint_rows = BASE.load_jsonl(args.label_checkpoint)
    if not all(
        isinstance(item, dict)
        for item in (freeze, av, label_jobs, label_result)
    ):
        raise ValueError("all evaluator JSON inputs must be objects")
    feature_rows, _, freeze_risks = BASE.validate_freeze(freeze, freeze_sha)
    BASE.validate_av(av, freeze_sha, feature_rows)
    labels = BASE.validate_labels(
        label_jobs,
        label_result,
        checkpoint_rows,
        freeze_sha,
        av_sha,
        feature_rows,
        label_freeze_sha=jobs_sha,
    )
    labeler_counts = validate_labeler_balance(labels)
    script_sha = sha256_file(Path(__file__))
    job, prompts = build_mixed_eval_job(
        feature_rows,
        labels,
        freeze_sha=freeze_sha,
        av_sha=av_sha,
        label_jobs_sha=jobs_sha,
        label_result_sha=result_sha,
        label_checkpoint_sha=checkpoint_sha,
        analysis_plan_sha=analysis_plan_sha,
        cli_amendment_sha=cli_amendment_sha,
        script_sha=script_sha,
    )
    job["labeler_counts"] = labeler_counts
    job["freeze_risks"] = freeze_risks
    job_sha = BASE.write_immutable(args.out_job, job)
    BASE.write_sidecar(Path(str(args.out_job) + ".sha256"), job_sha)
    print(
        f"mixed Terra job frozen: {args.out_job} sha256={job_sha} "
        f"scores={job['contract']['total_scores']}",
        flush=True,
    )
    if args.dry_run:
        print("dry-run: no Terra calls", flush=True)
        return 0
    if args.out_result.exists():
        existing_sha = BASE.verify_sidecar(args.out_result, required=True)
        existing = BASE.load_json(args.out_result)
        if (
            not isinstance(existing, dict)
            or existing.get("job_sha256") != job_sha
            or existing.get("status") != "EXPLORATORY_BLINDED_EVAL_MIXED_COMPLETE"
        ):
            raise ValueError("existing mixed Terra result is incompatible/incomplete")
        print(f"existing complete Terra result verified sha256={existing_sha}")
        return 0
    resolved = BASE.resolve_codex_command(args.codex_command)
    latest, history, evaluator = BASE.evaluate_all(
        job,
        prompts,
        job_sha,
        args.out_checkpoint,
        resolved_command=resolved,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout=args.timeout,
        dry_run=False,
    )
    scores, failures = score_records(latest, job)
    expected_scores = BASE.N_FEATURES * len(BASE.ARMS) * BASE.N_CONTEXTS
    status = (
        "EXPLORATORY_BLINDED_EVAL_MIXED_COMPLETE"
        if not failures and len(scores) == expected_scores
        else "NO_COMPLETE_ANALYSIS"
    )
    result = {
        "schema_version": 2,
        "experiment": "J1 exploratory mixed-labeler blinded held-out evaluator",
        "status": status,
        "confirmatory": False,
        "claim_scope": "discovery_only_no_confirmatory_inference",
        "job_sha256": job_sha,
        "freeze_sha256": freeze_sha,
        "av_result_sha256": av_sha,
        "label_job_freeze_sha256": jobs_sha,
        "label_job_result_sha256": result_sha,
        "label_job_checkpoint_sha256": checkpoint_sha,
        "mixed_analysis_plan_sha256": analysis_plan_sha,
        "cli_fallback_amendment_sha256": cli_amendment_sha,
        "evaluator": evaluator,
        "checkpoint": {
            "path": str(args.out_checkpoint),
            "history_rows": len(history),
        },
        "call_records": {
            str(feature): row for feature, row in sorted(latest.items())
        },
        "scores": scores,
        "failures": failures,
        "checks": {
            "expected_features": BASE.N_FEATURES,
            "expected_arms": len(BASE.ARMS),
            "expected_scores": expected_scores,
            "recorded_scores": len(scores),
            "recorded_failures": len(failures),
            "every_score_has_labeler": all(
                score.get("labeler") in ("fable", "luna") for score in scores
            ),
            "labeler_counts": labeler_counts,
        },
        "raw_final_json_retained_in_checkpoint": True,
    }
    if args.out_checkpoint.is_file():
        checkpoint_output_sha = sha256_file(args.out_checkpoint)
        BASE.write_sidecar(
            Path(str(args.out_checkpoint) + ".sha256"), checkpoint_output_sha
        )
        result["checkpoint"]["sha256"] = checkpoint_output_sha
    result_output_sha = BASE.write_immutable(args.out_result, result)
    BASE.write_sidecar(Path(str(args.out_result) + ".sha256"), result_output_sha)
    print(
        f"Terra status={status} scores={len(scores)} failures={len(failures)} "
        f"result_sha256={result_output_sha}",
        flush=True,
    )
    return 0 if status == "EXPLORATORY_BLINDED_EVAL_MIXED_COMPLETE" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"64_j1_discovery_evaluate_mixed: FAIL CLOSED: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
