#!/usr/bin/env python3
"""Render J2-P0 case-study material after the metric shortlist is frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def write_frozen(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"refusing to overwrite non-identical output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise SystemExit(f"missing sidecar: {sidecar}")
    actual = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").split()[0] != actual:
        raise SystemExit(f"sidecar mismatch: {path}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--j2-explanations", required=True, type=Path)
    parser.add_argument("--j2-result", required=True, type=Path)
    parser.add_argument("--n4-causal", required=True, type=Path)
    parser.add_argument("--j2-causal", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    input_paths = {
        "shortlist": args.shortlist,
        "analysis": args.analysis,
        "protocol": args.protocol,
        "j2_explanations": args.j2_explanations,
        "j2_result": args.j2_result,
        "n4_causal": args.n4_causal,
        "j2_causal": args.j2_causal,
    }
    input_hashes = {
        label: verify_sidecar(path) for label, path in input_paths.items()
    }
    shortlist = json.loads(args.shortlist.read_text(encoding="utf-8"))
    if shortlist.get("status") != "FROZEN_BEFORE_HUMAN_CASE_READING":
        raise SystemExit("case shortlist is not frozen")
    if (
        shortlist.get("selection_contract")
        != "nine protocol-defined categories x two SAE operating points x top3; "
        "overlap allowed; ties broken by idx"
    ):
        raise SystemExit("case shortlist selection contract mismatch")
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis.get("status") != "EXPLORATORY_ANALYSIS_COMPLETE":
        raise SystemExit("J2 analysis is not complete")
    if (
        analysis.get("case_shortlist", {}).get("sha256")
        != input_hashes["shortlist"]
        or analysis.get("integrity", {}).get("case_shortlist_sha256")
        != input_hashes["shortlist"]
    ):
        raise SystemExit("analysis does not bind the supplied shortlist")
    if analysis.get("inputs", {}).get("protocol") != input_hashes["protocol"]:
        raise SystemExit("analysis/protocol binding mismatch")
    analysis_bindings = {
        "j2_explanations": "j2_explanations",
        "j2_result": "j2_result",
        "n4_causal": "n4_causal",
        "j2_causal": "j2_causal",
    }
    for analysis_key, local_key in analysis_bindings.items():
        if (
            analysis.get("inputs", {}).get(analysis_key)
            != input_hashes[local_key]
        ):
            raise SystemExit(
                f"analysis input binding mismatch: {analysis_key}"
            )
    if shortlist.get("inputs") != analysis.get("inputs"):
        raise SystemExit("shortlist and analysis input manifests differ")
    explanations = json.loads(
        args.j2_explanations.read_text(encoding="utf-8")
    )
    result = json.loads(args.j2_result.read_text(encoding="utf-8"))
    n4_causal = json.loads(args.n4_causal.read_text(encoding="utf-8"))
    j2_causal = json.loads(args.j2_causal.read_text(encoding="utf-8"))

    artifact_rows = {
        "explanations": explanations["rows"],
        "result": result["rows"],
        "n4_causal": n4_causal["rows"],
        "j2_causal": j2_causal["rows"],
    }
    for name, source_rows in artifact_rows.items():
        indices = [int(row["idx"]) for row in source_rows]
        if len(source_rows) != 200 or indices != list(range(200)):
            raise SystemExit(
                f"{name} rows must be ordered, unique idx 0..199"
            )
    explanations_by_idx = {
        int(row["idx"]): row for row in artifact_rows["explanations"]
    }
    result_by_idx = {
        int(row["idx"]): row for row in artifact_rows["result"]
    }
    n4_causal_by_idx = {
        int(row["idx"]): row for row in artifact_rows["n4_causal"]
    }
    j2_causal_by_idx = {
        int(row["idx"]): row for row in artifact_rows["j2_causal"]
    }
    memberships: dict[int, list[dict[str, Any]]] = {}
    for category, cases in shortlist["categories"].items():
        for case in cases:
            memberships.setdefault(int(case["idx"]), []).append(
                {
                    "category": category,
                    "rank": int(case["rank"]),
                    "condition": case["condition"],
                    "selection_metrics": {
                        key: value
                        for key, value in case.items()
                        if key
                        not in {"rank", "idx", "doc_id", "condition"}
                    },
                }
            )

    expected_indices = set(shortlist.get("unique_indices", []))
    if set(memberships) != expected_indices:
        raise SystemExit("shortlist unique-index manifest mismatch")
    rows = []
    for idx in sorted(memberships):
        explanation = explanations_by_idx[idx]
        audit = result_by_idx[idx]
        old_causal = n4_causal_by_idx[idx]
        new_causal = j2_causal_by_idx[idx]
        if (
            int(explanation["doc_id"]) != int(audit["doc_id"])
            or int(explanation["position"]) != int(audit["position"])
            or int(explanation["doc_id"]) != int(old_causal["doc_id"])
            or int(explanation["position"]) != int(old_causal["position"])
            or int(explanation["doc_id"]) != int(new_causal["doc_id"])
            or int(explanation["position"]) != int(new_causal["position"])
            or explanation["token"] != audit["token"]
            or explanation["token"] != old_causal["token"]
            or explanation["token"] != new_causal["token"]
        ):
            raise SystemExit(f"case artifact metadata mismatch at idx {idx}")
        rows.append(
            {
                "idx": idx,
                "doc_id": int(explanation["doc_id"]),
                "position": int(explanation["position"]),
                "token": explanation["token"],
                "corpus": explanation["corpus"],
                "source": explanation["source"],
                "lang": explanation["lang"],
                "context_tail": explanation["context_tail"],
                "continuation": explanation["continuation"],
                "memberships": memberships[idx],
                "explanations": {
                    "direct_n4": explanation["direct_n4"],
                    "sae_small": explanation["sae_small"],
                    "sae_big": explanation["sae_big"],
                },
                "fixed_point": audit["fixed_point"],
                "causal_kl_at_pos": {
                    "nla_direct": old_causal["results"]["orig"]["kl_at_pos"],
                    "sae_small": old_causal["results"]["sae_small"]["kl_at_pos"],
                    "small_loop": new_causal["results"]["small_loop"]["kl_at_pos"],
                    "direct_small": new_causal["results"]["direct_small"][
                        "kl_at_pos"
                    ],
                    "sae_big": old_causal["results"]["sae_big"]["kl_at_pos"],
                    "big_loop": new_causal["results"]["big_loop"]["kl_at_pos"],
                    "direct_big": new_causal["results"]["direct_big"][
                        "kl_at_pos"
                    ],
                    "zero": old_causal["results"]["zero"]["kl_at_pos"],
                },
            }
        )

    payload = {
        "schema_version": 1,
        "experiment": "J2-P0 SAE projection language-loop case-study bundle",
        "status": "POST_SHORTLIST_CASE_MATERIAL",
        "confirmatory": False,
        "claim_scope": "post_hoc_mechanism_hypothesis_generation_only",
        "inputs": {
            **{f"{key}_sha256": value for key, value in input_hashes.items()},
            "script_sha256": sha256_file(__file__),
        },
        "rows": rows,
    }
    json_sha = write_frozen(args.out, canonical_bytes(payload))

    lines = [
        "# J2-P0 case-study bundle",
        "",
        "> Post-shortlist, exploratory material only. These cases generate "
        "mechanism hypotheses; they do not estimate population effects.",
        "",
    ]
    for row in rows:
        categories = ", ".join(
            f"`{item['category']}`#{item['rank']}"
            for item in row["memberships"]
        )
        lines += [
            f"## idx {row['idx']} — token `{row['token']}`",
            "",
            f"- doc/source/lang: `{row['doc_id']}` / `{row['source']}` / `{row['lang']}`",
            f"- selected by: {categories}",
            f"- KL: direct `{row['causal_kl_at_pos']['nla_direct']:.6g}`, "
            f"SAE-small `{row['causal_kl_at_pos']['sae_small']:.6g}`, "
            f"small-loop `{row['causal_kl_at_pos']['small_loop']:.6g}`, "
            f"SAE(direct)-small "
            f"`{row['causal_kl_at_pos']['direct_small']:.6g}`, "
            f"SAE-big `{row['causal_kl_at_pos']['sae_big']:.6g}`, "
            f"big-loop `{row['causal_kl_at_pos']['big_loop']:.6g}`, "
            f"SAE(direct)-big "
            f"`{row['causal_kl_at_pos']['direct_big']:.6g}`",
            "",
            "### Frozen context",
            "",
            "```text",
            str(row["context_tail"]),
            "```",
            "",
            "### Direct AV(x)",
            "",
            str(row["explanations"]["direct_n4"]),
            "",
            "### AV(SAE-small(x))",
            "",
            str(row["explanations"]["sae_small"]),
            "",
            "### AV(SAE-big(x))",
            "",
            str(row["explanations"]["sae_big"]),
            "",
        ]
    markdown_sha = write_frozen(
        args.markdown, ("\n".join(lines) + "\n").encode("utf-8")
    )
    print(
        f"J2_CASE_BUNDLE_COMPLETE json={json_sha} markdown={markdown_sha} "
        f"cases={len(rows)}"
    )


if __name__ == "__main__":
    main()
