#!/usr/bin/env python3
"""Create the v3r2 audit addendum by rebinding the frozen v3 checks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


FAILED_DRAFT_SHA256 = (
    "061903133b748ccffe2f85f697c5e6a7d53fd631e63fb1c0302ae42fbd59e6d5"
)
FAILED_AUDIT_SHA256 = (
    "5dcc267ad9f6bf9765bbb7c6ab3965498c275c7df3165780f5f1d79301c2a396"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-addendum", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--failed-anchor-audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for path in (
        args.base_addendum,
        args.anchors,
        args.failed_anchor_audit,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite addendum: {args.out}")

    base = read_json(args.base_addendum)
    anchors = read_json(args.anchors)
    failed_audit = read_json(args.failed_anchor_audit)
    if (
        base.get("status") != "frozen_before_v3_generation"
        or base.get("scenario_anchors", {}).get("sha256")
        != FAILED_DRAFT_SHA256
    ):
        raise ValueError("base addendum is not the failed v3-draft addendum")
    if (
        anchors.get("status") != "frozen_before_v3r2_generation"
        or len(anchors.get("concepts", [])) != 24
    ):
        raise ValueError("anchors are not the frozen v3r2 asset")
    sources = anchors.get("sources", {})
    if (
        sources.get("scenario_anchors_v3", {}).get("sha256")
        != FAILED_DRAFT_SHA256
        or sources.get("scenario_anchor_audit_v3", {}).get("sha256")
        != FAILED_AUDIT_SHA256
    ):
        raise ValueError("v3r2 anchors do not bind the failed draft and audit")
    if (
        str(failed_audit.get("status", "")).upper() != "FAIL"
        or sha256_file(args.failed_anchor_audit) != FAILED_AUDIT_SHA256
    ):
        raise ValueError("supplied failed anchor audit is not frozen evidence")

    payload = copy.deepcopy(base)
    payload["status"] = "frozen_before_v3r2_generation"
    payload["purpose"] = (
        "Apply the already frozen v3 semantic checks to the final v3r2 "
        "scenario-anchor asset before any generated v3 text exists."
    )
    payload["scenario_anchors"] = {
        "path": "server/c1_confirmatory_scenario_anchors_v3r2.json",
        "sha256": sha256_file(args.anchors),
        "status": "frozen_before_v3r2_generation",
    }
    payload["revision"] = {
        "type": "final pre-text anchor-design iteration",
        "failed_anchor_draft_sha256": FAILED_DRAFT_SHA256,
        "failed_anchor_audit_sha256": FAILED_AUDIT_SHA256,
        "generated_v3_text_existed_before_revision": False,
        "further_anchor_iteration_if_v3r2_fails": False,
        "all_document_batch_and_pair_checks_unchanged": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"C1_V3R2_AUDIT_ADDENDUM_FROZEN out={args.out} "
        f"sha256={sha256_file(args.out)}"
    )


if __name__ == "__main__":
    main()
