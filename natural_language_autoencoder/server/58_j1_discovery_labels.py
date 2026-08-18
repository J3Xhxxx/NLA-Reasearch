"""Cross-platform CPU/API runner for exploratory J1 discovery labels.

This module consumes the immutable J1 discovery freeze/result emitted by
``57_j1_discovery_pilot_gpu.py`` and sends one five-case labelling request per
feature to an installed Claude Code CLI.  It deliberately performs no
scientific analysis: the output is an auditable, resumable collection of
opaque case labels only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEED = 20260806
N_FEATURES = 45
N_DISCOVERY = 4
N_ROWS = N_FEATURES * N_DISCOVERY * 2
CONDITIONS = ("SAE_CONTEXT", "NLA_ASSISTED", "NLA_CONTRASTIVE", "NLA_MISMATCHED", "NLA_ONLY")
# Claude Code does not expose a provider-independent max-output-token switch;
# this protocol therefore freezes the same textual budget in every request and
# records it with the immutable job freeze.
MAX_OUTPUT_TOKENS = 768
MAX_RETRIES = 3
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(_canonical_bytes(value))


def _pretty_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    data = _pretty_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise RuntimeError(f"refusing to overwrite non-identical immutable artifact: {path}")
    else:
        with path.open("xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    return sha256_bytes(data)


def _immutable_sidecar(path: Path, digest: str) -> None:
    data = (digest + "\n").encode("ascii")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"sha256 sidecar mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _append_fsync(path: Path, row: Mapping[str, Any]) -> None:
    data = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _snippet_text(row: Mapping[str, Any]) -> str:
    """Resolve the actual AV/NLA snippet field without exposing metadata."""
    return _text(row, "explanation", "snippet", "text")


def _sha256_hex(value: Any) -> bool:
    """Return whether *value* is a conventional lowercase SHA-256 digest."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _finite_number(value: Any) -> bool:
    """Return whether *value* is a finite JSON-compatible number."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _usage_and_models(envelope: Mapping[str, Any], parsed: Mapping[str, Any]) -> tuple[dict[str, Any] | None,
                                                                                     dict[str, Any] | None,
                                                                                     list[str]]:
    """Extract usage provenance from a Claude envelope.

    Claude Code has emitted both ``usage`` and ``modelUsage`` over time.  The
    exact nested shape is provider-specific, so retain the objects verbatim
    and derive a conservative list of model names from explicit model fields
    and modelUsage mapping keys.
    """
    model_usage = envelope.get("modelUsage")
    if not isinstance(model_usage, dict):
        model_usage = parsed.get("modelUsage") if isinstance(parsed, dict) else None
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
    names: set[str] = set()

    def visit(value: Any, *, top_level: bool = False) -> None:
        if not isinstance(value, dict):
            return
        for key in ("model", "model_name", "modelName", "resolved_model", "resolvedModel"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                names.add(candidate.strip())
        for key, child in value.items():
            # modelUsage is commonly keyed by the resolved model name.  Keep
            # primitive-valued keys too: synthetic/older CLIs sometimes emit
            # token totals directly instead of a nested object.
            if top_level and isinstance(key, str) and key.strip() and key not in {
                "usage", "modelUsage", "input", "output", "tokens", "cache",
                "input_tokens", "output_tokens", "total_tokens",
                "inputTokens", "outputTokens", "totalTokens", "cacheReadInputTokens",
                "cacheCreationInputTokens",
            } and isinstance(value, dict) and value is model_usage:
                names.add(key.strip())
            if isinstance(child, dict):
                visit(child)

    visit(model_usage, top_level=True)
    visit(usage, top_level=True)
    return (model_usage if isinstance(model_usage, dict) else None,
            usage if isinstance(usage, dict) else None,
            sorted(names))


def _cost_from_envelope(envelope: Mapping[str, Any], parsed: Mapping[str, Any]) -> Any:
    for key in ("total_cost_usd", "cost", "totalCostUsd"):
        value = envelope.get(key)
        if _finite_number(value):
            return value
    if isinstance(parsed, dict):
        for key in ("total_cost_usd", "cost", "totalCostUsd"):
            value = parsed.get(key)
            if _finite_number(value):
                return value
    return None


def _verify_input_sidecar(path: Path) -> str:
    """Require and verify the immutable SHA-256 sidecar for an input."""
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing SHA-256 sidecar for {path}: {sidecar}")
    try:
        recorded = sidecar.read_text(encoding="ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"non-ASCII SHA-256 sidecar: {sidecar}") from exc
    # Accept the digest-only sidecars emitted by 57 and the conventional
    # ``sha256sum`` form (``<digest>  <basename>``) used by the frozen label
    # protocol addendum.  In the latter form bind the filename as well.
    fields = recorded.split()
    recorded_digest = fields[0] if fields else ""
    if len(fields) > 1 and fields[-1] != path.name:
        raise ValueError(f"SHA-256 sidecar filename mismatch for {path}")
    if recorded_digest != digest or len(fields) not in (1, 2):
        raise ValueError(f"SHA-256 sidecar mismatch for {path}")
    return digest


def _marked_context(context: Mapping[str, Any], fallback: str) -> str:
    """Render one discovery window with its active token explicitly marked.

    The freeze stores the tokenizer-decoded ``before``/``token``/``after``
    pieces.  If an older freeze omitted one of those fields, retain the exact
    result context text instead of fabricating a token boundary.
    """
    before = context.get("before")
    token = context.get("token")
    after = context.get("after")
    if all(isinstance(x, str) for x in (before, token, after)):
        return f"{before}<<<ACTIVE>>>{token}<<<END_ACTIVE>>>{after}"
    return fallback


def _extract_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise ValueError("J1 result lacks rows list")
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"result row {i} is not an object")
        out.append(dict(raw))
    return out


def _validate_freeze(freeze: Mapping[str, Any], freeze_sha: str) -> None:
    if freeze.get("status") != "EXPLORATORY_DISCOVERY_FROZEN_BEFORE_AV":
        raise ValueError("freeze status is not exploratory pre-AV")
    if freeze.get("confirmatory") is not False:
        raise ValueError("freeze confirmatory flag must be false")
    itt = freeze.get("itt")
    selection = freeze.get("selection")
    if not isinstance(itt, dict) or int(itt.get("n_features", -1)) != N_FEATURES:
        raise ValueError("freeze must record 45 features")
    if not isinstance(selection, dict) or int(selection.get("n_features", -1)) != N_FEATURES:
        raise ValueError("freeze selection must record 45 features")
    if freeze.get("freeze_sha256") not in (None, freeze_sha):
        raise ValueError("freeze self hash mismatch")
    features = freeze.get("features")
    if not isinstance(features, list) or len(features) != N_FEATURES:
        raise ValueError("freeze must contain exactly 45 features")
    seen: set[int] = set()
    for frow in features:
        if not isinstance(frow, dict):
            raise ValueError("freeze feature row is not an object")
        feature = int(frow.get("feature", -1))
        if feature in seen:
            raise ValueError(f"duplicate freeze feature {feature}")
        seen.add(feature)
        discovery = frow.get("discovery")
        if not isinstance(discovery, list) or len(discovery) != N_DISCOVERY:
            raise ValueError(f"feature {feature} must have four discovery contexts")
        heldout = frow.get("heldout_positive")
        if not isinstance(heldout, list) or len(heldout) != N_DISCOVERY:
            raise ValueError(f"feature {feature} must have four heldout contexts")
    if len(seen) != N_FEATURES:
        raise ValueError("freeze feature ids are not unique")


def _validate_result(result: Mapping[str, Any], freeze_sha: str,
                     expected_features: set[int],
                     freeze: Mapping[str, Any] | None = None,
                     protocol_sha: str | None = None) -> list[dict[str, Any]]:
    if result.get("confirmatory") is not False:
        raise ValueError("result confirmatory flag must be false")
    status = str(result.get("status", ""))
    if status != "EXPLORATORY_DISCOVERY_AV_COMPLETE":
        raise ValueError(f"result is not full discovery status: {status!r}")
    if result.get("freeze_sha256") != freeze_sha:
        raise ValueError("result is not bound to freeze sha256")
    if protocol_sha is not None and result.get("protocol_sha256") not in (None, protocol_sha):
        raise ValueError("result protocol SHA-256 does not match protocol input")
    rows = _extract_rows(result)
    if len(rows) != N_ROWS:
        raise ValueError(f"result must contain exactly {N_ROWS} rows, got {len(rows)}")
    discovery_refs: dict[tuple[int, int], Mapping[str, Any]] = {}
    if freeze is not None:
        features = freeze.get("features")
        if not isinstance(features, list):
            raise ValueError("freeze lacks feature rows needed for AV reference validation")
        for feature_row in features:
            if not isinstance(feature_row, dict):
                raise ValueError("freeze feature row is not an object")
            feature = int(feature_row.get("feature", -1))
            discovery = feature_row.get("discovery")
            if not isinstance(discovery, list) or len(discovery) != N_DISCOVERY:
                raise ValueError(f"freeze feature {feature} lacks four discovery rows")
            for di, context in enumerate(discovery):
                if not isinstance(context, dict):
                    raise ValueError(f"freeze feature {feature} discovery row {di} is not an object")
                discovery_refs[(feature, di)] = context
    seen: set[tuple[int, int, str]] = set()
    by_feature_index: dict[tuple[int, int], set[str]] = {}
    for i, row in enumerate(rows):
        feature = int(row.get("feature", -1))
        di = int(row.get("discovery_index", -1))
        role = str(row.get("role", ""))
        if role not in {"discovery", "sae_feature_ablated"}:
            raise ValueError(f"result row {i} has invalid role {role!r}")
        arm = str(row.get("arm", ""))
        if arm not in {"NLA_RAW", "NLA_CONTRASTIVE"}:
            raise ValueError(f"result row {i} has invalid arm {arm!r}")
        is_raw = role == "discovery" and arm == "NLA_RAW"
        is_abl = role == "sae_feature_ablated" and arm == "NLA_CONTRASTIVE"
        if not (is_raw or is_abl):
            raise ValueError(f"result row {i} role/arm mismatch")
        if not 0 <= di < N_DISCOVERY:
            raise ValueError(f"result row {i} discovery_index out of range")
        if feature not in expected_features:
            raise ValueError(f"result row {i} feature is outside freeze: {feature}")
        key = (feature, di, "raw" if is_raw else "ablated")
        if key in seen:
            raise ValueError(f"duplicate result row for feature/index/arm {key}")
        seen.add(key)
        by_feature_index.setdefault((feature, di), set()).add(key[2])
        for field in ("vector_sha256", "context_text", "raw_text_ref"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"result row {i} has empty/missing {field}")
        if not _sha256_hex(row["vector_sha256"]):
            raise ValueError(f"result row {i} vector hash is not sha256")
        if "doc_id" not in row or "position" not in row:
            raise ValueError(f"result row {i} lacks doc_id/position reference")
        try:
            row_doc_id = int(row["doc_id"])
            row_position = int(row["position"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"result row {i} doc_id/position is not integral") from exc
        if row_position < 0:
            raise ValueError(f"result row {i} position is negative")
        if freeze is not None:
            expected_context = discovery_refs.get((feature, di))
            if expected_context is None:
                raise ValueError(f"result row {i} lacks matching freeze discovery row")
            if row_doc_id != int(expected_context.get("doc_id", -1)):
                raise ValueError(f"result row {i} doc_id does not match freeze discovery row")
            if row_position != int(expected_context.get("position", -1)):
                raise ValueError(f"result row {i} position does not match freeze discovery row")
            expected_context_text = expected_context.get("context_text")
            if isinstance(expected_context_text, str) and row.get("context_text") != expected_context_text:
                raise ValueError(f"result row {i} context_text does not match freeze discovery row")
            expected_raw_ref = expected_context.get("raw_text_ref")
            actual_raw_ref = row.get("raw_text_ref")
            if not isinstance(actual_raw_ref, str) or not actual_raw_ref.strip():
                raise ValueError(f"result row {i} raw_text_ref is empty")
            if expected_raw_ref is not None:
                if str(actual_raw_ref) != str(expected_raw_ref):
                    raise ValueError(f"result row {i} raw_text_ref does not match freeze discovery row")
            elif str(actual_raw_ref) != str(row_doc_id):
                # 57's actual schema uses the opaque textual document ID as
                # raw_text_ref; reject any other reference when no richer
                # freeze reference exists.
                raise ValueError(f"result row {i} raw_text_ref is not the freeze document ID")
        explanation = row.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"result row {i} has empty/missing explanation")
        if is_abl:
            for field in ("ablation_activation", "ablation_norm_x", "ablation_norm_x_minus", "ablation_cosine"):
                if field not in row:
                    raise ValueError(f"ablated result row {i} lacks {field}")
    if len(seen) != N_ROWS or any(v != {"raw", "ablated"} for v in by_feature_index.values()):
        raise ValueError("result does not have one raw and one ablated row per feature/discovery index")
    if len(by_feature_index) != N_FEATURES * N_DISCOVERY:
        raise ValueError("result feature/discovery index cardinality mismatch")
    if {key[0] for key in by_feature_index} != expected_features:
        raise ValueError("result feature IDs do not exactly match the freeze")
    return rows


def _condition_donor_map(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[tuple[int, int], Mapping[str, Any]], list[dict[str, Any]]]:
    """Build an exact minimum-cost, same-stratum feature derangement.

    A target feature is matched to one donor feature in its stratum.  The
    assignment is a bijection with no self donors and minimizes the sum, over
    the four discovery indices, of absolute UTF-8 snippet-byte length
    differences.  A bitmask dynamic program is small and deterministic for
    the protocol's 15-feature strata; equal-cost assignments are resolved by
    the lexicographically smallest donor-feature sequence in ascending target
    feature order.
    """
    by_feature: dict[int, list[Mapping[str, Any]]] = {}
    strata_by_feature: dict[int, set[str]] = {}
    for row in rows:
        feature = int(row["feature"])
        by_feature.setdefault(feature, []).append(row)
        strata_by_feature.setdefault(feature, set()).add(str(row.get("stratum", "")))
    feature_meta: dict[int, dict[str, Any]] = {}
    for feature, members in by_feature.items():
        if (len(members) != N_DISCOVERY
                or {int(r.get("discovery_index", -1)) for r in members} != set(range(N_DISCOVERY))
                or len(strata_by_feature.get(feature, set())) != 1):
            raise ValueError(f"donor source feature {feature} lacks four raw snippets or has mixed strata")
        by_index = {int(r["discovery_index"]): r for r in members}
        lengths = [len(_snippet_text(by_index[i]).encode("utf-8"))
                   for i in range(N_DISCOVERY)]
        feature_meta[feature] = {
            "feature": feature,
            "stratum": next(iter(strata_by_feature[feature])),
            "snippet_lengths_utf8": lengths,
            "total_utf8_bytes": sum(lengths),
            "rows": by_index,
        }

    out: dict[tuple[int, int], Mapping[str, Any]] = {}
    stats: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for meta in feature_meta.values():
        groups.setdefault(str(meta["stratum"]), []).append(meta)
    for stratum, members in sorted(groups.items(), key=lambda item: item[0]):
        ordered = sorted(members, key=lambda m: int(m["feature"]))
        n = len(ordered)
        if n < 2:
            raise ValueError(f"same-stratum donor group {stratum!r} has fewer than two features")
        if n > 15:
            raise ValueError(f"same-stratum donor group {stratum!r} exceeds 15-feature bitmask DP")
        pair_cost = [[sum(abs(int(ordered[i]["snippet_lengths_utf8"][di])
                              - int(ordered[j]["snippet_lengths_utf8"][di]))
                            for di in range(N_DISCOVERY))
                      for j in range(n)] for i in range(n)]

        @functools.lru_cache(maxsize=None)
        def solve(target_index: int, used_mask: int) -> tuple[int, tuple[int, ...]] | None:
            if target_index == n:
                return 0, ()
            best: tuple[int, tuple[int, ...]] | None = None
            for donor_index in range(n):
                if donor_index == target_index or (used_mask & (1 << donor_index)):
                    continue
                suffix = solve(target_index + 1, used_mask | (1 << donor_index))
                if suffix is None:
                    continue
                candidate = (pair_cost[target_index][donor_index] + suffix[0],
                             (donor_index,) + suffix[1])
                if best is None or (candidate[0], tuple(ordered[i]["feature"] for i in candidate[1])) < \
                        (best[0], tuple(ordered[i]["feature"] for i in best[1])):
                    best = candidate
            return best

        assignment = solve(0, 0)
        if assignment is None:
            raise ValueError(f"no nonself derangement exists for same-stratum group {stratum!r}")
        total_cost, donor_indices = assignment
        if len(donor_indices) != n or set(donor_indices) != set(range(n)):
            raise ValueError(f"same-stratum donor assignment is not bijective for {stratum!r}")
        for target_index, donor_index in enumerate(donor_indices):
            target = ordered[target_index]
            donor = ordered[donor_index]
            if (int(target["feature"]) == int(donor["feature"])
                    or str(target["stratum"]) != str(donor["stratum"])):
                raise ValueError("same-stratum donor derangement violates nonself/stratum constraint")
            target_lengths = [int(x) for x in target["snippet_lengths_utf8"]]
            donor_lengths = [int(x) for x in donor["snippet_lengths_utf8"]]
            deltas = [abs(target_lengths[di] - donor_lengths[di]) for di in range(N_DISCOVERY)]
            for di in range(N_DISCOVERY):
                out[(int(target["feature"]), di)] = donor["rows"][di]
            stats.append({
                "stratum": stratum,
                "target_feature": int(target["feature"]),
                "donor_feature": int(donor["feature"]),
                "target_snippet_lengths_utf8": target_lengths,
                "donor_snippet_lengths_utf8": donor_lengths,
                "snippet_length_deltas_utf8": deltas,
                "target_total_utf8_bytes": int(sum(target_lengths)),
                "donor_total_utf8_bytes": int(sum(donor_lengths)),
                "length_difference_utf8": int(sum(donor_lengths) - sum(target_lengths)),
                "total_absolute_cost_utf8": int(sum(deltas)),
                "max_snippet_delta_utf8": int(max(deltas)),
            })

    expected_keys = {(int(feature), di) for feature in by_feature for di in range(N_DISCOVERY)}
    if set(out) != expected_keys:
        raise ValueError("same-stratum donor map is incomplete")
    assignments = {(int(target), int(stat["donor_feature"]))
                   for stat in stats for target in (stat["target_feature"],)}
    if len(assignments) != len(stats) or any(a == b for a, b in assignments):
        raise ValueError("same-stratum donor map is not a feature-level derangement")
    return out, sorted(stats, key=lambda x: (x["target_feature"], x["donor_feature"]))


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["cases"],
        "additionalProperties": False,
        "properties": {
            "cases": {
                "type": "array", "minItems": 5, "maxItems": 5,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["case_id", "hypothesis", "positive_cues",
                                  "exclusion_cues", "abstain", "confidence"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "hypothesis": {"type": "string", "maxLength": 256},
                        "positive_cues": {"type": "array", "items": {"type": "string"}},
                        "exclusion_cues": {"type": "array", "items": {"type": "string"}},
                        "abstain": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
        },
    }


def _case_id(rng: random.Random, used: set[str] | None = None) -> str:
    """Generate a deterministic opaque case ID, avoiding all prior IDs."""
    while True:
        candidate = "case_" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(18))
        if used is None or candidate not in used:
            if used is not None:
                used.add(candidate)
            return candidate


def _build_jobs(freeze: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 45 deterministic cross-feature batches of five opaque cases.

    Batch ``b`` receives condition ``j`` from feature index ``(b-j) mod 45``
    (features are sorted by their frozen integer IDs).  Thus each batch has
    five distinct features and each feature's five conditions occupy five
    distinct batches.  The public prompt contains only opaque case IDs and
    their payloads; the private condition map retains feature/condition
    provenance for downstream evaluation.
    """
    by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    strata: dict[int, str] = {}
    contexts_by_feature: dict[int, list[Mapping[str, Any]]] = {}
    for frow in freeze["features"]:
        feature = int(frow["feature"])
        strata[feature] = str(frow.get("stratum", ""))
        contexts_by_feature[feature] = list(frow["discovery"])
    for row in rows:
        arm = "raw" if row["role"] == "discovery" else "ablated"
        by_key[(int(row["feature"]), int(row["discovery_index"]), arm)] = row
    raw_rows: list[dict[str, Any]] = []
    for feature in sorted(strata):
        for di in range(N_DISCOVERY):
            raw = dict(by_key[(feature, di, "raw")])
            raw["stratum"] = strata[feature]
            raw_rows.append(raw)
    donor_map, donor_stats = _condition_donor_map(raw_rows)
    if len(donor_map) != N_FEATURES * N_DISCOVERY:
        raise ValueError("unable to construct complete same-stratum donor map")
    rng = random.Random(seed)
    used_case_ids: set[str] = set()
    feature_cases: dict[int, dict[str, dict[str, Any]]] = {}
    for feature in sorted(strata):
        raw = [dict(by_key[(feature, di, "raw")]) for di in range(N_DISCOVERY)]
        abl = [dict(by_key[(feature, di, "ablated")]) for di in range(N_DISCOVERY)]
        contexts = contexts_by_feature[feature]
        # Pull expected activations and decoder-marked token windows from the
        # freeze only; no held-out row is ever placed in a job payload.
        for di, row in enumerate(raw):
            row["stratum"] = strata[feature]
            context = contexts[di]
            row["prompt_context"] = _marked_context(context, _text(row, "context_text"))
        cases: list[dict[str, Any]] = []
        condition_map: dict[str, Any] = {}
        arm_input_utf8_bytes: dict[str, int] = {}
        for condition in CONDITIONS:
            if condition == "SAE_CONTEXT":
                # Exactly four marked raw contexts; no numeric SAE metadata is
                # visible to the interpreter in any Fable payload.
                payload = [{"context": r["prompt_context"]} for r in raw]
            elif condition == "NLA_ASSISTED":
                payload = [{"context": r["prompt_context"], "snippet": _snippet_text(r)} for r in raw]
            elif condition == "NLA_CONTRASTIVE":
                payload = [{"context": raw[i]["prompt_context"],
                            "text_a": _snippet_text(raw[i]),
                            "text_b": _snippet_text(abl[i])}
                           for i in range(N_DISCOVERY)]
            elif condition == "NLA_MISMATCHED":
                payload = [{"context": raw[i]["prompt_context"],
                            "snippet": _snippet_text(donor_map[(feature, i)])}
                           for i in range(N_DISCOVERY)]
            else:
                # NLA_ONLY is the protocol's diagnostic snippet-only arm.
                payload = [{"snippet": _snippet_text(raw[i])} for i in range(N_DISCOVERY)]
            cid = _case_id(rng, used_case_ids)
            # ``evidence`` is deliberately generic: the public prompt must
            # not reveal feature/condition provenance through field names.
            case = {"case_id": cid, "evidence": payload}
            cases.append(case)
            condition_map[cid] = {"feature": feature, "condition": condition,
                                  "donor": condition == "NLA_MISMATCHED",
                                  "input_utf8_bytes": len(_canonical_bytes(payload))}
            arm_input_utf8_bytes[condition] = len(_canonical_bytes(payload))
        feature_cases[feature] = {
            str(condition_map[c["case_id"]]["condition"]): c for c in cases
        }

    feature_ids = sorted(strata)
    if len(feature_ids) != N_FEATURES:
        raise ValueError(f"cross-feature batching requires exactly {N_FEATURES} features")
    jobs: list[dict[str, Any]] = []
    for batch_id in range(N_FEATURES):
        batch_cases: list[dict[str, Any]] = []
        batch_map: dict[str, Any] = {}
        batch_bytes: dict[str, int] = {}
        batch_features: list[int] = []
        for condition_index, condition in enumerate(CONDITIONS):
            feature = feature_ids[(batch_id - condition_index) % N_FEATURES]
            case = dict(feature_cases[feature][condition])
            cid = str(case["case_id"])
            case_map = {"feature": feature, "condition": condition,
                        "donor": condition == "NLA_MISMATCHED",
                        "input_utf8_bytes": len(_canonical_bytes(case["evidence"]))}
            # Keep the private map explicit and detached from public case
            # payloads.  It is the only place feature/condition provenance is
            # retained for the evaluator.
            batch_map[cid] = dict(case_map, batch_id=batch_id)
            batch_bytes[condition] = int(case_map["input_utf8_bytes"])
            batch_cases.append(case)
            batch_features.append(feature)
        if len(set(batch_features)) != len(CONDITIONS):
            raise ValueError(f"batch {batch_id} reuses a feature")
        # Randomize public case order with the fixed protocol RNG; this does
        # not alter the Latin/cyclic assignment or any private mapping.
        rng.shuffle(batch_cases)
        jobs.append({"batch_id": batch_id, "cases": batch_cases,
                     "condition_map": batch_map,
                     "arm_input_utf8_bytes": batch_bytes})

    # Strictly verify the cross-feature coverage invariants before freezing.
    seen_case_ids: set[str] = set()
    feature_batches: dict[int, set[int]] = {feature: set() for feature in feature_ids}
    for job in jobs:
        batch_id = int(job["batch_id"])
        cases_for_job = job["cases"]
        if len(cases_for_job) != len(CONDITIONS):
            raise ValueError(f"batch {batch_id} must contain exactly five cases")
        if set(job["condition_map"]) != {str(case["case_id"]) for case in cases_for_job}:
            raise ValueError(f"batch {batch_id} condition map/cases mismatch")
        batch_features_seen: set[int] = set()
        conditions_seen: set[str] = set()
        for case in cases_for_job:
            cid = str(case["case_id"])
            if cid in seen_case_ids:
                raise ValueError(f"duplicate opaque case ID {cid}")
            seen_case_ids.add(cid)
            mapping = job["condition_map"][cid]
            feature = int(mapping["feature"])
            condition = str(mapping["condition"])
            if feature in batch_features_seen or condition in conditions_seen:
                raise ValueError(f"batch {batch_id} violates distinct feature/condition assignment")
            if condition not in CONDITIONS:
                raise ValueError(f"batch {batch_id} has invalid condition {condition!r}")
            batch_features_seen.add(feature)
            conditions_seen.add(condition)
            feature_batches[feature].add(batch_id)
        if conditions_seen != set(CONDITIONS):
            raise ValueError(f"batch {batch_id} does not cover all five conditions")
    if len(seen_case_ids) != N_FEATURES * len(CONDITIONS):
        raise ValueError("cross-feature batches do not contain exactly 225 unique cases")
    if any(len(batch_ids) != len(CONDITIONS) for batch_ids in feature_batches.values()):
        raise ValueError("each feature's five conditions must occupy five distinct batches")
    # The private feature list is useful for audits but is never interpolated
    # into a prompt.
    return jobs, {"seed": seed, "donors": donor_stats}


def _prompt_for_job(job: Mapping[str, Any]) -> str:
    return ("Independently label five exploratory cases. Do not infer or discuss hidden grouping. "
            "Use only text shown. Return exactly one JSON object matching the supplied schema. For each case "
            "write a concise hypothesis of at most 32 words, positive_cues, exclusion_cues, abstain, and confidence 0..1. "
            f"Use the same schema and a maximum response budget of {MAX_OUTPUT_TOKENS} tokens for every request. "
            "Do not mention hidden metadata, IDs, unshown activations, or held-out contexts.\nCASE_PAYLOAD=" +
            json.dumps(job["cases"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _command_args(command: str, model: str, effort: str, schema_path: Path, prompt: str) -> list[str]:
    # ``--print`` is required for non-interactive JSON output.  An explicit
    # empty ``--tools`` value disables every built-in tool; shell=False keeps
    # the prompt and schema as literal subprocess arguments on Windows/Unix.
    return [command or "claude", "--print", "--model", model, "--effort", effort,
            "--tools", "", "--output-format", "json", "--json-schema",
            schema_path.read_text(encoding="utf-8"), prompt]


def _parse_cli_payload(stdout: bytes, expected_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Claude output is not valid UTF-8 JSON: {exc}") from exc
    envelope = payload if isinstance(payload, dict) else {}
    structured = envelope.get("structured_output") if isinstance(envelope, dict) else None
    if isinstance(structured, dict):
        payload = structured
    elif isinstance(envelope.get("result"), str):
        try:
            payload = json.loads(envelope["result"])
        except Exception as exc:
            raise ValueError(f"Claude result string is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"cases"} or not isinstance(payload.get("cases"), list):
        raise ValueError("Claude JSON lacks cases array")
    cases = payload["cases"]
    if len(cases) != len(expected_ids):
        raise ValueError("Claude returned wrong case count")
    got: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Claude case is not object")
        required_keys = {"case_id", "hypothesis", "positive_cues", "exclusion_cues", "abstain", "confidence"}
        if set(case) != required_keys:
            raise ValueError("Claude case has missing or extra fields")
        cid = case.get("case_id")
        if not isinstance(cid, str) or cid in got or cid not in expected_ids:
            raise ValueError("Claude case_id mismatch/duplicate")
        got.add(cid)
        hyp = case.get("hypothesis")
        if not isinstance(hyp, str) or len(hyp.split()) > 32:
            raise ValueError("hypothesis exceeds 32 words")
        if (not isinstance(case.get("positive_cues"), list) or
                not all(isinstance(x, str) for x in case["positive_cues"]) or
                not isinstance(case.get("exclusion_cues"), list) or
                not all(isinstance(x, str) for x in case["exclusion_cues"])):
            raise ValueError("cue arrays missing")
        if not isinstance(case.get("abstain"), bool):
            raise ValueError("abstain must be boolean")
        conf = case.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= float(conf) <= 1:
            raise ValueError("confidence outside [0,1]")
    if got != expected_ids:
        raise ValueError("Claude case ids are not exactly requested")
    return payload, envelope


def _validate_case_list(cases: Any, expected_ids: set[str]) -> None:
    """Apply the same fail-closed checks to a resumed checkpoint row."""
    wrapper = json.dumps({"cases": cases}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    _parse_cli_payload(wrapper, expected_ids)


def _validate_success_row(row: Mapping[str, Any], expected_job: Mapping[str, Any],
                          cli_version: str, *, expected_model: str,
                          expected_effort: str, expected_tools_disabled: bool,
                          expected_substitution: bool,
                          expected_command_args: Sequence[str] | None = None) -> None:
    """Fail closed when replaying a successful append-only checkpoint row."""
    batch_id = int(expected_job["batch_id"])
    expected_ids = {str(c["case_id"]) for c in expected_job["cases"]}
    if row.get("status") != "ok":
        raise ValueError(f"checkpoint batch {batch_id} is not successful")
    expected_prompt_sha = str(expected_job["prompt_sha256"])
    if row.get("prompt_sha256") != expected_prompt_sha or row.get("input_prompt_sha256") != expected_prompt_sha:
        raise ValueError(f"checkpoint batch {batch_id} prompt SHA mismatch")
    if row.get("claude_version") != cli_version:
        raise ValueError(f"checkpoint batch {batch_id} Claude version mismatch")
    if row.get("requested_model") != expected_model or row.get("effort") != expected_effort:
        raise ValueError(f"checkpoint batch {batch_id} requested model/effort mismatch")
    if row.get("tools_disabled") is not expected_tools_disabled:
        raise ValueError(f"checkpoint batch {batch_id} tools_disabled mismatch")
    if not isinstance(row.get("substitution_before_outcome"), bool) \
            or row.get("substitution_before_outcome") is not expected_substitution:
        raise ValueError(f"checkpoint batch {batch_id} substitution provenance mismatch")
    if expected_command_args is not None:
        expected_args = list(expected_command_args)
        for field in ("command_args", "command_args_without_prompt"):
            if field not in row or _canonical_bytes(row.get(field)) != _canonical_bytes(expected_args):
                raise ValueError(f"checkpoint batch {batch_id} {field} mismatch")
    elif "command_args" in row and "command_args_without_prompt" in row:
        if _canonical_bytes(row.get("command_args")) != _canonical_bytes(row.get("command_args_without_prompt")):
            raise ValueError(f"checkpoint batch {batch_id} command args aliases differ")
    raw_cli_json = row.get("raw_cli_json")
    if not isinstance(raw_cli_json, str):
        raise ValueError(f"checkpoint batch {batch_id} raw_cli_json is not text")
    try:
        reparsed, envelope = _parse_cli_payload(raw_cli_json.encode("utf-8"), expected_ids)
    except Exception as exc:
        raise ValueError(f"checkpoint batch {batch_id} raw_cli_json failed reparse: {exc}") from exc
    parsed_structured = row.get("parsed_structured_result")
    if not isinstance(parsed_structured, dict):
        raise ValueError(f"checkpoint batch {batch_id} lacks parsed_structured_result")
    if _canonical_bytes(reparsed) != _canonical_bytes(parsed_structured):
        raise ValueError(f"checkpoint batch {batch_id} parsed_structured_result differs from raw_cli_json")
    if _canonical_bytes(reparsed.get("cases")) != _canonical_bytes(row.get("cases")):
        raise ValueError(f"checkpoint batch {batch_id} cases differ from raw_cli_json")
    reparsed_model_usage, reparsed_usage, reparsed_models = _usage_and_models(envelope, reparsed)
    reparsed_cost = _cost_from_envelope(envelope, reparsed)
    if not _finite_number(reparsed_cost):
        raise ValueError(f"checkpoint batch {batch_id} raw envelope lacks finite numeric cost")
    if not (_nonempty_mapping(reparsed_model_usage) or _nonempty_mapping(reparsed_usage)):
        raise ValueError(f"checkpoint batch {batch_id} raw envelope lacks nonempty usage/modelUsage")
    for field, expected_value in (("modelUsage", reparsed_model_usage),
                                  ("usage", reparsed_usage),
                                  ("resolved_model_names", reparsed_models),
                                  ("cost", reparsed_cost)):
        if field not in row or _canonical_bytes(row.get(field)) != _canonical_bytes(expected_value):
            raise ValueError(f"checkpoint batch {batch_id} {field} differs from raw envelope")
    attempts = row.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"checkpoint batch {batch_id} attempts is not a list")
    stdout_sha = sha256_bytes(raw_cli_json.encode("utf-8"))
    if not any(isinstance(attempt, dict) and attempt.get("ok") is True
               and attempt.get("stdout_sha256") == stdout_sha for attempt in attempts):
        raise ValueError(f"checkpoint batch {batch_id} has no successful attempt matching raw stdout SHA")
    if not _finite_number(row.get("cost")):
        raise ValueError(f"checkpoint batch {batch_id} cost is not finite numeric")
    model_usage = row.get("modelUsage")
    usage = row.get("usage")
    if not (_nonempty_mapping(model_usage) or _nonempty_mapping(usage)):
        raise ValueError(f"checkpoint batch {batch_id} lacks nonempty usage/modelUsage")
    resolved = row.get("resolved_model_names")
    if resolved is not None and (not isinstance(resolved, list)
                                 or not all(isinstance(name, str) and name.strip() for name in resolved)):
        raise ValueError(f"checkpoint batch {batch_id} resolved model names are malformed")


def _run_job(job: Mapping[str, Any], args: argparse.Namespace, schema_path: Path,
             protocol_sha: str, label_protocol_sha: str, freeze_sha: str, result_sha: str,
             jobs_sha: str, script_sha: str, cli_version: str) -> dict[str, Any]:
    prompt = str(job["prompt"])
    prompt_hash = sha256_bytes(prompt.encode("utf-8"))
    if prompt_hash != str(job["prompt_sha256"]):
        raise ValueError(f"immutable prompt hash mismatch for batch {job['batch_id']}")
    argv = _command_args(args.command, args.model, args.effort, schema_path, prompt)
    start = time.time()
    attempts: list[dict[str, Any]] = []
    max_attempts = 1 + int(args.retries)
    for attempt in range(1, max_attempts + 1):
        stdout = b""
        try:
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, shell=False)
            stdout = proc.stdout
            stderr = proc.stderr
            if proc.returncode != 0:
                raise RuntimeError(f"Claude exited {proc.returncode}: {stderr.decode('utf-8', 'replace')[:1000]}")
            parsed, envelope = _parse_cli_payload(stdout, {str(c["case_id"]) for c in job["cases"]})
            model_usage, usage, resolved_model_names = _usage_and_models(envelope, parsed)
            cost = _cost_from_envelope(envelope, parsed)
            if not _finite_number(cost):
                raise ValueError("Claude response lacks a finite numeric cost")
            if not (_nonempty_mapping(model_usage) or _nonempty_mapping(usage)):
                raise ValueError("Claude response lacks a nonempty usage/modelUsage object")
            raw_cli_json = stdout.decode("utf-8")
            attempts.append({"attempt": attempt, "ok": True, "stdout_sha256": sha256_bytes(stdout), "stderr_sha256": sha256_bytes(stderr), "returncode": proc.returncode})
            row = {"batch_id": int(job["batch_id"]), "status": "ok",
                   "input_prompt_sha256": prompt_hash, "prompt_sha256": prompt_hash,
                   "cases": parsed["cases"], "parsed_structured_result": parsed,
                   "raw_cli_json": raw_cli_json,
                   "stderr": stderr.decode("utf-8", "replace"), "attempts": attempts,
                   "duration_seconds": time.time() - start, "duration_ms_cli": envelope.get("duration_ms"),
                   "modelUsage": model_usage, "usage": usage, "resolved_model_names": resolved_model_names,
                   "cost": cost, "command_args": argv[:-1],
                   "command_args_without_prompt": argv[:-1], "protocol_sha256": protocol_sha,
                   "label_protocol_sha256": label_protocol_sha,
                   "freeze_sha256": freeze_sha, "input_result_sha256": result_sha,
                   "jobs_sha256": jobs_sha, "script_sha256": script_sha,
                   "claude_version": cli_version, "requested_model": str(args.model),
                   "effort": str(args.effort), "tools_disabled": True,
                   "substitution_before_outcome": bool(getattr(args, "allow_model_substitution", False))}
            _validate_success_row(row, job, cli_version, expected_model=str(args.model),
                                  expected_effort=str(args.effort), expected_tools_disabled=True,
                                  expected_substitution=bool(getattr(args, "allow_model_substitution", False)),
                                  expected_command_args=argv[:-1])
            return row
        except Exception as exc:
            attempts.append({"attempt": attempt, "ok": False, "error": str(exc)})
            if attempt >= max_attempts:
                return {"batch_id": int(job["batch_id"]), "status": "error",
                        "input_prompt_sha256": prompt_hash, "prompt_sha256": prompt_hash,
                        "raw_cli_json": stdout.decode("utf-8", "replace"), "error": str(exc),
                        "attempts": attempts, "duration_seconds": time.time() - start,
                        "command_args": argv[:-1],
                        "command_args_without_prompt": argv[:-1], "protocol_sha256": protocol_sha,
                        "label_protocol_sha256": label_protocol_sha,
                        "freeze_sha256": freeze_sha, "input_result_sha256": result_sha,
                        "jobs_sha256": jobs_sha, "script_sha256": script_sha,
                        "claude_version": cli_version, "requested_model": str(args.model),
                        "effort": str(args.effort), "tools_disabled": True,
                        "substitution_before_outcome": bool(getattr(args, "allow_model_substitution", False))}
    raise AssertionError("unreachable")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_RESULTS / "j1_discovery_freeze_v1.json")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULTS / "j1_discovery_result_v1.json")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_RESULTS / "J1_DISCOVERY_PILOT_PROTOCOL_2026-08-06.md")
    parser.add_argument("--label-protocol", type=Path,
                        default=DEFAULT_RESULTS / "J1_DISCOVERY_LABEL_PROTOCOL_2026-08-06.md")
    parser.add_argument("--out-jobs", type=Path, default=DEFAULT_RESULTS / "j1_discovery_labels_jobs_v1.json")
    parser.add_argument("--out-checkpoint", type=Path, default=DEFAULT_RESULTS / "j1_discovery_labels_checkpoint_v1.jsonl")
    parser.add_argument("--out-result", type=Path, default=DEFAULT_RESULTS / "j1_discovery_labels_v1.json")
    parser.add_argument("--command", default="claude")
    parser.add_argument("--model", default="fable")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-model-substitution", action="store_true",
                        help="explicitly permit non-fable/low/claude invocation and record it before outcome")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed != SEED:
        raise ValueError(f"seed is fixed at {SEED}")
    if args.concurrency < 1 or not 0 <= args.retries <= MAX_RETRIES:
        raise ValueError(f"concurrency must be positive and retries must be in [0,{MAX_RETRIES}]")
    command_basename = os.path.basename(str(args.command).replace("\\", "/"))
    command_stem = os.path.splitext(command_basename)[0].lower()
    substitution = bool(args.allow_model_substitution)
    if not substitution and (str(args.model) != "fable" or str(args.effort) != "low"
                              or command_stem != "claude"):
        raise ValueError("frozen J1 invocation requires model=fable, effort=low, command basename claude; "
                         "pass --allow-model-substitution to override")
    for p in (args.freeze, args.result, args.protocol, args.label_protocol):
        if not p.is_file():
            raise FileNotFoundError(p)
    freeze_sha = _verify_input_sidecar(args.freeze)
    result_sha = _verify_input_sidecar(args.result)
    protocol_sha = _verify_input_sidecar(args.protocol)
    label_protocol_sha = _verify_input_sidecar(args.label_protocol)
    script_sha = sha256_file(Path(__file__))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    _validate_freeze(freeze, freeze_sha)
    freeze_inputs = freeze.get("inputs")
    if isinstance(freeze_inputs, dict):
        frozen_protocol_sha = freeze_inputs.get("protocol_sha256")
        if frozen_protocol_sha is not None and frozen_protocol_sha != protocol_sha:
            raise ValueError("freeze protocol SHA-256 does not match protocol input")
    for key in ("protocol_sha256", "protocol_sha"):
        frozen_protocol_sha = freeze.get(key)
        if frozen_protocol_sha is not None and frozen_protocol_sha != protocol_sha:
            raise ValueError("freeze protocol SHA-256 does not match protocol input")
    expected_features = {int(row["feature"]) for row in freeze["features"]}
    rows = _validate_result(result, freeze_sha, expected_features, freeze=freeze,
                            protocol_sha=protocol_sha)
    jobs, donor_info = _build_jobs(freeze, rows, args.seed)
    schema = _schema()
    schema_path = args.out_jobs.with_suffix(args.out_jobs.suffix + ".schema.json")
    schema_bytes = _immutable_json(schema_path, schema)
    _immutable_sidecar(Path(str(schema_path) + ".sha256"), schema_bytes)
    frozen_job_rows: list[dict[str, Any]] = []
    for job in jobs:
        prompt = _prompt_for_job(job)
        frozen_job_rows.append({"batch_id": int(job["batch_id"]), "cases": job["cases"],
                                "condition_map": job["condition_map"],
                                "arm_input_utf8_bytes": job["arm_input_utf8_bytes"],
                                "prompt": prompt,
                                "prompt_sha256": sha256_bytes(prompt.encode("utf-8"))})
    frozen_jobs = {"schema_version": 1, "status": "EXPLORATORY_LABEL_JOBS_FROZEN",
                   "seed": args.seed, "freeze_sha256": freeze_sha,
                   "input_result_sha256": result_sha, "protocol_sha256": protocol_sha,
                   "label_protocol_sha256": label_protocol_sha,
                   "script_sha256": script_sha, "schema_sha256": schema_bytes,
                   "command": args.command, "model": args.model, "effort": args.effort,
                   "requested_model": args.model,
                   "tools_disabled": True, "output_format": "json",
                   "max_output_tokens": MAX_OUTPUT_TOKENS,
                   "substitution_before_outcome": substitution,
                   "conditions": list(CONDITIONS), "donor_info": donor_info,
                   "jobs": frozen_job_rows}
    jobs_sha = _immutable_json(args.out_jobs, frozen_jobs)
    _immutable_sidecar(Path(str(args.out_jobs) + ".sha256"), jobs_sha)
    if args.dry_run:
        return 0
    version_proc = subprocess.run([args.command, "--version"], capture_output=True,
                                  text=False, check=False, shell=False)
    if version_proc.returncode != 0:
        raise RuntimeError(f"cannot determine Claude CLI version (exit {version_proc.returncode})")
    cli_version = version_proc.stdout.decode("utf-8", "replace").strip()
    if not cli_version:
        raise RuntimeError("Claude CLI returned an empty --version string")
    completed: dict[int, dict[str, Any]] = {}
    if args.out_checkpoint.exists():
        for line_no, line in enumerate(args.out_checkpoint.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            batch_id = int(row.get("batch_id", -1))
            expected_job = next((j for j in frozen_jobs["jobs"] if int(j["batch_id"]) == batch_id), None)
            if (expected_job is None or
                    row.get("freeze_sha256") != freeze_sha or
                    row.get("input_result_sha256") != result_sha or
                    row.get("jobs_sha256") != jobs_sha or
                    row.get("input_prompt_sha256") != expected_job["prompt_sha256"] or
                    row.get("prompt_sha256") != expected_job["prompt_sha256"] or
                    row.get("script_sha256") != script_sha or
                    row.get("protocol_sha256") != protocol_sha or
                    row.get("label_protocol_sha256") != label_protocol_sha or
                    row.get("claude_version") != cli_version or
                    row.get("requested_model") != args.model or
                    row.get("effort") != args.effort or
                    row.get("tools_disabled") is not True or
                    row.get("substitution_before_outcome") is not substitution):
                raise ValueError(f"checkpoint binding mismatch at line {line_no}")
            status = row.get("status")
            if status == "ok":
                if batch_id in completed:
                    raise ValueError(f"duplicate successful checkpoint row at line {line_no}")
                _validate_success_row(row, expected_job, cli_version,
                                      expected_model=str(args.model), expected_effort=str(args.effort),
                                      expected_tools_disabled=True, expected_substitution=substitution,
                                      expected_command_args=_command_args(
                                          args.command, args.model, args.effort, schema_path,
                                          str(expected_job["prompt"]))[:-1])
                completed[batch_id] = row
            elif status == "error":
                if batch_id in completed:
                    raise ValueError(f"error checkpoint row after success at line {line_no}")
                # Error rows are intentionally append-only and may be followed
                # by a later successful retry.
                if not isinstance(row.get("error"), str) or not row.get("error"):
                    raise ValueError(f"error checkpoint row lacks error text at line {line_no}")
            else:
                raise ValueError(f"checkpoint row has invalid status at line {line_no}")
    call_jobs = frozen_jobs["jobs"]
    missing = [j for j in call_jobs if int(j["batch_id"]) not in completed]
    # A pre-existing final result is immutable.  Never spend on missing calls
    # when a prior result would make the eventual write ambiguous; a complete
    # checkpoint can be replayed byte-identically and will be verified below.
    if missing and args.out_result.exists():
        raise RuntimeError("existing final result with incomplete checkpoint; refusing to call Claude")
    if args.out_result.exists():
        _verify_input_sidecar(args.out_result)
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_run_job, j, args, schema_path, protocol_sha,
                                   label_protocol_sha, freeze_sha, result_sha, jobs_sha, script_sha,
                                   cli_version) for j in missing]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                _append_fsync(args.out_checkpoint, row)
                if row.get("status") == "ok":
                    completed[int(row["batch_id"])] = row
    if len(completed) != len(call_jobs) or any(row.get("status") != "ok" for row in completed.values()):
        raise RuntimeError("one or more Claude label jobs failed; result withheld (fail closed)")
    first_prompt = str(call_jobs[0]["prompt"])
    exact_command_args = _command_args(args.command, args.model, args.effort, schema_path, first_prompt)[:-1]
    payload = {"schema_version": 1, "experiment": "J1 exploratory discovery labels",
               "status": "EXPLORATORY_DISCOVERY_LABELS_COMPLETE", "confirmatory": False,
               "claim_scope": "discovery_only_no_confirmatory_inference",
               "itt": {"n_features": N_FEATURES, "recorded": N_FEATURES},
               "freeze_sha256": freeze_sha, "input_result_sha256": result_sha,
               "jobs_sha256": jobs_sha, "protocol_sha256": protocol_sha,
               "label_protocol_sha256": label_protocol_sha,
               "script_sha256": script_sha, "schema_sha256": schema_bytes,
               "seed": args.seed, "claude_version": cli_version,
               "command_args_without_prompt": exact_command_args,
               "requested_model": args.model, "effort": args.effort,
               "tools_disabled": True, "output_format": "json",
               "max_output_tokens": MAX_OUTPUT_TOKENS,
               "substitution_before_outcome": substitution,
               "conditions": list(CONDITIONS), "donor_info": donor_info,
               "job_summaries": [{"batch_id": int(j["batch_id"]),
                                  "case_ids": [str(c["case_id"]) for c in j["cases"]],
                                  "condition_map": j["condition_map"],
                                  "prompt_sha256": j["prompt_sha256"],
                                  "arm_input_utf8_bytes": j["arm_input_utf8_bytes"]}
                                 for j in call_jobs],
               "rows": [completed[int(j["batch_id"])] for j in call_jobs],
               "checks": {"n_features": N_FEATURES, "n_batches": len(call_jobs),
                          "n_rows": len(call_jobs), "n_labels": len(call_jobs) * len(CONDITIONS),
                          "all_arms_input_utf8_bytes": {str(j["batch_id"]): j["arm_input_utf8_bytes"] for j in call_jobs}}}
    out_sha = _immutable_json(args.out_result, payload)
    _immutable_sidecar(Path(str(args.out_result) + ".sha256"), out_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
