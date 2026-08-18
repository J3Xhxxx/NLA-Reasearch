#!/usr/bin/env python3
"""Fail-closed provenance validation for C1 activation Parquet assets.

The validator is intentionally usable for either the discovery (``train``)
or held-out (``test``) manifest.  It binds every activation document back to
the frozen JSONL row, verifies that extraction retained every position from
``min_position`` onward, checks every residual vector, and fingerprints the
exact extractor and base-model bytes needed to reproduce the asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
OPTIONAL_MODEL_FILES = {
    "generation_config.json",
    "nla_meta.yaml",
}
EXTRACTION_METADATA_PREFIX = "nla.activation_extraction."
EXTRACTION_DTYPES = {"bfloat16", "float16", "float32"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {error}") from error


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    result = require_nonempty_string(value, label)
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return result


def aliased_string(
    row: dict[str, Any],
    primary: str,
    alias: str,
    label: str,
) -> str:
    present = [
        (key, row[key])
        for key in (primary, alias)
        if key in row
    ]
    if not present:
        raise ValueError(f"{label} lacks {primary!r}/{alias!r}")
    values = [
        require_nonempty_string(value, f"{label}.{key}")
        for key, value in present
    ]
    if len(set(values)) != 1:
        raise ValueError(
            f"{label} has conflicting {primary!r}/{alias!r} values"
        )
    return values[0]


def load_manifest(
    path: Path,
    expected_split: str,
    expected_documents: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = parse_json(line, f"manifest JSON on line {line_number}")
        if not isinstance(raw, dict):
            raise ValueError(
                f"manifest line {line_number} must be a JSON object"
            )
        label = f"manifest line {line_number}"
        prompt_id = require_nonempty_string(raw.get("id"), f"{label}.id")
        prompt = aliased_string(raw, "text", "prompt", label)
        topic = require_nonempty_string(raw.get("topic"), f"{label}.topic")
        domain = aliased_string(raw, "axis_domain", "domain", label)
        language = aliased_string(
            raw, "axis_language", "language", label
        )
        split = require_nonempty_string(raw.get("split"), f"{label}.split")
        if split != expected_split:
            raise ValueError(
                f"{label}.split is {split!r}, expected {expected_split!r}"
            )

        computed_prompt_sha = sha256_text(prompt)
        declared_prompt_hashes = [
            require_sha256(raw[key], f"{label}.{key}")
            for key in ("prompt_sha256", "prompt_sha")
            if key in raw
        ]
        if declared_prompt_hashes and (
            len(set(declared_prompt_hashes)) != 1
            or declared_prompt_hashes[0] != computed_prompt_sha
        ):
            raise ValueError(f"{label} has an invalid prompt SHA256")
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "topic": topic,
                "axis_domain": domain,
                "axis_language": language,
                "split": split,
                "prompt_sha256": computed_prompt_sha,
            }
        )

    if len(rows) != expected_documents:
        raise ValueError(
            f"manifest has {len(rows)} documents, expected "
            f"{expected_documents}"
        )
    prompt_ids = [row["prompt_id"] for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("manifest prompt IDs are not unique")
    return rows


def collect_key_values(value: Any, key: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key == key:
                result.append(child_value)
            result.extend(collect_key_values(child_value, key))
    elif isinstance(value, list):
        for child in value:
            result.extend(collect_key_values(child, key))
    return result


def manual_audit_manifest_sha(
    audit: dict[str, Any],
    expected_split: str,
) -> tuple[str, str]:
    key_priority = (
        (
            "discovery_manifest_sha256",
            "train_manifest_sha256",
            "manifest_sha256",
            "manifest_sha",
        )
        if expected_split == "train"
        else (
            "heldout_manifest_sha256",
            "test_manifest_sha256",
            "manifest_sha256",
            "manifest_sha",
        )
    )
    for key in key_priority:
        values = collect_key_values(audit, key)
        if not values:
            continue
        normalized = {
            require_sha256(value, f"manual audit {key}") for value in values
        }
        if len(normalized) != 1:
            raise ValueError(
                f"manual audit has conflicting values for {key}"
            )
        return next(iter(normalized)), key
    raise ValueError(
        "manual audit lacks a manifest SHA256 for "
        f"expected_split={expected_split!r}"
    )


def validate_manual_audit(
    path: Path,
    expected_split: str,
    actual_manifest_sha: str,
) -> dict[str, str]:
    value = parse_json(path.read_text(encoding="utf-8"), "manual audit JSON")
    if not isinstance(value, dict):
        raise ValueError("manual audit must be a JSON object")
    status = require_nonempty_string(
        value.get("status"), "manual audit status"
    )
    if status.upper() != "PASS":
        raise ValueError(
            f"manual audit status must be PASS, got {status!r}"
        )
    declared_sha, declared_field = manual_audit_manifest_sha(
        value, expected_split
    )
    if declared_sha != actual_manifest_sha:
        raise ValueError(
            "manual audit manifest SHA mismatch: "
            f"declared={declared_sha}, actual={actual_manifest_sha}"
        )
    return {
        "status": "PASS",
        "manifest_sha256": declared_sha,
        "manifest_sha256_source_field": declared_field,
    }


def nested_config_integer(
    config: dict[str, Any],
    key: str,
) -> int:
    candidates: list[Any] = []
    if key in config:
        candidates.append(config[key])
    for section in ("text_config", "language_config", "model_config"):
        child = config.get(section)
        if isinstance(child, dict) and key in child:
            candidates.append(child[key])
    integers = {
        int(value)
        for value in candidates
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    }
    if not integers:
        raise ValueError(f"base model config lacks a positive integer {key}")
    if len(integers) != 1:
        raise ValueError(
            f"base model config has conflicting {key} values: "
            f"{sorted(integers)}"
        )
    return next(iter(integers))


def validate_safetensor_index(
    model_directory: Path,
    shard_names: set[str],
) -> None:
    index_path = model_directory / "model.safetensors.index.json"
    index = parse_json(
        index_path.read_text(encoding="utf-8"),
        "model.safetensors.index.json",
    )
    if not isinstance(index, dict) or not isinstance(
        index.get("weight_map"), dict
    ):
        raise ValueError(
            "model.safetensors.index.json lacks an object weight_map"
        )
    referenced = {
        require_nonempty_string(
            name, "model.safetensors.index.json shard name"
        )
        for name in index["weight_map"].values()
    }
    unsafe = {
        name
        for name in referenced
        if Path(name).is_absolute() or ".." in Path(name).parts
    }
    if unsafe:
        raise ValueError(
            f"safetensor index contains unsafe shard names: {sorted(unsafe)}"
        )
    missing = referenced - shard_names
    if missing:
        raise FileNotFoundError(
            f"safetensor index references absent shards: {sorted(missing)}"
        )


def fingerprint_base_model(
    directory: Path,
    layer_index: int,
) -> tuple[dict[str, Any], int]:
    missing = [
        name
        for name in sorted(REQUIRED_MODEL_FILES)
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"base model is missing required identity files {missing}"
        )

    shard_paths = sorted(
        (
            path
            for path in directory.rglob("*.safetensors")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not shard_paths:
        raise ValueError("base model has no safetensor weights")
    shard_names = {
        path.relative_to(directory).as_posix() for path in shard_paths
    }
    validate_safetensor_index(directory, shard_names)

    # Preserve logical paths below the model directory even when Hugging Face
    # stores a file as a symlink into its blob cache.  Model-freeze uses these
    # same relative names while ``open``/``stat`` still follow the target.
    identity_paths = {
        directory / name
        for name in REQUIRED_MODEL_FILES
    }
    identity_paths.update(shard_paths)
    for name in OPTIONAL_MODEL_FILES:
        path = directory / name
        if path.is_file():
            identity_paths.add(path)

    files: list[dict[str, Any]] = []
    for path in identity_paths:
        try:
            name = path.relative_to(directory).as_posix()
        except ValueError as error:
            raise ValueError(
                f"base-model identity file resolves outside directory: {path}"
            ) from error
        files.append(
            {
                "name": name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    files.sort(key=lambda row: row["name"])
    names = [row["name"] for row in files]
    if len(names) != len(set(names)):
        raise ValueError("base-model identity filenames are not unique")
    fingerprint = sha256_text(canonical_json(files))

    config_value = parse_json(
        (directory / "config.json").read_text(encoding="utf-8"),
        "base model config",
    )
    if not isinstance(config_value, dict):
        raise ValueError("base model config must be a JSON object")
    d_model = nested_config_integer(config_value, "hidden_size")
    n_layers = nested_config_integer(config_value, "num_hidden_layers")
    if not 0 <= layer_index < n_layers:
        raise ValueError(
            f"layer_index={layer_index} is outside the model's "
            f"{n_layers} layers"
        )
    return (
        {
            "path": str(directory),
            "identity_files": files,
            "safetensor_shards": sorted(shard_names),
            "index_files": ["model.safetensors.index.json"],
            "configured_d_model": d_model,
            "configured_num_hidden_layers": n_layers,
            "identity_sha256": fingerprint,
        },
        d_model,
    )


def require_parquet_columns(parquet: pq.ParquetFile) -> set[str]:
    required = {
        "activation_vector",
        "position",
        "doc_id",
        "prompt_id",
        "axis_domain",
        "axis_language",
        "split",
        "topic",
        "prompt",
        "prompt_sha256",
    }
    columns = set(parquet.schema_arrow.names)
    missing = required - columns
    if missing:
        raise KeyError(
            f"activation parquet is missing columns {sorted(missing)}"
        )
    if not pa.types.is_integer(
        parquet.schema_arrow.field("doc_id").type
    ):
        raise TypeError("activation parquet doc_id must be an integer column")
    if not pa.types.is_integer(
        parquet.schema_arrow.field("position").type
    ):
        raise TypeError("activation parquet position must be an integer column")
    vector_type = parquet.schema_arrow.field("activation_vector").type
    if not (
        pa.types.is_list(vector_type)
        or pa.types.is_large_list(vector_type)
        or pa.types.is_fixed_size_list(vector_type)
    ):
        raise TypeError(
            "activation_vector must be a list or fixed-size-list column"
        )
    if not pa.types.is_floating(vector_type.value_type):
        raise TypeError("activation_vector values must have floating type")
    return required


def validate_extraction_schema_metadata(
    parquet: pq.ParquetFile,
    *,
    layer_index: int,
    min_position: int,
    max_per_prompt: int,
    dtype: str,
) -> dict[str, Any]:
    """Read extractor settings from Arrow schema metadata and match the CLI."""
    raw_metadata = parquet.schema_arrow.metadata or {}
    required_names = (
        "schema_version",
        "layer_index",
        "min_position",
        "max_per_prompt",
        "dtype",
    )
    decoded: dict[str, str] = {}
    for name in required_names:
        key = f"{EXTRACTION_METADATA_PREFIX}{name}".encode("ascii")
        if key not in raw_metadata:
            raise ValueError(
                "activation parquet schema metadata is missing "
                f"{key.decode('ascii')!r}"
            )
        try:
            decoded[name] = raw_metadata[key].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"activation parquet schema metadata {key!r} is not ASCII"
            ) from error

    try:
        schema_version = int(decoded["schema_version"])
        stored_layer_index = int(decoded["layer_index"])
        stored_min_position = int(decoded["min_position"])
        stored_max_per_prompt = int(decoded["max_per_prompt"])
    except ValueError as error:
        raise ValueError(
            "activation parquet extraction integer metadata is malformed"
        ) from error
    stored_dtype = decoded["dtype"]
    if schema_version != 1:
        raise ValueError(
            "activation parquet extraction metadata schema_version must be 1"
        )
    if stored_layer_index < 0 or stored_min_position < 0:
        raise ValueError(
            "activation parquet extraction layer/min-position must be nonnegative"
        )
    if stored_max_per_prompt < 0:
        raise ValueError(
            "activation parquet extraction max-per-prompt must be nonnegative"
        )
    if stored_dtype not in EXTRACTION_DTYPES:
        raise ValueError(
            f"activation parquet extraction dtype is invalid: {stored_dtype!r}"
        )

    expected = {
        "layer_index": layer_index,
        "min_position": min_position,
        "max_per_prompt": max_per_prompt,
        "dtype": dtype,
    }
    observed = {
        "layer_index": stored_layer_index,
        "min_position": stored_min_position,
        "max_per_prompt": stored_max_per_prompt,
        "dtype": stored_dtype,
    }
    if observed != expected:
        mismatches = {
            key: {"parquet": observed[key], "cli": expected[key]}
            for key in expected
            if observed[key] != expected[key]
        }
        raise ValueError(
            "activation parquet extraction metadata differs from validator "
            f"CLI: {mismatches}"
        )
    return {
        "schema_version": schema_version,
        **observed,
        "verified_against_cli": True,
    }


def metadata_values(
    table: pa.Table,
    name: str,
) -> list[Any]:
    return table.column(name).combine_chunks().to_pylist()


def validate_activation_vectors(
    path: Path,
    expected_rows: int,
    configured_d_model: int,
    batch_size: int = 512,
) -> tuple[int, int]:
    parquet = pq.ParquetFile(path)
    observed_rows = 0
    observed_d_model: int | None = None
    for batch_index, batch in enumerate(
        parquet.iter_batches(
            batch_size=batch_size,
            columns=["activation_vector"],
        )
    ):
        values = batch.column(0)
        if values.null_count:
            raise ValueError(
                f"activation batch {batch_index} contains null vectors"
            )
        try:
            vectors = np.asarray(values.to_pylist(), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"activation batch {batch_index} contains ragged or invalid "
                "vectors"
            ) from error
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise ValueError(
                f"activation batch {batch_index} has invalid shape "
                f"{vectors.shape}"
            )
        if vectors.shape[1] <= 0:
            raise ValueError("activation vectors must be nonempty")
        if not np.all(np.isfinite(vectors)):
            raise ValueError(
                f"activation batch {batch_index} contains non-finite values"
            )
        if observed_d_model is None:
            observed_d_model = int(vectors.shape[1])
        elif vectors.shape[1] != observed_d_model:
            raise ValueError(
                "activation vectors have inconsistent dimensions: "
                f"{vectors.shape[1]} != {observed_d_model}"
            )
        observed_rows += int(vectors.shape[0])

    if observed_rows != expected_rows:
        raise ValueError(
            f"vector row count {observed_rows} differs from parquet metadata "
            f"row count {expected_rows}"
        )
    if observed_d_model is None:
        raise ValueError("activation parquet contains no vectors")
    if observed_d_model != configured_d_model:
        raise ValueError(
            f"activation d_model={observed_d_model} differs from base config "
            f"hidden_size={configured_d_model}"
        )
    return observed_rows, observed_d_model


def validate_activation_metadata(
    path: Path,
    manifest: list[dict[str, Any]],
    expected_split: str,
    expected_documents: int,
    min_position: int,
    layer_index: int,
    max_per_prompt: int,
    dtype: str,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    required = require_parquet_columns(parquet)
    extraction_parameters = validate_extraction_schema_metadata(
        parquet,
        layer_index=layer_index,
        min_position=min_position,
        max_per_prompt=max_per_prompt,
        dtype=dtype,
    )
    metadata_columns = sorted(required - {"activation_vector"})
    table = pq.read_table(path, columns=metadata_columns)
    n_rows = int(table.num_rows)
    if n_rows <= 0 or n_rows != parquet.metadata.num_rows:
        raise ValueError("activation parquet has an invalid metadata row count")

    values = {
        name: metadata_values(table, name) for name in metadata_columns
    }
    if any(
        item is None
        for name in metadata_columns
        for item in values[name]
    ):
        raise ValueError("activation metadata contains null values")
    doc_ids = values["doc_id"]
    positions = values["position"]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in doc_ids
    ):
        raise TypeError("activation doc_id values must be integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in positions
    ):
        raise TypeError("activation position values must be integers")

    observed_block_order: list[int] = []
    for doc_id in doc_ids:
        if not observed_block_order or doc_id != observed_block_order[-1]:
            observed_block_order.append(doc_id)
    expected_doc_ids = list(range(expected_documents))
    if observed_block_order != expected_doc_ids:
        raise ValueError(
            "activation rows are not ordered in exactly one contiguous block "
            f"per doc_id 0..{expected_documents - 1}: "
            f"observed block order={observed_block_order}"
        )
    unique_doc_ids = sorted(set(doc_ids))
    if unique_doc_ids != expected_doc_ids:
        raise ValueError(
            f"activation doc IDs must be contiguous 0.."
            f"{expected_documents - 1}"
        )

    observed_prompt_ids = set(values["prompt_id"])
    expected_prompt_ids = {row["prompt_id"] for row in manifest}
    if observed_prompt_ids != expected_prompt_ids:
        raise ValueError(
            "activation prompt IDs differ from manifest "
            f"(unexpected={sorted(observed_prompt_ids - expected_prompt_ids)}, "
            f"missing={sorted(expected_prompt_ids - observed_prompt_ids)})"
        )

    documents: list[dict[str, Any]] = []
    for doc_id, expected in enumerate(manifest):
        indices = [
            index
            for index, row_doc_id in enumerate(doc_ids)
            if row_doc_id == doc_id
        ]
        if not indices:
            raise ValueError(f"activation doc_id {doc_id} has no rows")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"activation doc_id {doc_id} is not contiguous")

        expected_fields = {
            "prompt_id": expected["prompt_id"],
            "prompt": expected["prompt"],
            "topic": expected["topic"],
            "axis_domain": expected["axis_domain"],
            "axis_language": expected["axis_language"],
            "split": expected["split"],
            "prompt_sha256": expected["prompt_sha256"],
        }
        for field, expected_value in expected_fields.items():
            observed = {values[field][index] for index in indices}
            if len(observed) != 1:
                raise ValueError(
                    f"activation doc_id {doc_id} has inconsistent {field}"
                )
            actual_value = next(iter(observed))
            if actual_value != expected_value:
                raise ValueError(
                    f"activation doc_id {doc_id} {field} differs from "
                    f"manifest: {actual_value!r} != {expected_value!r}"
                )

        prompt_sha = require_sha256(
            expected["prompt_sha256"],
            f"manifest prompt SHA for doc_id {doc_id}",
        )
        if prompt_sha != sha256_text(expected["prompt"]):
            raise ValueError(f"prompt SHA mismatch for doc_id {doc_id}")
        doc_positions = [positions[index] for index in indices]
        if not doc_positions:
            raise ValueError(f"activation doc_id {doc_id} has no positions")
        expected_positions = list(
            range(min_position, int(doc_positions[-1]) + 1)
        )
        if doc_positions != expected_positions:
            raise ValueError(
                f"activation doc_id {doc_id} positions must be exactly "
                f"contiguous from {min_position}; observed first/last/count="
                f"{doc_positions[0]}/{doc_positions[-1]}/{len(doc_positions)}"
            )
        if expected["split"] != expected_split:
            raise ValueError(
                f"manifest doc_id {doc_id} has unexpected split "
                f"{expected['split']!r}"
            )
        documents.append(
            {
                "doc_id": doc_id,
                "prompt_id": expected["prompt_id"],
                "prompt_sha256": prompt_sha,
                "topic": expected["topic"],
                "axis_domain": expected["axis_domain"],
                "axis_language": expected["axis_language"],
                "split": expected_split,
                "first_position": int(doc_positions[0]),
                "last_position": int(doc_positions[-1]),
                "activation_rows": len(doc_positions),
            }
        )

    if len(documents) != expected_documents:
        raise ValueError(
            f"parquet has {len(documents)} documents, expected "
            f"{expected_documents}"
        )
    return documents, n_rows, extraction_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--manual-audit", required=True, type=Path)
    parser.add_argument(
        "--expected-split",
        required=True,
        choices=("train", "test"),
    )
    parser.add_argument("--expected-documents", required=True, type=int)
    parser.add_argument("--layer-index", type=int, default=32)
    parser.add_argument("--min-position", type=int, default=50)
    parser.add_argument("--max-per-prompt", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(EXTRACTION_DTYPES),
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_documents <= 0:
        raise ValueError("--expected-documents must be positive")
    if args.layer_index < 0:
        raise ValueError("--layer-index must be nonnegative")
    if args.min_position < 0:
        raise ValueError("--min-position must be nonnegative")
    if args.max_per_prompt < 0:
        raise ValueError("--max-per-prompt must be nonnegative")

    manifest_path = require_file(args.manifest, "manifest")
    activation_path = require_file(args.activations, "activations")
    extractor_path = require_file(args.extractor, "extractor")
    audit_path = require_file(args.manual_audit, "manual audit")
    model_directory = require_directory(args.base_model, "base model")
    output_path = args.out.expanduser().resolve()
    input_paths = {
        manifest_path,
        activation_path,
        extractor_path,
        audit_path,
    }
    if output_path in input_paths:
        raise ValueError("--out must not overwrite an input file")

    manifest_sha = sha256_file(manifest_path)
    audit_sha = sha256_file(audit_path)
    extractor_sha = sha256_file(extractor_path)
    manual_audit = validate_manual_audit(
        audit_path,
        args.expected_split,
        manifest_sha,
    )
    manifest = load_manifest(
        manifest_path,
        args.expected_split,
        args.expected_documents,
    )
    (
        documents,
        activation_rows,
        extraction_parameters,
    ) = validate_activation_metadata(
        activation_path,
        manifest,
        args.expected_split,
        args.expected_documents,
        args.min_position,
        args.layer_index,
        args.max_per_prompt,
        args.dtype,
    )
    base_model, configured_d_model = fingerprint_base_model(
        model_directory,
        args.layer_index,
    )
    vector_rows, d_model = validate_activation_vectors(
        activation_path,
        activation_rows,
        configured_d_model,
    )
    if vector_rows != activation_rows:
        raise ValueError("activation vector and metadata row counts differ")
    activation_sha = sha256_file(activation_path)

    identity_records = [
        {
            "doc_id": doc_id,
            **row,
        }
        for doc_id, row in enumerate(manifest)
    ]
    report = {
        "schema_version": 1,
        "experiment": "C1 activation provenance validation",
        "status": "PASS",
        "inputs": {
            "manifest": str(manifest_path),
            "activations": str(activation_path),
            "base_model": str(model_directory),
            "extractor": str(extractor_path),
            "manual_audit": str(audit_path),
        },
        "parameters": {
            "expected_split": args.expected_split,
            "expected_documents": args.expected_documents,
            "layer_index": extraction_parameters["layer_index"],
            "min_position": extraction_parameters["min_position"],
            "max_per_prompt": extraction_parameters["max_per_prompt"],
            "dtype": extraction_parameters["dtype"],
            "parquet_extraction_metadata_schema_version": (
                extraction_parameters["schema_version"]
            ),
            "verified_against_parquet_schema_metadata": (
                extraction_parameters["verified_against_cli"]
            ),
        },
        "counts": {
            "manifest_documents": len(manifest),
            "parquet_documents": len(documents),
            "activation_rows": activation_rows,
            "d_model": d_model,
        },
        "hashes": {
            "manifest_sha256": manifest_sha,
            "activations_sha256": activation_sha,
            "extractor_sha256": extractor_sha,
            "manual_audit_sha256": audit_sha,
            "base_model_identity_sha256": base_model["identity_sha256"],
            "document_identity_sha256": sha256_text(
                canonical_json(identity_records)
            ),
        },
        "manual_audit": manual_audit,
        "base_model": base_model,
        "documents": documents,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output_path)
    print("C1_ACTIVATION_PROVENANCE_PASS")
    print(
        json.dumps(
            {
                "split": args.expected_split,
                "documents": len(documents),
                "rows": activation_rows,
                "d_model": d_model,
                "manifest_sha256": manifest_sha,
                "activations_sha256": activation_sha,
                "base_model_identity_sha256": (
                    base_model["identity_sha256"]
                ),
                "out": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
