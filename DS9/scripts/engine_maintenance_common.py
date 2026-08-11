#!/usr/bin/env python3
"""Fail-closed, auditable TensorRT engine installation primitives for DS9."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from array import array
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.strict_json import StrictJSONError, strict_json_loads  # noqa: E402


CONTRACT = "noesis.ds9.engine_maintenance"
SCHEMA_VERSION = 1
DS9_TRTEXEC_BANNER = "TensorRT v101401"
MAPANYTHING_QUALITY_CONTRACT = "noesis.ds9.mapanything_functional_quality.v1"
MAPANYTHING_QUALITY_EVIDENCE_LABEL = "functional_quality"
MAPANYTHING_QUALITY_COMMAND_LABEL = "mapanything-functional-quality"
MAPANYTHING_QUALITY_OUTPUT_NAME = "mapanything-functional-output.json"
MAPANYTHING_QUALITY_OUTPUT_MAX_BYTES = 32 * 1024 * 1024
_MAPANYTHING_QUALITY_ARGS = (
    "--iterations=1",
    "--duration=0",
    "--warmUp=0",
    "--avgRuns=1",
)
_MAPANYTHING_OUTPUT_NAMES = ("conf", "depth", "mask")
_MAPANYTHING_DISTRIBUTION_METRICS = {
    "min",
    "max",
    "mean",
    "std",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
}
_MAPANYTHING_PLATFORM_KEYS = {
    "image",
    "image_id",
    "base_digest",
    "tensorrt_version",
    "cuda_version",
    "driver_version",
    "gpu_name",
    "gpu_uuid",
    "gpu_compute_capability",
    "gpu_memory_mib",
    "expected_trtexec_banner",
}
_PROVENANCE_ENV = {
    "NOESIS_DS9_MAINT_IMAGE": "image",
    "NOESIS_DS9_MAINT_IMAGE_ID": "image_id",
    "NOESIS_DS9_MAINT_BASE_DIGEST": "base_digest",
    "NOESIS_DS9_MAINT_TRT_VERSION": "tensorrt_version",
    "NOESIS_DS9_MAINT_CUDA_VERSION": "cuda_version",
    "NOESIS_DS9_MAINT_DRIVER_VERSION": "driver_version",
    "NOESIS_DS9_MAINT_GPU_NAME": "gpu_name",
    "NOESIS_DS9_MAINT_GPU_UUID": "gpu_uuid",
    "NOESIS_DS9_MAINT_GPU_COMPUTE_CAPABILITY": "gpu_compute_capability",
    "NOESIS_DS9_MAINT_GPU_MEMORY_MIB": "gpu_memory_mib",
}

_NEGATIVE_TRTEXEC_PATTERNS = (
    re.compile(r"\[E\]", re.IGNORECASE),
    re.compile(r"\bError\[\d+\]", re.IGNORECASE),
    re.compile(r"&&&&\s+FAILED\s+TensorRT\.trtexec", re.IGNORECASE),
    re.compile(r"engine\s+deserialization\s+failed", re.IGNORECASE),
    re.compile(r"(?:failed|unable|could not)\s+to\s+deserialize", re.IGNORECASE),
    re.compile(r"\bModel\s+missing\b", re.IGNORECASE),
)
_TRTEXEC_PASSED = re.compile(r"&&&&\s+PASSED\s+TensorRT\.trtexec", re.IGNORECASE)
_LOADED_ENGINE = re.compile(r"Loaded\s+engine\s+size:\s*(\d+)\s*MiB", re.IGNORECASE)
_DESERIALIZED = re.compile(r"Engine\s+deserialized\s+in\s+[0-9.eE+-]+\s+sec", re.IGNORECASE)
_SKIPPED_INFERENCE = re.compile(
    r"Skipped\s+inference\s+phase\s+since\s+--skipInference\s+is\s+added",
    re.IGNORECASE,
)
_STARTED_INFERENCE = re.compile(r"\bStarting\s+inference\b", re.IGNORECASE)
_ONE_QUERY_TRACE = re.compile(
    r"Timing\s+trace\s+has\s+1\s+queries\b", re.IGNORECASE
)
_WHOLEBODY_BUILDER_CONTRACT = (
    "[NOESIS_TRT_BUILDER] contract=noesis.ds9.wholebody49_builder.v1"
)
_WHOLEBODY_BUILDER_REQUIRED_LINES = (
    _WHOLEBODY_BUILDER_CONTRACT,
    "[NOESIS_TRT_BUILDER] network_mode=explicit_batch_trt10_default",
    "[NOESIS_TRT_BUILDER] profile=images:3x3x640x640",
    "[NOESIS_TRT_BUILDER] tactic_dram_bytes=2147483648",
    "[NOESIS_TRT_BUILDER] logger_minimum_severity=info",
    "[NOESIS_TRT_BUILDER] logger_verbose_policy=ignored_before_copy",
    "[NOESIS_TRT_BUILDER] logger_captured_truncation=fatal",
    "[NOESIS_TRT_BUILDER] logger_error_state=sticky_fatal",
    "[NOESIS_TRT_BUILDER] status=PASS",
)
_WHOLEBODY_WORKSPACE_BYTES_BY_VARIANT = {
    "s_masks": 6442450944,
    "x_boxes": 4294967296,
}
_WHOLEBODY_BUILDER_OPTIMIZATION_LEVEL_BY_VARIANT = {
    "s_masks": 0,
    "x_boxes": 3,
}
_WHOLEBODY_LOGGER_POLICY = {
    "minimum_severity": "info",
    "verbose": "ignored_before_copy",
    "captured_message_truncation": "fatal",
    "error_state": "sticky_fatal",
}
_WHOLEBODY_BUILDER_ENGINE_BYTES = re.compile(
    r"^\[NOESIS_TRT_BUILDER\] engine_bytes=([1-9][0-9]*)$", re.MULTILINE
)
_WHOLEBODY_BUILDER_FAILURE = re.compile(
    r"^\[NOESIS_TRT_BUILDER\] status=FAIL(?:\s|$)", re.MULTILINE
)


class EngineMaintenanceError(RuntimeError):
    """Raised when a guarded engine-maintenance contract fails."""


def validate_wholebody_memory_pool_limits(
    limits: object,
) -> dict[str, int]:
    """Validate the exact Wholebody TensorRT pool-size authority."""

    if not isinstance(limits, Mapping) or set(limits) != {
        "workspace",
        "tactic_dram",
    }:
        raise EngineMaintenanceError(
            "Wholebody49 memory-pool limits must name workspace and tactic_dram"
        )
    workspace = limits.get("workspace")
    if (
        isinstance(workspace, bool)
        or not isinstance(workspace, int)
        or workspace <= 0
        or workspace % (1024 * 1024) != 0
    ):
        raise EngineMaintenanceError(
            "Wholebody49 workspace memory-pool limit must be positive and MiB-aligned"
        )
    tactic_dram = limits.get("tactic_dram")
    if (
        isinstance(tactic_dram, bool)
        or not isinstance(tactic_dram, int)
        or tactic_dram <= 0
        or tactic_dram & (tactic_dram - 1)
    ):
        raise EngineMaintenanceError(
            "Wholebody49 tactic_dram memory-pool limit must be a positive power of two"
        )
    validated = {"workspace": workspace, "tactic_dram": tactic_dram}
    return validated


def validate_wholebody_builder_optimization_level(value: object) -> int:
    """Validate an exact TensorRT 10.14 builder optimization level."""

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
        raise EngineMaintenanceError(
            "Wholebody49 builder optimization level must be an integer from 0 to 5"
        )
    return value


def validate_wholebody_logger_policy(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or dict(value) != _WHOLEBODY_LOGGER_POLICY:
        raise EngineMaintenanceError(
            "Wholebody49 logger policy must capture INFO and higher, ignore "
            "VERBOSE before bounded copy, and fail on captured truncation or "
            "sticky errors"
        )
    return dict(_WHOLEBODY_LOGGER_POLICY)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_bundle_files(path: Path) -> list[tuple[str, Path]]:
    required_regular_file(path, "input bundle root")
    rows: list[tuple[str, Path]] = [("main", path)]
    if path.suffix.lower() != ".onnx":
        return rows
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    locations: set[Path] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {item.key: item.value for item in tensor.external_data}
        raw = str(metadata.get("location", "") or "").strip()
        relative = Path(raw)
        if not raw or relative.is_absolute() or ".." in relative.parts:
            raise EngineMaintenanceError(
                f"unsafe ONNX external-data location in {path}: {raw!r}"
            )
        locations.add(relative)
    for relative in sorted(locations, key=lambda value: value.as_posix()):
        sidecar = path.parent / relative
        required_regular_file(sidecar, f"ONNX external data {relative}")
        rows.append((f"external::{relative.as_posix()}", sidecar))
    return rows


def input_bundle_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files: list[dict[str, object]] = []
    for label, member in input_bundle_files(path):
        member_hash = sha256_file(member)
        member_size = member.stat().st_size
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
        files.append(
            {
                "label": label,
                "path": str(member.resolve()),
                "size_bytes": member_size,
                "sha256": member_hash,
            }
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "bundle_sha256": digest.hexdigest(),
        "files": files,
    }


def onnx_contract(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)

    def describe(value: Any) -> dict[str, Any]:
        tensor = value.type.tensor_type
        dimensions: list[int | str] = []
        for dimension in tensor.shape.dim:
            if dimension.dim_value:
                dimensions.append(int(dimension.dim_value))
            else:
                dimensions.append(str(dimension.dim_param or "?"))
        return {
            "name": str(value.name),
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": dimensions,
        }

    return {
        "opsets": [
            {"domain": str(row.domain or "ai.onnx"), "version": int(row.version)}
            for row in model.opset_import
        ],
        "inputs": [describe(row) for row in model.graph.input],
        "outputs": [describe(row) for row in model.graph.output],
        "external_initializer_count": sum(
            row.data_location == onnx.TensorProto.EXTERNAL
            for row in model.graph.initializer
        ),
    }


def load_source_contracts(path: Path) -> dict[str, Mapping[str, Any]]:
    try:
        payload = strict_json_loads(
            path.read_bytes(),
            label="engine source contract authority",
        )
    except (OSError, StrictJSONError) as exc:
        raise EngineMaintenanceError(
            f"invalid engine source contract file: {path}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get(
        "schema_version"
    ) != 1 or not isinstance(
        payload.get("contracts"), dict
    ):
        raise EngineMaintenanceError(f"invalid engine source contract file: {path}")
    return dict(payload["contracts"])


def validate_source_contract(
    name: str, path: Path, contracts_path: Path
) -> dict[str, object]:
    contracts = load_source_contracts(contracts_path)
    expected = contracts.get(name)
    if not isinstance(expected, Mapping):
        raise EngineMaintenanceError(f"missing pinned source contract for {name}")
    record = input_bundle_record(path)
    if record["sha256"] != expected.get("raw_sha256"):
        raise EngineMaintenanceError(
            f"pinned raw SHA-256 mismatch for {name}: "
            f"expected={expected.get('raw_sha256')} observed={record['sha256']}"
        )
    if record["bundle_sha256"] != expected.get("bundle_sha256"):
        raise EngineMaintenanceError(
            f"pinned bundle SHA-256 mismatch for {name}: "
            f"expected={expected.get('bundle_sha256')} observed={record['bundle_sha256']}"
        )
    expected_onnx = expected.get("onnx")
    if expected_onnx is not None:
        actual_onnx = onnx_contract(path)
        if actual_onnx != expected_onnx:
            raise EngineMaintenanceError(
                f"pinned ONNX tensor contract mismatch for {name}: "
                f"expected={expected_onnx!r} observed={actual_onnx!r}"
            )
        record["onnx"] = actual_onnx
    tensor_contract = expected.get("tensor_contract")
    if tensor_contract is not None:
        record["tensor_contract"] = tensor_contract
    return record


def _json_copy(value: object, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
        )
    except (TypeError, ValueError) as exc:
        raise EngineMaintenanceError(f"{label} is not finite JSON: {exc}") from exc


def _exact_mapping(
    value: object,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise EngineMaintenanceError(f"{label} has unexpected or missing fields")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EngineMaintenanceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EngineMaintenanceError(f"{label} must be finite")
    return result


def _quality_range(value: object, label: str) -> dict[str, float]:
    record = _exact_mapping(value, {"min", "max"}, label)
    lower = _finite_number(record["min"], f"{label}.min")
    upper = _finite_number(record["max"], f"{label}.max")
    if lower > upper:
        raise EngineMaintenanceError(f"{label} minimum exceeds its maximum")
    return {"min": lower, "max": upper}


def validate_mapanything_quality_gate_authority(
    value: object,
) -> dict[str, Any]:
    """Validate and canonicalize the tracked functional admission authority."""

    gate = _exact_mapping(
        value,
        {"contract", "fixture", "trtexec_args", "outputs"},
        "MapAnything quality gate",
    )
    if gate.get("contract") != MAPANYTHING_QUALITY_CONTRACT:
        raise EngineMaintenanceError("MapAnything quality-gate contract is invalid")

    fixture = _exact_mapping(
        gate.get("fixture"),
        {
            "artifact_relative_path",
            "sha256",
            "size_bytes",
            "tensor",
            "identical_batch_members",
        },
        "MapAnything quality fixture",
    )
    relative_raw = str(fixture.get("artifact_relative_path") or "")
    relative = Path(relative_raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or tuple(relative.parts[:3])
        != ("models", "engine_validation", "mapanything")
        or relative.name == ""
        or relative.as_posix() != relative_raw
    ):
        raise EngineMaintenanceError(
            "MapAnything quality fixture must stay below "
            "models/engine_validation/mapanything"
        )
    fixture_sha = str(fixture.get("sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", fixture_sha) is None:
        raise EngineMaintenanceError("MapAnything quality fixture SHA-256 is invalid")
    tensor = _exact_mapping(
        fixture.get("tensor"), {"name", "dtype", "shape"}, "quality input tensor"
    )
    shape = tensor.get("shape")
    if (
        tensor.get("name") != "images"
        or tensor.get("dtype") != "float32"
        or not isinstance(shape, list)
        or len(shape) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        or shape[0] < 2
        or shape[1] != 3
    ):
        raise EngineMaintenanceError(
            "MapAnything quality input must be float32 images with batch >= 2"
        )
    expected_fixture_size = math.prod(shape) * 4
    if (
        isinstance(fixture.get("size_bytes"), bool)
        or fixture.get("size_bytes") != expected_fixture_size
        or fixture.get("identical_batch_members") is not True
    ):
        raise EngineMaintenanceError(
            "MapAnything quality fixture size/batch-consistency authority is invalid"
        )

    trtexec_args = gate.get("trtexec_args")
    if not isinstance(trtexec_args, list) or tuple(trtexec_args) != _MAPANYTHING_QUALITY_ARGS:
        raise EngineMaintenanceError(
            "MapAnything quality inference arguments differ from reviewed authority"
        )
    if any(not isinstance(item, str) or not item for item in trtexec_args):
        raise EngineMaintenanceError("MapAnything quality inference arguments are invalid")

    outputs = gate.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(_MAPANYTHING_OUTPUT_NAMES):
        raise EngineMaintenanceError(
            "MapAnything quality outputs must be exactly conf, depth, and mask"
        )
    canonical_outputs: dict[str, Any] = {}
    for name in _MAPANYTHING_OUTPUT_NAMES:
        required = {
            "dtype",
            "shape",
            "finite_fraction",
            "positive_fraction",
            "distribution",
            "batch_max_abs_delta",
        }
        if name == "conf":
            required.add("forbidden_values")
        if name == "mask":
            required.update({"mask_threshold", "mask_coverage"})
        output = _exact_mapping(outputs[name], required, f"quality output {name}")
        output_shape = output.get("shape")
        if (
            output.get("dtype") != "float32"
            or output_shape != [shape[0], 1, shape[2], shape[3]]
        ):
            raise EngineMaintenanceError(
                f"MapAnything quality output {name} tensor contract is invalid"
            )
        finite_fraction = _quality_range(
            output.get("finite_fraction"), f"quality output {name} finite fraction"
        )
        positive_fraction = _quality_range(
            output.get("positive_fraction"),
            f"quality output {name} positive fraction",
        )
        for label, bounds in (
            ("finite", finite_fraction),
            ("positive", positive_fraction),
        ):
            if bounds["min"] < 0.0 or bounds["max"] > 1.0:
                raise EngineMaintenanceError(
                    f"quality output {name} {label} fraction leaves [0, 1]"
                )
        if finite_fraction != {"min": 1.0, "max": 1.0}:
            raise EngineMaintenanceError(
                f"quality output {name} must require every value to be finite"
            )
        if name == "depth" and positive_fraction["min"] < 0.99:
            raise EngineMaintenanceError(
                "MapAnything depth quality must require positive output"
            )

        distribution = output.get("distribution")
        if (
            not isinstance(distribution, Mapping)
            or not distribution
            or not set(distribution) <= _MAPANYTHING_DISTRIBUTION_METRICS
            or not {"min", "max", "mean", "p50"} <= set(distribution)
        ):
            raise EngineMaintenanceError(
                f"quality output {name} distribution authority is invalid"
            )
        canonical_distribution = {
            metric: _quality_range(
                distribution[metric], f"quality output {name} distribution {metric}"
            )
            for metric in sorted(distribution)
        }
        batch_delta = _finite_number(
            output.get("batch_max_abs_delta"),
            f"quality output {name} batch maximum delta",
        )
        if batch_delta < 0.0 or batch_delta > 1.0:
            raise EngineMaintenanceError(
                f"quality output {name} batch maximum delta is invalid"
            )
        canonical: dict[str, Any] = {
            "dtype": "float32",
            "shape": list(output_shape),
            "finite_fraction": finite_fraction,
            "positive_fraction": positive_fraction,
            "distribution": canonical_distribution,
            "batch_max_abs_delta": batch_delta,
        }
        if name == "conf":
            forbidden = output.get("forbidden_values")
            if not isinstance(forbidden, list) or not forbidden:
                raise EngineMaintenanceError(
                    "MapAnything confidence must declare forbidden sentinels"
                )
            canonical["forbidden_values"] = [
                _finite_number(item, "quality confidence forbidden value")
                for item in forbidden
            ]
        if name == "mask":
            threshold = _finite_number(
                output.get("mask_threshold"), "quality mask threshold"
            )
            coverage = _quality_range(
                output.get("mask_coverage"), "quality mask coverage"
            )
            if not 0.0 < threshold <= 1.0 or not (
                0.0 < coverage["min"] <= coverage["max"] <= 1.0
            ):
                raise EngineMaintenanceError(
                    "MapAnything mask threshold/coverage authority is invalid"
                )
            canonical["mask_threshold"] = threshold
            canonical["mask_coverage"] = coverage
        canonical_outputs[name] = canonical

    canonical_gate = {
        "contract": MAPANYTHING_QUALITY_CONTRACT,
        "fixture": {
            "artifact_relative_path": relative.as_posix(),
            "sha256": fixture_sha,
            "size_bytes": expected_fixture_size,
            "tensor": {
                "name": "images",
                "dtype": "float32",
                "shape": list(shape),
            },
            "identical_batch_members": True,
        },
        "trtexec_args": list(_MAPANYTHING_QUALITY_ARGS),
        "outputs": canonical_outputs,
    }
    return _json_copy(canonical_gate, "MapAnything quality-gate authority")


def mapanything_quality_gate_from_source_contracts(path: Path) -> dict[str, Any]:
    contracts = load_source_contracts(path)
    source = contracts.get("mapanything")
    if not isinstance(source, Mapping) or "quality_gate" not in source:
        raise EngineMaintenanceError(
            "MapAnything source authority lacks the mandatory functional quality gate"
        )
    return validate_mapanything_quality_gate_authority(source["quality_gate"])


def _private_quality_file(path: Path, label: str, *, max_bytes: int) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise EngineMaintenanceError(f"{label} must be a regular non-symlink file")
    info = path.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size <= 0
        or info.st_size > max_bytes
    ):
        raise EngineMaintenanceError(
            f"{label} must be owner-private, single-link, nonempty, and bounded"
        )
    return info


def _quality_fixture_record(path: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    fixture = authority["fixture"]
    info = _private_quality_file(
        path, "MapAnything quality fixture", max_bytes=int(fixture["size_bytes"])
    )
    if info.st_size != fixture["size_bytes"] or sha256_file(path) != fixture["sha256"]:
        raise EngineMaintenanceError(
            "MapAnything quality fixture differs from tracked authority"
        )
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    shape = fixture["tensor"]["shape"]
    expected_count = math.prod(shape)
    if len(values) != expected_count:
        raise EngineMaintenanceError("MapAnything quality fixture element count drifted")
    per_batch = expected_count // shape[0]
    first = values[:per_batch]
    if any(values[index * per_batch : (index + 1) * per_batch] != first for index in range(1, shape[0])):
        raise EngineMaintenanceError(
            "MapAnything quality fixture batch members are not identical"
        )
    numeric = [float(item) for item in values]
    if any(not math.isfinite(item) for item in numeric):
        raise EngineMaintenanceError("MapAnything quality fixture contains non-finite data")
    return {
        "sha256": fixture["sha256"],
        "size_bytes": info.st_size,
        "tensor": _json_copy(fixture["tensor"], "quality fixture tensor"),
        "identical_batch_members": True,
        "finite_count": len(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": math.fsum(numeric) / len(numeric),
    }


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _distribution_metrics(
    values: list[float], requested: set[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    if "min" in requested:
        result["min"] = min(values)
    if "max" in requested:
        result["max"] = max(values)
    mean = math.fsum(values) / len(values)
    if "mean" in requested:
        result["mean"] = mean
    if "std" in requested:
        result["std"] = math.sqrt(
            math.fsum((value - mean) ** 2 for value in values) / len(values)
        )
    percentile_quantiles = {
        "p01": 0.01,
        "p05": 0.05,
        "p50": 0.50,
        "p95": 0.95,
        "p99": 0.99,
    }
    selected = requested & set(percentile_quantiles)
    if selected:
        ordered = sorted(values)
        for name in sorted(selected):
            result[name] = _percentile(ordered, percentile_quantiles[name])
    return result


def _require_in_envelope(value: float, bounds: Mapping[str, Any], label: str) -> None:
    if value < float(bounds["min"]) or value > float(bounds["max"]):
        raise EngineMaintenanceError(
            f"{label} is outside the pinned reference envelope: {value} not in "
            f"[{bounds['min']}, {bounds['max']}]"
        )


def evaluate_mapanything_quality_output(
    output_path: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse one real trtexec export and enforce every functional bound."""

    canonical = validate_mapanything_quality_gate_authority(authority)
    _private_quality_file(
        output_path,
        "MapAnything trtexec output",
        max_bytes=MAPANYTHING_QUALITY_OUTPUT_MAX_BYTES,
    )
    try:
        payload = strict_json_loads(
            output_path.read_bytes(),
            label="MapAnything quality output",
        )
    except (OSError, StrictJSONError) as exc:
        raise EngineMaintenanceError(
            "MapAnything quality output is not strict JSON"
        ) from exc
    if not isinstance(payload, list) or len(payload) != len(_MAPANYTHING_OUTPUT_NAMES):
        raise EngineMaintenanceError(
            "MapAnything quality output must contain exactly three tensors"
        )
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in payload:
        row = _exact_mapping(raw, {"name", "dimensions", "values"}, "quality tensor")
        name = str(row.get("name") or "")
        if name in rows:
            raise EngineMaintenanceError(f"duplicate MapAnything quality tensor: {name}")
        rows[name] = row
    if set(rows) != set(_MAPANYTHING_OUTPUT_NAMES):
        raise EngineMaintenanceError(
            "MapAnything quality tensor names differ from conf/depth/mask"
        )

    observed: dict[str, Any] = {}
    for name in _MAPANYTHING_OUTPUT_NAMES:
        expected = canonical["outputs"][name]
        row = rows[name]
        dimensions = "x".join(str(item) for item in expected["shape"])
        if row.get("dimensions") != dimensions:
            raise EngineMaintenanceError(
                f"MapAnything {name} shape differs from {dimensions}"
            )
        raw_values = row.get("values")
        expected_count = math.prod(expected["shape"])
        if not isinstance(raw_values, list) or len(raw_values) != expected_count:
            raise EngineMaintenanceError(
                f"MapAnything {name} value count differs from its tensor shape"
            )
        values: list[float] = []
        for raw_value in raw_values:
            value = _finite_number(raw_value, f"MapAnything {name} output value")
            values.append(value)
        for sentinel in expected.get("forbidden_values", []):
            if any(value == sentinel for value in values):
                raise EngineMaintenanceError(
                    f"MapAnything {name} output contains forbidden sentinel {sentinel}"
                )
        total = len(values)
        positive = sum(value > 0.0 for value in values)
        zero = sum(value == 0.0 for value in values)
        negative = total - positive - zero
        finite_fraction = 1.0
        positive_fraction = positive / total
        _require_in_envelope(
            finite_fraction, expected["finite_fraction"], f"MapAnything {name} finite fraction"
        )
        _require_in_envelope(
            positive_fraction,
            expected["positive_fraction"],
            f"MapAnything {name} positive fraction",
        )
        distribution = _distribution_metrics(values, set(expected["distribution"]))
        for metric, value in distribution.items():
            _require_in_envelope(
                value,
                expected["distribution"][metric],
                f"MapAnything {name} distribution {metric}",
            )
        per_batch = total // expected["shape"][0]
        first = values[:per_batch]
        batch_max_abs_delta = 0.0
        for index in range(1, expected["shape"][0]):
            current = values[index * per_batch : (index + 1) * per_batch]
            batch_max_abs_delta = max(
                batch_max_abs_delta,
                max(abs(left - right) for left, right in zip(first, current)),
            )
        if batch_max_abs_delta > expected["batch_max_abs_delta"]:
            raise EngineMaintenanceError(
                f"MapAnything {name} batch consistency exceeded the pinned limit"
            )
        record: dict[str, Any] = {
            "dtype": expected["dtype"],
            "shape": list(expected["shape"]),
            "total_count": total,
            "finite_count": total,
            "positive_count": positive,
            "zero_count": zero,
            "negative_count": negative,
            "finite_fraction": finite_fraction,
            "positive_fraction": positive_fraction,
            "distribution": distribution,
            "batch_max_abs_delta": batch_max_abs_delta,
        }
        if name == "mask":
            threshold = expected["mask_threshold"]
            coverage_count = sum(value >= threshold for value in values)
            coverage = coverage_count / total
            _require_in_envelope(
                coverage, expected["mask_coverage"], "MapAnything mask coverage"
            )
            record.update(
                {
                    "mask_threshold": threshold,
                    "mask_coverage_count": coverage_count,
                    "mask_coverage": coverage,
                }
            )
        observed[name] = record
    return observed


def _quality_platform(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _MAPANYTHING_PLATFORM_KEYS:
        raise EngineMaintenanceError(
            "MapAnything quality runtime platform is incomplete or has drifted"
        )
    result = {key: str(value.get(key) or "").strip() for key in sorted(value)}
    if any(not item for item in result.values()):
        raise EngineMaintenanceError("MapAnything quality runtime platform is incomplete")
    if result["expected_trtexec_banner"] != DS9_TRTEXEC_BANNER:
        raise EngineMaintenanceError("MapAnything quality runtime TensorRT banner drifted")
    return result


def _quality_command(
    *,
    executable: str,
    engine_path: Path,
    fixture_path: Path,
    output_path: Path,
    authority: Mapping[str, Any],
) -> list[str]:
    if Path(executable).name != "trtexec" or not executable:
        raise EngineMaintenanceError("MapAnything quality executable must be trtexec")
    command = [
        executable,
        f"--loadEngine={engine_path}",
        f"--loadInputs=images:{fixture_path}",
        *authority["trtexec_args"],
        "--dumpOutput",
        f"--exportOutput={output_path}",
    ]
    if any(not item for item in command) or any("--skipInference" in item for item in command):
        raise EngineMaintenanceError(
            "MapAnything functional command must execute real inference"
        )
    return command


def _quality_receipt_digest(receipt: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_mapanything_functional_quality_receipt(
    *,
    candidate: Path,
    fixture_path: Path,
    output_path: Path,
    evidence_directory: Path,
    authority: Mapping[str, Any],
    platform: Mapping[str, Any],
    command_record: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = validate_mapanything_quality_gate_authority(authority)
    required_regular_file(candidate, "MapAnything quality candidate")
    fixture_record = _quality_fixture_record(fixture_path, canonical)
    if output_path.parent != evidence_directory or output_path.name != MAPANYTHING_QUALITY_OUTPUT_NAME:
        raise EngineMaintenanceError(
            "MapAnything quality output must stay in its maintenance evidence directory"
        )
    outputs = evaluate_mapanything_quality_output(output_path, canonical)
    command = command_record.get("command")
    expected_command = _quality_command(
        executable=str(command[0]) if isinstance(command, list) and command else "",
        engine_path=candidate,
        fixture_path=fixture_path,
        output_path=output_path,
        authority=canonical,
    )
    if (
        command_record.get("label") != MAPANYTHING_QUALITY_COMMAND_LABEL
        or command_record.get("proof") != "trtexec_inference"
        or command_record.get("status") != "passed"
        or command_record.get("returncode") != 0
        or command_record.get("timed_out") is not False
        or command_record.get("output_exceeded") is not False
        or command != expected_command
    ):
        raise EngineMaintenanceError(
            "MapAnything quality command lacks exact real-inference proof"
        )
    output_info = output_path.stat()
    receipt: dict[str, Any] = {
        "contract": MAPANYTHING_QUALITY_CONTRACT,
        "status": "passed",
        "engine": {
            "path": str(candidate),
            "sha256": sha256_file(candidate),
            "size_bytes": candidate.stat().st_size,
        },
        "fixture": {
            "path": str(fixture_path),
            "artifact_relative_path": canonical["fixture"]["artifact_relative_path"],
            **fixture_record,
        },
        "command": list(expected_command),
        "runtime": _quality_platform(platform),
        "output_evidence": {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "size_bytes": output_info.st_size,
        },
        "outputs": outputs,
        "reference_envelope": canonical,
        "passed_at_utc": utc_now(),
    }
    receipt["receipt_sha256"] = _quality_receipt_digest(receipt)
    return _json_copy(receipt, "MapAnything functional quality receipt")


def validate_mapanything_functional_quality_receipt(
    maintenance_payload: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    expected_engine_path: Path,
    evidence_directory: Path,
    fixture_path: Path | None = None,
) -> dict[str, Any]:
    """Independently re-evaluate a receipt against current installed bytes."""

    canonical = validate_mapanything_quality_gate_authority(authority)
    evidence = maintenance_payload.get("evidence")
    receipt = (
        evidence.get(MAPANYTHING_QUALITY_EVIDENCE_LABEL)
        if isinstance(evidence, Mapping)
        else None
    )
    if not isinstance(receipt, Mapping):
        raise EngineMaintenanceError(
            "MapAnything installation requires a functional quality receipt"
        )
    if (
        receipt.get("contract") != MAPANYTHING_QUALITY_CONTRACT
        or receipt.get("status") != "passed"
        or receipt.get("reference_envelope") != canonical
        or receipt.get("receipt_sha256") != _quality_receipt_digest(receipt)
    ):
        raise EngineMaintenanceError("MapAnything functional quality receipt is invalid")
    metadata = maintenance_payload.get("metadata")
    platform = metadata.get("platform") if isinstance(metadata, Mapping) else None
    declared_authority = (
        metadata.get("quality_gate_authority")
        if isinstance(metadata, Mapping)
        else None
    )
    if declared_authority != canonical:
        raise EngineMaintenanceError(
            "MapAnything maintenance quality authority differs from tracked authority"
        )
    if receipt.get("runtime") != _quality_platform(platform):
        raise EngineMaintenanceError(
            "MapAnything quality receipt runtime differs from maintenance provenance"
        )
    engine = receipt.get("engine")
    if not isinstance(engine, Mapping):
        raise EngineMaintenanceError("MapAnything quality receipt lacks engine identity")
    required_regular_file(expected_engine_path, "MapAnything admitted engine")
    if (
        engine.get("sha256") != sha256_file(expected_engine_path)
        or engine.get("size_bytes") != expected_engine_path.stat().st_size
    ):
        raise EngineMaintenanceError(
            "MapAnything quality receipt engine hash/size differs from candidate"
        )
    candidate = maintenance_payload.get("candidate")
    if not isinstance(candidate, Mapping) or any(
        candidate.get(key) != engine.get(key) for key in ("path", "sha256", "size_bytes")
    ):
        raise EngineMaintenanceError(
            "MapAnything quality receipt differs from the recorded candidate"
        )

    fixture = receipt.get("fixture")
    if not isinstance(fixture, Mapping):
        raise EngineMaintenanceError("MapAnything quality receipt lacks fixture identity")
    effective_fixture = fixture_path or Path(str(fixture.get("path") or ""))
    observed_fixture = _quality_fixture_record(effective_fixture, canonical)
    for key, value in observed_fixture.items():
        if fixture.get(key) != value:
            raise EngineMaintenanceError(
                f"MapAnything quality receipt fixture {key} differs from evidence"
            )
    if fixture.get("artifact_relative_path") != canonical["fixture"]["artifact_relative_path"]:
        raise EngineMaintenanceError("MapAnything quality receipt fixture path drifted")

    output_evidence = receipt.get("output_evidence")
    if not isinstance(output_evidence, Mapping) or set(output_evidence) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise EngineMaintenanceError("MapAnything quality output evidence is malformed")
    relative_output = Path(str(output_evidence.get("path") or ""))
    if relative_output != Path(MAPANYTHING_QUALITY_OUTPUT_NAME):
        raise EngineMaintenanceError("MapAnything quality output evidence path drifted")
    output_path = evidence_directory / relative_output
    output_info = _private_quality_file(
        output_path,
        "MapAnything quality output evidence",
        max_bytes=MAPANYTHING_QUALITY_OUTPUT_MAX_BYTES,
    )
    if (
        output_evidence.get("sha256") != sha256_file(output_path)
        or output_evidence.get("size_bytes") != output_info.st_size
    ):
        raise EngineMaintenanceError("MapAnything quality output evidence changed")
    observed_outputs = evaluate_mapanything_quality_output(output_path, canonical)
    if receipt.get("outputs") != observed_outputs:
        raise EngineMaintenanceError(
            "MapAnything quality receipt metrics differ from sealed output evidence"
        )

    commands = maintenance_payload.get("commands")
    matches = [
        row
        for row in commands if isinstance(row, Mapping) and row.get("label") == MAPANYTHING_QUALITY_COMMAND_LABEL
    ] if isinstance(commands, list) else []
    if (
        len(matches) != 1
        or matches[0].get("command") != receipt.get("command")
        or matches[0].get("proof") != "trtexec_inference"
        or matches[0].get("status") != "passed"
        or matches[0].get("returncode") != 0
        or matches[0].get("timed_out") is not False
        or matches[0].get("output_exceeded") is not False
    ):
        raise EngineMaintenanceError(
            "MapAnything quality receipt command differs from maintenance evidence"
        )
    command = receipt.get("command")
    receipt_candidate = Path(str(engine.get("path") or ""))
    receipt_fixture = Path(str(fixture.get("path") or ""))
    receipt_output = Path(str(command[-1]).removeprefix("--exportOutput=")) if isinstance(command, list) and command else Path()
    expected_command = _quality_command(
        executable=str(command[0]) if isinstance(command, list) and command else "",
        engine_path=receipt_candidate,
        fixture_path=receipt_fixture,
        output_path=receipt_output,
        authority=canonical,
    )
    if command != expected_command or receipt_output.name != MAPANYTHING_QUALITY_OUTPUT_NAME:
        raise EngineMaintenanceError("MapAnything quality command is not canonical")
    return _json_copy(receipt, "validated MapAnything quality receipt")


def validate_maintenance_build_contract(
    name: str, actual: Mapping[str, Any], contracts_path: Path
) -> dict[str, Any]:
    """Fail before GPU work when a builder drifts from reviewed build authority."""

    contracts = load_source_contracts(contracts_path)
    source_contract = contracts.get(name)
    if not isinstance(source_contract, Mapping):
        raise EngineMaintenanceError(f"missing pinned source contract for {name}")
    reviewed = source_contract.get("maintenance_build")
    if not isinstance(reviewed, Mapping):
        raise EngineMaintenanceError(
            f"missing reviewed maintenance build contract for {name}"
        )
    expected = dict(reviewed)
    if reviewed.get("builder") == _WHOLEBODY_BUILDER_CONTRACT:
        expected["memory_pool_limits_bytes"] = (
            validate_wholebody_memory_pool_limits(
                reviewed.get("memory_pool_limits_bytes")
            )
        )
        expected["builder_optimization_level"] = (
            validate_wholebody_builder_optimization_level(
                reviewed.get("builder_optimization_level")
            )
        )
        expected["logger_policy"] = validate_wholebody_logger_policy(
            reviewed.get("logger_policy")
        )
    tensor = source_contract.get("onnx") or source_contract.get("tensor_contract")
    if not isinstance(tensor, Mapping):
        raise EngineMaintenanceError(f"missing reviewed tensor contract for {name}")
    expected["onnx" if source_contract.get("onnx") is not None else "tensor_contract"] = dict(
        tensor
    )
    observed = dict(actual)
    if actual.get("builder") == _WHOLEBODY_BUILDER_CONTRACT:
        observed["memory_pool_limits_bytes"] = (
            validate_wholebody_memory_pool_limits(
                actual.get("memory_pool_limits_bytes")
            )
        )
        observed["builder_optimization_level"] = (
            validate_wholebody_builder_optimization_level(
                actual.get("builder_optimization_level")
            )
        )
        observed["logger_policy"] = validate_wholebody_logger_policy(
            actual.get("logger_policy")
        )
    if observed != expected:
        raise EngineMaintenanceError(
            f"maintenance build contract drift for {name}: "
            f"expected={expected!r} observed={observed!r}"
        )
    return expected


def validate_prepared_transaction_authority(
    *,
    transaction_manifest: Path,
    expected_sha256: str,
    engine_name: str,
    engine_target: Path,
) -> dict[str, Any]:
    """Authorize a real container build from the host's prepared transaction."""

    if transaction_manifest.is_symlink() or not transaction_manifest.is_file():
        raise EngineMaintenanceError(
            f"prepared transaction must be a regular non-symlink file: {transaction_manifest}"
        )
    info = transaction_manifest.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size <= 0
        or info.st_size > 2 * 1024 * 1024
    ):
        raise EngineMaintenanceError(
            "prepared transaction must be owner-private, single-link, and bounded"
        )
    raw = transaction_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise EngineMaintenanceError("prepared transaction digest changed")

    try:
        payload = strict_json_loads(
            raw,
            label="prepared engine transaction",
        )
    except StrictJSONError as exc:
        raise EngineMaintenanceError(
            "prepared transaction is not valid strict JSON"
        ) from exc
    if not isinstance(payload, dict) or (
        payload.get("schema_version") != 1
        or payload.get("contract") != "noesis.ds9.engine_finalize_transaction"
        or payload.get("state") != "prepared"
        or not re.fullmatch(
            r"[0-9A-Za-z_.-]+", str(payload.get("transaction_id") or "")
        )
        or payload.get("engine") != engine_name
        or payload.get("container_engine_output")
        != str(engine_target.expanduser().absolute())
    ):
        raise EngineMaintenanceError(
            "prepared transaction does not authorize this engine target"
        )
    prior = payload.get("prior_engine")
    if not isinstance(prior, Mapping):
        raise EngineMaintenanceError("prepared transaction lacks prior-engine authority")
    target = engine_target.expanduser().absolute()
    if bool(prior.get("exists")):
        required_regular_file(target, "engine target at prepared-authority gate")
        if (
            target.stat().st_size != prior.get("size_bytes")
            or sha256_file(target) != prior.get("sha256")
            or f"{stat.S_IMODE(target.stat().st_mode):04o}" != prior.get("mode")
        ):
            raise EngineMaintenanceError(
                "engine target differs from the prepared host snapshot"
            )
    elif target.exists() or target.is_symlink():
        raise EngineMaintenanceError(
            "engine target appeared after the prepared host snapshot"
        )
    return payload


def required_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise EngineMaintenanceError(f"{label} must not be a symlink: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise EngineMaintenanceError(f"{label} is missing or empty: {path}")


def require_absent_candidate_path(path: Path) -> None:
    """Fail without mutation if a unique transaction candidate already exists."""

    reject_symlink_ancestors(path.parent, "candidate engine parent")
    if path.exists() or path.is_symlink():
        raise EngineMaintenanceError(
            f"unique candidate path already exists; refusing to delete it: {path}"
        )


def reject_symlink_ancestors(path: Path, label: str) -> None:
    absolute = path.expanduser().absolute()
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink():
            raise EngineMaintenanceError(f"{label} contains a symlink: {ancestor}")


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stat_record(path: Path, info: os.stat_result) -> dict[str, object]:
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": info.st_uid,
        "gid": info.st_gid,
        "nlink": info.st_nlink,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _require_owned_regular_stat(
    path: Path, info: os.stat_result, label: str
) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_size <= 0
    ):
        raise EngineMaintenanceError(
            f"{label} must be a nonempty owner-owned single-link regular file: {path}"
        )


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 4 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def copy_regular_file_exclusive(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Durably stream-copy one stable owned file without following links.

    The destination is intentionally left in place after any post-create
    failure. Transaction callers must use an authorized hidden destination so
    the host finalizer can audit and remove a partial copy after interruption.
    """

    source = source.expanduser().absolute()
    destination = destination.expanduser().absolute()
    reject_symlink_ancestors(source, "exclusive-copy source")
    reject_symlink_ancestors(destination.parent, "exclusive-copy destination")
    require_absent_candidate_path(destination)
    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise EngineMaintenanceError("exclusive-copy expected SHA-256 is invalid")

    try:
        source_path_before = source.lstat()
    except OSError as exc:
        raise EngineMaintenanceError(
            f"exclusive-copy source is unavailable: {source}: {exc}"
        ) from exc
    _require_owned_regular_stat(source, source_path_before, "exclusive-copy source")

    source_flags = os.O_RDONLY
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        value = getattr(os, name, 0)
        source_flags |= value
        destination_flags |= value

    source_descriptor = -1
    destination_descriptor = -1
    try:
        source_descriptor = os.open(source, source_flags)
        source_fd_before = os.fstat(source_descriptor)
        _require_owned_regular_stat(
            source, source_fd_before, "opened exclusive-copy source"
        )
        if _stat_identity(source_fd_before) != _stat_identity(source_path_before):
            raise EngineMaintenanceError(
                f"exclusive-copy source changed while opened: {source}"
            )

        destination_descriptor = os.open(destination, destination_flags, 0o600)
        digest = hashlib.sha256()
        while True:
            block = os.read(source_descriptor, 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            remaining = memoryview(block)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise EngineMaintenanceError(
                        f"exclusive-copy write made no progress: {destination}"
                    )
                remaining = remaining[written:]

        os.fchmod(destination_descriptor, 0o600)
        os.fsync(destination_descriptor)
        source_fd_after = os.fstat(source_descriptor)
        destination_fd = os.fstat(destination_descriptor)
        if _stat_identity(source_fd_after) != _stat_identity(source_fd_before):
            raise EngineMaintenanceError(
                f"exclusive-copy source changed while copied: {source}"
            )
        _require_owned_regular_stat(
            destination, destination_fd, "exclusive-copy destination"
        )
        if stat.S_IMODE(destination_fd.st_mode) != 0o600:
            raise EngineMaintenanceError(
                f"exclusive-copy destination is not owner-private: {destination}"
            )
        source_sha256 = digest.hexdigest()
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise EngineMaintenanceError(
                "exclusive-copy source SHA-256 differs from the reviewed lock: "
                f"{source}"
            )
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)

    fsync_directory(destination.parent)
    destination_path = destination.lstat()
    if _stat_identity(destination_path) != _stat_identity(destination_fd):
        raise EngineMaintenanceError(
            f"exclusive-copy destination changed after close: {destination}"
        )
    source_path_after = source.lstat()
    if _stat_identity(source_path_after) != _stat_identity(source_path_before):
        raise EngineMaintenanceError(
            f"exclusive-copy source path changed after copy: {source}"
        )

    verify_flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        verify_flags |= getattr(os, name, 0)
    verify_descriptor = os.open(destination, verify_flags)
    try:
        destination_verify = os.fstat(verify_descriptor)
        if _stat_identity(destination_verify) != _stat_identity(destination_fd):
            raise EngineMaintenanceError(
                f"exclusive-copy destination changed while verified: {destination}"
            )
        destination_sha256 = _sha256_fd(verify_descriptor)
    finally:
        os.close(verify_descriptor)
    if destination_sha256 != source_sha256:
        raise EngineMaintenanceError(
            f"exclusive-copy destination hash differs from source: {destination}"
        )

    return {
        "method": "exclusive_nofollow_stream_copy",
        "source": {
            **_stat_record(source, source_path_after),
            "sha256": source_sha256,
        },
        "destination": {
            **_stat_record(destination, destination_path),
            "sha256": destination_sha256,
        },
        "durability": "file_and_destination_directory_fsynced",
    }


def ensure_private_directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    reject_symlink_ancestors(path, "private evidence path")
    probe = path
    missing: list[Path] = []
    while not probe.exists():
        if probe.is_symlink():
            raise EngineMaintenanceError(f"private evidence path must not be a symlink: {probe}")
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    if probe.is_symlink():
        raise EngineMaintenanceError(f"private evidence ancestor must not be a symlink: {probe}")
    if not probe.is_dir():
        raise EngineMaintenanceError(f"private evidence ancestor is not a directory: {probe}")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise EngineMaintenanceError(f"private evidence path is not a real directory: {path}")
    os.chmod(path, 0o700)
    return path


def _write_text_fsynced(path: Path, value: str) -> None:
    ensure_private_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def _copy_private(source: Path, destination: Path) -> None:
    ensure_private_directory(destination.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            descriptor = -1
            shutil.copyfileobj(source_handle, destination_handle, 4 * 1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.chmod(destination, 0o600)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.writing-{os.getpid()}")
    try:
        _write_text_fsynced(
            temporary,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def trtexec_negative_markers(transcript: str) -> list[str]:
    return [pattern.pattern for pattern in _NEGATIVE_TRTEXEC_PATTERNS if pattern.search(transcript)]


def validate_trtexec_transcript(transcript: str, *, require_load_proof: bool) -> None:
    negatives = trtexec_negative_markers(transcript)
    if negatives:
        raise EngineMaintenanceError(
            "trtexec transcript contains fail-closed marker(s): " + ", ".join(negatives)
        )
    if _TRTEXEC_PASSED.search(transcript) is None:
        raise EngineMaintenanceError("trtexec transcript lacks the PASSED completion marker")
    if not require_load_proof:
        return
    loaded = _LOADED_ENGINE.search(transcript)
    if loaded is None or int(loaded.group(1)) <= 0:
        raise EngineMaintenanceError("trtexec load lacks a positive Loaded engine size marker")
    if _DESERIALIZED.search(transcript) is None:
        raise EngineMaintenanceError("trtexec load lacks an Engine deserialized marker")
    if _SKIPPED_INFERENCE.search(transcript) is None:
        raise EngineMaintenanceError("trtexec load lacks the skipped-inference marker")


def validate_trtexec_inference_transcript(transcript: str) -> None:
    """Require actual candidate execution, not deserialization-only evidence."""

    negatives = trtexec_negative_markers(transcript)
    if negatives:
        raise EngineMaintenanceError(
            "trtexec inference transcript contains fail-closed marker(s): "
            + ", ".join(negatives)
        )
    if _TRTEXEC_PASSED.search(transcript) is None:
        raise EngineMaintenanceError(
            "trtexec inference transcript lacks the PASSED completion marker"
        )
    loaded = _LOADED_ENGINE.search(transcript)
    if loaded is None or int(loaded.group(1)) <= 0:
        raise EngineMaintenanceError(
            "trtexec inference lacks a positive Loaded engine size marker"
        )
    if _DESERIALIZED.search(transcript) is None:
        raise EngineMaintenanceError(
            "trtexec inference lacks an Engine deserialized marker"
        )
    if _SKIPPED_INFERENCE.search(transcript) is not None:
        raise EngineMaintenanceError(
            "trtexec functional quality proof used --skipInference"
        )
    if _STARTED_INFERENCE.search(transcript) is None:
        raise EngineMaintenanceError(
            "trtexec functional quality proof never started inference"
        )
    if _ONE_QUERY_TRACE.search(transcript) is None:
        raise EngineMaintenanceError(
            "trtexec functional quality proof lacks exactly one completed query"
        )


def validate_trtexec_probe(returncode: int, transcript: str) -> None:
    if returncode != 0:
        raise EngineMaintenanceError(f"trtexec --help exited {returncode}")
    negatives = trtexec_negative_markers(transcript)
    if negatives:
        raise EngineMaintenanceError(
            "trtexec --help contains fail-closed marker(s): " + ", ".join(negatives)
        )
    if DS9_TRTEXEC_BANNER not in transcript:
        raise EngineMaintenanceError(
            f"trtexec --help lacks exact DS9 banner {DS9_TRTEXEC_BANNER!r}"
        )


def validate_wholebody_builder_transcript(
    transcript: str, *, expected_variant: str
) -> None:
    """Require the exact positive contract emitted by the DS9 C++ builder."""

    if expected_variant not in {"s_masks", "x_boxes"}:
        raise EngineMaintenanceError(
            f"invalid expected Wholebody49 builder variant: {expected_variant}"
        )
    if _WHOLEBODY_BUILDER_FAILURE.search(transcript):
        raise EngineMaintenanceError(
            "Wholebody49 builder transcript contains its fail-closed marker"
        )
    lines = transcript.splitlines()
    for required in _WHOLEBODY_BUILDER_REQUIRED_LINES:
        if lines.count(required) != 1:
            raise EngineMaintenanceError(
                "Wholebody49 builder transcript lacks one exact marker: "
                f"{required}"
            )
    variant_line = f"[NOESIS_TRT_BUILDER] variant={expected_variant}"
    if lines.count(variant_line) != 1:
        raise EngineMaintenanceError(
            "Wholebody49 builder transcript differs from the selected variant: "
            f"{variant_line}"
        )
    workspace_line = (
        "[NOESIS_TRT_BUILDER] workspace_bytes="
        f"{_WHOLEBODY_WORKSPACE_BYTES_BY_VARIANT[expected_variant]}"
    )
    workspace_lines = [
        line
        for line in lines
        if line.startswith("[NOESIS_TRT_BUILDER] workspace_bytes=")
    ]
    if workspace_lines != [workspace_line]:
        raise EngineMaintenanceError(
            "Wholebody49 builder transcript lacks one exact variant workspace "
            f"marker: {workspace_line}"
        )
    optimization_line = (
        "[NOESIS_TRT_BUILDER] builder_optimization_level="
        f"{_WHOLEBODY_BUILDER_OPTIMIZATION_LEVEL_BY_VARIANT[expected_variant]}"
    )
    optimization_lines = [
        line
        for line in lines
        if line.startswith(
            "[NOESIS_TRT_BUILDER] builder_optimization_level="
        )
    ]
    if optimization_lines != [optimization_line]:
        raise EngineMaintenanceError(
            "Wholebody49 builder transcript lacks one exact variant optimization "
            f"marker: {optimization_line}"
        )
    engine_bytes = _WHOLEBODY_BUILDER_ENGINE_BYTES.findall(transcript)
    if len(engine_bytes) != 1 or int(engine_bytes[0]) <= 0:
        raise EngineMaintenanceError(
            "Wholebody49 builder transcript lacks one positive engine size marker"
        )


def maintenance_provenance_from_environment() -> dict[str, str]:
    provenance = {"expected_trtexec_banner": DS9_TRTEXEC_BANNER}
    for environment_name, manifest_name in _PROVENANCE_ENV.items():
        value = str(os.environ.get(environment_name, "") or "").strip()
        if value:
            provenance[manifest_name] = value
    return provenance


@contextmanager
def engine_maintenance_lock(lock_path: Path, *, dry_run: bool) -> Iterator[None]:
    if dry_run:
        yield
        return
    reject_symlink_ancestors(lock_path, "engine maintenance lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise EngineMaintenanceError(
                f"engine maintenance lock must be a private owner-only regular file: {lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EngineMaintenanceError(
                f"another DS9 engine maintenance process owns {lock_path}"
            ) from exc
        yield
    finally:
        os.close(descriptor)


class EngineMaintenanceRun:
    """Durable evidence and atomic install state for one engine."""

    def __init__(
        self,
        *,
        name: str,
        target: Path,
        evidence_root: Path,
        inputs: Mapping[str, Path],
        repo_root: Path,
        run_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.name = str(name)
        reject_symlink_ancestors(target, "engine target")
        self.target = target.expanduser().absolute()
        self.repo_root = repo_root.resolve()
        self.run_id = run_id or new_run_id()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.name).strip("-")
        private_root = ensure_private_directory(evidence_root)
        self.run_dir = private_root / f"{self.run_id}-{safe_name}"
        self.run_dir.mkdir(mode=0o700, exist_ok=False)
        os.chmod(self.run_dir, 0o700)
        self.manifest_path = self.run_dir / "manifest.json"
        self.payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "contract": CONTRACT,
            "run_id": self.run_id,
            "engine": self.name,
            "target": str(self.target),
            "status": "running",
            "created_at_utc": utc_now(),
            "inputs": {},
            "commands": [],
            "metadata": dict(metadata or {}),
        }
        for label, path in inputs.items():
            self.record_input(label, path, persist=False)
        self._preserve_prior()
        self.persist()

    def persist(self) -> None:
        _write_json_atomic(self.manifest_path, self.payload)

    def record_input(self, label: str, path: Path, *, persist: bool = True) -> None:
        inputs = self.payload.setdefault("inputs", {})
        assert isinstance(inputs, dict)
        inputs[str(label)] = input_bundle_record(path)
        if persist:
            self.persist()

    def revalidate_inputs(self) -> None:
        inputs = self.payload.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            raise EngineMaintenanceError("maintenance manifest has no recorded inputs")
        for label, expected in inputs.items():
            if not isinstance(expected, dict):
                raise EngineMaintenanceError(f"recorded input {label} is malformed")
            current = input_bundle_record(Path(str(expected.get("path") or "")))
            for key in ("size_bytes", "sha256", "bundle_sha256", "files"):
                if current.get(key) != expected.get(key):
                    raise EngineMaintenanceError(
                        f"recorded input changed before install: {label} ({key})"
                    )
        self.payload["inputs_revalidated_at_utc"] = utc_now()
        self.persist()

    def record_evidence(self, label: str, value: Mapping[str, object]) -> None:
        evidence = self.payload.setdefault("evidence", {})
        if not isinstance(evidence, dict):
            raise EngineMaintenanceError("maintenance evidence section is malformed")
        key = str(label)
        if key in evidence:
            raise EngineMaintenanceError(
                f"maintenance evidence label is already recorded: {key}"
            )
        evidence[key] = dict(value)
        self.persist()

    def record_mapanything_functional_quality(
        self,
        *,
        candidate: Path,
        fixture_path: Path,
        output_path: Path,
        authority: Mapping[str, Any],
        command_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.name != "mapanything":
            raise EngineMaintenanceError(
                "MapAnything functional quality evidence cannot authorize another engine"
            )
        metadata = self.payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise EngineMaintenanceError("maintenance metadata is malformed")
        receipt = build_mapanything_functional_quality_receipt(
            candidate=candidate,
            fixture_path=fixture_path,
            output_path=output_path,
            evidence_directory=self.run_dir,
            authority=authority,
            platform=metadata.get("platform") if isinstance(metadata, Mapping) else {},
            command_record=command_record,
        )
        self.record_evidence(MAPANYTHING_QUALITY_EVIDENCE_LABEL, receipt)
        return receipt

    def adopt_derived_candidate(
        self,
        derived: Path,
        candidate: Path,
        *,
        workspace_root: Path,
    ) -> dict[str, object]:
        """Copy one SDK-derived plan into the host-authorized candidate path."""

        workspace_raw = workspace_root.expanduser().absolute()
        reject_symlink_ancestors(workspace_raw, "derived-plan workspace")
        if workspace_raw.is_symlink() or not workspace_raw.is_dir():
            raise EngineMaintenanceError(
                f"derived-plan workspace is missing or unsafe: {workspace_raw}"
            )
        workspace_info = workspace_raw.stat()
        if (
            workspace_info.st_uid != os.getuid()
            or stat.S_IMODE(workspace_info.st_mode) != 0o700
        ):
            raise EngineMaintenanceError(
                "derived-plan workspace must be an owner-private directory: "
                f"{workspace_raw}"
            )
        workspace = workspace_raw.resolve(strict=True)

        derived_raw = derived.expanduser().absolute()
        reject_symlink_ancestors(derived_raw, "derived plan")
        if derived_raw.is_symlink():
            raise EngineMaintenanceError(
                f"derived plan must not be a symlink: {derived_raw}"
            )
        derived_path = derived_raw.resolve(strict=True)
        try:
            derived_path.relative_to(workspace)
        except ValueError as exc:
            raise EngineMaintenanceError(
                f"derived plan escaped its transaction workspace: {derived_path}"
            ) from exc
        derived_info = derived_path.lstat()
        _require_owned_regular_stat(derived_path, derived_info, "derived plan")

        candidate_path = candidate.expanduser().absolute()
        expected_candidate = self.target.with_name(
            f".{self.target.name}.building-{self.run_id}"
        )
        if candidate_path != expected_candidate:
            raise EngineMaintenanceError(
                "derived-plan candidate must be the exact host-authorized path: "
                f"expected={expected_candidate} observed={candidate_path}"
            )
        require_absent_candidate_path(candidate_path)

        adoption: dict[str, object] = {
            "status": "prepared",
            "prepared_at_utc": utc_now(),
            "workspace_root": str(workspace),
            "derived": _stat_record(derived_path, derived_info),
            "candidate": str(candidate_path),
            "failure_cleanup_authority": "host_engine_finalize_transaction",
        }
        self.payload["candidate_adoption"] = adoption
        self.persist()
        try:
            copy_record = copy_regular_file_exclusive(
                derived_path,
                candidate_path,
            )
        except BaseException as exc:
            adoption.update(
                {
                    "status": "failed_authorized_partial",
                    "failed_at_utc": utc_now(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            self.persist()
            raise

        adoption.update(
            {
                "status": "verified",
                "verified_at_utc": utc_now(),
                "copy": copy_record,
            }
        )
        self.persist()
        destination = copy_record.get("destination")
        if not isinstance(destination, dict):
            raise EngineMaintenanceError("derived candidate copy record is malformed")
        return destination

    def _preserve_prior(self) -> None:
        if not self.target.exists():
            self.payload["prior"] = {"exists": False, "path": str(self.target)}
            return
        required_regular_file(self.target, "prior engine")
        prior_hash = sha256_file(self.target)
        prior_size = self.target.stat().st_size
        prior_mode = stat.S_IMODE(self.target.stat().st_mode)
        prior_dir = self.run_dir / "prior"
        prior_dir.mkdir(mode=0o700, exist_ok=False)
        os.chmod(prior_dir, 0o700)
        preserved = prior_dir / self.target.name
        _copy_private(self.target, preserved)
        if preserved.stat().st_size != prior_size or sha256_file(preserved) != prior_hash:
            raise EngineMaintenanceError(f"preserved prior engine failed verification: {preserved}")
        fsync_directory(prior_dir)
        self.payload["prior"] = {
            "exists": True,
            "path": str(self.target),
            "size_bytes": prior_size,
            "sha256": prior_hash,
            "mode": f"{prior_mode:04o}",
        }
        self.payload["preserved_prior"] = {
            "path": str(preserved),
            "size_bytes": prior_size,
            "sha256": prior_hash,
            "mode": "0600",
        }

    def run_command(
        self,
        label: str,
        command: Sequence[str | Path],
        *,
        proof: str,
        timeout_seconds: int | None = None,
        env: Mapping[str, str] | None = None,
        max_output_bytes: int | None = None,
    ) -> dict[str, object]:
        rendered = [str(value) for value in command]
        print("[RUN]", " ".join(rendered))
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                rendered,
                cwd=str(self.repo_root),
                env=None if env is None else dict(env),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            output = proc.stdout or ""
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            raw_output = exc.stdout or ""
            output = raw_output.decode(errors="replace") if isinstance(raw_output, bytes) else raw_output
            returncode = -1
        duration = time.monotonic() - started
        output_exceeded = False
        if max_output_bytes is not None:
            if (
                isinstance(max_output_bytes, bool)
                or not isinstance(max_output_bytes, int)
                or max_output_bytes <= 0
            ):
                raise EngineMaintenanceError("command output bound must be positive")
            encoded = output.encode("utf-8", errors="replace")
            if len(encoded) > max_output_bytes:
                output_exceeded = True
                output = encoded[:max_output_bytes].decode(
                    "utf-8", errors="replace"
                ) + "\n[NOESIS] transcript truncated at mandatory bound\n"
        if output:
            print(output, end="" if output.endswith("\n") else "\n")
        log_path = self.run_dir / "logs" / f"{label}.log"
        _write_text_fsynced(log_path, output)
        record: dict[str, object] = {
            "label": label,
            "command": rendered,
            "proof": proof,
            "returncode": returncode,
            "duration_seconds": duration,
            "log": str(log_path),
            "timed_out": timed_out,
            "output_exceeded": output_exceeded,
            "status": "failed",
        }
        commands = self.payload.setdefault("commands", [])
        assert isinstance(commands, list)
        commands.append(record)
        try:
            if timed_out:
                raise EngineMaintenanceError(
                    f"{label} exceeded {timeout_seconds} seconds"
                )
            if returncode != 0:
                raise EngineMaintenanceError(f"{label} exited {returncode}")
            if output_exceeded:
                raise EngineMaintenanceError(
                    f"{label} exceeded the {max_output_bytes}-byte transcript bound"
                )
            if proof == "trtexec_build":
                validate_trtexec_transcript(output, require_load_proof=False)
            elif proof == "trtexec_load":
                validate_trtexec_transcript(output, require_load_proof=True)
            elif proof == "trtexec_probe":
                validate_trtexec_probe(returncode, output)
            elif proof == "trtexec_inference":
                validate_trtexec_inference_transcript(output)
            elif proof.startswith("wholebody_builder_"):
                validate_wholebody_builder_transcript(
                    output,
                    expected_variant=proof.removeprefix("wholebody_builder_"),
                )
            elif proof != "generic":
                raise EngineMaintenanceError(f"unknown command proof mode: {proof}")
        except BaseException:
            self.persist()
            raise
        record["status"] = "passed"
        self.persist()
        return record

    def record_candidate(self, candidate: Path) -> dict[str, object]:
        required_regular_file(candidate, "candidate engine")
        fsync_file(candidate)
        record = {
            "path": str(candidate.resolve()),
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
        self.payload["candidate"] = record
        self.persist()
        return record

    def install_candidate(self, candidate: Path) -> dict[str, object]:
        candidate_record = self.record_candidate(candidate)
        if self.name == "mapanything":
            metadata = self.payload.get("metadata")
            authority = (
                metadata.get("quality_gate_authority")
                if isinstance(metadata, Mapping)
                else None
            )
            validate_mapanything_functional_quality_receipt(
                self.payload,
                authority=authority if isinstance(authority, Mapping) else {},
                expected_engine_path=candidate,
                evidence_directory=self.run_dir,
            )
        self.target.parent.mkdir(parents=True, exist_ok=True)
        prior = self.payload.get("prior")
        if not isinstance(prior, dict):
            raise EngineMaintenanceError("manifest has no prior-engine record")
        if bool(prior.get("exists")):
            required_regular_file(self.target, "engine target before install")
            if (
                self.target.stat().st_size != int(prior.get("size_bytes") or 0)
                or sha256_file(self.target) != str(prior.get("sha256") or "")
            ):
                raise EngineMaintenanceError(
                    "engine target changed after prior preservation; refusing install"
                )
        elif self.target.exists() or self.target.is_symlink():
            raise EngineMaintenanceError(
                "engine target appeared after prior absence was recorded; refusing install"
            )

        transaction = {
            "status": "prepared",
            "prepared_at_utc": utc_now(),
            "target": str(self.target),
            "candidate": dict(candidate_record),
        }
        self.payload["install_transaction"] = transaction
        self.persist()
        replaced = False
        try:
            os.replace(candidate, self.target)
            replaced = True
            transaction["status"] = "replaced_pending_verification"
            transaction["replaced_at_utc"] = utc_now()
            fsync_directory(self.target.parent)
            required_regular_file(self.target, "installed engine")
            installed = {
                "path": str(self.target),
                "size_bytes": self.target.stat().st_size,
                "sha256": sha256_file(self.target),
                "installed_at_utc": utc_now(),
            }
            if (
                installed["size_bytes"] != candidate_record["size_bytes"]
                or installed["sha256"] != candidate_record["sha256"]
            ):
                raise EngineMaintenanceError(
                    "installed engine hash/size differs from candidate"
                )
            self.payload["installed"] = installed
            transaction["status"] = "installed_verified"
            transaction["verified_at_utc"] = utc_now()
            self.persist()
            return installed
        except BaseException as exc:
            if replaced:
                self.rollback_after_install_failure(exc)
            raise

    def rollback_after_install_failure(self, cause: BaseException) -> bool:
        rollback: dict[str, object] = {
            "requested_at_utc": utc_now(),
            "cause": {"type": type(cause).__name__, "message": str(cause)},
            "status": "manual_recovery_required",
        }
        self.payload["rollback_after_install_failure"] = rollback
        installed = self.payload.get("installed")
        if not isinstance(installed, dict):
            transaction = self.payload.get("install_transaction")
            installed = (
                transaction.get("candidate")
                if isinstance(transaction, dict)
                else None
            )
        if not isinstance(installed, dict):
            rollback["reason"] = "manifest has no installed candidate record"
            self.persist()
            return False
        expected_hash = str(installed.get("sha256") or "")
        expected_size = int(installed.get("size_bytes") or 0)
        try:
            required_regular_file(self.target, "installed engine before rollback")
            observed = {
                "path": str(self.target),
                "size_bytes": self.target.stat().st_size,
                "sha256": sha256_file(self.target),
            }
        except BaseException as exc:
            rollback["reason"] = f"unable to verify installed target before rollback: {exc}"
            self.persist()
            return False
        rollback["observed_target"] = observed
        if observed["sha256"] != expected_hash or observed["size_bytes"] != expected_size:
            rollback["reason"] = (
                "installed target no longer matches the candidate; refusing to overwrite "
                "possible external changes"
            )
            self.persist()
            return False

        prior = self.payload.get("prior")
        if not isinstance(prior, dict):
            rollback["reason"] = "manifest has no prior-engine record"
            self.persist()
            return False
        if not bool(prior.get("exists")):
            self.target.unlink()
            fsync_directory(self.target.parent)
            rollback.update(
                {
                    "status": "removed_candidate_no_prior",
                    "completed_at_utc": utc_now(),
                }
            )
            self.persist()
            return True

        preserved = self.payload.get("preserved_prior")
        if not isinstance(preserved, dict):
            rollback["reason"] = "manifest has no preserved-prior record"
            self.persist()
            return False
        preserved_path = Path(str(preserved.get("path") or ""))
        try:
            required_regular_file(preserved_path, "preserved prior engine")
            prior_hash = str(prior.get("sha256") or "")
            prior_size = int(prior.get("size_bytes") or 0)
            if (
                preserved_path.stat().st_size != prior_size
                or sha256_file(preserved_path) != prior_hash
            ):
                raise EngineMaintenanceError("preserved prior hash/size changed")
            temporary = self.target.with_name(
                f".{self.target.name}.rollback-{self.run_id}"
            )
            temporary.unlink(missing_ok=True)
            try:
                _copy_private(preserved_path, temporary)
                prior_mode = int(str(prior.get("mode") or "0600"), 8)
                os.chmod(temporary, prior_mode)
                fsync_file(temporary)
                if temporary.stat().st_size != prior_size or sha256_file(temporary) != prior_hash:
                    raise EngineMaintenanceError("rollback temporary failed hash/size verification")
                os.replace(temporary, self.target)
                fsync_directory(self.target.parent)
            finally:
                temporary.unlink(missing_ok=True)
            required_regular_file(self.target, "restored prior engine")
            restored = {
                "path": str(self.target),
                "size_bytes": self.target.stat().st_size,
                "sha256": sha256_file(self.target),
                "mode": f"{stat.S_IMODE(self.target.stat().st_mode):04o}",
            }
            if restored["size_bytes"] != prior_size or restored["sha256"] != prior_hash:
                raise EngineMaintenanceError("restored prior failed final hash/size verification")
        except BaseException as exc:
            rollback["reason"] = f"automatic prior restoration failed: {exc}"
            self.persist()
            return False
        rollback.update(
            {
                "status": "restored_prior",
                "restored": restored,
                "completed_at_utc": utc_now(),
            }
        )
        self.persist()
        return True

    def complete(self) -> None:
        self.payload["status"] = "complete"
        self.payload["completed_at_utc"] = utc_now()
        self.persist()

    def fail(self, exc: BaseException) -> None:
        self.payload["status"] = "failed"
        self.payload["completed_at_utc"] = utc_now()
        self.payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        self.persist()
