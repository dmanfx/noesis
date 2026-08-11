#!/usr/bin/env python3
"""Build unpromoted RF-DETR 1.8.3 TensorRT engines with the reviewed DS9 image.

This is intentionally separate from the canonical DS9 engine authority.  It
only writes the versioned RF-DETR 1.8.3 engine/evidence directories and never
updates the DS9 asset manifest, realization, runtime configuration, or model
selection.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from packaging.version import Version


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
MODEL_MATRIX_PATH = DS9_ROOT / "config" / "rfdetr_1_8_3_models.json"

MODEL_MATRIX_SCHEMA = "noesis.ds9.rfdetr-model-matrix.v1"
SOURCE_PROVENANCE_SCHEMA = "noesis.ds9.rfdetr-artifact-provenance.v1"
ENGINE_RECEIPT_SCHEMA = "noesis.ds9.rfdetr-engine-provenance.v1"
CANARY_ENGINE_RECEIPT_SCHEMA = (
    "noesis.ds9.rfdetr-engine-precision-canary-provenance.v1"
)
RUNTIME_ENGINE_RECEIPT_SCHEMA = (
    "noesis.ds9.rfdetr-runtime-engine-provenance.v1"
)
RELEASE_VERSION = "1.8.3"
REQUIRED_IMAGE_REF = "noesis-ds9-dev:9.0-20260710"
REQUIRED_IMAGE_ID = (
    "sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4"
)
REQUIRED_BASE_DIGEST = (
    "sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a"
)
REQUIRED_TRT_VERSION = "10.14.1.48"
REQUIRED_IMAGE_TRT_VERSION = "10.14.1.48+cuda13.0"
REQUIRED_CUDA_VERSION = "13.1.1.006"
REQUIRED_TRT_BANNER = "TensorRT v101401"
GPU_DEVICE_INDEX = 0
MAINTENANCE_MEMORY_BYTES = 25_769_803_776
MAINTENANCE_PIDS_LIMIT = 512
MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

BASELINE_PRECISION_PROFILE = "fp16_tf32"
RUNTIME_INPUT_CONTRACT = "rgb01_imagenet"
RUNTIME_ADAPTER_REVISION = "sub_div_float32_v1"
RUNTIME_ADAPTER_SPEC = {
    "input_contract": RUNTIME_INPUT_CONTRACT,
    "adapter_revision": RUNTIME_ADAPTER_REVISION,
    "input_dtype": "float32",
    "input_range": [0.0, 1.0],
    "operation": "(input - mean) / std",
    "channel_order": "RGB",
    "layout": "NCHW",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "nodes": [
        "NoesisInputNormalize/Sub",
        "NoesisInputNormalize/DivStd",
    ],
}
CANARY_PRECISION_PROFILES = {
    "fp32_tf32": {
        "precision": "fp32",
        "fp16_enabled": False,
        "tf32_enabled": True,
        "selective_fp32_heads": False,
        "trtexec_args": (),
    },
    "fp32_no_tf32": {
        "precision": "fp32",
        "fp16_enabled": False,
        "tf32_enabled": False,
        "selective_fp32_heads": False,
        "trtexec_args": ("--noTF32",),
    },
    "fp16_no_tf32": {
        "precision": "fp16",
        "fp16_enabled": True,
        "tf32_enabled": False,
        "selective_fp32_heads": False,
        "trtexec_args": ("--fp16", "--noTF32"),
    },
    "fp16_fp32_heads_no_tf32": {
        "precision": "fp16_with_fp32_heads",
        "fp16_enabled": True,
        "tf32_enabled": False,
        "selective_fp32_heads": True,
        "trtexec_args": (
            "--fp16",
            "--noTF32",
            "--precisionConstraints=obey",
        ),
    },
}
RUNTIME_ENGINE_PROFILES = (
    BASELINE_PRECISION_PROFILE,
    *tuple(sorted(CANARY_PRECISION_PROFILES)),
)

_FAILED_TRANSCRIPT_PATTERNS = (
    re.compile(r"\[E\]", re.IGNORECASE),
    re.compile(r"\bError\[\d+\]", re.IGNORECASE),
    re.compile(r"&&&&\s+FAILED\s+TensorRT\.trtexec", re.IGNORECASE),
    re.compile(r"engine\s+deserialization\s+failed", re.IGNORECASE),
    re.compile(r"(?:failed|unable|could not)\s+to\s+deserialize", re.IGNORECASE),
)
_TRTEXEC_PASSED = re.compile(
    r"&&&&\s+PASSED\s+TensorRT\.trtexec", re.IGNORECASE
)
_LOADED_ENGINE = re.compile(
    r"Loaded\s+engine\s+size:\s*(\d+)\s*MiB", re.IGNORECASE
)
_DESERIALIZED = re.compile(
    r"Engine\s+deserialized\s+in\s+[0-9.eE+-]+\s+sec", re.IGNORECASE
)
_SKIPPED_INFERENCE = re.compile(
    r"Skipped\s+inference\s+phase\s+since\s+--skipInference\s+is\s+added",
    re.IGNORECASE,
)


class BuilderError(RuntimeError):
    """Raised when the isolated RF-DETR engine-build contract is violated."""


def _safe_token(raw: object, label: str) -> str:
    value = str(raw or "")
    if (
        not SAFE_TOKEN.fullmatch(value)
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise BuilderError(f"unsafe {label} in RF-DETR model matrix: {value!r}")
    return value


def _precision_profile(raw: object) -> PrecisionProfile | None:
    value = str(raw or "").strip()
    if not value:
        return None
    row = CANARY_PRECISION_PROFILES.get(value)
    if row is None:
        raise BuilderError(
            f"unknown precision canary profile {value!r}; valid profiles: "
            + ", ".join(sorted(CANARY_PRECISION_PROFILES))
        )
    return PrecisionProfile(
        id=_safe_token(value, "precision canary profile"),
        precision=str(row["precision"]),
        fp16_enabled=bool(row["fp16_enabled"]),
        tf32_enabled=bool(row["tf32_enabled"]),
        selective_fp32_heads=bool(row["selective_fp32_heads"]),
        trtexec_args=tuple(str(item) for item in row["trtexec_args"]),
    )


def _runtime_engine_profile(raw: object) -> str | None:
    value = str(raw or "").strip()
    if not value:
        return None
    if value not in RUNTIME_ENGINE_PROFILES:
        raise BuilderError(
            f"unknown runtime engine profile {value!r}; valid profiles: "
            + ", ".join(RUNTIME_ENGINE_PROFILES)
        )
    return _safe_token(value, "runtime engine profile")


def _runtime_precision_profile(
    runtime_profile: str | None,
) -> PrecisionProfile | None:
    if runtime_profile in {None, BASELINE_PRECISION_PROFILE}:
        return None
    return _precision_profile(runtime_profile)


def _selective_fp32_layer_specs(model: Mapping[str, Any]) -> tuple[str, ...]:
    family = str(model.get("family") or "")
    common = (
        "/transformer/decoder/norm/*:fp32",
        "/class_embed/*:fp32",
        "/bbox_embed/*:fp32",
    )
    if family == "detection":
        return common
    if family == "segmentation":
        return (
            *common,
            "/segmentation_head/spatial_features_proj/*:fp32",
            "/segmentation_head/query_features_proj/*:fp32",
            "/segmentation_head/Einsum:fp32",
            "/segmentation_head/Add:fp32",
        )
    if family == "keypoint":
        keypoint_specs: list[str] = []
        for layer_index in range(4):
            prefix = f"/transformer/decoder/layers.{layer_index}"
            keypoint_specs.extend(
                (
                    f"{prefix}/kp_inst_norm/*:fp32",
                    f"{prefix}/kp_norm/*:fp32",
                    f"{prefix}/kp_cross_attn_norm/*:fp32",
                    f"{prefix}/kp_linear1/*:fp32",
                    f"{prefix}/kp_linear3/*:fp32",
                    f"{prefix}/kp_norm5/*:fp32",
                )
            )
        return (
            *common,
            *keypoint_specs,
            "/Gather_33:fp32",
            "/Reshape_17:fp32",
            "/Mul_71:fp32",
            "/ReduceSum:fp32",
            "/Add_2:fp32",
        )
    raise BuilderError(f"unsupported RF-DETR family for FP32 heads: {family!r}")


def _precision_trtexec_args(
    profile: PrecisionProfile | None,
    model: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if profile is None:
        return ("--fp16",)
    args = list(profile.trtexec_args)
    if profile.selective_fp32_heads:
        if model is None:
            raise BuilderError(
                "selective FP32-head canary requires an explicit model"
            )
        args.append(
            "--layerPrecisions="
            + ",".join(_selective_fp32_layer_specs(model))
        )
    return tuple(args)


@dataclass(frozen=True)
class ArtifactPaths:
    onnx: Path
    onnx_receipt: Path
    engine: Path
    engine_receipt: Path


@dataclass(frozen=True)
class PrecisionProfile:
    id: str
    precision: str
    fp16_enabled: bool
    tf32_enabled: bool
    selective_fp32_heads: bool
    trtexec_args: tuple[str, ...]


@dataclass(frozen=True)
class DockerContext:
    root: Path
    socket_path: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class CommandEvidence:
    label: str
    inner_command: tuple[str, ...]
    returncode: int
    duration_seconds: float
    log_path: Path
    log_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_exclusive(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) > MAX_TRANSCRIPT_BYTES:
            raise BuilderError(
                f"refusing a transcript larger than {MAX_TRANSCRIPT_BYTES} bytes"
            )
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise BuilderError(f"short write while recording transcript: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise BuilderError(f"{label} must be a nonempty regular file: {path}")


def _require_private_owned_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise BuilderError(f"{label} is missing or is not a directory: {path}")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise BuilderError(
            f"{label} must be caller-owned and mode 0700: {path}"
        )


def _require_owned_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise BuilderError(f"{label} is missing or is not a directory: {path}")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise BuilderError(
            f"{label} must be caller-owned and not group/world-writable: {path}"
        )


def _ensure_owned_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        _require_owned_directory(path, "artifact directory")
        return path
    path.mkdir(parents=True, mode=0o755)
    os.chmod(path, 0o755)
    _require_owned_directory(path, "artifact directory")
    return path


def _ensure_private_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        _require_private_owned_directory(path, "private artifact directory")
        return path
    path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)
    _require_private_owned_directory(path, "private artifact directory")
    return path


def _validate_output_destinations(
    artifact_root: Path,
    selected: Sequence[Mapping[str, Any]],
    precision_profile: PrecisionProfile | None = None,
    runtime_engine_profile: str | None = None,
) -> None:
    checked: set[Path] = set()
    for model in selected:
        paths = _paths_for_profile(
            artifact_root,
            model,
            precision_profile,
            runtime_engine_profile,
        )
        for directory, label in (
            (paths.engine.parent, "engine output directory"),
            (paths.engine_receipt.parent, "engine provenance directory"),
        ):
            if directory in checked:
                continue
            checked.add(directory)
            _reject_descendant_symlinks(artifact_root, directory, label)
            if directory.exists() or directory.is_symlink():
                _require_owned_directory(directory, label)


def _reject_descendant_symlinks(root: Path, path: Path, label: str) -> None:
    root_resolved = root.resolve()
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root_resolved)
    except ValueError as exc:
        raise BuilderError(f"{label} escaped the artifact root: {path}") from exc
    current = root_resolved
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise BuilderError(f"{label} contains a symlink: {current}")


def _explicit_root(raw: str, label: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise BuilderError(f"{label} must be set explicitly")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BuilderError(f"{label} must be absolute: {value}")
    resolved = path.resolve()
    if resolved in {Path("/"), REPO_ROOT.resolve(), DS9_ROOT.resolve()}:
        raise BuilderError(f"refusing unsafe {label}: {resolved}")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise BuilderError(f"{label} must not be inside the checkout: {resolved}")
    return resolved


def _load_matrix(path: Path = MODEL_MATRIX_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuilderError(f"invalid RF-DETR model matrix: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MODEL_MATRIX_SCHEMA:
        raise BuilderError(f"unexpected RF-DETR model matrix schema: {path}")
    release = payload.get("release")
    if (
        not isinstance(release, dict)
        or release.get("version") != RELEASE_VERSION
        or release.get("batch_size") != 3
        or release.get("dynamic_batch") is not False
    ):
        raise BuilderError(
            "RF-DETR 1.8.3 engine builds require the reviewed static-B3 release"
        )
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise BuilderError("RF-DETR model matrix has no model rows")
    ids = [
        _safe_token(row.get("id"), "model ID")
        for row in models
        if isinstance(row, dict)
    ]
    if len(ids) != len(models) or any(not item for item in ids):
        raise BuilderError("RF-DETR model matrix contains an invalid model row")
    if len(set(ids)) != len(ids):
        raise BuilderError("RF-DETR model matrix contains duplicate model IDs")
    for row in models:
        for key, label in (
            ("onnx_filename", "ONNX filename"),
            ("engine_filename", "engine filename"),
            ("checkpoint_filename", "checkpoint filename"),
            ("class_name", "model class"),
            ("package", "package"),
        ):
            _safe_token(row.get(key), label)
        if not str(row.get("checkpoint_url", "")).startswith(
            "https://storage.googleapis.com/rfdetr/"
        ):
            raise BuilderError(f"unexpected checkpoint URL for {row['id']}")
        runtime = row.get("runtime")
        if not isinstance(runtime, Mapping):
            raise BuilderError(
                f"{row['id']} lacks the reviewed runtime-input contract"
            )
        if (
            runtime.get("input_contract") != RUNTIME_INPUT_CONTRACT
            or runtime.get("adapter_revision") != RUNTIME_ADAPTER_REVISION
        ):
            raise BuilderError(
                f"{row['id']} runtime-input adapter contract drifted"
            )
        runtime_onnx = _safe_token(
            runtime.get("onnx_filename"), "runtime ONNX filename"
        )
        if not runtime_onnx.endswith(".onnx"):
            raise BuilderError(
                f"{row['id']} runtime ONNX filename must end in '.onnx'"
            )
        declared_profile = runtime.get("engine_profile")
        declared_engine = runtime.get("engine_filename")
        if (declared_profile is None) != (declared_engine is None):
            raise BuilderError(
                f"{row['id']} runtime engine profile/filename must be "
                "declared together"
            )
        if declared_profile is not None:
            profile = _runtime_engine_profile(declared_profile)
            assert profile is not None
            declared_name = _safe_token(
                declared_engine, "runtime engine filename"
            )
            expected_name = _runtime_engine_filename(row, profile)
            if declared_name != expected_name:
                raise BuilderError(
                    f"{row['id']} runtime engine filename does not match "
                    f"its ONNX/profile-derived name {expected_name!r}"
                )
    return payload


def _split_model_args(raw: Sequence[str]) -> list[str]:
    values: list[str] = []
    for item in raw:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    return values


def _select_models(
    matrix: Mapping[str, Any], requested: Sequence[str]
) -> list[dict[str, Any]]:
    requested_ids = _split_model_args(requested)
    if not requested_ids:
        raise BuilderError(
            "at least one explicit --model ID is required; unscoped builds are forbidden"
        )
    if len(set(requested_ids)) != len(requested_ids):
        raise BuilderError("duplicate RF-DETR model selection")
    by_id = {str(row["id"]): row for row in matrix["models"]}
    unknown = sorted(set(requested_ids) - set(by_id))
    if unknown:
        raise BuilderError(
            f"unknown RF-DETR model ID(s): {', '.join(unknown)}; "
            f"valid IDs: {', '.join(by_id)}"
        )
    disabled = [item for item in requested_ids if not bool(by_id[item].get("enabled"))]
    if disabled:
        reasons = "; ".join(
            f"{item}: {by_id[item].get('blocked_reason', 'disabled')}"
            for item in disabled
        )
        raise BuilderError(
            "licensed/disabled model rows are not authorized by this builder: "
            + reasons
        )
    selected = set(requested_ids)
    return [dict(row) for row in matrix["models"] if row["id"] in selected]


def _artifact_paths(root: Path, model: Mapping[str, Any]) -> ArtifactPaths:
    model_id = _safe_token(model["id"], "model ID")
    onnx_filename = _safe_token(model["onnx_filename"], "ONNX filename")
    engine_filename = _safe_token(model["engine_filename"], "engine filename")
    onnx_root = root / "models" / "onnx" / "rfdetr" / RELEASE_VERSION
    provenance_root = (
        root / "models" / "provenance" / "rfdetr" / RELEASE_VERSION
    )
    engine_root = root / "models" / "engines" / "rfdetr" / RELEASE_VERSION
    return ArtifactPaths(
        onnx=onnx_root / onnx_filename,
        onnx_receipt=provenance_root / f"{model_id}.onnx.json",
        engine=engine_root / engine_filename,
        engine_receipt=provenance_root / f"{model_id}.engine.json",
    )


def _runtime_engine_filename(
    model: Mapping[str, Any], runtime_profile: str
) -> str:
    runtime = model.get("runtime")
    if not isinstance(runtime, Mapping):
        raise BuilderError(f"{model['id']} lacks a runtime-input contract")
    onnx_filename = _safe_token(
        runtime.get("onnx_filename"), "runtime ONNX filename"
    )
    if not onnx_filename.endswith(".onnx"):
        raise BuilderError(
            f"{model['id']} runtime ONNX filename must end in '.onnx'"
        )
    profile_id = _safe_token(runtime_profile, "runtime engine profile")
    if profile_id not in RUNTIME_ENGINE_PROFILES:
        raise BuilderError(
            f"unsupported runtime engine profile {profile_id!r}"
        )
    stem = onnx_filename[: -len(".onnx")]
    return _safe_token(
        f"{stem}_{profile_id}.engine", "derived runtime engine filename"
    )


def _runtime_engine_artifact_paths(
    root: Path,
    model: Mapping[str, Any],
    runtime_profile: str,
) -> ArtifactPaths:
    model_id = _safe_token(model["id"], "model ID")
    runtime = model.get("runtime")
    if not isinstance(runtime, Mapping):
        raise BuilderError(f"{model_id} lacks a runtime-input contract")
    profile_id = _safe_token(runtime_profile, "runtime engine profile")
    onnx_filename = _safe_token(
        runtime.get("onnx_filename"), "runtime ONNX filename"
    )
    adapter_revision = _safe_token(
        runtime.get("adapter_revision"), "runtime adapter revision"
    )
    base = root / "models"
    onnx_root = base / "onnx" / "rfdetr" / RELEASE_VERSION / "runtime"
    engine_root = (
        base
        / "engines"
        / "rfdetr"
        / RELEASE_VERSION
        / "runtime"
        / profile_id
    )
    provenance_root = (
        base
        / "provenance"
        / "rfdetr"
        / RELEASE_VERSION
    )
    runtime_provenance_root = (
        provenance_root / "runtime" / profile_id
    )
    return ArtifactPaths(
        onnx=onnx_root / onnx_filename,
        onnx_receipt=provenance_root
        / f"{model_id}.runtime_onnx.{adapter_revision}.json",
        engine=engine_root
        / _runtime_engine_filename(model, profile_id),
        engine_receipt=runtime_provenance_root / f"{model_id}.engine.json",
    )


def _precision_canary_engine_filename(
    model: Mapping[str, Any], profile: PrecisionProfile
) -> str:
    baseline = _safe_token(model["engine_filename"], "engine filename")
    suffix = "_fp16.engine"
    if not baseline.endswith(suffix):
        raise BuilderError(
            f"{model['id']} baseline engine filename lacks {suffix!r}"
        )
    return f"{baseline[:-len(suffix)]}_{profile.id}.engine"


def _precision_canary_artifact_paths(
    root: Path,
    model: Mapping[str, Any],
    profile: PrecisionProfile,
) -> ArtifactPaths:
    baseline = _artifact_paths(root, model)
    model_id = _safe_token(model["id"], "model ID")
    profile_id = _safe_token(profile.id, "precision canary profile")
    engine_root = (
        root
        / "models"
        / "engines"
        / "rfdetr"
        / RELEASE_VERSION
        / "canaries"
        / profile_id
    )
    receipt_root = (
        root
        / "models"
        / "provenance"
        / "rfdetr"
        / RELEASE_VERSION
        / "canaries"
        / profile_id
    )
    return ArtifactPaths(
        onnx=baseline.onnx,
        onnx_receipt=baseline.onnx_receipt,
        engine=engine_root
        / _precision_canary_engine_filename(model, profile),
        engine_receipt=receipt_root / f"{model_id}.engine.json",
    )


def _paths_for_profile(
    root: Path,
    model: Mapping[str, Any],
    profile: PrecisionProfile | None,
    runtime_engine_profile: str | None = None,
) -> ArtifactPaths:
    if runtime_engine_profile is not None:
        return _runtime_engine_artifact_paths(
            root, model, runtime_engine_profile
        )
    if profile is None:
        return _artifact_paths(root, model)
    return _precision_canary_artifact_paths(root, model, profile)


def _expected_tensor_contract(
    model: Mapping[str, Any], release: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "input": {
            "input": [
                int(release["batch_size"]),
                3,
                int(model["resolution"]),
                int(model["resolution"]),
            ]
        },
        "outputs": {
            str(name): [int(value) for value in shape]
            for name, shape in dict(model["outputs"]).items()
        },
    }


def _validate_onnx_receipt(
    paths: ArtifactPaths,
    model: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_root = paths.onnx.parents[4]
    _reject_descendant_symlinks(artifact_root, paths.onnx, f"{model['id']} ONNX")
    _reject_descendant_symlinks(
        artifact_root,
        paths.onnx_receipt,
        f"{model['id']} ONNX provenance",
    )
    _require_regular_file(paths.onnx, f"{model['id']} ONNX")
    _require_regular_file(paths.onnx_receipt, f"{model['id']} ONNX provenance")
    try:
        receipt = json.loads(paths.onnx_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuilderError(
            f"invalid ONNX provenance receipt: {paths.onnx_receipt}"
        ) from exc
    expected_contract = _expected_tensor_contract(model, release)
    observed_contract = receipt.get("tensor_contract")
    if not isinstance(observed_contract, dict):
        raise BuilderError(f"{model['id']} ONNX receipt lacks a tensor contract")
    checkpoint = receipt.get("checkpoint")
    export = receipt.get("export")
    environment = receipt.get("export_environment")
    if (
        receipt.get("schema") != SOURCE_PROVENANCE_SCHEMA
        or receipt.get("artifact_kind") != "onnx"
        or receipt.get("model_id") != model["id"]
        or receipt.get("family") != model["family"]
        or receipt.get("variant") != model["variant"]
        or receipt.get("license") != model["license"]
        or receipt.get("filename") != paths.onnx.name
        or receipt.get("release") != release
        or observed_contract.get("input") != expected_contract["input"]
        or observed_contract.get("outputs") != expected_contract["outputs"]
        or observed_contract.get("opset_imports", {}).get("ai.onnx")
        != release.get("onnx_opset")
    ):
        raise BuilderError(
            f"{model['id']} ONNX provenance differs from the reviewed matrix"
        )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("filename")
        != _safe_token(model["checkpoint_filename"], "checkpoint filename")
        or checkpoint.get("md5") != model["checkpoint_md5"]
        or re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256") or ""))
        is None
    ):
        raise BuilderError(
            f"{model['id']} ONNX receipt has an invalid checkpoint binding"
        )
    observed_size = paths.onnx.stat().st_size
    observed_sha256 = _sha256(paths.onnx)
    if (
        receipt.get("size_bytes") != observed_size
        or receipt.get("sha256") != observed_sha256
    ):
        raise BuilderError(
            f"{model['id']} ONNX bytes differ from their provenance receipt"
        )
    if (
        not isinstance(export, dict)
        or export.get("class_name") != model["class_name"]
        or export.get("device") != "cpu"
        or export.get("batch_size") != 3
        or export.get("dynamic_batch") is not False
        or export.get("format") != "onnx"
        or export.get("opset") != release["onnx_opset"]
        or export.get("resolution")
        != [model["resolution"], model["resolution"]]
        or export.get("preserve_official_query_count") is not True
    ):
        raise BuilderError(
            f"{model['id']} ONNX is not the reviewed static-B3 export"
        )
    if not isinstance(environment, dict):
        raise BuilderError(f"{model['id']} ONNX receipt lacks export environment")
    try:
        environment_ok = (
            environment.get("rfdetr") == RELEASE_VERSION
            and Version("5.1.0")
            <= Version(str(environment.get("transformers")))
            < Version("6.0.0")
            and Version("1.16.0")
            <= Version(str(environment.get("onnx")))
            < Version("2.0.0")
        )
    except Exception as exc:
        raise BuilderError(
            f"{model['id']} ONNX receipt has invalid export versions"
        ) from exc
    recorded_commit = environment.get("rfdetr_source_commit")
    if recorded_commit is None:
        recorded_commit = release.get("git_commit")
    if not environment_ok or recorded_commit != release.get("git_commit"):
        raise BuilderError(
            f"{model['id']} ONNX receipt has an unreviewed export environment"
        )
    return {
        "path": paths.onnx,
        "size_bytes": observed_size,
        "sha256": observed_sha256,
        "receipt_path": paths.onnx_receipt,
        "receipt_sha256": _sha256(paths.onnx_receipt),
        "tensor_contract": observed_contract,
    }


def _validate_runtime_onnx_receipt(
    artifact_root: Path,
    paths: ArtifactPaths,
    model: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = model.get("runtime")
    if not isinstance(runtime, Mapping):
        raise BuilderError(
            f"{model['id']} lacks the reviewed runtime-input contract"
        )
    if (
        runtime.get("input_contract") != RUNTIME_INPUT_CONTRACT
        or runtime.get("adapter_revision") != RUNTIME_ADAPTER_REVISION
    ):
        raise BuilderError(
            f"{model['id']} runtime-input adapter contract drifted"
        )
    _reject_descendant_symlinks(
        artifact_root, paths.onnx, f"{model['id']} runtime ONNX"
    )
    _reject_descendant_symlinks(
        artifact_root,
        paths.onnx_receipt,
        f"{model['id']} runtime ONNX provenance",
    )
    _require_regular_file(paths.onnx, f"{model['id']} runtime ONNX")
    _require_regular_file(
        paths.onnx_receipt, f"{model['id']} runtime ONNX provenance"
    )
    try:
        receipt = json.loads(paths.onnx_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuilderError(
            f"invalid runtime ONNX provenance receipt: {paths.onnx_receipt}"
        ) from exc

    baseline_paths = _artifact_paths(artifact_root, model)
    baseline = _validate_onnx_receipt(baseline_paths, model, release)
    expected_contract = _expected_tensor_contract(model, release)
    observed_contract = receipt.get("tensor_contract")
    source = receipt.get("source")
    runtime_size = paths.onnx.stat().st_size
    runtime_sha256 = _sha256(paths.onnx)
    if (
        receipt.get("schema") != SOURCE_PROVENANCE_SCHEMA
        or receipt.get("artifact_kind") != "runtime_onnx"
        or receipt.get("model_id") != model["id"]
        or receipt.get("family") != model["family"]
        or receipt.get("variant") != model["variant"]
        or receipt.get("license") != model["license"]
        or receipt.get("release") != release
        or receipt.get("filename") != paths.onnx.name
        or receipt.get("size_bytes") != runtime_size
        or receipt.get("sha256") != runtime_sha256
        or receipt.get("adapter") != RUNTIME_ADAPTER_SPEC
        or not isinstance(observed_contract, Mapping)
        or observed_contract.get("input") != expected_contract["input"]
        or observed_contract.get("outputs") != expected_contract["outputs"]
        or observed_contract.get("opset_imports", {}).get("ai.onnx")
        != release.get("onnx_opset")
        or observed_contract.get("runtime_adapter")
        != RUNTIME_ADAPTER_SPEC
        or not isinstance(
            observed_contract.get("normalized_input_consumer_count"), int
        )
        or observed_contract.get("normalized_input_consumer_count", 0) <= 0
    ):
        raise BuilderError(
            f"{model['id']} runtime ONNX provenance differs from the "
            "reviewed adapter contract"
        )
    if (
        not isinstance(source, Mapping)
        or source.get("filename") != baseline_paths.onnx.name
        or source.get("size_bytes") != baseline["size_bytes"]
        or source.get("sha256") != baseline["sha256"]
        or source.get("receipt") != baseline_paths.onnx_receipt.name
        or source.get("receipt_sha256") != baseline["receipt_sha256"]
    ):
        raise BuilderError(
            f"{model['id']} runtime ONNX source binding drifted"
        )
    return {
        "path": paths.onnx,
        "size_bytes": runtime_size,
        "sha256": runtime_sha256,
        "receipt_path": paths.onnx_receipt,
        "receipt_sha256": _sha256(paths.onnx_receipt),
        "tensor_contract": dict(observed_contract),
        "adapter": dict(RUNTIME_ADAPTER_SPEC),
        "normalized_source": {
            "path": baseline["path"],
            "size_bytes": baseline["size_bytes"],
            "sha256": baseline["sha256"],
            "receipt_path": baseline["receipt_path"],
            "receipt_sha256": baseline["receipt_sha256"],
        },
    }


def _validate_baseline_engine(
    artifact_root: Path,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _artifact_paths(artifact_root, model)
    _reject_descendant_symlinks(
        artifact_root, paths.engine, f"{model['id']} baseline engine"
    )
    _reject_descendant_symlinks(
        artifact_root,
        paths.engine_receipt,
        f"{model['id']} baseline engine provenance",
    )
    _require_regular_file(paths.engine, f"{model['id']} baseline engine")
    _require_regular_file(
        paths.engine_receipt, f"{model['id']} baseline engine provenance"
    )
    try:
        receipt = json.loads(paths.engine_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuilderError(
            f"invalid baseline engine receipt: {paths.engine_receipt}"
        ) from exc
    engine = receipt.get("engine")
    build = receipt.get("build_contract")
    source = receipt.get("source")
    platform = receipt.get("platform")
    if (
        receipt.get("schema") != ENGINE_RECEIPT_SCHEMA
        or receipt.get("promotion_status") != "unpromoted"
        or receipt.get("runtime_selected") is not False
        or receipt.get("model_id") != model["id"]
        or receipt.get("family") != model["family"]
        or receipt.get("variant") != model["variant"]
        or not isinstance(engine, Mapping)
        or engine.get("path")
        != _relative_to_root(paths.engine, artifact_root)
        or engine.get("size_bytes") != paths.engine.stat().st_size
        or engine.get("sha256") != _sha256(paths.engine)
        or not isinstance(build, Mapping)
        or build.get("precision") != "fp16"
        or build.get("batch") != {"mode": "static", "size": 3}
        or not isinstance(source, Mapping)
        or source.get("onnx")
        != _relative_to_root(paths.onnx, artifact_root)
        or source.get("onnx_size_bytes") != paths.onnx.stat().st_size
        or source.get("onnx_sha256") != _sha256(paths.onnx)
        or source.get("onnx_receipt")
        != _relative_to_root(paths.onnx_receipt, artifact_root)
        or source.get("onnx_receipt_sha256")
        != _sha256(paths.onnx_receipt)
        or not isinstance(platform, Mapping)
        or platform.get("tensorrt_version") != REQUIRED_TRT_VERSION
        or platform.get("image_id") != REQUIRED_IMAGE_ID
    ):
        raise BuilderError(
            f"{model['id']} baseline FP16 engine receipt contract drifted"
        )
    return {
        "profile": BASELINE_PRECISION_PROFILE,
        "path": _relative_to_root(paths.engine, artifact_root),
        "size_bytes": paths.engine.stat().st_size,
        "sha256": _sha256(paths.engine),
        "receipt": _relative_to_root(paths.engine_receipt, artifact_root),
        "receipt_sha256": _sha256(paths.engine_receipt),
    }


def _docker_context(root: Path) -> DockerContext:
    socket_path = root / "run" / "docker.sock"
    try:
        mode = socket_path.stat().st_mode
    except OSError as exc:
        raise BuilderError(
            f"secondary Docker socket is unavailable: {socket_path}"
        ) from exc
    if not stat.S_ISSOCK(mode):
        raise BuilderError(f"secondary Docker endpoint is not a socket: {socket_path}")
    docker = shutil.which("docker")
    if not docker:
        raise BuilderError("docker CLI is unavailable")
    return DockerContext(
        root=root,
        socket_path=socket_path,
        command=(docker, "--host", f"unix://{socket_path}"),
    )


def _run_checked(
    command: Sequence[str],
    *,
    label: str,
    timeout_seconds: int = 60,
) -> str:
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuilderError(f"{label} failed: {exc}") from exc
    if result.returncode != 0:
        raise BuilderError(
            f"{label} exited {result.returncode}: {(result.stdout or '').strip()}"
        )
    return result.stdout or ""


def _inspect_secondary_docker(context: DockerContext) -> dict[str, Any]:
    actual_root = _run_checked(
        [
            *context.command,
            "info",
            "--format",
            "{{.DockerRootDir}}",
        ],
        label="secondary Docker info",
    ).strip()
    expected_root = (context.root / "data").resolve()
    try:
        actual_resolved = Path(actual_root).resolve(strict=True)
    except OSError as exc:
        raise BuilderError(
            f"secondary Docker reported an unavailable data root: {actual_root}"
        ) from exc
    if actual_resolved != expected_root:
        raise BuilderError(
            "secondary Docker root mismatch: "
            f"expected={expected_root} observed={actual_resolved}"
        )
    bridge_ids = _run_checked(
        [
            *context.command,
            "network",
            "ls",
            "--filter",
            "driver=bridge",
            "--format",
            "{{.ID}}",
        ],
        label="secondary Docker network inspection",
    ).strip()
    if bridge_ids:
        raise BuilderError(
            "secondary Docker unexpectedly exposes a bridge network"
        )
    raw = _run_checked(
        [*context.command, "image", "inspect", REQUIRED_IMAGE_REF],
        label="DS9 build image inspection",
    )
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BuilderError("DS9 build image inspection returned invalid JSON") from exc
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("Id") != REQUIRED_IMAGE_ID
    ):
        observed = rows[0].get("Id") if isinstance(rows, list) and rows else None
        raise BuilderError(
            "reviewed DS9 build image ID mismatch: "
            f"expected={REQUIRED_IMAGE_ID} observed={observed}"
        )
    config = rows[0].get("Config") or {}
    labels = config.get("Labels") or {}
    image_trt = labels.get("com.nvidia.tensorrt.version")
    base_digest = labels.get("org.opencontainers.image.base.digest")
    environment = {
        key: value
        for entry in (config.get("Env") or [])
        if "=" in entry
        for key, value in [entry.split("=", 1)]
    }
    cuda_version = environment.get("CUDA_VERSION")
    if (
        image_trt != REQUIRED_IMAGE_TRT_VERSION
        or base_digest != REQUIRED_BASE_DIGEST
        or cuda_version != REQUIRED_CUDA_VERSION
    ):
        raise BuilderError(
            "reviewed DS9 build image metadata mismatch: "
            f"tensorrt={image_trt!r} base={base_digest!r} cuda={cuda_version!r}"
        )
    return {
        "docker_root": str(expected_root),
        "docker_socket": str(context.socket_path),
        "image_ref": REQUIRED_IMAGE_REF,
        "image_id": REQUIRED_IMAGE_ID,
        "base_digest": base_digest,
        "image_tensorrt_version": image_trt,
        "cuda_version": cuda_version,
    }


def _container_security_args(*, gpu: bool) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()
    args = [
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user",
        f"{uid}:{gid}",
        "--init",
        "--memory",
        str(MAINTENANCE_MEMORY_BYTES),
        "--memory-swap",
        str(MAINTENANCE_MEMORY_BYTES),
        "--pids-limit",
        str(MAINTENANCE_PIDS_LIMIT),
        "--env",
        "HOME=/tmp/noesis-home",
        "--env",
        "CUDA_CACHE_PATH=/tmp/cuda-cache",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=2147483648",
    ]
    if gpu:
        args.extend(
            [
                "--runtime=nvidia",
                "--gpus",
                f"device={GPU_DEVICE_INDEX}",
            ]
        )
    else:
        args.extend(
            [
                "--runtime=runc",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
            ]
        )
    return args


def _probe_trtexec(context: DockerContext) -> str:
    command = [
        *context.command,
        "run",
        "--rm",
        *_container_security_args(gpu=False),
        "--entrypoint",
        "trtexec",
        REQUIRED_IMAGE_ID,
        "--help",
    ]
    output = _run_checked(command, label="DS9 TensorRT probe", timeout_seconds=120)
    if REQUIRED_TRT_BANNER not in output:
        raise BuilderError(
            f"DS9 TensorRT probe lacks exact banner {REQUIRED_TRT_BANNER!r}"
        )
    return output


def _query_compute_owners() -> str:
    command = [
        "nvidia-smi",
        f"--id={GPU_DEVICE_INDEX}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuilderError(f"unable to query GPU compute owners: {exc}") from exc
    if result.returncode != 0:
        raise BuilderError(
            "unable to query GPU compute owners: "
            + (result.stdout or "").strip()
        )
    return (result.stdout or "").strip()


def _assert_no_compute_owners(label: str) -> None:
    owners = _query_compute_owners()
    if owners:
        raise BuilderError(
            f"GPU compute owner(s) exist {label}; refusing RF-DETR build:\n{owners}"
        )


def _gpu_identity() -> dict[str, str]:
    command = [
        "nvidia-smi",
        f"--id={GPU_DEVICE_INDEX}",
        "--query-gpu=driver_version,name,uuid,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = _run_checked(command, label="GPU identity query", timeout_seconds=30)
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise BuilderError("GPU identity query did not return exactly one device")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 5 or any(not part for part in parts):
        raise BuilderError(f"GPU identity query returned an invalid row: {rows[0]}")
    identity = dict(
        zip(
            (
                "driver_version",
                "gpu_name",
                "gpu_uuid",
                "gpu_compute_capability",
                "gpu_memory_mib",
            ),
            parts,
            strict=True,
        )
    )
    try:
        driver_major = int(identity["driver_version"].split(".", 1)[0])
    except ValueError as exc:
        raise BuilderError(
            f"GPU driver version is invalid: {identity['driver_version']}"
        ) from exc
    if driver_major < 590:
        raise BuilderError(
            f"DS9 engine generation requires driver major >=590, "
            f"found {identity['driver_version']}"
        )
    return identity


def _inner_build_command(
    candidate_name: str,
    *,
    model: Mapping[str, Any] | None = None,
    precision_profile: PrecisionProfile | None = None,
    layer_info_name: str | None = None,
) -> tuple[str, ...]:
    command = [
        "trtexec",
        "--onnx=/inputs/model.onnx",
        *_precision_trtexec_args(precision_profile, model),
        "--memPoolSize=workspace:4096",
    ]
    if precision_profile is not None:
        if not layer_info_name:
            raise BuilderError(
                "precision canary build requires a layer-info filename"
            )
        safe_layer_info = _safe_token(
            layer_info_name, "layer-info filename"
        )
        command.extend(
            (
                "--profilingVerbosity=detailed",
                "--dumpLayerInfo",
                f"--exportLayerInfo=/work/{safe_layer_info}",
            )
        )
    command.extend(
        (
            f"--saveEngine=/work/{candidate_name}",
            "--skipInference",
        )
    )
    return tuple(command)


def _inner_load_command() -> tuple[str, ...]:
    return (
        "trtexec",
        "--loadEngine=/engine/model.engine",
        "--skipInference",
    )


def _validate_trtexec_transcript(value: str, *, require_load: bool) -> None:
    failures = [
        pattern.pattern for pattern in _FAILED_TRANSCRIPT_PATTERNS if pattern.search(value)
    ]
    if failures:
        raise BuilderError(
            "trtexec transcript contains failure marker(s): " + ", ".join(failures)
        )
    if _TRTEXEC_PASSED.search(value) is None:
        raise BuilderError("trtexec transcript lacks the PASSED completion marker")
    if not require_load:
        return
    loaded = _LOADED_ENGINE.search(value)
    if loaded is None or int(loaded.group(1)) <= 0:
        raise BuilderError("trtexec load lacks a positive engine-size marker")
    if _DESERIALIZED.search(value) is None:
        raise BuilderError("trtexec load lacks an engine-deserialized marker")
    if _SKIPPED_INFERENCE.search(value) is None:
        raise BuilderError("trtexec load lacks the skipped-inference marker")


def _container_name(model_id: str, phase: str, run_id: str) -> str:
    safe_id = re.sub(r"[^a-z0-9_.-]+", "-", model_id.lower())
    return f"noesis-rfdetr-183-{safe_id}-{phase}-{run_id.lower()}"[:128]


def _create_and_run_gpu_container(
    context: DockerContext,
    *,
    model_id: str,
    phase: str,
    run_id: str,
    mounts: Sequence[tuple[Path, str, bool]],
    inner_command: Sequence[str],
    timeout_seconds: int,
) -> tuple[int, str]:
    name = _container_name(model_id, phase, run_id)
    create_command = [
        *context.command,
        "create",
        "--name",
        name,
        *_container_security_args(gpu=True),
        "--label",
        "noesis.ds9.role=rfdetr-1.8.3-unpromoted-engine-build",
        "--label",
        f"noesis.ds9.model={model_id}",
        "--label",
        f"noesis.ds9.run={run_id}",
    ]
    for source, destination, writable in mounts:
        mount = f"type=bind,src={source},dst={destination}"
        if not writable:
            mount += ",readonly"
        create_command.extend(["--mount", mount])
    create_command.extend(
        [
            "--entrypoint",
            inner_command[0],
            REQUIRED_IMAGE_ID,
            *inner_command[1:],
        ]
    )

    _assert_no_compute_owners(f"before {model_id} {phase} container creation")
    raw_id = _run_checked(
        create_command,
        label=f"{model_id} {phase} container creation",
        timeout_seconds=120,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", raw_id):
        raise BuilderError(
            f"Docker returned an invalid {model_id} {phase} container ID: {raw_id}"
        )
    try:
        _assert_no_compute_owners(f"before {model_id} {phase} container start")
        started = time.monotonic()
        try:
            result = subprocess.run(
                [*context.command, "start", "--attach", raw_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            raise BuilderError(
                f"{model_id} {phase} exceeded {timeout_seconds} seconds\n{output}"
            ) from exc
        duration = time.monotonic() - started
        output = result.stdout or ""
        if len(output.encode("utf-8", errors="replace")) > MAX_TRANSCRIPT_BYTES:
            raise BuilderError(
                f"{model_id} {phase} transcript exceeded {MAX_TRANSCRIPT_BYTES} bytes"
            )
        print(
            f"[CONTAINER] {model_id} {phase} completed in {duration:.1f}s "
            f"with exit {result.returncode}",
            flush=True,
        )
        return int(result.returncode), output
    finally:
        cleanup = subprocess.run(
            [*context.command, "rm", "-f", raw_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        if cleanup.returncode != 0:
            raise BuilderError(
                f"unable to remove owned build container {raw_id}: "
                f"{(cleanup.stdout or '').strip()}"
            )


def _record_command(
    run_dir: Path,
    *,
    label: str,
    inner_command: Sequence[str],
    returncode: int,
    transcript: str,
    started: float,
) -> CommandEvidence:
    log_path = run_dir / f"{label}.log"
    _write_text_exclusive(log_path, transcript)
    return CommandEvidence(
        label=label,
        inner_command=tuple(inner_command),
        returncode=returncode,
        duration_seconds=time.monotonic() - started,
        log_path=log_path,
        log_sha256=_sha256(log_path),
    )


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _command_receipt(record: CommandEvidence, root: Path) -> dict[str, Any]:
    return {
        "label": record.label,
        "command": list(record.inner_command),
        "returncode": record.returncode,
        "duration_seconds": record.duration_seconds,
        "log": _relative_to_root(record.log_path, root),
        "log_sha256": record.log_sha256,
        "status": "passed",
    }


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise BuilderError(
                f"RF-DETR build lock must be caller-owned and mode 0600: {path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BuilderError(
                f"another RF-DETR 1.8.3 engine build owns {path}"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def _install_candidate(candidate: Path, target: Path) -> dict[str, Any]:
    _require_regular_file(candidate, "candidate engine")
    info = candidate.stat()
    if info.st_uid != os.getuid() or info.st_nlink != 1:
        raise BuilderError("candidate engine must be caller-owned and single-link")
    if target.exists() or target.is_symlink():
        raise BuilderError(
            f"refusing to replace an existing versioned engine: {target}"
        )
    if candidate.stat().st_dev != target.parent.stat().st_dev:
        raise BuilderError("candidate and target are not on the same filesystem")
    candidate_sha256 = _sha256(candidate)
    candidate_size = candidate.stat().st_size
    os.chmod(candidate, 0o600)
    _fsync_file(candidate)
    os.replace(candidate, target)
    _fsync_directory(target.parent)
    if (
        target.stat().st_size != candidate_size
        or _sha256(target) != candidate_sha256
    ):
        raise BuilderError("installed engine differs from the validated candidate")
    return {
        "size_bytes": candidate_size,
        "sha256": candidate_sha256,
        "mode": "0600",
    }


def _install_layer_info(source: Path, target: Path) -> dict[str, Any]:
    _require_regular_file(source, "TensorRT layer-info evidence")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuilderError("TensorRT layer-info evidence is not valid JSON") from exc
    if not isinstance(payload, (dict, list)) or not payload:
        raise BuilderError("TensorRT layer-info evidence is empty")
    if target.exists() or target.is_symlink():
        raise BuilderError(f"layer-info evidence already exists: {target}")
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise BuilderError("layer-info evidence crossed filesystems")
    source_sha256 = _sha256(source)
    source_size = source.stat().st_size
    os.chmod(source, 0o600)
    _fsync_file(source)
    os.replace(source, target)
    _fsync_directory(target.parent)
    if (
        target.stat().st_size != source_size
        or _sha256(target) != source_sha256
    ):
        raise BuilderError("installed TensorRT layer-info evidence changed")
    return {
        "path": target,
        "size_bytes": source_size,
        "sha256": source_sha256,
        "mode": "0600",
        "format": "trtexec_export_layer_info_json",
    }


def _build_contract(
    model: Mapping[str, Any],
    precision_profile: PrecisionProfile | None,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "batch": {
            "mode": "static",
            "size": 3,
        },
        "workspace_mib": 4096,
        "builder": "trtexec",
        "candidate_and_installed_deserialize_checks": True,
    }
    if precision_profile is None:
        return {
            "precision": "fp16",
            **common,
        }
    layer_precisions = (
        list(_selective_fp32_layer_specs(model))
        if precision_profile.selective_fp32_heads
        else []
    )
    return {
        "precision": precision_profile.precision,
        "profile": precision_profile.id,
        "fp16_enabled": precision_profile.fp16_enabled,
        "tf32_enabled": precision_profile.tf32_enabled,
        "strongly_typed": False,
        "precision_constraints": (
            "obey"
            if precision_profile.selective_fp32_heads
            else "none"
        ),
        "layer_precisions": layer_precisions,
        "trtexec_precision_args": list(
            _precision_trtexec_args(precision_profile, model)
        ),
        "layer_info_exported": True,
        **common,
    }


def _runtime_build_contract(
    model: Mapping[str, Any],
    runtime_engine_profile: str,
    precision_profile: PrecisionProfile | None,
) -> dict[str, Any]:
    if runtime_engine_profile == BASELINE_PRECISION_PROFILE:
        if precision_profile is not None:
            raise BuilderError(
                "baseline runtime profile cannot use canary precision flags"
            )
        baseline = _build_contract(model, None)
        return {
            **baseline,
            "profile": BASELINE_PRECISION_PROFILE,
            "fp16_enabled": True,
            "tf32_enabled": True,
            "strongly_typed": False,
            "precision_constraints": "none",
            "layer_precisions": [],
            "trtexec_precision_args": ["--fp16"],
            "layer_info_exported": False,
        }
    if precision_profile is None or precision_profile.id != runtime_engine_profile:
        raise BuilderError(
            f"runtime profile {runtime_engine_profile!r} lacks its exact "
            "precision contract"
        )
    return _build_contract(model, precision_profile)


def _build_one(
    context: DockerContext,
    *,
    artifact_root: Path,
    model: Mapping[str, Any],
    release: Mapping[str, Any],
    source: Mapping[str, Any],
    platform: Mapping[str, Any],
    build_timeout_seconds: int,
    load_timeout_seconds: int,
    precision_profile: PrecisionProfile | None = None,
    baseline_engine: Mapping[str, Any] | None = None,
    runtime_engine_profile: str | None = None,
) -> None:
    if runtime_engine_profile is not None and baseline_engine is not None:
        raise BuilderError("runtime-input engine must not bind a baseline engine")
    if (
        runtime_engine_profile is None
        and precision_profile is None
        and baseline_engine is not None
    ):
        raise BuilderError("canonical build cannot bind a canary baseline")
    if (
        runtime_engine_profile is None
        and precision_profile is not None
        and baseline_engine is None
    ):
        raise BuilderError("precision canary build lacks its baseline engine")
    if runtime_engine_profile is not None:
        expected_precision = _runtime_precision_profile(
            runtime_engine_profile
        )
        if precision_profile != expected_precision:
            raise BuilderError(
                f"runtime engine profile {runtime_engine_profile!r} "
                "does not match its precision flags"
            )
    paths = _paths_for_profile(
        artifact_root,
        model,
        precision_profile,
        runtime_engine_profile,
    )
    if paths.engine.exists() or paths.engine.is_symlink():
        raise BuilderError(
            f"versioned engine already exists; refusing overwrite: {paths.engine}"
        )
    if paths.engine_receipt.exists() or paths.engine_receipt.is_symlink():
        raise BuilderError(
            f"versioned engine receipt already exists: {paths.engine_receipt}"
        )

    run_id = _run_id()
    _reject_descendant_symlinks(
        artifact_root, paths.engine, f"{model['id']} engine target"
    )
    _reject_descendant_symlinks(
        artifact_root,
        paths.engine_receipt,
        f"{model['id']} engine provenance",
    )
    engine_root = _ensure_owned_directory(paths.engine.parent)
    _ensure_owned_directory(paths.engine_receipt.parent)
    evidence_root_path = (
        artifact_root
        / "models"
        / "engine_evidence"
        / "rfdetr"
        / RELEASE_VERSION
    )
    if runtime_engine_profile is not None:
        evidence_root_path = (
            evidence_root_path / "runtime" / runtime_engine_profile
        )
    elif precision_profile is not None:
        evidence_root_path = (
            evidence_root_path / "canaries" / precision_profile.id
        )
    _reject_descendant_symlinks(
        artifact_root, evidence_root_path, f"{model['id']} evidence root"
    )
    evidence_root = _ensure_private_directory(evidence_root_path)
    run_dir = evidence_root / f"{run_id}-{model['id']}"
    run_dir.mkdir(mode=0o700)
    os.chmod(run_dir, 0o700)
    work_root = _ensure_private_directory(engine_root / ".transactions")
    work_dir = work_root / f"{run_id}-{model['id']}"
    work_dir.mkdir(mode=0o700)
    os.chmod(work_dir, 0o700)
    candidate = work_dir / f"{paths.engine.name}.candidate"
    layer_info_candidate = (
        work_dir / f"{paths.engine.name}.layers.json"
        if precision_profile is not None
        else None
    )
    build_inner = _inner_build_command(
        candidate.name,
        model=model,
        precision_profile=precision_profile,
        layer_info_name=(
            layer_info_candidate.name
            if layer_info_candidate is not None
            else None
        ),
    )
    load_inner = _inner_load_command()
    records: list[CommandEvidence] = []
    installed = False
    layer_info_record: dict[str, Any] | None = None

    try:
        started = time.monotonic()
        build_code, build_output = _create_and_run_gpu_container(
            context,
            model_id=str(model["id"]),
            phase="build",
            run_id=run_id,
            mounts=(
                (paths.onnx, "/inputs/model.onnx", False),
                (work_dir, "/work", True),
            ),
            inner_command=build_inner,
            timeout_seconds=build_timeout_seconds,
        )
        records.append(
            _record_command(
                run_dir,
                label="build",
                inner_command=build_inner,
                returncode=build_code,
                transcript=build_output,
                started=started,
            )
        )
        if build_code != 0:
            raise BuilderError(f"{model['id']} trtexec build exited {build_code}")
        _validate_trtexec_transcript(build_output, require_load=False)
        _require_regular_file(candidate, f"{model['id']} candidate engine")
        if layer_info_candidate is not None:
            layer_info_record = _install_layer_info(
                layer_info_candidate, run_dir / "layer-info.json"
            )

        started = time.monotonic()
        candidate_code, candidate_output = _create_and_run_gpu_container(
            context,
            model_id=str(model["id"]),
            phase="candidate-load",
            run_id=run_id,
            mounts=((candidate, "/engine/model.engine", False),),
            inner_command=load_inner,
            timeout_seconds=load_timeout_seconds,
        )
        records.append(
            _record_command(
                run_dir,
                label="load-candidate",
                inner_command=load_inner,
                returncode=candidate_code,
                transcript=candidate_output,
                started=started,
            )
        )
        if candidate_code != 0:
            raise BuilderError(
                f"{model['id']} candidate deserialize exited {candidate_code}"
            )
        _validate_trtexec_transcript(candidate_output, require_load=True)

        engine_record = _install_candidate(candidate, paths.engine)
        installed = True
        print(f"[INSTALL] unpromoted {model['id']}: {paths.engine}", flush=True)

        started = time.monotonic()
        installed_code, installed_output = _create_and_run_gpu_container(
            context,
            model_id=str(model["id"]),
            phase="installed-load",
            run_id=run_id,
            mounts=((paths.engine, "/engine/model.engine", False),),
            inner_command=load_inner,
            timeout_seconds=load_timeout_seconds,
        )
        records.append(
            _record_command(
                run_dir,
                label="load-installed",
                inner_command=load_inner,
                returncode=installed_code,
                transcript=installed_output,
                started=started,
            )
        )
        if installed_code != 0:
            raise BuilderError(
                f"{model['id']} installed deserialize exited {installed_code}"
            )
        _validate_trtexec_transcript(installed_output, require_load=True)
        if (
            paths.engine.stat().st_size != engine_record["size_bytes"]
            or _sha256(paths.engine) != engine_record["sha256"]
        ):
            raise BuilderError(
                f"{model['id']} installed engine changed after deserialize"
            )

        if runtime_engine_profile is not None:
            source_receipt = {
                "runtime_onnx": _relative_to_root(
                    paths.onnx, artifact_root
                ),
                "runtime_onnx_size_bytes": source["size_bytes"],
                "runtime_onnx_sha256": source["sha256"],
                "runtime_onnx_receipt": _relative_to_root(
                    paths.onnx_receipt, artifact_root
                ),
                "runtime_onnx_receipt_sha256": source["receipt_sha256"],
                "tensor_contract": source["tensor_contract"],
            }
        else:
            source_receipt = {
                "onnx": _relative_to_root(paths.onnx, artifact_root),
                "onnx_size_bytes": source["size_bytes"],
                "onnx_sha256": source["sha256"],
                "onnx_receipt": _relative_to_root(
                    paths.onnx_receipt, artifact_root
                ),
                "onnx_receipt_sha256": source["receipt_sha256"],
                "tensor_contract": source["tensor_contract"],
            }
        receipt: dict[str, Any] = {
            "schema": (
                RUNTIME_ENGINE_RECEIPT_SCHEMA
                if runtime_engine_profile is not None
                else (
                    CANARY_ENGINE_RECEIPT_SCHEMA
                    if precision_profile is not None
                    else ENGINE_RECEIPT_SCHEMA
                )
            ),
            "promotion_status": "unpromoted",
            "runtime_selected": False,
            "model_id": model["id"],
            "family": model["family"],
            "variant": model["variant"],
            "release": dict(release),
            "source": source_receipt,
            "build_contract": (
                _runtime_build_contract(
                    model,
                    runtime_engine_profile,
                    precision_profile,
                )
                if runtime_engine_profile is not None
                else _build_contract(model, precision_profile)
            ),
            "container_contract": {
                "network": "none",
                "root_filesystem": "read_only",
                "capabilities": "drop_all",
                "no_new_privileges": True,
                "user": f"{os.getuid()}:{os.getgid()}",
                "runtime": "nvidia",
                "gpu_device": GPU_DEVICE_INDEX,
                "memory_bytes": MAINTENANCE_MEMORY_BYTES,
                "memory_swap_bytes": MAINTENANCE_MEMORY_BYTES,
                "pids_limit": MAINTENANCE_PIDS_LIMIT,
            },
            "platform": {
                **dict(platform),
                "expected_trtexec_banner": REQUIRED_TRT_BANNER,
            },
            "engine": {
                "path": _relative_to_root(paths.engine, artifact_root),
                **engine_record,
            },
            "commands": [
                _command_receipt(record, artifact_root) for record in records
            ],
            "evidence_directory": _relative_to_root(run_dir, artifact_root),
            "recorded_at_utc": _utc_now(),
        }
        if runtime_engine_profile is not None:
            receipt["artifact_role"] = "runtime_input_engine"
            receipt["runtime_input_contract"] = {
                "input_contract": RUNTIME_INPUT_CONTRACT,
                "adapter_revision": RUNTIME_ADAPTER_REVISION,
                "adapter": dict(source["adapter"]),
                "normalized_source": {
                    "onnx": _relative_to_root(
                        Path(source["normalized_source"]["path"]),
                        artifact_root,
                    ),
                    "onnx_size_bytes": source["normalized_source"][
                        "size_bytes"
                    ],
                    "onnx_sha256": source["normalized_source"]["sha256"],
                    "onnx_receipt": _relative_to_root(
                        Path(source["normalized_source"]["receipt_path"]),
                        artifact_root,
                    ),
                    "onnx_receipt_sha256": source["normalized_source"][
                        "receipt_sha256"
                    ],
                },
            }
        if precision_profile is not None:
            if layer_info_record is None:
                role = (
                    "canary"
                    if runtime_engine_profile is None
                    else "runtime nonbaseline build"
                )
                raise BuilderError(
                    f"{model['id']} {role} lacks layer-info evidence"
                )
            if runtime_engine_profile is None:
                receipt["artifact_role"] = "precision_canary"
                receipt["baseline_engine"] = dict(baseline_engine or {})
            receipt["layer_info"] = {
                **layer_info_record,
                "path": _relative_to_root(
                    Path(layer_info_record["path"]), artifact_root
                ),
            }
        receipt["receipt_sha256"] = _json_digest(receipt)
        _write_json_atomic(paths.engine_receipt, receipt)
        print(
            f"[OK] unpromoted {model['id']}: "
            f"sha256={engine_record['sha256']} receipt={paths.engine_receipt}",
            flush=True,
        )
    except BaseException:
        if installed and paths.engine.is_file() and not paths.engine.is_symlink():
            rollback = work_dir / f"{paths.engine.name}.failed-installed"
            if rollback.exists() or rollback.is_symlink():
                raise BuilderError(
                    f"rollback destination unexpectedly exists: {rollback}"
                )
            os.replace(paths.engine, rollback)
            _fsync_directory(paths.engine.parent)
            installed = False
        raise
    finally:
        if not any(work_dir.iterdir()):
            work_dir.rmdir()
        if work_root.is_dir() and not any(work_root.iterdir()):
            work_root.rmdir()


def _plan(
    *,
    artifact_root: Path,
    selected: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    platform: Mapping[str, Any],
    precision_profile: PrecisionProfile | None = None,
    baselines: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_engine_profile: str | None = None,
) -> None:
    rows = []
    for model in selected:
        paths = _paths_for_profile(
            artifact_root,
            model,
            precision_profile,
            runtime_engine_profile,
        )
        layer_info_name = (
            f"{paths.engine.name}.layers.json"
            if precision_profile is not None
            else None
        )
        row: dict[str, Any] = {
            "model_id": model["id"],
            "family": model["family"],
            "variant": model["variant"],
            "resolution": model["resolution"],
            "onnx": str(paths.onnx),
            "onnx_sha256": sources[str(model["id"])]["sha256"],
            "engine": str(paths.engine),
            "build_command": list(
                _inner_build_command(
                    f"{paths.engine.name}.candidate",
                    model=model,
                    precision_profile=precision_profile,
                    layer_info_name=layer_info_name,
                )
            ),
            "candidate_load_command": list(_inner_load_command()),
            "installed_load_command": list(_inner_load_command()),
            "promotion_status": "unpromoted",
        }
        if runtime_engine_profile is not None:
            row["artifact_role"] = "runtime_input_engine"
            row["runtime_engine_profile"] = runtime_engine_profile
            row["engine_receipt"] = str(paths.engine_receipt)
            row["runtime_input_contract"] = {
                "input_contract": RUNTIME_INPUT_CONTRACT,
                "adapter_revision": RUNTIME_ADAPTER_REVISION,
                "adapter": dict(
                    sources[str(model["id"])]["adapter"]
                ),
            }
            if precision_profile is not None:
                row["layer_info"] = str(
                    artifact_root
                    / "models"
                    / "engine_evidence"
                    / "rfdetr"
                    / RELEASE_VERSION
                    / "runtime"
                    / runtime_engine_profile
                    / f"<run-id>-{model['id']}"
                    / "layer-info.json"
                )
        elif precision_profile is not None:
            baseline = (baselines or {}).get(str(model["id"]))
            if baseline is None:
                raise BuilderError(
                    f"{model['id']} canary plan lacks its baseline engine"
                )
            row["artifact_role"] = "precision_canary"
            row["precision_profile"] = precision_profile.id
            row["engine_receipt"] = str(paths.engine_receipt)
            row["baseline_engine"] = dict(baseline)
            row["layer_info"] = str(
                artifact_root
                / "models"
                / "engine_evidence"
                / "rfdetr"
                / RELEASE_VERSION
                / "canaries"
                / precision_profile.id
                / f"<run-id>-{model['id']}"
                / "layer-info.json"
            )
        rows.append(row)
    envelope: dict[str, Any] = {
        "schema": "noesis.ds9.rfdetr-engine-build-plan.v1",
        "plan_is_artifact_read_only": True,
        "gpu_visible_to_probe": False,
        "release": RELEASE_VERSION,
        "platform": dict(platform),
        "models": rows,
    }
    if precision_profile is not None:
        if runtime_engine_profile is None:
            envelope["artifact_role"] = "precision_canary"
            envelope["precision_profile"] = precision_profile.id
    if runtime_engine_profile is not None:
        envelope["artifact_role"] = "runtime_input_engine"
        envelope["runtime_engine_profile"] = runtime_engine_profile
    print(
        json.dumps(
            envelope,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Explicit model ID from DS9/config/rfdetr_1_8_3_models.json. "
            "Repeat the option or pass comma-separated IDs."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Validate inputs/image and print the build plan without opening "
            "the GPU or writing artifact files."
        ),
    )
    build_kind = parser.add_mutually_exclusive_group()
    build_kind.add_argument(
        "--precision-canary-profile",
        choices=tuple(sorted(CANARY_PRECISION_PROFILES)),
        default="",
        help=(
            "Build a collision-proof, unpromoted precision canary instead "
            "of the canonical FP16 engine. Canary engines, receipts, and "
            "layer-info evidence are isolated by profile."
        ),
    )
    build_kind.add_argument(
        "--runtime-engine-profile",
        choices=RUNTIME_ENGINE_PROFILES,
        default="",
        help=(
            "Build a collision-proof, unpromoted engine from the reviewed "
            "RGB-[0,1] runtime ONNX adapter. Runtime engines and receipts "
            "are isolated below runtime/<profile>; nonbaseline profiles "
            "also retain TensorRT layer-info evidence."
        ),
    )
    parser.add_argument(
        "--build-timeout-seconds",
        type=int,
        default=7200,
    )
    parser.add_argument(
        "--load-timeout-seconds",
        type=int,
        default=300,
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    if os.getuid() == 0:
        raise SystemExit("RF-DETR engine generation must run as a non-root caller")
    if args.build_timeout_seconds < 600:
        raise SystemExit("--build-timeout-seconds must be at least 600")
    if args.load_timeout_seconds < 60:
        raise SystemExit("--load-timeout-seconds must be at least 60")

    try:
        artifact_root = _explicit_root(
            os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", ""),
            "NOESIS_DS9_ARTIFACT_ROOT",
        )
        docker_root = _explicit_root(
            os.environ.get("NOESIS_DS9_DOCKER_ROOT", ""),
            "NOESIS_DS9_DOCKER_ROOT",
        )
        if (
            artifact_root == docker_root
            or artifact_root in docker_root.parents
            or docker_root in artifact_root.parents
        ):
            raise BuilderError(
                "NOESIS_DS9_ARTIFACT_ROOT and NOESIS_DS9_DOCKER_ROOT "
                "must be disjoint"
            )
        if not artifact_root.is_dir():
            raise BuilderError(f"artifact root is unavailable: {artifact_root}")

        matrix = _load_matrix()
        selected = _select_models(matrix, args.model)
        runtime_engine_profile = _runtime_engine_profile(
            args.runtime_engine_profile
        )
        precision_profile = (
            _runtime_precision_profile(runtime_engine_profile)
            if runtime_engine_profile is not None
            else _precision_profile(args.precision_canary_profile)
        )
        release = dict(matrix["release"])
        if runtime_engine_profile is not None:
            sources = {
                str(model["id"]): _validate_runtime_onnx_receipt(
                    artifact_root,
                    _runtime_engine_artifact_paths(
                        artifact_root,
                        model,
                        runtime_engine_profile,
                    ),
                    model,
                    release,
                )
                for model in selected
            }
        else:
            sources = {
                str(model["id"]): _validate_onnx_receipt(
                    _artifact_paths(artifact_root, model),
                    model,
                    release,
                )
                for model in selected
            }
        baselines = (
            {
                str(model["id"]): _validate_baseline_engine(
                    artifact_root, model
                )
                for model in selected
            }
            if (
                precision_profile is not None
                and runtime_engine_profile is None
            )
            else {}
        )
        _validate_output_destinations(
            artifact_root,
            selected,
            precision_profile,
            runtime_engine_profile,
        )

        context = _docker_context(docker_root)
        platform = _inspect_secondary_docker(context)
        probe = _probe_trtexec(context)
        platform["tensorrt_version"] = REQUIRED_TRT_VERSION
        platform["trtexec_banner"] = REQUIRED_TRT_BANNER
        if args.plan:
            _plan(
                artifact_root=artifact_root,
                selected=selected,
                sources=sources,
                platform=platform,
                precision_profile=precision_profile,
                baselines=baselines,
                runtime_engine_profile=runtime_engine_profile,
            )
            return 0

        _assert_no_compute_owners("before the RF-DETR build transaction")
        platform.update(_gpu_identity())
        logs_root_path = artifact_root / "logs" / "rfdetr" / RELEASE_VERSION
        _reject_descendant_symlinks(
            artifact_root, logs_root_path, "RF-DETR TensorRT probe logs"
        )
        logs_root = _ensure_private_directory(logs_root_path)
        probe_path = logs_root / f"{_run_id()}-trtexec-probe.log"
        _write_text_exclusive(probe_path, probe)
        platform["trtexec_probe_log"] = _relative_to_root(
            probe_path, artifact_root
        )
        platform["trtexec_probe_log_sha256"] = _sha256(probe_path)
        lock_path = artifact_root / ".noesis-ds9-rfdetr-1.8.3-engine-build.lock"
        with _exclusive_lock(lock_path):
            for model in selected:
                _build_one(
                    context,
                    artifact_root=artifact_root,
                    model=model,
                    release=release,
                    source=sources[str(model["id"])],
                    platform=platform,
                    build_timeout_seconds=args.build_timeout_seconds,
                    load_timeout_seconds=args.load_timeout_seconds,
                    precision_profile=precision_profile,
                    baseline_engine=baselines.get(str(model["id"])),
                    runtime_engine_profile=runtime_engine_profile,
                )
        if runtime_engine_profile is not None:
            print(
                "[OK] RF-DETR 1.8.3 unpromoted DS9 runtime-input engine "
                f"profile {runtime_engine_profile} build complete; no "
                "canonical engine, asset-manifest, realization, runtime, or "
                "model-selection authority was changed"
            )
        elif precision_profile is None:
            print(
                "[OK] RF-DETR 1.8.3 unpromoted DS9 engine build complete; "
                "no asset-manifest, realization, runtime, or model-selection "
                "authority was changed"
            )
        else:
            print(
                "[OK] RF-DETR 1.8.3 unpromoted DS9 precision canary "
                f"{precision_profile.id} build complete; no canonical engine, "
                "asset-manifest, realization, runtime, or model-selection "
                "authority was changed"
            )
        return 0
    except BuilderError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
