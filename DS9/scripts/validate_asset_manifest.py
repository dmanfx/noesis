#!/usr/bin/env python3
"""Validate DS9 artifact ownership, compatibility, staging, and provenance."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import legacy_gpu_memory_guard_receipt as legacy_gpu_guard  # noqa: E402
from noesis_core.strict_json import (  # noqa: E402
    StrictJSONError,
    strict_json_loads,
)


DEFAULT_MANIFEST = REPO_ROOT / "DS9" / "asset_manifest.yaml"
SOURCE_CONTRACTS = REPO_ROOT / "DS9" / "config" / "engine_source_contracts.json"
REALIZATION_FILENAME = "asset_realization.json"
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ROOT_KEYS = {
    "schema_version",
    "manifest_id",
    "schema",
    "updated_at",
    "target",
    "runtime",
    "policy",
    "artifacts",
    "shared_app_data",
    "packaging_boundary",
}
REQUIRED_ARTIFACT_KEYS = {
    "id",
    "kind",
    "role",
    "output",
    "sources",
    "builder",
    "required_profiles",
    "state",
    "compatibility",
    "provenance",
}
REQUIRED_PROVENANCE_KEYS = {
    "source_sha256",
    "output_sha256",
    "built_at_utc",
    "build_host",
    "command",
}
REQUIRED_ENGINE_MAINTENANCE_KEYS = {
    "manifest",
    "output_size_bytes",
    "image",
    "image_id",
    "base_digest",
    "tensorrt_version",
    "cuda_version",
    "driver_version",
    "gpu",
    "precision",
    "batch",
    "tensor_contract",
}
ALLOWED_KINDS = {
    "native_extension",
    "gstreamer_plugin",
    "tensorrt_plugin",
    "nvinfer_parser",
    "tensorrt_engine",
}
MAX_PRIVATE_JSON_BYTES = 8 * 1024 * 1024
MAX_PRIVATE_LOG_BYTES = 16 * 1024 * 1024
WHOLEBODY_BUILDER_SOURCE = (
    REPO_ROOT
    / "DS9"
    / "csrc"
    / "wholebody49_engine_builder"
    / "wholebody49_engine_builder.cpp"
)
WHOLEBODY_BUILDER_CONTRACT = "noesis.ds9.wholebody49_builder.v1"
WHOLEBODY_WORKSPACE_BYTES_BY_VARIANT = {
    "s_masks": 6442450944,
    "x_boxes": 4294967296,
}
WHOLEBODY_BUILDER_OPTIMIZATION_LEVEL_BY_VARIANT = {
    "s_masks": 0,
    "x_boxes": 3,
}
WHOLEBODY_LOGGER_POLICY = {
    "minimum_severity": "info",
    "verbose": "ignored_before_copy",
    "captured_message_truncation": "fatal",
    "error_state": "sticky_fatal",
}
# Historical container receipts may carry a sealed NVML guard. Native-host
# maintenance never emits this legacy proof; the verifier below is read-only.
LEGACY_GPU_MEMORY_GUARD_EXEMPTIONS: dict[str, dict[str, str]] = {}
UNCONDITIONALLY_GUARD_REQUIRED_ARTIFACT_IDS = {
    "engine.wholebody49_s_masks",
    "engine.wholebody49_x_boxes",
}
NATIVE_HOST_AUTHORITY = {
    "operating_system": "ubuntu-24.04",
    "architecture": "x86_64",
    "driver_minimum": "595.58.03",
    "gstreamer": "1.24.2",
    "sdk_root": "/opt/nvidia/deepstream/deepstream-9.1",
    "cuda_root": "/usr/local/cuda-13.2",
    "native_root_env": "NOESIS_DS91_NATIVE_ROOT",
}
NATIVE_RUNTIME_AUTHORITY = {
    "backend": "native_host",
    "supervisor": "DS9/scripts/run_canonical_runtime_host.py",
    "artifact_root_env": "NOESIS_DS9_ARTIFACT_ROOT",
    "runtime_root_env": "NOESIS_DS9_RUNTIME_ROOT",
}
# Read-only compatibility for immutable engine receipts created before the
# native-host cutover. This does not authorize a build or runtime backend.
LEGACY_ENGINE_RECEIPT_PLATFORM = {
    "image": "noesis-ds9-dev:9.1-20260812",
    "image_id": "sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872",
    "base_digest": "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994",
    "tensorrt_version": "10.16.0.72",
    "cuda_version": "13.2.0.046",
}
NATIVE_ENGINE_RECEIPT_PLATFORM = {
    "image": "native_host",
    "image_id": "native_host",
    "base_digest": "native_host",
    "tensorrt_version": "10.16.0.72",
    "cuda_version": "13.2",
}
NATIVE_HOST_BUILD_AUTHORITY_KEYS = {
    "backend",
    "deepstream",
    "cuda",
    "tensorrt",
    "compiler",
    "python_abi",
    "source_sha256",
    "output_sha256",
    "command",
}


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _bounded_private_path(path: str | Path, root: Path, label: str) -> Path:
    """Return an exact lexical path below root without following symlinks."""

    candidate = _lexical_absolute(path)
    root = root.resolve(strict=True)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its private root: {candidate}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} contains a symlink: {current}")
    return candidate


def _read_private_bytes(path: str | Path, label: str) -> bytes:
    """Secure-read one stable owner-only regular file."""

    candidate = _lexical_absolute(path)
    try:
        lexical = candidate.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular file: {candidate}")
        if (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino):
            raise ValueError(f"{label} changed while it was opened: {candidate}")
        if opened.st_uid != os.getuid() or opened.st_nlink != 1:
            raise ValueError(f"{label} must be owned by the current uid with one link")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError(f"{label} must have mode 0600")
        if opened.st_size <= 0 or opened.st_size > MAX_PRIVATE_JSON_BYTES:
            raise ValueError(
                f"{label} size is outside the accepted evidence bound: {opened.st_size}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_PRIVATE_JSON_BYTES + 1)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) or len(raw) != opened.st_size:
            raise ValueError(f"{label} changed while it was read: {candidate}")
    finally:
        os.close(descriptor)
    return raw


def _read_private_log_bytes(path: str | Path, label: str) -> bytes:
    """Secure-read one stable owner-only evidence log, including an empty log."""

    candidate = _lexical_absolute(path)
    try:
        lexical = candidate.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ValueError(f"{label} must be an owner-only single-link regular file")
        if (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino):
            raise ValueError(f"{label} changed while it was opened: {candidate}")
        if opened.st_size < 0 or opened.st_size > MAX_PRIVATE_LOG_BYTES:
            raise ValueError(
                f"{label} size is outside the accepted evidence bound: {opened.st_size}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_PRIVATE_LOG_BYTES + 1)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) or len(raw) != opened.st_size:
            raise ValueError(f"{label} changed while it was read: {candidate}")
    finally:
        os.close(descriptor)
    return raw


def _load_private_json_with_bytes(
    path: str | Path, label: str
) -> tuple[Mapping[str, Any], bytes]:
    """Secure-read owner-only JSON and return the exact validated bytes."""

    raw = _read_private_bytes(path, label)
    try:
        payload = strict_json_loads(raw, label=label)
    except StrictJSONError as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} document root must be a mapping")
    return payload, raw


def _load_private_json(path: str | Path, label: str) -> Mapping[str, Any]:
    payload, _raw = _load_private_json_with_bytes(path, label)
    return payload


def _load_maintenance_manifest(path: Path, artifact_id: str) -> Mapping[str, Any]:
    payload = strict_json_loads(
        path.read_bytes(),
        label=f"{artifact_id} maintenance manifest",
    )
    if not isinstance(payload, Mapping):
        raise ValueError("engine maintenance manifest root must be a mapping")
    return payload


def _load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a mapping at document root: {path}")
    return payload


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate explicit mapping keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    explicit_keys: set[Any] = set()
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in explicit_keys
        except TypeError as exc:
            raise ValueError("YAML mapping key is not hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        explicit_keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _parse_yaml_mapping(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{label} is not valid unique-key UTF-8 YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} document root must be a mapping")
    return payload


def _relative_path(raw: Any) -> Path:
    text = str(raw or "").strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be non-empty and repository-relative: {raw!r}")
    return path


def _path_below(raw: str, roots: list[Path]) -> bool:
    path = _relative_path(raw)
    static_parts: list[str] = []
    for part in path.parts:
        if any(token in part for token in "*?["):
            break
        static_parts.append(part)
    static_path = Path(*static_parts)
    for root in roots:
        try:
            static_path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _physical_path(relative: Path, artifact_root: Path | None) -> Path:
    if artifact_root is not None and tuple(relative.parts[:2]) == ("DS9", "models"):
        return artifact_root / Path(*relative.parts[1:])
    return REPO_ROOT / relative


def _matches(raw: str, *, artifact_root: Path | None = None) -> list[Path]:
    relative = _relative_path(raw)
    physical = _physical_path(relative, artifact_root)
    return sorted(
        Path(value)
        for value in glob.glob(str(physical))
        if Path(value).is_file() and not Path(value).is_symlink()
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_entries(
    raw_sources: list[Any], *, artifact_root: Path | None = None
) -> list[tuple[str, Path]]:
    """Expand source files, including every ONNX external-data sidecar."""

    entries: dict[str, Path] = {}
    for raw in sorted(str(value) for value in raw_sources):
        relative = _relative_path(raw)
        path = _physical_path(relative, artifact_root)
        entries[raw] = path
        if path.suffix.lower() != ".onnx" or not path.is_file():
            continue
        try:
            import onnx
        except (
            ImportError
        ) as exc:  # pragma: no cover - DS9 build/runtime image pins ONNX
            raise RuntimeError(
                f"ONNX is required to verify external source data: {raw}"
            ) from exc
        model = onnx.load(str(path), load_external_data=False)
        locations: set[Path] = set()
        for tensor in model.graph.initializer:
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            metadata = {item.key: item.value for item in tensor.external_data}
            location_raw = str(metadata.get("location", "")).strip()
            location = Path(location_raw)
            if not location_raw or location.is_absolute() or ".." in location.parts:
                raise ValueError(
                    f"unsafe ONNX external-data location in {raw}: {location_raw!r}"
                )
            locations.add(location)
        for location in sorted(locations, key=lambda value: value.as_posix()):
            label = f"{raw}::external::{location.as_posix()}"
            entries[label] = path.parent / location
    return sorted(entries.items(), key=lambda item: item[0])


def _hash_sources(raw_sources: list[Any], *, artifact_root: Path | None = None) -> str:
    """Hash declared source paths and bytes in a deterministic order."""

    digest = hashlib.sha256()
    for label, path in _source_entries(raw_sources, artifact_root=artifact_root):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _selected(
    artifact: Mapping[str, Any], profile: str, effective_profiles: set[str]
) -> bool:
    if profile.startswith("artifact:"):
        return str(artifact.get("id") or "") == profile.removeprefix("artifact:")
    profiles = {str(value) for value in artifact.get("required_profiles", []) or []}
    return profile == "full" or bool(profiles & effective_profiles)


def _resolve_artifact_root(raw: str | Path | None) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"artifact root must be absolute: {raw}")
    resolved = candidate.resolve(strict=False)
    ds9_root = (REPO_ROOT / "DS9").resolve()
    if resolved in {Path("/"), REPO_ROOT.resolve(), ds9_root}:
        raise ValueError(f"refusing unsafe artifact root: {resolved}")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"artifact root must not be inside the checkout: {resolved}")
    return resolved


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    check_files: bool = False,
    profile: str = "canonical",
    require_provenance: bool = False,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    artifact_root = _resolve_artifact_root(artifact_root)
    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    missing_root = sorted(REQUIRED_ROOT_KEYS - set(manifest))
    if missing_root:
        errors.append(f"manifest is missing root keys: {', '.join(missing_root)}")
    if manifest.get("schema_version") != 3:
        errors.append("schema_version must be 3")

    schema_raw = str(manifest.get("schema", ""))
    try:
        schema_path = REPO_ROOT / _relative_path(schema_raw)
    except ValueError as exc:
        errors.append(f"schema: {exc}")
        schema_path = Path()
    if not schema_path.is_file():
        errors.append(f"schema file is missing: {schema_raw}")
    else:
        try:
            schema = strict_json_loads(
                schema_path.read_bytes(),
                label="DS9 asset manifest schema",
            )
        except Exception as exc:
            errors.append(f"schema file is not valid JSON: {exc}")
        else:
            if not isinstance(schema, Mapping) or schema.get("$id") != (
                "https://noesis.local/schemas/ds9-native-asset-manifest-v3.json"
            ):
                errors.append("schema $id is not the native DS9 manifest v3 identifier")

    target = manifest.get("target")
    if not isinstance(target, Mapping):
        errors.append("target must be a mapping")
        target = {}
    deepstream = target.get("deepstream") if isinstance(target, Mapping) else None
    if (
        not isinstance(deepstream, Mapping)
        or deepstream.get("major") != 9
        or deepstream.get("version") != "9.1"
    ):
        errors.append("target.deepstream must declare major 9 and version 9.1")
    if target.get("cuda") != "13.2":
        errors.append("target.cuda must be 13.2")
    if target.get("tensorrt") != "10.16.0.72":
        errors.append("target.tensorrt must be 10.16.0.72")
    if target.get("python") != "3.12":
        errors.append("target.python must be 3.12")
    native_host = target.get("native_host")
    if not isinstance(native_host, Mapping) or dict(native_host) != NATIVE_HOST_AUTHORITY:
        errors.append(
            "target.native_host does not match the reviewed native DS9.1 authority"
        )

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("runtime must be a mapping")
        runtime = {}
    for key, expected in NATIVE_RUNTIME_AUTHORITY.items():
        if runtime.get(key) != expected:
            errors.append(f"runtime.{key} must be {expected!r}")
    runtime_paths = {
        "entrypoint": ("DS9/",),
        "implementation": ("DS9/",),
        "pipeline_config": ("DS9/",),
        "v3dt_pipeline_config": ("DS9/",),
        "v3dt_cameras_config": ("DS9/",),
        "v3dt_tracker_config": ("DS9/",),
        "preflight": ("DS9/",),
        "canonical_world_factory": ("noesis_core/",),
        "capability_health_api": ("noesis/server/health_api.py",),
        "scene_release_api": ("noesis/server/scene_api.py",),
    }
    for key, allowed_prefixes in runtime_paths.items():
        raw = runtime.get(key)
        try:
            relative = _relative_path(raw)
        except ValueError as exc:
            errors.append(f"runtime.{key}: {exc}")
            continue
        if not any(str(relative).startswith(prefix) for prefix in allowed_prefixes):
            errors.append(
                f"runtime.{key} is outside the declared DS9/shared boundary: {relative}"
            )
        if not (REPO_ROOT / relative).is_file():
            errors.append(f"runtime.{key} is missing: {relative}")

    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        errors.append("policy must be a mapping")
        policy = {}
    for key in ("ds8_binary_reuse", "root_engine_reuse", "root_native_extension_reuse"):
        if policy.get(key) != "forbidden":
            errors.append(f"policy.{key} must be forbidden")
    for key in ("require_ds9_owned_outputs", "require_provenance_for_validated"):
        if policy.get(key) is not True:
            errors.append(f"policy.{key} must be true")
    if policy.get("virtual_model_root") != "DS9/models":
        errors.append("policy.virtual_model_root must be DS9/models")
    if policy.get("physical_model_root_env") != "NOESIS_DS9_ARTIFACT_ROOT":
        errors.append("policy.physical_model_root_env must be NOESIS_DS9_ARTIFACT_ROOT")
    if policy.get("minimum_residual_artifact_bytes") != 10 * 1024 * 1024 * 1024:
        errors.append("policy.minimum_residual_artifact_bytes must preserve 10 GiB")
    allowed_states = {str(value) for value in policy.get("allowed_states", []) or []}
    output_roots: list[Path] = []
    source_roots: list[Path] = []
    for key, destination in (
        ("output_roots", output_roots),
        ("source_roots", source_roots),
    ):
        for raw in policy.get(key, []) or []:
            try:
                path = _relative_path(raw)
            except ValueError as exc:
                errors.append(f"policy.{key}: {exc}")
                continue
            if key == "output_roots" and not str(path).startswith("DS9/"):
                errors.append(f"policy.{key} escapes DS9 ownership: {path}")
            if key == "source_roots" and not (
                str(path).startswith("DS9/")
                or str(path).startswith(
                    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg"
                )
            ):
                errors.append(
                    f"policy.{key} is not an approved DS9 or third-party source root: {path}"
                )
            destination.append(path)

    inheritance_raw = policy.get("profile_inheritance", {})
    if not isinstance(inheritance_raw, Mapping):
        errors.append("policy.profile_inheritance must be a mapping")
        inheritance_raw = {}
    inheritance: dict[str, tuple[str, ...]] = {}
    for child_raw, parents_raw in inheritance_raw.items():
        child = str(child_raw).strip()
        if not child or not isinstance(parents_raw, list) or not parents_raw:
            errors.append(
                f"policy.profile_inheritance.{child or '<empty>'} must be a non-empty list"
            )
            continue
        parents = tuple(str(value).strip() for value in parents_raw)
        if any(not value for value in parents):
            errors.append(
                f"policy.profile_inheritance.{child} contains an empty profile"
            )
            continue
        inheritance[child] = parents

    effective_profiles: set[str] = set()
    visiting: set[str] = set()

    def add_profile(name: str) -> None:
        if name in visiting:
            errors.append(f"policy.profile_inheritance contains a cycle at {name!r}")
            return
        if name in effective_profiles:
            return
        visiting.add(name)
        for parent in inheritance.get(name, ()):
            add_profile(parent)
        visiting.remove(name)
        effective_profiles.add(name)

    add_profile(profile)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty list")
        artifacts = []
    ids: set[str] = set()
    state_counts: dict[str, int] = {}
    selected_count = 0
    for index, artifact in enumerate(artifacts):
        label = f"artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        artifact_id = str(artifact.get("id", "")).strip()
        if not artifact_id:
            errors.append(f"{label}.id is required")
        elif artifact_id in ids:
            errors.append(f"duplicate artifact id: {artifact_id}")
        ids.add(artifact_id)
        missing_keys = sorted(REQUIRED_ARTIFACT_KEYS - set(artifact))
        if missing_keys:
            errors.append(f"{label} is missing keys: {', '.join(missing_keys)}")
            continue
        if artifact.get("kind") not in ALLOWED_KINDS:
            errors.append(
                f"{artifact_id}: unsupported artifact kind {artifact.get('kind')!r}"
            )
        if not str(artifact.get("role", "")).strip():
            errors.append(f"{artifact_id}: role is required")

        output_raw = str(artifact.get("output", ""))
        try:
            output_owned = _path_below(output_raw, output_roots)
        except ValueError as exc:
            errors.append(f"{artifact_id}.output: {exc}")
            output_owned = False
        if not output_owned:
            errors.append(
                f"{artifact_id}: output is not below a declared DS9 output root: {output_raw}"
            )

        sources = artifact.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{artifact_id}: sources must be a non-empty list")
            sources = []
        for raw in sources:
            try:
                source_owned = _path_below(str(raw), source_roots)
            except ValueError as exc:
                errors.append(f"{artifact_id}.sources: {exc}")
                source_owned = False
            if not source_owned:
                errors.append(
                    f"{artifact_id}: source is not below a declared DS9 source root: {raw}"
                )

        builder_raw = str(artifact.get("builder", ""))
        try:
            builder_relative = _relative_path(builder_raw)
            builder = REPO_ROOT / builder_relative
        except ValueError as exc:
            errors.append(f"{artifact_id}.builder: {exc}")
        else:
            if not str(builder_relative).startswith("DS9/scripts/"):
                errors.append(
                    f"{artifact_id}: builder is not DS9-owned: {builder_relative}"
                )
            if not builder.is_file():
                errors.append(
                    f"{artifact_id}: builder is missing: {builder.relative_to(REPO_ROOT)}"
                )

        profiles = artifact.get("required_profiles")
        if not isinstance(profiles, list) or not profiles:
            errors.append(f"{artifact_id}: required_profiles must be a non-empty list")
        state = str(artifact.get("state", ""))
        state_counts[state] = state_counts.get(state, 0) + 1
        if state not in allowed_states:
            errors.append(f"{artifact_id}: state {state!r} is not allowed")
        selected_for_profile = _selected(artifact, profile, effective_profiles)
        if selected_for_profile:
            selected_count += 1

        compatibility = artifact.get("compatibility")
        if not isinstance(compatibility, Mapping):
            errors.append(f"{artifact_id}: compatibility must be a mapping")
        else:
            if compatibility.get("deepstream_major") != 9:
                errors.append(f"{artifact_id}: deepstream_major must be 9")
            if compatibility.get("cuda") != target.get("cuda"):
                errors.append(
                    f"{artifact_id}: CUDA compatibility does not match target"
                )
            if artifact.get("kind") in {
                "tensorrt_engine",
                "tensorrt_plugin",
                "nvinfer_parser",
            }:
                if compatibility.get("tensorrt") != target.get("tensorrt"):
                    errors.append(
                        f"{artifact_id}: TensorRT compatibility does not match target"
                    )

        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            errors.append(f"{artifact_id}: provenance must be a mapping")
            provenance = {}
        missing_provenance = sorted(REQUIRED_PROVENANCE_KEYS - set(provenance))
        if missing_provenance:
            errors.append(
                f"{artifact_id}: provenance is missing keys: {', '.join(missing_provenance)}"
            )
        for key in ("source_sha256", "output_sha256"):
            value = provenance.get(key)
            if value is not None and not SHA256_RE.fullmatch(str(value)):
                errors.append(
                    f"{artifact_id}: provenance.{key} must be a lowercase SHA-256 digest"
                )

        matches = (
            _matches(output_raw, artifact_root=artifact_root)
            if check_files and output_owned
            else []
        )
        physical_state_selected = selected_for_profile or state == "validated"
        if check_files and physical_state_selected and state == "missing" and matches:
            errors.append(
                f"{artifact_id}: state is missing but output is staged: {output_raw}"
            )
        if (
            check_files
            and physical_state_selected
            and state in {"staged_unverified", "validated"}
            and not matches
        ):
            errors.append(
                f"{artifact_id}: state is {state} but no output matches {output_raw}"
            )
        if check_files and physical_state_selected and len(matches) > 1:
            errors.append(
                f"{artifact_id}: multiple outputs match reviewed path {output_raw}"
            )
        if (
            check_files
            and physical_state_selected
            and state in {"staged_unverified", "validated"}
        ):
            try:
                source_entries = _source_entries(
                    list(sources), artifact_root=artifact_root
                )
            except Exception as exc:
                errors.append(
                    f"{artifact_id}: staged source bundle cannot be inspected: {exc}"
                )
                source_entries = []
            for label, path in source_entries:
                if not path.is_file():
                    errors.append(
                        f"{artifact_id}: staged artifact source is missing: {label}"
                    )

        provenance_required = require_provenance or state == "validated"
        if provenance_required and (selected_for_profile or state == "validated"):
            empty = [
                key
                for key in REQUIRED_PROVENANCE_KEYS
                if not str(provenance.get(key) or "").strip()
            ]
            if empty:
                errors.append(
                    f"{artifact_id}: required provenance is incomplete: {', '.join(sorted(empty))}"
                )
            elif len(matches) == 1:
                expected_output = str(provenance.get("output_sha256"))
                actual_output = _hash_file(matches[0])
                if actual_output != expected_output:
                    errors.append(
                        f"{artifact_id}: output SHA-256 mismatch for {matches[0]}"
                    )
                source_paths = [
                    _physical_path(_relative_path(raw), artifact_root)
                    for raw in sources
                ]
                if all(path.is_file() for path in source_paths):
                    expected_source = str(provenance.get("source_sha256"))
                    try:
                        actual_source = _hash_sources(
                            list(sources), artifact_root=artifact_root
                        )
                    except Exception as exc:
                        errors.append(
                            f"{artifact_id}: source provenance cannot be verified: {exc}"
                        )
                    else:
                        if actual_source != expected_source:
                            errors.append(
                                f"{artifact_id}: declared source SHA-256 mismatch"
                            )
            if artifact.get("kind") == "tensorrt_engine":
                maintenance = provenance.get("maintenance")
                if not isinstance(maintenance, Mapping):
                    errors.append(
                        f"{artifact_id}: TensorRT provenance requires a maintenance mapping"
                    )
                else:
                    missing_maintenance = sorted(
                        REQUIRED_ENGINE_MAINTENANCE_KEYS - set(maintenance)
                    )
                    if missing_maintenance:
                        errors.append(
                            f"{artifact_id}: maintenance provenance is missing keys: "
                            + ", ".join(missing_maintenance)
                        )
                    empty_maintenance = sorted(
                        key
                        for key in REQUIRED_ENGINE_MAINTENANCE_KEYS
                        if key in maintenance
                        and key not in {"gpu", "tensor_contract"}
                        and not str(maintenance.get(key) or "").strip()
                    )
                    if empty_maintenance:
                        errors.append(
                            f"{artifact_id}: maintenance provenance is incomplete: "
                            + ", ".join(empty_maintenance)
                        )
                    if isinstance(compatibility, Mapping):
                        if maintenance.get("precision") != compatibility.get(
                            "precision"
                        ):
                            errors.append(
                                f"{artifact_id}: maintenance precision does not match compatibility"
                            )
                        if maintenance.get("batch") != compatibility.get("batch"):
                            errors.append(
                                f"{artifact_id}: maintenance batch does not match compatibility"
                            )
                    if not isinstance(
                        maintenance.get("tensor_contract"), Mapping
                    ) or not maintenance.get("tensor_contract"):
                        errors.append(
                            f"{artifact_id}: maintenance tensor_contract must be a non-empty mapping"
                        )
                    gpu = maintenance.get("gpu")
                    if not isinstance(gpu, Mapping) or any(
                        not str(gpu.get(key) or "").strip()
                        for key in ("name", "uuid", "compute_capability", "memory_mib")
                    ):
                        errors.append(
                            f"{artifact_id}: maintenance GPU provenance is incomplete"
                        )
                    trt_version = str(maintenance.get("tensorrt_version") or "")
                    cuda_version = str(maintenance.get("cuda_version") or "")
                    if not trt_version.startswith(str(target.get("tensorrt") or "")):
                        errors.append(
                            f"{artifact_id}: maintenance TensorRT version does not match target"
                        )
                    if not cuda_version.startswith(str(target.get("cuda") or "")):
                        errors.append(
                            f"{artifact_id}: maintenance CUDA version does not match target"
                        )
                    manifest_raw = str(maintenance.get("manifest") or "")
                    try:
                        manifest_relative = _relative_path(manifest_raw)
                    except ValueError as exc:
                        errors.append(
                            f"{artifact_id}: maintenance manifest path is invalid: {exc}"
                        )
                    else:
                        if not str(manifest_relative).startswith(
                            "DS9/models/engine_maintenance/"
                        ):
                            errors.append(
                                f"{artifact_id}: maintenance manifest must be DS9 external-model owned"
                            )
                        manifest_path = _physical_path(manifest_relative, artifact_root)
                        if check_files and not manifest_path.is_file():
                            blockers.append(
                                f"{artifact_id}: maintenance manifest missing ({manifest_raw})"
                            )
                        elif check_files and manifest_path.is_file():
                            try:
                                maintenance_payload = _load_maintenance_manifest(
                                    manifest_path,
                                    artifact_id,
                                )
                            except Exception as exc:
                                errors.append(
                                    f"{artifact_id}: maintenance manifest is invalid: {exc}"
                                )
                            else:
                                installed_record = (
                                    maintenance_payload.get("installed") or {}
                                )
                                if (
                                    maintenance_payload.get("contract")
                                    != "noesis.ds9.engine_maintenance"
                                    or maintenance_payload.get("status") != "complete"
                                    or not isinstance(installed_record, Mapping)
                                    or installed_record.get("sha256")
                                    != provenance.get("output_sha256")
                                    or installed_record.get("size_bytes")
                                    != maintenance.get("output_size_bytes")
                                ):
                                    errors.append(
                                        f"{artifact_id}: maintenance manifest does not prove the declared output"
                                    )

        if check_files and selected_for_profile:
            if not matches:
                blockers.append(f"{artifact_id}: output missing ({output_raw})")
            for raw in sources:
                source_path = _physical_path(_relative_path(raw), artifact_root)
                if not source_path.is_file():
                    blockers.append(f"{artifact_id}: source missing ({raw})")
                    continue
                try:
                    expanded = _source_entries([raw], artifact_root=artifact_root)
                except Exception as exc:
                    blockers.append(
                        f"{artifact_id}: source bundle cannot be inspected ({raw}: {exc})"
                    )
                    continue
                for label, path in expanded:
                    if not path.is_file():
                        blockers.append(
                            f"{artifact_id}: source dependency missing ({label})"
                        )

    if blockers:
        errors.extend(f"artifact blocker: {message}" for message in blockers)
    if profile != "full" and selected_count == 0:
        errors.append(f"artifact profile selects no outputs: {profile}")

    shared_app_data = manifest.get("shared_app_data")
    if not isinstance(shared_app_data, list):
        errors.append("shared_app_data must be a list")
    else:
        for raw in shared_app_data:
            try:
                relative = _relative_path(raw)
            except ValueError as exc:
                errors.append(f"shared_app_data: {exc}")
                continue
            if not (REPO_ROOT / relative).exists():
                errors.append(f"shared_app_data path is missing: {relative}")
    missing_count = state_counts.get("missing", 0)
    if missing_count and not check_files:
        warnings.append(
            f"{missing_count} artifact(s) are explicitly marked missing; use --check-files --profile canonical or full to gate staging"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "blockers": blockers,
        "artifact_count": len(artifacts),
        "selected_count": selected_count,
        "profile": profile,
        "effective_profiles": sorted(effective_profiles),
        "state_counts": state_counts,
    }


def _file_sha256(path: Path) -> str:
    return _hash_file(path)


ENGINE_NAME_BY_ARTIFACT_ID = {
    "engine.yolo11_seg_alternate": "yolo11_seg",
    "engine.yolo26_detect_m": "yolo26_m",
    "engine.yolo26_seg_s": "yolo26_seg_s",
    "engine.reid_swin_tiny": "reid_swin",
    "engine.pose_yolo26": "yolo26_pose_n",
    "engine.depth_tracking_dav2": "depth_anything_v2_tracking",
    "engine.mapanything": "mapanything",
    "engine.wholebody49_s_masks": "wholebody49_s_masks",
    "engine.wholebody49_x_boxes": "wholebody49_x_boxes",
    "engine.v3dt_bodypose": "bodypose3dnet",
    "engine.v3dt_tracker_reid": "v3dt_tracker_reid",
}
_SOURCE_CONTRACT_DOCUMENT_KEYS = {"schema_version", "contracts"}


def _read_regular_owned_bytes(path: Path, label: str) -> bytes:
    """Read one owned, stable, non-symlink authority file."""

    candidate = _lexical_absolute(path)
    try:
        lexical = candidate.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
        ):
            raise ValueError(f"{label} must be an owned, single-link regular file")
        if opened.st_size <= 0 or opened.st_size > MAX_PRIVATE_JSON_BYTES:
            raise ValueError(f"{label} size is outside the accepted bound")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) != opened.st_size or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise ValueError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    return raw


def _load_regular_json_with_bytes(
    path: Path, label: str
) -> tuple[Mapping[str, Any], bytes]:
    """Read one owned, stable, non-symlink JSON authority and exact bytes."""

    raw = _read_regular_owned_bytes(path, label)
    try:
        payload = strict_json_loads(raw, label=label)
    except StrictJSONError as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} document root must be a mapping")
    return payload, raw


def _require_sha256(value: object, label: str) -> str:
    result = str(value or "").strip()
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _validate_source_contract_document(
    payload: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    if set(payload) != _SOURCE_CONTRACT_DOCUMENT_KEYS:
        raise ValueError(f"{label} top-level schema keys drifted")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label} schema_version must be 1")
    contracts = payload.get("contracts")
    expected_names = set(ENGINE_NAME_BY_ARTIFACT_ID.values())
    if not isinstance(contracts, Mapping) or set(contracts) != expected_names:
        raise ValueError(
            f"{label} contract membership differs from the reviewed engine registry"
        )
    for key, value in contracts.items():
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{label} contract {key!r} must be a nonempty mapping")
    return contracts


def _validate_wholebody_snapshot_copy(
    record: object,
    *,
    label: str,
    source_input: Mapping[str, Any],
    expected_sha256: str,
    expected_destination: Path,
) -> None:
    if not isinstance(record, Mapping) or set(record) != {
        "method",
        "source",
        "destination",
        "durability",
    }:
        raise ValueError(f"{label} snapshot-copy evidence is malformed")
    if (
        record.get("method") != "exclusive_nofollow_stream_copy"
        or record.get("durability") != "file_and_destination_directory_fsynced"
    ):
        raise ValueError(f"{label} snapshot-copy method/durability drifted")
    source = record.get("source")
    destination = record.get("destination")
    exact_stat_keys = {
        "path",
        "device",
        "inode",
        "mode",
        "uid",
        "gid",
        "nlink",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "sha256",
    }
    if (
        not isinstance(source, Mapping)
        or not isinstance(destination, Mapping)
        or set(source) != exact_stat_keys
        or set(destination) != exact_stat_keys
    ):
        raise ValueError(f"{label} snapshot stat evidence is malformed")
    source_size = source_input.get("size_bytes")
    if (
        source.get("path") != source_input.get("path")
        or source.get("sha256") != expected_sha256
        or source.get("sha256") != source_input.get("sha256")
        or source.get("size_bytes") != source_size
        or source.get("uid") != os.getuid()
        or source.get("nlink") != 1
    ):
        raise ValueError(f"{label} snapshot source differs from reviewed input")
    if (
        destination.get("path") != str(expected_destination)
        or destination.get("sha256") != expected_sha256
        or destination.get("size_bytes") != source_size
        or destination.get("mode") != "0600"
        or destination.get("uid") != os.getuid()
        or destination.get("nlink") != 1
    ):
        raise ValueError(f"{label} private snapshot differs from reviewed bytes")


def _wholebody_private_log(
    *,
    command: Mapping[str, Any],
    label: str,
    maintenance_manifest_path: Path,
) -> bytes:
    run_directory = maintenance_manifest_path.parent
    host_path = run_directory / "logs" / f"{label}.log"
    virtual_path = Path(str(command.get("log") or ""))
    expected_virtual = (
        Path("/workspace/DS9/models/engine_maintenance")
        / run_directory.name
        / "logs"
        / f"{label}.log"
    )
    if virtual_path != expected_virtual:
        raise ValueError(f"Wholebody49 {label} log path is not canonical")
    return _read_private_log_bytes(host_path, f"Wholebody49 {label} log")


def _validate_wholebody_builder_maintenance(
    *,
    expected_engine: str,
    source_contract: Mapping[str, Any],
    expected_build: Mapping[str, Any],
    inputs: Mapping[str, Any],
    maintenance_payload: Mapping[str, Any],
    by_label: Mapping[str, Mapping[str, Any]],
    maintenance_manifest_path: Path | None,
    output_size_bytes: int,
) -> None:
    variants = {
        "wholebody49_s_masks": "s_masks",
        "wholebody49_x_boxes": "x_boxes",
    }
    expected_variant = variants.get(expected_engine)
    if expected_variant is None:
        return
    if maintenance_manifest_path is None:
        raise ValueError("Wholebody49 maintenance proof lacks its manifest path")

    builder_sha256 = _require_sha256(
        expected_build.get("builder_source_sha256"),
        "Wholebody49 reviewed builder-source digest",
    )
    memory_pool_limits = expected_build.get("memory_pool_limits_bytes")
    if not isinstance(memory_pool_limits, Mapping) or set(memory_pool_limits) != {
        "workspace",
        "tactic_dram",
    }:
        raise ValueError(
            "Wholebody49 memory-pool limits must name workspace and tactic_dram"
        )
    workspace = memory_pool_limits.get("workspace")
    if (
        isinstance(workspace, bool)
        or not isinstance(workspace, int)
        or workspace <= 0
        or workspace % (1024 * 1024) != 0
    ):
        raise ValueError(
            "Wholebody49 workspace memory-pool limit must be positive and MiB-aligned"
        )
    tactic_dram = memory_pool_limits.get("tactic_dram")
    if (
        isinstance(tactic_dram, bool)
        or not isinstance(tactic_dram, int)
        or tactic_dram <= 0
        or tactic_dram & (tactic_dram - 1)
    ):
        raise ValueError(
            "Wholebody49 tactic_dram memory-pool limit must be a positive power of two"
        )
    builder_optimization_level = expected_build.get("builder_optimization_level")
    if (
        isinstance(builder_optimization_level, bool)
        or not isinstance(builder_optimization_level, int)
        or not 0 <= builder_optimization_level <= 5
    ):
        raise ValueError(
            "Wholebody49 builder optimization level must be an integer from 0 to 5"
        )
    if (
        expected_build.get("builder") != WHOLEBODY_BUILDER_CONTRACT
        or expected_build.get("precision") != "fp16"
        or expected_build.get("network_mode") != "explicit_batch_trt10_default"
        or builder_optimization_level
        != WHOLEBODY_BUILDER_OPTIMIZATION_LEVEL_BY_VARIANT[expected_variant]
        or expected_build.get("logger_policy") != WHOLEBODY_LOGGER_POLICY
        or expected_build.get("variant") != expected_variant
        or expected_build.get("plugin_args") != []
        or expected_build.get("profile")
        != {
            "input": "images",
            "min": [3, 3, 640, 640],
            "opt": [3, 3, 640, 640],
            "max": [3, 3, 640, 640],
        }
        or dict(memory_pool_limits)
        != {
            "workspace": WHOLEBODY_WORKSPACE_BYTES_BY_VARIANT[expected_variant],
            "tactic_dram": 2147483648,
        }
    ):
        raise ValueError("Wholebody49 dedicated-builder authority drifted")
    if (
        WHOLEBODY_BUILDER_SOURCE.is_symlink()
        or not WHOLEBODY_BUILDER_SOURCE.is_file()
        or _file_sha256(WHOLEBODY_BUILDER_SOURCE) != builder_sha256
    ):
        raise ValueError("tracked Wholebody49 builder source differs from authority")

    builder_input = inputs.get("wholebody_builder_source")
    staged_onnx_input = inputs.get("staged_onnx")
    if not isinstance(builder_input, Mapping) or not isinstance(
        staged_onnx_input, Mapping
    ):
        raise ValueError("Wholebody49 maintenance inputs lack builder/ONNX authority")
    if (
        builder_input.get("sha256") != builder_sha256
        or builder_input.get("size_bytes") != WHOLEBODY_BUILDER_SOURCE.stat().st_size
        or not str(builder_input.get("path") or "").endswith(
            "/DS9/csrc/wholebody49_engine_builder/wholebody49_engine_builder.cpp"
        )
    ):
        raise ValueError("Wholebody49 builder-source input differs from authority")
    onnx_sha256 = _require_sha256(
        source_contract.get("raw_sha256"), "Wholebody49 reviewed ONNX digest"
    )
    if staged_onnx_input.get("sha256") != onnx_sha256:
        raise ValueError("Wholebody49 staged ONNX input differs from authority")

    run_id = str(maintenance_payload.get("run_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("Wholebody49 maintenance run ID is invalid")
    snapshot_root = Path(f"/tmp/noesis-wholebody49-build-{run_id}")
    evidence = maintenance_payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Wholebody49 maintenance evidence is missing")
    snapshots = evidence.get("wholebody_input_snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != {
        "directory",
        "builder_source",
        "onnx",
    }:
        raise ValueError("Wholebody49 private snapshot evidence is malformed")
    if snapshots.get("directory") != str(snapshot_root):
        raise ValueError("Wholebody49 private snapshot directory drifted")
    builder_snapshot = snapshot_root / "wholebody49_engine_builder.cpp"
    staged_name = Path(str(staged_onnx_input.get("path") or "")).name
    if not staged_name:
        raise ValueError("Wholebody49 staged ONNX input path is invalid")
    onnx_snapshot = snapshot_root / staged_name
    _validate_wholebody_snapshot_copy(
        snapshots.get("builder_source"),
        label="Wholebody49 builder source",
        source_input=builder_input,
        expected_sha256=builder_sha256,
        expected_destination=builder_snapshot,
    )
    _validate_wholebody_snapshot_copy(
        snapshots.get("onnx"),
        label="Wholebody49 ONNX",
        source_input=staged_onnx_input,
        expected_sha256=onnx_sha256,
        expected_destination=onnx_snapshot,
    )

    executable = evidence.get("wholebody_builder_executable")
    if not isinstance(executable, Mapping) or set(executable) != {
        "path",
        "size_bytes",
        "sha256",
        "mode",
        "compile_command",
    }:
        raise ValueError("Wholebody49 builder-executable evidence is malformed")
    executable_path = Path(f"/tmp/noesis-wholebody49-engine-builder-{run_id}")
    executable_sha256 = _require_sha256(
        executable.get("sha256"), "Wholebody49 builder-executable digest"
    )
    if (
        executable.get("path") != str(executable_path)
        or not isinstance(executable.get("size_bytes"), int)
        or int(executable["size_bytes"]) <= 0
        or executable.get("mode") != "0700"
        or not executable_sha256
    ):
        raise ValueError("Wholebody49 builder-executable evidence drifted")

    compiler = "/usr/bin/g++"
    compile_command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fstack-protector-strong",
        "-D_FORTIFY_SOURCE=3",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-Wno-deprecated-declarations",
        "-I/usr/local/cuda/include",
        str(builder_snapshot),
        "-o",
        str(executable_path),
        "-lnvonnxparser",
        "-lnvinfer",
        "-pthread",
    ]
    compile_row = by_label.get("compile-wholebody-builder")
    toolchain_row = by_label.get("probe-wholebody-builder-toolchain")
    if (
        not isinstance(compile_row, Mapping)
        or compile_row.get("command") != compile_command
        or executable.get("compile_command") != compile_command
        or not isinstance(toolchain_row, Mapping)
        or toolchain_row.get("command") != [compiler, "--version"]
    ):
        raise ValueError("Wholebody49 compile/toolchain command drifted")
    if (
        _wholebody_private_log(
            command=compile_row,
            label="compile-wholebody-builder",
            maintenance_manifest_path=maintenance_manifest_path,
        )
        != b""
    ):
        raise ValueError("Wholebody49 -Werror compilation emitted output")

    candidate = maintenance_payload.get("candidate")
    if not isinstance(candidate, Mapping) or (
        candidate.get("size_bytes") != output_size_bytes
        or candidate.get("sha256")
        != (maintenance_payload.get("installed") or {}).get("sha256")
    ):
        raise ValueError("Wholebody49 candidate evidence differs from installed engine")
    build_row = by_label.get("build")
    expected_build_command = [
        str(executable_path),
        "--onnx",
        str(onnx_snapshot),
        "--output",
        str(candidate.get("path") or ""),
        "--variant",
        expected_variant,
    ]
    if (
        not isinstance(build_row, Mapping)
        or build_row.get("command") != expected_build_command
        or build_row.get("proof") != f"wholebody_builder_{expected_variant}"
    ):
        raise ValueError("Wholebody49 variant-bound build command/proof drifted")
    build_log = _wholebody_private_log(
        command=build_row,
        label="build",
        maintenance_manifest_path=maintenance_manifest_path,
    ).decode("utf-8", errors="strict")
    required_lines = (
        "[NOESIS_TRT_BUILDER] contract=noesis.ds9.wholebody49_builder.v1",
        "[NOESIS_TRT_BUILDER] network_mode=explicit_batch_trt10_default",
        f"[NOESIS_TRT_BUILDER] variant={expected_variant}",
        "[NOESIS_TRT_BUILDER] profile=images:3x3x640x640",
        "[NOESIS_TRT_BUILDER] tactic_dram_bytes=2147483648",
        "[NOESIS_TRT_BUILDER] logger_minimum_severity=info",
        "[NOESIS_TRT_BUILDER] logger_verbose_policy=ignored_before_copy",
        "[NOESIS_TRT_BUILDER] logger_captured_truncation=fatal",
        "[NOESIS_TRT_BUILDER] logger_error_state=sticky_fatal",
        "[NOESIS_TRT_BUILDER] status=PASS",
    )
    build_lines = build_log.splitlines()
    workspace_line = (
        "[NOESIS_TRT_BUILDER] workspace_bytes="
        f"{WHOLEBODY_WORKSPACE_BYTES_BY_VARIANT[expected_variant]}"
    )
    workspace_lines = [
        line
        for line in build_lines
        if line.startswith("[NOESIS_TRT_BUILDER] workspace_bytes=")
    ]
    optimization_line = (
        "[NOESIS_TRT_BUILDER] builder_optimization_level="
        f"{WHOLEBODY_BUILDER_OPTIMIZATION_LEVEL_BY_VARIANT[expected_variant]}"
    )
    optimization_lines = [
        line
        for line in build_lines
        if line.startswith("[NOESIS_TRT_BUILDER] builder_optimization_level=")
    ]
    if (
        any(build_lines.count(line) != 1 for line in required_lines)
        or workspace_lines != [workspace_line]
        or optimization_lines != [optimization_line]
        or any(
            line.startswith("[NOESIS_TRT_BUILDER] status=FAIL") for line in build_lines
        )
    ):
        raise ValueError("Wholebody49 build transcript lacks exact positive markers")
    engine_sizes = re.findall(
        r"^\[NOESIS_TRT_BUILDER\] engine_bytes=([1-9][0-9]*)$",
        build_log,
        flags=re.MULTILINE,
    )
    if len(engine_sizes) != 1 or int(engine_sizes[0]) != output_size_bytes:
        raise ValueError("Wholebody49 build transcript engine size drifted")


_GPU_MEMORY_GUARD_RECORD_KEYS = {
    "contract",
    "path",
    "sha256",
    "engine",
    "transaction_id",
    "prepared_transaction_sha256",
    "artifact_root_id",
    "container_id",
    "wrapper_pid",
    "wrapper_start_time_ticks",
    "device_index",
    "gpu_uuid",
    "guard_mib",
    "sample_interval_ms",
    "maximum_gap_limit_ms",
    "summary",
}


def _validate_gpu_memory_guard_proof(
    *,
    artifact_root: Path,
    expected_engine: str,
    maintenance_payload: Mapping[str, Any],
    guard_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct and validate one realized NVML guard from sealed evidence."""

    if set(guard_record) != _GPU_MEMORY_GUARD_RECORD_KEYS:
        raise ValueError("realized GPU-memory guard fields drifted")
    metadata = maintenance_payload.get("metadata")
    platform = metadata.get("platform") if isinstance(metadata, Mapping) else None
    host_transaction = (
        metadata.get("host_transaction") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(platform, Mapping) or not isinstance(host_transaction, Mapping):
        raise ValueError(
            "GPU-memory guard lacks maintenance platform/transaction binding"
        )
    transaction_id = str(host_transaction.get("transaction_id") or "")
    prepared_sha256 = _require_sha256(
        host_transaction.get("transaction_sha256"),
        "GPU-memory guard prepared transaction digest",
    )
    expected_uuid = str(platform.get("gpu_uuid") or "")
    if not expected_uuid.startswith("GPU-"):
        raise ValueError("GPU-memory guard lacks a reviewed GPU UUID")
    root = artifact_root.resolve(strict=True)
    artifact_root_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    expected_guard_mib = legacy_gpu_guard.reviewed_guard_mib(expected_engine)
    container_id = str(guard_record.get("container_id") or "")
    wrapper_pid = guard_record.get("wrapper_pid")
    wrapper_start = guard_record.get("wrapper_start_time_ticks")
    if (
        not transaction_id
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", transaction_id)
        or SHA256_RE.fullmatch(container_id) is None
        or isinstance(wrapper_pid, bool)
        or not isinstance(wrapper_pid, int)
        or wrapper_pid <= 1
        or isinstance(wrapper_start, bool)
        or not isinstance(wrapper_start, int)
        or wrapper_start <= 0
    ):
        raise ValueError("realized GPU-memory guard identity is invalid")
    exact_fields = {
        "contract": legacy_gpu_guard.CONTRACT,
        "engine": expected_engine,
        "transaction_id": transaction_id,
        "prepared_transaction_sha256": prepared_sha256,
        "artifact_root_id": artifact_root_id,
        "device_index": 0,
        "gpu_uuid": expected_uuid,
        "guard_mib": expected_guard_mib,
        "sample_interval_ms": legacy_gpu_guard.REVIEWED_SAMPLE_INTERVAL_MS,
        "maximum_gap_limit_ms": legacy_gpu_guard.REVIEWED_MAX_GAP_MS,
    }
    for key, expected in exact_fields.items():
        if guard_record.get(key) != expected:
            raise ValueError(f"realized GPU-memory guard differs from authority: {key}")
    relative = _relative_path(guard_record.get("path"))
    expected_relative = (
        Path("models/engine_finalize")
        / f"{transaction_id}-{expected_engine}"
        / "gpu-memory.jsonl"
    )
    if relative != expected_relative:
        raise ValueError("GPU-memory guard evidence path is not canonical")
    cohort = root / expected_relative.parent
    evidence_path = _bounded_private_path(
        root / relative,
        cohort,
        "NVML GPU-memory guard evidence",
    )
    expected_evidence_sha256 = _require_sha256(
        guard_record.get("sha256"), "NVML GPU-memory guard evidence digest"
    )
    summary_args = argparse.Namespace(
        evidence=evidence_path,
        device_index=0,
        expected_uuid=expected_uuid,
        engine=expected_engine,
        transaction_id=transaction_id,
        prepared_transaction_sha256=prepared_sha256,
        artifact_root_id=artifact_root_id,
        container_id=container_id,
        guard_mib=expected_guard_mib,
        interval_ms=legacy_gpu_guard.REVIEWED_SAMPLE_INTERVAL_MS,
        max_gap_ms=legacy_gpu_guard.REVIEWED_MAX_GAP_MS,
        parent_pid=wrapper_pid,
        parent_start_time_ticks=wrapper_start,
        allow_active=False,
    )
    try:
        reconstructed = legacy_gpu_guard.summarize(summary_args)
    except Exception as exc:
        raise ValueError(f"NVML GPU-memory guard proof is invalid: {exc}") from exc
    declared_summary = guard_record.get("summary")
    if (
        not isinstance(declared_summary, Mapping)
        or dict(declared_summary) != reconstructed
    ):
        raise ValueError(
            "realized GPU-memory guard summary differs from sealed evidence"
        )
    if (
        reconstructed.get("state") != "stopped"
        or reconstructed.get("guard_ok") is not True
        or reconstructed.get("evidence_sha256") != expected_evidence_sha256
    ):
        raise ValueError("NVML GPU-memory guard proof is not a clean terminal run")
    return copy.deepcopy(dict(guard_record))


def _validate_engine_maintenance_proof(
    *,
    artifact_id: str,
    artifact: Mapping[str, Any],
    provenance: Mapping[str, Any],
    maintenance_payload: Mapping[str, Any],
    target: Mapping[str, Any],
    source_contracts_path: Path = SOURCE_CONTRACTS,
    source_contract_document: Mapping[str, Any] | None = None,
    current_source_contracts_sha256: str | None = None,
    artifact_root: Path,
    maintenance_manifest_path: Path,
    maintenance_manifest_sha256: str,
    output_sha256: str,
    output_size_bytes: int,
) -> None:
    """Validate a private maintenance manifest independently of its producer."""

    expected_engine = ENGINE_NAME_BY_ARTIFACT_ID.get(artifact_id)
    if expected_engine is None:
        raise ValueError(f"no reviewed maintenance engine mapping for {artifact_id}")
    bound_maintenance_sha256 = _require_sha256(
        maintenance_manifest_sha256,
        "caller-known maintenance manifest digest",
    )
    bound_payload, bound_raw = _load_private_json_with_bytes(
        maintenance_manifest_path, "caller-known maintenance manifest"
    )
    if _sha256_bytes(bound_raw) != bound_maintenance_sha256:
        raise ValueError("caller-known maintenance manifest digest mismatch")
    if dict(bound_payload) != dict(maintenance_payload):
        raise ValueError("maintenance proof payload differs from caller-known manifest")
    if (
        maintenance_payload.get("schema_version") != 1
        or maintenance_payload.get("contract") != "noesis.ds9.engine_maintenance"
        or maintenance_payload.get("status") != "complete"
        or maintenance_payload.get("engine") != expected_engine
    ):
        raise ValueError(
            f"maintenance evidence is not a complete {expected_engine} run"
        )
    if not str(maintenance_payload.get("inputs_revalidated_at_utc") or "").strip():
        raise ValueError(
            "maintenance evidence lacks the pre-install input revalidation"
        )

    installed = maintenance_payload.get("installed")
    if not isinstance(installed, Mapping) or (
        installed.get("sha256") != output_sha256
        or installed.get("size_bytes") != output_size_bytes
    ):
        raise ValueError("maintenance installed record differs from the engine output")

    inputs = maintenance_payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("maintenance inputs must be a mapping")
    source_contract_input = inputs.get("source_contracts")
    if not isinstance(source_contract_input, Mapping):
        raise ValueError("maintenance source-contract input must be a mapping")
    maintenance_source_contracts_sha256 = _require_sha256(
        source_contract_input.get("sha256"),
        "maintenance source-contract input digest",
    )
    authoritative_source_contracts_sha256 = _require_sha256(
        current_source_contracts_sha256 or _file_sha256(source_contracts_path),
        "current source-contract authority digest",
    )
    if maintenance_source_contracts_sha256 != authoritative_source_contracts_sha256:
        raise ValueError(
            "maintenance source-contract input digest differs from current authority"
        )

    if source_contract_document is None:
        loaded_source_contract_document, loaded_source_contract_raw = (
            _load_regular_json_with_bytes(
                source_contracts_path, "engine source contracts"
            )
        )
        if (
            _sha256_bytes(loaded_source_contract_raw)
            != authoritative_source_contracts_sha256
        ):
            raise ValueError("engine source contracts changed during validation")
        source_contract_document = loaded_source_contract_document
    contracts = source_contract_document.get("contracts")
    source_contract = (
        contracts.get(expected_engine) if isinstance(contracts, Mapping) else None
    )
    if not isinstance(source_contract, Mapping):
        raise ValueError(f"missing reviewed source contract for {expected_engine}")
    expected_build = source_contract.get("maintenance_build")
    if not isinstance(expected_build, Mapping):
        raise ValueError(
            f"missing reviewed maintenance build contract for {expected_engine}"
        )
    expected_build_contract = copy.deepcopy(dict(expected_build))
    expected_tensor = source_contract.get("onnx") or source_contract.get(
        "tensor_contract"
    )
    if not isinstance(expected_tensor, Mapping):
        raise ValueError(f"missing reviewed tensor contract for {expected_engine}")
    tensor_key = (
        "onnx" if source_contract.get("onnx") is not None else "tensor_contract"
    )
    expected_build_contract[tensor_key] = copy.deepcopy(dict(expected_tensor))

    metadata = maintenance_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("maintenance metadata must be a mapping")
    build_contract = metadata.get("build_contract")
    if (
        not isinstance(build_contract, Mapping)
        or dict(build_contract) != expected_build_contract
    ):
        raise ValueError(
            "maintenance build/tensor contract differs from reviewed authority"
        )

    platform = metadata.get("platform")
    if not isinstance(platform, Mapping):
        raise ValueError("maintenance platform must be a mapping")
    native_host = target.get("native_host")
    if not isinstance(native_host, Mapping) or dict(native_host) != NATIVE_HOST_AUTHORITY:
        raise ValueError("target native-host authority is missing")
    native_receipt = platform.get("image") == "native_host"
    exact_platform = (
        NATIVE_ENGINE_RECEIPT_PLATFORM
        if native_receipt
        else LEGACY_ENGINE_RECEIPT_PLATFORM
    )
    for key, expected in exact_platform.items():
        if platform.get(key) != expected:
            raise ValueError(
                f"maintenance platform {key} differs from reviewed receipt authority"
            )
    for key in (
        "driver_version",
        "gpu_name",
        "gpu_uuid",
        "gpu_compute_capability",
        "gpu_memory_mib",
    ):
        if not str(platform.get(key) or "").strip():
            raise ValueError(f"maintenance platform is missing {key}")

    commands = maintenance_payload.get("commands")
    if not isinstance(commands, list):
        raise ValueError("maintenance commands must be a list")
    by_label: dict[str, Mapping[str, Any]] = {}
    for row in commands:
        if not isinstance(row, Mapping):
            raise ValueError("maintenance command entries must be mappings")
        label = str(row.get("label") or "")
        if not label or label in by_label:
            raise ValueError("maintenance command labels must be non-empty and unique")
        by_label[label] = row
    build_label = (
        "build-tracker-engine" if expected_engine == "v3dt_tracker_reid" else "build"
    )
    wholebody_variants = {
        "wholebody49_s_masks": "s_masks",
        "wholebody49_x_boxes": "x_boxes",
    }
    wholebody_variant = wholebody_variants.get(expected_engine)
    required_commands = {
        "probe-trtexec": "trtexec_probe",
        build_label: (
            "generic"
            if build_label == "build-tracker-engine"
            else (
                f"wholebody_builder_{wholebody_variant}"
                if wholebody_variant is not None
                else "trtexec_build"
            )
        ),
        "load-candidate": "trtexec_load",
        "load-installed": "trtexec_load",
    }
    if expected_engine == "v3dt_tracker_reid":
        required_commands["compile-helper"] = "generic"
    if wholebody_variant is not None:
        required_commands.update(
            {
                "probe-wholebody-builder-toolchain": "generic",
                "compile-wholebody-builder": "generic",
            }
        )
    for label, proof in required_commands.items():
        row = by_label.get(label)
        if (
            not isinstance(row, Mapping)
            or row.get("status") != "passed"
            or row.get("returncode") != 0
            or row.get("timed_out") is not False
            or (native_receipt and row.get("output_exceeded") is not False)
            or row.get("proof") != proof
            or not isinstance(row.get("command"), list)
            or not row.get("command")
        ):
            raise ValueError(
                f"maintenance command did not pass its proof gate: {label}"
            )

    if native_receipt:
        if "host_transaction" in metadata:
            raise ValueError(
                "native-host maintenance must not carry a retired host transaction"
            )
        native_build = metadata.get("native_host")
        if (
            not isinstance(native_build, Mapping)
            or set(native_build) != NATIVE_HOST_BUILD_AUTHORITY_KEYS
        ):
            raise ValueError("native-host build authority fields drifted")
        source_input_label = (
            "staged_tracker_reid_source"
            if expected_engine == "v3dt_tracker_reid"
            else "staged_onnx"
        )
        source_input = inputs.get(source_input_label)
        if not isinstance(source_input, Mapping):
            raise ValueError(
                f"native-host maintenance lacks {source_input_label} input evidence"
            )
        expected_source_sha256 = _require_sha256(
            source_input.get("sha256"), "native-host source input digest"
        )
        build_command = by_label[build_label].get("command")
        assert isinstance(build_command, list)
        exact_native_build = {
            "backend": "native_host",
            "deepstream": native_host["sdk_root"],
            "cuda": native_host["cuda_root"],
            "tensorrt": "TensorRT v101600",
            "python_abi": "cp312",
            "source_sha256": expected_source_sha256,
            "output_sha256": output_sha256,
            "command": " ".join(str(value) for value in build_command),
        }
        for key, expected in exact_native_build.items():
            if native_build.get(key) != expected:
                raise ValueError(
                    f"native-host build authority differs from evidence: {key}"
                )
        compiler = str(native_build.get("compiler") or "")
        if not compiler or len(compiler.encode("utf-8")) > 4096:
            raise ValueError("native-host compiler identity is missing or unbounded")
        if len(str(native_build.get("command") or "").encode("utf-8")) > 65536:
            raise ValueError("native-host build command evidence is unbounded")

        candidate = maintenance_payload.get("candidate")
        install_transaction = maintenance_payload.get("install_transaction")
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("sha256") != output_sha256
            or candidate.get("size_bytes") != output_size_bytes
            or not isinstance(install_transaction, Mapping)
            or install_transaction.get("status") != "installed_verified"
            or install_transaction.get("target") != maintenance_payload.get("target")
            or install_transaction.get("candidate") != candidate
            or installed.get("path") != maintenance_payload.get("target")
        ):
            raise ValueError(
                "native-host maintenance lacks a verified atomic install transaction"
            )

    _validate_wholebody_builder_maintenance(
        expected_engine=expected_engine,
        source_contract=source_contract,
        expected_build=expected_build,
        inputs=inputs,
        maintenance_payload=maintenance_payload,
        by_label=by_label,
        maintenance_manifest_path=maintenance_manifest_path,
        output_size_bytes=output_size_bytes,
    )

    compatibility = artifact.get("compatibility")
    declared_maintenance = provenance.get("maintenance")
    if not isinstance(compatibility, Mapping) or not isinstance(
        declared_maintenance, Mapping
    ):
        raise ValueError("artifact compatibility/maintenance provenance is missing")
    guard_present = "gpu_memory_guard" in declared_maintenance
    guard = declared_maintenance.get("gpu_memory_guard")
    if native_receipt:
        if guard_present:
            raise ValueError(
                "native-host receipt must not carry the retired container GPU guard"
            )
        if declared_maintenance.get("manifest_sha256") != bound_maintenance_sha256:
            raise ValueError(
                "native-host realized maintenance digest differs from private evidence"
            )
    elif not guard_present:
        if artifact_id in UNCONDITIONALLY_GUARD_REQUIRED_ARTIFACT_IDS:
            raise ValueError(
                "Wholebody49 realized provenance requires a sealed GPU-memory guard"
            )
        legacy = LEGACY_GPU_MEMORY_GUARD_EXEMPTIONS.get(artifact_id)
        observed_legacy = {
            "output_sha256": output_sha256,
            "maintenance_manifest": declared_maintenance.get("manifest"),
            "maintenance_manifest_sha256": bound_maintenance_sha256,
        }
        if legacy != observed_legacy:
            raise ValueError(
                "realized engine lacks a sealed GPU-memory guard and is not an "
                "exact pre-guard trust anchor"
            )
        declared_manifest_sha256 = declared_maintenance.get("manifest_sha256")
        if (
            declared_manifest_sha256 is not None
            and declared_manifest_sha256 != bound_maintenance_sha256
        ):
            raise ValueError(
                "legacy realized maintenance digest differs from caller-known proof"
            )
    else:
        if not isinstance(guard, Mapping):
            raise ValueError("realized GPU-memory guard must be a mapping")
        if declared_maintenance.get("manifest_sha256") != bound_maintenance_sha256:
            raise ValueError(
                "realized maintenance digest differs from caller-known proof"
            )
        _validate_gpu_memory_guard_proof(
            artifact_root=artifact_root,
            expected_engine=expected_engine,
            maintenance_payload=maintenance_payload,
            guard_record=guard,
        )
    expected_declared_tensor = copy.deepcopy(dict(expected_tensor))
    exact_declared = {
        "output_size_bytes": output_size_bytes,
        "image": platform.get("image"),
        "image_id": platform.get("image_id"),
        "base_digest": platform.get("base_digest"),
        "tensorrt_version": platform.get("tensorrt_version"),
        "cuda_version": platform.get("cuda_version"),
        "driver_version": platform.get("driver_version"),
        "precision": compatibility.get("precision"),
        "batch": compatibility.get("batch"),
        "tensor_contract": expected_declared_tensor,
    }
    for key, expected in exact_declared.items():
        if declared_maintenance.get(key) != expected:
            raise ValueError(
                f"realized maintenance field differs from private evidence: {key}"
            )
    gpu = declared_maintenance.get("gpu")
    expected_gpu = {
        "name": str(platform.get("gpu_name")),
        "uuid": str(platform.get("gpu_uuid")),
        "compute_capability": str(platform.get("gpu_compute_capability")),
        "memory_mib": int(platform.get("gpu_memory_mib")),
    }
    if not isinstance(gpu, Mapping) or dict(gpu) != expected_gpu:
        raise ValueError("realized GPU provenance differs from private evidence")


def validate_asset_realization(
    base_manifest: Mapping[str, Any] | str | Path,
    realization_path: str | Path,
    artifact_root: str | Path,
    *,
    profile: str = "canonical",
    check_files: bool = True,
    require_provenance: bool = True,
) -> dict[str, Any]:
    try:
        if require_provenance and not check_files:
            raise ValueError(
                "authoritative realized provenance validation requires check_files"
            )
        if isinstance(base_manifest, Mapping):
            base_path = DEFAULT_MANIFEST
            base_raw = _read_regular_owned_bytes(
                base_path, "tracked base-manifest authority"
            )
            tracked = _parse_yaml_mapping(base_raw, "tracked base-manifest authority")
            if dict(base_manifest) != dict(tracked):
                raise ValueError(
                    "mapping base_manifest must exactly match tracked DS9/asset_manifest.yaml"
                )
            base_payload = copy.deepcopy(dict(tracked))
        else:
            base_path = _lexical_absolute(base_manifest)
            base_raw = _read_regular_owned_bytes(base_path, "base-manifest authority")
            base_payload = copy.deepcopy(
                dict(_parse_yaml_mapping(base_raw, "base-manifest authority"))
            )
        base_sha256 = _sha256_bytes(base_raw)
        source_contract_document, source_contract_raw = _load_regular_json_with_bytes(
            SOURCE_CONTRACTS, "engine source-contract authority"
        )
        current_source_contracts_sha256 = _sha256_bytes(source_contract_raw)
        _validate_source_contract_document(
            source_contract_document, "engine source-contract authority"
        )
        root = _resolve_artifact_root(artifact_root)
        if root is None:
            raise ValueError("artifact_root is required for realized validation")
        expected_realization = root / REALIZATION_FILENAME
        realized_path = _lexical_absolute(realization_path)
        if realized_path != expected_realization:
            raise ValueError(
                f"realization must be {expected_realization}, got {realized_path}"
            )
        realized_path = _bounded_private_path(realized_path, root, "asset realization")
        realization, realization_raw = _load_private_json_with_bytes(
            realized_path, "asset realization"
        )
        realization_sha256 = _sha256_bytes(realization_raw)
        allowed_top = {
            "schema_version",
            "contract",
            "base_manifest",
            "source_contracts",
            "created_at_utc",
            "updated_at_utc",
            "artifacts",
        }
        if set(realization) != allowed_top:
            raise ValueError(
                "asset realization has unexpected or missing top-level keys"
            )
        if (
            realization.get("schema_version") != 1
            or realization.get("contract") != REALIZATION_CONTRACT
        ):
            raise ValueError("asset realization contract mismatch")
        expected_base = {
            "path": "DS9/asset_manifest.yaml",
            "sha256": base_sha256,
        }
        expected_contracts = {
            "path": "DS9/config/engine_source_contracts.json",
            "sha256": current_source_contracts_sha256,
        }
        if realization.get("base_manifest") != expected_base:
            raise ValueError("asset realization base-manifest digest drift")
        if realization.get("source_contracts") != expected_contracts:
            raise ValueError("asset realization source-contract digest drift")
        if (
            not str(realization.get("created_at_utc") or "").strip()
            or not str(realization.get("updated_at_utc") or "").strip()
        ):
            raise ValueError("asset realization timestamps are required")

        artifacts = base_payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("base manifest artifacts must be a list")
        by_id = {str(row.get("id")): row for row in artifacts}
        engine_ids = {
            artifact_id
            for artifact_id, row in by_id.items()
            if row.get("kind") == "tensorrt_engine"
        }
        realized_artifacts = realization.get("artifacts")
        if not isinstance(realized_artifacts, dict):
            raise ValueError("asset realization artifacts must be a mapping")
        unknown = set(realized_artifacts) - engine_ids
        if unknown:
            raise ValueError(
                "asset realization contains unknown/non-engine artifact IDs: "
                + ", ".join(sorted(unknown))
            )
        for artifact_id, realized in realized_artifacts.items():
            if not isinstance(realized, Mapping) or set(realized) != {
                "state",
                "provenance",
            }:
                raise ValueError(
                    f"asset realization may override only state/provenance: {artifact_id}"
                )
            if realized.get("state") not in {"staged_unverified", "validated"}:
                raise ValueError(f"invalid realized state for {artifact_id}")
            if not isinstance(realized.get("provenance"), Mapping):
                raise ValueError(
                    f"realized provenance must be a mapping: {artifact_id}"
                )
            base_source_anchor = str(
                (by_id[artifact_id].get("provenance") or {}).get("source_sha256") or ""
            ).strip()
            realized_source_anchor = str(
                realized["provenance"].get("source_sha256") or ""
            ).strip()
            if not base_source_anchor or realized_source_anchor != base_source_anchor:
                raise ValueError(
                    f"realized source hash does not match portable trust anchor: {artifact_id}"
                )
            if check_files:
                provenance = realized["provenance"]
                maintenance = provenance.get("maintenance")
                if not isinstance(maintenance, Mapping):
                    raise ValueError(
                        f"realized TensorRT provenance lacks maintenance: {artifact_id}"
                    )
                maintenance_relative = _relative_path(maintenance.get("manifest"))
                if not str(maintenance_relative).startswith(
                    "DS9/models/engine_maintenance/"
                ):
                    raise ValueError(
                        f"maintenance evidence is outside the private root: {artifact_id}"
                    )
                maintenance_path = _physical_path(maintenance_relative, root)
                maintenance_path = _bounded_private_path(
                    maintenance_path,
                    root / "models/engine_maintenance",
                    "maintenance manifest",
                )
                maintenance_payload, maintenance_raw = _load_private_json_with_bytes(
                    maintenance_path, "maintenance manifest"
                )
                maintenance_inputs = maintenance_payload.get("inputs")
                maintenance_source_contract_input = (
                    maintenance_inputs.get("source_contracts")
                    if isinstance(maintenance_inputs, Mapping)
                    else None
                )
                maintenance_source_contracts_sha256 = (
                    str(maintenance_source_contract_input.get("sha256") or "").strip()
                    if isinstance(maintenance_source_contract_input, Mapping)
                    else ""
                )
                if (
                    maintenance_source_contracts_sha256
                    != current_source_contracts_sha256
                ):
                    raise ValueError(
                        "maintenance source-contract input digest differs from "
                        f"current authority: {artifact_id}"
                    )
                output = _physical_path(
                    _relative_path(by_id[artifact_id]["output"]), root
                )
                if output.is_symlink() or not output.is_file():
                    raise ValueError(
                        f"realized engine output is missing or unsafe: {artifact_id}"
                    )
                output_hash = _file_sha256(output)
                output_size = output.stat().st_size
                if provenance.get("output_sha256") != output_hash:
                    raise ValueError(
                        f"realized output digest differs from engine bytes: {artifact_id}"
                    )
                _validate_engine_maintenance_proof(
                    artifact_id=artifact_id,
                    artifact=by_id[artifact_id],
                    provenance=provenance,
                    maintenance_payload=maintenance_payload,
                    target=base_payload.get("target") or {},
                    source_contract_document=source_contract_document,
                    current_source_contracts_sha256=(current_source_contracts_sha256),
                    artifact_root=root,
                    maintenance_manifest_path=maintenance_path,
                    maintenance_manifest_sha256=_sha256_bytes(maintenance_raw),
                    output_sha256=output_hash,
                    output_size_bytes=output_size,
                )
            by_id[artifact_id]["state"] = realized["state"]
            by_id[artifact_id]["provenance"] = copy.deepcopy(
                dict(realized["provenance"])
            )
        result = validate_manifest(
            base_payload,
            check_files=check_files,
            profile=profile,
            require_provenance=require_provenance,
            artifact_root=root,
        )
        result["realization"] = str(realized_path)
        result["realization_sha256"] = realization_sha256
        result["realized_artifact_count"] = len(realized_artifacts)
        if (
            _sha256_bytes(
                _read_regular_owned_bytes(base_path, "base-manifest final CAS")
            )
            != base_sha256
        ):
            raise ValueError("base-manifest authority changed during validation")
        if (
            _sha256_bytes(
                _read_regular_owned_bytes(SOURCE_CONTRACTS, "source-contract final CAS")
            )
            != current_source_contracts_sha256
        ):
            raise ValueError("source-contract authority changed during validation")
        if (
            _sha256_bytes(
                _read_private_bytes(realized_path, "asset realization final CAS")
            )
            != realization_sha256
        ):
            raise ValueError("asset realization changed during validation")
        return result
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"asset realization: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "blockers": [],
            "profile": profile,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require outputs and sources for the selected profile.",
    )
    parser.add_argument(
        "--profile",
        default="canonical",
        help="Artifact profile to gate; use full for every artifact.",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Require complete provenance and verify output hashes.",
    )
    parser.add_argument(
        "--require-realization",
        action="store_true",
        help="Require and merge <artifact-root>/asset_realization.json before gating.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=(
            Path(os.environ["NOESIS_DS9_ARTIFACT_ROOT"])
            if str(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "")).strip()
            else None
        ),
        help=(
            "Physical root for virtual DS9/models paths. Defaults to "
            "NOESIS_DS9_ARTIFACT_ROOT when set."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path = args.manifest.resolve()
        manifest = _load_yaml(manifest_path)
        if args.require_realization:
            if args.artifact_root is None:
                raise ValueError("--require-realization requires --artifact-root")
            result = validate_asset_realization(
                manifest_path,
                Path(args.artifact_root) / REALIZATION_FILENAME,
                args.artifact_root,
                profile=str(args.profile),
                check_files=bool(args.check_files),
                require_provenance=bool(args.require_provenance),
            )
        else:
            result = validate_manifest(
                manifest,
                check_files=bool(args.check_files),
                profile=str(args.profile),
                require_provenance=bool(args.require_provenance),
                artifact_root=args.artifact_root,
            )
    except Exception as exc:
        result = {
            "ok": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
            "blockers": [],
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for message in result.get("errors", []):
            print(f"[FAIL] {message}", file=sys.stderr)
        for message in result.get("warnings", []):
            print(f"[WARN] {message}")
        if result.get("ok"):
            print(
                f"[OK] DS9 artifact manifest is structurally valid ({result.get('artifact_count')} artifacts; "
                f"profile={result.get('profile')}, selected={result.get('selected_count')})"
            )
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
