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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nvml_gpu_memory_sampler as gpu_sampler  # noqa: E402
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
# DeepStream 9.1 admits no pre-guard DS9.0 engine receipts. Every newly
# realized engine must carry the sealed NVML guard emitted by maintenance.
LEGACY_GPU_MEMORY_GUARD_EXEMPTIONS: dict[str, dict[str, str]] = {}
UNCONDITIONALLY_GUARD_REQUIRED_ARTIFACT_IDS = {
    "engine.wholebody49_s_masks",
    "engine.wholebody49_x_boxes",
}
RUNTIME_IMAGE_AUTHORITY = {
    "reference": "noesis-ds9-runtime:9.1-20260812",
    "image_id": "sha256:b97a32b082e74265c15e767bcaafa4dc1d8947e53feb36adb9baafdf69ba762e",
    "parent_reference": "noesis-ds9-dev:9.1-20260812",
    "parent_image_id": "sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872",
    "base_digest": "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994",
    "tensorrt_version": "10.16.0.72",
    "cuda_version": "13.2.0.046",
    "dockerfile": "DS9/docker/Dockerfile.runtime",
    "dockerfile_sha256": "4061b2dd98298aa07f0438ccf8ea9e1ae6e238621ba4b5ed363d4ce87d6cc133",
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
    if manifest.get("schema_version") != 2:
        errors.append("schema_version must be 2")

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
                "https://noesis.local/schemas/ds9-asset-manifest-v2.json"
            ):
                errors.append("schema $id is not the DS9 manifest v2 identifier")

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
    build_image = target.get("build_image")
    expected_build_image = {
        "reference": "noesis-ds9-dev:9.1-20260812",
        "image_id": "sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872",
        "base_digest": "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994",
        "tensorrt_version": "10.16.0.72",
        "cuda_version": "13.2.0.046",
        "dockerfile": "DS9/docker/Dockerfile",
        "dockerfile_sha256": "9f5f63a18c41256e06cab5514dcb6c5b47d290b8ea06a172490026776c56f01a",
        "requirements": "DS9/docker/requirements.lock.txt",
        "requirements_sha256": "de35fb439f5c9bfd05d7fbc23436122aee033bd7b584b2eb139358e50211be48",
    }
    if (
        not isinstance(build_image, Mapping)
        or dict(build_image) != expected_build_image
    ):
        errors.append(
            "target.build_image does not match the reviewed DS9 image authority"
        )
    else:
        for path_key, digest_key in (
            ("dockerfile", "dockerfile_sha256"),
            ("requirements", "requirements_sha256"),
        ):
            authority_path = REPO_ROOT / str(build_image[path_key])
            if (
                not authority_path.is_file()
                or _hash_file(authority_path) != build_image[digest_key]
            ):
                errors.append(
                    f"target.build_image {path_key} bytes differ from the declared digest"
                )

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("runtime must be a mapping")
        runtime = {}
    runtime_image = runtime.get("image")
    if (
        not isinstance(runtime_image, Mapping)
        or dict(runtime_image) != RUNTIME_IMAGE_AUTHORITY
    ):
        errors.append(
            "runtime.image does not match the reviewed DS9 runtime-image authority"
        )
    else:
        if (
            not isinstance(build_image, Mapping)
            or runtime_image["parent_reference"] != build_image.get("reference")
            or runtime_image["parent_image_id"] != build_image.get("image_id")
        ):
            errors.append("runtime.image parent does not match target.build_image")
        runtime_dockerfile = REPO_ROOT / str(runtime_image["dockerfile"])
        if (
            not runtime_dockerfile.is_file()
            or runtime_dockerfile.is_symlink()
            or _hash_file(runtime_dockerfile) != runtime_image["dockerfile_sha256"]
        ):
            errors.append(
                "runtime.image Dockerfile bytes differ from the declared digest"
            )
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
SOURCE_CONTRACT_REBASE_ROOT = "source_contract_rebase"
SOURCE_CONTRACT_REBASE_INPUT_ROOT = "inputs"
SOURCE_CONTRACT_REBASE_EVIDENCE_FILENAME = "source_contract_rebase_evidence.json"
SOURCE_CONTRACT_REBASE_CONTRACT = "noesis.ds9.source_contract_realization_rebase"
SOURCE_CONTRACT_REBASE_MAX_TRANSACTIONS = 4096
SOURCE_CONTRACT_PATH = "DS9/config/engine_source_contracts.json"
BASE_MANIFEST_PATH = "DS9/asset_manifest.yaml"
MANIFEST_REBASE_ROOT = "manifest_rebase"
MANIFEST_REBASE_EVIDENCE_FILENAME = "rebase_evidence.json"
MANIFEST_REBASE_CONTRACT = "noesis.ds9.asset_realization_rebase"
MANIFEST_REBASE_MAX_TRANSACTIONS = 4096
MANIFEST_REBASE_MAX_DIFF_PATHS = 4096
MAPANYTHING_TRANSITION_ROOT = "mapanything_authority_transition"
MAPANYTHING_TRANSITION_EVIDENCE_FILENAME = "transition_evidence.json"
MAPANYTHING_TRANSITION_CONTRACT = "noesis.ds9.mapanything_authority_transition"
MAPANYTHING_TRANSITION_PLAN_CONTRACT = (
    "noesis.ds9.mapanything_authority_transition.plan.v1"
)
MAPANYTHING_TRANSITION_MAX_TRANSACTIONS = 4096
MAPANYTHING_ARTIFACT_ID = "engine.mapanything"
MAPANYTHING_CONTRACT_NAME = "mapanything"
MAPANYTHING_OLD_MANIFEST_SHA256 = (
    "eed4340c1af587f903541f364824e643377b6149c5249e8314fc1d77fa12e789"
)
MAPANYTHING_NEW_MANIFEST_SHA256 = (
    "10e15351382e8a693a6acc048a05df3adff96a906fde50f0fa371fb0a8455198"
)
MAPANYTHING_OLD_SOURCE_CONTRACTS_SHA256 = (
    "7ddd449c82e80d5c3195c0ccfb4ba4c3d4d542c654234f5d1f31d4ea549a20e0"
)
MAPANYTHING_NEW_SOURCE_CONTRACTS_SHA256 = (
    "94144007b6e59f2eddd3239a6f0af02ce7d37224089e5f07a7a95d884f752b5a"
)
MAPANYTHING_OLD_REALIZED_RECORD_SHA256 = (
    "24504cc670bbeeb325f361c05ba360b749e1c4bb98465aeb636c5d7800fbd6d0"
)
MAPANYTHING_OLD_OUTPUT = "DS9/models/engines/mapanything_images_294x518_b3_fp16.plan"
MAPANYTHING_NEW_OUTPUT = "DS9/models/engines/mapanything_images_294x518_b3_fp32.plan"
MAPANYTHING_OLD_OUTPUT_SHA256 = (
    "aeb7140a56c31b8e420c7a1d31fb21ef4590c38d85e9eb41299dfe0d55b6891d"
)
MAPANYTHING_OLD_OUTPUT_SIZE_BYTES = 1_850_829_956
_MAPANYTHING_TRANSITION_SNAPSHOT_FILENAMES = {
    "old_asset_manifest.yaml",
    "new_asset_manifest.yaml",
    "old_engine_source_contracts.json",
    "new_engine_source_contracts.json",
    "asset_realization.before.json",
    "asset_realization.after.json",
}
_MAPANYTHING_TRANSITION_INVENTORY = _MAPANYTHING_TRANSITION_SNAPSHOT_FILENAMES | {
    MAPANYTHING_TRANSITION_EVIDENCE_FILENAME
}
_MAPANYTHING_TRANSITION_TERMINAL_ABORT_STATES = {
    "aborted_before_commit",
    "aborted_by_recovery",
    "rolled_back_after_evidence_failure",
}
_SOURCE_CONTRACT_DOCUMENT_KEYS = {"schema_version", "contracts"}
_SOURCE_REBASE_REQUIRED_EVIDENCE_KEYS = {
    "schema_version",
    "contract",
    "transaction_id",
    "state",
    "prepared_at_utc",
    "committed_at_utc",
    "old_source_contracts",
    "new_source_contracts",
    "base_manifest",
    "realization",
    "changed_contracts",
    "mapped_unrealized_artifact_ids",
    "realized_engine_ids",
    "mutation_paths",
    "semantic_checks",
}
_SOURCE_REBASE_OPTIONAL_EVIDENCE_KEYS = {"realization_replace_recovery"}
_SOURCE_REBASE_SEMANTIC_CHECK_KEYS = {
    "source_contract_schema_unchanged",
    "source_contract_membership_unchanged",
    "changed_contracts_exact_allowlist",
    "changed_contracts_map_only_to_unrealized_engines",
    "base_manifest_unchanged",
    "realized_artifact_records_unchanged",
    "proposal_mutation_is_exact",
}
_MANIFEST_REBASE_REQUIRED_EVIDENCE_KEYS = {
    "schema_version",
    "contract",
    "transaction_id",
    "state",
    "prepared_at_utc",
    "committed_at_utc",
    "old_manifest",
    "new_manifest",
    "source_contracts",
    "realization",
    "mutation_paths",
    "realized_engine_ids",
    "semantic_checks",
    "manifest_diff",
}
_MANIFEST_REBASE_OPTIONAL_EVIDENCE_KEYS = {
    "runtime_image_authority",
    "realization_replace_recovery",
}
_MANIFEST_REBASE_REQUIRED_SEMANTIC_CHECK_KEYS = {
    "target_unchanged",
    "source_contracts_unchanged",
    "all_engine_artifacts_unchanged",
    "realized_artifacts_unchanged",
}


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


def _parse_explicit_utc(value: object, label: str) -> datetime:
    raw = str(value or "").strip()
    if not raw.endswith("Z"):
        raise ValueError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must use UTC")
    return parsed


def _require_private_directory(path: Path, label: str) -> Path:
    candidate = _lexical_absolute(path)
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a real directory: {candidate}")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError(f"{label} must be owned by the current uid with mode 0700")
    return candidate


def _validate_private_parent_chain(path: Path, root: Path, label: str) -> None:
    candidate = _lexical_absolute(path)
    root = _require_private_directory(root, f"{label} root")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its private root") from exc
    current = root
    for component in relative.parts[:-1]:
        current /= component
        _require_private_directory(current, f"{label} parent")


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


def _require_exact_record(
    value: object, expected_keys: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{label} has unexpected or missing fields")
    return value


def _require_sorted_unique_strings(
    value: object,
    label: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(value)
        or len(set(value)) != len(value)
    ):
        qualifier = "possibly empty" if allow_empty else "nonempty"
        raise ValueError(f"{label} must be a sorted, unique, {qualifier} string list")
    return tuple(value)


def _validate_replace_recovery(value: object, label: str) -> None:
    recovery = _require_exact_record(
        value,
        {"outcome", "recovered_at_utc"},
        label,
    )
    if recovery.get("outcome") != "proposal_exact_bytes_refsynced":
        raise ValueError(f"{label} outcome is invalid")
    _parse_explicit_utc(recovery.get("recovered_at_utc"), f"{label} timestamp")


def _load_source_contract_snapshot(
    artifact_root: Path,
    evidence_root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes, str]:
    relative = _relative_path(record.get("path"))
    if tuple(relative.parts[:2]) != (
        SOURCE_CONTRACT_REBASE_ROOT,
        SOURCE_CONTRACT_REBASE_INPUT_ROOT,
    ):
        raise ValueError(f"{label} path is outside the reviewed input evidence root")
    path = _bounded_private_path(
        artifact_root / relative,
        evidence_root,
        label,
    )
    _validate_private_parent_chain(path, evidence_root, label)
    payload, raw = _load_private_json_with_bytes(path, label)
    observed_hash = _sha256_bytes(raw)
    expected_hash = _require_sha256(record.get("sha256"), f"{label} SHA-256")
    if observed_hash != expected_hash:
        raise ValueError(
            f"{label} digest mismatch: expected={expected_hash} observed={observed_hash}"
        )
    _validate_source_contract_document(payload, label)
    return payload, raw, observed_hash


def _require_owned_nonwritable_directory(path: Path, label: str) -> Path:
    candidate = _lexical_absolute(path)
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a real directory: {candidate}")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise ValueError(
            f"{label} must be owned by the current uid and not group/world writable"
        )
    return candidate


def _validate_manifest_snapshot_parent_chain(
    path: Path, root: Path, label: str
) -> None:
    candidate = _lexical_absolute(path)
    root = _require_owned_nonwritable_directory(root, f"{label} root")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its evidence root") from exc
    current = root
    for component in relative.parts[:-1]:
        current /= component
        _require_private_directory(current, f"{label} parent")


def _load_manifest_snapshot(
    artifact_root: Path,
    evidence_root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes, str]:
    relative = _relative_path(record.get("path"))
    if not relative.parts or relative.parts[0] != MANIFEST_REBASE_ROOT:
        raise ValueError(f"{label} path is outside the manifest evidence root")
    path = _bounded_private_path(
        artifact_root / relative,
        evidence_root,
        label,
    )
    _validate_manifest_snapshot_parent_chain(path, evidence_root, label)
    raw = _read_private_bytes(path, label)
    observed_hash = _sha256_bytes(raw)
    expected_hash = _require_sha256(record.get("sha256"), f"{label} SHA-256")
    if observed_hash != expected_hash:
        raise ValueError(
            f"{label} digest mismatch: expected={expected_hash} observed={observed_hash}"
        )
    return _parse_yaml_mapping(raw, label), raw, observed_hash


def _manifest_engine_records(
    payload: Mapping[str, Any], label: str
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    target = payload.get("target")
    artifacts = payload.get("artifacts")
    if not isinstance(target, Mapping) or not isinstance(artifacts, list):
        raise ValueError(f"{label} lacks target/artifact authority")
    seen: set[str] = set()
    engines: dict[str, Mapping[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} contains a malformed artifact record")
        artifact_id = str(row.get("id") or "").strip()
        if not artifact_id or artifact_id in seen:
            raise ValueError(f"{label} contains a missing or duplicate artifact ID")
        seen.add(artifact_id)
        if row.get("kind") == "tensorrt_engine":
            engines[artifact_id] = row
    if not set(ENGINE_NAME_BY_ARTIFACT_ID) <= set(engines):
        raise ValueError(f"{label} lacks a reviewed TensorRT engine artifact")
    return target, engines


def _manifest_artifact_records(
    payload: Mapping[str, Any], label: str
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{label} artifacts must be a list")
    all_rows: dict[str, Mapping[str, Any]] = {}
    engines: dict[str, Mapping[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} contains a malformed artifact record")
        artifact_id = str(row.get("id") or "").strip()
        if not artifact_id or artifact_id in all_rows:
            raise ValueError(f"{label} contains a missing or duplicate artifact ID")
        all_rows[artifact_id] = row
        if row.get("kind") == "tensorrt_engine":
            engines[artifact_id] = row
    if not set(ENGINE_NAME_BY_ARTIFACT_ID) <= set(engines):
        raise ValueError(f"{label} lacks a reviewed TensorRT engine artifact")
    return all_rows, engines


def _append_manifest_diff(
    paths: list[str], path: tuple[str, ...], left: object, right: object
) -> None:
    if len(paths) >= MANIFEST_REBASE_MAX_DIFF_PATHS:
        raise ValueError("manifest semantic diff exceeds the accepted bound")
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                paths.append(".".join((*path, str(key))))
            else:
                _append_manifest_diff(paths, (*path, str(key)), left[key], right[key])
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            paths.append(".".join((*path, "length")))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _append_manifest_diff(paths, (*path, str(index)), left_item, right_item)
        return
    if left != right:
        paths.append(".".join(path))


def _manifest_diff_summary(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    old_rows: Mapping[str, Mapping[str, Any]],
    new_rows: Mapping[str, Mapping[str, Any]],
    engine_ids: set[str],
) -> dict[str, Any]:
    paths: list[str] = []
    for key in sorted((set(old) | set(new)) - {"artifacts"}, key=str):
        if key not in old or key not in new:
            paths.append(str(key))
        else:
            _append_manifest_diff(paths, (str(key),), old[key], new[key])
    for artifact_id in sorted(set(old_rows) | set(new_rows)):
        if artifact_id not in old_rows or artifact_id not in new_rows:
            paths.append(f"artifacts.{artifact_id}")
        else:
            _append_manifest_diff(
                paths,
                ("artifacts", artifact_id),
                old_rows[artifact_id],
                new_rows[artifact_id],
            )
    changed_ids = sorted(
        artifact_id
        for artifact_id in set(old_rows) | set(new_rows)
        if old_rows.get(artifact_id) != new_rows.get(artifact_id)
    )
    return {
        "changed_path_count": len(paths),
        "changed_paths": paths,
        "changed_artifact_ids": changed_ids,
        "changed_non_engine_artifact_ids": [
            artifact_id for artifact_id in changed_ids if artifact_id not in engine_ids
        ],
    }


def _manifest_runtime_image_transition(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    old_runtime = old.get("runtime")
    new_runtime = new.get("runtime")
    if not isinstance(old_runtime, Mapping) or not isinstance(new_runtime, Mapping):
        raise ValueError("manifest runtime authority must be a mapping")
    old_without_image = dict(old_runtime)
    new_without_image = dict(new_runtime)
    old_image = old_without_image.pop("image", None)
    new_image = new_without_image.pop("image", None)
    if old_without_image != new_without_image:
        raise ValueError("manifest rebase changed runtime authority outside image")
    changed = old_image != new_image
    if (new_image is not None or changed) and (
        not isinstance(new_image, Mapping) or dict(new_image) != RUNTIME_IMAGE_AUTHORITY
    ):
        raise ValueError("manifest rebase runtime image is not reviewed authority")
    if old_image is not None and not isinstance(old_image, Mapping):
        raise ValueError("old manifest runtime image authority is malformed")
    return {
        "changed": changed,
        "before": copy.deepcopy(dict(old_image))
        if isinstance(old_image, Mapping)
        else None,
        "after": copy.deepcopy(dict(new_image))
        if isinstance(new_image, Mapping)
        else None,
    }


def _recompute_manifest_rebase_semantics(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if old.get("target") != new.get("target"):
        raise ValueError("manifest rebase changed DS9 target authority")
    runtime_transition = _manifest_runtime_image_transition(old, new)
    ignored_top_level = {"artifacts", "runtime", "updated_at"}
    old_top_level = set(old) - ignored_top_level
    new_top_level = set(new) - ignored_top_level
    if old_top_level != new_top_level:
        raise ValueError("manifest rebase changed unsupported top-level keys")
    for key in old_top_level:
        if old[key] != new[key]:
            raise ValueError(
                f"manifest rebase changed unsupported top-level authority: {key}"
            )
    old_rows, old_engines = _manifest_artifact_records(old, "old manifest")
    new_rows, new_engines = _manifest_artifact_records(new, "new manifest")
    if old_engines != new_engines:
        raise ValueError("manifest rebase changed TensorRT engine authority")
    return (
        _manifest_diff_summary(old, new, old_rows, new_rows, set(old_engines)),
        runtime_transition,
    )


def _transition_canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _parse_transition_json(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        payload = strict_json_loads(raw, label=label)
    except StrictJSONError as exc:
        raise ValueError(f"{label} is not valid unique-key UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} document root must be a mapping")
    return payload


def _load_transition_snapshot(
    transaction_dir: Path, filename: str, label: str
) -> bytes:
    if filename not in _MAPANYTHING_TRANSITION_SNAPSHOT_FILENAMES:
        raise ValueError(f"{label} has an unreviewed snapshot filename")
    path = _bounded_private_path(
        transaction_dir / filename,
        transaction_dir,
        label,
    )
    return _read_private_bytes(path, label)


def _mapanything_transition_manifest_semantics(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    old_top = dict(old)
    new_top = dict(new)
    old_top.pop("artifacts", None)
    new_top.pop("artifacts", None)
    if old_top != new_top:
        raise ValueError(
            "MapAnything transition manifest changed outside artifact inventory"
        )
    old_rows, _old_engines = _manifest_artifact_records(
        old, "old MapAnything transition manifest"
    )
    new_rows, _new_engines = _manifest_artifact_records(
        new, "new MapAnything transition manifest"
    )
    if set(old_rows) != set(new_rows) or MAPANYTHING_ARTIFACT_ID not in old_rows:
        raise ValueError("MapAnything transition manifest membership drifted")
    for artifact_id in set(old_rows) - {MAPANYTHING_ARTIFACT_ID}:
        if old_rows[artifact_id] != new_rows[artifact_id]:
            raise ValueError(
                "MapAnything transition changed another manifest artifact: "
                f"{artifact_id}"
            )
    old_map = old_rows[MAPANYTHING_ARTIFACT_ID]
    new_map = new_rows[MAPANYTHING_ARTIFACT_ID]
    changed_paths: list[str] = []
    _append_manifest_diff(changed_paths, (), old_map, new_map)
    if changed_paths != ["compatibility.precision", "output"]:
        raise ValueError("MapAnything transition manifest diff is not exact")
    old_compatibility = old_map.get("compatibility")
    new_compatibility = new_map.get("compatibility")
    if (
        old_map.get("kind") != "tensorrt_engine"
        or old_map.get("output") != MAPANYTHING_OLD_OUTPUT
        or new_map.get("output") != MAPANYTHING_NEW_OUTPUT
        or not isinstance(old_compatibility, Mapping)
        or not isinstance(new_compatibility, Mapping)
        or old_compatibility.get("precision") != "fp16"
        or new_compatibility.get("precision") != "fp32"
    ):
        raise ValueError("MapAnything transition manifest endpoints drifted")
    return {
        "artifact_id": MAPANYTHING_ARTIFACT_ID,
        "changed_paths": changed_paths,
        "old_record_sha256": _sha256_bytes(_transition_canonical_json(old_map)),
        "new_record_sha256": _sha256_bytes(_transition_canonical_json(new_map)),
        "old_output": MAPANYTHING_OLD_OUTPUT,
        "new_output": MAPANYTHING_NEW_OUTPUT,
    }


def _mapanything_transition_source_semantics(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    old_contracts = _validate_source_contract_document(
        old, "old MapAnything transition source contracts"
    )
    new_contracts = _validate_source_contract_document(
        new, "new MapAnything transition source contracts"
    )
    changed = sorted(
        name for name in old_contracts if old_contracts[name] != new_contracts[name]
    )
    if changed != [MAPANYTHING_CONTRACT_NAME]:
        raise ValueError("MapAnything transition source-contract diff is not exact")
    old_map = old_contracts[MAPANYTHING_CONTRACT_NAME]
    new_map = new_contracts[MAPANYTHING_CONTRACT_NAME]
    old_build = old_map.get("maintenance_build")
    new_build = new_map.get("maintenance_build")
    if (
        not isinstance(old_build, Mapping)
        or old_build.get("precision_arg") != "--fp16"
        or not isinstance(new_build, Mapping)
        or new_build.get("precision") != "fp32"
        or "precision_arg" in new_build
        or not isinstance(new_map.get("quality_gate"), Mapping)
    ):
        raise ValueError("MapAnything transition source-contract endpoints drifted")
    return {
        "changed_contracts": [MAPANYTHING_CONTRACT_NAME],
        "old_contract_sha256": _sha256_bytes(_transition_canonical_json(old_map)),
        "new_contract_sha256": _sha256_bytes(_transition_canonical_json(new_map)),
    }


def _verify_mapanything_historic_output(artifact_root: Path) -> None:
    relative = _relative_path(MAPANYTHING_OLD_OUTPUT)
    path = _bounded_private_path(
        _physical_path(relative, artifact_root),
        artifact_root,
        "historic MapAnything FP16 output",
    )
    try:
        lexical = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("historic MapAnything FP16 output is missing") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("historic MapAnything FP16 output is unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (lexical.st_dev, lexical.st_ino)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size != MAPANYTHING_OLD_OUTPUT_SIZE_BYTES
        ):
            raise ValueError(
                "historic MapAnything FP16 output ownership/link/size drifted"
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if remaining or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise ValueError("historic MapAnything FP16 output changed while read")
        if digest.hexdigest() != MAPANYTHING_OLD_OUTPUT_SHA256:
            raise ValueError("historic MapAnything FP16 output hash drifted")
    finally:
        os.close(descriptor)


def _mapanything_transition_plan(
    *,
    old_realization_hash: str,
    new_realization_hash: str,
    updated_before: str,
    updated_after: str,
    inventory_before: list[str],
    inventory_after: list[str],
    retired_artifact: Mapping[str, Any],
    manifest_diff: Mapping[str, Any],
    source_contract_diff: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": MAPANYTHING_TRANSITION_PLAN_CONTRACT,
        "authorities": {
            "manifest": {
                "old_sha256": MAPANYTHING_OLD_MANIFEST_SHA256,
                "new_sha256": MAPANYTHING_NEW_MANIFEST_SHA256,
            },
            "source_contracts": {
                "old_sha256": MAPANYTHING_OLD_SOURCE_CONTRACTS_SHA256,
                "new_sha256": MAPANYTHING_NEW_SOURCE_CONTRACTS_SHA256,
            },
        },
        "realization": {
            "old_sha256": old_realization_hash,
            "new_sha256": new_realization_hash,
            "updated_at_utc_before": updated_before,
            "updated_at_utc_after": updated_after,
            "realized_artifact_ids_before": inventory_before,
            "realized_artifact_ids_after": inventory_after,
        },
        "retired_artifact": copy.deepcopy(dict(retired_artifact)),
        "manifest_diff": copy.deepcopy(dict(manifest_diff)),
        "source_contract_diff": copy.deepcopy(dict(source_contract_diff)),
        "mutation_paths": [
            "artifacts.engine.mapanything",
            "base_manifest.sha256",
            "source_contracts.sha256",
            "updated_at_utc",
        ],
    }


def _mapanything_transition_committed_edge(
    *, artifact_root: Path, transaction_dir: Path
) -> dict[str, Any] | None:
    _require_private_directory(
        transaction_dir, "MapAnything authority-transition transaction"
    )
    inventory = list(transaction_dir.iterdir())
    if (
        len(inventory) != len(_MAPANYTHING_TRANSITION_INVENTORY)
        or {path.name for path in inventory} != _MAPANYTHING_TRANSITION_INVENTORY
        or any(path.is_symlink() or not path.is_file() for path in inventory)
    ):
        raise ValueError("MapAnything authority-transition inventory drifted")
    evidence_path = transaction_dir / MAPANYTHING_TRANSITION_EVIDENCE_FILENAME
    evidence, evidence_raw = _load_private_json_with_bytes(
        evidence_path, "MapAnything authority-transition evidence"
    )
    if _transition_canonical_json(evidence) != evidence_raw:
        raise ValueError("MapAnything authority-transition evidence is not canonical")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("contract") != MAPANYTHING_TRANSITION_CONTRACT
    ):
        raise ValueError("MapAnything authority-transition evidence contract drifted")
    transaction_id = str(evidence.get("transaction_id") or "")
    if (
        transaction_id != transaction_dir.name
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,127}", transaction_id) is None
    ):
        raise ValueError("MapAnything authority-transition transaction ID is invalid")
    state = str(evidence.get("state") or "")
    if state in _MAPANYTHING_TRANSITION_TERMINAL_ABORT_STATES:
        return None
    if state != "committed":
        raise ValueError("unresolved MapAnything authority-transition evidence")
    required_keys = {
        "schema_version",
        "contract",
        "transaction_id",
        "state",
        "prepared_at_utc",
        "committed_at_utc",
        "aborted_at_utc",
        "rolled_back_at_utc",
        "recovery",
        "realization_replace_recovery",
        "plan_sha256",
        "authorities",
        "realization",
        "retired_artifact",
        "manifest_diff",
        "source_contract_diff",
        "mutation_paths",
        "semantic_checks",
        "verification_stages",
    }
    if set(evidence) != required_keys:
        raise ValueError("MapAnything authority-transition evidence fields drifted")
    prepared_at = _parse_explicit_utc(
        evidence.get("prepared_at_utc"), "MapAnything transition prepared_at_utc"
    )
    committed_at = _parse_explicit_utc(
        evidence.get("committed_at_utc"), "MapAnything transition committed_at_utc"
    )
    if committed_at < prepared_at:
        raise ValueError("MapAnything transition commit predates preparation")
    if (
        evidence.get("aborted_at_utc") is not None
        or evidence.get("rolled_back_at_utc") is not None
    ):
        raise ValueError("committed MapAnything transition contains abort metadata")
    recovery = evidence.get("recovery")
    if recovery is not None:
        recovery = _require_exact_record(
            recovery,
            {"outcome", "recorded_at_utc"},
            "MapAnything transition recovery",
        )
        if recovery.get("outcome") != "exact_after_realization_terminalized":
            raise ValueError("committed MapAnything recovery outcome is invalid")
        _parse_explicit_utc(
            recovery.get("recorded_at_utc"), "MapAnything recovery timestamp"
        )
    replace_recovery = evidence.get("realization_replace_recovery")
    if replace_recovery is not None:
        _validate_replace_recovery(
            replace_recovery, "MapAnything realization replace recovery"
        )
    stages = evidence.get("verification_stages")
    if (
        not isinstance(stages, list)
        or len(stages) != len(set(stages))
        or stages[:2] != ["initial", "snapshots_written"]
    ):
        raise ValueError("MapAnything transition verification stages drifted")
    if recovery is None and stages != [
        "initial",
        "snapshots_written",
        "pre_cas",
        "post_cas",
        "pre_evidence_commit",
    ]:
        raise ValueError("MapAnything transition continuous verification is incomplete")
    if recovery is not None and stages[-1:] != ["recovery_final"]:
        raise ValueError("MapAnything recovery lacks final verification")

    authorities = _require_exact_record(
        evidence.get("authorities"),
        {"manifest", "source_contracts"},
        "MapAnything transition authorities",
    )
    manifest_authority = _require_exact_record(
        authorities.get("manifest"),
        {"old_path", "old_sha256", "new_path", "new_authority_path", "new_sha256"},
        "MapAnything transition manifest authority",
    )
    source_authority = _require_exact_record(
        authorities.get("source_contracts"),
        {"old_path", "old_sha256", "new_path", "new_authority_path", "new_sha256"},
        "MapAnything transition source authority",
    )
    expected_manifest_authority = {
        "old_path": "old_asset_manifest.yaml",
        "old_sha256": MAPANYTHING_OLD_MANIFEST_SHA256,
        "new_path": "new_asset_manifest.yaml",
        "new_authority_path": BASE_MANIFEST_PATH,
        "new_sha256": MAPANYTHING_NEW_MANIFEST_SHA256,
    }
    expected_source_authority = {
        "old_path": "old_engine_source_contracts.json",
        "old_sha256": MAPANYTHING_OLD_SOURCE_CONTRACTS_SHA256,
        "new_path": "new_engine_source_contracts.json",
        "new_authority_path": SOURCE_CONTRACT_PATH,
        "new_sha256": MAPANYTHING_NEW_SOURCE_CONTRACTS_SHA256,
    }
    if (
        dict(manifest_authority) != expected_manifest_authority
        or dict(source_authority) != expected_source_authority
    ):
        raise ValueError("MapAnything transition authority endpoints drifted")

    old_manifest_raw = _load_transition_snapshot(
        transaction_dir, "old_asset_manifest.yaml", "old transition manifest"
    )
    new_manifest_raw = _load_transition_snapshot(
        transaction_dir, "new_asset_manifest.yaml", "new transition manifest"
    )
    old_source_raw = _load_transition_snapshot(
        transaction_dir,
        "old_engine_source_contracts.json",
        "old transition source contracts",
    )
    new_source_raw = _load_transition_snapshot(
        transaction_dir,
        "new_engine_source_contracts.json",
        "new transition source contracts",
    )
    old_realization_raw = _load_transition_snapshot(
        transaction_dir,
        "asset_realization.before.json",
        "old transition realization",
    )
    new_realization_raw = _load_transition_snapshot(
        transaction_dir,
        "asset_realization.after.json",
        "new transition realization",
    )
    for observed, expected, label in (
        (
            _sha256_bytes(old_manifest_raw),
            MAPANYTHING_OLD_MANIFEST_SHA256,
            "old manifest",
        ),
        (
            _sha256_bytes(new_manifest_raw),
            MAPANYTHING_NEW_MANIFEST_SHA256,
            "new manifest",
        ),
        (
            _sha256_bytes(old_source_raw),
            MAPANYTHING_OLD_SOURCE_CONTRACTS_SHA256,
            "old source contracts",
        ),
        (
            _sha256_bytes(new_source_raw),
            MAPANYTHING_NEW_SOURCE_CONTRACTS_SHA256,
            "new source contracts",
        ),
    ):
        if observed != expected:
            raise ValueError(f"MapAnything transition {label} snapshot hash drifted")
    old_manifest = _parse_yaml_mapping(old_manifest_raw, "old transition manifest")
    new_manifest = _parse_yaml_mapping(new_manifest_raw, "new transition manifest")
    old_source = _parse_transition_json(
        old_source_raw, "old transition source contracts"
    )
    new_source = _parse_transition_json(
        new_source_raw, "new transition source contracts"
    )
    old_realization = _parse_transition_json(
        old_realization_raw, "old transition realization"
    )
    new_realization = _parse_transition_json(
        new_realization_raw, "new transition realization"
    )
    if (
        _transition_canonical_json(old_realization) != old_realization_raw
        or _transition_canonical_json(new_realization) != new_realization_raw
    ):
        raise ValueError(
            "MapAnything transition realization snapshots are not canonical"
        )
    manifest_diff = _mapanything_transition_manifest_semantics(
        old_manifest, new_manifest
    )
    source_diff = _mapanything_transition_source_semantics(old_source, new_source)
    if evidence.get("manifest_diff") != manifest_diff:
        raise ValueError("MapAnything transition manifest diff evidence drifted")
    if evidence.get("source_contract_diff") != source_diff:
        raise ValueError("MapAnything transition source diff evidence drifted")

    realization_record = _require_exact_record(
        evidence.get("realization"),
        {
            "path",
            "old_snapshot",
            "new_snapshot",
            "old_sha256",
            "new_sha256",
            "updated_at_utc_before",
            "updated_at_utc_after",
            "realized_artifact_ids_before",
            "realized_artifact_ids_after",
        },
        "MapAnything transition realization binding",
    )
    if (
        realization_record.get("path") != REALIZATION_FILENAME
        or realization_record.get("old_snapshot") != "asset_realization.before.json"
        or realization_record.get("new_snapshot") != "asset_realization.after.json"
    ):
        raise ValueError("MapAnything transition realization path drifted")
    old_realization_hash = _require_sha256(
        realization_record.get("old_sha256"), "old transition realization hash"
    )
    new_realization_hash = _require_sha256(
        realization_record.get("new_sha256"), "new transition realization hash"
    )
    if (
        _sha256_bytes(old_realization_raw) != old_realization_hash
        or _sha256_bytes(new_realization_raw) != new_realization_hash
        or old_realization_hash == new_realization_hash
    ):
        raise ValueError("MapAnything transition realization snapshot hash drifted")
    before_time = _parse_explicit_utc(
        realization_record.get("updated_at_utc_before"),
        "MapAnything transition old realization timestamp",
    )
    after_time = _parse_explicit_utc(
        realization_record.get("updated_at_utc_after"),
        "MapAnything transition new realization timestamp",
    )
    if after_time <= before_time:
        raise ValueError("MapAnything transition realization timestamp did not advance")
    if set(old_realization) != {
        "schema_version",
        "contract",
        "base_manifest",
        "source_contracts",
        "created_at_utc",
        "updated_at_utc",
        "artifacts",
    } or set(new_realization) != set(old_realization):
        raise ValueError("MapAnything transition realization schema drifted")
    if (
        old_realization.get("contract") != REALIZATION_CONTRACT
        or new_realization.get("contract") != REALIZATION_CONTRACT
        or old_realization.get("schema_version") != 1
        or new_realization.get("schema_version") != 1
        or old_realization.get("base_manifest")
        != {"path": BASE_MANIFEST_PATH, "sha256": MAPANYTHING_OLD_MANIFEST_SHA256}
        or new_realization.get("base_manifest")
        != {"path": BASE_MANIFEST_PATH, "sha256": MAPANYTHING_NEW_MANIFEST_SHA256}
        or old_realization.get("source_contracts")
        != {
            "path": SOURCE_CONTRACT_PATH,
            "sha256": MAPANYTHING_OLD_SOURCE_CONTRACTS_SHA256,
        }
        or new_realization.get("source_contracts")
        != {
            "path": SOURCE_CONTRACT_PATH,
            "sha256": MAPANYTHING_NEW_SOURCE_CONTRACTS_SHA256,
        }
        or old_realization.get("created_at_utc")
        != new_realization.get("created_at_utc")
        or old_realization.get("updated_at_utc")
        != realization_record.get("updated_at_utc_before")
        or new_realization.get("updated_at_utc")
        != realization_record.get("updated_at_utc_after")
    ):
        raise ValueError("MapAnything transition realization authority drifted")
    old_artifacts = old_realization.get("artifacts")
    new_artifacts = new_realization.get("artifacts")
    if not isinstance(old_artifacts, Mapping) or not isinstance(new_artifacts, Mapping):
        raise ValueError("MapAnything transition realized inventories are malformed")
    before_ids = _require_sorted_unique_strings(
        realization_record.get("realized_artifact_ids_before"),
        "MapAnything transition before inventory",
        allow_empty=False,
    )
    after_ids = _require_sorted_unique_strings(
        realization_record.get("realized_artifact_ids_after"),
        "MapAnything transition after inventory",
        allow_empty=True,
    )
    if (
        list(before_ids) != sorted(old_artifacts)
        or list(after_ids) != sorted(new_artifacts)
        or set(before_ids) - set(after_ids) != {MAPANYTHING_ARTIFACT_ID}
        or set(after_ids) - set(before_ids)
        or MAPANYTHING_ARTIFACT_ID in new_artifacts
    ):
        raise ValueError("MapAnything transition retirement inventory drifted")
    for artifact_id in after_ids:
        if old_artifacts.get(artifact_id) != new_artifacts.get(artifact_id):
            raise ValueError(
                "MapAnything transition changed a preserved realization record"
            )
    retired_record = old_artifacts.get(MAPANYTHING_ARTIFACT_ID)
    if not isinstance(retired_record, Mapping):
        raise ValueError("MapAnything transition old realization record is missing")
    retired_record_hash = _sha256_bytes(_transition_canonical_json(retired_record))
    if retired_record_hash != MAPANYTHING_OLD_REALIZED_RECORD_SHA256:
        raise ValueError("MapAnything transition old realization record hash drifted")
    retired_artifact = _require_exact_record(
        evidence.get("retired_artifact"),
        {"artifact_id", "realized_record_sha256", "output"},
        "MapAnything retired artifact evidence",
    )
    retired_output = _require_exact_record(
        retired_artifact.get("output"),
        {"path", "sha256", "size_bytes"},
        "MapAnything retired output evidence",
    )
    expected_retired = {
        "artifact_id": MAPANYTHING_ARTIFACT_ID,
        "realized_record_sha256": MAPANYTHING_OLD_REALIZED_RECORD_SHA256,
        "output": {
            "path": MAPANYTHING_OLD_OUTPUT,
            "sha256": MAPANYTHING_OLD_OUTPUT_SHA256,
            "size_bytes": MAPANYTHING_OLD_OUTPUT_SIZE_BYTES,
        },
    }
    if (
        dict(retired_artifact) != expected_retired
        or dict(retired_output) != expected_retired["output"]
    ):
        raise ValueError("MapAnything retired output evidence drifted")
    provenance = retired_record.get("provenance")
    maintenance = (
        provenance.get("maintenance") if isinstance(provenance, Mapping) else None
    )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("output_sha256") != MAPANYTHING_OLD_OUTPUT_SHA256
        or not isinstance(maintenance, Mapping)
        or maintenance.get("output_size_bytes") != MAPANYTHING_OLD_OUTPUT_SIZE_BYTES
        or maintenance.get("precision") != "fp16"
    ):
        raise ValueError("MapAnything retired realization proof drifted")
    _verify_mapanything_historic_output(artifact_root)

    expected_mutations = [
        "artifacts.engine.mapanything",
        "base_manifest.sha256",
        "source_contracts.sha256",
        "updated_at_utc",
    ]
    if evidence.get("mutation_paths") != expected_mutations:
        raise ValueError("MapAnything transition mutation paths drifted")
    semantic_checks = evidence.get("semantic_checks")
    expected_checks = {
        "exact_reviewed_authority_hashes",
        "manifest_change_is_mapanything_fp32_only",
        "source_contract_change_is_mapanything_fp32_only",
        "realization_authorities_advance_together",
        "only_mapanything_realization_is_retired",
        "other_realized_records_are_unchanged",
        "old_fp16_bytes_are_preserved_and_verified",
    }
    if (
        not isinstance(semantic_checks, Mapping)
        or set(semantic_checks) != expected_checks
        or any(value is not True for value in semantic_checks.values())
    ):
        raise ValueError("MapAnything transition semantic checks drifted")
    plan = _mapanything_transition_plan(
        old_realization_hash=old_realization_hash,
        new_realization_hash=new_realization_hash,
        updated_before=str(realization_record["updated_at_utc_before"]),
        updated_after=str(realization_record["updated_at_utc_after"]),
        inventory_before=list(before_ids),
        inventory_after=list(after_ids),
        retired_artifact=retired_artifact,
        manifest_diff=manifest_diff,
        source_contract_diff=source_diff,
    )
    if _sha256_bytes(_transition_canonical_json(plan)) != _require_sha256(
        evidence.get("plan_sha256"), "MapAnything transition plan hash"
    ):
        raise ValueError("MapAnything transition plan hash drifted")
    return {
        "edge_kind": "mapanything_authority_transition",
        "edge_uid": f"mapanything:{transaction_id}",
        "transaction_id": transaction_id,
        "evidence_sha256": _sha256_bytes(evidence_raw),
        "manifest_old_hash": MAPANYTHING_OLD_MANIFEST_SHA256,
        "manifest_new_hash": MAPANYTHING_NEW_MANIFEST_SHA256,
        "source_old_hash": MAPANYTHING_OLD_SOURCE_CONTRACTS_SHA256,
        "source_new_hash": MAPANYTHING_NEW_SOURCE_CONTRACTS_SHA256,
        "old_manifest_document": old_manifest,
        "new_manifest_document": new_manifest,
        "old_source_document": old_source,
        "new_source_document": new_source,
        "realized_ids_before": frozenset(before_ids),
        "realized_ids_after": frozenset(after_ids),
        "retired_ids": frozenset({MAPANYTHING_ARTIFACT_ID}),
        "old_realization_hash": old_realization_hash,
        "new_realization_hash": new_realization_hash,
        "updated_at_before": before_time,
        "updated_at_after": after_time,
        "prepared_at": prepared_at,
        "committed_at": committed_at,
        "changed_contracts": frozenset({MAPANYTHING_CONTRACT_NAME}),
        "mapped_artifacts": frozenset({MAPANYTHING_ARTIFACT_ID}),
    }


def _load_mapanything_transition_edges(
    artifact_root: Path,
) -> tuple[dict[str, Any], ...]:
    evidence_root = artifact_root / MAPANYTHING_TRANSITION_ROOT
    if not evidence_root.exists() and not evidence_root.is_symlink():
        return ()
    _require_private_directory(evidence_root, "MapAnything transition evidence root")
    entries = list(evidence_root.iterdir())
    if len(entries) > MAPANYTHING_TRANSITION_MAX_TRANSACTIONS:
        raise ValueError("MapAnything transition evidence exceeds safety bound")
    edges: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda value: value.name):
        if entry.is_symlink() or not entry.is_dir():
            raise ValueError("MapAnything transition evidence inventory is unsafe")
        edge = _mapanything_transition_committed_edge(
            artifact_root=artifact_root,
            transaction_dir=entry,
        )
        if edge is not None:
            edges.append(edge)
    return tuple(edges)


def _edge_realized_ids_before(edge: Mapping[str, Any]) -> frozenset[str]:
    value = edge.get("realized_ids_before", edge.get("realized_ids", frozenset()))
    return frozenset(str(item) for item in value)


def _edge_realized_ids_after(edge: Mapping[str, Any]) -> frozenset[str]:
    value = edge.get("realized_ids_after", edge.get("realized_ids", frozenset()))
    return frozenset(str(item) for item in value)


def _manifest_rebase_committed_edge(
    *,
    artifact_root: Path,
    evidence_root: Path,
    transaction_dir: Path,
    known_source_contract_hashes: set[str],
    current_realized_ids: set[str],
) -> dict[str, Any] | None:
    _require_private_directory(transaction_dir, "manifest-rebase transaction")
    inventory = list(transaction_dir.iterdir())
    if len(inventory) != 1 or inventory[0].name != MANIFEST_REBASE_EVIDENCE_FILENAME:
        raise ValueError(
            f"manifest-rebase transaction inventory is ambiguous: {transaction_dir}"
        )
    evidence, evidence_raw = _load_private_json_with_bytes(
        inventory[0], "manifest-rebase evidence"
    )
    if (
        evidence.get("schema_version") != 1
        or evidence.get("contract") != MANIFEST_REBASE_CONTRACT
    ):
        raise ValueError("manifest-rebase evidence contract is invalid")
    transaction_id = str(evidence.get("transaction_id") or "")
    if (
        transaction_id != transaction_dir.name
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,127}", transaction_id) is None
    ):
        raise ValueError("manifest-rebase transaction ID is invalid")
    state = str(evidence.get("state") or "")
    if state in {"aborted_before_commit", "rolled_back_after_evidence_failure"}:
        return None
    if state != "committed":
        raise ValueError("unresolved manifest-rebase evidence requires recovery")
    allowed_keys = (
        _MANIFEST_REBASE_REQUIRED_EVIDENCE_KEYS
        | _MANIFEST_REBASE_OPTIONAL_EVIDENCE_KEYS
    )
    if (
        not _MANIFEST_REBASE_REQUIRED_EVIDENCE_KEYS.issubset(evidence)
        or set(evidence) - allowed_keys
    ):
        raise ValueError("committed manifest-rebase evidence fields drifted")
    prepared_at = _parse_explicit_utc(
        evidence.get("prepared_at_utc"), "manifest-rebase prepared_at_utc"
    )
    committed_at = _parse_explicit_utc(
        evidence.get("committed_at_utc"), "manifest-rebase committed_at_utc"
    )
    if committed_at < prepared_at:
        raise ValueError("manifest-rebase commit predates preparation")
    if "realization_replace_recovery" in evidence:
        _validate_replace_recovery(
            evidence["realization_replace_recovery"],
            "manifest-rebase replace recovery",
        )

    old_record = _require_exact_record(
        evidence.get("old_manifest"),
        {"path", "sha256"},
        "old manifest evidence",
    )
    new_record = _require_exact_record(
        evidence.get("new_manifest"),
        {"path", "sha256"},
        "new manifest evidence",
    )
    if new_record.get("path") != BASE_MANIFEST_PATH:
        raise ValueError("new manifest evidence path drifted")
    old_hash = _require_sha256(old_record.get("sha256"), "old manifest hash")
    new_hash = _require_sha256(new_record.get("sha256"), "new manifest hash")
    if old_hash == new_hash:
        raise ValueError("manifest-rebase edge does not advance its authority")
    old_document, _old_raw, observed_old_hash = _load_manifest_snapshot(
        artifact_root,
        evidence_root,
        old_record,
        label="old manifest snapshot",
    )
    if observed_old_hash != old_hash:
        raise ValueError("old manifest snapshot hash drifted")

    source_record = _require_exact_record(
        evidence.get("source_contracts"),
        {"path", "sha256"},
        "manifest-rebase source-contract binding",
    )
    source_hash = _require_sha256(
        source_record.get("sha256"), "manifest-rebase source-contract hash"
    )
    if (
        source_record.get("path") != SOURCE_CONTRACT_PATH
        or source_hash not in known_source_contract_hashes
    ):
        raise ValueError("manifest-rebase source-contract binding is not authoritative")

    realization_record = _require_exact_record(
        evidence.get("realization"),
        {
            "path",
            "old_sha256",
            "new_sha256",
            "updated_at_utc_before",
            "updated_at_utc_after",
        },
        "manifest-rebase realization binding",
    )
    if realization_record.get("path") != REALIZATION_FILENAME:
        raise ValueError("manifest-rebase realization path drifted")
    old_realization_hash = _require_sha256(
        realization_record.get("old_sha256"), "manifest-rebase old realization hash"
    )
    new_realization_hash = _require_sha256(
        realization_record.get("new_sha256"), "manifest-rebase new realization hash"
    )
    if old_realization_hash == new_realization_hash:
        raise ValueError("manifest-rebase realization CAS did not advance")
    before_time = _parse_explicit_utc(
        realization_record.get("updated_at_utc_before"),
        "manifest-rebase prior realization timestamp",
    )
    after_time = _parse_explicit_utc(
        realization_record.get("updated_at_utc_after"),
        "manifest-rebase proposed realization timestamp",
    )
    if after_time <= before_time:
        raise ValueError("manifest-rebase realization timestamp did not advance")

    realized = _require_sorted_unique_strings(
        evidence.get("realized_engine_ids"),
        "manifest-rebase realized engines",
        allow_empty=True,
    )
    if not set(realized) <= current_realized_ids:
        raise ValueError("manifest-rebase names unknown current realized engines")
    if evidence.get("mutation_paths") != [
        "base_manifest.sha256",
        "updated_at_utc",
    ]:
        raise ValueError("manifest-rebase mutation paths drifted")
    semantic_checks = evidence.get("semantic_checks")
    allowed_semantic_key_sets = {
        frozenset(_MANIFEST_REBASE_REQUIRED_SEMANTIC_CHECK_KEYS),
        frozenset(
            _MANIFEST_REBASE_REQUIRED_SEMANTIC_CHECK_KEYS
            | {
                "runtime_fields_outside_image_unchanged",
                "runtime_image_change_is_exact_reviewed_authority",
            }
        ),
    }
    if (
        not isinstance(semantic_checks, Mapping)
        or frozenset(semantic_checks) not in allowed_semantic_key_sets
        or any(value is not True for value in semantic_checks.values())
    ):
        raise ValueError("manifest-rebase semantic checks are incomplete")
    manifest_diff = _require_exact_record(
        evidence.get("manifest_diff"),
        {
            "changed_artifact_ids",
            "changed_non_engine_artifact_ids",
            "changed_path_count",
            "changed_paths",
        },
        "manifest-rebase diff summary",
    )
    changed_ids = _require_sorted_unique_strings(
        manifest_diff.get("changed_artifact_ids"),
        "manifest-rebase changed artifact IDs",
        allow_empty=True,
    )
    changed_non_engine_ids = _require_sorted_unique_strings(
        manifest_diff.get("changed_non_engine_artifact_ids"),
        "manifest-rebase changed non-engine artifact IDs",
        allow_empty=True,
    )
    changed_paths = _require_sorted_unique_strings(
        manifest_diff.get("changed_paths"),
        "manifest-rebase changed paths",
        allow_empty=True,
    )
    if (
        not set(changed_non_engine_ids) <= set(changed_ids)
        or manifest_diff.get("changed_path_count") != len(changed_paths)
        or set(changed_ids) & set(ENGINE_NAME_BY_ARTIFACT_ID)
    ):
        raise ValueError("manifest-rebase diff summary is inconsistent")
    runtime_image_authority = evidence.get("runtime_image_authority")
    if runtime_image_authority is not None and not isinstance(
        runtime_image_authority, Mapping
    ):
        raise ValueError("manifest-rebase runtime-image evidence is malformed")

    return {
        "edge_kind": "manifest_rebase",
        "edge_uid": f"manifest:{transaction_id}",
        "transaction_id": transaction_id,
        "evidence_sha256": _sha256_bytes(evidence_raw),
        "old_hash": old_hash,
        "new_hash": new_hash,
        "old_document": old_document,
        "source_contracts_hash": source_hash,
        "realized_ids": frozenset(realized),
        "realized_ids_before": frozenset(realized),
        "realized_ids_after": frozenset(realized),
        "retired_ids": frozenset(),
        "old_realization_hash": old_realization_hash,
        "new_realization_hash": new_realization_hash,
        "updated_at_before": before_time,
        "updated_at_after": after_time,
        "prepared_at": prepared_at,
        "committed_at": committed_at,
        "semantic_checks": dict(semantic_checks),
        "manifest_diff": dict(manifest_diff),
        "runtime_image_authority": (
            dict(runtime_image_authority)
            if isinstance(runtime_image_authority, Mapping)
            else None
        ),
    }


def _validate_manifest_rebase_chain(
    *,
    artifact_root: Path,
    current_manifest_path: Path,
    current_manifest_sha256: str,
    current_manifest_document: Mapping[str, Any] | None = None,
    current_realized_ids: set[str],
    known_source_contract_hashes: set[str],
    required_manifest_hashes: set[str],
    combined_transition_edges: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    current_hash = _require_sha256(
        current_manifest_sha256, "current base-manifest authority hash"
    )
    if current_manifest_document is None:
        current_raw = _read_regular_owned_bytes(
            current_manifest_path, "current base-manifest authority"
        )
        if _sha256_bytes(current_raw) != current_hash:
            raise ValueError(
                "current base-manifest authority changed during validation"
            )
        current_document = _parse_yaml_mapping(
            current_raw, "current base-manifest authority"
        )
    else:
        current_document = current_manifest_document
    _manifest_engine_records(current_document, "current base-manifest authority")

    combined = tuple(
        combined_transition_edges
        if combined_transition_edges is not None
        else _load_mapanything_transition_edges(artifact_root)
    )
    known_realized_ids = set(current_realized_ids)
    for transition_edge in combined:
        known_realized_ids.update(_edge_realized_ids_before(transition_edge))
        known_realized_ids.update(_edge_realized_ids_after(transition_edge))
    edges: list[dict[str, Any]] = []
    evidence_root = artifact_root / MANIFEST_REBASE_ROOT
    if not evidence_root.exists() and not evidence_root.is_symlink():
        raise ValueError("manifest-rebase evidence root is missing")
    if evidence_root.exists() or evidence_root.is_symlink():
        _require_owned_nonwritable_directory(
            evidence_root, "manifest-rebase evidence root"
        )
        entries = list(evidence_root.iterdir())
        if len(entries) > MANIFEST_REBASE_MAX_TRANSACTIONS:
            raise ValueError("manifest-rebase evidence inventory exceeds safety bound")
        for entry in sorted(entries, key=lambda value: value.name):
            if entry.is_symlink():
                raise ValueError("manifest-rebase inventory contains a symlink")
            if not entry.is_dir():
                raise ValueError(
                    "manifest-rebase inventory contains an unexpected file"
                )
            evidence_path = entry / MANIFEST_REBASE_EVIDENCE_FILENAME
            if not evidence_path.exists() and not evidence_path.is_symlink():
                _require_owned_nonwritable_directory(
                    entry, "manifest-rebase input container"
                )
                continue
            edge = _manifest_rebase_committed_edge(
                artifact_root=artifact_root,
                evidence_root=evidence_root,
                transaction_dir=entry,
                known_source_contract_hashes=known_source_contract_hashes,
                current_realized_ids=known_realized_ids,
            )
            if edge is not None:
                edges.append(edge)
    for transition_edge in combined:
        edge = dict(transition_edge)
        edge.update(
            {
                "old_hash": transition_edge["manifest_old_hash"],
                "new_hash": transition_edge["manifest_new_hash"],
                "old_document": transition_edge["old_manifest_document"],
                "new_document": transition_edge["new_manifest_document"],
            }
        )
        edges.append(edge)
    if not edges:
        raise ValueError("no committed manifest-rebase chain is available")

    by_old: dict[str, dict[str, Any]] = {}
    by_new: dict[str, dict[str, Any]] = {}
    for edge in edges:
        old_hash = str(edge["old_hash"])
        new_hash = str(edge["new_hash"])
        if old_hash in by_old:
            raise ValueError("manifest-rebase chain branches from one old hash")
        if new_hash in by_new:
            raise ValueError("manifest-rebase chain has ambiguous predecessors")
        by_old[old_hash] = edge
        by_new[new_hash] = edge
    roots = [edge for edge in edges if edge["old_hash"] not in by_new]
    if len(roots) != 1:
        raise ValueError("manifest-rebase chain is cyclic or has ambiguous roots")
    ordered: list[dict[str, Any]] = []
    edge = roots[0]
    visited: set[str] = set()
    while True:
        edge_uid = str(edge.get("edge_uid") or edge["transaction_id"])
        if edge_uid in visited:
            raise ValueError("manifest-rebase chain contains a cycle")
        visited.add(edge_uid)
        ordered.append(edge)
        next_edge = by_old.get(str(edge["new_hash"]))
        if next_edge is None:
            break
        edge = next_edge
    if len(ordered) != len(edges):
        raise ValueError("manifest-rebase chain is disconnected or ambiguous")
    if ordered[-1]["new_hash"] != current_hash:
        raise ValueError("manifest-rebase chain does not reach current authority")

    positions = {str(edge["old_hash"]): index for index, edge in enumerate(ordered)}
    positions[current_hash] = len(ordered)
    if not required_manifest_hashes <= set(positions):
        raise ValueError(
            "required historic manifest binding is absent from the committed chain"
        )
    selected_start = min(positions[value] for value in required_manifest_hashes)
    selected = ordered[selected_start:]

    previous: dict[str, Any] | None = None
    for index, edge in enumerate(ordered):
        next_document = (
            ordered[index + 1]["old_document"]
            if index + 1 < len(ordered)
            else current_document
        )
        if edge.get("edge_kind") == "mapanything_authority_transition":
            if dict(next_document) != dict(edge["new_document"]):
                raise ValueError(
                    "MapAnything transition manifest successor is disconnected"
                )
            _mapanything_transition_manifest_semantics(
                edge["old_document"], next_document
            )
        else:
            old_target, old_engines = _manifest_engine_records(
                edge["old_document"], "old manifest transition authority"
            )
            next_target, next_engines = _manifest_engine_records(
                next_document, "next manifest transition authority"
            )
            if old_target != next_target or old_engines != next_engines:
                raise ValueError(
                    "manifest-rebase transition changed target or TensorRT engine authority"
                )
            expected_diff, expected_runtime_transition = (
                _recompute_manifest_rebase_semantics(
                    edge["old_document"], next_document
                )
            )
            if edge["manifest_diff"] != expected_diff:
                raise ValueError(
                    "manifest-rebase diff evidence differs from exact YAML semantics"
                )
            runtime_checks_present = {
                "runtime_fields_outside_image_unchanged",
                "runtime_image_change_is_exact_reviewed_authority",
            } <= set(edge["semantic_checks"])
            if runtime_checks_present:
                if edge["runtime_image_authority"] != expected_runtime_transition:
                    raise ValueError(
                        "manifest-rebase runtime-image evidence differs from exact YAML semantics"
                    )
            elif (
                edge["runtime_image_authority"] is not None
                or expected_runtime_transition["changed"]
            ):
                raise ValueError(
                    "legacy manifest-rebase evidence cannot authorize runtime-image drift"
                )
        if previous is not None:
            if previous["committed_at"] > edge["prepared_at"]:
                raise ValueError("manifest-rebase chronology is not monotonic")
            if not set(_edge_realized_ids_after(previous)) <= set(
                _edge_realized_ids_before(edge)
            ):
                raise ValueError(
                    "manifest-rebase realized-engine binding is not monotonic"
                )
        previous = edge
    return {
        "accepted_manifest_hashes": frozenset(
            {current_hash, *(str(edge["old_hash"]) for edge in selected)}
        ),
        "edges": tuple(selected),
    }


def _source_rebase_committed_edge(
    *,
    artifact_root: Path,
    evidence_root: Path,
    transaction_dir: Path,
    current_realized_ids: set[str],
) -> dict[str, Any] | None:
    _require_private_directory(transaction_dir, "source-contract rebase transaction")
    inventory = list(transaction_dir.iterdir())
    if (
        len(inventory) != 1
        or inventory[0].name != SOURCE_CONTRACT_REBASE_EVIDENCE_FILENAME
    ):
        raise ValueError(
            f"source-contract rebase transaction inventory is ambiguous: {transaction_dir}"
        )
    evidence_path = inventory[0]
    evidence, evidence_raw = _load_private_json_with_bytes(
        evidence_path,
        "source-contract rebase evidence",
    )
    if (
        evidence.get("schema_version") != 1
        or evidence.get("contract") != SOURCE_CONTRACT_REBASE_CONTRACT
    ):
        raise ValueError("source-contract rebase evidence contract is invalid")
    transaction_id = str(evidence.get("transaction_id") or "")
    if (
        transaction_id != transaction_dir.name
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,127}", transaction_id) is None
    ):
        raise ValueError("source-contract rebase transaction ID is invalid")
    state = str(evidence.get("state") or "")
    if state in {"aborted_before_commit", "rolled_back_after_evidence_failure"}:
        return None
    if state != "committed":
        raise ValueError("unresolved source-contract rebase evidence requires recovery")
    allowed_keys = (
        _SOURCE_REBASE_REQUIRED_EVIDENCE_KEYS | _SOURCE_REBASE_OPTIONAL_EVIDENCE_KEYS
    )
    if (
        not _SOURCE_REBASE_REQUIRED_EVIDENCE_KEYS.issubset(evidence)
        or set(evidence) - allowed_keys
    ):
        raise ValueError("committed source-contract rebase evidence fields drifted")
    prepared_at = _parse_explicit_utc(
        evidence.get("prepared_at_utc"),
        "source-contract rebase prepared_at_utc",
    )
    committed_at = _parse_explicit_utc(
        evidence.get("committed_at_utc"),
        "source-contract rebase committed_at_utc",
    )
    if committed_at < prepared_at:
        raise ValueError("source-contract rebase commit predates preparation")
    if "realization_replace_recovery" in evidence:
        _validate_replace_recovery(
            evidence["realization_replace_recovery"],
            "source-contract rebase replace recovery",
        )

    old_record = _require_exact_record(
        evidence.get("old_source_contracts"),
        {"path", "sha256"},
        "old source-contract evidence",
    )
    new_record = _require_exact_record(
        evidence.get("new_source_contracts"),
        {"path", "sha256"},
        "new source-contract evidence",
    )
    if new_record.get("path") != SOURCE_CONTRACT_PATH:
        raise ValueError("new source-contract evidence path drifted")
    old_hash = _require_sha256(
        old_record.get("sha256"), "old source-contract evidence hash"
    )
    new_hash = _require_sha256(
        new_record.get("sha256"), "new source-contract evidence hash"
    )
    if old_hash == new_hash:
        raise ValueError("source-contract rebase edge does not advance its authority")
    old_document, _old_raw, observed_old_hash = _load_source_contract_snapshot(
        artifact_root,
        evidence_root,
        old_record,
        label="old source-contract snapshot",
    )
    if observed_old_hash != old_hash:
        raise ValueError("old source-contract snapshot hash drifted")

    base_record = _require_exact_record(
        evidence.get("base_manifest"),
        {"path", "sha256"},
        "source-contract rebase base manifest",
    )
    if base_record.get("path") != BASE_MANIFEST_PATH:
        raise ValueError("source-contract rebase base-manifest path drifted")
    base_hash = _require_sha256(
        base_record.get("sha256"), "source-contract rebase base-manifest hash"
    )
    realization_record = _require_exact_record(
        evidence.get("realization"),
        {
            "path",
            "old_sha256",
            "new_sha256",
            "updated_at_utc_before",
            "updated_at_utc_after",
        },
        "source-contract rebase realization binding",
    )
    if realization_record.get("path") != REALIZATION_FILENAME:
        raise ValueError("source-contract rebase realization path drifted")
    old_realization_hash = _require_sha256(
        realization_record.get("old_sha256"),
        "source-contract rebase old realization hash",
    )
    new_realization_hash = _require_sha256(
        realization_record.get("new_sha256"),
        "source-contract rebase new realization hash",
    )
    if old_realization_hash == new_realization_hash:
        raise ValueError("source-contract rebase realization CAS did not advance")
    before_time = _parse_explicit_utc(
        realization_record.get("updated_at_utc_before"),
        "source-contract rebase prior realization timestamp",
    )
    after_time = _parse_explicit_utc(
        realization_record.get("updated_at_utc_after"),
        "source-contract rebase proposed realization timestamp",
    )
    if after_time <= before_time:
        raise ValueError("source-contract rebase realization timestamp did not advance")

    changed = _require_sorted_unique_strings(
        evidence.get("changed_contracts"),
        "source-contract rebase changed contracts",
        allow_empty=False,
    )
    mapped = _require_sorted_unique_strings(
        evidence.get("mapped_unrealized_artifact_ids"),
        "source-contract rebase mapped artifacts",
        allow_empty=False,
    )
    realized = _require_sorted_unique_strings(
        evidence.get("realized_engine_ids"),
        "source-contract rebase realized engines",
        allow_empty=True,
    )
    expected_contract_names = set(ENGINE_NAME_BY_ARTIFACT_ID.values())
    if not set(changed) <= expected_contract_names:
        raise ValueError("source-contract rebase names an unregistered contract")
    artifact_by_engine = {
        engine_name: artifact_id
        for artifact_id, engine_name in ENGINE_NAME_BY_ARTIFACT_ID.items()
    }
    expected_mapped = {artifact_by_engine[name] for name in changed}
    if set(mapped) != expected_mapped:
        raise ValueError("source-contract rebase mapped artifact evidence drifted")
    if set(realized) & expected_mapped:
        raise ValueError("source-contract rebase changed a realized engine contract")
    if not set(realized) <= current_realized_ids:
        raise ValueError(
            "source-contract rebase realization binding names unknown current engines"
        )
    if evidence.get("mutation_paths") != [
        "source_contracts.sha256",
        "updated_at_utc",
    ]:
        raise ValueError("source-contract rebase mutation paths drifted")
    semantic_checks = evidence.get("semantic_checks")
    if (
        not isinstance(semantic_checks, Mapping)
        or set(semantic_checks) != _SOURCE_REBASE_SEMANTIC_CHECK_KEYS
        or any(
            semantic_checks[key] is not True
            for key in _SOURCE_REBASE_SEMANTIC_CHECK_KEYS
        )
    ):
        raise ValueError("source-contract rebase semantic checks are incomplete")

    return {
        "edge_kind": "source_contract_rebase",
        "edge_uid": f"source:{transaction_id}",
        "transaction_id": transaction_id,
        "evidence_sha256": _sha256_bytes(evidence_raw),
        "old_hash": old_hash,
        "new_hash": new_hash,
        "old_document": old_document,
        "base_manifest_hash": base_hash,
        "changed_contracts": frozenset(str(value) for value in changed),
        "mapped_artifacts": frozenset(str(value) for value in mapped),
        "realized_ids": frozenset(str(value) for value in realized),
        "realized_ids_before": frozenset(str(value) for value in realized),
        "realized_ids_after": frozenset(str(value) for value in realized),
        "retired_ids": frozenset(),
        "old_realization_hash": old_realization_hash,
        "new_realization_hash": new_realization_hash,
        "updated_at_before": before_time,
        "updated_at_after": after_time,
        "prepared_at": prepared_at,
        "committed_at": committed_at,
    }


def _validate_source_contract_rebase_chain(
    *,
    artifact_root: Path,
    current_source_contracts_path: Path,
    current_source_contracts_sha256: str,
    current_source_contracts_document: Mapping[str, Any] | None = None,
    current_manifest_path: Path,
    current_manifest_sha256: str,
    current_manifest_document: Mapping[str, Any] | None = None,
    current_realization: Mapping[str, Any],
    current_realization_sha256: str,
) -> dict[str, frozenset[str]]:
    """Return historic source hashes equivalent for each realized engine."""

    current_hash = _require_sha256(
        current_source_contracts_sha256,
        "current source-contract authority hash",
    )
    if current_source_contracts_document is None:
        current_document, current_raw = _load_regular_json_with_bytes(
            current_source_contracts_path,
            "current source-contract authority",
        )
        if _sha256_bytes(current_raw) != current_hash:
            raise ValueError(
                "current source-contract authority changed during validation"
            )
    else:
        current_document = current_source_contracts_document
    current_contracts = _validate_source_contract_document(
        current_document,
        "current source-contract authority",
    )
    current_base = _require_exact_record(
        current_realization.get("base_manifest"),
        {"path", "sha256"},
        "current realization base manifest",
    )
    if current_base.get("path") != BASE_MANIFEST_PATH:
        raise ValueError("current realization base-manifest path drifted")
    current_base_hash = _require_sha256(
        current_base.get("sha256"), "current realization base-manifest hash"
    )
    if current_base_hash != _require_sha256(
        current_manifest_sha256, "current base-manifest authority hash"
    ):
        raise ValueError("current realization base-manifest binding is wrong")
    current_source_record = _require_exact_record(
        current_realization.get("source_contracts"),
        {"path", "sha256"},
        "current realization source-contract binding",
    )
    if current_source_record != {
        "path": SOURCE_CONTRACT_PATH,
        "sha256": current_hash,
    }:
        raise ValueError("current realization source-contract binding is wrong")
    current_realization_hash = _require_sha256(
        current_realization_sha256, "current realization hash"
    )
    current_realization_updated_at = _parse_explicit_utc(
        current_realization.get("updated_at_utc"),
        "current realization updated_at_utc",
    )
    current_realized_raw = current_realization.get("artifacts")
    if not isinstance(current_realized_raw, Mapping):
        raise ValueError("current realization artifacts must be a mapping")
    current_realized_ids = {str(value) for value in current_realized_raw}

    combined_transition_edges = _load_mapanything_transition_edges(artifact_root)
    known_realized_ids = set(current_realized_ids)
    for transition_edge in combined_transition_edges:
        known_realized_ids.update(_edge_realized_ids_before(transition_edge))
        known_realized_ids.update(_edge_realized_ids_after(transition_edge))
    edges: list[dict[str, Any]] = []
    evidence_root = artifact_root / SOURCE_CONTRACT_REBASE_ROOT
    if not evidence_root.exists() and not evidence_root.is_symlink():
        raise ValueError("source-contract rebase evidence root is missing")
    if evidence_root.exists() or evidence_root.is_symlink():
        _require_private_directory(
            evidence_root, "source-contract rebase evidence root"
        )
        entries = list(evidence_root.iterdir())
        if len(entries) > SOURCE_CONTRACT_REBASE_MAX_TRANSACTIONS + 1:
            raise ValueError(
                "source-contract rebase evidence inventory exceeds safety bound"
            )
        for entry in sorted(entries, key=lambda value: value.name):
            if entry.is_symlink():
                raise ValueError("source-contract rebase inventory contains a symlink")
            if entry.name == SOURCE_CONTRACT_REBASE_INPUT_ROOT:
                _require_private_directory(entry, "source-contract rebase input root")
                continue
            if not entry.is_dir():
                raise ValueError(
                    "source-contract rebase inventory contains an unexpected file"
                )
            edge = _source_rebase_committed_edge(
                artifact_root=artifact_root,
                evidence_root=evidence_root,
                transaction_dir=entry,
                current_realized_ids=known_realized_ids,
            )
            if edge is not None:
                edges.append(edge)
    for transition_edge in combined_transition_edges:
        edge = dict(transition_edge)
        edge.update(
            {
                "old_hash": transition_edge["source_old_hash"],
                "new_hash": transition_edge["source_new_hash"],
                "old_document": transition_edge["old_source_document"],
                "new_document": transition_edge["new_source_document"],
                "base_manifest_hash": transition_edge["manifest_old_hash"],
            }
        )
        edges.append(edge)
    if not edges:
        raise ValueError("no committed source-contract rebase chain is available")

    by_old: dict[str, dict[str, Any]] = {}
    by_new: dict[str, dict[str, Any]] = {}
    for edge in edges:
        old_hash = str(edge["old_hash"])
        new_hash = str(edge["new_hash"])
        if old_hash in by_old:
            raise ValueError("source-contract rebase chain branches from one old hash")
        if new_hash in by_new:
            raise ValueError("source-contract rebase chain has ambiguous predecessors")
        by_old[old_hash] = edge
        by_new[new_hash] = edge
    roots = [edge for edge in edges if edge["old_hash"] not in by_new]
    if len(roots) != 1:
        raise ValueError(
            "source-contract rebase chain is cyclic or has ambiguous roots"
        )
    ordered: list[dict[str, Any]] = []
    edge = roots[0]
    visited: set[str] = set()
    while True:
        edge_uid = str(edge.get("edge_uid") or edge["transaction_id"])
        if edge_uid in visited:
            raise ValueError("source-contract rebase chain contains a cycle")
        visited.add(edge_uid)
        ordered.append(edge)
        next_edge = by_old.get(str(edge["new_hash"]))
        if next_edge is None:
            break
        edge = next_edge
    if len(ordered) != len(edges):
        raise ValueError("source-contract rebase chain is disconnected or ambiguous")
    if ordered[-1]["new_hash"] != current_hash:
        raise ValueError(
            "source-contract rebase chain does not reach current authority"
        )

    known_source_hashes = {
        current_hash,
        *(str(edge["old_hash"]) for edge in ordered),
        *(str(edge["new_hash"]) for edge in ordered),
    }
    source_base_hashes = {str(edge["base_manifest_hash"]) for edge in ordered}
    manifest_chain: dict[str, Any] | None = None
    if source_base_hashes == {current_base_hash}:
        accepted_manifest_hashes = frozenset({current_base_hash})
        manifest_edges: tuple[dict[str, Any], ...] = ()
    else:
        manifest_chain = _validate_manifest_rebase_chain(
            artifact_root=artifact_root,
            current_manifest_path=current_manifest_path,
            current_manifest_sha256=current_base_hash,
            current_manifest_document=current_manifest_document,
            current_realized_ids=current_realized_ids,
            known_source_contract_hashes=known_source_hashes,
            required_manifest_hashes=source_base_hashes,
            combined_transition_edges=combined_transition_edges,
        )
        accepted_manifest_hashes = manifest_chain["accepted_manifest_hashes"]
        manifest_edges = manifest_chain["edges"]
    if not source_base_hashes <= set(accepted_manifest_hashes):
        raise ValueError(
            "source-contract rebase base-manifest binding is not in the proven manifest chain"
        )

    previous: dict[str, Any] | None = None
    for index, edge in enumerate(ordered):
        next_document = (
            ordered[index + 1]["old_document"]
            if index + 1 < len(ordered)
            else current_document
        )
        next_contracts = _validate_source_contract_document(
            next_document,
            "next source-contract transition authority",
        )
        if edge.get("edge_kind") == "mapanything_authority_transition" and dict(
            next_document
        ) != dict(edge["new_document"]):
            raise ValueError(
                "MapAnything transition source-contract successor is disconnected"
            )
        observed_changes = {
            key
            for key in current_contracts
            if edge["old_document"]["contracts"][key] != next_contracts[key]
        }
        if observed_changes != set(edge["changed_contracts"]):
            raise ValueError(
                "source-contract rebase allowlist differs from semantic contract changes"
            )
        if previous is not None:
            if previous["committed_at"] > edge["prepared_at"]:
                raise ValueError("source-contract rebase chronology is not monotonic")
            if not set(_edge_realized_ids_after(previous)) <= set(
                _edge_realized_ids_before(edge)
            ):
                raise ValueError(
                    "source-contract rebase realized-engine binding is not monotonic"
                )
        previous = edge
    realization_transitions: dict[str, str] = {}
    realization_predecessors: dict[str, str] = {}
    transition_edges: dict[str, dict[str, Any]] = {}
    unique_provenance_edges: dict[str, dict[str, Any]] = {}
    for provenance_edge in [*ordered, *manifest_edges]:
        edge_uid = str(
            provenance_edge.get("edge_uid")
            or f"legacy:{provenance_edge['transaction_id']}"
        )
        prior = unique_provenance_edges.get(edge_uid)
        if prior is not None:
            if (
                provenance_edge.get("edge_kind") != "mapanything_authority_transition"
                or prior.get("evidence_sha256")
                != provenance_edge.get("evidence_sha256")
                or prior.get("old_realization_hash")
                != provenance_edge.get("old_realization_hash")
                or prior.get("new_realization_hash")
                != provenance_edge.get("new_realization_hash")
            ):
                raise ValueError(
                    "realization provenance edge is duplicated ambiguously"
                )
            continue
        unique_provenance_edges[edge_uid] = provenance_edge
    for provenance_edge in unique_provenance_edges.values():
        old_realization_hash = str(provenance_edge["old_realization_hash"])
        new_realization_hash = str(provenance_edge["new_realization_hash"])
        if old_realization_hash in realization_transitions:
            raise ValueError("realization provenance graph branches ambiguously")
        if new_realization_hash in realization_predecessors:
            raise ValueError("realization provenance graph has ambiguous predecessors")
        realization_transitions[old_realization_hash] = new_realization_hash
        realization_predecessors[new_realization_hash] = old_realization_hash
        transition_edges[old_realization_hash] = provenance_edge

    realization_roots = set(realization_transitions) - set(realization_predecessors)
    if not realization_roots:
        raise ValueError("realization provenance graph is cyclic or has no root")

    # Engine maintenance may legitimately add or replace independently
    # validated engine records between two source/manifest rebase transactions.
    # Such a maintenance step has no rebase edge because it changes neither
    # authority document. Preserve the same fail-closed rule already used for a
    # newer current realization: each disconnected rebase component must be
    # chronologically ordered and its realized-engine inventory must be a
    # monotonic superset of the preceding component's terminal inventory.
    components: list[list[dict[str, Any]]] = []
    visited_realization_edges: set[str] = set()
    for root_hash in realization_roots:
        component: list[dict[str, Any]] = []
        edge = transition_edges[root_hash]
        while True:
            old_realization_hash = str(edge["old_realization_hash"])
            if old_realization_hash in visited_realization_edges:
                raise ValueError("realization provenance graph contains a cycle")
            visited_realization_edges.add(old_realization_hash)
            component.append(edge)
            following = transition_edges.get(str(edge["new_realization_hash"]))
            if following is None:
                break
            edge = following
        components.append(component)
    if len(visited_realization_edges) != len(transition_edges):
        raise ValueError("realization provenance graph contains a cycle")

    components.sort(key=lambda value: value[0]["updated_at_before"])
    for previous_component, next_component in zip(components, components[1:]):
        previous_terminal = previous_component[-1]
        next_root = next_component[0]
        if previous_terminal["updated_at_after"] >= next_root["updated_at_before"]:
            raise ValueError(
                "realization maintenance gap timestamps are not strictly monotonic"
            )
        if previous_terminal["committed_at"] > next_root["prepared_at"]:
            raise ValueError(
                "realization maintenance gap evidence chronology is not monotonic"
            )
        if not set(_edge_realized_ids_after(previous_terminal)) <= set(
            _edge_realized_ids_before(next_root)
        ):
            raise ValueError(
                "realization maintenance gap drops previously realized engines"
            )

    for component in components:
        for provenance_edge, following_edge in zip(component, component[1:]):
            if (
                _edge_realized_ids_after(provenance_edge)
                != _edge_realized_ids_before(following_edge)
                or provenance_edge["updated_at_after"]
                != following_edge["updated_at_before"]
            ):
                raise ValueError(
                    "realization rebase boundary has inconsistent state or timestamp binding"
                )

    terminal_edge = components[-1][-1]
    terminal_realization_hash = str(terminal_edge["new_realization_hash"])
    terminal_realized_ids = set(_edge_realized_ids_after(terminal_edge))
    terminal_updated_at = terminal_edge["updated_at_after"]
    if terminal_realization_hash == current_realization_hash:
        if (
            terminal_realized_ids != current_realized_ids
            or terminal_updated_at != current_realization_updated_at
        ):
            raise ValueError(
                "current realization state is not exactly bound to rebase evidence"
            )
    elif (
        not terminal_realized_ids <= current_realized_ids
        or terminal_updated_at >= current_realization_updated_at
    ):
        raise ValueError(
            "current realization does not monotonically succeed rebase evidence"
        )

    accepted: dict[str, set[str]] = {
        artifact_id: {current_hash} for artifact_id in current_realized_ids
    }
    equivalent_suffix = {artifact_id: True for artifact_id in current_realized_ids}
    for index in range(len(ordered) - 1, -1, -1):
        edge = ordered[index]
        next_document = (
            ordered[index + 1]["old_document"]
            if index + 1 < len(ordered)
            else current_document
        )
        old_contracts = edge["old_document"]["contracts"]
        next_contracts = next_document["contracts"]
        for artifact_id in current_realized_ids:
            engine_name = ENGINE_NAME_BY_ARTIFACT_ID.get(artifact_id)
            if engine_name is None:
                continue
            unchanged = (
                engine_name not in edge["changed_contracts"]
                and artifact_id not in edge["mapped_artifacts"]
                and old_contracts[engine_name] == next_contracts[engine_name]
            )
            equivalent_suffix[artifact_id] = (
                equivalent_suffix[artifact_id] and unchanged
            )
            if equivalent_suffix[artifact_id]:
                accepted[artifact_id].add(str(edge["old_hash"]))
    return {artifact_id: frozenset(hashes) for artifact_id, hashes in accepted.items()}


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
    expected_guard_mib = gpu_sampler.reviewed_guard_mib(expected_engine)
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
        "contract": gpu_sampler.CONTRACT,
        "engine": expected_engine,
        "transaction_id": transaction_id,
        "prepared_transaction_sha256": prepared_sha256,
        "artifact_root_id": artifact_root_id,
        "device_index": 0,
        "gpu_uuid": expected_uuid,
        "guard_mib": expected_guard_mib,
        "sample_interval_ms": gpu_sampler.REVIEWED_SAMPLE_INTERVAL_MS,
        "maximum_gap_limit_ms": gpu_sampler.REVIEWED_MAX_GAP_MS,
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
        interval_ms=gpu_sampler.REVIEWED_SAMPLE_INTERVAL_MS,
        max_gap_ms=gpu_sampler.REVIEWED_MAX_GAP_MS,
        parent_pid=wrapper_pid,
        parent_start_time_ticks=wrapper_start,
        allow_active=False,
    )
    try:
        reconstructed = gpu_sampler.summarize(summary_args)
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
    accepted_source_contract_hashes: frozenset[str] | None = None,
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
    accepted_hashes = accepted_source_contract_hashes or frozenset(
        {authoritative_source_contracts_sha256}
    )
    if (
        authoritative_source_contracts_sha256 not in accepted_hashes
        or maintenance_source_contracts_sha256 not in accepted_hashes
    ):
        raise ValueError(
            "maintenance source-contract input digest is not authoritative"
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
    build_image = target.get("build_image")
    if not isinstance(build_image, Mapping):
        raise ValueError("target build-image authority is missing")
    exact_platform = {
        "image": build_image.get("reference"),
        "image_id": build_image.get("image_id"),
        "base_digest": build_image.get("base_digest"),
        "tensorrt_version": build_image.get("tensorrt_version"),
        "cuda_version": build_image.get("cuda_version"),
    }
    for key, expected in exact_platform.items():
        if platform.get(key) != expected:
            raise ValueError(
                f"maintenance platform {key} differs from build-image authority"
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
            or row.get("proof") != proof
            or not isinstance(row.get("command"), list)
            or not row.get("command")
        ):
            raise ValueError(
                f"maintenance command did not pass its proof gate: {label}"
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
    if not guard_present:
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
        base_authority_document = copy.deepcopy(base_payload)
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
        accepted_source_contract_hashes_by_artifact: (
            dict[str, frozenset[str]] | None
        ) = None
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
                    and accepted_source_contract_hashes_by_artifact is None
                ):
                    accepted_source_contract_hashes_by_artifact = (
                        _validate_source_contract_rebase_chain(
                            artifact_root=root,
                            current_source_contracts_path=SOURCE_CONTRACTS,
                            current_source_contracts_sha256=(
                                current_source_contracts_sha256
                            ),
                            current_source_contracts_document=(
                                source_contract_document
                            ),
                            current_manifest_path=base_path,
                            current_manifest_sha256=expected_base["sha256"],
                            current_manifest_document=base_authority_document,
                            current_realization=realization,
                            current_realization_sha256=realization_sha256,
                        )
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
                    accepted_source_contract_hashes=(
                        accepted_source_contract_hashes_by_artifact.get(artifact_id)
                        if accepted_source_contract_hashes_by_artifact is not None
                        else None
                    ),
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
