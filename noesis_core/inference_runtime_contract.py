"""Fail-closed runtime materialization for TensorRT-backed DeepStream models.

The reviewed model/config files may retain ONNX, TAO, or other source inputs
for explicit offline maintenance.  Production ``nvinfer`` and ``nvtracker``
must receive derived configs that can only deserialize the selected engine.
"""

from __future__ import annotations

import configparser
import hashlib
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping

import yaml


class EngineOnlyRuntimeError(ValueError):
    """Raised when an engine-only production config cannot be proven."""


_NVINFER_BUILD_KEYS = frozenset(
    {
        "custom-network-config",
        "engine-create-func-name",
        "int8-calib-file",
        "model-file",
        "onnx-file",
        "proto-file",
        "tlt-encoded-model",
        "tlt-model-key",
        "uff-file",
        "uff-input-blob-name",
        "uff-input-dims",
        "uff-input-order",
    }
)
_NVINFER_RUNTIME_FILE_KEYS = frozenset(
    {
        "custom-lib-path",
        "labelfile-path",
        "mean-file",
    }
)
_FORBIDDEN_CUSTOM_BUILDER_SYMBOLS = frozenset(
    {
        "NvDsInferCreateModelParser",
        "NvDsInferCudaEngineGet",
    }
)
_NVTRACKER_BUILD_KEYS = frozenset(
    {
        "calibrationTableFile",
        "onnxFile",
        "tltEncodedModel",
        "tltModelKey",
        "uffFile",
    }
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _require_nonempty_file(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve(strict=False)
    try:
        valid = candidate.is_file() and candidate.stat().st_size > 0
    except OSError:
        valid = False
    if not valid:
        raise EngineOnlyRuntimeError(f"{label} is missing or empty: {candidate}")
    return candidate


def _safe_name(value: str, *, fallback: str) -> str:
    normalized = _SAFE_NAME_RE.sub("-", str(value or "").strip()).strip("-.")
    return normalized or fallback


def _resolve_reference(raw: Any, *, owner: Path, repo_root: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path("")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if candidate.parts:
        first = candidate.parts[0]
        if first == "DS9":
            return (repo_root / candidate).resolve(strict=False)
        if first in {
            "build",
            "config",
            "models",
            "native",
            "pipelines",
            "testpipelines",
        }:
            ds9_root = (repo_root / "DS9").resolve(strict=False)
            try:
                owner.resolve(strict=False).relative_to(ds9_root)
            except ValueError:
                stack_root = repo_root
            else:
                stack_root = ds9_root
            return (stack_root / candidate).resolve(strict=False)
    return (owner.parent / candidate).resolve(strict=False)


def _atomic_write_text(path: Path, text: str) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-{threading.get_ident()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(temporary, flags, 0o644)
        try:
            payload = text.encode("utf-8")
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _derived_destination(
    *,
    output_root: Path,
    family: str,
    component_name: str,
    basename: str,
    rendered: str,
) -> Path:
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
    return (
        Path(output_root).expanduser().resolve(strict=False)
        / "runtime_inference"
        / _safe_name(family, fallback="engine")
        / _safe_name(component_name, fallback="component")
        / digest
        / _safe_name(basename, fallback="runtime.conf")
    )


def _assert_parser_only_library(path: Path) -> None:
    library = _require_nonempty_file(path, label="nvinfer custom library")
    nm = shutil.which("nm")
    if nm is None:
        raise EngineOnlyRuntimeError(
            "nm is required to prove that an nvinfer custom library cannot build engines"
        )
    process = subprocess.run(
        [nm, "-D", "--defined-only", str(library)],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "symbol inspection failed").strip()
        raise EngineOnlyRuntimeError(
            f"unable to inspect nvinfer custom library symbols: {library}: {detail}"
        )
    exported = {
        line.rsplit(maxsplit=1)[-1]
        for line in process.stdout.splitlines()
        if line.strip()
    }
    forbidden = sorted(exported & _FORBIDDEN_CUSTOM_BUILDER_SYMBOLS)
    if forbidden:
        raise EngineOnlyRuntimeError(
            "production nvinfer custom library exports engine-building symbol(s): "
            f"{library}: {', '.join(forbidden)}"
        )


def render_nvinfer_engine_only_config(
    *,
    source_config: Path,
    engine_path: Path,
    repo_root: Path,
) -> str:
    """Render the exact production ``nvinfer`` INI without publishing it."""

    source = _require_nonempty_file(source_config, label="nvinfer source config")
    engine = _require_nonempty_file(engine_path, label="TensorRT engine")
    root = Path(repo_root).expanduser().resolve(strict=False)

    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#",),
        strict=False,
    )
    parser.optionxform = str
    try:
        with source.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except Exception as exc:
        raise EngineOnlyRuntimeError(
            f"unable to parse nvinfer source config {source}: {exc}"
        ) from exc
    if not parser.has_section("property"):
        raise EngineOnlyRuntimeError(
            f"nvinfer source config has no [property] section: {source}"
        )

    properties = parser["property"]
    for key in list(properties):
        normalized = str(key).strip().lower()
        if normalized in _NVINFER_BUILD_KEYS or normalized == "model-engine-file":
            del properties[key]

    properties["model-engine-file"] = str(engine)
    for key in list(properties):
        normalized = str(key).strip().lower()
        if normalized not in _NVINFER_RUNTIME_FILE_KEYS:
            continue
        resolved = _resolve_reference(properties[key], owner=source, repo_root=root)
        resolved = _require_nonempty_file(
            resolved,
            label=f"nvinfer {normalized}",
        )
        properties[key] = str(resolved)
        if normalized == "custom-lib-path":
            _assert_parser_only_library(resolved)

    from io import StringIO

    buffer = StringIO()
    parser.write(buffer, space_around_delimiters=False)
    rendered = buffer.getvalue()

    verification = configparser.ConfigParser(interpolation=None, strict=False)
    verification.read_string(rendered)
    output_properties: Mapping[str, str] = verification["property"]
    leaked = sorted(
        key
        for key in output_properties
        if str(key).strip().lower() in _NVINFER_BUILD_KEYS
    )
    if leaked:
        raise EngineOnlyRuntimeError(
            f"derived nvinfer config retained engine build inputs: {', '.join(leaked)}"
        )
    if Path(output_properties.get("model-engine-file", "")).resolve(strict=False) != engine:
        raise EngineOnlyRuntimeError(
            "derived nvinfer config does not select the requested TensorRT engine"
        )

    return rendered


def materialize_nvinfer_engine_only_config(
    *,
    source_config: Path,
    engine_path: Path,
    output_root: Path,
    component_name: str,
    repo_root: Path,
) -> Path:
    """Derive an atomic ``nvinfer`` INI containing no engine build inputs."""

    source = _require_nonempty_file(source_config, label="nvinfer source config")
    rendered = render_nvinfer_engine_only_config(
        source_config=source,
        engine_path=engine_path,
        repo_root=repo_root,
    )

    destination = _derived_destination(
        output_root=output_root,
        family="nvinfer",
        component_name=component_name,
        basename=source.name,
        rendered=rendered,
    )
    return _atomic_write_text(destination, rendered)


def _materialize_tracker_node(
    value: Any,
    *,
    owner: Path,
    repo_root: Path,
    label: str,
) -> Any:
    if isinstance(value, list):
        return [
            _materialize_tracker_node(
                item,
                owner=owner,
                repo_root=repo_root,
                label=f"{label}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value

    result = dict(value)
    build_inputs = sorted(key for key in result if key in _NVTRACKER_BUILD_KEYS)
    has_engine = "modelEngineFile" in result
    if build_inputs and not has_engine:
        raise EngineOnlyRuntimeError(
            f"{label} has tracker model source inputs without modelEngineFile: "
            f"{', '.join(build_inputs)}"
        )
    if has_engine:
        engine = _resolve_reference(
            result.get("modelEngineFile"),
            owner=owner,
            repo_root=repo_root,
        )
        engine = _require_nonempty_file(engine, label=f"{label}.modelEngineFile")
        result["modelEngineFile"] = str(engine)
    for key in _NVTRACKER_BUILD_KEYS:
        result.pop(key, None)

    camera_models = result.get("cameraModelFilepath")
    if isinstance(camera_models, list):
        resolved_models: list[str] = []
        for index, raw in enumerate(camera_models):
            resolved = _resolve_reference(raw, owner=owner, repo_root=repo_root)
            resolved = _require_nonempty_file(
                resolved,
                label=f"{label}.cameraModelFilepath[{index}]",
            )
            resolved_models.append(str(resolved))
        result["cameraModelFilepath"] = resolved_models

    for key, item in list(result.items()):
        result[key] = _materialize_tracker_node(
            item,
            owner=owner,
            repo_root=repo_root,
            label=f"{label}.{key}",
        )
    return result


def materialize_nvtracker_engine_only_config(
    *,
    source_config: Path,
    output_root: Path,
    component_name: str,
    repo_root: Path,
) -> Path:
    """Derive an atomic NvMOT YAML with engine files but no model sources."""

    source = _require_nonempty_file(source_config, label="nvtracker source config")
    root = Path(repo_root).expanduser().resolve(strict=False)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise EngineOnlyRuntimeError(
            f"unable to parse nvtracker source config {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EngineOnlyRuntimeError(
            f"nvtracker source config must be a mapping: {source}"
        )
    derived = _materialize_tracker_node(
        payload,
        owner=source,
        repo_root=root,
        label="nvtracker",
    )
    rendered = yaml.safe_dump(derived, sort_keys=False)
    for forbidden in _NVTRACKER_BUILD_KEYS:
        if re.search(rf"(?m)^\s*{re.escape(forbidden)}\s*:", rendered):
            raise EngineOnlyRuntimeError(
                f"derived nvtracker config retained engine build input: {forbidden}"
            )
    destination = _derived_destination(
        output_root=output_root,
        family="nvtracker",
        component_name=component_name,
        basename=source.name,
        rendered=rendered,
    )
    return _atomic_write_text(destination, rendered)


__all__ = [
    "EngineOnlyRuntimeError",
    "materialize_nvinfer_engine_only_config",
    "materialize_nvtracker_engine_only_config",
    "render_nvinfer_engine_only_config",
]
