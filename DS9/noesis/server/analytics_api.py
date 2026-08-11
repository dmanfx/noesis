from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import copy
import configparser
import hashlib
import io
import logging
import math
import os
import re
import stat
import tempfile
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, validator

from noesis.server.depth_api import ensure_pipeline_ready  # noqa: F401 - re-export for dependency parity
from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    measure_rest_response_model,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Noesis DS8 Analytics API")
app.router.route_class = BoundaryMetricsRoute

ANALYTICS_CONFIG_ENV = "NOESIS_ANALYTICS_CONFIG"
DEFAULT_ANALYTICS_CONFIG = Path("config/nvdsanalytics.yaml")
DEFAULT_EXCLUDE_CONFIG = Path("config/config_nvdsanalytics_exclude.ini")

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None
_RELOAD_HOOK: Optional[
    Callable[[str, Dict[str, Any], Mapping[str, Any]], Mapping[str, Any]]
] = None
_RELOAD_COUNTER: int = 0
_STATE_POISONED: Optional[str] = None
_POISON_HOOK: Optional[Callable[[str], None]] = None

_SAFE_ROI_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CANONICAL_STREAM_ID = re.compile(r"^(?:0|[1-9][0-9]{0,5})$")
ANALYTICS_YAML_MAX_BYTES = 4 * 1024 * 1024
ANALYTICS_EXCLUDE_INI_MAX_BYTES = 1024 * 1024


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _resolve_analytics_config() -> Path:
    """Return the analytics config path, preferring an env override."""
    env_path = os.environ.get(ANALYTICS_CONFIG_ENV)
    path = Path(env_path).expanduser() if env_path else DEFAULT_ANALYTICS_CONFIG
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalize_stream_keys(config: Dict[str, Any]) -> None:
    analytics = config.get("analytics")
    if not isinstance(analytics, dict):
        return
    stages = analytics.get("stages")
    if not isinstance(stages, dict):
        return

    for stage_name, stage_cfg in stages.items():
        if not isinstance(stage_cfg, dict):
            continue
        streams = stage_cfg.get("streams")
        if not isinstance(streams, dict):
            continue

        normalized: Dict[str, Any] = {}
        for key, value in streams.items():
            key_str = str(key)
            if key_str in normalized:
                raise ValueError(
                    f"Analytics stage '{stage_name}' contains duplicate stream key '{key_str}'"
                )
            if not _CANONICAL_STREAM_ID.fullmatch(key_str):
                raise ValueError(
                    f"Analytics stage '{stage_name}' has unsafe stream id '{key_str}'"
                )
            normalized[key_str] = value
        stage_cfg["streams"] = normalized


def _load_config(force: bool = False) -> Dict[str, Any]:
    """Load the nvdsanalytics configuration, caching for subsequent requests."""
    global _CONFIG_CACHE, _CONFIG_PATH

    if _STATE_POISONED is not None:
        raise RuntimeError(f"Analytics state is poisoned: {_STATE_POISONED}")

    path = _resolve_analytics_config()
    if not force and _CONFIG_CACHE is not None and _CONFIG_PATH == path:
        return copy.deepcopy(_CONFIG_CACHE)

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=_UniqueKeyLoader) or {}

    _normalize_stream_keys(config)

    _CONFIG_CACHE = copy.deepcopy(config)
    _CONFIG_PATH = path
    return config


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace one regular, non-symlink path with fsynced UTF-8 text."""
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    cursor = Path(target.anchor)
    for part in target.parts[1:]:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Analytics state path contains a symlink: {cursor}")
    parent_info = target.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError(f"Analytics state parent is not a directory: {target.parent}")

    original_identity: Optional[Tuple[int, int]] = None
    try:
        target_info = target.lstat()
    except FileNotFoundError:
        mode = 0o600
    else:
        if not stat.S_ISREG(target_info.st_mode) or target_info.st_nlink != 1:
            raise ValueError(f"Analytics state target must be a single-link regular file: {target}")
        mode = stat.S_IMODE(target_info.st_mode)
        original_identity = (int(target_info.st_dev), int(target_info.st_ino))

    fd = -1
    temporary: Optional[Path] = None
    try:
        fd, raw_temporary = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        temporary = Path(raw_temporary)
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            fd = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if original_identity is None:
            if target.exists() or target.is_symlink():
                raise RuntimeError(f"Analytics state target appeared during write: {target}")
        else:
            current = target.lstat()
            if (int(current.st_dev), int(current.st_ino)) != original_identity:
                raise RuntimeError(f"Analytics state target changed during write: {target}")
        os.replace(temporary, target)
        temporary = None
        directory_fd = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _snapshot_file(
    path: Path,
    *,
    max_bytes: int = ANALYTICS_YAML_MAX_BYTES,
) -> Optional[bytes]:
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        expected = target.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1:
        raise ValueError(f"Analytics snapshot target must be a single-link regular file: {target}")
    if expected.st_size > max_bytes:
        raise ValueError(f"Analytics snapshot exceeds {max_bytes} bytes: {target}")
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError(f"Analytics snapshot target changed while opening: {target}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > max_bytes:
        raise ValueError(f"Analytics snapshot exceeds {max_bytes} bytes: {target}")
    return payload


def _restore_file(path: Path, snapshot: Optional[bytes]) -> None:
    if snapshot is None:
        path.unlink(missing_ok=True)
        return
    _atomic_write_text(path, snapshot.decode("utf-8"))


def _rollback_files(
    files: List[Tuple[str, Path, Optional[bytes]]],
) -> List[str]:
    errors: List[str] = []
    for label, path, snapshot in files:
        try:
            _restore_file(path, snapshot)
            if _snapshot_file(path) != snapshot:
                raise RuntimeError("read-back verification mismatch")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    return errors


def _rollback_transaction(
    files: List[Tuple[str, Path, Optional[bytes]]],
    cache_snapshot: Optional[Dict[str, Any]],
    cache_path_snapshot: Optional[Path],
) -> List[str]:
    global _CONFIG_CACHE, _CONFIG_PATH
    errors = _rollback_files(files)
    if errors:
        _poison_state("analytics rollback failed: " + "; ".join(errors))
    else:
        _CONFIG_CACHE = copy.deepcopy(cache_snapshot)
        _CONFIG_PATH = cache_path_snapshot
    return errors


def _poison_state(reason: str) -> None:
    """Permanently fail this process closed after state/native divergence."""
    global _CONFIG_CACHE, _CONFIG_PATH, _STATE_POISONED
    normalized = str(reason).strip() or "unspecified analytics state divergence"
    if _STATE_POISONED and normalized not in _STATE_POISONED:
        normalized = f"{_STATE_POISONED}; {normalized}"
    _CONFIG_CACHE = None
    _CONFIG_PATH = None
    _STATE_POISONED = normalized
    logger.critical("Analytics state poisoned: %s", _STATE_POISONED)
    if _POISON_HOOK is not None:
        try:
            _POISON_HOOK(_STATE_POISONED)
        except Exception:
            logger.exception("Analytics poison hook failed")


def _validated_yaml_text(config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Return normalized YAML only when the persistent supervisor can reload it."""
    candidate = copy.deepcopy(config)
    _normalize_stream_keys(candidate)
    serialized = yaml.safe_dump(candidate, sort_keys=False)
    if len(serialized.encode("utf-8")) > ANALYTICS_YAML_MAX_BYTES:
        raise ValueError(
            f"Analytics YAML exceeds {ANALYTICS_YAML_MAX_BYTES} bytes"
        )
    return candidate, serialized


def _store_config(config: Dict[str, Any]) -> None:
    """Atomically persist config, publishing cache/path only after success."""
    global _CONFIG_CACHE, _CONFIG_PATH
    path = _CONFIG_PATH or _resolve_analytics_config()
    candidate, serialized = _validated_yaml_text(config)
    _atomic_write_text(path, serialized)
    _CONFIG_CACHE = copy.deepcopy(candidate)
    _CONFIG_PATH = path


def register_reload_hook(
    callback: Callable[[str, Dict[str, Any], Mapping[str, Any]], Mapping[str, Any]],
) -> None:
    """Allow the pipeline to subscribe for hot-reload notifications."""
    global _RELOAD_HOOK
    _RELOAD_HOOK = callback


def register_poison_hook(callback: Callable[[str], None]) -> None:
    """Register the runtime-fatal callback for unrecoverable state divergence."""
    global _POISON_HOOK
    _POISON_HOOK = callback


def clear_runtime_hooks() -> None:
    """Release process-local runtime callbacks during reversible startup abort."""
    global _POISON_HOOK, _RELOAD_HOOK
    _POISON_HOOK = None
    _RELOAD_HOOK = None


def _resolve_exclude_config_path() -> Path:
    """Determine the config file used by the exclusion element."""
    env_path = os.environ.get("NOESIS_ANALYTICS_EXCLUDE_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    try:
        from noesis.pipelines import ds8_pipeline
    except Exception as exc:
        raise RuntimeError("Analytics pipeline module is unavailable") from exc

    try:
        graph = ds8_pipeline.get_pipeline()
    except Exception as exc:
        raise RuntimeError(
            "Analytics exclusion path is unavailable before pipeline construction"
        ) from exc
    if graph is None:
        raise RuntimeError(
            "Analytics exclusion path is unavailable before pipeline construction"
        )
    component = graph.components.get("analytics_exclude")
    cfg_file = (
        component.config.get("config-file")
        if component is not None and isinstance(component.config, dict)
        else None
    )
    if not str(cfg_file or "").strip():
        raise RuntimeError(
            "Canonical analytics exclusion component has no config-file"
        )
    return Path(str(cfg_file)).expanduser()


def _coerce_roi_points(roi: Mapping[str, Any]) -> List[float]:
    coords: List[float] = []
    points = roi.get("points_px", []) or []
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Each exclusion ROI requires at least three points")
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each ROI vertex must contain exactly two coordinates")
        try:
            px = float(point[0])
            py = float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("ROI coordinates must be numeric") from exc
        if not math.isfinite(px) or not math.isfinite(py):
            raise ValueError("ROI coordinates must be finite")
        coords.extend([px, py])
    return coords


def _render_exclude_ini(stage_cfg: Dict[str, Any]) -> str:
    """Render and bound an nvdsroiexclude-style INI without touching disk."""
    parser = configparser.ConfigParser()
    parser.optionxform = str  # preserve hyphenated keys

    defaults = stage_cfg.get("defaults") or {}
    default_class_ids = defaults.get("class_ids") or []
    default_class_id = int(default_class_ids[0]) if default_class_ids else -1
    inverse_default = bool(defaults.get("inverse_roi", False))

    width = int(stage_cfg.get("config_width", 1920) or 1920)
    height = int(stage_cfg.get("config_height", 1080) or 1080)
    parser["property"] = {
        "enable": "1",
        "osd-mode": str(stage_cfg.get("osd_mode", 0) or 0),
        "display-font-size": str(stage_cfg.get("display_font_size", 12) or 12),
        "config-width": str(width),
        "config-height": str(height),
    }

    streams_cfg = stage_cfg.get("streams") or {}
    for stream_id, stream_cfg in streams_cfg.items():
        stream_id = str(stream_id)
        if not _CANONICAL_STREAM_ID.fullmatch(stream_id):
            raise ValueError(f"Unsafe analytics stream id: {stream_id!r}")
        section = f"roi-filtering-stream-{stream_id}"
        roi_filtering = stream_cfg.get("roi_filtering") or {}
        enable = bool(roi_filtering.get("enable", False))
        class_ids = (
            roi_filtering.get("class_ids")
            or stream_cfg.get("class_ids")
            or default_class_ids
        )
        class_id = int(class_ids[0]) if class_ids else default_class_id
        inverse_roi = roi_filtering.get(
            "inverse_roi", stream_cfg.get("inverse_roi", inverse_default)
        )
        section_values: Dict[str, str] = {
            "enable": "1" if enable else "0",
            "class-id": str(class_id),
            "inverse-roi": "1" if inverse_roi else "0",
        }

        rois = roi_filtering.get("rois") or []
        if enable and not rois:
            raise ValueError(
                f"Enabled analytics stream {stream_id} requires at least one exclusion ROI"
            )
        seen_roi_ids: set[str] = set()
        for roi in rois:
            roi_id = str(roi.get("id") or "").strip()
            if not _SAFE_ROI_ID.fullmatch(roi_id):
                raise ValueError(f"Unsafe exclusion ROI id: {roi_id!r}")
            if roi_id in seen_roi_ids:
                raise ValueError(f"Duplicate exclusion ROI id for stream {stream_id}: {roi_id}")
            seen_roi_ids.add(roi_id)
            coords = _coerce_roi_points(roi)
            rounded_values = [int(round(val)) for val in coords]
            rounded_points = list(zip(rounded_values[::2], rounded_values[1::2]))
            if any(x < 0 or x > width or y < 0 or y > height for x, y in rounded_points):
                raise ValueError(f"Exclusion ROI {roi_id!r} is outside {width}x{height}")
            if len(set(rounded_points)) < 3:
                raise ValueError(f"Exclusion ROI {roi_id!r} is degenerate after integer rounding")
            twice_area = abs(
                sum(
                    x1 * y2 - x2 * y1
                    for (x1, y1), (x2, y2) in zip(
                        rounded_points,
                        rounded_points[1:] + rounded_points[:1],
                    )
                )
            )
            if twice_area == 0:
                raise ValueError(f"Exclusion ROI {roi_id!r} is collinear after integer rounding")
            section_values[f"roi-{roi_id}"] = ";".join(str(value) for value in rounded_values)

        parser[section] = section_values

    stream = io.StringIO(newline="")
    parser.write(stream)
    rendered = stream.getvalue()
    if len(rendered.encode("utf-8")) > ANALYTICS_EXCLUDE_INI_MAX_BYTES:
        raise ValueError(
            "Analytics exclusion INI exceeds "
            f"{ANALYTICS_EXCLUDE_INI_MAX_BYTES} bytes"
        )
    return rendered


def _persist_exclude_ini(stage_cfg: Dict[str, Any], path: Path) -> None:
    """Persist a prevalidated nvdsroiexclude INI atomically."""
    _atomic_write_text(path, _render_exclude_ini(stage_cfg))


def _sync_exclude_stage(stage_name: str, stage_cfg: Dict[str, Any]) -> Optional[Path]:
    """Keep the exclusion INI in sync with the DS8 analytics YAML."""
    if stage_name != "exclude":
        return None
    target = _resolve_exclude_config_path()
    _persist_exclude_ini(stage_cfg, target)
    return target


def _build_exclude_reload_context(path: Path) -> Dict[str, str]:
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    payload = _snapshot_file(
        target,
        max_bytes=ANALYTICS_EXCLUDE_INI_MAX_BYTES,
    )
    if payload is None:
        raise FileNotFoundError(target)
    return {
        "config_path": str(target),
        "config_sha256": hashlib.sha256(payload).hexdigest(),
    }


class ROI(BaseModel):
    id: str = Field(..., min_length=1)
    description: Optional[str] = Field(None, max_length=512)
    points_px: List[List[float]] = Field(..., min_length=3, max_length=128)

    @validator("id")
    def _validate_id(cls, value: str) -> str:
        if not _SAFE_ROI_ID.fullmatch(value):
            raise ValueError("ROI id must use 1-64 letters, digits, dots, underscores, or hyphens")
        return value

    @validator("points_px", each_item=True)
    def _validate_points(cls, value: List[float]) -> List[float]:
        if len(value) != 2:
            raise ValueError("Each ROI vertex must be a two-element (x, y) list.")
        return [float(value[0]), float(value[1])]


class ROIStreamState(BaseModel):
    stream_id: str
    label: Optional[str] = None
    enable: bool = True
    rois: List[ROI] = Field(default_factory=list)


class ROIUpdateStream(BaseModel):
    stream_id: str
    label: Optional[str] = Field(None, max_length=128)
    enable: Optional[bool] = None
    rois: List[ROI] = Field(default_factory=list, max_length=64)

    @validator("stream_id")
    def _validate_stream_id(cls, value: str) -> str:
        if not _CANONICAL_STREAM_ID.fullmatch(value):
            raise ValueError("stream_id must be a canonical non-negative decimal id")
        return value

    @validator("rois")
    def _validate_unique_roi_ids(cls, value: List[ROI]) -> List[ROI]:
        identifiers = [roi.id for roi in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ROI ids must be unique within each stream")
        return value


class ROIListResponse(BaseModel):
    stage: str
    config_width: Optional[int] = None
    config_height: Optional[int] = None
    defaults: Dict[str, Any] = Field(default_factory=dict)
    streams: List[ROIStreamState] = Field(default_factory=list)
    config_path: Optional[str] = None


class ROIUpdateRequest(BaseModel):
    stage: str = Field("exclude", min_length=1)
    streams: List[ROIUpdateStream] = Field(..., min_length=1, max_length=64)

    @validator("stage")
    def _validate_stage(cls, value: str) -> str:
        if value != "exclude":
            raise ValueError("Only the 'exclude' analytics stage is writable")
        return value

    @validator("streams")
    def _validate_unique_streams(cls, value: List[ROIUpdateStream]) -> List[ROIUpdateStream]:
        identifiers = [stream.stream_id for stream in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("stream_id values must be unique per update")
        return value


class AnalyticsReloadReceipt(BaseModel):
    request_sequence: int = Field(..., ge=1)
    accepted_sequence: int = Field(..., ge=1)
    failed_sequence: int = Field(..., ge=0)
    active_config_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    reload_error_count: int = Field(..., ge=0)
    objects_removed_count: int = Field(..., ge=0)


class ROIUpdateResponse(ROIListResponse):
    reloaded: bool = False
    reload_receipt: Optional[AnalyticsReloadReceipt] = None


class NativeAnalyticsCommitAmbiguous(RuntimeError):
    """A native reload committed, but its Python-side publication did not."""

    native_state_ambiguous = True


def _extract_stage(config: Dict[str, Any], stage_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    analytics = config.get("analytics", {})
    stages = analytics.get("stages", {})
    stage_cfg = stages.get(stage_name)
    if stage_cfg is None:
        raise KeyError(stage_name)
    return analytics, stages, stage_cfg


def _stage_to_response(stage_name: str, stage_cfg: Dict[str, Any]) -> ROIListResponse:
    streams_cfg = stage_cfg.get("streams", {})
    streams: List[ROIStreamState] = []

    def _stream_sort_key(item: Tuple[str, Any]) -> Tuple[int, str]:
        key = item[0]
        try:
            return (0, f"{int(key):08d}")
        except Exception:
            return (1, str(key))

    for stream_id, stream_cfg in sorted(streams_cfg.items(), key=_stream_sort_key):
        roi_filtering = stream_cfg.get("roi_filtering", {})
        enable = bool(roi_filtering.get("enable", False))
        raw_rois = roi_filtering.get("rois", []) or []
        rois = [
            ROI(
                id=str(roi.get("id", "")),
                description=roi.get("description"),
                points_px=[list(map(float, point)) for point in roi.get("points_px", [])],
            )
            for roi in raw_rois
        ]
        streams.append(
            ROIStreamState(
                stream_id=str(stream_id),
                label=stream_cfg.get("label"),
                enable=enable,
                rois=rois,
            )
        )

    return ROIListResponse(
        stage=stage_name,
        config_width=stage_cfg.get("config_width"),
        config_height=stage_cfg.get("config_height"),
        defaults=stage_cfg.get("defaults", {}),
        streams=streams,
        config_path=str(_CONFIG_PATH) if _CONFIG_PATH else None,
    )


def _apply_updates(stage_cfg: Dict[str, Any], request: ROIUpdateRequest) -> None:
    streams_cfg = stage_cfg.setdefault("streams", {})

    for stream in request.streams:
        key = str(stream.stream_id)
        stream_cfg = streams_cfg.setdefault(key, {})
        if stream.label is not None:
            stream_cfg["label"] = stream.label

        roi_filtering = stream_cfg.setdefault("roi_filtering", {})
        enable = stream.enable if stream.enable is not None else roi_filtering.get("enable", True)
        roi_filtering["enable"] = bool(enable)
        roi_filtering["rois"] = [
            {
                "id": roi.id,
                "description": roi.description,
                "points_px": [[float(x), float(y)] for x, y in roi.points_px],
            }
            for roi in stream.rois
        ]


def _validate_update_bounds(stage_cfg: Dict[str, Any], request: ROIUpdateRequest) -> None:
    width = int(stage_cfg.get("config_width", 0) or 0)
    height = int(stage_cfg.get("config_height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Analytics stage must define positive config_width/config_height")
    for stream in request.streams:
        stream_cfg = (stage_cfg.get("streams") or {}).get(stream.stream_id)
        if stream_cfg is None:
            raise ValueError(f"Unknown analytics stream id: {stream.stream_id}")
        current_filtering = stream_cfg.get("roi_filtering") or {}
        effective_enable = (
            bool(stream.enable)
            if stream.enable is not None
            else bool(current_filtering.get("enable", True))
        )
        if effective_enable and not stream.rois:
            raise ValueError(
                f"Enabled analytics stream {stream.stream_id} requires at least one ROI; "
                "disable the stream when deleting its final ROI"
            )
        for roi in stream.rois:
            for x, y in roi.points_px:
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"ROI '{roi.id}' coordinates must be finite")
                if x < 0 or x > width or y < 0 or y > height:
                    raise ValueError(
                        f"ROI '{roi.id}' point ({x}, {y}) is outside {width}x{height}"
                    )


def _trigger_reload(stage_name: str, stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _RELOAD_COUNTER
    payload = copy.deepcopy(stage_cfg)
    exclude_path = _sync_exclude_stage(stage_name, payload)
    if exclude_path is None:
        raise RuntimeError("Analytics reload did not produce an exclusion config")
    reload_context = _build_exclude_reload_context(exclude_path)

    from noesis.pipelines import ds8_pipeline

    graph = ds8_pipeline.get_pipeline()

    if graph is None:
        raise RuntimeError("Analytics pipeline is unavailable")

    if _RELOAD_HOOK is None:
        raise RuntimeError("Analytics reload hook is not registered")

    exclude_component = graph.components.get("analytics_exclude")
    previous_exclude_path = None
    if stage_name == "exclude":
        if exclude_component is None:
            raise RuntimeError("Analytics exclusion component is unavailable")
        previous_exclude_path = exclude_component.config.get("config-file")
        if exclude_path is not None:
            exclude_component.config["config-file"] = str(exclude_path)

    try:
        raw_receipt = _RELOAD_HOOK(stage_name, payload, reload_context)
    except Exception:
        if stage_name == "exclude" and exclude_component is not None:
            exclude_component.config["config-file"] = previous_exclude_path
        raise
    try:
        receipt = AnalyticsReloadReceipt.model_validate(raw_receipt).model_dump()
        graph.config.setdefault("analytics", {}).setdefault("stages", {})[stage_name] = payload
        component = graph.components.get("analytics")
        if component is not None:
            component.config.setdefault("runtime_updates", {})[stage_name] = payload
        if exclude_component is not None:
            exclude_component.config.setdefault("runtime_updates", {})[stage_name] = payload
        graph.analytics_reload_receipt = copy.deepcopy(receipt)
    except Exception as exc:
        if stage_name == "exclude" and exclude_component is not None:
            exclude_component.config["config-file"] = previous_exclude_path
        raise NativeAnalyticsCommitAmbiguous(
            "Native analytics reload committed but its runtime receipt/publication failed"
        ) from exc

    _RELOAD_COUNTER += 1
    logger.info(
        "Analytics reload applied for stage %s (reload_count=%s)",
        stage_name,
        _RELOAD_COUNTER,
    )
    return receipt


@app.get("/api/v1/analytics/rois", response_model=ROIListResponse)
def list_rois(
    request: Request,
    stage: str = Query("exclude", min_length=1),
) -> ROIListResponse:
    with _CONFIG_LOCK:
        try:
            config = _load_config()
            _, _, stage_cfg = _extract_stage(config, stage)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=f"Analytics config missing: {exc}") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Stage '{stage}' not defined") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        with measure_rest_response_model(
            "/api/v1/analytics/rois:get", "ROIListResponse"
        ) as model_measurement:
            response = _stage_to_response(stage, stage_cfg)
        mark_rest_response(
            request,
            "/api/v1/analytics/rois:get",
            "ROIListResponse",
            model_duration_ms=model_measurement.elapsed_ms,
            include_budget=True,
        )
        return response


@app.post("/api/v1/analytics/rois", response_model=ROIUpdateResponse)
def update_rois(request: ROIUpdateRequest, http_request: Request) -> ROIUpdateResponse:
    global _CONFIG_CACHE, _CONFIG_PATH
    with _CONFIG_LOCK:
        try:
            config = _load_config()
            _, _, stage_cfg = _extract_stage(config, request.stage)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=f"Analytics config missing: {exc}") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Stage '{request.stage}' not defined") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            _validate_update_bounds(stage_cfg, request)
            config_path = _CONFIG_PATH or _resolve_analytics_config()
            exclude_path = _resolve_exclude_config_path()
            config_snapshot = _snapshot_file(config_path)
            exclude_snapshot = _snapshot_file(
                exclude_path,
                max_bytes=ANALYTICS_EXCLUDE_INI_MAX_BYTES,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to prepare analytics transaction: {exc}",
            ) from exc
        cache_snapshot = copy.deepcopy(_CONFIG_CACHE)
        cache_path_snapshot = _CONFIG_PATH

        _apply_updates(stage_cfg, request)
        try:
            _validated_yaml_text(config)
            _render_exclude_ini(stage_cfg)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            _store_config(config)
        except Exception as exc:
            logger.exception("Failed to persist analytics config")
            rollback_errors = _rollback_transaction(
                [
                    ("exclude", exclude_path, exclude_snapshot),
                    ("analytics", config_path, config_snapshot),
                ],
                cache_snapshot,
                cache_path_snapshot,
            )
            rollback_detail = (
                "; rollback failed: " + "; ".join(rollback_errors)
                if rollback_errors
                else ""
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to persist analytics config: {exc}{rollback_detail}",
            ) from exc

        with measure_rest_response_model(
            "/api/v1/analytics/rois:post", "ROIUpdateResponse"
        ) as stage_model_measurement:
            response = _stage_to_response(request.stage, stage_cfg)
        try:
            reload_receipt = _trigger_reload(request.stage, stage_cfg)
        except Exception as exc:
            logger.exception("Failed to apply analytics reload for stage %s", request.stage)
            rollback_errors = _rollback_transaction(
                [
                    ("exclude", exclude_path, exclude_snapshot),
                    ("analytics", config_path, config_snapshot),
                ],
                cache_snapshot,
                cache_path_snapshot,
            )
            rollback_detail = ""
            if rollback_errors:
                rollback_detail = "; rollback failed: " + "; ".join(rollback_errors)
            if getattr(exc, "native_state_ambiguous", False):
                _poison_state(
                    "native analytics reload may have committed without a complete receipt: "
                    f"{exc}"
                )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Failed to apply analytics reload for stage '{request.stage}': {exc}"
                    f"{rollback_detail}"
                ),
            ) from exc

    with measure_rest_response_model(
        "/api/v1/analytics/rois:post", "ROIUpdateResponse"
    ) as final_model_measurement:
        final_response = ROIUpdateResponse(
            **response.model_dump(),
            reloaded=True,
            reload_receipt=reload_receipt,
        )
    mark_rest_response(
        http_request,
        "/api/v1/analytics/rois:post",
        "ROIUpdateResponse",
        model_duration_ms=float(
            stage_model_measurement.elapsed_ms + final_model_measurement.elapsed_ms
        ),
        include_budget=True,
    )
    return final_response
