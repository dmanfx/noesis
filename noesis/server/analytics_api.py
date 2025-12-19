from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Mapping

import copy
import logging
import configparser
import os
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator

from noesis.server.depth_api import ensure_pipeline_ready  # noqa: F401 - re-export for dependency parity

logger = logging.getLogger(__name__)

app = FastAPI(title="Noesis DS8 Analytics API")

ANALYTICS_CONFIG_ENV = "NOESIS_ANALYTICS_CONFIG"
DEFAULT_ANALYTICS_CONFIG = Path("config/nvdsanalytics.yaml")
DEFAULT_EXCLUDE_CONFIG = Path("config/config_nvdsanalytics_exclude.ini")

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None
_RELOAD_HOOK: Optional[Callable[[str, Dict[str, Any]], None]] = None
_RELOAD_COUNTER: int = 0


def _resolve_analytics_config() -> Path:
    """Return the analytics config path, preferring an env override."""
    env_path = os.environ.get(ANALYTICS_CONFIG_ENV)
    path = Path(env_path).expanduser() if env_path else DEFAULT_ANALYTICS_CONFIG
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_config(force: bool = False) -> Dict[str, Any]:
    """Load the nvdsanalytics configuration, caching for subsequent requests."""
    global _CONFIG_CACHE, _CONFIG_PATH

    path = _resolve_analytics_config()
    if not force and _CONFIG_CACHE is not None and _CONFIG_PATH == path:
        return copy.deepcopy(_CONFIG_CACHE)

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    _CONFIG_CACHE = copy.deepcopy(config)
    _CONFIG_PATH = path
    return config


def _store_config(config: Dict[str, Any]) -> None:
    """Persist the in-memory cache and write through to disk."""
    global _CONFIG_CACHE
    path = _CONFIG_PATH or _resolve_analytics_config()
    try:
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    except Exception as exc:
        logger.exception("Failed to write analytics config to %s: %s", path, exc)
    _CONFIG_CACHE = copy.deepcopy(config)


def register_reload_hook(callback: Callable[[str, Dict[str, Any]], None]) -> None:
    """Allow the pipeline to subscribe for hot-reload notifications."""
    global _RELOAD_HOOK
    _RELOAD_HOOK = callback


def _resolve_exclude_config_path() -> Path:
    """Determine the config file used by the exclusion element."""
    env_path = os.environ.get("NOESIS_ANALYTICS_EXCLUDE_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    try:
        from noesis.pipelines import ds8_pipeline

        graph = ds8_pipeline.get_pipeline()
        component = graph.components.get("analytics_exclude") if graph else None
        cfg_file = None
        if component and isinstance(component.config, dict):
            cfg_file = component.config.get("config-file")
        if cfg_file:
            return Path(cfg_file).expanduser()
    except Exception:
        pass
    return DEFAULT_EXCLUDE_CONFIG


def _coerce_roi_points(roi: Mapping[str, Any]) -> List[float]:
    coords: List[float] = []
    for point in roi.get("points_px", []) or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            px = float(point[0])
            py = float(point[1])
        except Exception:
            continue
        coords.extend([px, py])
    return coords


def _persist_exclude_ini(stage_cfg: Dict[str, Any], path: Path) -> None:
    """Render an nvdsroiexclude-style INI from the DS8 analytics stage config."""
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
        for roi in rois:
            roi_id = str(roi.get("id") or roi.get("label") or "").strip() or "roi"
            coords = _coerce_roi_points(roi)
            if len(coords) < 6:  # need at least 3 points
                continue
            try:
                rounded = [str(int(round(val))) for val in coords]
            except Exception:
                rounded = [str(val) for val in coords]
            section_values[f"roi-{roi_id}"] = ";".join(rounded)

        parser[section] = section_values

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        parser.write(stream)


def _sync_exclude_stage(stage_name: str, stage_cfg: Dict[str, Any]) -> Optional[Path]:
    """Keep the exclusion INI in sync with the DS8 analytics YAML."""
    if stage_name != "exclude":
        return None
    try:
        target = _resolve_exclude_config_path()
        _persist_exclude_ini(stage_cfg, target)
        return target
    except Exception as exc:
        logger.warning("Failed to persist exclusion config: %s", exc)
        return None


class ROI(BaseModel):
    id: str = Field(..., min_length=1)
    description: Optional[str] = None
    points_px: List[List[float]] = Field(..., min_length=3)

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
    label: Optional[str] = None
    enable: Optional[bool] = None
    rois: List[ROI] = Field(default_factory=list)


class ROIListResponse(BaseModel):
    stage: str
    config_width: Optional[int] = None
    config_height: Optional[int] = None
    defaults: Dict[str, Any] = Field(default_factory=dict)
    streams: List[ROIStreamState] = Field(default_factory=list)
    config_path: Optional[str] = None


class ROIUpdateRequest(BaseModel):
    stage: str = Field("exclude", min_length=1)
    streams: List[ROIUpdateStream] = Field(..., min_length=1)


class ROIUpdateResponse(ROIListResponse):
    reloaded: bool = False


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

    for stream_id, stream_cfg in streams_cfg.items():
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


def _trigger_reload(stage_name: str, stage_cfg: Dict[str, Any]) -> bool:
    global _RELOAD_COUNTER
    payload = copy.deepcopy(stage_cfg)
    exclude_path = _sync_exclude_stage(stage_name, payload)
    _RELOAD_COUNTER += 1
    logger.info("Analytics reload applied for stage %s (reload_count=%s)", stage_name, _RELOAD_COUNTER)
    reloaded = False

    if _RELOAD_HOOK is not None:
        try:
            _RELOAD_HOOK(stage_name, payload)
            reloaded = True
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Analytics reload hook failed for stage %s", stage_name)

    graph = None
    try:
        from noesis.pipelines import ds8_pipeline

        graph = ds8_pipeline.get_pipeline()
    except Exception:
        graph = None

    if graph is not None:
        analytics_cfg = graph.config.setdefault("analytics", {})
        analytics_cfg.setdefault("stages", {})[stage_name] = payload

        component = graph.components.get("analytics")
        if component is not None:
            component.config.setdefault("runtime_updates", {})[stage_name] = payload
            if exclude_path and stage_name == "exclude":
                component.config.setdefault("exclude_config_path", str(exclude_path))

        if stage_name == "exclude":
            exclude_component = graph.components.get("analytics_exclude")
            if exclude_component is not None:
                exclude_component.config.setdefault("runtime_updates", {})[stage_name] = payload
                if exclude_path:
                    exclude_component.config["config-file"] = str(exclude_path)

    return reloaded or graph is not None


@app.get("/api/v1/analytics/rois", response_model=ROIListResponse)
def list_rois(stage: str = Query("exclude", min_length=1)) -> ROIListResponse:
    with _CONFIG_LOCK:
        try:
            config = _load_config()
            _, _, stage_cfg = _extract_stage(config, stage)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=f"Analytics config missing: {exc}") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Stage '{stage}' not defined") from exc

        return _stage_to_response(stage, stage_cfg)


@app.post("/api/v1/analytics/rois", response_model=ROIUpdateResponse)
def update_rois(request: ROIUpdateRequest) -> ROIUpdateResponse:
    with _CONFIG_LOCK:
        try:
            config = _load_config()
            _, _, stage_cfg = _extract_stage(config, request.stage)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=f"Analytics config missing: {exc}") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Stage '{request.stage}' not defined") from exc

        _apply_updates(stage_cfg, request)
        _store_config(config)

        response = _stage_to_response(request.stage, stage_cfg)

    reloaded = _trigger_reload(request.stage, stage_cfg)
    return ROIUpdateResponse(**response.model_dump(), reloaded=reloaded)
