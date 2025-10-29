from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import copy
import logging
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

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None
_RELOAD_HOOK: Optional[Callable[[str, Dict[str, Any]], None]] = None


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
    """Persist the in-memory cache to keep GET responses consistent."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = copy.deepcopy(config)


def register_reload_hook(callback: Callable[[str, Dict[str, Any]], None]) -> None:
    """Allow the pipeline to subscribe for hot-reload notifications."""
    global _RELOAD_HOOK
    _RELOAD_HOOK = callback


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
    payload = copy.deepcopy(stage_cfg)

    if _RELOAD_HOOK is not None:
        try:
            _RELOAD_HOOK(stage_name, payload)
            return True
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Analytics reload hook failed for stage %s", stage_name)
            return False

    try:
        from noesis.pipelines import ds8_pipeline

        graph = ds8_pipeline.get_pipeline()
    except Exception:
        return False

    component = graph.components.get("analytics") if graph else None
    if component is None:
        return False

    component.config.setdefault("runtime_updates", {})[stage_name] = payload
    return True


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
