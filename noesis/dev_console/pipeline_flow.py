from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from noesis.ds8_preflight import REPO_ROOT
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import build_effective_config


def _stage(
    stage_id: str,
    label: str,
    kind: str,
    status: str = "ready",
    detail: str = "",
    lane: str = "main",
    metrics: Optional[Mapping[str, Any]] = None,
    controls: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "id": stage_id,
        "label": label,
        "kind": kind,
        "status": status,
        "detail": detail,
        "lane": lane,
        "metrics": dict(metrics or {}),
        "controls": [dict(item) for item in controls or []],
    }


def _enabled(model: Mapping[str, Any], default: bool = True) -> bool:
    value = model.get("enable")
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def describe_flow(
    *,
    pipeline_config: str,
    pgie_profile: str,
    size: Optional[str],
    tracking_mode: str,
    rtsp_port: int = 8554,
    depth_enable_seconds: int = 0,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    env_map = {str(key): str(value) for key, value in dict(env or {}).items()}
    spec = LaunchSpec(
        pipeline_config=pipeline_config,
        pgie_profile=pgie_profile,
        size=size,
        tracking_mode=tracking_mode,
        rtsp_port=int(rtsp_port),
        depth_enable_seconds=int(depth_enable_seconds),
        env=env_map,
    )
    cfg = build_effective_config(spec)
    sources = cfg.get("sources") if isinstance(cfg.get("sources"), list) else []
    streammux = cfg.get("streammux") if isinstance(cfg.get("streammux"), Mapping) else {}
    models = cfg.get("models") if isinstance(cfg.get("models"), Mapping) else {}
    pgie = models.get("pgie") if isinstance(models.get("pgie"), Mapping) else {}
    tracker = cfg.get("tracker") if isinstance(cfg.get("tracker"), Mapping) else {}
    mosaic = cfg.get("mosaic_output") if isinstance(cfg.get("mosaic_output"), Mapping) else {}

    stages: List[Dict[str, Any]] = []
    stages.append(
        _stage(
            "sources",
            "Sources",
            "input",
            detail=f"{len(sources)} configured",
            metrics={"count": len(sources), "type": "rtsp" if any("rtsp://" in str(src.get("uri", "")) for src in sources if isinstance(src, Mapping)) else "file"},
        )
    )
    stages.append(
        _stage(
            "streammux",
            "Streammux",
            "gpu",
            detail=f"{streammux.get('width', '?')}x{streammux.get('height', '?')} batch {streammux.get('batch-size', cfg.get('batch_size', '?'))}",
        )
    )
    stages.append(_stage("preprocess", "Preprocess", "gpu", detail=str((cfg.get("preprocess") or {}).get("config-file", ""))))
    stages.append(
        _stage(
            "pgie",
            "PGIE",
            "inference",
            detail=f"{pgie_profile} {size or ''}".strip(),
            metrics={"config": pgie.get("config-file-path"), "engine": pgie.get("engine"), "gie_id": pgie.get("gie_id", 1)},
            controls=[
                {
                    "type": "select",
                    "label": "PGIE",
                    "target": "pgie_profile",
                    "value": pgie_profile,
                    "options": ["yolo11_seg", "yolo11", "yolo26_seg", "yolo26", "rfdetr_seg", "rfdetr", "wholebody49"],
                },
                {"type": "select", "label": "Size", "target": "size", "value": size or "", "options": ["", "n", "s", "m", "l", "x"]},
            ],
        )
    )
    stages.append(_stage("main_tee", "Main tee", "split", detail="tracker, depth, and map lanes"))

    depth_tracking = models.get("depth_tracking") if isinstance(models.get("depth_tracking"), Mapping) else {}
    stages.append(
        _stage(
            "depth_tracking",
            "DAv2 tracking depth",
            "inference",
            "ready" if _enabled(depth_tracking, default=False) else "disabled",
            detail=str(depth_tracking.get("config-file-path", "")),
            lane="depth",
            controls=[
                {
                    "type": "select",
                    "label": "Tracking",
                    "target": "tracking_mode",
                    "value": tracking_mode,
                    "options": ["baseline", "v3dt"],
                }
            ],
        )
    )
    mapanything = models.get("mapanything") if isinstance(models.get("mapanything"), Mapping) else {}
    stages.append(
        _stage(
            "mapanything",
            "MapAnything gate",
            "inference",
            "gated" if _enabled(mapanything, default=False) else "disabled",
            detail="on-demand full-frame depth",
            lane="depth",
            controls=[
                {
                    "type": "number",
                    "label": "Depth seconds",
                    "target": "depth_enable_seconds",
                    "value": int(depth_enable_seconds),
                    "min": 0,
                    "max": 300,
                },
                {
                    "type": "number",
                    "label": "Prime seconds",
                    "target": "NOESIS_MAPANYTHING_GATE_PRIME_SECONDS",
                    "value": env_map.get("NOESIS_MAPANYTHING_GATE_PRIME_SECONDS", "1.0"),
                    "min": 0,
                    "max": 30,
                    "step": 0.5,
                },
            ],
        )
    )

    stages.append(
        _stage(
            "tracker",
            "Tracker",
            "tracking",
            detail=str(tracker.get("config-file", "")),
            controls=[
                {
                    "type": "select",
                    "label": "Tracking",
                    "target": "tracking_mode",
                    "value": tracking_mode,
                    "options": ["baseline", "v3dt"],
                }
            ],
        )
    )
    stages.append(_stage("analytics", "Analytics", "metadata", detail=str((cfg.get("analytics") or {}).get("stages_config", ""))))
    reid = models.get("reid") if isinstance(models.get("reid"), Mapping) else {}
    stages.append(
        _stage(
            "reid",
            "ReID",
            "identity",
            "ready" if _enabled(reid, default=False) else "disabled",
            detail=str(reid.get("config-file-path", "")),
            controls=[
                {
                    "type": "toggle",
                    "label": "ReID",
                    "target": "NOESIS_REID_ENABLED",
                    "value": env_map.get("NOESIS_REID_ENABLED", "1"),
                }
            ],
        )
    )
    pose = models.get("pose") if isinstance(models.get("pose"), Mapping) else {}
    stages.append(
        _stage(
            "pose",
            "Pose",
            "pose",
            "ready" if _enabled(pose, default=False) else "disabled",
            detail=str(pose.get("config-file-path", "")),
            controls=[
                {
                    "type": "toggle",
                    "label": "Pose features",
                    "target": "NOESIS_POSE_FEATURES_ENABLED",
                    "value": env_map.get("NOESIS_POSE_FEATURES_ENABLED", "1"),
                },
                {
                    "type": "toggle",
                    "label": "ReID pose",
                    "target": "NOESIS_REID_POSE_ENABLED",
                    "value": env_map.get("NOESIS_REID_POSE_ENABLED", ""),
                },
            ],
        )
    )
    stages.append(
        _stage(
            "world",
            "World observation",
            "metadata",
            detail=tracking_mode,
            controls=[
                {
                    "type": "toggle",
                    "label": "Trails",
                    "target": "NOESIS_TRAILS_RENDER",
                    "value": env_map.get("NOESIS_TRAILS_RENDER", "1"),
                }
            ],
        )
    )
    stages.append(_stage("tiler", "Tiler", "gpu", detail="mosaic layout"))
    stages.append(
        _stage(
            "osd",
            "OSD",
            "display",
            detail=str(cfg.get("osd", {})),
            controls=[
                {
                    "type": "toggle",
                    "label": "Trails",
                    "target": "NOESIS_TRAILS_RENDER",
                    "value": env_map.get("NOESIS_TRAILS_RENDER", "1"),
                }
            ],
        )
    )
    stages.append(
        _stage(
            "rtsp",
            "RTSP mosaic",
            "output",
            "ready" if mosaic.get("rtsp_enabled", True) else "disabled",
            detail=f":{mosaic.get('rtsp_port', 8554)}/{mosaic.get('rtsp_path', 'mosaic')}",
            controls=[
                {"type": "number", "label": "RTSP port", "target": "rtsp_port", "value": int(rtsp_port), "min": 1024, "max": 65535},
                {
                    "type": "toggle",
                    "label": "RTSP",
                    "target": "NOESIS_MOSAIC_RTSP_ENABLED",
                    "value": env_map.get("NOESIS_MOSAIC_RTSP_ENABLED", "1"),
                },
            ],
        )
    )
    stages.append(
        _stage(
            "webrtc",
            "WebRTC gateway",
            "output",
            "ready" if mosaic.get("mosaic_webrtc_enabled", False) else "disabled",
            detail="browser mosaic video",
            controls=[
                {
                    "type": "toggle",
                    "label": "WebRTC",
                    "target": "NOESIS_MOSAIC_WEBRTC_ENABLED",
                    "value": env_map.get("NOESIS_MOSAIC_WEBRTC_ENABLED", "1"),
                }
            ],
        )
    )

    edges = [
        ["sources", "streammux"],
        ["streammux", "preprocess"],
        ["preprocess", "pgie"],
        ["pgie", "main_tee"],
        ["main_tee", "depth_tracking"],
        ["main_tee", "mapanything"],
        ["main_tee", "tracker"],
        ["tracker", "analytics"],
        ["analytics", "reid"],
        ["reid", "pose"],
        ["pose", "world"],
        ["world", "tiler"],
        ["tiler", "osd"],
        ["osd", "rtsp"],
        ["rtsp", "webrtc"],
    ]
    env_knobs = [{"key": str(key), "value": str(value)} for key, value in sorted(env_map.items())]
    return {
        "pipeline_config": str((REPO_ROOT / pipeline_config).resolve() if not Path(pipeline_config).is_absolute() else Path(pipeline_config)),
        "stages": stages,
        "edges": edges,
        "environment_knobs": env_knobs,
        "highlights": {
            "pgie_profile": pgie_profile,
            "size": size,
            "tracking_mode": tracking_mode,
            "rtsp_port": int(rtsp_port),
            "depth_enable_seconds": int(depth_enable_seconds),
            "source_count": len(sources),
            "mosaic_webrtc": bool(mosaic.get("mosaic_webrtc_enabled", False)),
        },
        "raw_config": yaml.safe_dump(cfg, sort_keys=False),
    }
