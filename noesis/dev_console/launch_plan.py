from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from noesis.ds8_preflight import REPO_ROOT
from noesis.dev_console.diagnostics import artifact_audit
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import build_effective_config, materialize_launch_pipeline


TRACKED_PATHS = [
    ("Preprocess config", ("preprocess", "config-file")),
    ("PGIE config", ("models", "pgie", "config-file-path")),
    ("PGIE engine", ("models", "pgie", "engine")),
    ("PGIE mask metadata", ("models", "pgie", "attach_tensor_meta")),
    ("ReID enabled", ("models", "reid", "enable")),
    ("ReID config", ("models", "reid", "config-file-path")),
    ("Pose enabled", ("models", "pose", "enable")),
    ("Pose config", ("models", "pose", "config-file-path")),
    ("Depth tracking enabled", ("models", "depth_tracking", "enable")),
    ("Depth tracking config", ("models", "depth_tracking", "config-file-path")),
    ("MapAnything enabled", ("models", "mapanything", "enable")),
    ("MapAnything config", ("models", "mapanything", "config-file-path")),
    ("Tracker config", ("tracker", "config-file")),
    ("Analytics stages", ("analytics", "stages_config")),
    ("RTSP enabled", ("mosaic_output", "rtsp_enabled")),
    ("RTSP port", ("mosaic_output", "rtsp_port")),
    ("WebRTC gateway", ("mosaic_output", "mosaic_webrtc_enabled")),
    ("OSD masks", ("osd", "display-mask")),
    ("OSD boxes", ("osd", "display-bbox")),
    ("OSD text", ("osd", "display-text")),
]


def _get(mapping: Mapping[str, Any], path: Iterable[str]) -> Any:
    value: Any = mapping
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _path_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    if "://" in text:
        return text
    path = Path(text)
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return text
    return text


def _normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    return _path_text(value)


def _tracked_changes(base: Mapping[str, Any], effective: Mapping[str, Any]) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    for label, path in TRACKED_PATHS:
        before = _normalize(_get(base, path))
        after = _normalize(_get(effective, path))
        changed = before != after
        if changed or after not in (None, ""):
            changes.append(
                {
                    "label": label,
                    "path": ".".join(path),
                    "before": before,
                    "after": after,
                    "changed": changed,
                }
            )
    return changes


def _gate_value(spec: LaunchSpec, effective: Mapping[str, Any], env: Mapping[str, str], key: str, path: Optional[Iterable[str]] = None, default: Any = "") -> Any:
    if key in env:
        return env[key]
    if path is not None:
        value = _get(effective, path)
        if value is not None:
            return value
    return default


def _active_gates(spec: LaunchSpec, effective: Mapping[str, Any]) -> List[Dict[str, Any]]:
    env = {str(key): str(value) for key, value in spec.env.items()}
    return [
        {
            "label": "Tracking mode",
            "key": "NOESIS_TRACKING_MODE",
            "value": spec.tracking_mode,
            "source": "launch arg",
        },
        {
            "label": "Startup depth gate",
            "key": "NOESIS_DEPTH_ENABLE_SECONDS",
            "value": int(spec.depth_enable_seconds),
            "source": "launch arg",
        },
        {
            "label": "MapAnything prime",
            "key": "NOESIS_MAPANYTHING_GATE_PRIME_SECONDS",
            "value": env.get("NOESIS_MAPANYTHING_GATE_PRIME_SECONDS", "1.0"),
            "source": "env" if "NOESIS_MAPANYTHING_GATE_PRIME_SECONDS" in env else "runtime default",
        },
        {
            "label": "ReID",
            "key": "NOESIS_REID_ENABLED",
            "value": _gate_value(spec, effective, env, "NOESIS_REID_ENABLED", ("models", "reid", "enable"), True),
            "source": "env" if "NOESIS_REID_ENABLED" in env else "pipeline",
        },
        {
            "label": "Pose features",
            "key": "NOESIS_POSE_FEATURES_ENABLED",
            "value": env.get("NOESIS_POSE_FEATURES_ENABLED", "1"),
            "source": "env" if "NOESIS_POSE_FEATURES_ENABLED" in env else "runtime default",
        },
        {
            "label": "Trails",
            "key": "NOESIS_TRAILS_RENDER",
            "value": env.get("NOESIS_TRAILS_RENDER", "1"),
            "source": "env" if "NOESIS_TRAILS_RENDER" in env else "runtime default",
        },
        {
            "label": "RTSP",
            "key": "NOESIS_MOSAIC_RTSP_ENABLED",
            "value": _gate_value(spec, effective, env, "NOESIS_MOSAIC_RTSP_ENABLED", ("mosaic_output", "rtsp_enabled"), False),
            "source": "env" if "NOESIS_MOSAIC_RTSP_ENABLED" in env else "pipeline",
        },
        {
            "label": "WebRTC",
            "key": "NOESIS_MOSAIC_WEBRTC_ENABLED",
            "value": _gate_value(spec, effective, env, "NOESIS_MOSAIC_WEBRTC_ENABLED", ("mosaic_output", "mosaic_webrtc_enabled"), False),
            "source": "env" if "NOESIS_MOSAIC_WEBRTC_ENABLED" in env else "pipeline",
        },
    ]


def build_launch_plan(spec: LaunchSpec) -> Dict[str, Any]:
    base_path = spec.pipeline_path
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError(f"pipeline YAML must be a mapping: {base_path}")
    materialized = materialize_launch_pipeline(spec, dry_run=True)
    effective = build_effective_config(spec)
    materialized_rel = str(materialized.relative_to(REPO_ROOT)) if materialized.is_relative_to(REPO_ROOT) else str(materialized)
    argv = spec.to_argv(pipeline_config=materialized_rel)
    artifact_snapshot = artifact_audit(spec)
    changed = _tracked_changes(base_cfg, effective)
    return {
        "spec": spec.to_dict(),
        "command": " ".join(argv),
        "argv": argv,
        "env_overlay": spec.to_dict().get("env_overlay", {}),
        "materialized_pipeline": str(materialized),
        "materialized_pipeline_rel": materialized_rel,
        "base_pipeline": str(base_path),
        "changes": changed,
        "change_count": sum(1 for item in changed if item["changed"]),
        "gates": _active_gates(spec, effective),
        "artifacts": artifact_snapshot,
        "ports": {
            "ws": {"host": spec.ws_host, "port": int(spec.ws_port)},
            "rest": {"host": spec.rest_host, "port": int(spec.rest_port), "enabled": bool(spec.enable_rest)},
            "rtsp": {"host": "127.0.0.1", "port": int(spec.rtsp_port)},
        },
    }
