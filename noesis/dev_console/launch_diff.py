from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import yaml

from noesis.ds8_preflight import REPO_ROOT
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import build_effective_config, canonicalize_runtime_paths


_HIGH_IMPACT_KEYS = {
    "engine",
    "config-file-path",
    "config-file",
    "rtsp_port",
    "rtsp_enabled",
    "mosaic_webrtc_enabled",
    "enable",
    "display-mask",
    "display-bbox",
    "display-text",
}

_CATEGORY_LABELS = {
    "models": "Models",
    "preprocess": "Preprocess",
    "tracker": "Tracking",
    "analytics": "Analytics",
    "mosaic_output": "Mosaic",
    "visualization": "Visualization",
    "depth_registration": "Depth",
    "sources": "Sources",
    "streammux": "Streammux",
    "osd": "OSD",
    "env": "Env",
    "launch": "Launch",
}


def _path_text(path: Tuple[Any, ...]) -> str:
    parts: List[str] = []
    for part in path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        elif parts:
            parts.append(f".{part}")
        else:
            parts.append(str(part))
    return "".join(parts)


def _category(path: Tuple[Any, ...]) -> str:
    root = str(path[0]) if path else "launch"
    return _CATEGORY_LABELS.get(root, root.replace("_", " ").title())


def _display_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _display_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_display_value(child) for child in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if "://" in value:
            return value
        path = Path(value)
        if path.is_absolute():
            try:
                return str(path.relative_to(REPO_ROOT))
            except ValueError:
                for anchor in ("models", "config", "pipelines", "build"):
                    if anchor in path.parts:
                        index = path.parts.index(anchor)
                        return str(Path(*path.parts[index:]))
                return value
    return value


def _short(value: Any) -> str:
    text = str(_display_value(value))
    if len(text) <= 140:
        return text
    return text[:139] + "..."


def _is_high_impact(path: Tuple[Any, ...]) -> bool:
    if not path:
        return False
    leaf = str(path[-1])
    return leaf in _HIGH_IMPACT_KEYS or any(str(part) == "pgie" for part in path)


def _append_change(changes: List[Dict[str, Any]], kind: str, path: Tuple[Any, ...], before: Any, after: Any) -> None:
    changes.append(
        {
            "kind": kind,
            "path": _path_text(path),
            "category": _category(path),
            "before": _display_value(before),
            "after": _display_value(after),
            "before_text": _short(before),
            "after_text": _short(after),
            "high_impact": _is_high_impact(path),
        }
    )


def _diff_values(changes: List[Dict[str, Any]], before: Any, after: Any, path: Tuple[Any, ...] = ()) -> None:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        for key in sorted(before_keys - after_keys, key=str):
            _append_change(changes, "removed", path + (key,), before[key], None)
        for key in sorted(after_keys - before_keys, key=str):
            _append_change(changes, "added", path + (key,), None, after[key])
        for key in sorted(before_keys & after_keys, key=str):
            _diff_values(changes, before[key], after[key], path + (key,))
        return
    if isinstance(before, list) and isinstance(after, list):
        shared = min(len(before), len(after))
        for index in range(shared):
            _diff_values(changes, before[index], after[index], path + (index,))
        for index in range(shared, len(before)):
            _append_change(changes, "removed", path + (index,), before[index], None)
        for index in range(shared, len(after)):
            _append_change(changes, "added", path + (index,), None, after[index])
        return
    if before != after:
        _append_change(changes, "changed", path, before, after)


def _load_base_config(spec: LaunchSpec) -> Dict[str, Any]:
    base_path = spec.pipeline_path
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"pipeline YAML must be a mapping: {base_path}")
    return canonicalize_runtime_paths(deepcopy(raw), base_path)


def _env_changes(spec: LaunchSpec) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    for key, value in sorted((spec.env or {}).items()):
        _append_change(changes, "added", ("env", str(key)), None, str(value))
    return changes


def _launch_changes(spec: LaunchSpec) -> List[Dict[str, Any]]:
    fields = {
        "pgie_profile": spec.pgie_profile,
        "size": spec.size or "auto",
        "tracking_mode": spec.tracking_mode,
        "depth_enable_seconds": int(spec.depth_enable_seconds),
        "strict_baseline": bool(spec.strict_baseline),
        "ws_port": int(spec.ws_port),
        "rest_port": int(spec.rest_port),
        "rtsp_port": int(spec.rtsp_port),
        "log_level": spec.log_level,
    }
    changes: List[Dict[str, Any]] = []
    for key, value in fields.items():
        _append_change(changes, "selected", ("launch", key), None, value)
    return changes


def build_launch_diff(spec: LaunchSpec) -> Dict[str, Any]:
    base = _load_base_config(spec)
    effective = build_effective_config(spec)
    changes: List[Dict[str, Any]] = []
    _diff_values(changes, base, effective)
    changes.extend(_env_changes(spec))
    launch_changes = _launch_changes(spec)

    categories: Dict[str, Dict[str, Any]] = {}
    for item in changes:
        category = str(item["category"])
        entry = categories.setdefault(category, {"label": category, "total": 0, "high_impact": 0})
        entry["total"] += 1
        if item.get("high_impact"):
            entry["high_impact"] += 1

    ordered_changes = sorted(
        changes,
        key=lambda item: (
            0 if item.get("high_impact") else 1,
            str(item.get("category")),
            str(item.get("path")),
        ),
    )
    return {
        "schema_version": 1,
        "base_pipeline": str(spec.pipeline_path),
        "summary": {
            "total": len(changes),
            "high_impact": sum(1 for item in changes if item.get("high_impact")),
            "env_overrides": len(spec.env or {}),
            "launch_fields": len(launch_changes),
            "categories": len(categories),
            "pgie_profile": spec.pgie_profile,
            "size": spec.size,
            "tracking_mode": spec.tracking_mode,
        },
        "categories": sorted(categories.values(), key=lambda item: str(item["label"])),
        "launch": launch_changes,
        "changes": ordered_changes,
        "base": _display_value(base),
        "effective": _display_value(effective),
    }
