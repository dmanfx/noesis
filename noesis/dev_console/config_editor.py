from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import build_effective_config, deep_merge, profile_overlay


def model_patch(
    pipeline_path: Path,
    *,
    pgie_profile: Optional[str] = None,
    size: Optional[str] = None,
    reid_enable: Optional[bool] = None,
    pose_enable: Optional[bool] = None,
    mapanything_enable: Optional[bool] = None,
    tracking_mode: str = "baseline",
) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(pipeline_path).read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"pipeline YAML must be a mapping: {pipeline_path}")
    merged = deepcopy(cfg)
    if pgie_profile:
        merged = deep_merge(merged, profile_overlay(pgie_profile, size))
    for key, value in {
        "reid": reid_enable,
        "pose": pose_enable,
        "mapanything": mapanything_enable,
    }.items():
        if value is None:
            continue
        merged = deep_merge(merged, {"models": {key: {"enable": bool(value)}}})
    spec = LaunchSpec(
        pipeline_config=str(Path(pipeline_path)),
        pgie_profile=pgie_profile or str(((merged.get("models") or {}).get("pgie") or {}).get("profile", "yolo11_seg")),
        size=size,
        tracking_mode=tracking_mode,
    )
    try:
        spec.pipeline_config = str(Path(pipeline_path).relative_to(Path.cwd()))
    except Exception:
        spec.pipeline_config = str(Path(pipeline_path))
    effective = build_effective_config(spec) if pgie_profile else merged
    if "osd" not in effective:
        effective["osd"] = {"process-mode": 0, "display-mask": 1, "display-bbox": 0, "display-text": 1}
    return effective
