from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from noesis.ds8_preflight import (
    REPO_ROOT,
    derive_osd_policy_from_ini,
    parse_nvinfer_ini,
    resolve_config_path,
    validate_metadata_compatibility,
)
from noesis.dev_console.launch_spec import LaunchSpec

LOGGER = logging.getLogger(__name__)


PROFILE_DEFAULT_SIZE = {
    "yolo11": "m",
    "yolo11_seg": "m",
    "yolo26": "m",
    "yolo26_seg": "s",
    "rfdetr": "m",
    "rfdetr_seg": "m",
    "wholebody49": "s",
}


def deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = deep_merge(merged.get(key), value) if key in merged else value
        return merged
    return deepcopy(overlay)


def _rel(path: str) -> str:
    return str(Path(path))


def profile_overlay(profile: str, size: Optional[str]) -> Dict[str, Any]:
    profile = str(profile or "yolo11_seg").strip()
    size_norm = str(size or PROFILE_DEFAULT_SIZE.get(profile, "m")).strip().lower()

    if profile == "yolo11":
        return {
            "preprocess": {"config-file": _rel(f"build/config_preproc_yolo11_{size_norm}.ini")},
            "models": {
                "pgie": {
                    "config-file-path": _rel(f"build/config_infer_primary_yolo11_{size_norm}.ini"),
                    "engine": _rel(f"models/engines/yolo11{size_norm}_b3_fp16.engine"),
                }
            },
        }
    if profile == "yolo11_seg":
        return {
            "preprocess": {"config-file": _rel(f"build/config_preproc_yolo11_seg_{size_norm}.ini")},
            "models": {
                "pgie": {
                    "config-file-path": _rel(f"build/config_infer_primary_yolo11_seg_{size_norm}.ini"),
                    "engine": _rel(f"models/engines/yolo11{size_norm}-seg_cust.engine"),
                }
            },
        }
    if profile == "yolo26":
        return {
            "preprocess": {"config-file": _rel(f"build/config_preproc_yolo26_{size_norm}.ini")},
            "models": {
                "pgie": {
                    "config-file-path": _rel(f"build/config_infer_primary_yolo26_{size_norm}.ini"),
                    "engine": _rel(f"models/engines/yolo26{size_norm}_dynamic_b1-3_fp16.engine"),
                }
            },
        }
    if profile == "yolo26_seg":
        return {
            "preprocess": {"config-file": _rel(f"build/config_preproc_yolo26_seg_{size_norm}_b3.ini")},
            "models": {
                "pgie": {
                    "config-file-path": _rel(f"build/config_infer_primary_yolo26_seg_{size_norm}.ini"),
                    "engine": _rel(f"models/engines/yolo26{size_norm}-seg_fused_b3_fp16.engine"),
                }
            },
        }
    if profile == "rfdetr":
        resolution = {"n": "384", "s": "512", "m": "576"}.get(size_norm, "576")
        return {
            "preprocess": {"config-file": _rel(f"pipelines/config_preproc_rfdetr_detect_{resolution}.ini")},
            "models": {
                "pgie": {
                    "config-file-path": _rel(f"build/config_infer_primary_rfdetr_{size_norm}.ini"),
                    "engine": _rel(f"models/engines/rfdetr_{size_norm}_{resolution}_b3_fp16.engine"),
                }
            },
        }
    if profile == "rfdetr_seg":
        resolution = {"n": "312", "s": "384", "m": "432"}.get(size_norm, "432")
        return {
            "preprocess": {"config-file": _rel(f"pipelines/config_preproc_rfdetr_{resolution}.ini")},
            "models": {
                "pgie": {
                    "config-file-path": _rel(f"build/config_infer_primary_rfdetr_seg_{size_norm}.ini"),
                    "engine": _rel(f"models/engines/rfdetr_seg_{size_norm}_{resolution}_b3_fp16.engine"),
                }
            },
        }
    if profile == "wholebody49":
        mode = "boxes" if size_norm == "x" else "masks"
        engine = (
            "models/engines/deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine"
            if size_norm == "x"
            else "models/engines/deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine"
        )
        return {
            "preprocess": {"config-file": _rel(f"build/config_preproc_wholebody49_{size_norm}_b3.ini")},
            "models": {
                "pgie": {
                    "config-file-path": _rel(f"build/config_infer_primary_wholebody49_{size_norm}.ini"),
                    "engine": _rel(engine),
                    "wholebody49_mode": mode,
                }
            },
        }
    raise ValueError(f"unknown PGIE profile: {profile}")


def _apply_tracking_overlay(cfg: Dict[str, Any], tracking_mode: str) -> Dict[str, Any]:
    mode = str(tracking_mode or "baseline").strip().lower()
    if mode == "v3dt":
        return deep_merge(cfg, {"models": {"depth_tracking": {"enable": False}}})
    return deep_merge(
        cfg,
        {
            "models": {
                "depth_tracking": {
                    "enable": True,
                    "name": "depth_tracking_fullframe",
                    "config-file-path": "build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini",
                    "engine": "models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
                    "batch_size": 3,
                    "gie_id": 5,
                    "attach_tensor_meta": True,
                }
            }
        },
    )


def _canonical_path(base_yaml_path: Path, raw: Any) -> str:
    if raw in (None, ""):
        return str(raw or "")
    text = str(raw)
    if "://" in text:
        return text
    return str(resolve_config_path(base_yaml_path, text))


def canonicalize_runtime_paths(cfg: Dict[str, Any], base_yaml_path: Path) -> Dict[str, Any]:
    effective = deepcopy(cfg)

    depth_registration = effective.get("depth_registration")
    if isinstance(depth_registration, dict) and depth_registration.get("path"):
        depth_registration["path"] = _canonical_path(base_yaml_path, depth_registration.get("path"))

    preprocess = effective.get("preprocess")
    if isinstance(preprocess, dict) and preprocess.get("config-file"):
        preprocess["config-file"] = _canonical_path(base_yaml_path, preprocess.get("config-file"))

    models = effective.get("models")
    if isinstance(models, dict):
        for model in models.values():
            if not isinstance(model, dict):
                continue
            for key in ("config-file-path", "engine"):
                if model.get(key):
                    model[key] = _canonical_path(base_yaml_path, model.get(key))

    tracker = effective.get("tracker")
    if isinstance(tracker, dict):
        for key in ("config-file", "ll-lib-file"):
            if tracker.get(key) and str(tracker.get(key)).startswith(("config/", "pipelines/", "models/", "build/")):
                tracker[key] = _canonical_path(base_yaml_path, tracker.get(key))

    analytics = effective.get("analytics")
    if isinstance(analytics, dict):
        for key in ("config-file", "stages_config"):
            if analytics.get(key):
                analytics[key] = _canonical_path(base_yaml_path, analytics.get(key))
        exclude = analytics.get("exclude")
        if isinstance(exclude, dict) and exclude.get("config-file"):
            exclude["config-file"] = _canonical_path(base_yaml_path, exclude.get("config-file"))

    sources = effective.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            dewarper = source.get("dewarper")
            if isinstance(dewarper, dict) and dewarper.get("config-file"):
                dewarper["config-file"] = _canonical_path(base_yaml_path, dewarper.get("config-file"))
    return effective


def build_effective_config(spec: LaunchSpec) -> Dict[str, Any]:
    base_path = spec.pipeline_path
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError(f"pipeline YAML must be a mapping: {base_path}")
    effective = deep_merge(base_cfg, profile_overlay(spec.pgie_profile, spec.size))
    effective = _apply_tracking_overlay(effective, spec.tracking_mode)
    effective = deep_merge(effective, {"mosaic_output": {"rtsp_port": int(spec.rtsp_port)}})

    pgie = ((effective.get("models") or {}).get("pgie") or {}) if isinstance(effective.get("models"), dict) else {}
    pgie_ini_raw = pgie.get("config-file-path") if isinstance(pgie, Mapping) else None
    if pgie_ini_raw:
        pgie_ini = resolve_config_path(base_path, pgie_ini_raw)
        if pgie_ini.exists():
            effective["osd"] = derive_osd_policy_from_ini(parse_nvinfer_ini(pgie_ini))
    return canonicalize_runtime_paths(effective, base_path)


def materialize_launch_pipeline(spec: LaunchSpec, *, dry_run: bool = False) -> Path:
    launch_dir = spec.launch_dir
    launch_dir.mkdir(parents=True, exist_ok=True)
    effective = build_effective_config(spec)
    suffix = f"{spec.pgie_profile}_{spec.size}" if spec.size else spec.pgie_profile
    out_path = launch_dir / f"effective_pipeline_{suffix}.yaml"
    out_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")

    materialized_rel = str(out_path.relative_to(REPO_ROOT)) if out_path.is_relative_to(REPO_ROOT) else str(out_path)
    spec.materialized_pipeline = materialized_rel

    validations = [
        item.to_dict()
        for item in validate_metadata_compatibility(
            effective,
            out_path,
            tracking_mode=spec.tracking_mode,
            strict_baseline=spec.strict_baseline,
        )
    ]
    manifest = {
        "launch_id": spec.launch_id,
        "dry_run": bool(dry_run),
        "pipeline": materialized_rel,
        "base_pipeline": spec.pipeline_config,
        "pgie_profile": spec.pgie_profile,
        "size": spec.size,
        "tracking_mode": spec.tracking_mode,
        "validations": validations,
    }
    (launch_dir / "launch_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (launch_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
