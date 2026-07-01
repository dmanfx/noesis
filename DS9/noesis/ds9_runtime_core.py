#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import configparser
import ctypes
import copy
import inspect
import json
import logging
import os
import re
import subprocess
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Mapping

import yaml
import numpy as np

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(DS9_ROOT), str(REPO_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DS9_ROOT))


def _agent_debug_log_path() -> Path:
    return Path(os.environ.get("NOESIS_AGENT_DEBUG_LOG", REPO_ROOT / ".cursor" / "debug.log")).expanduser()


from calibration_bundle import (
    assemble_calibration_bundle,
    load_alignment,
    load_extrinsics,
    load_intrinsics,
    pose_to_E_col_major,
    save_alignment,
    save_extrinsics,
)
from geometry.depth_source import DepthStorageManager
from mapanything_config import load_service_config
from noesis.calibration.depth_registration import (
    DepthRegistrationError,
    DepthRegistrationManager,
    model_profile_fingerprint as _depth_registration_model_profile_fingerprint,
)
from noesis.calibration.manager import CalibrationManager
from noesis.calibration.pose_v1 import normalize_pose_v1
from noesis.calibration.pose_v1 import POSE_V1_FRAME_BACKEND_WORLD_M
from noesis.depth_tracking_materialization import (
    DEFAULT_BATCH_SIZE as _DEPTH_TRACKING_BATCH_SIZE,
    DEFAULT_GIE_ID as _DEPTH_TRACKING_GIE_ID,
    DEFAULT_INPUT_SIZE as _DEPTH_TRACKING_INPUT_SIZE,
    DEFAULT_INTERVAL as _DEPTH_TRACKING_INTERVAL,
    ensure_native_depth_tracking_tensor_extension as _ensure_native_depth_tracking_tensor_extension,
    ensure_native_object_depth_extension as _ensure_native_object_depth_extension,
    materialize_depth_tracking_assets as _materialize_depth_tracking_assets,
)
from noesis.pipelines import ds8_pipeline, hooks
from noesis.metadata.intrinsics import CameraConfigLoader
from noesis.telemetry.publishers import DepthTelemetryPublisher, TrackingTelemetryPublisher, bind_occupancy_publisher
from noesis.diagnostics.telemetry_log import TrackingDiagnosticsLogger
from noesis.telemetry.bev import BevRenderer, CalibrationSnapshot
from noesis.yolo26_seg_materialization import (
    materialize_yolo26_seg_configs as _shared_materialize_yolo26_seg_configs,
)
from noesis.yolo26_seg_materialization import resolve_yolo26_seg_assets as _shared_resolve_yolo26_seg_assets
from websocket_server import WebSocketServer

# GLib/GObject for GStreamer main loop (required for bus event dispatch)
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import GLib, Gst
    _GLIB_AVAILABLE = True
except Exception:
    GLib = None  # type: ignore
    Gst = None  # type: ignore
    _GLIB_AVAILABLE = False

# pyservicemaker message types for pipeline event handling
try:
    import pyservicemaker._pydeepstream as _pyds
    PipelineMessage = _pyds.PipelineMessage
    EOSMessage = _pyds.EOSMessage
    StateTransitionMessage = _pyds.StateTransitionMessage
    _PYSERVICEMAKER_MSGS = True
except Exception:
    PipelineMessage = object  # type: ignore
    EOSMessage = object  # type: ignore
    StateTransitionMessage = object  # type: ignore
    _PYSERVICEMAKER_MSGS = False


_PGIE_PROFILES = ("yolo11_seg", "yolo11", "yolo26_seg", "yolo26", "rfdetr_seg", "rfdetr")
_SIZED_PGIE_PROFILES = ("yolo26_seg", "yolo26", "rfdetr_seg", "rfdetr")
_YOLO26_DETECT_SIZES = ("n", "s", "m", "l", "x")
_YOLO26_SEG_SIZES = ("n", "s", "m")
_RFDETR_SIZES = ("n", "s", "m")
_YOLO26_DETECT_SIZE_HELP = "/".join(_YOLO26_DETECT_SIZES)
_YOLO26_SEG_SIZE_HELP = "/".join(_YOLO26_SEG_SIZES)
_RFDETR_SIZE_HELP = "/".join(_RFDETR_SIZES)
_ENV_TRUE = ("1", "true", "yes", "y", "on")
_TRACKING_MODES = ("baseline", "v3dt")
_RFDETR_TRT_PLUGIN_LOADED = False


def _artifact_dir(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, default)).expanduser().resolve()


def _model_dir() -> Path:
    return _artifact_dir("NOESIS_MODEL_DIR", DS9_ROOT / "models")


def _onnx_dir() -> Path:
    return _artifact_dir("NOESIS_ONNX_DIR", _model_dir() / "onnx")


def _engine_dir() -> Path:
    return _artifact_dir("NOESIS_ENGINE_DIR", _model_dir() / "engines")


def _pipeline_dir() -> Path:
    return _artifact_dir("NOESIS_PIPELINE_DIR", DS9_ROOT / "pipelines")


def _build_dir() -> Path:
    return _artifact_dir("NOESIS_BUILD_DIR", DS9_ROOT / "build")


def _path_scope_for_yaml(yaml_path: Path) -> Path:
    try:
        yaml_path.expanduser().resolve().relative_to(DS9_ROOT.resolve())
        return DS9_ROOT
    except Exception:
        return REPO_ROOT


def _deep_merge_dict(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = _deep_merge_dict(merged.get(key), value) if key in merged else value
        return merged
    return overlay


def _resolve_pipeline_cfg_path(yaml_path: Path, raw: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        return Path("")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    base_dir = yaml_path.parent.resolve()
    if value.startswith("DS9/"):
        return (REPO_ROOT / candidate).resolve()
    if value.startswith(("config/", "models/", "pipelines/", "build/")):
        return (_path_scope_for_yaml(yaml_path) / candidate).resolve()
    return (base_dir / candidate).resolve()


def _resolve_pipeline_path_value(yaml_path: Path, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    return str(_resolve_pipeline_cfg_path(yaml_path, raw))


def _canonicalize_effective_pipeline_paths(cfg: Dict[str, Any], base_yaml_path: Path) -> Dict[str, Any]:
    effective = copy.deepcopy(cfg)

    depth_registration = effective.get("depth_registration")
    if isinstance(depth_registration, dict) and str(depth_registration.get("path") or "").strip():
        depth_registration["path"] = _resolve_pipeline_path_value(base_yaml_path, depth_registration["path"])

    sources = effective.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            dewarper = source.get("dewarper")
            if isinstance(dewarper, dict) and str(dewarper.get("config-file") or "").strip():
                dewarper["config-file"] = _resolve_pipeline_path_value(base_yaml_path, dewarper["config-file"])

    preprocess = effective.get("preprocess")
    if isinstance(preprocess, dict) and str(preprocess.get("config-file") or "").strip():
        preprocess["config-file"] = _resolve_pipeline_path_value(base_yaml_path, preprocess["config-file"])

    models = effective.get("models")
    if isinstance(models, dict):
        for model_cfg in models.values():
            if not isinstance(model_cfg, dict):
                continue
            for key in ("config-file-path", "engine"):
                if str(model_cfg.get(key) or "").strip():
                    model_cfg[key] = _resolve_pipeline_path_value(base_yaml_path, model_cfg[key])

    tracker = effective.get("tracker")
    if isinstance(tracker, dict) and str(tracker.get("config-file") or "").strip():
        tracker["config-file"] = _resolve_pipeline_path_value(base_yaml_path, tracker["config-file"])

    analytics = effective.get("analytics")
    if isinstance(analytics, dict):
        for key in ("config-file", "stages_config"):
            if str(analytics.get(key) or "").strip():
                analytics[key] = _resolve_pipeline_path_value(base_yaml_path, analytics[key])
        exclude = analytics.get("exclude")
        if isinstance(exclude, dict) and str(exclude.get("config-file") or "").strip():
            exclude["config-file"] = _resolve_pipeline_path_value(base_yaml_path, exclude["config-file"])

    return effective


def _parse_dewarper_dst_intrinsics(
    config_path: Path, logger: logging.Logger
) -> Optional[Tuple[float, float, float, float, Optional[int], Optional[int]]]:
    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Dewarp intrinsics check failed: unable to read %s: %s", config_path, exc)
        return None

    dst_focal: Optional[Tuple[float, float]] = None
    dst_pp: Optional[Tuple[float, float]] = None
    dst_width: Optional[int] = None
    dst_height: Optional[int] = None

    for line in text.splitlines():
        raw = line.split("#", 1)[0].strip()
        if not raw:
            continue
        if raw.startswith("dst-focal-length="):
            parts = raw.split("=", 1)[1].split(";")
            if len(parts) >= 2:
                try:
                    dst_focal = (float(parts[0].strip()), float(parts[1].strip()))
                except Exception:
                    dst_focal = None
        elif raw.startswith("dst-principal-point="):
            parts = raw.split("=", 1)[1].split(";")
            if len(parts) >= 2:
                try:
                    dst_pp = (float(parts[0].strip()), float(parts[1].strip()))
                except Exception:
                    dst_pp = None
        elif raw.startswith("output-width="):
            try:
                dst_width = int(float(raw.split("=", 1)[1].strip()))
            except Exception:
                dst_width = None
        elif raw.startswith("output-height="):
            try:
                dst_height = int(float(raw.split("=", 1)[1].strip()))
            except Exception:
                dst_height = None

    if dst_focal and dst_pp:
        return dst_focal[0], dst_focal[1], dst_pp[0], dst_pp[1], dst_width, dst_height
    return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        try:
            parsed = int(float(value))
        except Exception:
            return None
    return parsed if parsed > 0 else None


def _validate_dewarper_intrinsics_sync(
    pipeline_path: Path, cameras_path: Path, logger: logging.Logger
) -> bool:
    try:
        pipeline_cfg = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("Dewarp intrinsics check failed: unable to read %s: %s", pipeline_path, exc)
        return False

    sources = pipeline_cfg.get("sources")
    if not isinstance(sources, list):
        return True

    if not cameras_path.exists():
        logger.warning("Dewarp intrinsics check skipped: cameras config missing: %s", cameras_path)
        return True

    streammux_cfg = pipeline_cfg.get("streammux") if isinstance(pipeline_cfg, dict) else None
    mux_width = _coerce_positive_int((streammux_cfg or {}).get("width")) if isinstance(streammux_cfg, dict) else None
    mux_height = _coerce_positive_int((streammux_cfg or {}).get("height")) if isinstance(streammux_cfg, dict) else None

    loader = CameraConfigLoader(cameras_path)
    ok = True
    tol_abs = 1e-2
    tol_rel_scaled = 0.015

    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        dewarp_cfg = source.get("dewarper")
        if not (isinstance(dewarp_cfg, dict) and bool(dewarp_cfg.get("enable", False))):
            continue

        config_raw = str(dewarp_cfg.get("config-file") or "").strip()
        if not config_raw:
            logger.error(
                "Dewarp intrinsics check failed: source %d has dewarper enabled but no config-file",
                idx,
            )
            ok = False
            continue

        config_path = _resolve_pipeline_cfg_path(pipeline_path, config_raw)
        if not config_path.exists():
            logger.error(
                "Dewarp intrinsics check failed: dewarper config not found for source %d: %s",
                idx,
                config_path,
            )
            ok = False
            continue

        dst = _parse_dewarper_dst_intrinsics(config_path, logger)
        if dst is None:
            logger.warning(
                "Dewarp intrinsics check skipped: dst-focal-length/principal-point missing in %s",
                config_path,
            )
            continue

        intr = loader.get(idx)
        if intr is None:
            logger.error(
                "Dewarp intrinsics check failed: no intrinsics for source %d in %s",
                idx,
                cameras_path,
            )
            ok = False
            continue

        fx, fy, cx, cy, dst_width, dst_height = dst
        scaled = False
        if (
            dst_width
            and dst_height
            and mux_width
            and mux_height
            and (dst_width != mux_width or dst_height != mux_height)
        ):
            # Streammux scaling changes the effective intrinsics seen by SV3DT.
            scale_x = mux_width / float(dst_width)
            scale_y = mux_height / float(dst_height)
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y
            scaled = True

        tol_rel = tol_rel_scaled if scaled else 0.0

        def _within_tol(expected: float, actual: float) -> bool:
            return abs(expected - actual) <= max(tol_abs, tol_rel * max(abs(expected), abs(actual)))

        expected = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        actual = {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}
        diffs = {key: abs(expected[key] - actual[key]) for key in expected}
        mismatch = {key: val for key, val in diffs.items() if not _within_tol(expected[key], actual[key])}
        if mismatch:
            scale_note = ""
            if scaled:
                scale_note = (
                    f" scaled_from={dst_width}x{dst_height} to={mux_width}x{mux_height}"
                    f" tol_rel={tol_rel:.3f}"
                )
            logger.error(
                "Dewarp intrinsics mismatch for source %d (tol_abs=%.4f): "
                "dewarper dst=(fx=%.6f, fy=%.6f, cx=%.6f, cy=%.6f) "
                "cameras=(fx=%.6f, fy=%.6f, cx=%.6f, cy=%.6f)%s diffs=%s "
                "config=%s cameras=%s",
                idx,
                tol_abs,
                fx,
                fy,
                cx,
                cy,
                intr.fx,
                intr.fy,
                intr.cx,
                intr.cy,
                scale_note,
                mismatch,
                config_path,
                cameras_path,
            )
            ok = False

    return ok


def _newest_model_artifact(
    pattern: str,
    *,
    excluded_tokens: Tuple[str, ...] = (),
    root: Optional[Path] = None,
) -> Optional[Path]:
    models_dir = (root or _model_dir()).resolve()
    if not models_dir.exists():
        return None
    candidates = []
    for path in models_dir.rglob(pattern):
        name = path.name.lower()
        if any(token in name for token in excluded_tokens):
            continue
        if not path.is_file():
            continue
        candidates.append(path.resolve())
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.stat().st_mtime_ns, str(item)))


def _family_artifact_summary(*patterns: str, limit: int = 8) -> str:
    models_dir = _model_dir().resolve()
    if not models_dir.exists():
        return "<models directory missing>"
    matches = []
    seen = set()
    for pattern in patterns:
        for path in models_dir.rglob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(resolved)
    if not matches:
        return "<none>"
    newest = sorted(matches, key=lambda item: (item.stat().st_mtime_ns, str(item)), reverse=True)
    return "\n".join(f"  - {path}" for path in newest[:limit])


def _resolve_yolo_detect_assets(profile: str, size: Optional[str]) -> Dict[str, Any]:
    profile_norm = str(profile or "").strip().lower()
    if profile_norm == "yolo11":
        label = "YOLO11"
        family_prefix = "yolo11"
        size_norm = ""
        tensor_name = "input"
    elif profile_norm == "yolo26":
        size_norm = str(size or "").strip().lower()
        if size_norm not in _YOLO26_DETECT_SIZES:
            raise SystemExit(f"[FATAL] YOLO26 detection size must be one of {_YOLO26_DETECT_SIZE_HELP} (got: {size})")
        label = f"YOLO26 {size_norm}"
        family_prefix = f"yolo26{size_norm}"
        tensor_name = "images"
    else:
        raise SystemExit(f"[FATAL] Unsupported YOLO detection profile: {profile}")

    excluded = ("seg", "pose")
    onnx_path = _newest_model_artifact(f"{family_prefix}*.onnx", excluded_tokens=excluded, root=_model_dir())
    if onnx_path is None:
        observed = _family_artifact_summary(f"{family_prefix}*")
        raise SystemExit(
            f"[FATAL] {label} detection ONNX not found under {_model_dir()}.\n"
            f"Looked for {family_prefix}*.onnx excluding seg/pose variants.\n"
            f"Current matching artifacts:\n{observed}\n"
            f"Use {profile_norm}_seg for the current segmentation assets, or add a detector ONNX/engine pair."
        )

    engine_path = _newest_model_artifact(f"{family_prefix}*.engine", excluded_tokens=excluded, root=_engine_dir())
    if engine_path is None:
        engine_path = (_engine_dir() / f"{onnx_path.stem}_b3_fp16.engine").resolve()

    size_suffix = f"_{size_norm}" if size_norm else ""
    return {
        "label": label,
        "template": (_pipeline_dir() / "config_infer_primary_yolo11.ini").resolve(),
        "preprocess_template": (_pipeline_dir() / "config_preproc.ini").resolve(),
        "preprocess_output": (_build_dir() / f"config_preproc_{profile_norm}{size_suffix}.ini").resolve(),
        "tensor_name": tensor_name,
        "onnx": onnx_path,
        "engine": engine_path,
        "output": (_build_dir() / f"config_infer_primary_{profile_norm}{size_suffix}.ini").resolve(),
    }


def _resolve_yolo_detect_parser_override() -> Optional[Path]:
    raw = str(os.environ.get("NOESIS_YOLO_DETECT_PARSER_LIB", "") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(
            "[FATAL] NOESIS_YOLO_DETECT_PARSER_LIB points to a missing/empty YOLO detector parser.\n"
            f"NOESIS_YOLO_DETECT_PARSER_LIB={raw}\n"
            f"resolved: {path}"
        )
    return path


def _resolve_yolo_detect_parser_path() -> Path:
    override = _resolve_yolo_detect_parser_override()
    if override is not None:
        return override
    return (_pipeline_dir() / "nvdsinfer_yolo_detect" / "libnvdsparsebbox_yolo.so").resolve()


def _materialize_yolo_detect_preproc_ini(assets: Dict[str, Any], logger: logging.Logger) -> Path:
    template_path = assets["preprocess_template"]
    if not template_path.exists():
        raise SystemExit(f"[FATAL] YOLO detection preprocess template missing: {template_path}")

    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "network-input-shape": "3;3;640;640",
        "processing-width": "640",
        "processing-height": "640",
        "tensor-name": str(assets["tensor_name"]),
    }
    for key, value in replacements.items():
        text, count = re.subn(rf"(?m)^{re.escape(key)}=.*$", f"{key}={value}", text, count=1)
        if count != 1:
            raise SystemExit(f"[FATAL] YOLO detection preprocess template is missing {key}: {template_path}")

    out_path = assets["preprocess_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info(
        "%s detector preprocess config materialized: %s (tensor-name=%s)",
        assets["label"],
        out_path,
        assets["tensor_name"],
    )
    return out_path


def _materialize_yolo_detect_pgie_ini(profile: str, size: Optional[str], logger: logging.Logger) -> Dict[str, Any]:
    assets = _resolve_yolo_detect_assets(profile, size)
    template_path = assets["template"]
    if not template_path.exists():
        raise SystemExit(f"[FATAL] YOLO detection PGIE template missing: {template_path}")

    text = template_path.read_text(encoding="utf-8")
    text, onnx_count = re.subn(
        r"(?m)^onnx-file=.*$",
        f"onnx-file={assets['onnx']}",
        text,
        count=1,
    )
    text, engine_count = re.subn(
        r"(?m)^model-engine-file=.*$",
        f"model-engine-file={assets['engine']}",
        text,
        count=1,
    )
    parser_path = _resolve_yolo_detect_parser_path()
    text, lib_count = re.subn(
        r"(?m)^custom-lib-path=.*$",
        f"custom-lib-path={parser_path}",
        text,
        count=1,
    )
    if lib_count != 1:
        raise SystemExit(f"[FATAL] YOLO detection PGIE template is missing custom-lib-path: {template_path}")
    if onnx_count != 1 or engine_count != 1:
        raise SystemExit(f"[FATAL] YOLO detection PGIE template is missing onnx-file/model-engine-file: {template_path}")

    out_path = assets["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("%s detector PGIE config materialized: %s", assets["label"], out_path)
    preproc_path = _materialize_yolo_detect_preproc_ini(assets, logger)
    return {**assets, "pgie_config": out_path, "preprocess_config": preproc_path}


def _resolve_rfdetr_detect_assets(size: str) -> Dict[str, Any]:
    size_norm = str(size or "").strip().lower()
    if size_norm not in ("n", "s", "m"):
        raise SystemExit(f"[FATAL] RF-DETR detection size must be one of n/s/m (got: {size})")
    model_info = {
        "n": {"model": "rf-detr-nano", "weights": "rf-detr-nano.pth", "resolution": 384, "max_detections": 30},
        "s": {"model": "rf-detr-small", "weights": "rf-detr-small.pth", "resolution": 512, "max_detections": 50},
        "m": {"model": "rf-detr-medium", "weights": "rf-detr-medium.pth", "resolution": 576, "max_detections": 80},
    }[size_norm]
    resolution = int(model_info["resolution"])
    return {
        "model": model_info["model"],
        "resolution": resolution,
        "max_detections": int(model_info["max_detections"]),
        "template": (_pipeline_dir() / "config_infer_primary_rfdetr.template.ini").resolve(),
        "preproc": (_pipeline_dir() / f"config_preproc_rfdetr_detect_{resolution}.ini").resolve(),
        "weights": (REPO_ROOT / "models" / str(model_info["weights"])).resolve(),
        "onnx": (_onnx_dir() / f"rfdetr_{size_norm}_{resolution}.onnx").resolve(),
        "engine": (_engine_dir() / f"rfdetr_{size_norm}_{resolution}_b3_fp16.engine").resolve(),
        "output": (_build_dir() / f"config_infer_primary_rfdetr_{size_norm}.ini").resolve(),
    }


def _materialize_rfdetr_detect_pgie_ini(size: str, logger: logging.Logger) -> Path:
    assets = _resolve_rfdetr_detect_assets(size)
    template_path = assets["template"]
    if not template_path.exists():
        raise SystemExit(f"[FATAL] RF-DETR detection PGIE template missing: {template_path}")
    preproc_path = assets["preproc"]
    if not preproc_path.exists():
        raise SystemExit(f"[FATAL] RF-DETR detection preprocess config missing: {preproc_path}")
    out_path = assets["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = template_path.read_text(encoding="utf-8")
    text = text.replace("@ONNX_PATH@", str(assets["onnx"]))
    text = text.replace("@ENGINE_PATH@", str(assets["engine"]))
    text = text.replace("@TOPK@", str(assets["max_detections"]))
    text = re.sub(
        r"(?m)^labelfile-path=.*$",
        f"labelfile-path={(_model_dir() / 'coco_labels.txt').resolve()}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^custom-lib-path=.*$",
        f"custom-lib-path={(_pipeline_dir() / 'nvdsinfer_rfdetr' / 'libnvdsinfer_rfdetr.so').resolve()}",
        text,
        count=1,
    )
    out_path.write_text(text, encoding="utf-8")
    logger.info("RF-DETR detection PGIE config materialized: %s", out_path)
    return out_path


def _resolve_rfdetr_assets(size: str) -> Dict[str, Any]:
    size_norm = str(size or "").strip().lower()
    if size_norm not in ("n", "s", "m"):
        raise SystemExit(f"[FATAL] RF-DETR size must be one of n/s/m (got: {size})")

    model_info = {
        "n": {"model": "rfdetr-seg-nano", "resolution": 312, "max_detections": 10},
        "s": {"model": "rfdetr-seg-small", "resolution": 384, "max_detections": 20},
        "m": {"model": "rfdetr-seg-medium", "resolution": 432, "max_detections": 30},
    }[size_norm]
    resolution = int(model_info["resolution"])
    return {
        "model": model_info["model"],
        "resolution": resolution,
        "max_detections": int(model_info["max_detections"]),
        "template": (_pipeline_dir() / "config_infer_primary_rfdetr_seg.template.ini").resolve(),
        "preproc": (_pipeline_dir() / f"config_preproc_rfdetr_{resolution}.ini").resolve(),
        "weights": (REPO_ROOT / "models" / f"rf-detr-seg-{size_norm}.pt").resolve(),
        "onnx": (_onnx_dir() / f"rfdetr_seg_{size_norm}_{resolution}.onnx").resolve(),
        "engine": (_engine_dir() / f"rfdetr_seg_{size_norm}_{resolution}_b3_fp16.engine").resolve(),
        "output": (_build_dir() / f"config_infer_primary_rfdetr_seg_{size_norm}.ini").resolve(),
    }


def _materialize_rfdetr_pgie_ini(size: str, logger: logging.Logger) -> Path:
    assets = _resolve_rfdetr_assets(size)
    template_path = assets["template"]
    if not template_path.exists():
        raise SystemExit(f"[FATAL] RF-DETR PGIE template missing: {template_path}")
    preproc_path = assets["preproc"]
    if not preproc_path.exists():
        raise SystemExit(f"[FATAL] RF-DETR preprocess config missing: {preproc_path}")
    out_path = assets["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = template_path.read_text(encoding="utf-8")
    text = text.replace("@ONNX_PATH@", str(assets["onnx"]))
    text = text.replace("@ENGINE_PATH@", str(assets["engine"]))
    text = text.replace("@TOPK@", str(assets["max_detections"]))
    text = re.sub(
        r"(?m)^labelfile-path=.*$",
        f"labelfile-path={(_model_dir() / 'coco_labels.txt').resolve()}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^custom-lib-path=.*$",
        f"custom-lib-path={(_pipeline_dir() / 'nvdsinfer_rfdetr_seg' / 'libnvdsinfer_rfdetr_seg.so').resolve()}",
        text,
        count=1,
    )
    out_path.write_text(text, encoding="utf-8")
    logger.info("RF-DETR PGIE config materialized: %s", out_path)
    return out_path


def _resolve_yolo26_assets(size: str) -> Dict[str, Path]:
    try:
        return _shared_resolve_yolo26_assets(size)
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _materialize_yolo26_configs(size: str, src_ids: Tuple[int, ...], logger: logging.Logger) -> Dict[str, Path]:
    try:
        return _shared_materialize_yolo26_seg_configs(
            size=size,
            batch_size=3,
            src_ids=src_ids,
            logger=logger,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _load_rfdetr_trt_plugin_library(yaml_path: Path, logger: logging.Logger) -> None:
    global _RFDETR_TRT_PLUGIN_LOADED
    if _RFDETR_TRT_PLUGIN_LOADED:
        return

    env_path = str(os.environ.get("NOESIS_RFDETR_TRT_PLUGIN_LIB", "") or "").strip()
    if env_path:
        lib_path = _resolve_pipeline_cfg_path(yaml_path, env_path)
    else:
        lib_path = (
            REPO_ROOT
            / "external"
            / "DeepStream-Yolo-Seg"
            / "nvdsinfer_custom_impl_Yolo_seg"
            / "libnvdsinfer_custom_impl_Yolo_seg.so"
        ).resolve()

    if not lib_path.exists():
        raise SystemExit(
            "[FATAL] RF-DETR TensorRT plugin library missing.\n"
            f"resolved: {lib_path}\n"
            "Set NOESIS_RFDETR_TRT_PLUGIN_LIB to override, or build with:\n"
            "  make -C external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg\n"
        )

    try:
        import tensorrt as trt  # type: ignore

        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    except Exception as exc:
        logger.debug("TensorRT standard plugin init skipped: %s", exc)

    try:
        ctypes.CDLL(str(lib_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except Exception as exc:
        raise SystemExit(
            "[FATAL] Failed to load RF-DETR TensorRT plugin library.\n"
            f"resolved: {lib_path}\n"
            f"error: {exc}"
        ) from exc

    _RFDETR_TRT_PLUGIN_LOADED = True
    logger.info("RF-DETR TensorRT plugin library loaded: %s", lib_path)


def _preflight_pgie_profile(profile: str, pipeline_cfg: Dict[str, Any], yaml_path: Path, logger: logging.Logger) -> None:
    if profile not in ("yolo11", "yolo26", "rfdetr", "rfdetr_seg", "yolo26_seg"):
        return

    if profile in ("yolo11", "yolo26"):
        label = str(profile).upper()
        preprocess_cfg = pipeline_cfg.get("preprocess") if isinstance(pipeline_cfg, dict) else None
        preprocess_path_raw = (preprocess_cfg or {}).get("config-file") if isinstance(preprocess_cfg, dict) else None
        preprocess_path = _resolve_pipeline_cfg_path(yaml_path, str(preprocess_path_raw or ""))
        if not preprocess_path.exists():
            raise SystemExit(f"[FATAL] {label} detection profile requires preprocess config-file at: {preprocess_path}")

        preproc_parser = configparser.ConfigParser()
        preproc_parser.read(preprocess_path, encoding="utf-8")
        preproc_props = preproc_parser["property"] if preproc_parser.has_section("property") else {}
        tensor_name = str(preproc_props.get("tensor-name", "") or "").strip()
        expected_tensor_name = "images" if profile == "yolo26" else "input"
        if tensor_name != expected_tensor_name:
            raise SystemExit(
                f"[FATAL] {label} detection preprocess tensor-name must be {expected_tensor_name!r} "
                f"for the selected detector engine (got {tensor_name!r} in {preprocess_path})"
            )

        models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
        pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
        pgie_ini_raw = (pgie_cfg or {}).get("config-file-path") if isinstance(pgie_cfg, dict) else None
        pgie_ini = _resolve_pipeline_cfg_path(yaml_path, str(pgie_ini_raw or ""))
        if not pgie_ini.exists():
            raise SystemExit(f"[FATAL] {label} detection profile requires PGIE config-file-path at: {pgie_ini}")

        engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None
        engine_path = _resolve_pipeline_cfg_path(yaml_path, str(engine_raw or ""))
        if not str(engine_raw or "").strip():
            raise SystemExit(f"[FATAL] {label} detection profile requires models.pgie.engine to be set")

        parser = configparser.ConfigParser()
        parser.read(pgie_ini, encoding="utf-8")
        props = parser["property"] if parser.has_section("property") else {}

        lib_raw = str(props.get("custom-lib-path", "") or "").strip()
        lib_path = _resolve_pipeline_cfg_path(yaml_path, lib_raw)
        if not lib_raw or not lib_path.exists():
            raise SystemExit(
                f"[FATAL] {label} detection PGIE custom parser library missing.\n"
                f"PGIE INI: {pgie_ini}\n"
                f"custom-lib-path: {lib_raw or '<unset>'}\n"
                f"resolved: {lib_path}\n"
                "Build/stage a DS9-compatible parser that exports NvDsInferParseYolo, "
                "or set NOESIS_YOLO_DETECT_PARSER_LIB to that library.\n"
            )

        gie_uid = str(props.get("gie-unique-id", "") or "").strip()
        if gie_uid and gie_uid != "1":
            raise SystemExit(f"[FATAL] {label} detection PGIE gie-unique-id must remain 1 (got {gie_uid})")

        network_type = str(props.get("network-type", "") or "").strip()
        if network_type and network_type != "0":
            raise SystemExit(f"[FATAL] {label} detection PGIE network-type must be 0 (got {network_type})")

        batch_size = str(props.get("batch-size", "") or "").strip()
        if batch_size and batch_size != "3":
            raise SystemExit(f"[FATAL] {label} detection PGIE batch-size must be 3 for DS9 batch (got {batch_size})")

        if engine_path.exists():
            logger.info("%s detection PGIE engine found: %s", label, engine_path)
            return

        onnx_raw = str(props.get("onnx-file", "") or "").strip()
        onnx_path = _resolve_pipeline_cfg_path(yaml_path, onnx_raw)
        if not onnx_raw or not onnx_path.exists():
            raise SystemExit(
                f"[FATAL] {label} detection PGIE engine is missing and no ONNX is available to rebuild it.\n"
                f"engine (from YAML models.pgie.engine): {engine_path}\n"
                f"onnx-file (from PGIE INI): {onnx_raw or '<unset>'}\n"
                f"resolved: {onnx_path}\n"
            )

        logger.warning(
            "%s detection PGIE engine missing (%s); nvinfer will attempt to build it from ONNX (%s) on startup.",
            label,
            engine_path,
            onnx_path,
        )
        return

    if profile == "rfdetr":
        _load_rfdetr_trt_plugin_library(yaml_path, logger)
        preprocess_cfg = pipeline_cfg.get("preprocess") if isinstance(pipeline_cfg, dict) else None
        preprocess_path_raw = (preprocess_cfg or {}).get("config-file") if isinstance(preprocess_cfg, dict) else None
        preprocess_path = _resolve_pipeline_cfg_path(yaml_path, str(preprocess_path_raw or ""))
        if not preprocess_path.exists():
            raise SystemExit(f"[FATAL] RF-DETR detection profile requires preprocess config-file at: {preprocess_path}")

        models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
        pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
        pgie_ini_raw = (pgie_cfg or {}).get("config-file-path") if isinstance(pgie_cfg, dict) else None
        pgie_ini = _resolve_pipeline_cfg_path(yaml_path, str(pgie_ini_raw or ""))
        if not pgie_ini.exists():
            raise SystemExit(f"[FATAL] RF-DETR detection profile requires PGIE config-file-path at: {pgie_ini}")

        engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None
        engine_path = _resolve_pipeline_cfg_path(yaml_path, str(engine_raw or ""))
        if not str(engine_raw or "").strip():
            raise SystemExit("[FATAL] RF-DETR detection profile requires models.pgie.engine to be set")

        parser = configparser.ConfigParser()
        parser.read(pgie_ini, encoding="utf-8")
        props = parser["property"] if parser.has_section("property") else {}

        lib_raw = str(props.get("custom-lib-path", "") or "").strip()
        lib_path = _resolve_pipeline_cfg_path(yaml_path, lib_raw)
        if not lib_raw or not lib_path.exists():
            raise SystemExit(
                "[FATAL] RF-DETR detection PGIE custom parser library missing.\n"
                f"PGIE INI: {pgie_ini}\n"
                f"custom-lib-path: {lib_raw or '<unset>'}\n"
                f"resolved: {lib_path}\n"
                "Build it with: make -C pipelines/nvdsinfer_rfdetr\n"
            )

        gie_uid = str(props.get("gie-unique-id", "") or "").strip()
        if gie_uid and gie_uid != "1":
            raise SystemExit(f"[FATAL] RF-DETR detection PGIE gie-unique-id must remain 1 (got {gie_uid})")

        if engine_path.exists():
            logger.info("RF-DETR detection PGIE engine found: %s", engine_path)
            return

        onnx_raw = str(props.get("onnx-file", "") or "").strip()
        onnx_path = _resolve_pipeline_cfg_path(yaml_path, onnx_raw)
        if not onnx_raw or not onnx_path.exists():
            raise SystemExit(
                "[FATAL] RF-DETR detection PGIE engine is missing and no ONNX is available to rebuild it.\n"
                f"engine (from YAML models.pgie.engine): {engine_path}\n"
                f"onnx-file (from PGIE INI): {onnx_raw or '<unset>'}\n"
                f"resolved: {onnx_path}\n"
            )

        logger.warning(
            "RF-DETR detection PGIE engine missing (%s); nvinfer will attempt to build it from ONNX (%s) on startup.",
            engine_path,
            onnx_path,
        )
        return

    if profile == "rfdetr_seg":
        _load_rfdetr_trt_plugin_library(yaml_path, logger)
        preprocess_cfg = pipeline_cfg.get("preprocess") if isinstance(pipeline_cfg, dict) else None
        preprocess_path_raw = (preprocess_cfg or {}).get("config-file") if isinstance(preprocess_cfg, dict) else None
        preprocess_path = _resolve_pipeline_cfg_path(yaml_path, str(preprocess_path_raw or ""))
        if not preprocess_path.exists():
            raise SystemExit(f"[FATAL] RF-DETR profile requires preprocess config-file at: {preprocess_path}")

        models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
        pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
        pgie_ini_raw = (pgie_cfg or {}).get("config-file-path") if isinstance(pgie_cfg, dict) else None
        pgie_ini = _resolve_pipeline_cfg_path(yaml_path, str(pgie_ini_raw or ""))
        if not pgie_ini.exists():
            raise SystemExit(f"[FATAL] RF-DETR profile requires PGIE config-file-path at: {pgie_ini}")

        engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None
        engine_path = _resolve_pipeline_cfg_path(yaml_path, str(engine_raw or ""))
        if not str(engine_raw or "").strip():
            raise SystemExit("[FATAL] RF-DETR profile requires models.pgie.engine to be set")

        parser = configparser.ConfigParser()
        parser.read(pgie_ini, encoding="utf-8")
        props = parser["property"] if parser.has_section("property") else {}

        lib_raw = str(props.get("custom-lib-path", "") or "").strip()
        lib_path = _resolve_pipeline_cfg_path(yaml_path, lib_raw)
        if not lib_raw or not lib_path.exists():
            raise SystemExit(
                "[FATAL] RF-DETR PGIE custom parser library missing.\n"
                f"PGIE INI: {pgie_ini}\n"
                f"custom-lib-path: {lib_raw or '<unset>'}\n"
                f"resolved: {lib_path}\n"
                "Build it with: make -C pipelines/nvdsinfer_rfdetr_seg\n"
            )

        gie_uid = str(props.get("gie-unique-id", "") or "").strip()
        if gie_uid and gie_uid != "1":
            raise SystemExit(f"[FATAL] RF-DETR PGIE gie-unique-id must remain 1 (got {gie_uid})")

        if engine_path.exists():
            logger.info("RF-DETR PGIE engine found: %s", engine_path)
            return

        onnx_raw = str(props.get("onnx-file", "") or "").strip()
        onnx_path = _resolve_pipeline_cfg_path(yaml_path, onnx_raw)
        if not onnx_raw or not onnx_path.exists():
            raise SystemExit(
                "[FATAL] RF-DETR PGIE engine is missing and no ONNX is available to rebuild it.\n"
                f"engine (from YAML models.pgie.engine): {engine_path}\n"
                f"onnx-file (from PGIE INI): {onnx_raw or '<unset>'}\n"
                f"resolved: {onnx_path}\n"
            )

        logger.warning(
            "RF-DETR PGIE engine missing (%s); nvinfer will attempt to build it from ONNX (%s) on startup.",
            engine_path,
            onnx_path,
        )
        return

    if profile == "yolo26_seg":
        preprocess_cfg = pipeline_cfg.get("preprocess") if isinstance(pipeline_cfg, dict) else None
        preprocess_path_raw = (preprocess_cfg or {}).get("config-file") if isinstance(preprocess_cfg, dict) else None
        preprocess_path = _resolve_pipeline_cfg_path(yaml_path, str(preprocess_path_raw or ""))
        if not preprocess_path.exists():
            raise SystemExit(f"[FATAL] YOLO26 profile requires preprocess config-file at: {preprocess_path}")

        models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
        pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
        pgie_ini_raw = (pgie_cfg or {}).get("config-file-path") if isinstance(pgie_cfg, dict) else None
        pgie_ini = _resolve_pipeline_cfg_path(yaml_path, str(pgie_ini_raw or ""))
        if not pgie_ini.exists():
            raise SystemExit(f"[FATAL] YOLO26 profile requires PGIE config-file-path at: {pgie_ini}")

        engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None
        engine_path = _resolve_pipeline_cfg_path(yaml_path, str(engine_raw or ""))
        if not str(engine_raw or "").strip():
            raise SystemExit("[FATAL] YOLO26 profile requires models.pgie.engine to be set")
        if not engine_path.exists():
            raise SystemExit(f"[FATAL] YOLO26 PGIE engine missing: {engine_path}")

        parser = configparser.ConfigParser()
        parser.read(pgie_ini, encoding="utf-8")
        props = parser["property"] if parser.has_section("property") else {}

        lib_raw = str(props.get("custom-lib-path", "") or "").strip()
        lib_path = _resolve_pipeline_cfg_path(yaml_path, lib_raw)
        if not lib_raw or not lib_path.exists():
            raise SystemExit(
                "[FATAL] YOLO26 PGIE custom parser library missing.\n"
                f"PGIE INI: {pgie_ini}\n"
                f"custom-lib-path: {lib_raw or '<unset>'}\n"
                f"resolved: {lib_path}\n"
                "Build it with: make -C pipelines/nvdsinfer_yolo26_seg\n"
            )

        gie_uid = str(props.get("gie-unique-id", "") or "").strip()
        if gie_uid and gie_uid != "1":
            raise SystemExit(f"[FATAL] YOLO26 PGIE gie-unique-id must remain 1 (got {gie_uid})")

        batch_size = str(props.get("batch-size", "") or "").strip()
        if batch_size and batch_size != "3":
            raise SystemExit(f"[FATAL] YOLO26 PGIE batch-size must be 3 for DS9 b3 engines (got {batch_size})")


def _materialize_effective_pipeline_yaml(
    base_yaml_path: Path,
    profile: str,
    logger: logging.Logger,
    *,
    pgie_size: Optional[str] = None,
    tracking_mode: str = "baseline",
) -> Path:
    try:
        base_cfg = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(f"[FATAL] Unable to read DS9 pipeline YAML: {base_yaml_path} ({exc})") from exc
    if not isinstance(base_cfg, dict):
        raise SystemExit(f"[FATAL] DS9 pipeline YAML must be a mapping (got {type(base_cfg).__name__}): {base_yaml_path}")

    overlay: Dict[str, Any] = {}
    if profile in ("yolo11", "yolo26"):
        if profile == "yolo26" and not pgie_size:
            raise SystemExit(f"[FATAL] YOLO26 detection profile requires --size ({_YOLO26_DETECT_SIZE_HELP})")
        size_norm = str(pgie_size).strip().lower() if profile == "yolo26" else None
        assets = _materialize_yolo_detect_pgie_ini(profile, size_norm, logger)
        overlay = {
            "preprocess": {"config-file": str(assets["preprocess_config"])},
            "models": {
                "pgie": {
                    "config-file-path": str(assets["pgie_config"]),
                    "engine": str(assets["engine"]),
                }
            },
        }
        logger.info("%s detection PGIE selected", str(assets["label"]))
    if profile == "rfdetr":
        if not pgie_size:
            raise SystemExit(f"[FATAL] RF-DETR detection profile requires --size ({_RFDETR_SIZE_HELP})")
        size_norm = str(pgie_size).strip().lower()
        assets = _resolve_rfdetr_detect_assets(size_norm)
        pgie_ini = _materialize_rfdetr_detect_pgie_ini(size_norm, logger)
        overlay = {
            "preprocess": {"config-file": str(assets["preproc"])},
            "models": {
                "pgie": {
                    "config-file-path": str(pgie_ini),
                    "engine": str(assets["engine"]),
                }
            },
        }
        logger.info("RF-DETR detection PGIE size: %s", size_norm)
    if profile == "rfdetr_seg":
        if not pgie_size:
            raise SystemExit(f"[FATAL] RF-DETR profile requires --size ({_RFDETR_SIZE_HELP})")
        size_norm = str(pgie_size).strip().lower()
        assets = _resolve_rfdetr_assets(size_norm)
        pgie_ini = _materialize_rfdetr_pgie_ini(size_norm, logger)
        overlay = {
            "preprocess": {"config-file": str(assets["preproc"])},
            "models": {
                "pgie": {
                    "config-file-path": str(pgie_ini),
                    "engine": str(assets["engine"]),
                }
            },
        }
        logger.info("RF-DETR PGIE size: %s", size_norm)
    if profile == "yolo26_seg":
        if not pgie_size:
            raise SystemExit(f"[FATAL] YOLO26 profile requires --size ({_YOLO26_SEG_SIZE_HELP})")
        size_norm = str(pgie_size).strip().lower()
        sources_cfg = base_cfg.get("sources") if isinstance(base_cfg, dict) else None
        source_count = len(sources_cfg) if isinstance(sources_cfg, list) else 0
        src_ids = tuple(range(source_count)) or (0, 1, 2)
        assets = _materialize_yolo26_configs(size_norm, src_ids, logger)
        overlay = {
            "preprocess": {"config-file": str(assets["preprocess_config"])},
            "models": {
                "pgie": {
                    "config-file-path": str(assets["pgie_config"]),
                    "engine": str(assets["engine"]),
                }
            },
        }
        logger.info("YOLO26 PGIE size: %s", size_norm)

    if str(tracking_mode).strip().lower() == "baseline":
        models_cfg = base_cfg.get("models") if isinstance(base_cfg, dict) else None
        depth_cfg = (models_cfg or {}).get("depth_tracking") if isinstance(models_cfg, dict) else None
        ds9_runtime = str(os.environ.get("NOESIS_DEEPSTREAM_MAJOR", "")).strip() == "9"
        preserve_depth_assets = (
            ds9_runtime
            and isinstance(depth_cfg, dict)
            and str(depth_cfg.get("config-file-path") or "").strip()
            and str(depth_cfg.get("engine") or "").strip()
        )
        if preserve_depth_assets:
            logger.info(
                "Preserving baseline depth-tracking assets from DS9 pipeline config: config-file-path=%s, engine=%s",
                depth_cfg.get("config-file-path"),
                depth_cfg.get("engine"),
            )
        else:
            try:
                depth_assets = _materialize_depth_tracking_assets(
                    logger=logger,
                    batch_size=_DEPTH_TRACKING_BATCH_SIZE,
                    interval=_DEPTH_TRACKING_INTERVAL,
                    input_size=_DEPTH_TRACKING_INPUT_SIZE,
                    gie_id=_DEPTH_TRACKING_GIE_ID,
                )
            except Exception as exc:
                raise SystemExit(f"[FATAL] Unable to materialize baseline depth-tracking assets: {exc}") from exc
            overlay = _deep_merge_dict(
                overlay,
                {
                    "models": {
                        "depth_tracking": {
                            "enable": True,
                            "name": "depth_tracking_fullframe",
                            "config-file-path": str(depth_assets.config_path),
                            "engine": str(depth_assets.engine_path),
                            "batch_size": int(depth_assets.batch_size),
                            "gie_id": int(depth_assets.gie_id),
                            "attach_tensor_meta": True,
                        }
                    }
                },
            )
    else:
        overlay = _deep_merge_dict(
            overlay,
            {
                "models": {
                    "depth_tracking": {
                        "enable": False,
                    }
                }
            },
        )

    effective_cfg = _deep_merge_dict(base_cfg, overlay)
    if not isinstance(effective_cfg, dict):
        raise SystemExit("[FATAL] Internal error: effective pipeline config is not a mapping")

    from noesis.ds8_preflight import apply_osd_from_pgie_ini

    preprocess_cfg = effective_cfg.get("preprocess") if isinstance(effective_cfg, dict) else None
    preprocess_path_raw = (preprocess_cfg or {}).get("config-file") if isinstance(preprocess_cfg, dict) else None
    models_cfg = effective_cfg.get("models") if isinstance(effective_cfg, dict) else None
    pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
    pgie_ini_raw = (pgie_cfg or {}).get("config-file-path") if isinstance(pgie_cfg, dict) else None
    engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None

    logger.info("PGIE profile: %s", profile)
    logger.info(
        "PGIE (effective): preprocess.config-file=%s, models.pgie.config-file-path=%s, models.pgie.engine=%s",
        preprocess_path_raw,
        pgie_ini_raw,
        engine_raw,
    )

    launch_dir_raw = str(
        os.environ.get("NOESIS_DEV_CONSOLE_LAUNCH_DIR", "")
        or os.environ.get("NOESIS_BUILD_DIR", "")
        or ""
    ).strip()
    if launch_dir_raw:
        out_dir = Path(launch_dir_raw).resolve()
    else:
        out_dir = (REPO_ROOT / "build").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"effective_pipeline_{profile}.yaml"
    effective_cfg = _canonicalize_effective_pipeline_paths(effective_cfg, base_yaml_path)
    effective_cfg = apply_osd_from_pgie_ini(effective_cfg, out_path)
    out_path.write_text(yaml.safe_dump(effective_cfg, sort_keys=False), encoding="utf-8")

    _preflight_pgie_profile(profile, effective_cfg, out_path, logger)
    return out_path


def _maybe_autogen_v3dt_caminfo(pipeline_path: Path, cameras_path: Path, logger: logging.Logger) -> bool:
    """Optionally regenerate V3DT camInfo files from current calibration.

    Controlled by `NOESIS_V3DT_AUTOGEN_CAMINFO` (default: 0).
    """
    flag = str(os.environ.get("NOESIS_V3DT_AUTOGEN_CAMINFO", "0") or "").strip().lower()
    if flag not in _ENV_TRUE:
        return True

    try:
        pipeline_cfg = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("V3DT autogen camInfo failed: unable to read %s: %s", pipeline_path, exc)
        return False

    tracker_cfg = pipeline_cfg.get("tracker") or {}
    ll_cfg_path = str((tracker_cfg or {}).get("config-file") or "").strip()
    if not ll_cfg_path.startswith("config/v3dt/"):
        logger.info(
            "V3DT autogen camInfo: skipped (tracker config not under config/v3dt/): %s",
            ll_cfg_path or "<missing>",
        )
        return True

    # Use the streammux output resolution for camInfo generation.
    #
    # Even if `nvtracker` internally rescales frames for tracking, SV3DT’s camInfo projection
    # matrix is consumed in the pixel coordinate system of the frames flowing through the
    # pipeline (i.e., streammux output). Using `tracker-width/height` here can introduce
    # non-uniform scaling (e.g., 1920×1056 vs 1920×1080) and distort SV3DT’s projected 3D
    # object model, which in practice can cause sporadic tracks and “stretched line” cuboids.
    streammux_cfg = pipeline_cfg.get("streammux") or {}
    try:
        target_w = int((streammux_cfg or {}).get("width") or 1920)
        target_h = int((streammux_cfg or {}).get("height") or 1080)
    except Exception:
        target_w, target_h = 1920, 1080

    script = (REPO_ROOT / "scripts" / "generate_v3dt_caminfo.py").resolve()
    if not script.exists():
        logger.error("V3DT autogen camInfo failed: missing %s", script)
        return False

    cmd = [
        sys.executable,
        str(script),
        "--pipeline-config",
        str(pipeline_path),
        "--cameras-config",
        str(cameras_path),
        "--target-width",
        str(int(target_w)),
        "--target-height",
        str(int(target_h)),
    ]
    logger.warning("V3DT autogen camInfo enabled; running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=dict(os.environ))
    except Exception as exc:
        logger.error("V3DT autogen camInfo failed (command error): %s", exc)
        return False

    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noesis DS9 runtime harness")

    default_pgie_profile = str(os.environ.get("NOESIS_PGIE_PROFILE", "yolo11_seg") or "").strip() or "yolo11_seg"

    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help="Path to the DS9 pipeline YAML definition.",
    )
    parser.add_argument(
        "--pgie-profile",
        choices=_PGIE_PROFILES,
        default=default_pgie_profile,
        help="PGIE profile overlay (default: yolo11_seg). Env: NOESIS_PGIE_PROFILE",
    )
    parser.add_argument(
        "--size",
        choices=_YOLO26_DETECT_SIZES,
        default=None,
        help=(
            "Model size. YOLO26 detection supports n/s/m/l/x; "
            "YOLO26 segmentation and RF-DETR profiles currently support n/s/m. Default: m."
        ),
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=None,
        help="Path to the cameras YAML used for intrinsics.",
    )
    parser.add_argument(
        "--tracking-mode",
        choices=_TRACKING_MODES,
        default=None,
        help="Tracking mode selection (baseline or v3dt). Env: NOESIS_TRACKING_MODE",
    )
    parser.add_argument(
        "--v3dt",
        dest="v3dt",
        action="store_true",
        default=False,
        help="Shortcut for --tracking-mode v3dt.",
    )
    parser.add_argument(
        "--ws-host",
        default=os.environ.get("NOESIS_WS_HOST", "0.0.0.0"),
        help="WebSocket host to bind.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=int(os.environ.get("NOESIS_WS_PORT", "6008")),
        help="WebSocket port to bind.",
    )
    parser.add_argument(
        "--rest-host",
        default=os.environ.get("NOESIS_REST_HOST", "0.0.0.0"),
        help="REST host to bind when enabled.",
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=int(os.environ.get("NOESIS_REST_PORT", "8080")),
        help="REST port to bind when enabled.",
    )
    rest_group = parser.add_mutually_exclusive_group()
    rest_group.add_argument(
        "--enable-rest",
        dest="enable_rest",
        action="store_true",
        default=True,
        help="Start the DS9 FastAPI application (depth + analytics). Enabled by default.",
    )
    rest_group.add_argument(
        "--disable-rest",
        dest="enable_rest",
        action="store_false",
        help="Disable the DS9 FastAPI application (depth + analytics).",
    )
    parser.add_argument(
        "--storage-base",
        type=Path,
        default=None,
        help="Override MapAnything snapshot base directory.",
    )
    parser.add_argument(
        "--depth-registration-config",
        type=Path,
        default=None,
        help="Path to the DS9 room-registration artifact bundle used by baseline pose+depth tracking.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("NOESIS_LOG_LEVEL", "WARNING"),
        help="Logging level (default: WARNING).",
    )
    parser.add_argument(
        "--depth-enable-seconds",
        type=int,
        default=int(os.environ.get("NOESIS_DEPTH_ENABLE_SECONDS", "0")),
        help="Enable the MapAnything depth valve for this many seconds on startup (0 to disable).",
    )
    return parser.parse_args()


def _normalize_tracking_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in ("v3dt", "sv3dt", "mv3dt", "3d"):
        return "v3dt"
    if mode in ("2d", "baseline", "standard", "default"):
        return "baseline"
    if not mode or mode == "auto":
        return "baseline"
    logging.getLogger("ds9.runtime").warning("Unknown tracking mode '%s'; defaulting to baseline", value)
    return "baseline"


def _resolve_tracking_mode(args: argparse.Namespace) -> str:
    if args.tracking_mode:
        return _normalize_tracking_mode(args.tracking_mode)
    if getattr(args, "v3dt", False):
        return "v3dt"
    env_mode = os.environ.get("NOESIS_TRACKING_MODE", "")
    if str(env_mode).strip():
        return _normalize_tracking_mode(env_mode)
    return "baseline"


def _port_bindable(host: str, port: int) -> bool:
    if int(port) <= 0:
        return True
    try:
        infos = socket.getaddrinfo(
            host,
            int(port),
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except Exception:
        infos = []
    for family, socktype, proto, _canon, sockaddr in infos:
        s = None
        try:
            s = socket.socket(family, socktype, proto)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sockaddr)
            return True
        except Exception:
            continue
        finally:
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
    return False


def _select_ws_port(host: str, requested_port: int, max_fallback_tries: int, logger: logging.Logger) -> int:
    port = int(requested_port)
    if port <= 0:
        return port
    if _port_bindable(host, port):
        return port

    tries = max(0, int(max_fallback_tries))
    for offset in range(1, tries + 1):
        candidate = port + offset
        if candidate > 65535:
            break
        if _port_bindable(host, candidate):
            logger.warning(
                "Requested WS port %s unavailable on %s; using fallback port %s",
                port,
                host,
                candidate,
            )
            return candidate
    logger.error(
        "Requested WS port %s unavailable on %s and no fallback port found within %s tries",
        port,
        host,
        tries,
    )
    return port


def _cuda_runtime_preflight() -> Tuple[bool, str]:
    try:
        import ctypes
        import ctypes.util
    except Exception as exc:
        return False, f"ctypes_unavailable:{exc}"

    cudart_path = ctypes.util.find_library("cudart")
    if not cudart_path:
        return False, "cudart_not_found"

    try:
        cudart = ctypes.CDLL(cudart_path)
    except Exception as exc:
        return False, f"cudart_load_failed:{exc}"

    try:
        cuda_get_device_count = cudart.cudaGetDeviceCount
        cuda_get_device_count.restype = ctypes.c_int
        cuda_get_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        count = ctypes.c_int(0)
        rc = int(cuda_get_device_count(ctypes.byref(count)))
        if rc != 0:
            return False, f"cudaGetDeviceCount_error:{rc}"
        if int(count.value) <= 0:
            return False, "cuda_device_count_zero"
        return True, f"cuda_device_count:{int(count.value)}"
    except Exception as exc:
        return False, f"cuda_preflight_failed:{exc}"


def _mode_default_paths(mode: str) -> Tuple[Path, Path]:
    if mode == "v3dt":
        return (
            REPO_ROOT / "config" / "infer_v3dt_baseline.yaml",
            REPO_ROOT / "config" / "cameras_v3dt_baseline.yaml",
        )
    return (DS9_ROOT / "config" / "infer.yaml", REPO_ROOT / "config" / "cameras.yaml")


def _resolve_pipeline_and_camera_paths(
    args: argparse.Namespace, tracking_mode: str, logger: logging.Logger
) -> Tuple[Path, Path]:
    default_pipeline, default_cameras = _mode_default_paths(tracking_mode)
    pipeline_src = "default"
    cameras_src = "default"

    pipeline_path = args.pipeline_config
    if pipeline_path is not None:
        pipeline_src = "cli"
    else:
        env_pipeline = os.environ.get("NOESIS_DS9_PIPELINE_CONFIG", "") or os.environ.get(
            "NOESIS_DS8_PIPELINE_CONFIG",
            "",
        )
        if str(env_pipeline).strip():
            pipeline_path = Path(env_pipeline)
            pipeline_src = "env"
        else:
            pipeline_path = default_pipeline

    cameras_path = args.cameras_config
    if cameras_path is not None:
        cameras_src = "cli"
    else:
        env_cameras = os.environ.get("NOESIS_CAMERAS_CONFIG", "")
        if str(env_cameras).strip():
            cameras_path = Path(env_cameras)
            cameras_src = "env"
        else:
            cameras_path = default_cameras

    pipeline_path = Path(pipeline_path)
    cameras_path = Path(cameras_path)
    logger.info(
        "Tracking mode '%s' resolved pipeline=%s (%s), cameras=%s (%s)",
        tracking_mode,
        pipeline_path,
        pipeline_src,
        cameras_path,
        cameras_src,
    )
    return pipeline_path, cameras_path


def _load_pipeline_config(path: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("Unable to read pipeline config %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.error("Pipeline config must be a mapping: %s", path)
        return None
    return data


def _pipeline_expects_finite_source_eos(config: Mapping[str, Any]) -> bool:
    streammux = config.get("streammux") if isinstance(config, Mapping) else None
    if not isinstance(streammux, Mapping):
        return False
    live_source = streammux.get("live-source", streammux.get("live_source", 1))
    try:
        return int(live_source) == 0
    except Exception:
        return str(live_source).strip().lower() in {"0", "false", "no", "off"}


def _allow_depthless_reid_smoke(pipeline_cfg: Mapping[str, Any]) -> bool:
    env_enabled = str(os.environ.get("NOESIS_ALLOW_DEPTHLESS_REID_SMOKE", "")).strip().lower() in _ENV_TRUE
    if not env_enabled:
        return False
    validation_cfg = pipeline_cfg.get("validation") if isinstance(pipeline_cfg, Mapping) else None
    if not isinstance(validation_cfg, Mapping) or not bool(validation_cfg.get("reid_smoke_depthless", False)):
        return False
    models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, Mapping) else None
    if not isinstance(models_cfg, Mapping):
        return False
    reid_cfg = models_cfg.get("reid")
    if not isinstance(reid_cfg, Mapping) or not bool(reid_cfg.get("enable", True)):
        return False
    depth_cfg = models_cfg.get("depth_tracking")
    return not (isinstance(depth_cfg, Mapping) and bool(depth_cfg.get("enable", False)))


def _resolve_tracker_config_path(pipeline_cfg: Dict[str, Any], pipeline_path: Path) -> Optional[Path]:
    tracker_cfg = pipeline_cfg.get("tracker") or {}
    raw = (tracker_cfg or {}).get("config-file") if isinstance(tracker_cfg, dict) else None
    if not raw:
        return None
    return _resolve_pipeline_cfg_path(pipeline_path, raw)


def _tracker_under_v3dt_dir(path: Path) -> bool:
    v3dt_root = (REPO_ROOT / "config" / "v3dt").resolve()
    try:
        path.resolve().relative_to(v3dt_root)
        return True
    except Exception:
        return False


def _validate_v3dt_tracking_guardrails(pipeline_path: Path, logger: logging.Logger) -> bool:
    pipeline_cfg = _load_pipeline_config(pipeline_path, logger)
    if pipeline_cfg is None:
        return False
    tracker_path = _resolve_tracker_config_path(pipeline_cfg, pipeline_path)
    if tracker_path is None:
        logger.error("Tracking mode 'v3dt' requires tracker.config-file to be set in %s", pipeline_path)
        return False
    if not tracker_path.exists():
        logger.error("Tracking mode 'v3dt' tracker config missing: %s", tracker_path)
        return False
    if not _tracker_under_v3dt_dir(tracker_path):
        logger.error(
            "Tracking mode 'v3dt' requires tracker config under config/v3dt/ (got %s)",
            tracker_path,
        )
        return False
    try:
        tracker_cfg = yaml.safe_load(tracker_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("Unable to read V3DT tracker config %s: %s", tracker_path, exc)
        return False
    if not isinstance(tracker_cfg, dict):
        logger.error("V3DT tracker config must be a mapping: %s", tracker_path)
        return False
    omp = tracker_cfg.get("ObjectModelProjection") or {}
    caminfo = (omp or {}).get("cameraModelFilepath") if isinstance(omp, dict) else None
    if isinstance(caminfo, (list, tuple)) and caminfo:
        return True
    if isinstance(caminfo, str) and caminfo.strip():
        return True
    logger.error(
        "Tracking mode 'v3dt' requires ObjectModelProjection.cameraModelFilepath in %s",
        tracker_path,
    )
    return False


def _warn_baseline_with_v3dt_tracker(pipeline_path: Path, logger: logging.Logger) -> None:
    pipeline_cfg = _load_pipeline_config(pipeline_path, logger)
    if pipeline_cfg is None:
        return
    tracker_path = _resolve_tracker_config_path(pipeline_cfg, pipeline_path)
    if tracker_path is None:
        return
    if _tracker_under_v3dt_dir(tracker_path):
        logger.warning(
            "Tracking mode 'baseline' with V3DT tracker config %s; V3DT meta/bbox3d will be ignored",
            tracker_path,
        )


def _ensure_v3dt_meta_extension(logger: logging.Logger) -> bool:
    try:
        import noesis_v3dt_meta_ext  # type: ignore  # noqa: F401
    except Exception as exc:
        logger.error("Tracking mode 'v3dt' requires noesis_v3dt_meta_ext; import failed: %s", exc)
        return False
    return True


def _ensure_baseline_depth_tracking_guardrails(pipeline_path: Path, logger: logging.Logger) -> bool:
    pipeline_cfg = _load_pipeline_config(pipeline_path, logger)
    if pipeline_cfg is None:
        return False
    if _allow_depthless_reid_smoke(pipeline_cfg):
        logger.warning("Depth-tracking guardrail bypassed for explicit ReID smoke validation config")
        return True
    models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
    depth_cfg = (models_cfg or {}).get("depth_tracking") if isinstance(models_cfg, dict) else None
    if not isinstance(depth_cfg, dict) or not bool(depth_cfg.get("enable", False)):
        logger.error(
            "Baseline tracking requires models.depth_tracking.enable=true in %s",
            pipeline_path,
        )
        return False
    cfg_path_raw = str(depth_cfg.get("config-file-path") or "").strip()
    if not cfg_path_raw:
        logger.error("Baseline tracking requires models.depth_tracking.config-file-path in %s", pipeline_path)
        return False
    cfg_path = _resolve_pipeline_cfg_path(pipeline_path, cfg_path_raw)
    if not cfg_path.exists():
        logger.error(
            "Baseline tracking depth-tracking config missing: %s (from %s)",
            cfg_path,
            cfg_path_raw,
        )
        return False
    try:
        _ensure_native_object_depth_extension(logger)
    except Exception as exc:
        logger.error("Baseline tracking requires noesis_depth_meta_ext; build/import failed: %s", exc)
        return False
    try:
        _ensure_native_depth_tracking_tensor_extension(logger)
    except Exception as exc:
        logger.error("Baseline tracking requires noesis_depth_tracking_tensor_ext; build/import failed: %s", exc)
        return False
    return True


def _resolve_depth_registration_path(args: argparse.Namespace, *, pipeline_path: Path | None = None) -> Path:
    raw = args.depth_registration_config
    if raw is not None:
        return Path(raw).resolve()
    if pipeline_path is not None and pipeline_path.exists():
        try:
            pipeline_cfg = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pipeline_cfg = {}
        reg_cfg = pipeline_cfg.get("depth_registration") if isinstance(pipeline_cfg, Mapping) else None
        if isinstance(reg_cfg, Mapping):
            reg_path = str(reg_cfg.get("path") or "").strip()
            if reg_path:
                return _resolve_pipeline_cfg_path(pipeline_path, reg_path)
        elif isinstance(reg_cfg, str) and reg_cfg.strip():
            return _resolve_pipeline_cfg_path(pipeline_path, reg_cfg.strip())
    return (DS9_ROOT / "config" / "depth_registration.json").resolve()


def _build_depth_registration_profile_fingerprints(
    pipeline_cfg: Mapping[str, Any],
    *,
    pipeline_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, Mapping) else {}
    depth_cfg = (models_cfg or {}).get("depth_tracking") if isinstance(models_cfg, Mapping) else {}
    ma_cfg = (models_cfg or {}).get("mapanything") if isinstance(models_cfg, Mapping) else {}
    repo_root = REPO_ROOT.resolve()
    depth_profile = _depth_registration_model_profile_fingerprint(
        depth_cfg if isinstance(depth_cfg, Mapping) else {},
        repo_root=repo_root,
        extra={
            "model_name": "depth-anything-v2-metric-hypersim-vits",
            "input_size": list(_DEPTH_TRACKING_INPUT_SIZE),
            "batch_size": int(_DEPTH_TRACKING_BATCH_SIZE),
            "interval": int(_DEPTH_TRACKING_INTERVAL),
            "gie_id": int(_DEPTH_TRACKING_GIE_ID),
        },
    )
    mapanything_profile = _depth_registration_model_profile_fingerprint(
        ma_cfg if isinstance(ma_cfg, Mapping) else {},
        repo_root=repo_root,
        extra={"scope": "mapanything_reference_depth"},
    )
    return depth_profile, mapanything_profile


def _active_baseline_camera_ids(
    pipeline_cfg: Mapping[str, Any],
    camera_labels: Mapping[int, str],
) -> list[tuple[int, str]]:
    sources = pipeline_cfg.get("sources") if isinstance(pipeline_cfg, Mapping) else None
    if not isinstance(sources, list):
        return []
    active: list[tuple[int, str]] = []
    for index, _source in enumerate(sources):
        camera_id = camera_labels.get(int(index))
        if isinstance(camera_id, str) and camera_id.strip():
            active.append((int(index), camera_id.strip()))
    return active


def _load_depth_registration_manager(
    *,
    path: Path,
    pipeline_path: Path,
    pipeline_cfg: Mapping[str, Any],
    calibration_provider: Any,
    camera_labels: Mapping[int, str],
    logger: logging.Logger,
) -> DepthRegistrationManager:
    manager = DepthRegistrationManager.load(path)
    depth_profile, mapanything_profile = _build_depth_registration_profile_fingerprints(
        pipeline_cfg,
        pipeline_path=pipeline_path,
    )
    missing: list[str] = []
    for source_id, camera_id in _active_baseline_camera_ids(pipeline_cfg, camera_labels):
        snapshot = calibration_provider.snapshot(int(source_id), str(camera_id))
        if snapshot is None:
            missing.append(f"{camera_id}:calibration_unavailable")
            continue
        try:
            manager.validate_runtime(
                camera_id=str(camera_id),
                snapshot=snapshot,
                dav2_profile=depth_profile,
                mapanything_profile=mapanything_profile,
            )
        except DepthRegistrationError as exc:
            missing.append(f"{camera_id}:{exc}")
    if missing:
        raise DepthRegistrationError(
            "Baseline tracking depth registration invalid for active cameras: " + ", ".join(missing)
        )
    logger.info("Loaded depth registration artifact %s for %d active cameras", path, len(_active_baseline_camera_ids(pipeline_cfg, camera_labels)))
    return manager


def _load_camera_labels(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    except Exception:
        logging.getLogger(__name__).warning("Unable to read cameras config at %s", path)
        return {}

    cameras = data.get("cameras", {})
    labels: Dict[int, str] = {}
    for key, entry in cameras.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                labels[idx] = name.strip()
                continue
        labels[idx] = f"camera_{idx}"
    return labels


def _build_storage_manager(args: argparse.Namespace) -> DepthStorageManager:
    service_cfg = load_service_config()
    storage_cfg = service_cfg.storage
    base_path = Path(args.storage_base) if args.storage_base else Path(storage_cfg.depth_base)
    return DepthStorageManager(
        base_path=base_path,
        max_snapshots_per_camera=storage_cfg.max_snapshots_per_camera,
        retention_minutes=storage_cfg.snapshot_retention_minutes,
        max_total_bytes=storage_cfg.max_total_bytes,
        enable_async=storage_cfg.async_enabled,
        max_queue_size=storage_cfg.queue_size,
        worker_count=storage_cfg.async_workers,
        max_worker_count=storage_cfg.async_max_workers,
        enforce_async=storage_cfg.enforce_async,
        enforce_interval_s=storage_cfg.enforce_interval_s,
        size_hysteresis_ratio=storage_cfg.quota_hysteresis_ratio,
        zarr_clevel=storage_cfg.zarr_clevel,
        zarr_chunk_px=storage_cfg.zarr_chunk_px,
        min_conf=getattr(getattr(service_cfg, "performance", None), "min_conf", float("nan")),
    )


def _build_stable_id_manager(logger: logging.Logger, *, pipeline_config: Optional[Mapping[str, Any]] = None):
    """Instantiate StableIDManager if enabled and available."""
    flag = os.environ.get("NOESIS_REID_ENABLED", "1")
    if str(flag).strip().lower() not in ("1", "true", "yes", "on"):
        logger.info("Stable ID manager disabled (NOESIS_REID_ENABLED=%s)", flag)
        return None
    try:
        from reid.stable_id_manager import StableIDManager  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Stable ID manager unavailable (import failed): %s", exc)
        return None

    try:
        device = os.environ.get("NOESIS_REID_DEVICE", "cuda:0")
        model_path = os.environ.get("NOESIS_REID_MODEL_PATH")
        model_name = os.environ.get("NOESIS_REID_MODEL_NAME", "osnet_x1_0")
        img_h = int(os.environ.get("NOESIS_REID_IMAGE_H", "256") or 256)
        img_w = int(os.environ.get("NOESIS_REID_IMAGE_W", "128") or 128)
        embed_interval_s = float(os.environ.get("NOESIS_REID_EMBED_INTERVAL_S", "1.0") or 1.0)
        new_id_hysteresis_frames = int(os.environ.get("NOESIS_REID_NEW_ID_HYSTERESIS_FRAMES", "1") or 1)
        new_id_confirm_frames_at_cap = int(os.environ.get("NOESIS_REID_NEW_ID_CONFIRM_FRAMES_AT_CAP", "1") or 1)
        pose_flag = os.environ.get("NOESIS_REID_POSE_ENABLED", "")
        if str(pose_flag).strip():
            pose_enabled = str(pose_flag).strip().lower() in ("1", "true", "yes", "on")
        else:
            pose_enabled = False
            try:
                pose_cfg = (pipeline_config or {}).get("models", {}).get("pose") or {}
                if isinstance(pose_cfg, Mapping) and bool(pose_cfg.get("enable", False)):
                    pose_enabled = True
            except Exception:
                pose_enabled = False
        pose_feature_flag = os.environ.get("NOESIS_POSE_FEATURES_ENABLED", "")
        if str(pose_feature_flag).strip().lower() in ("0", "false", "no", "off"):
            pose_enabled = False
        try:
            pose_weight = float(os.environ.get("NOESIS_REID_POSE_WEIGHT", "0.15") or 0.15)
        except Exception:
            pose_weight = 0.15
        try:
            pose_sim_threshold = float(os.environ.get("NOESIS_REID_POSE_SIM_THRESHOLD", "0.55") or 0.55)
        except Exception:
            pose_sim_threshold = 0.55
        try:
            pose_sim_high_threshold = float(os.environ.get("NOESIS_REID_POSE_SIM_HIGH_THRESHOLD", "0.65") or 0.65)
        except Exception:
            pose_sim_high_threshold = 0.65
        try:
            pose_only_threshold = float(os.environ.get("NOESIS_REID_POSE_ONLY_THRESHOLD", "0.80") or 0.80)
        except Exception:
            pose_only_threshold = 0.80
        try:
            pose_min_valid_frac = float(os.environ.get("NOESIS_REID_POSE_MIN_VALID_FRAC", "0.45") or 0.45)
        except Exception:
            pose_min_valid_frac = 0.45
        try:
            pose_min_mean_conf = float(os.environ.get("NOESIS_REID_POSE_MIN_MEAN_CONF", "0.50") or 0.50)
        except Exception:
            pose_min_mean_conf = 0.50
        try:
            pose_min_features = int(os.environ.get("NOESIS_REID_POSE_MIN_FEATURES", "6") or 6)
        except Exception:
            pose_min_features = 6
        try:
            pose_interval_s = float(os.environ.get("NOESIS_REID_POSE_INTERVAL_S", "0.75") or 0.75)
        except Exception:
            pose_interval_s = 0.75
        try:
            pose_gallery_size = int(os.environ.get("NOESIS_REID_POSE_GALLERY_SIZE", "8") or 8)
        except Exception:
            pose_gallery_size = 8
        try:
            pose_max_age_s = float(os.environ.get("NOESIS_REID_POSE_MAX_AGE_S", "30.0") or 30.0)
        except Exception:
            pose_max_age_s = 30.0
        try:
            pose_max_total_entries = int(os.environ.get("NOESIS_REID_POSE_MAX_TOTAL_ENTRIES", "0") or 0)
        except Exception:
            pose_max_total_entries = 0
        aliases_enabled_env = os.environ.get("NOESIS_REID_ALIASES_ENABLED", "1")
        aliases_enabled = str(aliases_enabled_env).strip().lower() in ("1", "true", "yes", "on")
        reset_sid_pool_env = os.environ.get("NOESIS_REID_RESET_SID_POOL")
        if reset_sid_pool_env is None:
            # Backward compatibility for command-line typos observed in the field.
            reset_sid_pool_env = os.environ.get("NOESIS_REID_RESER_SID_POOL", "0")
        reset_sid_pool = str(reset_sid_pool_env).strip().lower() in ("1", "true", "yes", "on")
        try:
            max_total_ids = int(os.environ.get("NOESIS_REID_MAX_TOTAL_IDS", "12") or 12)
        except Exception:
            max_total_ids = 12
        total_id_reuse_env = os.environ.get("NOESIS_REID_TOTAL_ID_REUSE", "1")
        total_id_reuse = str(total_id_reuse_env).strip().lower() in ("1", "true", "yes", "on")
        try:
            total_id_reuse_min_age_s = float(os.environ.get("NOESIS_REID_TOTAL_ID_REUSE_MIN_AGE_S", "60") or 60.0)
        except Exception:
            total_id_reuse_min_age_s = 60.0
        auto_merge_enabled_env = os.environ.get("NOESIS_REID_AUTO_MERGE_ENABLED", "1")
        auto_merge_enabled = str(auto_merge_enabled_env).strip().lower() in ("1", "true", "yes", "on")
        try:
            auto_merge_interval_s = float(os.environ.get("NOESIS_REID_AUTO_MERGE_INTERVAL_S", "5") or 5.0)
        except Exception:
            auto_merge_interval_s = 5.0
        try:
            auto_merge_max_attempts = int(os.environ.get("NOESIS_REID_AUTO_MERGE_MAX_ATTEMPTS", "20") or 20)
        except Exception:
            auto_merge_max_attempts = 20
        try:
            auto_merge_min_sim = float(os.environ.get("NOESIS_REID_AUTO_MERGE_MIN_SIM", "0.92") or 0.92)
        except Exception:
            auto_merge_min_sim = 0.92
        try:
            auto_merge_min_embeddings_for_suggest = int(
                os.environ.get("NOESIS_REID_AUTO_MERGE_MIN_EMBEDDINGS_FOR_SUGGEST", "1") or 1
            )
        except Exception:
            auto_merge_min_embeddings_for_suggest = 1
        auto_merge_respect_inactive_env = os.environ.get("NOESIS_REID_AUTO_MERGE_RESPECT_INACTIVE", "0")
        auto_merge_respect_inactive = str(auto_merge_respect_inactive_env).strip().lower() in ("1", "true", "yes", "on")
        auto_merge_force_stuck_env = os.environ.get("NOESIS_REID_AUTO_MERGE_FORCE_STUCK", "0")
        auto_merge_force_stuck = str(auto_merge_force_stuck_env).strip().lower() in ("1", "true", "yes", "on")
        auto_merge_allow_both_active_env = os.environ.get("NOESIS_REID_AUTO_MERGE_ALLOW_BOTH_ACTIVE", "1")
        auto_merge_allow_both_active = str(auto_merge_allow_both_active_env).strip().lower() in ("1", "true", "yes", "on")
        try:
            auto_merge_both_active_min_sim = float(
                os.environ.get("NOESIS_REID_AUTO_MERGE_BOTH_ACTIVE_MIN_SIM", "0.97") or 0.97
            )
        except Exception:
            auto_merge_both_active_min_sim = 0.97
        alias_file = os.environ.get("NOESIS_REID_ALIAS_FILE", "~/.noesis/reid_aliases.json")
        alias_append_env = os.environ.get("NOESIS_REID_ALIAS_APPEND_DEFAULT", "1")
        alias_append_default = str(alias_append_env).strip().lower() in ("1", "true", "yes", "on")
        try:
            copresence_window_s = float(os.environ.get("NOESIS_REID_COPRESENCE_WINDOW_S", "60") or 60.0)
        except Exception:
            copresence_window_s = 60.0
        try:
            suggest_min_sim = float(os.environ.get("NOESIS_REID_SUGGEST_MIN_SIM", "0.92") or 0.92)
        except Exception:
            suggest_min_sim = 0.92
        try:
            suggest_mnn_margin = float(os.environ.get("NOESIS_REID_SUGGEST_MNN_MARGIN", "0.02") or 0.02)
        except Exception:
            suggest_mnn_margin = 0.02
        try:
            suggest_pose_sim_low = float(os.environ.get("NOESIS_REID_SUGGEST_POSE_SIM_LOW", "0.70") or 0.70)
        except Exception:
            suggest_pose_sim_low = 0.70
        try:
            suggest_pose_sim_high = float(os.environ.get("NOESIS_REID_SUGGEST_POSE_SIM_HIGH", "0.90") or 0.90)
        except Exception:
            suggest_pose_sim_high = 0.90
        try:
            min_embeddings_for_suggest = int(os.environ.get("NOESIS_REID_MIN_EMBEDDINGS_FOR_SUGGEST", "3") or 3)
        except Exception:
            min_embeddings_for_suggest = 3
        try:
            alias_history_max = int(os.environ.get("NOESIS_REID_ALIAS_HISTORY_MAX", "1000") or 1000)
        except Exception:
            alias_history_max = 1000
        stableid_gpu_enabled_env = str(os.environ.get("NOESIS_STABLEID_GPU_ENABLED", "1") or "1").strip().lower()
        if stableid_gpu_enabled_env not in ("1", "true", "yes", "on"):
            logger.error("NOESIS_STABLEID_GPU_ENABLED=%s is not supported in zero-copy hard-cutover", stableid_gpu_enabled_env)
            return None
        compute_backend = "gpu"
        gpu_device = str(os.environ.get("NOESIS_STABLEID_GPU_DEVICE", device) or device)
        try:
            gpu_min_gallery = int(os.environ.get("NOESIS_STABLEID_GPU_MIN_GALLERY", "32") or 32)
        except Exception:
            gpu_min_gallery = 32
        extra_kwargs = {
            "max_total_ids": max_total_ids,
            "total_id_reuse": total_id_reuse,
            "total_id_reuse_min_age_s": total_id_reuse_min_age_s,
            "reset_sid_pool_on_start": reset_sid_pool,
            "aliases_enabled": aliases_enabled,
            "alias_file": alias_file,
            "alias_append_default": alias_append_default,
            "copresence_window_s": copresence_window_s,
            "suggest_min_sim": suggest_min_sim,
            "suggest_mnn_margin": suggest_mnn_margin,
            "suggest_pose_sim_low": suggest_pose_sim_low,
            "suggest_pose_sim_high": suggest_pose_sim_high,
            "min_embeddings_for_suggest": min_embeddings_for_suggest,
            "alias_history_max": alias_history_max,
            "auto_merge_enabled": auto_merge_enabled,
            "auto_merge_interval_s": auto_merge_interval_s,
            "auto_merge_max_attempts": auto_merge_max_attempts,
            "auto_merge_min_sim": auto_merge_min_sim,
            "auto_merge_min_embeddings_for_suggest": auto_merge_min_embeddings_for_suggest,
            "auto_merge_respect_inactive": auto_merge_respect_inactive,
            "auto_merge_force_stuck": auto_merge_force_stuck,
            "auto_merge_allow_both_active": auto_merge_allow_both_active,
            "auto_merge_both_active_min_sim": auto_merge_both_active_min_sim,
            "compute_backend": compute_backend,
            "gpu_device": gpu_device,
            "gpu_min_gallery": gpu_min_gallery,
        }
        try:
            sig = inspect.signature(StableIDManager.__init__)
            valid_params = set(sig.parameters)
            valid_params.discard("self")
            extra_kwargs = {key: val for key, val in extra_kwargs.items() if key in valid_params}
        except Exception:
            pass
        mgr = StableIDManager(
            model_path=model_path,
            device=device,
            model_name=model_name,
            image_size=(img_h, img_w),
            allow_multi_zone_active=True,
            # Noesis stable IDs source embeddings from an explicit OSNet SGIE; do not load torchreid.
            use_extractor=False,
            embed_interval_s=embed_interval_s,
            new_id_hysteresis_frames=new_id_hysteresis_frames,
            new_id_confirm_frames_at_cap=new_id_confirm_frames_at_cap,
            pose_enabled=pose_enabled,
            pose_weight=pose_weight,
            pose_sim_threshold=pose_sim_threshold,
            pose_sim_high_threshold=pose_sim_high_threshold,
            pose_only_threshold=pose_only_threshold,
            pose_min_valid_frac=pose_min_valid_frac,
            pose_min_mean_conf=pose_min_mean_conf,
            pose_min_features=pose_min_features,
            pose_interval_s=pose_interval_s,
            pose_gallery_size=pose_gallery_size,
            pose_max_age_s=pose_max_age_s,
            pose_max_total_entries=pose_max_total_entries,
            **extra_kwargs,
        )
        logger.info(
            "Stable ID manager initialised (SGIE embeddings; allow_multi_zone_active=%s, embed_interval_s=%.3f, new_id_hysteresis_frames=%d, new_id_confirm_frames_at_cap=%d, pose_enabled=%s)",
            True,
            embed_interval_s,
            new_id_hysteresis_frames,
            new_id_confirm_frames_at_cap,
            pose_enabled,
        )
        try:
            sid_metrics = dict(mgr.get_sid_metrics() or {})
        except Exception:
            sid_metrics = {}
        backend_mode = str(sid_metrics.get("stableid_backend_mode") or "").strip().lower()
        if backend_mode != "gpu":
            logger.error(
                "Stable ID manager backend is '%s' (expected gpu); refusing CPU fallback in hard-cutover",
                backend_mode or "unknown",
            )
            return None
        return mgr
    except Exception as exc:
        logger.error("Stable ID manager init failed for hard-cutover: %s", exc)
        return None


class _CalibrationProvider:
    """Provide calibration snapshots and WS bundle for BEV rendering."""

    def __init__(
        self,
        cameras_path: Path,
        pipeline_cfg: Dict[str, object],
        *,
        tracking_mode: Optional[str] = None,
        extrinsics_path: Optional[Path] = None,
    ) -> None:
        self._loader = CameraConfigLoader(cameras_path)
        self._camera_model_res = self._load_camera_model_resolutions(cameras_path)
        self._intrinsics_models = load_intrinsics(str(REPO_ROOT / "intrinsics.json"))
        self._align = load_alignment(str(REPO_ROOT / "config" / "ply_alignment.json"))
        tracking_mode_norm = str(tracking_mode or "").strip().lower() or "baseline"
        default_extrinsics_path = REPO_ROOT / "config" / "camera_calibration.json"
        env_path = os.environ.get("NOESIS_CALIBRATION_EXTRINSICS", "")
        if not extrinsics_path and env_path.strip():
            extrinsics_path = Path(env_path.strip())
        self._extrinsics_path = Path(extrinsics_path) if extrinsics_path else default_extrinsics_path
        self._extrinsics = load_extrinsics(str(self._extrinsics_path), align_data=self._align)
        logging.getLogger(__name__).info("Calibration extrinsics path=%s", self._extrinsics_path)
        try:
            from config import config as app_config  # type: ignore

            calib_cfg = getattr(app_config, "calibration", None)
            self._model_map = dict(getattr(calib_cfg, "CAMERA_INTRINSICS_MODEL_MAP", {}) or {})
            self._camera_specs = dict(getattr(calib_cfg, "CAMERA_SPECS", {}) or {})
        except Exception:
            self._model_map = {}
            self._camera_specs = {}
        streammux_cfg = pipeline_cfg.get("streammux") or {}
        try:
            self._frame_size = (
                int((streammux_cfg or {}).get("width", 0) or 0),
                int((streammux_cfg or {}).get("height", 0) or 0),
            )
        except Exception:
            self._frame_size = (0, 0)
        self._camera_labels: Dict[int, str] = {}
        self._bundle_cache: Optional[Dict[str, object]] = None
        pose_only_env = str(os.environ.get("NOESIS_CALIBRATION_POSE_ONLY", "1") or "").strip().lower()
        self._pose_only = pose_only_env in ("1", "true", "yes", "on", "y")

    @staticmethod
    def _resolution_from_entry(entry: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(entry, dict):
            return None
        res = entry.get("resolution")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            try:
                w = int(res[0])
                h = int(res[1])
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                return None
        intr = entry.get("intrinsics")
        if isinstance(intr, dict):
            res = intr.get("resolution")
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                try:
                    w = int(res[0])
                    h = int(res[1])
                    if w > 0 and h > 0:
                        return w, h
                except Exception:
                    return None
        return None

    @classmethod
    def _load_camera_model_resolutions(cls, cameras_path: Path) -> Dict[str, Tuple[int, int]]:
        try:
            data = yaml.safe_load(Path(cameras_path).read_text()) or {}
        except Exception:
            return {}
        models = data.get("intrinsics_models") or {}
        cameras = data.get("cameras") or {}
        if not isinstance(models, dict) or not isinstance(cameras, dict):
            return {}
        model_res: Dict[str, Tuple[int, int]] = {}
        for key, entry in models.items():
            res = cls._resolution_from_entry(entry)
            if res:
                model_res[str(key)] = res
        cam_res: Dict[str, Tuple[int, int]] = {}
        for entry in cameras.values():
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            model = entry.get("model") or entry.get("intrinsics_model")
            if isinstance(name, str) and isinstance(model, str):
                res = model_res.get(model)
                if res:
                    cam_res[name] = res
        return cam_res

    def set_camera_labels(self, labels: Dict[int, str]) -> None:
        self._camera_labels = dict(labels or {})
        self._bundle_cache = None

    def pose_only_enabled(self) -> bool:
        return bool(self._pose_only)

    def extrinsics_path(self) -> Path:
        return Path(self._extrinsics_path)

    def validate_pose_coverage(self) -> Dict[str, str]:
        errors: Dict[str, str] = {}
        camera_ids = sorted({name for name in self._camera_labels.values() if isinstance(name, str) and str(name).strip()})
        if not camera_ids:
            return errors
        cams = self._extrinsics.get("cameras", {}) if isinstance(self._extrinsics, dict) else {}
        align = self._align if isinstance(self._align, dict) else {}
        try:
            floor_y = float(align.get("floor_y", 0.0) or 0.0)
        except Exception:
            floor_y = 0.0
        for camera_id in camera_ids:
            entry = cams.get(camera_id)
            if not isinstance(entry, dict):
                errors[camera_id] = "missing_camera_entry"
                continue
            pose = entry.get("pose")
            if not isinstance(pose, dict):
                errors[camera_id] = "missing_or_invalid_pose"
                continue
            E = entry.get("E")
            if not (isinstance(E, list) and len(E) == 16):
                errors[camera_id] = "pose_to_extrinsics_failed"
                continue
            try:
                Emat = np.array(E, dtype=np.float64).reshape((4, 4), order="F")
                Twc = np.linalg.inv(Emat)
                C_world = Twc[:3, 3].copy()
                if not np.all(np.isfinite(C_world)):
                    errors[camera_id] = "non_finite_camera_center"
                    continue
                if float(C_world[1]) <= float(floor_y) + 1e-3:
                    errors[camera_id] = "camera_not_above_floor"
                    continue
                R_wc = Twc[:3, :3].copy()
                forward = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                denom = float(forward[1])
                if abs(denom) < 1e-6:
                    errors[camera_id] = "camera_forward_parallel_to_floor"
                    continue
                t_hit = (float(floor_y) - float(C_world[1])) / denom
                if t_hit <= 0.0:
                    errors[camera_id] = "camera_forward_misses_floor"
            except Exception:
                errors[camera_id] = "invalid_pose_geometry"
        return errors

    def reload_extrinsics(self) -> None:
        """Reload extrinsics from the configured calibration path without touching alignment."""
        self._extrinsics = load_extrinsics(str(self._extrinsics_path), align_data=self._align)
        self._bundle_cache = None

    def reload_alignment(self) -> None:
        """Reload alignment from ply_alignment.json."""
        self._align = load_alignment(str(REPO_ROOT / "config" / "ply_alignment.json"))
        self._extrinsics = load_extrinsics(str(self._extrinsics_path), align_data=self._align)
        self._bundle_cache = None

    def snapshot(self, source_id: int, camera_id: str) -> Optional["CalibrationSnapshot"]:
        intr = self._loader.get(source_id)
        if intr is None:
            return None
        try:
            K = np.array(
                [
                    [float(intr.fx), 0.0, float(intr.cx)],
                    [0.0, float(intr.fy), float(intr.cy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        except Exception:
            return None
        extr_entry = self._extrinsics.get("cameras", {}).get(camera_id) if isinstance(self._extrinsics, dict) else None
        E = extr_entry.get("E") if isinstance(extr_entry, dict) else None
        if not isinstance(E, list) or len(E) != 16:
            return None
        align_dict = self._align if isinstance(self._align, dict) else {}
        floor_y = float(align_dict.get("floor_y", 0.0) or 0.0)
        unit_scale = 1.0
        frame_w, frame_h = self._frame_size
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 1920, 1080

        # Align intrinsics with the current streammux resolution.
        base_w = base_h = None
        res = self._camera_model_res.get(camera_id)
        if res:
            base_w, base_h = res
        if base_w is None or base_h is None:
            try:
                spec = (self._camera_specs or {}).get(camera_id) if isinstance(self._camera_specs, dict) else None
                if isinstance(spec, dict):
                    res = spec.get("resolution")
                    if isinstance(res, (list, tuple)) and len(res) >= 2:
                        base_w = base_w or int(res[0]) or None
                        base_h = base_h or int(res[1]) or None
            except Exception:
                pass
        if (base_w is None or base_h is None) and isinstance(self._model_map, dict):
            model_key = self._model_map.get(camera_id)
            model = (self._intrinsics_models or {}).get(model_key, {}) if model_key else {}
            res = model.get("resolution") if isinstance(model, dict) else None
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                try:
                    bw = int(res[0]); bh = int(res[1])
                    base_w = base_w or bw
                    base_h = base_h or bh
                except Exception:
                    pass
        if base_w is None or base_h is None:
            try:
                bw_guess = int(round(float(K[0, 2]) * 2.0))
                bh_guess = int(round(float(K[1, 2]) * 2.0))
                if bw_guess > 0 and bh_guess > 0:
                    base_w = base_w or bw_guess
                    base_h = base_h or bh_guess
            except Exception:
                pass
        if base_w and base_h and (base_w != frame_w or base_h != frame_h):
            try:
                sx = float(frame_w) / float(base_w)
                sy = float(frame_h) / float(base_h)
                K = K.copy()
                K[0, 0] *= sx
                K[0, 2] *= sx
                K[1, 1] *= sy
                K[1, 2] *= sy
            except Exception:
                pass
        return CalibrationSnapshot(
            camera_id=camera_id,
            intrinsics=K,
            extrinsics_col_major=list(E),
            floor_y=floor_y,
            image_size=(int(frame_w), int(frame_h)),
            unit_scale=unit_scale,
        )

    def calibration_bundle(self) -> Dict[str, object]:
        if self._bundle_cache is not None:
            return dict(self._bundle_cache)
        camera_ids = sorted({name for name in self._camera_labels.values() if isinstance(name, str)})
        if not camera_ids:
            return {}
        bundle = assemble_calibration_bundle(
            camera_ids=camera_ids,
            intrinsics_models=self._intrinsics_models,
            model_map=self._model_map,
            extrinsics_data=self._extrinsics,
            align_data=self._align,
            camera_specs=self._camera_specs,
        )
        cams_node = bundle.setdefault("cameras", {})
        k_table = cams_node.setdefault("K", {})
        frame_w, frame_h = self._frame_size
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 1920, 1080
        for src_id, cam_name in self._camera_labels.items():
            intr = self._loader.get(src_id)
            if intr is None:
                continue
            try:
                fx = float(intr.fx)
                fy = float(intr.fy)
                cx = float(intr.cx)
                cy = float(intr.cy)
            except Exception:
                continue
            base_w = base_h = 0
            res = self._camera_model_res.get(cam_name)
            if res:
                base_w, base_h = res
            if base_w <= 0 or base_h <= 0:
                try:
                    base_w = int(round(cx * 2.0))
                    base_h = int(round(cy * 2.0))
                except Exception:
                    base_w = base_h = 0
            if base_w > 0 and base_h > 0 and (base_w != frame_w or base_h != frame_h):
                try:
                    sx = float(frame_w) / float(base_w)
                    sy = float(frame_h) / float(base_h)
                    fx *= sx
                    cx *= sx
                    fy *= sy
                    cy *= sy
                except Exception:
                    pass
            k_table[cam_name] = [fx, fy, cx, cy]
        self._bundle_cache = bundle
        return dict(bundle)

def _build_stats_callback(
    pipeline: ds8_pipeline.DS8Pipeline,
    camera_labels: Dict[int, str],
    ws_metrics_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ws_metrics_resetter: Optional[Callable[[], None]] = None,
) -> Callable[[], Dict[str, object]]:
    start_time = time.time()
    stats_logger = logging.getLogger(__name__)
    sid_metrics_fetch_warned = False

    def _default_stableid_metrics() -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "stableid_backend_mode": None,
            "stableid_gpu_match_p50_ms": None,
            "stableid_gpu_match_p95_ms": None,
            "stableid_gallery_size": 0,
            "canonical_gallery_size": 0,
            "active_unique": 0,
            "ghost_unique": 0,
            "pending_new_count": 0,
            "auto_merge_last_candidate_count": 0,
            "auto_merge_last_blocked_count": 0,
            "auto_merge_zero_apply_streak": 0,
            "auto_merge_last_reason": None,
            "auto_merge_last_pressure_recycled": 0,
            "sid_new_alloc_count": 0,
            "sid_remap_count": 0,
            "sid_guard_reject_count": 0,
            "sid_no_embedding_count": 0,
            "sid_fragmentation_events": 0,
            "sid_pending_recycled_count": 0,
            "sid_same_frame_conflict_count": 0,
            "alias_candidate_pool_size": 0,
            "alias_candidate_sim_p50": None,
            "alias_candidate_sim_p95": None,
            "alias_support_ids_ge_1": 0,
            "alias_support_ids_ge_2": 0,
            "alias_support_ids_ge_3": 0,
        }

    def _mosaic_layout() -> Optional[Dict[str, object]]:
        tiler = pipeline.components.get("tiler")
        tiler_cfg = tiler.config if tiler and isinstance(tiler.config, dict) else {}
        try:
            frame_w, frame_h = pipeline.frame_size
        except Exception:
            frame_w, frame_h = (0, 0)

        sources = [
            {"source_id": int(source_id), "camera_id": str(name)}
            for source_id, name in sorted(camera_labels.items(), key=lambda item: int(item[0]))
        ]
        source_count = len(sources)

        mosaic_w = int(tiler_cfg.get("width", 0) or 0)
        mosaic_h = int(tiler_cfg.get("height", 0) or 0)
        cols_raw = tiler_cfg.get("columns")
        rows_raw = tiler_cfg.get("rows")
        cols = int(cols_raw) if cols_raw is not None else None
        rows = int(rows_raw) if rows_raw is not None else None
        square_seq_grid = bool(tiler_cfg.get("square-seq-grid", False))

        layout: Dict[str, object] = {
            "source_count": source_count,
            "sources": sources,
            "tile_order": "source-id",
            "frame_w": int(frame_w or 0),
            "frame_h": int(frame_h or 0),
            "square_seq_grid": square_seq_grid,
        }
        if mosaic_w:
            layout["mosaic_w"] = mosaic_w
        if mosaic_h:
            layout["mosaic_h"] = mosaic_h
        if cols is not None:
            layout["cols"] = cols
        if rows is not None:
            layout["rows"] = rows
        return layout

    def _normalize_room_name(name: Any) -> str:
        try:
            s = str(name)
            if not s:
                return s
            # Replace common separators with space
            s2 = s.replace("_", " ").replace("-", " ")
            # Insert spaces before capital letters for CamelCase
            import re

            if " " not in s2 and any(c.islower() for c in s2) and any(c.isupper() for c in s2):
                s2 = re.sub(r"(?<!^)(?=[A-Z])", " ", s2)
            # Normalize whitespace and Title Case
            s2 = " ".join(s2.split())
            return s2.title()
        except Exception:
            return str(name)

    def _stats() -> Dict[str, object]:
        nonlocal sid_metrics_fetch_warned
        now = time.time()
        try:
            depth_fps = pipeline.depth_fps()
        except Exception:
            depth_fps = 0.0
        reload_count = getattr(pipeline, "analytics_reload_count", 0)
        stableid_metrics: Dict[str, Any] = _default_stableid_metrics()
        try:
            sid_mgr = getattr(pipeline, "stable_id_mgr", None)
            if sid_mgr is not None:
                get_metrics = getattr(sid_mgr, "get_sid_metrics", None)
                if callable(get_metrics):
                    fetched = dict(get_metrics() or {})
                    if fetched:
                        stableid_metrics.update(fetched)
                        stableid_metrics["status"] = "ok"
                    else:
                        stableid_metrics["status"] = "empty"
                else:
                    stableid_metrics["status"] = "getter_missing"
            else:
                stableid_metrics["status"] = "manager_missing"
        except Exception:
            stableid_metrics["status"] = "fetch_error"
            stableid_metrics["fetch_error"] = "exception"
            if not sid_metrics_fetch_warned:
                stats_logger.warning("StableID metrics fetch failed in stats callback", exc_info=True)
                sid_metrics_fetch_warned = True
        cameras_stats: Dict[str, object] = {}
        core_instr = hooks.get_core_path_instrumentation_snapshot()
        core_counters = dict(core_instr.get("counters", {}))
        core_stage_timings = dict(core_instr.get("stage_timings", {}))
        core_violations = int(core_counters.get("core_path.cpu_copy_violation.total", 0))
        ws_boundary_metrics: Dict[str, Any] = {}
        if callable(ws_metrics_getter):
            try:
                ws_boundary_metrics = ws_metrics_getter() or {}
            except Exception:
                ws_boundary_metrics = {}
        rest_boundary_metrics: Dict[str, Any] = {}
        try:
            from noesis.server import boundary_metrics as _rest_boundary_metrics

            rest_boundary_metrics = _rest_boundary_metrics.get_boundary_serialization_metrics() or {}
        except Exception:
            rest_boundary_metrics = {}
        ws_p50 = ws_boundary_metrics.get("p50_ms")
        ws_p95 = ws_boundary_metrics.get("p95_ms")
        ws_p99 = ws_boundary_metrics.get("p99_ms")
        rest_p50 = rest_boundary_metrics.get("p50_ms")
        rest_p95 = rest_boundary_metrics.get("p95_ms")
        rest_p99 = rest_boundary_metrics.get("p99_ms")

        def _max_nullable(a: Any, b: Any) -> Any:
            try:
                if a is None:
                    return b
                if b is None:
                    return a
                return max(float(a), float(b))
            except Exception:
                return a if a is not None else b

        boundary_p50 = _max_nullable(ws_p50, rest_p50)
        boundary_p95 = _max_nullable(ws_p95, rest_p95)
        boundary_p99 = _max_nullable(ws_p99, rest_p99)
        latency_collector = getattr(pipeline, "latency_collector", None)
        latency_by_source: Dict[int, Dict[str, object]] = {}
        latency_aggregate: Optional[Dict[str, object]] = None
        latency_disabled: Optional[Dict[str, object]] = None
        if latency_collector is not None:
            try:
                latency_aggregate = latency_collector.snapshot_aggregate(now)
            except Exception:
                latency_aggregate = None
            try:
                latency_by_source = latency_collector.snapshot_by_source(now)
            except Exception:
                latency_by_source = {}
            try:
                latency_disabled = latency_collector.disabled_snapshot()
            except Exception:
                latency_disabled = None

        # Access analytics processor if attached to the pipeline components
        analytics_comp = pipeline.components.get("analytics")
        analytics_proc = analytics_comp.config.get("_analytics_processor") if analytics_comp else None

        for sensor_id, name in sorted(camera_labels.items()):
            cam_key = str(name)

            tracking: Dict[str, Any] = {
                "occupancy": {},
                "active_tracks": [],
                "transitions": [],
            }

            if analytics_proc:
                try:
                    raw_stats = analytics_proc.get_tracking_stats(sensor_id)
                    tracking["active_tracks"] = raw_stats.get("active_tracks", [])
                    tracking["transitions"] = raw_stats.get("transitions", [])

                    # Normalize occupancy names for frontend
                    raw_occ = raw_stats.get("occupancy", {})
                    norm_occ = {}
                    for k, v in raw_occ.items():
                        norm_occ[_normalize_room_name(k)] = v
                    tracking["occupancy"] = norm_occ
                except Exception:
                    pass

            latency_payload: Optional[Dict[str, object]] = None
            if latency_collector is not None:
                if getattr(latency_collector, "enabled", False):
                    latency_payload = latency_by_source.get(
                        int(sensor_id),
                        {
                            "enabled": True,
                            "window_sec": float(getattr(latency_collector, "window_sec", 10.0)),
                            "count": 0,
                            "p50": None,
                            "p95": None,
                            "max": None,
                            "last_sample_age_sec": None,
                        },
                    )
                elif latency_disabled is not None:
                    latency_payload = dict(latency_disabled)

            cameras_stats[cam_key] = {
                "fps": 0.0,
                "frames_processed": 0,
                "status": "running" if pipeline.activated else "unknown",
                "tracking": tracking,
                **({"latency_ms": latency_payload} if latency_payload is not None else {}),
            }
        return {
            "timestamp": now,
            "uptime": now - start_time,
            "stack": "ds8",
            "application": {
                "running": pipeline.activated,
                "cameras_active": len(camera_labels),
                "processors_active": 1 if pipeline.activated else 0,
            },
            "pipeline": {
                "stack": "ds8",
                "prepared": pipeline.prepared,
                "activated": pipeline.activated,
                "depth_enabled": pipeline.depth_enabled,
                "depth_fps": depth_fps,
                "zero_copy_profile": str(os.environ.get("NOESIS_ZERO_COPY_PROFILE", "strict") or "strict"),
                "zero_copy_core_enabled": True,
                "zero_copy_violations": core_violations,
                "stableid_backend_mode": stableid_metrics.get("stableid_backend_mode"),
                "stableid_gpu_match_p50_ms": stableid_metrics.get("stableid_gpu_match_p50_ms"),
                "stableid_gpu_match_p95_ms": stableid_metrics.get("stableid_gpu_match_p95_ms"),
                "stableid_gallery_size": stableid_metrics.get("stableid_gallery_size"),
                "boundary_cpu_serialization_p50_ms": boundary_p50,
                "boundary_cpu_serialization_p95_ms": boundary_p95,
                "boundary_cpu_serialization_p99_ms": boundary_p99,
                "boundary_cpu_serialization_ws_p99_ms": ws_p99,
                "boundary_cpu_serialization_rest_p99_ms": rest_p99,
                "analytics_reload_count": reload_count,
                "mosaic_layout": _mosaic_layout(),
                **({"latency_ms": latency_aggregate} if latency_aggregate is not None else {}),
                "zero_copy_core": {
                    "counters": core_counters,
                    "stage_timings": core_stage_timings,
                    "boundary_serialization_metrics": {
                        "ws": ws_boundary_metrics,
                        "rest": rest_boundary_metrics,
                    },
                },
                "stableid_metrics": stableid_metrics,
                "errors": list(pipeline.errors),
            },
            "cameras": cameras_stats,
        }

    def _clear_stats() -> None:
        try:
            collector = getattr(pipeline, "latency_collector", None)
            if collector is not None:
                collector.clear()
        except Exception:
            pass
        try:
            hooks.reset_core_path_instrumentation()
        except Exception:
            pass
        if callable(ws_metrics_resetter):
            try:
                ws_metrics_resetter()
            except Exception:
                pass
        try:
            from noesis.server import boundary_metrics as _rest_boundary_metrics

            _rest_boundary_metrics.reset_boundary_serialization_metrics()
        except Exception:
            pass

    setattr(_stats, "clear_stats", _clear_stats)
    return _stats


def _start_websocket_server(server: WebSocketServer) -> tuple[threading.Thread, Optional[asyncio.AbstractEventLoop]]:
    logger = logging.getLogger(__name__)
    started = threading.Event()
    loop_holder: Dict[str, asyncio.AbstractEventLoop] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        try:
            server.event_loop = loop
            loop.run_until_complete(server.start())
            started.set()
            loop.run_forever()
        except Exception:
            logger.exception("WebSocket server thread terminated unexpectedly")
            started.set()
        finally:
            try:
                loop.run_until_complete(server.stop())
            except Exception:
                pass
            finally:
                if not loop.is_closed():
                    loop.stop()
                    loop.close()

    thread = threading.Thread(target=_run, name="DS9-WebSocket", daemon=True)
    thread.start()
    started.wait(timeout=5.0)
    return thread, loop_holder.get("loop")


def _wait_for_rtsp_ready(host: str, port: int, timeout: float = 15.0, interval: float = 0.2) -> bool:
    """Wait for RTSP port to accept TCP connections before starting the gateway."""
    deadline = time.time() + timeout
    addr = (host, port)
    attempt = 0
    start_ts = time.time()
    last_log = 0.0
    #region agent log
    try:
        with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "ds9_runtime_core.py:_wait_for_rtsp_ready",
                        "message": "rtsp wait start",
                        "data": {"host": host, "port": port, "timeout_s": timeout},
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    #endregion
    while time.time() < deadline:
        attempt += 1
        try:
            with socket.create_connection(addr, timeout=0.3):
                #region agent log
                try:
                    with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds9_runtime_core.py:_wait_for_rtsp_ready",
                                    "message": "rtsp port ready",
                                    "data": {"host": host, "port": port, "attempt": attempt, "elapsed_ms": int((time.time() - (deadline - timeout)) * 1000)},
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                #endregion
                return True
        except Exception:
            now = time.time()
            if now - last_log >= 1.0:
                last_log = now
                #region agent log
                try:
                    with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds9_runtime_core.py:_wait_for_rtsp_ready",
                                    "message": "rtsp wait attempt",
                                    "data": {
                                        "host": host,
                                        "port": port,
                                        "attempt": attempt,
                                        "elapsed_ms": int((now - start_ts) * 1000),
                                    },
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                #endregion
            time.sleep(interval)
    #region agent log
    try:
        with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "ds9_runtime_core.py:_wait_for_rtsp_ready",
                        "message": "rtsp port not ready",
                        "data": {"host": host, "port": port, "timeout_s": timeout},
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    #endregion
    return False


def _stop_websocket_server(
    server: WebSocketServer,
    thread: Optional[threading.Thread],
    loop: Optional[asyncio.AbstractEventLoop],
    timeout: float = 5.0,
) -> None:
    if thread is None:
        return
    try:
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(server.stop(), loop)
            future.result(timeout=timeout)
            loop.call_soon_threadsafe(loop.stop)
    except Exception:
        pass
    finally:
        thread.join(timeout=timeout)


def _build_rest_app() -> "FastAPI":
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from noesis.server import analytics_api, depth_api, reid_api, virtual_twin_api

    app = FastAPI(title="Noesis DS9 Runtime API")
    origins_env = os.environ.get("NOESIS_REST_CORS_ORIGINS", "").strip()
    allow_all = os.environ.get("NOESIS_REST_CORS_ALLOW_ALL", "").strip().lower() in {"1", "true", "yes", "on"}
    origin_regex = os.environ.get("NOESIS_REST_CORS_ORIGIN_REGEX", "").strip()
    origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()] if origins_env else []
    if allow_all and "*" not in origins:
        origins = ["*"]
    if not origins and not origin_regex:
        origin_regex = (
            r"^https?://("
            r"localhost|127\.0\.0\.1|"
            r"10\.\d+\.\d+\.\d+|"
            r"172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+|"
            r"192\.168\.\d+\.\d+"
            r")(:\d+)?$"
        )
    if origins or origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_origin_regex=origin_regex or None,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(depth_api.app.router)
    app.include_router(analytics_api.app.router)
    app.include_router(reid_api.app.router)
    app.include_router(virtual_twin_api.app.router)
    return app


def _preload_rest_server_runtime() -> bool:
    try:
        import logging.config  # noqa: F401
        import logging.handlers  # noqa: F401
        import uvicorn  # noqa: F401
        import uvicorn.config  # noqa: F401
        import uvicorn.server  # noqa: F401
    except Exception:
        return False
    return True


def _start_rest_server(app: "FastAPI", host: str, port: int) -> tuple[Optional["uvicorn.Server"], Optional[threading.Thread]]:
    """Start the FastAPI REST server unless the port is already in use.

    If something is already listening on the requested host/port, we assume a
    REST instance is active and skip starting another one to avoid conflicts.
    """
    try:
        import uvicorn
    except Exception:
        logging.getLogger(__name__).warning("uvicorn not available; REST server disabled")
        return None, None

    # Safety check: skip starting another REST server if port is already in use
    try:
        import socket

        def _can_connect(_host: str, _port: int, timeout: float = 0.25) -> bool:
            try:
                with socket.create_connection((_host, int(_port)), timeout=timeout):
                    return True
            except Exception:
                return False

        # Normalize host for connectivity test when binding to all interfaces
        test_host = host
        if not test_host or test_host == "0.0.0.0":
            test_host = "127.0.0.1"
        elif test_host == "::":
            test_host = "::1"

        if _can_connect(test_host, port):
            logging.getLogger(__name__).info(
                "REST port %s is already in use on %s; assuming server active and skipping start",
                port,
                test_host,
            )
            return None, None
    except Exception:
        # Non-fatal: if the check fails, proceed to start server
        pass

    config = uvicorn.Config(app=app, host=host, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config=config)

    def _run() -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())
        server.run()

    thread = threading.Thread(target=_run, name="DS9-REST", daemon=True)
    thread.start()
    return server, thread


def _stop_rest_server(server: Optional["uvicorn.Server"], thread: Optional[threading.Thread], timeout: float = 5.0) -> None:
    if server is None or thread is None:
        return
    try:
        server.should_exit = True
    except Exception:
        pass
    thread.join(timeout=timeout)


def _setup_webrtc_signaling(
    pipeline: ds8_pipeline.DS8Pipeline,
    ws_server: WebSocketServer,
    logger: logging.Logger,
) -> None:
    """Attach WebRTC signaling handlers from webrtcbin to WebSocketServer."""
    from gi.repository import Gst

    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        logger.warning("DS9 pipeline handle unavailable; WebRTC signaling not attached")
        return

    mosaic_cfg = pipeline.config.get("mosaic_output") or {}
    webrtc_name = str(mosaic_cfg.get("webrtc_name", "mosaic_webrtc"))

    # Try to get the underlying GStreamer pipeline to use get_by_name()
    gst_pipeline: Gst.Pipeline = None
    webrtc_elem: Gst.Element = None

    # First, try to find the Gst.Pipeline handle
    for attr in ("pipeline", "_pipeline", "gst_pipeline", "_gst_pipeline", "handle", "_handle"):
        try:
            candidate = getattr(ds, attr, None)
        except Exception:
            candidate = None
        if isinstance(candidate, Gst.Pipeline):
            gst_pipeline = candidate
            break
        if candidate is not None and hasattr(candidate, "get_by_name"):
            gst_pipeline = candidate
            break

    # Also check inner attributes
    if gst_pipeline is None:
        for name in dir(ds):
            if name.startswith("__"):
                continue
            try:
                candidate = getattr(ds, name)
            except Exception:
                continue
            if isinstance(candidate, Gst.Pipeline):
                gst_pipeline = candidate
                break
            if hasattr(candidate, "get_by_name"):
                gst_pipeline = candidate
                break

    # Try to get webrtcbin element by name from the GStreamer pipeline
    if gst_pipeline is not None:
        try:
            webrtc_elem = gst_pipeline.get_by_name(webrtc_name)
        except Exception as exc:
            logger.debug("Failed to get webrtcbin by name: %s", exc)

    # Try to get rtsp_out for diagnostics
    if gst_pipeline is not None:
        try:
            rtsp_elem = gst_pipeline.get_by_name("rtsp_out")
            if rtsp_elem is not None:
                try:
                    with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds9_runtime_core.py:_setup_webrtc_signaling",
                                    "message": "rtsp element found",
                                    "data": {"name": rtsp_elem.get_name()},
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback: try pyservicemaker node attributes
    if webrtc_elem is None:
        try:
            node = ds[webrtc_name]
            for attr in ("element", "_element", "obj", "_obj", "gst_element", "_gst_element"):
                candidate = getattr(node, attr, None)
                if candidate is not None and isinstance(candidate, Gst.Element):
                    webrtc_elem = candidate
                    break
        except Exception as exc:
            logger.warning("webrtcbin '%s' not found in pipeline: %s", webrtc_name, exc)

    if webrtc_elem is None:
        logger.warning("webrtcbin element not found; WebRTC signaling not attached")
        return

    # Attach to WebSocket server for signaling
    ws_server.attach_webrtc_endpoint(webrtc_elem)
    logger.info("WebRTC signaling attached to webrtcbin '%s'", webrtc_name)


def _build_rtsp_keyframe_requester(
    pipeline: ds8_pipeline.DS8Pipeline,
    logger: logging.Logger,
) -> Optional[Callable[[str], None]]:
    """Best-effort keyframe/IDR request into the DS9 RTSP encoder pipeline.

    This is used to reduce "ICE connected but black" startups where the browser
    receives RTP bytes but decodes 0 frames until the next IDR arrives.
    """
    if not _GLIB_AVAILABLE or Gst is None:
        return None
    try:
        gi.require_version("GstVideo", "1.0")
        from gi.repository import GstVideo  # type: ignore
    except Exception:
        return None

    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        return None

    # Attempt to find the underlying Gst.Pipeline handle to look up rtsp_out by name.
    gst_pipeline: Optional[Gst.Pipeline] = None
    for attr in ("pipeline", "_pipeline", "gst_pipeline", "_gst_pipeline", "handle", "_handle"):
        try:
            candidate = getattr(ds, attr, None)
        except Exception:
            candidate = None
        if candidate is None:
            continue
        if isinstance(candidate, Gst.Pipeline):
            gst_pipeline = candidate
            break
        if hasattr(candidate, "get_by_name"):
            gst_pipeline = candidate  # type: ignore[assignment]
            break

    if gst_pipeline is None:
        for name in dir(ds):
            if name.startswith("__"):
                continue
            try:
                candidate = getattr(ds, name)
            except Exception:
                continue
            if isinstance(candidate, Gst.Pipeline):
                gst_pipeline = candidate
                break
            if candidate is not None and hasattr(candidate, "get_by_name"):
                gst_pipeline = candidate  # type: ignore[assignment]
                break

    rtsp_out = None
    if gst_pipeline is not None:
        try:
            rtsp_out = gst_pipeline.get_by_name("rtsp_out")
        except Exception:
            rtsp_out = None

    # Fallback: try pyservicemaker node attributes (some wheels don't expose a Gst.Pipeline handle).
    if rtsp_out is None:
        try:
            node = ds["rtsp_out"]
            for attr in ("element", "_element", "obj", "_obj", "gst_element", "_gst_element"):
                candidate = getattr(node, attr, None)
                if candidate is not None and isinstance(candidate, Gst.Element):
                    rtsp_out = candidate
                    break
        except Exception:
            rtsp_out = None

    if rtsp_out is None:
        return None

    # Prefer sending the upstream force-key-unit event from the internal RTP payloader if present.
    pay = None
    try:
        pay = rtsp_out.get_child_by_name("rtsp-video_rtppay")
    except Exception:
        pay = None
    if pay is None:
        try:
            n_children = int(rtsp_out.get_children_count())
        except Exception:
            n_children = 0
        for i in range(n_children):
            try:
                child = rtsp_out.get_child_by_index(i)
            except Exception:
                child = None
            if child is None:
                continue
            try:
                factory = child.get_factory()
                if factory is not None and factory.get_name() == "rtph264pay":
                    pay = child
                    break
            except Exception:
                continue

    def request_keyframe(reason: str) -> None:
        try:
            ev = GstVideo.video_event_new_upstream_force_key_unit(Gst.CLOCK_TIME_NONE, True, 0)
        except Exception:
            logger.debug("Failed to create upstream force-key-unit event", exc_info=True)
            return

        ok = False
        try:
            if pay is not None:
                sink_pad = pay.get_static_pad("sink")
                if sink_pad is not None:
                    ok = bool(sink_pad.send_event(ev))
                else:
                    ok = bool(pay.send_event(ev))
            else:
                ok = bool(rtsp_out.send_event(ev))
        except Exception:
            ok = False

        logger.info("Requested RTSP keyframe (reason=%s ok=%s)", reason, ok)
        try:
            with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "ds9_runtime_core.py:_build_rtsp_keyframe_requester",
                            "message": "rtsp keyframe requested",
                            "data": {"reason": reason, "ok": ok},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    return request_keyframe


def _on_bus_message(
    bus: "Gst.Bus",
    message: "Gst.Message",
    shutdown_event: threading.Event,
    logger: logging.Logger,
) -> bool:
    """Handle GStreamer pipeline bus messages for error/EOS/state diagnostics."""
    if not _GLIB_AVAILABLE or Gst is None:
        return True

    msg_type = message.type

    if msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        src_name = message.src.get_name() if message.src else "unknown"
        logger.error("🚨 Pipeline ERROR from '%s': %s", src_name, err.message)
        logger.error("🚨 Debug: %s", debug)
        #region agent log
        try:
            with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "ds9_runtime_core.py:_on_bus_message",
                            "message": "bus error",
                            "data": {"src": src_name, "error": err.message, "debug": debug},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion
        # Check if this is a source-related error suggesting stream issues
        if any(k in src_name.lower() for k in ("source", "urisrc", "rtspsrc", "decodebin")):
            logger.error("    → Source/decoder error; check RTSP stream connectivity.")
    elif msg_type == Gst.MessageType.EOS:
        logger.warning("⚠️ EOS received on pipeline (unexpected for live sources)")
        #region agent log
        try:
            with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "ds9_runtime_core.py:_on_bus_message",
                            "message": "bus eos",
                            "data": {},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        logger.warning("⚠️ Pipeline warning: %s", warn.message)
        #region agent log
        try:
            with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "ds9_runtime_core.py:_on_bus_message",
                            "message": "bus warning",
                            "data": {"warning": warn.message, "debug": debug},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion
    elif msg_type == Gst.MessageType.STATE_CHANGED:
        if message.src and hasattr(message.src, "get_name"):
            name = message.src.get_name()
            old, new, pending = message.parse_state_changed()
            # Log pipeline-level and RTSP element transitions
            if name == "noesis-ds8" or name == "noesis_rtsp_out" or "rtsp" in name:
                logger.info(
                    "🔄 [%s] state: %s → %s (pending: %s)",
                    name,
                    old.value_nick if old else "?",
                    new.value_nick if new else "?",
                    pending.value_nick if pending else "none",
                )
                #region agent log
                try:
                    with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds9_runtime_core.py:_on_bus_message",
                                    "message": "state change",
                                    "data": {
                                        "src": name,
                                        "old": old.value_nick if old else "?",
                                        "new": new.value_nick if new else "?",
                                        "pending": pending.value_nick if pending else "none",
                                    },
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                #endregion

    return True  # Keep receiving messages


def _start_glib_mainloop(
    pipeline: "ds8_pipeline.DS8Pipeline",
    shutdown_event: threading.Event,
    logger: logging.Logger,
) -> Tuple[Optional[threading.Thread], Optional["GLib.MainLoop"]]:
    """Start GLib main loop in a daemon thread for GStreamer event dispatch.

    Returns (thread, mainloop) tuple. Both may be None if GLib is unavailable.

    pyservicemaker does not expose the underlying Gst.Pipeline, so runtime control
    uses its native wait() path instead of a GLib main loop thread.
    """
    if not _GLIB_AVAILABLE or GLib is None or Gst is None:
        logger.warning("GLib unavailable; skipping main loop (may affect stream reconnection)")
        return None, None

    # pyservicemaker doesn't expose the underlying Gst.Pipeline.
    # Rely on native event handling via wait(); see _start_pyservicemaker_wait_loop.
    logger.info("Using pyservicemaker native event handling (GLib main loop not required)")
    return None, None


def _on_pyservicemaker_message(
    ds_pipeline: Any,
    message: Any,
    logger: logging.Logger,
    shutdown_event: threading.Event,
    state: Optional[Dict[str, Any]] = None,
) -> None:
    """Handle pyservicemaker pipeline messages (EOS, state transitions, etc.)."""
    if not _PYSERVICEMAKER_MSGS:
        return

    # State transition messages are extremely chatty and (depending on the backend build)
    # attribute access can be unsafe. Keep them OFF by default and enable only when needed.
    log_state = (
        str(os.environ.get("NOESIS_DS9_STATE_LOG", "") or os.environ.get("NOESIS_DS8_STATE_LOG", ""))
        .strip()
        .lower()
        in _ENV_TRUE
    )

    if isinstance(message, EOSMessage):
        expected_eos = shutdown_event.is_set() or bool((state or {}).get("expected_eos"))
        if state is not None:
            state["pipeline_eos_seen"] = True
        if expected_eos:
            logger.info("EOS received on pipeline during expected shutdown/finite-source completion")
            return
        logger.warning("⚠️ EOS received on pipeline (unexpected for live sources)")
        if state is not None:
            state["pipeline_failed"] = True
    elif isinstance(message, StateTransitionMessage):
        if log_state:
            # Avoid touching message attributes unless explicitly enabled.
            logger.info("DS9 state transition (origin=%s)", getattr(message, "origin", "unknown"))
    else:
        logger.debug("Pipeline message: %s", type(message).__name__)


def _start_pyservicemaker_wait_loop(
    ds_pipeline: Any,
    shutdown_event: threading.Event,
    logger: logging.Logger,
    state: Optional[Dict[str, Any]] = None,
) -> Optional[threading.Thread]:
    """Start a background thread that calls ds_pipeline.wait() to keep the pipeline alive.

    pyservicemaker's wait() blocks until the pipeline stops and processes internal events.
    Without this, the pipeline may stop processing after initial buffers.

    Returns the thread, or None if unavailable.
    """
    if ds_pipeline is None:
        logger.warning("No DSPipeline available for wait loop")
        return None

    if not hasattr(ds_pipeline, 'wait'):
        logger.warning("DSPipeline doesn't have wait() method; event handling may be limited")
        return None

    def _wait_loop() -> None:
        try:
            #region agent log
            try:
                with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds9_runtime_core.py:_start_pyservicemaker_wait_loop",
                                "message": "wait() entered",
                                "data": {},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
            logger.debug("pyservicemaker wait loop started")
            ds_pipeline.wait()
            logger.info("pyservicemaker wait() returned (pipeline stopped)")
        except Exception:
            logger.exception("pyservicemaker wait loop error")
            #region agent log
            try:
                with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds9_runtime_core.py:_start_pyservicemaker_wait_loop",
                                "message": "wait() exception",
                                "data": {},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
        finally:
            was_signalled = shutdown_event.is_set()
            # Signal shutdown when pipeline stops
            shutdown_event.set()
            expected_finite_eos = bool(
                state is not None
                and state.get("expected_eos")
                and state.get("pipeline_eos_seen")
            )
            if state is not None and not was_signalled and not expected_finite_eos:
                state["pipeline_failed"] = True
            #region agent log
            try:
                with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds9_runtime_core.py:_start_pyservicemaker_wait_loop",
                                "message": "wait() exited",
                                "data": {},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion

    thread = threading.Thread(target=_wait_loop, name="DS9-WaitLoop", daemon=True)
    thread.start()
    logger.info("pyservicemaker wait loop started for pipeline event handling")
    return thread


_ENCODE_LATENCY_FILTER_INSTALLED = False
_ENCODE_LATENCY_FILTER_LOCK = threading.Lock()
_ENCODE_LATENCY_FILTER_THREADS: list[threading.Thread] = []
_ENCODE_LATENCY_FILTER_ORIG_FDS: list[int] = []


def _install_fd_line_filter(
    logger: logging.Logger,
    *,
    fd: int,
    drop_substrings: Tuple[str, ...],
    label: str,
) -> None:
    """Redirect fd to a pipe and forward lines back to the original fd, dropping noise."""
    orig_fd = os.dup(fd)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, fd)
    os.close(w_fd)
    _ENCODE_LATENCY_FILTER_ORIG_FDS.append(orig_fd)

    def _reader() -> None:
        buf = b""
        try:
            with os.fdopen(r_fd, "rb", buffering=0) as r:
                while True:
                    chunk = r.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line += b"\n"
                        try:
                            text = line.decode("utf-8", errors="replace")
                        except Exception:
                            text = ""
                        if any(token in text for token in drop_substrings):
                            continue
                        try:
                            os.write(orig_fd, line)
                        except Exception:
                            return
                    # Avoid unbounded buffering if a producer writes without newlines.
                    if len(buf) > 8192:
                        try:
                            os.write(orig_fd, buf)
                        except Exception:
                            return
                        buf = b""
        except Exception:
            logger.debug("fd filter reader failed (%s)", label, exc_info=True)
        finally:
            if buf:
                try:
                    os.write(orig_fd, buf)
                except Exception:
                    pass

    t = threading.Thread(target=_reader, name=f"FDFilter-{label}", daemon=True)
    t.start()
    _ENCODE_LATENCY_FILTER_THREADS.append(t)


def _install_encode_latency_suppression(logger: logging.Logger) -> None:
    """Suppress gst-nvvideo4linux2 KPI prints (e.g. 'Encode Latency = ...')."""
    global _ENCODE_LATENCY_FILTER_INSTALLED

    with _ENCODE_LATENCY_FILTER_LOCK:
        if _ENCODE_LATENCY_FILTER_INSTALLED:
            return

        flag = os.environ.get("NOESIS_SUPPRESS_ENCODE_LATENCY")
        if flag is not None and str(flag).strip().lower() not in _ENV_TRUE:
            return

        # Only wrap stdout/stderr when NVDS latency measurement is enabled; otherwise
        # this noise does not appear and we avoid redirecting file descriptors.
        if str(os.environ.get("NVDS_ENABLE_LATENCY_MEASUREMENT", "")).strip().lower() not in _ENV_TRUE:
            return

        drop = ("Encode Latency =", "KPI: v4l2:")
        _install_fd_line_filter(logger, fd=1, drop_substrings=drop, label="stdout")
        _install_fd_line_filter(logger, fd=2, drop_substrings=drop, label="stderr")
        _ENCODE_LATENCY_FILTER_INSTALLED = True
        logger.info("Suppressed encoder KPI prints (NOESIS_SUPPRESS_ENCODE_LATENCY=0 to disable)")


def main() -> int:
    os.environ.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "1")
    os.environ.setdefault("NOESIS_DEPTH_ENABLE_SECONDS", "0")
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("ds9.runtime")
    ws_fallback_tries_raw = os.environ.get("NOESIS_WS_PORT_FALLBACK_TRIES", "32")
    try:
        ws_fallback_tries = int(str(ws_fallback_tries_raw).strip() or "32")
    except Exception:
        ws_fallback_tries = 32
    args.ws_port = _select_ws_port(args.ws_host, int(args.ws_port), ws_fallback_tries, logger)
    os.environ["NOESIS_WS_PORT"] = str(int(args.ws_port))
    # DeepStream's gst-nvvideo4linux2 encoder plugin can emit extremely noisy
    # "Encode Latency = ..." prints when NVDS latency measurement is enabled.
    # Suppress those lines by default; opt-out with NOESIS_SUPPRESS_ENCODE_LATENCY=0.
    try:
        _install_encode_latency_suppression(logger)
    except Exception:
        # Defensive: never break runtime due to log filtering helpers.
        pass
    runtime_state: Dict[str, Any] = {"pipeline_failed": False}
    pgie_size: Optional[str] = None

    if args.size is not None and str(args.pgie_profile) not in _SIZED_PGIE_PROFILES:
        sized = ", ".join(_SIZED_PGIE_PROFILES)
        raise SystemExit(f"[FATAL] --size is only valid with --pgie-profile in: {sized}")
    if str(args.pgie_profile) in _SIZED_PGIE_PROFILES:
        pgie_size = (args.size or "m").strip().lower()

    # Install SIGINT/SIGTERM handling early (before DS/GStreamer init), because
    # some backends install their own handlers/masks which can make `timeout(1)`
    # leave behind orphaned processes that keep ports bound.
    shutdown_event = threading.Event()

    def _signal_handler(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; initiating shutdown", signum)
        # Best-effort debug breadcrumb (useful when the backend swallows SIGTERM).
        try:
            with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H0",
                            "location": "ds9_runtime_core.py:_signal_handler",
                            "message": "signal received",
                            "data": {"signum": int(signum)},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        shutdown_event.set()
        # `timeout(1)` uses SIGTERM; DS/GStreamer backends can hang shutdown (and sometimes
        # starve Python threads), so hard-exit to avoid leaving orphaned processes/ports.
        if signum == signal.SIGTERM:
            try:
                with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H0",
                                "location": "ds9_runtime_core.py:_signal_handler",
                                "message": "sigterm hard exit",
                                "data": {},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            os._exit(0)

    # Some DS/GStreamer backends manipulate signal masks; ensure SIGINT/SIGTERM are unblocked.
    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    except Exception:
        pass

    signal.signal(signal.SIGINT, _signal_handler)
    # Prefer SIGTERM default behavior so external supervisors (e.g. `timeout(1)`) can
    # always terminate the process even if Python threads/GIL are starved by GI callbacks.
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except Exception:
        signal.signal(signal.SIGTERM, _signal_handler)

    tracking_mode = _resolve_tracking_mode(args)
    os.environ["NOESIS_TRACKING_MODE"] = tracking_mode
    if tracking_mode not in _TRACKING_MODES:
        logger.warning("Normalized tracking mode '%s' is unknown; defaulting to baseline", tracking_mode)
        tracking_mode = "baseline"

    pipeline_path, cameras_path = _resolve_pipeline_and_camera_paths(args, tracking_mode, logger)
    pipeline_path = pipeline_path.expanduser().resolve()
    cameras_path = cameras_path.expanduser().resolve()

    # Ensure process CWD is the repo root so relative paths in YAML (engines, configs)
    # resolve correctly for DS9 plugins and hooks.
    try:
        os.chdir(REPO_ROOT)
    except Exception:
        logging.getLogger("ds9.runtime").warning("Unable to chdir to REPO_ROOT %s", REPO_ROOT)

    if not pipeline_path.exists():
        logger.error("Pipeline configuration not found: %s", pipeline_path)
        return 1
    if tracking_mode == "v3dt":
        if not _ensure_v3dt_meta_extension(logger):
            return 1
        if not _validate_v3dt_tracking_guardrails(pipeline_path, logger):
            return 1
    else:
        _warn_baseline_with_v3dt_tracker(pipeline_path, logger)

    skip_cuda_preflight = str(os.environ.get("NOESIS_SKIP_CUDA_PREFLIGHT", "")).strip().lower() in _ENV_TRUE
    stub_pipeline = (
        str(os.environ.get("NOESIS_DS9_STUB_PIPELINE", "") or os.environ.get("NOESIS_DS8_STUB_PIPELINE", ""))
        .strip()
        .lower()
        in _ENV_TRUE
    )
    if not stub_pipeline and not skip_cuda_preflight:
        cuda_ok, cuda_msg = _cuda_runtime_preflight()
        if not cuda_ok:
            logger.error(
                "CUDA preflight failed (%s). Aborting before DS9 pipeline startup to avoid unstable runtime crashes.",
                cuda_msg,
            )
            logger.error("Set NOESIS_SKIP_CUDA_PREFLIGHT=1 to bypass this guardrail.")
            return 1
        logger.info("CUDA preflight passed (%s)", cuda_msg)
    elif skip_cuda_preflight:
        logger.warning("Skipping CUDA preflight because NOESIS_SKIP_CUDA_PREFLIGHT is enabled")

    base_pipeline_path = pipeline_path
    pipeline_path = _materialize_effective_pipeline_yaml(
        base_pipeline_path,
        str(args.pgie_profile),
        logger,
        pgie_size=pgie_size,
        tracking_mode=tracking_mode,
    )
    depth_registration_path = _resolve_depth_registration_path(args, pipeline_path=pipeline_path)
    logger.info("Building DS9 pipeline from %s (base: %s)", pipeline_path, base_pipeline_path)
    if tracking_mode == "baseline":
        if not _ensure_baseline_depth_tracking_guardrails(pipeline_path, logger):
            return 1
        if not depth_registration_path.exists():
            logger.error(
                "Baseline tracking requires a prebuilt depth registration artifact: %s",
                depth_registration_path,
            )
            return 1

    if not _maybe_autogen_v3dt_caminfo(pipeline_path, cameras_path, logger):
        return 1
    if not _validate_dewarper_intrinsics_sync(pipeline_path, cameras_path, logger):
        return 1

    os.environ["NOESIS_DS9_PIPELINE_CONFIG"] = str(pipeline_path)
    os.environ["NOESIS_DS8_PIPELINE_CONFIG"] = str(pipeline_path)
    try:
        from noesis.server import analytics_api

        os.environ.setdefault(analytics_api.ANALYTICS_CONFIG_ENV, str(REPO_ROOT / "config" / "nvdsanalytics.yaml"))
        analytics_api._load_config(force=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    prebuilt_rest_app = None
    if args.enable_rest:
        try:
            if not _preload_rest_server_runtime():
                logger.error("Failed to preload REST server runtime")
                return 1
            prebuilt_rest_app = _build_rest_app()
            logger.info("REST app and server runtime prebuilt before DeepStream pipeline startup")
        except Exception:
            logger.exception("Failed to build REST app")
            return 1

    camera_labels = _load_camera_labels(cameras_path)
    storage_manager = _build_storage_manager(args)
    auto_calibrate_lock = threading.Lock()

    def _resolve_auto_calibrate_cameras(camera_id: Optional[str]) -> list[str]:
        if camera_id is None or not str(camera_id).strip():
            ordered = [
                name
                for _, name in sorted(camera_labels.items(), key=lambda item: int(item[0]))
                if isinstance(name, str)
            ]
            seen: set[str] = set()
            cameras: list[str] = []
            for name in ordered:
                if name in seen:
                    continue
                cameras.append(name)
                seen.add(name)
            return cameras

        request_camera = str(camera_id).strip()
        canonical_camera = request_camera
        try:
            cam_idx = int(request_camera)
        except Exception:
            cam_idx = None
        if cam_idx is not None and cam_idx in camera_labels:
            canonical_camera = camera_labels[cam_idx]
        else:
            reverse_labels = {v: k for k, v in camera_labels.items()}
            if canonical_camera in reverse_labels:
                canonical_camera = request_camera
        return [canonical_camera]

    def _read_latest_depth_ts(camera_id: str) -> int:
        if storage_manager is None:
            return 0
        try:
            payload = storage_manager.load_latest_depth(camera_id, None)
        except Exception:
            return 0
        if not isinstance(payload, dict):
            return 0
        try:
            return int(payload.get("ts", 0) or 0)
        except Exception:
            return 0

    def _ds8_auto_calibrate_handler(camera_id: Optional[str] = None) -> Dict[str, Any]:
        if not auto_calibrate_lock.acquire(blocking=False):
            return {"ok": False, "results": [], "updated": [], "error": "busy"}
        try:
            if not pipeline.activated:
                return {"ok": False, "results": [], "updated": [], "error": "pipeline_not_running"}
            if storage_manager is None:
                return {"ok": False, "results": [], "updated": [], "error": "depth_source_unavailable"}
            depth_branch_present = bool(
                pipeline.depth_gate_attach and pipeline.depth_gate_attach in pipeline.components
            )
            if not depth_branch_present:
                return {"ok": False, "results": [], "updated": [], "error": "depth_branch_unavailable"}

            cameras = _resolve_auto_calibrate_cameras(camera_id)
            if not cameras:
                return {"ok": False, "results": [], "updated": [], "error": "no_cameras_configured"}

            baseline_ts = {cam: _read_latest_depth_ts(cam) for cam in cameras}

            enable_env = os.environ.get("NOESIS_AUTOCALIB_ENABLE_SECONDS", "10")
            try:
                enable_seconds = int(float(str(enable_env).strip()))
            except Exception:
                enable_seconds = 10
            enable_seconds = max(1, min(15, enable_seconds))

            if not pipeline.depth_enabled:
                try:
                    ds8_pipeline.enable_depth(seconds=enable_seconds)
                except Exception:
                    logger.warning("Auto-calibrate failed to enable depth burst", exc_info=True)

            wait_timeout_s = min(float(enable_seconds) + 2.0, 18.0)
            deadline = time.time() + wait_timeout_s
            fresh = set()
            while time.time() < deadline and len(fresh) < len(cameras):
                for cam in cameras:
                    latest_ts = _read_latest_depth_ts(cam)
                    if latest_ts > baseline_ts.get(cam, 0):
                        fresh.add(cam)
                if len(fresh) >= len(cameras):
                    break
                time.sleep(0.12)

            try:
                from scripts.auto_calibrate_from_depth import auto_calibrate_from_latest_depth
            except Exception as exc:
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": f"import_failed: {exc}",
                }

            try:
                res = auto_calibrate_from_latest_depth(cameras, persist=False)
            except Exception as exc:
                return {"ok": False, "results": [], "updated": [], "error": str(exc) or "calibration_failed"}

            results = res.get("results") if isinstance(res, dict) else []
            updated: list[str] = []
            persist_failed = False

            for entry in results or []:
                if not isinstance(entry, dict):
                    continue
                if not entry.get("ok"):
                    continue
                cam = entry.get("cameraId")
                e_mat = entry.get("E")
                if not cam or not e_mat:
                    continue
                persist_result = calibration_provider.set_extrinsics(str(cam), E=e_mat)
                if persist_result.get("ok"):
                    updated.append(cam)
                else:
                    logger.warning(
                        "Auto-calibrate persist failed camera=%s error=%s",
                        cam,
                        persist_result.get("error"),
                    )
                    persist_failed = True

            top_error = res.get("error") if isinstance(res, dict) else None
            if not updated and persist_failed and not top_error:
                top_error = "persist_failed"

            if updated:
                try:
                    storage_manager.calibration_bundle = calibration_provider.calibration_bundle()
                except Exception:
                    logger.debug("Unable to refresh storage calibration bundle", exc_info=True)
                try:
                    bundle = calibration_provider.calibration_bundle()
                    ws_server.broadcast_sync({"type": "calibration-bundle", "data": bundle})
                except Exception:
                    logger.debug("Failed to broadcast calibration bundle", exc_info=True)

            return {
                "ok": bool(updated),
                "results": results or [],
                "updated": updated,
                "error": top_error,
            }
        finally:
            auto_calibrate_lock.release()

    def _ds8_ma_depth_provider(
        cam_id: str,
        ts_max_us: Optional[object] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        request_camera = str(cam_id).strip()
        camera_key = request_camera
        canonical_camera = camera_key
        alt_keys: List[str] = []
        norm_ts_max: Optional[int] = None
        try:
            cam_idx = int(camera_key)
            if cam_idx in camera_labels:
                canonical_camera = camera_labels[cam_idx]
                camera_key = canonical_camera
                alt_keys.append(str(cam_idx))
        except Exception:
            reverse_labels = {v: k for k, v in camera_labels.items()}
            cam_idx = reverse_labels.get(camera_key)
            if cam_idx is not None:
                canonical_camera = camera_key
                alt_keys.append(str(cam_idx))

        def _normalize_ts_max_us(value: Optional[object]) -> Optional[int]:
            if value is None:
                return None
            try:
                if isinstance(value, str):
                    text = value.strip()
                    if not text:
                        return None
                    raw = int(float(text))
                else:
                    raw = int(value)  # type: ignore[arg-type]
            except Exception:
                return None
            if raw < 0:
                return None
            if raw < 1_000_000_000:
                return raw * 1_000_000
            if raw < 1_000_000_000_000:
                return raw * 1000
            return raw

        norm_ts_max = _normalize_ts_max_us(ts_max_us)

        def _payload_has_valid_depth(payload: Dict[str, Any]) -> bool:
            """Best-effort guard against invalid all-zero/NaN depth payloads."""
            try:
                shape = payload.get("shape")
                if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
                    return False
                height = int(shape[0] or 0)
                width = int(shape[1] or 0)
                if height <= 0 or width <= 0:
                    return False
                depth_b64 = payload.get("depth_b64") or payload.get("depth_z_b64")
                if not isinstance(depth_b64, str) or not depth_b64:
                    return False
                import base64

                raw = base64.b64decode(depth_b64)
                arr = np.frombuffer(raw, dtype=np.float32)
                needed = height * width
                if arr.size < needed:
                    return False
                arr = arr[:needed]
                finite = np.isfinite(arr)
                if not finite.any():
                    return False
                # Floorplan generation uses depth>0.1m as a validity threshold; align with it here.
                return bool(np.any(arr[finite] > 0.1))
            except Exception:
                return False

        def _maybe_attach_normals(payload: Dict[str, Any]) -> None:
            if not isinstance(payload, dict):
                return
            flag = os.environ.get("NOESIS_MAPANYTHING_NORMALS_ENABLE", "1")
            if str(flag).strip().lower() not in ("1", "true", "yes", "on"):
                return
            if storage_manager is None:
                return
            space = os.environ.get("NOESIS_MAPANYTHING_NORMALS_SPACE", "camera")
            dtype = os.environ.get("NOESIS_MAPANYTHING_NORMALS_DTYPE", "float16")
            try:
                storage_manager.attach_normals_to_payload(canonical_camera, payload, space=space, dtype=dtype)
            except Exception:
                logger.debug("Failed to attach depth normals for camera %s", canonical_camera, exc_info=True)

        def _response(
            served_from_cache: bool,
            payload: Optional[Dict[str, Any]] = None,
            ts_us: Optional[int] = None,
            error: Optional[str] = None,
        ) -> Dict[str, Any]:
            ts_val: Any = ts_us
            if ts_val is None and payload is not None:
                try:
                    ts_val = int(payload.get("ts", 0) or 0)
                except Exception:
                    ts_val = payload.get("ts", 0)
            resp: Dict[str, Any] = {
                "type": "ma_depth_response",
                "camera": canonical_camera,
                "served_from_cache": bool(served_from_cache),
                "ts_us": int(ts_val or 0),
            }
            if request_id:
                resp["request_id"] = request_id
            if payload is not None:
                _maybe_attach_normals(payload)
                resp["payload"] = payload
            if error:
                resp["error"] = error
            resp["ok"] = error is None
            return resp

        if storage_manager is None:
            return _response(False, error="depth_source_unavailable")

        def _load_latest(camera_id: str, ts_cutoff: Optional[int]) -> Optional[Dict[str, Any]]:
            try:
                return storage_manager.load_latest_depth(camera_id, ts_cutoff)
            except Exception:
                return None

        keys_to_check = [camera_key] + [k for k in alt_keys if k and k != camera_key]

        cached: Optional[Dict[str, Any]] = None
        cached_ts: Optional[int] = None
        if norm_ts_max is not None:
            for key in keys_to_check:
                cached = _load_latest(key, norm_ts_max)
                if cached is not None:
                    break
            if cached is not None and _payload_has_valid_depth(cached):
                try:
                    cached_ts = int(cached.get("ts", 0) or 0)
                except Exception:
                    cached_ts = 0
                return _response(True, payload=cached, ts_us=cached_ts)
            cached = None
        else:
            cached = _load_latest(camera_key, None)
            if cached is None:
                for alt_key in alt_keys:
                    cached = _load_latest(alt_key, None)
                    if cached is not None:
                        break
            if cached is not None and _payload_has_valid_depth(cached):
                try:
                    cached_ts = int(cached.get("ts", 0) or 0)
                except Exception:
                    cached_ts = 0
            else:
                cached = None
                cached_ts = None

        depth_branch_present = bool(pipeline.depth_gate_attach and pipeline.depth_gate_attach in pipeline.components)
        if not depth_branch_present:
            if cached is not None:
                return _response(True, payload=cached, ts_us=cached_ts, error="depth_branch_unavailable")
            return _response(False, error="depth_branch_unavailable")

        baseline_ts_by_key: Dict[str, int] = {}
        for key in keys_to_check:
            baseline_ts_by_key[key] = 0
            payload = _load_latest(key, None)
            if payload is None:
                continue
            try:
                baseline_ts_by_key[key] = int(payload.get("ts", 0) or 0)
            except Exception:
                baseline_ts_by_key[key] = 0

        # Trigger a short depth burst and wait for a newer cached snapshot to land.
        enable_env = os.environ.get("NOESIS_DEPTH_RPC_ENABLE_SECONDS", "4")
        try:
            enable_seconds = int(str(enable_env).strip())
        except Exception:
            enable_seconds = 4
        enable_seconds = max(1, min(20, enable_seconds))
        try:
            ds8_pipeline.enable_depth(seconds=enable_seconds)
        except Exception as exc:
            if cached is not None:
                return _response(True, payload=cached, ts_us=cached_ts, error=str(exc) or "depth_enable_failed")
            return _response(False, error=str(exc) or "depth_enable_failed")

        wait_env = os.environ.get("NOESIS_DEPTH_RPC_WAIT_TIMEOUT_SECONDS", "")
        try:
            wait_timeout_s = float(str(wait_env).strip()) if str(wait_env).strip() else min(12.0, float(enable_seconds) + 4.0)
        except Exception:
            wait_timeout_s = min(12.0, float(enable_seconds) + 4.0)
        wait_timeout_s = max(1.0, min(30.0, wait_timeout_s))
        deadline = time.time() + wait_timeout_s
        while time.time() < deadline:
            for key in keys_to_check:
                payload = _load_latest(key, None)
                if payload is None:
                    continue
                if not _payload_has_valid_depth(payload):
                    continue
                try:
                    payload_ts = int(payload.get("ts", 0) or 0)
                except Exception:
                    payload_ts = 0
                if payload_ts <= baseline_ts_by_key.get(key, 0):
                    continue
                return _response(False, payload=payload, ts_us=payload_ts)
            time.sleep(0.12)

        if cached is not None:
            return _response(True, payload=cached, ts_us=cached_ts, error="timeout_waiting_for_depth")
        payload = _load_latest(camera_key, None)
        if payload is None:
            for alt_key in alt_keys:
                payload = _load_latest(alt_key, None)
                if payload is not None:
                    break
        if payload is None or not _payload_has_valid_depth(payload):
            return _response(False, error="timeout_waiting_for_depth")
        try:
            payload_ts = int(payload.get("ts", 0) or 0)
        except Exception:
            payload_ts = 0
        return _response(True, payload=payload, ts_us=payload_ts, error="timeout_waiting_for_depth")

    def _ds8_floorplan_provider(
        camera: Optional[str] = None,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
        **_ignored: object,
    ) -> Dict[str, Any]:
        request_camera = str(camera or "").strip()
        camera_id = request_camera
        alt_keys: List[str] = []
        try:
            cam_idx = int(camera_id)
            if cam_idx in camera_labels:
                camera_id = camera_labels[cam_idx]
                alt_keys.append(str(cam_idx))
        except Exception:
            reverse_labels = {v: k for k, v in camera_labels.items()}
            cam_idx = reverse_labels.get(camera_id)
            if cam_idx is not None:
                alt_keys.append(str(cam_idx))

        if not camera_id:
            return {"error": "camera_required", "ts": int(time.time() * 1_000_000)}
        if storage_manager is None:
            return {"error": "depth_source_unavailable", "camera_id": camera_id}

        keys_to_check = [camera_id] + [key for key in alt_keys if key and key != camera_id]

        def _generate_for_key(key: str) -> Dict[str, Any]:
            return storage_manager.generate_topdown_floorplan(
                key,
                max_age_sec=max_age_sec,
                grid_res_m=grid_res_m,
                max_extent_m=max_extent_m,
                cache_only=cache_only,
            )

        def _generate_with_alternates() -> Dict[str, Any]:
            first_exc: Optional[Exception] = None
            try:
                return _generate_for_key(camera_id)
            except Exception as exc:
                first_exc = exc
            for alt_key in alt_keys:
                try:
                    return _generate_for_key(alt_key)
                except Exception:
                    continue
            if first_exc is not None:
                raise first_exc
            return {"error": "floorplan_failed", "camera_id": camera_id, "ts": int(time.time() * 1_000_000)}

        def _needs_live_depth_burst(payload: Mapping[str, Any]) -> bool:
            if cache_only:
                return False
            try:
                requested_max_age = float(max_age_sec)
            except Exception:
                requested_max_age = 60.0
            if requested_max_age <= 0.0:
                return True
            error = str(payload.get("error") or "").strip().lower()
            return error in {"no_depth", "stale_depth", "load_failed", "invalid_snapshot"}

        try:
            payload = _generate_with_alternates()
        except Exception as exc:
            logger.warning(
                "DS9 floorplan provider failed (camera=%s, cache_only=%s): %s",
                camera_id,
                cache_only,
                exc,
            )
            return {"error": str(exc) or "floorplan_failed", "camera_id": camera_id}
        if not _needs_live_depth_burst(payload):
            return payload

        depth_branch_present = bool(pipeline.depth_gate_attach and pipeline.depth_gate_attach in pipeline.components)
        if not depth_branch_present:
            payload = dict(payload)
            payload.setdefault("camera_id", camera_id)
            payload["error"] = "depth_branch_unavailable"
            payload["depth_burst_triggered"] = False
            return payload

        baseline_ts_by_key = {key: _read_latest_depth_ts(key) for key in keys_to_check}
        enable_env = os.environ.get(
            "NOESIS_FLOORPLAN_DEPTH_ENABLE_SECONDS",
            os.environ.get("NOESIS_DEPTH_RPC_ENABLE_SECONDS", "4"),
        )
        try:
            enable_seconds = int(float(str(enable_env).strip()))
        except Exception:
            enable_seconds = 4
        enable_seconds = max(1, min(20, enable_seconds))
        try:
            ds8_pipeline.enable_depth(seconds=enable_seconds)
        except Exception as exc:
            payload = dict(payload)
            payload.setdefault("camera_id", camera_id)
            payload["error"] = str(exc) or "depth_enable_failed"
            payload["depth_burst_triggered"] = False
            return payload

        deadline = time.time() + min(12.0, float(enable_seconds) + 4.0)
        fresh_depth = False
        while time.time() < deadline:
            for key in keys_to_check:
                latest_ts = _read_latest_depth_ts(key)
                if latest_ts > baseline_ts_by_key.get(key, 0):
                    fresh_depth = True
                    break
            if fresh_depth:
                break
            time.sleep(0.12)

        try:
            refreshed = dict(_generate_with_alternates())
        except Exception as exc:
            logger.warning(
                "DS9 floorplan provider failed after depth burst (camera=%s): %s",
                camera_id,
                exc,
            )
            refreshed = {"error": str(exc) or "floorplan_failed", "camera_id": camera_id}
        refreshed["depth_burst_triggered"] = True
        refreshed["depth_burst_fresh"] = bool(fresh_depth)
        if refreshed.get("error") and not fresh_depth:
            refreshed.setdefault("details", "timeout_waiting_for_depth")
        return refreshed

    pipeline = ds8_pipeline.build_pipeline(pipeline_path)
    runtime_state["expected_eos"] = _pipeline_expects_finite_source_eos(pipeline.config)
    runtime_state["pipeline_eos_seen"] = False
    setattr(pipeline, "camera_labels", camera_labels)
    if getattr(pipeline, "ds_pipeline", None) is None:
        logger.error("pyservicemaker unavailable; DS9 runtime cannot continue")
        return 1
    try:
        from noesis.server import analytics_api

        analytics_cfg = analytics_api._load_config()  # type: ignore[attr-defined]
        stage_cfg = (analytics_cfg.get("analytics") or {}).get("stages", {}).get("exclude") or {}
        if stage_cfg:
            analytics_api._sync_exclude_stage("exclude", stage_cfg)  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Unable to sync exclusion config at startup", exc_info=True)

    # Parse mosaic_output toggles from the *built pipeline config* (source of truth).
    # Do not re-apply env overrides here: env vars are consumed during build in ds8_pipeline,
    # and re-applying them here can desync runtime behavior from the actual pipeline graph.
    mosaic_cfg = pipeline.config.get("mosaic_output") or {}
    rtsp_port = int(mosaic_cfg.get("rtsp_port", 8554) or 8554)
    rtsp_path = str(mosaic_cfg.get("rtsp_path", "mosaic")).strip() or "mosaic"
    mosaic_webrtc_enabled = bool(mosaic_cfg.get("mosaic_webrtc_enabled", False))

    rtsp_built = "rtsp_out" in getattr(pipeline, "components", {})

    logger.info(
        "Mosaic output toggles (effective): RTSP=%s (built=%s), WebRTC_Gateway=%s",
        bool(mosaic_cfg.get("rtsp_enabled", False)),
        rtsp_built,
        mosaic_webrtc_enabled,
    )
    #region agent log
    try:
        with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "ds9_runtime_core.py:main",
                        "message": "mosaic toggles",
                        "data": {
                            "rtsp_enabled": bool(mosaic_cfg.get("rtsp_enabled", False)),
                            "rtsp_port": rtsp_port,
                            "rtsp_path": rtsp_path,
                            "webrtc_enabled": mosaic_webrtc_enabled,
                            "rtsp_built": rtsp_built,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    #endregion

    streammux_cfg = pipeline.config.get("streammux") or {}
    try:
        streammux_size = (
            int((streammux_cfg or {}).get("width", 0) or 0),
            int((streammux_cfg or {}).get("height", 0) or 0),
        )
    except Exception:
        streammux_size = (0, 0)
    if streammux_size[0] <= 0 or streammux_size[1] <= 0:
        streammux_size = (1920, 1080)

    calibration_provider = CalibrationManager(
        cameras_yaml_path=cameras_path,
        camera_calibration_json_path=REPO_ROOT / "config" / "camera_calibration.json",
        ply_alignment_json_path=REPO_ROOT / "config" / "ply_alignment.json",
        streammux_size=streammux_size,
    )
    calibration_provider.set_camera_labels(camera_labels)
    setattr(pipeline, "bev_calibration", calibration_provider)
    if calibration_provider.pose_only_enabled():
        pose_errors = calibration_provider.validate_pose_coverage()
        if pose_errors:
            logger.error("NOESIS_CALIBRATION_POSE_ONLY=1 startup validation failed; missing/invalid pose for cameras:")
            for camera_id, reason in sorted(pose_errors.items()):
                logger.error("  camera=%s reason=%s", camera_id, reason)
            logger.error("Aborting startup due to strict pose-only calibration mode.")
            return 1
        logger.warning("Strict pose-only calibration mode enabled (NOESIS_CALIBRATION_POSE_ONLY=1).")
    try:
        storage_manager.calibration_bundle = calibration_provider.calibration_bundle()
    except Exception:
        logger.debug("Unable to seed calibration bundle on storage manager", exc_info=True)
    depth_registration_manager: DepthRegistrationManager | None = None
    depthless_reid_smoke = _allow_depthless_reid_smoke(pipeline.config)
    if tracking_mode == "baseline" and not depthless_reid_smoke:
        try:
            depth_registration_manager = _load_depth_registration_manager(
                path=depth_registration_path,
                pipeline_path=pipeline_path,
                pipeline_cfg=pipeline.config,
                calibration_provider=calibration_provider,
                camera_labels=camera_labels,
                logger=logger,
            )
        except DepthRegistrationError as exc:
            logger.error("Baseline tracking requires a valid depth registration artifact: %s", exc)
            return 1
    elif depthless_reid_smoke:
        logger.warning("Skipping depth registration load for explicit ReID smoke validation config")
    stable_id_mgr = _build_stable_id_manager(logger, pipeline_config=pipeline.config)
    if stable_id_mgr is None:
        logger.error("Stable ID manager is required for zero-copy hard-cutover; aborting startup")
        return 1
    # Ensure occupancy publisher slot exists for telemetry hooks; real publisher can be bound later.
    bind_occupancy_publisher(pipeline, None)
    # Stable ID manager is optional; attach slot so hooks can discover it.
    setattr(pipeline, "stable_id_mgr", stable_id_mgr)
    try:
        from noesis.server import reid_api

        reid_api.register_reid_manager_getter(lambda: getattr(pipeline, "stable_id_mgr", None))
    except Exception:
        logger.debug("ReID API registration skipped", exc_info=True)
    # Intrinsics are served via the calibration bundle (`_CalibrationProvider`) rather than per-frame user meta.

    trails_cfg: Dict[str, Any] = {}
    try:
        vis_cfg = pipeline.config.get("visualization") or {}
        if isinstance(vis_cfg, dict):
            raw_trails = vis_cfg.get("trails") or {}
            if isinstance(raw_trails, dict):
                trails_cfg = raw_trails
    except Exception:
        trails_cfg = {}
    trail_settings = hooks.TrailOverlayConfig.from_mapping(trails_cfg)
    try:
        hooks.attach_trail_overlay_hook(pipeline, config=trails_cfg)
    except Exception:
        logger.exception("Failed to attach DS9 trail overlay hook")
    try:
        hooks.attach_pose_keypoint_overlay_hook(pipeline)
    except Exception:
        logger.exception("Failed to attach DS9 pose keypoint overlay hook")

    bev_cfg = pipeline.config.get("bev") or {}
    bev_smoothing_cfg = bev_cfg.get("smoothing") if isinstance(bev_cfg, dict) else None
    if not isinstance(bev_smoothing_cfg, dict):
        bev_smoothing_cfg = None
    bev_frame = None
    if isinstance(bev_cfg, dict):
        bev_frame = bev_cfg.get("frame") or bev_cfg.get("frame_mode")
    bev_frame_env = os.environ.get("NOESIS_BEV_FRAME")
    if bev_frame_env:
        bev_frame = bev_frame_env
    if not bev_frame:
        bev_frame = POSE_V1_FRAME_BACKEND_WORLD_M
    # JPEG BEV binary delivery retired (meta-only is the supported baseline per contracts + design decisions).
    # The old jpeg_enabled / NOESIS_BEV_JPEG_ENABLED knobs are ignored; only frame mode remains relevant.
    logger.info("BEV JPEG output retired (meta-only mode). frame mode=%s", bev_frame)

    ws_server = WebSocketServer(
        host=args.ws_host,
        port=args.ws_port,
        stats_callback=None,
        initial_trail_state=bool(trail_settings.enabled),
    )
    def _ws_trail_settings_getter() -> Dict[str, Any]:
        cfg = dict(trails_cfg or {})
        cfg["enabled"] = bool(ws_server.initial_trail_state)
        return cfg
    ws_server.trail_settings_getter = _ws_trail_settings_getter
    ws_server.stats_callback = _build_stats_callback(
        pipeline,
        camera_labels,
        ws_metrics_getter=ws_server.get_boundary_serialization_metrics,
        ws_metrics_resetter=ws_server.reset_boundary_serialization_metrics,
    )
    trail_processor = getattr(pipeline, "trail_overlay_processor", None)
    bev_renderer: Optional[BevRenderer] = None

    def _toggle_handler(toggle_name: str, enabled: bool) -> None:
        if toggle_name != "trail_visualization_enabled":
            return
        if trail_processor is not None:
            try:
                trail_processor.set_enabled(bool(enabled))
            except Exception:
                logger.exception("Failed to toggle trail overlay to %s", enabled)
        if bev_renderer is not None:
            try:
                bev_renderer.set_trails_enabled(bool(enabled))
            except Exception:
                logger.exception("Failed to toggle BEV trails to %s", enabled)
        try:
            ws_server.initial_trail_state = bool(enabled)
        except Exception:
            pass

    ws_server.toggle_callback = _toggle_handler
    ws_server.ma_depth_provider = _ds8_ma_depth_provider
    ws_server.floorplan_provider = _ds8_floorplan_provider
    ws_server.calibration_getter = calibration_provider.calibration_bundle
    ws_server.auto_calibrate_handler = _ds8_auto_calibrate_handler

    def _broadcast_calibration_bundle() -> None:
        try:
            bundle = calibration_provider.calibration_bundle()
        except Exception:
            logger.debug("Failed to build calibration bundle for broadcast", exc_info=True)
            return
        if storage_manager is not None:
            try:
                storage_manager.calibration_bundle = bundle
            except Exception:
                logger.debug("Failed to update storage calibration bundle", exc_info=True)
        try:
            ws_server.broadcast_sync({"type": "calibration-bundle", "data": bundle})
        except Exception:
            logger.debug("Failed to broadcast calibration bundle", exc_info=True)

    def _resolve_ws_camera_id(raw: object) -> Optional[str]:
        if isinstance(raw, dict):
            # Menon sometimes sends {"id","name"} objects.
            raw = raw.get("id") or raw.get("name") or raw.get("cameraId") or raw.get("camera")
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    def _camera_to_source_id(camera_id: str) -> Optional[int]:
        text = str(camera_id).strip()
        if not text:
            return None
        try:
            idx = int(text)
        except Exception:
            idx = None
        if idx is not None and idx in camera_labels:
            return idx
        for sid, cam in camera_labels.items():
            if cam == text:
                return int(sid)
        return None

    def _normalize_pose_payload(raw_pose: Any) -> Optional[Dict[str, Any]]:
        return normalize_pose_v1(raw_pose)

    def _set_extrinsics_handler(req: Dict[str, Any]) -> Dict[str, Any]:
        cam_id = _resolve_ws_camera_id(req.get("cameraId") or req.get("camera") or req.get("camId") or req.get("id"))
        if not cam_id:
            return {"ok": False, "error": "cameraId_required"}

        strict_pose_only = bool(calibration_provider.pose_only_enabled())
        raw_pose = req.get("pose")
        pose = _normalize_pose_payload(raw_pose)
        if strict_pose_only and raw_pose is not None and pose is None:
            return {"ok": False, "error": "pose_invalid"}
        if strict_pose_only and pose is None:
            return {"ok": False, "error": "pose_required"}

        E: Optional[list[float]] = None
        Twc_payload: Optional[list[float]] = None
        try:
            if pose is not None:
                E_pose = pose_to_E_col_major(pose, align_data=calibration_provider.alignment_data())
                if not (isinstance(E_pose, list) and len(E_pose) == 16):
                    return {"ok": False, "error": "pose_to_extrinsics_failed"}
                E = [float(x) for x in E_pose]
            elif isinstance(req.get("E"), list) and len(req["E"]) == 16:
                E = [float(x) for x in req["E"]]
            elif isinstance(req.get("Twc"), list) and len(req["Twc"]) == 16:
                Twc_payload = [float(x) for x in req["Twc"]]
                Twc = np.array(Twc_payload, dtype=np.float64).reshape((4, 4), order="F")
                Emat = np.linalg.inv(Twc)
                E = list(Emat.flatten(order="F"))
            else:
                return {"ok": False, "error": "E_or_Twc_required"}
        except Exception as exc:
            return {"ok": False, "error": f"parse_error: {exc}"}

        try:
            sid = _camera_to_source_id(cam_id)
            sid_text = "?" if sid is None else str(sid)
            logger.warning(
                "WS RX set_extrinsics camera=%s sid=%s strict_pose_only=%s pose_present=%s legacy_E_present=%s legacy_Twc_present=%s E_col_major=%s",
                cam_id,
                sid_text,
                strict_pose_only,
                bool(pose),
                isinstance(req.get("E"), list),
                isinstance(req.get("Twc"), list),
                E,
            )
            try:
                with np.printoptions(precision=6, suppress=True, linewidth=200):
                    Emat = np.array(E, dtype=np.float64).reshape((4, 4), order="F")
                    logger.warning("WS RX set_extrinsics camera=%s E_matrix=%s", cam_id, str(Emat))
            except Exception:
                pass
        except Exception:
            pass

        try:
            E_matrix = np.array(E, dtype=np.float64).reshape((4, 4), order="F")
            if np.allclose(E_matrix, np.eye(4), atol=1e-3):
                return {"ok": False, "error": "calibration_invalid_identity"}
        except Exception:
            return {"ok": False, "error": "bad_extrinsics"}

        persist_result = calibration_provider.set_extrinsics(cam_id, E=E, Twc=Twc_payload, pose=pose)
        if not persist_result.get("ok"):
            return persist_result

        logger.warning("WS set_extrinsics persisted camera=%s path=%s", cam_id, calibration_provider.extrinsics_path())
        _broadcast_calibration_bundle()
        return {"ok": True, "cameraId": cam_id}

    def _set_align_handler(req: Dict[str, Any]) -> Dict[str, Any]:
        align_update = req.get("align")
        if not isinstance(align_update, dict):
            return {"ok": False, "error": "align_required"}
        matrix = align_update.get("matrix")
        if matrix is not None and (not isinstance(matrix, list) or len(matrix) != 16):
            return {"ok": False, "error": "invalid_matrix"}
        floor_y = align_update.get("floor_y")
        if floor_y is not None and not isinstance(floor_y, (int, float)):
            return {"ok": False, "error": "invalid_floor_y"}
        units = align_update.get("units") if isinstance(align_update.get("units"), dict) else None
        if isinstance(units, dict) and "s_obj_to_m" in units:
            try:
                sval = float(units.get("s_obj_to_m"))
                if not (sval > 0):
                    return {"ok": False, "error": "invalid_s_obj_to_m"}
            except Exception:
                return {"ok": False, "error": "invalid_s_obj_to_m"}
        try:
            floor_y_log = align_update.get("floor_y")
            s_obj_to_m_log = (units or {}).get("s_obj_to_m") if isinstance(units, dict) else None
            matrix_log = align_update.get("matrix")
            matrix_len = len(matrix_log) if isinstance(matrix_log, list) else None
            logger.warning(
                "WS RX set_align floor_y=%s s_obj_to_m=%s matrix_len=%s",
                floor_y_log,
                s_obj_to_m_log,
                matrix_len,
            )
        except Exception:
            pass

        persist_result = calibration_provider.set_align(align_update)
        if not persist_result.get("ok"):
            return persist_result

        logger.warning("WS set_align persisted path=%s", calibration_provider.alignment_path())
        _broadcast_calibration_bundle()
        return {"ok": True}

    def _pixel_to_world_handler(req: Dict[str, Any]) -> Dict[str, Any]:
        cam_id = _resolve_ws_camera_id(req.get("cameraId") or req.get("camera") or req.get("camId") or req.get("id"))
        if not cam_id:
            return {"ok": False, "error": "cameraId_required"}
        src_id = _camera_to_source_id(cam_id)
        if src_id is None:
            return {"ok": False, "error": "unknown_camera"}

        try:
            u = float(req.get("u"))
            v = float(req.get("v"))
        except Exception:
            return {"ok": False, "error": "uv_required"}

        depth_m: Optional[float] = None
        try:
            raw_depth = req.get("depth_m", req.get("depth"))
            if raw_depth is not None:
                depth_m = float(raw_depth)
        except Exception:
            depth_m = None

        snap = calibration_provider.snapshot(int(src_id), str(camera_labels.get(int(src_id), cam_id)))
        if snap is None:
            return {"ok": False, "error": "calibration_missing"}

        K = np.array(snap.intrinsics, dtype=np.float64)
        E_col_major = list(snap.extrinsics_col_major)
        floor_y = float(snap.floor_y or 0.0)
        unit_scale = float(snap.unit_scale or 1.0)
        if not (np.isfinite(unit_scale) and unit_scale > 0):
            unit_scale = 1.0

        try:
            Emat = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
            Twc = np.linalg.inv(Emat)
            R_wc = Twc[:3, :3].copy()
            C_world = (Twc[:3, 3].copy()) * unit_scale
            floor_y_m = float(floor_y) * unit_scale
        except Exception:
            return {"ok": False, "error": "bad_extrinsics"}

        method = "floor"
        world_point: Optional[list[float]] = None

        if depth_m is not None and np.isfinite(depth_m) and depth_m > 0:
            try:
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                x_cam = (u - cx) / fx * depth_m * unit_scale
                y_cam = (v - cy) / fy * depth_m * unit_scale
                z_cam = depth_m * unit_scale
                p_world = (R_wc @ np.array([x_cam, y_cam, z_cam], dtype=np.float64)) + C_world
                world_point = [float(p_world[0]), float(p_world[1]), float(p_world[2])]
                method = "depth"
            except Exception:
                world_point = None
                method = "floor"

        if world_point is None:
            try:
                uv1 = np.array([u, v, 1.0], dtype=np.float64)
                dir_cam = np.linalg.inv(K) @ uv1
                norm = float(np.linalg.norm(dir_cam))
                if norm <= 1e-9:
                    return {"ok": False, "error": "invalid_ray"}
                dir_cam = dir_cam / norm
                dir_world = R_wc @ dir_cam
                denom = float(dir_world[1])
                if abs(denom) < 1e-9:
                    return {"ok": False, "error": "no_intersection"}
                t = (floor_y_m - float(C_world[1])) / denom
                if t < 0:
                    return {"ok": False, "error": "no_intersection"}
                hit = C_world + float(t) * dir_world
                world_point = [float(hit[0]), float(hit[1]), float(hit[2])]
                method = "floor"
            except Exception:
                return {"ok": False, "error": "no_intersection"}

        return {
            "ok": True,
            "world": list(world_point),
            "world_point": list(world_point),
            "method": method,
        }

    ws_server.set_extrinsics_handler = _set_extrinsics_handler
    ws_server.set_align_handler = _set_align_handler
    ws_server.pixel_to_world_handler = _pixel_to_world_handler

    def _tracking_contract_metadata(source_id: int, tracks: list[Mapping[str, Any]]) -> Dict[str, Any]:
        camera_id = str(camera_labels.get(int(source_id), f"camera_{int(source_id)}"))
        calibration_version = "unknown"
        coord_space = POSE_V1_FRAME_BACKEND_WORLD_M
        units = "meters"
        image_size = None
        try:
            bundle = calibration_provider.calibration_bundle()
            meta = bundle.get("meta") if isinstance(bundle, dict) else None
            if isinstance(meta, Mapping):
                raw_version = meta.get("calibration_version")
                if isinstance(raw_version, str) and raw_version.strip():
                    calibration_version = raw_version.strip()
                raw_coord = meta.get("coord_space")
                if isinstance(raw_coord, str) and raw_coord.strip():
                    coord_space = raw_coord.strip()
                raw_units = meta.get("units")
                if isinstance(raw_units, str) and raw_units.strip():
                    units = raw_units.strip()
        except Exception:
            pass
        for track in tracks or []:
            if not isinstance(track, Mapping):
                continue
            size = track.get("image_size") or track.get("frame_size")
            if not isinstance(size, (list, tuple)) or len(size) < 2:
                continue
            try:
                width = int(size[0])
                height = int(size[1])
            except Exception:
                continue
            if width > 8 and height > 8:
                image_size = [width, height]
                break
        if image_size is None:
            try:
                snapshot = calibration_provider.snapshot(int(source_id), camera_id)
                size = getattr(snapshot, "image_size", None)
                if isinstance(size, (list, tuple)) and len(size) >= 2:
                    width = int(size[0])
                    height = int(size[1])
                    if width > 8 and height > 8:
                        image_size = [width, height]
            except Exception:
                pass
        payload = {
            "camera_id": camera_id,
            "coord_space": coord_space,
            "units": units,
            "world_source": "backend_world_fused",
            "track_id_strategy": "camera_tracker_fallback",
            "calibration_version": calibration_version,
            "tracking_contract_version": 3,
            "world_frame": POSE_V1_FRAME_BACKEND_WORLD_M,
        }
        if image_size is not None:
            payload["image_size"] = list(image_size)
            payload["frame_size"] = list(image_size)
        return payload

    setattr(pipeline, "ws_server", ws_server)
    bev_renderer = BevRenderer(
        ws_server,
        trails_cfg=trails_cfg,
        smoothing_cfg=bev_smoothing_cfg,
        frame=str(bev_frame),
        # jpeg_* retired — meta-only mode (see BevRenderer and design decisions)
    )
    ws_server.bev_config_callback = lambda cam_id, cfg: bev_renderer.update_config(cam_id, cfg)
    ws_server.bev_overlay_callback = lambda cam_id, enabled: bev_renderer.update_config(cam_id, {"overlay": enabled})
    depth_pub = DepthTelemetryPublisher(ws_server)
    tracking_pub = TrackingTelemetryPublisher(ws_server, metadata_getter=_tracking_contract_metadata)
    diagnostics_logger = TrackingDiagnosticsLogger.from_env()
    if diagnostics_logger:
        logger.info("V3DT diagnostics logging enabled: %s", diagnostics_logger.output_path)

    # Attach MapAnything postprocess only if SGIE is present/enabled
    try:
        env_ma_post = os.environ.get("NOESIS_MAPANYTHING_POSTPROCESS_ENABLED", "1")
        ma_post_enabled = str(env_ma_post).strip().lower() in ("1", "true", "yes", "on")
        ma_cfg = (pipeline.config.get("models") or {}).get("mapanything") or {}
        ma_enabled = bool(ma_cfg.get("enable", True)) and any(
            key in ma_cfg for key in ("config-file-path", "engine", "name")
        )
        if not ma_post_enabled:
            logger.info("MapAnything postprocess disabled (NOESIS_MAPANYTHING_POSTPROCESS_ENABLED=%s)", env_ma_post)
        elif ma_enabled and "mapanything_fullframe" in pipeline.components:
            hooks.attach_mapanything_postprocess_hook(
                pipeline,
                storage=storage_manager,
                depth_pub=depth_pub,
                camera_labels=camera_labels,
            )
        else:
            logger.info("SGIE disabled or missing; skipping MapAnything postprocess hook")
    except Exception:
        logger.exception("Error while evaluating MapAnything postprocess attachment")
    try:
        hooks.attach_pose_feature_hook(pipeline, camera_labels=camera_labels)
    except Exception:
        logger.exception("Error while attaching pose feature hook")
    if tracking_mode == "baseline" and not depthless_reid_smoke:
        try:
            hooks.attach_object_depth_fusion_hook(
                pipeline,
                camera_labels=camera_labels,
                calibration_resolver=calibration_provider,
                depth_every_n_frames=2,
            )
        except Exception:
            logger.exception("Baseline tracking requires the DAv2 object-depth fusion hook")
            return 1
    elif depthless_reid_smoke:
        logger.warning("Skipping DAv2 object-depth fusion hook for explicit ReID smoke validation config")
    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=tracking_pub,
        tracking_mode=tracking_mode,
        camera_labels=camera_labels,
        bev_renderer=bev_renderer,
        bev_calibration=calibration_provider,
        depth_registration=depth_registration_manager,
        diagnostics_logger=diagnostics_logger,
    )
    hooks.attach_exclude_prune_hook(pipeline)
    hooks.attach_analytics_reload_bridge(pipeline)

    # Wire pyservicemaker Pipeline messages into our logger + shutdown handling.
    def _psm_message_cb(msg_obj) -> None:
        try:
            _on_pyservicemaker_message(
                getattr(pipeline, "ds_pipeline", None),
                msg_obj,
                logger,
                shutdown_event,
                runtime_state,
            )
        except Exception:
            logger.exception("Error in pyservicemaker message callback")

    logger.info("Preparing DS9 pipeline")
    if not ds8_pipeline.prepare(on_message=_psm_message_cb):
        logger.error("DS9 pipeline preparation failed: %s", pipeline.errors)
        return 1

    ws_thread, ws_loop = _start_websocket_server(ws_server)
    if getattr(ws_server, "server", None) is None:
        ws_bind_retry_raw = os.environ.get("NOESIS_WS_BIND_RETRY_TRIES", "4")
        try:
            ws_bind_retry_tries = max(0, int(str(ws_bind_retry_raw).strip() or "4"))
        except Exception:
            ws_bind_retry_tries = 4
        for retry_idx in range(ws_bind_retry_tries):
            next_port = _select_ws_port(args.ws_host, int(args.ws_port) + 1, 32, logger)
            if next_port == int(args.ws_port):
                break
            if not _port_bindable(args.ws_host, int(next_port)):
                logger.error(
                    "No bindable WebSocket fallback port available starting at %s",
                    int(args.ws_port) + 1,
                )
                break
            args.ws_port = int(next_port)
            os.environ["NOESIS_WS_PORT"] = str(int(args.ws_port))
            ws_server.port = int(args.ws_port)
            logger.warning(
                "Retrying WebSocket server start on fallback port %s (attempt %s/%s)",
                args.ws_port,
                retry_idx + 1,
                ws_bind_retry_tries,
            )
            ws_thread, ws_loop = _start_websocket_server(ws_server)
            if getattr(ws_server, "server", None) is not None:
                break
        if getattr(ws_server, "server", None) is None:
            logger.error("WebSocket server failed to start after retries; aborting DS9 runtime")
            return 1

    # Activate the DS9 pipeline after prepare() using activate() not start()
    # NOTE: We use activate() because prepare() was already called above.
    # Using start() after prepare() causes "Tried to add new watch while one was already there"
    # because start() internally calls prepare() + sets bus watch, conflicting with existing watch.
    if not ds8_pipeline.activate():
        logger.error("DS9 pipeline activation failed: %s", pipeline.errors)
        return 1
    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        logger.error("DS9 pipeline activated but ds_pipeline is missing")
        return 1
    logger.info("DS9 pipeline activated successfully")

    # Start pyservicemaker wait loop to keep pipeline alive and processing events
    # This is critical - without wait(), the pipeline may stop after initial buffers
    wait_thread = _start_pyservicemaker_wait_loop(ds, shutdown_event, logger, runtime_state)

    # Start WebRTC gateway(s) if enabled (requires RTSP output)
    webrtc_gateways = []
    if mosaic_webrtc_enabled:
        if rtsp_built:
            try:
                ready = _wait_for_rtsp_ready("127.0.0.1", rtsp_port, timeout=15.0, interval=0.2)
                if not ready:
                    logger.error("RTSP sink not ready on 127.0.0.1:%s; skipping WebRTC gateway start", rtsp_port)
                    #region agent log
                    try:
                        with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H2",
                                        "location": "ds9_runtime_core.py:main",
                                        "message": "gateway skipped - rtsp not ready",
                                        "data": {"host": "127.0.0.1", "port": rtsp_port},
                                        "timestamp": int(time.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    #endregion
                else:
                    from noesis.mosaic_webrtc_gateway import MosaicWebRTCGateway

                    rtsp_uri = f"rtsp://127.0.0.1:{rtsp_port}/{rtsp_path}"
                    try:
                        max_webrtc_clients = max(1, int(os.environ.get("NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS", "5")))
                    except Exception:
                        max_webrtc_clients = 5
                    rtsp_keyframe_requester = _build_rtsp_keyframe_requester(pipeline, logger)
                    if rtsp_keyframe_requester is None:
                        logger.debug("RTSP keyframe requester unavailable; falling back to natural IDR cadence")
                    for slot in range(max_webrtc_clients):
                        try:
                            gateway = MosaicWebRTCGateway(
                                ws_server=ws_server,
                                rtsp_uri=rtsp_uri,
                                request_rtsp_keyframe=rtsp_keyframe_requester,
                            )
                            gateway.start()
                            webrtc_gateways.append(gateway)
                            logger.info(
                                "WebRTC gateway slot %d/%d started, consuming RTSP at %s",
                                slot + 1,
                                max_webrtc_clients,
                                rtsp_uri,
                            )
                        except Exception:
                            logger.exception("Failed to start WebRTC gateway slot %d", slot + 1)
                    logger.info(
                        "WebRTC gateway capacity: %d active slot(s)",
                        len(webrtc_gateways),
                    )
                    #region agent log
                    try:
                        with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H4",
                                        "location": "ds9_runtime_core.py:main",
                                        "message": "gateway started",
                                        "data": {"rtsp_uri": rtsp_uri, "slots": len(webrtc_gateways)},
                                        "timestamp": int(time.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    #endregion
            except Exception:
                logger.exception("Failed to start WebRTC gateway")
        else:
            logger.error("WebRTC gateway enabled but RTSP branch was not built; cannot start gateway")

    depth_branch_present = bool(pipeline.depth_gate_attach and pipeline.depth_gate_attach in pipeline.components)
    if args.depth_enable_seconds > 0:
        if depth_branch_present:
            try:
                ds8_pipeline.enable_depth(seconds=args.depth_enable_seconds)
                logger.info(
                    "Depth branch enabled for %s seconds (startup burst)",
                    args.depth_enable_seconds,
                )
            except Exception:
                logger.exception("Failed to enable depth burst on startup")
        else:
            logger.info(
                "Depth startup enable skipped (no MapAnything branch present)"
            )
    if not getattr(pipeline, "activated", False):
        logger.warning("DS9 pipeline not activated; check pipeline.errors for details: %s", pipeline.errors)

    rest_server = None
    rest_thread = None

    if args.enable_rest:
        try:
            from noesis.server import analytics_api

            analytics_cfg_path = REPO_ROOT / "config" / "nvdsanalytics.yaml"
            os.environ.setdefault(analytics_api.ANALYTICS_CONFIG_ENV, str(analytics_cfg_path))
        except Exception:
            pass
        rest_app = prebuilt_rest_app if prebuilt_rest_app is not None else _build_rest_app()
        rest_server, rest_thread = _start_rest_server(rest_app, args.rest_host, args.rest_port)
        if rest_server:
            logger.info("REST server listening on http://%s:%s", args.rest_host, args.rest_port)

    # Re-assert SIGTERM default behavior after DS/GStreamer initialization.
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except Exception:
        pass

    logger.info("DS9 runtime is active. Press Ctrl+C to stop.")

    # Spawn a heartbeat logger to confirm liveness while waiting/processing
    def _heartbeat() -> None:
        for i in range(6):
            time.sleep(3)
            try:
                with _agent_debug_log_path().open("a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds9_runtime_core.py:main",
                                "message": "heartbeat",
                                "data": {"tick": i + 1},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass

    threading.Thread(target=_heartbeat, name="DS9-Heartbeat", daemon=True).start()
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()

    logger.info("Shutting down DS9 runtime")
    try:
        pipeline.mark_depth_enabled(False)
    except Exception:
        pass

    try:
        storage_manager.flush(timeout=5.0)
    except Exception:
        pass
    try:
        storage_manager.shutdown(wait=True)
    except Exception:
        pass
    try:
        if diagnostics_logger is not None:
            diagnostics_logger.close()
    except Exception:
        logger.exception("Error closing diagnostics logger")

    _stop_rest_server(rest_server, rest_thread)

    # Stop WebRTC gateways if running
    if webrtc_gateways:
        for idx, gateway in enumerate(webrtc_gateways, start=1):
            try:
                gateway.stop()
                logger.info("WebRTC gateway slot %d stopped", idx)
            except Exception:
                logger.exception("Error stopping WebRTC gateway slot %d", idx)

    # Stop pyservicemaker pipeline and wait thread
    if ds is not None:
        stop_done = threading.Event()

        def _stop_psm() -> None:
            try:
                ds.stop()
                logger.info("pyservicemaker pipeline stopped")
            except Exception:
                logger.exception("Error stopping pyservicemaker pipeline")
            finally:
                stop_done.set()

        threading.Thread(target=_stop_psm, name="DS9-StopPipeline", daemon=True).start()
        if not stop_done.wait(timeout=5.0):
            logger.warning("pyservicemaker pipeline stop timed out; continuing shutdown")

    if wait_thread is not None and wait_thread.is_alive():
        try:
            wait_thread.join(timeout=3.0)
            if wait_thread.is_alive():
                logger.warning("Wait thread did not terminate cleanly")
        except Exception:
            logger.exception("Error joining wait thread")

    _stop_websocket_server(ws_server, ws_thread, ws_loop)

    logger.info("Shutdown complete")
    fatal_errors: List[Any] = []
    try:
        for err in getattr(pipeline, "errors", []) or []:
            if isinstance(err, str) and "Depth gating not configured" in err:
                continue
            fatal_errors.append(err)
    except Exception:
        fatal_errors = getattr(pipeline, "errors", []) or []

    exit_code = 0
    if runtime_state.get("pipeline_failed") or fatal_errors:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
