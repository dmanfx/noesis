#!/usr/bin/env python3
# ruff: noqa: E402 - DS9 import authority must be established before runtime imports.
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import configparser
import ctypes
import copy
import hashlib
import inspect
import logging
import math
import os
import re
import subprocess
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, NamedTuple, NoReturn, Optional, Tuple

import yaml
import numpy as np

if TYPE_CHECKING:
    import uvicorn
    from fastapi import FastAPI

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(DS9_ROOT), str(REPO_ROOT)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DS9_ROOT))

from noesis.runtime_paths import (  # noqa: E402
    configure_ds9_runtime_import_paths,
    require_ds9_native_extension_origins,
    service_maker_system_site,
)
from noesis.native_artifact_provenance import (  # noqa: E402
    attest_ds9_native_artifacts,
)

_DS9_NATIVE_EXTENSION_DIR = configure_ds9_runtime_import_paths(
    ds9_root=DS9_ROOT,
    repo_root=REPO_ROOT,
    system_site=service_maker_system_site(),
)
attest_ds9_native_artifacts(
    ds9_root=DS9_ROOT,
    native_dir=_DS9_NATIVE_EXTENSION_DIR,
)
require_ds9_native_extension_origins(_DS9_NATIVE_EXTENSION_DIR)

from calibration_bundle import pose_to_E_col_major
from geometry.depth_source import (
    DepthStorageManager,
    StorageFailure,
    _floorplan_calibration_fingerprint,
)
from mapanything_config import load_service_config
from noesis.calibration.depth_registration import (
    DepthRegistrationError,
    DepthRegistrationManager,
    HARDENED_CONTRACT_VERSION as _DEPTH_REGISTRATION_HARDENED_CONTRACT_VERSION,
    MODEL_CONTENT_BINDING_CONTRACT as _DEPTH_REGISTRATION_MODEL_CONTENT_BINDING_CONTRACT,
    content_bundle_fingerprint as _depth_registration_content_bundle_fingerprint,
    content_file_fingerprint as _depth_registration_content_file_fingerprint,
    model_profile_fingerprint as _depth_registration_model_profile_fingerprint,
)
from noesis.calibration.world_fusion_policy import (
    WorldFusionPolicy,
    WorldFusionPolicyError,
    load_world_fusion_policy,
)
from noesis.calibration.manager import create_calibration_manager, load_camera_labels
from noesis.calibration.pose_v1 import normalize_pose_v1
from noesis.calibration.pose_v1 import POSE_V1_FRAME_BACKEND_WORLD_M
from noesis_core.scene_prior import ScenePriorError, ScenePriorSet
from noesis_core.scene_fusion import SceneFusionError, SceneFusionSet
from noesis.depth_tracking_materialization import (
    DEFAULT_BATCH_SIZE as _DEPTH_TRACKING_BATCH_SIZE,
    DEFAULT_GIE_ID as _DEPTH_TRACKING_GIE_ID,
    DEFAULT_INPUT_SIZE as _DEPTH_TRACKING_INPUT_SIZE,
    DEFAULT_INTERVAL as _DEPTH_TRACKING_INTERVAL,
    ensure_native_depth_tracking_tensor_extension as _ensure_native_depth_tracking_tensor_extension,
    ensure_native_object_depth_extension as _ensure_native_object_depth_extension,
    materialize_depth_tracking_assets as _materialize_depth_tracking_assets,
)
from noesis.deimv2_wholebody49_assets import (
    WHOLEBODY49_SIZES as _WHOLEBODY49_SIZES,
    materialize_wholebody49_configs as _shared_materialize_wholebody49_configs,
    resolve_wholebody49_assets as _shared_resolve_wholebody49_assets,
    validate_wholebody49_pgie_properties as _validate_wholebody49_pgie_properties,
    validate_wholebody49_preprocess_properties as _validate_wholebody49_preprocess_properties,
)
from noesis.rfdetr_1_8_3_assets import (
    RFDETR_DETECTION_SIZES as _RFDETR_DETECTION_SIZES,
    RFDETR_KEYPOINT_VARIANTS as _RFDETR_KEYPOINT_VARIANTS,
    RFDETR_SEGMENTATION_SIZES as _RFDETR_SEGMENTATION_SIZES,
    materialize_rfdetr_1_8_3_configs as _shared_materialize_rfdetr_configs,
    resolve_rfdetr_1_8_3_assets as _shared_resolve_rfdetr_assets,
    validate_rfdetr_class_attrs as _validate_rfdetr_class_attrs,
    validate_rfdetr_pgie_properties as _validate_rfdetr_pgie_properties,
)
from noesis.reid_swin_profile import (
    load_reid_swin_nvinfer_properties as _load_reid_swin_nvinfer_properties,
    validate_reid_swin_model_config as _validate_reid_swin_model_config,
    validate_reid_swin_nvinfer_properties as _validate_reid_swin_nvinfer_properties,
)
from noesis.capture_event_controller import (
    CanonicalCameraAliases,
    CaptureEventController,
)
from noesis.capture_event_rgb_provider import PipelineRgbFrameProvider
from noesis.capture_event_runtime import (
    CaptureEventRuntimeError,
    CaptureEventRuntimeProviders,
    resolve_capture_event_drain_timeout_s,
)
from noesis.manual_depth_models import (
    MANUAL_DEPTH_MODELS,
    build_manual_depth_overlay,
    normalize_manual_depth_model,
)
from noesis.pipelines import ds8_pipeline, hooks
from noesis.depth_capture_event import DepthStorageCaptureEventAdapter
from noesis.runtime_storage import close_depth_storage, storage_close_evidence
require_ds9_native_extension_origins(_DS9_NATIVE_EXTENSION_DIR)
from noesis.metadata.intrinsics import CameraConfigLoader
from noesis.telemetry.publishers import DepthTelemetryPublisher, TrackingTelemetryPublisher, bind_occupancy_publisher
from noesis_core.runtime_world import create_runtime_capability_monitor, create_runtime_world_service
from noesis_core.runtime_health import require_complete_camera_health
from noesis_core.runtime_publication import (
    RuntimePublicationGate,
    RuntimePublicationQuiescenceError,
)
from noesis_core.active_floorplan import ActiveFloorplanError, ActiveFloorplanRegistry
from noesis_core.capture_event_fusion import CaptureEventFusionError
from noesis_core.appliance import (
    ApplianceConfigurationError,
    optional_runtime_context_binding,
)
from noesis_core.servicemaker_shutdown import (
    OrderlyEosError,
    SyntheticStubEosMessage,
    pipeline_expects_finite_source_eos,
    request_orderly_eos,
    validate_synthetic_stub_eos_message,
)
from noesis_core.startup_lifecycle import (
    RuntimeStartupTransaction,
    StartupMainGuard,
    StartupOwnershipAmbiguous,
    StartupPhase,
    StartupResourceRegistrationError,
)
from noesis_core.runtime_secrets import (
    load_pipeline_config as load_runtime_pipeline_config,
    public_pipeline_config,
)
from noesis_core.inference_runtime_contract import render_nvinfer_engine_only_config
from noesis_core.strict_json import strict_json_loads
from noesis.diagnostics.telemetry_log import TrackingDiagnosticsLogger
from noesis.telemetry.bev import BevRenderer
from noesis.yolo26_seg_materialization import (
    materialize_yolo26_seg_configs as _shared_materialize_yolo26_seg_configs,
    resolve_yolo26_seg_assets as _shared_resolve_yolo26_assets,
)
from noesis.runtime_config import apply_osd_from_pgie_ini
from noesis.v3dt_assets import (
    EXPECTED_OBJECT_MODEL_HEIGHT_M,
    EXPECTED_OBJECT_MODEL_RADIUS_M,
    V3DTAssetBundle,
    V3DTAssetError,
    materialize_v3dt_tracker_config,
    validate_v3dt_assets,
)
from websocket_server import (
    WebSocketServer,
    WebSocketStartupError,
    WebSocketStartupReceipt,
)

SHUTDOWN_WATCHDOG_DEFAULT_S = 75
SHUTDOWN_EXTERNAL_SUPERVISOR_TIMEOUT_S = 90
SHUTDOWN_KNOWN_PHASE_BUDGET_S = (
    5.0  # REST and analytics drain
    + WebSocketServer.STATS_COLLECTOR_DRAIN_TIMEOUT_S
    + 5.0  # blocking WebSocket providers
    + 5.0  # retired WebRTC lifecycle workers
    + WebSocketServer.WEBRTC_GATEWAY_DRAIN_TIMEOUT_S
    + WebSocketServer.RUNTIME_SHUTDOWN_TIMEOUT_S
    + 15.0  # Service Maker EOS/wait thread
    + 5.0  # capture-controller admission/owner drain
    + 5.0  # MapAnything capture/worker drain
    + 5.0  # bounded storage writers/retention flush allowance
)
SHUTDOWN_WATCHDOG_MARGIN_S = (
    SHUTDOWN_WATCHDOG_DEFAULT_S - SHUTDOWN_KNOWN_PHASE_BUDGET_S
)
if SHUTDOWN_WATCHDOG_MARGIN_S <= 0.0:
    raise RuntimeError("shutdown phase budgets exceed the runtime watchdog")

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


_PGIE_PROFILES = ("yolo11_seg", "yolo11", "yolo26_seg", "yolo26", "rfdetr_seg", "rfdetr", "rfdetr_keypoint", "wholebody49")
_SIZED_PGIE_PROFILES = ("yolo26_seg", "yolo26", "rfdetr_seg", "rfdetr", "wholebody49")
_YOLO26_DETECT_SIZES = ("n", "s", "m", "l", "x")
_YOLO26_SEG_SIZES = ("n", "s", "m")
_PGIE_SIZE_CHOICES = tuple(
    dict.fromkeys(
        (
            *_YOLO26_DETECT_SIZES,
            *_RFDETR_SEGMENTATION_SIZES,
            *_WHOLEBODY49_SIZES,
        )
    )
)
_YOLO26_DETECT_SIZE_HELP = "/".join(_YOLO26_DETECT_SIZES)
_YOLO26_SEG_SIZE_HELP = "/".join(_YOLO26_SEG_SIZES)
_RFDETR_DETECTION_SIZE_HELP = "/".join(_RFDETR_DETECTION_SIZES)
_RFDETR_SEGMENTATION_SIZE_HELP = "/".join(_RFDETR_SEGMENTATION_SIZES)
_WHOLEBODY49_SIZE_HELP = "/".join(_WHOLEBODY49_SIZES)
_ENV_TRUE = ("1", "true", "yes", "y", "on")
_TRACKING_MODES = ("baseline", "v3dt", "mv3dt")
_MANUAL_DEPTH_MODELS = MANUAL_DEPTH_MODELS
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


def _manual_depth_overlay(
    base_cfg: Mapping[str, Any], manual_depth_model: str
) -> dict[str, Any]:
    try:
        return build_manual_depth_overlay(base_cfg, manual_depth_model)
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _resolve_pipeline_cfg_path(yaml_path: Path, raw: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        return Path("")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    base_dir = yaml_path.parent.resolve()
    for prefix, root in (
        ("DS9/models/engines/", _engine_dir()),
        ("DS9/models/onnx/", _onnx_dir()),
        ("DS9/models/", _model_dir()),
        ("models/engines/", _engine_dir()),
        ("models/onnx/", _onnx_dir()),
        ("models/", _model_dir()),
    ):
        if value.startswith(prefix):
            return (root / value[len(prefix) :]).resolve()
    if value.startswith("DS9/"):
        return (REPO_ROOT / candidate).resolve()
    if value.startswith(("config/", "pipelines/", "build/")):
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

    scene_priors = effective.get("scene_priors")
    if isinstance(scene_priors, dict) and str(scene_priors.get("path") or "").strip():
        scene_priors["path"] = _resolve_pipeline_path_value(base_yaml_path, scene_priors["path"])

    scene_fusions = effective.get("scene_fusions")
    if isinstance(scene_fusions, dict) and str(scene_fusions.get("path") or "").strip():
        scene_fusions["path"] = _resolve_pipeline_path_value(base_yaml_path, scene_fusions["path"])

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
        pipeline_cfg = load_runtime_pipeline_config(pipeline_path, materialize_secrets=False)
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


def _resolve_yolo_detect_assets(profile: str, size: Optional[str]) -> Dict[str, Any]:
    profile_norm = str(profile or "").strip().lower()
    if profile_norm == "yolo11":
        label = "YOLO11"
        size_norm = ""
        tensor_name = "input"
    elif profile_norm == "yolo26":
        size_norm = str(size or "").strip().lower()
        if size_norm not in _YOLO26_DETECT_SIZES:
            raise SystemExit(f"[FATAL] YOLO26 detection size must be one of {_YOLO26_DETECT_SIZE_HELP} (got: {size})")
        label = f"YOLO26 {size_norm}"
        tensor_name = "images"
    else:
        raise SystemExit(f"[FATAL] Unsupported YOLO detection profile: {profile}")

    if profile_norm == "yolo11":
        onnx_path = (_onnx_dir() / "yolo11m.onnx").resolve()
        engine_path = (_engine_dir() / "yolo11m_b3_fp16.engine").resolve()
    else:
        onnx_path = (_onnx_dir() / f"yolo26{size_norm}.onnx").resolve()
        engine_path = (_engine_dir() / f"yolo26{size_norm}_b3_fp16.engine").resolve()

    size_suffix = f"_{size_norm}" if size_norm else ""
    return {
        "label": label,
        "template": (_pipeline_dir() / "config_infer_primary_yolo11.ini").resolve(),
        "preprocess_template": (_pipeline_dir() / "config_preproc.ini").resolve(),
        "preprocess_output": (_build_dir() / f"config_preproc_{profile_norm}{size_suffix}.ini").resolve(),
        "tensor_name": tensor_name,
        "onnx": onnx_path,
        "engine": engine_path,
        "labels": (_model_dir() / "coco_labels.txt").resolve(),
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
    text = re.sub(r"(?m)^\s*onnx-file\s*=.*(?:\n|$)", "", text)
    text, engine_count = re.subn(
        r"(?m)^model-engine-file=.*$",
        f"model-engine-file={assets['engine']}",
        text,
        count=1,
    )
    labels_path = Path(assets["labels"])
    try:
        labels_ready = labels_path.is_file() and labels_path.stat().st_size > 0
    except OSError:
        labels_ready = False
    if not labels_ready:
        raise SystemExit(f"[FATAL] YOLO detection labels file missing or empty: {labels_path}")
    text, labels_count = re.subn(
        r"(?m)^labelfile-path=.*$",
        f"labelfile-path={labels_path}",
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
    if engine_count != 1 or labels_count != 1:
        raise SystemExit(
            "[FATAL] YOLO detection PGIE template is missing "
            f"model-engine-file/labelfile-path: {template_path}"
        )

    out_path = assets["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("%s detector PGIE config materialized: %s", assets["label"], out_path)
    preproc_path = _materialize_yolo_detect_preproc_ini(assets, logger)
    return {**assets, "pgie_config": out_path, "preprocess_config": preproc_path}


def _resolve_rfdetr_detect_assets(size: str) -> Dict[str, Any]:
    try:
        return dict(_shared_resolve_rfdetr_assets("detection", size))
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _materialize_rfdetr_detect_pgie_ini(size: str, logger: logging.Logger) -> Path:
    materialized = _materialize_rfdetr_configs(
        "detection", size, (0, 1, 2), logger
    )
    return Path(materialized["pgie_config"])


def _resolve_rfdetr_assets(size: str) -> Dict[str, Any]:
    try:
        return dict(_shared_resolve_rfdetr_assets("segmentation", size))
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _materialize_rfdetr_pgie_ini(size: str, logger: logging.Logger) -> Path:
    materialized = _materialize_rfdetr_configs(
        "segmentation", size, (0, 1, 2), logger
    )
    return Path(materialized["pgie_config"])


def _materialize_rfdetr_configs(
    family: str,
    size: str,
    src_ids: Tuple[int, ...],
    logger: logging.Logger,
) -> Dict[str, Any]:
    try:
        return dict(
            _shared_materialize_rfdetr_configs(
                family=family,
                size=size,
                batch_size=3,
                src_ids=src_ids,
                include_model_source=False,
                logger=logger,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _resolve_yolo26_assets(size: str) -> Dict[str, Path]:
    try:
        shared = _shared_resolve_yolo26_assets(size)
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc
    size_norm = str(size).strip().lower()
    # The shared resolver owns supported-size/name semantics. DS9 owns every
    # runtime artifact path and must never inherit root/DS8 binaries from it.
    return {
        **shared,
        "template": (_pipeline_dir() / "config_infer_primary_yolo26_seg.template.ini").resolve(),
        "preproc_template": (_pipeline_dir() / "config_preproc.ini").resolve(),
        "onnx": (_onnx_dir() / f"yolo26{size_norm}-seg_fused.onnx").resolve(),
        "engine": (_engine_dir() / f"yolo26{size_norm}-seg_fused_b3_fp16.engine").resolve(),
        "labels": (_model_dir() / "coco_labels.txt").resolve(),
        "parser": (_pipeline_dir() / "nvdsinfer_yolo26_seg" / "libnvdsinfer_yolo26_seg.so").resolve(),
        "output": (_build_dir() / f"config_infer_primary_yolo26_seg_{size_norm}.ini").resolve(),
        "default_output": (_build_dir() / f"config_infer_primary_yolo26_seg_{size_norm}.ini").resolve(),
    }


def _materialize_yolo26_configs(size: str, src_ids: Tuple[int, ...], logger: logging.Logger) -> Dict[str, Path]:
    assets = _resolve_yolo26_assets(size)
    try:
        materialized = _shared_materialize_yolo26_seg_configs(
            size=size,
            batch_size=3,
            src_ids=src_ids,
            logger=logger,
            onnx_path=assets["onnx"],
            engine_path=assets["engine"],
            pgie_output_path=assets["default_output"],
            preprocess_output_path=(_build_dir() / f"config_preproc_yolo26_seg_{str(size).strip().lower()}_b3.ini"),
            include_model_source=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc
    pgie_path = Path(materialized["pgie_config"])
    text = pgie_path.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^labelfile-path=.*$",
        f"labelfile-path={assets['labels']}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^custom-lib-path=.*$",
        f"custom-lib-path={assets['parser']}",
        text,
        count=1,
    )
    pgie_path.write_text(text, encoding="utf-8")
    return {**materialized, **assets, "pgie_config": pgie_path.resolve()}


def _resolve_wholebody49_assets(size: str) -> Dict[str, Any]:
    try:
        return dict(_shared_resolve_wholebody49_assets(size))
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _materialize_wholebody49_configs(
    size: str, src_ids: Tuple[int, ...], logger: logging.Logger
) -> Dict[str, Any]:
    try:
        return dict(
            _shared_materialize_wholebody49_configs(
                size=size,
                batch_size=3,
                src_ids=src_ids,
                include_model_source=False,
                logger=logger,
            )
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


def _preflight_reid_profile(
    pipeline_cfg: Dict[str, Any], yaml_path: Path, logger: logging.Logger
) -> None:
    models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
    reid_cfg = (models_cfg or {}).get("reid") if isinstance(models_cfg, dict) else None
    if not isinstance(reid_cfg, dict) or not bool(reid_cfg.get("enable", True)):
        return

    try:
        _validate_reid_swin_model_config(reid_cfg)
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc

    config_path = _resolve_pipeline_cfg_path(
        yaml_path, str(reid_cfg.get("config-file-path") or "")
    )
    if not config_path.is_file():
        raise SystemExit(f"[FATAL] DS9 ReID SGIE config missing: {config_path}")
    try:
        props = _load_reid_swin_nvinfer_properties(config_path)
        _validate_reid_swin_nvinfer_properties(props)
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc

    engine_path = _resolve_pipeline_cfg_path(
        yaml_path, str(reid_cfg.get("engine") or "")
    )
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        raise SystemExit(
            "[FATAL] Canonical DS9 TAO Swin-Tiny ReID engine missing.\n"
            f"engine: {engine_path}\n"
            "Build it during an exclusive-GPU window with:\n"
            "  DS9/scripts/run_canonical_engine_maintenance.sh --only reid_swin"
        )
    logger.info(
        "Canonical DS9 TAO Swin-Tiny ReID profile found: engine=%s",
        engine_path,
    )


def _require_nonempty_runtime_engine(engine_path: Path, *, label: str) -> None:
    try:
        ready = engine_path.is_file() and engine_path.stat().st_size > 0
    except OSError:
        ready = False
    if not ready:
        raise SystemExit(
            f"[FATAL] {label} engine missing or empty: {engine_path}\n"
            "Runtime startup cannot export ONNX or build TensorRT engines. "
            "Stage the engine with the explicit offline maintenance command first."
        )


def _preflight_pgie_profile(profile: str, pipeline_cfg: Dict[str, Any], yaml_path: Path, logger: logging.Logger) -> None:
    if profile not in _PGIE_PROFILES:
        return
    models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
    pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
    engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None
    if not str(engine_raw or "").strip():
        raise SystemExit(
            f"[FATAL] {profile} profile requires models.pgie.engine to be set"
        )
    profile_engine_labels = {
        "yolo11": "YOLO11 detection DS9 PGIE",
        "yolo11_seg": "YOLO11-seg DS9 PGIE",
        "yolo26": "YOLO26 detection DS9 PGIE",
        "rfdetr": "RF-DETR detection DS9 PGIE",
        "rfdetr_seg": "RF-DETR DS9 PGIE",
        "rfdetr_keypoint": "RF-DETR keypoint-preview DS9 PGIE",
        "yolo26_seg": "YOLO26 DS9 PGIE",
        "wholebody49": "Wholebody49 DS9 PGIE",
    }
    _require_nonempty_runtime_engine(
        _resolve_pipeline_cfg_path(yaml_path, str(engine_raw)),
        label=profile_engine_labels[profile],
    )

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

        _require_nonempty_runtime_engine(
            engine_path,
            label=f"{label} detection PGIE",
        )
        logger.info("%s detection PGIE engine found: %s", label, engine_path)
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
                "Build it with: make -C DS9/pipelines/nvdsinfer_rfdetr\n"
            )

        gie_uid = str(props.get("gie-unique-id", "") or "").strip()
        if gie_uid and gie_uid != "1":
            raise SystemExit(f"[FATAL] RF-DETR detection PGIE gie-unique-id must remain 1 (got {gie_uid})")

        _require_nonempty_runtime_engine(
            engine_path,
            label="RF-DETR detection PGIE",
        )
        logger.info("RF-DETR detection PGIE engine found: %s", engine_path)
        return

    if profile == "rfdetr_keypoint":
        _load_rfdetr_trt_plugin_library(yaml_path, logger)
        preprocess_cfg = (
            pipeline_cfg.get("preprocess")
            if isinstance(pipeline_cfg, dict)
            else None
        )
        if not isinstance(preprocess_cfg, dict) or preprocess_cfg.get("enable") is not False:
            raise SystemExit(
                "[FATAL] RF-DETR keypoint profile requires preprocess.enable=false "
                "so nvinfer owns direct-square frame preprocessing and emits "
                "frame-owned tensor metadata"
            )

        models_cfg = (
            pipeline_cfg.get("models")
            if isinstance(pipeline_cfg, dict)
            else None
        )
        pgie_cfg = (
            (models_cfg or {}).get("pgie")
            if isinstance(models_cfg, dict)
            else None
        )
        pgie_ini_raw = (
            (pgie_cfg or {}).get("config-file-path")
            if isinstance(pgie_cfg, dict)
            else None
        )
        pgie_ini = _resolve_pipeline_cfg_path(
            yaml_path, str(pgie_ini_raw or "")
        )
        if not pgie_ini.exists():
            raise SystemExit(
                "[FATAL] RF-DETR keypoint profile requires PGIE "
                f"config-file-path at: {pgie_ini}"
            )
        engine_raw = (
            (pgie_cfg or {}).get("engine")
            if isinstance(pgie_cfg, dict)
            else None
        )
        engine_path = _resolve_pipeline_cfg_path(
            yaml_path, str(engine_raw or "")
        )
        if not str(engine_raw or "").strip():
            raise SystemExit(
                "[FATAL] RF-DETR keypoint profile requires "
                "models.pgie.engine to be set"
            )

        parser = configparser.ConfigParser(interpolation=None)
        parser.read(pgie_ini, encoding="utf-8")
        props = (
            parser["property"]
            if parser.has_section("property")
            else {}
        )
        try:
            selected_assets = _shared_resolve_rfdetr_assets(
                "keypoint", _RFDETR_KEYPOINT_VARIANTS[0]
            )
            _validate_rfdetr_pgie_properties(
                props,
                family="keypoint",
                engine_profile=str(selected_assets["engine_profile"]),
                query_count=100,
                batch_size=3,
            )
            class_attrs = (
                parser["class-attrs-all"]
                if parser.has_section("class-attrs-all")
                else {}
            )
            parser_score_threshold = _validate_rfdetr_class_attrs(
                class_attrs,
                family="keypoint",
                query_count=100,
                expected_threshold=0.4,
            )
        except ValueError as exc:
            raise SystemExit(f"[FATAL] {exc} ({pgie_ini})") from exc

        lib_raw = str(props.get("custom-lib-path", "") or "").strip()
        lib_path = _resolve_pipeline_cfg_path(yaml_path, lib_raw)
        if not lib_raw or not lib_path.exists():
            raise SystemExit(
                "[FATAL] RF-DETR keypoint PGIE custom parser library missing.\n"
                f"PGIE INI: {pgie_ini}\n"
                f"custom-lib-path: {lib_raw or '<unset>'}\n"
                f"resolved: {lib_path}\n"
                "Build it with: make -C "
                "DS9/pipelines/nvdsinfer_rfdetr_keypoint\n"
            )
        pose_cfg = (
            (models_cfg or {}).get("pose")
            if isinstance(models_cfg, dict)
            else None
        )
        if not isinstance(pose_cfg, dict) or pose_cfg.get("enable") is not False:
            raise SystemExit(
                "[FATAL] RF-DETR keypoint profile must explicitly disable "
                "the YOLO pose SGIE"
            )
        bridge_cfg = (
            (models_cfg or {}).get("rfdetr_keypoint")
            if isinstance(models_cfg, dict)
            else None
        )
        required_bridge = {
            "enable": True,
            "tensor_source": "rfdetr_pgie_frame",
            "gie_id": 1,
            "attach_component": "world_observation_stage",
            "model_size": [576, 576],
            "score_threshold": parser_score_threshold,
            "kpt_threshold": 0.35,
            "letterbox": False,
            "match_min_iou": 0.7,
            "match_ambiguity_margin": 0.05,
            "pose_cache_max_age_frames": 0,
        }
        if not isinstance(bridge_cfg, dict) or any(
            bridge_cfg.get(key) != value
            for key, value in required_bridge.items()
        ):
            raise SystemExit(
                "[FATAL] RF-DETR keypoint metadata bridge config is missing "
                "or incompatible"
            )
        native = getattr(hooks, "noesis_pose_meta_ext", None)
        required_native_symbols = (
            "extract_rfdetr_keypoint_matches",
            "attach_pose_features",
        )
        missing_native_symbols = [
            symbol
            for symbol in required_native_symbols
            if native is None or not callable(getattr(native, symbol, None))
        ]
        if missing_native_symbols:
            raise SystemExit(
                "[FATAL] RF-DETR keypoint profile requires a DS9 "
                "noesis_pose_meta_ext build with callable symbols: "
                + ", ".join(required_native_symbols)
                + " (missing: "
                + ", ".join(missing_native_symbols)
                + ")"
            )
        _require_nonempty_runtime_engine(
            engine_path,
            label="RF-DETR keypoint-preview PGIE",
        )
        logger.info(
            "RF-DETR keypoint-preview PGIE and strict metadata bridge found: %s",
            engine_path,
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
                "Build it with: make -C DS9/pipelines/nvdsinfer_rfdetr_seg\n"
            )

        gie_uid = str(props.get("gie-unique-id", "") or "").strip()
        if gie_uid and gie_uid != "1":
            raise SystemExit(f"[FATAL] RF-DETR PGIE gie-unique-id must remain 1 (got {gie_uid})")

        _require_nonempty_runtime_engine(engine_path, label="RF-DETR PGIE")
        logger.info("RF-DETR PGIE engine found: %s", engine_path)
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
        _require_nonempty_runtime_engine(engine_path, label="YOLO26 PGIE")

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

    if profile == "wholebody49":
        preprocess_cfg = pipeline_cfg.get("preprocess") if isinstance(pipeline_cfg, dict) else None
        preprocess_path_raw = (preprocess_cfg or {}).get("config-file") if isinstance(preprocess_cfg, dict) else None
        preprocess_path = _resolve_pipeline_cfg_path(yaml_path, str(preprocess_path_raw or ""))
        if not preprocess_path.exists():
            raise SystemExit(f"[FATAL] Wholebody49 profile requires preprocess config-file at: {preprocess_path}")

        preproc_parser = configparser.ConfigParser()
        preproc_parser.read(preprocess_path, encoding="utf-8")
        preproc_props = preproc_parser["property"] if preproc_parser.has_section("property") else {}
        try:
            _validate_wholebody49_preprocess_properties(preproc_props, batch_size=3)
        except ValueError as exc:
            raise SystemExit(f"[FATAL] {exc} ({preprocess_path})") from exc

        models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, dict) else None
        pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
        pgie_ini_raw = (pgie_cfg or {}).get("config-file-path") if isinstance(pgie_cfg, dict) else None
        pgie_ini = _resolve_pipeline_cfg_path(yaml_path, str(pgie_ini_raw or ""))
        if not pgie_ini.exists():
            raise SystemExit(f"[FATAL] Wholebody49 profile requires PGIE config-file-path at: {pgie_ini}")

        engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None
        engine_path = _resolve_pipeline_cfg_path(yaml_path, str(engine_raw or ""))
        if not str(engine_raw or "").strip():
            raise SystemExit("[FATAL] Wholebody49 profile requires models.pgie.engine to be set")
        _require_nonempty_runtime_engine(engine_path, label="Wholebody49 DS9 PGIE")

        parser = configparser.ConfigParser()
        parser.read(pgie_ini, encoding="utf-8")
        props = parser["property"] if parser.has_section("property") else {}

        lib_raw = str(props.get("custom-lib-path", "") or "").strip()
        lib_path = _resolve_pipeline_cfg_path(yaml_path, lib_raw)
        if not lib_raw or not lib_path.exists():
            raise SystemExit(
                "[FATAL] Wholebody49 DS9 parser library missing.\n"
                f"PGIE INI: {pgie_ini}\n"
                f"custom-lib-path: {lib_raw or '<unset>'}\n"
                f"resolved: {lib_path}\n"
                "Build it with: make -C DS9/pipelines/nvdsinfer_deimv2_wholebody49\n"
            )

        labels_raw = str(props.get("labelfile-path", "") or "").strip()
        labels_path = _resolve_pipeline_cfg_path(yaml_path, labels_raw)
        if not labels_raw or not labels_path.exists():
            raise SystemExit(
                "[FATAL] Wholebody49 DS9 label file missing.\n"
                f"PGIE INI: {pgie_ini}\n"
                f"labelfile-path: {labels_raw or '<unset>'}\n"
                f"resolved: {labels_path}\n"
            )

        try:
            mode = _validate_wholebody49_pgie_properties(props, batch_size=3)
        except ValueError as exc:
            raise SystemExit(f"[FATAL] {exc} ({pgie_ini})") from exc
        logger.info("Wholebody49 DS9 PGIE engine found: %s (mode=%s)", engine_path, mode)


def _materialize_effective_pipeline_yaml(
    base_yaml_path: Path,
    profile: str,
    logger: logging.Logger,
    *,
    pgie_size: Optional[str] = None,
    tracking_mode: str = "baseline",
    manual_depth_model: str = "mapanything",
    v3dt_bundle: Optional[V3DTAssetBundle] = None,
) -> Path:
    try:
        base_cfg = load_runtime_pipeline_config(base_yaml_path, materialize_secrets=False)
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
            raise SystemExit(
                "[FATAL] RF-DETR detection profile requires --size "
                f"({_RFDETR_DETECTION_SIZE_HELP})"
            )
        size_norm = str(pgie_size).strip().lower()
        sources_cfg = base_cfg.get("sources") if isinstance(base_cfg, dict) else None
        source_count = len(sources_cfg) if isinstance(sources_cfg, list) else 0
        src_ids = tuple(range(source_count)) or (0, 1, 2)
        assets = _materialize_rfdetr_configs(
            "detection", size_norm, src_ids, logger
        )
        overlay = {
            "preprocess": {"config-file": str(assets["preprocess_config"])},
            "models": {
                "pgie": {
                    "config-file-path": str(assets["pgie_config"]),
                    "engine": str(assets["engine"]),
                }
            },
        }
        logger.info(
            "RF-DETR 1.8.3 detection PGIE size: %s (engine profile=%s)",
            size_norm,
            assets["engine_profile"],
        )
    if profile == "rfdetr_keypoint":
        sources_cfg = (
            base_cfg.get("sources") if isinstance(base_cfg, dict) else None
        )
        source_count = (
            len(sources_cfg) if isinstance(sources_cfg, list) else 0
        )
        src_ids = tuple(range(source_count)) or (0, 1, 2)
        assets = _materialize_rfdetr_configs(
            "keypoint", _RFDETR_KEYPOINT_VARIANTS[0], src_ids, logger
        )
        overlay = {
            "preprocess": {"enable": False},
            "models": {
                "pgie": {
                    "config-file-path": str(assets["pgie_config"]),
                    "engine": str(assets["engine"]),
                    "attach_tensor_meta": True,
                },
                "pose": {"enable": False},
                "rfdetr_keypoint": {
                    "enable": True,
                    "tensor_source": "rfdetr_pgie_frame",
                    "gie_id": 1,
                    "attach_component": "world_observation_stage",
                    "model_size": [576, 576],
                    "score_threshold": 0.4,
                    "kpt_threshold": 0.35,
                    "letterbox": False,
                    "match_min_iou": 0.7,
                    "match_ambiguity_margin": 0.05,
                    "pose_cache_max_age_frames": 0,
                },
            },
        }
        logger.info(
            "RF-DETR 1.8.3 keypoint-preview PGIE selected "
            "(engine profile=%s; strict frame tensor bridge)",
            assets["engine_profile"],
        )
    if profile == "rfdetr_seg":
        if not pgie_size:
            raise SystemExit(
                "[FATAL] RF-DETR segmentation profile requires --size "
                f"({_RFDETR_SEGMENTATION_SIZE_HELP})"
            )
        size_norm = str(pgie_size).strip().lower()
        sources_cfg = base_cfg.get("sources") if isinstance(base_cfg, dict) else None
        source_count = len(sources_cfg) if isinstance(sources_cfg, list) else 0
        src_ids = tuple(range(source_count)) or (0, 1, 2)
        assets = _materialize_rfdetr_configs(
            "segmentation", size_norm, src_ids, logger
        )
        overlay = {
            "preprocess": {"config-file": str(assets["preprocess_config"])},
            "models": {
                "pgie": {
                    "config-file-path": str(assets["pgie_config"]),
                    "engine": str(assets["engine"]),
                }
            },
        }
        logger.info(
            "RF-DETR 1.8.3 segmentation PGIE size: %s (engine profile=%s)",
            size_norm,
            assets["engine_profile"],
        )
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
    if profile == "wholebody49":
        if not pgie_size:
            raise SystemExit(f"[FATAL] Wholebody49 profile requires --size ({_WHOLEBODY49_SIZE_HELP})")
        size_norm = str(pgie_size).strip().lower()
        sources_cfg = base_cfg.get("sources") if isinstance(base_cfg, dict) else None
        source_count = len(sources_cfg) if isinstance(sources_cfg, list) else 0
        src_ids = tuple(range(source_count)) or (0, 1, 2)
        assets = _materialize_wholebody49_configs(size_norm, src_ids, logger)
        overlay = {
            "preprocess": {"config-file": str(assets["preprocess_config"])},
            "models": {
                "pgie": {
                    "config-file-path": str(assets["pgie_config"]),
                    "engine": str(assets["engine"]),
                }
            },
        }
        logger.info("Wholebody49 PGIE size: %s (%s)", size_norm, assets.get("mode"))

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

    manual_depth_model = _normalize_manual_depth_model(manual_depth_model)
    overlay = _deep_merge_dict(
        overlay,
        _manual_depth_overlay(base_cfg, manual_depth_model),
    )

    effective_cfg = _deep_merge_dict(base_cfg, overlay)
    if not isinstance(effective_cfg, dict):
        raise SystemExit("[FATAL] Internal error: effective pipeline config is not a mapping")
    effective_cfg["tracking_mode"] = str(tracking_mode).strip().lower()

    preprocess_cfg = effective_cfg.get("preprocess") if isinstance(effective_cfg, dict) else None
    preprocess_path_raw = (preprocess_cfg or {}).get("config-file") if isinstance(preprocess_cfg, dict) else None
    models_cfg = effective_cfg.get("models") if isinstance(effective_cfg, dict) else None
    pgie_cfg = (models_cfg or {}).get("pgie") if isinstance(models_cfg, dict) else None
    pgie_ini_raw = (pgie_cfg or {}).get("config-file-path") if isinstance(pgie_cfg, dict) else None
    engine_raw = (pgie_cfg or {}).get("engine") if isinstance(pgie_cfg, dict) else None

    logger.info("PGIE profile: %s", profile)
    logger.info("Manual depth model: %s", manual_depth_model)
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
    if str(tracking_mode).strip().lower() == "v3dt":
        if v3dt_bundle is None:
            raise SystemExit("[FATAL] V3DT runtime tracker materialization requires a validated asset bundle")
        tracker_out = materialize_v3dt_tracker_config(
            v3dt_bundle,
            out_dir / "config" / "v3dt" / "nvtracker_v3dt_runtime.yaml",
            output_root=out_dir,
        )
        tracker_cfg = effective_cfg.get("tracker")
        if not isinstance(tracker_cfg, dict):
            raise SystemExit("[FATAL] V3DT effective pipeline is missing tracker configuration")
        tracker_cfg["config-file"] = str(tracker_out)
        logger.info("Materialized absolute-path DS9 V3DT tracker config: %s", tracker_out)
    _preflight_reid_profile(effective_cfg, out_path, logger)
    effective_cfg = apply_osd_from_pgie_ini(effective_cfg, out_path)
    effective_cfg = public_pipeline_config(effective_cfg)
    out_path.write_text(yaml.safe_dump(effective_cfg, sort_keys=False), encoding="utf-8")

    _preflight_pgie_profile(profile, effective_cfg, out_path, logger)
    return out_path


def _maybe_autogen_v3dt_caminfo(
    pipeline_path: Path,
    cameras_path: Path,
    bundle: V3DTAssetBundle,
    logger: logging.Logger,
) -> bool:
    """Optionally regenerate V3DT camInfo files from current calibration.

    Controlled by `NOESIS_V3DT_AUTOGEN_CAMINFO` (default: 0).
    """
    flag = str(os.environ.get("NOESIS_V3DT_AUTOGEN_CAMINFO", "0") or "").strip().lower()
    if flag not in _ENV_TRUE:
        return True

    try:
        pipeline_cfg = load_runtime_pipeline_config(pipeline_path, materialize_secrets=False)
    except Exception as exc:
        logger.error("V3DT autogen camInfo failed: unable to read %s: %s", pipeline_path, exc)
        return False

    tracker_path = _resolve_tracker_config_path(pipeline_cfg, pipeline_path)
    if tracker_path is None or tracker_path.resolve() != bundle.tracker_config.resolve():
        logger.error(
            "V3DT autogen camInfo requires the validated DS9 tracker config: expected=%s actual=%s",
            bundle.tracker_config,
            tracker_path or "<missing>",
        )
        return False
    output_dirs = {path.parent.resolve() for path in bundle.camera_models}
    if len(output_dirs) != 1:
        logger.error("V3DT autogen camInfo requires one bounded camera-model directory: %s", output_dirs)
        return False
    output_dir = next(iter(output_dirs))

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
        "--output-dir",
        str(output_dir),
        "--model-height",
        str(EXPECTED_OBJECT_MODEL_HEIGHT_M),
        "--model-radius",
        str(EXPECTED_OBJECT_MODEL_RADIUS_M),
        "--calibration",
        str(REPO_ROOT / "config" / "camera_calibration.json"),
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

    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help="Path to the DS9 pipeline YAML definition.",
    )
    parser.add_argument(
        "--pgie-profile",
        "--pgie_profile",
        "-pgie-profile",
        "-pgie_profile",
        choices=_PGIE_PROFILES,
        default=None,
        help=(
            "PGIE profile overlay. Canonical defaults: baseline=yolo26/m, "
            "v3dt/mv3dt=yolo26_seg/s. Env: NOESIS_PGIE_PROFILE"
        ),
    )
    parser.add_argument(
        "--size",
        choices=_PGIE_SIZE_CHOICES,
        default=None,
        help=(
            "Model size. YOLO26 detection supports n/s/m/l/x; "
            "YOLO26 segmentation supports n/s/m; RF-DETR 1.8.3 detection "
            f"supports {_RFDETR_DETECTION_SIZE_HELP}; RF-DETR 1.8.3 "
            f"segmentation supports {_RFDETR_SEGMENTATION_SIZE_HELP}; "
            "RF-DETR keypoint uses its sole preview variant without --size; "
            "Wholebody49 supports s/x. Default: m, except Wholebody49 defaults to s."
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
        help=(
            "Tracking mode selection (baseline, v3dt, or deferred mv3dt). "
            "Env: NOESIS_TRACKING_MODE"
        ),
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
        default=os.environ.get("NOESIS_WS_HOST", "127.0.0.1"),
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
        default=os.environ.get("NOESIS_REST_HOST", "127.0.0.1"),
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
    parser.add_argument(
        "--manual-depth-model",
        choices=_MANUAL_DEPTH_MODELS,
        default=None,
        help=(
            "Manual full-frame depth model (mapanything or da3metric-large). "
            "Env: NOESIS_MANUAL_DEPTH_MODEL; default: mapanything."
        ),
    )
    args = parser.parse_args()
    env_profile = str(os.environ.get("NOESIS_PGIE_PROFILE", "") or "").strip()
    cli_profile = str(args.pgie_profile or "").strip()
    args._pgie_profile_explicit = bool(cli_profile or env_profile)
    args.pgie_profile = cli_profile or env_profile or "yolo26"
    return args


def _normalize_tracking_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode == "mv3dt":
        return "mv3dt"
    if mode in ("v3dt", "sv3dt", "3d"):
        return "v3dt"
    if mode in ("2d", "baseline", "standard", "default"):
        return "baseline"
    if not mode or mode == "auto":
        return "baseline"
    raise SystemExit(
        "[FATAL] Unsupported DS9 tracking mode "
        f"{value!r}; expected baseline, v3dt, mv3dt, or auto"
    )


def _resolve_tracking_mode(args: argparse.Namespace) -> str:
    if args.tracking_mode:
        return _normalize_tracking_mode(args.tracking_mode)
    if getattr(args, "v3dt", False):
        return "v3dt"
    env_mode = os.environ.get("NOESIS_TRACKING_MODE", "")
    if str(env_mode).strip():
        return _normalize_tracking_mode(env_mode)
    return "baseline"


def _normalize_manual_depth_model(value: Any) -> str:
    try:
        return normalize_manual_depth_model(value)
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc


def _resolve_manual_depth_model(args: argparse.Namespace) -> str:
    cli_value = str(getattr(args, "manual_depth_model", "") or "").strip()
    env_value = str(os.environ.get("NOESIS_MANUAL_DEPTH_MODEL", "") or "").strip()
    return _normalize_manual_depth_model(cli_value or env_value or "mapanything")


def _resolve_pgie_selection(
    args: argparse.Namespace,
    tracking_mode: str,
) -> tuple[str, Optional[str]]:
    profile = str(args.pgie_profile or "yolo26").strip().lower()
    explicit_profile = bool(getattr(args, "_pgie_profile_explicit", False))
    mode = _normalize_tracking_mode(tracking_mode)
    if mode in {"v3dt", "mv3dt"} and not explicit_profile:
        profile = "yolo26_seg"
    if profile not in _PGIE_PROFILES:
        raise SystemExit(f"[FATAL] Unsupported DS9 PGIE profile: {profile}")

    requested_size = getattr(args, "size", None)
    if requested_size is not None and profile not in _SIZED_PGIE_PROFILES:
        sized = ", ".join(_SIZED_PGIE_PROFILES)
        raise SystemExit(f"[FATAL] --size is only valid with --pgie-profile in: {sized}")
    if profile not in _SIZED_PGIE_PROFILES:
        return profile, None
    default_size = (
        "s"
        if profile == "wholebody49"
        or (mode in {"v3dt", "mv3dt"} and profile == "yolo26_seg")
        else "m"
    )
    return profile, str(requested_size or default_size).strip().lower()


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
    raise RuntimeError(
        f"required WebSocket endpoint is unavailable: host={host} port={port}"
    )


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
    if mode in {"v3dt", "mv3dt"}:
        return (
            DS9_ROOT
            / "config"
            / ("infer_mv3dt.yaml" if mode == "mv3dt" else "infer_v3dt.yaml"),
            DS9_ROOT / "config" / "cameras_v3dt.yaml",
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
        env_pipeline = os.environ.get("NOESIS_DS9_PIPELINE_CONFIG", "")
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
        data = load_runtime_pipeline_config(path, materialize_secrets=False)
    except Exception as exc:
        logger.error("Unable to read pipeline config %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.error("Pipeline config must be a mapping: %s", path)
        return None
    return data


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
    v3dt_root = (DS9_ROOT / "config" / "v3dt").resolve()
    try:
        path.resolve().relative_to(v3dt_root)
        return True
    except Exception:
        return False


def _v3dt_profile_for_tracking_mode(tracking_mode: str) -> str:
    """Translate the runtime selector to the profile name stored in YAML."""

    normalized = _normalize_tracking_mode(tracking_mode)
    if normalized == "v3dt":
        return "sv3dt"
    if normalized == "mv3dt":
        return "mv3dt"
    raise ValueError(
        f"V3DT profile validation requires v3dt or mv3dt, got {tracking_mode!r}"
    )


def _validate_v3dt_tracking_guardrails(
    pipeline_path: Path,
    cameras_path: Path,
    logger: logging.Logger,
    *,
    tracking_mode: str,
) -> Optional[V3DTAssetBundle]:
    try:
        bundle = validate_v3dt_assets(
            pipeline_path,
            cameras_config=cameras_path,
            require_engines=True,
            require_sources=False,
            expected_profile=_v3dt_profile_for_tracking_mode(tracking_mode),
        )
    except (OSError, V3DTAssetError) as exc:
        logger.error("%s", exc)
        return None
    logger.info(
        "DS9-owned V3DT asset graph validated: tracker=%s cameras=%d",
        bundle.tracker_config,
        len(bundle.camera_models),
    )
    return bundle


def _reject_baseline_with_v3dt_tracker(pipeline_path: Path, logger: logging.Logger) -> bool:
    pipeline_cfg = _load_pipeline_config(pipeline_path, logger)
    if pipeline_cfg is None:
        return False
    tracker_path = _resolve_tracker_config_path(pipeline_cfg, pipeline_path)
    if tracker_path is None:
        return True
    if _tracker_under_v3dt_dir(tracker_path):
        logger.error(
            "Tracking mode 'baseline' cannot use V3DT tracker config %s; select --tracking-mode v3dt",
            tracker_path,
        )
        return False
    return True


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
            pipeline_cfg = load_runtime_pipeline_config(pipeline_path, materialize_secrets=False)
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


def _logical_ds9_model_reference(value: Any) -> Any:
    """Map only reviewed DS9 model roots to their stable logical identity."""

    if not isinstance(value, (str, os.PathLike)):
        return value
    raw = os.fspath(value).strip()
    if not raw:
        return raw
    candidate = Path(raw).expanduser()
    logical_root = Path("DS9/models")
    if not candidate.is_absolute():
        try:
            relative = candidate.relative_to(logical_root)
        except ValueError:
            return raw
        return (logical_root / relative).as_posix()

    allowed_roots = [(DS9_ROOT / "models").resolve(strict=False)]
    artifact_root_raw = str(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "") or "").strip()
    if artifact_root_raw:
        artifact_root = Path(artifact_root_raw).expanduser()
        if artifact_root.is_absolute():
            allowed_roots.append((artifact_root / "models").resolve(strict=False))

    resolved = candidate.resolve(strict=False)
    for allowed_root in allowed_roots:
        try:
            relative = resolved.relative_to(allowed_root)
        except ValueError:
            continue
        return (logical_root / relative).as_posix()
    return raw


def _canonicalize_ds9_registration_model_paths(
    model_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = dict(model_cfg)
    for key in ("engine", "model-engine-file", "onnx", "onnx-file"):
        if key in canonical:
            canonical[key] = _logical_ds9_model_reference(canonical[key])
    return canonical


_DEPTH_REGISTRATION_SOURCE_CONTRACTS = DS9_ROOT / "config" / "engine_source_contracts.json"
_DEPTH_REGISTRATION_SOURCE_SELECTORS = {
    "depth_tracking": "depth_anything_v2_tracking",
    "mapanything": "mapanything",
}
_DEPTH_REGISTRATION_REVIEWED_CONFIGS = {
    "depth_tracking": DS9_ROOT
    / "pipelines"
    / "config_infer_secondary_depth_tracking_da2.ini",
    "mapanything": DS9_ROOT
    / "pipelines"
    / "config_infer_secondary_mapanything.ini",
}
_DEPTH_REGISTRATION_REVIEWED_LOGICAL_CONFIGS = {
    "depth_tracking": "DS9/pipelines/config_infer_secondary_depth_tracking_da2.ini",
    "mapanything": "DS9/pipelines/config_infer_secondary_mapanything.ini",
}


def _canonicalize_ds9_registration_model_profile(
    model_key: str,
    model_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = _canonicalize_ds9_registration_model_paths(model_cfg)
    canonical.pop("config-file", None)
    canonical["config-file-path"] = _DEPTH_REGISTRATION_REVIEWED_LOGICAL_CONFIGS[
        model_key
    ]
    return canonical


def _registration_logical_source_path(value: str) -> str:
    raw = str(value or "").strip()
    if raw.startswith("models/"):
        return f"DS9/{raw}"
    return str(_logical_ds9_model_reference(raw))


def _registration_source_contract(selector: str) -> tuple[Mapping[str, Any], str]:
    source_record = _depth_registration_content_file_fingerprint(
        _DEPTH_REGISTRATION_SOURCE_CONTRACTS,
        logical_path="DS9/config/engine_source_contracts.json",
    )
    try:
        raw = _DEPTH_REGISTRATION_SOURCE_CONTRACTS.read_bytes()
    except OSError as exc:
        raise DepthRegistrationError(
            f"unable to read DS9 engine source contracts: {exc}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != source_record["sha256"]:
        raise DepthRegistrationError("DS9 engine source contracts changed while reading")
    try:
        payload = strict_json_loads(raw, label="DS9 engine source contracts")
    except Exception as exc:
        raise DepthRegistrationError(
            f"DS9 engine source contracts are invalid: {exc}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "contracts"}
        or payload.get("schema_version") != 1
    ):
        raise DepthRegistrationError(
            "DS9 engine source contracts root does not match schema version 1"
        )
    contracts = payload.get("contracts")
    contract = contracts.get(selector) if isinstance(contracts, Mapping) else None
    if not isinstance(contract, Mapping):
        raise DepthRegistrationError(
            f"DS9 engine source contract selector is missing: {selector}"
        )
    for key in ("staged", "raw_sha256", "bundle_sha256"):
        if not str(contract.get(key) or "").strip():
            raise DepthRegistrationError(
                f"DS9 engine source contract {selector} is missing {key}"
            )
    staged = Path(str(contract.get("staged") or ""))
    if (
        staged.is_absolute()
        or ".." in staged.parts
        or staged.parts[:2] != ("models", "onnx")
    ):
        raise DepthRegistrationError(
            f"DS9 engine source contract {selector} staged ONNX path is not portable"
        )
    return contract, str(source_record["sha256"])


def _registration_onnx_bundle_binding(
    *,
    source_contract: Mapping[str, Any],
    pipeline_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    staged = str(source_contract.get("staged") or "").strip()
    logical_onnx = _registration_logical_source_path(staged)
    onnx_path = _resolve_pipeline_cfg_path(pipeline_path, staged)
    initial_main = _depth_registration_content_file_fingerprint(
        onnx_path,
        logical_path=logical_onnx,
    )
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception as exc:
        raise DepthRegistrationError(
            f"unable to inspect DS9 depth-registration ONNX bundle {logical_onnx}: {exc}"
        ) from exc

    locations: set[Path] = set()
    external_initializer_count = 0
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        external_initializer_count += 1
        metadata = {item.key: item.value for item in tensor.external_data}
        raw_location = str(metadata.get("location", "") or "").strip()
        relative = Path(raw_location)
        if not raw_location or relative.is_absolute() or ".." in relative.parts:
            raise DepthRegistrationError(
                f"unsafe DS9 depth-registration ONNX external-data location: {raw_location!r}"
            )
        locations.add(relative)
    expected_onnx = source_contract.get("onnx")
    if isinstance(expected_onnx, Mapping) and "external_initializer_count" in expected_onnx:
        if external_initializer_count != int(expected_onnx["external_initializer_count"]):
            raise DepthRegistrationError(
                "DS9 depth-registration ONNX external initializer count disagrees with source authority"
            )

    members: list[tuple[str, Path, str]] = [("main", onnx_path, logical_onnx)]
    logical_parent = Path(logical_onnx).parent
    for relative in sorted(locations, key=lambda value: value.as_posix()):
        members.append(
            (
                f"external::{relative.as_posix()}",
                onnx_path.parent / relative,
                (logical_parent / relative).as_posix(),
            )
        )
    observed = _depth_registration_content_bundle_fingerprint(members)
    files = observed.get("files")
    if not isinstance(files, list) or not files or files[0] != {"label": "main", **initial_main}:
        raise DepthRegistrationError(
            "DS9 depth-registration ONNX changed while its bundle was inspected"
        )
    if str(initial_main["sha256"]) != str(source_contract.get("raw_sha256") or ""):
        raise DepthRegistrationError(
            "DS9 depth-registration ONNX raw digest disagrees with source authority"
        )
    if str(observed.get("bundle_sha256") or "") != str(
        source_contract.get("bundle_sha256") or ""
    ):
        raise DepthRegistrationError(
            "DS9 depth-registration ONNX bundle digest disagrees with source authority"
        )
    return logical_onnx, [dict(record) for record in files]


def _registration_model_content_binding(
    *,
    model_key: str,
    model_cfg: Mapping[str, Any],
    pipeline_path: Path,
) -> dict[str, Any]:
    selector = _DEPTH_REGISTRATION_SOURCE_SELECTORS[model_key]
    source_contract, source_contract_sha256 = _registration_source_contract(selector)
    logical_onnx, onnx_bundle_files = _registration_onnx_bundle_binding(
        source_contract=source_contract,
        pipeline_path=pipeline_path,
    )
    engine_raw = str(model_cfg.get("engine") or model_cfg.get("model-engine-file") or "").strip()
    config_raw = str(model_cfg.get("config-file-path") or model_cfg.get("config-file") or "").strip()
    if not engine_raw or not config_raw:
        raise DepthRegistrationError(
            f"DS9 depth registration {model_key} requires engine and runtime config paths"
        )
    engine_path = _resolve_pipeline_cfg_path(pipeline_path, engine_raw)
    selected_config_path = _resolve_pipeline_cfg_path(pipeline_path, config_raw)
    reviewed_config_path = _DEPTH_REGISTRATION_REVIEWED_CONFIGS[model_key]
    logical_engine = str(_logical_ds9_model_reference(engine_path))
    reviewed_config_record = _depth_registration_content_file_fingerprint(
        reviewed_config_path,
        logical_path=_DEPTH_REGISTRATION_REVIEWED_LOGICAL_CONFIGS[model_key],
    )
    engine_record = _depth_registration_content_file_fingerprint(
        engine_path,
        logical_path=logical_engine,
    )
    try:
        rendered_runtime_config = render_nvinfer_engine_only_config(
            source_config=reviewed_config_path,
            engine_path=engine_path,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:
        raise DepthRegistrationError(
            f"unable to render DS9 depth registration runtime config for {model_key}: {exc}"
        ) from exc
    rendered_bytes = rendered_runtime_config.encode("utf-8")
    execution_runtime_config_record = {
        "logical_path": f"DS9/runtime_inference/{model_key}.ini",
        "size_bytes": int(len(rendered_bytes)),
        "sha256": hashlib.sha256(rendered_bytes).hexdigest(),
    }
    rendered_engine_path = engine_path.expanduser().resolve(strict=False)
    engine_line = f"model-engine-file={rendered_engine_path}\n"
    if rendered_runtime_config.count(engine_line) != 1:
        raise DepthRegistrationError(
            f"DS9 depth registration {model_key} runtime config engine binding is ambiguous"
        )
    if not logical_engine.startswith("DS9/models/engines/"):
        raise DepthRegistrationError(
            f"DS9 depth registration {model_key} engine lacks a portable logical identity"
        )
    portable_runtime_config = rendered_runtime_config.replace(
        engine_line,
        f"model-engine-file={logical_engine}\n",
        1,
    ).encode("utf-8")
    runtime_config_record = {
        "logical_path": f"DS9/runtime_inference/{model_key}.ini",
        "size_bytes": int(len(portable_runtime_config)),
        "sha256": hashlib.sha256(portable_runtime_config).hexdigest(),
    }
    if selected_config_path.resolve(strict=False) != reviewed_config_path.resolve(strict=False):
        selected_record = _depth_registration_content_file_fingerprint(
            selected_config_path,
            logical_path=f"DS9/runtime_inference/{model_key}.ini",
        )
        if selected_record != execution_runtime_config_record:
            raise DepthRegistrationError(
                f"DS9 depth registration {model_key} runtime config bytes disagree with the reviewed derivation"
            )

    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read_string(reviewed_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DepthRegistrationError(
            f"DS9 depth registration reviewed config is invalid: {reviewed_config_path}: {exc}"
        ) from exc
    if "property" not in parser:
        raise DepthRegistrationError(
            f"DS9 depth registration reviewed config lacks [property]: {reviewed_config_path}"
        )
    properties = parser["property"]
    configured_engine = str(properties.get("model-engine-file") or "").strip()
    if not configured_engine:
        raise DepthRegistrationError(
            f"DS9 depth registration reviewed config lacks model-engine-file: {reviewed_config_path}"
        )
    configured_engine_path = _resolve_pipeline_cfg_path(pipeline_path, configured_engine)
    if configured_engine_path.resolve(strict=False) != engine_path.resolve(strict=False):
        raise DepthRegistrationError(
            f"DS9 depth registration {model_key} engine disagrees with its runtime config"
        )

    configured_onnx = str(properties.get("onnx-file") or "").strip()
    if configured_onnx:
        configured_logical = _registration_logical_source_path(configured_onnx)
        if configured_logical != logical_onnx:
            raise DepthRegistrationError(
                f"DS9 depth registration {model_key} ONNX disagrees with source authority"
            )
    return {
        "contract": _DEPTH_REGISTRATION_MODEL_CONTENT_BINDING_CONTRACT,
        "engine": engine_record,
        "reviewed_config": reviewed_config_record,
        "runtime_config": runtime_config_record,
        "onnx_authority": {
            "logical_path": logical_onnx,
            "raw_sha256": str(source_contract.get("raw_sha256") or ""),
            "bundle_sha256": str(source_contract.get("bundle_sha256") or ""),
            "bundle_files": onnx_bundle_files,
            "source_contract_selector": selector,
            "source_contract_sha256": source_contract_sha256,
        },
    }


def _build_depth_registration_profile_fingerprints(
    pipeline_cfg: Mapping[str, Any],
    *,
    pipeline_path: Path,
    bind_artifact_content: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    models_cfg = pipeline_cfg.get("models") if isinstance(pipeline_cfg, Mapping) else {}
    depth_cfg = (models_cfg or {}).get("depth_tracking") if isinstance(models_cfg, Mapping) else {}
    ma_cfg = (models_cfg or {}).get("mapanything") if isinstance(models_cfg, Mapping) else {}
    registration_cfg = (
        pipeline_cfg.get("depth_registration")
        if isinstance(pipeline_cfg, Mapping)
        else {}
    )
    ma_reference_cfg = (
        (registration_cfg or {}).get("mapanything_reference")
        if isinstance(registration_cfg, Mapping)
        else None
    )
    if not isinstance(ma_reference_cfg, Mapping):
        ma_reference_cfg = ma_cfg
    repo_root = REPO_ROOT.resolve()
    depth_profile_cfg = (
        _canonicalize_ds9_registration_model_profile("depth_tracking", depth_cfg)
        if isinstance(depth_cfg, Mapping)
        else {}
    )
    mapanything_profile_cfg = (
        _canonicalize_ds9_registration_model_profile(
            "mapanything", ma_reference_cfg
        )
        if isinstance(ma_reference_cfg, Mapping)
        else {}
    )
    if bind_artifact_content:
        if not isinstance(depth_cfg, Mapping) or not isinstance(
            ma_reference_cfg, Mapping
        ):
            raise DepthRegistrationError(
                "DS9 hardened depth registration requires both active model profiles"
            )
        depth_profile_cfg["content_binding"] = _registration_model_content_binding(
            model_key="depth_tracking",
            model_cfg=depth_cfg,
            pipeline_path=pipeline_path,
        )
        mapanything_profile_cfg["content_binding"] = _registration_model_content_binding(
            model_key="mapanything",
            model_cfg=ma_reference_cfg,
            pipeline_path=pipeline_path,
        )
    depth_profile = _depth_registration_model_profile_fingerprint(
        depth_profile_cfg,
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
        mapanything_profile_cfg,
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
    manager = DepthRegistrationManager.load(
        path,
        required_contract_version=_DEPTH_REGISTRATION_HARDENED_CONTRACT_VERSION,
    )
    depth_profile, mapanything_profile = _build_depth_registration_profile_fingerprints(
        pipeline_cfg,
        pipeline_path=pipeline_path,
        bind_artifact_content=True,
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


def _load_world_measurement_fusion_policy(
    *,
    pipeline_cfg: Mapping[str, Any],
    camera_labels: Mapping[int, str],
    depth_registration: DepthRegistrationManager,
    logger: logging.Logger,
) -> WorldFusionPolicy:
    policy_cfg = pipeline_cfg.get("world_measurement_fusion")
    if not isinstance(policy_cfg, Mapping) or not str(policy_cfg.get("path") or "").strip():
        raise WorldFusionPolicyError(
            "baseline tracking requires world_measurement_fusion.path"
        )
    path = Path(str(policy_cfg["path"]).strip()).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    active = [
        camera_id
        for _source_id, camera_id in _active_baseline_camera_ids(
            pipeline_cfg,
            camera_labels,
        )
    ]
    policy = load_world_fusion_policy(
        path,
        runtime_lane="ds9",
        active_camera_ids=active,
        depth_registration=depth_registration,
    )
    logger.info(
        "Loaded calibrated world fusion policy %s id=%s cameras=%d",
        path,
        policy.policy_id,
        len(policy.cameras),
    )
    return policy


def _load_scene_priors(
    *,
    pipeline_cfg: Mapping[str, Any],
    pipeline_path: Path,
    logger: logging.Logger,
) -> ScenePriorSet | None:
    configured = pipeline_cfg.get("scene_priors")
    if configured is None:
        return None
    if not isinstance(configured, Mapping):
        raise ScenePriorError("scene_priors must be a mapping")
    raw_path = str(configured.get("path") or "").strip()
    if not raw_path:
        raise ScenePriorError("scene_priors.path is required when scene priors are configured")
    path = _resolve_pipeline_cfg_path(pipeline_path, raw_path)
    priors = ScenePriorSet.load(path)
    logger.info(
        "Loaded scene-prior catalog %s site=%s cameras=%s mode=shadow",
        path,
        priors.catalog.site_id,
        ",".join(priors.camera_ids) or "none",
    )
    return priors


def _load_scene_fusions(
    *,
    pipeline_cfg: Mapping[str, Any],
    pipeline_path: Path,
    logger: logging.Logger,
) -> SceneFusionSet | None:
    configured = pipeline_cfg.get("scene_fusions")
    if configured is None:
        return None
    if not isinstance(configured, Mapping):
        raise SceneFusionError("scene_fusions must be a mapping")
    raw_path = str(configured.get("path") or "").strip()
    if not raw_path:
        raise SceneFusionError(
            "scene_fusions.path is required when scene fusions are configured"
        )
    path = _resolve_pipeline_cfg_path(pipeline_path, raw_path)
    fusions = SceneFusionSet.load(path)
    logger.info(
        "Loaded scene-fusion catalog %s site=%s cameras=%s diagnostic_only=true",
        path,
        fusions.site_id,
        ",".join(fusions.camera_ids) or "none",
    )
    return fusions


def _build_storage_manager(
    args: argparse.Namespace,
    *,
    on_failure: Optional[Callable[[StorageFailure], None]] = None,
) -> DepthStorageManager:
    service_cfg = load_service_config()
    return _build_storage_manager_from_settings(
        args,
        storage_cfg=service_cfg.storage,
        min_conf=service_cfg.performance.min_conf,
        on_failure=on_failure,
    )


def _build_storage_manager_from_settings(
    args: argparse.Namespace,
    *,
    storage_cfg: Any,
    min_conf: float,
    on_failure: Optional[Callable[[StorageFailure], None]] = None,
) -> DepthStorageManager:
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
        min_conf=float(min_conf),
        on_failure=on_failure,
        legacy_policy="ignore",
    )


def _build_stable_id_manager(
    logger: logging.Logger,
    *,
    pipeline_config: Optional[Mapping[str, Any]] = None,
    camera_labels: Optional[Mapping[int, str]] = None,
    tracking_mode: str = "baseline",
):
    """Instantiate StableIDManager if enabled and available."""
    flag = os.environ.get("NOESIS_REID_ENABLED", "1")
    if str(flag).strip().lower() not in ("1", "true", "yes", "on"):
        logger.info("Stable ID manager disabled (NOESIS_REID_ENABLED=%s)", flag)
        return None
    try:
        from reid.stable_id_manager import StableIDManager  # type: ignore
        from reid.household_state import (  # type: ignore
            is_household_identity_enabled,
            prepare_household_stable_id_overrides,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error(
            "Stable ID/household policy unavailable for canonical DS9 runtime: %s",
            exc,
        )
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
        household_identity_enabled = is_household_identity_enabled()
        cos_sim_high_env_set = os.environ.get("NOESIS_REID_COS_SIM_HIGH_THRESHOLD") is not None
        try:
            cos_sim_high_threshold = float(
                os.environ.get("NOESIS_REID_COS_SIM_HIGH_THRESHOLD", "0.70") or 0.70
            )
        except Exception:
            cos_sim_high_threshold = 0.70
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
        allow_multi_zone_active = True
        household_overrides: Dict[str, Any] = {}
        if household_identity_enabled:
            household_overrides = prepare_household_stable_id_overrides(
                logger,
                repo_root=REPO_ROOT,
                cos_sim_high_threshold=cos_sim_high_threshold,
                cos_sim_high_env_set=cos_sim_high_env_set,
            )
            allow_multi_zone_active = bool(
                household_overrides.pop("allow_multi_zone_active", False)
            )
            auto_merge_enabled = bool(
                household_overrides.pop("auto_merge_enabled", False)
            )
            if "NOESIS_REID_POSE_ENABLED" not in os.environ and not str(
                pose_flag
            ).strip():
                pose_enabled = True
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
        v3dt_cfg = (
            pipeline_config.get("v3dt")
            if isinstance(pipeline_config, Mapping)
            else None
        )
        v3dt_tracking_active = str(tracking_mode).strip().lower() in {
            "v3dt",
            "sv3dt",
            "mv3dt",
        }
        if (
            v3dt_tracking_active
            and isinstance(v3dt_cfg, Mapping)
            and "household_confirm_embeddings" in v3dt_cfg
        ):
            try:
                v3dt_confirm_embeddings = int(
                    v3dt_cfg["household_confirm_embeddings"]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "v3dt.household_confirm_embeddings must be an integer"
                ) from exc
            if not 1 <= v3dt_confirm_embeddings <= 8:
                raise ValueError(
                    "v3dt.household_confirm_embeddings must be within 1..8"
                )
            extra_kwargs["household_confirm_embeddings"] = (
                v3dt_confirm_embeddings
            )
        if household_identity_enabled:
            extra_kwargs.update(household_overrides)
        if camera_labels is not None:
            extra_kwargs["camera_labels"] = dict(camera_labels)
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
            allow_multi_zone_active=allow_multi_zone_active,
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
            "Stable ID manager initialised (SGIE embeddings; household_mode=%s, allow_multi_zone_active=%s, embed_interval_s=%.3f, new_id_hysteresis_frames=%d, new_id_confirm_frames_at_cap=%d, household_confirm_embeddings=%d, pose_enabled=%s)",
            household_identity_enabled,
            allow_multi_zone_active,
            embed_interval_s,
            new_id_hysteresis_frames,
            new_id_confirm_frames_at_cap,
            int(getattr(mgr, "household_confirm_embeddings", 0)),
            pose_enabled,
        )
        if household_identity_enabled and (
            not bool(getattr(mgr, "household_mode", False))
            or bool(getattr(mgr, "auto_merge_enabled", True))
            or bool(getattr(mgr, "allow_multi_zone_active", True))
        ):
            raise RuntimeError(
                "canonical DS9 household StableID policy admission failed: "
                "household_mode=true, auto_merge_enabled=false, and "
                "allow_multi_zone_active=false are required"
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



def _build_stats_callback(
    pipeline: ds8_pipeline.DS8Pipeline,
    camera_labels: Dict[int, str],
    ws_metrics_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ws_metrics_resetter: Optional[Callable[[], None]] = None,
    runtime_state: Optional[Mapping[str, Any]] = None,
) -> Callable[[], Dict[str, object]]:
    start_time = time.time()
    stats_logger = logging.getLogger(__name__)
    sid_metrics_fetch_warned = False
    state = runtime_state if runtime_state is not None else {}

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
        reload_receipt = getattr(pipeline, "analytics_reload_receipt", None)
        initial_receipt = getattr(pipeline, "analytics_initial_receipt", None)
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
            ws_boundary_metrics = ws_metrics_getter()
            if not isinstance(ws_boundary_metrics, dict):
                raise TypeError("WebSocket boundary metrics getter must return a dict")
        from noesis.server import boundary_metrics as _rest_boundary_metrics

        compact_getter = getattr(
            _rest_boundary_metrics,
            "get_boundary_serialization_metrics_compact",
            _rest_boundary_metrics.get_boundary_serialization_metrics,
        )
        rest_boundary_metrics = compact_getter()
        if not isinstance(rest_boundary_metrics, dict):
            raise TypeError("REST boundary metrics getter must return a dict")
        ws_p50 = ws_boundary_metrics.get("p50_ms")
        ws_p95 = ws_boundary_metrics.get("p95_ms")
        ws_p99 = ws_boundary_metrics.get(
            "max_path_p99_ms", ws_boundary_metrics.get("p99_ms")
        )
        ws_p99_10s = ws_boundary_metrics.get(
            "max_path_p99_10s_ms", ws_boundary_metrics.get("p99_10s_ms")
        )
        ws_p99_60s = ws_boundary_metrics.get(
            "max_path_p99_60s_ms", ws_boundary_metrics.get("p99_60s_ms")
        )
        rest_p50 = rest_boundary_metrics.get("p50_ms")
        rest_p95 = rest_boundary_metrics.get("p95_ms")
        rest_p99 = rest_boundary_metrics.get(
            "max_path_p99_ms", rest_boundary_metrics.get("p99_ms")
        )
        rest_p99_10s = rest_boundary_metrics.get(
            "max_path_p99_10s_ms", rest_boundary_metrics.get("p99_10s_ms")
        )
        rest_p99_60s = rest_boundary_metrics.get(
            "max_path_p99_60s_ms", rest_boundary_metrics.get("p99_60s_ms")
        )

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
        boundary_p99_10s = _max_nullable(ws_p99_10s, rest_p99_10s)
        boundary_p99_60s = _max_nullable(ws_p99_60s, rest_p99_60s)
        boundary_p99 = _max_nullable(boundary_p99_10s, boundary_p99_60s)
        ws_boundary_errors = int(
            ws_boundary_metrics.get("boundary_serialization_errors_total", 0) or 0
        )
        rest_boundary_errors = int(
            rest_boundary_metrics.get("boundary_serialization_errors_total", 0) or 0
        )
        boundary_errors = ws_boundary_errors + rest_boundary_errors
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

        source_progress_monitor = getattr(pipeline, "source_progress_monitor", None)
        source_progress: Dict[str, Any] = {
            "healthy": False,
            "started": False,
            "sources": {},
            "fatal": None,
            "reason": "not_configured",
        }
        if source_progress_monitor is not None:
            source_progress = source_progress_monitor.snapshot()
        source_progress_rows = source_progress.get("sources", {})
        if not isinstance(source_progress_rows, dict):
            raise TypeError("decoded source progress snapshot sources must be a dict")

        # Access analytics processor if attached to the pipeline components
        analytics_comp = pipeline.components.get("analytics")
        analytics_proc = analytics_comp.config.get("_analytics_processor") if analytics_comp else None

        for sensor_id, name in sorted(camera_labels.items()):
            cam_key = str(name)
            progress_row = source_progress_rows.get(str(int(sensor_id)), {})
            if not isinstance(progress_row, dict):
                progress_row = {}

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
                "fps": float(progress_row.get("fps", 0.0) or 0.0),
                "frames_processed": int(progress_row.get("frames", 0) or 0),
                "status": str(
                    progress_row.get(
                        "status", "running" if pipeline.activated else "unknown"
                    )
                ),
                "decoded_progress": progress_row,
                "tracking": tracking,
                **({"latency_ms": latency_payload} if latency_payload is not None else {}),
            }
        bev_renderer = getattr(pipeline, "bev_renderer", None)
        active_floorplans = getattr(pipeline, "active_floorplan_registry", None)
        capture_controller = getattr(pipeline, "capture_event_controller", None)
        scene_priors = getattr(pipeline, "scene_priors", None)
        scene_fusions = getattr(pipeline, "scene_fusions", None)
        bev_health = (
            bev_renderer.health_snapshot()
            if bev_renderer is not None
            else {"healthy": False, "cameras": {}, "reason": "not_configured"}
        )
        bev_health = require_complete_camera_health(
            bev_health,
            camera_labels.values(),
        )
        active_floorplan_health = (
            active_floorplans.health_snapshot()
            if active_floorplans is not None
            else {"healthy": False, "cameras": {}, "reason": "not_configured"}
        )
        capture_event_health = (
            capture_controller.health_snapshot()
            if capture_controller is not None
            else {"healthy": False, "reason": "not_configured"}
        )
        scene_prior_health = (
            scene_priors.health_snapshot()
            if scene_priors is not None
            else {"contract": "noesis.scene_prior.health", "contract_version": 1, "mode": "shadow", "status": "not_configured"}
        )
        scene_fusion_health = (
            scene_fusions.health_snapshot()
            if scene_fusions is not None
            else {
                "contract": "noesis.scene_fusion.health",
                "contract_version": 1,
                "status": "not_configured",
            }
        )
        response_model_started_ns = time.perf_counter_ns()
        stats_payload = {
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
                "lifecycle_evidence": getattr(pipeline, "lifecycle_evidence", None),
                "bev": {
                    "frame": "camera_local_ground_m",
                    "health": bev_health,
                },
                "active_floorplan": active_floorplan_health,
                "capture_event_fusion": capture_event_health,
                "scene_prior": scene_prior_health,
                "scene_fusion": scene_fusion_health,
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
                "boundary_cpu_serialization_p99_10s_ms": boundary_p99_10s,
                "boundary_cpu_serialization_p99_60s_ms": boundary_p99_60s,
                "boundary_cpu_serialization_ws_p99_ms": ws_p99,
                "boundary_cpu_serialization_rest_p99_ms": rest_p99,
                "boundary_serialization_errors_total": boundary_errors,
                "boundary_serialization_ws_errors_total": ws_boundary_errors,
                "boundary_serialization_rest_errors_total": rest_boundary_errors,
                "analytics_reload_count": reload_count,
                "analytics_reload_receipt": reload_receipt,
                "analytics_initial_receipt": initial_receipt,
                "analytics_state_poisoned": state.get("analytics_state_poisoned"),
                "source_progress": source_progress,
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
        return WebSocketServer.timed_payload_since(
            stats_payload,
            response_model_started_ns,
        )

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


def _start_websocket_server(
    server: WebSocketServer,
    timeout_s: float = WebSocketServer.STARTUP_TIMEOUT_S,
) -> tuple[threading.Thread, asyncio.AbstractEventLoop]:
    logger = logging.getLogger(__name__)
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("WebSocket startup timeout must be positive")
    started = threading.Event()
    loop_holder: Dict[str, Any] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        try:
            server.event_loop = loop
            start_task = loop.create_task(
                server.start(),
                name="WebSocketServerStartup",
            )
            loop_holder["start_task"] = start_task
            loop.run_until_complete(
                asyncio.wait_for(start_task, timeout=timeout)
            )
            if getattr(server, "server", None) is None:
                raise RuntimeError(
                    "WebSocket startup completed without a bound listener"
                )
            started.set()
            loop.run_forever()
        except BaseException as exc:
            loop_holder["startup_error"] = exc
            logger.exception("WebSocket server thread terminated unexpectedly")
            server.report_lifecycle_failure(exc)
            started.set()
        finally:
            try:
                loop.run_until_complete(server.stop())
            except Exception as exc:
                logger.critical(
                    "WebSocket shutdown proof failed in event-loop owner",
                    exc_info=True,
                )
                server.report_lifecycle_failure(exc)
                loop.run_forever()
            finally:
                if not getattr(server, "_shutdown_quiesced", False):
                    logger.critical(
                        "WebSocket event-loop owner retained after unproven shutdown"
                    )
                    threading.Event().wait()
                elif not loop.is_closed():
                    loop.stop()
                    loop.close()

    thread = threading.Thread(target=_run, name="DS9-WebSocket", daemon=False)
    thread.start()
    startup_signaled = started.wait(timeout=timeout + 0.5)
    loop = loop_holder.get("loop")
    start_task = loop_holder.get("start_task")
    startup_error = loop_holder.get("startup_error")
    server_bound = getattr(server, "server", None) is not None
    if (
        startup_signaled
        and startup_error is None
        and server_bound
        and isinstance(loop, asyncio.AbstractEventLoop)
        and thread.is_alive()
    ):
        return thread, loop

    if (
        isinstance(loop, asyncio.AbstractEventLoop)
        and loop.is_running()
        and start_task is not None
        and not start_task.done()
    ):
        loop.call_soon_threadsafe(start_task.cancel)
    thread.join(timeout=WebSocketServer.RUNTIME_SHUTDOWN_TIMEOUT_S)
    receipt = WebSocketStartupReceipt(
        startup_signaled=bool(startup_signaled),
        server_bound=bool(getattr(server, "server", None) is not None),
        start_task_done=bool(start_task is not None and start_task.done()),
        thread_stopped=not thread.is_alive(),
        event_loop_closed=bool(
            isinstance(loop, asyncio.AbstractEventLoop) and loop.is_closed()
        ),
    )
    raise WebSocketStartupError(
        "WebSocket listener failed bounded startup",
        receipt,
    ) from startup_error


def _stop_websocket_server(
    server: WebSocketServer,
    thread: Optional[threading.Thread],
    loop: Optional[asyncio.AbstractEventLoop],
    timeout: float = WebSocketServer.RUNTIME_SHUTDOWN_TIMEOUT_S,
) -> None:
    if thread is None:
        return
    if loop is None or not loop.is_running():
        raise RuntimeError("WebSocket event loop was unavailable during shutdown")
    deadline = time.monotonic() + max(0.0, float(timeout))
    try:
        future = asyncio.run_coroutine_threadsafe(server.stop(), loop)
        async_bound = min(
            max(0.0, deadline - time.monotonic()),
            float(WebSocketServer.ASYNC_SHUTDOWN_TIMEOUT_S),
        )
        future.result(timeout=async_bound)
    except Exception as exc:
        raise RuntimeError("WebSocket server did not quiesce during shutdown") from exc
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=max(0.0, deadline - time.monotonic()))
    if thread.is_alive():
        raise RuntimeError("WebSocket server thread remained alive after shutdown")


def _build_rest_app(
    *, ws_host: str = "127.0.0.1", ws_port: int = 6008
) -> "FastAPI":
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from noesis.server import alignment_walk_api, analytics_api, depth_api, health_api, reid_api, reid_v2_api, scene_api, scene_prior_api, semantic_seg_api, virtual_twin_api
    from noesis.server.internal_auth import InternalAuthConfigurationError, configure_internal_rest_app

    app = FastAPI(title="Noesis DS9 Runtime API")
    origins_env = os.environ.get("NOESIS_REST_CORS_ORIGINS", "").strip()
    allow_all = os.environ.get("NOESIS_REST_CORS_ALLOW_ALL", "").strip().lower() in {"1", "true", "yes", "on"}
    origin_regex = os.environ.get("NOESIS_REST_CORS_ORIGIN_REGEX", "").strip()
    origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()] if origins_env else []
    if allow_all:
        raise InternalAuthConfigurationError("NOESIS_REST_CORS_ALLOW_ALL is forbidden; use the same-origin gateway")
    if origin_regex:
        raise InternalAuthConfigurationError("regex CORS is forbidden; declare exact NOESIS_REST_CORS_ORIGINS")
    if origins or origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(depth_api.app.router)
    app.include_router(analytics_api.app.router)
    app.include_router(reid_api.app.router)
    app.include_router(reid_v2_api.router)
    app.include_router(virtual_twin_api.app.router)
    app.include_router(health_api.router)
    app.include_router(scene_api.router)
    app.include_router(scene_prior_api.app.router)
    app.include_router(semantic_seg_api.router)
    auth_config = configure_internal_rest_app(app)
    alignment_walk_api.install_alignment_walk_api(
        app,
        ws_host=ws_host,
        ws_port=ws_port,
        auth_config=auth_config,
    )
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


class RestStartupError(RuntimeError):
    """REST startup failed after thread ownership may have been acquired."""

    def __init__(
        self,
        message: str,
        *,
        server: object,
        thread: threading.Thread,
        cleanup_proven: bool,
    ) -> None:
        self.server = server
        self.thread = thread
        self.cleanup_proven = bool(cleanup_proven)
        super().__init__(str(message))


def _start_rest_server(
    app: "FastAPI",
    host: str,
    port: int,
    *,
    startup_timeout_s: float = 10.0,
    cleanup_timeout_s: float = 2.0,
) -> tuple[Optional["uvicorn.Server"], Optional[threading.Thread]]:
    """Start the FastAPI REST server on its exact required endpoint."""
    from noesis.server.internal_auth import validate_internal_auth_listener

    auth_state = getattr(getattr(app, "state", None), "noesis_internal_auth", None)
    if not isinstance(auth_state, dict) or "mode" not in auth_state:
        raise RuntimeError("canonical REST application is missing internal auth state")
    validate_internal_auth_listener(str(auth_state["mode"]), host)
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("uvicorn is required for the canonical REST surface") from exc
    if not _port_bindable(host, int(port)):
        raise RuntimeError(f"required REST endpoint is unavailable: host={host} port={port}")

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        timeout_graceful_shutdown=None,
    )
    server = uvicorn.Server(config=config)

    def _run() -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())
        server.run()

    thread = threading.Thread(target=_run, name="DS9-REST", daemon=True)
    thread_started = False
    failure: BaseException | None = None
    try:
        thread.start()
        thread_started = True
        deadline = time.monotonic() + max(0.0, float(startup_timeout_s))
        while time.monotonic() < deadline:
            if bool(getattr(server, "started", False)):
                if not thread.is_alive():
                    raise RuntimeError("required REST server exited at readiness")
                return server, thread
            if not thread.is_alive():
                raise RuntimeError("required REST server exited before readiness")
            time.sleep(0.02)
        raise TimeoutError("required REST server readiness timed out")
    except BaseException as exc:
        failure = exc

    try:
        server.should_exit = True
    except BaseException as exc:
        if failure is None:
            failure = exc
    if thread_started:
        try:
            thread.join(timeout=max(0.0, float(cleanup_timeout_s)))
        except BaseException as exc:
            if failure is None:
                failure = exc
    cleanup_proven = bool(not thread_started or not thread.is_alive())
    message = str(failure or "required REST server startup failed")
    if not cleanup_proven:
        message = f"{message}; cleanup was not proven"
    raise RestStartupError(
        message,
        server=server,
        thread=thread,
        cleanup_proven=cleanup_proven,
    ) from failure


class RestShutdownReceipt(NamedTuple):
    """Proof that REST can no longer race callback-owned native resources."""

    rest_pair_consistent: bool
    stop_requested: bool
    server_thread_stopped: bool
    analytics_transaction_lock_retained: bool

    @property
    def quiesced(self) -> bool:
        return bool(
            self.rest_pair_consistent
            and self.stop_requested
            and self.server_thread_stopped
            and self.analytics_transaction_lock_retained
        )


def _stop_rest_server(
    server: Optional["uvicorn.Server"],
    thread: Optional[threading.Thread],
    analytics_transaction_lock: Any,
    timeout: float = 5.0,
) -> RestShutdownReceipt:
    """Stop REST and retain the analytics transaction lock through process exit.

    Uvicorn dispatches synchronous handlers through worker threads, so joining
    only its server thread is not sufficient proof that a native analytics
    reload has completed.  Once the listener thread is gone, retaining the
    analytics lock both proves any prior transaction completed and prevents a
    late worker from entering another transaction during native teardown.
    """

    try:
        bounded_timeout = max(0.0, float(timeout))
    except (TypeError, ValueError):
        bounded_timeout = 0.0
    deadline = time.monotonic() + bounded_timeout
    rest_pair_consistent = (server is None) == (thread is None)
    stop_requested = server is None
    if server is not None:
        try:
            server.should_exit = True
            stop_requested = True
        except Exception:
            stop_requested = False

    server_thread_stopped = bool(server is None and thread is None)
    if thread is not None:
        try:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            server_thread_stopped = not thread.is_alive()
        except Exception:
            server_thread_stopped = False
    if not rest_pair_consistent:
        server_thread_stopped = False

    analytics_lock_retained = False
    if server_thread_stopped:
        try:
            analytics_lock_retained = bool(
                analytics_transaction_lock.acquire(
                    timeout=max(0.0, deadline - time.monotonic())
                )
            )
        except Exception:
            analytics_lock_retained = False

    return RestShutdownReceipt(
        rest_pair_consistent=rest_pair_consistent,
        stop_requested=stop_requested,
        server_thread_stopped=server_thread_stopped,
        analytics_transaction_lock_retained=analytics_lock_retained,
    )


def _build_mosaic_keyframe_requester(
    pipeline: ds8_pipeline.DS8Pipeline,
    logger: logging.Logger,
    *,
    failure_callback: Optional[Callable[[BaseException], None]] = None,
) -> Optional[Callable[[str], None]]:
    """Build the supported Service Maker force-IDR control for mosaic H.264."""
    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        return None
    component = getattr(pipeline, "components", {}).get("mosaic_force_idr")
    if component is None or getattr(component, "element", None) != "noesisforceidr":
        return None
    try:
        trigger = ds["mosaic_force_idr"]
        if not callable(getattr(trigger, "set", None)):
            return None
        getter = getattr(trigger, "get", None)
        if not callable(getter):
            return None
        request_sequence = int(getter("accepted-sequence"))
        getter("last-request-ok")
    except Exception as exc:
        logger.error("Mosaic force-IDR trigger is unavailable: %s", exc)
        return None

    request_lock = threading.Lock()

    def request_keyframe(reason: str) -> None:
        nonlocal request_sequence
        try:
            with request_lock:
                request_sequence += 1
                requested = request_sequence
                trigger.set({"request-sequence": requested})
                accepted = int(getter("accepted-sequence"))
                request_ok = bool(getter("last-request-ok"))
                if not request_ok or accepted != requested:
                    raise RuntimeError(
                        f"force-IDR event rejected: requested={requested} accepted={accepted}"
                    )
        except Exception as exc:
            logger.error("Required mosaic force-IDR request failed (reason=%s): %s", reason, exc)
            if callable(failure_callback):
                failure_callback(exc)
            raise
        logger.info(
            "Requested mosaic IDR through NVIDIA encoder event (reason=%s sequence=%d)",
            reason,
            requested,
        )

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
        # Check if this is a source-related error suggesting stream issues
        if any(k in src_name.lower() for k in ("source", "urisrc", "rtspsrc", "decodebin")):
            logger.error("    → Source/decoder error; check RTSP stream connectivity.")
    elif msg_type == Gst.MessageType.EOS:
        logger.warning("⚠️ EOS received on pipeline (unexpected for live sources)")
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        logger.warning("⚠️ Pipeline warning: %s", warn.message)
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
    if type(message) is SyntheticStubEosMessage:
        try:
            sequence = validate_synthetic_stub_eos_message(ds_pipeline, message)
        except Exception:
            if state is not None:
                state["pipeline_failed"] = True
            raise
        shutdown_requested = shutdown_event.is_set()
        finite_source = bool((state or {}).get("expected_eos")) and not shutdown_requested
        eos_reason = "shutdown_requested" if shutdown_requested else "finite_source" if finite_source else "unexpected"
        if state is not None:
            state["pipeline_eos_seen"] = True
            state["pipeline_eos_reason"] = eos_reason
            state["synthetic_eos_request_sequence"] = sequence
            if not (shutdown_requested or finite_source):
                state["pipeline_failed"] = True
        if shutdown_requested or finite_source:
            logger.info("EOS received on pipeline (reason=%s)", eos_reason)
        else:
            logger.warning("EOS received on live pipeline (reason=unexpected)")
        return
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
        shutdown_requested = shutdown_event.is_set()
        finite_source = bool((state or {}).get("expected_eos")) and not shutdown_requested
        eos_reason = (
            "shutdown_requested"
            if shutdown_requested
            else "finite_source"
            if finite_source
            else "unexpected"
        )
        if state is not None:
            state["pipeline_eos_seen"] = True
            state["pipeline_eos_reason"] = eos_reason
        if shutdown_requested or finite_source:
            logger.info("EOS received on pipeline (reason=%s)", eos_reason)
            return
        logger.warning("EOS received on live pipeline (reason=unexpected)")
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
    """Join Service Maker's native loop without blocking the API/control thread."""
    if ds_pipeline is None:
        logger.warning("No DSPipeline available for wait loop")
        return None

    if not hasattr(ds_pipeline, 'wait'):
        logger.warning("DSPipeline doesn't have wait() method; event handling may be limited")
        return None

    def _wait_loop() -> None:
        try:
            logger.debug("pyservicemaker wait loop started")
            ds_pipeline.wait()
            logger.info("pyservicemaker wait() returned (pipeline stopped)")
        except Exception:
            if state is not None:
                state["wait_failed"] = True
                state["pipeline_failed"] = True
            logger.exception("pyservicemaker wait loop error")
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

    thread = threading.Thread(target=_wait_loop, name="DS9-WaitLoop", daemon=True)
    thread.start()
    logger.info("pyservicemaker wait loop started for pipeline event handling")
    return thread


_SERVICE_MAKER_STDIN_WRITE_FD: Optional[int] = None
_SERVICE_MAKER_STDIN_LOCK = threading.Lock()


def _install_servicemaker_stdin_keepalive(logger: logging.Logger) -> None:
    """Keep non-interactive stdin open while Service Maker owns its key watcher."""

    global _SERVICE_MAKER_STDIN_WRITE_FD

    with _SERVICE_MAKER_STDIN_LOCK:
        if _SERVICE_MAKER_STDIN_WRITE_FD is not None or os.isatty(0):
            return
        read_fd, write_fd = os.pipe()
        try:
            os.dup2(read_fd, 0)
        except Exception:
            os.close(write_fd)
            raise
        finally:
            os.close(read_fd)
        _SERVICE_MAKER_STDIN_WRITE_FD = write_fd
        logger.info("Service Maker non-interactive stdin keepalive installed")


def _close_servicemaker_stdin_keepalive() -> None:
    """Release the idle stdin writer only after Pipeline.wait() has returned."""

    global _SERVICE_MAKER_STDIN_WRITE_FD

    with _SERVICE_MAKER_STDIN_LOCK:
        write_fd = _SERVICE_MAKER_STDIN_WRITE_FD
        _SERVICE_MAKER_STDIN_WRITE_FD = None
    if write_fd is not None:
        os.close(write_fd)


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


def _run_main(startup_main_guard: StartupMainGuard) -> int:
    os.environ.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "1")
    os.environ.setdefault("NOESIS_DEPTH_ENABLE_SECONDS", "0")
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("ds9.runtime")
    tracking_mode = _resolve_tracking_mode(args)
    if tracking_mode == "mv3dt":
        logger.critical(
            "MV3DT activation is deferred until Kitchen geometry and synchronized "
            "occupied Kitchen/Family-Room overlap evidence are ready; Living Room "
            "has no MV3DT peer edge."
        )
        return 78
    try:
        appliance_binding = optional_runtime_context_binding(os.environ)
    except ApplianceConfigurationError as exc:
        logger.critical("Permanent appliance runtime-context rejection: %s", exc)
        return 78
    try:
        args.ws_port = _select_ws_port(args.ws_host, int(args.ws_port), 0, logger)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    if args.enable_rest and not _port_bindable(args.rest_host, int(args.rest_port)):
        logger.error(
            "required REST endpoint is unavailable: host=%s port=%s",
            args.rest_host,
            args.rest_port,
        )
        return 1
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

    # Install SIGINT/SIGTERM handling early (before DS/GStreamer init), because
    # some backends install their own handlers/masks which can make `timeout(1)`
    # leave behind orphaned processes that keep ports bound.
    shutdown_event = threading.Event()
    shutdown_watchdog_armed = False

    def _arm_shutdown_watchdog() -> None:
        nonlocal shutdown_watchdog_armed
        if shutdown_watchdog_armed:
            return
        try:
            grace_s = max(
                SHUTDOWN_WATCHDOG_DEFAULT_S,
                int(
                    os.environ.get(
                        "NOESIS_SHUTDOWN_GRACE_SECONDS",
                        str(SHUTDOWN_WATCHDOG_DEFAULT_S),
                    )
                ),
            )
        except Exception:
            grace_s = SHUTDOWN_WATCHDOG_DEFAULT_S
        signal.signal(signal.SIGALRM, lambda _sig, _frame: os._exit(2))
        signal.alarm(grace_s)
        shutdown_watchdog_armed = True

    def _signal_handler(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; initiating shutdown", signum)
        if signum == signal.SIGTERM:
            _arm_shutdown_watchdog()
        shutdown_event.set()

    # Some DS/GStreamer backends manipulate signal masks; ensure SIGINT/SIGTERM are unblocked.
    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    except Exception:
        pass

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    def _startup_ownership_ambiguous(reason: str) -> None:
        runtime_state["pipeline_failed"] = True
        runtime_state["startup_ownership_ambiguous"] = str(reason)
        logger.critical("Runtime startup ownership is ambiguous: %s", reason)
        shutdown_event.set()
        _arm_shutdown_watchdog()

    startup_transaction = RuntimeStartupTransaction(
        on_ambiguous=_startup_ownership_ambiguous
    )

    def _abort_startup(reason: str, code: int = 1) -> int:
        logger.error("Runtime startup aborted: %s", reason)
        receipt = startup_transaction.abort_reversible()
        runtime_state["startup_abort_receipt"] = {
            "phase_before": receipt.phase_before.value,
            "phase_after": receipt.phase_after.value,
            "completed": receipt.completed,
            "cleanup": [
                {
                    "name": row.name,
                    "completed": row.completed,
                    "error_type": row.error_type,
                    "error_message": row.error_message,
                }
                for row in receipt.cleanup
            ],
        }
        if not receipt.completed:
            runtime_state["pipeline_failed"] = True
            logger.critical(
                "Reversible startup cleanup was incomplete; waiting for watchdog"
            )
            _arm_shutdown_watchdog()
            while True:
                signal.pause()
        return int(code)

    def _record_startup_ambiguity(error: StartupOwnershipAmbiguous) -> None:
        runtime_state["startup_ambiguity_receipt"] = {
            "ingress_quiesced": error.ingress_quiesced,
            "cleanup": [
                {
                    "name": row.name,
                    "completed": row.completed,
                    "error_type": row.error_type,
                    "error_message": row.error_message,
                }
                for row in error.ingress_cleanup
            ],
        }
        if not error.ingress_quiesced:
            logger.critical(
                "Startup ingress cleanup was not proven; watchdog exit remains armed"
            )

    def _preserve_ambiguous_and_wait(
        reason: str,
        error: StartupOwnershipAmbiguous | None = None,
    ) -> NoReturn:
        _arm_shutdown_watchdog()
        if error is not None:
            _record_startup_ambiguity(error)
        logger.critical(
            "Preserving callback-owned startup resources until watchdog exit: %s",
            reason,
        )
        while True:
            signal.pause()

    def _abort_ambiguous_and_wait(reason: str) -> NoReturn:
        _arm_shutdown_watchdog()
        try:
            startup_transaction.abort_ambiguous(reason)
        except StartupOwnershipAmbiguous as exc:
            _preserve_ambiguous_and_wait(reason, exc)
        except BaseException as exc:
            runtime_state["startup_ambiguity_transaction_error"] = type(exc).__name__
            _startup_ownership_ambiguous(reason)
            logger.critical(
                "Startup ambiguity transaction failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            _preserve_ambiguous_and_wait(reason)

    def _handle_unexpected_startup_exception(exc: BaseException) -> int:
        phase = startup_transaction.phase
        logger.critical(
            "Unhandled runtime startup exception during phase=%s",
            phase.value,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        if phase in {StartupPhase.ASSEMBLING, StartupPhase.PREPARED}:
            code = _abort_startup(
                f"unhandled_startup_exception:{type(exc).__name__}"
            )
            if (
                isinstance(exc, StartupResourceRegistrationError)
                and not exc.rollback_completed
            ):
                runtime_state["startup_registration_rollback"] = [
                    {
                        "name": row.name,
                        "completed": row.completed,
                        "error_type": row.error_type,
                        "error_message": row.error_message,
                    }
                    for row in exc.rollback
                ]
                logger.critical(
                    "Newly acquired startup resource rollback was not proven"
                )
                _arm_shutdown_watchdog()
                while True:
                    signal.pause()
            return code
        if phase in {
            StartupPhase.ACTIVATION_ATTEMPTED,
            StartupPhase.ACTIVE_WAIT_OWNED,
        }:
            _abort_ambiguous_and_wait(
                f"unhandled_activation_exception:{type(exc).__name__}"
            )
        runtime_state["pipeline_failed"] = True
        logger.critical(
            "Unexpected startup guard phase=%s; preserving process ownership",
            phase.value,
        )
        _arm_shutdown_watchdog()
        while True:
            signal.pause()

    startup_main_guard.arm(_handle_unexpected_startup_exception)

    if tracking_mode not in _TRACKING_MODES:
        raise SystemExit(
            "[FATAL] Resolved DS9 tracking mode is outside the supported contract: "
            f"{tracking_mode!r}"
        )
    os.environ["NOESIS_TRACKING_MODE"] = tracking_mode
    pgie_profile, pgie_size = _resolve_pgie_selection(args, tracking_mode)
    manual_depth_model = _resolve_manual_depth_model(args)
    args.pgie_profile = pgie_profile
    args.manual_depth_model = manual_depth_model
    os.environ["NOESIS_MANUAL_DEPTH_MODEL"] = manual_depth_model
    if appliance_binding is not None:
        expected_variant = (
            f"ds9:wholebody49-{pgie_size}"
            if pgie_profile == "wholebody49"
            else "ds9:v3dt"
            if tracking_mode == "v3dt"
            else "ds9:baseline"
        )
        if (
            appliance_binding.runtime_family != "ds9"
            or appliance_binding.runtime_variant != expected_variant
            or args.ws_host != "127.0.0.1"
            or int(args.ws_port) != 6008
            or args.rest_host != "127.0.0.1"
            or int(args.rest_port) != 8080
            or not args.enable_rest
        ):
            logger.critical(
                "Permanent appliance runtime selection/endpoints differ from context"
            )
            return 78
    logger.info(
        "DS9 inference selection: tracking_mode=%s pgie_profile=%s size=%s "
        "manual_depth_model=%s explicit_profile=%s",
        tracking_mode,
        pgie_profile,
        pgie_size or "fixed",
        manual_depth_model,
        bool(getattr(args, "_pgie_profile_explicit", False)),
    )

    pipeline_path, cameras_path = _resolve_pipeline_and_camera_paths(args, tracking_mode, logger)
    pipeline_path = pipeline_path.expanduser().resolve()
    cameras_path = cameras_path.expanduser().resolve()

    # Ensure process CWD is the repo root so relative paths in YAML (engines, configs)
    # resolve correctly for DS9 plugins and hooks.
    try:
        os.chdir(REPO_ROOT)
    except Exception:
        logger.exception("Unable to chdir to required REPO_ROOT %s", REPO_ROOT)
        return _abort_startup("repo_root_chdir_failed")

    if not pipeline_path.exists():
        logger.error("Pipeline configuration not found: %s", pipeline_path)
        return 1
    v3dt_bundle: Optional[V3DTAssetBundle] = None
    if tracking_mode == "v3dt":
        if not _ensure_v3dt_meta_extension(logger):
            return 1
        v3dt_bundle = _validate_v3dt_tracking_guardrails(
            pipeline_path,
            cameras_path,
            logger,
            tracking_mode=tracking_mode,
        )
        if v3dt_bundle is None:
            return 1
        if not _maybe_autogen_v3dt_caminfo(
            pipeline_path, cameras_path, v3dt_bundle, logger
        ):
            return 1
        # Autogeneration mutates the camera-model artifacts deliberately; rerun
        # the complete ownership/shape/provenance contract before NvMOT sees them.
        v3dt_bundle = _validate_v3dt_tracking_guardrails(
            pipeline_path,
            cameras_path,
            logger,
            tracking_mode=tracking_mode,
        )
        if v3dt_bundle is None:
            return 1
    else:
        if not _reject_baseline_with_v3dt_tracker(pipeline_path, logger):
            return 1

    skip_cuda_preflight = str(os.environ.get("NOESIS_SKIP_CUDA_PREFLIGHT", "")).strip().lower() in _ENV_TRUE
    stub_pipeline = (
        str(os.environ.get("NOESIS_DS9_STUB_PIPELINE", ""))
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
        pgie_profile,
        logger,
        pgie_size=pgie_size,
        tracking_mode=tracking_mode,
        manual_depth_model=manual_depth_model,
        v3dt_bundle=v3dt_bundle,
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

    if not _validate_dewarper_intrinsics_sync(pipeline_path, cameras_path, logger):
        return 1

    os.environ["NOESIS_DS9_PIPELINE_CONFIG"] = str(pipeline_path)
    os.environ["NOESIS_DS8_PIPELINE_CONFIG"] = str(pipeline_path)
    try:
        from noesis.server import analytics_api

        pipeline_cfg_for_analytics = load_runtime_pipeline_config(
            pipeline_path,
            materialize_secrets=False,
        )
        analytics_pipeline_cfg = pipeline_cfg_for_analytics.get("analytics") or {}
        analytics_default_path = REPO_ROOT / "config" / "nvdsanalytics.yaml"
        if tracking_mode in ("v3dt", "sv3dt", "mv3dt"):
            stages_raw_path = str(
                analytics_pipeline_cfg.get("stages_config") or ""
            ).strip()
            if stages_raw_path:
                analytics_default_path = _resolve_pipeline_cfg_path(
                    pipeline_path, stages_raw_path
                )
        os.environ.setdefault(
            analytics_api.ANALYTICS_CONFIG_ENV,
            str(analytics_default_path),
        )
        exclude_cfg = analytics_pipeline_cfg.get("exclude") or {}
        exclude_raw_path = str(exclude_cfg.get("config-file") or "").strip()
        if not exclude_raw_path:
            raise RuntimeError("Analytics exclusion config path is missing")
        os.environ.setdefault(
            "NOESIS_ANALYTICS_EXCLUDE_CONFIG",
            str(_resolve_pipeline_cfg_path(pipeline_path, exclude_raw_path)),
        )
        analytics_cfg = analytics_api._load_config(force=True)  # type: ignore[attr-defined]
        stages_cfg = (analytics_cfg.get("analytics") or {}).get("stages") or {}
        if "exclude" not in stages_cfg or not isinstance(stages_cfg["exclude"], dict):
            raise RuntimeError("Writable analytics config is missing the required exclude stage")
        startup_exclude_path = analytics_api._sync_exclude_stage(  # type: ignore[attr-defined]
            "exclude", stages_cfg["exclude"]
        )
        if startup_exclude_path is None:
            raise RuntimeError("Writable analytics exclusion state did not materialize")
        analytics_startup_reload_context = analytics_api._build_exclude_reload_context(  # type: ignore[attr-defined]
            startup_exclude_path
        )

        def _analytics_state_poisoned(reason: str) -> None:
            runtime_state["analytics_state_poisoned"] = reason
            runtime_state["pipeline_failed"] = True
            shutdown_event.set()

        startup_transaction.bind(
            "analytics_runtime_hooks",
            lambda: analytics_api.register_poison_hook(_analytics_state_poisoned),
            analytics_api.clear_runtime_hooks,
        )
    except StartupResourceRegistrationError:
        raise
    except Exception:
        logger.exception("Failed to load and synchronize writable analytics state at startup")
        return _abort_startup("analytics_state_initialization_failed")

    prebuilt_rest_app = None
    if args.enable_rest:
        try:
            if not _preload_rest_server_runtime():
                logger.error("Failed to preload REST server runtime")
                return _abort_startup("rest_runtime_preload_failed")
            prebuilt_rest_app = _build_rest_app(
                ws_host=args.ws_host,
                ws_port=int(args.ws_port),
            )
            logger.info("REST app and server runtime prebuilt before DeepStream pipeline startup")
        except Exception:
            logger.exception("Failed to build REST app")
            return _abort_startup("rest_app_prebuild_failed")

    camera_labels = load_camera_labels(cameras_path, strict=True)
    try:
        capture_camera_aliases = CanonicalCameraAliases(camera_labels)
    except ValueError:
        logger.exception("Capture-event camera aliases are invalid")
        return _abort_startup("capture_event_camera_aliases_invalid")
    camera_aliases = {
        alias: camera
        for source_id, camera in camera_labels.items()
        for alias in (str(source_id), str(camera))
    }
    try:
        active_floorplan_registry = ActiveFloorplanRegistry(camera_aliases)
    except ActiveFloorplanError:
        logger.exception("Active floorplan registry configuration is invalid")
        return _abort_startup("active_floorplan_registry_invalid")

    def _depth_storage_failed(failure: StorageFailure) -> None:
        runtime_state["depth_storage_failure"] = {
            "write_id": failure.write_id,
            "sequence": failure.sequence,
            "camera_id": failure.camera_id,
            "ts_us": failure.ts_us,
            "error_type": failure.error_type,
            "message": failure.message,
        }
        runtime_state["pipeline_failed"] = True
        logger.critical(
            "Transactional depth storage failed: write_id=%s sequence=%s type=%s message=%s",
            failure.write_id,
            failure.sequence,
            failure.error_type,
            failure.message,
        )
        shutdown_event.set()

    try:
        storage_manager = startup_transaction.acquire(
            "depth_storage",
            lambda: _build_storage_manager(
                args,
                on_failure=_depth_storage_failed,
            ),
            lambda manager: close_depth_storage(manager, timeout_s=5.0),
            validator=lambda receipt: bool(receipt.completed),
        )
    except StartupResourceRegistrationError:
        raise
    except Exception:
        logger.exception("Failed to initialize transactional depth storage")
        return _abort_startup("depth_storage_initialization_failed")
    runtime_state["depth_storage_startup"] = storage_manager.startup_report()
    if runtime_state["depth_storage_startup"].get("legacy_entries"):
        logger.warning(
            "Ignoring %d legacy depth snapshots without commit manifests",
            len(runtime_state["depth_storage_startup"]["legacy_entries"]),
        )
    auto_calibrate_lock = threading.Lock()

    def _resolve_auto_calibrate_cameras(camera_id: Optional[str]) -> list[str]:
        if camera_id is None or not str(camera_id).strip():
            return [
                capture_camera_aliases.canonicalize(name)
                for _, name in sorted(
                    camera_labels.items(),
                    key=lambda item: int(item[0]),
                )
            ]
        return [capture_camera_aliases.canonicalize(camera_id)]

    def _ds8_auto_calibrate_handler(camera_id: Optional[str] = None) -> Dict[str, Any]:
        if not auto_calibrate_lock.acquire(blocking=False):
            return {"ok": False, "results": [], "updated": [], "error": "busy"}
        try:
            if shutdown_event.is_set():
                return {"ok": False, "results": [], "updated": [], "error": "shutting_down"}
            if not pipeline.activated:
                return {"ok": False, "results": [], "updated": [], "error": "pipeline_not_running"}
            if storage_manager is None:
                return {"ok": False, "results": [], "updated": [], "error": "depth_source_unavailable"}
            depth_branch_present = bool(
                pipeline.depth_gate_attach and pipeline.depth_gate_attach in pipeline.components
            )
            if not depth_branch_present:
                return {"ok": False, "results": [], "updated": [], "error": "depth_branch_unavailable"}

            try:
                cameras = _resolve_auto_calibrate_cameras(camera_id)
            except CaptureEventFusionError as exc:
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": exc.code,
                }
            if not cameras:
                return {"ok": False, "results": [], "updated": [], "error": "no_cameras_configured"}

            raw_enable_seconds = os.environ.get(
                "NOESIS_AUTOCALIB_ENABLE_SECONDS",
                "10",
            )
            try:
                enable_seconds = float(str(raw_enable_seconds).strip())
            except Exception:
                enable_seconds = float("nan")
            if (
                not math.isfinite(enable_seconds)
                or not 0.01 <= enable_seconds <= 15.0
            ):
                error = CaptureEventRuntimeError(
                    "capture_event_configuration_invalid"
                )
                _runtime_component_failed("capture_event", error)
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": error.code,
                }

            providers = getattr(
                pipeline,
                "capture_event_runtime_providers",
                None,
            )
            if not isinstance(providers, CaptureEventRuntimeProviders):
                error = CaptureEventRuntimeError(
                    "capture_event_controller_unavailable"
                )
                _runtime_component_failed("capture_event", error)
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": error.code,
                }
            try:
                capture_outcomes = providers.capture_for_auto_calibration(
                    cameras,
                    burst_seconds=enable_seconds,
                )
            except (CaptureEventFusionError, CaptureEventRuntimeError) as exc:
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": exc.code,
                }
            auto_capture_evidence = {
                outcome.canonical_camera: outcome.compact_evidence_payload()
                for outcome in capture_outcomes
            }

            if shutdown_event.is_set():
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": "shutting_down",
                }

            try:
                from scripts.auto_calibrate_from_depth import auto_calibrate_from_latest_depth
            except Exception:
                logger.exception("Auto-calibration implementation is unavailable")
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": "auto_calibration_unavailable",
                }

            try:
                res = auto_calibrate_from_latest_depth(cameras, persist=False)
            except Exception:
                logger.exception("Auto-calibration computation failed")
                return {
                    "ok": False,
                    "results": [],
                    "updated": [],
                    "error": "auto_calibration_failed",
                }

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
                    for updated_camera in updated:
                        active_floorplan_registry.clear(updated_camera)
                        storage_manager.invalidate_floorplan_cache(updated_camera)
                except Exception as exc:
                    _runtime_component_failed(
                        "calibration_cache_invalidation",
                        exc,
                    )
                    return {
                        "ok": False,
                        "results": results or [],
                        "updated": updated,
                        "error": "calibration_cache_invalidation_failed",
                        "capture_events": auto_capture_evidence,
                    }
                try:
                    storage_manager.calibration_bundle = calibration_provider.calibration_bundle()
                except Exception:
                    logger.debug("Unable to refresh storage calibration bundle", exc_info=True)
                try:
                    response_model_started_ns = time.perf_counter_ns()
                    bundle = calibration_provider.calibration_bundle()
                    message = {"type": "calibration-bundle", "data": bundle}
                    ws_server.broadcast_sync(
                        message,
                        response_model_timing=ws_server.response_model_timing_since(
                            response_model_started_ns
                        ),
                    )
                except Exception:
                    logger.debug("Failed to broadcast calibration bundle", exc_info=True)

            return {
                "ok": bool(updated),
                "results": results or [],
                "updated": updated,
                "error": top_error,
                "capture_events": auto_capture_evidence,
            }
        finally:
            auto_calibrate_lock.release()

    def _ds8_ma_depth_provider(
        cam_id: str,
        ts_max_us: Optional[object] = None,
        request_id: Optional[str] = None,
        cache_only: bool = False,
        **_ignored: object,
    ) -> Dict[str, Any]:
        providers = getattr(pipeline, "capture_event_runtime_providers", None)
        if not isinstance(providers, CaptureEventRuntimeProviders):
            error = CaptureEventRuntimeError(
                "capture_event_controller_unavailable"
            )
            _runtime_component_failed("capture_event", error)
            response: Dict[str, Any] = {
                "type": "ma_depth_response",
                "camera": str(cam_id or "").strip(),
                "cache_only": bool(cache_only),
                "served_from_cache": False,
                "ts_us": 0,
                "ok": False,
                "error": error.code,
            }
            if request_id:
                response["request_id"] = request_id
            return response
        return providers.depth_provider(
            cam_id,
            ts_max_us=ts_max_us,
            request_id=request_id,
            cache_only=cache_only,
            **_ignored,
        )

    def _record_active_floorplan_payload(
        camera_key: str,
        payload: Mapping[str, Any],
    ) -> bool:
        if payload.get("error"):
            return False
        try:
            return active_floorplan_registry.record(
                camera_key,
                payload,
                snapshot_ref=str(payload.get("snapshot_ref") or ""),
            )
        except ActiveFloorplanError as exc:
            _runtime_component_failed("active_floorplan", exc)
            raise

    def _bev_active_floorplan_bounds_provider(
        camera_key: str,
    ) -> Optional[Mapping[str, Any]]:
        return active_floorplan_registry.bounds_for(camera_key)

    def _ds8_floorplan_provider(
        camera: Optional[str] = None,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
        scene_prior_only: bool = False,
        **_ignored: object,
    ) -> Dict[str, Any]:
        requested_camera_id = str(camera or "").strip()
        if scene_prior_only:
            if scene_prior_set is None:
                return {
                    "camera_id": requested_camera_id,
                    "scene_prior_only": True,
                    "error": "scene_prior_not_configured",
                }
            binding = scene_prior_set.binding(requested_camera_id)
            if binding is None or not binding.include_floorplan_layers:
                return {
                    "camera_id": requested_camera_id,
                    "scene_prior_only": True,
                    "error": "scene_prior_camera_not_bound",
                }
            source_id = next(
                (
                    int(candidate)
                    for candidate, label in camera_labels.items()
                    if str(label) == requested_camera_id
                ),
                None,
            )
            if source_id is None:
                return {
                    "camera_id": requested_camera_id,
                    "scene_prior_only": True,
                    "error": "scene_prior_camera_not_active",
                }
            try:
                calibration = calibration_provider.snapshot(
                    source_id, requested_camera_id
                )
                if calibration is None or calibration.extrinsics_col_major is None:
                    return {
                        "camera_id": requested_camera_id,
                        "scene_prior_only": True,
                        "error": "scene_prior_calibration_unavailable",
                    }
                result = scene_prior_set.compose_static_floorplan(
                    requested_camera_id,
                    {
                        "camera_id": requested_camera_id,
                        "cache_only": True,
                        "scene_prior_only": True,
                        "served_from_cache": True,
                    },
                    extrinsics_col_major=calibration.extrinsics_col_major,
                )
                calibration_fingerprint = _floorplan_calibration_fingerprint(
                    calibration_provider.calibration_bundle(),
                    requested_camera_id,
                )
                if calibration_fingerprint is None:
                    raise ActiveFloorplanError(
                        "scene-prior calibration fingerprint is unavailable"
                    )
                active_floorplan_registry.record_scene_prior(
                    requested_camera_id,
                    result,
                    calibration_fingerprint=calibration_fingerprint,
                )
                return result
            except (ScenePriorError, ActiveFloorplanError) as exc:
                logger.error("Canonical scene-prior floorplan composition failed: %s", exc)
                return {
                    "camera_id": requested_camera_id,
                    "scene_prior_only": True,
                    "error": str(exc),
                }
        providers = getattr(pipeline, "capture_event_runtime_providers", None)
        if not isinstance(providers, CaptureEventRuntimeProviders):
            error = CaptureEventRuntimeError(
                "capture_event_controller_unavailable"
            )
            _runtime_component_failed("capture_event", error)
            return {
                "error": error.code,
                "camera_id": str(camera or "").strip(),
            }
        payload = providers.floorplan_provider(
            camera,
            max_age_sec=max_age_sec,
            grid_res_m=grid_res_m,
            max_extent_m=max_extent_m,
            cache_only=cache_only,
            **_ignored,
        )
        camera_id = str(payload.get("camera_id") or camera or "").strip()

        def _with_scene_fusion(result: Mapping[str, Any]) -> Dict[str, Any]:
            if scene_fusion_set is None:
                return dict(result)
            try:
                return scene_fusion_set.compose_floorplan(camera_id, result)
            except SceneFusionError as exc:
                failed = dict(result)
                failed["scene_fusion_error"] = str(exc)
                logger.error("Scene-fusion floorplan composition failed: %s", exc)
                return failed

        if scene_prior_set is None:
            return _with_scene_fusion(payload)
        binding = scene_prior_set.binding(camera_id)
        if binding is None or not binding.include_floorplan_layers:
            return _with_scene_fusion(payload)
        if payload.get("error"):
            try:
                return _with_scene_fusion(
                    scene_prior_set.compose_static_floorplan(camera_id, payload)
                )
            except ScenePriorError as exc:
                result = dict(payload)
                result["scene_prior_error"] = str(exc)
                logger.error("Scene-prior static floorplan composition failed: %s", exc)
                return _with_scene_fusion(result)
        source_id = next(
            (
                int(candidate)
                for candidate, label in camera_labels.items()
                if str(label) == camera_id
            ),
            None,
        )
        try:
            if source_id is None:
                raise ScenePriorError(
                    f"scene-prior camera {camera_id!r} is not an active calibrated source"
                )
            calibration = calibration_provider.snapshot(source_id, camera_id)
            if calibration is None or calibration.extrinsics_col_major is None:
                raise ScenePriorError(
                    f"scene-prior camera {camera_id!r} has no calibrated extrinsics"
                )
            revision = scene_prior_set.revision_for_camera(camera_id)
            if revision is None:
                raise ScenePriorError(
                    f"scene-prior camera {camera_id!r} has no loaded revision"
                )
            return _with_scene_fusion(
                scene_prior_set.compose_floorplan(
                    camera_id,
                    payload,
                    extrinsics_col_major=calibration.extrinsics_col_major,
                    floor_y_m=revision.manifest.derivation.floor_y_m,
                )
            )
        except ScenePriorError as exc:
            result = dict(payload)
            result["scene_prior_error"] = str(exc)
            result["scene_prior_meta"] = {
                "contract": "noesis.scene_prior.floorplan_composite",
                "contract_version": 1,
                "mode": "shadow",
                "status": "error",
                "camera_id": camera_id,
                "reason": str(exc),
            }
            logger.error("Scene-prior floorplan composition failed: %s", exc)
            return _with_scene_fusion(result)

    pipeline = ds8_pipeline.build_pipeline(pipeline_path)
    setattr(pipeline, "active_floorplan_registry", active_floorplan_registry)
    runtime_state["expected_eos"] = pipeline_expects_finite_source_eos(pipeline)
    runtime_state["pipeline_eos_seen"] = False
    runtime_state["pipeline_eos_reason"] = (
        "finite_source" if runtime_state["expected_eos"] else None
    )
    setattr(pipeline, "camera_labels", camera_labels)
    if getattr(pipeline, "ds_pipeline", None) is None:
        logger.error("pyservicemaker unavailable; DS9 runtime cannot continue")
        return _abort_startup("pyservicemaker_unavailable")
    try:
        scene_prior_set = _load_scene_priors(
            pipeline_cfg=pipeline.config,
            pipeline_path=pipeline_path,
            logger=logger,
        )
    except ScenePriorError as exc:
        logger.error("Configured scene priors are invalid: %s", exc)
        return _abort_startup("scene_prior_invalid")
    setattr(pipeline, "scene_priors", scene_prior_set)
    try:
        scene_fusion_set = _load_scene_fusions(
            pipeline_cfg=pipeline.config,
            pipeline_path=pipeline_path,
            logger=logger,
        )
    except SceneFusionError as exc:
        logger.error("Configured scene fusions are invalid: %s", exc)
        return _abort_startup("scene_fusion_invalid")
    setattr(pipeline, "scene_fusions", scene_fusion_set)
    # Parse mosaic_output toggles from the *built pipeline config* (source of truth).
    # Do not re-apply env overrides here: env vars are consumed during build in ds8_pipeline,
    # and re-applying them here can desync runtime behavior from the actual pipeline graph.
    mosaic_cfg = pipeline.config.get("mosaic_output") or {}
    mosaic_webrtc_enabled = bool(mosaic_cfg.get("mosaic_webrtc_enabled", False))

    rtsp_built = "rtsp_out" in getattr(pipeline, "components", {})

    mosaic_shm_built = "mosaic_h264_shmsink" in getattr(pipeline, "components", {})
    mosaic_h264_shm_socket = str(
        mosaic_cfg.get("mosaic_h264_shm_socket", "") or "/tmp/noesis-mosaic-h264"
    ).strip() or "/tmp/noesis-mosaic-h264"
    logger.info(
        "Mosaic output toggles (effective): RTSP=%s (built=%s), WebRTC_Gateway=%s (shm_built=%s socket=%s)",
        bool(mosaic_cfg.get("rtsp_enabled", False)),
        rtsp_built,
        mosaic_webrtc_enabled,
        mosaic_shm_built,
        mosaic_h264_shm_socket,
    )

    calibration_provider = create_calibration_manager(
        cameras_yaml_path=cameras_path,
        pipeline_config=pipeline.config,
        camera_calibration_json_path=Path(
            os.environ.get("NOESIS_CAMERA_CALIBRATION_FILE", "")
            or REPO_ROOT / "config" / "camera_calibration.json"
        ).expanduser().resolve(),
        ply_alignment_json_path=Path(
            os.environ.get("NOESIS_PLY_ALIGNMENT_FILE", "")
            or REPO_ROOT / "config" / "ply_alignment.json"
        ).expanduser().resolve(),
        camera_labels=camera_labels,
    )
    setattr(pipeline, "bev_calibration", calibration_provider)
    if calibration_provider.pose_only_enabled():
        pose_errors = calibration_provider.validate_pose_coverage()
        if pose_errors:
            logger.error("NOESIS_CALIBRATION_POSE_ONLY=1 startup validation failed; missing/invalid pose for cameras:")
            for camera_id, reason in sorted(pose_errors.items()):
                logger.error("  camera=%s reason=%s", camera_id, reason)
            logger.error("Aborting startup due to strict pose-only calibration mode.")
            return _abort_startup("pose_only_calibration_invalid")
        logger.warning("Strict pose-only calibration mode enabled (NOESIS_CALIBRATION_POSE_ONLY=1).")
    try:
        storage_manager.calibration_bundle = calibration_provider.calibration_bundle()
    except Exception:
        logger.debug("Unable to seed calibration bundle on storage manager", exc_info=True)
    depth_registration_manager: DepthRegistrationManager | None = None
    world_fusion_policy: WorldFusionPolicy | None = None
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
            world_fusion_policy = _load_world_measurement_fusion_policy(
                pipeline_cfg=pipeline.config,
                camera_labels=camera_labels,
                depth_registration=depth_registration_manager,
                logger=logger,
            )
        except (DepthRegistrationError, WorldFusionPolicyError) as exc:
            logger.error("Baseline tracking calibration policy is invalid: %s", exc)
            return _abort_startup("world_measurement_policy_invalid")
    elif depthless_reid_smoke:
        logger.warning("Skipping depth registration load for explicit ReID smoke validation config")
    stable_id_mgr = _build_stable_id_manager(
        logger,
        pipeline_config=pipeline.config,
        camera_labels=camera_labels,
        tracking_mode=tracking_mode,
    )
    if stable_id_mgr is None:
        logger.error("Stable ID manager is required for zero-copy hard-cutover; aborting startup")
        return _abort_startup("stable_id_manager_unavailable")
    # Ensure occupancy publisher slot exists for telemetry hooks; real publisher can be bound later.
    bind_occupancy_publisher(pipeline, None)
    # Stable ID manager is optional; attach slot so hooks can discover it.
    setattr(pipeline, "stable_id_mgr", stable_id_mgr)
    try:
        from noesis.server import reid_api

        startup_transaction.bind(
            "reid_api_binding",
            lambda: reid_api.register_reid_manager_getter(
                lambda: getattr(pipeline, "stable_id_mgr", None)
            ),
            reid_api.clear_reid_manager_getter,
        )
    except StartupResourceRegistrationError:
        raise
    except Exception:
        logger.exception("Required ReID API registration failed")
        return _abort_startup("reid_api_binding_failed")
    # Intrinsics come from the shared CalibrationManager rather than per-frame user meta.

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
    if tracking_mode in ("v3dt", "sv3dt", "mv3dt"):
        try:
            hooks.attach_v3dt_cuboid_overlay_hook(
                pipeline, tracking_mode=tracking_mode
            )
        except Exception:
            logger.exception("Failed to attach V3DT cuboid correction")
            return _abort_startup("v3dt_cuboid_correction_failed")
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
    bev_coverage_envelopes_cfg = (
        bev_cfg.get("coverage_envelopes")
        if isinstance(bev_cfg, dict)
        else None
    )
    locked_bev_frame = "camera_local_ground_m"
    bev_frame = (
        bev_cfg.get("frame") or bev_cfg.get("frame_mode")
        if isinstance(bev_cfg, dict)
        else None
    )
    bev_frame_env = str(os.environ.get("NOESIS_BEV_FRAME") or "").strip()
    if bev_frame_env and bev_frame_env != locked_bev_frame:
        logger.error("NOESIS_BEV_FRAME cannot override the locked local floorplan frame")
        return _abort_startup("bev_frame_override_rejected")
    if bev_frame != locked_bev_frame:
        logger.error("BEV frame must be %s; configured=%r", locked_bev_frame, bev_frame)
        return _abort_startup("bev_frame_contract_invalid")
    bev_frame = locked_bev_frame
    # JPEG BEV binary delivery retired (meta-only is the supported baseline per contracts + design decisions).
    # The old jpeg_enabled / NOESIS_BEV_JPEG_ENABLED knobs are ignored; only frame mode remains relevant.
    logger.info("BEV JPEG output retired (meta-only mode). frame mode=%s", bev_frame)

    def _runtime_component_failed(component: str, error: BaseException) -> None:
        message = f"{component}_failed:{type(error).__name__}:{error}"
        if message not in pipeline.errors:
            pipeline.errors.append(message)
        runtime_state["pipeline_failed"] = True
        capture_controller = getattr(pipeline, "capture_event_controller", None)
        if isinstance(capture_controller, CaptureEventController):
            capture_controller.close_admission()
        shutdown_event.set()

    def _ws_boundary_failed(error: BaseException) -> None:
        _runtime_component_failed("ws_boundary", error)

    def _bev_failed(error: BaseException) -> None:
        _runtime_component_failed("bev_renderer", error)

    def _mapanything_failed(error: BaseException) -> None:
        _runtime_component_failed("mapanything_postprocess", error)

    capture_event_runtime_providers = CaptureEventRuntimeProviders(
        aliases=capture_camera_aliases,
        storage=storage_manager,
        controller_getter=lambda: getattr(
            pipeline,
            "capture_event_controller",
            None,
        ),
        depth_branch_available=lambda: bool(
            pipeline.depth_gate_attach
            and pipeline.depth_gate_attach in pipeline.components
        ),
        shutdown_requested=shutdown_event.is_set,
        runtime_failure=_runtime_component_failed,
        record_floorplan=_record_active_floorplan_payload,
    )
    setattr(
        pipeline,
        "capture_event_runtime_providers",
        capture_event_runtime_providers,
    )

    ws_server = WebSocketServer(
        host=args.ws_host,
        port=args.ws_port,
        stats_callback=None,
        initial_trail_state=bool(trail_settings.enabled),
    )
    ws_server.boundary_failure_callback = _ws_boundary_failed
    ws_server.lifecycle_failure_callback = _ws_boundary_failed
    def _ws_trail_settings_getter() -> Dict[str, Any]:
        cfg = dict(trails_cfg or {})
        cfg["enabled"] = bool(ws_server.initial_trail_state)
        return cfg
    ws_server.trail_settings_getter = _ws_trail_settings_getter
    ws_server.stats_callback = _build_stats_callback(
        pipeline,
        camera_labels,
        ws_metrics_getter=ws_server.get_boundary_serialization_metrics_compact,
        ws_metrics_resetter=ws_server.reset_boundary_serialization_metrics,
        runtime_state=runtime_state,
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
            response_model_started_ns = time.perf_counter_ns()
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
            message = {"type": "calibration-bundle", "data": bundle}
            ws_server.broadcast_sync(
                message,
                response_model_timing=ws_server.response_model_timing_since(
                    response_model_started_ns
                ),
            )
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
        active_floorplan_registry.clear(cam_id)
        storage_manager.invalidate_floorplan_cache(
            active_floorplan_registry.canonical_camera(cam_id)
        )
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
        active_floorplan_registry.clear()
        storage_manager.invalidate_floorplan_cache()
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
        floorplan_bounds_provider=_bev_active_floorplan_bounds_provider,
        coverage_envelopes_cfg=bev_coverage_envelopes_cfg,
        failure_callback=_bev_failed,
        # jpeg_* retired — meta-only mode (see BevRenderer and design decisions)
    )
    setattr(pipeline, "bev_renderer", bev_renderer)
    ws_server.bev_config_callback = lambda cam_id, cfg: bev_renderer.update_config(cam_id, cfg)
    ws_server.bev_overlay_callback = lambda cam_id, enabled: bev_renderer.update_config(cam_id, {"overlay": enabled})
    runtime_publication_gate = RuntimePublicationGate()
    setattr(pipeline, "runtime_publication_gate", runtime_publication_gate)
    depth_pub = DepthTelemetryPublisher(
        ws_server,
        failure_callback=_ws_boundary_failed,
        publication_gate=runtime_publication_gate,
    )
    world_service = startup_transaction.acquire(
        "canonical_world",
        lambda: create_runtime_world_service(
            runtime="ds9",
            pipeline_config=pipeline.config,
            camera_labels=camera_labels,
            calibration_provider=calibration_provider,
            repo_root=REPO_ROOT,
            software_revision=(
                appliance_binding.software_revision
                if appliance_binding is not None
                else None
            ),
        ),
        lambda service: service.close(),
    )
    capability_monitor = create_runtime_capability_monitor(world_service)
    from noesis.server import health_api

    startup_transaction.bind(
        "health_api_bindings",
        lambda: (
            health_api.register_capability_monitor_getter(
                lambda: capability_monitor
            ),
            health_api.register_deployment_binding_getter(
                lambda: appliance_binding
            ),
        ),
        health_api.clear_runtime_bindings,
    )
    if appliance_binding is not None:
        appliance_binding.bind_producer(
            instance_id=world_service.producer.instance_id,
            run_id=world_service.producer.run_id,
        )
        ws_server.health_payload_getter = lambda: appliance_binding.websocket_health(
            capability_monitor
        ).model_dump(mode="json")

    def _canonical_publication_failed(error: BaseException) -> None:
        message = f"canonical_world_publication_failed:{type(error).__name__}:{error}"
        if message not in pipeline.errors:
            pipeline.errors.append(message)
        runtime_state["pipeline_failed"] = True
        shutdown_event.set()

    from noesis.identity_v2_service import (
        IdentityV2ConfigurationError,
        create_identity_v2_service,
    )
    try:
        identity_v2_service = create_identity_v2_service(
            pipeline_config=pipeline.config,
            pipeline_yaml_path=getattr(pipeline, "yaml_path", pipeline_path),
            camera_labels=camera_labels,
            repo_root=REPO_ROOT,
            run_id=world_service.producer.run_id,
        )
    except IdentityV2ConfigurationError as exc:
        logger.error("Identity-v2 startup validation failed: %s", exc)
        return _abort_startup("identity_v2_configuration_invalid")
    except Exception:
        logger.exception("Identity-v2 startup failed")
        return _abort_startup("identity_v2_startup_failed")
    setattr(pipeline, "identity_v2_service", identity_v2_service)
    if identity_v2_service is not None:
        identity_v2_service = startup_transaction.acquire(
            "identity_v2",
            lambda: identity_v2_service,
            lambda service: service.close(),
        )

    def _identity_v2_failed(error: BaseException) -> None:
        message = f"identity_v2_failed:{type(error).__name__}:{error}"
        if message not in pipeline.errors:
            pipeline.errors.append(message)
        runtime_state["pipeline_failed"] = True
        shutdown_event.set()

    setattr(pipeline, "identity_v2_failure_callback", _identity_v2_failed)
    from noesis.server import reid_v2_api

    startup_transaction.bind(
        "identity_v2_api_bindings",
        lambda: (
            reid_v2_api.register_identity_v2_runtime_getter(
                lambda: (
                    identity_v2_service.runtime
                    if identity_v2_service is not None
                    else None
                )
            ),
            reid_v2_api.register_identity_v2_service_getter(
                lambda: identity_v2_service
            ),
        ),
        reid_v2_api.clear_identity_v2_bindings,
    )
    if identity_v2_service is None:
        logger.info("Identity-v2 runtime disabled")
    else:
        logger.info(
            "Identity-v2 runtime ready: mode=%s run_id=%s model_sha256=%s layer=%s dim=%d scoring=%s",
            identity_v2_service.mode.value,
            identity_v2_service.run_id,
            identity_v2_service.model_fingerprint,
            identity_v2_service.model_layer,
            identity_v2_service.embedding_dim,
            identity_v2_service.runtime.scoring_calibration_status,
        )

    tracking_pub = TrackingTelemetryPublisher(
        ws_server,
        metadata_getter=_tracking_contract_metadata,
        world_service=world_service,
        health_monitor=capability_monitor,
        failure_callback=_canonical_publication_failed,
    )
    logger.info("Canonical world service ready: run_id=%s", world_service.producer.run_id)
    diagnostics_logger = TrackingDiagnosticsLogger.from_env()
    if diagnostics_logger:
        diagnostics_logger = startup_transaction.acquire(
            "tracking_diagnostics",
            lambda: diagnostics_logger,
            lambda diagnostics: diagnostics.close(),
        )
        logger.info("V3DT diagnostics logging enabled: %s", diagnostics_logger.output_path)

    capture_rgb_provider: PipelineRgbFrameProvider | None = None

    def _close_startup_mapanything_processor() -> None:
        processor = getattr(pipeline, "mapanything_processor", None)
        try:
            if processor is not None:
                processor.shutdown(wait=True, timeout_s=5.0)
        finally:
            if capture_rgb_provider is not None:
                capture_rgb_provider.close()

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
            capture_rgb_provider = PipelineRgbFrameProvider(
                camera_sources=camera_labels,
            )

            def _install_mapanything_processor() -> Any:
                hooks.attach_mapanything_postprocess_hook(
                    pipeline,
                    storage=storage_manager,
                    depth_pub=depth_pub,
                    camera_labels=camera_labels,
                    failure_callback=_mapanything_failed,
                    rgb_provider=capture_rgb_provider,
                )
                processor = getattr(pipeline, "mapanything_processor", None)
                if processor is None:
                    raise RuntimeError(
                        "MapAnything hook did not publish its owned processor"
                    )
                return processor

            mapanything_processor = startup_transaction.bind(
                "mapanything_processor",
                _install_mapanything_processor,
                _close_startup_mapanything_processor,
            )
            capture_event_controller = CaptureEventController(
                aliases=capture_camera_aliases,
                storage=DepthStorageCaptureEventAdapter(storage_manager),
                mapanything=mapanything_processor,
                set_depth_gate=pipeline.mark_depth_enabled,
                depth_gate_is_open=lambda: bool(pipeline.depth_enabled),
                burst_waiter=shutdown_event.wait,
                stop_requested=shutdown_event.is_set,
                failure_callback=lambda error: _runtime_component_failed(
                    "capture_event", error
                ),
                manual_drain_timeout_s=resolve_capture_event_drain_timeout_s(),
                rgb_provider=capture_rgb_provider,
                require_rgb=True,
            )
            setattr(
                pipeline,
                "capture_event_controller",
                capture_event_controller,
            )
        else:
            logger.info("SGIE disabled or missing; skipping MapAnything postprocess hook")
    except StartupResourceRegistrationError:
        raise
    except Exception as exc:
        logger.exception("Error while evaluating MapAnything postprocess attachment")
        _mapanything_failed(exc)
        return _abort_startup("mapanything_hook_failed")
    try:
        hooks.attach_pose_feature_hook(pipeline, camera_labels=camera_labels)
    except Exception:
        logger.exception("Error while attaching pose feature hook")
        return _abort_startup("pose_feature_hook_failed")
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
            return _abort_startup("object_depth_fusion_hook_failed")
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
        world_fusion_policy=world_fusion_policy,
        scene_priors=scene_prior_set,
        diagnostics_logger=diagnostics_logger,
        publication_gate=runtime_publication_gate,
    )
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

    if shutdown_event.is_set():
        return _abort_startup("shutdown_requested_before_prepare")
    logger.info("Preparing DS9 pipeline")
    if not ds8_pipeline.prepare(on_message=_psm_message_cb):
        logger.error("DS9 pipeline preparation failed: %s", pipeline.errors)
        return _abort_startup("pipeline_prepare_failed")
    startup_transaction.mark_prepared()
    if shutdown_event.is_set():
        return _abort_startup("shutdown_requested_after_prepare")

    try:
        ws_thread, ws_loop = _start_websocket_server(ws_server)
    except WebSocketStartupError as exc:
        if not exc.receipt.quiesced:
            logger.critical(
                "WebSocket startup cleanup was not proven; refusing normal return"
            )
            runtime_state["pipeline_failed"] = True
            _arm_shutdown_watchdog()
            while True:
                signal.pause()
        logger.error("WebSocket server failed bounded startup: %s", exc)
        return _abort_startup("websocket_startup_failed")
    ws_owner = (ws_thread, ws_loop)
    startup_transaction.acquire(
        "websocket_listener",
        lambda: ws_owner,
        lambda owner: _stop_websocket_server(ws_server, owner[0], owner[1]),
        ingress=True,
    )
    if getattr(ws_server, "server", None) is None:
        logger.error(
            "WebSocket server failed to bind required endpoint %s:%s; aborting DS9 runtime",
            args.ws_host,
            args.ws_port,
        )
        return _abort_startup("websocket_listener_missing")
    if shutdown_event.is_set():
        return _abort_startup("shutdown_requested_before_activation")

    # Activate the DS9 pipeline after prepare() using activate() not start()
    # NOTE: We use activate() because prepare() was already called above.
    # Using start() after prepare() causes "Tried to add new watch while one was already there"
    # because start() internally calls prepare() + sets bus watch, conflicting with existing watch.
    try:
        startup_transaction.bind(
            "servicemaker_stdin_keepalive",
            lambda: _install_servicemaker_stdin_keepalive(logger),
            _close_servicemaker_stdin_keepalive,
        )
    except StartupResourceRegistrationError:
        raise
    except Exception:
        logger.exception("Failed to establish Service Maker stdin lifecycle contract")
        return _abort_startup("servicemaker_stdin_contract_failed")
    if shutdown_event.is_set():
        return _abort_startup("shutdown_requested_before_activation")
    startup_transaction.mark_activation_attempted()
    if not ds8_pipeline.activate():
        logger.error("DS9 pipeline activation failed: %s", pipeline.errors)
        _abort_ambiguous_and_wait("pipeline_activation_failed")
    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        logger.error("DS9 pipeline activated but ds_pipeline is missing")
        _abort_ambiguous_and_wait("activated_pipeline_owner_missing")
    logger.info("DS9 pipeline activated successfully")

    # Start pyservicemaker wait loop to keep pipeline alive and processing events
    # This is critical - without wait(), the pipeline may stop after initial buffers
    wait_thread = _start_pyservicemaker_wait_loop(ds, shutdown_event, logger, runtime_state)
    try:
        startup_transaction.claim_wait_owner(wait_thread)
    except StartupOwnershipAmbiguous as exc:
        logger.critical("Service Maker wait ownership was not proven")
        _preserve_ambiguous_and_wait("servicemaker_wait_owner_missing", exc)
    startup_transaction.handoff_to_runtime()
    startup_main_guard.disarm()

    source_progress_monitor = getattr(pipeline, "source_progress_monitor", None)
    if source_progress_monitor is None:
        logger.critical(
            "Canonical per-source pipeline has no decoded-progress monitor"
        )
        runtime_state["pipeline_failed"] = True
        runtime_state["source_progress_failure"] = {
            "reason": "monitor_missing"
        }
        shutdown_event.set()
    else:
        def _source_progress_fatal(failure: Any) -> None:
            failure_payload = (
                failure.to_dict()
                if hasattr(failure, "to_dict")
                else {"reason": str(failure)}
            )
            runtime_state["source_progress_failure"] = failure_payload
            runtime_state["pipeline_failed"] = True
            logger.critical(
                "Decoded/dewarped source progress exhausted bounded recovery; "
                "terminating DS9 for supervisor-owned restart: %s",
                failure,
            )
            shutdown_event.set()

        try:
            source_progress_monitor.start(_source_progress_fatal)
            logger.info("Decoded/dewarped source progress monitor started")
        except Exception:
            logger.exception("Decoded/dewarped source progress monitor failed to start")
            runtime_state["pipeline_failed"] = True
            runtime_state["source_progress_failure"] = {
                "reason": "monitor_start_failed"
            }
            shutdown_event.set()

    try:
        hooks.verify_analytics_exclusion_initial_receipt(
            pipeline,
            analytics_startup_reload_context,
        )
    except Exception:
        logger.exception("Native analytics exclusion startup receipt validation failed")
        runtime_state["pipeline_failed"] = True
        shutdown_event.set()

    # Start WebRTC gateway(s) if enabled (consumes encoded AUs via SHM bridge).
    mosaic_h264_feeder = None
    if mosaic_webrtc_enabled and not shutdown_event.is_set():
        if not mosaic_shm_built:
            logger.error(
                "WebRTC gateway enabled but mosaic H.264 SHM sink was not built; cannot start gateway"
            )
            runtime_state["pipeline_failed"] = True
            shutdown_event.set()
        else:
            try:
                from noesis.mosaic_h264_bridge import MosaicH264ShmFeeder
                from noesis.mosaic_webrtc_gateway import MosaicWebRTCGateway

                try:
                    max_webrtc_clients = max(
                        1, int(os.environ.get("NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS", "5"))
                    )
                except Exception:
                    max_webrtc_clients = 5
                try:
                    initial_webrtc_clients = max(
                        0, int(os.environ.get("NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS", "1"))
                    )
                except Exception:
                    initial_webrtc_clients = 1
                initial_webrtc_clients = min(initial_webrtc_clients, max_webrtc_clients)

                def _keyframe_failed(error: BaseException) -> None:
                    message = f"mosaic_force_idr_failed:{type(error).__name__}:{error}"
                    if message not in pipeline.errors:
                        pipeline.errors.append(message)
                    runtime_state["pipeline_failed"] = True
                    shutdown_event.set()

                def _mosaic_transport_failed(error: BaseException) -> None:
                    message = (
                        "mosaic_h264_shm_failed:"
                        f"{type(error).__name__}:{error}"
                    )
                    if message not in pipeline.errors:
                        pipeline.errors.append(message)
                    runtime_state["pipeline_failed"] = True
                    shutdown_event.set()

                keyframe_requester = _build_mosaic_keyframe_requester(
                    pipeline,
                    logger,
                    failure_callback=_keyframe_failed,
                )
                if keyframe_requester is None:
                    raise RuntimeError(
                        "WebRTC late-viewer support requires a mosaic force-IDR requester"
                    )

                mosaic_h264_feeder = MosaicH264ShmFeeder(
                    mosaic_h264_shm_socket,
                    request_keyframe=keyframe_requester,
                    on_fatal_error=_mosaic_transport_failed,
                )
                mosaic_h264_feeder.start()
                logger.info(
                    "Mosaic H.264 SHM feeder started at %s",
                    mosaic_h264_shm_socket,
                )

                def _create_mosaic_gateway() -> MosaicWebRTCGateway:
                    gateway = MosaicWebRTCGateway(
                        ws_server=ws_server,
                        h264_feeder=mosaic_h264_feeder,
                        request_keyframe=keyframe_requester,
                    )
                    try:
                        gateway.start()
                    except Exception:
                        try:
                            gateway.stop()
                        except Exception:
                            logger.exception(
                                "Partially started WebRTC gateway did not quiesce"
                            )
                        raise
                    logger.info("WebRTC gateway started (H.264 AU feeder)")
                    return gateway

                ws_server.register_webrtc_gateway_factory(
                    _create_mosaic_gateway,
                    max_clients=max_webrtc_clients,
                    initial_clients=initial_webrtc_clients,
                )
                for slot in range(initial_webrtc_clients):
                    gateway = None
                    try:
                        gateway = _create_mosaic_gateway()
                        ws_server.register_webrtc_gateway(gateway)
                    except Exception:
                        if gateway is not None:
                            try:
                                gateway.stop()
                            except Exception:
                                logger.exception(
                                    "Unregistered WebRTC gateway slot %d did not stop",
                                    slot + 1,
                                )
                        logger.exception("Failed to start WebRTC gateway slot %d", slot + 1)
                        runtime_state["pipeline_failed"] = True
                        shutdown_event.set()
                logger.info(
                    "WebRTC gateway capacity: %d max, %d warm slot(s)",
                    max_webrtc_clients,
                    len(ws_server.webrtc_gateways),
                )
            except Exception:
                logger.exception("Failed to start WebRTC gateway")
                runtime_state["pipeline_failed"] = True
                shutdown_event.set()

    depth_branch_present = bool(pipeline.depth_gate_attach and pipeline.depth_gate_attach in pipeline.components)
    if args.depth_enable_seconds > 0:
        if depth_branch_present:
            try:
                depth_controller = getattr(pipeline, "capture_event_controller", None)
                if not isinstance(depth_controller, CaptureEventController):
                    raise RuntimeError("capture-event depth controller is unavailable")
                depth_controller.start_refresh(args.depth_enable_seconds)
                logger.info(
                    "Depth branch enabled for %s seconds (startup burst)",
                    args.depth_enable_seconds,
                )
            except Exception:
                logger.exception("Failed to enable depth burst on startup")
                runtime_state["pipeline_failed"] = True
                shutdown_event.set()
        else:
            logger.error("Requested startup depth burst but no MapAnything branch is present")
            runtime_state["pipeline_failed"] = True
            shutdown_event.set()
    if not getattr(pipeline, "activated", False):
        logger.warning("DS9 pipeline not activated; check pipeline.errors for details: %s", pipeline.errors)

    rest_server = None
    rest_thread = None

    if args.enable_rest and not shutdown_event.is_set():
        try:
            if prebuilt_rest_app is None:
                raise RuntimeError("required prebuilt DS9 REST app is unavailable")
            rest_app = prebuilt_rest_app
            depth_controller = getattr(pipeline, "capture_event_controller", None)
            if not isinstance(depth_controller, CaptureEventController):
                raise RuntimeError("capture-event depth controller is unavailable")
            rest_app.state.depth_refresh_provider = depth_controller.start_refresh
            rest_app.state.depth_storage = storage_manager
            rest_server, rest_thread = _start_rest_server(rest_app, args.rest_host, args.rest_port)
            logger.info("REST server listening on http://%s:%s", args.rest_host, args.rest_port)
        except RestStartupError as exc:
            runtime_state["rest_startup_receipt"] = {
                "cleanup_proven": exc.cleanup_proven,
                "thread_alive": exc.thread.is_alive(),
            }
            if not exc.cleanup_proven:
                logger.critical(
                    "REST startup ownership is unresolved; preserving callback-owned "
                    "resources until watchdog exit",
                    exc_info=True,
                )
                runtime_state["pipeline_failed"] = True
                _arm_shutdown_watchdog()
                while True:
                    signal.pause()
            logger.exception("Required REST server failed bounded startup")
            runtime_state["pipeline_failed"] = True
            shutdown_event.set()
        except Exception:
            logger.exception("Required REST server failed to start")
            runtime_state["pipeline_failed"] = True
            shutdown_event.set()

    # DeepStream dependencies may replace handlers during initialization.
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    logger.info("DS9 runtime is active. Press Ctrl+C to stop.")
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()

    logger.info("Shutting down DS9 runtime")
    _arm_shutdown_watchdog()
    runtime_state["expected_eos"] = True
    if wait_thread is None or wait_thread.is_alive():
        runtime_state["pipeline_eos_reason"] = "shutdown_requested"

    source_progress_receipt = {
        "present": source_progress_monitor is not None,
        "quiesced": source_progress_monitor is None,
        "fatal": False,
    }
    if source_progress_monitor is not None:
        try:
            source_progress_receipt = source_progress_monitor.stop(
                timeout_seconds=3.0
            )
        except Exception:
            logger.exception("Decoded source progress monitor shutdown failed")
            source_progress_receipt = {
                "present": True,
                "quiesced": False,
                "fatal": True,
            }
    runtime_state["source_progress_shutdown_receipt"] = source_progress_receipt
    if not bool(source_progress_receipt.get("quiesced")):
        runtime_state["pipeline_failed"] = True
        logger.error("Decoded source progress monitor did not quiesce")

    # Quiesce external request/consumer edges before touching native producers.
    # The retained lock is intentionally not released: it is the shutdown lease
    # that prevents a late REST worker from entering an analytics transaction.
    rest_shutdown_receipt = _stop_rest_server(
        rest_server,
        rest_thread,
        analytics_api._CONFIG_LOCK,  # type: ignore[attr-defined]
    )
    runtime_state["rest_shutdown_receipt"] = {
        **rest_shutdown_receipt._asdict(),
        "quiesced": rest_shutdown_receipt.quiesced,
    }
    if not rest_shutdown_receipt.quiesced:
        runtime_state["pipeline_failed"] = True
        logger.critical(
            "REST/analytics quiescence was not proven "
            "(rest_pair_consistent=%s stop_requested=%s server_thread_stopped=%s "
            "analytics_transaction_lock_retained=%s); preserving callback-owned "
            "native resources until watchdog exit",
            rest_shutdown_receipt.rest_pair_consistent,
            rest_shutdown_receipt.stop_requested,
            rest_shutdown_receipt.server_thread_stopped,
            rest_shutdown_receipt.analytics_transaction_lock_retained,
        )
        while True:
            signal.pause()

    try:
        stats_shutdown_receipt = ws_server.quiesce_stats_collector(
            timeout_s=WebSocketServer.STATS_COLLECTOR_DRAIN_TIMEOUT_S
        )
    except Exception as exc:
        failed_receipt = getattr(exc, "receipt", None)
        if failed_receipt is not None and hasattr(failed_receipt, "_asdict"):
            runtime_state["ws_stats_shutdown_receipt"] = {
                **failed_receipt._asdict(),
                "quiesced": False,
            }
        runtime_state["pipeline_failed"] = True
        logger.critical(
            "WebSocket stats collector did not quiesce; preserving native "
            "resources until watchdog exit: %s",
            exc,
        )
        while True:
            signal.pause()
    runtime_state["ws_stats_shutdown_receipt"] = {
        **stats_shutdown_receipt._asdict(),
        "quiesced": stats_shutdown_receipt.quiesced,
    }

    try:
        provider_shutdown_receipt = ws_server.quiesce_blocking_providers(
            timeout_s=5.0
        )
    except Exception as exc:
        failed_receipt = getattr(exc, "receipt", None)
        if failed_receipt is not None and hasattr(failed_receipt, "_asdict"):
            runtime_state["ws_provider_shutdown_receipt"] = {
                **failed_receipt._asdict(),
                "quiesced": False,
            }
        runtime_state["pipeline_failed"] = True
        logger.critical(
            "Blocking WebSocket providers did not quiesce; preserving native "
            "and storage resources until watchdog exit: %s",
            exc,
        )
        while True:
            signal.pause()
    runtime_state["ws_provider_shutdown_receipt"] = {
        **provider_shutdown_receipt._asdict(),
        "quiesced": provider_shutdown_receipt.quiesced,
    }

    try:
        detached_gateways = ws_server.begin_webrtc_shutdown(timeout_s=5.0)
    except Exception:
        runtime_state["pipeline_failed"] = True
        logger.exception("WebRTC gateway admission/lifecycle drain failed")
        while True:
            signal.pause()

    gateways_to_stop: List[Any] = []
    for gateway in list(detached_gateways):
        if gateway is not None and gateway not in gateways_to_stop:
            gateways_to_stop.append(gateway)
    gateways_quiesced = True
    gateway_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    if gateways_to_stop:
        gateway_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gateways_to_stop),
            thread_name_prefix="NoesisWebRTCStop",
        )
        gateway_futures = {
            gateway_executor.submit(gateway.stop): idx
            for idx, gateway in enumerate(gateways_to_stop, start=1)
        }
        done, pending = concurrent.futures.wait(
            gateway_futures,
            timeout=WebSocketServer.WEBRTC_GATEWAY_DRAIN_TIMEOUT_S,
        )
        for future in done:
            idx = gateway_futures[future]
            try:
                future.result()
                logger.info("WebRTC gateway slot %d stopped", idx)
            except Exception:
                logger.exception("Error stopping WebRTC gateway slot %d", idx)
                gateways_quiesced = False
        if pending:
            logger.error(
                "WebRTC gateway slots exceeded concurrent drain timeout: %s",
                sorted(gateway_futures[future] for future in pending),
            )
            gateways_quiesced = False
        gateway_executor.shutdown(
            wait=not pending,
            cancel_futures=False,
        )
    if not gateways_quiesced:
        runtime_state["pipeline_failed"] = True

    if not gateways_quiesced:
        logger.critical(
            "WebRTC gateways did not quiesce; preserving native resources "
            "until watchdog exit"
        )
        while True:
            signal.pause()

    if mosaic_h264_feeder is not None:
        try:
            mosaic_h264_feeder.stop()
            logger.info("Mosaic H.264 SHM feeder stopped")
        except Exception:
            runtime_state["pipeline_failed"] = True
            logger.exception("Failed to stop mosaic H.264 SHM feeder")
            while True:
                signal.pause()

    ws_server.webrtc_activity_callback = None

    try:
        pipeline.cancel_control_timers()
        pipeline.mark_depth_enabled(False)
    except Exception:
        runtime_state["pipeline_failed"] = True
        logger.exception("Failed to quiesce pipeline control timers")
        while True:
            signal.pause()

    try:
        publication_receipt = runtime_publication_gate.close_and_wait(
            timeout_s=5.0
        )
        runtime_state["runtime_publication_shutdown_receipt"] = (
            publication_receipt.to_dict()
        )
        logger.info(
            "Native runtime publication callbacks quiesced: "
            "admitted=%d completed=%d rejected=%d active=%d",
            publication_receipt.admitted,
            publication_receipt.completed,
            publication_receipt.rejected,
            publication_receipt.active,
        )
    except RuntimePublicationQuiescenceError as exc:
        runtime_state["runtime_publication_shutdown_receipt"] = (
            exc.receipt.to_dict()
        )
        runtime_state["pipeline_failed"] = True
        logger.critical(
            "Native runtime publication callbacks did not quiesce; "
            "preserving WebSocket and native resources until watchdog exit: %s",
            exc,
        )
        while True:
            signal.pause()
    except Exception:
        runtime_state["pipeline_failed"] = True
        logger.exception("Native runtime publication gate shutdown failed")
        while True:
            signal.pause()

    try:
        _stop_websocket_server(ws_server, ws_thread, ws_loop)
    except Exception:
        runtime_state["pipeline_failed"] = True
        logger.exception("WebSocket listener/worker quiescence failed")
        while True:
            signal.pause()
    runtime_state["websocket_shutdown_quiesced"] = True

    # The live-source reconnect probe inside nvurisrcbin intentionally drops
    # pipeline-level EOS.  Request EOS from the repo-owned bridge immediately
    # downstream of streammux; it emits asynchronously after Service Maker has
    # released setter locks.  Require its exact bounded acknowledgement, the
    # pipeline EOS callback, and wait-thread completion before releasing any
    # callback-owned resources.
    wait_was_alive = bool(wait_thread is not None and wait_thread.is_alive())
    eos_accepted = False
    if wait_was_alive:
        try:
            logger.info("Orderly pipeline EOS request initiated")
            eos_evidence = request_orderly_eos(pipeline)
            eos_accepted = True
            logger.info(
                "Orderly pipeline EOS accepted: component=%s request_sequence=%d",
                eos_evidence.component_name,
                eos_evidence.request_sequence,
            )
        except OrderlyEosError:
            logger.exception("Orderly pipeline EOS request failed")
            runtime_state["pipeline_failed"] = True
    else:
        eos_accepted = bool(runtime_state.get("pipeline_eos_seen"))

    wait_completed = wait_thread is not None and not wait_thread.is_alive()
    if wait_thread is not None and wait_thread.is_alive():
        try:
            wait_thread.join(timeout=15.0)
            wait_completed = not wait_thread.is_alive()
            if not wait_completed:
                logger.error("pyservicemaker wait thread did not terminate")
        except Exception:
            wait_completed = False
            logger.exception("Error joining wait thread")

    eos_seen = bool(runtime_state.get("pipeline_eos_seen"))
    wait_failed = bool(runtime_state.get("wait_failed"))
    pipeline_quiesced = bool(
        eos_accepted and eos_seen and wait_completed and not wait_failed
    )
    if not pipeline_quiesced:
        runtime_state["pipeline_failed"] = True
        logger.error(
            "Pipeline quiescence was not proven "
            "(eos_accepted=%s eos_seen=%s wait_completed=%s wait_failed=%s); "
            "preserving callback-owned resources until watchdog exit",
            eos_accepted,
            eos_seen,
            wait_completed,
            wait_failed,
        )
    else:
        capture_controller = getattr(pipeline, "capture_event_controller", None)
        capture_event_quiesced = capture_controller is None
        if capture_controller is not None:
            try:
                capture_controller.shutdown(timeout_s=5.0)
                capture_health = capture_controller.health_snapshot()
                manual_health = capture_health.get("manual_refresh", {})
                capture_event_quiesced = bool(
                    manual_health.get("phase") == "shutdown"
                    and not manual_health.get("active")
                    and not manual_health.get("thread_alive")
                    and not capture_health.get("shared_gate_owned")
                )
            except Exception:
                runtime_state["pipeline_failed"] = True
                logger.exception("Capture-event controller shutdown failed")
                capture_event_quiesced = False
        runtime_state["capture_event_shutdown_receipt"] = {
            "present": capture_controller is not None,
            "quiesced": capture_event_quiesced,
        }
        logger.info(
            "Capture-event controller shutdown quiesced: present=%d quiesced=%d",
            int(capture_controller is not None),
            int(capture_event_quiesced),
        )
        if not capture_event_quiesced:
            logger.critical(
                "Capture-event gate ownership remains unresolved; preserving "
                "MapAnything and storage resources until watchdog exit"
            )
            while True:
                signal.pause()

        mapanything_processor = getattr(pipeline, "mapanything_processor", None)
        mapanything_quiesced = mapanything_processor is None
        if mapanything_processor is not None:
            try:
                shutdown_mapanything = getattr(mapanything_processor, "shutdown", None)
                if not callable(shutdown_mapanything):
                    raise RuntimeError(
                        "MapAnything processor has no owned shutdown contract"
                    )
                shutdown_mapanything(wait=True, timeout_s=5.0)
                mapanything_quiesced = True
            except Exception:
                runtime_state["pipeline_failed"] = True
                logger.exception("MapAnything postprocess shutdown failed")
                quiescence_check = getattr(
                    mapanything_processor,
                    "async_shutdown_quiesced",
                    None,
                )
                try:
                    mapanything_quiesced = bool(
                        callable(quiescence_check) and quiescence_check()
                    )
                except Exception:
                    logger.exception(
                        "MapAnything postprocess quiescence proof failed"
                    )
                    mapanything_quiesced = False
        runtime_state["mapanything_shutdown_receipt"] = {
            "present": mapanything_processor is not None,
            "quiesced": mapanything_quiesced,
        }
        logger.info(
            "MapAnything source workers shutdown quiesced: "
            "present=%d quiesced=%d",
            int(mapanything_processor is not None),
            int(mapanything_quiesced),
        )
        if not mapanything_quiesced:
            logger.critical(
                "MapAnything worker ownership remains unresolved; preserving "
                "storage and callback resources until watchdog exit"
            )
            while True:
                signal.pause()

        publication_final_receipt = runtime_publication_gate.snapshot()
        runtime_state["runtime_publication_final_receipt"] = (
            publication_final_receipt.to_dict()
        )
        if not publication_final_receipt.quiesced:
            runtime_state["pipeline_failed"] = True
            logger.critical(
                "Native runtime publication gate lost quiescence after EOS"
            )
            while True:
                signal.pause()
        logger.info(
            "Native runtime publication shutdown finalized: "
            "admitted=%d completed=%d rejected=%d active=%d",
            publication_final_receipt.admitted,
            publication_final_receipt.completed,
            publication_final_receipt.rejected,
            publication_final_receipt.active,
        )

        runtime_finalization_failures: list[dict[str, str]] = []

        def _record_finalization_failure(
            name: str,
            error: BaseException,
        ) -> None:
            runtime_state["pipeline_failed"] = True
            runtime_finalization_failures.append(
                {
                    "name": str(name),
                    "error_type": type(error).__name__,
                }
            )

        try:
            _close_servicemaker_stdin_keepalive()
        except BaseException as exc:
            logger.exception("Service Maker stdin ownership did not close")
            _record_finalization_failure("servicemaker_stdin_keepalive", exc)

        try:
            mgr = getattr(pipeline, "stable_id_mgr", None)
            if mgr is not None and getattr(mgr, "gallery_persist_file", None):
                if not mgr.save_gallery():
                    raise RuntimeError("StableID gallery save returned false")
                logger.info(
                    "StableID gallery persisted to %s",
                    mgr.gallery_persist_file,
                )
        except BaseException as exc:
            logger.exception("Error persisting StableID gallery")
            _record_finalization_failure("stable_id_gallery", exc)

        try:
            identity_v2_service = getattr(pipeline, "identity_v2_service", None)
            if identity_v2_service is not None:
                identity_v2_service.close()
        except BaseException as exc:
            logger.exception("Failed to close identity-v2 store")
            _record_finalization_failure("identity_v2", exc)

        try:
            world_service.close()
        except BaseException as exc:
            logger.exception("Failed to flush canonical world journal")
            _record_finalization_failure("canonical_world", exc)

        try:
            storage_shutdown_receipt = close_depth_storage(
                storage_manager,
                timeout_s=5.0,
            )
            runtime_state["depth_storage_shutdown_receipt"] = storage_close_evidence(
                storage_shutdown_receipt
            )
        except BaseException as exc:
            logger.exception("Depth storage ownership did not quiesce")
            _record_finalization_failure("depth_storage", exc)

        try:
            if diagnostics_logger is not None:
                diagnostics_logger.close()
        except BaseException as exc:
            logger.exception("Error closing diagnostics logger")
            _record_finalization_failure("tracking_diagnostics", exc)

        for binding_name, clear_binding in (
            ("analytics_runtime_hooks", analytics_api.clear_runtime_hooks),
            ("reid_api_binding", reid_api.clear_reid_manager_getter),
            ("identity_v2_api_bindings", reid_v2_api.clear_identity_v2_bindings),
            ("health_api_bindings", health_api.clear_runtime_bindings),
        ):
            try:
                clear_binding()
            except BaseException as exc:
                logger.exception("Runtime binding cleanup failed: %s", binding_name)
                _record_finalization_failure(binding_name, exc)

        if not runtime_finalization_failures:
            try:
                startup_transaction.mark_quiesced()
            except BaseException as exc:
                logger.exception("Runtime ownership finalization failed")
                _record_finalization_failure("startup_transaction", exc)

        runtime_state["runtime_finalization_failures"] = list(
            runtime_finalization_failures
        )
        if runtime_finalization_failures:
            logger.critical(
                "Runtime finalization was not proven; waiting for shutdown watchdog: %s",
                [row["name"] for row in runtime_finalization_failures],
            )
            while True:
                signal.pause()

        try:
            signal.alarm(0)
        except BaseException as exc:
            logger.exception("Shutdown watchdog cancellation failed")
            _record_finalization_failure("shutdown_watchdog", exc)
            runtime_state["runtime_finalization_failures"] = list(
                runtime_finalization_failures
            )
            while True:
                signal.pause()
        logger.info("Shutdown complete")
    if not pipeline_quiesced:
        logger.critical("Native pipeline teardown failed; waiting for shutdown watchdog")
        while True:
            signal.pause()
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


def main() -> int:
    startup_main_guard = StartupMainGuard()
    return startup_main_guard.run(lambda: _run_main(startup_main_guard))


if __name__ == "__main__":
    sys.exit(main())
