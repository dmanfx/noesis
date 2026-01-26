#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import configparser
import json
import logging
import os
import subprocess
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration_bundle import (
    assemble_calibration_bundle,
    load_alignment,
    load_extrinsics,
    load_intrinsics,
    save_alignment,
    save_extrinsics,
)
from geometry.depth_source import DepthStorageManager
from mapanything_config import load_service_config
from noesis.pipelines import ds8_pipeline, hooks
from noesis.metadata.intrinsics import CameraConfigLoader
from noesis.telemetry.publishers import DepthTelemetryPublisher, TrackingTelemetryPublisher, bind_occupancy_publisher
from noesis.diagnostics.telemetry_log import TrackingDiagnosticsLogger
from noesis.telemetry.bev import BevRenderer, CalibrationSnapshot
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


_PGIE_PROFILES = ("yolo11_seg", "rfdetr_seg")
_ENV_TRUE = ("1", "true", "yes", "y", "on")
_TRACKING_MODES = ("legacy", "v3dt")


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
    repo_root = REPO_ROOT
    if value.startswith(("config/", "models/", "pipelines/")):
        return (repo_root / candidate).resolve()
    return (base_dir / candidate).resolve()


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


def _preflight_pgie_profile(profile: str, pipeline_cfg: Dict[str, Any], yaml_path: Path, logger: logging.Logger) -> None:
    if profile != "rfdetr_seg":
        return

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


def _materialize_effective_pipeline_yaml(
    base_yaml_path: Path,
    profile: str,
    logger: logging.Logger,
) -> Path:
    try:
        base_cfg = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(f"[FATAL] Unable to read DS8 pipeline YAML: {base_yaml_path} ({exc})") from exc
    if not isinstance(base_cfg, dict):
        raise SystemExit(f"[FATAL] DS8 pipeline YAML must be a mapping (got {type(base_cfg).__name__}): {base_yaml_path}")

    overlay: Dict[str, Any] = {}
    if profile == "rfdetr_seg":
        overlay = {
            "preprocess": {"config-file": "pipelines/config_preproc_rfdetr_432.ini"},
            "models": {
                "pgie": {
                    "config-file-path": "pipelines/config_infer_primary_rfdetr_seg.ini",
                    "engine": str((REPO_ROOT / "models" / "engines" / "rfdetr_seg_preview_432_b3_fp16.engine").resolve()),
                }
            },
        }

    effective_cfg = _deep_merge_dict(base_cfg, overlay)
    if not isinstance(effective_cfg, dict):
        raise SystemExit("[FATAL] Internal error: effective pipeline config is not a mapping")

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

    out_dir = (REPO_ROOT / "build").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"effective_pipeline_{profile}.yaml"
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
    parser = argparse.ArgumentParser(description="Noesis DS8 runtime harness")

    default_pgie_profile = str(os.environ.get("NOESIS_PGIE_PROFILE", "yolo11_seg") or "").strip() or "yolo11_seg"

    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help="Path to the DS8 pipeline YAML definition.",
    )
    parser.add_argument(
        "--pgie-profile",
        choices=_PGIE_PROFILES,
        default=default_pgie_profile,
        help="PGIE profile overlay (default: yolo11_seg). Env: NOESIS_PGIE_PROFILE",
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
        help="Tracking mode selection (legacy or v3dt). Env: NOESIS_TRACKING_MODE",
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
        help="Start the DS8 FastAPI application (depth + analytics). Enabled by default.",
    )
    rest_group.add_argument(
        "--disable-rest",
        dest="enable_rest",
        action="store_false",
        help="Disable the DS8 FastAPI application (depth + analytics).",
    )
    parser.add_argument(
        "--storage-base",
        type=Path,
        default=None,
        help="Override MapAnything snapshot base directory.",
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
    if mode in ("legacy", "2d", "baseline", "standard", "default"):
        return "legacy"
    if not mode or mode == "auto":
        return "legacy"
    logging.getLogger("ds8.runtime").warning("Unknown tracking mode '%s'; defaulting to legacy", value)
    return "legacy"


def _resolve_tracking_mode(args: argparse.Namespace) -> str:
    if args.tracking_mode:
        return _normalize_tracking_mode(args.tracking_mode)
    if getattr(args, "v3dt", False):
        return "v3dt"
    env_mode = os.environ.get("NOESIS_TRACKING_MODE", "")
    if str(env_mode).strip():
        return _normalize_tracking_mode(env_mode)
    return "legacy"


def _mode_default_paths(mode: str) -> Tuple[Path, Path]:
    if mode == "v3dt":
        return (
            REPO_ROOT / "config" / "infer_v3dt_baseline.yaml",
            REPO_ROOT / "config" / "cameras_v3dt_baseline.yaml",
        )
    return (REPO_ROOT / "config" / "infer.yaml", REPO_ROOT / "config" / "cameras.yaml")


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
        env_pipeline = os.environ.get("NOESIS_DS8_PIPELINE_CONFIG", "")
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


def _warn_legacy_with_v3dt_tracker(pipeline_path: Path, logger: logging.Logger) -> None:
    pipeline_cfg = _load_pipeline_config(pipeline_path, logger)
    if pipeline_cfg is None:
        return
    tracker_path = _resolve_tracker_config_path(pipeline_cfg, pipeline_path)
    if tracker_path is None:
        return
    if _tracker_under_v3dt_dir(tracker_path):
        logger.warning(
            "Tracking mode 'legacy' with V3DT tracker config %s; V3DT meta/bbox3d will be ignored",
            tracker_path,
        )


def _ensure_v3dt_meta_extension(logger: logging.Logger) -> bool:
    try:
        import noesis_v3dt_meta_ext  # type: ignore  # noqa: F401
    except Exception as exc:
        logger.error("Tracking mode 'v3dt' requires noesis_v3dt_meta_ext; import failed: %s", exc)
        return False
    return True


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


def _build_stable_id_manager(logger: logging.Logger):
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
        embed_interval_s = float(os.environ.get("NOESIS_REID_EMBED_INTERVAL_S", "0.0") or 0.0)
        new_id_hysteresis_frames = int(os.environ.get("NOESIS_REID_NEW_ID_HYSTERESIS_FRAMES", "1") or 1)
        new_id_confirm_frames_at_cap = int(os.environ.get("NOESIS_REID_NEW_ID_CONFIRM_FRAMES_AT_CAP", "1") or 1)
        mgr = StableIDManager(
            model_path=model_path,
            device=device,
            model_name=model_name,
            image_size=(img_h, img_w),
            allow_multi_zone_active=True,
            # DS8 stable IDs source embeddings from an explicit OSNet SGIE; do not load torchreid.
            use_extractor=False,
            embed_interval_s=embed_interval_s,
            new_id_hysteresis_frames=new_id_hysteresis_frames,
            new_id_confirm_frames_at_cap=new_id_confirm_frames_at_cap,
        )
        logger.info(
            "Stable ID manager initialised (SGIE embeddings; allow_multi_zone_active=%s, embed_interval_s=%.3f, new_id_hysteresis_frames=%d, new_id_confirm_frames_at_cap=%d)",
            True,
            embed_interval_s,
            new_id_hysteresis_frames,
            new_id_confirm_frames_at_cap,
        )
        return mgr
    except Exception as exc:
        logger.warning("Stable ID manager init failed; continuing without stable IDs: %s", exc)
        return None


class _CalibrationProvider:
    """Provide calibration snapshots and WS bundle for BEV rendering."""

    def __init__(self, cameras_path: Path, pipeline_cfg: Dict[str, object]) -> None:
        self._loader = CameraConfigLoader(cameras_path)
        self._camera_model_res = self._load_camera_model_resolutions(cameras_path)
        self._intrinsics_models = load_intrinsics(str(REPO_ROOT / "intrinsics.json"))
        self._align = load_alignment(str(REPO_ROOT / "config" / "ply_alignment.json"))
        self._extrinsics_path = REPO_ROOT / "config" / "camera_calibration.json"
        self._extrinsics = load_extrinsics(str(self._extrinsics_path))
        try:
            from config import config as legacy_config  # type: ignore

            calib_cfg = getattr(legacy_config, "calibration", None)
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

    def reload_extrinsics(self) -> None:
        """Reload extrinsics from camera_calibration.json without touching alignment."""
        self._extrinsics = load_extrinsics(str(self._extrinsics_path))
        self._bundle_cache = None

    def reload_alignment(self) -> None:
        """Reload alignment from ply_alignment.json."""
        self._align = load_alignment(str(REPO_ROOT / "config" / "ply_alignment.json"))
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
        try:
            unit_scale = float((align_dict.get("units") or {}).get("s_obj_to_m", 1.0))
        except Exception:
            unit_scale = 1.0
        frame_w, frame_h = self._frame_size
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 1920, 1080

        # Align intrinsics with the current streammux resolution (mirror DS7 scaling rules).
        base_w = base_h = None
        res = self._camera_model_res.get(camera_id)
        if res:
            base_w, base_h = res
        try:
            spec = (self._camera_specs or {}).get(camera_id) if isinstance(self._camera_specs, dict) else None
            if isinstance(spec, dict):
                res = spec.get("resolution")
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    base_w = int(res[0]) or None
                    base_h = int(res[1]) or None
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
) -> Callable[[], Dict[str, object]]:
    start_time = time.time()

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
        now = time.time()
        try:
            depth_fps = pipeline.depth_fps()
        except Exception:
            depth_fps = 0.0
        reload_count = getattr(pipeline, "analytics_reload_count", 0)
        cameras_stats: Dict[str, object] = {}

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

            cameras_stats[cam_key] = {
                "fps": 0.0,
                "frames_processed": 0,
                "status": "running" if pipeline.activated else "unknown",
                "tracking": tracking,
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
                "analytics_reload_count": reload_count,
                "mosaic_layout": _mosaic_layout(),
                "errors": list(pipeline.errors),
            },
            "cameras": cameras_stats,
        }

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

    thread = threading.Thread(target=_run, name="DS8-WebSocket", daemon=True)
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
        with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "ds8_runtime.py:_wait_for_rtsp_ready",
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
                    with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds8_runtime.py:_wait_for_rtsp_ready",
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
                    with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds8_runtime.py:_wait_for_rtsp_ready",
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
        with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "ds8_runtime.py:_wait_for_rtsp_ready",
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


def _attach_mosaic_gst_appsink_handler(
    pipeline: ds8_pipeline.DS8Pipeline,
    ws_server: WebSocketServer,
    logger: logging.Logger,
) -> None:
    """Attach a GI GstAppSink new-sample handler on mosaic_appsink to forward JPEG bytes over WS."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp
    except Exception as exc:  # pragma: no cover - runtime dependency
        msg = f"Gst/GstApp unavailable; mosaic appsink handler not attached: {exc}"
        logger.error(msg)
        pipeline.errors.append(msg)
        return

    if ws_server is None:
        msg = "WebSocket server missing; mosaic appsink handler not attached"
        logger.error(msg)
        pipeline.errors.append(msg)
        return

    ds_pipeline = getattr(pipeline, "ds_pipeline", None)
    if ds_pipeline is None:
        msg = "DS8 pipeline handle unavailable; mosaic appsink handler not attached"
        logger.error(msg)
        pipeline.errors.append(msg)
        return

    # Attempt to locate the underlying Gst.Pipeline or the Gst.AppSink element.
    appsink: Optional[GstApp.AppSink] = None
    gst_pipeline = None
    # Direct attributes that may surface the Gst.Pipeline
    for attr in ("pipeline", "_pipeline", "handle", "_handle", "gst_pipeline", "_gst_pipeline"):
        try:
            candidate = getattr(ds_pipeline, attr, None)
        except Exception:
            candidate = None
        if candidate is None:
            continue
        if isinstance(candidate, Gst.Pipeline):
            gst_pipeline = candidate
            break
        if hasattr(candidate, "get_by_name") and not gst_pipeline:
            gst_pipeline = candidate  # type: ignore[assignment]
            break
    # pyservicemaker.Pipeline keeps a _instance handle; inspect it as well.
    if gst_pipeline is None:
        try:
            inner = getattr(ds_pipeline, "_instance", None)
        except Exception:
            inner = None
        if inner is not None:
            for attr in ("pipeline", "_pipeline", "handle", "_handle", "gst_pipeline", "_gst_pipeline"):
                try:
                    candidate = getattr(inner, attr, None)
                except Exception:
                    candidate = None
                if candidate is None:
                    continue
                if isinstance(candidate, Gst.Pipeline):
                    gst_pipeline = candidate
                    break
                if hasattr(candidate, "get_by_name") and not gst_pipeline:
                    gst_pipeline = candidate  # type: ignore[assignment]
                    break
            if gst_pipeline is None:
                for name in dir(inner):
                    if name.startswith("_"):
                        continue
                    try:
                        candidate = getattr(inner, name)
                    except Exception:
                        continue
                    if isinstance(candidate, Gst.Pipeline):
                        gst_pipeline = candidate
                        break
                    if hasattr(candidate, "get_by_name") and not gst_pipeline:
                        gst_pipeline = candidate  # type: ignore[assignment]
                        break

    if gst_pipeline is None:
        for name in dir(ds_pipeline):
            if name.startswith("_"):
                continue
            try:
                candidate = getattr(ds_pipeline, name)
            except Exception:
                continue
            if isinstance(candidate, Gst.Pipeline):
                gst_pipeline = candidate
                break
            if hasattr(candidate, "get_by_name") and not gst_pipeline:
                gst_pipeline = candidate  # type: ignore[assignment]
                break

    if gst_pipeline is not None:
        try:
            elem = gst_pipeline.get_by_name("mosaic_appsink")
            if elem is not None:
                if isinstance(elem, GstApp.AppSink):
                    appsink = elem
                else:
                    cast_fn = getattr(GstApp.AppSink, "cast", None)
                    if callable(cast_fn):
                        try:
                            appsink = cast_fn(elem)
                        except Exception:
                            appsink = None
                    if appsink is None and isinstance(elem, Gst.Element):
                        appsink = elem  # type: ignore[assignment]
        except Exception:
            logger.debug("Failed to get mosaic_appsink via Gst.Pipeline", exc_info=True)

    # If pipeline handle lookup failed, try digging into the pyservicemaker node for a Gst element reference.
    if appsink is None:
        try:
            node = ds_pipeline["mosaic_appsink"]
        except Exception as exc:
            msg = f"mosaic_appsink component not found; handler not attached: {exc}"
            logger.error(msg, exc_info=True)
            pipeline.errors.append(msg)
            return

        for attr in (
            "element",
            "_element",
            "obj",
            "_obj",
            "gst_element",
            "_gst_element",
            "_gst",
            "handle",
            "_handle",
            "__gstelement__",
            "__gst_element__",
        ):
            try:
                candidate = getattr(node, attr, None)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            if isinstance(candidate, GstApp.AppSink):
                appsink = candidate
                break
            if isinstance(candidate, Gst.Element):
                appsink = candidate  # type: ignore[assignment]
                break
        if appsink is None:
            for name in dir(node):
                if name.startswith("_"):
                    continue
                try:
                    candidate = getattr(node, name)
                except Exception:
                    continue
                if isinstance(candidate, GstApp.AppSink):
                    appsink = candidate
                    break
                if isinstance(candidate, Gst.Element):
                    appsink = candidate  # type: ignore[assignment]
                    break

    if appsink is None:
        msg = "GstApp.AppSink handle not found for mosaic_appsink; handler not attached"
        logger.error(msg)
        pipeline.errors.append(msg)
        return

    try:
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
    except Exception:  # pragma: no cover - defensive; properties may already be set
        logger.warning("Failed to set mosaic_appsink properties", exc_info=True)

    state = {"count": 0, "debug_written": False}

    def _on_new_sample(sink: GstApp.AppSink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            logger.warning("mosaic_appsink: pull-sample returned None")
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        if buf is None:
            logger.warning("mosaic_appsink: sample has no buffer")
            return Gst.FlowReturn.ERROR

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            logger.warning("mosaic_appsink: GstBuffer.map() failed")
            return Gst.FlowReturn.ERROR

        try:
            payload = bytes(map_info.data) if map_info.data else b""
        finally:
            buf.unmap(map_info)

        if not payload:
            logger.warning("mosaic_appsink: empty JPEG payload")
            return Gst.FlowReturn.OK

        header = b"living-room"
        if len(header) > 255:
            header = header[:255]
        framed = bytes([len(header)]) + header + payload

        if not state["debug_written"]:
            debug_path = "/tmp/noesis_mosaic_test.jpg"
            try:
                with open(debug_path, "wb") as f:
                    f.write(payload)
                logger.info(
                    "mosaic_appsink: wrote first JPEG (%d bytes) to %s",
                    len(payload),
                    debug_path,
                )
                state["debug_written"] = True
            except Exception:
                logger.debug("mosaic_appsink: failed to write debug JPEG", exc_info=True)

        ws_server.broadcast_sync(framed)

        state["count"] += 1
        if state["count"] <= 3 or state["count"] % 30 == 0:
            #region agent log
            try:
                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                                "location": "ds8_runtime.py:_attach_mosaic_gst_appsink_handler",
                                "message": "mosaic appsink frame",
                                "data": {"count": state["count"], "bytes": len(payload)},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
        return Gst.FlowReturn.OK

    appsink.connect("new-sample", _on_new_sample)
    logger.info("Attached GstAppSink new-sample handler on 'mosaic_appsink'")


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
    from noesis.server import analytics_api, depth_api

    app = FastAPI(title="Noesis DS8 Runtime API")
    origins_env = os.environ.get("NOESIS_REST_CORS_ORIGINS", "").strip()
    allow_all = os.environ.get("NOESIS_REST_CORS_ALLOW_ALL", "").strip().lower() in {"1", "true", "yes", "on"}
    origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()] if origins_env else []
    if allow_all and "*" not in origins:
        origins = ["*"]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(depth_api.app.router)
    app.include_router(analytics_api.app.router)
    return app


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

    thread = threading.Thread(target=_run, name="DS8-REST", daemon=True)
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
        logger.warning("DS8 pipeline handle unavailable; WebRTC signaling not attached")
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
                    with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds8_runtime.py:_setup_webrtc_signaling",
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
    """Best-effort keyframe/IDR request into the DS8 RTSP encoder pipeline.

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
            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "ds8_runtime.py:_build_rtsp_keyframe_requester",
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
            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "ds8_runtime.py:_on_bus_message",
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
            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "ds8_runtime.py:_on_bus_message",
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
            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "ds8_runtime.py:_on_bus_message",
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
                    with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "ds8_runtime.py:_on_bus_message",
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
    
    NOTE: This function is kept for backwards compatibility but pyservicemaker
    does not expose the underlying Gst.Pipeline, so we use its native wait() instead.
    """
    if not _GLIB_AVAILABLE or GLib is None or Gst is None:
        logger.warning("GLib unavailable; skipping main loop (may affect stream reconnection)")
        return None, None

    # pyservicemaker doesn't expose the underlying Gst.Pipeline
    # Instead, we rely on its native event handling via wait()
    # This function exists for compatibility but returns None - see _start_pyservicemaker_wait_loop
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
    log_state = str(os.environ.get("NOESIS_DS8_STATE_LOG", "")).strip().lower() in ("1", "true", "yes", "on")

    if isinstance(message, EOSMessage):
        logger.warning("⚠️ EOS received on pipeline (unexpected for live sources)")
        if state is not None:
            state["pipeline_failed"] = True
    elif isinstance(message, StateTransitionMessage):
        if log_state:
            # Avoid touching message attributes unless explicitly enabled.
            logger.info("DS8 state transition (origin=%s)", getattr(message, "origin", "unknown"))
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
                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds8_runtime.py:_start_pyservicemaker_wait_loop",
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
                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds8_runtime.py:_start_pyservicemaker_wait_loop",
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
            if state is not None and not was_signalled:
                state["pipeline_failed"] = True
            #region agent log
            try:
                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds8_runtime.py:_start_pyservicemaker_wait_loop",
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

    thread = threading.Thread(target=_wait_loop, name="DS8-WaitLoop", daemon=True)
    thread.start()
    logger.info("pyservicemaker wait loop started for pipeline event handling")
    return thread


def main() -> int:
    os.environ.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "1")
    os.environ.setdefault("NOESIS_DEPTH_ENABLE_SECONDS", "0")
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("ds8.runtime")
    runtime_state: Dict[str, Any] = {"pipeline_failed": False}

    # Install SIGINT/SIGTERM handling early (before DS/GStreamer init), because
    # some backends install their own handlers/masks which can make `timeout(1)`
    # leave behind orphaned processes that keep ports bound.
    shutdown_event = threading.Event()

    def _signal_handler(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; initiating shutdown", signum)
        # Best-effort debug breadcrumb (useful when the backend swallows SIGTERM).
        try:
            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H0",
                            "location": "ds8_runtime.py:_signal_handler",
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
                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H0",
                                "location": "ds8_runtime.py:_signal_handler",
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
        logger.warning("Normalized tracking mode '%s' is unknown; defaulting to legacy", tracking_mode)
        tracking_mode = "legacy"

    pipeline_path, cameras_path = _resolve_pipeline_and_camera_paths(args, tracking_mode, logger)
    pipeline_path = pipeline_path.expanduser().resolve()
    cameras_path = cameras_path.expanduser().resolve()

    # Ensure process CWD is the repo root so relative paths in YAML (engines, configs)
    # resolve correctly for DS8 plugins and hooks.
    try:
        os.chdir(REPO_ROOT)
    except Exception:
        logging.getLogger("ds8.runtime").warning("Unable to chdir to REPO_ROOT %s", REPO_ROOT)

    if not pipeline_path.exists():
        logger.error("Pipeline configuration not found: %s", pipeline_path)
        return 1
    if tracking_mode == "v3dt":
        if not _ensure_v3dt_meta_extension(logger):
            return 1
        if not _validate_v3dt_tracking_guardrails(pipeline_path, logger):
            return 1
    else:
        _warn_legacy_with_v3dt_tracker(pipeline_path, logger)

    base_pipeline_path = pipeline_path
    pipeline_path = _materialize_effective_pipeline_yaml(base_pipeline_path, str(args.pgie_profile), logger)
    logger.info("Building DS8 pipeline from %s (base: %s)", pipeline_path, base_pipeline_path)

    if not _maybe_autogen_v3dt_caminfo(pipeline_path, cameras_path, logger):
        return 1
    if not _validate_dewarper_intrinsics_sync(pipeline_path, cameras_path, logger):
        return 1

    os.environ["NOESIS_DS8_PIPELINE_CONFIG"] = str(pipeline_path)
    try:
        from noesis.server import analytics_api

        os.environ.setdefault(analytics_api.ANALYTICS_CONFIG_ENV, str(REPO_ROOT / "config" / "nvdsanalytics.yaml"))
    except Exception:
        pass

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
            calib_path = str(REPO_ROOT / "config" / "camera_calibration.json")

            for entry in results or []:
                if not isinstance(entry, dict):
                    continue
                if not entry.get("ok"):
                    continue
                cam = entry.get("cameraId")
                e_mat = entry.get("E")
                if not cam or not e_mat:
                    continue
                if save_extrinsics(calib_path, cam, e_mat):
                    updated.append(cam)
                else:
                    persist_failed = True

            top_error = res.get("error") if isinstance(res, dict) else None
            if not updated and persist_failed and not top_error:
                top_error = "persist_failed"

            if updated:
                calibration_provider.reload_extrinsics()
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
        enable_env = os.environ.get("NOESIS_DEPTH_RPC_ENABLE_SECONDS", "2")
        try:
            enable_seconds = int(str(enable_env).strip())
        except Exception:
            enable_seconds = 2
        enable_seconds = max(1, min(20, enable_seconds))
        try:
            ds8_pipeline.enable_depth(seconds=enable_seconds)
        except Exception as exc:
            if cached is not None:
                return _response(True, payload=cached, ts_us=cached_ts, error=str(exc) or "depth_enable_failed")
            return _response(False, error=str(exc) or "depth_enable_failed")

        wait_timeout_s = min(3.0, float(enable_seconds) + 1.0)
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
        try:
            return storage_manager.generate_topdown_floorplan(
                camera_id,
                max_age_sec=max_age_sec,
                grid_res_m=grid_res_m,
                max_extent_m=max_extent_m,
                cache_only=cache_only,
            )
        except Exception as exc:
            for alt_key in alt_keys:
                try:
                    return storage_manager.generate_topdown_floorplan(
                        alt_key,
                        max_age_sec=max_age_sec,
                        grid_res_m=grid_res_m,
                        max_extent_m=max_extent_m,
                        cache_only=cache_only,
                    )
                except Exception:
                    continue
            logger.warning(
                "DS8 floorplan provider failed (camera=%s, cache_only=%s): %s",
                camera_id,
                cache_only,
                exc,
            )
            return {"error": str(exc) or "floorplan_failed", "camera_id": camera_id}

    pipeline = ds8_pipeline.build_pipeline(pipeline_path)
    setattr(pipeline, "camera_labels", camera_labels)
    if getattr(pipeline, "ds_pipeline", None) is None:
        logger.error("pyservicemaker unavailable; DS8 runtime cannot continue")
        return 1
    try:
        from noesis.server import analytics_api

        analytics_cfg = analytics_api._load_config(force=True)  # type: ignore[attr-defined]
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

    nvjpeg_built = "mosaic_appsink" in getattr(pipeline, "components", {})
    rtsp_built = "rtsp_out" in getattr(pipeline, "components", {})

    logger.info(
        "Mosaic output toggles (effective): JPEG=%s (built=%s), RTSP=%s (built=%s), WebRTC_Gateway=%s",
        bool(mosaic_cfg.get("jpeg_enabled", False)),
        nvjpeg_built,
        bool(mosaic_cfg.get("rtsp_enabled", False)),
        rtsp_built,
        mosaic_webrtc_enabled,
    )
    logger.info("NVJPEG mosaic branch %s (appsink_built=%s)", "ENABLED" if nvjpeg_built else "DISABLED", nvjpeg_built)
    #region agent log
    try:
        with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "ds8_runtime.py:main",
                        "message": "mosaic toggles",
                        "data": {
                            "jpeg_enabled": bool(mosaic_cfg.get("jpeg_enabled", False)),
                            "rtsp_enabled": bool(mosaic_cfg.get("rtsp_enabled", False)),
                            "rtsp_port": rtsp_port,
                            "rtsp_path": rtsp_path,
                            "webrtc_enabled": mosaic_webrtc_enabled,
                            "nvjpeg_built": nvjpeg_built,
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

    calibration_provider = _CalibrationProvider(cameras_path, pipeline.config)
    calibration_provider.set_camera_labels(camera_labels)
    try:
        storage_manager.calibration_bundle = calibration_provider.calibration_bundle()
    except Exception:
        logger.debug("Unable to seed calibration bundle on storage manager", exc_info=True)
    stable_id_mgr = _build_stable_id_manager(logger)
    # Ensure occupancy publisher slot exists for telemetry hooks; real publisher can be bound later.
    bind_occupancy_publisher(pipeline, None)
    # Stable ID manager is optional; attach slot so hooks can discover it.
    setattr(pipeline, "stable_id_mgr", stable_id_mgr)
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
        logger.exception("Failed to attach DS8 trail overlay hook")

    bev_cfg = pipeline.config.get("bev") or {}
    bev_jpeg_enabled = bool(bev_cfg.get("jpeg_enabled", False))
    bev_jpeg_quality = int(bev_cfg.get("jpeg_quality", 70) or 70)
    bev_smoothing_cfg = bev_cfg.get("smoothing") if isinstance(bev_cfg, dict) else None
    if not isinstance(bev_smoothing_cfg, dict):
        bev_smoothing_cfg = None
    bev_env = os.environ.get("NOESIS_BEV_JPEG_ENABLED")
    if bev_env is not None:
        env_text = str(bev_env).strip().lower()
        if env_text in ("1", "true", "yes", "on"):
            bev_jpeg_enabled = True
        elif env_text in ("0", "false", "no", "off"):
            bev_jpeg_enabled = False
    logger.info("BEV JPEG output enabled=%s (quality=%s)", bev_jpeg_enabled, bev_jpeg_quality)

    ws_server = WebSocketServer(
        host=args.ws_host,
        port=args.ws_port,
        stats_callback=_build_stats_callback(pipeline, camera_labels),
        initial_trail_state=bool(trail_settings.enabled),
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

    def _maybe_coerce_extrinsics_translation_to_meters(E_col_major: list[float]) -> tuple[list[float], Optional[str]]:
        mode = str(os.environ.get("NOESIS_EXTRINSICS_INPUT_UNITS", "auto") or "").strip().lower()
        if mode in ("m", "meter", "meters"):
            return E_col_major, None
        if mode in ("cm", "centimeter", "centimeters"):
            note = "cm→m (NOESIS_EXTRINSICS_INPUT_UNITS=cm)"
            try:
                Emat = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
                Emat[:3, 3] *= 0.01
                return list(Emat.flatten(order="F")), note
            except Exception:
                return E_col_major, note

        # auto: camera height in home scenes should not be tens/hundreds of meters.
        try:
            Emat = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
            Twc = np.linalg.inv(Emat)
            C_y = float(Twc[1, 3])
            if abs(C_y) > 20.0 and abs(C_y / 100.0) < 20.0:
                Emat[:3, 3] *= 0.01
                return list(Emat.flatten(order="F")), f"cm→m (auto; |C_y|={abs(C_y):.3f} too large for meters)"
        except Exception:
            pass
        return E_col_major, None

    def _set_extrinsics_handler(req: Dict[str, Any]) -> Dict[str, Any]:
        cam_id = _resolve_ws_camera_id(req.get("cameraId") or req.get("camera") or req.get("camId") or req.get("id"))
        if not cam_id:
            return {"ok": False, "error": "cameraId_required"}

        E: Optional[list[float]] = None
        try:
            if isinstance(req.get("E"), list) and len(req["E"]) == 16:
                E = [float(x) for x in req["E"]]
            elif isinstance(req.get("Twc"), list) and len(req["Twc"]) == 16:
                Twc = np.array(req["Twc"], dtype=np.float64).reshape((4, 4), order="F")
                Emat = np.linalg.inv(Twc)
                E = list(Emat.flatten(order="F"))
            else:
                return {"ok": False, "error": "E_or_Twc_required"}
        except Exception as exc:
            return {"ok": False, "error": f"parse_error: {exc}"}

        try:
            sid = _camera_to_source_id(cam_id)
            sid_text = "?" if sid is None else str(sid)
            logger.warning("WS RX set_extrinsics camera=%s sid=%s E_col_major=%s", cam_id, sid_text, E)
            try:
                with np.printoptions(precision=6, suppress=True, linewidth=200):
                    Emat = np.array(E, dtype=np.float64).reshape((4, 4), order="F")
                    logger.warning("WS RX set_extrinsics camera=%s E_matrix=%s", cam_id, str(Emat))
            except Exception:
                pass
        except Exception:
            pass

        E, units_note = _maybe_coerce_extrinsics_translation_to_meters(E)
        if units_note:
            logger.warning("set_extrinsics: coerced translation units for %s: %s", cam_id, units_note)
            try:
                logger.warning("WS RX set_extrinsics camera=%s E_col_major_coerced=%s", cam_id, E)
                with np.printoptions(precision=6, suppress=True, linewidth=200):
                    Emat = np.array(E, dtype=np.float64).reshape((4, 4), order="F")
                    logger.warning("WS RX set_extrinsics camera=%s E_matrix_coerced=%s", cam_id, str(Emat))
            except Exception:
                pass

        try:
            E_matrix = np.array(E, dtype=np.float64).reshape((4, 4), order="F")
            if np.allclose(E_matrix, np.eye(4), atol=1e-3):
                return {"ok": False, "error": "calibration_invalid_identity"}
        except Exception:
            return {"ok": False, "error": "bad_extrinsics"}

        calib_path = str(REPO_ROOT / "config" / "camera_calibration.json")
        if not save_extrinsics(calib_path, cam_id, E):
            return {"ok": False, "error": "persist_failed"}

        logger.warning("WS set_extrinsics persisted camera=%s path=%s", cam_id, calib_path)

        calibration_provider.reload_extrinsics()
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

        align_path = str(REPO_ROOT / "config" / "ply_alignment.json")
        if not save_alignment(align_path, align_update):
            return {"ok": False, "error": "persist_failed"}

        logger.warning("WS set_align persisted path=%s", align_path)

        calibration_provider.reload_alignment()
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

    setattr(pipeline, "ws_server", ws_server)
    bev_renderer = BevRenderer(
        ws_server,
        trails_cfg=trails_cfg,
        smoothing_cfg=bev_smoothing_cfg,
        jpeg_enabled=bev_jpeg_enabled,
        jpeg_quality=bev_jpeg_quality,
    )
    ws_server.bev_config_callback = lambda cam_id, cfg: bev_renderer.update_config(cam_id, cfg)
    ws_server.bev_overlay_callback = lambda cam_id, enabled: bev_renderer.update_config(cam_id, {"overlay": enabled})
    depth_pub = DepthTelemetryPublisher(ws_server)
    tracking_pub = TrackingTelemetryPublisher(ws_server)
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
    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=tracking_pub,
        tracking_mode=tracking_mode,
        camera_labels=camera_labels,
        bev_renderer=bev_renderer,
        bev_calibration=calibration_provider,
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

    logger.info("Preparing DS8 pipeline")
    if not ds8_pipeline.prepare(on_message=_psm_message_cb):
        logger.error("DS8 pipeline preparation failed: %s", pipeline.errors)
        return 1

    # Attach mosaic appsink GI handler after prepare and before activation.
    # Only attach if JPEG mosaic path is enabled (appsink only exists when jpeg_enabled)
    if nvjpeg_built:
        _attach_mosaic_gst_appsink_handler(pipeline, ws_server, logger)
        if pipeline.errors:
            # Non-fatal if RTSP/WebRTC path is available as alternative
            if rtsp_built:
                logger.warning("Mosaic appsink handler failed; RTSP path available as alternative. Errors: %s", pipeline.errors)
                pipeline.errors.clear()  # Clear errors since RTSP path is available
            else:
                logger.error("Mosaic appsink handler failed; errors: %s", pipeline.errors)
                return 1
    else:
        logger.info("JPEG mosaic disabled; skipping appsink handler attachment")

    ws_thread, ws_loop = _start_websocket_server(ws_server)
    if getattr(ws_server, "server", None) is None:
        logger.error("WebSocket server failed to start; aborting DS8 runtime")
        return 1

    # Activate the DS8 pipeline after prepare() using activate() not start()
    # NOTE: We use activate() because prepare() was already called above.
    # Using start() after prepare() causes "Tried to add new watch while one was already there"
    # because start() internally calls prepare() + sets bus watch, conflicting with existing watch.
    if not ds8_pipeline.activate():
        logger.error("DS8 pipeline activation failed: %s", pipeline.errors)
        return 1
    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        logger.error("DS8 pipeline activated but ds_pipeline is missing")
        return 1
    logger.info("DS8 pipeline activated successfully")

    # Start pyservicemaker wait loop to keep pipeline alive and processing events
    # This is critical - without wait(), the pipeline may stop after initial buffers
    wait_thread = _start_pyservicemaker_wait_loop(ds, shutdown_event, logger, runtime_state)

    # Start WebRTC gateway if enabled (requires RTSP output)
    webrtc_gateway = None
    if mosaic_webrtc_enabled:
        if rtsp_built:
            try:
                ready = _wait_for_rtsp_ready("127.0.0.1", rtsp_port, timeout=15.0, interval=0.2)
                if not ready:
                    logger.error("RTSP sink not ready on 127.0.0.1:%s; skipping WebRTC gateway start", rtsp_port)
                    #region agent log
                    try:
                        with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H2",
                                        "location": "ds8_runtime.py:main",
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
                    rtsp_keyframe_requester = _build_rtsp_keyframe_requester(pipeline, logger)
                    if rtsp_keyframe_requester is None:
                        logger.debug("RTSP keyframe requester unavailable; falling back to natural IDR cadence")
                    webrtc_gateway = MosaicWebRTCGateway(
                        ws_server=ws_server,
                        rtsp_uri=rtsp_uri,
                        request_rtsp_keyframe=rtsp_keyframe_requester,
                    )
                    webrtc_gateway.start()
                    logger.info("WebRTC gateway started, consuming RTSP at %s", rtsp_uri)
                    #region agent log
                    try:
                        with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H4",
                                        "location": "ds8_runtime.py:main",
                                        "message": "gateway started",
                                        "data": {"rtsp_uri": rtsp_uri},
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
        logger.warning("DS8 pipeline not activated; check pipeline.errors for details: %s", pipeline.errors)

    rest_server = None
    rest_thread = None

    if args.enable_rest:
        try:
            from noesis.server import analytics_api

            analytics_cfg_path = REPO_ROOT / "config" / "nvdsanalytics.yaml"
            os.environ.setdefault(analytics_api.ANALYTICS_CONFIG_ENV, str(analytics_cfg_path))
        except Exception:
            pass
        rest_app = _build_rest_app()
        rest_server, rest_thread = _start_rest_server(rest_app, args.rest_host, args.rest_port)
        if rest_server:
            logger.info("REST server listening on http://%s:%s", args.rest_host, args.rest_port)

    # Re-assert SIGTERM default behavior after DS/GStreamer initialization.
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except Exception:
        pass

    logger.info("DS8 runtime is active. Press Ctrl+C to stop.")

    # Spawn a heartbeat logger to confirm liveness while waiting/processing
    def _heartbeat() -> None:
        for i in range(6):
            time.sleep(3)
            try:
                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "ds8_runtime.py:main",
                                "message": "heartbeat",
                                "data": {"tick": i + 1},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass

    threading.Thread(target=_heartbeat, name="DS8-Heartbeat", daemon=True).start()
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()

    logger.info("Shutting down DS8 runtime")
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

    # Stop WebRTC gateway if running
    if webrtc_gateway is not None:
        try:
            webrtc_gateway.stop()
            logger.info("WebRTC gateway stopped")
        except Exception:
            logger.exception("Error stopping WebRTC gateway")

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

        threading.Thread(target=_stop_psm, name="DS8-StopPipeline", daemon=True).start()
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
