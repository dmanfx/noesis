#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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

from calibration_bundle import assemble_calibration_bundle, load_alignment, load_extrinsics, load_intrinsics
from geometry.depth_source import DepthStorageManager
from mapanything_config import load_service_config
from noesis.pipelines import ds8_pipeline, hooks
from noesis.metadata.intrinsics import CameraConfigLoader
from noesis.telemetry.publishers import DepthTelemetryPublisher, TrackingTelemetryPublisher, bind_occupancy_publisher
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noesis DS8 runtime harness")

    default_pipeline = Path(os.environ.get("NOESIS_DS8_PIPELINE_CONFIG", REPO_ROOT / "config" / "infer.yaml"))
    default_cameras = Path(os.environ.get("NOESIS_CAMERAS_CONFIG", REPO_ROOT / "config" / "cameras.yaml"))

    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=default_pipeline,
        help="Path to the DS8 pipeline YAML definition.",
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=default_cameras,
        help="Path to the cameras YAML used for intrinsics.",
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
    parser.add_argument(
        "--enable-rest",
        action="store_true",
        help="Start the DS8 FastAPI application (depth + analytics).",
    )
    parser.add_argument(
        "--storage-base",
        type=Path,
        default=None,
        help="Override MapAnything snapshot base directory.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("NOESIS_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--depth-enable-seconds",
        type=int,
        default=int(os.environ.get("NOESIS_DEPTH_ENABLE_SECONDS", "0")),
        help="Enable the MapAnything depth valve for this many seconds on startup (0 to disable).",
    )
    return parser.parse_args()


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
        mgr = StableIDManager(
            model_path=model_path,
            device=device,
            model_name=model_name,
            image_size=(img_h, img_w),
            allow_multi_zone_active=True,
        )
        logger.info("Stable ID manager initialised (device=%s, model=%s)", device, model_name)
        return mgr
    except Exception as exc:
        logger.warning("Stable ID manager init failed; continuing without stable IDs: %s", exc)
        return None


class _CalibrationProvider:
    """Provide calibration snapshots and WS bundle for BEV rendering."""

    def __init__(self, cameras_path: Path, pipeline_cfg: Dict[str, object]) -> None:
        self._loader = CameraConfigLoader(cameras_path)
        self._intrinsics_models = load_intrinsics(str(REPO_ROOT / "intrinsics.json"))
        self._align = load_alignment(str(REPO_ROOT / "config" / "ply_alignment.json"))
        self._extrinsics = load_extrinsics(str(REPO_ROOT / "config" / "camera_calibration.json"))
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

    def set_camera_labels(self, labels: Dict[int, str]) -> None:
        self._camera_labels = dict(labels or {})
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
        if not k_table:
            for src_id, cam_name in self._camera_labels.items():
                intr = self._loader.get(src_id)
                if intr is None:
                    continue
                try:
                    k_table[cam_name] = [float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)]
                except Exception:
                    continue
        self._bundle_cache = bundle
        return dict(bundle)

def _build_stats_callback(
    pipeline: ds8_pipeline.DS8Pipeline,
    camera_labels: Dict[int, str],
) -> Callable[[], Dict[str, object]]:
    start_time = time.time()

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
    from noesis.server import analytics_api, depth_api

    app = FastAPI(title="Noesis DS8 Runtime API")
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

    if isinstance(message, EOSMessage):
        logger.warning("⚠️ EOS received on pipeline (unexpected for live sources)")
        if state is not None:
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
                            "location": "ds8_runtime.py:_on_pyservicemaker_message",
                            "message": "EOS message received",
                            "data": {"type": "EOSMessage"},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion
    elif isinstance(message, StateTransitionMessage):
        logger.info(
            "🔄 Pipeline state: %s → %s (origin: %s)",
            message.old_state,
            message.new_state,
            getattr(message, 'origin', 'unknown'),
        )
        #region agent log
        try:
            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H1",
                            "location": "ds8_runtime.py:_on_pyservicemaker_message",
                            "message": "State transition",
                            "data": {
                                "old": getattr(message, "old_state", None),
                                "new": getattr(message, "new_state", None),
                                "origin": getattr(message, "origin", None),
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion
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
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
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

    pipeline_path = Path(args.pipeline_config).expanduser().resolve()
    cameras_path = Path(args.cameras_config).expanduser().resolve()

    # Ensure process CWD is the repo root so relative paths in YAML (engines, configs)
    # resolve correctly for DS8 plugins and hooks.
    try:
        os.chdir(REPO_ROOT)
    except Exception:
        logging.getLogger("ds8.runtime").warning("Unable to chdir to REPO_ROOT %s", REPO_ROOT)

    if not pipeline_path.exists():
        logger.error("Pipeline configuration not found: %s", pipeline_path)
        return 1

    os.environ.setdefault("NOESIS_DS8_PIPELINE_CONFIG", str(pipeline_path))
    try:
        from noesis.server import analytics_api

        os.environ.setdefault(analytics_api.ANALYTICS_CONFIG_ENV, str(REPO_ROOT / "config" / "nvdsanalytics.yaml"))
    except Exception:
        pass

    camera_labels = _load_camera_labels(cameras_path)
    storage_manager = _build_storage_manager(args)

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

    logger.info("Building DS8 pipeline from %s", pipeline_path)
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

    hooks.attach_intrinsics_hook(pipeline, config_path=cameras_path)

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
    setattr(pipeline, "ws_server", ws_server)
    bev_renderer = BevRenderer(ws_server, trails_cfg=trails_cfg)
    ws_server.bev_config_callback = lambda cam_id, cfg: bev_renderer.update_config(cam_id, cfg)
    ws_server.bev_overlay_callback = lambda cam_id, enabled: bev_renderer.update_config(cam_id, {"overlay": enabled})
    depth_pub = DepthTelemetryPublisher(ws_server)
    tracking_pub = TrackingTelemetryPublisher(ws_server)

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
        camera_labels=camera_labels,
        bev_renderer=bev_renderer,
        bev_calibration=calibration_provider,
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
