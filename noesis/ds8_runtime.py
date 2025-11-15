#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.depth_source import DepthStorageManager
from mapanything_config import load_service_config
from noesis.pipelines import ds8_pipeline, hooks
from noesis.telemetry.publishers import DepthTelemetryPublisher, TrackingTelemetryPublisher
from noesis.telemetry.bev import BevRenderer
from websocket_server import WebSocketServer
from pyservicemaker.flow import BufferRetriever, Receiver  # type: ignore
import logging as _logging


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
        default=int(os.environ.get("NOESIS_DEPTH_ENABLE_SECONDS", "120")),
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
    )


def _build_stats_callback(pipeline: ds8_pipeline.DS8Pipeline) -> Callable[[], Dict[str, object]]:
    start_time = time.time()

    def _stats() -> Dict[str, object]:
        now = time.time()
        try:
            depth_fps = pipeline.depth_fps()
        except Exception:
            depth_fps = 0.0
        return {
            "timestamp": now,
            "uptime": now - start_time,
            "pipeline": {
                "prepared": pipeline.prepared,
                "activated": pipeline.activated,
                "depth_enabled": pipeline.depth_enabled,
                "depth_fps": depth_fps,
                "errors": list(pipeline.errors),
            },
            "cameras": {},
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


class _JpegFrameRetriever(BufferRetriever):
    """Receiver for per-stream appsinks.

    Note: PyServiceMaker's Buffer only supports extracting raw surfaces via .extract().
    Since our branch ends with nvjpegenc, we cannot use .extract() to obtain encoded
    bytes directly. Instead, we rely on the C++ receiver to signal delivery and
    count frames for observability. The actual binary readout is handled by appsink
    in DS7 parity code paths.

    This retriever acts as a stub shim to validate attach/linking and can be
    extended with a native bridge if required.
    """

    def __init__(self, cam_id: str, ws: WebSocketServer, logger: _logging.Logger):
        super().__init__()
        self.cam_id = cam_id
        self.ws = ws
        self.logger = logger
        self._count = 0

    def consume(self, buffer) -> int:  # type: ignore[override]
        # We cannot extract JPEG bytes from Buffer (only raw surfaces are supported).
        # For now, emit a tiny heartbeat message so UI can confirm camera activity.
        try:
            self._count += 1
            if self._count <= 5 or (self._count % 30) == 0:
                self.logger.debug("Frame heartbeat for %s (count=%d)", self.cam_id, self._count)
            payload = {
                "type": "frame-heartbeat",
                "payload": {"cameraId": self.cam_id, "count": self._count},
            }
            self.ws.broadcast_sync(payload)
            return 0
        except Exception:
            return -1


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


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("ds8.runtime")

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

    camera_labels = _load_camera_labels(cameras_path)
    storage_manager = _build_storage_manager(args)

    logger.info("Building DS8 pipeline from %s", pipeline_path)
    pipeline = ds8_pipeline.build_pipeline(pipeline_path)
    if getattr(pipeline, "ds_pipeline", None) is None:
        logger.warning(
            "pyservicemaker not available or failed to initialize; DS8 running in dry-run (no GPU)."
        )

    hooks.attach_intrinsics_hook(pipeline, config_path=cameras_path)

    ws_server = WebSocketServer(
        host=args.ws_host,
        port=args.ws_port,
        stats_callback=_build_stats_callback(pipeline),
    )
    bev_renderer = BevRenderer(ws_server)
    ws_server.bev_config_callback = lambda cam_id, cfg: bev_renderer.update_config(cam_id, cfg)
    ws_server.bev_overlay_callback = lambda cam_id, enabled: bev_renderer.update_config(cam_id, {"overlay": enabled})
    depth_pub = DepthTelemetryPublisher(ws_server)
    tracking_pub = TrackingTelemetryPublisher(ws_server)

    # Attach MapAnything postprocess only if SGIE is present/enabled
    try:
        ma_cfg = (pipeline.config.get("models") or {}).get("mapanything") or {}
        ma_enabled = bool(ma_cfg.get("enable", True)) and any(
            key in ma_cfg for key in ("config-file-path", "engine", "name")
        )
        if ma_enabled and "mapanything_fullframe" in pipeline.components:
            hooks.attach_mapanything_postprocess_hook(
                pipeline,
                storage=storage_manager,
                depth_pub=depth_pub,
            )
        else:
            logger.info("SGIE disabled or missing; skipping MapAnything postprocess hook")
    except Exception:
        logger.exception("Error while evaluating MapAnything postprocess attachment")
    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=tracking_pub,
        camera_labels=camera_labels,
    )
    hooks.attach_exclude_prune_hook(pipeline)
    hooks.attach_analytics_reload_bridge(pipeline)

    # Tiled mode: no per-camera receivers are attached
    logger.info("DS8 tiled mode: skipping per-camera frame receivers")

    logger.info("Preparing DS8 pipeline")
    ds8_pipeline.prepare()
    ds8_pipeline.activate()

    if args.depth_enable_seconds > 0 and (pipeline.valve_name in pipeline.components):
        try:
            ds8_pipeline.enable_depth(seconds=args.depth_enable_seconds)
            logger.info(
                "Depth valve enabled for %s seconds (startup burst)",
                args.depth_enable_seconds,
            )
        except Exception:
            logger.exception("Failed to enable depth burst on startup")
    if not getattr(pipeline, "activated", False):
        logger.warning("DS8 pipeline not activated; check pipeline.errors for details: %s", pipeline.errors)

    ws_thread, ws_loop = _start_websocket_server(ws_server)
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

    shutdown_event = threading.Event()

    def _signal_handler(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; initiating shutdown", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("DS8 runtime is active. Press Ctrl+C to stop.")
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
    _stop_websocket_server(ws_server, ws_thread, ws_loop)

    logger.info("Shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
