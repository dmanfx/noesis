from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    mark_rest_response_exempt,
    measure_rest_response_model,
)

FORCE_STUB_ENV = "NOESIS_DEPTH_API_FORCE_STUB"
_FORCE_PIPELINE_STUB = os.environ.get(FORCE_STUB_ENV, "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

try:
    if _FORCE_PIPELINE_STUB:
        raise ImportError(f"{FORCE_STUB_ENV} requested pipeline stub")
    from noesis.pipelines.deepstream_pipeline import (
        activate,
        build_pipeline,
        get_pipeline,
        prepare,
    )
    USING_PIPELINE_STUB = False
except Exception:  # pragma: no cover - explicit test-only stub path
    if not _FORCE_PIPELINE_STUB:
        raise
    USING_PIPELINE_STUB = True

    class _StubPipeline:
        """Minimal in-memory graph used when the DS9 pipeline is unavailable."""

        def __init__(self) -> None:
            self.depth_enabled: bool = False
            self.config_path: Optional[Path] = None
            self.started_at: int = 0
            self.will_disable_at: int = 0
            self._timer: Optional[threading.Timer] = None
            self._lock = threading.Lock()

        def configure(self, path: Path) -> None:
            self.config_path = Path(path)

        def enable_depth(self, seconds: int) -> Dict[str, Any]:
            interval = max(1, int(seconds))
            now = int(time.time())
            ends_at = now + interval

            def _disable() -> None:
                with self._lock:
                    self.depth_enabled = False

            with self._lock:
                if self._timer is not None:
                    try:
                        self._timer.cancel()
                    except Exception:
                        pass
                self.depth_enabled = True
                self.started_at = now
                self.will_disable_at = ends_at
                timer = threading.Timer(interval, _disable)
                timer.daemon = True
                timer.start()
                self._timer = timer

            return {
                "started_at": self.started_at,
                "will_disable_at": self.will_disable_at,
                "enabled": True,
                "seconds": interval,
            }

    _STUB_PIPELINE: Optional[_StubPipeline] = None

    def _get_stub_pipeline() -> _StubPipeline:
        global _STUB_PIPELINE
        if _STUB_PIPELINE is None:
            _STUB_PIPELINE = _StubPipeline()
        return _STUB_PIPELINE

    def enable_depth(seconds: int = 20) -> Dict[str, Any]:  # type: ignore[misc]
        return _get_stub_pipeline().enable_depth(seconds)

    def build_pipeline(path: Path) -> _StubPipeline:  # type: ignore[misc]
        pipeline = _get_stub_pipeline()
        pipeline.configure(Path(path))
        return pipeline

    def get_pipeline() -> _StubPipeline:  # type: ignore[misc]
        return _get_stub_pipeline()

    def prepare() -> bool:  # type: ignore[misc]
        _get_stub_pipeline()
        return True

    def activate() -> bool:  # type: ignore[misc]
        pipeline = _get_stub_pipeline()
        pipeline.depth_enabled = False
        return True


app = FastAPI(title="Noesis DS9 Depth API")
app.router.route_class = BoundaryMetricsRoute
logger = logging.getLogger(__name__)

PIPELINE_CONFIG_ENV = "NOESIS_DS9_PIPELINE_CONFIG"
DEFAULT_PIPELINE_CONFIG = Path(__file__).resolve().parents[2] / "DS9/config/infer.yaml"
_STUB_WARNING_EMITTED = False


class DepthRefreshResponse(BaseModel):
    started_at: int
    will_disable_at: int
    enabled: bool
    seconds: int


class _LeaseClosingStreamingResponse(StreamingResponse):
    """Close a storage-backed stream on every ASGI terminal path."""

    def __init__(
        self,
        content: Any,
        *,
        media_type: str,
        headers: Dict[str, str],
        close_stream: Callable[[], None],
    ) -> None:
        super().__init__(content, media_type=media_type, headers=headers)
        self._close_stream = close_stream

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette skips a response BackgroundTask when an ASGI 2.4 send
            # raises OSError. Close on disconnect, cancellation, and success.
            self._close_stream()


def _bulk_http_error(exc: BaseException) -> HTTPException:
    code = str(getattr(exc, "code", "") or "bulk_component_open_failed")
    status_code = {
        "bulk_component_not_found": 404,
        "bulk_snapshot_identity_mismatch": 404,
        "bulk_snapshot_role_invalid": 404,
        "bulk_snapshot_unavailable": 404,
        "bulk_component_manifest_missing": 409,
        "bulk_component_manifest_invalid": 409,
        "bulk_snapshot_integrity_failed": 409,
        "bulk_snapshot_resource_limit_exceeded": 413,
    }.get(code, 500)
    return HTTPException(status_code=status_code, detail=code)


def _resolve_pipeline_config() -> Optional[Path]:
    """Return a usable pipeline config path if it exists."""
    env_path = os.environ.get(PIPELINE_CONFIG_ENV)
    candidate = Path(env_path).expanduser() if env_path else DEFAULT_PIPELINE_CONFIG
    return candidate if candidate.exists() else None


def ensure_pipeline_ready() -> bool:
    """Ensure the DS9 pipeline graph is built and primed.

    Returns True when the pipeline is ready; False otherwise.
    """
    global _STUB_WARNING_EMITTED
    if USING_PIPELINE_STUB and not _STUB_WARNING_EMITTED:
        logger.warning(
            "Depth API running with stub pipeline (NOESIS_DEPTH_API_FORCE_STUB=%s or DS9 bindings unavailable)",
            os.environ.get(FORCE_STUB_ENV, ""),
        )
        _STUB_WARNING_EMITTED = True
    try:
        get_pipeline()
        return True
    except Exception:
        cfg_path = _resolve_pipeline_config()
        if cfg_path is None:
            logger.warning(
                "Depth pipeline config missing; depth refresh disabled "
                "(env %s, default %s)",
                PIPELINE_CONFIG_ENV,
                DEFAULT_PIPELINE_CONFIG,
            )
            return False
        try:
            build_pipeline(cfg_path)
            prepare()
            activate()
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialize DS9 pipeline: %s", exc)
            return False


@app.on_event("startup")
def _bootstrap_pipeline() -> None:  # pragma: no cover - exercised in tests
    ensure_pipeline_ready()


@app.post(
    "/api/v1/depth/refresh",
    response_model=DepthRefreshResponse,
    response_model_exclude_none=True,
)
def refresh_depth(
    request: Request,
    seconds: int = Query(20, ge=1, le=300),
) -> DepthRefreshResponse:
    if not ensure_pipeline_ready():
        raise HTTPException(status_code=503, detail="Depth pipeline not ready")

    provider = getattr(request.app.state, "depth_refresh_provider", None)
    if callable(provider):
        try:
            payload = provider(seconds)
        except Exception as exc:
            code = str(getattr(exc, "code", "") or "")
            if code == "capture_event_busy":
                raise HTTPException(status_code=409, detail=code) from exc
            if code in {
                "capture_event_cancelled",
                "capture_event_controller_unavailable",
            }:
                raise HTTPException(status_code=503, detail=code) from exc
            logger.exception("Depth refresh controller failed")
            raise HTTPException(
                status_code=500,
                detail="Depth control failed",
            ) from exc
    elif USING_PIPELINE_STUB:
        payload = enable_depth(seconds=seconds)
    else:
        raise HTTPException(
            status_code=503,
            detail="Capture-event depth controller not ready",
        )
    try:
        with measure_rest_response_model(
            "/api/v1/depth/refresh", "DepthRefreshResponse"
        ) as model_measurement:
            response = DepthRefreshResponse(
                started_at=int(payload["started_at"]),
                will_disable_at=int(payload["will_disable_at"]),
                enabled=bool(payload.get("enabled", True)),
                seconds=int(payload.get("seconds", seconds)),
            )
        mark_rest_response(
            request,
            "/api/v1/depth/refresh",
            "DepthRefreshResponse",
            model_duration_ms=model_measurement.elapsed_ms,
            include_budget=True,
        )
        return response
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Depth refresh payload malformed: %s", exc)
        raise HTTPException(status_code=500, detail="Depth control failed") from exc


@app.get(
    "/api/v1/depth/snapshots/{camera_id}/{snapshot_id}/components/{component}",
    response_class=StreamingResponse,
)
def stream_depth_snapshot_component(
    request: Request,
    camera_id: str,
    snapshot_id: str,
    component: str,
    snapshot_ref: str = Query(..., min_length=1, max_length=512),
    content_sha256: str = Query(
        ...,
        min_length=64,
        max_length=64,
        pattern="^[0-9a-f]{64}$",
    ),
) -> StreamingResponse:
    query_items = list(request.query_params.multi_items())
    query_keys = [str(key) for key, _value in query_items]
    if (
        len(query_items) != 2
        or query_keys.count("snapshot_ref") != 1
        or query_keys.count("content_sha256") != 1
    ):
        raise HTTPException(
            status_code=400,
            detail="bulk_snapshot_identity_invalid",
        )
    storage = getattr(request.app.state, "depth_storage", None)
    opener = getattr(storage, "open_depth_snapshot_component", None)
    if not callable(opener):
        raise HTTPException(
            status_code=503,
            detail="depth_bulk_storage_unavailable",
        )
    if (
        not camera_id
        or len(camera_id) > 160
        or not snapshot_id
        or len(snapshot_id) > 160
    ):
        raise HTTPException(
            status_code=400,
            detail="bulk_snapshot_identity_invalid",
        )
    try:
        stream = opener(
            camera_id=camera_id,
            storage_ref=snapshot_ref,
            snapshot_id=snapshot_id,
            content_sha256=content_sha256,
            component=component,
        )
    except Exception as exc:
        raise _bulk_http_error(exc) from exc
    try:
        descriptor = stream.descriptor
        snapshot = stream.snapshot
        response = _LeaseClosingStreamingResponse(
            stream.iter_bytes(),
            media_type="application/octet-stream",
            close_stream=stream.close,
            headers={
                "Content-Length": str(int(descriptor.byte_count)),
                "X-Noesis-Component-Sha256": str(descriptor.sha256),
                "X-Noesis-Snapshot-Id": str(snapshot.write_id),
                "Cache-Control": "no-store",
            },
        )
        mark_rest_response_exempt(
            request,
            reason="dense_depth_bulk_stream",
        )
        return response
    except Exception:
        stream.close()
        raise
