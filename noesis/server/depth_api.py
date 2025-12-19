from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

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
    from noesis.pipelines.ds8_pipeline import (
        activate,
        build_pipeline,
        enable_depth,
        get_pipeline,
        prepare,
    )
    USING_PIPELINE_STUB = False
except Exception:  # pragma: no cover — import-safe fallback for analysis
    USING_PIPELINE_STUB = True

    class _StubPipeline:
        """Minimal in-memory graph used when the DS8 pipeline is unavailable."""

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


app = FastAPI(title="Noesis DS8 Depth API")
logger = logging.getLogger(__name__)

PIPELINE_CONFIG_ENV = "NOESIS_DS8_PIPELINE_CONFIG"
DEFAULT_PIPELINE_CONFIG = Path("config/infer.yaml")
_STUB_WARNING_EMITTED = False


class DepthRefreshResponse(BaseModel):
    started_at: int
    will_disable_at: int
    enabled: bool
    seconds: int


def _resolve_pipeline_config() -> Optional[Path]:
    """Return a usable pipeline config path if it exists."""
    env_path = os.environ.get(PIPELINE_CONFIG_ENV)
    candidate = Path(env_path).expanduser() if env_path else DEFAULT_PIPELINE_CONFIG
    return candidate if candidate.exists() else None


def ensure_pipeline_ready() -> bool:
    """Ensure the DS8 pipeline graph is built and primed.

    Returns True when the pipeline is ready; False otherwise.
    """
    global _STUB_WARNING_EMITTED
    if USING_PIPELINE_STUB and not _STUB_WARNING_EMITTED:
        logger.warning(
            "Depth API running with stub pipeline (NOESIS_DEPTH_API_FORCE_STUB=%s or DS8 libs unavailable)",
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
            logger.exception("Failed to initialize DS8 pipeline: %s", exc)
            return False


@app.on_event("startup")
def _bootstrap_pipeline() -> None:  # pragma: no cover - exercised in tests
    ensure_pipeline_ready()


@app.get(
    "/api/v1/depth/refresh",
    response_model=DepthRefreshResponse,
    response_model_exclude_none=True,
)
def refresh_depth(seconds: int = Query(20, ge=1, le=300)) -> DepthRefreshResponse:
    if not ensure_pipeline_ready():
        raise HTTPException(status_code=503, detail="Depth pipeline not ready")

    payload = enable_depth(seconds=seconds)
    try:
        return DepthRefreshResponse(
            started_at=int(payload["started_at"]),
            will_disable_at=int(payload["will_disable_at"]),
            enabled=bool(payload.get("enabled", True)),
            seconds=int(payload.get("seconds", seconds)),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Depth refresh payload malformed: %s", exc)
        raise HTTPException(status_code=500, detail="Depth control failed") from exc
