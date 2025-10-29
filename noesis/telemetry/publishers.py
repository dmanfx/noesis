from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional

from noesis.metadata.depth_result import DepthResult

try:
    from models import convert_numpy_types
except Exception:  # pragma: no cover - fallback when frontend utilities absent
    def convert_numpy_types(payload: Any) -> Any:  # type: ignore[override]
        return payload

logger = logging.getLogger(__name__)


class DepthTelemetryPublisher:
    """Bridge depth bursts to websocket clients."""

    def __init__(self, ws_server: Any) -> None:
        self._ws = ws_server

    def publish(self, result: DepthResult) -> None:
        message = {
            "type": "depth_result",
            "payload": result.to_dict(),
        }
        try:
            if hasattr(self._ws, "broadcast_sync"):
                self._ws.broadcast_sync(message)
            elif hasattr(self._ws, "broadcast"):
                # best effort async fallback
                self._ws.broadcast(message)  # type: ignore[call-arg]
            else:  # pragma: no cover - defensive
                raise AttributeError("WebSocket server has no broadcast interface")
        except Exception:
            logger.exception("Failed to publish DepthResult telemetry")


class TrackingTelemetryPublisher:
    """Broadcast tracking metadata (tracks, occupancy, transitions) to clients."""

    def __init__(self, ws_server: Any) -> None:
        self._ws = ws_server

    def publish(self, source_id: int, tracks: Iterable[Mapping[str, Any]]) -> None:
        payload = {
            "type": "tracking",
            "source_id": int(source_id),
            "tracks": convert_numpy_types(list(tracks)),
        }
        try:
            if hasattr(self._ws, "broadcast_sync"):
                self._ws.broadcast_sync(payload)
            else:  # pragma: no cover - defensive
                self._ws.broadcast(payload)  # type: ignore[call-arg]
        except Exception:
            logger.exception("Failed to publish tracking telemetry for source %s", source_id)


def bind_occupancy_publisher(pipeline: Any, occupancy_publisher: Optional[Any]) -> None:
    """Expose occupancy publisher to DS8 pipeline probes."""
    setattr(pipeline, "occupancy_publisher", occupancy_publisher)
    logger.info(
        "Occupancy publisher %s bound to pipeline",
        type(occupancy_publisher).__name__ if occupancy_publisher else "None",
    )

