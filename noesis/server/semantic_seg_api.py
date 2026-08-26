from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    mark_rest_response_exempt,
    measure_rest_response_model,
)
from noesis.semantic_capture import (
    SemanticCaptureBusy,
    SemanticCaptureError,
    SemanticCaptureManager,
    SemanticCaptureTimeout,
)


ModelSize = Literal["s", "l"]
ArtifactKind = Literal["raw", "class-map", "masked"]
router = APIRouter(route_class=BoundaryMetricsRoute, tags=["semantic-seg"])
_DEFAULT_MANAGER: SemanticCaptureManager | None = None


class SemanticCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelSize


def _manager(request: Request) -> SemanticCaptureManager:
    injected = getattr(request.app.state, "semantic_capture_manager", None)
    if isinstance(injected, SemanticCaptureManager) or (
        injected is not None
        and all(callable(getattr(injected, name, None)) for name in ("capture", "latest", "artifact_path"))
    ):
        return injected
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = SemanticCaptureManager()
    return _DEFAULT_MANAGER


@router.post("/api/v1/semantic-seg/captures")
def create_semantic_capture(payload: SemanticCaptureRequest, request: Request) -> dict[str, object]:
    try:
        response = _manager(request).capture(payload.model)
    except SemanticCaptureBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SemanticCaptureTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (SemanticCaptureError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with measure_rest_response_model(
        "/api/v1/semantic-seg/captures:post", "SemanticCaptureManifest"
    ) as measurement:
        rendered = response
    mark_rest_response(
        request,
        "/api/v1/semantic-seg/captures:post",
        "SemanticCaptureManifest",
        model_duration_ms=measurement.elapsed_ms,
    )
    return rendered


@router.get("/api/v1/semantic-seg/captures/latest/{model}")
def latest_semantic_capture(model: ModelSize, request: Request) -> dict[str, object]:
    try:
        response = _manager(request).latest(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SemanticCaptureError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with measure_rest_response_model(
        "/api/v1/semantic-seg/captures/latest/{model}:get", "SemanticCaptureManifest"
    ) as measurement:
        rendered = response
    mark_rest_response(
        request,
        "/api/v1/semantic-seg/captures/latest/{model}:get",
        "SemanticCaptureManifest",
        model_duration_ms=measurement.elapsed_ms,
    )
    return rendered


@router.get("/api/v1/semantic-seg/captures/{capture_id}/{camera_id}/{artifact}")
def semantic_capture_artifact(
    capture_id: str,
    camera_id: str,
    artifact: ArtifactKind,
    request: Request,
) -> FileResponse:
    try:
        path = _manager(request).artifact_path(capture_id, camera_id, artifact)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SemanticCaptureError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    media_type = "image/png" if artifact == "class-map" else "image/jpeg"
    mark_rest_response_exempt(request, reason="semantic_capture_file_response")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, no-store"})


__all__ = ["router"]
