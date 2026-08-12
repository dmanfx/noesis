from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Path, Request, Response

from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    mark_rest_response_exempt,
    measure_rest_response_model,
)
from noesis_core.scene_prior import ScenePriorError, ScenePriorSet


app = FastAPI(title="Noesis Scene Prior API")
app.router.route_class = BoundaryMetricsRoute


def _runtime_components(request: Request) -> tuple[ScenePriorSet, Any, Mapping[int, str]]:
    provider = getattr(request.app.state, "scene_prior_runtime_provider", None)
    if callable(provider):
        supplied = provider()
        if not isinstance(supplied, tuple) or len(supplied) != 3:
            raise HTTPException(status_code=503, detail="scene_prior_runtime_unavailable")
        priors, calibration, camera_labels = supplied
    else:
        try:
            from noesis.pipelines.ds8_pipeline import get_pipeline

            pipeline = get_pipeline()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="scene_prior_runtime_unavailable",
            ) from exc
        priors = getattr(pipeline, "scene_priors", None)
        calibration = getattr(pipeline, "bev_calibration", None)
        camera_labels = getattr(pipeline, "camera_labels", None)
    if not isinstance(priors, ScenePriorSet):
        raise HTTPException(status_code=404, detail="scene_prior_not_configured")
    if calibration is None or not isinstance(camera_labels, Mapping):
        raise HTTPException(status_code=503, detail="scene_prior_calibration_unavailable")
    return priors, calibration, camera_labels


def _camera_metadata(request: Request, camera_id: str) -> dict[str, Any]:
    priors, calibration, camera_labels = _runtime_components(request)
    source_id = next(
        (
            int(candidate)
            for candidate, label in camera_labels.items()
            if str(label) == camera_id
        ),
        None,
    )
    if source_id is None:
        raise HTTPException(status_code=404, detail="scene_prior_camera_not_active")
    try:
        snapshot = calibration.snapshot(source_id, camera_id)
        extrinsics = getattr(snapshot, "extrinsics_col_major", None)
        if extrinsics is None:
            raise ScenePriorError(
                f"scene-prior camera {camera_id!r} has no calibrated extrinsics"
            )
        metadata = priors.camera_view_metadata(
            camera_id,
            extrinsics_col_major=extrinsics,
        )
    except ScenePriorError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    encoded_camera = quote(camera_id, safe="")
    metadata["artifact_urls"] = {
        "points_glb": (
            f"/api/v1/scene-priors/cameras/{encoded_camera}/artifacts/points_glb"
        )
    }
    return metadata


@app.get("/api/v1/scene-priors/cameras/{camera_id}")
def get_scene_prior_camera(
    request: Request,
    camera_id: str = Path(
        ...,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    ),
) -> dict[str, Any]:
    metadata = _camera_metadata(request, camera_id)
    with measure_rest_response_model(
        "/api/v1/scene-priors/cameras/{camera_id}:get",
        "ScenePriorCameraView",
    ) as model_measurement:
        response = metadata
    mark_rest_response(
        request,
        "/api/v1/scene-priors/cameras/{camera_id}:get",
        "ScenePriorCameraView",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@app.get(
    "/api/v1/scene-priors/cameras/{camera_id}/artifacts/points_glb",
    response_class=Response,
)
def get_scene_prior_points(
    request: Request,
    camera_id: str = Path(
        ...,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    ),
) -> Response:
    priors, _calibration, _camera_labels = _runtime_components(request)
    try:
        artifact, payload = priors.artifact_for_camera(camera_id, "points_glb")
    except ScenePriorError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    response = Response(
        content=payload,
        media_type="model/gltf-binary",
        headers={
            "Content-Length": str(artifact.size_bytes),
            "ETag": f'"{artifact.sha256}"',
            "X-Noesis-Artifact-Sha256": artifact.sha256,
            "Cache-Control": "private, max-age=0, must-revalidate",
            "Content-Disposition": (
                f'inline; filename="{camera_id}-{artifact.sha256[:12]}.glb"'
            ),
        },
    )
    mark_rest_response_exempt(request, reason="scene_prior_immutable_artifact")
    return response


__all__ = ["app"]
