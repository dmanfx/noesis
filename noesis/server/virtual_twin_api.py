from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    mark_rest_response_exempt,
    measure_rest_response_model,
)
from noesis.virtual_twin.store import VirtualTwinStore, VirtualTwinStoreError


app = FastAPI(title="Noesis Virtual Twin API")
app.router.route_class = BoundaryMetricsRoute
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _store() -> VirtualTwinStore:
    return VirtualTwinStore()


def _http_error(exc: VirtualTwinStoreError) -> HTTPException:
    text = str(exc)
    status = 404 if "missing" in text or "no virtual-twin revision" in text else 400
    return HTTPException(status_code=status, detail=text)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"missing calibration file: {path.relative_to(REPO_ROOT)}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid calibration json: {path.relative_to(REPO_ROOT)}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"calibration json root must be an object: {path.relative_to(REPO_ROOT)}")
    return data


@app.get("/api/v1/virtual-twin/revisions")
def list_virtual_twin_revisions(request: Request) -> dict[str, Any]:
    revisions = _store().list_revisions()
    with measure_rest_response_model(
        "/api/v1/virtual-twin/revisions:get", "VirtualTwinRevisionList"
    ) as model_measurement:
        response = {"revisions": revisions}
    mark_rest_response(
        request,
        "/api/v1/virtual-twin/revisions:get",
        "VirtualTwinRevisionList",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@app.get("/api/v1/virtual-twin/latest")
def get_latest_virtual_twin(request: Request) -> dict[str, Any]:
    try:
        payload = _store().latest_payload()
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc
    with measure_rest_response_model(
        "/api/v1/virtual-twin/latest:get", "VirtualTwinLatestPayload"
    ) as model_measurement:
        response = payload
    mark_rest_response(
        request,
        "/api/v1/virtual-twin/latest:get",
        "VirtualTwinLatestPayload",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/manifest")
def get_virtual_twin_manifest(
    revision_id: str,
    request: Request,
) -> dict[str, Any]:
    try:
        store = _store()
        manifest = store.read_manifest(revision_id)
        artifact_urls = store.artifact_urls(revision_id, manifest)
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc
    with measure_rest_response_model(
        "/api/v1/virtual-twin/revisions/{revision_id}/manifest:get",
        "VirtualTwinManifestResponse",
    ) as model_measurement:
        response = {
            "revision_id": revision_id,
            "manifest": manifest,
            "artifact_urls": artifact_urls,
        }
    mark_rest_response(
        request,
        "/api/v1/virtual-twin/revisions/{revision_id}/manifest:get",
        "VirtualTwinManifestResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/metrics")
def get_virtual_twin_metrics(
    revision_id: str,
    request: Request,
) -> dict[str, Any]:
    try:
        metrics = _store().read_metrics(revision_id)
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc
    with measure_rest_response_model(
        "/api/v1/virtual-twin/revisions/{revision_id}/metrics:get",
        "VirtualTwinMetricsResponse",
    ) as model_measurement:
        response = {"revision_id": revision_id, "metrics": metrics}
    mark_rest_response(
        request,
        "/api/v1/virtual-twin/revisions/{revision_id}/metrics:get",
        "VirtualTwinMetricsResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/tracking-alignment")
def get_virtual_twin_tracking_alignment(
    revision_id: str,
    request: Request,
) -> dict[str, Any]:
    try:
        tracking_alignment = _store().read_tracking_alignment(revision_id)
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc
    with measure_rest_response_model(
        "/api/v1/virtual-twin/revisions/{revision_id}/tracking-alignment:get",
        "VirtualTwinTrackingAlignmentResponse",
    ) as model_measurement:
        response = {
            "revision_id": revision_id,
            "tracking_alignment": tracking_alignment,
        }
    mark_rest_response(
        request,
        "/api/v1/virtual-twin/revisions/{revision_id}/tracking-alignment:get",
        "VirtualTwinTrackingAlignmentResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@app.get("/api/v1/virtual-twin/calibration/scene-extrinsics")
def get_scene_extrinsics(request: Request) -> dict[str, Any]:
    """Return Menon-scene camera poses used as calibration, not room geometry."""
    rel_path = Path("config") / "camera_calibration_menon_obj.json"
    payload = _read_json(REPO_ROOT / rel_path)
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise HTTPException(status_code=404, detail=f"no cameras in calibration file: {rel_path}")
    with measure_rest_response_model(
        "/api/v1/virtual-twin/calibration/scene-extrinsics:get",
        "SceneExtrinsicsResponse",
    ) as model_measurement:
        response = {
            "source": "camera_calibration_menon_obj",
            "path": str(rel_path),
            "frame": "menon_scene",
            "cameras": cameras,
            "preview_meta": (
                payload.get("preview_meta")
                if isinstance(payload.get("preview_meta"), dict)
                else None
            ),
        }
    mark_rest_response(
        request,
        "/api/v1/virtual-twin/calibration/scene-extrinsics:get",
        "SceneExtrinsicsResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/artifacts/{artifact_path:path}")
def get_virtual_twin_artifact(
    revision_id: str,
    artifact_path: str,
    request: Request,
) -> FileResponse:
    try:
        path = _store().artifact_path(revision_id, artifact_path)
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc
    media_type = _media_type(path)
    response = FileResponse(path, media_type=media_type, filename=path.name)
    mark_rest_response_exempt(request, reason="file_response")
    return response


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".glb":
        return "model/gltf-binary"
    if suffix == ".png":
        return "image/png"
    if suffix == ".ply":
        return "application/octet-stream"
    if suffix == ".npz":
        return "application/octet-stream"
    return "application/octet-stream"


__all__ = ["app"]
