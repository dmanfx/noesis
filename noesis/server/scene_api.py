from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    mark_rest_response_exempt,
    measure_rest_response_model,
)
from noesis.virtual_twin.store import VirtualTwinStore
from noesis_core.contracts.scene import SceneRelease
from noesis_core.scene_store import (
    SceneReleaseConflict,
    SceneReleaseStore,
    SceneReleaseStoreError,
)
from noesis_core.scene_files import VerifiedSceneFile


router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["scene-releases"],
    route_class=BoundaryMetricsRoute,
)


class SceneRegistrationResponse(BaseModel):
    release_id: str
    release_sha256: str
    registered: bool = True


class SceneMutationRequest(BaseModel):
    actor_id: str = Field(min_length=1, max_length=160)
    expected_current_release_id: str | None = Field(default=None, max_length=200)


class ScenePromotionResponse(BaseModel):
    sequence: int
    event: Literal["promote", "rollback"]
    release_id: str
    previous_release_id: str | None
    actor_id: str
    occurred_at_us: int
    release_sha256: str


def _database_path() -> Path:
    configured = str(os.environ.get("NOESIS_SCENE_STORE_PATH", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "noesis" / "scene_releases.sqlite3"


def _store() -> SceneReleaseStore:
    virtual_twins = VirtualTwinStore()
    return SceneReleaseStore(
        _database_path(),
        artifact_root=virtual_twins.revisions_root,
        bundle_root=virtual_twins.root,
    )


def _http_error(exc: SceneReleaseStoreError) -> HTTPException:
    message = str(exc)
    if isinstance(exc, SceneReleaseConflict):
        return HTTPException(status_code=409, detail=message)
    if "unknown scene release" in message:
        return HTTPException(status_code=404, detail=message)
    return HTTPException(status_code=400, detail=message)


def _current_selection() -> tuple[SceneReleaseStore, SceneRelease]:
    try:
        store = _store()
        release = store.current()
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if release is None:
        raise HTTPException(status_code=404, detail="no scene release is promoted")
    return store, release


def _current_snapshot() -> tuple[SceneReleaseStore, SceneRelease]:
    store, release = _current_selection()
    try:
        store.validate_artifacts(release)
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return store, release


def _verified_response(
    artifact: VerifiedSceneFile,
    *,
    release_id: str,
    media_type: str,
) -> Response:
    return Response(
        content=artifact.data,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "ETag": f'"sha256-{artifact.sha256}"',
            "X-Noesis-Scene-Release": release_id,
        },
    )


@router.post("/releases", response_model=SceneRegistrationResponse, status_code=201)
def register_release(
    release: SceneRelease,
    request: Request,
) -> SceneRegistrationResponse:
    try:
        release_sha256 = _store().register(release)
    except SceneReleaseStoreError as exc:
        raise _http_error(exc) from exc
    with measure_rest_response_model(
        "/api/v1/scenes/releases:post", "SceneRegistrationResponse"
    ) as model_measurement:
        response = SceneRegistrationResponse(
            release_id=release.release_id,
            release_sha256=release_sha256,
        )
    mark_rest_response(
        request,
        "/api/v1/scenes/releases:post",
        "SceneRegistrationResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/releases", response_model=list[SceneRelease])
def list_releases(request: Request) -> list[SceneRelease]:
    try:
        releases = _store().list_releases()
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    with measure_rest_response_model(
        "/api/v1/scenes/releases:get", "SceneReleaseList"
    ) as model_measurement:
        response = list(releases)
    mark_rest_response(
        request,
        "/api/v1/scenes/releases:get",
        "SceneReleaseList",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/releases/{release_id}", response_model=SceneRelease)
def get_release(release_id: str, request: Request) -> SceneRelease:
    try:
        release = _store().get(release_id)
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if release is None:
        raise HTTPException(status_code=404, detail=f"unknown scene release: {release_id}")
    with measure_rest_response_model(
        "/api/v1/scenes/releases/{release_id}:get", "SceneRelease"
    ) as model_measurement:
        response = release
    mark_rest_response(
        request,
        "/api/v1/scenes/releases/{release_id}:get",
        "SceneRelease",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/current", response_model=SceneRelease)
def get_current_release(request: Request) -> SceneRelease:
    _, release = _current_snapshot()
    with measure_rest_response_model(
        "/api/v1/scenes/current:get", "SceneRelease"
    ) as model_measurement:
        response = release
    mark_rest_response(
        request,
        "/api/v1/scenes/current:get",
        "SceneRelease",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/current/payload")
def get_current_scene_payload(request: Request) -> dict[str, Any]:
    store, release = _current_snapshot()
    camera_manifests: list[tuple[Any, dict[str, Any]]] = []
    for camera in release.cameras:
        try:
            manifest = store.read_camera_manifest(camera)
        except SceneReleaseStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        camera_manifests.append((camera, manifest))
    with measure_rest_response_model(
        "/api/v1/scenes/current/payload:get", "ScenePayloadResponse"
    ) as model_measurement:
        cameras = [
            {
                "camera_id": camera.camera_id,
                "revision_id": camera.revision_id,
                "manifest_sha256": camera.manifest_sha256,
                "manifest": manifest,
                "artifact_urls": {
                    artifact.role: (
                        f"/api/v1/scenes/current/cameras/{camera.camera_id}/"
                        f"artifacts/{artifact.role}"
                    )
                    for artifact in camera.artifacts
                },
            }
            for camera, manifest in camera_manifests
        ]
        response = {
            "release": release.model_dump(mode="json"),
            "authored_scene_url": "/api/v1/scenes/current/authored-scene",
            "authored_scene_dependency_urls": {
                dependency.role: (
                    f"/api/v1/scenes/current/authored-dependencies/{dependency.role}"
                )
                for dependency in release.authored_scene_dependencies
            },
            "validation_report_url": "/api/v1/scenes/current/validation-report",
            "cameras": cameras,
        }
    mark_rest_response(
        request,
        "/api/v1/scenes/current/payload:get",
        "ScenePayloadResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/current/authored-scene", response_class=Response)
def get_current_authored_scene(request: Request) -> Response:
    store, release = _current_selection()
    try:
        artifact = store.read_authored_scene(release)
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response = _verified_response(
        artifact,
        release_id=release.release_id,
        media_type="application/octet-stream",
    )
    mark_rest_response_exempt(request, reason="pre_rendered_response")
    return response


@router.get(
    "/current/authored-dependencies/{dependency_role}",
    response_class=Response,
)
def get_current_authored_scene_dependency(
    dependency_role: str,
    request: Request,
) -> Response:
    store, release = _current_selection()
    dependency = next(
        (
            item
            for item in release.authored_scene_dependencies
            if item.role == dependency_role
        ),
        None,
    )
    if dependency is None:
        raise HTTPException(
            status_code=404,
            detail=f"authored scene dependency role is not in current scene: {dependency_role}",
        )
    try:
        artifact = store.read_authored_dependency(dependency)
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response = _verified_response(
        artifact,
        release_id=release.release_id,
        media_type="application/octet-stream",
    )
    mark_rest_response_exempt(request, reason="pre_rendered_response")
    return response


@router.get("/current/validation-report", response_class=Response)
def get_current_validation_report(request: Request) -> Response:
    store, release = _current_selection()
    try:
        artifact = store.read_validation_report(release)
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response = _verified_response(
        artifact,
        release_id=release.release_id,
        media_type="application/json",
    )
    mark_rest_response_exempt(request, reason="pre_rendered_response")
    return response


@router.get(
    "/current/cameras/{camera_id}/artifacts/{artifact_role}",
    response_class=Response,
)
def get_current_camera_artifact(
    camera_id: str,
    artifact_role: str,
    request: Request,
) -> Response:
    store, release = _current_selection()
    camera = next(
        (item for item in release.cameras if item.camera_id == camera_id),
        None,
    )
    if camera is None:
        raise HTTPException(status_code=404, detail=f"camera is not in current scene: {camera_id}")
    artifact = next(
        (item for item in camera.artifacts if item.role == artifact_role),
        None,
    )
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail=f"artifact role is not in current scene camera {camera_id}: {artifact_role}",
        )
    try:
        verified = store.read_camera_artifact(camera, artifact)
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response = _verified_response(
        verified,
        release_id=release.release_id,
        media_type="application/octet-stream",
    )
    mark_rest_response_exempt(request, reason="pre_rendered_response")
    return response


@router.post("/releases/{release_id}/promote", response_model=ScenePromotionResponse)
def promote_release(
    release_id: str,
    request: SceneMutationRequest,
    http_request: Request,
) -> ScenePromotionResponse:
    try:
        event = _store().promote(
            release_id,
            actor_id=request.actor_id,
            occurred_at_us=max(1, time.time_ns() // 1_000),
            expected_current_release_id=request.expected_current_release_id,
        )
    except SceneReleaseStoreError as exc:
        raise _http_error(exc) from exc
    with measure_rest_response_model(
        "/api/v1/scenes/releases/{release_id}/promote:post",
        "ScenePromotionResponse",
    ) as model_measurement:
        response = ScenePromotionResponse.model_validate(asdict(event))
    mark_rest_response(
        http_request,
        "/api/v1/scenes/releases/{release_id}/promote:post",
        "ScenePromotionResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.post("/releases/{release_id}/rollback", response_model=ScenePromotionResponse)
def rollback_release(
    release_id: str,
    request: SceneMutationRequest,
    http_request: Request,
) -> ScenePromotionResponse:
    if request.expected_current_release_id is None:
        raise HTTPException(status_code=422, detail="rollback requires expected_current_release_id")
    try:
        event = _store().rollback(
            release_id,
            actor_id=request.actor_id,
            occurred_at_us=max(1, time.time_ns() // 1_000),
            expected_current_release_id=request.expected_current_release_id,
        )
    except SceneReleaseStoreError as exc:
        raise _http_error(exc) from exc
    with measure_rest_response_model(
        "/api/v1/scenes/releases/{release_id}/rollback:post",
        "ScenePromotionResponse",
    ) as model_measurement:
        response = ScenePromotionResponse.model_validate(asdict(event))
    mark_rest_response(
        http_request,
        "/api/v1/scenes/releases/{release_id}/rollback:post",
        "ScenePromotionResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/history", response_model=list[ScenePromotionResponse])
def promotion_history(request: Request) -> list[ScenePromotionResponse]:
    try:
        events = _store().history()
    except SceneReleaseStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    with measure_rest_response_model(
        "/api/v1/scenes/history:get", "ScenePromotionResponseList"
    ) as model_measurement:
        response = [
            ScenePromotionResponse.model_validate(asdict(event)) for event in events
        ]
    mark_rest_response(
        request,
        "/api/v1/scenes/history:get",
        "ScenePromotionResponseList",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


__all__ = ["router"]
