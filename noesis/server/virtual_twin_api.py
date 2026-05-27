from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from noesis.virtual_twin.store import VirtualTwinStore, VirtualTwinStoreError


app = FastAPI(title="Noesis Virtual Twin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _store() -> VirtualTwinStore:
    return VirtualTwinStore()


def _http_error(exc: VirtualTwinStoreError) -> HTTPException:
    text = str(exc)
    status = 404 if "missing" in text or "no virtual-twin revision" in text else 400
    return HTTPException(status_code=status, detail=text)


@app.get("/api/v1/virtual-twin/revisions")
def list_virtual_twin_revisions() -> dict[str, Any]:
    return {"revisions": _store().list_revisions()}


@app.get("/api/v1/virtual-twin/latest")
def get_latest_virtual_twin() -> dict[str, Any]:
    try:
        return _store().latest_payload()
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/manifest")
def get_virtual_twin_manifest(revision_id: str) -> dict[str, Any]:
    try:
        store = _store()
        manifest = store.read_manifest(revision_id)
        return {
            "revision_id": revision_id,
            "manifest": manifest,
            "artifact_urls": store.artifact_urls(revision_id, manifest),
        }
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/metrics")
def get_virtual_twin_metrics(revision_id: str) -> dict[str, Any]:
    try:
        return {"revision_id": revision_id, "metrics": _store().read_metrics(revision_id)}
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/tracking-alignment")
def get_virtual_twin_tracking_alignment(revision_id: str) -> dict[str, Any]:
    try:
        return {"revision_id": revision_id, "tracking_alignment": _store().read_tracking_alignment(revision_id)}
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc


@app.get("/api/v1/virtual-twin/revisions/{revision_id}/artifacts/{artifact_path:path}")
def get_virtual_twin_artifact(revision_id: str, artifact_path: str) -> FileResponse:
    try:
        path = _store().artifact_path(revision_id, artifact_path)
    except VirtualTwinStoreError as exc:
        raise _http_error(exc) from exc
    media_type = _media_type(path)
    return FileResponse(path, media_type=media_type, filename=path.name)


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
