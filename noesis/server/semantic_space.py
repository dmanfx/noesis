"""Private HTTP boundary for owner-declared semantic names and relationships.

The existing internal bearer middleware authenticates the Menon gateway. That
gateway supplies its authenticated session actor; this API does not accept a
browser credential or derive identity from labels, headers, geometry or devices.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from noesis.server.boundary_metrics import mark_rest_response, measure_rest_response_model
from noesis_core.contracts.semantic_space import SpaceWrite
from noesis_core.private_paths import PrivatePathError
from noesis_core.scene_store import SceneReleaseConflict, SceneReleaseStoreError
from noesis_core.semantic_space_store import (
    MAX_BYTES, SemanticSpaceStore, SpaceLookupExpired, SpaceLookupMissing,
)

MAX_BODY = MAX_BYTES
BODY_TIMEOUT_SECONDS = 5
_workers = threading.BoundedSemaphore(2)


def _read_store() -> SemanticSpaceStore:
    from noesis.server.scene_api import _database_path
    return SemanticSpaceStore(database_path=_database_path())


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("nonfinite JSON")


async def _command(request: Request) -> SpaceWrite:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, detail="semantic_space_json_required")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BODY:
            raise HTTPException(413, detail="semantic_space_request_too_large")
        data.extend(chunk)
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite)
        return SpaceWrite.model_validate(raw)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise HTTPException(422, detail="semantic_space_request_invalid") from exc


def _response(request: Request, result) -> JSONResponse:
    route = "/api/v1/scenes/semantic-space/lookup:post" if request.method == "POST" else f"/api/v1/scenes/current/semantic-space:{request.method.lower()}"
    name = type(result).__name__
    with measure_rest_response_model(route, name) as measurement:
        response = JSONResponse(result.model_dump(mode="json"), headers={"Cache-Control": "private, no-store"})
        if len(response.body) > MAX_BYTES:
            raise HTTPException(503, detail="semantic_space_response_too_large")
    mark_rest_response(request, route, name, model_duration_ms=measurement.elapsed_ms)
    return response


def _perform(request: Request, action: str, command: SpaceWrite | None):
    try:
        store = _read_store()
        if action == "read":
            result = store.read()
        elif action == "lookup":
            result = store.lookup(command)
        else:
            # Exact replay is independent of current selection and never opens a
            # writable connection. A miss must still pass both current-scene CAS
            # checks inside the scene selector's own SQLite transaction.
            try:
                result = store.lookup(command)
            except SpaceLookupMissing:
                store.read()  # Absent/unpromoted stores must not be initialized.
                from noesis.server.scene_api import _store
                result = SemanticSpaceStore(_store()).replace(command)
        return _response(request, result)
    except SpaceLookupMissing as exc:
        raise HTTPException(404, detail="semantic_space_original_not_found") from exc
    except SpaceLookupExpired as exc:
        raise HTTPException(410, detail="semantic_space_original_expired") from exc
    except SceneReleaseConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except (SceneReleaseStoreError, sqlite3.Error, PrivatePathError, OSError, ValidationError, ValueError) as exc:
        raise HTTPException(503, detail="semantic_space_store_unavailable") from exc


async def handle(request: Request, action: str) -> JSONResponse:
    if request.query_params:
        raise HTTPException(422, detail="semantic_space_query_not_allowed")
    if not _workers.acquire(blocking=False):
        raise HTTPException(503, detail="semantic_space_busy", headers={"Retry-After": "1"})
    try:
        try:
            async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
                command = None if action == "read" else await _command(request)
        except TimeoutError as exc:
            raise HTTPException(408, detail="semantic_space_request_timeout") from exc
        return await run_in_threadpool(_perform, request, action, command)
    finally:
        _workers.release()
