"""Private bearer API for reviewed owner-declared model annotations.

The internal gateway authenticates its owner/session and supplies actor_id, as
for semantic-space. This API never treats a model point as measured world truth.
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
from noesis_core.contracts.semantic_shape import ShapeWrite
from noesis_core.private_paths import PrivatePathError
from noesis_core.scene_store import SceneReleaseConflict, SceneReleaseStoreError
from noesis_core.semantic_shape_store import MAX_BYTES, ShapeLookupExpired, ShapeLookupMissing, SemanticShapeStore

BODY_TIMEOUT_SECONDS = 5
_workers = threading.BoundedSemaphore(2)


def _read_store():
    from noesis.server.scene_api import _database_path
    return SemanticShapeStore(database_path=_database_path())


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("nonfinite JSON")


async def _command(request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, detail="semantic_shape_json_required")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, detail="semantic_shape_request_too_large")
        data.extend(chunk)
    try:
        return ShapeWrite.model_validate(json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise HTTPException(422, detail="semantic_shape_request_invalid") from exc


def _response(request, result):
    route = "/api/v1/scenes/semantic-shapes/lookup:post" if request.method == "POST" else f"/api/v1/scenes/current/semantic-shapes:{request.method.lower()}"
    name = type(result).__name__
    with measure_rest_response_model(route, name) as measurement:
        response = JSONResponse(result.model_dump(mode="json"), headers={"Cache-Control": "private, no-store"})
        if len(response.body) > MAX_BYTES:
            raise HTTPException(503, detail="semantic_shape_response_too_large")
    mark_rest_response(request, route, name, model_duration_ms=measurement.elapsed_ms)
    return response


def _perform(request, action, command):
    try:
        store = _read_store()
        if action == "read":
            result = store.read()
        elif action == "lookup":
            result = store.lookup(command)
        else:
            try:
                result = store.lookup(command)
            except ShapeLookupMissing:
                store.read()  # Never initialize an absent or unpromoted store.
                from noesis.server.scene_api import _store
                result = SemanticShapeStore(_store()).replace(command)
        return _response(request, result)
    except ShapeLookupMissing as exc:
        raise HTTPException(404, detail="semantic_shape_original_not_found") from exc
    except ShapeLookupExpired as exc:
        raise HTTPException(410, detail="semantic_shape_original_expired") from exc
    except SceneReleaseConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except (SceneReleaseStoreError, sqlite3.Error, PrivatePathError, OSError, ValidationError, ValueError) as exc:
        raise HTTPException(503, detail="semantic_shape_store_unavailable") from exc


async def handle(request: Request, action: str):
    if action not in ("read", "replace", "lookup"):
        raise HTTPException(422, detail="semantic_shape_action_invalid")
    if request.query_params or request.headers.get("transfer-encoding"):
        raise HTTPException(422, detail="semantic_shape_query_not_allowed")
    if not _workers.acquire(blocking=False):
        raise HTTPException(503, detail="semantic_shape_busy", headers={"Retry-After": "1"})
    try:
        try:
            async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
                if action == "read":
                    async for chunk in request.stream():
                        if chunk:
                            raise HTTPException(422, detail="semantic_shape_get_body_not_allowed")
                    command = None
                else:
                    command = await _command(request)
        except TimeoutError as exc:
            raise HTTPException(408, detail="semantic_shape_request_timeout") from exc
        return await run_in_threadpool(_perform, request, action, command)
    finally:
        _workers.release()
