"""Private owner-gateway-only inspection/review of declared object names."""
from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
import threading

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from noesis.server.boundary_metrics import BoundaryMetricsRoute, mark_rest_response, measure_rest_response_model
from noesis_core.contracts.object_associations import AssociationCommand, AssociationConfirm, AssociationInspect, AssociationPrepare, MAX_BYTES
from noesis_core.object_association_store import ObjectAssociationStore, AssociationConflict, AssociationError, AssociationExpired, AssociationMissing
from noesis_core.object_observations import ObjectObservationService
from noesis_core.private_paths import PrivatePathError

PATH = "/api/v1/objects/associations"
router = APIRouter(route_class=BoundaryMetricsRoute)
_workers = threading.BoundedSemaphore(2)
_owner_lock = threading.Lock()


def _store():
    from noesis.server.scene_api import _database_path
    return ObjectAssociationStore(_database_path())


def _owner(request):
    # A process-local token binds pending persisted inspections to the EXACT
    # admitted service instance and run. Restart/replacement never resumes them.
    with _owner_lock:
        owner = getattr(request.app.state, "object_observation_service", None)
        if type(owner) is not ObjectObservationService or owner._closed:
            request.app.state.object_association_source = None
            raise AssociationConflict("object observation owner unavailable")
        current = getattr(request.app.state, "object_association_source", None)
        if not current or current[0] is not owner or current[1] != owner.run_id:
            current = (owner, owner.run_id, secrets.token_urlsafe(32))
            request.app.state.object_association_source = current
    def check():
        return (getattr(request.app.state, "object_observation_service", None) is owner and not owner._closed
            and owner.run_id == current[1] and getattr(request.app.state, "object_association_source", None) is current)
    return owner, current[2], check


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("nonfinite JSON")


async def _body(request, model):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, detail="object_association_json_required")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, detail="object_association_request_too_large")
        data.extend(chunk)
    try:
        return model.model_validate(json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise HTTPException(422, detail="object_association_request_invalid") from exc


def _perform(request, action, command):
    try:
        store = _store()
        source = None
        if action == "read":
            result = store.read()
        elif action == "lookup":
            result = store.lookup(command)
        elif action == "confirm":
            # A completed original is independent of the current runtime owner.
            try:
                result = store.lookup(command.command)
            except AssociationMissing:
                source = _owner(request)
                result = store.confirm(command, *source)
        else:
            source = _owner(request)
            result = getattr(store, action)(command, *source)
        if action in {"inspect", "prepare"}:
            result = store._remaining(result)
        route = PATH + ":" + action
        with measure_rest_response_model(route, type(result).__name__) as measurement:
            response = JSONResponse(result.model_dump(mode="json"), headers={"Cache-Control": "private, no-store"})
            if len(response.body) > MAX_BYTES:
                raise HTTPException(503, detail="object_association_response_too_large")
        if source is not None and source[2]() is not True:
            raise AssociationConflict("object observation owner retired before delivery")
        mark_rest_response(request, route, type(result).__name__, model_duration_ms=measurement.elapsed_ms)
        return response
    except AssociationMissing as exc:
        raise HTTPException(404, detail="object_association_not_found") from exc
    except AssociationExpired as exc:
        raise HTTPException(410, detail="object_association_expired") from exc
    except AssociationConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except (AssociationError, PrivatePathError, sqlite3.Error, OSError, ValidationError, ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(503, detail="object_association_store_unavailable") from exc


async def _handle(request, action, model=None):
    if request.scope.get("query_string"):
        raise HTTPException(422, detail="object_association_query_not_allowed")
    if request.headers.get("transfer-encoding"):
        raise HTTPException(422, detail="object_association_transfer_encoding_not_allowed")
    if not _workers.acquire(blocking=False):
        raise HTTPException(503, detail="object_association_busy", headers={"Retry-After": "1"})
    try:
        try:
            async with asyncio.timeout(5):
                if model is None:
                    async for chunk in request.stream():
                        if chunk:
                            raise HTTPException(422, detail="object_association_get_body_not_allowed")
                    command = None
                else:
                    command = await _body(request, model)
        except TimeoutError as exc:
            raise HTTPException(408, detail="object_association_request_timeout") from exc
        return await run_in_threadpool(_perform, request, action, command)
    finally:
        _workers.release()


@router.get(PATH)
async def read_associations(request: Request):
    return await _handle(request, "read")


@router.post(PATH + "/inspect")
async def inspect_associations(request: Request):
    return await _handle(request, "inspect", AssociationInspect)


@router.post(PATH + "/prepare")
async def prepare_association(request: Request):
    return await _handle(request, "prepare", AssociationPrepare)


@router.put(PATH)
async def confirm_association(request: Request):
    return await _handle(request, "confirm", AssociationConfirm)


@router.post(PATH + "/lookup")
async def lookup_association(request: Request):
    return await _handle(request, "lookup", AssociationCommand)
