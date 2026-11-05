"""Bounded private as-of read of the actual canonical World owner."""
from __future__ import annotations

import asyncio
import json
import math
from threading import BoundedSemaphore

from fastapi import APIRouter, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from noesis.server.boundary_metrics import BoundaryMetricsRoute, mark_rest_response, measure_rest_response_model
from noesis_core.contracts.world import WorldSnapshot
from noesis_core.world_service import CanonicalWorldService
from noesis_core.spatial_relationships import build_spatial_relationships, encode_spatial_relationships, unknown_spatial_relationships
from noesis_core.world_history import world_history_reader, encode_world_history, unknown_world_history
from noesis_core.temporal_relationships import temporal_relationships_reader, encode_temporal_relationships, unknown_temporal_relationships

PATH = "/api/v1/world/current"
RELATIONSHIPS_PATH = "/api/v1/world/relationships"
HISTORY_PATH = "/api/v1/world/history"
TEMPORAL_RELATIONSHIPS_PATH = "/api/v1/world/temporal-relationships"
MAX_SNAPSHOT_AGE_US = 2_000_000
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ENTITIES = 256
MAX_SOURCE_ROWS = 4096
MAX_MODEL_NODES = 65_536
_READERS = BoundedSemaphore(2)
router = APIRouter(route_class=BoundaryMetricsRoute)


def _render_temporal(request: Request, body: bytes, status_code=200):
    mark_rest_response(request, TEMPORAL_RELATIONSHIPS_PATH + ":get", "TemporalRelationshipsRead",
        model_duration_ms=getattr(request.state, "temporal_model_duration_ms", 0.0))
    return Response(body, status_code=status_code, media_type="application/json", headers={"Cache-Control": "private, no-store"})


def _read_temporal(request):
    service = getattr(request.app.state, "canonical_world_service", None)
    if type(service) is not CanonicalWorldService:
        return encode_temporal_relationships(unknown_temporal_relationships("world_unbound"))
    with measure_rest_response_model(TEMPORAL_RELATIONSHIPS_PATH + ":get", "TemporalRelationshipsRead") as measurement:
        body = temporal_relationships_reader.read(service, is_current=lambda: getattr(request.app.state, "canonical_world_service", None) is service)
    request.state.temporal_model_duration_ms = measurement.elapsed_ms
    return body


@router.get(TEMPORAL_RELATIONSHIPS_PATH)
async def temporal_world_relationships(request: Request):
    invalid = lambda reason: _render_temporal(request, encode_temporal_relationships(unknown_temporal_relationships(reason)), 400)
    if request.scope.get("query_string") or request.headers.get("transfer-encoding") or request.headers.get("content-length") not in {None, "0"}:
        return invalid("query_or_body_not_allowed")
    try:
        message = await asyncio.wait_for(request.receive(), timeout=0.1)
    except (TimeoutError, asyncio.TimeoutError):
        return invalid("request_body_unavailable")
    if message.get("type") != "http.request" or message.get("body") or message.get("more_body"):
        return invalid("query_or_body_not_allowed")
    return _render_temporal(request, await run_in_threadpool(_read_temporal, request))


def _render_history(request: Request, body: bytes, status_code=200) -> Response:
    mark_rest_response(request, HISTORY_PATH + ":get", "WorldHistoryRead",
        model_duration_ms=getattr(request.state, "world_history_model_duration_ms", 0.0))
    return Response(body, status_code=status_code, media_type="application/json", headers={"Cache-Control": "private, no-store"})


def _history_read(request: Request, limit: int, cursor: str | None) -> bytes:
    service = getattr(request.app.state, "canonical_world_service", None)
    if type(service) is not CanonicalWorldService:
        return encode_world_history(unknown_world_history("world_unbound"))
    with measure_rest_response_model(HISTORY_PATH + ":get", "WorldHistoryRead") as measurement:
        body = world_history_reader.read(service, limit=limit, cursor=cursor,
            is_current=lambda: getattr(request.app.state, "canonical_world_service", None) is service)
    request.state.world_history_model_duration_ms = measurement.elapsed_ms
    return body


@router.get(HISTORY_PATH)
async def world_history(request: Request) -> Response:
    invalid = lambda reason: _render_history(request, encode_world_history(unknown_world_history(reason)), 400)
    if (len(request.scope.get("query_string", b"")) > 256 or request.headers.get("transfer-encoding")
            or request.headers.get("content-length") not in {None, "0"}):
        return invalid("invalid_query_or_body")
    pairs = list(request.query_params.multi_items())
    if len({key for key, _ in pairs}) != len(pairs) or any(key not in {"limit", "cursor"} for key, _ in pairs):
        return invalid("invalid_query")
    values = dict(pairs); raw_limit = values.get("limit", "50"); cursor = values.get("cursor")
    if (not raw_limit.isascii() or not raw_limit.isdecimal() or str(int(raw_limit)) != raw_limit
            or not 1 <= int(raw_limit) <= 50 or cursor is not None and (
                len(cursor) != 43 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in cursor))):
        return invalid("invalid_query")
    try:
        message = await asyncio.wait_for(request.receive(), timeout=0.1)
    except (TimeoutError, asyncio.TimeoutError):
        return invalid("request_body_unavailable")
    if message.get("type") != "http.request" or message.get("body") or message.get("more_body"):
        return invalid("invalid_query_or_body")
    return _render_history(request, await run_in_threadpool(_history_read, request, int(raw_limit), cursor))


def _unknown(reason: str, at_us: int | None = None) -> dict:
    return {"contract": "noesis.world.current-read", "version": 1, "status": "unknown", "reason": reason,
        "generated_at_us": at_us, "valid_until_us": None, "max_snapshot_age_us": MAX_SNAPSHOT_AGE_US, "snapshot": None}


def _encode(value: dict) -> bytes | None:
    chunks, size = [], 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for text in encoder.iterencode(value):
        chunk = text.encode("utf-8")
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _bounded_model(snapshot: WorldSnapshot) -> bool:
    """Preflight immutable models before allocating a detached JSON tree."""
    nodes = 0

    def visit(value, depth=0, name=""):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_MODEL_NODES or depth > 24:
            return False
        if name.endswith("_us") and value is not None and (type(value) is not int or not 0 < value <= 2**53 - 1):
            return False
        if value is None or isinstance(value, (str, bool)):
            return not isinstance(value, str) or len(value) <= MAX_RESPONSE_BYTES
        if type(value) is int:
            return not name.endswith("_us") or 0 < value <= 2**53 - 1
        if type(value) is float:
            return math.isfinite(value) and not name.endswith("_us")
        if isinstance(value, BaseModel):
            return all(visit(getattr(value, key), depth + 1, key) for key in type(value).model_fields)
        if isinstance(value, (tuple, list)):
            return len(value) <= MAX_MODEL_NODES - nodes and all(visit(item, depth + 1, name) for item in value)
        return False

    return visit(snapshot)


def _fresh(snapshot: WorldSnapshot, at_us: int) -> bool:
    published = snapshot.published_at_us
    return (type(published) is int and 0 < published <= 2**53 - 1 - MAX_SNAPSHOT_AGE_US
        and published <= at_us < published + MAX_SNAPSHOT_AGE_US)


def _guard(request: Request, service: CanonicalWorldService, snapshot: WorldSnapshot):
    if getattr(request.app.state, "canonical_world_service", None) is not service:
        return None, "world_owner_changed"
    latest, at_us, reason = service.read_current_snapshot()
    if getattr(request.app.state, "canonical_world_service", None) is not service:
        return None, "world_owner_changed"
    if reason != "available" or type(latest) is not WorldSnapshot:
        return at_us, reason if reason != "available" else "invalid_snapshot"
    if latest.producer != snapshot.producer or service.producer != snapshot.producer:
        return at_us, "world_owner_changed"
    if not _fresh(snapshot, at_us):
        return at_us, "snapshot_not_fresh"
    return at_us, None


def _render(request: Request, payload: dict, *, status_code=200) -> Response:
    with measure_rest_response_model(PATH + ":get", "WorldCurrentRead") as measurement:
        body = _encode(payload)
    mark_rest_response(request, PATH + ":get", "WorldCurrentRead", model_duration_ms=measurement.elapsed_ms)
    return Response(body, status_code=status_code, media_type="application/json", headers={"Cache-Control": "private, no-store"})


def _read(request: Request) -> Response:
    service = getattr(request.app.state, "canonical_world_service", None)
    if type(service) is not CanonicalWorldService:
        return _render(request, _unknown("world_unbound"))
    snapshot, at_us, reason = service.read_current_snapshot()
    if reason != "available" or type(snapshot) is not WorldSnapshot:
        return _render(request, _unknown(reason, at_us))
    if len(snapshot.entities) > MAX_ENTITIES:
        return _render(request, _unknown("entity_limit_exceeded", at_us))
    if sum(len(entity.sources) for entity in snapshot.entities) > MAX_SOURCE_ROWS:
        return _render(request, _unknown("source_evidence_limit_exceeded", at_us))
    if not _fresh(snapshot, at_us):
        return _render(request, _unknown("snapshot_not_fresh", at_us))
    with measure_rest_response_model(PATH + ":get", "WorldCurrentRead") as measurement:
        try:
            if not _bounded_model(snapshot):
                return _render(request, _unknown("snapshot_bound_exceeded", at_us))
            detached = snapshot.model_dump(mode="json")
            at_us, error = _guard(request, service, snapshot)
            if error:
                return _render(request, _unknown(error, at_us))
            payload = {"contract": "noesis.world.current-read", "version": 1, "status": "available", "reason": "committed_snapshot_as_of_read",
                "generated_at_us": at_us, "valid_until_us": snapshot.published_at_us + MAX_SNAPSHOT_AGE_US,
                "max_snapshot_age_us": MAX_SNAPSHOT_AGE_US, "snapshot": detached}
            body = _encode(payload)
            if body is None:
                return _render(request, _unknown("response_byte_limit_exceeded", at_us))
            final_us, error = _guard(request, service, snapshot)
            if error:
                return _render(request, _unknown(error, final_us))
        except (ValueError, TypeError, OverflowError, RecursionError):
            return _render(request, _unknown("invalid_snapshot", at_us))
    mark_rest_response(request, PATH + ":get", "WorldCurrentRead", model_duration_ms=measurement.elapsed_ms)
    return Response(body, media_type="application/json", headers={"Cache-Control": "private, no-store"})


@router.get(PATH)
async def current_world(request: Request) -> Response:
    # No query or body has meaning on this read. Inspect one bounded ASGI body
    # message without buffering an upload; never wait for it on the World lock.
    if request.scope.get("query_string") or request.headers.get("transfer-encoding"):
        return _render(request, _unknown("query_or_body_not_allowed"), status_code=400)
    length = request.headers.get("content-length")
    if length not in {None, "0"}:
        return _render(request, _unknown("query_or_body_not_allowed"), status_code=400)
    try:
        message = await asyncio.wait_for(request.receive(), timeout=0.1)
    except (TimeoutError, asyncio.TimeoutError):
        return _render(request, _unknown("request_body_unavailable"), status_code=400)
    if message.get("type") != "http.request" or message.get("body") or message.get("more_body"):
        return _render(request, _unknown("query_or_body_not_allowed"), status_code=400)
    if not _READERS.acquire(blocking=False):
        return _render(request, _unknown("reader_limit_exceeded"))
    try:
        return await run_in_threadpool(_read, request)
    finally:
        _READERS.release()


def _render_relationships(request: Request, result, *, status_code=200) -> Response:
    with measure_rest_response_model(RELATIONSHIPS_PATH + ":get", "SpatialRelationshipsRead") as measurement:
        body = encode_spatial_relationships(result)
    mark_rest_response(request, RELATIONSHIPS_PATH + ":get", "SpatialRelationshipsRead", model_duration_ms=measurement.elapsed_ms)
    return Response(body, status_code=status_code, media_type="application/json", headers={"Cache-Control": "private, no-store"})


def _read_relationships(request: Request) -> Response:
    service = getattr(request.app.state, "canonical_world_service", None)
    if type(service) is not CanonicalWorldService:
        return _render_relationships(request, unknown_spatial_relationships("world_unbound"))
    snapshot, at_us, reason = service.read_current_snapshot()
    if reason != "available" or type(snapshot) is not WorldSnapshot:
        return _render_relationships(request, unknown_spatial_relationships(reason, at_us))
    with measure_rest_response_model(RELATIONSHIPS_PATH + ":get", "SpatialRelationshipsRead") as measurement:
        result = build_spatial_relationships(snapshot, at_us)
        if result.status == "unknown":
            return _render_relationships(request, result)
        at_us, error = _guard(request, service, snapshot)
        if error:
            return _render_relationships(request, unknown_spatial_relationships(error, at_us))
        # Re-evaluate per-entity expiry using the final owner clock, without
        # replacing or renewing the captured original snapshot publication.
        result = build_spatial_relationships(snapshot, at_us)
        body = encode_spatial_relationships(result)
        if body is None:
            return _render_relationships(request, unknown_spatial_relationships("response_byte_limit_exceeded", at_us))
        final_us, error = _guard(request, service, snapshot)
        if error:
            return _render_relationships(request, unknown_spatial_relationships(error, final_us))
        if result.status == "available" and any(
            entity.assigned_to_room.state == "known" or entity.supported_by.state == "known"
            for entity in result.entities
            if final_us >= entity.stale_after_us
        ):
            return _render_relationships(request, unknown_spatial_relationships("entity_expired_during_read", final_us))
    mark_rest_response(request, RELATIONSHIPS_PATH + ":get", "SpatialRelationshipsRead", model_duration_ms=measurement.elapsed_ms)
    return Response(body, media_type="application/json", headers={"Cache-Control": "private, no-store"})


@router.get(RELATIONSHIPS_PATH)
async def current_world_relationships(request: Request) -> Response:
    if request.scope.get("query_string") or request.headers.get("transfer-encoding") or request.headers.get("content-length") not in {None, "0"}:
        return _render_relationships(request, unknown_spatial_relationships("query_or_body_not_allowed"), status_code=400)
    try:
        message = await asyncio.wait_for(request.receive(), timeout=0.1)
    except (TimeoutError, asyncio.TimeoutError):
        return _render_relationships(request, unknown_spatial_relationships("request_body_unavailable"), status_code=400)
    if message.get("type") != "http.request" or message.get("body") or message.get("more_body"):
        return _render_relationships(request, unknown_spatial_relationships("query_or_body_not_allowed"), status_code=400)
    if not _READERS.acquire(blocking=False):
        return _render_relationships(request, unknown_spatial_relationships("reader_limit_exceeded"))
    try:
        return await run_in_threadpool(_read_relationships, request)
    finally:
        _READERS.release()
