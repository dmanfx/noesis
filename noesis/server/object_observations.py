"""Fixed private read, protected by the runtime's existing internal auth."""
from __future__ import annotations

import asyncio
import json
from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from noesis_core.object_observations import ObjectObservationService
from noesis_core.contracts.object_observations import MAX_BYTES, TTL_MS
from noesis.server.boundary_metrics import BoundaryMetricsRoute, mark_rest_response, measure_rest_response_model

PATH = "/api/v1/objects/observations"
router = APIRouter(route_class=BoundaryMetricsRoute)


def _unknown(reason):
    return {"contract": "noesis.object-observations", "version": 1, "status": "unknown", "reason": reason,
        "run_id": None, "generated_at_us": None, "ttl_ms": TTL_MS, "dropped_frames": 0, "cameras": []}


def _encode(value):
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_BYTES:
        body = json.dumps(_unknown("response_byte_limit")).encode("utf-8")
    return body


def _render(request, value, status_code=200):
    with measure_rest_response_model(PATH + ":get", "ObjectObservationsRead") as measurement:
        body = _encode(value)
    mark_rest_response(request, PATH + ":get", "ObjectObservationsRead", model_duration_ms=measurement.elapsed_ms)
    return Response(body, status_code=status_code, media_type="application/json", headers={"Cache-Control": "private, no-store"})


def _read(request):
    owner = getattr(request.app.state, "object_observation_service", None)
    if type(owner) is not ObjectObservationService:
        return _render(request, _unknown("unbound"))
    value = owner.snapshot()
    if getattr(request.app.state, "object_observation_service", None) is not owner:
        return _render(request, _unknown("owner_changed"))
    with measure_rest_response_model(PATH + ":get", "ObjectObservationsRead") as measurement:
        body = _encode(value)
    if getattr(request.app.state, "object_observation_service", None) is not owner or owner._closed:
        return _render(request, _unknown("owner_changed"))
    mark_rest_response(request, PATH + ":get", "ObjectObservationsRead", model_duration_ms=measurement.elapsed_ms)
    return Response(body, media_type="application/json", headers={"Cache-Control": "private, no-store"})


@router.get(PATH)
async def observations(request: Request):
    if request.scope.get("query_string") or request.headers.get("transfer-encoding") or request.headers.get("content-length") not in {None, "0"}:
        return _render(request, _unknown("query_or_body_not_allowed"), 400)
    try:
        message = await asyncio.wait_for(request.receive(), timeout=0.1)
    except (TimeoutError, asyncio.TimeoutError):
        return _render(request, _unknown("request_body_unavailable"), 400)
    if message.get("type") != "http.request" or message.get("body") or message.get("more_body"):
        return _render(request, _unknown("query_or_body_not_allowed"), 400)
    return await run_in_threadpool(_read, request)


# Keep registration on the already installed object-observation API surface.
from noesis.server.object_associations import router as object_associations_router
router.include_router(object_associations_router)

from noesis.server.object_tracks import router as object_tracks_router
router.include_router(object_tracks_router)
