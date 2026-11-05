"""Private bounded view-analysis route on the existing scene-prior surface."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Mapping
from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from noesis.server.boundary_metrics import BoundaryMetricsRoute, mark_rest_response, measure_rest_response_model
from noesis_core.contracts.view_coverage import ViewAnalysisRequest, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES
from noesis_core.view_coverage import ViewCoverageService, ViewCoverageError, ViewCoverageRetired

PATH = "/api/v1/scene-priors/cameras/{camera_id}/view-analysis"
router = APIRouter(route_class=BoundaryMetricsRoute)
_workers = threading.BoundedSemaphore(2)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite JSON")


async def _body(request, read_only, model=ViewAnalysisRequest):
    if request.scope.get("query_string") or request.headers.get("transfer-encoding"):
        raise HTTPException(422, detail="view_analysis_query_or_transfer_denied")
    if not read_only and request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, detail="view_analysis_json_required")
    data = bytearray()
    async for chunk in request.stream():
        if read_only and chunk:
            raise HTTPException(422, detail="view_analysis_get_body_denied")
        if len(data) + len(chunk) > MAX_REQUEST_BYTES:
            raise HTTPException(413, detail="view_analysis_request_too_large")
        data.extend(chunk)
    if read_only:
        return None
    try:
        return model.model_validate(json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite))
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise HTTPException(422, detail="view_analysis_request_invalid") from error


def _perform(request, camera_id, command, canceled, deadline):
    try:
        from noesis.server.scene_prior_api import _runtime_components
        provider = getattr(request.app.state, "scene_prior_runtime_provider", None)
        priors, calibration, labels = _runtime_components(request)
        if not isinstance(labels, Mapping) or not 1 <= len(labels) <= 64 or any(type(key) is not int or key < 0 or type(value) is not str for key, value in labels.items()):
            raise HTTPException(503, detail="view_analysis_camera_binding_unavailable")
        original_labels = tuple(sorted(labels.items()))
        sources = [source for source, camera in original_labels if camera == camera_id]
        if len(sources) != 1:
            raise HTTPException(404, detail="view_analysis_camera_not_active")
        source = sources[0]
        snapshot = calibration.snapshot(source, camera_id)
        if snapshot is None:
            raise HTTPException(503, detail="view_analysis_calibration_unavailable")
        revision, frame = priors.revision_for_camera(camera_id), priors.frame_binding(camera_id)
        intrinsics = tuple(float(value) for value in snapshot.intrinsics.ravel())
        def current():
            try:
                if canceled.is_set() or time.monotonic() >= deadline or getattr(request.app.state, "scene_prior_runtime_provider", None) is not provider:
                    return False
                next_priors, next_calibration, next_labels = _runtime_components(request)
                return (next_priors is priors and next_calibration is calibration and tuple(sorted(next_labels.items())) == original_labels
                    and calibration.snapshot(source, camera_id) is snapshot and priors.revision_for_camera(camera_id) is revision
                    and priors.frame_binding(camera_id) is frame and tuple(float(value) for value in snapshot.intrinsics.ravel()) == intrinsics)
            except Exception:
                return False
        from noesis_core.contracts.room_coverage import RoomCoverageRequest
        from noesis_core.room_coverage import RoomCoverageService
        domain = isinstance(command, RoomCoverageRequest)
        service = (RoomCoverageService if domain else ViewCoverageService)(priors, snapshot, camera_id, is_current=current, deadline=deadline)
        result = service.context() if command is None else service.analyze(command)
        route = PATH.replace('view-analysis', 'domain-coverage') if domain else PATH + (":context" if command is None else ":post")
        with measure_rest_response_model(route, type(result).__name__) as measurement:
            response = JSONResponse(result.model_dump(mode="json"), headers={"Cache-Control": "private, no-store"})
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise HTTPException(503, detail="view_analysis_response_too_large")
        if not current():
            raise ViewCoverageRetired("view_analysis_owner_retired")
        mark_rest_response(request, route, type(result).__name__, model_duration_ms=measurement.elapsed_ms)
        return response
    except ViewCoverageRetired as error:
        raise HTTPException(409, detail="view_analysis_owner_retired") from error
    except ViewCoverageError as error:
        raise HTTPException(409, detail=str(error)) from error
    except (ValidationError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError) as error:
        raise HTTPException(503, detail="view_analysis_source_unavailable") from error
    finally:
        # The WORKER owns the slot after submission. A canceled HTTP request
        # cannot release it while its finite CPU work is still in progress.
        _workers.release()


async def _handle(request, camera_id, read_only, model=ViewAnalysisRequest):
    if not _workers.acquire(blocking=False):
        raise HTTPException(503, detail="view_analysis_busy", headers={"Retry-After": "1"})
    submitted = False
    canceled = threading.Event()
    try:
        async with asyncio.timeout(2):
            command = await _body(request, read_only, model)
        submitted = True
        deadline = time.monotonic() + 3
        async with asyncio.timeout(4):
            return await run_in_threadpool(_perform, request, camera_id, command, canceled, deadline)
    except TimeoutError as error:
        raise HTTPException(408, detail="view_analysis_request_timeout") from error
    finally:
        canceled.set()
        if not submitted:
            _workers.release()


@router.get(PATH)
async def view_analysis_context(request: Request, camera_id: str = Path(..., min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")):
    return await _handle(request, camera_id, True)


@router.post(PATH)
async def view_analysis(request: Request, camera_id: str = Path(..., min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")):
    return await _handle(request, camera_id, False)
