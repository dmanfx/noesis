"""Private, immutable measurement proposals. This module never applies a scale."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from noesis.server.boundary_metrics import mark_rest_response, measure_rest_response_model

from noesis_core.private_paths import (
    PrivatePathError, ensure_private_directory, prepare_private_writable_file,
    validate_private_file,
)
from noesis.validation.alignment_walk import validate_scene_binding, _target_owned_context

MAX_BODY = 65536
MAX_ACTOR_RECORDS = 256
MAX_RECORDS = 4096
MAX_STORED_BYTES = 64 * 1024 * 1024


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _uuid(value: Any) -> str:
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("request_id must be a canonical UUID")
    return value


def _text(value: Any, maximum: int, name: str, *, multiline: bool = False) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum or not value.strip() or any(ord(c) < 32 and not (multiline and c in "\n\t") for c in value):
        raise ValueError(f"{name} is invalid")
    return value


def normalize_request(raw: Any) -> tuple[dict[str, Any], float, float]:
    keys = {"request_id", "actor_id", "scene_binding", "points_scene", "distance_metres", "note"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("metric scale request fields are invalid")
    _uuid(raw["request_id"]); _text(raw["actor_id"], 160, "actor_id"); _text(raw["note"], 500, "note", multiline=True)
    if raw["note"] != raw["note"].strip():
        raise ValueError("note must already be trimmed")
    binding = raw["scene_binding"]
    if not isinstance(binding, dict) or type(binding.get("contract_version")) is not int or binding["contract_version"] != 2:
        raise ValueError("scene_binding requires exact V2 target authority")
    if validate_scene_binding(binding) != binding:
        raise ValueError("scene_binding must be exact")
    points = raw["points_scene"]
    if not isinstance(points, list) or len(points) != 2 or any(
        not isinstance(point, list) or len(point) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e9 for v in point)
        for point in points
    ):
        raise ValueError("points_scene requires two finite XYZ points in menon_scene")
    metres = raw["distance_metres"]
    if type(metres) not in (int, float) or not math.isfinite(metres) or not 0 < metres <= 100000:
        raise ValueError("distance_metres requires a positive finite number")
    distance = math.dist(*points)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("points_scene must be distinct with finite distance")
    proposed = metres / distance
    if not math.isfinite(proposed) or proposed <= 0:
        raise ValueError("proposed scale is not finite and positive")
    return json.loads(_json(raw)), distance, proposed


def _path() -> Path:
    from noesis.server.scene_api import _database_path
    return _database_path().parent / "metric-scale-requests" / "requests.sqlite3"


def _connect(*, create: bool) -> sqlite3.Connection | None:
    path = _path()
    if create:
        ensure_private_directory(path.parent, label="metric scale proposal directory")
        path = prepare_private_writable_file(path, label="metric scale proposal database")
        connection = sqlite3.connect(path, timeout=2)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS metric_scale_requests (request_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, fingerprint TEXT NOT NULL, response TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS metric_scale_actor ON metric_scale_requests(actor_id)")
            connection.commit()
        except BaseException:
            connection.close()
            raise
    else:
        if not path.exists():
            return None
        path = validate_private_file(path, label="metric scale proposal database")
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
        try:
            connection.execute("PRAGMA query_only=ON")
        except BaseException:
            connection.close()
            raise
    return connection


def _read(connection: sqlite3.Connection, request_id: str, actor_id: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT fingerprint,response FROM metric_scale_requests WHERE request_id=? AND actor_id=?", (request_id, actor_id)).fetchone()
    if row is None:
        return None
    value = json.loads(row[1])
    if not isinstance(value, dict) or set(value) != {"contract", "contract_version", "request_id", "actor_id", "created_at_us", "status", "applied", "request", "scene_distance", "proposed_metres_per_scene_unit"} or not isinstance(value.get("request"), dict) or set(value["request"]) != {"scene_binding", "points_scene", "distance_metres", "note"}:
        raise ValueError("invalid saved metric scale proposal fields")
    original = {"request_id": value["request_id"], "actor_id": value["actor_id"], **value["request"]}
    _, distance, proposed = normalize_request(original)
    if (hashlib.sha256(_json(original).encode("utf-8")).hexdigest() != row[0]
        or value.get("contract") != "noesis.scene.metric_scale_request" or type(value.get("contract_version")) is not int or value.get("contract_version") != 1
        or value.get("status") != "submitted_for_review" or value.get("applied") is not False
        or value.get("request_id") != request_id or value.get("actor_id") != actor_id
        or type(value.get("created_at_us")) is not int or value["created_at_us"] <= 0
        or type(value.get("scene_distance")) not in (int, float) or value["scene_distance"] != distance
        or type(value.get("proposed_metres_per_scene_unit")) not in (int, float) or value["proposed_metres_per_scene_unit"] != proposed):
        raise ValueError("invalid saved metric scale proposal")
    return value


def _validate_current(request: Request, binding: dict[str, Any]) -> None:
    from noesis.server.scene_api import _current_snapshot
    from noesis.server.scene_prior_api import _runtime_components
    try:
        _, release = _current_snapshot()
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(409, detail="metric_scale_current_scene_unavailable") from exc
        raise
    try:
        _, calibration, labels = _runtime_components(request)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(503, detail="metric_scale_authority_unavailable") from exc
        raise
    except Exception as exc:
        raise HTTPException(503, detail="metric_scale_authority_unavailable") from exc
    if not all(isinstance(label, str) for label in labels.values()):
        raise HTTPException(503, detail="metric_scale_authority_unavailable")
    if not set(binding["cameras"]).issubset({camera.camera_id for camera in release.cameras}) or not set(binding["cameras"]).issubset(set(labels.values())):
        raise HTTPException(409, detail="scene_binding_mismatch")
    method = getattr(calibration, "calibration_bundle", None)
    if not callable(method):
        raise HTTPException(503, detail="metric_scale_authority_unavailable")
    try:
        bundle = method()
    except Exception as exc:
        raise HTTPException(503, detail="metric_scale_authority_unavailable") from exc
    try:
        _target_owned_context(bundle, binding, {"authority": "noesis.scenes.current", "release_id": release.release_id,
            "authored_scene_sha256": release.authored_scene.sha256})
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(409, detail="metric_scale_scene_binding_unavailable") from exc


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate request field")
        result[key] = value
    return result


def _response(request: Request, value: dict[str, Any], status: int = 200) -> JSONResponse:
    route = "/api/v1/scenes/current/metric-scale-requests:post" if request.method == "POST" else "/api/v1/scenes/metric-scale-requests/{request_id}:get"
    with measure_rest_response_model(route, "MetricScaleRequest") as measurement:
        response = JSONResponse(value, status_code=status, headers={"Cache-Control": "no-store"})
    mark_rest_response(request, route, "MetricScaleRequest", model_duration_ms=measurement.elapsed_ms)
    return response


async def submit(request: Request) -> JSONResponse:
    if request.query_params:
        raise HTTPException(422, detail="metric_scale_request_invalid")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BODY:
            raise HTTPException(413, detail="metric_scale_request_too_large")
    try:
        raw = json.loads(data, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        value, distance, proposed = normalize_request(raw)
        fingerprint = hashlib.sha256(_json(value).encode("utf-8")).hexdigest()
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(422, detail="metric_scale_request_invalid") from exc
    return await run_in_threadpool(_persist, request, value, distance, proposed, fingerprint)


def _persist(request: Request, value: dict[str, Any], distance: float, proposed: float, fingerprint: str) -> JSONResponse:
    connection = None
    try:
        # Recovery of an already admitted exact request does not claim a current binding.
        connection = _connect(create=False)
        if connection is not None:
            existing = connection.execute("SELECT actor_id,fingerprint FROM metric_scale_requests WHERE request_id=?", (value["request_id"],)).fetchone()
            if existing:
                if existing != (value["actor_id"], fingerprint):
                    raise HTTPException(409, detail="metric_scale_request_conflict")
                return _response(request, _read(connection, value["request_id"], value["actor_id"]))
            connection.close(); connection = None
        _validate_current(request, value["scene_binding"])
        connection = _connect(create=True)
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute("SELECT actor_id,fingerprint FROM metric_scale_requests WHERE request_id=?", (value["request_id"],)).fetchone()
        if existing:
            if existing != (value["actor_id"], fingerprint):
                raise HTTPException(409, detail="metric_scale_request_conflict")
            return _response(request, _read(connection, value["request_id"], value["actor_id"]))
        response = {"contract": "noesis.scene.metric_scale_request", "contract_version": 1,
            "request_id": value["request_id"], "actor_id": value["actor_id"], "created_at_us": time.time_ns() // 1000,
            "status": "submitted_for_review", "applied": False,
            "request": {key: value[key] for key in ("scene_binding", "points_scene", "distance_metres", "note")},
            "scene_distance": distance, "proposed_metres_per_scene_unit": proposed}
        encoded = _json(response)
        count, size = connection.execute("SELECT count(*), coalesce(sum(length(CAST(response AS BLOB))),0) FROM metric_scale_requests").fetchone()
        actor_count = connection.execute("SELECT count(*) FROM metric_scale_requests WHERE actor_id=?", (value["actor_id"],)).fetchone()[0]
        if count >= MAX_RECORDS or actor_count >= MAX_ACTOR_RECORDS or size + len(encoded.encode("utf-8")) > MAX_STORED_BYTES:
            raise HTTPException(409, detail="metric_scale_request_capacity")
        _validate_current(request, value["scene_binding"])
        connection.execute("INSERT INTO metric_scale_requests VALUES (?,?,?,?)", (value["request_id"], value["actor_id"], fingerprint, encoded))
        saved = _read(connection, value["request_id"], value["actor_id"])
        if saved != response:
            raise ValueError("metric scale proposal readback differs")
        connection.commit()
        return _response(request, saved, 201)
    except (sqlite3.Error, PrivatePathError, OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, detail="metric_scale_request_store_unavailable") from exc
    finally:
        if connection is not None:
            connection.close()


def lookup(request: Request, request_id: str) -> JSONResponse:
    try:
        _uuid(request_id)
        if list(request.query_params.keys()) != ["actor_id"] or len(request.query_params.getlist("actor_id")) != 1:
            raise ValueError("actor_id required")
        actor = _text(request.query_params["actor_id"], 160, "actor_id")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail="metric_scale_request_invalid") from exc
    connection = None
    try:
        connection = _connect(create=False)
        saved = None if connection is None else _read(connection, request_id, actor)
        if saved is None:
            raise HTTPException(404, detail="metric_scale_request_not_found")
        return _response(request, saved)
    except (sqlite3.Error, PrivatePathError, OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, detail="metric_scale_request_store_unavailable") from exc
    finally:
        if connection is not None:
            connection.close()
