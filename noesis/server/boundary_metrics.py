from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Dict, Optional


_LOCK = threading.Lock()
_SAMPLES_MS = deque(maxlen=4096)
_COUNT = 0
_TOTAL_MS = 0.0
_MAX_MS = 0.0
_LAST_MS = 0.0
_TOTAL_BYTES = 0
_LAST_BYTES = 0
_BUDGET_MS = 3.0
_VIOLATIONS = 0
_ROUTE_METRICS: Dict[str, Dict[str, Any]] = {}
_STAGE_METRICS: Dict[str, Dict[str, Any]] = {}


def _new_bucket(maxlen: int = 1024) -> Dict[str, Any]:
    return {
        "samples": deque(maxlen=maxlen),
        "count": 0,
        "total_ms": 0.0,
        "max_ms": 0.0,
        "last_ms": 0.0,
        "total_bytes": 0,
        "last_bytes": 0,
    }


def _update_bucket(bucket: Dict[str, Any], duration_ms: float, payload_bytes: int) -> None:
    bucket["samples"].append(float(duration_ms))
    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket["total_ms"] = float(bucket.get("total_ms", 0.0)) + float(duration_ms)
    bucket["max_ms"] = max(float(bucket.get("max_ms", 0.0)), float(duration_ms))
    bucket["last_ms"] = float(duration_ms)
    bucket["total_bytes"] = int(bucket.get("total_bytes", 0)) + int(payload_bytes)
    bucket["last_bytes"] = int(payload_bytes)


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = int(round(q * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def record_boundary_stage(
    duration_ms: float,
    payload_bytes: int = 0,
    *,
    channel: str = "rest",
    route: str,
    message_type: str = "response",
    stage: str,
    outcome: str = "ok",
    include_budget: bool = False,
) -> None:
    global _COUNT, _TOTAL_MS, _MAX_MS, _LAST_MS, _TOTAL_BYTES, _LAST_BYTES, _VIOLATIONS
    d = max(0.0, float(duration_ms))
    b = max(0, int(payload_bytes))
    with _LOCK:
        if include_budget and stage == "total":
            _SAMPLES_MS.append(d)
            _COUNT += 1
            _TOTAL_MS += d
            _MAX_MS = max(_MAX_MS, d)
            _LAST_MS = d
            _TOTAL_BYTES += b
            _LAST_BYTES = b
            if d > float(_BUDGET_MS):
                _VIOLATIONS += 1

        route_key = f"{channel}|{route}|{message_type}|{outcome}"
        route_bucket = _ROUTE_METRICS.get(route_key)
        if route_bucket is None:
            route_bucket = _new_bucket()
            _ROUTE_METRICS[route_key] = route_bucket
        _update_bucket(route_bucket, d, b)

        stage_key = f"{channel}|{route}|{message_type}|{stage}|{outcome}"
        stage_bucket = _STAGE_METRICS.get(stage_key)
        if stage_bucket is None:
            stage_bucket = _new_bucket()
            _STAGE_METRICS[stage_key] = stage_bucket
        _update_bucket(stage_bucket, d, b)


def record_rest_response(
    route: str,
    message_type: str,
    *,
    model_duration_ms: float,
    payload: Any,
    include_budget: bool = True,
) -> None:
    record_boundary_stage(
        model_duration_ms,
        0,
        channel="rest",
        route=route,
        message_type=message_type,
        stage="response_model",
        outcome="ok",
        include_budget=False,
    )

    encode_start_ns = time.perf_counter_ns()
    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload
    encoded = json.dumps(payload_dict, separators=(",", ":"), default=str)
    payload_bytes = len(encoded.encode("utf-8"))
    encode_ms = (time.perf_counter_ns() - encode_start_ns) / 1_000_000.0
    record_boundary_stage(
        encode_ms,
        payload_bytes,
        channel="rest",
        route=route,
        message_type=message_type,
        stage="json_encode",
        outcome="ok",
        include_budget=False,
    )

    total_ms = float(model_duration_ms + encode_ms)
    record_boundary_stage(
        total_ms,
        payload_bytes,
        channel="rest",
        route=route,
        message_type=message_type,
        stage="total",
        outcome="ok",
        include_budget=include_budget,
    )


def get_boundary_serialization_metrics() -> Dict[str, Any]:
    with _LOCK:
        samples = list(_SAMPLES_MS)
        count = int(_COUNT)
        total_ms = float(_TOTAL_MS)
        max_ms = float(_MAX_MS)
        last_ms = float(_LAST_MS)
        total_bytes = int(_TOTAL_BYTES)
        last_bytes = int(_LAST_BYTES)
        violations = int(_VIOLATIONS)
        route_snapshot = {
            str(k): {
                "samples": list(v.get("samples", [])),
                "count": int(v.get("count", 0)),
                "total_ms": float(v.get("total_ms", 0.0)),
                "max_ms": float(v.get("max_ms", 0.0)),
                "last_ms": float(v.get("last_ms", 0.0)),
                "total_bytes": int(v.get("total_bytes", 0)),
                "last_bytes": int(v.get("last_bytes", 0)),
            }
            for k, v in _ROUTE_METRICS.items()
        }
        stage_snapshot = {
            str(k): {
                "samples": list(v.get("samples", [])),
                "count": int(v.get("count", 0)),
                "total_ms": float(v.get("total_ms", 0.0)),
                "max_ms": float(v.get("max_ms", 0.0)),
                "last_ms": float(v.get("last_ms", 0.0)),
                "total_bytes": int(v.get("total_bytes", 0)),
                "last_bytes": int(v.get("last_bytes", 0)),
            }
            for k, v in _STAGE_METRICS.items()
        }

    def _summarize(snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key, item in snapshot.items():
            values = [float(x) for x in item.get("samples", [])]
            icount = int(item.get("count", 0))
            itotal = float(item.get("total_ms", 0.0))
            out[key] = {
                "count": icount,
                "avg_ms": (itotal / float(icount)) if icount > 0 else None,
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
                "max_ms": float(item.get("max_ms", 0.0)) if icount > 0 else None,
                "last_ms": float(item.get("last_ms", 0.0)) if icount > 0 else None,
                "total_bytes": int(item.get("total_bytes", 0)),
                "last_payload_bytes": int(item.get("last_bytes", 0)),
            }
        return out

    avg_ms = (total_ms / float(count)) if count > 0 else None
    return {
        "count": count,
        "avg_ms": avg_ms,
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "max_ms": max_ms if count > 0 else None,
        "last_ms": last_ms if count > 0 else None,
        "total_bytes": total_bytes,
        "last_payload_bytes": last_bytes,
        "budget_ms": float(_BUDGET_MS),
        "violations": violations,
        "routes": _summarize(route_snapshot),
        "stages": _summarize(stage_snapshot),
    }


def reset_boundary_serialization_metrics() -> None:
    global _COUNT, _TOTAL_MS, _MAX_MS, _LAST_MS, _TOTAL_BYTES, _LAST_BYTES, _VIOLATIONS
    with _LOCK:
        _SAMPLES_MS.clear()
        _COUNT = 0
        _TOTAL_MS = 0.0
        _MAX_MS = 0.0
        _LAST_MS = 0.0
        _TOTAL_BYTES = 0
        _LAST_BYTES = 0
        _VIOLATIONS = 0
        _ROUTE_METRICS.clear()
        _STAGE_METRICS.clear()
