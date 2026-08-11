from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.exceptions import ResponseValidationError
from fastapi.routing import APIRoute
from starlette.responses import Response

_LOCK = threading.Lock()
_SAMPLES_MS = deque()
_GLOBAL_SAMPLE_SATURATION: Dict[str, Any] = {
    "dropped_total": 0,
    "active_dropped": 0,
    "last_drop_s": None,
    "max_dropped_ms": 0.0,
}
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
_BUDGET_PATH_SAMPLES: Dict[str, deque] = {}
_BUDGET_PATH_SATURATION: Dict[str, Dict[str, Any]] = {}
_ERROR_COUNTS: Dict[str, int] = {}
_REST_CONTEXT_STATE_KEY = "_noesis_rest_boundary_context"
_MAX_DETAIL_BUCKETS = 512
_MAX_ERROR_BUCKETS = 512
_MAX_WINDOW_SAMPLES = 4096
_TOP_BUDGET_PATHS_LIMIT = 8
_ROUTE_OVERFLOW_KEY = "rest|__overflow__|__overflow__|__overflow__"
_STAGE_OVERFLOW_KEY = "rest|__overflow__|__overflow__|__overflow__|__overflow__"
_ERROR_OVERFLOW_KEY = "rest|__overflow__|__overflow__|__overflow__|__overflow__"
_CLOCK = time.monotonic
_WINDOWS_S = (10.0, 60.0)
_MAX_WINDOW_S = max(_WINDOWS_S)
_SATURATION_FAIL_CLOSED_MS = _BUDGET_MS + 0.001


@dataclass(frozen=True)
class _RestResponseContext:
    route: str
    message_type: str
    model_duration_ms: float
    marked_at_ns: int
    include_budget: bool


@dataclass(frozen=True)
class _RestResponseExemption:
    reason: str


@dataclass
class RestResponseModelMeasurement:
    route: str
    message_type: str
    duration_ms: Optional[float] = None
    _started_at_ns: Optional[int] = None

    @property
    def elapsed_ms(self) -> float:
        if self.duration_ms is None:
            raise RuntimeError("REST response-model measurement is incomplete")
        return float(self.duration_ms)

    def __enter__(self) -> RestResponseModelMeasurement:
        self._started_at_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, _traceback) -> bool:  # type: ignore[no-untyped-def]
        started_at_ns = self._started_at_ns
        if started_at_ns is None:
            raise RuntimeError("REST response-model measurement was not started")
        self.duration_ms = max(
            0.0,
            (time.perf_counter_ns() - started_at_ns) / 1_000_000.0,
        )
        if exc is not None:
            _record_rest_boundary_error(
                route=self.route,
                message_type=self.message_type,
                stage="response_model",
                error=exc,
            )
        return False


class BoundaryMetricsRoute(APIRoute):
    """Measure the response FastAPI actually validated, rendered, and returned.

    Endpoint handlers mark the end of their response-model construction on the
    request.  FastAPI then performs its normal response-model filtering,
    aliasing, exclusion, and JSON rendering exactly once.  This wrapper observes
    the completed response body; it never serializes a surrogate payload.
    """

    def get_route_handler(self):  # type: ignore[no-untyped-def]
        route_handler = super().get_route_handler()

        async def measured_route_handler(request: Request) -> Response:
            try:
                response = await route_handler(request)
            except Exception as exc:
                context = getattr(request.state, _REST_CONTEXT_STATE_KEY, None)
                if isinstance(context, _RestResponseContext):
                    stage = (
                        "response_model"
                        if isinstance(exc, ResponseValidationError)
                        else (
                            "local_response"
                            if isinstance(exc, RuntimeError)
                            and "marked more than once" in str(exc)
                            else "json_encode"
                        )
                    )
                    _record_rest_boundary_error(
                        route=context.route,
                        message_type=context.message_type,
                        stage=stage,
                        error=exc,
                    )
                raise
            context = getattr(request.state, _REST_CONTEXT_STATE_KEY, None)
            if isinstance(context, _RestResponseExemption):
                return response
            if context is None:
                error = RuntimeError(
                    f"Successful REST route {self.path} did not declare boundary measurement or exemption"
                )
                _record_rest_boundary_error(
                    route=str(self.path),
                    message_type="unmarked_response",
                    stage="missing_boundary_context",
                    error=error,
                )
                raise error
            if not isinstance(context, _RestResponseContext):
                raise RuntimeError("Invalid REST boundary measurement context")

            finished_at_ns = time.perf_counter_ns()
            try:
                body = response.body
            except Exception as exc:
                _record_rest_boundary_error(
                    route=context.route,
                    message_type=context.message_type,
                    stage="local_response",
                    error=exc,
                )
                raise
            if not isinstance(body, (bytes, bytearray, memoryview)):
                error = RuntimeError(
                    "Measured REST response did not produce a bounded byte body"
                )
                _record_rest_boundary_error(
                    route=context.route,
                    message_type=context.message_type,
                    stage="local_response",
                    error=error,
                )
                raise error
            payload_bytes = len(body)
            render_ms = max(
                0.0,
                (finished_at_ns - context.marked_at_ns) / 1_000_000.0,
            )
            _record_rest_response(
                context,
                render_duration_ms=render_ms,
                payload_bytes=payload_bytes,
            )
            return response

        return measured_route_handler


def _new_bucket() -> Dict[str, Any]:
    return {
        "samples": deque(),
        "dropped_total": 0,
        "active_dropped": 0,
        "last_drop_s": None,
        "max_dropped_ms": 0.0,
        "count": 0,
        "total_ms": 0.0,
        "max_ms": 0.0,
        "last_ms": 0.0,
        "total_bytes": 0,
        "last_bytes": 0,
    }


def _update_bucket(
    bucket: Dict[str, Any],
    duration_ms: float,
    payload_bytes: int,
    observed_at_s: float,
) -> None:
    _append_window_sample(
        bucket["samples"],
        bucket,
        observed_at_s=observed_at_s,
        duration_ms=duration_ms,
        payload_bytes=payload_bytes,
    )
    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket["total_ms"] = float(bucket.get("total_ms", 0.0)) + float(duration_ms)
    bucket["max_ms"] = max(float(bucket.get("max_ms", 0.0)), float(duration_ms))
    bucket["last_ms"] = float(duration_ms)
    bucket["total_bytes"] = int(bucket.get("total_bytes", 0)) + int(payload_bytes)
    bucket["last_bytes"] = int(payload_bytes)


def _prune_samples(samples: deque, now_s: float) -> None:
    cutoff_s = float(now_s) - float(_MAX_WINDOW_S)
    while samples and float(samples[0][0]) < cutoff_s:
        samples.popleft()


def _new_saturation_state() -> Dict[str, Any]:
    return {
        "dropped_total": 0,
        "active_dropped": 0,
        "last_drop_s": None,
        "max_dropped_ms": 0.0,
    }


def _refresh_saturation(state: Dict[str, Any], now_s: float) -> None:
    last_drop_s = state.get("last_drop_s")
    if last_drop_s is None:
        return
    if float(last_drop_s) < (float(now_s) - float(_MAX_WINDOW_S)):
        state["active_dropped"] = 0
        state["last_drop_s"] = None
        state["max_dropped_ms"] = 0.0


def _append_window_sample(
    samples: deque,
    saturation: Dict[str, Any],
    *,
    observed_at_s: float,
    duration_ms: float,
    payload_bytes: int,
) -> None:
    _prune_samples(samples, observed_at_s)
    _refresh_saturation(saturation, observed_at_s)
    if len(samples) < int(_MAX_WINDOW_SAMPLES):
        samples.append((float(observed_at_s), float(duration_ms), int(payload_bytes)))
        return

    saturation["dropped_total"] = int(saturation.get("dropped_total", 0)) + 1
    saturation["active_dropped"] = int(saturation.get("active_dropped", 0)) + 1
    saturation["last_drop_s"] = float(observed_at_s)
    saturation["max_dropped_ms"] = max(
        float(saturation.get("max_dropped_ms", 0.0)),
        float(duration_ms),
    )


def _bounded_metric_key(
    metrics: Dict[str, Dict[str, Any]],
    desired_key: str,
    overflow_key: str,
) -> str:
    if desired_key in metrics:
        return desired_key
    if len(metrics) < (_MAX_DETAIL_BUCKETS - 1):
        return desired_key
    return overflow_key


def _record_rest_boundary_error(
    *,
    route: str,
    message_type: str,
    stage: str,
    error: BaseException,
) -> None:
    desired_key = f"rest|{route}|{message_type}|{stage}|{type(error).__name__}"
    with _LOCK:
        if desired_key in _ERROR_COUNTS:
            key = desired_key
        elif len(_ERROR_COUNTS) < (_MAX_ERROR_BUCKETS - 1):
            key = desired_key
        else:
            key = _ERROR_OVERFLOW_KEY
        _ERROR_COUNTS[key] = int(_ERROR_COUNTS.get(key, 0)) + 1


def measure_rest_response_model(
    route: str,
    message_type: str,
) -> RestResponseModelMeasurement:
    """Time one endpoint-owned model assembly and count only its failures."""

    return RestResponseModelMeasurement(
        route=str(route),
        message_type=str(message_type),
    )


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(values)
    return _percentile_from_ordered(ordered, q)


def _percentile_from_ordered(ordered: list[float], q: float) -> Optional[float]:
    if not ordered:
        return None
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
    observed_at_s = float(_CLOCK())
    with _LOCK:
        _prune_samples(_SAMPLES_MS, observed_at_s)
        if include_budget and stage == "total":
            _append_window_sample(
                _SAMPLES_MS,
                _GLOBAL_SAMPLE_SATURATION,
                observed_at_s=observed_at_s,
                duration_ms=d,
                payload_bytes=b,
            )
            _COUNT += 1
            _TOTAL_MS += d
            _MAX_MS = max(_MAX_MS, d)
            _LAST_MS = d
            _TOTAL_BYTES += b
            _LAST_BYTES = b
            if d > float(_BUDGET_MS):
                _VIOLATIONS += 1

        if stage == "total":
            route_key = _bounded_metric_key(
                _ROUTE_METRICS,
                f"{channel}|{route}|{message_type}|{outcome}",
                _ROUTE_OVERFLOW_KEY,
            )
            route_bucket = _ROUTE_METRICS.get(route_key)
            if route_bucket is None:
                route_bucket = _new_bucket()
                _ROUTE_METRICS[route_key] = route_bucket
            _prune_samples(route_bucket["samples"], observed_at_s)
            _update_bucket(route_bucket, d, b, observed_at_s)

        stage_key = _bounded_metric_key(
            _STAGE_METRICS,
            f"{channel}|{route}|{message_type}|{stage}|{outcome}",
            _STAGE_OVERFLOW_KEY,
        )
        stage_bucket = _STAGE_METRICS.get(stage_key)
        if stage_bucket is None:
            stage_bucket = _new_bucket()
            _STAGE_METRICS[stage_key] = stage_bucket
        _prune_samples(stage_bucket["samples"], observed_at_s)
        _update_bucket(stage_bucket, d, b, observed_at_s)
        if include_budget and stage == "total":
            budget_samples = _BUDGET_PATH_SAMPLES.get(stage_key)
            if budget_samples is None:
                budget_samples = deque()
                _BUDGET_PATH_SAMPLES[stage_key] = budget_samples
                _BUDGET_PATH_SATURATION[stage_key] = _new_saturation_state()
            _append_window_sample(
                budget_samples,
                _BUDGET_PATH_SATURATION[stage_key],
                observed_at_s=observed_at_s,
                duration_ms=d,
                payload_bytes=b,
            )


def mark_rest_response(
    request: Request,
    route: str,
    message_type: str,
    *,
    model_duration_ms: float,
    include_budget: bool = True,
) -> None:
    """Mark a REST response for measurement after FastAPI's one real render."""

    if getattr(request.state, _REST_CONTEXT_STATE_KEY, None) is not None:
        raise RuntimeError("REST response boundary was marked more than once")
    context = _RestResponseContext(
        route=str(route),
        message_type=str(message_type),
        model_duration_ms=max(0.0, float(model_duration_ms)),
        marked_at_ns=time.perf_counter_ns(),
        include_budget=bool(include_budget),
    )
    setattr(request.state, _REST_CONTEXT_STATE_KEY, context)


def mark_rest_response_exempt(request: Request, *, reason: str) -> None:
    """Declare one successful pre-rendered/file response outside JSON timing."""

    normalized_reason = str(reason or "").strip()
    if not normalized_reason:
        raise ValueError("REST boundary exemption requires a reason")
    if getattr(request.state, _REST_CONTEXT_STATE_KEY, None) is not None:
        raise RuntimeError("REST response boundary was marked more than once")
    setattr(
        request.state,
        _REST_CONTEXT_STATE_KEY,
        _RestResponseExemption(reason=normalized_reason),
    )


def _record_rest_response(
    context: _RestResponseContext,
    *,
    render_duration_ms: float,
    payload_bytes: int,
) -> None:
    record_boundary_stage(
        context.model_duration_ms,
        0,
        channel="rest",
        route=context.route,
        message_type=context.message_type,
        stage="response_model",
        outcome="ok",
        include_budget=False,
    )

    record_boundary_stage(
        render_duration_ms,
        payload_bytes,
        channel="rest",
        route=context.route,
        message_type=context.message_type,
        stage="json_encode",
        outcome="ok",
        include_budget=False,
    )

    total_ms = float(context.model_duration_ms + render_duration_ms)
    record_boundary_stage(
        total_ms,
        payload_bytes,
        channel="rest",
        route=context.route,
        message_type=context.message_type,
        stage="total",
        outcome="ok",
        include_budget=context.include_budget,
    )


def _window_summaries(
    samples: list[tuple[float, float, int]],
    *,
    now_s: float,
    saturation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    # Sorting the full 60-second set once lets the timestamp-filtered 10-second
    # subset retain duration order, so each bucket incurs only one sort.
    ordered = sorted(samples, key=lambda item: item[1])
    summaries: Dict[str, Dict[str, Any]] = {}
    for window_s in _WINDOWS_S:
        cutoff_s = float(now_s) - float(window_s)
        window_ordered = [item for item in ordered if item[0] >= cutoff_s]
        durations = [float(item[1]) for item in window_ordered]
        chronological = [item for item in samples if item[0] >= cutoff_s]
        count = len(durations)
        total_ms = sum(durations)
        key = f"{int(window_s)}s"
        last_drop_s = (saturation or {}).get("last_drop_s")
        sample_saturated = last_drop_s is not None and float(last_drop_s) >= cutoff_s
        summaries[key] = {
            "window_sec": float(window_s),
            "count": count,
            "avg_ms": (total_ms / float(count)) if count > 0 else None,
            "p50_ms": _percentile_from_ordered(durations, 0.50),
            "p95_ms": _percentile_from_ordered(durations, 0.95),
            "p99_ms": _percentile_from_ordered(durations, 0.99),
            "max_ms": durations[-1] if durations else None,
            "last_ms": float(chronological[-1][1]) if chronological else None,
            "total_bytes": sum(int(item[2]) for item in chronological),
            "last_payload_bytes": int(chronological[-1][2]) if chronological else 0,
            "violations": sum(1 for value in durations if value > float(_BUDGET_MS)),
            "sample_saturated": sample_saturated,
            "sample_dropped_total": int((saturation or {}).get("dropped_total", 0)),
            "sample_active_dropped": (
                int((saturation or {}).get("active_dropped", 0))
                if sample_saturated
                else 0
            ),
            "sample_saturation_max_ms": (
                float((saturation or {}).get("max_dropped_ms", 0.0))
                if sample_saturated
                else None
            ),
        }
    return summaries


def _worst_optional(*values: Any) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def get_boundary_serialization_metrics(
    *,
    include_details: bool = True,
) -> Dict[str, Any]:
    now_s = float(_CLOCK())
    with _LOCK:
        _prune_samples(_SAMPLES_MS, now_s)
        _refresh_saturation(_GLOBAL_SAMPLE_SATURATION, now_s)
        if include_details:
            for bucket in _ROUTE_METRICS.values():
                _prune_samples(bucket["samples"], now_s)
                _refresh_saturation(bucket, now_s)
            for bucket in _STAGE_METRICS.values():
                _prune_samples(bucket["samples"], now_s)
                _refresh_saturation(bucket, now_s)
        for budget_samples in _BUDGET_PATH_SAMPLES.values():
            _prune_samples(budget_samples, now_s)
        for saturation in _BUDGET_PATH_SATURATION.values():
            _refresh_saturation(saturation, now_s)

        samples = list(_SAMPLES_MS)
        global_saturation = dict(_GLOBAL_SAMPLE_SATURATION)
        lifetime_count = int(_COUNT)
        lifetime_total_ms = float(_TOTAL_MS)
        lifetime_max_ms = float(_MAX_MS)
        lifetime_last_ms = float(_LAST_MS)
        lifetime_total_bytes = int(_TOTAL_BYTES)
        lifetime_last_bytes = int(_LAST_BYTES)
        lifetime_violations = int(_VIOLATIONS)
        route_snapshot = (
            {
                str(key): {
                    "samples": list(bucket.get("samples", [])),
                    "saturation": {
                        "dropped_total": int(bucket.get("dropped_total", 0)),
                        "active_dropped": int(bucket.get("active_dropped", 0)),
                        "last_drop_s": bucket.get("last_drop_s"),
                        "max_dropped_ms": float(bucket.get("max_dropped_ms", 0.0)),
                    },
                    "lifetime_count": int(bucket.get("count", 0)),
                    "lifetime_total_ms": float(bucket.get("total_ms", 0.0)),
                    "lifetime_max_ms": float(bucket.get("max_ms", 0.0)),
                    "lifetime_last_ms": float(bucket.get("last_ms", 0.0)),
                    "lifetime_total_bytes": int(bucket.get("total_bytes", 0)),
                    "lifetime_last_bytes": int(bucket.get("last_bytes", 0)),
                }
                for key, bucket in _ROUTE_METRICS.items()
            }
            if include_details
            else {}
        )
        stage_snapshot = (
            {
                str(key): {
                    "samples": list(bucket.get("samples", [])),
                    "saturation": {
                        "dropped_total": int(bucket.get("dropped_total", 0)),
                        "active_dropped": int(bucket.get("active_dropped", 0)),
                        "last_drop_s": bucket.get("last_drop_s"),
                        "max_dropped_ms": float(bucket.get("max_dropped_ms", 0.0)),
                    },
                    "lifetime_count": int(bucket.get("count", 0)),
                    "lifetime_total_ms": float(bucket.get("total_ms", 0.0)),
                    "lifetime_max_ms": float(bucket.get("max_ms", 0.0)),
                    "lifetime_last_ms": float(bucket.get("last_ms", 0.0)),
                    "lifetime_total_bytes": int(bucket.get("total_bytes", 0)),
                    "lifetime_last_bytes": int(bucket.get("last_bytes", 0)),
                }
                for key, bucket in _STAGE_METRICS.items()
            }
            if include_details
            else {}
        )
        detail_truncated = (
            _ROUTE_OVERFLOW_KEY in _ROUTE_METRICS
            or _STAGE_OVERFLOW_KEY in _STAGE_METRICS
        )
        budget_path_snapshot = {
            key: list(path_samples)
            for key, path_samples in _BUDGET_PATH_SAMPLES.items()
        }
        budget_path_saturation = {
            key: dict(saturation) for key, saturation in _BUDGET_PATH_SATURATION.items()
        }
        error_counts = dict(_ERROR_COUNTS)

    def _summarize(
        snapshot: Dict[str, Dict[str, Any]],
        *,
        mark_budgeted: bool,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key, item in snapshot.items():
            windows = _window_summaries(
                item["samples"],
                now_s=now_s,
                saturation=item["saturation"],
            )
            summary_10s = windows["10s"]
            summary_60s = windows["60s"]
            out[key] = {
                **summary_60s,
                "p99_ms": _worst_optional(summary_10s["p99_ms"], summary_60s["p99_ms"]),
                "p99_10s_ms": summary_10s["p99_ms"],
                "p99_60s_ms": summary_60s["p99_ms"],
                "sample_saturated_10s": summary_10s["sample_saturated"],
                "sample_saturated_60s": summary_60s["sample_saturated"],
                "lifetime_count": int(item["lifetime_count"]),
                "lifetime_total_ms": float(item["lifetime_total_ms"]),
                "lifetime_max_ms": float(item["lifetime_max_ms"]),
                "lifetime_last_ms": float(item["lifetime_last_ms"]),
                "lifetime_total_bytes": int(item["lifetime_total_bytes"]),
                "lifetime_last_payload_bytes": int(item["lifetime_last_bytes"]),
            }
            if mark_budgeted:
                out[key]["budgeted"] = key in budget_path_snapshot
        return out

    aggregate_windows = _window_summaries(
        samples,
        now_s=now_s,
        saturation=global_saturation,
    )
    route_summary = _summarize(route_snapshot, mark_budgeted=False)
    stage_summary = _summarize(stage_snapshot, mark_budgeted=True)

    budget_path_windows = {
        key: _window_summaries(
            path_samples,
            now_s=now_s,
            saturation=budget_path_saturation[key],
        )
        for key, path_samples in budget_path_snapshot.items()
    }

    def _path_authority(key: str, window: Dict[str, Any]) -> Any:
        # Cardinality overflow combines distinct paths. Its pooled percentile
        # could hide a sparse slow path, so the overflow bucket fails safe on
        # its maximum while named paths retain p99 authority.
        metric = "max_ms" if key == _STAGE_OVERFLOW_KEY else "p99_ms"
        value = window.get(metric)
        if window.get("sample_saturated"):
            return _worst_optional(
                value,
                window.get("sample_saturation_max_ms"),
                _SATURATION_FAIL_CLOSED_MS,
            )
        return value

    max_path_10s = _worst_optional(
        *(
            _path_authority(key, windows["10s"])
            for key, windows in budget_path_windows.items()
        )
    )
    max_path_60s = _worst_optional(
        *(
            _path_authority(key, windows["60s"])
            for key, windows in budget_path_windows.items()
        )
    )
    aggregate_windows["10s"]["max_path_p99_ms"] = max_path_10s
    aggregate_windows["60s"]["max_path_p99_ms"] = max_path_60s
    path_saturated_10s = any(
        windows["10s"]["sample_saturated"] for windows in budget_path_windows.values()
    )
    path_saturated_60s = any(
        windows["60s"]["sample_saturated"] for windows in budget_path_windows.values()
    )
    budget_path_rows = [
        {
            "key": key,
            "p99_10s_ms": windows["10s"].get("p99_ms"),
            "p99_60s_ms": windows["60s"].get("p99_ms"),
            "max_10s_ms": windows["10s"].get("max_ms"),
            "max_60s_ms": windows["60s"].get("max_ms"),
            "authority_10s_ms": _path_authority(key, windows["10s"]),
            "authority_60s_ms": _path_authority(key, windows["60s"]),
            "sample_saturated_10s": windows["10s"].get("sample_saturated", False),
            "sample_saturated_60s": windows["60s"].get("sample_saturated", False),
        }
        for key, windows in budget_path_windows.items()
    ]
    budget_path_rows.sort(
        key=lambda item: (
            -float(
                _worst_optional(
                    item.get("authority_10s_ms"), item.get("authority_60s_ms")
                )
                or -1.0
            ),
            str(item["key"]),
        )
    )
    top_budget_paths = budget_path_rows[: int(_TOP_BUDGET_PATHS_LIMIT)]

    summary_10s = aggregate_windows["10s"]
    summary_60s = aggregate_windows["60s"]
    metrics = {
        **summary_60s,
        "p99_ms": _worst_optional(summary_10s["p99_ms"], summary_60s["p99_ms"]),
        "p99_10s_ms": summary_10s["p99_ms"],
        "p99_60s_ms": summary_60s["p99_ms"],
        "max_path_p99_ms": _worst_optional(max_path_10s, max_path_60s),
        "max_path_p99_10s_ms": max_path_10s,
        "max_path_p99_60s_ms": max_path_60s,
        "sample_limit": int(_MAX_WINDOW_SAMPLES),
        "sample_saturated_10s": summary_10s["sample_saturated"],
        "sample_saturated_60s": summary_60s["sample_saturated"],
        "path_sample_saturated_10s": path_saturated_10s,
        "path_sample_saturated_60s": path_saturated_60s,
        "budget_path_count": len(budget_path_rows),
        "top_budget_paths_limit": int(_TOP_BUDGET_PATHS_LIMIT),
        "top_budget_paths_truncated": (
            len(budget_path_rows) > int(_TOP_BUDGET_PATHS_LIMIT)
        ),
        "top_budget_paths": top_budget_paths,
        "windows": aggregate_windows,
        "budget_ms": float(_BUDGET_MS),
        "violations": lifetime_violations,
        "errors_total": sum(error_counts.values()),
        "errors": error_counts,
        "boundary_serialization_errors_total": sum(error_counts.values()),
        "boundary_serialization_errors": error_counts,
        "detail_limit": int(_MAX_DETAIL_BUCKETS),
        "detail_truncated": detail_truncated,
        "error_detail_limit": int(_MAX_ERROR_BUCKETS),
        "error_detail_truncated": _ERROR_OVERFLOW_KEY in error_counts,
        "lifetime_count": lifetime_count,
        "lifetime_avg_ms": (
            lifetime_total_ms / float(lifetime_count) if lifetime_count > 0 else None
        ),
        "lifetime_max_ms": lifetime_max_ms if lifetime_count > 0 else None,
        "lifetime_last_ms": lifetime_last_ms if lifetime_count > 0 else None,
        "lifetime_total_bytes": lifetime_total_bytes,
        "lifetime_last_payload_bytes": lifetime_last_bytes,
    }
    if include_details:
        metrics["routes"] = route_summary
        metrics["stages"] = stage_summary
    return metrics


def get_boundary_serialization_metrics_compact() -> Dict[str, Any]:
    """Return runtime gate metrics without materializing detailed buckets."""

    return get_boundary_serialization_metrics(include_details=False)


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
        _BUDGET_PATH_SAMPLES.clear()
        _BUDGET_PATH_SATURATION.clear()
        _ERROR_COUNTS.clear()
        _GLOBAL_SAMPLE_SATURATION.clear()
        _GLOBAL_SAMPLE_SATURATION.update(_new_saturation_state())
