"""Shared fixed-schema health aggregation for runtime-owned camera surfaces."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _activity_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def require_complete_camera_health(
    snapshot: Mapping[str, Any],
    configured_cameras: Iterable[object],
) -> dict[str, Any]:
    """Classify configured cameras without equating inactivity with failure.

    A BEV renderer records successful exact frames even when their footpoint
    set is empty.  A configured camera without a row has not reached its first
    frame attempt yet, so aggregation keeps that startup state inactive-ready
    instead of confusing absence of activity with a render failure.
    """

    expected = tuple(
        sorted(
            {
                str(value or "").strip()
                for value in configured_cameras
                if str(value or "").strip()
            }
        )
    )
    raw_cameras = snapshot.get("cameras")
    cameras = (
        {
            str(key): dict(value)
            for key, value in raw_cameras.items()
            if isinstance(value, Mapping)
        }
        if isinstance(raw_cameras, Mapping)
        else {}
    )
    observed = tuple(sorted(set(expected).intersection(cameras)))
    normalized_rows: dict[str, dict[str, Any]] = {}
    for camera, raw in cameras.items():
        row = dict(raw)
        success_count = _activity_count(row.get("success_count"))
        failure_count = _activity_count(row.get("failure_count"))
        count_contract_valid = success_count is not None and failure_count is not None
        if success_count is None:
            success_count = 0
        if failure_count is None:
            failure_count = 0
        # A failure before the first successful render cannot be represented as
        # inactive-ready. A later successful render may explicitly recover it.
        healthy = bool(row.get("healthy")) and count_contract_valid and not (
            success_count == 0 and failure_count > 0
        )
        row["healthy"] = healthy
        row["success_count"] = success_count
        row["failure_count"] = failure_count
        if not count_contract_valid:
            row["last_failure_stage"] = "health_contract"
            row["last_failure_type"] = "InvalidActivityCount"
        normalized_rows[camera] = row
    cameras = normalized_rows
    active = tuple(
        camera
        for camera in observed
        if cameras[camera]["healthy"] and cameras[camera]["success_count"] > 0
    )
    inactive = tuple(sorted(set(expected).difference(active)))
    failed = tuple(
        camera
        for camera, value in sorted(cameras.items())
        if not value["healthy"]
    )
    unexpected = tuple(sorted(set(cameras).difference(expected)))
    classified: dict[str, dict[str, Any]] = {}
    for camera in expected:
        raw = cameras.get(camera)
        if raw is None:
            classified[camera] = {
                "state": "inactive_ready",
                "configured": True,
                "active": False,
                "healthy": True,
                "success_count": 0,
                "failure_count": 0,
                "last_success_ts_us": None,
                "last_failure_ts_us": None,
                "last_failure_stage": None,
                "last_failure_type": None,
            }
            continue
        row = dict(raw)
        row_active = bool(row.get("healthy")) and row["success_count"] > 0
        row.update(
            {
                "state": (
                    "failed"
                    if not bool(row.get("healthy"))
                    else ("active_ready" if row_active else "inactive_ready")
                ),
                "configured": True,
                "active": row_active,
            }
        )
        classified[camera] = row
    renderer_ready = bool(snapshot.get("renderer_ready", snapshot.get("healthy", False)))
    config_ready = bool(expected)
    result = dict(snapshot)
    result.update(
        {
            "contract": "noesis.bev.health",
            "contract_version": 2,
            "healthy": config_ready and renderer_ready and not failed and not unexpected,
            "config_ready": config_ready,
            "renderer_ready": renderer_ready and not failed and not unexpected,
            "rendering_active": bool(active),
            "cameras": classified,
            "configured_camera_count": len(expected),
            "active_camera_count": len(active),
            "inactive_camera_count": len(inactive),
            "failed_camera_count": len(failed),
            "configured_cameras": list(expected),
            "active_cameras": list(active),
            "inactive_cameras": list(inactive),
            "failed_cameras": list(failed),
            "unexpected_cameras": list(unexpected),
            # Retained as explicit compatibility diagnostics.  Inactivity is
            # classified above and is no longer represented as missing.
            "expected_camera_count": len(expected),
            "observed_camera_count": len(observed),
            "expected_cameras": list(expected),
            "missing_cameras": [],
            "unhealthy_cameras": list(failed),
        }
    )
    return result


__all__ = ["require_complete_camera_health"]
