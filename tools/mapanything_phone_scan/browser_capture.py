"""Bounded import of browser camera and motion-sensor capture archives.

Browser MediaRecorder and motion events are useful RGB and sensor evidence,
but their callback clocks are not Camera2 acquisition timestamps.  This
module deliberately keeps that evidence in a separate schema and report so
the native ``noesis.phone_capture.v1`` importer and OpenVINS admission gates
remain unchanged.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .capture import (
    CaptureImportError,
    CaptureImportLimits,
    MAX_VIDEO_TIMESTAMPS,
    _probe_video_metadata,
    _probe_video_timestamps,
    _safe_member_name,
)


BROWSER_CAPTURE_SCHEMA = "noesis.phone_capture.browser.v1"
BROWSER_VIDEO_SUFFIXES = {".webm", ".mp4"}
# A 2 minute capture at 60 Hz for two streams is commonly several MiB.  This
# remains bounded and is deliberately separate from the native 512 KiB limit.
MAX_BROWSER_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_BROWSER_EVENTS = 100_000
MAX_BROWSER_VIDEO_OBSERVATIONS = MAX_VIDEO_TIMESTAMPS
MAX_BROWSER_INTERRUPTION_DETAILS = 256


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise CaptureImportError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CaptureImportError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise CaptureImportError(f"{field} must be finite")
    return number


def _required_string(value: Any, *, field: str, max_length: int = 512) -> str:
    if not isinstance(value, str):
        raise CaptureImportError(f"{field} must be a string")
    result = value.strip()
    if not result or len(result) > max_length:
        raise CaptureImportError(f"{field} must be a non-empty string")
    return result


def _browser_path(value: Any, *, field: str) -> str:
    value = _required_string(value, field=field, max_length=256)
    path = _safe_member_name(value)
    if Path(path).suffix.lower() not in BROWSER_VIDEO_SUFFIXES:
        raise CaptureImportError(f"{field} must reference a .webm or .mp4 video")
    return path


def _api_timestamp_source(api: str) -> str:
    normalized = api.strip().lower().replace(" ", "")
    if "devicemotion" in normalized or "deviceorientation" in normalized:
        return "DeviceMotionEvent.timestamp"
    if (
        "sensor.timestamp" in normalized
        or "genericsensor" in normalized
        or "generic-sensor" in normalized
        or normalized in {"sensor", "accelerometer", "gyroscope"}
    ):
        return "Sensor.timestamp"
    return "declared_browser_api_timestamp"


def _unit_for(kind: str, raw: Any) -> str:
    unit = _required_string(raw, field=f"sensors.{kind}.units", max_length=80).lower()
    if kind == "accelerometer" and unit not in {"m/s^2", "m/s²"}:
        raise CaptureImportError("sensors.accelerometer.units must be m/s^2")
    if kind == "gyroscope" and unit != "rad/s":
        raise CaptureImportError("sensors.gyroscope.units must be rad/s")
    return "m/s^2" if kind == "accelerometer" else "rad/s"


def _sample_row(raw: Any, *, kind: str, index: int) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, "sample_not_object"
    values: dict[str, float] = {}
    for key in ("timestamp_ms", "received_ms", "x", "y", "z"):
        if key not in raw:
            return None, f"missing_{key}"
        try:
            values[key] = _finite(raw[key], field=f"sensors.{kind}.samples[{index}].{key}")
        except CaptureImportError:
            return None, f"nonfinite_{key}"
    # Keep the exact browser units and timestamp values; this stream is never
    # materialized as native VIO input.
    return values, None


def _parse_sensor_stream(
    raw: Any,
    *,
    kind: str,
    limits: CaptureImportLimits,
) -> tuple[dict[str, Any], list[dict[str, float]], list[str]]:
    if not isinstance(raw, Mapping):
        raise CaptureImportError(f"sensors.{kind} must be an object")
    api = _required_string(raw.get("api"), field=f"sensors.{kind}.api")
    timestamp_source = _api_timestamp_source(api)
    sample_values = raw.get("samples")
    sample_domain = (
        sample_values[0].get("timestamp_domain")
        if isinstance(sample_values, list)
        and sample_values
        and isinstance(sample_values[0], Mapping)
        else None
    )
    timestamp_domain = _required_string(
        raw.get("timestamp_domain") or raw.get("time_domain")
        or sample_domain
        or (
            "sensor_timestamp_domain_unverified"
            if timestamp_source == "Sensor.timestamp"
            else "event_timestamp_domain_unverified"
        ),
        field=f"sensors.{kind}.timestamp_domain",
        max_length=160,
    )
    units = _unit_for(kind, raw.get("units"))
    reference_frame = _required_string(
        raw.get("reference_frame"), field=f"sensors.{kind}.reference_frame", max_length=80
    )
    if reference_frame != "device":
        raise CaptureImportError(f"sensors.{kind}.reference_frame must be device")
    samples = raw.get("samples")
    if not isinstance(samples, list):
        raise CaptureImportError(f"sensors.{kind}.samples must be an array")
    if len(samples) > limits.max_imu_rows:
        raise CaptureImportError(
            f"sensors.{kind}.samples exceeds {limits.max_imu_rows} rows"
        )

    valid: list[dict[str, float]] = []
    interruptions: list[str] = []
    previous_timestamp: float | None = None
    for index, raw_sample in enumerate(samples):
        row, reason = _sample_row(raw_sample, kind=kind, index=index)
        if reason is not None:
            if len(interruptions) < MAX_BROWSER_INTERRUPTION_DETAILS:
                interruptions.append(reason)
            continue
        assert row is not None
        timestamp = row["timestamp_ms"]
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            if len(interruptions) < MAX_BROWSER_INTERRUPTION_DETAILS:
                interruptions.append("timestamp_not_strictly_increasing")
            continue
        previous_timestamp = timestamp
        valid.append(row)

    if len(valid) < 2:
        raise CaptureImportError(
            f"sensors.{kind}.samples has fewer than two usable finite samples"
        )

    timestamp_start = valid[0]["timestamp_ms"]
    timestamp_end = valid[-1]["timestamp_ms"]
    received_start = min(row["received_ms"] for row in valid)
    received_end = max(row["received_ms"] for row in valid)
    gaps = [
        valid[index]["timestamp_ms"] - valid[index - 1]["timestamp_ms"]
        for index in range(1, len(valid))
    ]
    max_gap_ms = max(gaps, default=0.0)
    gap_limit_ms = limits.max_time_gap_s * 1000.0
    gap_count = sum(gap > gap_limit_ms for gap in gaps)
    duration_ms = timestamp_end - timestamp_start
    comparable_domains = {
        "performance_time_origin_ms",
        "performance_time_origin",
        "performance.now_ms",
        "domhighrestimestamp_ms",
        "relative_to_time_origin_ms",
    }
    lag_comparable = timestamp_domain.strip().lower() in comparable_domains
    receipt_lags = (
        [row["received_ms"] - row["timestamp_ms"] for row in valid]
        if lag_comparable
        else []
    )
    lag_unavailable_reason = (
        None if lag_comparable
        else "timestamp and received clocks have no declared common domain"
    )
    # A declaration alone cannot reconcile an observed clock-origin mismatch.
    # Preserve the samples; never turn receipt-time differences into a guessed
    # camera/IMU offset or report physically impossible callback delays.
    if receipt_lags and min(receipt_lags) < 0.0:
        lag_comparable = False
        receipt_lags = []
        lag_unavailable_reason = (
            "declared common clock contradicted by sensor timestamps after receipt"
        )
    summary = {
        "api": api,
        "timestamp_source": timestamp_source,
        "timestamp_domain": timestamp_domain,
        "units": units,
        "reference_frame": reference_frame,
        "sample_count": len(valid),
        "raw_sample_count": len(samples),
        "invalid_sample_count": len(samples) - len(valid),
        "timestamp_start_ms": timestamp_start,
        "timestamp_end_ms": timestamp_end,
        "duration_ms": duration_ms,
        "rate_hz": ((len(valid) - 1) * 1000.0 / duration_ms) if duration_ms > 0.0 else None,
        "received_start_ms": received_start,
        "received_end_ms": received_end,
        "callback_lag_comparable": lag_comparable,
        "callback_lag_unavailable_reason": lag_unavailable_reason,
        "max_callback_lag_ms": max(receipt_lags) if receipt_lags else None,
        "min_callback_lag_ms": min(receipt_lags) if receipt_lags else None,
        "max_gap_ms": max_gap_ms,
        "gap_limit_ms": gap_limit_ms,
        "gap_count_over_limit": gap_count,
        "strictly_increasing_after_repair": True,
        "interrupted": bool(interruptions or gap_count),
        "interruption_count": (len(samples) - len(valid)) + gap_count,
        "interruption_detail_count": len(interruptions),
        "calibrated": False,
    }
    return summary, valid, interruptions


def _parse_video_observations(
    raw: Any,
    *,
    limits: CaptureImportLimits,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    if not isinstance(raw, list):
        raise CaptureImportError("video_frames must be an array")
    if len(raw) > min(limits.max_video_timestamps, MAX_BROWSER_VIDEO_OBSERVATIONS):
        raise CaptureImportError("video_frames exceeds its observation limit")
    valid: list[dict[str, Any]] = []
    interruptions: list[str] = []
    previous_media_time = -math.inf
    capture_time_count = 0
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            if len(interruptions) < MAX_BROWSER_INTERRUPTION_DETAILS:
                interruptions.append("video_frame_not_object")
            continue
        try:
            media_time_s = _finite(item.get("media_time_s"), field=f"video_frames[{index}].media_time_s")
            callback_time_ms = _finite(item.get("callback_time_ms"), field=f"video_frames[{index}].callback_time_ms")
            capture_raw = item.get("capture_time_ms")
            capture_time_ms = None if capture_raw is None else _finite(capture_raw, field=f"video_frames[{index}].capture_time_ms")
        except CaptureImportError:
            if len(interruptions) < MAX_BROWSER_INTERRUPTION_DETAILS:
                interruptions.append("video_frame_timestamp_invalid")
            continue
        if media_time_s < 0.0 or media_time_s < previous_media_time:
            if len(interruptions) < MAX_BROWSER_INTERRUPTION_DETAILS:
                interruptions.append("video_media_time_not_monotonic")
            continue
        previous_media_time = media_time_s
        row = dict(item)
        row["media_time_s"] = media_time_s
        row["callback_time_ms"] = callback_time_ms
        row["capture_time_ms"] = capture_time_ms
        if capture_time_ms is not None:
            capture_time_count += 1
        valid.append(row)
    summary = {
        "observation_count": len(valid),
        "raw_observation_count": len(raw),
        "invalid_observation_count": len(raw) - len(valid),
        "capture_time_present_count": capture_time_count,
        "timestamp_source": "browser_callback_observation",
        "capture_time_source": (
            "browser_capture_time_observation_unverified" if capture_time_count else None
        ),
        "interrupted": bool(interruptions),
        "interruption_count": len(raw) - len(valid),
        "interruption_detail_count": len(interruptions),
        "empty": not valid,
    }
    return summary, valid, interruptions


def _validate_timing(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CaptureImportError("timing must be an object")
    time_origin_ms = _finite(raw.get("time_origin_ms"), field="timing.time_origin_ms")
    started_ms = _finite(raw.get("started_ms"), field="timing.started_ms")
    stopped_ms = _finite(raw.get("stopped_ms"), field="timing.stopped_ms")
    if stopped_ms < started_ms:
        raise CaptureImportError("timing.stopped_ms must be at or after started_ms")
    timing = dict(raw)
    timing.update(
        {
            "time_origin_ms": time_origin_ms,
            "started_ms": started_ms,
            "stopped_ms": stopped_ms,
            "duration_ms": stopped_ms - started_ms,
            "clock_provenance": "browser_performance_or_event_clock_unverified",
        }
    )
    return timing


def validate_browser_capture_manifest(
    manifest: Mapping[str, Any],
    *,
    member_names: set[str] | None = None,
    limits: CaptureImportLimits = CaptureImportLimits(),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a browser manifest and return report-safe normalized data.

    The original manifest is retained in the archive.  The normalized copy
    intentionally omits raw sample arrays; they are written to a separate
    bounded JSON artifact so report metadata remains compact.
    """

    if manifest.get("schema") != BROWSER_CAPTURE_SCHEMA:
        raise CaptureImportError(f"capture manifest must use {BROWSER_CAPTURE_SCHEMA}")
    capture_id = _required_string(manifest.get("capture_id"), field="capture_id", max_length=160)
    device = manifest.get("device")
    if not isinstance(device, Mapping):
        raise CaptureImportError("capture manifest has no browser device identity")
    device_id = _required_string(device.get("id"), field="device.id", max_length=240)
    expected_device_id = f"browser-session:{capture_id}"
    if device_id != expected_device_id:
        raise CaptureImportError("device.id must be browser-session:<capture_id>")
    model = _required_string(device.get("model"), field="device.model", max_length=240)
    user_agent = _required_string(device.get("user_agent"), field="device.user_agent", max_length=2048)

    video = manifest.get("video")
    if not isinstance(video, Mapping):
        raise CaptureImportError("capture manifest has no browser video stream")
    video_path = _browser_path(video.get("path"), field="video.path")
    mime_type = _required_string(video.get("mime_type"), field="video.mime_type", max_length=160).lower()
    suffix = Path(video_path).suffix.lower()
    if suffix == ".webm" and not mime_type.startswith("video/webm"):
        raise CaptureImportError("video.mime_type does not match the WebM path")
    if suffix == ".mp4" and not mime_type.startswith("video/mp4"):
        raise CaptureImportError("video.mime_type does not match the MP4 path")
    if member_names is not None and video_path not in member_names:
        raise CaptureImportError("capture manifest references a missing video file")

    camera = manifest.get("camera")
    if not isinstance(camera, Mapping):
        raise CaptureImportError("capture manifest has no browser camera metadata")
    settings = camera.get("settings")
    capabilities = camera.get("capabilities")
    if not isinstance(settings, Mapping) or not isinstance(capabilities, Mapping):
        raise CaptureImportError("camera.settings and camera.capabilities must be objects")
    requested = camera.get("requested", {})
    if not isinstance(requested, Mapping):
        raise CaptureImportError("camera.requested must be an object")
    strict_8k = requested.get("mode") == "8k"
    if strict_8k and (
        (settings.get("width"), settings.get("height")) not in ((7680, 4320), (4320, 7680))
        or settings.get("resizeMode") != "none"
        or settings.get("facingMode") != "environment"
    ):
        raise CaptureImportError("strict 8K capture requires verified unscaled rear-camera 8K settings")
    timing = _validate_timing(manifest.get("timing"))

    sensors = manifest.get("sensors")
    if not isinstance(sensors, Mapping):
        raise CaptureImportError("capture manifest has no browser sensors")
    sensor_summaries: dict[str, Any] = {}
    sensor_rows: dict[str, list[dict[str, float]]] = {}
    interruptions: list[dict[str, Any]] = []
    for kind in ("accelerometer", "gyroscope"):
        if kind not in sensors:
            raise CaptureImportError(f"capture manifest is missing sensors.{kind}")
        summary, rows, stream_interruptions = _parse_sensor_stream(
            sensors[kind], kind=kind, limits=limits
        )
        sensor_summaries[kind] = summary
        sensor_rows[kind] = rows
        for reason in stream_interruptions[:MAX_BROWSER_INTERRUPTION_DETAILS]:
            interruptions.append({"source": kind, "reason": reason})
        if summary["gap_count_over_limit"]:
            interruptions.append(
                {
                    "source": kind,
                    "reason": "sensor_timestamp_gap_over_limit",
                    "count": summary["gap_count_over_limit"],
                    "limit_ms": summary["gap_limit_ms"],
                }
            )

    video_summary, video_rows, video_interruptions = _parse_video_observations(
        manifest.get("video_frames"), limits=limits
    )
    for reason in video_interruptions[:MAX_BROWSER_INTERRUPTION_DETAILS]:
        interruptions.append({"source": "video_frames", "reason": reason})

    events = manifest.get("events", [])
    if not isinstance(events, list):
        raise CaptureImportError("events must be an array")
    if len(events) > MAX_BROWSER_EVENTS:
        raise CaptureImportError("events exceeds its row limit")
    stop_reason = _required_string(manifest.get("stop_reason"), field="stop_reason", max_length=240)

    normalized = deepcopy(dict(manifest))
    normalized["capture_id"] = capture_id
    normalized["device"] = {"id": device_id, "model": model, "user_agent": user_agent}
    normalized["video"] = {"path": video_path, "mime_type": mime_type}
    normalized["camera"] = {
        "settings": deepcopy(dict(settings)),
        "capabilities": deepcopy(dict(capabilities)),
        "requested": deepcopy(dict(requested)),
        "calibration": "unknown",
    }
    normalized["timing"] = timing
    normalized["sensors"] = {
        kind: {
            "api": sensor_summaries[kind]["api"],
            "units": sensor_summaries[kind]["units"],
            "reference_frame": "device",
            "timestamp_source": sensor_summaries[kind]["timestamp_source"],
            "timestamp_domain": sensor_summaries[kind]["timestamp_domain"],
        }
        for kind in ("accelerometer", "gyroscope")
    }
    normalized["video_frames"] = {"preserved_in_raw_manifest": True}
    normalized["events"] = {"preserved_in_raw_manifest": True, "count": len(events)}
    normalized["stop_reason"] = stop_reason

    parsed = {
        "sensors": sensor_summaries,
        "sensor_rows": sensor_rows,
        "video_frames": video_summary,
        "video_rows": video_rows,
    }
    return normalized, parsed, interruptions


def import_browser_capture_from_extracted(
    temporary_root: Path,
    scan_dir: Path,
    archive_path: Path,
    *,
    manifest_name: str,
    member_names: set[str],
    archive_uncompressed_bytes: int,
    manifest: Mapping[str, Any],
    limits: CaptureImportLimits,
) -> dict[str, Any]:
    """Finalize a browser import after shared safe archive extraction."""

    normalized, parsed, interruptions = validate_browser_capture_manifest(
        manifest, member_names=member_names, limits=limits
    )
    video_path = temporary_root / str(normalized["video"]["path"])
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise CaptureImportError("browser capture video is empty")
    encoded_width, encoded_height = _probe_video_metadata(video_path)
    strict_8k = normalized["camera"]["requested"].get("mode") == "8k"
    if strict_8k and (encoded_width, encoded_height) != (
        normalized["camera"]["settings"]["width"], normalized["camera"]["settings"]["height"]
    ):
        raise CaptureImportError(
            f"strict 8K recording encoded {encoded_width}x{encoded_height}, which differs from its verified camera settings"
        )
    encoded_timestamps = _probe_video_timestamps(
        video_path, maximum=limits.max_video_timestamps
    )
    if not encoded_timestamps:
        raise CaptureImportError("browser capture video has no encoded frames")

    video_summary = dict(parsed["video_frames"])
    video_summary.update(
        {
            "encoded_frame_count": len(encoded_timestamps),
            "encoded_width": encoded_width,
            "encoded_height": encoded_height,
            "requested_8k_verified": strict_8k,
            "encoded_duration_s": (
                max(encoded_timestamps[-1] - encoded_timestamps[0], 0.0)
                if len(encoded_timestamps) > 1
                else 0.0
            ),
            "encoded_timestamp_source": "ffprobe.best_effort_timestamp_time",
        }
    )
    if video_summary["observation_count"] == 0:
        interruptions.append(
            {"source": "video_frames", "reason": "no_valid_browser_frame_observations"}
        )

    capture_dir = temporary_root
    raw_sensor_path = capture_dir / "browser_sensor_samples.json"
    raw_sensor_path.write_text(
        json.dumps(
            {
                "schema": BROWSER_CAPTURE_SCHEMA,
                "timestamp_provenance": {
                    kind: parsed["sensors"][kind]["timestamp_source"]
                    for kind in ("accelerometer", "gyroscope")
                },
                "accelerometer": parsed["sensor_rows"]["accelerometer"],
                "gyroscope": parsed["sensor_rows"]["gyroscope"],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    raw_sensor_path.chmod(0o600)
    capture_import_path = capture_dir / "capture_import.json"
    manifest_relative = Path(manifest_name).as_posix()
    report: dict[str, Any] = {
        "schema": BROWSER_CAPTURE_SCHEMA,
        "capture_kind": "browser_camera_imu",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_uncompressed_bytes": archive_uncompressed_bytes,
        "manifest": normalized,
        "manifest_path": f"capture/{manifest_relative}",
        "video": video_summary,
        "sensors": parsed["sensors"],
        "timing": normalized["timing"],
        "coverage": {
            "cross_sensor_clock_verified": False,
            "video_sensor_alignment_verified": False,
            "time_domains": {
                "video": "browser_media_time_and_callback_time",
                "accelerometer": parsed["sensors"]["accelerometer"]["timestamp_domain"],
                "gyroscope": parsed["sensors"]["gyroscope"]["timestamp_domain"],
            },
        },
        "calibration": {
            "camera_intrinsics": False,
            "camera_to_imu_extrinsics": False,
            "time_offset": False,
            "clock_source_verified": False,
            "imu_noise": False,
            "camera_geometry": True,
            "complete_for_metric_vio": False,
            "reason": "Browser camera and IMU capture has no native metric calibration or synchronized acquisition clock",
        },
        "metric_vio_allowed": False,
        "native_vio_compatible": False,
        "camera_acquisition_timestamp_verified": False,
        "raw_streams_preserved": True,
        "interruptions": interruptions[:MAX_BROWSER_INTERRUPTION_DETAILS],
        "stop_reason": normalized["stop_reason"],
        "sensor_samples_path": "capture/browser_sensor_samples.json",
        "video_path": f"capture/{normalized['video']['path']}",
        "import_report_path": "capture/capture_import.json",
    }
    capture_import_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    capture_import_path.chmod(0o600)
    capture_root = scan_dir / "capture"
    if capture_root.exists():
        raise CaptureImportError("capture import destination already exists")
    os.replace(temporary_root, capture_root)
    return report


__all__ = [
    "BROWSER_CAPTURE_SCHEMA",
    "MAX_BROWSER_MANIFEST_BYTES",
    "import_browser_capture_from_extracted",
    "validate_browser_capture_manifest",
]
