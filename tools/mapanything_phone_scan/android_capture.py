"""Verify Android recorder timing while preserving incomplete RGB evidence."""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


ANDROID_CAPTURE_SCHEMA = "noesis.phone_capture.android.v1"
ANDROID_RESULT_SCHEMA = "noesis.phone_capture.android_result.v1"
ASSOCIATION_METHOD = "encoder_pts_us_equals_sensor_timestamp_ns_div_1000"
MAX_RESULT_BYTES = 512 * 1024
MAX_METADATA_LINE_BYTES = 64 * 1024


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be nonnegative")
    return parsed


def _csv_rows(path: Path, fields: tuple[str, ...], maximum: int) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in fields):
            raise ValueError(f"{path.name} has no required timing columns")
        for index, row in enumerate(reader):
            if index >= maximum:
                raise ValueError(f"{path.name} exceeds its timing row limit")
            rows.append({field: _integer(row.get(field), f"{path.name}[{index}].{field}") for field in fields})
    return rows


def _camera_rows(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        while line := handle.readline(MAX_METADATA_LINE_BYTES + 1):
            if len(line) > MAX_METADATA_LINE_BYTES:
                raise ValueError("camera result row exceeds its size limit")
            if not line.strip():
                continue
            if len(rows) >= maximum:
                raise ValueError("camera results exceed their timing row limit")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("camera result must be an object")
            rows.append({
                "ois_mode": row.get("ois_mode"),
                "eis_mode": row.get("eis_mode"),
                **{
                    field: _integer(row.get(field), f"camera result.{field}")
                    for field in ("frame_number", "sensor_timestamp_ns")
                },
            })
    return rows


def verify_android_capture(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    frame_count: int,
    container_timing: Mapping[str, Any],
    maximum_rows: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    """Check original recorder rows; a timing failure leaves raw RGB usable.

    MP4 presentation times may use a different origin or coarser timescale.
    Compare their deltas only to validate encoded order. That normalization is
    never used as a camera/IMU clock offset or acquisition timestamp.
    """

    evidence = manifest["android_capture"]
    errors: list[str] = []
    assessment: dict[str, Any] = {
        "schema": ANDROID_CAPTURE_SCHEMA,
        "camera_acquisition_timestamp_verified": False,
        "association_method": ASSOCIATION_METHOD,
        "decoded_frame_count": frame_count,
        "evidence_paths": {key: value for key, value in evidence.items() if key.endswith("_path")},
        "errors": errors,
    }

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        result_path = root / evidence["capture_result_path"]
        if result_path.stat().st_size > MAX_RESULT_BYTES:
            raise ValueError("Android capture result exceeds its size limit")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("schema") != ANDROID_RESULT_SCHEMA:
            raise ValueError(f"Android capture result must use {ANDROID_RESULT_SCHEMA}")
        camera = result.get("camera") or {}
        timing = result.get("timing") or {}
        if not isinstance(camera, dict) or not isinstance(timing, dict):
            raise ValueError("Android capture result has no camera/timing evidence")
        assessment["recorder"] = {
            key: result.get(key)
            for key in ("android_api_level", "status", "stop_reason", "partial", "failures", "dropped_metadata_records", "camera", "sensors")
        }
        assessment["reported_timing"] = timing
        check(_integer(result.get("android_api_level"), "android_api_level") >= 33, "Sensor-based encoder timestamps require Android API 33 or later")
        check(camera.get("timestamp_source") == "REALTIME", "Camera2 timestamp source is not REALTIME")
        check(camera.get("timestamp_base") == "SENSOR" and camera.get("timestamp_base_configured") is True, "Encoder output did not confirm TIMESTAMP_BASE_SENSOR")
        check(str(camera.get("id")) == manifest["camera"]["id"], "Recorder camera identity differs from the capture manifest")
        check([camera.get("width"), camera.get("height")] == manifest["video"]["encoded_resolution_px"], "Recorder dimensions differ from encoded video")
        check(timing.get("association_method") == ASSOCIATION_METHOD, "Recorder did not use exact sensor-to-encoder timestamp equality")
        check(timing.get("exact_frame_association_verified") is True, "Recorder did not verify every encoded frame association")
        check(result.get("dropped_metadata_records") == 0, "Recorder dropped camera or encoder metadata")

        encoded = _csv_rows(
            root / evidence["encoder_pts_path"],
            ("encoded_index", "encoded_pts_us", "flags", "size_bytes"), maximum_rows,
        )
        camera_rows = _camera_rows(root / evidence["camera_results_path"], maximum_rows)
        assessment["encoded_row_count"] = len(encoded)
        assessment["camera_result_row_count"] = len(camera_rows)
        check(len(encoded) == frame_count and frame_count >= 2, "Encoder timing count differs from decoded video frame count")
        check([row["encoded_index"] for row in encoded] == list(range(len(encoded))), "Encoder timing indices are not consecutive")
        encoded_pts = [row["encoded_pts_us"] for row in encoded]
        sensor_pts = [row["sensor_timestamp_ns"] // 1000 for row in camera_rows]
        check(all(left < right for left, right in zip(encoded_pts, encoded_pts[1:])), "Encoder presentation timestamps are not strictly increasing")
        check(all(left < right for left, right in zip(sensor_pts, sensor_pts[1:])), "Camera acquisition timestamps are not unique and increasing at encoder precision")
        check(all(row["size_bytes"] > 0 and not row["flags"] & 2 for row in encoded), "Encoder timing contains a non-video sample")
        by_pts = {row["sensor_timestamp_ns"] // 1000: row for row in camera_rows}
        matched_count = sum(pts in by_pts for pts in encoded_pts)
        assessment["matched_frame_count"] = matched_count
        check(matched_count == frame_count, "An encoded presentation timestamp has no exact Camera2 sensor match")
        expected_counts = {
            "encoded_frame_count": len(encoded),
            "matched_frame_count": matched_count,
            "unmatched_encoded_frame_count": len(encoded) - matched_count,
            "duplicate_encoded_pts_count": len(encoded_pts) - len(set(encoded_pts)),
            "duplicate_sensor_timestamp_us_count": len(sensor_pts) - len(set(sensor_pts)),
        }
        for key, count in expected_counts.items():
            check(timing.get(key) == count, f"Recorder {key} differs from its raw timing rows")

        timestamp_path = manifest["video"].get("frame_timestamps_path")
        if not timestamp_path:
            errors.append("Recorder supplied no acquisition timestamp mapping for the encoded video")
            return None, assessment
        check(manifest["video"].get("timestamp_unit") == "ns", "Android acquisition mapping must use integer nanoseconds")
        check(manifest["clocks"].get("camera_start_time_ns") in {None, 0}, "Android sensor timestamps must not receive an inferred epoch offset")
        associations = _csv_rows(
            root / timestamp_path,
            ("encoded_index", "encoded_pts_us", "frame_number", "timestamp_ns"), maximum_rows,
        )
        assessment["association_row_count"] = len(associations)
        check(len(associations) == len(encoded), "Acquisition mapping count differs from encoder timing rows")
        for index, (encoded_row, row) in enumerate(zip(encoded, associations)):
            sensor_row = by_pts.get(encoded_row["encoded_pts_us"])
            if not (
                sensor_row is not None
                and row["encoded_index"] == index
                and row["encoded_pts_us"] == encoded_row["encoded_pts_us"]
                and row["timestamp_ns"] == sensor_row["sensor_timestamp_ns"]
                and row["frame_number"] == sensor_row["frame_number"]
            ):
                errors.append(f"Acquisition mapping row {index} does not match original encoder and Camera2 evidence")
                break

        ticks = [_integer(value, "container timestamp") for value in container_timing.get("timestamps_ticks", [])]
        time_base = Fraction(str(container_timing.get("time_base")))
        check(len(ticks) == len(encoded) and time_base > 0, "Container presentation timing is incomplete")
        if ticks and encoded_pts and len(ticks) == len(encoded_pts):
            error_ns = max(
                abs((tick - ticks[0]) * time_base * 1_000_000_000 - (pts - encoded_pts[0]) * 1000)
                for tick, pts in zip(ticks, encoded_pts)
            )
            assessment["container_time_base"] = str(time_base)
            assessment["container_first_timestamp_ticks"] = ticks[0]
            assessment["encoder_first_pts_us"] = encoded_pts[0]
            assessment["container_pts_max_delta_error_ns"] = float(error_ns)
            check(error_ns <= time_base * 1_000_000_000 + 1000, "Encoded PTS cadence differs from the MP4 presentation timeline")

        if errors:
            return None, assessment
        settings_rows = [by_pts[pts] for pts in encoded_pts]
        modes = [row[key] for row in settings_rows for key in ("ois_mode", "eis_mode")]
        valid_modes = [value for value in modes if type(value) is int and value >= 0]
        stabilization = (
            "on" if any(value != 0 for value in valid_modes)
            else "off" if len(valid_modes) == len(modes)
            else "unknown"
        )
        assessment["camera_settings"] = {
            "stabilization": stabilization,
            "evidence": "encoded_frame_camera2_capture_results",
            "frame_count": len(settings_rows),
            "missing_or_invalid_mode_count": len(modes) - len(valid_modes),
        }
        assessment["camera_acquisition_timestamp_verified"] = True
        return [row["timestamp_ns"] for row in associations], assessment
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, csv.Error, ZeroDivisionError) as exc:
        errors.append(f"Android timing evidence could not be verified: {exc}")
        return None, assessment
