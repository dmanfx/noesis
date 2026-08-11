#!/usr/bin/env python3
"""Fail-closed occupied-scene gate for analytics ROI hot update and restore.

The gate mutates only the selected stream, proves the native reload receipt and
live tracking effect, and restores the exact original REST stage in ``finally``.
It never treats a missing person as a pass: an unoccupied target is BLOCKED.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    build_required_auth_request,
    configure_required_auth_environment,
    connect_required_websocket,
    load_required_internal_auth,
)


CONTRACT = "noesis.analytics.roi-hot-restore-gate"
CONTRACT_VERSION = 1
STAGE = "exclude"
FULL_FRAME_ROI_ID = "NOESIS_ROI_HOT_RESTORE_FULL_FRAME"
MAX_HTTP_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE_BYTES = 256 * 1024
MIN_ADVANCING_MESSAGES = 2
MIN_BASELINE_PERSON_FRAMES = 2
MIN_EXCLUDED_EMPTY_FRAMES = 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_FIELDS = (
    "request_sequence",
    "accepted_sequence",
    "failed_sequence",
    "active_config_sha256",
    "reload_error_count",
    "objects_removed_count",
)


class GateError(RuntimeError):
    """A bounded, evidence-safe gate failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = str(code)


@dataclass
class PhaseEvidence:
    name: str
    target_messages: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    sequence_advances: int = 0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    frame_advances: int = 0
    real_person_frames: int = 0
    real_person_frame_advances: int = 0
    person_tracks_seen: int = 0
    max_consecutive_empty_frames: int = 0
    pipeline_error_count: int = 0
    native_receipt: dict[str, object] | None = None
    _last_real_frame_id: int | None = field(default=None, repr=False)
    _consecutive_empty: int = field(default=0, repr=False)

    def public(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("_last_real_frame_id", None)
        payload.pop("_consecutive_empty", None)
        return payload


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _camera_key(value: object) -> str:
    return "".join(char for char in str(value or "").lower() if char.isalnum())


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _normalize_stage(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise GateError("analytics_stage_not_an_object")
    if payload.get("stage") != STAGE:
        raise GateError("analytics_stage_mismatch")
    width = payload.get("config_width")
    height = payload.get("config_height")
    if not _is_int(width) or int(width) <= 0:
        raise GateError("analytics_config_width_invalid")
    if not _is_int(height) or int(height) <= 0:
        raise GateError("analytics_config_height_invalid")
    defaults = payload.get("defaults")
    if not isinstance(defaults, Mapping):
        raise GateError("analytics_defaults_invalid")
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise GateError("analytics_streams_missing")

    streams: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_stream in raw_streams:
        if not isinstance(raw_stream, Mapping):
            raise GateError("analytics_stream_invalid")
        stream_id = str(raw_stream.get("stream_id") or "")
        if not stream_id.isdecimal() or stream_id in seen:
            raise GateError("analytics_stream_id_invalid")
        seen.add(stream_id)
        enable = raw_stream.get("enable")
        if not isinstance(enable, bool):
            raise GateError("analytics_stream_enable_invalid")
        raw_rois = raw_stream.get("rois")
        if not isinstance(raw_rois, list):
            raise GateError("analytics_rois_invalid")
        rois: list[dict[str, object]] = []
        for raw_roi in raw_rois:
            if not isinstance(raw_roi, Mapping):
                raise GateError("analytics_roi_invalid")
            roi_id = str(raw_roi.get("id") or "")
            points = raw_roi.get("points_px")
            if not roi_id or not isinstance(points, list) or len(points) < 3:
                raise GateError("analytics_roi_shape_invalid")
            normalized_points: list[list[float]] = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise GateError("analytics_roi_point_invalid")
                x = _finite_number(point[0])
                y = _finite_number(point[1])
                if x is None or y is None:
                    raise GateError("analytics_roi_coordinate_invalid")
                normalized_points.append([x, y])
            description = raw_roi.get("description")
            if description is not None and not isinstance(description, str):
                raise GateError("analytics_roi_description_invalid")
            rois.append(
                {
                    "id": roi_id,
                    "description": description,
                    "points_px": normalized_points,
                }
            )
        label = raw_stream.get("label")
        if label is not None and not isinstance(label, str):
            raise GateError("analytics_stream_label_invalid")
        streams.append(
            {
                "stream_id": stream_id,
                "label": label,
                "enable": enable,
                "rois": rois,
            }
        )
    streams.sort(key=lambda item: int(str(item["stream_id"])))
    return {
        "stage": STAGE,
        "config_width": int(width),
        "config_height": int(height),
        "defaults": copy.deepcopy(dict(defaults)),
        "streams": streams,
    }


def _load_camera_inventory(path: Path) -> dict[str, str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise GateError("camera_inventory_unreadable") from exc
    cameras = payload.get("cameras") if isinstance(payload, Mapping) else None
    if not isinstance(cameras, Mapping):
        raise GateError("camera_inventory_invalid")
    result: dict[str, str] = {}
    for raw_stream_id, raw_camera in cameras.items():
        if not isinstance(raw_camera, Mapping):
            raise GateError("camera_inventory_entry_invalid")
        stream_id = str(raw_stream_id)
        camera_name = str(raw_camera.get("name") or "").strip()
        if not stream_id.isdecimal() or not camera_name:
            raise GateError("camera_inventory_entry_invalid")
        result[stream_id] = camera_name
    return result


def _resolve_target_stream(
    stage: Mapping[str, object], camera: str, cameras_config: Path
) -> tuple[str, str]:
    inventory = _load_camera_inventory(cameras_config)
    requested_key = _camera_key(camera)
    if not requested_key:
        raise GateError("target_camera_empty")
    matches = [
        (stream_id, name)
        for stream_id, name in inventory.items()
        if stream_id == camera or _camera_key(name) == requested_key
    ]
    if len(matches) != 1:
        raise GateError("target_camera_not_unique")
    stream_id, canonical_camera = matches[0]
    streams = stage.get("streams")
    if not isinstance(streams, list):
        raise GateError("analytics_streams_missing")
    stage_rows = [
        row
        for row in streams
        if isinstance(row, Mapping) and str(row.get("stream_id")) == stream_id
    ]
    if len(stage_rows) != 1:
        raise GateError("target_stream_missing")
    label = stage_rows[0].get("label")
    if label is not None and _camera_key(label) != _camera_key(canonical_camera):
        raise GateError("target_stream_camera_mismatch")
    return stream_id, canonical_camera


def _temporary_full_frame_stage(
    original: Mapping[str, object], target_stream_id: str
) -> dict[str, object]:
    temporary = copy.deepcopy(dict(original))
    width = int(temporary["config_width"])
    height = int(temporary["config_height"])
    streams = temporary.get("streams")
    if not isinstance(streams, list):
        raise GateError("analytics_streams_missing")
    matches = 0
    for stream in streams:
        if not isinstance(stream, dict) or str(stream.get("stream_id")) != target_stream_id:
            continue
        matches += 1
        stream["enable"] = True
        stream["rois"] = [
            {
                "id": FULL_FRAME_ROI_ID,
                "description": "Occupied-scene hot-restore gate",
                "points_px": [
                    [0.0, 0.0],
                    [float(width), 0.0],
                    [float(width), float(height)],
                    [0.0, float(height)],
                ],
            }
        ]
    if matches != 1:
        raise GateError("target_stream_missing")
    if _json_sha256(temporary) == _json_sha256(original):
        raise GateError("temporary_stage_does_not_change_target")
    return temporary


def _update_payload(stage: Mapping[str, object]) -> dict[str, object]:
    streams = stage.get("streams")
    if not isinstance(streams, list):
        raise GateError("analytics_streams_missing")
    return {"stage": STAGE, "streams": copy.deepcopy(streams)}


def _normalize_receipt(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise GateError("native_receipt_missing")
    normalized: dict[str, object] = {}
    for field_name in RECEIPT_FIELDS:
        value = payload.get(field_name)
        if field_name == "active_config_sha256":
            digest = str(value or "")
            if SHA256_RE.fullmatch(digest) is None:
                raise GateError("native_receipt_hash_invalid")
            normalized[field_name] = digest
            continue
        if not _is_int(value) or int(value) < 0:
            raise GateError("native_receipt_counter_invalid")
        normalized[field_name] = int(value)
    return normalized


def _validate_initial_receipt(receipt: Mapping[str, object]) -> None:
    if int(receipt["reload_error_count"]) != 0:
        raise GateError("native_initial_error_count_nonzero")
    request_sequence = int(receipt["request_sequence"])
    accepted_sequence = int(receipt["accepted_sequence"])
    failed_sequence = int(receipt["failed_sequence"])
    if request_sequence == 0:
        valid_sequence = accepted_sequence == 0 and failed_sequence == 0
    else:
        valid_sequence = (
            accepted_sequence == request_sequence
            and failed_sequence != request_sequence
        )
    if not valid_sequence:
        raise GateError("native_initial_sequence_invalid")


def _validate_reload_receipt(
    receipt: Mapping[str, object],
    *,
    previous: Mapping[str, object],
    expected_hash: str | None = None,
) -> None:
    request_sequence = int(receipt["request_sequence"])
    if request_sequence < 1 or int(receipt["accepted_sequence"]) != request_sequence:
        raise GateError("native_reload_not_accepted")
    if int(receipt["failed_sequence"]) == request_sequence:
        raise GateError("native_reload_failed_sequence_matches")
    if int(receipt["reload_error_count"]) != 0:
        raise GateError("native_reload_error_count_nonzero")
    if request_sequence <= int(previous["request_sequence"]):
        raise GateError("native_reload_sequence_not_advanced")
    if int(receipt["objects_removed_count"]) < int(previous["objects_removed_count"]):
        raise GateError("native_removed_counter_regressed")
    if expected_hash is not None and receipt["active_config_sha256"] != expected_hash:
        raise GateError("native_active_hash_mismatch")


def _validate_update_response(
    response: object, expected_stage: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(response, Mapping):
        raise GateError("analytics_update_response_invalid")
    if response.get("reloaded") is not True:
        raise GateError("analytics_update_not_reloaded")
    if _normalize_stage(response) != dict(expected_stage):
        raise GateError("analytics_update_semantic_mismatch")
    return _normalize_receipt(response.get("reload_receipt"))


def _endpoint(base_url: str, path: str) -> str:
    parts = urlsplit(str(base_url).strip())
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise GateError("rest_base_url_invalid")
    base_path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme, parts.netloc, f"{base_path}/{path.lstrip('/')}", "", "")
    )


def _request_json(
    method: str,
    url: str,
    auth: RequiredInternalAuth,
    *,
    payload: Mapping[str, object] | None = None,
    timeout_s: float = 8.0,
) -> dict[str, object]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
        if len(data) > MAX_HTTP_BYTES:
            raise GateError("rest_request_too_large")
        headers["Content-Type"] = "application/json"
    request = build_required_auth_request(
        url,
        auth,
        method=method,
        data=data,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - validated HTTP(S) endpoint
            request, timeout=max(0.1, float(timeout_s))
        ) as response:
            status = int(getattr(response, "status", response.getcode()))
            raw = response.read(MAX_HTTP_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise GateError(f"rest_http_status_{int(exc.code)}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GateError("rest_transport_failed") from exc
    if status < 200 or status >= 300:
        raise GateError(f"rest_http_status_{status}")
    if len(raw) > MAX_HTTP_BYTES:
        raise GateError("rest_response_too_large")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise GateError("rest_response_json_invalid") from exc
    if not isinstance(decoded, dict):
        raise GateError("rest_response_not_an_object")
    return decoded


def _get_original_stage(
    rest_base: str,
    auth: RequiredInternalAuth,
    *,
    timeout_s: float,
) -> dict[str, object]:
    deadline = time.monotonic() + min(max(1.0, timeout_s), 15.0)
    endpoint = _endpoint(rest_base, "/api/v1/analytics/rois")
    # Add the fixed, non-secret query only after validating the
    # operator-provided base URL.
    endpoint += "?stage=exclude"
    last_error: GateError | None = None
    while time.monotonic() < deadline:
        try:
            return _normalize_stage(_request_json("GET", endpoint, auth))
        except GateError as exc:
            last_error = exc
            time.sleep(0.5)
    raise last_error or GateError("rest_get_stage_timeout")


def _post_stage(
    rest_base: str,
    auth: RequiredInternalAuth,
    stage: Mapping[str, object],
) -> dict[str, object]:
    endpoint = _endpoint(rest_base, "/api/v1/analytics/rois")
    return _request_json(
        "POST", endpoint, auth, payload=_update_payload(stage), timeout_s=10.0
    )


def _world_sequence(payload: Mapping[str, object]) -> int | None:
    snapshot = payload.get("world_snapshot")
    sequence = snapshot.get("sequence") if isinstance(snapshot, Mapping) else None
    if not _is_int(sequence) or int(sequence) < 0:
        return None
    return int(sequence)


def _tracking_frame_id(payload: Mapping[str, object]) -> int | None:
    frame_id = payload.get("frame_id")
    if not _is_int(frame_id) or int(frame_id) < 0:
        return None
    return int(frame_id)


def _real_person_frame_id(track: object) -> int | None:
    if not isinstance(track, Mapping):
        return None
    class_id = track.get("class_id")
    if not _is_int(class_id) or int(class_id) != 0:
        return None
    tracker_id = track.get("tracker_id")
    frame_id = track.get("frame_id")
    if not _is_int(tracker_id) or int(tracker_id) < 0:
        return None
    if not _is_int(frame_id) or int(frame_id) < 0:
        return None
    bbox = track.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    values = [_finite_number(value) for value in bbox]
    if any(value is None for value in values):
        return None
    _left, _top, width, height = (
        float(value) for value in values if value is not None
    )
    if width <= 0.0 or height <= 0.0:
        return None
    confidence = _finite_number(track.get("confidence"))
    tracker_confidence = _finite_number(track.get("tracker_confidence"))
    if not any(
        value is not None and value > 0.0
        for value in (confidence, tracker_confidence)
    ):
        return None
    return int(frame_id)


def _record_tracking(
    phase: PhaseEvidence,
    payload: Mapping[str, object],
    *,
    camera_id: str,
) -> None:
    if _camera_key(payload.get("camera_id")) != _camera_key(camera_id):
        return
    sequence = _world_sequence(payload)
    if sequence is None:
        raise GateError("target_tracking_sequence_missing")
    if phase.last_sequence is not None and sequence <= phase.last_sequence:
        raise GateError("target_tracking_sequence_not_advancing")
    if phase.first_sequence is None:
        phase.first_sequence = sequence
    else:
        phase.sequence_advances += 1
    phase.last_sequence = sequence

    frame_id = _tracking_frame_id(payload)
    if frame_id is None:
        raise GateError("target_tracking_frame_id_missing")
    if phase.last_frame_id is not None and frame_id <= phase.last_frame_id:
        raise GateError("target_tracking_frame_not_advancing")
    if phase.first_frame_id is None:
        phase.first_frame_id = frame_id
    else:
        phase.frame_advances += 1
    phase.last_frame_id = frame_id
    phase.target_messages += 1

    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        raise GateError("target_tracking_tracks_invalid")
    rows = [row for row in tracks if isinstance(row, Mapping)]
    phase.person_tracks_seen += len(rows)
    if rows:
        phase._consecutive_empty = 0
    else:
        phase._consecutive_empty += 1
        phase.max_consecutive_empty_frames = max(
            phase.max_consecutive_empty_frames, phase._consecutive_empty
        )

    real_frame_ids = [
        frame_id
        for frame_id in (_real_person_frame_id(row) for row in rows)
        if frame_id is not None
    ]
    if not real_frame_ids:
        return
    current_frame = max(real_frame_ids)
    phase.real_person_frames += 1
    if phase._last_real_frame_id is not None:
        if current_frame <= phase._last_real_frame_id:
            raise GateError("real_person_frame_not_advancing")
        phase.real_person_frame_advances += 1
    phase._last_real_frame_id = current_frame


def _select_stats_receipt(
    pipeline: Mapping[str, object], *, expected: Mapping[str, object] | None
) -> dict[str, object] | None:
    if expected is not None:
        raw = pipeline.get("analytics_reload_receipt")
        return _normalize_receipt(raw) if raw is not None else None
    raw = pipeline.get("analytics_reload_receipt")
    if raw is None:
        raw = pipeline.get("analytics_initial_receipt")
    return _normalize_receipt(raw) if raw is not None else None


def _phase_complete(
    phase: PhaseEvidence,
    *,
    expected_receipt: Mapping[str, object] | None,
) -> bool:
    advancing = (
        phase.target_messages >= MIN_ADVANCING_MESSAGES
        and phase.sequence_advances >= 1
        and phase.frame_advances >= 1
    )
    receipt_ready = phase.native_receipt is not None
    if phase.name == "baseline":
        return (
            advancing
            and receipt_ready
            and phase.real_person_frames >= MIN_BASELINE_PERSON_FRAMES
            and phase.real_person_frame_advances >= 1
        )
    if phase.name == "excluded":
        return (
            advancing
            and receipt_ready
            and phase.max_consecutive_empty_frames >= MIN_EXCLUDED_EMPTY_FRAMES
        )
    if phase.name == "restored":
        return advancing and receipt_ready and phase.real_person_frames >= 1
    raise GateError("phase_name_invalid")


async def _collect_phase(
    uri: str,
    auth: RequiredInternalAuth,
    *,
    name: str,
    camera_id: str,
    timeout_s: float,
    sequence_after: int | None = None,
    frame_after: int | None = None,
    expected_receipt: Mapping[str, object] | None = None,
) -> PhaseEvidence:
    phase = PhaseEvidence(name=name)
    deadline = time.monotonic() + float(timeout_s)
    async with connect_required_websocket(
        uri,
        auth,
        compression=None,
        max_size=4 * 1024 * 1024,
        open_timeout=min(10.0, float(timeout_s)),
        close_timeout=5.0,
    ) as websocket:
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(
                    websocket.recv(), timeout=min(2.0, remaining)
                )
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, str):
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            if payload.get("type") == "tracking":
                _record_tracking(phase, payload, camera_id=camera_id)
                if (
                    sequence_after is not None
                    and phase.first_sequence is not None
                    and phase.first_sequence <= sequence_after
                ):
                    raise GateError("target_frames_do_not_continue_across_phases")
                if (
                    frame_after is not None
                    and phase.first_frame_id is not None
                    and phase.first_frame_id <= frame_after
                ):
                    raise GateError("target_frames_do_not_continue_across_phases")
            elif payload.get("type") == "stats":
                stats = payload.get("payload")
                pipeline = stats.get("pipeline") if isinstance(stats, Mapping) else None
                if not isinstance(pipeline, Mapping):
                    continue
                errors = pipeline.get("errors")
                if isinstance(errors, list):
                    phase.pipeline_error_count = max(
                        phase.pipeline_error_count,
                        len([item for item in errors if str(item).strip()]),
                    )
                receipt = _select_stats_receipt(pipeline, expected=expected_receipt)
                if receipt is not None:
                    if expected_receipt is not None and receipt != dict(expected_receipt):
                        raise GateError("native_receipt_stats_mismatch")
                    phase.native_receipt = receipt
            if phase.pipeline_error_count:
                raise GateError("pipeline_errors_present")
            if _phase_complete(phase, expected_receipt=expected_receipt):
                return phase
    return phase


def _validate_phase_timeout(phase: PhaseEvidence) -> None:
    if (
        phase.target_messages < MIN_ADVANCING_MESSAGES
        or phase.sequence_advances < 1
        or phase.frame_advances < 1
    ):
        raise GateError(f"{phase.name}_target_frames_not_advancing")
    if phase.native_receipt is None:
        raise GateError(f"{phase.name}_native_receipt_unobserved")
    if phase.name == "baseline" and (
        phase.real_person_frames < MIN_BASELINE_PERSON_FRAMES
        or phase.real_person_frame_advances < 1
    ):
        raise GateError("baseline_real_person_absent")
    if (
        phase.name == "excluded"
        and phase.max_consecutive_empty_frames < MIN_EXCLUDED_EMPTY_FRAMES
    ):
        raise GateError("excluded_person_tracks_remain")
    if phase.name == "restored" and phase.real_person_frames < 1:
        raise GateError("restored_person_did_not_return")


def _spawn_runtime(
    args: argparse.Namespace, auth: RequiredInternalAuth
) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        str(REPO_ROOT / "noesis" / "ds8_runtime.py"),
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
        "--enable-rest",
        "--depth-enable-seconds",
        "0",
    ]
    environment = os.environ.copy()
    configure_required_auth_environment(environment, auth)
    environment.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    environment.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    return subprocess.Popen(command, cwd=str(REPO_ROOT), env=environment)


def _stop_runtime(process: subprocess.Popen[bytes]) -> None:
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _base_report(args: argparse.Namespace) -> dict[str, object]:
    return {
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "ok": False,
        "status": "fail",
        "camera_id": None,
        "target_stream_id": None,
        "checks": {
            "authenticated_rest_and_ws": False,
            "baseline_frames_advancing": False,
            "baseline_real_person": False,
            "temporary_stage_target_only": False,
            "temporary_native_receipt_exact": False,
            "excluded_frames_advancing": False,
            "person_tracks_disappeared": False,
            "objects_removed_count_rose": False,
            "restore_native_receipt_exact": False,
            "restored_get_semantic_equality": False,
            "restored_person_returned": False,
        },
        "stage_hashes": {},
        "receipts": {},
        "phases": {},
        "errors": [],
    }


def _append_error(report: dict[str, object], code: str) -> None:
    errors = report.setdefault("errors", [])
    if isinstance(errors, list) and code not in errors:
        errors.append(code)


def _run_gate(args: argparse.Namespace, auth: RequiredInternalAuth) -> dict[str, object]:
    report = _base_report(args)
    checks = report["checks"]
    assert isinstance(checks, dict)
    stage_hashes = report["stage_hashes"]
    receipts = report["receipts"]
    phases = report["phases"]
    assert isinstance(stage_hashes, dict)
    assert isinstance(receipts, dict)
    assert isinstance(phases, dict)

    original: dict[str, object] | None = None
    temporary: dict[str, object] | None = None
    original_receipt: dict[str, object] | None = None
    temporary_receipt: dict[str, object] | None = None
    target_stream_id: str | None = None
    canonical_camera = str(args.camera)
    mutation_started = False
    primary_error: GateError | None = None

    try:
        original = _get_original_stage(
            args.rest, auth, timeout_s=float(args.baseline_timeout)
        )
        target_stream_id, canonical_camera = _resolve_target_stream(
            original, str(args.camera), Path(args.cameras_config)
        )
        report["camera_id"] = canonical_camera
        report["target_stream_id"] = target_stream_id
        stage_hashes["original"] = _json_sha256(original)

        baseline = asyncio.run(
            _collect_phase(
                args.ws,
                auth,
                name="baseline",
                camera_id=canonical_camera,
                timeout_s=float(args.baseline_timeout),
            )
        )
        phases["baseline"] = baseline.public()
        _validate_phase_timeout(baseline)
        original_receipt = dict(baseline.native_receipt or {})
        _validate_initial_receipt(original_receipt)
        receipts["original"] = original_receipt
        checks["authenticated_rest_and_ws"] = True
        checks["baseline_frames_advancing"] = True
        checks["baseline_real_person"] = True

        # Refuse to overwrite a stage changed by another operator while the
        # occupied baseline was being collected.
        if _get_original_stage(
            args.rest, auth, timeout_s=float(args.baseline_timeout)
        ) != original:
            raise GateError("original_stage_changed_before_mutation")

        temporary = _temporary_full_frame_stage(original, target_stream_id)
        stage_hashes["temporary"] = _json_sha256(temporary)
        unchanged_original = copy.deepcopy(original)
        unchanged_temporary = copy.deepcopy(temporary)
        original_streams = {
            str(row["stream_id"]): row for row in original["streams"]  # type: ignore[index]
        }
        temporary_streams = {
            str(row["stream_id"]): row for row in temporary["streams"]  # type: ignore[index]
        }
        if any(
            temporary_streams[stream_id] != original_stream
            for stream_id, original_stream in original_streams.items()
            if stream_id != target_stream_id
        ):
            raise GateError("temporary_stage_changed_non_target_stream")
        if original != unchanged_original or temporary != unchanged_temporary:
            raise GateError("stage_snapshot_mutated_during_preparation")
        checks["temporary_stage_target_only"] = True

        mutation_started = True
        temporary_response = _post_stage(args.rest, auth, temporary)
        temporary_receipt = _validate_update_response(
            temporary_response, temporary
        )
        _validate_reload_receipt(
            temporary_receipt,
            previous=original_receipt,
        )
        if (
            temporary_receipt["active_config_sha256"]
            == original_receipt["active_config_sha256"]
        ):
            raise GateError("temporary_native_hash_did_not_change")
        receipts["temporary"] = temporary_receipt

        excluded = asyncio.run(
            _collect_phase(
                args.ws,
                auth,
                name="excluded",
                camera_id=canonical_camera,
                timeout_s=float(args.excluded_timeout),
                sequence_after=baseline.last_sequence,
                frame_after=baseline.last_frame_id,
                expected_receipt=temporary_receipt,
            )
        )
        phases["excluded"] = excluded.public()
        _validate_phase_timeout(excluded)
        checks["temporary_native_receipt_exact"] = True
        checks["excluded_frames_advancing"] = True
        checks["person_tracks_disappeared"] = True
    except GateError as exc:
        primary_error = exc
    except Exception as exc:
        primary_error = GateError(f"unexpected_{type(exc).__name__}")
    finally:
        if mutation_started:
            try:
                if original is None or original_receipt is None:
                    raise GateError("restore_snapshot_unavailable")
                restore_response = _post_stage(args.rest, auth, original)
                restore_receipt = _validate_update_response(
                    restore_response, original
                )
                previous_receipt = temporary_receipt or original_receipt
                _validate_reload_receipt(
                    restore_receipt,
                    previous=previous_receipt,
                    expected_hash=str(original_receipt["active_config_sha256"]),
                )
                receipts["restored"] = restore_receipt
                checks["restore_native_receipt_exact"] = True
                if temporary_receipt is None or int(
                    restore_receipt["objects_removed_count"]
                ) <= int(temporary_receipt["objects_removed_count"]):
                    raise GateError("objects_removed_count_did_not_rise")
                checks["objects_removed_count_rose"] = True

                restored_get = _get_original_stage(
                    args.rest, auth, timeout_s=float(args.restore_timeout)
                )
                stage_hashes["restored"] = _json_sha256(restored_get)
                if restored_get != original:
                    raise GateError("restored_get_semantic_mismatch")
                checks["restored_get_semantic_equality"] = True

                sequence_after = None
                frame_after = None
                excluded_public = phases.get("excluded")
                if isinstance(excluded_public, Mapping):
                    raw_last = excluded_public.get("last_sequence")
                    if _is_int(raw_last):
                        sequence_after = int(raw_last)
                    raw_last_frame = excluded_public.get("last_frame_id")
                    if _is_int(raw_last_frame):
                        frame_after = int(raw_last_frame)
                if sequence_after is None:
                    baseline_public = phases.get("baseline")
                    if isinstance(baseline_public, Mapping):
                        raw_last = baseline_public.get("last_sequence")
                        if _is_int(raw_last):
                            sequence_after = int(raw_last)
                if frame_after is None:
                    baseline_public = phases.get("baseline")
                    if isinstance(baseline_public, Mapping):
                        raw_last_frame = baseline_public.get("last_frame_id")
                        if _is_int(raw_last_frame):
                            frame_after = int(raw_last_frame)
                restored = asyncio.run(
                    _collect_phase(
                        args.ws,
                        auth,
                        name="restored",
                        camera_id=canonical_camera,
                        timeout_s=float(args.restore_timeout),
                        sequence_after=sequence_after,
                        frame_after=frame_after,
                        expected_receipt=restore_receipt,
                    )
                )
                phases["restored"] = restored.public()
                _validate_phase_timeout(restored)
                checks["restored_person_returned"] = True
            except GateError as exc:
                report["status"] = "restore_failed"
                _append_error(report, exc.code)
            except Exception as exc:
                report["status"] = "restore_failed"
                _append_error(report, f"restore_unexpected_{type(exc).__name__}")

    if primary_error is not None:
        if report["status"] != "restore_failed":
            report["status"] = (
                "blocked"
                if primary_error.code == "baseline_real_person_absent"
                else "fail"
            )
        _append_error(report, primary_error.code)
    elif report["status"] != "restore_failed":
        report["ok"] = all(bool(value) for value in checks.values())
        report["status"] = "pass" if report["ok"] else "fail"
    return report


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink():
        info = target.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise GateError("evidence_path_not_regular")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_EVIDENCE_BYTES:
            raise GateError("evidence_size_bound_exceeded")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600, follow_symlinks=False)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _positive_bounded_timeout(value: float, name: str) -> float:
    parsed = _finite_number(value)
    if parsed is None or parsed <= 0.0 or parsed > 300.0:
        raise GateError(f"{name}_invalid")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument(
        "--rest", default="http://127.0.0.1:8080", help="REST base URL"
    )
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/infer.yaml"))
    parser.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--no-spawn", action="store_true", help="Attach to an existing runtime")
    parser.add_argument(
        "--hot-restore",
        action="store_true",
        required=True,
        help="Explicitly authorize the temporary ROI transaction and mandatory restore",
    )
    parser.add_argument("--camera", required=True, help="Target camera name or source id")
    parser.add_argument("--baseline-timeout", type=float, default=45.0)
    parser.add_argument("--excluded-timeout", type=float, default=45.0)
    parser.add_argument("--restore-timeout", type=float, default=60.0)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Owner-only bounded JSON evidence path",
    )
    add_auth_token_file_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = _base_report(args)
    process: subprocess.Popen[bytes] | None = None
    try:
        args.baseline_timeout = _positive_bounded_timeout(
            args.baseline_timeout, "baseline_timeout"
        )
        args.excluded_timeout = _positive_bounded_timeout(
            args.excluded_timeout, "excluded_timeout"
        )
        args.restore_timeout = _positive_bounded_timeout(
            args.restore_timeout, "restore_timeout"
        )
        auth = load_required_internal_auth(args.auth_token_file)
        if not args.no_spawn:
            process = _spawn_runtime(args, auth)
        report = _run_gate(args, auth)
    except GateError as exc:
        _append_error(report, exc.code)
    except Exception as exc:
        _append_error(report, f"startup_unexpected_{type(exc).__name__}")
    finally:
        if process is not None:
            try:
                _stop_runtime(process)
            except Exception as exc:
                report["ok"] = False
                report["status"] = "fail"
                _append_error(report, f"runtime_stop_{type(exc).__name__}")

    try:
        _write_private_json(args.evidence, report)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "contract": CONTRACT,
                    "ok": False,
                    "status": "fail",
                    "errors": [f"evidence_write_{type(exc).__name__}"],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") == "blocked":
        return 2
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
