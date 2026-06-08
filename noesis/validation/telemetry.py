from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bev import validate_path_smoothness
from .core import CheckStatus, FailureType, SourceMetadata, ValidationCheck, ValidationReport
from .tracking import (
    BevPointSample,
    TrackSample,
    validate_bev_track_agreement,
    validate_identity_continuity,
    validate_occlusion_bridges,
    validate_reid_geometry_consistency,
    validate_track_motion,
)


@dataclass(frozen=True)
class TelemetrySamples:
    messages: tuple[Mapping[str, Any], ...] = ()
    track_samples: tuple[TrackSample, ...] = ()
    bev_points: tuple[BevPointSample, ...] = ()
    tracking_message_count: int = 0
    bev_message_count: int = 0
    raw_message_count: int = 0
    source_counts: Mapping[str, int] = field(default_factory=dict)


def _safe_int(value: Any) -> int | None:
    if value in (None, "", -1):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _first_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def _safe_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def _warnings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return tuple(out)
    return ()


def _timestamp_seconds(payload: Mapping[str, Any], fallback_s: float, *, ts_default_scale: float = 1.0) -> float:
    for key, scale in (
        ("ts_s", 1.0),
        ("timestamp_s", 1.0),
        ("time_s", 1.0),
        ("ts_us", 1e-6),
        ("timestamp_us", 1e-6),
        ("ts_ms", 1e-3),
        ("timestamp_ms", 1e-3),
        ("ts_ns", 1e-9),
        ("timestamp_ns", 1e-9),
    ):
        value = _safe_float(payload.get(key))
        if value is not None:
            return float(value * scale)
    value = _safe_float(payload.get("ts"))
    if value is None:
        value = _safe_float(payload.get("timestamp"))
    if value is None:
        return float(fallback_s)
    # DS8 BEV contracts use microseconds for `ts`; epoch milliseconds and
    # nanoseconds also show up in diagnostics. Prefer magnitude over guessing.
    if value > 1e17:
        return float(value / 1e9)
    if value > 1e14:
        return float(value / 1e6)
    if value > 1e11:
        return float(value / 1e3)
    return float(value * ts_default_scale)


def _finite_world(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 3:
        return None
    out = [_safe_float(v) for v in value]
    if any(v is None for v in out):
        return None
    return [float(out[0]), float(out[1]), float(out[2])]  # type: ignore[arg-type]


def read_ndjson(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"expected JSON object at line {line_no}: {path}")
        rows.append(payload)
    return rows


def write_ndjson(messages: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(dict(message), separators=(",", ":"), ensure_ascii=True) + "\n")
    return target


def extract_telemetry_samples(messages: Iterable[Mapping[str, Any]]) -> TelemetrySamples:
    raw_messages: list[Mapping[str, Any]] = []
    track_samples: list[TrackSample] = []
    bev_points: list[BevPointSample] = []
    tracking_count = 0
    bev_count = 0
    source_counts: dict[str, int] = defaultdict(int)

    for message_idx, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        raw_messages.append(message)
        msg_type = str(message.get("type") or "")
        if msg_type:
            source_counts[msg_type] += 1
        fallback_ts = float(message_idx) / 30.0
        if msg_type == "tracking":
            tracking_count += 1
            payload_ts = _timestamp_seconds(message, fallback_ts)
            top_camera = message.get("camera_id") or message.get("cameraId")
            tracks = message.get("tracks")
            if not isinstance(tracks, list):
                continue
            for track_idx, track in enumerate(tracks):
                if not isinstance(track, Mapping):
                    continue
                world = _finite_world(track.get("world"))
                if world is None or track.get("world_valid") is not True:
                    continue
                ts_s = _timestamp_seconds(track, payload_ts + track_idx * 1e-6)
                camera_id = track.get("camera_id") or track.get("cameraId") or top_camera
                track_samples.append(
                    TrackSample(
                        ts_s=ts_s,
                        world=world,
                        camera_id=str(camera_id) if camera_id not in (None, "") else None,
                        stable_id=_safe_int(track.get("stable_id") if "stable_id" in track else track.get("stableId")),
                        tracker_id=_safe_int(track.get("tracker_id") if "tracker_id" in track else track.get("trackerId", track.get("track_id"))),
                        room=str(track.get("zone") or track.get("room") or "") or None,
                        confidence=_safe_float(track.get("confidence")),
                        projection_confidence=_first_float(track, "projection_confidence", "projectionConfidence", "world_confidence", "worldConfidence"),
                        temporal_confidence=_first_float(track, "temporal_confidence", "temporalConfidence", "track_confidence", "trackConfidence"),
                        reid_confidence=_first_float(track, "reid_confidence", "reidConfidence", "appearance_confidence", "appearanceConfidence"),
                        reid_identity=_safe_label(_first_present(track, "reid_identity", "reidIdentity", "reid_id", "reidId")),
                        appearance_id=_safe_label(
                            _first_present(track, "appearance_id", "appearanceId", "appearance_cluster", "appearanceCluster", "appearance_label", "appearanceLabel")
                        ),
                        occluded=bool(track.get("occluded")) if track.get("occluded") is not None else None,
                        occlusion_uncertainty_m=_first_float(track, "occlusion_uncertainty_m", "occlusionUncertaintyM", "uncertainty_m", "uncertaintyM"),
                        warnings=_warnings(track.get("warnings") or track.get("warning")),
                    )
                )
        elif msg_type == "bev-frame":
            bev_count += 1
            payload_ts = _timestamp_seconds(message, fallback_ts, ts_default_scale=1e-6)
            camera_id = message.get("cameraId") or message.get("camera_id")
            footpoints = message.get("footpoints")
            if not isinstance(footpoints, list):
                continue
            for idx, footpoint in enumerate(footpoints):
                if not isinstance(footpoint, Mapping):
                    continue
                x = _safe_float(footpoint.get("x"))
                z = _safe_float(footpoint.get("z") if footpoint.get("z") is not None else footpoint.get("y"))
                if x is None or z is None:
                    continue
                bev_points.append(
                    BevPointSample(
                        ts_s=_timestamp_seconds(footpoint, payload_ts + idx * 1e-6, ts_default_scale=1e-3),
                        x=float(x),
                        z=float(z),
                        camera_id=str(camera_id) if camera_id not in (None, "") else None,
                        stable_id=_safe_int(footpoint.get("stableId") if "stableId" in footpoint else footpoint.get("stable_id")),
                        tracker_id=_safe_int(footpoint.get("trackerId") if "trackerId" in footpoint else footpoint.get("tracker_id")),
                    )
                )

    return TelemetrySamples(
        messages=tuple(raw_messages),
        track_samples=tuple(track_samples),
        bev_points=tuple(bev_points),
        tracking_message_count=tracking_count,
        bev_message_count=bev_count,
        raw_message_count=len(raw_messages),
        source_counts=dict(sorted(source_counts.items())),
    )


def validate_tracking_contract(
    samples: TelemetrySamples,
    *,
    expected_world_frame: str = "backend_world_m",
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    tracking_messages = [message for message in samples.messages if message.get("type") == "tracking"]
    checks.append(
        ValidationCheck(
            id="TELEMETRY.tracking.messages",
            domain="tracking",
            name="tracking_messages_present",
            status=CheckStatus.PASS if tracking_messages else CheckStatus.BLOCKED,
            failure_type=None if tracking_messages else FailureType.INFRASTRUCTURE,
            metric={"tracking_message_count": len(tracking_messages), "raw_message_count": samples.raw_message_count},
            detail="Tracking messages were observed." if tracking_messages else "No tracking messages were observed.",
            suggested_next_diagnostic=None if tracking_messages else "Capture DS8 WebSocket telemetry while the runtime is active.",
        )
    )
    world_valid_tracks = 0
    expected_frame_count = 0
    stable_id_count = 0
    for message in tracking_messages:
        for track in message.get("tracks") or []:
            if not isinstance(track, Mapping):
                continue
            if track.get("world_valid") is not True or _finite_world(track.get("world")) is None:
                continue
            world_valid_tracks += 1
            if track.get("world_frame") == expected_world_frame:
                expected_frame_count += 1
            sid = _safe_int(track.get("stable_id") if "stable_id" in track else track.get("stableId"))
            if sid is not None and sid >= 1:
                stable_id_count += 1
    checks.append(
        ValidationCheck(
            id="TELEMETRY.tracking.world_valid",
            domain="tracking",
            name="tracking_world_valid_samples",
            status=CheckStatus.PASS if world_valid_tracks > 0 else CheckStatus.BLOCKED,
            failure_type=None if world_valid_tracks > 0 else FailureType.PROJECTION,
            metric={"world_valid_track_count": world_valid_tracks},
            detail="World-valid tracking samples were observed." if world_valid_tracks else "No world-valid tracking samples were observed.",
            suggested_next_diagnostic=None if world_valid_tracks else "Check calibration/depth/world estimator status fields.",
        )
    )
    if world_valid_tracks > 0:
        frame_status = CheckStatus.PASS if expected_frame_count == world_valid_tracks else CheckStatus.FAIL
        checks.append(
            ValidationCheck(
                id="TELEMETRY.tracking.world_frame",
                domain="tracking",
                name="tracking_world_frame_contract",
                status=frame_status,
                failure_type=None if frame_status == CheckStatus.PASS else FailureType.TRANSFORM,
                metric={
                    "expected_world_frame": expected_world_frame,
                    "matching_world_frame_count": expected_frame_count,
                    "world_valid_track_count": world_valid_tracks,
                },
                detail="All world-valid tracks declare the expected world frame."
                if frame_status == CheckStatus.PASS
                else "One or more world-valid tracks do not declare the expected world frame.",
                suggested_next_diagnostic=None if frame_status == CheckStatus.PASS else "Audit backend world-frame ownership before validating BEV/Menon placement.",
            )
        )
        sid_status = CheckStatus.PASS if stable_id_count == world_valid_tracks else CheckStatus.WARNING
        checks.append(
            ValidationCheck(
                id="TELEMETRY.tracking.stable_id",
                domain="tracking",
                name="stable_id_presence",
                status=sid_status,
                failure_type=None if sid_status == CheckStatus.PASS else FailureType.TEMPORAL,
                metric={"stable_id_count": stable_id_count, "world_valid_track_count": world_valid_tracks},
                detail="All world-valid tracks carry stable_id." if sid_status == CheckStatus.PASS else "Some world-valid tracks are missing stable_id.",
                suggested_next_diagnostic=None if sid_status == CheckStatus.PASS else "Check StableIDManager/ReID/pose identity pipeline.",
            )
        )
    return checks


def validate_bev_contract(
    samples: TelemetrySamples,
    *,
    expected_world_frame: str = "backend_world_m",
    expected_frame_mode: str = "world",
) -> list[ValidationCheck]:
    bev_messages = [message for message in samples.messages if message.get("type") == "bev-frame"]
    checks = [
        ValidationCheck(
            id="TELEMETRY.bev.messages",
            domain="bev",
            name="bev_messages_present",
            status=CheckStatus.PASS if bev_messages else CheckStatus.BLOCKED,
            failure_type=None if bev_messages else FailureType.INFRASTRUCTURE,
            metric={"bev_message_count": len(bev_messages), "raw_message_count": samples.raw_message_count},
            detail="BEV messages were observed." if bev_messages else "No BEV messages were observed.",
            suggested_next_diagnostic=None if bev_messages else "Enable BEV telemetry or capture a longer WebSocket window.",
        )
    ]
    if not bev_messages:
        return checks
    matching = 0
    for message in bev_messages:
        if message.get("world_frame") == expected_world_frame and message.get("frame_mode") == expected_frame_mode:
            matching += 1
    if matching == len(bev_messages):
        status = CheckStatus.PASS
        failure_type = None
        detail = "All BEV frames declare the expected world frame and mode."
    elif matching > 0:
        status = CheckStatus.WARNING
        failure_type = FailureType.TRANSFORM
        detail = "Some BEV frames declare the expected world frame and mode."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.TRANSFORM
        detail = "No BEV frames declare the expected world frame and mode."
    checks.append(
        ValidationCheck(
            id="TELEMETRY.bev.frame_contract",
            domain="bev",
            name="bev_frame_contract",
            status=status,
            failure_type=failure_type,
            metric={
                "expected_world_frame": expected_world_frame,
                "expected_frame_mode": expected_frame_mode,
                "matching_frame_count": matching,
                "bev_message_count": len(bev_messages),
            },
            detail=detail,
            suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check BEV frame config and producer-owned world mode.",
        )
    )
    return checks


def _audit_confidence(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(finite[-1])


def _derived_temporal_confidence(max_speed_m_s: float, impossible_event: Mapping[str, Any] | None) -> float:
    if impossible_event is not None:
        return 0.25
    if max_speed_m_s <= 1.5:
        return 0.95
    if max_speed_m_s <= 3.0:
        return 0.80
    if max_speed_m_s <= 4.5:
        return 0.65
    return 0.35


def build_track_audit(
    samples: Sequence[TrackSample],
    *,
    max_speed_m_s: float = 4.5,
    max_accel_m_s2: float = 8.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str | None, Any], list[TrackSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.camera_id, sample.identity)].append(sample)
    rows: list[dict[str, Any]] = []
    for (camera_id, identity), values in sorted(grouped.items(), key=lambda item: str(item[0])):
        values.sort(key=lambda row: row.ts_s)
        latest = values[-1]
        speeds: list[float] = []
        timed_speeds: list[tuple[float, float]] = []
        impossible_event: dict[str, Any] | None = None
        for left, right in zip(values, values[1:]):
            dt = float(right.ts_s) - float(left.ts_s)
            if dt <= 1e-6:
                continue
            a = np.asarray(left.world, dtype=np.float64)
            b = np.asarray(right.world, dtype=np.float64)
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                speed = float(np.linalg.norm(b - a) / dt)
                speeds.append(speed)
                timed_speeds.append((float(right.ts_s), speed))
                if speed > max_speed_m_s and impossible_event is None:
                    impossible_event = {
                        "type": "speed_limit",
                        "ts_s": float(right.ts_s),
                        "speed_m_s": speed,
                        "threshold_m_s": max_speed_m_s,
                    }
        for left, right in zip(timed_speeds, timed_speeds[1:]):
            dt = float(right[0]) - float(left[0])
            if dt <= 1e-6:
                continue
            accel = abs(float(right[1]) - float(left[1])) / dt
            if accel > max_accel_m_s2:
                impossible_event = {
                    "type": "acceleration_limit",
                    "ts_s": float(right[0]),
                    "accel_m_s2": accel,
                    "threshold_m_s2": max_accel_m_s2,
                }
                break
        last_transition = None
        previous_room = values[0].room
        for value in values[1:]:
            if value.room and previous_room and value.room != previous_room:
                last_transition = {
                    "ts_s": float(value.ts_s),
                    "from_room": previous_room,
                    "to_room": value.room,
                }
            if value.room:
                previous_room = value.room
        last_occlusion_ts = None
        for value in values:
            if value.occluded:
                last_occlusion_ts = float(value.ts_s)
        warnings = sorted({warning for value in values for warning in value.warnings})
        if impossible_event is not None:
            warnings.append(f"impossible_motion:{impossible_event['type']}")
        projection_confidence = _audit_confidence([value.projection_confidence for value in values])
        temporal_confidence = _audit_confidence([value.temporal_confidence for value in values])
        reid_confidence = _audit_confidence([value.reid_confidence for value in values])
        if temporal_confidence is None:
            temporal_confidence = _derived_temporal_confidence(float(max(speeds)) if speeds else 0.0, impossible_event)
        appearance_keys = sorted({value.appearance_key for value in values if value.appearance_key is not None})
        rows.append(
            {
                "camera_id": camera_id,
                "identity": list(identity) if isinstance(identity, tuple) else str(identity),
                "stable_id": latest.stable_id,
                "tracker_id": latest.tracker_id,
                "reid_identity": latest.reid_identity,
                "appearance_id": latest.appearance_id,
                "appearance_key": latest.appearance_key,
                "appearance_keys_observed": appearance_keys,
                "current_room": latest.room,
                "world_position": {"x": float(latest.world[0]), "y": float(latest.world[1]), "z": float(latest.world[2])},
                "current_world_position": [float(v) for v in latest.world],
                "projection_confidence": projection_confidence,
                "temporal_confidence": temporal_confidence,
                "reid_confidence": reid_confidence,
                "last_doorway_transition": last_transition,
                "last_occlusion_age_s": None if last_occlusion_ts is None else max(0.0, float(latest.ts_s) - last_occlusion_ts),
                "current_occlusion_uncertainty_m": latest.occlusion_uncertainty_m,
                "last_impossible_motion_event": impossible_event,
                "warnings": warnings,
                "sample_count": len(values),
                "first_ts_s": float(values[0].ts_s),
                "last_ts_s": float(values[-1].ts_s),
                "p95_speed_m_s": float(np.percentile(np.asarray(speeds), 95)) if speeds else 0.0,
                "max_speed_m_s": float(max(speeds)) if speeds else 0.0,
            }
        )
    return rows


def build_telemetry_report(
    messages: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    source: SourceMetadata | None = None,
    expected_world_frame: str = "backend_world_m",
    expected_bev_frame_mode: str = "world",
) -> ValidationReport:
    samples = extract_telemetry_samples(messages)
    report = ValidationReport(
        run_id=run_id,
        source=source or SourceMetadata(),
        scope={
            "tiers": ["runtime" if samples.raw_message_count else "offline"],
            "message_counts": dict(samples.source_counts),
        },
    )
    report.add_check(
        ValidationCheck(
            id="TELEMETRY.messages",
            domain="core",
            name="telemetry_messages_present",
            status=CheckStatus.PASS if samples.raw_message_count > 0 else CheckStatus.BLOCKED,
            failure_type=None if samples.raw_message_count > 0 else FailureType.INFRASTRUCTURE,
            metric={"raw_message_count": samples.raw_message_count, "source_counts": dict(samples.source_counts)},
            detail="Telemetry messages were loaded." if samples.raw_message_count else "No telemetry messages were loaded.",
            suggested_next_diagnostic=None if samples.raw_message_count else "Check NDJSON input path or live WebSocket connection.",
        )
    )
    for check in validate_tracking_contract(samples, expected_world_frame=expected_world_frame):
        report.add_check(check)
    for check in validate_bev_contract(samples, expected_world_frame=expected_world_frame, expected_frame_mode=expected_bev_frame_mode):
        report.add_check(check)
    for check in validate_track_motion(samples.track_samples):
        report.add_check(check)
    report.add_check(validate_path_smoothness(samples.track_samples))
    report.add_check(validate_occlusion_bridges(samples.track_samples))
    report.add_check(validate_identity_continuity(samples.track_samples))
    report.add_check(validate_reid_geometry_consistency(samples.track_samples))
    report.add_check(validate_bev_track_agreement(samples.track_samples, samples.bev_points))
    return report
