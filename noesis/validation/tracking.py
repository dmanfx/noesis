from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np

from .core import CheckStatus, FailureType, ValidationCheck
from .geometry import point_in_polygon_xz, segments_intersect_2d


@dataclass(frozen=True)
class TrackSample:
    ts_s: float
    world: Sequence[float]
    camera_id: str | None = None
    stable_id: int | None = None
    tracker_id: int | None = None
    room: str | None = None
    confidence: float | None = None
    projection_confidence: float | None = None
    temporal_confidence: float | None = None
    reid_confidence: float | None = None
    reid_identity: str | None = None
    appearance_id: str | None = None
    occluded: bool | None = None
    occlusion_uncertainty_m: float | None = None
    warnings: Sequence[str] = ()

    @property
    def identity(self) -> Hashable:
        if self.stable_id not in (None, -1):
            return ("stable", int(self.stable_id))
        if self.tracker_id not in (None, -1):
            return ("tracker", int(self.tracker_id))
        return ("anonymous", self.camera_id or "")

    @property
    def appearance_key(self) -> str | None:
        for value in (self.reid_identity, self.appearance_id):
            if value is None:
                continue
            label = str(value).strip()
            if label and label.lower() not in {"-1", "none", "null", "unknown"}:
                return label
        return None


@dataclass(frozen=True)
class BevPointSample:
    ts_s: float
    x: float
    z: float
    camera_id: str | None = None
    stable_id: int | None = None
    tracker_id: int | None = None

    @property
    def identity(self) -> Hashable:
        if self.stable_id not in (None, -1):
            return ("stable", int(self.stable_id))
        if self.tracker_id not in (None, -1):
            return ("tracker", int(self.tracker_id))
        return ("anonymous", self.camera_id or "")


@dataclass(frozen=True)
class DetectionProjectionSample:
    entity_id: str
    world_point: Sequence[float]
    bbox_foot_world: Sequence[float] | None = None
    mask_foot_world: Sequence[float] | None = None
    pose_foot_world: Sequence[float] | None = None
    depth_world: Sequence[float] | None = None
    image_bbox_xyxy: Sequence[float] | None = None
    projected_bbox_xyxy: Sequence[float] | None = None
    projected_bbox_iou: float | None = None
    bbox_center_error_px: float | None = None
    reprojection_score: float | None = None
    temporal_smoothness_score: float | None = None
    semantic_validity_score: float | None = None
    floor_y: float = 0.0
    person_height_m: float | None = None
    ray_floor_valid: bool = True
    room_polygon_xz: Sequence[Sequence[float]] = ()
    camera_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


def _world_array(point: Sequence[float]) -> np.ndarray | None:
    try:
        arr = np.asarray(point, dtype=np.float64).reshape((3,))
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _optional_world_array(point: Sequence[float] | None) -> np.ndarray | None:
    if point is None:
        return None
    return _world_array(point)


def _bbox_xyxy(value: Sequence[float] | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape((4,))
    except Exception:
        return None
    if not np.all(np.isfinite(arr)) or arr[2] <= arr[0] or arr[3] <= arr[1]:
        return None
    return arr


def _bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return 0.0 if union <= 1e-12 else float(intersection / union)


def _bbox_center_error_px(left: np.ndarray, right: np.ndarray) -> float:
    left_center = np.asarray([(left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5], dtype=np.float64)
    right_center = np.asarray([(right[0] + right[2]) * 0.5, (right[1] + right[3]) * 0.5], dtype=np.float64)
    return float(np.linalg.norm(left_center - right_center))


def _score_or_zero(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.0


def _point_segment_distance_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def _segment_distance_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if segments_intersect_2d(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance_2d(a, c, d),
        _point_segment_distance_2d(b, c, d),
        _point_segment_distance_2d(c, a, b),
        _point_segment_distance_2d(d, a, b),
    )


def validate_track_motion(
    samples: Sequence[TrackSample],
    *,
    check_id_prefix: str = "TRACK.motion",
    max_speed_m_s: float = 4.5,
    max_accel_m_s2: float = 8.0,
    idle_speed_m_s: float = 0.15,
    idle_jitter_warning_m: float = 0.20,
) -> list[ValidationCheck]:
    grouped: dict[Hashable, list[TrackSample]] = defaultdict(list)
    for sample in samples:
        if _world_array(sample.world) is not None:
            grouped[(sample.camera_id, sample.identity)].append(sample)
    if not grouped:
        return [
            ValidationCheck(
                id=f"{check_id_prefix}.blocked",
                domain="tracking",
                name="track_motion",
                status=CheckStatus.BLOCKED,
                failure_type=FailureType.INFRASTRUCTURE,
                detail="No finite track samples were provided.",
                suggested_next_diagnostic="Capture tracking telemetry with world_valid=true.",
            )
        ]
    speeds: list[float] = []
    accelerations: list[float] = []
    idle_jitters: list[float] = []
    impossible_steps = 0
    impossible_accels = 0
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row.ts_s))
        row_speeds: list[tuple[float, float]] = []
        for left, right in zip(rows, rows[1:]):
            dt = float(right.ts_s) - float(left.ts_s)
            if dt <= 1e-6:
                continue
            a = _world_array(left.world)
            b = _world_array(right.world)
            if a is None or b is None:
                continue
            speed = float(np.linalg.norm(b - a) / dt)
            speeds.append(speed)
            row_speeds.append((float(right.ts_s), speed))
            if speed > max_speed_m_s:
                impossible_steps += 1
        for left, right in zip(row_speeds, row_speeds[1:]):
            dt = float(right[0]) - float(left[0])
            if dt <= 1e-6:
                continue
            accel = abs(float(right[1]) - float(left[1])) / dt
            accelerations.append(accel)
            if accel > max_accel_m_s2:
                impossible_accels += 1
        if rows and speeds:
            slow_rows = []
            for row in rows:
                if row.confidence is None or row.confidence >= 0.5:
                    point = _world_array(row.world)
                    if point is not None:
                        slow_rows.append(point)
            if len(slow_rows) >= 3 and (not row_speeds or np.median([s for _, s in row_speeds]) <= idle_speed_m_s):
                pts = np.asarray(slow_rows, dtype=np.float64)
                centroid = np.mean(pts, axis=0)
                jitter = float(np.max(np.linalg.norm(pts - centroid, axis=1)))
                idle_jitters.append(jitter)

    speed_status = CheckStatus.PASS if impossible_steps == 0 else CheckStatus.FAIL
    accel_status = CheckStatus.PASS if impossible_accels == 0 else CheckStatus.FAIL
    jitter_max = float(max(idle_jitters)) if idle_jitters else 0.0
    jitter_status = CheckStatus.PASS if jitter_max <= idle_jitter_warning_m else CheckStatus.WARNING
    p95_speed = float(np.percentile(np.asarray(speeds), 95)) if speeds else 0.0
    p95_accel = float(np.percentile(np.asarray(accelerations), 95)) if accelerations else 0.0
    return [
        ValidationCheck(
            id=f"{check_id_prefix}.speed",
            domain="tracking",
            name="human_speed_limit",
            status=speed_status,
            failure_type=None if speed_status == CheckStatus.PASS else FailureType.TEMPORAL,
            metric={"p95_speed_m_s": p95_speed, "impossible_step_count": impossible_steps, "sample_count": len(speeds)},
            threshold={"max_speed_m_s": max_speed_m_s},
            detail="Track speeds are plausible." if speed_status == CheckStatus.PASS else "One or more track steps exceed the human speed limit.",
            suggested_next_diagnostic=None if speed_status == CheckStatus.PASS else "Inspect timestamp sync, ID switches, and world transform jumps.",
        ),
        ValidationCheck(
            id=f"{check_id_prefix}.acceleration",
            domain="tracking",
            name="human_acceleration_limit",
            status=accel_status,
            failure_type=None if accel_status == CheckStatus.PASS else FailureType.TEMPORAL,
            metric={"p95_accel_m_s2": p95_accel, "impossible_accel_count": impossible_accels, "sample_count": len(accelerations)},
            threshold={"max_accel_m_s2": max_accel_m_s2},
            detail="Track accelerations are plausible." if accel_status == CheckStatus.PASS else "One or more track accelerations are implausible.",
            suggested_next_diagnostic=None if accel_status == CheckStatus.PASS else "Check smoothing ownership, occlusion reacquisition, and StableID continuity.",
        ),
        ValidationCheck(
            id=f"{check_id_prefix}.idle_jitter",
            domain="tracking",
            name="stationary_jitter",
            status=jitter_status,
            failure_type=None if jitter_status == CheckStatus.PASS else FailureType.TEMPORAL,
            metric={"max_idle_jitter_m": jitter_max, "track_count": len(idle_jitters)},
            threshold={"warning_jitter_m": idle_jitter_warning_m},
            detail="Idle track jitter is within tolerance." if jitter_status == CheckStatus.PASS else "Idle track jitter is above the warning threshold.",
            suggested_next_diagnostic=None if jitter_status == CheckStatus.PASS else "Check projection confidence near furniture edges and duplicate smoothing stages.",
        ),
    ]


def validate_occlusion_bridges(
    samples: Sequence[TrackSample],
    *,
    check_id: str = "TRACK.occlusion_bridge",
    max_bridge_s: float = 5.0,
    max_bridge_speed_m_s: float = 3.0,
    min_occlusion_uncertainty_m: float = 0.30,
) -> ValidationCheck:
    grouped: dict[Hashable, list[TrackSample]] = defaultdict(list)
    for sample in samples:
        if _world_array(sample.world) is not None:
            grouped[(sample.camera_id, sample.identity)].append(sample)
    if not grouped:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="occlusion_bridge",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No finite track samples were provided for occlusion bridge validation.",
            suggested_next_diagnostic="Capture ordered tracking samples with occlusion flags and world positions.",
        )
    bridge_count = 0
    implausible_bridge_count = 0
    unresolved_occlusion_count = 0
    low_uncertainty_count = 0
    max_bridge_speed = 0.0
    max_bridge_duration = 0.0
    occluded_sample_count = 0
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row.ts_s))
        idx = 0
        while idx < len(rows):
            if not rows[idx].occluded:
                idx += 1
                continue
            run_start = idx
            while idx < len(rows) and rows[idx].occluded:
                idx += 1
            run_end = idx - 1
            occluded_rows = rows[run_start : run_end + 1]
            occluded_sample_count += len(occluded_rows)
            for row in occluded_rows:
                if row.occlusion_uncertainty_m is not None and float(row.occlusion_uncertainty_m) < min_occlusion_uncertainty_m:
                    low_uncertainty_count += 1
            prev_visible = rows[run_start - 1] if run_start > 0 and not rows[run_start - 1].occluded else None
            next_visible = rows[idx] if idx < len(rows) and not rows[idx].occluded else None
            if prev_visible is None or next_visible is None:
                unresolved_occlusion_count += 1
                continue
            prev_point = _world_array(prev_visible.world)
            next_point = _world_array(next_visible.world)
            if prev_point is None or next_point is None:
                unresolved_occlusion_count += 1
                continue
            duration = float(next_visible.ts_s) - float(prev_visible.ts_s)
            if duration <= 1e-6:
                implausible_bridge_count += 1
                continue
            speed = float(np.linalg.norm(next_point - prev_point) / duration)
            bridge_count += 1
            max_bridge_speed = max(max_bridge_speed, speed)
            max_bridge_duration = max(max_bridge_duration, duration)
            if duration > max_bridge_s or speed > max_bridge_speed_m_s:
                implausible_bridge_count += 1
    if occluded_sample_count == 0:
        status = CheckStatus.PASS
        failure_type = None
        detail = "No occluded samples were observed; no occlusion bridge was required."
    elif implausible_bridge_count:
        status = CheckStatus.FAIL
        failure_type = FailureType.TEMPORAL
        detail = "One or more occlusion bridges imply impossible duration or speed."
    elif unresolved_occlusion_count or low_uncertainty_count:
        status = CheckStatus.WARNING
        failure_type = FailureType.TEMPORAL
        detail = "Occlusion bridges are plausible but have unresolved endpoints or understated uncertainty."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Occluded track segments bridge visible samples plausibly and carry uncertainty evidence."
    return ValidationCheck(
        id=check_id,
        domain="tracking",
        name="occlusion_bridge",
        status=status,
        failure_type=failure_type,
        metric={
            "occluded_sample_count": occluded_sample_count,
            "bridge_count": bridge_count,
            "implausible_bridge_count": implausible_bridge_count,
            "unresolved_occlusion_count": unresolved_occlusion_count,
            "low_uncertainty_count": low_uncertainty_count,
            "max_bridge_speed_m_s": max_bridge_speed,
            "max_bridge_duration_s": max_bridge_duration,
        },
        threshold={"max_bridge_s": max_bridge_s, "max_bridge_speed_m_s": max_bridge_speed_m_s, "min_occlusion_uncertainty_m": min_occlusion_uncertainty_m},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check occlusion state transitions, stale-hold duration, uncertainty widening, and identity reacquisition.",
    )


def validate_doorway_transitions(
    samples: Sequence[TrackSample],
    doorway_segments_xz: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    check_id: str = "TRACK.doorway_transitions",
    max_doorway_distance_m: float = 0.30,
) -> ValidationCheck:
    grouped: dict[Hashable, list[TrackSample]] = defaultdict(list)
    for sample in samples:
        if _world_array(sample.world) is not None:
            grouped[(sample.camera_id, sample.identity)].append(sample)
    if not grouped:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="doorway_transitions",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No finite track samples were provided for doorway transition validation.",
            suggested_next_diagnostic="Capture ordered tracking samples with room labels and world positions.",
        )
    parsed_doorways: list[tuple[np.ndarray, np.ndarray]] = []
    for raw_a, raw_b in doorway_segments_xz:
        try:
            a = np.asarray(raw_a, dtype=np.float64).reshape(2)
            b = np.asarray(raw_b, dtype=np.float64).reshape(2)
        except Exception:
            continue
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            parsed_doorways.append((a, b))
    transition_count = 0
    invalid_transition_count = 0
    min_transition_doorway_distance = math.inf
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row.ts_s))
        for left, right in zip(rows, rows[1:]):
            if not left.room or not right.room or left.room == right.room:
                continue
            transition_count += 1
            left_world = _world_array(left.world)
            right_world = _world_array(right.world)
            if left_world is None or right_world is None:
                invalid_transition_count += 1
                continue
            path_a = np.asarray([left_world[0], left_world[2]], dtype=np.float64)
            path_b = np.asarray([right_world[0], right_world[2]], dtype=np.float64)
            distances = [_segment_distance_2d(path_a, path_b, door_a, door_b) for door_a, door_b in parsed_doorways]
            if not distances:
                invalid_transition_count += 1
                continue
            best = float(min(distances))
            min_transition_doorway_distance = min(min_transition_doorway_distance, best)
            if best > max_doorway_distance_m:
                invalid_transition_count += 1
    if transition_count == 0:
        status = CheckStatus.PASS
        failure_type = None
        detail = "No room transitions were observed; no doorway crossing was required."
    elif invalid_transition_count == 0:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Observed room transitions pass through known doorway segments."
    elif parsed_doorways:
        status = CheckStatus.FAIL
        failure_type = FailureType.SEMANTIC
        detail = "One or more room transitions do not pass through known doorway segments."
    else:
        status = CheckStatus.BLOCKED
        failure_type = FailureType.INFRASTRUCTURE
        detail = "Room transitions were observed, but no doorway segment evidence was provided."
    return ValidationCheck(
        id=check_id,
        domain="tracking",
        name="doorway_transitions",
        status=status,
        failure_type=failure_type,
        metric={
            "transition_count": transition_count,
            "invalid_transition_count": invalid_transition_count,
            "doorway_segment_count": len(parsed_doorways),
            "min_transition_doorway_distance_m": None if not math.isfinite(min_transition_doorway_distance) else min_transition_doorway_distance,
        },
        threshold={"max_doorway_distance_m": max_doorway_distance_m},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check room labels, doorway geometry, world frame, and smoothing around room transitions.",
    )


def validate_identity_continuity(
    samples: Sequence[TrackSample],
    *,
    check_id: str = "TRACK.identity_continuity",
    max_switch_gap_s: float = 2.0,
    max_simultaneous_dt_s: float = 0.10,
    duplicate_stable_fail_distance_m: float = 0.75,
) -> ValidationCheck:
    finite_samples = [sample for sample in samples if _world_array(sample.world) is not None]
    if not finite_samples:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="identity_continuity",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No finite track samples were provided for identity continuity validation.",
            suggested_next_diagnostic="Capture tracking samples with stable_id, tracker_id, timestamps, and backend_world_m positions.",
        )
    stable_samples = [sample for sample in finite_samples if sample.stable_id not in (None, -1)]
    tracker_samples = [sample for sample in finite_samples if sample.tracker_id not in (None, -1)]
    if not stable_samples and not tracker_samples:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="identity_continuity",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.TEMPORAL,
            metric={"sample_count": len(finite_samples)},
            detail="Track samples did not include stable_id or tracker_id evidence.",
            suggested_next_diagnostic="Check StableIDManager output and tracker ID propagation in telemetry.",
        )

    tracker_stable_switches = 0
    tracker_groups: dict[tuple[str | None, int], list[TrackSample]] = defaultdict(list)
    for sample in tracker_samples:
        tracker_groups[(sample.camera_id, int(sample.tracker_id))].append(sample)  # type: ignore[arg-type]
    for rows in tracker_groups.values():
        rows.sort(key=lambda row: float(row.ts_s))
        for left, right in zip(rows, rows[1:]):
            if left.stable_id in (None, -1) or right.stable_id in (None, -1):
                continue
            if int(left.stable_id) == int(right.stable_id):
                continue
            dt = float(right.ts_s) - float(left.ts_s)
            if 0.0 <= dt <= max_switch_gap_s:
                tracker_stable_switches += 1

    duplicate_stable_far_events = 0
    same_camera_duplicate_events = 0
    simultaneous_stable_pairs = 0
    stable_groups: dict[int, list[TrackSample]] = defaultdict(list)
    for sample in stable_samples:
        stable_groups[int(sample.stable_id)].append(sample)  # type: ignore[arg-type]
    for rows in stable_groups.values():
        rows.sort(key=lambda row: float(row.ts_s))
        for idx, left in enumerate(rows):
            left_world = _world_array(left.world)
            if left_world is None:
                continue
            for right in rows[idx + 1 :]:
                dt = abs(float(right.ts_s) - float(left.ts_s))
                if dt > max_simultaneous_dt_s:
                    break
                if left.tracker_id == right.tracker_id and left.camera_id == right.camera_id:
                    continue
                right_world = _world_array(right.world)
                if right_world is None:
                    continue
                simultaneous_stable_pairs += 1
                distance = float(np.linalg.norm(right_world - left_world))
                if distance > duplicate_stable_fail_distance_m:
                    duplicate_stable_far_events += 1
                elif left.camera_id == right.camera_id:
                    same_camera_duplicate_events += 1

    tracker_multi_stable_events = 0
    for rows in tracker_groups.values():
        rows.sort(key=lambda row: float(row.ts_s))
        for idx, left in enumerate(rows):
            if left.stable_id in (None, -1):
                continue
            for right in rows[idx + 1 :]:
                dt = abs(float(right.ts_s) - float(left.ts_s))
                if dt > max_simultaneous_dt_s:
                    break
                if right.stable_id in (None, -1):
                    continue
                if int(left.stable_id) != int(right.stable_id):
                    tracker_multi_stable_events += 1

    issue_count = (
        tracker_stable_switches
        + duplicate_stable_far_events
        + same_camera_duplicate_events
        + tracker_multi_stable_events
    )
    if issue_count:
        status = CheckStatus.FAIL
        failure_type = FailureType.TEMPORAL
        detail = "Identity continuity has StableID switches, duplicate assignments, or split/merge evidence."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "StableID and tracker identity evidence remains continuous across the sampled tracks."
    return ValidationCheck(
        id=check_id,
        domain="tracking",
        name="identity_continuity",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(finite_samples),
            "stable_sample_count": len(stable_samples),
            "tracker_sample_count": len(tracker_samples),
            "tracker_stable_switch_count": tracker_stable_switches,
            "simultaneous_stable_pair_count": simultaneous_stable_pairs,
            "duplicate_stable_far_event_count": duplicate_stable_far_events,
            "same_camera_duplicate_stable_event_count": same_camera_duplicate_events,
            "tracker_multi_stable_event_count": tracker_multi_stable_events,
        },
        threshold={
            "max_switch_gap_s": max_switch_gap_s,
            "max_simultaneous_dt_s": max_simultaneous_dt_s,
            "duplicate_stable_fail_distance_m": duplicate_stable_fail_distance_m,
        },
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect StableID assignment, tracker reuse, cross-camera fusion, and split/merge handling near the reported timestamps.",
    )


def validate_reid_geometry_consistency(
    samples: Sequence[TrackSample],
    *,
    check_id: str = "TRACK.reid_geometry_consistency",
    max_geometry_gap_m: float = 0.75,
    max_geometry_gap_s: float = 2.0,
    max_simultaneous_dt_s: float = 0.10,
    duplicate_appearance_fail_distance_m: float = 1.50,
    min_reid_confidence: float = 0.50,
) -> ValidationCheck:
    finite_samples = [sample for sample in samples if _world_array(sample.world) is not None]
    if not finite_samples:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="reid_geometry_consistency",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No finite track samples were provided for ReID/geometry validation.",
            suggested_next_diagnostic="Capture tracking samples with backend_world_m positions and ReID appearance evidence.",
        )
    appearance_samples = [sample for sample in finite_samples if sample.appearance_key is not None]
    if not appearance_samples:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="reid_geometry_consistency",
            status=CheckStatus.SKIPPED,
            metric={"sample_count": len(finite_samples), "appearance_sample_count": 0},
            detail="No ReID appearance identity evidence was provided for this run.",
            suggested_next_diagnostic="Expose ReID identity or appearance cluster labels in telemetry when validating identity continuity.",
        )

    stable_reid_conflicts = 0
    low_reid_confidence_count = 0
    stable_groups: dict[int, list[TrackSample]] = defaultdict(list)
    for sample in appearance_samples:
        if sample.reid_confidence is not None and float(sample.reid_confidence) < min_reid_confidence:
            low_reid_confidence_count += 1
        if sample.stable_id not in (None, -1):
            stable_groups[int(sample.stable_id)].append(sample)  # type: ignore[arg-type]
    for rows in stable_groups.values():
        rows.sort(key=lambda row: float(row.ts_s))
        for left, right in zip(rows, rows[1:]):
            if left.appearance_key == right.appearance_key:
                continue
            dt = float(right.ts_s) - float(left.ts_s)
            if dt < 0.0 or dt > max_geometry_gap_s:
                continue
            left_world = _world_array(left.world)
            right_world = _world_array(right.world)
            if left_world is None or right_world is None:
                continue
            if float(np.linalg.norm(right_world - left_world)) <= max_geometry_gap_m:
                stable_reid_conflicts += 1

    reid_stable_churn = 0
    simultaneous_appearance_far_events = 0
    appearance_groups: dict[str, list[TrackSample]] = defaultdict(list)
    for sample in appearance_samples:
        appearance_groups[str(sample.appearance_key)].append(sample)
    for rows in appearance_groups.values():
        rows.sort(key=lambda row: float(row.ts_s))
        for left, right in zip(rows, rows[1:]):
            if left.stable_id in (None, -1) or right.stable_id in (None, -1):
                continue
            if int(left.stable_id) == int(right.stable_id):
                continue
            dt = float(right.ts_s) - float(left.ts_s)
            if dt < 0.0 or dt > max_geometry_gap_s:
                continue
            left_world = _world_array(left.world)
            right_world = _world_array(right.world)
            if left_world is None or right_world is None:
                continue
            if float(np.linalg.norm(right_world - left_world)) <= max_geometry_gap_m:
                reid_stable_churn += 1
        for idx, left in enumerate(rows):
            left_world = _world_array(left.world)
            if left_world is None:
                continue
            for right in rows[idx + 1 :]:
                dt = abs(float(right.ts_s) - float(left.ts_s))
                if dt > max_simultaneous_dt_s:
                    break
                right_world = _world_array(right.world)
                if right_world is None:
                    continue
                if float(np.linalg.norm(right_world - left_world)) > duplicate_appearance_fail_distance_m:
                    simultaneous_appearance_far_events += 1

    if stable_reid_conflicts:
        status = CheckStatus.FAIL
        failure_type = FailureType.TEMPORAL
        detail = "Geometry and StableID continuity say same person, but ReID identity changed over a continuous path."
    elif reid_stable_churn or simultaneous_appearance_far_events or low_reid_confidence_count:
        status = CheckStatus.WARNING
        failure_type = FailureType.TEMPORAL
        detail = "ReID evidence is usable but has identity churn, simultaneous appearance aliasing, or low-confidence samples."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "ReID appearance evidence agrees with geometry and StableID continuity."
    return ValidationCheck(
        id=check_id,
        domain="tracking",
        name="reid_geometry_consistency",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(finite_samples),
            "appearance_sample_count": len(appearance_samples),
            "stable_reid_conflict_count": stable_reid_conflicts,
            "reid_stable_churn_count": reid_stable_churn,
            "simultaneous_appearance_far_event_count": simultaneous_appearance_far_events,
            "low_reid_confidence_count": low_reid_confidence_count,
        },
        threshold={
            "max_geometry_gap_m": max_geometry_gap_m,
            "max_geometry_gap_s": max_geometry_gap_s,
            "max_simultaneous_dt_s": max_simultaneous_dt_s,
            "duplicate_appearance_fail_distance_m": duplicate_appearance_fail_distance_m,
            "min_reid_confidence": min_reid_confidence,
        },
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Compare ReID embeddings, StableID transitions, tracker continuity, and simultaneous camera observations for the flagged identities.",
    )


def validate_detection_world_projection(
    samples: Sequence[DetectionProjectionSample],
    *,
    check_id: str = "TRACK.detection_world_projection",
    good_foot_agreement_m: float = 0.15,
    fail_foot_agreement_m: float = 0.50,
    good_floor_error_m: float = 0.05,
    fail_floor_error_m: float = 0.20,
    min_person_height_m: float = 1.2,
    max_person_height_m: float = 2.3,
    good_min_bbox_aspect: float = 0.20,
    good_max_bbox_aspect: float = 0.85,
    fail_min_bbox_aspect: float = 0.10,
    fail_max_bbox_aspect: float = 1.20,
    good_bbox_center_error_px: float = 16.0,
    fail_bbox_center_error_px: float = 50.0,
    good_projected_bbox_iou: float = 0.50,
    fail_projected_bbox_iou: float = 0.25,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="detection_world_projection",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No detection-to-world projection samples were provided.",
            suggested_next_diagnostic="Capture bbox/mask/pose/depth footpoint evidence and final backend_world_m for the same detections.",
        )
    foot_errors: list[float] = []
    floor_errors: list[float] = []
    invalid_world = 0
    invalid_rays = 0
    room_misses = 0
    implausible_heights = 0
    bbox_aspect_failures = 0
    bbox_aspect_warnings = 0
    projected_bbox_failures = 0
    projected_bbox_warnings = 0
    invalid_bbox_values = 0
    bbox_center_errors_px: list[float] = []
    projected_bbox_ious: list[float] = []
    for sample in samples:
        world = _world_array(sample.world_point)
        if world is None:
            invalid_world += 1
            continue
        floor_errors.append(abs(float(world[1]) - float(sample.floor_y)))
        for candidate in (sample.bbox_foot_world, sample.mask_foot_world, sample.pose_foot_world, sample.depth_world):
            point = _optional_world_array(candidate)
            if point is not None:
                foot_errors.append(float(np.linalg.norm(point - world)))
        if not sample.ray_floor_valid:
            invalid_rays += 1
        if sample.room_polygon_xz and not point_in_polygon_xz(world, sample.room_polygon_xz):
            room_misses += 1
        if sample.person_height_m is not None and not (min_person_height_m <= float(sample.person_height_m) <= max_person_height_m):
            implausible_heights += 1
        image_bbox = _bbox_xyxy(sample.image_bbox_xyxy)
        projected_bbox = _bbox_xyxy(sample.projected_bbox_xyxy)
        if sample.image_bbox_xyxy is not None and image_bbox is None:
            invalid_bbox_values += 1
        if sample.projected_bbox_xyxy is not None and projected_bbox is None:
            invalid_bbox_values += 1
        if image_bbox is not None:
            aspect = float((image_bbox[2] - image_bbox[0]) / (image_bbox[3] - image_bbox[1]))
            if aspect < fail_min_bbox_aspect or aspect > fail_max_bbox_aspect:
                bbox_aspect_failures += 1
            elif aspect < good_min_bbox_aspect or aspect > good_max_bbox_aspect:
                bbox_aspect_warnings += 1
        if image_bbox is not None and projected_bbox is not None:
            projected_bbox_ious.append(_bbox_iou(image_bbox, projected_bbox))
            bbox_center_errors_px.append(_bbox_center_error_px(image_bbox, projected_bbox))
        if sample.projected_bbox_iou is not None:
            try:
                value = float(sample.projected_bbox_iou)
                if np.isfinite(value):
                    projected_bbox_ious.append(value)
                else:
                    invalid_bbox_values += 1
            except Exception:
                invalid_bbox_values += 1
        if sample.bbox_center_error_px is not None:
            try:
                value = abs(float(sample.bbox_center_error_px))
                if np.isfinite(value):
                    bbox_center_errors_px.append(value)
                else:
                    invalid_bbox_values += 1
            except Exception:
                invalid_bbox_values += 1
    for iou in projected_bbox_ious:
        if iou < fail_projected_bbox_iou:
            projected_bbox_failures += 1
        elif iou < good_projected_bbox_iou:
            projected_bbox_warnings += 1
    p95_foot = float(np.percentile(np.asarray(foot_errors, dtype=np.float64), 95)) if foot_errors else 0.0
    p95_floor = float(np.percentile(np.asarray(floor_errors, dtype=np.float64), 95)) if floor_errors else 0.0
    p95_bbox_center = float(np.percentile(np.asarray(bbox_center_errors_px, dtype=np.float64), 95)) if bbox_center_errors_px else 0.0
    min_projected_bbox_iou = float(min(projected_bbox_ious)) if projected_bbox_ious else 1.0
    if (
        invalid_world
        or invalid_rays
        or room_misses
        or invalid_bbox_values
        or bbox_aspect_failures
        or projected_bbox_failures
        or p95_foot > fail_foot_agreement_m
        or p95_floor > fail_floor_error_m
        or p95_bbox_center > fail_bbox_center_error_px
    ):
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "Detection-to-world projection violates footpoint, ray-floor, floor-contact, room-bound, bbox aspect, or bbox reprojection thresholds."
    elif (
        implausible_heights
        or bbox_aspect_warnings
        or projected_bbox_warnings
        or p95_foot > good_foot_agreement_m
        or p95_floor > good_floor_error_m
        or p95_bbox_center > good_bbox_center_error_px
    ):
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Detection-to-world projection is usable but outside the good band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Detection footpoints, floor contact, room bounds, bbox reprojection, aspect, and height assumptions agree."
    return ValidationCheck(
        id=check_id,
        domain="tracking",
        name="detection_world_projection",
        status=status,
        failure_type=failure_type,
        metric={
            "p95_foot_agreement_m": p95_foot,
            "p95_floor_error_m": p95_floor,
            "invalid_world_count": invalid_world,
            "invalid_ray_count": invalid_rays,
            "room_miss_count": room_misses,
            "implausible_height_count": implausible_heights,
            "bbox_aspect_failure_count": bbox_aspect_failures,
            "bbox_aspect_warning_count": bbox_aspect_warnings,
            "projected_bbox_failure_count": projected_bbox_failures,
            "projected_bbox_warning_count": projected_bbox_warnings,
            "invalid_bbox_value_count": invalid_bbox_values,
            "p95_bbox_center_error_px": p95_bbox_center,
            "min_projected_bbox_iou": min_projected_bbox_iou,
            "sample_count": len(samples),
        },
        threshold={
            "good_foot_agreement_m": good_foot_agreement_m,
            "fail_foot_agreement_m": fail_foot_agreement_m,
            "good_floor_error_m": good_floor_error_m,
            "fail_floor_error_m": fail_floor_error_m,
            "min_person_height_m": min_person_height_m,
            "max_person_height_m": max_person_height_m,
            "good_min_bbox_aspect": good_min_bbox_aspect,
            "good_max_bbox_aspect": good_max_bbox_aspect,
            "fail_min_bbox_aspect": fail_min_bbox_aspect,
            "fail_max_bbox_aspect": fail_max_bbox_aspect,
            "good_bbox_center_error_px": good_bbox_center_error_px,
            "fail_bbox_center_error_px": fail_bbox_center_error_px,
            "good_projected_bbox_iou": good_projected_bbox_iou,
            "fail_projected_bbox_iou": fail_projected_bbox_iou,
        },
        evidence=sorted({e for sample in samples for e in sample.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect bbox/mask/pose footpoint choice, floor-plane intersection, camera reprojection, bbox format, and room polygon frame.",
    )


def validate_projection_confidence(
    samples: Sequence[DetectionProjectionSample],
    *,
    check_id: str = "TRACK.projection_confidence",
    fail_foot_agreement_m: float = 0.50,
    fail_floor_error_m: float = 0.20,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="tracking",
            name="projection_confidence",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No projection confidence samples were provided.",
            suggested_next_diagnostic="Capture detection-to-world projection samples with component evidence.",
        )
    scores: list[float] = []
    component_counts = {
        "footpoint": 0,
        "floor_contact": 0,
        "ray_floor": 0,
        "room_bounds": 0,
        "height": 0,
        "bbox_aspect": 0,
        "bbox_overlap": 0,
        "bbox_center": 0,
        "reprojection": 0,
        "temporal_smoothness": 0,
        "semantic_validity": 0,
    }
    for sample in samples:
        world = _world_array(sample.world_point)
        if world is None:
            scores.append(0.0)
            continue
        component_scores: list[float] = []
        candidate_errors = []
        for candidate in (sample.bbox_foot_world, sample.mask_foot_world, sample.pose_foot_world, sample.depth_world):
            point = _optional_world_array(candidate)
            if point is not None:
                candidate_errors.append(float(np.linalg.norm(point - world)))
        if candidate_errors:
            component_scores.append(max(0.0, 1.0 - min(candidate_errors) / fail_foot_agreement_m))
            component_counts["footpoint"] += 1
        component_scores.append(max(0.0, 1.0 - abs(float(world[1]) - float(sample.floor_y)) / fail_floor_error_m))
        component_counts["floor_contact"] += 1
        component_scores.append(1.0 if sample.ray_floor_valid else 0.0)
        component_counts["ray_floor"] += 1
        if sample.room_polygon_xz:
            component_scores.append(1.0 if point_in_polygon_xz(world, sample.room_polygon_xz) else 0.0)
            component_counts["room_bounds"] += 1
        if sample.person_height_m is not None:
            component_scores.append(1.0 if 1.2 <= float(sample.person_height_m) <= 2.3 else 0.0)
            component_counts["height"] += 1
        image_bbox = _bbox_xyxy(sample.image_bbox_xyxy)
        projected_bbox = _bbox_xyxy(sample.projected_bbox_xyxy)
        if image_bbox is not None:
            aspect = float((image_bbox[2] - image_bbox[0]) / (image_bbox[3] - image_bbox[1]))
            component_scores.append(1.0 if 0.10 <= aspect <= 1.20 else 0.0)
            component_counts["bbox_aspect"] += 1
        if image_bbox is not None and projected_bbox is not None:
            component_scores.append(_bbox_iou(image_bbox, projected_bbox))
            component_counts["bbox_overlap"] += 1
            component_scores.append(max(0.0, 1.0 - _bbox_center_error_px(image_bbox, projected_bbox) / 50.0))
            component_counts["bbox_center"] += 1
        if sample.projected_bbox_iou is not None:
            try:
                component_scores.append(float(np.clip(float(sample.projected_bbox_iou), 0.0, 1.0)))
                component_counts["bbox_overlap"] += 1
            except Exception:
                component_scores.append(0.0)
                component_counts["bbox_overlap"] += 1
        if sample.bbox_center_error_px is not None:
            try:
                component_scores.append(max(0.0, 1.0 - abs(float(sample.bbox_center_error_px)) / 50.0))
                component_counts["bbox_center"] += 1
            except Exception:
                component_scores.append(0.0)
                component_counts["bbox_center"] += 1
        for component_name, component in (
            ("reprojection", sample.reprojection_score),
            ("temporal_smoothness", sample.temporal_smoothness_score),
            ("semantic_validity", sample.semantic_validity_score),
        ):
            score = _score_or_zero(component)
            if score is not None:
                component_scores.append(score)
                component_counts[component_name] += 1
        scores.append(float(np.mean(component_scores)) if component_scores else 0.0)
    p50 = float(np.percentile(np.asarray(scores, dtype=np.float64), 50))
    p05 = float(np.percentile(np.asarray(scores, dtype=np.float64), 5))
    if p50 >= 0.65 and p05 >= 0.45:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Projection confidence is usable across the sampled detections."
    elif p50 >= 0.45:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Projection confidence is weak or partially contradicted."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "Projection confidence is too low to trust world positions."
    return ValidationCheck(
        id=check_id,
        domain="tracking",
        name="projection_confidence",
        status=status,
        failure_type=failure_type,
        metric={"score": p50, "p05_score": p05, "sample_count": len(samples), "component_counts": component_counts},
        threshold={"strong": 0.85, "usable": 0.65, "weak": 0.45},
        evidence=sorted({e for sample in samples for e in sample.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect which projection confidence component collapsed: footpoint, floor, ray, room bounds, or person height.",
    )


def validate_bev_track_agreement(
    track_samples: Sequence[TrackSample],
    bev_points: Sequence[BevPointSample],
    *,
    check_id: str = "BEV.track_agreement",
    max_dt_s: float = 0.25,
    good_p95_m: float = 0.10,
    fail_p95_m: float = 0.50,
) -> ValidationCheck:
    bev_by_key: dict[tuple[str | None, Hashable], list[BevPointSample]] = defaultdict(list)
    for point in bev_points:
        bev_by_key[(point.camera_id, point.identity)].append(point)
    for rows in bev_by_key.values():
        rows.sort(key=lambda row: row.ts_s)

    errors: list[float] = []
    for track in track_samples:
        world = _world_array(track.world)
        if world is None:
            continue
        candidates = bev_by_key.get((track.camera_id, track.identity), [])
        if not candidates:
            continue
        best: BevPointSample | None = None
        best_dt = math.inf
        for candidate in candidates:
            dt = abs(float(candidate.ts_s) - float(track.ts_s))
            if dt < best_dt:
                best = candidate
                best_dt = dt
        if best is None or best_dt > max_dt_s:
            continue
        errors.append(float(math.hypot(float(best.x) - float(world[0]), float(best.z) - float(world[2]))))

    if not errors:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="bev_track_agreement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No comparable BEV/track samples were provided.",
            suggested_next_diagnostic="Capture BEV and tracking samples with shared camera and stable/tracker IDs.",
        )
    p95 = float(np.percentile(np.asarray(errors, dtype=np.float64), 95))
    if p95 <= good_p95_m:
        status = CheckStatus.PASS
        failure_type = None
        detail = "BEV points agree with canonical track world positions."
    elif p95 <= fail_p95_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "BEV/track agreement is outside the good band but below fail threshold."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "BEV points do not agree with canonical track world positions."
    return ValidationCheck(
        id=check_id,
        domain="bev",
        name="bev_track_agreement",
        status=status,
        failure_type=failure_type,
        metric={"p95_error_m": p95, "mean_error_m": float(np.mean(errors)), "comparison_count": len(errors)},
        threshold={"good_p95_m": good_p95_m, "fail_p95_m": fail_p95_m, "max_dt_s": max_dt_s},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect BEV frame declaration, backend world ownership, and duplicate coordinate conversion.",
    )
