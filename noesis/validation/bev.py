from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable
from typing import Mapping, Sequence

import numpy as np

from .core import CheckStatus, FailureType, ValidationCheck
from .geometry import point_in_polygon_xz
from .tracking import TrackSample


def _finite_xz_from_world(point: Sequence[float]) -> list[float] | None:
    try:
        x = float(point[0])
        z = float(point[2])
    except Exception:
        return None
    if not np.isfinite([x, z]).all():
        return None
    return [x, z]


def _angle_delta_deg(a_rad: float, b_rad: float) -> float:
    return abs(math.degrees((a_rad - b_rad + math.pi) % (2.0 * math.pi) - math.pi))


def validate_path_smoothness(
    track_samples: Sequence[TrackSample],
    *,
    check_id: str = "BEV.path_smoothness",
    min_step_m: float = 0.05,
    good_p95_turn_deg: float = 90.0,
    fail_p95_turn_deg: float = 150.0,
    good_path_ratio: float = 2.0,
    fail_path_ratio: float = 4.0,
) -> ValidationCheck:
    grouped: dict[tuple[str | None, Hashable], list[TrackSample]] = defaultdict(list)
    for sample in track_samples:
        if _finite_xz_from_world(sample.world) is not None:
            grouped[(sample.camera_id, sample.identity)].append(sample)
    if not grouped:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="path_smoothness",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No finite track samples were provided for BEV path smoothness validation.",
            suggested_next_diagnostic="Capture ordered track or BEV trail samples with backend_world_m positions.",
        )

    turn_angles: list[float] = []
    path_ratios: list[float] = []
    compared_tracks = 0
    short_step_count = 0
    for rows in grouped.values():
        rows.sort(key=lambda sample: float(sample.ts_s))
        points: list[np.ndarray] = []
        for sample in rows:
            point = _finite_xz_from_world(sample.world)
            if point is not None:
                points.append(np.asarray(point, dtype=np.float64))
        if len(points) < 3:
            continue
        compared_tracks += 1
        segment_lengths = [float(np.linalg.norm(right - left)) for left, right in zip(points, points[1:])]
        path_length = float(sum(segment_lengths))
        displacement = float(np.linalg.norm(points[-1] - points[0]))
        if displacement >= min_step_m:
            path_ratios.append(path_length / displacement)
        for left, middle, right in zip(points, points[1:], points[2:]):
            a = middle - left
            b = right - middle
            len_a = float(np.linalg.norm(a))
            len_b = float(np.linalg.norm(b))
            if len_a < min_step_m or len_b < min_step_m:
                short_step_count += 1
                continue
            turn_angles.append(_angle_delta_deg(float(math.atan2(a[1], a[0])), float(math.atan2(b[1], b[0]))))

    if not turn_angles and not path_ratios:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="path_smoothness",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            metric={"track_count": len(grouped), "compared_track_count": compared_tracks, "short_step_count": short_step_count},
            detail="Track samples did not include enough nontrivial path segments for smoothness validation.",
            suggested_next_diagnostic="Capture longer track trails or lower the minimum step threshold for fixture-only checks.",
        )

    p95_turn = float(np.percentile(np.asarray(turn_angles, dtype=np.float64), 95)) if turn_angles else 0.0
    max_path_ratio = float(max(path_ratios)) if path_ratios else 1.0
    if p95_turn > fail_p95_turn_deg or max_path_ratio > fail_path_ratio:
        status = CheckStatus.FAIL
        failure_type = FailureType.TEMPORAL
        detail = "BEV paths show abrupt zig-zagging or severe path inflation."
    elif p95_turn > good_p95_turn_deg or max_path_ratio > good_path_ratio:
        status = CheckStatus.WARNING
        failure_type = FailureType.TEMPORAL
        detail = "BEV paths are usable but outside the smoothness good band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "BEV paths are smooth and do not show suspicious zig-zagging."
    return ValidationCheck(
        id=check_id,
        domain="bev",
        name="path_smoothness",
        status=status,
        failure_type=failure_type,
        metric={
            "p95_turn_deg": p95_turn,
            "max_path_length_to_displacement_ratio": max_path_ratio,
            "turn_sample_count": len(turn_angles),
            "path_ratio_count": len(path_ratios),
            "short_step_count": short_step_count,
            "track_count": len(grouped),
            "compared_track_count": compared_tracks,
        },
        threshold={
            "min_step_m": min_step_m,
            "good_p95_turn_deg": good_p95_turn_deg,
            "fail_p95_turn_deg": fail_p95_turn_deg,
            "good_path_ratio": good_path_ratio,
            "fail_path_ratio": fail_path_ratio,
        },
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect BEV smoothing ownership, duplicate transform application, identity switches, and projection jitter near occlusions.",
    )


def validate_zone_consistency(
    track_samples: Sequence[TrackSample],
    room_polygons_xz: Mapping[str, Sequence[Sequence[float]]],
    *,
    check_id: str = "BEV.zone_consistency",
    min_consistency_ratio: float = 0.98,
) -> ValidationCheck:
    if not track_samples:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="zone_consistency",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No track samples were provided for zone consistency validation.",
            suggested_next_diagnostic="Capture track samples with room/zone labels and backend_world_m positions.",
        )
    checked = 0
    consistent = 0
    missing_zone = 0
    missing_polygon = 0
    for sample in track_samples:
        point = _finite_xz_from_world(sample.world)
        if point is None:
            continue
        if not sample.room:
            missing_zone += 1
            continue
        polygon = room_polygons_xz.get(str(sample.room))
        if not polygon:
            missing_polygon += 1
            continue
        checked += 1
        if point_in_polygon_xz(point, polygon):
            consistent += 1
    if checked <= 0:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="zone_consistency",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            metric={"missing_zone_count": missing_zone, "missing_polygon_count": missing_polygon},
            detail="No track samples had both a finite position and matching room polygon.",
            suggested_next_diagnostic="Check zone labels, room polygon names, and telemetry fixture coverage.",
        )
    ratio = float(consistent / checked)
    if ratio >= min_consistency_ratio and missing_zone == 0 and missing_polygon == 0:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Track room/zone labels agree with BEV room polygons."
    elif ratio > 0.0:
        status = CheckStatus.WARNING
        failure_type = FailureType.SEMANTIC
        detail = "Some track room/zone labels disagree with BEV room polygons or lack polygon evidence."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.SEMANTIC
        detail = "Track room/zone labels do not agree with BEV room polygons."
    return ValidationCheck(
        id=check_id,
        domain="bev",
        name="zone_consistency",
        status=status,
        failure_type=failure_type,
        metric={
            "consistency_ratio": ratio,
            "consistent_count": consistent,
            "checked_count": checked,
            "missing_zone_count": missing_zone,
            "missing_polygon_count": missing_polygon,
        },
        threshold={"min_consistency_ratio": min_consistency_ratio},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check zone labels, room polygon frame, and camera-specific track conversion.",
    )


def validate_camera_coverage(
    track_samples: Sequence[TrackSample],
    camera_frustums_xz: Mapping[str, Sequence[Sequence[float]]],
    *,
    check_id: str = "BEV.camera_coverage",
    min_covered_ratio: float = 0.95,
) -> ValidationCheck:
    if not track_samples:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="camera_coverage",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No track samples were provided for camera coverage validation.",
            suggested_next_diagnostic="Capture track samples with camera IDs and backend_world_m positions.",
        )
    checked = 0
    covered = 0
    missing_frustum = 0
    occluded_or_predicted = 0
    for sample in track_samples:
        point = _finite_xz_from_world(sample.world)
        if point is None:
            continue
        if sample.occluded:
            occluded_or_predicted += 1
            continue
        if not sample.camera_id or sample.camera_id not in camera_frustums_xz:
            missing_frustum += 1
            continue
        checked += 1
        if point_in_polygon_xz(point, camera_frustums_xz[str(sample.camera_id)]):
            covered += 1
    if checked <= 0:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="camera_coverage",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            metric={"missing_frustum_count": missing_frustum, "occluded_or_predicted_count": occluded_or_predicted},
            detail="No non-occluded track samples had matching camera frustum evidence.",
            suggested_next_diagnostic="Add camera frustum polygons or capture non-occluded track samples.",
        )
    ratio = float(covered / checked)
    if ratio >= min_covered_ratio and missing_frustum == 0:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Non-occluded track samples fall within their camera coverage footprints."
    elif ratio > 0.0:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Some non-occluded track samples fall outside camera coverage footprints."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "Non-occluded track samples are outside camera coverage footprints."
    return ValidationCheck(
        id=check_id,
        domain="bev",
        name="camera_coverage",
        status=status,
        failure_type=failure_type,
        metric={
            "covered_ratio": ratio,
            "covered_count": covered,
            "checked_count": checked,
            "missing_frustum_count": missing_frustum,
            "occluded_or_predicted_count": occluded_or_predicted,
        },
        threshold={"min_covered_ratio": min_covered_ratio},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check camera frustum projection, observability assumptions, and stale/predicted track handling.",
    )
