from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .core import CheckStatus, FailureType, ValidationCheck
from .transforms import matrix_from_col_major, round_trip_points, transform_points


@dataclass(frozen=True)
class MenonPlacementSample:
    entity_id: str
    world_point: Sequence[float]
    menon_point: Sequence[float]
    camera_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MenonTrailSample:
    entity_id: str
    noesis_world_points: Sequence[Sequence[float]]
    menon_points: Sequence[Sequence[float]]
    camera_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MenonBevTrailSample:
    entity_id: str
    bev_points_xz: Sequence[Sequence[float]]
    menon_points: Sequence[Sequence[float]]
    floor_y_world: float = 0.0
    camera_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class WorldBevSample:
    entity_id: str
    world_point: Sequence[float]
    bev_point_xz: Sequence[float]
    bev_y: float = 0.0
    camera_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MenonAvatarSample:
    entity_id: str
    menon_point: Sequence[float]
    radius_scene: float | None = None
    expected_height_m: float | None = None
    observed_height_scene: float | None = None
    scene_units_per_m: float = 1.0
    heading_deg: float | None = None
    movement_vector_xz: Sequence[float] | None = None
    allowed_collision: bool = False
    camera_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MenonObjectSample:
    object_id: str
    category: str
    menon_point: Sequence[float]
    world_point: Sequence[float] | None = None
    bbox_min_scene: Sequence[float] | None = None
    bbox_max_scene: Sequence[float] | None = None
    expected_dimensions_m: Sequence[float] | None = None
    observed_dimensions_scene: Sequence[float] | None = None
    scene_units_per_m: float = 1.0
    support_y_scene: float = 0.0
    allowed_rooms: Sequence[str] = ()
    allowed_collision: bool = False
    room: str | None = None
    camera_id: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MenonTransformAuditSample:
    entity_id: str
    stages: Sequence[Mapping[str, Any]]
    camera_id: str | None = None
    room: str | None = None


@dataclass(frozen=True)
class MenonCameraReprojectionSample:
    camera_id: str
    source_frame: str | None = None
    menon_render: str | None = None
    overlay_path: str | None = None
    layers: Sequence[str] = ()
    mean_error_px: float | None = None
    max_error_px: float | None = None
    anchor_mean_error_px: float | None = None
    anchor_max_error_px: float | None = None
    floor_grid_mean_error_px: float | None = None
    room_edge_mean_error_px: float | None = None
    bbox_iou: float | None = None
    avatar_iou: float | None = None
    mask_iou: float | None = None
    entity_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MenonLatencySample:
    entity_id: str
    noesis_ts_s: float
    telemetry_ts_s: float | None = None
    menon_update_ts_s: float | None = None
    menon_render_ts_s: float | None = None
    menon_display_ts_s: float | None = None
    camera_id: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


def _point_to_segment_distance_xz(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def _box_min_max_xz(raw_box: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(raw_box, Mapping):
        raw_min = raw_box.get("bbox_min_xz") or raw_box.get("min_xz") or raw_box.get("bbox_min") or raw_box.get("min")
        raw_max = raw_box.get("bbox_max_xz") or raw_box.get("max_xz") or raw_box.get("bbox_max") or raw_box.get("max")
    elif isinstance(raw_box, Sequence) and not isinstance(raw_box, (str, bytes, bytearray)) and len(raw_box) == 2:
        raw_min, raw_max = raw_box[0], raw_box[1]
    else:
        return None
    try:
        a = np.asarray(raw_min, dtype=np.float64).reshape((-1,))
        b = np.asarray(raw_max, dtype=np.float64).reshape((-1,))
        if a.size >= 3 and b.size >= 3:
            min_xz = np.asarray([min(a[0], b[0]), min(a[2], b[2])], dtype=np.float64)
            max_xz = np.asarray([max(a[0], b[0]), max(a[2], b[2])], dtype=np.float64)
        elif a.size >= 2 and b.size >= 2:
            min_xz = np.asarray([min(a[0], b[0]), min(a[1], b[1])], dtype=np.float64)
            max_xz = np.asarray([max(a[0], b[0]), max(a[1], b[1])], dtype=np.float64)
        else:
            return None
        if np.all(np.isfinite(min_xz)) and np.all(np.isfinite(max_xz)):
            return min_xz, max_xz
    except Exception:
        return None
    return None


def _point_in_box_xz(point: np.ndarray, min_xz: np.ndarray, max_xz: np.ndarray) -> bool:
    return bool(min_xz[0] <= point[0] <= max_xz[0] and min_xz[1] <= point[1] <= max_xz[1])


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect_xz(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-9
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    for p, q, r, orient in ((a, b, c, o1), (a, b, d, o2), (c, d, a, o3), (c, d, b, o4)):
        if abs(orient) <= eps and min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps:
            return True
    return False


def _segment_intersects_box_xz(a: np.ndarray, b: np.ndarray, min_xz: np.ndarray, max_xz: np.ndarray) -> bool:
    if _point_in_box_xz(a, min_xz, max_xz) or _point_in_box_xz(b, min_xz, max_xz):
        return True
    corners = [
        np.asarray([min_xz[0], min_xz[1]], dtype=np.float64),
        np.asarray([max_xz[0], min_xz[1]], dtype=np.float64),
        np.asarray([max_xz[0], max_xz[1]], dtype=np.float64),
        np.asarray([min_xz[0], max_xz[1]], dtype=np.float64),
    ]
    return any(_segments_intersect_xz(a, b, left, right) for left, right in zip(corners, corners[1:] + corners[:1]))


def _boxes_overlap_xz(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> bool:
    left_min, left_max = left
    right_min, right_max = right
    return bool(left_min[0] <= right_max[0] and left_max[0] >= right_min[0] and left_min[1] <= right_max[1] and left_max[1] >= right_min[1])


def _angle_delta_deg(a_deg: float, b_deg: float) -> float:
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def validate_world_menon_round_trip(
    world_to_menon_col_major: Sequence[float],
    samples: Sequence[MenonPlacementSample],
    *,
    check_id: str = "MENON.world_round_trip",
    good_max_error_m: float = 1e-6,
    fail_max_error_m: float = 1e-3,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="world_menon_round_trip",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon placement samples were provided.",
            suggested_next_diagnostic="Capture Noesis world and Menon scene positions for the same entities.",
        )
    try:
        matrix = matrix_from_col_major(world_to_menon_col_major, name="world_to_menon")
        world_points = np.asarray([sample.world_point for sample in samples], dtype=np.float64).reshape((-1, 3))
        round_trip = round_trip_points(world_points, matrix)
    except Exception as exc:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="world_menon_round_trip",
            status=CheckStatus.FAIL,
            failure_type=FailureType.TRANSFORM,
            detail=f"World/Menon round-trip failed: {exc}",
            suggested_next_diagnostic="Inspect scene_similarity.world_to_scene_col_major and matrix convention.",
        )
    if round_trip.max_error <= good_max_error_m:
        status = CheckStatus.PASS
        failure_type = None
        detail = "World/Menon transform round-trip is within tolerance."
    elif round_trip.max_error <= fail_max_error_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.TRANSFORM
        detail = "World/Menon transform round-trip is above the good band."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.TRANSFORM
        detail = "World/Menon transform round-trip exceeds fail threshold."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="world_menon_round_trip",
        status=status,
        failure_type=failure_type,
        metric={"max_error_m": round_trip.max_error, "mean_error_m": round_trip.mean_error, "sample_count": len(samples)},
        threshold={"good_max_error_m": good_max_error_m, "fail_max_error_m": fail_max_error_m},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check scale, origin, and one-time backend-world to Menon-scene conversion.",
    )


def validate_world_menon_placement_agreement(
    world_to_menon_col_major: Sequence[float],
    samples: Sequence[MenonPlacementSample],
    *,
    check_id: str = "MENON.placement_agreement",
    good_p95_scene_units: float = 0.10,
    fail_p95_scene_units: float = 0.50,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="world_menon_placement_agreement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon placement samples were provided.",
            suggested_next_diagnostic="Capture entity world and Menon scene positions from the same timestamp.",
        )
    try:
        matrix = matrix_from_col_major(world_to_menon_col_major, name="world_to_menon")
        world_points = np.asarray([sample.world_point for sample in samples], dtype=np.float64).reshape((-1, 3))
        expected_menon = transform_points(world_points, matrix)
        observed_menon = np.asarray([sample.menon_point for sample in samples], dtype=np.float64).reshape((-1, 3))
        errors = np.linalg.norm(expected_menon - observed_menon, axis=1)
    except Exception as exc:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="world_menon_placement_agreement",
            status=CheckStatus.FAIL,
            failure_type=FailureType.TRANSFORM,
            detail=f"Placement agreement failed: {exc}",
            suggested_next_diagnostic="Inspect transform audit fields and entity position snapshots.",
        )
    p95 = float(np.percentile(errors, 95))
    if p95 <= good_p95_scene_units:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon placements match Noesis world positions after the declared transform."
    elif p95 <= fail_p95_scene_units:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Menon placements are outside the good band but below fail threshold."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "Menon placements do not match Noesis world positions after the declared transform."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="world_menon_placement_agreement",
        status=status,
        failure_type=failure_type,
        metric={"p95_error_scene_units": p95, "mean_error_scene_units": float(np.mean(errors)), "sample_count": len(samples)},
        threshold={"good_p95_scene_units": good_p95_scene_units, "fail_p95_scene_units": fail_p95_scene_units},
        evidence=sorted({e for sample in samples for e in sample.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Audit whether Menon is converting backend_world_m exactly once.",
    )


def validate_menon_floor_contact(
    samples: Sequence[MenonPlacementSample],
    *,
    check_id: str = "MENON.floor_contact",
    floor_y_scene: float = 0.0,
    good_p95_scene_units: float = 0.05,
    fail_p95_scene_units: float = 0.20,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_floor_contact",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon placement samples were provided.",
            suggested_next_diagnostic="Capture Menon entity positions and floor height from the same render window.",
        )
    try:
        y_values = np.asarray([float(sample.menon_point[1]) for sample in samples], dtype=np.float64)
    except Exception as exc:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_floor_contact",
            status=CheckStatus.FAIL,
            failure_type=FailureType.PROJECTION,
            detail=f"Unable to read Menon placement Y values: {exc}",
            suggested_next_diagnostic="Check Menon trace placement schema.",
        )
    errors = np.abs(y_values - float(floor_y_scene))
    p95 = float(np.percentile(errors, 95))
    if p95 <= good_p95_scene_units:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon entity feet/anchors are in contact with the declared floor."
    elif p95 <= fail_p95_scene_units:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Menon floor contact is outside the good band."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "Menon floor contact exceeds fail threshold."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_floor_contact",
        status=status,
        failure_type=failure_type,
        metric={"p95_floor_error_scene_units": p95, "mean_floor_error_scene_units": float(np.mean(errors)), "sample_count": len(samples)},
        threshold={"floor_y_scene": float(floor_y_scene), "good_p95_scene_units": good_p95_scene_units, "fail_p95_scene_units": fail_p95_scene_units},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check avatar anchor origin, scene floor height, and backend-world to scene transform.",
    )


def validate_menon_trail_agreement(
    world_to_menon_col_major: Sequence[float],
    trails: Sequence[MenonTrailSample],
    *,
    check_id: str = "MENON.trail_agreement",
    good_p95_scene_units: float = 0.15,
    fail_p95_scene_units: float = 0.75,
) -> ValidationCheck:
    if not trails:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_trail_agreement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon trail samples were provided.",
            suggested_next_diagnostic="Capture Noesis world trail samples and Menon rendered trail points for matching identities.",
        )
    try:
        matrix = matrix_from_col_major(world_to_menon_col_major, name="world_to_menon")
    except Exception as exc:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_trail_agreement",
            status=CheckStatus.FAIL,
            failure_type=FailureType.TRANSFORM,
            detail=f"Unable to parse world_to_menon transform: {exc}",
            suggested_next_diagnostic="Inspect scene_similarity.world_to_scene_col_major.",
        )
    errors: list[float] = []
    compared_points = 0
    for trail in trails:
        n = min(len(trail.noesis_world_points), len(trail.menon_points))
        if n <= 0:
            continue
        try:
            expected = transform_points(np.asarray(trail.noesis_world_points[:n], dtype=np.float64), matrix)
            observed = np.asarray(trail.menon_points[:n], dtype=np.float64).reshape((-1, 3))
        except Exception:
            continue
        delta = np.linalg.norm(expected - observed, axis=1)
        errors.extend(float(v) for v in delta if np.isfinite(v))
        compared_points += int(delta.size)
    if not errors:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_trail_agreement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No comparable Menon/Noesis trail points were available.",
            suggested_next_diagnostic="Check trail identity matching and point dimensionality.",
        )
    p95 = float(np.percentile(np.asarray(errors, dtype=np.float64), 95))
    if p95 <= good_p95_scene_units:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon trail points match transformed Noesis world trail points."
    elif p95 <= fail_p95_scene_units:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Menon trail agreement is outside the good band."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "Menon trail points do not match transformed Noesis world trail points."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_trail_agreement",
        status=status,
        failure_type=failure_type,
        metric={"p95_error_scene_units": p95, "mean_error_scene_units": float(np.mean(errors)), "comparison_count": compared_points},
        threshold={"good_p95_scene_units": good_p95_scene_units, "fail_p95_scene_units": fail_p95_scene_units},
        evidence=sorted({e for trail in trails for e in trail.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check whether Menon applies backend_world_m to scene conversion exactly once for trails.",
    )


def validate_bev_menon_trail_agreement(
    world_to_menon_col_major: Sequence[float],
    trails: Sequence[MenonBevTrailSample],
    *,
    check_id: str = "MENON.bev_trail_agreement",
    good_p95_scene_units: float = 0.15,
    fail_p95_scene_units: float = 0.75,
) -> ValidationCheck:
    if not trails:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="bev_menon_trail_agreement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No BEV/Menon trail pairs were provided.",
            suggested_next_diagnostic="Capture BEV trail points and Menon rendered trail points for matching identities.",
        )
    try:
        matrix = matrix_from_col_major(world_to_menon_col_major, name="world_to_menon")
    except Exception as exc:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="bev_menon_trail_agreement",
            status=CheckStatus.FAIL,
            failure_type=FailureType.TRANSFORM,
            detail=f"Unable to parse world_to_menon transform: {exc}",
            suggested_next_diagnostic="Inspect scene_similarity.world_to_scene_col_major.",
        )
    errors: list[float] = []
    compared_points = 0
    for trail in trails:
        n = min(len(trail.bev_points_xz), len(trail.menon_points))
        if n <= 0:
            continue
        try:
            world_points = np.asarray(
                [[float(point[0]), float(trail.floor_y_world), float(point[1])] for point in trail.bev_points_xz[:n]],
                dtype=np.float64,
            )
            expected = transform_points(world_points, matrix)
            observed = np.asarray(trail.menon_points[:n], dtype=np.float64).reshape((-1, 3))
        except Exception:
            continue
        delta = np.linalg.norm(expected - observed, axis=1)
        errors.extend(float(v) for v in delta if np.isfinite(v))
        compared_points += int(delta.size)
    if not errors:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="bev_menon_trail_agreement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No comparable BEV/Menon trail points were available.",
            suggested_next_diagnostic="Check identity matching and BEV point dimensionality.",
        )
    p95 = float(np.percentile(np.asarray(errors, dtype=np.float64), 95))
    if p95 <= good_p95_scene_units:
        status = CheckStatus.PASS
        failure_type = None
        detail = "BEV trails match Menon rendered trails after the declared world-to-scene transform."
    elif p95 <= fail_p95_scene_units:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "BEV/Menon trail agreement is outside the good band."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "BEV trails do not match Menon rendered trails after the declared transform."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="bev_menon_trail_agreement",
        status=status,
        failure_type=failure_type,
        metric={"p95_error_scene_units": p95, "mean_error_scene_units": float(np.mean(errors)), "comparison_count": compared_points},
        threshold={"good_p95_scene_units": good_p95_scene_units, "fail_p95_scene_units": fail_p95_scene_units},
        evidence=sorted({e for trail in trails for e in trail.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check BEV/world axes and Menon's world-to-scene conversion.",
    )


def validate_world_bev_round_trip(
    world_to_bev_col_major: Sequence[float],
    samples: Sequence[WorldBevSample],
    *,
    check_id: str = "BEV.world_round_trip",
    good_p95_agreement_m: float = 0.10,
    fail_p95_agreement_m: float = 0.50,
    good_max_round_trip_m: float = 1e-6,
    fail_max_round_trip_m: float = 1e-3,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="world_bev_round_trip",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No world/BEV samples were provided.",
            suggested_next_diagnostic="Capture backend_world_m points, BEV X/Z points, and the declared world-to-BEV transform for matching entities.",
        )
    try:
        matrix = matrix_from_col_major(world_to_bev_col_major, name="world_to_bev")
        inverse = np.linalg.inv(matrix)
    except Exception as exc:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="world_bev_round_trip",
            status=CheckStatus.FAIL,
            failure_type=FailureType.TRANSFORM,
            detail=f"Unable to parse or invert world-to-BEV transform: {exc}",
            suggested_next_diagnostic="Inspect BEV transform matrix shape, convention, determinant, scale, and origin.",
        )
    agreement_errors: list[float] = []
    round_trip_errors: list[float] = []
    invalid_samples = 0
    for sample in samples:
        try:
            world = np.asarray(sample.world_point, dtype=np.float64).reshape((1, 3))
            bev_xz = np.asarray(sample.bev_point_xz, dtype=np.float64).reshape((2,))
            if not np.all(np.isfinite(world)) or not np.all(np.isfinite(bev_xz)):
                invalid_samples += 1
                continue
            expected_bev = transform_points(world, matrix)[0]
            agreement_errors.append(float(np.linalg.norm(expected_bev[[0, 2]] - bev_xz)))
            observed_bev_3d = np.asarray([[bev_xz[0], float(sample.bev_y), bev_xz[1]]], dtype=np.float64)
            world_round_trip = transform_points(observed_bev_3d, inverse)[0]
            round_trip_errors.append(float(np.linalg.norm(world_round_trip - world[0])))
        except Exception:
            invalid_samples += 1
    if not agreement_errors:
        return ValidationCheck(
            id=check_id,
            domain="bev",
            name="world_bev_round_trip",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No comparable finite world/BEV samples were available.",
            suggested_next_diagnostic="Check world point, BEV point, and transform dimensionality.",
        )
    p95_agreement = float(np.percentile(np.asarray(agreement_errors, dtype=np.float64), 95))
    max_round_trip = float(max(round_trip_errors)) if round_trip_errors else float("inf")
    if invalid_samples or p95_agreement > fail_p95_agreement_m or max_round_trip > fail_max_round_trip_m:
        status = CheckStatus.FAIL
        failure_type = FailureType.TRANSFORM
        detail = "World/BEV transform agreement or round-trip exceeds fail thresholds."
    elif p95_agreement > good_p95_agreement_m or max_round_trip > good_max_round_trip_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.TRANSFORM
        detail = "World/BEV transform agreement is outside the good band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "World points, BEV points, and the declared world-to-BEV transform agree and round-trip cleanly."
    return ValidationCheck(
        id=check_id,
        domain="bev",
        name="world_bev_round_trip",
        status=status,
        failure_type=failure_type,
        metric={
            "p95_agreement_m": p95_agreement,
            "mean_agreement_m": float(np.mean(agreement_errors)),
            "max_round_trip_error_m": max_round_trip,
            "invalid_sample_count": invalid_samples,
            "sample_count": len(samples),
        },
        threshold={
            "good_p95_agreement_m": good_p95_agreement_m,
            "fail_p95_agreement_m": fail_p95_agreement_m,
            "good_max_round_trip_m": good_max_round_trip_m,
            "fail_max_round_trip_m": fail_max_round_trip_m,
        },
        evidence=sorted({e for sample in samples for e in sample.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect BEV transform scale/origin, axis mapping, and whether backend_world_m was converted more than once.",
    )


def validate_transform_audit(
    samples: Sequence[MenonTransformAuditSample],
    *,
    check_id: str = "MENON.transform_audit",
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_transform_audit",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon transform audit samples were provided.",
            suggested_next_diagnostic="Capture transform audit stages from Menon placement code or browser debug state.",
        )
    missing: list[str] = []
    for sample in samples:
        frames: set[str] = set()
        stages: set[str] = set()
        for stage in sample.stages:
            for key in ("frame", "from_frame", "to_frame", "source_frame", "target_frame"):
                value = stage.get(key)
                if isinstance(value, str) and value.strip():
                    frames.add(value.strip())
            value = stage.get("stage") or stage.get("name")
            if isinstance(value, str) and value.strip():
                stages.add(value.strip())
        if "backend_world_m" not in frames or "menon_scene" not in frames:
            missing.append(sample.entity_id)
        if not stages:
            missing.append(f"{sample.entity_id}:no_stages")
    if not missing:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon transform audits declare backend_world_m to menon_scene stages."
    else:
        status = CheckStatus.WARNING
        failure_type = FailureType.TRANSFORM
        detail = "Some Menon transform audits are missing required frame/stage declarations."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_transform_audit",
        status=status,
        failure_type=failure_type,
        metric={"sample_count": len(samples), "missing_count": len(missing), "missing": missing[:20]},
        threshold={"required_frames": ["backend_world_m", "menon_scene"]},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Ensure Menon exports each transform stage used for placement.",
    )


def validate_menon_camera_reprojection(
    samples: Sequence[MenonCameraReprojectionSample],
    *,
    check_id: str = "MENON.camera_reprojection",
    required_layers: Sequence[str] = (
        "source_frame",
        "menon_render",
        "detected_bbox",
        "projected_avatar",
        "room_mesh_edges",
        "floor_grid",
        "anchors",
    ),
    good_mean_error_px: float = 12.0,
    fail_mean_error_px: float = 50.0,
    good_max_error_px: float = 30.0,
    fail_max_error_px: float = 100.0,
    good_iou: float = 0.50,
    fail_iou: float = 0.25,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_camera_reprojection",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon camera-view reprojection samples were provided.",
            suggested_next_diagnostic="Capture a Menon render from the selected camera pose with source-frame overlay evidence.",
        )
    missing_layers = 0
    missing_source_or_render = 0
    mean_errors: list[float] = []
    max_errors: list[float] = []
    ious: list[float] = []
    evidence: set[str] = set()
    for sample in samples:
        evidence.update(sample.evidence)
        evidence.update(path for path in (sample.source_frame, sample.menon_render, sample.overlay_path) if path)
        layers = {str(layer).strip() for layer in sample.layers if str(layer).strip()}
        if sample.source_frame:
            layers.add("source_frame")
        if sample.menon_render:
            layers.add("menon_render")
        if "source_frame" not in layers or "menon_render" not in layers:
            missing_source_or_render += 1
        missing_layers += len([layer for layer in required_layers if layer not in layers])
        for value in (
            sample.mean_error_px,
            sample.anchor_mean_error_px,
            sample.floor_grid_mean_error_px,
            sample.room_edge_mean_error_px,
        ):
            if value is not None and np.isfinite(float(value)):
                mean_errors.append(float(value))
        for value in (sample.max_error_px, sample.anchor_max_error_px):
            if value is not None and np.isfinite(float(value)):
                max_errors.append(float(value))
        for value in (sample.bbox_iou, sample.avatar_iou, sample.mask_iou):
            if value is not None and np.isfinite(float(value)):
                ious.append(float(value))

    worst_mean = float(max(mean_errors)) if mean_errors else None
    worst_max = float(max(max_errors)) if max_errors else None
    min_iou = float(min(ious)) if ious else None
    if missing_source_or_render or missing_layers:
        status = CheckStatus.BLOCKED
        failure_type = FailureType.INFRASTRUCTURE
        detail = "Menon camera reprojection evidence is missing required source/render layers."
    elif worst_mean is None and worst_max is None and min_iou is None:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Menon camera reprojection has visual layers but no numeric alignment metrics."
    elif (
        (worst_mean is not None and worst_mean > fail_mean_error_px)
        or (worst_max is not None and worst_max > fail_max_error_px)
        or (min_iou is not None and min_iou < fail_iou)
    ):
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "Menon camera reprojection alignment exceeds fail thresholds."
    elif (
        (worst_mean is not None and worst_mean > good_mean_error_px)
        or (worst_max is not None and worst_max > good_max_error_px)
        or (min_iou is not None and min_iou < good_iou)
    ):
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Menon camera reprojection alignment is outside the good band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon camera-view reprojection aligns source frame, room mesh, anchors, and avatar evidence."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_camera_reprojection",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(samples),
            "missing_layer_count": missing_layers,
            "missing_source_or_render_count": missing_source_or_render,
            "worst_mean_error_px": worst_mean,
            "worst_max_error_px": worst_max,
            "min_iou": min_iou,
            "numeric_metric_count": len(mean_errors) + len(max_errors) + len(ious),
        },
        threshold={
            "required_layers": list(required_layers),
            "good_mean_error_px": good_mean_error_px,
            "fail_mean_error_px": fail_mean_error_px,
            "good_max_error_px": good_max_error_px,
            "fail_max_error_px": fail_max_error_px,
            "good_iou": good_iou,
            "fail_iou": fail_iou,
        },
        evidence=sorted(evidence),
        detail=detail,
        suggested_next_diagnostic=None
        if status == CheckStatus.PASS
        else "Render Menon from the selected camera pose and compare projected anchors, room mesh edges, floor grid, bbox/avatar, and source frame alignment.",
    )


def validate_menon_avatar_scale(
    samples: Sequence[MenonAvatarSample],
    *,
    check_id: str = "MENON.avatar_scale",
    good_p95_scene_units: float = 0.15,
    fail_p95_scene_units: float = 0.40,
) -> ValidationCheck:
    pairs: list[tuple[MenonAvatarSample, float]] = []
    for sample in samples:
        if sample.expected_height_m is None or sample.observed_height_scene is None:
            continue
        try:
            expected = float(sample.expected_height_m) * float(sample.scene_units_per_m or 1.0)
            observed = float(sample.observed_height_scene)
        except Exception:
            continue
        if np.isfinite(expected) and np.isfinite(observed) and expected > 0.0 and observed > 0.0:
            pairs.append((sample, abs(observed - expected)))
    if not pairs:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_avatar_scale",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No avatar height samples with expected and observed heights were provided.",
            suggested_next_diagnostic="Capture expected person height and rendered avatar height in Menon scene units.",
        )
    errors = np.asarray([error for _, error in pairs], dtype=np.float64)
    p95 = float(np.percentile(errors, 95))
    if p95 <= good_p95_scene_units:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon avatar scale matches expected real-world height assumptions."
    elif p95 <= fail_p95_scene_units:
        status = CheckStatus.WARNING
        failure_type = FailureType.SEMANTIC
        detail = "Menon avatar scale is outside the good band."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.SEMANTIC
        detail = "Menon avatar scale exceeds fail threshold."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_avatar_scale",
        status=status,
        failure_type=failure_type,
        metric={"p95_height_error_scene_units": p95, "mean_height_error_scene_units": float(np.mean(errors)), "sample_count": len(pairs)},
        threshold={"good_p95_scene_units": good_p95_scene_units, "fail_p95_scene_units": fail_p95_scene_units},
        evidence=sorted({e for sample, _ in pairs for e in sample.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check avatar model scale, scene units per meter, and foot/head anchors.",
    )


def validate_menon_avatar_collision(
    samples: Sequence[MenonAvatarSample],
    wall_segments_xz: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    obstacle_boxes_xz: Sequence[Any] = (),
    *,
    check_id: str = "MENON.avatar_collision",
    default_radius_scene: float = 0.25,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_avatar_collision",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon avatar placement samples were provided.",
            suggested_next_diagnostic="Capture Menon avatar positions and collision geometry from the same scene revision.",
        )
    if not wall_segments_xz and not obstacle_boxes_xz:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_avatar_collision",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon wall or obstacle collision geometry was provided.",
            suggested_next_diagnostic="Export wall segments, furniture bounds, or collision proxy boxes with the trace.",
        )
    parsed_walls: list[tuple[np.ndarray, np.ndarray]] = []
    for raw_a, raw_b in wall_segments_xz:
        try:
            a = np.asarray(raw_a, dtype=np.float64).reshape(2)
            b = np.asarray(raw_b, dtype=np.float64).reshape(2)
        except Exception:
            continue
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            parsed_walls.append((a, b))
    parsed_boxes = [box for box in (_box_min_max_xz(item) for item in obstacle_boxes_xz) if box is not None]
    collision_ids: list[str] = []
    clearances: list[float] = []
    for sample in samples:
        if sample.allowed_collision:
            continue
        try:
            point = np.asarray([float(sample.menon_point[0]), float(sample.menon_point[2])], dtype=np.float64)
            radius = float(sample.radius_scene if sample.radius_scene is not None else default_radius_scene)
        except Exception:
            collision_ids.append(f"{sample.entity_id}:invalid_point")
            continue
        if not np.all(np.isfinite(point)) or not np.isfinite(radius):
            collision_ids.append(f"{sample.entity_id}:invalid_point")
            continue
        sample_clearance = np.inf
        collided = False
        for a, b in parsed_walls:
            clearance = _point_to_segment_distance_xz(point, a, b) - radius
            sample_clearance = min(sample_clearance, clearance)
            if clearance < 0.0:
                collided = True
        for min_xz, max_xz in parsed_boxes:
            expanded_min = min_xz - radius
            expanded_max = max_xz + radius
            dx = max(expanded_min[0] - point[0], 0.0, point[0] - expanded_max[0])
            dz = max(expanded_min[1] - point[1], 0.0, point[1] - expanded_max[1])
            clearance = float(np.hypot(dx, dz))
            if expanded_min[0] <= point[0] <= expanded_max[0] and expanded_min[1] <= point[1] <= expanded_max[1]:
                clearance = -min(point[0] - expanded_min[0], expanded_max[0] - point[0], point[1] - expanded_min[1], expanded_max[1] - point[1])
            sample_clearance = min(sample_clearance, clearance)
            if clearance < 0.0:
                collided = True
        if np.isfinite(sample_clearance):
            clearances.append(float(sample_clearance))
        if collided:
            collision_ids.append(sample.entity_id)
    status = CheckStatus.PASS if not collision_ids else CheckStatus.FAIL
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_avatar_collision",
        status=status,
        failure_type=None if status == CheckStatus.PASS else FailureType.SEMANTIC,
        metric={
            "sample_count": len(samples),
            "collision_count": len(collision_ids),
            "collision_ids": collision_ids[:20],
            "min_clearance_scene_units": min(clearances) if clearances else None,
            "wall_segment_count": len(parsed_walls),
            "obstacle_box_count": len(parsed_boxes),
        },
        threshold={"max_collision_count": 0, "default_radius_scene": default_radius_scene},
        evidence=sorted({e for sample in samples for e in sample.evidence}),
        detail="Menon avatars do not intersect supplied walls or obstacle boxes." if status == CheckStatus.PASS else "One or more Menon avatars intersect supplied walls or obstacle boxes.",
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check avatar anchor, scene collision proxies, wall/furniture bounds, and world-to-scene transform.",
    )


def validate_menon_object_placement(
    world_to_menon_col_major: Sequence[float] | None,
    samples: Sequence[MenonObjectSample],
    wall_segments_xz: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    obstacle_boxes_xz: Sequence[Any] = (),
    *,
    check_id: str = "MENON.object_placement",
    good_p95_scene_units: float = 0.10,
    fail_p95_scene_units: float = 0.50,
    good_dimension_error_scene_units: float = 0.10,
    fail_dimension_error_scene_units: float = 0.35,
    good_support_error_scene_units: float = 0.05,
    fail_support_error_scene_units: float = 0.20,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_object_placement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon object placement samples were provided.",
            suggested_next_diagnostic="Capture rendered object positions, dimensions, support surfaces, room labels, and collision proxies from the Menon scene.",
        )
    matrix = None
    if world_to_menon_col_major is not None:
        try:
            matrix = matrix_from_col_major(world_to_menon_col_major, name="world_to_menon")
        except Exception:
            return ValidationCheck(
                id=check_id,
                domain="menon",
                name="menon_object_placement",
                status=CheckStatus.FAIL,
                failure_type=FailureType.TRANSFORM,
                detail="Unable to parse world-to-Menon transform for object placement validation.",
                suggested_next_diagnostic="Inspect scene_similarity.world_to_scene_col_major and matrix convention.",
            )
    parsed_walls: list[tuple[np.ndarray, np.ndarray]] = []
    for raw_a, raw_b in wall_segments_xz:
        try:
            a = np.asarray(raw_a, dtype=np.float64).reshape(2)
            b = np.asarray(raw_b, dtype=np.float64).reshape(2)
        except Exception:
            continue
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            parsed_walls.append((a, b))
    parsed_boxes = [box for box in (_box_min_max_xz(item) for item in obstacle_boxes_xz) if box is not None]

    placement_errors: list[float] = []
    dimension_errors: list[float] = []
    support_errors: list[float] = []
    invalid_values = 0
    missing_transform_count = 0
    room_incompatibilities = 0
    wall_intersections = 0
    obstacle_overlaps = 0
    for sample in samples:
        try:
            menon_point = np.asarray(sample.menon_point, dtype=np.float64).reshape((1, 3))
            if not np.all(np.isfinite(menon_point)):
                invalid_values += 1
                continue
        except Exception:
            invalid_values += 1
            continue
        if sample.world_point is not None:
            if matrix is None:
                missing_transform_count += 1
            else:
                try:
                    world = np.asarray(sample.world_point, dtype=np.float64).reshape((1, 3))
                    expected = transform_points(world, matrix)
                    placement_errors.append(float(np.linalg.norm(expected[0] - menon_point[0])))
                except Exception:
                    invalid_values += 1
        if sample.allowed_rooms:
            allowed = {str(room).strip().lower() for room in sample.allowed_rooms if str(room).strip()}
            room = str(sample.room or "").strip().lower()
            if room not in allowed:
                room_incompatibilities += 1

        bbox: tuple[np.ndarray, np.ndarray] | None = None
        if sample.bbox_min_scene is not None and sample.bbox_max_scene is not None:
            try:
                bbox_min = np.asarray(sample.bbox_min_scene, dtype=np.float64).reshape((3,))
                bbox_max = np.asarray(sample.bbox_max_scene, dtype=np.float64).reshape((3,))
                if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
                    invalid_values += 1
                else:
                    support_errors.append(abs(float(min(bbox_min[1], bbox_max[1])) - float(sample.support_y_scene)))
                    min_xz = np.asarray([min(bbox_min[0], bbox_max[0]), min(bbox_min[2], bbox_max[2])], dtype=np.float64)
                    max_xz = np.asarray([max(bbox_min[0], bbox_max[0]), max(bbox_min[2], bbox_max[2])], dtype=np.float64)
                    bbox = (min_xz, max_xz)
                    if sample.observed_dimensions_scene is None:
                        observed_dims = np.abs(bbox_max - bbox_min)
                        if sample.expected_dimensions_m is not None:
                            expected_dims = np.asarray(sample.expected_dimensions_m, dtype=np.float64).reshape((3,)) * float(sample.scene_units_per_m or 1.0)
                            dimension_errors.append(float(np.max(np.abs(observed_dims - expected_dims))))
            except Exception:
                invalid_values += 1
        if sample.expected_dimensions_m is not None and sample.observed_dimensions_scene is not None:
            try:
                expected_dims = np.asarray(sample.expected_dimensions_m, dtype=np.float64).reshape((3,)) * float(sample.scene_units_per_m or 1.0)
                observed_dims = np.asarray(sample.observed_dimensions_scene, dtype=np.float64).reshape((3,))
                if np.all(np.isfinite(expected_dims)) and np.all(np.isfinite(observed_dims)):
                    dimension_errors.append(float(np.max(np.abs(observed_dims - expected_dims))))
                else:
                    invalid_values += 1
            except Exception:
                invalid_values += 1
        if bbox is not None and not sample.allowed_collision:
            for wall_a, wall_b in parsed_walls:
                if _segment_intersects_box_xz(wall_a, wall_b, bbox[0], bbox[1]):
                    wall_intersections += 1
                    break
            for obstacle in parsed_boxes:
                if _boxes_overlap_xz(bbox, obstacle):
                    obstacle_overlaps += 1
                    break

    p95_placement = float(np.percentile(np.asarray(placement_errors, dtype=np.float64), 95)) if placement_errors else 0.0
    p95_dimension = float(np.percentile(np.asarray(dimension_errors, dtype=np.float64), 95)) if dimension_errors else 0.0
    p95_support = float(np.percentile(np.asarray(support_errors, dtype=np.float64), 95)) if support_errors else 0.0
    hard_failures = (
        invalid_values
        + missing_transform_count
        + room_incompatibilities
        + wall_intersections
        + obstacle_overlaps
    )
    if (
        hard_failures
        or p95_placement > fail_p95_scene_units
        or p95_dimension > fail_dimension_error_scene_units
        or p95_support > fail_support_error_scene_units
    ):
        status = CheckStatus.FAIL
        failure_type = FailureType.SEMANTIC if (room_incompatibilities or wall_intersections or obstacle_overlaps) else FailureType.PROJECTION
        detail = "Menon objects violate placement, dimension, support, room, or collision thresholds."
    elif (
        p95_placement > good_p95_scene_units
        or p95_dimension > good_dimension_error_scene_units
        or p95_support > good_support_error_scene_units
    ):
        status = CheckStatus.WARNING
        failure_type = FailureType.SEMANTIC
        detail = "Menon objects are usable but outside the good placement, dimension, or support band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon object positions, dimensions, support surfaces, rooms, and collision proxies agree."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_object_placement",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(samples),
            "p95_placement_error_scene_units": p95_placement,
            "p95_dimension_error_scene_units": p95_dimension,
            "p95_support_error_scene_units": p95_support,
            "invalid_value_count": invalid_values,
            "missing_transform_count": missing_transform_count,
            "room_incompatibility_count": room_incompatibilities,
            "wall_intersection_count": wall_intersections,
            "obstacle_overlap_count": obstacle_overlaps,
            "dimension_sample_count": len(dimension_errors),
            "support_sample_count": len(support_errors),
            "wall_segment_count": len(parsed_walls),
            "obstacle_box_count": len(parsed_boxes),
        },
        threshold={
            "good_p95_scene_units": good_p95_scene_units,
            "fail_p95_scene_units": fail_p95_scene_units,
            "good_dimension_error_scene_units": good_dimension_error_scene_units,
            "fail_dimension_error_scene_units": fail_dimension_error_scene_units,
            "good_support_error_scene_units": good_support_error_scene_units,
            "fail_support_error_scene_units": fail_support_error_scene_units,
            "max_room_incompatibilities": 0,
            "max_wall_intersections": 0,
            "max_obstacle_overlaps": 0,
        },
        evidence=sorted({e for sample in samples for e in sample.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check object world-to-scene transform, asset scale, support snap, room assignment, wall segments, and furniture collision proxies.",
    )


def validate_menon_avatar_orientation(
    samples: Sequence[MenonAvatarSample],
    *,
    check_id: str = "MENON.avatar_orientation",
    min_motion_scene_units: float = 0.05,
    good_p95_error_deg: float = 45.0,
    fail_p95_error_deg: float = 120.0,
) -> ValidationCheck:
    errors: list[float] = []
    evidence: set[str] = set()
    for sample in samples:
        if sample.heading_deg is None or sample.movement_vector_xz is None:
            continue
        try:
            vx = float(sample.movement_vector_xz[0])
            vz = float(sample.movement_vector_xz[1])
            heading = float(sample.heading_deg)
        except Exception:
            continue
        motion = float(np.hypot(vx, vz))
        if not np.isfinite(motion) or motion < min_motion_scene_units or not np.isfinite(heading):
            continue
        movement_heading = float(np.degrees(np.arctan2(vz, vx)))
        errors.append(_angle_delta_deg(heading, movement_heading))
        evidence.update(sample.evidence)
    if not errors:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_avatar_orientation",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No moving avatar samples with heading and movement vectors were provided.",
            suggested_next_diagnostic="Capture avatar heading and recent Menon trail delta for moving tracks.",
        )
    p95 = float(np.percentile(np.asarray(errors, dtype=np.float64), 95))
    if p95 <= good_p95_error_deg:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon avatar orientation agrees with movement direction."
    elif p95 <= fail_p95_error_deg:
        status = CheckStatus.WARNING
        failure_type = FailureType.SEMANTIC
        detail = "Menon avatar orientation is outside the good band."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.SEMANTIC
        detail = "Menon avatar orientation contradicts movement direction."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_avatar_orientation",
        status=status,
        failure_type=failure_type,
        metric={"p95_heading_error_deg": p95, "mean_heading_error_deg": float(np.mean(errors)), "sample_count": len(errors)},
        threshold={"min_motion_scene_units": min_motion_scene_units, "good_p95_error_deg": good_p95_error_deg, "fail_p95_error_deg": fail_p95_error_deg},
        evidence=sorted(evidence),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check Menon heading convention, avatar forward axis, and trail smoothing lag.",
    )


def validate_timestamp_alignment(
    noesis_ts_s: Sequence[float],
    menon_ts_s: Sequence[float],
    *,
    check_id: str = "MENON.timestamp_alignment",
    good_p95_lag_s: float = 0.10,
    fail_p95_lag_s: float = 0.50,
) -> ValidationCheck:
    pairs = min(len(noesis_ts_s), len(menon_ts_s))
    if pairs <= 0:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="timestamp_alignment",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No timestamp pairs were provided.",
            suggested_next_diagnostic="Capture Noesis telemetry timestamp and Menon render/update timestamp together.",
        )
    errors = np.abs(np.asarray(menon_ts_s[:pairs], dtype=np.float64) - np.asarray(noesis_ts_s[:pairs], dtype=np.float64))
    p95 = float(np.percentile(errors, 95))
    if p95 <= good_p95_lag_s:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon timestamps are aligned with Noesis telemetry."
    elif p95 <= fail_p95_lag_s:
        status = CheckStatus.WARNING
        failure_type = FailureType.SYNC
        detail = "Menon timestamp lag is outside the good band."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.SYNC
        detail = "Menon timestamp lag exceeds fail threshold."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="timestamp_alignment",
        status=status,
        failure_type=failure_type,
        metric={"p95_lag_s": p95, "mean_lag_s": float(np.mean(errors)), "sample_count": pairs},
        threshold={"good_p95_lag_s": good_p95_lag_s, "fail_p95_lag_s": fail_p95_lag_s},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check websocket buffering, replay clocks, and Menon render queue lag.",
    )


def validate_menon_latency_alignment(
    samples: Sequence[MenonLatencySample],
    *,
    check_id: str = "MENON.latency_alignment",
    good_p95_total_lag_s: float = 0.15,
    fail_p95_total_lag_s: float = 0.75,
    good_p95_queue_lag_s: float = 0.08,
    fail_p95_queue_lag_s: float = 0.35,
) -> ValidationCheck:
    if not samples:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_latency_alignment",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon latency samples were provided.",
            suggested_next_diagnostic="Capture Noesis frame timestamps, telemetry timestamps, Menon update/render/display timestamps, and matching entity IDs.",
        )
    total_lags: list[float] = []
    queue_lags: list[float] = []
    telemetry_lags: list[float] = []
    update_lags: list[float] = []
    render_lags: list[float] = []
    display_lags: list[float] = []
    invalid_values = 0
    evidence: set[str] = set()
    for sample in samples:
        evidence.update(sample.evidence)
        try:
            noesis_ts = float(sample.noesis_ts_s)
        except Exception:
            invalid_values += 1
            continue
        if not np.isfinite(noesis_ts):
            invalid_values += 1
            continue
        stages: list[tuple[str, float]] = []
        for name, raw_value in (
            ("telemetry", sample.telemetry_ts_s),
            ("update", sample.menon_update_ts_s),
            ("render", sample.menon_render_ts_s),
            ("display", sample.menon_display_ts_s),
        ):
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except Exception:
                invalid_values += 1
                continue
            if not np.isfinite(value):
                invalid_values += 1
                continue
            lag = abs(value - noesis_ts)
            stages.append((name, value))
            if name == "telemetry":
                telemetry_lags.append(lag)
            elif name == "update":
                update_lags.append(lag)
            elif name == "render":
                render_lags.append(lag)
            elif name == "display":
                display_lags.append(lag)
        if stages:
            total_lags.append(abs(stages[-1][1] - noesis_ts))
        stage_values = {name: value for name, value in stages}
        for left, right in (("update", "render"), ("render", "display")):
            if left in stage_values and right in stage_values:
                queue_lags.append(abs(stage_values[right] - stage_values[left]))
    if not total_lags:
        return ValidationCheck(
            id=check_id,
            domain="menon",
            name="menon_latency_alignment",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="Latency samples did not include any comparable Menon telemetry/update/render/display timestamps.",
            suggested_next_diagnostic="Export at least one Menon stage timestamp with each Noesis frame timestamp.",
        )
    p95_total = float(np.percentile(np.asarray(total_lags, dtype=np.float64), 95))
    p95_queue = float(np.percentile(np.asarray(queue_lags, dtype=np.float64), 95)) if queue_lags else 0.0
    if invalid_values or p95_total > fail_p95_total_lag_s or p95_queue > fail_p95_queue_lag_s:
        status = CheckStatus.FAIL
        failure_type = FailureType.SYNC
        detail = "Menon latency alignment exceeds fail thresholds or contains invalid timestamp evidence."
    elif p95_total > good_p95_total_lag_s or p95_queue > good_p95_queue_lag_s:
        status = CheckStatus.WARNING
        failure_type = FailureType.SYNC
        detail = "Menon latency alignment is outside the good band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon telemetry, update, render, and display timestamps are aligned with Noesis frame timestamps."
    return ValidationCheck(
        id=check_id,
        domain="menon",
        name="menon_latency_alignment",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(samples),
            "p95_total_lag_s": p95_total,
            "mean_total_lag_s": float(np.mean(total_lags)),
            "p95_queue_lag_s": p95_queue,
            "telemetry_sample_count": len(telemetry_lags),
            "update_sample_count": len(update_lags),
            "render_sample_count": len(render_lags),
            "display_sample_count": len(display_lags),
            "invalid_value_count": invalid_values,
        },
        threshold={
            "good_p95_total_lag_s": good_p95_total_lag_s,
            "fail_p95_total_lag_s": fail_p95_total_lag_s,
            "good_p95_queue_lag_s": good_p95_queue_lag_s,
            "fail_p95_queue_lag_s": fail_p95_queue_lag_s,
        },
        evidence=sorted(evidence),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check DS8 telemetry timestamps, WebSocket buffering, Menon update scheduling, render queue lag, and browser display timing.",
    )
