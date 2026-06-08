from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import CheckStatus, FailureType, ValidationCheck
from .transforms import euclidean_distance, finite_point


@dataclass(frozen=True)
class AnchorTolerance:
    good_m: float
    warning_m: float


DEFAULT_ANCHOR_TOLERANCES: dict[str, AnchorTolerance] = {
    "room": AnchorTolerance(good_m=0.10, warning_m=0.20),
    "camera": AnchorTolerance(good_m=0.15, warning_m=0.30),
    "doorway": AnchorTolerance(good_m=0.10, warning_m=0.20),
    "object": AnchorTolerance(good_m=0.20, warning_m=0.40),
}


@dataclass(frozen=True)
class AnchorObservation:
    anchor_id: str
    expected_world: Sequence[float]
    observed_world: Sequence[float]
    anchor_type: str = "object"
    camera: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


def _tolerance_for(anchor_type: str, overrides: dict[str, AnchorTolerance] | None = None) -> AnchorTolerance:
    table = dict(DEFAULT_ANCHOR_TOLERANCES)
    if overrides:
        table.update(overrides)
    return table.get(str(anchor_type or "object"), table["object"])


def validate_known_anchors(
    anchors: Sequence[AnchorObservation],
    *,
    check_id_prefix: str = "ANCHOR",
    tolerances: dict[str, AnchorTolerance] | None = None,
) -> list[ValidationCheck]:
    if not anchors:
        return [
            ValidationCheck(
                id=f"{check_id_prefix}.blocked",
                domain="geometry",
                name="known_anchor_validation",
                status=CheckStatus.BLOCKED,
                failure_type=FailureType.INFRASTRUCTURE,
                detail="No known-anchor observations were provided.",
                suggested_next_diagnostic="Add 5 to 10 known anchors for the target room/camera.",
            )
        ]

    checks: list[ValidationCheck] = []
    for idx, anchor in enumerate(anchors):
        tol = _tolerance_for(anchor.anchor_type, tolerances)
        distance_m = euclidean_distance(anchor.expected_world, anchor.observed_world)
        if not math.isfinite(distance_m):
            status = CheckStatus.FAIL
            detail = "Anchor coordinates are missing or non-finite."
        elif distance_m <= tol.good_m:
            status = CheckStatus.PASS
            detail = "Anchor is within the good tolerance band."
        elif distance_m <= tol.warning_m:
            status = CheckStatus.WARNING
            detail = "Anchor is outside the good band but below the fail threshold."
        else:
            status = CheckStatus.FAIL
            detail = "Anchor error exceeds the fail threshold."
        checks.append(
            ValidationCheck(
                id=f"{check_id_prefix}.{idx:03d}",
                domain="geometry",
                name="known_anchor_error",
                status=status,
                failure_type=None if status == CheckStatus.PASS else FailureType.SCENE,
                camera=anchor.camera,
                room=anchor.room,
                metric={"anchor_id": anchor.anchor_id, "anchor_type": anchor.anchor_type, "error_m": distance_m},
                threshold={"good_m": tol.good_m, "fail_m": tol.warning_m},
                evidence=list(anchor.evidence),
                detail=detail,
                suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check scale, origin, and world-to-scene registration for this room.",
            )
        )
    return checks


def point_in_polygon_xz(point: Sequence[float], polygon_xz: Sequence[Sequence[float]]) -> bool:
    if len(polygon_xz) < 3:
        return False
    try:
        x = float(point[0])
        z = float(point[1] if len(point) == 2 else point[2])
    except Exception:
        return False
    inside = False
    j = len(polygon_xz) - 1
    for i, raw_pi in enumerate(polygon_xz):
        raw_pj = polygon_xz[j]
        xi, zi = float(raw_pi[0]), float(raw_pi[1])
        xj, zj = float(raw_pj[0]), float(raw_pj[1])
        intersects = ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def validate_points_inside_room(
    points_world: Sequence[Sequence[float]],
    polygon_xz: Sequence[Sequence[float]],
    *,
    check_id: str = "ROOM.points_inside",
    camera: str | None = None,
    room: str | None = None,
    min_inside_ratio: float = 0.98,
) -> ValidationCheck:
    points = [p for p in points_world if finite_point(p)]
    if not points:
        return ValidationCheck(
            id=check_id,
            domain="geometry",
            name="points_inside_room",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            camera=camera,
            room=room,
            detail="No finite world points were provided.",
            suggested_next_diagnostic="Capture tracking samples with world_valid=true.",
        )
    inside = [point_in_polygon_xz(point, polygon_xz) for point in points]
    ratio = float(np.count_nonzero(inside) / len(inside))
    if ratio >= min_inside_ratio:
        status = CheckStatus.PASS
        failure_type = None
        detail = "World points are inside the room polygon."
    elif ratio > 0.0:
        status = CheckStatus.WARNING
        failure_type = FailureType.PROJECTION
        detail = "Some world points are outside the room polygon."
    else:
        status = CheckStatus.FAIL
        failure_type = FailureType.PROJECTION
        detail = "No world points are inside the room polygon."
    return ValidationCheck(
        id=check_id,
        domain="geometry",
        name="points_inside_room",
        status=status,
        failure_type=failure_type,
        camera=camera,
        room=room,
        metric={"inside_ratio": ratio, "sample_count": len(points), "outside_count": len(points) - int(np.count_nonzero(inside))},
        threshold={"min_inside_ratio": min_inside_ratio},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect coordinate conversion, room polygon frame, and camera-specific calibration.",
    )


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect_2d(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    pa = np.asarray(a, dtype=np.float64).reshape(2)
    pb = np.asarray(b, dtype=np.float64).reshape(2)
    pc = np.asarray(c, dtype=np.float64).reshape(2)
    pd = np.asarray(d, dtype=np.float64).reshape(2)
    o1 = _orientation(pa, pb, pc)
    o2 = _orientation(pa, pb, pd)
    o3 = _orientation(pc, pd, pa)
    o4 = _orientation(pc, pd, pb)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def validate_no_wall_crossing(
    path_xz: Sequence[Sequence[float]],
    wall_segments_xz: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    check_id: str = "ROOM.no_wall_crossing",
    camera: str | None = None,
    room: str | None = None,
) -> ValidationCheck:
    path = [np.asarray(p, dtype=np.float64).reshape(2) for p in path_xz]
    if len(path) < 2:
        return ValidationCheck(
            id=check_id,
            domain="geometry",
            name="no_wall_crossing",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            camera=camera,
            room=room,
            metric={"path_segment_count": 0},
            detail="No track path segments were provided for wall-crossing validation.",
            suggested_next_diagnostic="Capture at least two ordered world points for the same track identity.",
        )
    crossing_count = 0
    for a, b in zip(path, path[1:]):
        for wall_a, wall_b in wall_segments_xz:
            if segments_intersect_2d(a, b, wall_a, wall_b):
                crossing_count += 1
    status = CheckStatus.PASS if crossing_count == 0 else CheckStatus.FAIL
    return ValidationCheck(
        id=check_id,
        domain="geometry",
        name="no_wall_crossing",
        status=status,
        failure_type=None if status == CheckStatus.PASS else FailureType.SEMANTIC,
        camera=camera,
        room=room,
        metric={"crossing_count": crossing_count, "path_segment_count": max(0, len(path) - 1)},
        threshold={"max_crossing_count": 0},
        detail="Path does not cross wall segments." if status == CheckStatus.PASS else "Path crosses one or more wall segments.",
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check room polygon/doorway definitions and world-to-scene transform.",
    )
