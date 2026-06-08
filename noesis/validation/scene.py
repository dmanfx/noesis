from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .core import CheckStatus, FailureType, ValidationCheck
from .transforms import matrix_from_col_major, round_trip_points


@dataclass(frozen=True)
class SceneCoordinateSystemObservation:
    scene_id: str
    units: str = "m"
    axis_convention: str = "x_right_y_up_z_forward"
    expected_axis_convention: str = "x_right_y_up_z_forward"
    camera_pose_convention: str = "world_to_camera"
    expected_camera_pose_convention: str = "world_to_camera"
    origin_delta_m: float | None = None
    transform_col_major: Sequence[float] | None = None
    round_trip_points: Sequence[Sequence[float]] = ()
    scale_anchor_expected_m: float | None = None
    scale_anchor_observed_m: float | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class PlaneObservation:
    plane_id: str
    plane_type: str
    normal: Sequence[float]
    rms_deviation_m: float | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class RoomDimensionObservation:
    room_id: str
    width_m: float | None = None
    length_m: float | None = None
    height_m: float | None = None
    expected_width_m: float | None = None
    expected_length_m: float | None = None
    expected_height_m: float | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class RoomGeometryConstraintObservation:
    constraint_id: str
    constraint_type: str
    gap_m: float | None = None
    overlap_m: float | None = None
    angle_error_deg: float | None = None
    doorway_width_m: float | None = None
    doorway_height_m: float | None = None
    doorway_bottom_gap_m: float | None = None
    window_width_m: float | None = None
    window_height_m: float | None = None
    window_sill_height_m: float | None = None
    surface_gap_m: float | None = None
    normal_dot: float | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MeshQualitySample:
    asset_id: str
    bbox_min: Sequence[float]
    bbox_max: Sequence[float]
    triangle_count: int | None = None
    vertex_count: int | None = None
    watertight_expected: bool = False
    watertight: bool | None = None
    non_manifold_edges: int | None = None
    inverted_faces: int | None = None
    duplicate_faces: int | None = None
    uv_overlap_ratio: float | None = None
    lod_max_error_m: float | None = None
    texture_alignment_error_px: float | None = None
    texture_stretch_ratio: float | None = None
    bvh_valid: bool | None = None
    collision_bvh_error_m: float | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class MenonAssetObservation:
    asset_id: str
    asset_type: str
    scale_error_m: float | None = None
    orientation_error_deg: float | None = None
    origin_error_m: float | None = None
    collision_mesh_error_m: float | None = None
    floor_flatness_m: float | None = None
    floor_alignment_error_m: float | None = None
    walkable: bool | None = None
    doorway_blocked: bool = False
    camera_position_error_m: float | None = None
    camera_aim_error_deg: float | None = None
    overlay_rendered_layers: Sequence[str] = ()
    required_overlay_layers: Sequence[str] = ()
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class DepthAnchorObservation:
    anchor_id: str
    expected_depth_m: float
    observed_depth_m: float
    order_group: str | None = None
    expected_order: int | None = None
    plane_residual_m: float | None = None
    object_depth_error_m: float | None = None
    temporal_std_m: float | None = None
    static_std_m: float | None = None
    edge_alignment_error_px: float | None = None
    confidence: float | None = None
    fusion_weight: float | None = None
    camera: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


@dataclass(frozen=True)
class SemanticObjectObservation:
    object_id: str
    category: str
    bbox_min: Sequence[float]
    bbox_max: Sequence[float]
    support_y: float = 0.0
    max_wall_intersection_m: float = 0.0
    max_static_shift_m: float | None = None
    allowed_rooms: Sequence[str] = ()
    doorway_clearance_m: float | None = None
    free_space_clearance_m: float | None = None
    walkable_area_blocked_ratio: float | None = None
    known_anchor_error_m: float | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


def _normal(value: Sequence[float]) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape((3,))
    except Exception:
        return None
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return arr / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def validate_scene_coordinate_system(
    observations: Sequence[SceneCoordinateSystemObservation],
    *,
    check_id: str = "SCENE.coordinate_system",
    allowed_units: Sequence[str] = ("m", "meter", "meters"),
    good_origin_delta_m: float = 0.05,
    fail_origin_delta_m: float = 0.20,
    good_scale_error_m: float = 0.05,
    fail_scale_error_m: float = 0.20,
    good_round_trip_m: float = 1e-6,
    fail_round_trip_m: float = 1e-3,
) -> ValidationCheck:
    if not observations:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="scene_coordinate_system",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No scene coordinate-system observations were provided.",
            suggested_next_diagnostic="Export units, axis convention, camera pose convention, origin stability, and scale anchors.",
        )
    unit_failures = 0
    axis_failures = 0
    pose_failures = 0
    handedness_failures = 0
    transform_failures = 0
    max_origin_delta = 0.0
    max_scale_error = 0.0
    max_round_trip = 0.0
    for obs in observations:
        unit = str(obs.units or "").strip().lower()
        if unit not in {str(item).strip().lower() for item in allowed_units}:
            unit_failures += 1
        if str(obs.axis_convention or "").strip() != str(obs.expected_axis_convention or "").strip():
            axis_failures += 1
        if str(obs.camera_pose_convention or "").strip() != str(obs.expected_camera_pose_convention or "").strip():
            pose_failures += 1
        if obs.origin_delta_m is not None:
            try:
                max_origin_delta = max(max_origin_delta, abs(float(obs.origin_delta_m)))
            except Exception:
                transform_failures += 1
        if obs.scale_anchor_expected_m is not None and obs.scale_anchor_observed_m is not None:
            try:
                max_scale_error = max(max_scale_error, abs(float(obs.scale_anchor_observed_m) - float(obs.scale_anchor_expected_m)))
            except Exception:
                transform_failures += 1
        if obs.transform_col_major is not None:
            try:
                matrix = matrix_from_col_major(obs.transform_col_major, name=f"{obs.scene_id}.transform")
                det = float(np.linalg.det(matrix[:3, :3]))
                if det <= 0.0:
                    handedness_failures += 1
                if obs.round_trip_points:
                    rt = round_trip_points(obs.round_trip_points, matrix)
                    max_round_trip = max(max_round_trip, float(rt.max_error))
            except Exception:
                transform_failures += 1
    hard_failures = unit_failures + axis_failures + pose_failures + handedness_failures + transform_failures
    if hard_failures or max_origin_delta > fail_origin_delta_m or max_scale_error > fail_scale_error_m or max_round_trip > fail_round_trip_m:
        status = CheckStatus.FAIL
        failure_type = FailureType.TRANSFORM
        detail = "Scene coordinate-system declarations or transform checks fail validation."
    elif max_origin_delta > good_origin_delta_m or max_scale_error > good_scale_error_m or max_round_trip > good_round_trip_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.TRANSFORM
        detail = "Scene coordinate-system checks pass but are outside the good tolerance band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Scene coordinate system declares meters, expected axes, pose convention, right-handed transform, stable origin, and scale anchors."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="scene_coordinate_system",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(observations),
            "unit_failure_count": unit_failures,
            "axis_failure_count": axis_failures,
            "pose_failure_count": pose_failures,
            "handedness_failure_count": handedness_failures,
            "transform_failure_count": transform_failures,
            "max_origin_delta_m": max_origin_delta,
            "max_scale_error_m": max_scale_error,
            "max_round_trip_m": max_round_trip,
        },
        threshold={
            "allowed_units": list(allowed_units),
            "good_origin_delta_m": good_origin_delta_m,
            "fail_origin_delta_m": fail_origin_delta_m,
            "good_scale_error_m": good_scale_error_m,
            "fail_scale_error_m": fail_scale_error_m,
            "good_round_trip_m": good_round_trip_m,
            "fail_round_trip_m": fail_round_trip_m,
        },
        evidence=sorted({e for obs in observations for e in obs.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check unit export, axis convention, pose matrix direction, handedness, origin anchors, and metric scale anchors.",
    )


def validate_plane_geometry(
    planes: Sequence[PlaneObservation],
    *,
    check_id: str = "SCENE.plane_geometry",
    good_angle_deg: float = 3.0,
    fail_angle_deg: float = 8.0,
    good_floor_rms_m: float = 0.03,
    fail_floor_rms_m: float = 0.10,
) -> ValidationCheck:
    if not planes:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="plane_geometry",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No scene plane observations were provided.",
            suggested_next_diagnostic="Export floor, wall, and ceiling plane normals from the reconstruction artifact.",
        )

    vertical_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    floor_angles: list[float] = []
    floor_rms: list[float] = []
    wall_tilts: list[float] = []
    ceiling_parallel_errors: list[float] = []
    invalid_planes = 0
    for plane in planes:
        normal = _normal(plane.normal)
        if normal is None:
            invalid_planes += 1
            continue
        plane_type = str(plane.plane_type or "").lower()
        if plane_type == "floor":
            floor_angles.append(_angle_deg(normal, vertical_axis))
            if plane.rms_deviation_m is not None:
                floor_rms.append(abs(float(plane.rms_deviation_m)))
        elif plane_type == "wall":
            wall_tilts.append(abs(float(math.degrees(math.asin(float(np.clip(normal[1], -1.0, 1.0)))))))
        elif plane_type == "ceiling":
            ceiling_parallel_errors.append(min(_angle_deg(normal, vertical_axis), _angle_deg(normal, -vertical_axis)))
    max_floor_angle = float(max(floor_angles)) if floor_angles else math.inf
    max_floor_rms = float(max(floor_rms)) if floor_rms else 0.0
    max_wall_tilt = float(max(wall_tilts)) if wall_tilts else 0.0
    max_ceiling_error = float(max(ceiling_parallel_errors)) if ceiling_parallel_errors else 0.0

    if invalid_planes or max_floor_angle > fail_angle_deg or max_wall_tilt > fail_angle_deg or max_ceiling_error > fail_angle_deg or max_floor_rms > fail_floor_rms_m:
        status = CheckStatus.FAIL
        failure_type = FailureType.SCENE
        detail = "Scene planes violate floor, wall, ceiling, or normal validity thresholds."
    elif max_floor_angle > good_angle_deg or max_wall_tilt > good_angle_deg or max_ceiling_error > good_angle_deg or max_floor_rms > good_floor_rms_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.SCENE
        detail = "Scene planes are valid but outside the good tolerance band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Scene planes are flat, vertical/parallel, and oriented as expected."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="plane_geometry",
        status=status,
        failure_type=failure_type,
        metric={
            "max_floor_angle_deg": max_floor_angle,
            "max_floor_rms_m": max_floor_rms,
            "max_wall_tilt_deg": max_wall_tilt,
            "max_ceiling_parallel_error_deg": max_ceiling_error,
            "invalid_plane_count": invalid_planes,
            "sample_count": len(planes),
        },
        threshold={
            "good_angle_deg": good_angle_deg,
            "fail_angle_deg": fail_angle_deg,
            "good_floor_rms_m": good_floor_rms_m,
            "fail_floor_rms_m": fail_floor_rms_m,
        },
        evidence=sorted({e for plane in planes for e in plane.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect plane extraction, gravity/up-axis convention, and room reconstruction scale.",
    )


def _dimension_error(observed: float | None, expected: float | None) -> float | None:
    if observed is None or expected is None:
        return None
    try:
        return abs(float(observed) - float(expected))
    except Exception:
        return None


def validate_room_dimensions(
    rooms: Sequence[RoomDimensionObservation],
    *,
    check_id: str = "SCENE.room_dimensions",
    good_error_m: float = 0.10,
    fail_error_m: float = 0.25,
    min_height_m: float = 2.0,
    max_height_m: float = 4.0,
) -> ValidationCheck:
    if not rooms:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="room_dimensions",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No room dimension observations were provided.",
            suggested_next_diagnostic="Add measured room width, length, and height anchors to the fixture.",
        )
    errors = []
    implausible_height = 0
    for room in rooms:
        for observed, expected in (
            (room.width_m, room.expected_width_m),
            (room.length_m, room.expected_length_m),
            (room.height_m, room.expected_height_m),
        ):
            error = _dimension_error(observed, expected)
            if error is not None and np.isfinite(error):
                errors.append(float(error))
        if room.height_m is not None and not (min_height_m <= float(room.height_m) <= max_height_m):
            implausible_height += 1
    max_error = float(max(errors)) if errors else 0.0
    if implausible_height or max_error > fail_error_m:
        status = CheckStatus.FAIL
        failure_type = FailureType.SCENE
        detail = "Room dimensions exceed fail thresholds or have implausible height."
    elif max_error > good_error_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.SCENE
        detail = "Room dimensions are outside the good tolerance band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Room dimensions match known anchors and plausible height limits."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="room_dimensions",
        status=status,
        failure_type=failure_type,
        metric={"max_dimension_error_m": max_error, "implausible_height_count": implausible_height, "sample_count": len(rooms)},
        threshold={"good_error_m": good_error_m, "fail_error_m": fail_error_m, "min_height_m": min_height_m, "max_height_m": max_height_m},
        evidence=sorted({e for room in rooms for e in room.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check scale anchors, floor/ceiling plane extraction, and room origin convention.",
    )


def validate_room_geometry_constraints(
    constraints: Sequence[RoomGeometryConstraintObservation],
    *,
    check_id: str = "SCENE.room_geometry_constraints",
    good_gap_m: float = 0.05,
    fail_gap_m: float = 0.15,
    good_overlap_m: float = 0.03,
    fail_overlap_m: float = 0.10,
    good_angle_deg: float = 3.0,
    fail_angle_deg: float = 8.0,
    min_door_width_m: float = 0.60,
    max_door_width_m: float = 1.50,
    min_door_height_m: float = 1.80,
    max_door_height_m: float = 2.40,
    min_window_width_m: float = 0.20,
    max_window_width_m: float = 3.00,
    min_window_height_m: float = 0.20,
    max_window_height_m: float = 2.20,
    min_window_sill_height_m: float = 0.20,
    max_window_sill_height_m: float = 1.80,
) -> ValidationCheck:
    if not constraints:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="room_geometry_constraints",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No room geometry constraints were provided.",
            suggested_next_diagnostic="Export wall junction, wall/floor angle, doorway, and surface-continuity measurements.",
        )
    max_gap = 0.0
    max_overlap = 0.0
    max_angle = 0.0
    max_surface_gap = 0.0
    impossible_openings = 0
    impossible_windows = 0
    invalid_normals = 0
    invalid_values = 0
    for item in constraints:
        for raw_value, accumulator in (
            (item.gap_m, "gap"),
            (item.overlap_m, "overlap"),
            (item.angle_error_deg, "angle"),
            (item.surface_gap_m, "surface_gap"),
        ):
            if raw_value is None:
                continue
            try:
                value = abs(float(raw_value))
            except Exception:
                invalid_values += 1
                continue
            if accumulator == "gap":
                max_gap = max(max_gap, value)
            elif accumulator == "overlap":
                max_overlap = max(max_overlap, value)
            elif accumulator == "angle":
                max_angle = max(max_angle, value)
            elif accumulator == "surface_gap":
                max_surface_gap = max(max_surface_gap, value)
        if item.normal_dot is not None:
            try:
                dot = float(item.normal_dot)
                if not np.isfinite(dot) or abs(dot) > 1.0:
                    invalid_normals += 1
            except Exception:
                invalid_normals += 1
        if str(item.constraint_type).lower() in {"doorway", "opening", "door"}:
            try:
                if item.doorway_width_m is not None:
                    width = float(item.doorway_width_m)
                    if not (min_door_width_m <= width <= max_door_width_m):
                        impossible_openings += 1
                if item.doorway_height_m is not None:
                    height = float(item.doorway_height_m)
                    if not (min_door_height_m <= height <= max_door_height_m):
                        impossible_openings += 1
                if item.doorway_bottom_gap_m is not None and abs(float(item.doorway_bottom_gap_m)) > fail_gap_m:
                    impossible_openings += 1
            except Exception:
                invalid_values += 1
        if str(item.constraint_type).lower() in {"window", "window_cutout"}:
            try:
                if item.window_width_m is not None:
                    width = float(item.window_width_m)
                    if not (min_window_width_m <= width <= max_window_width_m):
                        impossible_windows += 1
                if item.window_height_m is not None:
                    height = float(item.window_height_m)
                    if not (min_window_height_m <= height <= max_window_height_m):
                        impossible_windows += 1
                if item.window_sill_height_m is not None:
                    sill_height = float(item.window_sill_height_m)
                    if not (min_window_sill_height_m <= sill_height <= max_window_sill_height_m):
                        impossible_windows += 1
            except Exception:
                invalid_values += 1
    if (
        invalid_values
        or invalid_normals
        or impossible_openings
        or impossible_windows
        or max_gap > fail_gap_m
        or max_overlap > fail_overlap_m
        or max_angle > fail_angle_deg
        or max_surface_gap > fail_gap_m
    ):
        status = CheckStatus.FAIL
        failure_type = FailureType.SCENE
        detail = "Room geometry has invalid junctions, openings, normals, or continuity measurements."
    elif max_gap > good_gap_m or max_overlap > good_overlap_m or max_angle > good_angle_deg or max_surface_gap > good_gap_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.SCENE
        detail = "Room geometry is valid but outside the good tolerance band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Room walls, openings, and surface continuity satisfy geometry constraints."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="room_geometry_constraints",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(constraints),
            "max_gap_m": max_gap,
            "max_overlap_m": max_overlap,
            "max_angle_error_deg": max_angle,
            "max_surface_gap_m": max_surface_gap,
            "impossible_opening_count": impossible_openings,
            "impossible_window_count": impossible_windows,
            "invalid_normal_count": invalid_normals,
            "invalid_value_count": invalid_values,
        },
        threshold={
            "good_gap_m": good_gap_m,
            "fail_gap_m": fail_gap_m,
            "good_overlap_m": good_overlap_m,
            "fail_overlap_m": fail_overlap_m,
            "good_angle_deg": good_angle_deg,
            "fail_angle_deg": fail_angle_deg,
            "min_door_width_m": min_door_width_m,
            "max_door_width_m": max_door_width_m,
            "min_door_height_m": min_door_height_m,
            "max_door_height_m": max_door_height_m,
            "min_window_width_m": min_window_width_m,
            "max_window_width_m": max_window_width_m,
            "min_window_height_m": min_window_height_m,
            "max_window_height_m": max_window_height_m,
            "min_window_sill_height_m": min_window_sill_height_m,
            "max_window_sill_height_m": max_window_sill_height_m,
        },
        evidence=sorted({e for item in constraints for e in item.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect wall junction extraction, opening segmentation, floor contact, and surface merge thresholds.",
    )


def validate_mesh_quality(
    meshes: Sequence[MeshQualitySample],
    *,
    check_id: str = "SCENE.mesh_quality",
    max_triangles: int = 500_000,
    max_uv_overlap_ratio: float = 0.05,
    good_lod_error_m: float = 0.05,
    fail_lod_error_m: float = 0.20,
    good_texture_alignment_error_px: float = 4.0,
    fail_texture_alignment_error_px: float = 16.0,
    good_texture_stretch_ratio: float = 1.25,
    fail_texture_stretch_ratio: float = 2.0,
    good_collision_bvh_error_m: float = 0.05,
    fail_collision_bvh_error_m: float = 0.20,
) -> ValidationCheck:
    if not meshes:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="mesh_quality",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No mesh quality observations were provided.",
            suggested_next_diagnostic="Export mesh stats from GLB/OBJ/PLY generation or Menon asset build.",
        )
    failures = 0
    warnings = 0
    max_extent_m = 0.0
    max_triangles_seen = 0
    max_lod_error = 0.0
    max_texture_alignment_error = 0.0
    max_texture_stretch = 0.0
    max_collision_bvh_error = 0.0
    invalid_bvh_count = 0
    for mesh in meshes:
        try:
            bbox_min = np.asarray(mesh.bbox_min, dtype=np.float64).reshape((3,))
            bbox_max = np.asarray(mesh.bbox_max, dtype=np.float64).reshape((3,))
            extents = bbox_max - bbox_min
            if not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
                failures += 1
            max_extent_m = max(max_extent_m, float(np.max(np.abs(extents))))
        except Exception:
            failures += 1
        triangle_count = int(mesh.triangle_count or 0)
        max_triangles_seen = max(max_triangles_seen, triangle_count)
        if triangle_count > max_triangles:
            warnings += 1
        if mesh.watertight_expected and mesh.watertight is False:
            failures += 1
        for count in (mesh.non_manifold_edges, mesh.inverted_faces, mesh.duplicate_faces):
            if count is not None and int(count) > 0:
                failures += 1
        if mesh.uv_overlap_ratio is not None and float(mesh.uv_overlap_ratio) > max_uv_overlap_ratio:
            warnings += 1
        if mesh.bvh_valid is False:
            invalid_bvh_count += 1
            failures += 1
        for raw_value, good, fail, metric_name in (
            (mesh.lod_max_error_m, good_lod_error_m, fail_lod_error_m, "lod"),
            (mesh.texture_alignment_error_px, good_texture_alignment_error_px, fail_texture_alignment_error_px, "texture_alignment"),
            (mesh.texture_stretch_ratio, good_texture_stretch_ratio, fail_texture_stretch_ratio, "texture_stretch"),
            (mesh.collision_bvh_error_m, good_collision_bvh_error_m, fail_collision_bvh_error_m, "collision_bvh"),
        ):
            if raw_value is None:
                continue
            try:
                value = abs(float(raw_value))
            except Exception:
                failures += 1
                continue
            if not np.isfinite(value):
                failures += 1
                continue
            if metric_name == "lod":
                max_lod_error = max(max_lod_error, value)
            elif metric_name == "texture_alignment":
                max_texture_alignment_error = max(max_texture_alignment_error, value)
            elif metric_name == "texture_stretch":
                max_texture_stretch = max(max_texture_stretch, value)
            elif metric_name == "collision_bvh":
                max_collision_bvh_error = max(max_collision_bvh_error, value)
            if value > fail:
                failures += 1
            elif value > good:
                warnings += 1
    if failures:
        status = CheckStatus.FAIL
        failure_type = FailureType.SCENE
        detail = "One or more meshes have invalid topology, normals, bounding boxes, watertightness, LOD, texture, or collision/BVH evidence."
    elif warnings:
        status = CheckStatus.WARNING
        failure_type = FailureType.SCENE
        detail = "Meshes are usable but exceed triangle, UV, LOD, texture, or collision/BVH warning thresholds."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Mesh topology and bounds satisfy validation thresholds."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="mesh_quality",
        status=status,
        failure_type=failure_type,
        metric={
            "failure_count": failures,
            "warning_count": warnings,
            "invalid_bvh_count": invalid_bvh_count,
            "max_extent_m": max_extent_m,
            "max_triangle_count": max_triangles_seen,
            "max_lod_error_m": max_lod_error,
            "max_texture_alignment_error_px": max_texture_alignment_error,
            "max_texture_stretch_ratio": max_texture_stretch,
            "max_collision_bvh_error_m": max_collision_bvh_error,
            "sample_count": len(meshes),
        },
        threshold={
            "max_triangles": max_triangles,
            "max_uv_overlap_ratio": max_uv_overlap_ratio,
            "good_lod_error_m": good_lod_error_m,
            "fail_lod_error_m": fail_lod_error_m,
            "good_texture_alignment_error_px": good_texture_alignment_error_px,
            "fail_texture_alignment_error_px": fail_texture_alignment_error_px,
            "good_texture_stretch_ratio": good_texture_stretch_ratio,
            "fail_texture_stretch_ratio": fail_texture_stretch_ratio,
            "good_collision_bvh_error_m": good_collision_bvh_error_m,
            "fail_collision_bvh_error_m": fail_collision_bvh_error_m,
        },
        evidence=sorted({e for mesh in meshes for e in mesh.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect mesh export, decimation, LOD comparison, texture atlas bake, UV unwrap, normals, and collision/BVH generation.",
    )


def validate_menon_asset_sanity(
    assets: Sequence[MenonAssetObservation],
    *,
    check_id: str = "SCENE.menon_assets",
    good_scale_error_m: float = 0.05,
    fail_scale_error_m: float = 0.20,
    good_orientation_error_deg: float = 3.0,
    fail_orientation_error_deg: float = 8.0,
    good_origin_error_m: float = 0.05,
    fail_origin_error_m: float = 0.20,
    good_collision_error_m: float = 0.05,
    fail_collision_error_m: float = 0.20,
    good_floor_flatness_m: float = 0.03,
    fail_floor_flatness_m: float = 0.10,
    good_floor_alignment_error_m: float = 0.05,
    fail_floor_alignment_error_m: float = 0.20,
    good_camera_position_error_m: float = 0.10,
    fail_camera_position_error_m: float = 0.30,
    good_camera_aim_error_deg: float = 3.0,
    fail_camera_aim_error_deg: float = 8.0,
) -> ValidationCheck:
    if not assets:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="menon_asset_sanity",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No Menon asset observations were provided.",
            suggested_next_diagnostic="Export Menon room mesh, floor, wall, camera marker, collision, and debug overlay checks from the scene asset build.",
        )
    numeric_fields = {
        "scale_error_m": (good_scale_error_m, fail_scale_error_m),
        "orientation_error_deg": (good_orientation_error_deg, fail_orientation_error_deg),
        "origin_error_m": (good_origin_error_m, fail_origin_error_m),
        "collision_mesh_error_m": (good_collision_error_m, fail_collision_error_m),
        "floor_flatness_m": (good_floor_flatness_m, fail_floor_flatness_m),
        "floor_alignment_error_m": (good_floor_alignment_error_m, fail_floor_alignment_error_m),
        "camera_position_error_m": (good_camera_position_error_m, fail_camera_position_error_m),
        "camera_aim_error_deg": (good_camera_aim_error_deg, fail_camera_aim_error_deg),
    }
    max_values = {key: 0.0 for key in numeric_fields}
    warning_counts = {key: 0 for key in numeric_fields}
    failure_counts = {key: 0 for key in numeric_fields}
    invalid_values = 0
    non_walkable_floor_count = 0
    doorway_blocked_count = 0
    missing_overlay_layer_count = 0
    for asset in assets:
        for key, thresholds in numeric_fields.items():
            raw_value = getattr(asset, key)
            if raw_value is None:
                continue
            try:
                value = abs(float(raw_value))
            except Exception:
                invalid_values += 1
                continue
            if not np.isfinite(value):
                invalid_values += 1
                continue
            max_values[key] = max(max_values[key], value)
            good, fail = thresholds
            if value > fail:
                failure_counts[key] += 1
            elif value > good:
                warning_counts[key] += 1
        if asset.walkable is False:
            non_walkable_floor_count += 1
        if asset.doorway_blocked:
            doorway_blocked_count += 1
        if asset.required_overlay_layers:
            rendered = {str(layer).strip().lower() for layer in asset.overlay_rendered_layers if str(layer).strip()}
            required = {str(layer).strip().lower() for layer in asset.required_overlay_layers if str(layer).strip()}
            missing_overlay_layer_count += len(required - rendered)
    hard_failures = invalid_values + non_walkable_floor_count + doorway_blocked_count + missing_overlay_layer_count + sum(failure_counts.values())
    soft_warnings = sum(warning_counts.values())
    if hard_failures:
        status = CheckStatus.FAIL
        failure_type = FailureType.SCENE
        detail = "Menon assets violate room mesh, collision, floor, wall, camera marker, or debug overlay requirements."
    elif soft_warnings:
        status = CheckStatus.WARNING
        failure_type = FailureType.SCENE
        detail = "Menon assets are usable but outside the good tolerance band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Menon room, floor, wall, camera marker, collision, and debug overlay assets satisfy sanity checks."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="menon_asset_sanity",
        status=status,
        failure_type=failure_type,
        metric={
            "sample_count": len(assets),
            "invalid_value_count": invalid_values,
            "non_walkable_floor_count": non_walkable_floor_count,
            "doorway_blocked_count": doorway_blocked_count,
            "missing_overlay_layer_count": missing_overlay_layer_count,
            "warning_counts": warning_counts,
            "failure_counts": failure_counts,
            "max_values": max_values,
        },
        threshold={
            "good_scale_error_m": good_scale_error_m,
            "fail_scale_error_m": fail_scale_error_m,
            "good_orientation_error_deg": good_orientation_error_deg,
            "fail_orientation_error_deg": fail_orientation_error_deg,
            "good_origin_error_m": good_origin_error_m,
            "fail_origin_error_m": fail_origin_error_m,
            "good_collision_error_m": good_collision_error_m,
            "fail_collision_error_m": fail_collision_error_m,
            "good_floor_flatness_m": good_floor_flatness_m,
            "fail_floor_flatness_m": fail_floor_flatness_m,
            "good_floor_alignment_error_m": good_floor_alignment_error_m,
            "fail_floor_alignment_error_m": fail_floor_alignment_error_m,
            "good_camera_position_error_m": good_camera_position_error_m,
            "fail_camera_position_error_m": fail_camera_position_error_m,
            "good_camera_aim_error_deg": good_camera_aim_error_deg,
            "fail_camera_aim_error_deg": fail_camera_aim_error_deg,
        },
        evidence=sorted({e for asset in assets for e in asset.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Inspect Menon asset scale/origin, collision mesh export, floor walkability, doorway clearance, camera marker pose, and debug overlay layer rendering.",
    )


def validate_depth_consistency(
    anchors: Sequence[DepthAnchorObservation],
    *,
    check_id: str = "SCENE.depth_consistency",
    good_depth_error_m: float = 0.15,
    fail_depth_error_m: float = 0.50,
    good_plane_residual_m: float = 0.08,
    fail_plane_residual_m: float = 0.25,
    good_object_error_m: float = 0.15,
    fail_object_error_m: float = 0.50,
    good_temporal_std_m: float = 0.08,
    fail_temporal_std_m: float = 0.25,
    good_static_std_m: float = 0.06,
    fail_static_std_m: float = 0.20,
    good_edge_error_px: float = 6.0,
    fail_edge_error_px: float = 20.0,
    min_order_gap_m: float = 0.05,
    min_depth_confidence: float = 0.35,
    max_low_confidence_fusion_weight: float = 0.25,
) -> ValidationCheck:
    if not anchors:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="depth_consistency",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No depth anchor observations were provided.",
            suggested_next_diagnostic="Add known depth anchors, plane residuals, or temporal depth variance from a fixture clip.",
        )
    depth_errors: list[float] = []
    plane_residuals: list[float] = []
    object_errors: list[float] = []
    temporal_stds: list[float] = []
    static_stds: list[float] = []
    edge_errors: list[float] = []
    invalid_values = 0
    low_confidence_overweighted = 0
    for anchor in anchors:
        try:
            depth_errors.append(abs(float(anchor.observed_depth_m) - float(anchor.expected_depth_m)))
        except Exception:
            invalid_values += 1
        for raw_value, target in (
            (anchor.plane_residual_m, plane_residuals),
            (anchor.object_depth_error_m, object_errors),
            (anchor.temporal_std_m, temporal_stds),
            (anchor.static_std_m, static_stds),
            (anchor.edge_alignment_error_px, edge_errors),
        ):
            if raw_value is None:
                continue
            try:
                value = abs(float(raw_value))
            except Exception:
                invalid_values += 1
                continue
            if np.isfinite(value):
                target.append(value)
            else:
                invalid_values += 1
        if anchor.confidence is not None and anchor.fusion_weight is not None:
            try:
                confidence = float(anchor.confidence)
                fusion_weight = float(anchor.fusion_weight)
                if np.isfinite(confidence) and np.isfinite(fusion_weight) and confidence < min_depth_confidence and fusion_weight > max_low_confidence_fusion_weight:
                    low_confidence_overweighted += 1
            except Exception:
                invalid_values += 1

    ordering_inversions = 0
    order_comparison_count = 0
    groups: dict[str, list[DepthAnchorObservation]] = {}
    for anchor in anchors:
        if anchor.order_group is None or anchor.expected_order is None:
            continue
        groups.setdefault(str(anchor.order_group), []).append(anchor)
    for rows in groups.values():
        rows.sort(key=lambda item: int(item.expected_order if item.expected_order is not None else 0))
        for idx, near in enumerate(rows):
            for far in rows[idx + 1 :]:
                if near.expected_order == far.expected_order:
                    continue
                order_comparison_count += 1
                try:
                    near_depth = float(near.observed_depth_m)
                    far_depth = float(far.observed_depth_m)
                except Exception:
                    invalid_values += 1
                    continue
                if not np.isfinite(near_depth) or not np.isfinite(far_depth):
                    invalid_values += 1
                    continue
                if near_depth > far_depth - min_order_gap_m:
                    ordering_inversions += 1

    max_depth_error = float(max(depth_errors)) if depth_errors else math.inf
    max_plane_residual = float(max(plane_residuals)) if plane_residuals else 0.0
    max_object_error = float(max(object_errors)) if object_errors else 0.0
    max_temporal_std = float(max(temporal_stds)) if temporal_stds else 0.0
    max_static_std = float(max(static_stds)) if static_stds else 0.0
    max_edge_error = float(max(edge_errors)) if edge_errors else 0.0
    if (
        invalid_values
        or ordering_inversions
        or low_confidence_overweighted
        or max_depth_error > fail_depth_error_m
        or max_plane_residual > fail_plane_residual_m
        or max_object_error > fail_object_error_m
        or max_temporal_std > fail_temporal_std_m
        or max_static_std > fail_static_std_m
        or max_edge_error > fail_edge_error_px
    ):
        status = CheckStatus.FAIL
        failure_type = FailureType.MODEL
        detail = "Depth observations violate metric, ordering, plane, object, temporal, edge, or confidence-weighting thresholds."
    elif (
        max_depth_error > good_depth_error_m
        or max_plane_residual > good_plane_residual_m
        or max_object_error > good_object_error_m
        or max_temporal_std > good_temporal_std_m
        or max_static_std > good_static_std_m
        or max_edge_error > good_edge_error_px
    ):
        status = CheckStatus.WARNING
        failure_type = FailureType.DATA_QUALITY
        detail = "Depth observations are usable but outside the good band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Depth observations agree with anchors, ordering, planes, objects, temporal stability, edges, and confidence weighting."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="depth_consistency",
        status=status,
        failure_type=failure_type,
        metric={
            "max_depth_error_m": max_depth_error,
            "max_plane_residual_m": max_plane_residual,
            "max_object_depth_error_m": max_object_error,
            "max_temporal_std_m": max_temporal_std,
            "max_static_std_m": max_static_std,
            "max_edge_alignment_error_px": max_edge_error,
            "ordering_inversion_count": ordering_inversions,
            "order_comparison_count": order_comparison_count,
            "low_confidence_overweighted_count": low_confidence_overweighted,
            "invalid_value_count": invalid_values,
            "sample_count": len(anchors),
        },
        threshold={
            "good_depth_error_m": good_depth_error_m,
            "fail_depth_error_m": fail_depth_error_m,
            "good_plane_residual_m": good_plane_residual_m,
            "fail_plane_residual_m": fail_plane_residual_m,
            "good_object_error_m": good_object_error_m,
            "fail_object_error_m": fail_object_error_m,
            "good_temporal_std_m": good_temporal_std_m,
            "fail_temporal_std_m": fail_temporal_std_m,
            "good_static_std_m": good_static_std_m,
            "fail_static_std_m": fail_static_std_m,
            "good_edge_error_px": good_edge_error_px,
            "fail_edge_error_px": fail_edge_error_px,
            "min_order_gap_m": min_order_gap_m,
            "min_depth_confidence": min_depth_confidence,
            "max_low_confidence_fusion_weight": max_low_confidence_fusion_weight,
        },
        evidence=sorted({e for anchor in anchors for e in anchor.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check monocular depth scale anchors, relative ordering, reflective masks, edge alignment, static-scene variance, and plane/depth fusion weighting.",
    )


def validate_semantic_object_logic(
    objects: Sequence[SemanticObjectObservation],
    *,
    check_id: str = "SCENE.semantic_objects",
    good_support_error_m: float = 0.05,
    fail_support_error_m: float = 0.20,
    max_static_shift_warning_m: float = 0.40,
    good_doorway_clearance_m: float = 0.75,
    fail_doorway_clearance_m: float = 0.60,
    good_free_space_clearance_m: float = 0.40,
    fail_free_space_clearance_m: float = 0.20,
    good_walkable_blocked_ratio: float = 0.05,
    fail_walkable_blocked_ratio: float = 0.15,
    good_known_anchor_error_m: float = 0.20,
    fail_known_anchor_error_m: float = 0.40,
) -> ValidationCheck:
    if not objects:
        return ValidationCheck(
            id=check_id,
            domain="scene",
            name="semantic_object_logic",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.INFRASTRUCTURE,
            detail="No semantic object observations were provided.",
            suggested_next_diagnostic="Add object bounding boxes, support surfaces, wall intersections, and static-object history to the fixture.",
        )
    support_errors: list[float] = []
    wall_intersections = 0
    static_shift_warnings = 0
    implausible_dimensions = 0
    room_incompatibilities = 0
    doorway_clearance_failures = 0
    doorway_clearance_warnings = 0
    free_space_failures = 0
    free_space_warnings = 0
    walkable_failures = 0
    walkable_warnings = 0
    known_anchor_failures = 0
    known_anchor_warnings = 0
    invalid_values = 0
    for obj in objects:
        try:
            bbox_min = np.asarray(obj.bbox_min, dtype=np.float64).reshape((3,))
            bbox_max = np.asarray(obj.bbox_max, dtype=np.float64).reshape((3,))
            dims = bbox_max - bbox_min
            if not np.all(np.isfinite(dims)) or np.any(dims <= 0.0) or np.max(dims) > 8.0:
                implausible_dimensions += 1
            support_errors.append(abs(float(bbox_min[1]) - float(obj.support_y)))
        except Exception:
            implausible_dimensions += 1
        try:
            if float(obj.max_wall_intersection_m) > 0.0:
                wall_intersections += 1
        except Exception:
            invalid_values += 1
        if obj.max_static_shift_m is not None:
            try:
                if float(obj.max_static_shift_m) > max_static_shift_warning_m:
                    static_shift_warnings += 1
            except Exception:
                invalid_values += 1
        if obj.allowed_rooms:
            allowed = {str(room).strip().lower() for room in obj.allowed_rooms if str(room).strip()}
            room = str(obj.room or "").strip().lower()
            if room not in allowed:
                room_incompatibilities += 1
        if obj.doorway_clearance_m is not None:
            try:
                value = float(obj.doorway_clearance_m)
                if value < fail_doorway_clearance_m:
                    doorway_clearance_failures += 1
                elif value < good_doorway_clearance_m:
                    doorway_clearance_warnings += 1
            except Exception:
                invalid_values += 1
        if obj.free_space_clearance_m is not None:
            try:
                value = float(obj.free_space_clearance_m)
                if value < fail_free_space_clearance_m:
                    free_space_failures += 1
                elif value < good_free_space_clearance_m:
                    free_space_warnings += 1
            except Exception:
                invalid_values += 1
        if obj.walkable_area_blocked_ratio is not None:
            try:
                value = float(obj.walkable_area_blocked_ratio)
                if not np.isfinite(value) or value < 0.0:
                    invalid_values += 1
                elif value > fail_walkable_blocked_ratio:
                    walkable_failures += 1
                elif value > good_walkable_blocked_ratio:
                    walkable_warnings += 1
            except Exception:
                invalid_values += 1
        if obj.known_anchor_error_m is not None:
            try:
                value = abs(float(obj.known_anchor_error_m))
                if value > fail_known_anchor_error_m:
                    known_anchor_failures += 1
                elif value > good_known_anchor_error_m:
                    known_anchor_warnings += 1
            except Exception:
                invalid_values += 1
    max_support_error = float(max(support_errors)) if support_errors else math.inf
    hard_failures = (
        wall_intersections
        + implausible_dimensions
        + room_incompatibilities
        + doorway_clearance_failures
        + free_space_failures
        + walkable_failures
        + known_anchor_failures
        + invalid_values
    )
    soft_warnings = (
        static_shift_warnings
        + doorway_clearance_warnings
        + free_space_warnings
        + walkable_warnings
        + known_anchor_warnings
    )
    if hard_failures or max_support_error > fail_support_error_m:
        status = CheckStatus.FAIL
        failure_type = FailureType.SEMANTIC
        detail = "Objects violate support, collision, room compatibility, doorway, free-space, walkability, anchor, or dimension thresholds."
    elif soft_warnings or max_support_error > good_support_error_m:
        status = CheckStatus.WARNING
        failure_type = FailureType.SEMANTIC
        detail = "Objects are plausible but outside the good semantic consistency band."
    else:
        status = CheckStatus.PASS
        failure_type = None
        detail = "Objects rest on support surfaces and satisfy semantic room, doorway, free-space, and anchor checks."
    return ValidationCheck(
        id=check_id,
        domain="scene",
        name="semantic_object_logic",
        status=status,
        failure_type=failure_type,
        metric={
            "max_support_error_m": max_support_error,
            "wall_intersection_count": wall_intersections,
            "static_shift_warning_count": static_shift_warnings,
            "implausible_dimension_count": implausible_dimensions,
            "room_incompatibility_count": room_incompatibilities,
            "doorway_clearance_failure_count": doorway_clearance_failures,
            "doorway_clearance_warning_count": doorway_clearance_warnings,
            "free_space_failure_count": free_space_failures,
            "free_space_warning_count": free_space_warnings,
            "walkable_area_failure_count": walkable_failures,
            "walkable_area_warning_count": walkable_warnings,
            "known_anchor_failure_count": known_anchor_failures,
            "known_anchor_warning_count": known_anchor_warnings,
            "invalid_value_count": invalid_values,
            "sample_count": len(objects),
        },
        threshold={
            "good_support_error_m": good_support_error_m,
            "fail_support_error_m": fail_support_error_m,
            "max_static_shift_warning_m": max_static_shift_warning_m,
            "good_doorway_clearance_m": good_doorway_clearance_m,
            "fail_doorway_clearance_m": fail_doorway_clearance_m,
            "good_free_space_clearance_m": good_free_space_clearance_m,
            "fail_free_space_clearance_m": fail_free_space_clearance_m,
            "good_walkable_blocked_ratio": good_walkable_blocked_ratio,
            "fail_walkable_blocked_ratio": fail_walkable_blocked_ratio,
            "good_known_anchor_error_m": good_known_anchor_error_m,
            "fail_known_anchor_error_m": fail_known_anchor_error_m,
        },
        evidence=sorted({e for obj in objects for e in obj.evidence}),
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check object scale, floor/support snap, wall collision, room labels, doorway clearance, free-space masks, and persistent object anchors.",
    )
