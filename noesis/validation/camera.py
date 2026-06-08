from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from noesis.calibration.geometry import pixel_to_world

from .core import CheckStatus, FailureType, ValidationCheck
from .transforms import matrix_from_col_major


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    intrinsics: Sequence[Sequence[float]] | np.ndarray
    extrinsics_col_major: Sequence[float]
    image_size: tuple[int, int]
    floor_y: float = 0.0
    unit_scale: float = 1.0
    stream_kind: str = "unknown"
    applies_to_raw: bool | None = None
    applies_to_dewarped: bool | None = None

    @property
    def K(self) -> np.ndarray:
        return np.asarray(self.intrinsics, dtype=np.float64).reshape((3, 3))

    @property
    def E(self) -> np.ndarray:
        return matrix_from_col_major(self.extrinsics_col_major, name=f"{self.camera_id}.E")


@dataclass(frozen=True)
class ReprojectionAnchor:
    anchor_id: str
    world_point: Sequence[float]
    expected_pixel: Sequence[float]
    camera: str | None = None
    room: str | None = None
    evidence: Sequence[str] = ()


def validate_intrinsics(
    calibration: CameraCalibration,
    *,
    expected_resolution: tuple[int, int] | None = None,
    check_id_prefix: str = "CAM.intrinsics",
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    camera_id = calibration.camera_id
    try:
        K = calibration.K
    except Exception as exc:
        return [
            ValidationCheck(
                id=f"{check_id_prefix}.shape",
                domain="camera",
                name="intrinsics_shape",
                status=CheckStatus.FAIL,
                failure_type=FailureType.CALIBRATION,
                camera=camera_id,
                detail=f"Invalid intrinsics matrix: {exc}",
                suggested_next_diagnostic="Inspect config/cameras.yaml model intrinsics and resolution.",
            )
        ]
    width, height = int(calibration.image_size[0]), int(calibration.image_size[1])
    finite = bool(np.all(np.isfinite(K)))
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    positive_focal = finite and fx > 0.0 and fy > 0.0
    checks.append(
        ValidationCheck(
            id=f"{check_id_prefix}.focal",
            domain="camera",
            name="intrinsics_focal_sanity",
            status=CheckStatus.PASS if positive_focal else CheckStatus.FAIL,
            failure_type=None if positive_focal else FailureType.CALIBRATION,
            camera=camera_id,
            metric={"fx": fx, "fy": fy},
            detail="Focal lengths are finite and positive." if positive_focal else "Focal lengths are invalid.",
            suggested_next_diagnostic=None if positive_focal else "Verify camera intrinsics source and stream resolution.",
        )
    )
    near_center = abs(cx - width / 2.0) <= width * 0.20 and abs(cy - height / 2.0) <= height * 0.20
    checks.append(
        ValidationCheck(
            id=f"{check_id_prefix}.principal_point",
            domain="camera",
            name="intrinsics_principal_point",
            status=CheckStatus.PASS if near_center else CheckStatus.WARNING,
            failure_type=None if near_center else FailureType.CALIBRATION,
            camera=camera_id,
            metric={"cx": cx, "cy": cy, "image_width": width, "image_height": height},
            threshold={"max_center_offset_fraction": 0.20},
            detail="Principal point is near the image center." if near_center else "Principal point is far from center; verify this is intentional.",
            suggested_next_diagnostic=None if near_center else "Check native calibration resolution and dewarped virtual camera model.",
        )
    )
    aspect_expected = width / height if height else math.inf
    aspect_focal = fx / fy if fy else math.inf
    aspect_ok = math.isfinite(aspect_focal) and 0.5 <= (aspect_focal / aspect_expected) <= 2.0
    checks.append(
        ValidationCheck(
            id=f"{check_id_prefix}.aspect",
            domain="camera",
            name="intrinsics_aspect_consistency",
            status=CheckStatus.PASS if aspect_ok else CheckStatus.WARNING,
            failure_type=None if aspect_ok else FailureType.CALIBRATION,
            camera=camera_id,
            metric={"fx_fy_ratio": aspect_focal, "image_aspect": aspect_expected},
            detail="Focal ratio is plausible for the image aspect." if aspect_ok else "Focal ratio is suspicious for the image aspect.",
            suggested_next_diagnostic=None if aspect_ok else "Check whether K was scaled to the active inference/dewarped resolution.",
        )
    )
    if expected_resolution is not None:
        expected = (int(expected_resolution[0]), int(expected_resolution[1]))
        matches = (width, height) == expected
        checks.append(
            ValidationCheck(
                id=f"{check_id_prefix}.resolution",
                domain="camera",
                name="intrinsics_resolution_match",
                status=CheckStatus.PASS if matches else CheckStatus.FAIL,
                failure_type=None if matches else FailureType.CALIBRATION,
                camera=camera_id,
                metric={"image_size": [width, height], "expected_resolution": list(expected)},
                detail="Calibration image size matches expected resolution." if matches else "Calibration image size does not match expected resolution.",
                suggested_next_diagnostic=None if matches else "Verify raw/dewarped/inference resolution ownership.",
            )
        )
    if calibration.stream_kind == "dewarped":
        declared = calibration.applies_to_dewarped is True and calibration.applies_to_raw is not True
        checks.append(
            ValidationCheck(
                id=f"{check_id_prefix}.dewarped_scope",
                domain="camera",
                name="dewarped_virtual_camera_scope",
                status=CheckStatus.PASS if declared else CheckStatus.FAIL,
                failure_type=None if declared else FailureType.CALIBRATION,
                camera=camera_id,
                metric={"applies_to_raw": calibration.applies_to_raw, "applies_to_dewarped": calibration.applies_to_dewarped},
                detail="Calibration scope explicitly describes the dewarped stream." if declared else "Dewarped calibration scope is ambiguous.",
                suggested_next_diagnostic=None if declared else "Declare raw/dewarped ownership and active inference resolution.",
            )
        )
    return checks


def validate_extrinsics(
    calibration: CameraCalibration,
    *,
    check_id_prefix: str = "CAM.extrinsics",
    expected_height_range_m: tuple[float, float] = (1.0, 5.0),
) -> list[ValidationCheck]:
    camera_id = calibration.camera_id
    try:
        E = calibration.E
        Twc = np.linalg.inv(E)
    except Exception as exc:
        return [
            ValidationCheck(
                id=f"{check_id_prefix}.matrix",
                domain="camera",
                name="extrinsics_matrix",
                status=CheckStatus.FAIL,
                failure_type=FailureType.CALIBRATION,
                camera=camera_id,
                detail=f"Invalid extrinsics matrix: {exc}",
                suggested_next_diagnostic="Verify world-to-camera E column-major convention.",
            )
        ]
    center = Twc[:3, 3]
    floor_y = float(calibration.floor_y) * float(calibration.unit_scale or 1.0)
    height = float(center[1] - floor_y)
    min_h, max_h = expected_height_range_m
    height_ok = min_h <= height <= max_h
    checks = [
        ValidationCheck(
            id=f"{check_id_prefix}.height",
            domain="camera",
            name="camera_height_sanity",
            status=CheckStatus.PASS if height_ok else CheckStatus.WARNING,
            failure_type=None if height_ok else FailureType.CALIBRATION,
            camera=camera_id,
            metric={"camera_center": center.tolist(), "height_above_floor_m": height},
            threshold={"min_height_m": min_h, "max_height_m": max_h},
            detail="Camera height is plausible." if height_ok else "Camera height is outside the expected range.",
            suggested_next_diagnostic=None if height_ok else "Check pose scale, floor_y, and world-to-camera inversion.",
        )
    ]
    forward_world = Twc[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    denom = float(forward_world[1])
    if abs(denom) <= 1e-9:
        hit_ok = False
        t_hit = math.inf
    else:
        t_hit = (floor_y - float(center[1])) / denom
        hit_ok = math.isfinite(t_hit) and t_hit > 0.0
    checks.append(
        ValidationCheck(
            id=f"{check_id_prefix}.forward_floor_ray",
            domain="camera",
            name="camera_forward_floor_intersection",
            status=CheckStatus.PASS if hit_ok else CheckStatus.FAIL,
            failure_type=None if hit_ok else FailureType.CALIBRATION,
            camera=camera_id,
            metric={"t_hit": t_hit, "forward_world": forward_world.tolist()},
            detail="Camera forward ray intersects the floor in front of the camera." if hit_ok else "Camera forward ray does not intersect the floor in front of the camera.",
            suggested_next_diagnostic=None if hit_ok else "Audit yaw/pitch/roll and world-to-camera convention.",
        )
    )
    width, height_px = calibration.image_size
    result = pixel_to_world(
        calibration.K,
        list(calibration.extrinsics_col_major),
        calibration.floor_y,
        calibration.unit_scale,
        float(width) / 2.0,
        float(height_px) - 1.0,
    )
    checks.append(
        ValidationCheck(
            id=f"{check_id_prefix}.bottom_center_floor_ray",
            domain="camera",
            name="bottom_center_floor_ray",
            status=CheckStatus.PASS if result.ok else CheckStatus.FAIL,
            failure_type=None if result.ok else FailureType.PROJECTION,
            camera=camera_id,
            metric={"world_point": result.world_point, "method": result.method, "error": result.error},
            detail="Bottom-center ray intersects the floor." if result.ok else "Bottom-center ray does not produce a valid floor point.",
            suggested_next_diagnostic=None if result.ok else "Check intrinsics, extrinsics, and floor_y.",
        )
    )
    return checks


def project_world_to_pixel(calibration: CameraCalibration, world_point: Sequence[float]) -> tuple[float, float, float]:
    point = np.asarray([float(world_point[0]), float(world_point[1]), float(world_point[2]), 1.0], dtype=np.float64)
    camera_point = calibration.E @ point
    z = float(camera_point[2])
    if abs(z) <= 1e-12:
        raise ValueError("world point projects to z=0")
    uvw = calibration.K @ camera_point[:3]
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), z


def validate_reprojection_anchors(
    calibration: CameraCalibration,
    anchors: Sequence[ReprojectionAnchor],
    *,
    check_id_prefix: str = "CAM.reprojection",
    good_max_px: float = 12.0,
    fail_max_px: float = 50.0,
) -> list[ValidationCheck]:
    if not anchors:
        return [
            ValidationCheck(
                id=f"{check_id_prefix}.blocked",
                domain="camera",
                name="known_anchor_reprojection",
                status=CheckStatus.BLOCKED,
                failure_type=FailureType.INFRASTRUCTURE,
                camera=calibration.camera_id,
                detail="No reprojection anchors were provided.",
                suggested_next_diagnostic="Add known world/pixel anchor pairs for this camera.",
            )
        ]
    checks: list[ValidationCheck] = []
    for idx, anchor in enumerate(anchors):
        try:
            u, v, z = project_world_to_pixel(calibration, anchor.world_point)
            expected_u = float(anchor.expected_pixel[0])
            expected_v = float(anchor.expected_pixel[1])
            error_px = float(math.hypot(u - expected_u, v - expected_v))
        except Exception as exc:
            checks.append(
                ValidationCheck(
                    id=f"{check_id_prefix}.{idx:03d}",
                    domain="camera",
                    name="known_anchor_reprojection",
                    status=CheckStatus.FAIL,
                    failure_type=FailureType.PROJECTION,
                    camera=anchor.camera or calibration.camera_id,
                    room=anchor.room,
                    evidence=list(anchor.evidence),
                    detail=f"Anchor reprojection failed: {exc}",
                    suggested_next_diagnostic="Check anchor world frame and camera E convention.",
                )
            )
            continue
        if error_px <= good_max_px:
            status = CheckStatus.PASS
            failure_type = None
            detail = "Anchor reprojection is within the good pixel threshold."
        elif error_px <= fail_max_px:
            status = CheckStatus.WARNING
            failure_type = FailureType.PROJECTION
            detail = "Anchor reprojection is outside the good band but below fail threshold."
        else:
            status = CheckStatus.FAIL
            failure_type = FailureType.PROJECTION
            detail = "Anchor reprojection exceeds fail threshold."
        checks.append(
            ValidationCheck(
                id=f"{check_id_prefix}.{idx:03d}",
                domain="camera",
                name="known_anchor_reprojection",
                status=status,
                failure_type=failure_type,
                camera=anchor.camera or calibration.camera_id,
                room=anchor.room,
                metric={
                    "anchor_id": anchor.anchor_id,
                    "projected_pixel": [u, v],
                    "expected_pixel": [float(anchor.expected_pixel[0]), float(anchor.expected_pixel[1])],
                    "depth_z": z,
                    "error_px": error_px,
                },
                threshold={"good_max_px": good_max_px, "fail_max_px": fail_max_px},
                evidence=list(anchor.evidence),
                detail=detail,
                suggested_next_diagnostic=None if status == CheckStatus.PASS else "Run floor-ray and Menon camera-view alignment checks for this camera.",
            )
        )
    return checks
