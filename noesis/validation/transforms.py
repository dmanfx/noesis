from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .core import CheckStatus, FailureType, ValidationCheck


@dataclass(frozen=True)
class RoundTripResult:
    max_error: float
    mean_error: float
    errors: list[float]


def matrix_from_col_major(values: Sequence[float] | np.ndarray, *, name: str = "matrix") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"{name} must contain 16 values")
    matrix = arr.reshape((4, 4), order="F")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def matrix_to_col_major(matrix: Sequence[Sequence[float]] | np.ndarray) -> list[float]:
    arr = np.asarray(matrix, dtype=np.float64).reshape((4, 4))
    return [float(v) for v in arr.reshape(-1, order="F")]


def transform_points(points: Sequence[Sequence[float]] | np.ndarray, matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    mat = np.asarray(matrix, dtype=np.float64).reshape((4, 4))
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (mat @ hom.T).T[:, :3]


def transform_point(point: Sequence[float], matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    return transform_points([point], matrix)[0]


def round_trip_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    forward_matrix: Sequence[Sequence[float]] | np.ndarray,
    inverse_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> RoundTripResult:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if pts.size == 0:
        return RoundTripResult(max_error=0.0, mean_error=0.0, errors=[])
    forward = np.asarray(forward_matrix, dtype=np.float64).reshape((4, 4))
    inverse = np.linalg.inv(forward) if inverse_matrix is None else np.asarray(inverse_matrix, dtype=np.float64).reshape((4, 4))
    recovered = transform_points(transform_points(pts, forward), inverse)
    errors = np.linalg.norm(recovered - pts, axis=1)
    return RoundTripResult(
        max_error=float(np.max(errors)),
        mean_error=float(np.mean(errors)),
        errors=[float(v) for v in errors],
    )


def validate_transform_matrix(
    values: Sequence[float] | np.ndarray,
    *,
    check_id: str,
    domain: str = "transform",
    name: str = "transform_matrix",
    determinant_tolerance: float = 0.15,
    orthonormal_tolerance: float = 0.05,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    try:
        matrix = matrix_from_col_major(values, name=name)
    except Exception as exc:
        return [
            ValidationCheck(
                id=f"{check_id}.shape",
                domain=domain,
                name=f"{name}_shape",
                status=CheckStatus.FAIL,
                failure_type=FailureType.TRANSFORM,
                detail=str(exc),
                suggested_next_diagnostic="Inspect transform serialization and column-major convention.",
            )
        ]

    det = float(np.linalg.det(matrix[:3, :3]))
    try:
        inv = np.linalg.inv(matrix)
        inverse_ok = bool(np.all(np.isfinite(inv)))
    except Exception:
        inverse_ok = False
    checks.append(
        ValidationCheck(
            id=f"{check_id}.invertible",
            domain=domain,
            name=f"{name}_invertible",
            status=CheckStatus.PASS if inverse_ok else CheckStatus.FAIL,
            failure_type=None if inverse_ok else FailureType.TRANSFORM,
            metric={"determinant": det},
            detail="Transform is invertible." if inverse_ok else "Transform is singular or non-finite.",
            suggested_next_diagnostic=None if inverse_ok else "Check camera pose convention and matrix source.",
        )
    )

    rotation = matrix[:3, :3]
    should_be_identity = rotation.T @ rotation
    ortho_err = float(np.max(np.abs(should_be_identity - np.eye(3, dtype=np.float64))))
    det_err = abs(abs(det) - 1.0)
    handedness_ok = det > 0.0
    status = CheckStatus.PASS
    failure_type = None
    detail = "Rotation basis is right-handed and near orthonormal."
    if det_err > determinant_tolerance or ortho_err > orthonormal_tolerance:
        status = CheckStatus.FAIL
        failure_type = FailureType.TRANSFORM
        detail = "Rotation basis is not a valid rigid transform."
    elif not handedness_ok:
        status = CheckStatus.WARNING
        failure_type = FailureType.TRANSFORM
        detail = "Rotation basis appears mirrored."
    checks.append(
        ValidationCheck(
            id=f"{check_id}.rigid_basis",
            domain=domain,
            name=f"{name}_rigid_basis",
            status=status,
            failure_type=failure_type,
            metric={"determinant": det, "orthonormal_max_error": ortho_err},
            threshold={
                "determinant_tolerance": determinant_tolerance,
                "orthonormal_tolerance": orthonormal_tolerance,
            },
            detail=detail,
            suggested_next_diagnostic=None if status == CheckStatus.PASS else "Audit handedness and world-to-camera convention.",
        )
    )
    return checks


def validate_round_trip(
    points: Sequence[Sequence[float]] | np.ndarray,
    forward_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    check_id: str,
    domain: str = "transform",
    name: str = "round_trip",
    good_max_error: float = 1e-6,
    fail_max_error: float = 1e-3,
) -> ValidationCheck:
    try:
        result = round_trip_points(points, forward_matrix)
    except Exception as exc:
        return ValidationCheck(
            id=check_id,
            domain=domain,
            name=name,
            status=CheckStatus.FAIL,
            failure_type=FailureType.TRANSFORM,
            detail=f"Round-trip failed: {exc}",
            suggested_next_diagnostic="Inspect transform invertibility and point dimensionality.",
        )
    max_error = result.max_error
    if max_error <= good_max_error:
        status = CheckStatus.PASS
        detail = "Round-trip error is within tolerance."
        failure_type = None
    elif max_error <= fail_max_error:
        status = CheckStatus.WARNING
        detail = "Round-trip error is above the good band but below the fail threshold."
        failure_type = FailureType.TRANSFORM
    else:
        status = CheckStatus.FAIL
        detail = "Round-trip error exceeds fail threshold."
        failure_type = FailureType.TRANSFORM
    return ValidationCheck(
        id=check_id,
        domain=domain,
        name=name,
        status=status,
        failure_type=failure_type,
        metric={"max_error": max_error, "mean_error": result.mean_error, "sample_count": len(result.errors)},
        threshold={"good_max_error": good_max_error, "fail_max_error": fail_max_error},
        detail=detail,
        suggested_next_diagnostic=None if status == CheckStatus.PASS else "Check scale, origin, and matrix convention.",
    )


def validate_declared_coordinate_frame(payload: Mapping[str, Any], *, check_id: str, allowed_frames: Sequence[str]) -> ValidationCheck:
    frame = payload.get("frame") or payload.get("world_frame") or payload.get("coordinate_frame")
    ok = isinstance(frame, str) and frame in set(allowed_frames)
    return ValidationCheck(
        id=check_id,
        domain="transform",
        name="declared_coordinate_frame",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        failure_type=None if ok else FailureType.TRANSFORM,
        metric={"frame": frame, "allowed_frames": list(allowed_frames)},
        detail="Coordinate frame is declared and allowed." if ok else "Coordinate frame is missing or not allowed.",
        suggested_next_diagnostic=None if ok else "Verify producer emits backend_world_m, camera_local_ground_m, or the expected Menon frame explicitly.",
    )


def finite_point(point: Sequence[float]) -> bool:
    try:
        arr = np.asarray(point, dtype=np.float64).reshape((3,))
    except Exception:
        return False
    return bool(np.all(np.isfinite(arr)))


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not (finite_point(a) and finite_point(b)):
        return math.inf
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))
