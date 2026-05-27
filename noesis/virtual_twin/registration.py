from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from noesis.calibration.scene_registration import solve_similarity_transform

from .menon_obj import StructuralSurface, surface_to_json


class VirtualTwinRegistrationError(RuntimeError):
    """Raised when fused virtual-twin planes cannot be registered to Menon surfaces."""


@dataclass(frozen=True)
class SourcePlaneSurface:
    plane_id: str
    label: str
    centroid: np.ndarray
    normal: np.ndarray
    confidence: float = 1.0
    support_pixels: int = 0


def normalize_label(label: str | None) -> str:
    value = str(label or "").strip().lower()
    if value in {"floor", "ground", "walkable_floor"}:
        return "floor"
    if value in {"ceiling", "ceil"}:
        return "ceiling"
    if value in {"wall", "walls"}:
        return "wall"
    if value == "floor_or_ceiling":
        return value
    if "floor" in value:
        return "floor"
    if "wall" in value:
        return "wall"
    if "ceil" in value:
        return "ceiling"
    return value or "unknown"


def resolve_backend_plane_label(label: str | None, centroid: Sequence[float], normal: Sequence[float], *, floor_y: float) -> str:
    normalized = normalize_label(label)
    if normalized != "floor_or_ceiling":
        return normalized
    point = np.asarray(centroid, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    if point.shape == (3,) and abs(float(n[1])) >= 0.75:
        if float(point[1]) <= float(floor_y) + 0.45:
            return "floor"
        return "ceiling"
    return normalized


def _rotation_from_similarity(matrix: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[:3, :3]
    det = float(np.linalg.det(linear))
    scale = abs(det) ** (1.0 / 3.0) if abs(det) > 1e-12 else 1.0
    return linear / scale


def _matrix_from_col_major(values: Sequence[float] | np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.size != 16:
        raise VirtualTwinRegistrationError("initial world-to-scene transform must contain 16 values")
    matrix = raw.reshape((4, 4), order="F")
    if not np.all(np.isfinite(matrix)):
        raise VirtualTwinRegistrationError("initial world-to-scene transform contains non-finite values")
    return matrix


def _transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ np.asarray(point, dtype=np.float64).reshape(3)) + matrix[:3, 3]


def _transform_normal(matrix: np.ndarray, normal: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[:3, :3]
    n = np.linalg.inv(linear).T @ np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(n))
    return n / norm if norm > 1e-9 else np.zeros(3, dtype=np.float64)


def _normal_angle_deg(source_normal: np.ndarray, target_normal: np.ndarray, matrix: np.ndarray) -> float:
    n = _transform_normal(matrix, source_normal)
    n_norm = float(np.linalg.norm(n))
    t = np.asarray(target_normal, dtype=np.float64).reshape(3)
    t_norm = float(np.linalg.norm(t))
    if n_norm <= 1e-9 or t_norm <= 1e-9:
        return 180.0
    dot = abs(float(np.dot(n / n_norm, t / t_norm)))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def _target_registration_label(surface: StructuralSurface, *, floor_y_scene: float) -> str | None:
    normal = np.asarray(surface.normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.80:
        return None
    normal = normal / norm
    label = normalize_label(surface.label)
    if label == "floor_or_ceiling" and abs(float(normal[1])) >= 0.82:
        if float(surface.centroid[1]) <= float(floor_y_scene) + 40.0:
            return "floor"
        return None
    if label == "wall" and abs(float(normal[1])) <= 0.25:
        return "wall"
    return None


def _floor_y_scene_from_targets(target_surfaces: Sequence[StructuralSurface]) -> float:
    floorish: list[float] = []
    for surface in target_surfaces:
        if normalize_label(surface.label) != "floor_or_ceiling":
            continue
        normal = np.asarray(surface.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm > 0.80 and abs(float((normal / norm)[1])) >= 0.82:
            floorish.append(float(surface.centroid[1]))
    return float(min(floorish)) if floorish else 0.0


def _solve_scale_translation(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, float, float]:
    src = np.asarray(source_points, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise VirtualTwinRegistrationError("scale/translation registration requires at least three 3D pairs")
    src_mean = np.mean(src, axis=0)
    tgt_mean = np.mean(tgt, axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    denom = float(np.sum(src_centered * src_centered))
    scale = float(np.sum(src_centered * tgt_centered) / denom) if denom > 1e-12 else 1.0
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = tgt_mean - (scale * src_mean)
    transformed = (matrix[:3, :3] @ src.T).T + matrix[:3, 3]
    residuals = np.linalg.norm(transformed - tgt, axis=1)
    rmse = float(math.sqrt(float(np.mean(np.square(residuals))))) if residuals.size else 0.0
    return matrix, scale, rmse


def _register_with_initial_transform(
    normalized_sources: Sequence[SourcePlaneSurface],
    target_surfaces: Sequence[StructuralSurface],
    *,
    initial_matrix: np.ndarray,
    min_correspondences: int,
    max_match_normal_error_deg: float = 5.0,
    max_match_distance_scene_units: float = 350.0,
) -> dict[str, Any]:
    floor_y_scene = _floor_y_scene_from_targets(target_surfaces)
    targets_by_label: dict[str, list[StructuralSurface]] = {"floor": [], "wall": []}
    for surface in target_surfaces:
        label = _target_registration_label(surface, floor_y_scene=floor_y_scene)
        if label in targets_by_label:
            targets_by_label[label].append(surface)
    for rows in targets_by_label.values():
        rows.sort(key=lambda s: float(s.area), reverse=True)

    correspondences: list[dict[str, Any]] = []
    source_scene_points: list[np.ndarray] = []
    target_scene_points: list[np.ndarray] = []
    for source in normalized_sources:
        if source.label not in targets_by_label:
            continue
        source_scene = _transform_point(initial_matrix, source.centroid)
        source_scene_normal = _transform_normal(initial_matrix, source.normal)
        best: tuple[float, float, float, StructuralSurface] | None = None
        for target in targets_by_label[source.label]:
            distance = float(np.linalg.norm(source_scene - np.asarray(target.centroid, dtype=np.float64)))
            if distance > float(max_match_distance_scene_units):
                continue
            angle = _normal_angle_deg(source_scene_normal, target.normal, np.eye(4, dtype=np.float64))
            if angle > float(max_match_normal_error_deg):
                continue
            score = distance + (8.0 * angle)
            if best is None or score < best[0]:
                best = (score, distance, angle, target)
        if best is None:
            continue
        _, initial_distance, initial_angle, target = best
        source_scene_points.append(source_scene)
        target_scene_points.append(np.asarray(target.centroid, dtype=np.float64))
        correspondences.append(
            {
                "label": source.label,
                "source_plane_id": source.plane_id,
                "target_surface_id": target.surface_id,
                "source_centroid_m": [float(x) for x in source.centroid],
                "target_centroid_scene": [float(x) for x in target.centroid],
                "source_normal": [float(x) for x in source.normal],
                "target_normal": [float(x) for x in target.normal],
                "source_confidence": float(source.confidence),
                "target_area": float(target.area),
                "initial_distance_scene_units": float(initial_distance),
                "initial_normal_angle_error_deg": float(initial_angle),
            }
        )

    if len(correspondences) < int(min_correspondences):
        raise VirtualTwinRegistrationError(
            "virtual-twin registration needs at least "
            f"{int(min_correspondences)} initial-transform plane/surface correspondences; "
            f"got {len(correspondences)}"
        )

    correction, correction_scale, rmse = _solve_scale_translation(
        np.stack(source_scene_points, axis=0),
        np.stack(target_scene_points, axis=0),
    )
    matrix = correction @ initial_matrix
    residuals = []
    angular = []
    for item in correspondences:
        source_point = np.asarray(item["source_centroid_m"], dtype=np.float64)
        target_point = np.asarray(item["target_centroid_scene"], dtype=np.float64)
        transformed = _transform_point(matrix, source_point)
        residual = float(np.linalg.norm(transformed - target_point))
        angle = _normal_angle_deg(
            np.asarray(item["source_normal"], dtype=np.float64),
            np.asarray(item["target_normal"], dtype=np.float64),
            matrix,
        )
        item["residual_scene_units"] = residual
        item["normal_angle_error_deg"] = angle
        residuals.append(residual)
        angular.append(angle)

    scale = abs(float(np.linalg.det(matrix[:3, :3]))) ** (1.0 / 3.0)
    return {
        "status": "ok",
        "method": "initial_scene_similarity_scale_translation_refinement",
        "correspondence_count": len(correspondences),
        "world_to_menon_scene_col_major": [float(x) for x in matrix.flatten(order="F")],
        "matrix_row_major": [float(x) for x in matrix.reshape(-1)],
        "rotation_row_major": [float(x) for x in _rotation_from_similarity(matrix).reshape(-1)],
        "scene_per_m": float(scale),
        "s_obj_to_m": float(1.0 / scale) if scale > 1e-9 else 1.0,
        "position_rmse_scene_units": float(rmse),
        "median_residual_scene_units": float(np.median(residuals)) if residuals else 0.0,
        "p90_residual_scene_units": float(np.percentile(residuals, 90.0)) if residuals else 0.0,
        "median_normal_error_deg": float(np.median(angular)) if angular else 0.0,
        "p90_normal_error_deg": float(np.percentile(angular, 90.0)) if angular else 0.0,
        "max_normal_error_deg": float(np.max(angular)) if angular else 0.0,
        "correction_col_major": [float(x) for x in correction.flatten(order="F")],
        "correction_scale": float(correction_scale),
        "initial_world_to_menon_scene_col_major": [float(x) for x in initial_matrix.flatten(order="F")],
        "match_thresholds": {
            "max_match_normal_error_deg": float(max_match_normal_error_deg),
            "max_match_distance_scene_units": float(max_match_distance_scene_units),
            "floor_y_scene": float(floor_y_scene),
        },
        "correspondences": correspondences,
        "target_surface_summary": [surface_to_json(s) for s in target_surfaces[:64]],
    }


def register_planes_to_menon_surfaces(
    source_planes: Sequence[SourcePlaneSurface | Mapping[str, Any]],
    target_surfaces: Sequence[StructuralSurface],
    *,
    floor_y: float = 0.0,
    min_correspondences: int = 3,
    initial_world_to_scene_col_major: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    normalized_sources: list[SourcePlaneSurface] = []
    for item in source_planes:
        if isinstance(item, SourcePlaneSurface):
            source = item
        else:
            source = SourcePlaneSurface(
                plane_id=str(item.get("plane_id") or item.get("id") or ""),
                label=str(item.get("label") or item.get("semantic_label") or ""),
                centroid=np.asarray(item.get("centroid"), dtype=np.float64),
                normal=np.asarray(item.get("normal"), dtype=np.float64),
                confidence=float(item.get("confidence", 1.0) or 1.0),
                support_pixels=int(item.get("support_pixels", 0) or 0),
            )
        label = resolve_backend_plane_label(source.label, source.centroid, source.normal, floor_y=floor_y)
        if source.centroid.shape != (3,) or source.normal.shape != (3,) or label not in {"floor", "wall", "ceiling"}:
            continue
        normalized_sources.append(
            SourcePlaneSurface(
                plane_id=source.plane_id,
                label=label,
                centroid=np.asarray(source.centroid, dtype=np.float64),
                normal=np.asarray(source.normal, dtype=np.float64),
                confidence=float(source.confidence),
                support_pixels=int(source.support_pixels),
            )
        )

    if initial_world_to_scene_col_major is not None:
        return _register_with_initial_transform(
            normalized_sources,
            target_surfaces,
            initial_matrix=_matrix_from_col_major(initial_world_to_scene_col_major),
            min_correspondences=int(min_correspondences),
        )

    targets_by_label: dict[str, list[StructuralSurface]] = {"floor": [], "wall": [], "ceiling": []}
    for surface in target_surfaces:
        label = normalize_label(surface.label)
        if label == "floor_or_ceiling":
            label = "floor" if float(surface.normal[1]) >= 0.0 else "ceiling"
        if label in targets_by_label:
            targets_by_label[label].append(surface)
    for rows in targets_by_label.values():
        rows.sort(key=lambda s: float(s.area), reverse=True)

    sources_by_label: dict[str, list[SourcePlaneSurface]] = {"floor": [], "wall": [], "ceiling": []}
    for source in normalized_sources:
        sources_by_label[source.label].append(source)
    for rows in sources_by_label.values():
        rows.sort(key=lambda s: (float(s.confidence), int(s.support_pixels)), reverse=True)

    correspondences: list[dict[str, Any]] = []
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    for label in ("floor", "wall", "ceiling"):
        source_rows = sources_by_label[label]
        target_rows = targets_by_label[label]
        for source, target in zip(source_rows, target_rows):
            source_points.append(np.asarray(source.centroid, dtype=np.float64))
            target_points.append(np.asarray(target.centroid, dtype=np.float64))
            correspondences.append(
                {
                    "label": label,
                    "source_plane_id": source.plane_id,
                    "target_surface_id": target.surface_id,
                    "source_centroid_m": [float(x) for x in source.centroid],
                    "target_centroid_scene": [float(x) for x in target.centroid],
                    "source_normal": [float(x) for x in source.normal],
                    "target_normal": [float(x) for x in target.normal],
                    "source_confidence": float(source.confidence),
                    "target_area": float(target.area),
                }
            )

    if len(correspondences) < int(min_correspondences):
        raise VirtualTwinRegistrationError(
            "virtual-twin registration needs at least "
            f"{int(min_correspondences)} plane/surface correspondences; got {len(correspondences)}"
        )

    matrix, scale, rmse = solve_similarity_transform(np.stack(source_points, axis=0), np.stack(target_points, axis=0))
    residuals = []
    angular = []
    for item, source_point, target_point in zip(correspondences, source_points, target_points):
        transformed = (matrix[:3, :3] @ source_point) + matrix[:3, 3]
        residual = float(np.linalg.norm(transformed - target_point))
        angle = _normal_angle_deg(
            np.asarray(item["source_normal"], dtype=np.float64),
            np.asarray(item["target_normal"], dtype=np.float64),
            matrix,
        )
        item["residual_scene_units"] = residual
        item["normal_angle_error_deg"] = angle
        residuals.append(residual)
        angular.append(angle)

    return {
        "status": "ok",
        "correspondence_count": len(correspondences),
        "world_to_menon_scene_col_major": [float(x) for x in matrix.flatten(order="F")],
        "matrix_row_major": [float(x) for x in matrix.reshape(-1)],
        "rotation_row_major": [float(x) for x in _rotation_from_similarity(matrix).reshape(-1)],
        "scene_per_m": float(scale),
        "s_obj_to_m": float(1.0 / scale) if scale > 1e-9 else 1.0,
        "position_rmse_scene_units": float(rmse),
        "median_residual_scene_units": float(np.median(residuals)) if residuals else 0.0,
        "p90_residual_scene_units": float(np.percentile(residuals, 90.0)) if residuals else 0.0,
        "median_normal_error_deg": float(np.median(angular)) if angular else 0.0,
        "p90_normal_error_deg": float(np.percentile(angular, 90.0)) if angular else 0.0,
        "correspondences": correspondences,
        "target_surface_summary": [surface_to_json(s) for s in target_surfaces[:64]],
    }


__all__ = [
    "SourcePlaneSurface",
    "VirtualTwinRegistrationError",
    "normalize_label",
    "register_planes_to_menon_surfaces",
    "resolve_backend_plane_label",
]
