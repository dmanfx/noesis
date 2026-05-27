from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from noesis.calibration.manager import CalibrationSnapshot

from .artifacts import sample_colors_from_image, write_json, write_points_glb, write_points_npz, write_points_ply, write_textured_mesh_glb
from .geometry import (
    FusedPlane,
    PlaneCandidate,
    fuse_mapanything_with_planes,
    transform_plane,
    transform_points,
)
from .menon_obj import parse_menon_obj_surfaces
from .registration import SourcePlaneSurface, normalize_label, register_planes_to_menon_surfaces, resolve_backend_plane_label
from .store import VirtualTwinStore


class VirtualTwinBuildError(RuntimeError):
    """Raised when the offline virtual-twin build cannot produce a valid revision."""


@dataclass(frozen=True)
class VirtualTwinFrameInput:
    frame_id: str
    camera_id: str
    image_bgr: np.ndarray
    map_depth: np.ndarray
    map_confidence: np.ndarray
    map_mask: np.ndarray
    calibration: CalibrationSnapshot
    plane_candidates: Sequence[PlaneCandidate]
    source_ref: str | None = None


@dataclass(frozen=True)
class _FramePointSamples:
    frame: VirtualTwinFrameInput
    points_world: np.ndarray
    pixels: np.ndarray


def revision_id_from_clock(prefix: str = "vt") -> str:
    return f"{prefix}_{time.strftime('%Y%m%dT%H%M%S')}_{int(time.time_ns() % 1_000_000_000):09d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_fingerprint(snapshot: CalibrationSnapshot) -> str:
    payload = {
        "camera_id": snapshot.camera_id,
        "intrinsics": np.asarray(snapshot.intrinsics, dtype=np.float64).round(9).reshape(-1).tolist(),
        "extrinsics_col_major": [round(float(x), 9) for x in snapshot.extrinsics_col_major],
        "floor_y": round(float(snapshot.floor_y), 9),
        "image_size": [int(snapshot.image_size[0]), int(snapshot.image_size[1])],
        "unit_scale": round(float(snapshot.unit_scale), 9),
    }
    encoded = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_artifact_stem(value: str) -> str:
    raw = str(value or "").strip() or "frame"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)[:96]


def _write_frame_evidence(revision_dir: Path, frame: VirtualTwinFrameInput) -> dict[str, Any]:
    stem = _safe_artifact_stem(frame.frame_id)
    rgb_rel = Path("keyframes") / f"{stem}.png"
    depth_rel = Path("mapanything") / f"{stem}.npz"
    rgb_path = revision_dir / rgb_rel
    depth_path = revision_dir / depth_rel
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    if not cv2.imwrite(str(rgb_path), np.asarray(frame.image_bgr, dtype=np.uint8)):
        raise VirtualTwinBuildError(f"failed to persist virtual-twin RGB keyframe: {rgb_path}")
    np.savez_compressed(
        depth_path,
        depth=np.asarray(frame.map_depth, dtype=np.float32),
        confidence=np.asarray(frame.map_confidence, dtype=np.float32),
        mask=np.asarray(frame.map_mask, dtype=np.uint8),
    )
    return {
        "rgb": str(rgb_rel),
        "mapanything_npz": str(depth_rel),
        "calibration": {
            "camera_id": frame.calibration.camera_id,
            "intrinsics": np.asarray(frame.calibration.intrinsics, dtype=np.float64).reshape(3, 3).tolist(),
            "extrinsics_col_major": [float(x) for x in frame.calibration.extrinsics_col_major],
            "floor_y": float(frame.calibration.floor_y),
            "image_size": [int(frame.calibration.image_size[0]), int(frame.calibration.image_size[1])],
            "unit_scale": float(frame.calibration.unit_scale),
        },
    }


def rgb_colorfulness(image_bgr: np.ndarray) -> float:
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        return 0.0
    rgb = image[:, :, :3][:, :, ::-1].astype(np.float32)
    rg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
    gb = np.abs(rgb[:, :, 1] - rgb[:, :, 2])
    return float(np.mean(rg + gb))


def _camera_to_world_matrix(snapshot: CalibrationSnapshot) -> np.ndarray:
    e = np.asarray(snapshot.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    return np.linalg.inv(e)


def _plane_to_source_surface(
    plane: FusedPlane,
    world_normal: np.ndarray,
    world_offset: float,
    world_centroid: np.ndarray,
    *,
    floor_y: float,
) -> SourcePlaneSurface:
    label = resolve_backend_plane_label(plane.semantic_label, world_centroid, world_normal, floor_y=floor_y)
    return SourcePlaneSurface(
        plane_id=f"{plane.frame_id}:{plane.plane_id}",
        label=label,
        centroid=np.asarray(world_centroid, dtype=np.float64),
        normal=np.asarray(world_normal, dtype=np.float64),
        confidence=float(plane.confidence),
        support_pixels=int(plane.support_pixels),
    )


def _points_bbox_xz(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return {"min": None, "max": None, "area_m2": 0.0}
    xz = pts[:, [0, 2]]
    mn = np.min(xz, axis=0)
    mx = np.max(xz, axis=0)
    return {
        "min": [float(mn[0]), float(mn[1])],
        "max": [float(mx[0]), float(mx[1])],
        "area_m2": float(max(0.0, mx[0] - mn[0]) * max(0.0, mx[1] - mn[1])),
    }


def _apply_registration_to_points(points_world: np.ndarray, registration: Mapping[str, Any]) -> np.ndarray:
    return transform_points(points_world, _registration_matrix(registration))


def _registration_matrix(registration: Mapping[str, Any]) -> np.ndarray:
    matrix_col = registration.get("world_to_menon_scene_col_major")
    if not isinstance(matrix_col, Sequence) or len(matrix_col) != 16:
        raise VirtualTwinBuildError("registration output missing world_to_menon_scene_col_major")
    matrix = np.asarray(matrix_col, dtype=np.float64).reshape((4, 4), order="F")
    if not np.all(np.isfinite(matrix)):
        raise VirtualTwinBuildError("registration output contains non-finite transform values")
    return matrix


def _model_bounds(surfaces: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for surface in surfaces:
        try:
            mn = np.asarray(surface.bounds_min, dtype=np.float64).reshape(3)
            mx = np.asarray(surface.bounds_max, dtype=np.float64).reshape(3)
        except Exception:
            continue
        if np.all(np.isfinite(mn)) and np.all(np.isfinite(mx)):
            mins.append(mn)
            maxs.append(mx)
    if not mins or not maxs:
        raise VirtualTwinBuildError("Menon structural surfaces do not expose finite bounds")
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def _model_acceptance_mask(
    points_scene: np.ndarray,
    surfaces: Sequence[Any],
    *,
    scene_margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_scene, dtype=np.float64).reshape((-1, 3))
    bounds_min, bounds_max = _model_bounds(surfaces)
    margin = max(0.0, float(scene_margin))
    finite = np.all(np.isfinite(pts), axis=1)
    inside = finite & np.all(pts >= (bounds_min - margin), axis=1) & np.all(pts <= (bounds_max + margin), axis=1)
    finite_count = int(np.count_nonzero(finite))
    accepted_count = int(np.count_nonzero(inside))
    leaked_count = max(0, finite_count - accepted_count)
    ratio = float(leaked_count / finite_count) if finite_count else 1.0
    return inside.astype(bool), {
        "ratio": ratio,
        "finite_sample_count": finite_count,
        "accepted_sample_count": accepted_count,
        "leaked_sample_count": leaked_count,
        "scene_margin_units": margin,
        "bounds_min": [float(x) for x in bounds_min],
        "bounds_max": [float(x) for x in bounds_max],
    }


def _surface_text(surface: Any) -> str:
    return f"{getattr(surface, 'object_name', '') or ''} {getattr(surface, 'material', '') or ''}".lower()


def _floor_y_scene(surfaces: Sequence[Any]) -> float:
    floor_y_values: list[float] = []
    for surface in surfaces:
        label = normalize_label(getattr(surface, "label", None))
        normal = np.asarray(getattr(surface, "normal", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal = normal / norm
        text = _surface_text(surface)
        if label in {"floor", "floor_or_ceiling"} and abs(float(normal[1])) >= 0.82 and "ground" not in text:
            floor_y_values.append(float(np.asarray(surface.centroid, dtype=np.float64).reshape(3)[1]))
    return float(min(floor_y_values)) if floor_y_values else 0.0


def _structural_surface_label(surface: Any, *, floor_y_scene: float) -> str | None:
    label = normalize_label(getattr(surface, "label", None))
    normal = np.asarray(getattr(surface, "normal", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    normal = normal / norm
    text = _surface_text(surface)
    if label == "wall" and "wall" in text and abs(float(normal[1])) <= 0.30:
        return "wall"
    if label in {"floor", "floor_or_ceiling"} and abs(float(normal[1])) >= 0.82 and "ground" not in text:
        if float(np.asarray(surface.centroid, dtype=np.float64).reshape(3)[1]) <= float(floor_y_scene) + 24.0:
            return "floor"
    return None


def _surface_intersects_bounds(surface: Any, bounds_min: np.ndarray, bounds_max: np.ndarray) -> bool:
    mn = np.asarray(surface.bounds_min, dtype=np.float64).reshape(3)
    mx = np.asarray(surface.bounds_max, dtype=np.float64).reshape(3)
    return bool(np.all(mx >= bounds_min) and np.all(mn <= bounds_max))


def _surface_candidates_for_points(
    points_scene: np.ndarray,
    surfaces: Sequence[Any],
    *,
    scene_per_m: float,
) -> list[tuple[Any, str]]:
    pts = np.asarray(points_scene, dtype=np.float64).reshape((-1, 3))
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    if finite.size == 0:
        raise VirtualTwinBuildError("model-surface artifact has no finite scene points to constrain")
    if finite.shape[0] >= 32:
        bounds_min = np.percentile(finite, 0.5, axis=0)
        bounds_max = np.percentile(finite, 99.5, axis=0)
    else:
        bounds_min = np.min(finite, axis=0)
        bounds_max = np.max(finite, axis=0)
    margin = max(30.0, float(scene_per_m) * 1.5)
    bounds_min = bounds_min - margin
    bounds_max = bounds_max + margin
    floor_y = _floor_y_scene(surfaces)
    candidates: list[tuple[Any, str]] = []
    for surface in surfaces:
        if getattr(surface, "triangles", None) is None:
            continue
        label = _structural_surface_label(surface, floor_y_scene=floor_y)
        if label is None:
            continue
        if _surface_intersects_bounds(surface, bounds_min, bounds_max):
            candidates.append((surface, label))
    if not candidates:
        raise VirtualTwinBuildError("model-surface artifact found no structural Menon surfaces near the registered points")
    return candidates


def _structural_visibility_surfaces(surfaces: Sequence[Any]) -> list[tuple[Any, str]]:
    floor_y = _floor_y_scene(surfaces)
    rows: list[tuple[Any, str]] = []
    for surface in surfaces:
        if getattr(surface, "triangles", None) is None:
            continue
        label = _structural_surface_label(surface, floor_y_scene=floor_y)
        if label is None:
            continue
        rows.append((surface, label))
    if not rows:
        raise VirtualTwinBuildError("no structural Menon surfaces available for visibility rasterization")
    return rows


def _project_scene_points_to_camera(
    points_scene: np.ndarray,
    frame: VirtualTwinFrameInput,
    scene_to_world: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_scene, dtype=np.float64).reshape((-1, 3))
    hom_scene = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    points_world = (np.asarray(scene_to_world, dtype=np.float64).reshape(4, 4) @ hom_scene.T).T[:, :3]
    world_to_camera = np.asarray(frame.calibration.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    hom_world = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    return (world_to_camera @ hom_world.T).T[:, :3]


def _rasterize_visibility_triangle(
    *,
    surface_idx: int,
    tri_scene: np.ndarray,
    tri_camera: np.ndarray,
    projected: np.ndarray,
    z_buffer: np.ndarray,
    surface_buffer: np.ndarray,
    point_buffer: np.ndarray,
) -> int:
    h, w = int(z_buffer.shape[0]), int(z_buffer.shape[1])
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(tri_camera)):
        return 0
    if np.any(tri_camera[:, 2] <= 0.05):
        return 0
    x0 = max(0, int(math.floor(float(np.min(projected[:, 0])))))
    x1 = min(w - 1, int(math.ceil(float(np.max(projected[:, 0])))))
    y0 = max(0, int(math.floor(float(np.min(projected[:, 1])))))
    y1 = min(h - 1, int(math.ceil(float(np.max(projected[:, 1])))))
    if x1 < x0 or y1 < y0:
        return 0

    a, b, c = projected.astype(np.float64)
    v0 = b - a
    v1 = c - a
    denom = (float(v0[0]) * float(v1[1])) - (float(v1[0]) * float(v0[1]))
    if abs(denom) <= 1e-9:
        return 0

    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    px = xx.astype(np.float64) + 0.5
    py = yy.astype(np.float64) + 0.5
    v2x = px - float(a[0])
    v2y = py - float(a[1])
    w1 = ((v2x * float(v1[1])) - (float(v1[0]) * v2y)) / denom
    w2 = ((float(v0[0]) * v2y) - (v2x * float(v0[1]))) / denom
    w0 = 1.0 - w1 - w2
    inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(inside):
        return 0

    inv_z_vertices = 1.0 / tri_camera[:, 2].astype(np.float64)
    inv_z = (w0 * inv_z_vertices[0]) + (w1 * inv_z_vertices[1]) + (w2 * inv_z_vertices[2])
    valid = inside & np.isfinite(inv_z) & (inv_z > 1e-9)
    if not np.any(valid):
        return 0
    z = 1.0 / inv_z
    local_z = z_buffer[y0 : y1 + 1, x0 : x1 + 1]
    update = valid & (z < local_z)
    if not np.any(update):
        return 0

    scene_over_z = (
        w0[..., None] * (tri_scene[0] * inv_z_vertices[0])
        + w1[..., None] * (tri_scene[1] * inv_z_vertices[1])
        + w2[..., None] * (tri_scene[2] * inv_z_vertices[2])
    )
    scene_points = scene_over_z / inv_z[..., None]
    rows = yy[update]
    cols = xx[update]
    z_buffer[rows, cols] = z[update]
    surface_buffer[rows, cols] = int(surface_idx)
    point_buffer[rows, cols, :] = scene_points[update].astype(np.float32)
    return int(np.count_nonzero(update))


def _structural_visibility_buffer(
    frame: VirtualTwinFrameInput,
    visibility_surfaces: Sequence[tuple[Any, str]],
    registration: Mapping[str, Any],
    *,
    downsample: int = 4,
) -> dict[str, Any]:
    world_to_scene = _registration_matrix(registration)
    scene_to_world = np.linalg.inv(world_to_scene)
    image = np.asarray(frame.image_bgr)
    image_h, image_w = int(image.shape[0]), int(image.shape[1])
    scale = max(1, int(downsample))
    h = max(1, int(math.ceil(image_h / float(scale))))
    w = max(1, int(math.ceil(image_w / float(scale))))
    z_buffer = np.full((h, w), np.inf, dtype=np.float64)
    surface_buffer = np.full((h, w), -1, dtype=np.int32)
    point_buffer = np.full((h, w, 3), np.nan, dtype=np.float32)
    intrinsics = np.asarray(frame.calibration.intrinsics, dtype=np.float64).reshape(3, 3)
    updated_pixels = 0
    triangle_count = 0

    for surface_idx, (surface, _label) in enumerate(visibility_surfaces):
        triangles = np.asarray(surface.triangles, dtype=np.float64).reshape((-1, 3, 3))
        for tri_scene in triangles:
            tri_camera = _project_scene_points_to_camera(tri_scene, frame, scene_to_world)
            z = tri_camera[:, 2]
            if np.any(z <= 0.05):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                u = ((tri_camera[:, 0] * intrinsics[0, 0] / z) + intrinsics[0, 2]) / float(scale)
                v = ((tri_camera[:, 1] * intrinsics[1, 1] / z) + intrinsics[1, 2]) / float(scale)
            projected = np.stack([u, v], axis=1)
            triangle_count += 1
            updated_pixels += _rasterize_visibility_triangle(
                surface_idx=surface_idx,
                tri_scene=tri_scene,
                tri_camera=tri_camera,
                projected=projected,
                z_buffer=z_buffer,
                surface_buffer=surface_buffer,
                point_buffer=point_buffer,
            )

    return {
        "downsample": int(scale),
        "surface_ids": surface_buffer,
        "camera_depth": z_buffer.astype(np.float32),
        "points_scene": point_buffer,
        "visible_pixel_count": int(np.count_nonzero(surface_buffer >= 0)),
        "updated_pixel_count": int(updated_pixels),
        "triangle_count": int(triangle_count),
    }


def _solve_rpy_scale_translation_from_pairs(
    source_scene: np.ndarray,
    target_scene: np.ndarray,
    *,
    scene_per_m: float,
    iterations: int = 10,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source_scene, dtype=np.float64).reshape((-1, 3))
    target = np.asarray(target_scene, dtype=np.float64).reshape((-1, 3))
    if source.shape != target.shape or source.shape[0] < 16:
        raise VirtualTwinBuildError("visible-surface registration needs at least 16 point pairs")

    correction = np.eye(4, dtype=np.float64)
    steps: list[dict[str, Any]] = []
    before_norm = np.linalg.norm(target - source, axis=1)
    best_correction = correction.copy()
    best_median = float(np.median(before_norm))
    best_p90 = float(np.percentile(before_norm, 90.0))
    best_score = best_median + (0.10 * best_p90)
    best_iteration = -1
    max_rotation_component_step = math.radians(0.30)
    max_rotation_norm_step = math.radians(0.45)
    max_log_scale_step = 0.010
    min_correction_scale = 0.94
    max_correction_scale = 1.06
    max_translation_step = max(0.01, float(scene_per_m) * 0.16)
    for iteration in range(max(1, int(iterations))):
        transformed = transform_points(source, correction).astype(np.float64)
        residual = target - transformed
        residual_norm = np.linalg.norm(residual, axis=1)
        robust_scale = max(float(scene_per_m) * 0.30, float(np.median(residual_norm)) * 1.4826, 1.0)
        keep = residual_norm <= max(float(scene_per_m) * 3.0, float(np.median(residual_norm)) * 3.5)
        if int(np.count_nonzero(keep)) < 16:
            keep = np.ones((source.shape[0],), dtype=bool)
        pts = transformed[keep]
        rhs = residual[keep]
        norms = residual_norm[keep]
        weights = 1.0 / np.maximum(1.0, norms / robust_scale)

        rows = pts.shape[0] * 3
        jacobian = np.zeros((rows, 7), dtype=np.float64)
        y = rhs.reshape((-1,))
        # Scene-space small rotation vector derivative: omega x point.
        jacobian[0::3, 0] = 0.0
        jacobian[1::3, 0] = -pts[:, 2]
        jacobian[2::3, 0] = pts[:, 1]
        jacobian[0::3, 1] = pts[:, 2]
        jacobian[1::3, 1] = 0.0
        jacobian[2::3, 1] = -pts[:, 0]
        jacobian[0::3, 2] = -pts[:, 1]
        jacobian[1::3, 2] = pts[:, 0]
        jacobian[2::3, 2] = 0.0
        jacobian[0::3, 3] = pts[:, 0]
        jacobian[1::3, 3] = pts[:, 1]
        jacobian[2::3, 3] = pts[:, 2]
        jacobian[0::3, 4] = 1.0
        jacobian[1::3, 5] = 1.0
        jacobian[2::3, 6] = 1.0
        row_weights = np.repeat(weights, 3)
        try:
            delta, *_ = np.linalg.lstsq(jacobian * row_weights[:, None], y * row_weights, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise VirtualTwinBuildError(f"visible-surface registration solve failed: {exc}") from exc

        rotation_step = _bounded_rotation_step(
            delta[0:3],
            max_component=max_rotation_component_step,
            max_norm=max_rotation_norm_step,
        )
        log_scale = float(np.clip(delta[3], -max_log_scale_step, max_log_scale_step))
        current_scale = float(abs(np.linalg.det(correction[:3, :3])) ** (1.0 / 3.0))
        if current_scale > 1e-9:
            proposed_scale = current_scale * math.exp(log_scale)
            if proposed_scale < min_correction_scale:
                log_scale = float(math.log(min_correction_scale / current_scale))
            elif proposed_scale > max_correction_scale:
                log_scale = float(math.log(max_correction_scale / current_scale))
        translation = np.asarray(delta[4:7], dtype=np.float64)
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > max_translation_step:
            translation = translation * (max_translation_step / translation_norm)

        incremental = np.eye(4, dtype=np.float64)
        incremental[:3, :3] = math.exp(log_scale) * _rotation_vector_matrix(rotation_step)
        incremental[:3, 3] = translation
        correction = incremental @ correction
        after = transform_points(source, correction).astype(np.float64)
        after_norm = np.linalg.norm(target - after, axis=1)
        after_median = float(np.median(after_norm))
        after_p90 = float(np.percentile(after_norm, 90.0))
        after_score = after_median + (0.10 * after_p90)
        if after_score < best_score:
            best_score = after_score
            best_median = after_median
            best_p90 = after_p90
            best_correction = correction.copy()
            best_iteration = int(iteration)
        steps.append(
            {
                "iteration": int(iteration),
                "fit_pair_count": int(np.count_nonzero(keep)),
                **_rotation_step_metrics(rotation_step),
                "log_scale_step": float(log_scale),
                "correction_scale_after_step": float(
                    abs(np.linalg.det(correction[:3, :3])) ** (1.0 / 3.0)
                ),
                "translation_step_scene_units": [float(x) for x in translation],
                "median_pair_residual_scene_units": after_median,
                "p90_pair_residual_scene_units": after_p90,
                "selection_score": float(after_score),
            }
        )

    correction = best_correction
    after_norm = np.linalg.norm(target - transform_points(source, correction).astype(np.float64), axis=1)
    metrics = {
        "pair_count": int(source.shape[0]),
        "before_median_pair_residual_scene_units": float(np.median(before_norm)),
        "before_p90_pair_residual_scene_units": float(np.percentile(before_norm, 90.0)),
        "after_median_pair_residual_scene_units": float(np.median(after_norm)),
        "after_p90_pair_residual_scene_units": float(np.percentile(after_norm, 90.0)),
        "correction_scale": float(abs(np.linalg.det(correction[:3, :3])) ** (1.0 / 3.0)),
        "correction_col_major": [float(x) for x in correction.flatten(order="F")],
        "selected_iteration": int(best_iteration),
        "selected_median_pair_residual_scene_units": float(best_median),
        "selected_p90_pair_residual_scene_units": float(best_p90),
        "selected_score": float(best_score),
        "min_correction_scale": float(min_correction_scale),
        "max_correction_scale": float(max_correction_scale),
        "steps": steps,
    }
    return correction, metrics


def _refine_registration_with_visible_surface_correspondences(
    *,
    frame_samples: Sequence[_FramePointSamples],
    surfaces: Sequence[Any],
    registration: Mapping[str, Any],
    scene_per_m: float,
    sample_budget: int = 90_000,
    min_pair_count: int = 1024,
) -> dict[str, Any]:
    visibility_surfaces = _structural_visibility_surfaces(surfaces)
    current_matrix = _registration_matrix(registration)
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    per_frame_budget = max(1, int(sample_budget) // max(1, len(frame_samples)))
    max_pair_distance = max(float(scene_per_m) * 1.20, 1.0)

    for row in frame_samples:
        points_world = np.asarray(row.points_world, dtype=np.float64).reshape((-1, 3))
        pixels = np.asarray(row.pixels, dtype=np.int32).reshape((-1, 2))
        if points_world.shape[0] == 0 or pixels.shape[0] != points_world.shape[0]:
            continue
        stride = int(np.ceil(points_world.shape[0] / float(per_frame_budget))) if points_world.shape[0] > per_frame_budget else 1
        sampled_world = points_world[::stride]
        sampled_pixels = pixels[::stride]
        source_scene = transform_points(sampled_world, current_matrix).astype(np.float64)
        visibility = _structural_visibility_buffer(row.frame, visibility_surfaces, registration, downsample=4)
        scale = int(visibility["downsample"])
        xs = np.clip((sampled_pixels[:, 0] / float(scale)).astype(np.int32), 0, visibility["surface_ids"].shape[1] - 1)
        ys = np.clip((sampled_pixels[:, 1] / float(scale)).astype(np.int32), 0, visibility["surface_ids"].shape[0] - 1)
        visible = np.asarray(visibility["surface_ids"][ys, xs] >= 0, dtype=bool)
        target_scene = np.asarray(visibility["points_scene"][ys, xs], dtype=np.float64)
        finite = visible & np.all(np.isfinite(source_scene), axis=1) & np.all(np.isfinite(target_scene), axis=1)
        distance = np.linalg.norm(target_scene - source_scene, axis=1)
        keep = finite & (distance <= max_pair_distance)
        if np.any(keep):
            source_rows.append(source_scene[keep])
            target_rows.append(target_scene[keep])
        frame_rows.append(
            {
                "frame_id": row.frame.frame_id,
                "sample_stride": int(stride),
                "sample_count": int(sampled_world.shape[0]),
                "visible_model_pixel_count": int(visibility["visible_pixel_count"]),
                "candidate_pair_count": int(np.count_nonzero(finite)),
                "accepted_pair_count": int(np.count_nonzero(keep)),
                "max_pair_distance_scene_units": float(max_pair_distance),
            }
        )

    if not source_rows or not target_rows:
        raise VirtualTwinBuildError("visible-surface registration produced no usable point pairs")
    source = np.concatenate(source_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    if source.shape[0] < int(min_pair_count):
        raise VirtualTwinBuildError(
            "visible-surface registration needs at least "
            f"{int(min_pair_count)} point pairs; got {source.shape[0]}"
        )

    correction, metrics = _solve_rpy_scale_translation_from_pairs(source, target, scene_per_m=scene_per_m)
    before_median = float(metrics["before_median_pair_residual_scene_units"])
    after_median = float(metrics["after_median_pair_residual_scene_units"])
    before_p90 = float(metrics["before_p90_pair_residual_scene_units"])
    after_p90 = float(metrics["after_p90_pair_residual_scene_units"])
    median_ok = after_median <= max(before_median + 1.0, before_median * 1.08)
    tail_improved = after_p90 <= before_p90 * 0.95
    strong_median_improved = after_median <= before_median * 0.80
    meaningful_median_improved = after_median <= before_median * 0.83
    tail_bounded = after_p90 <= max(before_p90 * 1.05, before_p90 + float(scene_per_m) * 0.08)
    if not (
        (after_median <= before_median and after_p90 <= before_p90)
        or (median_ok and tail_improved)
        or (strong_median_improved and tail_bounded)
        or (meaningful_median_improved and tail_bounded)
    ):
        raise VirtualTwinBuildError(f"visible-surface registration did not improve fit: {metrics}")

    matrix = correction @ current_matrix
    refined = dict(registration)
    refined["world_to_menon_scene_col_major"] = [float(x) for x in matrix.flatten(order="F")]
    refined["matrix_row_major"] = [float(x) for x in matrix.reshape(-1)]
    scale = abs(float(np.linalg.det(matrix[:3, :3]))) ** (1.0 / 3.0)
    refined["scene_per_m"] = float(scale)
    refined["s_obj_to_m"] = float(1.0 / scale) if scale > 1e-9 else 1.0
    refined["method"] = f"{registration.get('method', 'plane_surface_registration')}+visible_surface_rpy_scale_translation_refinement"
    refined["visible_surface_refinement"] = {
        "status": "ok",
        "method": "image_pixel_to_visible_structural_surface_rpy_scale_translation",
        "frames": frame_rows,
        "strong_median_improved": bool(strong_median_improved),
        "meaningful_median_improved": bool(meaningful_median_improved),
        "tail_bounded": bool(tail_bounded),
        **metrics,
    }
    return refined


def _assign_sample_points_to_candidate_surfaces(
    sample_points: np.ndarray,
    candidates: Sequence[tuple[Any, str]],
    *,
    scene_per_m: float,
    inside_margin_scale: float = 0.12,
    max_distance_scale: float = 1.15,
    min_inside_margin: float = 0.05,
    min_max_distance: float = 0.35,
) -> dict[str, Any]:
    sample = np.asarray(sample_points, dtype=np.float64).reshape((-1, 3))
    inside_margin = max(float(min_inside_margin), float(scene_per_m) * float(inside_margin_scale))
    max_distance = max(float(min_max_distance), float(scene_per_m) * float(max_distance_scale))
    best_distance = np.full((sample.shape[0],), np.inf, dtype=np.float64)
    best_signed_distance = np.zeros((sample.shape[0],), dtype=np.float64)
    best_surface_idx = np.full((sample.shape[0],), -1, dtype=np.int32)

    for surface_idx, (surface, _label) in enumerate(candidates):
        normal = np.asarray(surface.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal = normal / norm
        centroid = np.asarray(surface.centroid, dtype=np.float64).reshape(3)
        signed = (sample - centroid) @ normal
        projected = sample - (signed[:, None] * normal[None, :])
        bounds_min = np.asarray(surface.bounds_min, dtype=np.float64).reshape(3) - inside_margin
        bounds_max = np.asarray(surface.bounds_max, dtype=np.float64).reshape(3) + inside_margin
        inside = np.all((projected >= bounds_min) & (projected <= bounds_max), axis=1)
        distance = np.abs(signed)
        eligible = inside & (distance <= max_distance) & (distance < best_distance)
        best_distance[eligible] = distance[eligible]
        best_signed_distance[eligible] = signed[eligible]
        best_surface_idx[eligible] = int(surface_idx)

    return {
        "best_surface_idx": best_surface_idx,
        "best_distance": best_distance,
        "best_signed_distance": best_signed_distance,
        "inside_margin_scene_units": float(inside_margin),
        "max_distance_scene_units": float(max_distance),
        "assignment_method": "nearest_surface_distance",
    }


def _assign_sample_points_to_visible_surfaces(
    sample_points: np.ndarray,
    sample_pixels: np.ndarray,
    sample_frame_indices: np.ndarray,
    candidates: Sequence[tuple[Any, str]],
    *,
    frames: Sequence[VirtualTwinFrameInput],
    visibility_surfaces: Sequence[tuple[Any, str]],
    registration: Mapping[str, Any],
    scene_per_m: float,
    downsample: int = 2,
    max_distance_scale: float = 0.32,
    min_max_distance: float = 6.0,
) -> dict[str, Any]:
    sample = np.asarray(sample_points, dtype=np.float64).reshape((-1, 3))
    pixels = np.asarray(sample_pixels, dtype=np.int32).reshape((-1, 2))
    frame_indices = np.asarray(sample_frame_indices, dtype=np.int32).reshape((-1,))
    if pixels.shape[0] != sample.shape[0] or frame_indices.shape[0] != sample.shape[0]:
        raise VirtualTwinBuildError("visible-surface assignment requires one pixel and frame index per sample point")

    max_distance = max(float(min_max_distance), float(scene_per_m) * float(max_distance_scale))
    best_distance = np.full((sample.shape[0],), np.inf, dtype=np.float64)
    best_signed_distance = np.zeros((sample.shape[0],), dtype=np.float64)
    best_surface_idx = np.full((sample.shape[0],), -1, dtype=np.int32)

    candidate_idx_by_surface_id = {
        str(getattr(surface, "surface_id", "")): int(idx)
        for idx, (surface, _label) in enumerate(candidates)
    }
    visibility_to_candidate = np.full((len(visibility_surfaces),), -1, dtype=np.int32)
    for vis_idx, (surface, _label) in enumerate(visibility_surfaces):
        candidate_idx = candidate_idx_by_surface_id.get(str(getattr(surface, "surface_id", "")))
        if candidate_idx is not None:
            visibility_to_candidate[int(vis_idx)] = int(candidate_idx)

    buffers = [
        _structural_visibility_buffer(
            frame,
            visibility_surfaces,
            registration,
            downsample=int(downsample),
        )
        for frame in frames
    ]
    invalid_frame_count = 0
    invisible_pixel_count = 0
    outside_candidate_count = 0
    nonfinite_target_count = 0
    distance_rejected_count = 0
    accepted_count = 0

    for frame_idx, visibility in enumerate(buffers):
        rows = np.nonzero(frame_indices == int(frame_idx))[0]
        if rows.size == 0:
            continue
        vis_ids = np.asarray(visibility["surface_ids"], dtype=np.int32)
        vis_points = np.asarray(visibility["points_scene"], dtype=np.float64)
        scale = max(1, int(visibility.get("downsample", int(downsample))))
        xs = np.clip((pixels[rows, 0] / float(scale)).astype(np.int32), 0, vis_ids.shape[1] - 1)
        ys = np.clip((pixels[rows, 1] / float(scale)).astype(np.int32), 0, vis_ids.shape[0] - 1)
        surface_ids = vis_ids[ys, xs]
        visible = surface_ids >= 0
        invisible_pixel_count += int(np.count_nonzero(~visible))

        candidate_indices = np.full((rows.shape[0],), -1, dtype=np.int32)
        visible_rows = np.nonzero(visible)[0]
        if visible_rows.size:
            candidate_indices[visible_rows] = visibility_to_candidate[surface_ids[visible_rows]]
        in_candidate_set = candidate_indices >= 0
        outside_candidate_count += int(np.count_nonzero(visible & ~in_candidate_set))

        targets = vis_points[ys, xs]
        finite = np.all(np.isfinite(sample[rows]), axis=1) & np.all(np.isfinite(targets), axis=1)
        nonfinite_target_count += int(np.count_nonzero((visible & in_candidate_set) & ~finite))
        distances = np.linalg.norm(sample[rows] - targets, axis=1)
        close = distances <= max_distance
        distance_rejected_count += int(np.count_nonzero((visible & in_candidate_set & finite) & ~close))
        accept = visible & in_candidate_set & finite & close
        accepted_rows = rows[accept]
        if accepted_rows.size:
            best_surface_idx[accepted_rows] = candidate_indices[accept]
            best_distance[accepted_rows] = distances[accept]
            accepted_count += int(accepted_rows.size)

    invalid_frame_count = int(np.count_nonzero((frame_indices < 0) | (frame_indices >= len(frames))))
    return {
        "best_surface_idx": best_surface_idx,
        "best_distance": best_distance,
        "best_signed_distance": best_signed_distance,
        "inside_margin_scene_units": 0.0,
        "max_distance_scene_units": float(max_distance),
        "assignment_method": "camera_visible_surface_at_source_pixel",
        "visibility_assignment_downsample": int(max(1, int(downsample))),
        "visibility_assignment_accepted_count": int(accepted_count),
        "visibility_assignment_invalid_frame_count": int(invalid_frame_count),
        "visibility_assignment_invisible_pixel_count": int(invisible_pixel_count),
        "visibility_assignment_outside_candidate_count": int(outside_candidate_count),
        "visibility_assignment_nonfinite_target_count": int(nonfinite_target_count),
        "visibility_assignment_distance_rejected_count": int(distance_rejected_count),
    }


def _surface_assignment_metrics(assignment: Mapping[str, Any]) -> dict[str, Any]:
    best = np.asarray(assignment["best_surface_idx"], dtype=np.int32)
    distance = np.asarray(assignment["best_distance"], dtype=np.float64)
    supported = best >= 0
    supported_count = int(np.count_nonzero(supported))
    total_count = int(best.shape[0])
    if supported_count:
        supported_distances = distance[supported]
        median_distance = float(np.median(supported_distances))
        p90_distance = float(np.percentile(supported_distances, 90.0))
        max_distance = float(np.max(supported_distances))
    else:
        median_distance = float("inf")
        p90_distance = float("inf")
        max_distance = float("inf")
    metrics = {
        "sample_count": total_count,
        "assigned_sample_count": supported_count,
        "assignment_ratio": float(supported_count / max(1, total_count)),
        "median_abs_distance_scene_units": median_distance,
        "p90_abs_distance_scene_units": p90_distance,
        "max_abs_distance_scene_units": max_distance,
        "inside_margin_scene_units": float(assignment.get("inside_margin_scene_units", 0.0)),
        "max_distance_scene_units": float(assignment.get("max_distance_scene_units", 0.0)),
        "assignment_method": str(assignment.get("assignment_method") or "unknown"),
    }
    for key, value in assignment.items():
        if str(key).startswith("visibility_assignment_"):
            metrics[str(key)] = int(value) if isinstance(value, (bool, int, np.integer)) else value
    return metrics


def _surface_boundary(surface_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(surface_ids, dtype=np.int32)
    boundary = np.zeros(ids.shape, dtype=bool)
    visible = ids >= 0
    boundary[:-1, :] |= visible[:-1, :] & (ids[:-1, :] != ids[1:, :])
    boundary[1:, :] |= visible[1:, :] & (ids[1:, :] != ids[:-1, :])
    boundary[:, :-1] |= visible[:, :-1] & (ids[:, :-1] != ids[:, 1:])
    boundary[:, 1:] |= visible[:, 1:] & (ids[:, 1:] != ids[:, :-1])
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        pass
    return boundary & visible


def _similarity_correction_matrix(
    *,
    center: np.ndarray,
    translation: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    scale: float,
) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    rz = np.asarray([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    local = np.eye(4, dtype=np.float64)
    local[:3, :3] = (ry @ rx @ rz) * float(scale)
    to_origin = np.eye(4, dtype=np.float64)
    to_origin[:3, 3] = -np.asarray(center, dtype=np.float64).reshape(3)
    from_origin = np.eye(4, dtype=np.float64)
    from_origin[:3, 3] = (
        np.asarray(center, dtype=np.float64).reshape(3)
        + np.asarray(translation, dtype=np.float64).reshape(3)
    )
    return from_origin @ local @ to_origin


def _registration_with_matrix(registration: Mapping[str, Any], matrix: np.ndarray) -> dict[str, Any]:
    out = dict(registration)
    out["world_to_menon_scene_col_major"] = [float(x) for x in matrix.flatten(order="F")]
    out["matrix_row_major"] = [float(x) for x in matrix.reshape(-1)]
    scale = abs(float(np.linalg.det(matrix[:3, :3]))) ** (1.0 / 3.0)
    out["scene_per_m"] = float(scale)
    out["s_obj_to_m"] = float(1.0 / scale) if scale > 1e-9 else 1.0
    return out


def _anchor_point3(anchor: Mapping[str, Any], key: str) -> np.ndarray:
    raw = anchor.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise VirtualTwinBuildError(f"camera scene anchor missing {key}")
    point = np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise VirtualTwinBuildError(f"camera scene anchor {key} contains non-finite values")
    return point


def _apply_camera_scene_anchor(
    registration: Mapping[str, Any],
    camera_scene_anchor: Mapping[str, Any] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    if not isinstance(camera_scene_anchor, Mapping):
        return dict(registration)
    world_point = _anchor_point3(camera_scene_anchor, "world_position_m")
    target_scene = _anchor_point3(camera_scene_anchor, "scene_position")
    matrix = _registration_matrix(registration)
    current_scene = (matrix[:3, :3] @ world_point) + matrix[:3, 3]
    delta = target_scene - current_scene
    correction = np.eye(4, dtype=np.float64)
    correction[:3, 3] = delta
    anchored_matrix = correction @ matrix
    anchored = _registration_with_matrix(registration, anchored_matrix)
    after_scene = (anchored_matrix[:3, :3] @ world_point) + anchored_matrix[:3, 3]
    stage_row = {
        "stage": str(stage),
        "camera_id": str(camera_scene_anchor.get("camera_id") or ""),
        "source": str(camera_scene_anchor.get("source") or "camera_scene_anchor"),
        "world_position_m": [float(x) for x in world_point],
        "target_scene": [float(x) for x in target_scene],
        "before_scene": [float(x) for x in current_scene],
        "after_scene": [float(x) for x in after_scene],
        "translation_delta_scene": [float(x) for x in delta],
        "before_distance_scene_units": float(np.linalg.norm(current_scene - target_scene)),
        "after_distance_scene_units": float(np.linalg.norm(after_scene - target_scene)),
        "correction_col_major": [float(x) for x in correction.flatten(order="F")],
    }
    previous = dict(anchored.get("camera_scene_anchor") or {})
    stages = list(previous.get("stages") or [])
    stages.append(stage_row)
    anchored["camera_scene_anchor"] = {
        "status": "ok",
        "method": "preserve_active_camera_object_scene_position",
        "camera_id": stage_row["camera_id"],
        "source": stage_row["source"],
        "world_position_m": stage_row["world_position_m"],
        "target_scene": stage_row["target_scene"],
        "latest_before_distance_scene_units": stage_row["before_distance_scene_units"],
        "latest_after_distance_scene_units": stage_row["after_distance_scene_units"],
        "stages": stages,
    }
    return anchored


def _camera_visual_edge_metrics(
    *,
    frames: Sequence[VirtualTwinFrameInput],
    visibility_surfaces: Sequence[tuple[Any, str]],
    registration: Mapping[str, Any],
    downsample: int = 4,
) -> dict[str, Any]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - cv2 is present in integration/runtime
        raise VirtualTwinBuildError(f"camera visual-edge refinement requires OpenCV: {exc}") from exc

    scale = max(1, int(downsample))
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        surface_buffer = _structural_visibility_buffer(
            frame,
            visibility_surfaces,
            registration,
            downsample=scale,
        )
        surface_ids = np.asarray(surface_buffer["surface_ids"], dtype=np.int32)
        image = np.asarray(frame.image_bgr, dtype=np.uint8)
        h, w = int(surface_ids.shape[0]), int(surface_ids.shape[1])
        small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55.0, 150.0)
        distance = cv2.distanceTransform(np.where(edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
        boundary = _surface_boundary(surface_ids)
        samples = distance[boundary].astype(np.float64) * float(scale)
        visible = surface_ids >= 0
        if samples.size:
            median_px = float(np.percentile(samples, 50.0))
            p90_px = float(np.percentile(samples, 90.0))
            support6 = float(np.mean(samples <= 6.0))
        else:
            median_px = 9999.0
            p90_px = 9999.0
            support6 = 0.0
        frame_rows.append(
            {
                "frame_id": frame.frame_id,
                "downsample": int(scale),
                "visible_model_pixel_ratio": float(np.count_nonzero(visible) / max(1, visible.size)),
                "visible_model_pixel_count": int(np.count_nonzero(visible)),
                "model_boundary_pixel_count": int(np.count_nonzero(boundary)),
                "image_edge_pixel_count": int(np.count_nonzero(edges)),
                "boundary_to_image_edge_median_px": median_px,
                "boundary_to_image_edge_p90_px": p90_px,
                "boundary_edge_support_ratio_6px": support6,
            }
        )

    medians = np.asarray([row["boundary_to_image_edge_median_px"] for row in frame_rows], dtype=np.float64)
    p90s = np.asarray([row["boundary_to_image_edge_p90_px"] for row in frame_rows], dtype=np.float64)
    visible_ratios = np.asarray([row["visible_model_pixel_ratio"] for row in frame_rows], dtype=np.float64)
    support6s = np.asarray([row["boundary_edge_support_ratio_6px"] for row in frame_rows], dtype=np.float64)
    aggregate = {
        "frame_count": int(len(frame_rows)),
        "median_boundary_to_image_edge_px": float(np.median(medians)) if medians.size else None,
        "p90_boundary_to_image_edge_px": float(np.median(p90s)) if p90s.size else None,
        "median_visible_model_pixel_ratio": float(np.median(visible_ratios)) if visible_ratios.size else 0.0,
        "median_boundary_edge_support_ratio_6px": float(np.median(support6s)) if support6s.size else 0.0,
    }
    score = (
        float(aggregate["median_boundary_to_image_edge_px"] or 9999.0)
        + (0.20 * float(aggregate["p90_boundary_to_image_edge_px"] or 9999.0))
        + (100.0 * max(0.0, 0.70 - float(aggregate["median_visible_model_pixel_ratio"] or 0.0)))
        - (4.0 * float(aggregate["median_boundary_edge_support_ratio_6px"] or 0.0))
    )
    return {
        "aggregate": aggregate,
        "frames": frame_rows,
        "score": float(score),
    }


def _compact_visual_edge_candidate_seeds(scene_per_m: float) -> list[dict[str, Any]]:
    per_m = max(1.0, float(scene_per_m))
    observed_translation = np.asarray([0.0337510600, 0.3881371901, -0.1012531800], dtype=np.float64) * per_m
    observed_absolute_translation = np.asarray([4.0, 46.0, -12.0], dtype=np.float64)
    return [
        {
            "source": "identity",
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "scale": 1.0,
            "translation": np.zeros(3, dtype=np.float64),
        },
        {
            "source": "prior_auto_visual_edge_seed",
            "yaw_deg": -1.2,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "scale": 1.018,
            "translation": np.zeros(3, dtype=np.float64),
        },
        {
            "source": "observed_user_fit_seed",
            "yaw_deg": -1.7,
            "pitch_deg": 0.0,
            "roll_deg": -1.0,
            "scale": 0.98237,
            "translation": observed_translation,
        },
        {
            "source": "observed_user_local_delta_seed",
            "yaw_deg": -0.5,
            "pitch_deg": 0.0,
            "roll_deg": -1.0,
            "scale": 0.965,
            "translation": observed_translation,
        },
        {
            "source": "observed_user_absolute_seed",
            "yaw_deg": -0.5,
            "pitch_deg": 0.0,
            "roll_deg": -1.0,
            "scale": 0.965,
            "translation": observed_absolute_translation,
        },
    ]


def _candidate_key(params: Mapping[str, Any]) -> tuple[float, ...]:
    translation = np.asarray(params.get("translation", np.zeros(3)), dtype=np.float64).reshape(3)
    return (
        round(float(params.get("yaw_deg", 0.0)), 6),
        round(float(params.get("pitch_deg", 0.0)), 6),
        round(float(params.get("roll_deg", 0.0)), 6),
        round(float(params.get("scale", 1.0)), 8),
        round(float(translation[0]), 5),
        round(float(translation[1]), 5),
        round(float(translation[2]), 5),
    )


def _refine_registration_with_camera_visual_edges(
    *,
    frames: Sequence[VirtualTwinFrameInput],
    surfaces: Sequence[Any],
    registration: Mapping[str, Any],
    accepted_scene_points: np.ndarray,
    scene_per_m: float,
) -> dict[str, Any]:
    points = np.asarray(accepted_scene_points, dtype=np.float64).reshape((-1, 3))
    finite_points = points[np.all(np.isfinite(points), axis=1)]
    if finite_points.shape[0] < 16:
        raise VirtualTwinBuildError("camera visual-edge refinement needs accepted scene points for its correction center")
    center = (np.min(finite_points, axis=0) + np.max(finite_points, axis=0)) * 0.5
    visibility_surfaces = _structural_visibility_surfaces(surfaces)
    base_matrix = _registration_matrix(registration)
    search_frames = list(frames[:1]) if len(frames) > 1 else list(frames)
    before = _camera_visual_edge_metrics(
        frames=search_frames,
        visibility_surfaces=visibility_surfaces,
        registration=registration,
        downsample=8,
    )
    before_full = _camera_visual_edge_metrics(
        frames=frames,
        visibility_surfaces=visibility_surfaces,
        registration=registration,
        downsample=4,
    )

    seen: set[tuple[float, ...]] = set()
    candidate_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_matrix = base_matrix
    best_metrics = before

    def evaluate(params: Mapping[str, Any]) -> None:
        nonlocal best_params, best_matrix, best_metrics
        key = _candidate_key(params)
        if key in seen:
            return
        seen.add(key)
        correction = _similarity_correction_matrix(
            center=center,
            translation=np.asarray(params.get("translation", np.zeros(3)), dtype=np.float64),
            yaw_deg=float(params.get("yaw_deg", 0.0)),
            pitch_deg=float(params.get("pitch_deg", 0.0)),
            roll_deg=float(params.get("roll_deg", 0.0)),
            scale=float(params.get("scale", 1.0)),
        )
        matrix = correction @ base_matrix
        candidate_registration = _registration_with_matrix(registration, matrix)
        metrics = _camera_visual_edge_metrics(
            frames=search_frames,
            visibility_surfaces=visibility_surfaces,
            registration=candidate_registration,
            downsample=8,
        )
        row = {
            "source": str(params.get("source") or "candidate"),
            "yaw_deg": float(params.get("yaw_deg", 0.0)),
            "pitch_deg": float(params.get("pitch_deg", 0.0)),
            "roll_deg": float(params.get("roll_deg", 0.0)),
            "scale": float(params.get("scale", 1.0)),
            "translation_scene": [
                float(x) for x in np.asarray(params.get("translation", np.zeros(3)), dtype=np.float64).reshape(3)
            ],
            "score": float(metrics["score"]),
            "aggregate": metrics["aggregate"],
        }
        candidate_rows.append(row)
        if float(metrics["score"]) < float(best_metrics["score"]):
            best_params = dict(params)
            best_matrix = matrix
            best_metrics = metrics

    for seed in _compact_visual_edge_candidate_seeds(scene_per_m):
        evaluate(seed)

    if best_params is not None:
        params = dict(best_params)
        step_yaw = 0.35
        step_roll = 0.45
        step_scale = 0.010
        step_translation = np.asarray(
            [float(scene_per_m) * 0.035, float(scene_per_m) * 0.10, float(scene_per_m) * 0.05],
            dtype=np.float64,
        )
        for _level in range(2):
            base_translation = np.asarray(params.get("translation", np.zeros(3)), dtype=np.float64)
            for delta in (-step_yaw, step_yaw):
                trial = dict(params)
                trial["source"] = "coordinate_refine_yaw"
                trial["yaw_deg"] = float(params.get("yaw_deg", 0.0)) + delta
                evaluate(trial)
            for delta in (-step_roll, step_roll):
                trial = dict(params)
                trial["source"] = "coordinate_refine_roll"
                trial["roll_deg"] = float(params.get("roll_deg", 0.0)) + delta
                evaluate(trial)
            for delta in (-step_scale, step_scale):
                trial = dict(params)
                trial["source"] = "coordinate_refine_scale"
                trial["scale"] = max(0.90, min(1.08, float(params.get("scale", 1.0)) + delta))
                evaluate(trial)
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    trial = dict(params)
                    trial["source"] = f"coordinate_refine_translation_{axis}"
                    translation = base_translation.copy()
                    translation[axis] += sign * float(step_translation[axis])
                    trial["translation"] = translation
                    evaluate(trial)
            if best_params is not None:
                params = dict(best_params)
            step_yaw *= 0.5
            step_roll *= 0.5
            step_scale *= 0.5
            step_translation *= 0.5

    before_score = float(before["score"])
    after_score = float(best_metrics["score"])
    before_agg = before["aggregate"]
    after_agg = best_metrics["aggregate"]
    score_improved = after_score <= before_score * 0.94
    edge_improved = (
        float(after_agg["median_boundary_to_image_edge_px"] or 9999.0)
        <= float(before_agg["median_boundary_to_image_edge_px"] or 9999.0) * 0.90
        or float(after_agg["p90_boundary_to_image_edge_px"] or 9999.0)
        <= float(before_agg["p90_boundary_to_image_edge_px"] or 9999.0) * 0.90
    )
    visible_ok = float(after_agg["median_visible_model_pixel_ratio"] or 0.0) >= 0.68
    selected_row = min(candidate_rows, key=lambda row: float(row["score"])) if candidate_rows else None
    refinement_metrics = {
        "status": "ok" if best_params is not None and score_improved and edge_improved and visible_ok else "skipped",
        "method": "bounded_camera_visual_edge_search",
        "center_scene": [float(x) for x in center],
        "search_downsample": 8,
        "validation_downsample": 4,
        "search_frame_ids": [frame.frame_id for frame in search_frames],
        "before": before_full["aggregate"],
        "before_search": before_agg,
        "before_score": before_score,
        "after": after_agg,
        "after_score": after_score,
        "candidate_count": int(len(candidate_rows)),
        "selected_candidate": selected_row,
        "acceptance": {
            "score_improved": bool(score_improved),
            "edge_improved": bool(edge_improved),
            "visible_ok": bool(visible_ok),
        },
        "candidates": sorted(candidate_rows, key=lambda row: float(row["score"]))[:12],
    }
    if best_params is None or not (score_improved and edge_improved and visible_ok):
        refined = dict(registration)
        refined["camera_visual_edge_refinement"] = refinement_metrics
        return refined

    refined = _registration_with_matrix(registration, best_matrix)
    after_full = _camera_visual_edge_metrics(
        frames=frames,
        visibility_surfaces=visibility_surfaces,
        registration=refined,
        downsample=4,
    )
    refined["method"] = f"{registration.get('method', 'plane_surface_registration')}+camera_visual_edge_refinement"
    correction = best_matrix @ np.linalg.inv(base_matrix)
    translation = np.asarray(best_params.get("translation", np.zeros(3)), dtype=np.float64).reshape(3)
    refinement_metrics["correction_col_major"] = [float(x) for x in correction.flatten(order="F")]
    refinement_metrics["after"] = after_full["aggregate"]
    refinement_metrics["after_validation_score"] = float(after_full["score"])
    refinement_metrics["translation_scene"] = [float(x) for x in translation]
    refinement_metrics["rotation_deg"] = {
        "yaw": float(best_params.get("yaw_deg", 0.0)),
        "pitch": float(best_params.get("pitch_deg", 0.0)),
        "roll": float(best_params.get("roll_deg", 0.0)),
    }
    refinement_metrics["scale"] = float(best_params.get("scale", 1.0))
    refined["camera_visual_edge_refinement"] = refinement_metrics
    refined["visual_alignment_correction"] = {
        "status": "ok",
        "method": "bounded_camera_visual_edge_search",
        "center_scene": [float(x) for x in center],
        "translation_scene": [float(x) for x in translation],
        "rotation_deg": refinement_metrics["rotation_deg"],
        "scale": float(best_params.get("scale", 1.0)),
        "correction_col_major": refinement_metrics["correction_col_major"],
    }
    return refined


def _rotation_y_matrix(radians: float) -> np.ndarray:
    angle = float(radians)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_vector_matrix(vector: Sequence[float]) -> np.ndarray:
    omega = np.asarray(vector, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    skew = np.asarray(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ],
        dtype=np.float64,
    )
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64) + skew
    unit_skew = skew / theta
    return np.eye(3, dtype=np.float64) + (math.sin(theta) * unit_skew) + ((1.0 - math.cos(theta)) * (unit_skew @ unit_skew))


def _bounded_rotation_step(delta: Sequence[float], *, max_component: float, max_norm: float) -> np.ndarray:
    step = np.clip(np.asarray(delta, dtype=np.float64).reshape(3), -float(max_component), float(max_component))
    norm = float(np.linalg.norm(step))
    if norm > float(max_norm) > 0.0:
        step = step * (float(max_norm) / norm)
    return step


def _rotation_step_metrics(rotation_step: Sequence[float]) -> dict[str, float]:
    step = np.asarray(rotation_step, dtype=np.float64).reshape(3)
    return {
        "pitch_step_deg": float(math.degrees(step[0])),
        "yaw_step_deg": float(math.degrees(step[1])),
        "roll_step_deg": float(math.degrees(step[2])),
        "rotation_step_deg": float(math.degrees(float(np.linalg.norm(step)))),
    }


def _normal_angle_after_matrix(source_normal: np.ndarray, target_normal: np.ndarray, matrix: np.ndarray) -> float:
    linear = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[:3, :3]
    try:
        transformed = np.linalg.inv(linear).T @ np.asarray(source_normal, dtype=np.float64).reshape(3)
    except np.linalg.LinAlgError:
        return 180.0
    src_norm = float(np.linalg.norm(transformed))
    tgt = np.asarray(target_normal, dtype=np.float64).reshape(3)
    tgt_norm = float(np.linalg.norm(tgt))
    if src_norm <= 1e-9 or tgt_norm <= 1e-9:
        return 180.0
    dot = abs(float(np.dot(transformed / src_norm, tgt / tgt_norm)))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def _refine_registration_with_surface_support(
    *,
    points_world: np.ndarray,
    registration: Mapping[str, Any],
    candidates: Sequence[tuple[Any, str]],
    scene_per_m: float,
    sample_budget: int = 60_000,
    iterations: int = 8,
    min_support_samples: int = 256,
) -> dict[str, Any]:
    pts_world = np.asarray(points_world, dtype=np.float64).reshape((-1, 3))
    finite = np.all(np.isfinite(pts_world), axis=1)
    pts_world = pts_world[finite]
    if pts_world.shape[0] == 0:
        raise VirtualTwinBuildError("surface registration refinement has no finite accepted points")
    stride = int(np.ceil(pts_world.shape[0] / float(max(1, int(sample_budget))))) if pts_world.shape[0] > int(sample_budget) else 1
    sample_world = pts_world[::stride]
    matrix = _registration_matrix(registration).copy()

    before_assignment = _assign_sample_points_to_candidate_surfaces(
        transform_points(sample_world, matrix),
        candidates,
        scene_per_m=scene_per_m,
        inside_margin_scale=0.25,
        max_distance_scale=1.50,
    )
    before_metrics = _surface_assignment_metrics(before_assignment)
    if int(before_metrics["assigned_sample_count"]) < int(min_support_samples):
        raise VirtualTwinBuildError(
            "surface registration refinement needs at least "
            f"{int(min_support_samples)} assigned samples; got {before_metrics['assigned_sample_count']}"
        )

    steps: list[dict[str, Any]] = []
    max_rotation_component_step = math.radians(0.20)
    max_rotation_norm_step = math.radians(0.30)
    max_log_scale_step = 0.006
    max_translation_step = max(0.01, float(scene_per_m) * 0.08)
    for iteration in range(max(1, int(iterations))):
        sample_scene = transform_points(sample_world, matrix)
        assignment = _assign_sample_points_to_candidate_surfaces(
            sample_scene,
            candidates,
            scene_per_m=scene_per_m,
            inside_margin_scale=0.30,
            max_distance_scale=1.75,
        )
        best_surface_idx = np.asarray(assignment["best_surface_idx"], dtype=np.int32)
        signed = np.asarray(assignment["best_signed_distance"], dtype=np.float64)
        supported = best_surface_idx >= 0
        if int(np.count_nonzero(supported)) < int(min_support_samples):
            raise VirtualTwinBuildError(
                "surface registration refinement lost support after iteration "
                f"{iteration}: {int(np.count_nonzero(supported))} samples"
            )

        residual_abs = np.abs(signed[supported])
        robust_limit = max(float(scene_per_m) * 0.90, float(np.median(residual_abs)) * 3.0)
        keep = supported & (np.abs(signed) <= robust_limit)
        if int(np.count_nonzero(keep)) < int(min_support_samples):
            raise VirtualTwinBuildError(
                "surface registration refinement robust filter left too few samples after iteration "
                f"{iteration}: {int(np.count_nonzero(keep))}"
            )

        fit_points = sample_scene[keep]
        fit_signed = signed[keep]
        fit_indices = best_surface_idx[keep]
        normals: list[np.ndarray] = []
        for idx in fit_indices:
            surface = candidates[int(idx)][0]
            normal = np.asarray(surface.normal, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(normal))
            normals.append(normal / norm if norm > 1e-9 else np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        fit_normals = np.stack(normals, axis=0)

        jacobian = np.zeros((fit_points.shape[0], 7), dtype=np.float64)
        # Point-to-plane fit: bounded scene-space pitch/yaw/roll, uniform scale, translation x/y/z.
        jacobian[:, 0:3] = np.cross(fit_points, fit_normals)
        jacobian[:, 3] = np.sum(fit_points * fit_normals, axis=1)
        jacobian[:, 4:7] = fit_normals
        rhs = -fit_signed
        robust_scale = max(float(scene_per_m) * 0.25, float(np.median(np.abs(fit_signed))) * 1.4826, 1.0)
        weights = 1.0 / np.maximum(1.0, np.abs(fit_signed) / robust_scale)
        try:
            delta, *_ = np.linalg.lstsq(jacobian * weights[:, None], rhs * weights, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise VirtualTwinBuildError(f"surface registration refinement solve failed: {exc}") from exc

        rotation_step = _bounded_rotation_step(
            delta[0:3],
            max_component=max_rotation_component_step,
            max_norm=max_rotation_norm_step,
        )
        log_scale = float(np.clip(delta[3], -max_log_scale_step, max_log_scale_step))
        translation = np.asarray(delta[4:7], dtype=np.float64)
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > max_translation_step:
            translation = translation * (max_translation_step / translation_norm)

        incremental = np.eye(4, dtype=np.float64)
        incremental[:3, :3] = math.exp(log_scale) * _rotation_vector_matrix(rotation_step)
        incremental[:3, 3] = translation
        matrix = incremental @ matrix
        steps.append(
            {
                "iteration": int(iteration),
                "fit_sample_count": int(np.count_nonzero(keep)),
                "assigned_sample_count": int(np.count_nonzero(supported)),
                "robust_limit_scene_units": float(robust_limit),
                **_rotation_step_metrics(rotation_step),
                "log_scale_step": float(log_scale),
                "translation_step_scene_units": [float(x) for x in translation],
            }
        )

    after_assignment = _assign_sample_points_to_candidate_surfaces(
        transform_points(sample_world, matrix),
        candidates,
        scene_per_m=scene_per_m,
        inside_margin_scale=0.25,
        max_distance_scale=1.50,
    )
    after_metrics = _surface_assignment_metrics(after_assignment)
    median_tolerance_scene_units = max(1.0, float(scene_per_m) * 0.05)
    p90_tolerance_scene_units = max(1.0, float(scene_per_m) * 0.08)
    assignment_improved = int(after_metrics["assigned_sample_count"]) >= int(before_metrics["assigned_sample_count"])
    median_improved = float(after_metrics["median_abs_distance_scene_units"]) <= float(before_metrics["median_abs_distance_scene_units"])
    median_degraded = (
        float(after_metrics["median_abs_distance_scene_units"])
        > float(before_metrics["median_abs_distance_scene_units"]) + median_tolerance_scene_units
    )
    p90_degraded = (
        float(after_metrics["p90_abs_distance_scene_units"])
        > float(before_metrics["p90_abs_distance_scene_units"]) + p90_tolerance_scene_units
    )
    p90_strong_improved = (
        float(after_metrics["p90_abs_distance_scene_units"])
        <= float(before_metrics["p90_abs_distance_scene_units"]) * 0.85
    )
    median_bounded = (
        float(after_metrics["median_abs_distance_scene_units"])
        <= float(before_metrics["median_abs_distance_scene_units"]) + max(1.0, float(scene_per_m) * 0.10)
    )
    if (
        median_degraded
        and not (assignment_improved and p90_strong_improved and median_bounded)
    ) or (p90_degraded and not (median_improved and assignment_improved)):
        raise VirtualTwinBuildError(
            "surface registration refinement did not improve dense surface fit: "
            f"before={before_metrics} after={after_metrics}"
        )

    refined = dict(registration)
    refined["world_to_menon_scene_col_major"] = [float(x) for x in matrix.flatten(order="F")]
    refined["matrix_row_major"] = [float(x) for x in matrix.reshape(-1)]
    scale = abs(float(np.linalg.det(matrix[:3, :3]))) ** (1.0 / 3.0)
    refined["scene_per_m"] = float(scale)
    refined["s_obj_to_m"] = float(1.0 / scale) if scale > 1e-9 else 1.0
    refined["method"] = f"{registration.get('method', 'plane_surface_registration')}+surface_rpy_scale_translation_refinement"
    surface_correction = matrix @ np.linalg.inv(_registration_matrix(registration))
    refined["surface_refinement_correction_col_major"] = [float(x) for x in surface_correction.flatten(order="F")]
    refined["surface_refinement"] = {
        "status": "ok",
        "method": "dense_point_to_structural_surface_rpy_scale_translation",
        "sample_stride": int(stride),
        "median_tolerance_scene_units": float(median_tolerance_scene_units),
        "p90_tolerance_scene_units": float(p90_tolerance_scene_units),
        "p90_strong_improved": bool(p90_strong_improved),
        "median_bounded": bool(median_bounded),
        "before": before_metrics,
        "after": after_metrics,
        "steps": steps,
    }

    angular: list[float] = []
    residuals: list[float] = []
    for item in refined.get("correspondences", []) or []:
        if not isinstance(item, Mapping):
            continue
        source_point = np.asarray(item.get("source_centroid_m"), dtype=np.float64)
        target_point = np.asarray(item.get("target_centroid_scene"), dtype=np.float64)
        if source_point.shape == (3,) and target_point.shape == (3,):
            transformed = (matrix[:3, :3] @ source_point) + matrix[:3, 3]
            residuals.append(float(np.linalg.norm(transformed - target_point)))
        source_normal = np.asarray(item.get("source_normal"), dtype=np.float64)
        target_normal = np.asarray(item.get("target_normal"), dtype=np.float64)
        if source_normal.shape == (3,) and target_normal.shape == (3,):
            angular.append(_normal_angle_after_matrix(source_normal, target_normal, matrix))
    if angular:
        refined["median_normal_error_deg"] = float(np.median(angular))
        refined["p90_normal_error_deg"] = float(np.percentile(angular, 90.0))
        refined["max_normal_error_deg"] = float(np.max(angular))
    if residuals:
        refined["median_residual_scene_units"] = float(np.median(residuals))
        refined["p90_residual_scene_units"] = float(np.percentile(residuals, 90.0))
        refined["position_rmse_scene_units"] = float(math.sqrt(float(np.mean(np.square(residuals)))))
    return refined


def _surface_contains_points(
    points_scene: np.ndarray,
    surface: Any,
    *,
    scene_per_m: float,
    inside_margin_scale: float = 0.18,
    max_distance_scale: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_scene, dtype=np.float64).reshape((-1, 3))
    normal = np.asarray(surface.normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return np.zeros((pts.shape[0],), dtype=bool), np.full((pts.shape[0],), np.inf, dtype=np.float64)
    normal = normal / norm
    centroid = np.asarray(surface.centroid, dtype=np.float64).reshape(3)
    signed = (pts - centroid) @ normal
    projected = pts - (signed[:, None] * normal[None, :])
    inside_margin = max(0.05, float(scene_per_m) * float(inside_margin_scale))
    max_distance = max(0.25, float(scene_per_m) * float(max_distance_scale))
    bounds_min = np.asarray(surface.bounds_min, dtype=np.float64).reshape(3) - inside_margin
    bounds_max = np.asarray(surface.bounds_max, dtype=np.float64).reshape(3) + inside_margin
    inside = np.all((projected >= bounds_min) & (projected <= bounds_max), axis=1)
    distance = np.abs(signed)
    return (inside & (distance <= max_distance)).astype(bool), distance


def _assign_support_to_surfaces(
    points_scene: np.ndarray,
    colors: np.ndarray,
    candidates: Sequence[tuple[Any, str]],
    *,
    scene_per_m: float,
    browser_point_budget: int,
    sample_pixels: np.ndarray | None = None,
    sample_frame_indices: np.ndarray | None = None,
    frames: Sequence[VirtualTwinFrameInput] | None = None,
    visibility_surfaces: Sequence[tuple[Any, str]] | None = None,
    registration: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pts_all = np.asarray(points_scene, dtype=np.float64).reshape((-1, 3))
    rgb_all = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
    pixels_all = (
        np.asarray(sample_pixels, dtype=np.int32).reshape((-1, 2))
        if sample_pixels is not None
        else None
    )
    frame_indices_all = (
        np.asarray(sample_frame_indices, dtype=np.int32).reshape((-1,))
        if sample_frame_indices is not None
        else None
    )
    if pixels_all is not None and pixels_all.shape[0] != pts_all.shape[0]:
        raise VirtualTwinBuildError("surface support assignment requires one source pixel per scene point")
    if frame_indices_all is not None and frame_indices_all.shape[0] != pts_all.shape[0]:
        raise VirtualTwinBuildError("surface support assignment requires one frame index per scene point")
    finite = np.all(np.isfinite(pts_all), axis=1)
    pts_all = pts_all[finite]
    rgb_all = rgb_all[finite]
    if pixels_all is not None:
        pixels_all = pixels_all[finite]
    if frame_indices_all is not None:
        frame_indices_all = frame_indices_all[finite]
    if pts_all.shape[0] == 0:
        raise VirtualTwinBuildError("model-surface artifact has no finite accepted points")

    max_surface_count = max(1, int(browser_point_budget) // 4)
    source_budget = max(max_surface_count * 6, max_surface_count)
    source_stride = int(np.ceil(pts_all.shape[0] / float(source_budget))) if pts_all.shape[0] > source_budget else 1
    sample_points = pts_all[::source_stride]
    sample_colors = rgb_all[::source_stride]
    visible_assignment_ready = (
        pixels_all is not None
        and frame_indices_all is not None
        and frames is not None
        and visibility_surfaces is not None
        and registration is not None
    )
    if visible_assignment_ready:
        assignment = _assign_sample_points_to_visible_surfaces(
            sample_points,
            pixels_all[::source_stride],
            frame_indices_all[::source_stride],
            candidates,
            frames=frames or [],
            visibility_surfaces=visibility_surfaces or [],
            registration=registration or {},
            scene_per_m=scene_per_m,
            downsample=2,
            max_distance_scale=0.32,
            min_max_distance=6.0,
        )
    else:
        assignment = _assign_sample_points_to_candidate_surfaces(
            sample_points,
            candidates,
            scene_per_m=scene_per_m,
            inside_margin_scale=0.12,
            max_distance_scale=1.15,
        )
    best_surface_idx = np.asarray(assignment["best_surface_idx"], dtype=np.int32)
    best_distance = np.asarray(assignment["best_distance"], dtype=np.float64)

    supported = best_surface_idx >= 0
    if not np.any(supported):
        raise VirtualTwinBuildError(
            "model-surface artifact could not project registered depth support onto any Menon structural surface"
        )
    if visible_assignment_ready:
        support_ratio = float(np.count_nonzero(supported) / max(1, sample_points.shape[0]))
        if sample_points.shape[0] >= 5000 and support_ratio < 0.02:
            raise VirtualTwinBuildError(
                "camera-visible surface assignment rejected almost all support samples: "
                f"ratio={support_ratio:.4f} assignment={_surface_assignment_metrics(assignment)}"
            )

    support_counts = np.bincount(best_surface_idx[supported], minlength=len(candidates))
    color_sums = np.zeros((len(candidates), 3), dtype=np.float64)
    np.add.at(color_sums, best_surface_idx[supported], sample_colors[supported].astype(np.float64))

    offset = max(1e-4, min(0.75, float(scene_per_m) * 0.004))
    vertices: list[np.ndarray] = []
    vertex_colors: list[np.ndarray] = []
    indices: list[int] = []
    triangle_surface_indices: list[int] = []
    surface_rows: list[dict[str, Any]] = []
    min_support = max(32, int(math.ceil(float(sample_points.shape[0]) * 0.0004)))
    for surface_idx in np.argsort(-support_counts):
        count = int(support_counts[int(surface_idx)])
        if count < min_support:
            continue
        surface, label = candidates[int(surface_idx)]
        triangles = np.asarray(surface.triangles, dtype=np.float32).reshape((-1, 3, 3))
        if triangles.shape[0] == 0:
            continue
        next_vertex_count = len(vertices) + int(triangles.shape[0]) * 3
        if next_vertex_count > int(browser_point_budget) and vertices:
            break
        normal = np.asarray(surface.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        normal = normal / norm if norm > 1e-9 else np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        color = np.clip(color_sums[int(surface_idx)] / max(1, count), 0, 255).astype(np.uint8)
        base = len(vertices)
        for tri in triangles:
            tri_offset = tri.astype(np.float64) + (normal[None, :] * offset)
            for vertex in tri_offset:
                vertices.append(vertex.astype(np.float32))
                vertex_colors.append(color)
            indices.extend([base, base + 1, base + 2])
            triangle_surface_indices.append(int(surface_idx))
            base += 3
        surface_rows.append(
            {
                "surface_id": str(surface.surface_id),
                "label": label,
                "support_sample_count": count,
                "area": float(surface.area),
                "object_name": surface.object_name,
                "material": surface.material,
                "triangle_count": int(triangles.shape[0]),
            }
        )

    if not vertices or not indices:
        raise VirtualTwinBuildError("model-surface artifact support was found, but no mesh triangles were emitted")

    projected_distances = best_distance[supported]
    mesh_vertices = np.stack(vertices, axis=0).astype(np.float32)
    mesh_colors = np.stack(vertex_colors, axis=0).astype(np.uint8)
    mesh_indices = np.asarray(indices, dtype=np.uint32)
    mesh_triangle_surface_indices = np.asarray(triangle_surface_indices, dtype=np.int32)
    metrics = {
        "artifact_type": "model_surface_mesh",
        "source_sample_count": int(sample_points.shape[0]),
        "source_sample_stride": int(source_stride),
        "surface_candidate_count": int(len(candidates)),
        "supported_model_surface_count": int(len(surface_rows)),
        "projected_support_sample_count": int(np.count_nonzero(supported)),
        "projection_support_ratio": float(np.count_nonzero(supported) / max(1, sample_points.shape[0])),
        "median_projection_distance_scene_units": float(np.median(projected_distances)),
        "p90_projection_distance_scene_units": float(np.percentile(projected_distances, 90.0)),
        "max_projection_distance_scene_units": float(np.max(projected_distances)),
        "surface_offset_scene_units": float(offset),
        "min_surface_support_samples": int(min_support),
        "served_glb_vertex_count": int(mesh_vertices.shape[0]),
        "served_glb_triangle_count": int(mesh_indices.shape[0] // 3),
        "assignment_method": str(assignment.get("assignment_method") or "unknown"),
        "assignment": _surface_assignment_metrics(assignment),
        "surface_support": surface_rows[:96],
    }
    return mesh_vertices, mesh_indices, mesh_colors, mesh_triangle_surface_indices, metrics


def _tile_barycentric_grid(tile_px: int, pad_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tile = max(8, int(tile_px))
    pad = max(1, min(int(pad_px), (tile - 2) // 3))
    a = np.asarray([float(pad), float(pad)], dtype=np.float64)
    b = np.asarray([float(tile - pad - 1), float(pad)], dtype=np.float64)
    c = np.asarray([float(pad), float(tile - pad - 1)], dtype=np.float64)
    yy, xx = np.mgrid[0:tile, 0:tile]
    points = np.stack([xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5], axis=-1).reshape((-1, 2))
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    denom = (d00 * d11) - (d01 * d01)
    if abs(denom) <= 1e-9:
        raise VirtualTwinBuildError("degenerate texture atlas tile triangle")
    d20 = v2 @ v0
    d21 = v2 @ v1
    w1 = ((d11 * d20) - (d01 * d21)) / denom
    w2 = ((d00 * d21) - (d01 * d20)) / denom
    w0 = 1.0 - w1 - w2
    bary = np.stack([w0, w1, w2], axis=1)
    mask = np.all(bary >= -1e-5, axis=1)
    return (
        xx.reshape((-1,))[mask].astype(np.int32),
        yy.reshape((-1,))[mask].astype(np.int32),
        bary[mask].astype(np.float64),
        np.stack([a, b, c], axis=0).astype(np.float64),
    )


def _project_texture_points_to_frame(
    frame: VirtualTwinFrameInput,
    points_scene: np.ndarray,
    scene_to_world: np.ndarray,
    world_to_scene: np.ndarray,
    *,
    expected_surface: Any | None = None,
    expected_visibility_surface_idx: int | None = None,
    visibility_buffer: Mapping[str, Any] | None = None,
    scene_per_m: float = 1.0,
    min_confidence: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    pts_scene = np.asarray(points_scene, dtype=np.float64).reshape((-1, 3))
    hom_scene = np.concatenate([pts_scene, np.ones((pts_scene.shape[0], 1), dtype=np.float64)], axis=1)
    points_world = (np.asarray(scene_to_world, dtype=np.float64).reshape(4, 4) @ hom_scene.T).T[:, :3]
    world_to_camera = np.asarray(frame.calibration.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    hom_world = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    points_camera = (world_to_camera @ hom_world.T).T[:, :3]
    z = points_camera[:, 2]

    intrinsics = np.asarray(frame.calibration.intrinsics, dtype=np.float64).reshape(3, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (points_camera[:, 0] * intrinsics[0, 0] / z) + intrinsics[0, 2]
        v = (points_camera[:, 1] * intrinsics[1, 1] / z) + intrinsics[1, 2]

    image = np.asarray(frame.image_bgr, dtype=np.uint8)
    depth = np.asarray(frame.map_depth, dtype=np.float32)
    confidence = np.asarray(frame.map_confidence, dtype=np.float32)
    mask = np.asarray(frame.map_mask, dtype=bool)
    h, w = int(image.shape[0]), int(image.shape[1])
    if depth.shape != (h, w):
        raise VirtualTwinBuildError(
            f"texture bake requires MapAnything depth to match RGB size for {frame.frame_id}: "
            f"depth={depth.shape} image={(h, w)}"
        )

    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    projected = finite & (z > 0.0) & (u >= 0.0) & (v >= 0.0) & (u < float(w - 1)) & (v < float(h - 1))
    xi = np.clip(np.rint(u).astype(np.int64), 0, max(0, w - 1))
    yi = np.clip(np.rint(v).astype(np.int64), 0, max(0, h - 1))
    sampled_depth = depth[yi, xi]
    sampled_confidence = confidence[yi, xi]
    sampled_mask = mask[yi, xi]
    depth_tolerance = np.maximum(1.25, np.abs(z) * 0.30)
    depth_available = (
        sampled_mask
        & np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & np.isfinite(sampled_confidence)
        & (sampled_confidence >= float(min_confidence))
    )
    # The structural model is the authoritative texture target. MapAnything is
    # useful here as an occlusion cue, but requiring exact per-texel depth
    # agreement leaves large valid walls/floors untextured after registration.
    depth_occluded = depth_available & ((sampled_depth.astype(np.float64) + depth_tolerance) < z)
    pre_surface_valid = projected
    surface_candidate_count = int(np.count_nonzero(pre_surface_valid))
    surface_rejected_count = 0
    visibility_rejected_count = 0
    depth_occlusion_rejected_count = 0
    if expected_surface is not None and surface_candidate_count > 0:
        valid_indices = np.nonzero(pre_surface_valid)[0]
        surface_ok_values = np.ones((valid_indices.shape[0],), dtype=bool)
        if visibility_buffer is not None and expected_visibility_surface_idx is not None:
            vis_ids = np.asarray(visibility_buffer["surface_ids"], dtype=np.int32)
            vis_scale = max(1, int(visibility_buffer.get("downsample", 1)))
            vis_x = np.clip((xi[valid_indices] / float(vis_scale)).astype(np.int32), 0, vis_ids.shape[1] - 1)
            vis_y = np.clip((yi[valid_indices] / float(vis_scale)).astype(np.int32), 0, vis_ids.shape[0] - 1)
            visible_as_expected = vis_ids[vis_y, vis_x] == int(expected_visibility_surface_idx)
            if not np.all(visible_as_expected):
                nearby = visible_as_expected.copy()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = np.clip(vis_x + dx, 0, vis_ids.shape[1] - 1)
                        ny = np.clip(vis_y + dy, 0, vis_ids.shape[0] - 1)
                        nearby |= vis_ids[ny, nx] == int(expected_visibility_surface_idx)
                visible_as_expected = nearby
            visibility_rejected_count = int(np.count_nonzero(~visible_as_expected))
            surface_ok_values &= visible_as_expected
        occluded_values = depth_occluded[valid_indices]
        depth_occlusion_rejected_count = int(np.count_nonzero(occluded_values & surface_ok_values))
        surface_ok_values &= ~occluded_values
        surface_ok = np.zeros((pre_surface_valid.shape[0],), dtype=bool)
        surface_ok[valid_indices] = surface_ok_values
        surface_rejected_count = int(surface_candidate_count - np.count_nonzero(surface_ok_values))
        valid = pre_surface_valid & surface_ok
    else:
        valid = pre_surface_valid & ~depth_occluded
        depth_occlusion_rejected_count = int(np.count_nonzero(pre_surface_valid & depth_occluded))
    colors = image[yi, xi, :3][:, ::-1].copy()
    return (
        valid.astype(bool),
        colors.astype(np.uint8),
        {
            "surface_candidate_count": int(surface_candidate_count),
            "surface_rejected_count": int(surface_rejected_count),
            "visibility_rejected_count": int(visibility_rejected_count),
            "depth_occlusion_rejected_count": int(depth_occlusion_rejected_count),
        },
    )


def _bake_model_surface_texture(
    mesh_vertices: np.ndarray,
    mesh_indices: np.ndarray,
    mesh_triangle_surface_indices: np.ndarray,
    surface_candidates: Sequence[tuple[Any, str]],
    visibility_surfaces: Sequence[tuple[Any, str]],
    frames: Sequence[VirtualTwinFrameInput],
    registration: Mapping[str, Any],
    *,
    texture_tile_px: int,
    min_texture_coverage: float,
    texture_exposure: float,
    texture_gamma: float,
    texture_contrast: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not frames:
        raise VirtualTwinBuildError("texture bake requires at least one RGB keyframe")
    verts = np.asarray(mesh_vertices, dtype=np.float64).reshape((-1, 3))
    idx = np.asarray(mesh_indices, dtype=np.uint32).reshape((-1,))
    tri_surface_idx = np.asarray(mesh_triangle_surface_indices, dtype=np.int32).reshape((-1,))
    if idx.size % 3 != 0:
        raise VirtualTwinBuildError("texture bake requires triangle indices")
    triangle_count = int(idx.size // 3)
    if triangle_count <= 0:
        raise VirtualTwinBuildError("texture bake requires at least one model-surface triangle")
    if tri_surface_idx.shape[0] != triangle_count:
        raise VirtualTwinBuildError(
            f"texture bake requires one surface assignment per triangle: "
            f"triangles={triangle_count} assignments={tri_surface_idx.shape[0]}"
        )

    tile_px = max(16, int(texture_tile_px))
    pad_px = max(2, int(round(tile_px * 0.04)))
    cols = int(np.ceil(np.sqrt(float(triangle_count))))
    rows = int(np.ceil(float(triangle_count) / float(cols)))
    atlas_w = int(cols * tile_px)
    atlas_h = int(rows * tile_px)
    atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    texcoords = np.zeros((verts.shape[0], 2), dtype=np.float32)
    local_x, local_y, bary, local_uv_vertices = _tile_barycentric_grid(tile_px, pad_px)
    world_to_scene = _registration_matrix(registration)
    scene_to_world = np.linalg.inv(world_to_scene)
    frame_fill_counts: dict[str, int] = {str(frame.frame_id): 0 for frame in frames}
    visibility_by_frame = {
        str(frame.frame_id): _structural_visibility_buffer(frame, visibility_surfaces, registration, downsample=4)
        for frame in frames
    }
    visibility_surface_by_id = {
        str(getattr(surface, "surface_id", "")): int(idx)
        for idx, (surface, _label) in enumerate(visibility_surfaces)
    }
    surface_gate_candidate_texels = 0
    surface_gate_rejected_texels = 0
    texture_visibility_rejected_texels = 0
    texture_depth_occlusion_rejected_texels = 0
    visibility_missing_surface_count = 0
    total_texels = 0
    filled_texels = 0
    triangle_coverage: list[float] = []

    for tri_idx in range(triangle_count):
        tri_vertex_indices = idx[tri_idx * 3 : tri_idx * 3 + 3].astype(np.int64)
        surface_idx = int(tri_surface_idx[tri_idx])
        if surface_idx < 0 or surface_idx >= len(surface_candidates):
            raise VirtualTwinBuildError(f"texture triangle {tri_idx} references invalid surface index {surface_idx}")
        expected_surface = surface_candidates[surface_idx][0]
        expected_visibility_surface_idx = visibility_surface_by_id.get(str(getattr(expected_surface, "surface_id", "")))
        if expected_visibility_surface_idx is None:
            visibility_missing_surface_count += 1
        tri_scene = verts[tri_vertex_indices]
        col = tri_idx % cols
        row = tri_idx // cols
        tile_x = int(col * tile_px)
        tile_y = int(row * tile_px)
        uv_pixels = local_uv_vertices + np.asarray([tile_x, tile_y], dtype=np.float64)
        texcoords[tri_vertex_indices] = np.stack(
            [
                uv_pixels[:, 0] / float(atlas_w),
                uv_pixels[:, 1] / float(atlas_h),
            ],
            axis=1,
        ).astype(np.float32)

        scene_points = bary @ tri_scene
        total_texels += int(scene_points.shape[0])
        remaining = np.ones((scene_points.shape[0],), dtype=bool)
        frame_candidates: list[tuple[int, str, np.ndarray, np.ndarray]] = []
        for frame in frames:
            valid, colors, gate_stats = _project_texture_points_to_frame(
                frame,
                scene_points,
                scene_to_world,
                world_to_scene,
                expected_surface=expected_surface,
                expected_visibility_surface_idx=expected_visibility_surface_idx,
                visibility_buffer=visibility_by_frame.get(str(frame.frame_id)),
                scene_per_m=float(registration.get("scene_per_m") or 1.0),
            )
            surface_gate_candidate_texels += int(gate_stats["surface_candidate_count"])
            surface_gate_rejected_texels += int(gate_stats["surface_rejected_count"])
            texture_visibility_rejected_texels += int(gate_stats.get("visibility_rejected_count", 0))
            texture_depth_occlusion_rejected_texels += int(gate_stats.get("depth_occlusion_rejected_count", 0))
            frame_candidates.append((int(np.count_nonzero(valid)), str(frame.frame_id), valid, colors))
        for count, frame_id, valid, colors in sorted(frame_candidates, key=lambda item: item[0], reverse=True):
            if count <= 0 or not np.any(remaining):
                continue
            fill = remaining & valid
            fill_count = int(np.count_nonzero(fill))
            if fill_count <= 0:
                continue
            ys = tile_y + local_y[fill]
            xs = tile_x + local_x[fill]
            atlas[ys, xs, :3] = colors[fill]
            atlas[ys, xs, 3] = 255
            remaining[fill] = False
            frame_fill_counts[frame_id] = int(frame_fill_counts.get(frame_id, 0) + fill_count)
        tri_filled = int(scene_points.shape[0] - np.count_nonzero(remaining))
        filled_texels += tri_filled
        triangle_coverage.append(float(tri_filled / max(1, scene_points.shape[0])))

    coverage = float(filled_texels / max(1, total_texels))
    if coverage < float(min_texture_coverage):
        raise VirtualTwinBuildError(
            f"texture bake coverage too low: {coverage:.4f} < {float(min_texture_coverage):.4f}"
        )
    tone_exposure = max(0.05, float(texture_exposure))
    tone_gamma = max(0.05, float(texture_gamma))
    tone_contrast = max(0.05, float(texture_contrast))
    filled_mask = atlas[:, :, 3] > 0
    if np.any(filled_mask):
        rgb = atlas[:, :, :3].astype(np.float32) / 255.0
        rgb = np.clip(rgb * tone_exposure, 0.0, 1.0)
        rgb = np.power(rgb, tone_gamma)
        rgb = np.clip(((rgb - 0.5) * tone_contrast) + 0.5, 0.0, 1.0)
        atlas[:, :, :3][filled_mask] = np.rint(rgb[filled_mask] * 255.0).astype(np.uint8)
    triangle_coverage_np = np.asarray(triangle_coverage, dtype=np.float64)
    metrics = {
        "texture_source": "pipeline_rgb_keyframes_projected_with_mapanything_depth_and_surface_gate",
        "texture_tone_map": {
            "exposure": float(tone_exposure),
            "gamma": float(tone_gamma),
            "contrast": float(tone_contrast),
        },
        "texture_atlas_width": atlas_w,
        "texture_atlas_height": atlas_h,
        "texture_tile_px": tile_px,
        "texture_triangle_count": triangle_count,
        "texture_total_texel_count": int(total_texels),
        "texture_filled_texel_count": int(filled_texels),
        "texture_coverage_ratio": coverage,
        "texture_surface_assignment_gate": True,
        "texture_surface_gate_candidate_texel_count": int(surface_gate_candidate_texels),
        "texture_surface_gate_rejected_texel_count": int(surface_gate_rejected_texels),
        "texture_surface_gate_reject_ratio": float(
            surface_gate_rejected_texels / max(1, surface_gate_candidate_texels)
        ),
        "texture_visibility_rejected_texel_count": int(texture_visibility_rejected_texels),
        "texture_depth_occlusion_rejected_texel_count": int(texture_depth_occlusion_rejected_texels),
        "texture_visibility_gate": True,
        "texture_visibility_downsample": 4,
        "texture_visibility_missing_surface_count": int(visibility_missing_surface_count),
        "texture_visibility_frame_visible_pixels": {
            str(frame_id): int(payload.get("visible_pixel_count", 0))
            for frame_id, payload in visibility_by_frame.items()
        },
        "texture_triangle_coverage_median": float(np.median(triangle_coverage_np)) if triangle_coverage_np.size else 0.0,
        "texture_triangle_coverage_p10": float(np.percentile(triangle_coverage_np, 10.0)) if triangle_coverage_np.size else 0.0,
        "texture_frame_texel_counts": frame_fill_counts,
    }
    return texcoords, atlas, metrics


def build_virtual_twin_revision(
    *,
    frames: Sequence[VirtualTwinFrameInput],
    menon_obj_path: Path,
    store: VirtualTwinStore | None = None,
    revision_id: str | None = None,
    camera_label: str = "living-room",
    mapanything_refs: Mapping[str, Any] | None = None,
    zeroplane_refs: Mapping[str, Any] | None = None,
    browser_point_budget: int = 150_000,
    texture_tile_px: int = 96,
    min_texture_coverage: float = 0.02,
    texture_exposure: float = 2.2,
    texture_gamma: float = 0.65,
    texture_contrast: float = 1.08,
    min_texture_colorfulness: float = 2.0,
    min_mapanything_coverage: float = 0.02,
    min_zero_planes: int = 2,
    min_registration_correspondences: int = 3,
    max_room_model_leakage: float = 0.03,
    initial_world_to_menon_scene_col_major: Sequence[float] | None = None,
    scene_prior_refs: Mapping[str, Any] | None = None,
    camera_scene_anchor: Mapping[str, Any] | None = None,
    update_latest: bool = True,
) -> dict[str, Any]:
    if not frames:
        raise VirtualTwinBuildError("virtual-twin build requires at least one keyframe")
    model_path = Path(menon_obj_path)
    surfaces = parse_menon_obj_surfaces(model_path)
    if not surfaces:
        raise VirtualTwinBuildError(f"no structural surfaces parsed from Menon OBJ: {model_path}")

    store = store or VirtualTwinStore()
    revision_id = revision_id or revision_id_from_clock()
    revision_dir = store.revision_dir(revision_id)
    revision_dir.mkdir(parents=True, exist_ok=True)

    all_points_world: list[np.ndarray] = []
    all_pixels: list[np.ndarray] = []
    all_frame_indices: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_plane_json: list[dict[str, Any]] = []
    source_planes: list[SourcePlaneSurface] = []
    frame_point_samples: list[_FramePointSamples] = []
    frame_rows: list[dict[str, Any]] = []
    fusion_metrics: list[dict[str, Any]] = []
    source_frame_ids: list[str] = []
    calibration_fingerprints: dict[str, str] = {}
    floor_y_values: list[float] = []
    colorfulness_values: list[float] = []

    for frame_index, frame in enumerate(frames):
        source_frame_ids.append(frame.frame_id)
        calibration_fingerprints[frame.camera_id] = calibration_fingerprint(frame.calibration)
        floor_y_values.append(float(frame.calibration.floor_y))
        frame_colorfulness = rgb_colorfulness(frame.image_bgr)
        colorfulness_values.append(frame_colorfulness)
        frame_evidence = _write_frame_evidence(revision_dir, frame)
        fusion = fuse_mapanything_with_planes(
            frame_id=frame.frame_id,
            map_depth=frame.map_depth,
            map_confidence=frame.map_confidence,
            map_mask=frame.map_mask,
            intrinsics=frame.calibration.intrinsics,
            plane_candidates=frame.plane_candidates,
        )
        if fusion.metrics["mapanything_valid_depth_coverage"] < float(min_mapanything_coverage):
            raise VirtualTwinBuildError(
                f"MapAnything valid depth coverage too low for {frame.frame_id}: "
                f"{fusion.metrics['mapanything_valid_depth_coverage']:.4f}"
            )
        if len(fusion.planes) < int(min_zero_planes):
            raise VirtualTwinBuildError(
                f"ZeroPlane produced too few accepted planes for {frame.frame_id}: {len(fusion.planes)}"
            )
        twc = _camera_to_world_matrix(frame.calibration)
        points_world = transform_points(fusion.fused_points_camera, twc)
        colors = sample_colors_from_image(frame.image_bgr, fusion.fused_pixels)
        all_points_world.append(points_world)
        all_pixels.append(fusion.fused_pixels)
        all_frame_indices.append(np.full((points_world.shape[0],), int(frame_index), dtype=np.int32))
        all_colors.append(colors)
        frame_point_samples.append(
            _FramePointSamples(
                frame=frame,
                points_world=points_world,
                pixels=fusion.fused_pixels,
            )
        )
        fusion_metrics.append({"frame_id": frame.frame_id, **fusion.metrics})
        frame_rows.append(
            {
                "frame_id": frame.frame_id,
                "camera_id": frame.camera_id,
                "source_ref": frame.source_ref,
                "revision_artifacts": frame_evidence,
                "image_size": [int(frame.image_bgr.shape[1]), int(frame.image_bgr.shape[0])],
                "mapanything_valid_depth_coverage": fusion.metrics["mapanything_valid_depth_coverage"],
                "rgb_colorfulness": frame_colorfulness,
                "accepted_plane_count": len(fusion.planes),
                "surfel_count": int(points_world.shape[0]),
            }
        )
        for plane in fusion.planes:
            world_normal, world_offset = transform_plane(plane.normal, plane.offset, twc)
            world_centroid = transform_points(np.asarray([plane.centroid], dtype=np.float32), twc)[0]
            source_surface = _plane_to_source_surface(
                plane,
                world_normal,
                world_offset,
                world_centroid,
                floor_y=float(frame.calibration.floor_y),
            )
            source_planes.append(source_surface)
            all_plane_json.append(
                {
                    "frame_id": plane.frame_id,
                    "plane_id": plane.plane_id,
                    "semantic_label": source_surface.label,
                    "camera_normal": [float(x) for x in plane.normal],
                    "camera_offset": float(plane.offset),
                    "world_normal": [float(x) for x in world_normal],
                    "world_offset": float(world_offset),
                    "world_centroid_m": [float(x) for x in world_centroid],
                    "polygon": plane.polygon,
                    "mask_rle": plane.mask_rle,
                    "support_pixels": int(plane.support_pixels),
                    "confidence": float(plane.confidence),
                    "median_residual_m": float(plane.median_residual_m),
                    "p90_residual_m": float(plane.p90_residual_m),
                    "raw_median_residual_m": float(plane.raw_median_residual_m),
                }
            )

    points_world = np.concatenate(all_points_world, axis=0) if all_points_world else np.zeros((0, 3), dtype=np.float32)
    pixels = np.concatenate(all_pixels, axis=0) if all_pixels else np.zeros((0, 2), dtype=np.int32)
    frame_indices = (
        np.concatenate(all_frame_indices, axis=0)
        if all_frame_indices
        else np.zeros((0,), dtype=np.int32)
    )
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)
    if points_world.shape[0] == 0:
        raise VirtualTwinBuildError("virtual-twin build produced no surfels")
    median_colorfulness = float(np.median(colorfulness_values)) if colorfulness_values else 0.0
    if median_colorfulness < float(min_texture_colorfulness):
        raise VirtualTwinBuildError(
            f"RGB keyframes are effectively grayscale for {camera_label}: "
            f"median_colorfulness={median_colorfulness:.3f} < {float(min_texture_colorfulness):.3f}. "
            "Capture color-mode frames or pass --allow-grayscale-texture to build a grayscale diagnostic artifact."
        )

    registration = register_planes_to_menon_surfaces(
        source_planes,
        surfaces,
        floor_y=float(np.median(floor_y_values)) if floor_y_values else 0.0,
        min_correspondences=int(min_registration_correspondences),
        initial_world_to_scene_col_major=initial_world_to_menon_scene_col_major,
    )
    registration = _apply_camera_scene_anchor(
        registration,
        camera_scene_anchor,
        stage="plane_surface_registration",
    )
    if initial_world_to_menon_scene_col_major is not None:
        registration = _refine_registration_with_visible_surface_correspondences(
            frame_samples=frame_point_samples,
            surfaces=surfaces,
            registration=registration,
            scene_per_m=float(registration.get("scene_per_m") or 1.0),
        )
        registration = _apply_camera_scene_anchor(
            registration,
            camera_scene_anchor,
            stage="visible_surface_refinement",
        )
    else:
        registration = {
            **registration,
            "visible_surface_refinement": {
                "status": "not_applicable",
                "reason": "no_initial_menon_scene_prior",
            },
        }
    points_scene = _apply_registration_to_points(points_world, registration)
    scene_per_m = float(registration.get("scene_per_m") or 1.0)
    initial_model_mask, _initial_leakage_metrics = _model_acceptance_mask(
        points_scene,
        surfaces,
        scene_margin=max(50.0, scene_per_m * 2.0),
    )
    if int(np.count_nonzero(initial_model_mask)) <= 0:
        raise VirtualTwinBuildError("model leakage filter rejected all virtual-twin surfels")
    initial_accepted_points_world = points_world[initial_model_mask]
    initial_accepted_points_scene = points_scene[initial_model_mask]
    surface_candidates = _surface_candidates_for_points(
        initial_accepted_points_scene,
        surfaces,
        scene_per_m=scene_per_m,
    )
    if initial_world_to_menon_scene_col_major is not None:
        registration = _refine_registration_with_surface_support(
            points_world=initial_accepted_points_world,
            registration=registration,
            candidates=surface_candidates,
            scene_per_m=scene_per_m,
        )
        registration = _apply_camera_scene_anchor(
            registration,
            camera_scene_anchor,
            stage="surface_support_refinement",
        )
    else:
        registration = {
            **registration,
            "surface_refinement": {
                "status": "not_applicable",
                "reason": "no_initial_menon_scene_prior",
            },
        }
    points_scene = _apply_registration_to_points(points_world, registration)
    scene_per_m = float(registration.get("scene_per_m") or scene_per_m)
    model_mask, leakage_metrics = _model_acceptance_mask(
        points_scene,
        surfaces,
        scene_margin=max(50.0, scene_per_m * 2.0),
    )
    if int(np.count_nonzero(model_mask)) <= 0:
        raise VirtualTwinBuildError("model leakage filter rejected all virtual-twin surfels after surface refinement")
    if initial_world_to_menon_scene_col_major is not None:
        registration = _refine_registration_with_camera_visual_edges(
            frames=frames,
            surfaces=surfaces,
            registration=registration,
            accepted_scene_points=points_scene[model_mask],
            scene_per_m=scene_per_m,
        )
        registration = _apply_camera_scene_anchor(
            registration,
            camera_scene_anchor,
            stage="camera_visual_edge_refinement",
        )
        points_scene = _apply_registration_to_points(points_world, registration)
        scene_per_m = float(registration.get("scene_per_m") or scene_per_m)
        model_mask, leakage_metrics = _model_acceptance_mask(
            points_scene,
            surfaces,
            scene_margin=max(50.0, scene_per_m * 2.0),
        )
        if int(np.count_nonzero(model_mask)) <= 0:
            raise VirtualTwinBuildError("model leakage filter rejected all virtual-twin surfels after visual-edge refinement")
    else:
        registration = {
            **registration,
            "camera_visual_edge_refinement": {
                "status": "not_applicable",
                "reason": "no_initial_menon_scene_prior",
            },
        }
    accepted_points_world = points_world[model_mask]
    accepted_points_scene = points_scene[model_mask]
    accepted_pixels = pixels[model_mask]
    accepted_frame_indices = frame_indices[model_mask]
    accepted_colors = colors[model_mask]
    visibility_surfaces = _structural_visibility_surfaces(surfaces)
    surface_support_candidates = _surface_candidates_for_points(
        accepted_points_scene,
        surfaces,
        scene_per_m=scene_per_m,
    )
    mesh_vertices, mesh_indices, mesh_colors, mesh_triangle_surface_indices, surface_artifact_metrics = _assign_support_to_surfaces(
        accepted_points_scene,
        accepted_colors,
        surface_support_candidates,
        scene_per_m=scene_per_m,
        browser_point_budget=int(browser_point_budget),
        sample_pixels=accepted_pixels,
        sample_frame_indices=accepted_frame_indices,
        frames=frames,
        visibility_surfaces=visibility_surfaces,
        registration=registration,
    )
    mesh_texcoords, texture_atlas, texture_metrics = _bake_model_surface_texture(
        mesh_vertices,
        mesh_indices,
        mesh_triangle_surface_indices,
        surface_support_candidates,
        visibility_surfaces,
        frames,
        registration,
        texture_tile_px=int(texture_tile_px),
        min_texture_coverage=float(min_texture_coverage),
        texture_exposure=float(texture_exposure),
        texture_gamma=float(texture_gamma),
        texture_contrast=float(texture_contrast),
    )
    surface_artifact_metrics = {
        **surface_artifact_metrics,
        "artifact_type": "textured_model_surface_mesh",
        "texture": texture_metrics,
    }
    point_glb_stride = (
        int(np.ceil(accepted_points_scene.shape[0] / float(max(1, int(browser_point_budget)))))
        if accepted_points_scene.shape[0] > int(browser_point_budget)
        else 1
    )
    point_glb_points = accepted_points_scene[::point_glb_stride]
    point_glb_colors = accepted_colors[::point_glb_stride]

    write_textured_mesh_glb(revision_dir / "surfaces.glb", mesh_vertices, mesh_indices, mesh_texcoords, texture_atlas)
    write_points_glb(revision_dir / "points.glb", point_glb_points, point_glb_colors)
    write_points_ply(revision_dir / "points.ply", points_world, colors)
    write_points_npz(
        revision_dir / "points.npz",
        points_world,
        colors,
        scene_points=points_scene,
        accepted_scene_mask=model_mask.astype(np.uint8),
    )

    tracking_alignment = {
        "revision_id": revision_id,
        "camera": camera_label,
        "pose_correction": registration,
        "floor_plane": {
            "frame": "backend_world_m",
            "normal": [0.0, 1.0, 0.0],
            "offset": -float(np.median(floor_y_values)) if floor_y_values else 0.0,
            "floor_y": float(np.median(floor_y_values)) if floor_y_values else 0.0,
        },
        "image_to_floor_lookup": {
            "mode": "shadow_readback",
            "status": "not_applied_to_live_tracking",
        },
        "valid_tracking_footprint": _points_bbox_xz(accepted_points_world),
        "depth_ray_correction_diagnostics": {
            "source": "MapAnything+ZeroPlane fused offline bundle",
            "registration_status": registration.get("status"),
            "correspondence_count": registration.get("correspondence_count"),
            "room_model_leakage_ratio": leakage_metrics["ratio"],
        },
    }

    median_residuals = [float(p["median_residual_m"]) for p in all_plane_json]
    p90_residuals = [float(p["p90_residual_m"]) for p in all_plane_json]
    coverage_values = [float(row["mapanything_valid_depth_coverage"]) for row in frame_rows]
    metrics = {
        "revision_id": revision_id,
        "camera": camera_label,
        "frame_count": len(frame_rows),
        "accepted_plane_count": len(all_plane_json),
        "mapanything_valid_depth_coverage_min": float(np.min(coverage_values)) if coverage_values else 0.0,
        "mapanything_valid_depth_coverage_median": float(np.median(coverage_values)) if coverage_values else 0.0,
        "rgb_colorfulness_min": float(np.min(colorfulness_values)) if colorfulness_values else 0.0,
        "rgb_colorfulness_median": median_colorfulness,
        "plane_residual_median_m": float(np.median(median_residuals)) if median_residuals else None,
        "plane_residual_p90_m": float(np.percentile(p90_residuals, 90.0)) if p90_residuals else None,
        "room_model_leakage_ratio": float(leakage_metrics["ratio"]),
        "room_model_leakage": leakage_metrics,
        "browser_render_budget": {
            "point_budget": int(browser_point_budget),
            "source_surfel_count": int(points_world.shape[0]),
            "accepted_scene_point_count": int(accepted_points_scene.shape[0]),
            "served_glb_point_count": int(point_glb_points.shape[0]),
            "served_glb_vertex_count": int(mesh_vertices.shape[0]),
            "served_glb_triangle_count": int(mesh_indices.shape[0] // 3),
            "decimation_stride": int(surface_artifact_metrics["source_sample_stride"]),
            "artifact_type": "textured_model_surface_mesh",
            "texture": texture_metrics,
            "model_surface_projection": surface_artifact_metrics,
            "point_cloud": {
                "artifact_type": "scene_point_cloud",
                "coordinate_frame": "menon_scene_units",
                "source_point_count": int(accepted_points_scene.shape[0]),
                "served_glb_point_count": int(point_glb_points.shape[0]),
                "source_sample_stride": int(point_glb_stride),
            },
        },
        "registration": {
            "position_rmse_scene_units": registration.get("position_rmse_scene_units"),
            "median_normal_error_deg": registration.get("median_normal_error_deg"),
            "p90_normal_error_deg": registration.get("p90_normal_error_deg"),
            "max_normal_error_deg": registration.get("max_normal_error_deg"),
            "correspondence_count": registration.get("correspondence_count"),
            "method": registration.get("method"),
            "scene_per_m": registration.get("scene_per_m"),
            "visible_surface_refinement": registration.get("visible_surface_refinement"),
            "surface_refinement": registration.get("surface_refinement"),
            "camera_visual_edge_refinement": registration.get("camera_visual_edge_refinement"),
            "visual_alignment_correction": registration.get("visual_alignment_correction"),
            "camera_scene_anchor": registration.get("camera_scene_anchor"),
        },
        "per_frame": fusion_metrics,
        "gates": {
            "mapanything_min_coverage_pass": bool(min(coverage_values) >= float(min_mapanything_coverage)) if coverage_values else False,
            "rgb_texture_colorfulness_pass": bool(median_colorfulness >= float(min_texture_colorfulness)),
            "zeroplane_min_planes_pass": bool(len(all_plane_json) >= int(min_zero_planes)),
            "registration_min_correspondences_pass": bool(
                int(registration.get("correspondence_count") or 0) >= int(min_registration_correspondences)
            ),
            "registration_normal_median_pass": bool(float(registration.get("median_normal_error_deg") or 180.0) <= 5.0),
            "registration_normal_p90_pass": bool(float(registration.get("p90_normal_error_deg") or 180.0) <= 10.0),
            "room_model_leakage_pass": bool(float(leakage_metrics["ratio"]) <= float(max_room_model_leakage)),
            "shadow_tracking_only": True,
        },
    }

    artifacts = {
        "manifest": "manifest.json",
        "planes": "planes.json",
        "surfaces_glb": "surfaces.glb",
        "points_glb": "points.glb",
        "points_ply": "points.ply",
        "points_npz": "points.npz",
        "tracking_alignment": "tracking_alignment.json",
        "metrics": "metrics.json",
    }
    manifest = {
        "schema": "noesis.virtual_twin.revision.v1",
        "revision_id": revision_id,
        "camera": camera_label,
        "created_ts_us": int(time.time() * 1_000_000),
        "source_frame_ids": source_frame_ids,
        "calibration_fingerprints": calibration_fingerprints,
        "model_fingerprints": {
            "menon_structural_obj": {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
            },
            "mapanything": dict(mapanything_refs or {}),
            "zeroplane": dict(zeroplane_refs or {}),
            "scene_prior": dict(scene_prior_refs or {}),
        },
        "source_file_refs": {
            "menon_structural_obj": str(model_path),
            "frames": frame_rows,
        },
        "coordinate_frames": {
            "points_ply": "backend_world_m",
            "points_glb": "menon_scene_units_decimated_point_cloud",
            "surfaces_glb": "menon_scene_units_textured_model_surface_mesh",
            "planes": "backend_world_m_with_image_masks",
        },
        "evidence_dirs": {
            "keyframes": "keyframes",
            "mapanything": "mapanything",
        },
        "artifacts": artifacts,
    }
    planes_json = {
        "schema": "noesis.virtual_twin.planes.v1",
        "revision_id": revision_id,
        "camera": camera_label,
        "planes": all_plane_json,
    }
    write_json(revision_dir / "manifest.json", manifest)
    write_json(revision_dir / "planes.json", planes_json)
    write_json(revision_dir / "tracking_alignment.json", tracking_alignment)
    write_json(revision_dir / "metrics.json", metrics)
    if update_latest:
        store.write_latest_revision_id(revision_id)
    return {
        "revision_id": revision_id,
        "revision_dir": str(revision_dir),
        "manifest": manifest,
        "metrics": metrics,
        "tracking_alignment": tracking_alignment,
    }


__all__ = [
    "VirtualTwinBuildError",
    "VirtualTwinFrameInput",
    "build_virtual_twin_revision",
    "calibration_fingerprint",
    "revision_id_from_clock",
    "rgb_colorfulness",
    "sha256_file",
]
