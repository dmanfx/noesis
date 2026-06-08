from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PlaneCandidate:
    frame_id: str
    plane_id: str
    mask: np.ndarray
    planar_depth: np.ndarray | None = None
    normal: np.ndarray | None = None
    offset: float | None = None
    confidence: float = 1.0
    semantic_label: str | None = None


@dataclass(frozen=True)
class FusedPlane:
    frame_id: str
    plane_id: str
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    support_pixels: int
    confidence: float
    median_residual_m: float
    p90_residual_m: float
    raw_median_residual_m: float
    mask_rle: dict[str, Any]
    polygon: list[list[float]]
    semantic_label: str
    depth_support: dict[str, Any]
    normal_support: dict[str, Any]
    fusion_score: float


@dataclass(frozen=True)
class FusionResult:
    fused_depth: np.ndarray
    fused_mask: np.ndarray
    fused_points_camera: np.ndarray
    fused_pixels: np.ndarray
    planes: list[FusedPlane]
    metrics: dict[str, Any]


def intrinsics_matrix(intrinsics: Mapping[str, Any] | Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    if isinstance(intrinsics, Mapping):
        if "K" in intrinsics:
            value = intrinsics["K"]
            mat = np.asarray(value, dtype=np.float64)
            if mat.size == 9:
                return mat.reshape(3, 3)
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    mat = np.asarray(intrinsics, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"intrinsics must be a 3x3 matrix, got {mat.shape}")
    return mat


def encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(mask, dtype=bool)
    flat = arr.astype(np.uint8).ravel(order="C")
    counts: list[int] = []
    run_value = 0
    run_len = 0
    for value in flat:
        value_int = int(value)
        if value_int == run_value:
            run_len += 1
        else:
            counts.append(run_len)
            run_value = value_int
            run_len = 1
    counts.append(run_len)
    return {"size": [int(arr.shape[0]), int(arr.shape[1])], "order": "C", "counts": counts}


def decode_mask_rle(payload: Mapping[str, Any]) -> np.ndarray:
    size = payload.get("size")
    counts = payload.get("counts")
    if not isinstance(size, Sequence) or len(size) != 2 or not isinstance(counts, Sequence):
        raise ValueError("mask RLE requires size=[h,w] and counts")
    h = int(size[0])
    w = int(size[1])
    values: list[int] = []
    value = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError("mask RLE counts must be non-negative")
        values.extend([value] * count)
        value = 1 - value
    arr = np.asarray(values[: h * w], dtype=bool)
    if arr.size != h * w:
        raise ValueError("mask RLE did not match declared size")
    return arr.reshape((h, w), order="C")


def resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.shape == shape:
        return arr
    import cv2

    return cv2.resize(arr.astype(np.uint8), (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST).astype(bool)


def resize_float_map(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == shape:
        return arr
    import cv2

    return cv2.resize(arr, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def compute_depth_normals_camera(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    intrinsics: Mapping[str, Any] | Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Compute camera-space normals from a dense depth map.

    The orientation matches the MapAnything runtime normals payload: normals
    are view-facing in camera coordinates, so a fronto-parallel wall has
    approximately negative Z.
    """

    z = np.asarray(depth, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"depth must be 2D, got {z.shape}")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != z.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match depth shape {z.shape}")
    valid = valid & np.isfinite(z) & (z > 0.0)
    k = intrinsics_matrix(intrinsics)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if abs(fx) <= 1e-9 or abs(fy) <= 1e-9:
        raise ValueError("intrinsics focal lengths must be non-zero")

    height, width = z.shape
    grid_u, grid_v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    x_cam = (grid_u - float(cx)) * z / float(fx)
    y_cam = (grid_v - float(cy)) * z / float(fy)
    points = np.stack([x_cam, y_cam, z], axis=-1).astype(np.float32, copy=False)
    points[~valid] = np.nan

    d_pdx = np.zeros_like(points)
    d_pdy = np.zeros_like(points)
    if width > 1:
        d_pdx[:, 1:-1] = points[:, 2:] - points[:, :-2]
        d_pdx[:, 0] = points[:, 1] - points[:, 0]
        d_pdx[:, -1] = points[:, -1] - points[:, -2]
    if height > 1:
        d_pdy[1:-1] = points[2:] - points[:-2]
        d_pdy[0] = points[1] - points[0]
        d_pdy[-1] = points[-1] - points[-2]

    normals = np.cross(d_pdx, d_pdy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.divide(normals, norm, out=np.zeros_like(normals), where=(norm > 1e-6))
    good = np.isfinite(normals).all(axis=-1) & valid
    normals[~good] = 0.0
    flip = normals[..., 2] > 0.0
    normals[flip] *= -1.0
    return normals.astype(np.float32, copy=False)


def mask_bbox_polygon(mask: np.ndarray) -> list[list[float]]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return []
    x0 = float(np.min(xs))
    x1 = float(np.max(xs))
    y0 = float(np.min(ys))
    y1 = float(np.max(ys))
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def backproject_depth(
    depth: np.ndarray,
    intrinsics: Mapping[str, Any] | Sequence[Sequence[float]] | np.ndarray,
    *,
    mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    min_confidence: float = 0.0,
    pixel_step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    depth_np = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_np) & (depth_np > 0.0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if confidence is not None:
        valid &= np.asarray(confidence, dtype=np.float32) >= float(min_confidence)
    step = max(1, int(pixel_step))
    if step > 1:
        grid = np.zeros_like(valid, dtype=bool)
        grid[::step, ::step] = True
        valid &= grid

    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    k = intrinsics_matrix(intrinsics)
    z = depth_np[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    pixels = np.stack([xs, ys], axis=1).astype(np.int32)
    return points, pixels


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (mat @ hom.T).T[:, :3]
    return out.astype(np.float32)


def transform_plane(normal: np.ndarray, offset: float, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    d = float(offset)
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    plane = np.asarray([n[0], n[1], n[2], d], dtype=np.float64)
    transformed = np.linalg.inv(mat).T @ plane
    out_n = transformed[:3]
    norm = float(np.linalg.norm(out_n))
    if norm <= 1e-12:
        raise ValueError("degenerate transformed plane")
    return (out_n / norm).astype(np.float32), float(transformed[3] / norm)


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("plane fit requires at least three 3D points")
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("degenerate plane normal")
    normal = normal / normal_norm
    offset = -float(np.dot(normal, centroid))
    residuals = np.abs(pts @ normal + offset)
    return normal.astype(np.float32), offset, residuals.astype(np.float32)


def fit_robust_plane(
    points: np.ndarray,
    *,
    threshold_m: float = 0.05,
    max_iterations: int = 128,
    min_support: int = 64,
    seed: int = 17,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("robust plane fit requires at least three points")
    if pts.shape[0] < max(3, int(min_support)):
        normal, offset, residuals = fit_plane_svd(pts)
        inliers = residuals <= float(threshold_m)
        return normal, offset, inliers.astype(bool), residuals

    rng = np.random.default_rng(int(seed))
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_median = math.inf
    threshold = float(threshold_m)
    n_points = pts.shape[0]
    for _ in range(max(1, int(max_iterations))):
        idx = rng.choice(n_points, size=3, replace=False)
        tri = pts[idx]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, tri[0]))
        residuals = np.abs(pts @ normal + offset)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residuals[inliers])) if count else math.inf
        if count > best_count or (count == best_count and median < best_median):
            best_inliers = inliers
            best_count = count
            best_median = median

    if best_inliers is None or int(np.count_nonzero(best_inliers)) < 3:
        normal, offset, residuals = fit_plane_svd(pts)
        return normal, offset, np.ones(pts.shape[0], dtype=bool), residuals

    if int(np.count_nonzero(best_inliers)) >= max(3, int(min_support)):
        normal, offset, _ = fit_plane_svd(pts[best_inliers])
    else:
        normal, offset, _ = fit_plane_svd(pts)
    residuals = np.abs(pts @ np.asarray(normal, dtype=np.float64) + float(offset))
    inliers = residuals <= threshold
    return normal, offset, inliers.astype(bool), residuals.astype(np.float32)


def plane_depth_for_pixels(
    normal: np.ndarray,
    offset: float,
    intrinsics: Mapping[str, Any] | Sequence[Sequence[float]] | np.ndarray,
    pixels: np.ndarray,
) -> np.ndarray:
    pix = np.asarray(pixels, dtype=np.float64)
    if pix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    k = intrinsics_matrix(intrinsics)
    rays = np.stack(
        [
            (pix[:, 0] - k[0, 2]) / k[0, 0],
            (pix[:, 1] - k[1, 2]) / k[1, 1],
            np.ones((pix.shape[0],), dtype=np.float64),
        ],
        axis=1,
    )
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -float(offset) / denom
    z[~np.isfinite(z)] = np.nan
    z[np.abs(denom) < 1e-9] = np.nan
    z[z <= 0.0] = np.nan
    return z.astype(np.float32)


def classify_plane_from_normal(normal: np.ndarray) -> str:
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-9:
        return "unknown"
    n = n / norm
    if abs(float(n[1])) >= 0.82:
        return "floor_or_ceiling"
    if abs(float(n[1])) <= 0.30:
        return "wall"
    return "sloped_plane"


def _unit_vector(value: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return arr / norm


def _normal_support_summary(
    *,
    plane_normal: np.ndarray,
    candidate_mask: np.ndarray,
    map_normals_camera: np.ndarray | None,
    map_normals_valid: np.ndarray | None,
    min_fraction: float,
    min_samples: int,
) -> dict[str, Any]:
    support_pixels = int(np.count_nonzero(candidate_mask))
    base = {
        "status": "not_provided",
        "sample_count": 0,
        "valid_fraction": 0.0,
        "mapanything_normal_camera": None,
        "normal_agreement_dot_median": None,
        "normal_angular_error_deg_median": None,
        "normal_angular_error_deg_p90": None,
        "normal_variance": None,
        "coherent_fraction": None,
    }
    if map_normals_camera is None:
        return base
    normals = np.asarray(map_normals_camera, dtype=np.float32)
    if normals.ndim != 3 or normals.shape[:2] != candidate_mask.shape or normals.shape[2] < 3:
        return {**base, "status": "bad_shape"}
    normal_mask = np.asarray(candidate_mask, dtype=bool)
    if map_normals_valid is not None:
        valid_arr = np.asarray(map_normals_valid, dtype=bool)
        if valid_arr.shape != candidate_mask.shape:
            return {**base, "status": "bad_valid_shape"}
        normal_mask &= valid_arr
    mags = np.linalg.norm(normals[:, :, :3], axis=-1)
    normal_mask &= np.isfinite(normals[:, :, :3]).all(axis=-1) & np.isfinite(mags) & (mags > 0.20)
    sample_count = int(np.count_nonzero(normal_mask))
    valid_fraction = float(sample_count / max(1, support_pixels))
    if sample_count <= 0:
        return {**base, "status": "no_valid_normals", "valid_fraction": valid_fraction}
    plane_unit = _unit_vector(plane_normal)
    if plane_unit is None:
        return {
            **base,
            "status": "bad_plane_normal",
            "sample_count": sample_count,
            "valid_fraction": valid_fraction,
        }
    samples = normals[:, :, :3][normal_mask].astype(np.float64)
    sample_norms = np.linalg.norm(samples, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        samples = np.divide(samples, sample_norms, out=np.zeros_like(samples), where=(sample_norms > 1e-9))
    dots = samples @ plane_unit
    finite = np.isfinite(dots)
    if not np.any(finite):
        return {
            **base,
            "status": "no_finite_normals",
            "sample_count": sample_count,
            "valid_fraction": valid_fraction,
        }
    samples = samples[finite]
    dots = dots[finite]
    sample_count = int(samples.shape[0])
    valid_fraction = float(sample_count / max(1, support_pixels))
    signs = np.where(dots < 0.0, -1.0, 1.0)
    aligned = samples * signs[:, None]
    abs_dots = np.clip(np.abs(dots), 0.0, 1.0)
    angles = np.degrees(np.arccos(abs_dots))
    median = np.median(aligned, axis=0)
    median_unit = _unit_vector(median)
    if median_unit is None:
        median_unit = _unit_vector(np.mean(aligned, axis=0))
    if median_unit is not None and float(np.dot(median_unit, plane_unit)) < 0.0:
        median_unit = -median_unit
    mean_aligned = np.mean(aligned, axis=0)
    variance = 1.0 - min(1.0, float(np.linalg.norm(mean_aligned)))
    coherent_fraction = float(np.count_nonzero(angles <= 35.0) / max(1, angles.size))
    median_angle = float(np.median(angles))
    p90_angle = float(np.percentile(angles, 90.0))
    if sample_count < max(1, int(min_samples)) or valid_fraction < float(min_fraction):
        status = "insufficient_support"
    elif median_angle <= 35.0 and coherent_fraction >= 0.45:
        status = "agree"
    else:
        status = "disagree"
    return {
        "status": status,
        "sample_count": sample_count,
        "valid_fraction": valid_fraction,
        "mapanything_normal_camera": [float(x) for x in median_unit] if median_unit is not None else None,
        "normal_agreement_dot_median": float(np.median(abs_dots)),
        "normal_angular_error_deg_median": median_angle,
        "normal_angular_error_deg_p90": p90_angle,
        "normal_variance": float(np.clip(variance, 0.0, 1.0)),
        "coherent_fraction": coherent_fraction,
    }


def _normal_support_rejects_plane(summary: Mapping[str, Any], *, max_angle_deg: float) -> bool:
    if str(summary.get("status") or "") != "disagree":
        return False
    median_angle = summary.get("normal_angular_error_deg_median")
    p90_angle = summary.get("normal_angular_error_deg_p90")
    coherent = summary.get("coherent_fraction")
    valid_fraction = float(summary.get("valid_fraction") or 0.0)
    sample_count = int(summary.get("sample_count") or 0)
    if sample_count < 24 or valid_fraction < 0.08:
        return False
    median_bad = median_angle is not None and float(median_angle) >= float(max_angle_deg)
    p90_bad = p90_angle is not None and float(p90_angle) >= max(82.0, float(max_angle_deg) + 10.0)
    coherent_bad = coherent is not None and float(coherent) < 0.20
    return bool(median_bad or (p90_bad and coherent_bad))


def fuse_mapanything_with_planes(
    *,
    frame_id: str,
    map_depth: np.ndarray,
    map_confidence: np.ndarray,
    map_mask: np.ndarray,
    map_normals_camera: np.ndarray | None = None,
    map_normals_valid: np.ndarray | None = None,
    intrinsics: Mapping[str, Any] | Sequence[Sequence[float]] | np.ndarray,
    plane_candidates: Sequence[PlaneCandidate],
    min_confidence: float = 0.5,
    plane_fit_threshold_m: float = 0.06,
    min_plane_support: int = 96,
    surfel_pixel_step: int = 2,
    min_normal_support_fraction: float = 0.08,
    max_normal_angle_deg: float = 70.0,
) -> FusionResult:
    depth = np.asarray(map_depth, dtype=np.float32)
    conf = np.asarray(map_confidence, dtype=np.float32)
    valid = np.asarray(map_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0) & np.isfinite(conf)
    valid &= conf >= float(min_confidence)
    fused_depth = depth.copy()
    fused_mask = valid.copy()
    replaced_mask = np.zeros_like(valid, dtype=bool)
    planes: list[FusedPlane] = []
    raw_residual_values: list[float] = []
    fused_residual_values: list[float] = []
    normal_angle_values: list[float] = []
    normal_supported_plane_count = 0
    normal_disagreement_plane_count = 0
    normal_rejected_plane_count = 0
    insufficient_support_rejected_count = 0

    for index, candidate in enumerate(plane_candidates):
        candidate_full_mask = resize_bool_mask(candidate.mask, depth.shape)
        candidate_mask = candidate_full_mask & valid
        support = int(np.count_nonzero(candidate_mask))
        if support < max(3, int(min_plane_support)):
            insufficient_support_rejected_count += 1
            continue
        points, pixels = backproject_depth(
            depth,
            intrinsics,
            mask=candidate_mask,
            confidence=conf,
            min_confidence=min_confidence,
            pixel_step=1,
        )
        if points.shape[0] < max(3, int(min_plane_support)):
            insufficient_support_rejected_count += 1
            continue
        normal, offset, inliers, residuals = fit_robust_plane(
            points,
            threshold_m=float(plane_fit_threshold_m),
            min_support=int(min_plane_support),
            seed=17 + index,
        )
        if candidate.normal is not None:
            candidate_normal = np.asarray(candidate.normal, dtype=np.float64).reshape(3)
            if float(np.dot(candidate_normal, normal)) < 0.0:
                normal = (-np.asarray(normal)).astype(np.float32)
                offset = -float(offset)
                residuals = np.abs(points @ np.asarray(normal, dtype=np.float64) + float(offset)).astype(np.float32)

        plane_z = plane_depth_for_pixels(normal, offset, intrinsics, pixels)
        ok = np.isfinite(plane_z) & (plane_z > 0.0)
        if not np.any(ok):
            continue
        observed_z = depth[pixels[:, 1], pixels[:, 0]].astype(np.float32, copy=False)
        depth_abs_error = np.abs(observed_z.astype(np.float64) - plane_z.astype(np.float64))
        depth_error_ok = np.isfinite(depth_abs_error) & ok
        depth_inlier_threshold = max(float(plane_fit_threshold_m) * 2.0, 0.08)
        depth_inlier_fraction = (
            float(np.count_nonzero(depth_error_ok & (depth_abs_error <= depth_inlier_threshold)))
            / max(1.0, float(np.count_nonzero(depth_error_ok)))
        )
        candidate_pixel_count = int(np.count_nonzero(candidate_full_mask))
        depth_support = {
            "support_pixels": support,
            "candidate_pixels": candidate_pixel_count,
            "valid_fraction": float(support / max(1, candidate_pixel_count)),
            "confidence_median": float(np.median(conf[candidate_mask])) if support else None,
            "depth_inlier_fraction": depth_inlier_fraction,
            "plane_depth_median_abs_error_m": (
                float(np.median(depth_abs_error[depth_error_ok])) if np.any(depth_error_ok) else None
            ),
            "plane_depth_p90_abs_error_m": (
                float(np.percentile(depth_abs_error[depth_error_ok], 90.0)) if np.any(depth_error_ok) else None
            ),
            "depth_inlier_threshold_m": float(depth_inlier_threshold),
        }
        normal_support = _normal_support_summary(
            plane_normal=normal,
            candidate_mask=candidate_mask,
            map_normals_camera=map_normals_camera,
            map_normals_valid=map_normals_valid,
            min_fraction=float(min_normal_support_fraction),
            min_samples=max(24, int(min_plane_support * 0.25)),
        )
        if normal_support.get("sample_count"):
            normal_supported_plane_count += 1
        if normal_support.get("normal_angular_error_deg_median") is not None:
            normal_angle_values.append(float(normal_support["normal_angular_error_deg_median"]))
        if str(normal_support.get("status") or "") == "disagree":
            normal_disagreement_plane_count += 1
        if _normal_support_rejects_plane(normal_support, max_angle_deg=float(max_normal_angle_deg)):
            normal_rejected_plane_count += 1
            continue
        px_ok = pixels[ok]
        fused_depth[px_ok[:, 1], px_ok[:, 0]] = plane_z[ok]
        replaced_mask[px_ok[:, 1], px_ok[:, 0]] = True
        fused_points, _ = backproject_depth(fused_depth, intrinsics, mask=candidate_mask, pixel_step=1)
        _, _, fused_residuals = fit_plane_svd(fused_points)
        raw_median = float(np.median(residuals)) if residuals.size else 0.0
        med = float(np.median(fused_residuals)) if fused_residuals.size else 0.0
        p90 = float(np.percentile(fused_residuals, 90.0)) if fused_residuals.size else 0.0
        inlier_ratio = float(np.count_nonzero(inliers)) / max(1.0, float(inliers.size))
        normal_status = str(normal_support.get("status") or "")
        if normal_status == "agree":
            normal_factor = 1.12
        elif normal_status == "disagree":
            normal_factor = 0.55
        elif normal_status in {"insufficient_support", "no_valid_normals", "bad_shape", "bad_valid_shape"}:
            normal_factor = 0.90
        else:
            normal_factor = 1.0
        depth_factor = float(np.clip(0.65 + (0.35 * depth_inlier_fraction), 0.2, 1.0))
        fusion_score = float(np.clip(float(candidate.confidence) * inlier_ratio * normal_factor * depth_factor, 0.0, 1.0))
        confidence = fusion_score
        centroid = np.mean(points[inliers] if np.any(inliers) else points, axis=0)
        semantic = candidate.semantic_label or classify_plane_from_normal(normal)
        planes.append(
            FusedPlane(
                frame_id=str(frame_id),
                plane_id=str(candidate.plane_id or f"plane_{index:02d}"),
                normal=np.asarray(normal, dtype=np.float32),
                offset=float(offset),
                centroid=np.asarray(centroid, dtype=np.float32),
                support_pixels=support,
                confidence=confidence,
                median_residual_m=med,
                p90_residual_m=p90,
                raw_median_residual_m=raw_median,
                mask_rle=encode_mask_rle(candidate_mask),
                polygon=mask_bbox_polygon(candidate_mask),
                semantic_label=semantic,
                depth_support=depth_support,
                normal_support=normal_support,
                fusion_score=fusion_score,
            )
        )
        raw_residual_values.extend(float(x) for x in np.ravel(residuals))
        fused_residual_values.extend(float(x) for x in np.ravel(fused_residuals))

    fused_mask |= replaced_mask
    points, pixels = backproject_depth(
        fused_depth,
        intrinsics,
        mask=fused_mask,
        confidence=None,
        min_confidence=0.0,
        pixel_step=max(1, int(surfel_pixel_step)),
    )
    coverage = float(np.count_nonzero(valid)) / float(valid.size) if valid.size else 0.0
    plane_coverage = float(np.count_nonzero(replaced_mask)) / max(1.0, float(np.count_nonzero(valid)))
    raw_med = float(np.median(raw_residual_values)) if raw_residual_values else None
    fused_med = float(np.median(fused_residual_values)) if fused_residual_values else None
    metrics = {
        "mapanything_valid_depth_coverage": coverage,
        "candidate_plane_count": len(plane_candidates),
        "accepted_plane_count": len(planes),
        "insufficient_support_rejected_plane_count": insufficient_support_rejected_count,
        "plane_pixel_coverage": plane_coverage,
        "raw_plane_median_residual_m": raw_med,
        "fused_plane_median_residual_m": fused_med,
        "fused_improved_raw_residual": (
            bool(fused_med <= raw_med) if raw_med is not None and fused_med is not None else None
        ),
        "surfel_count": int(points.shape[0]),
        "normal_fusion_status": "provided" if map_normals_camera is not None else "not_provided",
        "normal_supported_plane_count": normal_supported_plane_count,
        "normal_disagreement_plane_count": normal_disagreement_plane_count,
        "normal_rejected_plane_count": normal_rejected_plane_count,
        "normal_median_angular_error_deg": (
            float(np.median(normal_angle_values)) if normal_angle_values else None
        ),
    }
    return FusionResult(
        fused_depth=fused_depth.astype(np.float32),
        fused_mask=fused_mask.astype(bool),
        fused_points_camera=points.astype(np.float32),
        fused_pixels=pixels.astype(np.int32),
        planes=planes,
        metrics=metrics,
    )


__all__ = [
    "FusedPlane",
    "FusionResult",
    "PlaneCandidate",
    "backproject_depth",
    "classify_plane_from_normal",
    "compute_depth_normals_camera",
    "decode_mask_rle",
    "encode_mask_rle",
    "fit_plane_svd",
    "fit_robust_plane",
    "fuse_mapanything_with_planes",
    "intrinsics_matrix",
    "plane_depth_for_pixels",
    "resize_bool_mask",
    "resize_float_map",
    "transform_plane",
    "transform_points",
]
