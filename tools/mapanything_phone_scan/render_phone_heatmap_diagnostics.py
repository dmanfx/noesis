#!/usr/bin/env python3
"""Render Depth-panel-style diagnostics from phone-walk outputs only.

The comparison is deliberately kept out of Noesis/static-camera coordinates.
The integrated DA3 output is similarity-aligned to MapAnything using the
corresponding 48 phone camera poses, and both clouds are then levelled from the
phone reconstruction's floor plane. No fixed-camera image, depth map, cloud,
or calibration is read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.coordinate_frames import (  # noqa: E402
    CAMERA_LOCAL_RASTER_ORIENTATION,
    camera_ground_frame_from_camera_to_world,
    camera_local_raster_indices,
    transform_positions,
)


@dataclass
class PhoneCloud:
    points: np.ndarray
    colors: np.ndarray
    weights: np.ndarray
    ranges: np.ndarray
    camera_to_world: np.ndarray
    frame_zero_depth: np.ndarray
    frame_zero_rgb: np.ndarray
    presentation_camera_positions: np.ndarray | None = None


def _load_raw_phone_cloud(raw_root: Path, point_budget: int, provider: str) -> PhoneCloud:
    raw_paths = sorted(raw_root.glob("view_*.npz"))
    if len(raw_paths) < 2:
        raise ValueError(f"{provider} raw phone views are missing")
    per_view = max(1_000, point_budget // len(raw_paths))
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    first_depth = None
    first_rgb = None
    for path in raw_paths:
        with np.load(path) as row:
            world = np.asarray(row["world_points"], dtype=np.float32)
            depth = np.asarray(row["depth_z"], dtype=np.float32)
            confidence = np.asarray(row["confidence"], dtype=np.float32)
            mask = np.asarray(row["mask"], dtype=bool)
            rgb = np.asarray(row["model_rgb"])
            pose = np.asarray(row["camera_pose"], dtype=np.float64)
        valid = (
            mask
            & np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(confidence)
            & np.isfinite(world).all(axis=-1)
        )
        valid_conf = confidence[valid]
        threshold = float(np.percentile(valid_conf, 45.0))
        selected = np.flatnonzero((valid & (confidence >= threshold)).reshape(-1))
        stride = max(1, int(math.ceil(selected.size / per_view)))
        selected = selected[::stride][:per_view]
        rgb_u8 = np.clip(rgb * 255.0 if float(np.nanmax(rgb)) <= 1.5 else rgb, 0, 255).astype(np.uint8)
        conf_values = confidence.reshape(-1)[selected]
        conf_lo, conf_hi = np.percentile(valid_conf, (10.0, 95.0))
        normalized_conf = np.clip((conf_values - conf_lo) / max(conf_hi - conf_lo, 1e-8), 0.05, 1.0)
        points.append(world.reshape(-1, 3)[selected])
        colors.append(rgb_u8.reshape(-1, 3)[selected])
        weights.append(normalized_conf.astype(np.float32))
        ranges.append(depth.reshape(-1)[selected])
        poses.append(pose)
        if first_depth is None:
            first_depth = depth
            first_rgb = rgb_u8
    return PhoneCloud(
        points=np.concatenate(points),
        colors=np.concatenate(colors),
        weights=np.concatenate(weights),
        ranges=np.concatenate(ranges),
        camera_to_world=np.stack(poses),
        frame_zero_depth=np.asarray(first_depth),
        frame_zero_rgb=np.asarray(first_rgb),
    )


def _load_mapanything(
    scan_dir: Path,
    point_budget: int,
    raw_root: Path | None = None,
) -> PhoneCloud:
    return _load_raw_phone_cloud(
        raw_root or scan_dir / "outputs" / "raw", point_budget, "MapAnything"
    )


def _load_da3(
    scan_dir: Path,
    point_budget: int,
    raw_root: Path | None = None,
) -> PhoneCloud:
    return _load_raw_phone_cloud(
        raw_root or scan_dir / "da3_outputs" / "raw",
        point_budget,
        "integrated DA3",
    )


def _load_consensus(
    scan_dir: Path,
    point_budget: int,
    raw_root: Path | None = None,
) -> PhoneCloud:
    return _load_raw_phone_cloud(
        raw_root or scan_dir / "consensus_fusion" / "raw",
        point_budget,
        "Consensus Fusion",
    )


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return a proper Sim(3) mapping source to target (never a reflection)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise ValueError("Umeyama alignment produced an improper rotation")
    variance = float(np.sum(source_centered * source_centered) / source.shape[0])
    scale = float(np.sum(singular * np.diag(sign)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _apply_sim3(cloud: PhoneCloud, scale: float, rotation: np.ndarray, translation: np.ndarray) -> PhoneCloud:
    points = scale * (rotation @ cloud.points.T).T + translation
    poses = cloud.camera_to_world.copy()
    poses[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    poses[:, :3, 3] = scale * (rotation @ poses[:, :3, 3].T).T + translation
    return PhoneCloud(
        points=points.astype(np.float32),
        colors=cloud.colors,
        weights=cloud.weights,
        ranges=(cloud.ranges * scale).astype(np.float32),
        camera_to_world=poses,
        frame_zero_depth=(cloud.frame_zero_depth * scale).astype(np.float32),
        frame_zero_rgb=cloud.frame_zero_rgb,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag((1.0, -1.0, -1.0))
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _phone_floor_transform(cloud: PhoneCloud) -> tuple[np.ndarray, dict[str, float | list[float]]]:
    import open3d as o3d

    stride = max(1, cloud.points.shape[0] // 180_000)
    sampled = cloud.points[::stride].astype(np.float64)
    camera_up = _normalize(-np.mean(cloud.camera_to_world[:, :3, 1], axis=0))
    o3d.utility.random.seed(17)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled))
    plane, inliers = pcd.segment_plane(distance_threshold=0.06, ransac_n=3, num_iterations=1_600)
    normal = _normalize(np.asarray(plane[:3], dtype=np.float64))
    offset = float(plane[3]) / float(np.linalg.norm(plane[:3]))
    if float(np.dot(normal, camera_up)) < 0:
        normal = -normal
        offset = -offset
    rotation = _align_vectors(normal, np.array([0.0, 1.0, 0.0]))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[1, 3] = offset
    return transform, {
        "floor_normal_phone_world": normal.tolist(),
        "floor_plane_offset": offset,
        "floor_inlier_fraction": float(len(inliers) / sampled.shape[0]),
    }


def _apply_rigid(cloud: PhoneCloud, transform: np.ndarray) -> PhoneCloud:
    points = (transform[:3, :3] @ cloud.points.T).T + transform[:3, 3]
    poses = np.stack([transform @ pose for pose in cloud.camera_to_world])
    return PhoneCloud(
        points=points.astype(np.float32),
        colors=cloud.colors,
        weights=cloud.weights,
        ranges=cloud.ranges,
        camera_to_world=poses,
        frame_zero_depth=cloud.frame_zero_depth,
        frame_zero_rgb=cloud.frame_zero_rgb,
    )


def _camera_positions(cloud: PhoneCloud) -> np.ndarray:
    if cloud.presentation_camera_positions is not None:
        return np.asarray(cloud.presentation_camera_positions, dtype=np.float64)
    return np.asarray(cloud.camera_to_world[:, :3, 3], dtype=np.float64)


def _present_camera_ground(
    cloud: PhoneCloud,
    world_to_camera_local_display: np.ndarray,
) -> PhoneCloud:
    """Create a display-only cloud without transforming any pose rotation.

    The matrix is normally improper because OpenCV camera-down becomes
    camera-local height-up.  It is therefore applied only to point and path
    positions after all metric registration has completed.
    """

    matrix = np.asarray(world_to_camera_local_display, dtype=np.float64)
    return PhoneCloud(
        points=transform_positions(cloud.points, matrix).astype(np.float32),
        colors=cloud.colors,
        weights=cloud.weights,
        ranges=cloud.ranges,
        camera_to_world=cloud.camera_to_world,
        frame_zero_depth=cloud.frame_zero_depth,
        frame_zero_rgb=cloud.frame_zero_rgb,
        presentation_camera_positions=transform_positions(
            cloud.camera_to_world[:, :3, 3],
            matrix,
        ),
    )


def _shared_bounds(clouds: list[PhoneCloud]) -> tuple[float, float, float, float]:
    points = np.concatenate([cloud.points for cloud in clouds])
    cameras = np.concatenate([_camera_positions(cloud) for cloud in clouds])
    point_horizontal = points[:, [0, 2]]
    camera_horizontal = cameras[:, [0, 2]]
    low = np.minimum(
        np.percentile(point_horizontal, 0.6, axis=0),
        np.min(camera_horizontal, axis=0),
    ) - 0.35
    high = np.maximum(
        np.percentile(point_horizontal, 99.4, axis=0),
        np.max(camera_horizontal, axis=0),
    ) + 0.35
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def _rasterize(cloud: PhoneCloud, bounds: tuple[float, float, float, float], grid_res: float) -> dict[str, np.ndarray]:
    min_x, max_x, min_z, max_z = bounds
    cols = int(math.ceil((max_x - min_x) / grid_res))
    rows = int(math.ceil((max_z - min_z) / grid_res))
    x = cloud.points[:, 0]
    y = np.clip(cloud.points[:, 1], 0.0, 2.5)
    z = cloud.points[:, 2]
    valid = (
        np.isfinite(cloud.points).all(axis=1)
        & (x >= min_x) & (x < max_x)
        & (z >= min_z) & (z < max_z)
        & (cloud.points[:, 1] >= -0.15) & (cloud.points[:, 1] <= 2.7)
    )
    x = x[valid]
    y = y[valid]
    z = z[valid]
    weights = cloud.weights[valid]
    ranges = cloud.ranges[valid]
    colors = cloud.colors[valid]
    zi, xi, raster_valid = camera_local_raster_indices(
        x,
        z,
        min_x_m=min_x,
        min_z_m=min_z,
        resolution_m=grid_res,
        rows=rows,
        columns=cols,
    )
    if not np.all(raster_valid):
        raise ValueError("bounded camera-local points escaped their raster")
    xi = xi.astype(np.int32)
    zi = zi.astype(np.int32)
    index = (zi, xi)

    support = np.zeros((rows, cols), dtype=np.uint32)
    density = np.zeros((rows, cols), dtype=np.float32)
    weight_sum = np.zeros((rows, cols), dtype=np.float64)
    height_sum = np.zeros((rows, cols), dtype=np.float64)
    range_sum = np.zeros((rows, cols), dtype=np.float64)
    height_min = np.full((rows, cols), np.inf, dtype=np.float32)
    height_max = np.full((rows, cols), -np.inf, dtype=np.float32)
    floor_support = np.zeros((rows, cols), dtype=np.uint32)
    obstacle_support = np.zeros((rows, cols), dtype=np.uint32)
    color_sum = np.zeros((rows, cols, 3), dtype=np.float64)
    np.add.at(support, index, 1)
    np.add.at(density, index, 1.0)
    np.add.at(weight_sum, index, weights)
    np.add.at(height_sum, index, y * weights)
    np.add.at(range_sum, index, ranges * weights)
    np.minimum.at(height_min, index, y)
    np.maximum.at(height_max, index, y)
    np.add.at(floor_support, index, (y <= 0.12).astype(np.uint32))
    np.add.at(obstacle_support, index, ((y >= 0.35) & (y <= 1.3)).astype(np.uint32))
    for channel in range(3):
        np.add.at(color_sum[..., channel], index, colors[:, channel] * weights)

    observed = support > 0
    height = np.full((rows, cols), np.nan, dtype=np.float32)
    distance = np.zeros((rows, cols), dtype=np.float32)
    surface_rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    np.divide(height_sum, weight_sum, out=height, where=weight_sum > 1e-9)
    np.divide(range_sum, weight_sum, out=distance, where=weight_sum > 1e-9)
    for channel in range(3):
        channel_out = np.zeros((rows, cols), dtype=np.float64)
        np.divide(color_sum[..., channel], weight_sum, out=channel_out, where=weight_sum > 1e-9)
        surface_rgb[..., channel] = np.clip(channel_out, 0, 255).astype(np.uint8)
    height_min[~observed] = np.nan
    height_max[~observed] = np.nan
    density = np.log1p(density)
    density /= max(float(np.max(density)), 1e-9)

    inside = ndi.binary_fill_holes(ndi.binary_closing(observed, iterations=2))
    height_range = np.nan_to_num(height_max - height_min, nan=np.inf)
    obstacle = (
        inside & observed & (support >= 3) & (obstacle_support >= 2)
        & (height_min >= 0.25) & (height_range <= 0.22)
    )
    obstacle = ndi.binary_opening(obstacle, structure=np.ones((3, 3), dtype=bool))
    obstacle = ndi.binary_closing(obstacle, structure=np.ones((3, 3), dtype=bool))
    obstacle_height = np.zeros((rows, cols), dtype=np.float32)
    obstacle_height[obstacle] = np.clip(height_max[obstacle], 0.0, 1.8)
    walkable = inside & ~obstacle

    height_for_gradient = np.nan_to_num(height, nan=0.0)
    height_for_gradient[~observed] = 0.0
    gx = ndi.sobel(height_for_gradient, axis=1, mode="nearest")
    gz = ndi.sobel(height_for_gradient, axis=0, mode="nearest")
    gradient = np.hypot(gx, gz)
    p95 = float(np.percentile(gradient, 95.0))
    gradient = np.clip(gradient / max(p95, 1e-9), 0.0, 1.0).astype(np.float32)

    structural = surface_rgb.copy()
    structural[~inside] = (8, 10, 13)
    structural[walkable & ~observed] = (42, 50, 54)
    obstacle_edges = obstacle ^ ndi.binary_erosion(obstacle)
    floor_edges = inside ^ ndi.binary_erosion(inside)
    structural[obstacle_edges] = (255, 148, 44)
    structural[floor_edges] = (90, 220, 255)

    camera_positions = _camera_positions(cloud)
    camera_z, camera_x, camera_valid = camera_local_raster_indices(
        camera_positions[:, 0],
        camera_positions[:, 2],
        min_x_m=min_x,
        min_z_m=min_z,
        resolution_m=grid_res,
        rows=rows,
        columns=cols,
    )
    camera_x = np.clip(camera_x, 0, cols - 1).astype(int)
    camera_z = np.clip(camera_z, 0, rows - 1).astype(int)
    if not np.all(camera_valid):
        raise ValueError("bounded camera path escaped its raster")
    path = np.stack((camera_x, camera_z), axis=1).astype(np.int32)
    cv2.polylines(structural, [path.reshape(-1, 1, 2)], False, (58, 255, 118), 1, cv2.LINE_AA)
    cv2.circle(structural, tuple(path[0]), 2, (70, 255, 70), -1)
    cv2.circle(structural, tuple(path[-1]), 2, (255, 90, 90), -1)
    return {
        "structural": structural,
        "density": density,
        "height": height,
        "height_agl": height,
        "distance": distance,
        "obstacle_height": obstacle_height,
        "walkable": walkable.astype(np.float32),
        "gradient": gradient,
        "observed": observed,
    }


def _colorize(values: np.ndarray, cmap: str, valid: np.ndarray | None = None, limits: tuple[float, float] | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if limits is None:
        finite = values[valid]
        lo, hi = (np.percentile(finite, (2.0, 98.0)) if finite.size else (0.0, 1.0))
    else:
        lo, hi = limits
    normalized = np.clip((values - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    rgba = plt.get_cmap(cmap)(normalized)
    rgb = np.round(rgba[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = (7, 8, 10)
    return rgb


def _depth_panel(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    limits = tuple(float(value) for value in np.percentile(values, (2.0, 98.0)))
    return _colorize(depth, "turbo_r", valid, limits)


def _overlay_panel(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    colored = _depth_panel(depth)
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_u8.shape[:2] != colored.shape[:2]:
        rgb_u8 = cv2.resize(rgb_u8, (colored.shape[1], colored.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(rgb_u8, 0.58, colored, 0.42, 0)


def _save_panel(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise OSError(f"failed to write {path}")


def _point_splat(
    cloud: PhoneCloud,
    bounds: tuple[float, float, float, float],
    cell_res_m: float,
    height_range_m: tuple[float, float],
) -> tuple[np.ndarray, dict[str, int]]:
    """Project raw samples without averaging their color or height.

    A 2D image still cannot display multiple samples at the exact same X/Z
    coordinate. In that case the highest-confidence phone-walk sample wins;
    separate height bands keep floor, furniture, and upper structure from
    being collapsed into one vertical-column average.
    """
    min_x, max_x, min_z, max_z = bounds
    cols = int(math.ceil((max_x - min_x) / cell_res_m))
    rows = int(math.ceil((max_z - min_z) / cell_res_m))
    low_y, high_y = height_range_m
    points = cloud.points
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= min_x)
        & (points[:, 0] < max_x)
        & (points[:, 2] >= min_z)
        & (points[:, 2] < max_z)
        & (points[:, 1] >= low_y)
        & (points[:, 1] < high_y)
    )
    selected = np.flatnonzero(valid)
    image = np.full((rows, cols, 3), (7, 8, 10), dtype=np.uint8)
    if selected.size == 0:
        return image, {"point_count": 0, "occupied_cell_count": 0}
    zi, xi, raster_valid = camera_local_raster_indices(
        points[selected, 0],
        points[selected, 2],
        min_x_m=min_x,
        min_z_m=min_z,
        resolution_m=cell_res_m,
        rows=rows,
        columns=cols,
    )
    if not np.all(raster_valid):
        raise ValueError("bounded point splat escaped its raster")
    xi = xi.astype(np.int32)
    zi = zi.astype(np.int32)
    # Low-confidence samples are written first, so a collision is resolved by
    # an actual phone-walk sample rather than an average or synthetic surface.
    order = np.argsort(cloud.weights[selected], kind="stable")
    image[zi[order], xi[order]] = cloud.colors[selected[order]]
    occupied = np.unique(zi.astype(np.int64) * cols + xi.astype(np.int64)).size
    return image, {
        "point_count": int(selected.size),
        "occupied_cell_count": int(occupied),
    }


def _render_point_preserving_layers(
    name: str,
    cloud: PhoneCloud,
    bounds: tuple[float, float, float, float],
    cell_res_m: float,
    provider_dir: Path,
) -> tuple[str, dict[str, dict[str, int]]]:
    bands = (
        ("All accepted samples", "all", (-0.15, 2.70)),
        ("Floor", "floor", (-0.15, 0.20)),
        ("Low objects", "low_objects", (0.20, 0.75)),
        ("Furniture", "furniture", (0.75, 1.40)),
        ("Upper structure", "upper_structure", (1.40, 2.70)),
    )
    panels: list[tuple[str, np.ndarray]] = []
    metrics: dict[str, dict[str, int]] = {}
    resolution_slug = f"{cell_res_m * 100:g}cm".replace(".", "p")
    for title, slug, height_range in bands:
        image, layer_metrics = _point_splat(
            cloud, bounds, cell_res_m, height_range
        )
        panels.append((title, image))
        metrics[slug] = layer_metrics
        _save_panel(provider_dir / f"point_splat_{slug}_{resolution_slug}.png", image)

    figure, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0d1014")
    figure.suptitle(
        f"{name} · raw phone-walk point layers · {cell_res_m * 100:.1f} cm cells",
        color="white",
        fontsize=18,
        y=0.985,
    )
    for axis, (title, image) in zip(axes.ravel(), panels):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, color="white", fontsize=12)
        axis.axis("off")
    legend_axis = axes.ravel()[-1]
    legend_axis.set_facecolor("#0d1014")
    legend_axis.text(
        0.03,
        0.90,
        "No height/color averaging\nNo morphology or hole filling\nHighest-confidence phone sample wins a collision\nStatic-camera points are not included",
        transform=legend_axis.transAxes,
        color="white",
        fontsize=12,
        va="top",
        linespacing=1.6,
    )
    legend_axis.axis("off")
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95), h_pad=1.7)
    output_path = provider_dir / f"phone_walk_point_layers_{resolution_slug}.png"
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return str(output_path), metrics


def _render_provider(
    name: str,
    cloud: PhoneCloud,
    grids: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, str]:
    observed = grids["observed"]
    panels: list[tuple[str, np.ndarray]] = [
        ("Camera Depth · Primary", _depth_panel(cloud.frame_zero_depth)),
        ("Exact Phone Frame RGB + Depth", _overlay_panel(cloud.frame_zero_rgb, cloud.frame_zero_depth)),
        ("Observed Height Floorplan · Primary", _colorize(grids["height"], "inferno", observed)),
        ("Structural Composite (Diagnostic)", grids["structural"]),
        ("Density (Grayscale)", _colorize(grids["density"], "gray", observed, (0.0, 1.0))),
        ("Raw Height (Inferno)", _colorize(grids["height"], "inferno", observed)),
        ("Raw Height (Contrast)", _colorize(grids["height"], "turbo", observed)),
        ("Height Above Floor", _colorize(grids["height_agl"], "turbo", observed, (0.0, 1.2))),
        ("Distance (Viridis)", _colorize(grids["distance"], "viridis", observed)),
        ("Obstacle Height (Clean)", _colorize(grids["obstacle_height"], "inferno", grids["obstacle_height"] > 0, (0.0, 1.8))),
        ("Walkable (Binary)", _colorize(grids["walkable"], "gray", grids["walkable"] >= 0, (0.0, 1.0))),
        ("Gradient (Edges)", _colorize(grids["gradient"], "viridis", observed, (0.0, 1.0))),
    ]
    provider_dir = output_dir / name.lower().replace(" ", "_")
    provider_dir.mkdir(parents=True, exist_ok=True)
    for title, image in panels:
        slug = title.lower().replace(" · ", "_").replace(" ", "_").replace("(", "").replace(")", "")
        _save_panel(provider_dir / f"{slug}.png", image)

    figure, axes = plt.subplots(4, 3, figsize=(15, 18), facecolor="#0d1014")
    figure.suptitle(
        f"{name} · phone-walk only · no static-camera data",
        color="white",
        fontsize=18,
        y=0.992,
    )
    for index, (axis, (title, image)) in enumerate(zip(axes.ravel(), panels)):
        axis.imshow(image)
        axis.set_title(title, color="white", fontsize=11)
        axis.axis("off")
    figure.text(
        0.018,
        0.715,
        "Diagnostic layers",
        color="#7bdcff",
        fontsize=15,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.5,
        0.008,
        "Green path: phone trajectory · green dot: start · red dot: end · black: unknown",
        color="#bcc7d1",
        fontsize=10,
        ha="center",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.975), h_pad=2.0)
    montage = provider_dir / "phone_only_heatmap_diagnostics.png"
    figure.savefig(montage, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return {"montage": str(montage), "panels": str(provider_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--point-budget", type=int, default=720_000)
    parser.add_argument("--grid-res-m", type=float, default=0.10)
    parser.add_argument("--point-splat-res-m", type=float, default=0.025)
    parser.add_argument("--mapanything-raw", type=Path)
    parser.add_argument("--da3-raw", type=Path)
    parser.add_argument("--consensus-raw", type=Path)
    args = parser.parse_args()
    scan_dir = args.scan_dir.resolve()
    output_dir = (args.output_dir or scan_dir / "phone_only_heatmap_diagnostics").resolve()

    mapanything = _load_mapanything(scan_dir, args.point_budget, args.mapanything_raw)
    da3_raw = _load_da3(scan_dir, args.point_budget, args.da3_raw)
    consensus_raw = _load_consensus(scan_dir, args.point_budget, args.consensus_raw)
    scale, rotation, translation = _umeyama(
        da3_raw.camera_to_world[:, :3, 3],
        mapanything.camera_to_world[:, :3, 3],
    )
    da3 = _apply_sim3(da3_raw, scale, rotation, translation)
    consensus_scale, consensus_rotation, consensus_translation = _umeyama(
        consensus_raw.camera_to_world[:, :3, 3],
        mapanything.camera_to_world[:, :3, 3],
    )
    consensus = _apply_sim3(
        consensus_raw,
        consensus_scale,
        consensus_rotation,
        consensus_translation,
    )
    floor_transform, floor_metrics = _phone_floor_transform(mapanything)
    mapanything = _apply_rigid(mapanything, floor_transform)
    da3 = _apply_rigid(da3, floor_transform)
    consensus = _apply_rigid(consensus, floor_transform)
    presentation_frame = camera_ground_frame_from_camera_to_world(
        mapanything.camera_to_world[0]
    )
    presentation_matrix = presentation_frame.world_to_camera_local_display_matrix(
        0.0
    )
    mapanything = _present_camera_ground(mapanything, presentation_matrix)
    da3 = _present_camera_ground(da3, presentation_matrix)
    consensus = _present_camera_ground(consensus, presentation_matrix)
    bounds = _shared_bounds([mapanything, da3, consensus])

    outputs = {}
    metrics = {
        "schema": "noesis.phone_walk.heatmap_diagnostics.v1",
        "phone_walk_only": True,
        "static_camera_data_used": False,
        "frame_count": int(mapanything.camera_to_world.shape[0]),
        "grid_res_m": float(args.grid_res_m),
        "point_splat_res_m": float(args.point_splat_res_m),
        "shared_bounds_camera_local_ground_m": list(bounds),
        "presentation": {
            "source_coordinate_frame": "leveled_phone_world_m",
            "target_coordinate_frame": "camera_local_ground_m",
            "reference_view_index": 0,
            "presentation_only": True,
            "backend_geometry_mutated": False,
            "linear_determinant": float(
                np.linalg.det(presentation_matrix[:3, :3])
            ),
            "world_to_camera_local_row_major": presentation_matrix.tolist(),
            "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
            "screen_right": "camera_right_positive_x",
            "screen_up": "camera_forward_positive_z",
        },
        "da3_to_mapanything_phone_pose_sim3": {
            "scale": scale,
            "rotation_row_major": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "consensus_to_mapanything_phone_pose_sim3": {
            "scale": consensus_scale,
            "rotation_row_major": consensus_rotation.tolist(),
            "translation": consensus_translation.tolist(),
        },
        "phone_floor": floor_metrics,
        "providers": {},
    }
    for name, cloud in (
        ("MapAnything", mapanything),
        ("DA3 Integrated", da3),
        ("Consensus Fusion", consensus),
    ):
        grids = _rasterize(cloud, bounds, args.grid_res_m)
        outputs[name] = _render_provider(name, cloud, grids, output_dir)
        provider_dir = Path(outputs[name]["panels"])
        point_layers, point_layer_metrics = _render_point_preserving_layers(
            name,
            cloud,
            bounds,
            args.point_splat_res_m,
            provider_dir,
        )
        outputs[name]["point_layers"] = point_layers
        metrics["providers"][name] = {
            "point_count": int(cloud.points.shape[0]),
            "observed_cells": int(np.count_nonzero(grids["observed"])),
            "walkable_cells": int(np.count_nonzero(grids["walkable"] > 0.5)),
            "obstacle_cells": int(np.count_nonzero(grids["obstacle_height"] > 0)),
            "point_preserving_layers": point_layer_metrics,
        }
    metrics["artifacts"] = outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics_manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
