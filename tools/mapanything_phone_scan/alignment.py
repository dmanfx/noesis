from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


ProgressCallback = Callable[[float, str], None]


class NoesisAlignmentError(RuntimeError):
    """Raised when a phone reconstruction cannot be admitted into Noesis world space."""


@dataclass(frozen=True)
class NoesisAlignmentSettings:
    camera_id: str
    target_revision: Path
    calibration_path: Path
    review_point_budget: int = 600_000
    registration_points_per_view: int = 5_000
    confidence_percentile: float = 55.0
    floor_distance_threshold_m: float = 0.06
    voxel_size_m: float = 0.10
    random_seed: int = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise NoesisAlignmentError(f"cannot normalize {name}")
    return result / norm


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source, name="source direction")
    target = _normalize(target, name="target direction")
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-9:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.cross(source, np.asarray([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) <= 1e-9:
            axis = np.cross(source, np.asarray([0.0, 0.0, 1.0]))
        axis = _normalize(axis, name="opposite-vector rotation axis")
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _yaw_transform(parameters: np.ndarray) -> np.ndarray:
    yaw_deg, tx, tz = (float(value) for value in parameters)
    angle = math.radians(yaw_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    transform[:3, 3] = [tx, 0.0, tz]
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ np.asarray(points, dtype=np.float64).T).T + transform[:3, 3]


def _load_phone_clouds(
    scan_dir: Path,
    outputs: dict[str, Any],
    settings: NoesisAlignmentSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_root = scan_dir / "outputs" / "raw"
    raw_paths = sorted(raw_root.glob("view_*.npz"))
    view_count = int(outputs.get("view_count") or 0)
    if view_count < 2 or len(raw_paths) != view_count:
        raise NoesisAlignmentError(
            f"expected {view_count} MapAnything raw views, found {len(raw_paths)}"
        )
    per_view_budget = max(1_000, settings.review_point_budget // view_count)
    registration_points: list[np.ndarray] = []
    review_points: list[np.ndarray] = []
    review_colors: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for path in raw_paths:
        with np.load(path) as row:
            required = {
                "world_points",
                "depth_z",
                "confidence",
                "mask",
                "camera_pose",
                "model_rgb",
            }
            if not required.issubset(row.files):
                missing = ", ".join(sorted(required.difference(row.files)))
                raise NoesisAlignmentError(f"{path.name} is missing {missing}")
            points = np.asarray(row["world_points"], dtype=np.float32)
            depth = np.asarray(row["depth_z"], dtype=np.float32)
            confidence = np.asarray(row["confidence"], dtype=np.float32)
            mask = np.asarray(row["mask"], dtype=bool)
            image = np.asarray(row["model_rgb"])
            pose = np.asarray(row["camera_pose"], dtype=np.float64)
        valid = (
            mask
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(confidence)
            & np.isfinite(points).all(axis=2)
        )
        flat_valid = np.flatnonzero(valid.reshape(-1))
        if flat_valid.size == 0:
            raise NoesisAlignmentError(f"{path.name} has no finite valid points")
        review_stride = max(1, int(math.ceil(flat_valid.size / per_view_budget)))
        review_indices = flat_valid[::review_stride][:per_view_budget]
        image_u8 = np.clip(
            image * 255.0 if float(np.nanmax(image)) <= 1.5 else image,
            0,
            255,
        ).astype(np.uint8)
        review_points.append(points.reshape((-1, 3))[review_indices])
        review_colors.append(image_u8.reshape((-1, 3))[review_indices])

        threshold = float(np.percentile(confidence[valid], settings.confidence_percentile))
        registration_valid = valid & (confidence >= threshold)
        selected = points[registration_valid]
        stride = max(1, selected.shape[0] // settings.registration_points_per_view)
        registration_points.append(selected[::stride][: settings.registration_points_per_view])
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise NoesisAlignmentError(f"{path.name} has an invalid camera pose")
        poses.append(pose)
    return (
        np.concatenate(registration_points).astype(np.float64),
        np.concatenate(review_points).astype(np.float64),
        np.concatenate(review_colors).astype(np.uint8),
        np.stack(poses).astype(np.float64),
    )


def _level_phone_floor(
    points: np.ndarray,
    poses: np.ndarray,
    settings: NoesisAlignmentSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import open3d as o3d

    camera_up = _normalize(-np.mean(poses[:, :3, 1], axis=0), name="phone camera up")
    o3d.utility.random.seed(settings.random_seed)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[::3]))
    plane, inliers = cloud.segment_plane(
        distance_threshold=settings.floor_distance_threshold_m,
        ransac_n=3,
        num_iterations=1_600,
    )
    normal_raw = np.asarray(plane[:3], dtype=np.float64)
    normal_length = float(np.linalg.norm(normal_raw))
    normal = _normalize(normal_raw, name="phone floor normal")
    offset = float(plane[3]) / normal_length
    if float(np.dot(normal, camera_up)) < 0.0:
        normal = -normal
        offset = -offset
    rotation = _align_vectors(normal, np.asarray([0.0, 1.0, 0.0]))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[1, 3] = offset
    leveled_points = _transform_points(points, transform)
    leveled_poses = np.stack([transform @ pose for pose in poses])
    camera_heights = leveled_poses[:, 1, 3]
    metrics = {
        "normal_in_mapanything_world": normal.tolist(),
        "plane_offset": offset,
        "ransac_sample_count": int(len(cloud.points)),
        "ransac_inlier_count": int(len(inliers)),
        "ransac_inlier_fraction": float(len(inliers) / max(1, len(cloud.points))),
        "camera_height_m": {
            "min": float(np.min(camera_heights)),
            "median": float(np.median(camera_heights)),
            "max": float(np.max(camera_heights)),
        },
        "transform_row_major": transform.tolist(),
    }
    if metrics["ransac_inlier_fraction"] < 0.015:
        raise NoesisAlignmentError("phone floor plane does not have enough support")
    if not (0.7 <= metrics["camera_height_m"]["median"] <= 2.2):
        raise NoesisAlignmentError(
            "phone floor plane implies an implausible median camera height"
        )
    return leveled_points, leveled_poses, transform, metrics


def _voxel_points(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(cloud.voxel_down_sample(voxel_size_m).points, dtype=np.float64)


def _vertical_structure(
    points: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel_size_m)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(0.28, voxel_size_m * 2.8),
            max_nn=40,
        )
    )
    sampled = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    keep = (
        (sampled[:, 1] > 0.20)
        & (sampled[:, 1] < 2.30)
        & (np.abs(normals[:, 1]) < 0.30)
    )
    if int(np.count_nonzero(keep)) < 500:
        raise NoesisAlignmentError("not enough vertical room structure is available")
    return sampled[keep], normals[keep]


def _heading_deg(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[0]), float(vector[2])))


def _initial_candidates(
    leveled_poses: np.ndarray,
    target_camera_to_world: np.ndarray,
) -> list[np.ndarray]:
    target_center = target_camera_to_world[:3, 3]
    target_heading = _heading_deg(target_camera_to_world[:3, 2])
    candidates: list[np.ndarray] = []
    for pose in leveled_poses:
        yaw = target_heading - _heading_deg(pose[:3, 2])
        yaw = (yaw + 180.0) % 360.0 - 180.0
        rotation = _yaw_transform(np.asarray([yaw, 0.0, 0.0]))
        rotated_center = _transform_points(pose[None, :3, 3], rotation)[0]
        translation = target_center - rotated_center
        candidates.append(np.asarray([yaw, translation[0], translation[2]], dtype=np.float64))
    return candidates


def _apply_yaw_parameters(points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    return _transform_points(points, _yaw_transform(parameters))


def _basin_refine(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    parameters = initial.copy()
    lower = initial + np.asarray([-40.0, -3.0, -3.0])
    upper = initial + np.asarray([40.0, 3.0, 3.0])
    for _ in range(8):
        transformed = _apply_yaw_parameters(source, parameters)
        distances, indices = tree.query(transformed, k=1)
        cutoff = min(1.5, float(np.percentile(distances, 55.0)))
        selected = np.flatnonzero(distances <= cutoff)
        if selected.size < 200:
            break
        if selected.size > 5_000:
            selected = selected[:: max(1, selected.size // 5_000)][:5_000]

        def residual(candidate: np.ndarray) -> np.ndarray:
            return (
                _apply_yaw_parameters(source[selected], candidate) - target[indices[selected]]
            ).reshape(-1)

        parameters = least_squares(
            residual,
            parameters,
            bounds=(lower, upper),
            loss="huber",
            f_scale=0.15,
            max_nfev=35,
        ).x
    distances, _ = tree.query(_apply_yaw_parameters(source, parameters), k=1)
    return parameters, {
        "median_m": float(np.median(distances)),
        "overlap_0_30m": float(np.mean(distances < 0.30)),
    }


def _structure_score(
    parameters: np.ndarray,
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    target_normals: np.ndarray,
    full_source: np.ndarray,
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    transformed_structure = _apply_yaw_parameters(source_structure, parameters)
    target_tree = cKDTree(target_structure)
    source_distance, target_indices = target_tree.query(transformed_structure, k=1)
    plane_residual = np.abs(
        np.sum(
            (transformed_structure - target_structure[target_indices])
            * target_normals[target_indices],
            axis=1,
        )
    )
    usable = np.sort(plane_residual[source_distance < 0.80])
    if usable.size == 0:
        usable = np.asarray([1.0], dtype=np.float64)
    usable = usable[: max(1, int(0.75 * usable.size))]
    source_tree = cKDTree(_apply_yaw_parameters(full_source, parameters))
    target_distance, _ = source_tree.query(target_structure, k=1)
    trimmed_target = np.sort(target_distance)[: max(1, int(0.65 * target_distance.size))]
    objective = float(
        np.mean(np.clip(usable, 0.0, 0.50))
        + 0.5 * np.mean(np.clip(trimmed_target, 0.0, 0.80))
    )
    return {
        "objective": objective,
        "plane_residual_median_m": float(np.median(usable)),
        "plane_residual_p80_m": float(np.percentile(usable, 80.0)),
        "source_overlap_0_30m": float(np.mean(source_distance < 0.30)),
        "target_overlap_0_30m": float(np.mean(target_distance < 0.30)),
    }


def _structure_refine(
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    target_normals: np.ndarray,
    full_source: np.ndarray,
    initial: np.ndarray,
    bounds_delta: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    tree = cKDTree(target_structure)
    parameters = initial.copy()
    delta = (
        np.asarray(bounds_delta, dtype=np.float64)
        if bounds_delta is not None
        else np.asarray([30.0, 1.5, 1.5], dtype=np.float64)
    )
    lower = initial - delta
    upper = initial + delta
    for iteration in range(12):
        transformed = _apply_yaw_parameters(source_structure, parameters)
        distances, indices = tree.query(transformed, k=1)
        selected = np.flatnonzero(distances < min(0.75, 0.36 + iteration * 0.035))
        if selected.size < 100:
            break
        if selected.size > 5_000:
            selected = selected[:: max(1, selected.size // 5_000)][:5_000]

        def residual(candidate: np.ndarray) -> np.ndarray:
            transformed_selected = _apply_yaw_parameters(source_structure[selected], candidate)
            return np.sum(
                (transformed_selected - target_structure[indices[selected]])
                * target_normals[indices[selected]],
                axis=1,
            )

        parameters = least_squares(
            residual,
            parameters,
            bounds=(lower, upper),
            loss="huber",
            f_scale=0.08,
            max_nfev=50,
        ).x
    return parameters, _structure_score(
        parameters,
        source_structure,
        target_structure,
        target_normals,
        full_source,
    )


def _full_cloud_metrics(
    aligned_source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    source_distance, _ = cKDTree(target).query(aligned_source, k=1)
    target_distance, _ = cKDTree(aligned_source).query(target, k=1)
    return {
        "source_median_m": float(np.median(source_distance)),
        "target_median_m": float(np.median(target_distance)),
        "source_overlap_0_20m": float(np.mean(source_distance < 0.20)),
        "source_overlap_0_30m": float(np.mean(source_distance < 0.30)),
        "source_overlap_0_50m": float(np.mean(source_distance < 0.50)),
        "target_overlap_0_20m": float(np.mean(target_distance < 0.20)),
        "target_overlap_0_30m": float(np.mean(target_distance < 0.30)),
        "target_overlap_0_50m": float(np.mean(target_distance < 0.50)),
    }


def _cylinder_between(start: np.ndarray, end: np.ndarray, radius: float, color: list[int]) -> Any | None:
    import trimesh

    vector = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        return None
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vector / length)
    if transform is None:
        transform = np.eye(4)
    transform[:3, 3] = (np.asarray(start) + np.asarray(end)) * 0.5
    mesh.apply_transform(transform)
    mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _write_glbs(
    aligned_path: Path,
    comparison_path: Path,
    aligned_points: np.ndarray,
    phone_colors: np.ndarray,
    target_points: np.ndarray,
    camera_positions: np.ndarray,
    fixed_camera_position: np.ndarray,
) -> None:
    import trimesh

    aligned_scene = trimesh.Scene()
    aligned_scene.add_geometry(
        trimesh.points.PointCloud(aligned_points, colors=phone_colors),
        geom_name="phone_mapanything_aligned_rgb",
    )
    comparison_scene = trimesh.Scene()
    phone_orange = np.tile(np.asarray([255, 151, 58, 255], dtype=np.uint8), (len(aligned_points), 1))
    target_blue = np.tile(np.asarray([66, 151, 224, 255], dtype=np.uint8), (len(target_points), 1))
    comparison_scene.add_geometry(
        trimesh.points.PointCloud(target_points, colors=target_blue),
        geom_name="fixed_camera_noesis_points_blue",
    )
    comparison_scene.add_geometry(
        trimesh.points.PointCloud(aligned_points, colors=phone_orange),
        geom_name="aligned_phone_points_orange",
    )
    extent = max(float(np.linalg.norm(np.ptp(aligned_points, axis=0))), 1.0)
    radius = max(extent * 0.0013, 0.003)
    segments = []
    for start, end in zip(camera_positions[:-1], camera_positions[1:]):
        segment = _cylinder_between(start, end, radius, [75, 255, 114, 255])
        if segment is not None:
            segments.append(segment)
    if segments:
        path_mesh = trimesh.util.concatenate(segments)
        aligned_scene.add_geometry(path_mesh.copy(), geom_name="aligned_phone_camera_path")
        comparison_scene.add_geometry(path_mesh, geom_name="aligned_phone_camera_path_green")
    marker = trimesh.creation.icosphere(subdivisions=2, radius=max(radius * 4.0, 0.04))
    marker.apply_translation(fixed_camera_position)
    marker.visual.vertex_colors = np.tile(
        np.asarray([255, 45, 214, 255], dtype=np.uint8),
        (len(marker.vertices), 1),
    )
    comparison_scene.add_geometry(marker, geom_name="fixed_camera_magenta")
    rotation_x = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    aligned_scene.apply_transform(rotation_x)
    comparison_scene.apply_transform(rotation_x)
    aligned_scene.export(str(aligned_path))
    comparison_scene.export(str(comparison_path))


def _write_topdown(
    path: Path,
    aligned_points: np.ndarray,
    target_points: np.ndarray,
    camera_positions: np.ndarray,
    fixed_camera_position: np.ndarray,
) -> None:
    size = 1_200
    margin = 52
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    combined = np.concatenate([aligned_points[:, [0, 2]], target_points[:, [0, 2]]])
    low = np.percentile(combined, 1.0, axis=0)
    high = np.percentile(combined, 99.0, axis=0)
    span = np.maximum(high - low, 1e-6)

    def project(points: np.ndarray) -> np.ndarray:
        pixels = (points[:, [0, 2]] - low) / span * (size - 2 * margin) + margin
        pixels[:, 1] = size - pixels[:, 1]
        return np.round(pixels).astype(np.int32)

    target_stride = max(1, len(target_points) // 90_000)
    phone_stride = max(1, len(aligned_points) // 140_000)
    for x, y in project(target_points[::target_stride]):
        if 0 <= x < size and 0 <= y < size:
            canvas[y, x] = (220, 145, 62)
    for x, y in project(aligned_points[::phone_stride]):
        if 0 <= x < size and 0 <= y < size:
            canvas[y, x] = (58, 174, 255)
    path_pixels = project(camera_positions)
    if len(path_pixels) >= 2:
        cv2.polylines(canvas, [path_pixels[:, None, :]], False, (75, 255, 114), 4, cv2.LINE_AA)
    fixed_pixel = project(fixed_camera_position[None])[0]
    cv2.circle(canvas, tuple(fixed_pixel), 10, (255, 45, 214), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Noesis blue | aligned phone orange | phone walk green | fixed camera magenta",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise NoesisAlignmentError("failed to write top-down alignment preview")


def _project_depth_grid(
    points: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    cell_px: int = 8,
) -> np.ndarray:
    camera_points = _transform_points(points, camera_from_world)
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0.05)
    camera_points = camera_points[valid]
    projected = (intrinsics @ camera_points.T).T
    pixels = projected[:, :2] / projected[:, 2, None]
    cells = np.floor(pixels / cell_px).astype(np.int64)
    grid_width = int(math.ceil(image_width / cell_px))
    grid_height = int(math.ceil(image_height / cell_px))
    valid = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < grid_width)
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < grid_height)
    )
    cells = cells[valid]
    depth = camera_points[valid, 2]
    grid = np.full((grid_height, grid_width), np.inf, dtype=np.float64)
    np.minimum.at(grid, (cells[:, 1], cells[:, 0]), depth)
    return grid


def _write_reprojection(
    path: Path,
    source_frame_path: Path,
    aligned_points: np.ndarray,
    phone_colors: np.ndarray,
    target_points: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, float]:
    image = cv2.imread(str(source_frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise NoesisAlignmentError(f"fixed-camera keyframe is unreadable: {source_frame_path}")
    height, width = image.shape[:2]
    camera_points = _transform_points(aligned_points, camera_from_world)
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0.05)
    camera_points = camera_points[valid]
    colors = phone_colors[valid]
    projected = (intrinsics @ camera_points.T).T
    pixels = np.round(projected[:, :2] / projected[:, 2, None]).astype(np.int32)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[valid]
    colors = colors[valid]
    depth = camera_points[valid, 2]
    linear = pixels[:, 1].astype(np.int64) * width + pixels[:, 0]
    order = np.argsort(depth)
    _, first = np.unique(linear[order], return_index=True)
    visible = order[first]
    layer = np.zeros_like(image)
    mask = np.zeros((height, width), dtype=np.uint8)
    selected_pixels = pixels[visible]
    layer[selected_pixels[:, 1], selected_pixels[:, 0]] = colors[visible, ::-1]
    mask[selected_pixels[:, 1], selected_pixels[:, 0]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    layer = cv2.dilate(layer, kernel)
    mask = cv2.dilate(mask, kernel)
    overlay = image.copy()
    occupied = mask > 0
    overlay[occupied] = np.clip(
        overlay[occupied].astype(np.float32) * 0.35 + layer[occupied].astype(np.float32) * 0.65,
        0,
        255,
    ).astype(np.uint8)
    cv2.putText(
        overlay,
        "Aligned phone reconstruction through fixed living-room camera",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise NoesisAlignmentError("failed to write fixed-camera reprojection preview")

    phone_grid = _project_depth_grid(
        aligned_points,
        camera_from_world,
        intrinsics,
        width,
        height,
    )
    target_grid = _project_depth_grid(
        target_points,
        camera_from_world,
        intrinsics,
        width,
        height,
    )
    phone_valid = np.isfinite(phone_grid)
    target_valid = np.isfinite(target_grid)
    overlap = phone_valid & target_valid
    depth_delta = np.abs(phone_grid[overlap] - target_grid[overlap])
    return {
        "grid_cell_px": 8.0,
        "phone_visible_cell_count": float(np.count_nonzero(phone_valid)),
        "target_visible_cell_count": float(np.count_nonzero(target_valid)),
        "overlap_cell_count": float(np.count_nonzero(overlap)),
        "phone_cell_overlap_fraction": float(
            np.count_nonzero(overlap) / max(1, np.count_nonzero(phone_valid))
        ),
        "target_cell_coverage_fraction": float(
            np.count_nonzero(overlap) / max(1, np.count_nonzero(target_valid))
        ),
        "overlap_depth_delta_median_m": float(np.median(depth_delta)) if depth_delta.size else math.inf,
        "overlap_depth_delta_p80_m": float(np.percentile(depth_delta, 80.0)) if depth_delta.size else math.inf,
    }


def run_noesis_alignment(
    scan_dir: Path,
    output_dir: Path,
    outputs: dict[str, Any],
    settings: NoesisAlignmentSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    progress(0.02, "Loading saved phone and living-room reconstruction points")
    target_revision = settings.target_revision.resolve()
    target_npz = target_revision / "room_points.npz"
    target_meta_path = target_revision / "room_points_meta.json"
    if not target_npz.is_file() or not target_meta_path.is_file():
        raise NoesisAlignmentError(
            f"authoritative target revision is incomplete: {target_revision.name}"
        )
    if not settings.calibration_path.is_file():
        raise NoesisAlignmentError(
            f"camera calibration is missing: {settings.calibration_path}"
        )
    target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
    if target_meta.get("camera") != settings.camera_id:
        raise NoesisAlignmentError(
            f"target revision camera is {target_meta.get('camera')}, expected {settings.camera_id}"
        )
    coordinate_frame = str(target_meta.get("coordinate_frame") or "")
    if coordinate_frame != "backend_world_m_stream_points":
        raise NoesisAlignmentError(
            f"target revision has unsupported coordinate frame {coordinate_frame}"
        )
    with np.load(target_npz) as target_row:
        target_points = np.asarray(target_row["points"], dtype=np.float64)
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise NoesisAlignmentError("target room point cloud is malformed")

    registration_points, review_points, review_colors, phone_poses = _load_phone_clouds(
        scan_dir,
        outputs,
        settings,
    )
    progress(0.12, "Estimating the phone walk floor and gravity direction")
    leveled_registration, leveled_poses, floor_transform, floor_metrics = _level_phone_floor(
        registration_points,
        phone_poses,
        settings,
    )

    calibration_root = json.loads(settings.calibration_path.read_text(encoding="utf-8"))
    calibration_rows = calibration_root.get("cameras", calibration_root)
    calibration = calibration_rows.get(settings.camera_id)
    if not isinstance(calibration, dict) or not isinstance(calibration.get("E"), list):
        raise NoesisAlignmentError(f"camera {settings.camera_id} has no calibrated E matrix")
    camera_from_backend = np.asarray(calibration["E"], dtype=np.float64).reshape(
        (4, 4), order="F"
    )
    floor_alignment = target_meta.get("floor_alignment")
    if not isinstance(floor_alignment, dict) or not isinstance(
        floor_alignment.get("world_correction_col_major"), list
    ):
        raise NoesisAlignmentError("target revision is missing its floor-world correction")
    target_world_correction = np.asarray(
        floor_alignment["world_correction_col_major"], dtype=np.float64
    ).reshape((4, 4), order="F")
    target_camera_to_world = target_world_correction @ np.linalg.inv(camera_from_backend)
    target_camera_from_world = camera_from_backend @ np.linalg.inv(target_world_correction)
    intrinsics = np.asarray(target_meta.get("intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise NoesisAlignmentError("target revision intrinsics are malformed")

    progress(0.22, "Extracting gravity-preserving wall and doorway structure")
    source_voxel = _voxel_points(leveled_registration, 0.14)
    target_voxel = _voxel_points(target_points, 0.14)
    source_structure, _ = _vertical_structure(
        leveled_registration,
        settings.voxel_size_m,
    )
    target_structure, target_normals = _vertical_structure(
        target_points,
        settings.voxel_size_m,
    )

    progress(0.30, "Searching fixed-camera pose anchors for the room-registration basin")
    initials = _initial_candidates(leveled_poses, target_camera_to_world)
    raw_anchor_index = outputs.get("anchor_view_index")
    explicit_anchor = (
        type(raw_anchor_index) is int
        and 0 <= int(raw_anchor_index) < len(initials)
    )
    if explicit_anchor:
        anchor_index = int(raw_anchor_index)
        selected_basins = [
            (
                initials[anchor_index],
                {"median_m": math.nan, "overlap_0_30m": math.nan},
                anchor_index,
            )
        ]
        progress(
            0.58,
            "Using the jointly inferred fixed-camera view as the registration anchor",
        )
    else:
        basin_rows: list[tuple[np.ndarray, dict[str, float], int]] = []
        for index, initial in enumerate(initials):
            parameters, metrics = _basin_refine(source_voxel, target_voxel, initial)
            basin_rows.append((parameters, metrics, index))
            if index % max(1, len(initials) // 8) == 0:
                progress(
                    0.30 + 0.28 * ((index + 1) / len(initials)),
                    f"Testing phone pose anchor {index + 1} of {len(initials)}",
                )
        basin_rows.sort(
            key=lambda row: (
                row[1]["median_m"] - 0.35 * row[1]["overlap_0_30m"],
                row[2],
            )
        )
        selected_basins = []
        for row in basin_rows:
            if any(
                abs(float(row[0][0] - existing[0][0])) < 2.0
                and float(np.linalg.norm(row[0][1:] - existing[0][1:])) < 0.25
                for existing in selected_basins
            ):
                continue
            selected_basins.append(row)
            if len(selected_basins) >= 14:
                break

    progress(0.60, "Refining candidate fits against walls and vertical room structure")
    candidate_rows: list[dict[str, Any]] = []
    for index, (basin, basin_metrics, pose_index) in enumerate(selected_basins):
        parameters, metrics = _structure_refine(
            source_structure,
            target_structure,
            target_normals,
            leveled_registration,
            basin,
            bounds_delta=(
                np.asarray([5.0, 0.20, 0.20], dtype=np.float64)
                if explicit_anchor
                else None
            ),
        )
        candidate_rows.append(
            {
                "pose_anchor_index": pose_index,
                "parameters": parameters,
                "basin": basin_metrics,
                "metrics": metrics,
            }
        )
        progress(
            0.60 + 0.16 * ((index + 1) / len(selected_basins)),
            f"Refining structural candidate {index + 1} of {len(selected_basins)}",
        )
    candidate_rows.sort(key=lambda row: row["metrics"]["objective"])
    distinct_candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        parameters = np.asarray(row["parameters"], dtype=np.float64)
        duplicate = False
        for existing in distinct_candidates:
            existing_parameters = np.asarray(existing["parameters"], dtype=np.float64)
            yaw_delta = abs(
                (float(parameters[0] - existing_parameters[0]) + 180.0) % 360.0
                - 180.0
            )
            translation_delta = float(
                np.linalg.norm(parameters[1:] - existing_parameters[1:])
            )
            if yaw_delta < 2.0 and translation_delta < 0.25:
                duplicate = True
                break
        if not duplicate:
            distinct_candidates.append(row)
    best = distinct_candidates[0]
    runner_up = distinct_candidates[1] if len(distinct_candidates) > 1 else None
    world_from_phone = _yaw_transform(best["parameters"]) @ floor_transform
    inverse = np.linalg.inv(world_from_phone)
    identity_error = float(np.max(np.abs(inverse @ world_from_phone - np.eye(4))))
    aligned_registration = _transform_points(registration_points, world_from_phone)
    full_metrics = _full_cloud_metrics(aligned_registration, target_points)
    anchor_metrics: dict[str, float] | None = None
    if explicit_anchor:
        aligned_anchor = world_from_phone @ phone_poses[int(raw_anchor_index)]
        anchor_position_error = float(
            np.linalg.norm(
                aligned_anchor[:3, 3] - target_camera_to_world[:3, 3]
            )
        )
        anchor_heading_error = abs(
            (
                _heading_deg(aligned_anchor[:3, 2])
                - _heading_deg(target_camera_to_world[:3, 2])
                + 180.0
            )
            % 360.0
            - 180.0
        )
        anchor_metrics = {
            "position_error_m": anchor_position_error,
            "heading_error_deg": float(anchor_heading_error),
        }
    candidate_margin = (
        float(runner_up["metrics"]["objective"] - best["metrics"]["objective"])
        if runner_up is not None
        else math.inf
    )
    checks = {
        "proper_rigid_transform": bool(
            abs(float(np.linalg.det(world_from_phone[:3, :3])) - 1.0) <= 1e-5
        ),
        "round_trip": bool(identity_error <= 1e-6),
        "candidate_separation": bool(explicit_anchor or candidate_margin >= 0.05),
        "fixed_camera_anchor_position": bool(
            anchor_metrics is None or anchor_metrics["position_error_m"] <= 0.30
        ),
        "fixed_camera_anchor_heading": bool(
            anchor_metrics is None or anchor_metrics["heading_error_deg"] <= 7.0
        ),
        "vertical_source_overlap": bool(best["metrics"]["source_overlap_0_30m"] >= 0.55),
        "vertical_target_overlap": bool(best["metrics"]["target_overlap_0_30m"] >= 0.45),
        "vertical_plane_residual": bool(
            best["metrics"]["plane_residual_median_m"] <= 0.10
        ),
        "full_source_overlap": bool(full_metrics["source_overlap_0_30m"] >= 0.55),
        "full_target_overlap": bool(full_metrics["target_overlap_0_30m"] >= 0.40),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise NoesisAlignmentError(
            "automatic alignment did not clear the quality gate: "
            + ", ".join(failed_checks)
            + "; anchor="
            + json.dumps(anchor_metrics, sort_keys=True)
            + "; vertical="
            + json.dumps(best["metrics"], sort_keys=True)
            + "; full="
            + json.dumps(full_metrics, sort_keys=True)
        )

    progress(0.78, "Writing aligned point clouds, trajectory, and fixed-camera evidence")
    aligned_review = _transform_points(review_points, world_from_phone)
    aligned_poses = np.stack([world_from_phone @ pose for pose in phone_poses])
    aligned_glb = output_dir / "aligned_phone_points.glb"
    comparison_glb = output_dir / "noesis_phone_comparison.glb"
    _write_glbs(
        aligned_glb,
        comparison_glb,
        aligned_review,
        review_colors,
        target_points,
        aligned_poses[:, :3, 3],
        target_camera_to_world[:3, 3],
    )
    topdown_path = output_dir / "alignment_topdown.png"
    _write_topdown(
        topdown_path,
        aligned_review,
        target_points,
        aligned_poses[:, :3, 3],
        target_camera_to_world[:3, 3],
    )
    keyframes = target_meta.get("rgb_keyframes")
    if not isinstance(keyframes, dict) or not keyframes:
        raise NoesisAlignmentError("target revision has no fixed-camera keyframe")
    keyframe_relative = next(iter(keyframes.values()))
    keyframe_path = target_revision / str(keyframe_relative)
    reprojection_path = output_dir / "fixed_camera_reprojection.jpg"
    reprojection_metrics = _write_reprojection(
        reprojection_path,
        keyframe_path,
        aligned_review,
        review_colors,
        target_points,
        target_camera_from_world,
        intrinsics,
    )
    trajectory_path = output_dir / "aligned_camera_trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema": "noesis.mapanything.phone_scan.aligned_camera_trajectory.v1",
                "coordinate_frame": coordinate_frame,
                "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
                "camera_poses": aligned_poses.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    solution_path = output_dir / "aligned_camera_solution.npz"
    np.savez_compressed(
        solution_path,
        camera_poses=aligned_poses.astype(np.float32),
        world_from_mapanything=world_from_phone.astype(np.float64),
        mapanything_from_world=inverse.astype(np.float64),
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    transform_payload = {
        "schema": "noesis.mapanything.phone_scan.world_alignment.v1",
        "generated_at": generated_at,
        "source_coordinate_frame": str(outputs.get("coordinate_frame") or "mapanything_metric_world_unaligned_to_noesis"),
        "target_coordinate_frame": coordinate_frame,
        "scale": 1.0,
        "world_from_mapanything_row_major": world_from_phone.tolist(),
        "world_from_mapanything_col_major": world_from_phone.reshape(-1, order="F").tolist(),
        "mapanything_from_world_row_major": inverse.tolist(),
        "round_trip_max_abs_error": identity_error,
        "phone_floor": floor_metrics,
    }
    transform_path = output_dir / "phone_ma_to_noesis_world.json"
    transform_path.write_text(json.dumps(transform_payload, indent=2), encoding="utf-8")

    report = {
        "schema": "noesis.mapanything.phone_scan.alignment_report.v1",
        "generated_at": generated_at,
        "status": "passed",
        "admission": "saved_review_candidate_not_promoted_to_live_noesis",
        "target": {
            "camera_id": settings.camera_id,
            "revision_id": target_meta.get("revision_id"),
            "coordinate_frame": coordinate_frame,
            "revision_locator": f"data/virtual_twin/revisions/{target_revision.name}",
            "point_count": int(target_points.shape[0]),
        },
        "method": {
            "name": (
                "joint_fixed_camera_view_anchor_with_bounded_vertical_structure_refinement"
                if explicit_anchor
                else "gravity_preserving_pose_seed_search_and_vertical_structure_registration"
            ),
            "degrees_of_freedom": ["yaw", "translation_x", "translation_z"],
            "fixed_scale": 1.0,
            "source_vertical_point_count": int(source_structure.shape[0]),
            "target_vertical_point_count": int(target_structure.shape[0]),
            "candidate_count": len(distinct_candidates),
            "converged_seed_count": len(candidate_rows),
            "winning_pose_anchor_index": int(best["pose_anchor_index"]),
            "explicit_fixed_camera_anchor": bool(explicit_anchor),
            "fixed_camera_anchor_residual": anchor_metrics,
        },
        "quality_gate": {
            "passed": True,
            "checks": checks,
            "candidate_objective_margin": candidate_margin,
        },
        "vertical_structure": best["metrics"],
        "full_cloud": full_metrics,
        "fixed_camera_reprojection": reprojection_metrics,
        "transform": transform_payload,
        "candidate_summary": [
            {
                "pose_anchor_index": int(row["pose_anchor_index"]),
                "yaw_tx_tz": np.asarray(row["parameters"]).tolist(),
                "metrics": row["metrics"],
            }
            for row in distinct_candidates[:5]
        ],
        "inputs": {
            "phone_output_manifest_sha256": _sha256(
                scan_dir / "outputs" / "scan_outputs_manifest.json"
            ),
            "target_points_sha256": _sha256(target_npz),
            "target_meta_sha256": _sha256(target_meta_path),
            "fixed_camera_keyframe_sha256": _sha256(keyframe_path),
            "camera_calibration_sha256": _sha256(settings.calibration_path),
        },
    }
    report_path = output_dir / "alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress(0.96, "Hashing the saved alignment review assets")
    artifacts = {
        "aligned_phone_glb": "alignment/aligned_phone_points.glb",
        "comparison_glb": "alignment/noesis_phone_comparison.glb",
        "topdown_preview": "alignment/alignment_topdown.png",
        "fixed_camera_reprojection": "alignment/fixed_camera_reprojection.jpg",
        "transform": "alignment/phone_ma_to_noesis_world.json",
        "trajectory": "alignment/aligned_camera_trajectory.json",
        "camera_solution_npz": "alignment/aligned_camera_solution.npz",
        "report": "alignment/alignment_report.json",
    }
    files = [
        {
            "path": f"alignment/{path.relative_to(output_dir).as_posix()}",
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    ]
    progress(1.0, "Phone reconstruction is aligned to the living-room Noesis world")
    return {
        "schema": "noesis.phone_scan.alignment_outputs.v2",
        "generated_at": generated_at,
        "coordinate_frame": coordinate_frame,
        "target_camera_id": settings.camera_id,
        "target_revision_id": target_meta.get("revision_id"),
        "admission": "saved_review_candidate_not_promoted_to_live_noesis",
        "quality_gate": report["quality_gate"],
        "vertical_structure": best["metrics"],
        "full_cloud": full_metrics,
        "fixed_camera_reprojection": reprojection_metrics,
        "artifacts": artifacts,
        "files": files,
    }


__all__ = [
    "NoesisAlignmentError",
    "NoesisAlignmentSettings",
    "run_noesis_alignment",
]
