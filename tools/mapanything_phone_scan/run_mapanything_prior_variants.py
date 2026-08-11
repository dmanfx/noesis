#!/usr/bin/env python3
"""Run controlled MapAnything reconstructions with DA3 geometric priors.

The runner keeps the existing image-only MapAnything result untouched and
produces four review variants:

* DA3 camera poses only;
* reliable sparse DA3 metric depth only;
* DA3 poses plus reliable sparse DA3 metric depth;
* the same combined priors with a calibrated static-camera view as view zero.

For pose-conditioned variants, MapAnything is allowed to predict its own pose
solution, but saved points are back-projected through the supplied DA3 poses.
Those poses may remain in DA3's native metric frame for phone-only fusion, or
may be transformed into the Noesis backend world after an independent static
alignment has passed. MapAnything's predicted poses remain separate for
disagreement diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.build_consensus_fusion import (  # noqa: E402
    Sequence,
    _depth_boundary_weight,
    _load_sequence,
    _multiview_consistency,
)
from tools.mapanything_phone_scan.inference import (  # noqa: E402
    MapAnythingScanError,
    _map2d,
    _map3d,
    _matrix,
    _scalar,
    _write_confidence_preview,
    _write_depth_preview,
    _write_mask,
    _write_reconstruction_glb,
    _write_rgb,
    _write_trajectory_preview,
)
from tools.mapanything_phone_scan.alignment import (  # noqa: E402
    NoesisAlignmentError,
    _resolve_target_cloud_for_calibrated_camera,
)


VARIANT_SPECS = {
    "da3_pose": {
        "use_pose": True,
        "use_depth": False,
        "use_static": False,
        "label": "MapAnything + DA3 poses",
    },
    "da3_sparse_depth": {
        "use_pose": False,
        "use_depth": True,
        "use_static": False,
        "label": "MapAnything + reliable sparse DA3 metric depth",
    },
    "da3_pose_sparse_depth": {
        "use_pose": True,
        "use_depth": True,
        "use_static": False,
        "label": "MapAnything + DA3 poses + reliable sparse DA3 metric depth",
    },
    "da3_pose_sparse_depth_static": {
        "use_pose": True,
        "use_depth": True,
        "use_static": True,
        "label": "MapAnything + DA3 poses/depth + calibrated static reference",
    },
}


@dataclass(frozen=True)
class StaticReference:
    image: np.ndarray
    depth_z: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    camera_from_world: np.ndarray
    source_image: Path
    revision: Path
    point_count: int
    projected_point_count: int


@dataclass(frozen=True)
class SparseDepthPriors:
    depth_z: np.ndarray
    mask: np.ndarray
    reliability_mask: np.ndarray
    consistency_score: np.ndarray
    consistency_support: np.ndarray
    boundary_weight: np.ndarray
    metrics: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MapAnythingScanError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MapAnythingScanError(f"{path} does not contain a JSON object")
    return payload


def _validate_rigid(transform: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(transform, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise MapAnythingScanError(f"{name} is not a finite 4x4 matrix")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise MapAnythingScanError(f"{name} has an invalid homogeneous final row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise MapAnythingScanError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-4):
        raise MapAnythingScanError(f"{name} rotation is not proper")
    return result


def _load_world_from_da3(path: Path) -> np.ndarray:
    payload = _load_json(path)
    matrix = payload.get("world_from_mapanything_row_major")
    if not isinstance(matrix, list):
        matrix = payload.get("world_from_da3_row_major")
    if not isinstance(matrix, list):
        raise MapAnythingScanError(
            f"{path} has neither world_from_mapanything_row_major nor world_from_da3_row_major"
        )
    return _validate_rigid(np.asarray(matrix, dtype=np.float64), name="world_from_da3")


def _resolve_carrier_frame(
    world_from_da3_path: Path | None,
) -> tuple[np.ndarray, str, Path | None]:
    if world_from_da3_path is None:
        return (
            np.eye(4, dtype=np.float64),
            "da3_metric_world_unaligned_to_noesis",
            None,
        )
    resolved = world_from_da3_path.resolve()
    return _load_world_from_da3(resolved), "backend_world_m_stream_points", resolved


def _transform_poses(transform: np.ndarray, poses: np.ndarray) -> np.ndarray:
    transform = _validate_rigid(transform, name="pose transform")
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise MapAnythingScanError(f"camera poses have invalid shape {poses.shape}")
    transformed = np.einsum("ij,njk->nik", transform, poses)
    for index, pose in enumerate(transformed):
        _validate_rigid(pose, name=f"transformed camera pose {index}")
    return transformed


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first[:3, :3], dtype=np.float64).T @ np.asarray(
        second[:3, :3], dtype=np.float64
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _transform_pointmap(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape((-1, 3))
    transformed = (transform[:3, :3] @ flat.T).T + transform[:3, 3]
    return transformed.reshape(points.shape).astype(np.float32)


def _backproject_depth(
    depth_z: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Back-project Z depth through an OpenCV camera-to-world transform."""
    depth = np.asarray(depth_z, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    camera_to_world = _validate_rigid(camera_to_world, name="camera_to_world")
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise MapAnythingScanError("depth back-projection received malformed inputs")
    height, width = depth.shape
    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    local = np.stack(
        (
            (uu - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (vv - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )
    world = (
        camera_to_world[:3, :3] @ local.reshape((-1, 3)).T
    ).T + camera_to_world[:3, 3]
    return world.reshape((height, width, 3)).astype(np.float32)


def _top_up_reliable_samples(
    reliable: np.ndarray,
    sampled: np.ndarray,
    *,
    rng: np.random.Generator,
    minimum_per_view: int,
) -> tuple[np.ndarray, list[int], int]:
    reliable = np.asarray(reliable, dtype=bool)
    result = np.asarray(sampled, dtype=bool).copy()
    if reliable.shape != result.shape or reliable.ndim != 3:
        raise MapAnythingScanError("reliable and sampled masks must be matching NxHxW arrays")
    topped_up_views: list[int] = []
    added_total = 0
    for index in range(reliable.shape[0]):
        sampled_flat = result[index].reshape(-1)
        count = int(np.count_nonzero(sampled_flat))
        if count >= minimum_per_view:
            continue
        candidates = np.flatnonzero(
            reliable[index].reshape(-1) & ~sampled_flat
        )
        needed = minimum_per_view - count
        if candidates.size < needed:
            raise MapAnythingScanError(
                "reliable sparse DA3 depth has fewer than "
                f"{minimum_per_view} reliable samples in view {index}"
            )
        selected = rng.choice(candidates, size=needed, replace=False)
        sampled_flat[selected] = True
        topped_up_views.append(index)
        added_total += needed
    return result, topped_up_views, added_total


def _build_sparse_depth_priors(
    da3: Sequence,
    *,
    consistency_threshold: float,
    boundary_threshold: float,
    sample_fraction: float,
    random_seed: int,
) -> SparseDepthPriors:
    if not (0.0 < sample_fraction <= 1.0):
        raise MapAnythingScanError("sparse depth sample fraction must be in (0, 1]")
    consistency = _multiview_consistency(
        da3.depth,
        da3.mask,
        da3.intrinsics,
        da3.poses,
    )
    boundary = _depth_boundary_weight(da3.depth, da3.mask)
    reliable = (
        da3.mask
        & np.isfinite(da3.depth)
        & (da3.depth > 0.05)
        & (consistency.support >= 1)
        & (consistency.score >= consistency_threshold)
        & (boundary >= boundary_threshold)
    )
    rng = np.random.default_rng(random_seed)
    sampled = reliable & (rng.random(reliable.shape) < sample_fraction)
    sampled, topped_up_views, top_up_sample_count = _top_up_reliable_samples(
        reliable,
        sampled,
        rng=rng,
        minimum_per_view=500,
    )
    per_view_count = np.count_nonzero(sampled, axis=(1, 2))
    sparse_depth = np.where(sampled, da3.depth, 0.0).astype(np.float32)
    metrics = {
        "method": (
            "da3_only_temporal_reprojection_plus_depth_boundary_then_seeded_sampling_"
            "with_reliable_per_view_minimum"
        ),
        "consistency_threshold": float(consistency_threshold),
        "boundary_threshold": float(boundary_threshold),
        "sample_fraction_of_reliable_pixels": float(sample_fraction),
        "random_seed": int(random_seed),
        "minimum_samples_per_view": 500,
        "minimum_top_up_views": topped_up_views,
        "minimum_top_up_sample_count": int(top_up_sample_count),
        "multiview_reprojection_error_median_m": float(consistency.median_error_m),
        "multiview_reprojection_error_p80_m": float(consistency.p80_error_m),
        "reliable_fraction_of_all_pixels": float(np.mean(reliable)),
        "sampled_fraction_of_all_pixels": float(np.mean(sampled)),
        "sampled_fraction_of_da3_valid_pixels": float(
            np.count_nonzero(sampled) / max(1, np.count_nonzero(da3.mask))
        ),
        "samples_per_view": {
            "min": int(np.min(per_view_count)),
            "median": float(np.median(per_view_count)),
            "max": int(np.max(per_view_count)),
        },
    }
    return SparseDepthPriors(
        depth_z=sparse_depth,
        mask=sampled,
        reliability_mask=reliable,
        consistency_score=consistency.score.astype(np.float32),
        consistency_support=consistency.support.astype(np.uint8),
        boundary_weight=boundary.astype(np.float32),
        metrics=metrics,
    )


def _zbuffer_depth(
    points_world: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Project a world cloud to one-pixel nearest-surface Z-depth samples."""
    height, width = image_shape
    points = np.asarray(points_world, dtype=np.float64)
    camera_from_world = _validate_rigid(camera_from_world, name="camera_from_world")
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    camera = (camera_from_world[:3, :3] @ points.T).T + camera_from_world[:3, 3]
    projected = (intrinsics @ camera.T).T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
    valid = (
        np.isfinite(camera).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (camera[:, 2] > 0.05)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.float32), 0
    pixels = np.rint(uv[valid]).astype(np.int64)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    depths = camera[valid, 2]
    linear = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((depths, linear))
    ordered_linear = linear[order]
    first = np.concatenate(
        ([True], ordered_linear[1:] != ordered_linear[:-1])
    )
    selected = order[first]
    depth_map = np.zeros((height, width), dtype=np.float32)
    chosen_pixels = pixels[selected]
    depth_map[chosen_pixels[:, 1], chosen_pixels[:, 0]] = depths[selected].astype(
        np.float32
    )
    return depth_map, int(selected.size)


def _load_static_reference(
    revision: Path,
    calibration_path: Path,
    camera_id: str,
) -> StaticReference:
    revision = revision.resolve()
    meta_path = revision / "room_points_meta.json"
    points_path = revision / "room_points.npz"
    if not meta_path.is_file() or not points_path.is_file():
        raise MapAnythingScanError(f"static revision is incomplete: {revision}")
    meta = _load_json(meta_path)
    if meta.get("camera") != camera_id:
        raise MapAnythingScanError(
            f"static revision camera is {meta.get('camera')}, expected {camera_id}"
        )
    if meta.get("coordinate_frame") != "backend_world_m_stream_points":
        raise MapAnythingScanError("static revision is not in backend world coordinates")
    keyframes = meta.get("rgb_keyframes")
    if not isinstance(keyframes, dict) or not keyframes:
        raise MapAnythingScanError("static revision has no RGB keyframe")
    source_image = revision / str(next(iter(keyframes.values())))
    image_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise MapAnythingScanError(f"static keyframe is unreadable: {source_image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    intrinsics = np.asarray(meta.get("intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise MapAnythingScanError("static revision intrinsics are malformed")

    calibration_root = _load_json(calibration_path)
    calibration_rows = calibration_root.get("cameras", calibration_root)
    calibration = calibration_rows.get(camera_id)
    if not isinstance(calibration, dict) or not isinstance(calibration.get("E"), list):
        raise MapAnythingScanError(f"camera {camera_id} has no calibrated E matrix")
    camera_from_backend = np.asarray(calibration["E"], dtype=np.float64).reshape(
        (4, 4), order="F"
    )
    floor_alignment = meta.get("floor_alignment")
    if not isinstance(floor_alignment, dict) or not isinstance(
        floor_alignment.get("world_correction_col_major"), list
    ):
        raise MapAnythingScanError("static revision has no floor-world correction")
    world_correction = np.asarray(
        floor_alignment["world_correction_col_major"], dtype=np.float64
    ).reshape((4, 4), order="F")
    calibrated_camera_to_world = _validate_rigid(
        world_correction @ np.linalg.inv(camera_from_backend),
        name="calibrated static camera_to_world",
    )
    with np.load(points_path) as row:
        points = np.asarray(row["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise MapAnythingScanError("static room point cloud is malformed")
    try:
        points, _, _ = _resolve_target_cloud_for_calibrated_camera(
            points,
            calibrated_camera_to_world,
        )
    except NoesisAlignmentError as exc:
        raise MapAnythingScanError(str(exc)) from exc
    camera_to_world = _validate_rigid(
        calibrated_camera_to_world,
        name="resolved static camera_to_world",
    )
    camera_from_world = _validate_rigid(
        np.linalg.inv(camera_to_world),
        name="resolved static camera_from_world",
    )
    depth, projected = _zbuffer_depth(
        points,
        camera_from_world,
        intrinsics,
        image.shape[:2],
    )
    if projected < 5_000:
        raise MapAnythingScanError("too few static points project into the keyframe")
    return StaticReference(
        image=image,
        depth_z=depth,
        intrinsics=intrinsics.astype(np.float32),
        camera_to_world=camera_to_world,
        camera_from_world=camera_from_world,
        source_image=source_image,
        revision=revision,
        point_count=int(points.shape[0]),
        projected_point_count=projected,
    )


def _load_frame_rows(scan_dir: Path) -> list[dict[str, Any]]:
    manifest_path = scan_dir / "prepared_frames_manifest.json"
    payload = _load_json(manifest_path)
    rows = payload.get("frames")
    if not isinstance(rows, list) or len(rows) < 2:
        raise MapAnythingScanError(f"{manifest_path} has no prepared multiview frame set")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("frame"), str):
            raise MapAnythingScanError(f"prepared frame row {index} is malformed")
        frame_path = scan_dir / row["frame"]
        if not frame_path.is_file():
            raise MapAnythingScanError(f"prepared frame is missing: {frame_path}")
    return rows


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _build_input_views(
    variant_name: str,
    scan_dir: Path,
    frame_rows: list[dict[str, Any]],
    da3: Sequence,
    backend_poses: np.ndarray,
    sparse: SparseDepthPriors,
    static: StaticReference | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray | None]:
    spec = VARIANT_SPECS[variant_name]
    use_pose = bool(spec["use_pose"])
    use_depth = bool(spec["use_depth"])
    use_static = bool(spec["use_static"])
    views: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    carrier_poses: list[np.ndarray] = []
    if use_static:
        if static is None:
            raise MapAnythingScanError(
                f"{variant_name} requires a calibrated static-camera reference"
            )
        views.append(
            {
                "img": static.image,
                "depth_z": static.depth_z,
                "intrinsics": static.intrinsics,
                "camera_poses": static.camera_to_world.astype(np.float32),
            }
        )
        source_rows.append(
            {
                "source_frame": str(static.source_image),
                "timestamp_s": None,
                "fixed_camera_anchor": True,
                "phone_frame_index": None,
            }
        )
        carrier_poses.append(static.camera_to_world)
    for index, row in enumerate(frame_rows):
        if use_depth:
            image = da3.rgb[index]
        else:
            image = _load_rgb(scan_dir / str(row["frame"]))
        view: dict[str, Any] = {"img": image}
        if use_depth:
            view["depth_z"] = sparse.depth_z[index]
            view["intrinsics"] = da3.intrinsics[index].astype(np.float32)
        if use_pose:
            view["camera_poses"] = backend_poses[index].astype(np.float32)
            carrier_poses.append(backend_poses[index])
        source_rows.append(
            {
                "source_frame": str(row["frame"]),
                "timestamp_s": row.get("timestamp_s"),
                "fixed_camera_anchor": False,
                "phone_frame_index": index,
            }
        )
        views.append(view)
    return (
        views,
        source_rows,
        np.stack(carrier_poses).astype(np.float64) if use_pose else None,
    )


def _numpy_from_processed(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _save_prior_bundle(
    path: Path,
    sparse: SparseDepthPriors,
    da3: Sequence,
) -> None:
    np.savez_compressed(
        path,
        depth_z=sparse.depth_z,
        sparse_mask=sparse.mask.astype(np.uint8),
        reliability_mask=sparse.reliability_mask.astype(np.uint8),
        consistency_score=sparse.consistency_score,
        consistency_support=sparse.consistency_support,
        boundary_weight=sparse.boundary_weight,
        intrinsics=da3.intrinsics.astype(np.float32),
        da3_camera_poses=da3.poses.astype(np.float32),
    )


def _file_inventory(root: Path, excluded: Iterable[Path] = ()) -> list[dict[str, Any]]:
    excluded_resolved = {path.resolve() for path in excluded}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.resolve() not in excluded_resolved:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    return rows


def _save_variant_outputs(
    output_dir: Path,
    variant_name: str,
    outputs: list[dict[str, Any]],
    processed_views: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    carrier_poses: np.ndarray | None,
    sparse: SparseDepthPriors,
    da3: Sequence,
    static: StaticReference | None,
    frame_rows: list[dict[str, Any]],
    carrier_from_da3: np.ndarray,
    coordinate_frame: str,
    model_metadata: dict[str, Any],
    point_budget: int,
) -> dict[str, Any]:
    if len(outputs) != len(source_rows) or len(outputs) != len(processed_views):
        raise MapAnythingScanError("prediction, input, and source row counts disagree")
    spec = VARIANT_SPECS[variant_name]
    preview_root = output_dir / "views"
    raw_root = output_dir / "raw"
    preview_root.mkdir(parents=True, exist_ok=False)
    raw_root.mkdir(parents=True, exist_ok=False)

    predicted_reference_pose = _matrix(
        outputs[0].get("camera_poses"),
        name="camera_poses",
        shape=(4, 4),
    ).astype(np.float64)
    predicted_reference_pose = _validate_rigid(
        predicted_reference_pose,
        name="predicted reference pose",
    )
    reference_transform: np.ndarray | None = None
    if carrier_poses is not None:
        reference_transform = _validate_rigid(
            carrier_poses[0] @ np.linalg.inv(predicted_reference_pose),
            name="backend_from_predicted_reference",
        )

    per_view_budget = max(1_000, int(point_budget) // len(outputs))
    frame_results: list[dict[str, Any]] = []
    sampled_points: list[np.ndarray] = []
    sampled_colors: list[np.ndarray] = []
    saved_poses: list[np.ndarray] = []
    model_poses_world: list[np.ndarray] = []
    intrinsics_rows: list[np.ndarray] = []
    scales: list[float] = []
    position_errors: list[float] = []
    rotation_errors: list[float] = []

    for index, pred in enumerate(outputs):
        if not isinstance(pred, dict):
            raise MapAnythingScanError(f"MapAnything prediction {index} is malformed")
        model_points = _map3d(pred.get("pts3d"), name="pts3d").astype(np.float32)
        depth = _map2d(pred.get("depth_z"), name="depth_z").astype(np.float32)
        confidence = _map2d(pred.get("conf"), name="conf").astype(np.float32)
        mask = _map2d(pred.get("mask"), name="mask").astype(bool)
        image_rgb = _map3d(pred.get("img_no_norm"), name="img_no_norm")
        model_pose = _validate_rigid(
            _matrix(pred.get("camera_poses"), name="camera_poses", shape=(4, 4)),
            name=f"predicted camera pose {index}",
        )
        intrinsics = _matrix(
            pred.get("intrinsics"), name="intrinsics", shape=(3, 3)
        ).astype(np.float32)
        scale = _scalar(pred.get("metric_scaling_factor"), name="metric_scaling_factor")
        if depth.shape != mask.shape or confidence.shape != mask.shape:
            raise MapAnythingScanError(f"prediction {index} has inconsistent map shapes")

        model_pose_world = (
            reference_transform @ model_pose
            if reference_transform is not None
            else model_pose
        )
        if carrier_poses is not None:
            saved_pose = carrier_poses[index]
            world_points = _backproject_depth(depth, intrinsics, saved_pose)
            position_errors.append(
                float(np.linalg.norm(model_pose_world[:3, 3] - saved_pose[:3, 3]))
            )
            rotation_errors.append(_rotation_error_deg(model_pose_world, saved_pose))
            point_source = "predicted_depth_backprojected_through_supplied_pose_prior"
        else:
            saved_pose = model_pose_world
            world_points = model_points
            point_source = "mapanything_predicted_world_points"
        valid = (
            mask
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(world_points).all(axis=2)
        )

        stem = f"view_{index:04d}"
        rgb_path = preview_root / f"{stem}_rgb.png"
        depth_path = preview_root / f"{stem}_depth.png"
        confidence_path = preview_root / f"{stem}_confidence.png"
        mask_path = preview_root / f"{stem}_mask.png"
        raw_path = raw_root / f"{stem}.npz"
        _write_rgb(rgb_path, image_rgb)
        depth_stats = _write_depth_preview(depth_path, depth, valid)
        confidence_stats = _write_confidence_preview(
            confidence_path, confidence, valid
        )
        _write_mask(mask_path, valid)

        prior_depth = processed_views[index].get("depth_z")
        if prior_depth is None:
            prior_depth_array = np.zeros_like(depth, dtype=np.float32)
        else:
            prior_depth_array = np.squeeze(_numpy_from_processed(prior_depth)).astype(
                np.float32
            )
            if prior_depth_array.shape != depth.shape:
                raise MapAnythingScanError(
                    f"processed prior depth {index} has shape {prior_depth_array.shape}, expected {depth.shape}"
                )
        np.savez_compressed(
            raw_path,
            world_points=world_points.astype(np.float32),
            depth_z=depth,
            confidence=confidence,
            mask=valid.astype(np.uint8),
            camera_pose=np.asarray(saved_pose, dtype=np.float32),
            model_camera_pose=np.asarray(model_pose_world, dtype=np.float32),
            intrinsics=intrinsics,
            metric_scaling_factor=np.asarray([scale], dtype=np.float32),
            model_rgb=np.asarray(image_rgb, dtype=np.float32),
            input_depth_prior_z=prior_depth_array,
        )

        valid_indices = np.flatnonzero(valid.reshape(-1))
        if valid_indices.size:
            stride = max(1, int(math.ceil(valid_indices.size / per_view_budget)))
            selected = valid_indices[::stride][:per_view_budget]
            image_u8 = np.clip(
                image_rgb * 255.0 if float(np.nanmax(image_rgb)) <= 1.5 else image_rgb,
                0,
                255,
            ).astype(np.uint8)
            sampled_points.append(world_points.reshape((-1, 3))[selected])
            sampled_colors.append(image_u8.reshape((-1, 3))[selected])
        saved_poses.append(np.asarray(saved_pose, dtype=np.float64))
        model_poses_world.append(np.asarray(model_pose_world, dtype=np.float64))
        intrinsics_rows.append(intrinsics)
        scales.append(scale)
        source = source_rows[index]
        frame_results.append(
            {
                "index": index,
                **source,
                "model_rgb": f"views/{rgb_path.name}",
                "depth_preview": f"views/{depth_path.name}",
                "confidence_preview": f"views/{confidence_path.name}",
                "mask_preview": f"views/{mask_path.name}",
                "raw_npz": f"raw/{raw_path.name}",
                "depth": depth_stats,
                "confidence": confidence_stats,
                "camera_pose": np.asarray(saved_pose).tolist(),
                "model_camera_pose_world": np.asarray(model_pose_world).tolist(),
                "intrinsics": intrinsics.tolist(),
                "metric_scaling_factor": scale,
                "input_depth_prior_valid_fraction": float(
                    np.mean(prior_depth_array > 0.0)
                ),
                "world_point_source": point_source,
            }
        )
        outputs[index] = {}
        print(
            f"[{variant_name}] saved view {index + 1}/{len(source_rows)}",
            flush=True,
        )

    if not sampled_points:
        raise MapAnythingScanError(f"{variant_name} produced no valid world points")
    points = np.concatenate(sampled_points).astype(np.float32)
    colors = np.concatenate(sampled_colors).astype(np.uint8)
    poses = np.stack(saved_poses).astype(np.float32)
    model_poses = np.stack(model_poses_world).astype(np.float32)
    intrinsics_array = np.stack(intrinsics_rows).astype(np.float32)
    scales_array = np.asarray(scales, dtype=np.float32)

    _write_reconstruction_glb(
        output_dir / "reconstruction_points.glb",
        points,
        colors,
        poses[:, :3, 3],
    )
    _write_trajectory_preview(
        output_dir / "camera_trajectory_topdown.png",
        points,
        poses[:, :3, 3],
    )
    camera_solution = {
        "camera_poses": poses,
        "model_camera_poses_world": model_poses,
        "intrinsics": intrinsics_array,
        "metric_scaling_factors": scales_array,
        "carrier_from_da3": carrier_from_da3.astype(np.float64),
        "carrier_from_predicted_reference": (
            reference_transform.astype(np.float64)
            if reference_transform is not None
            else np.eye(4, dtype=np.float64)
        ),
    }
    if coordinate_frame == "backend_world_m_stream_points":
        camera_solution["world_from_da3"] = carrier_from_da3.astype(np.float64)
        camera_solution["backend_from_predicted_reference"] = camera_solution[
            "carrier_from_predicted_reference"
        ]
    np.savez_compressed(output_dir / "camera_solution.npz", **camera_solution)
    (output_dir / "camera_trajectory.json").write_text(
        json.dumps(
            {
                "schema": "noesis.mapanything.prior_variant.camera_trajectory.v1",
                "coordinate_frame": (
                    coordinate_frame
                    if carrier_poses is not None
                    else "mapanything_metric_world_unaligned_to_noesis"
                ),
                "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
                "camera_poses": poses.tolist(),
                "model_camera_poses_world": model_poses.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_prior_bundle(output_dir / "da3_sparse_depth_prior.npz", sparse, da3)

    phone_raw_relative: str | None = None
    if bool(spec["use_static"]):
        phone_raw = output_dir / "phone_raw"
        phone_raw.mkdir(parents=True, exist_ok=False)
        for phone_index in range(len(frame_rows)):
            source = raw_root / f"view_{phone_index + 1:04d}.npz"
            target = phone_raw / f"view_{phone_index:04d}.npz"
            os.link(source, target)
        phone_raw_relative = "phone_raw"

    bounds_min = np.min(points[np.isfinite(points).all(axis=1)], axis=0)
    bounds_max = np.max(points[np.isfinite(points).all(axis=1)], axis=0)
    pose_disagreement: dict[str, Any] | None = None
    if position_errors:
        pose_disagreement = {
            "meaning": "MapAnything predicted pose after reference anchoring versus supplied pose carrier",
            "position_error_m": {
                "median": float(np.median(position_errors)),
                "p80": float(np.percentile(position_errors, 80.0)),
                "max": float(np.max(position_errors)),
            },
            "rotation_error_deg": {
                "median": float(np.median(rotation_errors)),
                "p80": float(np.percentile(rotation_errors, 80.0)),
                "max": float(np.max(rotation_errors)),
            },
        }
    reference_transforms = {
        "carrier_from_da3_row_major": carrier_from_da3.tolist(),
        "carrier_from_predicted_reference_row_major": (
            reference_transform.tolist()
            if reference_transform is not None
            else None
        ),
    }
    if coordinate_frame == "backend_world_m_stream_points":
        reference_transforms["world_from_da3_row_major"] = carrier_from_da3.tolist()
        reference_transforms["backend_from_predicted_reference_row_major"] = (
            reference_transform.tolist()
            if reference_transform is not None
            else None
        )
    summary: dict[str, Any] = {
        "schema": "noesis.mapanything.prior_variant.outputs.v1",
        "generated_at": _utc_now(),
        "variant": variant_name,
        "label": spec["label"],
        "inputs": {
            "uses_da3_pose_prior": bool(spec["use_pose"]),
            "uses_da3_sparse_depth_prior": bool(spec["use_depth"]),
            "uses_static_reference": bool(spec["use_static"]),
            "phone_view_count": len(frame_rows),
            "total_view_count": len(outputs),
            "static_revision": str(static.revision) if static is not None else None,
            "static_keyframe": str(static.source_image) if static is not None else None,
            "static_point_count": static.point_count if static is not None else None,
            "static_projected_depth_sample_count": (
                static.projected_point_count if static is not None else None
            ),
            "sparse_da3_depth": sparse.metrics,
        },
        "model": model_metadata,
        "world_output_policy": (
            "mapanything_depth_and_intrinsics_backprojected_through_da3_pose_carrier"
            if carrier_poses is not None
            else "mapanything_predicted_world_points"
        ),
        "coordinate_frame": (
            coordinate_frame
            if carrier_poses is not None
            else "mapanything_metric_world_unaligned_to_noesis"
        ),
        "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
        "anchor_view_index": 0 if bool(spec["use_static"]) else None,
        "pose_disagreement": pose_disagreement,
        "review_point_count": int(points.shape[0]),
        "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "metric_scaling_factor": {
            "min": float(np.min(scales_array)),
            "median": float(np.median(scales_array)),
            "max": float(np.max(scales_array)),
        },
        "reference_transforms": reference_transforms,
        "artifacts": {
            "raw": "raw",
            "phone_raw": phone_raw_relative,
            "reconstruction_glb": "reconstruction_points.glb",
            "trajectory_preview": "camera_trajectory_topdown.png",
            "trajectory_json": "camera_trajectory.json",
            "camera_solution_npz": "camera_solution.npz",
            "da3_sparse_depth_prior_npz": "da3_sparse_depth_prior.npz",
            "manifest": "variant_manifest.json",
        },
        "frames": frame_results,
    }
    manifest_path = output_dir / "variant_manifest.json"
    summary["files"] = _file_inventory(output_dir, excluded=[manifest_path])
    manifest_row = {"path": "variant_manifest.json", "size_bytes": 0}
    summary["files"].append(manifest_row)
    for _ in range(3):
        manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        size = int(manifest_path.stat().st_size)
        if manifest_row["size_bytes"] == size:
            break
        manifest_row["size_bytes"] = size
    return summary


def _variant_output_name(variant: str) -> str:
    return f"mapanything_{variant}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--da3-raw", type=Path, required=True)
    parser.add_argument(
        "--world-from-da3",
        type=Path,
        help=(
            "Passed Noesis alignment transform. Omit for an explicitly unaligned "
            "DA3-native phone-only carrier frame."
        ),
    )
    parser.add_argument(
        "--target-revision",
        type=Path,
        help="Static-camera revision; required only by a *_static variant.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Camera calibration; required only by a *_static variant.",
    )
    parser.add_argument("--camera", default="living-room")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_SPECS),
        default=list(VARIANT_SPECS),
    )
    parser.add_argument("--model-id", default="facebook/map-anything-apache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--point-budget", type=int, default=600_000)
    parser.add_argument("--consistency-threshold", type=float, default=0.55)
    parser.add_argument("--boundary-threshold", type=float, default=0.50)
    parser.add_argument("--sparse-sample-fraction", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=8_675_309)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit model downloads. The default is local-files-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.model_id.endswith("-apache"):
        raise MapAnythingScanError("only the Apache-licensed MapAnything model is permitted")
    scan_dir = args.scan_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame_rows = _load_frame_rows(scan_dir)
    da3 = _load_sequence(args.da3_raw.resolve(), "DA3")
    if da3.depth.shape[0] != len(frame_rows):
        raise MapAnythingScanError(
            f"DA3 has {da3.depth.shape[0]} views, prepared scan has {len(frame_rows)}"
        )
    carrier_from_da3, coordinate_frame, carrier_transform_source = _resolve_carrier_frame(
        args.world_from_da3
    )
    carrier_poses = _transform_poses(carrier_from_da3, da3.poses)
    print("Computing DA3-only temporal reliability for sparse depth priors", flush=True)
    sparse = _build_sparse_depth_priors(
        da3,
        consistency_threshold=args.consistency_threshold,
        boundary_threshold=args.boundary_threshold,
        sample_fraction=args.sparse_sample_fraction,
        random_seed=args.random_seed,
    )
    print(json.dumps(sparse.metrics, indent=2), flush=True)
    needs_static = any(bool(VARIANT_SPECS[name]["use_static"]) for name in args.variants)
    static: StaticReference | None = None
    if needs_static:
        if (
            args.world_from_da3 is None
            or args.target_revision is None
            or args.calibration is None
        ):
            raise MapAnythingScanError(
                "a *_static variant requires --world-from-da3, --target-revision, and --calibration"
            )
        static = _load_static_reference(
            args.target_revision,
            args.calibration.resolve(),
            args.camera,
        )
        print(
            f"Static reference: {static.projected_point_count}/{static.point_count} points project into {static.source_image.name}",
            flush=True,
        )
    else:
        print(f"Pose carrier frame: {coordinate_frame}", flush=True)

    existing = [
        output_root / _variant_output_name(variant)
        for variant in args.variants
        if (output_root / _variant_output_name(variant)).exists()
    ]
    if existing:
        raise MapAnythingScanError(
            "refusing to overwrite existing variant outputs: "
            + ", ".join(str(path) for path in existing)
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from importlib.metadata import version
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise MapAnythingScanError(f"CUDA device {args.device} is unavailable")
    device = torch.device(args.device)
    print(f"Loading {args.model_id} on {device}", flush=True)
    model = MapAnything.from_pretrained(
        args.model_id,
        local_files_only=not args.allow_download,
    ).to(device)
    model.eval()
    model_metadata = {
        "id": args.model_id,
        "mapanything_version": version("mapanything"),
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "memory_efficient_inference": True,
        "apply_mask": True,
        "mask_edges": True,
        "apply_confidence_mask": False,
        "local_files_only": not args.allow_download,
    }
    suite_rows: list[dict[str, Any]] = []
    try:
        for variant in args.variants:
            name = _variant_output_name(variant)
            final_dir = output_root / name
            stage_dir = output_root / f".{name}.incomplete-{os.getpid()}"
            if stage_dir.exists():
                raise MapAnythingScanError(f"staging directory already exists: {stage_dir}")
            stage_dir.mkdir(parents=False, exist_ok=False)
            print(f"[{variant}] preparing multimodal views", flush=True)
            raw_views, source_rows, carrier_poses = _build_input_views(
                variant,
                scan_dir,
                frame_rows,
                da3,
                carrier_poses,
                sparse,
                static,
            )
            processed_views = preprocess_inputs(raw_views, verbose=True)
            del raw_views
            expected_shape = tuple(processed_views[0]["img"].shape[-2:])
            if any(tuple(view["img"].shape[-2:]) != expected_shape for view in processed_views):
                raise MapAnythingScanError("preprocessed views do not share one image shape")
            if bool(VARIANT_SPECS[variant]["use_depth"]):
                prior_counts = [
                    int(np.count_nonzero(_numpy_from_processed(view["depth_z"])))
                    for view in processed_views
                ]
                if min(prior_counts) < 500:
                    raise MapAnythingScanError(
                        f"{variant} has fewer than 500 preprocessed depth-prior samples in a view"
                    )
                print(
                    f"[{variant}] preprocessed shape={expected_shape}, depth samples/view min={min(prior_counts)} median={np.median(prior_counts):.0f} max={max(prior_counts)}",
                    flush=True,
                )
            print(
                f"[{variant}] running joint inference over {len(processed_views)} views",
                flush=True,
            )
            outputs: list[dict[str, Any]] | None = None
            try:
                with torch.inference_mode():
                    outputs = model.infer(
                        processed_views,
                        memory_efficient_inference=True,
                        use_amp=True,
                        amp_dtype=args.amp_dtype,
                        apply_mask=True,
                        mask_edges=True,
                        apply_confidence_mask=False,
                        confidence_percentile=10,
                    )
                if not isinstance(outputs, list) or len(outputs) != len(processed_views):
                    raise MapAnythingScanError(
                        f"{variant} returned an invalid prediction list"
                    )
                print(f"[{variant}] inference complete; writing review artifacts", flush=True)
                summary = _save_variant_outputs(
                    stage_dir,
                    variant,
                    outputs,
                    processed_views,
                    source_rows,
                    carrier_poses,
                    sparse,
                    da3,
                    static,
                    frame_rows,
                    carrier_from_da3,
                    coordinate_frame,
                    model_metadata,
                    args.point_budget,
                )
                os.replace(stage_dir, final_dir)
                suite_rows.append(
                    {
                        "variant": variant,
                        "label": VARIANT_SPECS[variant]["label"],
                        "output": name,
                        "coordinate_frame": summary["coordinate_frame"],
                        "pose_disagreement": summary["pose_disagreement"],
                    }
                )
                print(f"[{variant}] complete: {final_dir}", flush=True)
            except Exception:
                if stage_dir.exists():
                    failed = output_root / f"{stage_dir.name}.failed-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
                    os.replace(stage_dir, failed)
                    print(f"[{variant}] preserved incomplete output at {failed}", flush=True)
                raise
            finally:
                outputs = None
                processed_views = []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        suite = {
            "schema": "noesis.mapanything.prior_variant_suite.v1",
            "generated_at": _utc_now(),
            "scan_dir": str(scan_dir),
            "prepared_frames_manifest_sha256": _sha256(
                scan_dir / "prepared_frames_manifest.json"
            ),
            "carrier": {
                "coordinate_frame": coordinate_frame,
                "transform_source": (
                    str(carrier_transform_source)
                    if carrier_transform_source is not None
                    else None
                ),
                "transform_source_sha256": (
                    _sha256(carrier_transform_source)
                    if carrier_transform_source is not None
                    else None
                ),
                "aligned_to_noesis": carrier_transform_source is not None,
            },
            "world_from_da3_source": (
                str(carrier_transform_source)
                if carrier_transform_source is not None
                else None
            ),
            "world_from_da3_source_sha256": (
                _sha256(carrier_transform_source)
                if carrier_transform_source is not None
                else None
            ),
            "static_revision": str(static.revision) if static is not None else None,
            "calibration": (
                str(args.calibration.resolve()) if args.calibration is not None else None
            ),
            "model": model_metadata,
            "sparse_depth_prior": sparse.metrics,
            "variants": suite_rows,
        }
        (output_root / "variant_suite_manifest.json").write_text(
            json.dumps(suite, indent=2), encoding="utf-8"
        )
        print(f"Variant suite complete: {output_root}", flush=True)
    finally:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MapAnythingScanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
