#!/usr/bin/env python3
"""Run Roomform's Point Transformer V3 object-lifting stage locally.

The semantic model is Pointcept's released ScanNet-20 PTv3 checkpoint.  This
module intentionally contains no Modal integration: it turns an RGB PLY into
Roomform-compatible ``labels.npz`` and can lift those labels into SceneObjects.

Pointcept code and its released PTv3 checkpoints are MIT licensed.  The
checkpoint was trained on ScanNet v2 data, whose terms are non-commercial.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


GRID_M = 0.02
DEFAULT_MAX_POINTS = 150_000
CLASSES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower_curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

EXCLUDE = {"wall", "floor"}
MIN_CLUSTER_PTS = 200
CLUSTER_RADIUS_M = 0.05
MAX_OBJECT_XY_M = 4.0
MAX_OBJECT_Z_M = 3.2


# This is Roomform's released pointlabel configuration: the shipped ScanNet
# base model with FlashAttention disabled in favor of plain upcast attention.
MODEL_CONFIG = {
    "type": "DefaultSegmentorV2",
    "num_classes": 20,
    "backbone_out_channels": 64,
    "backbone": {
        "type": "PT-v3m1",
        "in_channels": 6,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 2, 2, 2),
        "enc_depths": (2, 2, 2, 6, 2),
        "enc_channels": (32, 64, 128, 256, 512),
        "enc_num_head": (2, 4, 8, 16, 32),
        "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
        "dec_depths": (2, 2, 2, 2),
        "dec_channels": (64, 64, 128, 256),
        "dec_num_head": (4, 4, 8, 16),
        "dec_patch_size": (1024, 1024, 1024, 1024),
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path": 0.3,
        "shuffle_orders": True,
        "pre_norm": True,
        "enable_rpe": False,
        "enable_flash": False,
        "upcast_attention": True,
        "upcast_softmax": True,
        "cls_mode": False,
        "pdnorm_bn": False,
        "pdnorm_ln": False,
        "pdnorm_decouple": True,
        "pdnorm_adaptive": False,
        "pdnorm_affine": True,
        "pdnorm_conditions": ("ScanNet", "S3DIS", "Structured3D"),
    },
    "criteria": [{"type": "CrossEntropyLoss", "loss_weight": 1.0}],
}


def _pointcept_builder(pointcept_root: Path) -> Any:
    """Import only PTv3's registry surface from a Pointcept checkout.

    Pointcept's stock model-package initializer eagerly imports optional
    backbones such as pointops-based models.  Roomform's remote image replaces
    that initializer with three imports.  The local adapter achieves the same
    isolation in memory, leaving the external checkout untouched.
    """
    root = pointcept_root.expanduser().resolve()
    package_dir = root / "pointcept"
    models_dir = package_dir / "models"
    required = (
        package_dir / "__init__.py",
        models_dir / "builder.py",
        models_dir / "default.py",
        models_dir / "point_transformer_v3" / "point_transformer_v3m1_base.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"invalid Pointcept checkout at {root}; missing: {', '.join(missing)}"
        )

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    loaded_pointcept = sys.modules.get("pointcept")
    if loaded_pointcept is not None:
        loaded_file = Path(getattr(loaded_pointcept, "__file__", "")).resolve()
        if package_dir not in loaded_file.parents:
            raise RuntimeError(
                "a different Pointcept checkout is already imported: "
                f"{loaded_file}"
            )
    else:
        loaded_pointcept = importlib.import_module("pointcept")

    if "pointcept.models" not in sys.modules:
        models_package = types.ModuleType("pointcept.models")
        models_package.__path__ = [str(models_dir)]
        models_package.__package__ = "pointcept.models"
        models_package.__spec__ = importlib.machinery.ModuleSpec(
            "pointcept.models", loader=None, is_package=True
        )
        sys.modules["pointcept.models"] = models_package
        setattr(loaded_pointcept, "models", models_package)

    try:
        builder = importlib.import_module("pointcept.models.builder")
        importlib.import_module("pointcept.models.default")
        importlib.import_module(
            "pointcept.models.point_transformer_v3.point_transformer_v3m1_base"
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "local PTv3 dependencies are incomplete; missing Python module "
            f"{exc.name!r}"
        ) from exc
    return builder.build_model


def _voxel_keep(points: np.ndarray, grid_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic first-point indices and non-negative voxel coords."""
    grid = np.floor((points - points.min(axis=0)) / grid_m).astype(np.int64)
    _, keep = np.unique(grid, axis=0, return_index=True)
    keep.sort()
    return keep, grid


def _bounded_sample(
    points: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if max_points < 1_024:
        raise ValueError("max_points must be at least 1024")
    grid_m = GRID_M
    while True:
        keep, grid = _voxel_keep(points, grid_m)
        if len(keep) <= max_points:
            return keep, grid[keep], grid_m
        if grid_m < 0.03:
            grid_m = round(grid_m + 0.005, 8)
        elif grid_m < 0.04:
            grid_m = round(grid_m + 0.0025, 8)
        elif grid_m < 0.08:
            grid_m = round(grid_m + 0.01, 8)
        else:
            grid_m *= 1.25
        if grid_m > 1.0:
            raise RuntimeError(
                f"could not bound {len(points):,} points below {max_points:,} "
                "without exceeding a 1 m semantic voxel"
            )


def _load_ply(
    point_cloud: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    try:
        from plyfile import PlyData
        from scipy.spatial import cKDTree
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"local PTv3 dependencies are incomplete; missing {exc.name!r}"
        ) from exc

    vertex = PlyData.read(str(point_cloud))["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"PLY is missing required fields: {', '.join(missing)}")

    points = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=1)
    points = np.asarray(points, dtype=np.float32)
    colors = np.stack([vertex[channel] for channel in ("red", "green", "blue")], axis=1)
    colors = np.asarray(colors, dtype=np.float32)
    valid = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(colors), axis=1)
    points, colors = points[valid], colors[valid]
    valid_input_points = len(points)
    if len(points) < 1_024:
        raise ValueError(f"PLY has too few valid RGB points for PTv3: {len(points):,}")
    if float(colors.min()) < 0.0 or float(colors.max()) > 255.0:
        raise ValueError("PLY RGB values must be in the 0..255 range")

    normal_fields = {"nx", "ny", "nz"}
    if normal_fields.issubset(names):
        normals = np.stack([vertex[name][valid] for name in ("nx", "ny", "nz")], axis=1)
        normals = np.asarray(normals, dtype=np.float32)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        if not np.all(np.isfinite(normals)) or np.any(norm < 1e-8):
            raise ValueError("PLY contains invalid or zero-length normals")
        normals /= norm
        return points, colors, normals, "ply", valid_input_points

    # Estimate normals after the checkpoint's 2 cm base voxelization.  This
    # keeps the k-NN calculation bounded and matches Roomform's remote stage.
    keep, _ = _voxel_keep(points, GRID_M)
    points, colors = points[keep], colors[keep]
    neighbors = min(16, len(points))
    _, neighbor_index = cKDTree(points).query(points, k=neighbors, workers=-1)
    if neighbors == 1:
        neighbor_index = neighbor_index[:, None]
    neighborhood = points[neighbor_index]
    neighborhood -= neighborhood.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", neighborhood, neighborhood) / neighbors
    normals = np.linalg.eigh(covariance)[1][:, :, 0].astype(np.float32)
    return points, colors, normals, "estimated_knn16", valid_input_points


def segment_local(
    point_cloud: Path,
    out_path: Path,
    *,
    pointcept_root: Path,
    checkpoint: Path,
    max_points: int = DEFAULT_MAX_POINTS,
    semantic_grid_m: float | None = None,
    tile_overlap_m: float = 0.75,
) -> dict[str, Any]:
    """Run PTv3 locally and write Roomform-compatible semantic labels."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("local PTv3 requires PyTorch") from exc

    point_cloud = point_cloud.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    if not point_cloud.is_file():
        raise FileNotFoundError(f"point cloud does not exist: {point_cloud}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PTv3 checkpoint does not exist: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("local PTv3 inference requires a CUDA GPU")

    started = time.perf_counter()
    points, colors, normals, normal_source, input_points = _load_ply(point_cloud)
    normal_points = len(points)

    lo, hi = points.min(axis=0), points.max(axis=0)
    center = np.asarray(
        [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]],
        dtype=np.float32,
    )
    coord = points - center
    if semantic_grid_m is None:
        keep, grid_coord, grid_m = _bounded_sample(coord, max_points)
    else:
        if semantic_grid_m < GRID_M:
            raise ValueError(
                f"semantic_grid_m must be at least PTv3's {GRID_M:.2f} m "
                "training grid"
            )
        keep, full_grid = _voxel_keep(coord, semantic_grid_m)
        grid_coord = full_grid[keep]
        grid_m = semantic_grid_m
    sampled_points = np.ascontiguousarray(points[keep], dtype=np.float32)
    feat = np.concatenate(
        (colors[keep] / 127.5 - 1.0, normals[keep]),
        axis=1,
        dtype=np.float32,
    )

    build_model = _pointcept_builder(pointcept_root)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.cuda.reset_peak_memory_stats(device)

    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("state_dict"), dict):
        raise RuntimeError("PTv3 checkpoint has no state_dict mapping")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in loaded["state_dict"].items()
    }
    model = build_model(MODEL_CONFIG).to(device).eval()
    model.load_state_dict(state_dict)

    sampled_coord = np.ascontiguousarray(coord[keep], dtype=np.float32)

    def infer(active_model: Any, indices: np.ndarray) -> np.ndarray:
        data = {
            "coord": torch.from_numpy(sampled_coord[indices]).float().to(device),
            "grid_coord": torch.from_numpy(
                np.ascontiguousarray(grid_coord[indices])
            ).long().to(device),
            "feat": torch.from_numpy(np.ascontiguousarray(feat[indices])).to(device),
            "offset": torch.tensor([len(indices)], dtype=torch.int64, device=device),
        }
        with torch.inference_mode():
            tile_label = active_model(data)["seg_logits"].argmax(dim=1)
        result = tile_label.cpu().numpy().astype(np.uint8, copy=False)
        del tile_label, data
        return result

    inference_started = time.perf_counter()
    tile_sizes: list[int] = []
    if len(sampled_points) <= max_points:
        labels = infer(model, np.arange(len(sampled_points)))
        execution = "whole_room"
    else:
        execution = "overlapping_xy_slabs"
        axis = int(np.argmax(np.ptp(sampled_coord[:, :2], axis=0)))
        axis_values = sampled_coord[:, axis]
        tile_count = max(2, int(np.ceil(len(sampled_points) / max_points)))
        while True:
            edges = np.quantile(axis_values, np.linspace(0.0, 1.0, tile_count + 1))
            expanded_sizes = []
            for tile_index in range(tile_count):
                lo, hi = edges[tile_index], edges[tile_index + 1]
                expanded_sizes.append(
                    int(
                        np.count_nonzero(
                            (axis_values >= lo - tile_overlap_m)
                            & (axis_values <= hi + tile_overlap_m)
                        )
                    )
                )
            if max(expanded_sizes) <= max_points:
                break
            tile_count += 1
            if tile_count > 32:
                raise RuntimeError(
                    f"could not tile {len(sampled_points):,} semantic points "
                    f"below {max_points:,} points per tile"
                )
        labels = np.full(len(sampled_points), 255, dtype=np.uint8)
        for tile_index in range(tile_count):
            lo, hi = edges[tile_index], edges[tile_index + 1]
            core = (axis_values >= lo) & (
                axis_values <= hi if tile_index == tile_count - 1 else axis_values < hi
            )
            expanded = (axis_values >= lo - tile_overlap_m) & (
                axis_values <= hi + tile_overlap_m
            )
            expanded_indices = np.flatnonzero(expanded)
            core_positions = core[expanded_indices]
            tile_labels = infer(model, expanded_indices)
            labels[expanded_indices[core_positions]] = tile_labels[core_positions]
            tile_sizes.append(int(len(expanded_indices)))
        if np.any(labels == 255):
            raise RuntimeError("PTv3 tiling left semantic points unlabeled")
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    peak_gpu_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    del model
    torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pts=sampled_points,
        label=labels,
        classes=np.asarray(CLASSES),
    )
    cluster_radius_m, min_cluster_points = _cluster_parameters(grid_m)
    return {
        "output": str(out_path),
        "input_points": int(input_points),
        "normal_points": int(normal_points),
        "semantic_points": int(len(sampled_points)),
        "semantic_grid_m": float(grid_m),
        "execution": execution,
        "tile_count": int(len(tile_sizes) or 1),
        "tile_points": tile_sizes or [int(len(sampled_points))],
        "tile_overlap_m": float(tile_overlap_m if tile_sizes else 0.0),
        "cluster_radius_m": float(cluster_radius_m),
        "min_cluster_points": int(min_cluster_points),
        "normal_source": normal_source,
        "inference_seconds": float(inference_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "peak_gpu_mib": float(peak_gpu_mib),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "fp32_tf32",
    }


def _cluster_parameters(sample_grid_m: float) -> tuple[float, int]:
    """Preserve Roomform's physical cluster scale after GPU-driven coarsening."""
    radius = max(CLUSTER_RADIUS_M, sample_grid_m * 1.75)
    min_points = max(
        30,
        round(MIN_CLUSTER_PTS * (GRID_M / sample_grid_m) ** 2),
    )
    return radius, min_points


def _components(points: np.ndarray, radius_m: float, min_points: int):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(points).query_pairs(radius_m, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(len(points), len(points)),
    )
    _, component = connected_components(graph, directed=False)
    for component_id in np.unique(component):
        mask = component == component_id
        if mask.sum() >= min_points:
            yield points[mask]


def _footprint_axis(xy: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError:
        return np.asarray([1.0, 0.0])
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    best_angle, best_area = 0.0, np.inf
    for angle in angles:
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = hull @ np.asarray([[cosine, -sine], [sine, cosine]])
        area = np.ptp(rotated[:, 0]) * np.ptp(rotated[:, 1])
        if area < best_area:
            best_angle, best_area = angle, area
    return np.asarray([np.cos(best_angle), np.sin(best_angle)])


def _box(class_name: str, points: np.ndarray, frame_shift: np.ndarray) -> Any | None:
    from roomform.contracts import SceneObject

    xy = points[:, :2]
    mean = xy.mean(axis=0)
    centered = xy - mean
    axis = _footprint_axis(centered)
    perpendicular = np.asarray([-axis[1], axis[0]])
    u, t = centered @ axis, centered @ perpendicular
    u0, u1 = np.percentile(u, [1, 99])
    t0, t1 = np.percentile(t, [1, 99])
    z0, z1 = np.percentile(points[:, 2], [1, 99])
    if max(u1 - u0, t1 - t0) > MAX_OBJECT_XY_M or z1 - z0 > MAX_OBJECT_Z_M:
        return None
    center_xy = mean + axis * (u0 + u1) / 2
    center_xy += perpendicular * (t0 + t1) / 2
    return SceneObject(
        cls=class_name,
        center=(
            float(center_xy[0] - frame_shift[0]),
            float(center_xy[1] - frame_shift[1]),
            float((z0 + z1) / 2 - frame_shift[2]),
        ),
        size=(
            max(float(u1 - u0), GRID_M),
            max(float(t1 - t0), GRID_M),
            max(float(z1 - z0), GRID_M),
        ),
        heading=float(np.arctan2(axis[1], axis[0])),
        source="pointlabel",
    )


def lift_local_labels(
    labels_path: Path,
    frame_shift: tuple[float, float, float],
    *,
    sample_grid_m: float = GRID_M,
) -> list[Any]:
    """Lift semantic labels into Roomform SceneObjects without Modal imports."""
    labels_path = labels_path.expanduser().resolve()
    with np.load(labels_path, allow_pickle=False) as data:
        required = {"pts", "label", "classes"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"labels NPZ is missing: {', '.join(missing)}")
        points = np.asarray(data["pts"], dtype=np.float32)
        labels = np.asarray(data["label"])
        classes = [str(value) for value in data["classes"]]
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
        raise ValueError(
            f"invalid labels arrays: pts={points.shape}, label={labels.shape}"
        )

    shift = np.asarray(frame_shift, dtype=np.float64)
    cluster_radius_m, min_cluster_points = _cluster_parameters(sample_grid_m)
    objects: list[Any] = []
    for class_id, class_name in enumerate(classes):
        if class_name in EXCLUDE:
            continue
        class_points = points[labels == class_id]
        if len(class_points) < min_cluster_points:
            continue
        objects.extend(
            box
            for component in _components(
                class_points, cluster_radius_m, min_cluster_points
            )
            if (box := _box(class_name, component, shift)) is not None
        )
    return objects


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_cloud", type=Path, help="RGB PLY in metric Z-up coordinates")
    parser.add_argument("out", type=Path, help="output labels.npz")
    parser.add_argument("--pointcept-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument("--grid-m", type=float, choices=(0.02,), default=0.02)
    parser.add_argument("--tile-overlap-m", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = segment_local(
        args.point_cloud,
        args.out,
        pointcept_root=args.pointcept_root,
        checkpoint=args.checkpoint,
        max_points=args.max_points,
        semantic_grid_m=args.grid_m,
        tile_overlap_m=args.tile_overlap_m,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
