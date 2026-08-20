#!/usr/bin/env python3
"""Render a dashboard-style BEV from an existing multi-room PCF join.

The renderer is presentation-only.  It derives the reference-camera ground
frame from an immutable Scene Prior manifest, builds the same 2.5-D evidence
layers used by the dashboard, and writes a clean review PNG plus a provenance
manifest.  The registered backend-world NPZ is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.coordinate_frames import transform_positions  # noqa: E402
from tools.mapanything_phone_scan.render_pcf_multiroom_presentation import (  # noqa: E402
    derive_presentation_frame,
)


class MultiroomBEVError(RuntimeError):
    """Raised when a joined BEV cannot be derived without ambiguity."""


@dataclass(frozen=True)
class BEVConfig:
    resolution_m: float = 0.025
    safety_padding_m: float = 1.0
    floor_support_band_m: float = 0.12
    obstacle_min_height_m: float = 0.18
    obstacle_max_height_m: float = 2.20
    obstacle_min_support: int = 3
    max_source_height_m: float = 3.20
    planar_floor_expansion_cells: int = 2
    planar_floor_min_component_fraction: float = 0.02
    display_scale: int = 2


@dataclass(frozen=True)
class BEVGrid:
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    rows: int
    columns: int
    resolution_m: float


_REQUIRED_NPZ_FIELDS = {
    "points",
    "colors",
    "owner_room_id",
    "fixed_camera_positions",
    "moving_camera_positions",
}

_FLOOR_RGB = np.asarray([184, 180, 170], dtype=np.uint8)
_INFERNO_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (0, 0, 4)),
    (0.2, (35, 6, 59)),
    (0.4, (99, 23, 94)),
    (0.6, (159, 43, 73)),
    (0.8, (218, 83, 32)),
    (1.0, (252, 255, 164)),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MultiroomBEVError(f"{path} does not contain a JSON object")
    return value


def _full_extent_grid(points: np.ndarray, config: BEVConfig) -> BEVGrid:
    finite_xz = points[np.isfinite(points).all(axis=1)][:, [0, 2]]
    if not len(finite_xz):
        raise MultiroomBEVError("source contains no finite presentation points")
    resolution = float(config.resolution_m)
    padding = float(config.safety_padding_m)
    low = np.floor((np.min(finite_xz, axis=0) - padding) / resolution) * resolution
    high = np.ceil((np.max(finite_xz, axis=0) + padding) / resolution) * resolution
    columns = int(round((float(high[0]) - float(low[0])) / resolution))
    rows = int(round((float(high[1]) - float(low[1])) / resolution))
    if rows <= 0 or columns <= 0 or rows * columns > 20_000_000:
        raise MultiroomBEVError(
            f"unsafe BEV grid {columns}x{rows} from full source extent"
        )
    return BEVGrid(
        min_x=float(low[0]),
        max_x=float(high[0]),
        min_z=float(low[1]),
        max_z=float(high[1]),
        rows=rows,
        columns=columns,
        resolution_m=resolution,
    )


def _group_percentile(
    linear: np.ndarray,
    values: np.ndarray,
    *,
    cell_count: int,
    percentile: float,
) -> np.ndarray:
    """Vectorized NumPy linear percentile for each occupied raster cell."""

    output = np.full(cell_count, np.nan, dtype=np.float32)
    if not len(linear):
        return output
    order = np.lexsort((values, linear))
    sorted_linear = linear[order]
    sorted_values = values[order].astype(np.float64, copy=False)
    starts = np.flatnonzero(np.r_[True, sorted_linear[1:] != sorted_linear[:-1]])
    stops = np.r_[starts[1:], len(sorted_linear)]
    counts = stops - starts
    positions = (float(percentile) / 100.0) * (counts - 1)
    lows = np.floor(positions).astype(np.int64)
    highs = np.ceil(positions).astype(np.int64)
    fractions = positions - lows
    result = (
        sorted_values[starts + lows] * (1.0 - fractions)
        + sorted_values[starts + highs] * fractions
    )
    output[sorted_linear[starts]] = result.astype(np.float32)
    return output


def _derive_planar_floor(
    floor_supported: np.ndarray,
    *,
    expansion_cells: int,
    min_component_fraction: float,
) -> np.ndarray:
    """Python equivalent of oai2-fe/src/lib/planarFloor.mjs."""

    from scipy import ndimage

    seeds = np.asarray(floor_supported, dtype=bool)
    if not np.any(seeds):
        return seeds.copy()
    expanded = ndimage.binary_dilation(
        seeds,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(0, min(8, int(expansion_cells))),
    )
    # scipy's default fill-hole connectivity is four-neighbour, matching the
    # browser implementation's edge flood of the complement.
    candidate = ndimage.binary_fill_holes(expanded)
    labels, component_count = ndimage.label(
        candidate, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if component_count <= 0:
        return np.zeros_like(seeds)
    areas = np.bincount(labels.reshape(-1), minlength=component_count + 1)
    supports = np.bincount(
        labels.reshape(-1),
        weights=seeds.reshape(-1).astype(np.uint8),
        minlength=component_count + 1,
    )
    component_ids = np.arange(1, component_count + 1)
    primary = int(
        component_ids[
            np.lexsort((areas[component_ids], supports[component_ids]))[-1]
        ]
    )
    fraction = min(0.25, max(0.0, float(min_component_fraction)))
    minimum_area = max(16, math.ceil(int(areas[primary]) * fraction))
    minimum_support = max(2, math.ceil(float(supports[primary]) * fraction))
    retained = np.zeros(component_count + 1, dtype=bool)
    retained[primary] = True
    retained[component_ids] |= (
        (areas[component_ids] >= minimum_area)
        & (supports[component_ids] >= minimum_support)
    )
    return retained[labels]


def _inferno(values: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    result = np.empty(normalized.shape + (3,), dtype=np.uint8)
    for index in range(len(_INFERNO_STOPS) - 1):
        position_a, color_a = _INFERNO_STOPS[index]
        position_b, color_b = _INFERNO_STOPS[index + 1]
        selected = (normalized >= position_a) & (normalized <= position_b)
        if not np.any(selected):
            continue
        fraction = (normalized[selected] - position_a) / (position_b - position_a)
        rgb_a = np.asarray(color_a, dtype=np.float32)
        rgb_b = np.asarray(color_b, dtype=np.float32)
        result[selected] = np.rint(
            rgb_a[None] + (rgb_b - rgb_a)[None] * fraction[:, None]
        ).astype(np.uint8)
    return result


def _build_layers(
    points: np.ndarray,
    grid: BEVGrid,
    config: BEVConfig,
) -> dict[str, np.ndarray | int | float]:
    finite = np.isfinite(points).all(axis=1)
    columns = np.floor((points[:, 0] - grid.min_x) / grid.resolution_m).astype(
        np.int64
    )
    rows = np.floor((grid.max_z - points[:, 2]) / grid.resolution_m).astype(
        np.int64
    )
    inside = (
        finite
        & (columns >= 0)
        & (columns < grid.columns)
        & (rows >= 0)
        & (rows < grid.rows)
    )
    vertical = (
        (points[:, 1] >= -float(config.floor_support_band_m))
        & (points[:, 1] <= float(config.max_source_height_m))
    )
    selected = inside & vertical
    linear = rows[selected] * grid.columns + columns[selected]
    heights = points[selected, 1].astype(np.float32)
    cell_count = grid.rows * grid.columns
    point_count = np.bincount(linear, minlength=cell_count).astype(np.uint32)
    floor_point = np.abs(heights) <= float(config.floor_support_band_m)
    obstacle_point = (
        (heights >= float(config.obstacle_min_height_m))
        & (heights <= float(config.obstacle_max_height_m))
    )
    floor_count = np.bincount(
        linear[floor_point], minlength=cell_count
    ).astype(np.uint32)
    obstacle_count = np.bincount(
        linear[obstacle_point], minlength=cell_count
    ).astype(np.uint32)
    relevant_height = (
        (heights >= 0.0) & (heights <= float(config.obstacle_max_height_m))
    )
    height_p95 = _group_percentile(
        linear[relevant_height],
        heights[relevant_height],
        cell_count=cell_count,
        percentile=95.0,
    )
    shape = (grid.rows, grid.columns)
    observed = (point_count > 0).reshape(shape)
    floor_supported = (floor_count > 0).reshape(shape)
    obstacle = (obstacle_count >= int(config.obstacle_min_support)).reshape(shape)
    walkable = observed & floor_supported & ~obstacle
    obstacle_height = np.where(
        obstacle, height_p95.reshape(shape), np.nan
    ).astype(np.float32)
    planar_floor = _derive_planar_floor(
        floor_supported,
        expansion_cells=config.planar_floor_expansion_cells,
        min_component_fraction=config.planar_floor_min_component_fraction,
    )
    return {
        "observed": observed,
        "floor_supported": floor_supported,
        "obstacle": obstacle,
        "walkable": walkable,
        "obstacle_height": obstacle_height,
        "planar_floor": planar_floor,
        "finite_point_count": int(np.count_nonzero(finite)),
        "raster_admitted_point_count": int(np.count_nonzero(selected)),
        "vertical_rejected_point_count": int(np.count_nonzero(inside & ~vertical)),
    }


def _compose_image(
    layers: dict[str, np.ndarray | int | float],
) -> tuple[np.ndarray, dict[str, Any]]:
    observed = np.asarray(layers["observed"], dtype=bool)
    walkable = np.asarray(layers["walkable"], dtype=bool)
    planar_floor = np.asarray(layers["planar_floor"], dtype=bool)
    obstacle_height = np.asarray(layers["obstacle_height"], dtype=np.float32)
    obstacle_visible = np.isfinite(obstacle_height) & (obstacle_height > 0.05)
    finite_obstacle = obstacle_height[np.isfinite(obstacle_height)]
    obstacle_value_max = (
        float(np.percentile(finite_obstacle, 99.0))
        if len(finite_obstacle)
        else 1.0
    )
    if not math.isfinite(obstacle_value_max) or obstacle_value_max <= 0.0:
        obstacle_value_max = 1.0
    image = np.zeros(observed.shape + (3,), dtype=np.uint8)
    floor_visible = (planar_floor | walkable) & ~obstacle_visible
    image[floor_visible] = _FLOOR_RGB
    normalized = np.clip(obstacle_height / obstacle_value_max, 0.0, 1.0)
    image[obstacle_visible] = _inferno(normalized[obstacle_visible])
    metrics = {
        "observed_cell_count": int(np.count_nonzero(observed)),
        "floor_supported_cell_count": int(
            np.count_nonzero(layers["floor_supported"])
        ),
        "walkable_cell_count": int(np.count_nonzero(walkable)),
        "obstacle_cell_count": int(np.count_nonzero(layers["obstacle"])),
        "planar_floor_cell_count": int(np.count_nonzero(planar_floor)),
        "matte_floor_pixel_count_source_grid": int(np.count_nonzero(floor_visible)),
        "inferno_obstacle_pixel_count_source_grid": int(
            np.count_nonzero(obstacle_visible)
        ),
        "black_pixel_count_source_grid": int(
            image.shape[0] * image.shape[1]
            - np.count_nonzero(floor_visible | obstacle_visible)
        ),
        "obstacle_height_value_min_m": 0.0,
        "obstacle_height_value_max_p99_m": obstacle_value_max,
    }
    return image, metrics


def _write_png(
    path: Path,
    source_image: np.ndarray,
    *,
    grid: BEVGrid,
    display_scale: int,
) -> tuple[int, int]:
    from PIL import Image, ImageDraw, ImageFont

    scale = max(1, int(display_scale))
    source = Image.fromarray(source_image, mode="RGB")
    if scale > 1:
        output = source.resize(
            (source.width * scale, source.height * scale),
            resample=Image.Resampling.BICUBIC,
        )
    else:
        output = source

    # Same camera-local raster convention as the dashboard: +X right and +Z
    # forward/up.  This marker is presentation metadata, not scene geometry.
    if grid.min_x <= 0.0 <= grid.max_x and grid.min_z <= 0.0 <= grid.max_z:
        x = ((0.0 - grid.min_x) / (grid.max_x - grid.min_x)) * output.width
        y = ((grid.max_z - 0.0) / (grid.max_z - grid.min_z)) * output.height
        draw = ImageDraw.Draw(output)
        size = max(8, int(round(min(output.size) * 0.012)))
        triangle = [
            (x, y - 1.45 * size),
            (x + size, y + 0.45 * size),
            (x - size, y + 0.45 * size),
        ]
        draw.polygon(triangle, fill=(255, 215, 64), outline=(0, 0, 0))
        draw.line(
            [(x, y - 1.45 * size), (x, y - 3.0 * size)],
            fill=(0, 0, 0),
            width=max(2, size // 5),
        )
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(11, size))
        except OSError:
            font = ImageFont.load_default()
        label_position = (x + 1.25 * size, y - 0.65 * size)
        draw.text(
            label_position,
            "CAM",
            font=font,
            fill=(255, 224, 102),
            stroke_width=max(1, size // 7),
            stroke_fill=(0, 0, 0),
        )
    output.save(path, format="PNG", optimize=True)
    return output.width, output.height


def render_joined_bev(
    *,
    source_npz: Path,
    scene_prior_manifest: Path,
    output_dir: Path,
    source_reintegration_manifest: Path | None = None,
    source_presentation_manifest: Path | None = None,
    config: BEVConfig = BEVConfig(),
    output_stem: str = "kitchen_family_joined_bev",
) -> dict[str, Any]:
    source_npz = source_npz.resolve()
    scene_prior_manifest = scene_prior_manifest.resolve()
    output_dir = output_dir.resolve()
    source_reintegration_manifest = (
        source_reintegration_manifest.resolve()
        if source_reintegration_manifest is not None
        else None
    )
    source_presentation_manifest = (
        source_presentation_manifest.resolve()
        if source_presentation_manifest is not None
        else None
    )
    if output_dir.exists():
        raise MultiroomBEVError(f"output directory already exists: {output_dir}")
    if config.resolution_m != 0.025:
        raise MultiroomBEVError("dashboard-faithful review rendering requires 0.025 m")
    if not output_stem or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in output_stem
    ):
        raise MultiroomBEVError("output_stem contains unsupported characters")

    source_hash_before = _sha256(source_npz)
    source_size_before = source_npz.stat().st_size
    frame = derive_presentation_frame(scene_prior_manifest)
    reintegration_payload = _load_json(source_reintegration_manifest)
    presentation_payload = _load_json(source_presentation_manifest)
    with np.load(source_npz, allow_pickle=False) as archive:
        missing = sorted(_REQUIRED_NPZ_FIELDS.difference(archive.files))
        if missing:
            raise MultiroomBEVError(f"{source_npz} is missing {missing}")
        backend_points = np.asarray(archive["points"], dtype=np.float32)
        colors = np.asarray(archive["colors"], dtype=np.uint8)
        owners = np.asarray(archive["owner_room_id"], dtype=np.uint8)
        fixed_path = np.asarray(archive["fixed_camera_positions"], dtype=np.float32)
        moving_path = np.asarray(archive["moving_camera_positions"], dtype=np.float32)
    if backend_points.ndim != 2 or backend_points.shape[1] != 3 or not len(backend_points):
        raise MultiroomBEVError("source points must be one non-empty N x 3 array")
    if colors.shape != backend_points.shape or owners.shape != (len(backend_points),):
        raise MultiroomBEVError("source color/owner arrays do not match source points")
    unique_owners = sorted(np.unique(owners).tolist())
    if any(owner < 1 or owner > 255 for owner in unique_owners):
        raise MultiroomBEVError("owner_room_id must contain positive uint8 IDs")

    presentation_points = transform_positions(
        backend_points, frame.world_to_presentation
    ).astype(np.float32)
    grid = _full_extent_grid(presentation_points, config)
    layers = _build_layers(presentation_points, grid, config)
    source_image, layer_metrics = _compose_image(layers)
    output_dir.mkdir(parents=True)
    png_path = output_dir / f"{output_stem}_floorplan.png"
    image_width, image_height = _write_png(
        png_path,
        source_image,
        grid=grid,
        display_scale=config.display_scale,
    )

    source_hash_after = _sha256(source_npz)
    source_size_after = source_npz.stat().st_size
    source_unchanged = (
        source_hash_before == source_hash_after
        and source_size_before == source_size_after
    )
    if not source_unchanged:
        raise MultiroomBEVError("backend source NPZ changed during BEV rendering")
    finite_xz = presentation_points[
        np.isfinite(presentation_points).all(axis=1)
    ][:, [0, 2]]
    transform = frame.world_to_presentation
    linear = transform[:3, :3]
    camera_ground = transform_positions(
        frame.camera_position_world_m[None], transform
    )[0]
    manifest: dict[str, Any] = {
        "schema": "noesis.pcf.multiroom_bev_presentation.v1",
        "generated_at": _utc_now(),
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "presentation_only": True,
        "backend_geometry": {
            "mutated": False,
            "transformed_npz_written": False,
            "source_coordinate_frame": frame.source_coordinate_frame,
            "source_npz": str(source_npz),
            "source_npz_size_bytes_before": source_size_before,
            "source_npz_size_bytes_after": source_size_after,
            "source_npz_sha256_before": source_hash_before,
            "source_npz_sha256_after": source_hash_after,
            "source_npz_unchanged": source_unchanged,
            "backend_points_sha256": _array_sha256(backend_points),
            "backend_fixed_camera_path_sha256": _array_sha256(fixed_path),
            "backend_moving_camera_path_sha256": _array_sha256(moving_path),
            "point_count": int(len(backend_points)),
            "owner_point_counts": {
                str(owner): int(np.count_nonzero(owners == owner))
                for owner in unique_owners
            },
        },
        "source_reintegration_manifest": (
            {
                "path": str(source_reintegration_manifest),
                "sha256": _sha256(source_reintegration_manifest),
                "status": reintegration_payload.get("status"),
                "registration": reintegration_payload.get("registration"),
            }
            if source_reintegration_manifest is not None
            and reintegration_payload is not None
            else None
        ),
        "source_presentation_manifest": (
            {
                "path": str(source_presentation_manifest),
                "sha256": _sha256(source_presentation_manifest),
                "status": presentation_payload.get("status"),
                "presentation_only": presentation_payload.get("presentation_only"),
            }
            if source_presentation_manifest is not None
            and presentation_payload is not None
            else None
        ),
        "scene_prior_manifest": {
            "path": str(scene_prior_manifest),
            "sha256": _sha256(scene_prior_manifest),
            "prior_id": _load_json(scene_prior_manifest).get("prior_id"),
        },
        "presentation_frame": {
            "reference_camera_id": frame.reference_camera_id,
            "target_coordinate_frame": frame.target_coordinate_frame,
            "world_to_presentation_row_major": transform.tolist(),
            "linear_determinant": float(np.linalg.det(linear)),
            "orthonormal_max_abs_error": float(
                np.max(np.abs(linear.T @ linear - np.eye(3)))
            ),
            "reference_camera_presentation_xyz_m": camera_ground.tolist(),
            "screen_contract": {
                "horizontal": "camera_right_positive_x",
                "vertical": "camera_forward_positive_z_up",
                "camera_ground_origin_xz_m": [0.0, 0.0],
            },
        },
        "raster_derivation": {
            "algorithm": "dashboard_pcf_walkable_obstacle_composite_v1",
            "grid_resolution_m": config.resolution_m,
            "full_finite_point_extent_used": True,
            "percentile_or_authored_crop_applied": False,
            "safety_padding_m": config.safety_padding_m,
            "bounds_camera_local_ground_m": {
                "min_x": grid.min_x,
                "max_x": grid.max_x,
                "min_z": grid.min_z,
                "max_z": grid.max_z,
            },
            "raw_finite_point_bounds_camera_local_ground_m": {
                "min_x": float(np.min(finite_xz[:, 0])),
                "max_x": float(np.max(finite_xz[:, 0])),
                "min_z": float(np.min(finite_xz[:, 1])),
                "max_z": float(np.max(finite_xz[:, 1])),
            },
            "rows": grid.rows,
            "columns": grid.columns,
            "floor_support_band_m": config.floor_support_band_m,
            "obstacle_height_range_m": [
                config.obstacle_min_height_m,
                config.obstacle_max_height_m,
            ],
            "obstacle_min_support": config.obstacle_min_support,
            "max_source_height_m": config.max_source_height_m,
            "planar_floor": {
                "expansion_cells": config.planar_floor_expansion_cells,
                "enclosed_sampling_gaps_filled": True,
                "connectivity": "8-neighbour components; 4-neighbour exterior flood",
                "min_component_fraction": config.planar_floor_min_component_fraction,
            },
            "palette": {
                "floor_rgb_u8": _FLOOR_RGB.tolist(),
                "non_floor": "oai2-fe inferno stops",
                "unknown_rgb_u8": [0, 0, 0],
            },
            "display_resampling": "bicubic",
            "display_scale": config.display_scale,
            "background_guides_or_grid_rectangles_drawn": False,
        },
        "evidence_counts": {
            "finite_point_count": layers["finite_point_count"],
            "raster_admitted_point_count": layers["raster_admitted_point_count"],
            "vertical_rejected_point_count": layers[
                "vertical_rejected_point_count"
            ],
            **layer_metrics,
        },
        "outputs": {
            "png": {
                "path": png_path.name,
                "sha256": _sha256(png_path),
                "size_bytes": png_path.stat().st_size,
                "width_px": image_width,
                "height_px": image_height,
                "source_grid_width_px": grid.columns,
                "source_grid_height_px": grid.rows,
            }
        },
    }
    manifest_path = output_dir / f"{output_stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_test_scene_prior(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "contract": "noesis.scene_prior.revision",
                "prior_id": "sceneprior-test",
                "grid": {"coordinate_frame": "backend_world_m"},
                "derivation": {"floor_y_m": 0.0},
                "preview": {
                    "coordinate_frame": "camera_local_ground_m",
                    "orientation": "row_increases_camera_forward_column_increases_camera_right",
                    "reference_camera_id": "camera-test",
                    "camera_position_world_m": [0.0, 1.5, 0.0],
                    "camera_right_world_xz": [1.0, 0.0],
                    "camera_forward_world_xz": [0.0, -1.0],
                },
            }
        ),
        encoding="utf-8",
    )


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="multiroom_bev_smoke_") as value:
        root = Path(value)
        scene_prior = root / "scene_prior.json"
        _write_test_scene_prior(scene_prior)
        xy = np.asarray(
            [
                [x, z]
                for x in np.linspace(-0.5, 0.5, 21)
                for z in np.linspace(0.0, 1.0, 21)
            ],
            dtype=np.float32,
        )
        floor = np.column_stack((xy[:, 0], np.zeros(len(xy)), xy[:, 1]))
        obstacle = np.asarray(
            [[0.0, y, 0.5] for y in (0.25, 0.5, 0.75, 1.0)], dtype=np.float32
        )
        points = np.vstack((floor, obstacle))
        source = root / "source.npz"
        np.savez_compressed(
            source,
            points=points,
            colors=np.full(points.shape, 128, dtype=np.uint8),
            owner_room_id=np.r_[
                np.ones(len(floor), dtype=np.uint8),
                np.full(len(obstacle), 2, dtype=np.uint8),
            ],
            fixed_camera_positions=np.asarray([[0.0, 1.5, 0.0]], dtype=np.float32),
            moving_camera_positions=np.asarray([[0.0, 1.5, 0.5]], dtype=np.float32),
        )
        source_hash = _sha256(source)
        result = render_joined_bev(
            source_npz=source,
            scene_prior_manifest=scene_prior,
            output_dir=root / "output",
        )
        if _sha256(source) != source_hash:
            raise AssertionError("source NPZ changed during BEV self-test")
        if not result["backend_geometry"]["source_npz_unchanged"]:
            raise AssertionError("source immutability proof failed")
        counts = result["evidence_counts"]
        if counts["planar_floor_cell_count"] <= counts["floor_supported_cell_count"]:
            raise AssertionError("planar floor did not expand measured floor support")
        if counts["inferno_obstacle_pixel_count_source_grid"] < 1:
            raise AssertionError("synthetic obstacle was not rendered")
        if result["raster_derivation"]["percentile_or_authored_crop_applied"]:
            raise AssertionError("self-test unexpectedly enabled cropping")
        for name in (
            "kitchen_family_joined_bev_floorplan.png",
            "kitchen_family_joined_bev_manifest.json",
        ):
            if not (root / "output" / name).is_file():
                raise AssertionError(f"missing self-test output {name}")
    print("PCF multi-room joined BEV smoke: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path)
    parser.add_argument("--scene-prior-manifest", type=Path)
    parser.add_argument("--source-reintegration-manifest", type=Path)
    parser.add_argument("--source-presentation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-stem", default="kitchen_family_joined_bev")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.self_test:
        _self_test()
        return 0
    missing = [
        name
        for name in ("source_npz", "scene_prior_manifest", "output_dir")
        if getattr(arguments, name) is None
    ]
    if missing:
        raise MultiroomBEVError(
            "missing required arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    result = render_joined_bev(
        source_npz=arguments.source_npz,
        scene_prior_manifest=arguments.scene_prior_manifest,
        source_reintegration_manifest=arguments.source_reintegration_manifest,
        source_presentation_manifest=arguments.source_presentation_manifest,
        output_dir=arguments.output_dir,
        output_stem=arguments.output_stem,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
