"""Strict loading and floorplan serialization for cached scene-fusion diagnostics."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from noesis_core.coordinate_frames import (
    CoordinateFrameError,
    camera_ground_frame_from_camera_to_world,
    transform_positions,
)


MAX_CATALOG_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_POINTS_BYTES = 128 * 1024 * 1024
MAX_GRID_BYTES = 64 * 1024 * 1024
_POINT_ARRAYS = (
    "points_world_m",
    "colors_rgb_u8",
    "confidence",
    "provenance",
    "view_support",
)
_GRID_ARRAYS = (
    "height_agl_p95_m",
    "observed",
    "confidence",
    "point_count",
    "obstacle_height_m",
    "obstacle_mask",
    "floor_support_count",
    "floor_supported",
    "provenance",
)
_DIAGNOSTIC_SURFACE_HEIGHT_MAX_M = 1.80
_DIAGNOSTIC_FURNITURE_MIN_M = 0.12
_DIAGNOSTIC_FURNITURE_MAX_M = 1.65
_DIAGNOSTIC_HEIGHT_BIN_M = 0.05


class SceneFusionError(RuntimeError):
    """Raised when a configured scene-fusion artifact is invalid."""


def _read_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
        if size <= 0 or size > maximum:
            raise SceneFusionError(f"{label} size is outside its bound")
        return path.read_bytes()
    except SceneFusionError:
        raise
    except OSError as exc:
        raise SceneFusionError(f"cannot read {label}: {exc}") from exc


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneFusionError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SceneFusionError(f"{label} must contain an object")
    return value


def _relative(root: Path, value: Any, *, label: str) -> Path:
    raw = str(value or "").strip()
    posix = PurePosixPath(raw)
    if not raw or posix.is_absolute() or ".." in posix.parts or "\\" in raw:
        raise SceneFusionError(f"{label} must be a normalized relative path")
    candidate = root.joinpath(*posix.parts)
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SceneFusionError(f"{label} escapes its catalog root") from exc
    return candidate


def _verified_artifact(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    maximum: int,
    label: str,
) -> tuple[Path, bytes]:
    path = _relative(root, descriptor.get("path"), label=f"{label} path")
    data = _read_bytes(path, maximum=maximum, label=label)
    expected_size = int(descriptor.get("size_bytes") or 0)
    expected_sha = str(descriptor.get("sha256") or "")
    if len(data) != expected_size:
        raise SceneFusionError(f"{label} size fingerprint does not match")
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise SceneFusionError(f"{label} content fingerprint does not match")
    return path, data


def _npz(path: Path, names: tuple[str, ...], *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(names):
                raise SceneFusionError(f"{label} has an unexpected array inventory")
            return {name: np.array(loaded[name], copy=True) for name in names}
    except SceneFusionError:
        raise
    except (OSError, ValueError) as exc:
        raise SceneFusionError(f"cannot decode {label}") from exc


def _encoded_layer(values: np.ndarray, *, value_min: float, value_max: float) -> dict[str, Any]:
    grid = np.ascontiguousarray(values, dtype=np.float32)
    return {
        "grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"),
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "value_min": float(value_min),
        "value_max": float(value_max),
    }


def _encoded_rgb_layer(rgb: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    colors = np.ascontiguousarray(rgb, dtype=np.uint8)
    mask = np.ascontiguousarray(observed, dtype=np.float32)
    return {
        "rgb_b64": base64.b64encode(colors.tobytes()).decode("ascii"),
        "rgb_shape": [int(colors.shape[0]), int(colors.shape[1]), 3],
        "observed_b64": base64.b64encode(mask.tobytes()).decode("ascii"),
    }


def _finite_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float(default)


def _binary_perimeter(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(values, dtype=bool)
    for row_offset in range(3):
        for column_offset in range(3):
            eroded &= padded[
                row_offset : row_offset + values.shape[0],
                column_offset : column_offset + values.shape[1],
            ]
    return values & ~eroded


def _sobel_gradient(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    source = np.where(
        np.asarray(observed, dtype=bool) & np.isfinite(values),
        np.asarray(values, dtype=np.float32),
        0.0,
    )
    padded = np.pad(source, 1, mode="edge")
    gx = (
        padded[:-2, 2:]
        + (2.0 * padded[1:-1, 2:])
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - (2.0 * padded[1:-1, :-2])
        - padded[2:, :-2]
    )
    gz = (
        padded[2:, :-2]
        + (2.0 * padded[2:, 1:-1])
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - (2.0 * padded[:-2, 1:-1])
        - padded[:-2, 2:]
    )
    magnitude = np.sqrt((gx * gx) + (gz * gz)).astype(np.float32)
    scale = _finite_percentile(magnitude, 95.0)
    if scale <= 1e-6:
        return np.zeros_like(source, dtype=np.float32)
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def _derive_diagnostic_grids(
    *,
    local_points: np.ndarray,
    points: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Derive Heatmap-style review layers from one admitted common-frame result.

    This is deterministic raster post-processing. It does not invoke MapAnything,
    estimate a new camera transform, or modify the immutable fusion artifact.
    """

    source_grid = manifest["grid"]
    rows = int(source_grid["rows"])
    columns = int(source_grid["columns"])
    bounds = source_grid["bounds"]
    min_x = float(bounds["min_x"])
    max_x = float(bounds["max_x"])
    min_z = float(bounds["min_z"])
    max_z = float(bounds["max_z"])
    dx = (max_x - min_x) / columns
    dz = (max_z - min_z) / rows
    cell_count = rows * columns

    local = np.asarray(local_points, dtype=np.float32)
    colors = np.asarray(points["colors_rgb_u8"], dtype=np.uint8)
    confidence = np.clip(
        np.asarray(points["confidence"], dtype=np.float32), 0.0, 1.0
    )
    point_columns = np.floor((local[:, 0] - min_x) / dx).astype(np.int64)
    point_rows = np.floor((max_z - local[:, 2]) / dz).astype(np.int64)
    valid = (
        np.isfinite(local).all(axis=1)
        & np.isfinite(confidence)
        & (point_columns >= 0)
        & (point_columns < columns)
        & (point_rows >= 0)
        & (point_rows < rows)
    )
    point_columns = point_columns[valid]
    point_rows = point_rows[valid]
    heights = local[valid, 1]
    camera_distance = np.hypot(local[valid, 0], local[valid, 2]).astype(
        np.float32
    )
    colors = colors[valid]
    confidence = confidence[valid]
    weights = np.maximum(confidence, 1e-3)
    linear = (point_rows * columns) + point_columns

    weight_sum = np.bincount(linear, weights=weights, minlength=cell_count)
    height_sum = np.bincount(
        linear, weights=heights * weights, minlength=cell_count
    )
    distance_sum = np.bincount(
        linear, weights=camera_distance * weights, minlength=cell_count
    )
    mean_height = np.full(cell_count, np.nan, dtype=np.float32)
    mean_distance = np.full(cell_count, np.nan, dtype=np.float32)
    np.divide(height_sum, weight_sum, out=mean_height, where=weight_sum > 1e-9)
    np.divide(distance_sum, weight_sum, out=mean_distance, where=weight_sum > 1e-9)
    mean_height = mean_height.reshape((rows, columns))
    mean_height = np.where(
        np.isfinite(mean_height), np.clip(mean_height, 0.0, 4.0), np.nan
    ).astype(np.float32)
    mean_distance = mean_distance.reshape((rows, columns))

    observed = np.asarray(grid["observed"], dtype=bool)
    density_count = np.asarray(grid["point_count"], dtype=np.float32)
    density_scale = float(np.max(density_count)) if density_count.size else 0.0
    density = (
        np.clip(density_count / density_scale, 0.0, 1.0)
        if density_scale > 0.0
        else np.zeros_like(density_count, dtype=np.float32)
    ).astype(np.float32)

    gradient = _sobel_gradient(mean_height, observed)
    walkable = (
        (np.asarray(grid["floor_supported"]) > 0)
        & ~(np.asarray(grid["obstacle_mask"]) > 0)
    ).astype(np.float32)

    surface_bin_count = int(
        math.ceil(_DIAGNOSTIC_SURFACE_HEIGHT_MAX_M / _DIAGNOSTIC_HEIGHT_BIN_M)
    )
    surface_histogram = np.zeros(
        (cell_count, surface_bin_count), dtype=np.float32
    )
    surface_candidates = (
        (heights >= 0.0) & (heights <= _DIAGNOSTIC_SURFACE_HEIGHT_MAX_M)
    )
    if np.any(surface_candidates):
        height_bins = np.clip(
            np.floor(
                heights[surface_candidates] / _DIAGNOSTIC_HEIGHT_BIN_M
            ).astype(np.int64),
            0,
            surface_bin_count - 1,
        )
        np.add.at(
            surface_histogram,
            (linear[surface_candidates], height_bins),
            weights[surface_candidates],
        )
    smoothed_histogram = surface_histogram.copy()
    smoothed_histogram[:, 1:] += surface_histogram[:, :-1] * 0.25
    smoothed_histogram[:, :-1] += surface_histogram[:, 1:] * 0.25
    modal_bin = np.argmax(smoothed_histogram, axis=1)
    modal_support = smoothed_histogram[np.arange(cell_count), modal_bin]
    surface_observed = modal_support > 0.0
    modal_height = (
        (modal_bin.astype(np.float32) + 0.5) * _DIAGNOSTIC_HEIGHT_BIN_M
    )
    structural_height = np.full(cell_count, np.nan, dtype=np.float32)
    structural_height[observed.reshape(-1)] = 0.0
    furniture = (
        surface_observed
        & (modal_height >= _DIAGNOSTIC_FURNITURE_MIN_M)
        & (modal_height <= _DIAGNOSTIC_FURNITURE_MAX_M)
    )
    structural_height[furniture] = modal_height[furniture]

    surface_rgb_sum = np.zeros((cell_count, 3), dtype=np.float64)
    surface_rgb_weight = np.zeros(cell_count, dtype=np.float64)
    point_modal_height = modal_height[np.clip(linear, 0, cell_count - 1)]
    rgb_valid = (
        surface_candidates
        & surface_observed[np.clip(linear, 0, cell_count - 1)]
        & (
            np.abs(heights - point_modal_height)
            <= (_DIAGNOSTIC_HEIGHT_BIN_M * 1.5)
        )
    )
    if np.any(rgb_valid):
        selected_cells = linear[rgb_valid]
        selected_weights = weights[rgb_valid].astype(np.float64)
        np.add.at(surface_rgb_weight, selected_cells, selected_weights)
        for channel in range(3):
            np.add.at(
                surface_rgb_sum[:, channel],
                selected_cells,
                colors[rgb_valid, channel].astype(np.float64) * selected_weights,
            )
    surface_rgb = np.zeros((cell_count, 3), dtype=np.uint8)
    np.divide(
        surface_rgb_sum,
        surface_rgb_weight[:, None],
        out=surface_rgb_sum,
        where=surface_rgb_weight[:, None] > 1e-9,
    )
    surface_rgb[:] = np.clip(np.rint(surface_rgb_sum), 0, 255).astype(np.uint8)
    surface_rgb_observed = surface_rgb_weight > 1e-9

    vertical_bin_m = 0.10
    vertical_bin_count = int(math.ceil(2.40 / vertical_bin_m))
    vertical_histogram = np.zeros(
        (cell_count, vertical_bin_count), dtype=np.uint16
    )
    vertical_candidates = (heights >= 0.0) & (heights <= 2.40)
    if np.any(vertical_candidates):
        vertical_bins = np.clip(
            np.floor(heights[vertical_candidates] / vertical_bin_m).astype(
                np.int64
            ),
            0,
            vertical_bin_count - 1,
        )
        np.add.at(
            vertical_histogram,
            (linear[vertical_candidates], vertical_bins),
            1,
        )
    occupied_bins = vertical_histogram > 0
    occupied_count = np.count_nonzero(occupied_bins, axis=1)
    first_bin = np.argmax(occupied_bins, axis=1)
    last_bin = vertical_bin_count - 1 - np.argmax(
        occupied_bins[:, ::-1], axis=1
    )
    vertical_span = np.where(
        occupied_count > 0,
        (last_bin - first_bin).astype(np.float32) * vertical_bin_m,
        0.0,
    )
    wall_support = (
        np.clip((occupied_count.astype(np.float32) - 2.0) / 8.0, 0.0, 1.0)
        * np.clip((vertical_span - 0.30) / 1.20, 0.0, 1.0)
    )
    wall_support[occupied_count < 4] = 0.0
    wall_support = wall_support.reshape((rows, columns)).astype(np.float32)
    footprint = observed.astype(np.float32)
    perimeter = _binary_perimeter(observed).astype(np.float32)
    room_boundary = np.maximum(wall_support, perimeter * 0.35).astype(np.float32)

    return {
        "density": density,
        "height": mean_height,
        "height_agl": mean_height.copy(),
        "distance": mean_distance,
        "gradient": gradient,
        "obstacle_height": np.asarray(
            grid["obstacle_height_m"], dtype=np.float32
        ).copy(),
        "walkable": walkable,
        "structural_height": structural_height.reshape((rows, columns)),
        "surface_observed": surface_observed.reshape((rows, columns)).astype(
            np.float32
        ),
        "room_footprint": footprint,
        "wall_support": wall_support,
        "room_boundary": room_boundary,
        "surface_rgb": surface_rgb.reshape((rows, columns, 3)),
        "surface_rgb_observed": surface_rgb_observed.reshape(
            (rows, columns)
        ).astype(np.float32),
    }


def _target_shape(payload: Mapping[str, Any], fallback: tuple[int, int]) -> tuple[int, int]:
    for name in ("height_agl", "scene_composite_height_agl", "scene_static_height_agl"):
        layer = payload.get(name)
        shape = layer.get("grid_shape") if isinstance(layer, Mapping) else None
        if isinstance(shape, list) and len(shape) == 2:
            rows, columns = int(shape[0]), int(shape[1])
            if rows > 0 and columns > 0:
                return rows, columns
    return fallback


def _resample_grid(
    values: np.ndarray,
    *,
    source_bounds: Mapping[str, float],
    target_bounds: Mapping[str, float],
    target_shape: tuple[int, int],
    fill: float,
) -> np.ndarray:
    source = np.asarray(values)
    source_rows, source_columns = source.shape
    target_rows, target_columns = target_shape
    target_x = float(target_bounds["min_x"]) + (
        np.arange(target_columns, dtype=np.float64) + 0.5
    ) * (
        (float(target_bounds["max_x"]) - float(target_bounds["min_x"]))
        / target_columns
    )
    target_z = float(target_bounds["max_z"]) - (
        np.arange(target_rows, dtype=np.float64) + 0.5
    ) * (
        (float(target_bounds["max_z"]) - float(target_bounds["min_z"]))
        / target_rows
    )
    columns = np.floor(
        (target_x - float(source_bounds["min_x"]))
        / ((float(source_bounds["max_x"]) - float(source_bounds["min_x"])) / source_columns)
    ).astype(np.int64)
    rows = np.floor(
        (float(source_bounds["max_z"]) - target_z)
        / ((float(source_bounds["max_z"]) - float(source_bounds["min_z"])) / source_rows)
    ).astype(np.int64)
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    inside = (
        (row_grid >= 0)
        & (row_grid < source_rows)
        & (column_grid >= 0)
        & (column_grid < source_columns)
    )
    safe_rows = np.clip(row_grid, 0, source_rows - 1)
    safe_columns = np.clip(column_grid, 0, source_columns - 1)
    return np.where(inside, source[safe_rows, safe_columns], fill)


@dataclass(frozen=True)
class LoadedSceneFusion:
    manifest: Mapping[str, Any]
    points: Mapping[str, np.ndarray]
    grid: Mapping[str, np.ndarray]
    local_points: np.ndarray
    diagnostics: Mapping[str, np.ndarray]

    def compose_floorplan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("frame") != "camera_local_ground_m" or payload.get("units") != "meters":
            raise SceneFusionError("scene fusion requires camera_local_ground_m meters")
        target_bounds = payload.get("bounds")
        source_grid = self.manifest["grid"]
        source_bounds = source_grid["bounds"]
        if not isinstance(target_bounds, Mapping):
            raise SceneFusionError("scene fusion requires floorplan bounds")
        source_shape = (int(source_grid["rows"]), int(source_grid["columns"]))
        target_shape = _target_shape(payload, source_shape)

        resampled: dict[str, np.ndarray] = {}
        for name in _GRID_ARRAYS:
            fill = math.nan if name in ("height_agl_p95_m", "obstacle_height_m") else 0.0
            resampled[name] = _resample_grid(
                self.grid[name],
                source_bounds=source_bounds,
                target_bounds=target_bounds,
                target_shape=target_shape,
                fill=fill,
            )
        finite_height = resampled["height_agl_p95_m"][
            np.isfinite(resampled["height_agl_p95_m"])
        ]
        finite_obstacle = resampled["obstacle_height_m"][
            np.isfinite(resampled["obstacle_height_m"])
        ]
        result = dict(payload)
        result.update(
            {
                "scene_fusion_height_agl": _encoded_layer(
                    resampled["height_agl_p95_m"],
                    value_min=0.0,
                    value_max=float(np.percentile(finite_height, 99.0)) if finite_height.size else 0.0,
                ),
                "scene_fusion_observed": _encoded_layer(
                    resampled["observed"], value_min=0.0, value_max=1.0
                ),
                "scene_fusion_confidence": _encoded_layer(
                    resampled["confidence"], value_min=0.0, value_max=1.0
                ),
                "scene_fusion_obstacle_height": _encoded_layer(
                    resampled["obstacle_height_m"],
                    value_min=0.0,
                    value_max=float(np.percentile(finite_obstacle, 99.0))
                    if finite_obstacle.size
                    else 0.0,
                ),
                "scene_fusion_obstacle_mask": _encoded_layer(
                    resampled["obstacle_mask"], value_min=0.0, value_max=1.0
                ),
                "scene_fusion_floor_supported": _encoded_layer(
                    resampled["floor_supported"], value_min=0.0, value_max=1.0
                ),
                "scene_fusion_floor_height": _encoded_layer(
                    np.where(
                        resampled["floor_supported"] > 0,
                        0.0,
                        np.nan,
                    ).astype(np.float32),
                    value_min=0.0,
                    value_max=0.0,
                ),
                "scene_fusion_provenance": _encoded_layer(
                    resampled["provenance"], value_min=0.0, value_max=3.0
                ),
                "scene_fusion_diagnostic_density": _encoded_layer(
                    self.diagnostics["density"], value_min=0.0, value_max=1.0
                ),
                "scene_fusion_diagnostic_height": _encoded_layer(
                    self.diagnostics["height"],
                    value_min=_finite_percentile(self.diagnostics["height"], 1.0),
                    value_max=_finite_percentile(self.diagnostics["height"], 99.0),
                ),
                "scene_fusion_diagnostic_height_agl": _encoded_layer(
                    self.diagnostics["height_agl"],
                    value_min=0.0,
                    value_max=_finite_percentile(
                        self.diagnostics["height_agl"], 99.0
                    ),
                ),
                "scene_fusion_diagnostic_distance": _encoded_layer(
                    self.diagnostics["distance"],
                    value_min=_finite_percentile(
                        self.diagnostics["distance"], 1.0
                    ),
                    value_max=_finite_percentile(
                        self.diagnostics["distance"], 99.0
                    ),
                ),
                "scene_fusion_diagnostic_gradient": _encoded_layer(
                    self.diagnostics["gradient"], value_min=0.0, value_max=1.0
                ),
                "scene_fusion_diagnostic_obstacle_height": _encoded_layer(
                    self.diagnostics["obstacle_height"],
                    value_min=0.0,
                    value_max=_finite_percentile(
                        self.diagnostics["obstacle_height"], 99.0
                    ),
                ),
                "scene_fusion_diagnostic_walkable": _encoded_layer(
                    self.diagnostics["walkable"], value_min=0.0, value_max=1.0
                ),
                "scene_fusion_diagnostic_structural_height": _encoded_layer(
                    self.diagnostics["structural_height"],
                    value_min=0.0,
                    value_max=_DIAGNOSTIC_FURNITURE_MAX_M,
                ),
                "scene_fusion_diagnostic_surface_observed": _encoded_layer(
                    self.diagnostics["surface_observed"],
                    value_min=0.0,
                    value_max=1.0,
                ),
                "scene_fusion_diagnostic_room_footprint": _encoded_layer(
                    self.diagnostics["room_footprint"],
                    value_min=0.0,
                    value_max=1.0,
                ),
                "scene_fusion_diagnostic_wall_support": _encoded_layer(
                    self.diagnostics["wall_support"],
                    value_min=0.0,
                    value_max=1.0,
                ),
                "scene_fusion_diagnostic_room_boundary": _encoded_layer(
                    self.diagnostics["room_boundary"],
                    value_min=0.0,
                    value_max=1.0,
                ),
                "scene_fusion_diagnostic_surface_rgb": _encoded_rgb_layer(
                    self.diagnostics["surface_rgb"],
                    self.diagnostics["surface_rgb_observed"],
                ),
                "scene_fusion_points": {
                    "frame": "camera_local_ground_m",
                    "units": "meters",
                    "point_count": int(self.local_points.shape[0]),
                    "positions_f32_b64": base64.b64encode(
                        np.ascontiguousarray(self.local_points, dtype=np.float32).tobytes()
                    ).decode("ascii"),
                    "colors_rgb_u8_b64": base64.b64encode(
                        np.ascontiguousarray(self.points["colors_rgb_u8"], dtype=np.uint8).tobytes()
                    ).decode("ascii"),
                    "confidence_f32_b64": base64.b64encode(
                        np.ascontiguousarray(self.points["confidence"], dtype=np.float32).tobytes()
                    ).decode("ascii"),
                    "provenance_u8_b64": base64.b64encode(
                        np.ascontiguousarray(self.points["provenance"], dtype=np.uint8).tobytes()
                    ).decode("ascii"),
                },
                "scene_fusion_meta": {
                    "contract": self.manifest["contract"],
                    "contract_version": self.manifest["contract_version"],
                    "fusion_id": self.manifest["fusion_id"],
                    "status": self.manifest["status"],
                    "diagnostic_only": bool(self.manifest.get("diagnostic_only", True)),
                    "space_id": self.manifest["space_id"],
                    "camera_id": self.manifest["camera_id"],
                    "inference": self.manifest["method"]["inference"],
                    "quality": self.manifest["quality"],
                    "source_counts": self.manifest["point_cloud"]["source_counts"],
                    "raster_orientation": source_grid["orientation"],
                    "diagnostic_layers": {
                        "contract": "noesis.scene_fusion.diagnostic_layers",
                        "contract_version": 1,
                        "source": "admitted_common_frame_fused_points_and_grid",
                        "derived_at_catalog_load": True,
                        "mapanything_inference_triggered": False,
                        "registration_triggered": False,
                        "bounds": dict(source_bounds),
                        "grid_shape": [source_shape[0], source_shape[1]],
                        "raster_orientation": source_grid["orientation"],
                        "fixed_anchor_view_count": int(
                            self.manifest["method"].get("fixed_anchor_view_count", 0)
                        ),
                        "phone_view_count": int(
                            self.manifest.get("sources", {}).get("phone_frame_count", 0)
                        ),
                        "predicted_anchor_geometry_used_in_fusion": bool(
                            self.manifest["method"].get(
                                "predicted_anchor_geometry_used_in_fusion", False
                            )
                        ),
                    },
                },
            }
        )
        return result


class SceneFusionSet:
    """Configured immutable diagnostic fusions indexed by reference camera."""

    def __init__(self, site_id: str, fusions: Mapping[str, LoadedSceneFusion]) -> None:
        self.site_id = str(site_id)
        self._fusions = dict(fusions)

    @classmethod
    def load(cls, catalog_path: str | Path) -> "SceneFusionSet":
        path = Path(catalog_path)
        catalog = _json_object(
            _read_bytes(path, maximum=MAX_CATALOG_BYTES, label="scene-fusion catalog"),
            label="scene-fusion catalog",
        )
        if catalog.get("contract") != "noesis.scene_fusion.catalog" or catalog.get(
            "contract_version"
        ) != 1:
            raise SceneFusionError("scene-fusion catalog contract is unsupported")
        bindings = catalog.get("camera_bindings")
        if not isinstance(bindings, list):
            raise SceneFusionError("scene-fusion catalog camera_bindings must be a list")
        root = path.parent
        fusions: dict[str, LoadedSceneFusion] = {}
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise SceneFusionError("scene-fusion binding must be an object")
            camera_id = str(binding.get("camera_id") or "")
            if not camera_id or camera_id in fusions:
                raise SceneFusionError("scene-fusion camera bindings must be unique")
            _, manifest_bytes = _verified_artifact(
                root, binding["manifest"], maximum=MAX_MANIFEST_BYTES, label=f"{camera_id} fusion manifest"
            )
            manifest = _json_object(manifest_bytes, label=f"{camera_id} fusion manifest")
            if (
                manifest.get("contract") != "noesis.scene_fusion.diagnostic"
                or manifest.get("contract_version") != 1
                or manifest.get("status") != "passed"
                or manifest.get("camera_id") != camera_id
                or manifest.get("site_id") != catalog.get("site_id")
            ):
                raise SceneFusionError(f"{camera_id} fusion manifest is not an admitted diagnostic")
            points_path, _ = _verified_artifact(
                root, binding["points"], maximum=MAX_POINTS_BYTES, label=f"{camera_id} fusion points"
            )
            grid_path, _ = _verified_artifact(
                root, binding["grid"], maximum=MAX_GRID_BYTES, label=f"{camera_id} fusion grid"
            )
            points = _npz(points_path, _POINT_ARRAYS, label=f"{camera_id} fusion points")
            grid = _npz(grid_path, _GRID_ARRAYS, label=f"{camera_id} fusion grid")
            point_count = int(manifest["point_cloud"]["point_count"])
            if (
                points["points_world_m"].shape != (point_count, 3)
                or points["colors_rgb_u8"].shape != (point_count, 3)
                or any(points[name].shape != (point_count,) for name in ("confidence", "provenance", "view_support"))
                or not np.isfinite(points["points_world_m"]).all()
                or not np.isfinite(points["confidence"]).all()
            ):
                raise SceneFusionError(f"{camera_id} fusion point arrays are invalid")
            expected_grid_shape = (int(manifest["grid"]["rows"]), int(manifest["grid"]["columns"]))
            if any(grid[name].shape != expected_grid_shape for name in _GRID_ARRAYS):
                raise SceneFusionError(f"{camera_id} fusion grid arrays have invalid shapes")
            pose = np.asarray(manifest.get("reference_camera_to_world_row_major"), dtype=np.float64)
            if pose.size != 16 or not np.isfinite(pose).all():
                raise SceneFusionError(f"{camera_id} fusion reference camera pose is invalid")
            pose = pose.reshape((4, 4))
            try:
                camera_frame = camera_ground_frame_from_camera_to_world(pose)
                world_to_camera_local = (
                    camera_frame.world_to_camera_local_display_matrix(
                        float(manifest["floor_y_m"])
                    )
                )
                local_points = transform_positions(
                    points["points_world_m"],
                    world_to_camera_local,
                ).astype(np.float32)
            except (CoordinateFrameError, TypeError, ValueError) as exc:
                raise SceneFusionError(
                    f"{camera_id} fusion reference camera pose is invalid: {exc}"
                ) from exc
            diagnostics = _derive_diagnostic_grids(
                local_points=local_points,
                points=points,
                grid=grid,
                manifest=manifest,
            )
            fusions[camera_id] = LoadedSceneFusion(
                manifest=manifest,
                points=points,
                grid=grid,
                local_points=local_points,
                diagnostics=diagnostics,
            )
        return cls(str(catalog.get("site_id") or ""), fusions)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._fusions))

    def compose_floorplan(self, camera_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        fusion = self._fusions.get(str(camera_id))
        return dict(payload) if fusion is None else fusion.compose_floorplan(payload)

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "contract": "noesis.scene_fusion.health",
            "contract_version": 1,
            "site_id": self.site_id,
            "status": "loaded",
            "camera_count": len(self._fusions),
            "cameras": {
                camera_id: {
                    "fusion_id": fusion.manifest["fusion_id"],
                    "status": fusion.manifest["status"],
                    "diagnostic_only": True,
                }
                for camera_id, fusion in sorted(self._fusions.items())
            },
        }


__all__ = ["LoadedSceneFusion", "SceneFusionError", "SceneFusionSet"]
