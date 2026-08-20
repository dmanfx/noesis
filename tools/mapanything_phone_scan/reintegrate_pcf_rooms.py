#!/usr/bin/env python3
"""Reintegrate two registered PCF room walks without losing provenance.

This module consumes the accepted per-view PCF ``raw/view_*.npz`` artifacts.
It does not estimate registration and it never runs ICP.  A caller supplies the
accepted local-to-world transforms plus any cross-session/global and per-view
world-space corrections.  All accepted pixels are streamed into 2.5 cm
weighted voxels, with source-room, source-view, observation, consensus-
agreement, and model-reliability provenance retained in the output NPZ.

The fixed room owns overlapping evidence.  By default, overlap is the moving
room evidence whose X/Z column is covered by at least two fixed-room views
(with a small configurable dilation); exact shared voxels always belong to the
fixed room.  A reviewed fixed-room ownership polygon may be supplied when the
door/opening boundary is known more precisely.  Moving-room-only evidence is
retained, so the second room continues beyond the shared opening.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class PCFReintegrationError(RuntimeError):
    """Raised when PCF evidence cannot be reintegrated without ambiguity."""


@dataclass(frozen=True)
class ReintegrationSettings:
    """Conservative evidence and ownership settings for two-room fusion."""

    voxel_size_m: float = 0.025
    minimum_confidence: float = 0.35
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 12.0
    agreement_weight_bonus: float = 0.25
    ownership_cell_m: float = 0.10
    ownership_dilation_m: float = 0.10
    fixed_owner_minimum_views: int = 2
    merge_batch_size: int = 8
    hash_raw_views: bool = True
    write_glb: bool = True
    write_topdown: bool = True


@dataclass(frozen=True)
class RoomSource:
    """One accepted PCF room and its explicitly composed world corrections.

    Point transforms are composed in this order::

        corrected_world = per_view_world_correction
                          @ global_world_correction
                          @ accepted_world_from_local
                          @ local_point

    Both correction kinds therefore operate in the fixed/output world frame.
    ``per_view_world_corrections`` must contain only refinements that were
    validated by the registration stage; missing views receive identity.
    """

    name: str
    raw_root: Path
    world_manifest: Path
    prior_id: str | None = None
    scan_id: str | None = None
    global_world_correction: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    per_view_world_corrections: Mapping[int, np.ndarray] = field(
        default_factory=dict
    )
    included_view_indices: frozenset[int] | None = None


@dataclass
class RoomFusion:
    source: RoomSource
    records: np.ndarray
    camera_positions: np.ndarray
    view_files: list[dict[str, Any]]
    counters: dict[str, int]
    view_count: int
    view_mask_words: int


_REQUIRED_RAW_FIELDS = {
    "world_points",
    "depth_z",
    "confidence",
    "mask",
    "camera_pose",
    "model_rgb",
    "source_selection",
    "cross_model_uncertain",
    "cross_model_agreement",
    "absolute_depth_disagreement_m",
    "mapanything_reliability",
    "da3_reliability",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.5 else 1.0
    return np.clip(array * scale, 0.0, 255.0).astype(np.uint8)


def _validate_matrix(
    matrix: np.ndarray,
    *,
    label: str,
    permit_metric_scale: bool,
) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise PCFReintegrationError(f"{label} must be one finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise PCFReintegrationError(f"{label} has an invalid homogeneous row")
    linear = value[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise PCFReintegrationError(f"{label} is reflected or singular")
    scale = determinant ** (1.0 / 3.0)
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise PCFReintegrationError(f"{label} contains shear/non-rigid distortion")
    if permit_metric_scale:
        if not 0.98 <= scale <= 1.02:
            raise PCFReintegrationError(
                f"{label} changes accepted metric scale by {scale:.6f}"
            )
    elif not math.isclose(scale, 1.0, abs_tol=2e-3):
        raise PCFReintegrationError(
            f"{label} must be rigid; observed scale {scale:.6f}"
        )
    return value


def _manifest_transform(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    alignment = payload.get("alignment")
    if not isinstance(alignment, dict):
        raise PCFReintegrationError(f"missing alignment in {path}")
    rotation = np.asarray(alignment.get("rotation_row_major"), dtype=np.float64)
    translation = np.asarray(alignment.get("translation"), dtype=np.float64)
    scale = float(alignment.get("scale", float("nan")))
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise PCFReintegrationError(f"malformed alignment in {path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return (
        _validate_matrix(
            transform,
            label=f"accepted world transform from {path}",
            permit_metric_scale=True,
        ),
        payload,
    )


def _effective_transform(
    source: RoomSource,
    accepted_world_from_local: np.ndarray,
    view_index: int,
) -> np.ndarray:
    global_correction = _validate_matrix(
        source.global_world_correction,
        label=f"{source.name} global world correction",
        permit_metric_scale=False,
    )
    view_correction = _validate_matrix(
        source.per_view_world_corrections.get(view_index, np.eye(4)),
        label=f"{source.name} view {view_index} world correction",
        permit_metric_scale=False,
    )
    return view_correction @ global_correction @ accepted_world_from_local


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ array.T).T + transform[:3, 3]


def _record_dtype(view_mask_words: int) -> np.dtype:
    return np.dtype(
        [
            ("key", "<i4", (3,)),
            ("point_weighted_sum", "<f8", (3,)),
            ("color_weighted_sum", "<f8", (3,)),
            ("weight_sum", "<f8"),
            ("confidence_sum", "<f8"),
            ("observation_count", "<u4"),
            ("agreement_observation_count", "<u4"),
            ("agreement_weight_sum", "<f8"),
            ("disagreement_weighted_sum", "<f8"),
            ("disagreement_weight_sum", "<f8"),
            ("map_reliability_sum", "<f8"),
            ("map_reliability_count", "<u4"),
            ("da3_reliability_sum", "<f8"),
            ("da3_reliability_count", "<u4"),
            ("source_selection_counts", "<u4", (6,)),
            ("view_mask_words", "<u8", (view_mask_words,)),
        ]
    )


def _sort_order(keys: np.ndarray) -> np.ndarray:
    return np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    if not len(sorted_keys):
        return np.empty(0, dtype=np.int64)
    changed = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    return np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(changed) + 1)
    )


def _reduce_records(rows: Sequence[np.ndarray], dtype: np.dtype) -> np.ndarray:
    available = [row for row in rows if len(row)]
    if not available:
        return np.empty(0, dtype=dtype)
    combined = np.concatenate(available)
    order = _sort_order(combined["key"])
    combined = combined[order]
    starts = _group_starts(combined["key"])
    result = np.zeros(len(starts), dtype=dtype)
    result["key"] = combined["key"][starts]
    additive = (
        "point_weighted_sum",
        "color_weighted_sum",
        "weight_sum",
        "confidence_sum",
        "observation_count",
        "agreement_observation_count",
        "agreement_weight_sum",
        "disagreement_weighted_sum",
        "disagreement_weight_sum",
        "map_reliability_sum",
        "map_reliability_count",
        "da3_reliability_sum",
        "da3_reliability_count",
        "source_selection_counts",
    )
    for field_name in additive:
        result[field_name] = np.add.reduceat(combined[field_name], starts, axis=0)
    result["view_mask_words"] = np.bitwise_or.reduceat(
        combined["view_mask_words"], starts, axis=0
    )
    return result


def _pad_view_mask_words(records: np.ndarray, target_words: int) -> np.ndarray:
    current_words = records.dtype["view_mask_words"].shape[0]
    if current_words == target_words:
        return records
    if current_words > target_words:
        raise PCFReintegrationError("cannot truncate source-view provenance masks")
    result = np.zeros(len(records), dtype=_record_dtype(target_words))
    for field_name in records.dtype.names or ():
        if field_name == "view_mask_words":
            result[field_name][:, :current_words] = records[field_name]
        else:
            result[field_name] = records[field_name]
    return result


def _aggregate_view(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    agreement: np.ndarray,
    disagreement: np.ndarray,
    map_reliability: np.ndarray,
    da3_reliability: np.ndarray,
    source_selection: np.ndarray,
    view_index: int,
    view_mask_words: int,
    settings: ReintegrationSettings,
) -> np.ndarray:
    dtype = _record_dtype(view_mask_words)
    if not len(points):
        return np.empty(0, dtype=dtype)
    keys = np.floor(points / settings.voxel_size_m).astype(np.int32)
    order = _sort_order(keys)
    keys = keys[order]
    points = points[order].astype(np.float64, copy=False)
    colors = colors[order].astype(np.float64, copy=False)
    confidence = confidence[order].astype(np.float64, copy=False)
    agreement = agreement[order]
    disagreement = disagreement[order].astype(np.float64, copy=False)
    map_reliability = map_reliability[order].astype(np.float64, copy=False)
    da3_reliability = da3_reliability[order].astype(np.float64, copy=False)
    source_selection = source_selection[order]
    starts = _group_starts(keys)
    weights = confidence * (
        1.0 + settings.agreement_weight_bonus * agreement.astype(np.float64)
    )
    result = np.zeros(len(starts), dtype=dtype)
    result["key"] = keys[starts]
    result["point_weighted_sum"] = np.add.reduceat(
        points * weights[:, None], starts, axis=0
    )
    result["color_weighted_sum"] = np.add.reduceat(
        colors * weights[:, None], starts, axis=0
    )
    result["weight_sum"] = np.add.reduceat(weights, starts)
    result["confidence_sum"] = np.add.reduceat(confidence, starts)
    result["observation_count"] = np.diff(
        np.append(starts, len(keys))
    ).astype(np.uint32)
    result["agreement_observation_count"] = np.add.reduceat(
        agreement.astype(np.uint32), starts
    )
    result["agreement_weight_sum"] = np.add.reduceat(
        weights * agreement.astype(np.float64), starts
    )
    valid_disagreement = np.isfinite(disagreement)
    result["disagreement_weighted_sum"] = np.add.reduceat(
        np.where(valid_disagreement, disagreement * weights, 0.0), starts
    )
    result["disagreement_weight_sum"] = np.add.reduceat(
        np.where(valid_disagreement, weights, 0.0), starts
    )
    valid_map = np.isfinite(map_reliability)
    result["map_reliability_sum"] = np.add.reduceat(
        np.where(valid_map, map_reliability, 0.0), starts
    )
    result["map_reliability_count"] = np.add.reduceat(
        valid_map.astype(np.uint32), starts
    )
    valid_da3 = np.isfinite(da3_reliability)
    result["da3_reliability_sum"] = np.add.reduceat(
        np.where(valid_da3, da3_reliability, 0.0), starts
    )
    result["da3_reliability_count"] = np.add.reduceat(
        valid_da3.astype(np.uint32), starts
    )
    one_hot = np.eye(6, dtype=np.uint32)[source_selection]
    result["source_selection_counts"] = np.add.reduceat(one_hot, starts, axis=0)
    word = view_index // 64
    bit = view_index % 64
    result["view_mask_words"][:, word] = np.left_shift(
        np.uint64(1), np.uint64(bit)
    )
    return result


def _compact_runs(
    run_paths: list[Path],
    *,
    dtype: np.dtype,
    batch_size: int,
    temporary_root: Path,
) -> np.ndarray:
    current = list(run_paths)
    generation = 0
    while len(current) > 1:
        next_generation: list[Path] = []
        for offset in range(0, len(current), batch_size):
            batch = current[offset : offset + batch_size]
            rows = [np.load(path, allow_pickle=False) for path in batch]
            merged = _reduce_records(rows, dtype)
            path = temporary_root / f"merge_{generation:02d}_{offset // batch_size:04d}.npy"
            np.save(path, merged, allow_pickle=False)
            next_generation.append(path)
        for path in current:
            path.unlink(missing_ok=True)
        current = next_generation
        generation += 1
    if not current:
        return np.empty(0, dtype=dtype)
    result = np.load(current[0], allow_pickle=False)
    current[0].unlink(missing_ok=True)
    return result


def _popcount_words(words: np.ndarray) -> np.ndarray:
    values = np.asarray(words, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.sum(np.bitwise_count(values), axis=1, dtype=np.uint16)
    bytes_view = values.view(np.uint8).reshape(values.shape[0], -1)
    return np.unpackbits(bytes_view, axis=1).sum(axis=1, dtype=np.uint16)


def _fuse_room(
    source: RoomSource,
    settings: ReintegrationSettings,
    temporary_root: Path,
) -> RoomFusion:
    raw_paths = sorted(source.raw_root.glob("view_*.npz"))
    if len(raw_paths) < 2:
        raise PCFReintegrationError(
            f"{source.name} has too few raw PCF views in {source.raw_root}"
        )
    unexpected = sorted(
        index
        for index in source.per_view_world_corrections
        if index < 0 or index >= len(raw_paths)
    )
    if unexpected:
        raise PCFReintegrationError(
            f"{source.name} corrections reference absent views: {unexpected}"
        )
    if source.included_view_indices is not None:
        unexpected_included = sorted(
            index
            for index in source.included_view_indices
            if index < 0 or index >= len(raw_paths)
        )
        if unexpected_included:
            raise PCFReintegrationError(
                f"{source.name} view selection references absent views: "
                f"{unexpected_included}"
            )
        if len(source.included_view_indices) < 2:
            raise PCFReintegrationError(
                f"{source.name} view selection must retain at least two views"
            )
    accepted_world_from_local, _ = _manifest_transform(source.world_manifest)
    view_mask_words = int(math.ceil(len(raw_paths) / 64.0))
    dtype = _record_dtype(view_mask_words)
    run_paths: list[Path] = []
    camera_positions: list[np.ndarray] = []
    view_files: list[dict[str, Any]] = []
    counters = {
        "raw_pixel_count": 0,
        "raw_mask_accepted_count": 0,
        "uncertain_rejected_count": 0,
        "confidence_rejected_count": 0,
        "depth_rejected_count": 0,
        "nonfinite_rejected_count": 0,
        "selected_observation_count": 0,
    }
    for view_index, path in enumerate(raw_paths):
        if (
            source.included_view_indices is not None
            and view_index not in source.included_view_indices
        ):
            continue
        sha256 = _sha256(path) if settings.hash_raw_views else None
        with np.load(path, allow_pickle=False) as row:
            missing = sorted(_REQUIRED_RAW_FIELDS.difference(row.files))
            if missing:
                raise PCFReintegrationError(f"{path} is missing {missing}")
            world_points = np.asarray(row["world_points"], dtype=np.float64)
            depth = np.asarray(row["depth_z"], dtype=np.float32)
            confidence = np.asarray(row["confidence"], dtype=np.float32)
            mask = np.asarray(row["mask"], dtype=bool)
            rgb = _as_u8(row["model_rgb"])
            uncertain = np.asarray(row["cross_model_uncertain"], dtype=bool)
            agreement = np.asarray(row["cross_model_agreement"], dtype=bool)
            disagreement = np.asarray(
                row["absolute_depth_disagreement_m"], dtype=np.float32
            )
            map_reliability = np.asarray(
                row["mapanything_reliability"], dtype=np.float32
            )
            da3_reliability = np.asarray(row["da3_reliability"], dtype=np.float32)
            source_selection = np.asarray(row["source_selection"], dtype=np.uint8)
            camera_pose = np.asarray(row["camera_pose"], dtype=np.float64)
        expected = depth.shape
        pixel_fields = (
            confidence,
            mask,
            uncertain,
            agreement,
            disagreement,
            map_reliability,
            da3_reliability,
            source_selection,
        )
        if world_points.shape != (*expected, 3) or rgb.shape != (*expected, 3):
            raise PCFReintegrationError(f"RGB-D geometry shape mismatch in {path}")
        if any(value.shape != expected for value in pixel_fields):
            raise PCFReintegrationError(f"provenance field shape mismatch in {path}")
        if camera_pose.shape != (4, 4):
            raise PCFReintegrationError(f"camera pose is malformed in {path}")
        if np.any(source_selection > 5):
            raise PCFReintegrationError(f"unknown source-selection code in {path}")

        finite = np.isfinite(world_points).all(axis=2) & np.isfinite(depth)
        depth_valid = (depth >= settings.minimum_depth_m) & (
            depth <= settings.maximum_depth_m
        )
        confidence_valid = np.isfinite(confidence) & (
            confidence >= settings.minimum_confidence
        )
        selected = (
            mask
            & ~uncertain
            & finite
            & depth_valid
            & confidence_valid
            & (source_selection > 0)
        )
        counters["raw_pixel_count"] += int(mask.size)
        counters["raw_mask_accepted_count"] += int(np.count_nonzero(mask))
        counters["uncertain_rejected_count"] += int(np.count_nonzero(mask & uncertain))
        counters["confidence_rejected_count"] += int(
            np.count_nonzero(mask & ~uncertain & ~confidence_valid)
        )
        counters["depth_rejected_count"] += int(
            np.count_nonzero(mask & ~uncertain & confidence_valid & ~depth_valid)
        )
        counters["nonfinite_rejected_count"] += int(
            np.count_nonzero(
                mask & ~uncertain & confidence_valid & depth_valid & ~finite
            )
        )
        counters["selected_observation_count"] += int(np.count_nonzero(selected))

        transform = _effective_transform(source, accepted_world_from_local, view_index)
        points = _transform_points(world_points[selected], transform)
        finite_transformed = np.isfinite(points).all(axis=1)
        if not np.all(finite_transformed):
            counters["nonfinite_rejected_count"] += int(
                np.count_nonzero(~finite_transformed)
            )
            points = points[finite_transformed]
        selected_flat = np.flatnonzero(selected.reshape(-1))[finite_transformed]
        aggregated = _aggregate_view(
            points=points,
            colors=rgb.reshape(-1, 3)[selected_flat],
            confidence=confidence.reshape(-1)[selected_flat],
            agreement=agreement.reshape(-1)[selected_flat],
            disagreement=disagreement.reshape(-1)[selected_flat],
            map_reliability=map_reliability.reshape(-1)[selected_flat],
            da3_reliability=da3_reliability.reshape(-1)[selected_flat],
            source_selection=source_selection.reshape(-1)[selected_flat],
            view_index=view_index,
            view_mask_words=view_mask_words,
            settings=settings,
        )
        run_path = temporary_root / f"{source.name}_{view_index:04d}.npy"
        np.save(run_path, aggregated, allow_pickle=False)
        run_paths.append(run_path)
        camera_positions.append(_transform_points(camera_pose[None, :3, 3], transform)[0])
        view_files.append(
            {
                "view_index": view_index,
                "path": str(path),
                "sha256": sha256,
                "selected_observation_count": int(np.count_nonzero(selected)),
                "effective_world_from_local_row_major": transform.tolist(),
            }
        )
    records = _compact_runs(
        run_paths,
        dtype=dtype,
        batch_size=settings.merge_batch_size,
        temporary_root=temporary_root,
    )
    if not len(records):
        raise PCFReintegrationError(f"{source.name} produced no accepted voxels")
    return RoomFusion(
        source=source,
        records=records,
        camera_positions=np.asarray(camera_positions, dtype=np.float64),
        view_files=view_files,
        counters=counters,
        view_count=(
            len(raw_paths)
            if source.included_view_indices is None
            else len(source.included_view_indices)
        ),
        view_mask_words=view_mask_words,
    )


def _points_from_records(records: np.ndarray) -> np.ndarray:
    return (
        records["point_weighted_sum"]
        / np.maximum(records["weight_sum"][:, None], 1e-12)
    )


def _keys_as_structured(keys: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(keys, dtype=np.int32)
    return value.view(
        np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
    ).reshape(-1)


def _polygon_contains(points_xz: np.ndarray, polygon_xz: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_xz, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise PCFReintegrationError("fixed ownership polygon needs at least 3 X/Z points")
    x = points_xz[:, 0]
    z = points_xz[:, 1]
    inside = np.zeros(len(points_xz), dtype=bool)
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        x1, z1 = polygon[current]
        x2, z2 = polygon[previous]
        crosses = (z1 > z) != (z2 > z)
        boundary_x = (x2 - x1) * (z - z1) / (z2 - z1 + 1e-15) + x1
        inside ^= crosses & (x < boundary_x)
        previous = current
    return inside


def _fixed_owner_columns(
    fixed: RoomFusion,
    settings: ReintegrationSettings,
) -> set[tuple[int, int]]:
    points = _points_from_records(fixed.records)
    columns = np.floor(points[:, [0, 2]] / settings.ownership_cell_m).astype(np.int32)
    order = np.lexsort((columns[:, 1], columns[:, 0]))
    columns = columns[order]
    view_masks = fixed.records["view_mask_words"][order]
    starts = _group_starts(columns)
    unique_columns = columns[starts]
    column_masks = np.bitwise_or.reduceat(view_masks, starts, axis=0)
    supported = _popcount_words(column_masks) >= settings.fixed_owner_minimum_views
    base = {tuple(value) for value in unique_columns[supported].tolist()}
    radius = int(math.ceil(settings.ownership_dilation_m / settings.ownership_cell_m))
    if radius <= 0:
        return base
    offsets = [
        (dx, dz)
        for dx in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if math.hypot(dx, dz) * settings.ownership_cell_m
        <= settings.ownership_dilation_m + 1e-9
    ]
    return {
        (column[0] + dx, column[1] + dz)
        for column in base
        for dx, dz in offsets
    }


def _match_exact_keys(
    source_keys: np.ndarray,
    query_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = _keys_as_structured(source_keys)
    query = _keys_as_structured(query_keys)
    positions = np.searchsorted(source, query)
    valid = positions < len(source)
    matched = np.zeros(len(query), dtype=bool)
    matched[valid] = source[positions[valid]] == query[valid]
    return matched, positions


def _copy_room_provenance(
    destination: dict[str, np.ndarray],
    prefix: str,
    target_indices: np.ndarray,
    records: np.ndarray,
) -> None:
    destination[f"{prefix}_observation_count"][target_indices] = records[
        "observation_count"
    ]
    destination[f"{prefix}_view_mask_words"][target_indices] = records[
        "view_mask_words"
    ]
    destination[f"{prefix}_source_selection_counts"][target_indices] = records[
        "source_selection_counts"
    ]


def _compose_output(
    fixed: RoomFusion,
    moving: RoomFusion,
    settings: ReintegrationSettings,
    fixed_owner_polygon_xz: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    moving_points = _points_from_records(moving.records)
    moving_columns = np.floor(
        moving_points[:, [0, 2]] / settings.ownership_cell_m
    ).astype(np.int32)
    exact_fixed, exact_positions = _match_exact_keys(
        fixed.records["key"], moving.records["key"]
    )
    if fixed_owner_polygon_xz is None:
        fixed_columns = _fixed_owner_columns(fixed, settings)
        column_owned = np.fromiter(
            (tuple(value) in fixed_columns for value in moving_columns),
            dtype=bool,
            count=len(moving_columns),
        )
        ownership_method = "fixed_multiview_xz_coverage_with_dilation"
    else:
        column_owned = _polygon_contains(moving_points[:, [0, 2]], fixed_owner_polygon_xz)
        ownership_method = "reviewed_fixed_room_xz_polygon"
    suppressed = exact_fixed | column_owned
    retained_moving = moving.records[~suppressed]
    suppressed_moving = moving.records[suppressed]

    combined_records = np.concatenate((fixed.records, retained_moving))
    owner_room_id = np.concatenate(
        (
            np.ones(len(fixed.records), dtype=np.uint8),
            np.full(len(retained_moving), 2, dtype=np.uint8),
        )
    )
    order = _sort_order(combined_records["key"])
    combined_records = combined_records[order]
    owner_room_id = owner_room_id[order]
    if np.any(np.all(combined_records["key"][1:] == combined_records["key"][:-1], axis=1)):
        raise PCFReintegrationError("ownership policy left duplicate output voxels")

    points = _points_from_records(combined_records).astype(np.float32)
    colors = np.clip(
        combined_records["color_weighted_sum"]
        / np.maximum(combined_records["weight_sum"][:, None], 1e-12),
        0.0,
        255.0,
    ).astype(np.uint8)
    observation_count = combined_records["observation_count"].astype(np.uint32)
    view_count = _popcount_words(combined_records["view_mask_words"])
    agreement_observation_count = combined_records[
        "agreement_observation_count"
    ].astype(np.uint32)
    disagreement_mean = np.divide(
        combined_records["disagreement_weighted_sum"],
        combined_records["disagreement_weight_sum"],
        out=np.full(len(combined_records), np.nan, dtype=np.float64),
        where=combined_records["disagreement_weight_sum"] > 0.0,
    ).astype(np.float32)
    map_mean = np.divide(
        combined_records["map_reliability_sum"],
        combined_records["map_reliability_count"],
        out=np.full(len(combined_records), np.nan, dtype=np.float64),
        where=combined_records["map_reliability_count"] > 0,
    ).astype(np.float32)
    da3_mean = np.divide(
        combined_records["da3_reliability_sum"],
        combined_records["da3_reliability_count"],
        out=np.full(len(combined_records), np.nan, dtype=np.float64),
        where=combined_records["da3_reliability_count"] > 0,
    ).astype(np.float32)

    output: dict[str, np.ndarray] = {
        "points": points,
        "colors": colors,
        "voxel_keys": combined_records["key"].astype(np.int32),
        "fusion_weight_sum": combined_records["weight_sum"].astype(np.float32),
        "mean_confidence": (
            combined_records["confidence_sum"] / np.maximum(observation_count, 1)
        ).astype(np.float32),
        "observation_count": observation_count,
        "view_count": view_count.astype(np.uint16),
        "agreement_observation_count": agreement_observation_count,
        "agreement_fraction": (
            agreement_observation_count / np.maximum(observation_count, 1)
        ).astype(np.float32),
        "agreement_weight_sum": combined_records["agreement_weight_sum"].astype(
            np.float32
        ),
        "mean_absolute_depth_disagreement_m": disagreement_mean,
        "mean_mapanything_reliability": map_mean,
        "mean_da3_reliability": da3_mean,
        "source_selection_counts": combined_records[
            "source_selection_counts"
        ].astype(np.uint32),
        "owner_room_id": owner_room_id,
        "room_contribution_mask": np.where(owner_room_id == 1, 1, 2).astype(np.uint8),
        "room_presence_mask": np.where(owner_room_id == 1, 1, 2).astype(np.uint8),
        "fixed_observation_count": np.zeros(len(combined_records), dtype=np.uint32),
        "moving_observation_count": np.zeros(len(combined_records), dtype=np.uint32),
        "fixed_view_mask_words": np.zeros(
            (len(combined_records), fixed.view_mask_words), dtype=np.uint64
        ),
        "moving_view_mask_words": np.zeros(
            (len(combined_records), moving.view_mask_words), dtype=np.uint64
        ),
        "fixed_source_selection_counts": np.zeros(
            (len(combined_records), 6), dtype=np.uint32
        ),
        "moving_source_selection_counts": np.zeros(
            (len(combined_records), 6), dtype=np.uint32
        ),
        "fixed_camera_positions": fixed.camera_positions.astype(np.float32),
        "moving_camera_positions": moving.camera_positions.astype(np.float32),
        "voxel_size_m": np.asarray([settings.voxel_size_m], dtype=np.float32),
    }
    fixed_output = np.flatnonzero(owner_room_id == 1)
    moving_output = np.flatnonzero(owner_room_id == 2)
    _copy_room_provenance(output, "fixed", fixed_output, combined_records[fixed_output])
    _copy_room_provenance(output, "moving", moving_output, combined_records[moving_output])

    # Exact shared voxels retain both rooms' pre-ownership provenance even
    # though only fixed-room geometry contributes to the output centroid.
    suppressed_exact_indices = np.flatnonzero(exact_fixed)
    if len(suppressed_exact_indices):
        fixed_record_indices = exact_positions[suppressed_exact_indices]
        fixed_key_structured = _keys_as_structured(fixed.records["key"])
        output_key_structured = _keys_as_structured(combined_records["key"])
        fixed_output_indices = np.searchsorted(
            output_key_structured, fixed_key_structured[fixed_record_indices]
        )
        _copy_room_provenance(
            output,
            "moving",
            fixed_output_indices,
            moving.records[suppressed_exact_indices],
        )
        output["room_presence_mask"][fixed_output_indices] |= np.uint8(2)

    ownership = {
        "method": ownership_method,
        "fixed_room_owns_overlap": True,
        "fixed_room_id": 1,
        "moving_room_id": 2,
        "fixed_input_voxel_count": int(len(fixed.records)),
        "moving_input_voxel_count": int(len(moving.records)),
        "exact_shared_voxel_count": int(np.count_nonzero(exact_fixed)),
        "moving_suppressed_voxel_count": int(np.count_nonzero(suppressed)),
        "moving_retained_voxel_count": int(len(retained_moving)),
        "moving_suppressed_observation_count": int(
            np.sum(moving.records["observation_count"][suppressed], dtype=np.uint64)
        ),
        "ownership_cell_m": float(settings.ownership_cell_m),
        "ownership_dilation_m": float(settings.ownership_dilation_m),
        "fixed_owner_minimum_views": int(settings.fixed_owner_minimum_views),
        "fixed_owner_polygon_xz_m": (
            np.asarray(fixed_owner_polygon_xz, dtype=np.float64).tolist()
            if fixed_owner_polygon_xz is not None
            else None
        ),
    }
    suppressed_observations = suppressed_moving["observation_count"]
    suppressed_audit = {
        "points": _points_from_records(suppressed_moving).astype(np.float32),
        "colors": np.clip(
            suppressed_moving["color_weighted_sum"]
            / np.maximum(suppressed_moving["weight_sum"][:, None], 1e-12),
            0.0,
            255.0,
        ).astype(np.uint8),
        "voxel_keys": suppressed_moving["key"].astype(np.int32),
        "fusion_weight_sum": suppressed_moving["weight_sum"].astype(np.float32),
        "observation_count": suppressed_observations.astype(np.uint32),
        "view_mask_words": suppressed_moving["view_mask_words"].astype(np.uint64),
        "view_count": _popcount_words(
            suppressed_moving["view_mask_words"]
        ).astype(np.uint16),
        "agreement_observation_count": suppressed_moving[
            "agreement_observation_count"
        ].astype(np.uint32),
        "agreement_weight_sum": suppressed_moving["agreement_weight_sum"].astype(
            np.float32
        ),
        "source_selection_counts": suppressed_moving[
            "source_selection_counts"
        ].astype(np.uint32),
        "suppression_reason_mask": (
            exact_fixed[suppressed].astype(np.uint8)
            | (column_owned[suppressed].astype(np.uint8) << np.uint8(1))
        ),
        "voxel_size_m": np.asarray([settings.voxel_size_m], dtype=np.float32),
    }
    return output, suppressed_audit, ownership


def _write_glb(
    path: Path,
    data: Mapping[str, np.ndarray],
    fixed_name: str,
    moving_name: str,
) -> None:
    import trimesh

    scene = trimesh.Scene()
    owners = np.asarray(data["owner_room_id"])
    for room_id, name in ((1, fixed_name), (2, moving_name)):
        selected = owners == room_id
        if np.any(selected):
            rgba = np.column_stack(
                (
                    data["colors"][selected],
                    np.full(np.count_nonzero(selected), 235, dtype=np.uint8),
                )
            )
            scene.add_geometry(
                trimesh.points.PointCloud(data["points"][selected], colors=rgba),
                node_name=f"pcf_{name}_owned_surfels",
            )
    scene.export(path)


def _write_topdown(
    path: Path,
    data: Mapping[str, np.ndarray],
    fixed_name: str,
    moving_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(data["points"])
    colors = np.asarray(data["colors"])
    owners = np.asarray(data["owner_room_id"])
    # Keep every voxel in the NPZ/GLB.  The review raster suppresses isolated
    # one-view splats so capture rays do not obscure the actual room surfaces.
    supported = np.asarray(data["view_count"]) >= 2
    if int(np.count_nonzero(supported)) < 10_000:
        supported = np.ones(len(points), dtype=bool)
    supported_indices = np.flatnonzero(supported)
    stride = max(1, int(math.ceil(len(supported_indices) / 260_000)))
    sampled = supported_indices[::stride]
    figure, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    figure.patch.set_facecolor("#11151c")
    for axis in axes:
        axis.set_facecolor("#11151c")
        axis.tick_params(colors="#d8dde6")
        axis.xaxis.label.set_color("#d8dde6")
        axis.yaxis.label.set_color("#d8dde6")
        for spine in axis.spines.values():
            spine.set_color("#697386")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("fixed backend world X (m)")
        axis.set_ylabel("fixed backend world Z (m)")
        axis.grid(alpha=0.12)
    axes[0].scatter(
        points[sampled, 0],
        points[sampled, 2],
        s=0.42,
        c=colors[sampled] / 255.0,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("Continuous PCF RGB surfels (2+ source views)", color="white")
    palette = np.asarray([[0, 0, 0], [92, 177, 255], [244, 173, 82]], dtype=np.float32) / 255.0
    axes[1].scatter(
        points[sampled, 0],
        points[sampled, 2],
        s=0.42,
        c=palette[owners[sampled]],
        linewidths=0,
        rasterized=True,
    )
    axes[1].scatter([], [], color="#66d9ff", label=fixed_name)
    axes[1].scatter([], [], color="#ffad52", label=moving_name)
    axes[1].legend(facecolor="#11151c", labelcolor="white", framealpha=0.8)
    axes[1].set_title(
        f"Ownership: {fixed_name} wins overlap; {moving_name} extends outside",
        color="white",
    )
    # Use the complete finite extent; this review intentionally does not apply
    # percentile cropping, the source of an earlier misleading room cutoff.
    finite_xz = points[supported & np.isfinite(points).all(axis=1)][:, [0, 2]]
    low = np.min(finite_xz, axis=0) - 0.35
    high = np.max(finite_xz, axis=0) + 0.35
    for axis in axes:
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
    figure.suptitle(
        "Two-room PCF reintegration · accepted phone-walk evidence only",
        color="white",
        fontsize=16,
    )
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def reintegrate(
    *,
    fixed: RoomSource,
    moving: RoomSource,
    output_dir: Path,
    settings: ReintegrationSettings = ReintegrationSettings(),
    fixed_owner_polygon_xz: np.ndarray | None = None,
    registration_disposition: str = "caller_supplied_unvalidated",
    registration_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse registered PCF raw evidence and write review artifacts.

    The caller remains responsible for accepting the registration and any
    per-view corrections.  This function only realizes those corrections.
    """

    if settings.voxel_size_m <= 0.0 or settings.ownership_cell_m <= 0.0:
        raise PCFReintegrationError("voxel and ownership cells must be positive")
    if settings.merge_batch_size < 2:
        raise PCFReintegrationError("merge_batch_size must be at least 2")
    allowed_dispositions = {
        "caller_supplied_unvalidated",
        "validated_review_candidate",
        "accepted_cross_session_registration",
    }
    if registration_disposition not in allowed_dispositions:
        raise PCFReintegrationError(
            "registration_disposition must be one of "
            + ", ".join(sorted(allowed_dispositions))
        )
    accepted_registration = (
        registration_disposition == "accepted_cross_session_registration"
    )
    if accepted_registration and not registration_provenance:
        raise PCFReintegrationError(
            "an accepted registration requires explicit registration provenance"
        )
    if output_dir.exists():
        raise PCFReintegrationError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="pcf_reintegration_", dir=output_dir))
    try:
        fixed_fusion = _fuse_room(fixed, settings, temporary_root)
        moving_fusion = _fuse_room(moving, settings, temporary_root)
        # A shared record dtype permits rooms with different numbers of views
        # (for example 48 and 80) without truncating either provenance bitset.
        common_view_mask_words = max(
            fixed_fusion.view_mask_words, moving_fusion.view_mask_words
        )
        fixed_fusion.records = _pad_view_mask_words(
            fixed_fusion.records, common_view_mask_words
        )
        moving_fusion.records = _pad_view_mask_words(
            moving_fusion.records, common_view_mask_words
        )
        fixed_fusion.view_mask_words = common_view_mask_words
        moving_fusion.view_mask_words = common_view_mask_words
        data, suppressed_audit, ownership = _compose_output(
            fixed_fusion,
            moving_fusion,
            settings,
            fixed_owner_polygon_xz,
        )
        if not len(data["points"]):
            raise PCFReintegrationError("ownership and fusion rejected every voxel")
        npz_path = output_dir / "multiroom_pcf_surfels.npz"
        np.savez_compressed(npz_path, **data)
        suppressed_path = output_dir / "suppressed_moving_overlap_evidence.npz"
        np.savez_compressed(suppressed_path, **suppressed_audit)
        artifacts: dict[str, str] = {
            "surfels_npz": npz_path.name,
            "suppressed_overlap_audit_npz": suppressed_path.name,
        }
        if settings.write_glb:
            glb_path = output_dir / "multiroom_pcf_reconstruction.glb"
            _write_glb(glb_path, data, fixed.name, moving.name)
            artifacts["reconstruction_glb"] = glb_path.name
        if settings.write_topdown:
            topdown_path = output_dir / "multiroom_pcf_topdown.png"
            _write_topdown(topdown_path, data, fixed.name, moving.name)
            artifacts["topdown_review"] = topdown_path.name

        _, fixed_manifest_payload = _manifest_transform(fixed.world_manifest)
        _, moving_manifest_payload = _manifest_transform(moving.world_manifest)
        report: dict[str, Any] = {
            "schema": "noesis.pcf.multiroom_reintegration.v1",
            "generated_at": _utc_now(),
            "status": "complete" if accepted_registration else "review_only",
            "method": "all_accepted_raw_pcf_points_confidence_and_agreement_weighted_2_5cm_voxels",
            "phone_walk_only": True,
            "static_camera_points_included": False,
            "whole_cloud_icp_used": False,
            "output_coordinate_frame": "fixed_room_accepted_backend_world_m",
            "registration": {
                "disposition": registration_disposition,
                "accepted_for_canonical_use": accepted_registration,
                "provenance": dict(registration_provenance or {}),
            },
            "fixed_room": _room_report(fixed_fusion, fixed_manifest_payload),
            "moving_room": _room_report(moving_fusion, moving_manifest_payload),
            "fusion": {
                "voxel_size_m": float(settings.voxel_size_m),
                "minimum_confidence": float(settings.minimum_confidence),
                "minimum_depth_m": float(settings.minimum_depth_m),
                "maximum_depth_m": float(settings.maximum_depth_m),
                "agreement_weight_bonus": float(settings.agreement_weight_bonus),
                "output_voxel_count": int(len(data["points"])),
                "output_observation_count": int(
                    np.sum(data["observation_count"], dtype=np.uint64)
                ),
                "output_distinct_view_support_median": float(
                    np.median(data["view_count"])
                ),
                "agreement_fraction_weighted_by_observations": float(
                    np.sum(data["agreement_observation_count"], dtype=np.uint64)
                    / max(1, np.sum(data["observation_count"], dtype=np.uint64))
                ),
                "point_sampling": "none_all_accepted_raw_pixels_streamed",
                "centroid": "confidence_times_consensus_agreement_weighted_mean_per_voxel",
            },
            "ownership": ownership,
            "npz_contract": {
                "room_ids": {"1": fixed.name, "2": moving.name},
                "room_presence_mask_bits": {"1": fixed.name, "2": moving.name},
                "room_contribution_mask_bits": {"1": fixed.name, "2": moving.name},
                "source_selection_columns": [
                    "rejected_or_unknown",
                    "cross_model_consensus",
                    "mapanything_selected_by_consistency",
                    "da3_selected_by_consistency",
                    "mapanything_only_validated_fill",
                    "da3_only_validated_fill",
                ],
                "fixed_view_mask_word_count": fixed_fusion.view_mask_words,
                "moving_view_mask_word_count": moving_fusion.view_mask_words,
            },
            "artifacts": artifacts,
        }
        manifest_path = output_dir / "multiroom_pcf_manifest.json"
        manifest_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["manifest"] = manifest_path.name
        report["artifact_sha256"] = {
            name: _sha256(output_dir / relative)
            for name, relative in artifacts.items()
            if name != "manifest"
        }
        manifest_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception:
        # Keep the output directory and any completed evidence for diagnosis,
        # but never leave large temporary reduction runs behind.
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _room_report(
    fusion: RoomFusion,
    world_manifest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    source = fusion.source
    return {
        "name": source.name,
        "prior_id": source.prior_id,
        "scan_id": source.scan_id,
        "raw_root": str(source.raw_root),
        "view_count": fusion.view_count,
        "selected_voxel_count_before_ownership": int(len(fusion.records)),
        "accepted_world_manifest": str(source.world_manifest),
        "accepted_world_manifest_sha256": _sha256(source.world_manifest),
        "accepted_source_coordinate_frame": world_manifest_payload.get(
            "source_coordinate_frame"
        ),
        "accepted_output_coordinate_frame": world_manifest_payload.get(
            "output_coordinate_frame"
        ),
        "global_world_correction_row_major": np.asarray(
            source.global_world_correction, dtype=np.float64
        ).tolist(),
        "per_view_world_corrections_row_major": {
            str(index): np.asarray(matrix, dtype=np.float64).tolist()
            for index, matrix in sorted(source.per_view_world_corrections.items())
        },
        "selection_counts": fusion.counters,
        "raw_views": fusion.view_files,
    }


def _correction_payload(path: Path) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    solution = payload.get("solution") if isinstance(payload.get("solution"), dict) else payload
    passed = payload.get("status") == "passed" or solution.get("passed") is True
    global_value = solution.get("moving_correction_in_backend_world_row_major")
    per_view_value = solution.get("per_view_world_corrections_row_major")
    if per_view_value is None:
        per_view_value = payload.get("per_view_world_corrections_row_major", {})
    if global_value is not None and not passed:
        raise PCFReintegrationError(
            f"refusing a non-passed registration correction from {path}"
        )
    global_correction = (
        _validate_matrix(
            np.asarray(global_value, dtype=np.float64),
            label=f"global correction from {path}",
            permit_metric_scale=False,
        )
        if global_value is not None
        else None
    )
    if not isinstance(per_view_value, dict):
        raise PCFReintegrationError(f"per-view corrections in {path} are not a mapping")
    per_view = {
        int(index): _validate_matrix(
            np.asarray(matrix, dtype=np.float64),
            label=f"view {index} correction from {path}",
            permit_metric_scale=False,
        )
        for index, matrix in per_view_value.items()
    }
    return global_correction, per_view


def _ownership_polygon(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("fixed_owner_polygon_xz_m", payload)
    return np.asarray(value, dtype=np.float64)


def _write_smoke_raw(path: Path, points: np.ndarray, view_index: int) -> None:
    height, width = points.shape[:2]
    confidence = np.full((height, width), 0.8, dtype=np.float32)
    np.savez_compressed(
        path,
        world_points=points.astype(np.float32),
        depth_z=np.full((height, width), 2.0, dtype=np.float32),
        confidence=confidence,
        mask=np.ones((height, width), dtype=bool),
        camera_pose=np.eye(4, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        metric_scaling_factor=np.asarray([1.0], dtype=np.float32),
        model_rgb=np.full((height, width, 3), 80 + view_index * 20, dtype=np.uint8),
        source_selection=np.ones((height, width), dtype=np.uint8),
        cross_model_uncertain=np.zeros((height, width), dtype=bool),
        cross_model_agreement=np.ones((height, width), dtype=bool),
        absolute_depth_disagreement_m=np.full((height, width), 0.03, dtype=np.float32),
        mapanything_reliability=np.full((height, width), 0.8, dtype=np.float32),
        da3_reliability=np.full((height, width), 0.85, dtype=np.float32),
    )


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="pcf_reintegration_smoke_") as root_value:
        root = Path(root_value)
        fixed_raw = root / "fixed" / "raw"
        moving_raw = root / "moving" / "raw"
        fixed_raw.mkdir(parents=True)
        moving_raw.mkdir(parents=True)
        manifest = {
            "alignment": {
                "scale": 1.0,
                "rotation_row_major": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            },
            "source_coordinate_frame": "synthetic_local",
            "output_coordinate_frame": "synthetic_world",
        }
        fixed_manifest = root / "fixed_world.json"
        moving_manifest = root / "moving_world.json"
        fixed_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        moving_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        fixed_points = np.asarray(
            [[[0.012, 0.0, 0.012], [0.112, 0.0, 0.012], [0.212, 0.0, 0.012]]],
            dtype=np.float32,
        )
        moving_points = np.asarray(
            [[[0.012, 0.0, 0.012], [0.112, 0.0, 0.012], [0.912, 0.0, 0.012]]],
            dtype=np.float32,
        )
        for index in range(2):
            _write_smoke_raw(fixed_raw / f"view_{index:04d}.npz", fixed_points, index)
            # Exercise both correction levels: +10 cm globally, plus +2 cm on
            # the second moving view, returns both synthetic views to the same
            # fixed-world points before ownership is evaluated.
            local_offset = 0.10 + (0.02 if index == 1 else 0.0)
            shifted = moving_points.copy()
            shifted[..., 0] -= local_offset
            _write_smoke_raw(moving_raw / f"view_{index:04d}.npz", shifted, index)
        output = root / "output"
        moving_global = np.eye(4, dtype=np.float64)
        moving_global[0, 3] = 0.10
        moving_view_one = np.eye(4, dtype=np.float64)
        moving_view_one[0, 3] = 0.02
        report = reintegrate(
            fixed=RoomSource("family", fixed_raw, fixed_manifest),
            moving=RoomSource(
                "kitchen",
                moving_raw,
                moving_manifest,
                global_world_correction=moving_global,
                per_view_world_corrections={1: moving_view_one},
            ),
            output_dir=output,
            settings=ReintegrationSettings(
                hash_raw_views=False,
                write_glb=True,
                write_topdown=True,
                ownership_dilation_m=0.0,
            ),
            registration_disposition="caller_supplied_unvalidated",
        )
        with np.load(output / "multiroom_pcf_surfels.npz", allow_pickle=False) as row:
            owners = np.asarray(row["owner_room_id"])
            points = np.asarray(row["points"])
            if np.count_nonzero(owners == 1) != 3:
                raise AssertionError("fixed room did not retain its owned voxels")
            if np.count_nonzero(owners == 2) != 1:
                raise AssertionError("moving-only evidence outside overlap was lost")
            if not np.any(np.isclose(points[owners == 2, 0], 0.912, atol=0.02)):
                raise AssertionError("moving-room extension was not preserved")
            shared = np.asarray(row["room_presence_mask"]) == 3
            if np.count_nonzero(shared) != 2:
                raise AssertionError("exact overlap provenance was not retained")
            if not np.all(np.asarray(row["view_count"]) >= 2):
                raise AssertionError("per-voxel view support was not retained")
        if report["ownership"]["moving_suppressed_voxel_count"] != 2:
            raise AssertionError("unexpected ownership suppression count")
        if report["status"] != "review_only" or report["registration"][
            "accepted_for_canonical_use"
        ]:
            raise AssertionError("default registration disposition did not fail closed")
        with np.load(
            output / "suppressed_moving_overlap_evidence.npz", allow_pickle=False
        ) as row:
            if len(row["points"]) != 2 or not np.all(row["view_count"] == 2):
                raise AssertionError("suppressed overlap provenance was not preserved")
        if not (output / "multiroom_pcf_reconstruction.glb").is_file():
            raise AssertionError("GLB writer did not produce an artifact")
        if not (output / "multiroom_pcf_topdown.png").is_file():
            raise AssertionError("top-down writer did not produce an artifact")
    print("PCF two-room reintegration smoke: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-name")
    parser.add_argument("--fixed-prior-id")
    parser.add_argument("--fixed-scan-id")
    parser.add_argument("--fixed-raw-root", type=Path)
    parser.add_argument("--fixed-world-manifest", type=Path)
    parser.add_argument("--fixed-corrections", type=Path)
    parser.add_argument("--moving-name")
    parser.add_argument("--moving-prior-id")
    parser.add_argument("--moving-scan-id")
    parser.add_argument("--moving-raw-root", type=Path)
    parser.add_argument("--moving-world-manifest", type=Path)
    parser.add_argument("--moving-registration-report", type=Path)
    parser.add_argument("--moving-corrections", type=Path)
    parser.add_argument(
        "--registration-disposition",
        choices=(
            "caller_supplied_unvalidated",
            "validated_review_candidate",
            "accepted_cross_session_registration",
        ),
        default="caller_supplied_unvalidated",
    )
    parser.add_argument("--fixed-owner-polygon", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--voxel-m", type=float, default=0.025)
    parser.add_argument("--ownership-cell-m", type=float, default=0.10)
    parser.add_argument("--ownership-dilation-m", type=float, default=0.10)
    parser.add_argument("--no-hash-raw-views", action="store_true")
    parser.add_argument("--no-glb", action="store_true")
    parser.add_argument("--no-topdown", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _require_cli(arguments: argparse.Namespace, names: Iterable[str]) -> None:
    missing = [name for name in names if getattr(arguments, name) in (None, "")]
    if missing:
        raise PCFReintegrationError(
            "missing required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.self_test:
        _self_test()
        return 0
    _require_cli(
        arguments,
        (
            "fixed_name",
            "fixed_raw_root",
            "fixed_world_manifest",
            "moving_name",
            "moving_raw_root",
            "moving_world_manifest",
            "moving_registration_report",
            "output_dir",
        ),
    )
    fixed_global = np.eye(4, dtype=np.float64)
    fixed_per_view: dict[int, np.ndarray] = {}
    if arguments.fixed_corrections is not None:
        global_value, fixed_per_view = _correction_payload(arguments.fixed_corrections)
        if global_value is not None:
            fixed_global = global_value
    moving_global, moving_per_view = _correction_payload(
        arguments.moving_registration_report
    )
    if moving_global is None:
        raise PCFReintegrationError("moving registration report has no global correction")
    if arguments.moving_corrections is not None:
        supplemental_global, supplemental_per_view = _correction_payload(
            arguments.moving_corrections
        )
        if supplemental_global is not None:
            moving_global = supplemental_global @ moving_global
        moving_per_view.update(supplemental_per_view)
    report = reintegrate(
        fixed=RoomSource(
            name=arguments.fixed_name,
            prior_id=arguments.fixed_prior_id,
            scan_id=arguments.fixed_scan_id,
            raw_root=arguments.fixed_raw_root.resolve(),
            world_manifest=arguments.fixed_world_manifest.resolve(),
            global_world_correction=fixed_global,
            per_view_world_corrections=fixed_per_view,
        ),
        moving=RoomSource(
            name=arguments.moving_name,
            prior_id=arguments.moving_prior_id,
            scan_id=arguments.moving_scan_id,
            raw_root=arguments.moving_raw_root.resolve(),
            world_manifest=arguments.moving_world_manifest.resolve(),
            global_world_correction=moving_global,
            per_view_world_corrections=moving_per_view,
        ),
        output_dir=arguments.output_dir.resolve(),
        settings=ReintegrationSettings(
            voxel_size_m=arguments.voxel_m,
            ownership_cell_m=arguments.ownership_cell_m,
            ownership_dilation_m=arguments.ownership_dilation_m,
            hash_raw_views=not arguments.no_hash_raw_views,
            write_glb=not arguments.no_glb,
            write_topdown=not arguments.no_topdown,
        ),
        fixed_owner_polygon_xz=_ownership_polygon(arguments.fixed_owner_polygon),
        registration_disposition=arguments.registration_disposition,
        registration_provenance={
            "registration_report": str(arguments.moving_registration_report.resolve()),
            "registration_report_sha256": _sha256(
                arguments.moving_registration_report.resolve()
            ),
            "registration_report_status": json.loads(
                arguments.moving_registration_report.read_text(encoding="utf-8")
            ).get("status"),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
