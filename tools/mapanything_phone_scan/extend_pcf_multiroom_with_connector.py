#!/usr/bin/env python3
"""Extend an existing PCF multi-room join with an accepted room and bridge walk.

Existing joined voxels are immutable ownership authority.  Accepted room-walk
voxels extend the world using the buffered X/Z envelope as an authority
candidate, then require compatible three-dimensional geometry and retained
camera/depth/pose support before suppressing a source voxel.  Different-height
surfaces and measured passage evidence therefore remain available inside the
same X/Z column.

The connector transform may come from a rejected pose-graph candidate, but in
that case the output is unconditionally marked review-only and non-canonical.
No whole-cloud ICP or nearest-neighbour surface fitting is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.reintegrate_pcf_rooms import (  # noqa: E402
    ReintegrationSettings,
    RoomSource,
    ViewEvidence,
    _fuse_room,
    _points_from_records,
    _popcount_words,
    _three_dimensional_compatibility,
)


class MultiroomExtensionError(RuntimeError):
    """Raised when a source cannot be joined without losing authority."""


_CORE_FIELDS = (
    "points",
    "colors",
    "voxel_keys",
    "fusion_weight_sum",
    "mean_confidence",
    "observation_count",
    "view_count",
    "agreement_observation_count",
    "agreement_fraction",
    "agreement_weight_sum",
    "mean_absolute_depth_disagreement_m",
    "mean_mapanything_reliability",
    "mean_da3_reliability",
    "source_selection_counts",
    "owner_room_id",
    "room_contribution_mask",
    "room_presence_mask",
    "geometry_status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MultiroomExtensionError(f"{path} does not contain a JSON object")
    return value


def _parse_view_indices(value: str | None) -> frozenset[int] | None:
    if value is None:
        return None
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise MultiroomExtensionError(
                    f"connector view range is reversed: {token}"
                )
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    if len(selected) < 2:
        raise MultiroomExtensionError(
            "connector view selection must contain at least two views"
        )
    return frozenset(selected)


def _keys_as_structured(keys: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(keys, dtype=np.int32)
    return value.view(
        np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
    ).reshape(-1)


def _record_payload(records: np.ndarray) -> dict[str, np.ndarray]:
    observations = records["observation_count"].astype(np.uint32)
    points = _points_from_records(records).astype(np.float32)
    agreement_count = records["agreement_observation_count"].astype(np.uint32)
    disagreement = np.divide(
        records["disagreement_weighted_sum"],
        records["disagreement_weight_sum"],
        out=np.full(len(records), np.nan, dtype=np.float64),
        where=records["disagreement_weight_sum"] > 0.0,
    ).astype(np.float32)
    map_reliability = np.divide(
        records["map_reliability_sum"],
        records["map_reliability_count"],
        out=np.full(len(records), np.nan, dtype=np.float64),
        where=records["map_reliability_count"] > 0,
    ).astype(np.float32)
    da3_reliability = np.divide(
        records["da3_reliability_sum"],
        records["da3_reliability_count"],
        out=np.full(len(records), np.nan, dtype=np.float64),
        where=records["da3_reliability_count"] > 0,
    ).astype(np.float32)
    return {
        "points": points,
        "colors": np.clip(
            records["color_weighted_sum"]
            / np.maximum(records["weight_sum"][:, None], 1e-12),
            0.0,
            255.0,
        ).astype(np.uint8),
        "voxel_keys": records["key"].astype(np.int32),
        "fusion_weight_sum": records["weight_sum"].astype(np.float32),
        "mean_confidence": (
            records["confidence_sum"] / np.maximum(observations, 1)
        ).astype(np.float32),
        "observation_count": observations,
        "view_count": _popcount_words(records["view_mask_words"]).astype(np.uint16),
        "agreement_observation_count": agreement_count,
        "agreement_fraction": (
            agreement_count / np.maximum(observations, 1)
        ).astype(np.float32),
        "agreement_weight_sum": records["agreement_weight_sum"].astype(np.float32),
        "mean_absolute_depth_disagreement_m": disagreement,
        "mean_mapanything_reliability": map_reliability,
        "mean_da3_reliability": da3_reliability,
        "source_selection_counts": records["source_selection_counts"].astype(
            np.uint32
        ),
        "geometry_status": np.zeros(len(records), dtype=np.uint8),
    }


def _manifest_view_evidence(room_report: Mapping[str, Any]) -> list[ViewEvidence]:
    """Reload the exact retained RGB-D view evidence named by a prior report."""

    raw_views = room_report.get("raw_views")
    if not isinstance(raw_views, list) or not raw_views:
        raise MultiroomExtensionError(
            "existing room report has no explicit raw-view visibility provenance"
        )
    evidence: list[ViewEvidence] = []
    for item in raw_views:
        if not isinstance(item, Mapping):
            raise MultiroomExtensionError("existing raw-view provenance is malformed")
        path = Path(str(item.get("path") or ""))
        try:
            view_index = int(item["view_index"])
            transform = np.asarray(
                item["effective_world_from_local_row_major"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MultiroomExtensionError(
                "existing raw-view provenance lacks an effective transform"
            ) from exc
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise MultiroomExtensionError("existing raw-view transform is malformed")
        try:
            with np.load(path, allow_pickle=False) as row:
                depth = np.asarray(row["depth_z"], dtype=np.float32).copy()
                mask = np.asarray(row["mask"], dtype=bool).copy()
                intrinsics = np.asarray(row["intrinsics"], dtype=np.float64).copy()
                camera_pose = np.asarray(row["camera_pose"], dtype=np.float64)
        except (OSError, KeyError, ValueError) as exc:
            raise MultiroomExtensionError(
                f"existing raw-view visibility evidence cannot be read: {path}"
            ) from exc
        if (
            depth.ndim != 2
            or mask.shape != depth.shape
            or intrinsics.shape != (3, 3)
            or camera_pose.shape != (4, 4)
            or not np.isfinite(intrinsics).all()
            or not np.isfinite(camera_pose).all()
        ):
            raise MultiroomExtensionError(
                f"existing raw-view visibility evidence is malformed: {path}"
            )
        evidence.append(
            ViewEvidence(
                view_index=view_index,
                intrinsics=intrinsics,
                depth_z=depth,
                mask=mask,
                camera_to_world=transform @ camera_pose,
            )
        )
    return evidence


def _coverage_columns(points: np.ndarray, cell_m: float) -> np.ndarray:
    columns = np.floor(points[:, [0, 2]] / cell_m).astype(np.int32)
    return np.unique(columns, axis=0)


def _buffer_columns(
    columns: np.ndarray, *, cell_m: float, buffer_m: float
) -> np.ndarray:
    if buffer_m <= 0.0 or len(columns) == 0:
        return columns
    radius = int(math.ceil(buffer_m / cell_m))
    minimum = np.min(columns, axis=0) - radius
    maximum = np.max(columns, axis=0) + radius
    shape = tuple((maximum - minimum + 1).tolist())
    source = np.zeros(shape, dtype=bool)
    local = columns - minimum
    source[local[:, 0], local[:, 1]] = True
    buffered = np.zeros_like(source)
    for delta_x in range(-radius, radius + 1):
        for delta_z in range(-radius, radius + 1):
            if math.hypot(delta_x, delta_z) * cell_m > buffer_m + 1e-9:
                continue
            destination_x = slice(
                max(0, delta_x), min(shape[0], shape[0] + delta_x)
            )
            destination_z = slice(
                max(0, delta_z), min(shape[1], shape[1] + delta_z)
            )
            source_x = slice(
                max(0, -delta_x), min(shape[0], shape[0] - delta_x)
            )
            source_z = slice(
                max(0, -delta_z), min(shape[1], shape[1] - delta_z)
            )
            buffered[destination_x, destination_z] |= source[source_x, source_z]
    return np.argwhere(buffered).astype(np.int32) + minimum


def _column_membership(query_points: np.ndarray, columns: np.ndarray, cell_m: float) -> np.ndarray:
    query = np.floor(query_points[:, [0, 2]] / cell_m).astype(np.int32)
    source = np.ascontiguousarray(columns).view(
        np.dtype([("x", "<i4"), ("z", "<i4")])
    ).reshape(-1)
    query_structured = np.ascontiguousarray(query).view(
        np.dtype([("x", "<i4"), ("z", "<i4")])
    ).reshape(-1)
    positions = np.searchsorted(source, query_structured)
    valid = positions < len(source)
    result = np.zeros(len(query), dtype=bool)
    result[valid] = source[positions[valid]] == query_structured[valid]
    return result


def _append_source(
    output: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
    *,
    owner_id: int,
    room_bit: int,
    ownership_cell_m: float,
    authority_buffer_m: float,
    existing_view_evidence: Sequence[ViewEvidence],
    source_view_evidence: Sequence[ViewEvidence],
    geometry_compatibility_distance_m: float = 0.06,
    maximum_depth_m: float = 12.0,
    minimum_depth_m: float = 0.05,
    depth_tolerance_m: float = 0.05,
    depth_tolerance_fraction: float = 0.03,
    retain_geometry: bool = True,
) -> dict[str, int]:
    existing_keys = _keys_as_structured(output["voxel_keys"])
    source_keys = _keys_as_structured(source["voxel_keys"])
    positions = np.searchsorted(existing_keys, source_keys)
    exact = positions < len(existing_keys)
    exact[exact] &= existing_keys[positions[exact]] == source_keys[exact]
    measured_columns = _coverage_columns(output["points"], ownership_cell_m)
    covered_columns = _buffer_columns(
        measured_columns,
        cell_m=ownership_cell_m,
        buffer_m=authority_buffer_m,
    )
    if not retain_geometry:
        return {
            "input_voxel_count": int(len(source["points"])),
            "exact_existing_voxel_count": 0,
            "covered_existing_xz_column_voxel_count": 0,
            "measured_authority_column_count": int(len(measured_columns)),
            "buffered_authority_column_count": int(len(covered_columns)),
            "authority_buffer_m": authority_buffer_m,
            "suppressed_voxel_count": int(len(source["points"])),
            "retained_voxel_count": 0,
            "geometry_omitted_by_policy": True,
        }
    column_owned = _column_membership(
        source["points"], covered_columns, ownership_cell_m
    )
    (
        compatible_3d,
        source_ray_valid,
        existing_ray_valid,
        nearest_existing_distance,
        source_cross_free_space,
        source_cross_unknown,
        source_own_contradiction,
        existing_own_contradiction,
        source_own_unknown,
        existing_own_unknown,
    ) = (
        _three_dimensional_compatibility(
            output["points"],
            source["points"],
            existing_view_evidence,
            source_view_evidence,
            distance_m=geometry_compatibility_distance_m,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
            depth_tolerance_m=depth_tolerance_m,
            depth_tolerance_fraction=depth_tolerance_fraction,
        )
    )
    geometry_overlap = column_owned & compatible_3d
    suppressed = exact | geometry_overlap
    source_uncertain = source_own_contradiction | source_cross_free_space
    contradictory_geometry = (
        ~exact
        & source_uncertain
        & np.isfinite(nearest_existing_distance)
        & (nearest_existing_distance > geometry_compatibility_distance_m)
    )
    source_existing_status = np.asarray(source["geometry_status"], dtype=np.uint8)
    source_geometry_status = np.where(
        source_uncertain | (source_existing_status == 1),
        np.uint8(1),
        np.where(
            source_own_unknown | (source_existing_status == 2),
            np.uint8(2),
            np.uint8(0),
        ),
    )
    retained = ~suppressed

    exact_source = np.flatnonzero(exact)
    if len(exact_source):
        output["room_presence_mask"][positions[exact_source]] |= np.uint8(room_bit)

    base_count = len(output["points"])
    for name in _CORE_FIELDS:
        if name in {"owner_room_id", "room_contribution_mask", "room_presence_mask"}:
            continue
        output[name] = np.concatenate((output[name], source[name][retained]), axis=0)
    retained_count = int(np.count_nonzero(retained))
    # Preserve source uncertainty and mark a co-visible but geometrically
    # contradictory source row as uncertain.  It remains available for review
    # with provenance, while it cannot be mistaken for an observed overlap.
    if retained_count:
        output["geometry_status"][-retained_count:] = source_geometry_status[retained]
    output["owner_room_id"] = np.concatenate(
        (
            output["owner_room_id"],
            np.full(retained_count, owner_id, dtype=np.uint8),
        )
    )
    output["room_contribution_mask"] = np.concatenate(
        (
            output["room_contribution_mask"],
            np.full(retained_count, room_bit, dtype=np.uint8),
        )
    )
    output["room_presence_mask"] = np.concatenate(
        (
            output["room_presence_mask"],
            np.full(retained_count, room_bit, dtype=np.uint8),
        )
    )
    order = np.lexsort(
        (
            output["voxel_keys"][:, 2],
            output["voxel_keys"][:, 1],
            output["voxel_keys"][:, 0],
        )
    )
    for name in _CORE_FIELDS:
        output[name] = output[name][order]
    if len(output["voxel_keys"]) != base_count + retained_count:
        raise MultiroomExtensionError("source append changed the expected row count")
    duplicated = np.all(
        output["voxel_keys"][1:] == output["voxel_keys"][:-1], axis=1
    )
    if np.any(duplicated):
        raise MultiroomExtensionError("ownership append left duplicate voxels")
    return {
        "input_voxel_count": int(len(source["points"])),
        "exact_existing_voxel_count": int(np.count_nonzero(exact)),
        "covered_existing_xz_column_voxel_count": int(
            np.count_nonzero(column_owned)
        ),
        "three_dimensional_compatible_voxel_count": int(
            np.count_nonzero(compatible_3d)
        ),
        "source_ray_depth_pose_valid_count": int(np.count_nonzero(source_ray_valid)),
        "existing_ray_depth_pose_valid_count": int(
            np.count_nonzero(existing_ray_valid)
        ),
        "source_cross_room_unknown_count": int(
            np.count_nonzero(source_cross_unknown)
        ),
        "source_cross_room_free_space_contradiction_count": int(
            np.count_nonzero(source_cross_free_space)
        ),
        "source_own_view_free_space_contradiction_count": int(
            np.count_nonzero(source_own_contradiction)
        ),
        "existing_own_view_free_space_contradiction_count": int(
            np.count_nonzero(existing_own_contradiction)
        ),
        "source_own_view_unknown_count": int(np.count_nonzero(source_own_unknown)),
        "existing_own_view_unknown_count": int(
            np.count_nonzero(existing_own_unknown)
        ),
        "measured_authority_column_count": int(len(measured_columns)),
        "buffered_authority_column_count": int(len(covered_columns)),
        "authority_buffer_m": authority_buffer_m,
        "suppressed_voxel_count": int(np.count_nonzero(suppressed)),
        "suppressed_by_3d_compatibility_count": int(
            np.count_nonzero(geometry_overlap & ~exact)
        ),
        "complementary_height_retained_count": int(
            np.count_nonzero(column_owned & ~compatible_3d & ~exact)
        ),
        "contradictory_geometry_uncertain_count": int(
            np.count_nonzero(contradictory_geometry & ~suppressed)
        ),
        "contradictory_geometry_within_conflict_band_count": int(
            np.count_nonzero(
                contradictory_geometry
                & (nearest_existing_distance <= 0.30)
            )
        ),
        "retained_voxel_count": retained_count,
    }


def _write_glb(path: Path, output: Mapping[str, np.ndarray]) -> None:
    import trimesh

    labels = {1: "family", 2: "kitchen", 3: "living", 4: "connector"}
    scene = trimesh.Scene()
    for owner_id, label in labels.items():
        selected = output["owner_room_id"] == owner_id
        if not np.any(selected):
            continue
        rgba = np.column_stack(
            (
                output["colors"][selected],
                np.full(np.count_nonzero(selected), 235, dtype=np.uint8),
            )
        )
        scene.add_geometry(
            trimesh.points.PointCloud(output["points"][selected], colors=rgba),
            node_name=f"pcf_{label}_owned_surfels",
        )
    scene.export(path)


def extend(
    *,
    existing_npz_path: Path,
    existing_manifest_path: Path,
    living_raw_root: Path,
    living_world_manifest: Path,
    connector_raw_root: Path,
    connector_world_manifest: Path,
    multianchor_report_path: Path,
    output_dir: Path,
    ownership_cell_m: float = 0.10,
    accepted_room_authority_buffer_m: float = 0.0,
    connector_endpoint_authority_buffer_m: float = 0.75,
    connector_view_indices: frozenset[int] | None = None,
    include_connector_geometry: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise MultiroomExtensionError(f"output directory already exists: {output_dir}")
    if not math.isclose(ownership_cell_m, 0.10, abs_tol=1e-9):
        raise MultiroomExtensionError("review extension currently requires 0.10 m ownership cells")
    if not 0.0 <= accepted_room_authority_buffer_m <= 2.0:
        raise MultiroomExtensionError(
            "accepted-room authority buffer must be between 0 and 2 metres"
        )
    if not 0.0 <= connector_endpoint_authority_buffer_m <= 2.0:
        raise MultiroomExtensionError(
            "connector-endpoint authority buffer must be between 0 and 2 metres"
        )
    source_hash_before = _sha256(existing_npz_path)
    existing_manifest = _load_json(existing_manifest_path)
    registration = _load_json(multianchor_report_path)
    candidate = registration.get("candidate")
    endpoints = registration.get("derived_endpoint_transforms")
    if not isinstance(candidate, dict) or not isinstance(endpoints, dict):
        raise MultiroomExtensionError("multi-anchor report has no review candidate")
    living_to_family = np.asarray(
        endpoints.get("living_to_family_row_major"), dtype=np.float64
    )
    if living_to_family.shape != (4, 4):
        raise MultiroomExtensionError("multi-anchor report lacks Living-to-Family")
    transforms_value = candidate.get("per_view_transforms_moving_to_fixed_row_major")
    if not isinstance(transforms_value, dict) or not transforms_value:
        raise MultiroomExtensionError("multi-anchor report lacks per-view transforms")
    connector_transforms = {
        int(index): np.asarray(value, dtype=np.float64)
        for index, value in transforms_value.items()
    }

    output_dir.mkdir(parents=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="pcf_three_room_", dir=output_dir))
    settings = ReintegrationSettings(
        voxel_size_m=0.025,
        ownership_cell_m=ownership_cell_m,
        hash_raw_views=True,
        write_glb=False,
        write_topdown=False,
    )
    try:
        living_fusion = _fuse_room(
            RoomSource(
                name="living",
                raw_root=living_raw_root,
                world_manifest=living_world_manifest,
                global_world_correction=living_to_family,
            ),
            settings,
            temporary_root,
        )
        connector_fusion = _fuse_room(
            RoomSource(
                name="connector",
                raw_root=connector_raw_root,
                world_manifest=connector_world_manifest,
                per_view_world_corrections=connector_transforms,
                included_view_indices=connector_view_indices,
            ),
            settings,
            temporary_root,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    with np.load(existing_npz_path, allow_pickle=False) as archive:
        # geometry_status was added after v1 artifacts were emitted.  Existing
        # room ownership remains valid; absent status means observed/unknown
        # was not recorded and is represented as the neutral value 0.
        missing = sorted(
            set(_CORE_FIELDS).difference(archive.files).difference({"geometry_status"})
        )
        if missing:
            raise MultiroomExtensionError(f"existing NPZ is missing {missing}")
        output = {
            name: np.asarray(archive[name]).copy()
            for name in _CORE_FIELDS
            if name in archive.files
        }
        if "geometry_status" not in output:
            output["geometry_status"] = np.zeros(
                len(output["points"]), dtype=np.uint8
            )
        fixed_camera_positions = np.asarray(
            archive["fixed_camera_positions"], dtype=np.float32
        )
        moving_camera_positions = np.asarray(
            archive["moving_camera_positions"], dtype=np.float32
        )
        voxel_size = float(np.asarray(archive["voxel_size_m"]).reshape(-1)[0])
    if not math.isclose(voxel_size, 0.025, abs_tol=1e-6):
        raise MultiroomExtensionError(f"existing voxel size is {voxel_size}, not 0.025")
    existing_view_evidence = (
        _manifest_view_evidence(existing_manifest.get("fixed_room", {}))
        + _manifest_view_evidence(existing_manifest.get("moving_room", {}))
    )

    living_payload = _record_payload(living_fusion.records)
    connector_payload = _record_payload(connector_fusion.records)
    living_ownership = _append_source(
        output,
        living_payload,
        owner_id=3,
        room_bit=4,
        ownership_cell_m=ownership_cell_m,
        authority_buffer_m=accepted_room_authority_buffer_m,
        existing_view_evidence=existing_view_evidence,
        source_view_evidence=living_fusion.view_evidence,
    )
    connector_ownership = _append_source(
        output,
        connector_payload,
        owner_id=4,
        room_bit=8,
        ownership_cell_m=ownership_cell_m,
        authority_buffer_m=connector_endpoint_authority_buffer_m,
        existing_view_evidence=existing_view_evidence + living_fusion.view_evidence,
        source_view_evidence=connector_fusion.view_evidence,
        retain_geometry=include_connector_geometry,
    )
    output.update(
        {
            "fixed_camera_positions": fixed_camera_positions,
            "moving_camera_positions": moving_camera_positions,
            "living_camera_positions": living_fusion.camera_positions.astype(
                np.float32
            ),
            "connector_camera_positions": connector_fusion.camera_positions.astype(
                np.float32
            ),
            "voxel_size_m": np.asarray([0.025], dtype=np.float32),
        }
    )
    npz_path = output_dir / "multiroom_pcf_surfels.npz"
    np.savez_compressed(npz_path, **output)
    glb_path = output_dir / "multiroom_pcf_reconstruction.glb"
    _write_glb(glb_path, output)

    source_hash_after = _sha256(existing_npz_path)
    if source_hash_before != source_hash_after:
        raise MultiroomExtensionError("existing Kitchen/Family NPZ changed during extension")
    owner_counts = {
        str(owner): int(np.count_nonzero(output["owner_room_id"] == owner))
        for owner in sorted(np.unique(output["owner_room_id"]).tolist())
    }
    report: dict[str, Any] = {
        "schema": "noesis.pcf.multiroom_connector_extension.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "method": (
            "existing_join_authority_then_accepted_living_with_connector_"
            + (
                "transition_gap_fill_"
                if include_connector_geometry
                else "registration_constraints_only_"
            )
            + "confidence_weighted_2_5cm_voxels_with_3d_overlap_gating"
        ),
        "whole_cloud_icp_used": False,
        "nearest_neighbor_surface_fitting_used": False,
        "phone_walk_only": True,
        "static_camera_points_included": False,
        "output_coordinate_frame": "family_accepted_backend_world_m",
        "canonical_limitation": (
            "multi-anchor connector registration did not pass all held-out accuracy gates"
            if not registration.get("accepted_for_canonical_use")
            else None
        ),
        "registration": {
            "path": str(multianchor_report_path),
            "sha256": _sha256(multianchor_report_path),
            "status": registration.get("status"),
            "reason_codes": registration.get("reason_codes"),
        },
        "existing_kitchen_family_authority": {
            "npz": str(existing_npz_path),
            "npz_sha256_before": source_hash_before,
            "npz_sha256_after": source_hash_after,
            "npz_unchanged": True,
            "manifest": str(existing_manifest_path),
            "manifest_sha256": _sha256(existing_manifest_path),
            "status": existing_manifest.get("status"),
        },
        "sources": {
            "living": {
                "raw_root": str(living_raw_root),
                "world_manifest": str(living_world_manifest),
                "world_manifest_sha256": _sha256(living_world_manifest),
                "view_count": living_fusion.view_count,
                "selected_observation_count": living_fusion.counters[
                    "selected_observation_count"
                ],
                "visibility_views": living_fusion.view_files,
            },
            "connector": {
                "raw_root": str(connector_raw_root),
                "world_manifest": str(connector_world_manifest),
                "world_manifest_sha256": _sha256(connector_world_manifest),
                "view_count": connector_fusion.view_count,
                "included_view_indices": (
                    sorted(connector_view_indices)
                    if connector_view_indices is not None
                    else None
                ),
                "geometry_included": include_connector_geometry,
                "selected_observation_count": connector_fusion.counters[
                    "selected_observation_count"
                ],
                "visibility_views": connector_fusion.view_files,
            },
        },
        "ownership": {
            "priority": [
                "existing_family_kitchen",
                "accepted_living",
                "connector_gap_fill",
            ],
            "ownership_cell_m": ownership_cell_m,
            "accepted_room_authority_buffer_m": accepted_room_authority_buffer_m,
            "connector_endpoint_authority_buffer_m": (
                connector_endpoint_authority_buffer_m
            ),
            "existing_xz_columns_are_authority_candidates_only": True,
            "three_dimensional_compatibility_required_for_suppression": True,
            "existing_xz_column_coverage_suppresses_later_sources": False,
            "buffered_endpoint_room_envelopes_suppress_connector_duplicates": False,
            "connector_used_as_registration_constraint_only": (
                not include_connector_geometry
            ),
            "living": living_ownership,
            "connector": connector_ownership,
        },
        "npz_contract": {
            "owner_room_ids": {
                "1": "family",
                "2": "kitchen",
                "3": "living",
                "4": "connector_gap_fill",
            },
            "room_mask_bits": {
                "1": "family",
                "2": "kitchen",
                "4": "living",
                "8": "connector_gap_fill",
            },
            "voxel_size_m": 0.025,
            "owner_voxel_counts": owner_counts,
            "output_voxel_count": int(len(output["points"])),
            "output_observation_count": int(
                np.sum(output["observation_count"], dtype=np.uint64)
            ),
            "geometry_status_values": {
                "0": "observed_or_authoritative",
                "1": "uncertain_own_or_cross_room_free_space_contradiction",
                "2": "occluded_or_unknown_without_own_view_ray_support",
            },
        },
        "artifacts": {
            "surfels_npz": {
                "path": npz_path.name,
                "sha256": _sha256(npz_path),
                "size_bytes": npz_path.stat().st_size,
            },
            "reconstruction_glb": {
                "path": glb_path.name,
                "sha256": _sha256(glb_path),
                "size_bytes": glb_path.stat().st_size,
            },
        },
    }
    manifest_path = output_dir / "multiroom_pcf_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-npz", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--living-raw-root", type=Path, required=True)
    parser.add_argument("--living-world-manifest", type=Path, required=True)
    parser.add_argument("--connector-raw-root", type=Path, required=True)
    parser.add_argument("--connector-world-manifest", type=Path, required=True)
    parser.add_argument("--multianchor-report", type=Path, required=True)
    parser.add_argument(
        "--accepted-room-authority-buffer-m", type=float, default=0.0
    )
    parser.add_argument(
        "--connector-endpoint-authority-buffer-m", type=float, default=0.75
    )
    parser.add_argument(
        "--connector-view-indices",
        help="Comma-separated connector view indices or inclusive ranges, e.g. 23-30",
    )
    parser.add_argument(
        "--omit-connector-geometry",
        action="store_true",
        help="Use connector matches for registration but exclude its unstable points",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = extend(
        existing_npz_path=args.existing_npz,
        existing_manifest_path=args.existing_manifest,
        living_raw_root=args.living_raw_root,
        living_world_manifest=args.living_world_manifest,
        connector_raw_root=args.connector_raw_root,
        connector_world_manifest=args.connector_world_manifest,
        multianchor_report_path=args.multianchor_report,
        output_dir=args.output_dir,
        accepted_room_authority_buffer_m=args.accepted_room_authority_buffer_m,
        connector_endpoint_authority_buffer_m=(
            args.connector_endpoint_authority_buffer_m
        ),
        connector_view_indices=_parse_view_indices(args.connector_view_indices),
        include_connector_geometry=not args.omit_connector_geometry,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_voxel_count": report["npz_contract"][
                    "output_voxel_count"
                ],
                "owner_voxel_counts": report["npz_contract"][
                    "owner_voxel_counts"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
