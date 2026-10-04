"""Revision-bound proof for image-identified planes in an admitted PCF.

The phone views select support from the already admitted point GLB.  A full
normal/connected-support extraction is replayed only by the offline builder;
the presentation reader checks immutable bytes, measured plane geometry and
the exact retained image rays without importing a point-cloud fitting backend.
This proves source provenance, not a passed structural solve or physical survey.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from noesis.calibration.structural_carrier_evidence import (
    KIND as RAY_KIND, MAX_NPZ_BYTES, _alignment, _binding, _consensus,
    _file, _json, _npz, verify_carrier_image_point,
)
from noesis.calibration.structural_registration import (
    StructuralRegistrationError, _compare_replay, read_pcf_points_glb,
)
from noesis.calibration.structural_roi_planes import (
    estimate_admitted_normals, fit_supported_region_plane,
    grow_connected_plane_support, select_admitted_region_support,
)


KIND = "retained_carrier_normal_supported_pcf_roi"
MAX_REGIONS = 12
MAX_WALLS = 8
MAX_VIEWS = 16
MAX_POINTS = 4


def _fail(message: str) -> None:
    raise StructuralRegistrationError(message)


def _keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(f"{label} has missing or unknown fields")
    return value


def _close(left: Any, right: Any, label: str) -> None:
    _compare_replay(left, right, path=label)


def _review_ref_matches(reference: Mapping[str, Any], frozen: Mapping[str, Any]) -> bool:
    return (reference.get("sha256") == frozen.get("sha256")
            and reference.get("size_bytes") == frozen.get("size_bytes"))


def _plane_segment_replay(plane: Mapping[str, Any], segment: Mapping[str, Any],
                          reference_y: float) -> None:
    """Replay the shared-height line without fitting or flattening measured tilt."""
    if not plane.get("review_candidate_passed") or plane.get("reason_codes"):
        _fail("source wall was not a passed measured plane in its frozen review")
    normal = np.asarray(plane.get("normal_xyz"), dtype=float)
    offset = float(plane.get("offset_m", math.nan))
    tangent_interval = np.asarray(plane.get("tangent_interval_m"), dtype=float)
    if (normal.shape != (3,) or tangent_interval.shape != (2,)
            or not np.isfinite(normal).all() or not np.isfinite(tangent_interval).all()
            or not math.isfinite(offset) or not math.isclose(np.linalg.norm(normal), 1, abs_tol=1e-6)
            or np.linalg.norm(normal[[0, 2]]) <= .95
            or tangent_interval[1] - tangent_interval[0] < .80
            or float(plane.get("vertical_span_m", 0)) < 1.20
            or int(plane.get("voxel_support_count", 0)) < 200):
        _fail("frozen source wall lacks the measured support/height/tilt gates")
    horizontal = normal[[0, 2]] / np.linalg.norm(normal[[0, 2]])
    tangent = np.asarray((-horizontal[1], horizontal[0]))
    line_offset = -(offset + normal[1] * reference_y) / np.linalg.norm(normal[[0, 2]])
    ends = [line_offset * horizontal + extent * tangent for extent in tangent_interval]
    for key, expected in zip(("start_xz", "end_xz"), ends):
        actual = np.asarray(segment.get(key), dtype=float)
        if actual.shape != (2,) or not np.isfinite(actual).all() or not np.allclose(
                actual, expected, atol=1e-7, rtol=0):
            _fail("source wall segment does not replay from its tilted plane and observed height")
    if (segment.get("support_count") != plane["voxel_support_count"]
            or segment.get("vertical_span") != plane["vertical_span_m"]):
        _fail("source wall segment lost measured support")
    refs = segment.get("source_refs")
    if not isinstance(refs, list) or len(refs) > 8:
        _fail("source wall has invalid bounded plane refs")
    try:
        fields = dict(ref.split(":", 1) for ref in refs)
        parsed_normal = [float(v) for v in fields["normal_xyz"].split(",")]
        parsed_offset = float(fields["plane_d_m"])
        parsed_y = float(fields["reference_y_m"])
    except (ValueError, KeyError, AttributeError) as exc:
        raise StructuralRegistrationError("source wall plane refs are malformed") from exc
    if (len(fields) != len(refs) or fields.get("method") != "admitted_pcf_image_roi_plane"
            or fields.get("support_sha256") != plane["support_xyz_sha256"]
            or not np.allclose(parsed_normal, normal, atol=1e-9, rtol=0)
            or not math.isclose(parsed_offset, offset, abs_tol=1e-9)
            or parsed_y != reference_y):
        _fail("source wall plane refs do not identify the measured review")


def _load_region_view(ref: Mapping[str, Any], root: Path) -> dict[str, np.ndarray]:
    """Read only the six arrays used by the original bounded ROI selection."""
    path = _file(ref, root, maximum=MAX_NPZ_BYTES)
    # _npz validates zip size, K/pose, RGB and required geometric array shapes.
    raw = _npz(ref, root)
    try:
        with np.load(path, allow_pickle=False) as archive:
            mask = np.asarray(archive["mask"])
    except (OSError, ValueError, KeyError) as exc:
        raise StructuralRegistrationError("carrier ROI mask cannot be decoded") from exc
    if mask.shape != raw["model_rgb"].shape[:2]:
        _fail("carrier ROI mask differs from exact retained raster")
    return {**raw, "mask": mask}


def _reextract(review: Mapping[str, Any], recipe: Mapping[str, Any],
               admitted: np.ndarray, views: Mapping[str, Any],
               alignment: np.ndarray, root: Path) -> None:
    """Expensive artifact-build replay of every reviewed patch and connected plane."""
    regions = recipe["regions"]
    support: dict[str, list[np.ndarray]] = {}
    for observed, frozen in zip(regions, review["regions"]):
        if (observed["surface"] != frozen.get("surface")
                or observed["view_index"] != frozen.get("view_index")
                or observed["box_xyxy"] != frozen.get("box_xyxy")):
            _fail("frozen ROI differs from its image selection recipe")
        ref = views[f'{observed["view_index"]:04d}']
        if not _review_ref_matches(ref, frozen["view"]):
            _fail("reviewed ROI view bytes differ from frozen extraction")
        raw = _load_region_view(ref, root)
        x0, y0, x1, y1 = observed["box_xyxy"]
        height, width = raw["model_rgb"].shape[:2]
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            _fail("reviewed ROI lies outside exact model raster")
        roi = np.s_[y0:y1, x0:x1]
        xyz = raw["world_points"][roi].reshape(-1, 3)
        good = ((raw["confidence"][roi].reshape(-1) >= .70)
                & raw["mask"][roi].reshape(-1).astype(bool)
                & (raw["source_selection"][roi].reshape(-1) == 1)
                & ~raw["cross_model_uncertain"][roi].reshape(-1).astype(bool)
                & np.isfinite(xyz).all(axis=1))
        observed_points = xyz[good] @ alignment[:3, :3].T + alignment[:3, 3]
        indices = (select_admitted_region_support(
            admitted, observed_points, maximum_distance_m=recipe["association_distance_m"])
            if len(observed_points) >= 3 else np.empty(0, dtype=int))
        if (len(observed_points) != frozen["quality_point_count"]
                or len(indices) != frozen["selected_admitted_count"]):
            _fail("reviewed ROI quality or admitted support count changed")
        if len(indices) >= 3:
            _close(frozen["plane"], fit_supported_region_plane(admitted[indices]), "ROI plane")
            support.setdefault(observed["surface"], []).append(indices)
        elif frozen["plane"].get("extraction_support_passed") is not False:
            _fail("unsupported ROI gained a measured plane")
    if not support:
        _fail("review has no admitted wall support")
    pooled = np.unique(np.concatenate([indices for rows in support.values() for indices in rows]))
    reference_y = float(np.median(admitted[pooled, 1]))
    _close(review["common_observed_reference_y_m"], reference_y, "observed reference height")
    normals = estimate_admitted_normals(admitted)
    frozen_surfaces = {row["surface"]: row for row in review["surfaces"]}
    if set(frozen_surfaces) != set(support):
        _fail("reviewed surfaces differ from exact region extraction")
    for surface, rows in support.items():
        seeds = np.unique(np.concatenate(rows))
        frozen = frozen_surfaces[surface]
        plane = fit_supported_region_plane(admitted[seeds])
        _close(frozen["plane"], plane, f"{surface} pooled plane")
        indices = grow_connected_plane_support(admitted, seeds, plane,
                                               local_normals=normals)
        if len(indices) != frozen.get("connected_support_count"):
            _fail(f"{surface} connected PCF support count changed")
        if len(indices) >= 3:
            _close(frozen["connected_plane"], fit_supported_region_plane(admitted[indices]),
                   f"{surface} connected plane")
        if frozen.get("connected_segment") is not None:
            _plane_segment_replay(frozen["connected_plane"], frozen["connected_segment"], reference_y)


def verify_source_feature_evidence(candidate: Mapping[str, Any], points_path: str | Path,
                                   evidence: Mapping[str, Any], root: Path, *, offline: bool,
                                   binding: Mapping[str, Any]) -> None:
    """Verify a bounded source proof; replay GLB extraction only when offline."""
    _keys(evidence, {"kind", "capture_id", "target_world_revision",
                     "target_transform_sha256", "target_calibration_sha256",
                     "admitted_prior", "source_bundle_manifest", "source_manifest",
                     "consensus_manifest", "carrier_surfels", "review", "recipe",
                     "source_feature_review", "views", "image_file", "walls",
                     "point_evidence"}, "normal-supported source feature evidence")
    if evidence["kind"] != KIND:
        _fail("unsupported normal-supported source proof")
    _binding(evidence, binding)
    if (binding.get("world_frame", {}).get("frame_id") != "backend_world_m"
            or binding.get("scene_prior_id") != evidence["target_world_revision"]):
        _fail("normal-supported proof lacks the exact backend-world prior binding")
    root = Path(root)
    points_path = Path(points_path)
    prior = _json(evidence["admitted_prior"], root)
    bundle = _json(evidence["source_bundle_manifest"], root)
    source = _json(evidence["source_manifest"], root)
    consensus = _json(evidence["consensus_manifest"], root)
    review = _json(evidence["review"], root)
    recipe = _json(evidence["recipe"], root)
    feature_review = _json(evidence["source_feature_review"], root)
    views = evidence["views"]
    if (not isinstance(views, Mapping) or not 1 <= len(views) <= MAX_VIEWS
            or any(type(key) is not str or len(key) != 4 or not key.isdecimal() for key in views)):
        _fail("normal-supported proof has an invalid bounded view set")
    _consensus(consensus, views)
    alignment = _alignment(source)
    capture = evidence["capture_id"]
    if (not isinstance(capture, str) or not 1 <= len(capture) <= 160
            or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in capture)
            or capture not in source.get("source", "")):
        _fail("source capture is not retained in the phone-only manifest")
    artifacts = [row for row in prior.get("artifacts", []) if row.get("role") == "points_glb"]
    source_rows = [row for row in bundle.get("files", [])
                   if row.get("path") == "evaluation/static_world_point_cloud_manifest.json"]
    if (prior.get("prior_id") != evidence["target_world_revision"]
            or prior.get("source", {}).get("capture_id") != capture
            or prior.get("source", {}).get("bundle_manifest_sha256") != evidence["source_bundle_manifest"]["sha256"]
            or bundle.get("schema") != "noesis.reference.room_scan_bundle_manifest.v1"
            or bundle.get("bundle_id") != prior.get("source", {}).get("bundle_id")
            or len(source_rows) != 1 or not _review_ref_matches(source_rows[0], evidence["source_manifest"])
            or len(artifacts) != 1 or artifacts[0].get("relative_path") != points_path.name
            or not points_path.is_file() or points_path.is_symlink()
            or points_path.parent != root / Path(evidence["admitted_prior"]["path"]).parent
            or points_path.stat().st_size != artifacts[0].get("size_bytes")
            or _file({"path": points_path.relative_to(root).as_posix(),
                      "sha256": artifacts[0]["sha256"],
                      "size_bytes": artifacts[0]["size_bytes"]}, root, maximum=64*1024*1024) != points_path):
        _fail("source proof does not bind the exact admitted PCF revision")
    if (review.get("schema") != "noesis.review.pcf_image_wall_planes.v1"
            or review.get("status") != "review_only"
            or review.get("runtime_changed") is not False
            or review.get("structural_mapping_admitted") is not False
            or not _review_ref_matches(review.get("prior", {}), evidence["admitted_prior"])
            or review.get("pcf", {}).get("sha256") != artifacts[0]["sha256"]
            or review.get("pcf", {}).get("size_bytes") != artifacts[0]["size_bytes"]
            or not _review_ref_matches(review.get("source_bundle", {}), evidence["source_bundle_manifest"])
            or not _review_ref_matches(review.get("source_manifest", {}), evidence["source_manifest"])
            or not _review_ref_matches(review.get("carrier_consensus", {}), evidence["consensus_manifest"])
            or not _review_ref_matches(review.get("carrier_surfels", {}), evidence["carrier_surfels"])
            or not _review_ref_matches(review.get("recipe", {}), evidence["recipe"])
            or review.get("alignment") != source.get("alignment")
            or recipe.get("schema") != "noesis.review.pcf_image_wall_regions.v1"):
        _fail("normal-supported review does not replay the admitted source identity")
    regions = recipe.get("regions")
    if (not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS
            or len(review.get("regions", [])) != len(regions)
            or type(recipe.get("association_distance_m")) not in (float, int)
            or not 0 < recipe["association_distance_m"] <= .18):
        _fail("normal-supported region recipe is unbounded or invalid")
    region_views = {f'{row["view_index"]:04d}' for row in regions}
    feature_views = {f'{row["view_index"]:04d}' for row in feature_review.get("holdouts", [])}
    if len(feature_views) != 1:
        _fail("one exact retained raster must carry this bounded heldout point set")
    heldout_view = next(iter(feature_views))
    if set(views) != region_views | feature_views:
        _fail("normal-supported view inventory omits or adds source rasters")
    for index, frozen in enumerate(review["regions"]):
        if not _review_ref_matches(frozen.get("view", {}), views[f'{regions[index]["view_index"]:04d}']):
            _fail("normal-supported region view digest changed")
    for ref in views.values():
        _file(ref, root, maximum=MAX_NPZ_BYTES)
    _file(evidence["carrier_surfels"], root, maximum=16*1024*1024)
    _file(evidence["image_file"], root, maximum=32*1024*1024)
    feature_schema = feature_review.get("schema")
    feature_source = feature_review.get("source", {})
    if (not isinstance(feature_schema, str)
            or not feature_schema.startswith("noesis.review.")
            or not feature_schema.endswith("_source_structural_holdouts.v1")
            or feature_review.get("source", {}).get("review_sha256") != evidence["review"]["sha256"]
            or feature_source.get(f"view_{heldout_view}_npz_sha256") != views[heldout_view]["sha256"]
            or feature_source.get(f"view_{heldout_view}_png_sha256") != evidence["image_file"]["sha256"]
            or feature_review.get("frozen_geometry", {}).get("source_to_backend_alignment") is None):
        _fail("heldout image review is not bound to the measured source review")
    # The review records the actual phone K/pose. They remain image geometry,
    # never horizontal registration controls or live camera calibration.
    raw = _npz(views[heldout_view], root, image_only=True)
    frozen_geometry = feature_review["frozen_geometry"]
    if (not np.array_equal(raw["intrinsics"], np.asarray(frozen_geometry[f"view_{heldout_view}_intrinsics"]))
            or not np.array_equal(raw["camera_pose"], np.asarray(frozen_geometry[f"view_{heldout_view}_camera_pose"]))):
        _fail("heldout K/pose differ from exact retained view")
    surfaces = review.get("surfaces")
    walls = evidence["walls"]
    if (not isinstance(surfaces, list) or not isinstance(walls, list)
            or not 1 <= len(walls) <= MAX_WALLS):
        _fail("normal-supported wall proof is missing or unbounded")
    reviewed = {row["surface"]: row for row in surfaces}
    candidate_walls = {row["segment_id"]: row for row in candidate["inputs"]["source_segments"]}
    if (len(reviewed) != len(surfaces) or len(candidate_walls) != len(walls)
            or {row.get("segment_id") for row in walls} != set(candidate_walls)):
        _fail("normal-supported proof does not cover exactly the candidate walls")
    reference_y = float(review["common_observed_reference_y_m"])
    if not math.isfinite(reference_y):
        _fail("normal-supported observed height is not finite")
    fit_by_id: dict[str, Mapping[str, Any]] = {}
    for frozen in walls:
        _keys(frozen, {"segment_id", "surface", "plane", "segment"}, "normal-supported wall")
        surface = reviewed.get(frozen["surface"])
        if (surface is None or surface.get("connected_plane") != frozen["plane"]
                or surface.get("connected_segment") != frozen["segment"]
                or frozen["segment_id"] != frozen["surface"]):
            _fail("normal-supported candidate wall differs from its frozen review")
        _close(candidate_walls[frozen["segment_id"]], frozen["segment"], "source wall")
        _plane_segment_replay(frozen["plane"], frozen["segment"], reference_y)
        fit_by_id[frozen["segment_id"]] = frozen["plane"]
    points = candidate["inputs"]["source_points"]
    point_evidence = evidence["point_evidence"]
    holdouts = {row["id"]: row for row in feature_review.get("holdouts", [])}
    if (not isinstance(points, list) or not isinstance(point_evidence, list)
            or not 1 <= len(points) <= MAX_POINTS or len(points) != len(point_evidence)
            or len(holdouts) != len(points)):
        _fail("normal-supported proof does not cover exactly the reviewed points")
    point_matches = candidate["inputs"]["point_correspondences"]
    if (len(point_matches) != len(points)
            or {row["source_id"] for row in point_matches} != {row["point_id"] for row in points}
            or any(row["role"] != "heldout" for row in point_matches)):
        _fail("reviewed source opening points must remain independent holdouts")
    for frozen, point in zip(point_evidence, points):
        wall_id = frozen.get("segment_id")
        held = holdouts.get(point["point_id"])
        plane = fit_by_id.get(wall_id)
        if (frozen.get("kind") != RAY_KIND or frozen.get("point") != point
                or plane is None or point["source_wall_ids"] != [wall_id]
                or held is None or frozen.get("pixel_uv") != held["pixel_xy"]
                or frozen.get("identity_review") != held["physical_identity"]
                or frozen.get("pixel_bounds") != [held["pixel_bounds_xyxy_inclusive"][:2],
                                                       held["pixel_bounds_xyxy_inclusive"][2:]]
                or frozen.get("view_index") != held["view_index"]
                or frozen.get("image_file") != evidence["image_file"]
                or frozen.get("projection") != {
                    "view": views[f'{held["view_index"]:04d}'],
                    "source_manifest": evidence["source_manifest"],
                    "normal_xyz": plane["normal_xyz"],
                    "offset_m": plane["offset_m"],
                    "reference_y_m": reference_y,
                }):
            _fail("normal-supported point differs from heldout raster/PCF wall")
        _close(point["xz"], held["reference_y_xz_m"], "heldout ray XZ")
        held_view = views[f'{held["view_index"]:04d}']
        if point.get("evidence_refs") != [
            f'image_sha256:{evidence["image_file"]["sha256"]}',
            f'pixel_uv:{held["pixel_xy"][0]},{held["pixel_xy"][1]}',
            f'view_npz_sha256:{held_view["sha256"]}',
            f'wall_review_sha256:{evidence["review"]["sha256"]}',
        ]:
            _fail("source opening evidence refs differ from exact raster/review bytes")
        if (frozen.get("target_world_revision") != evidence["target_world_revision"]
                or frozen.get("target_transform_sha256") != evidence["target_transform_sha256"]
                or frozen.get("target_calibration_sha256") != evidence["target_calibration_sha256"]):
            _fail("heldout ray lost its exact target binding")
        verify_carrier_image_point(frozen, point, binding, root)
    if offline:
        admitted = read_pcf_points_glb(points_path)
        from scipy.spatial import cKDTree
        surfel_path = _file(evidence["carrier_surfels"], root, maximum=16*1024*1024)
        with np.load(surfel_path, allow_pickle=False) as archive:
            surfels = np.asarray(archive["points"])
        if surfels.shape != (source.get("source_point_count"), 3) or not np.isfinite(surfels).all():
            _fail("retained carrier surfel source count or geometry changed")
        aligned = surfels @ alignment[:3, :3].T + alignment[:3, 3]
        distance, _ = cKDTree(aligned).query(admitted, workers=1)
        if float(distance.max()) > 2e-5:
            _fail("admitted PCF does not replay from the retained carrier source")
        _reextract(review, recipe, admitted, views, alignment, root)
