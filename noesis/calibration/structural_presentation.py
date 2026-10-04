"""Revision-bound structural world-to-scene presentation, not localization.

The expensive feature extraction/registration is offline. Startup verifies the
exact geometry bytes and frozen mathematical solve; bundle publication only
checks the already admitted camera/frame/support identities. Camera-device
positions, people and phone optical-center paths are not inputs to this fit.
"""
from __future__ import annotations

import json
import itertools
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from noesis.calibration.presentation_registration import (
    PresentationRegistrationError, _canonical_sha256, _exact_keys, _matrix,
    _read_json, _sha, sha256_file,
)
from noesis.validation.authored_scene import AuthoredSceneGeometry
from noesis_core.contracts.scene_prior import (
    ScenePriorCatalog, ScenePriorFrameRef, ScenePriorRevision, ScenePriorWorldToScenePresentation,
    scene_revision_sha256, world_to_scene_presentation_sha256,
)
from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.support_surface import SupportSurfaceSet


CONTRACT = "noesis.scene_presentation.structural_registration"
OWNER_SELECTED_CONTRACT = "noesis.scene_presentation.owner_selected_structural_registration"
CONTRACT_VERSION = 1
MAX_BYTES = 512 * 1024
MAX_ENTRIES = 8
MAX_IMAGE_POINTS = 4
REPO_ROOT = Path(__file__).resolve().parents[2]


def _relative_file(path: str | Path, root: Path) -> dict[str, Any]:
    source = Path(path).absolute()
    relative = source.relative_to(root.absolute()).as_posix()
    if source.is_symlink() or not source.is_file():
        raise PresentationRegistrationError("structural input is not a regular file")
    return {"path": relative, "sha256": sha256_file(source), "size_bytes": source.stat().st_size}


def _verified_file(ref: Any, root: Path, *, maximum: int) -> Path:
    if not isinstance(ref, Mapping):
        raise PresentationRegistrationError("structural evidence reference is missing")
    _exact_keys(ref, {"path", "sha256", "size_bytes"}, label="structural evidence file")
    relative = ref["path"]
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or relative.startswith("/") or "\x00" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise PresentationRegistrationError("structural evidence path must be portable and root-relative")
    _sha(ref["sha256"], label="structural evidence SHA")
    size = ref["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= maximum:
        raise PresentationRegistrationError("structural evidence file exceeds its bound")
    source = root / PurePosixPath(relative)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PresentationRegistrationError("structural evidence path contains a symlink")
    if not source.is_file() or source.stat().st_size != size or sha256_file(source) != ref["sha256"]:
        raise PresentationRegistrationError("structural evidence bytes changed")
    return source


def _floor_y(geometry: AuthoredSceneGeometry, groups: list[str]) -> float:
    values = [geometry.vertices[i][1] for item in geometry.walkable_triangles
              if item.authority_group in groups for i in item.triangle.vertex_indices]
    if not values or max(values) - min(values) > 1e-3:
        raise PresentationRegistrationError("selected authored room has no single horizontal floor")
    return float(np.mean(values))


def _floor_plane(binding: Mapping[str, Any]) -> dict[str, Any]:
    floor = binding.get("world_floor_plane")
    if not isinstance(floor, Mapping) or set(floor) != {"normal", "offset_m"}:
        raise PresentationRegistrationError("bound PCF floor plane is missing")
    normal = np.asarray(floor["normal"], dtype=float)
    offset = floor["offset_m"]
    if (normal.shape != (3,) or not np.allclose(normal, (0, 1, 0), atol=1e-6, rtol=0)
            or isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset)):
        raise PresentationRegistrationError("structural presentation requires an admitted gravity-aligned floor")
    return {"normal": [0.0, 1.0, 0.0], "offset_m": float(offset)}


def _verify_capture(
    evidence: Mapping[str, Any], *, bundle: Mapping[str, Any], camera: str, root: Path,
) -> None:
    """Bind reviewed pixels to their retained rectified raster and camera rays.

    Only the compact manifest/calibration/metadata are read, not depth or video.
    This does not admit the retained reconstruction as live localization.
    """
    capture = evidence["capture"]
    _exact_keys(capture, {"manifest", "calibration", "metadata"}, label="structural image capture")
    manifest_path = _verified_file(capture["manifest"], root, maximum=1024 * 1024)
    calibration_path = _verified_file(capture["calibration"], root, maximum=256 * 1024)
    metadata_path = _verified_file(capture["metadata"], root, maximum=256 * 1024)
    manifest, calibration, metadata = (_read_json(p) for p in (manifest_path, calibration_path, metadata_path))
    if (manifest.get("schema") != "noesis.paired_static_reference.v1"
            or calibration.get("schema") != "noesis.paired_static_reference.calibration.v1"
            or metadata.get("schema") != "noesis.room_reconstruction.stream_points.v4"
            or manifest.get("reference_kind") != "paired_static"
            or calibration.get("E_semantics") != "camera_from_calibration_frame_raw"):
        raise PresentationRegistrationError("unsupported retained rectified image binding")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) > 128:
        raise PresentationRegistrationError("retained image manifest files are missing or unbounded")
    for ref in (evidence["image_file"], capture["calibration"], capture["metadata"]):
        actual = root / ref["path"]
        try:
            relative = actual.relative_to(manifest_path.parent).as_posix()
        except ValueError as exc:
            raise PresentationRegistrationError("image evidence is outside its retained capture") from exc
        if {"path": relative, "sha256": ref["sha256"], "size_bytes": ref["size_bytes"]} not in files:
            raise PresentationRegistrationError("retained manifest does not bind the image evidence bytes")
    projection = _projection_evidence(bundle, camera)
    binding = projection["frame_binding"]
    if (manifest.get("camera_frame_binding") != binding
            or calibration.get("frame_binding") != binding
            or metadata.get("camera_frame_binding") != binding
            or calibration.get("source_calibration_sha256") != binding["camera_calibration_sha256"]
            or calibration.get("cameras", {}).get(camera) != {"K": projection["K"], "E": projection["E"]}):
        raise PresentationRegistrationError("retained image rays no longer match the admitted calibration/PCF")
    k = projection["K"]
    expected_k = [[k[0], 0.0, k[2]], [0.0, k[1], k[3]], [0.0, 0.0, 1.0]]
    measured = evidence["image_evidence"]
    if (metadata.get("image_size") != [measured["image"]["width"], measured["image"]["height"]]
            or metadata.get("rgb_keyframes", {}).get(camera) != Path(evidence["image_file"]["path"]).name
            or metadata.get("keyframe_source") != "pixelwise_median_of_six_captured_rectified_observations"
            or metadata.get("intrinsics") != expected_k
            or metadata.get("extrinsics_col_major") != projection["E"]):
        raise PresentationRegistrationError("reviewed image is not in its captured rectified pixel space")


def _verify_image_points(
    *, candidate: Mapping[str, Any], bundle: Mapping[str, Any], camera: str,
    image_evidence: list[dict[str, Any]], root: Path,
    support: SupportSurfaceSet | None,
) -> None:
    from noesis.calibration.structural_image_evidence import verify_structural_image_point
    inputs = candidate["inputs"]
    if not isinstance(image_evidence, list) or len(image_evidence) != len(inputs["source_points"]):
        raise PresentationRegistrationError("every source point requires exact measured image evidence")
    walls = {s["segment_id"]: s for s in inputs["source_segments"]}
    floor_y = -_floor_plane(bundle["cameras"]["frame_bindings"][camera])["offset_m"]
    for point, evidence in zip(inputs["source_points"], image_evidence):
        if evidence.get("kind") == "retained_carrier_pcf_roi_plane":
            from noesis.calibration.structural_carrier_evidence import verify_carrier_image_point
            verify_carrier_image_point(evidence, point, bundle["cameras"]["frame_bindings"][camera], root)
            continue
        _exact_keys(evidence, {"point", "image_evidence", "image_file", "capture", "identity_review", "pixel_bounds"}, label="structural image evidence")
        if evidence["point"] != point or not isinstance(evidence["identity_review"], str) or not evidence["identity_review"].strip():
            raise PresentationRegistrationError("source point differs from reviewed image identity")
        image_path = _verified_file(evidence["image_file"], root, maximum=32 * 1024 * 1024)
        _verify_capture(evidence, bundle=bundle, camera=camera, root=root)
        measured = evidence["image_evidence"]
        plane = measured["wall_plane"]
        height = measured["surface_y_m"]
        if height is not None and abs(float(height) - floor_y) > 1e-6:
            if support is None or not any(
                np.allclose(s.normal_world, (0, 1, 0), atol=1e-6, rtol=0)
                and abs(float(height) + s.offset_m) <= 1e-6 for s in support.surfaces
            ):
                raise PresentationRegistrationError("image floor contact uses an unmeasured support height")
        verify_structural_image_point(
            {"point": point, "image_evidence": measured}, bundle=bundle, camera_id=camera,
            image_path=image_path,
            image_size=[measured["image"]["width"], measured["image"]["height"]],
            pixel=measured["pixel_uv"], kind=point["kind"], point_id=point["point_id"],
            plane_segment=walls[plane["segment_id"]] if plane else None, surface_y_m=height,
        )


def _annotation_stability(*, candidate: Mapping[str, Any], bundle: Mapping[str, Any],
                          camera: str, image_evidence: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Re-solve all bounded combinations of reviewed raster intervals.

    The bounds are site evidence, not a default room correction. Half a pixel
    is the raster quantization floor; rough annotations must declare their
    wider observed ambiguity. Corners plus centres are a finite stress check,
    not a proof of physical feature identity or all nonlinear interior extrema.
    """
    from noesis.calibration.structural_image_evidence import project_structural_image_point
    from noesis.calibration.structural_registration import solve_structural_registration
    if len(image_evidence) > MAX_IMAGE_POINTS:
        raise PresentationRegistrationError("annotation stability requires at most four independently reviewed image points")
    inputs = candidate["inputs"]
    walls = {s["segment_id"]: s for s in inputs["source_segments"]}
    variants = []
    for evidence in image_evidence:
        measured = evidence["image_evidence"]
        centre = np.asarray(measured["pixel_uv"], dtype=float)
        bounds = np.asarray(evidence["pixel_bounds"], dtype=float)
        size = np.asarray([measured["image"]["width"], measured["image"]["height"]])
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[0] < 0) or np.any(bounds[1] >= size)
                or np.any(centre - bounds[0] < 0.5) or np.any(bounds[1] - centre < 0.5)
                or np.any(bounds[1] - bounds[0] > 64)):
            raise PresentationRegistrationError("annotation bounds must enclose its pixel quantization and reviewed local ambiguity")
        image_path = root / evidence["image_file"]["path"]
        plane = measured["wall_plane"]
        choices = [(evidence["point"], centre.tolist())]
        for u, v in itertools.product(bounds[:, 0], bounds[:, 1]):
            pixel = [float(u), float(v)]
            if evidence.get("kind") == "retained_carrier_pcf_roi_plane":
                from noesis.calibration.structural_carrier_evidence import project_carrier_image_point
                varied_point = project_carrier_image_point(evidence, pixel, evidence["point"], root)
            else:
                varied_point = project_structural_image_point(bundle=bundle, camera_id=camera,
                    image_path=image_path, image_size=size.tolist(), pixel=pixel,
                    kind=evidence["point"]["kind"], point_id=evidence["point"]["point_id"],
                    plane_segment=walls[plane["segment_id"]] if plane else None,
                    surface_y_m=measured["surface_y_m"])["point"]
            choices.append((varied_point, pixel))
        variants.append(choices)
    result = {"method": "reviewed_raster_intervals_joint_corners_and_centres_v1",
              "passed": True, "combinations_checked": 0, "failed_case": None,
              "max_point_error_m": 0.0, "max_wall_perpendicular_m": 0.0,
              "max_wall_angle_deg": 0.0}
    for choices in itertools.product(*variants):
        varied = {**inputs, "source_points": [point for point, _ in choices]}
        replayed = solve_structural_registration(**varied)
        result["combinations_checked"] += 1
        for measurement in (replayed["solution"] or {}).get("measurements", []):
            if measurement["kind"] == "point":
                result["max_point_error_m"] = max(result["max_point_error_m"], measurement["error_m"])
            else:
                result["max_wall_perpendicular_m"] = max(result["max_wall_perpendicular_m"], measurement["perpendicular_m"])
                result["max_wall_angle_deg"] = max(result["max_wall_angle_deg"], measurement["angle_deg"])
        if replayed["solver_status"] != "accepted":
            result["passed"] = False
            result["failed_case"] = {"pixels": {point["point_id"]: pixel for point, pixel in choices},
                                     "reason_codes": replayed["reason_codes"]}
            break
    return result


def _owner_selection(value: Any, cameras: set[str]) -> dict[str, Any]:
    """Explicit trusted-owner configuration, never a passed-quality claim."""
    if not isinstance(value, Mapping):
        raise PresentationRegistrationError("owner-selected fit requires explicit owner authorization")
    _exact_keys(value, {"selection_id", "authorized_by", "authorization", "reason",
                       "selected_matrix_sha256s", "structurally_qualified"}, label="owner selection")
    if value["authorized_by"] != "owner" or value["structurally_qualified"] is not False:
        raise PresentationRegistrationError("owner selection cannot claim structural qualification")
    for key, limit in (("selection_id", 160), ("authorization", 2000), ("reason", 2000)):
        if not isinstance(value[key], str) or not 1 <= len(value[key].strip()) <= limit:
            raise PresentationRegistrationError("owner selection authorization is empty or unbounded")
    selected = value["selected_matrix_sha256s"]
    if not isinstance(selected, Mapping) or set(selected) != cameras:
        raise PresentationRegistrationError("owner selection must name the exact selected cameras")
    for digest in selected.values():
        _sha(digest, label="owner-selected matrix SHA")
    return dict(value)


def _selection_quality_only(stability: Mapping[str, Any]) -> None:
    if not stability["passed"] and not set(stability["failed_case"]["reason_codes"]).issubset(
        {"heldout_wall_residual", "heldout_point_residual"}
    ):
        raise PresentationRegistrationError("owner selection cannot bypass annotation structural validity failures")


def _matrix_for_floor(candidate: dict[str, Any], world_floor: dict[str, Any], authored_y: float,
                      *, owner_selection: Mapping[str, Any] | None = None, camera: str | None = None) -> np.ndarray:
    from noesis.calibration.structural_registration import validate_solve_result
    verified = validate_solve_result(candidate)
    if owner_selection is not None:
        selected = owner_selection["selected_matrix_sha256s"].get(camera)
        if verified["solution"] is None or selected != _canonical_sha256({"matrix": verified["solution"]["matrix_col_major"]}):
            raise PresentationRegistrationError("solve differs from the explicitly owner-selected matrix")
        # Source identity, rank, fit support, proper metric geometry and solver
        # failures remain fatal. Only the known independent QUALITY rejection
        # can be explicitly selected for interim presentation by the owner.
        if not set(verified["reason_codes"]).issubset({"heldout_wall_residual", "heldout_point_residual"}):
            raise PresentationRegistrationError("owner selection cannot bypass structural validity failures")
    elif verified["solver_status"] != "accepted":
        raise PresentationRegistrationError("structural registration has not passed its geometry gates")
    solution = verified["solution"]
    result = _matrix(solution["matrix_col_major"], label="structural solve matrix")
    scale = float(solution["scene_per_m"])
    if not math.isfinite(scale) or scale <= 0:
        raise PresentationRegistrationError("structural registration has invalid scale")
    if (not np.allclose(np.linalg.svd(result[:3, :3], compute_uv=False), scale, atol=scale * 1e-6, rtol=0)
            or np.linalg.det(result[:3, :3]) <= 0
            or not np.allclose(result[1, :3], (0, scale, 0), atol=1e-6, rtol=0)
            or abs(result[0, 1]) > 1e-6 or abs(result[2, 1]) > 1e-6):
        raise PresentationRegistrationError("structural solve is not a proper gravity-locked uniform similarity")
    result[1, 3] = authored_y + scale * world_floor["offset_m"]
    return result


def _verify_geometry_features(
    *, candidate: Mapping[str, Any], points_path: Path,
    geometry: AuthoredSceneGeometry, groups: list[str],
    bundle: Mapping[str, Any], camera: str,
    image_evidence: list[dict[str, Any]], authored_selectors: list[dict[str, Any]],
    root: Path, support: SupportSurfaceSet | None,
    source_feature_evidence: Mapping[str, Any] | None = None,
    authored_wall_selectors: list[dict[str, Any]] | None = None,
) -> None:
    """Offline proof of feature provenance, separate from optimization replay."""
    from noesis.calibration.authored_structures import extract_authored_structures, selected_mesh_wall_face
    from noesis.calibration.structural_registration import (
        verify_authored_segments, verify_pcf_source_segments,
    )
    inputs = candidate["inputs"]
    if source_feature_evidence is None:
        if any(row.get("kind") == "retained_carrier_pcf_roi_plane" for row in image_evidence):
            raise PresentationRegistrationError("carrier image rays require their measured PCF feature proof")
        verify_pcf_source_segments(points_path, inputs["source_segments"])
    else:
        if image_evidence != source_feature_evidence.get("point_evidence"):
            raise PresentationRegistrationError("carrier image evidence differs from the verified source feature proof")
        _verify_source_features(candidate, points_path, source_feature_evidence, root, offline=True,
                                binding=bundle["cameras"]["frame_bindings"][camera])
    extracted = extract_authored_structures(geometry, groups,
        initial_scene_units_per_meter=inputs["scene_per_m_hint"])
    mesh_walls = [selected_mesh_wall_face(geometry, row) for row in (authored_wall_selectors or [])]
    mesh_ids = {row["segment_id"] for row in mesh_walls}
    if len(mesh_ids) != len(mesh_walls) or not mesh_ids.issubset({row["segment_id"] for row in inputs["authored_segments"]}):
        raise PresentationRegistrationError("mesh wall selectors are duplicate or not used by the solve")
    from noesis.calibration.structural_registration import _compare_replay, _segments
    for wall in mesh_walls:
        _compare_replay(next(row for row in inputs["authored_segments"] if row["segment_id"] == wall["segment_id"]),
                        _segments([wall], label="authored_segments", minimum_length=0.4 * inputs["scene_per_m_hint"])[0], path="selected mesh wall")
    verify_authored_segments(extracted.wall_segments, [row for row in inputs["authored_segments"] if row["segment_id"] not in mesh_ids],
                            scene_per_m_hint=inputs["scene_per_m_hint"])
    if len(authored_selectors) != len(inputs["authored_points"]):
        raise PresentationRegistrationError("every authored point requires an exact structural selector")
    _verify_image_points(candidate=candidate, bundle=bundle, camera=camera,
                         image_evidence=image_evidence, root=root, support=support)
    openings = {o.opening_id: o for o in extracted.openings}
    authored_walls = {s.segment_id: s for s in extracted.wall_segments}
    point_matches = {m["authored_id"]: m["source_id"] for m in inputs["point_correspondences"]}
    source_points = {p["point_id"]: p for p in inputs["source_points"]}
    matched_wall_planes = {m["source_id"]: authored_walls[m["authored_id"]].plane_id
                           for m in inputs["correspondences"] if m["authored_id"] in authored_walls}
    floor_vertices = {i for t in geometry.walkable_triangles
                      if t.authority_group in groups for i in t.triangle.vertex_indices}
    wall_triangle_ids = {int(ref.rsplit("#triangle:", 1)[1])
                         for wall in extracted.wall_segments for ref in wall.source_refs}
    structural_vertices = floor_vertices | {i for t in geometry.triangles
        if t.index in wall_triangle_ids for i in t.vertex_indices}
    floor_y = _floor_y(geometry, groups)
    for point, selector in zip(inputs["authored_points"], authored_selectors):
        if selector["point_id"] != point["point_id"] or selector["kind"] != point["kind"]:
            raise PresentationRegistrationError("authored point identity differs from its selector")
        if selector.get("selector_type") == "wall_vertex":
            _exact_keys(selector, {"point_id", "kind", "selector_type", "segment_id", "wall_group", "vertex_index", "identity_review"}, label="authored wall vertex")
            wall = authored_walls.get(selector["segment_id"])
            index = selector["vertex_index"]
            source_point = source_points.get(point_matches.get(point["point_id"]))
            if (wall is None or type(index) is not int or not 1 <= index <= len(geometry.vertices)
                    or source_point is None or not any(matched_wall_planes.get(s) == wall.plane_id for s in source_point["source_wall_ids"])
                    or not any(selector["wall_group"] in t.groups and index - 1 in t.vertex_indices for t in geometry.triangles)
                    or not any(ref.startswith(selector["wall_group"] + "#") for ref in wall.source_refs)):
                raise PresentationRegistrationError("authored wall vertex is not bound to its matched architectural wall")
            vertex = geometry.vertices[index - 1]
            expected = (vertex[0], vertex[2])
            tangent = np.asarray(wall.end_xz) - wall.start_xz
            line_normal = np.asarray((-tangent[1], tangent[0])) / np.linalg.norm(tangent)
            if abs(line_normal @ (np.asarray(expected) - wall.start_xz)) > 1e-3 * inputs["scene_per_m_hint"]:
                raise PresentationRegistrationError("authored vertex is outside its matched wall plane")
        elif selector.get("selector_type") == "wall_segment_endpoint":
            _exact_keys(selector, {"point_id", "kind", "selector_type", "segment_id", "endpoint", "identity_review"},
                        label="authored wall endpoint")
            wall = authored_walls.get(selector["segment_id"])
            if wall is None or selector["endpoint"] not in ("start", "end"):
                raise PresentationRegistrationError("authored wall endpoint selector is unavailable")
            source_point = source_points.get(point_matches.get(point["point_id"]))
            if (source_point is None or not any(matched_wall_planes.get(source_wall) == wall.plane_id
                                               for source_wall in source_point["source_wall_ids"])):
                raise PresentationRegistrationError("reviewed wall endpoint does not belong to its corresponding measured wall")
            expected = wall.start_xz if selector["endpoint"] == "start" else wall.end_xz
        elif selector["kind"] == "opening":
            _exact_keys(selector, {"point_id", "kind", "opening_id", "endpoint", "identity_review"}, label="authored opening")
            opening = openings.get(selector["opening_id"])
            if opening is None or selector["endpoint"] not in ("start", "end"):
                raise PresentationRegistrationError("authored opening selector is unavailable")
            source_point = source_points.get(point_matches.get(point["point_id"]))
            if (source_point is None or not any(matched_wall_planes.get(wall) == opening.plane_id
                                               for wall in source_point["source_wall_ids"])):
                raise PresentationRegistrationError("reviewed opening does not belong to its corresponding measured wall")
            expected = opening.start_xz if selector["endpoint"] == "start" else opening.end_xz
        else:
            _exact_keys(selector, {"point_id", "kind", "vertex_index", "identity_review"}, label="authored corner")
            index = selector["vertex_index"]
            if type(index) is not int or index - 1 not in structural_vertices:
                raise PresentationRegistrationError("authored corner is not on the selected structural floor")
            vertex = geometry.vertices[index - 1]
            if abs(vertex[1] - floor_y) > 1e-3:
                raise PresentationRegistrationError("raised authored corner is not a lower-floor contact")
            expected = (vertex[0], vertex[2])
        if not isinstance(selector["identity_review"], str) or not selector["identity_review"].strip():
            raise PresentationRegistrationError("authored structural identity has not been reviewed")
        if not np.allclose(expected, point["xz"], atol=1e-8, rtol=0):
            raise PresentationRegistrationError("authored point differs from measured OBJ structure")


def _verify_source_features(candidate, points_path, evidence, root, *, offline, binding):
    if evidence.get("kind") == "retained_carrier_normal_supported_pcf_roi":
        from noesis.calibration.structural_roi_evidence import verify_source_feature_evidence
    else:
        from noesis.calibration.structural_carrier_evidence import verify_source_feature_evidence
    verify_source_feature_evidence(candidate, points_path, evidence, root, offline=offline, binding=binding)


def _authored_groups(manifest, room_map, rebinding, root):
    """Keep PCF/frame bytes unchanged when an owner re-exports the home model."""
    labels = list(manifest.semantic_binding.room_labels)
    groups = sorted({g for label in labels for g in room_map.get("rooms", {}).get(label, [])})
    if not groups:
        raise PresentationRegistrationError("authored room map omits the admitted room labels")
    if rebinding is None:
        if (room_map.get("authored_scene_sha256") != manifest.authored_scene.sha256
                or groups != sorted(manifest.semantic_binding.authored_groups)):
            raise PresentationRegistrationError("semantic room map and PCF authored parent disagree")
    else:
        _exact_keys(rebinding, {"source_room_group_map", "identity_review"}, label="authored room rebinding")
        if not isinstance(rebinding["identity_review"], str) or not rebinding["identity_review"].strip():
            raise PresentationRegistrationError("authored room identities have not been reviewed")
        ref = rebinding["source_room_group_map"]
        old_map = _read_json(_verified_file(ref, root, maximum=256 * 1024))
        if (ref["sha256"] != manifest.semantic_binding.room_group_map.sha256
                or old_map.get("authored_scene_sha256") != manifest.authored_scene.sha256
                or sorted({g for label in labels for g in old_map.get("rooms", {}).get(label, [])}) != sorted(manifest.semantic_binding.authored_groups)):
            raise PresentationRegistrationError("authored rebinding lost the original PCF semantic identity")
    return groups


def _projection_evidence(bundle: Mapping[str, Any], camera: str) -> dict[str, Any]:
    return {"K": list(bundle["cameras"]["K"][camera]),
            "E": list(bundle["cameras"]["E"][camera]),
            "frame_binding": dict(bundle["cameras"]["frame_bindings"][camera])}


def _support_binding(support: SupportSurfaceSet, camera: str, binding: Mapping[str, Any]) -> None:
    if (support.camera_id != camera or support.authority != "measured_calibration"
            or support.world_frame_revision != binding["world_frame"]["revision"]
            or support.calibration_revision != binding["camera_calibration_sha256"]
            or support.world_transform_sha256 != binding["transform_sha256"]):
        raise PresentationRegistrationError("measured support is not bound to the admitted camera/world revision")


def _catalog_manifest(catalog_path: Path, catalog: ScenePriorCatalog, camera: str, root: Path):
    selected = next((b for b in catalog.camera_bindings if b.camera_id == camera), None)
    if selected is None or selected.frame_binding is None:
        raise PresentationRegistrationError("structural source has no admitted PCF camera binding")
    reference = next(r for r in catalog.revisions if r.prior_id == selected.prior_id)
    path = catalog_path.parent / reference.manifest_path
    ref = {"path": path.relative_to(root).as_posix(), "sha256": reference.manifest_sha256,
           "size_bytes": reference.manifest_size_bytes}
    _verified_file(ref, root, maximum=1024 * 1024)
    manifest = ScenePriorRevision.model_validate(_read_json(path))
    if (manifest.prior_id != selected.frame_binding.target_frame.revision
            or manifest.preview.target_revision_metadata.sha256 != selected.frame_binding.target_revision_metadata_sha256):
        raise PresentationRegistrationError("structural source is not the exact admitted PCF target metadata")
    return selected.frame_binding, path, ref, manifest


def _unchanged_room(old: AuthoredSceneGeometry, new: AuthoredSceneGeometry, groups: list[str], hint: float,
                    measured_adjacent_groups: set[str] | None = None) -> None:
    from noesis.calibration.authored_structures import extract_authored_structures
    def geometry_signature(geometry):
        structures = extract_authored_structures(geometry, groups, initial_scene_units_per_meter=hint)
        # Global plane numbering and a plane's numerical averaging can change
        # when a different room's floor is derived. Compare the actual local
        # input triangles instead, including all extracted wall/opening support
        # and every selected floor face. No coordinate tolerance hides edits.
        refs = {ref for item in (*structures.wall_segments, *structures.openings)
                for ref in item.source_refs}
        signature = {}
        for triangle in geometry.triangles:
            ref = f"{'+'.join(triangle.groups)}#triangle:{triangle.index}"
            if ref in refs or set(triangle.groups).intersection(groups):
                signature[ref] = tuple(geometry.vertices[i] for i in triangle.vertex_indices)
        if not refs.issubset(signature):
            raise PresentationRegistrationError("retained room structural support is unavailable")
        return signature
    before, after = geometry_signature(old), geometry_signature(new)
    if set(before) != set(after):
        raise PresentationRegistrationError("retained room structural membership changed")
    # A neighbouring measured slab may share a short riser with this room.
    # Its explicit level is checked separately. Retention permits only that
    # slab's top-Y derivation, never changed X/Z, walls or this room's own floor.
    allowed = set(measured_adjacent_groups or ()) - set(groups)
    levels = {group: (_floor_y(old, [group]), _floor_y(new, [group])) for group in allowed}
    for ref, previous in before.items():
        current = after[ref]
        if previous == current:
            continue
        group_names = set(ref.rsplit("#triangle:", 1)[0].split("+"))
        candidates = group_names.intersection(allowed)
        if (not candidates or group_names.intersection(groups)
                or not any(all(a[0] == b[0] and a[2] == b[2]
                    and (a[1] == b[1] or (abs(a[1] - lower) < 1e-9 and abs(b[1] - upper) < 1e-9))
                    for a, b in zip(previous, current)) for lower, upper in (levels[g] for g in candidates))):
            raise PresentationRegistrationError("retained room geometry changed; it requires its own structural solve")


def build_structural_presentation(
    *, bundle_path: str | Path, authored_obj_path: str | Path,
    room_group_map_path: str | Path,
    scene_prior_catalog_path: str | Path,
    room_solutions: Mapping[str, Mapping[str, Any]],
    support_surfaces: Mapping[str, SupportSurfaceSet] | None = None,
    measured_floor_groups: Mapping[str, Mapping[str, list[str]]] | None = None,
    retained_registration_path: str | Path | None = None,
    retained_authored_obj_path: str | Path | None = None,
    parent_authored_obj_path: str | Path | None = None,
    owner_selection: Mapping[str, Any] | None = None,
    authored_room_rebinding: Mapping[str, Any] | None = None,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build from passed room solves; explicitly retain untouched room entries.

    Each room solution supplies ``manifest_path`` and the unmodified solver
    ``candidate``. ``measured_floor_groups`` maps camera -> surface ID -> actual
    authored floor groups. It is site data, never a generic room/height policy.
    """
    bundle = _read_json(bundle_path)
    room_map = _read_json(room_group_map_path)
    catalog_path = Path(scene_prior_catalog_path)
    catalog = ScenePriorCatalog.model_validate(_read_json(catalog_path))
    geometry = AuthoredSceneGeometry.from_obj(authored_obj_path)
    retained = None
    retained_ref = None
    retained_geometry = None
    if retained_registration_path is not None or retained_authored_obj_path is not None:
        raise PresentationRegistrationError("inherited camera-marker retention is retired; solve every current room")
    parent_ref = None
    if geometry.source_sha256 != room_map.get("authored_scene_sha256"):
        if parent_authored_obj_path is None:
            raise PresentationRegistrationError("derived authored geometry requires its exact semantic parent")
        parent_ref = _relative_file(parent_authored_obj_path, root)
        if parent_ref["sha256"] != room_map.get("authored_scene_sha256"):
            raise PresentationRegistrationError("authored semantic parent bytes do not match the room map")
    bindings = bundle.get("cameras", {}).get("frame_bindings")
    if not isinstance(bindings, Mapping) or not 1 <= len(bindings) <= MAX_ENTRIES:
        raise PresentationRegistrationError("current target frame bindings are unavailable or unbounded")
    if not room_solutions or not set(room_solutions).issubset(bindings):
        raise PresentationRegistrationError("structural solutions do not name current cameras")
    selection = _owner_selection(owner_selection, set(room_solutions)) if owner_selection is not None else None
    sets = dict(support_surfaces or {})
    floor_groups = dict(measured_floor_groups or {})
    measured_adjacent_groups = {group for camera_groups in floor_groups.values()
                                for groups in camera_groups.values() for group in groups}
    if not set(sets).issubset(bindings) or not set(floor_groups).issubset(room_solutions):
        raise PresentationRegistrationError("measured floor evidence names an unavailable structural room")
    for camera, measured in sets.items():
        _support_binding(measured, camera, bindings[camera])
        if camera in room_solutions and set(floor_groups.get(camera, {})) != {s.surface_id for s in measured.surfaces}:
            raise PresentationRegistrationError("every measured support level must be mapped into the authored scene")
    entries = {}
    prior_bindings = {}
    for camera, binding in sorted(bindings.items()):
        prior, admitted_path, manifest_ref, admitted = _catalog_manifest(catalog_path, catalog, camera, root)
        prior_bindings[camera] = prior.model_dump(mode="json")
        if (prior.target_frame.model_dump(mode="json") != binding["world_frame"]
                or prior.target_from_source_sha256 != binding["transform_sha256"]):
            raise PresentationRegistrationError("bundle and admitted PCF frame binding disagree")
        frame = ScenePriorFrameRef.model_validate(binding["world_frame"])
        if frame.revision in entries:
            raise PresentationRegistrationError("one target frame requires one jointly solved presentation")
        if camera not in room_solutions:
            raise PresentationRegistrationError(f"{camera}: a structural solve is required; marker retention is retired")
        solution = room_solutions[camera]
        manifest_path = Path(solution["manifest_path"])
        if manifest_path.absolute() != admitted_path.absolute():
            raise PresentationRegistrationError("structural solve source differs from admitted PCF manifest")
        manifest = ScenePriorRevision.model_validate(_read_json(manifest_path))
        if (manifest.prior_id != frame.revision or manifest.preview.reference_camera_id != camera
                or manifest.preview.camera_calibration.sha256 != binding["camera_calibration_sha256"]):
            raise PresentationRegistrationError("structural source is not the admitted camera/PCF revision")
        groups = _authored_groups(manifest, room_map, authored_room_rebinding, root)
        labels = list(manifest.semantic_binding.room_labels)
        if sorted(groups) != sorted({g for name in labels for g in room_map.get("rooms", {}).get(name, [])}):
            raise PresentationRegistrationError("selected structural room groups disagree with semantic binding")
        artifact = next((row for row in manifest.artifacts if row.role == "points_glb"), None)
        if artifact is None:
            raise PresentationRegistrationError("admitted structural PCF points are unavailable")
        points_path = manifest_path.parent / artifact.relative_path
        points_ref = _relative_file(points_path, root)
        if points_ref["sha256"] != artifact.sha256 or points_ref["size_bytes"] != artifact.size_bytes:
            raise PresentationRegistrationError("structural PCF points differ from immutable manifest")
        world_floor = _floor_plane(binding)
        authored_y = _floor_y(geometry, groups)
        candidate = dict(solution["candidate"])
        matrix = _matrix_for_floor(candidate, world_floor, authored_y, owner_selection=selection, camera=camera)
        image_evidence = list(solution.get("source_image_evidence", []))
        authored_selectors = list(solution.get("authored_point_selectors", []))
        source_feature_evidence = solution.get("source_feature_evidence")
        authored_wall_selectors = solution.get("authored_wall_selectors", [])
        _verify_geometry_features(candidate=candidate, points_path=points_path, geometry=geometry,
            groups=groups, bundle=bundle, camera=camera, image_evidence=image_evidence,
            authored_selectors=authored_selectors, root=root, support=sets.get(camera),
            source_feature_evidence=source_feature_evidence, authored_wall_selectors=authored_wall_selectors)
        stability = _annotation_stability(candidate=candidate, bundle=bundle, camera=camera,
                                          image_evidence=image_evidence, root=root)
        if selection is not None:
            _selection_quality_only(stability)
        if not stability["passed"] and selection is None:
            raise PresentationRegistrationError("reviewed annotation uncertainty fails structural fit/holdout gates: "
                                                + json.dumps(stability["failed_case"], sort_keys=True))
        scale = float(candidate["solution"]["scene_per_m"])
        measured = sets.get(camera)
        levels = []
        if camera in floor_groups:
            if measured is None or measured.authority != "measured_calibration":
                raise PresentationRegistrationError("measured authored levels require the bound measured support set")
            for surface_id, authored_groups in sorted(floor_groups[camera].items()):
                surface = next((s for s in measured.surfaces if s.surface_id == surface_id), None)
                if surface is None or not np.allclose(surface.normal_world, (0, 1, 0), atol=1e-6, rtol=0):
                    raise PresentationRegistrationError("measured authored level has no parallel support plane")
                expected_y = authored_y + scale * (world_floor["offset_m"] - surface.offset_m)
                actual_y = _floor_y(geometry, list(authored_groups))
                if abs(expected_y - actual_y) > 1e-3:
                    raise PresentationRegistrationError("authored level no longer represents its measured rise at the solved scale")
                levels.append({"surface_id": surface_id, "surface_revision": surface.revision,
                               "authored_groups": list(authored_groups), "authored_y": actual_y})
        values = [float(v) for v in matrix.flatten(order="F")]
        entry = {
            "camera_id": camera, "world_frame": frame.model_dump(mode="json"),
            "calibration_frame_revision": binding["calibration_frame"]["revision"],
            "calibration_revision": binding["camera_calibration_sha256"],
            "source_transform_sha256": binding["transform_sha256"],
            "source_manifest": _relative_file(manifest_path, root), "source_points": points_ref,
            "room_labels": labels, "room_groups": groups,
            "world_floor_plane": world_floor, "authored_floor_y": authored_y,
            "candidate": candidate, "candidate_sha256": _canonical_sha256(candidate),
            "source_image_evidence": image_evidence,
            "authored_point_selectors": authored_selectors,
            "annotation_stability": stability,
            "image_projection": _projection_evidence(bundle, camera),
            "measured_levels": levels,
            "world_to_scene_col_major": values,
            "world_to_scene_sha256": world_to_scene_presentation_sha256(frame, "menon_scene", tuple(values)),
            "scene_revision_id": scene_revision_sha256(frame, "menon_scene", tuple(values)),
        }
        entries[frame.revision] = {"kind": "structural", "entry": entry}
        if source_feature_evidence is not None:
            entry["source_feature_evidence"] = source_feature_evidence
        if authored_wall_selectors:
            entry["authored_wall_selectors"] = authored_wall_selectors
    result = {
        "contract": OWNER_SELECTED_CONTRACT if selection else CONTRACT, "contract_version": CONTRACT_VERSION,
        "authored_scene_sha256": geometry.source_sha256,
        "room_group_map": _relative_file(room_group_map_path, root),
        "scene_prior_catalog": _relative_file(catalog_path, root),
        "scene_prior_bindings": prior_bindings,
        "authored_parent": parent_ref,
        "support_sets": {camera: value.model_dump(mode="json") for camera, value in sorted(sets.items())},
        "retained_registration": retained_ref if any(e["kind"] == "retained" for e in entries.values()) else None,
        "presentations": entries,
    }
    if selection is not None:
        result["owner_selection"] = selection
    if authored_room_rebinding is not None:
        result["authored_room_rebinding"] = dict(authored_room_rebinding)
    result["registration_sha256"] = _canonical_sha256(result)
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_BYTES:
        raise PresentationRegistrationError("structural presentation exceeds its startup bound")
    return result


def load_structural_presentation(path: str | Path, *, authored_obj_path: str | Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_BYTES:
        raise PresentationRegistrationError("structural presentation is missing or oversized")
    result = _read_json(source)
    selected_by_owner = result.get("contract") == OWNER_SELECTED_CONTRACT
    expected_root = {"contract", "contract_version", "authored_scene_sha256", "room_group_map",
                       "scene_prior_catalog", "scene_prior_bindings",
                       "authored_parent", "support_sets", "retained_registration", "presentations", "registration_sha256"}
    if selected_by_owner:
        expected_root.add("owner_selection")
    if "authored_room_rebinding" in result:
        expected_root.add("authored_room_rebinding")
    _exact_keys(result, expected_root, label="structural presentation")
    if result["contract"] not in (CONTRACT, OWNER_SELECTED_CONTRACT) or result["contract_version"] != CONTRACT_VERSION:
        raise PresentationRegistrationError("unsupported structural presentation contract")
    _sha(result["authored_scene_sha256"], label="authored scene SHA")
    _sha(result["registration_sha256"], label="registration SHA")
    if result["registration_sha256"] != _canonical_sha256({k: v for k, v in result.items() if k != "registration_sha256"}):
        raise PresentationRegistrationError("structural registration digest is invalid")
    geometry = AuthoredSceneGeometry.from_obj(authored_obj_path)
    if geometry.source_sha256 != result["authored_scene_sha256"]:
        raise PresentationRegistrationError("loaded authored OBJ does not match structural registration")
    room_map_path = _verified_file(result["room_group_map"], root, maximum=256 * 1024)
    room_map = _read_json(room_map_path)
    catalog_path = _verified_file(result["scene_prior_catalog"], root, maximum=1024 * 1024)
    catalog = ScenePriorCatalog.model_validate(_read_json(catalog_path))
    parent = result["authored_parent"]
    semantic_sha = result["authored_scene_sha256"]
    if parent is not None:
        _verified_file(parent, root, maximum=64 * 1024 * 1024)
        semantic_sha = parent["sha256"]
    if semantic_sha != room_map.get("authored_scene_sha256"):
        raise PresentationRegistrationError("authored scene has lost its semantic parent binding")
    entries = result["presentations"]
    if not isinstance(entries, dict) or not 1 <= len(entries) <= MAX_ENTRIES:
        raise PresentationRegistrationError("structural presentation entries are missing or unbounded")
    selection = _owner_selection(result["owner_selection"], {e["entry"]["camera_id"] for e in entries.values()
        if e.get("kind") == "structural"}) if selected_by_owner else None
    sets = result["support_sets"]
    if not isinstance(sets, dict) or len(sets) > MAX_ENTRIES:
        raise PresentationRegistrationError("structural support evidence is unbounded")
    for camera, payload in sets.items():
        measured = SupportSurfaceSet.model_validate(payload)
        if measured.camera_id != camera or measured.authority != "measured_calibration":
            raise PresentationRegistrationError("structural support evidence is not measured or correctly bound")
    cameras = set()
    measured_adjacent_groups = {group for item in entries.values() if item.get("kind") == "structural"
                                for level in item["entry"].get("measured_levels", []) for group in level["authored_groups"]}
    retained = None
    retained_geometry = None
    if result["retained_registration"] is not None:
        raise PresentationRegistrationError("inherited camera-marker retention is retired")
    for revision, item in entries.items():
        expected_keys = {"kind", "entry", "source_manifest", "room_groups"} if item.get("kind") == "retained" else {"kind", "entry"}
        _exact_keys(item, expected_keys, label="structural presentation selection")
        entry = item["entry"]
        if not isinstance(entry, Mapping):
            raise PresentationRegistrationError("structural entry is not an object")
        prior, _, manifest_ref, admitted = _catalog_manifest(catalog_path, catalog, entry["camera_id"], root)
        if result["scene_prior_bindings"].get(entry["camera_id"]) != prior.model_dump(mode="json"):
            raise PresentationRegistrationError("structural admitted frame metadata changed")
        if item["kind"] == "structural":
            entry_keys = {"camera_id", "world_frame", "calibration_frame_revision", "calibration_revision",
                "source_transform_sha256", "source_manifest", "source_points", "room_labels", "room_groups",
                "world_floor_plane", "authored_floor_y", "candidate", "candidate_sha256", "measured_levels",
                "source_image_evidence", "authored_point_selectors", "image_projection",
                "annotation_stability",
                "world_to_scene_col_major", "world_to_scene_sha256", "scene_revision_id"}
            if "source_feature_evidence" in entry:
                entry_keys.add("source_feature_evidence")
            if "authored_wall_selectors" in entry:
                entry_keys.add("authored_wall_selectors")
            _exact_keys(entry, entry_keys, label="structural room entry")
            manifest_path = _verified_file(entry["source_manifest"], root, maximum=1024 * 1024)
            if entry["source_manifest"] != manifest_ref:
                raise PresentationRegistrationError("structural source differs from admitted manifest bytes")
            points_path = _verified_file(entry["source_points"], root, maximum=64 * 1024 * 1024)
            manifest = ScenePriorRevision.model_validate(_read_json(manifest_path))
            points = next((a for a in manifest.artifacts if a.role == "points_glb"), None)
            if (manifest.prior_id != revision or manifest.preview.reference_camera_id != entry["camera_id"]
                    or manifest.preview.camera_calibration.sha256 != entry["calibration_revision"]
                    or _authored_groups(manifest, room_map, result.get("authored_room_rebinding"), root) != entry["room_groups"]
                    or list(manifest.semantic_binding.room_labels) != entry["room_labels"]
                    or points is None or points.sha256 != entry["source_points"]["sha256"]
                    or points.size_bytes != entry["source_points"]["size_bytes"]
                    or manifest_path.parent / points.relative_path != points_path
                    or (result.get("authored_room_rebinding") is None and manifest.semantic_binding.room_group_map.sha256 != result["room_group_map"]["sha256"])):
                raise PresentationRegistrationError("structural evidence does not match admitted PCF/semantic identity")
            _sha(entry["candidate_sha256"], label="candidate SHA")
            if entry["candidate_sha256"] != _canonical_sha256(entry["candidate"]):
                raise PresentationRegistrationError("structural candidate proof was altered")
            authored_y = _floor_y(geometry, entry["room_groups"])
            if abs(authored_y - entry["authored_floor_y"]) > 1e-6:
                raise PresentationRegistrationError("structural authored lower floor changed")
            world_floor = _floor_plane({"world_floor_plane": entry["world_floor_plane"]})
            expected = _matrix_for_floor(entry["candidate"], world_floor, authored_y,
                                         owner_selection=selection, camera=entry["camera_id"])
            if not np.allclose(expected.flatten(order="F"), entry["world_to_scene_col_major"], atol=1e-9, rtol=0):
                raise PresentationRegistrationError("presentation matrix differs from passed structural solve")
            projection = entry["image_projection"]
            _exact_keys(projection, {"K", "E", "frame_binding"}, label="structural image projection")
            binding = projection["frame_binding"]
            if (binding["camera_calibration_sha256"] != entry["calibration_revision"]
                    or binding["world_frame"] != entry["world_frame"]
                    or binding["calibration_frame"]["revision"] != entry["calibration_frame_revision"]
                    or binding["transform_sha256"] != entry["source_transform_sha256"]
                    or _floor_plane(binding) != world_floor):
                raise PresentationRegistrationError("structural image projection lost its admitted frame binding")
            camera = entry["camera_id"]
            frozen_bundle = {"cameras": {"K": {camera: projection["K"]},
                "E": {camera: projection["E"]}, "frame_bindings": {camera: binding}}}
            feature_proof = entry.get("source_feature_evidence")
            if feature_proof is not None:
                if entry["source_image_evidence"] != feature_proof.get("point_evidence"):
                    raise PresentationRegistrationError("carrier rays differ from their verified source feature proof")
                _verify_source_features(entry["candidate"], points_path, feature_proof, root, offline=False, binding=binding)
            elif any(row.get("kind") == "retained_carrier_pcf_roi_plane" for row in entry["source_image_evidence"]):
                raise PresentationRegistrationError("carrier rays require their measured PCF feature proof")
            measured = SupportSurfaceSet.model_validate(sets[camera]) if camera in sets else None
            if measured is not None:
                _support_binding(measured, camera, binding)
                if {level["surface_id"] for level in entry["measured_levels"]} != {s.surface_id for s in measured.surfaces}:
                    raise PresentationRegistrationError("authored scene omits a measured support level")
            # Feature extraction and its exact geometry replay belong to the
            # offline builder. Native startup verifies immutable source bytes,
            # frozen solve/rays and floor geometry; it does not load Open3D or
            # trimesh, or perform a second stochastic point-cloud fit.
            _verify_image_points(candidate=entry["candidate"], bundle=frozen_bundle,
                camera=camera, image_evidence=entry["source_image_evidence"],
                root=root, support=measured)
            if entry.get("authored_wall_selectors"):
                from noesis.calibration.authored_structures import selected_mesh_wall_face
                from noesis.calibration.structural_registration import _compare_replay, _segments
                walls = {w["segment_id"]: w for w in entry["candidate"]["inputs"]["authored_segments"]}
                for selector in entry["authored_wall_selectors"]:
                    _compare_replay(walls[selector["segment_id"]], _segments([selected_mesh_wall_face(geometry, selector)],
                        label="authored_segments", minimum_length=0.4 * entry["candidate"]["inputs"]["scene_per_m_hint"])[0], path="selected mesh wall")
            stability = _annotation_stability(candidate=entry["candidate"], bundle=frozen_bundle, camera=camera,
                                              image_evidence=entry["source_image_evidence"], root=root)
            if selection is not None:
                _selection_quality_only(stability)
            if not stability["passed"] and selection is None:
                raise PresentationRegistrationError("structural annotation uncertainty changed or fails its holdout gates")
            from noesis.calibration.structural_registration import _compare_replay
            _compare_replay(entry["annotation_stability"], stability, path="annotation stability")
            for level in entry["measured_levels"]:
                _exact_keys(level, {"surface_id", "surface_revision", "authored_groups", "authored_y"}, label="measured authored level")
                measured = SupportSurfaceSet.model_validate(sets[entry["camera_id"]])
                surface = next((s for s in measured.surfaces if s.surface_id == level["surface_id"]), None)
                if surface is None or surface.revision != level["surface_revision"]:
                    raise PresentationRegistrationError("measured authored support identity changed")
                expected_y = authored_y + float(entry["candidate"]["solution"]["scene_per_m"]) * (world_floor["offset_m"] - surface.offset_m)
                if (abs(_floor_y(geometry, level["authored_groups"]) - expected_y) > 1e-3
                        or abs(level["authored_y"] - expected_y) > 1e-3):
                    raise PresentationRegistrationError("authored floor no longer represents measured support elevation")
        else:
            raise PresentationRegistrationError("unknown structural presentation kind")
        frame = ScenePriorFrameRef.model_validate(entry["world_frame"])
        if frame.frame_id != "backend_world_m" or frame.revision != revision:
            raise PresentationRegistrationError("structural target revision key is invalid")
        if not isinstance(entry["camera_id"], str) or not entry["camera_id"] or entry["camera_id"] in cameras:
            raise PresentationRegistrationError("structural camera identities are duplicate or invalid")
        cameras.add(entry["camera_id"])
        for key in ("calibration_frame_revision", "calibration_revision"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise PresentationRegistrationError("structural calibration identity is empty")
        for key in ("source_transform_sha256", "world_to_scene_sha256", "scene_revision_id"):
            _sha(entry[key], label=key)
        values = tuple(entry["world_to_scene_col_major"])
        _matrix(list(values), label="structural target presentation matrix")
        if (entry["world_to_scene_sha256"] != world_to_scene_presentation_sha256(frame, "menon_scene", values)
                or entry["scene_revision_id"] != scene_revision_sha256(frame, "menon_scene", values)):
            raise PresentationRegistrationError("structural target matrix digest is invalid")
    return result


def current_structural_presentations(
    artifact: Mapping[str, Any], *, frame_contracts: Mapping[str, Any],
    support_surfaces: Mapping[str, SupportSurfaceSet], calibration_revision: str,
    camera_projection: Mapping[str, Mapping[str, Any]],
    scene_prior_bindings: Mapping[str, Any],
    prior_similarity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Cheap current-input admission. No file I/O or extraction at publication."""
    current_bindings = {camera: value.model_dump(mode="json") for camera, value in scene_prior_bindings.items()}
    if current_bindings != artifact["scene_prior_bindings"]:
        raise PresentationRegistrationError("structural admitted PCF metadata or frame binding changed")
    if set(artifact["support_sets"]) != set(support_surfaces):
        raise PresentationRegistrationError("structural presentation omits current measured support evidence")
    for camera, payload in artifact["support_sets"].items():
        current = support_surfaces.get(camera)
        if current is None or current.model_dump(mode="json") != payload:
            raise PresentationRegistrationError("structural presentation measured support evidence changed")
    entries = artifact["presentations"]
    output = {}
    if set(e["entry"]["camera_id"] for e in entries.values()) != set(frame_contracts):
        raise PresentationRegistrationError("structural presentation camera set changed")
    for camera, contract in sorted(frame_contracts.items()):
        revision = contract.target_frame.revision
        selected = entries.get(revision)
        if selected is None:
            raise PresentationRegistrationError("structural presentation target revision changed")
        entry = selected["entry"]
        if (entry["camera_id"] != camera or entry["source_transform_sha256"] != contract.transform_sha256
                or entry["calibration_frame_revision"] != contract.source_frame.revision
                or entry["calibration_revision"] != calibration_revision):
            raise PresentationRegistrationError("structural presentation camera/frame/calibration evidence changed")
        binding = {"calibration_frame": {"frame_id": contract.source_frame.frame_id, "revision": contract.source_frame.revision},
                   "world_frame": {"frame_id": contract.target_frame.frame_id, "revision": revision},
                   "target_from_calibration_col_major": list(contract.target_from_source_col_major),
                   "transform_sha256": contract.transform_sha256,
                   "world_floor_plane": {"normal": list(contract.target_floor_plane.normal), "offset_m": contract.target_floor_plane.offset_m}}
        if selected["kind"] == "structural":
            if _floor_plane(binding) != entry["world_floor_plane"]:
                raise PresentationRegistrationError("structural presentation bound floor changed")
            current = camera_projection.get(camera)
            projection = entry["image_projection"]
            if current is None or current.get("K") != projection["K"] or current.get("E") != projection["E"]:
                raise PresentationRegistrationError("structural presentation current camera rays changed")
        else:
            raise PresentationRegistrationError("inherited camera-marker retention is retired")
        frame = ScenePriorFrameRef(frame_id=contract.target_frame.frame_id, revision=revision)
        presentation = ScenePriorWorldToScenePresentation(
            contract="noesis.scene_prior.world_to_scene_presentation", contract_version=1,
            world_frame=frame, render_frame="menon_scene",
            world_to_scene_col_major=tuple(entry["world_to_scene_col_major"]),
            world_to_scene_sha256=entry["world_to_scene_sha256"], scene_revision_id=entry["scene_revision_id"],
            source_transform_sha256s=(contract.transform_sha256,),
            provenance=ArtifactFingerprint(role="world_to_scene_presentation", sha256=artifact["registration_sha256"],
                                           version=f"{artifact.get('contract', CONTRACT)}.v1", producer="noesis.calibration.structural_presentation"),
            accepted=True, authored_scene_sha256=artifact["authored_scene_sha256"], registration_sha256=artifact["registration_sha256"],
        )
        output[revision] = presentation.model_dump(mode="json")
    return output
