#!/usr/bin/env python3
"""Run fixture-based Noesis/Menon validation checks.

This is the offline/fixture entrypoint for the shared validation toolbox. It is
intentionally lightweight and GPU-free; DS8 runtime and Menon live validators can
feed the same JSON report schema as they are added.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.camera import (  # noqa: E402
    CameraCalibration,
    ReprojectionAnchor,
    validate_extrinsics,
    validate_intrinsics,
    validate_reprojection_anchors,
)
from noesis.validation.bev import validate_camera_coverage, validate_path_smoothness, validate_zone_consistency  # noqa: E402
from noesis.validation.artifacts import ArtifactIndex  # noqa: E402
from noesis.validation.core import ConfidenceScores, SourceMetadata, ValidationCheck, ValidationReport  # noqa: E402
from noesis.validation.fixtures import FixtureRegistry  # noqa: E402
from noesis.validation.geometry import (  # noqa: E402
    AnchorObservation,
    validate_known_anchors,
    validate_no_wall_crossing,
    validate_points_inside_room,
)
from noesis.validation.menon import (  # noqa: E402
    MenonAvatarSample,
    MenonBevTrailSample,
    MenonLatencySample,
    MenonObjectSample,
    MenonPlacementSample,
    WorldBevSample,
    validate_bev_menon_trail_agreement,
    validate_menon_avatar_collision,
    validate_menon_avatar_orientation,
    validate_menon_avatar_scale,
    validate_menon_latency_alignment,
    validate_menon_object_placement,
    validate_timestamp_alignment,
    validate_world_bev_round_trip,
    validate_world_menon_placement_agreement,
    validate_world_menon_round_trip,
)
from noesis.validation.reports import write_markdown  # noqa: E402
from noesis.validation.scene import (  # noqa: E402
    DepthAnchorObservation,
    MenonAssetObservation,
    MeshQualitySample,
    PlaneObservation,
    RoomGeometryConstraintObservation,
    RoomDimensionObservation,
    SceneCoordinateSystemObservation,
    SemanticObjectObservation,
    validate_depth_consistency,
    validate_menon_asset_sanity,
    validate_mesh_quality,
    validate_plane_geometry,
    validate_room_geometry_constraints,
    validate_room_dimensions,
    validate_scene_coordinate_system,
    validate_semantic_object_logic,
)
from noesis.validation.tracking import (  # noqa: E402
    BevPointSample,
    DetectionProjectionSample,
    TrackSample,
    validate_bev_track_agreement,
    validate_detection_world_projection,
    validate_doorway_transitions,
    validate_identity_continuity,
    validate_occlusion_bridges,
    validate_projection_confidence,
    validate_reid_geometry_consistency,
    validate_track_motion,
)
from noesis.validation.transforms import validate_round_trip, validate_transform_matrix  # noqa: E402
from noesis.validation.visuals import render_bev_diagnostic_overlay, render_camera_reprojection_overlay  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return payload


def _git_revision() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        revision = proc.stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        ).returncode != 0
        return f"{revision}-dirty" if dirty else revision
    except Exception:
        return None


def _source_from_fixture(payload: Mapping[str, Any]) -> SourceMetadata:
    source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
    return SourceMetadata(
        repo=str(source.get("repo") or "Noesis_Devel"),
        git_revision=str(source.get("git_revision") or _git_revision() or ""),
        pipeline_config=source.get("pipeline_config"),
        cameras_config=source.get("cameras_config"),
        menon_available=source.get("menon_available"),
        menon_revision=source.get("menon_revision"),
    )


def _seq(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _first_present(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)
    return None


def _string_seq(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in _seq(value)]


def _resolve_fixture_path(value: Any, *, fixture_dir: Path | None) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute() and fixture_dir is not None:
        path = fixture_dir / path
    return path


def _wall_segments(raw_segments: Any) -> list[tuple[Sequence[float], Sequence[float]]]:
    segments: list[tuple[Sequence[float], Sequence[float]]] = []
    for item in _seq(raw_segments):
        if isinstance(item, Mapping):
            a = item.get("a") or item.get("start")
            b = item.get("b") or item.get("end")
        elif isinstance(item, Sequence) and len(item) == 2:
            a, b = item[0], item[1]
        else:
            continue
        if isinstance(a, Sequence) and isinstance(b, Sequence) and len(a) >= 2 and len(b) >= 2:
            segments.append((a, b))
    return segments


def _track_path_xz(samples: Sequence[TrackSample]) -> list[list[float]]:
    path: list[list[float]] = []
    for sample in sorted(samples, key=lambda item: float(item.ts_s)):
        try:
            point = sample.world
            path.append([float(point[0]), float(point[2])])
        except Exception:
            continue
    return path


def _run_fixture(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    artifact_dir: Path | None = None,
    fixture_dir: Path | None = None,
) -> ValidationReport:
    artifact_index = ArtifactIndex(artifact_dir) if artifact_dir is not None else None
    report = ValidationReport(
        run_id=run_id,
        source=_source_from_fixture(payload),
        scope=payload.get("scope") if isinstance(payload.get("scope"), Mapping) else {},
        confidence=ConfidenceScores.from_mapping(payload.get("confidence") if isinstance(payload.get("confidence"), Mapping) else None),
    )

    for idx, item in enumerate(_seq(payload.get("transforms"))):
        if not isinstance(item, Mapping):
            continue
        matrix = item.get("matrix_col_major") or item.get("matrix")
        check_prefix = str(item.get("id") or f"TRANSFORM.{idx:03d}")
        if matrix is not None:
            for check in validate_transform_matrix(matrix, check_id=check_prefix):
                report.add_check(check)
            points = item.get("points")
            if isinstance(points, Sequence):
                report.add_check(validate_round_trip(points, matrix, check_id=f"{check_prefix}.round_trip"))

    anchors: list[AnchorObservation] = []
    for item in _seq(payload.get("anchors")):
        if not isinstance(item, Mapping):
            continue
        anchors.append(
            AnchorObservation(
                anchor_id=str(item.get("anchor_id") or item.get("id") or "anchor"),
                expected_world=item.get("expected_world") or item.get("expected") or [],
                observed_world=item.get("observed_world") or item.get("observed") or [],
                anchor_type=str(item.get("anchor_type") or item.get("type") or "object"),
                camera=item.get("camera"),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if anchors or "anchors" in payload:
        for check in validate_known_anchors(anchors):
            report.add_check(check)

    scene = payload.get("scene") if isinstance(payload.get("scene"), Mapping) else {}
    coordinate_systems: list[SceneCoordinateSystemObservation] = []
    for item in _seq(scene.get("coordinate_systems") or scene.get("coordinate_system")):
        if not isinstance(item, Mapping):
            continue
        coordinate_systems.append(
            SceneCoordinateSystemObservation(
                scene_id=str(item.get("scene_id") or item.get("id") or item.get("room") or "scene"),
                units=str(item.get("units") or item.get("unit") or "m"),
                axis_convention=str(item.get("axis_convention") or "x_right_y_up_z_forward"),
                expected_axis_convention=str(item.get("expected_axis_convention") or "x_right_y_up_z_forward"),
                camera_pose_convention=str(item.get("camera_pose_convention") or "world_to_camera"),
                expected_camera_pose_convention=str(item.get("expected_camera_pose_convention") or "world_to_camera"),
                origin_delta_m=item.get("origin_delta_m"),
                transform_col_major=item.get("transform_col_major") or item.get("matrix_col_major"),
                round_trip_points=item.get("round_trip_points") or item.get("points") or [],
                scale_anchor_expected_m=item.get("scale_anchor_expected_m"),
                scale_anchor_observed_m=item.get("scale_anchor_observed_m"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if coordinate_systems or "coordinate_systems" in scene or "coordinate_system" in scene:
        report.add_check(validate_scene_coordinate_system(coordinate_systems))

    planes: list[PlaneObservation] = []
    for item in _seq(scene.get("planes")):
        if not isinstance(item, Mapping):
            continue
        planes.append(
            PlaneObservation(
                plane_id=str(item.get("plane_id") or item.get("id") or "plane"),
                plane_type=str(item.get("plane_type") or item.get("type") or "unknown"),
                normal=item.get("normal") or [],
                rms_deviation_m=item.get("rms_deviation_m"),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if planes or "planes" in scene:
        report.add_check(validate_plane_geometry(planes))

    room_dimensions: list[RoomDimensionObservation] = []
    for item in _seq(scene.get("room_dimensions")):
        if not isinstance(item, Mapping):
            continue
        room_dimensions.append(
            RoomDimensionObservation(
                room_id=str(item.get("room_id") or item.get("room") or item.get("id") or "room"),
                width_m=item.get("width_m"),
                length_m=item.get("length_m"),
                height_m=item.get("height_m"),
                expected_width_m=item.get("expected_width_m"),
                expected_length_m=item.get("expected_length_m"),
                expected_height_m=item.get("expected_height_m"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if room_dimensions or "room_dimensions" in scene:
        report.add_check(validate_room_dimensions(room_dimensions))

    room_constraints: list[RoomGeometryConstraintObservation] = []
    for item in _seq(scene.get("room_geometry_constraints") or scene.get("geometry_constraints")):
        if not isinstance(item, Mapping):
            continue
        room_constraints.append(
            RoomGeometryConstraintObservation(
                constraint_id=str(item.get("constraint_id") or item.get("id") or "constraint"),
                constraint_type=str(item.get("constraint_type") or item.get("type") or "unknown"),
                gap_m=item.get("gap_m"),
                overlap_m=item.get("overlap_m"),
                angle_error_deg=item.get("angle_error_deg"),
                doorway_width_m=item.get("doorway_width_m"),
                doorway_height_m=item.get("doorway_height_m"),
                doorway_bottom_gap_m=item.get("doorway_bottom_gap_m"),
                window_width_m=_first_present(item, "window_width_m", "windowWidthM"),
                window_height_m=_first_present(item, "window_height_m", "windowHeightM"),
                window_sill_height_m=_first_present(item, "window_sill_height_m", "windowSillHeightM"),
                surface_gap_m=item.get("surface_gap_m"),
                normal_dot=item.get("normal_dot"),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if room_constraints or "room_geometry_constraints" in scene or "geometry_constraints" in scene:
        report.add_check(validate_room_geometry_constraints(room_constraints))

    meshes: list[MeshQualitySample] = []
    for item in _seq(scene.get("meshes")):
        if not isinstance(item, Mapping):
            continue
        meshes.append(
            MeshQualitySample(
                asset_id=str(item.get("asset_id") or item.get("id") or "mesh"),
                bbox_min=item.get("bbox_min") or [],
                bbox_max=item.get("bbox_max") or [],
                triangle_count=item.get("triangle_count"),
                vertex_count=item.get("vertex_count"),
                watertight_expected=bool(item.get("watertight_expected", False)),
                watertight=item.get("watertight"),
                non_manifold_edges=item.get("non_manifold_edges"),
                inverted_faces=item.get("inverted_faces"),
                duplicate_faces=item.get("duplicate_faces"),
                uv_overlap_ratio=item.get("uv_overlap_ratio"),
                lod_max_error_m=_first_present(item, "lod_max_error_m", "lodMaxErrorM"),
                texture_alignment_error_px=_first_present(item, "texture_alignment_error_px", "textureAlignmentErrorPx", "atlas_alignment_error_px", "atlasAlignmentErrorPx"),
                texture_stretch_ratio=_first_present(item, "texture_stretch_ratio", "textureStretchRatio"),
                bvh_valid=_first_present(item, "bvh_valid", "bvhValid", "collision_bvh_valid", "collisionBvhValid"),
                collision_bvh_error_m=_first_present(item, "collision_bvh_error_m", "collisionBvhErrorM"),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if meshes or "meshes" in scene:
        report.add_check(validate_mesh_quality(meshes))

    menon_assets: list[MenonAssetObservation] = []
    for item in _seq(scene.get("menon_assets") or scene.get("menon_asset_checks")):
        if not isinstance(item, Mapping):
            continue
        menon_assets.append(
            MenonAssetObservation(
                asset_id=str(item.get("asset_id") or item.get("id") or "menon_asset"),
                asset_type=str(item.get("asset_type") or item.get("type") or "asset"),
                scale_error_m=_first_present(item, "scale_error_m", "scaleErrorM"),
                orientation_error_deg=_first_present(item, "orientation_error_deg", "orientationErrorDeg"),
                origin_error_m=_first_present(item, "origin_error_m", "originErrorM"),
                collision_mesh_error_m=_first_present(item, "collision_mesh_error_m", "collisionMeshErrorM"),
                floor_flatness_m=_first_present(item, "floor_flatness_m", "floorFlatnessM"),
                floor_alignment_error_m=_first_present(item, "floor_alignment_error_m", "floorAlignmentErrorM"),
                walkable=item.get("walkable"),
                doorway_blocked=bool(item.get("doorway_blocked", item.get("doorwayBlocked", False))),
                camera_position_error_m=_first_present(item, "camera_position_error_m", "cameraPositionErrorM"),
                camera_aim_error_deg=_first_present(item, "camera_aim_error_deg", "cameraAimErrorDeg"),
                overlay_rendered_layers=_string_seq(_first_present(item, "overlay_rendered_layers", "overlayRenderedLayers", "rendered_layers", "renderedLayers")),
                required_overlay_layers=_string_seq(_first_present(item, "required_overlay_layers", "requiredOverlayLayers", "required_layers", "requiredLayers")),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if menon_assets or "menon_assets" in scene or "menon_asset_checks" in scene:
        report.add_check(validate_menon_asset_sanity(menon_assets))

    depth_anchors: list[DepthAnchorObservation] = []
    for item in _seq(scene.get("depth_anchors")):
        if not isinstance(item, Mapping):
            continue
        depth_anchors.append(
            DepthAnchorObservation(
                anchor_id=str(item.get("anchor_id") or item.get("id") or "depth_anchor"),
                expected_depth_m=float(_first_present(item, "expected_depth_m", "expectedDepthM") or 0.0),
                observed_depth_m=float(_first_present(item, "observed_depth_m", "observedDepthM") or 0.0),
                order_group=_first_present(item, "order_group", "orderGroup", "relative_order_group", "relativeOrderGroup"),
                expected_order=_first_present(item, "expected_order", "expectedOrder", "relative_order", "relativeOrder"),
                plane_residual_m=item.get("plane_residual_m"),
                object_depth_error_m=_first_present(item, "object_depth_error_m", "objectDepthErrorM"),
                temporal_std_m=item.get("temporal_std_m"),
                static_std_m=_first_present(item, "static_std_m", "staticSceneStdM", "static_scene_std_m", "staticSceneStdM"),
                edge_alignment_error_px=_first_present(item, "edge_alignment_error_px", "edgeAlignmentErrorPx", "depth_edge_error_px", "depthEdgeErrorPx"),
                confidence=_first_present(item, "confidence", "depth_confidence", "depthConfidence"),
                fusion_weight=_first_present(item, "fusion_weight", "fusionWeight", "depth_fusion_weight", "depthFusionWeight"),
                camera=item.get("camera_id") or item.get("camera"),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if depth_anchors or "depth_anchors" in scene:
        report.add_check(validate_depth_consistency(depth_anchors))

    semantic_objects: list[SemanticObjectObservation] = []
    for item in _seq(scene.get("semantic_objects")):
        if not isinstance(item, Mapping):
            continue
        semantic_objects.append(
            SemanticObjectObservation(
                object_id=str(item.get("object_id") or item.get("id") or "object"),
                category=str(item.get("category") or item.get("class") or "object"),
                bbox_min=item.get("bbox_min") or [],
                bbox_max=item.get("bbox_max") or [],
                support_y=float(_first_present(item, "support_y", "supportY") or 0.0),
                max_wall_intersection_m=float(_first_present(item, "max_wall_intersection_m", "maxWallIntersectionM") or 0.0),
                max_static_shift_m=_first_present(item, "max_static_shift_m", "maxStaticShiftM"),
                allowed_rooms=_string_seq(_first_present(item, "allowed_rooms", "allowedRooms", "compatible_rooms", "compatibleRooms")),
                doorway_clearance_m=_first_present(item, "doorway_clearance_m", "doorwayClearanceM"),
                free_space_clearance_m=_first_present(item, "free_space_clearance_m", "freeSpaceClearanceM"),
                walkable_area_blocked_ratio=_first_present(item, "walkable_area_blocked_ratio", "walkableAreaBlockedRatio"),
                known_anchor_error_m=_first_present(item, "known_anchor_error_m", "knownAnchorErrorM", "anchor_error_m", "anchorErrorM"),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if semantic_objects or "semantic_objects" in scene:
        report.add_check(validate_semantic_object_logic(semantic_objects))

    camera_frustums_xz: list[Sequence[Sequence[float]]] = []
    camera_frustums_by_id: dict[str, Sequence[Sequence[float]]] = {}
    for item in _seq(payload.get("cameras")):
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("frustum_xz"), Sequence):
            camera_frustums_xz.append(item.get("frustum_xz"))  # type: ignore[arg-type]
            camera_name = str(item.get("camera_id") or item.get("id") or "camera")
            camera_frustums_by_id[camera_name] = item.get("frustum_xz")  # type: ignore[assignment]
        try:
            calibration = CameraCalibration(
                camera_id=str(item.get("camera_id") or item.get("id") or "camera"),
                intrinsics=item.get("intrinsics") or item.get("K"),
                extrinsics_col_major=item.get("extrinsics_col_major") or item.get("E"),
                image_size=tuple(item.get("image_size") or item.get("resolution") or [0, 0]),  # type: ignore[arg-type]
                floor_y=float(item.get("floor_y") or 0.0),
                unit_scale=float(item.get("unit_scale") or 1.0),
                stream_kind=str(item.get("stream_kind") or "unknown"),
                applies_to_raw=item.get("applies_to_raw"),
                applies_to_dewarped=item.get("applies_to_dewarped"),
            )
        except Exception as exc:
            raise RuntimeError(f"invalid camera fixture entry: {exc}") from exc
        expected_resolution = item.get("expected_resolution")
        for check in validate_intrinsics(
            calibration,
            expected_resolution=tuple(expected_resolution) if isinstance(expected_resolution, Sequence) else None,  # type: ignore[arg-type]
        ):
            report.add_check(check)
        for check in validate_extrinsics(calibration):
            report.add_check(check)
        reprojection_anchors = []
        for anchor in _seq(item.get("reprojection_anchors")):
            if not isinstance(anchor, Mapping):
                continue
            reprojection_anchors.append(
                ReprojectionAnchor(
                    anchor_id=str(anchor.get("anchor_id") or anchor.get("id") or "anchor"),
                    world_point=anchor.get("world_point") or anchor.get("world") or [],
                    expected_pixel=anchor.get("expected_pixel") or anchor.get("pixel") or [],
                    camera=anchor.get("camera"),
                    room=anchor.get("room"),
                    evidence=_seq(anchor.get("evidence")),
                )
            )
        if reprojection_anchors or "reprojection_anchors" in item:
            for check in validate_reprojection_anchors(calibration, reprojection_anchors):
                report.add_check(check)
        if artifact_index is not None and (reprojection_anchors or item.get("render_overlay")):
            image_path = _resolve_fixture_path(item.get("image_path"), fixture_dir=fixture_dir)
            overlay_path = artifact_index.path(f"visual/camera_{calibration.camera_id}_reprojection.png")
            try:
                render_camera_reprojection_overlay(
                    calibration,
                    reprojection_anchors,
                    overlay_path,
                    background_path=image_path,
                    room_outline_world=_seq(item.get("room_outline_world")),
                    detected_footpoints_world=_seq(item.get("detected_footpoints_world")),
                    mesh_edges_world=_seq(item.get("mesh_edges_world")),
                    image_bboxes_xyxy=_seq(item.get("image_bboxes_xyxy") or item.get("detected_bboxes_xyxy")),
                    projected_bboxes_xyxy=_seq(item.get("projected_bboxes_xyxy") or item.get("projected_bbox_xyxy") or item.get("projected_3d_bboxes_xyxy")),
                    avatar_bboxes_xyxy=_seq(item.get("avatar_bboxes_xyxy") or item.get("projected_avatar_bboxes_xyxy")),
                    mask_polygons_xy=_seq(item.get("mask_polygons_xy") or item.get("segmentation_masks_xy")),
                )
                artifact_index.add(
                    f"camera_{calibration.camera_id}_reprojection_overlay",
                    overlay_path,
                    metadata={"camera": calibration.camera_id, "kind": "camera_reprojection_overlay"},
                )
            except Exception as exc:
                report.add_check(
                    ValidationCheck(
                        id=f"VISUAL.{calibration.camera_id}.reprojection_overlay",
                        domain="camera",
                        name="camera_reprojection_overlay",
                        status="warning",
                        failure_type="infrastructure_failure",
                        camera=calibration.camera_id,
                        detail=f"Unable to render camera reprojection overlay: {exc}",
                        suggested_next_diagnostic="Check cv2/Pillow availability and fixture image path.",
                    )
                )

    track_samples: list[TrackSample] = []
    for item in _seq(payload.get("tracks")):
        if not isinstance(item, Mapping):
            continue
        track_samples.append(
            TrackSample(
                ts_s=float(item.get("ts_s") or item.get("timestamp_s") or 0.0),
                world=item.get("world") or [],
                camera_id=item.get("camera_id") or item.get("camera"),
                stable_id=_first_present(item, "stable_id", "stableId"),
                tracker_id=_first_present(item, "tracker_id", "trackerId", "track_id", "trackId"),
                room=item.get("room"),
                confidence=item.get("confidence"),
                projection_confidence=_first_present(item, "projection_confidence", "projectionConfidence"),
                temporal_confidence=_first_present(item, "temporal_confidence", "temporalConfidence"),
                reid_confidence=_first_present(item, "reid_confidence", "reidConfidence", "appearance_confidence", "appearanceConfidence"),
                reid_identity=_first_present(item, "reid_identity", "reidIdentity", "reid_id", "reidId"),
                appearance_id=_first_present(item, "appearance_id", "appearanceId", "appearance_cluster", "appearanceCluster", "appearance_label", "appearanceLabel"),
                occluded=item.get("occluded"),
                occlusion_uncertainty_m=_first_present(item, "occlusion_uncertainty_m", "occlusionUncertaintyM", "uncertainty_m", "uncertaintyM"),
                warnings=_seq(item.get("warnings")),
            )
        )
    if track_samples or "tracks" in payload:
        for check in validate_track_motion(track_samples):
            report.add_check(check)
        report.add_check(validate_path_smoothness(track_samples))
        report.add_check(validate_occlusion_bridges(track_samples))
        report.add_check(validate_identity_continuity(track_samples))
        report.add_check(validate_reid_geometry_consistency(track_samples))

    projection_samples: list[DetectionProjectionSample] = []
    for item in _seq(payload.get("projection_samples")):
        if not isinstance(item, Mapping):
            continue
        projection_samples.append(
            DetectionProjectionSample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or "entity"),
                world_point=item.get("world_point") or item.get("world") or [],
                bbox_foot_world=item.get("bbox_foot_world"),
                mask_foot_world=item.get("mask_foot_world"),
                pose_foot_world=item.get("pose_foot_world"),
                depth_world=item.get("depth_world"),
                image_bbox_xyxy=_first_present(item, "image_bbox_xyxy", "imageBboxXyxy", "bbox_xyxy", "bboxXyxy"),
                projected_bbox_xyxy=_first_present(item, "projected_bbox_xyxy", "projectedBboxXyxy", "bbox3d_projected_xyxy", "bbox3dProjectedXyxy"),
                projected_bbox_iou=_first_present(item, "projected_bbox_iou", "projectedBboxIou", "bbox3d_iou", "bbox3dIou"),
                bbox_center_error_px=_first_present(item, "bbox_center_error_px", "bboxCenterErrorPx"),
                reprojection_score=_first_present(item, "reprojection_score", "reprojectionScore"),
                temporal_smoothness_score=_first_present(item, "temporal_smoothness_score", "temporalSmoothnessScore"),
                semantic_validity_score=_first_present(item, "semantic_validity_score", "semanticValidityScore"),
                floor_y=float(item.get("floor_y") or 0.0),
                person_height_m=item.get("person_height_m"),
                ray_floor_valid=bool(item.get("ray_floor_valid", True)),
                room_polygon_xz=item.get("room_polygon_xz") or [],
                camera_id=item.get("camera_id") or item.get("camera"),
                room=item.get("room"),
                evidence=_seq(item.get("evidence")),
            )
        )
    if projection_samples or "projection_samples" in payload:
        report.add_check(validate_detection_world_projection(projection_samples))
        report.add_check(validate_projection_confidence(projection_samples))

    room_defs = [item for item in _seq(payload.get("rooms")) if isinstance(item, Mapping)]
    room_polygons_xz: list[Sequence[Sequence[float]]] = []
    room_polygons_by_id: dict[str, Sequence[Sequence[float]]] = {}
    wall_segments_xz: list[tuple[Sequence[float], Sequence[float]]] = []
    doorway_segments_xz: list[tuple[Sequence[float], Sequence[float]]] = []
    for room_idx, room_def in enumerate(room_defs):
        room_id = str(room_def.get("room") or room_def.get("id") or room_def.get("name") or f"room_{room_idx}")
        polygon = room_def.get("polygon_xz") or room_def.get("room_polygon_xz")
        if isinstance(polygon, Sequence) and len(polygon) >= 3:
            room_polygons_xz.append(polygon)  # type: ignore[arg-type]
            room_polygons_by_id[room_id] = polygon  # type: ignore[assignment]
            room_tracks = [sample for sample in track_samples if sample.room == room_id]
            if not room_tracks and len(room_defs) == 1:
                room_tracks = list(track_samples)
            report.add_check(
                validate_points_inside_room(
                    [sample.world for sample in room_tracks],
                    polygon,  # type: ignore[arg-type]
                    check_id=f"ROOM.{room_id}.points_inside",
                    room=room_id,
                    min_inside_ratio=float(room_def.get("min_inside_ratio") or 0.98),
                )
            )
        walls = _wall_segments(room_def.get("wall_segments_xz") or room_def.get("walls_xz"))
        wall_segments_xz.extend(walls)
        doorway_segments_xz.extend(_wall_segments(room_def.get("doorways_xz") or room_def.get("doorway_segments_xz")))
        if walls:
            grouped: dict[tuple[str | None, object], list[TrackSample]] = {}
            for sample in track_samples:
                if sample.room not in (None, room_id) and len(room_defs) > 1:
                    continue
                grouped.setdefault((sample.camera_id, sample.identity), []).append(sample)
            if not grouped:
                report.add_check(
                    validate_no_wall_crossing(
                        [],
                        walls,
                        check_id=f"ROOM.{room_id}.no_wall_crossing",
                        room=room_id,
                    )
                )
            for group_idx, samples_for_path in enumerate(grouped.values()):
                path_xz = _track_path_xz(samples_for_path)
                if len(path_xz) < 2:
                    continue
                report.add_check(
                    validate_no_wall_crossing(
                        path_xz,
                        walls,
                        check_id=f"ROOM.{room_id}.no_wall_crossing.{group_idx:03d}",
                        room=room_id,
                    )
                )
    if track_samples and room_defs:
        report.add_check(validate_doorway_transitions(track_samples, doorway_segments_xz))

    bev_points: list[BevPointSample] = []
    for item in _seq(payload.get("bev_points")):
        if not isinstance(item, Mapping):
            continue
        bev_points.append(
            BevPointSample(
                ts_s=float(item.get("ts_s") or item.get("timestamp_s") or 0.0),
                x=float(item.get("x") or 0.0),
                z=float(item.get("z") if item.get("z") is not None else item.get("y") or 0.0),
                camera_id=item.get("camera_id") or item.get("camera"),
                stable_id=item.get("stable_id") or item.get("stableId"),
                tracker_id=item.get("tracker_id") or item.get("trackerId"),
            )
        )
    if track_samples and (bev_points or "bev_points" in payload):
        report.add_check(validate_bev_track_agreement(track_samples, bev_points))
    if track_samples and room_polygons_by_id:
        report.add_check(validate_zone_consistency(track_samples, room_polygons_by_id))
    if track_samples and camera_frustums_by_id:
        report.add_check(validate_camera_coverage(track_samples, camera_frustums_by_id))

    if artifact_index is not None and (payload.get("render_bev_overlay") or isinstance(payload.get("bev_overlay"), Mapping)):
        bev_config = payload.get("bev_overlay") if isinstance(payload.get("bev_overlay"), Mapping) else {}
        track_paths = []
        grouped_tracks: dict[tuple[str | None, object], list[TrackSample]] = {}
        for sample in track_samples:
            grouped_tracks.setdefault((sample.camera_id, sample.identity), []).append(sample)
        for samples_for_path in grouped_tracks.values():
            path = _track_path_xz(samples_for_path)
            if path:
                track_paths.append(path)
        raw_footpoints = []
        for sample in projection_samples:
            try:
                raw_footpoints.append({"entity_id": sample.entity_id, "point": [float(sample.world_point[0]), float(sample.world_point[2])]})
            except Exception:
                continue
        overlay_path = artifact_index.path("visual/bev_diagnostic.png")
        try:
            render_bev_diagnostic_overlay(
                overlay_path,
                room_polygons_xz=room_polygons_xz,
                wall_segments_xz=wall_segments_xz,
                doorway_segments_xz=doorway_segments_xz,
                camera_frustums_xz=camera_frustums_xz,
                track_paths_xz=track_paths,
                raw_footpoints_xz=raw_footpoints,
                confidence_ellipses_xz=_seq(bev_config.get("confidence_ellipses")),
                track_annotations_xz=_seq(bev_config.get("track_annotations")),
                width=int(bev_config.get("width") or 900),
                height=int(bev_config.get("height") or 700),
            )
            artifact_index.add("bev_diagnostic_overlay", overlay_path, metadata={"kind": "bev_diagnostic_overlay"})
        except Exception as exc:
            report.add_check(
                ValidationCheck(
                    id="VISUAL.bev.diagnostic_overlay",
                    domain="bev",
                    name="bev_diagnostic_overlay",
                    status="warning",
                    failure_type="infrastructure_failure",
                    detail=f"Unable to render BEV diagnostic overlay: {exc}",
                    suggested_next_diagnostic="Check BEV overlay fixture geometry and cv2/Pillow availability.",
                )
            )

    menon = payload.get("menon") if isinstance(payload.get("menon"), Mapping) else {}
    world_to_bev = menon.get("world_to_bev_col_major") if isinstance(menon, Mapping) else None
    if world_to_bev is not None:
        world_bev_samples: list[WorldBevSample] = []
        for item in _seq(menon.get("world_bev_points") or menon.get("world_bev_samples")):
            if not isinstance(item, Mapping):
                continue
            world_bev_samples.append(
                WorldBevSample(
                    entity_id=str(item.get("entity_id") or item.get("id") or "entity"),
                    world_point=item.get("world_point") or item.get("world") or [],
                    bev_point_xz=item.get("bev_point_xz") or item.get("bev_xz") or item.get("bev") or [],
                    bev_y=float(item.get("bev_y") or 0.0),
                    camera_id=item.get("camera_id") or item.get("camera"),
                    room=item.get("room"),
                    evidence=_seq(item.get("evidence")),
                )
            )
        report.add_check(validate_world_bev_round_trip(world_to_bev, world_bev_samples))
    world_to_menon = menon.get("world_to_menon_col_major") if isinstance(menon, Mapping) else None
    if world_to_menon is not None:
        placements: list[MenonPlacementSample] = []
        for item in _seq(menon.get("placements")):
            if not isinstance(item, Mapping):
                continue
            placements.append(
                MenonPlacementSample(
                    entity_id=str(item.get("entity_id") or item.get("id") or "entity"),
                    world_point=item.get("world_point") or item.get("world") or [],
                    menon_point=item.get("menon_point") or item.get("menon") or [],
                    camera_id=item.get("camera_id") or item.get("camera"),
                    room=item.get("room"),
                    evidence=_seq(item.get("evidence")),
                )
            )
        report.add_check(validate_world_menon_round_trip(world_to_menon, placements))
        report.add_check(validate_world_menon_placement_agreement(world_to_menon, placements))
        bev_trails: list[MenonBevTrailSample] = []
        for item in _seq(menon.get("bev_trails") or menon.get("bev_menon_trails")):
            if not isinstance(item, Mapping):
                continue
            bev_trails.append(
                MenonBevTrailSample(
                    entity_id=str(item.get("entity_id") or item.get("id") or "entity"),
                    bev_points_xz=item.get("bev_points_xz") or item.get("bev_points") or item.get("bev_path_xz") or [],
                    menon_points=item.get("menon_points") or item.get("menon_scene_points") or item.get("scene_points") or [],
                    floor_y_world=float(item.get("floor_y_world") or item.get("world_floor_y") or 0.0),
                    camera_id=item.get("camera_id") or item.get("camera"),
                    room=item.get("room"),
                    evidence=_seq(item.get("evidence")),
                )
            )
        if bev_trails or "bev_trails" in menon or "bev_menon_trails" in menon:
            report.add_check(validate_bev_menon_trail_agreement(world_to_menon, bev_trails))
    avatar_samples: list[MenonAvatarSample] = []
    if isinstance(menon, Mapping):
        for item in _seq(menon.get("avatars") or menon.get("avatar_samples")):
            if not isinstance(item, Mapping):
                continue
            avatar_samples.append(
                MenonAvatarSample(
                    entity_id=str(item.get("entity_id") or item.get("id") or "entity"),
                    menon_point=item.get("menon_point") or item.get("menon_scene") or item.get("scene") or item.get("position") or [],
                    radius_scene=_first_present(item, "radius_scene", "radius", "collision_radius_scene"),
                    expected_height_m=_first_present(item, "expected_height_m", "person_height_m"),
                    observed_height_scene=_first_present(item, "observed_height_scene", "avatar_height_scene", "height_scene"),
                    scene_units_per_m=float(item.get("scene_units_per_m") or 1.0),
                    heading_deg=_first_present(item, "heading_deg", "yaw_deg", "orientation_deg"),
                    movement_vector_xz=_first_present(item, "movement_vector_xz", "velocity_xz", "trail_delta_xz"),
                    allowed_collision=bool(item.get("allowed_collision", False)),
                    camera_id=item.get("camera_id") or item.get("camera"),
                    room=item.get("room"),
                    evidence=_seq(item.get("evidence")),
                )
            )
    if avatar_samples or "avatars" in menon or "avatar_samples" in menon:
        collision = menon.get("collision") if isinstance(menon.get("collision"), Mapping) else {}
        avatar_walls = _wall_segments(collision.get("wall_segments_xz") or menon.get("wall_segments_xz") or wall_segments_xz)
        avatar_obstacles = _seq(collision.get("obstacle_boxes_xz") or menon.get("obstacle_boxes_xz"))
        report.add_check(validate_menon_avatar_scale(avatar_samples))
        report.add_check(validate_menon_avatar_collision(avatar_samples, avatar_walls, avatar_obstacles))
        report.add_check(validate_menon_avatar_orientation(avatar_samples))
    object_samples: list[MenonObjectSample] = []
    if isinstance(menon, Mapping):
        for item in _seq(menon.get("objects") or menon.get("object_samples")):
            if not isinstance(item, Mapping):
                continue
            object_samples.append(
                MenonObjectSample(
                    object_id=str(item.get("object_id") or item.get("id") or "object"),
                    category=str(item.get("category") or item.get("class") or "object"),
                    world_point=_first_present(item, "world_point", "world", "noesis_world"),
                    menon_point=item.get("menon_point") or item.get("menon_scene") or item.get("scene") or item.get("position") or [],
                    bbox_min_scene=_first_present(item, "bbox_min_scene", "bboxMinScene", "bbox_min", "bboxMin"),
                    bbox_max_scene=_first_present(item, "bbox_max_scene", "bboxMaxScene", "bbox_max", "bboxMax"),
                    expected_dimensions_m=_first_present(item, "expected_dimensions_m", "expectedDimensionsM", "expected_dims_m", "expectedDimsM"),
                    observed_dimensions_scene=_first_present(item, "observed_dimensions_scene", "observedDimensionsScene", "dims_scene", "dimsScene"),
                    scene_units_per_m=float(item.get("scene_units_per_m") or 1.0),
                    support_y_scene=float(item.get("support_y_scene") or item.get("supportYScene") or 0.0),
                    allowed_rooms=_string_seq(_first_present(item, "allowed_rooms", "allowedRooms", "compatible_rooms", "compatibleRooms")),
                    allowed_collision=bool(item.get("allowed_collision", False)),
                    room=item.get("room"),
                    camera_id=item.get("camera_id") or item.get("camera"),
                    evidence=_seq(item.get("evidence")),
                )
            )
    if object_samples or "objects" in menon or "object_samples" in menon:
        collision = menon.get("collision") if isinstance(menon.get("collision"), Mapping) else {}
        object_walls = _wall_segments(collision.get("wall_segments_xz") or menon.get("wall_segments_xz") or wall_segments_xz)
        object_obstacles = _seq(collision.get("object_obstacle_boxes_xz") or collision.get("obstacle_boxes_xz") or menon.get("obstacle_boxes_xz"))
        report.add_check(validate_menon_object_placement(world_to_menon, object_samples, object_walls, object_obstacles))
    latency_samples: list[MenonLatencySample] = []
    if isinstance(menon, Mapping):
        for item in _seq(menon.get("latency_samples") or menon.get("latency")):
            if not isinstance(item, Mapping):
                continue
            latency_samples.append(
                MenonLatencySample(
                    entity_id=str(item.get("entity_id") or item.get("id") or "entity"),
                    noesis_ts_s=float(_first_present(item, "noesis_ts_s", "frame_ts_s", "source_ts_s") or 0.0),
                    telemetry_ts_s=_first_present(item, "telemetry_ts_s", "telemetryTsS"),
                    menon_update_ts_s=_first_present(item, "menon_update_ts_s", "update_ts_s", "menonUpdateTsS"),
                    menon_render_ts_s=_first_present(item, "menon_render_ts_s", "render_ts_s", "menonRenderTsS"),
                    menon_display_ts_s=_first_present(item, "menon_display_ts_s", "display_ts_s", "menonDisplayTsS"),
                    camera_id=item.get("camera_id") or item.get("camera"),
                    room=item.get("room"),
                    evidence=_seq(item.get("evidence")),
                )
            )
    if latency_samples or "latency_samples" in menon or "latency" in menon:
        report.add_check(validate_menon_latency_alignment(latency_samples))
    if isinstance(menon, Mapping) and ("noesis_ts_s" in menon or "menon_ts_s" in menon):
        report.add_check(validate_timestamp_alignment(_seq(menon.get("noesis_ts_s")), _seq(menon.get("menon_ts_s"))))

    if artifact_index is not None:
        index_path = artifact_index.path("visual/index.json")
        artifact_index.add("visual_index", index_path)
        artifact_index.write("visual/index.json")
        report.artifacts = artifact_index.artifacts

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixture-based Noesis/Menon validation checks.")
    parser.add_argument("--fixture", default="", help="JSON fixture containing validation inputs.")
    parser.add_argument("--fixture-registry", default="", help="JSON fixture registry.")
    parser.add_argument("--fixture-id", default="", help="Fixture id to resolve from --fixture-registry.")
    parser.add_argument("--output-dir", default="diagnostics/validation", help="Output directory for reports.")
    parser.add_argument("--run-id", default="", help="Override report run id.")
    args = parser.parse_args()

    if args.fixture_id:
        if not args.fixture_registry:
            parser.error("--fixture-id requires --fixture-registry")
        fixture_path = FixtureRegistry.load(args.fixture_registry).resolve(args.fixture_id)
    elif args.fixture:
        fixture_path = Path(args.fixture)
    else:
        parser.error("provide --fixture or --fixture-id with --fixture-registry")
    payload = _load_json(fixture_path)
    run_id = str(args.run_id or payload.get("run_id") or fixture_path.stem)
    output_dir = Path(args.output_dir) / run_id
    report = _run_fixture(payload, run_id=run_id, artifact_dir=output_dir, fixture_dir=fixture_path.parent)
    json_path = report.write_json(output_dir / "validation_report.json")
    md_path = write_markdown(report, output_dir / "validation_report.md")
    print(json.dumps({"status": report.status.value, "level": report.level.value, "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 1 if report.status.value == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
