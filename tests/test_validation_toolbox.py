from __future__ import annotations

import json
import copy
import os
from pathlib import Path

import numpy as np
import pytest

from noesis.validation.camera import (
    CameraCalibration,
    ReprojectionAnchor,
    validate_extrinsics,
    validate_intrinsics,
    validate_reprojection_anchors,
)
from noesis.validation.bev import validate_camera_coverage, validate_path_smoothness, validate_zone_consistency
from noesis.validation.core import CheckStatus, SourceMetadata, ValidationCheck, ValidationReport
from noesis.validation.fixtures import FixtureRegistry
from noesis.validation.golden import compare_artifact_expectations, compare_image_files
from noesis.validation.geometry import AnchorObservation, validate_known_anchors, validate_no_wall_crossing, validate_points_inside_room
from noesis.validation.menon import (
    MenonAvatarSample,
    MenonBevTrailSample,
    MenonCameraReprojectionSample,
    MenonLatencySample,
    MenonObjectSample,
    MenonPlacementSample,
    MenonTrailSample,
    MenonTransformAuditSample,
    WorldBevSample,
    validate_bev_menon_trail_agreement,
    validate_menon_avatar_collision,
    validate_menon_avatar_orientation,
    validate_menon_avatar_scale,
    validate_menon_camera_reprojection,
    validate_menon_floor_contact,
    validate_menon_latency_alignment,
    validate_menon_object_placement,
    validate_menon_trail_agreement,
    validate_timestamp_alignment,
    validate_transform_audit,
    validate_world_bev_round_trip,
    validate_world_menon_placement_agreement,
)
from noesis.validation.menon_trace import build_menon_trace_report, load_menon_trace
from noesis.validation.menon_browser import browser_snapshot_to_menon_trace
from noesis.validation.reports import render_markdown
from noesis.validation.scene import (
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
from noesis.validation.telemetry import build_telemetry_report, build_track_audit, extract_telemetry_samples, read_ndjson
from noesis.validation.tracking import (
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
from noesis.validation.transforms import matrix_to_col_major, validate_round_trip, validate_transform_matrix
from scripts.noesis_validation_runner import _run_fixture
from scripts.noesis_validation_menon_trace_report import run_trace_report
from scripts.noesis_validation_capture_menon_trace import (
    validate_browser_auth_proof,
    validate_storage_state_path,
    write_trace_from_snapshot,
)
from scripts.noesis_validation_regression_runner import _failure_categories, run_regression_suite
from scripts.noesis_validation_telemetry_report import _load_messages


def _camera_extrinsics_col_major() -> list[float]:
    # Camera center is 2m above the floor. Camera +Z points downward, and +Y
    # points toward positive world Z, so floor rays are valid and deterministic.
    twc = np.eye(4, dtype=np.float64)
    twc[:3, :3] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    twc[:3, 3] = [0.0, 2.0, 0.0]
    return matrix_to_col_major(np.linalg.inv(twc))


def test_validation_report_serializes_summary(tmp_path: Path) -> None:
    report = ValidationReport(
        run_id="unit",
        source=SourceMetadata(pipeline_config="config/infer.yaml"),
        checks=[
            ValidationCheck(id="A", domain="core", name="a", status=CheckStatus.PASS, detail="ok"),
            ValidationCheck(id="B", domain="core", name="b", status=CheckStatus.WARNING, detail="warn"),
        ],
    )
    output = report.write_json(tmp_path / "report.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "warning"
    assert payload["summary"]["pass_count"] == 1
    assert payload["summary"]["warning_count"] == 1
    assert "Validation Report" in render_markdown(report)


def test_transform_validation_and_round_trip_pass() -> None:
    matrix = matrix_to_col_major(np.eye(4, dtype=np.float64))
    checks = validate_transform_matrix(matrix, check_id="T")
    assert [check.status for check in checks] == [CheckStatus.PASS, CheckStatus.PASS]
    round_trip = validate_round_trip([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], np.eye(4), check_id="T.round")
    assert round_trip.status == CheckStatus.PASS


def test_anchor_validation_classifies_warning_and_fail() -> None:
    checks = validate_known_anchors(
        [
            AnchorObservation("good", [0, 0, 0], [0.05, 0, 0], anchor_type="room"),
            AnchorObservation("warn", [0, 0, 0], [0.15, 0, 0], anchor_type="room"),
            AnchorObservation("fail", [0, 0, 0], [0.35, 0, 0], anchor_type="room"),
        ]
    )
    assert [check.status for check in checks] == [CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL]


def test_camera_intrinsics_extrinsics_and_reprojection_pass() -> None:
    calibration = CameraCalibration(
        camera_id="test-camera",
        intrinsics=np.asarray([[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        extrinsics_col_major=_camera_extrinsics_col_major(),
        image_size=(640, 480),
        floor_y=0.0,
        unit_scale=1.0,
        stream_kind="dewarped",
        applies_to_raw=False,
        applies_to_dewarped=True,
    )
    assert all(check.status == CheckStatus.PASS for check in validate_intrinsics(calibration, expected_resolution=(640, 480)))
    assert all(check.status == CheckStatus.PASS for check in validate_extrinsics(calibration))
    checks = validate_reprojection_anchors(
        calibration,
        [ReprojectionAnchor("floor-center", [0.0, 0.0, 0.0], [320.0, 240.0])],
    )
    assert checks[0].status == CheckStatus.PASS


def test_dewarped_intrinsics_scope_requires_explicit_dewarped_contract() -> None:
    calibration = CameraCalibration(
        camera_id="ambiguous-dewarped-camera",
        intrinsics=np.asarray([[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        extrinsics_col_major=_camera_extrinsics_col_major(),
        image_size=(640, 480),
        stream_kind="dewarped",
        applies_to_raw=True,
        applies_to_dewarped=None,
    )
    checks = validate_intrinsics(calibration, expected_resolution=(640, 480))
    scope = next(check for check in checks if check.id == "CAM.intrinsics.dewarped_scope")
    resolution = next(check for check in checks if check.id == "CAM.intrinsics.resolution")
    assert resolution.status == CheckStatus.PASS
    assert scope.status == CheckStatus.FAIL
    assert scope.metric["applies_to_raw"] is True
    assert scope.metric["applies_to_dewarped"] is None


def test_track_motion_and_bev_agreement_detect_clean_path() -> None:
    tracks = [
        TrackSample(ts_s=0.0, world=[0.0, 0.0, 0.0], camera_id="cam", stable_id=1, tracker_id=101, reid_identity="person-a"),
        TrackSample(
            ts_s=1.0,
            world=[0.5, 0.0, 0.0],
            camera_id="cam",
            stable_id=1,
            tracker_id=101,
            reid_identity="person-a",
            occluded=True,
            occlusion_uncertainty_m=0.75,
        ),
        TrackSample(ts_s=2.0, world=[1.0, 0.0, 0.0], camera_id="cam", stable_id=1, tracker_id=101, reid_identity="person-a"),
    ]
    motion = validate_track_motion(tracks)
    assert motion[0].status == CheckStatus.PASS
    assert motion[1].status == CheckStatus.PASS
    assert validate_occlusion_bridges(tracks).status == CheckStatus.PASS
    assert validate_identity_continuity(tracks).status == CheckStatus.PASS
    assert validate_reid_geometry_consistency(tracks).status == CheckStatus.PASS
    assert validate_path_smoothness(tracks).status == CheckStatus.PASS
    doorway_tracks = [
        TrackSample(ts_s=0.0, world=[0.0, 0.0, -0.5], camera_id="cam", stable_id=1, room="a"),
        TrackSample(ts_s=1.0, world=[0.0, 0.0, 0.5], camera_id="cam", stable_id=1, room="b"),
    ]
    assert validate_doorway_transitions(doorway_tracks, [([-0.5, 0.0], [0.5, 0.0])]).status == CheckStatus.PASS
    bev = [
        BevPointSample(ts_s=0.0, x=0.0, z=0.0, camera_id="cam", stable_id=1),
        BevPointSample(ts_s=1.0, x=0.5, z=0.0, camera_id="cam", stable_id=1),
        BevPointSample(ts_s=2.0, x=1.0, z=0.0, camera_id="cam", stable_id=1),
    ]
    assert validate_bev_track_agreement(tracks, bev).status == CheckStatus.PASS
    room_polygons = {"fixture-room": [[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]]}
    frustums = {"cam": [[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]]}
    room_tracks = [TrackSample(ts_s=t.ts_s, world=t.world, camera_id=t.camera_id, stable_id=t.stable_id, room="fixture-room") for t in tracks]
    assert validate_zone_consistency(room_tracks, room_polygons).status == CheckStatus.PASS
    assert validate_camera_coverage(room_tracks, frustums).status == CheckStatus.PASS


def test_bev_path_smoothness_flags_zigzag() -> None:
    tracks = [
        TrackSample(ts_s=0.0, world=[0.0, 0.0, 0.0], camera_id="cam", stable_id=1),
        TrackSample(ts_s=1.0, world=[1.0, 0.0, 0.0], camera_id="cam", stable_id=1),
        TrackSample(ts_s=2.0, world=[0.0, 0.0, 0.0], camera_id="cam", stable_id=1),
        TrackSample(ts_s=3.0, world=[1.0, 0.0, 0.0], camera_id="cam", stable_id=1),
    ]
    check = validate_path_smoothness(tracks)
    assert check.status == CheckStatus.FAIL
    assert check.metric["p95_turn_deg"] == 180.0


def test_identity_and_reid_validators_detect_contradictions() -> None:
    stable_switch = [
        TrackSample(ts_s=0.0, world=[0.0, 0.0, 0.0], camera_id="cam", stable_id=1, tracker_id=10, reid_identity="person-a"),
        TrackSample(ts_s=1.0, world=[0.2, 0.0, 0.0], camera_id="cam", stable_id=2, tracker_id=10, reid_identity="person-a"),
    ]
    continuity = validate_identity_continuity(stable_switch)
    assert continuity.status == CheckStatus.FAIL
    assert continuity.metric["tracker_stable_switch_count"] == 1

    duplicate_stable = [
        TrackSample(ts_s=1.0, world=[0.0, 0.0, 0.0], camera_id="cam-a", stable_id=7, tracker_id=10),
        TrackSample(ts_s=1.05, world=[2.0, 0.0, 0.0], camera_id="cam-b", stable_id=7, tracker_id=20),
    ]
    split_merge = validate_identity_continuity(duplicate_stable)
    assert split_merge.status == CheckStatus.FAIL
    assert split_merge.metric["duplicate_stable_far_event_count"] == 1

    reid_conflict = [
        TrackSample(ts_s=0.0, world=[0.0, 0.0, 0.0], camera_id="cam", stable_id=3, tracker_id=30, reid_identity="person-a"),
        TrackSample(ts_s=1.0, world=[0.2, 0.0, 0.0], camera_id="cam", stable_id=3, tracker_id=30, reid_identity="person-b"),
    ]
    reid = validate_reid_geometry_consistency(reid_conflict)
    assert reid.status == CheckStatus.FAIL
    assert reid.metric["stable_reid_conflict_count"] == 1


def test_track_motion_flags_teleport() -> None:
    tracks = [
        TrackSample(ts_s=0.0, world=[0.0, 0.0, 0.0], camera_id="cam", stable_id=1),
        TrackSample(ts_s=0.1, world=[5.0, 0.0, 0.0], camera_id="cam", stable_id=1),
    ]
    assert validate_track_motion(tracks)[0].status == CheckStatus.FAIL
    audit = build_track_audit(tracks)
    assert audit[0]["last_impossible_motion_event"]["type"] == "speed_limit"
    assert "impossible_motion:speed_limit" in audit[0]["warnings"]


def test_detection_projection_and_confidence_pass() -> None:
    samples = [
        DetectionProjectionSample(
            "p1",
            [0.5, 0.0, 0.0],
            bbox_foot_world=[0.49, 0.0, 0.01],
            mask_foot_world=[0.51, 0.0, -0.01],
            pose_foot_world=[0.5, 0.0, 0.02],
            depth_world=[0.52, 0.0, 0.0],
            image_bbox_xyxy=[300.0, 100.0, 380.0, 420.0],
            projected_bbox_xyxy=[304.0, 106.0, 376.0, 416.0],
            reprojection_score=0.94,
            temporal_smoothness_score=0.92,
            semantic_validity_score=0.95,
            floor_y=0.0,
            person_height_m=1.72,
            ray_floor_valid=True,
            room_polygon_xz=[[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]],
        )
    ]
    assert validate_detection_world_projection(samples).status == CheckStatus.PASS
    confidence = validate_projection_confidence(samples)
    assert confidence.status == CheckStatus.PASS
    assert confidence.metric["score"] > 0.9
    assert confidence.metric["component_counts"]["temporal_smoothness"] == 1
    assert confidence.metric["component_counts"]["semantic_validity"] == 1


def test_detection_projection_flags_bbox_aspect_and_overlap_failures() -> None:
    samples = [
        DetectionProjectionSample(
            "p1",
            [0.5, 0.0, 0.0],
            bbox_foot_world=[0.49, 0.0, 0.01],
            image_bbox_xyxy=[10.0, 10.0, 210.0, 60.0],
            projected_bbox_xyxy=[300.0, 300.0, 340.0, 380.0],
            floor_y=0.0,
            person_height_m=1.72,
            ray_floor_valid=True,
            room_polygon_xz=[[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]],
        )
    ]
    check = validate_detection_world_projection(samples)
    assert check.status == CheckStatus.FAIL
    assert check.metric["bbox_aspect_failure_count"] == 1
    assert check.metric["projected_bbox_failure_count"] == 1
    assert check.metric["p95_bbox_center_error_px"] > 50.0


def test_projection_confidence_uses_explicit_temporal_and_semantic_scores() -> None:
    samples = [
        DetectionProjectionSample(
            "p1",
            [0.5, 0.0, 0.0],
            bbox_foot_world=[0.49, 0.0, 0.01],
            floor_y=0.0,
            person_height_m=1.72,
            ray_floor_valid=True,
            room_polygon_xz=[[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]],
            reprojection_score=0.05,
            temporal_smoothness_score=0.05,
            semantic_validity_score=0.05,
        )
    ]
    confidence = validate_projection_confidence(samples)
    assert confidence.status == CheckStatus.WARNING
    assert confidence.metric["component_counts"]["reprojection"] == 1
    assert confidence.metric["component_counts"]["temporal_smoothness"] == 1
    assert confidence.metric["component_counts"]["semantic_validity"] == 1


def test_menon_placement_agreement_passes() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [10.0, 0.0, 0.0]
    samples = [MenonPlacementSample("p1", [1.0, 0.0, 2.0], [11.0, 0.0, 2.0])]
    check = validate_world_menon_placement_agreement(matrix_to_col_major(matrix), samples)
    assert check.status == CheckStatus.PASS


def test_world_bev_round_trip_passes_and_flags_scale_drift() -> None:
    identity = matrix_to_col_major(np.eye(4, dtype=np.float64))
    samples = [
        WorldBevSample("p1", [0.0, 0.0, 0.0], [0.0, 0.0]),
        WorldBevSample("p1", [0.5, 0.0, 0.2], [0.5, 0.2]),
    ]
    assert validate_world_bev_round_trip(identity, samples).status == CheckStatus.PASS

    drifted = [WorldBevSample("p1", [0.5, 0.0, 0.2], [1.5, 0.2])]
    check = validate_world_bev_round_trip(identity, drifted)
    assert check.status == CheckStatus.FAIL
    assert check.metric["p95_agreement_m"] > 0.5


def test_menon_trace_validators_pass_clean_trace() -> None:
    matrix = matrix_to_col_major(np.eye(4, dtype=np.float64))
    placements = [
        MenonPlacementSample("p1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        MenonPlacementSample("p1", [0.5, 0.0, 0.2], [0.5, 0.0, 0.2]),
    ]
    trails = [MenonTrailSample("p1", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.2]], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.2]])]
    audits = [
        MenonTransformAuditSample(
            "p1",
            [
                {"stage": "source_payload", "frame": "backend_world_m"},
                {"stage": "scene_similarity", "from_frame": "backend_world_m", "to_frame": "menon_scene"},
                {"stage": "render_anchor", "frame": "menon_scene"},
            ],
        )
    ]
    assert validate_menon_floor_contact(placements).status == CheckStatus.PASS
    assert validate_menon_trail_agreement(matrix, trails).status == CheckStatus.PASS
    bev_trails = [MenonBevTrailSample("p1", [[0.0, 0.0], [0.5, 0.2]], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.2]])]
    assert validate_bev_menon_trail_agreement(matrix, bev_trails).status == CheckStatus.PASS
    avatars = [
        MenonAvatarSample(
            "p1",
            [0.5, 0.0, 0.2],
            radius_scene=0.18,
            expected_height_m=1.72,
            observed_height_scene=1.70,
            heading_deg=22.0,
            movement_vector_xz=[0.5, 0.2],
        )
    ]
    walls = [([-1.0, -1.0], [2.0, -1.0]), ([2.0, -1.0], [2.0, 1.0]), ([2.0, 1.0], [-1.0, 1.0]), ([-1.0, 1.0], [-1.0, -1.0])]
    obstacles = [{"bbox_min_xz": [1.2, -0.5], "bbox_max_xz": [1.6, -0.2]}]
    assert validate_menon_avatar_scale(avatars).status == CheckStatus.PASS
    assert validate_menon_avatar_collision(avatars, walls, obstacles).status == CheckStatus.PASS
    assert validate_menon_avatar_orientation(avatars).status == CheckStatus.PASS
    objects = [
        MenonObjectSample(
            "couch",
            "couch",
            menon_point=[0.45, 0.0, 0.15],
            world_point=[0.45, 0.0, 0.15],
            bbox_min_scene=[0.25, 0.0, -0.05],
            bbox_max_scene=[0.95, 0.8, 0.35],
            expected_dimensions_m=[0.70, 0.80, 0.40],
            observed_dimensions_scene=[0.70, 0.80, 0.40],
            support_y_scene=0.0,
            allowed_rooms=["fixture-room"],
            room="fixture-room",
        )
    ]
    object_walls = [([-1.0, -1.0], [2.0, -1.0]), ([2.0, -1.0], [2.0, 1.0]), ([2.0, 1.0], [-1.0, 1.0]), ([-1.0, 1.0], [-1.0, -1.0])]
    assert validate_menon_object_placement(matrix, objects, object_walls, []).status == CheckStatus.PASS
    assert validate_transform_audit(audits).status == CheckStatus.PASS
    camera_reprojection = [
        MenonCameraReprojectionSample(
            "fixture-camera",
            source_frame="visual/source.png",
            menon_render="visual/render.png",
            overlay_path="visual/overlay.png",
            layers=["detected_bbox", "projected_avatar", "room_mesh_edges", "floor_grid", "anchors"],
            anchor_mean_error_px=6.0,
            anchor_max_error_px=14.0,
            floor_grid_mean_error_px=7.5,
            room_edge_mean_error_px=8.0,
            bbox_iou=0.68,
            avatar_iou=0.71,
        )
    ]
    assert validate_menon_camera_reprojection(camera_reprojection).status == CheckStatus.PASS
    assert validate_timestamp_alignment([1.0, 2.0], [1.02, 2.03]).status == CheckStatus.PASS
    latency = [
        MenonLatencySample(
            "p1",
            noesis_ts_s=10.0,
            telemetry_ts_s=10.01,
            menon_update_ts_s=10.03,
            menon_render_ts_s=10.05,
            menon_display_ts_s=10.06,
        )
    ]
    latency_check = validate_menon_latency_alignment(latency)
    assert latency_check.status == CheckStatus.PASS
    assert latency_check.metric["display_sample_count"] == 1


def test_menon_object_placement_flags_room_support_and_collision_failures() -> None:
    matrix = matrix_to_col_major(np.eye(4, dtype=np.float64))
    objects = [
        MenonObjectSample(
            "bad-couch",
            "couch",
            menon_point=[1.2, 0.0, 0.0],
            world_point=[0.0, 0.0, 0.0],
            bbox_min_scene=[0.8, 0.35, -0.2],
            bbox_max_scene=[1.6, 1.1, 0.2],
            expected_dimensions_m=[0.70, 0.80, 0.40],
            observed_dimensions_scene=[1.10, 0.75, 0.40],
            support_y_scene=0.0,
            allowed_rooms=["living-room"],
            room="kitchen",
        )
    ]
    walls = [([1.0, -1.0], [1.0, 1.0])]
    obstacles = [{"bbox_min_xz": [1.1, -0.1], "bbox_max_xz": [1.4, 0.1]}]
    check = validate_menon_object_placement(matrix, objects, walls, obstacles)
    assert check.status == CheckStatus.FAIL
    assert check.metric["room_incompatibility_count"] == 1
    assert check.metric["wall_intersection_count"] == 1
    assert check.metric["obstacle_overlap_count"] == 1
    assert check.metric["p95_placement_error_scene_units"] > 0.5
    assert check.metric["p95_support_error_scene_units"] > 0.2


def test_menon_latency_alignment_flags_display_lag() -> None:
    samples = [
        MenonLatencySample(
            "p1",
            noesis_ts_s=10.0,
            telemetry_ts_s=10.1,
            menon_update_ts_s=10.4,
            menon_render_ts_s=10.9,
            menon_display_ts_s=11.2,
        )
    ]
    check = validate_menon_latency_alignment(samples)
    assert check.status == CheckStatus.FAIL
    assert check.metric["p95_total_lag_s"] > 0.75
    assert check.metric["p95_queue_lag_s"] > 0.35


def test_room_bounds_and_wall_crossing_validators() -> None:
    polygon = [[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]]
    inside = validate_points_inside_room([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], polygon, min_inside_ratio=1.0)
    assert inside.status == CheckStatus.PASS
    outside = validate_points_inside_room([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], polygon, min_inside_ratio=1.0)
    assert outside.status == CheckStatus.WARNING

    wall = [([0.5, -1.0], [0.5, 1.0])]
    assert validate_no_wall_crossing([[0.0, 0.0], [0.4, 0.0]], wall).status == CheckStatus.PASS
    assert validate_no_wall_crossing([[0.0, 0.0], [1.0, 0.0]], wall).status == CheckStatus.FAIL


def test_scene_geometry_mesh_depth_and_semantic_validators_pass() -> None:
    coordinates = [
        SceneCoordinateSystemObservation(
            "scene",
            units="m",
            axis_convention="x_right_y_up_z_forward",
            expected_axis_convention="x_right_y_up_z_forward",
            camera_pose_convention="world_to_camera",
            expected_camera_pose_convention="world_to_camera",
            origin_delta_m=0.02,
            transform_col_major=matrix_to_col_major(np.eye(4, dtype=np.float64)),
            round_trip_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            scale_anchor_expected_m=3.0,
            scale_anchor_observed_m=3.02,
        )
    ]
    assert validate_scene_coordinate_system(coordinates).status == CheckStatus.PASS
    planes = [
        PlaneObservation("floor", "floor", [0.0, 1.0, 0.0], rms_deviation_m=0.01),
        PlaneObservation("wall", "wall", [1.0, 0.0, 0.0], rms_deviation_m=0.02),
        PlaneObservation("ceiling", "ceiling", [0.0, -1.0, 0.0], rms_deviation_m=0.01),
    ]
    assert validate_plane_geometry(planes).status == CheckStatus.PASS
    dimensions = [RoomDimensionObservation("room", 3.0, 2.0, 2.45, 3.02, 2.01, 2.44)]
    assert validate_room_dimensions(dimensions).status == CheckStatus.PASS
    constraints = [
        RoomGeometryConstraintObservation("wall-floor", "orthogonality", gap_m=0.01, angle_error_deg=1.2),
        RoomGeometryConstraintObservation("door", "doorway", doorway_width_m=0.86, doorway_height_m=2.03, doorway_bottom_gap_m=0.0),
        RoomGeometryConstraintObservation("window", "window", window_width_m=1.10, window_height_m=0.80, window_sill_height_m=0.92),
        RoomGeometryConstraintObservation("junction", "wall_intersection", gap_m=0.02, overlap_m=0.01, normal_dot=0.0),
    ]
    assert validate_room_geometry_constraints(constraints).status == CheckStatus.PASS
    meshes = [
        MeshQualitySample(
            "room-shell",
            [-1.0, 0.0, -1.0],
            [2.0, 2.45, 1.0],
            triangle_count=2400,
            vertex_count=1400,
            watertight_expected=False,
            watertight=False,
            non_manifold_edges=0,
            inverted_faces=0,
            duplicate_faces=0,
            uv_overlap_ratio=0.0,
            lod_max_error_m=0.02,
            texture_alignment_error_px=2.0,
            texture_stretch_ratio=1.05,
            bvh_valid=True,
            collision_bvh_error_m=0.02,
        )
    ]
    assert validate_mesh_quality(meshes).status == CheckStatus.PASS
    menon_assets = [
        MenonAssetObservation(
            "room-shell",
            "room_mesh",
            scale_error_m=0.02,
            orientation_error_deg=1.0,
            origin_error_m=0.02,
            collision_mesh_error_m=0.01,
        ),
        MenonAssetObservation(
            "floor",
            "floor_mesh",
            floor_flatness_m=0.01,
            floor_alignment_error_m=0.02,
            walkable=True,
        ),
        MenonAssetObservation("wall", "wall_mesh", doorway_blocked=False),
        MenonAssetObservation(
            "camera-marker",
            "camera_marker",
            camera_position_error_m=0.04,
            camera_aim_error_deg=1.5,
        ),
        MenonAssetObservation(
            "debug-overlays",
            "debug_overlay",
            overlay_rendered_layers=["anchors", "frustums", "floor_grid", "track_trails"],
            required_overlay_layers=["anchors", "frustums", "floor_grid", "track_trails"],
        ),
    ]
    assert validate_menon_asset_sanity(menon_assets).status == CheckStatus.PASS
    depth = [
        DepthAnchorObservation(
            "near-depth",
            expected_depth_m=2.0,
            observed_depth_m=2.04,
            order_group="depth-order",
            expected_order=1,
            plane_residual_m=0.03,
            object_depth_error_m=0.04,
            temporal_std_m=0.02,
            static_std_m=0.02,
            edge_alignment_error_px=3.0,
            confidence=0.82,
            fusion_weight=0.70,
        ),
        DepthAnchorObservation(
            "far-depth",
            expected_depth_m=3.0,
            observed_depth_m=3.08,
            order_group="depth-order",
            expected_order=2,
            plane_residual_m=0.04,
            object_depth_error_m=0.05,
            temporal_std_m=0.03,
            static_std_m=0.025,
            edge_alignment_error_px=4.0,
            confidence=0.76,
            fusion_weight=0.60,
        ),
    ]
    assert validate_depth_consistency(depth).status == CheckStatus.PASS
    objects = [
        SemanticObjectObservation(
            "couch",
            "couch",
            [0.2, 0.0, -0.5],
            [1.4, 0.8, 0.2],
            support_y=0.0,
            max_wall_intersection_m=0.0,
            max_static_shift_m=0.02,
            allowed_rooms=["fixture-room"],
            doorway_clearance_m=0.86,
            free_space_clearance_m=0.55,
            walkable_area_blocked_ratio=0.02,
            known_anchor_error_m=0.05,
            room="fixture-room",
        )
    ]
    assert validate_semantic_object_logic(objects).status == CheckStatus.PASS


def test_menon_asset_sanity_flags_collision_and_overlay_failures() -> None:
    assets = [
        MenonAssetObservation(
            "bad-room-shell",
            "room_mesh",
            scale_error_m=0.35,
            collision_mesh_error_m=0.30,
        ),
        MenonAssetObservation("floor", "floor_mesh", walkable=False),
        MenonAssetObservation("wall", "wall_mesh", doorway_blocked=True),
        MenonAssetObservation(
            "debug-overlays",
            "debug_overlay",
            overlay_rendered_layers=["anchors"],
            required_overlay_layers=["anchors", "frustums", "floor_grid"],
        ),
    ]
    check = validate_menon_asset_sanity(assets)
    assert check.status == CheckStatus.FAIL
    assert check.metric["non_walkable_floor_count"] == 1
    assert check.metric["doorway_blocked_count"] == 1
    assert check.metric["missing_overlay_layer_count"] == 2
    assert check.metric["failure_counts"]["scale_error_m"] == 1
    assert check.metric["failure_counts"]["collision_mesh_error_m"] == 1


def test_room_geometry_constraints_flag_impossible_window() -> None:
    constraints = [
        RoomGeometryConstraintObservation(
            "bad-window",
            "window",
            window_width_m=0.05,
            window_height_m=0.10,
            window_sill_height_m=2.20,
        )
    ]
    check = validate_room_geometry_constraints(constraints)
    assert check.status == CheckStatus.FAIL
    assert check.metric["impossible_window_count"] == 3


def test_mesh_quality_flags_lod_texture_and_bvh_failures() -> None:
    meshes = [
        MeshQualitySample(
            "bad-room-shell",
            [-1.0, 0.0, -1.0],
            [2.0, 2.45, 1.0],
            triangle_count=2400,
            watertight_expected=False,
            non_manifold_edges=0,
            inverted_faces=0,
            duplicate_faces=0,
            uv_overlap_ratio=0.0,
            lod_max_error_m=0.30,
            texture_alignment_error_px=22.0,
            texture_stretch_ratio=2.50,
            bvh_valid=False,
            collision_bvh_error_m=0.35,
        )
    ]
    check = validate_mesh_quality(meshes)
    assert check.status == CheckStatus.FAIL
    assert check.metric["invalid_bvh_count"] == 1
    assert check.metric["max_lod_error_m"] == 0.30
    assert check.metric["max_texture_alignment_error_px"] == 22.0
    assert check.metric["max_texture_stretch_ratio"] == 2.50
    assert check.metric["max_collision_bvh_error_m"] == 0.35


def test_semantic_object_logic_flags_room_doorway_and_walkability_failures() -> None:
    objects = [
        SemanticObjectObservation(
            "blocked-couch",
            "couch",
            [0.2, 0.0, -0.5],
            [1.4, 0.8, 0.2],
            support_y=0.0,
            max_wall_intersection_m=0.0,
            max_static_shift_m=0.02,
            allowed_rooms=["living-room"],
            doorway_clearance_m=0.40,
            free_space_clearance_m=0.10,
            walkable_area_blocked_ratio=0.25,
            known_anchor_error_m=0.50,
            room="kitchen",
        )
    ]
    check = validate_semantic_object_logic(objects)
    assert check.status == CheckStatus.FAIL
    assert check.metric["room_incompatibility_count"] == 1
    assert check.metric["doorway_clearance_failure_count"] == 1
    assert check.metric["free_space_failure_count"] == 1
    assert check.metric["walkable_area_failure_count"] == 1
    assert check.metric["known_anchor_failure_count"] == 1


def test_depth_consistency_flags_ordering_and_confidence_failures() -> None:
    depth = [
        DepthAnchorObservation(
            "near-depth",
            expected_depth_m=2.0,
            observed_depth_m=3.2,
            order_group="depth-order",
            expected_order=1,
            edge_alignment_error_px=24.0,
            confidence=0.20,
            fusion_weight=0.80,
        ),
        DepthAnchorObservation(
            "far-depth",
            expected_depth_m=3.0,
            observed_depth_m=2.1,
            order_group="depth-order",
            expected_order=2,
            confidence=0.80,
            fusion_weight=0.60,
        ),
    ]
    check = validate_depth_consistency(depth)
    assert check.status == CheckStatus.FAIL
    assert check.metric["ordering_inversion_count"] == 1
    assert check.metric["low_confidence_overweighted_count"] == 1
    assert check.metric["max_edge_alignment_error_px"] == 24.0


def test_fixture_registry_resolves_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"fixtures": [{"id": "fixture", "path": "fixture.json", "rooms": ["room"], "cameras": ["cam"]}]}),
        encoding="utf-8",
    )
    registry = FixtureRegistry.load(registry_path)
    assert registry.resolve("fixture") == fixture


def test_fixture_runner_emits_common_report() -> None:
    matrix = matrix_to_col_major(np.eye(4, dtype=np.float64))
    fixture = {
        "run_id": "fixture",
        "transforms": [{"id": "T", "matrix_col_major": matrix, "points": [[0, 0, 0], [1, 2, 3]]}],
        "anchors": [{"anchor_id": "a", "expected_world": [0, 0, 0], "observed_world": [0.05, 0, 0], "anchor_type": "room"}],
        "tracks": [
            {"ts_s": 0.0, "world": [0, 0, 0], "camera_id": "cam", "stable_id": 1},
            {"ts_s": 1.0, "world": [0.5, 0, 0], "camera_id": "cam", "stable_id": 1},
        ],
        "bev_points": [
            {"ts_s": 0.0, "x": 0, "z": 0, "camera_id": "cam", "stable_id": 1},
            {"ts_s": 1.0, "x": 0.5, "z": 0, "camera_id": "cam", "stable_id": 1},
        ],
    }
    report = _run_fixture(fixture, run_id="fixture")
    payload = report.to_dict()
    assert payload["summary"]["check_count"] >= 6
    assert payload["summary"]["failure_count"] == 0


def test_fixture_runner_writes_visual_artifacts(tmp_path: Path) -> None:
    fixture = json.loads(Path("plans/noesis_menon_validation/minimal_fixture.json").read_text(encoding="utf-8"))
    report = _run_fixture(
        fixture,
        run_id="fixture_visual",
        artifact_dir=tmp_path / "fixture_visual",
        fixture_dir=Path("plans/noesis_menon_validation"),
    )
    payload = report.to_dict()
    assert payload["summary"]["failure_count"] == 0
    assert (tmp_path / "fixture_visual" / "visual" / "camera_fixture-camera_reprojection.png").is_file()
    bev_overlay = tmp_path / "fixture_visual" / "visual" / "bev_diagnostic.png"
    assert bev_overlay.is_file()
    assert bev_overlay.stat().st_size > 0
    assert (tmp_path / "fixture_visual" / "visual" / "index.json").is_file()
    assert "camera_fixture-camera_reprojection_overlay" in payload["artifacts"]
    assert "bev_diagnostic_overlay" in payload["artifacts"]


def test_telemetry_extraction_and_report_from_ndjson() -> None:
    messages = read_ndjson("plans/noesis_menon_validation/minimal_telemetry.ndjson")
    samples = extract_telemetry_samples(messages)
    assert samples.tracking_message_count == 3
    assert samples.bev_message_count == 3
    assert len(samples.track_samples) == 3
    assert len(samples.bev_points) == 3
    report = build_telemetry_report(messages, run_id="telemetry_fixture")
    payload = report.to_dict()
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["blocked_count"] == 0
    assert any(check["id"] == "TELEMETRY.tracking.world_frame" for check in payload["checks"])
    assert any(check["id"] == "BEV.track_agreement" for check in payload["checks"])
    assert any(check["id"] == "TRACK.occlusion_bridge" for check in payload["checks"])
    audit = build_track_audit(samples.track_samples)
    assert audit[0]["stable_id"] == 1
    assert audit[0]["sample_count"] == 3
    assert audit[0]["projection_confidence"] == 0.85
    assert audit[0]["temporal_confidence"] == 0.94
    assert audit[0]["reid_confidence"] == 0.8
    assert audit[0]["appearance_key"] == "person-fixture"
    assert audit[0]["last_occlusion_age_s"] == 1.0
    assert audit[0]["last_impossible_motion_event"] is None
    assert audit[0]["warnings"] == []


def test_telemetry_cli_loader_accepts_input() -> None:
    class Args:
        input = ["plans/noesis_menon_validation/minimal_telemetry.ndjson"]
        ws = ""
        duration = 1.0

    messages = _load_messages(Args())
    assert len(messages) == 6


def test_menon_trace_fixture_report_passes() -> None:
    payload = load_menon_trace("plans/noesis_menon_validation/minimal_menon_trace.json")
    report = build_menon_trace_report(payload, run_id="menon_trace_fixture")
    summary = report.to_dict()["summary"]
    assert summary["failure_count"] == 0
    assert summary["blocked_count"] == 0
    assert summary["pass_count"] >= 10


def test_menon_browser_snapshot_converts_to_trace(tmp_path: Path) -> None:
    fixture_path = Path("plans/noesis_menon_validation/minimal_menon_browser_snapshot.json")
    fixture_text = fixture_path.read_text(encoding="utf-8")
    snapshot = json.loads(fixture_text)
    trace = browser_snapshot_to_menon_trace(
        snapshot,
        run_id="menon_browser_fixture",
        page_url="http://127.0.0.1:5175",
        raw_snapshot_path="browser/browser_snapshot.json",
    )
    assert len(trace["placements"]) == 1
    assert {item["entity_id"] for item in trace["placements"]} == {"resident:resident-a"}
    assert all(item["transform_audit"][0]["frame"] == "backend_world_m" for item in trace["placements"])
    assert trace["browser"]["projection_mode"] == "canonical_world_snapshot"
    assert trace["browser"]["canonical_admission"] == "accepted"
    assert trace["browser"]["authenticated_role"] == "owner"
    assert trace["source"]["runtime"] == "ds9"
    assert trace["source"]["pipeline_config"] == "DS9/config/infer.yaml"
    assert "backendWorldPosition" in fixture_text
    assert "backendWorldRaw" not in fixture_text
    assert "frontend_reproject" not in fixture_text
    assert trace["world_to_menon_col_major"] == matrix_to_col_major(np.eye(4, dtype=np.float64))
    assert len(trace["camera_reprojections"]) == 1
    assert trace["camera_reprojections"][0]["bbox_iou"] == 0.68
    report = build_menon_trace_report(trace, run_id="menon_browser_fixture")
    summary = report.to_dict()["summary"]
    assert summary["failure_count"] == 0
    assert summary["blocked_count"] == 0
    assert any(check["id"] == "MENON.camera_reprojection" for check in report.to_dict()["checks"])

    tmp_path.chmod(0o700)
    output = tmp_path / "trace.json"
    written = write_trace_from_snapshot(snapshot, output_path=output, run_id="menon_browser_written")
    assert output.is_file()
    assert (tmp_path / "browser_snapshot.json").is_file()
    assert len(written["placements"]) == 1
    assert output.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "browser_snapshot.json").stat().st_mode & 0o777 == 0o600


def test_menon_browser_storage_state_must_be_private_regular_json(tmp_path: Path) -> None:
    state = tmp_path / "storage-state.json"
    state.write_text('{"cookies": [], "origins": []}\n', encoding="utf-8")
    state.chmod(0o600)
    assert validate_storage_state_path(state) == state.resolve()

    state.chmod(0o644)
    with pytest.raises(RuntimeError, match="mode must be 0600"):
        validate_storage_state_path(state)

    state.chmod(0o600)
    symlink = tmp_path / "storage-state-link.json"
    symlink.symlink_to(state)
    with pytest.raises(RuntimeError, match="symlink"):
        validate_storage_state_path(symlink)

    symlink.unlink()
    hardlink = tmp_path / "storage-state-hardlink.json"
    hardlink.hardlink_to(state)
    with pytest.raises(RuntimeError, match="exactly one hard link"):
        validate_storage_state_path(state)


def test_menon_browser_auth_proof_binds_live_session_and_final_origin() -> None:
    snapshot = {
        "href": "http://127.0.0.1:5175/",
        "authState": {"authenticated": True, "role": "owner"},
    }
    session = {
        "authenticated": True,
        "session": {
            "id": "session-1",
            "user": {"role": "owner"},
            "expiresAt": "2026-07-11T12:00:00.000Z",
        },
    }
    proof = validate_browser_auth_proof(
        snapshot,
        session,
        requested_url="http://127.0.0.1:5175/",
        final_url="http://127.0.0.1:5175/",
        checked_at_ms=10_031,
    )
    assert proof["sessionId"] == "session-1"
    assert proof["origin"] == "http://127.0.0.1:5175"

    ipv6_snapshot = copy.deepcopy(snapshot)
    ipv6_snapshot["href"] = "http://[::1]:5175/"
    ipv6_proof = validate_browser_auth_proof(
        ipv6_snapshot,
        session,
        requested_url="http://[::1]:5175/",
        final_url="http://[::1]:5175/",
        checked_at_ms=10_031,
    )
    assert ipv6_proof["origin"] == "http://[::1]:5175"

    with pytest.raises(RuntimeError, match="changed origin"):
        validate_browser_auth_proof(
            snapshot,
            session,
            requested_url="http://127.0.0.1:5175/",
            final_url="http://127.0.0.1:5176/",
            checked_at_ms=10_031,
        )
    expired = copy.deepcopy(session)
    expired["session"]["expiresAt"] = "1970-01-01T00:00:01.000Z"
    with pytest.raises(RuntimeError, match="current authenticated"):
        validate_browser_auth_proof(
            snapshot,
            expired,
            requested_url="http://127.0.0.1:5175/",
            final_url="http://127.0.0.1:5175/",
            checked_at_ms=10_031,
        )


def _mutate_every_canonical_path(snapshot: dict, field: str, value) -> None:
    for path in snapshot["menonTrackDebug"]["canonicalPaths"]:
        path[field] = copy.deepcopy(value)
    for path in snapshot["menonTrackDebug"]["canonicalInfos"]:
        path[field] = copy.deepcopy(value)
    for path in snapshot["latestTrackingPaths"]:
        path[field] = copy.deepcopy(value)
    for path in snapshot["latestTrackInfos"]:
        path[field] = copy.deepcopy(value)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda payload: payload.pop("authSessionProof"), "fresh_auth_session_proof_required"),
        (lambda payload: payload.pop("canonicalWorldState"), "canonical_world_state_shape_invalid"),
        (
            lambda payload: payload["canonicalWorldPresentation"].__setitem__("sequence", 11),
            "canonical_presentation_coherence_invalid",
        ),
        (
            lambda payload: payload["menonTrackDebug"].__setitem__("canonicalPathCount", 2),
            "canonical_debug_paths_incoherent",
        ),
        (
            lambda payload: payload["promotedSceneCohort"].__setitem__("loading", True),
            "scene_cohort_not_current",
        ),
        (
            lambda payload: payload["canonicalWorldState"]["entities"][0].__setitem__("stale_after_us", 10_020_000),
            "canonical_entity_contract_invalid",
        ),
        (
            lambda payload: payload["menonTrackDebug"]["canonicalPaths"][0].__setitem__(
                "sceneTransformCount", "1"
            ),
            "canonical_debug_paths_incoherent",
        ),
        (
            lambda payload: payload["menonTrackDebug"]["canonicalPaths"][0].__setitem__(
                "backendWorldPosition", [99.0, 0.0, 0.0]
            ),
            "canonical_debug_paths_incoherent",
        ),
    ],
)
def test_menon_browser_adapter_rejects_spoofed_or_incoherent_canonical_evidence(
    mutation,
    reason: str,
) -> None:
    snapshot = json.loads(
        Path("plans/noesis_menon_validation/minimal_menon_browser_snapshot.json").read_text(encoding="utf-8")
    )
    mutation(snapshot)
    trace = browser_snapshot_to_menon_trace(snapshot)
    assert trace["placements"] == []
    assert trace["browser"]["canonical_admission"] == reason
    assert "camera_reprojections" not in trace


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backendWorldPosition", [99.0, 0.0, 0.0]),
        ("position", [99.0, 0.0, 0.0]),
        ("sceneTransformCount", "1"),
        ("lifecycle", "invented"),
        ("subjectId", "resident:other"),
    ],
)
def test_menon_browser_adapter_binds_each_current_path_to_the_canonical_entity(
    field: str,
    value,
) -> None:
    snapshot = json.loads(
        Path("plans/noesis_menon_validation/minimal_menon_browser_snapshot.json").read_text(encoding="utf-8")
    )
    _mutate_every_canonical_path(snapshot, field, value)
    trace = browser_snapshot_to_menon_trace(snapshot)
    assert trace["placements"] == []
    assert trace["browser"]["canonical_admission"] == "canonical_render_entity_mismatch"


def test_menon_browser_adapter_does_not_promote_legacy_per_camera_projection() -> None:
    snapshot = {
        "trackingProjectionMode": "frontend_reproject",
        "calibrationData": {
            "align": {"scene_similarity": {"world_to_scene_col_major": matrix_to_col_major(np.eye(4))}}
        },
        "menonTrackDebug": {
            "rawByCamera": {
                "camera-a": {
                    "tracks": [{
                        "stableId": 1,
                        "trackerId": 2,
                        "backendWorldRaw": [0.0, 0.0, 0.0],
                        "world": [0.0, 0.0, 0.0],
                    }]
                }
            }
        },
    }
    trace = browser_snapshot_to_menon_trace(snapshot)
    assert trace["placements"] == []
    assert trace["browser"]["projection_mode"] == "frontend_reproject"


def test_menon_trace_cli_writes_report_and_artifacts(tmp_path: Path) -> None:
    json_path, md_path, payload = run_trace_report(
        "plans/noesis_menon_validation/minimal_menon_trace.json",
        output_dir=tmp_path,
        run_id="menon_trace_cli",
    )
    assert json_path.is_file()
    assert md_path.is_file()
    assert payload["summary"]["failure_count"] == 0
    assert (tmp_path / "menon_trace_cli" / "menon" / "trace.json").is_file()
    assert (tmp_path / "menon_trace_cli" / "menon" / "trace_audit.json").is_file()
    assert (tmp_path / "menon_trace_cli" / "visual" / "index.json").is_file()
    run_dir = tmp_path / "menon_trace_cli"
    assert run_dir.stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
        for path in run_dir.rglob("*")
    )


def test_menon_trace_required_checkout_reports_blocked(tmp_path: Path) -> None:
    _, _, payload = run_trace_report(
        "plans/noesis_menon_validation/minimal_menon_trace.json",
        output_dir=tmp_path,
        run_id="menon_trace_requires_checkout",
        require_menon_root=True,
    )
    assert payload["summary"]["status"] == "blocked"
    assert payload["summary"]["blocked_count"] == 1


def test_regression_runner_executes_registered_fixtures(tmp_path: Path) -> None:
    previous_umask = os.umask(0o027)
    try:
        summary_path, payload = run_regression_suite(
            "plans/noesis_menon_validation/fixture_registry.json",
            output_dir=tmp_path,
            run_id="regression_fixture",
        )
        restored_umask = os.umask(0o027)
        assert restored_umask == 0o027
    finally:
        os.umask(previous_umask)
    assert summary_path.is_file()
    assert payload["status"] == "pass"
    assert payload["case_count"] == 4
    assert payload["pass_count"] == 4
    assert payload["regression_failure_count"] == 0
    assert payload["failure_categories"] == {}
    assert all(case["regression_status"] == "pass" for case in payload["cases"])
    fixture_case = next(case for case in payload["cases"] if case["fixture_id"] == "minimal_validation_fixture")
    assert len(fixture_case["artifact_comparisons"]) == 2
    assert all(item["status"] == "pass" for item in fixture_case["artifact_comparisons"])
    menon_case = next(case for case in payload["cases"] if case["fixture_id"] == "minimal_menon_trace")
    assert menon_case["check_count"] >= 10
    suite_dir = summary_path.parent
    assert suite_dir.stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
        for path in suite_dir.rglob("*")
    )

    unsafe_dir = tmp_path / "unsafe_regression"
    unsafe_dir.mkdir()
    unsafe_dir.chmod(0o755)
    with pytest.raises(RuntimeError, match="mode must be 0700"):
        run_regression_suite(
            "plans/noesis_menon_validation/fixture_registry.json",
            output_dir=tmp_path,
            run_id="unsafe_regression",
            fixture_ids=["minimal_menon_browser_snapshot"],
        )

    symlink_dir = tmp_path / "symlink_regression"
    symlink_dir.symlink_to(suite_dir, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        run_regression_suite(
            "plans/noesis_menon_validation/fixture_registry.json",
            output_dir=tmp_path,
            run_id="symlink_regression",
            fixture_ids=["minimal_menon_browser_snapshot"],
        )

    hardlink_dir = tmp_path / "hardlink_regression"
    hardlink_dir.mkdir(mode=0o700)
    linked_file = hardlink_dir / "artifact.json"
    linked_file.write_text("{}\n", encoding="utf-8")
    linked_file.chmod(0o600)
    (hardlink_dir / "artifact-copy.json").hardlink_to(linked_file)
    with pytest.raises(RuntimeError, match="exactly one hard link"):
        run_regression_suite(
            "plans/noesis_menon_validation/fixture_registry.json",
            output_dir=tmp_path,
            run_id="hardlink_regression",
            fixture_ids=["minimal_menon_browser_snapshot"],
        )


def test_regression_failure_categories_include_check_and_expectation_failures() -> None:
    payload = {
        "checks": [
            {"status": "fail", "failure_type": "projection_failure"},
            {"status": "warning", "failure_type": "temporal_failure"},
            {"status": "pass"},
        ]
    }
    categories, dominant, focus = _failure_categories(payload, ["check_count below expected minimum"])
    assert categories == {"projection_failure": 1, "regression_failure": 1, "temporal_failure": 1}
    assert dominant == "temporal_failure"
    assert "smoothing" in str(focus)


def test_golden_image_diff_and_artifact_expectations(tmp_path: Path) -> None:
    from PIL import Image

    golden = tmp_path / "golden.png"
    actual = tmp_path / "actual.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(golden)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(actual)
    diff = tmp_path / "diff.png"
    result = compare_image_files(actual, golden, diff_path=diff)
    assert result["status"] == "pass"
    assert diff.is_file()

    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "visual"
    artifact_dir.mkdir(parents=True)
    copied = artifact_dir / "actual.png"
    copied.write_bytes(actual.read_bytes())
    payload = {"artifacts": {"actual_image": "visual/actual.png"}}
    failures, comparisons = compare_artifact_expectations(
        payload,
        run_dir=run_dir,
        expectations=[{"key": "actual_image", "min_bytes": 10, "golden_path": str(golden)}],
        registry_dir=tmp_path,
        diff_dir=run_dir / "regression_diffs",
    )
    assert failures == []
    assert comparisons[0]["status"] == "pass"
    assert Path(comparisons[0]["image_diff"]["diff_path"]).is_file()
