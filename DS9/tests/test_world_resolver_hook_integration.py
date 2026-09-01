from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from noesis_core.contracts.base import Matrix3, Vector3
from geometry.homography import project_world_to_image
from noesis_core.contracts.world_measurement import (
    WorldMeasurementCohort,
    WorldMeasurementHypothesis,
    WorldMeasurementSet,
    WorldPriorEvidence,
)
from noesis.metadata.object_depth import ObjectDepthResult
from noesis.pipelines import hooks


def _processor(config: dict[str, object] | None = None):
    return hooks._AnalyticsTelemetryProcessor(  # type: ignore[attr-defined]
        pipeline=SimpleNamespace(
            config=config or {
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    }
                }
            },
            frame_size=(1920, 1080),
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "camera-0"},
        sensor_id_map={0: 0},
    )


def _upright_projection_calibration() -> SimpleNamespace:
    intrinsics = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_world = np.array([0.0, 2.2, -6.0], dtype=np.float64)
    forward = np.array([0.0, -0.25, 1.0], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    down /= np.linalg.norm(down)
    rotation_cw = np.column_stack([right, down, forward]).T
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation_cw
    extrinsics[:3, 3] = -rotation_cw @ camera_world
    return SimpleNamespace(
        intrinsics=intrinsics,
        extrinsics_col_major=list(extrinsics.flatten(order="F")),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
    )


def _occluded_upright_pose(
    calibration: SimpleNamespace,
) -> tuple[np.ndarray, list[float]]:
    def project(world_xyz: tuple[float, float, float]) -> tuple[float, float]:
        uv = project_world_to_image(
            world_xyz,
            calibration.intrinsics,
            calibration.extrinsics_col_major,
            calibration.image_size,
            unit_scale=1.0,
        )
        assert uv is not None
        return float(uv[0]), float(uv[1])

    pose = np.zeros((17, 3), dtype=np.float64)
    for name, world_xyz in {
        "nose": (0.0, 1.692, 6.0),
        "left_shoulder": (-0.18, 1.476, 6.0),
        "right_shoulder": (0.18, 1.476, 6.0),
        "left_hip": (-0.10, 0.99, 6.0),
        "right_hip": (0.10, 0.99, 6.0),
    }.items():
        u, v = project(world_xyz)
        pose[hooks._POSE_KPT_INDEX[name]] = [u, v, 0.98]

    foot_u, foot_v = project((0.0, 0.0, 6.0))
    _head_u, head_v = project((0.0, 1.8, 6.0))
    # The detector terminates on furniture, well above the actual floor.
    bbox = [foot_u - 42.0, head_v, 84.0, 0.68 * (foot_v - head_v)]
    return pose, [float(value) for value in bbox]


def _cohort(*, pcf_revision: str | None = None) -> WorldMeasurementCohort:
    return WorldMeasurementCohort(
        track_key="5:17:3",
        tracker_lifecycle_generation=3,
        camera_id="camera-0",
        source_id=5,
        tracker_id=17,
        frame_id=42,
        observed_at_us=123456,
        world_revision="world-revision",
        calibration_revision="calibration-revision",
        pcf_revision=pcf_revision,
    )


def _hypothesis(
    cohort: WorldMeasurementCohort,
    *,
    candidate_id: str,
    kind: str = "floor_ray",
    x: float = 1.0,
) -> WorldMeasurementHypothesis:
    return WorldMeasurementHypothesis(
        candidate_id=candidate_id,
        cohort=cohort,
        kind=kind,  # type: ignore[arg-type]
        position=Vector3(x=x, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
            )
        ),
        anchor="pose_ankle_support",
        support_state="floor",
        posture="standing",
        confidence=0.9,
        support_score=0.9,
        ray_incidence_sin=0.8,
        pixel_uncertainty_px=2.0,
    )


def test_hook_emits_snake_case_full_covariance_resolver_diagnostics() -> None:
    processor = _processor()
    processor.set_world_resolver_diagnostics_enabled(True)
    hooks.reset_core_path_instrumentation()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(
            _hypothesis(cohort, candidate_id="floor_ray"),
            _hypothesis(
                cohort,
                candidate_id="registered_depth",
                kind="registered_depth",
                x=1.02,
            ),
        ),
    )
    track: dict[str, object] = {}

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is not None
    assert result is not None
    diagnostic = track["world_resolver"]
    assert isinstance(diagnostic, dict)
    assert diagnostic["frame_id"] == 42
    assert diagnostic["world_frame_revision"] == "world-revision"
    assert diagnostic["selected_id"]
    assert diagnostic["contributor_ids"]
    assert diagnostic["resolved"]["position"] == {"x": point[0], "y": point[1], "z": point[2]}
    assert len(diagnostic["resolved"]["covariance"]["values"]) == 9
    assert len(diagnostic["candidates"]) == 2
    assert diagnostic["candidates"][0]["id"] == "floor_ray"
    assert len(diagnostic["candidates"][0]["covariance"]["values"]) == 9
    assert "world" not in diagnostic["candidates"][0]
    assert "covarianceXZ" not in diagnostic["candidates"][0]
    # Resolver covariance is private until PersonGroundState accepts the
    # current measurement and the emitted point is known.
    assert "world_covariance" not in track
    assert track["world_quantity"] == "ground_footprint"
    assert track["world_support_state"] == "floor"
    timing = hooks.get_core_path_instrumentation_snapshot()["stage_timings"][
        "world_resolver.resolve"
    ]
    assert timing["count"] == 1
    assert timing["total_ns"] >= 0


def test_request_gated_legacy_comparison_reconstructs_retired_room_policy_only() -> None:
    processor = _processor()
    processor.legacy_world_fusion_policy = SimpleNamespace(
        policy_id="historical-policy-id",
        profile=lambda _camera_id: SimpleNamespace(
            floor_weight_scale=1.0,
            depth_weight_scale=1.0,
            floor_only_allowed=False,
        ),
    )
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(
            _hypothesis(cohort, candidate_id="floor_ray", x=1.0),
            _hypothesis(
                cohort,
                candidate_id="registered_depth",
                kind="registered_depth",
                x=1.8,
            ),
        ),
    )

    compact_track: dict[str, object] = {}
    compact_point, _compact_result = processor._apply_resolved_world_measurement(
        compact_track,
        measurement_set,
    )
    assert compact_point is not None
    assert "world_resolver" not in compact_track

    processor.set_world_resolver_diagnostics_enabled(True)
    detailed_track: dict[str, object] = {}
    detailed_point, _detailed_result = processor._apply_resolved_world_measurement(
        detailed_track,
        measurement_set,
    )
    assert detailed_point is not None
    assert detailed_point.tolist() == pytest.approx(compact_point.tolist())
    diagnostics = detailed_track["world_resolver"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["legacy"] == {
        "id": "legacy_room_policy",
        "kind": "legacy_policy",
        "position": {"x": 1.8, "y": 0.0, "z": 2.0},
        "covariance": {
            "values": [
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
            ]
        },
        "source": "historical_depth_labeled_fused",
        "policy_id": "historical-policy-id",
    }


def test_ndarray_candidates_survive_hook_contract_boundary() -> None:
    processor = _processor()
    cohort = _cohort()
    candidate_set = processor._build_world_measurement_set(
        cohort=cohort,
        candidates=[
            {
                "candidate_id": "floor_ray",
                "kind": "floor_ray",
                "position": np.asarray([1.0, 0.0, 2.0], dtype=np.float64),
                "covariance": Matrix3(
                    values=(
                        0.04,
                        0.0,
                        0.0,
                        0.0,
                        0.04,
                        0.0,
                        0.0,
                        0.0,
                        0.04,
                    )
                ),
                "anchor": "pose_ankle_support",
                "support_state": "floor",
                "posture": "standing",
            }
        ],
    )

    assert candidate_set is not None
    assert len(candidate_set.hypotheses) == 1
    point, result = processor._apply_resolved_world_measurement({}, candidate_set)
    assert point is not None
    assert result is not None


def test_hook_cohort_uses_track_source_and_observed_time() -> None:
    processor = _processor()
    track = {
        "tracker_id": 17,
        "frame_id": 42,
        "source_id": 5,
        "observed_at_us": 123456,
        "tracker_lifecycle_generation": 3,
    }
    calib = SimpleNamespace(camera_calibration_sha256="calibration-revision")

    cohort = processor._world_measurement_cohort(
        99,
        "camera-0",
        track,
        calib=calib,
        world_frame_revision="world-revision",
        world_transform_sha256="a" * 64,
    )

    assert cohort is not None
    assert cohort.source_id == 5
    assert cohort.observed_at_us == 123456
    assert cohort.calibration_revision == "calibration-revision"
    assert cohort.world_revision == "world-revision"
    assert cohort.track_key == "5:17:3"
    assert cohort.tracker_lifecycle_generation == 3
    assert cohort.world_transform_sha256 == "a" * 64


def test_hook_cohort_rejects_missing_exact_lifecycle_or_transform_identity() -> None:
    processor = _processor()
    calib = SimpleNamespace(camera_calibration_sha256="calibration-revision")
    base = {
        "tracker_id": 17,
        "frame_id": 42,
        "source_id": 5,
        "observed_at_us": 123456,
        "tracker_lifecycle_generation": 3,
    }
    assert (
        processor._world_measurement_cohort(
            99,
            "camera-0",
            {**base, "tracker_lifecycle_generation": None},
            calib=calib,
            world_frame_revision="world-revision",
            world_transform_sha256="a" * 64,
        )
        is None
    )
    assert (
        processor._world_measurement_cohort(
            99,
            "camera-0",
            base,
            calib=calib,
            world_frame_revision="world-revision",
            world_transform_sha256=None,
        )
        is None
    )


def test_invalid_enabled_resolver_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="measurement_resolver"):
        _processor(
            {
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_candidates": "not-an-int",
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "patch",
    (
        {"enabled": "true"},
        {"legacy_comparator": False},
        {"max_range_m": 0.5},
        {"max_disagreement_m": float("nan")},
        {"max_candidates": 5},
        {"max_candidates": True},
    ),
)
def test_resolver_config_rejects_coercion_and_out_of_range_values(
    patch: dict[str, object],
) -> None:
    config: dict[str, object] = {
        "enabled": True,
        "max_range_m": 22.0,
        "max_disagreement_m": 1.25,
        "max_candidates": 4,
    }
    config.update(patch)
    with pytest.raises(ValueError, match="measurement_resolver"):
        _processor({"canonical_world": {"measurement_resolver": config}})


@pytest.mark.parametrize(
    "person_admission",
    (
        {},
        {"min_unposed_detection_confidence": "0.5"},
        {"min_unposed_detection_confidence": True},
        {"min_unposed_detection_confidence": -0.1},
        {"min_unposed_detection_confidence": 1.1},
        {"unexpected": 0.5},
    ),
)
def test_person_admission_config_rejects_missing_coercion_and_out_of_range_values(
    person_admission: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="person_admission"):
        _processor({"canonical_world": {"person_admission": person_admission}})


def test_person_admission_config_uses_explicit_default_when_omitted() -> None:
    assert _processor()._min_unposed_detection_confidence == pytest.approx(0.50)


def test_filtered_covariance_includes_resolver_to_emitted_displacement() -> None:
    processor = _processor()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(_hypothesis(cohort, candidate_id="floor_ray"),),
    )
    track: dict[str, object] = {}
    point, result = processor._apply_resolved_world_measurement(track, measurement_set)
    assert point is not None and result is not None

    processor._publish_filtered_world_covariance(
        track,
        resolved=result,
        resolved_point=point,
        emitted_point=np.asarray([point[0] + 0.5, point[1], point[2]], dtype=np.float64),
    )

    covariance = track["world_covariance"]
    assert isinstance(covariance, list)
    assert len(covariance) == 9
    assert covariance[0] >= 0.29


def test_resolver_preserves_selected_anatomical_contact_basis() -> None:
    processor = _processor()
    processor.set_world_resolver_diagnostics_enabled(True)
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(_hypothesis(cohort, candidate_id="floor_ray"),),
    )
    track: dict[str, object] = {}

    point, result = processor._apply_resolved_world_measurement(
        track,
        measurement_set,
        contact_basis_by_candidate={"floor_ray": "pose:left_ankle"},
    )

    assert point is not None
    assert result is not None and result.status == "measured"
    assert track["world_resolver_contact_basis"] == "pose:left_ankle"
    candidate = track["world_resolver"]["candidates"][0]
    assert candidate["anchor"] == "pose_ankle_support"
    assert candidate["contact_basis"] == "pose:left_ankle"


def test_person_ground_consensus_collapses_only_observed_ankle_sides() -> None:
    processor = _processor()
    state = hooks._WorldAnchorState(
        ts=0.0,
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.0,
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=0.0,
    )

    for index, (basis, expected_count) in enumerate(
        (
            ("pose:left_ankle", 1),
            ("pose:right_ankle", 2),
            ("depth:pose_torso_support", 1),
            ("depth:lower_body_band", 1),
        ),
        start=1,
    ):
        processor._update_track_world_state(
            {},
            state,
            measurement=np.asarray([5.0 + index * 0.1, 0.0, 0.0]),
            floor_y=0.0,
            now_ts=index * 0.1,
            alpha=0.45,
            beta=0.15,
            quality="good",
            contact_basis=processor._person_ground_consensus_contact_basis(basis),
            image_motion_supported=True,
        )
        assert state.measurement_accepted is False
        assert state.reacquire_count == expected_count

    assert processor._person_ground_consensus_contact_basis(
        "pose:ankle_pair"
    ) == "pose_ankle_floor"
    assert processor._person_ground_consensus_contact_basis(
        "pose:left_ankle"
    ) == "pose_single_ankle_floor"
    assert processor._person_ground_consensus_contact_basis(
        "pose:right_ankle"
    ) == "pose_single_ankle_floor"
    assert processor._person_ground_consensus_contact_basis(
        "depth:pose_torso_support"
    ) == "depth:pose_torso_support"


def test_resolver_apply_clears_compact_fields_on_rejection() -> None:
    processor = _processor()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(cohort=cohort, hypotheses=())
    track: dict[str, object] = {
        "world_quantity": "ground_footprint",
        "world_posture": "standing",
        "world_support_state": "floor",
        "world_resolver_confidence": 0.99,
        "world_resolver_selected_id": "stale",
        "world_resolver_fused": True,
        "world_resolver_disagreement_m": 2.0,
        "world_resolver_contact_basis": "stale",
        "world_resolver_source_continuity_match": True,
        "world_resolver": {"candidates": [{"stale": True}]},
        "world_covariance": [1.0] * 9,
    }

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is None
    assert result is not None and result.status == "rejected"
    assert "world_resolver" not in track
    assert "world_covariance" not in track
    assert "world_resolver_selected_id" not in track
    assert "world_resolver_fused" not in track
    assert "world_resolver_disagreement_m" not in track
    assert "world_resolver_contact_basis" not in track
    assert track["world_resolver_confidence"] == 0.0


def test_torso_depth_anchor_never_enters_floor_ray_hypothesis() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.scene_priors = None
    processor._floor_candidate_covariance = lambda *_args, **_kwargs: Matrix3(  # type: ignore[method-assign]
        values=(
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
        )
    )
    torso_anchor = hooks._PoseAnchorCandidate(
        u=100.0,
        v=200.0,
        source="person_mask_floor",
        contact_basis="depth:torso_core",
        quality="estimated",
    )
    candidates: list[dict[str, object]] = []
    processor._append_universal_world_candidates(
        candidates,
        sensor_id=0,
        track={"tracker_id": 7, "world_floor_incidence_sin": 0.8},
        camera_id="cam0",
        calib=SimpleNamespace(),
        floor_candidate=np.asarray([1.0, 0.0, 2.0], dtype=np.float64),
        floor_ray_admitted=True,
        anchor_candidate=torso_anchor,
        pose_uv=(100.0, 200.0),
        contact_basis="depth:torso_core",
        posture="standing",
        occlusion_fraction=0.0,
        image_motion_supported=True,
        depth_obs=None,
        depth_weight=0.0,
        depth_observation=hooks._DepthObservationResult(None, 0.0, "none"),
        depth_result=None,
        person_anchor=torso_anchor,
        reject_current_geometry=False,
        flip_u=False,
        flip_v=False,
    )

    assert candidates == []
    assert processor._anchor_is_verified_ground_contact(torso_anchor) is False


def test_torso_depth_remains_registered_depth_with_unknown_support() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.scene_priors = None
    processor.depth_registration = None
    processor._depth_candidate_covariance = lambda *_args, **_kwargs: Matrix3(  # type: ignore[method-assign]
        values=(
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
        )
    )
    torso_anchor = hooks._PoseAnchorCandidate(
        u=100.0,
        v=200.0,
        source="person_mask_floor",
        contact_basis="depth:torso_core",
        quality="estimated",
    )
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=7,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=64,
        valid_fraction=0.9,
        anchor_source="torso_core",
        anchor_depth_m=5.0,
        anchor_sample_count=64,
        anchor_valid_fraction=0.9,
    )
    candidates: list[dict[str, object]] = []
    processor._append_universal_world_candidates(
        candidates,
        sensor_id=0,
        track={"tracker_id": 7, "world_floor_incidence_sin": 0.8},
        camera_id="cam0",
        calib=SimpleNamespace(floor_y=0.0),
        floor_candidate=None,
        floor_ray_admitted=False,
        anchor_candidate=torso_anchor,
        pose_uv=None,
        contact_basis="depth:torso_core",
        posture="standing",
        occlusion_fraction=0.75,
        image_motion_supported=False,
        depth_obs=np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
        depth_weight=0.8,
        depth_observation=hooks._DepthObservationResult(
            np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
            0.8,
            "ok",
            registered_depth_m=5.0,
            registration_status="raw_passthrough",
        ),
        depth_result=depth_result,
        person_anchor=torso_anchor,
        reject_current_geometry=False,
        flip_u=False,
        flip_v=False,
    )

    assert [item["kind"] for item in candidates] == ["registered_depth"]
    assert candidates[0]["support_state"] == "unknown"


def test_moving_upright_lateral_truncation_excludes_interior_seating() -> None:
    processor = _processor()
    state = hooks._WorldAnchorState(
        height_ref_scene=1.7,
        last_full_body_ts=9.5,
        last_non_upright_ts=-1.0,
        upright_bbox_height_px=600.0,
        last_accepted_bbox_geometry=(1347.0, 524.25, 283.5, 396.0),
    )

    edge = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(1400.0, 200.0, 300.0, 520.0),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )
    interior = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(700.0, 400.0, 300.0, 300.0),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )
    state.last_accepted_bbox_geometry = (1400.0, 200.0, 300.0, 520.0)
    stationary_edge = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(1400.0, 200.0, 300.0, 520.0),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )
    hard_clip = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(1558.5, 436.5, 360.0, 415.5),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )

    assert edge[:3] == (True, True, False)
    assert edge[3] == pytest.approx(520.0 / 600.0)
    assert interior[:3] == (False, False, False)
    assert interior[3] == pytest.approx(0.5)
    assert stationary_edge[:3] == (False, True, False)
    assert hard_clip[:3] == (True, True, True)


def test_occluded_upright_body_planes_recover_floor_without_bbox_bottom() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, furniture_truncated_bbox = _occluded_upright_pose(calibration)

    projection = processor._upright_pose_ground_projection(
        calibration,
        kpts_abs=pose,
        bbox=furniture_truncated_bbox,
        state=None,
        flip_u=False,
        flip_v=False,
    )

    assert projection is not None
    point, covariance, height_m, scatter_m, anchor_count, strong_proof = projection
    assert point == pytest.approx([0.0, 0.0, 6.0], abs=0.20)
    assert height_m == pytest.approx(1.8, abs=0.10)
    assert scatter_m < 0.30
    assert anchor_count == 5
    assert strong_proof is True
    assert covariance.values[0] >= 0.04
    assert covariance.values[8] >= 0.04


def test_torso_motion_anchor_tolerates_one_joint_dropout_only() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, bbox = _occluded_upright_pose(calibration)

    pose[hooks._POSE_KPT_INDEX["right_shoulder"], 2] = 0.0
    assert processor._resolve_pose_torso_motion_anchor(pose, bbox=bbox) is not None

    pose[hooks._POSE_KPT_INDEX["right_hip"], 2] = 0.0
    assert processor._resolve_pose_torso_motion_anchor(pose, bbox=bbox) is None


def test_four_body_planes_remain_weak_without_independent_head_plane() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, bbox = _occluded_upright_pose(calibration)
    pose[hooks._POSE_KPT_INDEX["nose"], 2] = 0.0

    projection = processor._upright_pose_ground_projection(
        calibration,
        kpts_abs=pose,
        bbox=bbox,
        state=None,
        flip_u=False,
        flip_v=False,
    )

    assert projection is not None
    _point, _covariance, _height_m, _scatter_m, anchor_count, strong_proof = (
        projection
    )
    assert anchor_count == 4
    assert strong_proof is False


def test_complete_side_profile_torso_does_not_require_apparent_pair_width() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, bbox = _occluded_upright_pose(calibration)

    for left_name, right_name in (
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ):
        left_index = hooks._POSE_KPT_INDEX[left_name]
        right_index = hooks._POSE_KPT_INDEX[right_name]
        center_x = 0.5 * (pose[left_index, 0] + pose[right_index, 0])
        pose[left_index, 0] = center_x - 1.0
        pose[right_index, 0] = center_x + 1.0

    assert processor._resolve_pose_torso_motion_anchor(pose, bbox=bbox) is not None
    assert (
        processor._upright_pose_ground_projection(
            calibration,
            kpts_abs=pose,
            bbox=bbox,
            state=None,
            flip_u=False,
            flip_v=False,
        )
        is not None
    )


@pytest.mark.parametrize(
    "kind, anchor, support_state, posture, expected",
    (
        ("pose_scale", "seated_torso_plane", "seat", "sitting", True),
        ("pose_scale", "upright_body_plane", "unknown", "standing", True),
        (
            "registered_depth",
            "person_body_projection",
            "seat",
            "sitting",
            True,
        ),
        (
            "registered_depth",
            "person_body_projection",
            "unknown",
            "sitting",
            False,
        ),
        ("pose_scale", "seated_torso_plane", "seat", "standing", False),
    ),
)
def test_only_typed_body_projections_support_ground_footprint(
    kind: str,
    anchor: str,
    support_state: str,
    posture: str,
    expected: bool,
) -> None:
    candidate = SimpleNamespace(
        kind=kind,
        anchor=anchor,
        support_state=support_state,
        posture=posture,
    )

    assert (
        hooks._AnalyticsTelemetryProcessor
        ._resolver_candidate_is_ground_footprint_supported(candidate)
        is expected
    )


@pytest.mark.parametrize(
    "field, value",
    (
        ("source_id", 6),
        ("tracker_id", 18),
        ("frame_id", 43),
        ("media_pts_ns", 123457000),
    ),
)
def test_depth_measurement_requires_exact_current_track_cohort(field, value) -> None:
    result = ObjectDepthResult(
        source_id=5,
        frame_id=42,
        object_id=17,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=32,
        valid_fraction=0.9,
        measurement_frame_id=42,
        measurement_ts_us=123456,
        measurement_age_us=0,
        measurement_cached=False,
        depth_tensor_frame_id=42,
        depth_tensor_ts_us=123456,
        depth_tensor_age_frames=0,
        depth_tensor_age_us=0,
        ts_us=123456,
    )
    track = {
        "source_id": 5,
        "tracker_id": 17,
        "frame_id": 42,
        "media_pts_ns": 123456000,
    }
    assert hooks._AnalyticsTelemetryProcessor._depth_measurement_is_current(  # type: ignore[attr-defined]
        result,
        track=track,
    ) is True
    mismatched = dict(track)
    mismatched[field] = value
    assert hooks._AnalyticsTelemetryProcessor._depth_measurement_is_current(  # type: ignore[attr-defined]
        result,
        track=mismatched,
    ) is False


def test_depth_measurement_rejects_lagged_tensor_even_with_current_object_cohort() -> None:
    result = ObjectDepthResult(
        source_id=5,
        frame_id=42,
        object_id=17,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=32,
        valid_fraction=0.9,
        measurement_frame_id=42,
        measurement_ts_us=123456,
        measurement_age_us=0,
        measurement_cached=False,
        depth_tensor_frame_id=41,
        depth_tensor_ts_us=90123,
        depth_tensor_age_frames=1,
        depth_tensor_age_us=33333,
        ts_us=123456,
    )
    assert hooks._AnalyticsTelemetryProcessor._depth_measurement_is_current(  # type: ignore[attr-defined]
        result,
        track={
            "source_id": 5,
            "tracker_id": 17,
            "frame_id": 42,
            "media_pts_ns": 123456000,
        },
    ) is False


def test_weak_resolver_result_is_diagnostic_only() -> None:
    processor = _processor()
    processor.set_world_resolver_diagnostics_enabled(True)
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(
            WorldMeasurementHypothesis(
                candidate_id="weak",
                cohort=cohort,
                kind="gravity_reconstruction",
                position=Vector3(x=1.0, y=0.0, z=2.0),
                covariance=Matrix3(
                    values=(
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                    )
                ),
                anchor="learned_body_height",
                confidence=0.01,
                support_score=0.01,
                source_reliability=0.01,
            ),
        ),
    )
    track: dict[str, object] = {}

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is not None
    assert result is not None
    assert result.quality == "weak"
    assert track["world_resolver"]["decision"] == "weak"  # type: ignore[index]


def test_weak_bbox_requires_motion_or_trusted_lifecycle_consensus() -> None:
    processor = _processor()
    cohort = _cohort(pcf_revision="revision")
    supported = WorldMeasurementHypothesis(
        candidate_id="floor",
        cohort=cohort,
        kind="floor_ray",
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04)
        ),
        anchor="bbox_bottom",
        confidence=0.75,
        support_score=0.72,
        support_state="floor",
        posture="standing",
        pcf=WorldPriorEvidence(
            prior_id="prior",
            revision_id="revision",
            status="fail",
            inside_extent=True,
            inside_authored_space=False,
            boundary_signed_distance_m=-2.0,
        ),
    )
    unsupported = WorldMeasurementHypothesis(
        candidate_id="gravity",
        cohort=cohort,
        kind="gravity_reconstruction",
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(4.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 4.0)
        ),
        anchor="learned_body_height",
        confidence=0.75,
        support_score=0.72,
        source_reliability=0.01,
        support_state="floor",
        posture="standing",
    )
    floor_set = WorldMeasurementSet(cohort=cohort, hypotheses=(supported,))
    gravity_set = WorldMeasurementSet(cohort=cohort, hypotheses=(unsupported,))

    floor_result = processor._world_resolver.resolve(floor_set)
    gravity_result = processor._world_resolver.resolve(gravity_set)

    assert floor_result.quality == "weak"
    assert processor._weak_resolver_measurement_is_ground_supported(
        floor_result,
        floor_set,
    ) is False
    assert processor._weak_resolver_measurement_is_ground_supported(
        floor_result,
        floor_set,
        image_motion_supported=True,
    ) is True
    trusted_state = hooks._WorldAnchorState(
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=1.0,
        last_output_world_x=0.0,
        last_output_world_z=0.0,
    )
    assert processor._weak_resolver_measurement_is_ground_supported(
        floor_result,
        floor_set,
        state=trusted_state,
    ) is True
    assert gravity_result.quality == "weak"
    assert processor._weak_resolver_measurement_is_ground_supported(
        gravity_result,
        gravity_set,
    ) is False


def test_weak_observed_ankle_pcf_conflict_requires_metric_lifecycle() -> None:
    processor = _processor()
    cohort = _cohort(pcf_revision="revision")
    ankle = WorldMeasurementHypothesis(
        candidate_id="ankle",
        cohort=cohort,
        kind="floor_ray",
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04)
        ),
        anchor="pose_ankle_floor",
        confidence=0.95,
        support_score=0.95,
        support_state="floor",
        posture="sitting",
        pcf=WorldPriorEvidence(
            prior_id="prior",
            revision_id="revision",
            status="fail",
            inside_extent=True,
            inside_authored_space=False,
            boundary_signed_distance_m=-2.0,
        ),
    )
    measurement_set = WorldMeasurementSet(cohort=cohort, hypotheses=(ankle,))
    result = processor._world_resolver.resolve(measurement_set)

    assert result.quality == "weak"
    assert processor._weak_resolver_measurement_is_ground_supported(
        result,
        measurement_set,
    ) is False
    assert processor._weak_resolver_measurement_is_ground_supported(
        result,
        measurement_set,
        verified_first_output_contact=True,
    ) is True
    trusted_state = hooks._WorldAnchorState(
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=1.0,
        last_output_world_x=0.0,
        last_output_world_z=0.0,
    )
    assert processor._weak_resolver_measurement_is_ground_supported(
        result,
        measurement_set,
        state=trusted_state,
    ) is True

    tight_contact = {
        "pose_present": True,
        "world_floor_admitted": True,
        "world_floor_contact_plausible": True,
        "world_floor_contact_gap_ratio": 0.0,
        "world_floor_contact_range_delta_m": 0.0,
        "world_floor_incidence_sin": 0.30,
    }
    assert processor._first_output_ankle_contact_is_tight(tight_contact) is True
    reflected_contact = dict(
        tight_contact,
        world_floor_contact_gap_ratio=0.11,
        world_floor_contact_range_delta_m=1.54,
    )
    assert processor._first_output_ankle_contact_is_tight(reflected_contact) is False
    near_horizon_contact = dict(
        tight_contact,
        world_floor_incidence_sin=0.16,
    )
    assert (
        processor._first_output_ankle_contact_is_tight(near_horizon_contact)
        is False
    )
    assert (
        processor._cold_floor_ray_incidence_is_adequate(tight_contact) is True
    )
    assert (
        processor._cold_floor_ray_incidence_is_adequate(near_horizon_contact)
        is False
    )

def test_rich_resolver_diagnostics_are_off_by_default_but_summary_remains() -> None:
    processor = _processor()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(_hypothesis(cohort, candidate_id="floor_ray"),),
    )
    track: dict[str, object] = {
        "world_resolver": {"stale": True},
    }

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is not None
    assert result is not None
    assert "world_resolver" not in track
    assert track["world_resolver_confidence"] == pytest.approx(result.confidence)
    assert track["world_resolver_selected_id"] == "floor_ray"
    assert track["world_quantity"] == "ground_footprint"


def test_canonical_tracking_copy_excludes_rich_resolver_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor()
    published: list[object] = []
    monkeypatch.setattr(processor, "_publish_tracking_work", published.append)
    track: dict[str, object] = {
        "camera_id": "camera-0",
        "tracker_id": 17,
        "frame_id": 42,
        "class_id": 0,
        "world_resolver": {
            "contract": "noesis.world_resolver_diagnostics",
            "candidates": [{"id": "floor_ray", "covariance": {"values": [1.0] * 9}}],
        },
        "world_resolver_confidence": 0.9,
        "world_resolver_selected_id": "floor_ray",
        "_world_force_first_metric_publication": True,
    }
    continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=42,
        observed_at_us=123456,
        tracks=[track],
        footpoints=[],
    )

    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(frame_number=42, buf_pts=123456000),
        tracks=[track],
        footpoints=[],
        now_ts=1.0,
        temporal_contract={"observed_at_us": 123456},
        continuity=continuity,
        force=True,
    )

    assert len(published) == 1
    queued_track = published[0].tracks[0]  # type: ignore[attr-defined]
    assert "world_resolver" not in queued_track
    assert "_world_force_first_metric_publication" not in queued_track
    assert queued_track["world_resolver_confidence"] == 0.9
    assert queued_track["world_resolver_selected_id"] == "floor_ray"
    assert "world_resolver" in track
    assert track["_world_force_first_metric_publication"] is True


def test_universal_resolver_source_continuity_is_evidence_not_a_veto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor()
    state = SimpleNamespace()
    track: dict[str, object] = {"world_resolver": {"decision": "good"}}
    monkeypatch.setattr(processor, "_admit_live_world_source", lambda *_args, **_kwargs: False)

    continuity_match = processor._observe_resolver_source_continuity(
        track,
        state,
        candidate_source="pose_floor_only",
        quality="good",
        depth_weight=0.0,
        posture="standing",
    )

    assert continuity_match is False
    assert track["world_resolver_source_continuity_match"] is False
    assert track["world_resolver"]["source_continuity_match"] is False  # type: ignore[index]
