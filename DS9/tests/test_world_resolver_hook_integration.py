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
from noesis_core.contracts.world_measurement import (
    WorldMeasurementCohort,
    WorldMeasurementHypothesis,
    WorldMeasurementSet,
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


def _cohort() -> WorldMeasurementCohort:
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
    assert queued_track["world_resolver_confidence"] == 0.9
    assert queued_track["world_resolver_selected_id"] == "floor_ray"
    assert "world_resolver" in track


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
