from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from noesis.calibration.world_fusion_policy import (
    CameraWorldFusionProfile,
    WorldFusionPolicy,
)
from noesis.calibration.pose_v1 import pose_to_E_col_major
from noesis.pipelines import hooks
from noesis.telemetry.person_ground_state import PersonGroundState
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.runtime_publication import RuntimePublicationGate
from noesis_core.world import GlobalWorldFusion
from noesis_core.world_service import CanonicalWorldService, WorldArtifacts


def _extrinsics_for_camera(camera_world: tuple[float, float, float]) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = -np.asarray(camera_world, dtype=np.float64)
    return list(matrix.flatten(order="F"))


def _calibration(
    camera_world: tuple[float, float, float],
    *,
    revision: str,
    transform: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        camera_id="cam0",
        camera_calibration_sha256="b" * 64,
        intrinsics=np.array(
            [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        extrinsics_col_major=_extrinsics_for_camera(camera_world),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
        world_frame_id="backend_world_m",
        world_frame_revision=revision,
        frame_transform_sha256=transform,
    )


def _processor(raw: SimpleNamespace, active: SimpleNamespace) -> hooks._AnalyticsTelemetryProcessor:
    provider = SimpleNamespace(
        snapshot=lambda _source_id, _camera_id: raw,
        world_snapshot=lambda _source_id, _camera_id: active,
    )
    policy = WorldFusionPolicy(
        policy_id="world-boundary-test",
        evidence={},
        cameras={
            "cam0": CameraWorldFusionProfile(
                camera_id="cam0",
                calibration_fingerprint_sha256="a" * 64,
                registration_id="reg-test",
                floor_weight_scale=1.0,
                depth_weight_scale=1.0,
                floor_only_allowed=True,
                floor_ray_max_range_m=20.0,
            )
        },
    )
    return hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={"models": {"pose": {"kpt_threshold": 0.35}}}),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=provider,
        world_fusion_policy=policy,
    )


def _seeded_track(tracker_id: int, *, source: str, world: list[float]) -> dict[str, object]:
    return {
        "tracker_id": tracker_id,
        "bbox": [100.0, 100.0, 80.0, 180.0],
        "bbox3d": {"x": world[0], "y": world[1], "z": world[2]},
        "world": list(world),
        "world_valid": True,
        "world_source": source,
        "world_frame": "backend_world_m",
    }


def _canonical_world_service() -> CanonicalWorldService:
    producer = ProducerRef(
        runtime="ds9",
        instance_id="test",
        run_id="run-seeded",
        software_revision="test-revision",
    )
    return CanonicalWorldService(
        producer=producer,
        artifacts=WorldArtifacts(
            calibration=ArtifactFingerprint(
                role="camera_calibration",
                sha256="c" * 64,
            ),
            model=ArtifactFingerprint(role="tracking_models", sha256="d" * 64),
            config=ArtifactFingerprint(role="runtime_config", sha256="e" * 64),
        ),
        fusion=GlobalWorldFusion(producer),
        clock_us=lambda: 101_000_000,
    )


def _complete_seeded_track(
    tracker_id: int,
    *,
    frame_id: int,
    media_pts_ns: int,
    world: list[float],
) -> dict[str, object]:
    track = _seeded_track(tracker_id, source="bbox3d", world=world)
    track.update(
        {
            "camera_id": "cam0",
            "source_id": 0,
            "frame_id": frame_id,
            "observed_at_us": media_pts_ns // 1_000,
            "media_pts_ns": media_pts_ns,
            "tracker_lifecycle_generation": 1,
            "image_size": [1280, 720],
            "confidence": 0.95,
            "tracker_confidence": 0.93,
            "stable_id": 41,
            "identity_kind": "resident",
            "resident_uuid": "resident-seeded",
        }
    )
    return track


def _v3dt_pose_recovery_case():
    calibration = _calibration((0.0, 2.5, 0.0), revision="pose-test", transform="a" * 64)
    calibration.extrinsics_col_major = pose_to_E_col_major({
        "position": [0.0, 2.5, 0.0],
        "yaw_pitch_roll_deg": [0.0, 15.0, 0.0],
        "rotation_order": "YXZ", "frame": "backend_world_m",
    })
    processor = _processor(calibration, calibration)
    processor._tracking_mode = "v3dt"
    first = _complete_seeded_track(29, frame_id=1, media_pts_ns=100_000_000_000, world=[0.0, 0.0, 6.0])
    processor._refine_seeded_world_with_ground_state(
        0, "cam0", first, world_source_label="bbox3d", world_now_ts=100.0,
    )
    assert first["world_valid"] is True
    processor._commit_enqueued_world_output_watermarks(0, [first], filter_ts=100.0)
    processor._clear_absent_world_state(0, [], now_ts=100.05)
    projection = calibration.intrinsics @ np.array(calibration.extrinsics_col_major).reshape(4, 4, order="F")[:3]

    def sample(frame_id, timestamp, *, pose_x=3.1, current=True):
        track = _complete_seeded_track(29, frame_id=frame_id, media_pts_ns=int(timestamp * 1e9), world=[3.0, 0.0, 6.0])
        foot = projection @ np.array([3.0, 0.0, 6.0, 1.0])
        top = projection @ np.array([3.0, 1.7, 6.0, 1.0])
        foot, top = foot[:2] / foot[2], top[:2] / top[2]
        track["bbox"] = [float(foot[0] - 45), float(top[1]), 90.0, float(foot[1] - top[1] + 8)]
        keypoints = None
        if pose_x is not None:
            observed = projection @ np.array([pose_x, 0.0, 6.0, 1.0])
            u, v = observed[:2] / observed[2]
            keypoints = np.zeros((17, 3), dtype=float)
            keypoints[15] = [u - 3, v, 0.99]
            keypoints[16] = [u + 3, v, 0.99]
        processor._refine_seeded_world_with_ground_state(
            0, "cam0", track, world_source_label="bbox3d", world_now_ts=timestamp,
            pose_kpts_abs=keypoints, pose_is_current=current,
        )
        return track

    return processor, sample


def test_v3dt_recovery_uses_three_current_pose_proofs_and_keeps_sdk_measurement():
    processor, sample = _v3dt_pose_recovery_case()
    first = sample(2, 100.2)
    assert first["world_valid"] is False
    assert first["world_reacquire_count"] == 1
    missing = sample(3, 100.3, pose_x=None)
    assert missing["world_valid"] is False
    assert missing["world_reacquire_count"] == 1
    second = sample(4, 100.4)
    assert second["world_valid"] is False
    assert second["world_reacquire_count"] == 2
    recovered = sample(5, 100.6)
    assert recovered["world_valid"] is True
    assert recovered["world_source"] == "bbox3d"
    assert recovered["world_reacquired"] is True
    assert recovered["trail_break_required"] is True
    assert recovered["world"] == pytest.approx([3.0, 0.0, 6.0])
    assert recovered["world_prefilter_measurement"] == [3.0, 0.0, 6.0]
    assert not processor._world_state_by_track[(0, 29)].post_ghost_position_support_required


@pytest.mark.parametrize("proof", ("missing", "cached", "contradictory"))
def test_v3dt_recovery_cannot_be_authorized_by_missing_cached_or_conflicting_pose(proof):
    _processor_instance, sample = _v3dt_pose_recovery_case()
    for index in range(4):
        track = sample(
            index + 2, 100.2 + index * 0.1,
            pose_x=None if proof == "missing" else 0.0 if proof == "contradictory" else 3.1,
            current=proof != "cached",
        )
        assert track["world_valid"] is False
        assert not track["world_v3dt_pose_support"]
        assert track["world_reacquire_count"] == 0
        assert "world" not in track


def test_v3dt_conflicting_current_pose_clears_pending_recovery():
    _processor_instance, sample = _v3dt_pose_recovery_case()
    assert sample(2, 100.2)["world_reacquire_count"] == 1
    conflict = sample(3, 100.3, pose_x=0.0)
    assert conflict["world_v3dt_pose_support_reason"] == "pose_sdk_ground_disagreement"
    assert conflict["world_reacquire_count"] == 0
    assert sample(4, 100.4)["world_reacquire_count"] == 1


@pytest.mark.parametrize("tamper", ("pose_cache_reused", "pose_cache_age_frames", "source_id", "object_id", "frame_id", "ts_us"))
def test_v3dt_pose_proof_rejects_cache_and_cohort_mismatch(tamper):
    payload = {"source_id": 0, "object_id": 29, "frame_id": 8, "ts_us": 100_200_000}
    cohort = (0, 29, 8, 100_200_000_000)
    assert hooks._AnalyticsTelemetryProcessor._pose_payload_is_current(payload, cohort)
    payload[tamper] = True if tamper == "pose_cache_reused" else int(payload.get(tamper, 0)) + 1
    assert not hooks._AnalyticsTelemetryProcessor._pose_payload_is_current(payload, cohort)


def test_v3dt_cached_pose_remains_present_without_becoming_recovery_proof(monkeypatch):
    processor, _sample = _v3dt_pose_recovery_case()
    payload = {
        "source_id": 0, "object_id": 29, "frame_id": 8, "ts_us": 100_200_000,
        "pose_cache_reused": True, "pose_cache_age_frames": 2,
        "keypoints_abs": np.ones((17, 3)).tolist(),
    }
    monkeypatch.setattr(processor, "_extract_pose_payload_for_anchor", lambda _obj: payload)
    status = {}
    points = processor._extract_pose_keypoints_for_anchor(
        object(), [0, 0, 100, 200],
        current_cohort=(0, 29, 8, 100_200_000_000), current_cohort_status=status,
    )
    assert points is not None
    assert status == {"current": False}


def test_v3dt_invalid_payload_cannot_lend_freshness_to_native_fallback(monkeypatch):
    processor, _sample = _v3dt_pose_recovery_case()
    processor._pose_anchor_native_remaining = 1
    payload = {"source_id": 0, "object_id": 29, "frame_id": 8, "ts_us": 100_200_000}
    monkeypatch.setattr(processor, "_extract_pose_payload_for_anchor", lambda _obj: payload)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", SimpleNamespace(
        extract_pose_keypoints=lambda *_args: {"keypoints_abs": np.ones((17, 3)).tolist()}
    ))
    status = {}
    points = processor._extract_pose_keypoints_for_anchor(
        object(), [0, 0, 100, 200],
        current_cohort=(0, 29, 8, 100_200_000_000), current_cohort_status=status,
    )
    assert points is not None
    assert status == {"current": False}


def test_v3dt_duplicate_pose_cohort_cannot_advance_recovery_consensus():
    _processor_instance, sample = _v3dt_pose_recovery_case()
    assert sample(2, 100.2)["world_reacquire_count"] == 1
    for _ in range(4):
        duplicate = sample(2, 100.2)
        assert duplicate["world_reacquire_count"] == 1
        assert duplicate["world_valid"] is False
    assert sample(3, 100.4)["world_reacquire_count"] == 2
    assert sample(4, 100.6)["world_reacquired"] is True


def test_bbox3d_range_uses_revision_bound_world_snapshot_with_nonidentity_transform() -> None:
    # The active target-frame camera is translated 10 m from the raw source
    # frame. The candidate is inside the 20 m envelope only in that active
    # frame; a raw-snapshot check would reject it.
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    track = _seeded_track(11, source="bbox3d", world=[20.0, 0.0, 10.0])

    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )

    assert track["world_valid"] is True
    assert track["world_frame_revision"] == "rev-1"
    assert track["world_observation_range_admitted"] is True
    assert track["world_observation_range_m"] < 20.0


def test_non_bbox3d_preseed_is_admitted_or_rejected_by_canonical_range_gate() -> None:
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    track = _seeded_track(12, source="external_preseed", world=[40.0, 0.0, 10.0])
    track["world_frame_revision"] = "rev-1"
    track["world_transform_sha256"] = "active"
    track.pop("bbox3d")

    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "world_observation_range_exceeded"
    assert track["world_observation_range_admitted"] is False


def test_out_of_range_seeded_hold_is_invalid_and_does_not_publish() -> None:
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    processor._world_state_by_track[(0, 13)] = PersonGroundState(  # type: ignore[attr-defined]
        world_x=40.0,
        world_z=10.0,
        filtered_ts=99.0,
        last_good_world=(40.0, 0.0, 10.0),
        last_good_ts=99.0,
    )
    track = _seeded_track(13, source="bbox3d", world=[40.0, 0.0, 10.0])

    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "world_observation_range_exceeded"
    assert track["trail_append_allowed"] is False


def test_output_speed_rejected_seed_roundtrips_as_strict_output_hold() -> None:
    calibration = _calibration(
        (0.0, 2.2, -6.0),
        revision="rev-1",
        transform="a" * 64,
    )
    processor = _processor(calibration, calibration)
    service = _canonical_world_service()
    first = _complete_seeded_track(
        21,
        frame_id=1,
        media_pts_ns=99_900_000_000,
        world=[0.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        first,
        world_source_label="bbox3d",
        world_now_ts=99.9,
    )
    assert first["world_measurement_accepted"] is True
    processor._commit_enqueued_world_output_watermarks(  # type: ignore[attr-defined]
        0,
        [first],
        filter_ts=99.9,
    )
    first_publication = service.publish(0, [first], metadata={})
    assert first_publication.observations[0].payload.world is not None

    state = processor._world_state_by_track[(0, 21)]  # type: ignore[attr-defined]
    state.motion_mode = "walk"
    state.vel_world_x = 0.0
    second = _complete_seeded_track(
        21,
        frame_id=2,
        media_pts_ns=100_000_000_000,
        world=[1.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        second,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )

    assert second["world_source"] == "anchor_hold"
    assert second["world_quality"] == "held"
    assert second["world_measurement_accepted"] is False
    assert second["world_rejection_reason"] == "physical_output_speed_exceeded"
    assert second["world"] == pytest.approx(first["world"])
    provenance = second["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["type"] == "bounded_output_hold"
    assert provenance["origin"] == "last_published_output"
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    assert transition["kind"] == "output_hold"
    assert transition["position_gain"] == pytest.approx(0.0)
    assert transition["origin_world"] == [0.0, 0.0, 6.0]
    assert transition["metric_origin_world"] == [0.0, 0.0, 6.0]

    publication = service.publish(0, [second], metadata={})

    world = publication.observations[0].payload.world
    assert world is not None
    assert world.source == "anchor_hold"
    assert world.quality == "held"
    assert world.position.x == pytest.approx(float(second["world"][0]))
    assert publication.snapshot.entities[0].position is not None
    assert publication.snapshot.entities[0].position.x == pytest.approx(
        float(second["world"][0])
    )


def test_uncommitted_output_speed_rejection_fails_closed_and_clears_state() -> None:
    calibration = _calibration(
        (0.0, 2.2, -6.0),
        revision="rev-1",
        transform="b" * 64,
    )
    processor = _processor(calibration, calibration)
    first = _complete_seeded_track(
        22,
        frame_id=1,
        media_pts_ns=99_900_000_000,
        world=[0.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        first,
        world_source_label="bbox3d",
        world_now_ts=99.9,
    )
    # Model a rate-suppressed callback: the filter saw the metric point, but
    # the ordered tracking/world/BEV queue never committed it.
    state = processor._world_state_by_track[(0, 22)]  # type: ignore[attr-defined]
    state.motion_mode = "walk"
    state.vel_world_x = 0.0
    second = _complete_seeded_track(
        22,
        frame_id=2,
        media_pts_ns=100_000_000_000,
        world=[1.0, 0.0, 6.0],
    )

    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        second,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )

    assert second["world_valid"] is False
    assert "world" not in second
    assert "world_source" not in second
    assert "world_prediction_provenance" not in second
    assert second["world_quality_reason"] == (
        "canonical_continuity_provenance_unavailable"
    )
    assert second["trail_append_allowed"] is False
    assert state.last_output_world_x is None
    assert state.last_output_world_z is None
    assert state.last_output_media_pts_ns is None


@pytest.mark.parametrize("ghost_gap_s", (0.2, 0.6))
def test_restored_short_ghost_requires_current_position_evidence(
    ghost_gap_s: float,
) -> None:
    calibration = _calibration(
        (0.0, 2.2, -6.0),
        revision="rev-1",
        transform="f" * 64,
    )
    processor = _processor(calibration, calibration)
    first = _complete_seeded_track(
        25,
        frame_id=1,
        media_pts_ns=100_000_000_000,
        world=[0.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        first,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )
    assert first["world_valid"] is True
    processor._commit_enqueued_world_output_watermarks(  # type: ignore[attr-defined]
        0,
        [first],
        filter_ts=100.0,
    )
    processor._clear_absent_world_state(  # type: ignore[attr-defined]
        0,
        [],
        now_ts=100.05,
    )

    media_pts_ns = int(round((100.0 + ghost_gap_s) * 1_000_000_000.0))
    restored = _complete_seeded_track(
        25,
        frame_id=2,
        media_pts_ns=media_pts_ns,
        world=[5.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        restored,
        world_source_label="bbox3d",
        world_now_ts=100.0 + ghost_gap_s,
    )

    assert restored["world_state_continuity"] == "restored_short_ghost"
    assert restored["world_valid"] is False
    assert "world" not in restored
    assert "world_source" not in restored
    assert restored["world_quality_reason"] == (
        "restored_short_ghost_requires_current_position_evidence"
    )
    assert restored["trail_append_allowed"] is False

    persistent = _complete_seeded_track(
        25,
        frame_id=3,
        media_pts_ns=media_pts_ns + 100_000_000,
        world=[5.2, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        persistent,
        world_source_label="bbox3d",
        world_now_ts=100.1 + ghost_gap_s,
    )

    assert persistent.get("world_state_continuity") is None
    assert persistent["world_valid"] is False
    assert "world" not in persistent
    assert persistent["world_quality_reason"] == (
        "restored_short_ghost_requires_current_position_evidence"
    )


def test_restored_short_ghost_accepts_current_metric_recovery() -> None:
    calibration = _calibration(
        (0.0, 2.2, -6.0),
        revision="rev-1",
        transform="9" * 64,
    )
    processor = _processor(calibration, calibration)
    first = _complete_seeded_track(
        26,
        frame_id=1,
        media_pts_ns=100_000_000_000,
        world=[0.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        first,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )
    processor._commit_enqueued_world_output_watermarks(  # type: ignore[attr-defined]
        0,
        [first],
        filter_ts=100.0,
    )
    processor._clear_absent_world_state(  # type: ignore[attr-defined]
        0,
        [],
        now_ts=100.05,
    )

    restored = _complete_seeded_track(
        26,
        frame_id=2,
        media_pts_ns=100_200_000_000,
        world=[0.2, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        restored,
        world_source_label="bbox3d",
        world_now_ts=100.2,
    )

    assert restored["world_state_continuity"] == "restored_short_ghost"
    assert restored["world_valid"] is True
    assert restored["world_measurement_accepted"] is True
    assert restored["world_source"] == "bbox3d"


@pytest.mark.parametrize(
    "tamper",
    ("missing", "partial_transition", "forged_origin"),
)
def test_noncanonical_process_row_cannot_seed_image_motion_watermark(
    tamper: str,
) -> None:
    calibration = _calibration(
        (0.0, 2.2, -6.0),
        revision="rev-1",
        transform="c" * 64,
    )
    processor = _processor(calibration, calibration)
    first = _complete_seeded_track(
        23,
        frame_id=1,
        media_pts_ns=99_900_000_000,
        world=[0.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        first,
        world_source_label="bbox3d",
        world_now_ts=99.9,
    )
    processor._commit_enqueued_world_output_watermarks(  # type: ignore[attr-defined]
        0,
        [first],
        filter_ts=99.9,
    )

    state = processor._world_state_by_track[(0, 23)]  # type: ignore[attr-defined]
    state.motion_mode = "walk"
    state.vel_world_x = 0.0
    track = _complete_seeded_track(
        23,
        frame_id=2,
        media_pts_ns=100_000_000_000,
        world=[1.0, 0.0, 6.0],
    )
    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )
    assert track["world_source"] == "anchor_hold"
    assert track["world_prediction_provenance"]
    if tamper == "missing":
        track.pop("world_prediction_provenance")
    elif tamper == "partial_transition":
        provenance = dict(track["world_prediction_provenance"])
        provenance["filter_transition"] = {}
        track["world_prediction_provenance"] = provenance
    else:
        provenance = dict(track["world_prediction_provenance"])
        transition = dict(provenance["filter_transition"])
        transition["origin_world"] = [99.0, 0.0, 99.0]
        provenance["filter_transition"] = transition
        track["world_prediction_provenance"] = provenance

    processor._commit_enqueued_world_output_watermarks(  # type: ignore[attr-defined]
        0,
        [track],
        filter_ts=100.0,
    )
    key = processor._world_output_watermark_key(  # type: ignore[attr-defined]
        0,
        track,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-1",
        world_transform_sha256="c" * 64,
    )

    # Projective integration requires this exact committed reference.  A
    # malformed process row cannot replace the legitimate metric origin with
    # its newer PTS and become the next image-motion origin.
    assert processor._world_output_reference(key) == pytest.approx(  # type: ignore[attr-defined]
        (0.0, 6.0, 99_900_000_000, 99.9, 0)
    )


def test_refinement_exception_clears_arbitrary_world_valid() -> None:
    provider = SimpleNamespace(
        snapshot=lambda _source_id, _camera_id: (_ for _ in ()).throw(RuntimeError("calibration boom")),
        world_snapshot=lambda _source_id, _camera_id: (_ for _ in ()).throw(RuntimeError("calibration boom")),
    )
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=provider,
    )
    track = _seeded_track(14, source="external_preseed", world=[1.0, 0.0, 1.0])
    track.pop("bbox3d")

    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "world_refinement_failed"


def test_v3dt_bbox3d_without_calibration_is_invalid() -> None:
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    processor._tracking_mode = "v3dt"  # type: ignore[attr-defined]
    processor.bev_calibration = None
    track = _seeded_track(15, source="bbox3d", world=[1.0, 0.0, 1.0])

    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "calibration_unavailable"
