from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from noesis.calibration.world_fusion_policy import (
    CameraWorldFusionProfile,
    WorldFusionPolicy,
)
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
