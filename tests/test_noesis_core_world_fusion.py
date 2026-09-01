from __future__ import annotations

import pytest

from noesis_core.contracts.base import ArtifactFingerprint, Matrix3, ProducerRef, Vector3
from noesis_core.contracts.identity import IdentityKind, SubjectRef, TrackletRef
from noesis_core.contracts.observation import ObservationEnvelope, PersonObservation, WorldPositionObservation
from noesis_core.contracts.world import EntityLifecycle
from noesis_core.world import GlobalWorldFusion, ObservationOrderError, WorldFusionConfig


SHA = "a" * 64


def _producer(run_id: str = "source-run") -> ProducerRef:
    return ProducerRef(runtime="ds8", instance_id="appliance", run_id=run_id, software_revision="rev")


def _world_producer() -> ProducerRef:
    return ProducerRef(runtime="ds8", instance_id="appliance", run_id="world-run", software_revision="rev")


def _subject() -> SubjectRef:
    return SubjectRef(
        subject_id="resident:abc",
        kind=IdentityKind.RESIDENT,
        resident_uuid="abc",
        display_name="Resident",
        stable_id=1,
    )


def _observation(
    camera_id: str,
    source_id: int,
    sequence: int,
    observed_at_us: int,
    x: float,
    *,
    tracker_id: int | None = None,
    media_pts_ns: int | None = None,
    variance: float = 0.04,
    covariance: tuple[float, ...] | None = None,
    confidence: float = 0.9,
    quality: str = "good",
    source: str = "pose_depth_fused",
    run_id: str = "source-run",
    zone: str | None = None,
    zone_source: str | None = None,
    zone_authoritative: bool = False,
    world_frame_revision: str | None = None,
    world_transform_sha256: str | None = None,
    calibration_revision: str | None = None,
) -> ObservationEnvelope:
    fingerprint = ArtifactFingerprint(role="input", sha256=SHA)
    tracklet = TrackletRef(
        run_id=run_id,
        camera_id=camera_id,
        source_id=source_id,
        tracker_id=sequence if tracker_id is None else tracker_id,
        frame_id=sequence,
        observed_at_us=observed_at_us,
    )
    return ObservationEnvelope(
        contract="noesis.observation.person",
        contract_version=1,
        observation_id=f"{run_id}:{camera_id}:{sequence}",
        producer=_producer(run_id),
        sequence=sequence,
        captured_at_us=observed_at_us - 10,
        observed_at_us=observed_at_us,
        published_at_us=observed_at_us + 10,
        capture_time_status="estimated",
        media_pts_ns=media_pts_ns,
        coordinate_frame="backend_world_m",
        units="meters",
        calibration=fingerprint.model_copy(update={"role": "calibration"}),
        model=fingerprint.model_copy(update={"role": "model"}),
        config=fingerprint.model_copy(update={"role": "config"}),
        payload=PersonObservation(
            tracklet=tracklet,
            bbox_xywh=(1.0, 2.0, 3.0, 4.0),
            image_size=(1920, 1080),
            zone=zone,
            zone_source=zone_source,  # type: ignore[arg-type]
            zone_authoritative=zone_authoritative,
            world=WorldPositionObservation(
                position=Vector3(x=x, y=0.0, z=2.0),
                covariance=Matrix3(
                    values=covariance
                    or (variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance)
                ),
                frame="backend_world_m",
                units="meters",
                world_frame_revision=world_frame_revision,
                world_transform_sha256=world_transform_sha256,
                calibration_revision=calibration_revision,
                source=source,
                quality=quality,  # type: ignore[arg-type]
                confidence=confidence,
            ),
        ),
    )


def test_fuses_compatible_contemporaneous_camera_observations() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(_observation("kitchen", 0, 1, 1_000_000, 1.0), _subject())
    fusion.ingest(_observation("family-room", 1, 1, 1_050_000, 1.2), _subject())

    snapshot = fusion.snapshot(published_at_us=1_100_000)
    entity = snapshot.entities[0]
    assert entity.lifecycle == EntityLifecycle.PRESENT
    assert entity.conflict is False
    assert entity.position is not None
    assert entity.position.x == pytest.approx(1.1)
    assert len(entity.sources) == 2
    assert all(source.accepted for source in entity.sources)


def test_held_continuation_does_not_pull_fresh_metric_camera_position() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation("hall", 0, 1, 1_000_000, 1.1),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "kitchen",
            1,
            1,
            1_050_000,
            1.2,
            variance=0.64,
            confidence=0.35,
            quality="held",
            source="cv_prediction",
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.lifecycle == EntityLifecycle.PRESENT
    assert entity.position is not None
    assert entity.position.x == pytest.approx(1.1)
    assert entity.conflict is False
    by_camera = {source.camera_id: source for source in entity.sources}
    assert by_camera["hall"].accepted is True
    assert by_camera["kitchen"].accepted is False
    assert by_camera["kitchen"].rejection_reason == (
        "non_authoritative_held_continuation"
    )


def test_only_held_continuations_select_newest_exact_point_without_averaging() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "hall",
            0,
            1,
            1_000_000,
            1.0,
            variance=0.64,
            quality="held",
            source="anchor_hold",
        ),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "kitchen",
            1,
            1,
            1_050_000,
            1.2,
            variance=0.64,
            quality="held",
            source="cv_prediction",
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.lifecycle == EntityLifecycle.PRESENT
    assert entity.position is not None
    assert entity.position.x == pytest.approx(1.2)
    assert entity.velocity_mps is None
    assert entity.conflict is False
    by_camera = {source.camera_id: source for source in entity.sources}
    assert by_camera["kitchen"].accepted is True
    assert by_camera["hall"].accepted is False
    assert by_camera["hall"].rejection_reason == (
        "non_authoritative_held_continuation"
    )


def test_implausible_fused_step_retains_prior_position_without_advancing_baseline() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation("kitchen", 0, 1, 1_000_000, 0.0),
        _subject(),
    )
    first = fusion.snapshot(published_at_us=1_050_000).entities[0]
    assert first.velocity_mps is None

    fusion.ingest(
        _observation("kitchen", 0, 2, 2_000_000, 4.1),
        _subject(),
    )
    second = fusion.snapshot(published_at_us=2_050_000).entities[0]

    assert second.position is not None
    assert second.position.x == pytest.approx(0.0)
    assert second.velocity_mps is None
    assert second.lifecycle == EntityLifecycle.HELD
    assert second.observed_at_us == 1_000_000
    assert second.stale_after_us == 3_000_000
    assert second.conflict is True
    assert second.conflict_reason == (
        "implausible same-identity position step rejected; prior fused position retained"
    )
    current = next(
        source
        for source in second.sources
        if source.observation_id == "source-run:kitchen:2"
    )
    assert current.accepted is False
    assert current.rejection_reason == (
        "velocity_gate:4.100m/1.000s=4.100mps>4.000mps"
    )
    type(second).model_validate(second.model_dump(mode="python"))

    # This physically plausible recovery is admitted relative to the original
    # x=0 baseline. It would fail if the rejected x=4.1 point had advanced the
    # admission clock and position.
    fusion.ingest(
        _observation("kitchen", 0, 3, 2_500_000, 2.0),
        _subject(),
    )
    recovered = fusion.snapshot(published_at_us=2_550_000).entities[0]

    assert recovered.position is not None
    assert recovered.position.x == pytest.approx(2.0)
    assert recovered.velocity_mps is not None
    assert recovered.velocity_mps.x == pytest.approx(4.0 / 3.0)
    assert recovered.conflict is False


def test_same_source_velocity_admission_uses_monotonic_media_time() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            0.0,
            tracker_id=7,
            media_pts_ns=1_000_000_000,
        ),
        _subject(),
    )
    fusion.snapshot(published_at_us=1_010_000)

    # Arrival jitter compresses this 0.4517 m step into 0.0879 s (5.139 m/s),
    # while the authoritative same-source media clock spans 0.1667 s
    # (2.710 m/s). Physical admission must use the latter.
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            2,
            1_087_900,
            0.4517,
            tracker_id=7,
            media_pts_ns=1_166_700_000,
        ),
        _subject(),
    )
    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.position is not None
    assert entity.position.x == pytest.approx(0.4517)
    assert entity.velocity_mps is not None
    assert entity.velocity_mps.x == pytest.approx(0.4517 / 0.1667)
    assert entity.conflict is False


def test_source_epoch_change_resets_same_run_media_velocity_baseline() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            0.0,
            tracker_id=7,
            media_pts_ns=100_000_000_000,
        ),
        _subject(),
        source_epoch=0,
    )
    fusion.snapshot(published_at_us=1_010_000)

    # The process run is unchanged, but this reconnect starts a distinct media
    # timeline at 1 s. Arrival time is only 100 ms later, so treating the rewind
    # as an ordinary arrival-clock step would reject this valid new baseline.
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            2,
            1_100_000,
            10.0,
            tracker_id=7,
            media_pts_ns=1_000_000_000,
        ),
        _subject(),
        source_epoch=1,
    )
    reset = fusion.snapshot(published_at_us=1_110_000).entities[0]

    assert reset.position is not None
    assert reset.position.x == pytest.approx(10.0)
    assert reset.velocity_mps is None
    assert reset.conflict is False

    # Once seeded in epoch 1, the next point again uses that epoch's monotonic
    # media clock and produces an ordinary bounded velocity.
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            3,
            1_200_000,
            10.2,
            tracker_id=7,
            media_pts_ns=1_100_000_000,
        ),
        _subject(),
        source_epoch=1,
    )
    continued = fusion.snapshot(published_at_us=1_210_000).entities[0]

    assert continued.position is not None
    assert continued.position.x == pytest.approx(10.2)
    assert continued.velocity_mps is not None
    assert continued.velocity_mps.x == pytest.approx(2.0)
    assert continued.conflict is False


def test_velocity_rejected_position_is_retained_only_until_existing_lost_ttl() -> None:
    fusion = GlobalWorldFusion(
        _world_producer(),
        config=WorldFusionConfig(present_ttl_us=100, lost_ttl_us=500),
    )
    fusion.ingest(_observation("kitchen", 0, 1, 1_000, 0.0), _subject())
    assert fusion.snapshot(published_at_us=1_050).entities[0].position.x == pytest.approx(0.0)

    fusion.ingest(_observation("kitchen", 0, 2, 1_200, 1.0), _subject())
    retained = fusion.snapshot(published_at_us=1_250).entities[0]
    assert retained.position is not None
    assert retained.position.x == pytest.approx(0.0)
    assert retained.lifecycle == EntityLifecycle.HELD

    expired = fusion.snapshot(published_at_us=1_600)
    assert expired.entities == ()


def test_velocity_admission_gate_spans_camera_handoff_in_same_world_registration() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            0.0,
            world_frame_revision="world-r1",
            world_transform_sha256="a" * 64,
        ),
        _subject(),
    )
    fusion.snapshot(published_at_us=1_050_000)

    fusion.ingest(
        _observation(
            "family-room",
            1,
            1,
            2_000_000,
            4.1,
            world_frame_revision="world-r1",
            world_transform_sha256="b" * 64,
        ),
        _subject(),
    )
    entity = fusion.snapshot(published_at_us=2_050_000).entities[0]

    assert entity.position is not None
    assert entity.position.x == pytest.approx(0.0)
    family_evidence = next(
        source for source in entity.sources if source.camera_id == "family-room"
    )
    assert family_evidence.accepted is False
    assert family_evidence.rejection_reason and family_evidence.rejection_reason.startswith(
        "velocity_gate:"
    )


def test_registration_change_resets_velocity_admission_baseline() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            0.0,
            world_frame_revision="world-r1",
            world_transform_sha256="a" * 64,
        ),
        _subject(),
    )
    fusion.snapshot(published_at_us=1_050_000)

    fusion.ingest(
        _observation(
            "family-room",
            1,
            1,
            2_000_000,
            10.0,
            world_frame_revision="world-r2",
            world_transform_sha256="b" * 64,
        ),
        _subject(),
    )
    entity = fusion.snapshot(published_at_us=2_050_000).entities[0]

    assert entity.position is not None
    assert entity.position.x == pytest.approx(10.0)
    assert entity.conflict is False
    assert entity.velocity_mps is None


def test_explicit_trail_segment_break_resets_velocity_gate_exactly_once() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            0.0,
            tracker_id=7,
        ),
        _subject(),
        trail_segment_id=9,
        tracker_lifecycle_generation=1,
    )
    fusion.snapshot(published_at_us=1_050_000)

    fusion.ingest(
        _observation(
            "kitchen",
            0,
            2,
            1_100_000,
            10.0,
            tracker_id=7,
        ),
        _subject(),
        trail_segment_id=10,
        trail_break_required=True,
        tracker_lifecycle_generation=1,
    )
    reset = fusion.snapshot(published_at_us=1_150_000).entities[0]

    assert reset.position is not None
    assert reset.position.x == pytest.approx(10.0)
    assert reset.velocity_mps is None
    assert reset.conflict is False

    # Repeating the same explicit break flag cannot grant another reset. The
    # source/lifecycle/segment token was consumed by the exact reset point.
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            3,
            1_200_000,
            20.0,
            tracker_id=7,
        ),
        _subject(),
        trail_segment_id=10,
        trail_break_required=True,
        tracker_lifecycle_generation=1,
    )
    rejected = fusion.snapshot(published_at_us=1_250_000).entities[0]

    assert rejected.position is not None
    assert rejected.position.x == pytest.approx(10.0)
    assert rejected.conflict is True
    current = next(
        source
        for source in rejected.sources
        if source.observation_id == "source-run:kitchen:3"
    )
    assert current.accepted is False
    assert current.rejection_reason and current.rejection_reason.startswith(
        "velocity_gate:"
    )


def test_changed_same_lifecycle_trail_segment_resets_velocity_gate_exactly_once() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            0.0,
            tracker_id=7,
        ),
        _subject(),
        trail_segment_id=1,
        tracker_lifecycle_generation=3,
    )
    fusion.snapshot(published_at_us=1_050_000)

    # The break pulse may have appeared on an earlier world-invalid row. The
    # segment transition remains an intrinsic one-time reset boundary.
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            2,
            1_100_000,
            10.0,
            tracker_id=7,
        ),
        _subject(),
        trail_segment_id=2,
        trail_break_required=False,
        tracker_lifecycle_generation=3,
    )
    reset = fusion.snapshot(published_at_us=1_150_000).entities[0]

    assert reset.position is not None
    assert reset.position.x == pytest.approx(10.0)
    assert reset.velocity_mps is None
    assert reset.conflict is False

    fusion.ingest(
        _observation(
            "kitchen",
            0,
            3,
            1_200_000,
            20.0,
            tracker_id=7,
        ),
        _subject(),
        trail_segment_id=2,
        trail_break_required=False,
        tracker_lifecycle_generation=3,
    )
    rejected = fusion.snapshot(published_at_us=1_250_000).entities[0]

    assert rejected.position is not None
    assert rejected.position.x == pytest.approx(10.0)
    assert rejected.conflict is True
    current = next(
        source
        for source in rejected.sources
        if source.observation_id == "source-run:kitchen:3"
    )
    assert current.accepted is False
    assert current.rejection_reason and current.rejection_reason.startswith(
        "velocity_gate:"
    )


def test_nonlatest_camera_trail_segment_change_resets_velocity_exactly_once() -> None:
    fusion = GlobalWorldFusion(_world_producer())

    def ingest_pair(
        *,
        sequence: int,
        observed_at_us: int,
        x: float,
        kitchen_segment: int,
        kitchen_break: bool = False,
    ) -> None:
        # Kitchen is deliberately the older member of each simultaneous pair.
        # A singleton "latest token" therefore observes only Family Room and
        # misses Kitchen's segment transition.
        fusion.ingest(
            _observation(
                "kitchen",
                0,
                sequence,
                observed_at_us,
                x,
                tracker_id=7,
            ),
            _subject(),
            trail_segment_id=kitchen_segment,
            trail_break_required=kitchen_break,
            tracker_lifecycle_generation=3,
        )
        fusion.ingest(
            _observation(
                "family-room",
                1,
                sequence,
                observed_at_us + 50_000,
                x,
                tracker_id=8,
            ),
            _subject(),
            trail_segment_id=1,
            tracker_lifecycle_generation=4,
        )

    ingest_pair(
        sequence=1,
        observed_at_us=1_000_000,
        x=0.0,
        kitchen_segment=1,
    )
    fusion.snapshot(published_at_us=1_060_000)

    ingest_pair(
        sequence=2,
        observed_at_us=1_100_000,
        x=10.0,
        kitchen_segment=2,
    )
    reset = fusion.snapshot(published_at_us=1_160_000).entities[0]

    assert reset.position is not None
    assert reset.position.x == pytest.approx(10.0)
    assert reset.velocity_mps is None
    assert reset.conflict is False

    # A delayed explicit pulse for the already-consumed segment transition
    # cannot spend the same per-source reset a second time.
    ingest_pair(
        sequence=3,
        observed_at_us=1_200_000,
        x=20.0,
        kitchen_segment=2,
        kitchen_break=True,
    )
    rejected = fusion.snapshot(published_at_us=1_260_000).entities[0]

    assert rejected.position is not None
    assert rejected.position.x == pytest.approx(10.0)
    assert rejected.conflict is True
    assert rejected.conflict_reason == (
        "implausible same-identity position step rejected; prior fused position retained"
    )


def test_covariance_precision_is_not_multiplied_by_duplicate_confidence() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            0.0,
            variance=0.04,
            confidence=0.1,
        ),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "family-room",
            1,
            1,
            1_050_000,
            1.0,
            variance=0.04,
            confidence=0.9,
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.position is not None
    assert entity.position.x == pytest.approx(0.5)
    assert entity.covariance is not None
    # CI deliberately does not claim independent camera errors; equal inputs
    # retain their original uncertainty instead of becoming overconfident.
    assert entity.covariance.values[0] == pytest.approx(0.04)


def test_correlated_covariance_is_preserved_by_full_matrix_fusion() -> None:
    covariance = (0.09, 0.03, 0.0, 0.03, 0.04, 0.0, 0.0, 0.0, 0.16)
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation("kitchen", 0, 1, 1_000_000, 1.0, covariance=covariance),
        _subject(),
    )
    fusion.ingest(
        _observation("hall", 1, 1, 1_050_000, 1.2, covariance=covariance),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.covariance is not None
    assert entity.covariance.values[1] == pytest.approx(0.03)
    assert entity.covariance.values[3] == pytest.approx(0.03)
    assert entity.covariance.values[0] == pytest.approx(0.09)


def test_mixed_target_frame_revisions_are_conflict_evidence_not_fused() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            1.0,
            world_frame_revision="world-r1",
            world_transform_sha256="a" * 64,
            calibration_revision="calibration-k",
        ),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "family-room",
            1,
            1,
            1_050_000,
            1.2,
            world_frame_revision="world-r2",
            world_transform_sha256="b" * 64,
            calibration_revision="calibration-f",
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.conflict is True
    assert sum(source.accepted for source in entity.sources) == 1
    assert {
        source.world_frame_revision for source in entity.sources
    } == {"world-r1", "world-r2"}
    assert any(
        source.rejection_reason == "registration_identity_conflict"
        for source in entity.sources
    )


def test_distinct_camera_edges_can_fuse_when_the_target_revision_matches() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            1.0,
            world_frame_revision="world-r1",
            world_transform_sha256="a" * 64,
            calibration_revision="calibration-k",
        ),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "family-room",
            1,
            1,
            1_050_000,
            1.2,
            world_frame_revision="world-r1",
            world_transform_sha256="b" * 64,
            calibration_revision="calibration-f",
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.conflict is False
    assert all(source.accepted for source in entity.sources)
    assert entity.world_frame_revision == "world-r1"
    assert entity.world_transform_sha256 is None
    assert entity.calibration_revision is None
    assert {source.world_transform_sha256 for source in entity.sources} == {
        "a" * 64,
        "b" * 64,
    }


def test_missing_target_frame_identity_is_not_fused_with_canonical_evidence() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            1.0,
            world_frame_revision="world-r1",
            world_transform_sha256="a" * 64,
        ),
        _subject(),
    )
    fusion.ingest(
        _observation("hall", 1, 1, 1_050_000, 1.2),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]

    assert entity.conflict is True
    assert sum(source.accepted for source in entity.sources) == 1
    assert any(
        source.rejection_reason == "registration_identity_missing"
        for source in entity.sources
    )


def test_conflicting_cameras_are_not_averaged() -> None:
    fusion = GlobalWorldFusion(
        _world_producer(),
        config=WorldFusionConfig(conflict_distance_m=1.0),
    )
    fusion.ingest(_observation("kitchen", 0, 1, 1_000_000, 1.0, variance=0.01), _subject())
    fusion.ingest(_observation("family-room", 1, 1, 1_010_000, 8.0, variance=0.20), _subject())

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]
    assert entity.conflict is True
    assert entity.position is not None
    assert entity.position.x == pytest.approx(1.0)
    assert sum(source.accepted for source in entity.sources) == 1
    assert any(source.rejection_reason and source.rejection_reason.startswith("position_conflict") for source in entity.sources)


def test_compatible_room_evidence_populates_canonical_room() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            1.0,
            zone="Kitchen",
            zone_source="nvdsanalytics_roi",
            zone_authoritative=True,
        ),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "hall",
            1,
            1,
            1_050_000,
            1.2,
            zone="Kitchen",
            zone_source="nvdsanalytics_roi",
            zone_authoritative=True,
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]
    assert entity.room_id == "Kitchen"
    assert {source.zone for source in entity.sources} == {"Kitchen"}


def test_differing_accepted_room_evidence_fails_closed_with_conflict() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            1.0,
            zone="Kitchen",
            zone_source="nvdsanalytics_roi",
            zone_authoritative=True,
        ),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "family-room",
            1,
            1,
            1_050_000,
            1.2,
            zone="FamilyRoom",
            zone_source="nvdsanalytics_roi",
            zone_authoritative=True,
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]
    assert entity.room_id is None
    assert entity.conflict is True
    assert {source.zone for source in entity.sources} == {
        "Kitchen",
        "FamilyRoom",
    }


def test_rejected_position_source_room_cannot_override_accepted_room() -> None:
    fusion = GlobalWorldFusion(
        _world_producer(),
        config=WorldFusionConfig(conflict_distance_m=0.5),
    )
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            1.0,
            zone="Kitchen",
            zone_source="nvdsanalytics_roi",
            zone_authoritative=True,
        ),
        _subject(),
    )
    fusion.ingest(
        _observation(
            "family-room",
            1,
            1,
            1_050_000,
            8.0,
            zone="FamilyRoom",
            zone_source="nvdsanalytics_roi",
            zone_authoritative=True,
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]
    admitted = next(source for source in entity.sources if source.accepted)
    rejected = next(source for source in entity.sources if not source.accepted)
    assert entity.room_id == admitted.zone
    assert entity.room_id != rejected.zone
    assert entity.conflict is True
    assert sum(source.accepted for source in entity.sources) == 1


def test_camera_default_zone_cannot_populate_canonical_room() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation(
            "kitchen",
            0,
            1,
            1_000_000,
            1.0,
            zone="Kitchen",
            zone_source="camera_default",
            zone_authoritative=False,
        ),
        _subject(),
    )

    entity = fusion.snapshot(published_at_us=1_100_000).entities[0]
    assert entity.room_id is None
    assert entity.sources[0].zone == "Kitchen"
    assert entity.sources[0].zone_source == "camera_default"
    assert entity.sources[0].zone_authoritative is False


def test_rejects_non_monotonic_source_sequence() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(_observation("kitchen", 0, 2, 1_000_000, 1.0), _subject())
    with pytest.raises(ObservationOrderError):
        fusion.ingest(_observation("kitchen", 0, 2, 1_100_000, 1.1), _subject())


def test_empty_source_clear_removes_evidence_without_resetting_order() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(
        _observation("kitchen", 0, 2, 1_000_000, 1.0),
        _subject(),
    )

    assert fusion.clear_source("kitchen", 0, "source-run") == 1
    assert fusion.snapshot(published_at_us=1_010_000).entities == ()

    with pytest.raises(ObservationOrderError):
        fusion.ingest(
            _observation("kitchen", 0, 2, 1_020_000, 1.1),
            _subject(),
        )
    fusion.ingest(
        _observation("kitchen", 0, 3, 1_030_000, 1.2),
        _subject(),
    )
    assert len(fusion.snapshot(published_at_us=1_040_000).entities) == 1


def test_new_source_run_replaces_old_run_without_order_collision() -> None:
    fusion = GlobalWorldFusion(_world_producer())
    fusion.ingest(_observation("kitchen", 0, 9, 1_000_000, 1.0, run_id="run-a"), _subject())
    fusion.ingest(_observation("kitchen", 0, 1, 2_000_000, 20.0, run_id="run-b"), _subject())
    entity = fusion.snapshot(published_at_us=2_010_000).entities[0]
    assert entity.position is not None
    assert entity.position.x == pytest.approx(20.0)
    assert entity.velocity_mps is None
    assert len(entity.sources) == 1


def test_entity_is_held_then_expires() -> None:
    fusion = GlobalWorldFusion(
        _world_producer(),
        config=WorldFusionConfig(present_ttl_us=100, lost_ttl_us=500),
    )
    fusion.ingest(_observation("kitchen", 0, 1, 1_000, 1.0), _subject())
    held = fusion.snapshot(published_at_us=1_200)
    assert held.entities[0].lifecycle == EntityLifecycle.HELD
    expired = fusion.snapshot(published_at_us=1_600)
    assert expired.entities == ()
