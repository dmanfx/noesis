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
    variance: float = 0.04,
    confidence: float = 0.9,
    run_id: str = "source-run",
    zone: str | None = None,
    zone_source: str | None = None,
    zone_authoritative: bool = False,
) -> ObservationEnvelope:
    fingerprint = ArtifactFingerprint(role="input", sha256=SHA)
    tracklet = TrackletRef(
        run_id=run_id,
        camera_id=camera_id,
        source_id=source_id,
        tracker_id=sequence,
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
                covariance=Matrix3(values=(variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance)),
                frame="backend_world_m",
                units="meters",
                source="pose_depth_fused",
                quality="good",
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
    fusion.ingest(_observation("kitchen", 0, 1, 2_000_000, 2.0, run_id="run-b"), _subject())
    entity = fusion.snapshot(published_at_us=2_010_000).entities[0]
    assert entity.position is not None
    assert entity.position.x == pytest.approx(2.0)
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
