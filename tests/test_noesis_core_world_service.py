from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from noesis.telemetry.publishers import (
    TrackingPublicationPoisoned,
    TrackingTelemetryPublisher,
)
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.health import CapabilityMonitor, CapabilityPolicy
from noesis_core.journal import AsyncContractJournal, ContractJournal
from noesis_core.world import GlobalWorldFusion, WorldFusionConfig
from noesis_core.world_service import CanonicalWorldService, WorldArtifacts
from noesis.server.websocket import WebSocketServer


def _producer() -> ProducerRef:
    return ProducerRef(
        runtime="ds8",
        instance_id="appliance",
        run_id="run-1",
        software_revision="revision-1",
    )


def _artifacts() -> WorldArtifacts:
    return WorldArtifacts(
        calibration=ArtifactFingerprint(role="camera_calibration", sha256="a" * 64),
        model=ArtifactFingerprint(role="tracking_models", sha256="b" * 64),
        config=ArtifactFingerprint(role="runtime_config", sha256="c" * 64),
    )


def _track(
    *,
    camera_id: str,
    tracker_id: int,
    frame_id: int,
    observed_at_us: int,
    x: float | None,
    identity_kind: str = "resident",
    resident_uuid: str | None = "resident-1",
    stable_id: int = 1,
    visitor_generation: int | None = None,
) -> dict[str, Any]:
    track: dict[str, Any] = {
        "camera_id": camera_id,
        "tracker_id": tracker_id,
        "frame_id": frame_id,
        "observed_at_us": observed_at_us,
        "capture_time_status": "estimated",
        "media_pts_ns": frame_id * 1_000,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "image_size": [1920, 1080],
        "confidence": 0.91,
        "tracker_confidence": 0.88,
        "stable_id": stable_id,
        "identity_kind": identity_kind,
        "resident_uuid": resident_uuid,
    }
    if x is not None:
        track.update(
            {
                "world": [x, 0.0, 2.0],
                "world_valid": True,
                "world_frame": "backend_world_m",
                "world_frame_revision": "world-rev",
                "world_transform_sha256": "d" * 64,
                "calibration_revision": "calibration-rev",
                "world_quality": "good",
                "world_source": "pose_depth_fused",
            }
        )
    if visitor_generation is not None:
        track["visitor_generation"] = visitor_generation
    return track


def _service(
    *,
    conflict_distance_m: float = 1.25,
    journal: ContractJournal | AsyncContractJournal | None = None,
) -> CanonicalWorldService:
    producer = _producer()
    return CanonicalWorldService(
        producer=producer,
        artifacts=_artifacts(),
        fusion=GlobalWorldFusion(
            producer,
            config=WorldFusionConfig(conflict_distance_m=conflict_distance_m),
        ),
        clock_us=lambda: 2_000_000,
        journal=journal,
    )


def test_resident_observations_across_cameras_fuse_into_one_entity() -> None:
    service = _service()
    first = service.publish(
        0,
        [_track(camera_id="kitchen", tracker_id=7, frame_id=1, observed_at_us=1_900_000, x=1.0)],
        metadata={"camera_id": "kitchen"},
    )
    second = service.publish(
        1,
        [_track(camera_id="hall", tracker_id=19, frame_id=4, observed_at_us=1_950_000, x=1.2)],
        metadata={"camera_id": "hall"},
    )

    assert len(first.observations) == 1
    assert len(second.snapshot.entities) == 1
    entity = second.snapshot.entities[0]
    assert entity.entity_id == "resident:resident-1"
    assert entity.position is not None
    assert entity.position.x == pytest.approx(1.1)
    assert {source.camera_id for source in entity.sources} == {"kitchen", "hall"}


def test_world_observation_preserves_resolver_anisotropic_covariance() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    covariance = [
        0.09,
        0.0,
        0.03,
        0.0,
        0.01,
        0.0,
        0.03,
        0.0,
        0.25,
    ]
    track["world_covariance"] = covariance
    track["world_resolver_confidence"] = 0.72
    track["world_quantity"] = "ground_footprint"
    track["world_support_state"] = "floor"
    track["world_posture"] = "standing"
    track["source_id"] = 0
    track["world_resolver"] = {
        "contract": "noesis.world_resolver_diagnostics",
        "version": 1,
        "camera_id": "kitchen",
        "source_id": 0,
        "tracker_id": 7,
        "frame_id": 1,
        "observed_at_us": 1_900_000,
        "world_frame": "backend_world_m",
        "world_frame_revision": "world-rev",
        "selected_id": "floor_ray",
        "selected_kind": "floor_ray",
        "contributor_ids": ["floor_ray"],
        "alternate_id": None,
        "fused": False,
        "confidence": 0.72,
        "pcf_score": 0.9,
        "agreement_mahalanobis_sq": None,
        "reason": "selected_single_valid_hypothesis",
        "candidates": [
            {
                "id": "floor_ray",
                "kind": "floor_ray",
                "score": 0.72,
                "pcf_score": 0.9,
                "innovation_m": None,
                "agreement_mahalanobis_sq": None,
                "compatible_with_selected": True,
                "selected": True,
                "retained_as_alternate": False,
                "rejection_reason": None,
            }
        ],
    }
    track["world_frame_revision"] = "world-rev"

    publication = service.publish(0, [track], metadata={})

    world = publication.observations[0].payload.world
    assert world is not None
    assert world.covariance.values == pytest.approx(tuple(covariance))
    assert world.quantity == "ground_footprint"
    assert world.support_state == "floor"
    assert world.posture == "standing"
    diagnostics = publication.observations[0].payload.world_diagnostics
    assert diagnostics is not None
    assert diagnostics.resolver_selected_kind == "floor_ray"
    assert diagnostics.resolver_contributor_ids == ("floor_ray",)
    assert diagnostics.resolver_candidate_diagnostics[0].selected is True
    # Tracker confidence remains an independent upper bound.
    assert world.confidence == pytest.approx(0.72)
    entity = publication.snapshot.entities[0]
    assert entity.position_quantity == "ground_footprint"
    assert entity.support_state == "floor"
    assert entity.posture == "standing"


def test_world_observation_drops_stale_resolver_diagnostics_only() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["source_id"] = 0
    track["world_frame_revision"] = "world-rev"
    track["world_resolver"] = {
        "contract": "noesis.world_resolver_diagnostics",
        "version": 1,
        "camera_id": "kitchen",
        "source_id": 0,
        "tracker_id": 7,
        "frame_id": 2,
        "observed_at_us": 1_900_000,
        "world_frame": "backend_world_m",
        "world_frame_revision": "world-rev",
        "selected_id": None,
        "selected_kind": None,
        "contributor_ids": [],
        "alternate_id": None,
        "fused": False,
        "confidence": 0.0,
        "candidates": [],
    }

    publication = service.publish(0, [track], metadata={})

    world = publication.observations[0].payload.world
    assert world is not None
    diagnostics = publication.observations[0].payload.world_diagnostics
    assert diagnostics is not None
    assert diagnostics.resolver_selected_kind is None
    assert diagnostics.resolver_candidate_diagnostics == ()


@pytest.mark.parametrize(
    "covariance",
    (
        [0.1] * 8,
        [0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
        [-0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
        [0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1],
    ),
)
def test_world_observation_rejects_malformed_resolver_covariance(
    covariance: list[float],
) -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["world_covariance"] = covariance

    publication = service.publish(0, [track], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities == ()


def test_world_observation_rejects_a_non_ground_quantity_in_world_field() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["world_quantity"] = "body_root_3d"

    publication = service.publish(0, [track], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities == ()


@pytest.mark.parametrize("missing_field", ("world_frame_revision", "world_transform_sha256"))
def test_canonical_world_observation_requires_registration_identity(
    missing_field: str,
) -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track.pop(missing_field)

    publication = service.publish(0, [track], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.observations[0].payload.world_diagnostics is not None
    assert publication.observations[0].payload.world_diagnostics.first_divergence_reason == (
        f"{missing_field}_missing"
    )
    assert publication.snapshot.entities == ()


@pytest.mark.parametrize(
    "continuity_source",
    ("cv_prediction", "anchor_hold", "image_motion_prediction"),
)
def test_display_continuity_never_enters_authoritative_world_fusion(
    continuity_source: str,
) -> None:
    service = _service()
    authoritative = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    service.publish(0, [authoritative], metadata={})
    continuation = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=2,
        observed_at_us=1_950_000,
        x=99.0,
    )
    continuation["world_source"] = continuity_source

    publication = service.publish(0, [continuation], metadata={})

    observation = publication.observations[0]
    assert observation.payload.world is None
    assert observation.coordinate_frame == "image_px"
    assert publication.snapshot.entities[0].position is not None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_visitor_generation_prevents_recycled_id_aliasing() -> None:
    service = _service()
    service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
                identity_kind="visitor",
                resident_uuid=None,
                stable_id=1001,
                visitor_generation=2,
            )
        ],
        metadata={},
    )
    publication = service.publish(
        1,
        [
            _track(
                camera_id="hall",
                tracker_id=8,
                frame_id=2,
                observed_at_us=1_950_000,
                x=1.1,
                identity_kind="visitor",
                resident_uuid=None,
                stable_id=1001,
                visitor_generation=3,
            )
        ],
        metadata={},
    )

    assert {entity.entity_id for entity in publication.snapshot.entities} == {
        "visitor:run-1:1001:g2",
        "visitor:run-1:1001:g3",
    }


def test_conflicting_positions_are_exposed_and_not_averaged() -> None:
    service = _service(conflict_distance_m=0.5)
    service.publish(
        0,
        [_track(camera_id="kitchen", tracker_id=7, frame_id=1, observed_at_us=1_900_000, x=1.0)],
        metadata={},
    )
    publication = service.publish(
        1,
        [_track(camera_id="hall", tracker_id=8, frame_id=2, observed_at_us=1_950_000, x=8.0)],
        metadata={},
    )

    entity = publication.snapshot.entities[0]
    assert entity.conflict is True
    assert entity.position is not None
    assert entity.position.x == pytest.approx(1.0) or entity.position.x == pytest.approx(8.0)
    assert sum(source.accepted for source in entity.sources) == 1


def test_image_only_observation_is_published_but_not_fused() -> None:
    service = _service()
    publication = service.publish(
        0,
        [_track(camera_id="kitchen", tracker_id=7, frame_id=1, observed_at_us=1_900_000, x=None)],
        metadata={},
    )

    assert len(publication.observations) == 1
    assert publication.observations[0].coordinate_frame == "image_px"
    assert publication.snapshot.entities == ()


def test_invalid_world_diagnostics_are_preserved_in_observation_and_journal(
    tmp_path: Path,
) -> None:
    journal = ContractJournal(tmp_path / "world.sqlite3")
    service = _service(journal=journal)
    track = _track(
        camera_id="living-room",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=None,
    )
    track.update(
        {
            "world_estimator_evaluated": True,
            "world_quality_reason": "fusion_policy_requires_registered_depth",
            "world_floor_candidate": [3.34, 0.0, 1.99],
            "world_floor_range_m": 3.88,
            "world_floor_range_limit_m": 22.0,
            "world_floor_incidence_sin": 0.31,
            "world_floor_admitted": True,
            "world_depth_candidate": None,
            "world_prefilter_measurement": None,
            "world_filter_prediction": [3.2, 0.0, 2.0],
            "world_measurement_accepted": False,
            "world_fusion_policy_id": "policy-sha",
            "world_floor_weight_scale": 1.0,
            "world_depth_weight_scale": 1.0,
            "world_floor_weight_effective": 1.0,
            "world_depth_weight_effective": 0.0,
            "depth_status": "ok",
            "depth_anchor_m": 3.9738,
            "depth_registered_m": None,
            "depth_used_m": None,
            "depth_registration_status": "out_of_domain_or_invalid",
            "depth_registration_id": "living-registration",
        }
    )

    publication = service.publish(0, [track], metadata={})
    observation = publication.observations[0]
    diagnostics = observation.payload.world_diagnostics
    assert observation.payload.world is None
    assert diagnostics is not None
    assert diagnostics.first_divergence_reason == (
        "fusion_policy_requires_registered_depth"
    )
    assert diagnostics.floor_candidate_m is not None
    assert diagnostics.floor_candidate_m.x == pytest.approx(3.34)
    assert diagnostics.depth_anchor_m == pytest.approx(3.9738)
    assert diagnostics.depth_registration_status == (
        "out_of_domain_or_invalid"
    )

    records = journal.records()
    persisted = records[0].payload
    assert persisted["observation_id"] == observation.observation_id
    persisted_diagnostics = persisted["payload"]["world_diagnostics"]
    assert persisted_diagnostics["first_divergence_reason"] == (
        "fusion_policy_requires_registered_depth"
    )
    assert persisted_diagnostics["floor_candidate_m"] == {
        "x": 3.34,
        "y": 0.0,
        "z": 1.99,
    }
    assert "identity_v2" not in json.dumps(persisted, sort_keys=True)


def test_valid_observation_id_and_room_evidence_reach_world_source() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track.update(
        {
            "zone": "Kitchen",
            "zone_source": "nvdsanalytics_roi",
            "zone_authoritative": True,
        }
    )

    publication = service.publish(0, [track], metadata={})

    observation = publication.observations[0]
    entity = publication.snapshot.entities[0]
    assert entity.room_id == "Kitchen"
    assert entity.sources[0].zone == "Kitchen"
    assert entity.sources[0].zone_source == "nvdsanalytics_roi"
    assert entity.sources[0].zone_authoritative is True
    assert entity.sources[0].observation_id == observation.observation_id


def test_camera_default_zone_is_diagnostic_but_never_canonical_room() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track.update(
        {
            "zone": "Kitchen",
            "zone_source": "camera_default",
            "zone_authoritative": False,
        }
    )

    entity = service.publish(0, [track], metadata={}).snapshot.entities[0]

    assert entity.room_id is None
    assert entity.sources[0].zone == "Kitchen"
    assert entity.sources[0].zone_source == "camera_default"
    assert entity.sources[0].zone_authoritative is False


@pytest.mark.parametrize("zone", [" Kitchen", "Kitchen ", "x" * 161, 7])
def test_world_service_rejects_nonexact_zone_before_observation(
    zone: object,
) -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track.update(
        {
            "zone": zone,
            "zone_source": "nvdsanalytics_roi",
            "zone_authoritative": True,
        }
    )

    with pytest.raises(ValueError, match="zone labels must be"):
        service.publish(0, [track], metadata={})


@pytest.mark.parametrize(
    "zone_source",
    [" nvdsanalytics_roi", "nvdsanalytics_roi ", "unknown", 7],
)
def test_world_service_rejects_nonexact_zone_provenance(
    zone_source: object,
) -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track.update(
        {
            "zone": "Kitchen",
            "zone_source": zone_source,
            "zone_authoritative": True,
        }
    )

    with pytest.raises(ValueError, match="zone_source must be an exact"):
        service.publish(0, [track], metadata={})


def test_unprovenanced_legacy_zone_fails_closed_for_room_derivation() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["zone"] = "Kitchen"

    publication = service.publish(0, [track], metadata={})
    observation = publication.observations[0]
    entity = publication.snapshot.entities[0]

    assert observation.payload.zone == "Kitchen"
    assert observation.payload.zone_source is None
    assert observation.payload.zone_authoritative is False
    assert entity.room_id is None


def test_conflicting_accepted_room_evidence_fails_closed() -> None:
    service = _service()
    kitchen = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    kitchen.update(
        {
            "zone": "Kitchen",
            "zone_source": "nvdsanalytics_roi",
            "zone_authoritative": True,
        }
    )
    living = _track(
        camera_id="living-room",
        tracker_id=19,
        frame_id=2,
        observed_at_us=1_950_000,
        x=1.1,
    )
    living.update(
        {
            "zone": "Living Room",
            "zone_source": "nvdsanalytics_roi",
            "zone_authoritative": True,
        }
    )
    service.publish(0, [kitchen], metadata={})

    entity = service.publish(1, [living], metadata={}).snapshot.entities[0]

    assert entity.room_id is None
    assert entity.conflict is True
    assert entity.conflict_reason == (
        "incompatible simultaneous accepted source room observations"
    )
    assert {source.zone for source in entity.sources} == {
        "Kitchen",
        "Living Room",
    }


def test_rejected_position_source_cannot_override_admitted_room() -> None:
    service = _service(conflict_distance_m=0.5)
    kitchen = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    kitchen.update(
        {
            "zone": "Kitchen",
            "zone_source": "nvdsanalytics_roi",
            "zone_authoritative": True,
        }
    )
    living = _track(
        camera_id="living-room",
        tracker_id=19,
        frame_id=2,
        observed_at_us=1_950_000,
        x=8.0,
    )
    living.update(
        {
            "zone": "Living Room",
            "zone_source": "nvdsanalytics_roi",
            "zone_authoritative": True,
        }
    )
    service.publish(0, [kitchen], metadata={})

    entity = service.publish(1, [living], metadata={}).snapshot.entities[0]
    admitted = next(source for source in entity.sources if source.accepted)

    assert entity.room_id == admitted.zone
    assert entity.conflict is True


def test_complete_embedding_provenance_is_copied_into_observation() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=None,
    )
    track.update(
        {
            "embedding_sequence": 42,
            "embedding_model_sha256": "d" * 64,
            "embedding_dimension": 256,
        }
    )

    observation = service.publish(0, [track], metadata={}).observations[0].payload

    assert observation.embedding_sequence == 42
    assert observation.embedding_model_sha256 == "d" * 64
    assert observation.embedding_dimension == 256


@pytest.mark.parametrize(
    "partial",
    [
        {"embedding_sequence": 42},
        {"embedding_model_sha256": "d" * 64},
        {"embedding_dimension": 256},
        {
            "embedding_sequence": 42,
            "embedding_model_sha256": "d" * 64,
        },
    ],
)
def test_partial_embedding_provenance_fails_publication(
    partial: dict[str, Any],
) -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=None,
    )
    track.update(partial)

    with pytest.raises(ValueError, match="partial embedding provenance"):
        service.publish(0, [track], metadata={})


@pytest.mark.parametrize(
    ("status", "depth_fields", "expected"),
    [
        (None, {}, False),
        ("unavailable", {"depth_used_m": 2.0}, False),
        ("depth_not_ready", {"depth_used_m": 2.0}, False),
        ("no_valid_depth", {"depth_anchor_m": 2.0}, False),
        ("transform_mismatch", {"depth_registered_m": 2.0}, False),
        ("native_stats_unavailable", {"depth_median_m": 2.0}, False),
        ("ok", {}, False),
        ("ok", {"depth_used_m": float("nan")}, False),
        ("ok", {"depth_used_m": float("inf")}, False),
        ("ok", {"depth_used_m": -1.0}, False),
        (
            "ok",
            {
                "depth_anchor_m": 2.0,
                "depth_registration_status": "out_of_domain_or_invalid",
            },
            False,
        ),
        (
            "ok",
            {"depth_anchor_m": 2.0, "depth_registration_status": "ok"},
            False,
        ),
        ("ok", {"depth_used_m": 2.0}, False),
        (
            "ok",
            {"depth_registered_m": 1.25, "depth_registration_status": "ok"},
            True,
        ),
        ("OK", {"depth_registered_m": 1.25}, False),
        (
            "ok",
            {
                "depth_registered_m": 1.25,
                "depth_used_m": 1.5,
                "depth_registration_status": "ok",
            },
            False,
        ),
    ],
)
def test_depth_present_requires_ok_status_and_usable_finite_depth(
    status: str | None,
    depth_fields: dict[str, Any],
    expected: bool,
) -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=None,
    )
    track.update({"depth_status": status, **depth_fields})

    observation = service.publish(0, [track], metadata={}).observations[0].payload

    assert observation.depth_present is expected


def test_sequences_are_monotonic_per_source() -> None:
    service = _service()
    publication = service.publish(
        0,
        [
            _track(camera_id="kitchen", tracker_id=7, frame_id=1, observed_at_us=1_900_000, x=1.0),
            _track(camera_id="kitchen", tracker_id=8, frame_id=1, observed_at_us=1_900_000, x=2.0),
        ],
        metadata={},
    )
    later = service.publish(
        0,
        [_track(camera_id="kitchen", tracker_id=7, frame_id=2, observed_at_us=1_950_000, x=1.1)],
        metadata={},
    )

    assert [item.sequence for item in publication.observations] == [0, 1]
    assert [item.sequence for item in later.observations] == [2]


def test_provisional_subject_is_stable_across_frames() -> None:
    service = _service()
    first = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
                identity_kind="provisional",
                resident_uuid=None,
            )
        ],
        metadata={},
    )
    second = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=2,
                observed_at_us=1_950_000,
                x=1.1,
                identity_kind="provisional",
                resident_uuid=None,
            )
        ],
        metadata={},
    )

    assert first.snapshot.entities[0].entity_id == second.snapshot.entities[0].entity_id
    assert len(second.snapshot.entities) == 1


def test_identity_resolution_atomically_replaces_provisional_entity() -> None:
    service = _service()
    first = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
                identity_kind="provisional",
                resident_uuid=None,
            )
        ],
        metadata={},
    )
    resolved = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=2,
                observed_at_us=1_950_000,
                x=1.1,
                identity_kind="resident",
                resident_uuid="resident-1",
            )
        ],
        metadata={},
    )

    assert [entity.entity_id for entity in first.snapshot.entities] == [
        "tracklet:run-1:kitchen:7"
    ]
    assert [entity.entity_id for entity in resolved.snapshot.entities] == [
        "resident:resident-1"
    ]
    assert [event.event_type for event in resolved.events] == ["appeared", "lost"]


def test_open_set_unknown_remains_explicit_world_identity_kind() -> None:
    service = _service()
    publication = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
                identity_kind="unknown",
                resident_uuid=None,
            )
        ],
        metadata={},
    )

    entity = publication.snapshot.entities[0]
    assert entity.entity_id == "unknown:run-1:kitchen:7"
    assert entity.subject.kind.value == "unknown"
    assert entity.subject.display_name is None


def test_world_service_emits_lifecycle_events() -> None:
    now = [2_000_000]
    producer = _producer()
    service = CanonicalWorldService(
        producer=producer,
        artifacts=_artifacts(),
        fusion=GlobalWorldFusion(
            producer,
            config=WorldFusionConfig(present_ttl_us=100, lost_ttl_us=500),
        ),
        clock_us=lambda: now[0],
    )
    appeared = service.publish(
        0,
        [_track(camera_id="kitchen", tracker_id=7, frame_id=1, observed_at_us=1_999_900, x=1.0)],
        metadata={},
    )
    assert [event.event_type for event in appeared.events] == ["appeared"]

    now[0] = 2_000_300
    held = service.publish(1, [], metadata={"camera_id": "hall"})
    assert [event.event_type for event in held.events] == ["held"]

    now[0] = 2_000_600
    lost = service.publish(1, [], metadata={"camera_id": "hall"})
    assert [event.event_type for event in lost.events] == ["lost"]
    assert [event.sequence for event in (*appeared.events, *held.events, *lost.events)] == [0, 1, 2]


def test_empty_source_frame_clears_world_evidence_and_emits_lost() -> None:
    service = _service()
    appeared = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            )
        ],
        metadata={"camera_id": "kitchen", "frame_id": 1},
    )
    cleared = service.publish(
        0,
        [],
        metadata={"camera_id": "kitchen", "frame_id": 2},
    )

    assert [event.event_type for event in appeared.events] == ["appeared"]
    assert cleared.observations == ()
    assert cleared.snapshot.entities == ()
    assert [event.event_type for event in cleared.events] == ["lost"]


def test_empty_source_frame_requires_explicit_camera_identity() -> None:
    service = _service()
    with pytest.raises(
        ValueError,
        match="empty source publication requires metadata.camera_id",
    ):
        service.publish(0, [], metadata={})


def test_empty_source_frame_removes_only_that_camera_and_clears_conflict() -> None:
    service = _service(conflict_distance_m=0.5)
    service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            )
        ],
        metadata={"camera_id": "kitchen"},
    )
    conflicted = service.publish(
        1,
        [
            _track(
                camera_id="hall",
                tracker_id=8,
                frame_id=1,
                observed_at_us=1_950_000,
                x=8.0,
            )
        ],
        metadata={"camera_id": "hall"},
    )
    cleared = service.publish(
        1,
        [],
        metadata={"camera_id": "hall", "frame_id": 2},
    )

    assert conflicted.snapshot.entities[0].conflict is True
    assert [event.event_type for event in cleared.events] == ["conflict_cleared"]
    entity = cleared.snapshot.entities[0]
    assert entity.conflict is False
    assert [source.camera_id for source in entity.sources] == ["kitchen"]


class _WebSocketRecorder:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.next_submission_id = 1

    def admit_broadcast_batch_sync(
        self,
        messages: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> Any:
        submission_id = self.next_submission_id
        self.next_submission_id += 1
        receipt = SimpleNamespace(
            submission_id=submission_id,
            message_count=len(messages),
        )

        class _Admission:
            def __init__(self, owner: _WebSocketRecorder) -> None:
                self.receipt = receipt
                self._owner = owner

            def commit_then_release(self, commit: Any) -> Any:
                result = commit()
                self._owner.messages.extend(messages)
                return result

        return _Admission(self)


def test_tracking_publisher_emits_observations_and_separate_world_snapshot() -> None:
    websocket = _WebSocketRecorder()
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {"camera_id": "kitchen"},
        world_service=_service(),
    )

    receipt = publisher.publish(
        0,
        [_track(camera_id="kitchen", tracker_id=7, frame_id=1, observed_at_us=1_900_000, x=1.0)],
    )

    assert [message["type"] for message in websocket.messages] == [
        "tracking",
        "world_snapshot",
        "world_event",
    ]
    tracking = websocket.messages[0]
    assert tracking["observation_contract"] == "noesis.observation.person"
    assert tracking["tracking_continuity_contract"] == (
        "noesis.tracking.publication-continuity"
    )
    assert tracking["tracking_continuity_contract_version"] == 1
    assert tracking["tracking_publication_sequence"] == 0
    assert receipt.tracking_publication_sequence == 0
    assert receipt.outbound_submission_id == 1
    assert receipt.outbound_message_count == 3
    assert all(message["cohort"] == tracking["cohort"] for message in websocket.messages)
    assert tracking["observations"][0]["contract_version"] == 1
    assert tracking["world_snapshot"]["contract"] == "noesis.world.snapshot"
    assert tracking["world_events"][0]["event_type"] == "appeared"


def test_tracking_publisher_emits_advancing_empty_frame_and_clears_world() -> None:
    websocket = _WebSocketRecorder()
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {
            "camera_id": "kitchen",
            "image_size": [1920, 1080],
        },
        world_service=_service(),
    )
    publisher.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            )
        ],
        frame_metadata={
            "frame_id": 1,
            "observed_at_us": 1_900_000,
            "capture_time_status": "estimated",
            "media_pts_ns": 1_000,
        },
    )
    publisher.publish(
        0,
        [],
        frame_metadata={
            "frame_id": 2,
            "observed_at_us": 1_950_000,
            "capture_time_status": "estimated",
            "media_pts_ns": 2_000,
        },
    )

    tracking = [
        message for message in websocket.messages if message["type"] == "tracking"
    ]
    assert [message["frame_id"] for message in tracking] == [1, 2]
    assert [message["tracking_publication_sequence"] for message in tracking] == [
        0,
        1,
    ]
    assert tracking[1]["track_count"] == 0
    assert tracking[1]["tracks"] == []
    assert tracking[1]["world_snapshot"]["entities"] == []
    assert [event["event_type"] for event in tracking[1]["world_events"]] == [
        "lost"
    ]


def test_tracking_publication_sequence_commits_only_after_successful_broadcast() -> None:
    class FailFirstRecorder:
        def __init__(self) -> None:
            self.fail = True
            self.messages: list[dict[str, Any]] = []
            self.next_submission_id = 1

        def admit_broadcast_batch_sync(
            self,
            messages: list[dict[str, Any]],
            **_kwargs: Any,
        ) -> Any:
            if self.fail:
                self.fail = False
                raise RuntimeError("tracking broadcast failed")
            submission_id = self.next_submission_id
            self.next_submission_id += 1
            receipt = SimpleNamespace(
                submission_id=submission_id,
                message_count=len(messages),
            )

            class _Admission:
                def __init__(self, owner: FailFirstRecorder) -> None:
                    self.receipt = receipt
                    self._owner = owner

                def commit_then_release(self, commit: Any) -> Any:
                    result = commit()
                    self._owner.messages.extend(messages)
                    return result

            return _Admission(self)

    websocket = FailFirstRecorder()
    publisher = TrackingTelemetryPublisher(websocket)

    with pytest.raises(RuntimeError, match="tracking broadcast failed"):
        publisher.publish(0, [])
    publisher.publish(0, [])

    assert len(websocket.messages) == 1
    assert websocket.messages[0]["type"] == "tracking"
    assert websocket.messages[0]["tracking_publication_sequence"] == 0


def test_batch_admission_failure_rolls_back_world_and_reuses_sequence_zero(
    tmp_path: Path,
) -> None:
    class FailFirstBatchRecorder:
        def __init__(self) -> None:
            self.fail = True
            self.messages: list[dict[str, Any]] = []
            self.next_submission_id = 1

        def admit_broadcast_batch_sync(
            self,
            messages: list[dict[str, Any]],
            **_kwargs: Any,
        ) -> Any:
            if self.fail:
                self.fail = False
                raise RuntimeError("canonical batch admission failed")
            submission_id = self.next_submission_id
            self.next_submission_id += 1
            receipt = SimpleNamespace(
                submission_id=submission_id,
                message_count=len(messages),
            )

            class _Admission:
                def __init__(self, owner: FailFirstBatchRecorder) -> None:
                    self.receipt = receipt
                    self._owner = owner

                def commit_then_release(self, commit: Any) -> Any:
                    result = commit()
                    self._owner.messages.extend(messages)
                    return result

            return _Admission(self)

    failures: list[BaseException] = []
    websocket = FailFirstBatchRecorder()
    journal = ContractJournal(tmp_path / "world.sqlite3")
    service = _service(journal=journal)
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {"camera_id": "kitchen"},
        world_service=service,
        failure_callback=failures.append,
    )

    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    with pytest.raises(RuntimeError, match="canonical batch admission failed"):
        publisher.publish(0, [track])

    assert websocket.messages == []
    assert journal.records() == ()
    assert service.current_snapshot() is None

    receipt = publisher.publish(0, [track])

    tracking = [
        message for message in websocket.messages if message["type"] == "tracking"
    ]
    assert [message["tracking_publication_sequence"] for message in tracking] == [0]
    assert tracking[0]["world_snapshot"]["entities"][0]["entity_id"] == (
        "resident:resident-1"
    )
    assert tracking[0]["world_events"][0]["sequence"] == 0
    assert receipt.tracking_publication_sequence == 0
    assert len(journal.records()) == 3
    assert len(failures) == 1
    assert str(failures[0]) == "canonical batch admission failed"


def test_nonfinite_batch_rejection_precedes_tracking_and_world_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    class _Future:
        def add_done_callback(self, _callback: Any) -> None:
            return None

        def done(self) -> bool:
            return False

        def cancelled(self) -> bool:
            return False

        def exception(self) -> None:
            return None

    def _capture(coroutine: object, _loop: object) -> _Future:
        scheduled.append(coroutine)
        return _Future()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    websocket = WebSocketServer(stats_callback=None)
    websocket.event_loop = type(
        "OpenLoop",
        (),
        {"is_closed": lambda self: False},
    )()
    service = _service()
    metadata = {"camera_id": "kitchen", "diagnostic": float("nan")}
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: metadata,
        world_service=service,
    )
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        publisher.publish(0, [track])

    assert scheduled == []
    assert service.current_snapshot() is None
    metadata["diagnostic"] = 0.0
    receipt = publisher.publish(0, [track])
    assert receipt.tracking_publication_sequence == 0
    assert len(scheduled) == 1
    scheduled.pop().close()


def test_health_progress_uses_publication_clock_when_snapshot_window_regresses() -> None:
    websocket = _WebSocketRecorder()
    monitor = CapabilityMonitor(
        instance_id="appliance",
        run_id="run-1",
        policies={
            capability: CapabilityPolicy(
                stale_after_us=1_000_000,
                fail_after_us=2_000_000,
            )
            for capability in ("tracking_observations", "global_world")
        },
    )
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda source_id, _tracks: {
            "camera_id": "kitchen" if source_id == 0 else "hall",
        },
        world_service=_service(),
        health_monitor=monitor,
    )

    publisher.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_990_000,
                x=1.0,
            )
        ],
    )
    publisher.publish(
        1,
        [
            _track(
                camera_id="hall",
                tracker_id=8,
                frame_id=1,
                observed_at_us=1_950_000,
                x=1.1,
                resident_uuid="resident-2",
                stable_id=2,
            )
        ],
    )
    publisher.publish(0, [])

    snapshots = [
        message["payload"]
        for message in websocket.messages
        if message["type"] == "world_snapshot"
    ]
    assert snapshots[-2]["observed_end_us"] == 1_990_000
    assert snapshots[-1]["observed_end_us"] == 1_950_000
    health = monitor.snapshot(generated_at_us=2_000_001)
    assert {row.status.value for row in health.capabilities} == {"healthy"}
    assert {
        row.evidence["world_observed_end_us"] for row in health.capabilities
    } == {1_950_000}
    assert {row.evidence["observed_at_us"] for row in health.capabilities} == {
        2_000_000
    }


def test_tracking_publisher_does_not_hide_world_service_failure() -> None:
    class BrokenWorldService:
        def prepare(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("world failed")

        def commit(self, _prepared: Any) -> None:
            raise AssertionError("commit must not run after prepare failure")

        def discard(self, _prepared: Any) -> None:
            raise AssertionError("nothing was prepared")

    failures: list[BaseException] = []
    publisher = TrackingTelemetryPublisher(
        _WebSocketRecorder(),
        world_service=BrokenWorldService(),
        failure_callback=failures.append,
    )
    with pytest.raises(RuntimeError, match="world failed"):
        publisher.publish(0, [])
    assert len(failures) == 1
    assert str(failures[0]) == "world failed"


def test_tracking_metadata_getter_failure_is_fatal() -> None:
    failures: list[BaseException] = []

    def _broken_metadata(_source_id: int, _tracks: list[dict[str, Any]]) -> None:
        raise RuntimeError("metadata unavailable")

    websocket = _WebSocketRecorder()
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=_broken_metadata,
        failure_callback=failures.append,
    )

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        publisher.publish(0, [])

    assert websocket.messages == []
    assert len(failures) == 1
    assert str(failures[0]) == "metadata unavailable"


def test_world_commit_failure_after_admission_is_loud_and_consumes_receipt() -> None:
    delegate = _service()

    class FailFirstCommitService:
        def __init__(self) -> None:
            self.fail = True

        def prepare(self, *args: Any, **kwargs: Any) -> Any:
            return delegate.prepare(*args, **kwargs)

        def commit(self, prepared: Any) -> Any:
            if self.fail:
                self.fail = False
                raise RuntimeError("world commit failed")
            return delegate.commit(prepared)

        def discard(self, prepared: Any) -> None:
            delegate.discard(prepared)

    failures: list[BaseException] = []
    websocket = _WebSocketRecorder()
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {"camera_id": "kitchen"},
        world_service=FailFirstCommitService(),
        failure_callback=failures.append,
    )
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )

    with pytest.raises(RuntimeError, match="world commit failed"):
        publisher.publish(0, [track])

    assert websocket.messages == []
    assert delegate.current_snapshot() is None
    with pytest.raises(TrackingPublicationPoisoned) as exc_info:
        publisher.publish(0, [track])
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "world commit failed"
    assert websocket.messages == []
    assert delegate.current_snapshot() is None
    assert len(failures) == 1
    assert str(failures[0]) == "world commit failed"


def test_tracking_delivery_waits_for_world_authority_commit() -> None:
    async def _run() -> tuple[list[str], Any, CanonicalWorldService]:
        delegate = _service()
        commit_entered = threading.Event()
        release_commit = threading.Event()

        class BlockingCommitService:
            def prepare(self, *args: Any, **kwargs: Any) -> Any:
                return delegate.prepare(*args, **kwargs)

            def commit(self, prepared: Any) -> Any:
                commit_entered.set()
                assert release_commit.wait(timeout=5.0)
                return delegate.commit(prepared)

            def discard(self, prepared: Any) -> None:
                delegate.discard(prepared)

        class Socket:
            remote_address = ("127.0.0.1", 6008)

            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send(self, payload: str) -> None:
                self.sent.append(payload)

        websocket = WebSocketServer(stats_callback=None)
        websocket.event_loop = asyncio.get_running_loop()
        socket = Socket()
        websocket.connected_clients.add(socket)
        publisher = TrackingTelemetryPublisher(
            websocket,
            metadata_getter=lambda _source_id, _tracks: {
                "camera_id": "kitchen"
            },
            world_service=BlockingCommitService(),
        )
        track = _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
        publish_task = asyncio.create_task(
            asyncio.to_thread(publisher.publish, 0, [track])
        )
        assert await asyncio.to_thread(commit_entered.wait, 2.0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert socket.sent == []
        assert delegate.current_snapshot() is None

        release_commit.set()
        receipt = await publish_task
        await websocket.quiesce_outbound_submissions(timeout_s=1.0)
        return socket.sent, receipt, delegate

    encoded, receipt, service = asyncio.run(_run())
    assert [json.loads(item)["type"] for item in encoded] == [
        "tracking",
        "world_snapshot",
        "world_event",
    ]
    assert receipt.tracking_publication_sequence == 0
    snapshot = service.current_snapshot()
    assert snapshot is not None
    assert len(snapshot.entities) == 1


def test_prepared_world_candidate_is_owner_bound_and_identity_bound() -> None:
    owner = _service()
    other = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    prepared = owner.prepare(0, [track], metadata={"camera_id": "kitchen"})

    with pytest.raises(RuntimeError, match="belongs to another service"):
        other.commit(prepared)
    with pytest.raises(RuntimeError, match="identity mismatch|unknown or consumed"):
        owner.commit(replace(prepared))

    assert owner.current_snapshot() is None
    owner.discard(prepared)


def test_prepared_world_candidates_have_a_hard_capacity() -> None:
    service = _service()
    prepared = [
        service.prepare(
            0,
            [
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            ],
            metadata={"camera_id": "kitchen"},
        )
        for _ in range(service.MAX_PREPARED_CANDIDATES)
    ]

    with pytest.raises(RuntimeError, match="preparation capacity"):
        service.prepare(
            0,
            [],
            metadata={"camera_id": "kitchen"},
        )

    for candidate in prepared:
        service.discard(candidate)


def test_current_snapshot_is_immutable_read_authority_and_consumes_no_sequence() -> None:
    service = _service()
    first = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            )
        ],
        metadata={"camera_id": "kitchen"},
    )

    assert service.current_snapshot() is first.snapshot
    assert service.current_snapshot() is first.snapshot
    assert not hasattr(service, "fusion")

    second = service.publish(
        1,
        [],
        metadata={"camera_id": "hall"},
    )
    assert second.snapshot.sequence == first.snapshot.sequence + 1
    assert service.current_snapshot() is second.snapshot


def test_injected_fusion_is_detached_from_later_caller_mutation() -> None:
    injected = GlobalWorldFusion(_producer())
    service = CanonicalWorldService(
        producer=_producer(),
        artifacts=_artifacts(),
        fusion=injected,
        clock_us=lambda: 2_000_000,
    )

    # This mutates only the caller-owned instance's diagnostic sequence.
    assert injected.snapshot(published_at_us=2_000_000).sequence == 1
    publication = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            )
        ],
        metadata={"camera_id": "kitchen"},
    )
    assert publication.snapshot.sequence == 1


def test_world_authority_accepts_async_optional_journal(tmp_path: Path) -> None:
    path = tmp_path / "world.sqlite3"
    asynchronous = AsyncContractJournal(
        ContractJournal(path)
    )
    service = _service(journal=asynchronous)

    publication = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            )
        ],
        metadata={"camera_id": "kitchen"},
    )

    assert publication.snapshot.sequence == 1
    assert service.persistence_health()["status"] == "healthy"
    service.close()
    reopened = ContractJournal(path)
    try:
        assert len(reopened.records()) == len(publication.observations) + 2
    finally:
        reopened.close()


def test_async_world_persistence_never_waits_in_live_publication() -> None:
    persistence_started = threading.Event()
    release_persistence = threading.Event()

    class SlowJournal:
        max_records = 10_000

        def append_entries(self, entries: Any) -> None:
            tuple(entries)
            persistence_started.set()
            assert release_persistence.wait(timeout=5.0)

        def close(self) -> None:
            return None

    asynchronous = AsyncContractJournal(SlowJournal())  # type: ignore[arg-type]
    service = _service(journal=asynchronous)

    started = time.perf_counter()
    publication = service.publish(
        0,
        [
            _track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            )
        ],
        metadata={"camera_id": "kitchen"},
    )
    elapsed_s = time.perf_counter() - started

    assert publication.snapshot.sequence == 1
    assert persistence_started.wait(timeout=1.0)
    assert not release_persistence.is_set()
    assert elapsed_s < 0.1
    release_persistence.set()
    service.close()


def test_world_commit_rejects_false_journal_append_acknowledgement() -> None:
    class FalseAckJournal:
        def append_many(
            self,
            _payloads: Any,
            *,
            recorded_at_us: int,
        ) -> tuple[Any, ...]:
            assert recorded_at_us > 0
            return ()

    service = _service(journal=FalseAckJournal())  # type: ignore[arg-type]
    with pytest.raises(
        RuntimeError,
        match="journal did not confirm the exact append count",
    ):
        service.publish(
            0,
            [
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            ],
            metadata={"camera_id": "kitchen"},
        )
    assert service.current_snapshot() is None


def test_journal_ack_failure_aborts_admitted_tracking_without_delivery() -> None:
    class FalseAckJournal:
        def append_many(
            self,
            _payloads: Any,
            *,
            recorded_at_us: int,
        ) -> tuple[Any, ...]:
            assert recorded_at_us > 0
            return ()

    websocket = _WebSocketRecorder()
    service = _service(journal=FalseAckJournal())  # type: ignore[arg-type]
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {
            "camera_id": "kitchen"
        },
        world_service=service,
    )
    with pytest.raises(
        RuntimeError,
        match="journal did not confirm the exact append count",
    ):
        publisher.publish(
            0,
            [
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            ],
        )
    assert websocket.messages == []
    assert service.current_snapshot() is None
    with pytest.raises(TrackingPublicationPoisoned):
        publisher.publish(0, [])


def test_journal_retention_must_hold_complete_cohort_before_delivery(
    tmp_path: Path,
) -> None:
    websocket = _WebSocketRecorder()
    journal = ContractJournal(
        tmp_path / "world.sqlite3",
        max_records=1,
    )
    service = _service(journal=journal)
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {
            "camera_id": "kitchen"
        },
        world_service=service,
    )
    with pytest.raises(
        RuntimeError,
        match="retention cannot hold the complete cohort",
    ):
        publisher.publish(
            0,
            [
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            ],
        )
    assert websocket.messages == []
    assert journal.records() == ()
    assert service.current_snapshot() is None


def test_synchronous_world_journal_meets_three_source_aggregate_cadence(
    tmp_path: Path,
) -> None:
    journal = ContractJournal(
        tmp_path / "world.sqlite3",
        max_records=10_000,
    )
    service = _service(journal=journal)
    durations_s: list[float] = []
    try:
        # Three warm-up frames plus 31 measured aggregates keep the median
        # representative when the host schedules an unrelated I/O outlier.
        for frame_id in range(1, 35):
            started = time.perf_counter()
            for source_id, camera_id in enumerate(("kitchen", "hall", "living")):
                tracks = [
                    _track(
                        camera_id=camera_id,
                        tracker_id=entity_index,
                        frame_id=frame_id,
                        observed_at_us=(
                            1_900_000 + frame_id * 1_000 + source_id
                        ),
                        x=entity_index / 10.0 + source_id * 0.01,
                        resident_uuid=f"resident-{entity_index}",
                        stable_id=entity_index + 1,
                    )
                    # Six simultaneous people on every camera is already above the
                    # expected occupied household load. The independent 64-entity
                    # test remains the capacity stress contract.
                    for entity_index in range(6)
                ]
                service.publish(
                    source_id,
                    tracks,
                    metadata={"camera_id": camera_id},
                )
            if frame_id > 3:
                durations_s.append(time.perf_counter() - started)
    finally:
        service.close()

    # Pair-safe publication is approximately 12 Hz/source. Three serialized
    # sources therefore have a 27.8 ms aggregate service period. WAL with FULL
    # synchronization keeps every release-gating commit power-loss durable.
    assert len(durations_s) == 31
    assert statistics.median(durations_s) < 0.027
    assert statistics.fmean(durations_s) < 0.027


def test_transactional_prepare_stays_within_full_world_cadence_budget() -> None:
    service = _service()
    cameras = ("kitchen", "hall", "living", "entry")
    for source_id, camera_id in enumerate(cameras):
        service.publish(
            source_id,
            [
                _track(
                    camera_id=camera_id,
                    tracker_id=entity_index,
                    frame_id=source_id + 1,
                    observed_at_us=1_900_000 + source_id * 10_000,
                    x=entity_index / 10.0,
                    resident_uuid=f"resident-{entity_index}",
                    stable_id=entity_index + 1,
                )
                for entity_index in range(64)
            ],
            metadata={"camera_id": camera_id},
        )
    next_frame = [
        _track(
            camera_id="kitchen",
            tracker_id=entity_index,
            frame_id=10,
            observed_at_us=1_990_000,
            x=entity_index / 10.0 + 0.01,
            resident_uuid=f"resident-{entity_index}",
            stable_id=entity_index + 1,
        )
        for entity_index in range(64)
    ]

    warmup = service.prepare(
        0,
        next_frame,
        metadata={"camera_id": "kitchen"},
    )
    service.discard(warmup)
    durations_s: list[float] = []
    for _ in range(7):
        started = time.perf_counter()
        prepared = service.prepare(
            0,
            next_frame,
            metadata={"camera_id": "kitchen"},
        )
        durations_s.append(time.perf_counter() - started)
        service.discard(prepared)

    tracemalloc.start()
    measured = service.prepare(
        0,
        next_frame,
        metadata={"camera_id": "kitchen"},
    )
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    service.discard(measured)
    tracemalloc.stop()

    # Max-capacity (64 entities), four-camera authority state must remain
    # comfortably inside a 15 Hz publication period on the reference host.
    assert statistics.median(durations_s) < 0.030
    assert peak_bytes < 8 * 1024 * 1024


def test_world_service_close_waits_for_inflight_publication(tmp_path) -> None:
    publication_started = threading.Event()
    release_publication = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    publish_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def _blocked_artifacts(_source_id: int, _metadata: Any) -> WorldArtifacts:
        publication_started.set()
        assert release_publication.wait(timeout=5.0)
        return _artifacts()

    journal = ContractJournal(tmp_path / "world.sqlite3")
    service = CanonicalWorldService(
        producer=_producer(),
        artifacts=_blocked_artifacts,
        clock_us=lambda: 2_000_000,
        journal=journal,
    )

    def _publish() -> None:
        try:
            service.publish(
                0,
                [
                    _track(
                        camera_id="kitchen",
                        tracker_id=7,
                        frame_id=1,
                        observed_at_us=1_900_000,
                        x=1.0,
                    )
                ],
                metadata={"camera_id": "kitchen"},
            )
        except BaseException as exc:
            publish_errors.append(exc)

    def _close() -> None:
        close_started.set()
        try:
            service.close()
        except BaseException as exc:
            close_errors.append(exc)
        finally:
            close_finished.set()

    publish_thread = threading.Thread(target=_publish, daemon=True)
    close_thread = threading.Thread(target=_close, daemon=True)
    publish_thread.start()
    assert publication_started.wait(timeout=2.0)
    close_thread.start()
    assert close_started.wait(timeout=2.0)
    close_was_blocked = not close_finished.wait(timeout=0.2)
    release_publication.set()
    publish_thread.join(timeout=5.0)
    close_thread.join(timeout=5.0)

    assert close_was_blocked
    assert not publish_thread.is_alive()
    assert not close_thread.is_alive()
    assert publish_errors == []
    assert close_errors == []
    assert service.closed
    persisted = ContractJournal(journal.path)
    try:
        assert len(persisted.records()) == 3
    finally:
        persisted.close()
    with pytest.raises(RuntimeError, match="canonical world service is closed"):
        service.publish(0, [], metadata={})
    service.close()
