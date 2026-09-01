from __future__ import annotations

import asyncio
import json
import math
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
from noesis_core.contracts.world import EntityLifecycle
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


def _image_motion_filter_transition(
    *,
    process_observation: list[float],
    position_base: list[float],
    position_gain: float,
    origin_media_pts_ns: int = 1_000_000_000,
    current_media_pts_ns: int = 1_100_000_000,
    trail_segment_id: int = 0,
) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "innovation_update",
        "origin_kind": "queue_admitted_world_output",
        "origin_world": [1.0, 0.0, 2.0],
        "origin_media_pts_ns": origin_media_pts_ns,
        "current_media_pts_ns": current_media_pts_ns,
        "origin_trail_segment_id": trail_segment_id,
        "gate_dt_s": 0.1,
        "position_base": position_base,
        "position_gain": position_gain,
        "max_speed_mps": 4.0,
        "max_jump_m": 0.75,
        "reset_after_s": 1.25,
    }


def _set_image_motion_origin(
    track: dict[str, Any],
    *,
    trail_segment_id: int = 0,
) -> dict[str, Any]:
    track["media_pts_ns"] = 1_000_000_000
    track["tracker_lifecycle_generation"] = 1
    track["trail_segment_id"] = trail_segment_id
    track["world_measurement_accepted"] = True
    return track


def _bounded_process_track(
    *,
    continuity_source: str = "cv_prediction",
    provenance_type: str = "bounded_cv_process",
    provenance_origin: str = "last_metric_process",
    trail_segment_id: int = 0,
) -> dict[str, Any]:
    output_hold = provenance_type == "bounded_output_hold"
    world_x = 1.0 if output_hold else 1.2
    process_x = 1.4 if output_hold else world_x
    continuation = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=2,
        observed_at_us=1_950_000,
        x=world_x,
    )
    continuation.update(
        {
            "media_pts_ns": 1_100_000_000,
            "tracker_lifecycle_generation": 1,
            "trail_segment_id": trail_segment_id,
            "world_source": continuity_source,
            "world_quality": "held",
            "world_quality_reason": "physical_measurement_rejected",
            "world_measurement_accepted": False,
            "world_filter_prediction": [world_x, 0.0, 2.0],
            "world_prediction_provenance": {
                "type": provenance_type,
                "non_authoritative": True,
                "state_integrated": True,
                "origin": provenance_origin,
                "reason": "physical_measurement_rejected",
                "process_observation": [process_x, 0.0, 2.0],
                "filter_transition": {
                    "version": 1,
                    "kind": "output_hold" if output_hold else "bounded_process_step",
                    "origin_kind": "queue_admitted_world_output",
                    "origin_world": [1.0, 0.0, 2.0],
                    "origin_media_pts_ns": 1_000_000_000,
                    "current_media_pts_ns": 1_100_000_000,
                    "origin_trail_segment_id": trail_segment_id,
                    "metric_origin_kind": "queue_admitted_metric_world_output",
                    "metric_origin_world": [1.0, 0.0, 2.0],
                    "metric_origin_media_pts_ns": 1_000_000_000,
                    "metric_origin_trail_segment_id": trail_segment_id,
                    "tracker_lifecycle_generation": 1,
                    "world_frame": "backend_world_m",
                    "world_frame_revision": "world-rev",
                    "world_transform_sha256": "d" * 64,
                    "gate_dt_s": 0.1,
                    "position_base": [1.0, 0.0, 2.0],
                    "position_gain": 0.0 if output_hold else 1.0,
                    "max_speed_mps": 4.0,
                    "reset_after_s": 1.25,
                },
            },
        }
    )
    return continuation


def _stationary_output_hold(
    *,
    metric_age_ns: int,
    include_evidence: bool = True,
    stationary_supported: bool = True,
    posture: str = "sitting",
    reason: str = "physical_measurement_rejected",
) -> dict[str, Any]:
    hold = _bounded_process_track(
        continuity_source="anchor_hold",
        provenance_type="bounded_output_hold",
        provenance_origin="last_published_output",
    )
    current_media_pts_ns = 1_000_000_000 + int(metric_age_ns)
    hold["media_pts_ns"] = current_media_pts_ns
    hold["posture"] = posture
    hold["world_posture"] = posture
    hold["world_bbox_stationary_supported"] = bool(stationary_supported)
    hold["world_quality_reason"] = reason
    provenance = hold["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["reason"] = reason
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition["current_media_pts_ns"] = current_media_pts_ns
    transition["gate_dt_s"] = min(
        metric_age_ns / 1_000_000_000.0,
        1.25,
    )
    if include_evidence:
        provenance["stationary_evidence"] = {
            "version": 1,
            "kind": "bbox_stationary",
            "frame_id": hold["frame_id"],
            "media_pts_ns": current_media_pts_ns,
            "posture": posture,
        }
    return hold


def _upright_presence_output_hold(
    *,
    origin_media_pts_ns: int,
    current_media_pts_ns: int,
    floor_candidate_x: float = 1.2,
    posture: str = "standing",
    motion_mode: str = "walk",
    bbox_stationary_supported: bool = True,
) -> dict[str, Any]:
    hold = _retimed_bounded_output_hold(
        origin_x=1.0,
        origin_media_pts_ns=origin_media_pts_ns,
        current_media_pts_ns=current_media_pts_ns,
    )
    hold.update(
        {
            "frame_id": int(current_media_pts_ns // 100_000_000),
            "observed_at_us": int(current_media_pts_ns // 1_000),
            "pose_present": True,
            "posture": posture,
            "world_posture": posture,
            "motion_mode": motion_mode,
            "lower_body_occluded": False,
            "world_bbox_stationary_supported": bbox_stationary_supported,
            "world_contact_basis": "pose:torso_motion",
            "world_floor_admitted": True,
            "world_floor_contact_plausible": True,
            "world_observation_range_admitted": True,
            "world_support_state": "floor",
            "world_floor_candidate": [floor_candidate_x, 0.0, 2.0],
            "bbox": [736.0, 382.0, 75.0, 154.0],
            "image_size": [1920, 1080],
            "confidence": 0.90,
            "tracker_confidence": 0.70,
        }
    )
    distance_m = abs(float(floor_candidate_x) - 1.0)
    provenance = hold["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["upright_presence_evidence"] = {
        "version": 1,
        "kind": "pose_confirmed_nonseated_floor_near_output",
        "frame_id": hold["frame_id"],
        "media_pts_ns": current_media_pts_ns,
        "posture": posture,
        "motion_mode": motion_mode,
        "contact_basis": "pose:torso_motion",
        "detector_confidence": 0.90,
        "tracker_confidence": 0.70,
        "floor_candidate": [floor_candidate_x, 0.0, 2.0],
        "output_distance_m": distance_m,
        "output_distance_limit_m": 1.25,
    }
    return hold


def _current_presence_output_hold(
    *,
    visible_origin_media_pts_ns: int,
    kinematic_origin_media_pts_ns: int,
    current_media_pts_ns: int,
) -> dict[str, Any]:
    hold = _retimed_bounded_output_hold(
        origin_x=1.0,
        origin_media_pts_ns=visible_origin_media_pts_ns,
        current_media_pts_ns=current_media_pts_ns,
    )
    root_bbox = [10.0, 20.0, 30.0, 40.0]
    current_bbox = [11.0, 20.0, 31.0, 40.0]
    root_center = [25.0, 40.0]
    current_center = [26.5, 40.0]
    bbox_scale = 0.5 * (
        math.hypot(root_bbox[2], root_bbox[3])
        + math.hypot(current_bbox[2], current_bbox[3])
    )
    size_ratio = max(
        root_bbox[2] / current_bbox[2],
        current_bbox[2] / root_bbox[2],
        root_bbox[3] / current_bbox[3],
        current_bbox[3] / root_bbox[3],
    )
    displacement_norm = math.hypot(
        current_center[0] - root_center[0],
        current_center[1] - root_center[1],
    ) / bbox_scale
    hold.update(
        {
            "class_id": 0,
            "frame_id": int(current_media_pts_ns // 100_000_000),
            "observed_at_us": int(current_media_pts_ns // 1_000),
            "bbox": current_bbox,
            "pose_present": True,
            "confidence": 0.39,
            "tracker_confidence": 0.71,
        }
    )
    provenance = hold["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["current_presence_evidence"] = {
        "version": 1,
        "kind": "same_lifecycle_bbox_from_kinematic_output",
        "frame_id": hold["frame_id"],
        "media_pts_ns": current_media_pts_ns,
        "kinematic_origin_media_pts_ns": kinematic_origin_media_pts_ns,
        "kinematic_origin_trail_segment_id": 0,
        "kinematic_origin_bbox": root_bbox,
        "current_bbox": current_bbox,
        "bbox_size_ratio": size_ratio,
        "bbox_center_displacement_norm": displacement_norm,
        "detector_confidence": 0.39,
        "tracker_confidence": 0.71,
        "pose_present": True,
    }
    return hold


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


def test_service_trail_break_resets_fusion_once_then_same_segment_teleport_is_gated() -> None:
    service = _service()
    origin = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_800_000,
        x=0.0,
    )
    origin.update(
        {
            "tracker_lifecycle_generation": 1,
            "trail_segment_id": 9,
            "trail_break_required": False,
            "trail_append_allowed": True,
        }
    )
    service.publish(0, [origin], metadata={})

    reset_point = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=2,
        observed_at_us=1_900_000,
        x=10.0,
    )
    reset_point.update(
        {
            "tracker_lifecycle_generation": 1,
            "trail_segment_id": 10,
            "trail_break_required": True,
            "trail_append_allowed": False,
        }
    )
    reset_publication = service.publish(0, [reset_point], metadata={})

    reset_world = reset_publication.observations[0].payload.world
    assert reset_world is not None
    assert reset_world.position.x == pytest.approx(10.0)
    reset_entity = reset_publication.snapshot.entities[0]
    assert reset_entity.position is not None
    assert reset_entity.position.x == pytest.approx(10.0)
    assert reset_entity.velocity_mps is None
    assert reset_entity.conflict is False

    teleport = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=3,
        observed_at_us=1_950_000,
        x=20.0,
    )
    teleport.update(
        {
            "tracker_lifecycle_generation": 1,
            "trail_segment_id": 10,
            "trail_break_required": False,
            "trail_append_allowed": True,
        }
    )
    rejected_publication = service.publish(0, [teleport], metadata={})

    # The strict observation remains an exact record of the producer row;
    # only global same-episode position admission is rejected.
    rejected_world = rejected_publication.observations[0].payload.world
    assert rejected_world is not None
    assert rejected_world.position.x == pytest.approx(20.0)
    retained_entity = rejected_publication.snapshot.entities[0]
    assert retained_entity.position is not None
    assert retained_entity.position.x == pytest.approx(10.0)
    assert retained_entity.conflict is True
    current = next(
        source
        for source in retained_entity.sources
        if source.observation_id == rejected_publication.observations[0].observation_id
    )
    assert current.accepted is False
    assert current.rejection_reason and current.rejection_reason.startswith(
        "velocity_gate:"
    )


def test_service_passes_private_source_epoch_to_fusion_velocity_clock() -> None:
    service = _service()

    def epoch_track(
        *,
        frame_id: int,
        observed_at_us: int,
        media_pts_ns: int,
        source_epoch: int,
        x: float,
    ) -> dict[str, Any]:
        track = _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=frame_id,
            observed_at_us=observed_at_us,
            x=x,
        )
        track.update(
            {
                "media_pts_ns": media_pts_ns,
                "source_epoch": source_epoch,
                "tracker_lifecycle_generation": 1,
                "trail_segment_id": 0,
            }
        )
        return track

    service.publish(
        0,
        [
            epoch_track(
                frame_id=1,
                observed_at_us=1_800_000,
                media_pts_ns=100_000_000_000,
                source_epoch=0,
                x=0.0,
            )
        ],
        metadata={"camera_id": "kitchen"},
    )
    reset = service.publish(
        0,
        [
            epoch_track(
                frame_id=2,
                observed_at_us=1_900_000,
                media_pts_ns=1_000_000_000,
                source_epoch=1,
                x=10.0,
            )
        ],
        metadata={"camera_id": "kitchen"},
    ).snapshot.entities[0]

    assert reset.position is not None
    assert reset.position.x == pytest.approx(10.0)
    assert reset.velocity_mps is None
    assert reset.conflict is False

    continued = service.publish(
        0,
        [
            epoch_track(
                frame_id=3,
                observed_at_us=2_000_000,
                media_pts_ns=1_100_000_000,
                source_epoch=1,
                x=10.2,
            )
        ],
        metadata={"camera_id": "kitchen"},
    ).snapshot.entities[0]

    assert continued.position is not None
    assert continued.position.x == pytest.approx(10.2)
    assert continued.velocity_mps is not None
    assert continued.velocity_mps.x == pytest.approx(2.0)
    assert continued.conflict is False


def test_world_observation_bounds_quality_reason_to_contract_limit() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["world_quality_reason"] = "diagnostic," * 40

    publication = service.publish(0, [track], metadata={})

    world = publication.observations[0].payload.world
    assert world is not None
    assert world.reason == str(track["world_quality_reason"])[:200]
    assert len(world.reason) == 200


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
    ("cv_prediction", "anchor_hold", "relative_motion_prediction"),
)
def test_unproven_display_continuity_never_enters_canonical_world(
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


@pytest.mark.parametrize(
    (
        "continuity_source",
        "provenance_type",
        "provenance_origin",
        "expected_x",
    ),
    (
        ("cv_prediction", "bounded_cv_process", "last_metric_process", 1.2),
        ("anchor_hold", "bounded_cv_process", "last_metric_process", 1.2),
        ("anchor_hold", "bounded_output_hold", "last_published_output", 1.0),
    ),
)
def test_bounded_process_continuity_updates_canonical_held_position(
    continuity_source: str,
    provenance_type: str,
    provenance_origin: str,
    expected_x: float,
) -> None:
    service = _service()
    authoritative = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )
    service.publish(0, [authoritative], metadata={})
    continuation = _bounded_process_track(
        continuity_source=continuity_source,
        provenance_type=provenance_type,
        provenance_origin=provenance_origin,
    )

    publication = service.publish(0, [continuation], metadata={})

    observation = publication.observations[0]
    assert observation.payload.world is not None
    assert observation.payload.world.source == continuity_source
    assert observation.payload.world.quality == "held"
    assert observation.payload.world.covariance.values[0] == pytest.approx(0.64)
    assert observation.coordinate_frame == "backend_world_m"
    assert (
        observation.payload.world.position.x,
        observation.payload.world.position.y,
        observation.payload.world.position.z,
    ) == pytest.approx((expected_x, 0.0, 2.0))
    assert publication.snapshot.entities[0].position is not None
    assert (
        publication.snapshot.entities[0].position.x,
        publication.snapshot.entities[0].position.y,
        publication.snapshot.entities[0].position.z,
    ) == pytest.approx((expected_x, 0.0, 2.0))
    assert publication.snapshot.entities[0].lifecycle == EntityLifecycle.PRESENT


@pytest.mark.parametrize(
    ("reason_length", "accepted"),
    (
        (240, True),
        (241, False),
    ),
)
def test_bounded_process_reason_has_a_fixed_boundary_limit(
    reason_length: int,
    accepted: bool,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bounded_process_track()
    reason = "r" * reason_length
    continuation["world_quality_reason"] = reason
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["reason"] = reason

    publication = service.publish(0, [continuation], metadata={})

    assert (publication.observations[0].payload.world is not None) is accepted


def test_bounded_process_continuity_is_not_cross_camera_fusion_authority() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(
        1,
        [
            _track(
                camera_id="hall",
                tracker_id=9,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.1,
            )
        ],
        metadata={},
    )
    image_publication = service.publish(
        0,
        [_bbox_affine_process_track()],
        metadata={},
    )
    assert image_publication.observations[0].payload.world is not None
    continuation = _recent_projective_bridge_track(
        origin_x=1.2,
        origin_media_pts_ns=1_100_000_000,
        current_x=1.2,
        current_media_pts_ns=1_200_000_000,
    )

    publication = service.publish(0, [continuation], metadata={})

    held_world = publication.observations[0].payload.world
    assert held_world is not None
    assert held_world.position.x == pytest.approx(1.2)
    entity = publication.snapshot.entities[0]
    assert entity.position is not None
    assert entity.position.x == pytest.approx(1.1)
    assert entity.conflict is False
    by_camera = {source.camera_id: source for source in entity.sources}
    assert by_camera["hall"].accepted is True
    assert by_camera["kitchen"].accepted is False
    assert by_camera["kitchen"].rejection_reason == (
        "non_authoritative_held_continuation"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("world_measurement_accepted", True),
        ("world_quality", "estimated"),
        ("world_filter_prediction", [1.3, 0.0, 2.0]),
        ("world_prediction_provenance", {"type": "bounded_cv_process"}),
    ),
)
def test_unproven_bounded_process_continuity_remains_diagnostic_only(
    field: str,
    value: Any,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bounded_process_track(
        provenance_origin="recent_projective_process"
    )
    continuation[field] = value

    publication = service.publish(0, [continuation], metadata={})

    observation = publication.observations[0]
    assert observation.payload.world is None
    assert observation.payload.world_diagnostics is not None
    assert observation.payload.world_diagnostics.first_divergence_reason == (
        "bounded_process_continuity_provenance_invalid"
    )
    assert publication.snapshot.entities[0].position is not None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    (
        ("transition", "origin_world", [1.1, 0.0, 2.0]),
        ("transition", "origin_media_pts_ns", 999_000_000),
        ("transition", "current_media_pts_ns", 1_099_000_000),
        ("transition", "origin_trail_segment_id", 1),
        ("transition", "metric_origin_world", [1.1, 0.0, 2.0]),
        ("transition", "metric_origin_media_pts_ns", 999_000_000),
        ("transition", "metric_origin_trail_segment_id", 1),
        ("transition", "tracker_lifecycle_generation", 2),
        ("transition", "position_base", [1.1, 0.0, 2.0]),
        ("transition", "position_gain", 0.5),
        ("transition", "max_speed_mps", 40.0),
        ("provenance", "reason", "different_reason"),
        ("track", "media_pts_ns", None),
        ("track", "trail_segment_id", 1),
        ("track", "tracker_lifecycle_generation", 2),
    ),
)
def test_bounded_process_rejects_mutated_origin_or_transition(
    scope: str,
    field: str,
    value: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bounded_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    if scope == "transition":
        transition = provenance["filter_transition"]
        assert isinstance(transition, dict)
        transition[field] = value
    elif scope == "provenance":
        provenance[field] = value
    else:
        continuation[field] = value

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_bounded_process_rejects_jointly_forged_posterior_and_origins() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bounded_process_track()
    continuation["world"] = [99.0, 0.0, 99.0]
    continuation["world_filter_prediction"] = [99.0, 0.0, 99.0]
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["process_observation"] = [99.0, 0.0, 99.0]
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition["origin_world"] = [98.9, 0.0, 99.0]
    transition["metric_origin_world"] = [98.9, 0.0, 99.0]
    transition["position_base"] = [98.9, 0.0, 99.0]

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("continuity_source", "provenance_type", "provenance_origin"),
    (
        ("cv_prediction", "bounded_cv_process", "last_metric_process"),
        ("anchor_hold", "bounded_cv_process", "last_metric_process"),
        ("anchor_hold", "bounded_output_hold", "last_published_output"),
    ),
)
@pytest.mark.parametrize(
    ("metric_age_ns", "accepted"),
    (
        (405_000_000, True),
        (405_000_001, False),
    ),
)
def test_nonstationary_bounded_process_is_bound_to_metric_horizon(
    continuity_source: str,
    provenance_type: str,
    provenance_origin: str,
    metric_age_ns: int,
    accepted: bool,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bounded_process_track(
        continuity_source=continuity_source,
        provenance_type=provenance_type,
        provenance_origin=provenance_origin,
    )
    current_pts = 1_000_000_000 + metric_age_ns
    continuation["media_pts_ns"] = current_pts
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition["current_media_pts_ns"] = current_pts
    transition["gate_dt_s"] = min(metric_age_ns / 1_000_000_000.0, 1.25)

    publication = service.publish(0, [continuation], metadata={})

    assert (publication.observations[0].payload.world is not None) is accepted


@pytest.mark.parametrize(
    ("metric_age_ns", "accepted"),
    (
        (2_005_000_000, True),
        (2_005_000_001, False),
    ),
)
def test_typed_stationary_exact_hold_uses_one_fixed_metric_horizon(
    metric_age_ns: int,
    accepted: bool,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _stationary_output_hold(
        metric_age_ns=metric_age_ns,
    )

    publication = service.publish(0, [continuation], metadata={})

    assert (publication.observations[0].payload.world is not None) is accepted


def test_pose_confirmed_nonseated_hold_renews_from_each_current_person_row() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    origin_pts = 1_000_000_000
    for current_pts in (
        1_300_000_000,
        1_600_000_000,
        1_900_000_000,
        2_200_000_000,
        2_500_000_000,
        2_800_000_000,
        3_005_000_000,
    ):
        publication = service.publish(
            0,
            [
                _upright_presence_output_hold(
                    origin_media_pts_ns=origin_pts,
                    current_media_pts_ns=current_pts,
                )
            ],
            metadata={},
        )
        assert publication.observations[0].payload.world is not None
        origin_pts = current_pts

    expired = service.publish(
        0,
        [
            _upright_presence_output_hold(
                origin_media_pts_ns=origin_pts,
                current_media_pts_ns=origin_pts + 2_005_000_001,
            )
        ],
        metadata={},
    )
    assert expired.observations[0].payload.world is None


def test_current_bbox_presence_hold_uses_one_fixed_kinematic_root() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    first = service.publish(
        0,
        [
            _current_presence_output_hold(
                visible_origin_media_pts_ns=1_000_000_000,
                kinematic_origin_media_pts_ns=1_000_000_000,
                current_media_pts_ns=1_500_000_000,
            )
        ],
        metadata={},
    )
    assert first.observations[0].payload.world is not None

    boundary = service.publish(
        0,
        [
            _current_presence_output_hold(
                visible_origin_media_pts_ns=1_500_000_000,
                kinematic_origin_media_pts_ns=1_000_000_000,
                current_media_pts_ns=2_005_000_000,
            )
        ],
        metadata={},
    )
    assert boundary.observations[0].payload.world is not None
    output = next(iter(service._canonical_track_outputs.values()))
    assert output.kinematic_media_pts_ns == 1_000_000_000
    assert output.kinematic_bbox_geometry == (10.0, 20.0, 30.0, 40.0)

    expired = service.publish(
        0,
        [
            _current_presence_output_hold(
                visible_origin_media_pts_ns=2_005_000_000,
                kinematic_origin_media_pts_ns=1_000_000_000,
                current_media_pts_ns=2_105_000_000,
            )
        ],
        metadata={},
    )
    assert expired.observations[0].payload.world is None


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    (
        ("track", "pose_present", False),
        ("track", "world_contact_basis", "bbox:bottom_center"),
        ("track", "world_floor_admitted", False),
        ("track", "world_observation_range_admitted", False),
        ("track", "lower_body_occluded", True),
        ("track", "bbox", [736.0, 382.0, 20.0, 40.0]),
        ("track", "world_floor_candidate", [1.2, 0.0, 2.0, 99.0]),
        ("evidence", "detector_confidence", 0.60),
        ("evidence", "floor_candidate", [1.2, 0.0]),
    ),
)
def test_upright_presence_evidence_is_bound_to_current_person_row(
    scope: str,
    field: str,
    value: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    first = _upright_presence_output_hold(
        origin_media_pts_ns=1_000_000_000,
        current_media_pts_ns=1_300_000_000,
    )
    assert service.publish(
        0, [first], metadata={}
    ).observations[0].payload.world is not None

    continuation = _upright_presence_output_hold(
        origin_media_pts_ns=1_300_000_000,
        current_media_pts_ns=1_600_000_000,
    )
    if scope == "track":
        continuation[field] = value
    else:
        provenance = continuation["world_prediction_provenance"]
        assert isinstance(provenance, dict)
        evidence = provenance["upright_presence_evidence"]
        assert isinstance(evidence, dict)
        evidence[field] = value

    publication = service.publish(0, [continuation], metadata={})
    assert publication.observations[0].payload.world is None


@pytest.mark.parametrize("motion_mode", ("sit", "lie"))
def test_nonseated_presence_hold_rejects_seated_motion_mode(
    motion_mode: str,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _upright_presence_output_hold(
        origin_media_pts_ns=1_000_000_000,
        current_media_pts_ns=1_600_000_000,
        motion_mode=motion_mode,
    )

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None


@pytest.mark.parametrize(
    ("track_field", "evidence_field"),
    (
        ("confidence", "detector_confidence"),
        ("tracker_confidence", "tracker_confidence"),
    ),
)
def test_upright_presence_confidence_must_remain_unit_bounded(
    track_field: str,
    evidence_field: str,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _upright_presence_output_hold(
        origin_media_pts_ns=1_000_000_000,
        current_media_pts_ns=1_600_000_000,
    )
    continuation[track_field] = 1.5
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    evidence = provenance["upright_presence_evidence"]
    assert isinstance(evidence, dict)
    evidence[evidence_field] = 1.5

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None


def test_upright_presence_hold_does_not_require_stationary_motion_label() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    first = _upright_presence_output_hold(
        origin_media_pts_ns=1_000_000_000,
        current_media_pts_ns=1_300_000_000,
    )
    assert service.publish(
        0, [first], metadata={}
    ).observations[0].payload.world is not None
    continuation = _upright_presence_output_hold(
        origin_media_pts_ns=1_300_000_000,
        current_media_pts_ns=1_600_000_000,
        motion_mode="unknown",
        bbox_stationary_supported=False,
    )

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is not None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_unknown_posture_presence_hold_survives_old_metric_origin() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    first = _upright_presence_output_hold(
        origin_media_pts_ns=1_000_000_000,
        current_media_pts_ns=1_500_000_000,
    )
    assert service.publish(
        0, [first], metadata={}
    ).observations[0].payload.world is not None

    # The metric root is now 2.066 s old, but the latest exact public output is
    # only 1.566 s old and this row carries fresh pose/floor presence evidence.
    continuation = _upright_presence_output_hold(
        origin_media_pts_ns=1_500_000_000,
        current_media_pts_ns=3_066_000_000,
        posture="unknown",
        motion_mode="unknown",
        bbox_stationary_supported=False,
    )
    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is not None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_upright_presence_floor_candidate_cannot_be_far_from_output() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_000_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    first = _upright_presence_output_hold(
        origin_media_pts_ns=1_000_000_000,
        current_media_pts_ns=1_300_000_000,
    )
    assert service.publish(
        0, [first], metadata={}
    ).observations[0].payload.world is not None
    far = _upright_presence_output_hold(
        origin_media_pts_ns=1_300_000_000,
        current_media_pts_ns=1_600_000_000,
        floor_candidate_x=2.30,
    )

    publication = service.publish(0, [far], metadata={})
    assert publication.observations[0].payload.world is None


def test_stationary_reason_token_alone_cannot_expand_metric_horizon() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _stationary_output_hold(
        metric_age_ns=2_000_000_000,
        include_evidence=False,
        reason="physical_measurement_rejected,stationary_bbox_hold",
    )

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_stationary_evidence_cannot_expand_moving_cv_horizon() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bounded_process_track(
        continuity_source="anchor_hold",
        provenance_type="bounded_cv_process",
        provenance_origin="last_metric_process",
    )
    current_media_pts_ns = 3_000_000_000
    continuation["media_pts_ns"] = current_media_pts_ns
    continuation["world_posture"] = "sitting"
    continuation["world_bbox_stationary_supported"] = True
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["stationary_evidence"] = {
        "version": 1,
        "kind": "bbox_stationary",
        "frame_id": continuation["frame_id"],
        "media_pts_ns": current_media_pts_ns,
        "posture": "sitting",
    }
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition["current_media_pts_ns"] = current_media_pts_ns
    transition["gate_dt_s"] = 1.25

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    (
        ("evidence", "version", 2),
        ("evidence", "kind", "reason_token_only"),
        ("evidence", "frame_id", 99),
        ("evidence", "media_pts_ns", 2_999_999_999),
        ("evidence", "posture", "lying"),
        ("track", "world_bbox_stationary_supported", False),
        ("track", "posture", "standing"),
    ),
)
def test_stationary_evidence_is_exactly_bound_to_current_track(
    scope: str,
    field: str,
    value: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _stationary_output_hold(
        metric_age_ns=2_000_000_000,
    )
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    evidence = provenance["stationary_evidence"]
    assert isinstance(evidence, dict)
    if scope == "evidence":
        evidence[field] = value
    else:
        continuation[field] = value

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None


def test_stationary_evidence_uses_tracking_posture_not_resolver_diagnostic() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _stationary_output_hold(
        metric_age_ns=2_000_000_000,
    )
    continuation["world_posture"] = "unknown"

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is not None

    # A resolver diagnostic cannot grant stationary authority when tracking
    # posture disagrees with the typed evidence.
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    forged = _stationary_output_hold(
        metric_age_ns=2_000_000_000,
    )
    forged["posture"] = "standing"
    forged["world_posture"] = "sitting"
    rejected = service.publish(0, [forged], metadata={})

    assert rejected.observations[0].payload.world is None


def test_bounded_cv_chain_cannot_renew_its_metric_horizon() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )

    prior_x = 1.0
    prior_pts = 1_000_000_000
    for step in range(1, 7):
        current_x = 1.0 + 0.05 * step
        current_pts = 1_000_000_000 + 100_000_000 * step
        continuation = _bounded_process_track()
        continuation["frame_id"] = step + 1
        continuation["observed_at_us"] = 1_900_000 + 100_000 * step
        continuation["media_pts_ns"] = current_pts
        continuation["world"] = [current_x, 0.0, 2.0]
        continuation["world_filter_prediction"] = [current_x, 0.0, 2.0]
        provenance = continuation["world_prediction_provenance"]
        assert isinstance(provenance, dict)
        provenance["process_observation"] = [current_x, 0.0, 2.0]
        transition = provenance["filter_transition"]
        assert isinstance(transition, dict)
        transition["origin_world"] = [prior_x, 0.0, 2.0]
        transition["origin_media_pts_ns"] = prior_pts
        transition["current_media_pts_ns"] = current_pts
        transition["position_base"] = [prior_x, 0.0, 2.0]
        transition["gate_dt_s"] = 0.1

        publication = service.publish(0, [continuation], metadata={})
        admitted = publication.observations[0].payload.world is not None
        assert admitted is (step <= 4)
        if admitted:
            prior_x = current_x
            prior_pts = current_pts


def test_canonical_track_output_cache_replaces_superseded_bindings() -> None:
    service = _service()

    first = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )
    service.publish(0, [first], metadata={})

    next_lifecycle = dict(first)
    next_lifecycle["frame_id"] = 2
    next_lifecycle["media_pts_ns"] = 1_100_000_000
    next_lifecycle["tracker_lifecycle_generation"] = 2
    service.publish(0, [next_lifecycle], metadata={})

    assert len(service._canonical_track_outputs) == 1
    lifecycle_key = next(iter(service._canonical_track_outputs))
    assert lifecycle_key[2:4] == (7, 2)

    next_revision = dict(next_lifecycle)
    next_revision["frame_id"] = 3
    next_revision["media_pts_ns"] = 1_200_000_000
    next_revision["world_frame_revision"] = "world-rev-2"
    next_revision["world_transform_sha256"] = "e" * 64
    service.publish(0, [next_revision], metadata={})

    assert len(service._canonical_track_outputs) == 1
    revision_key = next(iter(service._canonical_track_outputs))
    assert revision_key[3:] == (
        2,
        "world-rev-2",
        "e" * 64,
        "a" * 64,
        0,
    )


def test_canonical_continuation_never_crosses_source_epoch() -> None:
    service = _service()
    metric = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )
    metric["source_epoch"] = 0
    service.publish(0, [metric], metadata={})
    image = _bbox_affine_process_track()
    image["source_epoch"] = 0
    assert service.publish(
        0, [image], metadata={}
    ).observations[0].payload.world is not None

    stale_after_reconnect = _recent_projective_bridge_track(
        origin_x=1.2,
        origin_media_pts_ns=1_100_000_000,
        current_x=1.25,
        current_media_pts_ns=1_200_000_000,
    )
    stale_after_reconnect["source_epoch"] = 1
    publication = service.publish(
        0,
        [stale_after_reconnect],
        metadata={},
    )

    assert publication.observations[0].payload.world is None
    assert service._canonical_track_outputs == {}


def test_canonical_track_output_never_crosses_calibration_artifact_revision() -> None:
    calibration_sha256 = {"value": "a" * 64}

    def artifacts(_source_id: int, _metadata: object) -> WorldArtifacts:
        baseline = _artifacts()
        return WorldArtifacts(
            calibration=ArtifactFingerprint(
                role="camera_calibration",
                sha256=calibration_sha256["value"],
            ),
            model=baseline.model,
            config=baseline.config,
        )

    producer = _producer()
    service = CanonicalWorldService(
        producer=producer,
        artifacts=artifacts,
        fusion=GlobalWorldFusion(producer),
        clock_us=lambda: 2_000_000,
    )
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    initial_key = next(iter(service._canonical_track_outputs))
    assert initial_key[-2:] == ("a" * 64, 0)

    calibration_sha256["value"] = "f" * 64
    publication = service.publish(
        0,
        [_bounded_process_track()],
        metadata={},
    )

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)
    assert service._canonical_track_outputs == {}


def test_world_observation_requires_exact_raw_camera_calibration_digest() -> None:
    baseline = _artifacts()
    current_digest = "b" * 64
    service = CanonicalWorldService(
        producer=_producer(),
        artifacts=WorldArtifacts(
            calibration=ArtifactFingerprint(
                role="camera_calibration",
                sha256="e" * 64,
            ),
            model=baseline.model,
            config=baseline.config,
            camera_calibration_sha256=current_digest,
        ),
        clock_us=lambda: 2_000_000,
    )
    stale = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    stale["world_calibration_sha256"] = "a" * 64
    stale["world_floor_candidate"] = [99.0, 0.0, 99.0]

    publication = service.publish(0, [stale], metadata={})

    assert publication.observations[0].calibration.sha256 == "e" * 64
    assert publication.observations[0].payload.world is None
    diagnostics = publication.observations[0].payload.world_diagnostics
    assert diagnostics is not None
    assert diagnostics.first_divergence_reason == (
        "camera_calibration_sha256_mismatch"
    )
    assert diagnostics.floor_candidate_m is None
    assert publication.snapshot.entities == ()
    assert service._canonical_track_outputs == {}


def test_prepared_world_commit_rejects_camera_calibration_provider_switch() -> None:
    calibration_sha256 = {"value": "a" * 64}

    def artifacts(_source_id: int, _metadata: object) -> WorldArtifacts:
        baseline = _artifacts()
        digest = calibration_sha256["value"]
        return WorldArtifacts(
            calibration=ArtifactFingerprint(
                role="camera_calibration",
                sha256=digest,
            ),
            model=baseline.model,
            config=baseline.config,
            camera_calibration_sha256=digest,
        )

    service = CanonicalWorldService(
        producer=_producer(),
        artifacts=artifacts,
        clock_us=lambda: 2_000_000,
    )
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["world_calibration_sha256"] = "a" * 64
    prepared = service.prepare(
        0,
        [track],
        metadata={"camera_id": "kitchen"},
    )
    assert prepared.publication.observations[0].payload.world is not None

    calibration_sha256["value"] = "b" * 64
    with pytest.raises(
        RuntimeError,
        match="camera calibration authority changed before commit",
    ):
        service.commit(prepared)

    assert service.current_snapshot() is None
    assert service._revision == 0
    assert service._canonical_track_outputs == {}
    service.discard(prepared)


def test_empty_source_publication_preserves_short_gap_origins() -> None:
    service = _service()
    kitchen = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )
    hall = _set_image_motion_origin(
        _track(
            camera_id="hall",
            tracker_id=8,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )
    service.publish(0, [kitchen], metadata={})
    service.publish(1, [hall], metadata={})

    service.publish(
        0,
        [],
        metadata={"camera_id": "kitchen", "media_pts_ns": 1_200_000_000},
    )

    assert {key[0] for key in service._canonical_track_outputs} == {0, 1}
    kitchen_continuation = _bounded_process_track()
    kitchen_continuation["media_pts_ns"] = 1_300_000_000
    kitchen_provenance = kitchen_continuation["world_prediction_provenance"]
    assert isinstance(kitchen_provenance, dict)
    kitchen_transition = kitchen_provenance["filter_transition"]
    assert isinstance(kitchen_transition, dict)
    kitchen_transition["current_media_pts_ns"] = 1_300_000_000
    kitchen_transition["gate_dt_s"] = 0.3
    publication = service.publish(0, [kitchen_continuation], metadata={})
    assert publication.observations[0].payload.world is not None

    continuation = _bounded_process_track()
    continuation["camera_id"] = "hall"
    continuation["tracker_id"] = 8
    publication = service.publish(1, [continuation], metadata={})
    assert publication.observations[0].payload.world is not None


def test_canonical_track_output_cache_is_hard_bounded_and_keeps_live_origin() -> None:
    service = _service()
    service.MAX_CANONICAL_TRACK_OUTPUTS = 3
    for tracker_id in range(1, 4):
        track = _set_image_motion_origin(
            _track(
                camera_id="kitchen",
                tracker_id=tracker_id,
                frame_id=tracker_id,
                observed_at_us=1_900_000 + tracker_id,
                x=1.0,
            )
        )
        service.publish(0, [track], metadata={})

    # Refresh the oldest key before introducing more churn. Plain dict
    # replacement alone would leave it first in eviction order.
    hot = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=1,
            frame_id=4,
            observed_at_us=1_900_100,
            x=1.0,
        )
    )
    service.publish(0, [hot], metadata={})
    fourth = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=4,
            frame_id=5,
            observed_at_us=1_900_200,
            x=1.0,
        )
    )
    service.publish(0, [fourth], metadata={})

    assert len(service._canonical_track_outputs) == 3
    assert {key[2] for key in service._canonical_track_outputs} == {1, 3, 4}

    continuation = _bounded_process_track()
    continuation["tracker_id"] = 1
    publication = service.publish(0, [continuation], metadata={})
    assert publication.observations[0].payload.world is not None


def test_canonical_track_output_cache_prunes_when_all_horizons_expire() -> None:
    service = _service()
    origin = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )
    service.publish(0, [origin], metadata={})

    service.publish(
        0,
        [],
        metadata={
            "camera_id": "kitchen",
            "media_pts_ns": 3_005_000_001,
        },
    )

    assert service._canonical_track_outputs == {}


def _bbox_affine_process_track() -> dict[str, Any]:
    continuation = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=2,
        observed_at_us=1_950_000,
        x=1.2,
    )
    continuation.update(
        {
            "media_pts_ns": 1_100_000_000,
            "tracker_lifecycle_generation": 1,
            "trail_segment_id": 0,
            "world_source": "image_motion_prediction",
            "world_quality": "held",
            "world_measurement_accepted": False,
            "world_filter_prediction": [1.2, 0.0, 2.0],
            "world_prediction_provenance": {
                "type": "bbox_affine_floor_projection",
                "non_authoritative": True,
                "state_integrated": True,
                "origin": "last_accepted_pose_projective_origin",
                "transport": "pose_torso_translation",
                "image_foot": [640.0, 500.0],
                "age_s": 0.1,
                "projective_origin_world": [1.05, 0.0, 2.0],
                "process_observation": [1.25, 0.0, 2.0],
                "world_delta_m": 0.2,
                "world_delta_limit_m": 0.75,
                "filter_transition": _image_motion_filter_transition(
                    process_observation=[1.25, 0.0, 2.0],
                    position_base=[1.0, 0.0, 2.0],
                    position_gain=0.8,
                ),
            },
        }
    )
    return continuation


def _recent_projective_bridge_track(
    *,
    origin_x: float,
    origin_media_pts_ns: int,
    current_x: float,
    current_media_pts_ns: int,
) -> dict[str, Any]:
    continuation = _bounded_process_track(
        provenance_origin="recent_projective_process"
    )
    continuation["media_pts_ns"] = current_media_pts_ns
    continuation["world"] = [current_x, 0.0, 2.0]
    continuation["world_filter_prediction"] = [current_x, 0.0, 2.0]
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["process_observation"] = [current_x, 0.0, 2.0]
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition["origin_world"] = [origin_x, 0.0, 2.0]
    transition["origin_media_pts_ns"] = origin_media_pts_ns
    transition["current_media_pts_ns"] = current_media_pts_ns
    transition["position_base"] = [origin_x, 0.0, 2.0]
    transition["gate_dt_s"] = min(
        (current_media_pts_ns - origin_media_pts_ns) / 1_000_000_000.0,
        1.25,
    )
    return continuation


def _retimed_bounded_output_hold(
    *,
    origin_x: float,
    origin_media_pts_ns: int,
    current_media_pts_ns: int,
    trail_segment_id: int = 0,
    active_inferred_occlusion: bool = False,
) -> dict[str, Any]:
    hold = _bounded_process_track(
        continuity_source="anchor_hold",
        provenance_type="bounded_output_hold",
        provenance_origin="last_published_output",
        trail_segment_id=trail_segment_id,
    )
    hold["media_pts_ns"] = current_media_pts_ns
    hold["world"] = [origin_x, 0.0, 2.0]
    hold["world_filter_prediction"] = [origin_x, 0.0, 2.0]
    if active_inferred_occlusion:
        hold.update(
            {
                "posture": "standing",
                "motion_mode": "walk",
                "lower_body_occluded": True,
                "lower_body_occlusion_level": "knees",
            }
        )
    provenance = hold["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition.update(
        {
            "origin_world": [origin_x, 0.0, 2.0],
            "origin_media_pts_ns": origin_media_pts_ns,
            "current_media_pts_ns": current_media_pts_ns,
            "gate_dt_s": min(
                (current_media_pts_ns - origin_media_pts_ns)
                / 1_000_000_000.0,
                1.25,
            ),
            "position_base": [origin_x, 0.0, 2.0],
        }
    )
    return hold


@pytest.mark.parametrize(
    ("output_age_ns", "accepted"),
    (
        (1_655_000_000, True),
        (1_655_000_001, False),
    ),
)
def test_recent_projective_bridge_is_bound_to_image_output_age(
    output_age_ns: int,
    accepted: bool,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    image_publication = service.publish(
        0,
        [_bbox_affine_process_track()],
        metadata={},
    )
    assert image_publication.observations[0].payload.world is not None

    publication = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.25,
                current_media_pts_ns=1_100_000_000 + output_age_ns,
            )
        ],
        metadata={},
    )

    assert (publication.observations[0].payload.world is not None) is accepted


def test_delayed_projective_proof_can_name_root_after_newer_descendant() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    assert service.publish(
        0, [_bbox_affine_process_track()], metadata={}
    ).observations[0].payload.world is not None

    newer_descendant = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.25,
                current_media_pts_ns=2_400_000_000,
            )
        ],
        metadata={},
    )
    assert newer_descendant.observations[0].payload.world is not None

    delayed_root_proof = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.30,
                current_media_pts_ns=2_500_000_000,
            )
        ],
        metadata={},
    )

    assert delayed_root_proof.observations[0].payload.world is not None
    output = next(iter(service._canonical_track_outputs.values()))
    assert output.projective_bridge_root_media_pts_ns == 1_100_000_000


def test_recent_projective_bridge_chain_keeps_one_nonrenewing_image_root() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})
    first = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.25,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )
    assert first.observations[0].payload.world is not None
    first_output = next(iter(service._canonical_track_outputs.values()))
    assert first_output.projective_bridge_root_media_pts_ns == 1_100_000_000

    second = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.25,
                origin_media_pts_ns=1_200_000_000,
                current_x=1.30,
                current_media_pts_ns=1_300_000_000,
            )
        ],
        metadata={},
    )
    assert second.observations[0].payload.world is not None
    second_output = next(iter(service._canonical_track_outputs.values()))
    assert second_output.projective_bridge_root_media_pts_ns == 1_100_000_000

    boundary = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.30,
                origin_media_pts_ns=1_300_000_000,
                current_x=1.35,
                current_media_pts_ns=2_755_000_000,
            )
        ],
        metadata={},
    )
    assert boundary.observations[0].payload.world is not None
    boundary_output = next(iter(service._canonical_track_outputs.values()))
    assert boundary_output.projective_bridge_root_media_pts_ns == 1_100_000_000

    expired = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.35,
                origin_media_pts_ns=2_755_000_000,
                current_x=1.35,
                current_media_pts_ns=2_755_000_001,
            )
        ],
        metadata={},
    )
    assert expired.observations[0].payload.world is None
    assert expired.snapshot.entities[0].position.x == pytest.approx(1.35)


def test_queued_older_projective_origin_inherits_latest_service_image_root() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})
    newer_image = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.3,
                process_x=1.3,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )
    assert newer_image.observations[0].payload.world is not None

    first_bridge = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.25,
                current_media_pts_ns=1_300_000_000,
            )
        ],
        metadata={},
    )
    assert first_bridge.observations[0].payload.world is not None
    output = next(iter(service._canonical_track_outputs.values()))
    assert output.projective_bridge_root_media_pts_ns == 1_200_000_000

    older_origin_again = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.30,
                current_media_pts_ns=1_400_000_000,
            )
        ],
        metadata={},
    )
    assert older_origin_again.observations[0].payload.world is not None
    output = next(iter(service._canonical_track_outputs.values()))
    assert output.projective_bridge_root_media_pts_ns == 1_200_000_000

    expired = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.30,
                origin_media_pts_ns=1_400_000_000,
                current_x=1.30,
                current_media_pts_ns=2_855_000_001,
            )
        ],
        metadata={},
    )
    assert expired.observations[0].payload.world is None
    assert expired.snapshot.entities[0].position.x == pytest.approx(1.30)


def test_output_hold_inherits_recent_projective_bridge_episode() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})
    bridge = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.25,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )
    assert bridge.observations[0].payload.world is not None

    hold = service.publish(
        0,
        [
            _retimed_bounded_output_hold(
                origin_x=1.25,
                origin_media_pts_ns=1_200_000_000,
                current_media_pts_ns=1_300_000_000,
            )
        ],
        metadata={},
    )
    assert hold.observations[0].payload.world is not None
    output = next(iter(service._canonical_track_outputs.values()))
    assert output.projective_bridge_root_media_pts_ns == 1_100_000_000

    continued = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.25,
                origin_media_pts_ns=1_300_000_000,
                current_x=1.30,
                current_media_pts_ns=1_350_000_000,
            )
        ],
        metadata={},
    )
    assert continued.observations[0].payload.world is not None
    output = next(iter(service._canonical_track_outputs.values()))
    assert output.projective_bridge_root_media_pts_ns == 1_100_000_000


def test_recent_projective_bridge_accepts_older_committed_image_origin() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    first_image = service.publish(
        0,
        [_bbox_affine_process_track()],
        metadata={},
    )
    assert first_image.observations[0].payload.world is not None
    second_image = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.3,
                process_x=1.3,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )
    assert second_image.observations[0].payload.world is not None

    publication = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.25,
                current_media_pts_ns=1_300_000_000,
            )
        ],
        metadata={},
    )

    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.x == pytest.approx(1.25)


def test_recent_projective_history_rejects_after_metric_reanchor() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})
    metric_reanchor = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=3,
            observed_at_us=2_100_000,
            x=1.3,
        )
    )
    metric_reanchor["media_pts_ns"] = 1_200_000_000
    service.publish(0, [metric_reanchor], metadata={})

    publication = service.publish(
        0,
        [
            _recent_projective_bridge_track(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_x=1.25,
                current_media_pts_ns=1_300_000_000,
            )
        ],
        metadata={},
    )

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.3)


def test_bounded_output_hold_remains_latest_origin_only() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})
    service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.3,
                process_x=1.3,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )
    hold = _bounded_process_track(
        continuity_source="anchor_hold",
        provenance_type="bounded_output_hold",
        provenance_origin="last_published_output",
    )
    hold["media_pts_ns"] = 1_300_000_000
    hold["world"] = [1.2, 0.0, 2.0]
    hold["world_filter_prediction"] = [1.2, 0.0, 2.0]
    provenance = hold["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition.update(
        {
            "origin_world": [1.2, 0.0, 2.0],
            "origin_media_pts_ns": 1_100_000_000,
            "current_media_pts_ns": 1_300_000_000,
            "gate_dt_s": 0.2,
            "position_base": [1.2, 0.0, 2.0],
        }
    )

    publication = service.publish(0, [hold], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.3)


def test_latest_projective_image_allows_multiple_holds_under_one_root() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    image = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.2,
                process_x=1.2,
                current_media_pts_ns=1_400_000_000,
            )
        ],
        metadata={},
    )
    assert image.observations[0].payload.world is not None

    def _hold(*, origin_media_pts_ns: int, current_media_pts_ns: int) -> dict[str, Any]:
        track = _bounded_process_track(
            continuity_source="anchor_hold",
            provenance_type="bounded_output_hold",
            provenance_origin="last_published_output",
        )
        track["media_pts_ns"] = current_media_pts_ns
        track["world"] = [1.2, 0.0, 2.0]
        track["world_filter_prediction"] = [1.2, 0.0, 2.0]
        provenance = track["world_prediction_provenance"]
        assert isinstance(provenance, dict)
        transition = provenance["filter_transition"]
        assert isinstance(transition, dict)
        transition.update(
            {
                "origin_world": [1.2, 0.0, 2.0],
                    "origin_media_pts_ns": origin_media_pts_ns,
                    "current_media_pts_ns": current_media_pts_ns,
                    "gate_dt_s": min(
                        (current_media_pts_ns - origin_media_pts_ns)
                        / 1_000_000_000.0,
                        1.25,
                    ),
                "position_base": [1.2, 0.0, 2.0],
            }
        )
        return track

    first_hold = service.publish(
        0,
        [
            _hold(
                origin_media_pts_ns=1_400_000_000,
                current_media_pts_ns=1_500_000_000,
            )
        ],
        metadata={},
    )
    assert first_hold.observations[0].payload.world is not None
    held_output = next(iter(service._canonical_track_outputs.values()))
    assert held_output.committed_world_source == "anchor_hold"
    assert held_output.projective_bridge_root_media_pts_ns == 1_400_000_000

    second_hold = service.publish(
        0,
        [
            _hold(
                # The media worker formed this proof before the first hold's
                # admission receipt arrived. Both service-owned origins have
                # the exact same gain-zero coordinate and trail segment.
                origin_media_pts_ns=1_400_000_000,
                current_media_pts_ns=1_600_000_000,
            )
        ],
        metadata={},
    )
    assert second_hold.observations[0].payload.world is not None
    held_output = next(iter(service._canonical_track_outputs.values()))
    assert held_output.projective_bridge_root_media_pts_ns == 1_400_000_000

    boundary_hold = service.publish(
        0,
        [
            _hold(
                origin_media_pts_ns=1_600_000_000,
                current_media_pts_ns=3_055_000_000,
            )
        ],
        metadata={},
    )
    assert boundary_hold.observations[0].payload.world is not None
    held_output = next(iter(service._canonical_track_outputs.values()))
    assert held_output.projective_bridge_root_media_pts_ns == 1_400_000_000

    expired_hold = service.publish(
        0,
        [
            _hold(
                origin_media_pts_ns=3_055_000_000,
                current_media_pts_ns=3_055_000_001,
            )
        ],
        metadata={},
    )
    assert expired_hold.observations[0].payload.world is None
    assert expired_hold.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_gain_zero_hold_preserves_motion_bearing_kinematic_origin() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    image = service.publish(
        0,
        [_bbox_affine_process_track()],
        metadata={},
    )
    assert image.observations[0].payload.world is not None

    hold = service.publish(
        0,
        [
            _retimed_bounded_output_hold(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )
    assert hold.observations[0].payload.world is not None
    held_output = next(iter(service._canonical_track_outputs.values()))
    assert held_output.kinematic_position == pytest.approx((1.2, 0.0, 2.0))
    assert held_output.kinematic_media_pts_ns == 1_100_000_000

    # The held row did not observe the person at a new physical coordinate.
    # A later process step therefore receives its filter budget from the image
    # posterior at 1.1s. Its emitted coordinate is nevertheless limited by the
    # latest visible 1.2s row, preventing that retained budget from becoming a
    # one-frame display teleport.
    continuation = _recent_projective_bridge_track(
        origin_x=1.2,
        origin_media_pts_ns=1_100_000_000,
        current_x=2.0,
        current_media_pts_ns=1_400_000_000,
    )
    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.x == pytest.approx(2.0)


def test_gain_zero_hold_rejects_kinematically_valid_visible_teleport() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})
    service.publish(
        0,
        [
            _retimed_bounded_output_hold(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )

    teleport = _recent_projective_bridge_track(
        origin_x=1.2,
        origin_media_pts_ns=1_100_000_000,
        current_x=2.2,
        current_media_pts_ns=1_400_000_000,
    )
    publication = service.publish(0, [teleport], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_calibrated_image_motion_updates_canonical_held_position() -> None:
    service = _service()
    authoritative = _set_image_motion_origin(_track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    ))
    service.publish(0, [authoritative], metadata={})
    continuation = _bbox_affine_process_track()

    publication = service.publish(0, [continuation], metadata={})

    observation = publication.observations[0]
    assert observation.payload.world is not None
    assert observation.payload.world.source == "image_motion_prediction"
    assert observation.payload.world.quality == "held"
    assert observation.payload.world.covariance.values[0] == pytest.approx(0.64)
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def _retimed_bbox_affine_process_track(
    *,
    origin_x: float,
    origin_media_pts_ns: int,
    current_x: float,
    process_x: float,
    current_media_pts_ns: int,
) -> dict[str, Any]:
    continuation = _bbox_affine_process_track()
    continuation["media_pts_ns"] = current_media_pts_ns
    continuation["world"] = [current_x, 0.0, 2.0]
    continuation["world_filter_prediction"] = [current_x, 0.0, 2.0]
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["process_observation"] = [process_x, 0.0, 2.0]
    provenance["world_delta_m"] = abs(process_x - 1.05)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition.update(
        {
            "origin_world": [origin_x, 0.0, 2.0],
            "origin_media_pts_ns": origin_media_pts_ns,
            "current_media_pts_ns": current_media_pts_ns,
            "gate_dt_s": min(
                (current_media_pts_ns - origin_media_pts_ns)
                / 1_000_000_000.0,
                1.25,
            ),
            "position_base": [origin_x, 0.0, 2.0],
            "position_gain": 1.0,
        }
    )
    return continuation


def test_calibrated_image_motion_accepts_bounded_older_committed_origin() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    latest = service.publish(0, [_bbox_affine_process_track()], metadata={})
    assert latest.observations[0].payload.world is not None
    assert latest.observations[0].payload.world.position.x == pytest.approx(1.2)

    publication = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.3,
                process_x=1.3,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )

    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.x == pytest.approx(1.3)


def test_calibrated_image_motion_rejects_uncommitted_older_origin() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})

    publication = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.1,
                origin_media_pts_ns=1_050_000_000,
                current_x=1.3,
                process_x=1.3,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_calibrated_image_motion_rejects_old_origin_jump_from_latest() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})

    publication = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.7,
                process_x=1.7,
                current_media_pts_ns=1_200_000_000,
            )
        ],
        metadata={},
    )

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("origin_age_ns", "accepted"),
    ((1_250_000_000, True), (1_250_000_001, False)),
)
def test_calibrated_image_motion_committed_history_has_time_horizon(
    origin_age_ns: int,
    accepted: bool,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    service.publish(0, [_bbox_affine_process_track()], metadata={})
    current_media_pts_ns = 1_000_000_000 + origin_age_ns

    publication = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.3,
                process_x=1.3,
                current_media_pts_ns=current_media_pts_ns,
            )
        ],
        metadata={},
    )

    if accepted:
        assert publication.observations[0].payload.world is not None
        assert publication.observations[0].payload.world.position.x == pytest.approx(1.3)
    else:
        assert publication.observations[0].payload.world is None
        assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_calibrated_image_motion_prunes_expired_latest_before_validation() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )

    publication = service.publish(
        0,
        [
            _retimed_bbox_affine_process_track(
                origin_x=1.0,
                origin_media_pts_ns=1_000_000_000,
                current_x=1.2,
                process_x=1.2,
                current_media_pts_ns=4_000_000_000,
            )
        ],
        metadata={},
    )

    assert publication.observations[0].payload.world is None


def test_calibrated_image_motion_accepts_epoch_float_gate_clock_roundoff() -> None:
    service = _service()
    authoritative = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )
    service.publish(0, [authoritative], metadata={})
    continuation = _bbox_affine_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition["gate_dt_s"] = 1_780_000_000.1 - 1_780_000_000.0

    publication = service.publish(0, [continuation], metadata={})

    observation = publication.observations[0]
    assert observation.payload.world is not None
    assert observation.payload.world.position.x == pytest.approx(1.2)
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


@pytest.mark.parametrize(
    "posterior",
    (None, [1.3, 0.0, 2.0], [float("nan"), 0.0, 2.0]),
)
def test_calibrated_image_motion_requires_exact_integrated_posterior(
    posterior: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(_track(
                camera_id="kitchen",
                tracker_id=7,
                frame_id=1,
                observed_at_us=1_900_000,
                x=1.0,
            ))
        ],
        metadata={},
    )
    continuation = _bbox_affine_process_track()
    continuation["world_filter_prediction"] = posterior

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("origin_world", [1.1, 0.0, 2.0]),
        ("origin_media_pts_ns", 999_000_000),
        ("current_media_pts_ns", 1_099_000_000),
        ("origin_trail_segment_id", 1),
        ("gate_dt_s", 0.2),
        ("position_base", [1.1, 0.0, 2.0]),
        ("position_gain", 0.9),
        ("max_speed_mps", 40.0),
        ("max_jump_m", 7.5),
        ("reset_after_s", 12.5),
    ),
)
def test_calibrated_image_motion_rejects_mutated_filter_transition(
    field: str,
    value: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bbox_affine_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition[field] = value

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_calibrated_image_motion_rejects_jointly_forged_posterior() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                )
            )
        ],
        metadata={},
    )
    continuation = _bbox_affine_process_track()
    continuation["world"] = [99.0, 0.0, 99.0]
    continuation["world_filter_prediction"] = [99.0, 0.0, 99.0]

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def _inferred_ground_process_track() -> dict[str, object]:
    continuation = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=2,
        observed_at_us=1_950_000,
        x=1.2,
    )
    continuation.update(
        {
            "world_source": "image_motion_prediction",
            "world_quality": "held",
            "world_quality_reason": (
                "world_resolver_non_floor_diagnostic_only,"
                "guided_by_inferred_body_geometry"
            ),
            "world_measurement_accepted": False,
            "world_filter_prediction": [1.2, 0.0, 2.0],
            "media_pts_ns": 1_100_000_000,
            "world_observation_range_admitted": True,
            "tracker_lifecycle_generation": 1,
            "trail_segment_id": 4,
            "posture": "standing",
            "motion_mode": "walk",
            "lower_body_occluded": True,
            "lower_body_occlusion_level": "knees",
            "lower_body_occlusion_confidence": 0.89,
            "world_inferred_raw_observation": [10.5, 0.0, 19.5],
            "world_inferred_process_observation": [1.25, 0.0, 2.05],
            "world_prediction_provenance": {
                "type": "inferred_ground_process_observation",
                "non_authoritative": True,
                "state_integrated": True,
                "origin": "learned_body_height",
                "transport": "fixed_occlusion_origin_raw_world_delta",
                "support_kind": "lower_body_occlusion",
                "image_motion_supported": False,
                "image_motion_streak": 0,
                "image_motion_contact_basis": "",
                "detector_confidence": 0.91,
                "trusted_origin_kind": "queue_admitted_metric_world_output",
                "age_s": 0.05,
                "raw_origin_ts_s": 1.9,
                "raw_consensus_ts_s": 1.95,
                "current_ts_s": 1.95,
                "trusted_origin_filter_ts_s": 1.89,
                "origin_observed_at_us": 1_900_000,
                "raw_consensus_observed_at_us": 1_950_000,
                "observed_at_us": 1_950_000,
                "raw_origin_media_pts_ns": 1_050_000_000,
                "raw_consensus_media_pts_ns": 1_100_000_000,
                "trusted_origin_media_pts_ns": 1_000_000_000,
                "current_media_pts_ns": 1_100_000_000,
                "raw_consensus_method": "xz_medoid",
                "raw_consensus_count": 2,
                "raw_consensus_span_s": 0.05,
                "raw_evidence_gap_s": 0.05,
                "tracker_lifecycle_generation": 1,
                "origin_trail_segment_id": 4,
                "world_frame": "backend_world_m",
                "world_frame_revision": "world-rev",
                "world_transform_sha256": "d" * 64,
                "resolver_candidate_id": "gravity_reconstruction",
                "raw_sample": [10.5, 0.0, 19.5],
                "raw_consensus": [10.5, 0.0, 19.5],
                "raw_origin": [10.25, 0.0, 19.45],
                "raw_delta": [0.25, 0.0, 0.05],
                "trusted_world_origin": [1.0, 0.0, 2.0],
                "process_observation": [1.25, 0.0, 2.05],
                "filter_transition": _image_motion_filter_transition(
                    process_observation=[1.25, 0.0, 2.05],
                    position_base=[1.0, 0.0, 1.8],
                    position_gain=0.8,
                    trail_segment_id=4,
                ),
                "height_ref_scene": 1.78,
                "occlusion_level": "knees",
                "occlusion_confidence": 0.89,
            },
        }
    )
    return continuation


def _coherent_torso_inferred_ground_process_track() -> dict[str, object]:
    continuation = _inferred_ground_process_track()
    continuation.update(
        {
            "lower_body_occluded": False,
            "lower_body_occlusion_level": "none",
            "lower_body_occlusion_confidence": 0.0,
            "world_image_motion_supported": True,
            "world_image_motion_streak": 3,
            "world_contact_basis": "pose:torso_motion",
        }
    )
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance.update(
        {
            "transport": "fixed_torso_origin_raw_world_delta",
            "support_kind": "coherent_torso_motion",
            "image_motion_supported": True,
            "image_motion_streak": 3,
            "image_motion_contact_basis": "pose:torso_motion",
            "detector_confidence": 0.91,
            "occlusion_level": "none",
            "occlusion_confidence": 0.0,
        }
    )
    return continuation


def _robust_innovation_inferred_ground_process_track() -> dict[str, object]:
    continuation = _inferred_ground_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    raw_consensus = [12.25, 0.0, 19.45]
    raw_delta = [2.0, 0.0, 0.0]
    raw_process = [3.0, 0.0, 2.0]
    filter_observation = [2.15, 0.0, 2.0]
    posterior = [1.345, 0.0, 2.0]
    continuation.update(
        {
            "world": posterior,
            "world_filter_prediction": posterior,
            "world_inferred_raw_observation": raw_consensus,
            "world_inferred_process_observation": raw_process,
        }
    )
    provenance.update(
        {
            "raw_sample": raw_consensus,
            "raw_consensus": raw_consensus,
            "raw_delta": raw_delta,
            "process_observation": raw_process,
        }
    )
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition.update(
        {
            "position_base": [1.0, 0.0, 2.0],
            "position_gain": 0.3,
            "innovation_limit_applied": True,
            "raw_position_target": raw_process,
            "filter_observation": filter_observation,
            "raw_innovation_m": 2.0,
            "innovation_limit_m": 1.15,
            "innovation_scale": 0.575,
        }
    )
    return continuation


def _retimed_inferred_ground_process_track(
    *,
    origin_world: tuple[float, float, float] = (1.2, 0.0, 2.0),
    origin_media_pts_ns: int = 1_100_000_000,
    current_media_pts_ns: int = 1_833_000_000,
    current_ts_s: float = 2.683,
    raw_evidence_gap_s: float = 0.733,
) -> dict[str, object]:
    continuation = _inferred_ground_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    raw_origin = provenance["raw_origin"]
    trusted_origin = provenance["trusted_world_origin"]
    assert isinstance(raw_origin, list)
    assert isinstance(trusted_origin, list)
    raw_consensus = [10.5, 0.0, 19.5]
    raw_delta = [
        raw_consensus[index] - float(raw_origin[index])
        for index in range(3)
    ]
    process_observation = [
        float(trusted_origin[index]) + raw_delta[index]
        for index in range(3)
    ]
    position_gain = 0.8
    posterior = [
        float(origin_world[index])
        + position_gain
        * (process_observation[index] - float(origin_world[index]))
        for index in range(3)
    ]
    observed_at_us = int(round(current_ts_s * 1_000_000.0))
    continuation.update(
        {
            "observed_at_us": observed_at_us,
            "media_pts_ns": current_media_pts_ns,
            "world": posterior,
            "world_filter_prediction": posterior,
            "world_inferred_raw_observation": raw_consensus,
            "world_inferred_process_observation": process_observation,
        }
    )
    provenance.update(
        {
            "age_s": current_ts_s - float(provenance["raw_origin_ts_s"]),
            "raw_consensus_ts_s": current_ts_s,
            "current_ts_s": current_ts_s,
            "raw_consensus_observed_at_us": observed_at_us,
            "observed_at_us": observed_at_us,
            "raw_consensus_media_pts_ns": current_media_pts_ns,
            "current_media_pts_ns": current_media_pts_ns,
            "raw_consensus_count": 1,
            "raw_consensus_span_s": 0.0,
            "raw_evidence_gap_s": raw_evidence_gap_s,
            "raw_sample": raw_consensus,
            "raw_consensus": raw_consensus,
            "raw_delta": raw_delta,
            "process_observation": process_observation,
        }
    )
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition.update(
        {
            "origin_world": list(origin_world),
            "origin_media_pts_ns": origin_media_pts_ns,
            "current_media_pts_ns": current_media_pts_ns,
            "gate_dt_s": min(
                (current_media_pts_ns - origin_media_pts_ns)
                / 1_000_000_000.0,
                1.25,
            ),
            "position_base": list(origin_world),
            "position_gain": position_gain,
        }
    )
    return continuation


def test_inferred_ground_process_updates_exact_canonical_held_position() -> None:
    service = _service()
    authoritative = _set_image_motion_origin(_track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    ), trail_segment_id=4)
    service.publish(0, [authoritative], metadata={})

    publication = service.publish(
        0,
        [_inferred_ground_process_track()],
        metadata={},
    )

    observation = publication.observations[0]
    assert observation.payload.world is not None
    assert observation.payload.world.source == "image_motion_prediction"
    assert observation.payload.world.quality == "held"
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_coherent_torso_inferred_process_closes_established_contact_gap() -> None:
    service = _service()
    authoritative = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        ),
        trail_segment_id=4,
    )
    service.publish(0, [authoritative], metadata={})

    publication = service.publish(
        0,
        [_coherent_torso_inferred_ground_process_track()],
        metadata={},
    )

    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.source == (
        "image_motion_prediction"
    )
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    (
        ("provenance", "support_kind", "lower_body_occlusion"),
        (
            "provenance",
            "transport",
            "fixed_occlusion_origin_raw_world_delta",
        ),
        ("provenance", "image_motion_supported", False),
        ("provenance", "image_motion_streak", 2),
        ("provenance", "image_motion_contact_basis", "bbox_bottom"),
        ("provenance", "detector_confidence", 0.64),
        ("track", "lower_body_occluded", True),
        ("track", "world_image_motion_supported", False),
        ("track", "world_image_motion_streak", 2),
        ("track", "world_contact_basis", "bbox_bottom"),
    ),
)
def test_coherent_torso_inferred_process_rejects_incomplete_current_proof(
    scope: str,
    field: str,
    value: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    continuation = _coherent_torso_inferred_ground_process_track()
    if scope == "provenance":
        provenance = continuation["world_prediction_provenance"]
        assert isinstance(provenance, dict)
        provenance[field] = value
    else:
        continuation[field] = value

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_inferred_ground_robust_innovation_proof_is_canonical() -> None:
    service = _service()
    authoritative = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        ),
        trail_segment_id=4,
    )
    service.publish(0, [authoritative], metadata={})

    publication = service.publish(
        0,
        [_robust_innovation_inferred_ground_process_track()],
        metadata={},
    )

    assert publication.observations[0].payload.world is not None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.345)


def test_inferred_ground_robust_innovation_rejects_tampered_scale() -> None:
    service = _service()
    authoritative = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        ),
        trail_segment_id=4,
    )
    service.publish(0, [authoritative], metadata={})
    continuation = _robust_innovation_inferred_ground_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition["innovation_scale"] = 0.6

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("innovation_limit_applied", False),
        ("raw_position_target", [3.01, 0.0, 2.0]),
        ("filter_observation", [2.14, 0.0, 2.0]),
        ("raw_innovation_m", 1.99),
        ("innovation_limit_m", 1.14),
        ("innovation_scale", 0.574),
    ),
)
def test_inferred_ground_robust_innovation_rejects_each_mutated_proof_member(
    field: str,
    value: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    continuation = _robust_innovation_inferred_ground_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition[field] = value

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field",
    (
        "raw_position_target",
        "filter_observation",
        "raw_innovation_m",
        "innovation_limit_m",
        "innovation_scale",
    ),
)
def test_inferred_ground_robust_innovation_rejects_partial_proof(
    field: str,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    continuation = _robust_innovation_inferred_ground_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition.pop(field)

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def _service_with_inferred_ground_root_and_hold() -> CanonicalWorldService:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    first = service.publish(
        0,
        [_inferred_ground_process_track()],
        metadata={},
    )
    assert first.observations[0].payload.world is not None
    hold = service.publish(
        0,
        [
            _retimed_bounded_output_hold(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_media_pts_ns=1_200_000_000,
                trail_segment_id=4,
                active_inferred_occlusion=True,
            )
        ],
        metadata={},
    )
    assert hold.observations[0].payload.world is not None
    return service


def test_inferred_ground_root_ends_when_current_occlusion_episode_ends() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    first = service.publish(
        0,
        [_inferred_ground_process_track()],
        metadata={},
    )
    assert first.observations[0].payload.world is not None
    assert next(
        iter(service._canonical_track_outputs.values())
    ).inferred_ground_episode_root is not None

    ended = service.publish(
        0,
        [
            _retimed_bounded_output_hold(
                origin_x=1.2,
                origin_media_pts_ns=1_100_000_000,
                current_media_pts_ns=1_200_000_000,
                trail_segment_id=4,
            )
        ],
        metadata={},
    )

    assert ended.observations[0].payload.world is not None
    assert next(
        iter(service._canonical_track_outputs.values())
    ).inferred_ground_episode_root is None


def test_long_gap_inferred_ground_restart_keeps_exact_service_episode_root() -> None:
    service = _service_with_inferred_ground_root_and_hold()
    held_output = next(iter(service._canonical_track_outputs.values()))
    episode_root = held_output.inferred_ground_episode_root
    assert episode_root is not None
    assert held_output.prior_outputs[0].inferred_ground_episode_root is episode_root

    restart = _retimed_inferred_ground_process_track()
    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.x == pytest.approx(
        1.24
    )
    restarted_output = next(iter(service._canonical_track_outputs.values()))
    assert restarted_output.inferred_ground_episode_root is episode_root

    following = _retimed_inferred_ground_process_track(
        origin_world=(1.24, 0.0, 2.04),
        origin_media_pts_ns=1_833_000_000,
        current_media_pts_ns=1_933_000_000,
        current_ts_s=2.783,
        raw_evidence_gap_s=0.1,
    )
    following_publication = service.publish(0, [following], metadata={})

    assert following_publication.observations[0].payload.world is not None
    following_output = next(iter(service._canonical_track_outputs.values()))
    assert following_output.inferred_ground_episode_root is episode_root


def test_long_gap_inferred_ground_restart_requires_service_owned_root() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    restart = _retimed_inferred_ground_process_track(
        origin_world=(1.0, 0.0, 2.0),
        origin_media_pts_ns=1_000_000_000,
    )

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("raw_origin_media_pts_ns", 1_049_000_000),
        ("origin_observed_at_us", 1_899_000),
        ("trusted_origin_media_pts_ns", 999_000_000),
        ("trusted_origin_filter_ts_s", 1.88),
        ("height_ref_scene", 1.79),
    ),
)
def test_long_gap_inferred_ground_restart_rejects_changed_root_scalar(
    field: str,
    value: object,
) -> None:
    service = _service_with_inferred_ground_root_and_hold()
    restart = _retimed_inferred_ground_process_track()
    provenance = restart["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance[field] = value

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_long_gap_inferred_ground_restart_rejects_changed_raw_origin_time() -> None:
    service = _service_with_inferred_ground_root_and_hold()
    restart = _retimed_inferred_ground_process_track()
    provenance = restart["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["raw_origin_ts_s"] = 1.89
    provenance["age_s"] = float(provenance["current_ts_s"]) - 1.89

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_long_gap_inferred_ground_restart_rejects_changed_raw_root_with_valid_algebra() -> None:
    service = _service_with_inferred_ground_root_and_hold()
    restart = _retimed_inferred_ground_process_track()
    provenance = restart["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["raw_origin"] = [10.2, 0.0, 19.45]
    raw_consensus = provenance["raw_consensus"]
    trusted_origin = provenance["trusted_world_origin"]
    assert isinstance(raw_consensus, list)
    assert isinstance(trusted_origin, list)
    raw_delta = [
        float(raw_consensus[index])
        - float(provenance["raw_origin"][index])
        for index in range(3)
    ]
    process = [
        float(trusted_origin[index]) + raw_delta[index]
        for index in range(3)
    ]
    posterior = [
        1.2 + 0.8 * (process[0] - 1.2),
        0.0,
        2.0 + 0.8 * (process[2] - 2.0),
    ]
    provenance["raw_delta"] = raw_delta
    provenance["process_observation"] = process
    restart["world_inferred_process_observation"] = process
    restart["world"] = posterior
    restart["world_filter_prediction"] = posterior

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_long_gap_inferred_ground_restart_rejects_changed_trusted_root_with_valid_algebra() -> None:
    service = _service_with_inferred_ground_root_and_hold()
    restart = _retimed_inferred_ground_process_track()
    provenance = restart["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    provenance["trusted_world_origin"] = [1.01, 0.0, 2.0]
    raw_delta = provenance["raw_delta"]
    assert isinstance(raw_delta, list)
    process = [
        float(provenance["trusted_world_origin"][index])
        + float(raw_delta[index])
        for index in range(3)
    ]
    posterior = [
        1.2 + 0.8 * (process[0] - 1.2),
        0.0,
        2.0 + 0.8 * (process[2] - 2.0),
    ]
    provenance["process_observation"] = process
    restart["world_inferred_process_observation"] = process
    restart["world"] = posterior
    restart["world_filter_prediction"] = posterior

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


@pytest.mark.parametrize(
    "boundary",
    ("lifecycle", "world_frame", "revision", "transform", "segment"),
)
def test_long_gap_inferred_ground_restart_rejects_changed_episode_boundary(
    boundary: str,
) -> None:
    service = _service_with_inferred_ground_root_and_hold()
    restart = _retimed_inferred_ground_process_track()
    provenance = restart["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    if boundary == "lifecycle":
        restart["tracker_lifecycle_generation"] = 2
        provenance["tracker_lifecycle_generation"] = 2
    elif boundary == "world_frame":
        restart["world_frame"] = "other_world_m"
        provenance["world_frame"] = "other_world_m"
    elif boundary == "revision":
        restart["world_frame_revision"] = "other-revision"
        provenance["world_frame_revision"] = "other-revision"
    elif boundary == "transform":
        restart["world_transform_sha256"] = "e" * 64
        provenance["world_transform_sha256"] = "e" * 64
    else:
        restart["trail_segment_id"] = 5
        provenance["origin_trail_segment_id"] = 5
        transition["origin_trail_segment_id"] = 5

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_long_gap_inferred_ground_restart_rejects_excessive_output_age() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    service.publish(0, [_inferred_ground_process_track()], metadata={})
    restart = _retimed_inferred_ground_process_track(
        current_media_pts_ns=2_350_000_001,
    )

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


def test_long_gap_inferred_ground_restart_rejects_gap_beyond_reset_age() -> None:
    service = _service_with_inferred_ground_root_and_hold()
    restart = _retimed_inferred_ground_process_track(
        current_ts_s=3.150002,
        raw_evidence_gap_s=1.250002,
    )

    publication = service.publish(0, [restart], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    (
        ("provenance", "origin", "last_accepted_image_foot"),
        ("provenance", "transport", "current_reference_plane_projection"),
        ("provenance", "trusted_origin_kind", "filter_state"),
        ("provenance", "tracker_lifecycle_generation", 99),
        ("provenance", "observed_at_us", 1_949_999),
        ("provenance", "raw_delta", [0.6, 0.0, -0.5]),
        ("provenance", "raw_consensus", [10.6, 0.0, 19.5]),
        ("provenance", "raw_consensus_ts_s", 1.96),
        ("provenance", "raw_consensus_observed_at_us", 1_950_001),
        ("provenance", "raw_consensus_media_pts_ns", 2_001),
        ("provenance", "raw_consensus_method", "mean"),
        ("provenance", "raw_consensus_count", 6),
        ("provenance", "raw_consensus_span_s", -0.1),
        ("provenance", "raw_evidence_gap_s", 0.41),
        ("provenance", "trusted_world_origin", [0.8, 0.0, 2.55]),
        ("provenance", "process_observation", [1.3, 0.0, 2.05]),
        ("provenance", "age_s", 0.06),
        ("provenance", "current_ts_s", 1.96),
        ("provenance", "current_media_pts_ns", 1_999),
        ("provenance", "origin_trail_segment_id", 3),
        ("provenance", "world_frame_revision", "other-world"),
        ("track", "lower_body_occluded", False),
        ("track", "world_observation_range_admitted", False),
        ("track", "world_inferred_raw_observation", [10.6, 0.0, 19.5]),
        ("track", "world_inferred_process_observation", [1.3, 0.0, 2.05]),
        ("track", "trail_segment_id", 3),
        ("track", "world_filter_prediction", [1.3, 0.0, 2.0]),
    ),
)
def test_malformed_inferred_ground_process_remains_diagnostic_only(
    scope: str,
    field: str,
    value: object,
) -> None:
    service = _service()
    authoritative = _set_image_motion_origin(_track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    ), trail_segment_id=4)
    service.publish(0, [authoritative], metadata={})
    continuation = _inferred_ground_process_track()
    if scope == "provenance":
        provenance = continuation["world_prediction_provenance"]
        assert isinstance(provenance, dict)
        provenance[field] = value
    else:
        continuation[field] = value

    publication = service.publish(0, [continuation], metadata={})

    observation = publication.observations[0]
    assert observation.payload.world is None
    assert observation.payload.world_diagnostics is not None
    assert observation.payload.world_diagnostics.first_divergence_reason == (
        "image_motion_continuity_provenance_invalid"
    )
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("origin_world", [1.1, 0.0, 2.0]),
        ("origin_media_pts_ns", 999_000_000),
        ("gate_dt_s", 0.2),
        ("position_base", [1.0, 0.0, 1.9]),
        ("position_gain", 0.9),
    ),
)
def test_inferred_ground_process_rejects_mutated_filter_transition(
    field: str,
    value: object,
) -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    continuation = _inferred_ground_process_track()
    provenance = continuation["world_prediction_provenance"]
    assert isinstance(provenance, dict)
    transition = provenance["filter_transition"]
    assert isinstance(transition, dict)
    transition[field] = value

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_inferred_ground_process_rejects_jointly_forged_posterior() -> None:
    service = _service()
    service.publish(
        0,
        [
            _set_image_motion_origin(
                _track(
                    camera_id="kitchen",
                    tracker_id=7,
                    frame_id=1,
                    observed_at_us=1_900_000,
                    x=1.0,
                ),
                trail_segment_id=4,
            )
        ],
        metadata={},
    )
    continuation = _inferred_ground_process_track()
    continuation["world"] = [99.0, 0.0, 99.0]
    continuation["world_filter_prediction"] = [99.0, 0.0, 99.0]

    publication = service.publish(0, [continuation], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities[0].position.x == pytest.approx(1.0)


def test_unproven_image_motion_remains_diagnostic_only() -> None:
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
    continuation["world_source"] = "image_motion_prediction"
    continuation["world_quality"] = "held"
    continuation["world_measurement_accepted"] = False
    continuation["world_prediction_provenance"] = {
        "type": "bbox_affine_floor_projection",
        "non_authoritative": True,
        "state_integrated": False,
        "origin": "last_accepted_image_foot",
        "transport": "bbox_affine",
        "image_foot": [640.0, 500.0],
        "age_s": 0.1,
        "world_delta_m": 0.2,
        "world_delta_limit_m": 0.75,
    }

    publication = service.publish(0, [continuation], metadata={})

    observation = publication.observations[0]
    assert observation.payload.world is None
    assert observation.payload.world_diagnostics is not None
    assert (
        observation.payload.world_diagnostics.first_divergence_reason
        == "image_motion_continuity_provenance_invalid"
    )
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


def test_explicitly_rejected_ordinary_measurement_is_not_fused() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["world_measurement_accepted"] = False
    track["world_quality_reason"] = "physical_innovation_exceeded"

    publication = service.publish(0, [track], metadata={})

    assert publication.observations[0].payload.world is None
    assert publication.snapshot.entities == ()


def test_ordinary_source_cannot_masquerade_as_held_process_output() -> None:
    service = _service()
    track = _track(
        camera_id="kitchen",
        tracker_id=7,
        frame_id=1,
        observed_at_us=1_900_000,
        x=1.0,
    )
    track["world_quality"] = "held"

    publication = service.publish(0, [track], metadata={})

    assert publication.observations[0].payload.world is None
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


def test_tracking_publisher_clears_world_row_rejected_by_canonical_service() -> None:
    websocket = _WebSocketRecorder()
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {"camera_id": "kitchen"},
        world_service=_service(),
    )
    proposed = _bounded_process_track()

    receipt = publisher.publish(0, [proposed])

    tracking = websocket.messages[0]
    public_track = tracking["tracks"][0]
    assert proposed["world_valid"] is True
    assert public_track["world_valid"] is False
    assert public_track["world_quality"] == "invalid"
    assert public_track["world_quality_reason"] == (
        "canonical_world_service_rejected"
    )
    assert public_track["trail_append_allowed"] is False
    assert public_track["trail_break_required"] is True
    assert "world" not in public_track
    assert "world_source" not in public_track
    assert "world_filter_prediction" not in public_track
    assert "world_prediction_provenance" not in public_track
    assert tracking["observations"][0]["payload"]["world"] is None
    assert tracking["world_snapshot"]["entities"] == []
    assert receipt.canonical_world_admission_bound is True
    assert receipt.canonical_world_track_keys == ()


def test_tracking_publisher_receipt_binds_exact_admitted_world_row() -> None:
    websocket = _WebSocketRecorder()
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {"camera_id": "kitchen"},
        world_service=_service(),
    )
    track = _set_image_motion_origin(
        _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=1,
            observed_at_us=1_900_000,
            x=1.0,
        )
    )

    receipt = publisher.publish(0, [track])

    assert websocket.messages[0]["tracks"][0]["world_valid"] is True
    assert receipt.canonical_world_admission_bound is True
    assert receipt.canonical_world_track_keys == ((7, 1, 1),)


def test_velocity_rejection_clears_tracking_authority_and_process_origin() -> None:
    websocket = _WebSocketRecorder()
    service = _service()
    publisher = TrackingTelemetryPublisher(
        websocket,
        metadata_getter=lambda _source_id, _tracks: {"camera_id": "kitchen"},
        world_service=service,
    )

    def point(*, frame_id: int, media_pts_ns: int, x: float) -> dict[str, Any]:
        track = _track(
            camera_id="kitchen",
            tracker_id=7,
            frame_id=frame_id,
            observed_at_us=1_800_000 + (frame_id - 1) * 100_000,
            x=x,
        )
        track.update(
            {
                "media_pts_ns": media_pts_ns,
                "tracker_lifecycle_generation": 1,
                "trail_segment_id": 0,
                "trail_break_required": False,
                "trail_append_allowed": True,
            }
        )
        return track

    publisher.publish(
        0,
        [point(frame_id=1, media_pts_ns=1_000_000_000, x=0.0)],
    )
    rejected = point(frame_id=2, media_pts_ns=1_100_000_000, x=2.0)
    receipt = publisher.publish(0, [rejected])

    tracking_messages = [
        message
        for message in websocket.messages
        if message["type"] == "tracking"
    ]
    tracking = tracking_messages[1]
    public_track = tracking["tracks"][0]
    assert public_track["world_valid"] is False
    assert "world" not in public_track
    assert public_track["world_quality_reason"] == (
        "canonical_world_service_rejected"
    )
    assert receipt.canonical_world_track_keys == ()

    entity = tracking["world_snapshot"]["entities"][0]
    assert entity["position"]["x"] == pytest.approx(0.0)
    current_source = next(
        source
        for source in entity["sources"]
        if source["observation_id"].endswith(":1:7")
    )
    assert current_source["accepted"] is False
    assert current_source["rejection_reason"].startswith("velocity_gate:")

    output = next(iter(service._canonical_track_outputs.values()))
    assert output.position[0] == pytest.approx(0.0)
    assert output.media_pts_ns == 1_000_000_000

    process = _recent_projective_bridge_track(
        origin_x=2.0,
        origin_media_pts_ns=1_100_000_000,
        current_x=2.0,
        current_media_pts_ns=1_200_000_000,
    )
    process["frame_id"] = 3
    process["observed_at_us"] = 2_000_000
    later = service.publish(0, [process], metadata={})

    assert later.observations[0].payload.world is None
    assert later.snapshot.entities[0].position.x == pytest.approx(0.0)


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
