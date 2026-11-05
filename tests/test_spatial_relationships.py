from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from noesis_core.contracts.spatial_relationships import SpatialRelationshipsRead
from noesis_core.contracts.world import WorldAttributionClaim, WorldAttributionQuarantine, WorldSnapshot
from noesis_core.spatial_relationships import build_spatial_relationships, encode_spatial_relationships
from tests.test_world_read_api import world  # Actual disposable committed WorldService fixture.


def supported_snapshot(world, *, entity_changes=None, source_changes=None):
    payload = world.snapshot.model_dump(mode="json")
    entity = payload["entities"][0]
    surface = {"surface_id": "actual-selected-plane", "revision": "measured-fixture-v1", "normal_world": [0, 1, 0],
        "offset_m": 0, "extent_xz": [[-3, -3], [3, -3], [3, 3], [-3, 3]]}
    entity["support_state"] = "floor"; entity["support_surface"] = surface
    entity["subject"]["display_name"] = "<literal source name>"
    for source in entity["sources"]:
        source["support_surface"] = copy.deepcopy(surface)
        if source_changes: source.update(source_changes)
    if entity_changes: entity.update(entity_changes)
    return WorldSnapshot.model_validate(payload)


def quarantined_snapshot(world):
    run = world.snapshot.producer.run_id
    owners = [WorldAttributionClaim(run_id=run, physical_serial=serial, camera_id="kitchen", source_id=0,
        source_epoch=0, raw_id=serial, lifecycle_generation=1, lifecycle_started_frame=1, frame_id=2,
        observation_id=f"fixture:{serial}", acquired_at_s=1.9, clock_domain="synthetic-native", media_time_s=0.1,
        clock_basis="synthetic", household_state="resident", household_identity_key="contested-fixture",
        household_stable_id=9, household_visitor_generation=None, household_constraint="eligible") for serial in (1, 2)]
    payload = dict(status="quarantined", purpose="physical_only_quarantine", reason="settled_household_claim_conflict",
        run_id=run, association_revision=3, subject_ids=["resident:contested-fixture"], stable_ids=[9],
        owners=[owner.model_dump(mode="json") for owner in owners], retained_owners=[], component_entity_ids=[f"physical:{run}:{serial}" for serial in (1, 2)])
    record = WorldAttributionQuarantine(conflict_id=WorldAttributionQuarantine.digest(payload), **payload)
    data = supported_snapshot(world).model_dump(mode="json"); base = data["entities"][0]; data["entities"] = []
    for serial in (1, 2):
        entity = copy.deepcopy(base)
        entity.update(entity_id=f"physical:{run}:{serial}", subject={"subject_id": f"quarantine:{run}:{serial}", "kind": "provisional", "generation": 0,
            "resident_uuid": None, "display_name": None, "stable_id": None}, attribution_quarantine=record.model_dump(mode="json"))
        data["entities"].append(entity)
    return WorldSnapshot.model_validate(data)


def test_exact_original_room_support_and_evidence_are_immutable_as_of_projection_without_geometry_or_duration_inference(world):
    original = supported_snapshot(world)
    before = original.model_dump(mode="json")
    result = build_spatial_relationships(original, world.clock[0])
    assert result.status == "available" and result.valid_until_us == original.published_at_us + 2_000_000
    assert result.producer is original.producer and result.snapshot_id == original.snapshot_id and result.sequence == original.sequence
    assert result.frame == original.frame and result.units == original.units
    row, entity = result.entities[0], original.entities[0]
    assert row.entity_id == entity.entity_id and row.display_label == entity.subject.display_name
    assert row.assigned_to_room.state == "known" and row.assigned_to_room.room_id == entity.room_id == "Kitchen"
    assert row.assigned_to_room.reason == "recorded_authoritative_room_assignment"
    assert row.supported_by.state == "known" and row.supported_by.support_surface is entity.support_surface
    assert row.supported_by.support_state == entity.support_state == "floor"
    assert row.sources == entity.sources and all(a is b for a, b in zip(row.sources, entity.sources, strict=True))
    assert row.observed_at_us == entity.observed_at_us and row.stale_after_us == entity.stale_after_us
    assert row.world_frame_revision == entity.world_frame_revision and row.world_transform_sha256 == entity.world_transform_sha256
    assert row.dwell.state == "unknown" and row.dwell.duration_us is None
    assert not {"position", "distance", "seat_id", "coverage", "occupancy"} & set(type(row).model_fields)
    assert original.model_dump(mode="json") == before
    assert SpatialRelationshipsRead.model_validate_json(encode_spatial_relationships(result)) == result
    with pytest.raises(ValidationError): row.room_id = "invented"
    with pytest.raises(ValidationError): result.entities = ()
    data = result.model_dump(mode="json"); data["invented"] = True
    with pytest.raises(ValidationError): SpatialRelationshipsRead.model_validate(data)


@pytest.mark.parametrize(("entity_changes", "source_changes", "reason"), [
    ({"lifecycle": "held"}, None, "entity_held"),
    ({"lifecycle": "lost", "position": None, "covariance": None}, None, "entity_lost"),
    ({"conflict": True, "conflict_reason": "Original conflict"}, None, "entity_conflict"),
    ({"position_state": "held"}, None, "entity_not_measured"),
    ({"position_state": "predicted"}, None, "entity_not_measured"),
    ({"stale_after_us": 2_000_000}, None, "entity_expired"),
    ({"observed_at_us": 2_000_001}, None, "entity_observation_in_future"),
    ({"world_frame_revision": None, "world_transform_sha256": None, "calibration_revision": None},
        {"world_frame_revision": None, "world_transform_sha256": None, "calibration_revision": None}, "binding_unavailable"),
    (None, {"position_state": "held"}, "source_evidence_not_current"),
])
def test_held_lost_conflicting_expired_predicted_or_unbound_never_become_settled_relations(world, entity_changes, source_changes, reason):
    original = supported_snapshot(world, entity_changes=entity_changes, source_changes=source_changes)
    result = build_spatial_relationships(original, world.clock[0]); row = result.entities[0]
    assert result.status == "available"
    for relation in (row.assigned_to_room, row.supported_by): assert relation.state == "unknown" and relation.reason == reason
    assert row.assigned_to_room.room_id is None and row.supported_by.support_surface is None
    assert row.room_id == original.entities[0].room_id and row.sources == original.entities[0].sources
    assert row.dwell.duration_us is None


def test_quarantine_preserves_original_attribution_component_without_settled_identity_or_room_support_claims(world):
    original = quarantined_snapshot(world); result = build_spatial_relationships(original, world.clock[0])
    assert len(result.entities) == 2
    for row, entity in zip(result.entities, original.entities, strict=True):
        assert row.display_label == entity.entity_id
        assert row.attribution_quarantine is entity.attribution_quarantine
        assert row.assigned_to_room.reason == row.supported_by.reason == "attribution_quarantined"
        assert row.assigned_to_room.room_id is None and row.supported_by.support_surface is None


def test_room_requires_exact_authoritative_accepted_zone_and_support_needs_exact_selected_plane(world):
    original = supported_snapshot(world, entity_changes={"room_id": None}, source_changes={"zone": "Kitchen", "zone_source": "camera_default", "zone_authoritative": False})
    result = build_spatial_relationships(original, world.clock[0]); assert result.entities[0].assigned_to_room.reason == "room_unassigned"
    original = supported_snapshot(world, entity_changes={"support_surface": None})
    result = build_spatial_relationships(original, world.clock[0]); assert result.entities[0].supported_by.reason == "support_surface_unavailable"
    original = supported_snapshot(world, source_changes={"support_surface": None})
    result = build_spatial_relationships(original, world.clock[0]); assert result.entities[0].supported_by.reason == "support_source_unavailable"
    original = supported_snapshot(world, entity_changes={"support_state": "unknown"})
    result = build_spatial_relationships(original, world.clock[0]); assert result.entities[0].supported_by.reason == "support_unknown"


def test_multiple_camera_specific_complete_bindings_do_not_require_invented_common_transform(world):
    payload = supported_snapshot(world).model_dump(mode="json"); entity = payload["entities"][0]
    other = copy.deepcopy(entity["sources"][0]); other.update(observation_id="other-original", camera_id="other-camera", world_transform_sha256="e" * 64, calibration_revision="other-calibration")
    entity["sources"].append(other); entity["world_transform_sha256"] = entity["calibration_revision"] = None
    original = WorldSnapshot.model_validate(payload); row = build_spatial_relationships(original, world.clock[0]).entities[0]
    assert row.assigned_to_room.state == row.supported_by.state == "known"
    assert row.world_transform_sha256 is None and len(row.sources) == 2
    assert row.sources[0].world_transform_sha256 != row.sources[1].world_transform_sha256


@pytest.mark.parametrize("at", [None, 0, -1, True, 2.5, float("nan"), 2**53])
def test_bad_clocks_are_unknown_not_epoch_reinterpreted(world, at):
    result = build_spatial_relationships(world.snapshot, at)
    assert result.status == "unknown" and result.reason == "invalid_clock" and result.generated_at_us is None
    assert result.entities == () and result.producer is None


def test_no_read_renews_snapshot_expiry_and_empty_snapshot_does_not_enumerate_rooms_or_infer_empty_home(world):
    original = supported_snapshot(world)
    result = build_spatial_relationships(original, 2_000_000); later = build_spatial_relationships(original, 2_100_000)
    assert result.valid_until_us == later.valid_until_us == 4_000_000 and later.generated_at_us == 2_100_000
    for at in (1_999_999, 4_000_000, 4_000_001): assert build_spatial_relationships(original, at).reason == "snapshot_not_fresh"
    empty = original.model_copy(update={"entities": ()}); result = build_spatial_relationships(empty, 2_000_000)
    assert result.status == "available" and result.entities == ()


def test_entity_evidence_node_native_field_and_wire_bounds_fail_as_whole_result_without_truncation(world):
    original = supported_snapshot(world); entity = original.entities[0]
    assert build_spatial_relationships(original.model_copy(update={"entities": (entity,) * 257}), 2_000_000).reason == "entity_limit_exceeded"
    excessive = entity.model_copy(update={"sources": entity.sources * 4097})
    assert build_spatial_relationships(original.model_copy(update={"entities": (excessive,)}), 2_000_000).reason == "source_evidence_limit_exceeded"
    assert build_spatial_relationships(original.model_copy(update={"observed_start_us": 2**53}), 2_000_000).reason == "snapshot_bound_exceeded"
    # Real valid large retained evidence is bounded before response exposure.
    payload = original.model_dump(mode="json"); entity = payload["entities"][0]
    source = copy.deepcopy(entity["sources"][0]); source.update(camera_id="🟦" * 160, observation_id="🟦" * 200)
    entity["sources"] = [source] * 120
    result = build_spatial_relationships(WorldSnapshot.model_validate(payload), 2_000_000)
    assert result.status == "unknown" and result.reason == "response_byte_limit_exceeded" and not result.entities
    assert len(encode_spatial_relationships(result)) < 1024
