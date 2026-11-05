"""Pure read-side relationships from one exact immutable World snapshot."""
from __future__ import annotations

import json
import math

from pydantic import BaseModel, ValidationError

from noesis_core.contracts.spatial_relationships import (
    AssignedToRoom, EntitySpatialRelationships, MAX_ENTITIES, MAX_RESPONSE_BYTES,
    MAX_SNAPSHOT_AGE_US, SpatialRelationshipsRead, SupportedBy, UnknownDwell,
)
from noesis_core.contracts.world import WorldEntity, WorldSnapshot, WorldSourceEvidence, validate_world_identity_uniqueness

MAX_SOURCE_ROWS = 4096
MAX_MODEL_NODES = 65_536


def unknown_spatial_relationships(reason: str, generated_at_us: int | None = None) -> SpatialRelationshipsRead:
    if type(generated_at_us) is not int or not 0 < generated_at_us <= 2**53 - 1:
        generated_at_us = None
    return SpatialRelationshipsRead(status="unknown", reason=reason, generated_at_us=generated_at_us,
        valid_until_us=None, producer=None, snapshot_id=None, sequence=None, published_at_us=None,
        observed_start_us=None, observed_end_us=None, frame=None, units=None, entities=())


def encode_spatial_relationships(value: SpatialRelationshipsRead) -> bytes | None:
    """Finite response encoding; never create an unbounded serialized string."""
    size, chunks = 0, []
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for text in encoder.iterencode(value.model_dump(mode="json")):
        chunk = text.encode("utf-8")
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _bounded(snapshot: WorldSnapshot) -> bool:
    nodes = 0

    def visit(value, name="", depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_MODEL_NODES or depth > 24:
            return False
        if name.endswith("_us") and value is not None and (type(value) is not int or not 0 < value <= 2**53 - 1):
            return False
        if value is None or isinstance(value, (str, bool)):
            return not isinstance(value, str) or len(value) <= 4096
        if type(value) is int:
            return abs(value) <= 2**53 - 1
        if type(value) is float:
            return math.isfinite(value)
        if isinstance(value, BaseModel):
            return all(visit(getattr(value, field), field, depth + 1) for field in type(value).model_fields)
        if isinstance(value, tuple):
            return len(value) <= MAX_MODEL_NODES - nodes and all(visit(item, name, depth + 1) for item in value)
        return False

    return visit(snapshot)


def _unsettled(entity: WorldEntity, generated_at_us: int) -> str | None:
    if entity.attribution_quarantine is not None:
        return "attribution_quarantined"
    if entity.lifecycle == "lost":
        return "entity_lost"
    if entity.lifecycle == "held":
        return "entity_held"
    if entity.conflict:
        return "entity_conflict"
    if entity.position_state != "measured":
        return "entity_not_measured"
    if entity.observed_at_us > generated_at_us:
        return "entity_observation_in_future"
    if generated_at_us >= entity.stale_after_us:
        return "entity_expired"
    accepted = tuple(source for source in entity.sources if source.accepted)
    if not accepted:
        return "source_evidence_unavailable"
    if entity.world_frame_revision is None or any(
        source.world_frame != entity.world_frame or source.world_frame_revision != entity.world_frame_revision
        or source.world_transform_sha256 is None or source.calibration_revision is None
        or entity.world_transform_sha256 is not None and source.world_transform_sha256 != entity.world_transform_sha256
        or entity.calibration_revision is not None and source.calibration_revision != entity.calibration_revision
        for source in accepted
    ):
        return "binding_unavailable"
    if any(source.position_state != "measured" or source.position_quality == "held"
           or source.observed_at_us > generated_at_us or source.observed_at_us > entity.observed_at_us
           for source in accepted):
        return "source_evidence_not_current"
    return None


def _entity(entity: WorldEntity, generated_at_us: int) -> EntitySpatialRelationships:
    blocked = _unsettled(entity, generated_at_us)
    accepted = tuple(source for source in entity.sources if source.accepted)
    room_reason = blocked
    if room_reason is None:
        rooms = {source.zone for source in accepted if source.zone_source == "nvdsanalytics_roi" and source.zone_authoritative and source.zone is not None}
        if entity.room_id is None:
            room_reason = "room_unassigned"
        elif rooms != {entity.room_id}:
            room_reason = "room_source_unavailable"
    support_reason = blocked
    if support_reason is None:
        if entity.support_state == "unknown":
            support_reason = "support_unknown"
        elif entity.support_surface is None:
            support_reason = "support_surface_unavailable"
        elif any(source.support_surface != entity.support_surface for source in accepted):
            support_reason = "support_source_unavailable"
    label = entity.entity_id
    if entity.attribution_quarantine is None and entity.subject.display_name:
        label = entity.subject.display_name
    return EntitySpatialRelationships(entity_id=entity.entity_id, display_label=label, lifecycle=entity.lifecycle,
        position_state=entity.position_state, conflict=entity.conflict, conflict_reason=entity.conflict_reason,
        attribution_quarantine=entity.attribution_quarantine, observed_at_us=entity.observed_at_us, stale_after_us=entity.stale_after_us,
        world_frame=entity.world_frame, world_frame_revision=entity.world_frame_revision, world_transform_sha256=entity.world_transform_sha256,
        calibration_revision=entity.calibration_revision, room_id=entity.room_id, support_state=entity.support_state,
        support_surface=entity.support_surface, sources=entity.sources,
        assigned_to_room=AssignedToRoom(state="unknown" if room_reason else "known", reason=room_reason or "recorded_authoritative_room_assignment",
            room_id=None if room_reason else entity.room_id),
        supported_by=SupportedBy(state="unknown" if support_reason else "known", reason=support_reason or "recorded_support_surface",
            support_state=None if support_reason else entity.support_state, support_surface=None if support_reason else entity.support_surface),
        dwell=UnknownDwell())


def build_spatial_relationships(snapshot: WorldSnapshot, generated_at_us: int) -> SpatialRelationshipsRead:
    """No clock calls, state retention, publication, inference or writes."""
    if type(generated_at_us) is not int or not 0 < generated_at_us <= 2**53 - 1:
        return unknown_spatial_relationships("invalid_clock")
    if type(snapshot) is not WorldSnapshot:
        return unknown_spatial_relationships("snapshot_unavailable", generated_at_us)
    if not isinstance(snapshot.entities, tuple):
        return unknown_spatial_relationships("invalid_snapshot", generated_at_us)
    if len(snapshot.entities) > MAX_ENTITIES:
        return unknown_spatial_relationships("entity_limit_exceeded", generated_at_us)
    if any(type(entity) is not WorldEntity for entity in snapshot.entities):
        return unknown_spatial_relationships("invalid_snapshot", generated_at_us)
    source_count = 0
    for entity in snapshot.entities:
        if not isinstance(entity.sources, tuple):
            return unknown_spatial_relationships("invalid_snapshot", generated_at_us)
        source_count += len(entity.sources)
        if source_count > MAX_SOURCE_ROWS:
            return unknown_spatial_relationships("source_evidence_limit_exceeded", generated_at_us)
        if any(type(source) is not WorldSourceEvidence for source in entity.sources):
            return unknown_spatial_relationships("invalid_snapshot", generated_at_us)
    if not _bounded(snapshot):
        return unknown_spatial_relationships("snapshot_bound_exceeded", generated_at_us)
    published = snapshot.published_at_us
    if published > 2**53 - 1 - MAX_SNAPSHOT_AGE_US or not published <= generated_at_us < published + MAX_SNAPSHOT_AGE_US:
        return unknown_spatial_relationships("snapshot_not_fresh", generated_at_us)
    try:
        validate_world_identity_uniqueness(snapshot.entities)
        result = SpatialRelationshipsRead(status="available", reason="recorded_world_relationships_as_of_read", generated_at_us=generated_at_us,
            valid_until_us=published + MAX_SNAPSHOT_AGE_US, producer=snapshot.producer, snapshot_id=snapshot.snapshot_id,
            sequence=snapshot.sequence, published_at_us=published, observed_start_us=snapshot.observed_start_us, observed_end_us=snapshot.observed_end_us,
            frame=snapshot.frame, units=snapshot.units, entities=tuple(_entity(entity, generated_at_us) for entity in snapshot.entities))
        if encode_spatial_relationships(result) is None:
            return unknown_spatial_relationships("response_byte_limit_exceeded", generated_at_us)
        return result
    except (ValidationError, ValueError, TypeError, OverflowError, RecursionError):
        return unknown_spatial_relationships("invalid_snapshot", generated_at_us)
