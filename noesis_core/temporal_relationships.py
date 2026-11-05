"""Bounded request-side observation spans over original persisted snapshots."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from noesis_core.contracts.world import WorldSnapshot
from noesis_core.contracts.world_history import WorldHistoryPersistence
from noesis_core.contracts.temporal_relationships import (MAX_TEMPORAL_BYTES, MAX_TEMPORAL_ENTITIES,
    ObservedAssignmentSpan, TemporalEntityRelationships, TemporalRelationshipSource,
    TemporalRelationshipsRead, TemporalSnapshotEvidence)
from noesis_core.journal import AsyncContractJournal, ContractJournal, ContractJournalError
from noesis_core.spatial_relationships import build_spatial_relationships
from noesis_core.world_history import HistoryUnavailable, WorldHistoryReader

READ_DEADLINE_SECONDS = 0.75


def unknown_temporal_relationships(reason, at_us=None):
    if type(at_us) is not int or not 0 < at_us <= 2**53 - 1:
        at_us = None
    return TemporalRelationshipsRead(status="unknown", reason=reason, generated_at_us=at_us)


def encode_temporal_relationships(value):
    chunks, size = [], 0
    for text in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(value.model_dump(mode="json")):
        chunk = text.encode("utf-8"); size += len(chunk)
        if size > MAX_TEMPORAL_BYTES:
            raise HistoryUnavailable("temporal_response_byte_limit_exceeded")
        chunks.append(chunk)
    return b"".join(chunks)


def _binding(entity):
    return (entity.subject, entity.world_frame, entity.world_frame_revision, entity.world_transform_sha256,
        entity.calibration_revision, tuple(sorted((s.camera_id, s.world_frame, s.world_frame_revision,
        s.world_transform_sha256, s.calibration_revision) for s in entity.sources if s.accepted)))


def _evidence(record, snapshot, entity):
    return TemporalSnapshotEvidence(journal_sequence=record.sequence, recorded_at_us=record.recorded_at_us,
        previous_sha256=record.previous_sha256, record_sha256=record.record_sha256,
        snapshot_id=snapshot.snapshot_id, snapshot_sequence=snapshot.sequence, producer=snapshot.producer,
        published_at_us=snapshot.published_at_us, observed_start_us=snapshot.observed_start_us,
        observed_end_us=snapshot.observed_end_us, frame=snapshot.frame, units=snapshot.units,
        entity_id=entity.entity_id, subject=entity.subject, lifecycle=entity.lifecycle, position_state=entity.position_state,
        conflict=entity.conflict, quarantined=entity.attribution_quarantine is not None, room_id=entity.room_id,
        entity_observed_at_us=entity.observed_at_us, entity_stale_after_us=entity.stale_after_us,
        world_frame_revision=entity.world_frame_revision, world_transform_sha256=entity.world_transform_sha256,
        calibration_revision=entity.calibration_revision, sources=entity.sources)


def build_temporal_relationships(records, *, current, at_us, persistence, retained_first, retained_previous, omitted):
    """Pure projection. Distinct point observations never prove continuous dwell."""
    if not records:
        return unknown_temporal_relationships("persisted_snapshot_unavailable", at_us)
    snapshots = tuple(WorldSnapshot.model_validate(record.payload) for record in records)
    head = snapshots[-1]
    if current.producer != head.producer:
        return unknown_temporal_relationships("persisted_run_changed", at_us)
    if current.sequence < head.sequence or current.published_at_us < head.published_at_us:
        return unknown_temporal_relationships("current_snapshot_precedes_persisted_head", at_us)
    if not head.published_at_us <= at_us < head.published_at_us + 2_000_000:
        return unknown_temporal_relationships("persisted_head_not_fresh", at_us)
    if len(head.entities) > MAX_TEMPORAL_ENTITIES:
        return unknown_temporal_relationships("temporal_entity_limit_exceeded", at_us)
    projections = tuple(build_spatial_relationships(snapshot, snapshot.published_at_us) for snapshot in snapshots)
    if any(item.status != "available" for item in projections):
        return unknown_temporal_relationships("persisted_snapshot_invalid_or_over_limit", at_us)
    rows = []
    for entity, relationship in zip(head.entities, projections[-1].entities, strict=True):
        evidence = [_evidence(records[-1], head, entity)]
        boundary = "snapshot_window_limit" if omitted else "retained_history_boundary"
        blocked = relationship.assigned_to_room.reason if relationship.assigned_to_room.state != "known" else None
        if at_us >= entity.stale_after_us:
            blocked = "head_entity_expired"
        distinct = {entity.observed_at_us} if blocked is None else set()
        if blocked is None:
            newer_snapshot, newer_entity = head, entity
            for index in range(len(snapshots) - 2, -1, -1):
                older_snapshot = snapshots[index]
                if older_snapshot.producer != head.producer:
                    boundary = "producer_changed"; break
                if older_snapshot.sequence + 1 != newer_snapshot.sequence:
                    boundary = "snapshot_sequence_gap"; break
                if (older_snapshot.published_at_us >= newer_snapshot.published_at_us
                        or older_snapshot.observed_end_us > newer_snapshot.observed_end_us):
                    boundary = "snapshot_clock_regression"; break
                older_entity = next((item for item in older_snapshot.entities if item.entity_id == entity.entity_id), None)
                if older_entity is None:
                    boundary = "entity_missing"; break
                older_relation = next(item for item in projections[index].entities if item.entity_id == entity.entity_id)
                if older_relation.assigned_to_room.state != "known":
                    boundary = older_relation.assigned_to_room.reason; break
                if _binding(older_entity) != _binding(entity):
                    boundary = "identity_or_frame_binding_changed"; break
                if older_entity.room_id != entity.room_id:
                    boundary = "room_assignment_changed"; break
                if older_entity.observed_at_us > newer_entity.observed_at_us:
                    boundary = "entity_observation_clock_regression"; break
                if (older_snapshot.published_at_us + 2_000_000 <= newer_snapshot.published_at_us
                        or older_entity.stale_after_us <= newer_entity.observed_at_us):
                    boundary = "observation_gap"; break
                evidence.append(_evidence(records[index], older_snapshot, older_entity))
                distinct.add(older_entity.observed_at_us)
                newer_snapshot, newer_entity = older_snapshot, older_entity
        known = blocked is None and len(distinct) >= 2
        span = ObservedAssignmentSpan(state="known" if known else "unknown",
            reason="consecutive_recorded_assignment_observations" if known else blocked or "insufficient_distinct_observations",
            room_id=entity.room_id if known else None, first_observed_at_us=min(distinct) if known else None,
            last_observed_at_us=max(distinct) if known else None, span_us=max(distinct) - min(distinct) if known else None,
            observation_count=len(distinct), snapshot_report_count=len(evidence), boundary_reason=blocked or boundary,
            evidence=tuple(reversed(evidence)))
        rows.append(TemporalEntityRelationships(entity_id=entity.entity_id, display_label=relationship.display_label,
            observed_assignment_span=span))
    same_head = current.snapshot_id == head.snapshot_id and current.sequence == head.sequence
    partial = (not same_head or omitted or retained_first > 0 or persistence.status == "failed" or persistence.worker_alive is False
        or persistence.error_reported or persistence.rejected_cohorts or persistence.pending_batches or persistence.inflight_batches)
    valid_until = min([head.published_at_us + 2_000_000] + [entity.stale_after_us for entity, row in zip(head.entities, rows, strict=True)
        if row.observed_assignment_span.state == "known"])
    return TemporalRelationshipsRead(status="partial" if partial else "available",
        reason="bounded_or_incomplete_persisted_window" if partial else "consecutive_persisted_observation_reports",
        generated_at_us=at_us, valid_until_us=valid_until, persistence=persistence,
        source=TemporalRelationshipSource(producer=head.producer, head_snapshot_id=head.snapshot_id,
            head_snapshot_sequence=head.sequence, head_published_at_us=head.published_at_us,
            head_journal_sequence=records[-1].sequence, head_record_sha256=records[-1].record_sha256,
            current_snapshot_id_at_read=current.snapshot_id, head_matches_current_at_read=same_head,
            inspected_snapshot_count=len(records), retained_first_journal_sequence=retained_first,
            retained_first_previous_sha256=retained_previous, earlier_snapshots_omitted=omitted,
            retention_pruned=retained_first > 0), entities=tuple(rows))


class TemporalRelationshipsReader:
    def __init__(self):
        self._readers = threading.BoundedSemaphore(1)

    def read(self, service, *, is_current=lambda: True):
        if not self._readers.acquire(blocking=False):
            return encode_temporal_relationships(unknown_temporal_relationships("temporal_reader_limit_exceeded"))
        try:
            return self._read(service, is_current)
        except HistoryUnavailable as exc:
            return encode_temporal_relationships(unknown_temporal_relationships(str(exc)))
        except (ContractJournalError, sqlite3.Error, ValueError, TypeError, OverflowError, RecursionError):
            return encode_temporal_relationships(unknown_temporal_relationships("temporal_journal_unavailable"))
        finally:
            self._readers.release()

    def _read(self, service, is_current):
        deadline = time.monotonic() + READ_DEADLINE_SECONDS
        source = WorldHistoryReader._capture(service, is_current)
        journal = source[0]; durable = journal.journal if type(journal) is AsyncContractJournal else journal
        if type(durable) is not ContractJournal:
            raise HistoryUnavailable("persistence_source_unavailable")
        health = WorldHistoryReader._health(source)
        if health.status == "closed":
            raise HistoryUnavailable("persistence_closed")
        try:
            records, retained_first, retained_previous, omitted = durable.inspect_world_snapshots(deadline=deadline)
        except ContractJournalError as exc:
            if str(exc).startswith("temporal_"):
                raise HistoryUnavailable(str(exc)) from exc
            raise
        except sqlite3.OperationalError as exc:
            raise HistoryUnavailable("temporal_read_deadline_exceeded" if time.monotonic() >= deadline else "temporal_journal_busy") from exc
        latest_source = WorldHistoryReader._capture(service, is_current)
        if latest_source[0] is not journal or latest_source[1] != source[1] or latest_source[2] is not source[2]:
            raise HistoryUnavailable("world_source_changed")
        current, at_us, reason = service.read_current_snapshot()
        if reason != "available" or type(current) is not WorldSnapshot or current.producer != source[1]:
            raise HistoryUnavailable(reason if reason != "available" else "world_owner_changed")
        later = WorldHistoryReader._health(latest_source)
        if later.status == "closed":
            raise HistoryUnavailable("persistence_closed")
        health = WorldHistoryPersistence(status="failed" if "failed" in {health.status, later.status} else later.status,
            pending_batches=max(health.pending_batches or 0, later.pending_batches or 0) if type(journal) is AsyncContractJournal else None,
            inflight_batches=max(health.inflight_batches or 0, later.inflight_batches or 0) if type(journal) is AsyncContractJournal else None,
            worker_alive=later.worker_alive, rejected_cohorts=max(health.rejected_cohorts, later.rejected_cohorts),
            error_reported=health.error_reported or later.error_reported)
        result = build_temporal_relationships(records, current=current, at_us=at_us, persistence=health,
            retained_first=retained_first, retained_previous=retained_previous, omitted=omitted)
        body = encode_temporal_relationships(result)
        final_source = WorldHistoryReader._capture(service, is_current)
        if (final_source[0] is not journal or final_source[1] != source[1] or final_source[2] is not source[2]
                or type(journal) is AsyncContractJournal and journal.journal is not durable):
            raise HistoryUnavailable("world_source_changed")
        final, final_us, reason = service.read_current_snapshot()
        if reason != "available" or final.producer != source[1]:
            raise HistoryUnavailable("world_owner_changed")
        if result.status != "unknown" and (not result.generated_at_us <= final_us < result.valid_until_us):
            raise HistoryUnavailable("temporal_head_expired_during_read")
        durable._validate_main_binding()
        if durable._closed or type(journal) is AsyncContractJournal and journal._closed:
            raise HistoryUnavailable("persistence_closed")
        if time.monotonic() >= deadline:
            raise HistoryUnavailable("temporal_read_deadline_exceeded")
        return body


temporal_relationships_reader = TemporalRelationshipsReader()
