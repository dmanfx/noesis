"""Bounded inspection of already durable canonical WorldEvent records."""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from noesis_core.contracts.world_history import (HISTORY_LEASE_US, MAX_HISTORY_BYTES, MAX_HISTORY_ROWS,
    WorldHistoryInspection, WorldHistoryPersistence, WorldHistoryRead, WorldHistoryRecord)
from noesis_core.journal import AsyncContractJournal, ContractJournal, ContractJournalError

MAX_SCAN_ROWS = 10_000
MAX_SCAN_BYTES = 32 * 1024 * 1024
READ_DEADLINE_SECONDS = 0.75
MAX_INSPECTIONS = 64


class HistoryUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class _Cursor:
    service: Any
    journal: Any
    durable: ContractJournal
    producer: Any
    inspection: WorldHistoryInspection
    expires_monotonic: float
    before: int


def unknown_world_history(reason: str) -> WorldHistoryRead:
    return WorldHistoryRead(status="unavailable", reason=reason)


def encode_world_history(value: WorldHistoryRead) -> bytes:
    chunks, size = [], 0
    for text in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(value.model_dump(mode="json")):
        chunk = text.encode("utf-8"); size += len(chunk)
        if size > MAX_HISTORY_BYTES:
            raise HistoryUnavailable("response_byte_limit_exceeded")
        chunks.append(chunk)
    return b"".join(chunks)


class WorldHistoryReader:
    """One request reader; bounded process-local cursors grant inspection only."""

    def __init__(self):
        self._readers = threading.BoundedSemaphore(1)
        self._cursors: OrderedDict[str, _Cursor] = OrderedDict()

    @staticmethod
    def _clock(source):
        try:
            stamp = source[2]()
        except Exception as exc:
            raise HistoryUnavailable("invalid_clock") from exc
        if type(stamp) is not int or not 0 < stamp <= 2**53 - 1 - HISTORY_LEASE_US:
            raise HistoryUnavailable("invalid_clock")
        return stamp

    @staticmethod
    def _capture(service, is_current):
        if not is_current():
            raise HistoryUnavailable("world_owner_changed")
        source, reason = service.capture_history_source()
        if source is None:
            raise HistoryUnavailable(reason)
        if type(source[0]) not in {ContractJournal, AsyncContractJournal}:
            raise HistoryUnavailable("persistence_disabled" if source[0] is None else "persistence_source_unavailable")
        return source

    @staticmethod
    def _health(source):
        journal = source[0]
        if type(journal) is AsyncContractJournal:
            health = journal.inspect_health()
            if health is None:
                raise HistoryUnavailable("persistence_busy")
        else:
            health = {"status": "closed" if journal._closed else "synchronous"}
        return WorldHistoryPersistence(**health, rejected_cohorts=source[3], error_reported=source[4])

    def read(self, service, *, limit=MAX_HISTORY_ROWS, cursor=None, is_current=lambda: True) -> bytes:
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_ROWS or cursor is not None and (
                not isinstance(cursor, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", cursor) is None):
            return encode_world_history(unknown_world_history("invalid_query"))
        if not self._readers.acquire(blocking=False):
            return encode_world_history(unknown_world_history("reader_limit_exceeded"))
        try:
            return self._read(service, limit, cursor, is_current)
        except HistoryUnavailable as exc:
            return encode_world_history(unknown_world_history(str(exc)))
        except (ContractJournalError, sqlite3.Error, ValueError, TypeError, OverflowError, RecursionError):
            return encode_world_history(unknown_world_history("journal_integrity_unavailable"))
        finally:
            self._readers.release()

    def _read(self, service, limit, cursor, is_current):
        started = time.monotonic(); deadline = started + READ_DEADLINE_SECONDS
        for token in tuple(self._cursors):
            if self._cursors[token].expires_monotonic <= started:
                del self._cursors[token]
        prior = self._cursors.get(cursor) if cursor else None
        if cursor and prior is None:
            raise HistoryUnavailable("inspection_expired")
        source = self._capture(service, is_current); at_us = self._clock(source)
        journal = source[0]; durable = journal.journal if type(journal) is AsyncContractJournal else journal
        if type(durable) is not ContractJournal:
            raise HistoryUnavailable("persistence_source_unavailable")
        if prior and (prior.service is not service or prior.journal is not journal or prior.durable is not durable
                      or prior.producer != source[1] or not prior.inspection.created_at_us <= at_us < prior.inspection.expires_at_us):
            raise HistoryUnavailable("inspection_expired")
        health = self._health(source)
        if health.status == "closed":
            raise HistoryUnavailable("persistence_closed")
        try:
            records, eligible, next_sequence, floor_previous = durable.inspect_world_events(limit=limit,
                before=prior.before if prior else None, deadline=deadline, max_rows=MAX_SCAN_ROWS, max_bytes=MAX_SCAN_BYTES,
                anchor_sequence=prior.inspection.anchor_sequence if prior else None)
        except ContractJournalError as exc:
            reason = str(exc)
            if reason in {"history_deadline_exceeded", "history_scan_bound_exceeded", "history_journal_closed", "inspection_expired"}:
                raise HistoryUnavailable(reason) from exc
            raise
        except sqlite3.OperationalError as exc:
            raise HistoryUnavailable("history_deadline_exceeded" if time.monotonic() >= deadline else "journal_busy_or_unavailable") from exc
        first = records[0].sequence if records else next_sequence
        anchor_sequence = records[-1].sequence if records else next_sequence - 1
        anchor_hash = records[-1].record_sha256 if records else floor_previous
        if prior:
            inspection = prior.inspection
            anchor = next((r for r in records if r.sequence == inspection.anchor_sequence), None)
            if (first != inspection.retained_first_sequence or floor_previous != inspection.retained_first_previous_sha256
                    or anchor is None or anchor.record_sha256 != inspection.anchor_sha256):
                raise HistoryUnavailable("inspection_expired")
        else:
            inspection = WorldHistoryInspection(anchor_sequence=anchor_sequence, anchor_sha256=anchor_hash,
                retained_first_sequence=first, retained_first_previous_sha256=floor_previous,
                created_at_us=at_us, expires_at_us=at_us + HISTORY_LEASE_US)
        indexed = {r.sequence: r for r in records}
        events = []
        for sequence in eligible[:limit]:
            record = indexed[sequence]
            if sequence > inspection.anchor_sequence:
                raise HistoryUnavailable("inspection_anchor_mismatch")
            events.append(WorldHistoryRecord(sequence=sequence, recorded_at_us=record.recorded_at_us,
                previous_sha256=record.previous_sha256, record_sha256=record.record_sha256, payload=dict(record.payload)))
        final = self._capture(service, is_current)
        if final[0] is not journal or final[1] != source[1] or final[2] is not source[2] or (
                type(journal) is AsyncContractJournal and journal.journal is not durable):
            raise HistoryUnavailable("world_source_changed")
        final_health = self._health(final)
        if final_health.status == "closed":
            raise HistoryUnavailable("persistence_closed")
        # Report the less complete of both request-side observations, including
        # dequeued in-flight batches; a queue drain cannot erase initial partiality.
        health = WorldHistoryPersistence(status="failed" if "failed" in {health.status, final_health.status} else final_health.status,
            pending_batches=max(health.pending_batches or 0, final_health.pending_batches or 0) if type(journal) is AsyncContractJournal else None,
            inflight_batches=max(health.inflight_batches or 0, final_health.inflight_batches or 0) if type(journal) is AsyncContractJournal else None,
            worker_alive=final_health.worker_alive, rejected_cohorts=max(health.rejected_cohorts, final_health.rejected_cohorts),
            error_reported=health.error_reported or final_health.error_reported)
        partial = (health.status == "failed" or health.worker_alive is False or health.error_reported
                   or health.rejected_cohorts or health.pending_batches or health.inflight_batches)
        next_token = None
        if len(eligible) > limit:
            if len(self._cursors) >= MAX_INSPECTIONS:
                raise HistoryUnavailable("inspection_capacity_exceeded")
            next_token = secrets.token_urlsafe(32)
            next_state = _Cursor(service, journal, durable, source[1], inspection,
                prior.expires_monotonic if prior else started + HISTORY_LEASE_US / 1_000_000, events[-1].sequence)
        result = WorldHistoryRead(status="partial" if partial else "available",
            reason="persistence_incomplete" if partial else "persisted_events_as_of_inspection", inspection=inspection,
            persistence=health, events=tuple(events), next_cursor=next_token)
        body = encode_world_history(result)
        final_us = self._clock(final)
        if not inspection.created_at_us <= final_us < inspection.expires_at_us or time.monotonic() >= deadline:
            raise HistoryUnavailable("history_deadline_exceeded")
        # Guard after response materialization as well; no data from a retired
        # service/journal may escape merely because its SQL read completed.
        guard = self._capture(service, is_current)
        if guard[0] is not journal or guard[1] != source[1] or guard[2] is not source[2] or (
                type(journal) is AsyncContractJournal and journal.journal is not durable):
            raise HistoryUnavailable("world_source_changed")
        durable._validate_main_binding()
        if durable._closed or type(journal) is AsyncContractJournal and journal._closed:
            raise HistoryUnavailable("persistence_closed")
        if next_token:
            self._cursors[next_token] = next_state
        return body


world_history_reader = WorldHistoryReader()
