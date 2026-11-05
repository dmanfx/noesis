"""Bounded reviewed owner declarations; no detector identity or live matching."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts.object_associations import (
    AssociationCommand, AssociationConfirm, AssociationInspect, AssociationInspection,
    AssociationPrepare, AssociationRecord, AssociationResult, AssociationReview,
    AssociationSnapshot, ObjectAssociation, ObservationCapture,
    MAX_BYTES, MAX_ASSOCIATIONS, MAX_INSPECTIONS, MAX_INSPECTION_ROWS, MAX_REVIEWS,
    MAX_OPERATIONS, MAX_STORED_BYTES, REVIEW_US, REPLAY_US,
)
from .object_observations import ObjectObservationService
from .private_paths import prepare_private_writable_file, validate_private_file
from .semantic_space_store import SemanticSpaceStore, canonical


class AssociationError(RuntimeError):
    pass


class AssociationConflict(AssociationError):
    pass


class AssociationMissing(AssociationError):
    pass


class AssociationExpired(AssociationError):
    pass


class ObjectAssociationStore:
    def __init__(self, database_path: Path, *, now_us=None):
        self.database_path = Path(database_path).expanduser().absolute()
        self._reader = SemanticSpaceStore(database_path=self.database_path)
        self.now_us = now_us or (lambda: time.time_ns() // 1000)

    def _now(self):
        value = self.now_us()
        if type(value) is not int or not 0 < value <= 2**53 - 1:
            raise AssociationError("invalid association clock")
        return value

    def _remaining(self, result):
        remaining = (result.expires_at_us - self._now()) // 1000
        if not 0 < remaining <= 120000:
            raise AssociationExpired("original authoring lease expired")
        return result.model_copy(update={"remaining_ms": remaining})

    @staticmethod
    def _raw(model):
        raw = canonical(model.model_dump(mode="json"))
        if len(raw.encode("utf-8")) > MAX_BYTES:
            raise AssociationError("association payload exceeds byte limit")
        return raw

    @classmethod
    def _stored(cls, model, raw):
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_BYTES:
            raise AssociationError("invalid stored association bytes")
        result = model.model_validate_json(raw)
        if cls._raw(result) != raw:
            raise AssociationError("stored association is not canonical")
        return result

    @classmethod
    def _fingerprint(cls, command):
        return hashlib.sha256(cls._raw(command).encode("utf-8")).hexdigest()

    @contextmanager
    def _transaction(self):
        path = prepare_private_writable_file(self.database_path, label="object association store")
        connection = sqlite3.connect(path, timeout=2, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS object_association_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload_json TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS object_association_inspections (actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, source_token TEXT NOT NULL, created_at_us INTEGER NOT NULL, expires_at_us INTEGER NOT NULL, inspection_json TEXT NOT NULL, PRIMARY KEY(actor_id,operation_id))")
            connection.execute("CREATE TABLE IF NOT EXISTS object_association_reviews (actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, source_token TEXT NOT NULL, created_at_us INTEGER NOT NULL, expires_at_us INTEGER NOT NULL, review_json TEXT NOT NULL, PRIMARY KEY(actor_id,operation_id))")
            connection.execute("CREATE TABLE IF NOT EXISTS object_association_operations (actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, fingerprint TEXT NOT NULL, expires_at_us INTEGER NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY(actor_id,operation_id))")
            yield connection
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()
            validate_private_file(self.database_path, label="object association store")

    def _record(self, connection):
        if not self._reader._has_table(connection, "object_association_state"):
            return AssociationRecord(revision=None, updated_at_us=None, actor_id=None, associations=())
        row = connection.execute("SELECT payload_json FROM object_association_state WHERE singleton=1").fetchone()
        return self._stored(AssociationRecord, row[0]) if row else AssociationRecord(revision=None, updated_at_us=None, actor_id=None, associations=())

    def read(self):
        with self._reader._read_connection() as connection:
            record = self._record(connection)
        result = AssociationSnapshot(status="unknown" if record.revision is None else "available", record=record)
        self._raw(result)
        return result

    @staticmethod
    def _source(service, source_token, is_current, run_id=None):
        if (type(service) is not ObjectObservationService or service._closed or not isinstance(source_token, str)
            or not source_token or is_current() is not True or run_id is not None and service.run_id != run_id):
            raise AssociationConflict("object observation owner or run retired")

    @staticmethod
    def _lease(row, now, source_token):
        if row["source_token"] != source_token:
            raise AssociationConflict("original observation owner retired")
        if type(row["created_at_us"]) is not int or type(row["expires_at_us"]) is not int or row["expires_at_us"] != row["created_at_us"] + REVIEW_US:
            raise AssociationError("invalid stored authoring lease")
        if not row["created_at_us"] <= now < row["expires_at_us"]:
            raise AssociationExpired("original authoring lease expired")

    @staticmethod
    def _prune(connection, now):
        for name in ("inspections", "reviews", "operations"):
            connection.execute("DELETE FROM object_association_" + name + " WHERE expires_at_us<=?", (now,))

    @staticmethod
    def _capacity(connection, extra=0):
        size = connection.execute("SELECT coalesce(sum(length(CAST(payload_json AS BLOB))),0) FROM object_association_state").fetchone()[0]
        for name, column, limit in (("inspections", "inspection_json", MAX_INSPECTIONS), ("reviews", "review_json", MAX_REVIEWS), ("operations", "result_json", MAX_OPERATIONS)):
            count, used = connection.execute("SELECT count(*),coalesce(sum(length(CAST(" + column + " AS BLOB))),0) FROM object_association_" + name).fetchone()
            if count > limit:
                raise AssociationConflict("association retention is full")
            size += used
        if size + extra > MAX_STORED_BYTES:
            raise AssociationConflict("association byte retention is full")

    def _lookup(self, connection, command, now):
        if not self._reader._has_table(connection, "object_association_operations"):
            raise AssociationMissing("original association operation not found")
        row = connection.execute("SELECT * FROM object_association_operations WHERE actor_id=? AND operation_id=?", (command.actor_id, command.operation_id)).fetchone()
        if row is None:
            raise AssociationMissing("original association operation not found")
        if row["fingerprint"] != self._fingerprint(command):
            raise AssociationConflict("original association operation differs")
        if type(row["expires_at_us"]) is not int or row["expires_at_us"] <= now:
            raise AssociationExpired("original association operation expired")
        result = self._stored(AssociationResult, row["result_json"])
        exact = next((item for item in result.record.associations if item.id == command.association.id), None)
        if result.replayed or result.operation_id != command.operation_id or result.association_id != command.association.id or exact != command.association or result.record.actor_id != command.actor_id or result.record.revision is None:
            raise AssociationError("stored association original differs")
        return result.model_copy(update={"replayed": True})

    def lookup(self, command):
        command = AssociationCommand.model_validate(command.model_dump(mode="json") if isinstance(command, AssociationCommand) else command)
        self._raw(command)
        with self._reader._read_connection() as connection:
            return self._lookup(connection, command, self._now())

    def inspect(self, request, service, source_token, is_current):
        request = AssociationInspect.model_validate(request.model_dump(mode="json") if isinstance(request, AssociationInspect) else request)
        started = self._now()
        self._source(service, source_token, is_current)
        # Replay the server-issued inspection without recapturing or renewing it.
        with self._reader._read_connection() as connection:
            if self._reader._has_table(connection, "object_association_inspections"):
                previous = connection.execute("SELECT * FROM object_association_inspections WHERE actor_id=? AND operation_id=?", (request.actor_id, request.operation_id)).fetchone()
                if previous:
                    self._lease(previous, started, source_token)
                    inspection = self._stored(AssociationInspection, previous["inspection_json"])
                    self._source(service, source_token, is_current, inspection.run_id)
                    return self._remaining(inspection)
        snapshot = service.snapshot()
        if snapshot.get("run_id") != service.run_id or snapshot.get("status") not in {"available", "partial", "unknown"}:
            raise AssociationConflict("observation report unavailable")
        cameras = snapshot.get("cameras")
        if not isinstance(cameras, list) or len(cameras) > 16:
            raise AssociationError("observation camera bound exceeded")
        captures, capture_bytes, omitted, seen = [], 0, 0, set()
        for frame in cameras:
            rows = frame.get("observations")
            if not isinstance(rows, list) or len(rows) > 32:
                raise AssociationError("observation row bound exceeded")
            metadata = {key: value for key, value in frame.items() if key != "observations"}
            omitted += frame.get("omitted_count", 0) + frame.get("invalid_count", 0)
            for row in rows:
                capture = ObservationCapture(run_id=service.run_id, **metadata, observation=row)
                key = (capture.camera_id, capture.source_id, capture.source_epoch, capture.frame_id, capture.observation.observation_id)
                if key in seen or not capture.observed_at_us <= snapshot.get("generated_at_us", 0) < capture.valid_until_us:
                    raise AssociationConflict("observation capture is invalid or expired")
                seen.add(key)
                size = len(self._raw(capture).encode("utf-8"))
                if len(captures) >= MAX_INSPECTION_ROWS or capture_bytes + size > MAX_BYTES - 4096:
                    omitted += 1
                else:
                    captures.append(capture); capture_bytes += size
        with self._transaction() as connection:
            self._source(service, source_token, is_current, snapshot["run_id"])
            self._prune(connection, started)
            if connection.execute("SELECT 1 FROM object_association_operations WHERE actor_id=? AND operation_id=?", (request.actor_id, request.operation_id)).fetchone():
                raise AssociationConflict("operation ID already saved")
            if connection.execute("SELECT 1 FROM object_association_inspections WHERE actor_id=? AND operation_id=?", (request.actor_id, request.operation_id)).fetchone():
                raise AssociationConflict("inspection already issued")
            if connection.execute("SELECT count(*) FROM object_association_inspections").fetchone()[0] >= MAX_INSPECTIONS:
                raise AssociationConflict("association inspection retention is full")
            inspection = AssociationInspection(inspection_token=secrets.token_urlsafe(32), expires_at_us=started + REVIEW_US, remaining_ms=120000, run_id=snapshot["run_id"], registry_revision=self._record(connection).revision, captures=tuple(captures), omitted_count=omitted)
            raw = self._raw(inspection); self._capacity(connection, len(raw.encode("utf-8")))
            connection.execute("INSERT INTO object_association_inspections VALUES(?,?,?,?,?,?)", (request.actor_id, request.operation_id, source_token, started, inspection.expires_at_us, raw))
            self._source(service, source_token, is_current, inspection.run_id)
            if self._now() >= inspection.expires_at_us:
                raise AssociationExpired("inspection authoring lease expired")
            connection.execute("COMMIT")
            return self._remaining(inspection)

    def prepare(self, request, service, source_token, is_current):
        request = AssociationPrepare.model_validate(request.model_dump(mode="json") if isinstance(request, AssociationPrepare) else request)
        now = self._now(); self._source(service, source_token, is_current)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM object_association_inspections WHERE actor_id=? AND operation_id=?", (request.actor_id, request.operation_id)).fetchone()
            if row is None:
                raise AssociationMissing("server-issued inspection not found")
            self._lease(row, now, source_token)
            inspection = self._stored(AssociationInspection, row["inspection_json"])
            if not secrets.compare_digest(request.inspection_token, inspection.inspection_token):
                raise AssociationConflict("inspection token differs")
            self._source(service, source_token, is_current, inspection.run_id)
            record = self._record(connection)
            if request.expected_revision != inspection.registry_revision or record.revision != request.expected_revision:
                raise AssociationConflict("association registry revision changed")
            old = next((item for item in record.associations if item.id == request.association_id), None)
            if old:
                if request.observation is not None or old.retired and not request.retired:
                    raise AssociationConflict("stable declaration capture and retirement cannot be replaced")
                capture = old.capture
            else:
                if request.retired or request.observation is None or len(record.associations) >= MAX_ASSOCIATIONS:
                    raise AssociationConflict("new declaration requires one inspected observation")
                reference = request.observation
                matches = [item for item in inspection.captures if item.run_id == reference.run_id and item.camera_id == reference.camera_id
                    and item.source_id == reference.source_id and item.source_epoch == reference.source_epoch and item.frame_id == reference.frame_id and item.observation.observation_id == reference.observation_id]
                if len(matches) != 1:
                    raise AssociationConflict("exact inspected observation not found")
                capture = matches[0]
            association = ObjectAssociation(id=request.association_id, label=request.label, retired=request.retired, declaration="owner_declared", capture=capture)
            command = AssociationCommand(actor_id=request.actor_id, operation_id=request.operation_id, expected_revision=request.expected_revision, association=association)
            previous = connection.execute("SELECT * FROM object_association_reviews WHERE actor_id=? AND operation_id=?", (request.actor_id, request.operation_id)).fetchone()
            if previous:
                self._lease(previous, now, source_token)
                review = self._stored(AssociationReview, previous["review_json"])
                if review.command != command:
                    raise AssociationConflict("original review command differs")
                return self._remaining(review)
            self._prune(connection, now)
            if connection.execute("SELECT count(*) FROM object_association_reviews").fetchone()[0] >= MAX_REVIEWS:
                raise AssociationConflict("association review retention is full")
            review = AssociationReview(review_token=secrets.token_urlsafe(32), expires_at_us=inspection.expires_at_us, remaining_ms=(inspection.expires_at_us - now) // 1000, command=command)
            raw = self._raw(review); self._capacity(connection, len(raw.encode("utf-8")))
            connection.execute("INSERT INTO object_association_reviews VALUES(?,?,?,?,?,?)", (request.actor_id, request.operation_id, source_token, row["created_at_us"], review.expires_at_us, raw))
            self._source(service, source_token, is_current, inspection.run_id)
            if self._now() >= review.expires_at_us:
                raise AssociationExpired("original authoring lease expired")
            connection.execute("COMMIT")
            return self._remaining(review)

    def confirm(self, request, service, source_token, is_current):
        request = AssociationConfirm.model_validate(request.model_dump(mode="json") if isinstance(request, AssociationConfirm) else request)
        command = request.command
        try:
            return self.lookup(command)
        except AssociationMissing:
            pass
        now = self._now(); self._source(service, source_token, is_current)
        with self._transaction() as connection:
            try:
                return self._lookup(connection, command, now)
            except AssociationMissing:
                pass
            row = connection.execute("SELECT * FROM object_association_reviews WHERE actor_id=? AND operation_id=?", (command.actor_id, command.operation_id)).fetchone()
            if row is None:
                raise AssociationMissing("issued association review not found")
            self._lease(row, now, source_token)
            review = self._stored(AssociationReview, row["review_json"])
            if not secrets.compare_digest(review.review_token, request.review_token) or review.command != command:
                raise AssociationConflict("exact issued association review differs")
            self._source(service, source_token, is_current, review.command.association.capture.run_id if command.expected_revision is None else None)
            previous = self._record(connection)
            if previous.revision != command.expected_revision:
                raise AssociationConflict("association registry revision changed")
            by_id = {item.id: item for item in previous.associations}
            old = by_id.get(command.association.id)
            if old and (old.capture != command.association.capture or old.retired and not command.association.retired):
                raise AssociationConflict("stable declaration capture cannot be rebound")
            by_id[command.association.id] = command.association
            record = AssociationRecord(revision=str(uuid.uuid4()), updated_at_us=now, actor_id=command.actor_id, associations=tuple(by_id.values()))
            result = AssociationResult(operation_id=command.operation_id, association_id=command.association.id, record=record, replayed=False)
            raw, payload = self._raw(result), self._raw(record)
            self._prune(connection, now)
            if connection.execute("SELECT count(*) FROM object_association_operations").fetchone()[0] >= MAX_OPERATIONS:
                raise AssociationConflict("association original retention is full")
            self._capacity(connection, len(raw.encode("utf-8")) + len(payload.encode("utf-8")))
            connection.execute("INSERT OR REPLACE INTO object_association_state VALUES(1,?)", (payload,))
            connection.execute("INSERT INTO object_association_operations VALUES(?,?,?,?,?)", (command.actor_id, command.operation_id, self._fingerprint(command), now + REPLAY_US, raw))
            if self._record(connection) != record or self._lookup(connection, command, now) != result.model_copy(update={"replayed": True}):
                raise AssociationError("association exact original readback failed")
            self._source(service, source_token, is_current)
            if self._now() >= review.expires_at_us:
                raise AssociationExpired("original authoring lease expired")
            connection.execute("DELETE FROM object_association_reviews WHERE actor_id=? AND operation_id=?", (command.actor_id, command.operation_id))
            connection.execute("DELETE FROM object_association_inspections WHERE actor_id=? AND operation_id=?", (command.actor_id, command.operation_id))
            connection.execute("COMMIT")
            return result
