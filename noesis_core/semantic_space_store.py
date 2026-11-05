"""Stable semantic IDs share the scene selector's SQLite transaction boundary.

There are deliberately no coordinates, body measurements, room assignments,
provider identities or live spatial inferences in this registry.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts.semantic_space import SpaceBinding, SpaceRecord, SpaceResult, SpaceReview, SpaceSnapshot, SpaceWrite
from .private_paths import validate_private_file
from .scene_store import SceneReleaseConflict, SceneReleaseStore, SceneReleaseStoreError

MAX_BYTES = 256 * 1024
REPLAY_US = 24 * 60 * 60 * 1_000_000
MAX_OPERATIONS = 128
MAX_STORED_BYTES = 32 * 1024 * 1024


class SpaceLookupMissing(SceneReleaseStoreError):
    pass


class SpaceLookupExpired(SceneReleaseStoreError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class SemanticSpaceStore:
    def __init__(self, scenes: SceneReleaseStore | None = None, *, database_path: Path | None = None, now_us=None):
        self.scenes = scenes
        if scenes is None and database_path is None:
            raise ValueError("an explicit scene store or database path is required")
        self.database_path = scenes.database_path if scenes is not None else Path(database_path).expanduser().absolute()
        self.now_us = now_us or (lambda: time.time_ns() // 1000)

    @contextmanager
    def _read_connection(self):
        # In particular, original-operation recovery must not initialize a store,
        # prune expired rows, update pragmas, or create a database on a miss.
        if not self.database_path.exists() and not self.database_path.is_symlink():
            yield None
            return
        path = validate_private_file(self.database_path, label="scene release store")
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            yield connection
        finally:
            connection.close()
            validate_private_file(self.database_path, label="scene release store")

    @staticmethod
    def _has_table(connection, name):
        return connection is not None and connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    @staticmethod
    def _binding(connection):
        if connection is None:
            raise SceneReleaseConflict("no scene release is promoted")
        row = connection.execute("SELECT r.payload_json,r.payload_sha256 FROM singleton_state s LEFT JOIN releases r ON r.release_id=s.current_release_id WHERE s.singleton=1").fetchone()
        if not row or row["payload_json"] is None:
            raise SceneReleaseConflict("no scene release is promoted")
        release = SceneReleaseStore._release_from_row(row)
        return SpaceBinding(scope="scene_release", release_id=release.release_id,
            release_sha256=row["payload_sha256"], authored_scene_sha256=release.authored_scene.sha256,
            calibration_sha256=release.calibration.sha256, model_sha256=release.model.sha256,
            config_sha256=release.config.sha256)

    def _record(self, connection):
        if not self._has_table(connection, "semantic_space_state"):
            return SpaceRecord(revision=None, updated_at_us=None, actor_id=None, objects=())
        row = connection.execute("SELECT payload_json FROM semantic_space_state WHERE singleton=1").fetchone()
        if row is None:
            return SpaceRecord(revision=None, updated_at_us=None, actor_id=None, objects=())
        if not isinstance(row[0], str) or len(row[0].encode("utf-8")) > MAX_BYTES:
            raise SceneReleaseStoreError("stored semantic registry exceeds its byte limit")
        record = SpaceRecord.model_validate_json(row[0])
        if canonical(record.model_dump(mode="json")) != row[0]:
            raise SceneReleaseStoreError("stored semantic registry is not canonical")
        return record

    @staticmethod
    def _reviews(record, binding):
        return tuple(SpaceReview(id=item.id, status=("retired" if item.retired else
            "unreviewed" if item.reviewed_binding is None else
            "reviewed" if item.reviewed_binding == binding else "stale")) for item in record.objects)

    @staticmethod
    def _fingerprint(command):
        return hashlib.sha256(canonical(command.model_dump(mode="json")).encode()).hexdigest()

    def _lookup(self, connection, command, now):
        if not self._has_table(connection, "semantic_space_operations"):
            return None
        row = connection.execute("SELECT * FROM semantic_space_operations WHERE actor_id=? AND operation_id=?", (command.actor_id, command.operation_id)).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != self._fingerprint(command):
            raise SceneReleaseConflict("original operation fingerprint differs")
        if type(row["expires_at_us"]) is not int or row["expires_at_us"] < 1:
            raise SceneReleaseStoreError("stored semantic operation expiry is invalid")
        if row["expires_at_us"] <= now:
            raise SpaceLookupExpired("original operation retention expired")
        if not isinstance(row["result_json"], str) or len(row["result_json"].encode("utf-8")) > MAX_BYTES:
            raise SceneReleaseStoreError("stored semantic operation exceeds its byte limit")
        result = SpaceResult.model_validate_json(row["result_json"])
        if (canonical(result.model_dump(mode="json")) != row["result_json"] or result.replayed
            or result.operation_id != command.operation_id or result.binding != command.expected_binding
            or result.record.actor_id != command.actor_id or result.record.objects != command.objects
            or result.record.revision is None or result.review_states != self._reviews(result.record, result.binding)):
            raise SceneReleaseStoreError("stored semantic operation does not match its original")
        return result.model_copy(update={"replayed": True})

    def read(self):
        with self._read_connection() as connection:
            if connection is None:
                raise SceneReleaseConflict("no scene release is promoted")
            connection.execute("BEGIN")
            binding, record = self._binding(connection), self._record(connection)
            result = SpaceSnapshot(binding=binding, record=record,
                status="unknown" if record.revision is None else "available",
                review_states=self._reviews(record, binding))
            connection.execute("COMMIT")
            return result

    def lookup(self, command: SpaceWrite):
        with self._read_connection() as connection:
            result = self._lookup(connection, command, self.now_us())
        if result is None:
            raise SpaceLookupMissing("original operation is unavailable; no write was attempted")
        return result

    def replace(self, command: SpaceWrite):
        if self.scenes is None:
            raise SceneReleaseStoreError("semantic mutation requires the canonical scene store")
        # Fail malformed or oversized direct callers before opening a transaction.
        command = SpaceWrite.model_validate(command)
        if len(canonical(command.model_dump(mode="json")).encode("utf-8")) > MAX_BYTES:
            raise SceneReleaseStoreError("semantic command exceeds its byte limit")
        now = self.now_us()
        with self.scenes._connection() as connection:
            connection.execute("PRAGMA busy_timeout=2000")
            connection.execute("BEGIN IMMEDIATE")
            original = self._lookup(connection, command, now)
            if original is not None:
                connection.execute("COMMIT")
                return original
            binding, previous = self._binding(connection), self._record(connection)
            if binding != command.expected_binding:
                raise SceneReleaseConflict("scene binding changed; reload before reviewing")
            if previous.revision != command.expected_revision:
                raise SceneReleaseConflict("semantic revision changed; reload before reviewing")
            by_id = {item.id: item for item in command.objects}
            for old in previous.objects:
                new = by_id.get(old.id)
                if new is None or new.kind != old.kind or (old.retired and not new.retired):
                    raise SceneReleaseConflict("stable IDs must be retained with immutable kind and retirement")
            old_by_id = {item.id: item for item in previous.objects}
            for item in command.objects:
                if item != old_by_id.get(item.id) and not item.retired and item.reviewed_binding not in (None, binding):
                    raise SceneReleaseConflict("reviewed semantic label uses another scene binding")
            record = SpaceRecord(revision=str(uuid.uuid4()), updated_at_us=now, actor_id=command.actor_id, objects=command.objects)
            result = SpaceResult(operation_id=command.operation_id, record=record, binding=binding,
                replayed=False, review_states=self._reviews(record, binding))
            payload, result_json = canonical(record.model_dump(mode="json")), canonical(result.model_dump(mode="json"))
            if len(result_json.encode()) > MAX_BYTES:
                raise SceneReleaseStoreError("semantic space exceeds the response byte limit")
            connection.execute("CREATE TABLE IF NOT EXISTS semantic_space_state (singleton INTEGER PRIMARY KEY CHECK (singleton=1), payload_json TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS semantic_space_operations (actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, fingerprint TEXT NOT NULL, expires_at_us INTEGER NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY(actor_id,operation_id))")
            connection.execute("DELETE FROM semantic_space_operations WHERE expires_at_us<=?", (now,))
            count, size = connection.execute("SELECT count(*),coalesce(sum(length(CAST(result_json AS BLOB))),0) FROM semantic_space_operations").fetchone()
            if count >= MAX_OPERATIONS or size + len(result_json.encode()) + len(payload.encode()) > MAX_STORED_BYTES:
                raise SceneReleaseConflict("semantic operation retention is full; try after original operations expire")
            connection.execute("INSERT OR REPLACE INTO semantic_space_state VALUES(1,?)", (payload,))
            connection.execute("INSERT INTO semantic_space_operations VALUES(?,?,?,?,?)", (command.actor_id, command.operation_id, self._fingerprint(command), now + REPLAY_US, result_json))
            connection.execute("COMMIT")
            return result
