"""Reviewed model annotations using the canonical scene SQLite transaction.

The authored artifact supplies identity and its own coordinates, never a new
world transform. Original lookup is independent of current scene/semantic state.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path, PurePosixPath

from .contracts.semantic_anchor import AnchorArtifact, AnchorEligibility, AnchorRecord, AnchorResult, AnchorReview, AnchorSnapshot, AnchorWrite
from .scene_store import SceneReleaseConflict, SceneReleaseStore, SceneReleaseStoreError
from .semantic_space_store import SemanticSpaceStore, canonical

MAX_BYTES = 256 * 1024
REPLAY_US = 24 * 60 * 60 * 1_000_000
MAX_OPERATIONS = 128
MAX_STORED_BYTES = 32 * 1024 * 1024


class AnchorLookupMissing(SceneReleaseStoreError):
    pass


class AnchorLookupExpired(SceneReleaseStoreError):
    pass


class SemanticAnchorStore:
    def __init__(self, scenes: SceneReleaseStore | None = None, *, database_path: Path | None = None, now_us=None):
        self.scenes = scenes
        self.semantic = SemanticSpaceStore(scenes, database_path=database_path)
        self.database_path = self.semantic.database_path
        self.now_us = now_us or (lambda: time.time_ns() // 1000)

    @staticmethod
    def _eligibility(artifact):
        supported = PurePosixPath(artifact.relative_path).suffix.lower() == ".obj"
        return AnchorEligibility(status="supported" if supported else "unsupported",
            reason="authored_obj_contract" if supported else "unsupported_authored_scene_format")

    def _source(self, connection):
        binding = self.semantic._binding(connection)
        row = connection.execute("SELECT payload_json,payload_sha256 FROM releases WHERE release_id=?", (binding.release_id,)).fetchone()
        release = SceneReleaseStore._release_from_row(row)
        artifact = AnchorArtifact(role=release.authored_scene.role, relative_path=release.authored_scene_path,
            sha256=release.authored_scene.sha256, size_bytes=release.authored_scene_size_bytes)
        return binding, artifact, self.semantic._record(connection), release

    def _record(self, connection):
        if not self.semantic._has_table(connection, "semantic_anchor_state"):
            return AnchorRecord(revision=None, updated_at_us=None, actor_id=None, anchors=())
        row = connection.execute("SELECT payload_json FROM semantic_anchor_state WHERE singleton=1").fetchone()
        if row is None:
            return AnchorRecord(revision=None, updated_at_us=None, actor_id=None, anchors=())
        if not isinstance(row[0], str) or len(row[0].encode("utf-8")) > MAX_BYTES:
            raise SceneReleaseStoreError("stored anchor registry exceeds its byte limit")
        record = AnchorRecord.model_validate_json(row[0])
        if canonical(record.model_dump(mode="json")) != row[0]:
            raise SceneReleaseStoreError("stored anchor registry is not canonical")
        return record

    @staticmethod
    def _reviews(record, binding, artifact, semantic):
        by_id = {item.id: item for item in semantic.objects}
        rows = []
        for anchor in record.anchors:
            item = by_id.get(anchor.semantic_id)
            status = ("retired" if anchor.retired else
                "stale_scene" if anchor.binding != binding or anchor.artifact != artifact else
                "semantic_missing" if item is None else "semantic_retired" if item.retired else
                "semantic_unreviewed" if item.reviewed_binding != binding else
                "stale_semantic" if anchor.semantic_revision != semantic.revision else "reviewed")
            rows.append(AnchorReview(id=anchor.id, status=status))
        return tuple(rows)

    @staticmethod
    def _fingerprint(command):
        return hashlib.sha256(canonical(command.model_dump(mode="json")).encode("utf-8")).hexdigest()

    @staticmethod
    def _command(command):
        command = AnchorWrite.model_validate(command.model_dump(mode="json") if isinstance(command, AnchorWrite) else command)
        if len(canonical(command.model_dump(mode="json")).encode("utf-8")) > MAX_BYTES:
            raise SceneReleaseStoreError("anchor command exceeds its byte limit")
        return command

    def _lookup(self, connection, command, now):
        if not self.semantic._has_table(connection, "semantic_anchor_operations"):
            return None
        row = connection.execute("SELECT * FROM semantic_anchor_operations WHERE actor_id=? AND operation_id=?", (command.actor_id, command.operation_id)).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != self._fingerprint(command):
            raise SceneReleaseConflict("original anchor operation fingerprint differs")
        if type(row["expires_at_us"]) is not int or row["expires_at_us"] < 1:
            raise SceneReleaseStoreError("stored anchor operation expiry is invalid")
        if row["expires_at_us"] <= now:
            raise AnchorLookupExpired("original anchor operation retention expired")
        raw = row["result_json"]
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_BYTES:
            raise SceneReleaseStoreError("stored anchor operation exceeds its byte limit")
        result = AnchorResult.model_validate_json(raw)
        expected_reviews = tuple(AnchorReview(id=item.id, status="retired" if item.retired else "reviewed") for item in command.anchors)
        if (canonical(result.model_dump(mode="json")) != raw or result.replayed
            or result.operation_id != command.operation_id or result.binding != command.expected_binding
            or result.artifact != command.expected_artifact or result.semantic_revision != command.expected_semantic_revision
            or result.record.actor_id != command.actor_id or result.record.anchors != command.anchors
            or result.record.revision is None or result.review_states != expected_reviews):
            raise SceneReleaseStoreError("stored anchor operation does not match its original")
        return result.model_copy(update={"replayed": True})

    def read(self):
        with self.semantic._read_connection() as connection:
            if connection is None:
                raise SceneReleaseConflict("no scene release is promoted")
            connection.execute("BEGIN")
            binding, artifact, semantic, _ = self._source(connection)
            record = self._record(connection)
            result = AnchorSnapshot(binding=binding, artifact=artifact, semantic_revision=semantic.revision, record=record,
                eligibility=self._eligibility(artifact), status="unknown" if record.revision is None else "available",
                review_states=self._reviews(record, binding, artifact, semantic))
            if len(canonical(result.model_dump(mode="json")).encode("utf-8")) > MAX_BYTES:
                raise SceneReleaseStoreError("anchor response exceeds its byte limit")
            connection.execute("COMMIT")
            return result

    def lookup(self, command):
        command = self._command(command)
        with self.semantic._read_connection() as connection:
            result = self._lookup(connection, command, self.now_us())
        if result is None:
            raise AnchorLookupMissing("original anchor operation is unavailable; no write was attempted")
        return result

    def replace(self, command):
        command = self._command(command)
        if self.scenes is None:
            raise SceneReleaseStoreError("anchor mutation requires the canonical scene store")
        now = self.now_us()
        with self.scenes._connection() as connection:
            connection.execute("PRAGMA busy_timeout=2000")
            connection.execute("BEGIN IMMEDIATE")
            original = self._lookup(connection, command, now)
            if original is not None:
                connection.execute("COMMIT")
                return original
            binding, artifact, semantic, release = self._source(connection)
            previous = self._record(connection)
            if binding != command.expected_binding or artifact != command.expected_artifact:
                raise SceneReleaseConflict("scene artifact binding changed; reload before reviewing")
            declares_geometry = any(not anchor.retired for anchor in command.anchors)
            if declares_geometry and self._eligibility(artifact).status != "supported":
                raise SceneReleaseConflict("semantic anchor geometry unsupported: current authored artifact is not an admitted OBJ geometry source")
            if semantic.revision != command.expected_semantic_revision:
                raise SceneReleaseConflict("semantic revision changed; reload before reviewing")
            if previous.revision != command.expected_revision:
                raise SceneReleaseConflict("anchor revision changed; reload before reviewing")
            by_id = {item.id: item for item in command.anchors}
            for old in previous.anchors:
                item = by_id.get(old.id)
                if item is None or item.semantic_id != old.semantic_id or old.retired and not item.retired:
                    raise SceneReleaseConflict("stable anchor IDs retain semantic identity and retirement")
            semantic_ids = {item.id: item for item in semantic.objects}
            old_by_id = {item.id: item for item in previous.anchors}
            for anchor in command.anchors:
                # Retirement preserves the exact original declaration even when
                # its scene/semantic source is no longer current or active. The
                # top-level current-source and registry CAS checks still apply.
                if anchor.retired:
                    old = old_by_id.get(anchor.id)
                    if old is not None and canonical(anchor.model_dump(mode="json")) == canonical(old.model_copy(update={"retired": True}).model_dump(mode="json")):
                        continue
                    raise SceneReleaseConflict("retirement must preserve an existing anchor's exact original declaration")
                item = semantic_ids.get(anchor.semantic_id)
                if (anchor.binding != binding or anchor.artifact != artifact or anchor.semantic_revision != semantic.revision
                    or item is None or item.retired or item.reviewed_binding != binding):
                    raise SceneReleaseConflict("anchor requires the exact current reviewed active semantic object and artifact")
            # Verify only the selected authored artifact, never a substitute or a
            # full-model scan. This API does not parse or change its geometry.
            if declares_geometry:
                verified = self.scenes.read_authored_scene(release)
                if not re.search(rb"(?m)^\s*v[ \t]+[^\r\n]+$", verified.data) or not re.search(rb"(?m)^\s*f[ \t]+[^\r\n]+$", verified.data):
                    raise SceneReleaseConflict("semantic anchor geometry unsupported: verified authored OBJ has no declared vertex/face geometry")
            record = AnchorRecord(revision=str(uuid.uuid4()), updated_at_us=now, actor_id=command.actor_id, anchors=command.anchors)
            result = AnchorResult(operation_id=command.operation_id, binding=binding, artifact=artifact,
                semantic_revision=semantic.revision, record=record, replayed=False,
                review_states=self._reviews(record, binding, artifact, semantic))
            payload, raw = canonical(record.model_dump(mode="json")), canonical(result.model_dump(mode="json"))
            if len(raw.encode("utf-8")) > MAX_BYTES:
                raise SceneReleaseStoreError("anchor response exceeds its byte limit")
            connection.execute("CREATE TABLE IF NOT EXISTS semantic_anchor_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload_json TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS semantic_anchor_operations (actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, fingerprint TEXT NOT NULL, expires_at_us INTEGER NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY(actor_id,operation_id))")
            connection.execute("DELETE FROM semantic_anchor_operations WHERE expires_at_us<=?", (now,))
            count, size = connection.execute("SELECT count(*),coalesce(sum(length(CAST(result_json AS BLOB))),0) FROM semantic_anchor_operations").fetchone()
            if count >= MAX_OPERATIONS or size + len(raw.encode("utf-8")) + len(payload.encode("utf-8")) > MAX_STORED_BYTES:
                raise SceneReleaseConflict("anchor operation retention is full")
            connection.execute("INSERT OR REPLACE INTO semantic_anchor_state VALUES(1,?)", (payload,))
            connection.execute("INSERT INTO semantic_anchor_operations VALUES(?,?,?,?,?)", (command.actor_id, command.operation_id, self._fingerprint(command), now + REPLAY_US, raw))
            if self._record(connection) != record or self._lookup(connection, command, now) != result.model_copy(update={"replayed": True}):
                raise SceneReleaseStoreError("anchor exact write readback failed")
            connection.execute("COMMIT")
            return result
