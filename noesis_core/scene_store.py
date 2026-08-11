from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import ValidationError

from noesis_core.contracts.scene import SceneArtifact, SceneCameraRevision, SceneRelease
from noesis_core.private_paths import (
    PrivatePathError,
    prepare_private_writable_file,
    validate_private_file,
)
from noesis_core.scene_files import (
    MAX_AUTHORED_SCENE_BYTES,
    MAX_SCENE_ARTIFACT_BYTES,
    MAX_SCENE_MANIFEST_BYTES,
    MAX_SCENE_VALIDATION_BYTES,
    SceneFileError,
    VerifiedSceneFile,
    absolute_path_without_resolving,
    load_strict_json,
    read_scene_root_file,
)


class SceneReleaseStoreError(RuntimeError):
    pass


class SceneReleaseConflict(SceneReleaseStoreError):
    pass


@dataclass(frozen=True)
class ScenePromotion:
    sequence: int
    event: Literal["promote", "rollback"]
    release_id: str
    previous_release_id: str | None
    actor_id: str
    occurred_at_us: int
    release_sha256: str


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


MAX_SCENE_RELEASE_JSON_BYTES = 5 * 1024 * 1024


class SceneReleaseStore:
    """Immutable release registry with atomic promotion and rollback history."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        database_path: str | Path,
        *,
        artifact_root: str | Path | None = None,
        bundle_root: str | Path | None = None,
    ) -> None:
        configured_database_path = Path(database_path).expanduser()
        try:
            self.database_path = prepare_private_writable_file(
                configured_database_path,
                label="scene release store",
            )
        except PrivatePathError as exc:
            raise SceneReleaseStoreError(str(exc)) from exc
        self.artifact_root = (
            absolute_path_without_resolving(artifact_root)
            if artifact_root is not None
            else None
        )
        self.bundle_root = (
            absolute_path_without_resolving(bundle_root)
            if bundle_root is not None
            else None
        )
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()
            try:
                validate_private_file(
                    self.database_path,
                    label="scene release store",
                )
            except PrivatePathError as exc:
                raise SceneReleaseStoreError(str(exc)) from exc

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, self.SCHEMA_VERSION}:
                raise SceneReleaseStoreError(
                    f"unsupported scene store schema: {version}; expected {self.SCHEMA_VERSION}"
                )
            if version == 0:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE releases (
                        release_id TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        payload_sha256 TEXT NOT NULL UNIQUE,
                        created_at_us INTEGER NOT NULL,
                        created_by TEXT NOT NULL
                    );
                    CREATE TABLE singleton_state (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        current_release_id TEXT REFERENCES releases(release_id)
                    );
                    INSERT INTO singleton_state(singleton, current_release_id) VALUES (1, NULL);
                    CREATE TABLE promotion_history (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        event TEXT NOT NULL CHECK (event IN ('promote', 'rollback')),
                        release_id TEXT NOT NULL REFERENCES releases(release_id),
                        previous_release_id TEXT REFERENCES releases(release_id),
                        actor_id TEXT NOT NULL,
                        occurred_at_us INTEGER NOT NULL,
                        release_sha256 TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                )
        try:
            validate_private_file(
                self.database_path,
                label="scene release store",
            )
        except PrivatePathError as exc:
            raise SceneReleaseStoreError(str(exc)) from exc

    def register(self, release: SceneRelease) -> str:
        validated = SceneRelease.model_validate(release)
        if self.artifact_root is not None or self.bundle_root is not None:
            self.validate_artifacts(validated)
        payload = validated.model_dump(mode="json")
        payload_json = _canonical_json(payload)
        if len(payload_json.encode("utf-8")) > MAX_SCENE_RELEASE_JSON_BYTES:
            raise SceneReleaseStoreError(
                f"scene release exceeds the {MAX_SCENE_RELEASE_JSON_BYTES}-byte JSON limit"
            )
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json, payload_sha256 FROM releases WHERE release_id = ?",
                (validated.release_id,),
            ).fetchone()
            if existing is not None:
                connection.execute("ROLLBACK")
                self._release_from_row(existing)
                if str(existing["payload_sha256"]) != payload_sha256:
                    raise SceneReleaseConflict(
                        f"release_id {validated.release_id!r} already names different immutable content"
                    )
                return payload_sha256
            connection.execute(
                """
                INSERT INTO releases(release_id, payload_json, payload_sha256, created_at_us, created_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    validated.release_id,
                    payload_json,
                    payload_sha256,
                    validated.created_at_us,
                    validated.created_by,
                ),
            )
            connection.execute("COMMIT")
        return payload_sha256

    def get(self, release_id: str) -> SceneRelease | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json, payload_sha256 FROM releases WHERE release_id = ?",
                (str(release_id),),
            ).fetchone()
        if row is None:
            return None
        return self._release_from_row(row)

    def current(self) -> SceneRelease | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                """
                SELECT r.payload_json, r.payload_sha256
                FROM singleton_state AS s
                LEFT JOIN releases AS r ON r.release_id = s.current_release_id
                WHERE s.singleton = 1
                """
            ).fetchone()
        if row is None or row["payload_json"] is None:
            return None
        return self._release_from_row(row)

    def promote(
        self,
        release_id: str,
        *,
        actor_id: str,
        occurred_at_us: int,
        expected_current_release_id: str | None,
        event: Literal["promote", "rollback"] = "promote",
    ) -> ScenePromotion:
        actor = str(actor_id).strip()
        if not actor:
            raise SceneReleaseStoreError("actor_id is required")
        if int(occurred_at_us) <= 0:
            raise SceneReleaseStoreError("occurred_at_us must be positive")
        release = self.get(str(release_id))
        if release is None:
            raise SceneReleaseStoreError(f"unknown scene release: {release_id}")
        with self._lock:
            self.validate_artifacts(release)
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                state = connection.execute(
                    "SELECT current_release_id FROM singleton_state WHERE singleton = 1"
                ).fetchone()
                current_id = str(state["current_release_id"]) if state and state["current_release_id"] else None
                if current_id != expected_current_release_id:
                    connection.execute("ROLLBACK")
                    raise SceneReleaseConflict(
                        f"scene promotion compare-and-swap failed: expected={expected_current_release_id!r} current={current_id!r}"
                    )
                row = connection.execute(
                    "SELECT payload_json, payload_sha256 FROM releases WHERE release_id = ?",
                    (release.release_id,),
                ).fetchone()
                if row is None:
                    connection.execute("ROLLBACK")
                    raise SceneReleaseStoreError(f"scene release disappeared during promotion: {release.release_id}")
                if self._release_from_row(row) != release:
                    connection.execute("ROLLBACK")
                    raise SceneReleaseStoreError(
                        f"scene release changed during promotion: {release.release_id}"
                    )
                release_sha256 = str(row["payload_sha256"])
                connection.execute(
                    "UPDATE singleton_state SET current_release_id = ? WHERE singleton = 1",
                    (release.release_id,),
                )
                cursor = connection.execute(
                    """
                    INSERT INTO promotion_history(
                        event, release_id, previous_release_id, actor_id, occurred_at_us, release_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (event, release.release_id, current_id, actor, int(occurred_at_us), release_sha256),
                )
                sequence = int(cursor.lastrowid)
                connection.execute("COMMIT")
        return ScenePromotion(
            sequence=sequence,
            event=event,
            release_id=release.release_id,
            previous_release_id=current_id,
            actor_id=actor,
            occurred_at_us=int(occurred_at_us),
            release_sha256=release_sha256,
        )

    def rollback(
        self,
        release_id: str,
        *,
        actor_id: str,
        occurred_at_us: int,
        expected_current_release_id: str,
    ) -> ScenePromotion:
        return self.promote(
            release_id,
            actor_id=actor_id,
            occurred_at_us=occurred_at_us,
            expected_current_release_id=expected_current_release_id,
            event="rollback",
        )

    def history(self) -> tuple[ScenePromotion, ...]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event, release_id, previous_release_id, actor_id, occurred_at_us, release_sha256
                FROM promotion_history ORDER BY sequence ASC
                """
            ).fetchall()
        return tuple(
            ScenePromotion(
                sequence=int(row["sequence"]),
                event=str(row["event"]),  # type: ignore[arg-type]
                release_id=str(row["release_id"]),
                previous_release_id=(
                    str(row["previous_release_id"]) if row["previous_release_id"] else None
                ),
                actor_id=str(row["actor_id"]),
                occurred_at_us=int(row["occurred_at_us"]),
                release_sha256=str(row["release_sha256"]),
            )
            for row in rows
        )

    def list_releases(self) -> tuple[SceneRelease, ...]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT payload_json, payload_sha256 FROM releases "
                "ORDER BY created_at_us DESC, release_id ASC"
            ).fetchall()
        return tuple(self._release_from_row(row) for row in rows)

    @staticmethod
    def _release_from_row(row: sqlite3.Row) -> SceneRelease:
        payload_json = str(row["payload_json"])
        payload_bytes = payload_json.encode("utf-8")
        if len(payload_bytes) > MAX_SCENE_RELEASE_JSON_BYTES:
            raise SceneReleaseStoreError("stored scene release JSON exceeds its size limit")
        expected_sha256 = str(row["payload_sha256"])
        if hashlib.sha256(payload_bytes).hexdigest() != expected_sha256:
            raise SceneReleaseStoreError("stored scene release fingerprint mismatch")
        try:
            release = SceneRelease.model_validate_json(payload_json)
        except (ValidationError, ValueError) as exc:
            raise SceneReleaseStoreError("stored scene release is invalid") from exc
        if _canonical_json(release.model_dump(mode="json")) != payload_json:
            raise SceneReleaseStoreError("stored scene release is not canonical JSON")
        return release

    @staticmethod
    def _wrap_file_error(exc: SceneFileError) -> SceneReleaseStoreError:
        return SceneReleaseStoreError(str(exc))

    def read_authored_scene(self, release: SceneRelease) -> VerifiedSceneFile:
        if self.bundle_root is None:
            raise SceneReleaseStoreError("scene bundle root is not configured")
        try:
            return read_scene_root_file(
                self.bundle_root,
                release.authored_scene_path,
                label="authored scene",
                max_bytes=MAX_AUTHORED_SCENE_BYTES,
                expected_sha256=release.authored_scene.sha256,
                expected_size=release.authored_scene_size_bytes,
            )
        except SceneFileError as exc:
            raise self._wrap_file_error(exc) from exc

    def read_authored_dependency(
        self,
        dependency: SceneArtifact,
    ) -> VerifiedSceneFile:
        if self.bundle_root is None:
            raise SceneReleaseStoreError("scene bundle root is not configured")
        try:
            return read_scene_root_file(
                self.bundle_root,
                dependency.relative_path,
                label=f"authored scene dependency {dependency.role}",
                max_bytes=MAX_SCENE_ARTIFACT_BYTES,
                expected_sha256=dependency.sha256,
                expected_size=dependency.size_bytes,
            )
        except SceneFileError as exc:
            raise self._wrap_file_error(exc) from exc

    def read_validation_report(self, release: SceneRelease) -> VerifiedSceneFile:
        if self.bundle_root is None:
            raise SceneReleaseStoreError("scene bundle root is not configured")
        try:
            return read_scene_root_file(
                self.bundle_root,
                release.validation_report_path,
                label="scene validation report",
                max_bytes=MAX_SCENE_VALIDATION_BYTES,
                expected_sha256=release.validation_report_sha256,
                expected_size=release.validation_report_size_bytes,
            )
        except SceneFileError as exc:
            raise self._wrap_file_error(exc) from exc

    def read_camera_manifest(self, camera: SceneCameraRevision) -> dict[str, Any]:
        if self.artifact_root is None:
            raise SceneReleaseStoreError("scene artifact root is not configured")
        manifest_relative = (
            PurePosixPath(camera.artifact_path) / "manifest.json"
        ).as_posix()
        try:
            manifest_file = read_scene_root_file(
                self.artifact_root,
                manifest_relative,
                label=f"scene manifest for camera {camera.camera_id}",
                max_bytes=MAX_SCENE_MANIFEST_BYTES,
                expected_sha256=camera.manifest_sha256,
            )
        except SceneFileError as exc:
            raise self._wrap_file_error(exc) from exc
        try:
            manifest = load_strict_json(
                manifest_file.data,
                label=f"scene manifest for camera {camera.camera_id}",
            )
        except SceneFileError as exc:
            raise SceneReleaseStoreError(
                f"scene manifest is invalid for camera {camera.camera_id}"
            ) from exc
        if not isinstance(manifest, dict):
            raise SceneReleaseStoreError(
                f"scene manifest root is invalid for camera {camera.camera_id}"
            )
        if str(manifest.get("revision_id") or "") != camera.revision_id:
            raise SceneReleaseStoreError(
                f"scene manifest revision mismatch for camera {camera.camera_id}"
            )
        if str(manifest.get("camera") or "") != camera.camera_id:
            raise SceneReleaseStoreError(
                f"scene manifest camera mismatch for camera {camera.camera_id}"
            )
        declared = manifest.get("artifacts")
        if not isinstance(declared, dict):
            raise SceneReleaseStoreError(
                f"scene manifest artifact inventory is invalid for camera {camera.camera_id}"
            )
        expected = {
            artifact.role: artifact.relative_path for artifact in camera.artifacts
        }
        normalized_declared = {
            str(role): str(relative_path)
            for role, relative_path in declared.items()
            if isinstance(relative_path, str)
        }
        if normalized_declared != expected:
            raise SceneReleaseStoreError(
                f"scene artifact inventory mismatch for camera {camera.camera_id}"
            )
        return manifest

    def read_camera_artifact(
        self,
        camera: SceneCameraRevision,
        artifact: SceneArtifact,
    ) -> VerifiedSceneFile:
        if self.artifact_root is None:
            raise SceneReleaseStoreError("scene artifact root is not configured")
        relative_path = (
            PurePosixPath(camera.artifact_path) / artifact.relative_path
        ).as_posix()
        try:
            return read_scene_root_file(
                self.artifact_root,
                relative_path,
                label=f"scene artifact {camera.camera_id}/{artifact.role}",
                max_bytes=MAX_SCENE_ARTIFACT_BYTES,
                expected_sha256=artifact.sha256,
                expected_size=artifact.size_bytes,
            )
        except SceneFileError as exc:
            raise self._wrap_file_error(exc) from exc

    def validate_artifacts(self, release: SceneRelease) -> None:
        if self.bundle_root is not None:
            self.read_authored_scene(release)
            for dependency in release.authored_scene_dependencies:
                self.read_authored_dependency(dependency)
            self.read_validation_report(release)

        if self.artifact_root is None:
            return
        for camera in release.cameras:
            self.read_camera_manifest(camera)
            for artifact in camera.artifacts:
                self.read_camera_artifact(camera, artifact)
