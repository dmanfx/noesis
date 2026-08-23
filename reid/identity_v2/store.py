"""Versioned local SQLite persistence for household identity v2."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sqlite3
import struct
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence, Tuple

from noesis_core.private_paths import (
    PrivatePathError,
    prepare_private_writable_file,
    validate_private_file,
)
from noesis_core.strict_json import strict_json_loads

SCHEMA_VERSION = 2
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
_ANCHOR_ROLE = "enrollment_anchor"
_QUARANTINE_ROLE = "quarantine"
_ADAPTIVE_ROLE = "adaptive"


class IdentityStoreError(RuntimeError):
    pass


class DuplicateResidentError(IdentityStoreError):
    pass


class ModelProfileMismatch(IdentityStoreError):
    pass


class VisitorSlotInUse(IdentityStoreError):
    pass


class MigrationConflict(IdentityStoreError):
    pass


class EnrollmentProposalConflict(IdentityStoreError):
    pass


class EnrollmentProposalExpired(IdentityStoreError):
    pass


@dataclass(frozen=True)
class ResidentRecord:
    resident_uuid: str
    display_name: str
    normalized_name: str
    compatibility_sid: int
    model_fingerprint: str
    embedding_dim: int
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class VisitorSession:
    session_uuid: str
    slot: int
    generation: int
    state: str
    model_fingerprint: str
    embedding_dim: int
    created_at: float
    last_seen_at: float
    expires_at: float


@dataclass(frozen=True)
class VisitorExemplarRecord:
    exemplar_uuid: str
    session_uuid: str
    vector: Tuple[float, ...]
    model_fingerprint: str
    embedding_dim: int
    observation_id: str
    created_at: float


@dataclass(frozen=True)
class ExemplarRecord:
    exemplar_uuid: str
    resident_uuid: str
    role: str
    vector: Tuple[float, ...]
    model_fingerprint: str
    embedding_dim: int
    observation_id: str
    independence_key: str
    created_at: float
    promoted_at: Optional[float]
    expires_at: Optional[float]
    corroboration_count: int


@dataclass(frozen=True)
class PromotionResult:
    exemplar_uuid: str
    corroboration_count: int
    promoted: bool
    duplicate_observation: bool


@dataclass(frozen=True)
class PurgeResult:
    visitor_sessions: int
    provisional_sessions: int
    quarantine_exemplars: int
    enrollment_proposals: int = 0


@dataclass(frozen=True)
class DeletionResult:
    resident_deleted: bool
    exemplars_deleted: int


@dataclass(frozen=True)
class IdentityHealth:
    schema_version: int
    resident_count: int
    residents_with_anchors: int
    residents_without_anchors: int
    enrollment_anchor_count: int
    adaptive_exemplar_count: int
    quarantine_exemplar_count: int
    active_visitor_sessions: int
    visitor_slot_count: int
    active_provisional_sessions: int
    visitor_exemplar_count: int
    pending_enrollment_proposals: int
    confirmed_enrollment_proposals: int
    model_profiles: Tuple[Tuple[str, int, int], ...]


@dataclass(frozen=True)
class EnrollmentObservationKey:
    run_id: str
    camera_id: str
    tracker_id: str
    frame_id: int
    observation_id: str

    def __post_init__(self) -> None:
        for field in ("run_id", "camera_id", "tracker_id", "observation_id"):
            object.__setattr__(self, field, _require_text(getattr(self, field), field))
        frame = int(self.frame_id)
        if frame < 0:
            raise ValueError("frame_id must be non-negative")
        object.__setattr__(self, "frame_id", frame)

    @property
    def tracklet_id(self) -> str:
        return json.dumps(
            [self.run_id, self.camera_id, self.tracker_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class EnrollmentProposalRecord:
    proposal_uuid: str
    key: EnrollmentObservationKey
    observation_quality: float
    observation_evidence: Tuple[str, ...]
    display_name: str
    normalized_name: str
    compatibility_sid: Optional[int]
    target_resident_uuid: Optional[str]
    proposed_resident_uuid: Optional[str]
    action: str
    model_fingerprint: str
    embedding_dim: int
    evidence_digest: str
    created_at: float
    expires_at: float
    state: str
    confirmed_at: Optional[float]
    result_resident_uuid: Optional[str]
    anchor_exemplar_uuid: Optional[str]

    def effective_state(self, *, now: Optional[float] = None) -> str:
        timestamp = float(time.time() if now is None else now)
        if self.state == "pending" and self.expires_at <= timestamp:
            return "expired"
        return self.state


@dataclass(frozen=True)
class EnrollmentProposalResult:
    proposal: EnrollmentProposalRecord
    idempotent: bool


@dataclass(frozen=True)
class EnrollmentConfirmationResult:
    proposal: EnrollmentProposalRecord
    resident: ResidentRecord
    anchor: ExemplarRecord
    idempotent: bool


@dataclass(frozen=True)
class LegacyResidentImport:
    resident_uuid: str
    display_name: str
    compatibility_sid: int
    model_fingerprint: str
    embedding_dim: int
    anchor_vectors: Tuple[Tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class LegacyImportResult:
    idempotent: bool
    residents_created: int
    residents_existing: int
    anchors_created: int


def normalize_display_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = " ".join(normalized.split()).casefold()
    if not normalized:
        raise ValueError("display_name must be non-empty")
    return normalized


def _require_text(value: str, field: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise ValueError(f"{field} must be non-empty")
    return out


def _canonical_uuid(value: str, field: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid UUID") from exc


def _encode_vector(vector: Sequence[float], expected_dim: int) -> bytes:
    values = tuple(float(value) for value in vector)
    if len(values) != int(expected_dim):
        raise ModelProfileMismatch(
            f"embedding dimension {len(values)} does not match expected {expected_dim}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding values must be finite")
    return struct.pack(f"<{len(values)}f", *values)


def _decode_vector(blob: bytes, dim: int) -> Tuple[float, ...]:
    expected_bytes = int(dim) * 4
    if len(blob) != expected_bytes:
        raise IdentityStoreError(
            f"stored embedding has {len(blob)} bytes; expected {expected_bytes}"
        )
    return tuple(float(value) for value in struct.unpack(f"<{int(dim)}f", blob))


def _enrollment_evidence_digest(
    *,
    key: EnrollmentObservationKey,
    observation_quality: float,
    observation_evidence: Sequence[str],
    display_name: str,
    compatibility_sid: Optional[int],
    target_resident_uuid: Optional[str],
    proposed_resident_uuid: Optional[str],
    action: str,
    model_fingerprint: str,
    embedding_dim: int,
    embedding_blob: bytes,
) -> str:
    metadata = json.dumps(
        {
            "version": 1,
            "run_id": key.run_id,
            "camera_id": key.camera_id,
            "tracker_id": key.tracker_id,
            "frame_id": key.frame_id,
            "observation_id": key.observation_id,
            "observation_quality": observation_quality,
            "observation_evidence": list(observation_evidence),
            "display_name": display_name,
            "compatibility_sid": compatibility_sid,
            "target_resident_uuid": target_resident_uuid,
            "proposed_resident_uuid": proposed_resident_uuid,
            "action": action,
            "model_fingerprint": model_fingerprint,
            "embedding_dim": embedding_dim,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(b"noesis-enrollment-evidence-v1\0")
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(embedding_blob)
    return digest.hexdigest()


class IdentityStore:
    """Fail-loud transactional identity persistence for a single local home."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        try:
            self.path = prepare_private_writable_file(
                Path(path).expanduser(),
                label="identity-v2 store",
            )
        except PrivatePathError as exc:
            raise IdentityStoreError(str(exc)) from exc
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = DELETE")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._migrate()
        try:
            validate_private_file(self.path, label="identity-v2 store")
        except PrivatePathError as exc:
            self._conn.close()
            raise IdentityStoreError(str(exc)) from exc

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            try:
                validate_private_file(self.path, label="identity-v2 store")
            except PrivatePathError as exc:
                raise IdentityStoreError(str(exc)) from exc

    def __enter__(self) -> "IdentityStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
                try:
                    validate_private_file(self.path, label="identity-v2 store")
                except PrivatePathError as exc:
                    raise IdentityStoreError(str(exc)) from exc

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def _migrate(self) -> None:
        current = self.schema_version
        if current > SCHEMA_VERSION:
            raise IdentityStoreError(
                f"identity DB schema {current} is newer than supported {SCHEMA_VERSION}"
            )
        while current < SCHEMA_VERSION:
            target = current + 1
            migration = getattr(self, f"_migrate_{current}_to_{target}", None)
            if migration is None:
                raise IdentityStoreError(
                    f"missing schema migration {current}->{target}"
                )
            with self.transaction() as conn:
                migration(conn)
                conn.execute(f"PRAGMA user_version = {target}")
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (target, time.time()),
                )
            current = target

    @staticmethod
    def _migrate_0_to_1(conn: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE residents (
                resident_uuid TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE,
                compatibility_sid INTEGER NOT NULL UNIQUE CHECK (compatibility_sid > 0),
                model_fingerprint TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE visitor_slots (
                slot INTEGER PRIMARY KEY CHECK (slot > 0),
                generation INTEGER NOT NULL CHECK (generation >= 0),
                current_session_uuid TEXT UNIQUE,
                last_released_at REAL
            )
            """,
            """
            CREATE TABLE visitor_sessions (
                session_uuid TEXT PRIMARY KEY,
                slot INTEGER NOT NULL,
                generation INTEGER NOT NULL CHECK (generation > 0),
                state TEXT NOT NULL CHECK (state IN ('active', 'released')),
                model_fingerprint TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                ttl_s REAL NOT NULL CHECK (ttl_s > 0),
                UNIQUE(slot, generation),
                FOREIGN KEY(slot) REFERENCES visitor_slots(slot) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE provisional_sessions (
                session_uuid TEXT PRIMARY KEY,
                tracklet_key TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                ttl_s REAL NOT NULL CHECK (ttl_s > 0)
            )
            """,
            """
            CREATE TABLE resident_exemplars (
                exemplar_uuid TEXT PRIMARY KEY,
                resident_uuid TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('enrollment_anchor', 'quarantine', 'adaptive')),
                embedding BLOB NOT NULL,
                model_fingerprint TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                observation_id TEXT NOT NULL,
                independence_key TEXT NOT NULL,
                created_at REAL NOT NULL,
                promoted_at REAL,
                expires_at REAL,
                UNIQUE(resident_uuid, observation_id),
                FOREIGN KEY(resident_uuid) REFERENCES residents(resident_uuid) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE exemplar_corroborations (
                exemplar_uuid TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                independence_key TEXT NOT NULL,
                observed_at REAL NOT NULL,
                PRIMARY KEY(exemplar_uuid, independence_key),
                UNIQUE(exemplar_uuid, observation_id),
                FOREIGN KEY(exemplar_uuid) REFERENCES resident_exemplars(exemplar_uuid) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE legacy_migrations (
                migration_key TEXT PRIMARY KEY,
                source_digest TEXT NOT NULL,
                applied_at REAL NOT NULL,
                result_json TEXT NOT NULL
            )
            """,
            """
            CREATE TRIGGER enrollment_anchor_immutable
            BEFORE UPDATE ON resident_exemplars
            WHEN OLD.role = 'enrollment_anchor'
            BEGIN
                SELECT RAISE(ABORT, 'enrollment anchors are immutable');
            END
            """,
            """
            CREATE TRIGGER enrollment_anchor_delete_guard
            BEFORE DELETE ON resident_exemplars
            WHEN OLD.role = 'enrollment_anchor'
             AND EXISTS (
                 SELECT 1 FROM residents WHERE resident_uuid = OLD.resident_uuid
             )
            BEGIN
                SELECT RAISE(ABORT, 'enrollment anchors may only be deleted with their resident');
            END
            """,
            "CREATE INDEX resident_exemplars_owner_role ON resident_exemplars(resident_uuid, role, created_at, exemplar_uuid)",
            "CREATE INDEX visitor_sessions_expiry ON visitor_sessions(expires_at)",
            "CREATE INDEX provisional_sessions_expiry ON provisional_sessions(expires_at)",
            "CREATE INDEX resident_exemplars_expiry ON resident_exemplars(role, expires_at)",
        )
        for statement in statements:
            conn.execute(statement)

    @staticmethod
    def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE visitor_exemplars (
                exemplar_uuid TEXT PRIMARY KEY,
                session_uuid TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_fingerprint TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                observation_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(session_uuid, observation_id),
                FOREIGN KEY(session_uuid) REFERENCES visitor_sessions(session_uuid) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE enrollment_proposals (
                proposal_uuid TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                tracker_id TEXT NOT NULL,
                frame_id INTEGER NOT NULL CHECK (frame_id >= 0),
                observation_id TEXT NOT NULL,
                observation_quality REAL NOT NULL CHECK (observation_quality >= 0.0 AND observation_quality <= 1.0),
                observation_evidence_json TEXT NOT NULL,
                display_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                compatibility_sid INTEGER,
                target_resident_uuid TEXT,
                proposed_resident_uuid TEXT,
                action TEXT NOT NULL CHECK (action IN ('create_resident', 'add_anchor')),
                embedding BLOB NOT NULL,
                model_fingerprint TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                evidence_digest TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'confirmed', 'cancelled')),
                confirmed_at REAL,
                result_resident_uuid TEXT,
                anchor_exemplar_uuid TEXT,
                UNIQUE(run_id, camera_id, tracker_id, frame_id, observation_id),
                FOREIGN KEY(target_resident_uuid) REFERENCES residents(resident_uuid) ON DELETE CASCADE,
                FOREIGN KEY(result_resident_uuid) REFERENCES residents(resident_uuid) ON DELETE CASCADE,
                FOREIGN KEY(anchor_exemplar_uuid) REFERENCES resident_exemplars(exemplar_uuid) ON DELETE CASCADE,
                CHECK (
                    (state IN ('pending', 'cancelled')
                     AND confirmed_at IS NULL
                     AND result_resident_uuid IS NULL
                     AND anchor_exemplar_uuid IS NULL)
                    OR
                    (state = 'confirmed'
                     AND confirmed_at IS NOT NULL
                     AND result_resident_uuid IS NOT NULL
                     AND anchor_exemplar_uuid IS NOT NULL)
                )
            )
            """,
            """
            CREATE TRIGGER enrollment_proposal_evidence_immutable
            BEFORE UPDATE ON enrollment_proposals
            WHEN OLD.run_id IS NOT NEW.run_id
              OR OLD.camera_id IS NOT NEW.camera_id
              OR OLD.tracker_id IS NOT NEW.tracker_id
              OR OLD.frame_id IS NOT NEW.frame_id
              OR OLD.observation_id IS NOT NEW.observation_id
              OR OLD.observation_quality IS NOT NEW.observation_quality
              OR OLD.observation_evidence_json IS NOT NEW.observation_evidence_json
              OR OLD.display_name IS NOT NEW.display_name
              OR OLD.normalized_name IS NOT NEW.normalized_name
              OR OLD.compatibility_sid IS NOT NEW.compatibility_sid
              OR OLD.target_resident_uuid IS NOT NEW.target_resident_uuid
              OR OLD.proposed_resident_uuid IS NOT NEW.proposed_resident_uuid
              OR OLD.action IS NOT NEW.action
              OR OLD.embedding IS NOT NEW.embedding
              OR OLD.model_fingerprint IS NOT NEW.model_fingerprint
              OR OLD.embedding_dim IS NOT NEW.embedding_dim
              OR OLD.evidence_digest IS NOT NEW.evidence_digest
              OR OLD.created_at IS NOT NEW.created_at
              OR OLD.expires_at IS NOT NEW.expires_at
            BEGIN
                SELECT RAISE(ABORT, 'enrollment proposal evidence is immutable');
            END
            """,
            """
            CREATE TRIGGER confirmed_enrollment_proposal_delete_guard
            BEFORE DELETE ON enrollment_proposals
            WHEN OLD.state = 'confirmed'
             AND OLD.result_resident_uuid IS NOT NULL
             AND EXISTS (
                 SELECT 1 FROM residents WHERE resident_uuid = OLD.result_resident_uuid
             )
            BEGIN
                SELECT RAISE(ABORT, 'confirmed enrollment evidence may only be deleted with its resident');
            END
            """,
            """
            CREATE TRIGGER confirmed_enrollment_proposal_immutable
            BEFORE UPDATE ON enrollment_proposals
            WHEN OLD.state = 'confirmed'
            BEGIN
                SELECT RAISE(ABORT, 'confirmed enrollment proposal is immutable');
            END
            """,
            "CREATE INDEX visitor_exemplars_session ON visitor_exemplars(session_uuid, created_at, exemplar_uuid)",
            "CREATE INDEX enrollment_proposals_expiry ON enrollment_proposals(state, expires_at)",
            "CREATE INDEX enrollment_proposals_target ON enrollment_proposals(target_resident_uuid)",
        )
        for statement in statements:
            conn.execute(statement)

    def create_resident(
        self,
        *,
        display_name: str,
        compatibility_sid: int,
        model_fingerprint: str,
        embedding_dim: int,
        resident_uuid: Optional[str] = None,
        now: Optional[float] = None,
    ) -> ResidentRecord:
        timestamp = float(time.time() if now is None else now)
        name = " ".join(str(display_name or "").split())
        normalized = normalize_display_name(name)
        fingerprint = _require_text(model_fingerprint, "model_fingerprint")
        sid = int(compatibility_sid)
        dim = int(embedding_dim)
        if sid <= 0 or dim <= 0:
            raise ValueError("compatibility_sid and embedding_dim must be positive")
        resident_id = (
            _canonical_uuid(resident_uuid, "resident_uuid")
            if resident_uuid is not None
            else str(uuid.uuid4())
        )
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO residents(
                        resident_uuid, display_name, normalized_name,
                        compatibility_sid, model_fingerprint, embedding_dim,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resident_id,
                        name,
                        normalized,
                        sid,
                        fingerprint,
                        dim,
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateResidentError(str(exc)) from exc
        return self.get_resident(resident_id)

    def get_resident(self, resident_uuid: str) -> ResidentRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM residents WHERE resident_uuid = ?",
                (str(resident_uuid),),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown resident UUID: {resident_uuid}")
        return self._resident_from_row(row)

    def list_residents(self) -> Tuple[ResidentRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM residents ORDER BY compatibility_sid, resident_uuid"
            ).fetchall()
        return tuple(self._resident_from_row(row) for row in rows)

    def find_resident_by_display_name(
        self, display_name: str
    ) -> Optional[ResidentRecord]:
        normalized = normalize_display_name(display_name)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM residents WHERE normalized_name = ?",
                (normalized,),
            ).fetchone()
        return self._resident_from_row(row) if row is not None else None

    def update_resident_display_name(
        self,
        resident_uuid: str,
        *,
        display_name: str,
        now: Optional[float] = None,
    ) -> ResidentRecord:
        resident_id = _canonical_uuid(resident_uuid, "resident_uuid")
        name = " ".join(str(display_name or "").split())
        normalized = normalize_display_name(name)
        timestamp = float(time.time() if now is None else now)
        try:
            with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE residents
                    SET display_name = ?, normalized_name = ?, updated_at = ?
                    WHERE resident_uuid = ?
                    """,
                    (name, normalized, timestamp, resident_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown resident UUID: {resident_uuid}")
        except sqlite3.IntegrityError as exc:
            raise DuplicateResidentError(str(exc)) from exc
        return self.get_resident(resident_id)

    @staticmethod
    def _resident_from_row(row: sqlite3.Row) -> ResidentRecord:
        return ResidentRecord(
            resident_uuid=str(row["resident_uuid"]),
            display_name=str(row["display_name"]),
            normalized_name=str(row["normalized_name"]),
            compatibility_sid=int(row["compatibility_sid"]),
            model_fingerprint=str(row["model_fingerprint"]),
            embedding_dim=int(row["embedding_dim"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def delete_resident(self, resident_uuid: str) -> DeletionResult:
        with self.transaction() as conn:
            exemplar_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM resident_exemplars WHERE resident_uuid = ?",
                    (str(resident_uuid),),
                ).fetchone()[0]
            )
            cursor = conn.execute(
                "DELETE FROM residents WHERE resident_uuid = ?",
                (str(resident_uuid),),
            )
        return DeletionResult(
            bool(cursor.rowcount), exemplar_count if cursor.rowcount else 0
        )

    def _assert_resident_profile(
        self,
        conn: sqlite3.Connection,
        resident_uuid: str,
        model_fingerprint: str,
        embedding_dim: int,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM residents WHERE resident_uuid = ?",
            (str(resident_uuid),),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown resident UUID: {resident_uuid}")
        if str(row["model_fingerprint"]) != str(model_fingerprint):
            raise ModelProfileMismatch(
                f"model fingerprint {model_fingerprint!r} does not match resident profile {row['model_fingerprint']!r}"
            )
        if int(row["embedding_dim"]) != int(embedding_dim):
            raise ModelProfileMismatch(
                f"embedding dimension {embedding_dim} does not match resident profile {row['embedding_dim']}"
            )
        return row

    def add_enrollment_anchor(
        self,
        *,
        resident_uuid: str,
        vector: Sequence[float],
        model_fingerprint: str,
        embedding_dim: int,
        observation_id: str,
        independence_key: str,
        exemplar_uuid: Optional[str] = None,
        now: Optional[float] = None,
    ) -> ExemplarRecord:
        return self._add_exemplar(
            resident_uuid=resident_uuid,
            vector=vector,
            model_fingerprint=model_fingerprint,
            embedding_dim=embedding_dim,
            observation_id=observation_id,
            independence_key=independence_key,
            role=_ANCHOR_ROLE,
            exemplar_uuid=exemplar_uuid,
            now=now,
            expires_at=None,
        )

    def add_quarantine_exemplar(
        self,
        *,
        resident_uuid: str,
        vector: Sequence[float],
        model_fingerprint: str,
        embedding_dim: int,
        observation_id: str,
        independence_key: str,
        ttl_s: float,
        exemplar_uuid: Optional[str] = None,
        now: Optional[float] = None,
    ) -> ExemplarRecord:
        timestamp = float(time.time() if now is None else now)
        ttl = float(ttl_s)
        if ttl <= 0.0:
            raise ValueError("ttl_s must be positive")
        return self._add_exemplar(
            resident_uuid=resident_uuid,
            vector=vector,
            model_fingerprint=model_fingerprint,
            embedding_dim=embedding_dim,
            observation_id=observation_id,
            independence_key=independence_key,
            role=_QUARANTINE_ROLE,
            exemplar_uuid=exemplar_uuid,
            now=timestamp,
            expires_at=timestamp + ttl,
        )

    def _add_exemplar(
        self,
        *,
        resident_uuid: str,
        vector: Sequence[float],
        model_fingerprint: str,
        embedding_dim: int,
        observation_id: str,
        independence_key: str,
        role: str,
        exemplar_uuid: Optional[str],
        now: Optional[float],
        expires_at: Optional[float],
    ) -> ExemplarRecord:
        timestamp = float(time.time() if now is None else now)
        fingerprint = _require_text(model_fingerprint, "model_fingerprint")
        observation = _require_text(observation_id, "observation_id")
        independent = _require_text(independence_key, "independence_key")
        dim = int(embedding_dim)
        blob = _encode_vector(vector, dim)
        exemplar_id = (
            _canonical_uuid(exemplar_uuid, "exemplar_uuid")
            if exemplar_uuid is not None
            else str(uuid.uuid4())
        )
        with self.transaction() as conn:
            self._assert_resident_profile(conn, resident_uuid, fingerprint, dim)
            conn.execute(
                """
                INSERT INTO resident_exemplars(
                    exemplar_uuid, resident_uuid, role, embedding,
                    model_fingerprint, embedding_dim, observation_id,
                    independence_key, created_at, promoted_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exemplar_id,
                    str(resident_uuid),
                    role,
                    blob,
                    fingerprint,
                    dim,
                    observation,
                    independent,
                    timestamp,
                    timestamp if role == _ANCHOR_ROLE else None,
                    expires_at,
                ),
            )
            if role == _QUARANTINE_ROLE:
                conn.execute(
                    """
                    INSERT INTO exemplar_corroborations(
                        exemplar_uuid, observation_id, independence_key, observed_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (exemplar_id, observation, independent, timestamp),
                )
        return self.get_exemplar(exemplar_id)

    def corroborate_quarantine(
        self,
        exemplar_uuid: str,
        *,
        observation_id: str,
        independence_key: str,
        required_observations: int = 3,
        now: Optional[float] = None,
    ) -> PromotionResult:
        required = int(required_observations)
        if required < 2:
            raise ValueError("required_observations must be at least 2")
        observation = _require_text(observation_id, "observation_id")
        independent = _require_text(independence_key, "independence_key")
        timestamp = float(time.time() if now is None else now)
        duplicate = False
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT role, expires_at FROM resident_exemplars WHERE exemplar_uuid = ?",
                (str(exemplar_uuid),),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown exemplar UUID: {exemplar_uuid}")
            role = str(row["role"])
            if role == _ANCHOR_ROLE:
                raise IdentityStoreError(
                    "enrollment anchors do not require corroboration"
                )
            if role == _QUARANTINE_ROLE and row["expires_at"] is not None:
                if float(row["expires_at"]) <= timestamp:
                    raise IdentityStoreError("quarantine exemplar has expired")
            existing = conn.execute(
                """
                SELECT 1 FROM exemplar_corroborations
                WHERE exemplar_uuid = ? AND (observation_id = ? OR independence_key = ?)
                """,
                (str(exemplar_uuid), observation, independent),
            ).fetchone()
            if existing is not None:
                duplicate = True
            elif role == _QUARANTINE_ROLE:
                conn.execute(
                    """
                    INSERT INTO exemplar_corroborations(
                        exemplar_uuid, observation_id, independence_key, observed_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (str(exemplar_uuid), observation, independent, timestamp),
                )
            count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM exemplar_corroborations WHERE exemplar_uuid = ?",
                    (str(exemplar_uuid),),
                ).fetchone()[0]
            )
            promoted = role == _ADAPTIVE_ROLE
            if role == _QUARANTINE_ROLE and count >= required:
                conn.execute(
                    """
                    UPDATE resident_exemplars
                    SET role = ?, promoted_at = ?, expires_at = NULL
                    WHERE exemplar_uuid = ?
                    """,
                    (_ADAPTIVE_ROLE, timestamp, str(exemplar_uuid)),
                )
                promoted = True
        return PromotionResult(str(exemplar_uuid), count, promoted, duplicate)

    def get_exemplar(self, exemplar_uuid: str) -> ExemplarRecord:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT e.*, COUNT(c.independence_key) AS corroboration_count
                FROM resident_exemplars e
                LEFT JOIN exemplar_corroborations c ON c.exemplar_uuid = e.exemplar_uuid
                WHERE e.exemplar_uuid = ?
                GROUP BY e.exemplar_uuid
                """,
                (str(exemplar_uuid),),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown exemplar UUID: {exemplar_uuid}")
        return self._exemplar_from_row(row)

    def load_resident_gallery(
        self,
        resident_uuid: str,
        *,
        model_fingerprint: str,
        embedding_dim: int,
        include_quarantine: bool = False,
    ) -> Tuple[ExemplarRecord, ...]:
        with self._lock:
            self._assert_resident_profile(
                self._conn,
                resident_uuid,
                model_fingerprint,
                int(embedding_dim),
            )
            roles = (
                (_ANCHOR_ROLE, _ADAPTIVE_ROLE, _QUARANTINE_ROLE)
                if include_quarantine
                else (
                    _ANCHOR_ROLE,
                    _ADAPTIVE_ROLE,
                )
            )
            placeholders = ",".join("?" for _ in roles)
            rows = self._conn.execute(
                f"""
                SELECT e.*, COUNT(c.independence_key) AS corroboration_count
                FROM resident_exemplars e
                LEFT JOIN exemplar_corroborations c ON c.exemplar_uuid = e.exemplar_uuid
                WHERE e.resident_uuid = ? AND e.role IN ({placeholders})
                GROUP BY e.exemplar_uuid
                ORDER BY CASE e.role
                    WHEN 'enrollment_anchor' THEN 0
                    WHEN 'adaptive' THEN 1
                    ELSE 2 END,
                    e.created_at, e.exemplar_uuid
                """,
                (str(resident_uuid), *roles),
            ).fetchall()
        return tuple(self._exemplar_from_row(row) for row in rows)

    @staticmethod
    def _exemplar_from_row(row: sqlite3.Row) -> ExemplarRecord:
        dim = int(row["embedding_dim"])
        return ExemplarRecord(
            exemplar_uuid=str(row["exemplar_uuid"]),
            resident_uuid=str(row["resident_uuid"]),
            role=str(row["role"]),
            vector=_decode_vector(bytes(row["embedding"]), dim),
            model_fingerprint=str(row["model_fingerprint"]),
            embedding_dim=dim,
            observation_id=str(row["observation_id"]),
            independence_key=str(row["independence_key"]),
            created_at=float(row["created_at"]),
            promoted_at=float(row["promoted_at"])
            if row["promoted_at"] is not None
            else None,
            expires_at=float(row["expires_at"])
            if row["expires_at"] is not None
            else None,
            corroboration_count=int(row["corroboration_count"]),
        )

    def open_visitor_session(
        self,
        *,
        slot: int,
        model_fingerprint: str,
        embedding_dim: int,
        ttl_s: float,
        session_uuid: Optional[str] = None,
        now: Optional[float] = None,
    ) -> VisitorSession:
        slot_int = int(slot)
        dim = int(embedding_dim)
        ttl = float(ttl_s)
        timestamp = float(time.time() if now is None else now)
        if slot_int <= 0 or dim <= 0 or ttl <= 0.0:
            raise ValueError("slot, embedding_dim, and ttl_s must be positive")
        fingerprint = _require_text(model_fingerprint, "model_fingerprint")
        session_id = (
            _canonical_uuid(session_uuid, "session_uuid")
            if session_uuid is not None
            else str(uuid.uuid4())
        )
        with self.transaction() as conn:
            slot_row = conn.execute(
                "SELECT * FROM visitor_slots WHERE slot = ?",
                (slot_int,),
            ).fetchone()
            if slot_row is None:
                generation = 1
                conn.execute(
                    """
                    INSERT INTO visitor_slots(slot, generation, current_session_uuid, last_released_at)
                    VALUES (?, 0, NULL, NULL)
                    """,
                    (slot_int,),
                )
            else:
                generation = int(slot_row["generation"]) + 1
                current = slot_row["current_session_uuid"]
                if current:
                    current_row = conn.execute(
                        "SELECT state, expires_at FROM visitor_sessions WHERE session_uuid = ?",
                        (str(current),),
                    ).fetchone()
                    if (
                        current_row is not None
                        and str(current_row["state"]) == "active"
                        and float(current_row["expires_at"]) > timestamp
                    ):
                        raise VisitorSlotInUse(
                            f"visitor slot {slot_int} is still active"
                        )
                    if current_row is not None:
                        conn.execute(
                            "UPDATE visitor_sessions SET state = 'released', expires_at = ? WHERE session_uuid = ?",
                            (timestamp, str(current)),
                        )
                    conn.execute(
                        "UPDATE visitor_slots SET current_session_uuid = NULL, last_released_at = ? WHERE slot = ?",
                        (timestamp, slot_int),
                    )
            conn.execute(
                """
                INSERT INTO visitor_sessions(
                    session_uuid, slot, generation, state, model_fingerprint,
                    embedding_dim, created_at, last_seen_at, expires_at, ttl_s
                ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    slot_int,
                    generation,
                    fingerprint,
                    dim,
                    timestamp,
                    timestamp,
                    timestamp + ttl,
                    ttl,
                ),
            )
            conn.execute(
                """
                UPDATE visitor_slots
                SET generation = ?, current_session_uuid = ?
                WHERE slot = ?
                """,
                (generation, session_id, slot_int),
            )
        return self.get_visitor_session(session_id)

    def get_visitor_session(self, session_uuid: str) -> VisitorSession:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM visitor_sessions WHERE session_uuid = ?",
                (str(session_uuid),),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown visitor session UUID: {session_uuid}")
        return VisitorSession(
            session_uuid=str(row["session_uuid"]),
            slot=int(row["slot"]),
            generation=int(row["generation"]),
            state=str(row["state"]),
            model_fingerprint=str(row["model_fingerprint"]),
            embedding_dim=int(row["embedding_dim"]),
            created_at=float(row["created_at"]),
            last_seen_at=float(row["last_seen_at"]),
            expires_at=float(row["expires_at"]),
        )

    def list_active_visitor_sessions(
        self,
        *,
        now: Optional[float] = None,
    ) -> Tuple[VisitorSession, ...]:
        timestamp = float(time.time() if now is None else now)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM visitor_sessions
                WHERE state = 'active' AND expires_at > ?
                ORDER BY slot, generation, session_uuid
                """,
                (timestamp,),
            ).fetchall()
        return tuple(
            VisitorSession(
                session_uuid=str(row["session_uuid"]),
                slot=int(row["slot"]),
                generation=int(row["generation"]),
                state=str(row["state"]),
                model_fingerprint=str(row["model_fingerprint"]),
                embedding_dim=int(row["embedding_dim"]),
                created_at=float(row["created_at"]),
                last_seen_at=float(row["last_seen_at"]),
                expires_at=float(row["expires_at"]),
            )
            for row in rows
        )

    def add_visitor_exemplar(
        self,
        *,
        session_uuid: str,
        vector: Sequence[float],
        model_fingerprint: str,
        embedding_dim: int,
        observation_id: str,
        exemplar_uuid: Optional[str] = None,
        max_exemplars: int = 32,
        now: Optional[float] = None,
    ) -> VisitorExemplarRecord:
        session_id = _canonical_uuid(session_uuid, "session_uuid")
        exemplar_id = (
            _canonical_uuid(exemplar_uuid, "exemplar_uuid")
            if exemplar_uuid is not None
            else str(uuid.uuid4())
        )
        fingerprint = _require_text(model_fingerprint, "model_fingerprint")
        dimension = int(embedding_dim)
        observation = _require_text(observation_id, "observation_id")
        timestamp = float(time.time() if now is None else now)
        maximum = int(max_exemplars)
        if maximum < 1:
            raise ValueError("max_exemplars must be positive")
        blob = _encode_vector(vector, dimension)
        with self.transaction() as conn:
            session = conn.execute(
                "SELECT * FROM visitor_sessions WHERE session_uuid = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"unknown visitor session UUID: {session_uuid}")
            if str(session["state"]) != "active":
                raise IdentityStoreError("visitor session is no longer active")
            if float(session["expires_at"]) <= timestamp:
                raise IdentityStoreError("visitor session has expired")
            if str(session["model_fingerprint"]) != fingerprint:
                raise ModelProfileMismatch("visitor model fingerprint mismatch")
            if int(session["embedding_dim"]) != dimension:
                raise ModelProfileMismatch("visitor embedding dimension mismatch")
            conn.execute(
                """
                INSERT INTO visitor_exemplars(
                    exemplar_uuid, session_uuid, embedding, model_fingerprint,
                    embedding_dim, observation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exemplar_id,
                    session_id,
                    blob,
                    fingerprint,
                    dimension,
                    observation,
                    timestamp,
                ),
            )
            conn.execute(
                """
                UPDATE visitor_sessions
                SET last_seen_at = ?, expires_at = ?
                WHERE session_uuid = ?
                """,
                (timestamp, timestamp + float(session["ttl_s"]), session_id),
            )
            conn.execute(
                """
                DELETE FROM visitor_exemplars
                WHERE exemplar_uuid IN (
                    SELECT exemplar_uuid FROM visitor_exemplars
                    WHERE session_uuid = ? AND exemplar_uuid != ?
                    ORDER BY created_at DESC, exemplar_uuid DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (session_id, exemplar_id, maximum - 1),
            )
        return self.get_visitor_exemplar(exemplar_id)

    def get_visitor_exemplar(self, exemplar_uuid: str) -> VisitorExemplarRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM visitor_exemplars WHERE exemplar_uuid = ?",
                (str(exemplar_uuid),),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown visitor exemplar UUID: {exemplar_uuid}")
        return self._visitor_exemplar_from_row(row)

    @staticmethod
    def _visitor_exemplar_from_row(row: sqlite3.Row) -> VisitorExemplarRecord:
        dimension = int(row["embedding_dim"])
        return VisitorExemplarRecord(
            exemplar_uuid=str(row["exemplar_uuid"]),
            session_uuid=str(row["session_uuid"]),
            vector=_decode_vector(bytes(row["embedding"]), dimension),
            model_fingerprint=str(row["model_fingerprint"]),
            embedding_dim=dimension,
            observation_id=str(row["observation_id"]),
            created_at=float(row["created_at"]),
        )

    def load_active_visitor_galleries(
        self,
        *,
        now: Optional[float] = None,
    ) -> Tuple[Tuple[VisitorSession, Tuple[VisitorExemplarRecord, ...]], ...]:
        out = []
        with self._lock:
            sessions = self.list_active_visitor_sessions(now=now)
            for session in sessions:
                rows = self._conn.execute(
                    """
                    SELECT * FROM visitor_exemplars
                    WHERE session_uuid = ?
                    ORDER BY created_at, exemplar_uuid
                    """,
                    (session.session_uuid,),
                ).fetchall()
                out.append(
                    (
                        session,
                        tuple(self._visitor_exemplar_from_row(row) for row in rows),
                    )
                )
        return tuple(out)

    def load_visitor_gallery(
        self,
        session_uuid: str,
    ) -> Tuple[VisitorSession, Tuple[VisitorExemplarRecord, ...]]:
        """Load one visitor session and its bounded gallery.

        This is the mutation-side counterpart to
        :meth:`load_active_visitor_galleries`.  Runtime updates already know
        the exact session that changed, so reading that gallery directly
        avoids reloading every active visitor gallery after each observation.
        The caller remains responsible for applying active/expiry semantics.
        """

        session_id = str(session_uuid)
        with self._lock:
            session_row = self._conn.execute(
                "SELECT * FROM visitor_sessions WHERE session_uuid = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"unknown visitor session UUID: {session_uuid}")
            session = VisitorSession(
                session_uuid=str(session_row["session_uuid"]),
                slot=int(session_row["slot"]),
                generation=int(session_row["generation"]),
                state=str(session_row["state"]),
                model_fingerprint=str(session_row["model_fingerprint"]),
                embedding_dim=int(session_row["embedding_dim"]),
                created_at=float(session_row["created_at"]),
                last_seen_at=float(session_row["last_seen_at"]),
                expires_at=float(session_row["expires_at"]),
            )
            rows = self._conn.execute(
                """
                SELECT * FROM visitor_exemplars
                WHERE session_uuid = ?
                ORDER BY created_at, exemplar_uuid
                """,
                (session_id,),
            ).fetchall()
        return session, tuple(self._visitor_exemplar_from_row(row) for row in rows)

    def release_visitor_session(
        self, session_uuid: str, *, now: Optional[float] = None
    ) -> None:
        timestamp = float(time.time() if now is None else now)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT slot FROM visitor_sessions WHERE session_uuid = ?",
                (str(session_uuid),),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown visitor session UUID: {session_uuid}")
            slot = int(row["slot"])
            conn.execute(
                "UPDATE visitor_sessions SET state = 'released', expires_at = ? WHERE session_uuid = ?",
                (timestamp, str(session_uuid)),
            )
            conn.execute(
                """
                UPDATE visitor_slots
                SET current_session_uuid = NULL, last_released_at = ?
                WHERE slot = ? AND current_session_uuid = ?
                """,
                (timestamp, slot, str(session_uuid)),
            )

    def touch_visitor_session(
        self,
        session_uuid: str,
        *,
        now: Optional[float] = None,
    ) -> VisitorSession:
        """Renew one exact visitor incarnation without relying on its recycled slot."""

        timestamp = float(time.time() if now is None else now)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state, expires_at, ttl_s FROM visitor_sessions WHERE session_uuid = ?",
                (str(session_uuid),),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown visitor session UUID: {session_uuid}")
            if str(row["state"]) != "active":
                raise IdentityStoreError("visitor session is no longer active")
            if float(row["expires_at"]) <= timestamp:
                raise IdentityStoreError("visitor session has expired")
            conn.execute(
                """
                UPDATE visitor_sessions
                SET last_seen_at = ?, expires_at = ?
                WHERE session_uuid = ?
                """,
                (timestamp, timestamp + float(row["ttl_s"]), str(session_uuid)),
            )
        return self.get_visitor_session(session_uuid)

    def create_provisional_session(
        self,
        *,
        tracklet_key: str,
        ttl_s: float,
        session_uuid: Optional[str] = None,
        now: Optional[float] = None,
    ) -> str:
        tracklet = _require_text(tracklet_key, "tracklet_key")
        ttl = float(ttl_s)
        timestamp = float(time.time() if now is None else now)
        if ttl <= 0.0:
            raise ValueError("ttl_s must be positive")
        session_id = (
            _canonical_uuid(session_uuid, "session_uuid")
            if session_uuid is not None
            else str(uuid.uuid4())
        )
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO provisional_sessions(
                    session_uuid, tracklet_key, created_at, last_seen_at, expires_at, ttl_s
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, tracklet, timestamp, timestamp, timestamp + ttl, ttl),
            )
        return session_id

    def touch_provisional_session(
        self,
        session_uuid: str,
        *,
        now: Optional[float] = None,
    ) -> None:
        timestamp = float(time.time() if now is None else now)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT expires_at, ttl_s FROM provisional_sessions WHERE session_uuid = ?",
                (str(session_uuid),),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown provisional session UUID: {session_uuid}")
            if float(row["expires_at"]) <= timestamp:
                raise IdentityStoreError("provisional session has expired")
            conn.execute(
                """
                UPDATE provisional_sessions
                SET last_seen_at = ?, expires_at = ?
                WHERE session_uuid = ?
                """,
                (timestamp, timestamp + float(row["ttl_s"]), str(session_uuid)),
            )

    def create_enrollment_proposal(
        self,
        *,
        key: EnrollmentObservationKey,
        observation_quality: float,
        observation_evidence: Sequence[str] = (),
        display_name: str,
        vector: Sequence[float],
        model_fingerprint: str,
        embedding_dim: int,
        ttl_s: float,
        compatibility_sid: Optional[int] = None,
        update_resident_uuid: Optional[str] = None,
        now: Optional[float] = None,
    ) -> EnrollmentProposalResult:
        if not isinstance(key, EnrollmentObservationKey):
            raise TypeError("key must be an EnrollmentObservationKey")
        quality = float(observation_quality)
        evidence = tuple(str(item) for item in observation_evidence)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("observation_quality must be finite and within [0, 1]")
        name = " ".join(str(display_name or "").split())
        normalized = normalize_display_name(name)
        fingerprint = _require_text(model_fingerprint, "model_fingerprint")
        dimension = int(embedding_dim)
        ttl = float(ttl_s)
        timestamp = float(time.time() if now is None else now)
        if dimension <= 0 or ttl <= 0.0:
            raise ValueError("embedding_dim and ttl_s must be positive")
        blob = _encode_vector(vector, dimension)
        target_id = (
            _canonical_uuid(update_resident_uuid, "update_resident_uuid")
            if update_resident_uuid is not None
            else None
        )
        sid = int(compatibility_sid) if compatibility_sid is not None else None
        if sid is not None and sid <= 0:
            raise ValueError("compatibility_sid must be positive")
        action = "add_anchor" if target_id is not None else "create_resident"
        if action == "create_resident" and sid is None:
            raise ValueError("compatibility_sid is required for a new resident")
        proposal_key = json.dumps(
            {
                "run_id": key.run_id,
                "camera_id": key.camera_id,
                "tracker_id": key.tracker_id,
                "frame_id": key.frame_id,
                "observation_id": key.observation_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        proposal_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"noesis-enrollment-proposal:{proposal_key}")
        )
        proposed_resident_id = (
            str(uuid.uuid5(uuid.UUID(proposal_id), "resident"))
            if action == "create_resident"
            else None
        )
        digest = _enrollment_evidence_digest(
            key=key,
            observation_quality=quality,
            observation_evidence=evidence,
            display_name=name,
            compatibility_sid=sid,
            target_resident_uuid=target_id,
            proposed_resident_uuid=proposed_resident_id,
            action=action,
            model_fingerprint=fingerprint,
            embedding_dim=dimension,
            embedding_blob=blob,
        )
        with self.transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM enrollment_proposals
                WHERE run_id = ? AND camera_id = ? AND tracker_id = ?
                  AND frame_id = ? AND observation_id = ?
                """,
                (
                    key.run_id,
                    key.camera_id,
                    key.tracker_id,
                    key.frame_id,
                    key.observation_id,
                ),
            ).fetchone()
            if existing is not None:
                if str(existing["evidence_digest"]) != digest:
                    raise EnrollmentProposalConflict(
                        "exact observation already has a different enrollment proposal"
                    )
                return EnrollmentProposalResult(
                    proposal=self._proposal_from_row(existing),
                    idempotent=True,
                )

            name_owner = conn.execute(
                "SELECT * FROM residents WHERE normalized_name = ?",
                (normalized,),
            ).fetchone()
            if action == "add_anchor":
                target = conn.execute(
                    "SELECT * FROM residents WHERE resident_uuid = ?",
                    (target_id,),
                ).fetchone()
                if target is None:
                    raise KeyError(f"unknown resident UUID: {target_id}")
                if str(target["normalized_name"]) != normalized:
                    raise EnrollmentProposalConflict(
                        "adding an anchor to an existing resident requires its current display name"
                    )
                if name_owner is None or str(name_owner["resident_uuid"]) != target_id:
                    raise EnrollmentProposalConflict(
                        "normalized display name does not belong to the explicit update resident"
                    )
                if sid is not None and int(target["compatibility_sid"]) != sid:
                    raise EnrollmentProposalConflict(
                        "compatibility SID does not match the explicit update resident"
                    )
                if str(target["model_fingerprint"]) != fingerprint:
                    raise ModelProfileMismatch("resident model fingerprint mismatch")
                if int(target["embedding_dim"]) != dimension:
                    raise ModelProfileMismatch("resident embedding dimension mismatch")
            else:
                if name_owner is not None:
                    raise DuplicateResidentError(
                        "normalized display name already exists; provide update_resident_uuid explicitly"
                    )
                sid_owner = conn.execute(
                    "SELECT resident_uuid FROM residents WHERE compatibility_sid = ?",
                    (sid,),
                ).fetchone()
                if sid_owner is not None:
                    raise DuplicateResidentError(
                        f"compatibility SID already belongs to resident {sid_owner['resident_uuid']}"
                    )
            conn.execute(
                """
                INSERT INTO enrollment_proposals(
                    proposal_uuid, run_id, camera_id, tracker_id, frame_id,
                    observation_id, observation_quality, observation_evidence_json,
                    display_name, normalized_name,
                    compatibility_sid, target_resident_uuid,
                    proposed_resident_uuid, action, embedding,
                    model_fingerprint, embedding_dim, evidence_digest,
                    created_at, expires_at, state, confirmed_at,
                    result_resident_uuid, anchor_exemplar_uuid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'pending', NULL, NULL, NULL)
                """,
                (
                    proposal_id,
                    key.run_id,
                    key.camera_id,
                    key.tracker_id,
                    key.frame_id,
                    key.observation_id,
                    quality,
                    json.dumps(evidence, separators=(",", ":")),
                    name,
                    normalized,
                    sid,
                    target_id,
                    proposed_resident_id,
                    action,
                    blob,
                    fingerprint,
                    dimension,
                    digest,
                    timestamp,
                    timestamp + ttl,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enrollment_proposals WHERE proposal_uuid = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise IdentityStoreError("enrollment proposal disappeared after insert")
        return EnrollmentProposalResult(self._proposal_from_row(row), False)

    def get_enrollment_proposal(self, proposal_uuid: str) -> EnrollmentProposalRecord:
        proposal_id = _canonical_uuid(proposal_uuid, "proposal_uuid")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM enrollment_proposals WHERE proposal_uuid = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown enrollment proposal UUID: {proposal_uuid}")
        return self._proposal_from_row(row)

    def find_enrollment_proposal_by_key(
        self,
        key: EnrollmentObservationKey,
    ) -> Optional[EnrollmentProposalRecord]:
        if not isinstance(key, EnrollmentObservationKey):
            raise TypeError("key must be an EnrollmentObservationKey")
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM enrollment_proposals
                WHERE run_id = ? AND camera_id = ? AND tracker_id = ?
                  AND frame_id = ? AND observation_id = ?
                """,
                (
                    key.run_id,
                    key.camera_id,
                    key.tracker_id,
                    key.frame_id,
                    key.observation_id,
                ),
            ).fetchone()
        return self._proposal_from_row(row) if row is not None else None

    def list_enrollment_proposals(
        self,
        *,
        limit: int = 100,
    ) -> Tuple[EnrollmentProposalRecord, ...]:
        count = int(limit)
        if count < 1 or count > 1000:
            raise ValueError("limit must be within [1, 1000]")
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM enrollment_proposals
                ORDER BY created_at DESC, proposal_uuid DESC
                LIMIT ?
                """,
                (count,),
            ).fetchall()
        return tuple(self._proposal_from_row(row) for row in rows)

    @staticmethod
    def _proposal_from_row(row: sqlite3.Row) -> EnrollmentProposalRecord:
        evidence = strict_json_loads(
            str(row["observation_evidence_json"]),
            label="stored enrollment proposal evidence",
        )
        if not isinstance(evidence, list):
            raise IdentityStoreError(
                "stored enrollment proposal evidence must be a JSON list"
            )
        return EnrollmentProposalRecord(
            proposal_uuid=str(row["proposal_uuid"]),
            key=EnrollmentObservationKey(
                run_id=str(row["run_id"]),
                camera_id=str(row["camera_id"]),
                tracker_id=str(row["tracker_id"]),
                frame_id=int(row["frame_id"]),
                observation_id=str(row["observation_id"]),
            ),
            observation_quality=float(row["observation_quality"]),
            observation_evidence=tuple(str(item) for item in evidence),
            display_name=str(row["display_name"]),
            normalized_name=str(row["normalized_name"]),
            compatibility_sid=(
                int(row["compatibility_sid"])
                if row["compatibility_sid"] is not None
                else None
            ),
            target_resident_uuid=(
                str(row["target_resident_uuid"])
                if row["target_resident_uuid"] is not None
                else None
            ),
            proposed_resident_uuid=(
                str(row["proposed_resident_uuid"])
                if row["proposed_resident_uuid"] is not None
                else None
            ),
            action=str(row["action"]),
            model_fingerprint=str(row["model_fingerprint"]),
            embedding_dim=int(row["embedding_dim"]),
            evidence_digest=str(row["evidence_digest"]),
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
            state=str(row["state"]),
            confirmed_at=(
                float(row["confirmed_at"]) if row["confirmed_at"] is not None else None
            ),
            result_resident_uuid=(
                str(row["result_resident_uuid"])
                if row["result_resident_uuid"] is not None
                else None
            ),
            anchor_exemplar_uuid=(
                str(row["anchor_exemplar_uuid"])
                if row["anchor_exemplar_uuid"] is not None
                else None
            ),
        )

    def confirm_enrollment_proposal(
        self,
        proposal_uuid: str,
        *,
        expected_key: EnrollmentObservationKey,
        evidence_digest: str,
        now: Optional[float] = None,
    ) -> EnrollmentConfirmationResult:
        proposal_id = _canonical_uuid(proposal_uuid, "proposal_uuid")
        if not isinstance(expected_key, EnrollmentObservationKey):
            raise TypeError("expected_key must be an EnrollmentObservationKey")
        expected_digest = _require_text(evidence_digest, "evidence_digest")
        timestamp = float(time.time() if now is None else now)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM enrollment_proposals WHERE proposal_uuid = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown enrollment proposal UUID: {proposal_uuid}")
            proposal = self._proposal_from_row(row)
            if (
                proposal.key != expected_key
                or proposal.evidence_digest != expected_digest
            ):
                raise EnrollmentProposalConflict(
                    "confirmation does not match the proposal's exact observation evidence"
                )
            if proposal.state == "confirmed":
                return self._confirmed_result_from_rows(conn, proposal, idempotent=True)
            if proposal.state != "pending":
                raise EnrollmentProposalConflict(
                    f"enrollment proposal is {proposal.state}, not pending"
                )
            if proposal.expires_at <= timestamp:
                raise EnrollmentProposalExpired("enrollment proposal has expired")

            if proposal.action == "create_resident":
                resident_id = str(proposal.proposed_resident_uuid)
                if (
                    not proposal.proposed_resident_uuid
                    or proposal.compatibility_sid is None
                ):
                    raise IdentityStoreError(
                        "new-resident proposal is structurally incomplete"
                    )
                conflict = conn.execute(
                    """
                    SELECT resident_uuid FROM residents
                    WHERE normalized_name = ? OR compatibility_sid = ?
                       OR resident_uuid = ?
                    """,
                    (
                        proposal.normalized_name,
                        proposal.compatibility_sid,
                        resident_id,
                    ),
                ).fetchone()
                if conflict is not None:
                    raise DuplicateResidentError(
                        f"resident state changed after proposal; conflict with {conflict['resident_uuid']}"
                    )
                conn.execute(
                    """
                    INSERT INTO residents(
                        resident_uuid, display_name, normalized_name,
                        compatibility_sid, model_fingerprint, embedding_dim,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resident_id,
                        proposal.display_name,
                        proposal.normalized_name,
                        proposal.compatibility_sid,
                        proposal.model_fingerprint,
                        proposal.embedding_dim,
                        timestamp,
                        timestamp,
                    ),
                )
            elif proposal.action == "add_anchor":
                if not proposal.target_resident_uuid:
                    raise IdentityStoreError(
                        "add-anchor proposal is structurally incomplete"
                    )
                resident_id = proposal.target_resident_uuid
                resident = conn.execute(
                    "SELECT * FROM residents WHERE resident_uuid = ?",
                    (resident_id,),
                ).fetchone()
                if resident is None:
                    raise KeyError(f"unknown resident UUID: {resident_id}")
                if str(resident["normalized_name"]) != proposal.normalized_name:
                    raise EnrollmentProposalConflict(
                        "resident display name changed after enrollment proposal"
                    )
                if str(resident["model_fingerprint"]) != proposal.model_fingerprint:
                    raise ModelProfileMismatch("resident model fingerprint changed")
                if int(resident["embedding_dim"]) != proposal.embedding_dim:
                    raise ModelProfileMismatch("resident embedding dimension changed")
            else:
                raise IdentityStoreError(
                    f"unsupported enrollment action {proposal.action!r}"
                )

            anchor_id = str(uuid.uuid5(uuid.UUID(proposal_id), "enrollment-anchor"))
            observation = f"enrollment:{proposal.evidence_digest}"
            conn.execute(
                """
                INSERT INTO resident_exemplars(
                    exemplar_uuid, resident_uuid, role, embedding,
                    model_fingerprint, embedding_dim, observation_id,
                    independence_key, created_at, promoted_at, expires_at
                ) VALUES (?, ?, 'enrollment_anchor', ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    anchor_id,
                    resident_id,
                    bytes(row["embedding"]),
                    proposal.model_fingerprint,
                    proposal.embedding_dim,
                    observation,
                    observation,
                    timestamp,
                    timestamp,
                ),
            )
            conn.execute(
                """
                UPDATE enrollment_proposals
                SET state = 'confirmed', confirmed_at = ?,
                    result_resident_uuid = ?, anchor_exemplar_uuid = ?
                WHERE proposal_uuid = ?
                """,
                (timestamp, resident_id, anchor_id, proposal_id),
            )
            confirmed_row = conn.execute(
                "SELECT * FROM enrollment_proposals WHERE proposal_uuid = ?",
                (proposal_id,),
            ).fetchone()
            if confirmed_row is None:
                raise IdentityStoreError("confirmed enrollment proposal disappeared")
            confirmed = self._proposal_from_row(confirmed_row)
            return self._confirmed_result_from_rows(conn, confirmed, idempotent=False)

    def _confirmed_result_from_rows(
        self,
        conn: sqlite3.Connection,
        proposal: EnrollmentProposalRecord,
        *,
        idempotent: bool,
    ) -> EnrollmentConfirmationResult:
        if not proposal.result_resident_uuid or not proposal.anchor_exemplar_uuid:
            raise IdentityStoreError(
                "confirmed enrollment proposal lacks durable result links"
            )
        resident_row = conn.execute(
            "SELECT * FROM residents WHERE resident_uuid = ?",
            (proposal.result_resident_uuid,),
        ).fetchone()
        anchor_row = conn.execute(
            """
            SELECT e.*, 0 AS corroboration_count
            FROM resident_exemplars e WHERE exemplar_uuid = ?
            """,
            (proposal.anchor_exemplar_uuid,),
        ).fetchone()
        if resident_row is None or anchor_row is None:
            raise IdentityStoreError(
                "confirmed enrollment result is missing from the store"
            )
        return EnrollmentConfirmationResult(
            proposal=proposal,
            resident=self._resident_from_row(resident_row),
            anchor=self._exemplar_from_row(anchor_row),
            idempotent=idempotent,
        )

    def purge_expired(self, *, now: Optional[float] = None) -> PurgeResult:
        timestamp = float(time.time() if now is None else now)
        with self.transaction() as conn:
            expired_visitors = conn.execute(
                "SELECT session_uuid, slot FROM visitor_sessions WHERE expires_at <= ?",
                (timestamp,),
            ).fetchall()
            for row in expired_visitors:
                conn.execute(
                    """
                    UPDATE visitor_slots
                    SET current_session_uuid = NULL, last_released_at = ?
                    WHERE slot = ? AND current_session_uuid = ?
                    """,
                    (timestamp, int(row["slot"]), str(row["session_uuid"])),
                )
            visitor_deleted = conn.execute(
                "DELETE FROM visitor_sessions WHERE expires_at <= ?",
                (timestamp,),
            ).rowcount
            provisional_deleted = conn.execute(
                "DELETE FROM provisional_sessions WHERE expires_at <= ?",
                (timestamp,),
            ).rowcount
            quarantine_deleted = conn.execute(
                """
                DELETE FROM resident_exemplars
                WHERE role = 'quarantine' AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (timestamp,),
            ).rowcount
            proposal_deleted = conn.execute(
                """
                DELETE FROM enrollment_proposals
                WHERE state = 'pending' AND expires_at <= ?
                """,
                (timestamp,),
            ).rowcount
        return PurgeResult(
            visitor_sessions=int(visitor_deleted),
            provisional_sessions=int(provisional_deleted),
            quarantine_exemplars=int(quarantine_deleted),
            enrollment_proposals=int(proposal_deleted),
        )

    def health(self, *, now: Optional[float] = None) -> IdentityHealth:
        timestamp = float(time.time() if now is None else now)
        with self._lock:
            resident_count = int(
                self._conn.execute("SELECT COUNT(*) FROM residents").fetchone()[0]
            )
            anchor_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM resident_exemplars WHERE role = 'enrollment_anchor'"
                ).fetchone()[0]
            )
            adaptive_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM resident_exemplars WHERE role = 'adaptive'"
                ).fetchone()[0]
            )
            quarantine_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM resident_exemplars WHERE role = 'quarantine'"
                ).fetchone()[0]
            )
            with_anchors = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM residents r
                    WHERE EXISTS (
                        SELECT 1 FROM resident_exemplars e
                        WHERE e.resident_uuid = r.resident_uuid
                          AND e.role = 'enrollment_anchor'
                    )
                    """
                ).fetchone()[0]
            )
            active_visitors = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM visitor_sessions
                    WHERE state = 'active' AND expires_at > ?
                    """,
                    (timestamp,),
                ).fetchone()[0]
            )
            visitor_slots = int(
                self._conn.execute("SELECT COUNT(*) FROM visitor_slots").fetchone()[0]
            )
            active_provisional = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM provisional_sessions WHERE expires_at > ?",
                    (timestamp,),
                ).fetchone()[0]
            )
            visitor_exemplars = int(
                self._conn.execute("SELECT COUNT(*) FROM visitor_exemplars").fetchone()[
                    0
                ]
            )
            pending_proposals = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM enrollment_proposals
                    WHERE state = 'pending' AND expires_at > ?
                    """,
                    (timestamp,),
                ).fetchone()[0]
            )
            confirmed_proposals = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM enrollment_proposals WHERE state = 'confirmed'"
                ).fetchone()[0]
            )
            profiles = tuple(
                (str(row[0]), int(row[1]), int(row[2]))
                for row in self._conn.execute(
                    """
                    SELECT model_fingerprint, embedding_dim, COUNT(*)
                    FROM residents
                    GROUP BY model_fingerprint, embedding_dim
                    ORDER BY model_fingerprint, embedding_dim
                    """
                ).fetchall()
            )
        return IdentityHealth(
            schema_version=self.schema_version,
            resident_count=resident_count,
            residents_with_anchors=with_anchors,
            residents_without_anchors=resident_count - with_anchors,
            enrollment_anchor_count=anchor_count,
            adaptive_exemplar_count=adaptive_count,
            quarantine_exemplar_count=quarantine_count,
            active_visitor_sessions=active_visitors,
            visitor_slot_count=visitor_slots,
            active_provisional_sessions=active_provisional,
            visitor_exemplar_count=visitor_exemplars,
            pending_enrollment_proposals=pending_proposals,
            confirmed_enrollment_proposals=confirmed_proposals,
            model_profiles=profiles,
        )

    def legacy_migration_applied(self, migration_key: str, source_digest: str) -> bool:
        key = _require_text(migration_key, "migration_key")
        digest = _require_text(source_digest, "source_digest")
        with self._lock:
            row = self._conn.execute(
                "SELECT source_digest FROM legacy_migrations WHERE migration_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return False
        if str(row["source_digest"]) != digest:
            raise MigrationConflict(
                "legacy source digest differs from recorded migration"
            )
        return True

    def import_legacy_residents_once(
        self,
        *,
        migration_key: str,
        source_digest: str,
        residents: Sequence[LegacyResidentImport],
        now: Optional[float] = None,
    ) -> LegacyImportResult:
        key = _require_text(migration_key, "migration_key")
        digest = _require_text(source_digest, "source_digest")
        timestamp = float(time.time() if now is None else now)
        with self.transaction() as conn:
            prior = conn.execute(
                "SELECT source_digest, result_json FROM legacy_migrations WHERE migration_key = ?",
                (key,),
            ).fetchone()
            if prior is not None:
                if str(prior["source_digest"]) != digest:
                    raise MigrationConflict(
                        "legacy source changed after migration was recorded"
                    )
                payload = strict_json_loads(
                    str(prior["result_json"]),
                    label="stored legacy identity migration result",
                )
                if not isinstance(payload, Mapping):
                    raise MigrationConflict(
                        "stored legacy identity migration result must be a JSON object"
                    )
                return LegacyImportResult(
                    idempotent=True,
                    residents_created=int(payload.get("residents_created", 0)),
                    residents_existing=int(payload.get("residents_existing", 0)),
                    anchors_created=int(payload.get("anchors_created", 0)),
                )

            validated = []
            seen_uuids: set[str] = set()
            seen_names: set[str] = set()
            seen_sids: set[int] = set()
            for item in residents:
                resident_id = _canonical_uuid(item.resident_uuid, "resident_uuid")
                normalized = normalize_display_name(item.display_name)
                sid = int(item.compatibility_sid)
                fingerprint = _require_text(item.model_fingerprint, "model_fingerprint")
                dimension = int(item.embedding_dim)
                if sid <= 0 or dimension <= 0:
                    raise MigrationConflict(
                        "legacy compatibility SID and embedding dimension must be positive"
                    )
                if resident_id in seen_uuids:
                    raise MigrationConflict(
                        f"duplicate legacy resident UUID {resident_id}"
                    )
                if normalized in seen_names:
                    raise MigrationConflict(
                        f"duplicate normalized legacy display name {normalized!r}"
                    )
                if sid in seen_sids:
                    raise MigrationConflict(f"duplicate legacy compatibility SID {sid}")
                seen_uuids.add(resident_id)
                seen_names.add(normalized)
                seen_sids.add(sid)
                validated.append(
                    (item, resident_id, normalized, sid, fingerprint, dimension)
                )

            created = 0
            existing = 0
            anchors = 0
            for item, resident_id, normalized, sid, fingerprint, dimension in sorted(
                validated,
                key=lambda row: (row[3], row[1]),
            ):
                resident_row = conn.execute(
                    "SELECT * FROM residents WHERE resident_uuid = ?",
                    (resident_id,),
                ).fetchone()
                if resident_row is None:
                    conflict = conn.execute(
                        """
                        SELECT resident_uuid FROM residents
                        WHERE normalized_name = ? OR compatibility_sid = ?
                        """,
                        (normalized, sid),
                    ).fetchone()
                    if conflict is not None:
                        raise MigrationConflict(
                            f"legacy resident conflicts with existing resident {conflict['resident_uuid']}"
                        )
                    conn.execute(
                        """
                        INSERT INTO residents(
                            resident_uuid, display_name, normalized_name,
                            compatibility_sid, model_fingerprint, embedding_dim,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            resident_id,
                            " ".join(item.display_name.split()),
                            normalized,
                            sid,
                            fingerprint,
                            dimension,
                            timestamp,
                            timestamp,
                        ),
                    )
                    created += 1
                else:
                    expected = (
                        normalized,
                        sid,
                        fingerprint,
                        dimension,
                    )
                    actual = (
                        str(resident_row["normalized_name"]),
                        int(resident_row["compatibility_sid"]),
                        str(resident_row["model_fingerprint"]),
                        int(resident_row["embedding_dim"]),
                    )
                    if actual != expected:
                        raise MigrationConflict(
                            f"legacy UUID {resident_id} exists with different identity data"
                        )
                    existing += 1

                for index, vector in enumerate(item.anchor_vectors):
                    blob = _encode_vector(vector, dimension)
                    exemplar_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{key}:{digest}:{resident_id}:anchor:{index}",
                        )
                    )
                    observation_id = f"legacy:{digest}:{resident_id}:{index}"
                    found = conn.execute(
                        "SELECT 1 FROM resident_exemplars WHERE exemplar_uuid = ?",
                        (exemplar_id,),
                    ).fetchone()
                    if found is not None:
                        continue
                    conn.execute(
                        """
                        INSERT INTO resident_exemplars(
                            exemplar_uuid, resident_uuid, role, embedding,
                            model_fingerprint, embedding_dim, observation_id,
                            independence_key, created_at, promoted_at, expires_at
                        ) VALUES (?, ?, 'enrollment_anchor', ?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            exemplar_id,
                            resident_id,
                            blob,
                            fingerprint,
                            dimension,
                            observation_id,
                            observation_id,
                            timestamp,
                            timestamp,
                        ),
                    )
                    anchors += 1

            payload = {
                "residents_created": created,
                "residents_existing": existing,
                "anchors_created": anchors,
            }
            conn.execute(
                """
                INSERT INTO legacy_migrations(migration_key, source_digest, applied_at, result_json)
                VALUES (?, ?, ?, ?)
                """,
                (key, digest, timestamp, json.dumps(payload, sort_keys=True)),
            )
        return LegacyImportResult(False, created, existing, anchors)
