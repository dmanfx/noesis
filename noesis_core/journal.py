from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Iterable, Mapping, NamedTuple

from noesis_core.private_paths import (
    PrivatePathError,
    prepare_private_writable_file,
    validate_private_file,
)
from noesis_core.replay import (
    ReplayValidationError,
    validate_contract_payload,
    validate_stored_contract_payload,
)
from noesis_core.strict_json import StrictJSONError, strict_json_loads


class ContractJournalError(RuntimeError):
    pass


@dataclass(frozen=True)
class JournalRecord:
    sequence: int
    recorded_at_us: int
    payload: Mapping[str, Any]
    previous_sha256: str
    record_sha256: str


class AsyncJournalAdmissionReceipt(NamedTuple):
    """Bounded non-durable admission receipt for optional persistence."""

    payload_count: int
    pending_batches: int
    max_pending_batches: int


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _record_sha256(
    *,
    sequence: int,
    recorded_at_us: int,
    payload_json: bytes,
    previous_sha256: str,
) -> str:
    """Hash one journal core without serializing its payload a second time.

    The journal core has a fixed, lexicographically sorted key order.  The
    payload was already canonicalized for its SQLite column, so splice that
    exact byte sequence into the equivalent canonical wrapper and encode only
    the scalar hash field.  ``previous_sha256`` is deliberately JSON-encoded
    rather than interpolated so malformed/corrupted state retains the same
    escaping behavior as :func:`_canonical_bytes`.
    """

    previous_json = json.dumps(
        str(previous_sha256),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    encoded = (
        b'{"payload":'
        + payload_json
        + b',"previous_sha256":'
        + previous_json
        + b',"recorded_at_us":'
        + str(int(recorded_at_us)).encode("ascii")
        + b',"sequence":'
        + str(int(sequence)).encode("ascii")
        + b"}"
    )
    return hashlib.sha256(encoded).hexdigest()


def _normalized_schema_sql(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class ContractJournal:
    """Bounded, integrity-chained SQLite WAL for canonical contract output."""

    LEGACY_SCHEMA_VERSION = 1
    SCHEMA_VERSION = 2
    GENESIS_SHA256 = "0" * 64
    SQLITE_WAL_SYNCHRONOUS_FULL = 2
    SQLITE_WAL_AUTOCHECKPOINT_PAGES = 1_000
    _EXPECTED_SCHEMA_SQL = {
        "journal_state": _normalized_schema_sql(
            """
            CREATE TABLE journal_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                next_sequence INTEGER NOT NULL,
                anchor_previous_sha256 TEXT NOT NULL
            )
            """
        ),
        "records": _normalized_schema_sql(
            """
            CREATE TABLE records (
                sequence INTEGER PRIMARY KEY,
                recorded_at_us INTEGER NOT NULL,
                contract TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_sha256 TEXT NOT NULL,
                record_sha256 TEXT NOT NULL UNIQUE
            )
            """
        ),
        "records_recorded_at": _normalized_schema_sql(
            "CREATE INDEX records_recorded_at "
            "ON records(recorded_at_us, sequence)"
        ),
    }
    _EXPECTED_TABLE_COLUMNS = {
        "journal_state": (
            (0, "singleton", "INTEGER", 0, None, 1),
            (1, "next_sequence", "INTEGER", 1, None, 0),
            (2, "anchor_previous_sha256", "TEXT", 1, None, 0),
        ),
        "records": (
            (0, "sequence", "INTEGER", 0, None, 1),
            (1, "recorded_at_us", "INTEGER", 1, None, 0),
            (2, "contract", "TEXT", 1, None, 0),
            (3, "payload_json", "TEXT", 1, None, 0),
            (4, "previous_sha256", "TEXT", 1, None, 0),
            (5, "record_sha256", "TEXT", 1, None, 0),
        ),
    }

    def __init__(
        self,
        path: str | Path,
        *,
        max_records: int = 10_000,
        max_age_us: int = 86_400_000_000,
    ) -> None:
        if int(max_records) <= 0 or int(max_age_us) <= 0:
            raise ValueError("journal retention limits must be positive")
        configured_path = Path(path).expanduser()
        try:
            self.path = prepare_private_writable_file(
                configured_path,
                label="contract journal",
            )
        except PrivatePathError as exc:
            raise ContractJournalError(str(exc)) from exc
        self.max_records = int(max_records)
        self.max_age_us = int(max_age_us)
        self._lock = threading.RLock()
        self._closed = False
        self._close_failure: ContractJournalError | None = None
        self._connection_handle: sqlite3.Connection | None = None
        self._preflight_validated = False
        self._preflight_version: int | None = None
        database = self.path.lstat()
        parent = self.path.parent.lstat()
        self._database_identity = (int(database.st_dev), int(database.st_ino))
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._sidecar_path_cache = (
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        )
        self._sidecar_identities: dict[Path, tuple[int, int]] = {}
        try:
            self._initialize()
        except BaseException as exc:
            connection = self._connection_handle
            self._connection_handle = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass
            if isinstance(exc, sqlite3.Error):
                raise ContractJournalError(
                    "contract journal SQLite database cannot be validated"
                ) from exc
            raise

    @property
    def closed(self) -> bool:
        with self._lock:
            return bool(self._closed)

    def _raise_if_closed(self) -> None:
        if not self._closed:
            return
        if self._close_failure is not None:
            raise self._close_failure
        raise ContractJournalError("contract journal is closed")

    def _sidecar_paths(self) -> tuple[Path, Path]:
        return self._sidecar_path_cache

    def _validate_main_binding(self) -> None:
        """Require the configured path to name the exact opened database."""

        try:
            validated = validate_private_file(
                self.path,
                label="contract journal",
            )
            # Re-stat after path-policy validation.  The first stat proves the
            # named file was private; this second one is the binding receipt
            # compared with the inode SQLite opened.  Reusing the validation
            # stat would miss a rename/replacement performed between those two
            # operations.
            database = validated.lstat()
            parent = validated.parent.lstat()
            if (
                (int(database.st_dev), int(database.st_ino))
                != self._database_identity
                or not stat.S_ISREG(database.st_mode)
                or database.st_uid != os.geteuid()
                or stat.S_IMODE(database.st_mode) != 0o600
                or database.st_nlink != 1
            ):
                raise PrivatePathError(
                    "contract journal database inode changed or became unsafe while open"
                )
            if (
                (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
                or not stat.S_ISDIR(parent.st_mode)
                or parent.st_uid != os.geteuid()
                or stat.S_IMODE(parent.st_mode) != 0o700
            ):
                raise PrivatePathError(
                    "contract journal parent changed or became unsafe while open"
                )
        except (OSError, PrivatePathError) as exc:
            raise ContractJournalError(str(exc)) from exc

    def _validate_storage_files(self, *, bind_sidecars: bool = True) -> None:
        """Bind the live connection to one exact private SQLite file cohort."""

        self._validate_main_binding()
        try:
            for sidecar in self._sidecar_paths():
                if not os.path.lexists(sidecar):
                    if bind_sidecars and sidecar in self._sidecar_identities:
                        raise PrivatePathError(
                            f"contract journal sidecar disappeared while open: {sidecar.name}"
                        )
                    continue
                validated_sidecar = validate_private_file(
                    sidecar,
                    label=f"contract journal sidecar {sidecar.name}",
                )
                info = validated_sidecar.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_nlink != 1
                ):
                    raise PrivatePathError(
                        "contract journal sidecar changed or became unsafe while open: "
                        f"{sidecar.name}"
                    )
                identity = (int(info.st_dev), int(info.st_ino))
                expected = self._sidecar_identities.get(sidecar)
                if bind_sidecars and expected is None:
                    self._sidecar_identities[sidecar] = identity
                elif bind_sidecars and identity != expected:
                    raise PrivatePathError(
                        f"contract journal sidecar inode changed while open: {sidecar.name}"
                    )
        except (OSError, PrivatePathError) as exc:
            raise ContractJournalError(str(exc)) from exc

    def _connect(self) -> sqlite3.Connection:
        self._raise_if_closed()
        if self._connection_handle is not None:
            return self._connection_handle
        # A crash-recovery open may legitimately recreate stale WAL/SHM files.
        # Validate their privacy before SQLite sees them, then bind the exact
        # recovered sidecar inodes after the connection is configured.
        self._validate_storage_files(bind_sidecars=False)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=10.0,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            preflight_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if preflight_version not in {
                0,
                self.LEGACY_SCHEMA_VERSION,
                self.SCHEMA_VERSION,
            }:
                raise ContractJournalError(
                    f"unsupported contract journal schema: {preflight_version}"
                )
            if preflight_version == 0:
                foreign_objects = connection.execute(
                    """
                    SELECT type, name FROM sqlite_schema
                    WHERE name NOT LIKE 'sqlite_%'
                    ORDER BY type, name
                    """
                ).fetchall()
                if foreign_objects:
                    raise ContractJournalError(
                        "uninitialized contract journal database is not empty"
                    )
            else:
                # Reject schema or retained-history corruption before changing
                # any persistent database setting on a restart.
                self._validate_schema(connection)
                self._validated_records(connection)
                self._preflight_validated = True
            existing_mode_row = connection.execute("PRAGMA journal_mode").fetchone()
            existing_mode = (
                str(existing_mode_row[0]).strip().lower()
                if existing_mode_row
                else ""
            )
            if preflight_version == self.SCHEMA_VERSION and existing_mode != "wal":
                raise ContractJournalError(
                    "contract journal storage contract v2 requires existing WAL "
                    f"mode; found {existing_mode or 'no mode'}"
                )
            if preflight_version == self.LEGACY_SCHEMA_VERSION and existing_mode not in {
                "delete",
                "wal",
            }:
                raise ContractJournalError(
                    "legacy contract journal has unsupported persistent mode: "
                    f"{existing_mode or 'no mode'}"
                )
            if preflight_version == 0 and existing_mode not in {"delete", "wal"}:
                raise ContractJournalError(
                    "uninitialized contract journal has unsupported persistent mode: "
                    f"{existing_mode or 'no mode'}"
                )
            if existing_mode == "wal":
                mode = existing_mode
            else:
                mode_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
                mode = str(mode_row[0]).strip().lower() if mode_row else ""
            if mode != "wal":
                raise ContractJournalError(
                    "contract journal requires SQLite WAL mode; "
                    f"database selected {mode or 'no mode'}"
                )
            connection.execute("PRAGMA synchronous = FULL")
            synchronous_row = connection.execute("PRAGMA synchronous").fetchone()
            synchronous = int(synchronous_row[0]) if synchronous_row else -1
            if synchronous != self.SQLITE_WAL_SYNCHRONOUS_FULL:
                raise ContractJournalError(
                    "contract journal requires SQLite synchronous=FULL; "
                    f"database selected {synchronous}"
                )
            checkpoint_row = connection.execute(
                f"PRAGMA wal_autocheckpoint = {self.SQLITE_WAL_AUTOCHECKPOINT_PAGES}"
            ).fetchone()
            checkpoint_pages = int(checkpoint_row[0]) if checkpoint_row else -1
            if checkpoint_pages != self.SQLITE_WAL_AUTOCHECKPOINT_PAGES:
                raise ContractJournalError(
                    "contract journal could not configure the exact WAL checkpoint "
                    f"interval: expected={self.SQLITE_WAL_AUTOCHECKPOINT_PAGES} "
                    f"actual={checkpoint_pages}"
                )
            self._preflight_version = preflight_version
            self._connection_handle = connection
            self._sidecar_identities.clear()
            self._validate_storage_files()
            self._validate_connection_configuration(connection)
            return connection
        except BaseException:
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass
            self._connection_handle = None
            raise

    def _validate_connection_configuration(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        mode_row = connection.execute("PRAGMA journal_mode").fetchone()
        mode = str(mode_row[0]).strip().lower() if mode_row else ""
        synchronous_row = connection.execute("PRAGMA synchronous").fetchone()
        synchronous = int(synchronous_row[0]) if synchronous_row else -1
        checkpoint_row = connection.execute("PRAGMA wal_autocheckpoint").fetchone()
        checkpoint_pages = int(checkpoint_row[0]) if checkpoint_row else -1
        if mode != "wal":
            raise ContractJournalError(
                "contract journal SQLite mode changed while open: "
                f"expected=wal actual={mode or 'no mode'}"
            )
        if synchronous != self.SQLITE_WAL_SYNCHRONOUS_FULL:
            raise ContractJournalError(
                "contract journal SQLite synchronous policy changed while open: "
                f"expected={self.SQLITE_WAL_SYNCHRONOUS_FULL} actual={synchronous}"
            )
        if checkpoint_pages != self.SQLITE_WAL_AUTOCHECKPOINT_PAGES:
            raise ContractJournalError(
                "contract journal SQLite checkpoint interval changed while open: "
                f"expected={self.SQLITE_WAL_AUTOCHECKPOINT_PAGES} "
                f"actual={checkpoint_pages}"
            )

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        quick_check = tuple(
            str(row[0])
            for row in connection.execute("PRAGMA quick_check").fetchall()
        )
        if quick_check != ("ok",):
            raise ContractJournalError(
                "contract journal SQLite integrity check failed"
            )
        objects = connection.execute(
            """
            SELECT type, name, tbl_name, sql FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        observed_objects = tuple(
            (str(row["type"]), str(row["name"]), str(row["tbl_name"]))
            for row in objects
        )
        expected_objects = (
            ("index", "records_recorded_at", "records"),
            ("table", "journal_state", "journal_state"),
            ("table", "records", "records"),
        )
        if observed_objects != expected_objects:
            raise ContractJournalError(
                "contract journal schema objects do not match the storage contract"
            )
        for row in objects:
            name = str(row["name"])
            if _normalized_schema_sql(row["sql"]) != self._EXPECTED_SCHEMA_SQL[name]:
                raise ContractJournalError(
                    "contract journal schema SQL does not match the storage "
                    f"contract: {name}"
                )
        for table, expected_columns in self._EXPECTED_TABLE_COLUMNS.items():
            columns = tuple(
                tuple(row)
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
            if columns != expected_columns:
                raise ContractJournalError(
                    "contract journal table layout does not match the storage "
                    f"contract: {table}"
                )
        state_indexes = tuple(
            tuple(row)
            for row in connection.execute("PRAGMA index_list(journal_state)").fetchall()
        )
        if state_indexes:
            raise ContractJournalError(
                "contract journal state table has unexpected indexes"
            )
        record_indexes = {
            (str(row[1]), int(row[2]), str(row[3]), int(row[4]))
            for row in connection.execute("PRAGMA index_list(records)").fetchall()
        }
        if record_indexes != {
            ("records_recorded_at", 0, "c", 0),
            ("sqlite_autoindex_records_1", 1, "u", 0),
        }:
            raise ContractJournalError(
                "contract journal record indexes do not match the storage contract"
            )
        recorded_at_index = tuple(
            tuple(row)
            for row in connection.execute(
                "PRAGMA index_info(records_recorded_at)"
            ).fetchall()
        )
        checksum_index = tuple(
            tuple(row)
            for row in connection.execute(
                "PRAGMA index_info(sqlite_autoindex_records_1)"
            ).fetchall()
        )
        if recorded_at_index != (
            (0, 1, "recorded_at_us"),
            (1, 0, "sequence"),
        ) or checksum_index != ((0, 5, "record_sha256"),):
            raise ContractJournalError(
                "contract journal index columns do not match the storage contract"
            )

    def _validated_records(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[JournalRecord, ...]:
        state_rows = connection.execute(
            """
            SELECT singleton, next_sequence, anchor_previous_sha256
            FROM journal_state ORDER BY singleton
            """
        ).fetchall()
        if len(state_rows) != 1:
            raise ContractJournalError(
                "contract journal must contain exactly one state row"
            )
        state = state_rows[0]
        if type(state["singleton"]) is not int or int(state["singleton"]) != 1:
            raise ContractJournalError("contract journal state singleton is invalid")
        if type(state["next_sequence"]) is not int:
            raise ContractJournalError("contract journal next sequence is invalid")
        next_sequence = int(state["next_sequence"])
        if next_sequence < 0:
            raise ContractJournalError("contract journal next sequence is invalid")
        anchor = state["anchor_previous_sha256"]
        if not _is_sha256(anchor):
            raise ContractJournalError("contract journal anchor checksum is invalid")

        rows = connection.execute(
            """
            SELECT sequence, recorded_at_us, contract, payload_json,
                   previous_sha256, record_sha256
            FROM records ORDER BY sequence ASC
            """
        ).fetchall()
        if len(rows) > self.max_records:
            raise ContractJournalError(
                "contract journal retained record count exceeds its configured limit"
            )
        first_sequence = next_sequence - len(rows)
        if first_sequence < 0:
            raise ContractJournalError(
                "contract journal next sequence precedes retained history"
            )
        if (first_sequence == 0) != (anchor == self.GENESIS_SHA256):
            raise ContractJournalError(
                "contract journal anchor does not match its retained-history boundary"
            )
        expected_previous = str(anchor)
        result: list[JournalRecord] = []
        for offset, row in enumerate(rows):
            expected_sequence = first_sequence + offset
            if type(row["sequence"]) is not int or int(row["sequence"]) != expected_sequence:
                raise ContractJournalError(
                    "contract journal retained sequences are not contiguous"
                )
            sequence = int(row["sequence"])
            if (
                type(row["recorded_at_us"]) is not int
                or int(row["recorded_at_us"]) <= 0
            ):
                raise ContractJournalError(
                    f"journal timestamp is invalid at sequence {sequence}"
                )
            recorded_at_us = int(row["recorded_at_us"])
            raw_payload_value = row["payload_json"]
            if not isinstance(raw_payload_value, str):
                raise ContractJournalError(
                    f"journal payload JSON is invalid at sequence {sequence}"
                )
            raw_payload = raw_payload_value
            try:
                payload = strict_json_loads(
                    raw_payload,
                    label="stored journal payload",
                )
            except StrictJSONError as exc:
                raise ContractJournalError(
                    f"journal payload JSON is ambiguous or invalid at sequence {sequence}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ContractJournalError(
                    f"journal payload is not an object at sequence {sequence}"
                )
            if raw_payload.encode("utf-8") != _canonical_bytes(payload):
                raise ContractJournalError(
                    f"journal payload is not canonical JSON at sequence {sequence}"
                )
            previous = row["previous_sha256"]
            if not _is_sha256(previous) or previous != expected_previous:
                raise ContractJournalError(
                    f"journal hash-chain mismatch at sequence {sequence}"
                )
            stored_sha = row["record_sha256"]
            if not _is_sha256(stored_sha):
                raise ContractJournalError(
                    f"journal record checksum is invalid at sequence {sequence}"
                )
            core = {
                "sequence": sequence,
                "recorded_at_us": recorded_at_us,
                "payload": payload,
                "previous_sha256": previous,
            }
            expected_sha = _sha256(core)
            if stored_sha != expected_sha:
                raise ContractJournalError(
                    f"journal record checksum mismatch at sequence {sequence}"
                )
            stored_contract = row["contract"]
            if (
                not isinstance(stored_contract, str)
                or payload.get("contract") != stored_contract
            ):
                raise ContractJournalError(
                    f"journal contract column mismatch at sequence {sequence}"
                )
            try:
                validated = validate_stored_contract_payload(payload)
            except ReplayValidationError as exc:
                raise ContractJournalError(
                    f"journal payload contract is invalid at sequence {sequence}"
                ) from exc
            if _canonical_bytes(validated) != raw_payload.encode("utf-8"):
                raise ContractJournalError(
                    f"journal payload is not normalized at sequence {sequence}"
                )
            result.append(
                JournalRecord(
                    sequence=sequence,
                    recorded_at_us=recorded_at_us,
                    payload=validated,
                    previous_sha256=previous,
                    record_sha256=expected_sha,
                )
            )
            expected_previous = expected_sha
        return tuple(result)

    @contextmanager
    def _connection(self):
        connection = self._connect()
        self._validate_storage_files()
        self._validate_connection_configuration(connection)
        try:
            yield connection
        finally:
            if self._connection_handle is connection and not self._closed:
                self._validate_connection_configuration(connection)
                self._validate_storage_files()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {
                0,
                self.LEGACY_SCHEMA_VERSION,
                self.SCHEMA_VERSION,
            }:
                raise ContractJournalError(
                    f"unsupported contract journal schema: {version}"
                )
            if version != self._preflight_version:
                raise ContractJournalError(
                    "contract journal schema version changed during admission"
                )
            if version == 0:
                connection.executescript(
                    f"""
                    BEGIN IMMEDIATE;
                    CREATE TABLE journal_state (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        next_sequence INTEGER NOT NULL,
                        anchor_previous_sha256 TEXT NOT NULL
                    );
                    INSERT INTO journal_state(singleton, next_sequence, anchor_previous_sha256)
                    VALUES (1, 0, '{self.GENESIS_SHA256}');
                    CREATE TABLE records (
                        sequence INTEGER PRIMARY KEY,
                        recorded_at_us INTEGER NOT NULL,
                        contract TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        previous_sha256 TEXT NOT NULL,
                        record_sha256 TEXT NOT NULL UNIQUE
                    );
                    CREATE INDEX records_recorded_at ON records(recorded_at_us, sequence);
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
                self._validate_schema(connection)
                self._validated_records(connection)
            elif version == self.LEGACY_SCHEMA_VERSION:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        f"PRAGMA user_version = {self.SCHEMA_VERSION}"
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
            elif not self._preflight_validated:
                self._validate_schema(connection)
                self._validated_records(connection)
            migrated_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if migrated_version != self.SCHEMA_VERSION:
                raise ContractJournalError(
                    "contract journal storage contract migration did not commit"
                )

    def append_many(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        recorded_at_us: int,
    ) -> tuple[JournalRecord, ...]:
        timestamp = int(recorded_at_us)
        if timestamp <= 0:
            raise ContractJournalError("recorded_at_us must be positive")
        return self.append_entries((payload, timestamp) for payload in payloads)

    def append_entries(
        self,
        entries: Iterable[tuple[Mapping[str, Any], int]],
    ) -> tuple[JournalRecord, ...]:
        with self._lock:
            self._raise_if_closed()
        try:
            validated = [
                (validate_contract_payload(payload), int(timestamp))
                for payload, timestamp in entries
            ]
        except ReplayValidationError as exc:
            raise ContractJournalError(str(exc)) from exc
        if any(timestamp <= 0 for _payload, timestamp in validated):
            raise ContractJournalError("recorded_at_us must be positive")
        if not validated:
            return ()
        if len(validated) > self.max_records:
            raise ContractJournalError(
                "contract journal append exceeds retained record capacity: "
                f"required={len(validated)} max_records={self.max_records}"
            )
        appended: list[JournalRecord] = []
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = connection.execute(
                    """
                    SELECT next_sequence, anchor_previous_sha256,
                           (SELECT record_sha256 FROM records
                            ORDER BY sequence DESC LIMIT 1)
                               AS last_record_sha256
                    FROM journal_state WHERE singleton = 1
                    """
                ).fetchone()
                if state is None:
                    raise ContractJournalError("contract journal state is missing")
                sequence = int(state["next_sequence"])
                append_start_sequence = sequence
                previous = (
                    str(state["last_record_sha256"])
                    if state["last_record_sha256"] is not None
                    else str(state["anchor_previous_sha256"])
                )
                encoded_validated = [
                    (payload, timestamp, _canonical_bytes(payload))
                    for payload, timestamp in validated
                ]
                for payload, timestamp, payload_json in encoded_validated:
                    record_sha = _record_sha256(
                        sequence=sequence,
                        recorded_at_us=timestamp,
                        payload_json=payload_json,
                        previous_sha256=previous,
                    )
                    connection.execute(
                        """
                        INSERT INTO records(
                            sequence, recorded_at_us, contract, payload_json,
                            previous_sha256, record_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sequence,
                            timestamp,
                            str(payload["contract"]),
                            payload_json.decode("utf-8"),
                            previous,
                            record_sha,
                        ),
                    )
                    appended.append(
                        JournalRecord(
                            sequence=sequence,
                            recorded_at_us=timestamp,
                            payload=payload,
                            previous_sha256=previous,
                            record_sha256=record_sha,
                        )
                    )
                    previous = record_sha
                    sequence += 1
                connection.execute(
                    "UPDATE journal_state SET next_sequence = ? WHERE singleton = 1",
                    (sequence,),
                )
                self._prune(
                    connection,
                    now_us=max(timestamp for _payload, timestamp in validated),
                    last_sequence=sequence - 1,
                )
                retained = connection.execute(
                    """
                    SELECT COUNT(*) AS retained_count,
                           MIN(sequence) AS first_sequence,
                           MAX(sequence) AS last_sequence
                    FROM records
                    WHERE sequence >= ? AND sequence < ?
                    """,
                    (append_start_sequence, sequence),
                ).fetchone()
                if (
                    retained is None
                    or int(retained["retained_count"]) != len(validated)
                    or int(retained["first_sequence"]) != append_start_sequence
                    or int(retained["last_sequence"]) != sequence - 1
                ):
                    raise ContractJournalError(
                        "contract journal retention would discard records from "
                        "the acknowledged append cohort"
                    )
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except BaseException as rollback_error:
                    failure = ContractJournalError(
                        "contract journal transaction rollback failed; "
                        "the journal is closed because commit state is uncertain"
                    )
                    try:
                        connection.close()
                    except BaseException:
                        pass
                    self._connection_handle = None
                    self._closed = True
                    self._close_failure = failure
                    raise failure from rollback_error
                raise
        return tuple(appended)

    def _prune(
        self,
        connection: sqlite3.Connection,
        *,
        now_us: int,
        last_sequence: int | None = None,
    ) -> None:
        """Prune only a contiguous prefix so the retained hash chain stays valid.

        Capture/publication timestamps should normally be monotonic, but replay,
        clock correction, or delayed producer evidence can legitimately arrive
        out of timestamp order.  Retention must therefore never delete an old
        record from the middle of the sequence chain.
        """
        cutoff = int(now_us) - self.max_age_us
        delete_through: int | None = None

        if last_sequence is None:
            last = connection.execute(
                "SELECT sequence FROM records ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            observed_last_sequence = (
                int(last["sequence"]) if last is not None else None
            )
        else:
            observed_last_sequence = int(last_sequence)
        if observed_last_sequence is not None:
            # Retained records are always one contiguous sequence range.  The
            # previous OFFSET query walked that whole range on every append;
            # derive the same boundary directly from its final sequence.
            delete_through = observed_last_sequence - self.max_records

        age_delete_through = self._age_delete_through(
            connection,
            cutoff=cutoff,
            observed_last_sequence=observed_last_sequence,
        )
        if age_delete_through is not None:
            if delete_through is None or age_delete_through > delete_through:
                delete_through = age_delete_through
        if delete_through is None:
            return

        last_deleted = connection.execute(
            "SELECT record_sha256 FROM records WHERE sequence = ?",
            (delete_through,),
        ).fetchone()
        if last_deleted is None:
            return
        connection.execute(
            "DELETE FROM records WHERE sequence <= ?",
            (delete_through,),
        )
        connection.execute(
            "UPDATE journal_state SET anchor_previous_sha256 = ? WHERE singleton = 1",
            (str(last_deleted["record_sha256"]),),
        )

    @staticmethod
    def _age_delete_through(
        connection: sqlite3.Connection,
        *,
        cutoff: int,
        observed_last_sequence: int | None,
    ) -> int | None:
        """Return the final sequence in the safely age-prunable prefix."""

        oldest = connection.execute(
            """
            SELECT sequence, recorded_at_us FROM records
            ORDER BY sequence ASC LIMIT 1
            """
        ).fetchone()
        if oldest is None or int(oldest["recorded_at_us"]) >= int(cutoff):
            return None

        age_delete_through: int | None = None
        if oldest is not None:
            # A short sequence-prefix probe is cheapest when only a few rows
            # have just crossed the age boundary.  After a long idle period,
            # avoid walking the entire old prefix synchronously: find the
            # earliest fresh sequence through the timestamp index instead.
            prefix = connection.execute(
                """
                SELECT sequence, recorded_at_us FROM records
                ORDER BY sequence ASC LIMIT 256
                """
            ).fetchall()
            first_fresh_sequence = next(
                (
                    int(row["sequence"])
                    for row in prefix
                    if int(row["recorded_at_us"]) >= cutoff
                ),
                None,
            )
            if first_fresh_sequence is None and len(prefix) == 256:
                indexed_first_fresh = connection.execute(
                    """
                    SELECT MIN(sequence) AS sequence
                    FROM records INDEXED BY records_recorded_at
                    WHERE recorded_at_us >= ?
                    """,
                    (cutoff,),
                ).fetchone()
                if (
                    indexed_first_fresh is not None
                    and indexed_first_fresh["sequence"] is not None
                ):
                    first_fresh_sequence = int(indexed_first_fresh["sequence"])
            if first_fresh_sequence is None:
                age_delete_through = observed_last_sequence
            else:
                previous = connection.execute(
                    """
                    SELECT sequence FROM records
                    WHERE sequence < ?
                    ORDER BY sequence DESC LIMIT 1
                    """,
                    (first_fresh_sequence,),
                ).fetchone()
                age_delete_through = (
                    int(previous["sequence"]) if previous is not None else None
                )
        return age_delete_through

    def records(self) -> tuple[JournalRecord, ...]:
        with self._lock, self._connection() as connection:
            return self._validated_records(connection)

    def close(self) -> None:
        """Checkpoint and close the durable WAL connection exactly once."""

        failure: BaseException | None = None
        normalized_failure: ContractJournalError | None = None
        with self._lock:
            if self._closed:
                if self._close_failure is not None:
                    raise self._close_failure
                return
            connection = self._connection_handle
            try:
                if connection is not None:
                    self._validate_storage_files()
                    self._validate_connection_configuration(connection)
                    checkpoint = connection.execute(
                        "PRAGMA wal_checkpoint(TRUNCATE)"
                    ).fetchone()
                    if checkpoint is None or len(checkpoint) != 3:
                        raise ContractJournalError(
                            "contract journal WAL checkpoint returned no exact receipt"
                        )
                    receipt = (
                        int(checkpoint[0]),
                        int(checkpoint[1]),
                        int(checkpoint[2]),
                    )
                    if receipt != (0, 0, 0):
                        raise ContractJournalError(
                            "contract journal WAL checkpoint did not complete: "
                            f"busy={receipt[0]} wal_frames={receipt[1]} "
                            f"checkpointed={receipt[2]}"
                        )
                    self._validate_connection_configuration(connection)
                    self._validate_storage_files()
            except BaseException as exc:
                failure = exc
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                self._connection_handle = None
                self._closed = True
            try:
                self._validate_main_binding()
                for sidecar in self._sidecar_paths():
                    if os.path.lexists(sidecar):
                        validate_private_file(
                            sidecar,
                            label=f"contract journal sidecar {sidecar.name}",
                        )
            except (ContractJournalError, PrivatePathError) as exc:
                if failure is None:
                    failure = (
                        exc
                        if isinstance(exc, ContractJournalError)
                        else ContractJournalError(str(exc))
                    )
            if failure is not None:
                normalized_failure = (
                    failure
                    if isinstance(failure, ContractJournalError)
                    else ContractJournalError(
                        "contract journal could not close its WAL connection: "
                        f"{failure}"
                    )
                )
                self._close_failure = normalized_failure
        if normalized_failure is not None:
            raise normalized_failure from (
                None if normalized_failure is failure else failure
            )


class AsyncContractJournal:
    """Non-blocking bounded batch writer that surfaces worker failure on submit."""

    def __init__(
        self,
        journal: ContractJournal,
        *,
        max_pending_batches: int = 1024,
        max_transaction_batches: int = 128,
    ) -> None:
        if max_pending_batches <= 0 or max_transaction_batches <= 0:
            raise ValueError("async journal queue and batch limits must be positive")
        self.journal = journal
        self.max_transaction_batches = int(max_transaction_batches)
        self._queue: Queue[tuple[tuple[dict[str, Any], ...], int] | None] = Queue(
            maxsize=int(max_pending_batches)
        )
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="Noesis-ContractJournal",
            daemon=True,
        )
        self._thread.start()

    @property
    def max_records(self) -> int | None:
        value = getattr(self.journal, "max_records", None)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def health_snapshot(self) -> dict[str, Any]:
        with self._failure_lock:
            failure = self._failure
        with self._state_lock:
            closed = bool(self._closed)
            pending = int(self._queue.qsize())
        return {
            "status": "failed" if failure is not None else ("closed" if closed else "healthy"),
            "pending_batches": pending,
            "max_pending_batches": int(self._queue.maxsize),
            "worker_alive": bool(self._thread.is_alive()),
            "last_error": (
                f"{type(failure).__name__}: {failure}"
                if failure is not None
                else None
            ),
        }

    def append_many(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        recorded_at_us: int,
    ) -> AsyncJournalAdmissionReceipt:
        self._raise_if_failed()
        timestamp = int(recorded_at_us)
        if timestamp <= 0:
            raise ContractJournalError("recorded_at_us must be positive")
        try:
            validated = tuple(validate_contract_payload(payload) for payload in payloads)
        except ReplayValidationError as exc:
            raise ContractJournalError(str(exc)) from exc
        with self._state_lock:
            self._raise_if_failed()
            if self._closed:
                raise ContractJournalError("async contract journal is closed")
            if not validated:
                return AsyncJournalAdmissionReceipt(
                    payload_count=0,
                    pending_batches=int(self._queue.qsize()),
                    max_pending_batches=int(self._queue.maxsize),
                )
            try:
                self._queue.put_nowait((validated, timestamp))
            except Full as exc:
                raise ContractJournalError(
                    "async contract journal queue is full; persistence is not keeping up"
                ) from exc
            return AsyncJournalAdmissionReceipt(
                payload_count=len(validated),
                pending_batches=int(self._queue.qsize()),
                max_pending_batches=int(self._queue.maxsize),
            )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            batch = [item]
            stop_after_batch = False
            while len(batch) < self.max_transaction_batches:
                try:
                    next_item = self._queue.get_nowait()
                except Empty:
                    break
                if next_item is None:
                    stop_after_batch = True
                    self._queue.task_done()
                    break
                batch.append(next_item)
            try:
                self.journal.append_entries(
                    (payload, timestamp)
                    for payloads, timestamp in batch
                    for payload in payloads
                )
            except BaseException as exc:
                with self._failure_lock:
                    self._failure = exc
            finally:
                for _item in batch:
                    self._queue.task_done()
            if stop_after_batch:
                return

    def _raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise ContractJournalError(
                f"async contract journal worker failed: {failure}"
            ) from failure

    def flush(self) -> None:
        with self._state_lock:
            self._queue.join()
            self._raise_if_failed()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                if self._thread.is_alive():
                    raise ContractJournalError("async contract journal did not stop")
                self._raise_if_failed()
                return
            # Close submission before draining.  Otherwise an append can pass
            # the open check, enqueue behind the stop sentinel, and report
            # success even though no worker remains to persist the record.
            self._closed = True
            # Drain and stop the worker even when a prior batch failed.  Calling
            # ``flush`` directly here would raise before the sentinel is delivered,
            # leaving the daemon thread alive during a best-effort runtime teardown.
            self._queue.join()
            self._queue.put(None)
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                raise ContractJournalError("async contract journal did not stop")
            self._raise_if_failed()


__all__ = [
    "AsyncJournalAdmissionReceipt",
    "AsyncContractJournal",
    "ContractJournal",
    "ContractJournalError",
    "JournalRecord",
]
