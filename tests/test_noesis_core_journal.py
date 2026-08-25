from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from pathlib import Path

import pytest

from noesis_core.contracts.health import CapabilityHealth, CapabilityState, CapabilityStatus
from noesis_core.journal import AsyncContractJournal, ContractJournal, ContractJournalError


def _payload(sequence: int) -> dict[str, object]:
    return CapabilityHealth(
        contract="noesis.capability.health",
        contract_version=1,
        instance_id="appliance",
        run_id="run-1",
        generated_at_us=1_000 + sequence,
        capabilities=(
            CapabilityState(
                capability="world",
                status=CapabilityStatus.HEALTHY,
                checked_at_us=1_000 + sequence,
                last_success_at_us=1_000 + sequence,
                evidence={"sequence": sequence},
            ),
        ),
    ).model_dump(mode="json")


def _sqlite_sidecars(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    )


def test_journal_uses_exact_durable_wal_profile_and_clean_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)

    connection = journal._connection_handle
    assert connection is not None
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1_000
    for sidecar in _sqlite_sidecars(path):
        info = sidecar.lstat()
        assert info.st_mode & 0o777 == 0o600
        assert info.st_uid == os.geteuid()
        assert info.st_nlink == 1

    journal.close()
    journal.close()

    assert journal.closed
    assert all(not sidecar.exists() for sidecar in _sqlite_sidecars(path))
    with pytest.raises(ContractJournalError, match="journal is closed"):
        journal.records()
    with pytest.raises(ContractJournalError, match="journal is closed"):
        journal.append_many([_payload(2)], recorded_at_us=2_001)
    with pytest.raises(ContractJournalError, match="journal is closed"):
        journal.append_entries(())

    reopened = ContractJournal(path)
    try:
        assert [record.payload for record in reopened.records()] == [_payload(1)]
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("pragma", "expected_error"),
    (
        ("PRAGMA synchronous = NORMAL", "synchronous policy changed"),
        ("PRAGMA wal_autocheckpoint = 7", "checkpoint interval changed"),
    ),
)
def test_journal_rechecks_connection_durability_before_every_operation(
    tmp_path: Path,
    pragma: str,
    expected_error: str,
) -> None:
    journal = ContractJournal(tmp_path / "world.sqlite3")
    connection = journal._connection_handle
    assert connection is not None
    connection.execute(pragma)

    with pytest.raises(ContractJournalError, match=expected_error):
        journal.records()

    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA wal_autocheckpoint = 1000")
    assert journal.records() == ()
    journal.close()


def test_journal_rejects_insecure_existing_wal_sidecar_before_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    path.touch(mode=0o600)
    wal, _shm = _sqlite_sidecars(path)
    wal.touch(mode=0o644)

    with pytest.raises(ContractJournalError, match="mode must be 0600"):
        ContractJournal(path)

    assert wal.stat().st_mode & 0o777 == 0o644


def test_journal_rejects_wal_permission_link_and_inode_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)
    wal, _shm = _sqlite_sidecars(path)

    wal.chmod(0o644)
    with pytest.raises(ContractJournalError, match="mode must be 0600"):
        journal.records()
    wal.chmod(0o600)

    alias = tmp_path / "wal-alias"
    os.link(wal, alias)
    with pytest.raises(ContractJournalError, match="exactly one hard link"):
        journal.records()
    alias.unlink()

    original = tmp_path / "original-wal"
    wal.rename(original)
    wal.touch(mode=0o600)
    with pytest.raises(ContractJournalError, match="sidecar inode changed"):
        journal.records()
    wal.unlink()
    original.rename(wal)

    assert len(journal.records()) == 1
    journal.close()


def test_journal_rejects_main_database_inode_swap_while_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)
    original = tmp_path / "original.sqlite3"

    path.rename(original)
    path.touch(mode=0o600)
    with pytest.raises(ContractJournalError, match="database inode changed"):
        journal.records()
    path.unlink()
    original.rename(path)

    assert len(journal.records()) == 1
    journal.close()


def test_journal_rejects_main_database_swap_after_path_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import noesis_core.journal as journal_module

    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)
    original = tmp_path / "original.sqlite3"
    real_validate = journal_module.validate_private_file
    swapped = False

    def validate_then_swap(*args, **kwargs):
        nonlocal swapped
        validated = real_validate(*args, **kwargs)
        if not swapped and kwargs.get("label") == "contract journal":
            swapped = True
            path.rename(original)
            path.touch(mode=0o600)
        return validated

    monkeypatch.setattr(journal_module, "validate_private_file", validate_then_swap)
    with pytest.raises(ContractJournalError, match="database inode changed"):
        journal.records()

    path.unlink()
    original.rename(path)
    monkeypatch.setattr(journal_module, "validate_private_file", real_validate)
    assert len(journal.records()) == 1
    journal.close()


def test_journal_revalidates_main_inode_after_connection_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)
    real_connection = journal._connection_handle
    assert real_connection is not None
    original = tmp_path / "original.sqlite3"

    class SwapOnClose:
        def __getattr__(self, name):
            return getattr(real_connection, name)

        def close(self) -> None:
            real_connection.close()
            path.rename(original)
            path.touch(mode=0o600)

    journal._connection_handle = SwapOnClose()  # type: ignore[assignment]
    with pytest.raises(ContractJournalError, match="database inode changed"):
        journal.close()

    assert journal.closed
    path.unlink()
    original.rename(path)
    assert all(not sidecar.exists() for sidecar in _sqlite_sidecars(path))


def test_journal_serializes_one_connection_across_threads(tmp_path: Path) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path, max_records=1_000)
    errors: list[BaseException] = []
    start = threading.Barrier(5)

    def append_range(offset: int) -> None:
        try:
            start.wait(timeout=5.0)
            for sequence in range(offset, offset + 10):
                journal.append_many(
                    [_payload(sequence)],
                    recorded_at_us=2_000 + sequence,
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=append_range, args=(offset,), daemon=True)
        for offset in (0, 10, 20, 30)
    ]
    for thread in threads:
        thread.start()
    start.wait(timeout=5.0)
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    records = journal.records()
    assert len(records) == 40
    assert [record.sequence for record in records] == list(range(40))
    journal.close()


def test_journal_recovers_committed_wal_after_unclean_process_exit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    child = """
import json
import os
import sys
from noesis_core.journal import ContractJournal

journal = ContractJournal(sys.argv[1])
journal.append_many([json.loads(sys.argv[2])], recorded_at_us=2000)
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", child, str(path), json.dumps(_payload(7))],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    assert any(sidecar.exists() for sidecar in _sqlite_sidecars(path))
    recovered = ContractJournal(path)
    try:
        records = recovered.records()
        assert len(records) == 1
        assert records[0].payload == _payload(7)
    finally:
        recovered.close()
    assert all(not sidecar.exists() for sidecar in _sqlite_sidecars(path))


def test_journal_is_bounded_restartable_and_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "private" / "world.sqlite3"
    journal = ContractJournal(path, max_records=3, max_age_us=1_000_000)
    for sequence in range(5):
        journal.append_many([_payload(sequence)], recorded_at_us=2_000 + sequence)

    records = journal.records()
    assert [record.payload["capabilities"][0]["evidence"]["sequence"] for record in records] == [2, 3, 4]
    assert records[0].sequence == 2
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert ContractJournal(path, max_records=3, max_age_us=1_000_000).records() == records


def test_journal_retention_append_work_does_not_scale_with_capacity(
    tmp_path: Path,
) -> None:
    def append_vm_steps(capacity: int) -> int:
        journal = ContractJournal(
            tmp_path / f"world-{capacity}.sqlite3",
            max_records=capacity,
            max_age_us=10_000_000,
        )
        journal.append_many(
            [_payload(sequence) for sequence in range(capacity)],
            recorded_at_us=2_000,
        )
        connection = journal._connection_handle
        assert connection is not None
        steps = 0

        def count_step() -> None:
            nonlocal steps
            steps += 1

        connection.set_progress_handler(count_step, 1)
        try:
            journal.append_many([_payload(capacity)], recorded_at_us=2_001)
        finally:
            connection.set_progress_handler(None, 0)
        assert [record.sequence for record in journal.records()] == list(
            range(1, capacity + 1)
        )
        journal.close()
        return steps

    small_steps = append_vm_steps(64)
    large_steps = append_vm_steps(2_048)

    assert large_steps <= small_steps + 100


def test_journal_long_aged_prefix_uses_timestamp_index(
    tmp_path: Path,
) -> None:
    def prune_vm_steps(capacity: int) -> int:
        journal = ContractJournal(
            tmp_path / f"aged-{capacity}.sqlite3",
            max_records=capacity + 1,
            max_age_us=50,
        )
        journal.append_many(
            [_payload(sequence) for sequence in range(capacity)],
            recorded_at_us=100,
        )
        connection = journal._connection_handle
        assert connection is not None
        steps = 0

        def count_step() -> None:
            nonlocal steps
            steps += 1

        connection.execute(
            "INSERT INTO records("
            "sequence, recorded_at_us, contract, payload_json, "
            "previous_sha256, record_sha256"
            ") SELECT ?, ?, contract, payload_json, previous_sha256, "
            "printf('%064x', ?) "
            "FROM records WHERE sequence = 0",
            (capacity, 1_000, capacity + 1),
        )
        connection.set_progress_handler(count_step, 1)
        try:
            boundary = journal._age_delete_through(
                connection,
                cutoff=950,
                observed_last_sequence=capacity,
            )
        finally:
            connection.set_progress_handler(None, 0)
        assert boundary == capacity - 1
        journal.close()
        return steps

    small_steps = prune_vm_steps(512)
    large_steps = prune_vm_steps(8_192)

    assert large_steps <= small_steps + 200


def test_journal_detects_database_tampering(tmp_path: Path) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE records SET payload_json = '{}' WHERE sequence = 0")
    connection.commit()
    connection.close()
    with pytest.raises(ContractJournalError, match="checksum mismatch"):
        journal.records()


@pytest.mark.parametrize(
    "tamper_payload",
    (
        lambda original: '{"contract":"tamper-secret",' + original[1:],
        lambda original: original.replace(
            '"evidence":{"sequence":1}',
            '"evidence":{"sequence":"tamper-secret","sequence":1}',
        ),
    ),
    ids=("duplicate-root-key", "duplicate-nested-key"),
)
def test_journal_rejects_duplicate_keys_before_rehash(
    tmp_path: Path,
    tamper_payload,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)

    with sqlite3.connect(path) as connection:
        original = str(
            connection.execute(
                "SELECT payload_json FROM records WHERE sequence = 0"
            ).fetchone()[0]
        )
        tampered = tamper_payload(original)
        assert tampered != original
        connection.execute(
            "UPDATE records SET payload_json = ? WHERE sequence = 0",
            (tampered,),
        )

    with pytest.raises(
        ContractJournalError,
        match="journal payload JSON is ambiguous or invalid at sequence 0",
    ) as caught:
        journal.records()
    assert "tamper-secret" not in str(caught.value)


@pytest.mark.parametrize("variant", ("whitespace", "key_order"))
def test_journal_rejects_noncanonical_payload_bytes_before_rehash(
    tmp_path: Path,
    variant: str,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)

    with sqlite3.connect(path) as connection:
        original = str(
            connection.execute(
                "SELECT payload_json FROM records WHERE sequence = 0"
            ).fetchone()[0]
        )
        if variant == "whitespace":
            tampered = "  " + original
        else:
            payload = json.loads(original)
            reordered = dict(reversed(tuple(payload.items())))
            tampered = json.dumps(
                reordered,
                sort_keys=False,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        assert tampered != original
        connection.execute(
            "UPDATE records SET payload_json = ? WHERE sequence = 0",
            (tampered,),
        )

    with pytest.raises(ContractJournalError, match="not canonical JSON"):
        journal.records()


@pytest.mark.parametrize("numeric_token", ("NaN", "Infinity", "1e309"))
def test_journal_rejects_nonfinite_or_overflow_payload_before_rehash(
    tmp_path: Path,
    numeric_token: str,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    journal.append_many([_payload(1)], recorded_at_us=2_000)

    with sqlite3.connect(path) as connection:
        original = str(
            connection.execute(
                "SELECT payload_json FROM records WHERE sequence = 0"
            ).fetchone()[0]
        )
        tampered = original.replace(
            '"evidence":{"sequence":1}',
            f'"evidence":{{"sequence":{numeric_token}}}',
        )
        assert tampered != original
        connection.execute(
            "UPDATE records SET payload_json = ? WHERE sequence = 0",
            (tampered,),
        )

    with pytest.raises(
        ContractJournalError,
        match="journal payload JSON is ambiguous or invalid at sequence 0",
    ):
        journal.records()


def test_journal_rejects_unsupported_payload_without_partial_write(tmp_path: Path) -> None:
    journal = ContractJournal(tmp_path / "world.sqlite3")
    with pytest.raises(ContractJournalError, match="unsupported replay contract"):
        journal.append_many(
            [_payload(1), {"contract": "noesis.future", "contract_version": 1}],
            recorded_at_us=2_000,
        )
    assert journal.records() == ()


def test_async_journal_batches_and_flushes_without_blocking_caller(tmp_path: Path) -> None:
    journal = ContractJournal(tmp_path / "world.sqlite3", max_records=500)
    asynchronous = AsyncContractJournal(journal, max_pending_batches=200)
    for sequence in range(100):
        receipt = asynchronous.append_many(
            [_payload(sequence)],
            recorded_at_us=2_000 + sequence,
        )
        assert receipt.payload_count == 1
        assert receipt.max_pending_batches == 200
    asynchronous.close()
    assert len(journal.records()) == 100


def test_async_journal_surfaces_worker_failure() -> None:
    class BrokenJournal:
        def append_entries(self, _entries) -> None:
            raise OSError("disk failed")

    asynchronous = AsyncContractJournal(BrokenJournal())  # type: ignore[arg-type]
    asynchronous.append_many([_payload(1)], recorded_at_us=2_000)
    with pytest.raises(ContractJournalError, match="disk failed"):
        asynchronous.flush()


def test_age_pruning_keeps_a_contiguous_chain_with_out_of_order_times(
    tmp_path: Path,
) -> None:
    journal = ContractJournal(
        tmp_path / "world.sqlite3",
        max_records=10,
        max_age_us=50,
    )
    journal.append_entries(((_payload(0), 100),))
    journal.append_entries(
        (
            (_payload(1), 200),
            (_payload(2), 120),
        )
    )

    records = journal.records()
    assert [record.sequence for record in records] == [1, 2]
    assert [
        record.payload["capabilities"][0]["evidence"]["sequence"]
        for record in records
    ] == [1, 2]


def test_async_journal_close_stops_worker_after_failure() -> None:
    class BrokenJournal:
        def append_entries(self, _entries) -> None:
            raise OSError("disk failed")

    asynchronous = AsyncContractJournal(BrokenJournal())  # type: ignore[arg-type]
    asynchronous.append_many([_payload(1)], recorded_at_us=2_000)

    with pytest.raises(ContractJournalError, match="disk failed"):
        asynchronous.close()

    assert not asynchronous._thread.is_alive()


def test_async_journal_close_rejects_inflight_uncommitted_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import noesis_core.journal as journal_module

    validation_started = threading.Event()
    release_validation = threading.Event()
    append_errors: list[BaseException] = []
    real_validate = journal_module.validate_contract_payload

    def _blocked_validate(payload):
        validation_started.set()
        assert release_validation.wait(timeout=5.0)
        return real_validate(payload)

    monkeypatch.setattr(journal_module, "validate_contract_payload", _blocked_validate)
    journal = ContractJournal(tmp_path / "world.sqlite3")
    asynchronous = AsyncContractJournal(journal)

    def _append() -> None:
        try:
            asynchronous.append_many([_payload(1)], recorded_at_us=2_000)
        except BaseException as exc:
            append_errors.append(exc)

    append_thread = threading.Thread(target=_append, daemon=True)
    append_thread.start()
    assert validation_started.wait(timeout=2.0)

    asynchronous.close()
    release_validation.set()
    append_thread.join(timeout=5.0)

    assert not append_thread.is_alive()
    assert len(append_errors) == 1
    assert isinstance(append_errors[0], ContractJournalError)
    assert "async contract journal is closed" in str(append_errors[0])
    assert not asynchronous._thread.is_alive()
    assert asynchronous._queue.unfinished_tasks == 0
    assert journal.records() == ()


def _seed_closed_journal(path: Path, *, count: int = 1) -> None:
    journal = ContractJournal(path, max_records=max(10, count))
    try:
        for sequence in range(count):
            journal.append_many(
                [_payload(sequence)],
                recorded_at_us=2_000 + sequence,
            )
    finally:
        journal.close()


def _stored_record_sha256(
    *,
    sequence: int,
    recorded_at_us: int,
    payload: object,
    previous_sha256: str,
) -> str:
    encoded = json.dumps(
        {
            "sequence": sequence,
            "recorded_at_us": recorded_at_us,
            "payload": payload,
            "previous_sha256": previous_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_journal_append_hashes_match_canonical_core(tmp_path: Path) -> None:
    journal = ContractJournal(tmp_path / "world.sqlite3")
    appended = journal.append_entries(
        (
            (_payload(1), 2_000),
            (_payload(2), 2_001),
        )
    )

    assert [record.record_sha256 for record in appended] == [
        _stored_record_sha256(
            sequence=record.sequence,
            recorded_at_us=record.recorded_at_us,
            payload=record.payload,
            previous_sha256=record.previous_sha256,
        )
        for record in appended
    ]
    assert journal.records() == appended
    journal.close()


def test_journal_reopens_additive_legacy_shape_without_mutating_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        row = connection.execute(
            "SELECT payload_json, previous_sha256 FROM records WHERE sequence = 0"
        ).fetchone()
        payload = json.loads(str(row[0]))
        del payload["capabilities"][0]["blockers"]
        payload_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        checksum = _stored_record_sha256(
            sequence=0,
            recorded_at_us=2_000,
            payload=payload,
            previous_sha256=str(row[1]),
        )
        connection.execute(
            "UPDATE records SET payload_json = ?, record_sha256 = ? "
            "WHERE sequence = 0",
            (payload_json, checksum),
        )

    before = path.read_bytes()
    journal = ContractJournal(path)
    try:
        records = journal.records()
        assert "blockers" not in records[0].payload["capabilities"][0]
        assert records[0].record_sha256 == checksum
    finally:
        journal.close()
    after = path.read_bytes()

    assert before == after


def _assert_reopen_rejects_and_cleans_up(path: Path) -> None:
    with pytest.raises(ContractJournalError):
        ContractJournal(path)
    assert all(not sidecar.exists() for sidecar in _sqlite_sidecars(path))


def test_journal_rejects_corrupt_history_during_restart_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE records SET payload_json = '{}' WHERE sequence = 0"
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_rejects_corrupt_v1_schema_during_restart_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP INDEX records_recorded_at")

    _assert_reopen_rejects_and_cleans_up(path)


@pytest.mark.parametrize(
    "statement",
    (
        "DELETE FROM journal_state",
        "UPDATE journal_state SET next_sequence = 99 WHERE singleton = 1",
        "UPDATE journal_state SET anchor_previous_sha256 = 'f' || "
        "substr(anchor_previous_sha256, 2) WHERE singleton = 1",
    ),
    ids=("missing-state", "next-sequence", "anchor"),
)
def test_journal_rejects_corrupt_state_during_restart_admission(
    tmp_path: Path,
    statement: str,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(statement)

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_rejects_contract_column_payload_disagreement_on_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE records SET contract = 'noesis.world.snapshot' "
            "WHERE sequence = 0"
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_rejects_nonpositive_stored_timestamp_even_with_valid_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        payload_json, previous_sha256 = connection.execute(
            "SELECT payload_json, previous_sha256 FROM records WHERE sequence = 0"
        ).fetchone()
        payload = json.loads(str(payload_json))
        record_sha256 = _stored_record_sha256(
            sequence=0,
            recorded_at_us=0,
            payload=payload,
            previous_sha256=str(previous_sha256),
        )
        connection.execute(
            "UPDATE records SET recorded_at_us = 0, record_sha256 = ? "
            "WHERE sequence = 0",
            (record_sha256,),
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_rejects_payload_that_model_validation_would_normalize(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        payload_json, previous_sha256 = connection.execute(
            "SELECT payload_json, previous_sha256 FROM records WHERE sequence = 0"
        ).fetchone()
        payload = json.loads(str(payload_json))
        payload["instance_id"] = " appliance "
        canonical_payload = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        record_sha256 = _stored_record_sha256(
            sequence=0,
            recorded_at_us=2_000,
            payload=payload,
            previous_sha256=str(previous_sha256),
        )
        connection.execute(
            "UPDATE records SET payload_json = ?, record_sha256 = ? "
            "WHERE sequence = 0",
            (canonical_payload, record_sha256),
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_rejects_noncontiguous_sequence_even_with_valid_hash_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path, count=2)
    with closing(sqlite3.connect(path)) as connection, connection:
        payload_json, previous_sha256 = connection.execute(
            "SELECT payload_json, previous_sha256 FROM records WHERE sequence = 1"
        ).fetchone()
        payload = json.loads(str(payload_json))
        record_sha256 = _stored_record_sha256(
            sequence=2,
            recorded_at_us=2_001,
            payload=payload,
            previous_sha256=str(previous_sha256),
        )
        connection.execute(
            "UPDATE records SET sequence = 2, record_sha256 = ? "
            "WHERE sequence = 1",
            (record_sha256,),
        )
        connection.execute(
            "UPDATE journal_state SET next_sequence = 3 WHERE singleton = 1"
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_never_adopts_or_mutates_an_unrelated_v0_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE unrelated(value TEXT NOT NULL)")
        connection.execute("INSERT INTO unrelated(value) VALUES ('preserve-me')")
    path.chmod(0o600)

    with pytest.raises(ContractJournalError):
        ContractJournal(path)

    assert all(not sidecar.exists() for sidecar in _sqlite_sidecars(path))
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("SELECT value FROM unrelated").fetchone()[0] == (
            "preserve-me"
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
    assert tables == {"unrelated"}


def test_journal_rolls_back_transaction_after_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeliberateAbort(BaseException):
        pass

    journal = ContractJournal(tmp_path / "world.sqlite3")
    real_prune = journal._prune

    def abort_prune(
        _connection,
        *,
        now_us: int,
        last_sequence: int,
    ) -> None:
        assert now_us == 2_000
        assert last_sequence == 0
        raise DeliberateAbort

    monkeypatch.setattr(journal, "_prune", abort_prune)
    with pytest.raises(DeliberateAbort):
        journal.append_many([_payload(0)], recorded_at_us=2_000)
    monkeypatch.setattr(journal, "_prune", real_prune)

    assert journal.records() == ()
    appended = journal.append_many([_payload(1)], recorded_at_us=2_001)
    assert [record.sequence for record in appended] == [0]
    journal.close()


def test_journal_requires_exact_checkpoint_receipt_and_repeats_close_failure(
    tmp_path: Path,
) -> None:
    journal = ContractJournal(tmp_path / "world.sqlite3")
    journal.append_many([_payload(0)], recorded_at_us=2_000)
    real_connection = journal._connection_handle
    assert real_connection is not None

    class InexactCheckpointReceipt:
        @staticmethod
        def fetchone() -> tuple[int, int, int]:
            return (0, 1, 1)

    class InexactCheckpointConnection:
        def execute(self, statement: str, *args, **kwargs):
            if statement.strip().lower() == "pragma wal_checkpoint(truncate)":
                return InexactCheckpointReceipt()
            return real_connection.execute(statement, *args, **kwargs)

        def close(self) -> None:
            real_connection.close()

    journal._connection_handle = InexactCheckpointConnection()  # type: ignore[assignment]
    for _attempt in range(2):
        with pytest.raises(ContractJournalError, match="checkpoint"):
            journal.close()
    assert journal.closed


def test_journal_rejects_invalid_empty_state_anchor_on_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path, count=0)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE journal_state SET anchor_previous_sha256 = 'not-a-sha256' "
            "WHERE singleton = 1"
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_never_acknowledges_a_batch_larger_than_retention(
    tmp_path: Path,
) -> None:
    journal = ContractJournal(tmp_path / "world.sqlite3", max_records=2)
    with pytest.raises(ContractJournalError, match="retention|max_records"):
        journal.append_many(
            [_payload(0), _payload(1), _payload(2)],
            recorded_at_us=2_000,
        )
    assert journal.records() == ()
    journal.close()


def test_empty_initial_journal_requires_genesis_anchor_on_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path, count=0)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE journal_state SET anchor_previous_sha256 = ? "
            "WHERE singleton = 1",
            ("f" * 64,),
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_first_sequence_zero_requires_genesis_even_with_recomputed_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    forged_anchor = "f" * 64
    with closing(sqlite3.connect(path)) as connection, connection:
        payload_json = connection.execute(
            "SELECT payload_json FROM records WHERE sequence = 0"
        ).fetchone()[0]
        payload = json.loads(str(payload_json))
        forged_sha = _stored_record_sha256(
            sequence=0,
            recorded_at_us=2_000,
            payload=payload,
            previous_sha256=forged_anchor,
        )
        connection.execute(
            "UPDATE journal_state SET anchor_previous_sha256 = ? "
            "WHERE singleton = 1",
            (forged_anchor,),
        )
        connection.execute(
            "UPDATE records SET previous_sha256 = ?, record_sha256 = ? "
            "WHERE sequence = 0",
            (forged_anchor, forged_sha),
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_pruned_first_sequence_rejects_genesis_even_with_recomputed_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path, max_records=1)
    journal.append_many([_payload(0)], recorded_at_us=2_000)
    journal.append_many([_payload(1)], recorded_at_us=2_001)
    journal.close()

    with closing(sqlite3.connect(path)) as connection, connection:
        payload_json = connection.execute(
            "SELECT payload_json FROM records WHERE sequence = 1"
        ).fetchone()[0]
        payload = json.loads(str(payload_json))
        forged_sha = _stored_record_sha256(
            sequence=1,
            recorded_at_us=2_001,
            payload=payload,
            previous_sha256=ContractJournal.GENESIS_SHA256,
        )
        connection.execute(
            "UPDATE journal_state SET anchor_previous_sha256 = ? "
            "WHERE singleton = 1",
            (ContractJournal.GENESIS_SHA256,),
        )
        connection.execute(
            "UPDATE records SET previous_sha256 = ?, record_sha256 = ? "
            "WHERE sequence = 1",
            (ContractJournal.GENESIS_SHA256, forged_sha),
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_empty_noninitial_journal_rejects_genesis_anchor_on_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM records")
        connection.execute(
            "UPDATE journal_state SET anchor_previous_sha256 = ? "
            "WHERE singleton = 1",
            (ContractJournal.GENESIS_SHA256,),
        )

    _assert_reopen_rejects_and_cleans_up(path)


def test_journal_rejects_age_self_pruning_batch_without_state_advance(
    tmp_path: Path,
) -> None:
    journal = ContractJournal(
        tmp_path / "world.sqlite3",
        max_records=10,
        max_age_us=50,
    )
    with pytest.raises(ContractJournalError, match="retention|cohort|prun"):
        journal.append_entries(
            (
                (_payload(0), 100),
                (_payload(1), 200),
            )
        )

    assert journal.records() == ()
    appended = journal.append_many([_payload(2)], recorded_at_us=300)
    assert [record.sequence for record in appended] == [0]
    journal.close()


def test_journal_rejects_persistent_wal_mode_drift_without_repair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        selected_mode = connection.execute(
            "PRAGMA journal_mode = DELETE"
        ).fetchone()[0]
    assert selected_mode == "delete"

    _assert_reopen_rejects_and_cleans_up(path)

    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_journal_normalizes_physical_database_corruption_and_cleans_up(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    corrupt_bytes = b"not-a-sqlite-database\x00private-journal-corruption"
    path.write_bytes(corrupt_bytes)
    path.chmod(0o600)

    with pytest.raises(ContractJournalError, match="database|SQLite|corrupt"):
        ContractJournal(path)

    assert path.read_bytes() == corrupt_bytes
    assert all(not sidecar.exists() for sidecar in _sqlite_sidecars(path))


def test_journal_migrates_validated_legacy_v1_to_exact_v2_wal_profile(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute(
            "PRAGMA journal_mode = DELETE"
        ).fetchone()[0] == "delete"
        connection.execute("PRAGMA user_version = 1")

    migrated = ContractJournal(path)
    try:
        assert [record.payload for record in migrated.records()] == [_payload(0)]
        connection = migrated._connection_handle
        assert connection is not None
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1_000
    finally:
        migrated.close()

    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    restarted = ContractJournal(path)
    try:
        assert [record.payload for record in restarted.records()] == [_payload(0)]
        connection = restarted._connection_handle
        assert connection is not None
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        restarted.close()


def test_journal_rejects_corrupt_legacy_v1_before_any_migration_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world.sqlite3"
    _seed_closed_journal(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute(
            "PRAGMA journal_mode = DELETE"
        ).fetchone()[0] == "delete"
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            "UPDATE records SET payload_json = '{}' WHERE sequence = 0"
        )

    _assert_reopen_rejects_and_cleans_up(path)

    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute(
            "SELECT payload_json FROM records WHERE sequence = 0"
        ).fetchone()[0] == "{}"


def test_journal_rollback_failure_poison_is_sticky_and_restart_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeliberateAbort(BaseException):
        pass

    path = tmp_path / "world.sqlite3"
    journal = ContractJournal(path)
    real_connection = journal._connection_handle
    assert real_connection is not None

    class RollbackFailureConnection:
        def execute(self, statement: str, *args, **kwargs):
            if statement.strip().upper() == "ROLLBACK":
                raise sqlite3.OperationalError("injected rollback failure")
            return real_connection.execute(statement, *args, **kwargs)

        def close(self) -> None:
            real_connection.close()

    def abort_prune(
        _connection,
        *,
        now_us: int,
        last_sequence: int,
    ) -> None:
        assert now_us == 2_000
        assert last_sequence == 0
        raise DeliberateAbort

    journal._connection_handle = RollbackFailureConnection()  # type: ignore[assignment]
    monkeypatch.setattr(journal, "_prune", abort_prune)

    for operation in (
        lambda: journal.append_many([_payload(0)], recorded_at_us=2_000),
        journal.records,
        journal.close,
    ):
        with pytest.raises(
            ContractJournalError,
            match="rollback failed|commit state is uncertain",
        ):
            operation()

    assert journal.closed
    assert journal._connection_handle is None
    assert all(not sidecar.exists() for sidecar in _sqlite_sidecars(path))

    recovered = ContractJournal(path)
    try:
        assert recovered.records() == ()
        appended = recovered.append_many([_payload(1)], recorded_at_us=2_001)
        assert [record.sequence for record in appended] == [0]
    finally:
        recovered.close()
