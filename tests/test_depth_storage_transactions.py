from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zarr

from geometry.depth_source import (
    CommitReceipt,
    DepthStorageClosedError,
    DepthStorageError,
    DepthStorageManager,
    DepthStoragePoisonedError,
    DepthStorageQueueFullError,
    DuplicateSnapshotError,
    SnapshotDescriptor,
    StorageLifecycle,
    resolve_depth_store_commit_timeout_s,
)


def _arrays(value: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((4, 6), value, dtype=np.float32)
    confidence = np.ones_like(depth, dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)
    return depth, confidence, mask


def _manager(tmp_path: Path, **kwargs: object) -> DepthStorageManager:
    options = {
        "base_path": tmp_path,
        "max_snapshots_per_camera": 0,
        "retention_minutes": 0.0,
        "enable_async": True,
        "enforce_async": False,
        "max_queue_size": 4,
        "worker_count": 1,
        "max_worker_count": 1,
        "zarr_clevel": 0,
        "zarr_chunk_px": 0,
    }
    options.update(kwargs)
    return DepthStorageManager(**options)  # type: ignore[arg-type]


def _close(manager: DepthStorageManager) -> None:
    receipt = manager.shutdown(timeout=5.0)
    assert receipt.completed, receipt
    assert receipt.state is StorageLifecycle.CLOSED


def test_public_depth_commit_timeout_is_finite_and_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S", raising=False)
    assert resolve_depth_store_commit_timeout_s() == 30.0
    assert resolve_depth_store_commit_timeout_s("0") == 0.1
    assert resolve_depth_store_commit_timeout_s("999") == 60.0
    for invalid in ("not-a-number", "nan", "inf", "-inf"):
        with pytest.raises(ValueError, match="finite number"):
            resolve_depth_store_commit_timeout_s(invalid)


def test_blocked_writer_returns_deadline_receipts_and_retains_live_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = manager._write_snapshot  # type: ignore[attr-defined]

    def blocked(job):
        entered.set()
        assert release.wait(5.0)
        return original(job)

    monkeypatch.setattr(manager, "_write_snapshot", blocked)
    depth, confidence, mask = _arrays()
    handle = manager.store("camera", 1_700_000_000_000_001, depth, confidence, mask)
    assert entered.wait(2.0)

    flush = manager.flush(timeout=0.01)
    assert not flush.completed
    assert flush.timed_out
    assert flush.pending_sequences == (handle.sequence,)

    first_shutdown = manager.shutdown(timeout=0.01)
    assert not first_shutdown.completed
    assert first_shutdown.state is StorageLifecycle.CLOSING
    assert first_shutdown.alive_writer_names
    assert manager._queue is not None  # type: ignore[attr-defined]
    assert manager._writer_threads  # type: ignore[attr-defined]
    with pytest.raises(DepthStorageClosedError):
        manager.store("camera", 1_700_000_000_000_002, depth, confidence, mask)

    release.set()
    assert isinstance(handle.wait(timeout=5.0), CommitReceipt)
    _close(manager)


def test_queue_saturation_rejects_without_synchronous_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path, max_queue_size=1, queue_put_timeout_s=0.01)
    entered = threading.Event()
    release = threading.Event()
    original = manager._write_snapshot  # type: ignore[attr-defined]

    def blocked(job):
        entered.set()
        assert release.wait(5.0)
        return original(job)

    monkeypatch.setattr(manager, "_write_snapshot", blocked)
    depth, confidence, mask = _arrays()
    first = manager.store("camera", 1_700_000_000_000_011, depth, confidence, mask)
    assert entered.wait(2.0)
    second = manager.store("camera", 1_700_000_000_000_012, depth, confidence, mask)
    with pytest.raises(DepthStorageQueueFullError):
        manager.store("camera", 1_700_000_000_000_013, depth, confidence, mask)
    assert not manager._snapshot_destination("camera", 1_700_000_000_000_013).exists()  # type: ignore[attr-defined]

    release.set()
    first.wait(timeout=5.0)
    second.wait(timeout=5.0)
    _close(manager)


def test_first_disk_failure_poison_is_immutable_and_callback_fires_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures = []
    manager = _manager(tmp_path, max_queue_size=2, on_failure=failures.append)
    entered = threading.Event()
    release = threading.Event()

    def fail_first(job):
        entered.set()
        assert release.wait(5.0)
        raise OSError("injected disk failure")

    monkeypatch.setattr(manager, "_write_snapshot", fail_first)
    depth, confidence, mask = _arrays()
    first = manager.store("camera", 1_700_000_000_000_021, depth, confidence, mask)
    assert entered.wait(2.0)
    second = manager.store("camera", 1_700_000_000_000_022, depth, confidence, mask)
    release.set()

    with pytest.raises(DepthStoragePoisonedError):
        first.wait(timeout=5.0)
    with pytest.raises(DepthStoragePoisonedError):
        second.wait(timeout=5.0)
    poison = manager.poison
    assert poison is not None
    assert poison.write_id == first.write_id
    assert poison.message == "injected disk failure"
    assert failures == [poison]
    with pytest.raises(DepthStoragePoisonedError, match=first.write_id):
        manager.store("camera", 1_700_000_000_000_023, depth, confidence, mask)
    assert manager.poison == poison
    _close(manager)


def test_two_worker_reordered_completion_keeps_timestamp_sorted_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path, worker_count=2, max_worker_count=2)
    older_started = threading.Event()
    newer_committed = threading.Event()
    release_older = threading.Event()
    original = manager._write_snapshot  # type: ignore[attr-defined]
    older_ts = 1_700_000_000_000_031
    newer_ts = older_ts + 1

    def reordered(job):
        if job.ts_us == older_ts:
            older_started.set()
            assert release_older.wait(5.0)
            return original(job)
        receipt = original(job)
        newer_committed.set()
        release_older.set()
        return receipt

    monkeypatch.setattr(manager, "_write_snapshot", reordered)
    depth, confidence, mask = _arrays()
    older = manager.store("camera", older_ts, depth, confidence, mask)
    assert older_started.wait(2.0)
    newer = manager.store("camera", newer_ts, depth, confidence, mask)
    assert newer_committed.wait(5.0)
    newer.wait(timeout=5.0)
    older.wait(timeout=5.0)

    assert [ts for ts, _path in manager.list_snapshot_entries("camera")] == [
        older_ts,
        newer_ts,
    ]
    assert manager.latest_entry("camera", None) == newer.path
    _close(manager)


def test_duplicate_camera_timestamp_is_reserved_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = manager._write_snapshot  # type: ignore[attr-defined]

    def blocked(job):
        entered.set()
        assert release.wait(5.0)
        return original(job)

    monkeypatch.setattr(manager, "_write_snapshot", blocked)
    depth, confidence, mask = _arrays()
    ts_us = 1_700_000_000_000_041
    handle = manager.store("camera", ts_us, depth, confidence, mask)
    assert entered.wait(2.0)
    with pytest.raises(DuplicateSnapshotError):
        manager.store("camera", ts_us, depth, confidence, mask)
    release.set()
    handle.wait(timeout=5.0)
    with pytest.raises(DuplicateSnapshotError):
        manager.store("camera", ts_us, depth, confidence, mask)
    _close(manager)


def test_atomic_publish_never_overwrites_external_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = manager._write_snapshot  # type: ignore[attr-defined]

    def blocked(job):
        entered.set()
        assert release.wait(5.0)
        return original(job)

    monkeypatch.setattr(manager, "_write_snapshot", blocked)
    depth, confidence, mask = _arrays()
    handle = manager.store(
        "camera",
        1_700_000_000_000_049,
        depth,
        confidence,
        mask,
    )
    assert entered.wait(2.0)
    handle.path.mkdir(parents=True)
    sentinel = handle.path / "external-owner"
    sentinel.write_text("must survive", encoding="utf-8")
    release.set()

    with pytest.raises(DepthStoragePoisonedError, match="destination already exists"):
        handle.wait(timeout=5.0)
    assert sentinel.read_text(encoding="utf-8") == "must survive"
    _close(manager)


def _write_legacy_snapshot(path: Path, camera_id: str, ts_us: int) -> None:
    depth, confidence, mask = _arrays()
    group = zarr.open_group(str(path), mode="w")
    group.create_array("depth_z", data=depth)
    group.create_array("conf", data=confidence)
    group.create_array("mask", data=mask)
    group.attrs.update(
        camera_id=camera_id, timestamp_us=ts_us, snapshot_role="legacy_raw"
    )


def test_restart_ignores_interrupted_and_legacy_trees_then_explicitly_migrates(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, enable_async=False)
    depth, confidence, mask = _arrays()
    committed_ts = 1_700_000_000_000_051
    committed = manager.store("camera", committed_ts, depth, confidence, mask)
    committed.wait()
    _close(manager)

    legacy_ts = committed_ts + 1
    legacy_path = manager._snapshot_destination("camera", legacy_ts)  # type: ignore[attr-defined]
    _write_legacy_snapshot(legacy_path, "camera", legacy_ts)
    interrupted = legacy_path.parent / f".{legacy_ts + 1}.deadbeef.staging"
    _write_legacy_snapshot(interrupted, "camera", legacy_ts + 1)

    restarted = _manager(tmp_path)
    assert [ts for ts, _path in restarted.list_snapshot_entries("camera")] == [
        committed_ts
    ]
    report = restarted.startup_report()
    assert report["committed_snapshot_count"] == 1
    assert report["legacy_entries"] == [
        {"path": str(legacy_path), "reason": "legacy_missing_commit_manifest"}
    ]
    assert interrupted.exists()

    migrated = restarted.migrate_legacy_snapshot(legacy_path)
    assert migrated.ts_us == legacy_ts
    assert (
        DepthStorageManager._read_and_validate_commit_manifest(migrated.path)["state"]
        == "committed"
    )
    assert [ts for ts, _path in restarted.list_snapshot_entries("camera")] == [
        committed_ts,
        legacy_ts,
    ]
    _close(restarted)


def test_restart_rejects_corrupt_committed_snapshot(tmp_path: Path) -> None:
    manager = _manager(tmp_path, enable_async=False)
    depth, confidence, mask = _arrays()
    receipt = manager.store(
        "camera",
        1_700_000_000_000_059,
        depth,
        confidence,
        mask,
    ).wait()
    _close(manager)
    manifest_path = receipt.path / DepthStorageManager._COMMIT_MANIFEST_NAME
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(DepthStorageError, match="invalid committed depth snapshots"):
        _manager(tmp_path)


def test_fusion_holds_source_leases_and_returns_exact_atomic_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    depth, confidence, mask = _arrays()
    entries = []
    source_write_ids = []
    for offset in range(3):
        ts_us = 1_700_000_000_000_061 + offset
        handle = manager.store("camera", ts_us, depth + offset * 0.01, confidence, mask)
        receipt = handle.wait(timeout=5.0)
        entries.append((ts_us, receipt.path))
        source_write_ids.append(receipt.write_id)

    original = manager._fuse_depth_datasets  # type: ignore[attr-defined]
    observed_pins: list[tuple[int, ...]] = []

    def inspect_pins(snapshots, **kwargs):
        observed_pins.append(
            tuple(manager.read_pin_count(path) for _ts, path, _data in snapshots)
        )
        old_retention = manager._retention_us  # type: ignore[attr-defined]
        manager._retention_us = 1  # type: ignore[attr-defined]
        assert manager.prune("camera") == 0
        try:
            return original(snapshots, **kwargs)
        finally:
            manager._retention_us = old_retention  # type: ignore[attr-defined]

    monkeypatch.setattr(manager, "_fuse_depth_datasets", inspect_pins)
    fused_path, meta = manager.fuse_snapshot_entries(
        "camera",
        entries,
        rgb=np.full((4, 6, 3), 99, dtype=np.uint8),
        ts_us=1_700_000_000_000_071,
    )

    assert observed_pins == [(1, 1, 1)]
    assert fused_path.exists()
    manifest = DepthStorageManager._read_and_validate_commit_manifest(fused_path)
    assert manifest["state"] == "committed"
    attrs = DepthStorageManager._snapshot_attr_dict(fused_path)
    assert attrs["snapshot_role"] == "capture_event_fused"
    assert attrs["source_snapshot_count"] == 3
    assert meta["fused_snapshot_path"] == str(fused_path)
    assert [row["write_id"] for row in meta["source_snapshots"]] == source_write_ids
    assert meta["fused_write_id"]
    assert meta["fused_manifest_sha256"]
    descriptor = manager.describe_snapshot(fused_path)
    assert isinstance(descriptor, SnapshotDescriptor)
    assert descriptor.write_id == meta["fused_write_id"]
    assert all(manager.read_pin_count(path) == 0 for _ts, path in entries)
    _close(manager)


def test_fusion_rejects_any_invalid_or_derived_member_without_changing_cohort(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    depth, confidence, mask = _arrays()
    entries = []
    for offset in range(2):
        ts_us = 1_700_000_000_000_072 + offset
        receipt = manager.store("camera", ts_us, depth, confidence, mask).wait(
            timeout=5.0
        )
        entries.append((ts_us, receipt.path))

    before = manager.list_snapshot_entries("camera")
    with pytest.raises(DepthStorageError, match="not a committed readable entry"):
        manager.fuse_snapshot_entries(
            "camera",
            [entries[0], (entries[1][0] + 100, tmp_path / "missing.zarr")],
        )
    assert manager.list_snapshot_entries("camera") == before

    small = np.ones((2, 2), dtype=np.float32)
    small_receipt = manager.store(
        "camera",
        1_700_000_000_000_077,
        small,
        small,
        np.ones((2, 2), dtype=np.uint8),
    ).wait(timeout=5.0)
    with pytest.raises(DepthStorageError, match="shape mismatch"):
        manager.fuse_snapshot_entries(
            "camera",
            [entries[0], (small_receipt.ts_us, small_receipt.path)],
        )

    derived_path, _meta = manager.fuse_snapshot_entries(
        "camera",
        entries,
        ts_us=1_700_000_000_000_078,
    )
    with pytest.raises(DepthStorageError, match="derived snapshot"):
        manager.fuse_snapshot_entries(
            "camera",
            [entries[0], (1_700_000_000_000_078, derived_path)],
            ts_us=1_700_000_000_000_079,
        )
    assert manager.latest_entry("camera", None) == derived_path
    _close(manager)


def test_read_lease_prevents_prune_until_release(tmp_path: Path) -> None:
    manager = _manager(tmp_path, enable_async=False)
    depth, confidence, mask = _arrays()
    receipts = []
    for offset in range(3):
        handle = manager.store(
            "camera",
            1_700_000_000_000_081 + offset,
            depth,
            confidence,
            mask,
        )
        receipts.append(handle.wait())

    manager._retention_us = 1  # type: ignore[attr-defined]
    lease = manager.acquire_read_lease(receipts[0].path)
    manager.prune("camera")
    assert receipts[0].path.exists()
    assert manager.list_snapshot_entries("camera") == [
        (receipts[0].ts_us, receipts[0].path)
    ]
    lease.release()
    manager.prune("camera")
    assert not receipts[0].path.exists()
    assert manager.list_snapshot_entries("camera") == []
    _close(manager)


def test_committed_read_corruption_raises_instead_of_becoming_no_data(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, enable_async=False)
    depth, confidence, mask = _arrays()
    receipt = manager.store(
        "camera",
        1_700_000_000_000_089,
        depth,
        confidence,
        mask,
    ).wait()
    manifest = DepthStorageManager._read_and_validate_commit_manifest(receipt.path)
    first_file = receipt.path / manifest["files"][0]["path"]
    with first_file.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(DepthStorageError, match="commit_file_manifest_mismatch"):
        manager.load_datasets(receipt.path)
    _close(manager)


def test_new_parent_chain_is_fsynced_through_storage_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsynced: list[Path] = []

    def record(path: Path) -> None:
        fsynced.append(Path(path))

    monkeypatch.setattr(DepthStorageManager, "_fsync_directory", staticmethod(record))
    manager = _manager(tmp_path, enable_async=False)
    depth, confidence, mask = _arrays()
    receipt = manager.store(
        "new-camera",
        1_700_000_000_000_090,
        depth,
        confidence,
        mask,
    ).wait()

    hour = receipt.path.parent
    date = hour.parent
    camera = date.parent
    assert {tmp_path.resolve(), camera, date, hour}.issubset(set(fsynced))
    _close(manager)


def test_retention_enforcer_failure_poison_is_observable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures = []
    invoked = threading.Event()

    def fail_prune(_manager, _camera_id=None):
        invoked.set()
        raise OSError("injected retention failure")

    monkeypatch.setattr(DepthStorageManager, "prune", fail_prune)
    manager = _manager(
        tmp_path,
        on_failure=failures.append,
        enforce_async=True,
    )
    assert invoked.wait(2.0)
    deadline = time.monotonic() + 2.0
    while manager.poison is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.poison is not None
    assert manager.poison.message == "injected retention failure"
    assert failures == [manager.poison]
    _close(manager)


def test_store_copies_rgb_and_manifest_prevents_post_commit_mutation(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, enable_async=False)
    depth, confidence, mask = _arrays()
    rgb = np.zeros((4, 6, 4), dtype=np.uint8)
    rgb[..., 0] = 17
    original = rgb.copy()
    handle = manager.store(
        "camera", 1_700_000_000_000_091, depth, confidence, mask, rgb=rgb
    )
    receipt = handle.wait()
    rgb[...] = 255

    datasets = manager.load_datasets(receipt.path)
    assert datasets is not None
    assert np.array_equal(
        original, np.concatenate((datasets["rgb"], original[..., 3:4]), axis=2)
    )
    manifest = DepthStorageManager._read_and_validate_commit_manifest(receipt.path)
    assert manifest["write_id"] == handle.write_id
    assert manifest["sequence"] == handle.sequence
    _close(manager)
