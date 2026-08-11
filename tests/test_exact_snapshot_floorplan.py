from __future__ import annotations

import copy
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from geometry.depth_source import DepthStorageError, DepthStorageManager


def _manager(tmp_path: Path, camera_id: str) -> DepthStorageManager:
    manager = DepthStorageManager(
        base_path=tmp_path / "depth",
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        max_total_bytes=None,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
        min_conf=0.0,
    )
    manager.calibration_bundle = {
        "cameras": {
            "K": {camera_id: [4.0, 4.0, 1.5, 1.5]},
            "E": {
                camera_id: np.eye(4, dtype=np.float32)
                .reshape(-1, order="F")
                .astype(float)
                .tolist()
            },
        }
    }
    return manager


def _write(
    manager: DepthStorageManager,
    camera_id: str,
    timestamp_us: int,
    value: float,
):
    depth = np.full((4, 4), value, dtype=np.float32)
    confidence = np.ones_like(depth, dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)
    receipt = manager.store(camera_id, timestamp_us, depth, confidence, mask).wait(
        timeout=2.0
    )
    return manager.describe_snapshot(receipt.path)


def test_exact_floorplan_cannot_race_to_newer_snapshot(tmp_path: Path) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        base_timestamp_us = time.time_ns() // 1_000 - 10_000
        older = _write(manager, camera_id, base_timestamp_us, 1.0)
        newer = _write(manager, camera_id, base_timestamp_us + 1, 2.0)

        exact = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.1,
            max_extent_m=5.0,
            snapshot_ref=older.storage_ref,
            snapshot_id=older.write_id,
            snapshot_content_sha256=older.content_sha256,
        )
        latest = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.1,
            max_extent_m=5.0,
        )

        assert exact.get("error") is None
        assert exact["snapshot_ts"] == older.ts_us
        assert exact["snapshot_ref"] == older.storage_ref
        assert exact["snapshot_id"] == older.write_id
        assert exact["snapshot_content_sha256"] == older.content_sha256
        assert latest.get("error") is None, latest
        assert latest["snapshot_ts"] == newer.ts_us
        assert latest["snapshot_id"] == newer.write_id
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed


def test_exact_depth_read_revalidates_full_snapshot_identity(tmp_path: Path) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        descriptor = _write(manager, camera_id, time.time_ns() // 1_000, 3.0)
        payload = manager.load_depth_snapshot_exact(
            camera_id=camera_id,
            storage_ref=descriptor.storage_ref,
            snapshot_id=descriptor.write_id,
            content_sha256=descriptor.content_sha256,
        )
        assert payload["ts"] == descriptor.ts_us
        assert payload["snapshot_ref"] == descriptor.storage_ref
        assert payload["snapshot_id"] == descriptor.write_id
        assert payload["snapshot_content_sha256"] == descriptor.content_sha256
        assert payload["shape"] == [4, 4]
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed


def test_floorplan_cache_only_miss_performs_no_generation_or_persist(
    tmp_path: Path,
) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        _write(manager, camera_id, time.time_ns() // 1_000, 2.0)
        payload = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )
        assert payload["error"] == "no_cached_floorplan"
        assert not manager._floorplan_cache  # type: ignore[attr-defined]
        assert not list((manager.base_path / "floorplans").rglob("*.json"))
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed


def test_exact_floorplan_atomically_publishes_cache_only_latest_alias(
    tmp_path: Path,
) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        descriptor = _write(manager, camera_id, time.time_ns() // 1_000, 2.5)
        exact = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            snapshot_ref=descriptor.storage_ref,
            snapshot_id=descriptor.write_id,
            snapshot_content_sha256=descriptor.content_sha256,
        )
        assert exact.get("error") is None, exact
        assert exact["served_from_cache"] is False

        keys_before = tuple(manager._floorplan_cache)  # type: ignore[attr-defined]
        assert len(keys_before) == 2
        assert {key[-1] for key in keys_before} == {
            descriptor.write_id,
            "latest",
        }
        cached = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )

        assert cached["served_from_cache"] is True
        assert cached["snapshot_ref"] == exact["snapshot_ref"]
        assert cached["snapshot_id"] == exact["snapshot_id"]
        assert (
            cached["snapshot_content_sha256"]
            == exact["snapshot_content_sha256"]
        )
        assert cached["snapshot_ts"] == exact["snapshot_ts"]
        assert tuple(manager._floorplan_cache) == keys_before  # type: ignore[attr-defined]
        persisted = list((manager.base_path / "floorplans").rglob("*.json"))
        assert len(persisted) == 1

        cached["bounds"]["min_x"] = 999.0
        cached_again = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )
        assert cached_again["bounds"] == exact["bounds"]
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed


def test_older_exact_floorplan_cannot_downgrade_newer_persisted_alias(
    tmp_path: Path,
) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        base_timestamp_us = time.time_ns() // 1_000 - 10_000
        older = _write(manager, camera_id, base_timestamp_us, 1.5)
        newer = _write(manager, camera_id, base_timestamp_us + 1, 2.5)

        newer_floorplan = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            snapshot_ref=newer.storage_ref,
            snapshot_id=newer.write_id,
            snapshot_content_sha256=newer.content_sha256,
        )
        assert newer_floorplan.get("error") is None, newer_floorplan

        # This ordering models an older concurrent generation completing after
        # the newer accepted artifact. Its exact response remains valid, but it
        # must not replace either latest alias.
        older_floorplan = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            snapshot_ref=older.storage_ref,
            snapshot_id=older.write_id,
            snapshot_content_sha256=older.content_sha256,
        )
        assert older_floorplan.get("error") is None, older_floorplan
        assert older_floorplan["snapshot_id"] == older.write_id

        latest_memory = manager.generate_topdown_floorplan(
            camera_id,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )
        assert latest_memory["snapshot_id"] == newer.write_id

        manager._floorplan_cache.clear()  # type: ignore[attr-defined]
        latest_disk = manager.generate_topdown_floorplan(
            camera_id,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )
        assert latest_disk["snapshot_id"] == newer.write_id

        # A memory-cache reset/eviction within the runtime session leaves disk
        # authoritative and memory empty. An exact request for an older retained
        # snapshot must still return it without installing it as the latest alias.
        manager._floorplan_cache.clear()  # type: ignore[attr-defined]
        stale_exact = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            snapshot_ref=older.storage_ref,
            snapshot_id=older.write_id,
            snapshot_content_sha256=older.content_sha256,
        )
        assert stale_exact["snapshot_id"] == older.write_id
        latest_after_stale = manager.generate_topdown_floorplan(
            camera_id,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )
        assert latest_after_stale["snapshot_id"] == newer.write_id
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed


def test_disk_hydration_and_newer_publication_are_lock_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        base_timestamp_us = time.time_ns() // 1_000 - 10_000
        older = _write(manager, camera_id, base_timestamp_us, 1.5)
        newer = _write(manager, camera_id, base_timestamp_us + 1, 2.5)
        older_floorplan = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            snapshot_ref=older.storage_ref,
            snapshot_id=older.write_id,
            snapshot_content_sha256=older.content_sha256,
        )
        newer_floorplan = copy.deepcopy(older_floorplan)
        newer_floorplan.update(
            {
                "ts": int(older_floorplan["ts"]) + 1,
                "snapshot_ts": newer.ts_us,
                "snapshot_ref": newer.storage_ref,
                "snapshot_id": newer.write_id,
                "snapshot_content_sha256": newer.content_sha256,
                "served_from_cache": False,
            }
        )
        manager._floorplan_cache.clear()  # type: ignore[attr-defined]

        original_load = manager._load_floorplan_from_disk  # type: ignore[attr-defined]
        disk_read_started = threading.Event()
        release_disk_read = threading.Event()

        def paused_disk_load(*args, **kwargs):  # type: ignore[no-untyped-def]
            payload = original_load(*args, **kwargs)
            disk_read_started.set()
            assert release_disk_read.wait(timeout=2.0)
            return payload

        monkeypatch.setattr(manager, "_load_floorplan_from_disk", paused_disk_load)
        cache_result: list[dict] = []
        failures: list[BaseException] = []

        def hydrate_cache() -> None:
            try:
                cache_result.append(
                    manager.generate_topdown_floorplan(
                        camera_id,
                        grid_res_m=0.2,
                        max_extent_m=5.0,
                        cache_only=True,
                    )
                )
            except BaseException as exc:  # pragma: no cover - assertion aid
                failures.append(exc)

        publication_started = threading.Event()
        publication_done = threading.Event()

        def publish_newer() -> None:
            publication_started.set()
            try:
                manager._publish_floorplan_alias_transaction(  # type: ignore[attr-defined]
                    camera_id,
                    0.2,
                    5.0,
                    newer_floorplan,
                    exact_cache_key=(camera_id, 0.2, 5.0, newer.write_id),
                )
            except BaseException as exc:  # pragma: no cover - assertion aid
                failures.append(exc)
            finally:
                publication_done.set()

        hydration_thread = threading.Thread(target=hydrate_cache)
        hydration_thread.start()
        assert disk_read_started.wait(timeout=2.0)
        publication_thread = threading.Thread(target=publish_newer)
        publication_thread.start()
        assert publication_started.wait(timeout=2.0)
        assert manager._cache_lock.locked()  # type: ignore[attr-defined]
        assert not publication_done.is_set()
        release_disk_read.set()
        hydration_thread.join(timeout=2.0)
        publication_thread.join(timeout=2.0)

        assert not hydration_thread.is_alive()
        assert not publication_thread.is_alive()
        assert not failures
        assert cache_result[0]["snapshot_id"] == older.write_id
        latest = manager.generate_topdown_floorplan(
            camera_id,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )
        assert latest["snapshot_id"] == newer.write_id
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed


def test_exact_floorplan_persistence_failure_is_not_reported_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        descriptor = _write(
            manager,
            camera_id,
            time.time_ns() // 1_000,
            2.5,
        )
        floorplan_path = manager._floorplan_path(  # type: ignore[attr-defined]
            camera_id,
            0.2,
            5.0,
        )
        original_replace = Path.replace

        def fail_floorplan_replace(path: Path, target: Path) -> Path:
            if path == floorplan_path.with_suffix(floorplan_path.suffix + ".tmp"):
                raise OSError("simulated floorplan persistence failure")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_floorplan_replace)
        with pytest.raises(DepthStorageError, match="floorplan_persistence_failed"):
            manager.generate_topdown_floorplan(
                camera_id,
                max_age_sec=0.0,
                grid_res_m=0.2,
                max_extent_m=5.0,
                snapshot_ref=descriptor.storage_ref,
                snapshot_id=descriptor.write_id,
                snapshot_content_sha256=descriptor.content_sha256,
            )

        assert not manager._floorplan_cache  # type: ignore[attr-defined]
        assert not floorplan_path.exists()
        assert not floorplan_path.with_suffix(floorplan_path.suffix + ".tmp").exists()
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed


def test_exact_floorplan_and_public_depth_survive_memory_cache_reset_and_retention(
    tmp_path: Path,
) -> None:
    camera_id = "living"
    manager = _manager(tmp_path, camera_id)
    try:
        base_timestamp_us = time.time_ns() // 1_000 - 10_000
        first = _write(manager, camera_id, base_timestamp_us, 2.5)
        second = _write(manager, camera_id, base_timestamp_us + 1, 2.5)
        fused_path, _meta = manager.fuse_snapshot_entries(
            camera_id,
            ((first.ts_us, first.path), (second.ts_us, second.path)),
            ts_us=base_timestamp_us + 2,
        )
        fused = manager.describe_snapshot(fused_path)
        exact = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=0.0,
            grid_res_m=0.2,
            max_extent_m=5.0,
            snapshot_ref=fused.storage_ref,
            snapshot_id=fused.write_id,
            snapshot_content_sha256=fused.content_sha256,
        )
        assert exact.get("error") is None, exact

        manager._floorplan_cache.clear()  # type: ignore[attr-defined]
        manager._retention_us = 1  # type: ignore[attr-defined]
        manager.prune(camera_id)

        assert fused_path.exists()
        cached_depth = manager.describe_latest_depth_bulk(camera_id)
        assert cached_depth is not None
        assert cached_depth["snapshot_id"] == fused.write_id
        cached_floorplan = manager.generate_topdown_floorplan(
            camera_id,
            grid_res_m=0.2,
            max_extent_m=5.0,
            cache_only=True,
        )
        assert cached_floorplan.get("error") is None, cached_floorplan
        assert cached_floorplan["served_from_cache"] is True
        assert cached_floorplan["snapshot_id"] == fused.write_id
    finally:
        assert manager.shutdown(wait=True, timeout=2.0).completed
