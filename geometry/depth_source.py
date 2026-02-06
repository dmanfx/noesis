"""HTTP client and storage utilities for MapAnything depth inference."""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import queue
import shutil
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple, Sequence

from concurrent.futures import Future

import cv2
import numpy as np
import requests
import scipy.ndimage as ndi
import zarr

from adapters.mapanything_adapter import ViewBuildResult, build_mono_view
from geometry.homography import parse_extrinsics
from mapanything_config import ServiceConfig, load_service_config
from utils.rate_limited_logger import RateLimitedLogger

try:  # zarr v3 codec shim
    from zarr.codecs import Blosc as _ZarrBlosc  # type: ignore
except Exception:
    _ZarrBlosc = None  # type: ignore

try:
    from numcodecs import Blosc as _NumcodecsBlosc  # type: ignore
except Exception:
    _NumcodecsBlosc = None  # type: ignore

_FLOORPLAN_FRAME = "camera_local_ground"


def _infer_image_flips_from_extrinsics(
    extrinsics_col_major: Sequence[float],
) -> Optional[Tuple[bool, bool]]:
    try:
        R_wc, _ = parse_extrinsics(extrinsics_col_major)
        forward = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        f_norm = float(np.linalg.norm(forward))
        if f_norm > 1e-6:
            forward = forward / f_norm
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right_ref = np.cross(world_up, forward)
        r_norm = float(np.linalg.norm(right_ref))
        if r_norm <= 1e-6:
            right_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right_ref = right_ref / r_norm
        up_ref = np.cross(forward, right_ref)
        u_norm = float(np.linalg.norm(up_ref))
        if u_norm <= 1e-6:
            up_ref = world_up
        else:
            up_ref = up_ref / u_norm

        right = R_wc @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = R_wc @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r_actual = float(np.dot(right, right_ref))
        u_actual = float(np.dot(up, up_ref))

        flip_u = r_actual < 0.0
        flip_v = u_actual > 0.0
        return bool(flip_u), bool(flip_v)
    except Exception:
        return None


def _expected_floorplan_flip(
    calib_bundle: Any,
    camera_id: str,
) -> Optional[Tuple[bool, bool]]:
    if not isinstance(calib_bundle, dict):
        return None
    cameras_node = calib_bundle.get("cameras")
    if not isinstance(cameras_node, dict):
        return None
    e_table = cameras_node.get("E") if isinstance(cameras_node, dict) else None
    extr = e_table.get(camera_id) if isinstance(e_table, dict) else None
    if not isinstance(extr, (list, tuple)) or len(extr) != 16:
        return None
    return _infer_image_flips_from_extrinsics(extr)


def _flip_payload_matches(
    payload: Any,
    expected: Optional[Tuple[bool, bool]],
) -> bool:
    if expected is None:
        return True
    if not isinstance(payload, dict):
        return False
    try:
        flip_u = bool(payload.get("u"))
        flip_v = bool(payload.get("v"))
    except Exception:
        return False
    return (flip_u, flip_v) == (bool(expected[0]), bool(expected[1]))


@dataclass(frozen=True)
class DepthSummary:
    median: float
    p10: float
    p90: float
    conf_mean: float
    valid_ratio: float
    sample_count: int


@dataclass(frozen=True)
class DepthResult:
    camera_id: str
    ts_us: int
    depth: np.ndarray
    conf: np.ndarray
    mask: np.ndarray
    intrinsics: Optional[np.ndarray]
    native_intrinsics: Optional[np.ndarray]
    summary: DepthSummary
    storage_path: Path


@dataclass(frozen=True)
class _SnapshotJob:
    camera_id: str
    ts_us: int
    depth: np.ndarray
    conf: np.ndarray
    mask: np.ndarray
    dest_path: Path


@dataclass(frozen=True)
class _BatchItem:
    camera_id: str
    timestamp_s: float
    view_result: ViewBuildResult
    view_payload: Dict[str, object]
    future: Future


class DepthStorageManager:
    """Persist depth outputs to Zarr for later consumption with retention enforcement."""

    def __init__(
        self,
        base_path: Path,
        max_snapshots_per_camera: int,
        retention_minutes: float,
        max_total_bytes: Optional[int] = None,
        *,
        enable_async: bool = True,
        max_queue_size: int = 32,
        worker_count: int = 1,
        max_worker_count: int = 0,
        # New tuning knobs
        enforce_async: bool = True,
        enforce_interval_s: float = 1.0,
        size_hysteresis_ratio: float = 0.9,
        zarr_clevel: int = 5,
        zarr_chunk_px: int = 128,
        min_conf: float = float("nan"),
        floorplan_store_dir: Optional[Path] = None,
        max_depth_cache_entries: int = 16,
        max_floorplan_cache_entries: int = 24,
        max_normals_cache_entries: int = 8,
    ) -> None:
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._indices: Dict[str, Deque[Tuple[int, Path]]] = {}
        self._logger = logging.getLogger(__name__)
        self._max_snapshots = max(max_snapshots_per_camera, 0)
        self._retention_us = max(0, int(retention_minutes * 60.0 * 1_000_000))
        self._max_total_bytes = max_total_bytes if (max_total_bytes is not None and max_total_bytes > 0) else None
        self._async_enabled = bool(enable_async)
        self._max_queue_size = max(1, int(max_queue_size)) if self._async_enabled else 0
        self._initial_workers = 0
        self._max_worker_count = 0
        if self._async_enabled:
            requested_workers = int(worker_count)
            default_workers = self._default_worker_target()
            if requested_workers <= 0:
                requested_workers = default_workers
            self._initial_workers = max(1, requested_workers)
            requested_max = int(max_worker_count)
            if requested_max <= 0:
                requested_max = max(self._initial_workers, default_workers)
            self._max_worker_count = max(self._initial_workers, requested_max)
            self._max_queue_size = max(self._max_queue_size, self._initial_workers * 4)
        self._queue: Optional["queue.Queue[_SnapshotJob]"] = None
        self._stop_event: Optional[threading.Event] = None
        self._writer_threads: List[threading.Thread] = []
        self._worker_lock = threading.Lock()
        self._worker_name_counter = 0
        self._queue_put_timeout = 1.0
        self._last_queue_full_warning: float = 0.0
        # Retention/size enforcement tuning
        self._enforce_async = bool(enforce_async)
        self._enforce_interval_s = max(0.05, float(enforce_interval_s))
        self._size_hysteresis_ratio = min(1.0, max(0.1, float(size_hysteresis_ratio)))
        self._zarr_clevel = max(0, int(zarr_clevel))
        self._zarr_chunk_px = int(zarr_chunk_px)
        self._cache_lock = threading.Lock()
        self._depth_payload_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._floorplan_cache: "OrderedDict[Tuple[str, float, float], Dict[str, Any]]" = OrderedDict()
        self._normals_cache: "OrderedDict[Tuple[str, int, str, str], Dict[str, Any]]" = OrderedDict()
        self._floorplan_store_dir = Path(floorplan_store_dir).resolve() if floorplan_store_dir else (self.base_path / "floorplans")
        self._floorplan_store_dir.mkdir(parents=True, exist_ok=True)
        self._max_depth_cache_entries = max(1, int(max_depth_cache_entries))
        self._max_floorplan_cache_entries = max(1, int(max_floorplan_cache_entries))
        self._max_normals_cache_entries = max(1, int(max_normals_cache_entries))
        self.min_conf = float(min_conf)
        # Background enforcement thread
        self._enforce_thread: Optional[threading.Thread] = None
        self._enforce_stop = threading.Event()
        if self._max_total_bytes is not None and self._max_total_bytes < 10 * 1024 * 1024:
            self._logger.warning(
                "Configured max_total_bytes=%s is very small; increasing to 10MB minimum",
                self._max_total_bytes,
            )
            self._max_total_bytes = 10 * 1024 * 1024
        self._seed_existing_entries()
        if self._async_enabled:
            self._start_writer()
        # Start async enforcement if enabled
        if self._enforce_async:
            self._start_enforcer()

    def _default_worker_target(self) -> int:
        cpu_count = os.cpu_count() or 1
        return max(2, min(8, cpu_count))

    def _seed_existing_entries(self) -> None:
        """Populate in-memory indices from disk on startup and prune if needed."""
        for camera_dir in sorted(self.base_path.glob("*")):
            if not camera_dir.is_dir():
                continue
            camera_id = camera_dir.name
            index = self._indices.setdefault(camera_id, deque())
            zarr_paths = []
            for path in camera_dir.rglob("*.zarr"):
                try:
                    ts = int(path.stem)
                except ValueError:
                    continue
                zarr_paths.append((ts, path))
            if not zarr_paths:
                continue
            zarr_paths.sort(key=lambda item: item[0])
            index.extend(zarr_paths)
            self._enforce_limits(camera_id, index, now_ts=self._current_time_us())

    def _current_time_us(self) -> int:
        return int(time.time() * 1_000_000)

    def _get_lock(self, camera_id: str) -> threading.Lock:
        return self._locks.setdefault(camera_id, threading.Lock())

    def _get_index(self, camera_id: str) -> Deque[Tuple[int, Path]]:
        return self._indices.setdefault(camera_id, deque())

    def _start_writer(self) -> None:
        if self._queue is not None or self._initial_workers <= 0:
            return
        self._queue = queue.Queue(maxsize=self._max_queue_size)
        self._stop_event = threading.Event()
        with self._worker_lock:
            for _ in range(self._initial_workers):
                self._spawn_worker_locked()

    def _spawn_worker_locked(self) -> None:
        if self._queue is None or self._stop_event is None:
            return
        self._worker_name_counter += 1
        thread = threading.Thread(
            target=self._writer_loop,
            name=f"DepthSnapshotWriter-{self._worker_name_counter}",
            daemon=True,
        )
        self._writer_threads.append(thread)
        thread.start()

    def _writer_loop(self) -> None:
        # Defensive loop that tolerates shutdown races and empty queue timeouts.
        # This is intentionally conservative to avoid noisy thread exceptions
        # while retaining async snapshot functionality.
        try:
            q = self._queue
            stop = self._stop_event
        except Exception:
            q = None
            stop = None
        while True:
            try:
                # Refresh local refs each iteration in case shutdown mutated them
                if q is None or stop is None:
                    q = self._queue
                    stop = self._stop_event
                if q is None:
                    # Queue no longer available; exit quietly
                    break
                try:
                    job = q.get(timeout=0.2)
                except queue.Empty:
                    try:
                        if stop is None:
                            stop = self._stop_event
                        if stop is not None and stop.is_set():
                            break
                    except Exception:
                        # If stop flag is unavailable, exit defensively
                        break
                    continue
                try:
                    self._write_snapshot(job)
                except Exception as exc:
                    # Keep processing other jobs; log at error level
                    self._logger.error(
                        "Depth snapshot write failed for %s to %s: %s",
                        getattr(job, "camera_id", "unknown"),
                        getattr(job, "dest_path", None),
                        exc,
                    )
                finally:
                    try:
                        q.task_done()
                    except Exception:
                        pass
            except Exception:
                # Any unexpected error should not tear down the thread noisily.
                # Re-check shutdown and either continue or exit quietly.
                try:
                    stop = self._stop_event
                    if stop is not None and stop.is_set():
                        break
                except Exception:
                    break
                time.sleep(0.05)

    def _start_enforcer(self) -> None:
        if self._enforce_thread is not None:
            return
        def _loop() -> None:
            while not self._enforce_stop.is_set():
                try:
                    # Prune all cameras; uses per-camera locks internally
                    self.prune()
                except Exception:
                    pass
                # Sleep a bit to batch work and avoid thrash
                self._enforce_stop.wait(self._enforce_interval_s)
        self._enforce_thread = threading.Thread(target=_loop, name="DepthRetentionEnforcer", daemon=True)
        self._enforce_thread.start()

    def shutdown(self, *, wait: bool = True) -> None:
        # Stop background enforcer
        try:
            self._enforce_stop.set()
            if self._enforce_thread and wait:
                self._enforce_thread.join(timeout=2.0)
        except Exception:
            pass
        self._enforce_thread = None
        # Stop async writers
        if not self._async_enabled or self._stop_event is None:
            return
        self._stop_event.set()
        # Always give workers a small window to exit to avoid races
        join_timeout = 2.0 if wait else 0.5
        threads = self._collect_alive_threads()
        for thread in threads:
            try:
                thread.join(timeout=join_timeout)
            except Exception:
                pass
        with self._worker_lock:
            self._writer_threads.clear()
        # Only clear references after attempting joins to prevent attr races
        self._queue = None
        self._stop_event = None

    def flush(self, timeout: Optional[float] = None) -> None:
        if not self._async_enabled or self._queue is None:
            return
        if timeout is None:
            self._queue.join()
            return
        deadline = time.time() + max(0.0, timeout)
        while getattr(self._queue, "unfinished_tasks", 0) > 0:
            if time.time() >= deadline:
                break
            time.sleep(0.01)

    def _create_job(
        self,
        camera_id: str,
        ts_us: int,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
        dest_path: Path,
    ) -> _SnapshotJob:
        depth_c = np.ascontiguousarray(depth, dtype=np.float32).copy()
        conf_c = np.ascontiguousarray(conf, dtype=np.float32).copy()
        mask_c = np.ascontiguousarray(mask, dtype=np.uint8).copy()
        return _SnapshotJob(camera_id, ts_us, depth_c, conf_c, mask_c, dest_path)

    def _register_snapshot(self, camera_id: str, ts_us: int, dest_path: Path) -> None:
        # Keep lock ordering consistent with load_latest_depth (cache_lock -> camera_lock)
        # to avoid deadlocks and to ensure we don't repopulate stale cached payloads
        # while a new snapshot is being registered.
        with self._cache_lock:
            lock = self._get_lock(camera_id)
            with lock:
                index = self._get_index(camera_id)
                index.append((ts_us, dest_path))
                # Defer enforcement to background thread if enabled
                if not self._enforce_async:
                    self._enforce_limits(camera_id, index)
            # Invalidate cached WS payload so subsequent RPCs see the newest snapshot.
            self._depth_payload_cache.pop(camera_id, None)

    def _make_blosc_compressor(self) -> Optional[Any]:
        if self._zarr_clevel <= 0:
            return None
        if _ZarrBlosc is not None:
            try:
                return _ZarrBlosc(cname="zstd", clevel=int(self._zarr_clevel))
            except Exception:
                self._logger.debug("zarr.codecs.Blosc unavailable; falling back to numcodecs")
        if _NumcodecsBlosc is not None:
            try:
                shuffle = getattr(_NumcodecsBlosc, "SHUFFLE", 1)
                return _NumcodecsBlosc(cname="zstd", clevel=int(self._zarr_clevel), shuffle=shuffle, blocksize=0)
            except Exception:
                self._logger.debug("numcodecs.Blosc unavailable; storing uncompressed")
        return None

    def _create_zarr_dataset(
        self,
        root: "zarr.hierarchy.Group",
        name: str,
        data: np.ndarray,
        chunk_shape: Tuple[int, int],
        compressor: Optional[Any],
    ) -> None:
        create_kwargs = {
            "shape": tuple(int(dim) for dim in data.shape),
            "data": data,
            "chunks": tuple(int(dim) for dim in chunk_shape),
            "overwrite": True,
        }
        if compressor is not None:
            create_kwargs["compressor"] = compressor
        try:
            root.create_dataset(name, **create_kwargs)
            return
        except TypeError as exc:
            # Retry using the zarr v3 compressors API, otherwise fall back to uncompressed.
            self._logger.debug("create_dataset fallback for %s due to %s", name, exc)
            create_kwargs.pop("compressor", None)
            if compressor is not None:
                try:
                    create_kwargs["compressors"] = [compressor]
                    root.create_dataset(name, **create_kwargs)
                    return
                except TypeError:
                    create_kwargs.pop("compressors", None)
        root.create_dataset(name, **create_kwargs)

    def _write_snapshot(self, job: _SnapshotJob) -> None:
        job.dest_path.parent.mkdir(parents=True, exist_ok=True)
        compressor = self._make_blosc_compressor()
        root = zarr.open_group(str(job.dest_path), mode="w")
        if self._zarr_chunk_px and self._zarr_chunk_px > 0:
            chunk_shape = (min(self._zarr_chunk_px, job.depth.shape[0]), min(self._zarr_chunk_px, job.depth.shape[1]))
        else:
            # Single-chunk per array to minimize file count
            chunk_shape = job.depth.shape
        self._create_zarr_dataset(root, "depth_z", job.depth, chunk_shape, compressor)
        self._create_zarr_dataset(root, "conf", job.conf, chunk_shape, compressor)
        self._create_zarr_dataset(root, "mask", job.mask, chunk_shape, compressor)
        root.attrs.update(
            camera_id=job.camera_id,
            timestamp_us=int(job.ts_us),
            stored_at=time.time(),
            shape=json.dumps(job.depth.shape),
        )
        self._register_snapshot(job.camera_id, job.ts_us, job.dest_path)

    def _remove_snapshot(self, path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                # Clean up empty parent directories up to camera root
                parent = path.parent
                for _ in range(2):
                    if parent == self.base_path or not parent.exists():
                        break
                    try:
                        next(parent.iterdir())
                    except StopIteration:
                        parent.rmdir()
                    parent = parent.parent
        except Exception as exc:
            self._logger.debug("Failed to remove snapshot %s: %s", path, exc)

    def _enforce_limits(
        self,
        camera_id: str,
        index: Deque[Tuple[int, Path]],
        now_ts: Optional[int] = None,
    ) -> None:
        if not index:
            return
        now_ts = now_ts if now_ts is not None else self._current_time_us()
        retention_cutoff = None
        if self._retention_us > 0:
            retention_cutoff = now_ts - self._retention_us

        removed = 0
        # Enforce retention duration first so age limit always wins
        if retention_cutoff is not None:
            while index and index[0][0] < retention_cutoff:
                _, path = index.popleft()
                self._remove_snapshot(path)
                removed += 1

        # Enforce max snapshot count with hysteresis to prevent thrash
        if self._max_snapshots > 0 and len(index) > self._max_snapshots:
            # determine floor based on 90% of max (at least 1)
            target_len = max(1, int(self._max_snapshots * 0.9))
            while len(index) > target_len:
                _, path = index.popleft()
                self._remove_snapshot(path)
                removed += 1

        if self._max_total_bytes is not None:
            removed += self._enforce_total_size(camera_id, index)

        if removed:
            self._logger.info(
                "Pruned %s depth snapshots for camera %s (max=%s, retention_us=%s)",
                removed,
                camera_id,
                self._max_snapshots,
                self._retention_us,
            )

    def _enforce_total_size(self, camera_id: str, index: Deque[Tuple[int, Path]]) -> int:
        if not index:
            return 0
        total_bytes = 0
        sizes: List[int] = []
        for _, path in index:
            try:
                size = sum(file.stat().st_size for file in path.rglob('*') if file.is_file())
            except FileNotFoundError:
                size = 0
            sizes.append(size)
            total_bytes += size

        removed = 0
        # Hysteresis: prune down to a target below the hard cap to avoid frequent rescans
        target_bytes = int(self._max_total_bytes * self._size_hysteresis_ratio)
        while total_bytes > target_bytes and index:
            ts, path = index.popleft()
            size = sizes.pop(0)
            total_bytes -= size
            self._remove_snapshot(path)
            removed += 1

        if removed and total_bytes > self._max_total_bytes:
            self._logger.warning(
                "Total depth storage still above quota after pruning (camera=%s remaining=%s max=%s)",
                camera_id,
                total_bytes,
                self._max_total_bytes,
            )
        return removed

    def prune(self, camera_id: Optional[str] = None) -> int:
        """Manual pruning entrypoint; returns number of snapshots removed."""
        total_removed = 0
        if camera_id:
            lock = self._get_lock(camera_id)
            with lock:
                index = self._get_index(camera_id)
                before = len(index)
                self._enforce_limits(camera_id, index)
                total_removed += max(0, before - len(index))
            return total_removed

        for cam_id in list(self._indices.keys()):
            total_removed += self.prune(cam_id)
        return total_removed

    def purge_all(self, camera_id: Optional[str] = None) -> int:
        """Remove all stored depth snapshots to start fresh."""
        if camera_id is not None:
            return self._purge_camera(camera_id)

        total = 0
        for cam_id in list(self._indices.keys()):
            total += self._purge_camera(cam_id)
        return total

    def _purge_camera(self, camera_id: str) -> int:
        removed = 0
        lock = self._get_lock(camera_id)
        with lock:
            index = self._get_index(camera_id)
            while index:
                _, path = index.popleft()
                self._remove_snapshot(path)
                removed += 1
        cam_dir = self.base_path / camera_id
        if cam_dir.exists():
            try:
                next(cam_dir.iterdir())
            except StopIteration:
                cam_dir.rmdir()
        return removed

    def store(
        self,
        camera_id: str,
        ts_us: int,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> Path:
        timestamp = datetime.utcfromtimestamp(ts_us / 1_000_000.0)
        date_dir = timestamp.strftime("%Y%m%d")
        hour_dir = timestamp.strftime("%H")
        dest_dir = self.base_path / camera_id / date_dir / hour_dir
        dest_path = dest_dir / f"{ts_us}.zarr"

        job = self._create_job(camera_id, ts_us, depth, conf, mask, dest_path)
        if self._async_enabled and self._queue is not None:
            try:
                self._queue.put(job, timeout=self._queue_put_timeout)
                return dest_path
            except queue.Full:
                if self._maybe_scale_workers():
                    try:
                        self._queue.put(job, timeout=self._queue_put_timeout)
                        return dest_path
                    except queue.Full:
                        pass
                now = time.time()
                if now - self._last_queue_full_warning >= 5.0:
                    self._logger.warning(
                        "Depth snapshot queue full; writing synchronously (size=%s)",
                        self._max_queue_size,
                    )
                    self._last_queue_full_warning = now

        self._write_snapshot(job)
        return dest_path

    def _collect_alive_threads(self) -> List[threading.Thread]:
        with self._worker_lock:
            alive = [thread for thread in self._writer_threads if thread.is_alive()]
            self._writer_threads = alive
            return list(alive)

    def _maybe_scale_workers(self) -> bool:
        if not self._async_enabled or self._queue is None:
            return False
        with self._worker_lock:
            alive = [thread for thread in self._writer_threads if thread.is_alive()]
            self._writer_threads = alive
            if len(alive) >= self._max_worker_count:
                return False
            self._spawn_worker_locked()
            return True

    def latest_entry(self, camera_id: str, ts_max: Optional[int]) -> Optional[Path]:
        lock = self._get_lock(camera_id)
        with lock:
            index = self._get_index(camera_id)
            if not index:
                return None
            # Drop missing files from the tail
            while index and not index[-1][1].exists():
                index.pop()
            if not index:
                return None
            if ts_max is None:
                return index[-1][1]
            # Find newest entry <= ts_max
            for ts, path in reversed(index):
                if ts <= ts_max and path.exists():
                    return path
            return None

    def all_cameras(self) -> Iterable[str]:
        return list(self._indices.keys())

    def load_datasets(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        try:
            group = zarr.open_group(str(path), mode='r')
            depth = np.array(group['depth_z'])
            conf = np.array(group['conf'])
            mask = np.array(group['mask'])
            return {'depth': depth, 'conf': conf, 'mask': mask}
        except Exception:
            return None

    def load_latest_depth(self, camera_id: str, ts_max_us: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the newest cached payload up to ts_max_us (microseconds)."""
        cache_key = camera_id
        ts_cutoff = None
        if ts_max_us is not None:
            try:
                ts_cutoff = int(ts_max_us)
            except Exception:
                ts_cutoff = None
        with self._cache_lock:
            cached = self._depth_payload_cache.get(cache_key)
            if cached:
                try:
                    cached_ts = int(cached.get('ts', 0) or 0)
                except Exception:
                    cached_ts = 0
                if ts_cutoff is None or cached_ts <= ts_cutoff:
                    return dict(cached)

        path = self.latest_entry(camera_id, ts_cutoff)
        if not path:
            return None
        datasets = self.load_datasets(path)
        if not datasets:
            return None
        depth = datasets['depth'].astype(np.float32, copy=False)
        conf = datasets['conf'].astype(np.float32, copy=False)
        mask = datasets['mask'].astype(np.uint8, copy=False)
        height, width = depth.shape[:2]
        ts_us = int(path.stem)
        payload = {
            'ts': ts_us,
            'depth_b64': base64.b64encode(depth.tobytes()).decode('ascii'),
            'conf_b64': base64.b64encode(conf.tobytes()).decode('ascii'),
            'mask_b64': base64.b64encode(mask.tobytes()).decode('ascii'),
            'shape': [int(height), int(width)],
        }
        with self._cache_lock:
            self._depth_payload_cache[cache_key] = dict(payload)
            self._depth_payload_cache.move_to_end(cache_key, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)
        return payload

    def _resolve_intrinsics_for_depth(
        self,
        camera_id: str,
        depth_shape: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        calib_bundle = getattr(self, "calibration_bundle", None) or {}
        cameras_node = calib_bundle.get("cameras") if isinstance(calib_bundle, dict) else {}
        k_table = cameras_node.get("K") if isinstance(cameras_node, dict) else {}
        intr = k_table.get(camera_id) if isinstance(k_table, dict) else None

        if intr is None:
            raise ValueError("missing_calibration")

        intr_arr = np.asarray(intr, dtype=np.float32).reshape(-1)
        if intr_arr.size == 4:
            fx, fy, cx, cy = [float(v) for v in intr_arr]
        elif intr_arr.size == 9:
            k_mat = intr_arr.reshape(3, 3)
            fx = float(k_mat[0, 0])
            fy = float(k_mat[1, 1])
            cx = float(k_mat[0, 2])
            cy = float(k_mat[1, 2])
        else:
            raise ValueError("bad_intrinsics")

        if not all(np.isfinite([fx, fy, cx, cy])) or fx == 0.0 or fy == 0.0:
            raise ValueError("invalid_intrinsics")

        target_h, target_w = depth_shape
        base_w = None
        base_h = None
        meta_node = calib_bundle.get("meta") if isinstance(calib_bundle, dict) else None
        specs_node = meta_node.get("camera_specs") if isinstance(meta_node, dict) else None
        spec = specs_node.get(camera_id) if isinstance(specs_node, dict) else None
        if isinstance(spec, dict):
            res = spec.get("resolution")
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                try:
                    base_w = int(res[0])
                    base_h = int(res[1])
                except Exception:
                    base_w = None
                    base_h = None
            if base_w is None or base_h is None:
                try:
                    base_w = int(spec.get("width", 0) or 0) or base_w
                    base_h = int(spec.get("height", 0) or 0) or base_h
                except Exception:
                    base_w = base_w
                    base_h = base_h
        if base_w is None or base_h is None:
            try:
                base_w = int(round(float(cx) * 2.0))
                base_h = int(round(float(cy) * 2.0))
            except Exception:
                base_w = None
                base_h = None
        if (
            base_w and base_h and target_w and target_h
            and base_w > 0 and base_h > 0
            and target_w > 0 and target_h > 0
        ):
            s = min(float(target_w) / float(base_w), float(target_h) / float(base_h))
            pad_x = (float(target_w) - float(base_w) * s) * 0.5
            pad_y = (float(target_h) - float(base_h) * s) * 0.5
            fx *= s
            fy *= s
            cx = cx * s + pad_x
            cy = cy * s + pad_y

        return fx, fy, cx, cy

    def _resolve_extrinsics(self, camera_id: str) -> Optional[Sequence[float]]:
        calib_bundle = getattr(self, "calibration_bundle", None) or {}
        cameras_node = calib_bundle.get("cameras") if isinstance(calib_bundle, dict) else {}
        e_table = cameras_node.get("E") if isinstance(cameras_node, dict) else {}
        extr = e_table.get(camera_id) if isinstance(e_table, dict) else None
        if not isinstance(extr, (list, tuple)) or len(extr) != 16:
            return None
        return extr

    @staticmethod
    def _compute_normals(
        depth: np.ndarray,
        valid_mask: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        height, width = depth.shape
        grid_u, grid_v = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing="xy",
        )
        x_cam = (grid_u - float(cx)) * depth / float(fx)
        y_cam = (grid_v - float(cy)) * depth / float(fy)
        z_cam = depth
        points = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32, copy=False)
        invalid = ~valid_mask
        if np.any(invalid):
            points[invalid] = np.nan

        dPdx = np.zeros_like(points)
        dPdy = np.zeros_like(points)
        if width > 1:
            dPdx[:, 1:-1] = points[:, 2:] - points[:, :-2]
            dPdx[:, 0] = points[:, 1] - points[:, 0]
            dPdx[:, -1] = points[:, -1] - points[:, -2]
        if height > 1:
            dPdy[1:-1] = points[2:] - points[:-2]
            dPdy[0] = points[1] - points[0]
            dPdy[-1] = points[-1] - points[-2]

        normals = np.cross(dPdx, dPdy)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normals = np.divide(normals, norm, out=np.zeros_like(normals), where=(norm > 1e-6))

        good = np.isfinite(normals).all(axis=-1) & valid_mask
        normals[~good] = 0.0

        flip = normals[..., 2] > 0
        normals[flip] *= -1.0
        return normals

    def attach_normals_to_payload(
        self,
        camera_id: str,
        payload: Dict[str, Any],
        *,
        space: str = "camera",
        dtype: str = "float16",
    ) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("normals_b64"):
            return

        depth_b64 = payload.get("depth_b64") or payload.get("depth_z_b64")
        shape = payload.get("shape")
        if not isinstance(depth_b64, str) or not depth_b64:
            payload["normals_error"] = "missing_depth"
            return
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            payload["normals_error"] = "bad_shape"
            return

        try:
            height = int(shape[0])
            width = int(shape[1])
        except Exception:
            payload["normals_error"] = "bad_shape"
            return
        if height <= 0 or width <= 0:
            payload["normals_error"] = "bad_shape"
            return

        ts_raw = payload.get("ts") or payload.get("ts_us") or 0
        try:
            ts_us = int(ts_raw)
        except Exception:
            ts_us = 0

        space_norm = str(space or "camera").strip().lower() or "camera"
        dtype_norm = str(dtype or "float16").strip().lower() or "float16"
        if space_norm not in ("camera", "world"):
            payload["normals_error"] = "unsupported_space"
            return
        if dtype_norm not in ("float16", "float32"):
            payload["normals_error"] = "unsupported_dtype"
            return

        cache_key = (str(camera_id), int(ts_us or 0), space_norm, dtype_norm)
        with self._cache_lock:
            cached = self._normals_cache.get(cache_key)
            if cached:
                payload.pop("normals_error", None)
                payload.update(cached)
                return

        try:
            raw = base64.b64decode(depth_b64)
            depth = np.frombuffer(raw, dtype=np.float32)
            needed = height * width
            if depth.size < needed:
                payload["normals_error"] = "depth_too_small"
                return
            depth = depth[:needed].reshape((height, width))
        except Exception:
            payload["normals_error"] = "depth_decode_failed"
            return

        mask = None
        mask_b64 = payload.get("mask_b64")
        if isinstance(mask_b64, str) and mask_b64:
            try:
                raw_mask = base64.b64decode(mask_b64)
                mask_arr = np.frombuffer(raw_mask, dtype=np.uint8)
                if mask_arr.size >= height * width:
                    mask = mask_arr[: height * width].reshape((height, width)) > 0
            except Exception:
                mask = None

        valid = np.isfinite(depth)
        valid &= depth > 0.1
        valid &= depth < 50.0
        if mask is not None:
            valid &= mask
        if not np.any(valid):
            payload["normals_error"] = "no_valid_depth"
            return

        try:
            fx, fy, cx, cy = self._resolve_intrinsics_for_depth(camera_id, depth.shape)
        except Exception as exc:
            payload["normals_error"] = str(exc) or "intrinsics_failed"
            return

        normals = self._compute_normals(depth.astype(np.float32, copy=False), valid, fx, fy, cx, cy)

        if space_norm == "world":
            extr = self._resolve_extrinsics(camera_id)
            if extr is None:
                payload["normals_error"] = "missing_extrinsics"
                return
            try:
                r_wc, _ = parse_extrinsics(extr)
                normals = (r_wc @ normals.reshape(-1, 3).T).T.reshape((height, width, 3))
                norm = np.linalg.norm(normals, axis=-1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=(norm > 1e-6))
            except Exception:
                payload["normals_error"] = "extrinsics_failed"
                return

        normals = np.asarray(normals, dtype=np.float32, copy=False)
        if dtype_norm == "float16":
            normals_out = normals.astype(np.float16)
        else:
            normals_out = normals.astype(np.float32)

        try:
            normals_b64 = base64.b64encode(normals_out.tobytes()).decode("ascii")
        except Exception:
            payload["normals_error"] = "normals_encode_failed"
            return

        normals_payload = {
            "normals_b64": normals_b64,
            "normals_shape": [int(height), int(width), 3],
            "normals_dtype": dtype_norm,
            "normals_space": space_norm,
        }
        payload.pop("normals_error", None)
        payload.update(normals_payload)
        with self._cache_lock:
            self._normals_cache[cache_key] = dict(normals_payload)
            self._normals_cache.move_to_end(cache_key, last=True)
            while len(self._normals_cache) > self._max_normals_cache_entries:
                self._normals_cache.popitem(last=False)

    @staticmethod
    def _sanitize_camera_id(camera_id: str) -> str:
        safe = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in camera_id.strip())
        return safe or "camera"

    @staticmethod
    def _format_param(value: float) -> str:
        text = f"{float(value):.6f}".rstrip('0').rstrip('.')
        if not text:
            text = "0"
        if text.startswith('-'):
            text = 'neg' + text[1:]
        return text.replace('.', 'p')

    def _floorplan_path(self, camera_id: str, grid_res_m: float, max_extent_m: float) -> Path:
        safe_cam = self._sanitize_camera_id(camera_id)
        filename = f"grid{self._format_param(grid_res_m)}__ext{self._format_param(max_extent_m)}.json"
        return self._floorplan_store_dir / safe_cam / filename

    def _load_floorplan_from_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        expected_flip: Optional[Tuple[bool, bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        if not path.exists():
            return None
        try:
            with path.open('r', encoding='utf-8') as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                frame = payload.get("frame")
                if frame != _FLOORPLAN_FRAME:
                    return None
                if expected_flip is not None:
                    if not _flip_payload_matches(payload.get("image_flip"), expected_flip):
                        return None
                payload.setdefault('camera_id', camera_id)
                return payload
        except Exception as exc:
            self._logger.debug(f"Failed to load cached floorplan for {camera_id}: {exc}")
        return None

    def _persist_floorplan_to_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        payload: Mapping[str, Any],
    ) -> None:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            to_store = dict(payload)
            to_store.setdefault("frame", _FLOORPLAN_FRAME)
            to_store.pop('served_from_cache', None)
            tmp_path = path.with_suffix(path.suffix + '.tmp')
            with tmp_path.open('w', encoding='utf-8') as fh:
                json.dump(to_store, fh, separators=(',', ':'))
            tmp_path.replace(path)
        except Exception as exc:
            self._logger.debug(f"Failed to persist floorplan for {camera_id}: {exc}")

    def update_depth_cache(self, result: DepthResult) -> None:
        try:
            depth_bytes = result.depth.astype(np.float32, copy=False).tobytes()
            conf_bytes = result.conf.astype(np.float32, copy=False).tobytes()
            mask_bytes = result.mask.astype(np.uint8, copy=False).tobytes()
        except Exception as exc:
            self._logger.debug(f"Depth cache serialization failed for {result.camera_id}: {exc}")
            return
        payload = {
            'ts': result.ts_us,
            'depth_b64': base64.b64encode(depth_bytes).decode('ascii'),
            'conf_b64': base64.b64encode(conf_bytes).decode('ascii'),
            'mask_b64': base64.b64encode(mask_bytes).decode('ascii'),
            'shape': [int(result.depth.shape[0]), int(result.depth.shape[1])],
        }
        with self._cache_lock:
            self._depth_payload_cache[result.camera_id] = payload
            self._depth_payload_cache.move_to_end(result.camera_id, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)

    def precompute_for_cameras(
        self,
        camera_ids: Iterable[str],
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
    ) -> None:
        for cam_id in camera_ids:
            if not cam_id:
                continue
            try:
                self.load_latest_depth(cam_id)
            except Exception as exc:
                self._logger.debug(f"Depth cache warmup for {cam_id} failed: {exc}")
                continue
            try:
                self.generate_topdown_floorplan(
                    cam_id,
                    max_age_sec=max_age_sec,
                    grid_res_m=grid_res_m,
                    max_extent_m=max_extent_m,
                )
            except Exception as exc:
                self._logger.debug(f"Floorplan warmup for {cam_id} failed: {exc}")

    def generate_topdown_floorplan(
        self,
        camera_id: str,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.15,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
    ) -> Dict[str, Any]:
        """Generate a per-camera top-down (XZ) blueprint view from the latest depth snapshot."""
        if not camera_id:
            return {'error': 'camera_required', 'ts': int(time.time() * 1_000_000)}

        if not cache_only:
            try:
                self.flush(timeout=1.5)
            except Exception:
                pass

        expected_flip = _expected_floorplan_flip(getattr(self, "calibration_bundle", None) or {}, camera_id)

        cache_key = (camera_id, float(grid_res_m), float(max_extent_m))
        now_us = int(time.time() * 1_000_000)
        with self._cache_lock:
            cached = self._floorplan_cache.get(cache_key)
            if cached:
                if expected_flip is not None and not _flip_payload_matches(cached.get("image_flip"), expected_flip):
                    cached = None
            if cached:
                if cache_only:
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload
                age_us = now_us - cached.get('snapshot_ts', cached.get('ts', 0))
                if age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload

        disk_payload = self._load_floorplan_from_disk(camera_id, grid_res_m, max_extent_m, expected_flip=expected_flip)
        if disk_payload:
            snapshot_ts = disk_payload.get('snapshot_ts', disk_payload.get('ts'))
            if isinstance(snapshot_ts, (int, float)):
                age_us = now_us - int(snapshot_ts)
            else:
                age_us = None
            if cache_only or age_us is None or age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                with self._cache_lock:
                    self._floorplan_cache[cache_key] = dict(disk_payload)
                    self._floorplan_cache.move_to_end(cache_key, last=True)
                    while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                        self._floorplan_cache.popitem(last=False)
                payload = dict(disk_payload)
                payload['served_from_cache'] = True
                return payload

        max_age_us = int(max(0.0, max_age_sec) * 1_000_000)
        ts_cutoff = now_us - max_age_us if max_age_us > 0 else None

        path_entry = self.latest_entry(camera_id, now_us)
        if not path_entry:
            return {'error': 'no_depth', 'camera_id': camera_id, 'ts': now_us}

        if ts_cutoff is not None:
            try:
                snapshot_ts = int(path_entry.stem)
            except ValueError:
                snapshot_ts = None
            if snapshot_ts is None or snapshot_ts < ts_cutoff:
                return {'error': 'stale_depth', 'camera_id': camera_id, 'ts': now_us}

        datasets = self.load_datasets(path_entry)
        if not datasets:
            return {'error': 'load_failed', 'camera_id': camera_id, 'ts': now_us}

        depth = datasets.get('depth')
        conf = datasets.get('conf')
        mask = datasets.get('mask')
        if depth is None or conf is None or mask is None:
            return {'error': 'invalid_snapshot', 'camera_id': camera_id, 'ts': now_us}

        depth = np.asarray(depth, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.uint8) > 0
        if depth.ndim != 2 or conf.shape != depth.shape or mask.shape != depth.shape:
            return {'error': 'shape_mismatch', 'camera_id': camera_id, 'ts': now_us}

        intr_arr = np.asarray([], dtype=np.float32)
        calib_bundle = getattr(self, 'calibration_bundle', None) or {}
        cameras_node = calib_bundle.get('cameras') if isinstance(calib_bundle, dict) else {}
        k_table = cameras_node.get('K') if isinstance(cameras_node, dict) else {}
        e_table = cameras_node.get('E') if isinstance(cameras_node, dict) else {}

        intr = None
        extr = None
        if isinstance(k_table, dict):
            intr = k_table.get(camera_id)
        if isinstance(e_table, dict):
            extr = e_table.get(camera_id)

        if intr is None or extr is None:
            return {'error': 'missing_calibration', 'camera_id': camera_id, 'ts': now_us}

        flip_pair = _infer_image_flips_from_extrinsics(extr)
        if flip_pair is None:
            flip_u = False
            flip_v = False
        else:
            flip_u, flip_v = flip_pair

        intr_arr = np.asarray(intr, dtype=np.float32).reshape(-1)
        if intr_arr.size == 4:
            fx, fy, cx, cy = [float(v) for v in intr_arr]
        elif intr_arr.size == 9:
            k_mat = intr_arr.reshape(3, 3)
            fx = float(k_mat[0, 0])
            fy = float(k_mat[1, 1])
            cx = float(k_mat[0, 2])
            cy = float(k_mat[1, 2])
        else:
            return {'error': 'bad_intrinsics', 'camera_id': camera_id, 'ts': now_us}

        if not all(np.isfinite([fx, fy, cx, cy])) or fx == 0.0 or fy == 0.0:
            return {'error': 'invalid_intrinsics', 'camera_id': camera_id, 'ts': now_us}

        # Align intrinsics to the depth raster resolution, preserving aspect (letterbox).
        target_h, target_w = depth.shape
        base_w = None
        base_h = None
        meta_node = calib_bundle.get('meta') if isinstance(calib_bundle, dict) else None
        specs_node = meta_node.get('camera_specs') if isinstance(meta_node, dict) else None
        spec = specs_node.get(camera_id) if isinstance(specs_node, dict) else None
        if isinstance(spec, dict):
            res = spec.get('resolution')
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                try:
                    base_w = int(res[0])
                    base_h = int(res[1])
                except Exception:
                    base_w = None
                    base_h = None
            if base_w is None or base_h is None:
                try:
                    base_w = int(spec.get('width', 0) or 0) or base_w
                    base_h = int(spec.get('height', 0) or 0) or base_h
                except Exception:
                    base_w = base_w
                    base_h = base_h
        if base_w is None or base_h is None:
            try:
                base_w = int(round(float(cx) * 2.0))
                base_h = int(round(float(cy) * 2.0))
            except Exception:
                base_w = None
                base_h = None
        if (
            base_w and base_h and target_w and target_h
            and base_w > 0 and base_h > 0
            and target_w > 0 and target_h > 0
        ):
            s = min(float(target_w) / float(base_w), float(target_h) / float(base_h))
            pad_x = (float(target_w) - float(base_w) * s) * 0.5
            pad_y = (float(target_h) - float(base_h) * s) * 0.5
            fx *= s
            fy *= s
            cx = cx * s + pad_x
            cy = cy * s + pad_y

        # PERMISSIVE validity: only reject truly invalid depth values
        # Do NOT hard-filter by mask or confidence - use them as soft weights instead
        valid = np.isfinite(depth)
        valid &= depth > 0.1
        valid &= depth < 50.0
        # Note: mask and conf are used as weights below, not hard filters

        if not np.any(valid):
            grid = np.zeros((1, 1), dtype=np.float32)
            payload = {
                'camera_id': camera_id,
                'ts': now_us,
                'snapshot_ts': int(path_entry.stem) if path_entry.stem.isdigit() else None,
                'frame': _FLOORPLAN_FRAME,
                'bounds': {'min_x': -grid_res_m * 0.5, 'max_x': grid_res_m * 0.5, 'min_z': 0.0, 'max_z': max(grid_res_m, 1.0)},
                'scale_m_per_px': float(grid_res_m),
                'point_count': 0,
                'density': {
                    'grid_b64': base64.b64encode(grid.tobytes()).decode('ascii'),
                    'grid_shape': [1, 1],
                    'value_min': 0.0,
                    'value_max': 0.0,
                },
                'height': {
                    'grid_b64': base64.b64encode(grid.tobytes()).decode('ascii'),
                    'grid_shape': [1, 1],
                    'value_min': 0.0,
                    'value_max': 0.0,
                },
                'distance': {
                    'grid_b64': base64.b64encode(grid.tobytes()).decode('ascii'),
                    'grid_shape': [1, 1],
                    'value_min': 0.0,
                    'value_max': 0.0,
                },
                'served_from_cache': False,
                'grid_res_m': float(grid_res_m),
                'max_extent_m': float(max_extent_m),
            }
            payload['image_flip'] = {'u': bool(flip_u), 'v': bool(flip_v)}
            self._persist_floorplan_to_disk(camera_id, grid_res_m, max_extent_m, payload)
            with self._cache_lock:
                self._floorplan_cache[cache_key] = dict(payload)
                self._floorplan_cache.move_to_end(cache_key, last=True)
                while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                    self._floorplan_cache.popitem(last=False)
            return payload

        # Extract confidence values for valid points - use as weights, not filter
        # Combine mask (as 0/1) and conf into a single weight
        # Points inside mask with high conf get weight ~1.0
        # Points outside mask or low conf get lower weights but still contribute
        # Soften the mask: give masked-out points a small weight (0.15) instead of 0
        soft_mask = np.where(mask, 1.0, 0.15).astype(np.float32)
        # Clamp confidence to [0.05, 1.0] to avoid zero weights
        conf_clamped = np.clip(conf, 0.05, 1.0)
        # Combined weight = soft_mask * confidence
        combined_weight = soft_mask * conf_clamped
        # Extract weights for valid points
        pts_weight = combined_weight[valid].astype(np.float32)

        h_img, w_img = depth.shape
        grid_u, grid_v = np.meshgrid(
            np.arange(w_img, dtype=np.float32),
            np.arange(h_img, dtype=np.float32),
            indexing='xy'
        )
        if flip_u:
            grid_u = (float(w_img - 1)) - grid_u
        if flip_v:
            grid_v = (float(h_img - 1)) - grid_v

        x_cam = (grid_u - cx) * depth / fx
        y_cam = (grid_v - cy) * depth / fy
        z_cam = depth

        pts_cam = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=1)

        e_arr = np.asarray(extr, dtype=np.float32)
        if e_arr.size == 16:
            e_mat = e_arr.reshape(4, 4, order='F')
        elif e_arr.shape == (3, 4):
            e_mat = np.eye(4, dtype=np.float32)
            e_mat[:3, :4] = e_arr
        elif e_arr.shape == (4, 4):
            e_mat = e_arr
        else:
            return {'error': 'bad_extrinsics', 'camera_id': camera_id, 'ts': now_us}

        try:
            twc = np.linalg.inv(e_mat)
        except np.linalg.LinAlgError:
            return {'error': 'extrinsics_singular', 'camera_id': camera_id, 'ts': now_us}

        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world_h = pts_cam_h @ twc.T
        pts_world = pts_world_h[:, :3]

        pts_depth = pts_cam[:, 2]
        pts_y = pts_world[:, 1]
        # Align floorplan coordinates with BEV camera-local ground-plane frame.
        R_wc = twc[:3, :3]
        C_world = twc[:3, 3]
        dir_world = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        yaw = math.atan2(float(dir_world[0]), float(dir_world[2]))
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        dx = pts_world[:, 0] - float(C_world[0])
        dz = pts_world[:, 2] - float(C_world[2])
        x_cam_pts = dx * cos_yaw - dz * sin_yaw
        z_cam_pts = dx * sin_yaw + dz * cos_yaw

        if x_cam_pts.size == 0 or z_cam_pts.size == 0:
            return {'error': 'no_points', 'camera_id': camera_id, 'ts': now_us, 'point_count': 0}

        pad_x = max(0.5, grid_res_m * 2.0)
        pad_z = max(0.5, grid_res_m * 2.0)

        max_x_abs = float(np.max(np.abs(x_cam_pts))) if x_cam_pts.size else 0.0
        if not np.isfinite(max_x_abs):
            max_x_abs = 0.0
        forward_max = float(np.max(z_cam_pts)) if z_cam_pts.size else 0.0
        if not np.isfinite(forward_max):
            forward_max = 0.0

        half_width = max_x_abs + pad_x
        forward_extent = max(0.0, forward_max) + pad_z
        if max_extent_m > 0:
            half_width = min(half_width, max_extent_m * 0.5)
            forward_extent = min(forward_extent, max_extent_m)

        half_width = max(half_width, grid_res_m * 0.5)
        forward_extent = max(forward_extent, grid_res_m)

        min_x = -half_width
        max_x = half_width
        min_z = 0.0
        max_z = forward_extent

        width_m = max_x - min_x
        height_m = max_z - min_z

        w_px = max(1, int(np.ceil(width_m / grid_res_m)))
        h_px = max(1, int(np.ceil(height_m / grid_res_m)))

        x_norm = np.clip((x_cam_pts - min_x) / width_m, 0.0, 0.999999)
        z_norm = np.clip((z_cam_pts - min_z) / height_m, 0.0, 0.999999)
        x_idx = np.clip(np.floor(x_norm * w_px).astype(np.int32), 0, w_px - 1)
        z_idx = np.clip(np.floor((1.0 - z_norm) * h_px).astype(np.int32), 0, h_px - 1)

        density_grid = np.zeros((h_px, w_px), dtype=np.float32)
        distance_sum = np.zeros((h_px, w_px), dtype=np.float32)
        distance_count = np.zeros((h_px, w_px), dtype=np.uint32)
        height_grid = np.full((h_px, w_px), -np.inf, dtype=np.float32)

        # Confidence-weighted height aggregation
        weighted_height_sum = np.zeros((h_px, w_px), dtype=np.float64)
        weight_sum = np.zeros((h_px, w_px), dtype=np.float64)

        indices = (z_idx, x_idx)
        # Density: count of points (unweighted for backward compat)
        np.add.at(density_grid, indices, 1.0)
        # Distance: weighted by confidence
        np.add.at(distance_sum, indices, (pts_depth * pts_weight).astype(np.float32, copy=False))
        np.add.at(distance_count, indices, 1)
        # Height: accumulate weighted sum and weights for weighted mean
        np.add.at(weighted_height_sum, indices, (pts_y * pts_weight).astype(np.float64))
        np.add.at(weight_sum, indices, pts_weight.astype(np.float64))
        # Also track max height (still useful for some visualizations)
        np.maximum.at(height_grid, indices, np.asarray(pts_y, dtype=np.float32))

        # Compute confidence-weighted mean height
        # Use weighted mean where we have weights, otherwise fall back to max
        has_weight = weight_sum > 1e-9
        # Avoid invalid division warnings: np.where evaluates both branches eagerly.
        weighted_mean_height = height_grid.astype(np.float32, copy=True)
        np.divide(
            weighted_height_sum,
            weight_sum,
            out=weighted_mean_height,
            where=has_weight,
        )

        # For cells with no points at all, mark as NaN
        empty_cells = height_grid == -np.inf
        if np.any(empty_cells):
            weighted_mean_height[empty_cells] = np.nan
            height_grid[empty_cells] = np.nan

        # Use weighted mean as the primary height grid (preserves detail better than max)
        height_grid = weighted_mean_height

        density_max = float(np.max(density_grid)) if density_grid.size else 0.0
        if density_max > 0.0:
            density_grid /= density_max

        if np.isnan(height_grid).any():
            if not np.isnan(height_grid).all():
                min_height = float(np.nanmin(height_grid))
            else:
                min_height = 0.0
            height_grid = np.nan_to_num(height_grid, nan=min_height)
        height_min = float(np.min(height_grid)) if height_grid.size else 0.0
        height_max = float(np.max(height_grid)) if height_grid.size else 0.0
        if height_grid.size:
            height_grid = height_grid - height_min
            height_min = 0.0
            height_max = float(np.max(height_grid)) if height_grid.size else 0.0

        # Compute height gradient magnitude for edge detection
        # First, fill empty/NaN cells with floor level so boundaries don't create false edges
        height_for_gradient = height_grid.copy()
        floor_level = float(np.nanmin(height_grid)) if np.any(np.isfinite(height_grid)) else 0.0
        height_for_gradient = np.nan_to_num(
            height_for_gradient,
            nan=floor_level,
            posinf=floor_level,
            neginf=floor_level,
        )

        # Also mask cells with zero density (no points) to floor level
        if density_grid is not None:
            empty_mask = density_grid < 1e-6
            height_for_gradient[empty_mask] = floor_level

        # Sobel filters capture directional derivatives
        gx = ndi.sobel(height_for_gradient, axis=1, mode='nearest')
        gz = ndi.sobel(height_for_gradient, axis=0, mode='nearest')
        gradient_mag = np.sqrt(gx ** 2 + gz ** 2)

        # Use 95th percentile normalization to prevent outliers from dominating
        gradient_flat = gradient_mag[np.isfinite(gradient_mag)]
        if gradient_flat.size > 0:
            gradient_p95 = float(np.percentile(gradient_flat, 95))
            if gradient_p95 > 1e-6:
                gradient_grid = np.clip(gradient_mag / gradient_p95, 0.0, 1.0).astype(np.float32)
            else:
                gradient_grid = np.zeros_like(height_grid, dtype=np.float32)
        else:
            gradient_grid = np.zeros_like(height_grid, dtype=np.float32)

        # Clean up any remaining NaN
        gradient_grid = np.nan_to_num(gradient_grid, nan=0.0, posinf=0.0, neginf=0.0)

        distance_grid = np.zeros((h_px, w_px), dtype=np.float32)
        nonzero_mask = distance_count > 0
        if np.any(nonzero_mask):
            distance_grid[nonzero_mask] = distance_sum[nonzero_mask] / distance_count[nonzero_mask]
            min_distance = float(np.min(distance_grid[nonzero_mask]))
            max_distance = float(np.max(distance_grid[nonzero_mask]))
        else:
            min_distance = 0.0
            max_distance = 0.0

        bounds = {
            'min_x': float(min_x),
            'max_x': float(max_x),
            'min_z': float(min_z),
            'max_z': float(max_z),
        }

        payload: Dict[str, Any] = {
            'camera_id': camera_id,
            'ts': now_us,
            'snapshot_ts': int(path_entry.stem) if path_entry.stem.isdigit() else None,
            'frame': _FLOORPLAN_FRAME,
            'bounds': bounds,
            'scale_m_per_px': float(width_m / w_px if w_px else grid_res_m),
            'point_count': int(pts_cam.shape[0]),
            'density': {
                'grid_b64': base64.b64encode(density_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
            'height': {
                'grid_b64': base64.b64encode(height_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(height_min),
                'value_max': float(height_max),
            },
            'distance': {
                'grid_b64': base64.b64encode(distance_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(min_distance),
                'value_max': float(max_distance),
            },
            'gradient': {
                'grid_b64': base64.b64encode(gradient_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
        }
        payload['served_from_cache'] = False
        payload['grid_res_m'] = float(grid_res_m)
        payload['max_extent_m'] = float(max_extent_m)
        payload['image_flip'] = {'u': bool(flip_u), 'v': bool(flip_v)}

        self._persist_floorplan_to_disk(camera_id, grid_res_m, max_extent_m, payload)

        with self._cache_lock:
            self._floorplan_cache[cache_key] = dict(payload)
            self._floorplan_cache.move_to_end(cache_key, last=True)
            while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                self._floorplan_cache.popitem(last=False)

        return payload


class MapAnythingDepthSource:
    """Client for invoking MapAnything inference service and persisting results."""

    def __init__(self, config: Optional[ServiceConfig] = None) -> None:
        self.config = config or load_service_config()
        self.session = requests.Session()
        self.logger = RateLimitedLogger(logging.getLogger(__name__), rate_limit_seconds=2.0)
        self.min_conf = float(self.config.performance.min_conf)
        self.storage = DepthStorageManager(
            Path(self.config.storage.depth_base),
            max_snapshots_per_camera=self.config.storage.max_snapshots_per_camera,
            retention_minutes=self.config.storage.snapshot_retention_minutes,
            max_total_bytes=getattr(self.config.storage, 'max_total_bytes', None),
            enable_async=getattr(self.config.storage, 'async_enabled', True),
            max_queue_size=max(1, int(getattr(self.config.storage, 'queue_size', 32))),
            worker_count=int(getattr(self.config.storage, 'async_workers', 0)),
            max_worker_count=int(getattr(self.config.storage, 'async_max_workers', 0)),
            enforce_async=bool(getattr(self.config.storage, 'enforce_async', True)),
            enforce_interval_s=float(getattr(self.config.storage, 'enforce_interval_s', 1.0)),
            size_hysteresis_ratio=float(getattr(self.config.storage, 'quota_hysteresis_ratio', 0.9)),
            zarr_clevel=int(getattr(self.config.storage, 'zarr_clevel', 5)),
            zarr_chunk_px=int(getattr(self.config.storage, 'zarr_chunk_px', 128)),
            min_conf=self.min_conf,
        )
        self.mono_interval = 1.0 / max(self.config.performance.mono_freq_hz, 1e-6)
        self.last_request_per_camera: Dict[str, float] = {}
        self.timeout = 5.0
        self._cache_lock = threading.Lock()
        self._depth_payload_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._floorplan_cache: "OrderedDict[Tuple[str, float, float], Dict[str, Any]]" = OrderedDict()
        self._floorplan_store_dir = Path(self.config.storage.depth_base) / "floorplans"
        self._floorplan_store_dir.mkdir(parents=True, exist_ok=True)
        self._max_depth_cache_entries = 16
        self._max_floorplan_cache_entries = 24
        self._batch_lock = threading.Lock()
        self._batch_condition = threading.Condition(self._batch_lock)
        self._batch_queue: Deque[_BatchItem] = deque()
        self._batch_worker: Optional[threading.Thread] = None
        self._batch_shutdown = False
        batch_size = max(1, int(getattr(self.config.performance, 'multi_batch_size', 1)))
        self._batch_size = batch_size
        # Flush quickly enough to avoid latency while still gathering a few cameras.
        candidate_flush = self.mono_interval * 0.25
        self._batch_flush_s = max(0.005, min(0.02, candidate_flush))
        self._multi_batches_attempted = 0
        self._multi_batches_succeeded = 0
        self._multi_batches_fallback = 0
        self._multi_scene_counter = 0

    def close(self) -> None:
        try:
            self._stop_batch_worker()
        except Exception:
            pass
        try:
            self.storage.flush(timeout=2.0)
            self.storage.shutdown(wait=False)
        except Exception:
            pass

    def should_infer(self, camera_id: str, timestamp_s: float) -> bool:
        last = self.last_request_per_camera.get(camera_id)
        if last is None:
            return True
        return (timestamp_s - last) >= self.mono_interval

    def maybe_infer_mono(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        calib_bundle: Optional[Mapping[str, object]],
        timestamp_s: Optional[float] = None,
    ) -> Optional[DepthResult]:
        ts = timestamp_s or time.time()
        if not self.should_infer(camera_id, ts):
            return None
        try:
            view_result, view_payload = self._prepare_view(camera_id, frame_bgr, calib_bundle)
            if self._batch_size <= 1:
                result = self._run_single_request(camera_id, ts, view_result, view_payload)
            else:
                result = self._submit_batch_request(camera_id, ts, view_result, view_payload)
            self.last_request_per_camera[camera_id] = ts
            return result
        except Exception as exc:
            self.logger.error(f"Mono depth inference failed for {camera_id}: {exc}")
            return None

    def _prepare_view(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        calib_bundle: Optional[Mapping[str, object]],
    ) -> Tuple[ViewBuildResult, Dict[str, object]]:
        view_result = build_mono_view(frame_bgr, camera_id, calib_bundle)
        payload = self._serialize_view_payload(camera_id, view_result)
        return view_result, payload

    def _serialize_view_payload(self, camera_id: str, view_result: ViewBuildResult) -> Dict[str, object]:
        view_payload = dict(view_result.payload)
        view_payload['cam_id'] = camera_id

        shape = view_payload.get('shape') or view_result.resized_shape
        view_payload['shape'] = [int(shape[0]), int(shape[1]), int(shape[2])]

        if 'img_b64' not in view_payload:
            frame_rgb = view_payload.get('img')
            if frame_rgb is None:
                raise ValueError("MapAnything view payload missing img_b64 data")
            img_bytes = np.ascontiguousarray(frame_rgb).tobytes()
            view_payload['img_b64'] = base64.b64encode(img_bytes).decode('ascii')
        view_payload.pop('img', None)

        if 'intrinsics' not in view_payload and view_result.intrinsics is not None:
            view_payload['intrinsics'] = view_result.intrinsics.tolist()

        return view_payload

    def _submit_batch_request(
        self,
        camera_id: str,
        timestamp_s: float,
        view_result: ViewBuildResult,
        view_payload: Dict[str, object],
    ) -> DepthResult:
        future: Future = Future()
        item = _BatchItem(
            camera_id=camera_id,
            timestamp_s=timestamp_s,
            view_result=view_result,
            view_payload=view_payload,
            future=future,
        )
        with self._batch_condition:
            self._start_batch_worker_locked()
            self._batch_queue.append(item)
            self._batch_condition.notify()
        return future.result()

    def _start_batch_worker_locked(self) -> None:
        if self._batch_worker is not None and self._batch_worker.is_alive():
            return
        self._batch_shutdown = False
        self._batch_worker = threading.Thread(target=self._batch_loop, name="MapAnythingBatcher", daemon=True)
        self._batch_worker.start()

    def _batch_loop(self) -> None:
        while True:
            with self._batch_condition:
                while not self._batch_queue and not self._batch_shutdown:
                    self._batch_condition.wait()
                if self._batch_shutdown and not self._batch_queue:
                    return
                if not self._batch_queue:
                    continue
                first = self._batch_queue.popleft()
                batch: List[_BatchItem] = [first]
                if self._batch_size > 1:
                    deadline = time.perf_counter() + self._batch_flush_s
                    while len(batch) < self._batch_size:
                        if self._batch_queue:
                            batch.append(self._batch_queue.popleft())
                            continue
                        if self._batch_shutdown:
                            break
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._batch_condition.wait(timeout=remaining)
                    while self._batch_queue and len(batch) < self._batch_size:
                        batch.append(self._batch_queue.popleft())
            try:
                self._process_batch(batch)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(f"Batch processing failed: {exc}")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _process_batch(self, batch: List[_BatchItem]) -> None:
        if not batch:
            return
        if len(batch) == 1:
            self._execute_single(batch[0])
            return
        self._multi_batches_attempted += 1
        try:
            parsed = self._invoke_multi(batch)
        except Exception as exc:
            self._multi_batches_fallback += 1
            self.logger.warning(f"/infer_multi failed ({exc}); falling back to individual requests")
            for item in batch:
                self._execute_single(item, suppress_error_log=True)
            return
        self._multi_batches_succeeded += 1
        for item in batch:
            entry = parsed.get(item.camera_id)
            if entry is None:
                self.logger.warning(f"/infer_multi response missing camera {item.camera_id}; retrying singly")
                self._execute_single(item, suppress_error_log=True)
                continue
            depth, conf, mask = entry
            self._resolve_item(item, depth, conf, mask)

    def _invoke_multi(self, batch: List[_BatchItem]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        scene_id = self._next_scene_id()
        payload = {
            'scene_id': scene_id,
            'views': [dict(item.view_payload) for item in batch],
        }
        response_json = self._post_json('/infer_multi', payload)
        return self._parse_multi_response(response_json)

    def _execute_single(self, item: _BatchItem, *, suppress_error_log: bool = False) -> None:
        try:
            payload = {'view': dict(item.view_payload)}
            response_json = self._post_json('/infer_mono', payload)
            depth, conf, mask = self._parse_response(response_json)
            self._resolve_item(item, depth, conf, mask)
        except Exception as exc:
            if not suppress_error_log:
                self.logger.warning(f"/infer_mono fallback failed for {item.camera_id}: {exc}")
            if not item.future.done():
                item.future.set_exception(exc)

    def _resolve_item(
        self,
        item: _BatchItem,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        try:
            result = self._finalize_depth_result(item.camera_id, item.timestamp_s, item.view_result, depth, conf, mask)
            if not item.future.done():
                item.future.set_result(result)
        except Exception as exc:
            if not item.future.done():
                item.future.set_exception(exc)

    def _finalize_depth_result(
        self,
        camera_id: str,
        timestamp_s: float,
        view_result: ViewBuildResult,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> DepthResult:
        depth_aligned, conf_aligned, mask_aligned = self._align_to_original_shape(depth, conf, mask, view_result)
        ts_us = int(timestamp_s * 1_000_000)
        storage_path = self.storage.store(camera_id, ts_us, depth_aligned, conf_aligned, mask_aligned)
        summary = self._compute_summary(depth_aligned, conf_aligned, mask_aligned)
        return DepthResult(
            camera_id=camera_id,
            ts_us=ts_us,
            depth=depth_aligned,
            conf=conf_aligned,
            mask=mask_aligned,
            intrinsics=view_result.native_intrinsics.copy() if view_result.native_intrinsics is not None else None,
            native_intrinsics=view_result.native_intrinsics.copy() if view_result.native_intrinsics is not None else None,
            summary=summary,
            storage_path=storage_path,
        )

    def _run_single_request(
        self,
        camera_id: str,
        timestamp_s: float,
        view_result: ViewBuildResult,
        view_payload: Dict[str, object],
    ) -> DepthResult:
        response_json = self._post_json('/infer_mono', {'view': dict(view_payload)})
        depth, conf, mask = self._parse_response(response_json)
        return self._finalize_depth_result(camera_id, timestamp_s, view_result, depth, conf, mask)

    def _parse_multi_response(
        self,
        response: Dict[str, object],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        depth_map = response.get('depth_b64')
        conf_map = response.get('conf_b64')
        mask_map = response.get('mask_b64') or {}
        shapes_map = response.get('shapes')
        if not isinstance(depth_map, dict) or not isinstance(conf_map, dict) or not isinstance(shapes_map, dict):
            raise ValueError('Multi response missing depth/conf/shape dictionaries')
        results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for cam_id, depth_b64 in depth_map.items():
            shape = shapes_map.get(cam_id)
            if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
                raise ValueError(f"Missing shape metadata for camera {cam_id}")
            height, width = int(shape[0]), int(shape[1])
            conf_b64 = conf_map.get(cam_id)
            if not isinstance(conf_b64, str):
                raise ValueError(f"Missing confidence tensor for camera {cam_id}")
            mask_b64 = mask_map.get(cam_id)
            try:
                depth = np.frombuffer(base64.b64decode(depth_b64), dtype=np.float32).reshape((height, width))
                conf = np.frombuffer(base64.b64decode(conf_b64), dtype=np.float32).reshape((height, width))
                if isinstance(mask_b64, str):
                    mask = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8).reshape((height, width)).astype(bool)
                else:
                    mask = np.ones((height, width), dtype=bool)
            except Exception as exc:
                raise ValueError(f"Failed to decode multi response for camera {cam_id}: {exc}") from exc
            results[cam_id] = (depth, conf, mask)
        return results

    def _next_scene_id(self) -> str:
        self._multi_scene_counter = (self._multi_scene_counter + 1) % 1_000_000
        return f"mono-batch-{self._multi_scene_counter}"

    def _stop_batch_worker(self) -> None:
        with self._batch_condition:
            self._batch_shutdown = True
            self._batch_condition.notify_all()
        if self._batch_worker is not None and self._batch_worker.is_alive():
            self._batch_worker.join(timeout=1.0)
        pending: List[_BatchItem] = []
        with self._batch_condition:
            while self._batch_queue:
                pending.append(self._batch_queue.popleft())
        for item in pending:
            if not item.future.done():
                item.future.set_exception(RuntimeError('Batch worker stopped before completion'))

    def _post_json(self, endpoint: str, payload: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.config.service.base_url}{endpoint}"
        headers = {"X-API-Key": self.config.service.api_key}
        backoff = 0.5
        for attempt in range(1, 4):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                self.logger.warning(f"MapAnything service returned {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                self.logger.warning(f"MapAnything request failed (attempt {attempt}): {exc}")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 4.0)
        raise RuntimeError("MapAnything request failed after retries")

    def _parse_response(self, response: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = response.get("shape")
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            raise ValueError(f"Depth response missing shape metadata (got {shape!r})")
        height, width = int(shape[0]), int(shape[1])

        depth_b64 = response.get("depth_b64") or response.get("depth_z_b64")
        conf_b64 = response.get("conf_b64")
        mask_b64 = response.get("mask_b64")
        missing = [name for name, value in (
            ("depth_b64", depth_b64),
            ("conf_b64", conf_b64),
            ("mask_b64", mask_b64),
        ) if not isinstance(value, str)]
        if missing:
            raise ValueError(f"Depth response missing encoded tensors: {missing}")

        try:
            depth = np.frombuffer(base64.b64decode(depth_b64), dtype=np.float32).reshape((height, width))
            conf = np.frombuffer(base64.b64decode(conf_b64), dtype=np.float32).reshape((height, width))
            mask = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8).reshape((height, width)).astype(bool)
        except Exception as exc:
            raise ValueError(f"Failed to decode depth response: {exc}") from exc
        return depth, conf, mask

    def _align_to_original_shape(
        self,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
        view_result: ViewBuildResult,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_h, target_w = view_result.original_shape[:2]
        current_h, current_w = depth.shape[:2]
        if (target_h, target_w) == (current_h, current_w):
            return depth, conf, mask
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        conf_resized = cv2.resize(conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_uint8.astype(bool)
        return depth_resized, conf_resized, mask_resized

    def _compute_summary(self, depth: np.ndarray, conf: np.ndarray, mask: np.ndarray) -> DepthSummary:
        valid = mask & np.isfinite(depth) & (conf >= self.min_conf) & (depth > 0.0)
        total = depth.size
        if total == 0:
            return DepthSummary(median=0.0, p10=0.0, p90=0.0, conf_mean=0.0, valid_ratio=0.0, sample_count=0)
        if not np.any(valid):
            return DepthSummary(median=0.0, p10=0.0, p90=0.0, conf_mean=float(conf.mean()), valid_ratio=0.0, sample_count=0)
        valid_depth = depth[valid]
        return DepthSummary(
            median=float(np.median(valid_depth)),
            p10=float(np.percentile(valid_depth, 10)),
            p90=float(np.percentile(valid_depth, 90)),
            conf_mean=float(conf[valid].mean()),
            valid_ratio=float(valid.sum() / total),
            sample_count=int(valid.sum()),
        )

    def load_latest_depth(self, camera_id: str, ts_max_us: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the newest cached payload up to ts_max_us (microseconds)."""
        cache_key = camera_id
        ts_cutoff = None
        if ts_max_us is not None:
            try:
                ts_cutoff = int(ts_max_us)
            except Exception:
                ts_cutoff = None
        with self._cache_lock:
            cached = self._depth_payload_cache.get(cache_key)
            if cached:
                try:
                    cached_ts = int(cached.get('ts', 0) or 0)
                except Exception:
                    cached_ts = 0
                if ts_cutoff is None or cached_ts <= ts_cutoff:
                    return dict(cached)

        path = self.storage.latest_entry(camera_id, ts_cutoff)
        if not path:
            return None
        datasets = self.storage.load_datasets(path)
        if not datasets:
            return None
        depth = datasets['depth'].astype(np.float32, copy=False)
        conf = datasets['conf'].astype(np.float32, copy=False)
        mask = datasets['mask'].astype(np.uint8, copy=False)
        height, width = depth.shape[:2]
        ts_us = int(path.stem)
        payload = {
            'ts': ts_us,
            'depth_b64': base64.b64encode(depth.tobytes()).decode('ascii'),
            'conf_b64': base64.b64encode(conf.tobytes()).decode('ascii'),
            'mask_b64': base64.b64encode(mask.tobytes()).decode('ascii'),
            'shape': [int(height), int(width)],
        }
        with self._cache_lock:
            self._depth_payload_cache[cache_key] = dict(payload)
            self._depth_payload_cache.move_to_end(cache_key, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)
        return payload

    @staticmethod
    def _sanitize_camera_id(camera_id: str) -> str:
        safe = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in camera_id.strip())
        return safe or "camera"

    @staticmethod
    def _format_param(value: float) -> str:
        text = f"{float(value):.6f}".rstrip('0').rstrip('.')
        if not text:
            text = "0"
        if text.startswith('-'):
            text = 'neg' + text[1:]
        return text.replace('.', 'p')

    def _floorplan_path(self, camera_id: str, grid_res_m: float, max_extent_m: float) -> Path:
        safe_cam = self._sanitize_camera_id(camera_id)
        filename = f"grid{self._format_param(grid_res_m)}__ext{self._format_param(max_extent_m)}.json"
        return self._floorplan_store_dir / safe_cam / filename

    def _load_floorplan_from_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        expected_flip: Optional[Tuple[bool, bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        if not path.exists():
            return None
        try:
            with path.open('r', encoding='utf-8') as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                frame = payload.get("frame")
                if frame != _FLOORPLAN_FRAME:
                    return None
                if expected_flip is not None:
                    if not _flip_payload_matches(payload.get("image_flip"), expected_flip):
                        return None
                payload.setdefault('camera_id', camera_id)
                return payload
        except Exception as exc:
            self.logger.debug(f"Failed to load cached floorplan for {camera_id}: {exc}")
        return None

    def _persist_floorplan_to_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        payload: Mapping[str, Any],
    ) -> None:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            to_store = dict(payload)
            to_store.setdefault("frame", _FLOORPLAN_FRAME)
            to_store.pop('served_from_cache', None)
            tmp_path = path.with_suffix(path.suffix + '.tmp')
            with tmp_path.open('w', encoding='utf-8') as fh:
                json.dump(to_store, fh, separators=(',', ':'))
            tmp_path.replace(path)
        except Exception as exc:
            self.logger.debug(f"Failed to persist floorplan for {camera_id}: {exc}")

    def update_depth_cache(self, result: DepthResult) -> None:
        try:
            depth_bytes = result.depth.astype(np.float32, copy=False).tobytes()
            conf_bytes = result.conf.astype(np.float32, copy=False).tobytes()
            mask_bytes = result.mask.astype(np.uint8, copy=False).tobytes()
        except Exception as exc:
            self.logger.debug(f"Depth cache serialization failed for {result.camera_id}: {exc}")
            return
        payload = {
            'ts': result.ts_us,
            'depth_b64': base64.b64encode(depth_bytes).decode('ascii'),
            'conf_b64': base64.b64encode(conf_bytes).decode('ascii'),
            'mask_b64': base64.b64encode(mask_bytes).decode('ascii'),
            'shape': [int(result.depth.shape[0]), int(result.depth.shape[1])],
        }
        with self._cache_lock:
            self._depth_payload_cache[result.camera_id] = payload
            self._depth_payload_cache.move_to_end(result.camera_id, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)

    def precompute_for_cameras(
        self,
        camera_ids: Iterable[str],
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
    ) -> None:
        for cam_id in camera_ids:
            if not cam_id:
                continue
            try:
                self.load_latest_depth(cam_id)
            except Exception as exc:
                self.logger.debug(f"Depth cache warmup for {cam_id} failed: {exc}")
                continue
            try:
                self.generate_topdown_floorplan(
                    cam_id,
                    max_age_sec=max_age_sec,
                    grid_res_m=grid_res_m,
                    max_extent_m=max_extent_m,
                )
            except Exception as exc:
                self.logger.debug(f"Floorplan warmup for {cam_id} failed: {exc}")

    def generate_topdown_floorplan(
        self,
        camera_id: str,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
    ) -> Dict[str, Any]:
        """Generate a per-camera top-down (XZ) blueprint view from the latest depth snapshot."""
        if not camera_id:
            return {'error': 'camera_required', 'ts': int(time.time() * 1_000_000)}

        # Ensure any recently queued depth snapshots are flushed to disk before attempting
        # to read for a regenerate request. This avoids a race where floorplan generation
        # runs immediately after inference but before the async writer persists the zarr.
        if not cache_only:
            try:
                self.storage.flush(timeout=1.5)
            except Exception:
                pass

        expected_flip = _expected_floorplan_flip(getattr(self.storage, "calibration_bundle", None) or {}, camera_id)

        cache_key = (camera_id, float(grid_res_m), float(max_extent_m))
        now_us = int(time.time() * 1_000_000)
        with self._cache_lock:
            cached = self._floorplan_cache.get(cache_key)
            if cached:
                if expected_flip is not None and not _flip_payload_matches(cached.get("image_flip"), expected_flip):
                    cached = None
            if cached:
                if cache_only:
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload
                age_us = now_us - cached.get('snapshot_ts', cached.get('ts', 0))
                if age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload
            # Fall through to load from disk or recompute when cache is empty
            # so callers receive a floorplan without additional interaction.

        disk_payload = self._load_floorplan_from_disk(camera_id, grid_res_m, max_extent_m, expected_flip=expected_flip)
        if disk_payload:
            snapshot_ts = disk_payload.get('snapshot_ts', disk_payload.get('ts'))
            if isinstance(snapshot_ts, (int, float)):
                age_us = now_us - int(snapshot_ts)
            else:
                age_us = None
            if cache_only or age_us is None or age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                with self._cache_lock:
                    self._floorplan_cache[cache_key] = dict(disk_payload)
                    self._floorplan_cache.move_to_end(cache_key, last=True)
                    while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                        self._floorplan_cache.popitem(last=False)
                payload = dict(disk_payload)
                payload['served_from_cache'] = True
                return payload

        max_age_us = int(max(0.0, max_age_sec) * 1_000_000)
        ts_cutoff = now_us - max_age_us if max_age_us > 0 else None

        path_entry = self.storage.latest_entry(camera_id, now_us)
        if not path_entry:
            return {'error': 'no_depth', 'camera_id': camera_id, 'ts': now_us}

        if ts_cutoff is not None:
            try:
                snapshot_ts = int(path_entry.stem)
            except ValueError:
                snapshot_ts = None
            if snapshot_ts is None or snapshot_ts < ts_cutoff:
                return {'error': 'stale_depth', 'camera_id': camera_id, 'ts': now_us}

        datasets = self.storage.load_datasets(path_entry)
        if not datasets:
            return {'error': 'load_failed', 'camera_id': camera_id, 'ts': now_us}

        depth = datasets.get('depth')
        conf = datasets.get('conf')
        mask = datasets.get('mask')
        if depth is None or conf is None or mask is None:
            return {'error': 'invalid_snapshot', 'camera_id': camera_id, 'ts': now_us}

        calib_bundle = getattr(self, 'calibration_bundle', None) or {}
        cameras_node = calib_bundle.get('cameras') if isinstance(calib_bundle, dict) else {}
        k_table = cameras_node.get('K') if isinstance(cameras_node, dict) else {}
        e_table = cameras_node.get('E') if isinstance(cameras_node, dict) else {}

        intr = None
        extr = None
        if isinstance(k_table, dict):
            intr = k_table.get(camera_id)
        if isinstance(e_table, dict):
            extr = e_table.get(camera_id)

        if intr is None or extr is None:
            return {'error': 'missing_calibration', 'camera_id': camera_id, 'ts': now_us}

        flip_pair = _infer_image_flips_from_extrinsics(extr)
        if flip_pair is None:
            flip_u = False
            flip_v = False
        else:
            flip_u, flip_v = flip_pair

        depth = np.asarray(depth, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.uint8) > 0
        if depth.ndim != 2 or conf.shape != depth.shape or mask.shape != depth.shape:
            return {'error': 'shape_mismatch', 'camera_id': camera_id, 'ts': now_us}

        intr_arr = np.asarray(intr, dtype=np.float32).reshape(-1)
        if intr_arr.size == 4:
            fx, fy, cx, cy = [float(v) for v in intr_arr]
        elif intr_arr.size == 9:
            k_mat = intr_arr.reshape(3, 3)
            fx = float(k_mat[0, 0])
            fy = float(k_mat[1, 1])
            cx = float(k_mat[0, 2])
            cy = float(k_mat[1, 2])
        else:
            return {'error': 'bad_intrinsics', 'camera_id': camera_id, 'ts': now_us}

        if not all(np.isfinite([fx, fy, cx, cy])) or fx == 0.0 or fy == 0.0:
            return {'error': 'invalid_intrinsics', 'camera_id': camera_id, 'ts': now_us}

        # PERMISSIVE validity: only reject truly invalid depth values
        # Do NOT hard-filter by mask or confidence - use them as soft weights instead
        valid = np.isfinite(depth)
        valid &= depth > 0.1
        valid &= depth < 50.0
        # Note: mask and conf are used as weights below, not hard filters

        if not np.any(valid):
            return {'error': 'no_points', 'camera_id': camera_id, 'ts': now_us, 'point_count': 0}

        # Extract confidence values for valid points - use as weights, not filter
        # Combine mask (as 0/1) and conf into a single weight
        # Points inside mask with high conf get weight ~1.0
        # Points outside mask or low conf get lower weights but still contribute
        soft_mask = np.where(mask, 1.0, 0.15).astype(np.float32)
        # Clamp confidence to [0.05, 1.0] to avoid zero weights
        conf_clamped = np.clip(conf, 0.05, 1.0)
        # Combined weight = soft_mask * confidence
        combined_weight = soft_mask * conf_clamped
        # Extract weights for valid points
        pts_weight = combined_weight[valid].astype(np.float32)

        h_img, w_img = depth.shape
        grid_u, grid_v = np.meshgrid(
            np.arange(w_img, dtype=np.float32),
            np.arange(h_img, dtype=np.float32),
            indexing='xy'
        )
        if flip_u:
            grid_u = (float(w_img - 1)) - grid_u
        if flip_v:
            grid_v = (float(h_img - 1)) - grid_v

        x_cam = (grid_u - cx) * depth / fx
        y_cam = (grid_v - cy) * depth / fy
        z_cam = depth

        pts_cam = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=1)

        e_arr = np.asarray(extr, dtype=np.float32)
        if e_arr.size == 16:
            e_mat = e_arr.reshape(4, 4, order='F')
        elif e_arr.shape == (3, 4):
            e_mat = np.eye(4, dtype=np.float32)
            e_mat[:3, :4] = e_arr
        elif e_arr.shape == (4, 4):
            e_mat = e_arr
        else:
            return {'error': 'bad_extrinsics', 'camera_id': camera_id, 'ts': now_us}

        try:
            twc = np.linalg.inv(e_mat)
        except np.linalg.LinAlgError:
            return {'error': 'extrinsics_singular', 'camera_id': camera_id, 'ts': now_us}

        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world_h = pts_cam_h @ twc.T
        pts_world = pts_world_h[:, :3]

        pts_depth = pts_cam[:, 2]
        pts_y = pts_world[:, 1]
        # Align floorplan coordinates with BEV camera-local ground-plane frame.
        R_wc = twc[:3, :3]
        C_world = twc[:3, 3]
        dir_world = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        yaw = math.atan2(float(dir_world[0]), float(dir_world[2]))
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        dx = pts_world[:, 0] - float(C_world[0])
        dz = pts_world[:, 2] - float(C_world[2])
        x_cam_pts = dx * cos_yaw - dz * sin_yaw
        z_cam_pts = dx * sin_yaw + dz * cos_yaw

        if x_cam_pts.size == 0 or z_cam_pts.size == 0:
            return {'error': 'no_points', 'camera_id': camera_id, 'ts': now_us, 'point_count': 0}

        pad_x = max(0.5, grid_res_m * 2.0)
        pad_z = max(0.5, grid_res_m * 2.0)

        max_x_abs = float(np.max(np.abs(x_cam_pts))) if x_cam_pts.size else 0.0
        if not np.isfinite(max_x_abs):
            max_x_abs = 0.0
        forward_max = float(np.max(z_cam_pts)) if z_cam_pts.size else 0.0
        if not np.isfinite(forward_max):
            forward_max = 0.0

        half_width = max_x_abs + pad_x
        forward_extent = max(0.0, forward_max) + pad_z
        if max_extent_m > 0:
            half_width = min(half_width, max_extent_m * 0.5)
            forward_extent = min(forward_extent, max_extent_m)

        half_width = max(half_width, grid_res_m * 0.5)
        forward_extent = max(forward_extent, grid_res_m)

        min_x = -half_width
        max_x = half_width
        min_z = 0.0
        max_z = forward_extent

        width_m = max_x - min_x
        height_m = max_z - min_z

        w_px = max(1, int(np.ceil(width_m / grid_res_m)))
        h_px = max(1, int(np.ceil(height_m / grid_res_m)))

        x_norm = np.clip((x_cam_pts - min_x) / width_m, 0.0, 0.999999)
        z_norm = np.clip((z_cam_pts - min_z) / height_m, 0.0, 0.999999)
        x_idx = np.clip(np.floor(x_norm * w_px).astype(np.int32), 0, w_px - 1)
        z_idx = np.clip(np.floor((1.0 - z_norm) * h_px).astype(np.int32), 0, h_px - 1)

        density_grid = np.zeros((h_px, w_px), dtype=np.float32)
        distance_sum = np.zeros((h_px, w_px), dtype=np.float32)
        distance_count = np.zeros((h_px, w_px), dtype=np.uint32)
        height_grid = np.full((h_px, w_px), -np.inf, dtype=np.float32)

        # Confidence-weighted height aggregation
        weighted_height_sum = np.zeros((h_px, w_px), dtype=np.float64)
        weight_sum = np.zeros((h_px, w_px), dtype=np.float64)

        indices = (z_idx, x_idx)
        # Density: count of points (unweighted for backward compat)
        np.add.at(density_grid, indices, 1.0)
        # Distance: weighted by confidence
        np.add.at(distance_sum, indices, (pts_depth * pts_weight).astype(np.float32, copy=False))
        np.add.at(distance_count, indices, 1)
        # Height: accumulate weighted sum and weights for weighted mean
        np.add.at(weighted_height_sum, indices, (pts_y * pts_weight).astype(np.float64))
        np.add.at(weight_sum, indices, pts_weight.astype(np.float64))
        # Also track max height (still useful for some visualizations)
        np.maximum.at(height_grid, indices, np.asarray(pts_y, dtype=np.float32))

        # Compute confidence-weighted mean height
        # Use weighted mean where we have weights, otherwise fall back to max
        has_weight = weight_sum > 1e-9
        # Avoid invalid division warnings: np.where evaluates both branches eagerly.
        weighted_mean_height = height_grid.astype(np.float32, copy=True)
        np.divide(
            weighted_height_sum,
            weight_sum,
            out=weighted_mean_height,
            where=has_weight,
        )

        # For cells with no points at all, mark as NaN
        empty_cells = height_grid == -np.inf
        if np.any(empty_cells):
            weighted_mean_height[empty_cells] = np.nan
            height_grid[empty_cells] = np.nan

        # Use weighted mean as the primary height grid (preserves detail better than max)
        height_grid = weighted_mean_height

        density_max = float(np.max(density_grid)) if density_grid.size else 0.0
        if density_max > 0.0:
            density_grid /= density_max

        if np.isnan(height_grid).any():
            if not np.isnan(height_grid).all():
                min_height = float(np.nanmin(height_grid))
            else:
                min_height = 0.0
            height_grid = np.nan_to_num(height_grid, nan=min_height)
        height_min = float(np.min(height_grid)) if height_grid.size else 0.0
        height_max = float(np.max(height_grid)) if height_grid.size else 0.0
        if height_grid.size:
            height_grid = height_grid - height_min
            height_min = 0.0
            height_max = float(np.max(height_grid)) if height_grid.size else 0.0

        # Compute height gradient magnitude for edge detection
        # First, fill empty/NaN cells with floor level so boundaries don't create false edges
        height_for_gradient = height_grid.copy()
        floor_level = float(np.nanmin(height_grid)) if np.any(np.isfinite(height_grid)) else 0.0
        height_for_gradient = np.nan_to_num(
            height_for_gradient,
            nan=floor_level,
            posinf=floor_level,
            neginf=floor_level,
        )

        # Also mask cells with zero density (no points) to floor level
        if density_grid is not None:
            empty_mask = density_grid < 1e-6
            height_for_gradient[empty_mask] = floor_level

        # Sobel filters capture directional derivatives
        gx = ndi.sobel(height_for_gradient, axis=1, mode='nearest')
        gz = ndi.sobel(height_for_gradient, axis=0, mode='nearest')
        gradient_mag = np.sqrt(gx ** 2 + gz ** 2)

        # Use 95th percentile normalization to prevent outliers from dominating
        gradient_flat = gradient_mag[np.isfinite(gradient_mag)]
        if gradient_flat.size > 0:
            gradient_p95 = float(np.percentile(gradient_flat, 95))
            if gradient_p95 > 1e-6:
                gradient_grid = np.clip(gradient_mag / gradient_p95, 0.0, 1.0).astype(np.float32)
            else:
                gradient_grid = np.zeros_like(height_grid, dtype=np.float32)
        else:
            gradient_grid = np.zeros_like(height_grid, dtype=np.float32)

        # Clean up any remaining NaN
        gradient_grid = np.nan_to_num(gradient_grid, nan=0.0, posinf=0.0, neginf=0.0)

        distance_grid = np.zeros((h_px, w_px), dtype=np.float32)
        nonzero_mask = distance_count > 0
        if np.any(nonzero_mask):
            distance_grid[nonzero_mask] = distance_sum[nonzero_mask] / distance_count[nonzero_mask]
            min_distance = float(np.min(distance_grid[nonzero_mask]))
            max_distance = float(np.max(distance_grid[nonzero_mask]))
        else:
            min_distance = 0.0
            max_distance = 0.0

        bounds = {
            'min_x': float(min_x),
            'max_x': float(max_x),
            'min_z': float(min_z),
            'max_z': float(max_z),
        }

        payload: Dict[str, Any] = {
            'camera_id': camera_id,
            'ts': now_us,
            'snapshot_ts': int(path_entry.stem) if path_entry.stem.isdigit() else None,
            'frame': _FLOORPLAN_FRAME,
            'bounds': bounds,
            'scale_m_per_px': float(width_m / w_px if w_px else grid_res_m),
            'point_count': int(pts_cam.shape[0]),
            'density': {
                'grid_b64': base64.b64encode(density_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
            'height': {
                'grid_b64': base64.b64encode(height_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(height_min),
                'value_max': float(height_max),
            },
            'distance': {
                'grid_b64': base64.b64encode(distance_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(min_distance),
                'value_max': float(max_distance),
            },
            'gradient': {
                'grid_b64': base64.b64encode(gradient_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
        }
        payload['served_from_cache'] = False
        payload['grid_res_m'] = float(grid_res_m)
        payload['max_extent_m'] = float(max_extent_m)
        payload['image_flip'] = {'u': bool(flip_u), 'v': bool(flip_v)}

        self._persist_floorplan_to_disk(camera_id, grid_res_m, max_extent_m, payload)

        with self._cache_lock:
            self._floorplan_cache[cache_key] = dict(payload)
            self._floorplan_cache.move_to_end(cache_key, last=True)
            while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                self._floorplan_cache.popitem(last=False)

        return payload


__all__ = [
    "DepthStorageManager",
    "MapAnythingDepthSource",
    "DepthResult",
    "DepthSummary",
]
