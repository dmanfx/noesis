"""HTTP client and storage utilities for MapAnything depth inference."""
from __future__ import annotations

import base64
import json
import logging
import os
import queue
import shutil
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from concurrent.futures import Future

import cv2
import numpy as np
import requests
import zarr
from numcodecs import Blosc

from adapters.mapanything_adapter import ViewBuildResult, build_mono_view
from mapanything_config import ServiceConfig, load_service_config
from utils.rate_limited_logger import RateLimitedLogger


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
    ) -> None:
        self.base_path = base_path
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
        if self._max_total_bytes is not None and self._max_total_bytes < 10 * 1024 * 1024:
            self._logger.warning(
                "Configured max_total_bytes=%s is very small; increasing to 10MB minimum",
                self._max_total_bytes,
            )
            self._max_total_bytes = 10 * 1024 * 1024
        self._seed_existing_entries()
        if self._async_enabled:
            self._start_writer()

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
        assert self._queue is not None
        assert self._stop_event is not None
        while True:
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            try:
                self._write_snapshot(job)
            except Exception as exc:
                self._logger.error(f"Depth snapshot write failed for {job.camera_id}: {exc}")
            finally:
                self._queue.task_done()

    def shutdown(self, *, wait: bool = True) -> None:
        if not self._async_enabled or self._stop_event is None:
            return
        self._stop_event.set()
        if wait:
            threads = self._collect_alive_threads()
            for thread in threads:
                thread.join(timeout=2.0)
        with self._worker_lock:
            self._writer_threads.clear()
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
        lock = self._get_lock(camera_id)
        with lock:
            index = self._get_index(camera_id)
            index.append((ts_us, dest_path))
            self._enforce_limits(camera_id, index)

    def _write_snapshot(self, job: _SnapshotJob) -> None:
        job.dest_path.parent.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        root = zarr.open_group(str(job.dest_path), mode="w")
        chunk_shape = (min(128, job.depth.shape[0]), min(128, job.depth.shape[1]))
        root.create_dataset(
            "depth_z",
            data=job.depth,
            compressor=compressor,
            chunks=chunk_shape,
            overwrite=True,
        )
        root.create_dataset(
            "conf",
            data=job.conf,
            compressor=compressor,
            chunks=chunk_shape,
            overwrite=True,
        )
        root.create_dataset(
            "mask",
            data=job.mask,
            compressor=compressor,
            chunks=chunk_shape,
            overwrite=True,
        )
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
        while total_bytes > self._max_total_bytes and index:
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


class MapAnythingDepthSource:
    """Client for invoking MapAnything inference service and persisting results."""

    def __init__(self, config: Optional[ServiceConfig] = None) -> None:
        self.config = config or load_service_config()
        self.session = requests.Session()
        self.logger = RateLimitedLogger(logging.getLogger(__name__), rate_limit_seconds=2.0)
        self.storage = DepthStorageManager(
            Path(self.config.storage.depth_base),
            max_snapshots_per_camera=self.config.storage.max_snapshots_per_camera,
            retention_minutes=self.config.storage.snapshot_retention_minutes,
            max_total_bytes=getattr(self.config.storage, 'max_total_bytes', None),
            enable_async=getattr(self.config.storage, 'async_enabled', True),
            max_queue_size=max(1, int(getattr(self.config.storage, 'queue_size', 32))),
            worker_count=int(getattr(self.config.storage, 'async_workers', 0)),
            max_worker_count=int(getattr(self.config.storage, 'async_max_workers', 0)),
        )
        self.min_conf = float(self.config.performance.min_conf)
        self.mono_interval = 1.0 / max(self.config.performance.mono_freq_hz, 1e-6)
        self.last_request_per_camera: Dict[str, float] = {}
        self.timeout = 5.0
        self._cache_lock = threading.Lock()
        self._depth_payload_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._floorplan_cache: "OrderedDict[Tuple[str, float, float], Dict[str, Any]]" = OrderedDict()
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

    def load_latest_depth(self, camera_id: str, ts_max: Optional[int] = None) -> Optional[Dict[str, Any]]:
        cache_key = camera_id
        with self._cache_lock:
            cached = self._depth_payload_cache.get(cache_key)
            if cached:
                cached_ts = cached.get('ts', 0)
                if ts_max is None or cached_ts <= (ts_max or cached_ts):
                    return dict(cached)

        path = self.storage.latest_entry(camera_id, ts_max)
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
    ) -> Dict[str, Any]:
        """Generate a per-camera top-down (XZ) blueprint view from the latest depth snapshot."""
        if not camera_id:
            return {'error': 'camera_required', 'ts': int(time.time() * 1_000_000)}

        cache_key = (camera_id, float(grid_res_m), float(max_extent_m))
        with self._cache_lock:
            cached = self._floorplan_cache.get(cache_key)
            if cached:
                age_us = int(time.time() * 1_000_000) - cached.get('snapshot_ts', cached.get('ts', 0))
                if age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                    return dict(cached)

        now_us = int(time.time() * 1_000_000)
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
            legacy_cam = cameras_node.get(camera_id) if isinstance(cameras_node, dict) else None
            if isinstance(legacy_cam, dict):
                if intr is None:
                    maybe_intr = legacy_cam.get('intrinsics')
                    if isinstance(maybe_intr, (list, tuple)) and len(maybe_intr) == 9:
                        intr = maybe_intr
                if extr is None:
                    maybe_extr = legacy_cam.get('extrinsics')
                    if isinstance(maybe_extr, dict):
                        extr = maybe_extr.get('E')

        if intr is None or extr is None:
            return {'error': 'missing_calibration', 'camera_id': camera_id, 'ts': now_us}

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

        valid = np.isfinite(depth)
        valid &= depth > 0.1
        valid &= depth < 50.0
        valid &= mask
        if np.isfinite(self.min_conf):
            valid &= conf >= float(self.min_conf)

        if not np.any(valid):
            return {'error': 'no_points', 'camera_id': camera_id, 'ts': now_us, 'point_count': 0}

        h_img, w_img = depth.shape
        grid_u, grid_v = np.meshgrid(
            np.arange(w_img, dtype=np.float32),
            np.arange(h_img, dtype=np.float32),
            indexing='xy'
        )

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
        x_cam_pts = pts_cam[:, 0]
        z_cam_pts = pts_cam[:, 2]

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

        indices = (z_idx, x_idx)
        np.add.at(density_grid, indices, 1.0)
        np.add.at(distance_sum, indices, pts_depth.astype(np.float32, copy=False))
        np.add.at(distance_count, indices, 1)
        np.maximum.at(height_grid, indices, np.asarray(pts_y, dtype=np.float32))

        # Restore NaNs for empty cells after vectorised max accumulation
        empty_cells = height_grid == -np.inf
        if np.any(empty_cells):
            height_grid = height_grid.astype(np.float32, copy=False)
            height_grid[empty_cells] = np.nan

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
        }

        with self._cache_lock:
            self._floorplan_cache[cache_key] = dict(payload)
            self._floorplan_cache.move_to_end(cache_key, last=True)
            while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                self._floorplan_cache.popitem(last=False)

        return payload


__all__ = [
    "MapAnythingDepthSource",
    "DepthResult",
    "DepthSummary",
]
