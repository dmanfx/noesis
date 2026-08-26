from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # DS metadata bindings (optional for unit tests / non-DS hosts)
    import pyds  # type: ignore
except Exception:  # pragma: no cover - import-safe fallback when DS libs absent
    pyds = None  # type: ignore

# Optional Service Maker latency shim
try:  # pragma: no cover - runtime optional
    import noesis_latency_ext  # type: ignore
except Exception:  # pragma: no cover
    noesis_latency_ext = None  # type: ignore

# Optional Service Maker latency shim
try:  # pragma: no cover - runtime optional
    import noesis_latency_ext  # type: ignore
except Exception:  # pragma: no cover
    noesis_latency_ext = None  # type: ignore


class NvDsFrameLatencyInfo(ctypes.Structure):
    """ctypes mirror of NvDsFrameLatencyInfo from nvds_latency_meta.h."""

    _fields_ = [
        ("source_id", ctypes.c_uint),
        ("frame_num", ctypes.c_uint),
        ("comp_in_timestamp", ctypes.c_double),
        ("latency", ctypes.c_double),  # milliseconds
    ]


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _load_nvds_meta() -> Optional[ctypes.CDLL]:
    """Best-effort load of the DeepStream latency meta library.

    On some DeepStream installs the `nvds_measure_buffer_latency` symbol is
    exported by `libnvdsgst_meta.so` (not `libnvds_meta.so`), so we try both.
    """
    candidates: List[str] = []
    override = os.environ.get("NOESIS_NVDS_LATENCY_LIB", "").strip()
    if override:
        for part in override.split(","):
            p = part.strip()
            if p:
                candidates.append(p)
    for libname in ("nvdsgst_meta", "nvds_meta"):
        found = ctypes.util.find_library(libname)
        if found:
            candidates.append(found)
    # Common DeepStream install locations
    candidates.extend(
        [
            "/opt/nvidia/deepstream/deepstream/lib/libnvdsgst_meta.so",
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_meta.so",
            "libnvdsgst_meta.so",
            "libnvds_meta.so",
        ]
    )
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            getattr(lib, "nvds_measure_buffer_latency")
            return lib
        except Exception:
            continue
    return None


def _percentile_linear(sorted_values: List[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0,1]) over sorted values."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    p = min(1.0, max(0.0, float(p)))
    pos = p * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


@dataclass
class LatencySnapshot:
    enabled: bool
    window_sec: float
    count: int
    p50: Optional[float]
    p95: Optional[float]
    max: Optional[float]
    last_sample_age_sec: Optional[float]
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "enabled": bool(self.enabled),
            "window_sec": float(self.window_sec),
            "count": int(self.count),
            "p50": self.p50,
            "p95": self.p95,
            "max": self.max,
            "last_sample_age_sec": self.last_sample_age_sec,
        }
        if self.reason:
            payload["reason"] = str(self.reason)
        return payload


class LatencyWindow:
    def __init__(self, window_sec: float) -> None:
        self.window_sec = float(max(0.1, window_sec))
        self._samples: Deque[Tuple[float, float]] = deque()

    def add(self, now_sec: float, latency_ms: float) -> None:
        self._samples.append((float(now_sec), float(latency_ms)))
        self.prune(now_sec)

    def prune(self, now_sec: float) -> None:
        cutoff = float(now_sec) - self.window_sec
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def snapshot(self, now_sec: float) -> LatencySnapshot:
        self.prune(now_sec)
        if not self._samples:
            return LatencySnapshot(
                enabled=True,
                window_sec=self.window_sec,
                count=0,
                p50=None,
                p95=None,
                max=None,
                last_sample_age_sec=None,
            )
        values = [lat for _, lat in self._samples]
        values.sort()
        p50 = _percentile_linear(values, 0.50)
        p95 = _percentile_linear(values, 0.95)
        vmax = float(values[-1])
        last_ts = float(self._samples[-1][0])
        age = max(0.0, float(now_sec) - last_ts)
        return LatencySnapshot(
            enabled=True,
            window_sec=self.window_sec,
            count=len(values),
            p50=float(p50),
            p95=float(p95),
            max=vmax,
            last_sample_age_sec=float(age),
        )


class LatencyCollector:
    """Collect per-frame latency samples from DeepStream built-in latency measurement.

    This collector is safe to construct on non-DeepStream hosts; it simply stays disabled.
    """

    def __init__(self, window_sec: float = 10.0) -> None:
        self.window_sec = float(max(0.1, window_sec))
        self._lock = threading.Lock()
        self._windows: Dict[int, LatencyWindow] = {}
        self._disabled_reason: Optional[str] = None

        self._enabled_by_env = _env_truthy("NVDS_ENABLE_LATENCY_MEASUREMENT", default=False)
        self._lib = _load_nvds_meta() if self._enabled_by_env else None
        self._scratch_n: int = 0
        self._scratch: Optional[ctypes.Array] = None

        self._warned_missing_pyds = False
        self._warned_missing_lib = False
        self._warned_measure_failed = False

        if self._enabled_by_env and pyds is None:
            self._disabled_reason = "pyds_not_available"
        elif self._enabled_by_env and self._lib is None:
            self._disabled_reason = "libnvds_meta_not_found"

        if self._lib is not None:
            try:
                fn = self._lib.nvds_measure_buffer_latency
                fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvDsFrameLatencyInfo)]
                fn.restype = ctypes.c_uint
            except Exception:
                self._disabled_reason = "nvds_measure_buffer_latency_unavailable"
                self._lib = None

    @property
    def enabled(self) -> bool:
        return bool(self._enabled_by_env and self._disabled_reason is None)

    def disabled_snapshot(self) -> Dict[str, object]:
        reason = self._disabled_reason or ("env_disabled" if not self._enabled_by_env else "disabled")
        return LatencySnapshot(False, self.window_sec, 0, None, None, None, None, reason=reason).to_dict()

    def disable(self, reason: str) -> None:
        with self._lock:
            self._disabled_reason = str(reason or "disabled")

    def clear(self) -> None:
        with self._lock:
            self._windows.clear()

    def _ensure_scratch(self, n: int) -> ctypes.Array:
        if n <= 0:
            raise ValueError("scratch size must be > 0")
        if self._scratch is None or self._scratch_n != n:
            self._scratch_n = int(n)
            self._scratch = (NvDsFrameLatencyInfo * int(n))()
        return self._scratch

    def record_from_gst_buffer_ptr(self, gst_buffer_ptr: int) -> None:
        """Extract latency samples from a GstBuffer* and record them into rolling windows."""
        if not self.enabled:
            # Surface the most important disable reasons once, but do not spam logs.
            if self._enabled_by_env and pyds is None and not self._warned_missing_pyds:
                self._warned_missing_pyds = True
                logger.warning("Latency enabled via env but pyds is not available; disabling latency stats")
            if self._enabled_by_env and self._lib is None and not self._warned_missing_lib:
                self._warned_missing_lib = True
                logger.warning("Latency enabled via env but libnvds_meta.so not found; disabling latency stats")
            return
        if not gst_buffer_ptr:
            return
        if pyds is None or self._lib is None:
            return

        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(int(gst_buffer_ptr))
        except Exception:
            if not self._warned_measure_failed:
                self._warned_measure_failed = True
                logger.exception("Failed to retrieve NvDsBatchMeta for latency measurement; disabling latency stats")
            self.disable("batch_meta_unavailable")
            return
        if not batch_meta:
            return

        max_frames = 0
        try:
            pyds.nvds_acquire_meta_lock(batch_meta)
            max_frames = int(getattr(batch_meta, "max_frames_in_batch", 0) or 0)
            if not max_frames:
                max_frames = int(getattr(batch_meta, "num_frames_in_batch", 0) or 0)
        except Exception:
            max_frames = 0
        finally:
            try:
                pyds.nvds_release_meta_lock(batch_meta)
            except Exception:
                pass
        if max_frames <= 0:
            return
        now = time.time()
        with self._lock:
            scratch = self._ensure_scratch(max_frames)
            try:
                fn = self._lib.nvds_measure_buffer_latency
                num_sources = int(fn(ctypes.c_void_p(int(gst_buffer_ptr)), scratch))
            except Exception:
                if not self._warned_measure_failed:
                    self._warned_measure_failed = True
                    logger.exception("nvds_measure_buffer_latency failed; disabling latency stats")
                self._disabled_reason = "measure_failed"
                return
            if num_sources <= 0:
                return

            for idx in range(num_sources):
                try:
                    info = scratch[idx]
                    source_id = int(info.source_id)
                    latency_ms = float(info.latency)
                    if math.isnan(latency_ms) or latency_ms < 0.0:
                        continue
                    window = self._windows.get(source_id)
                    if window is None:
                        window = LatencyWindow(self.window_sec)
                        self._windows[source_id] = window
                    window.add(now, latency_ms)
                except Exception:
                    # Defensive: never let a single bad sample break telemetry.
                    continue

    # -------------------- Service Maker path --------------------

    def record_from_sm_buffer(self, buffer: object) -> None:
        """Extract latency samples from a Service Maker Buffer via native shim."""
        if not self.enabled:
            return
        if noesis_latency_ext is None:
            if not self._warned_measure_failed:
                self._warned_measure_failed = True
                logger.warning("Latency shim unavailable (noesis_latency_ext); disabling latency stats")
            self.disable("sm_latency_shim_unavailable")
            return
        try:
            samples = noesis_latency_ext.measure_buffer_latency(buffer)
        except Exception:
            if not self._warned_measure_failed:
                self._warned_measure_failed = True
                logger.exception("SM latency measurement failed; disabling latency stats")
            self.disable("sm_measure_failed")
            return
        now = time.time()
        with self._lock:
            for sample in samples:
                try:
                    source_id = int(sample.get("source_id", 0))
                    latency_ms = float(sample.get("latency_ms", 0.0))
                except Exception:
                    continue
                self._windows.setdefault(source_id, LatencyWindow(self.window_sec)).add(now, latency_ms)

    def snapshot_by_source(self, now_sec: Optional[float] = None) -> Dict[int, Dict[str, object]]:
        now = float(time.time() if now_sec is None else now_sec)
        if not self.enabled:
            return {}
        with self._lock:
            snapshots: Dict[int, Dict[str, object]] = {}
            for source_id, window in list(self._windows.items()):
                try:
                    snapshots[int(source_id)] = window.snapshot(now).to_dict()
                except Exception:
                    continue
            return snapshots

    def snapshot_aggregate(self, now_sec: Optional[float] = None) -> Dict[str, object]:
        now = float(time.time() if now_sec is None else now_sec)
        if not self.enabled:
            return self.disabled_snapshot()

        with self._lock:
            values: List[float] = []
            last_ts: Optional[float] = None
            for window in list(self._windows.values()):
                window.prune(now)
                # Aggregate across all sources. Copy the small per-window sample deque.
                samples = list(window._samples)  # type: ignore[attr-defined]
                if samples:
                    last_ts = max(last_ts or samples[-1][0], samples[-1][0])
                    values.extend([lat for _, lat in samples])

        if not values:
            return LatencySnapshot(True, self.window_sec, 0, None, None, None, None).to_dict()

        values.sort()
        p50 = _percentile_linear(values, 0.50)
        p95 = _percentile_linear(values, 0.95)
        vmax = float(values[-1])
        age = max(0.0, now - float(last_ts)) if last_ts is not None else None
        return LatencySnapshot(True, self.window_sec, len(values), float(p50), float(p95), vmax, age).to_dict()
