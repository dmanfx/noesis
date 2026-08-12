from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Mapping


LOGGER = logging.getLogger(__name__)


def _bounded_float(
    env: Mapping[str, str],
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = str(env.get(name, default)).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value


def _bounded_int(
    env: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = str(env.get(name, default)).strip()
    try:
        value = int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if str(value) != raw or value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class SourceProgressPolicy:
    stall_seconds: float = 20.0
    recovery_interval_seconds: float = 10.0
    max_recovery_attempts: int = 4
    startup_grace_seconds: float = 45.0
    poll_interval_seconds: float = 1.0
    fps_window_seconds: float = 5.0

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None
    ) -> "SourceProgressPolicy":
        values = os.environ if env is None else env
        return cls(
            stall_seconds=_bounded_float(
                values,
                "NOESIS_SOURCE_PROGRESS_STALL_SECONDS",
                default=20.0,
                minimum=2.0,
                maximum=300.0,
            ),
            recovery_interval_seconds=_bounded_float(
                values,
                "NOESIS_SOURCE_PROGRESS_RECOVERY_INTERVAL_SECONDS",
                default=10.0,
                minimum=1.0,
                maximum=120.0,
            ),
            max_recovery_attempts=_bounded_int(
                values,
                "NOESIS_SOURCE_PROGRESS_MAX_ATTEMPTS",
                default=4,
                minimum=2,
                maximum=12,
            ),
            startup_grace_seconds=_bounded_float(
                values,
                "NOESIS_SOURCE_PROGRESS_STARTUP_GRACE_SECONDS",
                default=45.0,
                minimum=5.0,
                maximum=600.0,
            ),
            poll_interval_seconds=_bounded_float(
                values,
                "NOESIS_SOURCE_PROGRESS_POLL_SECONDS",
                default=1.0,
                minimum=0.1,
                maximum=10.0,
            ),
            fps_window_seconds=_bounded_float(
                values,
                "NOESIS_SOURCE_PROGRESS_FPS_WINDOW_SECONDS",
                default=5.0,
                minimum=1.0,
                maximum=30.0,
            ),
        )


@dataclass(frozen=True)
class SourceProgressFailure(RuntimeError):
    source_id: int
    camera_id: str
    attempts: int
    progress_age_seconds: float

    def __str__(self) -> str:
        return (
            "decoded/dewarped frame progress did not recover "
            f"(source_id={self.source_id}, camera_id={self.camera_id}, "
            f"attempts={self.attempts}, age_seconds={self.progress_age_seconds:.3f})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "camera_id": self.camera_id,
            "attempts": self.attempts,
            "progress_age_seconds": self.progress_age_seconds,
        }


@dataclass
class _SourceState:
    camera_id: str
    frame_count: int = 0
    last_progress_at: float | None = None
    recovery_attempts: int = 0
    last_recovery_attempt_at: float | None = None
    recoveries: int = 0
    recent_frames: deque[float] = field(default_factory=deque)


class DecodedSourceProgressMonitor:
    """Process-owned liveness contract for decoded/dewarped source buffers.

    Packet arrival inside ``nvurisrcbin`` is deliberately not considered
    progress. Only a buffer observed at the configured post-source graph
    boundary resets recovery attempts.
    """

    def __init__(
        self,
        source_labels: Mapping[int, str],
        *,
        policy: SourceProgressPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not source_labels:
            raise ValueError("decoded source progress requires at least one source")
        normalized: dict[int, _SourceState] = {}
        for raw_source_id, raw_camera_id in source_labels.items():
            if isinstance(raw_source_id, bool):
                raise ValueError("decoded source progress source IDs must be integers")
            source_id = int(raw_source_id)
            camera_id = str(raw_camera_id).strip()
            if source_id < 0 or not camera_id or source_id in normalized:
                raise ValueError("decoded source progress source mapping is invalid")
            normalized[source_id] = _SourceState(camera_id=camera_id)
        self._states = normalized
        self.policy = policy or SourceProgressPolicy.from_environment()
        self._clock = clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._fatal_failure: SourceProgressFailure | None = None
        self._on_fatal: Callable[[SourceProgressFailure], None] | None = None

    def record_progress(self, source_id: int, *, now: float | None = None) -> None:
        observed_at = self._clock() if now is None else float(now)
        recovered: tuple[int, str, int, float] | None = None
        with self._lock:
            state = self._states.get(int(source_id))
            if state is None:
                raise KeyError(f"unknown decoded source progress ID: {source_id}")
            previous_progress = state.last_progress_at
            if previous_progress is not None and observed_at < previous_progress:
                raise ValueError("decoded source progress clock regressed")
            if state.recovery_attempts:
                age = (
                    observed_at - previous_progress
                    if previous_progress is not None
                    else observed_at - (self._started_at or observed_at)
                )
                recovered = (
                    int(source_id),
                    state.camera_id,
                    state.recovery_attempts,
                    max(0.0, age),
                )
                state.recoveries += 1
            state.frame_count += 1
            state.last_progress_at = observed_at
            state.recovery_attempts = 0
            state.last_recovery_attempt_at = None
            state.recent_frames.append(observed_at)
            self._trim_recent_locked(state, observed_at)
        if recovered is not None:
            sid, camera_id, attempts, age = recovered
            LOGGER.info(
                json.dumps(
                    {
                        "event": "source_decoded_progress_recovered",
                        "source_id": sid,
                        "camera_id": camera_id,
                        "attempts_before_recovery": attempts,
                        "stalled_seconds": round(age, 3),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )

    def start(
        self, on_fatal: Callable[[SourceProgressFailure], None]
    ) -> threading.Thread:
        if not callable(on_fatal):
            raise TypeError("decoded source progress fatal callback must be callable")
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("decoded source progress monitor is already started")
            self._started_at = self._clock()
            self._fatal_failure = None
            self._on_fatal = on_fatal
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name="NoesisDecodedSourceProgress",
                daemon=False,
            )
            self._thread = thread
            thread.start()
            return thread

    def stop(self, *, timeout_seconds: float = 3.0) -> dict[str, object]:
        with self._lock:
            thread = self._thread
        self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        quiesced = thread is None or not thread.is_alive()
        if quiesced:
            with self._lock:
                if self._thread is thread:
                    self._thread = None
        return {
            "present": True,
            "quiesced": quiesced,
            "fatal": self._fatal_failure is not None,
        }

    def evaluate(self, *, now: float | None = None) -> SourceProgressFailure | None:
        checked_at = self._clock() if now is None else float(now)
        attempt_logs: list[dict[str, object]] = []
        fatal: SourceProgressFailure | None = None
        with self._lock:
            if self._fatal_failure is not None:
                return self._fatal_failure
            started_at = self._started_at
            if started_at is None:
                raise RuntimeError("decoded source progress monitor has not started")
            if checked_at < started_at:
                raise ValueError("decoded source progress clock regressed")
            if checked_at - started_at < self.policy.startup_grace_seconds:
                return None
            for source_id, state in sorted(self._states.items()):
                reference = (
                    state.last_progress_at
                    if state.last_progress_at is not None
                    else started_at
                )
                age = checked_at - reference
                if age < self.policy.stall_seconds:
                    continue
                if (
                    state.last_recovery_attempt_at is not None
                    and checked_at - state.last_recovery_attempt_at
                    < self.policy.recovery_interval_seconds
                ):
                    continue
                state.recovery_attempts += 1
                state.last_recovery_attempt_at = checked_at
                attempt_logs.append(
                    {
                        "event": "source_decoded_progress_stalled",
                        "source_id": source_id,
                        "camera_id": state.camera_id,
                        "attempt": state.recovery_attempts,
                        "max_attempts": self.policy.max_recovery_attempts,
                        "progress_age_seconds": round(age, 3),
                    }
                )
                if state.recovery_attempts >= self.policy.max_recovery_attempts:
                    fatal = SourceProgressFailure(
                        source_id=source_id,
                        camera_id=state.camera_id,
                        attempts=state.recovery_attempts,
                        progress_age_seconds=max(0.0, age),
                    )
                    self._fatal_failure = fatal
                    break
        for payload in attempt_logs:
            LOGGER.warning(
                json.dumps(payload, separators=(",", ":"), sort_keys=True)
            )
        return fatal

    def snapshot(self, *, now: float | None = None) -> dict[str, object]:
        checked_at = self._clock() if now is None else float(now)
        with self._lock:
            started_at = self._started_at
            rows: dict[str, dict[str, object]] = {}
            healthy = self._fatal_failure is None
            for source_id, state in sorted(self._states.items()):
                self._trim_recent_locked(state, checked_at)
                age = (
                    max(0.0, checked_at - state.last_progress_at)
                    if state.last_progress_at is not None
                    else None
                )
                if state.recovery_attempts:
                    status = "recovering"
                    healthy = False
                elif state.last_progress_at is None:
                    status = "starting"
                else:
                    status = "running"
                rows[str(source_id)] = {
                    "camera_id": state.camera_id,
                    "status": status,
                    "frames": state.frame_count,
                    "fps": len(state.recent_frames) / self.policy.fps_window_seconds,
                    "last_progress_age_seconds": age,
                    "recovery_attempts": state.recovery_attempts,
                    "recoveries": state.recoveries,
                }
            return {
                "healthy": healthy,
                "started": started_at is not None,
                "startup_grace_remaining_seconds": (
                    max(
                        0.0,
                        self.policy.startup_grace_seconds - (checked_at - started_at),
                    )
                    if started_at is not None
                    else self.policy.startup_grace_seconds
                ),
                "policy": {
                    "stall_seconds": self.policy.stall_seconds,
                    "recovery_interval_seconds": self.policy.recovery_interval_seconds,
                    "max_recovery_attempts": self.policy.max_recovery_attempts,
                    "startup_grace_seconds": self.policy.startup_grace_seconds,
                },
                "sources": rows,
                "fatal": (
                    self._fatal_failure.to_dict()
                    if self._fatal_failure is not None
                    else None
                ),
            }

    def _run(self) -> None:
        while not self._stop.wait(self.policy.poll_interval_seconds):
            failure = self.evaluate()
            if failure is None:
                continue
            callback = self._on_fatal
            if callback is not None:
                try:
                    callback(failure)
                except BaseException:
                    LOGGER.exception(
                        "Decoded source progress fatal callback failed"
                    )
            return

    def _trim_recent_locked(self, state: _SourceState, now: float) -> None:
        cutoff = now - self.policy.fps_window_seconds
        while state.recent_frames and state.recent_frames[0] < cutoff:
            state.recent_frames.popleft()


__all__ = [
    "DecodedSourceProgressMonitor",
    "SourceProgressFailure",
    "SourceProgressPolicy",
]
