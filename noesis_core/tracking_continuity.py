from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock
from typing import Any, Mapping, MutableMapping, Sequence


TRACKING_CONTINUITY_CONTRACT = "noesis.tracking.publication-continuity"
TRACKING_CONTINUITY_CONTRACT_VERSION = 1
TRACKER_LIFECYCLE_GENERATION_FIELD = "tracker_lifecycle_generation"


def _exact_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _tracker_id(track: Mapping[str, Any]) -> int:
    raw = track.get("tracker_id", track.get("track_id"))
    return _exact_nonnegative_int(raw, "tracker_id")


def pair_safe_publication_interval_s(
    *,
    track_count: int,
    tracking_interval_s: float,
    empty_tracking_interval_s: float,
    bev_interval_s: float,
    bev_active: bool,
) -> float:
    """Return one cadence that cannot emit an unpaired BEV frame."""

    count = _exact_nonnegative_int(track_count, "track_count")

    def _interval(value: float, label: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be finite and nonnegative") from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
        return parsed

    tracking = _interval(
        empty_tracking_interval_s if count == 0 else tracking_interval_s,
        "tracking interval",
    )
    if not bev_active:
        return tracking
    bev = _interval(bev_interval_s, "BEV interval")
    return max(tracking, bev)


@dataclass(frozen=True)
class _ActiveTrackLifecycle:
    camera_id: str
    tracker_id: int
    generation: int
    last_seen_frame_id: int
    last_seen_observed_at_us: int


@dataclass(frozen=True)
class TrackingContinuityUpdate:
    source_id: int
    frame_id: int
    observed_at_us: int
    tracker_keys_changed: bool
    tombstones: tuple[dict[str, object], ...]
    active_lifecycles: tuple[_ActiveTrackLifecycle, ...]


class TrackingLifecycleRegistry:
    """Stamp process-frame tracker generations and explicit disappearance rows.

    The registry is updated for every processed frame, including frames that are
    later skipped by WebSocket rate limiting.  Reappearance after even one
    processed absence therefore receives a new generation.  Published presence
    is committed separately after telemetry publication succeeds, so a
    disappearance tombstone always names the exact last *published* frame/time,
    never a processed frame that downstream consumers did not observe.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._active_by_source: dict[int, dict[int, _ActiveTrackLifecycle]] = {}
        self._next_generation_by_source: dict[int, int] = {}
        self._published_by_source: dict[int, dict[int, _ActiveTrackLifecycle]] = {}
        self._pending_tombstones_by_source: dict[
            int, dict[tuple[int, int], _ActiveTrackLifecycle]
        ] = {}
        self._last_published_frame_by_source: dict[int, int] = {}

    def update_frame(
        self,
        *,
        source_id: int,
        camera_id: str,
        frame_id: int,
        observed_at_us: int,
        tracks: Sequence[MutableMapping[str, Any]],
    ) -> TrackingContinuityUpdate:
        source = _exact_nonnegative_int(source_id, "source_id")
        frame = _exact_nonnegative_int(frame_id, "frame_id")
        observed = _exact_nonnegative_int(observed_at_us, "observed_at_us")
        if observed <= 0:
            raise ValueError("observed_at_us must be positive")
        camera = str(camera_id or "").strip()
        if not camera:
            raise ValueError("camera_id is required")

        with self._lock:
            previous = self._active_by_source.get(source, {})
            current_ids: list[int] = []
            indexed: dict[int, MutableMapping[str, Any]] = {}
            for track in tracks:
                if not isinstance(track, MutableMapping):
                    raise ValueError("tracking lifecycle requires mutable track mappings")
                tracker = _tracker_id(track)
                if tracker in indexed:
                    raise ValueError("source frame contains a duplicate tracker_id")
                track_camera = str(track.get("camera_id") or camera).strip()
                if track_camera != camera:
                    raise ValueError("track camera_id differs from source frame camera_id")
                current_ids.append(tracker)
                indexed[tracker] = track

            previous_ids = set(previous)
            current_id_set = set(current_ids)
            published = self._published_by_source.get(source, {})
            pending = dict(self._pending_tombstones_by_source.get(source, {}))
            next_generation = self._next_generation_by_source.get(source, 1)
            current: dict[int, _ActiveTrackLifecycle] = {}
            for tracker in current_ids:
                prior = previous.get(tracker)
                if prior is None:
                    generation = next_generation
                    next_generation += 1
                else:
                    generation = prior.generation
                indexed[tracker][TRACKER_LIFECYCLE_GENERATION_FIELD] = generation
                current[tracker] = _ActiveTrackLifecycle(
                    camera_id=camera,
                    tracker_id=tracker,
                    generation=generation,
                    last_seen_frame_id=frame,
                    last_seen_observed_at_us=observed,
                )

            for tracker in sorted(previous_ids - current_id_set):
                processed_prior = previous[tracker]
                published_prior = published.get(tracker)
                if (
                    published_prior is None
                    or published_prior.generation != processed_prior.generation
                ):
                    continue
                pending.setdefault(
                    (published_prior.tracker_id, published_prior.generation),
                    published_prior,
                )
            tombstones = tuple(
                {
                    "camera_id": pending[key].camera_id,
                    "tracker_id": pending[key].tracker_id,
                    TRACKER_LIFECYCLE_GENERATION_FIELD: pending[key].generation,
                    "last_seen_frame_id": pending[key].last_seen_frame_id,
                    "last_seen_observed_at_us": (
                        pending[key].last_seen_observed_at_us
                    ),
                    "disappeared_at_frame_id": frame,
                    "disappeared_at_observed_at_us": observed,
                }
                for key in sorted(pending)
            )
            self._active_by_source[source] = current
            self._next_generation_by_source[source] = next_generation
            self._pending_tombstones_by_source[source] = pending
            return TrackingContinuityUpdate(
                source_id=source,
                frame_id=frame,
                observed_at_us=observed,
                tracker_keys_changed=(
                    previous_ids != current_id_set or bool(tombstones)
                ),
                tombstones=tombstones,
                active_lifecycles=tuple(
                    current[tracker] for tracker in sorted(current)
                ),
            )

    def mark_published(self, update: TrackingContinuityUpdate) -> None:
        """Commit the exact public presence represented by a successful publish."""

        if not isinstance(update, TrackingContinuityUpdate):
            raise ValueError("tracking publication requires its continuity update")
        with self._lock:
            previous_frame = self._last_published_frame_by_source.get(update.source_id)
            if previous_frame is not None and update.frame_id <= previous_frame:
                raise ValueError("published source frame must advance monotonically")
            pending = self._pending_tombstones_by_source.get(update.source_id, {})
            expected_tombstones = tuple(
                {
                    "camera_id": pending[key].camera_id,
                    "tracker_id": pending[key].tracker_id,
                    TRACKER_LIFECYCLE_GENERATION_FIELD: pending[key].generation,
                    "last_seen_frame_id": pending[key].last_seen_frame_id,
                    "last_seen_observed_at_us": (
                        pending[key].last_seen_observed_at_us
                    ),
                    "disappeared_at_frame_id": update.frame_id,
                    "disappeared_at_observed_at_us": update.observed_at_us,
                }
                for key in sorted(pending)
            )
            if update.tombstones != expected_tombstones:
                raise ValueError("published tombstone set is stale")
            published = dict(self._published_by_source.get(update.source_id, {}))
            for tombstone in update.tombstones:
                tracker = int(tombstone["tracker_id"])
                generation = int(
                    tombstone[TRACKER_LIFECYCLE_GENERATION_FIELD]
                )
                prior = published.get(tracker)
                if prior is not None and prior.generation == generation:
                    published.pop(tracker, None)
                pending.pop((tracker, generation), None)
            for lifecycle in update.active_lifecycles:
                published[lifecycle.tracker_id] = lifecycle
            self._published_by_source[update.source_id] = published
            self._pending_tombstones_by_source[update.source_id] = pending
            self._last_published_frame_by_source[update.source_id] = update.frame_id


__all__ = [
    "TRACKER_LIFECYCLE_GENERATION_FIELD",
    "TRACKING_CONTINUITY_CONTRACT",
    "TRACKING_CONTINUITY_CONTRACT_VERSION",
    "TrackingContinuityUpdate",
    "TrackingLifecycleRegistry",
    "pair_safe_publication_interval_s",
]
