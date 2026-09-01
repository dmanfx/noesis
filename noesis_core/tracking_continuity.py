from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, replace
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


def _track_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        left, top, width, height = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (left, top, width, height)):
        return None
    if width <= 1.0 or height <= 1.0:
        return None
    return left, top, width, height


def _media_pts_ns(value: object) -> int | None:
    """Normalize one usable source PTS without accepting CLOCK_TIME_NONE."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed <= 0 or parsed >= (1 << 64) - 1:
        return None
    return parsed


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
    last_seen_media_pts_ns: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class TrackingContinuityUpdate:
    source_id: int
    frame_id: int
    observed_at_us: int
    tracker_keys_changed: bool
    tombstones: tuple[dict[str, object], ...]
    active_lifecycles: tuple[_ActiveTrackLifecycle, ...]
    # A source epoch changes when a camera/file source reconnects and its
    # frame counter rewinds.  Keeping this scalar with the cohort prevents a
    # queued publication from being mistaken for an older frame in the new
    # epoch.  Epoch zero is the original process epoch.
    source_epoch: int = 0
    # Media-side ``update_frame`` can run ahead of the publication worker. A
    # worker must call ``prepare_publication`` against the then-current
    # last-published lifecycle immediately before emitting this cohort.
    publication_prepared: bool = False


class TrackingLifecycleRegistry:
    """Stamp process-frame tracker generations and explicit disappearance rows.

    The registry is updated for every processed frame, including frames that are
    later skipped by WebSocket rate limiting.  A brief same-source reappearance
    may reuse its generation only when the numeric tracker ID, source-media
    elapsed time, and bounding-box continuity all agree. Published presence is
    committed separately after telemetry publication succeeds, so a
    disappearance tombstone always names the exact last *published* frame/time,
    never a processed frame that downstream consumers did not observe.
    """

    def __init__(
        self,
        *,
        reappearance_grace_s: float = 0.75,
        max_retired_per_source: int = 256,
        max_center_displacement_norm: float = 0.65,
        max_bbox_size_ratio: float = 1.8,
    ) -> None:
        self._lock = RLock()
        self._active_by_source: dict[int, dict[int, _ActiveTrackLifecycle]] = {}
        self._next_generation_by_source: dict[int, int] = {}
        self._published_by_source: dict[int, dict[int, _ActiveTrackLifecycle]] = {}
        self._last_published_frame_by_source: dict[int, int] = {}
        self._source_epoch_by_source: dict[int, int] = {}
        self._reappearance_grace_us = max(
            0,
            int(float(reappearance_grace_s) * 1_000_000),
        )
        self._reappearance_grace_ns = int(self._reappearance_grace_us) * 1_000
        self._max_retired_per_source = max(1, int(max_retired_per_source))
        self._max_center_displacement_norm = max(
            0.0,
            float(max_center_displacement_norm),
        )
        self._max_bbox_size_ratio = max(1.0, float(max_bbox_size_ratio))
        # Only compact lifecycle+bbox scalars are retained. Ordered eviction
        # makes the memory bound deterministic even under tracker-ID churn.
        self._retired_by_source: dict[
            int, OrderedDict[int, _ActiveTrackLifecycle]
        ] = {}
        # New tracker IDs are encountered while the media callback is still
        # assembling world rows.  Reserve their generations atomically per
        # source frame so two people in one frame cannot both use the same
        # value before update_frame commits the final public subset.
        self._generation_preview_by_source: dict[
            int, tuple[int, int, int | None, dict[int, int]]
        ] = {}

    def reset_source(self, source_id: int) -> int:
        """Start a new source epoch after a validated reconnect/rewind.

        The numeric tracker generation counter is intentionally retained so a
        reused tracker ID cannot collide with a prior epoch. Published
        presence is cleared because its frame IDs belong to the previous
        source timeline. The caller must serialize this reset with its
        source-frame admission.
        """

        source = _exact_nonnegative_int(source_id, "source_id")
        with self._lock:
            self._active_by_source.pop(source, None)
            self._published_by_source.pop(source, None)
            self._last_published_frame_by_source.pop(source, None)
            self._retired_by_source.pop(source, None)
            self._generation_preview_by_source.pop(source, None)
            epoch = int(self._source_epoch_by_source.get(source, 0)) + 1
            self._source_epoch_by_source[source] = epoch
            return epoch

    def source_epoch(self, source_id: int) -> int:
        source = _exact_nonnegative_int(source_id, "source_id")
        with self._lock:
            return int(self._source_epoch_by_source.get(source, 0))

    @staticmethod
    def _reappearance_gap_ns(
        lifecycle: _ActiveTrackLifecycle,
        *,
        observed_at_us: int,
        media_pts_ns: int | None,
    ) -> int | None:
        """Prefer strictly advancing source time; use host time only as fallback."""

        prior_media = lifecycle.last_seen_media_pts_ns
        if prior_media is not None and media_pts_ns is not None:
            if int(media_pts_ns) <= int(prior_media):
                # Decreasing PTS is a source-boundary condition in the DS9
                # producer. Equal PTS cannot prove any elapsed media interval.
                return None
            return int(media_pts_ns) - int(prior_media)
        gap_us = int(observed_at_us) - int(lifecycle.last_seen_observed_at_us)
        if gap_us <= 0:
            return None
        return int(gap_us) * 1_000

    def _prune_retired_locked(
        self,
        source_id: int,
        observed_at_us: int,
        media_pts_ns: int | None,
    ) -> None:
        retired = self._retired_by_source.get(int(source_id))
        if not retired:
            return
        expired: list[int] = []
        for tracker_id, lifecycle in retired.items():
            gap_ns = self._reappearance_gap_ns(
                lifecycle,
                observed_at_us=int(observed_at_us),
                media_pts_ns=media_pts_ns,
            )
            if gap_ns is not None and gap_ns > int(self._reappearance_grace_ns):
                expired.append(int(tracker_id))
        for tracker_id in expired:
            retired.pop(int(tracker_id), None)
        if not retired:
            self._retired_by_source.pop(int(source_id), None)

    def _bbox_is_compatible(
        self,
        previous: tuple[float, float, float, float] | None,
        current: tuple[float, float, float, float] | None,
    ) -> bool:
        if previous is None or current is None:
            return False
        prev_left, prev_top, prev_width, prev_height = previous
        curr_left, curr_top, curr_width, curr_height = current
        size_ratio = max(
            prev_width / curr_width,
            curr_width / prev_width,
            prev_height / curr_height,
            curr_height / prev_height,
        )
        if size_ratio > float(self._max_bbox_size_ratio):
            return False
        prev_center = (
            prev_left + (0.5 * prev_width),
            prev_top + (0.5 * prev_height),
        )
        curr_center = (
            curr_left + (0.5 * curr_width),
            curr_top + (0.5 * curr_height),
        )
        displacement = math.hypot(
            curr_center[0] - prev_center[0],
            curr_center[1] - prev_center[1],
        )
        # Normalize by both boxes so a plausible displacement scales with the
        # person's current image footprint rather than absolute resolution.
        normalizer = 0.5 * (
            math.hypot(prev_width, prev_height)
            + math.hypot(curr_width, curr_height)
        )
        if normalizer <= 1e-6:
            return False
        return (
            displacement / normalizer
            <= float(self._max_center_displacement_norm)
        )

    def _retired_is_reusable_locked(
        self,
        lifecycle: _ActiveTrackLifecycle,
        *,
        observed_at_us: int,
        media_pts_ns: int | None,
        bbox: tuple[float, float, float, float] | None,
    ) -> bool:
        gap_ns = self._reappearance_gap_ns(
            lifecycle,
            observed_at_us=int(observed_at_us),
            media_pts_ns=media_pts_ns,
        )
        return bool(
            gap_ns is not None
            and 0 < gap_ns <= int(self._reappearance_grace_ns)
            and self._bbox_is_compatible(lifecycle.bbox, bbox)
        )

    def _retire_lifecycle_locked(
        self,
        source_id: int,
        lifecycle: _ActiveTrackLifecycle,
    ) -> None:
        retired = self._retired_by_source.setdefault(int(source_id), OrderedDict())
        retired[int(lifecycle.tracker_id)] = lifecycle
        retired.move_to_end(int(lifecycle.tracker_id))
        while len(retired) > int(self._max_retired_per_source):
            retired.popitem(last=False)

    def retired_lifecycle_count(self, source_id: int) -> int:
        """Return the bounded scalar grace-state count for diagnostics/tests."""

        source = _exact_nonnegative_int(source_id, "source_id")
        with self._lock:
            return len(self._retired_by_source.get(source, ()))

    def peek_generation(
        self,
        source_id: int,
        tracker_id: int,
        *,
        frame_id: int | None = None,
        observed_at_us: int | None = None,
        media_pts_ns: object = None,
        bbox: object = None,
    ) -> int:
        """Return the generation that the next observation would receive.

        Without a frame context this is a read-only lookup.  With a frame
        context it also reserves a bounded scalar generation for that exact
        frame; ``update_frame`` consumes those reservations and remains the
        authority that stamps and commits the final rows.  This lets world
        augmentation use the same value for every new tracker in a frame.
        """

        source = _exact_nonnegative_int(source_id, "source_id")
        tracker = _exact_nonnegative_int(tracker_id, "tracker_id")
        with self._lock:
            active = self._active_by_source.get(source, {}).get(tracker)
            if active is not None:
                return int(active.generation)
            if frame_id is not None or observed_at_us is not None:
                frame = _exact_nonnegative_int(frame_id, "frame_id")
                observed = _exact_nonnegative_int(
                    observed_at_us, "observed_at_us"
                )
                if observed <= 0:
                    raise ValueError("observed_at_us must be positive")
                media = _media_pts_ns(media_pts_ns)
                self._prune_retired_locked(source, observed, media)
                preview = self._generation_preview_by_source.get(source)
                if preview is None or preview[:3] != (frame, observed, media):
                    preview = (frame, observed, media, {})
                    self._generation_preview_by_source[source] = preview
                generations = preview[3]
                reserved = generations.get(tracker)
                if reserved is None:
                    next_generation = self._next_generation_by_source.get(
                        source, 1
                    )
                    retired = self._retired_by_source.get(source, {}).get(tracker)
                    current_bbox = _track_bbox(bbox)
                    if retired is not None and self._retired_is_reusable_locked(
                        retired,
                        observed_at_us=observed,
                        media_pts_ns=media,
                        bbox=current_bbox,
                    ):
                        reserved = int(retired.generation)
                    else:
                        # Reused generations are below the source's next-new
                        # counter and must not consume a fresh reservation.
                        fresh_reservations = sum(
                            1
                            for value in generations.values()
                            if int(value) >= int(next_generation)
                        )
                        reserved = int(next_generation) + fresh_reservations
                    generations[tracker] = reserved
                return int(reserved)
            return int(self._next_generation_by_source.get(source, 1))

    def update_frame(
        self,
        *,
        source_id: int,
        camera_id: str,
        frame_id: int,
        observed_at_us: int,
        media_pts_ns: object = None,
        tracks: Sequence[MutableMapping[str, Any]],
    ) -> TrackingContinuityUpdate:
        source = _exact_nonnegative_int(source_id, "source_id")
        frame = _exact_nonnegative_int(frame_id, "frame_id")
        observed = _exact_nonnegative_int(observed_at_us, "observed_at_us")
        if observed <= 0:
            raise ValueError("observed_at_us must be positive")
        media = _media_pts_ns(media_pts_ns)
        camera = str(camera_id or "").strip()
        if not camera:
            raise ValueError("camera_id is required")

        with self._lock:
            previous = self._active_by_source.get(source, {})
            current_ids: list[int] = []
            indexed: dict[int, MutableMapping[str, Any]] = {}
            indexed_bbox: dict[
                int, tuple[float, float, float, float] | None
            ] = {}
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
                indexed_bbox[tracker] = _track_bbox(track.get("bbox"))

            previous_ids = set(previous)
            current_id_set = set(current_ids)
            self._prune_retired_locked(source, observed, media)
            for tracker in sorted(previous_ids - current_id_set):
                self._retire_lifecycle_locked(source, previous[tracker])
            next_generation = self._next_generation_by_source.get(source, 1)
            preview = self._generation_preview_by_source.get(source)
            preview_generations: dict[int, int] = {}
            if preview is not None and preview[:3] == (frame, observed, media):
                preview_generations = dict(preview[3])
            current: dict[int, _ActiveTrackLifecycle] = {}
            for tracker in current_ids:
                prior = previous.get(tracker)
                if prior is None:
                    generation = preview_generations.get(tracker)
                    if generation is None:
                        retired = self._retired_by_source.get(source, {}).get(
                            tracker
                        )
                        if retired is not None and self._retired_is_reusable_locked(
                            retired,
                            observed_at_us=observed,
                            media_pts_ns=media,
                            bbox=indexed_bbox[tracker],
                        ):
                            generation = int(retired.generation)
                        else:
                            generation = next_generation
                            next_generation += 1
                    else:
                        next_generation = max(
                            int(next_generation), int(generation) + 1
                        )
                    retired_for_source = self._retired_by_source.get(source)
                    if retired_for_source is not None:
                        retired_for_source.pop(tracker, None)
                        if not retired_for_source:
                            self._retired_by_source.pop(source, None)
                else:
                    generation = prior.generation
                    retired_for_source = self._retired_by_source.get(source)
                    if retired_for_source is not None:
                        retired_for_source.pop(tracker, None)
                indexed[tracker][TRACKER_LIFECYCLE_GENERATION_FIELD] = generation
                current[tracker] = _ActiveTrackLifecycle(
                    camera_id=camera,
                    tracker_id=tracker,
                    generation=generation,
                    last_seen_frame_id=frame,
                    last_seen_observed_at_us=observed,
                    last_seen_media_pts_ns=media,
                    bbox=indexed_bbox[tracker],
                )

            # This is a useful media-side preview for key-change/diagnostic
            # consumers, but it is deliberately not publication authority.
            # The worker recomputes the same set immediately before publish so
            # queued frames cannot compare against a later media-side state.
            tombstones = self._derive_tombstones_locked(
                self._published_by_source.get(source, {}),
                current,
                frame_id=frame,
                observed_at_us=observed,
            )
            self._active_by_source[source] = current
            self._next_generation_by_source[source] = next_generation
            self._generation_preview_by_source.pop(source, None)
            return TrackingContinuityUpdate(
                source_id=source,
                frame_id=frame,
                observed_at_us=observed,
                source_epoch=int(self._source_epoch_by_source.get(source, 0)),
                # Force publication only on the processed key transition.
                # Repeated empty frames must not stay forced while the worker
                # is draining an earlier disappearance receipt.
                tracker_keys_changed=previous_ids != current_id_set,
                tombstones=tombstones,
                active_lifecycles=tuple(
                    current[tracker] for tracker in sorted(current)
                ),
            )

    @staticmethod
    def _derive_tombstones_locked(
        published: Mapping[int, _ActiveTrackLifecycle],
        active_lifecycles: Mapping[int, _ActiveTrackLifecycle],
        *,
        frame_id: int,
        observed_at_us: int,
    ) -> tuple[dict[str, object], ...]:
        """Derive disappearance rows from the last *published* state.

        ``update_frame`` advances processed state on the media callback, while
        publication can be queued behind it. Comparing against processed
        presence makes an absent frame lose its tombstone if its preceding
        present frame has not published yet. This helper intentionally sees
        only the worker's current published lifecycle and the immutable target
        cohort.
        """

        tombstones: list[dict[str, object]] = []
        for tracker in sorted(published):
            prior = published[tracker]
            current = active_lifecycles.get(int(tracker))
            if current is not None and current.generation == prior.generation:
                continue
            tombstones.append(
                {
                    "camera_id": prior.camera_id,
                    "tracker_id": prior.tracker_id,
                    TRACKER_LIFECYCLE_GENERATION_FIELD: prior.generation,
                    "last_seen_frame_id": prior.last_seen_frame_id,
                    "last_seen_observed_at_us": prior.last_seen_observed_at_us,
                    "disappeared_at_frame_id": int(frame_id),
                    "disappeared_at_observed_at_us": int(observed_at_us),
                }
            )
        return tuple(tombstones)

    def prepare_publication(
        self,
        update: TrackingContinuityUpdate,
    ) -> TrackingContinuityUpdate:
        """Bind tombstones to the worker's current publication state.

        The media callback may prepare many exact cohorts before the worker
        publishes any of them. Therefore ``update.tombstones`` is only a
        media-time preview. This method must run immediately before the
        tracking publication and returns the receipt that
        ``mark_published`` validates.
        """

        if not isinstance(update, TrackingContinuityUpdate):
            raise ValueError("tracking publication requires its continuity update")
        with self._lock:
            current_epoch = int(self._source_epoch_by_source.get(update.source_id, 0))
            if int(update.source_epoch) > current_epoch:
                raise ValueError(
                    "tracking continuity receipt belongs to a future source epoch"
                )
            if int(update.source_epoch) < current_epoch:
                # A source reconnect clears the previous epoch's published
                # lifecycle. Older queued rows can still drain FIFO, but they
                # must not create a tombstone in the new epoch.
                tombstones: tuple[dict[str, object], ...] = ()
            else:
                active = {
                    int(item.tracker_id): item
                    for item in update.active_lifecycles
                }
                tombstones = self._derive_tombstones_locked(
                    self._published_by_source.get(update.source_id, {}),
                    active,
                    frame_id=int(update.frame_id),
                    observed_at_us=int(update.observed_at_us),
                )
            return replace(
                update,
                tombstones=tombstones,
                publication_prepared=True,
            )

    def mark_published(self, update: TrackingContinuityUpdate) -> None:
        """Commit the exact public presence represented by a successful publish."""

        if not isinstance(update, TrackingContinuityUpdate):
            raise ValueError("tracking publication requires its continuity update")
        # Keep direct registry callers source-compatible while ensuring the
        # canonical worker path supplies an explicit worker-prepared receipt.
        if not update.publication_prepared:
            update = self.prepare_publication(update)
        with self._lock:
            current_epoch = int(self._source_epoch_by_source.get(update.source_id, 0))
            if int(update.source_epoch) > current_epoch:
                raise ValueError(
                    "tracking continuity receipt belongs to a future source epoch"
                )
            if int(update.source_epoch) < current_epoch:
                return
            previous_frame = self._last_published_frame_by_source.get(update.source_id)
            if previous_frame is not None and update.frame_id <= previous_frame:
                raise ValueError("published source frame must advance monotonically")
            active = {
                int(item.tracker_id): item
                for item in update.active_lifecycles
            }
            expected_tombstones = self._derive_tombstones_locked(
                self._published_by_source.get(update.source_id, {}),
                active,
                frame_id=int(update.frame_id),
                observed_at_us=int(update.observed_at_us),
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
            for lifecycle in update.active_lifecycles:
                published[lifecycle.tracker_id] = lifecycle
            self._published_by_source[update.source_id] = published
            self._last_published_frame_by_source[update.source_id] = update.frame_id


__all__ = [
    "TRACKER_LIFECYCLE_GENERATION_FIELD",
    "TRACKING_CONTINUITY_CONTRACT",
    "TRACKING_CONTINUITY_CONTRACT_VERSION",
    "TrackingContinuityUpdate",
    "TrackingLifecycleRegistry",
    "pair_safe_publication_interval_s",
]
