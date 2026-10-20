"""Private, SDK-neutral compact acquisition state. No identity or World authority."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, asdict, fields, field
import hashlib
import math
from collections.abc import Mapping
import threading
import time

CAMERAS = ("living-room", "kitchen", "family-room")
MAX_CLOCK = (1 << 64) - 1
MAX_BATCH_FRAMES = 3


class AcquisitionUnavailable(ValueError):
    pass


def _integer(value, field):
    if type(value) is not int or not 0 <= value < MAX_CLOCK:
        raise AcquisitionUnavailable(f"invalid_{field}")
    return value


def current_frame_key(frame):
    """SDK-neutral synthetic adapter. Runtime injects native verified getter API."""
    key = tuple(getattr(frame, name, None) for name in
                ("batch_id", "pad_index", "source_id", "frame_number", "buf_pts", "ntp_timestamp"))
    for value in key:
        _integer(value, "current_frame_key")
    if key[0] >= MAX_BATCH_FRAMES or key[1] >= len(CAMERAS):
        raise AcquisitionUnavailable("current_frame_bounds")
    return key


class ScopedFrameReceipts:
    """Callback-local exact frame lookup. Never store this object in queued work.

    Native extraction scope is fresh each call; original buffer scope was stamped
    on the SDK-owned frame. Structurally bad rows deny only their slot. Duplicate
    slots deny all rows for that slot. No source-keyed fallback is permitted.
    """
    def __init__(self, batch, payload, *, frame_key=current_frame_key):
        self._batch = batch
        self._frame_key = frame_key
        self.rows, self.errors = {}, {}
        if not isinstance(payload, Mapping):
            raise AcquisitionUnavailable("private_batch_unavailable")
        self.scope = _integer(payload.get("scope"), "extraction_scope")
        self.original_scope = payload.get("original_buffer_scope")
        if self.original_scope is not None:
            _integer(self.original_scope, "current_original_buffer_scope")
        rows = payload.get("rows")
        if not isinstance(rows, (list, tuple)) or len(rows) > MAX_BATCH_FRAMES:
            raise AcquisitionUnavailable("private_batch_bounds")
        for row in rows:
            if not isinstance(row, Mapping):
                raise AcquisitionUnavailable("private_row_scope_unavailable")
            slot = row.get("slot")
            if type(slot) is not int or not 0 <= slot < MAX_BATCH_FRAMES:
                raise AcquisitionUnavailable("private_slot_bounds")
            if slot in self.rows or slot in self.errors:
                self.rows.pop(slot, None)
                self.errors[slot] = "duplicate_private_slot"
                continue
            try:
                copied = dict(row)
                if "receipt_valid" in copied and self.original_scope is None:
                    raise AcquisitionUnavailable("private_current_original_buffer_scope_absent")
                for field in ("source_id", "sdk_source_id", "post_tracker_frame",
                              "post_tracker_pts_ns", "post_tracker_ntp_ns", "scope"):
                    _integer(copied.get(field), field)
                if copied["scope"] != self.scope or copied["source_id"] >= len(CAMERAS):
                    raise AcquisitionUnavailable("private_scope_source_mismatch")
                self.rows[slot] = copied
            except AcquisitionUnavailable as exc:
                self.errors[slot] = str(exc)

    def lookup(self, batch, frame):
        if batch is not self._batch:
            raise AcquisitionUnavailable("private_current_buffer_scope_mismatch")
        return self.lookup_key(batch, self._frame_key(frame))

    def lookup_key(self, batch, key):
        """Lookup by already owned scalars, never a retained iterator view."""
        if batch is not self._batch:
            raise AcquisitionUnavailable("private_current_buffer_scope_mismatch")
        if type(key) is not tuple or len(key) != 6:
            raise AcquisitionUnavailable("current_native_frame_identity_shape")
        slot, source, sdk_source, number, pts, ntp = key
        for value in (slot, source, sdk_source, number, pts, ntp):
            _integer(value, "current_native_frame_identity")
        if slot >= MAX_BATCH_FRAMES or source >= len(CAMERAS):
            raise AcquisitionUnavailable("current_native_frame_identity_bounds")
        if slot in self.errors:
            raise AcquisitionUnavailable(self.errors[slot])
        row = self.rows.get(slot)
        if row is None:
            raise AcquisitionUnavailable("exact_frame_receipt_absent")
        if (row["source_id"], row["sdk_source_id"], row["post_tracker_frame"],
            row["post_tracker_pts_ns"], row["post_tracker_ntp_ns"]) != (source, sdk_source, number, pts, ntp):
            raise AcquisitionUnavailable("exact_current_frame_mismatch")
        if self.original_scope is not None and row.get("original_buffer_scope") != self.original_scope:
            raise AcquisitionUnavailable("exact_original_current_buffer_scope_mismatch")
        return dict(row)


def ordered_current_frames(frames, *, frame_key=current_frame_key):
    """Return ONLY owned scalar keys, read before advancing borrowed views.

    SDK AbstractIterator owns/reuses its metadata view. Never materialize the
    wrappers, copy them by an unverified API, or dereference after exhaustion.
    Sort reordered enumeration only when frame counters AND PTS agree.
    An intra-buffer epoch rewind has conflicting order and is integrity failure,
    not a sequence which may be sorted into an invented monotonic clock.
    """
    keys = []
    for frame in frames:
        if len(keys) >= MAX_BATCH_FRAMES:
            raise AcquisitionUnavailable("current_batch_frame_bounds")
        key = frame_key(frame)  # view and owning iterator are still current
        if type(key) is not tuple or len(key) != 6:
            raise AcquisitionUnavailable("current_native_frame_identity_shape")
        for value in key:
            _integer(value, "current_native_frame_identity")
        if key[0] >= MAX_BATCH_FRAMES or key[1] >= len(CAMERAS):
            raise AcquisitionUnavailable("current_native_frame_identity_bounds")
        keys.append(key)
        del frame
    if not keys: raise AcquisitionUnavailable("current_batch_frame_bounds")
    if len({key[0] for key in keys}) != len(keys):
        raise AcquisitionUnavailable("duplicate_current_slot")
    grouped = {}
    for key in keys:
        grouped.setdefault(key[1], []).append(key)
    ordered = []
    for group in grouped.values():
        group.sort(key=lambda key: key[3])
        for old, new in zip(group, group[1:]):
            if new[3] <= old[3] or new[4] <= old[4]:
                raise AcquisitionUnavailable("ambiguous_intrabuffer_epoch_or_clock_order")
        ordered.extend(group)
    return tuple(ordered)


def consume_ordered_current_frames(batch, keys, consume, *, frame_key=current_frame_key):
    """Visit exact views INSIDE their iterator lifetime; no wrapper escapes.

    Each frame_items access initiates a fresh callback-local SDK iterator.
    At most three keys/three frames => nine scalar comparisons, no SDK copy.
    The original batch remains callback-owned throughout validation/consumption.
    """
    if type(keys) is not tuple or not 0 < len(keys) <= MAX_BATCH_FRAMES:
        raise AcquisitionUnavailable("current_batch_frame_bounds")
    for expected in keys:
        iterator = iter(batch.frame_items)
        found = False
        try:
            for index, frame in enumerate(iterator):
                if index >= MAX_BATCH_FRAMES:
                    raise AcquisitionUnavailable("current_batch_frame_bounds")
                if frame_key(frame) == expected:
                    consume(frame, expected)  # before next()/iterator destruction
                    found = True
                    del frame
                    break
                del frame
        finally:
            del iterator
        if not found: raise AcquisitionUnavailable("exact_current_frame_disappeared")


@dataclass(frozen=True)
class OriginalCohort:
    run_id: str
    graph_id: str
    camera_id: str
    source_id: int
    source_clock_epoch: int
    original_frame: int
    pts_ns: int
    ntp_ns: int
    timestamp_origin: str
    buffer_scope: int = 0
    slot: int = 0
    sdk_source_id: int = 0


@dataclass(frozen=True)
class OwnerRetentionReceipt:
    """Private SAME canonical-cohort scalar snapshot, not identity/World authority."""
    run_id: str
    graph_id: str
    source_id: int
    source_epoch: int
    clock_epoch: int
    frame_id: int
    observed_at_us: int
    media_pts_ns: int | None
    owner_grace_ns: int
    keys: tuple  # public owner (source, epoch, raw, generation) keys
    active: tuple  # raw, generation, last ACTUAL frame/observed_us/media_pts_ns
    shadow: tuple
    terminated: tuple


@dataclass(frozen=True, slots=True)
class OwnerSourceAvailability:
    """SAME prequeue owner/progress snapshot; never a fresh member observation."""
    original: OriginalCohort
    owner: OwnerRetentionReceipt
    ingress_monotonic_s: float
    checked_monotonic_s: float
    last_progress_age_s: float
    stall_seconds: float
    status: str


def freeze_owner_availability(private, continuity, monitor, *, ingress_monotonic_s):
    """Query the existing own-source monitor ONCE before throttle, no I/O/SDK."""
    from noesis_core.source_progress import DecodedSourceProgressMonitor
    try:
        if (type(monitor) is not DecodedSourceProgressMonitor or private is None
                or private.denial is not None or type(private.original) is not OriginalCohort
                or type(ingress_monotonic_s) not in (int, float)
                or not math.isfinite(ingress_monotonic_s) or ingress_monotonic_s < 0):
            return None
        original, owner = private.original, private.owner_retention
        owner_retention_members(owner, original, continuity.source_epoch, continuity.frame_id)
        if (continuity.source_id, continuity.observed_at_us, continuity.media_pts_ns) != (
                original.source_id, owner.observed_at_us, original.pts_ns):
            return None
        if any(len(v) > 64 for v in (owner.keys, owner.active, owner.shadow, owner.terminated)):
            return None
        snapshot = monitor.snapshot(now=ingress_monotonic_s)
        row = snapshot['sources'][str(original.source_id)]
        age, stall = row['last_progress_age_seconds'], snapshot['policy']['stall_seconds']
        if (snapshot['started'] is not True or snapshot['fatal'] is not None
                or row['camera_id'] != original.camera_id or row['frames'] <= 0
                or row['status'] not in {'running', 'recovering'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (age, stall))
                or stall <= 0):
            return None
        status = row['status']
        return OwnerSourceAvailability(original, owner, float(ingress_monotonic_s),
            float(ingress_monotonic_s), float(age), float(stall), status)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None  # Optional custody unavailable, never an invented source stop.


def freeze_owner_retention(update, keys, bound, original):
    """Called under the adapter source lock immediately after owner update/query."""
    from noesis_core.tracking_continuity import TrackingContinuityUpdate
    if (type(update) is not TrackingContinuityUpdate or type(original) is not OriginalCohort
            or type(keys) is not frozenset or type(bound) is not int or bound < 0):
        raise AcquisitionUnavailable("exact_owner_retention_snapshot_required")
    receipt = OwnerRetentionReceipt(original.run_id, original.graph_id, update.source_id,
        update.source_epoch, original.source_clock_epoch, update.frame_id, update.observed_at_us, update.media_pts_ns, bound,
        tuple(sorted(keys)), tuple((item.tracker_id, item.generation, item.last_seen_frame_id,
            item.last_seen_observed_at_us, item.last_seen_media_pts_ns) for item in update.active_lifecycles),
        tuple(update.shadow_lifecycles), tuple(update.terminated_tracker_ids))
    owner_retention_members(receipt, original, update.source_epoch, update.frame_id)
    return receipt


def owner_retention_members(receipt, original, source_epoch, canonical_frame):
    """Validate scope/classes; retired eligibility is ONLY the frozen public query."""
    if (type(receipt) is not OwnerRetentionReceipt or type(original) is not OriginalCohort
            or (receipt.run_id, receipt.graph_id, receipt.source_id, receipt.source_epoch,
                receipt.clock_epoch, receipt.frame_id, receipt.media_pts_ns) !=
               (original.run_id, original.graph_id, original.source_id, source_epoch,
                original.source_clock_epoch, canonical_frame, original.pts_ns)
            or type(receipt.owner_grace_ns) is not int or receipt.owner_grace_ns < 0
            or any(type(value) is not int or value < 0 for value in (
                receipt.source_id, receipt.source_epoch, receipt.clock_epoch,
                receipt.frame_id, receipt.media_pts_ns))
            or type(receipt.observed_at_us) is not int or receipt.observed_at_us <= 0
            or any(type(rows) is not tuple or len(rows) > 256
                   for rows in (receipt.keys, receipt.active, receipt.shadow, receipt.terminated))):
        raise AcquisitionUnavailable("same_cohort_owner_retention_unavailable")
    retained = set()
    for key in receipt.keys:
        if (type(key) is not tuple or len(key) != 4 or any(type(v) is not int or v < 0 for v in key)
                or key[:2] != (receipt.source_id, source_epoch) or key in retained):
            raise AcquisitionUnavailable("owner_retention_key_scope_or_duplicate")
        retained.add(key)
    active = {}
    for row in receipt.active:
        if (type(row) is not tuple or len(row) != 5 or
                any(type(v) is not int or v < 0 for v in row[:4]) or
                row[4] is not None and (type(row[4]) is not int or row[4] < 0) or
                row[:2] in active or row[2] > canonical_frame or row[3] > receipt.observed_at_us or
                row[4] is not None and row[4] > receipt.media_pts_ns):
            raise AcquisitionUnavailable("owner_retention_actual_row_invalid")
        active[row[:2]] = row
        if (receipt.source_id, source_epoch, *row[:2]) not in retained:
            raise AcquisitionUnavailable("owner_retention_active_key_missing")
    shadow = set()
    for key in receipt.shadow:
        if (type(key) is not tuple or len(key) != 2 or any(type(v) is not int or v < 0 for v in key)
                or key not in active or key in shadow):
            raise AcquisitionUnavailable("owner_retention_unknown_or_duplicate_shadow")
        shadow.add(key)
    if any(type(raw) is not int or raw < 0 for raw in receipt.terminated) or len(set(receipt.terminated)) != len(receipt.terminated):
        raise AcquisitionUnavailable("owner_retention_termination_invalid")
    if any(key[2] in receipt.terminated for key in retained):
        raise AcquisitionUnavailable("owner_retention_termination_wins")
    for key, row in active.items():
        if key not in shadow and (row[2] != canonical_frame or row[3] != receipt.observed_at_us or row[4] != receipt.media_pts_ns):
            raise AcquisitionUnavailable("owner_retention_measurement_not_current")
    return frozenset(key[1:] for key in retained), frozenset((source_epoch, *key) for key in active if key not in shadow)


@dataclass(frozen=True)
class PrivateAssociationWork:
    """Private diagnostic evidence only; NOT the accepted Identity DTO."""
    source_id: int
    canonical_frame: int
    media_pts_ns: int
    clock_epoch: int
    original: OriginalCohort | None
    denial: str | None
    terminated: tuple[int, ...] = ()
    owner_retention: OwnerRetentionReceipt | None = None


@dataclass(frozen=True)
class OriginalSwinExtraction:
    """Private compact copy from proven selected synchronous fresh ingress."""
    original: OriginalCohort
    raw_id: int
    profile: object  # actual accepted AppearanceProfile, not a parallel DTO
    vector: tuple[float, ...] = field(repr=False)
    ingress_monotonic_s: float | None = None  # availability, NOT acquisition time

    @property
    def key(self):
        return (self.original.buffer_scope, self.original.slot, self.raw_id)


@dataclass(frozen=True)
class SwinOwnerBinding:
    member: tuple  # actual owner (source_epoch, raw_id, lifecycle_generation)
    original_birth: OriginalCohort
    lifecycle_started_frame: int  # ORIGINAL owner canonical first-admitted frame


@dataclass(frozen=True)
class CachedSwinUse:
    appearance: object  # actual frozen AppearanceEvidence
    vector: tuple[float, ...] = field(repr=False)
    fresh_extraction: bool = False
    appearance_ingress_monotonic_s: float | None = None


class OriginalSwinCache:
    """Private <=256 pending receipts AND <=256 members per source; no SDK state.

    Media acquires the lock nonblocking; exhaustion degrades provenance only.
    The FIFO owner alone binds/reset/removes canonical lifecycle ownership.
    Queued work uses frozen receipts/evidence, never this member map as last-seen.
    """
    def __init__(self, *, coverage_diagnostics=False, run_id=None, graph_id=None):
        self.pending = [{} for _ in CAMERAS]
        self.frozen_extractions = [{} for _ in CAMERAS]  # <=256 genuine queued references/source
        self.members = [{} for _ in CAMERAS]
        self.owners = [{} for _ in CAMERAS]
        self.holdovers = deque(maxlen=64)
        self.counters = Counter()
        self.coverage_diagnostics = bool(coverage_diagnostics)
        self.birth_coverage_counters = Counter()
        self.birth_coverage_samples = {}  # <=64 aggregate scalar events, no authority
        self.owner_watermarks = [None] * 3
        self.producer_counts = Counter()
        self.producer_examples = {}  # <=64 scalar examples, never vectors/authority.
        self.run_id, self.graph_id = run_id, graph_id
        self._lock = threading.Lock()

    def producer_event(self, event, original, *, current_frame=None, **details):
        key = (original.source_id, event)
        self.producer_counts[f"source_{key[0]}.{event}"] += 1
        if key not in self.producer_examples and len(self.producer_examples) < 64:
            self.producer_examples[key] = dict(source_id=original.source_id,event=event,
                original_frame=original.original_frame,original_pts_ns=original.pts_ns,
                scope=original.buffer_scope,slot=original.slot,current_frame=current_frame,**details)

    def producer_summary(self):
        with self._lock:
            return dict(counts=dict(self.producer_counts),examples=tuple(self.producer_examples.values()))

    def _birth_coverage(self, event, reason, original, key, birth, canonical_frame,
                        owner=None, status=None):
        if not self.coverage_diagnostics: return
        source = original.source_id
        pending = self.pending[source].get((original.buffer_scope, original.slot, key[1]))
        fresh = pending is not None and pending.original == original
        counts = self.birth_coverage_counters
        counts["events"] += 1
        counts[f"source_{source}.{event}.{reason}"] += 1
        index = (event, source, key, birth)
        sample = self.birth_coverage_samples.get(index)
        current = asdict(original)  # only this known frozen scalar dataclass
        if sample is None:
            if len(self.birth_coverage_samples) >= 64:
                self.birth_coverage_samples.pop(next(iter(self.birth_coverage_samples)))
                counts["sample_evictions"] += 1
            sample = {"event": event, "reason": reason, "source_id": source,
                "source_epoch": key[0], "raw_id": key[1], "lifecycle_generation": key[2],
                "original_birth": asdict(birth), "first_current_original": current,
                "first_canonical_frame": canonical_frame, "first_sequence": counts["events"],
                "cache_lifecycle_started_frame": owner.lifecycle_started_frame if owner is not None else None,
                "occurrences": 0, "current_selected_extraction_occurrences": 0}
            self.birth_coverage_samples[index] = sample
        sample.update(last_current_original=current, last_canonical_frame=canonical_frame,
            last_sequence=counts["events"], owner_binding_present=owner is not None and event != "owner_withdrawn",
            owner_result_status=status[:128] if type(status) is str else None,
            current_selected_extraction_present=fresh,
            original_birth_mismatch_fields=[field.name for field in fields(OriginalCohort)
                if getattr(birth, field.name) != getattr(original, field.name)])
        sample["occurrences"] += 1
        sample["current_selected_extraction_occurrences"] += int(fresh)

    def record(self, value):
        from reid.mv3dt_person_association import AppearanceProfile
        if (type(value) is not OriginalSwinExtraction or type(value.original) is not OriginalCohort
                or type(value.profile) is not AppearanceProfile or value.profile.dimensions != 256
                or type(value.vector) is not tuple or len(value.vector) != 256
                or not all(type(item) is float and math.isfinite(item) for item in value.vector)
                or not any(value.vector) or type(value.raw_id) is not int or value.raw_id < 0
                or value.ingress_monotonic_s is not None and (
                    type(value.ingress_monotonic_s) not in (int, float)
                    or not math.isfinite(value.ingress_monotonic_s) or value.ingress_monotonic_s < 0)):
            raise AcquisitionUnavailable("bounded_selected_swin_copy_required")
        source = value.original.source_id
        if not 0 <= source < 3: raise AcquisitionUnavailable("swin_source_bounds")
        if not self._lock.acquire(blocking=False):
            self.counters["media_lock_unavailable"] += 1; return False
        try:
            rows = self.pending[source]
            if value.key in rows:
                self.counters["duplicate_extraction"] += 1; return False
            if len(rows) >= 256:
                self.counters["pending_capacity"] += 1; return False
            rows[value.key] = value
            self.counters["fresh_extractions"] += 1
            self.producer_event('fresh_F',value.original,raw_id=value.raw_id,
                original_ingress_monotonic_s=value.ingress_monotonic_s,profile=value.profile.fingerprint)
            if self.coverage_diagnostics:
                self.birth_coverage_counters[f"source_{source}.fresh_extractions"] += 1
            return True
        finally: self._lock.release()

    def exact(self, original, raw):
        if original is None or not self._lock.acquire(blocking=False): return None
        try:
            value = self.pending[original.source_id].get((original.buffer_scope, original.slot, raw))
            if value is not None and value.original == original: return value
            # Same CURRENT frame after FIFO binding; never a cached prior frame.
            return next((item[0] for item in self.members[original.source_id].values()
                if item[0].key == (original.buffer_scope, original.slot, raw) and item[0].original == original), None)
        finally: self._lock.release()

    def freeze_exact(self,original,raw):
        if original is None or not self._lock.acquire(blocking=False):return None
        try:
            value=self.pending[original.source_id].get((original.buffer_scope,original.slot,raw))
            frozen=self.frozen_extractions[original.source_id]
            if value is None or value.original!=original:return None
            if value.key not in frozen and len(frozen)>=256:
                self.counters['frozen_capacity']+=1
                return None
            frozen[value.key]=value
            return value
        finally:self._lock.release()

    def finish_frozen(self,current):
        with self._lock:
            rows=self.frozen_extractions[current.source_id]
            now=time.monotonic()
            for key,value in tuple(rows.items()):
                if (value.original.run_id,value.original.graph_id)!=(current.run_id,current.graph_id) or (
                        value.original.source_clock_epoch,value.original.original_frame)<= (
                        current.source_clock_epoch,current.original_frame) or (
                        value.ingress_monotonic_s is not None and now-value.ingress_monotonic_s>.35):rows.pop(key)

    def bind_owner(self, original, source_epoch, canonical_frame, members, *, owner_retention=None,
                   diagnostic_status=None, diagnostic_terminated=()):
        """Called only from the actual successful FIFO owner receipt/barrier."""
        source = original.source_id
        if len(members) > 256: raise AcquisitionUnavailable("swin_owner_capacity")
        keys = set()
        with self._lock:
            owners, bound = self.owners[source], self.members[source]
            if self.run_id is not None and (original.run_id, original.graph_id) != (self.run_id, self.graph_id):
                owners.clear(); bound.clear(); self._prune_pending(source, original)
                raise AcquisitionUnavailable("swin_owner_run_graph_barrier")
            previous = self.owner_watermarks[source]
            order = (source_epoch, original.source_clock_epoch, canonical_frame, original.pts_ns)
            if previous is not None and (order[:2] < previous[:2] or
                    order[:2] == previous[:2] and (order[2] <= previous[2] or order[3] <= previous[3])):
                raise AcquisitionUnavailable("swin_owner_fifo_order_invalid")
            try:
                retained, measured = owner_retention_members(owner_retention, original, source_epoch, canonical_frame)
            except AcquisitionUnavailable:
                owners.clear(); bound.clear()
                self._prune_pending(source, original)
                self.counters["owner_retention_unavailable"] += 1
                raise
            self.owner_watermarks[source] = order
            if self.run_id is None: self.run_id, self.graph_id = original.run_id, original.graph_id
            # Validate every current member before changing any binding.
            for key, birth in members:
                if (type(key) is not tuple or len(key) != 3 or any(type(v) is not int or v < 0 for v in key)
                        or key[0] != source_epoch or key not in measured or type(birth) is not OriginalCohort
                        or key in keys
                        or (birth.run_id, birth.graph_id, birth.source_id, birth.source_clock_epoch) !=
                        (original.run_id, original.graph_id, source, original.source_clock_epoch)):
                    owners.clear(); bound.clear(); self._prune_pending(source, original)
                    raise AcquisitionUnavailable("exact_current_measured_owner_birth_required")
                old = owners.get(key)
                if (old is not None and
                        (old.original_birth.run_id, old.original_birth.graph_id, old.original_birth.source_clock_epoch) ==
                        (original.run_id, original.graph_id, original.source_clock_epoch)
                        and old.original_birth != birth):
                    owners.clear(); bound.clear(); self._prune_pending(source, original)
                    raise AcquisitionUnavailable("swin_original_birth_changed")
                keys.add(key)
            for key in tuple(owners):
                birth = owners[key].original_birth
                if (key not in retained or key[0] != source_epoch or key[1] in owner_retention.terminated or
                        (birth.run_id, birth.graph_id, birth.source_clock_epoch) !=
                        (original.run_id, original.graph_id, original.source_clock_epoch)):
                    removed = owners.pop(key); bound.pop(key, None)
                    self.counters["owner_withdrawals"] += 1
                    self.counters["lineage_hard_prunes"] += 1
                    self._prune_pending(source, original, {key[1]}, preserve_current=key[1] in {item[1] for item in measured})
                    if self.coverage_diagnostics:
                        reason = ("source_epoch_changed" if key[0] != source_epoch else
                                  "original_termination_barrier" if key[1] in owner_retention.terminated else
                                  "canonical_retention_or_run_clock_prune")
                        self._birth_coverage("owner_withdrawn", reason, original, key, removed.original_birth,
                                             canonical_frame, removed, diagnostic_status)
                elif key not in keys:
                    if bound.pop(key, None) is not None:
                        self.counters["appearance_withdrawals"] += 1
                    self.counters["lineage_retained_without_appearance"] += 1
                    self._birth_coverage("lineage_retained", "current_shadow_or_retired_owner", original,
                                         key, birth, canonical_frame, owners[key], diagnostic_status)
            for key, birth in members:
                if (type(key) is not tuple or len(key) != 3 or key[0] != source_epoch
                        or type(birth) is not OriginalCohort or birth.run_id != original.run_id
                        or birth.source_id != source):
                    raise AcquisitionUnavailable("exact_swin_owner_birth_required")
                if key not in measured:
                    raise AcquisitionUnavailable("current_measured_owner_member_required")
                owner = owners.get(key)
                if owner is None:
                    if birth != original:
                        self.counters["original_canonical_birth_unavailable"] += 1
                        self._birth_coverage("birth_refused", "owner_absent_birth_not_current", original,
                                             key, birth, canonical_frame, status=diagnostic_status)
                        continue
                    owner = SwinOwnerBinding(key, birth, canonical_frame)
                    owners[key] = owner
                    self._birth_coverage("owner_created", "genuine_current_original_birth", original,
                                         key, birth, canonical_frame, owner, diagnostic_status)
                elif owner.original_birth != birth:
                    self._birth_coverage("birth_conflict", "existing_owner_birth_changed", original,
                                         key, birth, canonical_frame, owner, diagnostic_status)
                    raise AcquisitionUnavailable("swin_original_birth_changed")
                extraction = self.pending[source].pop((original.buffer_scope, original.slot, key[1]), None)
                if extraction is not None and extraction.original == original:
                    bound[key] = (extraction, owner, canonical_frame, None)
                    if self.coverage_diagnostics:
                        self.birth_coverage_counters[f"source_{source}.owned_extractions"] += 1
                elif key in bound and canonical_frame > bound[key][2]:
                    # Later ACTUAL FIFO-owned frame, SAME lifecycle, no fresh
                    # selected tensor for this frame. Observe retention only;
                    # do not renew the original extraction/stamp or grant use.
                    retained, retained_owner, retained_frame, _appearance = bound[key]
                    if retained_owner != owner:
                        raise AcquisitionUnavailable("swin_holdover_owner_changed")
                    self.counters["fifo_owned_holdovers"] += 1
                    if self.coverage_diagnostics:
                        self.birth_coverage_counters[f"source_{source}.fifo_owned_holdovers"] += 1
                    self.holdovers.append({"source_id": source, "source_epoch": key[0],
                        "raw_id": key[1], "lifecycle_generation": key[2],
                        "birth_frame": owner.original_birth.original_frame,
                        "lifecycle_started_frame": owner.lifecycle_started_frame,
                        "extraction_frame": retained.original.original_frame,
                        "extraction_pts_ns": retained.original.pts_ns,
                        "extraction_ntp_ns": retained.original.ntp_ns,
                        "extraction_scope": retained.original.buffer_scope,
                        "extraction_slot": retained.original.slot,
                        "extraction_canonical_frame": retained_frame,
                        "later_frame": original.original_frame, "later_pts_ns": original.pts_ns,
                        "later_ntp_ns": original.ntp_ns, "later_scope": original.buffer_scope,
                        "later_slot": original.slot, "later_canonical_frame": canonical_frame,
                        "profile": retained.profile.fingerprint})
            # No past/exact unbound extraction may survive this ordered barrier.
            # Future detached media receipts remain available to their own FIFO work.
            self._prune_pending(source, original)

    def _prune_pending(self, source, original, raws=None, *, preserve_current=False):
        """Do not erase future exact receipts while older FIFO work drains."""
        for key, value in tuple(self.pending[source].items()):
            stamp = value.original
            old = (stamp.run_id, stamp.graph_id) != (original.run_id, original.graph_id) or (
                stamp.source_clock_epoch < original.source_clock_epoch or
                stamp.source_clock_epoch == original.source_clock_epoch and stamp.original_frame <= original.original_frame)
            if old and (raws is None or value.raw_id in raws) and not (preserve_current and stamp == original):
                self.producer_event('pruned_unbound_F',stamp,current_frame=original.original_frame,raw_id=value.raw_id)
                self.pending[source].pop(key)

    def bind_retained_original(self, extraction, observation, current, owner, profile, now):
        """Sole FIFO actor, after proven original lifecycle extent; never renew F."""
        from reid.mv3dt_person_association import SourceObservation
        if type(extraction) is not OriginalSwinExtraction or type(observation) is not SourceObservation:
            raise AcquisitionUnavailable('swin_retained_type')
        f,stamp=extraction.original,observation.stamp
        member=stamp.member
        key=(member.source_epoch,member.raw_id,member.lifecycle_generation)
        with self._lock:
            if (self.owners[current.source_id].get(key) is not owner or type(owner) is not SwinOwnerBinding
                    or self.frozen_extractions[current.source_id].get(extraction.key) is not extraction
                    or self.owner_watermarks[current.source_id] is None
                    or not stamp.frame_id<self.owner_watermarks[current.source_id][2]
                    or (f.run_id,f.graph_id,f.source_id,f.camera_id,f.source_clock_epoch) !=
                       (current.run_id,current.graph_id,current.source_id,current.camera_id,current.source_clock_epoch)
                    or (member.run_id,member.camera_id,member.source_id)!=(f.run_id,f.camera_id,f.source_id)
                    or extraction.raw_id!=member.raw_id or extraction.profile!=profile
                    or owner.lifecycle_started_frame>stamp.frame_id
                    or owner.original_birth.original_frame>f.original_frame
                    or stamp.lifecycle_started_frame!=owner.lifecycle_started_frame
                    or stamp.media_time_s!=f.pts_ns/1e9
                    or not 0<=current.pts_ns/1e9-stamp.acquired_at_s<=.35
                    or type(extraction.ingress_monotonic_s) not in (int,float)
                    or not 0<=now-extraction.ingress_monotonic_s<=.35):
                raise AcquisitionUnavailable('swin_retained_binding_or_expiry')
            old=self.members[current.source_id].get(key)
            if old is not None and old[2]>=stamp.frame_id:
                if old[2]==stamp.frame_id and old[0]!=extraction:
                    raise AcquisitionUnavailable('swin_retained_duplicate_changed')
                self.producer_event('retained_duplicate_or_superseded',f,current_frame=current.original_frame)
                return False
            self.members[current.source_id][key]=(extraction,owner,stamp.frame_id,None)
        self.bind_observation(f,observation)
        self.producer_event('bound_original_F',f,current_frame=current.original_frame,
            raw_id=member.raw_id,source_epoch=member.source_epoch,generation=member.lifecycle_generation,
            lifecycle_started_frame=owner.lifecycle_started_frame,original_ingress_monotonic_s=extraction.ingress_monotonic_s,
            evaluation_monotonic_s=now,source_age_s=(current.pts_ns-f.pts_ns)/1e9,
            availability_age_s=now-extraction.ingress_monotonic_s,profile=extraction.profile.fingerprint)
        return True

    def bind_observation(self, original, observation):
        """Actual accepted observation supplies truthful clock; never manufacture it."""
        from reid.mv3dt_person_association import (SourceObservation, AppearanceEvidence,
                                                 appearance_extraction_identity)
        if type(observation) is not SourceObservation:
            raise AcquisitionUnavailable("actual_source_observation_required")
        member, stamp = observation.stamp.member, observation.stamp
        key = (member.source_epoch, member.raw_id, member.lifecycle_generation)
        with self._lock:
            row = self.members[member.source_id].get(key)
            if row is None: raise AcquisitionUnavailable("original_swin_owner_binding_unavailable")
            extraction, owner, canonical_frame, old = row
            if (extraction.original != original or member.run_id != original.run_id
                    or member.camera_id != original.camera_id or stamp.frame_id != canonical_frame
                    or stamp.lifecycle_started_frame != owner.lifecycle_started_frame
                    or stamp.media_time_s != original.pts_ns / 1e9):
                raise AcquisitionUnavailable("exact_original_swin_observation_mismatch")
            identity = appearance_extraction_identity(stamp, extraction.profile)
            appearance = AppearanceEvidence(stamp, identity, extraction.profile.model_id,
                stamp.acquired_at_s, stamp.clock_domain, identity, extraction.profile)
            if old is not None and old != appearance:
                raise AcquisitionUnavailable("original_swin_stamp_retagged")
            self.members[member.source_id][key] = (extraction, owner, canonical_frame, appearance)
            return appearance

    def use(self, observation, at):
        from reid.mv3dt_person_association import SourceObservation, AcquisitionTime
        if type(observation) is not SourceObservation or type(at) is not AcquisitionTime: return None
        member = observation.stamp.member
        key = (member.source_epoch, member.raw_id, member.lifecycle_generation)
        with self._lock:
            row = self.members[member.source_id].get(key)
            if row is None:
                self.counters['use_owner_extraction_unavailable'] += 1
                return None
            if row[3] is None:
                self.counters['use_original_appearance_stamp_unavailable'] += 1
                return None
            extraction, owner, _frame, appearance = row
            if (appearance.stamp.member != member or observation.stamp.lifecycle_started_frame != owner.lifecycle_started_frame
                    or at.clock_domain != appearance.clock_domain or at.clock_basis != appearance.stamp.clock_basis
                    or not 0 <= at.at_s - appearance.acquired_at_s <= .35):
                self.counters["unavailable_or_expired"] += 1
                self.counters['use_binding_unavailable' if (
                    appearance.stamp.member != member or observation.stamp.lifecycle_started_frame != owner.lifecycle_started_frame
                    or at.clock_domain != appearance.clock_domain or at.clock_basis != appearance.stamp.clock_basis)
                    else 'use_original_source_time_expired'] += 1
                return None
            return CachedSwinUse(appearance, extraction.vector, observation.stamp == appearance.stamp,
                                 extraction.ingress_monotonic_s)


def selected_swin_profile(ini, *, source_ini=None, engine_path=None, model_config=None,
                          runtime_properties=None, pipeline_config=None):
    """Validate ORIGINAL source plus exact normal engine-only derivation.

    The canonical source validator deliberately requires its ONNX reference.
    Runtime receives ONLY the unchanged public renderer's engine-only output;
    never inject that build input or validate a fabricated hybrid property map.
    """
    from noesis.reid_swin_profile import (load_reid_swin_nvinfer_properties,
        validate_reid_swin_source_config, validate_reid_swin_model_config,
        REID_SWIN_ONNX_SHA256)
    from reid.mv3dt_person_association import AppearanceProfile
    source = source_ini if source_ini is not None else ini
    props = dict(validate_reid_swin_source_config(source, pipeline_config=pipeline_config))
    if source_ini is not None:
        from pathlib import Path
        from noesis_core.inference_runtime_contract import render_nvinfer_engine_only_config
        if engine_path is None or model_config is None or runtime_properties is None:
            raise AcquisitionUnavailable("selected_swin_original_source_and_effective_engine_required")
        validate_reid_swin_model_config(model_config)
        if Path(model_config["config-file-path"]).resolve() != Path(ini).resolve():
            raise AcquisitionUnavailable("selected_swin_effective_ini_mismatch")
        if Path(model_config["engine"]).resolve() != Path(engine_path).resolve():
            raise AcquisitionUnavailable("selected_swin_effective_engine_mismatch")
        rendered = render_nvinfer_engine_only_config(source_config=Path(source),
            engine_path=Path(engine_path), repo_root=Path(__file__).resolve().parents[3])
        if Path(ini).read_text() != rendered or props != dict(load_reid_swin_nvinfer_properties(source)):
            raise AcquisitionUnavailable("selected_swin_effective_derivation_or_source_drift")
        effective = dict(load_reid_swin_nvinfer_properties(ini))
        aliases = {"unique-id": "gie-unique-id"}
        defaults = {"interval": "0", "input-tensor-meta": "0", "output-instance-mask": "0"}
        for name, value in runtime_properties.items():
            if name == "config-file-path":
                if Path(value).resolve() != Path(ini).resolve():
                    raise AcquisitionUnavailable("selected_swin_component_ini_mismatch")
                continue
            key = aliases.get(name, name)
            if key not in effective and key not in defaults: continue
            actual = str(int(value)) if type(value) is bool else str(value)
            if actual != effective.get(key, defaults.get(key)):
                raise AcquisitionUnavailable("selected_swin_component_override_mismatch:" + name)
    fingerprint = hashlib.sha256(repr((REID_SWIN_ONNX_SHA256, tuple(sorted(props.items())))).encode()).hexdigest()
    return AppearanceProfile("tao-swin-tiny-market1501-aicity156", "fc_pred", 256, fingerprint)


class SelectedSwinIngress:
    """Paired exact adjacent-before-SGIE output/src, no freshness heuristic.

    Installed sync classifier cached path attaches classifiers ONLY. Prove
    selected tensor count zero immediately before SGIE, unique at its src,
    on the SAME owned frame.
    Copy/reused output of a previous buffer cannot create a new extraction.
    """
    def __init__(self, run_id, graph_id, profile, analytics, reid, *, timestamp_origin, budget=256,
                 coverage_diagnostics=False):
        self.profile, self.analytics, self.reid = profile, analytics, reid
        self.state = SourceAcquisitionState(run_id, graph_id, timestamp_origin)
        self.cache = OriginalSwinCache(coverage_diagnostics=coverage_diagnostics, run_id=run_id, graph_id=graph_id)
        self.pending = [{} for _ in CAMERAS]
        self.frontier = [None for _ in CAMERAS]
        self.budget = min(256, max(0, budget))
        self.counters = Counter()

    def _frames(self, batch, consume):
        receipts = ScopedFrameReceipts(batch, self.analytics.extract_source_acquisition(batch),
                                      frame_key=self.analytics.extract_frame_identity)
        keys = ordered_current_frames(batch.frame_items, frame_key=self.analytics.extract_frame_identity)
        rows = {key: receipts.lookup_key(batch, key) for key in keys}
        consume_ordered_current_frames(batch, keys, lambda frame, key: consume(frame, rows[key]),
                                       frame_key=self.analytics.extract_frame_identity)

    @staticmethod
    def _identity(row):
        return (row["original_buffer_scope"], row["slot"], row["source_id"], row["original_frame"],
                row["original_pts_ns"], row["original_ntp_ns"], row["sdk_source_id"])

    def enter(self, batch):
        def consume(frame, row):
            source, identity = row["source_id"], self._identity(row)
            frontier = self.frontier[source]
            if frontier is not None and (identity[0] < frontier[0] or identity == frontier):
                self.counters["copied_or_replayed_input"] += 1; return
            if len(self.pending[source]) >= 256:
                self.counters["ingress_pending_capacity"] += 1; return
            original = self.state.observe_native(row)
            objects = {}
            for index, obj in enumerate(frame.object_items):
                if index >= 256: self.counters["object_capacity"] += 1; break
                if obj.class_id != 0: continue
                raw = obj.object_id
                if raw in objects: raise AcquisitionUnavailable("duplicate_swin_raw_id_in_exact_frame")
                objects[raw] = self.reid.count_reid_tensor_records(obj, 3)
            self.pending[source][identity] = (original, objects)
            self.frontier[source] = identity
        self._frames(batch, consume)

    def leave(self, batch):
        def consume(frame, row):
            source, identity = row["source_id"], self._identity(row)
            before = self.pending[source].pop(identity, None)
            if before is None: self.counters["exact_sgie_input_unavailable"] += 1; return
            original, objects = before
            remaining, seen = self.budget, set()
            for index, obj in enumerate(frame.object_items):
                if index >= 256: self.counters["object_capacity"] += 1; break
                if obj.class_id != 0: continue
                raw = obj.object_id
                if raw in seen: raise AcquisitionUnavailable("duplicate_swin_output_raw_id")
                seen.add(raw)
                if objects.get(raw) != 0:
                    self.counters["selected_input_tensor_or_member_mismatch"] += 1; continue
                if remaining <= 0: self.counters["existing_extraction_budget"] += 1; continue
                descriptor = self.reid.extract_reid_ingress(obj, 3, "fc_pred", 256)
                if not isinstance(descriptor, dict) or descriptor.get("matching_metadata_count") != 1:
                    self.counters["cached_or_duplicate_tensor"] += 1; continue
                if (descriptor.get("gie_id"), descriptor.get("layer"), descriptor.get("dimensions")) != (3, "fc_pred", 256):
                    self.counters["profile_mismatch"] += 1; continue
                payload = descriptor.get("embedding")
                if payload is None: self.counters["invalid_tensor"] += 1; continue
                remaining -= 1
                vector = tuple(float(value) for value in payload)
                self.cache.record(OriginalSwinExtraction(original, raw, self.profile, vector, time.monotonic()))
        self._frames(batch, consume)


@dataclass(frozen=True)
class OriginalAssociationOperation:
    """Exact accepted operation, not a substitute proof/event schema."""
    kind: str  # link, transition, or local observe
    observation: object = None  # actual frozen SourceObservation
    proof: object = None  # actual frozen LinkProof
    event: object = None  # actual frozen NativeTransition


@dataclass(frozen=True)
class WorldFrameInput:
    """Private detached queue container; public World rows/observations unchanged."""
    source_id: int
    rows: tuple  # recursively frozen existing producer dictionaries
    metadata: tuple
    observations: tuple  # actual SourceObservation inputs, original opaque tokens


def freeze_world_input(value):
    if isinstance(value, Mapping):
        return tuple((key, freeze_world_input(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)): return tuple(freeze_world_input(item) for item in value)
    if value is None or type(value) in (str, bool, int, float): return value
    raise AcquisitionUnavailable("world_input_not_detached")


def thaw_world_input(value):
    if isinstance(value, tuple):
        if value and all(type(item) is tuple and len(item) == 2 and type(item[0]) is str for item in value):
            return {key: thaw_world_input(item) for key, item in value}
        return [thaw_world_input(item) for item in value]
    return value


@dataclass(frozen=True)
class OrderedProducerWork:
    run_id: str
    source_id: int
    source_epoch: int
    frame_id: int
    at: object  # actual frozen AcquisitionTime, or unavailable
    mutations: tuple = ()  # actual frozen owner service inputs, in exact order
    operations: tuple[OriginalAssociationOperation, ...] = ()
    measurements: tuple = ()  # actual frozen MeasurementReferences for ALL members
    denial: str | None = None
    native_evidence: PrivateAssociationWork | None = None
    world_inputs: tuple[WorldFrameInput, ...] = ()

    def __post_init__(self):
        # Reject SDK objects/vectors or arbitrary records BEFORE queue admission,
        # not only when the worker eventually touches the real service.
        from noesis.identity_association_service import MeasurementReference, ProducerAdmission
        from reid.mv3dt_person_association import (AcquisitionTime, AppearanceEvidence, AppearanceProfile,
            AssociationPolicy, HouseholdDescriptor, LifecycleRetention, LinkProof, MeasurementEvidence,
            MemberKey, MemberRemoval, NativeTransition, ObservationStamp, SourceAbsence, SourceBinding,
            SourceObservation)
        allowed = (MeasurementReference, ProducerAdmission, AcquisitionTime, AppearanceEvidence,
            AppearanceProfile, AssociationPolicy, HouseholdDescriptor, LifecycleRetention, LinkProof,
            MeasurementEvidence, MemberKey, MemberRemoval, NativeTransition, ObservationStamp, SourceAbsence,
            SourceBinding, SourceObservation, OriginalAssociationOperation, PrivateAssociationWork, OriginalCohort,
            WorldFrameInput)
        budget = [8192]
        def detached(value, depth=0):
            budget[0] -= 1
            if budget[0] < 0 or depth > 12:
                raise ValueError("bounded detached producer evidence required")
            if value is None or type(value) in (bool,int): return
            if type(value) is float and math.isfinite(value): return
            if type(value) is str and len(value) <= 256: return
            if type(value) is tuple and len(value) <= 512:
                for item in value: detached(item,depth+1)
                return
            if type(value) in allowed:
                for field in fields(value): detached(getattr(value,field.name),depth+1)
                return
            raise TypeError("producer work contains unsupported/SDK/embedding object")
        for field in fields(self): detached(getattr(self,field.name))


@dataclass(frozen=True)
class OrderedProducerPreparation:
    status: str
    reason: str
    contributions: tuple = ()  # actual (AssociationDTO, MeasurementCohort) objects
    withdrawn_members: tuple = ()
    retired_keys: tuple = ()


class OrderedAssociationIntegrityError(RuntimeError):
    pass


class Xc3dWorldIngress:
    """One real publisher's prepare/commit adapter; never another publisher.

    Real CanonicalWorldService owns receipts/grants/fusion/journal. Stage exact
    FIFO work only; no media-side mutation, private verifier or last-seen join.
    """
    def __init__(self, world):
        self.world = world
        self.staged = None
        self.pending = None

    def stage(self, work, identity):
        if self.staged is not None or self.pending is not None:
            raise OrderedAssociationIntegrityError("one_ordered_world_preparation_required")
        self.staged = (work, identity)

    def prepare(self, source, tracks, *, metadata):
        staged, self.staged = self.staged, None
        if staged is None:
            return self.world.prepare(source, tracks, metadata=metadata)
        work, identity = staged
        effects = identity.contributions or identity.withdrawn_members or identity.retired_keys
        if not effects:
            return self.world.prepare(source, tracks, metadata=metadata)
        from reid.mv3dt_person_association import AcquisitionTime
        if type(work.at) is not AcquisitionTime or not work.world_inputs:
            raise OrderedAssociationIntegrityError("exact_physical_clock_and_original_world_inputs_required")
        receipts = []
        try:
            for original in work.world_inputs:
                if type(original) is not WorldFrameInput:
                    raise OrderedAssociationIntegrityError("exact_detached_world_input_required")
                receipts.append(self.world.prepare(original.source_id, thaw_world_input(original.rows),
                    metadata=thaw_world_input(original.metadata), physical_observations=original.observations))
            prepared = self.world.prepare_physical(source, tuple(receipts), identity.contributions,
                at=work.at, withdrawn_members=identity.withdrawn_members, retired_keys=identity.retired_keys)
            self.pending = (prepared, work.at)
            return prepared
        except BaseException:
            for receipt in receipts: self.world.discard(receipt)
            raise

    def commit(self, prepared):
        if self.pending is None:
            return self.world.commit(prepared)
        original, at = self.pending
        if prepared is not original: raise OrderedAssociationIntegrityError("world_receipt_identity_mismatch")
        try: return self.world.commit(original, association_at=at)
        finally: self.pending = None

    def discard(self, prepared):
        self.world.discard(prepared)
        if self.pending is not None and self.pending[0] is prepared: self.pending = None


class OrderedIdentityProducer:
    """One FIFO actor owns ACTUAL service mutations BEFORE World publication.

    Only exact frozen service/kernel inputs are accepted. No private ledger,
    inferred adoption, DTO copies, token changes, last-seen proof, or vectors.
    World owns the real transaction/consume API; this actor performs no public
    per-person commit and cannot turn source strings into physical admission.
    """
    def __init__(self, run_id, admission):
        from noesis.identity_association_service import IdentityAssociationService, ProducerAdmission
        if type(admission) is not ProducerAdmission:
            raise ValueError("actual ProducerAdmission required")
        self.run_id, self.admission = run_id, admission
        self.service = IdentityAssociationService(run_id, admission)
        self._actor = None
        self._frontiers = {}  # <=3 source FIFO frontiers, NOT an association ledger.

    def _operation(self, operation, at):
        from reid.mv3dt_person_association import LinkProof, NativeTransition, SourceObservation
        if type(operation) is not OriginalAssociationOperation:
            raise OrderedAssociationIntegrityError("original_operation_type")
        if operation.kind == "link" and type(operation.proof) is LinkProof and operation.event is None:
            return self.service.link(operation.proof, at)
        if (operation.kind == "transition" and type(operation.event) is NativeTransition and
                type(operation.observation) is SourceObservation and
                (operation.proof is None or type(operation.proof) is LinkProof)):
            return self.service.transition(operation.event, operation.observation, at, operation.proof)
        if operation.kind == "local" and type(operation.observation) is SourceObservation and operation.proof is None and operation.event is None:
            return self.service.observe(operation.observation)
        raise OrderedAssociationIntegrityError("original_operation_unavailable")

    def prepare(self, work):
        from noesis.identity_association_service import (MeasurementCohort, MeasurementReference,
                                                        ProducerAdmission)
        from reid.mv3dt_person_association import (AcquisitionTime, SourceObservation,
            MemberRemoval, SourceAbsence, SourceBinding)
        actor = threading.get_ident()
        if self._actor is None:
            self._actor = actor
        if actor != self._actor:
            raise OrderedAssociationIntegrityError("one_ordered_identity_actor_required")
        if (type(work) is not OrderedProducerWork or work.run_id != self.run_id or
                type(work.source_id) is not int or not 0 <= work.source_id < 3 or
                type(work.source_epoch) is not int or work.source_epoch < 0 or
                type(work.frame_id) is not int or work.frame_id < 0 or
                type(work.mutations) is not tuple or len(work.mutations) > 512 or
                type(work.operations) is not tuple or len(work.operations) > 64 or
                type(work.measurements) is not tuple or len(work.measurements) > 128):
            raise OrderedAssociationIntegrityError("bounded_exact_producer_work_required")
        previous = self._frontiers.get(work.source_id)
        order = (work.source_epoch, work.frame_id)
        if previous is not None and (order[0] < previous[0] or
                (order[0] == previous[0] and order[1] <= previous[1])):
            raise OrderedAssociationIntegrityError("delayed_or_replayed_producer_epoch_work")
        withdrawn, retired = set(), set()
        def collect(result, *, observation_progress=False):
            withdrawn.update(result.withdrawn_members)
            retired.update(result.retired_keys)
            # Updating a linked member legitimately invalidates its old proof
            # before the SAME FIFO work's fresh link is issued. This is private
            # observation progress, never a public contribution or grant.
            unresolved = result.disposition == "unresolved" and not observation_progress
            if unresolved or result.disposition in {"conflict", "capacity", "denied", "rejected", "unavailable"}:
                from noesis_core.world_service import WorldPublicationRefused
                raise WorldPublicationRefused.from_association_result(result)
            return result
        # Exact removals/reset/admission/observations precede EVERY reservation.
        # Even no-fresh-proof work carries and applies truthful lifecycle inputs.
        for mutation in work.mutations:
            if type(mutation) is SourceObservation: result = self.service.observe(mutation)
            elif type(mutation) is MemberRemoval: result = self.service.withdraw(mutation)
            elif type(mutation) is SourceAbsence: result = self.service.clear_source(mutation)
            elif type(mutation) is SourceBinding: result = self.service.reset_source(mutation)
            elif type(mutation) is ProducerAdmission:
                result = self.service.bind_admission(mutation)
                self.admission = mutation
            else: raise OrderedAssociationIntegrityError("accepted_mutation_api_required")
            collect(result, observation_progress=type(mutation) is SourceObservation)
        self._frontiers[work.source_id] = order
        if self.admission.status != "admitted" or work.denial or type(work.at) is not AcquisitionTime:
            return OrderedProducerPreparation("unavailable", work.denial or "producer_admission_or_clock_unavailable",
                withdrawn_members=tuple(sorted(withdrawn)), retired_keys=tuple(sorted(retired)))
        binding = next((source for source in self.admission.policy.sources
                        if source.source_id == work.source_id), None)
        if binding is None or binding.source_epoch != work.source_epoch:
            raise OrderedAssociationIntegrityError("exact_producer_epoch_admission_required")
        # ALL genuine operations first. No person gets an early public commit.
        for operation in work.operations:
            collect(self._operation(operation, work.at))
        revision = self.service.stats().association_revision
        references = {}
        for measured in work.measurements:
            if type(measured) is not MeasurementReference or measured.stamp in references:
                raise OrderedAssociationIntegrityError("exact_all_member_measurement_references_required")
            references[measured.stamp] = measured
        contributions = {}
        for operation in work.operations:
            # R5: adoption reissues ORIGINAL transition, NOT link(adoption proof).
            result = collect(self._operation(operation, work.at))
            if self.service.stats().association_revision != revision:
                raise OrderedAssociationIntegrityError("final_issuance_mutated_revision_reconcile_required")
            dto = result.dto
            if dto is None or dto.state not in {"accepted", "local"}:
                raise OrderedAssociationIntegrityError(result.reason)
            try:
                exact = tuple(references[member.observation] for member in dto.members)
            except KeyError as exc:
                raise OrderedAssociationIntegrityError("measurement_reference_for_current_member_unavailable") from exc
            if dto.proof is not None and exact != dto.proof.measurements:
                raise OrderedAssociationIntegrityError("original_measurement_proof_reference_mismatch")
            cohort = MeasurementCohort(dto.run_id, dto.association_revision, dto.policy_identity,
                                       dto.policy.target_frame, dto.policy.world_revision, exact)
            contributions[dto.physical_key] = (dto, cohort)
        return OrderedProducerPreparation("ready", "exact_original_operations_final_revision",
            tuple(contributions.values()), tuple(sorted(withdrawn)), tuple(sorted(retired)))

    def native_unavailable_work(self, source, epoch, frame, evidence):
        # Real source/epoch/generation/original birth are preserved on the exact
        # native work. Current comparability, geometry, adoption and authentic
        # ORIGINAL Swin extraction stamp are absent: create NO fake observation.
        return OrderedProducerWork(self.run_id, source, epoch, frame, None,
            denial="native_metric_clock_camera_epoch_original_swin_unavailable", native_evidence=evidence)


def private_owner_diagnostic(evidence, source_epoch, result):
    """Bounded JSON-shaped PRIVATE diagnostic, never a public/service DTO.

    Its current exact work receipt and each original birth remain distinct.
    No embedding, surface, SDK pointer, geometry estimate or alias is serialized.
    """
    members = result.get("members", ())
    if len(members) > 256:
        raise AcquisitionUnavailable("private_diagnostic_member_bounds")
    diagnostic = {key: value for key, value in result.items() if key != "members"}
    diagnostic["current_work"] = asdict(evidence)
    diagnostic["source_epoch"] = source_epoch
    diagnostic["members"] = [{"owner_lifecycle": list(key), "original_birth": asdict(birth)}
                             for key, birth in members]
    return diagnostic


class SourceAcquisitionState:
    """At most 3 sources, 32 SRs/source, 256 frames and owner members/source.

    Original owner first-admitted cohorts are retained separately from the
    frame ring. Renewals and history compaction cannot retimestamp their birth.
    SDK birth and previous/adopted ID are unavailable, never inferred.
    """

    def __init__(self, run_id, graph_id, timestamp_origin="host_receive"):
        if not run_id or not graph_id or timestamp_origin not in {"host_receive", "source_sr", "source_pts"}:
            raise AcquisitionUnavailable("missing_run_graph_or_origin")
        self.run_id, self.graph_id, self.origin = run_id, graph_id, timestamp_origin
        self.epochs = [0] * 3
        self.frames = [deque(maxlen=256) for _ in CAMERAS]
        self.srs = [deque(maxlen=32) for _ in CAMERAS]
        self.members = [{} for _ in CAMERAS]
        self.births = [{} for _ in CAMERAS]
        self.last = [None] * 3
        self.sessions = [None] * 3
        self.counters = Counter()
        self.owner_watermarks = [None] * 3
        self.owner_clock_epochs = [None] * 3

    def _source(self, source):
        if type(source) is not int or not 0 <= source < 3:
            raise AcquisitionUnavailable("source_out_of_bounds")
        return source

    def reset(self, source, reason):
        source = self._source(source)
        self.epochs[source] += 1
        self.frames[source].clear()
        self.srs[source].clear()
        self.members[source].clear()
        self.births[source].clear()
        self.last[source] = None
        self.counters[reason] += 1

    def observe_sr(self, source, session, ssrc, rtp, ntp_ns, host_arrival_ns):
        source = self._source(source)
        for value, field in ((ssrc, "ssrc"), (rtp, "rtp"), (ntp_ns, "ntp"), (host_arrival_ns, "arrival")):
            _integer(value, field)
        if not session or not ntp_ns or ssrc > 0xffffffff or rtp > 0xffffffff:
            raise AcquisitionUnavailable("sr_invalid")
        epoch = (str(session), ssrc)
        if self.sessions[source] is not None and self.sessions[source] != epoch:
            if str(session).isdigit() and self.sessions[source][0].isdigit() and int(session) < int(self.sessions[source][0]):
                raise AcquisitionUnavailable("stale_sr_session")
            self.reset(source, "session_ssrc_reset")
        self.sessions[source] = epoch
        reports = self.srs[source]
        if reports and (ntp_ns <= reports[-1][1] or host_arrival_ns <= reports[-1][2]
                        or not 0 < ((rtp - reports[-1][0]) & 0xffffffff) < 0x80000000):
            self.reset(source, "sr_clock_regression")
            self.sessions[source] = None
            raise AcquisitionUnavailable("sr_clock_regression")
        if len(reports) == reports.maxlen:
            self.counters["sr_retained_drop"] += 1
        reports.append((rtp, ntp_ns, host_arrival_ns))
        self.counters["sr_observed"] += 1

    def observe_native(self, row):
        """Consume one exact-buffer pre/post-tracker scalar receipt, never last-seen."""
        source = self._source(row.get("source_id"))
        try:
            if row.get("receipt_valid") is not True:
                raise AcquisitionUnavailable("exact_buffer_receipt_missing")
            original = _integer(row.get("original_frame"), "original_frame")
            pts = _integer(row.get("original_pts_ns"), "pts")
            ntp = _integer(row.get("original_ntp_ns"), "ntp")
            _integer(row.get("post_tracker_frame"), "post_tracker_frame")
            if row.get("post_tracker_pts_ns") != pts or row.get("original_source_id") != source:
                raise AcquisitionUnavailable("exact_buffer_join_mismatch")
            if (row.get("original_batch_id") != row.get("slot")
                or row.get("original_sdk_source_id") != row.get("sdk_source_id")
                or row.get("post_tracker_ntp_ns") != ntp
                or not _integer(row.get("original_buffer_scope"), "original_buffer_scope")):
                raise AcquisitionUnavailable("exact_original_slot_source_ntp_scope_mismatch")
            previous = self.last[source]
            if previous is not None and (original <= previous[0] or pts <= previous[1]
                                         or (ntp and previous[2] and ntp <= previous[2])):
                raise AcquisitionUnavailable("frame_clock_duplicate_or_regression")
        except AcquisitionUnavailable:
            self.reset(source, "invalid_native_clock")
            raise
        cohort = OriginalCohort(self.run_id, self.graph_id, CAMERAS[source], source,
                                self.epochs[source], original, pts, ntp, self.origin,
                                row["original_buffer_scope"], row["slot"], row["sdk_source_id"])
        if len(self.frames[source]) == 256:
            self.counters["frame_retained_drop"] += 1
        self.frames[source].append(cohort)
        self.last[source] = (original, pts, ntp)
        self.counters["native_frames"] += 1
        self.counters["native_ntp_frames"] += bool(ntp)
        return cohort

    def owner_frame(self, source, source_epoch, frame_receipt, tracks, terminated=(), *, detached=False):
        """Observe canonical OWNER lifecycle, preserving its ORIGINAL birth cohort.

        This private diagnostic is not MemberKey/LinkReceipt/service DTO. Generation
        is supplied by the existing canonical lifecycle owner, never inferred here.
        """
        source = self._source(source)
        if frame_receipt is None or (not detached and frame_receipt not in self.frames[source]):
            self.reset(source, "owner_receipt_unavailable")
            return {"status": "acquisition_unavailable", "members": ()}
        current = self.members[source]
        births = self.births[source]
        for key in tuple(births):
            if key[0] != source_epoch or key[1] in terminated:
                births.pop(key)
                current.pop(key, None)
        keys = set()
        for track in tracks:
            aliases = tuple(track[name] for name in ("tracker_id", "id", "track_id") if name in track)
            if aliases and any(value != aliases[0] for value in aliases[1:]):
                raise AcquisitionUnavailable("ambiguous_owner_tracker_id_alias")
            raw = aliases[0] if aliases else None
            generation = track.get("tracker_lifecycle_generation")
            if type(raw) is not int or type(generation) is not int:
                continue
            key = (source_epoch, raw, generation)
            if raw in terminated:
                continue
            keys.add(key)
            for old in tuple(births):
                if old[1] == raw and old[2] != generation:
                    births.pop(old)
                    current.pop(old, None)
            if key not in current:
                if key not in births and len(births) >= 256:
                    raise AcquisitionUnavailable("owner_member_capacity")
                births.setdefault(key, frame_receipt)
                current[key] = births[key]
        # Exact absence, termination and ID change withdraw old membership.
        for key in tuple(current):
            if key not in keys:
                del current[key]
        return {"status": "accepted_identity_dto_unavailable",
                "transition": "transition_unavailable", "sdk_birth": None,
                "geometry": "geometry_unavailable", "geometry_camera_epoch": None,
                "original_swin_evidence": "unavailable_until_accepted_owner_stamp_interface",
                "fresh_appearance_extraction": False,
                "members": tuple((key, birth) for key, birth in current.items())}

    def consume_owner_work(self, evidence, source_epoch, tracks):
        """ONE FIFO publication owner applies detached private barriers.

        No media callback mutates this owner's members/reservations. This does
        not reserve/commit the unaccepted Identity service or World transaction.
        """
        source = self._source(evidence.source_id)
        watermark = (source_epoch, evidence.canonical_frame, evidence.media_pts_ns)
        previous = self.owner_watermarks[source]
        if previous is not None and (watermark[0] < previous[0] or
                (watermark[0] == previous[0] and
                 (watermark[1] <= previous[1] or watermark[2] <= previous[2]))):
            raise AcquisitionUnavailable("private_owner_fifo_order_invalid")
        if (self.owner_clock_epochs[source] != evidence.clock_epoch or
                (previous is not None and previous[0] != source_epoch)):
            self.members[source].clear()
            self.births[source].clear()
        self.owner_clock_epochs[source] = evidence.clock_epoch
        self.owner_watermarks[source] = watermark
        if evidence.denial is not None or evidence.original is None:
            self.members[source].clear()
            self.births[source].clear()
            self.counters["ordered_association_unavailable"] += 1
            return {"status": "acquisition_unavailable", "reason": evidence.denial, "members": ()}
        if (evidence.original.source_id != source or evidence.original.pts_ns != evidence.media_pts_ns
                or evidence.original.source_clock_epoch != evidence.clock_epoch or
                (evidence.original.run_id, evidence.original.graph_id) != (self.run_id, self.graph_id)):
            self.members[source].clear(); self.births[source].clear()
            raise AcquisitionUnavailable("private_owner_exact_cohort_mismatch")
        try:
            retained, measured = owner_retention_members(evidence.owner_retention, evidence.original,
                                                        source_epoch, evidence.canonical_frame)
            if tuple(evidence.terminated) != evidence.owner_retention.terminated:
                raise AcquisitionUnavailable("same_cohort_owner_termination_required")
        except AcquisitionUnavailable:
            self.members[source].clear(); self.births[source].clear()
            self.counters["ordered_owner_retention_unavailable"] += 1
            return {"status": "acquisition_unavailable", "reason": "same_cohort_owner_retention_unavailable", "members": ()}
        for key in tuple(self.births[source]):
            if key not in retained or key[1] in evidence.terminated:
                self.births[source].pop(key); self.members[source].pop(key, None)
        filtered = []
        for track in tracks:
            raw = track.get("tracker_id", track.get("id", track.get("track_id")))
            if (source_epoch, raw, track.get("tracker_lifecycle_generation")) in measured:
                filtered.append(track)
        if len(filtered) != len(tracks):
            self.members[source].clear()
            return {"status": "acquisition_unavailable", "reason": "current_measured_owner_member_required", "members": ()}
        return self.owner_frame(source, source_epoch, evidence.original, filtered,
                                evidence.terminated, detached=True)

    def mapping(self, source):
        reports = self.srs[self._source(source)]
        if len(reports) < 2:
            return {"valid": False, "reason": "sr_mapping_unavailable", "comparable": False}
        first, last = reports[0], reports[-1]
        ticks = (last[0] - first[0]) & 0xffffffff
        span = last[1] - first[1]
        if not ticks or span <= 0:
            return {"valid": False, "reason": "sr_mapping_invalid", "comparable": False}
        rate = ticks * 1e9 / span
        residuals = [abs(ntp - (first[1] + ((rtp - first[0]) & 0xffffffff) * 1e9 / rate))
                     for rtp, ntp, _arrival in reports]
        return {"valid": math.isfinite(rate), "rtp_rate_hz": rate,
                "rtp_origin": first[0], "ntp_origin_ns": first[1],
                "mapping_epoch": (self.epochs[source], self.sessions[source]),
                "residual_max_ns": max(residuals), "sr_count": len(reports),
                "comparable": False, "reason": "independent_clock_offset_unqualified",
                "host_arrival_is_latency_only": True}

    def summary(self):
        total = self.counters["native_frames"]
        return {"timestamp_origin": self.origin, "comparable": False,
                "ntp_fraction": self.counters["native_ntp_frames"] / total if total else 0,
                "sources": [self.mapping(i) for i in range(3)],
                "counters": dict(self.counters), "offset_skew_drift": "unqualified"}
