"""Bounded private quality handoff; no SDK, filter, camera selection or grants.

Public World measurement/depth/lifecycle objects are already immutable scalar
contracts. Keep their originals, not diagnostic JSON or a later pipeline row.
The actual World transaction is bound only after its owning implementation returns.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import inspect
import logging
import math
import threading
import time
from collections import deque
from numbers import Real
from collections.abc import Mapping, Sequence

from noesis.metadata.object_depth import ObjectDepthResult, NativePosePatchSupport, PoseDepthSample
from noesis_core.tracking_continuity import TrackingContinuityUpdate, TrackingIdentityObservation
from noesis_core.contracts.world_measurement import WorldMeasurementSet, ResolvedGroundMeasurement, WorldMeasurementCohort
from reid.mv3dt_person_association import HouseholdDescriptor
from noesis.pipelines.source_acquisition import (
    OriginalCohort, PrivateAssociationWork, SwinOwnerBinding, owner_retention_members, AcquisitionUnavailable,
    SelectedSwinIngress, OriginalSwinCache, OriginalSwinExtraction, OwnerSourceAvailability,
)

MAX_MEMBERS = 128
_RAW_MEDIA_UNSET = object()
logger = logging.getLogger(__name__)


def _binding_refusal(row, reason):
    """ONE bounded scalar diagnostic in the ordered actor, never SDK/vector data."""
    from noesis_core.world_service import WorldPublicationRefused
    fields = dict(row.evidence.world)
    cohort = row.evidence.measurements.cohort if row.evidence.measurements is not None else None
    scalar = lambda value: value if value is None or type(value) in (bool, int, float, str) else "<non-scalar>"
    logger.error("World quality binding mismatch: %s", {
        "source_id": row.original.source_id, "frame_id": row.evidence.lifecycle.frame_id,
        "reason": reason, "resolver_status": getattr(row.evidence.resolution, "status", None),
        "resolver_binding": {name: getattr(cohort, name, None) for name in
            ("source_id", "camera_id", "frame_id", "world_frame", "world_revision", "calibration_revision", "world_transform_sha256")},
        "emitted": {name: scalar(fields.get(name)) for name in ("world_valid", "world_quality", "world_source", "world_measurement_accepted",
            "world_frame", "world_frame_revision", "world_calibration_sha256", "world_transform_sha256", "calibration_revision")},
        "position_present": fields.get("world") is not None,
        "covariance_present": fields.get("world_covariance") is not None})
    return WorldPublicationRefused(reason)


def snapshot_swin_lineage(pipeline):
    """One bounded owned snapshot in the sole ordered owners actor, no SDK/lock."""
    ingress = getattr(pipeline, "xc3d_swin_ingress", None)
    if type(ingress) is not SelectedSwinIngress:
        raise AcquisitionUnavailable("world_quality_selected_swin_ingress_unavailable")
    cache = getattr(ingress, "cache", None)
    if type(cache) is not OriginalSwinCache:
        raise AcquisitionUnavailable("world_quality_original_swin_cache_unavailable")
    run, graph = ingress.state.run_id, ingress.state.graph_id
    if (any(type(value) is not str or not 0 < len(value) <= 240 for value in (run, graph))
            or (cache.run_id, cache.graph_id) != (run, graph)):
        raise AcquisitionUnavailable("world_quality_lineage_run_graph_unavailable")
    if type(cache.owners) is not list or len(cache.owners) != 3:
        raise AcquisitionUnavailable("world_quality_lineage_source_arity")
    snapshot = {}
    for source, owners in enumerate(cache.owners):
        if type(owners) is not dict or len(owners) > 256:
            raise AcquisitionUnavailable("world_quality_lineage_source_map_unavailable")
        copied = dict(owners)
        for key, binding in copied.items():
            if (type(key) is not tuple or len(key) != 3 or
                    any(type(value) is not int or value < 0 for value in key)):
                raise AcquisitionUnavailable("world_quality_lineage_member_key_invalid")
            if (type(binding) is not SwinOwnerBinding or type(binding.member) is not tuple
                    or any(type(value) is not int or value < 0 for value in binding.member)
                    or binding.member != key
                    or type(binding.original_birth) is not OriginalCohort
                    or type(binding.lifecycle_started_frame) is not int or binding.lifecycle_started_frame < 0):
                raise AcquisitionUnavailable("world_quality_lineage_original_binding_invalid")
            birth = binding.original_birth
            if (type(birth.source_id) is not int or birth.source_id != source
                    or (birth.run_id, birth.graph_id) != (run, graph)):
                raise AcquisitionUnavailable("world_quality_lineage_original_source_run_graph_mismatch")
        snapshot[source] = copied
    return snapshot


def require_world_quality_api():
    """Fail before SDK initialization if the actual normal public seam is absent."""
    from noesis_core.runtime_world import create_runtime_world_service
    from noesis_core.world_service import CanonicalWorldService
    from noesis.identity_association_service import IdentityAssociationService
    prepare = getattr(CanonicalWorldService, "prepare_spatial", None)
    mode = inspect.signature(prepare).parameters.get("attribution_mode") if callable(prepare) else None
    if ("spatial_bindings" not in inspect.signature(create_runtime_world_service).parameters
            or "spatial_bindings" not in inspect.signature(CanonicalWorldService).parameters
            or mode is None or mode.kind is not inspect.Parameter.KEYWORD_ONLY or mode.default != "strict"
            or not callable(getattr(IdentityAssociationService, "prepare_spatial_physical_member", None))
            or not callable(getattr(IdentityAssociationService, "commit_spatial_physical_member", None))
            or not callable(getattr(IdentityAssociationService, "for_spatial", None))):
        raise ValueError("xc3d_world_quality_public_api_unavailable")


def require_world_quality_attachment(pipeline):
    """The new opt-in must not start sources with a swallowed probe failure."""
    status = getattr(pipeline, "xc3d_analytics_attachment_status", {})
    if (status.get("status") != "attached"
            or getattr(pipeline, "xc3d_swin_ingress", None) is None
            or getattr(pipeline, "xc3d_source_acquisition", None) is None):
        raise ValueError("xc3d_world_quality_analytics_attachment_unavailable")


def runtime_spatial_bindings(run_id, camera_labels, calibration_provider, scene_priors):
    """Read real already-guarded frame inputs; no identity transform/PCF synthesis."""
    from noesis_core.contracts.scene_prior import ScenePriorFrameBinding
    from reid.mv3dt_person_association import SourceBinding
    if scene_priors is None or not callable(getattr(scene_priors, "frame_bindings", None)):
        raise ValueError("world_quality_explicit_frame_bindings_unavailable")
    frames = scene_priors.frame_bindings()
    if not 1 <= len(camera_labels) <= 3:
        raise ValueError("world_quality_source_bounds")
    batch = getattr(calibration_provider, 'snapshots', None)
    if not callable(batch):
        raise ValueError("world_quality_atomic_calibration_snapshots_unavailable")
    snapshots = batch(camera_labels)
    if not isinstance(snapshots, Mapping) or set(snapshots) != set(camera_labels):
        raise ValueError("world_quality_atomic_calibration_snapshot_sources")
    active, sources = {}, []
    for source, camera in sorted(camera_labels.items()):
        binding = frames.get(camera)
        snapshot = snapshots[source]
        contract = getattr(snapshot, "frame_contract", None)
        if (type(binding) is not ScenePriorFrameBinding or snapshot is None or contract is None
                or (contract.source_frame.frame_id, contract.source_frame.revision,
                    contract.target_frame.frame_id, contract.target_frame.revision, contract.transform_sha256) !=
                   (binding.source_frame.frame_id, binding.source_frame.revision,
                    binding.target_frame.frame_id, binding.target_frame.revision, binding.target_from_source_sha256)
                or snapshot.camera_calibration_sha256 != binding.source_camera_calibration_sha256):
            raise ValueError(f"world_quality_actual_frame_binding_unavailable:{source}:{camera}")
        active[source] = binding
        sources.append(SourceBinding(camera, source, 0, binding.source_frame.revision,
            binding.target_from_source_sha256, f"source-media:{run_id}:{source}:0:0", "source-media"))
    return tuple(sources), active
WORLD_FIELDS = (
    "world", "world_valid", "world_covariance", "world_measurement_accepted",
    "world_source", "world_quality", "world_quality_reason", "world_frame",
    "world_frame_revision", "world_transform_sha256", "world_calibration_sha256",
    "world_resolver_confidence", "world_resolver_selected_id", "world_resolver_fused",
    "world_quantity", "world_posture", "world_support_state", "world_resolver_contact_basis",
    "world_observation_ts_us", "world_measurement_frame_id", "world_measurement_ts_us",
    "trail_segment_id", "trail_break_required",
    "calibration_revision",
)


def _scalar(value):
    if value is None or type(value) in (bool, int): return value
    if type(value) is float and math.isfinite(value): return value
    if type(value) is str and len(value) <= 240: return value
    if type(value) in (list, tuple) and len(value) <= 9:
        return tuple(_scalar(item) for item in value)
    raise ValueError("world_quality_nonfinite_unbounded_or_borrowed_value")


@dataclass(frozen=True, slots=True)
class ObservedBodyImagePatch:
    """CURRENT observed pose geometry only; no depth sample/range/3D claim."""
    anatomy: str
    anchor_uv: tuple[float,float]
    capsule: tuple[float,float,float,float,float,float]

    def __post_init__(self):
        if (self.anatomy not in {'pose_torso_support','pose_shoulder_support'}
                or type(self.anchor_uv) is not tuple or len(self.anchor_uv)!=2
                or type(self.capsule) is not tuple or len(self.capsule)!=6
                or not all(type(v) in (int,float) and math.isfinite(v) for v in self.anchor_uv+self.capsule)
                or min(self.capsule[4:])<=0):
            raise ValueError('invalid current observed image patch')

    @property
    def support_radius_px(self):
        u,v=self.anchor_uv
        c=self.capsule
        return max(math.hypot(c[0]-u,c[1]-v)+c[4],math.hypot(c[2]-u,c[3]-v)+max(c[4:]))


@dataclass(frozen=True, slots=True)
class OriginalBodySupport:
    """Private original pixels, NOT proof that a detection is a physical person.

    Issued only from the callback's already extracted, exact-current native pose.
    ``freeze_quality_work`` binds this lifecycle to the actual original raster
    cohort. No SDK object, array, public label or borrowed box survives the copy.
    Confidence is preserved as a score, never converted into a distinctness claim.
    """

    lifecycle: TrackingIdentityObservation
    bbox_xywh: tuple[float, float, float, float]
    image_size: tuple[int, int]
    keypoints: tuple[tuple[float, float, float], ...]
    detector_confidence: float | None
    tracker_confidence: float | None
    image_patches: tuple[ObservedBodyImagePatch,...] = ()


def capture_body_support(track, lifecycle, keypoints, *, pose_is_current, image_patches=()):
    """Detach at most 17x3 existing scalars; cached/missing support is unavailable.

    The boolean is the producer's exact native payload provenance check, NOT a
    caller-supplied physical-correspondence vote. This private helper performs
    no extraction, inference, clock renewal or coordinate conversion.
    """
    if (pose_is_current is not True or type(lifecycle) is not TrackingIdentityObservation
            or not isinstance(track, Mapping)):
        return None
    def finite(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
    try:
        size, box = track.get("image_size"), track.get("bbox")
        if (type(size) not in (tuple, list) or len(size) != 2
                or any(type(v) is not int or not 8 < v <= 32768 for v in size)
                or type(box) not in (tuple, list) or len(box) != 4
                or not all(finite(v) for v in box) or len(keypoints) != 17):
            return None
        width, height = size
        x, y, w, h = (float(v) for v in box)
        if not (w > 0 and h > 0 and 0 <= x < x+w <= width and 0 <= y < y+h <= height):
            return None
        points = []
        for index in range(17):
            row = keypoints[index]
            if len(row) != 3 or not all(finite(row[j]) for j in range(3)):
                return None
            u, v, confidence = (float(row[j]) for j in range(3))
            if not (0 <= confidence <= 1 and 0 <= u <= width and 0 <= v <= height):
                return None
            points.append((u, v, confidence))
        scores = []
        for field in ("confidence", "tracker_confidence"):
            score = track.get(field)
            scores.append(float(score) if finite(score) and 0 <= score <= 1 else None)
        if (type(image_patches) is not tuple or len(image_patches)>2
                or any(type(p) is not ObservedBodyImagePatch for p in image_patches)
                or len({p.anatomy for p in image_patches})!=len(image_patches)):
            return None
        patches=tuple(p for p in image_patches if x<=p.anchor_uv[0]<=x+w and y<=p.anchor_uv[1]<=y+h)
        return OriginalBodySupport(lifecycle, (x,y,w,h), (width,height), tuple(points), *scores, patches)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


@dataclass(frozen=True, slots=True)
class OriginalBodyDepth:
    """Private unflattened registered point; NOT a ground-position permission."""
    cohort: WorldMeasurementCohort
    patch: NativePosePatchSupport
    original_point: tuple[float, float, float]
    raw_depth_m: float
    registered_depth_m: float
    registration_status: str
    registration_id: str | None
    image_flips: tuple[bool, bool]


@dataclass(frozen=True, slots=True)
class OriginalAnatomyDepth:
    """One ORIGINAL sampled constraint, not an independent confidence vote.

    Ankle UV is the actual sampler's representative contact UV; two disjoint
    ankle patches need not contain that mean. Raw quantiles are NOT registered
    ranges or uncertainty bounds. Missing registration stays explicitly absent.
    """
    receipt: OriginalBodyDepth
    sample: PoseDepthSample
    registered_depth_interval_m: tuple[float, float] | None = None
    depth_convention: str | None = None

    def __post_init__(self):
        if (type(self.receipt) is not OriginalBodyDepth or type(self.sample) is not PoseDepthSample
                or self.receipt.patch.anchor_source != self.sample.anchor_source
                or self.receipt.patch.anchor_uv != self.sample.anchor_uv
                or self.sample.evidence_quality == 'rejected' or self.sample.evidence_reason is not None
                or self.sample.depth_median != self.receipt.raw_depth_m):
            raise ValueError('world_quality_original_anatomy_sample_mismatch')
        interval = self.registered_depth_interval_m
        if interval is not None and (type(interval) is not tuple or len(interval) != 2
                or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in interval)
                or interval[0] > self.receipt.registered_depth_m or interval[1] < self.receipt.registered_depth_m
                or self.depth_convention != 'camera_z_m'):
            raise ValueError('world_quality_original_anatomy_registered_interval_invalid')
        if interval is None and self.depth_convention is not None:
            raise ValueError('world_quality_original_anatomy_convention_without_interval')


@dataclass(frozen=True, slots=True)
class QualityInput:
    """One original resolver/PGS row captured in its source callback, once."""
    lifecycle: TrackingIdentityObservation
    measurements: WorldMeasurementSet | None
    resolution: ResolvedGroundMeasurement | None
    depth: ObjectDepthResult | None
    world: tuple
    household: HouseholdDescriptor
    body: OriginalBodySupport | None = None
    body_depth: OriginalBodyDepth | None = None
    anatomy_depths: tuple[OriginalAnatomyDepth, ...] = ()


def household_from_final_track(track):
    """Read explicit public household truth; provisional compatibility SID is not settled."""
    kind, state = track.get("identity_kind"), track.get("identity_state")
    if any(value is not None and type(value) is not str for value in (kind, state)):
        raise ValueError("world_quality_original_household_invalid")
    kind = kind if kind is not None else state if state is not None else "unknown"
    sid, generation, resident = track.get("stable_id"), track.get("visitor_generation"), track.get("resident_uuid")
    if sid is not None and (type(sid) is not int or sid <= 0):
        raise ValueError("world_quality_original_household_invalid")
    if kind in {"unknown", "provisional"}:
        identity = track.get("identity_v2")
        subject = identity.get("subject_id") if isinstance(identity, Mapping) else None
        if (state not in (None, kind) or generation is not None or resident is not None
                or subject is not None or track.get("reid_identity") is not None
                or kind == "unknown" and sid is not None):
            raise ValueError("world_quality_original_household_descriptor_unavailable")
        return HouseholdDescriptor(kind)
    if kind == "resident" and state in (None, "resident", "handoff") and type(resident) is str and 0 < len(resident) <= 240 and generation is None:
        return HouseholdDescriptor(kind, resident, sid)
    if kind == "visitor" and state in (None, "visitor", "handoff") and resident is None and sid is not None and type(generation) is int and generation >= 0:
        return HouseholdDescriptor(kind, f"visitor:{sid}:g{generation}", sid, generation)
    raise ValueError("world_quality_original_household_descriptor_unavailable")


def capture_quality_input(track, lifecycle, measurements, resolution, depth, body=None, body_depth=None, anatomy_depths=()):
    if type(lifecycle) is not TrackingIdentityObservation:
        raise ValueError("world_quality_actual_lifecycle_observation_required")
    if measurements is not None and type(measurements) is not WorldMeasurementSet:
        raise ValueError("world_quality_actual_measurement_set_required")
    if resolution is not None and type(resolution) is not ResolvedGroundMeasurement:
        raise ValueError("world_quality_actual_resolution_required")
    if depth is not None and type(depth) is not ObjectDepthResult:
        raise ValueError("world_quality_actual_compact_depth_required")
    if not isinstance(track, Mapping): raise ValueError("world_quality_track_required")
    if body is not None and (type(body) is not OriginalBodySupport or body.lifecycle != lifecycle):
        raise ValueError("world_quality_body_lifecycle_mismatch")
    if body is not None and (tuple(track.get("bbox", ())) != body.bbox_xywh
            or tuple(track.get("image_size", ())) != body.image_size):
        raise ValueError("world_quality_body_raster_changed")
    if body_depth is not None and (type(body_depth) is not OriginalBodyDepth
            or measurements is None or body_depth.cohort != measurements.cohort):
        raise ValueError("world_quality_body_depth_cohort_mismatch")
    if (type(anatomy_depths) is not tuple or len(anatomy_depths) > 2
            or any(type(a) is not OriginalAnatomyDepth for a in anatomy_depths)
            or len({a.sample.anchor_source for a in anatomy_depths}) != len(anatomy_depths)):
        raise ValueError('world_quality_original_anatomy_budget_or_duplicate')
    for anatomy in anatomy_depths:
        receipt, sample = anatomy.receipt, anatomy.sample
        if (body is None or depth is None or measurements is None
                or receipt.cohort != measurements.cohort
                or receipt.patch not in depth.native_pose_support or sample not in depth.pose_depth_samples
                or receipt.patch.image_size != body.image_size
                or (receipt.patch.measurement_frame_id, receipt.patch.measurement_ts_us) !=
                    (depth.measurement_frame_id, depth.measurement_ts_us)):
            raise ValueError('world_quality_original_anatomy_binding_mismatch')
    if (track.get("source_id"), track.get("tracker_id"), track.get("tracker_lifecycle_generation"),
            track.get("frame_id"), track.get("media_pts_ns")) != (
            lifecycle.source_id, lifecycle.tracker_id, lifecycle.generation,
            lifecycle.frame_id, lifecycle.media_pts_ns):
        raise ValueError("world_quality_track_lifecycle_mismatch")
    world = tuple((key, _scalar(track.get(key))) for key in WORLD_FIELDS)
    return QualityInput(lifecycle, measurements, resolution, depth, world, household_from_final_track(track), body, body_depth, anatomy_depths)


@dataclass(frozen=True, slots=True)
class QualityRow:
    original: OriginalCohort
    evidence: QualityInput
    depth_current: bool
    current_measured: bool
    denial: str | None = None


@dataclass(frozen=True, slots=True)
class QualityWork:
    """Private exact FIFO cohort, not a World/Identity measurement permission."""
    private: PrivateAssociationWork | None
    continuity: TrackingContinuityUpdate
    rows: tuple[QualityRow, ...]
    denial: str | None = None
    raw_media_pts_ns: int | None = None
    ingress_monotonic_s: float | None = None  # real callback availability, not capture time
    swin_extractions: tuple[OriginalSwinExtraction, ...] = ()  # exact pre-throttle owned receipts
    owner_availability: OwnerSourceAvailability | None = None


def freeze_swin_work(work, ingress):
    """Share already copied fresh SGIE F; never extract SDK metadata again."""
    from dataclasses import replace
    if type(ingress) is not SelectedSwinIngress or work.denial or work.private is None:
        return work
    original=work.private.original
    copies=[]
    for row in work.rows:
        value=ingress.cache.freeze_exact(original,row.evidence.lifecycle.tracker_id)
        if value is not None and value.original==original and value.profile==ingress.profile:
            copies.append(value)
    return replace(work,swin_extractions=tuple(copies))


class OriginalQualityRetention:
    """Optional historical scalars only; canonical publications never enter it.

    <=128 original rows/works, no SDK/frame/device objects, no waits or separate
    person state. Overflow drops ONLY optional history. The sole FIFO actor
    validates original lifecycle and age before it can use a retained row.
    """
    def __init__(self, *, include_barriers=False):
        self._lock = threading.Lock()
        self._works = deque()
        self._rows = 0
        self.dropped_rows = 0
        self.include_barriers=include_barriers

    def retain(self, work):
        from dataclasses import replace
        if (type(work) is not QualityWork or work.denial is not None
                or type(work.rows) is not tuple or len(work.rows) > MAX_MEMBERS
                or type(work.swin_extractions) is not tuple or len(work.swin_extractions)>MAX_MEMBERS
                or any(type(e) is not OriginalSwinExtraction for e in work.swin_extractions)):
            return
        raws={e.raw_id for e in work.swin_extractions}
        rows = tuple(r for r in work.rows if r.evidence.lifecycle.tracker_id in raws or
            r.denial is None and r.depth_current and (r.evidence.body_depth is not None or r.evidence.anatomy_depths))
        if not rows and not self.include_barriers:
            return
        with self._lock:
            if any(value.private.original == work.private.original for value in self._works):
                return  # A duplicate cannot refresh F's original availability.
            while self._works and (self._rows + len(rows) > MAX_MEMBERS or len(self._works)>=MAX_MEMBERS):
                self.dropped_rows += len(self._works[0].rows)
                self._rows -= len(self._works.popleft().rows)
            self._works.append(replace(work, rows=rows))
            self._rows += len(rows)

    def take(self):
        with self._lock:
            result = tuple(self._works)
            self._works.clear()
            self._rows = 0
        return result


def _empty_initial_zero_media(work):
    private, continuity = work.private, work.continuity
    original = private.original if type(private) is PrivateAssociationWork else None
    return (type(original) is OriginalCohort and work.denial is None and private.denial is None
        and type(work.raw_media_pts_ns) is int and work.raw_media_pts_ns == 0
        and type(private.media_pts_ns) is int and private.media_pts_ns == 0
        and type(original.pts_ns) is int and original.pts_ns == 0
        and continuity.media_pts_ns is None
        and all(type(value) is int and value == 0 for value in (
            continuity.frame_id, private.canonical_frame, original.original_frame,
            continuity.source_epoch, private.clock_epoch, original.source_clock_epoch))
        and (private.source_id, original.source_id, original.camera_id) ==
            (continuity.source_id, continuity.source_id, continuity.camera_id)
        and type(continuity.source_id) is int and 0 <= continuity.source_id < 3
        and type(original.buffer_scope) is int and original.buffer_scope > 0
        and type(original.slot) is int and 0 <= original.slot < 9
        and type(original.sdk_source_id) is int and original.sdk_source_id >= 0
        and type(original.ntp_ns) is int and original.ntp_ns >= 0
        and all(type(value) is str and 0 < len(value) <= 240 for value in (original.run_id, original.graph_id, original.camera_id))
        and original.timestamp_origin in {"host_receive", "source_sr", "source_pts"}
        and not work.rows and private.owner_retention is None and private.terminated == ()
        and all(value == () for value in (continuity.active_lifecycles, continuity.shadow_lifecycles,
            continuity.terminated_tracker_ids, continuity.tombstones)))


def freeze_quality_work(private, continuity, inputs: Sequence[QualityInput], *, raw_media_pts_ns=_RAW_MEDIA_UNSET):
    if type(continuity) is not TrackingContinuityUpdate:
        raise ValueError("world_quality_actual_continuity_required")
    if len(inputs) > MAX_MEMBERS:
        return QualityWork(private, continuity, (), "world_quality_member_capacity")
    if type(private) is not PrivateAssociationWork:
        return QualityWork(private, continuity, (), "world_quality_original_cohort_unavailable")
    # Only old positive-PTS callers may omit this scalar. Zero requires explicit
    # callback evidence; None is never interchangeable with a zero raw clock.
    raw = private.media_pts_ns if raw_media_pts_ns is _RAW_MEDIA_UNSET and type(private.media_pts_ns) is int and private.media_pts_ns > 0 else raw_media_pts_ns
    if private.denial is not None:
        return QualityWork(private, continuity, (), private.denial, raw if type(raw) is int else None)
    if type(private.original) is not OriginalCohort:
        return QualityWork(private, continuity, (), "world_quality_original_cohort_unavailable", raw if type(raw) is int else None)
    original = private.original
    if type(raw) is not int or raw < 0:
        return QualityWork(private, continuity, (), "world_quality_raw_media_unavailable")
    empty = QualityWork(private, continuity, (), raw_media_pts_ns=raw)
    if raw == 0:
        if type(inputs) in (tuple, list) and not inputs and _empty_initial_zero_media(empty):
            return empty
        return QualityWork(private, continuity, (), "world_quality_zero_media_not_empty_initial", raw)
    try:
        if (type(private.media_pts_ns) is not int or type(original.pts_ns) is not int
                or type(continuity.media_pts_ns) is not int or raw != private.media_pts_ns or raw != original.pts_ns
                or (private.source_id, private.canonical_frame, private.media_pts_ns) != (
                continuity.source_id, continuity.frame_id, continuity.media_pts_ns)):
            raise AcquisitionUnavailable("world_quality_exact_original_cohort_mismatch")
        if (original.source_id, original.camera_id, original.source_clock_epoch) != (
                continuity.source_id, continuity.camera_id, private.clock_epoch):
            raise AcquisitionUnavailable("world_quality_original_source_camera_clock_mismatch")
        _retained, measured = owner_retention_members(private.owner_retention, original,
            continuity.source_epoch, continuity.frame_id)
        receipt = private.owner_retention
        if (receipt.observed_at_us, receipt.terminated) != (continuity.observed_at_us, continuity.terminated_tracker_ids):
            raise AcquisitionUnavailable("world_quality_exact_owner_barrier_mismatch")
    except AcquisitionUnavailable as exc:
        return QualityWork(private, continuity, (), str(exc), raw)
    rows, seen = [], set()
    for value in inputs:
        if type(value) is not QualityInput:
            return QualityWork(private, continuity, (), "world_quality_borrowed_input_refused")
        life = value.lifecycle
        if value.body is not None and (type(value.body) is not OriginalBodySupport or value.body.lifecycle != life):
            return QualityWork(private, continuity, (), "world_quality_body_lifecycle_mismatch")
        key = (life.source_epoch, life.tracker_id, life.generation)
        if key in seen or key not in measured or (life.source_id, life.frame_id,
                life.observed_at_us, life.media_pts_ns) != (continuity.source_id, continuity.frame_id,
                continuity.observed_at_us, continuity.media_pts_ns):
            return QualityWork(private, continuity, (), "world_quality_exact_current_member_mismatch")
        seen.add(key)
        depth = value.depth
        depth_current = bool(depth is not None and depth.is_metric is True and depth.unit == "m"
            and depth.measurement_cached is False and depth.measurement_age_us == 0
            and (depth.source_id, depth.object_id, depth.measurement_frame_id, depth.depth_tensor_frame_id) ==
                (original.source_id, life.tracker_id, original.original_frame, original.original_frame)
            and (depth.measurement_ts_us, depth.depth_tensor_ts_us) == (original.pts_ns // 1000, original.pts_ns // 1000)
            and depth.depth_tensor_age_frames == 0 and depth.depth_tensor_age_us == 0)
        fields = dict(value.world)
        measurement_set, resolution = value.measurements, value.resolution
        denial = None
        if measurement_set is None or resolution is None:
            denial = "world_quality_original_resolver_evidence_unavailable"
        elif measurement_set.cohort != resolution.cohort or measurement_set.cohort.camera_id != original.camera_id or (
                measurement_set.cohort.source_id, measurement_set.cohort.tracker_id,
                measurement_set.cohort.tracker_lifecycle_generation, measurement_set.cohort.frame_id,
                measurement_set.cohort.observed_at_us) != (
                original.source_id, life.tracker_id, life.generation, life.frame_id, life.observed_at_us):
            denial = "world_quality_original_resolver_cohort_mismatch"
        else:
            cohort = measurement_set.cohort
            expected = {"world_frame": cohort.world_frame, "world_frame_revision": cohort.world_revision,
                "world_calibration_sha256": cohort.calibration_revision, "world_transform_sha256": cohort.world_transform_sha256}
            # Exact producer invalidation removes point/covariance/source/frame labels.
            # ORIGINAL resolver evidence remains strict but cannot repair that absent point.
            absent = (fields.get("world_valid") is False and fields.get("world_quality") == "invalid"
                and type(fields.get("world_quality_reason")) is str and bool(fields["world_quality_reason"].strip())
                and (fields.get("world_measurement_accepted") is None or fields.get("world_measurement_accepted") is False)
                and all(fields.get(name) is None for name in ("world", "world_covariance", "world_source",
                    "world_frame", "world_frame_revision", "world_transform_sha256")))
            retained_mismatch = any(fields.get(name) is not None and fields[name] != value for name, value in expected.items())
            if retained_mismatch or not absent and any(fields.get(name) != value for name, value in expected.items()):
                denial = "world_quality_emitted_frame_binding_mismatch"
        covariance = fields.get("world_covariance")
        position = fields.get("world")
        current = bool(denial is None and resolution.status == "measured"
            and fields.get("world_valid") is True and fields.get("world_measurement_accepted") is True
            and type(covariance) is tuple and len(covariance) == 9
            and all(type(value) in (int, float) for value in covariance)
            and type(position) is tuple and len(position) == 3
            and all(type(value) in (int, float) for value in position)
            and fields.get("world_source") not in ("anchor_hold", "cv_prediction", "image_motion_prediction"))
        if current and resolution.selected_kind == "registered_depth" and not depth_current:
            denial, current = "world_quality_depth_not_exact_current", False
        rows.append(QualityRow(original, value, depth_current, current, denial))
    return QualityWork(private, continuity, tuple(rows), raw_media_pts_ns=raw)


class OrderedQualityWorldIngress:
    """Existing publication actor's bounded ORIGINAL-input adapter, not a ledger.

    Identity owns person membership; World owns selection/filtering and the one
    transaction. The only retained rows here are exact source observations for
    ordered removal, three source clocks and the one staged compact cohort.
    No cached camera measurement is inserted into another camera's preparation.
    """

    def __init__(self, world, identity, bindings, *, origin_bindings, projective_bridge=None):
        from noesis.identity_association_service import IdentityAssociationService
        from noesis_core.world_service import CanonicalWorldService
        from reid.mv3dt_person_association import SourceBinding
        if (type(world) is not CanonicalWorldService or type(identity) is not IdentityAssociationService
                or type(bindings) is not tuple or not 1 <= len(bindings) <= 3
                or any(type(b) is not SourceBinding for b in bindings)
                or len({b.source_id for b in bindings}) != len(bindings) or not callable(origin_bindings)):
            raise ValueError("world_quality_actual_service_bindings_required")
        self.world, self.identity = world, identity
        self.bindings = {b.source_id: b for b in bindings}
        self.origin_bindings = origin_bindings
        self.originals = {}  # <=128 ORIGINAL SourceObservation removal receipts, no person keys.
        self.times, self.watermarks = {}, {}
        self._staged = None
        self._expired_preparation = None
        self._expiry_retry_used = False
        self._context = None
        self.projective_bridge = projective_bridge
        self._swin_history=OriginalQualityRetention(include_barriers=True)
        self.swin_pair_examples={}
        self._image_source_stops = {}  # <=3 actual terminal sources, graph-bound/run.
        self._image_source_stop_bindings = {}
        self._image_source_stop_results = {}
        self._lifecycle_refusal_logged = False  # ONE failure-only diagnostic/run.

    def _diagnose_lifecycle_refusal(self, result, original, work, retained, measured, lineage):
        """Observe the exact failed boundary; never recompute or alter admission."""
        if self._lifecycle_refusal_logged or result.reason not in (
                'lifecycle_frontier', 'closed_lifecycle', 'retired_lifecycle_generation'):
            return
        self._lifecycle_refusal_logged = True
        try:
            member = original.stamp.member
            key = (member.source_epoch, member.raw_id, member.lifecycle_generation)
            owner = lineage.get(key)
            receipt = work.private.owner_retention
            row = next((r for r in work.rows if
                (r.evidence.lifecycle.source_epoch, r.evidence.lifecycle.tracker_id,
                 r.evidence.lifecycle.generation) == key), None)
            kernel = self.identity.ledger.observation_refusal_diagnostic()
            if kernel is not None and kernel.get('stamp') != asdict(original.stamp):
                kernel = None  # Never graft another failure's observer snapshot.
            note = {'contract': 'noesis.world_quality.lifecycle_refusal.v1',
                'reason': result.reason, 'disposition': result.disposition,
                'kernel': kernel, 'original': asdict(row.original) if row else None,
                'native_observation': asdict(row.evidence.lifecycle) if row else None,
                'submitted_native_state': 'ACTIVE',
                'native_birth': asdict(owner.original_birth) if owner is not None else None,
                'native_lifecycle_started_frame': owner.lifecycle_started_frame if owner else None,
                'retained': key in retained, 'measured': key in measured,
                'owner_receipt': {name: getattr(receipt, name) for name in (
                    'run_id', 'graph_id', 'source_id', 'source_epoch', 'clock_epoch',
                    'frame_id', 'observed_at_us', 'media_pts_ns', 'owner_grace_ns')},
                'active_row': next((list(v) for v in receipt.active if v[:2] == key[1:]), None),
                'shadow': key[1:] in receipt.shadow,
                'terminated': member.raw_id in receipt.terminated}
            encoded = json.dumps(note, allow_nan=False, separators=(',', ':'))
            if len(encoded.encode()) > 8192:
                encoded = json.dumps({'contract': note['contract'], 'reason': result.reason,
                                      'diagnostic_unavailable': 'byte_bound'})
            logger.error('World quality lifecycle refusal: %s', encoded)
        except Exception:
            # The typed original refusal is raised unchanged by checked().
            pass

    def close_image_region_run(self):
        from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
        if type(self.projective_bridge) is ImageRegionQualityBridge:
            self.identity.close_image_region_run()

    def stop_image_region_source(self, source_id, *, graph_id):
        """Ordered real-stop entrypoint; silence/shadow must NEVER call this."""
        from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
        if type(self.projective_bridge) is not ImageRegionQualityBridge:
            raise ValueError('image_region_stop_requires_image_mode')
        from noesis_core.world_service import WorldPublicationRefused
        if type(source_id) is not int or source_id not in self.bindings:
            raise WorldPublicationRefused('image_region_stop_source_binding')
        binding=self.bindings[source_id]
        old=self._image_source_stops.get(source_id)
        if old is not None and old!=graph_id:
            raise WorldPublicationRefused('image_region_owner_availability_binding')
        if self._image_source_stop_bindings.get(source_id)==binding:
            return self._image_source_stop_results[source_id]
        result=self.identity.stop_image_region_source(binding, graph_id=graph_id)
        if result.disposition!='observed' or result.reason!='image_region_source_stopped':
            raise WorldPublicationRefused.from_association_result(result)
        self._image_source_stops[source_id]=graph_id
        self._image_source_stop_bindings[source_id]=binding
        self._image_source_stop_results[source_id]=result
        self.projective_bridge.stop_source(source_id)
        return result

    def _diagnose_swin_pairs(self,cache):
        """Bounded scalar provenance only, no decision/counter-changing cache.use."""
        now=time.monotonic()
        originals=tuple(self.originals.values())
        remaining=32  # Diagnostic work as well as retained output stays bounded.
        for left in (o for o in originals if o.stamp.member.camera_id=='kitchen'):
            for right in (o for o in originals if o.stamp.member.camera_id=='family-room'):
                if not remaining:return
                remaining-=1
                members=tuple(o.stamp.member for o in (left,right))
                values=tuple(cache.members[m.source_id].get((m.source_epoch,m.raw_id,m.lifecycle_generation)) for m in members)
                ready=all(v is not None and v[3] is not None for v in values)
                key=members
                current_ingresses=tuple(self.projective_bridge.ingresses.get(m)
                    if self.projective_bridge is not None else None for m in members)
                current_prequeue=tuple(v[1] if v is not None and v[0]==o.stamp else None
                    for o,v in zip((left,right),current_ingresses))
                sample={'members':tuple((m.source_id,m.source_epoch,m.raw_id,m.lifecycle_generation) for m in members),
                    'current_frames':tuple(o.stamp.frame_id for o in (left,right)),
                    'current_source_s':tuple(o.stamp.acquired_at_s for o in (left,right)),
                    'current_media_s':tuple(o.stamp.media_time_s for o in (left,right)),
                    'current_prequeue_monotonic_s':current_prequeue,
                    'current_availability_ages_s':tuple(None if v is None else now-v for v in current_prequeue),
                    'evaluation_monotonic_s':now,'original_appearance_present':ready}
                if ready:
                    a,b=(v[0].vector for v in values)
                    sample.update(cosine=sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b)),
                        original_frames=tuple(v[3].stamp.frame_id for v in values),
                        original_source_s=tuple(v[3].acquired_at_s for v in values),
                        source_ages_s=tuple(o.stamp.acquired_at_s-v[3].acquired_at_s for o,v in zip((left,right),values)),
                        original_prequeue_monotonic_s=tuple(v[0].ingress_monotonic_s for v in values),
                        availability_ages_s=tuple(None if v[0].ingress_monotonic_s is None else now-v[0].ingress_monotonic_s for v in values))
                self.swin_pair_examples.pop(key,None)
                if len(self.swin_pair_examples)>=32:
                    self.swin_pair_examples.pop(next(iter(self.swin_pair_examples)))
                self.swin_pair_examples[key]=sample

    def _bind_retained_swin(self,cache,current):
        from reid.mv3dt_person_association import SourceObservation,ObservationStamp
        works=self._swin_history.take()
        selected=[]
        for w in works:
            if w.private.original.source_id==current.source_id:selected.append(w)
            else:self._swin_history.retain(w)
        selected.sort(key=lambda w:w.continuity.frame_id)
        if not selected:return
        endpoint=selected[-1]
        if endpoint.private.original!=current:return
        now=time.monotonic()
        for w in selected[:-1]:
            f=w.private.original
            for extraction in w.swin_extractions:
                try:
                    row=next(r for r in w.rows if r.evidence.lifecycle.tracker_id==extraction.raw_id)
                    life=row.evidence.lifecycle
                    key=(life.source_epoch,life.tracker_id,life.generation)
                    owner=cache.owners[current.source_id].get(key)
                    extent=[v for v in selected if w.continuity.frame_id<=v.continuity.frame_id<=endpoint.continuity.frame_id]
                    if (owner is None or extraction.original!=f
                            or len(extent)!=endpoint.continuity.frame_id-w.continuity.frame_id+1
                            or any(v.denial or v.private.denial or
                                (v.private.original.run_id,v.private.original.graph_id,v.private.clock_epoch)!=
                                (current.run_id,current.graph_id,current.source_clock_epoch)
                                or key not in owner_retention_members(v.private.owner_retention,v.private.original,
                                    v.continuity.source_epoch,v.continuity.frame_id)[1] for v in extent)
                            or extraction.ingress_monotonic_s is None or w.ingress_monotonic_s is None
                            or not extraction.ingress_monotonic_s<=w.ingress_monotonic_s<=endpoint.ingress_monotonic_s<=now):
                        raise AcquisitionUnavailable('swin_retained_extent_unproven')
                    observation=next(o for k,o in self.originals.items() if
                        (k.source_epoch,k.raw_id,k.lifecycle_generation)==key and k.source_id==current.source_id)
                    member=observation.stamp.member
                    token=hashlib.sha256(repr((f,member,owner.lifecycle_started_frame)).encode()).hexdigest()
                    stamp=ObservationStamp(member,life.frame_id,'native-quality:'+token,f.pts_ns/1e9,
                        observation.stamp.clock_domain,f.pts_ns/1e9,observation.stamp.clock_basis,owner.lifecycle_started_frame)
                    cache.bind_retained_original(extraction,SourceObservation(stamp,row.evidence.household),
                        current,owner,self.identity.ledger.policy.appearance_profile,now)
                except (AcquisitionUnavailable,StopIteration,TypeError,ValueError):
                    cache.producer_event('retained_F_refused',f,current_frame=current.original_frame)

    def bind_published_swin(self, cache, original):
        """Bind actual original appearance only AFTER its genuine owner publication.

        A source may have no newly extracted embedding on this frame. That is
        absence of fresh evidence, not permission to retag the retained tensor.
        No extraction, appearance scoring or physical joining happens here.
        """
        if type(cache) is not OriginalSwinCache or type(original) is not OriginalCohort:
            raise AcquisitionUnavailable("world_quality_original_swin_cache_unavailable")
        bound = []
        for observation in self.originals.values():
            stamp, member = observation.stamp, observation.stamp.member
            if (member.source_id != original.source_id or member.run_id != original.run_id
                    or member.camera_id != original.camera_id or stamp.media_time_s != original.pts_ns / 1e9):
                continue
            # An exact current copy is required. ``exact`` never returns an
            # earlier retained member extraction as a new original frame.
            if cache.exact(original, member.raw_id) is not None:
                bound.append(cache.bind_observation(original, observation))
        self._bind_retained_swin(cache,original)
        cache.finish_frozen(original)
        return tuple(bound)

    def stage(self, work, tracks, *, historical_quality=(), original_depth_resolver=None):
        """Native calls once BEFORE publisher.prepare; bundles are SDK-neutral.

        An explicitly supplied bundle contains only actual detached current
        source cohorts, <=3 distinct sources. Repeated-source mux frames remain
        separate FIFO transactions, never collapsed into a last-seen join.
        """
        from dataclasses import replace
        from noesis_core.world_service import WorldPublicationRefused
        from reid.mv3dt_person_association import (
            MemberKey, ObservationStamp, SourceObservation, MemberRemoval,
            SpatialClockInterval, SpatialClockContext, SpatialLifecycleReceipt,
        )
        from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
        image_mode = type(self.projective_bridge) is ImageRegionQualityBridge
        image_sources = self.projective_bridge.cameras if image_mode else ()
        if self._staged is not None:
            raise RuntimeError("world_quality_ordered_cohort_already_pending")
        works = work if type(work) is tuple else (work,)
        track_groups = tracks if type(work) is tuple else ((work.continuity.source_id, tracks),)
        if (not 1 <= len(works) <= 3 or any(type(w) is not QualityWork for w in works)
                or len({w.continuity.source_id for w in works}) != len(works)
                or len(track_groups) != len(works)):
            raise WorldPublicationRefused("world_quality_current_source_cohort_bounds")
        sources = tuple(w.continuity.source_id for w in works)
        if tuple(source for source, _ in track_groups) != sources:
            raise WorldPublicationRefused("world_quality_original_track_sources")
        try:
            origins = self.origin_bindings()  # ONCE in this ordered actor, before any authority mutation.
        except Exception as exc:
            raise WorldPublicationRefused(str(exc) or "world_quality_original_lineage_provider_failed") from exc
        if not isinstance(origins, Mapping):
            raise WorldPublicationRefused("world_quality_original_lineage_unavailable")
        # Validate the ENTIRE detached input before touching the real owner.
        plans = []
        measured_by_source = {}
        for w, (_, rows) in zip(works, track_groups, strict=True):
            if w.denial:
                raise WorldPublicationRefused(w.denial)
            if type(w.private) is not PrivateAssociationWork or type(w.private.original) is not OriginalCohort:
                raise WorldPublicationRefused("world_quality_original_cohort_unavailable")
            if w.private.denial is not None:
                raise WorldPublicationRefused(w.private.denial)
            availability = w.owner_availability
            if availability is not None and (
                    type(availability) is not OwnerSourceAvailability
                    or availability.original != w.private.original
                    or availability.owner != w.private.owner_retention
                    or availability.ingress_monotonic_s != w.ingress_monotonic_s
                    or availability.checked_monotonic_s != w.ingress_monotonic_s):
                raise WorldPublicationRefused('world_quality_owner_availability_binding')
            source, original = w.continuity.source_id, w.private.original
            zero_media = _empty_initial_zero_media(w)
            binding = self.bindings.get(source)
            if binding is None or original.run_id != self.identity.run_id:
                raise WorldPublicationRefused("world_quality_original_run_source_binding")
            if ((w.private.source_id, w.private.canonical_frame, w.private.clock_epoch) !=
                    (source, w.continuity.frame_id, original.source_clock_epoch)
                    or original.source_id != source or original.camera_id != binding.camera_id or not zero_media and
                    (type(w.raw_media_pts_ns) is not int or w.raw_media_pts_ns <= 0
                     or w.raw_media_pts_ns != w.private.media_pts_ns
                     or w.raw_media_pts_ns != original.pts_ns or original.pts_ns != w.continuity.media_pts_ns)):
                raise WorldPublicationRefused("world_quality_original_camera_media_binding")
            watermark = (w.continuity.source_epoch, original.source_clock_epoch,
                         w.continuity.frame_id, original.pts_ns)
            previous = self.watermarks.get(source)
            if previous is not None and (watermark[:2] < previous[:2] or watermark[:2] == previous[:2]
                    and (watermark[2] <= previous[2] or watermark[3] <= previous[3])):
                raise WorldPublicationRefused("world_quality_fifo_order_invalid")
            if previous is not None and watermark[1] != previous[1] and watermark[0] == previous[0]:
                raise WorldPublicationRefused("world_quality_clock_requires_canonical_epoch_reset")
            if w.continuity.source_epoch != binding.source_epoch:
                if w.continuity.source_epoch <= binding.source_epoch:
                    raise WorldPublicationRefused("world_quality_canonical_epoch_reordered")
                binding = replace(binding, source_epoch=w.continuity.source_epoch,
                    clock_domain=f"source-media:{original.run_id}:{source}:{w.continuity.source_epoch}:{original.source_clock_epoch}")
            retained = frozenset()
            if not zero_media:
                retained, measured = owner_retention_members(w.private.owner_retention, original,
                    w.continuity.source_epoch, w.continuity.frame_id)
                measured_by_source[source] = measured
            else:
                measured_by_source[source] = frozenset()
            if len(rows) != len(w.rows) or len(rows) > MAX_MEMBERS:
                raise WorldPublicationRefused("world_quality_original_row_count")
            by_key = {(r.evidence.lifecycle.tracker_id, r.evidence.lifecycle.generation): r for r in w.rows}
            copies, observations = [], []
            lineage = origins[source] if source in origins else None
            if not isinstance(lineage, Mapping):
                raise WorldPublicationRefused("world_quality_original_lineage_unavailable")
            if zero_media and (type(rows) not in (tuple, list) or rows or previous is not None or source in self.times or lineage
                    or binding.source_epoch != 0 or any(member.source_id == source for member in self.originals)):
                raise WorldPublicationRefused("world_quality_zero_media_source_not_initial_empty")
            seen = set()
            for track in rows:
                key = (track.get("tracker_id"), track.get("tracker_lifecycle_generation"))
                r = by_key.get(key)
                if r is None or key in seen:
                    raise WorldPublicationRefused("world_quality_original_row_missing")
                seen.add(key)
                life, fields = r.evidence.lifecycle, dict(r.evidence.world)
                try:
                    if household_from_final_track(track) != r.evidence.household:
                        raise ValueError("world_quality_final_household_changed")
                except ValueError as exc:
                    raise WorldPublicationRefused(str(exc)) from exc
                if any(_scalar(track.get(name)) != value for name, value in r.evidence.world):
                    raise WorldPublicationRefused("world_quality_original_emitted_row_changed")
                if r.denial and r.denial != "world_quality_depth_not_exact_current":
                    if r.denial in {"world_quality_emitted_frame_binding_mismatch", "world_quality_original_resolver_cohort_mismatch"}:
                        raise _binding_refusal(r, r.denial)
                    raise WorldPublicationRefused(r.denial)
                if fields.get("calibration_revision") not in (None, binding.calibration_revision):
                    raise _binding_refusal(r, "world_quality_source_frame_revision_changed")
                if fields.get("world_valid") is True and fields.get("calibration_revision") is None:
                    raise WorldPublicationRefused("world_quality_original_source_frame_revision_unavailable")
                if source in image_sources and (life.source_epoch,*key) not in measured_by_source[source]:
                    # Emitted SDK shadow rows are diagnostics, not a fresh
                    # observation/body/position, even with a cached valid point.
                    continue
                owner = lineage.get((life.source_epoch, life.tracker_id, life.generation))
                if owner is not None:
                    if (type(owner) is not SwinOwnerBinding or type(owner.original_birth) is not OriginalCohort
                            or owner.member != (life.source_epoch, *key)
                            or (owner.original_birth.run_id, owner.original_birth.graph_id,
                                owner.original_birth.source_id, owner.original_birth.source_clock_epoch) !=
                               (original.run_id, original.graph_id, source, original.source_clock_epoch)
                            or type(owner.lifecycle_started_frame) is not int
                            or not 0 <= owner.lifecycle_started_frame <= life.frame_id
                            or owner.original_birth.original_frame > original.original_frame
                            or owner.original_birth.pts_ns > original.pts_ns):
                        raise WorldPublicationRefused("world_quality_original_birth_binding")
                    started = owner.lifecycle_started_frame
                elif life.disposition == "new":
                    # Genuine canonical first-admitted owner cohort, NOT SDK birth.
                    started = w.continuity.frame_id
                else:
                    raise WorldPublicationRefused("world_quality_original_canonical_birth_unavailable")
                member = MemberKey(original.run_id, binding.camera_id, source, life.source_epoch, *key)
                token = hashlib.sha256(repr((original, member, started)).encode()).hexdigest()
                stamp = ObservationStamp(member, life.frame_id, "native-quality:" + token,
                    original.pts_ns / 1e9, binding.clock_domain, original.pts_ns / 1e9, binding.clock_basis, started)
                observations.append(SourceObservation(stamp, r.evidence.household))
                copy = dict(track)
                copy.update(calibration_revision=fields.get("calibration_revision"), lifecycle_started_frame=started,
                    world_measurement_accepted=r.current_measured)
                copies.append(copy)  # Original point/covariance/confidence/time untouched.
            plans.append((w, binding, watermark, retained, tuple(copies), tuple(observations)))
        prospective = set(self.originals)
        for w, binding, _watermark, retained, _rows, observations in plans:
            prospective = {key for key in prospective if key.source_id != binding.source_id or
                (binding == self.bindings[binding.source_id] and
                 (key.source_epoch, key.raw_id, key.lifecycle_generation) in retained)}
            prospective.update(o.stamp.member for o in observations)
        if len(prospective) > MAX_MEMBERS:
            raise WorldPublicationRefused("world_quality_original_member_capacity")
        withdrawals, retirements = set(), set()
        def checked(result, *, progress=False):
            if result.disposition not in {"observed", "removed", "reset"} and not (progress
                    and result.disposition == "unresolved" and result.reason == "current_cross_source_proof_required"):
                raise WorldPublicationRefused.from_association_result(result)
            withdrawals.update(result.withdrawn_members); retirements.update(getattr(result, "retired_keys", ()))
        for w, binding, watermark, retained, _rows, observations in plans:
            source = binding.source_id
            if binding != self.bindings[source]:
                checked(self.identity.reset_source(binding))
                self.originals = {key: value for key, value in self.originals.items() if key.source_id != source}
                self.times.pop(source, None)
                self.bindings[source] = binding
                if source in self._image_source_stops:
                    # Canonical reset is real, but cannot invent reactivation of
                    # a source whose native stop was terminal in this static run.
                    self.stop_image_region_source(source,graph_id=self._image_source_stops[source])
            def observe_original(original):
                observed = self.identity.observe_spatial(original, SpatialLifecycleReceipt(
                    "native-active:" + original.stamp.observation_id, original.stamp, "ACTIVE"))
                self._diagnose_lifecycle_refusal(observed, original, w, retained,
                    measured_by_source[source], origins[source])
                checked(observed, progress=True)
                self.originals[original.stamp.member] = original
            preobserved = set()
            pending_removal = any(member.source_id == source and
                (member.source_epoch,member.raw_id,member.lifecycle_generation) not in retained
                for member in self.originals)
            if pending_removal:
                # The complete CURRENT cohort precedes its unrelated end-of-
                # cohort withdrawals. Admit ONLY a genuinely new current birth
                # against the PRIOR kernel frontier/history; never move an old
                # birth or a reused raw ID ahead of its required removal.
                occupied_raw = {m.raw_id for m in self.originals if m.source_id == source}
                lifecycles = {(r.evidence.lifecycle.tracker_id,r.evidence.lifecycle.generation):
                              r.evidence.lifecycle for r in w.rows}
                for original in observations:
                    stamp = original.stamp
                    member = stamp.member
                    life = lifecycles[(member.raw_id,member.lifecycle_generation)]
                    if (member not in self.originals and member.raw_id not in occupied_raw
                            and life.disposition == "new"
                            and life.previous_media_pts_ns is None
                            and stamp.lifecycle_started_frame == stamp.frame_id == w.continuity.frame_id
                            and (member.source_epoch,member.raw_id,member.lifecycle_generation)
                                in measured_by_source[source]):
                        observe_original(original)
                        preobserved.add(member)
            for member, previous in tuple(self.originals.items()):
                if member.source_id == source and (member.source_epoch, member.raw_id, member.lifecycle_generation) not in retained:
                    kind = "termination" if member.raw_id in w.private.terminated else "absence"
                    removal = MemberRemoval(f"quality-barrier:{watermark}:{member}", previous.stamp,
                        kind, w.continuity.frame_id, w.private.media_pts_ns / 1e9,
                        binding.clock_domain, w.private.media_pts_ns / 1e9)
                    checked(self.identity.withdraw(removal))
                    del self.originals[member]
            if source in image_sources:
                from reid.mv3dt_person_association import SpatialLifecycleUpdate
                for member, previous in self.originals.items():
                    member_key=(member.source_epoch,member.raw_id,member.lifecycle_generation)
                    if member.source_id==source and member_key in retained and member_key not in measured_by_source[source]:
                        token=hashlib.sha256(repr((watermark,member,previous.stamp)).encode()).hexdigest()
                        checked(self.identity.update_spatial_lifecycle(SpatialLifecycleUpdate(
                            'quality-retained-unmeasured:'+token,previous.stamp,w.continuity.frame_id,
                            w.private.media_pts_ns/1e9,binding.clock_domain,w.private.media_pts_ns/1e9)))
                        self.projective_bridge.counts['retained_unmeasured_updates'] += 1
            for original in observations:
                if original.stamp.member not in preobserved:
                    observe_original(original)
            at = w.private.media_pts_ns / 1e9
            self.times[source] = at
            self.watermarks[source] = watermark
        self._context = SpatialClockContext(tuple(SpatialClockInterval(self.bindings[source], at, at, at, at)
            for source, at in sorted(self.times.items())))
        if self.projective_bridge is not None:
            for frozen in historical_quality+works:self._swin_history.retain(frozen)
            self._diagnose_swin_pairs(self.projective_bridge.cache)
            if original_depth_resolver is not None:
                combined=OriginalQualityRetention()
                for original_work in historical_quality + original_depth_resolver(self.originals,self._context,time.monotonic()):
                    combined.retain(original_work)
                historical_quality=combined.take()
            if historical_quality:
                self.projective_bridge.ingest_historical_quality(historical_quality, self.originals,
                    self._context, time.monotonic())
            result = self.projective_bridge.step(plans,self.originals,self._context,self.identity)
            withdrawals.update(result.withdrawn_members)
            retirements.update(result.retired_keys)
        self._staged = (plans, tuple(sorted(withdrawals)), tuple(sorted(retirements)))

    def prepare(self, source_id, tracks, *, metadata):
        from noesis_core.world_service import WorldPublicationRefused
        if self._staged is None:
            raise WorldPublicationRefused("world_quality_exact_fifo_cohort_unavailable")
        if self._expiry_retry_used:
            error = self._expired_preparation[1]
            try:
                current = self.identity.spatial_memberships(at_interval=self._context)
            except Exception as changed:
                self._staged = self._expired_preparation = None
                self._expiry_retry_used = False
                if isinstance(changed,ValueError):raise WorldPublicationRefused(str(changed)) from changed
                raise
            if (error.context is not self._context or len(current) != len(error.memberships)
                    or any(new is not old for new,old in zip(current,error.memberships))):
                self._staged = self._expired_preparation = None
                self._expiry_retry_used = False
                raise WorldPublicationRefused("spatial_membership_revision_changed")
        plans, withdrawals, retirements = self._staged
        trigger = next((p for p in plans if p[0].continuity.source_id == source_id), None)
        if trigger is None or (metadata.get("camera_id"), metadata.get("observed_at_us"), metadata.get("frame_id"),
                metadata.get("source_epoch"), metadata.get("media_pts_ns")) != (
                trigger[0].continuity.camera_id, trigger[0].continuity.observed_at_us, trigger[0].continuity.frame_id,
                trigger[0].continuity.source_epoch, trigger[0].continuity.media_pts_ns):
            raise WorldPublicationRefused("world_quality_exact_publisher_cohort_mismatch")
        if _empty_initial_zero_media(trigger[0]) and tracks:
            raise WorldPublicationRefused("world_quality_zero_media_source_not_initial_empty")
        measurements = []
        try:
            for w, _binding, _watermark, _retained, rows, originals in plans:
                meta = dict(metadata) if w.continuity.source_id == source_id else {
                    "camera_id": w.continuity.camera_id, "frame_id": w.continuity.frame_id,
                    "source_epoch": w.continuity.source_epoch, "observed_at_us": w.continuity.observed_at_us,
                    "media_pts_ns": w.continuity.media_pts_ns}
                measurements.append(self.world.prepare(w.continuity.source_id, rows, metadata=meta,
                    physical_observations=originals))
            # Final SAME revision issuance, after all owner barriers/mutations.
            members = self.identity.spatial_memberships(at_interval=self._context)
            return self.world.prepare_spatial(source_id, tuple(measurements), members,
                at_interval=self._context, withdrawn_members=withdrawals, retired_keys=retirements,
                attribution_mode="physical_only_quarantine")
        except Exception as exc:
            for measurement in measurements: self.world.discard(measurement)
            self._staged = self._expired_preparation = None
            self._expiry_retry_used = False
            if isinstance(exc, ValueError): raise WorldPublicationRefused(str(exc)) from exc
            raise

    def commit(self, prepared):
        from noesis_core.world_service import WorldPreparedHoldExpired
        keep = False
        try:
            return self.world.commit(prepared, association_at_interval=self._context)
        except WorldPreparedHoldExpired as error:
            if not self._expiry_retry_used and error.prepared is prepared and self._staged is not None:
                self._expired_preparation = (prepared,error,self._staged)
                keep = True
            raise
        finally:
            if not keep:
                self._staged = self._expired_preparation = None
                self._expiry_retry_used = False

    def retry_expired_preparation(self, prepared, error):
        """One SAME detached plan, only after its exact uncommitted expiry."""
        pending = self._expired_preparation
        if (self._expiry_retry_used or pending is None or pending[0] is not prepared
                or pending[1] is not error or pending[2] is not self._staged):
            return False
        self._expiry_retry_used = True
        return True

    def discard(self, prepared):
        try: return self.world.discard(prepared)
        finally:
            self._staged = self._expired_preparation = None
            self._expiry_retry_used = False
