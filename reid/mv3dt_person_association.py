"""Bounded, SDK-neutral physical association; this is not household identity.

The caller owns extraction and validation of native/lifecycle/measurement evidence.
This module checks coherent scalar provenance against explicit source bindings; it
does not authenticate evidence from untrusted callers. No raw ID equality grants a
link. A downstream owner must consume an accepted *current* LinkReceipt, not just
person-key equality. Updating either observation revokes that receipt immediately.

All input/output records are immutable. The single-writer ledger has no I/O,
models, galleries, SDK imports or host clock. Superseded detail compacts only
behind monotonic cohort/lifecycle frontiers; current authority/grace is protected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict, field
from functools import lru_cache
import json
from hashlib import sha256
from math import hypot, isfinite, ulp, nextafter


def _text(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= 256 and value == value.strip()


def _integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value < 2**64


def _number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


_IMAGE_DECISION_SNAPSHOT_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class ImageRegionDecisionSnapshot:
    """Private immutable completed-cohort JSON, never a live owner/capability.

    Only the ledger seals its already validated encoded records. Materializing
    creates a detached graph and applies the existing diagnostic detail budget.
    """
    envelope_json: str
    record_json: tuple[str, ...]
    record_kinds: tuple[str, ...]
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def _freeze(cls, envelope, records, kinds, seal):
        if seal is not _IMAGE_DECISION_SNAPSHOT_SEAL:
            raise ValueError('unissued image diagnostic snapshot')
        value = object.__new__(cls)
        for name, item in (('envelope_json', envelope), ('record_json', records),
                           ('record_kinds', kinds), ('_seal', seal)):
            object.__setattr__(value, name, item)
        value.check_immutable()
        return value

    def check_immutable(self):
        if (self._seal is not _IMAGE_DECISION_SNAPSHOT_SEAL
                or type(self.envelope_json) is not str or len(self.envelope_json) > 4096
                or type(self.record_json) is not tuple or len(self.record_json) > 8
                or type(self.record_kinds) is not tuple or len(self.record_kinds) != len(self.record_json)
                or any(type(s) is not str or len(s) > 1048576 for s in self.record_json)
                or any(type(k) is not str or k not in ('transition', 'latest_evaluation',
                    'transition_and_latest_evaluation') for k in self.record_kinds)):
            raise ValueError('invalid immutable image diagnostic snapshot')

    def materialize(self) -> dict:
        self.check_immutable()
        result = json.loads(self.envelope_json[:-1] + ',"transitions":[' + ','.join(self.record_json) + ']}')
        for record, kind in zip(result['transitions'], self.record_kinds):
            record['record_kind'] = kind
        budget = 8192
        # Creation/refusal is small scalar evidence; reserve it before verbose
        # member continuity details so the actual failed guard remains visible.
        for record in reversed(result['transitions']):
            note=record.get('split_custody_creation')
            if note is not None:
                size=len(json.dumps(note,separators=(',', ':'),allow_nan=False).encode())
                if size<=budget:budget-=size
                else:record['split_custody_creation']={'failure':'diagnostic_byte_bound'}
        names = ('floor_advance_note', 'paired_native_seed', 'owner_availability',
                 'live_native_checks', 'custody_check', 'lineage')
        for record in reversed(result['transitions']):
            seeds = {}
            for row in record['members']:
                if row.get('paired_native_seed'):
                    token = json.dumps(row['paired_native_seed'], sort_keys=True, separators=(',', ':'))
                    if token in seeds:
                        row['paired_native_seed'] = {'member_ref': seeds[token]}
                    else:
                        seeds[token] = row['member']['source_id']
                extras = {k: row[k] for k in names if k in row}
                size = len(json.dumps(extras, separators=(',', ':'), allow_nan=False).encode())
                if size <= budget:
                    budget -= size
                else:
                    for k in names:
                        row.pop(k, None)
                    row['continuity_detail_status'] = 'byte_bound_omitted'
        return result


def _clock_basis(value: object) -> bool:
    return type(value) is str and value in {"ntp", "synchronized-media"}


@dataclass(frozen=True, slots=True, order=True)
class MemberKey:
    run_id: str
    camera_id: str
    source_id: int
    source_epoch: int
    raw_id: int
    lifecycle_generation: int


@dataclass(frozen=True, slots=True, order=True)
class PhysicalPersonKey:
    """Opaque run-local allocation, unrelated to a native ID or household SID."""

    run_id: str
    serial: int


@dataclass(frozen=True, slots=True)
class SourceBinding:
    camera_id: str
    source_id: int
    source_epoch: int
    calibration_revision: str
    transform_id: str
    clock_domain: str
    clock_basis: str


@dataclass(frozen=True, slots=True)
class AppearanceProfile:
    model_id: str
    layer_id: str
    dimensions: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class AssociationPolicy:
    sources: tuple[SourceBinding, ...]
    target_frame: str
    world_revision: str
    appearance_model: str
    appearance_profile: AppearanceProfile
    overlap_pairs: tuple[tuple[str, str], ...]
    max_entities: int = 64
    max_history: int = 128
    max_age_s: float = 0.35
    max_distance_m: float = 1.25
    min_similarity: float = 0.60
    max_return_grace_s: float = 0.35

    def __post_init__(self) -> None:
        if (type(self.sources) is not tuple or not 1 <= len(self.sources) <= 3
                or any(type(b) is not SourceBinding for b in self.sources)):
            raise ValueError("one to three explicit immutable source bindings required")
        cameras: set[str] = set()
        sources: set[int] = set()
        for binding in self.sources:
            if (binding.camera_id not in {"kitchen", "family-room", "living-room"}
                    or not _integer(binding.source_id) or not _integer(binding.source_epoch)
                    or not all(_text(v) for v in (binding.calibration_revision,
                                                  binding.transform_id, binding.clock_domain))
                    or not _clock_basis(binding.clock_basis)
                    or binding.camera_id in cameras or binding.source_id in sources):
                raise ValueError("invalid or duplicate source binding")
            cameras.add(binding.camera_id)
            sources.add(binding.source_id)
        if not all(_text(v) for v in (self.target_frame, self.world_revision, self.appearance_model)):
            raise ValueError("target frame/revision and selected appearance model required")
        if (type(self.appearance_profile) is not AppearanceProfile
                or self.appearance_profile.model_id != self.appearance_model
                or not all(_text(v) for v in (self.appearance_profile.model_id,
                                              self.appearance_profile.layer_id, self.appearance_profile.fingerprint))
                or not _integer(self.appearance_profile.dimensions, 1)):
            raise ValueError("explicit selected model/layer/dimension/fingerprint required")
        if (type(self.overlap_pairs) is not tuple or len(self.overlap_pairs) > 1
                or any(type(pair) is not tuple or len(pair) != 2
                       or not all(type(camera) is str for camera in pair)
                       or set(pair) != {"kitchen", "family-room"}
                       or not set(pair) <= cameras for pair in self.overlap_pairs)):
            raise ValueError("only the exact enabled Kitchen/Family overlap is permitted")
        if (not _integer(self.max_entities, 1) or self.max_entities > 64
                or not _integer(self.max_history, 1)
                or self.max_history > 2 * self.max_entities):
            raise ValueError("entity/history limits exceed the admitted envelope")
        if (not all(_number(v) for v in (self.max_age_s, self.max_distance_m,
                                        self.min_similarity, self.max_return_grace_s))
                or not 0 < self.max_age_s <= 0.35 or not 0 < self.max_distance_m <= 1.25
                or not 0.60 <= self.min_similarity <= 1.0
                or not 0 <= self.max_return_grace_s <= 2.0):
            raise ValueError("invalid or weakened proof/grace limits")


@dataclass(frozen=True, slots=True)
class SpatialAssociationPolicy:
    """Explicit nonmetric route; no invented common target or admission receipt."""

    sources: tuple[SourceBinding, ...]
    appearance_profile: AppearanceProfile
    overlap_pairs: tuple[tuple[str, str], ...]
    max_entities: int = 64
    max_history: int = 128
    max_age_s: float = .35
    max_distance_m: float = 1.25
    min_similarity: float = .60
    max_return_grace_s: float = .35

    @property
    def appearance_model(self) -> str:
        return self.appearance_profile.model_id

    def __post_init__(self) -> None:
        if (type(self.sources) is not tuple or not 1 <= len(self.sources) <= 3
                or any(type(b) is not SourceBinding for b in self.sources)
                or len({b.source_id for b in self.sources}) != len(self.sources)
                or len({b.camera_id for b in self.sources}) != len(self.sources)):
            raise ValueError("one to three exact spatial source bindings required")
        for b in self.sources:
            if (b.camera_id not in {"kitchen", "family-room", "living-room"}
                    or not _integer(b.source_id) or not _integer(b.source_epoch)
                    or not all(_text(v) for v in (b.calibration_revision, b.transform_id, b.clock_domain))
                    or b.clock_basis not in {"ntp", "synchronized-media", "source-media"}):
                raise ValueError("invalid spatial source binding")
        p = self.appearance_profile
        if (type(p) is not AppearanceProfile or not _integer(p.dimensions, 1)
                or not all(_text(v) for v in (p.model_id, p.layer_id, p.fingerprint))):
            raise ValueError("exact selected appearance profile required")
        if (type(self.overlap_pairs) is not tuple or len(self.overlap_pairs) > 1
                or any(type(pair) is not tuple or len(pair) != 2
                       or set(pair) != {"kitchen", "family-room"}
                       or not set(pair) <= {b.camera_id for b in self.sources} for pair in self.overlap_pairs)):
            raise ValueError("only explicit Kitchen/Family topology allowed")
        if (not _integer(self.max_entities, 1) or self.max_entities > 64
                or not _integer(self.max_history, 1) or self.max_history > 2 * self.max_entities
                or not all(_number(v) for v in (self.max_age_s, self.max_distance_m,
                                               self.min_similarity, self.max_return_grace_s))
                or not 0 < self.max_age_s <= .35 or not 0 < self.max_distance_m <= 1.25
                or not .60 <= self.min_similarity <= 1 or not 0 <= self.max_return_grace_s <= .35):
            raise ValueError("invalid or weakened spatial limits")


@dataclass(frozen=True, slots=True)
class SpatialClockInterval:
    binding: SourceBinding
    acquisition_min_s: float
    acquisition_max_s: float
    evaluation_min_s: float
    evaluation_max_s: float


@dataclass(frozen=True, slots=True)
class SpatialClockContext:
    intervals: tuple[SpatialClockInterval, ...]
    comparability_receipt_id: str | None = None
    common_domain: str | None = None
    # Explicit source-clock -> common-clock offset uncertainty and validity.
    offsets: tuple[tuple[int, float, float], ...] = ()
    valid_ranges: tuple[tuple[int, float, float], ...] = ()
    qualification_scope: str | None = None


@dataclass(frozen=True, slots=True)
class ObservationStamp:
    member: MemberKey
    frame_id: int
    observation_id: str
    acquired_at_s: float
    clock_domain: str
    media_time_s: float | None
    clock_basis: str
    lifecycle_started_frame: int


@dataclass(frozen=True, slots=True)
class HouseholdDescriptor:
    """Owner-supplied household result, passed through without promotion/aliasing."""

    state: str = "unknown"
    identity_key: str | None = None
    stable_id: int | None = None
    visitor_generation: int | None = None
    constraint: str = "eligible"


@dataclass(frozen=True, slots=True)
class SourceObservation:
    stamp: ObservationStamp
    household: HouseholdDescriptor = HouseholdDescriptor()


@dataclass(frozen=True, slots=True)
class AcquisitionTime:
    at_s: float
    clock_domain: str
    clock_basis: str


@dataclass(frozen=True, slots=True)
class MeasurementEvidence:
    stamp: ObservationStamp
    owner: MemberKey
    evidence_id: str
    target_frame: str
    world_revision: str
    calibration_revision: str
    transform_id: str
    x_m: float
    y_m: float
    origin: str = "measured"
    units: str = "m"
    independence_id: str = ""


@dataclass(frozen=True, slots=True)
class AppearanceEvidence:
    stamp: ObservationStamp
    evidence_id: str
    model: str
    acquired_at_s: float
    clock_domain: str
    extraction_id: str
    profile: AppearanceProfile


@dataclass(frozen=True, slots=True)
class SpatialLifecycleReceipt:
    receipt_id: str
    stamp: ObservationStamp
    native_state: str


@dataclass(frozen=True, slots=True)
class SpatialLifecycleUpdate:
    """Current native-owner classification, NOT a new measurement observation."""
    receipt_id: str
    last_observation: ObservationStamp
    at_frame_id: int
    at_acquired_at_s: float
    clock_domain: str
    at_media_time_s: float
    classification: str = "retained_unmeasured"


@dataclass(frozen=True, slots=True)
class SpatialPersonCandidate:
    left: ObservationStamp
    right: ObservationStamp
    left_appearance: AppearanceEvidence
    right_appearance: AppearanceEvidence
    similarity: float
    left_copy_id: str
    right_copy_id: str
    separation_m: float | None = None
    geometry_receipt_id: str | None = None
    geometry_frame: str | None = None
    geometry_revision: str | None = None
    calibration_revisions: tuple[str, ...] = ()
    transform_ids: tuple[str, ...] = ()
    independence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpatialCurrentPersonProof:
    proof_id: str
    cohort: tuple[ObservationStamp, ...]
    candidates: tuple[SpatialPersonCandidate, ...]
    selected_pairs: tuple[tuple[MemberKey, MemberKey], ...]
    evidence_scope: str


@dataclass(frozen=True, slots=True)
class CurrentVisualProof:
    proof_id: str
    cohort: tuple[ObservationStamp, ...]
    candidates: tuple[SpatialPersonCandidate, ...]
    selected_pairs: tuple[tuple[MemberKey, MemberKey], ...]


@dataclass(frozen=True, slots=True)
class CurrentVisualIngress:
    observation: ObservationStamp
    ingress_monotonic_s: float
    appearance_observation: ObservationStamp | None = None
    appearance_ingress_monotonic_s: float | None = None


@dataclass(frozen=True, slots=True)
class CurrentVisualDeliveryContext:
    actor_id: str
    sequence: int
    evaluation_monotonic_s: float
    ingresses: tuple[CurrentVisualIngress, ...]


@dataclass(frozen=True, slots=True)
class ProjectiveCameraBinding:
    source: SourceBinding
    prior_id: str
    raster_id: str
    image_size: tuple[int, int]
    target_frame: str
    world_revision: str
    measurement_calibration_sha256: str
    own_projection: tuple[tuple[float, ...], ...]
    peer_projection: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ProjectiveContinuityPolicy:
    cameras: tuple[ProjectiveCameraBinding, ...]

    def __post_init__(self) -> None:
        if (type(self.cameras) is not tuple or len(self.cameras) != 2
                or any(type(c) is not ProjectiveCameraBinding or type(c.source) is not SourceBinding for c in self.cameras)
                or {c.source.camera_id for c in self.cameras} != {"kitchen", "family-room"}
                or len({c.source.source_id for c in self.cameras}) != 2
                or len({c.prior_id for c in self.cameras if type(c.prior_id) is str}) != 1):
            raise ValueError("two exact same-prior Kitchen/Family projective bindings required")
        for c in self.cameras:
            if (not all(_text(v) for v in (c.prior_id, c.raster_id, c.target_frame, c.world_revision, c.measurement_calibration_sha256))
                    or type(c.image_size) is not tuple or len(c.image_size) != 2
                    or not all(_integer(v, 1) for v in c.image_size)
                    or any(type(m) is not tuple or len(m) != 3 or any(type(r) is not tuple or len(r) != 4
                        or not all(_number(v) for v in r) for r in m) for m in (c.own_projection, c.peer_projection))):
                raise ValueError("finite immutable 3x4 original-raster projection matrices required")


@dataclass(frozen=True, slots=True)
class ObservedImageRegion:
    anatomy: str
    anchor_uv: tuple[float, float]
    capsule: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class ImageRegionOriginal:
    stamp: ObservationStamp
    ingress_monotonic_s: float
    body_evidence_id: str
    raster_id: str
    prior_id: str
    image_size: tuple[int, int]
    bbox_xywh: tuple[float, float, float, float]
    patches: tuple[ObservedImageRegion, ...]
    semantic_joints: tuple[tuple[int, float, float, float], ...] = ()
    head_joints: tuple[tuple[int, float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ImageRegionNativeBox:
    """CURRENT native extent for negative NEW-birth ambiguity, never anatomy."""
    stamp: ObservationStamp
    ingress_monotonic_s: float
    raster_id: str
    prior_id: str
    image_size: tuple[int, int]
    bbox_xywh: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ImageRegionOwnerAvailability:
    """Original queued owner/decoded-progress scalars; custody, never a vote."""
    observation: ObservationStamp
    graph_id: str
    source_frame_id: int
    source_media_time_s: float
    source_observed_at_us: int
    source_ingress_monotonic_s: float
    owner_grace_ns: int
    retained_members: tuple[MemberKey, ...]
    active_members: tuple[MemberKey, ...]
    shadow_members: tuple[MemberKey, ...]
    terminated_members: tuple[MemberKey, ...]
    availability_checked_monotonic_s: float
    last_progress_age_s: float
    stall_seconds: float
    availability_status: str
    owner_ingress_monotonic_s: float | None = None


@dataclass(frozen=True, slots=True)
class EstimatedSplitCustodyView:
    """Finite old-point attenuation only; NOT membership or measurement proof."""
    episode_id: str
    origin_key: PhysicalPersonKey
    split_revision: int
    prior_seed: tuple[ObservationStamp, ObservationStamp]
    split_stamps: tuple[ObservationStamp, ObservationStamp]
    successor_keys: tuple[PhysicalPersonKey, PhysicalPersonKey]
    policy_identity: str
    native_owner_receipts: tuple[ImageRegionOwnerAvailability, ImageRegionOwnerAvailability]
    owner_deadline_monotonic_s: float
    rejoined_key: PhysicalPersonKey | None = None
    rejoined_stamps: tuple[ObservationStamp, ...] = ()
    rejoin_revision: int | None = None
    # Exact native ownership of an OLD original after the genuine split.
    # Neither this destination nor a key change grants a current measurement.
    native_destinations: tuple[tuple[ObservationStamp, PhysicalPersonKey], ...] = ()


@dataclass(frozen=True, slots=True)
class ImageRegionAppearance:
    """Selected ORIGINAL physical appearance; never retagged or household ID."""
    appearance: AppearanceEvidence
    ingress_monotonic_s: float
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EstimatedImageRegionProof:
    proof_id: str
    cohort: tuple[ObservationStamp, ...]
    originals: tuple[ImageRegionOriginal, ...]
    owner_availability: tuple[ImageRegionOwnerAvailability, ...] = ()
    appearances: tuple[ImageRegionAppearance, ...] = ()
    native_boxes: tuple[ImageRegionNativeBox, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectiveMeasurementBinding:
    evidence_id: str
    target_frame: str
    world_revision: str
    calibration_revision: str
    transform_id: str
    measured_stage_id: str
    origin: str = "measured"
    units: str = "m"


@dataclass(frozen=True, slots=True)
class ProjectiveMotionSample:
    stamp: ObservationStamp
    ingress_monotonic_s: float
    body_evidence_id: str
    raster_id: str
    image_size: tuple[int, int]
    bbox_xywh: tuple[float, float, float, float]
    original_point: tuple[float, float, float]
    measurement: ProjectiveMeasurementBinding
    observed_anchor_uv: tuple[float, float]
    pixel_uncertainty_px: float | None
    # An observed capsule enclosure is geometric image support, NOT Gaussian
    # localization uncertainty. Legacy floor samples retain their actual sigma.
    support_basis: str = "observed_ankle_floor"
    observed_support_radius_px: float | None = None


@dataclass(frozen=True, slots=True)
class ProjectiveMotionPair:
    left: tuple[ProjectiveMotionSample, ...]
    right: tuple[ProjectiveMotionSample, ...]


@dataclass(frozen=True, slots=True)
class ProjectiveContinuityProof:
    visual: CurrentVisualProof
    pairs: tuple[ProjectiveMotionPair, ...]


@dataclass(frozen=True, slots=True)
class ProjectiveContinuityInspection:
    """Numeric diagnostics only: no association, frontier or grant capability."""
    error: str | None
    pairs: tuple[tuple[MemberKey, MemberKey, bool], ...] = ()


@dataclass(frozen=True, slots=True)
class StaticAnatomyQualification:
    qualification_id: str
    projective_policy_sha256: str
    artifact_sha256: str
    scope: str = "static_pair_estimated_anatomical_support_only"
    depth_convention: str = "camera_z_m"


@dataclass(frozen=True, slots=True)
class AnatomicalConsensusPolicy:
    projective_policy: ProjectiveContinuityPolicy
    qualification: StaticAnatomyQualification | None = None

    def __post_init__(self) -> None:
        if type(self.projective_policy) is not ProjectiveContinuityPolicy:
            raise ValueError("exact projective source policy required")


@dataclass(frozen=True, slots=True)
class AnatomicalDepthSupport:
    original: ProjectiveMotionSample
    capsules: tuple[tuple[float, ...], ...]
    registered_depth_interval_m: tuple[float, float] | None = None
    depth_convention: str | None = None


@dataclass(frozen=True, slots=True)
class AnatomicalPairSupport:
    left: ObservationStamp
    right: ObservationStamp
    left_originals: tuple[AnatomicalDepthSupport, ...]
    right_originals: tuple[AnatomicalDepthSupport, ...]


@dataclass(frozen=True, slots=True)
class AnatomicalConsensusProof:
    visual: CurrentVisualProof
    pairs: tuple[AnatomicalPairSupport, ...]


def appearance_extraction_identity(stamp: ObservationStamp, profile: AppearanceProfile) -> str:
    """Canonical origin identity; does not authenticate an untrusted extractor.

    The producer supplies its true original stamp/profile. Reusing this identity
    with retagged metadata fails even after historical detail has compacted.
    """
    return "appearance:" + sha256(repr((stamp, profile)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LinkProof:
    proof_id: str
    left: MeasurementEvidence
    right: MeasurementEvidence
    left_appearance: AppearanceEvidence
    right_appearance: AppearanceEvidence
    similarity: float


@dataclass(frozen=True, slots=True)
class NativeTransition:
    event_id: str
    kind: str  # adoption or correction; numeric ID changes alone are not events
    previous: ObservationStamp
    current: ObservationStamp
    peer: ObservationStamp | None
    occurred_at_s: float
    clock_domain: str


@dataclass(frozen=True, slots=True)
class LifecycleRetention:
    receipt_id: str
    last_observation: ObservationStamp
    owner: MemberKey
    lifecycle: str
    grace_s: float


@dataclass(frozen=True, slots=True)
class MemberRemoval:
    receipt_id: str
    last_observation: ObservationStamp
    kind: str  # absence or termination
    frame_id: int
    acquired_at_s: float
    clock_domain: str
    media_time_s: float | None
    retention: LifecycleRetention | None = None


@dataclass(frozen=True, slots=True)
class SourceAbsence:
    """An exact empty/reset source receipt, never last-seen presence."""

    receipt_id: str
    camera_id: str
    source_id: int
    source_epoch: int
    frame_id: int
    acquired_at_s: float
    clock_domain: str


@dataclass(frozen=True, slots=True)
class LinkReceipt:
    person_key: PhysicalPersonKey
    members: tuple[ObservationStamp, ObservationStamp]
    proof_id: str
    event_id: str | None
    admitted_at: AcquisitionTime
    appearances: tuple[AppearanceEvidence, AppearanceEvidence]
    expires_at_s: float


@dataclass(frozen=True, slots=True)
class ContinuityGap:
    person_key: PhysicalPersonKey
    members: tuple[MemberKey, ...]
    first_missing_cohort: AcquisitionTime
    deadline_s: float
    previous_receipt: LinkReceipt


@dataclass(frozen=True, slots=True)
class MemberView:
    observation: ObservationStamp
    household: HouseholdDescriptor


@dataclass(frozen=True, slots=True)
class PersonView:
    person_key: PhysicalPersonKey
    members: tuple[MemberView, ...]
    state: str  # local, accepted, or unresolved (must not grant dual contribution)
    link: LinkReceipt | None
    continuity: ContinuityGap | None = None


@dataclass(frozen=True, slots=True)
class AssociationResult:
    disposition: str  # observed, accepted, removed, reset, rejected, unresolved, capacity
    reason: str
    person_key: PhysicalPersonKey | None = None
    members: tuple[MemberView, ...] = ()
    link: LinkReceipt | None = None
    retired_keys: tuple[PhysicalPersonKey, ...] = ()
    consumed_tokens: tuple[str, ...] = ()
    corrected_members: tuple[MemberKey, ...] = ()


@dataclass(frozen=True, slots=True)
class LedgerStats:
    active_persons: int
    active_members: int
    retained_members: int
    closed_members: int
    receipts: int
    retired_keys: int
    blocked_sources: int
    lifecycle_frontiers: int
    compacted_receipts: int
    compacted_closed: int
    compacted_retired: int
    retired_serial_frontier: int
    allocated_serial: int
    appearance_origins: int
    continuity_gaps: int
    previous_shared: int


@dataclass(frozen=True, slots=True)
class _FloorAdvanceNote:
    cause: str
    previous_original_stamp: ObservationStamp | None
    current_original_stamp: ObservationStamp
    previous_ingress_monotonic_s: float | None
    current_ingress_monotonic_s: float | None
    ingress_gap_s: float | None
    native_event_frame: int | None = None


@dataclass(frozen=True, slots=True)
class _SplitBirthPending:
    origin: PhysicalPersonKey
    seeds: tuple[ObservationStamp, ...]
    native_floors: tuple[int, ...]
    policy: str
    expires_monotonic_s: float


@dataclass(frozen=True, slots=True)
class _ProvisionalExtraBirth:
    original: ObservationStamp
    body_evidence_id: str
    raster_id: str
    image_size: tuple[int, int]
    anchors: tuple[ObservationStamp, ...]
    policy: str
    basis: str = 'current_torso'
    box_anchors: tuple[ObservationStamp, ...] = ()


@dataclass(frozen=True, slots=True)
class _Member:
    observation: SourceObservation
    person_key: PhysicalPersonKey
    appearance_after_frame: int
    spatial_lifecycle: SpatialLifecycleReceipt | None = None
    spatial_join_required: bool = False
    association_basis: str = "source_local"
    visual_confirmed_frame: int = -1
    anatomical_originals: tuple[AnatomicalDepthSupport, ...] = ()
    anatomical_confirmed: ObservationStamp | None = None
    anatomical_after_frame: int = -1
    image_region_original: ImageRegionOriginal | None = None
    image_region_policy: str | None = None
    image_region_front_confirmed: bool = False
    image_region_native_seed: tuple[ObservationStamp, ...] = ()
    image_region_pair_hypothesis: bool = False
    image_region_shadow_membership: bool = False
    image_region_vote_blocked_stamp: ObservationStamp | None = None
    image_region_vote_pending: bool = False
    image_region_participant: bool = True
    image_region_owner_availability: ImageRegionOwnerAvailability | None = None
    image_region_dormant_custody: bool = False
    image_region_conflict_custody: bool = False
    image_region_semantic: bool = False
    image_region_birth_quarantined: bool = False
    image_region_birth_established: bool = False
    image_region_birth_seed: ObservationStamp | None = None
    image_region_birth_peers: tuple[ObservationStamp, ...] = ()
    image_region_provisional_extra: _ProvisionalExtraBirth | None = None
    image_region_fragment_return: ObservationStamp | None = None
    image_region_fragment_provisional: _ProvisionalExtraBirth | None = None
    image_region_split_birth: _SplitBirthPending | None = None
    image_region_rejected_birth_peer: ObservationStamp | None = None
    image_region_rejected_birth_native_floor: int = -1
    image_region_rejected_birth_origin: PhysicalPersonKey | None = None
    image_region_split_custody: EstimatedSplitCustodyView | None = None
    image_region_split_custody_native_floor: int = -1
    split_custody_refusal: tuple | None = field(default=None,compare=False,repr=False)
    spatial_lifecycle_update: SpatialLifecycleUpdate | None = None
    # Scalar continuity barriers, not retained measurements or another ledger.
    projective_native_floor: int = -1
    projective_delivery_floor: int = -1
    floor_advance_note: _FloorAdvanceNote | None = field(default=None,compare=False,repr=False)


@dataclass(frozen=True, slots=True)
class _Closed:
    last_observation: ObservationStamp
    person_key: PhysicalPersonKey
    household: HouseholdDescriptor
    removal_frame: int
    kind: str
    expires_media_s: float | None


@dataclass(frozen=True, slots=True)
class _Receipt:
    fingerprint: str
    result: AssociationResult
    sources: tuple[int, ...]
    cohorts: tuple[tuple[int, int, int], ...]


class PersonAssociationLedger:
    """Single-writer run-owned ledger; proof histories are explicitly bounded.

    observe() accepts source observations, not shared contributions. link() is the
    only common-metric dual-camera admission. Explicit spatial policy adds
    join_spatial() for qualified current membership, not a World measurement.
    transition() atomically replaces/detaches an exact
    current member. withdraw()/clear_source() remove presence before returning.
    Calls return typed decisions for malformed/stale/capacity inputs. Invalid
    construction/configuration raises ValueError before any ledger exists.
    """

    def __init__(self, run_id: str, policy: AssociationPolicy | SpatialAssociationPolicy) -> None:
        if not _text(run_id) or type(policy) not in {AssociationPolicy, SpatialAssociationPolicy}:
            raise ValueError("valid run ID and explicit AssociationPolicy required")
        self.run_id = run_id
        self.policy = policy
        self._bindings = {b.source_id: b for b in policy.sources}
        self._members: dict[MemberKey, _Member] = {}
        self._persons: dict[PhysicalPersonKey, set[MemberKey]] = {}
        self._links: dict[PhysicalPersonKey, LinkReceipt] = {}
        self._closed: dict[MemberKey, _Closed] = {}
        self._receipts: dict[str, _Receipt] = {}
        self._retired: dict[PhysicalPersonKey, tuple[int, ...]] = {}
        self._cursors: dict[int, tuple[int, float, float | None]] = {}
        self._blocked: set[int] = set()
        self._serial = 0
        self._authority_revision = 0
        self._lifecycle_frontiers: dict[int, int] = {}
        self._retired_serial_frontier = 0
        self._compacted_receipts = 0
        self._compacted_closed = 0
        self._compacted_retired = 0
        self._observation_refusal_json: str | None = None  # ONE failure-only scalar snapshot.
        self._appearance_origins: dict[MemberKey, AppearanceEvidence] = {}
        self._gaps: dict[PhysicalPersonKey, ContinuityGap] = {}
        self._last_shared: dict[PhysicalPersonKey, LinkReceipt] = {}
        self._visual_actor: str | None = None
        self._visual_sequence = -1
        self._visual_evaluation = -1.0
        self._visual_ingresses: dict[MemberKey, CurrentVisualIngress] = {}
        self._visual_last: tuple[str, int, AssociationResult] | None = None
        # One immutable endpoint fingerprint + original support-age scalars/member.
        self._projective_frontiers: dict[MemberKey, tuple] = {}
        self._image_decision_records: list[dict] = []
        self._image_decision_last_completed: dict | None = None
        self._image_decision_evicted = 0
        self._image_decision_capture_errors = 0
        self._image_decision_pending: dict | None = None
        self._image_decision_pairs: dict = {}
        # At most the selected eight COMPLETED immutable observer records.
        # Cached JSON is private scalar data, never a capability or live view.
        self._image_decision_json: tuple[tuple[dict, str], ...] = ()
        self._image_graph_id: str | None = None
        self._image_stopped_sources: set[int] = set()
        self._image_run_closed = False
        self._image_support_targets = ()  # ONE task-target capture; never authority/history.
        self._image_support_active = None

    def arm_image_region_support_diagnostic(self, stamps=()):
        """Arm existing target output for the NEXT actual membership checks.

        No classifier is invoked here. The caller owns the scalar sink; the
        single pending handle is cleared before the next bridge/lifecycle step.
        """
        self._image_support_targets = ()
        sink = {'limit':8,'byte_limit':8192,'records':[],'omitted_records':0,'errors':0}
        if type(stamps) is not tuple or len(stamps)>64:
            sink['errors']=1
            return sink
        targets=[]
        for stamp in stamps:
            if type(stamp) is not ObservationStamp:
                sink['omitted_records']+=1;continue
            m=self._members.get(stamp.member)
            if m is None or m.observation.stamp!=stamp or len(self._persons[m.person_key])!=2:continue
            if len(targets)>=8:
                sink['omitted_records']+=1;continue
            row={'query':[stamp.member.source_id,stamp.member.raw_id,stamp.member.lifecycle_generation,
                stamp.frame_id,stamp.lifecycle_started_frame], 'stage':'unobserved_stage'}
            sink['records'].append(row);targets.append((stamp,row,sink,self._authority_revision))
        self._image_support_targets=tuple(targets)
        return sink

    def _image_support_note(self, cause, blocker=None):
        try:
            row=self._image_support_active
            if row is not None:
                row['cause']=cause
                row['blocking_source']=blocker.source_id if blocker else None
        except Exception:pass

    def _image_support_members(self, stamp, context):
        m=self._members.get(stamp.member)
        if m is None:return []
        rows=[]
        for key in sorted(self._persons[m.person_key]):
            v=self._members[key];s=v.observation.stamp;o=v.image_region_original;c=v.image_region_owner_availability
            interval=next((i for i in context.intervals if i.binding.source_id==key.source_id),None)
            evaluation=interval.evaluation_max_s if interval else None
            def age(a,b):return a-b if _number(a) and _number(b) and _number(a-b) else None
            rows.append({'stamp':[key.source_id,key.raw_id,key.lifecycle_generation,s.frame_id,s.lifecycle_started_frame],
                'original_frame':o.stamp.frame_id if o else None,'original_is_current':bool(o and o.stamp==s),
                'basis':v.association_basis,'native':v.spatial_lifecycle.native_state if v.spatial_lifecycle else None,
                'participant':v.image_region_participant,'rejoin':v.spatial_join_required,
                'front':v.image_region_front_confirmed,'pair_only':v.image_region_pair_hypothesis,
                'shadow':v.image_region_shadow_membership,'dormant':v.image_region_dormant_custody,
                'vote_blocked':v.image_region_vote_blocked_stamp is not None,'vote_pending':v.image_region_vote_pending,
                'policy_matches':v.image_region_policy==m.image_region_policy,
                'floors':[v.appearance_after_frame,v.projective_native_floor,v.projective_delivery_floor],
                'seed':[[q.member.source_id,q.member.raw_id,q.member.lifecycle_generation,q.frame_id,q.lifecycle_started_frame]
                    for q in v.image_region_native_seed if q.member in self._persons[m.person_key]][:2],
                'seed_total':len(v.image_region_native_seed),'current_age_s':age(evaluation,s.acquired_at_s),
                'original_age_s':age(evaluation,o.stamp.acquired_at_s) if o else None,
                'original_wall_age_s':age(self._visual_evaluation,o.ingress_monotonic_s) if o else None,
                'caps':None if c is None else {'frame':c.source_frame_id,'status':c.availability_status,
                    'stamp_matches':c.observation==s,'checked_mono_s':c.availability_checked_monotonic_s,
                    'retained':key in c.retained_members,'active':key in c.active_members,
                    'shadow':key in c.shadow_members,'terminated':key in c.terminated_members,
                    'remaining_s':age(c.availability_checked_monotonic_s-c.last_progress_age_s+c.stall_seconds,self._visual_evaluation)}})
        return rows

    def image_region_decision_diagnostics(self) -> dict:
        """Detached scalar diagnostics only; no decision, lease or authority."""
        return self.image_region_decision_snapshot().materialize()

    def image_region_decision_snapshot(self) -> ImageRegionDecisionSnapshot:
        """Freeze EXACT completed selection without decoding or owner references."""
        latest=self._image_decision_last_completed
        distinct=latest is not None and not any(r is latest for r in self._image_decision_records)
        history=self._image_decision_records[-7 if distinct else -8:]
        selected=history+([latest] if distinct else [])
        cached={id(record):(record,encoded) for record,encoded in self._image_decision_json}
        snapshots=[]
        for record in selected:
            prior=cached.get(id(record))
            encoded=prior[1] if prior is not None and prior[0] is record else self._image_diag_json(record)
            snapshots.append((record,encoded))
        envelope={"contract":"noesis.identity.image_region_decisions.v1","limit":8,
            "evicted":self._image_decision_evicted,"capture_errors":self._image_decision_capture_errors,
            "last_evaluated_sequence":latest['sequence'] if latest is not None else None,
            "transition_history_omitted":len(self._image_decision_records)-len(history)}
        encoded=json.dumps(envelope,allow_nan=False,separators=(',',':'))
        self._image_decision_json=tuple(snapshots)
        kinds=tuple(('latest_evaluation' if distinct else 'transition_and_latest_evaluation')
                    if original is latest else 'transition' for original in selected)
        return ImageRegionDecisionSnapshot._freeze(encoded,tuple(s for _,s in snapshots),kinds,
                                                    _IMAGE_DECISION_SNAPSHOT_SEAL)

    @staticmethod
    def _image_diag_json(record):
        """Finite JSON-only completed record; malformed observer data may fail.

        Existing producer bounds are 64 members/128 pairs. These defensive
        serialization bounds exceed that schema, not any authority capacity.
        """
        pending=[(record,0)];nodes=0;string_bytes=0
        while pending:
            value,depth=pending.pop();nodes+=1
            if nodes>65536 or depth>16:raise ValueError('image_region_diagnostic_json_bound')
            kind=type(value)
            if kind is dict:
                if nodes+len(pending)+len(value)>65536:raise ValueError('image_region_diagnostic_json_bound')
                if any(type(k) is not str for k in value):raise ValueError('image_region_diagnostic_json_type')
                for key in value:
                    if len(key)>1048576:raise ValueError('image_region_diagnostic_json_bound')
                    string_bytes+=len(key.encode())
                pending.extend((v,depth+1) for v in value.values())
            elif kind is list:
                if nodes+len(pending)+len(value)>65536:raise ValueError('image_region_diagnostic_json_bound')
                pending.extend((v,depth+1) for v in value)
            elif kind is str:
                if len(value)>1048576:raise ValueError('image_region_diagnostic_json_bound')
                string_bytes+=len(value.encode())
            elif kind is float:
                if not isfinite(value):raise ValueError('image_region_diagnostic_json_nonfinite')
            elif kind not in (str,int,bool,type(None)):raise ValueError('image_region_diagnostic_json_type')
            if string_bytes>1048576:raise ValueError('image_region_diagnostic_json_bound')
        encoded=json.dumps(record,allow_nan=False,separators=(',',':'))
        if len(encoded.encode())>1048576:raise ValueError('image_region_diagnostic_json_bound')
        return encoded

    def _image_diag_check(self, member, mode, reason, blocker=None):
        active=self._image_support_active
        if active is not None:
            try:
                checks=active.setdefault('checks',{})
                key=str(member.observation.stamp.member.source_id)+':'+mode
                if key in checks or len(checks)<6:
                    checks[key]={'failure':reason,'blocking_source':blocker.source_id if blocker else None}
            except Exception:pass
        if self._image_decision_pending is None:return
        row=next((r for r in self._image_decision_pending['members'] if r['member']==asdict(member.observation.stamp.member)),None)
        if row is not None:
            value={'failure':reason,'blocking_source':blocker.source_id if blocker else None}
            if mode=='custody':row['custody_check']=value
            else:row.setdefault('live_native_checks',{})[mode]=value

    @staticmethod
    def _image_floor_note(cause,previous,current,oldmono=None,newmono=None,event=None):
        oldmono=oldmono if _number(oldmono) else None;newmono=newmono if _number(newmono) else None
        gap=newmono-oldmono if oldmono is not None and newmono is not None else None
        return _FloorAdvanceNote(cause,previous,current,oldmono,newmono,gap if _number(gap) else None,event)

    def _image_diag_components(self):
        return [{"physical_serial":p.serial,"members":[asdict(k) for k in sorted(keys)]}
                for p,keys in sorted(self._persons.items())]

    def _image_diag_input_original(self, original, current):
        """Bounded finite CURRENT input copy, not an admission or old fallback."""
        if (type(original) is not ImageRegionOriginal or original.stamp!=current
                or self._stamp_error(original.stamp)
                or not _number(original.ingress_monotonic_s) or original.ingress_monotonic_s<0
                or not all(_text(v) for v in (original.body_evidence_id,original.prior_id,original.raster_id))
                or type(original.image_size) is not tuple or len(original.image_size)!=2
                or not all(_integer(v,1) for v in original.image_size)
                or type(original.bbox_xywh) is not tuple or len(original.bbox_xywh)!=4
                or not all(_number(v) for v in original.bbox_xywh)
                or type(original.patches) is not tuple or len(original.patches)>2): return None
        patches=[]
        for patch in original.patches:
            if (type(patch) is not ObservedImageRegion or not _text(patch.anatomy)
                    or type(patch.anchor_uv) is not tuple or len(patch.anchor_uv)!=2
                    or type(patch.capsule) is not tuple or len(patch.capsule)!=6
                    or not all(_number(v) for v in patch.anchor_uv+patch.capsule)): return None
            patches.append({"anatomy":patch.anatomy,"anchor_uv":list(patch.anchor_uv),"capsule":list(patch.capsule)})
        return {"stamp":asdict(original.stamp),"ingress_monotonic_s":original.ingress_monotonic_s,
            "body_evidence_id":original.body_evidence_id,"prior_id":original.prior_id,"raster_id":original.raster_id,
            "image_size":list(original.image_size),"bbox_xywh":list(original.bbox_xywh),"patches":patches}

    def _image_diag_begin(self, proof, context, delivery):
        self._image_decision_pending=None
        self._image_decision_pairs={}
        if (type(proof) is not EstimatedImageRegionProof or type(context) is not SpatialClockContext
                or type(delivery) is not CurrentVisualDeliveryContext or len(self._members)>64
                or type(proof.cohort) is not tuple or len(proof.cohort)>128
                or type(proof.originals) is not tuple or len(proof.originals)>64
                or type(context.intervals) is not tuple or len(context.intervals)>3
                or type(delivery.ingresses) is not tuple or len(delivery.ingresses)>128): return
        def finite(v): return v if _number(v) else None
        def age(a,b): return finite(a-b) if _number(a) and _number(b) else None
        intervals={i.binding.source_id:i for i in context.intervals if type(i) is SpatialClockInterval and type(i.binding) is SourceBinding}
        ingress={r.observation.member:r for r in delivery.ingresses if type(r) is CurrentVisualIngress and not self._stamp_error(r.observation)}
        input_members={o.stamp.member for o in proof.originals if type(o) is ImageRegionOriginal and not self._stamp_error(o.stamp)}
        members=[]
        for key,m in sorted(self._members.items()):
            stamp=m.observation.stamp; o=m.image_region_original; row=ingress.get(key)
            interval=intervals.get(key.source_id)
            evaluation=interval.evaluation_max_s if interval else None
            mono=row.ingress_monotonic_s if row else None
            confirmed=o.stamp if o else None
            members.append({"member":asdict(key),"current_stamp":asdict(stamp),
                "confirmed_stamp":asdict(confirmed) if confirmed else None,
                "current_ingress_monotonic_s":finite(mono),
                "confirmed_ingress_monotonic_s":finite(o.ingress_monotonic_s) if o else None,
                "source_evaluation_s":finite(evaluation),"current_source_age_s":age(evaluation,stamp.acquired_at_s),
                "confirmed_source_age_s":age(evaluation,confirmed.acquired_at_s) if confirmed else None,
                "current_wall_age_s":age(delivery.evaluation_monotonic_s,mono),
                "confirmed_wall_age_s":age(delivery.evaluation_monotonic_s,o.ingress_monotonic_s) if o else None,
                "floors":{"appearance":m.appearance_after_frame,"native":m.projective_native_floor,"delivery":m.projective_delivery_floor},
                "floors_before":{"appearance":m.appearance_after_frame,"native":m.projective_native_floor,"delivery":m.projective_delivery_floor},
                "input_status":"present_unvalidated" if key in input_members else "absent",
                "input_is_current":None if key in input_members else False,
                "input_validation_passed":None,"current_input_original":None,
                "continuity_hypothesis_before":m.image_region_pair_hypothesis,
                "continuity_hypothesis_after":None,
                "vote_block_before":asdict(m.image_region_vote_blocked_stamp) if m.image_region_vote_blocked_stamp else None,
                "vote_block_after":None,"vote_pending_before":m.image_region_vote_pending,"vote_pending_after":None,
                "effective_status":"not_evaluated","valid":None})
            if len(self._persons[m.person_key])==2:
                members[-1].update(floor_advance_note=asdict(m.floor_advance_note) if m.floor_advance_note else None,
                    paired_native_seed=[asdict(s) for s in m.image_region_native_seed
                        if s.member in self._persons[m.person_key]][:2],live_native_checks={},custody_check={'failure':'not_evaluated'},
                    lineage={'before':{'participant':m.image_region_participant,'rejoin_required':m.spatial_join_required,
                        'shadow_membership':m.image_region_shadow_membership,'front_confirmed':m.image_region_front_confirmed,
                        'native_state':m.spatial_lifecycle.native_state if m.spatial_lifecycle else None}})
        self._image_decision_pending={"run_id":self.run_id,"proof_id":proof.proof_id if _text(proof.proof_id) else None,
            "actor_id":delivery.actor_id if _text(delivery.actor_id) else None,"sequence":delivery.sequence if _integer(delivery.sequence) else None,
            "evaluation_monotonic_s":finite(delivery.evaluation_monotonic_s),
            "authority_revision_before":self._authority_revision,"authority_revision_after":None,
            "pre_components":self._image_diag_components(),"post_components":[],"members":members,"pairs":[],"outcome":None}

    def _image_diag_finish(self, result):
        record=self._image_decision_pending
        if record is None:return
        record["post_components"]=self._image_diag_components()
        # Native/delivery floors may advance during the actual ordered decision.
        for row in record["members"]:
            m=self._members.get(MemberKey(**row["member"]))
            if m is not None:
                row["floors"]={"appearance":m.appearance_after_frame,"native":m.projective_native_floor,"delivery":m.projective_delivery_floor}
                row["continuity_hypothesis_after"]=m.image_region_pair_hypothesis
                row["vote_block_after"]=asdict(m.image_region_vote_blocked_stamp) if m.image_region_vote_blocked_stamp else None
                row["vote_pending_after"]=m.image_region_vote_pending
                if 'paired_native_seed' in row:
                    row['floor_advance_note']=asdict(m.floor_advance_note) if m.floor_advance_note else None
                    c=m.image_region_owner_availability
                    remaining=c.availability_checked_monotonic_s-c.last_progress_age_s+c.stall_seconds-self._visual_evaluation if c else None
                    row['owner_availability']={'present':c is not None,'status':c.availability_status if c else None,
                        'source_frame':c.source_frame_id if c else None,'remaining_s':remaining if _number(remaining) else None,
                        'retained':bool(c and c.observation.member in c.retained_members),
                        'active':bool(c and c.observation.member in c.active_members),
                        'shadow':bool(c and c.observation.member in c.shadow_members),
                        'terminated':bool(c and c.observation.member in c.terminated_members)}
                    row['lineage']['after']={'participant':m.image_region_participant,'rejoin_required':m.spatial_join_required,
                        'shadow_membership':m.image_region_shadow_membership,'front_confirmed':m.image_region_front_confirmed,
                        'native_state':m.spatial_lifecycle.native_state if m.spatial_lifecycle else None}
        revoked=(result.disposition not in {"accepted","observed"} and any(len(c["members"])>1 for c in record["pre_components"]))
        hypothesis_changed=any(r["continuity_hypothesis_before"]!=r["continuity_hypothesis_after"] for r in record["members"])
        vote_changed=any(r["vote_block_before"]!=r["vote_block_after"] or r["vote_pending_before"]!=r["vote_pending_after"] for r in record["members"])
        # Completed evaluations are observable even when topology/history stays
        # unchanged. Never expose the in-progress pending record to consumers.
        record["authority_revision_after"]=self._authority_revision
        record["outcome"]={"disposition":result.disposition,"reason":result.reason,
            "corrected_members":[asdict(k) for k in result.corrected_members]}
        self._image_decision_last_completed=record
        if record["pre_components"]!=record["post_components"] or result.corrected_members or revoked or hypothesis_changed or vote_changed:
            self._image_decision_records.append(record)
            if len(self._image_decision_records)>8:
                self._image_decision_records.pop(0);self._image_decision_evicted+=1

    def _reject(self, reason: str, disposition: str = "rejected") -> AssociationResult:
        return AssociationResult(disposition, reason)

    def observation_refusal_diagnostic(self) -> dict | None:
        """Detached private last refusal, not an admission/frontier capability."""
        return json.loads(self._observation_refusal_json) if self._observation_refusal_json else None

    def _observe_lifecycle_refusal(self, reason: str, stamp: ObservationStamp) -> AssociationResult:
        # Called AT the actual rejected branch, before any observer can see a
        # later state. Never retain the observation/household or mutable graph.
        self._observation_refusal_json = None
        try:
            key = stamp.member
            current, closed = self._members.get(key), self._closed.get(key)
            binding = self._bindings.get(key.source_id)
            note = {'contract': 'noesis.identity.observation_refusal.v1', 'reason': reason,
                'stamp': asdict(stamp), 'authority_revision': self._authority_revision,
                'source_frontier': self._lifecycle_frontiers.get(key.source_id),
                'source_cursor': self._cursors.get(key.source_id),
                'source_blocked': key.source_id in self._blocked,
                'binding_epoch': binding.source_epoch if binding else None,
                'current_stamp': asdict(current.observation.stamp) if current else None,
                'current_native_state': current.spatial_lifecycle.native_state
                    if current and current.spatial_lifecycle else None,
                'closed_stamp': asdict(closed.last_observation) if closed else None,
                'closed_kind': closed.kind if closed else None,
                'closed_removal_frame': closed.removal_frame if closed else None,
                'closed_expires_media_s': closed.expires_media_s if closed else None,
                'same_raw_closed_count': sum(k.source_id == key.source_id and k.raw_id == key.raw_id
                                             for k in self._closed),
                'active_count': len(self._members), 'closed_count': len(self._closed),
                'compacted_closed_total': self._compacted_closed,
                'compacted_receipts_total': self._compacted_receipts,
                'compacted_retired_total': self._compacted_retired}
            encoded = json.dumps(note, allow_nan=False, separators=(',', ':'))
            if len(encoded.encode()) <= 8192:
                self._observation_refusal_json = encoded
        except Exception:
            pass  # Optional diagnostics must NEVER replace the real refusal.
        return self._reject(reason, 'unresolved')

    def _binding(self, key: MemberKey) -> SourceBinding | None:
        binding = self._bindings.get(key.source_id)
        if (binding is not None and key.run_id == self.run_id
                and key.camera_id == binding.camera_id
                and key.source_epoch == binding.source_epoch):
            return binding
        return None

    def _stamp_error(self, stamp: object) -> str | None:
        if type(stamp) is not ObservationStamp or type(stamp.member) is not MemberKey:
            return "malformed_observation"
        key = stamp.member
        if (not _text(key.run_id) or not _text(key.camera_id)
                or not all(_integer(v) for v in (key.source_id, key.source_epoch, key.raw_id))
                or key.raw_id == 2**64 - 1
                or not _integer(key.lifecycle_generation, 1) or not _integer(stamp.frame_id)
                or not _integer(stamp.lifecycle_started_frame)
                or stamp.lifecycle_started_frame > stamp.frame_id
                or not _text(stamp.observation_id) or not _text(stamp.clock_domain)
                or not _number(stamp.acquired_at_s) or stamp.acquired_at_s < 0
                or (stamp.media_time_s is not None
                    and (not _number(stamp.media_time_s) or stamp.media_time_s < 0))):
            return "malformed_observation"
        binding = self._binding(key)
        if binding is None:
            return "source_binding"
        spatial_media = type(self.policy) is SpatialAssociationPolicy and stamp.clock_basis == "source-media"
        if (stamp.clock_domain != binding.clock_domain
                or not (_clock_basis(stamp.clock_basis) or spatial_media) or stamp.clock_basis != binding.clock_basis
                or (spatial_media and stamp.media_time_s != stamp.acquired_at_s)):
            return "clock_domain"
        return None

    def _household_error(self, descriptor: object) -> str | None:
        if type(descriptor) is not HouseholdDescriptor:
            return "malformed_household"
        if (not _text(descriptor.state) or not _text(descriptor.constraint)
                or descriptor.state not in {"unknown", "provisional", "resident", "visitor", "handoff"}
                or descriptor.constraint not in {"eligible", "unresolved", "rejected"}
                or (descriptor.identity_key is not None and not _text(descriptor.identity_key))
                or (descriptor.stable_id is not None and not _integer(descriptor.stable_id, 1))
                or (descriptor.visitor_generation is not None
                    and not _integer(descriptor.visitor_generation))):
            return "malformed_household"
        if descriptor.state in {"unknown", "provisional"} and any(
                v is not None for v in (descriptor.identity_key, descriptor.stable_id,
                                        descriptor.visitor_generation)):
            return "malformed_household"
        if descriptor.state in {"resident", "visitor", "handoff"} and descriptor.identity_key is None:
            return "malformed_household"
        if descriptor.state == "visitor" and descriptor.visitor_generation is None:
            return "malformed_household"
        if descriptor.state != "visitor" and descriptor.visitor_generation is not None:
            return "malformed_household"
        return None

    def _clock_error(self, clock: object) -> str | None:
        if (type(clock) is not AcquisitionTime or not _number(clock.at_s)
                or clock.at_s < 0 or not _text(clock.clock_domain)
                or not (_clock_basis(clock.clock_basis)
                        or type(self.policy) is SpatialAssociationPolicy and clock.clock_basis == "source-media")):
            return "clock_domain"
        return None

    def _fresh(self, stamp: ObservationStamp, at: AcquisitionTime) -> str | None:
        if stamp.clock_domain != at.clock_domain or stamp.clock_basis != at.clock_basis:
            return "clock_domain"
        if stamp.acquired_at_s > at.at_s:
            return "future_evidence"
        if at.at_s - stamp.acquired_at_s > self.policy.max_age_s:
            return "stale_evidence"
        return None

    def _order_error(self, stamp: ObservationStamp) -> str | None:
        cursor = self._cursors.get(stamp.member.source_id)
        if cursor is not None:
            frame, acquired, media = cursor
            if stamp.frame_id < frame or stamp.acquired_at_s < acquired:
                return "reordered_observation"
            if stamp.frame_id == frame and (stamp.acquired_at_s != acquired
                    or stamp.media_time_s != media):
                return "contradictory_cohort"
            if stamp.media_time_s is not None and media is not None and stamp.media_time_s < media:
                return "reordered_media_time"
        return None

    def _views(self, key: PhysicalPersonKey) -> tuple[MemberView, ...]:
        return tuple(MemberView(self._members[k].observation.stamp,
                                self._members[k].observation.household)
                     for k in sorted(self._persons.get(key, ())))

    def _result(self, disposition: str, reason: str, key: PhysicalPersonKey,
                *, retired: tuple[PhysicalPersonKey, ...] = (),
                tokens: tuple[str, ...] = ()) -> AssociationResult:
        return AssociationResult(disposition, reason, key, self._views(key),
                                 self._links.get(key), retired, tokens)

    def _owned_key_count(self) -> int:
        return len(set(self._persons) | {c.person_key for c in self._closed.values()
                                        if c.expires_media_s is not None})

    def _allocate(self) -> PhysicalPersonKey:
        self._serial += 1
        return PhysicalPersonKey(self.run_id, self._serial)

    def _remove(self, key: MemberKey) -> _Member:
        member = self._members.pop(key)
        if member.image_region_split_custody is not None:
            self._image_revoke_split_custody(member.image_region_split_custody)
        for other,m in tuple(self._members.items()):
            if m.image_region_rejected_birth_peer is not None and m.image_region_rejected_birth_peer.member==key:
                self._members[other]=replace(m,image_region_split_birth=None,image_region_rejected_birth_peer=None,
                    image_region_rejected_birth_native_floor=-1,
                    image_region_rejected_birth_origin=None)
        group = self._persons[member.person_key]
        group.remove(key)
        self._links.pop(member.person_key, None)
        self._gaps.pop(member.person_key, None)
        self._last_shared.pop(member.person_key, None)
        self._appearance_origins.pop(key, None)
        if not group:
            del self._persons[member.person_key]
        elif len(group) == 1:
            remaining = next(iter(group))
            current = self._members[remaining]
            if current.image_region_pair_hypothesis or current.image_region_shadow_membership or current.image_region_conflict_custody:
                self._members[remaining]=replace(current,association_basis="source_local",
                    image_region_conflict_custody=False,
                    image_region_pair_hypothesis=False,image_region_shadow_membership=False,
                    image_region_vote_blocked_stamp=None,image_region_vote_pending=False,
                    image_region_native_seed=(),image_region_front_confirmed=False)
            elif current.association_basis.startswith("prior_projective_continuity"):
                # Genuine peer removal leaves the SAME physical person with
                # ordinary local authority, not a demand for absent peer proof.
                # Preserve any REAL native rejoin/gap flag on the survivor.
                self._members[remaining] = replace(current, association_basis="source_local")
        self._authority_revision += 1
        return member

    def observe(self, observation: SourceObservation) -> AssociationResult:
        if type(observation) is not SourceObservation:
            return self._reject("malformed_observation")
        error = self._stamp_error(observation.stamp) or self._household_error(observation.household)
        if error:
            return self._reject(error)
        stamp = observation.stamp
        key = stamp.member
        if key.source_id in self._blocked:
            return self._reject("source_reset_required", "unresolved")
        error = self._order_error(stamp)
        if error:
            return self._reject(error)
        previous = self._members.get(key)
        if previous is not None:
            if previous.observation == observation:
                return self._result("observed", "duplicate_observation", previous.person_key)
            old_stamp = previous.observation.stamp
            if stamp.lifecycle_started_frame != old_stamp.lifecycle_started_frame:
                return self._reject("lifecycle_binding")
            if stamp.frame_id <= old_stamp.frame_id or stamp.observation_id == old_stamp.observation_id:
                return self._reject("reordered_observation")
            self.expire_retention(key.source_id, stamp.media_time_s)
            person_key = previous.person_key
            appearance_floor = previous.appearance_after_frame
            self._links.pop(person_key, None)
        else:
            # Same numeric ID in a new lifecycle is never an inherited lock.
            for existing in self._members:
                if (existing.source_id == key.source_id and existing.raw_id == key.raw_id
                        and existing != key):
                    return self._reject("lifecycle_reset_required", "unresolved")
            closed = self._closed.get(key)
            if closed is not None:
                if (closed.kind != "absence" or closed.expires_media_s is None
                        or stamp.media_time_s is None or stamp.media_time_s > closed.expires_media_s
                        or closed.last_observation.media_time_s is None
                        or stamp.media_time_s < closed.last_observation.media_time_s
                        or stamp.frame_id <= closed.removal_frame):
                    return self._observe_lifecycle_refusal("closed_lifecycle", stamp)
                person_key = closed.person_key
                appearance_floor = closed.removal_frame
                if self._persons.get(person_key):
                    # Returning source remains pending until a new proof is supplied.
                    if any(k.camera_id == key.camera_id for k in self._persons[person_key]):
                        return self._reject("same_camera_collision", "unresolved")
                del self._closed[key]
            else:
                if any(k.source_id == key.source_id and k.raw_id == key.raw_id
                       and k.lifecycle_generation >= key.lifecycle_generation for k in self._closed):
                    return self._observe_lifecycle_refusal("retired_lifecycle_generation", stamp)
                if stamp.lifecycle_started_frame <= self._lifecycle_frontiers.get(key.source_id, -1):
                    return self._observe_lifecycle_refusal("lifecycle_frontier", stamp)
                self.expire_retention(key.source_id, stamp.media_time_s)
                if self._owned_key_count() >= self.policy.max_entities:
                    return self._reject("entity_capacity", "capacity")
                person_key = self._allocate()
                appearance_floor = stamp.lifecycle_started_frame - 1
        self._members[key] = _Member(observation, person_key, appearance_floor)
        self._authority_revision += 1
        self._persons.setdefault(person_key, set()).add(key)
        self._cursors[key.source_id] = (stamp.frame_id, stamp.acquired_at_s, stamp.media_time_s)
        self.compact_history()
        if len(self._persons[person_key]) > 1:
            return self._result("unresolved", "current_cross_source_proof_required", person_key)
        return self._result("observed", "local_observation", person_key)

    def observe_spatial(self, original: SourceObservation,
                        lifecycle_receipt: SpatialLifecycleReceipt, *,
                        image_source_local_reactivation: bool = False,
                        retain_image_membership: bool = False,
                        quarantine_unresolved_birth: bool = False) -> AssociationResult:
        self._image_support_targets=()
        if type(retain_image_membership) is not bool:
            return self._reject("image_region_live_track_boolean_required","association_unresolved")
        if type(image_source_local_reactivation) is not bool:
            return self._reject("image_source_local_boolean_required","association_unresolved")
        if type(quarantine_unresolved_birth) is not bool:
            return self._reject("image_region_birth_boolean_required","association_unresolved")
        if type(self.policy) is not SpatialAssociationPolicy:
            return self._reject("spatial_policy_required")
        if (type(original) is not SourceObservation or self._stamp_error(original.stamp)
                or type(lifecycle_receipt) is not SpatialLifecycleReceipt
                or lifecycle_receipt.stamp != original.stamp or not _text(lifecycle_receipt.receipt_id)
                or lifecycle_receipt.native_state not in {"ACTIVE", "INACTIVE", "TENTATIVE", "UNKNOWN", "TERMINATED"}):
            return self._reject("spatial_lifecycle_binding")
        key = original.stamp.member
        prior = self._members.get(key)
        if prior is not None and prior.spatial_lifecycle_update is not None:
            event=prior.spatial_lifecycle_update
            if (original.stamp.frame_id<=event.at_frame_id
                    or original.stamp.acquired_at_s<event.at_acquired_at_s
                    or original.stamp.media_time_s is None
                    or original.stamp.media_time_s<event.at_media_time_s):
                return self._reject("spatial_lifecycle_reactivation_requires_new_observation","association_unresolved")
        if lifecycle_receipt.native_state == "TERMINATED":
            stamp = original.stamp
            return self.withdraw(MemberRemoval(lifecycle_receipt.receipt_id, stamp, "termination",
                                               stamp.frame_id, stamp.acquired_at_s, stamp.clock_domain,
                                               stamp.media_time_s))
        returning = key in self._closed
        result = self.observe(original)
        current = self._members.get(key)
        if result.person_key is not None and current is not None and current.observation == original:
            needs_join = returning or (prior is not None and prior.spatial_join_required)
            shadow_return=(retain_image_membership and not returning and prior is not None
                and prior.image_region_shadow_membership and prior.spatial_lifecycle_update is not None
                and lifecycle_receipt.native_state=="ACTIVE"
                and self._image_shadow_event_continuous(prior,original.stamp.acquired_at_s,original.stamp.media_time_s))
            if shadow_return: needs_join=False
            local_reactivation=(image_source_local_reactivation and not returning and prior is not None
                and prior.spatial_lifecycle_update is not None and not prior.image_region_participant
                and prior.association_basis=="source_local" and len(self._persons[current.person_key])==1
                and current.person_key==prior.person_key and prior.spatial_lifecycle is not None
                and prior.spatial_lifecycle.native_state==lifecycle_receipt.native_state=="ACTIVE")
            if local_reactivation:
                # A real source-local observation authorizes only this already
                # separate singleton. It does not restore any cross-view seed.
                needs_join=False
            continuous = (not returning and prior is not None and prior.spatial_lifecycle is not None
                and prior.spatial_lifecycle.native_state == lifecycle_receipt.native_state == "ACTIVE"
                and original.stamp.lifecycle_started_frame == prior.observation.stamp.lifecycle_started_frame
                and original.stamp.media_time_s is not None and prior.observation.stamp.media_time_s is not None
                and 0 <= original.stamp.acquired_at_s - prior.observation.stamp.acquired_at_s <= self.policy.max_age_s
                and 0 <= original.stamp.media_time_s - prior.observation.stamp.media_time_s <= self.policy.max_age_s)
            native_floor = (prior.projective_native_floor if continuous and prior.projective_native_floor >= 0
                            else original.stamp.frame_id)
            note=prior.floor_advance_note if prior else None
            if prior is None or native_floor>prior.projective_native_floor:
                before=prior.observation.stamp if prior else None
                reason=('native_initial' if prior is None else 'native_return' if returning else
                    'native_birth_change' if before.lifecycle_started_frame!=original.stamp.lifecycle_started_frame else
                    'native_state_change' if prior.spatial_lifecycle is None or prior.spatial_lifecycle.native_state!='ACTIVE' or lifecycle_receipt.native_state!='ACTIVE' else
                    'native_continuity_gap')
                previous_ingress=self._visual_ingresses.get(key)
                note=self._image_floor_note(reason,before,original.stamp,
                    previous_ingress.ingress_monotonic_s if previous_ingress else None)
            if (prior is not None and prior.association_basis.startswith(("prior_projective_continuity", "current_registered_anatomical_consensus", "estimated_current_image_regions", "estimated_image_region_continuation", "estimated_live_track_continuity"))
                    and (prior.spatial_lifecycle is None or prior.spatial_lifecycle.native_state != "ACTIVE"
                         or lifecycle_receipt.native_state != "ACTIVE")):
                for peer in self._persons[current.person_key]:
                    self._members[peer] = replace(self._members[peer], spatial_join_required=True,
                        image_region_pair_hypothesis=False,image_region_shadow_membership=False)
                needs_join = True
            established=bool(prior and prior.image_region_birth_established
                and (continuous or shadow_return or local_reactivation) and not returning
                and lifecycle_receipt.native_state=='ACTIVE' and not needs_join)
            event=prior.spatial_lifecycle_update if prior else None
            caps=prior.image_region_owner_availability if prior else None
            fragment_return=bool(quarantine_unresolved_birth and retain_image_membership and established
                and prior is not None and not prior.image_region_participant and event is not None
                and event.classification=='retained_unmeasured' and event.last_observation==prior.observation.stamp
                and caps is not None and caps.observation==prior.observation.stamp
                and caps.graph_id==self._image_graph_id and caps.availability_status=='running'
                and caps.source_frame_id==event.at_frame_id and key in caps.shadow_members
                and key in caps.active_members and key in caps.retained_members and key not in caps.terminated_members
                and self._image_shadow_event_continuous(prior,original.stamp.acquired_at_s,original.stamp.media_time_s))
            updated = replace(current, spatial_lifecycle=lifecycle_receipt, spatial_join_required=needs_join,
                              image_region_fragment_return=original.stamp if fragment_return else None,
                              image_region_fragment_provisional=prior.image_region_fragment_provisional if prior
                                  and not returning and lifecycle_receipt.native_state=='ACTIVE'
                                  and prior.observation.stamp.lifecycle_started_frame==original.stamp.lifecycle_started_frame else None,
                              image_region_provisional_extra=prior.image_region_provisional_extra if prior
                                  and not returning and prior.observation.stamp.member==original.stamp.member
                                  and prior.observation.stamp.lifecycle_started_frame==original.stamp.lifecycle_started_frame else None,
                              image_region_rejected_birth_origin=prior.image_region_rejected_birth_origin if prior
                                  and (continuous or shadow_return or local_reactivation) and not returning
                                  and lifecycle_receipt.native_state=='ACTIVE' and not needs_join else None,
                              image_region_rejected_birth_peer=prior.image_region_rejected_birth_peer if prior
                                  and (continuous or shadow_return or local_reactivation) and not returning
                                  and lifecycle_receipt.native_state=='ACTIVE' and not needs_join else None,
                              image_region_rejected_birth_native_floor=prior.image_region_rejected_birth_native_floor if prior
                                  and (continuous or shadow_return or local_reactivation) and not returning
                                  and lifecycle_receipt.native_state=='ACTIVE' and not needs_join else -1,
                              image_region_split_custody=prior.image_region_split_custody if prior
                                  and (continuous or shadow_return or local_reactivation) and not returning
                                  and lifecycle_receipt.native_state=='ACTIVE' else None,
                              image_region_split_custody_native_floor=prior.image_region_split_custody_native_floor if prior
                                  and (continuous or shadow_return or local_reactivation) and not returning
                                  and lifecycle_receipt.native_state=='ACTIVE' else -1,
                              image_region_split_birth=prior.image_region_split_birth if prior
                                  and (continuous or shadow_return or local_reactivation) and not returning
                                  and lifecycle_receipt.native_state=='ACTIVE' and not needs_join else None,
                              image_region_birth_established=established,
                              image_region_birth_quarantined=bool(quarantine_unresolved_birth and not established),
                              image_region_birth_seed=prior.image_region_birth_seed if established else None,
                              image_region_birth_peers=prior.image_region_birth_peers if established else (),
                              association_basis=prior.association_basis if prior else "source_local",
                              visual_confirmed_frame=-1 if local_reactivation else prior.visual_confirmed_frame if prior else -1,
                              anatomical_originals=prior.anatomical_originals if prior else (),
                              anatomical_confirmed=prior.anatomical_confirmed if prior else None,
                              anatomical_after_frame=max(prior.anatomical_after_frame,
                                  prior.observation.stamp.frame_id if needs_join and prior.association_basis.startswith("current_registered_anatomical_consensus") else -1) if prior else -1,
                              image_region_original=prior.image_region_original if prior else None,
                              image_region_policy=prior.image_region_policy if prior else None,
                              image_region_native_seed=() if local_reactivation else prior.image_region_native_seed if prior else (),
                              image_region_front_confirmed=False if local_reactivation else prior.image_region_front_confirmed if prior else False,
                              image_region_pair_hypothesis=False if local_reactivation else prior.image_region_pair_hypothesis if prior else False,
                              image_region_shadow_membership=bool(prior and prior.image_region_shadow_membership
                                  and lifecycle_receipt.native_state=="ACTIVE" and (continuous or shadow_return)),
                              image_region_vote_blocked_stamp=prior.image_region_vote_blocked_stamp if prior else None,
                              image_region_conflict_custody=prior.image_region_conflict_custody if prior else False,
                              image_region_semantic=prior.image_region_semantic if prior else False,
                              image_region_vote_pending=bool(prior and prior.image_region_vote_blocked_stamp is not None
                                  and (prior.image_region_vote_pending or prior.observation.stamp!=original.stamp)),
                              projective_native_floor=native_floor,
                              floor_advance_note=note,
                              projective_delivery_floor=prior.projective_delivery_floor if continuous else -1)
            if updated != current:
                self._members[key] = updated
                self._authority_revision += 1
        return result

    def image_region_active_cohort(self) -> tuple[ObservationStamp, ...]:
        """Literal participant roster; stale ACTIVE members are NOT filtered."""
        rows=tuple(sorted((m.observation.stamp for m in self._members.values()
            if m.observation.stamp.member.camera_id in {"kitchen","family-room"}
            and m.spatial_lifecycle is not None and m.spatial_lifecycle.native_state=="ACTIVE"
            and m.image_region_participant),key=lambda s:s.member))
        if len(rows)>128: raise ValueError("image_region_cohort_capacity")
        return rows

    def _image_shadow_event_continuous(self, member, source_time, media_time):
        event=member.spatial_lifecycle_update
        source=event.at_acquired_at_s if event else member.observation.stamp.acquired_at_s
        media=event.at_media_time_s if event else member.observation.stamp.media_time_s
        return (media is not None and media_time is not None
            and 0<=source_time-source<=self.policy.max_age_s
            and 0<=media_time-media<=self.policy.max_age_s)

    def update_spatial_lifecycle(self, update: SpatialLifecycleUpdate, *, retain_image_membership: bool=False) -> AssociationResult:
        """Apply a genuine CURRENT retained-minus-measured owner receipt only."""
        self._image_support_targets=()
        if type(retain_image_membership) is not bool:
            return self._reject("image_region_live_track_boolean_required","association_unresolved")
        if (type(self.policy) is not SpatialAssociationPolicy or type(update) is not SpatialLifecycleUpdate
                or not _text(update.receipt_id) or self._stamp_error(update.last_observation)
                or not _integer(update.at_frame_id) or not _number(update.at_acquired_at_s)
                or not _number(update.at_media_time_s) or update.at_media_time_s<0
                or update.classification!="retained_unmeasured"):
            return self._reject("spatial_lifecycle_update_binding","association_unresolved")
        stamp=update.last_observation; member=self._members.get(stamp.member)
        if (member is None or member.observation.stamp!=stamp
                or member.spatial_lifecycle is None or member.spatial_lifecycle.native_state!="ACTIVE"):
            return self._reject("spatial_lifecycle_update_noncurrent","association_unresolved")
        if member.spatial_lifecycle_update==update:
            return self._result("observed","retained_unmeasured",member.person_key)
        prior=member.spatial_lifecycle_update
        cursor=self._cursors.get(stamp.member.source_id)
        if (update.clock_domain!=stamp.clock_domain or stamp.media_time_s is None
                or update.at_frame_id<stamp.frame_id or update.at_acquired_at_s<stamp.acquired_at_s
                or update.at_media_time_s<stamp.media_time_s
                or prior is not None and (update.receipt_id==prior.receipt_id or update.at_frame_id<=prior.at_frame_id
                    or update.at_acquired_at_s<prior.at_acquired_at_s or update.at_media_time_s<prior.at_media_time_s)
                or cursor is not None and (update.at_frame_id<cursor[0] or update.at_acquired_at_s<cursor[1]
                    or cursor[2] is not None and update.at_media_time_s<cursor[2]
                    or update.at_frame_id==cursor[0] and (update.at_acquired_at_s!=cursor[1] or update.at_media_time_s!=cursor[2]))):
            return self._reject("spatial_lifecycle_update_reordered","association_unresolved")
        group=self._persons[member.person_key]
        keep=(retain_image_membership and len(group)==2
            and self._image_shadow_event_continuous(member,update.at_acquired_at_s,update.at_media_time_s)
            and all(self._members[k].image_region_front_confirmed and self._members[k].image_region_native_seed
                and self._members[k].spatial_lifecycle is not None
                and self._members[k].spatial_lifecycle.native_state=="ACTIVE"
                and self._members[k].association_basis in {"estimated_current_image_regions","estimated_image_region_continuation","estimated_live_track_continuity"}
                for k in group))
        if len(group)>1 and not keep and self._owned_key_count()>=self.policy.max_entities:
            return self._reject("spatial_lifecycle_split_capacity","capacity")
        corrected=()
        if keep:
            for key in group:
                self._members[key]=replace(self._members[key],image_region_shadow_membership=True,
                    association_basis="estimated_live_track_continuity")
            member=self._members[stamp.member]
        if len(group)>1 and not keep:
            # The measured continuing source keeps the old physical key. A
            # shadow is retained ownership, not current cross-camera presence.
            group.remove(stamp.member)
            detached=self._allocate();self._persons[detached]={stamp.member}
            member=replace(member,person_key=detached)
            corrected=(stamp.member,)
            for key in group:
                peer=self._members[key]
                self._members[key]=replace(peer,association_basis="source_local",
                    spatial_join_required=False,image_region_native_seed=(),image_region_front_confirmed=False,
                    image_region_pair_hypothesis=False,image_region_shadow_membership=False)
        self._members[stamp.member]=replace(member,image_region_participant=False,
            floor_advance_note=self._image_floor_note('retained_unmeasured',stamp,stamp,
                self._visual_ingresses[stamp.member].ingress_monotonic_s if stamp.member in self._visual_ingresses else None,
                event=update.at_frame_id),
            spatial_lifecycle_update=update,spatial_join_required=True,association_basis="estimated_live_track_continuity" if keep else "source_local",
            image_region_native_seed=member.image_region_native_seed if keep else (),image_region_front_confirmed=keep,
            image_region_shadow_membership=keep,
            image_region_pair_hypothesis=False,
            projective_native_floor=max(member.projective_native_floor,update.at_frame_id),
            projective_delivery_floor=max(member.projective_delivery_floor,update.at_frame_id))
        if (retain_image_membership and member.image_region_split_custody is not None
                and member.image_region_split_custody_native_floor==member.projective_native_floor
                and self._image_shadow_event_continuous(member,update.at_acquired_at_s,update.at_media_time_s)):
            current=self._members[stamp.member]
            self._members[stamp.member]=replace(current,
                image_region_split_custody_native_floor=current.projective_native_floor)
        if retain_image_membership and self._image_shadow_event_continuous(member,update.at_acquired_at_s,update.at_media_time_s):
            # Carry a NEGATIVE birth fact only across this actual ordered native
            # retention event. Never infer continuity from diagnostics or bridge
            # an earlier ordinary native gap with a later shadow receipt.
            for key,other in tuple(self._members.items()):
                rejected=other.image_region_rejected_birth_peer
                if (rejected is not None and rejected.member==stamp.member
                        and rejected.lifecycle_started_frame==stamp.lifecycle_started_frame
                        and other.image_region_rejected_birth_origin==member.person_key
                        and other.image_region_rejected_birth_native_floor==member.projective_native_floor):
                    self._members[key]=replace(other,
                        image_region_rejected_birth_native_floor=max(member.projective_native_floor,update.at_frame_id))
        self._cursors[stamp.member.source_id]=(update.at_frame_id,update.at_acquired_at_s,update.at_media_time_s)
        self._authority_revision+=1
        return AssociationResult("observed","retained_unmeasured",member.person_key,
            corrected_members=corrected)

    def spatial_clock_error(self, context: object) -> str | None:
        if (type(context) is not SpatialClockContext or type(context.intervals) is not tuple
                or not 1 <= len(context.intervals) <= 3
                or any(type(i) is not SpatialClockInterval or type(i.binding) is not SourceBinding
                       for i in context.intervals)):
            return "spatial_clock_context"
        sources = set()
        for interval in context.intervals:
            b = interval.binding
            if b.source_id in sources or self._bindings.get(b.source_id) != b:
                return "spatial_clock_binding"
            sources.add(b.source_id)
            values = (interval.acquisition_min_s, interval.acquisition_max_s,
                      interval.evaluation_min_s, interval.evaluation_max_s)
            if (not all(_number(v) and v >= 0 for v in values)
                    or not values[0] <= values[1] <= values[2] <= values[3]):
                return "spatial_clock_interval"
        return None

    def spatial_member_error(self, stamp: ObservationStamp, context: SpatialClockContext) -> str | None:
        error = self._stamp_error(stamp) or self.spatial_clock_error(context)
        if error:
            return error
        member = self._members.get(stamp.member)
        if member is None or member.observation.stamp != stamp:
            return "noncurrent_spatial_member"
        if (member.spatial_lifecycle is None or member.spatial_lifecycle.stamp != stamp
                or member.spatial_lifecycle.native_state != "ACTIVE"):
            return "native_active_required"
        if member.spatial_join_required or not member.image_region_participant:
            return "current_rejoin_proof_required"
        return self._spatial_stamp_age(stamp, context)

    def _spatial_stamp_age(self, stamp: ObservationStamp, context: SpatialClockContext) -> str | None:
        interval = next((i for i in context.intervals if i.binding.source_id == stamp.member.source_id), None)
        if interval is None:
            return "source_clock_unavailable"
        if not interval.acquisition_min_s <= stamp.acquired_at_s <= interval.acquisition_max_s:
            return "original_acquisition_interval"
        if interval.evaluation_max_s - interval.acquisition_min_s > self.policy.max_age_s:
            return "stale_spatial_member"
        return None

    def _spatial_common_ranges(self, context: SpatialClockContext) -> dict[int, tuple[float, float, float, float]] | None:
        if (not _text(context.comparability_receipt_id) or not _text(context.common_domain)
                or context.qualification_scope not in {"synthetic", "canonical"}
                or type(context.offsets) is not tuple or type(context.valid_ranges) is not tuple
                or not len(context.offsets) == len(context.valid_ranges) == len(context.intervals)):
            return None
        offsets, validity = {}, {}
        for rows, target in ((context.offsets, offsets), (context.valid_ranges, validity)):
            for row in rows:
                if (type(row) is not tuple or len(row) != 3 or not _integer(row[0])
                        or row[0] in target or not all(_number(v) for v in row[1:]) or row[1] > row[2]):
                    return None
                target[row[0]] = row[1:]
        if set(offsets) != set(validity) or set(offsets) != {i.binding.source_id for i in context.intervals}:
            return None
        for i in context.intervals:
            low, high = validity[i.binding.source_id]
            if not low <= i.acquisition_min_s <= i.evaluation_max_s <= high:
                return None
        return {source: offsets[source] + validity[source] for source in offsets}

    def _spatial_candidate_error(self, candidate: SpatialPersonCandidate, context: SpatialClockContext,
                                 mapping: dict[int, tuple[float, float, float, float]]) -> str | None:
        if type(candidate) is not SpatialPersonCandidate:
            return "malformed_spatial_candidate"
        stamps = candidate.left, candidate.right
        bindings, ranges = [], []
        appearances = candidate.left_appearance, candidate.right_appearance
        for stamp, appearance in zip(stamps, appearances):
            error = self._stamp_error(stamp)
            if error:
                return error
            member = self._members.get(stamp.member)
            if (member is None or member.observation.stamp != stamp or member.spatial_lifecycle is None
                    or member.spatial_lifecycle.native_state != "ACTIVE"):
                return "noncurrent_spatial_member"
            error = self._spatial_stamp_age(stamp, context)
            if error:
                return error
            binding = self._bindings[stamp.member.source_id]
            interval = next(i for i in context.intervals if i.binding == binding)
            lo, hi, valid_lo, valid_hi = mapping[binding.source_id]
            ranges.append((interval.acquisition_min_s + lo, interval.acquisition_max_s + hi,
                           interval.evaluation_min_s + lo, interval.evaluation_max_s + hi))
            bindings.append(binding)
            if (type(appearance) is not AppearanceEvidence or self._stamp_error(appearance.stamp)
                    or appearance.stamp.member != stamp.member
                    or appearance.stamp.lifecycle_started_frame != stamp.lifecycle_started_frame
                    or appearance.stamp.frame_id <= member.appearance_after_frame
                    or appearance.stamp.frame_id > stamp.frame_id
                    or appearance.model != self.policy.appearance_model
                    or appearance.profile != self.policy.appearance_profile
                    or not all(_text(v) for v in (appearance.evidence_id, appearance.extraction_id))
                    or appearance.extraction_id != appearance_extraction_identity(appearance.stamp, appearance.profile)
                    or appearance.acquired_at_s != appearance.stamp.acquired_at_s
                    or appearance.clock_domain != binding.clock_domain
                    or not valid_lo <= appearance.acquired_at_s <= stamp.acquired_at_s <= valid_hi
                    or interval.evaluation_max_s - appearance.acquired_at_s > self.policy.max_age_s):
                return "original_spatial_appearance_required"
            origin = self._appearance_origins.get(stamp.member)
            if origin is not None and (appearance.stamp.frame_id < origin.stamp.frame_id
                    or (appearance.stamp.frame_id == origin.stamp.frame_id and appearance != origin)):
                return "superseded_spatial_appearance"
        if (stamps[0].member.camera_id != "kitchen" or stamps[1].member.camera_id != "family-room"
                or not self.policy.overlap_pairs):
            return "overlap_denied"
        if (max(r[3] for r in ranges) - min(r[0] for r in ranges) > self.policy.max_age_s
                or max(r[1] for r in ranges) - min(r[0] for r in ranges) > self.policy.max_age_s):
            return "qualified_time_separation"
        appearance_ranges = [(a.acquired_at_s + mapping[a.stamp.member.source_id][0],
                              a.acquired_at_s + mapping[a.stamp.member.source_id][1]) for a in appearances]
        if (max(r[1] for r in appearance_ranges) - min(r[0] for r in appearance_ranges) > self.policy.max_age_s
                or max(r[3] for r in ranges) - min(r[0] for r in appearance_ranges) > self.policy.max_age_s):
            return "qualified_appearance_time_separation"
        if (not all(_text(v) for v in (candidate.left_copy_id, candidate.right_copy_id))
                or candidate.left_copy_id == candidate.right_copy_id
                or appearances[0].extraction_id == appearances[1].extraction_id
                or not _number(candidate.similarity) or not -1 <= candidate.similarity <= 1):
            return "spatial_appearance_copy_binding"
        if (not _number(candidate.separation_m) or candidate.separation_m < 0
                or not all(_text(v) for v in (candidate.geometry_receipt_id, candidate.geometry_frame,
                                              candidate.geometry_revision))
                or candidate.calibration_revisions != tuple(b.calibration_revision for b in bindings)
                or candidate.transform_ids != tuple(b.transform_id for b in bindings)
                or type(candidate.independence_ids) is not tuple or len(candidate.independence_ids) != 2
                or not all(_text(v) for v in candidate.independence_ids)
                or candidate.independence_ids[0] == candidate.independence_ids[1]):
            return "supported_separation_required"
        return None

    def join_spatial(self, current_person_proof: SpatialCurrentPersonProof,
                     *, at_interval: SpatialClockContext) -> AssociationResult:
        if type(self.policy) is not SpatialAssociationPolicy:
            return self._reject("spatial_policy_required")
        error = self.spatial_clock_error(at_interval)
        proof = current_person_proof
        if error:
            return self._reject(error, "association_unresolved")
        if (type(proof) is not SpatialCurrentPersonProof or not _text(proof.proof_id)
                or type(proof.cohort) is not tuple or len(proof.cohort) > 128
                or any(self._stamp_error(s) for s in proof.cohort)
                or type(proof.candidates) is not tuple or not 1 <= len(proof.candidates) <= 128
                or type(proof.selected_pairs) is not tuple or not 1 <= len(proof.selected_pairs) <= 64
                or proof.evidence_scope not in {"synthetic", "canonical"}):
            return self._reject("malformed_current_person_proof", "association_unresolved")
        mapping = self._spatial_common_ranges(at_interval)
        if mapping is None or proof.evidence_scope != at_interval.qualification_scope:
            return self._reject("qualified_cross_source_time_required", "association_unresolved")
        actual = tuple(sorted((m.observation.stamp for m in self._members.values()
                               if m.observation.stamp.member.camera_id in {"kitchen", "family-room"}),
                              key=lambda s: s.member))
        if tuple(sorted(proof.cohort, key=lambda s: s.member)) != actual:
            return self._reject("complete_current_cohort_required", "association_unresolved")
        kitchens = [s for s in actual if s.member.camera_id == "kitchen"]
        families = [s for s in actual if s.member.camera_id == "family-room"]
        if len(kitchens) * len(families) > 128:
            return self._reject("competing_candidate_capacity", "association_unresolved")
        expected = {(k.member, f.member) for k in kitchens for f in families}
        candidates, viable, copies = {}, set(), {}
        for candidate in proof.candidates:
            error = self._spatial_candidate_error(candidate, at_interval, mapping)
            if error:
                return self._reject(error, "association_unresolved")
            pair = candidate.left.member, candidate.right.member
            if pair in candidates:
                return self._reject("duplicate_spatial_candidate", "association_unresolved")
            candidates[pair] = candidate
            for copy, appearance in ((candidate.left_copy_id, candidate.left_appearance),
                                     (candidate.right_copy_id, candidate.right_appearance)):
                if copy in copies and copies[copy] != appearance:
                    return self._reject("spatial_copy_alias", "association_unresolved")
                copies[copy] = appearance
            if candidate.similarity >= self.policy.min_similarity and candidate.separation_m <= self.policy.max_distance_m:
                viable.add(pair)
        if set(candidates) != expected:
            return self._reject("complete_competing_candidates_required", "association_unresolved")
        selected = proof.selected_pairs
        if (any(type(p) is not tuple or len(p) != 2 or any(type(k) is not MemberKey for k in p) for p in selected)
                or len(set(selected)) != len(selected) or not set(selected) <= viable):
            return self._reject("selected_pair_not_supported", "association_unresolved")
        if any(sum(key in pair for pair in viable) != 1 for pair in selected for key in pair):
            return self._reject("competing_person_ambiguity", "association_unresolved")
        for pair in selected:
            error = self._merge_error(pair)
            houses = [self._members[k].observation.household for k in pair]
            settled = [h for h in houses if h.identity_key is not None]
            if error or any(h.constraint != "eligible" for h in houses) or (len(settled) == 2 and settled[0] != settled[1]):
                return self._reject(error or "household_conflict", "association_unresolved")
        token = "spatial-proof:" + proof.proof_id
        fingerprint = sha256(repr(proof).encode()).hexdigest()
        prior = self._replay(token, fingerprint)
        if prior is not None:
            return prior
        evidence = "spatial-cohort:" + sha256(repr((actual, tuple(sorted(selected)))).encode()).hexdigest()
        if evidence in self._receipts:
            return self._reject("consumed_spatial_proof", "association_unresolved")
        retired = tuple(sorted({max((self._members[k].person_key for k in pair), key=lambda p: p.serial)
                                for pair in selected if len({self._members[k].person_key for k in pair}) == 2}))
        if not self._history_available((token, evidence), retired):
            return self._reject("receipt_capacity", "capacity")
        keys = []
        for pair in selected:
            people = {self._members[k].person_key for k in pair}
            keep = min(people, key=lambda p: p.serial)
            for person in people - {keep}:
                for k in tuple(self._persons.pop(person)):
                    self._members[k] = replace(self._members[k], person_key=keep)
                    self._persons[keep].add(k)
                self._retired[person] = tuple(k.source_id for k in pair)
            candidate = candidates[pair]
            for k, appearance in zip(pair, (candidate.left_appearance, candidate.right_appearance)):
                self._members[k] = replace(self._members[k], spatial_join_required=False,
                                           association_basis="qualified_spatial")
                self._appearance_origins[k] = appearance
            keys.append(keep)
        self._authority_revision += 1
        result = AssociationResult("accepted", "supported_current_person", keys[0] if len(keys) == 1 else None,
                                   retired_keys=retired, consumed_tokens=(token, evidence))
        self._remember((token, evidence), fingerprint, result, tuple(sorted({s.member.source_id for s in actual})))
        return result

    def _visual_delivery(self, context: object, *, retain: bool = True) -> tuple[str | None, dict[MemberKey, CurrentVisualIngress]]:
        if (type(context) is not CurrentVisualDeliveryContext or not _text(context.actor_id)
                or not _integer(context.sequence) or not _number(context.evaluation_monotonic_s)
                or context.evaluation_monotonic_s < 0 or type(context.ingresses) is not tuple
                or len(context.ingresses) > 128):
            return "current_visual_delivery_context", {}
        if (self._visual_actor not in {None, context.actor_id}
                or context.sequence < self._visual_sequence
                or context.evaluation_monotonic_s < self._visual_evaluation):
            return "current_visual_actor_rewind", {}
        rows = {}
        for row in context.ingresses:
            if (type(row) is not CurrentVisualIngress or self._stamp_error(row.observation)
                    or not _number(row.ingress_monotonic_s)
                    or not 0 <= row.ingress_monotonic_s <= context.evaluation_monotonic_s):
                return "current_visual_ingress_binding", {}
            key = row.observation.member
            member = self._members.get(key)
            if key in rows or member is None or member.observation.stamp != row.observation:
                return "current_visual_noncurrent_ingress", {}
            prior = self._visual_ingresses.get(key)
            if prior is not None and (
                    prior.observation == row.observation and prior.ingress_monotonic_s != row.ingress_monotonic_s
                    or prior.observation != row.observation and (
                        row.observation.frame_id <= prior.observation.frame_id
                        or row.ingress_monotonic_s < prior.ingress_monotonic_s)):
                return "current_visual_ingress_retimestamp", {}
            rows[key] = row
        protected = set(self._members) | {k for k, c in self._closed.items() if c.expires_media_s is not None}
        retained = {k: row for k, row in self._visual_ingresses.items() if k in protected}
        if retain:
            self._visual_ingresses = retained
        if len(set(retained) | set(rows)) > 128:
            return "current_visual_ingress_capacity", {}
        return None, rows

    def current_visual_liveness_error(self, stamp: ObservationStamp) -> str | None:
        """Read-only current delivery age, not capture time or appearance renewal."""
        row = self._visual_ingresses.get(stamp.member)
        if row is None or row.observation != stamp:
            return "current_visual_delivery_unavailable"
        if self._visual_evaluation - row.ingress_monotonic_s > self.policy.max_age_s:
            return "current_visual_stale_member_delivery"
        return None

    def image_region_position_delivery_error(self, stamp: ObservationStamp, context: SpatialClockContext,
                                             *, live_track_continuity: bool = False) -> str | None:
        """Position-only custody, never freshness for a new visual association.

        Worker delay does not retime the exact source-local measurement. World
        still validates that measurement and preserves its original wire expiry.
        """
        error=self.current_visual_liveness_error(stamp)
        if error!='current_visual_stale_member_delivery' or live_track_continuity is not True:
            return error
        m=self._members.get(stamp.member)
        if m is None or self.spatial_member_error(stamp,context) is not None:
            return error
        group=self._persons[m.person_key]
        for k in group:
            peer=self._members[k]
            if (peer.image_region_birth_quarantined or peer.image_region_conflict_custody
                    or peer.image_region_vote_blocked_stamp is not None or peer.image_region_vote_pending
                    or peer.image_region_split_birth is not None or peer.image_region_rejected_birth_peer is not None
                    or peer.spatial_join_required or not peer.image_region_participant):
                return error
        if len(group)==1:
            seed=m.image_region_birth_seed
            if (not m.image_region_birth_established or m.association_basis!='source_local'
                    or seed is None or seed.member!=stamp.member
                    or seed.lifecycle_started_frame!=stamp.lifecycle_started_frame
                    or seed.frame_id<=m.appearance_after_frame or not self._image_custody_valid(m)):
                return error
            return None
        if (len(group)==2 and all(self._members[k].image_region_dormant_custody
                and self._members[k].association_basis=='estimated_live_track_continuity' for k in group)
                and self._image_pair_custody(m,context,m.image_region_policy,fresh_delivery=False)):
            return None
        return error

    def _visual_member_age(self, stamp: ObservationStamp, context: SpatialClockContext) -> str | None:
        # Grouping is asynchronous estimation, not a measurement grant in the
        # latest camera cohort. An older still-current ACTIVE competitor must
        # not disappear just because another member advanced that source.
        interval = next((i for i in context.intervals if i.binding.source_id == stamp.member.source_id), None)
        if interval is None:
            return "source_clock_unavailable"
        if stamp.acquired_at_s > interval.evaluation_min_s:
            return "current_visual_future_member"
        if interval.evaluation_max_s - stamp.acquired_at_s > self.policy.max_age_s:
            return "stale_spatial_member"
        return None

    def _visual_candidate_error(self, candidate: object, context: SpatialClockContext,
                                delivery: CurrentVisualDeliveryContext,
                                ingresses: dict[MemberKey, CurrentVisualIngress]) -> str | None:
        if type(candidate) is not SpatialPersonCandidate:
            return "current_visual_candidate_type"
        if (any(v is not None for v in (candidate.separation_m, candidate.geometry_receipt_id,
                                       candidate.geometry_frame, candidate.geometry_revision))
                or candidate.calibration_revisions != () or candidate.transform_ids != ()
                or candidate.independence_ids != ()):
            return "current_visual_geometry_forbidden"
        if (not _number(candidate.similarity) or not -1 <= candidate.similarity <= 1
                or not all(_text(v) for v in (candidate.left_copy_id, candidate.right_copy_id))
                or candidate.left_copy_id == candidate.right_copy_id):
            return "current_visual_cosine_copy_binding"
        for stamp, appearance in ((candidate.left, candidate.left_appearance),
                                  (candidate.right, candidate.right_appearance)):
            if self._stamp_error(stamp):
                return "current_visual_member_binding"
            member = self._members.get(stamp.member)
            if (member is None or member.observation.stamp != stamp or member.spatial_lifecycle is None
                    or member.spatial_lifecycle.native_state != "ACTIVE"):
                return "current_visual_active_required"
            error = self._visual_member_age(stamp, context)
            if error:
                return error
            ingress = ingresses.get(stamp.member)
            if ingress is None or delivery.evaluation_monotonic_s - ingress.ingress_monotonic_s > self.policy.max_age_s:
                return "current_visual_stale_ingress"
            interval = next(i for i in context.intervals if i.binding.source_id == stamp.member.source_id)
            if (type(appearance) is not AppearanceEvidence or self._stamp_error(appearance.stamp)
                    or appearance.stamp.member != stamp.member
                    or appearance.stamp.lifecycle_started_frame != stamp.lifecycle_started_frame
                    or not member.appearance_after_frame < appearance.stamp.frame_id <= stamp.frame_id
                    or appearance.profile != self.policy.appearance_profile
                    or appearance.model != self.policy.appearance_model
                    or not all(_text(v) for v in (appearance.evidence_id, appearance.extraction_id))
                    or appearance.extraction_id != appearance_extraction_identity(appearance.stamp, appearance.profile)
                    or appearance.acquired_at_s != appearance.stamp.acquired_at_s
                    or appearance.clock_domain != stamp.clock_domain
                    or not appearance.acquired_at_s <= stamp.acquired_at_s
                    or interval.evaluation_max_s - appearance.acquired_at_s > self.policy.max_age_s):
                return "current_visual_original_appearance_required"
            origin = self._appearance_origins.get(stamp.member)
            if origin is not None and (appearance.stamp.frame_id < origin.stamp.frame_id
                    or appearance.stamp.frame_id == origin.stamp.frame_id and appearance != origin):
                return "current_visual_superseded_appearance"
            original_stamp = ingress.appearance_observation or ingress.observation
            original_time = (ingress.appearance_ingress_monotonic_s
                             if ingress.appearance_observation is not None else ingress.ingress_monotonic_s)
            if (original_stamp != appearance.stamp or not _number(original_time)
                    or not 0 <= original_time <= ingress.ingress_monotonic_s
                    or delivery.evaluation_monotonic_s - original_time > self.policy.max_age_s
                    or ingress.appearance_observation is None and ingress.appearance_ingress_monotonic_s is not None):
                return "current_visual_appearance_ingress_binding"
            previous = self._visual_ingresses.get(stamp.member)
            if (previous is not None and previous.appearance_observation == appearance.stamp
                    and previous.appearance_ingress_monotonic_s != original_time):
                return "current_visual_appearance_ingress_retimestamp"
        if (candidate.left.member.camera_id != "kitchen" or candidate.right.member.camera_id != "family-room"
                or not self.policy.overlap_pairs):
            return "current_visual_topology"
        if candidate.left_appearance.extraction_id == candidate.right_appearance.extraction_id:
            return "current_visual_copied_extraction"
        return None

    @staticmethod
    def _project_pixel(matrix: tuple[tuple[float, ...], ...], point: tuple[float, ...]) -> tuple[float, float] | None:
        homogeneous = point + (1.,)
        q = tuple(sum(a * b for a, b in zip(row, homogeneous)) for row in matrix)
        if not all(_number(v) for v in q) or q[2] <= 0:
            return None
        uv = q[0] / q[2], q[1] / q[2]
        return uv if all(_number(v) for v in uv) else None

    @staticmethod
    def _support_radius(sample: ProjectiveMotionSample) -> float | None:
        if sample.support_basis == "observed_ankle_floor" and sample.observed_support_radius_px is None:
            value = sample.pixel_uncertainty_px
        elif (sample.support_basis in {"pose_torso_support", "pose_shoulder_support"}
                and sample.pixel_uncertainty_px is None):
            value = sample.observed_support_radius_px
        else:
            return None
        return value if _number(value) and value > 0 else None

    @staticmethod
    def _body_contains(sample: ProjectiveMotionSample, uv: tuple[float, float] | None) -> bool:
        radius = PersonAssociationLedger._support_radius(sample)
        if uv is None or radius is None:
            return False
        x, y, w, h = sample.bbox_xywh
        return (x <= uv[0] <= x + w and y <= uv[1] <= y + h
                and hypot(uv[0] - sample.observed_anchor_uv[0], uv[1] - sample.observed_anchor_uv[1]) <= radius)

    def _projective_seeded_lag_support(self, witness: ProjectiveMotionPair,
                                      bindings: dict[str, ProjectiveCameraBinding]) -> bool:
        """At most9 ORIGINAL reciprocal matches; no new point/time interpolation."""
        left, right = witness.left, witness.right
        lc, rc = bindings["kitchen"], bindings["family-room"]
        lp = tuple(self._project_pixel(lc.peer_projection, s.original_point) for s in left)
        rp = tuple(self._project_pixel(rc.peer_projection, s.original_point) for s in right)
        if not all(self._body_contains(s, self._project_pixel(c.own_projection, s.original_point))
                   for side, c in ((left, lc), (right, rc)) for s in side):
            return False
        # An uninterrupted seed can stop on its NEXT original callback. Older
        # repeated positions are not a reason to discard fresh reciprocal CURRENT
        # support; this helper is unreachable for a NEW correspondence.
        if (abs(left[2].ingress_monotonic_s - right[2].ingress_monotonic_s) <= self.policy.max_age_s
                and self._body_contains(right[2], lp[2]) and self._body_contains(left[2], rp[2])):
            return True
        matches = {(i, j) for i in range(3) for j in range(3)
                   if abs(left[i].ingress_monotonic_s - right[j].ingress_monotonic_s) <= self.policy.max_age_s
                   and self._body_contains(right[j], lp[i]) and self._body_contains(left[i], rp[j])}
        unique = sorted((i, j) for i, j in matches
                        if sum(a == i for a, _ in matches) == 1 and sum(b == j for _, b in matches) == 1)
        def delta(a, b):
            return b[0] - a[0], b[1] - a[1]
        def dot(a, b):
            return a[0] * b[0] + a[1] * b[1]
        for i0, j0 in unique:
            for i1, j1 in unique:
                if not (i0 < i1 and j0 < j1 and (i1 == 2 or j1 == 2)):
                    continue
                good = True
                for side, peer, projected, a0, a1, b0, b1, camera in (
                        (left, right, lp, i0, i1, j0, j1, lc), (right, left, rp, j0, j1, i0, i1, rc)):
                    observed = delta(side[a0].observed_anchor_uv, side[a1].observed_anchor_uv)
                    target = delta(peer[b0].observed_anchor_uv, peer[b1].observed_anchor_uv)
                    mapped = delta(projected[a0], projected[a1])
                    good &= (hypot(*observed) > self._support_radius(side[a0]) + self._support_radius(side[a1])
                             and hypot(*mapped) > self._support_radius(peer[b0]) + self._support_radius(peer[b1])
                             and dot(mapped, target) > 0)
                    if a1 < 2:
                        # Only one real original frame may lead its matched peer.
                        current = delta(side[a1].observed_anchor_uv, side[2].observed_anchor_uv)
                        if projected[2] is None:
                            good = False
                        else:
                            leading = delta(projected[a1], projected[2])
                            own = tuple(self._project_pixel(camera.own_projection, s.original_point) for s in side)
                            good &= (side[2].stamp.frame_id - side[a1].stamp.frame_id == 1
                                     and dot(current, observed) > 0 and dot(leading, target) > 0
                                     and dot(delta(own[a1], own[2]), delta(own[a0], own[a1])) > 0)
                if good:
                    return True
        return False

    def _projective_evidence(self, proof: ProjectiveContinuityProof, policy: ProjectiveContinuityPolicy,
                             context: SpatialClockContext, delivery: CurrentVisualDeliveryContext,
                             ingresses: dict[MemberKey, CurrentVisualIngress], *, allow_seeded: bool = True
                             ) -> tuple[str | None, dict[tuple[MemberKey, MemberKey], bool], dict[MemberKey, tuple]]:
        states, frontiers = {}, {}
        if (type(policy) is not ProjectiveContinuityPolicy or type(proof.pairs) is not tuple
                or len(proof.pairs) * 6 > 128):
            return "projective_evidence_capacity_or_type", {}, {}
        bindings = {c.source.camera_id: c for c in policy.cameras}
        if any(self._bindings.get(c.source.source_id) != c.source for c in policy.cameras):
            return "projective_source_binding", {}, {}
        for witness in proof.pairs:
            if (type(witness) is not ProjectiveMotionPair or any(type(side) is not tuple or len(side) != 3
                    or any(type(s) is not ProjectiveMotionSample for s in side) for side in (witness.left, witness.right))):
                return "projective_three_original_samples_required", {}, {}
            endpoints = witness.left[-1].stamp, witness.right[-1].stamp
            if len({s.support_basis for s in witness.left + witness.right}) != 1:
                return "projective_same_observed_anatomy_required", {}, {}
            if any(self._stamp_error(s) or not _number(s.media_time_s) for s in endpoints):
                return "projective_original_stamp", {}, {}
            pair = endpoints[0].member, endpoints[1].member
            if (pair in states or pair[0].camera_id != "kitchen" or pair[1].camera_id != "family-room"):
                return "projective_pair_binding", {}, {}
            for side, camera in ((witness.left, "kitchen"), (witness.right, "family-room")):
                c = bindings[camera]
                endpoint = side[-1].stamp
                ingress = ingresses.get(endpoint.member)
                interval = next((i for i in context.intervals if i.binding == c.source), None)
                if ingress is None or ingress.observation != endpoint or interval is None:
                    return "projective_current_endpoint", {}, {}
                for index, s in enumerate(side):
                    m = s.measurement
                    if (self._stamp_error(s.stamp) or s.stamp.member != endpoint.member
                            or s.stamp.lifecycle_started_frame != endpoint.lifecycle_started_frame
                            or not _number(s.stamp.media_time_s) or s.stamp.media_time_s < 0
                            or not _number(s.ingress_monotonic_s)
                            or not 0 <= s.ingress_monotonic_s <= delivery.evaluation_monotonic_s
                            or side[-1].ingress_monotonic_s - s.ingress_monotonic_s > 2 * self.policy.max_age_s
                            or s.stamp.acquired_at_s > interval.evaluation_min_s
                            or endpoint.acquired_at_s - s.stamp.acquired_at_s > 2 * self.policy.max_age_s
                            or endpoint.media_time_s - s.stamp.media_time_s > 2 * self.policy.max_age_s
                            or s.raster_id != c.raster_id or s.image_size != c.image_size
                            or not _text(s.body_evidence_id) or type(m) is not ProjectiveMeasurementBinding):
                        return "projective_original_binding_or_age", {}, {}
                    if (not all(_text(v) for v in (m.evidence_id, m.measured_stage_id))
                            or m.target_frame != c.target_frame or m.world_revision != c.world_revision
                            or m.calibration_revision != c.measurement_calibration_sha256
                            or m.transform_id != c.source.transform_id or m.origin != "measured" or m.units != "m"):
                        return "projective_original_measurement_binding", {}, {}
                    if (type(s.bbox_xywh) is not tuple or len(s.bbox_xywh) != 4
                            or not all(_number(v) for v in s.bbox_xywh)
                            or min(s.bbox_xywh[:2]) < 0 or min(s.bbox_xywh[2:]) <= 0
                            or s.bbox_xywh[0] + s.bbox_xywh[2] > c.image_size[0]
                            or s.bbox_xywh[1] + s.bbox_xywh[3] > c.image_size[1]
                            or type(s.original_point) is not tuple or len(s.original_point) != 3
                            or not all(_number(v) for v in s.original_point)
                            or type(s.observed_anchor_uv) is not tuple or len(s.observed_anchor_uv) != 2
                            or not all(_number(v) for v in s.observed_anchor_uv)
                            or self._support_radius(s) is None
                            or not self._body_contains(s, s.observed_anchor_uv)):
                        return "projective_actual_body_point_uncertainty_required", {}, {}
                    if index:
                        previous = side[index - 1]
                        if (not previous.stamp.frame_id < s.stamp.frame_id
                                or not previous.stamp.acquired_at_s < s.stamp.acquired_at_s
                                or not previous.stamp.media_time_s < s.stamp.media_time_s
                                or not previous.ingress_monotonic_s < s.ingress_monotonic_s
                                or previous.body_evidence_id == s.body_evidence_id
                                or previous.measurement.evidence_id == m.evidence_id):
                            return "projective_original_order", {}, {}
                        if (s.stamp.acquired_at_s - previous.stamp.acquired_at_s > self.policy.max_age_s
                                or s.stamp.media_time_s - previous.stamp.media_time_s > self.policy.max_age_s
                                or s.ingress_monotonic_s - previous.ingress_monotonic_s > self.policy.max_age_s):
                            return "projective_historical_gap", {}, {}
                    prior = self._projective_frontiers.get(endpoint.member)
                    if prior is not None and s.stamp.frame_id == prior[0].frame_id and (
                            s.stamp != prior[0] or sha256(repr(s).encode()).hexdigest() != prior[1]):
                        return "projective_original_retimestamp", {}, {}
                if side[-1].ingress_monotonic_s != ingress.ingress_monotonic_s:
                    return "projective_endpoint_ingress", {}, {}
                if len({s.body_evidence_id for s in side}) != 3 or len({s.measurement.evidence_id for s in side}) != 3:
                    return "projective_original_evidence_alias", {}, {}
                # Historical anchors describe past motion, never fresh position
                # authority. Three genuine 5Hz originals span .40s; only their
                # endpoint retains the .35s CURRENT TTL. Two adjacent continuity
                # intervals bound the history to .70s without freshening a sample.
                if (delivery.evaluation_monotonic_s - side[-1].ingress_monotonic_s > self.policy.max_age_s
                        or interval.evaluation_max_s - endpoint.acquired_at_s > self.policy.max_age_s):
                    return "projective_current_endpoint_age", {}, {}
                historical = (delivery.evaluation_monotonic_s - side[0].ingress_monotonic_s > self.policy.max_age_s
                              or interval.evaluation_max_s - side[0].stamp.acquired_at_s > self.policy.max_age_s
                              or endpoint.media_time_s - side[0].stamp.media_time_s > self.policy.max_age_s)
                member = self._members[endpoint.member]
                if historical and (member.projective_native_floor < 0 or member.projective_delivery_floor < 0
                        or side[0].stamp.frame_id < max(member.projective_native_floor, member.projective_delivery_floor)):
                    return "projective_historical_continuity_required", {}, {}
                frontiers[endpoint.member] = (endpoint, sha256(repr(side[-1]).encode()).hexdigest(),
                                              side[0].stamp.acquired_at_s, side[0].ingress_monotonic_s,
                                              side[-1].support_basis)
            if any(abs(a.ingress_monotonic_s - b.ingress_monotonic_s) > self.policy.max_age_s
                   for a, b in zip(witness.left, witness.right)):
                return "projective_availability_lag", {}, {}
            # A NEW correspondence needs motion. An already seeded EXACT pair
            # may stop while fresh reciprocal current support continuously renews.
            # No unseeded, returned, reset, corrected, skipped or prior-changed pair
            # can obtain that privilege from merely similar/stationary samples.
            policy_fingerprint = sha256(repr(policy).encode()).hexdigest()
            people = {self._members[k].person_key for k in pair}
            seeded = allow_seeded and len(people) == 1 and self._persons[next(iter(people))] == set(pair)
            for key, side in zip(pair, (witness.left, witness.right)):
                member, prior = self._members[key], self._projective_frontiers.get(key)
                endpoint = side[-1]
                seeded &= (member.association_basis == "prior_projective_continuity"
                           and not member.spatial_join_required and prior is not None
                           and member.spatial_lifecycle is not None and member.spatial_lifecycle.native_state == "ACTIVE"
                           and prior[4] == member.person_key and prior[5] == policy_fingerprint
                           and len(prior) == 9 and prior[8] == endpoint.support_basis
                           and endpoint.stamp.lifecycle_started_frame == prior[0].lifecycle_started_frame
                           and 0 <= endpoint.stamp.frame_id - prior[0].frame_id <= 1
                           and 0 <= endpoint.stamp.acquired_at_s - prior[0].acquired_at_s <= self.policy.max_age_s
                           and 0 <= endpoint.ingress_monotonic_s - prior[6] <= self.policy.max_age_s)
            if not seeded:
                for key, side in zip(pair, (witness.left, witness.right)):
                    prior = self._projective_frontiers.get(key)
                    floor = max(self._members[key].appearance_after_frame, prior[7] if prior else -1)
                    if prior is not None and prior[5] != policy_fingerprint:
                        floor = max(floor, prior[0].frame_id)
                    if side[0].stamp.frame_id <= floor:
                        return "projective_new_motion_after_barrier_required", {}, {}
            supported = True
            for side, peer, c in ((witness.left, witness.right, bindings["kitchen"]),
                                  (witness.right, witness.left, bindings["family-room"])):
                projected = [self._project_pixel(c.peer_projection, s.original_point) for s in side]
                supported &= all(self._body_contains(s, self._project_pixel(c.own_projection, s.original_point))
                                 and self._body_contains(p, uv) for s, p, uv in zip(side, peer, projected))
                for index in (() if seeded else (1, 2)):
                    a, b = side[index - 1], side[index]
                    observed = tuple(b.observed_anchor_uv[j] - a.observed_anchor_uv[j] for j in (0, 1))
                    supported &= hypot(*observed) > self._support_radius(a) + self._support_radius(b)
                    if projected[index] is None or projected[index - 1] is None:
                        supported = False
                    else:
                        motion = tuple(projected[index][j] - projected[index - 1][j] for j in (0, 1))
                        target_motion = tuple(peer[index].observed_anchor_uv[j] - peer[index - 1].observed_anchor_uv[j] for j in (0, 1))
                        supported &= (hypot(*motion) > self._support_radius(peer[index]) + self._support_radius(peer[index - 1])
                                      and sum(a * b for a, b in zip(motion, target_motion)) > 0)
            if seeded and not supported:
                supported = self._projective_seeded_lag_support(witness, bindings)
            states[pair] = bool(supported)
        protected = set(self._members) | {k for k, c in self._closed.items() if c.expires_media_s is not None}
        if len((set(self._projective_frontiers) & protected) | set(frontiers)) > 128:
            return "projective_frontier_capacity", {}, {}
        return None, states, frontiers

    def inspect_projective_continuity(self, proof: ProjectiveContinuityProof, *,
                                     projective_policy: ProjectiveContinuityPolicy,
                                     at_interval: SpatialClockContext,
                                     delivery_context: CurrentVisualDeliveryContext) -> ProjectiveContinuityInspection:
        """Read-only original numeric gate; appearance and seeded exemptions are not decisions here."""
        if type(self.policy) is not SpatialAssociationPolicy:
            return ProjectiveContinuityInspection("spatial_policy_required")
        if (type(proof) is not ProjectiveContinuityProof or type(proof.pairs) is not tuple
                or len(proof.pairs) * 6 > 128 or type(projective_policy) is not ProjectiveContinuityPolicy):
            return ProjectiveContinuityInspection("projective_evidence_capacity_or_type")
        error = self.spatial_clock_error(at_interval)
        if error:
            return ProjectiveContinuityInspection(error)
        if (at_interval.comparability_receipt_id is not None or at_interval.common_domain is not None
                or at_interval.offsets != () or at_interval.valid_ranges != ()
                or at_interval.qualification_scope is not None):
            return ProjectiveContinuityInspection("current_visual_common_qualification_forbidden")
        error, ingresses = self._visual_delivery(delivery_context, retain=False)
        if error:
            return ProjectiveContinuityInspection(error)
        for pair in proof.pairs:
            if (type(pair) is not ProjectiveMotionPair or any(type(side) is not tuple or len(side) != 3
                    or any(type(s) is not ProjectiveMotionSample for s in side) for side in (pair.left, pair.right))):
                return ProjectiveContinuityInspection("projective_three_original_samples_required")
            for side in (pair.left, pair.right):
                if any(self._stamp_error(s.stamp) for s in side):
                    return ProjectiveContinuityInspection("projective_original_stamp")
                endpoint = side[-1].stamp
                error = self.spatial_member_error(endpoint, at_interval)
                if error:
                    return ProjectiveContinuityInspection(error)
        error, states, _ = self._projective_evidence(proof, projective_policy, at_interval,
            delivery_context, ingresses, allow_seeded=False)
        return ProjectiveContinuityInspection(error, tuple((left, right, supported)
            for (left, right), supported in states.items()))

    def projective_continuity_error(self, stamp: ObservationStamp, context: SpatialClockContext) -> str | None:
        member = self._members.get(stamp.member)
        if member is None or member.observation.stamp != stamp:
            return "projective_noncurrent_member"
        if len(self._persons[member.person_key]) == 1:
            return self.spatial_member_error(stamp, context)
        if not member.association_basis.startswith("prior_projective_continuity"):
            return None  # Separate source-local people still have ordinary local authority.
        for key in self._persons[member.person_key]:
            current, original = self._members[key], self._projective_frontiers.get(key)
            interval = next((i for i in context.intervals if i.binding.source_id == key.source_id), None)
            if (current.association_basis != "prior_projective_continuity" or current.spatial_join_required
                    or original is None or original[0] != current.observation.stamp
                    # [2]/[3] are retrospective history, not CURRENT support.
                    or interval is None or interval.evaluation_max_s - original[0].acquired_at_s > self.policy.max_age_s
                    or self._visual_evaluation - original[6] > self.policy.max_age_s):
                return "projective_current_support_required"
        return None

    def join_projective_continuity(self, proof: ProjectiveContinuityProof, *, projective_policy: ProjectiveContinuityPolicy,
                                  at_interval: SpatialClockContext, delivery_context: CurrentVisualDeliveryContext) -> AssociationResult:
        if type(proof) is not ProjectiveContinuityProof:
            return self._reject("projective_proof_required", "association_unresolved")
        result = self._join_visual(proof.visual, at_interval=at_interval, delivery_context=delivery_context,
                                    projective=proof, projective_policy=projective_policy)
        # Appearance/provenance failures can return before numeric validation.
        # They must not leave a stationary-renewal seed silently usable later.
        if result.disposition != "accepted":
            for key, member in tuple(self._members.items()):
                if member.association_basis.startswith("prior_projective_continuity"):
                    prior = self._projective_frontiers.get(key)
                    if prior is not None:
                        self._projective_frontiers[key] = prior[:7] + (max(prior[7], member.observation.stamp.frame_id),) + prior[8:]
                    self._members[key] = replace(member, association_basis="prior_projective_continuity_unresolved")
                    self._authority_revision += 1
                    self._visual_last = None
        return result

    @staticmethod
    def anatomical_qualification_error(policy: AnatomicalConsensusPolicy) -> str | None:
        if type(policy) is not AnatomicalConsensusPolicy:
            return "anatomical_policy_required"
        q = policy.qualification
        if q is None:
            return "anatomical_static_pair_qualification_unavailable"
        if (type(q) is not StaticAnatomyQualification or not _text(q.qualification_id)
                or q.scope != "static_pair_estimated_anatomical_support_only" or q.depth_convention != "camera_z_m"
                or q.projective_policy_sha256 != sha256(repr(policy.projective_policy).encode()).hexdigest()
                or type(q.artifact_sha256) is not str or len(q.artifact_sha256) != 64
                or any(c not in "0123456789abcdef" for c in q.artifact_sha256)):
            return "anatomical_static_pair_qualification_binding"
        return None

    @staticmethod
    def _anatomical_contains(capsules: tuple, uv: tuple[float, float] | None) -> bool:
        if uv is None:
            return False
        for ax, ay, bx, by, r, end_r in capsules:
            dx, dy = bx - ax, by - ay
            den = dx * dx + dy * dy
            t = max(0., min(1., ((uv[0] - ax) * dx + (uv[1] - ay) * dy) / den)) if den else 0.
            if (hypot(uv[0] - ax - t * dx, uv[1] - ay - t * dy) <= r
                    or min(hypot(uv[0] - ax, uv[1] - ay), hypot(uv[0] - bx, uv[1] - by)) <= end_r):
                return True
        return False

    def _anatomical_evidence(self, proof, policy, context, delivery, ingresses):
        error = self.anatomical_qualification_error(policy)
        if error:
            return error, {}, {}
        if (type(proof) is not AnatomicalConsensusProof or type(proof.pairs) is not tuple
                or len(proof.pairs) > 32):  # <=128 original anatomy samples.
            return "anatomical_evidence_capacity_or_type", {}, {}
        bindings = {c.source.camera_id: c for c in policy.projective_policy.cameras}
        if any(self._bindings.get(c.source.source_id) != c.source for c in bindings.values()):
            return "anatomical_source_binding", {}, {}
        states, originals = {}, {}
        for pair in proof.pairs:
            if (type(pair) is not AnatomicalPairSupport or self._stamp_error(pair.left) or self._stamp_error(pair.right)
                    or pair.left.member.camera_id != "kitchen" or pair.right.member.camera_id != "family-room"):
                return "anatomical_pair_binding", {}, {}
            keys = pair.left.member, pair.right.member
            if keys in states:
                return "anatomical_duplicate_pair", {}, {}
            sides = pair.left_originals, pair.right_originals
            for endpoint, side in zip((pair.left, pair.right), sides):
                member = self._members.get(endpoint.member)
                row = ingresses.get(endpoint.member)
                if (member is None or member.observation.stamp != endpoint or row is None or row.observation != endpoint
                        or member.spatial_lifecycle is None or member.spatial_lifecycle.native_state != "ACTIVE"
                        or delivery.evaluation_monotonic_s - row.ingress_monotonic_s > self.policy.max_age_s):
                    return "anatomical_current_endpoint_required", {}, {}
                if (type(side) is not tuple or len(side) != 2 or any(type(s) is not AnatomicalDepthSupport
                        or type(s.original) is not ProjectiveMotionSample for s in side)):
                    return "anatomical_two_original_supports_required", {}, {}
                if tuple(s.original.support_basis for s in side) != ("pose_ankle_support", "pose_torso_support"):
                    return "anatomical_ankle_torso_required", {}, {}
                camera = bindings[endpoint.member.camera_id]
                interval = next((i for i in context.intervals if i.binding == camera.source), None)
                if interval is None:
                    return "anatomical_source_clock_unavailable", {}, {}
                if not interval.acquisition_min_s <= endpoint.acquired_at_s <= interval.acquisition_max_s:
                    return "anatomical_current_source_interval_required", {}, {}
                if side[0].original.stamp != side[1].original.stamp:
                    return "anatomical_exact_original_cohort_required", {}, {}
                for support in side:
                    s, ranges = support.original, support.registered_depth_interval_m
                    m = s.measurement
                    if (self._stamp_error(s.stamp) or s.stamp.member != endpoint.member
                            or s.stamp.lifecycle_started_frame != endpoint.lifecycle_started_frame
                            or not max(member.appearance_after_frame, member.anatomical_after_frame) < s.stamp.frame_id <= endpoint.frame_id
                            or not _number(s.stamp.media_time_s) or not _number(endpoint.media_time_s)
                            or not 0 <= endpoint.media_time_s - s.stamp.media_time_s <= self.policy.max_age_s
                            or not 0 <= interval.evaluation_max_s - s.stamp.acquired_at_s <= self.policy.max_age_s
                            or not _number(s.ingress_monotonic_s)
                            or not 0 <= s.ingress_monotonic_s <= row.ingress_monotonic_s
                            or delivery.evaluation_monotonic_s - s.ingress_monotonic_s > self.policy.max_age_s
                            or s.raster_id != camera.raster_id or s.image_size != camera.image_size
                            or not _text(s.body_evidence_id) or type(m) is not ProjectiveMeasurementBinding
                            or m.target_frame != camera.target_frame or m.world_revision != camera.world_revision
                            or m.calibration_revision != camera.measurement_calibration_sha256
                            or m.transform_id != camera.source.transform_id or m.origin != "measured" or m.units != "m"
                            or not _text(m.evidence_id) or not _text(m.measured_stage_id)):
                        return "anatomical_original_binding_or_age", {}, {}
                    if ranges is None or support.depth_convention is None:
                        return "anatomical_registered_interval_unavailable", {}, {}
                    if (support.depth_convention != "camera_z_m" or type(ranges) is not tuple or len(ranges) != 2
                            or not all(_number(v) and v > 0 for v in ranges) or ranges[0] > ranges[1]
                            or type(s.original_point) is not tuple or len(s.original_point) != 3
                            or not all(_number(v) for v in s.original_point)
                            or type(s.observed_anchor_uv) is not tuple or len(s.observed_anchor_uv) != 2
                            or not all(_number(v) for v in s.observed_anchor_uv)
                            or s.pixel_uncertainty_px is not None
                            or type(s.bbox_xywh) is not tuple or len(s.bbox_xywh) != 4
                            or not all(_number(v) for v in s.bbox_xywh) or min(s.bbox_xywh[:2]) < 0
                            or min(s.bbox_xywh[2:]) <= 0 or s.bbox_xywh[0] + s.bbox_xywh[2] > camera.image_size[0]
                            or s.bbox_xywh[1] + s.bbox_xywh[3] > camera.image_size[1]
                            or type(support.capsules) is not tuple or not 1 <= len(support.capsules) <= 2
                            or any(type(c) is not tuple or len(c) != 6 or not all(_number(v) for v in c)
                                   or min(c[4:]) <= 0 for c in support.capsules)):
                        return "anatomical_original_support_binding", {}, {}
                    if not self._anatomical_contains(support.capsules, s.observed_anchor_uv):
                        return "anatomical_representative_outside_sampled_support", {}, {}
                    if member.anatomical_originals:
                        prior = member.anatomical_originals[0].original.stamp
                        if s.stamp.frame_id < prior.frame_id or (s.stamp.frame_id == prior.frame_id
                                and support not in member.anatomical_originals):
                            return "anatomical_original_retimestamp_or_rewind", {}, {}
                # Conservative exact separation proof via enclosing circles;
                # overlapping enclosures are unavailable, never independent votes.
                for a in side[0].capsules:
                    for b in side[1].capsules:
                        bound = lambda c: ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
                        ac, bc = bound(a), bound(b)
                        if hypot(ac[0] - bc[0], ac[1] - bc[1]) <= (
                                hypot(a[2] - a[0], a[3] - a[1]) / 2 + max(a[4:])
                                + hypot(b[2] - b[0], b[3] - b[1]) / 2 + max(b[4:])):
                            return "anatomical_distinct_support_regions_unavailable", {}, {}
                if endpoint.member in originals and originals[endpoint.member] != side:
                    return "anatomical_original_alias", {}, {}
                originals[endpoint.member] = side
            supported = True
            for side, peer, camera in ((sides[0], sides[1], bindings["kitchen"]),
                                       (sides[1], sides[0], bindings["family-room"])):
                for a, b in zip(side, peer):
                    s = a.original
                    own = self._project_pixel(camera.own_projection, s.original_point)
                    uv = self._project_pixel(camera.peer_projection, s.original_point)
                    z = sum(x * y for x, y in zip(camera.peer_projection[2], s.original_point + (1.,)))
                    oz = sum(x * y for x, y in zip(camera.own_projection[2], s.original_point + (1.,)))
                    x, y, w, h = b.original.bbox_xywh
                    supported &= (self._anatomical_contains(a.capsules, own)
                                  and self._anatomical_contains(b.capsules, uv)
                                  and uv is not None and x <= uv[0] <= x + w and y <= uv[1] <= y + h
                                  and a.registered_depth_interval_m[0] <= oz <= a.registered_depth_interval_m[1]
                                  and b.registered_depth_interval_m[0] <= z <= b.registered_depth_interval_m[1])
            states[keys] = bool(supported)
        if sum(len(originals.get(k,m.anatomical_originals)) for k,m in self._members.items()) > 128:
            return "anatomical_original_capacity", {}, {}
        return None, states, originals

    def anatomical_consensus_error(self, stamp, context):
        member = self._members.get(stamp.member)
        if member is None or member.observation.stamp != stamp:
            return "anatomical_noncurrent_member"
        group = self._persons[member.person_key]
        if len(group) == 1 or not member.association_basis.startswith("current_registered_anatomical_consensus"):
            return None
        for key in group:
            m = self._members[key]
            interval = next((i for i in context.intervals if i.binding.source_id == key.source_id), None)
            if (m.association_basis != "current_registered_anatomical_consensus" or m.anatomical_confirmed != m.observation.stamp
                    or interval is None or len(m.anatomical_originals) != 2
                    or any(interval.evaluation_max_s - a.original.stamp.acquired_at_s > self.policy.max_age_s
                           or self._visual_evaluation - a.original.ingress_monotonic_s > self.policy.max_age_s
                           for a in m.anatomical_originals)):
                return "anatomical_current_support_required"
        return None

    def join_anatomical_consensus(self, proof: AnatomicalConsensusProof, *, anatomical_policy: AnatomicalConsensusPolicy,
                                  at_interval: SpatialClockContext, delivery_context: CurrentVisualDeliveryContext) -> AssociationResult:
        if type(proof) is not AnatomicalConsensusProof:
            return self._reject("anatomical_proof_required", "association_unresolved")
        result = self._join_visual(proof.visual, at_interval=at_interval, delivery_context=delivery_context,
                                   anatomical=proof, anatomical_policy=anatomical_policy)
        if result.disposition != "accepted":
            for key, m in tuple(self._members.items()):
                if m.association_basis == "current_registered_anatomical_consensus":
                    self._members[key] = replace(m, association_basis="current_registered_anatomical_consensus_unresolved")
                    self._authority_revision += 1
            self._visual_last = None
        return result

    def join_current_visual(self, proof: CurrentVisualProof, *, at_interval: SpatialClockContext,
                            delivery_context: CurrentVisualDeliveryContext) -> AssociationResult:
        return self._join_visual(proof, at_interval=at_interval, delivery_context=delivery_context)

    @staticmethod
    def _region_fundamental(left, right):
        # Cache only the bounded literal immutable form accepted by the prior
        # contract. Direct mutable/malformed/exotic calls retain the original
        # arithmetic and exception behavior below. Float hex preserves signed
        # zero; explicit tags prevent equal-valued int/float keys aliasing.
        if all(type(p) is tuple and len(p) == 3 and all(
                type(row) is tuple and len(row) == 4 and all(_number(v) for v in row)
                for row in p) for p in (left, right)):
            exact = tuple(tuple(tuple(('f', v.hex()) if type(v) is float else ('i', v)
                                      for v in row) for row in p) for p in (left, right))
            return PersonAssociationLedger._region_fundamental_cached(exact, left, right)
        return PersonAssociationLedger._region_fundamental_uncached(left, right)

    @staticmethod
    @lru_cache(maxsize=16)
    def _region_fundamental_cached(exact, left, right):
        # Only projections and their immutable numeric result are retained;
        # no policy, member, source observation, SDK object or authority is held.
        return PersonAssociationLedger._region_fundamental_uncached(left, right)

    @staticmethod
    def _region_fundamental_uncached(left, right):
        """4x4 minors of exact prior cameras; no inverse/depth/metric admission."""
        def determinant(rows):
            if len(rows) == 1:
                return rows[0][0]
            return sum((-1)**j * rows[0][j] * determinant(tuple(tuple(r[k] for k in range(len(rows)) if k != j)
                       for r in rows[1:])) for j in range(len(rows)))
        if any(not any(determinant(tuple(tuple(row[j] for j in range(4) if j != dropped) for row in p)) != 0
                       for dropped in range(4)) for p in (left,right)):
            return None
        f = tuple(tuple((-1)**(i+j) * determinant(tuple(left[k] for k in range(3) if k != i)
                    + tuple(right[k] for k in range(3) if k != j)) for i in range(3)) for j in range(3))
        scale = max(abs(v) for row in f for v in row)
        return tuple(tuple(v/scale for v in row) for row in f) if scale > 0 and all(_number(v) for row in f for v in row) else None

    @staticmethod
    def _region_front_witness(left, right, p, q):
        """Finite oriented ray intersection; internal sign test, no metric output."""
        rows=tuple(tuple(uv*camera[2][j]-camera[axis][j] for j in range(4))
                   for camera,point in ((left,p),(right,q)) for axis,uv in enumerate(point))
        scales=tuple(max(abs(v) for v in row) for row in rows)
        if any(not _number(v) or v==0 for v in scales): return False
        rows=tuple(tuple(v/scale for v in row) for row,scale in zip(rows,scales))
        def det3(r):
            a,b,c=r
            return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        candidates=[]
        for indices in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
            candidates.append(tuple((-1)**j*det3(tuple(tuple(rows[i][k] for k in range(4) if k!=j) for i in indices)) for j in range(4)))
        x=max(candidates,key=lambda v:max(abs(c) for c in v))
        scale=max(abs(v) for v in x)
        if not _number(scale) or scale<=256*ulp(1.): return False
        x=tuple(v/scale for v in x)
        if abs(x[3])<=64*ulp(1.): return False
        if any(abs(sum(a*b for a,b in zip(row,x)))>256*ulp(1.) for row in rows): return False
        depths=tuple(sum(a*b for a,b in zip(camera[2],x))/x[3] for camera in (left,right))
        return all(_number(v) and v>0 for v in depths)

    @staticmethod
    def _region_feasible(f, a, b, *, cameras=None):
        """Symmetric bounded feasibility: safe outer exclusion / inner witnesses.

        Endpoint discs' cardinal points lie inside the observed capsule. Their
        convex hull witnesses feasibility; rectangular outer bounds only prove
        incompatibility. The undecided curved remainder is UNKNOWN, not inflated.
        """
        def points(c, outer):
            x,y,u,v,r,s = c
            if outer:
                lo,hi = min(x-r,u-s),max(x+r,u+s)
                bottom,top = min(y-r,v-s),max(y+r,v+s)
                return ((lo,bottom),(lo,top),(hi,bottom),(hi,top))
            return tuple(p for cx,cy,radius in ((x,y,r),(u,v,s)) for p in
                         ((cx,cy),(cx-radius,cy),(cx+radius,cy),(cx,cy-radius),(cx,cy+radius)))
        def values(pa,pb):
            lines=[tuple(row[0]*x+row[1]*y+row[2] for row in f) for x,y in pa]
            return [a*x+b*y+c for a,b,c in lines for x,y in pb]
        outside=values(points(a,True),points(b,True))
        # Arithmetic roundoff only, not a pixel tolerance or enlarged region.
        magnitude=max(1.,max(abs(v) for row in f for v in row)
                      * max(1.,max(abs(v) for p in points(a,True) for v in p))
                      * max(1.,max(abs(v) for p in points(b,True) for v in p)))
        roundoff=64*ulp(magnitude)
        if min(outside)>roundoff or max(outside)<-roundoff:
            return False
        pa,pb=points(a,False),points(b,False)
        inside=values(pa,pb)
        if cameras is None: return None
        # <=100 exact vertices and <=20 linear zero witnesses. Interpolation
        # varies one image region only, so stays within its convex inner hull.
        witnesses=[(p,q) for i,p in enumerate(pa) for j,q in enumerate(pb)
                   if abs(inside[i*len(pb)+j])<=roundoff]
        for fixed,other,transpose in ((pa,pb,False),(pb,pa,True)):
            for i,p in enumerate(fixed):
                line=[inside[j*len(pb)+i] if transpose else inside[i*len(pb)+j] for j in range(len(other))]
                lo,hi=min(range(len(line)),key=line.__getitem__),max(range(len(line)),key=line.__getitem__)
                if line[lo]<-roundoff and line[hi]>roundoff:
                    t=-line[lo]/(line[hi]-line[lo])
                    q=tuple(other[lo][k]+t*(other[hi][k]-other[lo][k]) for k in range(2))
                    witnesses.append((q,p) if transpose else (p,q))
        # Positive-only curved-disc witnesses. Literal endpoint discs are inside
        # the capsule; neither radius nor negative exclusion is enlarged.
        for own,peer,matrix,transpose in ((a,b,f,False),(b,a,tuple(zip(*f)),True)):
            for x,y,r in ((own[0],own[1],own[4]),(own[2],own[3],own[5])):
                for u,v,s in ((peer[0],peer[1],peer[4]),(peer[2],peer[3],peer[5])):
                    gx=matrix[0][0]*u+matrix[1][0]*v+matrix[2][0]
                    gy=matrix[0][1]*u+matrix[1][1]*v+matrix[2][1]
                    length=hypot(gx,gy)
                    if not _number(length) or length==0: continue
                    for sign in (-1,1):
                        p=(x+sign*r*gx/length,y+sign*r*gy/length)
                        # Move rounding-only overshoot inward, NEVER pad a disc.
                        for _ in range(4):
                            if hypot(p[0]-x,p[1]-y)<=r: break
                            p=(nextafter(p[0],x),nextafter(p[1],y))
                        if hypot(p[0]-x,p[1]-y)>r: continue
                        line=tuple(row[0]*p[0]+row[1]*p[1]+row[2] for row in matrix)
                        norm=line[0]*line[0]+line[1]*line[1]
                        if not _number(norm) or norm==0: continue
                        distance=(line[0]*u+line[1]*v+line[2])/norm
                        q=(u-distance*line[0],v-distance*line[1])
                        if all(_number(c) for c in q) and hypot(q[0]-u,q[1]-v)<=s:
                            witnesses.append((q,p) if transpose else (p,q))
        return True if any(PersonAssociationLedger._region_front_witness(*cameras,p,q) for p,q in witnesses) else None

    def _image_region_retained(self, member, context, fingerprint):
        o=member.image_region_original
        if (self._image_run_closed or member.observation.stamp.member.source_id in self._image_stopped_sources
                or o is None or not member.image_region_front_confirmed
                or member.association_basis not in {"estimated_current_image_regions","estimated_image_region_continuation"}
                or len(self._persons[member.person_key])!=2 or member.spatial_join_required
                or member.image_region_policy!=fingerprint or member.spatial_lifecycle is None
                or member.spatial_lifecycle.native_state!="ACTIVE"):
            return False
        current=member.observation.stamp
        return (o.stamp.member==current.member and o.stamp.lifecycle_started_frame==current.lifecycle_started_frame
            and o.stamp.frame_id>=max(member.appearance_after_frame+1,member.projective_native_floor,member.projective_delivery_floor)
            and o.stamp.media_time_s is not None and current.media_time_s is not None
            and 0<=current.media_time_s-o.stamp.media_time_s<=self.policy.max_age_s
            and self._visual_member_age(o.stamp,context) is None
            and self._visual_member_age(current,context) is None
            and 0<=self._visual_evaluation-o.ingress_monotonic_s<=self.policy.max_age_s
            and self.current_visual_liveness_error(current) is None)

    def _image_live_native(self, member, context, fingerprint, *, pair_only=False, dormant=False):
        """Membership lease, NOT a pose/measurement freshness extension."""
        def answer(reason=None,blocker=None):
            try:self._image_diag_check(member,'dormant' if dormant else 'fresh',reason,blocker)
            except Exception:self._image_decision_capture_errors+=1
            return reason is None
        seed=member.image_region_native_seed
        if (not seed or not member.image_region_front_confirmed or member.image_region_original is None
                or member.image_region_policy!=fingerprint or member.spatial_join_required
                or len(self._persons[member.person_key])!=2
                or member.association_basis not in {"estimated_current_image_regions",
                    "estimated_image_region_continuation","estimated_live_track_continuity"}):
            return answer('seed_or_membership_unavailable')
        current={s.member:self._members[s.member] for s in self.image_region_active_cohort()}
        required=self._persons[member.person_key] if pair_only else {s.member for s in seed}
        if (not required<=set(current) or not required<={s.member for s in seed}
                or not pair_only and set(current)!=required): return answer('participant_or_seed_roster')
        for s in seed:
            if s.member not in required: continue
            m=current[s.member]; now=m.observation.stamp
            if self._image_run_closed:return answer('run_closed',s.member)
            if now.member.source_id in self._image_stopped_sources:return answer('source_stopped',s.member)
            if now.lifecycle_started_frame!=s.lifecycle_started_frame:return answer('birth_changed',s.member)
            if m.spatial_join_required:return answer('rejoin_required',s.member)
            if s.frame_id<max(m.appearance_after_frame+1,m.projective_native_floor,m.projective_delivery_floor):return answer('seed_before_floor',s.member)
            error=self._visual_member_age(now,context)
            if error:return answer(error,s.member)
            if dormant:
                if not self._image_custody_valid(m):return answer('owner_custody_unavailable',s.member)
            else:
                error=self.current_visual_liveness_error(now)
                if error:return answer(error,s.member)
        return answer()

    def _image_owner_capsules(self, proof, delivery):
        caps=proof.owner_availability
        if type(caps) is not tuple or len(caps)>64:
            return "image_region_owner_availability_binding"
        seen=set(); unavailable=set(); graph=self._image_graph_id
        for c in caps:
            owner_ingress=c.source_ingress_monotonic_s if c.owner_ingress_monotonic_s is None else c.owner_ingress_monotonic_s
            if (type(c) is not ImageRegionOwnerAvailability or self._stamp_error(c.observation)
                    or not _text(c.graph_id) or not _integer(c.source_frame_id)
                    or not _integer(c.source_observed_at_us) or not _integer(c.owner_grace_ns)
                    or not all(_number(v) and v>=0 for v in (owner_ingress,c.source_media_time_s,c.source_ingress_monotonic_s,
                        c.availability_checked_monotonic_s,c.last_progress_age_s,c.stall_seconds))
                    or c.stall_seconds<=0 or c.availability_status not in {"running","starting","recovering","stopped","failed"}):
                return "image_region_owner_availability_binding"
            k=c.observation.member;m=self._members.get(k);cursor=self._cursors.get(k.source_id)
            if (k in seen or k.source_id in self._image_stopped_sources or self._image_run_closed
                    or m is None or m.observation.stamp!=c.observation or cursor is None
                    or graph is not None and graph!=c.graph_id
                    or c.source_frame_id<c.observation.frame_id or c.source_frame_id!=cursor[0]
                    or c.observation.media_time_s is None or cursor[2] is None
                    or not c.observation.media_time_s<=c.source_media_time_s<=cursor[2]
                    or not c.source_ingress_monotonic_s<=owner_ingress<=delivery.evaluation_monotonic_s
                    or c.availability_checked_monotonic_s!=owner_ingress):
                return "image_region_owner_availability_binding"
            graph=c.graph_id;seen.add(k)
            groups=(c.retained_members,c.active_members,c.shadow_members,c.terminated_members)
            if any(type(g) is not tuple or len(g)>64 or any(type(j) is not MemberKey
                    or (j.run_id,j.source_id,j.source_epoch)!=(k.run_id,k.source_id,k.source_epoch) for j in g)
                    or len(set(g))!=len(g) for g in groups):return "image_region_owner_availability_binding"
            if (not set(c.active_members)<=set(c.retained_members) or not set(c.shadow_members)<=set(c.active_members)
                    or set(c.terminated_members)&set(c.retained_members)):
                return "image_region_owner_availability_binding"
            ingress=self._visual_ingresses.get(k)
            if ingress is None or c.source_ingress_monotonic_s!=ingress.ingress_monotonic_s:
                return "image_region_owner_availability_binding"
            old=m.image_region_owner_availability
            old_owner=(old.source_ingress_monotonic_s if old.owner_ingress_monotonic_s is None else old.owner_ingress_monotonic_s) if old else None
            if (old is not None and not m.image_region_participant
                    and c.source_frame_id>old.source_frame_id
                    and c.source_media_time_s>=old.source_media_time_s
                    and c.source_observed_at_us>=old.source_observed_at_us
                    and owner_ingress>old_owner
                    and (owner_ingress-old_owner>self.policy.max_age_s
                        or c.source_media_time_s-old.source_media_time_s>self.policy.max_age_s)):
                # A forward gap denies this OLD-original custody, not unrelated
                # current bodies. All structural/rewind checks remain strict.
                unavailable.add(k)
                continue
            if old is not None and (c.source_frame_id<old.source_frame_id
                    or c.source_frame_id==old.source_frame_id and c!=old
                    or c.source_frame_id>old.source_frame_id and (c.source_media_time_s<old.source_media_time_s
                        or c.source_observed_at_us<old.source_observed_at_us
                        or not 0<owner_ingress-old_owner<=self.policy.max_age_s
                        or c.source_media_time_s-old.source_media_time_s>self.policy.max_age_s)):
                return "image_region_owner_availability_reordered"
        self._image_graph_id=graph
        for k in unavailable:
            m=self._members[k]
            if m.image_region_split_custody is not None:
                self._image_revoke_split_custody(m.image_region_split_custody)
            m=self._members[k]
            updated=replace(m,image_region_native_seed=(),image_region_front_confirmed=False)
            if updated!=m:
                self._members[k]=updated;self._authority_revision+=1
        for k,m in tuple(self._members.items()):
            if (k not in seen or k in unavailable) and m.image_region_owner_availability is not None:
                self._members[k]=replace(m,image_region_owner_availability=None)
                self._authority_revision+=1
        for c in caps:
            if c.observation.member in unavailable:continue
            m=self._members[c.observation.member]
            if m.image_region_owner_availability!=c:
                self._members[c.observation.member]=replace(m,image_region_owner_availability=c)
                self._authority_revision+=1
        return None

    def _image_quarantine_custody(self, member, context, fingerprint):
        return self._image_pair_custody(member,context,fingerprint,fresh_delivery=True)

    def _image_pair_custody(self, member, context, fingerprint, *, fresh_delivery):
        """B/R40-only custody of an existing pair, never a measurement lease.

        A delivery/pose frontier is not native-person death. A native frontier
        may cross the unique seed only through retained-shadow lineage already
        accepted by the ordered lifecycle owner, not an ordinary native gap.
        """
        group=self._persons[member.person_key]
        if self._image_run_closed or len(group)!=2:
            return False
        for key in group:
            m=self._members[key];now=m.observation.stamp
            seed=next((s for s in m.image_region_native_seed if s.member==key),None)
            if (seed is None or not group<={s.member for s in m.image_region_native_seed}
                    or not m.image_region_front_confirmed or m.image_region_original is None
                    or m.image_region_policy!=fingerprint or m.spatial_join_required
                    or m.association_basis not in {"estimated_current_image_regions",
                        "estimated_image_region_continuation","estimated_live_track_continuity"}
                    or not m.image_region_participant or m.spatial_lifecycle is None
                    or m.spatial_lifecycle.native_state!="ACTIVE"
                    or seed.lifecycle_started_frame!=now.lifecycle_started_frame
                    or seed.frame_id<=m.appearance_after_frame
                    or seed.frame_id<m.projective_native_floor and not m.image_region_shadow_membership
                    or self._visual_member_age(now,context) is not None
                    or fresh_delivery and self.current_visual_liveness_error(now) is not None
                    or not self._image_custody_valid(m)):
                return False
        return True

    def _image_custody_valid(self, member):
        c=member.image_region_owner_availability;s=member.observation.stamp
        reason=('run_closed' if self._image_run_closed else 'source_stopped' if s.member.source_id in self._image_stopped_sources
            else 'receipt_missing' if c is None else 'observation_mismatch' if c.observation!=s
            else 'graph_mismatch' if c.graph_id!=self._image_graph_id else 'source_not_running' if c.availability_status!='running'
            else 'not_retained' if s.member not in c.retained_members else 'not_active' if s.member not in c.active_members
            else 'shadow' if s.member in c.shadow_members else 'terminated' if s.member in c.terminated_members
            else 'media_unavailable' if s.media_time_s is None
            else 'native_grace_expired' if not 0<=c.source_media_time_s-s.media_time_s<=c.owner_grace_ns/1e9
            else 'source_availability_expired' if not 0<=self._visual_evaluation-c.availability_checked_monotonic_s+c.last_progress_age_s<c.stall_seconds else None)
        try:self._image_diag_check(member,'custody',reason,s.member)
        except Exception:self._image_decision_capture_errors+=1
        return reason is None

    def _image_split_shadow_custody(self, member, context, fingerprint):
        """OLD-original transfer only; never used by current vote/join paths."""
        group=self._persons[member.person_key]
        if self._image_run_closed or len(group)!=2:return False
        shadow=False
        for key in group:
            m=self._members[key];s=m.observation.stamp;c=m.image_region_owner_availability
            seed=next((v for v in m.image_region_native_seed if v.member==key),None)
            if (seed is None or not group<={v.member for v in m.image_region_native_seed}
                    or not m.image_region_front_confirmed or m.image_region_original is None
                    or m.image_region_policy!=fingerprint or not m.image_region_shadow_membership
                    or m.association_basis!='estimated_live_track_continuity'
                    or s.lifecycle_started_frame!=seed.lifecycle_started_frame
                    or seed.frame_id<=m.appearance_after_frame
                    or m.spatial_lifecycle is None or m.spatial_lifecycle.native_state!='ACTIVE'
                    or key.source_id in self._image_stopped_sources
                    or c is None or c.observation!=s or c.graph_id!=self._image_graph_id
                    or c.availability_status!='running' or key not in c.retained_members
                    or key not in c.active_members or key in c.terminated_members
                    or s.media_time_s is None
                    or not 0<=c.source_media_time_s-s.media_time_s<=c.owner_grace_ns/1e9
                    or not 0<=self._visual_evaluation-c.availability_checked_monotonic_s+c.last_progress_age_s<c.stall_seconds):
                return False
            interval=next((i for i in context.intervals if i.binding.source_id==key.source_id),None)
            if (interval is None or interval.binding!=self._bindings.get(key.source_id)
                    or not 0<=interval.evaluation_max_s-s.acquired_at_s<=c.owner_grace_ns/1e9):return False
            if not m.image_region_participant:
                event=m.spatial_lifecycle_update;shadow=True
                if (event is None or event.last_observation!=s or event.classification!='retained_unmeasured'
                        or event.at_frame_id!=c.source_frame_id or event.at_media_time_s!=c.source_media_time_s
                        or m.projective_native_floor!=event.at_frame_id
                        or key not in c.shadow_members):return False
            elif (m.spatial_join_required or key in c.shadow_members
                    or seed.frame_id<m.projective_native_floor
                    or self._visual_member_age(s,context) is not None):return False
        return shadow

    def _image_split_birth_pending(self, member, fingerprint):
        """A rejected view is not a second person; never a measurement grant.

        The fixed deadline is captured from real source leases at the split;
        neither observations nor subsequent healthy capsules renew this hold.
        """
        pending=member.image_region_split_birth
        if (pending is None or pending.policy!=fingerprint or self._image_run_closed
                or not self._visual_evaluation<pending.expires_monotonic_s):
            return False
        for seed,floor in zip(pending.seeds,pending.native_floors):
            m=self._members.get(seed.member)
            if (m is None or seed.member.source_id in self._image_stopped_sources
                    or m.observation.stamp.lifecycle_started_frame!=seed.lifecycle_started_frame
                    or seed.frame_id<=m.appearance_after_frame or m.spatial_join_required
                    or m.spatial_lifecycle is None or m.spatial_lifecycle.native_state!='ACTIVE'
                    or m.projective_native_floor!=floor
                    or (seed.member!=member.observation.stamp.member and m.person_key!=pending.origin)):
                return False
        return True

    def _image_revoke_split_custody(self, view):
        for key,m in tuple(self._members.items()):
            if m.image_region_split_custody is view:
                self._members[key]=replace(m,image_region_split_custody=None,
                    image_region_split_custody_native_floor=-1)
                self._authority_revision+=1

    @staticmethod
    def _image_split_deadline(caps):
        deadlines=tuple(c.availability_checked_monotonic_s+c.stall_seconds-c.last_progress_age_s for c in caps)
        return min(deadlines) if deadlines and all(_number(t) for t in deadlines) else None

    def _image_split_custody_error(self, view, context):
        error=self._image_split_custody_error_raw(view,context)
        if error is not None:
            # Failure-only observer state; never read by an authority predicate.
            try:
                guards=tuple(self._split_custody_member_diagnostic(s) for s in view.prior_seed)
                note=(view.episode_id,self._visual_sequence,self._visual_evaluation,error,guards)
                for s in view.prior_seed:
                    m=self._members.get(s.member)
                    if m is not None and (m.split_custody_refusal is None
                            or m.split_custody_refusal[:2]!=note[:2]):
                        self._members[s.member]=replace(m,split_custody_refusal=note)
            except Exception:pass
        return error

    def _split_custody_member_diagnostic(self, seed):
        m=self._members.get(seed.member)
        if m is None:return (seed.member.source_id,seed.member.raw_id,None,None,None,None,None,None,None)
        c=m.image_region_owner_availability;s=m.observation.stamp
        return (seed.member.source_id,seed.member.raw_id,s.frame_id,
            s.lifecycle_started_frame==seed.lifecycle_started_frame,
            bool(m.spatial_lifecycle and m.spatial_lifecycle.native_state=='ACTIVE'),
            seed.member.source_id in self._image_stopped_sources,
            m.image_region_participant,m.spatial_join_required,
            bool(c and c.observation==s and seed.member in c.retained_members
                and seed.member in c.active_members and seed.member not in c.terminated_members))

    def estimated_split_custody_diagnostic(self, view, *, at_interval, observed_monotonic_s):
        """Detached optional observer, NOT validation or a custody permission."""
        if (type(view) is not EstimatedSplitCustodyView or not _number(observed_monotonic_s)
                or not _number(view.owner_deadline_monotonic_s)
                or type(view.prior_seed) is not tuple or len(view.prior_seed)!=2
                or any(type(s) is not ObservationStamp for s in view.prior_seed)):
            return {'available':False}
        note=next((m.split_custody_refusal for s in view.prior_seed
            if (m:=self._members.get(s.member)) is not None and m.split_custody_refusal
            and m.split_custody_refusal[0]==view.episode_id),None)
        return {'available':True,'fixed_deadline_mono_s':view.owner_deadline_monotonic_s,
            'checked_mono_s':observed_monotonic_s,
            'expired_now':not observed_monotonic_s<view.owner_deadline_monotonic_s,
            'filter_reason':note[3] if note else None,
            'filter_sequence':note[1] if note else None,
            'filter_evaluation_mono_s':note[2] if note else None,
            'same_decision':bool(note and note[1]==self._visual_sequence),
            'current_query_reason':self._image_split_custody_error_raw(view,at_interval),
            'member_guards_basis':'captured_filter' if note else 'current_query',
            'member_guard_fields':['source','raw','frame','same_birth','native_active','stopped','participant','rejoin','exact_retained_active_owner'],
            'member_guards':[list(row) for row in (note[4] if note else
                tuple(self._split_custody_member_diagnostic(s) for s in view.prior_seed))]}

    def _image_split_custody_error_raw(self, view, context):
        if self._image_run_closed or not self._visual_evaluation<view.owner_deadline_monotonic_s:
            return 'split_custody_expired'
        if view.origin_key not in view.successor_keys or len(set(view.successor_keys))!=2:
            return 'split_custody_keys'
        reunited=view.rejoined_key is not None
        if view.native_destinations and (type(view.native_destinations) is not tuple
                or len(view.native_destinations)!=2
                or any(type(row) is not tuple or len(row)!=2 or type(row[0]) is not ObservationStamp
                    or type(row[1]) is not PhysicalPersonKey for row in view.native_destinations)
                or tuple(s for s,k in view.native_destinations)!=view.split_stamps
                or {k for s,k in view.native_destinations}!=set(view.successor_keys)):
            return 'split_custody_native_destination'
        if reunited and (view.rejoined_key!=view.origin_key
                or not _integer(view.rejoin_revision) or not view.split_revision<=view.rejoin_revision<=self._authority_revision
                or len(view.rejoined_stamps)!=2
                or {s.member for s in view.rejoined_stamps}!={s.member for s in view.prior_seed}):
            return 'split_custody_rejoin_binding'
        for seed,split in zip(view.prior_seed,view.split_stamps):
            key=seed.member;m=self._members.get(key)
            if (m is None or m.image_region_split_custody is not view
                    or m.person_key not in view.successor_keys
                    or reunited and m.person_key!=view.origin_key
                    or self._persons.get(m.person_key)!=({s.member for s in view.prior_seed} if reunited else {key})
                    or key.source_id in self._image_stopped_sources
                    or m.image_region_policy!=view.policy_identity
                    or m.observation.stamp.lifecycle_started_frame!=seed.lifecycle_started_frame
                    or seed.member!=split.member or seed.lifecycle_started_frame!=split.lifecycle_started_frame
                    or seed.frame_id<=m.appearance_after_frame
                    or m.projective_native_floor>m.image_region_split_custody_native_floor
                    or m.spatial_lifecycle is None or m.spatial_lifecycle.native_state!='ACTIVE'
                    or m.observation.household.constraint!='eligible'):
                return 'split_custody_native_binding'
            s=m.observation.stamp;c=m.image_region_owner_availability
            if (view.native_destinations and not reunited
                    and next(k for original,k in view.native_destinations if original.member==key)!=m.person_key):
                return 'split_custody_native_destination'
            if reunited:
                joined=next(j for j in view.rejoined_stamps if j.member==key)
                if (joined.lifecycle_started_frame!=s.lifecycle_started_frame or joined.frame_id>s.frame_id
                        or joined.acquired_at_s>s.acquired_at_s or joined.clock_domain!=s.clock_domain
                        or joined.clock_basis!=s.clock_basis):return 'split_custody_rejoin_binding'
            interval=next((i for i in context.intervals if i.binding.source_id==key.source_id),None)
            if (c is None or c.observation!=s or c.graph_id!=self._image_graph_id
                    or c.availability_status!='running' or key not in c.retained_members
                    or key not in c.active_members or key in c.terminated_members
                    or s.media_time_s is None or not 0<=c.source_media_time_s-s.media_time_s<=c.owner_grace_ns/1e9
                    or not 0<=self._visual_evaluation-c.availability_checked_monotonic_s+c.last_progress_age_s<c.stall_seconds
                    or interval is None or interval.binding!=self._bindings.get(key.source_id)
                    or not 0<=interval.evaluation_max_s-s.acquired_at_s<=c.owner_grace_ns/1e9):
                return 'split_custody_owner_unavailable'
            if not m.image_region_participant:
                event=m.spatial_lifecycle_update
                if (event is None or event.last_observation!=s or event.classification!='retained_unmeasured'
                        or event.at_frame_id!=c.source_frame_id or event.at_media_time_s!=c.source_media_time_s
                        or key not in c.shadow_members):return 'split_custody_shadow_binding'
            elif m.spatial_join_required or key in c.shadow_members:
                return 'split_custody_rejoin_required'
        if {self._members[s.member].person_key for s in view.prior_seed}!=(
                {view.origin_key} if reunited else set(view.successor_keys)):
            return 'split_custody_reassigned'
        return None

    def estimated_split_custodies(self, *, at_interval):
        """Current immutable old-point episodes; actual wall deadline is external.

        World must check time.monotonic() against the FIXED deadline again at
        prepare/commit, as this SDK-neutral owner never reads a host clock.
        """
        if self.spatial_clock_error(at_interval):return ()
        views={id(m.image_region_split_custody):m.image_region_split_custody
            for m in self._members.values() if m.image_region_split_custody is not None}
        return tuple(sorted((v for v in views.values() if self._image_split_custody_error(v,at_interval) is None),
            key=lambda v:v.origin_key))

    def _image_finish_split_custody(self, context, selected, fingerprint):
        views={id(m.image_region_split_custody):m.image_region_split_custody
            for m in self._members.values() if m.image_region_split_custody is not None}
        for view in views.values():
            pair=tuple(s.member for s in view.prior_seed)
            if (view.policy_identity!=fingerprint or self._image_split_custody_error(view,context) is not None
                    or any(q!=pair and set(q)&set(pair) for q in selected)):
                self._image_revoke_split_custody(view)

    def _image_rejected_birth_custody(self, member, peer):
        """Negative additional-birth provenance, NEVER presence or a point grant.

        A genuine retained/unmeasured event may advance a measurement frontier
        without ending this same native owner. Remember only that accepted
        frontier; an ordinary subsequent native gap cannot cross it.
        """
        rejected=member.image_region_rejected_birth_peer
        if rejected is None or peer is None:return None
        s=peer.observation.stamp;c=peer.image_region_owner_availability
        if (peer.person_key!=member.image_region_rejected_birth_origin
                or s.member!=rejected.member or s.lifecycle_started_frame!=rejected.lifecycle_started_frame
                or rejected.frame_id<=peer.appearance_after_frame
                or peer.spatial_lifecycle is None or peer.spatial_lifecycle.native_state!='ACTIVE'
                or self._image_run_closed or s.member.source_id in self._image_stopped_sources
                or c is None or c.observation!=s or c.graph_id!=self._image_graph_id
                or peer.projective_native_floor>member.image_region_rejected_birth_native_floor
                or c.availability_status!='running' or s.member not in c.retained_members
                or s.member not in c.active_members or s.member in c.terminated_members
                or s.media_time_s is None or not 0<=c.source_media_time_s-s.media_time_s<=c.owner_grace_ns/1e9
                or not 0<=self._visual_evaluation-c.availability_checked_monotonic_s+c.last_progress_age_s<c.stall_seconds):
            return None
        event=peer.spatial_lifecycle_update
        shadow=(event is not None and not peer.image_region_participant
            and event.last_observation==s and event.classification=='retained_unmeasured'
            and event.at_frame_id==c.source_frame_id and event.at_media_time_s==c.source_media_time_s
            and s.member in c.shadow_members
            and 0<=event.at_acquired_at_s-s.acquired_at_s<=self.policy.max_age_s
            and 0<=event.at_media_time_s-s.media_time_s<=self.policy.max_age_s)
        if shadow:return member.image_region_rejected_birth_native_floor
        if (peer.spatial_join_required or not peer.image_region_participant or s.member in c.shadow_members
                or peer.projective_native_floor>member.image_region_rejected_birth_native_floor):return None
        return member.image_region_rejected_birth_native_floor

    def stop_image_region_source(self, binding, *, graph_id):
        self._image_support_targets=()
        if type(binding) is not SourceBinding or self._bindings.get(binding.source_id)!=binding or not _text(graph_id) or self._image_graph_id not in (None,graph_id):
            return self._reject("image_region_owner_availability_binding","association_unresolved")
        self._image_stopped_sources.add(binding.source_id);self._authority_revision+=1
        return AssociationResult("observed","image_region_source_stopped")

    def close_image_region_run(self):
        self._image_support_targets=()
        self._image_run_closed=True;self._authority_revision+=1
        return AssociationResult("observed","image_region_run_closed")

    def _image_region_evidence(self, proof, policy, context, delivery, ingresses):
        if (type(policy) is not ProjectiveContinuityPolicy
                or any(self._bindings.get(c.source.source_id) != c.source for c in policy.cameras)):
            return "image_region_source_binding", {}, {}, {}, {}, {}
        if (type(proof.originals) is not tuple or len(proof.originals)>64
                or any(type(o) is not ImageRegionOriginal for o in proof.originals)):
            return "image_region_original_capacity_or_type", {}, {}, {}, {}, {}
        cameras={c.source.camera_id:c for c in policy.cameras}
        originals, valid, integrity, evidence_ids = {}, {}, {}, set()
        for o in proof.originals:
            if self._stamp_error(o.stamp) or o.stamp.member in originals or o.stamp not in proof.cohort:
                return "image_region_original_binding", {}, {}, {}, {}, {}
            key=o.stamp.member
            if not _text(o.body_evidence_id) or o.body_evidence_id in evidence_ids:
                return "image_region_original_alias", {}, {}, {}, {}, {}
            evidence_ids.add(o.body_evidence_id)
            originals[key]=o
            m=self._members.get(key); c=cameras[key.camera_id]; ingress=ingresses.get(key)
            age_error=self._visual_member_age(o.stamp,context)
            good=(not self._image_run_closed and key.source_id not in self._image_stopped_sources
                and m is not None and m.observation.stamp==o.stamp and ingress is not None
                and age_error in (None,"stale_spatial_member")
                and ingress.ingress_monotonic_s==o.ingress_monotonic_s
                and o.stamp.frame_id>m.appearance_after_frame
                and o.raster_id==c.raster_id and o.prior_id==c.prior_id and o.image_size==c.image_size
                and _text(o.body_evidence_id) and type(o.bbox_xywh) is tuple and len(o.bbox_xywh)==4
                and all(_number(v) for v in o.bbox_xywh)
                and type(o.patches) is tuple and 1<=len(o.patches)<=2
                and len({p.anatomy for p in o.patches if type(p) is ObservedImageRegion})==len(o.patches))
            if good:
                x,y,w,h=o.bbox_xywh
                good=0<=x<x+w<=o.image_size[0] and 0<=y<y+h<=o.image_size[1]
            joints=o.semantic_joints
            good=good and type(joints) is tuple and len(joints)<=4 and all(
                    type(j) is tuple and len(j)==4 and type(j[0]) is int and j[0] in (5,6,11,12)
                    and all(_number(v) for v in j[1:]) and 0<=j[1]<o.image_size[0]
                    and 0<=j[2]<o.image_size[1] and .35<=j[3]<=1.
                    for j in joints) and all(a[0]<b[0] for a,b in zip(joints,joints[1:]))
            heads=o.head_joints
            good=good and type(heads) is tuple and len(heads)<=5 and all(
                type(j) is tuple and len(j)==4 and type(j[0]) is int and 0<=j[0]<=4
                and all(_number(v) for v in j[1:]) and 0<=j[1]<o.image_size[0]
                and 0<=j[2]<o.image_size[1] and .35<=j[3]<=1. for j in heads
                ) and all(a[0]<b[0] for a,b in zip(heads,heads[1:]))
            for p in o.patches if type(o.patches) is tuple and len(o.patches)<=2 else ():
                good=good and (type(p) is ObservedImageRegion and p.anatomy in {"pose_torso_support","pose_shoulder_support"}
                    and type(p.anchor_uv) is tuple and len(p.anchor_uv)==2 and type(p.capsule) is tuple and len(p.capsule)==6
                    and all(_number(v) for v in p.anchor_uv+p.capsule) and min(p.capsule[4:])>0)
                if good:
                    u,v=p.anchor_uv; a,b,d,e,r,s=p.capsule
                    good=(0<=min(a-r,d-s) and max(a+r,d+s)<=o.image_size[0]
                        and 0<=min(b-r,e-s) and max(b+r,e+s)<=o.image_size[1]
                        and x<=u<=x+w and y<=v<=y+h and self._anatomical_contains((p.capsule,),p.anchor_uv))
            if (m is not None and m.image_region_original is not None
                    and (m.image_region_original.stamp==o.stamp and m.image_region_original!=o
                         or m.image_region_original.stamp!=o.stamp and m.image_region_original.body_evidence_id==o.body_evidence_id)):
                good=False
            # Validate binding/geometry once, independently of ordinary expiry.
            # A stale valid original is UNKNOWN, never a fresh geometric vote.
            integrity[key]=good
            fresh=good and age_error is None and delivery.evaluation_monotonic_s-o.ingress_monotonic_s<=self.policy.max_age_s
            valid[key]=fresh
            if self._image_decision_pending is not None:
                diagnostic=next(r for r in self._image_decision_pending["members"] if r["member"]==asdict(key))
                try: original_copy=self._image_diag_input_original(o,m.observation.stamp) if m is not None else None
                except Exception:
                    original_copy=None;self._image_decision_capture_errors+=1
                diagnostic.update(input_status="current_valid" if fresh else "current_stale" if good else "current_invalid",
                    effective_status="fresh" if fresh else "unavailable",valid=fresh,
                    input_is_current=bool(m is not None and m.observation.stamp==o.stamp),
                    input_validation_passed=good,
                    current_input_original=original_copy)
        fingerprint=sha256(repr(policy).encode()).hexdigest()
        for stamp in proof.cohort:
            key=stamp.member
            if key not in originals and self._image_region_retained(self._members[key],context,fingerprint):
                o=self._members[key].image_region_original
                originals[key]=o
                valid[key]=o.body_evidence_id not in evidence_ids
                evidence_ids.add(o.body_evidence_id)
        if self._image_decision_pending is not None:
            for diagnostic in self._image_decision_pending["members"]:
                key=MemberKey(**diagnostic["member"])
                if diagnostic["input_status"]=="absent":
                    diagnostic.update(effective_status="retained" if valid.get(key) else "unavailable",valid=bool(valid.get(key)))
        left=cameras["kitchen"]; right=cameras["family-room"]
        f=self._region_fundamental(left.own_projection,left.peer_projection)
        reverse=self._region_fundamental(right.own_projection,right.peer_projection)
        states={}; cue_states={}
        for a in proof.cohort:
            for b in proof.cohort:
                if a.member.camera_id!="kitchen" or b.member.camera_id!="family-room": continue
                pair=a.member,b.member
                states[pair]=None
                cue_states[pair]={}
                diagnostic={"left":asdict(pair[0]),"right":asdict(pair[1]),"cues":[],"state":None,
                    "possible":None,"unique":None,"supported":None}
                if self._image_decision_pending is not None:
                    self._image_decision_pending["pairs"].append(diagnostic)
                    self._image_decision_pairs[pair]=diagnostic
                if f is None or reverse is None or not valid.get(pair[0]) or not valid.get(pair[1]): continue
                ap={p.anatomy:p for p in originals[pair[0]].patches}
                bp={p.anatomy:p for p in originals[pair[1]].patches}
                shared=ap.keys() & bp.keys()
                votes=[]
                for name in sorted(shared):
                    forward=self._region_feasible(f,ap[name].capsule,bp[name].capsule,cameras=(left.own_projection,left.peer_projection))
                    backward=self._region_feasible(reverse,bp[name].capsule,ap[name].capsule,cameras=(right.own_projection,right.peer_projection))
                    votes.extend((forward,backward))
                    cue_states[pair][name]=(forward,backward)
                    diagnostic["cues"].append({"anatomy":name,"forward":forward,"reverse":backward})
                if votes:
                    states[pair]=False if False in votes else True if all(v is True for v in votes) else None
                retained=any(originals[k].stamp!=self._members[k].observation.stamp for k in pair)
                if retained and (self._members[pair[0]].person_key!=self._members[pair[1]].person_key
                        or not all(self._members[k].image_region_front_confirmed for k in pair)):
                    # Retained data may block a competing edge but never support
                    # a NEW edge. Unknown remains a competitor, not an exclusion.
                    if states[pair] is True: states[pair]=None
        return None, states, originals, valid, integrity, cue_states

    @staticmethod
    def _semantic_joint_terms(forward, reverse, left, right):
        """The unchanged four nonnegative directional distances per joint."""
        if forward is None or reverse is None:return None
        values=[]
        for a,b in zip(left,right):
            x,y=(a[1],a[2],1.),(b[1],b[2],1.)
            for matrix,own,peer in ((forward,x,y),(reverse,y,x)):
                line=tuple(sum(matrix[i][j]*own[j] for j in range(3)) for i in range(3))
                back=tuple(sum(matrix[i][j]*peer[i] for i in range(3)) for j in range(3))
                numerator=abs(sum(peer[i]*line[i] for i in range(3)))
                norms=(hypot(line[0],line[1]),hypot(back[0],back[1]))
                if not all(_number(n) and n>0 for n in norms):return None
                values.extend(numerator/n for n in norms)
        return tuple(values) if all(_number(v) for v in values) else None

    @staticmethod
    def _semantic_joint_mean(forward, reverse, left, right):
        """Frozen COMPLETE-quartet residual, not uncertainty or identity truth."""
        values=PersonAssociationLedger._semantic_joint_terms(forward,reverse,left,right)
        return sum(values)/16 if values is not None and len(values)==16 else None

    @staticmethod
    def _semantic_joint_lower_bound(forward, reverse, left, right):
        """Partial named correspondences can EXCLUDE, never positively join.

        Missing terms are unknown and nonnegative; the full denominator stays
        sixteen. Inputs have already passed original integrity/currentness.
        """
        a={j[0]:j for j in left};b={j[0]:j for j in right}
        shared=sorted(a.keys() & b.keys())
        values=PersonAssociationLedger._semantic_joint_terms(forward,reverse,
            tuple(a[i] for i in shared),tuple(b[i] for i in shared))
        return sum(values)/16 if values and len(values)<16 else None

    @staticmethod
    def _image_observed_supports_separate(a,b):
        """Positive disjoint CURRENT head AND full torso supports, not IDs.

        Enclosures describe observed points, not confidence/covariance. Strict
        parent-interior required joints cannot have been clipped to the parent.
        Overlap, weak/missing heads and duplicate bodies remain UNKNOWN.
        """
        def supports(o):
            if tuple(j[0] for j in o.semantic_joints)!=(5,6,11,12):return None
            head={j[0]:j[1:3] for j in o.head_joints}
            body={j[0]:j[1:3] for j in o.semantic_joints}
            x,y,w,h=o.bbox_xywh
            inside=lambda p:x<p[0]<x+w and y<p[1]<y+h
            if not all(inside(p) for p in body.values()):return None
            discs=[]
            for ids in ((0,1,3),(0,2,4)):
                if not all(i in head for i in ids):continue
                ps=tuple(head[i] for i in ids)
                if not all(inside(p) for p in ps) or len(set(ps))!=3:return None
                lo=tuple(min(p[i] for p in ps) for i in (0,1))
                hi=tuple(max(p[i] for p in ps) for i in (0,1))
                radius=hypot(hi[0]-lo[0],hi[1]-lo[1])/2
                if radius<=0:return None
                discs.append(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,radius))
            if not discs:return None
            shoulder=tuple((body[5][i]+body[6][i])/2 for i in (0,1))
            hip=tuple((body[11][i]+body[12][i])/2 for i in (0,1))
            radius=max(hypot(*(body[5][i]-body[6][i] for i in (0,1))),
                hypot(*(body[11][i]-body[12][i] for i in (0,1))))/2
            return (discs,shoulder,hip,radius) if radius>0 else None
        left,right=supports(a),supports(b)
        if left is None or right is None:return False
        if not all(hypot(p[0]-q[0],p[1]-q[1])>p[2]+q[2]
            for p in left[0] for q in right[0]):return False
        def point_segment(p,u,v):
            dx,dy=v[0]-u[0],v[1]-u[1];norm=dx*dx+dy*dy
            t=max(0.,min(1.,((p[0]-u[0])*dx+(p[1]-u[1])*dy)/norm)) if norm else 0.
            return hypot(p[0]-u[0]-t*dx,p[1]-u[1]-t*dy)
        u,v=left[1:3];p,q=right[1:3]
        cross=lambda a,b,c:(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        if cross(u,v,p)*cross(u,v,q)<=0 and cross(p,q,u)*cross(p,q,v)<=0:return False
        distance=min(point_segment(u,p,q),point_segment(v,p,q),point_segment(p,u,v),point_segment(q,u,v))
        return distance>left[3]+right[3]

    def _image_provisional_extras(self,originals,valid,integrity,context,fingerprint,native_boxes=()):
        """Birth-only ambiguity: positively overlapping torso, unresolved head.

        This does not assert SAME person or physical absence. The approved
        recall tradeoff delays an unestablished head-occluded overlapping
        newcomer. Existing tracks and new reliable-head proposals remain
        untouched; an already provisional body needs independent head support.
        All anchors come from the pre-birth CURRENT snapshot, never iteration
        order; two weak unestablished bodies cannot suppress each other. Once
        observed, competition remains a negative fact UNTIL positive current
        independent anatomy, not the disappearance of its supporting input.
        """
        def supports(o):
            x,y,w,h=o.bbox_xywh
            inside=lambda p:x<p[0]<x+w and y<p[1]<y+h
            head={j[0]:j[1:3] for j in o.head_joints}
            discs=[]
            for ids in ((0,1,3),(0,2,4)):
                if (not all(i in head for i in ids) or len({head[i] for i in ids})!=3
                        or not all(inside(head[i]) for i in ids)):continue
                lo=tuple(min(head[i][axis] for i in ids) for axis in (0,1))
                hi=tuple(max(head[i][axis] for i in ids) for axis in (0,1))
                discs.append(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,hypot(hi[0]-lo[0],hi[1]-lo[1])/2))
            reliable=bool(discs)
            body={j[0]:j[1:3] for j in o.semantic_joints}
            if tuple(body)!=(5,6,11,12) or not all(inside(p) for p in body.values()):
                return (reliable,None,None,0.,discs)
            shoulder=tuple((body[5][i]+body[6][i])/2 for i in (0,1))
            hip=tuple((body[11][i]+body[12][i])/2 for i in (0,1))
            radius=max(hypot(*(body[5][i]-body[6][i] for i in (0,1))),
                hypot(*(body[11][i]-body[12][i] for i in (0,1))))/2
            return (reliable,shoulder,hip,radius,discs)
        def distance(a,b):
            def point_segment(p,u,v):
                dx,dy=v[0]-u[0],v[1]-u[1];norm=dx*dx+dy*dy
                t=max(0.,min(1.,((p[0]-u[0])*dx+(p[1]-u[1])*dy)/norm)) if norm else 0.
                return hypot(p[0]-u[0]-t*dx,p[1]-u[1]-t*dy)
            u,v=a[1:3];p,q=b[1:3]
            cross=lambda a,b,c:(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            intersects=(cross(u,v,p)*cross(u,v,q)<=0 and cross(p,q,u)*cross(p,q,v)<=0
                and all(max(min(u[i],v[i]),min(p[i],q[i]))<=min(max(u[i],v[i]),max(p[i],q[i])) for i in (0,1)))
            return 0. if intersects else min(point_segment(u,p,q),point_segment(v,p,q),
                point_segment(p,u,v),point_segment(q,u,v))
        def back_head(o,support):
            # A restricted return/box-basis release enclosure, not a general
            # NEW-birth head qualification or SAME. All joints passed the
            # original's finite/current/raster/confidence integrity checks.
            if support[3]<=0:return None
            head={j[0]:j[1:3] for j in o.head_joints}
            if not all(i in head for i in (3,4)):return None
            body={j[0]:j[1:3] for j in o.semantic_joints}
            x,y,w,h=o.bbox_xywh;a,b=head[3],head[4]
            if (not all(x<p[0]<x+w and y<p[1]<y+h for p in (a,b))
                    or max(a[1],b[1])>=min(body[5][1],body[6][1])):return None
            dx,dy=body[6][0]-body[5][0],body[6][1]-body[5][1]
            shoulder2=dx*dx+dy*dy;ear2=(a[0]-b[0])**2+(a[1]-b[1])**2
            center=((a[0]+b[0])/2,(a[1]+b[1])/2)
            projection=(center[0]-body[5][0])*dx+(center[1]-body[5][1])*dy
            if not 0<ear2<=shoulder2 or not 0<projection<shoulder2:return None
            return (*center,hypot(a[0]-b[0],a[1]-b[1])/2)
        current={}
        for k,o in originals.items():
            m=self._members[k];s=m.observation.stamp
            if (not valid.get(k) or not integrity.get(k) or o.stamp!=s
                    or not m.image_region_participant or not self._image_custody_valid(m)
                    or self.spatial_member_error(s,context) is not None
                    or self._visual_member_age(s,context) is not None
                    or self.current_visual_liveness_error(s) is not None):continue
            current[k]=supports(o)
        boxes={}
        for box in native_boxes:
            if not self._image_native_box_shape(box):continue
            s=box.stamp;k=s.member;m=self._members.get(k);ingress=self._visual_ingresses.get(k)
            if (m is None or m.observation.stamp!=s or not m.image_region_participant
                    or m.spatial_lifecycle is None or m.spatial_lifecycle.native_state!='ACTIVE'
                    or not self._image_custody_valid(m) or ingress is None or ingress.observation!=s
                    or ingress.ingress_monotonic_s!=box.ingress_monotonic_s
                    or self.spatial_member_error(s,context) is not None
                    or self._visual_member_age(s,context) is not None
                    or self.current_visual_liveness_error(s) is not None):continue
            boxes[k]=box
        box_anchors={k for k in boxes if self._members[k].image_region_birth_established
            and not self._members[k].image_region_birth_quarantined
            and self._members[k].image_region_provisional_extra is None
            and self._members[k].image_region_fragment_provisional is None}
        def aligned_box(k,peer):
            a,b=originals[k],boxes[peer]
            return (k.source_id==peer.source_id and k.camera_id==peer.camera_id and k.source_epoch==peer.source_epoch
                and (a.stamp.frame_id,a.stamp.acquired_at_s,a.stamp.media_time_s,a.ingress_monotonic_s,a.raster_id,a.prior_id,a.image_size)
                ==(b.stamp.frame_id,b.stamp.acquired_at_s,b.stamp.media_time_s,b.ingress_monotonic_s,b.raster_id,b.prior_id,b.image_size))
        def enclosure(support):
            a,b=support[1:3];r=support[3]
            return min(a[0],b[0])-r,min(a[1],b[1])-r,max(a[0],b[0])+r,max(a[1],b[1])+r
        anchors={k for k,support in current.items()
            if support[3]>0 and (self._members[k].image_region_birth_established or support[0])}
        # An ambiguous return cannot bootstrap another return's anchor. Freeze
        # qualification before this pass changes any negative fact.
        return_anchors={k for k in anchors if current[k][0]
            and self._members[k].image_region_birth_established
            and self._members[k].image_region_fragment_return is None
            and self._members[k].image_region_fragment_provisional is None
            and self.image_region_error(self._members[k].observation.stamp,context) is None}
        def same_originals(k,peer):
            a,b=originals[k],originals[peer]
            return (k.source_id==peer.source_id and k.camera_id==peer.camera_id
                and k.source_epoch==peer.source_epoch
                and (a.stamp.frame_id,a.stamp.acquired_at_s,a.stamp.media_time_s,a.raster_id,a.image_size)
                ==(b.stamp.frame_id,b.stamp.acquired_at_s,b.stamp.media_time_s,b.raster_id,b.image_size))
        provisional=set()
        for k,m in tuple(self._members.items()):
            s=m.observation.stamp;fact=m.image_region_provisional_extra;support=current.get(k)
            if fact is not None and (fact.policy!=fingerprint or fact.original.member!=k
                    or fact.original.lifecycle_started_frame!=s.lifecycle_started_frame
                    or m.image_region_birth_established):
                fact=None
            if not m.image_region_birth_established:
                competing=tuple(sorted(peer for peer in anchors-{k}
                    if support is not None and support[3]>0 and same_originals(k,peer)
                    and distance(support,current[peer])<support[3]+current[peer][3]))
                # A qualified head is not necessarily independent. For this
                # extra case BOTH current supports must positively overlap,
                # against an already established qualified owner, never a
                # just-born reliable-head peer or an UNKNOWN/missing head.
                birth_competing=(tuple(peer for peer in competing if peer in return_anchors
                    and any(hypot(a[0]-b[0],a[1]-b[1])<a[2]+b[2]
                        for a in support[4] for b in current[peer][4]))
                    if support is not None and support[0] else competing)
                if fact is None and support is not None and birth_competing:
                    a=originals[k]
                    fact=_ProvisionalExtraBirth(s,a.body_evidence_id,a.raster_id,a.image_size,
                        tuple(originals[peer].stamp for peer in birth_competing),fingerprint)
                box_scope_overflow=False
                if support is not None and not support[0] and support[3]>0:
                    l,t,r,b=enclosure(support)
                    box_peers=tuple(sorted(peer for peer in box_anchors-{k} if aligned_box(k,peer)
                        and (lambda q:q[0]<l and q[1]<t and r<q[0]+q[2] and b<q[1]+q[3])(boxes[peer].bbox_xywh)))
                    if fact is None and box_peers:
                        a=originals[k]
                        fact=_ProvisionalExtraBirth(s,a.body_evidence_id,a.raster_id,a.image_size,
                            tuple(boxes[peer].stamp for peer in box_peers),fingerprint,'current_native_box')
                    elif fact is not None:
                        # A still-NEW proposal cannot escape by moving from an
                        # old torso OR box anchor into another current extent.
                        # Keep the original basis, stamp and its release proof;
                        # additional box competition is a separate conjunct.
                        by={p.member:p for p in (fact.anchors if fact.basis=='current_native_box' else fact.box_anchors)}
                        for peer in box_peers:by.setdefault(peer,boxes[peer].stamp)
                        box_scope_overflow=len(set(by)|{p.member for p in fact.anchors})>64
                        if not box_scope_overflow:
                            field='anchors' if fact.basis=='current_native_box' else 'box_anchors'
                            fact=replace(fact,**{field:tuple(by[p] for p in sorted(by))})
                if fact is not None:
                    # Same observed head enclosures as the existing positive
                    # separation helper. Reliability/overlap/UNKNOWN is not
                    # independent anatomy and is never a native SAME claim.
                    independent_head=(support is not None and support[0]
                        and originals[k].raster_id==fact.raster_id and originals[k].image_size==fact.image_size
                        and all(prior.member in current and current[prior.member][0]
                            and self._members[prior.member].observation.stamp.lifecycle_started_frame==prior.lifecycle_started_frame
                            and same_originals(k,prior.member)
                            and all(hypot(a[0]-b[0],a[1]-b[1])>a[2]+b[2]
                                for a in support[4] for b in current[prior.member][4])
                            for prior in fact.anchors))
                    disjoint=(support is not None and support[3]>0 and not competing
                        and originals[k].raster_id==fact.raster_id and originals[k].image_size==fact.image_size
                        and all(prior.member in current and current[prior.member][3]>0
                            and self._members[prior.member].observation.stamp.lifecycle_started_frame==prior.lifecycle_started_frame
                            and same_originals(k,prior.member)
                            and distance(support,current[prior.member])>support[3]+current[prior.member][3]
                            for prior in fact.anchors))
                    release=(independent_head or disjoint) and not box_scope_overflow
                    required_boxes=fact.anchors if fact.basis=='current_native_box' else fact.box_anchors
                    if required_boxes:
                        box_current=(not box_scope_overflow and support is not None and support[3]>0
                            and originals[k].raster_id==fact.raster_id and originals[k].image_size==fact.image_size
                            and all(prior.member in boxes and aligned_box(k,prior.member)
                                and boxes[prior.member].stamp.lifecycle_started_frame==prior.lifecycle_started_frame
                                for prior in required_boxes))
                        disjoint=False
                        if box_current:
                            l,t,r,b=enclosure(support)
                            disjoint=all((lambda q:r<q[0] or q[0]+q[2]<l or b<q[1] or q[1]+q[3]<t)(boxes[p.member].bbox_xywh)
                                for p in required_boxes)
                        front=(box_current and support[0] and all(p.member in current and current[p.member][0]
                            and same_originals(k,p.member)
                            and all(hypot(a[0]-b[0],a[1]-b[1])>a[2]+b[2]
                                for a in support[4] for b in current[p.member][4]) for p in required_boxes))
                        back=back_head(originals[k],support) if box_current else None
                        independent_head=(box_current and (front or back is not None
                            and all(p.member in current and current[p.member][0] and current[p.member][3]>0
                                and same_originals(k,p.member)
                                and all(hypot(back[0]-h[0],back[1]-h[1])>back[2]+h[2]
                                    for h in current[p.member][4]) for p in required_boxes)))
                        release=(independent_head or disjoint) and (fact.basis=='current_native_box' or release)
                    if release:fact=None
                    else:provisional.add(k)
            if fact!=m.image_region_provisional_extra:
                self._members[k]=replace(m,image_region_provisional_extra=fact)
                self._authority_revision+=1
            # Separate position-only veto for a genuinely observed SHADOW ->
            # current same-birth return. Never demote native identity or the
            # established birth, and never turn competition into SAME.
            m=self._members[k];old=m.image_region_fragment_provisional;fact=old
            if fact is not None and (fact.policy!=fingerprint or fact.original.member!=k
                    or fact.original.lifecycle_started_frame!=s.lifecycle_started_frame):fact=None
            if (fact is None and m.image_region_fragment_return==s and support is not None
                    and not support[0] and support[3]>0):
                competing=tuple(sorted(peer for peer in return_anchors-{k}
                    if same_originals(k,peer) and distance(support,current[peer])<support[3]+current[peer][3]))
                if competing:
                    o=originals[k]
                    fact=_ProvisionalExtraBirth(s,o.body_evidence_id,o.raster_id,o.image_size,
                        tuple(originals[peer].stamp for peer in competing),fingerprint)
            if fact is not None:
                aligned=(support is not None and originals[k].raster_id==fact.raster_id
                    and originals[k].image_size==fact.image_size
                    and all(prior.member in current and same_originals(k,prior.member)
                        and self._members[prior.member].observation.stamp.lifecycle_started_frame==prior.lifecycle_started_frame
                        for prior in fact.anchors))
                head=(aligned and support[0] and all(current[prior.member][0]
                    and all(hypot(a[0]-b[0],a[1]-b[1])>a[2]+b[2]
                        for a in support[4] for b in current[prior.member][4]) for prior in fact.anchors))
                back=(back_head(originals[k],support) if aligned
                    and m.image_region_birth_established and fact.anchors else None)
                independent_back=(back is not None and all(current[prior.member][0]
                    and current[prior.member][3]>0
                    and all(hypot(back[0]-b[0],back[1]-b[1])>back[2]+b[2]
                        for b in current[prior.member][4]) for prior in fact.anchors))
                torso=(aligned and support[3]>0 and all(current[prior.member][3]>0
                    and distance(support,current[prior.member])>support[3]+current[prior.member][3]
                    for prior in fact.anchors))
                if head or independent_back or torso:fact=None
                else:provisional.add(k)
            if fact!=old:
                self._members[k]=replace(m,image_region_fragment_provisional=fact)
                self._authority_revision+=1
        return provisional

    def _image_native_box_shape(self,b):
        return (type(b) is ImageRegionNativeBox and self._stamp_error(b.stamp) is None
            and _number(b.ingress_monotonic_s) and b.ingress_monotonic_s>=0
            and _text(b.raster_id) and _text(b.prior_id) and type(b.image_size) is tuple
            and len(b.image_size)==2 and all(_integer(v,1) for v in b.image_size)
            and type(b.bbox_xywh) is tuple and len(b.bbox_xywh)==4 and all(_number(v) for v in b.bbox_xywh)
            and 0<=b.bbox_xywh[0]<b.bbox_xywh[0]+b.bbox_xywh[2]<=b.image_size[0]
            and 0<=b.bbox_xywh[1]<b.bbox_xywh[1]+b.bbox_xywh[3]<=b.image_size[1])

    def _image_split_local_survivor(self,pair,originals,valid,integrity,context,fingerprint):
        """Keep only an already established current local endpoint's old key.

        The peer's genuine native/delivery gap still requires rejoin. It cannot
        erase the other endpoint's independently current source observation.
        No cross-view seed, birth fact, measurement or lease is created here.
        """
        stopped=tuple(k for k in pair if k.source_id in self._image_stopped_sources)
        if stopped:
            if len(stopped)!=1:return None
        elif not all(self._image_custody_valid(self._members[k]) for k in pair):return None
        continuous=[]
        for k in pair:
            m=self._members[k];s=m.observation.stamp;o=originals.get(k)
            seed=next((q for q in m.image_region_native_seed if q.member==k),None)
            if stopped:
                birth=m.image_region_birth_seed
                if (not self._image_custody_valid(m) or birth is None or birth.member!=k
                        or birth.lifecycle_started_frame!=s.lifecycle_started_frame
                        or not m.appearance_after_frame<birth.frame_id<=s.frame_id
                        or m.spatial_lifecycle is None or m.spatial_lifecycle.native_state!='ACTIVE'):
                    continue
            if (m.image_region_birth_established and not m.image_region_birth_quarantined
                    and not m.image_region_conflict_custody and m.image_region_vote_blocked_stamp is None
                    and not m.image_region_vote_pending and m.image_region_split_birth is None
                    and m.image_region_rejected_birth_peer is None and m.image_region_front_confirmed
                    and m.image_region_policy==fingerprint and seed is not None
                    and seed.lifecycle_started_frame==s.lifecycle_started_frame
                    and seed.frame_id>=max(m.appearance_after_frame+1,m.projective_native_floor,m.projective_delivery_floor)
                    and o is not None and o.stamp==s and valid.get(k) and integrity.get(k)
                    and self.spatial_member_error(s,context) is None
                    and self._visual_member_age(s,context) is None
                    and self.current_visual_liveness_error(s) is None):
                continuous.append(k)
        if len(continuous)!=1:return None
        keep=continuous[0];other=next(k for k in pair if k!=keep);m=self._members[other]
        # An explicit terminal peer source is not evidence against this SAME
        # current native endpoint. Detach the stopped peer, retaining only the
        # live endpoint's existing birth/key; no stopped point is transferred.
        if stopped:return keep if other==stopped[0] else None
        seed=next((q for q in m.image_region_native_seed if q.member==other),None)
        if (m.image_region_birth_established or seed is None or m.image_region_policy!=fingerprint
                or seed.lifecycle_started_frame!=m.observation.stamp.lifecycle_started_frame
                or m.spatial_lifecycle is None or m.spatial_lifecycle.native_state!='ACTIVE'
                or not (m.spatial_join_required or seed.frame_id<max(m.projective_native_floor,m.projective_delivery_floor))):
            return None
        return keep

    def _image_local_occupancy(self,k,comparisons,states,originals,valid,integrity,
                               quarantine_pairs,context,fingerprint,established,exclusions):
        """No cross-view join: all possible peers are occupied, nonvoting.

        A current independently separated local body is not erased by a peer
        already retained in another established local body's component.
        """
        pending=tuple(p for p in comparisons if states.get(p) is not False)
        if not pending:return False
        candidate=self._members[k]
        if (candidate.image_region_vote_blocked_stamp is not None or candidate.image_region_vote_pending
                or candidate.image_region_conflict_custody):return False
        def current(key):
            m=self._members[key];o=originals.get(key)
            return (o is not None and valid.get(key) and integrity.get(key)
                and o.stamp==m.observation.stamp and not m.spatial_join_required
                and m.image_region_participant and self._image_custody_valid(m)
                and self._visual_member_age(o.stamp,context) is None
                and self.current_visual_liveness_error(o.stamp) is None)
        if not current(k):return False
        peers={next(j for j in p if j!=k) for p in pending}
        # Prove separation from EVERY same-peer local competitor, including
        # pending ones. Duplicate candidate rows cannot each manufacture a
        # third person merely because both are distinct from the old body.
        rivals={j for p,state in states.items() if state is not False and set(p)&peers
            for j in p if j.camera_id==k.camera_id and j!=k}
        for other in rivals:
            if not current(other):return False
            a,b=originals[k],originals[other]
            if (a.stamp.frame_id!=b.stamp.frame_id or a.stamp.acquired_at_s!=b.stamp.acquired_at_s
                    or a.stamp.media_time_s!=b.stamp.media_time_s or a.raster_id!=b.raster_id
                    or a.image_size!=b.image_size or not self._image_observed_supports_separate(a,b)):return False
        for pair in pending:
            peer=next(j for j in pair if j!=k);m=self._members[peer]
            group=self._persons[m.person_key]
            if (len(group)!=2 or k in group or not current(peer)
                    or m.image_region_vote_blocked_stamp!=m.observation.stamp or m.image_region_vote_pending
                    or m.image_region_birth_quarantined or m.image_region_conflict_custody
                    or not self._image_pair_custody(m,context,fingerprint,fresh_delivery=True)):return False
            other=next(j for j in group if j!=peer);local=self._members[other]
            own_pair=tuple(sorted(group,key=lambda j:j.source_id))
            if (other.camera_id!=k.camera_id or other.source_id!=k.source_id or not current(other)
                    or own_pair not in quarantine_pairs or other not in established
                    or self.image_region_error(local.observation.stamp,context) is not None
                    or not group<=established
                    or any(self._members[j].image_region_split_birth is not None for j in group)):return False
        # Other edges still need literal complete exclusions. UNKNOWN is never
        # silently discarded because one occupied peer happened to be present.
        return all(states.get(p) is not False or p in exclusions and all(valid.get(j) and integrity.get(j) for j in p)
            for p in comparisons)

    def image_region_error(self, stamp, context):
        if not self._image_support_targets:return self._image_region_support_error(stamp,context)
        try:
            target=next((v for v in self._image_support_targets if v[0]==stamp
                and v[1].get('stage')=='unobserved_stage' and v[3]==self._authority_revision),None)
        except Exception:target=None
        if target is None:return self._image_region_support_error(stamp,context)
        row,sink=target[1:3]
        self._image_support_active=row
        try:
            result=self._image_region_support_error(stamp,context)
        finally:
            self._image_support_active=None
        try:
            row.update(stage='actual_post_join_membership_check',reason=result,
                evaluation_monotonic_s=self._visual_evaluation,
                cause='accepted' if result is None else row.get('cause','unobserved_branch'),
                members=self._image_support_members(stamp,context))
            while len(json.dumps(sink,allow_nan=False).encode())>8192:
                sink['records'].pop();sink['omitted_records']+=1
        except Exception:
            try:
                row.clear();row.update(stage='diagnostic_unavailable')
                sink['errors']+=1
            except Exception:pass
        return result

    def _image_region_support_error(self, stamp, context):
        member=self._members.get(stamp.member)
        if member is None or member.observation.stamp!=stamp:
            self._image_support_note('noncurrent_member')
            return "image_region_noncurrent_member"
        if self._image_run_closed or stamp.member.source_id in self._image_stopped_sources:
            self._image_support_note('run_or_source_stopped',stamp.member)
            return "image_region_current_support_required"
        if member.image_region_fragment_provisional is not None:
            self._image_support_note('fragment_provisional')
            return "image_region_ambiguous_member"
        if member.image_region_birth_quarantined:
            self._image_support_note('birth_quarantined')
            return "image_region_ambiguous_member"
        if member.image_region_conflict_custody:
            self._image_support_note('conflict_custody')
            return "image_region_contradictory_cue_member"
        if member.image_region_vote_blocked_stamp is not None:
            self._image_support_note('vote_blocked')
            return "image_region_ambiguous_member"
        group=self._persons[member.person_key]
        if len(group)==1: return None
        # Existing native-pair custody can admit this endpoint's own current
        # position while its peer's pending vote remains blocked above. It is
        # not a new SAME decision or freshness for the nonvoting peer.
        if all(self._members[k].image_region_dormant_custody for k in group):
            if (self._image_pair_custody(member,context,member.image_region_policy,fresh_delivery=False)
                    and self._spatial_stamp_age(stamp,context) is None
                    and self.image_region_position_delivery_error(stamp,context,live_track_continuity=True) is None):return None
            self._image_support_note('group_dormant_custody')
            return "image_region_current_support_required"
        # Only an actually classified CURRENT R40 quarantine can use the narrow
        # custody fallback. New observations never clear/reissue that decision.
        if (any(self._members[k].image_region_vote_blocked_stamp is not None for k in group)
                and all(self._members[k].association_basis=="estimated_live_track_continuity"
                    and self._members[k].image_region_pair_hypothesis for k in group)):
            classified=all((self._members[k].image_region_vote_blocked_stamp is None or
                        self._members[k].image_region_vote_blocked_stamp==self._members[k].observation.stamp
                        and not self._members[k].image_region_vote_pending) for k in group)
            if classified and (all(self._image_live_native(self._members[k],context,member.image_region_policy,
                    pair_only=True) for k in group)
                    or self._image_quarantine_custody(member,context,member.image_region_policy)):
                return None
            self._image_support_note('group_vote_classification' if not classified else 'group_vote_custody')
            return "image_region_current_support_required"
        if all(self._members[k].image_region_shadow_membership
                and self._members[k].association_basis=="estimated_live_track_continuity"
                and self._members[k].image_region_front_confirmed
                and self._members[k].image_region_policy==member.image_region_policy
                and self._members[k].spatial_lifecycle is not None
                and self._members[k].spatial_lifecycle.native_state=="ACTIVE" for k in group):
            # Membership custody is not a vote from the shadow or a fresh pose.
            if (member.image_region_participant and not member.spatial_join_required
                    and member.spatial_lifecycle is not None and member.spatial_lifecycle.native_state=="ACTIVE"
                    and self._spatial_stamp_age(stamp,context) is None
                    and self.current_visual_liveness_error(stamp) is None):
                return None
            self._image_support_note('group_shadow_current_support')
            return "image_region_current_support_required"
        for key in group:
            m=self._members[key]; o=m.image_region_original
            if (m.association_basis not in {"estimated_current_image_regions","estimated_image_region_continuation","estimated_live_track_continuity"} or o is None
                    or m.spatial_lifecycle is None or m.spatial_lifecycle.native_state!="ACTIVE"
                    or m.spatial_join_required or m.image_region_policy!=member.image_region_policy
                    or not (self._image_live_native(m,context,member.image_region_policy,pair_only=m.image_region_pair_hypothesis)
                        if m.association_basis=="estimated_live_track_continuity"
                        else self._image_region_retained(m,context,member.image_region_policy))):
                self._image_support_note('group_member_support',key)
                return "image_region_current_support_required"
        return None

    def _image_appearance_support(self, proof, states, originals, valid, integrity, context, delivery, ingresses):
        """Positive joint corroboration only; no appearance cannot-link evidence."""
        rows=proof.appearances
        if not rows or len(proof.cohort)>64:return set()
        # Honest absent/partial pose is UNKNOWN. Supplied malformed or expired
        # geometry is not absence and cannot be excused by another positive cue.
        if any(not integrity.get(o.stamp.member) or not valid.get(o.stamp.member)
                or originals[o.stamp.member].stamp!=self._members[o.stamp.member].observation.stamp
                for o in proof.originals):return set()
        by={r.appearance.stamp.member:r for r in rows}
        if len(by)!=len(rows) or set(by)!={s.member for s in proof.cohort}:return set()
        endpoints={s.member:s for s in proof.cohort}
        local=dict(ingresses)
        for stamp in proof.cohort:
            row=by[stamp.member];a=row.appearance;m=self._members[stamp.member]
            if (a.stamp.member!=stamp.member or a.profile!=self.policy.appearance_profile
                    or a.stamp.frame_id==stamp.frame_id and a.stamp!=stamp
                    or a.model!=self.policy.appearance_model or len(row.vector)!=a.profile.dimensions
                    or stamp.member not in local
                    or not self._image_custody_valid(m)):
                return set()
            local[stamp.member]=replace(local[stamp.member],appearance_observation=a.stamp,
                appearance_ingress_monotonic_s=row.ingress_monotonic_s)
        viable=set()
        for pair in states:
            a,b=(by[k] for k in pair)
            aa=sum(v*v for v in a.vector);bb=sum(v*v for v in b.vector)
            norm=aa*bb
            if not _number(norm) or norm<=0:return set()
            score=sum(x*y for x,y in zip(a.vector,b.vector))/norm**.5
            if not _number(score):return set()
            score=max(-1.,min(1.,score))
            candidate=SpatialPersonCandidate(endpoints[pair[0]],endpoints[pair[1]],
                a.appearance,b.appearance,score,a.appearance.extraction_id,b.appearance.extraction_id)
            if self._visual_candidate_error(candidate,context,delivery,local):return set()
            if score>=self.policy.min_similarity:viable.add(pair)
        return {p for p in viable if states[p] is True
            and all(sum(k in q for q in viable)==1 for k in p)
            and all(valid.get(k) and k in originals
                and originals[k].stamp==self._members[k].observation.stamp for k in p)}

    def join_estimated_image_regions(self, proof: EstimatedImageRegionProof, *, projective_policy: ProjectiveContinuityPolicy,
                                    at_interval: SpatialClockContext, delivery_context: CurrentVisualDeliveryContext,
                                    live_track_continuity: bool = False,
                                    contradictory_cue_quarantine: bool = False,
                                    semantic_joint_max_mean_px: float | None = None,
                                    unresolved_birth_quarantine: bool = False) -> AssociationResult:
        if type(live_track_continuity) is not bool:
            return self._reject("image_region_live_track_boolean_required","association_unresolved")
        if type(contradictory_cue_quarantine) is not bool or contradictory_cue_quarantine and not live_track_continuity:
            return self._reject("image_region_contradictory_cue_opt_in_required","association_unresolved")
        if semantic_joint_max_mean_px is not None and (not _number(semantic_joint_max_mean_px)
                or semantic_joint_max_mean_px<=0 or not live_track_continuity):
            return self._reject("semantic_joint_threshold_opt_in_required","association_unresolved")
        if type(unresolved_birth_quarantine) is not bool or unresolved_birth_quarantine and (
                semantic_joint_max_mean_px is None or not live_track_continuity):
            return self._reject("image_region_birth_opt_in_required","association_unresolved")
        if self._image_run_closed:
            return self._reject("image_region_run_closed","association_unresolved")
        # Optional malformed/incomplete corroboration cannot amplify repr/hash
        # work or relax the original geometry decision. No vector is diagnosed.
        if type(proof) is EstimatedImageRegionProof:
            boxes=proof.native_boxes
            if (type(boxes) is not tuple or len(boxes)>64
                    or any(not self._image_native_box_shape(b) for b in boxes)
                    or len({b.stamp.member for b in boxes})!=len(boxes)):
                proof=replace(proof,native_boxes=())
            rows=proof.appearances
            if (type(rows) is not tuple or len(rows)>64 or any(type(r) is not ImageRegionAppearance
                    or type(r.appearance) is not AppearanceEvidence or self._stamp_error(r.appearance.stamp)
                    or not all(_text(v) for v in (r.appearance.evidence_id,r.appearance.extraction_id,
                        r.appearance.model,r.appearance.clock_domain)) or not _number(r.appearance.acquired_at_s)
                    or type(r.appearance.profile) is not AppearanceProfile
                    or r.appearance.profile!=self.policy.appearance_profile
                    or not _number(r.ingress_monotonic_s) or r.ingress_monotonic_s<0
                    or type(r.vector) is not tuple or len(r.vector)!=self.policy.appearance_profile.dimensions
                    or len(r.vector)>256 or any(type(v) is not float or not _number(v) for v in r.vector)
                    for r in rows)):
                proof=replace(proof,appearances=())
        if (type(proof) is EstimatedImageRegionProof and (type(proof.owner_availability) is not tuple
                or len(proof.owner_availability)>64 or any(type(c) is not ImageRegionOwnerAvailability
                    or self._stamp_error(c.observation) or not _text(c.graph_id)
                    or not _text(c.availability_status)
                    or not all(_integer(v) for v in (c.source_frame_id,c.source_observed_at_us,c.owner_grace_ns))
                    or any(not _number(v) for v in (c.source_media_time_s,c.source_ingress_monotonic_s,
                        c.availability_checked_monotonic_s,c.last_progress_age_s,c.stall_seconds))
                    or c.owner_ingress_monotonic_s is not None and not _number(c.owner_ingress_monotonic_s)
                    or any(type(g) is not tuple or len(g)>64 or any(type(k) is not MemberKey
                        or not _text(k.run_id) or not _text(k.camera_id)
                        or not all(_integer(v) for v in (k.source_id,k.source_epoch,k.raw_id,k.lifecycle_generation)) for k in g)
                        for g in (c.retained_members,c.active_members,c.shadow_members,c.terminated_members))
                    for c in proof.owner_availability))):
            return self._reject("image_region_owner_availability_binding","association_unresolved")
        try:self._image_diag_begin(proof,at_interval,delivery_context)
        except Exception:
            self._image_decision_pending=None;self._image_decision_capture_errors+=1
        if type(proof) is not EstimatedImageRegionProof:
            result=self._reject("image_region_proof_required","association_unresolved")
        else:
            result=self._join_visual(CurrentVisualProof(proof.proof_id,proof.cohort,(),()),at_interval=at_interval,
                delivery_context=delivery_context, regions=proof, region_policy=projective_policy,
                region_live=live_track_continuity is True,region_conflict=contradictory_cue_quarantine,
                region_semantic=semantic_joint_max_mean_px,region_birth=unresolved_birth_quarantine)
        if result.disposition not in {"accepted","observed"}:
            for view in tuple({id(m.image_region_split_custody):m.image_region_split_custody
                    for m in self._members.values() if m.image_region_split_custody is not None}.values()):
                self._image_revoke_split_custody(view)
            for key,m in tuple(self._members.items()):
                if m.association_basis in {"estimated_current_image_regions","estimated_image_region_continuation","estimated_live_track_continuity"}:
                    self._members[key]=replace(m,association_basis="estimated_current_image_regions_unresolved",
                        image_region_conflict_custody=False,
                        image_region_pair_hypothesis=False)
                    self._authority_revision+=1
            self._visual_last=None
        try:self._image_diag_finish(result)
        except Exception:self._image_decision_capture_errors+=1
        finally:
            self._image_decision_pending=None;self._image_decision_pairs={}
        return result

    def _join_visual(self, proof: CurrentVisualProof, *, at_interval: SpatialClockContext,
                     delivery_context: CurrentVisualDeliveryContext, projective: ProjectiveContinuityProof | None = None,
                     projective_policy: ProjectiveContinuityPolicy | None = None,
                     anatomical: AnatomicalConsensusProof | None = None,
                     anatomical_policy: AnatomicalConsensusPolicy | None = None,
                     regions: EstimatedImageRegionProof | None = None,
                     region_policy: ProjectiveContinuityPolicy | None = None,
                     region_live: bool = False, region_conflict: bool = False,
                     region_semantic: float | None = None, region_birth: bool = False) -> AssociationResult:
        """Revocable ESTIMATE only; no household permit, metric or common-time proof."""
        if type(self.policy) is not SpatialAssociationPolicy:
            return self._reject("spatial_policy_required", "association_unresolved")
        error = self.spatial_clock_error(at_interval)
        if error:
            return self._reject(error, "association_unresolved")
        if (at_interval.comparability_receipt_id is not None or at_interval.common_domain is not None
                or at_interval.offsets != () or at_interval.valid_ranges != ()
                or at_interval.qualification_scope is not None):
            return self._reject("current_visual_common_qualification_forbidden", "association_unresolved")
        if (type(proof) is not CurrentVisualProof or not _text(proof.proof_id)
                or type(proof.cohort) is not tuple or len(proof.cohort) > 128
                or any(self._stamp_error(s) for s in proof.cohort)
                or type(proof.candidates) is not tuple or len(proof.candidates) > 128
                or type(proof.selected_pairs) is not tuple or len(proof.selected_pairs) > 64):
            return self._reject("current_visual_proof_binding", "association_unresolved")
        error, ingresses = self._visual_delivery(delivery_context)
        if error:
            return self._reject(error, "association_unresolved")
        digest = sha256(repr((proof, at_interval, delivery_context, projective, projective_policy,
                             anatomical, anatomical_policy, regions, region_policy, region_live,region_conflict,region_semantic,region_birth)).encode()).hexdigest()
        if delivery_context.sequence == self._visual_sequence:
            if self._visual_last is not None and self._visual_last[:2] == (digest, self._authority_revision):
                return self._visual_last[2]
            return self._reject("current_visual_sequence_reused", "association_unresolved")
        # Record one process's delivery frontier, never acquisition offsets.
        self._visual_actor = delivery_context.actor_id
        self._visual_sequence = delivery_context.sequence
        self._visual_evaluation = delivery_context.evaluation_monotonic_s
        self._visual_last = None
        for key, row in ingresses.items():
            previous = self._visual_ingresses.get(key)
            member = self._members[key]
            if (previous is None or member.projective_delivery_floor < 0
                    or row.ingress_monotonic_s - previous.ingress_monotonic_s > self.policy.max_age_s
                    or row.observation.lifecycle_started_frame != previous.observation.lifecycle_started_frame):
                cause=('delivery_previous_unavailable' if previous is None else
                    'delivery_initial' if member.projective_delivery_floor<0 else
                    'delivery_ingress_gap' if row.ingress_monotonic_s-previous.ingress_monotonic_s>self.policy.max_age_s else 'delivery_birth_change')
                self._members[key] = replace(member, projective_delivery_floor=row.observation.frame_id,
                    floor_advance_note=self._image_floor_note(cause,previous.observation if previous else None,row.observation,
                        previous.ingress_monotonic_s if previous else None,row.ingress_monotonic_s))
            # Bind current ingress immediately, but keep a known original-copy
            # ingress until the new appearance is validated below. A rejected
            # use may never erase the cache's original monotonic provenance.
            self._visual_ingresses[key] = replace(row,
                appearance_observation=previous.appearance_observation if previous else None,
                appearance_ingress_monotonic_s=previous.appearance_ingress_monotonic_s if previous else None)
        stale_corrected = []
        if projective is not None or anatomical is not None:
            for person in tuple(sorted(self._persons)):
                group = self._persons[person]
                if len(group) != 2 or not any(self._members[k].association_basis.startswith(("prior_projective_continuity", "current_registered_anatomical_consensus")) for k in group):
                    continue
                stale = [k for k in group if self.current_visual_liveness_error(self._members[k].observation.stamp) == "current_visual_stale_member_delivery"]
                live = [k for k in group if self.current_visual_liveness_error(self._members[k].observation.stamp) is None]
                if len(stale) != 1 or len(live) != 1:
                    continue
                if self._owned_key_count() >= self.policy.max_entities:
                    for k in group:
                        self._members[k] = replace(self._members[k], association_basis="prior_projective_continuity_unresolved")
                    self._authority_revision += 1
                    return self._reject("projective_stale_split_capacity", "capacity")
                # Delivery expiry revokes ESTIMATED association, not native life.
                # Keep the continuing source's physical key; no fabricated absence.
                detached = self._allocate()
                key = stale[0]
                group.remove(key)
                self._persons[detached] = {key}
                self._members[key] = replace(self._members[key], person_key=detached, association_basis="source_local")
                self._members[live[0]] = replace(self._members[live[0]], association_basis="source_local")
                self._authority_revision += 1
                stale_corrected.append(key)
        actual = tuple(sorted((m.observation.stamp for m in self._members.values()
                               if m.observation.stamp.member.camera_id in {"kitchen", "family-room"}
                               and m.spatial_lifecycle is not None and m.spatial_lifecycle.native_state == "ACTIVE"
                               and (regions is None or m.image_region_participant)
                               and (projective is None and anatomical is None or self.current_visual_liveness_error(m.observation.stamp) != "current_visual_stale_member_delivery")
                               and (regions is not None or self._visual_member_age(m.observation.stamp, at_interval) != "stale_spatial_member")), key=lambda s: s.member))
        if tuple(sorted(proof.cohort, key=lambda s: s.member)) != actual:
            return self._reject("current_visual_complete_cohort_required", "association_unresolved")
        kitchens = [s for s in actual if s.member.camera_id == "kitchen"]
        families = [s for s in actual if s.member.camera_id == "family-room"]
        if len(kitchens) * len(families) > 128:
            return self._reject("current_visual_candidate_capacity", "association_unresolved")
        expected = {(k.member, f.member) for k in kitchens for f in families}
        candidates, viable, copies = {}, set(), {}
        for candidate in proof.candidates:
            error = self._visual_candidate_error(candidate, at_interval, delivery_context, ingresses)
            if error:
                return self._reject(error, "association_unresolved")
            pair = candidate.left.member, candidate.right.member
            if pair in candidates:
                return self._reject("current_visual_duplicate_candidate", "association_unresolved")
            candidates[pair] = candidate
            for copy, appearance in ((candidate.left_copy_id, candidate.left_appearance),
                                     (candidate.right_copy_id, candidate.right_appearance)):
                if copy in copies and copies[copy] != appearance:
                    return self._reject("current_visual_copy_alias", "association_unresolved")
                copies[copy] = appearance
            if candidate.similarity >= self.policy.min_similarity:
                viable.add(pair)
        if regions is None and set(candidates) != expected:
            return self._reject("current_visual_complete_candidates_required", "association_unresolved")
        selected = proof.selected_pairs
        if (any(type(p) is not tuple or len(p) != 2 or any(type(k) is not MemberKey for k in p) for p in selected)
                or len(set(selected)) != len(selected) or not set(selected) <= set(candidates)):
            return self._reject("current_visual_selected_pair_binding", "association_unresolved")
        supported = {p for p in viable if all(sum(k in edge for edge in viable) == 1 for k in p)}
        appearance_supported = supported.copy()
        states, frontiers = {}, {}
        if projective is not None:
            error, states, frontiers = self._projective_evidence(projective, projective_policy, at_interval, delivery_context, ingresses)
            # Missing/malformed cues never renew a previous positive group.
            for member in tuple(self._members.values()):
                if member.association_basis.startswith("prior_projective_continuity"):
                    key = member.observation.stamp.member
                    self._members[key] = replace(member, association_basis="prior_projective_continuity_unresolved")
                    if self._members[key] != member:
                        self._authority_revision += 1
            if error:
                return self._reject(error, "association_unresolved")
            if not set(states) <= set(candidates):
                return self._reject("projective_candidate_binding", "association_unresolved")
            supported &= {p for p, accepted in states.items() if accepted}
        anatomy_originals = {}
        region_originals = {}
        live_pairs=set()
        hypothesis_pairs=set()
        quarantine_pairs=set()
        quarantine_votes=set()
        dormant_pairs=set()
        conflict_pairs=set()
        provisional_extras=set()
        if regions is not None:
            error, states, region_originals, region_valid, region_integrity, cue_states = self._image_region_evidence(regions, region_policy, at_interval, delivery_context, ingresses)
            if error:
                return self._reject(error,"association_unresolved")
            candidates={p:None for p in expected}
            image_states=states.copy();semantic_scores={};semantic_lower_bounds={}
            if region_semantic is not None:
                cameras={c.source.camera_id:c for c in region_policy.cameras}
                left,right=cameras['kitchen'],cameras['family-room']
                ff=self._region_fundamental(left.own_projection,left.peer_projection)
                rr=self._region_fundamental(right.own_projection,right.peer_projection)
                for p in expected:
                    score=None;lower=None;torso=cue_states.get(p,{}).get('pose_torso_support')
                    if all(region_valid.get(k) and k in region_originals
                            and region_originals[k].stamp==self._members[k].observation.stamp
                            and region_originals[k].stamp.media_time_s is not None for k in p) and abs(
                                region_originals[p[0]].stamp.media_time_s-region_originals[p[1]].stamp.media_time_s)<=.10:
                        score=self._semantic_joint_mean(ff,rr,region_originals[p[0]].semantic_joints,region_originals[p[1]].semantic_joints)
                        if score is None:
                            lower=self._semantic_joint_lower_bound(ff,rr,region_originals[p[0]].semantic_joints,region_originals[p[1]].semantic_joints)
                    semantic_scores[p]=score
                    semantic_lower_bounds[p]=lower
                    states[p]=(False if torso is not None and False in torso else
                        False if lower is not None and lower>=region_semantic else
                        None if score is None else False if score>=region_semantic else
                        True if torso==(True,True) else None)
            possible={p for p in expected if states.get(p) is not False}
            supported={p for p in possible if states.get(p) is True and all(sum(k in q for q in possible)==1 for k in p)}
            if region_semantic is not None:
                supported={p for p in supported if all(semantic_scores.get(q) is not None
                    and semantic_scores[p]<semantic_scores[q] for q in expected if q!=p and set(q)&set(p))}
            appearance_joint_pairs=set()
            if region_live:
                error=self._image_owner_capsules(regions,delivery_context)
                if error:return self._reject(error,"association_unresolved")
                fingerprint=sha256(repr(region_policy if region_semantic is None else (region_policy,region_semantic)).encode()).hexdigest()
                if region_birth:
                    boxes=regions.native_boxes
                    if any(b.stamp not in regions.cohort or b.stamp.member.camera_id not in cameras
                            or (b.prior_id,b.raster_id,b.image_size)!=(cameras[b.stamp.member.camera_id].prior_id,
                                cameras[b.stamp.member.camera_id].raster_id,cameras[b.stamp.member.camera_id].image_size)
                            for b in boxes):boxes=()
                    provisional_extras=self._image_provisional_extras(region_originals,region_valid,
                        region_integrity,at_interval,fingerprint,boxes)
                appearance_joint_pairs=self._image_appearance_support(regions,states,region_originals,region_valid,region_integrity,
                    at_interval,delivery_context,ingresses)
                supported |= appearance_joint_pairs
                # Invalid supplied originals are not an honest missing-pose gap.
                invalid=any(not region_integrity.get(o.stamp.member,False) for o in regions.originals)
                supplied={o.stamp.member for o in regions.originals}
                for p in expected:
                    # NEW opt-in custody policy. FALSE stays FALSE; neither
                    # camera is identified as correct, so BOTH votes are denied.
                    if (region_semantic is None and region_conflict and not invalid and states.get(p) is False
                            and cue_states.get(p)=={'pose_shoulder_support':(False,False),'pose_torso_support':(True,True)}
                            and self._members[p[0]].person_key==self._members[p[1]].person_key
                            and all(k in supplied and region_valid.get(k) is True
                                and region_originals[k].stamp==self._members[k].observation.stamp for k in p)
                            and self._image_pair_custody(self._members[p[0]],at_interval,fingerprint,fresh_delivery=False)
                            and not any(q!=p and states.get(q) is True and set(q)&set(p) for q in expected)):
                        conflict_pairs.add(p);live_pairs.add(p);quarantine_votes.update(p)
                    if region_semantic is None and any(self._members[k].image_region_conflict_custody for k in p):
                        recovery=(p in supported and all(k in supplied and region_valid.get(k) is True
                            and region_originals[k].stamp==self._members[k].observation.stamp
                            and {q.anatomy for q in region_originals[k].patches}=={'pose_shoulder_support','pose_torso_support'} for k in p))
                        if not recovery:
                            supported.discard(p)
                            continue
                    if (not invalid and states.get(p) is not False and p not in supported
                            and self._members[p[0]].person_key==self._members[p[1]].person_key
                            and self._image_pair_custody(self._members[p[0]],at_interval,fingerprint,fresh_delivery=False)
                            and not any(q!=p and states.get(q) is True and set(q)&set(p) for q in expected)):
                        dormant_pairs.add(p);live_pairs.add(p)
                    if (not invalid and states.get(p) is True
                            and self._members[p[0]].person_key==self._members[p[1]].person_key
                            and all(k in supplied and region_originals[k].stamp==self._members[k].observation.stamp
                                and (self._image_live_native(self._members[k],at_interval,fingerprint,pair_only=True)
                                    or self._image_quarantine_custody(self._members[k],at_interval,fingerprint)) for k in p)
                            and any(q!=p and states.get(q) is True and set(q)&set(p) for q in expected)):
                        # Retain custody, NOT ambiguous source ownership/votes.
                        quarantine_pairs.add(p)
                        quarantine_votes.update(k for k in p if sum(k in q for q in expected if states.get(q) is True)>1)
                    if (not invalid and states.get(p) is not False
                            and self._members[p[0]].person_key==self._members[p[1]].person_key
                            and all(self._members[k].image_region_shadow_membership
                                and not self._members[k].spatial_join_required
                                and self._members[k].spatial_lifecycle.native_state=="ACTIVE"
                                and self._spatial_stamp_age(self._members[k].observation.stamp,at_interval) is None
                                and self.current_visual_liveness_error(self._members[k].observation.stamp) is None for k in p)
                            and not any(q!=p and states.get(q) is True and set(q)&set(p) for q in expected)):
                        live_pairs.add(p)
                    if (not invalid and p not in supported and states.get(p) is True
                            and self._members[p[0]].person_key==self._members[p[1]].person_key
                            and all(k in supplied and region_originals[k].stamp==self._members[k].observation.stamp
                                and {q.anatomy for q in region_originals[k].patches}=={"pose_torso_support","pose_shoulder_support"}
                                and self._image_live_native(self._members[k],at_interval,fingerprint,pair_only=True) for k in p)
                            and not any(q!=p and states.get(q) is True and set(q)&set(p) for q in expected)):
                        hypothesis_pairs.add(p)
                    partial_hypothesis=all(self._members[k].image_region_pair_hypothesis for k in p) and any(
                        k in supplied and {q.anatomy for q in region_originals[k].patches}!={"pose_torso_support","pose_shoulder_support"} for k in p)
                    if (not invalid and (any(k not in supplied for k in p) or partial_hypothesis) and states.get(p) is not False
                            and self._members[p[0]].person_key==self._members[p[1]].person_key
                            and all(self._image_live_native(self._members[k],at_interval,fingerprint,
                                pair_only=self._members[k].image_region_pair_hypothesis) for k in p)
                            and not any(q!=p and states.get(q) is True and set(q)&set(p) for q in expected)):
                        live_pairs.add(p)
                live_pairs |= hypothesis_pairs | quarantine_pairs
                supported |= live_pairs
                # Birth proposals cannot join. A returning established member
                # also cannot acquire a NEW join, but its position veto alone
                # must not split an otherwise supported existing membership.
                supported={p for p in supported if not any(k in provisional_extras and
                    (self._members[k].image_region_fragment_provisional is None
                     or self._members[p[0]].person_key!=self._members[p[1]].person_key) for k in p)}
            for pair,diagnostic in self._image_decision_pairs.items():
                diagnostic.update(state=image_states.get(pair),possible=pair in possible,
                    unique=pair in possible and all(sum(k in q for q in possible)==1 for k in pair),supported=pair in supported,
                    continuity_hypothesis=pair in hypothesis_pairs)
                diagnostic["vote_quarantine"]=pair in quarantine_pairs
                diagnostic["appearance_corroborated"]=pair in appearance_joint_pairs
                if pair in conflict_pairs:diagnostic["contradictory_cue_quarantine"]=True
                if region_semantic is not None:
                    diagnostic.update(semantic_state=states.get(pair),semantic_mean_px=semantic_scores.get(pair),
                        semantic_partial_lower_bound_px=semantic_lower_bounds.get(pair))
            selected=tuple(sorted(supported))
            appearance_supported=supported.copy()
        if anatomical is not None:
            if selected:
                return self._reject("anatomical_selected_pairs_forbidden", "association_unresolved")
            error, states, anatomy_originals = self._anatomical_evidence(anatomical, anatomical_policy,
                                                                       at_interval, delivery_context, ingresses)
            if error:
                return self._reject(error, "association_unresolved")
            if not set(states) <= set(candidates):
                return self._reject("anatomical_candidate_binding", "association_unresolved")
            possible = {p for p in viable if states.get(p) is not False}
            supported = {p for p in possible if states.get(p) is True
                         and all(sum(k in q for q in possible) == 1 for k in p)}
            selected = tuple(sorted(supported))
            appearance_supported = {p for p in viable if states.get(p) is not False}
        # Fresh COMPLETE evidence can contradict an old visual estimate. Missing
        # appearance/coverage returned above and is never invented contradiction.
        corrected = list(stale_corrected)
        correction_births = {}
        for person in tuple(sorted(self._persons)):
            members = tuple(self._persons[person])
            if len(members) != 2 or not any(self._members[k].association_basis.startswith(("current_visual", "prior_projective_continuity", "current_registered_anatomical_consensus", "estimated_current_image_regions", "estimated_image_region_continuation", "estimated_live_track_continuity")) for k in members):
                continue
            pair = tuple(sorted(members, key=lambda k: k.source_id))
            contradictory = pair not in appearance_supported or states.get(pair) is False
            shadow_alternative=(regions is not None and pair not in quarantine_pairs and any(self._members[k].image_region_shadow_membership for k in members)
                and any(q!=pair and states.get(q) is True and set(q)&set(pair) for q in expected))
            if not shadow_alternative and (pair not in candidates or pair in supported or ((projective is not None or anatomical is not None) and not contradictory)):
                continue
            if self._owned_key_count() >= self.policy.max_entities:
                for k in members:
                    self._members[k] = replace(self._members[k], spatial_join_required=True,
                                               association_basis="prior_projective_continuity_unresolved" if projective else "current_visual_unresolved")
                self._authority_revision += 1
                return AssociationResult("capacity", "current_visual_split_capacity", corrected_members=tuple(sorted(members)))
            changed = [k for k in members if self._members[k].observation.stamp.frame_id > self._members[k].visual_confirmed_frame]
            detach = changed[0] if len(changed) == 1 else max(members, key=lambda k: k.source_id)
            custody_mode=region_birth and region_live and region_semantic is not None
            selected_alternative=any(q!=pair and set(q)&set(pair) for q in selected)
            if custody_mode and not invalid and states.get(pair) is not False and not selected_alternative:
                keep=self._image_split_local_survivor(pair,region_originals,region_valid,region_integrity,
                    at_interval,fingerprint)
                if keep is not None:detach=next(k for k in pair if k!=keep)
            old = self._members[detach]
            custody=None
            native_custody=((self._image_pair_custody(old,at_interval,fingerprint,fresh_delivery=False)
                or self._image_split_shadow_custody(old,at_interval,fingerprint))
                if custody_mode and not invalid else False)
            births=all(self._members[k].image_region_birth_established for k in members)
            renewed=any(self._members[k].image_region_split_custody is not None
                        and self._members[k].image_region_split_custody.rejoined_key is not None for k in members)
            deadline=None
            if (custody_mode and not invalid and native_custody and births
                    and not renewed and not selected_alternative):
                seeds=tuple(next(s for s in self._members[k].image_region_native_seed if s.member==k) for k in pair)
                stamps=tuple(self._members[k].observation.stamp for k in pair)
                caps=tuple(self._members[k].image_region_owner_availability for k in pair)
                deadline=self._image_split_deadline(caps)
                if deadline is not None and self._visual_evaluation<deadline:
                    custody=(seeds,stamps,caps,deadline)
            try:
                if custody_mode and self._image_decision_pending is not None:
                    failure=('invalid_original' if invalid else 'native_custody' if not native_custody
                        else 'birth_unestablished' if not births else 'rejoined_episode_not_renewed' if renewed
                        else 'selected_alternative' if selected_alternative else 'deadline_invalid' if deadline is None
                        else 'deadline_expired' if self._visual_evaluation>=deadline else 'issued')
                    self._image_decision_pending['split_custody_creation']={
                        'origin_serial':person.serial,'failure':failure,'native_custody':bool(native_custody),
                        'births_established':births,'rejoined_episode':renewed,'selected_alternative':selected_alternative,
                        'owner_deadline_monotonic_s':deadline if _number(deadline) else None,
                        'members':[{'source':k.source_id,'raw':k.raw_id,
                            'seed_frame':next((s.frame_id for s in self._members[k].image_region_native_seed if s.member==k),None),
                            'native_floor':self._members[k].projective_native_floor,
                            'native_active':bool(self._members[k].spatial_lifecycle and self._members[k].spatial_lifecycle.native_state=='ACTIVE'),
                            'owner_present':self._members[k].image_region_owner_availability is not None,
                            'retained':bool(self._members[k].image_region_owner_availability and k in self._members[k].image_region_owner_availability.retained_members),
                            'active':bool(self._members[k].image_region_owner_availability and k in self._members[k].image_region_owner_availability.active_members),
                            'shadow':bool(self._members[k].image_region_owner_availability and k in self._members[k].image_region_owner_availability.shadow_members),
                            'terminated':bool(self._members[k].image_region_owner_availability and k in self._members[k].image_region_owner_availability.terminated_members)} for k in pair]}
            except Exception:self._image_decision_capture_errors+=1
            pending=None
            if (region_birth and states.get(pair) is False
                    and self._image_pair_custody(old,at_interval,fingerprint,fresh_delivery=True)
                    and all(self._members[k].image_region_birth_established for k in members)):
                seeds=tuple(next(s for s in self._members[k].image_region_native_seed if s.member==k) for k in pair)
                caps=tuple(self._members[k].image_region_owner_availability for k in pair)
                pending=_SplitBirthPending(person,seeds,
                    tuple(self._members[k].projective_native_floor for k in pair),fingerprint,
                    min(c.availability_checked_monotonic_s+c.stall_seconds-c.last_progress_age_s for c in caps))
            if (custody_mode and not invalid and states.get(pair) is not False and selected_alternative
                    and old.image_region_policy==fingerprint and old.image_region_birth_established
                    and not old.image_region_birth_quarantined and not old.image_region_conflict_custody
                    and old.image_region_split_birth is None and old.image_region_rejected_birth_peer is None
                    and old.image_region_birth_seed is not None
                    and old.image_region_birth_seed.member==detach
                    and old.image_region_birth_seed.lifecycle_started_frame==old.observation.stamp.lifecycle_started_frame
                    and old.appearance_after_frame<old.image_region_birth_seed.frame_id<=old.observation.stamp.frame_id):
                # Local pre-correction provenance only. Restoration still needs
                # the accepted CURRENT reassignment + ordered SHADOW certificate
                # below; ordinary splits never retain birth by default.
                correction_births[detach]=(old.observation.stamp,old.image_region_birth_seed,old.image_region_birth_peers)
            self._persons[person].remove(detach)
            fresh = self._allocate()
            self._persons[fresh] = {detach}
            self._members[detach] = replace(old, person_key=fresh, association_basis="source_local",
                image_region_split_birth=pending,
                image_region_rejected_birth_peer=self._members[next(k for k in members if k!=detach)].observation.stamp if region_birth else None,
                image_region_rejected_birth_native_floor=self._members[next(k for k in members if k!=detach)].projective_native_floor if region_birth else -1,
                image_region_rejected_birth_origin=person if region_birth else None,
                image_region_birth_established=False if region_birth else old.image_region_birth_established,
                image_region_birth_quarantined=True if region_birth else old.image_region_birth_quarantined,
                image_region_conflict_custody=False,
                image_region_pair_hypothesis=False,image_region_shadow_membership=False,
                image_region_vote_blocked_stamp=None,image_region_vote_pending=False)
            if custody is not None:
                seeds,stamps,caps,deadline=custody
                revision=self._authority_revision+1
                episode='split-custody:'+sha256(repr((person,fresh,seeds,stamps,revision,fingerprint)).encode()).hexdigest()
                view=EstimatedSplitCustodyView(episode,person,revision,seeds,stamps,
                    tuple(sorted((person,fresh))),fingerprint,caps,deadline,
                    native_destinations=tuple((s,self._members[s.member].person_key) for s in stamps))
                for key in pair:
                    m=self._members[key]
                    self._members[key]=replace(m,image_region_split_custody=view,
                        image_region_split_custody_native_floor=m.projective_native_floor)
                self._authority_revision+=1
            for k in self._persons[person]:
                self._members[k] = replace(self._members[k], association_basis="source_local",
                    image_region_conflict_custody=False,
                    image_region_pair_hypothesis=False,image_region_shadow_membership=False,
                    image_region_vote_blocked_stamp=None,image_region_vote_pending=False)
            if anatomical is not None:
                for k in members:
                    m = self._members[k]
                    self._members[k] = replace(m, anatomical_after_frame=max(m.anatomical_after_frame,m.observation.stamp.frame_id))
            self._authority_revision += 1
            corrected.append(detach)
        if not set(selected) <= supported:
            return AssociationResult("association_unresolved", "current_visual_ambiguous_or_denied",
                                     corrected_members=tuple(sorted(corrected)))
        for pair in selected:
            error = self._merge_error(pair)
            houses = [self._members[k].observation.household for k in pair]
            settled = [h for h in houses if h.identity_key is not None]
            if error or any(h.constraint != "eligible" for h in houses) or (regions is None and len(settled) == 2 and settled[0] != settled[1]):
                return AssociationResult("association_unresolved", error or "current_visual_household_conflict",
                                         corrected_members=tuple(sorted(corrected)))
        # Capture the VALID original split episode before a genuinely current,
        # unique SAME-pair merge changes its two-singleton topology. This local
        # bounded work list is not another retained registry or lease renewal.
        reunions=[]
        if regions is not None and region_live and region_birth and region_semantic is not None:
            for pair in selected:
                view=self._members[pair[0]].image_region_split_custody
                if (view is not None and view.rejoined_key is None
                        and self._members[pair[1]].image_region_split_custody is view
                        and {s.member for s in view.prior_seed}==set(pair)
                        and pair not in live_pairs and states.get(pair) is True
                        and all(k in supplied and region_valid.get(k) and region_integrity.get(k)
                            and region_originals[k].stamp==self._members[k].observation.stamp
                            and sum(k in q for q in possible)==1 for k in pair)
                        and self._image_split_custody_error(view,at_interval) is None):
                    reunions.append((view,pair,tuple(self._members[k].observation.stamp for k in pair)))
        retired, keys = [], []
        for pair in selected:
            people = {self._members[k].person_key for k in pair}
            keep = min(people, key=lambda k: k.serial)
            for old_person in people - {keep}:
                for k in tuple(self._persons.pop(old_person)):
                    self._persons[keep].add(k)
                    self._members[k] = replace(self._members[k], person_key=keep)
                retired.append(old_person)
                self._retired_serial_frontier = max(self._retired_serial_frontier, old_person.serial)
                self._authority_revision += 1
            for k in pair:
                member = self._members[k]
                updated = replace(member, spatial_join_required=False, association_basis="current_visual",
                                  visual_confirmed_frame=member.visual_confirmed_frame if pair in live_pairs else member.observation.stamp.frame_id)
                if projective is not None:
                    updated = replace(updated, association_basis="prior_projective_continuity")
                if anatomical is not None:
                    updated = replace(updated, association_basis="current_registered_anatomical_consensus",
                                      anatomical_originals=anatomy_originals[k], anatomical_confirmed=member.observation.stamp)
                if updated != member:
                    self._members[k] = updated
                    self._authority_revision += 1
            keys.append(keep)
        if projective is not None:
            protected = set(self._members) | {k for k, c in self._closed.items() if c.expires_media_s is not None}
            self._projective_frontiers = {k: v for k, v in self._projective_frontiers.items() if k in protected}
            fingerprint = sha256(repr(projective_policy).encode()).hexdigest()
            self._projective_frontiers.update({k: frontiers[k][:4] + (self._members[k].person_key, fingerprint,
                ingresses[k].ingress_monotonic_s, self._projective_frontiers[k][7] if k in self._projective_frontiers else -1,
                frontiers[k][4])
                for pair in selected for k in pair})
        for candidate in (() if regions is not None else candidates.values()):
            for appearance in (candidate.left_appearance, candidate.right_appearance):
                key = appearance.stamp.member
                self._appearance_origins[key] = appearance
                row = ingresses[key]
                original_time = (row.appearance_ingress_monotonic_s if row.appearance_observation is not None
                                 else row.ingress_monotonic_s)
                self._visual_ingresses[key] = replace(row, appearance_observation=appearance.stamp,
                                                      appearance_ingress_monotonic_s=original_time)
        if regions is not None and appearance_joint_pairs:
            # Reuse preserves the existing ORIGINAL provenance frontier, not
            # merely a vector: later proofs cannot roll back or retime this F.
            for value in regions.appearances:
                appearance=value.appearance;key=appearance.stamp.member
                self._appearance_origins[key]=appearance
                self._visual_ingresses[key]=replace(self._visual_ingresses[key],
                    appearance_observation=appearance.stamp,
                    appearance_ingress_monotonic_s=value.ingress_monotonic_s)
        if regions is not None:
            for pair in selected:
                if pair in live_pairs:
                    for k in pair:
                        member=self._members[k]
                        current_unique=(states.get(pair) is True and all(j in supplied
                            and region_originals[j].stamp==self._members[j].observation.stamp for j in pair)
                            and sum(k in q for q in possible)==1)
                        blocked=(member.observation.stamp if k in quarantine_votes else
                            None if current_unique else member.image_region_vote_blocked_stamp)
                        self._members[k]=replace(member,association_basis="estimated_live_track_continuity",
                            image_region_semantic=region_semantic is not None,
                            image_region_conflict_custody=pair in conflict_pairs,
                            image_region_dormant_custody=pair in dormant_pairs,
                            image_region_pair_hypothesis=pair in hypothesis_pairs or pair in quarantine_pairs or member.image_region_pair_hypothesis,
                            image_region_vote_blocked_stamp=blocked,
                            image_region_vote_pending=False if k in quarantine_votes or current_unique else member.image_region_vote_pending)
                        self._authority_revision+=1
                    continue
                continuing=any(region_originals[k].stamp!=self._members[k].observation.stamp for k in pair)
                for k in pair:
                    m=self._members[k]
                    self._members[k]=replace(m,association_basis="estimated_image_region_continuation" if continuing else "estimated_current_image_regions",
                        image_region_original=m.image_region_original if continuing else region_originals[k],
                        image_region_policy=m.image_region_policy if continuing else sha256(repr(region_policy).encode()).hexdigest(),
                        image_region_semantic=region_semantic is not None,
                        image_region_native_seed=m.image_region_native_seed if continuing else actual if region_live else (),
                        image_region_pair_hypothesis=False,
                        image_region_conflict_custody=False,
                        image_region_dormant_custody=False,
                        image_region_shadow_membership=False,
                        image_region_vote_blocked_stamp=m.image_region_vote_blocked_stamp if continuing else None,
                        image_region_vote_pending=m.image_region_vote_pending if continuing else False,
                        image_region_front_confirmed=True)
                    self._authority_revision+=1
        if region_semantic is not None:
            for pair in selected:
                for k in pair:
                    self._members[k]=replace(self._members[k],image_region_policy=fingerprint)
        if region_birth:
            # New local measurement authority is not proof of an additional
            # person. Classify only from this completed CURRENT full cohort.
            joined={k for pair in selected for k in pair}
            # An ordered native SHADOW is absent from actual, but must not
            # miss an already accepted CURRENT positive peer reassignment.
            # Retire the obsolete negative fact. The same-decision certificate
            # may also preserve EXISTING physical birth provenance; SHADOW still
            # gets no membership, measurement, fresh vote or renewed lease.
            for k,m in tuple(self._members.items()):
                rejected=m.image_region_rejected_birth_peer
                if m.image_region_participant or rejected is None:continue
                s=m.observation.stamp;c=m.image_region_owner_availability;event=m.spatial_lifecycle_update
                peer=self._members.get(rejected.member)
                if (m.image_region_policy!=fingerprint or self._image_run_closed
                        or k.source_id in self._image_stopped_sources
                        or m.spatial_lifecycle is None or m.spatial_lifecycle.stamp!=s
                        or m.spatial_lifecycle.native_state!='ACTIVE'
                        or event is None or event.classification!='retained_unmeasured' or event.last_observation!=s
                        or c is None or c.observation!=s or c.graph_id!=self._image_graph_id
                        or c.availability_status!='running' or k not in c.retained_members or k not in c.active_members
                        or k not in c.shadow_members or k in c.terminated_members
                        or event.at_frame_id!=c.source_frame_id or event.at_media_time_s!=c.source_media_time_s
                        or m.projective_native_floor!=event.at_frame_id
                        or s.media_time_s is None or not 0<=c.source_media_time_s-s.media_time_s<=c.owner_grace_ns/1e9
                        or not 0<=self._visual_evaluation-c.availability_checked_monotonic_s+c.last_progress_age_s<c.stall_seconds
                        or self._visual_member_age(s,at_interval) is not None
                        or self.current_visual_liveness_error(s) is not None
                        or self._image_rejected_birth_custody(m,peer) is None):continue
                reassigned=any(rejected.member in pair and k not in pair and pair not in live_pairs
                    and states.get(pair) is True and all(j in supplied and region_valid.get(j)
                        and region_integrity.get(j) and region_originals[j].stamp==self._members[j].observation.stamp
                        and self._members[j].image_region_policy==fingerprint
                        and self._members[j].observation.stamp in self._members[j].image_region_native_seed
                        and self.spatial_member_error(self._members[j].observation.stamp,at_interval) is None
                        and self.current_visual_liveness_error(self._members[j].observation.stamp) is None
                        and self._image_custody_valid(self._members[j]) for j in pair) for pair in selected)
                if reassigned:
                    birth=correction_births.get(k)
                    preserved=bool(birth is not None and birth==(s,m.image_region_birth_seed,m.image_region_birth_peers)
                        and m.image_region_split_birth is None)
                    self._members[k]=replace(m,image_region_split_birth=None,image_region_rejected_birth_peer=None,
                        image_region_rejected_birth_native_floor=-1,image_region_rejected_birth_origin=None,
                        image_region_birth_established=True if preserved else m.image_region_birth_established,
                        image_region_birth_quarantined=False if preserved else m.image_region_birth_quarantined)
                    self._authority_revision+=1
            established_before={k for k,m in self._members.items() if m.image_region_birth_established}
            exclusions={p for p in expected if states.get(p) is False and (
                semantic_scores.get(p) is not None or semantic_lower_bounds.get(p) is not None
                and semantic_lower_bounds[p]>=region_semantic)}
            for stamp in actual:
                k=stamp.member;m=self._members[k]
                rejected=m.image_region_rejected_birth_peer
                peer=self._members.get(rejected.member) if rejected else None
                reassigned=bool(peer and (peer.person_key!=m.image_region_rejected_birth_origin
                    or any(rejected.member in p and k not in p and p not in live_pairs for p in selected)))
                if (k in joined or reassigned) and (m.image_region_split_birth is not None
                        or m.image_region_rejected_birth_peer is not None):
                    m=replace(m,image_region_split_birth=None,image_region_rejected_birth_peer=None,
                        image_region_rejected_birth_native_floor=-1,
                        image_region_rejected_birth_origin=None)
                    self._members[k]=m;self._authority_revision+=1
                if self._image_split_birth_pending(m,fingerprint):
                    continue
                if m.image_region_split_birth is not None:
                    m=replace(m,image_region_split_birth=None)
                    self._members[k]=m;self._authority_revision+=1
                if m.image_region_birth_established:continue
                rejected=m.image_region_rejected_birth_peer
                peer=self._members.get(rejected.member) if rejected else None
                rejected_floor=self._image_rejected_birth_custody(m,peer)
                rejected_current=rejected_floor is not None
                comparisons={p for p in expected if k in p}
                peers=tuple(s for s in actual if s.member.camera_id!=k.camera_id)
                fresh=(m.spatial_lifecycle is not None and m.spatial_lifecycle.native_state=='ACTIVE'
                    and not m.spatial_join_required and self._spatial_stamp_age(stamp,at_interval) is None
                    and self.current_visual_liveness_error(stamp) is None
                    and k.source_id not in self._image_stopped_sources and not self._image_run_closed)
                independent=(k not in provisional_extras and not rejected_current and len(self._persons[m.person_key])==1 and fresh and all(
                    states.get(p) is False and (semantic_scores.get(p) is not None
                        or semantic_lower_bounds.get(p) is not None and semantic_lower_bounds[p]>=region_semantic)
                    and all(region_integrity.get(j) and region_valid.get(j)
                        and j in region_originals and region_originals[j].stamp==self._members[j].observation.stamp for j in p)
                    for p in comparisons))
                if (k not in provisional_extras and not independent and not rejected_current and not invalid and fresh
                        and len(self._persons[m.person_key])==1):
                    independent=self._image_local_occupancy(k,comparisons,states,region_originals,
                        region_valid,region_integrity,quarantine_pairs,at_interval,fingerprint,
                        established_before,exclusions)
                established=k in joined or independent
                update=replace(m,image_region_birth_quarantined=not established,
                    image_region_rejected_birth_peer=rejected if rejected_current and k not in joined else None,
                    image_region_rejected_birth_origin=m.image_region_rejected_birth_origin if rejected_current and k not in joined else None,
                    image_region_rejected_birth_native_floor=rejected_floor if rejected_current and k not in joined else -1,
                    image_region_birth_established=established,image_region_birth_seed=stamp if established else None,
                    image_region_birth_peers=peers)
                if update!=m:self._members[k]=update;self._authority_revision+=1
        region_basis=("estimated_semantic_joint_correspondence" if region_semantic is not None else "estimated_image_region_conflict_custody" if conflict_pairs else "estimated_live_track_continuity" if live_pairs else "estimated_image_region_continuation" if regions is not None and any(self._members[k].association_basis=="estimated_image_region_continuation" for pair in selected for k in pair) else "estimated_current_image_regions")
        for prior,pair,stamps in reunions:
            if all(self._members[k].person_key==prior.origin_key for k in pair):
                view=replace(prior,rejoined_key=prior.origin_key,rejoined_stamps=stamps,
                    rejoin_revision=self._authority_revision+1)
                for k in pair:self._members[k]=replace(self._members[k],image_region_split_custody=view)
                self._authority_revision+=1
        if regions is not None and region_live:
            self._image_finish_split_custody(at_interval,selected,fingerprint)
        result = AssociationResult("accepted" if selected else "observed", (region_basis if regions is not None else "current_registered_anatomical_consensus" if anatomical else "prior_projective_continuity" if projective else "current_visual") if selected else ("image_region_unavailable_or_ambiguous" if regions is not None else "anatomical_support_unavailable_or_ambiguous" if anatomical and viable else "current_visual_local_only"),
                                   keys[0] if len(keys) == 1 else None, retired_keys=tuple(sorted(retired)),
                                   corrected_members=tuple(sorted(corrected)))
        self._visual_last = digest, self._authority_revision, result
        return result

    def _proof_error(self, proof: object, at: AcquisitionTime,
                     observations: dict[MemberKey, SourceObservation] | None = None,
                     appearance_floors: dict[MemberKey, int] | None = None) -> str | None:
        if type(self.policy) is not AssociationPolicy:
            return "common_metric_policy_required"
        if type(proof) is not LinkProof or not _text(proof.proof_id):
            return "malformed_proof"
        if self._clock_error(at):
            return "clock_domain"
        if type(proof.left) is not MeasurementEvidence or type(proof.right) is not MeasurementEvidence:
            return "malformed_proof"
        sides = (proof.left, proof.right)
        stamps = (proof.left.stamp, proof.right.stamp)
        for stamp in stamps:
            error = self._stamp_error(stamp)
            if error:
                return error
        keys = tuple(s.member for s in stamps)
        if (keys[0] == keys[1] or keys[0].source_id == keys[1].source_id
                or keys[0].camera_id == keys[1].camera_id):
            return "same_camera_collision"
        pair = {keys[0].camera_id, keys[1].camera_id}
        if pair != {"kitchen", "family-room"} or not any(
                set(p) == pair for p in self.policy.overlap_pairs):
            return "overlap_denied"
        members = observations if observations is not None else {
            k: m.observation for k, m in self._members.items()}
        descriptors: list[HouseholdDescriptor] = []
        for evidence in sides:
            stamp = evidence.stamp
            observation = members.get(stamp.member)
            if observation is None or observation.stamp != stamp:
                return "noncurrent_observation"
            descriptors.append(observation.household)
            error = self._fresh(stamp, at)
            if error:
                return error
            binding = self._bindings[stamp.member.source_id]
            if (evidence.owner != stamp.member or evidence.origin != "measured"
                    or evidence.units != "m" or not _text(evidence.evidence_id)
                    or not _text(evidence.independence_id)):
                return "independent_measurement_required"
            if (evidence.target_frame != self.policy.target_frame
                    or evidence.world_revision != self.policy.world_revision
                    or evidence.calibration_revision != binding.calibration_revision
                    or evidence.transform_id != binding.transform_id):
                return "measurement_binding"
            if not _number(evidence.x_m) or not _number(evidence.y_m):
                return "nonfinite_measurement"
        if (sides[0].independence_id == sides[1].independence_id
                or sides[0].evidence_id == sides[1].evidence_id):
            return "correlated_measurements"
        if any(d.constraint != "eligible" for d in descriptors):
            return "household_constraint"
        resolved = [d for d in descriptors if d.identity_key is not None]
        if len(resolved) == 2 and (resolved[0].state != resolved[1].state
                or resolved[0].identity_key != resolved[1].identity_key
                or resolved[0].stable_id != resolved[1].stable_id
                or resolved[0].visitor_generation != resolved[1].visitor_generation):
            return "identity_conflict"
        if abs(stamps[0].acquired_at_s - stamps[1].acquired_at_s) > self.policy.max_age_s:
            return "time_separation"
        delta_x = sides[0].x_m - sides[1].x_m
        delta_y = sides[0].y_m - sides[1].y_m
        if not _number(delta_x) or not _number(delta_y):
            return "nonfinite_measurement"
        if hypot(delta_x, delta_y) > self.policy.max_distance_m:
            return "metric_separation"
        appearances = (proof.left_appearance, proof.right_appearance)
        missing = False
        stale = False
        for appearance, stamp in zip(appearances, stamps):
            if appearance is None:
                missing = True
                continue
            if (type(appearance) is not AppearanceEvidence or self._stamp_error(appearance.stamp)
                    or appearance.stamp.member != stamp.member
                    or appearance.stamp.lifecycle_started_frame != stamp.lifecycle_started_frame
                    or not _text(appearance.evidence_id)
                    or not _text(appearance.extraction_id)
                    or appearance.model != self.policy.appearance_model
                    or appearance.profile != self.policy.appearance_profile):
                return "appearance_binding"
            if appearance.extraction_id != appearance_extraction_identity(appearance.stamp, appearance.profile):
                return "appearance_origin_conflict"
            floor = (appearance_floors or {}).get(stamp.member,
                       self._members[stamp.member].appearance_after_frame
                       if stamp.member in self._members else stamp.lifecycle_started_frame - 1)
            if (appearance.stamp.frame_id <= floor or appearance.stamp.frame_id > stamp.frame_id
                    or appearance.stamp.clock_basis != stamp.clock_basis):
                return "appearance_lineage"
            if (not _number(appearance.acquired_at_s) or appearance.acquired_at_s < 0
                    or appearance.clock_domain != at.clock_domain
                    or appearance.acquired_at_s > stamp.acquired_at_s):
                return "appearance_clock"
            age = at.at_s - appearance.acquired_at_s
            if appearance.acquired_at_s != appearance.stamp.acquired_at_s:
                return "appearance_observation"
            if age < 0:
                return "appearance_clock"
            stale = stale or age > self.policy.max_age_s
            origin = self._appearance_origins.get(stamp.member)
            if origin is not None:
                if appearance.stamp.frame_id < origin.stamp.frame_id:
                    return "superseded_appearance"
                if ((appearance.stamp.frame_id == origin.stamp.frame_id
                     or appearance.extraction_id == origin.extraction_id) and appearance != origin):
                    return "appearance_origin_conflict"
        if missing:
            return "appearance_missing"
        if appearances[0].evidence_id == appearances[1].evidence_id:
            return "independent_appearance_required"
        if appearances[0].extraction_id == appearances[1].extraction_id:
            return "independent_appearance_required"
        if stale:
            return "stale_appearance"
        if abs(appearances[0].acquired_at_s - appearances[1].acquired_at_s) > self.policy.max_age_s:
            return "appearance_time_separation"
        if (not _number(proof.similarity) or not -1 <= proof.similarity <= 1
                or proof.similarity < self.policy.min_similarity):
            return "appearance_denied"
        return None

    def _proof_fingerprint(self, proof: LinkProof) -> str:
        # Canonical order makes a repeated swapped pair the same event/proof.
        sides = sorted(((proof.left, proof.left_appearance),
                        (proof.right, proof.right_appearance)), key=lambda pair: pair[0].stamp.member)
        return sha256(repr((proof.proof_id, sides, proof.similarity)).encode()).hexdigest()

    def _evidence_token(self, proof: LinkProof) -> str:
        # Renaming proof/measurement tokens is not a new source observation.
        sides = sorted((proof.left.stamp, proof.right.stamp), key=lambda stamp: stamp.member)
        return "evidence:" + sha256(repr(sides).encode()).hexdigest()

    def _replay(self, token: str, fingerprint: str) -> AssociationResult | None:
        receipt = self._receipts.get(token)
        if receipt is None:
            return None
        if receipt.fingerprint != fingerprint:
            return self._reject("token_conflict", "unresolved")
        if (receipt.result.disposition == "accepted"
                and self._links.get(receipt.result.person_key) != receipt.result.link):
            return AssociationResult("unresolved", "receipt_invalidated", consumed_tokens=(token,))
        return receipt.result

    def _remember(self, tokens: tuple[str, ...], fingerprint: str,
                  result: AssociationResult, sources: tuple[int, ...]) -> None:
        cohorts = tuple((source, self._bindings[source].source_epoch,
                         self._cursors.get(source, (-1, 0, None))[0]) for source in sources)
        for token in tokens:
            self._receipts[token] = _Receipt(fingerprint, result, sources, cohorts)

    def compact_history(self) -> None:
        """Drop only detail whose authority cannot be current again.

        Cohort cursors never rewind. A new unknown lifecycle must be born strictly
        after its source's closed-through frontier; an existing member or exact
        owned media grace is exempt. Birth provenance is immutable owner evidence,
        not a caller's retagged first-seen timestamp. Retired serials are never
        allocated again. Current receipts/correction duplicates/grace stay pinned.
        This operation changes no observation, physical key, link or saved clock.
        """
        for token, receipt in tuple(self._receipts.items()):
            result = receipt.result
            if result.link is not None and self._links.get(result.person_key) == result.link:
                continue
            if (result.disposition == "observed" and result.members
                    and self._views(result.person_key) == result.members):
                continue
            if any(c.expires_media_s is not None and c.person_key == result.person_key
                   for c in self._closed.values()):
                continue
            obsolete = any(self._bindings[source].source_epoch != epoch
                           or self._cursors.get(source, (-1, 0, None))[0] > frame
                           for source, epoch, frame in receipt.cohorts)
            if obsolete:
                del self._receipts[token]
                self._compacted_receipts += 1
        for key, closed in tuple(self._closed.items()):
            if (closed.expires_media_s is None
                    and self._cursors.get(key.source_id, (-1, 0, None))[0] > closed.removal_frame):
                self._lifecycle_frontiers[key.source_id] = max(
                    self._lifecycle_frontiers.get(key.source_id, -1), closed.removal_frame)
                del self._closed[key]
                self._compacted_closed += 1
        protected = set(self._persons) | {c.person_key for c in self._closed.values()
                                         if c.expires_media_s is not None}
        for key in tuple(self._retired):
            if key not in protected:
                self._retired_serial_frontier = max(self._retired_serial_frontier, key.serial)
                del self._retired[key]
                self._compacted_retired += 1

    def _history_available(self, tokens: tuple[str, ...],
                           retired: tuple[PhysicalPersonKey, ...] = ()) -> bool:
        self.compact_history()
        return (len(self._receipts) + len(self._retired)
                + sum(t not in self._receipts for t in tokens)
                + sum(k not in self._retired for k in retired) <= self.policy.max_history)

    def _merge_error(self, keys: tuple[MemberKey, MemberKey]) -> str | None:
        person_keys = {self._members[k].person_key for k in keys}
        all_members = set().union(*(self._persons[k] for k in person_keys))
        if len(all_members) > 2 or len({k.camera_id for k in all_members}) != len(all_members):
            return "same_camera_collision"
        if len(person_keys) > 1 and len(self._retired) >= self.policy.max_history:
            return "retired_key_capacity"
        return None

    def _admit(self, proof: LinkProof, at: AcquisitionTime,
               event_id: str | None, tokens: tuple[str, ...]) -> AssociationResult:
        keys = (proof.left.stamp.member, proof.right.stamp.member)
        person_keys = {self._members[k].person_key for k in keys}
        keep = min(person_keys, key=lambda key: key.serial)
        retired = tuple(sorted(person_keys - {keep}))
        for person_key in retired:
            for key in tuple(self._persons[person_key]):
                member = self._members[key]
                self._members[key] = _Member(member.observation, keep, member.appearance_after_frame)
                self._persons[keep].add(key)
            del self._persons[person_key]
            self._links.pop(person_key, None)
            self._retired[person_key] = tuple(sorted(k.source_id for k in keys))
        stamps = tuple(sorted((proof.left.stamp, proof.right.stamp), key=lambda s: s.member))
        appearances = tuple(sorted((proof.left_appearance, proof.right_appearance), key=lambda a: a.stamp.member))
        for appearance in appearances:
            self._appearance_origins[appearance.stamp.member] = appearance
        expiry = min(s.acquired_at_s + self.policy.max_age_s for s in stamps + tuple(a.stamp for a in appearances))
        self._links[keep] = LinkReceipt(keep, stamps, proof.proof_id, event_id, at, appearances, expiry)
        self._last_shared[keep] = self._links[keep]
        self._gaps.pop(keep, None)
        self._authority_revision += 1
        return self._result("accepted", "fresh_overlap", keep, retired=retired, tokens=tokens)

    def link(self, proof: LinkProof, at: AcquisitionTime) -> AssociationResult:
        if self._clock_error(at) is None:
            self._expire_gaps(at)
        error = self._proof_error(proof, at)
        if error:
            return self._missing_proof(proof, at, error)
        fingerprint = self._proof_fingerprint(proof)
        token = "proof:" + proof.proof_id
        replay = self._replay(token, fingerprint)
        if replay is not None:
            return replay
        keys = (proof.left.stamp.member, proof.right.stamp.member)
        error = self._merge_error(keys)
        if error:
            return self._reject(error, "capacity" if error.endswith("capacity") else "unresolved")
        evidence_token = self._evidence_token(proof)
        if evidence_token in self._receipts:
            return self._reject("consumed_evidence", "unresolved")
        tokens = (token, evidence_token)
        person_keys = {self._members[k].person_key for k in keys}
        oldest = min(person_keys, key=lambda key: key.serial)
        retired = tuple(person_keys - {oldest})
        if not self._history_available(tokens, retired):
            return self._reject("receipt_capacity", "capacity")
        result = self._admit(proof, at, None, tokens)
        self._remember(tokens, fingerprint, result, tuple(k.source_id for k in keys))
        return result

    def _event_error(self, event: object, at: AcquisitionTime) -> str | None:
        if (type(event) is not NativeTransition or not _text(event.event_id)
                or not _text(event.kind) or not _text(event.clock_domain)
                or event.kind not in {"adoption", "correction"}
                or not _number(event.occurred_at_s) or event.occurred_at_s < 0):
            return "malformed_event"
        error = self._clock_error(at) or self._stamp_error(event.previous) or self._stamp_error(event.current)
        if error:
            return error
        if event.peer is not None:
            error = self._stamp_error(event.peer)
            if error:
                return error
        old, new = event.previous.member, event.current.member
        if (old.source_id != new.source_id or old.camera_id != new.camera_id
                or old.source_epoch != new.source_epoch
                or old.lifecycle_generation != new.lifecycle_generation
                or event.previous.lifecycle_started_frame != event.current.lifecycle_started_frame):
            return "event_lineage"
        if (event.current.frame_id < event.previous.frame_id
                or event.current.acquired_at_s < event.previous.acquired_at_s
                or (event.current != event.previous
                    and event.current.observation_id == event.previous.observation_id)):
            return "event_lineage"
        if (event.clock_domain != at.clock_domain or event.previous.clock_domain != at.clock_domain
                or event.current.clock_domain != at.clock_domain
                or event.previous.clock_basis != at.clock_basis
                or event.current.clock_basis != at.clock_basis):
            return "clock_domain"
        if event.occurred_at_s > at.at_s:
            return "future_event"
        if (event.occurred_at_s < max(event.previous.acquired_at_s, event.current.acquired_at_s)
                or at.at_s - event.occurred_at_s > self.policy.max_age_s):
            return "stale_event"
        if event.kind == "adoption" and self._fresh(event.previous, at):
            return "stale_event_lineage"
        return None

    def transition(self, event: NativeTransition, current: SourceObservation,
                   at: AcquisitionTime, proof: LinkProof | None = None) -> AssociationResult:
        """Adoption needs fresh pair proof; correction withdraws before detaching.

        Call before replacing the old member by a generic observation. If native
        lineage is unavailable use exact withdrawal plus independent re-admission,
        never this method with an inferred old/new numeric-ID relationship.
        """
        error = self._event_error(event, at)
        if error:
            return self._reject(error, "unresolved")
        if (type(current) is not SourceObservation or current.stamp != event.current
                or self._household_error(current.household)):
            return self._reject("event_observation", "unresolved")
        # Fingerprints contain only bounded immutable scalar inputs, no embeddings.
        if proof is not None:
            virtual = {k: m.observation for k, m in self._members.items()}
            virtual[current.stamp.member] = current
            floors = {event.current.member: event.current.frame_id - 1} if event.previous.member != event.current.member else None
            error = self._proof_error(proof, at, virtual, floors)
            if error:
                return self._reject(error)
        fingerprint = sha256(repr((event, current, self._proof_fingerprint(proof)
                                   if proof is not None else None)).encode()).hexdigest()
        event_token = "event:" + event.event_id
        # After correction/rekey a completed receipt is diagnostic/idempotent;
        # it never rewrites current state or grants a current link.
        replay = self._replay(event_token, fingerprint)
        if replay is not None:
            return replay
        old_key, new_key = event.previous.member, event.current.member
        old = self._members.get(old_key)
        if old is None or old.observation.stamp != event.previous:
            return self._reject("noncurrent_event_lineage", "unresolved")
        if event.kind == "adoption" and old.person_key in self._gaps:
            return self._reject("continuity_only_no_transition", "unresolved")
        if event.peer is not None:
            peer = self._members.get(event.peer.member)
            if (peer is None or peer.observation.stamp != event.peer
                    or event.peer.member.camera_id == old_key.camera_id):
                return self._reject("noncurrent_peer_lineage", "unresolved")
        if self._order_error(current.stamp):
            return self._reject("reordered_observation", "unresolved")
        if new_key != old_key and new_key in self._closed:
            return self._reject("closed_lifecycle", "unresolved")
        existing_new = self._members.get(new_key)
        if existing_new is not None and new_key != old_key and existing_new.observation != current:
            return self._reject("contradictory_new_member", "unresolved")
        if len(self._closed) >= self.policy.max_history and new_key != old_key:
            self.compact_history()
        if len(self._closed) >= self.policy.max_history and new_key != old_key:
            if event.kind == "correction":
                self._remove(old_key)
                self._block_source(old_key.source_id)
                return self._reject("correction_capacity_source_reset_required", "capacity")
            return self._reject("closed_member_capacity", "capacity")
        tokens = (event_token,) + (("proof:" + proof.proof_id, self._evidence_token(proof))
                                  if proof is not None else ())
        if not self._history_available(tokens):
            if event.kind == "correction":
                self._remove(old_key)
                self._block_source(old_key.source_id)
                return self._reject("correction_capacity_source_reset_required", "capacity")
            return self._reject("receipt_capacity", "capacity")
        if proof is not None and any(token in self._receipts for token in tokens[1:]):
            return self._reject("consumed_proof", "unresolved")
        if event.kind == "adoption":
            if proof is None or event.peer is None:
                return self._reject("fresh_pair_proof_required")
            if {proof.left.stamp, proof.right.stamp} != {event.current, event.peer}:
                return self._reject("event_proof_lineage", "unresolved")
            old_group = self._persons[old.person_key] - {old_key}
            peer_key = self._members[event.peer.member].person_key
            combined = old_group | (self._persons[peer_key] - {old_key}) | {new_key}
            if existing_new is not None:
                combined |= self._persons[existing_new.person_key] - {old_key}
            if len(combined) > 2 or len({k.camera_id for k in combined}) != len(combined):
                return self._reject("same_camera_collision", "unresolved")
            if old.observation.household.identity_key is not None and (
                    current.household != old.observation.household):
                return self._reject("identity_conflict")
            to_retire = {old.person_key, peer_key}
            if existing_new is not None:
                to_retire.add(existing_new.person_key)
            keep = min(to_retire, key=lambda key: key.serial)
            retired = tuple(sorted(to_retire - {keep}))
            if not self._history_available(tokens, retired):
                return self._reject("retired_key_capacity", "capacity")
            if new_key != old_key:
                self._remove(old_key)
                self._lifecycle_frontiers[old_key.source_id] = max(
                    self._lifecycle_frontiers.get(old_key.source_id, -1), event.current.frame_id)
                self._closed[old_key] = _Closed(event.previous, keep, old.observation.household,
                                              event.current.frame_id, "adoption", None)
            if existing_new is not None and new_key != old_key:
                self._remove(new_key)
            self._persons.setdefault(keep, set())
            for key in tuple(to_retire - {keep}):
                for member_key in tuple(self._persons.pop(key, ())):
                    member = self._members[member_key]
                    self._members[member_key] = _Member(member.observation, keep, member.appearance_after_frame)
                    self._persons[keep].add(member_key)
                self._links.pop(key, None)
                self._retired[key] = (old_key.source_id, event.peer.member.source_id)
            self._members[new_key] = _Member(current, keep, event.current.frame_id - 1
                                             if new_key != old_key else old.appearance_after_frame)
            self._persons[keep].add(new_key)
            result = self._admit(proof, at, event.event_id, tokens)
            result = AssociationResult(result.disposition, result.reason, result.person_key,
                                       result.members, result.link, retired, tokens)
        else:
            if proof is not None:
                return self._reject("correction_requires_separate_readmission", "unresolved")
            group = self._persons[old.person_key]
            if len(group) > 1 and (event.peer is None or event.peer.member not in group):
                return self._reject("correction_peer_lineage", "unresolved")
            if existing_new is not None and new_key != old_key:
                return self._reject("correction_new_member_collision", "unresolved")
            # Withdrawal wins, including exhaustion. The unaffected peer key lives.
            self._remove(old_key)
            if new_key != old_key:
                self._lifecycle_frontiers[old_key.source_id] = max(
                    self._lifecycle_frontiers.get(old_key.source_id, -1), event.current.frame_id)
                self._closed[old_key] = _Closed(event.previous, old.person_key,
                                              old.observation.household,
                                              event.current.frame_id, "correction", None)
            if self._owned_key_count() >= self.policy.max_entities:
                result = self._reject("detached_entity_capacity", "capacity")
            else:
                detached = self._allocate()
                self._members[new_key] = _Member(current, detached, event.current.frame_id)
                self._persons[detached] = {new_key}
                result = self._result("observed", "correction_detached", detached, tokens=tokens)
        self._cursors[new_key.source_id] = (current.stamp.frame_id, current.stamp.acquired_at_s,
                                           current.stamp.media_time_s)
        self._authority_revision += 1
        self._remember(tokens, fingerprint, result, tuple(sorted({old_key.source_id}
                        | ({event.peer.member.source_id} if event.peer is not None else set()))))
        return result

    def _retention_deadline(self, removal: MemberRemoval) -> float | None:
        receipt = removal.retention
        stamp = removal.last_observation
        if (removal.kind != "absence" or type(receipt) is not LifecycleRetention
                or not _text(receipt.receipt_id) or receipt.last_observation != stamp
                or receipt.owner != stamp.member or receipt.lifecycle != "inactive"
                or not _number(receipt.grace_s)
                or not 0 < receipt.grace_s <= self.policy.max_return_grace_s
                or stamp.media_time_s is None or removal.media_time_s is None
                or not stamp.media_time_s <= removal.media_time_s <= stamp.media_time_s + receipt.grace_s):
            return None
        return stamp.media_time_s + receipt.grace_s

    def withdraw(self, removal: MemberRemoval) -> AssociationResult:
        self._image_support_targets=()
        if (type(removal) is not MemberRemoval or not _text(removal.receipt_id)
                or not _text(removal.kind)
                or removal.kind not in {"absence", "termination"}
                or not _integer(removal.frame_id) or not _number(removal.acquired_at_s)
                or (removal.media_time_s is not None
                    and (not _number(removal.media_time_s) or removal.media_time_s < 0))):
            return self._reject("malformed_removal")
        error = self._stamp_error(removal.last_observation)
        if error:
            return self._reject(error)
        stamp = removal.last_observation
        if (removal.clock_domain != stamp.clock_domain or removal.frame_id < stamp.frame_id
                or removal.acquired_at_s < stamp.acquired_at_s):
            return self._reject("removal_lineage", "unresolved")
        # Invalid/unbounded retention input cannot be retained or hashed. It
        # invalidates grace while the exact absence still removes presence.
        deadline = self._retention_deadline(removal)
        fingerprint = sha256(repr((removal.receipt_id, stamp, removal.kind,
                                   removal.frame_id, removal.acquired_at_s,
                                   removal.clock_domain, removal.media_time_s, deadline)).encode()).hexdigest()
        token = "removal:" + removal.receipt_id
        replay = self._replay(token, fingerprint)
        if replay is not None:
            return replay
        member = self._members.get(stamp.member)
        closed = self._closed.get(stamp.member)
        if member is None:
            # Termination supersedes an earlier valid absence/grace receipt.
            if (removal.kind == "termination" and closed is not None
                    and closed.last_observation == stamp):
                self._closed[stamp.member] = _Closed(stamp, closed.person_key, closed.household,
                                                   removal.frame_id, "termination", None)
                self._authority_revision += 1
                result = self._result("removed", "termination", closed.person_key, tokens=(token,))
                if self._history_available((token,)):
                    self._remember((token,), fingerprint, result, (stamp.member.source_id,))
                return result
            return self._reject("noncurrent_removal", "unresolved")
        if member.observation.stamp != stamp:
            return self._reject("noncurrent_removal", "unresolved")
        cursor = self._cursors.get(stamp.member.source_id)
        if cursor is not None and (removal.frame_id < cursor[0]
                                   or removal.acquired_at_s < cursor[1]):
            return self._reject("reordered_removal", "unresolved")
        self._remove(stamp.member)
        self._lifecycle_frontiers[stamp.member.source_id] = max(
            self._lifecycle_frontiers.get(stamp.member.source_id, -1), removal.frame_id)
        self.compact_history()
        if len(self._closed) >= self.policy.max_history or not self._history_available((token,)):
            # Never retain live presence merely because the tombstone budget is full.
            self._block_source(stamp.member.source_id)
            return self._reject("removal_capacity_source_reset_required", "capacity")
        self._closed[stamp.member] = _Closed(stamp, member.person_key, member.observation.household,
                                           removal.frame_id, removal.kind,
                                           deadline)
        self._cursors[stamp.member.source_id] = (removal.frame_id, removal.acquired_at_s,
                                               removal.media_time_s)
        result = self._result("removed", removal.kind, member.person_key, tokens=(token,))
        self._remember((token,), fingerprint, result, (stamp.member.source_id,))
        return result

    def _block_source(self, source_id: int) -> None:
        for key in tuple(self._members):
            if key.source_id == source_id:
                self._remove(key)
        for key, closed in tuple(self._closed.items()):
            if key.source_id == source_id and closed.expires_media_s is not None:
                self._closed[key] = _Closed(closed.last_observation, closed.person_key,
                                           closed.household, closed.removal_frame, closed.kind, None)
        self._blocked.add(source_id)
        self._authority_revision += 1

    def clear_source(self, absence: SourceAbsence) -> AssociationResult:
        """Invalidate every member/proof for an exact empty source; no grace inferred."""
        self._image_support_targets=()
        if (type(absence) is not SourceAbsence or not _text(absence.receipt_id)
                or not _integer(absence.source_id) or not _integer(absence.source_epoch)
                or not _integer(absence.frame_id) or not _number(absence.acquired_at_s)
                or absence.acquired_at_s < 0):
            return self._reject("malformed_source_absence")
        binding = self._bindings.get(absence.source_id)
        if (binding is None or binding.camera_id != absence.camera_id
                or binding.source_epoch != absence.source_epoch
                or binding.clock_domain != absence.clock_domain):
            return self._reject("source_binding")
        cursor = self._cursors.get(absence.source_id)
        if cursor is not None and (absence.frame_id < cursor[0]
                                   or absence.acquired_at_s < cursor[1]):
            return self._reject("reordered_source_absence", "unresolved")
        fingerprint = sha256(repr(absence).encode()).hexdigest()
        token = "source_absence:" + absence.receipt_id
        replay = self._replay(token, fingerprint)
        if replay is not None:
            return replay
        for key in tuple(self._members):
            if key.source_id != absence.source_id:
                continue
            member = self._remove(key)
            if len(self._closed) < self.policy.max_history:
                self._closed[key] = _Closed(member.observation.stamp, member.person_key,
                                           member.observation.household, absence.frame_id, "absence", None)
            else:
                self._block_source(absence.source_id)
        # Empty source explicitly cancels any earlier private return grace.
        for key, closed in tuple(self._closed.items()):
            if key.source_id == absence.source_id and closed.expires_media_s is not None:
                self._closed[key] = _Closed(closed.last_observation, closed.person_key,
                                           closed.household, absence.frame_id, "absence", None)
        self._cursors[absence.source_id] = (absence.frame_id, absence.acquired_at_s, None)
        self._lifecycle_frontiers[absence.source_id] = max(
            self._lifecycle_frontiers.get(absence.source_id, -1), absence.frame_id)
        self._authority_revision += 1
        result = AssociationResult("removed", "source_absent", consumed_tokens=(token,))
        if self._history_available((token,)):
            self._remember((token,), fingerprint, result, (absence.source_id,))
        else:
            self._block_source(absence.source_id)
            return self._reject("removal_capacity_source_reset_required", "capacity")
        return result

    def reset_source(self, binding: SourceBinding) -> AssociationResult:
        self._image_support_targets=()
        if type(binding) is not SourceBinding or binding.source_id not in self._bindings:
            return self._reject("source_binding")
        old = self._bindings[binding.source_id]
        if (binding.camera_id != old.camera_id or not _integer(binding.source_epoch)
                or binding.source_epoch <= old.source_epoch
                or not all(_text(v) for v in (binding.calibration_revision,
                                             binding.transform_id, binding.clock_domain))
                or not (_clock_basis(binding.clock_basis)
                        or type(self.policy) is SpatialAssociationPolicy and binding.clock_basis == "source-media")):
            return self._reject("source_epoch_must_advance", "unresolved")
        for key in tuple(self._members):
            if key.source_id == binding.source_id:
                self._remove(key)
        self._closed = {k: c for k, c in self._closed.items() if k.source_id != binding.source_id}
        self._receipts = {t: r for t, r in self._receipts.items() if binding.source_id not in r.sources}
        self._retired = {k: sources for k, sources in self._retired.items() if binding.source_id not in sources}
        self._cursors.pop(binding.source_id, None)
        self._blocked.discard(binding.source_id)
        self._lifecycle_frontiers.pop(binding.source_id, None)
        self._bindings[binding.source_id] = binding
        self._image_stopped_sources.discard(binding.source_id)
        self._appearance_origins = {k: a for k, a in self._appearance_origins.items() if k.source_id != binding.source_id}
        self._authority_revision += 1
        return AssociationResult("reset", "source_epoch_advanced")

    def reset_run(self, run_id: str) -> AssociationResult:
        self._image_support_targets=()
        if not _text(run_id) or run_id == self.run_id:
            return self._reject("new_run_id_required", "unresolved")
        self.run_id = run_id
        self._members.clear()
        self._persons.clear()
        self._links.clear()
        self._closed.clear()
        self._receipts.clear()
        self._retired.clear()
        self._cursors.clear()
        self._blocked.clear()
        self._lifecycle_frontiers.clear()
        self._retired_serial_frontier = 0
        self._compacted_receipts = self._compacted_closed = self._compacted_retired = 0
        self._observation_refusal_json = None
        self._serial = 0
        self._appearance_origins.clear()
        self._gaps.clear()
        self._last_shared.clear()
        self._visual_actor = None
        self._visual_sequence = -1
        self._visual_evaluation = -1.0
        self._visual_ingresses.clear()
        self._visual_last = None
        self._projective_frontiers.clear()
        self._image_decision_records.clear()
        self._image_decision_json=()
        self._image_decision_last_completed=None
        self._image_decision_evicted=0
        self._image_decision_capture_errors=0
        self._image_decision_pending=None
        self._image_decision_pairs={}
        self._image_graph_id=None;self._image_stopped_sources.clear();self._image_run_closed=False
        self._authority_revision += 1
        return AssociationResult("reset", "run_restarted")

    def snapshot(self, at: AcquisitionTime) -> tuple[PersonView, ...]:
        if self._clock_error(at):
            raise ValueError("snapshot requires explicit valid acquisition time")
        self._expire_gaps(at)
        views: list[PersonView] = []
        for key in sorted(self._persons):
            members = self._views(key)
            receipt = self._links.get(key)
            if receipt is not None and (at.at_s > receipt.expires_at_s
                                       or any(self._fresh(s, at) for s in receipt.members)):
                receipt = None
            fresh = all(self._fresh(m.observation, at) is None for m in members)
            state = ("local" if fresh and len(members) == 1 else
                     "accepted" if receipt is not None else "unresolved")
            gap = self._gaps.get(key)
            if gap is not None:
                state = "continuity-only"
            views.append(PersonView(key, members, state, receipt, gap))
        return tuple(views)

    def validate_receipt(self, receipt: LinkReceipt, at: AcquisitionTime) -> AssociationResult:
        """Read-only owner check for prepare/commit consumers of a shared receipt.

        Historical results are diagnostic. Only an exact still-current receipt
        from this ledger with fresh original acquisition times grants a shared
        contribution. This method never renews the saved clocks or proof history.
        """
        if self._clock_error(at):
            return self._reject("clock_domain")
        if (type(receipt) is not LinkReceipt or type(receipt.person_key) is not PhysicalPersonKey
                or not _text(receipt.person_key.run_id) or not _integer(receipt.person_key.serial, 1)):
            return self._reject("malformed_receipt")
        if self._links.get(receipt.person_key) != receipt:
            return self._reject("receipt_invalidated", "unresolved")
        for stamp in receipt.members:
            member = self._members.get(stamp.member)
            if member is None or member.observation.stamp != stamp:
                return self._reject("receipt_invalidated", "unresolved")
            error = self._fresh(stamp, at)
            if error:
                return self._reject(error)
        if at.at_s > receipt.expires_at_s:
            return self._reject("stale_appearance")
        for appearance in receipt.appearances:
            error = self._fresh(appearance.stamp, at)
            member = self._members.get(appearance.stamp.member)
            if (error or member is None or appearance.stamp.frame_id <= member.appearance_after_frame
                    or self._appearance_origins.get(appearance.stamp.member) != appearance):
                return self._reject(error or "appearance_lineage")
        return self._result("accepted", "current_owner_receipt", receipt.person_key)

    def _expire_gaps(self, at: AcquisitionTime) -> None:
        for key, gap in tuple(self._gaps.items()):
            if at.clock_domain == gap.first_missing_cohort.clock_domain and at.clock_basis == gap.first_missing_cohort.clock_basis and at.at_s >= gap.deadline_s:
                self._gaps.pop(key)
                self._last_shared.pop(key, None)
                self._links.pop(key, None)
                self._authority_revision += 1

    def _missing_proof(self, proof: object, at: AcquisitionTime, reason: str) -> AssociationResult:
        if reason not in {"appearance_missing", "stale_appearance"}:
            if type(proof) is LinkProof and type(proof.left) is MeasurementEvidence and type(proof.right) is MeasurementEvidence:
                stamps = (proof.left.stamp, proof.right.stamp)
                if all(self._stamp_error(s) is None and s.member in self._members
                       and self._members[s.member].observation.stamp == s for s in stamps):
                    person = self._members[stamps[0].member].person_key
                    if (person in self._gaps and self._members[stamps[1].member].person_key == person):
                        self._gaps.pop(person)
                        self._last_shared.pop(person, None)
                        self._authority_revision += 1
            return self._reject(reason)
        if type(proof) is not LinkProof or self._clock_error(at):
            return self._reject(reason)
        stamps = (proof.left.stamp, proof.right.stamp)
        keys = tuple(s.member for s in stamps)
        if any(k not in self._members for k in keys):
            return self._reject(reason)
        person = self._members[keys[0]].person_key
        if self._members[keys[1]].person_key != person or person not in self._last_shared:
            return self._reject(reason)
        if self._persons[person] != set(keys):
            return self._reject(reason)
        gap = self._gaps.get(person)
        if gap is None:
            anchor = AcquisitionTime(max(s.acquired_at_s for s in stamps), at.clock_domain, at.clock_basis)
            gap = ContinuityGap(person, tuple(sorted(keys)), anchor,
                                anchor.at_s + self.policy.max_age_s, self._last_shared[person])
            if at.at_s >= gap.deadline_s:
                self._last_shared.pop(person, None)
                return self._reject("continuity_expired", "unresolved")
            self._gaps[person] = gap
            self._links.pop(person, None)
            self._authority_revision += 1
        return self._result("unresolved", "continuity_only", person)

    def expire_retention(self, source_id: int, media_time_s: float | None) -> None:
        """Release private grace only; never refresh observations or proof clocks."""
        if not _integer(source_id) or source_id not in self._bindings or (media_time_s is not None and
                (not _number(media_time_s) or media_time_s < 0)):
            raise ValueError("valid source/media time required")
        cursor = self._cursors.get(source_id)
        if cursor is not None and cursor[2] is not None and media_time_s is not None and media_time_s < cursor[2]:
            raise ValueError("media time moved backwards")
        for key, closed in tuple(self._closed.items()):
            if key.source_id == source_id and closed.expires_media_s is not None and (
                    media_time_s is None or media_time_s > closed.expires_media_s):
                self._closed[key] = _Closed(closed.last_observation, closed.person_key,
                                           closed.household, closed.removal_frame, closed.kind, None)
                self._authority_revision += 1

    @property
    def authority_revision(self) -> int:
        return self._authority_revision

    def source_bindings(self) -> tuple[SourceBinding, ...]:
        return tuple(self._bindings[source] for source in sorted(self._bindings))

    def household_claim_owners(self) -> tuple[tuple[PhysicalPersonKey, SourceObservation], ...]:
        """Read-only current membership plus genuinely owned original media grace.

        No host clock, inferred retention, registry, expiration or household
        resolution is introduced. Closed records retain the exact original
        stamp and descriptor; the immutable wrapper reconstructs those fields.
        At most max_entities physical owners and max_history closed records
        plus two current members per owner can participate. Publication owners
        must fail closed if their smaller conflict-output budget is exceeded.
        """
        owners = [(member.person_key, member.observation) for member in self._members.values()]
        owners.extend((closed.person_key, SourceObservation(closed.last_observation, closed.household))
                      for closed in self._closed.values() if closed.expires_media_s is not None)
        return tuple(sorted(owners, key=lambda row: (row[0], row[1].stamp.member)))

    def stats(self) -> LedgerStats:
        return LedgerStats(len(self._persons), len(self._members),
                           sum(c.expires_media_s is not None for c in self._closed.values()),
                           len(self._closed), len(self._receipts), len(self._retired), len(self._blocked),
                           len(self._lifecycle_frontiers), self._compacted_receipts,
                           self._compacted_closed, self._compacted_retired,
                           self._retired_serial_frontier, self._serial,
                           len(self._appearance_origins), len(self._gaps), len(self._last_shared))
