"""Process-owned association DTOs and exact single-writer consumer transactions.

No native extraction, household resolution, metric admission, threads or I/O live
here. The strict constructor needs an explicit admitted common-metric receipt;
denied admission constructs no ledger. Explicit for_spatial supplies source-local
membership only, without a common-metric receipt or a second person ledger.
Owner tokens and object-identity registries are process-local, not public wire
credentials. Synthetic admissions must be labelled synthetic by their test owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import math
from typing import Callable

from reid.mv3dt_person_association import (
    AcquisitionTime, AppearanceEvidence, AssociationPolicy, AssociationResult,
    ContinuityGap, LinkProof, LinkReceipt, MeasurementEvidence, MemberKey, MemberRemoval,
    MemberView, NativeTransition, ObservationStamp, PersonAssociationLedger,
    PhysicalPersonKey, SourceAbsence, SourceBinding, SourceObservation,
    HouseholdDescriptor, SpatialAssociationPolicy, SpatialClockContext, SpatialClockInterval,
    SpatialCurrentPersonProof, SpatialLifecycleReceipt, SpatialLifecycleUpdate,
    CurrentVisualProof, CurrentVisualDeliveryContext,
    ProjectiveContinuityPolicy, ProjectiveContinuityProof, ProjectiveContinuityInspection,
    AnatomicalConsensusPolicy, AnatomicalConsensusProof,
    EstimatedImageRegionProof, ImageRegionOwnerAvailability, ImageRegionAppearance, EstimatedSplitCustodyView,
)


def policy_identity(policy: AssociationPolicy) -> str:
    canonical = replace(policy, sources=tuple(sorted(policy.sources, key=lambda b: b.source_id)))
    return sha256(repr(canonical).encode()).hexdigest()


def _token(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= 256 and value == value.strip()


@dataclass(frozen=True, slots=True)
class ProducerAdmission:
    status: str
    source_receipt_id: str
    reason: str
    run_id: str | None = None
    policy: AssociationPolicy | None = None
    policy_identity: str | None = None
    metric_receipt_id: str | None = None
    acquisition_receipt_id: str | None = None
    independent_source_receipt_id: str | None = None
    scope: str = "unavailable"

    def __post_init__(self) -> None:
        if (not all(_token(v) for v in (self.status, self.source_receipt_id, self.reason, self.scope))
                or self.status not in {"admitted", "unavailable"}
                or self.scope not in {"synthetic", "canonical", "unavailable"}
                or (self.policy is not None and type(self.policy) is not AssociationPolicy)
                or any(v is not None and not _token(v) for v in (
                    self.run_id, self.policy_identity, self.metric_receipt_id,
                    self.acquisition_receipt_id, self.independent_source_receipt_id))):
            raise ValueError("immutable bounded producer admission fields required")


@dataclass(frozen=True, slots=True)
class MeasurementReference:
    stamp: ObservationStamp
    owner: MemberKey
    evidence_id: str
    target_frame: str
    world_revision: str
    calibration_revision: str
    transform_id: str
    independence_id: str
    origin: str
    units: str

    @classmethod
    def from_evidence(cls, evidence: MeasurementEvidence) -> MeasurementReference:
        return cls(evidence.stamp, evidence.owner, evidence.evidence_id,
                   evidence.target_frame, evidence.world_revision,
                   evidence.calibration_revision, evidence.transform_id,
                   evidence.independence_id, evidence.origin, evidence.units)


@dataclass(frozen=True, slots=True)
class ProofIdentity:
    receipt: LinkReceipt
    measurements: tuple[MeasurementReference, ...]
    appearances: tuple[AppearanceEvidence, ...]
    native_transition: NativeTransition | None
    fresh_extractions: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class AssociationDTO:
    owner_instance: object = field(repr=False, compare=False)
    run_id: str
    physical_key: PhysicalPersonKey
    association_revision: int
    state: str
    members: tuple[MemberView, ...]
    proof: ProofIdentity | None
    policy: AssociationPolicy
    policy_identity: str
    producer_admission: ProducerAdmission
    native_transition: NativeTransition | None
    withdrawn_members: tuple[MemberKey, ...]
    retired_keys: tuple[PhysicalPersonKey, ...]
    reason: str
    continuity: ContinuityGap | None


@dataclass(frozen=True, slots=True)
class HouseholdClaimConflict:
    """Original contested claims: duplicate across people or divergent within one.

    The image-only within-person case has >=2 distinct original MemberKeys, not
    historical versions of one member. Owners are never fictitious entities.
    """

    run_id: str
    association_revision: int
    subject_ids: tuple[str, ...]
    stable_ids: tuple[int, ...]
    owners: tuple[tuple[PhysicalPersonKey, SourceObservation], ...]
    reason: str = "settled_household_claim_conflict"


@dataclass(frozen=True, slots=True)
class ServiceResult:
    disposition: str
    reason: str
    dto: AssociationDTO | None = None
    withdrawn_members: tuple[MemberKey, ...] = ()
    retired_keys: tuple[PhysicalPersonKey, ...] = ()
    household_conflict: HouseholdClaimConflict | None = None


@dataclass(frozen=True, slots=True)
class MeasurementCohort:
    run_id: str
    association_revision: int
    policy_identity: str
    target_frame: str
    world_revision: str
    measurements: tuple[MeasurementReference, ...]


@dataclass(frozen=True, slots=True)
class AssociationReservation:
    owner_instance: object = field(repr=False, compare=False)
    serial: int
    dto: AssociationDTO
    cohort: MeasurementCohort
    prepared_at: AcquisitionTime


@dataclass(frozen=True, slots=True)
class ReservationResult:
    disposition: str
    reason: str
    reservation: AssociationReservation | None = None
    household_conflict: HouseholdClaimConflict | None = None


@dataclass(frozen=True, slots=True)
class CommitResult:
    disposition: str
    reason: str
    consumer_value: object = None
    household_conflict: HouseholdClaimConflict | None = None


@dataclass(frozen=True, slots=True)
class ServiceStats:
    association_revision: int
    current_dtos: int
    current_proofs: int
    pending_reservations: int
    reservation_serial: int
    clock_frontiers: int
    committed_cohorts: int
    spatial_member_frontiers: int = 0


@dataclass(frozen=True, slots=True)
class SpatialMemberEligibility:
    observation: ObservationStamp
    lifecycle_receipt: SpatialLifecycleReceipt | None
    eligible: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ImageRegionSupportView:
    """Immutable seed provenance, not a fresh pose or unique-person claim.

    native_seed_cohort describes the ORIGINAL seed, not the current competitor
    roster of an estimated current-pair continuity hypothesis.
    """
    current_stamp: ObservationStamp
    original_stamp: ObservationStamp
    original_ingress_monotonic_s: float
    native_seed_cohort: tuple[ObservationStamp, ...] = ()


@dataclass(frozen=True, slots=True)
class SpatialMembershipDTO:
    owner_instance: object = field(repr=False, compare=False)
    run_id: str
    physical_key: PhysicalPersonKey
    association_revision: int
    members: tuple[MemberView, ...]
    source_bindings: tuple[SourceBinding, ...]
    eligibility: tuple[SpatialMemberEligibility, ...]
    household_claim: HouseholdDescriptor | None
    withdrawn_members: tuple[MemberKey, ...]
    retired_keys: tuple[PhysicalPersonKey, ...]
    household_conflict: HouseholdClaimConflict | None = None
    association_basis: str = "source_local"
    image_region_support: tuple[ImageRegionSupportView, ...] = ()
    estimated_split_custody: EstimatedSplitCustodyView | None = None


@dataclass(frozen=True, slots=True)
class SpatialMeasurementReference:
    stamp: ObservationStamp
    owner: MemberKey
    evidence_id: str
    target_frame: str
    world_revision: str
    calibration_revision: str
    transform_id: str
    origin: str
    units: str
    measured_stage_id: str
    acquisition_interval: SpatialClockInterval


@dataclass(frozen=True, slots=True)
class SpatialMemberReservation:
    owner_instance: object = field(repr=False, compare=False)
    serial: int
    dto: SpatialMembershipDTO
    reference: SpatialMeasurementReference
    prepared_context: SpatialClockContext
    purpose: str = "strict"
    household_conflict: HouseholdClaimConflict | None = None


@dataclass(frozen=True, slots=True)
class SpatialMemberResult:
    disposition: str
    reason: str
    dto: SpatialMembershipDTO | None = None
    reservation: SpatialMemberReservation | None = None
    consumer_value: object = None
    household_conflict: HouseholdClaimConflict | None = None
    withdrawn_members: tuple[MemberKey, ...] = ()
    retired_keys: tuple[PhysicalPersonKey, ...] = ()
    association_basis: str = "source_local"
    purpose: str = "strict"


class IdentityAssociationService:
    def __init__(self, run_id: str, admission: ProducerAdmission,
                 *, max_reservations: int = 64) -> None:
        if not _token(run_id) or type(admission) is not ProducerAdmission:
            raise ValueError("explicit run and producer admission required")
        if type(max_reservations) is not int or not 1 <= max_reservations <= 64:
            raise ValueError("reservation capacity must be in 1..64")
        self.run_id = run_id
        self._owner = object()
        self._admission = admission
        self._ledger: PersonAssociationLedger | None = None
        self._revision = 0
        self._ledger_revision = 0
        self._issued: dict[PhysicalPersonKey, AssociationDTO] = {}
        self._proofs: dict[PhysicalPersonKey, ProofIdentity] = {}
        self._reservations: dict[int, AssociationReservation] = {}
        self._reservation_serial = 0
        self._max_reservations = max_reservations
        self._times: dict[tuple[str, str], float] = {}
        self._committing = False
        self._committed: dict[PhysicalPersonKey, tuple[ObservationStamp, ...]] = {}
        self._spatial_policy: SpatialAssociationPolicy | None = None
        self._spatial_issued: dict[PhysicalPersonKey, SpatialMembershipDTO] = {}
        self._spatial_times: dict[int, SpatialClockInterval] = {}
        self._spatial_consumed: dict[MemberKey, int] = {}
        self._spatial_withdrawn: set[MemberKey] = set()
        self._spatial_retired: set[PhysicalPersonKey] = set()
        self._spatial_barrier_overflow = False
        self._current_visual_enabled = False
        self._projective_policy: ProjectiveContinuityPolicy | None = None
        self._anatomical_policy: AnatomicalConsensusPolicy | None = None
        self._image_region_policy: ProjectiveContinuityPolicy | None = None
        self._image_live_track_continuity = False
        self._image_contradictory_cue_quarantine = False
        self._semantic_joint_max_mean_px = None
        self._unresolved_birth_quarantine = False
        if self._admission_error(admission) is None:
            self._ledger = PersonAssociationLedger(run_id, admission.policy)

    @classmethod
    def for_spatial(cls, run_id: str, spatial_policy: SpatialAssociationPolicy) -> IdentityAssociationService:
        if type(spatial_policy) is not SpatialAssociationPolicy:
            raise ValueError("explicit nonmetric spatial policy required")
        owner = cls(run_id, ProducerAdmission("unavailable", "explicit-spatial-route", "no-common-metric-admission"))
        owner._spatial_policy = spatial_policy
        owner._ledger = PersonAssociationLedger(run_id, spatial_policy)
        return owner

    @classmethod
    def for_current_visual(cls, run_id: str, spatial_policy: SpatialAssociationPolicy) -> IdentityAssociationService:
        """Opt-in estimated physical grouping; no household overlap/metric/CI permit."""
        owner = cls.for_spatial(run_id, spatial_policy)
        owner._current_visual_enabled = True
        return owner

    @classmethod
    def for_projective_continuity(cls, run_id: str, spatial_policy: SpatialAssociationPolicy,
                                  projective_policy: ProjectiveContinuityPolicy) -> IdentityAssociationService:
        if (type(spatial_policy) is not SpatialAssociationPolicy or type(projective_policy) is not ProjectiveContinuityPolicy
                or any(c.source not in spatial_policy.sources for c in projective_policy.cameras)):
            raise ValueError("exact original source projective policy required")
        owner = cls.for_spatial(run_id, spatial_policy)
        owner._current_visual_enabled = True
        owner._projective_policy = projective_policy
        return owner

    @property
    def ledger(self) -> PersonAssociationLedger | None:
        """Diagnostics only; mutation belongs to this single service owner."""
        return self._ledger

    def image_region_decision_diagnostics(self) -> dict:
        """Detached owner-captured diagnostics; never issues/synchronizes a grant."""
        if self._ledger is None:
            return {"contract":"noesis.identity.image_region_decisions.v1","limit":8,
                    "evicted":0,"capture_errors":0,"transitions":[]}
        return self._ledger.image_region_decision_diagnostics()

    _default_image_region_diagnostics = image_region_decision_diagnostics

    def image_region_decision_snapshot(self):
        """Same-completed-join immutable observer; preserve injected accessors."""
        accessor = self.image_region_decision_diagnostics
        if (self._ledger is None or getattr(accessor, '__func__', None)
                is not IdentityAssociationService._default_image_region_diagnostics):
            return accessor()  # Existing optional validation still owns overrides.
        return self._ledger.image_region_decision_snapshot()

    @classmethod
    def for_estimated_image_regions(cls, run_id: str, spatial_policy: SpatialAssociationPolicy,
                                    projective_policy: ProjectiveContinuityPolicy, *,
                                    live_track_continuity: bool = False,
                                    contradictory_cue_quarantine: bool = False,
                                    semantic_joint_max_mean_px: float | None = None,
                                    unresolved_birth_quarantine: bool = False) -> IdentityAssociationService:
        """Task-only estimates; live continuity never seeds an ambiguous NEW pair."""
        if type(live_track_continuity) is not bool:
            raise ValueError("image_region_live_track_boolean_required")
        if type(contradictory_cue_quarantine) is not bool or contradictory_cue_quarantine and not live_track_continuity:
            raise ValueError("image_region_contradictory_cue_opt_in_required")
        if semantic_joint_max_mean_px is not None and (type(semantic_joint_max_mean_px) not in (int,float)
                or not math.isfinite(semantic_joint_max_mean_px) or semantic_joint_max_mean_px<=0 or not live_track_continuity):
            raise ValueError("semantic_joint_threshold_opt_in_required")
        if type(unresolved_birth_quarantine) is not bool or unresolved_birth_quarantine and (
                not live_track_continuity or semantic_joint_max_mean_px is None):
            raise ValueError("image_region_birth_opt_in_required")
        if (type(spatial_policy) is not SpatialAssociationPolicy or type(projective_policy) is not ProjectiveContinuityPolicy
                or any(c.source not in spatial_policy.sources for c in projective_policy.cameras)):
            raise ValueError("exact estimated image-region source policy required")
        owner=cls.for_current_visual(run_id,spatial_policy)
        owner._image_region_policy=projective_policy
        owner._image_live_track_continuity=live_track_continuity is True
        owner._image_contradictory_cue_quarantine=contradictory_cue_quarantine
        owner._semantic_joint_max_mean_px=semantic_joint_max_mean_px
        owner._unresolved_birth_quarantine=unresolved_birth_quarantine
        return owner

    @classmethod
    def for_anatomical_consensus(cls, run_id: str, spatial_policy: SpatialAssociationPolicy,
                                 anatomical_policy: AnatomicalConsensusPolicy) -> IdentityAssociationService:
        if (type(anatomical_policy) is not AnatomicalConsensusPolicy
                or any(c.source not in spatial_policy.sources for c in anatomical_policy.projective_policy.cameras)):
            raise ValueError("exact anatomical source policy required")
        owner = cls.for_current_visual(run_id, spatial_policy)
        owner._anatomical_policy = anatomical_policy
        return owner

    def _admission_error(self, admission: ProducerAdmission) -> str | None:
        if admission.status != "admitted":
            return "producer_unavailable"
        if (admission.run_id != self.run_id or type(admission.policy) is not AssociationPolicy
                or not _token(admission.scope) or admission.scope not in {"synthetic", "canonical"}
                or not all(_token(v) for v in (admission.source_receipt_id,
                                              admission.metric_receipt_id,
                                              admission.acquisition_receipt_id,
                                              admission.independent_source_receipt_id))
                or admission.policy_identity != policy_identity(admission.policy)):
            return "producer_admission_invalid"
        return None

    def _available(self) -> str | None:
        if self._committing:
            return "reentrant_transaction"
        if self._spatial_policy is not None:
            return None if self._ledger is not None and self._ledger.run_id == self.run_id else "spatial_run_changed"
        error = self._admission_error(self._admission)
        if error or self._ledger is None:
            return error or "producer_unavailable"
        if self._ledger.run_id != self.run_id:
            return "producer_run_changed"
        if tuple(sorted(self._admission.policy.sources, key=lambda b: b.source_id)) != self._ledger.source_bindings():
            return "producer_source_binding_changed"
        return None

    def _sync(self) -> None:
        if self._ledger is not None and self._ledger_revision != self._ledger.authority_revision:
            self._ledger_revision = self._ledger.authority_revision
            self._revision += 1
            self._issued.clear()
            self._proofs.clear()
            self._reservations.clear()
            if self._spatial_issued:
                self._spatial_withdrawn.clear()
                self._spatial_retired.clear()
            self._spatial_issued.clear()
            protected = set(self._ledger._members) | {k for k, c in self._ledger._closed.items()
                                                     if c.expires_media_s is not None}
            self._spatial_consumed = {k: frame for k, frame in self._spatial_consumed.items() if k in protected}
            active = set(self._ledger._persons) | {c.person_key for c in self._ledger._closed.values()
                                                 if c.expires_media_s is not None}
            self._committed = {k: stamps for k, stamps in self._committed.items() if k in active}

    def _clock(self, at: AcquisitionTime) -> str | None:
        if self._ledger is None or self._ledger._clock_error(at):
            return "clock_domain"
        identity = (at.clock_domain, at.clock_basis)
        if identity not in {(b.clock_domain, b.clock_basis) for b in self._ledger.source_bindings()}:
            return "clock_domain"
        if at.at_s < self._times.get(identity, -1):
            return "reordered_clock"
        self._times[identity] = at.at_s
        return None

    def _members(self) -> dict[MemberKey, PhysicalPersonKey]:
        return {k: m.person_key for k, m in self._ledger._members.items()} if self._ledger is not None else {}

    def _withdrawn(self, before: dict[MemberKey, PhysicalPersonKey]) -> tuple[MemberKey, ...]:
        current = self._members()
        return tuple(sorted(k for k, person in before.items() if current.get(k) != person))

    def _household_check(self, key: PhysicalPersonKey) -> tuple[str | None, HouseholdClaimConflict | None]:
        """Aggregate once over finite live/grace membership, never pairwise history.

        This is grant eligibility, not assignment or physical resolution. A
        handoff supplies no accepted public kind in this DTO contract, so only
        its non-null SID is checked; no resident/visitor truth is inferred.
        """
        rows = self._ledger.household_claim_owners()
        if self._image_region_policy is not None:
            latest={}
            for person,original in rows:
                previous=latest.get(original.stamp.member)
                if previous is None or original.stamp.frame_id>previous[1].stamp.frame_id:
                    latest[original.stamp.member]=(person,original)
            for member in self._ledger._members.values():
                previous=latest.get(member.observation.stamp.member)
                if previous is not None and member.observation.stamp.frame_id>=previous[1].stamp.frame_id:
                    latest[member.observation.stamp.member]=(member.person_key,member.observation)
            rows=tuple(sorted(latest.values(),key=lambda row:(row[0],row[1].stamp.member)))
        subjects: dict[str, set[PhysicalPersonKey]] = {}
        stable: dict[int, set[PhysicalPersonKey]] = {}
        claims = []
        for person, original in rows:
            house = original.household
            subject = None
            if house.state == "resident":
                subject = f"resident:{house.identity_key}"
            elif house.state == "visitor" and house.stable_id is not None:
                subject = f"visitor:{self.run_id}:{house.stable_id}:g{house.visitor_generation}"
            if subject is not None:
                subjects.setdefault(subject, set()).add(person)
            if house.stable_id is not None:
                stable.setdefault(house.stable_id, set()).add(person)
            claims.append((person, original, subject, house.stable_id))
        divergent=set()
        if self._image_region_policy is not None:
            grouped={}
            for person,original,subject,sid in claims:
                h=original.household
                if h.constraint=="eligible" and h.identity_key is not None and h.state in {"resident","visitor","handoff"}:
                    grouped.setdefault(person,[]).append(original)
            for person,originals in grouped.items():
                if (len({o.stamp.member for o in originals})>=2
                        and len({(o.household.state,o.household.identity_key,o.household.stable_id,o.household.visitor_generation)
                                 for o in originals})>1):
                    divergent.add(person)
        clashing_subjects = tuple(sorted({s for s, owners in subjects.items() if len(owners) > 1}
            | {subject for person,original,subject,sid in claims if person in divergent and subject is not None}))
        clashing_stable = tuple(sorted({s for s, owners in stable.items() if len(owners) > 1}
            | {sid for person,original,subject,sid in claims if person in divergent and sid is not None}))
        subject_set, stable_set = set(clashing_subjects), set(clashing_stable)
        owners = tuple((person, original) for person, original, subject, sid in claims
                       if person in divergent or subject in subject_set or sid in stable_set)
        if not any(person == key for person, _ in owners):
            return None, None
        # Live/grace tables are separately bounded. Never truncate original
        # clashing evidence to pretend this smaller public DTO is complete.
        limit = self._ledger.policy.max_history
        if (len(owners) > limit or len(clashing_subjects) > limit or len(clashing_stable) > limit
                or len({person for person, _ in owners}) > self._ledger.policy.max_entities):
            return "household_claim_capacity", None
        conflict = HouseholdClaimConflict(self.run_id, self._revision, clashing_subjects,
                                          clashing_stable, owners)
        return conflict.reason, conflict

    def _dto_household_check(self, dto: object) -> tuple[str | None, HouseholdClaimConflict | None]:
        # An old owner-issued DTO can describe a now-conflicting key after a
        # global revision clears grants. This check cannot grant a forged DTO.
        if (type(dto) is AssociationDTO and dto.owner_instance is self._owner
                and dto.run_id == self.run_id and type(dto.physical_key) is PhysicalPersonKey):
            return self._household_check(dto.physical_key)
        return None, None

    def _return(self, result: AssociationResult, at: AcquisitionTime,
                before: dict[MemberKey, PhysicalPersonKey], proof: LinkProof | None = None,
                event: NativeTransition | None = None) -> ServiceResult:
        self._sync()
        withdrawn = self._withdrawn(before)
        if result.person_key is None or result.disposition in {"rejected", "capacity", "removed", "reset"}:
            return ServiceResult(result.disposition, result.reason, withdrawn_members=withdrawn,
                                 retired_keys=result.retired_keys)
        if result.disposition == "accepted" and proof is not None:
            identities = tuple(sorted((MeasurementReference.from_evidence(proof.left),
                                       MeasurementReference.from_evidence(proof.right)),
                                      key=lambda m: m.stamp.member))
            appearances = tuple(sorted((proof.left_appearance, proof.right_appearance),
                                       key=lambda a: a.stamp.member))
            existing = self._proofs.get(result.person_key)
            if existing is None or existing.receipt != result.link:
                current_stamps = {m.stamp.member: m.stamp for m in identities}
                fresh = tuple(a.stamp == current_stamps[a.stamp.member] for a in appearances)
                self._proofs[result.person_key] = ProofIdentity(result.link, identities, appearances, event, fresh)
        views = self._ledger.snapshot(at)
        view = next((v for v in views if v.person_key == result.person_key), None)
        if view is None:
            return ServiceResult("unresolved", "current_person_unavailable", withdrawn_members=withdrawn)
        error, conflict = self._household_check(view.person_key)
        if error:
            return ServiceResult("conflict" if conflict is not None else "capacity", error,
                                 withdrawn_members=withdrawn, retired_keys=result.retired_keys,
                                 household_conflict=conflict)
        prior = self._issued.get(view.person_key)
        if prior is not None:
            return ServiceResult(result.disposition, result.reason, prior, withdrawn, result.retired_keys)
        effective = replace(self._ledger.policy, sources=self._ledger.source_bindings())
        identity = self._proofs.get(view.person_key)
        if view.state == "accepted" and (identity is None or identity.receipt != view.link):
            return ServiceResult("unresolved", "current_proof_identity_unavailable", withdrawn_members=withdrawn)
        dto = AssociationDTO(self._owner, self.run_id, view.person_key, self._revision,
                             view.state, view.members, identity, effective, policy_identity(effective),
                             self._admission, event, withdrawn, result.retired_keys, result.reason, view.continuity)
        self._issued[view.person_key] = dto
        return ServiceResult(result.disposition, result.reason, dto, withdrawn, result.retired_keys)

    def observe(self, observation: SourceObservation) -> ServiceResult:
        if self._spatial_policy is not None:
            return ServiceResult("unavailable", "use_observe_spatial_with_native_lifecycle")
        error = self._available()
        if error:
            return ServiceResult("unavailable", error)
        before = self._members()
        result = self._ledger.observe(observation)
        if result.person_key is None:
            self._sync()
            return ServiceResult(result.disposition, result.reason)
        at = AcquisitionTime(observation.stamp.acquired_at_s, observation.stamp.clock_domain,
                             observation.stamp.clock_basis)
        return self._return(result, at, before)

    def link(self, proof: LinkProof, at: AcquisitionTime) -> ServiceResult:
        if self._spatial_policy is not None:
            return ServiceResult("unavailable", "use_join_spatial_with_qualified_intervals")
        error = self._available() or self._clock(at)
        if error:
            return ServiceResult("unavailable", error)
        before = self._members()
        result = self._ledger.link(proof, at)
        return self._return(result, at, before, proof=proof if result.disposition == "accepted" else None)

    def transition(self, event: NativeTransition, current: SourceObservation,
                   at: AcquisitionTime, proof: LinkProof | None = None) -> ServiceResult:
        if self._spatial_policy is not None:
            error = self._available()
            if error or type(event) is not NativeTransition or event.kind != "correction" or proof is not None:
                return ServiceResult("unavailable", error or "spatial_native_adoption_requires_current_person_proof")
            before = self._members()
            result = self._ledger.transition(event, current, at)
            member = self._ledger._members.get(current.stamp.member) if type(current) is SourceObservation else None
            if result.person_key is not None and member is not None:
                self._ledger._members[current.stamp.member] = replace(member, spatial_join_required=True)
            self._sync()
            self._spatial_note(before, result.retired_keys)
            return ServiceResult(result.disposition, result.reason, withdrawn_members=self._withdrawn(before),
                                 retired_keys=result.retired_keys)
        error = self._available() or self._clock(at)
        if error:
            return ServiceResult("unavailable", error)
        before = self._members()
        return self._return(self._ledger.transition(event, current, at, proof), at, before, proof, event)

    def withdraw(self, removal: MemberRemoval) -> ServiceResult:
        error = self._available()
        if error:
            return ServiceResult("unavailable", error)
        before = self._members()
        result = self._ledger.withdraw(removal)
        self._sync()
        self._spatial_note(before)
        return ServiceResult(result.disposition, result.reason,
                             withdrawn_members=self._withdrawn(before))

    def clear_source(self, absence: SourceAbsence) -> ServiceResult:
        error = self._available()
        if error:
            return ServiceResult("unavailable", error)
        before = self._members()
        result = self._ledger.clear_source(absence)
        self._sync()
        self._spatial_note(before)
        return ServiceResult(result.disposition, result.reason,
                             withdrawn_members=self._withdrawn(before))

    def reset_source(self, binding: SourceBinding) -> ServiceResult:
        error = self._available()
        if error:
            return ServiceResult("unavailable", error)
        before = self._members()
        result = self._ledger.reset_source(binding)
        if self._spatial_policy is not None and result.disposition == "reset":
            self._spatial_policy = replace(self._spatial_policy, sources=self._ledger.source_bindings())
            self._ledger.policy = self._spatial_policy
            self._spatial_times.pop(binding.source_id, None)
        self._sync()
        self._spatial_note(before)
        return ServiceResult(result.disposition, result.reason,
                             withdrawn_members=self._withdrawn(before))

    def reset_run(self, run_id: str) -> ServiceResult:
        error = self._available()
        if error:
            return ServiceResult("unavailable", error)
        before = self._members()
        result = self._ledger.reset_run(run_id)
        if result.disposition == "reset":
            self.run_id = run_id
            self._times.clear()
            self._committed.clear()
            self._spatial_times.clear()
            self._spatial_consumed.clear()
            self._spatial_barrier_overflow = False
        self._sync()
        self._spatial_note(before)
        return ServiceResult(result.disposition, result.reason,
                             withdrawn_members=self._withdrawn(before))

    def bind_admission(self, admission: ProducerAdmission) -> ServiceResult:
        if self._spatial_policy is not None:
            return ServiceResult("unavailable", "spatial_route_has_no_common_metric_admission")
        if self._committing or type(admission) is not ProducerAdmission:
            return ServiceResult("unavailable", "invalid_admission_update")
        error = self._admission_error(admission)
        if error:
            self._admission = admission
            self._revision += 1
            self._issued.clear()
            self._proofs.clear()
            self._reservations.clear()
            return ServiceResult("unavailable", error)
        if self._ledger is not None:
            expected = replace(self._ledger.policy, sources=self._ledger.source_bindings())
            if policy_identity(expected) != admission.policy_identity:
                return ServiceResult("unavailable", "policy_binding_mismatch")
        else:
            self._ledger = PersonAssociationLedger(self.run_id, admission.policy)
        self._admission = admission
        self._revision += 1
        self._issued.clear()
        self._proofs.clear()
        self._reservations.clear()
        return ServiceResult("bound", "explicit_producer_admission")

    def _current(self, dto: AssociationDTO, at: AcquisitionTime) -> str | None:
        self._sync()
        error = self._available()
        if error:
            return error
        if (type(dto) is not AssociationDTO or dto.owner_instance is not self._owner
                or type(dto.physical_key) is not PhysicalPersonKey
                or not _token(dto.physical_key.run_id) or type(dto.physical_key.serial) is not int
                or self._issued.get(dto.physical_key) is not dto):
            return "foreign_or_unissued_dto"
        effective = replace(self._ledger.policy, sources=self._ledger.source_bindings())
        if (dto.run_id != self.run_id or dto.association_revision != self._revision
                or dto.policy_identity != policy_identity(effective)
                or dto.producer_admission is not self._admission):
            return "association_revision_changed"
        if dto.state not in {"local", "accepted"}:
            return "current_cross_source_proof_required"
        current = next((v for v in self._ledger.snapshot(at) if v.person_key == dto.physical_key), None)
        if current is None or current.members != dto.members or current.state != dto.state:
            return "current_members_changed_or_expired"
        if dto.state == "accepted":
            if dto.proof is None or self._ledger.validate_receipt(dto.proof.receipt, at).disposition != "accepted":
                return "current_receipt_invalidated"
        return None

    def _cohort(self, dto: AssociationDTO, cohort: MeasurementCohort) -> str | None:
        if (type(cohort) is not MeasurementCohort or type(cohort.measurements) is not tuple
                or not 1 <= len(cohort.measurements) <= 2
                or any(type(m) is not MeasurementReference for m in cohort.measurements)):
            return "malformed_measurement_cohort"
        if any(self._ledger._stamp_error(m.stamp) for m in cohort.measurements):
            return "malformed_measurement_cohort"
        if (cohort.run_id != dto.run_id or cohort.association_revision != dto.association_revision
                or cohort.policy_identity != dto.policy_identity
                or cohort.target_frame != dto.policy.target_frame
                or cohort.world_revision != dto.policy.world_revision):
            return "measurement_cohort_binding"
        ordered = tuple(sorted(cohort.measurements, key=lambda m: m.stamp.member))
        if tuple(m.stamp for m in ordered) != tuple(m.observation for m in dto.members):
            return "measurement_members_mismatch"
        for measured in ordered:
            binding = next((b for b in dto.policy.sources if b.source_id == measured.stamp.member.source_id), None)
            if (binding is None or measured.owner != measured.stamp.member
                    or measured.origin != "measured" or measured.units != "m"
                    or measured.target_frame != dto.policy.target_frame
                    or measured.world_revision != dto.policy.world_revision
                    or measured.calibration_revision != binding.calibration_revision
                    or measured.transform_id != binding.transform_id
                    or not _token(measured.evidence_id) or not _token(measured.independence_id)):
                return "measurement_source_binding"
        if dto.proof is not None and ordered != dto.proof.measurements:
            return "measurement_proof_identity"
        return None

    def prepare(self, dto: AssociationDTO, cohort: MeasurementCohort,
                at: AcquisitionTime) -> ReservationResult:
        if self._spatial_policy is not None:
            return ReservationResult("denied", "use_prepare_spatial_member")
        error = self._available() or self._clock(at)
        if error:
            return ReservationResult("denied", error)
        self._sync()
        error, conflict = self._dto_household_check(dto)
        if error:
            return ReservationResult("denied", error, household_conflict=conflict)
        error = self._current(dto, at)
        if error:
            return ReservationResult("denied", error)
        if self._committed.get(dto.physical_key) == tuple(m.observation for m in dto.members):
            return ReservationResult("denied", "cohort_already_committed")
        error = self._cohort(dto, cohort)
        if error:
            return ReservationResult("denied", error)
        if len(self._reservations) >= min(self._max_reservations, self._ledger.policy.max_entities):
            return ReservationResult("capacity", "reservation_capacity")
        self._reservation_serial += 1
        reservation = AssociationReservation(self._owner, self._reservation_serial, dto, cohort, at)
        self._reservations[reservation.serial] = reservation
        return ReservationResult("reserved", "exact_owner_cohort", reservation)

    def discard(self, reservation: AssociationReservation) -> ReservationResult:
        if (self._committing or type(reservation) is not AssociationReservation
                or type(reservation.serial) is not int
                or reservation.owner_instance is not self._owner
                or self._reservations.get(reservation.serial) is not reservation):
            return ReservationResult("denied", "foreign_or_consumed_reservation")
        del self._reservations[reservation.serial]
        return ReservationResult("discarded", "owner_reservation_discarded")

    def commit(self, reservation: AssociationReservation, at: AcquisitionTime,
               consumer: Callable[[AssociationDTO, MeasurementCohort], object]) -> CommitResult:
        if self._spatial_policy is not None:
            return CommitResult("denied", "use_commit_spatial_member")
        self._sync()
        if (self._committing or type(reservation) is not AssociationReservation
                or type(reservation.serial) is not int
                or reservation.owner_instance is not self._owner):
            return CommitResult("denied", "foreign_or_consumed_reservation")
        error = self._available()
        if error:
            if self._reservations.get(reservation.serial) is reservation:
                del self._reservations[reservation.serial]
            return CommitResult("denied", error)
        error, conflict = self._dto_household_check(reservation.dto)
        if error:
            if self._reservations.get(reservation.serial) is reservation:
                del self._reservations[reservation.serial]
            return CommitResult("denied", error, household_conflict=conflict)
        if self._reservations.get(reservation.serial) is not reservation:
            return CommitResult("denied", "foreign_or_consumed_reservation")
        # Consume intent before invoking the application owner, including failures.
        del self._reservations[reservation.serial]
        error = self._available() or self._clock(at) or self._current(reservation.dto, at)
        if error:
            return CommitResult("denied", error)
        if at.at_s < reservation.prepared_at.at_s:
            return CommitResult("denied", "reordered_clock")
        error = self._cohort(reservation.dto, reservation.cohort)
        if error or not callable(consumer):
            return CommitResult("denied", error or "invalid_consumer")
        stamps = tuple(m.observation for m in reservation.dto.members)
        if self._committed.get(reservation.dto.physical_key) == stamps:
            return CommitResult("denied", "cohort_already_committed")
        self._committed[reservation.dto.physical_key] = stamps
        self._committing = True
        try:
            value = consumer(reservation.dto, reservation.cohort)
        finally:
            self._committing = False
        return CommitResult("committed", "exact_owner_cohort", value)

    def compact_history(self) -> None:
        if self._ledger is not None and not self._committing:
            self._ledger.compact_history()
            self._sync()

    def expire_retention(self, source_id: int, media_time_s: float | None) -> ServiceResult:
        error = self._available()
        if error:
            return ServiceResult("unavailable", error)
        try:
            self._ledger.expire_retention(source_id, media_time_s)
        except ValueError:
            return ServiceResult("rejected", "invalid_media_clock")
        self._sync()
        return ServiceResult("expired", "original_media_grace_checked")

    def stats(self) -> ServiceStats:
        self._sync()
        return ServiceStats(self._revision, len(self._issued) + len(self._spatial_issued), len(self._proofs),
                            len(self._reservations), self._reservation_serial, len(self._times) + len(self._spatial_times),
                            len(self._committed), len(self._spatial_consumed))

    def _spatial_available(self) -> str | None:
        return self._available() or ("spatial_route_required" if self._spatial_policy is None else
                                    "spatial_barrier_capacity" if self._spatial_barrier_overflow else None)

    def _spatial_note(self, before: dict[MemberKey, PhysicalPersonKey],
                      retired: tuple[PhysicalPersonKey, ...] = (),
                      corrected: tuple[MemberKey, ...] = ()) -> None:
        if self._spatial_policy is not None:
            withdrawals = self._spatial_withdrawn | set(self._withdrawn(before)) | set(corrected)
            retirements = self._spatial_retired | set(retired)
            if len(withdrawals) > 128 or len(retirements) > 128:
                # Per-operation results still carry the complete real barrier.
                # Refuse publication; do not retain an unbounded hidden journal.
                self._spatial_barrier_overflow = True
            else:
                self._spatial_withdrawn, self._spatial_retired = withdrawals, retirements

    def _spatial_clock(self, context: SpatialClockContext) -> str | None:
        error = self._ledger.spatial_clock_error(context)
        if error:
            return error
        for interval in context.intervals:
            prior = self._spatial_times.get(interval.binding.source_id)
            cursor = self._ledger._cursors.get(interval.binding.source_id)
            if (prior is not None and (prior.binding != interval.binding
                    or interval.evaluation_min_s < prior.evaluation_min_s
                    or interval.evaluation_max_s < prior.evaluation_max_s)
                    or cursor is not None and interval.evaluation_max_s < cursor[1]):
                return "spatial_clock_rewind"
        for interval in context.intervals:
            self._spatial_times[interval.binding.source_id] = interval
        return None

    def observe_spatial(self, original: SourceObservation,
                        lifecycle_receipt: SpatialLifecycleReceipt) -> SpatialMemberResult:
        error = self._spatial_available()
        if error:
            return SpatialMemberResult("unavailable", error)
        before = self._members()
        result = self._ledger.observe_spatial(original, lifecycle_receipt,
            image_source_local_reactivation=self._image_region_policy is not None,
            retain_image_membership=self._image_region_policy is not None and self._image_live_track_continuity,
            quarantine_unresolved_birth=self._unresolved_birth_quarantine)
        self._sync()
        self._spatial_note(before, result.retired_keys)
        # Final capabilities are issued only after ALL ordered cohort mutations.
        return SpatialMemberResult(result.disposition, result.reason,
                                   withdrawn_members=self._withdrawn(before), retired_keys=result.retired_keys)

    def join_spatial(self, current_person_proof: SpatialCurrentPersonProof,
                     *, at_interval: SpatialClockContext) -> SpatialMemberResult:
        if self._current_visual_enabled:
            return SpatialMemberResult("association_unresolved", "strict_spatial_factory_required")
        error = self._spatial_available() or self._spatial_clock(at_interval)
        if error:
            return SpatialMemberResult("association_unresolved", error)
        before = self._members()
        result = self._ledger.join_spatial(current_person_proof, at_interval=at_interval)
        self._sync()
        self._spatial_note(before, result.retired_keys)
        return SpatialMemberResult(result.disposition, result.reason,
                                   withdrawn_members=self._withdrawn(before), retired_keys=result.retired_keys)

    def image_region_active_cohort(self) -> tuple[ObservationStamp, ...]:
        if self._image_region_policy is None:
            raise ValueError("estimated_image_regions_opt_in_required")
        return self._ledger.image_region_active_cohort()

    def stop_image_region_source(self, binding: SourceBinding, *, graph_id: str) -> SpatialMemberResult:
        if self._image_region_policy is None:return SpatialMemberResult("association_unresolved","estimated_image_regions_opt_in_required")
        result=self._ledger.stop_image_region_source(binding,graph_id=graph_id);self._sync()
        return SpatialMemberResult(result.disposition,result.reason)

    def close_image_region_run(self) -> SpatialMemberResult:
        if self._image_region_policy is None:return SpatialMemberResult("association_unresolved","estimated_image_regions_opt_in_required")
        result=self._ledger.close_image_region_run();self._sync()
        return SpatialMemberResult(result.disposition,result.reason)

    def update_spatial_lifecycle(self, update: SpatialLifecycleUpdate) -> SpatialMemberResult:
        if self._image_region_policy is None:
            return SpatialMemberResult("association_unresolved","estimated_image_regions_opt_in_required")
        error=self._spatial_available()
        if error: return SpatialMemberResult("unavailable",error)
        before=self._members()
        result=self._ledger.update_spatial_lifecycle(update,retain_image_membership=self._image_live_track_continuity)
        self._sync()
        self._spatial_note(before,result.retired_keys,result.corrected_members)
        member=self._ledger._members.get(update.last_observation.member) if type(update) is SpatialLifecycleUpdate else None
        return SpatialMemberResult(result.disposition,result.reason,
            withdrawn_members=tuple(sorted(set(self._withdrawn(before))|set(result.corrected_members))),
            retired_keys=result.retired_keys,association_basis=member.association_basis if member else "source_local")

    def join_current_visual(self, proof: CurrentVisualProof, *, at_interval: SpatialClockContext,
                            delivery_context: CurrentVisualDeliveryContext) -> SpatialMemberResult:
        if self._image_region_policy is not None:
            return SpatialMemberResult("association_unresolved", "estimated_image_regions_route_required")
        if self._anatomical_policy is not None:
            return SpatialMemberResult("association_unresolved", "anatomical_consensus_route_required")
        if self._projective_policy is not None:
            return SpatialMemberResult("association_unresolved", "projective_continuity_route_required")
        if not self._current_visual_enabled:
            return SpatialMemberResult("association_unresolved", "current_visual_opt_in_required")
        error = self._spatial_available() or self._spatial_clock(at_interval)
        if error:
            return SpatialMemberResult("association_unresolved", error, association_basis="current_visual")
        before = self._members()
        result = self._ledger.join_current_visual(proof, at_interval=at_interval, delivery_context=delivery_context)
        self._sync()
        self._spatial_note(before, result.retired_keys, result.corrected_members)
        withdrawn = tuple(sorted(set(self._withdrawn(before)) | set(result.corrected_members)))
        return SpatialMemberResult(result.disposition, result.reason, withdrawn_members=withdrawn,
                                   retired_keys=result.retired_keys, association_basis="current_visual")

    def inspect_projective_continuity(self, proof: ProjectiveContinuityProof, *, at_interval: SpatialClockContext,
                                      delivery_context: CurrentVisualDeliveryContext) -> ProjectiveContinuityInspection:
        """Diagnostic only; does not synchronize, observe, reserve, grant or renew a seed."""
        if self._projective_policy is None:
            return ProjectiveContinuityInspection("projective_continuity_opt_in_required")
        error = self._spatial_available()
        if error:
            return ProjectiveContinuityInspection(error)
        return self._ledger.inspect_projective_continuity(proof, projective_policy=self._projective_policy,
            at_interval=at_interval, delivery_context=delivery_context)

    def join_projective_continuity(self, proof: ProjectiveContinuityProof, *, at_interval: SpatialClockContext,
                                   delivery_context: CurrentVisualDeliveryContext) -> SpatialMemberResult:
        if self._projective_policy is None:
            return SpatialMemberResult("association_unresolved", "projective_continuity_opt_in_required")
        error = self._spatial_available() or self._spatial_clock(at_interval)
        if error:
            return SpatialMemberResult("association_unresolved", error, association_basis="prior_projective_continuity")
        before = self._members()
        result = self._ledger.join_projective_continuity(proof, projective_policy=self._projective_policy,
                                                       at_interval=at_interval, delivery_context=delivery_context)
        self._sync()
        self._spatial_note(before, result.retired_keys, result.corrected_members)
        return SpatialMemberResult(result.disposition, result.reason,
            withdrawn_members=tuple(sorted(set(self._withdrawn(before)) | set(result.corrected_members))),
            retired_keys=result.retired_keys, association_basis="prior_projective_continuity")

    def join_anatomical_consensus(self, proof: AnatomicalConsensusProof, *, at_interval: SpatialClockContext,
                                  delivery_context: CurrentVisualDeliveryContext) -> SpatialMemberResult:
        if self._anatomical_policy is None:
            return SpatialMemberResult("association_unresolved", "anatomical_consensus_opt_in_required")
        error = self._spatial_available() or self._spatial_clock(at_interval)
        if error:
            return SpatialMemberResult("association_unresolved", error)
        before = self._members()
        result = self._ledger.join_anatomical_consensus(proof, anatomical_policy=self._anatomical_policy,
            at_interval=at_interval, delivery_context=delivery_context)
        self._sync()
        self._spatial_note(before, result.retired_keys, result.corrected_members)
        return SpatialMemberResult(result.disposition, result.reason,
            withdrawn_members=tuple(sorted(set(self._withdrawn(before)) | set(result.corrected_members))),
            retired_keys=result.retired_keys, association_basis="current_registered_anatomical_consensus")

    def _private_image_region_ambiguity(self, stamp):
        """Detached bounded inspection only; never sync, validate or grant custody."""
        value=dict(available=False,stamp_match=False,cause='unavailable',
            birth_established=None,birth_quarantined=None,vote_blocked=None,
            provisional_extra=None,fragment_return_current=None,
            fragment_original=None,fragment_anchors=[],fragment_anchor_total=0,fragment_anchor_omitted=0)
        if type(stamp) is not ObservationStamp or type(stamp.member) is not MemberKey:return value
        try:m=self._ledger._members.get(stamp.member) if self._ledger is not None else None
        except TypeError:return value
        if m is None:return value
        if m.observation.stamp!=stamp:
            value['cause']='stale_stamp';return value
        fact=m.image_region_fragment_provisional
        value.update(available=True,stamp_match=True,
            cause=('fragment_provisional' if fact is not None else 'birth_quarantined'
                if m.image_region_birth_quarantined else 'vote_blocked'
                if m.image_region_vote_blocked_stamp is not None else None),
            birth_established=m.image_region_birth_established,
            birth_quarantined=m.image_region_birth_quarantined,
            vote_blocked=m.image_region_vote_blocked_stamp is not None,
            provisional_extra=m.image_region_provisional_extra is not None,
            fragment_return_current=m.image_region_fragment_return==stamp)
        def compact(s):
            if type(s) is not ObservationStamp or type(s.member) is not MemberKey:return None
            k=s.member
            return [k.source_id,k.source_epoch,k.raw_id,k.lifecycle_generation,s.lifecycle_started_frame,s.frame_id]
        if fact is not None:
            value['fragment_original']=compact(fact.original)
            anchors=fact.anchors
            value['fragment_anchor_total']=len(anchors)
            value['fragment_anchors']=[v for s in anchors[:8] if (v:=compact(s)) is not None]
            value['fragment_anchor_omitted']=len(anchors)-len(value['fragment_anchors'])
        return value

    def join_estimated_image_regions(self, proof: EstimatedImageRegionProof, *, at_interval: SpatialClockContext,
                                     delivery_context: CurrentVisualDeliveryContext) -> SpatialMemberResult:
        if self._image_region_policy is None:
            return SpatialMemberResult("association_unresolved","estimated_image_regions_opt_in_required")
        error=self._spatial_available() or self._spatial_clock(at_interval)
        if error:
            return SpatialMemberResult("association_unresolved",error)
        before=self._members()
        result=self._ledger.join_estimated_image_regions(proof,projective_policy=self._image_region_policy,
            at_interval=at_interval,delivery_context=delivery_context,
            live_track_continuity=self._image_live_track_continuity,
            contradictory_cue_quarantine=self._image_contradictory_cue_quarantine,
            semantic_joint_max_mean_px=self._semantic_joint_max_mean_px,
            unresolved_birth_quarantine=self._unresolved_birth_quarantine)
        self._sync()
        self._spatial_note(before,result.retired_keys,result.corrected_members)
        return SpatialMemberResult(result.disposition,result.reason,
            withdrawn_members=tuple(sorted(set(self._withdrawn(before)) | set(result.corrected_members))),
            retired_keys=result.retired_keys,association_basis=result.reason if result.disposition=="accepted" else "estimated_current_image_regions")

    def estimated_split_custody_diagnostic(self, view, *, at_interval, observed_monotonic_s):
        """Optional detached refusal observation, never an owner permission."""
        return self._ledger.estimated_split_custody_diagnostic(view,
            at_interval=at_interval,observed_monotonic_s=observed_monotonic_s)

    def estimated_split_original_destination(self, view: EstimatedSplitCustodyView,
            original: ObservationStamp, *, at_interval: SpatialClockContext) -> PhysicalPersonKey:
        """Read-only finite OLD-original routing, never a measurement grant."""
        error=self._spatial_available() or self._spatial_clock(at_interval)
        if error:raise ValueError(error)
        self._sync()
        if (type(view) is not EstimatedSplitCustodyView or type(original) is not ObservationStamp
                or not any(v is view for v in self._ledger.estimated_split_custodies(at_interval=at_interval))):
            raise ValueError('split_custody_owner_changed')
        split=next((s for s in view.split_stamps if s.member==original.member),None)
        if (split is None or original.lifecycle_started_frame!=split.lifecycle_started_frame
                or original.clock_domain!=split.clock_domain or original.clock_basis!=split.clock_basis
                or original.frame_id>split.frame_id or original.acquired_at_s>split.acquired_at_s
                or original.media_time_s is None or split.media_time_s is None
                or original.media_time_s>split.media_time_s):
            raise ValueError('split_custody_original_binding')
        if view.rejoined_key is not None or not view.native_destinations:return view.origin_key
        return next(key for stamp,key in view.native_destinations if stamp.member==original.member)

    def spatial_memberships(self, *, at_interval: SpatialClockContext) -> tuple[SpatialMembershipDTO, ...]:
        error = self._spatial_available() or self._spatial_clock(at_interval)
        if error:
            raise ValueError(error)
        self._sync()
        if len(self._spatial_withdrawn) > 128 or len(self._spatial_retired) > 128:
            raise ValueError("spatial_barrier_capacity")
        episodes=(self._ledger.estimated_split_custodies(at_interval=at_interval)
            if self._image_region_policy is not None and self._image_live_track_continuity
            and self._semantic_joint_max_mean_px is not None and self._unresolved_birth_quarantine else ())
        custody={key:view for view in episodes for key in (
            (view.rejoined_key,) if view.rejoined_key is not None else view.successor_keys)}
        issued = {}
        for key in sorted(self._ledger._persons):
            members = self._ledger._views(key)
            eligible = tuple(SpatialMemberEligibility(m.observation,
                             self._ledger._members[m.observation.member].spatial_lifecycle,
                             (reason := (self._spatial_member_reason(m.observation, at_interval)
                                         or self._position_delivery_error(m.observation,at_interval)
                                         or (self._ledger.projective_continuity_error(m.observation, at_interval)
                                             if self._projective_policy is not None else None)
                                         or (self._ledger.anatomical_consensus_error(m.observation, at_interval)
                                             if self._anatomical_policy is not None else None)
                                         or (self._ledger.image_region_error(m.observation, at_interval)
                                             if self._image_region_policy is not None else None))) is None,
                             reason or "current_active_own_source") for m in members)
            conflict_reason, conflict = self._household_check(key)
            if conflict_reason and conflict is None:
                raise ValueError(conflict_reason)
            originals = [m.household for m, e in zip(members, eligible)
                         if e.eligible and m.household.identity_key is not None and m.household.constraint == "eligible"]
            claim = originals[0] if originals and all(h == originals[0] for h in originals) and conflict is None else None
            bases = {self._ledger._members[m.observation.member].association_basis for m in members}
            basis = next((b for b in ("estimated_current_image_regions_unresolved", "estimated_live_track_continuity", "estimated_image_region_continuation", "estimated_current_image_regions", "current_registered_anatomical_consensus_unresolved", "current_registered_anatomical_consensus",
                                     "prior_projective_continuity_unresolved", "prior_projective_continuity",
                                     "current_visual_unresolved", "current_visual", "qualified_spatial") if b in bases), "source_local")
            if any(self._ledger._members[m.observation.member].image_region_conflict_custody for m in members):
                basis="estimated_image_region_conflict_custody"
            elif basis in {"estimated_current_image_regions","estimated_image_region_continuation","estimated_live_track_continuity"} and all(
                    self._ledger._members[m.observation.member].image_region_semantic for m in members):
                basis="estimated_semantic_joint_correspondence"
            dto = SpatialMembershipDTO(self._owner, self.run_id, key, self._revision, members,
                                       self._ledger.source_bindings(), eligible, claim,
                                       tuple(sorted(self._spatial_withdrawn)), tuple(sorted(self._spatial_retired)), conflict, basis,
                                       tuple(ImageRegionSupportView(m.observation,original.stamp,original.ingress_monotonic_s,
                                           self._ledger._members[m.observation.member].image_region_native_seed)
                                           for m in members if self._image_region_policy is not None
                                           and basis in {"estimated_current_image_regions","estimated_image_region_continuation","estimated_live_track_continuity","estimated_image_region_conflict_custody","estimated_semantic_joint_correspondence"}
                                           and (original:=self._ledger._members[m.observation.member].image_region_original) is not None),custody.get(key))
            prior = self._spatial_issued.get(key)
            issued[key] = prior if prior == dto else dto
        self._spatial_issued = issued
        return tuple(issued.values())

    def _spatial_member_reason(self, stamp: ObservationStamp, context: SpatialClockContext) -> str | None:
        reason=self._ledger.spatial_member_error(stamp,context)
        if self._image_region_policy is not None and reason=="current_rejoin_proof_required":
            return "image_region_rejoin_proof_required"
        return reason

    def _position_delivery_error(self, stamp: ObservationStamp, context: SpatialClockContext) -> str | None:
        if not self._current_visual_enabled:
            return None
        if self._image_region_policy is not None and self._image_live_track_continuity:
            return self._ledger.image_region_position_delivery_error(stamp,context,live_track_continuity=True)
        return self._ledger.current_visual_liveness_error(stamp)

    @staticmethod
    def _spatial_original_binding_error(reference: SpatialMeasurementReference, context: SpatialClockContext) -> str | None:
        current=next(i for i in context.intervals if i.binding.source_id==reference.stamp.member.source_id)
        original=reference.acquisition_interval
        if (original.binding != current.binding or original.acquisition_min_s != current.acquisition_min_s
                or original.acquisition_max_s != current.acquisition_max_s
                or reference.calibration_revision != current.binding.calibration_revision
                or reference.transform_id != current.binding.transform_id
                or not all(_token(v) for v in (reference.evidence_id, reference.target_frame,
                                               reference.world_revision, reference.measured_stage_id))
                or reference.origin != "measured" or reference.units != "m"):
            return "spatial_original_measurement_binding"
        return None

    def _spatial_reference_error(self, dto: SpatialMembershipDTO, reference: SpatialMeasurementReference,
                                 context: SpatialClockContext) -> str | None:
        if (type(reference) is not SpatialMeasurementReference or self._ledger._stamp_error(reference.stamp)
                or type(reference.acquisition_interval) is not SpatialClockInterval):
            return "malformed_spatial_measurement_reference"
        stamp = reference.stamp
        if reference.owner != stamp.member or not any(m.observation == stamp for m in dto.members):
            return "spatial_reference_member"
        error = self._ledger.spatial_member_error(stamp, context)
        if error:
            if self._image_region_policy is not None and error=="current_rejoin_proof_required":
                # Expected missing proof never launders a forged original binding.
                return (self._spatial_original_binding_error(reference,context)
                        or "image_region_rejoin_proof_required")
            return error
        error = self._position_delivery_error(stamp,context)
        if error:
            return error
        if self._projective_policy is not None:
            error = self._ledger.projective_continuity_error(stamp, context)
            if error:
                return error
        if self._anatomical_policy is not None:
            error = self._ledger.anatomical_consensus_error(stamp, context)
            if error:
                return error
        if self._image_region_policy is not None:
            error = self._ledger.image_region_error(stamp, context)
            if error:
                return (self._spatial_original_binding_error(reference,context) or error
                    if error in {"image_region_ambiguous_member","image_region_contradictory_cue_member"} else error)
        error=self._spatial_original_binding_error(reference,context)
        if error: return error
        if stamp.frame_id <= self._spatial_consumed.get(stamp.member, -1):
            return "spatial_member_measurement_consumed"
        if stamp.member not in self._spatial_consumed and len(self._spatial_consumed) >= 128:
            return "spatial_member_frontier_capacity"
        return None

    def _spatial_grant_error(self, dto: object, reference: SpatialMeasurementReference,
                             context: SpatialClockContext, *, purpose: str = "strict"
                             ) -> tuple[str | None, HouseholdClaimConflict | None]:
        error = self._spatial_available() or self._spatial_clock(context)
        if error:
            return error, None
        self._sync()
        if (type(dto) is not SpatialMembershipDTO or dto.owner_instance is not self._owner
                or type(dto.physical_key) is not PhysicalPersonKey or dto.run_id != self.run_id
                or dto.physical_key.run_id != self.run_id or type(dto.physical_key.serial) is not int
                or not 0 < dto.physical_key.serial < 2**64):
            return "foreign_or_unissued_spatial_dto", None
        current = (self._spatial_issued.get(dto.physical_key) is dto and dto.association_revision == self._revision
                   and dto.source_bindings == self._ledger.source_bindings()
                   and dto.members == self._ledger._views(dto.physical_key))
        physical_only = purpose == "physical_only_quarantine"
        if purpose not in {"strict", "physical_only_quarantine"}:
            return "spatial_reservation_purpose_mismatch", None
        # Only the exact issued DTO can carry the original quarantine capability.
        # Strict order/denials stay unchanged, including reporting a new conflict.
        if physical_only and not current:
            return "spatial_membership_revision_changed", None
        original = dto.household_conflict if physical_only else None
        error, conflict = self._household_check(dto.physical_key)
        if error and (not physical_only or error != "settled_household_claim_conflict"):
            return error, original if physical_only else conflict
        if not current:
            return "spatial_membership_revision_changed", None
        houses = [m.household for m in dto.members if m.household.identity_key is not None]
        image_quarantine=(physical_only and self._image_region_policy is not None
            and dto.association_basis in {"estimated_current_image_regions","estimated_image_region_continuation","estimated_live_track_continuity","estimated_image_region_conflict_custody","estimated_semantic_joint_correspondence"}
            and type(original) is HouseholdClaimConflict and conflict==original and error=="settled_household_claim_conflict"
            and all((dto.physical_key,SourceObservation(m.observation,m.household)) in original.owners
                    for m in dto.members if m.household.identity_key is not None))
        if any(m.household.constraint != "eligible" for m in dto.members) or (houses and any(h != houses[0] for h in houses) and not image_quarantine):
            return "spatial_household_conflict", original
        if physical_only:
            if type(original) is not HouseholdClaimConflict:
                return "physical_quarantine_conflict_required", None
            if (error != "settled_household_claim_conflict" or conflict != original
                    or original.reason != "settled_household_claim_conflict"
                    or original.run_id != self.run_id or original.association_revision != self._revision):
                return "physical_quarantine_conflict_changed", original
        # Physical-only bypasses EXACTLY the checked settled attribution conflict,
        # never the original membership, lifecycle, measurement or consumption gate.
        return self._spatial_reference_error(dto, reference, context), original

    def prepare_spatial_member(self, dto: SpatialMembershipDTO, reference: SpatialMeasurementReference,
                               *, at_interval: SpatialClockContext) -> SpatialMemberResult:
        error, conflict = self._spatial_grant_error(dto, reference, at_interval)
        if error:
            return SpatialMemberResult("denied", error, household_conflict=conflict)
        if len(self._reservations) >= min(self._max_reservations, self._spatial_policy.max_entities):
            return SpatialMemberResult("capacity", "reservation_capacity")
        self._reservation_serial += 1
        reservation = SpatialMemberReservation(self._owner, self._reservation_serial, dto, reference, at_interval)
        self._reservations[reservation.serial] = reservation
        return SpatialMemberResult("reserved", "exact_current_spatial_member", reservation=reservation,
                                   association_basis=dto.association_basis)

    def prepare_spatial_physical_member(self, dto: SpatialMembershipDTO, reference: SpatialMeasurementReference,
                                       *, at_interval: SpatialClockContext) -> SpatialMemberResult:
        """Reserve an original physical measurement; never grant settled attribution."""
        purpose = "physical_only_quarantine"
        error, conflict = self._spatial_grant_error(dto, reference, at_interval, purpose=purpose)
        if error:
            return SpatialMemberResult("denied", error, household_conflict=conflict, purpose=purpose)
        if len(self._reservations) >= min(self._max_reservations, self._spatial_policy.max_entities):
            return SpatialMemberResult("capacity", "reservation_capacity", household_conflict=conflict, purpose=purpose)
        self._reservation_serial += 1
        reservation = SpatialMemberReservation(self._owner, self._reservation_serial, dto, reference,
                                               at_interval, purpose, conflict)
        self._reservations[reservation.serial] = reservation
        return SpatialMemberResult("reserved", "exact_current_spatial_physical_member", reservation=reservation,
                                   household_conflict=conflict, association_basis=dto.association_basis, purpose=purpose)

    def discard_spatial_member(self, reservation: SpatialMemberReservation) -> SpatialMemberResult:
        if (self._committing or type(reservation) is not SpatialMemberReservation or type(reservation.serial) is not int
                or reservation.owner_instance is not self._owner
                or self._reservations.get(reservation.serial) is not reservation):
            return SpatialMemberResult("denied", "foreign_or_consumed_spatial_reservation")
        del self._reservations[reservation.serial]
        return SpatialMemberResult("discarded", "spatial_intent_discarded",
                                   household_conflict=reservation.household_conflict,
                                   association_basis=reservation.dto.association_basis, purpose=reservation.purpose)

    def commit_spatial_member(self, reservation: SpatialMemberReservation, *, at_interval: SpatialClockContext,
                              consumer: Callable[[SpatialMembershipDTO, SpatialMeasurementReference], object]) -> SpatialMemberResult:
        self._sync()
        if (self._committing or type(reservation) is not SpatialMemberReservation or type(reservation.serial) is not int
                or reservation.owner_instance is not self._owner):
            return SpatialMemberResult("denied", "foreign_or_consumed_spatial_reservation")
        if reservation.purpose != "strict" or reservation.household_conflict is not None:
            return SpatialMemberResult("denied", "spatial_reservation_purpose_mismatch")
        error, conflict = self._spatial_grant_error(reservation.dto, reservation.reference, at_interval)
        exact = self._reservations.get(reservation.serial) is reservation
        if exact:
            del self._reservations[reservation.serial]  # Consume before callback, including all failures.
        if error or not exact or not callable(consumer):
            return SpatialMemberResult("denied", error or ("foreign_or_consumed_spatial_reservation" if not exact else "invalid_consumer"),
                                       household_conflict=conflict)
        self._spatial_consumed[reservation.reference.stamp.member] = reservation.reference.stamp.frame_id
        self._committing = True
        try:
            value = consumer(reservation.dto, reservation.reference)
        finally:
            self._committing = False
        return SpatialMemberResult("committed", "exact_current_spatial_member", consumer_value=value,
                                   association_basis=reservation.dto.association_basis)

    def commit_spatial_physical_member(self, reservation: SpatialMemberReservation, *, at_interval: SpatialClockContext,
                                      consumer: Callable[[SpatialMembershipDTO, SpatialMeasurementReference], object]
                                      ) -> SpatialMemberResult:
        """Consume one exact quarantine intent after rechecking original ownership."""
        purpose = "physical_only_quarantine"
        self._sync()
        if (self._committing or type(reservation) is not SpatialMemberReservation or type(reservation.serial) is not int
                or reservation.owner_instance is not self._owner
                or self._reservations.get(reservation.serial) is not reservation):
            return SpatialMemberResult("denied", "foreign_or_consumed_spatial_reservation", purpose=purpose)
        if (reservation.purpose != purpose or type(reservation.household_conflict) is not HouseholdClaimConflict
                or reservation.household_conflict is not reservation.dto.household_conflict):
            return SpatialMemberResult("denied", "spatial_reservation_purpose_mismatch", purpose=purpose)
        error, conflict = self._spatial_grant_error(reservation.dto, reservation.reference, at_interval, purpose=purpose)
        # Consume before callback, including invalid/expired/failed commits. A
        # refused transaction never advances the original measurement frontier.
        del self._reservations[reservation.serial]
        if error or not callable(consumer):
            return SpatialMemberResult("denied", error or "invalid_consumer",
                                       household_conflict=conflict, purpose=purpose)
        self._spatial_consumed[reservation.reference.stamp.member] = reservation.reference.stamp.frame_id
        self._committing = True
        try:
            value = consumer(reservation.dto, reservation.reference)
        finally:
            self._committing = False
        return SpatialMemberResult("committed", "exact_current_spatial_physical_member", consumer_value=value,
                                   household_conflict=conflict, association_basis=reservation.dto.association_basis,
                                   purpose=purpose)
