from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator
from scipy.special import betaincinv

from .base import Confidence, ContractModel, Sha256, TimestampUs

IDENTITY_CALIBRATION_GENERATOR = "scripts/identity_v2_calibrate.py"

# Generic open-set authority comes from an independent benchmark holdout.  The
# household stratum is deliberately smaller: it is a local domain-shift gate,
# not a substitute for the large-N safety claim.
IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_KNOWN_PERSONS = 50
IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_UNKNOWN_PERSONS = 50
IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_KNOWN_PERSONS = 300
IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_UNKNOWN_PERSONS = 300
IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_KNOWN_ENCOUNTERS = 10
IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_UNKNOWN_ENCOUNTERS = 10
IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_KNOWN_ENCOUNTERS = 20
IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_UNKNOWN_ENCOUNTERS = 20

IDENTITY_CALIBRATION_EVIDENCE_UNIT_BUCKET_US = 5_000_000
IDENTITY_CALIBRATION_MAXIMUM_OBSERVATIONS_PER_UNIT = 300
IDENTITY_CALIBRATION_MAXIMUM_FIT_UNITS_PER_ENCOUNTER = 8
IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_CANDIDATES = 2
IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_IMPOSTORS = 1
IDENTITY_CALIBRATION_CONFIDENCE_LEVEL = 0.95
IDENTITY_CALIBRATION_CONFIDENCE_METHOD = "clopper_pearson_one_sided"

IDENTITY_CALIBRATION_MAX_HOLDOUT_FAR = 0.01
IDENTITY_CALIBRATION_MAX_HOLDOUT_FRR = 0.35
IDENTITY_CALIBRATION_MAX_HOLDOUT_MISIDENTIFICATION_RATE = 0.01
IDENTITY_CALIBRATION_MAX_HARMFUL_PRIOR_CHANGES = 0
IDENTITY_CALIBRATION_MAX_PRIOR_REJECTION_RESCUES = 0


class CalibrationEvidenceSource(StrEnum):
    SHADOW = "shadow"
    REPLAY = "replay"


class CalibrationEvidenceStratum(StrEnum):
    BENCHMARK = "benchmark"
    HOUSEHOLD = "household"


class CalibrationPartition(StrEnum):
    TRAIN = "train"
    HOLDOUT = "holdout"


class CalibrationSplitStrategy(StrEnum):
    SUBJECT_DISJOINT = "subject_disjoint"
    SESSION_DISJOINT = "session_disjoint"


class CalibrationTruthKind(StrEnum):
    RESIDENT = "resident"
    VISITOR = "visitor"
    UNKNOWN = "unknown"


def one_sided_binomial_upper_confidence_bound(
    failure_count: int,
    trial_count: int,
    *,
    confidence_level: float = IDENTITY_CALIBRATION_CONFIDENCE_LEVEL,
) -> float:
    """Return the exact Clopper-Pearson one-sided upper bound.

    Benchmark authority passes encounter-worst outcomes into truth-person-worst
    Bernoulli trials. In particular, zero failures in 300 independent people
    produces a 95% upper bound of approximately 0.9936%, while repeated frames,
    units, tracklets, or encounters never increase ``trial_count``.
    """

    failures = int(failure_count)
    trials = int(trial_count)
    confidence = float(confidence_level)
    if trials < 1:
        raise ValueError("binomial confidence bounds require at least one trial")
    if failures < 0 or failures > trials:
        raise ValueError("binomial failures must be within [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be within (0, 1)")
    if failures == trials:
        return 1.0
    value = float(betaincinv(failures + 1, trials - failures, confidence))
    if not math.isfinite(value):
        raise ValueError("binomial confidence bound is not finite")
    return min(1.0, max(0.0, value))


class CalibrationCandidateScore(ContractModel):
    subject_id: str = Field(min_length=1, max_length=240)
    identity_kind: Literal["resident", "visitor"]
    raw_similarity: float = Field(ge=-1.0, le=1.0)
    hard_allowed: bool = True
    hard_constraint_reason: str | None = Field(default=None, max_length=200)
    gallery_exemplar_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _constraint_reason_is_coherent(self) -> "CalibrationCandidateScore":
        if self.hard_allowed and self.hard_constraint_reason is not None:
            raise ValueError(
                "allowed candidates may not carry a hard constraint reason"
            )
        if not self.hard_allowed and not self.hard_constraint_reason:
            raise ValueError("blocked candidates require a hard constraint reason")
        return self


class ShadowIdentityEvidenceRecord(ContractModel):
    """Score-only shadow/replay evidence; biometric vectors are never serialized."""

    contract: Literal["noesis.identity.shadow_score_evidence"]
    contract_version: Literal[2]
    event_id: Sha256
    sequence: int = Field(ge=0)
    previous_event_id: Sha256 | None = None
    observed_at_us: TimestampUs
    source: CalibrationEvidenceSource
    session_id: str = Field(min_length=1, max_length=200)
    runtime: Literal["ds8", "ds9", "replay", "test"]
    runtime_mode: Literal["shadow", "authoritative"]
    run_id: str = Field(min_length=1, max_length=200)
    camera_id: str = Field(min_length=1, max_length=200)
    tracker_id: str = Field(min_length=1, max_length=200)
    frame_id: int = Field(ge=0)
    observation_id: str = Field(min_length=1, max_length=200)
    model_sha256: Sha256
    model_semantic_profile_sha256: Sha256
    model_layer: str = Field(min_length=1, max_length=200)
    embedding_dim: int = Field(ge=1)
    quality: Confidence
    candidates: tuple[CalibrationCandidateScore, ...]
    pre_prior_winner_id: str | None = Field(default=None, max_length=240)
    final_winner_id: str | None = Field(default=None, max_length=240)
    final_outcome: Literal["resident", "visitor", "unknown"]
    calibrated_confidence: Confidence
    reject_reason: str | None = Field(default=None, max_length=200)
    prior_changed_winner: bool = False

    @model_validator(mode="after")
    def _record_is_coherent(self) -> "ShadowIdentityEvidenceRecord":
        if self.sequence == 0 and self.previous_event_id is not None:
            raise ValueError("first evidence event may not reference a predecessor")
        if self.sequence > 0 and self.previous_event_id is None:
            raise ValueError("non-first evidence events require a predecessor")
        ids = [row.subject_id for row in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate subject IDs must be unique")
        candidate_ids = set(ids)
        if (
            self.pre_prior_winner_id is not None
            and self.pre_prior_winner_id not in candidate_ids
        ):
            raise ValueError("pre-prior winner must be a serialized candidate")
        if self.final_outcome == "unknown":
            if self.final_winner_id is not None:
                raise ValueError("unknown outcomes may not carry a final winner")
            if not self.reject_reason:
                raise ValueError("unknown outcomes require a reject reason")
            if self.prior_changed_winner:
                raise ValueError(
                    "resident prior may not convert rejection into acceptance"
                )
        else:
            if self.final_winner_id not in candidate_ids:
                raise ValueError("accepted outcomes require a serialized final winner")
            if self.reject_reason is not None:
                raise ValueError("accepted outcomes may not carry a reject reason")
            winner = next(
                row for row in self.candidates if row.subject_id == self.final_winner_id
            )
            if winner.identity_kind != self.final_outcome:
                raise ValueError("accepted outcome disagrees with winner identity kind")
        if (
            self.prior_changed_winner
            and self.pre_prior_winner_id == self.final_winner_id
        ):
            raise ValueError("prior_changed_winner requires different winners")
        return self


class IdentityEvidenceChainCheckpoint(ContractModel):
    """Tail/head checkpoint for one bounded score-evidence JSONL chain."""

    contract: Literal["noesis.identity.shadow_score_evidence_chain"]
    contract_version: Literal[1]
    next_sequence: int = Field(ge=0)
    first_sequence: int | None = Field(default=None, ge=0)
    previous_event_id: Sha256 | None = None
    tail_event_id: Sha256 | None = None
    retained_count: int = Field(ge=0)
    retained_bytes: int = Field(ge=0)
    updated_at_us: TimestampUs
    checkpoint_sha256: Sha256

    @model_validator(mode="after")
    def _chain_window_is_coherent(self) -> "IdentityEvidenceChainCheckpoint":
        if self.retained_count == 0:
            if self.first_sequence is not None or self.tail_event_id is not None:
                raise ValueError(
                    "empty evidence chain may not declare retained endpoints"
                )
            if self.retained_bytes != 0:
                raise ValueError("empty evidence chain must have zero retained bytes")
            if self.next_sequence == 0 and self.previous_event_id is not None:
                raise ValueError("new evidence chain may not reference a predecessor")
            if self.next_sequence > 0 and self.previous_event_id is None:
                raise ValueError(
                    "pruned evidence chain requires its predecessor checkpoint"
                )
            return self
        if self.first_sequence is None or self.tail_event_id is None:
            raise ValueError("retained evidence chain requires head and tail endpoints")
        if self.first_sequence + self.retained_count != self.next_sequence:
            raise ValueError("evidence chain sequence window is not contiguous")
        if self.first_sequence == 0 and self.previous_event_id is not None:
            raise ValueError("unpruned evidence chain may not reference a predecessor")
        if self.first_sequence > 0 and self.previous_event_id is None:
            raise ValueError("pruned evidence chain requires a predecessor checkpoint")
        if self.retained_bytes <= 0:
            raise ValueError("retained evidence chain requires positive bytes")
        return self


class CalibrationDatasetProvenance(ContractModel):
    """Immutable provenance for one benchmark or household capture corpus."""

    source_name: str = Field(min_length=1, max_length=200)
    source_revision: str = Field(min_length=1, max_length=200)
    source_manifest_sha256: Sha256
    model_semantic_profile_sha256: Sha256
    labeling_protocol_revision: str = Field(min_length=1, max_length=200)
    population_basis: Literal[
        "licensed_real_people",
        "owner_consented_household_people",
    ]


class CalibrationTruthLabel(ContractModel):
    event_id: Sha256
    partition: CalibrationPartition
    truth_kind: CalibrationTruthKind
    truth_subject_id: str | None = Field(default=None, max_length=240)
    truth_person_key: str = Field(min_length=1, max_length=240)
    encounter_id: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _truth_is_coherent(self) -> "CalibrationTruthLabel":
        if self.truth_kind is CalibrationTruthKind.UNKNOWN:
            if self.truth_subject_id is not None:
                raise ValueError("unknown truth may not carry a subject ID")
        elif not self.truth_subject_id:
            raise ValueError("known truth requires a subject ID")
        return self


class IdentityEvidenceLabelSet(ContractModel):
    contract: Literal["noesis.identity.evidence_labels"]
    contract_version: Literal[2]
    evidence_sha256: Sha256
    evidence_stratum: CalibrationEvidenceStratum
    split_strategy: CalibrationSplitStrategy
    provenance: CalibrationDatasetProvenance
    labeled_by: str = Field(min_length=1, max_length=200)
    labeling_revision: str = Field(min_length=1, max_length=200)
    labels: tuple[CalibrationTruthLabel, ...] = Field(min_length=1)

    @field_validator("labels")
    @classmethod
    def _labels_are_unique(
        cls, labels: tuple[CalibrationTruthLabel, ...]
    ) -> tuple[CalibrationTruthLabel, ...]:
        event_ids = [row.event_id for row in labels]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("evidence labels contain duplicate event IDs")
        return labels


class LabeledCalibrationSample(ContractModel):
    evidence: ShadowIdentityEvidenceRecord
    label: CalibrationTruthLabel

    @model_validator(mode="after")
    def _event_ids_match(self) -> "LabeledCalibrationSample":
        if self.evidence.event_id != self.label.event_id:
            raise ValueError("evidence and truth label event IDs disagree")
        truth_subject_id = self.label.truth_subject_id
        if truth_subject_id is not None:
            truth_rows = tuple(
                row
                for row in self.evidence.candidates
                if row.subject_id == truth_subject_id
            )
            if (
                truth_rows
                and truth_rows[0].identity_kind != self.label.truth_kind.value
            ):
                raise ValueError(
                    "known truth kind disagrees with candidate identity kind"
                )
        return self


class CalibrationEvidenceUnitPolicy(ContractModel):
    algorithm: Literal["session_tracklet_time_bucket_v1"]
    bucket_duration_us: Literal[5_000_000]
    maximum_observations_per_unit: Literal[300]
    fit_thinning: Literal["temporal_cover_center_representative_v1"]
    maximum_fit_units_per_encounter: Literal[8]
    fit_representatives_per_unit: Literal[1]
    authority_aggregation: Literal["encounter_worst_case_v1"]


def calibration_evidence_unit_id(
    *,
    session_id: str,
    run_id: str,
    source: CalibrationEvidenceSource,
    camera_id: str,
    tracker_id: str,
    bucket_index: int,
) -> str:
    payload = {
        "algorithm": "session_tracklet_time_bucket_v1",
        "bucket_index": int(bucket_index),
        "camera_id": str(camera_id),
        "run_id": str(run_id),
        "session_id": str(session_id),
        "source": source.value,
        "tracker_id": str(tracker_id),
    }
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"noesis-identity-evidence-unit-v1\0" + body).hexdigest()


class CalibrationEvidenceUnit(ContractModel):
    unit_id: Sha256
    partition: CalibrationPartition
    truth_kind: CalibrationTruthKind
    truth_subject_id: str | None = Field(default=None, max_length=240)
    truth_person_key: str = Field(min_length=1, max_length=240)
    encounter_id: str = Field(min_length=1, max_length=240)
    session_id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(min_length=1, max_length=200)
    source: CalibrationEvidenceSource
    camera_id: str = Field(min_length=1, max_length=200)
    tracker_id: str = Field(min_length=1, max_length=200)
    bucket_index: int = Field(ge=0)
    event_ids: tuple[Sha256, ...] = Field(min_length=1, max_length=300)
    fit_event_id: Sha256 | None = None

    @model_validator(mode="after")
    def _unit_is_coherent(self) -> "CalibrationEvidenceUnit":
        if len(self.event_ids) != len(set(self.event_ids)):
            raise ValueError("evidence unit contains duplicate event IDs")
        if self.fit_event_id is not None and self.fit_event_id not in self.event_ids:
            raise ValueError("fit representative must belong to its evidence unit")
        expected = calibration_evidence_unit_id(
            session_id=self.session_id,
            run_id=self.run_id,
            source=self.source,
            camera_id=self.camera_id,
            tracker_id=self.tracker_id,
            bucket_index=self.bucket_index,
        )
        if self.unit_id != expected:
            raise ValueError("evidence unit ID does not match its deterministic key")
        return self


def _temporal_cover_indexes(count: int, maximum: int) -> tuple[int, ...]:
    if count <= maximum:
        return tuple(range(count))
    if maximum <= 1:
        return (count // 2,)
    return tuple((index * (count - 1)) // (maximum - 1) for index in range(maximum))


class IdentityCalibrationDataset(ContractModel):
    contract: Literal["noesis.identity.calibration_dataset"]
    contract_version: Literal[2]
    evidence_sha256: Sha256
    labels_sha256: Sha256
    evidence_stratum: CalibrationEvidenceStratum
    split_strategy: CalibrationSplitStrategy
    provenance: CalibrationDatasetProvenance
    labeled_by: str = Field(min_length=1, max_length=200)
    labeling_revision: str = Field(min_length=1, max_length=200)
    model_sha256: Sha256
    model_layer: str = Field(min_length=1, max_length=200)
    embedding_dim: int = Field(ge=1)
    evidence_chain: IdentityEvidenceChainCheckpoint
    evidence_unit_policy: CalibrationEvidenceUnitPolicy
    samples: tuple[LabeledCalibrationSample, ...] = Field(min_length=1)
    evidence_units: tuple[CalibrationEvidenceUnit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _dataset_profile_and_units_are_closed(self) -> "IdentityCalibrationDataset":
        sample_by_id = {row.evidence.event_id: row for row in self.samples}
        if len(sample_by_id) != len(self.samples):
            raise ValueError("calibration dataset contains duplicate events")
        for row in self.samples:
            evidence = row.evidence
            if (
                evidence.model_sha256 != self.model_sha256
                or evidence.model_layer != self.model_layer
                or evidence.embedding_dim != self.embedding_dim
                or evidence.model_semantic_profile_sha256
                != self.provenance.model_semantic_profile_sha256
            ):
                raise ValueError("calibration dataset mixes model profiles")

        ordered = tuple(sorted(self.samples, key=lambda item: item.evidence.sequence))
        checkpoint = self.evidence_chain
        if len(ordered) != checkpoint.retained_count:
            raise ValueError(
                "calibration dataset count disagrees with evidence checkpoint"
            )
        if ordered[0].evidence.sequence != checkpoint.first_sequence:
            raise ValueError(
                "calibration dataset head disagrees with evidence checkpoint"
            )
        expected_previous = checkpoint.previous_event_id
        expected_sequence = int(checkpoint.first_sequence or 0)
        for row in ordered:
            if row.evidence.sequence != expected_sequence:
                raise ValueError("calibration dataset evidence sequence has a gap")
            if row.evidence.previous_event_id != expected_previous:
                raise ValueError("calibration dataset evidence chain is broken")
            expected_previous = row.evidence.event_id
            expected_sequence += 1
        if expected_previous != checkpoint.tail_event_id:
            raise ValueError(
                "calibration dataset tail disagrees with evidence checkpoint"
            )

        partitions = {row.label.partition for row in self.samples}
        if partitions != {CalibrationPartition.TRAIN, CalibrationPartition.HOLDOUT}:
            raise ValueError("calibration dataset requires train and holdout samples")

        unit_ids = [row.unit_id for row in self.evidence_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("calibration dataset contains duplicate evidence units")
        covered_events = [
            event_id for unit in self.evidence_units for event_id in unit.event_ids
        ]
        if len(covered_events) != len(set(covered_events)):
            raise ValueError("an evidence event belongs to more than one evidence unit")
        if set(covered_events) != set(sample_by_id):
            raise ValueError(
                "evidence units must cover every dataset event exactly once"
            )

        encounter_facts: dict[str, tuple[object, ...]] = {}
        tracklet_encounters: dict[tuple[str, str, str, str], str] = {}
        session_facts: dict[
            str, tuple[CalibrationPartition, str, CalibrationEvidenceSource]
        ] = {}
        run_partitions: dict[str, CalibrationPartition] = {}
        subject_people: dict[str, str] = {}
        person_truth: dict[str, tuple[CalibrationTruthKind, str | None]] = {}
        candidate_kinds: dict[str, str] = {}
        for unit in self.evidence_units:
            if (
                len(unit.event_ids)
                > self.evidence_unit_policy.maximum_observations_per_unit
            ):
                raise ValueError("evidence unit exceeds the observation cap")
            facts = (
                unit.partition,
                unit.truth_kind,
                unit.truth_subject_id,
                unit.truth_person_key,
                unit.session_id,
                unit.run_id,
                unit.source,
            )
            previous_facts = encounter_facts.setdefault(unit.encounter_id, facts)
            if previous_facts != facts:
                raise ValueError("one encounter has conflicting truth or provenance")
            tracklet_key = (
                unit.session_id,
                unit.run_id,
                unit.camera_id,
                unit.tracker_id,
            )
            previous_encounter = tracklet_encounters.setdefault(
                tracklet_key, unit.encounter_id
            )
            if previous_encounter != unit.encounter_id:
                raise ValueError("one capture tracklet is split across encounters")
            session_fact = (unit.partition, unit.run_id, unit.source)
            previous_session_fact = session_facts.setdefault(
                unit.session_id, session_fact
            )
            if previous_session_fact != session_fact:
                raise ValueError("capture session crosses split, run, or source")
            previous_run_partition = run_partitions.setdefault(
                unit.run_id, unit.partition
            )
            if previous_run_partition is not unit.partition:
                raise ValueError("runtime run crosses train/holdout")
            truth_fact = (unit.truth_kind, unit.truth_subject_id)
            previous_truth_fact = person_truth.setdefault(
                unit.truth_person_key, truth_fact
            )
            if previous_truth_fact != truth_fact:
                raise ValueError("one truth person key maps to conflicting identities")
            if unit.truth_subject_id is not None:
                previous_person = subject_people.setdefault(
                    unit.truth_subject_id, unit.truth_person_key
                )
                if previous_person != unit.truth_person_key:
                    raise ValueError(
                        "one known truth subject maps to multiple person keys"
                    )

            expected_bucket = None
            for event_id in unit.event_ids:
                sample = sample_by_id[event_id]
                evidence = sample.evidence
                label = sample.label
                bucket = (
                    evidence.observed_at_us
                    // self.evidence_unit_policy.bucket_duration_us
                )
                if expected_bucket is None:
                    expected_bucket = bucket
                if (
                    label.partition is not unit.partition
                    or label.truth_kind is not unit.truth_kind
                    or label.truth_subject_id != unit.truth_subject_id
                    or label.truth_person_key != unit.truth_person_key
                    or label.encounter_id != unit.encounter_id
                    or evidence.session_id != unit.session_id
                    or evidence.run_id != unit.run_id
                    or evidence.source is not unit.source
                    or evidence.camera_id != unit.camera_id
                    or evidence.tracker_id != unit.tracker_id
                    or bucket != unit.bucket_index
                ):
                    raise ValueError("evidence unit conflicts with a member event")
                for candidate in evidence.candidates:
                    previous_kind = candidate_kinds.setdefault(
                        candidate.subject_id, candidate.identity_kind
                    )
                    if previous_kind != candidate.identity_kind:
                        raise ValueError(
                            "one candidate subject changes resident/visitor identity kind"
                        )

        if self.split_strategy is CalibrationSplitStrategy.SUBJECT_DISJOINT:
            train_people = {
                row.label.truth_person_key
                for row in self.samples
                if row.label.partition is CalibrationPartition.TRAIN
            }
            holdout_people = {
                row.label.truth_person_key
                for row in self.samples
                if row.label.partition is CalibrationPartition.HOLDOUT
            }
            if train_people & holdout_people:
                raise ValueError("truth people cross a subject-disjoint split")
            train_known_people = {
                row.label.truth_person_key
                for row in self.samples
                if row.label.partition is CalibrationPartition.TRAIN
                and row.label.truth_kind is not CalibrationTruthKind.UNKNOWN
            }
            holdout_known_people = {
                row.label.truth_person_key
                for row in self.samples
                if row.label.partition is CalibrationPartition.HOLDOUT
                and row.label.truth_kind is not CalibrationTruthKind.UNKNOWN
            }
            if len(train_known_people) < 2 or len(holdout_known_people) < 1:
                raise ValueError(
                    "subject-disjoint calibration requires two train and one holdout known people"
                )

        units_by_encounter: dict[str, list[CalibrationEvidenceUnit]] = {}
        for unit in self.evidence_units:
            units_by_encounter.setdefault(unit.encounter_id, []).append(unit)
        for units in units_by_encounter.values():
            ordered_units = sorted(
                units, key=lambda row: (row.bucket_index, row.unit_id)
            )
            selected_indexes = set(
                _temporal_cover_indexes(
                    len(ordered_units),
                    self.evidence_unit_policy.maximum_fit_units_per_encounter,
                )
            )
            for index, unit in enumerate(ordered_units):
                expected_fit_event: str | None = None
                if (
                    unit.partition is CalibrationPartition.TRAIN
                    and index in selected_indexes
                ):
                    midpoint = (
                        unit.bucket_index * self.evidence_unit_policy.bucket_duration_us
                        + self.evidence_unit_policy.bucket_duration_us // 2
                    )
                    expected_fit_event = min(
                        unit.event_ids,
                        key=lambda event_id: (
                            abs(
                                sample_by_id[event_id].evidence.observed_at_us
                                - midpoint
                            ),
                            sample_by_id[event_id].evidence.observed_at_us,
                            event_id,
                        ),
                    )
                if unit.fit_event_id != expected_fit_event:
                    raise ValueError("evidence-unit fit thinning is not deterministic")
        return self


class IdentityOpenSetPolicyContract(ContractModel):
    appearance_floor: float = Field(ge=-1.0, le=1.0)
    quality_floor: Confidence
    calibrated_confidence_floor: Confidence
    ambiguity_margin_floor: float = Field(ge=0.0, le=1.0)
    calibration_slope: float = Field(gt=0.0, le=200.0)
    calibration_midpoint: float = Field(ge=-1.0, le=1.0)
    resident_prior_bonus: float = Field(ge=0.0, le=0.05)
    resident_prior_cap: float = Field(ge=0.0, le=0.05)
    unknown_utility: Confidence

    @model_validator(mode="after")
    def _prior_is_bounded(self) -> "IdentityOpenSetPolicyContract":
        if self.resident_prior_bonus > self.resident_prior_cap:
            raise ValueError("resident prior bonus exceeds its cap")
        return self


class CalibrationMetrics(ContractModel):
    confidence_level: Literal[0.95]
    confidence_method: Literal["clopper_pearson_one_sided"]
    confidence_unit: Literal["truth_person_worst_case"]
    observation_count: int = Field(ge=1)
    known_observation_count: int = Field(ge=1)
    unknown_observation_count: int = Field(ge=1)
    challenge_eligible_observation_count: int = Field(ge=1)
    evidence_unit_count: int = Field(ge=1)
    known_evidence_unit_count: int = Field(ge=1)
    unknown_evidence_unit_count: int = Field(ge=1)
    encounter_count: int = Field(ge=1)
    known_encounter_count: int = Field(ge=1)
    unknown_encounter_count: int = Field(ge=1)
    challenge_eligible_encounter_count: int = Field(ge=1)
    correct_accept_encounter_count: int = Field(ge=0)
    false_reject_encounter_count: int = Field(ge=0)
    misidentification_encounter_count: int = Field(ge=0)
    false_accept_encounter_count: int = Field(ge=0)
    person_count: int = Field(ge=1)
    known_person_count: int = Field(ge=1)
    unknown_person_count: int = Field(ge=1)
    known_challenge_person_count: int = Field(ge=1)
    unknown_challenge_person_count: int = Field(ge=1)
    correct_accept_person_count: int = Field(ge=0)
    false_reject_person_count: int = Field(ge=0)
    misidentification_person_count: int = Field(ge=0)
    false_accept_person_count: int = Field(ge=0)
    misidentification_challenge_person_count: int = Field(ge=0)
    false_accept_challenge_person_count: int = Field(ge=0)
    person_far: Confidence
    person_frr: Confidence
    person_misidentification_rate: Confidence
    person_identification_accuracy: Confidence
    far: Confidence
    far_upper_confidence_bound: Confidence
    frr: Confidence
    frr_upper_confidence_bound: Confidence
    misidentification_rate: Confidence
    misidentification_upper_confidence_bound: Confidence
    unknown_rejection_rate: Confidence
    identification_accuracy: Confidence
    resident_prior_changed_encounter_count: int = Field(ge=0)
    resident_prior_beneficial_encounter_count: int = Field(ge=0)
    resident_prior_harmful_encounter_count: int = Field(ge=0)
    resident_prior_rejection_rescue_encounter_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _counts_are_coherent(self) -> "CalibrationMetrics":
        if (
            self.known_observation_count + self.unknown_observation_count
            != self.observation_count
        ):
            raise ValueError(
                "known and unknown observations do not sum to observations"
            )
        if (
            self.known_evidence_unit_count + self.unknown_evidence_unit_count
            != self.evidence_unit_count
        ):
            raise ValueError("known and unknown units do not sum to evidence units")
        if (
            self.known_encounter_count + self.unknown_encounter_count
            != self.encounter_count
        ):
            raise ValueError("known and unknown encounters do not sum to encounters")
        if (
            self.correct_accept_encounter_count
            + self.false_reject_encounter_count
            + self.misidentification_encounter_count
            != self.known_encounter_count
        ):
            raise ValueError("known encounter outcomes do not sum to known encounters")
        if self.false_accept_encounter_count > self.unknown_encounter_count:
            raise ValueError("false accepts exceed unknown encounters")
        if self.known_person_count + self.unknown_person_count != self.person_count:
            raise ValueError("known and unknown people do not sum to people")
        if (
            self.correct_accept_person_count
            + self.false_reject_person_count
            + self.misidentification_person_count
            != self.known_person_count
        ):
            raise ValueError("known person outcomes do not sum to known people")
        if self.false_accept_person_count > self.unknown_person_count:
            raise ValueError("false accepts exceed unknown people")
        for encounter_failures, person_failures, label in (
            (
                self.false_accept_encounter_count,
                self.false_accept_person_count,
                "false accept",
            ),
            (
                self.misidentification_encounter_count,
                self.misidentification_person_count,
                "misidentification",
            ),
            (
                self.false_reject_encounter_count,
                self.false_reject_person_count,
                "false reject",
            ),
        ):
            if person_failures > encounter_failures or (
                label != "false reject"
                and bool(person_failures) != bool(encounter_failures)
            ):
                raise ValueError(
                    f"{label} person outcomes disagree with encounter outcomes"
                )
        if (
            self.challenge_eligible_observation_count > self.observation_count
            or self.challenge_eligible_encounter_count > self.encounter_count
            or self.known_challenge_person_count > self.known_person_count
            or self.unknown_challenge_person_count > self.unknown_person_count
            or self.misidentification_challenge_person_count
            > min(
                self.misidentification_person_count,
                self.known_challenge_person_count,
            )
            or self.false_accept_challenge_person_count
            > min(
                self.false_accept_person_count,
                self.unknown_challenge_person_count,
            )
        ):
            raise ValueError("challenge-coverage metric counts are incoherent")
        if (
            self.known_challenge_person_count == self.known_person_count
            and self.misidentification_challenge_person_count
            != self.misidentification_person_count
        ) or (
            self.unknown_challenge_person_count == self.unknown_person_count
            and self.false_accept_challenge_person_count
            != self.false_accept_person_count
        ):
            raise ValueError(
                "fully challenge-covered person failures must match all-person failures"
            )
        if (
            self.resident_prior_beneficial_encounter_count
            + self.resident_prior_harmful_encounter_count
            > self.resident_prior_changed_encounter_count
        ):
            raise ValueError("resident-prior categories exceed changed encounters")
        if (
            self.resident_prior_changed_encounter_count > self.encounter_count
            or self.resident_prior_rejection_rescue_encounter_count
            > self.resident_prior_changed_encounter_count
        ):
            raise ValueError("resident-prior encounter counts are incoherent")

        expected_rates = {
            "far": self.false_accept_encounter_count / self.unknown_encounter_count,
            "frr": self.false_reject_encounter_count / self.known_encounter_count,
            "misidentification_rate": (
                self.misidentification_encounter_count / self.known_encounter_count
            ),
            "unknown_rejection_rate": 1.0
            - (self.false_accept_encounter_count / self.unknown_encounter_count),
            "identification_accuracy": (
                self.correct_accept_encounter_count / self.known_encounter_count
            ),
            "person_far": self.false_accept_person_count / self.unknown_person_count,
            "person_frr": self.false_reject_person_count / self.known_person_count,
            "person_misidentification_rate": (
                self.misidentification_person_count / self.known_person_count
            ),
            "person_identification_accuracy": (
                self.correct_accept_person_count / self.known_person_count
            ),
        }
        for field, value in expected_rates.items():
            if abs(float(getattr(self, field)) - float(value)) > 1.0e-9:
                raise ValueError(f"{field} does not match its encounter outcomes")

        expected_bounds = {
            "far_upper_confidence_bound": one_sided_binomial_upper_confidence_bound(
                self.false_accept_challenge_person_count,
                self.unknown_challenge_person_count,
                confidence_level=self.confidence_level,
            ),
            "frr_upper_confidence_bound": one_sided_binomial_upper_confidence_bound(
                self.false_reject_person_count,
                self.known_person_count,
                confidence_level=self.confidence_level,
            ),
            "misidentification_upper_confidence_bound": one_sided_binomial_upper_confidence_bound(
                self.misidentification_challenge_person_count,
                self.known_challenge_person_count,
                confidence_level=self.confidence_level,
            ),
        }
        for field, value in expected_bounds.items():
            if abs(float(getattr(self, field)) - float(value)) > 1.0e-9:
                raise ValueError(f"{field} is not the exact declared confidence bound")
        return self


class CalibrationFitSummary(ContractModel):
    score_semantics: Literal["balanced_monotonic_match_score_not_posterior"]
    selected_evidence_unit_count: int = Field(ge=1)
    selected_encounter_count: int = Field(ge=1)
    selected_person_count: int = Field(ge=1)
    positive_pair_count: int = Field(ge=1)
    negative_pair_count: int = Field(ge=1)
    balanced_log_loss: float = Field(ge=0.0)
    balanced_brier_score: float = Field(ge=0.0, le=1.0)
    iterations: int = Field(ge=1)


class CalibrationGalleryLimits(ContractModel):
    maximum_resident_candidate_count: int = Field(ge=1, le=1024)
    maximum_visitor_candidate_count: int = Field(ge=1, le=1024)
    maximum_total_candidate_count: int = Field(ge=2, le=2048)
    maximum_exemplars_per_candidate: int = Field(ge=1, le=4096)

    @model_validator(mode="after")
    def _limits_are_coherent(self) -> "CalibrationGalleryLimits":
        if self.maximum_total_candidate_count < max(
            self.maximum_resident_candidate_count,
            self.maximum_visitor_candidate_count,
        ):
            raise ValueError("total gallery limit is below a kind-specific limit")
        return self


class CalibrationDatasetBinding(ContractModel):
    evidence_stratum: CalibrationEvidenceStratum
    dataset_sha256: Sha256
    evidence_sha256: Sha256
    labels_sha256: Sha256
    split_strategy: CalibrationSplitStrategy
    evidence_sources: tuple[CalibrationEvidenceSource, ...] = Field(
        min_length=1, max_length=2
    )
    runtimes: tuple[str, ...] = Field(min_length=1, max_length=4)
    runtime_modes: tuple[str, ...] = Field(min_length=1, max_length=4)
    truth_kinds: tuple[CalibrationTruthKind, ...] = Field(min_length=2, max_length=3)
    provenance: CalibrationDatasetProvenance
    observation_count: int = Field(ge=1)
    evidence_unit_count: int = Field(ge=1)
    encounter_count: int = Field(ge=1)

    @field_validator("evidence_sources")
    @classmethod
    def _sources_are_canonical(
        cls, sources: tuple[CalibrationEvidenceSource, ...]
    ) -> tuple[CalibrationEvidenceSource, ...]:
        if len(sources) != len(set(sources)):
            raise ValueError("dataset binding contains duplicate evidence sources")
        if sources != tuple(sorted(sources, key=lambda value: value.value)):
            raise ValueError("dataset binding evidence sources must be sorted")
        return sources

    @field_validator("runtimes", "runtime_modes")
    @classmethod
    def _text_sets_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(value).strip().lower() for value in values)
        if any(not value for value in normalized):
            raise ValueError("dataset binding runtime values must be non-empty")
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("dataset binding runtime values must be sorted and unique")
        return normalized

    @field_validator("truth_kinds")
    @classmethod
    def _truth_kinds_are_canonical(
        cls, values: tuple[CalibrationTruthKind, ...]
    ) -> tuple[CalibrationTruthKind, ...]:
        if values != tuple(sorted(set(values), key=lambda value: value.value)):
            raise ValueError("dataset binding truth kinds must be sorted and unique")
        return values


class CalibrationAcceptance(ContractModel):
    passed: Literal[True]
    confidence_level: Literal[0.95]
    confidence_method: Literal["clopper_pearson_one_sided"]
    max_benchmark_holdout_far_upper_confidence_bound: float = Field(
        ge=0.0, le=IDENTITY_CALIBRATION_MAX_HOLDOUT_FAR
    )
    max_benchmark_holdout_frr: float = Field(
        ge=0.0, le=IDENTITY_CALIBRATION_MAX_HOLDOUT_FRR
    )
    max_benchmark_holdout_misidentification_upper_confidence_bound: float = Field(
        ge=0.0,
        le=IDENTITY_CALIBRATION_MAX_HOLDOUT_MISIDENTIFICATION_RATE,
    )
    max_household_false_accept_encounters: Literal[0]
    max_household_misidentification_encounters: Literal[0]
    max_household_holdout_frr: float = Field(
        ge=0.0, le=IDENTITY_CALIBRATION_MAX_HOLDOUT_FRR
    )
    max_harmful_prior_encounters: Literal[0]
    max_prior_rejection_rescue_encounters: Literal[0]
    minimum_benchmark_train_known_persons: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_KNOWN_PERSONS
    )
    minimum_benchmark_train_unknown_persons: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_UNKNOWN_PERSONS
    )
    minimum_benchmark_holdout_known_persons: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_KNOWN_PERSONS
    )
    minimum_benchmark_holdout_unknown_persons: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_UNKNOWN_PERSONS
    )
    minimum_household_train_known_encounters: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_KNOWN_ENCOUNTERS
    )
    minimum_household_train_unknown_encounters: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_UNKNOWN_ENCOUNTERS
    )
    minimum_household_holdout_known_encounters: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_KNOWN_ENCOUNTERS
    )
    minimum_household_holdout_unknown_encounters: int = Field(
        ge=IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_UNKNOWN_ENCOUNTERS
    )


class IdentityAuthorityGateEvidence(ContractModel):
    """One immutable, owner-reviewed prerequisite for public identity cutover."""

    evidence_kind: Literal[
        "whole_frame_coordinator_replay",
        "occupied_scene_runtime",
    ]
    evidence_path: str = Field(min_length=1, max_length=1024)
    evidence_sha256: Sha256
    evidence_size_bytes: int = Field(ge=1, le=64 * 1024 * 1024)
    evidence_revision: str = Field(min_length=1, max_length=200)
    completed_at_us: TimestampUs
    passed: Literal[True]

    @field_validator("evidence_path")
    @classmethod
    def _path_is_lexically_safe(cls, value: str) -> str:
        normalized = str(value).strip()
        if "\x00" in normalized:
            raise ValueError("authority evidence path contains a NUL byte")
        return normalized


class IdentityAuthorityCutoverArtifact(ContractModel):
    """Pinned evidence that may promote scorer output to public authority.

    The calibration artifact deliberately authorizes only scorer policy.  This
    separate artifact binds the exact runtime/code/topology surface to both a
    coordinator replay and an occupied-scene validation report.
    """

    contract: Literal["noesis.identity.authority_cutover"]
    contract_version: Literal[1]
    approved_at_us: TimestampUs
    approved_by: str = Field(min_length=1, max_length=200)
    runtime: Literal["ds9"]
    model_sha256: Sha256
    model_layer: str = Field(min_length=1, max_length=200)
    embedding_dim: int = Field(ge=1)
    model_semantic_profile_sha256: Sha256
    scoring_artifact_sha256: Sha256
    authority_runtime_profile_sha256: Sha256
    camera_topology_sha256: Sha256
    camera_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    authority_scope: Literal["full_identity_runtime_public_cutover"]
    coordinator_replay: IdentityAuthorityGateEvidence
    occupied_scene: IdentityAuthorityGateEvidence

    @field_validator("camera_ids")
    @classmethod
    def _camera_ids_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(value).strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("authority cutover camera IDs must be non-empty")
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("authority cutover camera IDs must be sorted and unique")
        return normalized

    @model_validator(mode="after")
    def _independent_gates_are_coherent(self) -> "IdentityAuthorityCutoverArtifact":
        if self.coordinator_replay.evidence_kind != "whole_frame_coordinator_replay":
            raise ValueError("coordinator replay evidence has the wrong gate kind")
        if self.occupied_scene.evidence_kind != "occupied_scene_runtime":
            raise ValueError("occupied-scene evidence has the wrong gate kind")
        if (
            self.coordinator_replay.evidence_sha256
            == self.occupied_scene.evidence_sha256
            or self.coordinator_replay.evidence_path
            == self.occupied_scene.evidence_path
        ):
            raise ValueError(
                "coordinator replay and occupied-scene gates require distinct evidence"
            )
        if (
            self.coordinator_replay.completed_at_us > self.approved_at_us
            or self.occupied_scene.completed_at_us > self.approved_at_us
        ):
            raise ValueError("authority evidence may not postdate its approval")
        return self


class IdentityOpenSetCalibrationArtifact(ContractModel):
    contract: Literal["noesis.identity.open_set_calibration"]
    contract_version: Literal[2]
    generated_at_us: TimestampUs
    generator: Literal["scripts/identity_v2_calibrate.py"]
    generator_revision: str = Field(min_length=1, max_length=200)
    model_sha256: Sha256
    model_layer: str = Field(min_length=1, max_length=200)
    embedding_dim: int = Field(ge=1)
    model_semantic_profile_sha256: Sha256
    authority_scope: Literal["open_set_scorer_policy_only"]
    gallery_limits: CalibrationGalleryLimits
    evidence_unit_policy: CalibrationEvidenceUnitPolicy
    benchmark_dataset: CalibrationDatasetBinding
    household_dataset: CalibrationDatasetBinding
    benchmark_policy: IdentityOpenSetPolicyContract
    policy: IdentityOpenSetPolicyContract
    fit: CalibrationFitSummary
    benchmark_training_metrics: CalibrationMetrics
    benchmark_holdout_metrics: CalibrationMetrics
    household_training_metrics: CalibrationMetrics
    household_holdout_metrics: CalibrationMetrics
    acceptance: CalibrationAcceptance

    @model_validator(mode="after")
    def _acceptance_is_evidence_backed(self) -> "IdentityOpenSetCalibrationArtifact":
        if (
            self.benchmark_dataset.evidence_stratum
            is not CalibrationEvidenceStratum.BENCHMARK
        ):
            raise ValueError("benchmark binding must identify benchmark evidence")
        if self.benchmark_dataset.provenance.population_basis != "licensed_real_people":
            raise ValueError("benchmark authority requires licensed real-human truth")
        if (
            self.benchmark_dataset.split_strategy
            is not CalibrationSplitStrategy.SUBJECT_DISJOINT
            or self.benchmark_dataset.evidence_sources
            != (CalibrationEvidenceSource.REPLAY,)
            or self.benchmark_dataset.runtimes != ("replay",)
            or self.benchmark_dataset.runtime_modes != ("shadow",)
            or self.benchmark_dataset.truth_kinds
            != (
                CalibrationTruthKind.RESIDENT,
                CalibrationTruthKind.UNKNOWN,
            )
        ):
            raise ValueError(
                "benchmark binding must be subject-disjoint shadow replay evidence "
                "with resident and unknown truth"
            )
        if self.household_dataset.runtime_modes != ("shadow",):
            raise ValueError("household binding must contain shadow-mode evidence")
        if (
            self.household_dataset.evidence_stratum
            is not CalibrationEvidenceStratum.HOUSEHOLD
        ):
            raise ValueError("household binding must identify household evidence")
        if self.household_dataset.truth_kinds != (
            CalibrationTruthKind.RESIDENT,
            CalibrationTruthKind.UNKNOWN,
        ):
            raise ValueError(
                "household calibration must verify resident and unknown truth"
            )
        if (
            self.household_dataset.provenance.population_basis
            != "owner_consented_household_people"
        ):
            raise ValueError("household authority requires owner-consented human truth")
        if (
            self.benchmark_dataset.dataset_sha256
            == self.household_dataset.dataset_sha256
            or self.benchmark_dataset.evidence_sha256
            == self.household_dataset.evidence_sha256
            or self.benchmark_dataset.labels_sha256
            == self.household_dataset.labels_sha256
            or self.benchmark_dataset.provenance.source_manifest_sha256
            == self.household_dataset.provenance.source_manifest_sha256
        ):
            raise ValueError(
                "benchmark and household strata must bind distinct corpora"
            )
        if (
            self.benchmark_dataset.provenance.model_semantic_profile_sha256
            != self.model_semantic_profile_sha256
            or self.household_dataset.provenance.model_semantic_profile_sha256
            != self.model_semantic_profile_sha256
        ):
            raise ValueError(
                "dataset provenance disagrees with the artifact semantic profile"
            )

        for binding, train, holdout in (
            (
                self.benchmark_dataset,
                self.benchmark_training_metrics,
                self.benchmark_holdout_metrics,
            ),
            (
                self.household_dataset,
                self.household_training_metrics,
                self.household_holdout_metrics,
            ),
        ):
            if (
                train.observation_count + holdout.observation_count
                != binding.observation_count
            ):
                raise ValueError(
                    "dataset binding observation count disagrees with metrics"
                )
            if (
                train.evidence_unit_count + holdout.evidence_unit_count
                != binding.evidence_unit_count
            ):
                raise ValueError(
                    "dataset binding evidence-unit count disagrees with metrics"
                )
            if (
                train.encounter_count + holdout.encounter_count
                != binding.encounter_count
            ):
                raise ValueError(
                    "dataset binding encounter count disagrees with metrics"
                )

        monotonic_fields = (
            "appearance_floor",
            "quality_floor",
            "calibrated_confidence_floor",
            "ambiguity_margin_floor",
            "unknown_utility",
        )
        for field in monotonic_fields:
            if float(getattr(self.policy, field)) < float(
                getattr(self.benchmark_policy, field)
            ):
                raise ValueError(
                    "household policy may not lower a benchmark rejection gate"
                )
        invariant_fields = (
            "calibration_slope",
            "calibration_midpoint",
            "resident_prior_bonus",
            "resident_prior_cap",
        )
        for field in invariant_fields:
            if (
                abs(
                    float(getattr(self.policy, field))
                    - float(getattr(self.benchmark_policy, field))
                )
                > 1.0e-12
            ):
                raise ValueError(
                    "household policy may not change benchmark calibration or prior"
                )

        minima = self.acceptance
        benchmark_required_counts = (
            (
                self.benchmark_training_metrics,
                minima.minimum_benchmark_train_known_persons,
                minima.minimum_benchmark_train_unknown_persons,
                "benchmark training",
            ),
            (
                self.benchmark_holdout_metrics,
                minima.minimum_benchmark_holdout_known_persons,
                minima.minimum_benchmark_holdout_unknown_persons,
                "benchmark holdout",
            ),
        )
        for metrics, known_minimum, unknown_minimum, label in benchmark_required_counts:
            if (
                metrics.known_challenge_person_count < known_minimum
                or metrics.unknown_challenge_person_count < unknown_minimum
            ):
                raise ValueError(
                    f"{label} evidence is below the challenge-person minimum"
                )
            if (
                metrics.known_challenge_person_count != metrics.known_person_count
                or metrics.unknown_challenge_person_count
                != metrics.unknown_person_count
            ):
                raise ValueError(f"{label} must challenge-cover every truth person")

        household_required_counts = (
            (
                self.household_training_metrics,
                minima.minimum_household_train_known_encounters,
                minima.minimum_household_train_unknown_encounters,
                "household training",
            ),
            (
                self.household_holdout_metrics,
                minima.minimum_household_holdout_known_encounters,
                minima.minimum_household_holdout_unknown_encounters,
                "household holdout",
            ),
        )
        for metrics, known_minimum, unknown_minimum, label in household_required_counts:
            if (
                metrics.known_encounter_count < known_minimum
                or metrics.unknown_encounter_count < unknown_minimum
            ):
                raise ValueError(f"{label} evidence is below the encounter minimum")

        if (
            self.fit.selected_encounter_count
            != self.benchmark_training_metrics.encounter_count
            or self.fit.selected_person_count
            != self.benchmark_training_metrics.person_count
            or self.fit.selected_person_count > self.fit.selected_encounter_count
            or self.fit.selected_evidence_unit_count < self.fit.selected_encounter_count
            or self.fit.selected_evidence_unit_count
            > self.benchmark_training_metrics.evidence_unit_count
            or self.fit.selected_evidence_unit_count
            > (
                self.fit.selected_encounter_count
                * self.evidence_unit_policy.maximum_fit_units_per_encounter
            )
        ):
            raise ValueError(
                "fit summary is inconsistent with benchmark training units"
            )

        benchmark_holdout = self.benchmark_holdout_metrics
        if (
            benchmark_holdout.far_upper_confidence_bound
            > minima.max_benchmark_holdout_far_upper_confidence_bound
        ):
            raise ValueError("benchmark holdout FAR upper bound exceeds acceptance")
        if benchmark_holdout.person_frr > minima.max_benchmark_holdout_frr:
            raise ValueError("benchmark holdout FRR exceeds acceptance")
        if (
            benchmark_holdout.misidentification_upper_confidence_bound
            > minima.max_benchmark_holdout_misidentification_upper_confidence_bound
        ):
            raise ValueError(
                "benchmark holdout misidentification upper bound exceeds acceptance"
            )

        # Training is policy selection evidence, not a large-N confidence claim.
        benchmark_train = self.benchmark_training_metrics
        if (
            benchmark_train.person_far
            > minima.max_benchmark_holdout_far_upper_confidence_bound
        ):
            raise ValueError("benchmark training FAR exceeds the product point gate")
        if benchmark_train.person_frr > minima.max_benchmark_holdout_frr:
            raise ValueError("benchmark training FRR exceeds acceptance")
        if (
            benchmark_train.person_misidentification_rate
            > minima.max_benchmark_holdout_misidentification_upper_confidence_bound
        ):
            raise ValueError(
                "benchmark training misidentification exceeds the product point gate"
            )

        for label, metrics in (
            ("household training", self.household_training_metrics),
            ("household holdout", self.household_holdout_metrics),
        ):
            if (
                metrics.false_accept_encounter_count
                > minima.max_household_false_accept_encounters
            ):
                raise ValueError(f"{label} contains a false accept")
            if (
                metrics.misidentification_encounter_count
                > minima.max_household_misidentification_encounters
            ):
                raise ValueError(f"{label} contains a misidentification")
            if metrics.frr > minima.max_household_holdout_frr:
                raise ValueError(f"{label} FRR exceeds acceptance")

        for label, metrics in (
            ("benchmark training", self.benchmark_training_metrics),
            ("benchmark holdout", self.benchmark_holdout_metrics),
            ("household training", self.household_training_metrics),
            ("household holdout", self.household_holdout_metrics),
        ):
            if (
                metrics.resident_prior_harmful_encounter_count
                > minima.max_harmful_prior_encounters
            ):
                raise ValueError(f"{label} has a harmful resident-prior change")
            if (
                metrics.resident_prior_rejection_rescue_encounter_count
                > minima.max_prior_rejection_rescue_encounters
            ):
                raise ValueError(f"{label} has a resident-prior rejection rescue")
        return self


__all__ = [
    "CalibrationAcceptance",
    "CalibrationCandidateScore",
    "CalibrationDatasetBinding",
    "CalibrationDatasetProvenance",
    "CalibrationEvidenceSource",
    "CalibrationEvidenceStratum",
    "CalibrationEvidenceUnit",
    "CalibrationEvidenceUnitPolicy",
    "CalibrationFitSummary",
    "CalibrationGalleryLimits",
    "CalibrationMetrics",
    "CalibrationPartition",
    "CalibrationSplitStrategy",
    "CalibrationTruthKind",
    "CalibrationTruthLabel",
    "IdentityCalibrationDataset",
    "IdentityAuthorityCutoverArtifact",
    "IdentityAuthorityGateEvidence",
    "IdentityEvidenceLabelSet",
    "IdentityEvidenceChainCheckpoint",
    "IdentityOpenSetCalibrationArtifact",
    "IdentityOpenSetPolicyContract",
    "LabeledCalibrationSample",
    "ShadowIdentityEvidenceRecord",
    "calibration_evidence_unit_id",
    "one_sided_binomial_upper_confidence_bound",
    "IDENTITY_CALIBRATION_CONFIDENCE_LEVEL",
    "IDENTITY_CALIBRATION_CONFIDENCE_METHOD",
    "IDENTITY_CALIBRATION_EVIDENCE_UNIT_BUCKET_US",
    "IDENTITY_CALIBRATION_GENERATOR",
    "IDENTITY_CALIBRATION_MAX_HARMFUL_PRIOR_CHANGES",
    "IDENTITY_CALIBRATION_MAX_HOLDOUT_FAR",
    "IDENTITY_CALIBRATION_MAX_HOLDOUT_FRR",
    "IDENTITY_CALIBRATION_MAX_HOLDOUT_MISIDENTIFICATION_RATE",
    "IDENTITY_CALIBRATION_MAX_PRIOR_REJECTION_RESCUES",
    "IDENTITY_CALIBRATION_MAXIMUM_FIT_UNITS_PER_ENCOUNTER",
    "IDENTITY_CALIBRATION_MAXIMUM_OBSERVATIONS_PER_UNIT",
    "IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_KNOWN_PERSONS",
    "IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_UNKNOWN_PERSONS",
    "IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_KNOWN_PERSONS",
    "IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_UNKNOWN_PERSONS",
    "IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_CANDIDATES",
    "IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_IMPOSTORS",
    "IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_KNOWN_ENCOUNTERS",
    "IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_UNKNOWN_ENCOUNTERS",
    "IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_KNOWN_ENCOUNTERS",
    "IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_UNKNOWN_ENCOUNTERS",
]
