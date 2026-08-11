"""Deterministic, leakage-resistant calibration of identity-v2 open-set scores."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from scipy.optimize import minimize

from noesis_core.contracts.identity_calibration import (
    CalibrationAcceptance,
    CalibrationDatasetBinding,
    CalibrationEvidenceStratum,
    CalibrationEvidenceUnit,
    CalibrationEvidenceUnitPolicy,
    CalibrationFitSummary,
    CalibrationGalleryLimits,
    CalibrationMetrics,
    CalibrationPartition,
    CalibrationSplitStrategy,
    CalibrationTruthKind,
    IdentityCalibrationDataset,
    IdentityEvidenceChainCheckpoint,
    IdentityEvidenceLabelSet,
    IdentityOpenSetCalibrationArtifact,
    IdentityOpenSetPolicyContract,
    calibration_evidence_unit_id,
    one_sided_binomial_upper_confidence_bound,
    IDENTITY_CALIBRATION_CONFIDENCE_LEVEL,
    IDENTITY_CALIBRATION_CONFIDENCE_METHOD,
    IDENTITY_CALIBRATION_EVIDENCE_UNIT_BUCKET_US,
    IDENTITY_CALIBRATION_GENERATOR,
    IDENTITY_CALIBRATION_MAX_HARMFUL_PRIOR_CHANGES,
    IDENTITY_CALIBRATION_MAX_HOLDOUT_FAR,
    IDENTITY_CALIBRATION_MAX_HOLDOUT_FRR,
    IDENTITY_CALIBRATION_MAX_HOLDOUT_MISIDENTIFICATION_RATE,
    IDENTITY_CALIBRATION_MAX_PRIOR_REJECTION_RESCUES,
    IDENTITY_CALIBRATION_MAXIMUM_FIT_UNITS_PER_ENCOUNTER,
    IDENTITY_CALIBRATION_MAXIMUM_OBSERVATIONS_PER_UNIT,
    IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_KNOWN_PERSONS,
    IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_UNKNOWN_PERSONS,
    IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_KNOWN_PERSONS,
    IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_UNKNOWN_PERSONS,
    IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_CANDIDATES,
    IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_IMPOSTORS,
    IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_KNOWN_ENCOUNTERS,
    IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_UNKNOWN_ENCOUNTERS,
    IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_KNOWN_ENCOUNTERS,
    IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_UNKNOWN_ENCOUNTERS,
    LabeledCalibrationSample,
    ShadowIdentityEvidenceRecord,
)
from noesis_core.private_paths import (
    PrivatePathError,
    atomic_write_private_file,
    read_private_file,
    validate_private_file,
)
from noesis_core.strict_json import strict_json_loads

from .models import CandidateEvidence, IdentityKind, TrackletObservation
from .scoring import OpenSetPolicy, OpenSetScorer
from .evidence import load_evidence_checkpoint, validate_evidence_checkpoint

MAXIMUM_EVIDENCE_RECORDS = 100_000
MAXIMUM_EVIDENCE_BYTES = 256 * 1024 * 1024
MAXIMUM_EVIDENCE_RECORD_BYTES = 1024 * 1024


class IdentityCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceValidationSummary:
    event_count: int
    session_count: int
    candidate_pair_count: int
    model_sha256: str
    model_semantic_profile_sha256: str
    model_layer: str
    embedding_dim: int
    evidence_sha256: str
    evidence_chain: IdentityEvidenceChainCheckpoint


def canonical_json_bytes(payload: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _private_source(path: str | Path, *, label: str) -> Path:
    try:
        return validate_private_file(Path(path).expanduser(), label=label)
    except PrivatePathError as exc:
        raise IdentityCalibrationError(str(exc)) from exc


def sha256_file(path: str | Path) -> str:
    source = _private_source(path, label="identity calibration input")
    try:
        payload = read_private_file(
            source,
            label="identity calibration input",
            max_bytes=MAXIMUM_EVIDENCE_BYTES,
        )
    except PrivatePathError as exc:
        raise IdentityCalibrationError(str(exc)) from exc
    return hashlib.sha256(payload).hexdigest()


def _expected_event_id(record: ShadowIdentityEvidenceRecord) -> str:
    body = record.model_dump(mode="json")
    body.pop("event_id", None)
    return hashlib.sha256(
        b"noesis-identity-shadow-evidence-v2\0" + canonical_json_bytes(body)
    ).hexdigest()


def load_and_validate_evidence(
    path: str | Path,
) -> Tuple[Tuple[ShadowIdentityEvidenceRecord, ...], EvidenceValidationSummary]:
    source = _private_source(path, label="identity evidence JSONL")
    if not source.is_file() or source.stat().st_size <= 0:
        raise IdentityCalibrationError(f"evidence JSONL is missing or empty: {source}")
    if source.stat().st_size > MAXIMUM_EVIDENCE_BYTES:
        raise IdentityCalibrationError(
            "evidence JSONL exceeds the calibrated byte bound "
            f"({source.stat().st_size} > {MAXIMUM_EVIDENCE_BYTES})"
        )
    try:
        evidence_bytes = read_private_file(
            source,
            label="identity evidence JSONL",
            max_bytes=MAXIMUM_EVIDENCE_BYTES,
        )
    except PrivatePathError as exc:
        raise IdentityCalibrationError(str(exc)) from exc
    rows = []
    try:
        checkpoint = load_evidence_checkpoint(source)
    except Exception as exc:
        raise IdentityCalibrationError(str(exc)) from exc
    if checkpoint.retained_count <= 0 or checkpoint.first_sequence is None:
        raise IdentityCalibrationError("evidence chain checkpoint contains no records")
    if checkpoint.retained_bytes != len(evidence_bytes):
        raise IdentityCalibrationError(
            "evidence JSONL byte count disagrees with its chain checkpoint"
        )
    seen_events: set[str] = set()
    seen_observations: set[tuple[object, ...]] = set()
    last_observed_at_us: int | None = None
    expected_sequence = checkpoint.first_sequence
    expected_previous = checkpoint.previous_event_id
    with io.BytesIO(evidence_bytes) as stream:
        line_number = 0
        while True:
            line = stream.readline(MAXIMUM_EVIDENCE_RECORD_BYTES + 1)
            if not line:
                break
            line_number += 1
            if len(line) > MAXIMUM_EVIDENCE_RECORD_BYTES:
                raise IdentityCalibrationError(
                    f"evidence record exceeds the byte bound at line {line_number}"
                )
            if not line.endswith(b"\n"):
                raise IdentityCalibrationError(
                    f"evidence JSONL has an incomplete final record at line {line_number}"
                )
            if not line.strip():
                raise IdentityCalibrationError(
                    f"evidence JSONL contains a blank line at {line_number}"
                )
            try:
                row = ShadowIdentityEvidenceRecord.model_validate(
                    strict_json_loads(
                        line,
                        label=f"identity evidence line {line_number}",
                    )
                )
            except Exception as exc:
                raise IdentityCalibrationError(
                    f"invalid evidence record at line {line_number}: {exc}"
                ) from exc
            if line != canonical_json_bytes(row.model_dump(mode="json")) + b"\n":
                raise IdentityCalibrationError(
                    f"evidence record is not canonical JSON at line {line_number}"
                )
            if row.event_id != _expected_event_id(row):
                raise IdentityCalibrationError(
                    f"evidence event digest mismatch at line {line_number}"
                )
            if row.event_id in seen_events:
                raise IdentityCalibrationError(
                    f"duplicate evidence event ID at line {line_number}"
                )
            if row.sequence != expected_sequence:
                raise IdentityCalibrationError(
                    f"evidence sequence gap at line {line_number}"
                )
            if row.previous_event_id != expected_previous:
                raise IdentityCalibrationError(
                    f"evidence hash-chain mismatch at line {line_number}"
                )
            observation_key = (
                row.run_id,
                row.camera_id,
                row.tracker_id,
                row.frame_id,
                row.observation_id,
            )
            if observation_key in seen_observations:
                raise IdentityCalibrationError(
                    f"duplicate exact observation at line {line_number}"
                )
            if (
                last_observed_at_us is not None
                and row.observed_at_us < last_observed_at_us
            ):
                raise IdentityCalibrationError(
                    f"evidence timestamps move backwards at line {line_number}"
                )
            seen_events.add(row.event_id)
            seen_observations.add(observation_key)
            rows.append(row)
            expected_sequence += 1
            expected_previous = row.event_id
            last_observed_at_us = row.observed_at_us
            if len(rows) > MAXIMUM_EVIDENCE_RECORDS:
                raise IdentityCalibrationError(
                    "evidence JSONL exceeds the calibrated record bound "
                    f"({len(rows)} > {MAXIMUM_EVIDENCE_RECORDS})"
                )
    if (
        len(rows) != checkpoint.retained_count
        or expected_sequence != checkpoint.next_sequence
        or expected_previous != checkpoint.tail_event_id
    ):
        raise IdentityCalibrationError(
            "evidence JSONL endpoints disagree with its chain checkpoint"
        )
    profiles = {
        (
            row.model_sha256,
            row.model_semantic_profile_sha256,
            row.model_layer,
            row.embedding_dim,
        )
        for row in rows
    }
    if len(profiles) != 1:
        raise IdentityCalibrationError("evidence JSONL mixes model profiles")
    (
        model_sha256,
        model_semantic_profile_sha256,
        model_layer,
        embedding_dim,
    ) = next(iter(profiles))
    ordered = tuple(sorted(rows, key=lambda row: row.event_id))
    return ordered, EvidenceValidationSummary(
        event_count=len(ordered),
        session_count=len({row.session_id for row in ordered}),
        candidate_pair_count=sum(len(row.candidates) for row in ordered),
        model_sha256=model_sha256,
        model_semantic_profile_sha256=model_semantic_profile_sha256,
        model_layer=model_layer,
        embedding_dim=embedding_dim,
        evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        evidence_chain=checkpoint,
    )


def _load_label_set_snapshot(
    path: str | Path,
) -> tuple[IdentityEvidenceLabelSet, str]:
    source = _private_source(path, label="identity evidence label set")
    if not source.is_file() or source.stat().st_size <= 0:
        raise IdentityCalibrationError(f"label set is missing or empty: {source}")
    try:
        payload = read_private_file(
            source,
            label="identity evidence label set",
            max_bytes=MAXIMUM_EVIDENCE_BYTES,
        )
        return (
            IdentityEvidenceLabelSet.model_validate(
                strict_json_loads(payload, label="identity evidence label set")
            ),
            hashlib.sha256(payload).hexdigest(),
        )
    except Exception as exc:
        raise IdentityCalibrationError(f"invalid evidence label set: {exc}") from exc


def load_label_set(path: str | Path) -> IdentityEvidenceLabelSet:
    return _load_label_set_snapshot(path)[0]


def _partition_rows(
    samples: Sequence[LabeledCalibrationSample],
    partition: CalibrationPartition,
) -> Tuple[LabeledCalibrationSample, ...]:
    return tuple(row for row in samples if row.label.partition is partition)


def _partition_units(
    units: Sequence[CalibrationEvidenceUnit],
    partition: CalibrationPartition,
) -> Tuple[CalibrationEvidenceUnit, ...]:
    return tuple(row for row in units if row.partition is partition)


def _unit_policy() -> CalibrationEvidenceUnitPolicy:
    return CalibrationEvidenceUnitPolicy(
        algorithm="session_tracklet_time_bucket_v1",
        bucket_duration_us=IDENTITY_CALIBRATION_EVIDENCE_UNIT_BUCKET_US,
        maximum_observations_per_unit=(
            IDENTITY_CALIBRATION_MAXIMUM_OBSERVATIONS_PER_UNIT
        ),
        fit_thinning="temporal_cover_center_representative_v1",
        maximum_fit_units_per_encounter=(
            IDENTITY_CALIBRATION_MAXIMUM_FIT_UNITS_PER_ENCOUNTER
        ),
        fit_representatives_per_unit=1,
        authority_aggregation="encounter_worst_case_v1",
    )


def _temporal_cover_indexes(count: int, maximum: int) -> tuple[int, ...]:
    if count <= maximum:
        return tuple(range(count))
    if maximum <= 1:
        return (count // 2,)
    return tuple((index * (count - 1)) // (maximum - 1) for index in range(maximum))


@dataclass(frozen=True)
class _EvidenceUnitDraft:
    unit_id: str
    partition: CalibrationPartition
    truth_kind: CalibrationTruthKind
    truth_subject_id: str | None
    truth_person_key: str
    encounter_id: str
    session_id: str
    run_id: str
    source: object
    camera_id: str
    tracker_id: str
    bucket_index: int
    samples: tuple[LabeledCalibrationSample, ...]


def _build_evidence_units(
    samples: Sequence[LabeledCalibrationSample],
    policy: CalibrationEvidenceUnitPolicy,
) -> tuple[CalibrationEvidenceUnit, ...]:
    grouped: dict[tuple[object, ...], list[LabeledCalibrationSample]] = {}
    for sample in samples:
        evidence = sample.evidence
        bucket_index = evidence.observed_at_us // policy.bucket_duration_us
        key = (
            evidence.session_id,
            evidence.run_id,
            evidence.source,
            evidence.camera_id,
            evidence.tracker_id,
            bucket_index,
        )
        grouped.setdefault(key, []).append(sample)

    drafts: list[_EvidenceUnitDraft] = []
    for key, rows in sorted(
        grouped.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        session_id, run_id, source, camera_id, tracker_id, bucket_index = key
        ordered = tuple(
            sorted(rows, key=lambda row: (row.evidence.sequence, row.evidence.event_id))
        )
        if len(ordered) > policy.maximum_observations_per_unit:
            raise IdentityCalibrationError(
                "deterministic evidence unit exceeds the observation cap: "
                f"session={session_id} camera={camera_id} tracker={tracker_id} "
                f"bucket={bucket_index} count={len(ordered)} "
                f"cap={policy.maximum_observations_per_unit}"
            )
        first = ordered[0]
        expected_label = (
            first.label.partition,
            first.label.truth_kind,
            first.label.truth_subject_id,
            first.label.truth_person_key,
            first.label.encounter_id,
        )
        if any(
            (
                row.label.partition,
                row.label.truth_kind,
                row.label.truth_subject_id,
                row.label.truth_person_key,
                row.label.encounter_id,
            )
            != expected_label
            for row in ordered[1:]
        ):
            raise IdentityCalibrationError(
                "one deterministic session/tracklet/time evidence unit has conflicting "
                "partition, person, truth, or encounter labels"
            )
        drafts.append(
            _EvidenceUnitDraft(
                unit_id=calibration_evidence_unit_id(
                    session_id=str(session_id),
                    run_id=str(run_id),
                    source=source,
                    camera_id=str(camera_id),
                    tracker_id=str(tracker_id),
                    bucket_index=int(bucket_index),
                ),
                partition=first.label.partition,
                truth_kind=first.label.truth_kind,
                truth_subject_id=first.label.truth_subject_id,
                truth_person_key=first.label.truth_person_key,
                encounter_id=first.label.encounter_id,
                session_id=str(session_id),
                run_id=str(run_id),
                source=source,
                camera_id=str(camera_id),
                tracker_id=str(tracker_id),
                bucket_index=int(bucket_index),
                samples=ordered,
            )
        )

    drafts_by_encounter: dict[str, list[_EvidenceUnitDraft]] = {}
    for draft in drafts:
        drafts_by_encounter.setdefault(draft.encounter_id, []).append(draft)
    fit_unit_ids: set[str] = set()
    for encounter_drafts in drafts_by_encounter.values():
        ordered_drafts = sorted(
            encounter_drafts, key=lambda row: (row.bucket_index, row.unit_id)
        )
        if ordered_drafts[0].partition is not CalibrationPartition.TRAIN:
            continue
        for index in _temporal_cover_indexes(
            len(ordered_drafts), policy.maximum_fit_units_per_encounter
        ):
            fit_unit_ids.add(ordered_drafts[index].unit_id)

    units = []
    for draft in drafts:
        fit_event_id = None
        if draft.unit_id in fit_unit_ids:
            midpoint = (
                draft.bucket_index * policy.bucket_duration_us
                + policy.bucket_duration_us // 2
            )
            fit_event_id = min(
                draft.samples,
                key=lambda row: (
                    abs(row.evidence.observed_at_us - midpoint),
                    row.evidence.observed_at_us,
                    row.evidence.event_id,
                ),
            ).evidence.event_id
        units.append(
            CalibrationEvidenceUnit(
                unit_id=draft.unit_id,
                partition=draft.partition,
                truth_kind=draft.truth_kind,
                truth_subject_id=draft.truth_subject_id,
                truth_person_key=draft.truth_person_key,
                encounter_id=draft.encounter_id,
                session_id=draft.session_id,
                run_id=draft.run_id,
                source=draft.source,
                camera_id=draft.camera_id,
                tracker_id=draft.tracker_id,
                bucket_index=draft.bucket_index,
                event_ids=tuple(row.evidence.event_id for row in draft.samples),
                fit_event_id=fit_event_id,
            )
        )
    return tuple(
        sorted(
            units,
            key=lambda row: (
                row.partition.value,
                row.encounter_id,
                row.bucket_index,
                row.unit_id,
            ),
        )
    )


def _validate_disjointness(dataset: IdentityCalibrationDataset) -> None:
    try:
        validate_evidence_checkpoint(dataset.evidence_chain)
    except Exception as exc:
        raise IdentityCalibrationError(str(exc)) from exc
    for sample in dataset.samples:
        if sample.evidence.event_id != _expected_event_id(sample.evidence):
            raise IdentityCalibrationError(
                f"dataset evidence event digest mismatch: {sample.evidence.event_id}"
            )
    train = _partition_rows(dataset.samples, CalibrationPartition.TRAIN)
    holdout = _partition_rows(dataset.samples, CalibrationPartition.HOLDOUT)
    train_encounters = {row.label.encounter_id for row in train}
    holdout_encounters = {row.label.encounter_id for row in holdout}
    leaked_encounters = train_encounters & holdout_encounters
    if leaked_encounters:
        raise IdentityCalibrationError(
            "encounters cross train/holdout: " + ", ".join(sorted(leaked_encounters))
        )
    train_sessions = {row.evidence.session_id for row in train}
    holdout_sessions = {row.evidence.session_id for row in holdout}
    if train_sessions & holdout_sessions:
        raise IdentityCalibrationError("capture sessions cross train/holdout")
    train_runs = {row.evidence.run_id for row in train}
    holdout_runs = {row.evidence.run_id for row in holdout}
    if train_runs & holdout_runs:
        raise IdentityCalibrationError("runtime runs cross train/holdout")
    if dataset.split_strategy is CalibrationSplitStrategy.SUBJECT_DISJOINT:
        train_people = {row.label.truth_person_key for row in train}
        holdout_people = {row.label.truth_person_key for row in holdout}
        leaked_people = train_people & holdout_people
        if leaked_people:
            raise IdentityCalibrationError(
                "truth people cross a subject-disjoint train/holdout: "
                + ", ".join(sorted(leaked_people))
            )
    if dataset.evidence_stratum is CalibrationEvidenceStratum.HOUSEHOLD:
        train_unknown_people = {
            row.label.truth_person_key
            for row in train
            if row.label.truth_kind is CalibrationTruthKind.UNKNOWN
        }
        holdout_unknown_people = {
            row.label.truth_person_key
            for row in holdout
            if row.label.truth_kind is CalibrationTruthKind.UNKNOWN
        }
        if train_unknown_people & holdout_unknown_people:
            raise IdentityCalibrationError(
                "household unknown truth people cross train/holdout"
            )


def _encounter_counts(
    dataset: IdentityCalibrationDataset,
    partition: CalibrationPartition,
) -> tuple[int, int]:
    units = _partition_units(dataset.evidence_units, partition)
    known = {
        row.encounter_id
        for row in units
        if row.truth_kind is not CalibrationTruthKind.UNKNOWN
    }
    unknown = {
        row.encounter_id
        for row in units
        if row.truth_kind is CalibrationTruthKind.UNKNOWN
    }
    return len(known), len(unknown)


def _challenge_eligible(
    sample: LabeledCalibrationSample,
    *,
    require_truth: bool,
) -> bool:
    allowed = tuple(
        row
        for row in sample.evidence.candidates
        if row.hard_allowed and row.gallery_exemplar_count >= 1
    )
    if len(allowed) < IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_CANDIDATES:
        return False
    if not any(row.identity_kind == "resident" for row in allowed):
        return False
    if not any(row.identity_kind == "visitor" for row in allowed):
        return False
    truth = sample.label.truth_subject_id
    if truth is None:
        return True
    impostors = tuple(row for row in allowed if row.subject_id != truth)
    if len(impostors) < IDENTITY_CALIBRATION_MINIMUM_HARD_ALLOWED_IMPOSTORS:
        return False
    if require_truth and not any(row.subject_id == truth for row in allowed):
        return False
    return True


def _challenge_person_counts(
    dataset: IdentityCalibrationDataset,
    partition: CalibrationPartition,
) -> tuple[int, int]:
    challenged: dict[str, CalibrationTruthKind] = {}
    for sample in _partition_rows(dataset.samples, partition):
        if _challenge_eligible(
            sample,
            require_truth=partition is CalibrationPartition.TRAIN,
        ):
            challenged[sample.label.truth_person_key] = sample.label.truth_kind
    known = sum(
        kind is not CalibrationTruthKind.UNKNOWN for kind in challenged.values()
    )
    return known, len(challenged) - known


def _validate_challenge_coverage(dataset: IdentityCalibrationDataset) -> None:
    samples_by_encounter: dict[str, list[LabeledCalibrationSample]] = {}
    for sample in dataset.samples:
        samples_by_encounter.setdefault(sample.label.encounter_id, []).append(sample)
    for encounter_id, samples in samples_by_encounter.items():
        partition = samples[0].label.partition
        if not any(
            _challenge_eligible(
                sample,
                require_truth=partition is CalibrationPartition.TRAIN,
            )
            for sample in samples
        ):
            raise IdentityCalibrationError(
                f"encounter {encounter_id} lacks deployment-representative "
                "hard-allowed gallery challenge coverage"
            )

    sample_by_id = {row.evidence.event_id: row for row in dataset.samples}
    for unit in dataset.evidence_units:
        if unit.fit_event_id is None:
            continue
        if not _challenge_eligible(sample_by_id[unit.fit_event_id], require_truth=True):
            raise IdentityCalibrationError(
                "deterministic fit representative lacks genuine/impostor gallery coverage"
            )


def _validate_sufficiency(dataset: IdentityCalibrationDataset) -> None:
    if (
        dataset.evidence_stratum is CalibrationEvidenceStratum.BENCHMARK
        and dataset.split_strategy is not CalibrationSplitStrategy.SUBJECT_DISJOINT
    ):
        raise IdentityCalibrationError(
            "benchmark authority requires a subject-disjoint train/holdout"
        )
    if dataset.evidence_stratum is CalibrationEvidenceStratum.BENCHMARK and {
        (
            row.evidence.source.value,
            row.evidence.runtime,
            row.evidence.runtime_mode,
        )
        for row in dataset.samples
    } != {("replay", "replay", "shadow")}:
        raise IdentityCalibrationError(
            "benchmark authority requires provenance-locked shadow replay evidence"
        )
    if any(
        row.label.truth_kind is CalibrationTruthKind.VISITOR for row in dataset.samples
    ):
        raise IdentityCalibrationError(
            f"{dataset.evidence_stratum.value} known-person authority requires "
            "resident truth; visitor continuity is outside the calibration recall claim"
        )
    required_population_basis = (
        "licensed_real_people"
        if dataset.evidence_stratum is CalibrationEvidenceStratum.BENCHMARK
        else "owner_consented_household_people"
    )
    if dataset.provenance.population_basis != required_population_basis:
        raise IdentityCalibrationError(
            f"{dataset.evidence_stratum.value} authority requires "
            f"population_basis={required_population_basis}"
        )
    if dataset.evidence_stratum is CalibrationEvidenceStratum.BENCHMARK:
        minima = {
            CalibrationPartition.TRAIN: (
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_KNOWN_PERSONS,
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_UNKNOWN_PERSONS,
            ),
            CalibrationPartition.HOLDOUT: (
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_KNOWN_PERSONS,
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_UNKNOWN_PERSONS,
            ),
        }
    else:
        minima = {
            CalibrationPartition.TRAIN: (
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_KNOWN_ENCOUNTERS,
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_UNKNOWN_ENCOUNTERS,
            ),
            CalibrationPartition.HOLDOUT: (
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_KNOWN_ENCOUNTERS,
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_UNKNOWN_ENCOUNTERS,
            ),
        }
    for partition in (CalibrationPartition.TRAIN, CalibrationPartition.HOLDOUT):
        if dataset.evidence_stratum is CalibrationEvidenceStratum.BENCHMARK:
            known, unknown = _challenge_person_counts(dataset, partition)
            unit_label = "challenge-covered people"
        else:
            known, unknown = _encounter_counts(dataset, partition)
            unit_label = "independent encounters"
        minimum_known, minimum_unknown = minima[partition]
        if known < minimum_known or unknown < minimum_unknown:
            raise IdentityCalibrationError(
                f"{dataset.evidence_stratum.value} {partition.value} requires at least "
                f"{minimum_known} known and {minimum_unknown} unknown {unit_label}; "
                f"got {known} and {unknown}"
            )
    _validate_challenge_coverage(dataset)

    if dataset.evidence_stratum is CalibrationEvidenceStratum.BENCHMARK:
        session_people: dict[str, set[str]] = {}
        for sample in dataset.samples:
            session_people.setdefault(sample.evidence.session_id, set()).add(
                sample.label.truth_person_key
            )
        if any(len(people) != 1 for people in session_people.values()):
            raise IdentityCalibrationError(
                "benchmark capture sessions must each contain exactly one truth person"
            )


def build_calibration_dataset(
    evidence_path: str | Path,
    labels_path: str | Path,
) -> IdentityCalibrationDataset:
    evidence, summary = load_and_validate_evidence(evidence_path)
    label_set, labels_sha256 = _load_label_set_snapshot(labels_path)
    if label_set.evidence_sha256 != summary.evidence_sha256:
        raise IdentityCalibrationError(
            "label set evidence_sha256 does not match the exact evidence file"
        )
    if (
        label_set.provenance.model_semantic_profile_sha256
        != summary.model_semantic_profile_sha256
    ):
        raise IdentityCalibrationError(
            "label-set semantic profile does not match the profile recorded by "
            "the active evidence runtime"
        )
    evidence_by_id = {row.event_id: row for row in evidence}
    labels_by_id = {row.event_id: row for row in label_set.labels}
    missing = sorted(set(evidence_by_id) - set(labels_by_id))
    orphaned = sorted(set(labels_by_id) - set(evidence_by_id))
    if missing or orphaned:
        raise IdentityCalibrationError(
            "every evidence event requires exactly one label "
            f"(missing={len(missing)}, orphaned={len(orphaned)})"
        )
    samples = tuple(
        LabeledCalibrationSample(
            evidence=evidence_by_id[event_id],
            label=labels_by_id[event_id],
        )
        for event_id in sorted(
            evidence_by_id,
            key=lambda value: evidence_by_id[value].sequence,
        )
    )
    policy = _unit_policy()
    units = _build_evidence_units(samples, policy)
    try:
        dataset = IdentityCalibrationDataset(
            contract="noesis.identity.calibration_dataset",
            contract_version=2,
            evidence_sha256=summary.evidence_sha256,
            labels_sha256=labels_sha256,
            evidence_stratum=label_set.evidence_stratum,
            split_strategy=label_set.split_strategy,
            provenance=label_set.provenance,
            labeled_by=label_set.labeled_by,
            labeling_revision=label_set.labeling_revision,
            model_sha256=summary.model_sha256,
            model_layer=summary.model_layer,
            embedding_dim=summary.embedding_dim,
            evidence_chain=summary.evidence_chain,
            evidence_unit_policy=policy,
            samples=samples,
            evidence_units=units,
        )
    except Exception as exc:
        raise IdentityCalibrationError(
            f"invalid correlation-aware calibration dataset: {exc}"
        ) from exc
    _validate_disjointness(dataset)
    _validate_sufficiency(dataset)
    return dataset


def write_private_json(path: str | Path, payload: object) -> Path:
    body = canonical_json_bytes(payload, pretty=True)
    try:
        return atomic_write_private_file(
            Path(path).expanduser(),
            body,
            label="identity calibration JSON",
        )
    except PrivatePathError as exc:
        raise IdentityCalibrationError(str(exc)) from exc


def _load_calibration_dataset_snapshot(
    path: str | Path,
) -> tuple[IdentityCalibrationDataset, str]:
    source = _private_source(path, label="identity calibration dataset")
    try:
        payload = read_private_file(
            source,
            label="identity calibration dataset",
            max_bytes=MAXIMUM_EVIDENCE_BYTES,
        )
        dataset = IdentityCalibrationDataset.model_validate(
            strict_json_loads(payload, label="identity calibration dataset")
        )
    except Exception as exc:
        raise IdentityCalibrationError(f"invalid calibration dataset: {exc}") from exc
    _validate_disjointness(dataset)
    _validate_sufficiency(dataset)
    return dataset, hashlib.sha256(payload).hexdigest()


def load_calibration_dataset(path: str | Path) -> IdentityCalibrationDataset:
    return _load_calibration_dataset_snapshot(path)[0]


def dataset_review(dataset: IdentityCalibrationDataset) -> dict[str, object]:
    partitions: dict[str, object] = {}
    for partition in (CalibrationPartition.TRAIN, CalibrationPartition.HOLDOUT):
        samples = _partition_rows(dataset.samples, partition)
        units = _partition_units(dataset.evidence_units, partition)
        known, unknown = _encounter_counts(dataset, partition)
        known_people = {
            row.label.truth_person_key
            for row in samples
            if row.label.truth_kind is not CalibrationTruthKind.UNKNOWN
        }
        unknown_people = {
            row.label.truth_person_key
            for row in samples
            if row.label.truth_kind is CalibrationTruthKind.UNKNOWN
        }
        challenged_known, challenged_unknown = _challenge_person_counts(
            dataset, partition
        )
        partitions[partition.value] = {
            "observation_count": len(samples),
            "evidence_unit_count": len(units),
            "encounter_count": known + unknown,
            "known_encounter_count": known,
            "unknown_encounter_count": unknown,
            "session_count": len({row.evidence.session_id for row in samples}),
            "run_count": len({row.evidence.run_id for row in samples}),
            "person_count": len(known_people) + len(unknown_people),
            "known_person_count": len(known_people),
            "unknown_person_count": len(unknown_people),
            "known_challenge_person_count": challenged_known,
            "unknown_challenge_person_count": challenged_unknown,
            "zero_error_known_upper_95": one_sided_binomial_upper_confidence_bound(
                0, challenged_known
            ),
            "zero_error_unknown_upper_95": one_sided_binomial_upper_confidence_bound(
                0, challenged_unknown
            ),
        }
    return {
        "contract_version": dataset.contract_version,
        "evidence_stratum": dataset.evidence_stratum.value,
        "split_strategy": dataset.split_strategy.value,
        "model_sha256": dataset.model_sha256,
        "model_layer": dataset.model_layer,
        "embedding_dim": dataset.embedding_dim,
        "provenance": dataset.provenance.model_dump(mode="json"),
        "evidence_unit_policy": dataset.evidence_unit_policy.model_dump(mode="json"),
        "partitions": partitions,
    }


def _candidate_pairs(
    dataset: IdentityCalibrationDataset,
) -> Tuple[Tuple[float, int, float], ...]:
    sample_by_id = {row.evidence.event_id: row for row in dataset.samples}
    selected_by_person: dict[str, dict[str, list[LabeledCalibrationSample]]] = {}
    for unit in dataset.evidence_units:
        if (
            unit.partition is CalibrationPartition.TRAIN
            and unit.fit_event_id is not None
        ):
            sample = sample_by_id[unit.fit_event_id]
            selected_by_person.setdefault(sample.label.truth_person_key, {}).setdefault(
                unit.encounter_id, []
            ).append(sample)

    pairs: list[tuple[float, int, float]] = []
    for person_key in sorted(selected_by_person):
        encounters = selected_by_person[person_key]
        encounter_weight = 1.0 / len(encounters)
        for encounter_id in sorted(encounters):
            selected = sorted(
                encounters[encounter_id],
                key=lambda row: row.evidence.event_id,
            )
            unit_weight = encounter_weight / len(selected)
            for sample in selected:
                truth = sample.label.truth_subject_id
                positives = [
                    row
                    for row in sample.evidence.candidates
                    if row.subject_id == truth
                    and row.hard_allowed
                    and row.gallery_exemplar_count >= 1
                ]
                negatives = [
                    row
                    for row in sample.evidence.candidates
                    if row.subject_id != truth
                    and row.hard_allowed
                    and row.gallery_exemplar_count >= 1
                ]
                if positives:
                    positive_weight = unit_weight / len(positives)
                    pairs.extend(
                        (float(row.raw_similarity), 1, positive_weight)
                        for row in positives
                    )
                if negatives:
                    negative_weight = unit_weight / len(negatives)
                    pairs.extend(
                        (float(row.raw_similarity), 0, negative_weight)
                        for row in negatives
                    )
    return tuple(pairs)


def _fit_logistic(
    pairs: Sequence[Tuple[float, int, float]],
    *,
    selected_evidence_unit_count: int,
    selected_encounter_count: int,
    selected_person_count: int,
) -> tuple[float, float, CalibrationFitSummary]:
    positives = sum(label for _, label, _ in pairs)
    negatives = len(pairs) - positives
    if positives < 1 or negatives < 1:
        raise IdentityCalibrationError(
            "calibration requires both genuine and impostor candidate pairs"
        )
    positive_mass = sum(weight for _, label, weight in pairs if label)
    negative_mass = sum(weight for _, label, weight in pairs if not label)
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise IdentityCalibrationError("calibration candidate-pair weights are empty")
    weighted_pairs = tuple(
        (
            similarity,
            label,
            0.5 * base_weight / (positive_mass if label else negative_mass),
        )
        for similarity, label, base_weight in pairs
    )

    def objective(parameters: Sequence[float]) -> float:
        slope, midpoint = float(parameters[0]), float(parameters[1])
        loss = 0.0
        for similarity, label, weight in weighted_pairs:
            z = max(-60.0, min(60.0, slope * (similarity - midpoint)))
            probability = 1.0 / (1.0 + math.exp(-z))
            probability = min(1.0 - 1.0e-12, max(1.0e-12, probability))
            loss -= weight * (
                label * math.log(probability)
                + (1 - label) * math.log(1.0 - probability)
            )
        return float(loss + 1.0e-6 * slope * slope)

    positive_scores = sorted(score for score, label, _ in pairs if label)
    negative_scores = sorted(score for score, label, _ in pairs if not label)
    initial_midpoint = (
        positive_scores[len(positive_scores) // 2]
        + negative_scores[len(negative_scores) // 2]
    ) / 2.0
    result = minimize(
        objective,
        x0=(12.0, max(-1.0, min(1.0, initial_midpoint))),
        method="L-BFGS-B",
        bounds=((0.1, 200.0), (-1.0, 1.0)),
        options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 500},
    )
    if not bool(result.success):
        raise IdentityCalibrationError(
            f"deterministic logistic fit failed: {result.message}"
        )
    slope, midpoint = float(result.x[0]), float(result.x[1])
    balanced_loss = 0.0
    balanced_brier = 0.0
    for similarity, label, weight in weighted_pairs:
        z = max(-60.0, min(60.0, slope * (similarity - midpoint)))
        probability = 1.0 / (1.0 + math.exp(-z))
        bounded_probability = min(1.0 - 1.0e-12, max(1.0e-12, probability))
        balanced_loss -= weight * (
            label * math.log(bounded_probability)
            + (1 - label) * math.log(1.0 - bounded_probability)
        )
        balanced_brier += weight * ((probability - label) ** 2)
    return (
        slope,
        midpoint,
        CalibrationFitSummary(
            score_semantics="balanced_monotonic_match_score_not_posterior",
            selected_evidence_unit_count=selected_evidence_unit_count,
            selected_encounter_count=selected_encounter_count,
            selected_person_count=selected_person_count,
            positive_pair_count=positives,
            negative_pair_count=negatives,
            balanced_log_loss=float(balanced_loss),
            balanced_brier_score=float(balanced_brier),
            iterations=max(1, int(result.nit)),
        ),
    )


def _tracklet(sample: LabeledCalibrationSample) -> TrackletObservation:
    evidence = sample.evidence
    return TrackletObservation(
        tracklet_id=evidence.event_id,
        quality=evidence.quality,
        candidates=tuple(
            CandidateEvidence(
                identity_id=row.subject_id,
                identity_kind=IdentityKind(row.identity_kind),
                raw_similarity=row.raw_similarity,
                hard_allowed=row.hard_allowed,
                hard_constraint_reason=row.hard_constraint_reason,
            )
            for row in evidence.candidates
        ),
    )


def _single_decision(
    scorer: OpenSetScorer, observation: TrackletObservation
) -> tuple[str | None, str | None]:
    scored = scorer.score_tracklet(observation)
    if not scored.assignment_eligible:
        return None, scored.pre_prior_winner_id
    eligible = sorted(
        (row for row in scored.candidates if row.eligible),
        key=lambda row: (-row.adjusted_utility, row.identity_id),
    )
    if not eligible or eligible[0].adjusted_utility <= scorer.policy.unknown_utility:
        return None, scored.pre_prior_winner_id
    return eligible[0].identity_id, scored.pre_prior_winner_id


def _metrics(
    dataset: IdentityCalibrationDataset,
    partition: CalibrationPartition,
    policy: OpenSetPolicy,
) -> CalibrationMetrics:
    scorer = OpenSetScorer(policy)
    samples = _partition_rows(dataset.samples, partition)
    units = _partition_units(dataset.evidence_units, partition)
    samples_by_encounter: dict[str, list[LabeledCalibrationSample]] = {}
    for sample in samples:
        samples_by_encounter.setdefault(sample.label.encounter_id, []).append(sample)

    correct = false_reject = misidentified = false_accept = 0
    prior_changes = beneficial = harmful = rescues = 0
    encounter_outcomes: dict[str, tuple[str, CalibrationTruthKind, str, bool]] = {}
    for encounter_id in sorted(samples_by_encounter):
        encounter_samples = samples_by_encounter[encounter_id]
        truth_kind = encounter_samples[0].label.truth_kind
        truth = encounter_samples[0].label.truth_subject_id
        person_key = encounter_samples[0].label.truth_person_key
        challenge_eligible = any(
            _challenge_eligible(
                sample,
                require_truth=partition is CalibrationPartition.TRAIN,
            )
            for sample in encounter_samples
        )
        saw_correct = saw_reject = saw_wrong = False
        saw_prior_change = saw_beneficial = saw_harmful = saw_rescue = False
        for sample in encounter_samples:
            selected, pre_prior = _single_decision(scorer, _tracklet(sample))
            if truth_kind is CalibrationTruthKind.UNKNOWN:
                saw_wrong = saw_wrong or selected is not None
            elif selected is None:
                saw_reject = True
            elif selected == truth:
                saw_correct = True
            else:
                saw_wrong = True

            if selected is not None and pre_prior is None:
                saw_prior_change = True
                saw_rescue = True
                saw_harmful = True
            elif (
                selected is not None and pre_prior is not None and selected != pre_prior
            ):
                saw_prior_change = True
                if selected == truth and pre_prior != truth:
                    saw_beneficial = True
                elif selected != truth:
                    saw_harmful = True

        if truth_kind is CalibrationTruthKind.UNKNOWN:
            if saw_wrong:
                false_accept += 1
                outcome = "false_accept"
            else:
                outcome = "correct_reject"
        elif saw_wrong:
            misidentified += 1
            outcome = "misidentification"
        elif saw_reject or not saw_correct:
            false_reject += 1
            outcome = "false_reject"
        else:
            correct += 1
            outcome = "correct_accept"
        encounter_outcomes[encounter_id] = (
            person_key,
            truth_kind,
            outcome,
            challenge_eligible,
        )

        if saw_prior_change:
            prior_changes += 1
        if saw_harmful:
            harmful += 1
        elif saw_beneficial:
            beneficial += 1
        if saw_rescue:
            rescues += 1

    known_count = correct + false_reject + misidentified
    unknown_count = len(samples_by_encounter) - known_count
    outcomes_by_person: dict[str, tuple[CalibrationTruthKind, list[str], bool]] = {}
    for person_key, truth_kind, outcome, challenged in encounter_outcomes.values():
        existing = outcomes_by_person.get(person_key)
        if existing is None:
            outcomes_by_person[person_key] = (truth_kind, [outcome], challenged)
        else:
            existing[1].append(outcome)
            outcomes_by_person[person_key] = (
                existing[0],
                existing[1],
                existing[2] or challenged,
            )
    person_correct = person_false_reject = person_misidentified = 0
    person_false_accept = 0
    known_challenge_people = unknown_challenge_people = 0
    challenge_misidentified_people = challenge_false_accept_people = 0
    for truth_kind, outcomes, challenged in outcomes_by_person.values():
        if truth_kind is CalibrationTruthKind.UNKNOWN:
            if challenged:
                unknown_challenge_people += 1
            if "false_accept" in outcomes:
                person_false_accept += 1
                if challenged:
                    challenge_false_accept_people += 1
            continue
        if challenged:
            known_challenge_people += 1
        if "misidentification" in outcomes:
            person_misidentified += 1
            if challenged:
                challenge_misidentified_people += 1
        elif "false_reject" in outcomes:
            person_false_reject += 1
        else:
            person_correct += 1
    known_people = person_correct + person_false_reject + person_misidentified
    unknown_people = len(outcomes_by_person) - known_people
    known_observations = sum(
        row.label.truth_kind is not CalibrationTruthKind.UNKNOWN for row in samples
    )
    unknown_observations = len(samples) - known_observations
    known_units = sum(
        row.truth_kind is not CalibrationTruthKind.UNKNOWN for row in units
    )
    unknown_units = len(units) - known_units
    confidence = IDENTITY_CALIBRATION_CONFIDENCE_LEVEL
    challenge_observations = sum(
        _challenge_eligible(
            row,
            require_truth=partition is CalibrationPartition.TRAIN,
        )
        for row in samples
    )
    challenge_encounters = sum(
        challenged for _, _, _, challenged in encounter_outcomes.values()
    )
    return CalibrationMetrics(
        confidence_level=confidence,
        confidence_method=IDENTITY_CALIBRATION_CONFIDENCE_METHOD,
        confidence_unit="truth_person_worst_case",
        observation_count=len(samples),
        known_observation_count=known_observations,
        unknown_observation_count=unknown_observations,
        challenge_eligible_observation_count=challenge_observations,
        evidence_unit_count=len(units),
        known_evidence_unit_count=known_units,
        unknown_evidence_unit_count=unknown_units,
        encounter_count=len(samples_by_encounter),
        known_encounter_count=known_count,
        unknown_encounter_count=unknown_count,
        challenge_eligible_encounter_count=challenge_encounters,
        correct_accept_encounter_count=correct,
        false_reject_encounter_count=false_reject,
        misidentification_encounter_count=misidentified,
        false_accept_encounter_count=false_accept,
        person_count=len(outcomes_by_person),
        known_person_count=known_people,
        unknown_person_count=unknown_people,
        known_challenge_person_count=known_challenge_people,
        unknown_challenge_person_count=unknown_challenge_people,
        correct_accept_person_count=person_correct,
        false_reject_person_count=person_false_reject,
        misidentification_person_count=person_misidentified,
        false_accept_person_count=person_false_accept,
        misidentification_challenge_person_count=challenge_misidentified_people,
        false_accept_challenge_person_count=challenge_false_accept_people,
        person_far=person_false_accept / unknown_people,
        person_frr=person_false_reject / known_people,
        person_misidentification_rate=person_misidentified / known_people,
        person_identification_accuracy=person_correct / known_people,
        far=false_accept / unknown_count,
        far_upper_confidence_bound=one_sided_binomial_upper_confidence_bound(
            challenge_false_accept_people,
            unknown_challenge_people,
            confidence_level=confidence,
        ),
        frr=false_reject / known_count,
        frr_upper_confidence_bound=one_sided_binomial_upper_confidence_bound(
            person_false_reject, known_people, confidence_level=confidence
        ),
        misidentification_rate=misidentified / known_count,
        misidentification_upper_confidence_bound=(
            one_sided_binomial_upper_confidence_bound(
                challenge_misidentified_people,
                known_challenge_people,
                confidence_level=confidence,
            )
        ),
        unknown_rejection_rate=1.0 - (false_accept / unknown_count),
        identification_accuracy=correct / known_count,
        resident_prior_changed_encounter_count=prior_changes,
        resident_prior_beneficial_encounter_count=beneficial,
        resident_prior_harmful_encounter_count=harmful,
        resident_prior_rejection_rescue_encounter_count=rescues,
    )


def _bounded_breakpoints(
    values: Iterable[float],
    *,
    defaults: Sequence[float],
    lower: float,
    upper: float,
    maximum_points: int,
) -> tuple[float, ...]:
    candidates = {
        max(lower, min(upper, float(value)))
        for value in (*tuple(values), *tuple(defaults))
        if math.isfinite(float(value))
    }
    ordered = sorted(candidates)
    if len(ordered) <= maximum_points:
        return tuple(ordered)
    required = {max(lower, min(upper, float(value))) for value in defaults} | {
        ordered[0],
        ordered[-1],
    }
    slots = maximum_points - len(required)
    if slots < 0:
        raise IdentityCalibrationError("policy breakpoint defaults exceed search cap")
    selected = set(required)
    if slots:
        denominator = slots + 1
        for index in range(1, slots + 1):
            position = round(index * (len(ordered) - 1) / denominator)
            selected.add(ordered[position])
    return tuple(sorted(selected))


def _policy_breakpoints(
    dataset: IdentityCalibrationDataset,
    *,
    slope: float,
    midpoint: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    samples = _partition_rows(dataset.samples, CalibrationPartition.TRAIN)
    raw_scores = tuple(
        float(candidate.raw_similarity)
        for sample in samples
        for candidate in sample.evidence.candidates
        if candidate.hard_allowed and candidate.gallery_exemplar_count >= 1
    )
    appearance_values = tuple(
        value
        for score in raw_scores
        for value in (score, math.nextafter(score, math.inf))
    )
    quality_values = tuple(
        value
        for sample in samples
        for value in (
            float(sample.evidence.quality),
            math.nextafter(float(sample.evidence.quality), math.inf),
        )
    )
    scorer = OpenSetScorer(
        OpenSetPolicy(
            appearance_floor=-1.0,
            quality_floor=0.0,
            calibrated_confidence_floor=0.0,
            ambiguity_margin_floor=0.0,
            calibration_slope=slope,
            calibration_midpoint=midpoint,
            resident_prior_bonus=0.0,
            resident_prior_cap=0.05,
        )
    )
    margins: list[float] = []
    for sample in samples:
        confidences = sorted(
            (
                scorer.calibrate(candidate.raw_similarity)
                for candidate in sample.evidence.candidates
                if candidate.hard_allowed and candidate.gallery_exemplar_count >= 1
            ),
            reverse=True,
        )
        if not confidences:
            continue
        runner = confidences[1] if len(confidences) > 1 else 0.0
        margin = max(0.0, min(1.0, confidences[0] - runner))
        margins.extend((margin, math.nextafter(margin, math.inf)))
    return (
        _bounded_breakpoints(
            appearance_values,
            defaults=(-1.0, 0.70),
            lower=-1.0,
            upper=1.0,
            maximum_points=12,
        ),
        _bounded_breakpoints(
            quality_values,
            defaults=(0.0, 0.50),
            lower=0.0,
            upper=1.0,
            maximum_points=5,
        ),
        _bounded_breakpoints(
            margins,
            defaults=(0.0, 0.025),
            lower=0.0,
            upper=1.0,
            maximum_points=5,
        ),
    )


def _candidate_policies(
    dataset: IdentityCalibrationDataset,
    *,
    slope: float,
    midpoint: float,
) -> Iterable[OpenSetPolicy]:
    appearance_floors, quality_floors, ambiguity_floors = _policy_breakpoints(
        dataset,
        slope=slope,
        midpoint=midpoint,
    )
    prior_bonuses = (0.0, 0.01, 0.015, 0.025, 0.04, 0.05)
    for appearance in appearance_floors:
        for quality in quality_floors:
            for ambiguity in ambiguity_floors:
                for prior in prior_bonuses:
                    yield OpenSetPolicy(
                        appearance_floor=appearance,
                        quality_floor=quality,
                        calibrated_confidence_floor=0.50,
                        ambiguity_margin_floor=ambiguity,
                        calibration_slope=slope,
                        calibration_midpoint=midpoint,
                        resident_prior_bonus=prior,
                        resident_prior_cap=0.05,
                        unknown_utility=0.0,
                    )


def _policy_contract(policy: OpenSetPolicy) -> IdentityOpenSetPolicyContract:
    return IdentityOpenSetPolicyContract(
        **{
            key: value
            for key, value in policy.__dict__.items()
            if key in IdentityOpenSetPolicyContract.model_fields
        }
    )


def _household_candidate_policies(
    benchmark_policy: OpenSetPolicy,
    household_dataset: IdentityCalibrationDataset,
) -> Iterable[OpenSetPolicy]:
    appearance_breakpoints, quality_breakpoints, ambiguity_breakpoints = (
        _policy_breakpoints(
            household_dataset,
            slope=benchmark_policy.calibration_slope,
            midpoint=benchmark_policy.calibration_midpoint,
        )
    )
    appearances = sorted(
        {benchmark_policy.appearance_floor}
        | {
            value
            for value in appearance_breakpoints
            if value >= benchmark_policy.appearance_floor
        }
    )
    qualities = sorted(
        {benchmark_policy.quality_floor}
        | {
            value
            for value in quality_breakpoints
            if value >= benchmark_policy.quality_floor
        }
    )
    ambiguities = sorted(
        {benchmark_policy.ambiguity_margin_floor}
        | {
            value
            for value in ambiguity_breakpoints
            if value >= benchmark_policy.ambiguity_margin_floor
        }
    )
    for appearance in appearances:
        for quality in qualities:
            for ambiguity in ambiguities:
                yield OpenSetPolicy(
                    appearance_floor=appearance,
                    quality_floor=quality,
                    calibrated_confidence_floor=(
                        benchmark_policy.calibrated_confidence_floor
                    ),
                    ambiguity_margin_floor=ambiguity,
                    calibration_slope=benchmark_policy.calibration_slope,
                    calibration_midpoint=benchmark_policy.calibration_midpoint,
                    resident_prior_bonus=benchmark_policy.resident_prior_bonus,
                    resident_prior_cap=benchmark_policy.resident_prior_cap,
                    unknown_utility=benchmark_policy.unknown_utility,
                )


def _binding(
    dataset: IdentityCalibrationDataset,
    dataset_sha256: str,
) -> CalibrationDatasetBinding:
    encounters = {row.encounter_id for row in dataset.evidence_units}
    return CalibrationDatasetBinding(
        evidence_stratum=dataset.evidence_stratum,
        dataset_sha256=dataset_sha256,
        evidence_sha256=dataset.evidence_sha256,
        labels_sha256=dataset.labels_sha256,
        split_strategy=dataset.split_strategy,
        evidence_sources=tuple(
            sorted(
                {row.evidence.source for row in dataset.samples},
                key=lambda value: value.value,
            )
        ),
        runtimes=tuple(sorted({row.evidence.runtime for row in dataset.samples})),
        runtime_modes=tuple(
            sorted({row.evidence.runtime_mode for row in dataset.samples})
        ),
        truth_kinds=tuple(
            sorted(
                {row.label.truth_kind for row in dataset.samples},
                key=lambda value: value.value,
            )
        ),
        provenance=dataset.provenance,
        observation_count=len(dataset.samples),
        evidence_unit_count=len(dataset.evidence_units),
        encounter_count=len(encounters),
    )


def _gallery_limits(
    benchmark_dataset: IdentityCalibrationDataset,
    household_dataset: IdentityCalibrationDataset,
) -> CalibrationGalleryLimits:
    supported_rows: list[tuple[int, int, int, int]] = []
    for dataset in (benchmark_dataset, household_dataset):
        for sample in dataset.samples:
            if not _challenge_eligible(
                sample,
                require_truth=sample.label.partition is CalibrationPartition.TRAIN,
            ):
                continue
            candidates = tuple(
                row
                for row in sample.evidence.candidates
                if row.hard_allowed and row.gallery_exemplar_count >= 1
            )
            supported_rows.append(
                (
                    sum(row.identity_kind == "resident" for row in candidates),
                    sum(row.identity_kind == "visitor" for row in candidates),
                    len(candidates),
                    min(row.gallery_exemplar_count for row in candidates),
                )
            )
    if not supported_rows:
        raise IdentityCalibrationError("calibration has no gallery challenge rows")
    supported = tuple(min(row[index] for row in supported_rows) for index in range(4))
    if supported[0] < 1 or supported[2] < 2 or supported[3] < 1:
        raise IdentityCalibrationError(
            "both strata require deployment-representative gallery coverage"
        )
    return CalibrationGalleryLimits(
        maximum_resident_candidate_count=supported[0],
        maximum_visitor_candidate_count=supported[1],
        maximum_total_candidate_count=supported[2],
        maximum_exemplars_per_candidate=supported[3],
    )


def calibrate_datasets(
    benchmark_dataset: IdentityCalibrationDataset,
    household_dataset: IdentityCalibrationDataset,
    *,
    benchmark_dataset_sha256: str,
    household_dataset_sha256: str,
    generated_at_us: int,
    generator_revision: str,
    max_benchmark_holdout_far_upper_confidence_bound: float = (
        IDENTITY_CALIBRATION_MAX_HOLDOUT_FAR
    ),
    max_benchmark_holdout_frr: float = IDENTITY_CALIBRATION_MAX_HOLDOUT_FRR,
    max_benchmark_holdout_misidentification_upper_confidence_bound: float = (
        IDENTITY_CALIBRATION_MAX_HOLDOUT_MISIDENTIFICATION_RATE
    ),
) -> IdentityOpenSetCalibrationArtifact:
    try:
        benchmark_dataset = IdentityCalibrationDataset.model_validate(
            benchmark_dataset.model_dump(mode="python")
        )
        household_dataset = IdentityCalibrationDataset.model_validate(
            household_dataset.model_dump(mode="python")
        )
    except Exception as exc:
        raise IdentityCalibrationError(
            f"calibration input dataset contract is invalid: {exc}"
        ) from exc
    requested_gates = (
        (
            "max_benchmark_holdout_far_upper_confidence_bound",
            float(max_benchmark_holdout_far_upper_confidence_bound),
            IDENTITY_CALIBRATION_MAX_HOLDOUT_FAR,
        ),
        (
            "max_benchmark_holdout_frr",
            float(max_benchmark_holdout_frr),
            IDENTITY_CALIBRATION_MAX_HOLDOUT_FRR,
        ),
        (
            "max_benchmark_holdout_misidentification_upper_confidence_bound",
            float(max_benchmark_holdout_misidentification_upper_confidence_bound),
            IDENTITY_CALIBRATION_MAX_HOLDOUT_MISIDENTIFICATION_RATE,
        ),
    )
    for name, requested, product_maximum in requested_gates:
        if not math.isfinite(requested) or requested < 0.0:
            raise IdentityCalibrationError(f"{name} must be finite and non-negative")
        if requested > product_maximum:
            raise IdentityCalibrationError(
                f"{name}={requested} weakens the product maximum {product_maximum}"
            )
    if (
        IDENTITY_CALIBRATION_MAX_HARMFUL_PRIOR_CHANGES != 0
        or IDENTITY_CALIBRATION_MAX_PRIOR_REJECTION_RESCUES != 0
    ):
        raise IdentityCalibrationError(
            "identity calibration safety invariants must remain zero"
        )

    if benchmark_dataset.evidence_stratum is not CalibrationEvidenceStratum.BENCHMARK:
        raise IdentityCalibrationError(
            "first calibration dataset must be benchmark evidence"
        )
    if household_dataset.evidence_stratum is not CalibrationEvidenceStratum.HOUSEHOLD:
        raise IdentityCalibrationError(
            "second calibration dataset must be household evidence"
        )
    if (
        str(benchmark_dataset_sha256) == str(household_dataset_sha256)
        or benchmark_dataset.evidence_sha256 == household_dataset.evidence_sha256
        or benchmark_dataset.labels_sha256 == household_dataset.labels_sha256
        or benchmark_dataset.provenance.source_manifest_sha256
        == household_dataset.provenance.source_manifest_sha256
    ):
        raise IdentityCalibrationError(
            "benchmark and household strata must bind distinct corpora"
        )
    profiles = {
        (
            dataset.model_sha256,
            dataset.model_layer,
            dataset.embedding_dim,
        )
        for dataset in (benchmark_dataset, household_dataset)
    }
    if len(profiles) != 1:
        raise IdentityCalibrationError(
            "benchmark and household datasets must bind the same model profile"
        )
    if benchmark_dataset.evidence_unit_policy != household_dataset.evidence_unit_policy:
        raise IdentityCalibrationError(
            "benchmark and household datasets must use the same evidence-unit policy"
        )
    semantic_profiles = {
        dataset.provenance.model_semantic_profile_sha256
        for dataset in (benchmark_dataset, household_dataset)
    }
    if len(semantic_profiles) != 1:
        raise IdentityCalibrationError(
            "benchmark and household datasets bind different model semantic profiles"
        )
    cross_stratum_keys = (
        (
            "evidence events",
            {row.evidence.event_id for row in benchmark_dataset.samples},
            {row.evidence.event_id for row in household_dataset.samples},
        ),
        (
            "evidence units",
            {row.unit_id for row in benchmark_dataset.evidence_units},
            {row.unit_id for row in household_dataset.evidence_units},
        ),
        (
            "capture sessions",
            {row.evidence.session_id for row in benchmark_dataset.samples},
            {row.evidence.session_id for row in household_dataset.samples},
        ),
        (
            "runtime runs",
            {row.evidence.run_id for row in benchmark_dataset.samples},
            {row.evidence.run_id for row in household_dataset.samples},
        ),
        (
            "encounters",
            {row.label.encounter_id for row in benchmark_dataset.samples},
            {row.label.encounter_id for row in household_dataset.samples},
        ),
        (
            "truth people",
            {row.label.truth_person_key for row in benchmark_dataset.samples},
            {row.label.truth_person_key for row in household_dataset.samples},
        ),
    )
    for label, benchmark_keys, household_keys in cross_stratum_keys:
        overlap = benchmark_keys & household_keys
        if overlap:
            raise IdentityCalibrationError(
                f"benchmark and household {label} overlap; strata must be disjoint"
            )
    candidate_kinds: dict[str, str] = {}
    for dataset in (benchmark_dataset, household_dataset):
        for sample in dataset.samples:
            for candidate in sample.evidence.candidates:
                previous = candidate_kinds.setdefault(
                    candidate.subject_id, candidate.identity_kind
                )
                if previous != candidate.identity_kind:
                    raise IdentityCalibrationError(
                        "a candidate subject changes resident/visitor kind across strata"
                    )
    for dataset in (benchmark_dataset, household_dataset):
        _validate_disjointness(dataset)
        _validate_sufficiency(dataset)

    selected_units = tuple(
        unit
        for unit in benchmark_dataset.evidence_units
        if unit.partition is CalibrationPartition.TRAIN
        and unit.fit_event_id is not None
    )
    slope, midpoint, fit = _fit_logistic(
        _candidate_pairs(benchmark_dataset),
        selected_evidence_unit_count=len(selected_units),
        selected_encounter_count=len({row.encounter_id for row in selected_units}),
        selected_person_count=len({row.truth_person_key for row in selected_units}),
    )

    best: tuple[tuple[object, ...], OpenSetPolicy, CalibrationMetrics] | None = None
    for policy in _candidate_policies(
        benchmark_dataset,
        slope=slope,
        midpoint=midpoint,
    ):
        metrics = _metrics(benchmark_dataset, CalibrationPartition.TRAIN, policy)
        if (
            metrics.person_far > float(max_benchmark_holdout_far_upper_confidence_bound)
            or metrics.person_frr > float(max_benchmark_holdout_frr)
            or metrics.person_misidentification_rate
            > float(max_benchmark_holdout_misidentification_upper_confidence_bound)
            or metrics.resident_prior_harmful_encounter_count > 0
            or metrics.resident_prior_rejection_rescue_encounter_count > 0
        ):
            continue
        rank = (
            metrics.correct_accept_person_count,
            -metrics.false_accept_person_count,
            -metrics.misidentification_person_count,
            metrics.resident_prior_beneficial_encounter_count,
            -abs(policy.resident_prior_bonus - 0.025),
            -policy.appearance_floor,
            -policy.quality_floor,
            -policy.ambiguity_margin_floor,
        )
        if best is None or rank > best[0]:
            best = (rank, policy, metrics)
    if best is None:
        raise IdentityCalibrationError(
            "no benchmark training policy satisfies the point FAR/FRR/misidentification gates"
        )
    _, benchmark_policy, _ = best

    local_best: (
        tuple[tuple[object, ...], OpenSetPolicy, CalibrationMetrics, CalibrationMetrics]
        | None
    ) = None
    for policy in _household_candidate_policies(
        benchmark_policy,
        household_dataset,
    ):
        benchmark_training_metrics = _metrics(
            benchmark_dataset, CalibrationPartition.TRAIN, policy
        )
        household_training_metrics = _metrics(
            household_dataset, CalibrationPartition.TRAIN, policy
        )
        if (
            benchmark_training_metrics.person_far
            > float(max_benchmark_holdout_far_upper_confidence_bound)
            or benchmark_training_metrics.person_frr > float(max_benchmark_holdout_frr)
            or benchmark_training_metrics.person_misidentification_rate
            > float(max_benchmark_holdout_misidentification_upper_confidence_bound)
            or household_training_metrics.false_accept_encounter_count > 0
            or household_training_metrics.misidentification_encounter_count > 0
            or household_training_metrics.frr > float(max_benchmark_holdout_frr)
            or any(
                metrics.resident_prior_harmful_encounter_count > 0
                or metrics.resident_prior_rejection_rescue_encounter_count > 0
                for metrics in (
                    benchmark_training_metrics,
                    household_training_metrics,
                )
            )
        ):
            continue
        tightening = (
            (policy.appearance_floor - benchmark_policy.appearance_floor)
            + (policy.quality_floor - benchmark_policy.quality_floor)
            + (policy.ambiguity_margin_floor - benchmark_policy.ambiguity_margin_floor)
        )
        rank = (
            household_training_metrics.correct_accept_encounter_count,
            -tightening,
            benchmark_training_metrics.correct_accept_person_count,
            -policy.appearance_floor,
            -policy.quality_floor,
            -policy.ambiguity_margin_floor,
        )
        if local_best is None or rank > local_best[0]:
            local_best = (
                rank,
                policy,
                benchmark_training_metrics,
                household_training_metrics,
            )
    if local_best is None:
        raise IdentityCalibrationError(
            "no household-tightened policy satisfies the local zero-error gate"
        )
    _, policy, benchmark_training_metrics, household_training_metrics = local_best
    benchmark_holdout_metrics = _metrics(
        benchmark_dataset, CalibrationPartition.HOLDOUT, policy
    )
    household_holdout_metrics = _metrics(
        household_dataset, CalibrationPartition.HOLDOUT, policy
    )

    failures = []
    if benchmark_holdout_metrics.far_upper_confidence_bound > float(
        max_benchmark_holdout_far_upper_confidence_bound
    ):
        failures.append(
            "benchmark FAR one-sided upper bound "
            f"{benchmark_holdout_metrics.far_upper_confidence_bound:.6f} > "
            f"{float(max_benchmark_holdout_far_upper_confidence_bound):.6f}"
        )
    if benchmark_holdout_metrics.person_frr > float(max_benchmark_holdout_frr):
        failures.append(
            f"benchmark person-worst FRR {benchmark_holdout_metrics.person_frr:.6f} > "
            f"{float(max_benchmark_holdout_frr):.6f}"
        )
    if benchmark_holdout_metrics.misidentification_upper_confidence_bound > float(
        max_benchmark_holdout_misidentification_upper_confidence_bound
    ):
        failures.append(
            "benchmark misidentification one-sided upper bound "
            f"{benchmark_holdout_metrics.misidentification_upper_confidence_bound:.6f} > "
            f"{float(max_benchmark_holdout_misidentification_upper_confidence_bound):.6f}"
        )
    if household_holdout_metrics.false_accept_encounter_count > 0:
        failures.append("household holdout contains a false-accept encounter")
    if household_holdout_metrics.misidentification_encounter_count > 0:
        failures.append("household holdout contains a misidentification encounter")
    if household_holdout_metrics.frr > float(max_benchmark_holdout_frr):
        failures.append(
            f"household FRR {household_holdout_metrics.frr:.6f} > "
            f"{float(max_benchmark_holdout_frr):.6f}"
        )
    for label, metrics in (
        ("benchmark holdout", benchmark_holdout_metrics),
        ("household holdout", household_holdout_metrics),
    ):
        if metrics.resident_prior_harmful_encounter_count > 0:
            failures.append(f"resident prior caused a harmful {label} winner change")
        if metrics.resident_prior_rejection_rescue_encounter_count > 0:
            failures.append(f"resident prior rescued a rejected {label} encounter")
    if failures:
        raise IdentityCalibrationError(
            "holdout acceptance failed: " + "; ".join(failures)
        )
    return IdentityOpenSetCalibrationArtifact(
        contract="noesis.identity.open_set_calibration",
        contract_version=2,
        generated_at_us=int(generated_at_us),
        generator=IDENTITY_CALIBRATION_GENERATOR,
        generator_revision=str(generator_revision),
        model_sha256=benchmark_dataset.model_sha256,
        model_layer=benchmark_dataset.model_layer,
        embedding_dim=benchmark_dataset.embedding_dim,
        model_semantic_profile_sha256=next(iter(semantic_profiles)),
        authority_scope="open_set_scorer_policy_only",
        gallery_limits=_gallery_limits(benchmark_dataset, household_dataset),
        evidence_unit_policy=benchmark_dataset.evidence_unit_policy,
        benchmark_dataset=_binding(benchmark_dataset, str(benchmark_dataset_sha256)),
        household_dataset=_binding(household_dataset, str(household_dataset_sha256)),
        benchmark_policy=_policy_contract(benchmark_policy),
        policy=_policy_contract(policy),
        fit=fit,
        benchmark_training_metrics=benchmark_training_metrics,
        benchmark_holdout_metrics=benchmark_holdout_metrics,
        household_training_metrics=household_training_metrics,
        household_holdout_metrics=household_holdout_metrics,
        acceptance=CalibrationAcceptance(
            passed=True,
            confidence_level=IDENTITY_CALIBRATION_CONFIDENCE_LEVEL,
            confidence_method=IDENTITY_CALIBRATION_CONFIDENCE_METHOD,
            max_benchmark_holdout_far_upper_confidence_bound=float(
                max_benchmark_holdout_far_upper_confidence_bound
            ),
            max_benchmark_holdout_frr=float(max_benchmark_holdout_frr),
            max_benchmark_holdout_misidentification_upper_confidence_bound=float(
                max_benchmark_holdout_misidentification_upper_confidence_bound
            ),
            max_household_false_accept_encounters=0,
            max_household_misidentification_encounters=0,
            max_household_holdout_frr=float(max_benchmark_holdout_frr),
            max_harmful_prior_encounters=0,
            max_prior_rejection_rescue_encounters=0,
            minimum_benchmark_train_known_persons=(
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_KNOWN_PERSONS
            ),
            minimum_benchmark_train_unknown_persons=(
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_TRAIN_UNKNOWN_PERSONS
            ),
            minimum_benchmark_holdout_known_persons=(
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_KNOWN_PERSONS
            ),
            minimum_benchmark_holdout_unknown_persons=(
                IDENTITY_CALIBRATION_MINIMUM_BENCHMARK_HOLDOUT_UNKNOWN_PERSONS
            ),
            minimum_household_train_known_encounters=(
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_KNOWN_ENCOUNTERS
            ),
            minimum_household_train_unknown_encounters=(
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_TRAIN_UNKNOWN_ENCOUNTERS
            ),
            minimum_household_holdout_known_encounters=(
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_KNOWN_ENCOUNTERS
            ),
            minimum_household_holdout_unknown_encounters=(
                IDENTITY_CALIBRATION_MINIMUM_HOUSEHOLD_HOLDOUT_UNKNOWN_ENCOUNTERS
            ),
        ),
    )


def calibrate_dataset_files(
    benchmark_dataset_path: str | Path,
    household_dataset_path: str | Path,
    **kwargs: object,
) -> IdentityOpenSetCalibrationArtifact:
    benchmark_dataset, benchmark_sha256 = _load_calibration_dataset_snapshot(
        benchmark_dataset_path
    )
    household_dataset, household_sha256 = _load_calibration_dataset_snapshot(
        household_dataset_path
    )
    return calibrate_datasets(
        benchmark_dataset,
        household_dataset,
        benchmark_dataset_sha256=benchmark_sha256,
        household_dataset_sha256=household_sha256,
        **kwargs,
    )


__all__ = [
    "EvidenceValidationSummary",
    "IdentityCalibrationError",
    "build_calibration_dataset",
    "calibrate_datasets",
    "calibrate_dataset_files",
    "canonical_json_bytes",
    "dataset_review",
    "load_and_validate_evidence",
    "load_calibration_dataset",
    "load_label_set",
    "sha256_file",
    "write_private_json",
]
