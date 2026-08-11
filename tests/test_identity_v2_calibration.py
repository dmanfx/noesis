from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from noesis.identity_v2_service import _load_open_set_policy
from noesis_core.contracts.identity_calibration import (
    CalibrationCandidateScore,
    CalibrationDatasetProvenance,
    CalibrationEvidenceStratum,
    CalibrationPartition,
    CalibrationSplitStrategy,
    CalibrationTruthKind,
    CalibrationTruthLabel,
    IdentityCalibrationDataset,
    IdentityEvidenceLabelSet,
    ShadowIdentityEvidenceRecord,
    one_sided_binomial_upper_confidence_bound,
)
from reid.identity_v2 import (
    FrameBatchResult,
    IdentityCalibrationError,
    IdentityDecision,
    IdentityEvidenceError,
    IdentityEvidenceRecorder,
    OpenSetScorer,
    PrimitiveFrameObservation,
    PublicIdentityOverlay,
    TrackletObservation,
    build_calibration_dataset,
    calibrate_datasets,
    dataset_review,
    evidence_checkpoint_path,
    load_calibration_dataset,
    load_and_validate_evidence,
)
from reid.identity_v2.calibration import (
    canonical_json_bytes,
    load_label_set,
    sha256_file,
    write_private_json,
)
from reid.identity_v2.store import EnrollmentObservationKey, PurgeResult

MODEL_SHA = "a" * 64
MODEL_LAYER = "features"
DIMENSION = 4


def _event(body: dict) -> ShadowIdentityEvidenceRecord:
    event_id = hashlib.sha256(
        b"noesis-identity-shadow-evidence-v2\0" + canonical_json_bytes(body)
    ).hexdigest()
    return ShadowIdentityEvidenceRecord.model_validate({"event_id": event_id, **body})


def _default_counts(
    stratum: CalibrationEvidenceStratum,
) -> dict[tuple[CalibrationPartition, str], int]:
    if stratum is CalibrationEvidenceStratum.BENCHMARK:
        return {
            (CalibrationPartition.TRAIN, "known"): 50,
            (CalibrationPartition.TRAIN, "unknown"): 50,
            (CalibrationPartition.HOLDOUT, "known"): 300,
            (CalibrationPartition.HOLDOUT, "unknown"): 300,
        }
    return {
        (CalibrationPartition.TRAIN, "known"): 10,
        (CalibrationPartition.TRAIN, "unknown"): 10,
        (CalibrationPartition.HOLDOUT, "known"): 20,
        (CalibrationPartition.HOLDOUT, "unknown"): 20,
    }


def _fixture_files(
    tmp_path: Path,
    *,
    stratum: CalibrationEvidenceStratum,
    counts: dict[tuple[CalibrationPartition, str], int] | None = None,
    frames_per_encounter: int = 1,
    first_train_known_windows: int = 0,
    leak_person: bool = False,
    leak_session: bool = False,
    one_bad_unknown_holdout_frame: bool = False,
    benchmark_known_person_modulo: int | None = None,
    benchmark_unknown_person_modulo: int | None = None,
    omit_first_holdout_truth_candidate: bool = False,
    all_hard_blocked_first_train: bool = False,
    include_zero_exemplar_decoy: bool = False,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tmp_path.chmod(0o700)
    split_strategy = (
        CalibrationSplitStrategy.SUBJECT_DISJOINT
        if stratum is CalibrationEvidenceStratum.BENCHMARK
        else CalibrationSplitStrategy.SESSION_DISJOINT
    )
    sample_counts = counts or _default_counts(stratum)
    residents = ("resident:a", "resident:b", "resident:c")
    records: list[ShadowIdentityEvidenceRecord] = []
    labels: list[CalibrationTruthLabel] = []
    frame = 0
    sequence = 0
    observed_at_us = 1_000_000
    previous_event_id = None
    for partition in (CalibrationPartition.TRAIN, CalibrationPartition.HOLDOUT):
        for truth_class in ("known", "unknown"):
            for index in range(sample_counts[(partition, truth_class)]):
                if truth_class == "known":
                    if stratum is CalibrationEvidenceStratum.BENCHMARK:
                        person_index = (
                            index
                            if benchmark_known_person_modulo is None
                            else index % benchmark_known_person_modulo
                        )
                        truth = (
                            "benchmark-train-known-person-0"
                            if leak_person
                            and partition is CalibrationPartition.HOLDOUT
                            and index == 0
                            else f"benchmark-{partition.value}-known-person-{person_index}"
                        )
                        gallery_subjects = (
                            truth,
                            "benchmark-resident-impostor-a",
                            "benchmark-resident-impostor-b",
                        )
                    else:
                        truth = residents[index % len(residents)]
                        gallery_subjects = residents
                    truth_person_key = truth
                    scores = {
                        candidate: (0.90 if candidate == truth else 0.42)
                        for candidate in gallery_subjects
                    }
                    outcome = "resident"
                    winner = truth
                    truth_kind = CalibrationTruthKind.RESIDENT
                else:
                    truth = None
                    truth_person_key = (
                        f"unknown-{partition.value}-{index if benchmark_unknown_person_modulo is None else index % benchmark_unknown_person_modulo}"
                        if stratum is CalibrationEvidenceStratum.BENCHMARK
                        else f"local-unknown-{partition.value}-{index}"
                    )
                    unknown_gallery = (
                        (
                            "benchmark-resident-impostor-a",
                            "benchmark-resident-impostor-b",
                            "benchmark-resident-impostor-c",
                        )
                        if stratum is CalibrationEvidenceStratum.BENCHMARK
                        else residents
                    )
                    scores = {
                        candidate: score
                        for candidate, score in zip(
                            unknown_gallery, (0.55, 0.50, 0.45), strict=True
                        )
                    }
                    outcome = "unknown"
                    winner = None
                    truth_kind = CalibrationTruthKind.UNKNOWN
                scores["visitor:challenge"] = 0.40
                if (
                    omit_first_holdout_truth_candidate
                    and partition is CalibrationPartition.HOLDOUT
                    and truth_class == "known"
                    and index == 0
                ):
                    scores.pop(str(truth), None)
                    outcome = "unknown"
                    winner = None
                if include_zero_exemplar_decoy:
                    scores["resident:zero-exemplar-decoy"] = 0.99

                encounter_id = (
                    f"{stratum.value}-{partition.value}-{truth_class}-encounter-{index}"
                )
                session_partition = (
                    CalibrationPartition.TRAIN
                    if leak_session
                    and partition is CalibrationPartition.HOLDOUT
                    and index == 0
                    else partition
                )
                session = (
                    f"{stratum.value}-{session_partition.value}-{truth_class}-session-"
                    f"{index}"
                )
                run = f"{stratum.value}-{partition.value}-run"
                source = (
                    "replay"
                    if stratum is CalibrationEvidenceStratum.BENCHMARK
                    else "shadow"
                )
                runtime = "replay" if source == "replay" else "test"
                frame_count = (
                    first_train_known_windows
                    if first_train_known_windows > 0
                    and partition is CalibrationPartition.TRAIN
                    and truth_class == "known"
                    and index == 0
                    else frames_per_encounter
                )
                for encounter_frame in range(frame_count):
                    frame += 1
                    observed_at_us += (
                        6_000_000
                        if first_train_known_windows > 0
                        and partition is CalibrationPartition.TRAIN
                        and truth_class == "known"
                        and index == 0
                        else 1_000
                    )
                    frame_scores = dict(scores)
                    if (
                        one_bad_unknown_holdout_frame
                        and partition is CalibrationPartition.HOLDOUT
                        and truth_class == "unknown"
                        and index == 0
                        and encounter_frame == 0
                    ):
                        frame_scores[residents[0]] = 0.92
                    body = {
                        "contract": "noesis.identity.shadow_score_evidence",
                        "contract_version": 2,
                        "sequence": sequence,
                        "previous_event_id": previous_event_id,
                        "observed_at_us": observed_at_us,
                        "source": source,
                        "session_id": session,
                        "runtime": runtime,
                        "runtime_mode": "shadow",
                        "run_id": run,
                        "camera_id": "camera-a",
                        "tracker_id": (
                            f"tracker-{partition.value}-{truth_class}-{index}"
                        ),
                        "frame_id": frame,
                        "observation_id": (
                            f"observation-{partition.value}-{truth_class}-{index}-"
                            f"{encounter_frame}"
                        ),
                        "model_sha256": MODEL_SHA,
                        "model_semantic_profile_sha256": "1" * 64,
                        "model_layer": MODEL_LAYER,
                        "embedding_dim": DIMENSION,
                        "quality": 0.90,
                        "candidates": [
                            CalibrationCandidateScore(
                                subject_id=candidate,
                                identity_kind=(
                                    "visitor"
                                    if candidate.startswith("visitor:")
                                    else "resident"
                                ),
                                raw_similarity=score,
                                hard_allowed=not (
                                    all_hard_blocked_first_train
                                    and partition is CalibrationPartition.TRAIN
                                    and truth_class == "known"
                                    and index == 0
                                ),
                                hard_constraint_reason=(
                                    "fixture_hard_block"
                                    if all_hard_blocked_first_train
                                    and partition is CalibrationPartition.TRAIN
                                    and truth_class == "known"
                                    and index == 0
                                    else None
                                ),
                                gallery_exemplar_count=(
                                    0
                                    if candidate == "resident:zero-exemplar-decoy"
                                    else 4
                                ),
                            ).model_dump(mode="json")
                            for candidate, score in sorted(frame_scores.items())
                        ],
                        "pre_prior_winner_id": winner,
                        "final_winner_id": winner,
                        "final_outcome": outcome,
                        "calibrated_confidence": 0.90 if winner else 0.50,
                        "reject_reason": None if winner else "appearance_below_floor",
                        "prior_changed_winner": False,
                    }
                    record = _event(body)
                    records.append(record)
                    sequence += 1
                    previous_event_id = record.event_id
                    labels.append(
                        CalibrationTruthLabel(
                            event_id=record.event_id,
                            partition=partition,
                            truth_kind=truth_kind,
                            truth_subject_id=truth,
                            truth_person_key=truth_person_key,
                            encounter_id=encounter_id,
                        )
                    )

    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_bytes(
        b"".join(
            canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in records
        )
    )
    evidence_path.chmod(0o600)
    checkpoint_body = {
        "contract": "noesis.identity.shadow_score_evidence_chain",
        "contract_version": 1,
        "next_sequence": len(records),
        "first_sequence": 0,
        "previous_event_id": None,
        "tail_event_id": records[-1].event_id,
        "retained_count": len(records),
        "retained_bytes": evidence_path.stat().st_size,
        "updated_at_us": records[-1].observed_at_us,
    }
    checkpoint_body["checkpoint_sha256"] = hashlib.sha256(
        b"noesis-identity-shadow-evidence-chain-v1\0"
        + canonical_json_bytes(checkpoint_body)
    ).hexdigest()
    checkpoint_path = evidence_checkpoint_path(evidence_path)
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint_body) + b"\n")
    checkpoint_path.chmod(0o600)

    labels_path = tmp_path / "labels.json"
    label_set = IdentityEvidenceLabelSet(
        contract="noesis.identity.evidence_labels",
        contract_version=2,
        evidence_sha256=sha256_file(evidence_path),
        evidence_stratum=stratum,
        split_strategy=split_strategy,
        provenance=CalibrationDatasetProvenance(
            source_name=f"fixture-{stratum.value}",
            source_revision="fixture-v2",
            source_manifest_sha256=(
                "b" if stratum is CalibrationEvidenceStratum.BENCHMARK else "c"
            )
            * 64,
            model_semantic_profile_sha256="1" * 64,
            labeling_protocol_revision="encounter-labeling-v2",
            population_basis=(
                "licensed_real_people"
                if stratum is CalibrationEvidenceStratum.BENCHMARK
                else "owner_consented_household_people"
            ),
        ),
        labeled_by="owner",
        labeling_revision="fixture-v2",
        labels=tuple(labels),
    )
    write_private_json(labels_path, label_set.model_dump(mode="json"))
    return evidence_path, labels_path


def _dataset_files(tmp_path: Path):
    benchmark_evidence, benchmark_labels = _fixture_files(
        tmp_path / "benchmark", stratum=CalibrationEvidenceStratum.BENCHMARK
    )
    household_evidence, household_labels = _fixture_files(
        tmp_path / "household", stratum=CalibrationEvidenceStratum.HOUSEHOLD
    )
    benchmark = build_calibration_dataset(benchmark_evidence, benchmark_labels)
    household = build_calibration_dataset(household_evidence, household_labels)
    benchmark_path = write_private_json(
        tmp_path / "benchmark-dataset.json", benchmark.model_dump(mode="json")
    )
    household_path = write_private_json(
        tmp_path / "household-dataset.json", household.model_dump(mode="json")
    )
    return benchmark, household, benchmark_path, household_path


def _calibrate(benchmark, household, benchmark_path, household_path):
    return calibrate_datasets(
        benchmark,
        household,
        benchmark_dataset_sha256=sha256_file(benchmark_path),
        household_dataset_sha256=sha256_file(household_path),
        generated_at_us=123456789,
        generator_revision="test-revision",
    )


def _artifact_env(path: Path) -> dict[str, str]:
    return {
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT": str(path),
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
        "NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256": "1" * 64,
    }


def test_score_only_recorder_never_serializes_embedding_vectors(tmp_path: Path) -> None:
    recorder = IdentityEvidenceRecorder(
        tmp_path / "private" / "evidence.jsonl",
        session_id="session-a",
        source="shadow",
        runtime="test",
    )
    observation = PrimitiveFrameObservation(
        run_id="run-a",
        camera_id="camera-a",
        tracker_id="tracker-a",
        frame_id=1,
        observation_id="observation-a",
        quality=0.9,
        embedding=(1.0, 0.0, 0.0, 0.0),
    )
    key = EnrollmentObservationKey(
        run_id="run-a",
        camera_id="camera-a",
        tracker_id="tracker-a",
        frame_id=1,
        observation_id="observation-a",
    )
    decision = IdentityDecision(
        tracklet_id=key.tracklet_id,
        identity_id=None,
        identity_kind=None,
        is_unknown=True,
        raw_similarity=None,
        calibrated_confidence=1.0,
        ambiguity_margin=None,
        prior_contribution=0.0,
        assignment_utility=0.0,
        reason="no_candidates",
    )
    overlay = PublicIdentityOverlay(
        key=key,
        identity_state="unknown",
        subject_id=None,
        compatibility_sid=None,
        display_name=None,
        visitor_generation=None,
        calibrated_confidence=1.0,
        reason="no_candidates",
        provisional_evidence_count=0,
        resolver_decision=decision,
    )
    recorder.append_frame(
        run_id="run-a",
        model_sha256=MODEL_SHA,
        model_semantic_profile_sha256="1" * 64,
        model_layer=MODEL_LAYER,
        embedding_dim=DIMENSION,
        observed_at_us=1,
        observations=(observation,),
        candidate_rows=(TrackletObservation(tracklet_id=key.tracklet_id, quality=0.9),),
        batch=FrameBatchResult(
            overlays=(overlay,),
            minted_visitor_sessions=(),
            released_visitor_session_uuids=(),
            retention=PurgeResult(0, 0, 0, 0),
            active_claims=(),
        ),
        scorer=OpenSetScorer(),
        runtime_mode="shadow",
    )
    payload = json.loads((tmp_path / "private" / "evidence.jsonl").read_text())
    assert "embedding" not in payload
    assert "vector" not in payload
    assert payload["embedding_dim"] == DIMENSION

    tampered = json.loads(recorder.path.read_text(encoding="utf-8"))
    tampered["quality"] = 0.1
    recorder.path.write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(IdentityCalibrationError):
        load_and_validate_evidence(recorder.path)
    with pytest.raises(IdentityEvidenceError, match="event digest mismatch"):
        IdentityEvidenceRecorder(
            recorder.path,
            session_id="session-b",
            source="shadow",
            runtime="test",
        )


def test_calibration_replay_rejects_noncanonical_identity_row_bytes(
    tmp_path: Path,
) -> None:
    path, _labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
    )
    path.write_bytes(b"  " + path.read_bytes())
    checkpoint_path = evidence_checkpoint_path(path)
    checkpoint = json.loads(checkpoint_path.read_bytes())
    checkpoint["retained_bytes"] = path.stat().st_size
    checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        b"noesis-identity-shadow-evidence-chain-v1\0"
        + canonical_json_bytes(checkpoint)
    ).hexdigest()
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint) + b"\n")
    checkpoint_path.chmod(0o600)

    with pytest.raises(IdentityCalibrationError, match="not canonical JSON"):
        load_and_validate_evidence(path)


def test_evidence_chain_rejects_selective_interior_deletion(tmp_path: Path) -> None:
    evidence, _ = _fixture_files(tmp_path, stratum=CalibrationEvidenceStratum.HOUSEHOLD)
    lines = evidence.read_bytes().splitlines(keepends=True)
    del lines[len(lines) // 2]
    evidence.write_bytes(b"".join(lines))
    checkpoint_path = evidence_checkpoint_path(evidence)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["retained_count"] -= 1
    checkpoint["next_sequence"] -= 1
    checkpoint["retained_bytes"] = evidence.stat().st_size
    checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        b"noesis-identity-shadow-evidence-chain-v1\0" + canonical_json_bytes(checkpoint)
    ).hexdigest()
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint) + b"\n")
    checkpoint_path.chmod(0o600)
    with pytest.raises(
        IdentityCalibrationError,
        match="sequence gap|hash-chain mismatch|endpoints disagree",
    ):
        load_and_validate_evidence(evidence)


def test_repeated_frames_never_satisfy_encounter_minima(tmp_path: Path) -> None:
    counts = {
        (CalibrationPartition.TRAIN, "known"): 2,
        (CalibrationPartition.TRAIN, "unknown"): 2,
        (CalibrationPartition.HOLDOUT, "known"): 5,
        (CalibrationPartition.HOLDOUT, "unknown"): 5,
    }
    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
        counts=counts,
        frames_per_encounter=25,
    )
    with pytest.raises(
        IdentityCalibrationError,
        match="requires at least 10 known and 10 unknown independent encounters",
    ):
        build_calibration_dataset(evidence, labels)


def test_evidence_units_are_capped_and_fit_thinning_is_deterministic(
    tmp_path: Path,
) -> None:
    evidence, labels = _fixture_files(
        tmp_path / "valid",
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
        first_train_known_windows=12,
    )
    dataset = build_calibration_dataset(evidence, labels)
    long_encounter = [
        row
        for row in dataset.evidence_units
        if row.encounter_id == "household-train-known-encounter-0"
    ]
    assert len(long_encounter) == 12
    assert sum(row.fit_event_id is not None for row in long_encounter) == 8
    assert len({row.fit_event_id for row in long_encounter if row.fit_event_id}) == 8

    tiny_counts = {
        (partition, truth): 1
        for partition in (CalibrationPartition.TRAIN, CalibrationPartition.HOLDOUT)
        for truth in ("known", "unknown")
    }
    over_cap_evidence, over_cap_labels = _fixture_files(
        tmp_path / "over-cap",
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
        counts=tiny_counts,
        frames_per_encounter=301,
    )
    with pytest.raises(IdentityCalibrationError, match="exceeds the observation cap"):
        build_calibration_dataset(over_cap_evidence, over_cap_labels)


def test_split_leakage_is_rejected_for_people_sessions_and_runs(tmp_path: Path) -> None:
    evidence, labels = _fixture_files(
        tmp_path / "people",
        stratum=CalibrationEvidenceStratum.BENCHMARK,
        leak_person=True,
    )
    with pytest.raises(IdentityCalibrationError, match="truth people cross"):
        build_calibration_dataset(evidence, labels)

    evidence, labels = _fixture_files(
        tmp_path / "aliased-people",
        stratum=CalibrationEvidenceStratum.BENCHMARK,
    )
    payload = json.loads(labels.read_text(encoding="utf-8"))
    for row in payload["labels"]:
        if row["partition"] == "holdout" and row["truth_kind"] == "resident":
            row["truth_subject_id"] = "benchmark-train-known-person-0"
    write_private_json(labels, payload)
    with pytest.raises(
        IdentityCalibrationError,
        match="known truth subject maps to multiple person keys",
    ):
        build_calibration_dataset(evidence, labels)

    evidence, labels = _fixture_files(
        tmp_path / "sessions",
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
        leak_session=True,
    )
    with pytest.raises(IdentityCalibrationError, match="capture session"):
        build_calibration_dataset(evidence, labels)

    evidence, labels = _fixture_files(
        tmp_path / "household-unknown-people",
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
    )
    payload = json.loads(labels.read_text(encoding="utf-8"))
    train_unknown = next(
        row
        for row in payload["labels"]
        if row["partition"] == "train" and row["truth_kind"] == "unknown"
    )
    holdout_unknown = next(
        row
        for row in payload["labels"]
        if row["partition"] == "holdout" and row["truth_kind"] == "unknown"
    )
    holdout_unknown["truth_person_key"] = train_unknown["truth_person_key"]
    write_private_json(labels, payload)
    with pytest.raises(
        IdentityCalibrationError, match="household unknown truth people"
    ):
        build_calibration_dataset(evidence, labels)


def test_benchmark_minima_count_challenge_covered_resident_people_not_encounters(
    tmp_path: Path,
) -> None:
    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.BENCHMARK,
        benchmark_known_person_modulo=10,
    )
    with pytest.raises(
        IdentityCalibrationError,
        match="challenge-covered people; got 10 and",
    ):
        build_calibration_dataset(evidence, labels)


def test_repeated_person_encounters_do_not_inflate_benchmark_confidence(
    tmp_path: Path,
) -> None:
    from reid.identity_v2.calibration import _metrics

    counts = _default_counts(CalibrationEvidenceStratum.BENCHMARK)
    counts[(CalibrationPartition.HOLDOUT, "known")] = 301
    counts[(CalibrationPartition.HOLDOUT, "unknown")] = 301
    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.BENCHMARK,
        counts=counts,
        benchmark_known_person_modulo=300,
        benchmark_unknown_person_modulo=300,
    )
    dataset = build_calibration_dataset(evidence, labels)
    metrics = _metrics(
        dataset,
        CalibrationPartition.HOLDOUT,
        OpenSetScorer().policy,
    )
    assert metrics.known_encounter_count == 301
    assert metrics.unknown_encounter_count == 301
    assert metrics.known_person_count == 300
    assert metrics.unknown_person_count == 300
    assert metrics.far_upper_confidence_bound == pytest.approx(
        one_sided_binomial_upper_confidence_bound(0, 300)
    )
    assert metrics.misidentification_upper_confidence_bound == pytest.approx(
        one_sided_binomial_upper_confidence_bound(0, 300)
    )


def test_holdout_missing_truth_is_counted_as_a_challenge_failure(
    tmp_path: Path,
) -> None:
    from reid.identity_v2.calibration import _metrics

    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.BENCHMARK,
        omit_first_holdout_truth_candidate=True,
    )
    dataset = build_calibration_dataset(evidence, labels)
    metrics = _metrics(
        dataset,
        CalibrationPartition.HOLDOUT,
        OpenSetScorer().policy,
    )
    assert metrics.known_challenge_person_count == 300
    assert metrics.false_reject_person_count == 1
    assert metrics.person_frr == pytest.approx(1.0 / 300.0)


def test_training_challenge_must_have_usable_resident_and_visitor_competition(
    tmp_path: Path,
) -> None:
    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
        all_hard_blocked_first_train=True,
    )
    with pytest.raises(
        IdentityCalibrationError,
        match="lacks deployment-representative hard-allowed gallery challenge",
    ):
        build_calibration_dataset(evidence, labels)


def test_zero_exemplar_candidates_never_influence_fit_pairs(tmp_path: Path) -> None:
    from reid.identity_v2.calibration import _candidate_pairs

    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.BENCHMARK,
        include_zero_exemplar_decoy=True,
    )
    dataset = build_calibration_dataset(evidence, labels)
    pairs = _candidate_pairs(dataset)
    assert len(pairs) == 400
    assert all(score < 0.99 for score, _label, _weight in pairs)


def test_candidate_kind_flip_is_rejected_by_dataset_contract(tmp_path: Path) -> None:
    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
    )
    payload = build_calibration_dataset(evidence, labels).model_dump(mode="json")
    visitor_rows = [
        sample
        for sample in payload["samples"]
        if any(
            candidate["subject_id"] == "visitor:challenge"
            for candidate in sample["evidence"]["candidates"]
        )
    ]
    visitor = next(
        candidate
        for candidate in visitor_rows[0]["evidence"]["candidates"]
        if candidate["subject_id"] == "visitor:challenge"
    )
    visitor["identity_kind"] = "resident"
    with pytest.raises(Exception, match="changes resident/visitor identity kind"):
        IdentityCalibrationDataset.model_validate(payload)


def test_gallery_authority_uses_worst_covered_row_not_largest_row(
    tmp_path: Path,
) -> None:
    from reid.identity_v2.calibration import _gallery_limits

    benchmark, household, _benchmark_path, _household_path = _dataset_files(tmp_path)
    sample = benchmark.samples[0]
    extra = tuple(
        CalibrationCandidateScore(
            subject_id=f"resident:oversized-{index}",
            identity_kind="resident",
            raw_similarity=0.1,
            gallery_exemplar_count=20,
        )
        for index in range(20)
    )
    evidence = sample.evidence.model_copy(
        update={"candidates": sample.evidence.candidates + extra}
    )
    oversized = benchmark.model_copy(
        update={
            "samples": (
                sample.model_copy(update={"evidence": evidence}),
                *benchmark.samples[1:],
            )
        }
    )
    limits = _gallery_limits(oversized, household)
    assert limits.maximum_resident_candidate_count == 3
    assert limits.maximum_visitor_candidate_count == 1
    assert limits.maximum_total_candidate_count == 4
    assert limits.maximum_exemplars_per_candidate == 4


def test_policy_search_contains_observed_fail_closed_boundaries(tmp_path: Path) -> None:
    from reid.identity_v2.calibration import _policy_breakpoints

    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.BENCHMARK,
    )
    dataset = build_calibration_dataset(evidence, labels)
    appearances, qualities, ambiguities = _policy_breakpoints(
        dataset,
        slope=12.0,
        midpoint=0.70,
    )
    assert math.nextafter(0.55, math.inf) in appearances
    assert math.nextafter(0.90, math.inf) in qualities
    assert 0.0 in ambiguities


def test_old_ambiguous_label_and_dataset_contracts_reject(tmp_path: Path) -> None:
    evidence, labels = _fixture_files(
        tmp_path, stratum=CalibrationEvidenceStratum.HOUSEHOLD
    )
    old_labels = json.loads(labels.read_text(encoding="utf-8"))
    old_labels["contract_version"] = 1
    old_labels.pop("evidence_stratum")
    old_labels.pop("provenance")
    for row in old_labels["labels"]:
        row["independence_key"] = row.pop("encounter_id")
        row.pop("truth_person_key")
    write_private_json(labels, old_labels)
    with pytest.raises(IdentityCalibrationError, match="invalid evidence label set"):
        build_calibration_dataset(evidence, labels)

    valid_evidence, valid_labels = _fixture_files(
        tmp_path / "valid", stratum=CalibrationEvidenceStratum.HOUSEHOLD
    )
    dataset = build_calibration_dataset(valid_evidence, valid_labels).model_dump(
        mode="json"
    )
    dataset["contract_version"] = 1
    path = write_private_json(tmp_path / "old-dataset.json", dataset)
    with pytest.raises(IdentityCalibrationError, match="invalid calibration dataset"):
        load_calibration_dataset(path)


def test_label_set_and_dataset_reject_duplicate_authority_keys(
    tmp_path: Path,
) -> None:
    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
    )
    dataset = build_calibration_dataset(evidence, labels)
    dataset_path = write_private_json(
        tmp_path / "dataset.json",
        dataset.model_dump(mode="json"),
    )

    for path in (labels, dataset_path):
        canonical = path.read_text(encoding="utf-8")
        marker = '"contract_version":'
        offset = canonical.index(marker)
        path.write_text(
            canonical[:offset] + '"contract_version":1,' + canonical[offset:],
            encoding="utf-8",
        )
        path.chmod(0o600)

    with pytest.raises(IdentityCalibrationError, match="duplicate JSON object key"):
        load_label_set(labels)
    with pytest.raises(IdentityCalibrationError, match="duplicate JSON object key"):
        load_calibration_dataset(dataset_path)


def test_dataset_review_separates_observations_units_encounters_and_strata(
    tmp_path: Path,
) -> None:
    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
        frames_per_encounter=3,
    )
    review = dataset_review(build_calibration_dataset(evidence, labels))
    assert review["evidence_stratum"] == "household"
    train = review["partitions"]["train"]
    assert train["observation_count"] == 60
    assert train["encounter_count"] == 20
    assert train["evidence_unit_count"] == 20
    assert train["zero_error_unknown_upper_95"] > 0.20


def test_owner_cli_review_reports_stratum_and_correlation_units(tmp_path: Path) -> None:
    evidence, labels = _fixture_files(
        tmp_path, stratum=CalibrationEvidenceStratum.HOUSEHOLD
    )
    dataset = build_calibration_dataset(evidence, labels)
    dataset_path = write_private_json(
        tmp_path / "household-dataset.json", dataset.model_dump(mode="json")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "identity_v2_calibrate.py"
            ),
            "validate-dataset",
            "--dataset",
            str(dataset_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["review"]["evidence_stratum"] == "household"
    assert payload["review"]["partitions"]["holdout"]["encounter_count"] == 40


def test_authority_metrics_are_encounter_worst_case_not_frame_rates(
    tmp_path: Path,
) -> None:
    from reid.identity_v2.calibration import _metrics

    evidence, labels = _fixture_files(
        tmp_path,
        stratum=CalibrationEvidenceStratum.HOUSEHOLD,
        frames_per_encounter=5,
        one_bad_unknown_holdout_frame=True,
    )
    dataset = build_calibration_dataset(evidence, labels)
    metrics = _metrics(
        dataset,
        CalibrationPartition.HOLDOUT,
        OpenSetScorer().policy,
    )
    assert metrics.unknown_observation_count == 100
    assert metrics.unknown_encounter_count == 20
    assert metrics.false_accept_encounter_count == 1
    assert metrics.far == pytest.approx(1.0 / 20.0)


def test_calibration_is_deterministic_two_stratum_and_runtime_accepted(
    tmp_path: Path,
) -> None:
    benchmark, household, benchmark_path, household_path = _dataset_files(tmp_path)
    first = _calibrate(benchmark, household, benchmark_path, household_path)
    second = _calibrate(benchmark, household, benchmark_path, household_path)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.contract_version == 2
    assert first.benchmark_holdout_metrics.far == 0.0
    assert first.benchmark_holdout_metrics.far_upper_confidence_bound == pytest.approx(
        one_sided_binomial_upper_confidence_bound(0, 300)
    )
    assert first.benchmark_holdout_metrics.far_upper_confidence_bound < 0.01
    assert (
        first.benchmark_holdout_metrics.misidentification_upper_confidence_bound < 0.01
    )
    assert first.household_holdout_metrics.known_encounter_count == 20
    assert first.household_holdout_metrics.unknown_encounter_count == 20
    assert first.household_holdout_metrics.false_accept_encounter_count == 0
    assert first.household_holdout_metrics.misidentification_encounter_count == 0
    assert first.household_holdout_metrics.far_upper_confidence_bound > 0.10
    assert first.policy.appearance_floor >= first.benchmark_policy.appearance_floor
    assert first.policy.quality_floor >= first.benchmark_policy.quality_floor
    assert (
        first.policy.ambiguity_margin_floor
        >= first.benchmark_policy.ambiguity_margin_floor
    )
    assert (
        first.policy.resident_prior_bonus == first.benchmark_policy.resident_prior_bonus
    )
    assert (
        first.household_holdout_metrics.resident_prior_rejection_rescue_encounter_count
        == 0
    )

    artifact_path = write_private_json(
        tmp_path / "calibration.json", first.model_dump(mode="json")
    )
    policy = _load_open_set_policy(
        environ=_artifact_env(artifact_path),
        repo_root=tmp_path,
        model_fingerprint=MODEL_SHA,
        model_layer=MODEL_LAYER,
        embedding_dim=DIMENSION,
        model_semantic_profile_sha256="1" * 64,
    )
    assert policy.calibration_status == "artifact_backed"
    assert policy.calibration_artifact_id == f"sha256:{sha256_file(artifact_path)}"


def test_scoring_artifact_rejects_duplicate_authority_keys(tmp_path: Path) -> None:
    path = tmp_path / "scoring.json"
    path.write_text(
        '{"contract_version":1,"contract_version":2}',
        encoding="utf-8",
    )
    path.chmod(0o600)
    env = {
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT": str(path),
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
    }

    with pytest.raises(Exception, match="duplicate JSON object key"):
        _load_open_set_policy(
            environ=env,
            repo_root=tmp_path,
            model_fingerprint=MODEL_SHA,
            model_layer=MODEL_LAYER,
            embedding_dim=DIMENSION,
            model_semantic_profile_sha256="1" * 64,
        )


def test_calibration_rejects_cross_stratum_truth_person_overlap(tmp_path: Path) -> None:
    benchmark, household, benchmark_path, household_path = _dataset_files(tmp_path)
    payload = household.model_dump(mode="json")
    sample = next(
        row
        for row in payload["samples"]
        if row["label"]["partition"] == "train"
        and row["label"]["truth_kind"] == "unknown"
    )
    old_person_key = sample["label"]["truth_person_key"]
    sample["label"]["truth_person_key"] = "unknown-train-0"
    for unit in payload["evidence_units"]:
        if unit["truth_person_key"] == old_person_key:
            unit["truth_person_key"] = "unknown-train-0"
    overlapping = IdentityCalibrationDataset.model_validate(payload)
    with pytest.raises(IdentityCalibrationError, match="truth people overlap"):
        _calibrate(benchmark, overlapping, benchmark_path, household_path)


def test_artifact_rejects_claimed_pass_when_confidence_or_monotonicity_fails(
    tmp_path: Path,
) -> None:
    benchmark, household, benchmark_path, household_path = _dataset_files(tmp_path)
    artifact = _calibrate(
        benchmark, household, benchmark_path, household_path
    ).model_dump(mode="json")
    metrics = artifact["benchmark_holdout_metrics"]
    metrics["false_accept_encounter_count"] = 1
    metrics["false_accept_person_count"] = 1
    metrics["false_accept_challenge_person_count"] = 1
    metrics["far"] = 1.0 / metrics["unknown_encounter_count"]
    metrics["person_far"] = 1.0 / metrics["unknown_person_count"]
    metrics["unknown_rejection_rate"] = 1.0 - metrics["far"]
    metrics["far_upper_confidence_bound"] = one_sided_binomial_upper_confidence_bound(
        1, metrics["unknown_encounter_count"]
    )
    bad_path = write_private_json(tmp_path / "bad-bound.json", artifact)
    with pytest.raises(Exception, match="benchmark holdout FAR upper bound"):
        _load_open_set_policy(
            environ=_artifact_env(bad_path),
            repo_root=tmp_path,
            model_fingerprint=MODEL_SHA,
            model_layer=MODEL_LAYER,
            embedding_dim=DIMENSION,
            model_semantic_profile_sha256="1" * 64,
        )

    artifact = _calibrate(
        benchmark, household, benchmark_path, household_path
    ).model_dump(mode="json")
    artifact["policy"]["calibrated_confidence_floor"] = (
        artifact["benchmark_policy"]["calibrated_confidence_floor"] - 0.01
    )
    bad_path = write_private_json(tmp_path / "bad-local-policy.json", artifact)
    with pytest.raises(Exception, match="may not lower a benchmark rejection gate"):
        _load_open_set_policy(
            environ=_artifact_env(bad_path),
            repo_root=tmp_path,
            model_fingerprint=MODEL_SHA,
            model_layer=MODEL_LAYER,
            embedding_dim=DIMENSION,
            model_semantic_profile_sha256="1" * 64,
        )


def test_v1_artifact_and_weaker_product_gates_reject(tmp_path: Path) -> None:
    benchmark, household, benchmark_path, household_path = _dataset_files(tmp_path)
    artifact = _calibrate(
        benchmark, household, benchmark_path, household_path
    ).model_dump(mode="json")
    artifact["contract_version"] = 1
    old_path = write_private_json(tmp_path / "old-artifact.json", artifact)
    with pytest.raises(Exception, match="contract is invalid"):
        _load_open_set_policy(
            environ=_artifact_env(old_path),
            repo_root=tmp_path,
            model_fingerprint=MODEL_SHA,
            model_layer=MODEL_LAYER,
            embedding_dim=DIMENSION,
            model_semantic_profile_sha256="1" * 64,
        )

    with pytest.raises(IdentityCalibrationError, match="weakens the product maximum"):
        calibrate_datasets(
            benchmark,
            household,
            benchmark_dataset_sha256=sha256_file(benchmark_path),
            household_dataset_sha256=sha256_file(household_path),
            generated_at_us=1,
            generator_revision="test",
            max_benchmark_holdout_far_upper_confidence_bound=0.011,
        )

    with pytest.raises(IdentityCalibrationError, match="distinct corpora"):
        calibrate_datasets(
            benchmark,
            household,
            benchmark_dataset_sha256=sha256_file(benchmark_path),
            household_dataset_sha256=sha256_file(benchmark_path),
            generated_at_us=1,
            generator_revision="test",
        )

    cloned_payload = benchmark.model_dump(mode="json")
    cloned_payload["evidence_stratum"] = "household"
    cloned_payload["evidence_sha256"] = "d" * 64
    cloned_payload["labels_sha256"] = "e" * 64
    cloned_payload["provenance"]["source_manifest_sha256"] = "f" * 64
    cloned_household = IdentityCalibrationDataset.model_validate(cloned_payload)
    with pytest.raises(IdentityCalibrationError, match="evidence events overlap"):
        calibrate_datasets(
            benchmark,
            cloned_household,
            benchmark_dataset_sha256=sha256_file(benchmark_path),
            household_dataset_sha256="9" * 64,
            generated_at_us=1,
            generator_revision="test",
        )
