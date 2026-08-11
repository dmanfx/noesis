from __future__ import annotations

import json
from pathlib import Path

import pytest

from reid.identity_v2.coordinator import (
    FrameBatchResult,
    PrimitiveFrameObservation,
    PublicIdentityOverlay,
)
from reid.identity_v2.evidence import (
    IdentityEvidenceError,
    IdentityEvidenceRecorder,
    evidence_checkpoint_path,
    load_evidence_checkpoint,
)
from reid.identity_v2.models import (
    CandidateEvidence,
    IdentityDecision,
    IdentityKind,
    TrackletObservation,
)
from reid.identity_v2.scoring import OpenSetPolicy, OpenSetScorer
from reid.identity_v2.store import EnrollmentObservationKey, PurgeResult

MODEL_SHA = "a" * 64
MODEL_LAYER = "features"
DIMENSION = 4


def _append_unknown(
    recorder: IdentityEvidenceRecorder,
    *,
    frame: int,
    observed_at_us: int,
) -> None:
    tracker = f"tracker-{frame}"
    observation_id = f"observation-{frame}"
    observation = PrimitiveFrameObservation(
        run_id="run-a",
        camera_id="camera-a",
        tracker_id=tracker,
        frame_id=frame,
        observation_id=observation_id,
        quality=0.9,
        embedding=(1.0, 0.0, 0.0, 0.0),
    )
    key = EnrollmentObservationKey(
        run_id="run-a",
        camera_id="camera-a",
        tracker_id=tracker,
        frame_id=frame,
        observation_id=observation_id,
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
        observed_at_us=observed_at_us,
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


def test_resident_prior_never_rescues_below_floor_or_ambiguity() -> None:
    policy = OpenSetPolicy(
        appearance_floor=0.70,
        ambiguity_margin_floor=0.03,
        resident_prior_bonus=0.05,
        resident_prior_cap=0.05,
    )
    scorer = OpenSetScorer(policy)
    below = scorer.score_tracklet(
        TrackletObservation(
            tracklet_id="below",
            quality=0.95,
            candidates=(
                CandidateEvidence(
                    "resident:a", IdentityKind.RESIDENT, raw_similarity=0.699
                ),
            ),
        )
    )
    assert below.assignment_eligible is False
    assert below.reason == "appearance_below_floor"
    assert below.candidates[0].prior_contribution == 0.0

    ambiguous = scorer.score_tracklet(
        TrackletObservation(
            tracklet_id="ambiguous",
            quality=0.95,
            candidates=(
                CandidateEvidence("resident:a", IdentityKind.RESIDENT, 0.82),
                CandidateEvidence("visitor:b", IdentityKind.VISITOR, 0.819),
            ),
        )
    )
    assert ambiguous.assignment_eligible is False
    assert ambiguous.reason == "ambiguous"


def test_evidence_restart_rejects_incomplete_final_record(tmp_path: Path) -> None:
    now = [100.0]
    path = tmp_path / "private" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        clock=lambda: now[0],
    )
    _append_unknown(recorder, frame=1, observed_at_us=100_000_000)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    with pytest.raises(IdentityEvidenceError, match="incomplete"):
        IdentityEvidenceRecorder(
            path,
            session_id="session-a",
            source="shadow",
            runtime="test",
            clock=lambda: now[0],
        )


def test_evidence_reader_rejects_duplicate_key_hidden_embedding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        clock=lambda: 100.0,
    )
    _append_unknown(recorder, frame=1, observed_at_us=100_000_000)
    canonical = path.read_text(encoding="utf-8")
    marker = '"candidates":'
    offset = canonical.index(marker)
    hidden = '"candidates":[{"embedding":[' + ",".join(["0.25"] * 256) + "]}],"
    ambiguous = (canonical[:offset] + hidden + canonical[offset:]).encode("utf-8")

    with pytest.raises(IdentityEvidenceError, match="duplicate JSON object key"):
        IdentityEvidenceRecorder._validate_line(ambiguous, line_number=1)


@pytest.mark.parametrize("variant", ("whitespace", "key_order"))
def test_evidence_reader_rejects_noncanonical_exact_bytes(
    tmp_path: Path,
    variant: str,
) -> None:
    path = tmp_path / "private" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        clock=lambda: 100.0,
    )
    _append_unknown(recorder, frame=1, observed_at_us=100_000_000)
    canonical = path.read_bytes()
    if variant == "whitespace":
        mutated = b"  " + canonical
    else:
        document = json.loads(canonical)
        reordered = dict(reversed(tuple(document.items())))
        mutated = (
            json.dumps(
                reordered,
                sort_keys=False,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    assert mutated != canonical

    with pytest.raises(IdentityEvidenceError, match="not canonical JSON"):
        IdentityEvidenceRecorder._validate_line(mutated, line_number=1)


def test_evidence_checkpoint_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "private" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        clock=lambda: 100.0,
    )
    _append_unknown(recorder, frame=1, observed_at_us=100_000_000)
    checkpoint_path = evidence_checkpoint_path(path)
    canonical = checkpoint_path.read_text(encoding="utf-8")
    marker = '"next_sequence":'
    offset = canonical.index(marker)
    checkpoint_path.write_text(
        canonical[:offset] + '"next_sequence":999,' + canonical[offset:],
        encoding="utf-8",
    )
    checkpoint_path.chmod(0o600)

    with pytest.raises(IdentityEvidenceError, match="duplicate JSON object key"):
        load_evidence_checkpoint(path)


@pytest.mark.parametrize("variant", ("whitespace", "key_order"))
def test_evidence_checkpoint_rejects_noncanonical_exact_bytes(
    tmp_path: Path,
    variant: str,
) -> None:
    path = tmp_path / "private" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        clock=lambda: 100.0,
    )
    _append_unknown(recorder, frame=1, observed_at_us=100_000_000)
    checkpoint_path = evidence_checkpoint_path(path)
    canonical = checkpoint_path.read_bytes()
    if variant == "whitespace":
        mutated = b"  " + canonical
    else:
        document = json.loads(canonical)
        reordered = dict(reversed(tuple(document.items())))
        mutated = (
            json.dumps(
                reordered,
                sort_keys=False,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    assert mutated != canonical
    checkpoint_path.write_bytes(mutated)
    checkpoint_path.chmod(0o600)

    with pytest.raises(IdentityEvidenceError, match="not canonical JSON"):
        load_evidence_checkpoint(path)


def test_evidence_retention_is_bounded_deterministic_and_restart_safe(
    tmp_path: Path,
) -> None:
    now = [100.0]
    path = tmp_path / "private" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        max_records=2,
        max_bytes=1024 * 1024,
        max_age_s=10.0,
        prune_interval_s=1.0,
        clock=lambda: now[0],
    )
    for frame in (1, 2, 3):
        now[0] = 99.0 + frame
        _append_unknown(
            recorder,
            frame=frame,
            observed_at_us=int(now[0] * 1_000_000),
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["frame_id"] for line in lines] == [2, 3]
    health = recorder.health()
    assert health.recorded_event_count == 2
    assert health.pruned_event_count == 1
    assert health.retained_bytes == path.stat().st_size <= health.max_bytes

    restarted = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        max_records=2,
        max_bytes=1024 * 1024,
        max_age_s=10.0,
        prune_interval_s=1.0,
        clock=lambda: now[0],
    )
    assert restarted.health().recorded_event_count == 2

    now[0] = 120.0
    expired = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="test",
        max_records=2,
        max_bytes=1024 * 1024,
        max_age_s=10.0,
        prune_interval_s=1.0,
        clock=lambda: now[0],
    )
    assert expired.health().recorded_event_count == 0
    assert expired.health().pruned_event_count == 2
    assert path.read_bytes() == b""


def test_evidence_writer_refuses_shared_parent_and_links_without_chmod(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    target = shared / "evidence.jsonl"
    with pytest.raises(IdentityEvidenceError, match="directory mode must be 0700"):
        IdentityEvidenceRecorder(
            target,
            session_id="session-a",
            source="shadow",
            runtime="test",
        )
    assert shared.stat().st_mode & 0o777 == 0o755
    assert not target.exists()

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    real = private / "real.jsonl"
    real.touch(mode=0o600)
    linked = private / "linked.jsonl"
    linked.symlink_to(real)
    with pytest.raises(IdentityEvidenceError, match="symlink"):
        IdentityEvidenceRecorder(
            linked,
            session_id="session-a",
            source="shadow",
            runtime="test",
        )
    assert real.stat().st_mode & 0o777 == 0o600
