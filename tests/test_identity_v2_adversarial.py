from __future__ import annotations

import json
import threading
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


def test_async_evidence_writer_is_ordered_and_flushes_durably(tmp_path: Path) -> None:
    path = tmp_path / "async" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
        queue_capacity=4,
    )
    try:
        _append_unknown(recorder, frame=1, observed_at_us=1)
        _append_unknown(recorder, frame=2, observed_at_us=2)
        assert recorder.health().pending_event_count >= 0
        recorder.flush()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["sequence"] for row in rows] == [0, 1]
        assert recorder.health().pending_event_count == 0
        assert recorder.health().failed is False
    finally:
        recorder.close()


def test_async_health_keeps_durability_boundary_explicit_while_writer_is_blocked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "async" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
    )
    started = threading.Event()
    release = threading.Event()
    original_append = recorder._append_payload

    def blocked_append(payload: bytes) -> None:
        started.set()
        assert release.wait(timeout=30.0)
        original_append(payload)

    recorder._append_payload = blocked_append  # type: ignore[method-assign]
    try:
        _append_unknown(recorder, frame=1, observed_at_us=1)
        assert started.wait(timeout=2.0)
        health = recorder.health()
        assert health.recorded_event_count == 0
        assert health.last_observed_at_us is None
        assert health.pending_event_count == 1
        release.set()
        recorder.flush()
        assert recorder.health().recorded_event_count == 1
    finally:
        release.set()
        recorder.close()


def test_async_reservations_reconcile_after_retention(tmp_path: Path) -> None:
    path = tmp_path / "async" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
        max_records=2,
    )
    try:
        for frame in range(8):
            _append_unknown(recorder, frame=frame, observed_at_us=frame + 1)
        recorder.flush()
        assert recorder.health().recorded_event_count == 2
        assert len(recorder._reserved_event_ids) == 2
    finally:
        recorder.close()


def test_async_writer_failure_is_health_visible_and_next_append_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "async" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
    )

    def fail_append(_payload: bytes) -> None:
        raise OSError("synthetic evidence disk failure")

    recorder._append_payload = fail_append  # type: ignore[method-assign]
    try:
        _append_unknown(recorder, frame=1, observed_at_us=1)
        for _ in range(100):
            if recorder.health().failed:
                break
            threading.Event().wait(0.001)
        health = recorder.health()
        assert health.failed is True
        assert "synthetic evidence disk failure" in (health.last_error or "")
        with pytest.raises(IdentityEvidenceError, match="synthetic evidence"):
            _append_unknown(recorder, frame=2, observed_at_us=2)
    finally:
        with pytest.raises(IdentityEvidenceError, match="synthetic evidence"):
            recorder.close()


def test_async_close_closes_admission_and_is_idempotent(tmp_path: Path) -> None:
    recorder = IdentityEvidenceRecorder(
        tmp_path / "async" / "evidence.jsonl",
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
    )
    recorder.close()
    recorder.close()
    with pytest.raises(IdentityEvidenceError, match="closed"):
        _append_unknown(recorder, frame=1, observed_at_us=1)


def test_async_close_admission_wins_against_concurrent_append(tmp_path: Path) -> None:
    recorder = IdentityEvidenceRecorder(
        tmp_path / "async" / "evidence.jsonl",
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
    )
    started = threading.Event()
    release = threading.Event()
    original_append = recorder._append_payload

    def blocked_append(payload: bytes) -> None:
        started.set()
        assert release.wait(timeout=30.0)
        original_append(payload)

    recorder._append_payload = blocked_append  # type: ignore[method-assign]
    _append_unknown(recorder, frame=1, observed_at_us=1)
    assert started.wait(timeout=2.0)
    close_done = threading.Event()

    def close_recorder() -> None:
        try:
            recorder.close()
        finally:
            close_done.set()

    closer = threading.Thread(target=close_recorder)
    closer.start()
    for _ in range(100):
        with recorder._lock:
            if recorder._admission_closed:
                break
        threading.Event().wait(0.001)
    with pytest.raises(IdentityEvidenceError, match="closed"):
        _append_unknown(recorder, frame=2, observed_at_us=2)
    release.set()
    closer.join(timeout=5.0)
    assert close_done.is_set()


def test_async_close_surfaces_failure_that_occurs_during_drain(tmp_path: Path) -> None:
    recorder = IdentityEvidenceRecorder(
        tmp_path / "async" / "evidence.jsonl",
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
        queue_capacity=2,
    )
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0
    original_append = recorder._append_payload

    def block_then_fail(payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert release_first.wait(timeout=30.0)
            original_append(payload)
            return
        raise OSError("synthetic final evidence failure")

    recorder._append_payload = block_then_fail  # type: ignore[method-assign]
    _append_unknown(recorder, frame=1, observed_at_us=1)
    assert first_started.wait(timeout=2.0)
    _append_unknown(recorder, frame=2, observed_at_us=2)
    outcome: list[BaseException] = []
    done = threading.Event()

    def close_recorder() -> None:
        try:
            recorder.close()
        except BaseException as exc:  # capture the required close failure
            outcome.append(exc)
        finally:
            done.set()

    closer = threading.Thread(target=close_recorder)
    closer.start()
    release_first.set()
    closer.join(timeout=5.0)
    assert done.is_set(), "close hung while writer failed during drain"
    assert outcome and isinstance(outcome[0], IdentityEvidenceError)
    assert "synthetic final evidence failure" in str(outcome[0])


def test_async_evidence_queue_drop_is_explicit_and_chain_remains_contiguous(
    tmp_path: Path,
) -> None:
    path = tmp_path / "async" / "evidence.jsonl"
    recorder = IdentityEvidenceRecorder(
        path,
        session_id="session-a",
        source="shadow",
        runtime="replay",
        async_mode=True,
        queue_capacity=2,
    )
    started = threading.Event()
    release = threading.Event()
    original_append = recorder._append_payload

    def blocked_append(payload: bytes) -> None:
        started.set()
        assert release.wait(timeout=30.0)
        original_append(payload)

    recorder._append_payload = blocked_append  # type: ignore[method-assign]
    try:
        _append_unknown(recorder, frame=1, observed_at_us=1)
        assert started.wait(timeout=2.0)
        _append_unknown(recorder, frame=2, observed_at_us=2)
        _append_unknown(recorder, frame=3, observed_at_us=3)
        health = recorder.health()
        assert health.dropped_event_count == 1
        assert health.failed is False
        release.set()
        recorder.flush()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["sequence"] for row in rows] == [0, 1]
        assert load_evidence_checkpoint(path).next_sequence == 2
    finally:
        release.set()
        recorder.close()
