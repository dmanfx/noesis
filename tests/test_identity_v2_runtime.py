from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from reid.identity_v2 import (
    DuplicateResidentError,
    EnrollmentObservationKey,
    EnrollmentProposalConflict,
    EnrollmentProposalExpired,
    IdentityKind,
    IdentityStore,
    IdentityV2Runtime,
    ModelProfileMismatch,
    ObservationEvidenceConflict,
    ObservationEvidenceConsumed,
    ObservationEvidenceExpired,
    ObservationEvidenceUnavailable,
    OpenSetPolicy,
    RuntimeObservation,
    resident_subject_id,
    visitor_subject_id,
)

FINGERPRINT = "identity-v2-runtime-test"
DIMENSION = 4


def _key(
    observation_id: str,
    *,
    run: str = "run-1",
    camera: str = "camera-a",
    tracker: str = "tracker-7",
    frame: int = 100,
) -> EnrollmentObservationKey:
    return EnrollmentObservationKey(
        run_id=run,
        camera_id=camera,
        tracker_id=tracker,
        frame_id=frame,
        observation_id=observation_id,
    )


def _observation(
    observation_id: str,
    vector=(1.0, 0.0, 0.0, 0.0),
    *,
    run: str = "run-1",
    camera: str = "camera-a",
    tracker: str = "tracker-7",
    frame: int = 100,
    quality: float = 0.95,
) -> RuntimeObservation:
    return RuntimeObservation(
        key=_key(
            observation_id,
            run=run,
            camera=camera,
            tracker=tracker,
            frame=frame,
        ),
        quality=quality,
        embedding=tuple(vector),
    )


def _runtime(tmp_path, *, now: float = 10.0, **runtime_kwargs):
    store = IdentityStore(tmp_path / "identity.sqlite3")
    runtime = IdentityV2Runtime(
        store,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        clock=lambda: now,
        **runtime_kwargs,
    )
    return store, runtime


def _enroll(
    runtime: IdentityV2Runtime, observation: RuntimeObservation, *, name="Alice", sid=1
):
    runtime.ingest_observations((observation,), now=10.0)
    proposal = runtime.propose_enrollment_from_cache(
        observation.key,
        display_name=name,
        compatibility_sid=sid,
        now=10.0,
    )
    confirmation = runtime.confirm_enrollment(
        proposal.proposal.proposal_uuid,
        expected_key=observation.key,
        evidence_digest=proposal.proposal.evidence_digest,
        now=11.0,
    )
    return proposal, confirmation


def test_exact_enrollment_is_stale_safe_replay_safe_and_restart_safe(tmp_path) -> None:
    db_path = tmp_path / "identity.sqlite3"
    store = IdentityStore(db_path)
    runtime = IdentityV2Runtime(
        store,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        clock=lambda: 10.0,
    )
    observation = _observation("observation-a")
    before = runtime.resolve_batch((observation,), now=10.0)[0]
    assert before.is_unknown
    assert before.decision.reason == "no_candidates"

    runtime.ingest_observations((observation,), now=10.0)
    proposal = runtime.propose_enrollment_from_cache(
        observation.key,
        display_name="Alice",
        compatibility_sid=1,
        ttl_s=30.0,
        now=10.0,
    )
    assert not proposal.idempotent
    assert proposal.proposal.effective_state(now=10.0) == "pending"
    with pytest.raises(ObservationEvidenceConsumed, match="already consumed"):
        runtime.propose_enrollment_from_cache(
            observation.key,
            display_name="Alice",
            compatibility_sid=1,
            ttl_s=30.0,
            now=12.0,
        )

    with pytest.raises(ObservationEvidenceConsumed, match="already consumed"):
        runtime.propose_enrollment_from_cache(
            observation.key,
            display_name="Different Person",
            compatibility_sid=2,
            now=12.0,
        )
    with pytest.raises(EnrollmentProposalConflict, match="exact observation evidence"):
        runtime.confirm_enrollment(
            proposal.proposal.proposal_uuid,
            expected_key=_key("wrong-observation"),
            evidence_digest=proposal.proposal.evidence_digest,
            now=12.0,
        )
    with pytest.raises(EnrollmentProposalConflict, match="exact observation evidence"):
        runtime.confirm_enrollment(
            proposal.proposal.proposal_uuid,
            expected_key=observation.key,
            evidence_digest="wrong-digest",
            now=12.0,
        )

    confirmation = runtime.confirm_enrollment(
        proposal.proposal.proposal_uuid,
        expected_key=observation.key,
        evidence_digest=proposal.proposal.evidence_digest,
        now=12.0,
    )
    assert not confirmation.idempotent
    assert confirmation.anchor.role == "enrollment_anchor"
    replay = runtime.confirm_enrollment(
        proposal.proposal.proposal_uuid,
        expected_key=observation.key,
        evidence_digest=proposal.proposal.evidence_digest,
        now=13.0,
    )
    assert replay.idempotent
    assert replay.anchor.exemplar_uuid == confirmation.anchor.exemplar_uuid
    assert runtime.health(now=13.0).enrollment_anchor_count == 1
    resident_uuid = confirmation.resident.resident_uuid
    store.close()

    with IdentityStore(db_path) as reopened_store:
        reopened = IdentityV2Runtime(
            reopened_store,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            clock=lambda: 20.0,
        )
        decision = reopened.resolve_batch((observation,), now=20.0)[0]
        assert not decision.is_unknown
        assert decision.subject is not None
        assert decision.subject.subject_id == resident_subject_id(resident_uuid)
        assert reopened.health(now=20.0).confirmed_enrollment_proposals == 1
        with pytest.raises(ObservationEvidenceConsumed):
            reopened.propose_enrollment_from_cache(
                observation.key,
                display_name="Alice",
                compatibility_sid=1,
                now=20.0,
            )


def test_observation_cache_is_bounded_stale_safe_and_profile_checked(tmp_path) -> None:
    store, runtime = _runtime(
        tmp_path,
        observation_cache_ttl_s=2.0,
        observation_cache_max_entries=2,
    )
    try:
        first = _observation("cache-1", frame=1)
        second = _observation("cache-2", frame=2)
        third = _observation("cache-3", frame=3)
        with pytest.raises(ObservationEvidenceUnavailable):
            runtime.propose_enrollment_from_cache(
                first.key,
                display_name="Alice",
                compatibility_sid=1,
                now=10.0,
            )
        runtime.ingest_observations((first, second, third), now=10.0)
        health = runtime.observation_cache_health(now=10.0)
        assert health.entry_count == 2
        assert health.capacity_evictions == 1
        with pytest.raises(ObservationEvidenceUnavailable):
            runtime.propose_enrollment_from_cache(
                first.key,
                display_name="Alice",
                compatibility_sid=1,
                now=10.0,
            )
        with pytest.raises(ObservationEvidenceConflict, match="different evidence"):
            runtime.ingest_observations(
                (
                    _observation(
                        "cache-3",
                        vector=(0.0, 1.0, 0.0, 0.0),
                        frame=3,
                    ),
                ),
                now=10.5,
            )
        with pytest.raises(ModelProfileMismatch, match="fingerprint"):
            runtime.ingest_observations(
                (second,),
                model_fingerprint="wrong-profile",
                now=10.5,
            )
        with pytest.raises(ObservationEvidenceExpired):
            runtime.propose_enrollment_from_cache(
                second.key,
                display_name="Bob",
                compatibility_sid=2,
                now=12.0,
            )
        assert runtime.observation_cache_health(now=12.0).entry_count == 0
    finally:
        store.close()


def test_concurrent_proposal_consumes_server_evidence_exactly_once(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        observation = _observation("proposal-race")
        runtime.ingest_observations((observation,), now=10.0)

        def propose(_index):
            try:
                return runtime.propose_enrollment_from_cache(
                    observation.key,
                    display_name="Alice",
                    compatibility_sid=1,
                    now=10.0,
                )
            except ObservationEvidenceConsumed as exc:
                return exc

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = tuple(executor.map(propose, range(16)))
        proposals = [row for row in results if not isinstance(row, Exception)]
        consumed = [
            row for row in results if isinstance(row, ObservationEvidenceConsumed)
        ]
        assert len(proposals) == 1
        assert len(consumed) == 15
        assert {row.proposal_uuid for row in consumed} == {
            proposals[0].proposal.proposal_uuid
        }
        assert runtime.observation_cache_health(now=10.0).entry_count == 0
    finally:
        store.close()


def test_expired_proposal_cannot_be_confirmed_and_retention_purges_evidence(
    tmp_path,
) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        observation = _observation("expires")
        runtime.ingest_observations((observation,), now=10.0)
        proposal = runtime.propose_enrollment_from_cache(
            observation.key,
            display_name="Alice",
            compatibility_sid=1,
            ttl_s=2.0,
            now=10.0,
        )
        with pytest.raises(EnrollmentProposalExpired):
            runtime.confirm_enrollment(
                proposal.proposal.proposal_uuid,
                expected_key=observation.key,
                evidence_digest=proposal.proposal.evidence_digest,
                now=12.0,
            )
        assert runtime.health(now=12.0).pending_enrollment_proposals == 0
        purge = runtime.run_retention(now=12.0)
        assert purge.enrollment_proposals == 1
        with pytest.raises(KeyError):
            runtime.get_enrollment_proposal(proposal.proposal.proposal_uuid)
    finally:
        store.close()


def test_duplicate_normalized_name_requires_explicit_existing_resident_update(
    tmp_path,
) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        _, first = _enroll(runtime, _observation("first"))
        second_observation = _observation("second", frame=200)
        runtime.ingest_observations((second_observation,), now=20.0)
        with pytest.raises(DuplicateResidentError, match="update_resident_uuid"):
            runtime.propose_enrollment_from_cache(
                second_observation.key,
                display_name="  ALICE ",
                compatibility_sid=2,
                now=20.0,
            )
        proposal = runtime.propose_enrollment_from_cache(
            second_observation.key,
            display_name="  ALICE ",
            update_resident_uuid=first.resident.resident_uuid,
            now=20.0,
        )
        assert proposal.proposal.action == "add_anchor"
        confirmed = runtime.confirm_enrollment(
            proposal.proposal.proposal_uuid,
            expected_key=second_observation.key,
            evidence_digest=proposal.proposal.evidence_digest,
            now=21.0,
        )
        assert confirmed.resident.resident_uuid == first.resident.resident_uuid
        assert runtime.health(now=21.0).enrollment_anchor_count == 2

        _, bob = _enroll(
            runtime,
            _observation("bob", tracker="tracker-9", frame=300),
            name="Bob",
            sid=2,
        )
        with pytest.raises(DuplicateResidentError):
            runtime.update_resident_display_name(
                bob.resident.resident_uuid,
                display_name="alice",
            )
    finally:
        store.close()


def test_open_set_negative_stays_unknown_and_hot_resolution_avoids_store_reads(
    tmp_path,
    monkeypatch,
) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        _, confirmation = _enroll(runtime, _observation("enroll"))
        monkeypatch.setattr(
            store,
            "load_resident_gallery",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cold store read")
            ),
        )
        monkeypatch.setattr(
            store,
            "load_active_visitor_galleries",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cold store read")
            ),
        )

        positive = runtime.resolve_batch(
            (_observation("positive", frame=400),), now=30.0
        )[0]
        assert positive.subject is not None
        assert positive.subject.resident_uuid == confirmation.resident.resident_uuid
        negative = runtime.resolve_batch(
            (_observation("negative", vector=(0.0, 1.0, 0.0, 0.0), frame=401),),
            now=30.0,
        )[0]
        assert negative.is_unknown
        assert negative.decision.reason == "appearance_below_floor"
    finally:
        store.close()


def test_visitor_subject_identity_includes_uuid_and_generation_across_reuse_and_restart(
    tmp_path,
) -> None:
    db_path = tmp_path / "identity.sqlite3"
    store = IdentityStore(db_path)
    runtime = IdentityV2Runtime(
        store,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        clock=lambda: 10.0,
    )
    first = runtime.open_visitor_session(slot=1000, ttl_s=30.0, now=10.0)
    runtime.record_visitor_observation(
        first.session_uuid,
        _observation("visitor-first", vector=(0.0, 1.0, 0.0, 0.0)),
        now=11.0,
    )
    first_decision = runtime.resolve_batch(
        (_observation("match-first", vector=(0.0, 1.0, 0.0, 0.0), frame=101),),
        now=12.0,
    )[0]
    assert first_decision.subject is not None
    assert first_decision.subject.identity_kind is IdentityKind.VISITOR
    assert first_decision.subject.subject_id == visitor_subject_id(
        first.session_uuid, 1
    )
    runtime.release_visitor_session(first.session_uuid, now=13.0)

    second = runtime.open_visitor_session(slot=1000, ttl_s=30.0, now=14.0)
    assert second.generation == 2
    assert second.session_uuid != first.session_uuid
    assert runtime.resolve_batch(
        (_observation("no-gallery", vector=(0.0, 1.0, 0.0, 0.0), frame=102),),
        now=14.0,
    )[0].is_unknown
    runtime.record_visitor_observation(
        second.session_uuid,
        _observation("visitor-second", vector=(0.0, 1.0, 0.0, 0.0), frame=103),
        now=15.0,
    )
    store.close()

    with IdentityStore(db_path) as reopened_store:
        reopened = IdentityV2Runtime(
            reopened_store,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            clock=lambda: 16.0,
        )
        decision = reopened.resolve_batch(
            (_observation("restart-match", vector=(0.0, 1.0, 0.0, 0.0), frame=104),),
            now=16.0,
        )[0]
        assert decision.subject is not None
        assert decision.subject.subject_id == visitor_subject_id(second.session_uuid, 2)
        assert first_decision.subject.subject_id != decision.subject.subject_id


def test_calibrated_gallery_envelope_rejects_growth_before_store_mutation(
    tmp_path,
) -> None:
    policy = OpenSetPolicy(
        maximum_resident_candidates=1,
        maximum_visitor_candidates=1,
        maximum_total_candidates=1,
        maximum_exemplars_per_candidate=1,
    )
    store, runtime = _runtime(tmp_path, policy=policy)
    try:
        _proposal, confirmation = _enroll(runtime, _observation("resident-a"))
        assert runtime.health(now=12.0).resident_count == 1
        assert runtime.health(now=12.0).enrollment_anchor_count == 1

        second = _observation("resident-b", frame=101)
        runtime.ingest_observations((second,), now=12.0)
        second_proposal = runtime.propose_enrollment_from_cache(
            second.key,
            display_name="Bob",
            compatibility_sid=2,
            now=12.0,
        )
        with pytest.raises(RuntimeError, match="before mutation: resident candidates"):
            runtime.confirm_enrollment(
                second_proposal.proposal.proposal_uuid,
                expected_key=second.key,
                evidence_digest=second_proposal.proposal.evidence_digest,
                now=13.0,
            )
        assert runtime.health(now=13.0).resident_count == 1
        assert (
            second_proposal.proposal.proposal_uuid
            == runtime.get_enrollment_proposal(
                second_proposal.proposal.proposal_uuid
            ).proposal_uuid
        )

        anchor = _observation("resident-a-second-anchor", frame=102)
        runtime.ingest_observations((anchor,), now=14.0)
        anchor_proposal = runtime.propose_enrollment_from_cache(
            anchor.key,
            display_name="Alice",
            update_resident_uuid=confirmation.resident.resident_uuid,
            now=14.0,
        )
        with pytest.raises(RuntimeError, match="before mutation: exemplars"):
            runtime.confirm_enrollment(
                anchor_proposal.proposal.proposal_uuid,
                expected_key=anchor.key,
                evidence_digest=anchor_proposal.proposal.evidence_digest,
                now=15.0,
            )
        assert runtime.health(now=15.0).enrollment_anchor_count == 1

        visitor = runtime.open_visitor_session(slot=1000, ttl_s=30.0, now=16.0)
        with pytest.raises(RuntimeError, match="before mutation: total candidates"):
            runtime.record_visitor_observation(
                visitor.session_uuid,
                _observation(
                    "visitor-over-envelope",
                    vector=(0.0, 1.0, 0.0, 0.0),
                    frame=103,
                ),
                now=17.0,
            )
        assert runtime.health(now=17.0).visitor_exemplar_count == 0
    finally:
        store.close()


def test_profile_and_dimension_mismatch_fail_loudly(tmp_path) -> None:
    with IdentityStore(tmp_path / "wrong.sqlite3") as store:
        store.create_resident(
            display_name="Wrong Profile",
            compatibility_sid=1,
            model_fingerprint="other-model",
            embedding_dim=DIMENSION,
        )
        with pytest.raises(ModelProfileMismatch, match="fingerprint"):
            IdentityV2Runtime(
                store,
                model_fingerprint=FINGERPRINT,
                embedding_dim=DIMENSION,
            )

    store, runtime = _runtime(tmp_path / "second")
    try:
        with pytest.raises(ModelProfileMismatch, match="dimension"):
            runtime.resolve_batch((_observation("wrong-dim", vector=(1.0, 0.0)),))
        with pytest.raises(ValueError, match="zero norm"):
            runtime.ingest_observations(
                (_observation("zero", vector=(0.0, 0.0, 0.0, 0.0)),)
            )
        weak = _observation("weak", quality=0.1)
        runtime.ingest_observations((weak,))
        with pytest.raises(ValueError, match="quality floor"):
            runtime.propose_enrollment_from_cache(
                weak.key,
                display_name="Weak",
                compatibility_sid=1,
            )
    finally:
        store.close()


def test_concurrent_confirmation_creates_one_resident_and_one_anchor(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        observation = _observation("concurrent")
        runtime.ingest_observations((observation,), now=10.0)
        proposal = runtime.propose_enrollment_from_cache(
            observation.key,
            display_name="Alice",
            compatibility_sid=1,
            now=10.0,
        )

        def confirm():
            return runtime.confirm_enrollment(
                proposal.proposal.proposal_uuid,
                expected_key=observation.key,
                evidence_digest=proposal.proposal.evidence_digest,
                now=11.0,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = tuple(executor.map(lambda _index: confirm(), range(16)))
        assert sum(not result.idempotent for result in results) == 1
        assert len({result.resident.resident_uuid for result in results}) == 1
        health = runtime.health(now=12.0)
        assert health.resident_count == 1
        assert health.enrollment_anchor_count == 1
        assert health.confirmed_enrollment_proposals == 1
    finally:
        store.close()


def test_deletion_removes_biometrics_proposal_and_hot_subject(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        observation = _observation("delete")
        proposal, confirmation = _enroll(runtime, observation)
        result = runtime.delete_resident(confirmation.resident.resident_uuid)
        assert result.resident_deleted
        assert result.exemplars_deleted == 1
        assert runtime.health(now=20.0).resident_count == 0
        assert runtime.health(now=20.0).confirmed_enrollment_proposals == 0
        with pytest.raises(KeyError):
            runtime.get_enrollment_proposal(proposal.proposal.proposal_uuid)
        decision = runtime.resolve_batch(
            (_observation("after-delete", frame=500),), now=20.0
        )[0]
        assert decision.is_unknown
        assert decision.subject is None
    finally:
        store.close()


def test_batch_resolution_is_deterministic_under_input_order(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        _enroll(runtime, _observation("alice"), name="Alice", sid=1)
        _enroll(
            runtime,
            _observation(
                "bob",
                vector=(0.0, 1.0, 0.0, 0.0),
                tracker="tracker-8",
                frame=200,
            ),
            name="Bob",
            sid=2,
        )
        rows = (
            _observation("live-a", tracker="tracker-a", frame=300),
            _observation(
                "live-b",
                vector=(0.0, 1.0, 0.0, 0.0),
                tracker="tracker-b",
                frame=300,
            ),
        )
        forward = runtime.resolve_batch(rows, now=30.0)
        reverse = runtime.resolve_batch(tuple(reversed(rows)), now=30.0)

        def summarize(values):
            return {
                row.observation_key.tracklet_id: row.decision.identity_id
                for row in values
            }

        assert summarize(forward) == summarize(reverse)
    finally:
        store.close()
