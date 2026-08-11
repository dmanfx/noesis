from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from reid.identity_v2 import (
    CameraOverlapEdge,
    FrameCoordinatorConfig,
    FrameReplayError,
    IdentityFrameCoordinator,
    IdentityStore,
    IdentityV2Runtime,
    OpenSetPolicy,
    OverlapSharePermit,
    PrimitiveFrameObservation,
)

FINGERPRINT = "coordinator-test-profile"
DIMENSION = 4


def _observation(
    tracker: str,
    frame: int,
    *,
    vector=(1.0, 0.0, 0.0, 0.0),
    camera="camera-a",
    quality=0.95,
    run="run-a",
) -> PrimitiveFrameObservation:
    return PrimitiveFrameObservation(
        run_id=run,
        camera_id=camera,
        tracker_id=tracker,
        frame_id=frame,
        observation_id=f"{camera}:{tracker}:{frame}",
        quality=quality,
        embedding=tuple(vector),
    )


def _runtime(tmp_path, *, visitor_gallery_max_exemplars=32, policy=None):
    store = IdentityStore(tmp_path / "identity.sqlite3")
    runtime = IdentityV2Runtime(
        store,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        policy=policy,
        visitor_gallery_max_exemplars=visitor_gallery_max_exemplars,
    )
    return store, runtime


def _resident(store, runtime, *, name="Alice", sid=1, vector=(1.0, 0.0, 0.0, 0.0)):
    resident = store.create_resident(
        display_name=name,
        compatibility_sid=sid,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
    )
    store.add_enrollment_anchor(
        resident_uuid=resident.resident_uuid,
        vector=vector,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        observation_id=f"enrollment-{sid}",
        independence_key=f"enrollment-{sid}",
    )
    runtime.refresh()
    return resident


def test_resident_hysteresis_never_forces_name_or_subject_on_unknown(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        resident = _resident(store, runtime)
        coordinator = IdentityFrameCoordinator(
            runtime,
            config=FrameCoordinatorConfig(
                resident_confirmation_frames=2,
                provisional_frames_for_visitor=3,
            ),
        )
        first = coordinator.process_frame(
            (_observation("track-1", 1),), timestamp=1.0
        ).overlays[0]
        assert first.identity_state == "unknown"
        assert first.subject_id is None
        assert first.display_name is None
        assert first.reason == "identity_hysteresis_pending"

        second = coordinator.process_frame(
            (_observation("track-1", 2),), timestamp=2.0
        ).overlays[0]
        assert second.identity_state == "resident"
        assert second.subject_id == f"resident:{resident.resident_uuid}"
        assert second.display_name == "Alice"

        negative = coordinator.process_frame(
            (
                _observation(
                    "track-1",
                    3,
                    vector=(0.0, 1.0, 0.0, 0.0),
                ),
            ),
            timestamp=3.0,
        ).overlays[0]
        assert negative.identity_state == "unknown"
        assert negative.subject_id is None
        assert negative.display_name is None
        assert negative.resolver_decision.is_unknown
    finally:
        store.close()


def test_tracker_continuity_replays_joint_assignment_flip_without_switching(
    tmp_path,
) -> None:
    """Regression for the occupied DS9 frame-453/frame-455 subject flip.

    The first batch makes the continuing tracker consume its eligible
    second-choice identity because another tracker wins its local best.  When
    that competing tracker disappears, the continuing tracker must retain the
    already accepted subject while it remains independently admissible.  A
    real evidence gap may expire the tracker state and permit a new subject.
    """

    policy = OpenSetPolicy(
        appearance_floor=0.50,
        calibrated_confidence_floor=0.20,
        ambiguity_margin_floor=0.01,
        resident_prior_bonus=0.0,
    )
    store, runtime = _runtime(tmp_path, policy=policy)
    try:
        resident_a = _resident(
            store,
            runtime,
            name="Resident A",
            sid=1,
            vector=(1.0, 0.0, 0.0, 0.0),
        )
        resident_b = _resident(
            store,
            runtime,
            name="Resident B",
            sid=2,
            vector=(0.0, 1.0, 0.0, 0.0),
        )
        subject_a = f"resident:{resident_a.resident_uuid}"
        subject_b = f"resident:{resident_b.resident_uuid}"
        coordinator = IdentityFrameCoordinator(
            runtime,
            config=FrameCoordinatorConfig(
                resident_confirmation_frames=1,
                provisional_frames_for_visitor=99,
                active_claim_ttl_s=0.1,
                tracker_state_ttl_s=2.0,
            ),
        )
        contested = coordinator.process_frame(
            (
                _observation(
                    "continuing",
                    1,
                    vector=(0.8, 0.6, 0.0, 0.0),
                ),
                _observation(
                    "winner",
                    1,
                    vector=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            timestamp=1.0,
        )
        by_tracker = {row.key.tracker_id: row for row in contested.overlays}
        assert by_tracker["continuing"].subject_id == subject_b
        assert by_tracker["winner"].subject_id == subject_a
        assert by_tracker["continuing"].resolver_decision.reason == "joint_assignment"

        retained_batch = coordinator.process_frame(
            (
                _observation(
                    "continuing",
                    2,
                    vector=(0.8, 0.6, 0.0, 0.0),
                ),
            ),
            timestamp=1.2,
        )
        retained = retained_batch.overlays[0]
        assert retained.subject_id == subject_b
        assert retained.subject_id != subject_a
        assert (
            f"rejected_candidate={subject_a}:tracker_continuity_lock"
            in retained.resolver_decision.evidence
        )
        candidates = {
            row.identity_id: row for row in retained_batch.candidate_rows[0].candidates
        }
        assert candidates[subject_a].hard_allowed is False
        assert candidates[subject_a].hard_constraint_reason == (
            "tracker_continuity_lock"
        )
        assert candidates[subject_b].hard_allowed is True

        rejected = coordinator.process_frame(
            (
                _observation(
                    "continuing",
                    3,
                    vector=(-1.0, 0.0, 0.0, 0.0),
                ),
            ),
            timestamp=1.4,
        ).overlays[0]
        assert rejected.identity_state == "unknown"
        assert rejected.subject_id is None

        after_gap = coordinator.process_frame(
            (
                _observation(
                    "continuing",
                    4,
                    vector=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            timestamp=3.5,
        ).overlays[0]
        assert after_gap.subject_id == subject_a
    finally:
        store.close()


def test_frame_order_invariance_one_to_one_and_explicit_overlap_only(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        resident = _resident(store, runtime)
        config = FrameCoordinatorConfig(
            resident_confirmation_frames=1,
            provisional_frames_for_visitor=99,
        )
        rows = (
            _observation("track-b", 1, camera="camera-b"),
            _observation("track-a", 1, camera="camera-a"),
        )
        first = IdentityFrameCoordinator(runtime, config=config).process_frame(
            rows, timestamp=1.0
        )
        second = IdentityFrameCoordinator(runtime, config=config).process_frame(
            tuple(reversed(rows)), timestamp=1.0
        )

        def summarize(result):
            return {
                row.key.tracklet_id: (row.identity_state, row.subject_id)
                for row in result.overlays
            }

        assert summarize(first) == summarize(second)
        known = [row for row in first.overlays if row.identity_state == "resident"]
        assert len(known) == 1

        keys = [row.to_runtime().key for row in rows]
        subject_id = f"resident:{resident.resident_uuid}"
        permit = OverlapSharePermit(
            subject_id, keys[0].tracklet_id, keys[1].tracklet_id
        )
        shared = IdentityFrameCoordinator(
            runtime,
            config=config,
            camera_overlap_edges=(CameraOverlapEdge("camera-a", "camera-b"),),
        ).process_frame(
            rows,
            timestamp=1.0,
            overlap_permits=(permit,),
        )
        assert [row.subject_id for row in shared.overlays] == [subject_id, subject_id]
        assert all(
            "overlap_share_permitted" in row.resolver_decision.evidence
            for row in shared.overlays
        )
    finally:
        store.close()


def test_recent_cross_camera_claim_blocks_duplicate_without_exact_permit(
    tmp_path,
) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        resident = _resident(store, runtime)
        config = FrameCoordinatorConfig(
            resident_confirmation_frames=1,
            provisional_frames_for_visitor=99,
            active_claim_ttl_s=1.5,
        )
        coordinator = IdentityFrameCoordinator(
            runtime,
            config=config,
            camera_overlap_edges=(
                CameraOverlapEdge("camera-a", "camera-b", max_batch_gap_s=0.5),
            ),
        )
        camera_a = _observation("track-a", 1, camera="camera-a")
        first = coordinator.process_frame((camera_a,), timestamp=10.0)
        subject_id = f"resident:{resident.resident_uuid}"
        assert first.overlays[0].subject_id == subject_id

        camera_b = _observation("track-b", 1, camera="camera-b")
        blocked = coordinator.process_frame((camera_b,), timestamp=10.1)
        assert blocked.overlays[0].identity_state == "unknown"
        assert blocked.overlays[0].subject_id is None
        assert blocked.overlays[0].resolver_decision.reason == "active_copresence_claim"

        camera_b_next = _observation("track-b", 2, camera="camera-b")
        permit = OverlapSharePermit(
            subject_id,
            camera_a.to_runtime().key.tracklet_id,
            camera_b_next.to_runtime().key.tracklet_id,
        )
        shared = coordinator.process_frame(
            (camera_b_next,),
            timestamp=10.2,
            overlap_permits=(permit,),
        )
        assert shared.overlays[0].subject_id == subject_id
        assert (
            "cross_batch_overlap_share_permitted"
            in shared.overlays[0].resolver_decision.evidence
        )
        assert len(shared.active_claims) == 2

        # The permit is a current proof, not a sticky exemption.
        camera_a_next = _observation("track-a", 2, camera="camera-a")
        blocked_again = coordinator.process_frame((camera_a_next,), timestamp=10.3)
        assert blocked_again.overlays[0].identity_state == "unknown"
        assert blocked_again.overlays[0].subject_id is None

        # Once every conflicting lease expires, ordinary matching resumes.
        camera_b_after_expiry = _observation("track-b", 3, camera="camera-b")
        reacquired = coordinator.process_frame((camera_b_after_expiry,), timestamp=12.0)
        assert reacquired.overlays[0].subject_id == subject_id
    finally:
        store.close()


def test_cross_batch_overlap_permit_requires_configured_topology(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        resident = _resident(store, runtime)
        coordinator = IdentityFrameCoordinator(
            runtime,
            config=FrameCoordinatorConfig(
                resident_confirmation_frames=1,
                provisional_frames_for_visitor=99,
            ),
        )
        camera_a = _observation("track-a", 1, camera="camera-a")
        coordinator.process_frame((camera_a,), timestamp=1.0)
        camera_b = _observation("track-b", 1, camera="camera-b")
        permit = OverlapSharePermit(
            f"resident:{resident.resident_uuid}",
            camera_a.to_runtime().key.tracklet_id,
            camera_b.to_runtime().key.tracklet_id,
        )
        with pytest.raises(ValueError, match="not configured"):
            coordinator.process_frame(
                (camera_b,),
                timestamp=1.1,
                overlap_permits=(permit,),
            )
    finally:
        store.close()


def test_provisional_visitor_mint_record_release_and_generation_reuse(tmp_path) -> None:
    store, runtime = _runtime(tmp_path, visitor_gallery_max_exemplars=3)
    try:
        coordinator = IdentityFrameCoordinator(
            runtime,
            config=FrameCoordinatorConfig(
                provisional_frames_for_visitor=2,
                visitor_match_confirmation_frames=1,
                visitor_slot_min=1000,
                visitor_slot_max=1000,
                visitor_record_interval_frames=1,
                visitor_release_after_s=3.0,
                visitor_session_ttl_s=20.0,
                tracker_state_ttl_s=1.5,
            ),
        )
        first = coordinator.process_frame(
            (_observation("visitor-track", 1),), timestamp=1.0
        )
        assert first.overlays[0].identity_state == "unknown"
        assert first.overlays[0].provisional_evidence_count == 1
        second = coordinator.process_frame(
            (_observation("visitor-track", 2),), timestamp=2.0
        )
        assert len(second.minted_visitor_sessions) == 1
        session_one = second.minted_visitor_sessions[0]
        assert session_one.slot == 1000
        assert session_one.generation == 1
        assert second.overlays[0].identity_state == "visitor"
        assert second.overlays[0].visitor_generation == 1
        assert second.overlays[0].display_name is None

        for frame in range(3, 8):
            matched = coordinator.process_frame(
                (_observation("visitor-track", frame),),
                timestamp=float(frame),
            )
            assert matched.overlays[0].identity_state == "visitor"
            assert matched.overlays[0].subject_id == second.overlays[0].subject_id
        assert runtime.health(now=7.0).visitor_exemplar_count == 3

        released = coordinator.process_frame((), timestamp=11.0)
        assert released.released_visitor_session_uuids == (session_one.session_uuid,)
        assert runtime.active_visitor_sessions(now=11.0) == ()

        coordinator.process_frame(
            (_observation("new-track", 1, run="run-b"),), timestamp=12.0
        )
        reminted = coordinator.process_frame(
            (_observation("new-track", 2, run="run-b"),), timestamp=13.0
        )
        session_two = reminted.minted_visitor_sessions[0]
        assert session_two.slot == 1000
        assert session_two.generation == 2
        assert session_two.session_uuid != session_one.session_uuid
        assert reminted.overlays[0].subject_id != second.overlays[0].subject_id
    finally:
        store.close()


def test_coordinator_restart_reacquires_durable_visitor_without_remint(
    tmp_path,
) -> None:
    db_path = tmp_path / "identity.sqlite3"
    store = IdentityStore(db_path)
    runtime = IdentityV2Runtime(
        store,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
    )
    config = FrameCoordinatorConfig(
        provisional_frames_for_visitor=2,
        visitor_match_confirmation_frames=1,
        visitor_session_ttl_s=30.0,
    )
    coordinator = IdentityFrameCoordinator(runtime, config=config)
    coordinator.process_frame((_observation("old-track", 1),), timestamp=1.0)
    minted = coordinator.process_frame(
        (_observation("old-track", 2),), timestamp=2.0
    ).minted_visitor_sessions[0]
    store.close()

    with IdentityStore(db_path) as reopened_store:
        reopened_runtime = IdentityV2Runtime(
            reopened_store,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
        )
        restarted = IdentityFrameCoordinator(reopened_runtime, config=config)
        result = restarted.process_frame(
            (_observation("new-camera-track", 3, camera="camera-b"),),
            timestamp=3.0,
        )
        assert result.minted_visitor_sessions == ()
        assert result.overlays[0].identity_state == "visitor"
        assert result.overlays[0].subject_id is not None
        assert minted.session_uuid in result.overlays[0].subject_id
        assert result.overlays[0].visitor_generation == 1


def test_frame_replay_is_rejected_and_concurrent_distinct_tracks_are_safe(
    tmp_path,
) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        coordinator = IdentityFrameCoordinator(
            runtime,
            config=FrameCoordinatorConfig(provisional_frames_for_visitor=99),
        )
        row = _observation("track", 1)
        coordinator.process_frame((row,), timestamp=1.0)
        with pytest.raises(FrameReplayError):
            coordinator.process_frame((row,), timestamp=1.1)

        def process(index):
            return coordinator.process_frame(
                (
                    _observation(
                        f"concurrent-{index}",
                        1,
                        run=f"run-{index}",
                    ),
                ),
                timestamp=2.0,
            ).overlays[0]

        with ThreadPoolExecutor(max_workers=8) as executor:
            overlays = tuple(executor.map(process, range(16)))
        assert len(overlays) == 16
        assert all(row.identity_state == "unknown" for row in overlays)
        assert all(row.display_name is None for row in overlays)
        assert coordinator.tracker_state_count() >= 16
    finally:
        store.close()


def test_coordinator_household_batch_latency(tmp_path) -> None:
    store, runtime = _runtime(tmp_path)
    try:
        basis = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        for index, vector in enumerate(basis, start=1):
            _resident(
                store,
                runtime,
                name=f"Resident {index}",
                sid=index,
                vector=vector,
            )
        coordinator = IdentityFrameCoordinator(
            runtime,
            config=FrameCoordinatorConfig(
                resident_confirmation_frames=1,
                provisional_frames_for_visitor=999,
            ),
        )
        samples = []
        for frame in range(1, 51):
            rows = tuple(
                _observation(
                    f"track-{index}",
                    frame,
                    vector=basis[index % 4],
                    camera=f"camera-{index % 4}",
                )
                for index in range(8)
            )
            started = time.perf_counter()
            result = coordinator.process_frame(rows, timestamp=float(frame))
            samples.append((time.perf_counter() - started) * 1000.0)
            assert len(result.overlays) == 8
        average_ms = sum(samples) / len(samples)
        assert average_ms < 100.0, f"coordinator averaged {average_ms:.2f} ms"
    finally:
        store.close()
