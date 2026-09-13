"""Source-owner identity handoff, including public absence and reuse negatives."""
from __future__ import annotations

import numpy as np
import pytest

from noesis_core.tracking_continuity import TrackingLifecycleRegistry
from reid.household_identity import PROVISIONAL_ID_MIN
from reid.stable_id_manager import StableIDManager


BOX = (10.0, 20.0, 40.0, 120.0)
EMB = np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def flow(tmp_path):
    manager = StableIDManager(
        use_extractor=False, household_mode=True,
        residents_file=str(tmp_path / "residents.json"),
        visitor_pool_file=str(tmp_path / "visitor_pool.json"),
        gallery_persist_file=str(tmp_path / "gallery.npz"),
        new_id_hysteresis_frames=1, household_confirm_embeddings=1,
        cos_sim_threshold=0.55, cos_sim_high_threshold=0.6,
        auto_merge_enabled=False, embed_interval_s=0.0,
    )
    owner = TrackingLifecycleRegistry()
    counters = {}

    def frame(media_ns, *, present=True, embedding=EMB, box=BOX, source=0,
              tracker=7, observed_us=None):
        counters[source] = counters.get(source, 0) + 1
        frame_id = counters[source]
        observed = observed_us or (10_000_000 + media_ns // 1000)
        tracks = []
        receipt = None
        sid = None
        if present:
            receipt = owner.identity_observation(
                source, tracker, frame_id=frame_id, observed_at_us=observed,
                media_pts_ns=media_ns, bbox=box,
            )
            sid = manager.update(
                source, tracker, box, observed / 1e6, "living-room",
                embedding=embedding, lifecycle_observation=receipt,
            )
            tracks.append({"tracker_id": tracker, "bbox": list(box), "stable_id": sid})
        manager.remove_missing_tracks(source, [tracker] if present else [], observed / 1e6)
        cohort = owner.update_frame(
            source_id=source, camera_id="living-room", frame_id=frame_id,
            observed_at_us=observed, media_pts_ns=media_ns, tracks=tracks,
        )
        owner.mark_published(cohort)
        manager.prune_identity_lifecycles(source, owner.identity_retention_keys(
            source, media_pts_ns=media_ns, max_gap_ns=manager.identity_reappearance_grace_ns,
        ))
        return sid, receipt, cohort, tracks

    return manager, owner, frame


@pytest.mark.parametrize("embedding", [None, EMB])
def test_short_return_keeps_confirmed_identity_but_publishes_absence(flow, embedding):
    manager, owner, frame = flow
    sid, first, _, _ = frame(1_000_000_000)
    assert sid < PROVISIONAL_ID_MIN
    _, _, absent, tracks = frame(1_100_000_000, present=False)
    assert tracks == []
    assert len(absent.tombstones) == 1
    assert absent.tombstones[0]["tracker_lifecycle_generation"] == first.generation
    assert manager.active_tracks == {}
    assert sid not in manager.active_zones
    assert len(manager._retained_identity_tracks) == 1
    returned, receipt, cohort, _ = frame(1_350_000_000, embedding=embedding)
    assert returned == sid
    assert receipt.key == first.key
    assert receipt.disposition == "reappeared"
    assert receipt.media_gap_ns == 350_000_000
    assert cohort.tracker_keys_changed
    assert manager.get_track_diagnostics(0, 7)["id_event"] == "reuse_lifecycle"
    assert manager.get_track_diagnostics(0, 7)["identity_retention_decision"] == "retained"
    assert not manager._retained_identity_tracks


@pytest.mark.parametrize("media_ns", [1_350_000_001, 1_800_000_000])
def test_expired_return_cannot_use_numeric_tracker_ghost_without_appearance(flow, media_ns):
    manager, owner, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    frame(1_100_000_000, present=False)
    returned, _, _, _ = frame(media_ns, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != sid
    assert manager.get_track_diagnostics(0, 7)["id_event"] != "reuse_lifecycle"


def test_media_time_controls_retention_despite_large_host_delay(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000, observed_us=10_000_000)
    frame(1_050_000_000, present=False, observed_us=20_000_000)
    returned, _, _, _ = frame(1_100_000_000, observed_us=30_000_000, embedding=None)
    assert returned == sid


@pytest.mark.parametrize("change", ["epoch", "bbox", "tracker"])
def test_new_lifecycle_never_inherits_binding_without_appearance(flow, change):
    manager, owner, frame = flow
    sid, first, _, _ = frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    if change == "epoch":
        owner.reset_source(0)
    returned, receipt, _, _ = frame(
        1_100_000_000, embedding=None,
        box=(900.0, 20.0, 40.0, 120.0) if change == "bbox" else BOX,
        tracker=8 if change == "tracker" else 7,
    )
    assert receipt.key != first.key
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != sid


def test_source_reset_invalidates_an_active_binding(flow):
    manager, owner, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    owner.reset_source(0)
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != sid
    assert not manager._retained_identity_tracks


def test_competing_camera_claim_vetoes_private_return(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    other, _, _, _ = frame(1_060_000_000, source=1, tracker=9)
    assert other == sid  # General appearance matching owns this new claim.
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != sid
    assert manager.active_tracks[(1, 9)]["stable_id"] == sid
    assert manager.get_track_diagnostics(0, 7)["identity_retention_decision"] == "conflicting_active_claim"


def test_contradictory_fresh_appearance_is_not_frozen_by_lifecycle(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    returned, _, _, _ = frame(1_100_000_000, embedding=-EMB)
    assert returned != sid
    assert manager.get_track_diagnostics(0, 7)["id_event"] != "reuse_lifecycle"
    assert manager.get_track_diagnostics(0, 7)["identity_retention_decision"] == "appearance_mismatch"


def test_provisional_identity_is_never_retained(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000, embedding=None)
    assert sid >= PROVISIONAL_ID_MIN
    frame(1_050_000_000, present=False)
    assert manager.active_tracks == {}
    assert manager._retained_identity_tracks == {}


def test_owner_expiry_clears_retention_on_empty_frame(flow):
    manager, _, frame = flow
    frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    frame(1_350_000_001, present=False)
    assert not manager._retained_identity_tracks


def test_private_retention_has_a_hard_capacity(flow):
    manager, _, frame = flow
    manager._retained_identity_max = 1
    frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    _, second, _, _ = frame(1_060_000_000, tracker=8, embedding=-EMB)
    frame(1_100_000_000, present=False)
    assert list(manager._retained_identity_tracks) == [second.key]


def test_enrollment_during_absence_invalidates_old_private_identity(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    enrolled = manager.enroll_resident(display_name="Test person", visitor_id=sid)
    assert not manager._retained_identity_tracks
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != enrolled["stable_id"]


def test_deleting_dormant_resident_cannot_resurrect_enrollment(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    enrolled = manager.enroll_resident(display_name="Test person", visitor_id=sid)
    frame(1_040_000_000)
    frame(1_050_000_000, present=False)
    manager.delete_resident(enrolled["uuid"])
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != enrolled["stable_id"]


def test_mismatched_last_observation_cannot_resume_private_binding(flow):
    from dataclasses import replace
    manager, owner, frame = flow
    sid, first, _, _ = frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    manager._retained_identity_tracks[first.key]["_identity_lifecycle"] = replace(
        first, media_pts_ns=999_999_999,
    )
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert manager.get_track_diagnostics(0, 7)["identity_retention_decision"] == "receipt_mismatch"


def test_released_visitor_without_new_generation_cannot_resume(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    generation = manager._visitor_pool.generation_for(sid)
    manager._visitor_pool.release(sid)
    assert manager._visitor_pool.generation_for(sid) == generation
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != sid
    assert manager.get_track_diagnostics(0, 7)["identity_retention_decision"] == "identity_registry_changed"


def test_visitor_expiry_purges_private_binding_before_next_frame(flow):
    manager, _, frame = flow
    sid, _, _, _ = frame(1_000_000_000)
    frame(1_050_000_000, present=False)
    manager.max_ghost_age_s = 0.01
    manager._visitor_pool.ttl_s = 0.01
    manager.prune_ghosts(12.0)
    assert sid not in manager._visitor_pool.used_sids()
    assert not manager._retained_identity_tracks
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN


@pytest.mark.parametrize("through_manager", [True, False])
def test_resident_sid_reuse_cannot_substitute_another_uuid(flow, through_manager):
    manager, _, frame = flow
    visitor, _, _, _ = frame(1_000_000_000)
    first = manager.enroll_resident(display_name="First person", visitor_id=visitor)
    frame(1_040_000_000)
    frame(1_050_000_000, present=False)
    manager.max_ghost_age_s = 0.01
    manager.prune_ghosts(12.0)
    assert not any(manager.ghosts.values())
    if through_manager:
        manager.delete_resident(first["uuid"])
        assert not manager._retained_identity_tracks
        second = manager.enroll_resident(display_name="Second person", stable_id=first["stable_id"])
    else:
        # Also protect the receiver if registry state changes independently of
        # the manager invalidation callback (for example a registry reload).
        manager._resident_registry.delete(first["uuid"])
        second = manager._resident_registry.enroll(
            display_name="Second person", stable_id=first["stable_id"],
        ).to_dict()
    assert second["uuid"] != first["uuid"]
    returned, _, _, _ = frame(1_100_000_000, embedding=None)
    assert returned >= PROVISIONAL_ID_MIN
    assert returned != first["stable_id"]
    if not through_manager:
        assert manager.get_track_diagnostics(0, 7)["identity_retention_decision"] == "identity_registry_changed"
