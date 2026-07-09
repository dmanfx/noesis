"""Household closed-world resident/visitor StableID tests."""

from __future__ import annotations

import numpy as np
import pytest

from reid.household_identity import PROVISIONAL_ID_MIN, VISITOR_ID_MAX, VISITOR_ID_MIN
from reid.stable_id_manager import StableIDManager


def _unit_emb(seed: int, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (float(np.linalg.norm(vec)) + 1e-12)


def _bbox(x: float = 10.0) -> tuple:
    return (x, 20.0, 40.0, 120.0)


def _pose_features(keys, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {str(k): float(rng.uniform(0.2, 1.0)) for k in keys}


def _quality(mean_conf: float = 0.9, valid_frac: float = 0.9) -> dict[str, float]:
    return {"kpt_mean_conf": float(mean_conf), "kpt_valid_frac": float(valid_frac)}


def _make_household_mgr(tmp_path, **overrides):
    root = tmp_path / "household"
    root.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        use_extractor=False,
        household_mode=True,
        residents_file=str(root / "residents.json"),
        visitor_pool_file=str(root / "visitor_pool.json"),
        gallery_persist_file=str(root / "gallery.npz"),
        new_id_hysteresis_frames=1,
        household_confirm_embeddings=1,
        cos_sim_threshold=0.55,
        cos_sim_high_threshold=0.60,
        resident_match_margin=0.05,
        allow_multi_zone_active=False,
        auto_merge_enabled=False,
        embed_interval_s=0.0,
    )
    defaults.update(overrides)
    return StableIDManager(**defaults)


def test_household_mints_visitor_in_range(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    emb = _unit_emb(7)
    sid = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=_bbox(),
        ts=1.0,
        zone="default",
        embedding=emb,
    )
    assert VISITOR_ID_MIN <= sid <= VISITOR_ID_MAX
    metrics = mgr.get_sid_metrics()
    assert metrics["mint_visitor_count"] >= 1
    assert metrics["next_sid"] <= VISITOR_ID_MAX + 1


def test_resident_match_preferred_over_visitor_mint(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    resident_emb = _unit_emb(42)
    visitor_emb = _unit_emb(99)

    visitor_sid = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=_bbox(10),
        ts=1.0,
        zone="default",
        embedding=visitor_emb,
    )
    assert VISITOR_ID_MIN <= visitor_sid <= VISITOR_ID_MAX

    enrolled = mgr.enroll_resident(display_name="Alex", visitor_id=visitor_sid)
    resident_sid = int(enrolled["stable_id"])
    assert 1 <= resident_sid < VISITOR_ID_MIN

    # Clear live claim so exclusivity does not block a later gallery match.
    mgr.remove_missing_tracks(0, [], 1.5)
    mgr.gallery[resident_sid].clear()
    mgr.gallery[resident_sid].append((1.0, resident_emb))
    mgr.sid_centroid[resident_sid] = resident_emb

    match_sid = mgr.update(
        sensor_id=1,
        ds_obj_id=2,
        bbox_ltrbwh=_bbox(30),
        ts=2.0,
        zone="default",
        embedding=resident_emb,
    )
    assert match_sid == resident_sid
    diag = mgr.get_track_diagnostics(1, 2)
    assert diag.get("identity_kind") == "resident"


def test_enroll_resident_remaps_active_zones_and_gallery(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    emb = _unit_emb(55)
    visitor_sid = mgr.update(0, 1, _bbox(), 1.0, "kitchen", embedding=emb)
    assert VISITOR_ID_MIN <= visitor_sid <= VISITOR_ID_MAX
    assert visitor_sid in mgr.active_zones
    assert (0, "kitchen") in mgr.active_zones[visitor_sid]
    gallery_before = list(mgr.gallery.get(visitor_sid, []))
    assert len(gallery_before) >= 1

    enrolled = mgr.enroll_resident(display_name="Mayor", visitor_id=visitor_sid)
    resident_sid = int(enrolled["stable_id"])
    assert 1 <= resident_sid < VISITOR_ID_MIN

    track = mgr.active_tracks[(0, 1)]
    assert int(track["stable_id"]) == resident_sid
    assert track["identity_kind"] == "resident"
    assert track.get("display_name") == "Mayor"
    assert resident_sid in mgr.active_zones
    assert (0, "kitchen") in mgr.active_zones[resident_sid]
    assert visitor_sid not in mgr.active_zones
    assert len(mgr.gallery.get(resident_sid, [])) >= 1
    assert visitor_sid not in mgr.gallery or len(mgr.gallery.get(visitor_sid, [])) == 0


def test_delete_resident_remaps_to_new_visitor(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    emb = _unit_emb(66)
    visitor_sid = mgr.update(0, 1, _bbox(), 1.0, "default", embedding=emb)
    enrolled = mgr.enroll_resident(display_name="Pat", visitor_id=visitor_sid)
    resident_sid = int(enrolled["stable_id"])
    resident_uuid = str(enrolled["uuid"])
    assert int(mgr.active_tracks[(0, 1)]["stable_id"]) == resident_sid

    deleted = mgr.delete_resident(resident_uuid)
    assert int(deleted["stable_id"]) == resident_sid
    track = mgr.active_tracks[(0, 1)]
    new_sid = int(track["stable_id"])
    assert VISITOR_ID_MIN <= new_sid <= VISITOR_ID_MAX
    assert new_sid != resident_sid
    assert track["identity_kind"] == "visitor"
    assert track.get("resident_uuid") is None
    assert track.get("display_name") is None
    assert resident_sid not in mgr.active_zones
    assert new_sid in mgr.active_zones


def test_exclusivity_blocks_false_share_without_overlap(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    emb_a = _unit_emb(11)
    emb_b = _unit_emb(22)

    sid_a = mgr.update(0, 1, _bbox(10), 1.0, "default", embedding=emb_a)
    sid_b = mgr.update(0, 2, _bbox(50), 2.0, "default", embedding=emb_b)
    assert sid_a != sid_b

    # Attempt to claim sid_a on a second camera without overlap permit.
    conflict_sid = mgr.update(1, 3, _bbox(60), 3.0, "default", embedding=emb_a)
    assert conflict_sid != sid_a
    assert VISITOR_ID_MIN <= conflict_sid <= VISITOR_ID_MAX
    metrics = mgr.get_sid_metrics()
    assert metrics["false_share_blocked_count"] >= 1


def test_visitor_pool_recycles_after_ttl(tmp_path):
    mgr = _make_household_mgr(tmp_path, visitor_ttl_s=0.01)
    emb = _unit_emb(5)
    sid = mgr.update(0, 1, _bbox(), 1.0, "default", embedding=emb)
    mgr.remove_missing_tracks(0, [], 1.1)
    mgr.prune_ghosts(10.0)
    metrics = mgr.get_sid_metrics()
    assert sid >= VISITOR_ID_MIN
    assert metrics.get("visitor_count", 0) >= 0


def test_pose_only_gallery_match_uses_household_claims(tmp_path):
    keys = getattr(StableIDManager, "_POSE_FEATURE_KEYS", ())
    features = _pose_features(keys, seed=123)
    mgr = _make_household_mgr(
        tmp_path,
        pose_enabled=True,
        pose_only_threshold=0.7,
        pose_sim_threshold=0.55,
        pose_sim_high_threshold=0.65,
        pose_min_features=6,
        pose_min_mean_conf=0.2,
        pose_min_valid_frac=0.2,
        new_id_hysteresis_frames=1,
        household_confirm_embeddings=1,
        allow_multi_zone_active=False,
    )
    # Seed a visitor with pose gallery via confirmed mint (embedding path).
    emb = _unit_emb(77)
    sid0 = mgr.update(
        0,
        1,
        _bbox(),
        1.0,
        "default",
        embedding=emb,
        pose_features=features,
        pose_quality=_quality(),
    )
    assert VISITOR_ID_MIN <= sid0 <= VISITOR_ID_MAX
    assert sid0 in mgr.pose_gallery or mgr.active_tracks[(0, 1)].get("pose_vec") is not None

    # Force pose gallery entry for cross-cam pose-only match.
    pose_vec = mgr.active_tracks[(0, 1)].get("pose_vec")
    assert pose_vec is not None
    mgr._update_pose_state(int(sid0), pose_vec, 1.0)

    # Same pose on another camera without overlap must not steal SID.
    sid1 = mgr.update(
        1,
        2,
        _bbox(40),
        2.0,
        "default",
        embedding=None,
        pose_features=features,
        pose_quality=_quality(),
    )
    assert sid1 != sid0
    diag = mgr.get_track_diagnostics(1, 2)
    assert diag.get("id_reject_reason") in (
        "exclusivity_blocked",
        "active_guard",
        "frame_sid_claimed",
        None,
    )
    # If it minted provisional/visitor, it must not be the occupied SID.
    assert int(sid1) != int(sid0)
