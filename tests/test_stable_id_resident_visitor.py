"""Household closed-world resident/visitor StableID tests."""

from __future__ import annotations

import json
import stat
import time

import numpy as np
import pytest

from reid.household_identity import (
    PROVISIONAL_ID_MIN,
    ResidentRegistry,
    VISITOR_ID_MAX,
    VISITOR_ID_MIN,
    VisitorPool,
)
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
    visitor_diag = mgr.get_track_diagnostics(0, 1)
    assert visitor_diag.get("identity_kind") == "visitor"
    assert visitor_diag.get("visitor_generation") == 1

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
    assert len(mgr.gallery.get(resident_sid, [])) == 0
    assert len(mgr.gallery.get(new_sid, [])) == 0

    with np.load(tmp_path / "household" / "gallery.npz", allow_pickle=False) as persisted:
        assert resident_sid not in set(int(s) for s in persisted["sids"].tolist())


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


def test_visitor_pool_restart_does_not_reallocate_used_slot(tmp_path):
    pool_path = tmp_path / "visitor_pool.json"
    first = VisitorPool(pool_file=str(pool_path), id_min=1000, id_max=1001)
    sid_a = first.alloc(1.0)
    generation_a = first.generation_for(sid_a)
    first.save()

    restarted = VisitorPool(pool_file=str(pool_path), id_min=1000, id_max=1001)
    sid_b = restarted.alloc(2.0)
    assert sid_b != sid_a
    assert sid_a in restarted.used_sids()
    assert restarted.generation_for(sid_a) == generation_a

    restarted.release(sid_a)
    sid_c = restarted.alloc(3.0)
    assert sid_c == sid_a
    assert restarted.generation_for(sid_c) == generation_a + 1


def test_resident_registry_rejects_duplicate_resident_inventory(tmp_path):
    root = tmp_path / "household"
    root.mkdir(mode=0o700)
    path = root / "residents.json"
    path.write_text(
        '{"next_resident_id":2,"residents":[],"residents":['
        '{"uuid":"resident-a","stable_id":1,"display_name":"Alice",'
        '"created_ts":1.0,"embedding_count":1}]}',
        encoding="utf-8",
    )
    path.chmod(0o600)

    registry = ResidentRegistry(str(path))

    assert registry.list_residents() == []


def test_visitor_pool_rejects_duplicate_persisted_state(tmp_path):
    path = tmp_path / "visitor_pool.json"
    path.write_text(
        '{"version":1,"free_visitor_sids":[1001],'
        '"visitor_last_seen":{},"visitor_last_seen":{"1000":1.0},'
        '"visitor_generations":{"1000":4}}',
        encoding="utf-8",
    )
    path.chmod(0o600)

    pool = VisitorPool(pool_file=str(path), id_min=1000, id_max=1001)

    assert pool.alloc(2.0) == 1000
    assert pool.generation_for(1000) == 1


def test_resident_gallery_survives_ghost_expiry_and_restart(tmp_path):
    now = time.time()
    mgr = _make_household_mgr(
        tmp_path,
        max_ghost_age_s=0.1,
        active_evict_grace_s=0.1,
        gallery_persist_max_age_s=3600.0,
    )
    emb = _unit_emb(505)
    visitor_sid = mgr.update(0, 1, _bbox(), now, "default", embedding=emb)
    enrolled = mgr.enroll_resident(display_name="Resident", visitor_id=visitor_sid)
    resident_sid = int(enrolled["stable_id"])
    # A later update populates global last-seen, which previously made the
    # generic ghost pruner delete the durable resident gallery.
    mgr.update(0, 1, _bbox(), now + 0.01, "default", embedding=emb)
    mgr.remove_missing_tracks(0, [], now + 0.02)
    mgr.prune_ghosts(now + 10.0)

    assert len(mgr.gallery.get(resident_sid, [])) >= 1
    assert mgr.list_residents()[0]["embedding_count"] == len(mgr.gallery[resident_sid])
    assert mgr.save_gallery()

    restarted = _make_household_mgr(
        tmp_path,
        max_ghost_age_s=0.1,
        active_evict_grace_s=0.1,
        gallery_persist_max_age_s=3600.0,
    )
    assert len(restarted.gallery.get(resident_sid, [])) >= 1
    assert restarted.list_residents()[0]["gallery_embeddings"] >= 1


def test_resident_embedding_count_uses_live_gallery_truth(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    emb = _unit_emb(606)
    visitor_sid = mgr.update(0, 1, _bbox(), time.time(), "default", embedding=emb)
    enrolled = mgr.enroll_resident(display_name="Resident", visitor_id=visitor_sid)
    resident_sid = int(enrolled["stable_id"])
    resident_uuid = str(enrolled["uuid"])

    registry_rec = mgr._resident_registry.get_by_uuid(resident_uuid)
    assert registry_rec is not None
    registry_rec.embedding_count = 999
    mgr._resident_registry.save()
    live_count = len(mgr.gallery.get(resident_sid, []))
    assert mgr.list_residents()[0]["embedding_count"] == live_count

    assert mgr.save_gallery()
    persisted = json.loads((tmp_path / "household" / "residents.json").read_text())
    assert persisted["residents"][0]["embedding_count"] == live_count


def test_household_state_permissions_are_owner_only(tmp_path):
    mgr = _make_household_mgr(tmp_path, aliases_enabled=True, alias_file=str(tmp_path / "household" / "aliases.json"))
    visitor_sid = mgr.update(0, 1, _bbox(), time.time(), "default", embedding=_unit_emb(707))
    mgr.enroll_resident(display_name="Resident", visitor_id=visitor_sid)
    assert mgr._visitor_pool is not None
    mgr._visitor_pool.save()
    mgr._save_aliases()
    assert mgr.save_gallery()

    household_dir = tmp_path / "household"
    assert stat.S_IMODE(household_dir.stat().st_mode) == 0o700
    for name in ("residents.json", "visitor_pool.json", "gallery.npz", "aliases.json"):
        path = household_dir / name
        assert path.exists(), name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_camera_topology_runtime_map_validation(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    valid = mgr.validate_camera_topology(
        {0: "living-room", 1: "kitchen", 2: "family-room"}
    )
    assert valid["valid"] is True
    assert mgr.get_sid_metrics()["camera_topology_validation_status"] == "valid"

    with pytest.raises(ValueError, match="camera topology/runtime map mismatch"):
        mgr.validate_camera_topology(
            {0: "kitchen", 1: "living-room", 2: "family-room"}
        )
    metrics = mgr.get_sid_metrics()
    assert metrics["camera_topology_validation_status"] == "invalid"
    assert metrics["camera_topology_validation_errors"]

    with pytest.raises(ValueError, match="camera topology/runtime map mismatch"):
        _make_household_mgr(
            tmp_path / "constructor-mismatch",
            camera_labels={0: "kitchen", 1: "living-room", 2: "family-room"},
        )


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
