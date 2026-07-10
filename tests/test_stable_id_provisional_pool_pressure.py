"""Provisional/visitor pool pressure: never crash on exhaustion under churn."""

from __future__ import annotations

import numpy as np

from reid.household_identity import PROVISIONAL_ID_MIN, VISITOR_ID_MAX, VISITOR_ID_MIN
from reid.stable_id_manager import StableIDManager


def _unit_emb(seed: int, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (float(np.linalg.norm(vec)) + 1e-12)


def _bbox() -> tuple:
    return (10.0, 20.0, 40.0, 120.0)


def _make_mgr(tmp_path, **overrides):
    root = tmp_path / "household"
    root.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        use_extractor=False,
        household_mode=True,
        residents_file=str(root / "residents.json"),
        visitor_pool_file=str(root / "visitor_pool.json"),
        gallery_persist_file=str(root / "gallery.npz"),
        new_id_hysteresis_frames=5,
        household_confirm_embeddings=5,
        cos_sim_high_threshold=0.99,
        allow_multi_zone_active=False,
        embed_interval_s=0.0,
        visitor_ttl_s=3600.0,
    )
    defaults.update(overrides)
    return StableIDManager(**defaults)


def test_provisional_pool_survives_more_than_capacity(tmp_path):
    mgr = _make_mgr(tmp_path)
    # Mint more simultaneous provisionals than the 100-slot pool.
    for i in range(130):
        sid = mgr.update(0, i + 1, _bbox(), 1.0, "default", embedding=_unit_emb(i + 1))
        assert sid >= PROVISIONAL_ID_MIN
    metrics = mgr.get_sid_metrics()
    assert metrics["provisional_pool_used"] <= 100
    assert metrics["provisional_pool_free"] >= 0
    # Eviction under pressure keeps the manager alive.
    assert len(mgr.active_tracks) <= 100


def test_provisional_pool_recycles_after_remove(tmp_path):
    mgr = _make_mgr(tmp_path)
    for i in range(100):
        mgr.update(0, i + 1, _bbox(), 1.0 + i * 0.001, "default", embedding=_unit_emb(i + 1))
    assert mgr.get_sid_metrics()["provisional_pool_used"] == 100
    mgr.remove_missing_tracks(0, [], 2.0)
    assert mgr.get_sid_metrics()["provisional_pool_used"] == 0
    sid = mgr.update(0, 999, _bbox(), 3.0, "default", embedding=_unit_emb(999))
    assert sid >= PROVISIONAL_ID_MIN
    assert mgr.get_sid_metrics()["provisional_pool_used"] == 1


def test_same_frame_conflict_does_not_leak_abandoned_provisional(tmp_path):
    mgr = _make_mgr(tmp_path, new_id_hysteresis_frames=50, household_confirm_embeddings=50)
    s1 = mgr.update(0, 1, _bbox(), 10.0, "default", embedding=_unit_emb(1))
    assert s1 >= PROVISIONAL_ID_MIN
    # Plant a second track claiming the same provisional SID at the same timestamp.
    mgr.active_tracks[(0, 2)] = dict(mgr.active_tracks[(0, 1)])
    mgr.active_tracks[(0, 2)]["stable_id"] = int(s1)
    mgr.active_zones[int(s1)].add((0, "default"))
    s2 = mgr.update(0, 2, _bbox(), 10.0, "default", embedding=_unit_emb(2))
    assert s2 != s1
    assert s2 >= PROVISIONAL_ID_MIN
    # Both tracks still hold distinct provisionals (no leak of a third unused SID).
    used = mgr.get_sid_metrics()["provisional_pool_used"]
    assert used == 2
    mgr.remove_missing_tracks(0, [1], 11.0)
    assert mgr.get_sid_metrics()["provisional_pool_used"] == 1
    mgr.remove_missing_tracks(0, [], 12.0)
    assert mgr.get_sid_metrics()["provisional_pool_used"] == 0


def test_visitor_pool_pressure_recycles_inactive(tmp_path):
    mgr = _make_mgr(
        tmp_path,
        new_id_hysteresis_frames=1,
        household_confirm_embeddings=1,
        cos_sim_high_threshold=0.99,
        visitor_ttl_s=3600.0,
        max_active_ids_per_sensor=64,
        new_id_confirm_frames_at_cap=1,
    )
    # Fill visitor range with confirmed identities, then drop them.
    for i in range(VISITOR_ID_MAX - VISITOR_ID_MIN + 1):
        sid = mgr.update(0, i + 1, _bbox(), 1.0 + i * 0.01, "default", embedding=_unit_emb(1000 + i))
        assert VISITOR_ID_MIN <= int(sid) <= VISITOR_ID_MAX, f"track {i+1} got {sid}"
    mgr.remove_missing_tracks(0, [], 2.0)
    # Ghosts would normally block TTL recycle; pressure mint must still succeed.
    sid = mgr.update(0, 500, _bbox(), 3.0, "default", embedding=_unit_emb(500))
    assert VISITOR_ID_MIN <= int(sid) <= VISITOR_ID_MAX
