"""Provisional suppression tests for household identity."""

from __future__ import annotations

import numpy as np
import pytest

from reid.household_identity import PROVISIONAL_ID_MIN, VISITOR_ID_MAX, VISITOR_ID_MIN
from reid.stable_id_manager import StableIDManager


def _unit_emb(seed: int, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (float(np.linalg.norm(vec)) + 1e-12)


def _bbox() -> tuple:
    return (10.0, 20.0, 40.0, 120.0)


def _make_household_mgr(tmp_path, **overrides):
    root = tmp_path / "household"
    root.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        use_extractor=False,
        household_mode=True,
        residents_file=str(root / "residents.json"),
        visitor_pool_file=str(root / "visitor_pool.json"),
        gallery_persist_file=str(root / "gallery.npz"),
        new_id_hysteresis_frames=3,
        household_confirm_embeddings=3,
        cos_sim_high_threshold=0.70,
        allow_multi_zone_active=False,
        embed_interval_s=0.0,
    )
    defaults.update(overrides)
    return StableIDManager(**defaults)


def test_provisional_id_before_confirmation(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    emb = _unit_emb(3)
    sid = mgr.update(0, 1, _bbox(), 1.0, "default", embedding=emb)
    assert sid >= PROVISIONAL_ID_MIN
    assert sid not in mgr.gallery or len(mgr.gallery.get(sid, [])) == 0
    assert (0, 1) in mgr.active_tracks
    assert sid in mgr.active_zones
    diag = mgr.get_track_diagnostics(0, 1)
    assert diag.get("identity_kind") == "provisional"
    assert diag.get("identity_state") == "provisional"


def test_provisional_stable_across_frames(tmp_path):
    # hysteresis=3 / confirm_emb=3 → frames 1–2 stay provisional; frame 3 promotes.
    mgr = _make_household_mgr(tmp_path)
    emb = _unit_emb(11)
    sid1 = mgr.update(0, 1, _bbox(), 1.0, "default", embedding=emb)
    sid2 = mgr.update(0, 1, _bbox(), 2.0, "default", embedding=emb)
    assert sid1 >= PROVISIONAL_ID_MIN
    assert sid2 == sid1
    assert (0, 1) in mgr.active_tracks
    assert mgr.active_tracks[(0, 1)]["identity_kind"] == "provisional"
    metrics = mgr.get_sid_metrics()
    assert metrics["provisional_count"] == 1
    assert metrics["provisional_active_count"] == 1
    assert metrics["provisional_event_count"] == 1


def test_provisional_promotes_to_visitor_after_confirm(tmp_path):
    mgr = _make_household_mgr(
        tmp_path,
        new_id_hysteresis_frames=2,
        household_confirm_embeddings=2,
        cos_sim_high_threshold=0.55,
    )
    emb = _unit_emb(21)
    sid_p = mgr.update(0, 1, _bbox(), 1.0, "default", embedding=emb)
    assert sid_p >= PROVISIONAL_ID_MIN
    sid_v = mgr.update(0, 1, _bbox(), 2.0, "default", embedding=emb)
    assert VISITOR_ID_MIN <= sid_v <= VISITOR_ID_MAX
    assert sid_v != sid_p
    assert mgr.active_tracks[(0, 1)]["identity_kind"] == "visitor"
    assert sid_v in mgr.active_zones
    assert sid_p not in mgr.active_zones
    assert len(mgr.gallery.get(sid_v, [])) >= 1
    metrics = mgr.get_sid_metrics()
    assert metrics["provisional_count"] == 0
    assert metrics["provisional_event_count"] == 1


def test_provisional_does_not_grow_monotonic_sid(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    for i in range(5):
        emb = _unit_emb(100 + i)
        mgr.update(0, i + 1, _bbox(), float(i + 1), "default", embedding=emb)
        mgr.remove_missing_tracks(0, [], float(i + 1.5))
    metrics = mgr.get_sid_metrics()
    assert metrics["next_sid"] <= VISITOR_ID_MAX + 1


def test_confirmed_visitor_mints_after_gates(tmp_path):
    mgr = _make_household_mgr(
        tmp_path,
        new_id_hysteresis_frames=1,
        household_confirm_embeddings=1,
        cos_sim_high_threshold=0.55,
    )
    emb = _unit_emb(8)
    sid = mgr.update(0, 1, _bbox(), 1.0, "default", embedding=emb)
    assert VISITOR_ID_MIN <= sid <= VISITOR_ID_MAX
    assert len(mgr.gallery.get(sid, [])) >= 1
    diag = mgr.get_track_diagnostics(0, 1)
    assert diag.get("identity_kind") == "visitor"


def test_many_provisional_tracks_do_not_pollute_gallery(tmp_path):
    mgr = _make_household_mgr(tmp_path)
    for i in range(8):
        emb = _unit_emb(200 + i)
        sid = mgr.update(0, i + 1, _bbox(), 1.0 + i * 0.01, "default", embedding=emb)
        assert sid >= PROVISIONAL_ID_MIN
        mgr.remove_missing_tracks(0, [], 1.5 + i * 0.01)
    gallery_sids = [sid for sid in mgr.gallery.keys() if len(mgr.gallery[sid]) > 0]
    for sid in gallery_sids:
        assert sid >= VISITOR_ID_MIN or sid < PROVISIONAL_ID_MIN
