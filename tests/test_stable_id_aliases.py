from __future__ import annotations

import threading
import json
from typing import Dict

import numpy as np

from reid.stable_id_manager import StableIDManager


def _unit_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(dim,)).astype(np.float32)
    emb /= float(np.linalg.norm(emb) + 1e-12)
    return emb


def _pose_features(scale: float = 1.0) -> Dict[str, float]:
    keys = StableIDManager._POSE_FEATURE_KEYS
    return {keys[i]: float(i + 1) * scale for i in range(6)}


def _pose_quality() -> Dict[str, float]:
    return {"kpt_mean_conf": 1.0, "kpt_valid_frac": 1.0}


def _make_mgr(tmp_path, **kwargs) -> StableIDManager:
    alias_file = kwargs.pop("alias_file", str(tmp_path / "aliases.json"))
    sid_pool_file = kwargs.pop("sid_pool_file", str(tmp_path / "sid_pool.json"))
    return StableIDManager(
        use_extractor=False,
        new_id_hysteresis_frames=1,
        new_id_confirm_frames_at_cap=1,
        embed_interval_s=0.0,
        sid_pool_file=str(sid_pool_file),
        alias_file=alias_file,
        aliases_enabled=True,
        **kwargs,
    )


def _update_with_embedding(mgr: StableIDManager, sensor_id: int, ds_obj_id: int, ts: float, emb: np.ndarray) -> int:
    return mgr.update(
        sensor_id=sensor_id,
        ds_obj_id=ds_obj_id,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=ts,
        zone=None,
        frame_bgr=None,
        embedding=emb,
    )


def _vec2d(x: float, y: float) -> np.ndarray:
    vec = np.asarray([x, y], dtype=np.float32)
    return vec / (float(np.linalg.norm(vec)) + 1e-12)


def test_alias_canonicalizes_update_return_value(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=1)
    emb_b = _unit_embedding(seed=2)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    assert sid_a != sid_b

    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]

    sid_b2 = _update_with_embedding(mgr, 0, 2, 2.0, emb_b)
    assert sid_b2 == mgr.canonical_sid(sid_b)
    rec = mgr.active_tracks.get((0, 2))
    assert rec and int(rec.get("stable_id")) == mgr.canonical_sid(sid_b)


def test_alias_appends_embeddings_to_canonical_gallery(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, gallery_size=5, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=3)
    emb_b = _unit_embedding(seed=4)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    _update_with_embedding(mgr, 0, 1, 2.0, emb_a)
    _update_with_embedding(mgr, 0, 2, 2.0, emb_b)

    before = len(mgr.gallery.get(int(sid_a), []))
    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]
    after = len(mgr.gallery.get(int(res["canonical"]), []))
    assert after > before
    assert after <= mgr.gallery_size


def test_alias_merges_pose_state(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        pose_enabled=True,
        pose_min_features=1,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
    )
    emb_a = _unit_embedding(seed=10)
    emb_b = _unit_embedding(seed=11)
    sid_a = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=1.0,
        zone=None,
        frame_bgr=None,
        embedding=emb_a,
        pose_features=_pose_features(1.0),
        pose_quality=_pose_quality(),
    )
    sid_b = mgr.update(
        sensor_id=0,
        ds_obj_id=2,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=2.0,
        zone=None,
        frame_bgr=None,
        embedding=emb_b,
        pose_features=_pose_features(2.0),
        pose_quality=_pose_quality(),
    )
    len_a = len(mgr.pose_gallery.get(int(sid_a), []))
    len_b = len(mgr.pose_gallery.get(int(sid_b), []))

    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]
    canon = res["canonical"]
    assert len(mgr.pose_gallery.get(int(canon), [])) >= (len_a + len_b)
    assert mgr.pose_last_seen.get(int(canon)) == max(
        mgr.pose_last_seen.get(int(sid_a), 0.0),
        mgr.pose_last_seen.get(int(sid_b), 0.0),
    )


def test_alias_embedding_merge_respects_maxlen(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, gallery_size=10, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=21)
    emb_b = _unit_embedding(seed=22)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 101.0, emb_b)
    for i in range(2, 9):
        _update_with_embedding(mgr, 0, 1, float(i), emb_a)
    for i in range(102, 109):
        _update_with_embedding(mgr, 0, 2, float(i), emb_b)

    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]
    canon = res["canonical"]
    merged = list(mgr.gallery.get(int(canon), []))
    ts_list = [float(ts) for (ts, _emb) in merged]
    assert len(ts_list) == 10
    assert ts_list == sorted(ts_list, reverse=True)
    assert min(ts_list) == 7.0


def test_gallery_stores_timestamps(tmp_path) -> None:
    mgr = _make_mgr(tmp_path)
    emb = _unit_embedding(seed=30)
    sid = _update_with_embedding(mgr, 0, 1, 5.0, emb)
    entry = list(mgr.gallery.get(int(sid), []))[0]
    assert isinstance(entry, tuple)
    assert float(entry[0]) == 5.0


def test_suggest_aliases_filters_copresence(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, min_embeddings_for_suggest=1, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb = _unit_embedding(seed=40)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb)
    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=2.0)
    mgr.active_zones.clear()
    mgr.observe_copresence([sid_a, sid_b], ts=100.0)
    candidates = mgr.suggest_aliases(min_sim=0.0, limit=5, require_inactive=True, now_ts=101.0)
    assert candidates
    assert candidates[0]["blocked"]
    assert candidates[0]["block_reason"] == "copresent_recently"


def test_suggest_aliases_mutual_nearest_neighbor(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, min_embeddings_for_suggest=1, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    e1 = np.zeros((512,), dtype=np.float32)
    e1[0] = 1.0
    e2 = np.zeros((512,), dtype=np.float32)
    e2[1] = 1.0
    emb_a = e1
    emb_b = (e1 + 0.01 * e2) / float(np.linalg.norm(e1 + 0.01 * e2) + 1e-12)
    emb_c = e2
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    sid_c = _update_with_embedding(mgr, 0, 3, 1.0, emb_c)
    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=2.0)
    mgr.active_zones.clear()
    candidates = mgr.suggest_aliases(min_sim=0.0, limit=10, require_inactive=True, now_ts=3.0)
    assert len(candidates) == 1
    pair = {candidates[0]["a"], candidates[0]["b"]}
    assert pair == {sid_a, sid_b}
    assert not candidates[0]["blocked"]
    assert sid_c not in pair


def test_suggest_aliases_min_embedding_gate(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, min_embeddings_for_suggest=3, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb = _unit_embedding(seed=50)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb)
    _update_with_embedding(mgr, 0, 1, 2.0, emb)
    _update_with_embedding(mgr, 0, 1, 3.0, emb)
    _update_with_embedding(mgr, 0, 2, 1.0, emb)
    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=4.0)
    candidates = mgr.suggest_aliases(min_sim=0.0, limit=5, require_inactive=True, now_ts=5.0)
    assert candidates == []


def test_batch_merge_connected_components(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=60)
    emb_b = _unit_embedding(seed=61)
    emb_c = _unit_embedding(seed=62)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    sid_c = _update_with_embedding(mgr, 0, 3, 1.0, emb_c)
    payload = mgr.set_aliases_batch(
        [{"a": sid_a, "b": sid_b}, {"a": sid_b, "b": sid_c}],
        force=True,
        now_ts=10.0,
    )
    assert payload["applied_count"] == 2
    canonical = min(sid_a, sid_b, sid_c)
    assert mgr.canonical_sid(sid_b) == canonical
    assert mgr.canonical_sid(sid_c) == canonical


def test_batch_merge_conflicting_canonicals_fails(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=70)
    emb_b = _unit_embedding(seed=71)
    emb_c = _unit_embedding(seed=72)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    sid_c = _update_with_embedding(mgr, 0, 3, 1.0, emb_c)
    payload = mgr.set_aliases_batch(
        [
            {"a": sid_a, "b": sid_b, "canonical": sid_a},
            {"a": sid_a, "b": sid_c, "canonical": sid_c},
        ],
        force=True,
        now_ts=10.0,
    )
    assert payload.get("error") == "conflicting_canonical"
    assert mgr.list_aliases() == {}


def test_batch_merge_cycle_detection(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=80)
    emb_b = _unit_embedding(seed=81)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    payload = mgr.set_aliases_batch(
        [{"a": sid_a, "b": sid_b}, {"a": sid_b, "b": sid_a}],
        force=True,
        now_ts=10.0,
    )
    reasons = [item.get("reason") for item in payload.get("results", [])]
    assert "cycle_detected" in reasons


def test_alias_persistence_roundtrip(tmp_path) -> None:
    alias_file = tmp_path / "aliases.json"
    mgr = _make_mgr(tmp_path, alias_file=str(alias_file), cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=90)
    emb_b = _unit_embedding(seed=91)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]

    mgr2 = _make_mgr(tmp_path, alias_file=str(alias_file))
    assert mgr2.canonical_sid(sid_b) == mgr2.canonical_sid(sid_a)


def test_alias_registry_rejects_duplicate_root_keys(tmp_path) -> None:
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(
        '{"aliases":{},"aliases":{"101":"102"},"history":[]}',
        encoding="utf-8",
    )

    mgr = _make_mgr(tmp_path, alias_file=str(alias_file))

    assert mgr.list_aliases() == {}
    assert mgr.canonical_sid(101) == 101


def test_alias_history_persisted(tmp_path) -> None:
    alias_file = tmp_path / "aliases.json"
    mgr = _make_mgr(tmp_path, alias_file=str(alias_file), cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=100)
    emb_b = _unit_embedding(seed=101)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]

    mgr2 = _make_mgr(tmp_path, alias_file=str(alias_file))
    assert any(entry.get("action") == "merge" for entry in mgr2.alias_history)


def test_alias_reserved_ids_not_reused(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=110)
    emb_b = _unit_embedding(seed=111)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]
    mgr._free_sids = [sid_a, sid_b]
    mgr._free_sids_set = {sid_a, sid_b}
    reused = mgr._alloc_sid()
    assert reused not in {sid_a, sid_b}


def test_unset_alias_removes_mapping(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=120)
    emb_b = _unit_embedding(seed=121)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]
    mgr.unset_alias(sid_b)
    assert mgr.canonical_sid(sid_b) == sid_b


def test_auto_merge_reduces_id_count_when_over_limit(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        max_total_ids=2,
        auto_merge_enabled=True,
        auto_merge_interval_s=0.0,
        auto_merge_max_attempts=10,
        auto_merge_min_sim=0.90,
        auto_merge_respect_inactive=False,
        auto_merge_force_stuck=True,
        suggest_mnn_margin=0.0,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
        min_embeddings_for_suggest=1,
    )
    emb = _vec2d(1.0, 0.0)
    for i in range(6):
        _update_with_embedding(mgr, 0, i + 1, float(i), emb)

    metrics = mgr.get_sid_metrics()
    assert metrics["canonical_gallery_size"] <= 3
    assert metrics["auto_merge_applied"] > 0
    assert metrics["auto_merge_runs"] == metrics["auto_merge_triggers"]


def test_auto_merge_interval_prevents_immediate_repeat_runs(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        max_total_ids=2,
        auto_merge_enabled=True,
        auto_merge_interval_s=5.0,
        auto_merge_max_attempts=10,
        auto_merge_min_sim=0.90,
        auto_merge_respect_inactive=False,
        auto_merge_force_stuck=True,
        suggest_mnn_margin=0.0,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
        min_embeddings_for_suggest=1,
    )
    emb = _vec2d(1.0, 0.0)
    for i in range(4):
        _update_with_embedding(mgr, 0, i + 1, float(i), emb)

    runs_after_first = mgr.get_sid_metrics()["auto_merge_runs"]

    _update_with_embedding(mgr, 0, 5, 4.4, emb)
    metrics = mgr.get_sid_metrics()
    assert metrics["auto_merge_runs"] == runs_after_first
    assert metrics["canonical_gallery_size"] >= 3

    _update_with_embedding(mgr, 0, 6, 9.9, emb)
    metrics = mgr.get_sid_metrics()
    assert metrics["auto_merge_runs"] > runs_after_first
    assert metrics["canonical_gallery_size"] <= 4


def test_auto_merge_respects_active_guard_by_default(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        max_total_ids=2,
        auto_merge_enabled=True,
        auto_merge_interval_s=0.0,
        auto_merge_max_attempts=10,
        auto_merge_min_sim=0.90,
        auto_merge_respect_inactive=True,
        suggest_mnn_margin=0.0,
        auto_merge_force_stuck=False,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
        min_embeddings_for_suggest=1,
    )
    emb = _vec2d(1.0, 0.0)

    # First 3 active tracks exceed the max ID cap.
    for i in range(3):
        _update_with_embedding(mgr, 0, i + 1, float(i), emb)

    _update_with_embedding(mgr, 0, 4, 3.1, emb)
    metrics = mgr.get_sid_metrics()
    assert metrics["auto_merge_runs"] >= 1
    assert metrics["auto_merge_applied"] == 0
    assert metrics["auto_merge_suppressed"] > 0
    assert metrics["canonical_gallery_size"] > 2

    # Once tracks disappear and are inactive, auto-merge can apply immediately.
    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=10.0)
    _update_with_embedding(mgr, 0, 5, 10.1, emb)
    metrics = mgr.get_sid_metrics()
    assert metrics["auto_merge_applied"] >= 1
    assert metrics["canonical_gallery_size"] <= 3


def test_auto_merge_uses_own_min_embedding_gate(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        max_total_ids=2,
        auto_merge_enabled=True,
        auto_merge_interval_s=0.0,
        auto_merge_max_attempts=10,
        auto_merge_min_sim=0.90,
        auto_merge_min_embeddings_for_suggest=1,
        auto_merge_respect_inactive=False,
        auto_merge_force_stuck=True,
        suggest_mnn_margin=0.0,
        min_embeddings_for_suggest=3,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
    )
    emb = _vec2d(1.0, 0.0)
    for i in range(3):
        _update_with_embedding(mgr, 0, i + 1, float(i), emb)

    # Manual/default suggestion path still respects min_embeddings_for_suggest=3.
    manual = mgr.suggest_aliases(min_sim=0.0, limit=10, require_inactive=False, now_ts=3.0)
    assert manual == []

    # Auto-merge path should still run with its own support gate of 1.
    _update_with_embedding(mgr, 0, 4, 4.0, emb)
    metrics = mgr.get_sid_metrics()
    assert metrics["auto_merge_applied"] > 0
    assert metrics["auto_merge_last_support_gate"] == 1


def test_auto_merge_collapses_new_sid_same_update_call(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        max_total_ids=1,
        auto_merge_enabled=True,
        auto_merge_interval_s=0.0,
        auto_merge_max_attempts=10,
        auto_merge_min_sim=0.90,
        auto_merge_min_embeddings_for_suggest=1,
        auto_merge_respect_inactive=False,
        auto_merge_force_stuck=True,
        suggest_mnn_margin=0.0,
        min_embeddings_for_suggest=3,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
    )
    emb = _vec2d(1.0, 0.0)
    sid1 = _update_with_embedding(mgr, 0, 1, 1.0, emb)
    sid2 = _update_with_embedding(mgr, 0, 2, 2.0, emb)
    assert sid1 == 1
    # Post-create auto-merge should canonicalize immediately.
    assert sid2 == sid1
    metrics = mgr.get_sid_metrics()
    assert metrics["canonical_gallery_size"] == 1
    assert metrics["auto_merge_applied"] >= 1


def test_auto_merge_can_merge_both_active_when_enabled(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        max_total_ids=2,
        auto_merge_enabled=True,
        auto_merge_interval_s=0.0,
        auto_merge_max_attempts=10,
        auto_merge_min_sim=0.90,
        auto_merge_min_embeddings_for_suggest=1,
        auto_merge_respect_inactive=False,
        auto_merge_force_stuck=False,
        auto_merge_allow_both_active=True,
        auto_merge_both_active_min_sim=0.90,
        suggest_mnn_margin=0.0,
        min_embeddings_for_suggest=3,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
    )
    emb = _vec2d(1.0, 0.0)
    for i in range(4):
        _update_with_embedding(mgr, 0, i + 1, float(i), emb)
    metrics = mgr.get_sid_metrics()
    assert metrics["auto_merge_applied"] >= 1
    assert metrics["canonical_gallery_size"] <= 3


def test_auto_merge_both_active_still_respects_copresence_guard(tmp_path) -> None:
    mgr = _make_mgr(
        tmp_path,
        max_total_ids=2,
        auto_merge_enabled=True,
        auto_merge_interval_s=0.0,
        auto_merge_max_attempts=10,
        auto_merge_min_sim=0.90,
        auto_merge_min_embeddings_for_suggest=1,
        auto_merge_respect_inactive=False,
        auto_merge_force_stuck=False,
        auto_merge_allow_both_active=True,
        auto_merge_both_active_min_sim=0.90,
        copresence_window_s=600.0,
        suggest_mnn_margin=0.0,
        min_embeddings_for_suggest=1,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
    )
    emb = _vec2d(1.0, 0.0)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb)
    sid_b = _update_with_embedding(mgr, 0, 2, 2.0, emb)
    mgr.observe_copresence([sid_a, sid_b], ts=2.0)
    _update_with_embedding(mgr, 0, 3, 3.0, emb)
    metrics = mgr.get_sid_metrics()
    assert metrics["auto_merge_applied"] == 0
    assert metrics["auto_merge_suppressed"] >= 1
    assert metrics["canonical_gallery_size"] >= 3


def test_reset_sid_pool_on_start(tmp_path) -> None:
    sid_pool_path = tmp_path / "sid_pool.json"
    sid_pool_path.write_text(json.dumps({"free_sids": [101, 102, 103]}), encoding="utf-8")
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text("{}", encoding="utf-8")

    mgr = _make_mgr(
        tmp_path,
        sid_pool_file=str(sid_pool_path),
        alias_file=str(alias_path),
        reset_sid_pool_on_start=True,
        auto_merge_enabled=False,
    )

    # Existing alias state is unchanged and should still be readable.
    assert mgr.aliases_enabled
    metrics = mgr.get_sid_metrics()
    assert metrics["free_sid_pool_size"] == 0
    assert not sid_pool_path.exists()

    sid = _update_with_embedding(mgr, 0, 1, 1.0, _vec2d(1.0, 0.0))
    assert sid == 1


def test_thread_safety_concurrent_merge_update(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=130)
    emb_b = _unit_embedding(seed=131)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    errors = []

    def _updater() -> None:
        try:
            for i in range(20):
                _update_with_embedding(mgr, 0, 1, 2.0 + i, emb_a)
        except Exception as exc:
            errors.append(exc)

    def _merger() -> None:
        try:
            mgr.set_alias(sid_b, sid_a, force=True)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_updater)
    t2 = threading.Thread(target=_merger)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors


def test_sid_global_first_seen_set_on_first_append(tmp_path) -> None:
    mgr = _make_mgr(tmp_path)
    emb = _unit_embedding(seed=140)
    sid = _update_with_embedding(mgr, 0, 1, 5.0, emb)
    assert mgr.sid_global_first_seen.get(int(sid)) == 5.0


def test_sid_global_first_seen_preserved_on_merge(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=150)
    emb_b = _unit_embedding(seed=151)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 5.0, emb_b)
    mgr.sid_global_first_seen[int(sid_a)] = 1.0
    mgr.sid_global_first_seen[int(sid_b)] = 5.0
    res = mgr.set_alias(sid_b, sid_a, force=True)
    assert res["applied"]
    assert mgr.sid_global_first_seen.get(int(res["canonical"])) == 1.0


def test_sid_global_first_seen_cleared_on_reuse(tmp_path) -> None:
    mgr = _make_mgr(tmp_path)
    emb = _unit_embedding(seed=160)
    sid = _update_with_embedding(mgr, 0, 1, 1.0, emb)
    assert sid in mgr.sid_global_first_seen
    mgr._purge_sid_state(sid)
    mgr._free_sid(sid)
    reused = mgr._alloc_sid()
    assert reused == sid
    assert sid not in mgr.sid_global_first_seen


def test_alias_history_capped(tmp_path) -> None:
    alias_file = tmp_path / "aliases.json"
    mgr = _make_mgr(
        tmp_path,
        alias_file=str(alias_file),
        alias_history_max=5,
        cos_sim_threshold=1.1,
        cos_sim_high_threshold=1.1,
    )
    for i in range(10):
        emb_a = _unit_embedding(seed=200 + i * 2)
        emb_b = _unit_embedding(seed=200 + i * 2 + 1)
        sid_a = _update_with_embedding(mgr, 0, 100 + i * 2, float(i), emb_a)
        sid_b = _update_with_embedding(mgr, 0, 101 + i * 2, float(i), emb_b)
        mgr.set_alias(sid_b, sid_a, force=True)
    mgr2 = _make_mgr(tmp_path, alias_file=str(alias_file), alias_history_max=5)
    assert len(mgr2.alias_history) == 5


def test_unset_alias_preserves_src_state(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=300)
    emb_b = _unit_embedding(seed=301)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    _update_with_embedding(mgr, 0, 1, 2.0, emb_a)
    before_gallery = list(mgr.gallery.get(int(sid_b), []))
    before_centroid = mgr.sid_centroid.get(int(sid_b)).copy()
    mgr.set_alias(sid_b, sid_a, force=True)
    mgr.unset_alias(sid_b)
    after_gallery = list(mgr.gallery.get(int(sid_b), []))
    after_centroid = mgr.sid_centroid.get(int(sid_b)).copy()
    assert len(before_gallery) == len(after_gallery)
    assert np.allclose(before_centroid, after_centroid)
