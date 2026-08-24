from __future__ import annotations

import numpy as np

from reid.stable_id_manager import StableIDManager


def _unit_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(dim,)).astype(np.float32)
    emb /= float(np.linalg.norm(emb) + 1e-12)
    return emb


def test_stable_id_external_embedding_cross_camera_match(tmp_path) -> None:
    mgr = StableIDManager(
        use_extractor=False,
        new_id_hysteresis_frames=1,
        cos_sim_threshold=0.60,
        cos_sim_high_threshold=0.70,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    emb = _unit_embedding(seed=123)
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=1.0,
        zone=None,
        frame_bgr=None,
        embedding=emb,
    )
    assert sid0 > 0

    sid1 = mgr.update(
        sensor_id=1,
        ds_obj_id=1,
        bbox_ltrbwh=(12.0, 12.0, 52.0, 118.0),
        ts=2.0,
        zone=None,
        frame_bgr=None,
        embedding=emb,
    )
    assert sid1 == sid0


def test_stable_id_ghost_reassociation_same_camera(tmp_path) -> None:
    mgr = StableIDManager(
        use_extractor=False,
        new_id_hysteresis_frames=1,
        max_ghost_age_s=60.0,
        cos_sim_threshold=0.60,
        cos_sim_high_threshold=0.70,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    emb = _unit_embedding(seed=456)
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=11,
        bbox_ltrbwh=(100.0, 50.0, 40.0, 110.0),
        ts=10.0,
        zone="zone_a",
        frame_bgr=None,
        embedding=emb,
    )
    assert sid0 > 0

    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=11.0)
    mgr.prune_ghosts(now_ts=11.0)

    sid1 = mgr.update(
        sensor_id=0,
        ds_obj_id=12,
        bbox_ltrbwh=(100.0, 50.0, 40.0, 110.0),
        ts=12.0,
        zone="zone_a",
        frame_bgr=None,
        embedding=emb,
    )
    assert sid1 == sid0


def test_stable_id_tracker_lifecycle_continuity_without_fresh_embedding(tmp_path) -> None:
    """A short tracker restart must not expose a new provisional identity."""
    mgr = StableIDManager(
        use_extractor=False,
        new_id_hysteresis_frames=1,
        max_ghost_age_s=60.0,
        cos_sim_threshold=0.60,
        cos_sim_high_threshold=0.70,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    emb = _unit_embedding(seed=789)
    bbox = (100.0, 50.0, 40.0, 110.0)
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=19,
        bbox_ltrbwh=bbox,
        ts=10.0,
        zone="family",
        frame_bgr=None,
        embedding=emb,
    )
    assert sid0 > 0

    # The tracker disappears for one frame.  No fresh SGIE tensor is supplied
    # when the same tracker-local ID returns.
    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=10.05)
    sid1 = mgr.update(
        sensor_id=0,
        ds_obj_id=19,
        bbox_ltrbwh=(101.0, 50.5, 40.0, 109.0),
        ts=10.10,
        zone="family",
        frame_bgr=None,
        embedding=None,
    )
    assert sid1 == sid0
    diag = mgr.get_track_diagnostics(sensor_id=0, ds_obj_id=19)
    assert diag["id_event"] == "match_ghost_continuity"
    assert diag["embedding_present"] is False


def test_stable_id_tracker_continuity_is_strict_and_time_bounded(tmp_path) -> None:
    mgr = StableIDManager(
        use_extractor=False,
        new_id_hysteresis_frames=1,
        max_ghost_age_s=60.0,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    emb = _unit_embedding(seed=890)
    bbox = (100.0, 50.0, 40.0, 110.0)
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=19,
        bbox_ltrbwh=bbox,
        ts=20.0,
        zone=None,
        frame_bgr=None,
        embedding=emb,
    )
    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=20.05)

    # A different tracker-local ID cannot inherit the ghost without ReID.
    sid_different_tracker = mgr.update(
        sensor_id=0,
        ds_obj_id=20,
        bbox_ltrbwh=bbox,
        ts=20.10,
        zone=None,
        frame_bgr=None,
        embedding=None,
    )
    assert sid_different_tracker != sid0

    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=20.15)
    # The same tracker ID outside the bounded handoff window also gets a new
    # identity; the continuity path must not become a long-lived appearance-
    # free matcher.
    sid_expired = mgr.update(
        sensor_id=0,
        ds_obj_id=19,
        bbox_ltrbwh=bbox,
        ts=20.95,
        zone=None,
        frame_bgr=None,
        embedding=None,
    )
    assert sid_expired != sid0
