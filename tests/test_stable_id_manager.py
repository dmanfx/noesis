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
