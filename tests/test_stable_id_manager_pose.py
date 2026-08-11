from __future__ import annotations

import numpy as np

from reid.stable_id_manager import StableIDManager


def _pose_features(keys, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {str(k): float(rng.uniform(0.2, 1.0)) for k in keys}


def _quality(mean_conf: float = 0.9, valid_frac: float = 0.9) -> dict[str, float]:
    return {"kpt_mean_conf": float(mean_conf), "kpt_valid_frac": float(valid_frac)}


def test_pose_only_cross_camera_match(tmp_path) -> None:
    keys = getattr(StableIDManager, "_POSE_FEATURE_KEYS", ())
    features = _pose_features(keys, seed=123)
    mgr = StableIDManager(
        use_extractor=False,
        pose_enabled=True,
        pose_only_threshold=0.7,
        pose_sim_threshold=0.55,
        pose_sim_high_threshold=0.65,
        pose_min_features=6,
        pose_min_mean_conf=0.2,
        pose_min_valid_frac=0.2,
        new_id_hysteresis_frames=1,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=(10.0, 10.0, 40.0, 120.0),
        ts=1.0,
        zone=None,
        frame_bgr=None,
        embedding=None,
        pose_features=features,
        pose_quality=_quality(),
    )
    sid1 = mgr.update(
        sensor_id=1,
        ds_obj_id=1,
        bbox_ltrbwh=(12.0, 12.0, 42.0, 118.0),
        ts=2.0,
        zone=None,
        frame_bgr=None,
        embedding=None,
        pose_features=features,
        pose_quality=_quality(),
    )
    assert sid1 == sid0


def test_pose_quality_blocks_match(tmp_path) -> None:
    keys = getattr(StableIDManager, "_POSE_FEATURE_KEYS", ())
    features = _pose_features(keys, seed=456)
    mgr = StableIDManager(
        use_extractor=False,
        pose_enabled=True,
        pose_only_threshold=0.7,
        pose_min_features=6,
        pose_min_mean_conf=0.8,
        pose_min_valid_frac=0.8,
        new_id_hysteresis_frames=1,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=(20.0, 15.0, 40.0, 120.0),
        ts=1.0,
        zone=None,
        frame_bgr=None,
        embedding=None,
        pose_features=features,
        pose_quality=_quality(mean_conf=0.2, valid_frac=0.2),
    )
    sid1 = mgr.update(
        sensor_id=1,
        ds_obj_id=1,
        bbox_ltrbwh=(22.0, 16.0, 42.0, 118.0),
        ts=2.0,
        zone=None,
        frame_bgr=None,
        embedding=None,
        pose_features=features,
        pose_quality=_quality(mean_conf=0.2, valid_frac=0.2),
    )
    assert sid1 != sid0


def test_pose_gallery_prunes_by_age(tmp_path) -> None:
    keys = getattr(StableIDManager, "_POSE_FEATURE_KEYS", ())
    features = _pose_features(keys, seed=789)
    mgr = StableIDManager(
        use_extractor=False,
        pose_enabled=True,
        pose_max_age_s=1.0,
        pose_gallery_size=2,
        new_id_hysteresis_frames=1,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=(5.0, 5.0, 30.0, 90.0),
        ts=0.0,
        zone=None,
        frame_bgr=None,
        embedding=None,
        pose_features=features,
        pose_quality=_quality(),
    )
    assert mgr.pose_gallery.get(int(sid0))
    mgr.prune_ghosts(now_ts=10.0)
    assert not mgr.pose_gallery.get(int(sid0))
