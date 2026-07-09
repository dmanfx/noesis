from __future__ import annotations

from pathlib import Path

import numpy as np

from reid.stable_id_manager import StableIDManager

TOPOLOGY = Path(__file__).resolve().parents[1] / "config" / "camera_topology.yaml"


def _unit_embedding(dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(dim,)).astype(np.float32)
    emb /= float(np.linalg.norm(emb) + 1e-12)
    return emb


def _household_mgr(tmp_path, **kwargs) -> StableIDManager:
    # confirm_embeddings=1 so the first update mints a visitor and populates the
    # gallery — exclusivity/overlap only apply on gallery match, not provisional.
    params = dict(
        use_extractor=False,
        compute_backend="cpu",
        household_mode=True,
        camera_topology_file=str(TOPOLOGY),
        new_id_hysteresis_frames=1,
        household_confirm_embeddings=1,
        cos_sim_threshold=0.60,
        cos_sim_high_threshold=0.70,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
        visitor_pool_file=str(tmp_path / "visitor_pool.json"),
        residents_file=str(tmp_path / "residents.json"),
        gallery_persist_file=str(tmp_path / "gallery.npz"),
        embed_interval_s=0.0,
    )
    params.update(kwargs)
    return StableIDManager(**params)


def test_household_blocks_cross_camera_share_without_overlap(tmp_path) -> None:
    """Living-room (0) vs kitchen (1): no overlap → distinct SIDs."""
    mgr = _household_mgr(tmp_path)
    emb = _unit_embedding(seed=11)

    sid_a = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=1.0,
        zone=None,
        embedding=emb,
        world_xy=(0.0, 0.0),
        world_valid=True,
    )
    assert sid_a > 0

    sid_b = mgr.update(
        sensor_id=1,
        ds_obj_id=1,
        bbox_ltrbwh=(12.0, 12.0, 52.0, 118.0),
        ts=1.1,
        zone=None,
        embedding=emb,
        world_xy=(5.0, 5.0),
        world_valid=True,
    )
    assert sid_b != sid_a

    metrics = mgr.get_sid_metrics()
    assert metrics["false_share_blocked_count"] >= 1
    assert metrics["overlap_permit_deny_count"] >= 1
    assert metrics["household_mode"] is True


def test_legacy_mode_allows_multi_zone_active(tmp_path) -> None:
    """Backward compat: allow_multi_zone_active=True still shares across cameras."""
    mgr = StableIDManager(
        use_extractor=False,
        compute_backend="cpu",
        household_mode=False,
        allow_multi_zone_active=True,
        new_id_hysteresis_frames=1,
        cos_sim_threshold=0.60,
        cos_sim_high_threshold=0.70,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
    )
    emb = _unit_embedding(seed=22)
    sid0 = mgr.update(
        sensor_id=0,
        ds_obj_id=1,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=1.0,
        zone=None,
        embedding=emb,
    )
    sid1 = mgr.update(
        sensor_id=1,
        ds_obj_id=1,
        bbox_ltrbwh=(12.0, 12.0, 52.0, 118.0),
        ts=2.0,
        zone=None,
        embedding=emb,
    )
    assert sid1 == sid0
