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


def _household_mgr(tmp_path) -> StableIDManager:
    return StableIDManager(
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


def test_overlap_permit_grants_dual_active_kitchen_family(tmp_path) -> None:
    """Kitchen (1) + family-room (2) overlap with consistent world points."""
    mgr = _household_mgr(tmp_path)
    emb = _unit_embedding(seed=33)

    sid_kitchen = mgr.update(
        sensor_id=1,
        ds_obj_id=1,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=10.0,
        zone=None,
        embedding=emb,
        world_xy=(1.0, 2.0),
        world_valid=True,
    )
    assert sid_kitchen > 0

    sid_family = mgr.update(
        sensor_id=2,
        ds_obj_id=1,
        bbox_ltrbwh=(12.0, 12.0, 52.0, 118.0),
        ts=10.1,
        zone=None,
        embedding=emb,
        world_xy=(1.2, 2.1),
        world_valid=True,
    )
    assert sid_family == sid_kitchen

    diag = mgr.get_track_diagnostics(sensor_id=2, ds_obj_id=1)
    assert diag.get("overlap_permit") is True

    metrics = mgr.get_sid_metrics()
    assert metrics["overlap_permit_grant_count"] >= 1
    assert metrics["active_unique"] == 1


def test_overlap_permit_denies_distant_world_points(tmp_path) -> None:
    mgr = _household_mgr(tmp_path)
    emb = _unit_embedding(seed=44)

    sid_kitchen = mgr.update(
        sensor_id=1,
        ds_obj_id=1,
        bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
        ts=20.0,
        zone=None,
        embedding=emb,
        world_xy=(0.0, 0.0),
        world_valid=True,
    )

    sid_family = mgr.update(
        sensor_id=2,
        ds_obj_id=2,
        bbox_ltrbwh=(12.0, 12.0, 52.0, 118.0),
        ts=20.1,
        zone=None,
        embedding=emb,
        world_xy=(10.0, 10.0),
        world_valid=True,
    )
    assert sid_family != sid_kitchen

    granted, reason = mgr.overlap_permit(
        sid_kitchen,
        sensor_id=2,
        world_xy=(10.0, 10.0),
        ts=20.1,
        world_valid=True,
        appearance_sim=1.0,
    )
    assert granted is False
    assert reason == "world_dist"
