from __future__ import annotations

import numpy as np

from reid.stable_id_manager import StableIDManager


def _unit_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(dim,)).astype(np.float32)
    emb /= float(np.linalg.norm(emb) + 1e-12)
    return emb


def test_stable_id_gpu_backend_fallback_when_unavailable(tmp_path) -> None:
    try:
        mgr = StableIDManager(
            use_extractor=False,
            compute_backend="gpu",
            gpu_device="cuda:0",
            sid_pool_file=str(tmp_path / "sid_pool.json"),
        )
    except RuntimeError:
        return
    metrics = mgr.get_sid_metrics()
    mode = str(metrics.get("stableid_backend_mode") or "")
    assert mode == "gpu"


def test_stable_id_assignment_parity_cpu_vs_auto(tmp_path) -> None:
    kwargs = dict(
        use_extractor=False,
        cos_sim_threshold=0.60,
        cos_sim_high_threshold=0.70,
        new_id_hysteresis_frames=1,
        embed_interval_s=0.0,
    )
    mgr_cpu = StableIDManager(compute_backend="cpu", sid_pool_file=str(tmp_path / "sid_cpu.json"), **kwargs)
    mgr_auto = StableIDManager(compute_backend="auto", sid_pool_file=str(tmp_path / "sid_auto.json"), **kwargs)

    seq = [
        (0, 1, 1.0, _unit_embedding(seed=11)),
        (0, 1, 2.0, _unit_embedding(seed=11)),
        (1, 9, 3.0, _unit_embedding(seed=11)),
        (1, 9, 4.0, _unit_embedding(seed=11)),
        (0, 2, 5.0, _unit_embedding(seed=22)),
        (1, 10, 6.0, _unit_embedding(seed=22)),
    ]

    cpu_ids = []
    auto_ids = []
    for sensor_id, ds_obj_id, ts, emb in seq:
        cpu_ids.append(
            mgr_cpu.update(
                sensor_id=sensor_id,
                ds_obj_id=ds_obj_id,
                bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
                ts=ts,
                zone=None,
                frame_bgr=None,
                embedding=emb,
            )
        )
        auto_ids.append(
            mgr_auto.update(
                sensor_id=sensor_id,
                ds_obj_id=ds_obj_id,
                bbox_ltrbwh=(10.0, 10.0, 50.0, 120.0),
                ts=ts,
                zone=None,
                frame_bgr=None,
                embedding=emb,
            )
        )
    assert auto_ids == cpu_ids
