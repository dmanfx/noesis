"""Phase 2 mutual-nearest and frame-claim assignment guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reid.assignment import assign_tracklets_mnn, is_mutual_nearest_match
from reid.stable_id_manager import StableIDManager

TOPOLOGY = Path(__file__).resolve().parents[1] / "config" / "camera_topology.yaml"


def _unit(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr / (float(np.linalg.norm(arr)) + 1e-12)


def _basis(dim: int, idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _household_mgr(tmp_path, **kwargs) -> StableIDManager:
    params = dict(
        use_extractor=False,
        compute_backend="cpu",
        household_mode=True,
        camera_topology_file=str(TOPOLOGY),
        new_id_hysteresis_frames=1,
        household_confirm_embeddings=1,
        cos_sim_threshold=0.55,
        cos_sim_high_threshold=0.60,
        gallery_size=8,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
        residents_file=str(tmp_path / "residents.json"),
        visitor_pool_file=str(tmp_path / "visitor_pool.json"),
    )
    params.update(kwargs)
    return StableIDManager(**params)


def test_mnn_rejects_ambiguous_pair() -> None:
    dim = 4
    sid_a, sid_b = 101, 102
    emb_a = _unit(_basis(dim, 0))
    emb_b = _unit(_basis(dim, 1))
    # Track 1 leans to A; track 2 is a stronger match for A (non-mutual for track 1 → A).
    emb_track1 = _unit(0.95 * _basis(dim, 0) + 0.31 * _basis(dim, 1))
    emb_track2 = _unit(0.99 * _basis(dim, 0) + 0.14 * _basis(dim, 1))

    def sim_fn(emb: np.ndarray, candidates: list[int]) -> dict[int, float]:
        gallery = {sid_a: emb_a, sid_b: emb_b}
        out = {}
        for sid in candidates:
            out[int(sid)] = float(np.dot(emb, gallery[int(sid)]))
        return out

    peers = {
        (0, 1): emb_track1,
        (0, 2): emb_track2,
    }
    assert not is_mutual_nearest_match((0, 1), emb_track1, sid_a, peers, [sid_a, sid_b], sim_fn)

    assigned = assign_tracklets_mnn(peers, [sid_a, sid_b], sim_fn, min_score=0.5)
    assert (0, 1) not in assigned or assigned.get((0, 1), (0, 0.0))[0] != sid_a


def test_reject_mnn_when_identity_prefers_other_track(tmp_path) -> None:
    from collections import deque

    mgr = _household_mgr(tmp_path)
    dim = 8
    sid = 1001
    emb_owner = _unit(_basis(dim, 0))
    emb_impostor = _unit(0.92 * _basis(dim, 0) + 0.38 * _basis(dim, 1))

    mgr.gallery[sid] = deque([(1.0, emb_owner)], maxlen=8)
    mgr.sid_centroid[sid] = emb_owner
    mgr.active_tracks[(0, 1)] = {
        "emb": emb_owner,
        "last_seen_ts": 1.0,
        "stable_id": sid,
    }
    mgr.active_tracks[(0, 2)] = {
        "emb": emb_impostor,
        "last_seen_ts": 1.0,
        "stable_id": 9001,
    }

    reject, reason = mgr._reject_household_gallery_match(
        track_key=(0, 2),
        emb=emb_impostor,
        best_sid=sid,
        ts=1.0,
        sensor_id=0,
        world_xy=None,
        world_valid=False,
        appearance_sim=0.95,
    )
    assert reject
    assert reason == "mnn_conflict"
    assert mgr._mnn_reject_count >= 1
