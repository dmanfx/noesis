"""Phase 2 gallery quality gates and clustered exemplar storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reid.gallery_quality import add_clustered_exemplar, should_accept_gallery_embedding
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
        gallery_size=8,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
        residents_file=str(tmp_path / "residents.json"),
        visitor_pool_file=str(tmp_path / "visitor_pool.json"),
    )
    params.update(kwargs)
    return StableIDManager(**params)


def test_quality_gate_rejects_small_crop_household() -> None:
    ok, reason = should_accept_gallery_embedding(
        household_mode=True,
        bbox=(0.0, 0.0, 40.0, 32.0),
        blur_var=100.0,
        pose_quality=None,
        identity_state="visitor",
        identity_quality="strong",
        sid_has_gallery=True,
        min_crop_h=64,
        min_laplacian_var=12.0,
    )
    assert not ok
    assert reason == "crop_too_small"


def test_quality_gate_legacy_permissive() -> None:
    ok, reason = should_accept_gallery_embedding(
        household_mode=False,
        bbox=(0.0, 0.0, 10.0, 10.0),
        blur_var=1.0,
        pose_quality=None,
        identity_state="provisional",
        identity_quality="weak_no_embedding",
        sid_has_gallery=False,
        min_crop_h=64,
        min_laplacian_var=12.0,
    )
    assert ok
    assert reason is None


def test_clustered_exemplar_keeps_distant_modes() -> None:
    from collections import deque

    dim = 8
    dq: deque = deque(maxlen=3)
    modes = [_unit(_basis(dim, i)) for i in range(3)]
    for i, emb in enumerate(modes):
        add_clustered_exemplar(dq, float(i), emb)
    assert len(dq) == 3

    new_mode = _unit(_basis(dim, 7))
    add_clustered_exemplar(dq, 99.0, new_mode)
    stored = np.stack([e for (_t, e) in dq], axis=0)
    assert len(dq) == 3
    assert float(np.max(stored @ new_mode)) == pytest.approx(1.0, abs=1e-5)


def test_household_gallery_size_capped(tmp_path) -> None:
    mgr = _household_mgr(tmp_path, gallery_size=20)
    assert mgr.gallery_size == 8
    assert mgr.gpu_min_gallery == 8


def test_provisional_track_skips_gallery_add(tmp_path) -> None:
    mgr = _household_mgr(tmp_path)
    dim = 16
    emb = _unit(_basis(dim, 0))
    bbox = (10.0, 10.0, 50.0, 120.0)
    rec = {
        "stable_id": 10001,
        "identity_kind": "provisional",
        "identity_state": "provisional",
        "identity_quality": "weak_no_embedding",
    }
    added = mgr._persist_track_embedding(10001, 1.0, emb, rec=rec, bbox=bbox)
    assert not added
    assert 10001 not in mgr.gallery
