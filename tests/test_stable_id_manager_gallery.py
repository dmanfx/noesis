"""Long-term ReID memory tests for StableIDManager.

Covers:
- Multi-exemplar similarity (recall from any stored appearance, not just the
  recency-weighted centroid).
- Diversity-preserving gallery retention (full gallery replaces the most
  similar exemplar instead of evicting the oldest).
- Persistent gallery save/load roundtrip (memory across restarts).
"""

from __future__ import annotations

import numpy as np
import pytest

from reid.stable_id_manager import StableIDManager


def _unit(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr / (np.linalg.norm(arr) + 1e-12)


def _make_manager(tmp_path, **overrides) -> StableIDManager:
    kwargs = dict(
        use_extractor=False,
        compute_backend="cpu",
        embed_interval_s=0.0,
        gallery_size=4,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
        gallery_persist_file=str(tmp_path / "gallery.npz"),
        gallery_autosave_interval_s=0.0,
        aliases_enabled=False,
    )
    kwargs.update(overrides)
    return StableIDManager(**kwargs)


def _basis_vec(dim: int, idx: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def test_multi_exemplar_similarity_recalls_old_appearance(tmp_path):
    mgr = _make_manager(tmp_path)
    dim = 16
    old_appearance = _basis_vec(dim, 0)
    new_appearance = _basis_vec(dim, 1)

    sid = 7
    mgr.gallery[sid].append((100.0, _unit(old_appearance)))
    mgr.gallery[sid].append((200.0, _unit(new_appearance)))
    # Centroid dominated by the new appearance.
    mgr.sid_centroid[sid] = _unit(new_appearance)

    sims = mgr._similarity_for_candidates(_unit(old_appearance), [sid])
    # Exemplar-max matching must recall the old appearance (cos ~1.0), even
    # though the centroid similarity is ~0.
    assert sims[sid] == pytest.approx(1.0, abs=1e-5)


def test_gallery_add_preserves_diverse_exemplars(tmp_path):
    mgr = _make_manager(tmp_path, gallery_size=3)
    dim = 8
    sid = 3
    distinct = [_unit(_basis_vec(dim, i)) for i in range(3)]
    for i, emb in enumerate(distinct):
        mgr._gallery_add(sid, float(i), emb)
    assert len(mgr.gallery[sid]) == 3

    # A near-duplicate of exemplar 2 should replace exemplar 2, not exemplar 0.
    near_dup = _unit(_basis_vec(dim, 2) + 0.05 * _basis_vec(dim, 3))
    mgr._gallery_add(sid, 99.0, near_dup)
    entries = list(mgr.gallery[sid])
    assert len(entries) == 3
    stored = np.stack([e for (_t, e) in entries], axis=0)
    # Exemplars 0 and 1 survive.
    assert float(np.max(stored @ distinct[0])) == pytest.approx(1.0, abs=1e-5)
    assert float(np.max(stored @ distinct[1])) == pytest.approx(1.0, abs=1e-5)
    # The near-duplicate replaced its closest neighbor.
    assert any(t == 99.0 for (t, _e) in entries)


def test_gallery_persistence_roundtrip(tmp_path):
    import time as _time

    now = _time.time()
    mgr = _make_manager(tmp_path)
    dim = 16
    emb_a = _unit(_basis_vec(dim, 0))
    emb_b = _unit(_basis_vec(dim, 5))
    mgr.gallery[2].append((now - 10.0, emb_a))
    mgr.gallery[2].append((now - 5.0, emb_b))
    mgr.sid_centroid[2] = _unit(emb_a + emb_b)
    mgr.sid_global_first_seen[2] = now - 10.0
    mgr.sid_global_last_seen[2] = now - 5.0
    assert mgr.save_gallery()

    mgr2 = _make_manager(tmp_path)
    assert 2 in mgr2.gallery
    assert len(mgr2.gallery[2]) == 2
    assert mgr2.next_stable_id >= 3
    sims = mgr2._similarity_for_candidates(emb_a, [2])
    assert sims[2] == pytest.approx(1.0, abs=1e-5)
    # Loaded SID must not be re-allocatable.
    assert 2 not in mgr2._free_sids_set


def test_gallery_persistence_drops_stale_identities(tmp_path):
    import time as _time

    now = _time.time()
    mgr = _make_manager(tmp_path, gallery_persist_max_age_s=60.0)
    dim = 8
    mgr.gallery[4].append((now - 3600.0, _unit(_basis_vec(dim, 0))))
    mgr.sid_global_last_seen[4] = now - 3600.0
    mgr.gallery[5].append((now - 1.0, _unit(_basis_vec(dim, 1))))
    mgr.sid_global_last_seen[5] = now - 1.0
    assert mgr.save_gallery()

    mgr2 = _make_manager(tmp_path, gallery_persist_max_age_s=60.0)
    assert 4 not in mgr2.gallery
    assert 5 in mgr2.gallery


def test_cross_camera_handoff_ignores_scale_penalty(tmp_path):
    """A person walking to a camera with a very different bbox scale must keep
    their stable ID; scale penalties only apply within the same camera."""
    mgr = _make_manager(tmp_path, cos_sim_threshold=0.62, cos_sim_high_threshold=0.72)
    dim = 32
    person = _unit(np.linspace(0.1, 1.0, dim))

    t0 = 2000.0
    big_bbox = (200.0, 100.0, 300.0, 800.0)  # close to camera 0
    sid_first = mgr.update(0, 21, big_bbox, t0, None, embedding=person)
    mgr.update(0, 21, big_bbox, t0 + 1.0, None, embedding=person)

    # Appears on camera 1 seconds later, 16x smaller bbox area (far away).
    small_bbox = (50.0, 50.0, 75.0, 200.0)
    sid_xcam = mgr.update(1, 33, small_bbox, t0 + 3.0, None, embedding=person)
    assert sid_xcam == sid_first


def test_same_camera_scale_penalty_still_applies(tmp_path):
    mgr = _make_manager(tmp_path)
    dim = 16
    emb = _unit(_basis_vec(dim, 0))
    sid = 9
    mgr.gallery[sid].append((10.0, emb))
    mgr.sid_centroid[sid] = emb
    mgr.sid_last_bbox[sid] = (0.0, 0.0, 100.0, 300.0)
    mgr.sid_last_bbox_sensor[sid] = 0

    # Same sensor: scale change (4x area) is penalized below raw cosine.
    best_same, reid_same, _ = mgr._gallery_best(
        emb, sensor_id=0, curr_bbox=(0.0, 0.0, 50.0, 150.0), min_reid=0.0
    )
    # Different sensor: no scale penalty.
    best_xcam, reid_xcam, _ = mgr._gallery_best(
        emb, sensor_id=1, curr_bbox=(0.0, 0.0, 50.0, 150.0), min_reid=0.0
    )
    assert best_same == sid and best_xcam == sid
    assert reid_xcam > reid_same
    assert reid_xcam == pytest.approx(1.0, abs=1e-5)


def test_update_assigns_same_sid_after_long_absence(tmp_path):
    """A person leaving and returning much later must get the same stable ID."""
    mgr = _make_manager(tmp_path, cos_sim_threshold=0.62, cos_sim_high_threshold=0.72)
    dim = 32
    person = _unit(np.linspace(0.1, 1.0, dim))
    bbox = (100.0, 100.0, 60.0, 160.0)

    t0 = 1000.0
    sid_first = mgr.update(0, 11, bbox, t0, None, embedding=person)
    mgr.update(0, 11, bbox, t0 + 1.0, None, embedding=person)
    # Track disappears.
    mgr.remove_missing_tracks(0, [], t0 + 2.0)
    # Ghosts expire; only the gallery (long-term memory) remains.
    mgr.prune_ghosts(now_ts=t0 + 100000.0)

    t1 = t0 + 100000.0
    sid_back = mgr.update(1, 99, bbox, t1, None, embedding=person)
    assert sid_back == sid_first
