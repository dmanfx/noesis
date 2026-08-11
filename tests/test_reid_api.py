from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from noesis.server import reid_api
from reid.stable_id_manager import StableIDManager


def _unit_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(dim,)).astype(np.float32)
    emb /= float(np.linalg.norm(emb) + 1e-12)
    return emb


def _make_mgr(tmp_path, **kwargs) -> StableIDManager:
    alias_file = kwargs.pop("alias_file", str(tmp_path / "aliases.json"))
    return StableIDManager(
        use_extractor=False,
        new_id_hysteresis_frames=1,
        new_id_confirm_frames_at_cap=1,
        embed_interval_s=0.0,
        sid_pool_file=str(tmp_path / "sid_pool.json"),
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


def test_reid_api_alias_lifecycle(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=1)
    emb_b = _unit_embedding(seed=2)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    reid_api.register_reid_manager_getter(lambda: mgr)
    client = TestClient(reid_api.app)

    resp = client.get("/api/v1/reid/aliases")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["aliases"] == {}

    resp = client.post("/api/v1/reid/aliases/merge", json={"a": sid_a, "b": sid_b, "force": True})
    assert resp.status_code == 200
    merge_body = resp.json()
    assert merge_body["applied"] is True

    resp = client.get("/api/v1/reid/aliases")
    assert resp.status_code == 200
    assert resp.json()["aliases"]

    resp = client.post("/api/v1/reid/aliases/unset", json={"src": merge_body["src"]})
    assert resp.status_code == 200
    assert resp.json()["removed"] is True

    resp = client.post("/api/v1/reid/aliases/clear", json={})
    assert resp.status_code == 200
    assert resp.json()["cleared"] >= 0

    resp = client.get("/api/v1/reid/aliases/history?limit=50")
    assert resp.status_code == 200
    history = resp.json().get("history", [])
    assert any(entry.get("action") == "merge" for entry in history)


def test_reid_api_merge_with_canonical_and_batch(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb_a = _unit_embedding(seed=10)
    emb_b = _unit_embedding(seed=11)
    emb_c = _unit_embedding(seed=12)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb_a)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb_b)
    sid_c = _update_with_embedding(mgr, 0, 3, 1.0, emb_c)
    reid_api.register_reid_manager_getter(lambda: mgr)
    client = TestClient(reid_api.app)

    resp = client.post(
        "/api/v1/reid/aliases/merge",
        json={"a": sid_a, "b": sid_b, "canonical": sid_b, "force": True},
    )
    assert resp.status_code == 200
    assert resp.json()["canonical"] == sid_b

    resp = client.post(
        "/api/v1/reid/aliases/merge-batch",
        json={"pairs": [{"a": sid_a, "b": sid_c}, {"a": sid_c, "b": sid_b}], "force": True},
    )
    assert resp.status_code == 200
    batch_body = resp.json()
    assert batch_body["applied_count"] >= 1


def test_reid_api_merge_batch_conflict_and_suggest(tmp_path) -> None:
    mgr = _make_mgr(tmp_path, min_embeddings_for_suggest=1, cos_sim_threshold=1.1, cos_sim_high_threshold=1.1)
    emb = _unit_embedding(seed=20)
    sid_a = _update_with_embedding(mgr, 0, 1, 1.0, emb)
    sid_b = _update_with_embedding(mgr, 0, 2, 1.0, emb)
    sid_c = _update_with_embedding(mgr, 0, 3, 1.0, emb)
    mgr.remove_missing_tracks(sensor_id=0, present_ds_ids=[], ts=2.0)
    reid_api.register_reid_manager_getter(lambda: mgr)
    client = TestClient(reid_api.app)

    resp = client.post(
        "/api/v1/reid/aliases/merge-batch",
        json={
            "pairs": [
                {"a": sid_a, "b": sid_b, "canonical": sid_a},
                {"a": sid_a, "b": sid_c, "canonical": sid_c},
            ],
            "force": True,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["applied_count"] == 0

    resp = client.post("/api/v1/reid/aliases/suggest", json={"min_sim": 0.0, "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "candidates" in body
    assert "default_min_sim" in body
