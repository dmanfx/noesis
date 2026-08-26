from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from noesis.pipelines import hooks


@dataclass
class _Rect:
    left: float = 10.0
    top: float = 20.0
    width: float = 80.0
    height: float = 160.0


@dataclass
class _ObjMeta:
    object_id: int = 7
    class_id: int = 0
    confidence: float = 0.95
    rect_params: _Rect = field(default_factory=_Rect)


@dataclass
class _FrameMeta:
    object_items: List[Any]
    source_id: int = 0
    frame_number: int = 1
    buf_pts: int = 0


class _TrackingPub:
    def __init__(self) -> None:
        self.calls: List[tuple[int, list[dict[str, Any]]]] = []

    def publish(
        self,
        sensor_id: int,
        tracks: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> None:
        self.calls.append((int(sensor_id), list(tracks)))


class _StableIDMgr:
    def __init__(self, *, need_embedding: bool) -> None:
        self.need_embedding = bool(need_embedding)
        self.active_tracks: Dict[tuple[int, int], Dict[str, Any]] = {}
        self.embeddings_seen: List[Optional[np.ndarray]] = []

    def needs_embedding(self, sensor_id: int, ds_obj_id: int, ts: float) -> bool:
        return bool(self.need_embedding)

    def update(
        self,
        *,
        sensor_id: int,
        ds_obj_id: int,
        bbox_ltrbwh: tuple[float, float, float, float],
        ts: float,
        zone: Optional[str],
        frame_bgr: Optional[np.ndarray],
        embedding: Optional[np.ndarray],
        pose_features: Optional[np.ndarray] = None,
        pose_quality: float | None = None,
        world_xy: tuple[float, float] | None = None,
        world_valid: bool = False,
    ) -> int:
        self.embeddings_seen.append(embedding)
        sid = 101
        self.active_tracks[(int(sensor_id), int(ds_obj_id))] = {"stable_id": sid, "emb": embedding}
        return sid

    def remove_missing_tracks(self, sensor_id: int, present_track_ids: list[int], ts: float) -> None:
        return None

    def prune_ghosts(self, ts: float) -> None:
        return None

    def observe_copresence(self, stable_ids: list[int], ts: float) -> None:
        return None



def _make_processor(stable_mgr: _StableIDMgr) -> tuple[hooks._AnalyticsTelemetryProcessor, _TrackingPub]:  # type: ignore[attr-defined]
    tracking_pub = _TrackingPub()
    pipeline = SimpleNamespace(
        frame_size=(1920, 1080),
        stable_id_mgr=stable_mgr,
        config={},
    )
    proc = hooks._AnalyticsTelemetryProcessor(  # type: ignore[attr-defined]
        pipeline=pipeline,
        tracking_pub=tracking_pub,
        camera_labels={0: "cam0"},
        sensor_id_map={0: 0},
    )
    return proc, tracking_pub


def test_reid_embedding_extraction_requires_needs_embedding_true(monkeypatch) -> None:
    monkeypatch.setattr(hooks, "noesis_analytics_meta_ext", None)
    obj = _ObjMeta()
    frame = _FrameMeta(object_items=[obj])

    mgr_no = _StableIDMgr(need_embedding=False)
    proc_no, tracking_no = _make_processor(mgr_no)
    calls_no = {"extract": 0}

    def _extract_no(_obj_meta: Any) -> np.ndarray:
        calls_no["extract"] += 1
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return emb

    monkeypatch.setattr(proc_no, "_extract_reid_embedding_servicemaker", _extract_no)
    proc_no.handle_servicemaker_frame(frame)

    assert calls_no["extract"] == 0
    assert tracking_no.calls
    assert len(mgr_no.embeddings_seen) == 1
    assert mgr_no.embeddings_seen[0] is None

    mgr_yes = _StableIDMgr(need_embedding=True)
    proc_yes, tracking_yes = _make_processor(mgr_yes)
    calls_yes = {"extract": 0}

    def _extract_yes(_obj_meta: Any) -> np.ndarray:
        calls_yes["extract"] += 1
        return np.array([0.2, 0.3, 0.4], dtype=np.float32)

    monkeypatch.setattr(proc_yes, "_extract_reid_embedding_servicemaker", _extract_yes)
    proc_yes.handle_servicemaker_frame(frame)

    assert calls_yes["extract"] == 1
    assert tracking_yes.calls
    assert len(mgr_yes.embeddings_seen) == 1
    assert isinstance(mgr_yes.embeddings_seen[0], np.ndarray)


def test_reid_embedding_contract_is_finite_float32_and_normalized(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    proc, _ = _make_processor(_StableIDMgr(need_embedding=True))
    obj = _ObjMeta()
    calls: list[tuple[int, str, int, bool]] = []

    class _ReidNative:
        @staticmethod
        def extract_reid_embedding(
            obj_meta: Any,
            gie_id: int,
            layer_name: str,
            dim: int,
            normalize: bool,
        ) -> list[float]:
            calls.append((gie_id, layer_name, dim, normalize))
            return [3.0, 4.0, 0.0]

    monkeypatch.setattr(hooks, "noesis_reid_meta_ext", _ReidNative())
    emb = proc._extract_reid_embedding_servicemaker(obj)

    assert emb is not None
    assert calls == [(3, "fc_pred", 256, True)]
    assert emb.dtype == np.float32
    assert emb.ndim == 1
    assert bool(np.all(np.isfinite(emb)))
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3

    snap = hooks.get_core_path_instrumentation_snapshot()
    counters = snap.get("counters", {})
    assert int(counters.get("tensor_host_copies_total.reid", 0)) >= 1
