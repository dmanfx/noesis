from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

from noesis.pipelines import hooks


@dataclass
class _Rect:
    left: float = 12.0
    top: float = 18.0
    width: float = 140.0
    height: float = 220.0


@dataclass
class _ObjMeta:
    object_id: int = 2
    class_id: int = 0
    confidence: float = 0.9
    rect_params: _Rect = field(default_factory=_Rect)
    _pose_payload: str = ""


@dataclass
class _FrameMeta:
    object_items: List[Any]
    source_id: int = 0
    frame_number: int = 1
    buf_pts: int = 0


class _PoseNativeStub:
    def __init__(self) -> None:
        self.attach_calls = 0

    def extract_pose_keypoints(
        self,
        obj_meta: Any,
        gie_id: int,
        model_w: int,
        model_h: int,
        score_threshold: float,
        letterbox: bool,
    ) -> Dict[str, Any]:
        left = float(getattr(obj_meta.rect_params, "left", 0.0))
        top = float(getattr(obj_meta.rect_params, "top", 0.0))
        keypoints_roi = []
        keypoints_abs = []
        for idx in range(17):
            x = 8.0 + float(idx)
            y = 6.0 + float(idx)
            c = 0.92
            keypoints_roi.append((x, y, c))
            keypoints_abs.append((x + left, y + top, c))
        return {
            "score": 0.9,
            "keypoints_roi": keypoints_roi,
            "keypoints_abs": keypoints_abs,
        }

    def attach_pose_features(
        self,
        _batch_meta: Any,
        obj_meta: Any,
        payload_json: str,
        replace_existing: bool = True,
    ) -> bool:
        self.attach_calls += 1
        obj_meta._pose_payload = str(payload_json)
        return True

    def extract_pose_features(self, obj_meta: Any) -> str:
        return str(getattr(obj_meta, "_pose_payload", ""))


def _pipeline_stub() -> Any:
    return SimpleNamespace(stable_id_mgr=SimpleNamespace(active_tracks={}), frame_size=(1920, 1080), config={})


def test_pose_payload_limit_blocks_oversized_attach(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    native = _PoseNativeStub()
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", native)
    monkeypatch.setenv("NOESIS_POSE_META_MAX_JSON_BYTES", "128")

    proc = hooks.PoseFeatureProcessor(pipeline=_pipeline_stub(), gie_id=4)
    obj = _ObjMeta()
    frame = _FrameMeta(object_items=[obj])

    proc.handle_servicemaker_frame(object(), frame)

    assert native.attach_calls == 0
    assert obj._pose_payload == ""


def test_pose_payload_roundtrip_allows_valid_payload(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    native = _PoseNativeStub()
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", native)
    monkeypatch.setenv("NOESIS_POSE_META_MAX_JSON_BYTES", "65536")

    pipeline = _pipeline_stub()
    proc = hooks.PoseFeatureProcessor(pipeline=pipeline, gie_id=4)
    overlay_proc = hooks.PoseKeypointOverlayProcessor(pipeline=pipeline, gie_id=4)

    obj = _ObjMeta()
    frame = _FrameMeta(object_items=[obj])

    proc.handle_servicemaker_frame(object(), frame)

    assert native.attach_calls == 1
    assert obj._pose_payload

    payload = overlay_proc._extract_pose_payload(obj)
    assert isinstance(payload, dict)
    assert "keypoints_roi" in payload
    assert "keypoints_abs" in payload

    snap = hooks.get_core_path_instrumentation_snapshot()
    counters = snap.get("counters", {})
    assert int(counters.get("tensor_boundary_copy_bytes_total.pose_meta", 0)) > 0
