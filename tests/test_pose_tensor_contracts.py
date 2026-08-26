from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from noesis.pipelines import hooks


@dataclass
class _Rect:
    left: float = 30.0
    top: float = 40.0
    width: float = 120.0
    height: float = 240.0


@dataclass
class _ObjMeta:
    object_id: int = 5
    class_id: int = 0
    confidence: float = 0.95
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
        self.extract_calls = 0

    def _kpts(self, left: float, top: float) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        roi: list[tuple[float, float, float]] = []
        abs_k: list[tuple[float, float, float]] = []
        for idx in range(17):
            x = 5.0 + float(idx)
            y = 7.0 + float(idx)
            c = 0.9
            roi.append((x, y, c))
            abs_k.append((x + left, y + top, c))
        return roi, abs_k

    def extract_pose_keypoints(
        self,
        obj_meta: Any,
        gie_id: int,
        model_w: int,
        model_h: int,
        score_threshold: float,
        letterbox: bool,
    ) -> Dict[str, Any]:
        self.extract_calls += 1
        left = float(getattr(obj_meta.rect_params, "left", 0.0))
        top = float(getattr(obj_meta.rect_params, "top", 0.0))
        roi, abs_k = self._kpts(left, top)
        return {
            "score": 0.93,
            "keypoints_roi": roi,
            "keypoints_abs": abs_k,
        }

    def attach_pose_features(
        self,
        _batch_meta: Any,
        obj_meta: Any,
        payload_json: str,
        replace_existing: bool = True,
    ) -> bool:
        obj_meta._pose_payload = str(payload_json)
        return True

    def extract_pose_features(self, obj_meta: Any) -> str:
        return str(getattr(obj_meta, "_pose_payload", ""))


def _build_pipeline_stub() -> Any:
    return SimpleNamespace(stable_id_mgr=SimpleNamespace(active_tracks={}), frame_size=(1920, 1080), config={})


def test_pose_native_extract_returns_expected_shape_and_counter(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    native = _PoseNativeStub()
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", native)

    proc = hooks.PoseFeatureProcessor(pipeline=_build_pipeline_stub(), gie_id=4)
    obj = _ObjMeta()

    extracted = proc._extract_pose_native(obj)
    assert extracted is not None
    score, roi, abs_k = extracted
    assert score > 0.0
    assert roi.shape == (17, 3)
    assert abs_k.shape == (17, 3)
    assert roi.dtype == np.float32
    assert abs_k.dtype == np.float32

    snap = hooks.get_core_path_instrumentation_snapshot()
    counters = snap.get("counters", {})
    assert int(counters.get("tensor_host_copies_total.pose", 0)) >= 1


def test_pose_overlay_reuses_attached_payload_without_second_tensor_extract(monkeypatch) -> None:
    native = _PoseNativeStub()
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", native)

    pipeline = _build_pipeline_stub()
    feature_proc = hooks.PoseFeatureProcessor(pipeline=pipeline, gie_id=4)
    overlay_proc = hooks.PoseKeypointOverlayProcessor(pipeline=pipeline, gie_id=4)

    obj = _ObjMeta()
    frame = _FrameMeta(object_items=[obj])

    feature_proc.handle_servicemaker_frame(object(), frame)
    payload = overlay_proc._extract_pose_payload(obj)

    assert payload is not None
    assert "keypoints_roi" in payload
    assert "keypoints_abs" in payload

    bbox = (obj.rect_params.left, obj.rect_params.top, obj.rect_params.width, obj.rect_params.height)
    kpts = overlay_proc._keypoints_from_payload(payload, bbox)
    assert kpts is not None
    assert kpts.shape == (17, 3)

    # Only the feature processor should decode tensor meta; overlay reads attached payload.
    assert native.extract_calls == 1


def test_pose_feature_processor_does_not_attach_when_sgie_skips(monkeypatch) -> None:
    class _SkippingPoseNativeStub(_PoseNativeStub):
        def extract_pose_keypoints(
            self,
            obj_meta: Any,
            gie_id: int,
            model_w: int,
            model_h: int,
            score_threshold: float,
            letterbox: bool,
        ) -> Dict[str, Any] | None:
            if self.extract_calls >= 1:
                self.extract_calls += 1
                return None
            return super().extract_pose_keypoints(
                obj_meta,
                gie_id,
                model_w,
                model_h,
                score_threshold,
                letterbox,
            )

    native = _SkippingPoseNativeStub()
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", native)

    proc = hooks.PoseFeatureProcessor(pipeline=_build_pipeline_stub(), gie_id=4)
    obj_first = _ObjMeta()
    frame_first = _FrameMeta(object_items=[obj_first], frame_number=1)

    proc.handle_servicemaker_frame(object(), frame_first)

    obj_second = _ObjMeta(rect_params=_Rect(left=36.0, top=46.0, width=126.0, height=252.0))
    frame_second = _FrameMeta(object_items=[obj_second], frame_number=2)
    proc.handle_servicemaker_frame(object(), frame_second)

    assert obj_first._pose_payload
    assert obj_second._pose_payload == ""
    assert native.extract_calls == 2
