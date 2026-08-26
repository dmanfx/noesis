from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List

import numpy as np

from noesis.pipelines import hooks


@dataclass
class _FrameMeta:
    source_id: int = 0
    frame_number: int = 11
    frame_width: int = 1920
    frame_height: int = 1080
    buf_pts: int = 123456789000
    object_items: List[Any] = field(default_factory=list)


@dataclass
class _Rect:
    left: float = 1.0
    top: float = 1.0
    width: float = 2.0
    height: float = 2.0


@dataclass
class _ObjMeta:
    object_id: int = 7
    class_id: int = 0
    confidence: float = 0.9
    rect_params: _Rect = field(default_factory=_Rect)


class _DeviceDepthFrame:
    def __init__(self, *, value: float, frame_w: int, frame_h: int, depth_w: int = 518, depth_h: int = 294) -> None:
        self._value = float(value)
        self.frame_width = int(frame_w)
        self.frame_height = int(frame_h)
        self.depth_width = int(depth_w)
        self.depth_height = int(depth_h)
        self.roi_calls: list[tuple[int, int, int, int]] = []

    def copy_roi_to_numpy(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        self.roi_calls.append((int(left), int(top), int(width), int(height)))
        return np.full((int(height), int(width)), self._value, dtype=np.float32)


class _NativeStatsDepthFrame(_DeviceDepthFrame):
    def __init__(self, *, value: float, frame_w: int, frame_h: int) -> None:
        super().__init__(value=value, frame_w=frame_w, frame_h=frame_h)
        self.stats_calls: list[tuple[int, int, int, int, int]] = []
        self.mask_stats_calls: list[tuple[int, int, int, int, int]] = []
        self.mask_dtypes: list[np.dtype] = []

    def sample_roi_stats(self, left: int, top: int, width: int, height: int, max_samples: int = 4096) -> dict[str, float | int]:
        self.stats_calls.append((int(left), int(top), int(width), int(height), int(max_samples)))
        area = int(width) * int(height)
        return {
            "roi_area_px": area,
            "sampled_area_px": area,
            "sample_count": area,
            "valid_fraction": 1.0,
            "depth_center": self._value,
            "depth_median": self._value,
            "depth_mean": self._value,
            "depth_p10": self._value,
            "depth_p90": self._value,
            "depth_min": self._value,
            "depth_max": self._value,
        }

    def sample_masked_roi_stats(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        mask: np.ndarray,
        threshold: float = 0.5,
        max_samples: int = 4096,
    ) -> dict[str, float | int]:
        self.mask_stats_calls.append((int(left), int(top), int(width), int(height), int(max_samples)))
        self.mask_dtypes.append(np.asarray(mask).dtype)
        mask_area = int(np.count_nonzero(np.asarray(mask, dtype=np.float32) > float(threshold)))
        return {
            "roi_area_px": int(width) * int(height),
            "mask_area_px": mask_area,
            "sampled_area_px": int(width) * int(height),
            "sampled_mask_area_px": mask_area,
            "sample_count": mask_area,
            "valid_fraction": 1.0 if mask_area > 0 else 0.0,
            "depth_center": self._value,
            "depth_median": self._value if mask_area > 0 else None,
            "depth_mean": self._value if mask_area > 0 else None,
            "depth_p10": self._value if mask_area > 0 else None,
            "depth_p90": self._value if mask_area > 0 else None,
            "depth_min": self._value if mask_area > 0 else None,
            "depth_max": self._value if mask_area > 0 else None,
        }


class _NativePersonStatsDepthFrame(_NativeStatsDepthFrame):
    def __init__(self, *, value: float, frame_w: int, frame_h: int) -> None:
        super().__init__(value=value, frame_w=frame_w, frame_h=frame_h)
        self.person_stats_calls: list[tuple[int, int, int, int, int]] = []
        self.person_mask_dtypes: list[np.dtype] = []

    def sample_masked_person_roi_stats(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        mask: np.ndarray,
        threshold: float = 0.5,
        max_samples: int = 4096,
    ) -> dict[str, float | int]:
        self.person_stats_calls.append((int(left), int(top), int(width), int(height), int(max_samples)))
        self.person_mask_dtypes.append(np.asarray(mask).dtype)
        mask_area = int(np.count_nonzero(np.asarray(mask, dtype=np.float32) > float(threshold)))
        return {
            "roi_area_px": int(width) * int(height),
            "mask_area_px": mask_area,
            "lower_mask_area_px": mask_area,
            "torso_mask_area_px": 0,
            "sampled_area_px": int(width) * int(height),
            "sampled_mask_area_px": mask_area,
            "sampled_lower_mask_area_px": mask_area,
            "sampled_torso_mask_area_px": 0,
            "foot_u": float(left) + (float(width) * 0.5),
            "foot_v": float(top) + float(height) - 1.0,
            "sample_count": mask_area,
            "valid_fraction": 1.0 if mask_area > 0 else 0.0,
            "depth_center": self._value,
            "depth_median": self._value if mask_area > 0 else None,
            "depth_mean": self._value if mask_area > 0 else None,
            "depth_p10": self._value if mask_area > 0 else None,
            "depth_p90": self._value if mask_area > 0 else None,
            "depth_min": self._value if mask_area > 0 else None,
            "depth_max": self._value if mask_area > 0 else None,
            "lower_sample_count": mask_area,
            "lower_valid_fraction": 1.0 if mask_area > 0 else 0.0,
            "lower_depth_median": self._value if mask_area > 0 else None,
            "lower_depth_mean": self._value if mask_area > 0 else None,
            "lower_depth_p10": self._value if mask_area > 0 else None,
            "lower_depth_p90": self._value if mask_area > 0 else None,
            "lower_depth_min": self._value if mask_area > 0 else None,
            "lower_depth_max": self._value if mask_area > 0 else None,
            "torso_sample_count": 0,
            "torso_valid_fraction": 0.0,
            "torso_depth_median": None,
            "torso_depth_mean": None,
            "torso_depth_p10": None,
            "torso_depth_p90": None,
            "torso_depth_min": None,
            "torso_depth_max": None,
        }


def _depth_processor() -> hooks._DepthTrackingFrameProcessor:  # type: ignore[attr-defined]
    return hooks._DepthTrackingFrameProcessor(  # type: ignore[attr-defined]
        depth_store=hooks._AlignedDepthFrameStore(),  # type: ignore[attr-defined]
        depth_gie_id=5,
        fallback_frame_size=(1920, 1080),
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
    )


def _fusion_processor(store: hooks._AlignedDepthFrameStore) -> hooks._ObjectDepthFusionProcessor:  # type: ignore[attr-defined]
    return hooks._ObjectDepthFusionProcessor(  # type: ignore[attr-defined]
        depth_store=store,
        fallback_frame_size=(1920, 1080),
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        depth_every_n_frames=2,
        calibration_resolver=None,
        camera_labels={0: "cam0"},
    )


def _aligned_frame(
    *,
    source_id: int,
    frame_id: int,
    pts_us: int,
    device_frame: Any | None = None,
) -> hooks._AlignedDepthFrame:  # type: ignore[attr-defined]
    return hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
        key=(int(source_id), int(frame_id), int(pts_us)),
        source_id=int(source_id),
        frame_id=int(frame_id),
        pts_us=int(pts_us),
        depth_map=None,
        valid_mask=None,
        frame_w=1920,
        frame_h=1080,
        depth_w=518,
        depth_h=294,
        unit="m",
        is_metric=True,
        model_name="depth-anything-v2-metric-hypersim-vits",
        depth_device_frame=device_frame,
    )


def test_aligned_depth_store_bypasses_wait_before_lagged_fallback() -> None:
    hooks.reset_core_path_instrumentation()
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    lagged = _aligned_frame(source_id=0, frame_id=10, pts_us=100_000)
    exact = _aligned_frame(source_id=0, frame_id=11, pts_us=133_333)
    store.put(lagged)

    started = threading.Event()
    resolved: list[tuple[Any, int, float]] = []

    def _resolve() -> None:
        started.set()
        resolved.append(
            store.resolve(
                source_id=0,
                frame_id=11,
                pts_us=133_333,
                max_age_frames=1,
                wait_ms=150.0,
            )
        )

    thread = threading.Thread(target=_resolve, daemon=True)
    thread.start()
    assert started.wait(timeout=1.0)
    time.sleep(0.02)
    store.put(exact)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert resolved == [(lagged, 1, 33.333)]
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["depth_bridge_put_total"] == 2
    assert counters["depth_bridge_wait_bypassed_total"] == 1
    assert counters.get("depth_bridge_exact_resolve_total", 0) == 0
    assert counters["depth_bridge_lagged_resolve_total"] == 1


def test_aligned_depth_store_bypasses_wait_then_uses_bounded_prior_frame() -> None:
    hooks.reset_core_path_instrumentation()
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    lagged = _aligned_frame(source_id=0, frame_id=10, pts_us=100_000)
    store.put(lagged)

    resolved, age_frames, age_ms = store.resolve(
        source_id=0,
        frame_id=11,
        pts_us=133_333,
        max_age_frames=1,
        wait_ms=5.0,
    )

    assert resolved is lagged
    assert age_frames == 1
    assert age_ms == 33.333
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["depth_bridge_wait_bypassed_total"] == 1
    assert counters.get("depth_bridge_wait_total", 0) == 0
    assert counters.get("depth_bridge_wait_timeout_total", 0) == 0
    assert counters["depth_bridge_lagged_resolve_total"] == 1
    assert counters["depth_bridge_lagged_age_frames_total"] == 1
    assert counters["depth_bridge_lagged_age_us_total"] == 33_333


def test_aligned_depth_store_rejects_wrong_source_pts_and_age() -> None:
    hooks.reset_core_path_instrumentation()
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    store.put(_aligned_frame(source_id=1, frame_id=11, pts_us=133_333))
    store.put(_aligned_frame(source_id=0, frame_id=11, pts_us=100_000))
    store.put(_aligned_frame(source_id=0, frame_id=10, pts_us=150_000))
    store.put(_aligned_frame(source_id=0, frame_id=9, pts_us=90_000))

    resolved, age_frames, age_ms = store.resolve(
        source_id=0,
        frame_id=11,
        pts_us=133_333,
        max_age_frames=1,
    )

    assert resolved is None
    assert age_frames == 0
    assert age_ms == 0.0
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["depth_bridge_miss_total"] == 1
    assert counters.get("depth_bridge_lagged_resolve_total", 0) == 0


def test_object_depth_rendezvous_wait_defaults_to_nonblocking_and_is_bounded(monkeypatch) -> None:
    processor = _fusion_processor(hooks._AlignedDepthFrameStore())  # type: ignore[attr-defined]

    monkeypatch.setenv("NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS", "inf")
    assert processor._exact_frame_wait_ms() == 0.0  # type: ignore[attr-defined]
    monkeypatch.setenv("NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS", "999999")
    assert processor._exact_frame_wait_ms() == 250.0  # type: ignore[attr-defined]
    monkeypatch.setenv("NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS", "-5")
    assert processor._exact_frame_wait_ms() == 0.0  # type: ignore[attr-defined]


def test_depth_tracking_captures_device_aligned_frame(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    processor = _depth_processor()
    frame_meta = _FrameMeta()
    device_frame = _DeviceDepthFrame(value=3.5, frame_w=1920, frame_h=1080, depth_w=924, depth_h=518)

    class _NativeExt:
        @staticmethod
        def capture_aligned_depth_frame(frame: object, gie_id: int, frame_w: int, frame_h: int) -> _DeviceDepthFrame:
            assert frame is frame_meta
            assert gie_id == 5
            assert frame_w == 1920
            assert frame_h == 1080
            return device_frame

    monkeypatch.setattr(hooks, "noesis_depth_tracking_tensor_ext", _NativeExt)

    processor.handle_servicemaker_frame(frame_meta)  # type: ignore[attr-defined]

    stored, age_frames, age_ms = processor.depth_store.resolve(  # type: ignore[attr-defined]
        source_id=0,
        frame_id=11,
        pts_us=123456789,
        max_age_frames=0,
    )
    assert age_frames == 0
    assert age_ms == 0.0
    assert stored is not None
    assert stored.depth_device_frame is device_frame
    assert stored.depth_map is None
    assert stored.depth_w == 924
    assert stored.depth_h == 518
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["depth_tracking_device_frames_total"] == 1


def test_depth_tracking_returns_none_when_native_ext_has_no_tensor(monkeypatch) -> None:
    processor = _depth_processor()
    frame_meta = _FrameMeta()

    class _NativeExt:
        @staticmethod
        def capture_aligned_depth_frame(frame: object, gie_id: int, frame_w: int, frame_h: int) -> None:
            assert frame is frame_meta
            assert gie_id == 5
            assert frame_w == 1920
            assert frame_h == 1080
            return None

    monkeypatch.setattr(hooks, "noesis_depth_tracking_tensor_ext", _NativeExt)

    processor.handle_servicemaker_frame(frame_meta)  # type: ignore[attr-defined]

    stored, _, _ = processor.depth_store.resolve(  # type: ignore[attr-defined]
        source_id=0,
        frame_id=11,
        pts_us=123456789,
        max_age_frames=0,
    )
    assert stored is None


def test_object_depth_fusion_copies_only_object_roi_from_device_frame(monkeypatch) -> None:
    monkeypatch.setenv("NOESIS_OBJECT_DEPTH_ALLOW_HOST_ROI_COPY", "1")
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    device_frame = _DeviceDepthFrame(value=4.25, frame_w=1920, frame_h=1080)
    store.put(
        hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device_frame,
        )
    )
    processor = _fusion_processor(store)
    attached: list[str] = []

    def _attach(
        _batch_meta: Any,
        _obj_meta: Any,
        payload_json: str,
        _replace_existing: bool = True,
    ) -> bool:
        attached.append(payload_json)
        return True

    native_ext = SimpleNamespace(
        extract_object_mask=lambda _obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
        attach_object_depth=_attach,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", native_ext)

    frame_meta = _FrameMeta(object_items=[_ObjMeta(rect_params=_Rect(left=10.0, top=20.0, width=30.0, height=40.0))])
    processor.handle_servicemaker_frame(object(), frame_meta)  # type: ignore[attr-defined]

    assert device_frame.roi_calls == [(10, 20, 30, 40)]
    assert len(attached) == 1
    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["sample_count"] == 1200
    assert payload["depth_center"] == 4.25
    assert payload["depth_median"] == 4.25


def test_object_depth_fusion_uses_bounded_host_roi_when_native_stats_are_unavailable(monkeypatch) -> None:
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    device_frame = _DeviceDepthFrame(value=4.25, frame_w=1920, frame_h=1080)
    store.put(
        hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device_frame,
        )
    )
    processor = _fusion_processor(store)
    attached: list[str] = []

    def _attach(
        _batch_meta: Any,
        _obj_meta: Any,
        payload_json: str,
        _replace_existing: bool = True,
    ) -> bool:
        attached.append(payload_json)
        return True

    native_ext = SimpleNamespace(
        extract_object_mask=lambda _obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
        attach_object_depth=_attach,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", native_ext)
    hooks.reset_core_path_instrumentation()

    frame_meta = _FrameMeta(object_items=[_ObjMeta(rect_params=_Rect(left=10.0, top=20.0, width=30.0, height=40.0))])
    processor.handle_servicemaker_frame(object(), frame_meta)  # type: ignore[attr-defined]

    assert device_frame.roi_calls == [(10, 20, 30, 40)]
    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["sampling_mode"] == "instance_mask"
    counters = hooks.get_core_path_instrumentation_snapshot().get("counters", {})
    assert int(counters.get("object_depth_gpu_roi_copies_total", 0)) == 1


def test_object_depth_fusion_uses_native_mask_stats_when_available(monkeypatch) -> None:
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    device_frame = _NativeStatsDepthFrame(value=4.25, frame_w=1920, frame_h=1080)
    store.put(
        hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device_frame,
        )
    )
    processor = _fusion_processor(store)
    attached: list[str] = []

    def _attach(
        _batch_meta: Any,
        _obj_meta: Any,
        payload_json: str,
        _replace_existing: bool = True,
    ) -> bool:
        attached.append(payload_json)
        return True

    native_ext = SimpleNamespace(
        extract_object_mask=lambda _obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
        attach_object_depth=_attach,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", native_ext)
    hooks.reset_core_path_instrumentation()

    frame_meta = _FrameMeta(object_items=[_ObjMeta(rect_params=_Rect(left=10.0, top=20.0, width=30.0, height=40.0))])
    processor.handle_servicemaker_frame(object(), frame_meta)  # type: ignore[attr-defined]

    assert device_frame.roi_calls == []
    assert device_frame.mask_stats_calls[0] == (10, 20, 30, 40, 4096)
    assert device_frame.mask_dtypes
    assert all(dtype == np.dtype("float32") for dtype in device_frame.mask_dtypes)
    assert len(attached) == 1
    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["sampling_mode"] == "instance_mask"
    assert payload["sample_count"] == 1200
    assert payload["depth_center"] == 4.25
    assert payload["depth_median"] == 4.25
    assert payload["anchor_source"] == "lower_body_band"
    counters = hooks.get_core_path_instrumentation_snapshot().get("counters", {})
    assert int(counters.get("detection_wake.object_depth_native_mask_stats", 0)) == 1
    assert int(counters.get("object_depth_gpu_roi_copies_total", 0)) == 1
    assert int(counters.get("object_depth_attach_total", 0)) == 1
    assert int(counters.get("object_depth_status_total.ok", 0)) == 1


def test_object_depth_attach_false_is_observable(monkeypatch, caplog) -> None:
    processor = _fusion_processor(hooks._AlignedDepthFrameStore())  # type: ignore[attr-defined]
    monkeypatch.setattr(
        hooks,
        "noesis_depth_meta_ext",
        SimpleNamespace(attach_object_depth=lambda *_args, **_kwargs: False),
    )
    hooks.reset_core_path_instrumentation()

    with caplog.at_level("WARNING"):
        attached = processor._attach_object_depth_payload(  # type: ignore[attr-defined]
            object(),
            _ObjMeta(),
            {"status": "ok"},
        )

    assert attached is False
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters.get("object_depth_attach_total", 0) == 0
    assert counters["object_depth_attach_failure_total"] == 1
    assert counters["object_depth_attach_failure_total.native_rejected"] == 1
    assert counters.get("object_depth_status_total.ok", 0) == 0
    assert "attachment failed (reason=native_rejected" in caplog.text


def test_object_depth_fusion_prefers_combined_native_person_mask_stats(monkeypatch) -> None:
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    device_frame = _NativePersonStatsDepthFrame(value=4.25, frame_w=1920, frame_h=1080)
    store.put(
        hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device_frame,
        )
    )
    processor = _fusion_processor(store)
    attached: list[str] = []

    def _attach(
        _batch_meta: Any,
        _obj_meta: Any,
        payload_json: str,
        _replace_existing: bool = True,
    ) -> bool:
        attached.append(payload_json)
        return True

    native_ext = SimpleNamespace(
        extract_object_mask=lambda _obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
        attach_object_depth=_attach,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", native_ext)

    frame_meta = _FrameMeta(object_items=[_ObjMeta(rect_params=_Rect(left=10.0, top=20.0, width=30.0, height=40.0))])
    processor.handle_servicemaker_frame(object(), frame_meta)  # type: ignore[attr-defined]

    assert device_frame.roi_calls == []
    assert device_frame.mask_stats_calls == []
    assert device_frame.person_stats_calls == [(10, 20, 30, 40, 4096)]
    assert device_frame.person_mask_dtypes == [np.dtype("float32")]
    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["sampling_mode"] == "instance_mask"
    assert payload["anchor_uv"] == [24.0, 59.0]
    assert payload["anchor_source"] == "lower_body_band"
    assert payload["anchor_depth_m"] == 4.25


def test_object_depth_fusion_reuses_recent_cached_payload(monkeypatch) -> None:
    monkeypatch.setenv("NOESIS_OBJECT_DEPTH_MAX_HZ_PER_TRACK", "1.0")
    monkeypatch.setenv("NOESIS_OBJECT_DEPTH_CACHE_MAX_AGE_MS", "1000")
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    device_frame = _NativePersonStatsDepthFrame(value=4.25, frame_w=1920, frame_h=1080)
    store.put(
        hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device_frame,
        )
    )
    processor = _fusion_processor(store)
    attached: list[str] = []

    def _attach(
        _batch_meta: Any,
        _obj_meta: Any,
        payload_json: str,
        _replace_existing: bool = True,
    ) -> bool:
        attached.append(payload_json)
        return True

    native_ext = SimpleNamespace(
        extract_object_mask=lambda _obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
        attach_object_depth=_attach,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", native_ext)
    hooks.reset_core_path_instrumentation()

    obj = _ObjMeta(rect_params=_Rect(left=10.0, top=20.0, width=30.0, height=40.0))
    processor.handle_servicemaker_frame(object(), _FrameMeta(object_items=[obj]))  # type: ignore[attr-defined]
    processor.handle_servicemaker_frame(
        object(),
        _FrameMeta(
            frame_number=12,
            buf_pts=123556789000,
            object_items=[_ObjMeta(rect_params=_Rect(left=11.0, top=20.0, width=30.0, height=40.0))],
        )
    )  # type: ignore[attr-defined]

    assert device_frame.roi_calls == []
    assert device_frame.person_stats_calls == [(10, 20, 30, 40, 4096)]
    cache_entry = processor._result_cache[(0, 7)]  # type: ignore[attr-defined]
    assert cache_entry.get("_payload_json") is None
    assert len(attached) == 2
    first = json.loads(attached[0])
    second = json.loads(attached[1])
    assert "_payload_json" not in first
    assert "_payload_json" not in second
    assert first["frame_id"] == 11
    assert second["frame_id"] == 12
    assert second["depth_median"] == 4.25
    counters = hooks.get_core_path_instrumentation_snapshot().get("counters", {})
    assert int(counters.get("detection_wake.object_depth_sampled", 0)) == 1
    assert int(counters.get("detection_wake.object_depth_cache_hit", 0)) == 1


def test_object_depth_fusion_uses_bbox_band_when_person_mask_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("NOESIS_OBJECT_DEPTH_ALLOW_HOST_ROI_COPY", "1")
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    device_frame = _DeviceDepthFrame(value=5.5, frame_w=1920, frame_h=1080)
    store.put(
        hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device_frame,
        )
    )
    processor = _fusion_processor(store)
    attached: list[str] = []

    def _attach(
        _batch_meta: Any,
        _obj_meta: Any,
        payload_json: str,
        _replace_existing: bool = True,
    ) -> bool:
        attached.append(payload_json)
        return True

    native_ext = SimpleNamespace(
        extract_object_mask=lambda _obj_meta: None,
        attach_object_depth=_attach,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", native_ext)

    frame_meta = _FrameMeta(object_items=[_ObjMeta(rect_params=_Rect(left=10.0, top=20.0, width=30.0, height=40.0))])
    processor.handle_servicemaker_frame(object(), frame_meta)  # type: ignore[attr-defined]

    assert device_frame.roi_calls == [(20, 50, 10, 10)]
    assert len(attached) == 1
    payload = json.loads(attached[0])
    assert payload["status"] == "no_ground_contact"
    assert payload["sampling_mode"] == "bbox_core"
    assert payload["sample_count"] == 100
    assert payload["depth_center"] == 5.5
    assert payload["depth_median"] == 5.5
    assert "anchor_source" not in payload
    assert "anchor_depth_m" not in payload
    assert payload["evidence_reason"] == "bbox_only_without_person_contact_support"


def test_object_depth_fusion_uses_native_bbox_band_stats_when_available(monkeypatch) -> None:
    store = hooks._AlignedDepthFrameStore()  # type: ignore[attr-defined]
    device_frame = _NativeStatsDepthFrame(value=5.5, frame_w=1920, frame_h=1080)
    store.put(
        hooks._AlignedDepthFrame(  # type: ignore[attr-defined]
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device_frame,
        )
    )
    processor = _fusion_processor(store)
    attached: list[str] = []

    def _attach(
        _batch_meta: Any,
        _obj_meta: Any,
        payload_json: str,
        _replace_existing: bool = True,
    ) -> bool:
        attached.append(payload_json)
        return True

    native_ext = SimpleNamespace(
        extract_object_mask=lambda _obj_meta: None,
        attach_object_depth=_attach,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", native_ext)
    hooks.reset_core_path_instrumentation()

    frame_meta = _FrameMeta(object_items=[_ObjMeta(rect_params=_Rect(left=10.0, top=20.0, width=30.0, height=40.0))])
    processor.handle_servicemaker_frame(object(), frame_meta)  # type: ignore[attr-defined]

    assert device_frame.roi_calls == []
    assert device_frame.stats_calls == [(20, 50, 10, 10, 4096)]
    assert len(attached) == 1
    payload = json.loads(attached[0])
    assert payload["status"] == "no_ground_contact"
    assert payload["sampling_mode"] == "bbox_core_native"
    assert payload["sample_count"] == 100
    assert payload["depth_median"] == 5.5
    assert "anchor_source" not in payload
    assert payload["evidence_reason"] == "bbox_only_without_person_contact_support"
    counters = hooks.get_core_path_instrumentation_snapshot().get("counters", {})
    assert int(counters.get("detection_wake.object_depth_native_stats", 0)) == 1
