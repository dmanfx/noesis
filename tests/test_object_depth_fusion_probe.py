from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np


def _load_probe_module(monkeypatch, *, native_ext: Any | None = None) -> Any:
    module_name = "test_yolo26_seg_depth_debug_probes"
    module_path = Path(__file__).resolve().parents[1] / "testpipelines" / "yolo26-seg-depth" / "debug_probes.py"
    if module_name in sys.modules:
        del sys.modules[module_name]
    fake_ext = native_ext or SimpleNamespace()
    monkeypatch.setitem(sys.modules, "noesis_depth_meta_ext", fake_ext)
    monkeypatch.setitem(
        sys.modules,
        "pyservicemaker",
        SimpleNamespace(BatchMetadataOperator=object),
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _Rect:
    left: float = 1.0
    top: float = 1.0
    width: float = 2.0
    height: float = 2.0


@dataclass
class _TextParams:
    display_text: str = "person 0.90"
    x_offset: int = 0
    y_offset: int = 0
    font_params: Any = field(default_factory=lambda: SimpleNamespace(size=16))


@dataclass
class _ObjMeta:
    object_id: int = 7
    class_id: int = 0
    confidence: float = 0.9
    obj_label: str = "Person"
    rect_params: _Rect = field(default_factory=_Rect)
    text_params: _TextParams = field(default_factory=_TextParams)
    attached_payloads: List[str] = field(default_factory=list)


@dataclass
class _TensorMeta:
    unique_id: int = 2
    layers: Dict[str, Any] = field(default_factory=dict)

    def get_layers(self) -> Dict[str, Any]:
        return dict(self.layers)


@dataclass
class _TensorItem:
    tensor_meta: _TensorMeta

    def as_tensor_output(self) -> _TensorMeta:
        return self.tensor_meta


@dataclass
class _FrameMeta:
    tensor_items: List[Any]
    object_items: List[Any]
    source_id: int = 0
    frame_number: int = 11
    source_frame_width: int = 4
    source_frame_height: int = 4
    frame_width: int = 6
    frame_height: int = 4
    buf_pts: int = 123456789000


@dataclass
class _BatchMeta:
    frame_items: List[Any]


@dataclass
class _CalibrationSnapshot:
    intrinsics: np.ndarray
    extrinsics_col_major: List[float]
    floor_y: float = 0.0
    unit_scale: float = 1.0
    image_size: tuple[int, int] = (1920, 1080)


@dataclass
class _CalibrationResolver:
    snapshot_payload: Any
    labels: Dict[int, str] = field(default_factory=lambda: {0: "family-room"})

    def camera_labels(self) -> Dict[int, str]:
        return dict(self.labels)

    def snapshot(self, source_id: int, camera_id: str | None = None) -> Any:
        return self.snapshot_payload


def _quadrants(mean: float) -> Dict[str, Dict[str, float]]:
    return {
        "left": {"finite_fraction": 1.0, "mean": mean},
        "right": {"finite_fraction": 1.0, "mean": mean},
        "top": {"finite_fraction": 1.0, "mean": mean},
        "bottom": {"finite_fraction": 1.0, "mean": mean},
    }


def _aligned_depth_frame(module: Any, *, frame_w: int, frame_h: int, value: float = 4.0) -> Any:
    return module.AlignedDepthFrame(
        key=(0, 11, 123456789),
        source_id=0,
        frame_id=11,
        pts_us=123456789,
        depth_map=np.ones((frame_h, frame_w), dtype=np.float32) * float(value),
        valid_mask=np.ones((frame_h, frame_w), dtype=bool),
        frame_w=frame_w,
        frame_h=frame_h,
        source_frame_w=frame_w,
        source_frame_h=frame_h,
        depth_w=frame_w,
        depth_h=frame_h,
        unit="m",
        is_metric=True,
        model_name="depth-anything-v2-metric-hypersim-vits",
        finite_fraction=1.0,
        quadrant_stats=_quadrants(float(value)),
        transform_desc=f"resize({frame_w}x{frame_h}->{frame_w}x{frame_h})",
    )


def test_depth_frame_probe_aligns_to_frame_space_not_source_space(monkeypatch) -> None:
    module = _load_probe_module(monkeypatch)
    store = module.AlignedDepthFrameStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_map = np.array(
        [[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]],
        dtype=np.float32,
    )
    frame_meta = _FrameMeta(
        tensor_items=[_TensorItem(_TensorMeta(layers={"depth": depth_map}))],
        object_items=[],
        source_frame_width=4,
        source_frame_height=2,
        frame_width=6,
        frame_height=4,
    )
    probe = module.DepthFrameProbe(
        depth_store=store,
        stats=stats,
        depth_gie_id=2,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        fallback_frame_size=(6, 4),
    )
    probe.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    stored = store.get((0, 11, 123456789))
    assert stored is not None
    assert stored.depth_map.shape == (4, 6)
    assert stored.frame_w == 6
    assert stored.source_frame_w == 4
    assert stored.transform_desc == "resize(4x2->6x4)"


def test_object_depth_fusion_uses_canonical_frame_for_right_side_object(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        obj_meta.attached_payloads.append(payload_json)
        return True

    native = SimpleNamespace(
        attach_object_depth=_attach,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_map = np.array(
        [[[0.0, 0.2, 0.4, 0.6], [0.8, 1.0, 1.2, 1.4]]],
        dtype=np.float32,
    )
    frame_meta = _FrameMeta(
        tensor_items=[_TensorItem(_TensorMeta(layers={"depth": depth_map}))],
        object_items=[],
        source_frame_width=4,
        source_frame_height=2,
        frame_width=6,
        frame_height=4,
    )
    depth_probe = module.DepthFrameProbe(
        depth_store=depth_store,
        stats=stats,
        depth_gie_id=2,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        fallback_frame_size=(6, 4),
    )
    depth_probe.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    obj_meta = _ObjMeta(rect_params=_Rect(left=4.0, top=1.0, width=2.0, height=2.0))
    frame_meta.object_items = [obj_meta]
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        report_interval_s=9999.0,
    )
    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    assert len(attached) == 1
    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["sample_count"] == 4
    assert payload["bbox"] == [4.0, 1.0, 2.0, 2.0]
    depth_suffix = f"z={payload['depth_median']:.2f}m"
    assert obj_meta.text_params.display_text.endswith(depth_suffix)
    assert obj_meta.obj_label.endswith(depth_suffix)
    assert overlay_store.pop((0, 11, 123456789))


def test_object_depth_fusion_marks_transform_mismatch_when_bbox_is_outside_frame(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        return True

    native = SimpleNamespace(
        attach_object_depth=_attach,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_store.put(
        module.AlignedDepthFrame(
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=np.ones((4, 6), dtype=np.float32),
            valid_mask=np.ones((4, 6), dtype=bool),
            frame_w=6,
            frame_h=4,
            source_frame_w=4,
            source_frame_h=4,
            depth_w=6,
            depth_h=4,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            finite_fraction=1.0,
            quadrant_stats=_quadrants(1.0),
            transform_desc="resize(6x4->6x4)",
        )
    )
    obj_meta = _ObjMeta(rect_params=_Rect(left=8.0, top=1.0, width=2.0, height=2.0))
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=6, frame_height=4)
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        report_interval_s=9999.0,
    )
    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    payload = json.loads(attached[0])
    assert payload["status"] == "transform_mismatch"
    assert payload["sample_count"] == 0


def test_object_depth_fusion_reports_missing_mask_without_bbox_fallback(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        return True

    native = SimpleNamespace(attach_object_depth=_attach, extract_object_mask=lambda obj_meta: None)
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_store.put(
        module.AlignedDepthFrame(
            key=(0, 11, 123456789),
            source_id=0,
            frame_id=11,
            pts_us=123456789,
            depth_map=np.ones((4, 6), dtype=np.float32),
            valid_mask=np.ones((4, 6), dtype=bool),
            frame_w=6,
            frame_h=4,
            source_frame_w=4,
            source_frame_h=4,
            depth_w=6,
            depth_h=4,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            finite_fraction=1.0,
            quadrant_stats=_quadrants(1.0),
            transform_desc="resize(6x4->6x4)",
        )
    )
    obj_meta = _ObjMeta(rect_params=_Rect(left=1.0, top=1.0, width=2.0, height=2.0))
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=6, frame_height=4)
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        report_interval_s=9999.0,
    )
    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    payload = json.loads(attached[0])
    assert payload["status"] == "missing_mask"
    assert payload["sample_count"] == 0


def test_extract_person_anchor_prefers_lower_body_band(monkeypatch) -> None:
    module = _load_probe_module(monkeypatch)
    mask = np.ones((40, 40), dtype=bool)
    depth_crop = np.ones((40, 40), dtype=np.float32) * 3.25

    anchor = module._extract_person_anchor(mask, depth_crop, frame_origin=(100, 50))

    assert anchor.anchor_source == "lower_body_band"
    assert anchor.anchor_depth_m == 3.25
    assert anchor.lower_body_sample_count >= 64
    assert anchor.lower_body_valid_fraction == 1.0
    assert anchor.foot_uv is not None
    assert 100.0 <= anchor.foot_uv[0] <= 140.0
    assert anchor.foot_uv[1] > 80.0


def test_extract_person_anchor_falls_back_to_torso_core_when_lower_band_invalid(monkeypatch) -> None:
    module = _load_probe_module(monkeypatch)
    mask = np.ones((40, 40), dtype=bool)
    depth_crop = np.full((40, 40), np.nan, dtype=np.float32)
    depth_crop[14:28, 10:30] = 6.5

    anchor = module._extract_person_anchor(mask, depth_crop, frame_origin=(0, 0))

    assert anchor.anchor_source == "torso_core"
    assert anchor.anchor_depth_m == 6.5
    assert anchor.lower_body_sample_count == 0
    assert anchor.torso_sample_count > 0
    assert anchor.foot_uv is not None


def test_object_depth_fusion_projects_person_anchor_when_depth_and_floor_agree(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        return True

    native = SimpleNamespace(
        attach_object_depth=_attach,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((40, 40), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)
    monkeypatch.setattr(
        module,
        "pixel_to_world",
        lambda K, E, floor_y, unit_scale, u, v, depth_m=None: SimpleNamespace(
            ok=True,
            world_point=[1.2, 0.0, 3.6] if depth_m is not None else [1.0, 0.0, 3.2],
        ),
    )

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_store.put(_aligned_depth_frame(module, frame_w=80, frame_h=60, value=4.5))
    obj_meta = _ObjMeta(rect_params=_Rect(left=20.0, top=10.0, width=40.0, height=40.0))
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=80, frame_height=60)
    calibration = _CalibrationResolver(
        _CalibrationSnapshot(
            intrinsics=np.eye(3, dtype=np.float64),
            extrinsics_col_major=[1.0, 0.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.0, 0.0, 1.0],
            image_size=(80, 60),
        )
    )
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        calibration_resolver=calibration,
        report_interval_s=9999.0,
    )

    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["projection_method"] == "depth"
    assert payload["spatial_status"] == "ok"
    assert payload["anchor_source"] == "lower_body_band"
    assert payload["anchor_depth_m"] == 4.5
    assert payload["world_point"] == [1.2, 0.0, 3.6]
    assert payload["world_point_floor"] == [1.0, 0.0, 3.2]


def test_object_depth_fusion_floor_guards_when_depth_projection_disagrees(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        return True

    native = SimpleNamespace(
        attach_object_depth=_attach,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((40, 40), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)
    monkeypatch.setattr(
        module,
        "pixel_to_world",
        lambda K, E, floor_y, unit_scale, u, v, depth_m=None: SimpleNamespace(
            ok=True,
            world_point=[9.0, 0.0, 9.0] if depth_m is not None else [1.0, 0.0, 3.0],
        ),
    )

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_store.put(_aligned_depth_frame(module, frame_w=80, frame_h=60, value=5.0))
    obj_meta = _ObjMeta(rect_params=_Rect(left=20.0, top=10.0, width=40.0, height=40.0))
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=80, frame_height=60)
    calibration = _CalibrationResolver(
        _CalibrationSnapshot(
            intrinsics=np.eye(3, dtype=np.float64),
            extrinsics_col_major=[1.0, 0.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.0, 0.0, 1.0],
            image_size=(80, 60),
        )
    )
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        calibration_resolver=calibration,
        report_interval_s=9999.0,
    )

    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    payload = json.loads(attached[0])
    assert payload["projection_method"] == "floor_guarded"
    assert payload["spatial_status"] == "ok"
    assert payload["world_point"] == [1.0, 0.0, 3.0]
    assert payload["world_point_depth"] == [9.0, 0.0, 9.0]


def test_object_depth_fusion_marks_geometry_unavailable_without_calibration(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        return True

    native = SimpleNamespace(
        attach_object_depth=_attach,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((40, 40), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_store.put(_aligned_depth_frame(module, frame_w=80, frame_h=60, value=4.0))
    obj_meta = _ObjMeta(rect_params=_Rect(left=20.0, top=10.0, width=40.0, height=40.0))
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=80, frame_height=60)
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        calibration_resolver=_CalibrationResolver(None),
        report_interval_s=9999.0,
    )

    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["spatial_status"] == "geometry_unavailable"
    assert "world_point" not in payload


def test_object_depth_fusion_keeps_non_person_payload_raw_only(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        return True

    native = SimpleNamespace(
        attach_object_depth=_attach,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((20, 20), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=1)
    depth_store.put(_aligned_depth_frame(module, frame_w=40, frame_h=40, value=2.0))
    obj_meta = _ObjMeta(class_id=56, obj_label="Chair", rect_params=_Rect(left=10.0, top=10.0, width=20.0, height=20.0))
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=40, frame_height=40)
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        calibration_resolver=_CalibrationResolver(None),
        report_interval_s=9999.0,
    )

    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert "projection_method" not in payload
    assert "world_point" not in payload


def test_depth_overlay_probe_clamps_right_edge_labels(monkeypatch) -> None:
    module = _load_probe_module(monkeypatch)
    overlay_store = module.OverlayStateStore()
    result = module.ObjectDepthResult(
        source_id=0,
        frame_id=11,
        object_id=7,
        class_id=0,
        bbox=(80.0, 12.0, 18.0, 20.0),
        score=0.9,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=16,
        valid_fraction=1.0,
        depth_median=2.5,
    )
    obj_meta = _ObjMeta(rect_params=_Rect(left=80.0, top=12.0, width=18.0, height=20.0))
    obj_meta.text_params.x_offset = 96
    obj_meta.text_params.y_offset = 10
    key = (0, 11, 123456789)
    overlay_store.put(key, {module._object_signature(obj_meta): result})
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=100, frame_height=60)
    overlay = module.DepthOverlayProbe(overlay_store=overlay_store, fallback_frame_size=(100, 60))
    overlay.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    assert obj_meta.text_params.display_text.endswith("z=2.50m")
    assert obj_meta.obj_label.endswith("z=2.50m")
    assert obj_meta.text_params.x_offset < 96


def test_depth_overlay_probe_uses_object_label_as_base_text(monkeypatch) -> None:
    module = _load_probe_module(monkeypatch)
    overlay_store = module.OverlayStateStore()
    result = module.ObjectDepthResult(
        source_id=0,
        frame_id=11,
        object_id=7,
        class_id=0,
        bbox=(10.0, 12.0, 18.0, 20.0),
        score=0.9,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=16,
        valid_fraction=1.0,
        depth_median=7.52,
    )
    obj_meta = _ObjMeta(rect_params=_Rect(left=10.0, top=12.0, width=18.0, height=20.0))
    obj_meta.obj_label = "Person"
    obj_meta.text_params.display_text = ""
    key = (0, 11, 123456789)
    overlay_store.put(key, {module._object_signature(obj_meta): result})
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=100, frame_height=60)
    overlay = module.DepthOverlayProbe(overlay_store=overlay_store, fallback_frame_size=(100, 60))

    overlay.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    assert obj_meta.obj_label == "Person z=7.52m"
    assert obj_meta.text_params.display_text == "Person z=7.52m"


def test_object_depth_fusion_reuses_previous_depth_frame_when_depth_interval_skips(monkeypatch) -> None:
    attached: List[str] = []

    def _attach(obj_meta: Any, payload_json: str, replace_existing: bool = True) -> bool:
        attached.append(payload_json)
        obj_meta.attached_payloads.append(payload_json)
        return True

    native = SimpleNamespace(
        attach_object_depth=_attach,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)

    depth_store = module.AlignedDepthFrameStore()
    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=True, depth_every_n_frames=2)
    depth_store.put(
        module.AlignedDepthFrame(
            key=(0, 10, 123455789),
            source_id=0,
            frame_id=10,
            pts_us=123455789,
            depth_map=np.ones((4, 6), dtype=np.float32) * 3.5,
            valid_mask=np.ones((4, 6), dtype=bool),
            frame_w=6,
            frame_h=4,
            source_frame_w=4,
            source_frame_h=4,
            depth_w=6,
            depth_h=4,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            finite_fraction=1.0,
            quadrant_stats=_quadrants(3.5),
            transform_desc="resize(6x4->6x4)",
        )
    )
    obj_meta = _ObjMeta(rect_params=_Rect(left=1.0, top=1.0, width=2.0, height=2.0))
    frame_meta = _FrameMeta(
        tensor_items=[],
        object_items=[obj_meta],
        frame_width=6,
        frame_height=4,
        frame_number=11,
        buf_pts=123456789000,
    )
    fusion = module.ObjectDepthFusionProbe(
        depth_store=depth_store,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        depth_every_n_frames=2,
        report_interval_s=9999.0,
    )

    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    payload = json.loads(attached[0])
    assert payload["status"] == "ok"
    assert payload["depth_median"] == 3.5
    assert stats.depth_reused_frames == 1


def test_object_depth_fusion_depth_disabled_skips_attachment(monkeypatch) -> None:
    native = SimpleNamespace(
        attach_object_depth=lambda obj_meta, payload_json, replace_existing=True: True,
        extract_object_mask=lambda obj_meta: {
            "threshold": 0.5,
            "data": np.ones((2, 2), dtype=np.float32),
        },
    )
    module = _load_probe_module(monkeypatch, native_ext=native)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: native)

    overlay_store = module.OverlayStateStore()
    stats = module.PrototypeRuntimeStats(depth_enabled=False, depth_every_n_frames=1)
    obj_meta = _ObjMeta(rect_params=_Rect(left=1.0, top=1.0, width=2.0, height=2.0))
    frame_meta = _FrameMeta(tensor_items=[], object_items=[obj_meta], frame_width=6, frame_height=4)
    fusion = module.ObjectDepthFusionProbe(
        depth_store=None,
        overlay_store=overlay_store,
        stats=stats,
        depth_model_name="depth-disabled",
        depth_unit="m",
        depth_is_metric=True,
        report_interval_s=9999.0,
    )

    fusion.handle_metadata(_BatchMeta(frame_items=[frame_meta]))

    assert obj_meta.attached_payloads == []
    assert stats.frames_total == 1
    assert stats.detections_total == 1
