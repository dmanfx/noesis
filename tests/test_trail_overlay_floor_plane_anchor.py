from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from geometry.homography import project_world_to_image
from noesis.pipelines import hooks
from noesis.telemetry.bev import CalibrationSnapshot


@dataclass
class _Rect:
    left: float
    top: float
    width: float
    height: float


@dataclass
class _ObjMeta:
    object_id: int
    rect_params: _Rect
    class_id: int = 0


@dataclass
class _FrameMeta:
    object_items: List[Any]
    source_id: int = 0
    compositor_rect: _Rect = field(default_factory=lambda: _Rect(0.0, 0.0, 1280.0, 720.0))
    source_frame_width: int = 1280
    source_frame_height: int = 720
    appended: List[Any] = field(default_factory=list)

    def append(self, item: Any) -> None:
        self.appended.append(item)


class _CalibrationProvider:
    def __init__(self, snapshot: CalibrationSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, source_id: int, camera_id: str) -> CalibrationSnapshot:
        return self._snapshot


class _AnalyticsStub:
    def __init__(self) -> None:
        self.tracks_by_sensor: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def set_tracks(self, sensor_id: int, tracks: Dict[int, Dict[str, Any]]) -> None:
        self.tracks_by_sensor[int(sensor_id)] = dict(tracks)

    def get_active_track_map(self, sensor_id: int) -> Dict[int, Dict[str, Any]]:
        tracks = self.tracks_by_sensor.get(int(sensor_id), {})
        return {int(k): dict(v) for k, v in tracks.items()}


class _DisplayMeta:
    def __init__(self) -> None:
        self.n_lines = 0
        self.n_labels = 0
        self.lines: List[Dict[str, float]] = []

    def add_line(self, line: Any) -> None:
        self.lines.append(
            {
                "x1": float(line.x1),
                "y1": float(line.y1),
                "x2": float(line.x2),
                "y2": float(line.y2),
                "a": float(line.color.a),
                "width": float(line.width),
            }
        )
        self.n_lines += 1

    def add_text(self, text: Any) -> None:
        self.n_labels += 1


class _BatchMeta:
    def acquire_display_meta(self) -> _DisplayMeta:
        return _DisplayMeta()


def _build_snapshot() -> CalibrationSnapshot:
    K = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_world = np.array([0.0, 2.2, -6.0], dtype=np.float64)
    forward = np.array([0.0, -0.25, 1.0], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    down /= np.linalg.norm(down)
    R_wc = np.column_stack([right, down, forward])
    R_cw = R_wc.T
    t = -R_cw @ camera_world
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_cw
    E[:3, 3] = t
    return CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=K,
        extrinsics_col_major=list(E.flatten(order="F")),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
    )


def _project(calib: CalibrationSnapshot, world_xyz: List[float]) -> tuple[float, float]:
    uv = project_world_to_image(
        world_xyz,
        calib.intrinsics,
        calib.extrinsics_col_major,
        calib.image_size,
        unit_scale=1.0,
    )
    assert uv is not None
    return float(uv[0]), float(uv[1])


def _person_bbox(
    calib: CalibrationSnapshot,
    *,
    foot_world: List[float],
    height_scene: float = 1.8,
    width_px: float = 80.0,
) -> tuple[list[float], tuple[float, float]]:
    u_foot, v_foot = _project(calib, foot_world)
    u_top, v_top = _project(calib, [float(foot_world[0]), float(foot_world[1]) + float(height_scene), float(foot_world[2])])
    bbox = [float(u_foot - (width_px * 0.5)), float(v_top), float(width_px), float(v_foot - v_top)]
    assert bbox[3] > 0.0
    return bbox, (float(u_foot), float(v_foot))


def _build_processor(
    *,
    anchor_mode: str = "floor_plane_gravity_drop",
    predicted_alpha_scale: float = 0.5,
    analytics: Optional[_AnalyticsStub] = None,
) -> tuple[hooks.TrailOverlayProcessor, CalibrationSnapshot, _AnalyticsStub]:
    calib = _build_snapshot()
    analytics = analytics or _AnalyticsStub()
    pipeline = SimpleNamespace(
        components={},
        frame_size=calib.image_size,
        camera_labels={0: "cam0"},
        bev_calibration=_CalibrationProvider(calib),
        analytics_telemetry_processor=analytics,
        stable_id_mgr=None,
    )
    config = hooks.TrailOverlayConfig(
        enabled=True,
        class_ids=frozenset({0}),
        anchor_mode=anchor_mode,
        window_s=8.0,
        draw_stride=1,
        min_step_px=0.0,
        min_dt_s=0.0,
        smooth_tau_s=0.0,
        max_speed_px_per_s=0.0,
        gap_predict_ttl_s=1.0,
        gap_predict_decay_tau_s=0.75,
        predicted_alpha_scale=predicted_alpha_scale,
        min_alpha=0.0,
    )
    return hooks.TrailOverlayProcessor(pipeline=pipeline, config=config), calib, analytics


def test_gravity_drop_anchor_stays_floor_locked_under_partial_occlusion() -> None:
    processor, calib, analytics = _build_processor()
    bbox_full, foot_uv = _person_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    analytics.set_tracks(0, {7: {"tracker_id": 7, "bbox": bbox_full}})
    frame_1 = _FrameMeta(object_items=[_ObjMeta(object_id=7, rect_params=_Rect(*bbox_full))])

    processor._handle_frame(frame_1, SimpleNamespace(), now=10.0)  # type: ignore[attr-defined]

    state = processor._tracks[0][7]  # type: ignore[attr-defined]
    assert state.height_ref_scene is not None
    first_point = state.points[-1]
    assert first_point.predicted is False

    bbox_partial = [bbox_full[0], bbox_full[1], bbox_full[2], bbox_full[3] * 0.25]
    analytics.set_tracks(0, {7: {"tracker_id": 7, "bbox": bbox_partial}})
    frame_2 = _FrameMeta(object_items=[_ObjMeta(object_id=7, rect_params=_Rect(*bbox_partial))])

    processor._handle_frame(frame_2, SimpleNamespace(), now=10.1)  # type: ignore[attr-defined]

    second_point = state.points[-1]
    raw_bbox_bottom_y = float(bbox_partial[1] + bbox_partial[3])
    assert second_point.predicted is False
    assert abs(float(second_point.y) - float(foot_uv[1])) < 20.0
    assert abs(float(second_point.y) - raw_bbox_bottom_y) > 80.0
    assert abs(float(second_point.y) - float(first_point.y)) < 20.0


def test_gap_prediction_only_extends_after_track_loss_until_ttl() -> None:
    processor, calib, analytics = _build_processor()
    bbox_1, _ = _person_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    bbox_2, _ = _person_bbox(calib, foot_world=[0.4, 0.0, 6.0])

    analytics.set_tracks(0, {7: {"tracker_id": 7, "bbox": bbox_1}})
    processor._handle_frame(
        _FrameMeta(object_items=[_ObjMeta(object_id=7, rect_params=_Rect(*bbox_1))]),
        SimpleNamespace(),
        now=10.0,
    )  # type: ignore[attr-defined]

    analytics.set_tracks(0, {7: {"tracker_id": 7, "bbox": bbox_2}})
    processor._handle_frame(
        _FrameMeta(object_items=[_ObjMeta(object_id=7, rect_params=_Rect(*bbox_2))]),
        SimpleNamespace(),
        now=10.2,
    )  # type: ignore[attr-defined]

    state = processor._tracks[0][7]  # type: ignore[attr-defined]
    measured_len = len(state.points)

    analytics.set_tracks(0, {})
    processor._handle_frame(_FrameMeta(object_items=[]), SimpleNamespace(), now=10.6)  # type: ignore[attr-defined]

    assert len(state.points) == measured_len + 1
    assert state.points[-1].predicted is True
    assert float(state.points[-1].x) > float(state.points[-2].x)

    after_gap_len = len(state.points)
    processor._handle_frame(_FrameMeta(object_items=[]), SimpleNamespace(), now=11.4)  # type: ignore[attr-defined]
    assert len(state.points) == after_gap_len


def test_floor_plane_anchor_falls_back_to_bbox_bottom_without_analytics() -> None:
    processor, _calib, _analytics = _build_processor()
    processor.pipeline.analytics_telemetry_processor = None
    frame = _FrameMeta(object_items=[_ObjMeta(object_id=9, rect_params=_Rect(100.0, 50.0, 80.0, 200.0))])

    processor._handle_frame(frame, SimpleNamespace(), now=0.0)  # type: ignore[attr-defined]

    point = processor._tracks[0][9].points[-1]  # type: ignore[attr-defined]
    assert point.predicted is False
    assert point.x == pytest.approx(140.0)
    assert point.y == pytest.approx(250.0)


def test_source_to_mosaic_uses_configured_tiler_rect_when_compositor_missing() -> None:
    processor, _calib, _analytics = _build_processor()
    processor.pipeline.components = {
        "tiler": SimpleNamespace(config={"width": 3840, "height": 720, "columns": 3, "rows": 1})
    }
    processor.pipeline.camera_labels = {0: "living-room", 1: "kitchen", 2: "family-room"}
    frame = _FrameMeta(
        object_items=[],
        source_id=2,
        compositor_rect=None,  # type: ignore[arg-type]
        source_frame_width=1280,
        source_frame_height=720,
    )

    x, y = processor._source_to_mosaic(frame, 640.0, 360.0, (1280, 720))  # type: ignore[attr-defined]

    assert x == pytest.approx(3200.0)
    assert y == pytest.approx(360.0)


def test_floor_plane_anchor_maps_analytics_uv_into_source_tile_without_compositor_rect() -> None:
    processor, calib, analytics = _build_processor()
    processor.pipeline.components = {
        "tiler": SimpleNamespace(config={"width": 3840, "height": 720, "columns": 3, "rows": 1})
    }
    processor.pipeline.camera_labels = {0: "living-room", 1: "kitchen", 2: "family-room"}
    bbox, foot_uv = _person_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    analytics.set_tracks(
        1,
        {
            7: {
                "tracker_id": 7,
                "bbox": bbox,
                "world": [0.0, 0.0, 6.0],
                "world_valid": True,
                "image_foot": [float(foot_uv[0]), float(foot_uv[1])],
            }
        },
    )
    frame = _FrameMeta(
        object_items=[_ObjMeta(object_id=7, rect_params=_Rect(*bbox))],
        source_id=1,
        compositor_rect=None,  # type: ignore[arg-type]
        source_frame_width=1280,
        source_frame_height=720,
    )

    processor._handle_frame(frame, SimpleNamespace(), now=10.0)  # type: ignore[attr-defined]

    point = processor._tracks[1][7].points[-1]  # type: ignore[attr-defined]
    assert point.x == pytest.approx(1280.0 + foot_uv[0], abs=1.0)
    assert point.y == pytest.approx(foot_uv[1], abs=1.0)


def test_predicted_segments_render_with_reduced_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    processor, _calib, _analytics = _build_processor(anchor_mode="bbox_bottom", predicted_alpha_scale=0.5)
    processor._tracks[0] = {
        7: hooks._TrailTrackState(  # type: ignore[attr-defined]
            points=deque(
                [
                    hooks._TrailPoint(ts=0.0, x=10.0, y=20.0, predicted=False),  # type: ignore[attr-defined]
                    hooks._TrailPoint(ts=0.5, x=40.0, y=60.0, predicted=True),  # type: ignore[attr-defined]
                ]
            ),
            last_seen_ts=0.5,
        )
    }

    class _Color:
        def __init__(self) -> None:
            self.r = 0.0
            self.g = 0.0
            self.b = 0.0
            self.a = 0.0

    class _Line:
        def __init__(self) -> None:
            self.x1 = 0
            self.y1 = 0
            self.x2 = 0
            self.y2 = 0
            self.width = 0
            self.color = _Color()

    class _Font:
        def __init__(self) -> None:
            self.name = ""
            self.size = 0
            self.color = _Color()

    class _Text:
        def __init__(self) -> None:
            self.display_text = ""
            self.x_offset = 0
            self.y_offset = 0
            self.font = _Font()
            self.set_bg_color = False

    fake_osd = SimpleNamespace(Line=_Line, Text=_Text, FontFamily=SimpleNamespace(Serif="Serif"))
    monkeypatch.setattr(hooks, "ds_osd", fake_osd)

    frame = _FrameMeta(object_items=[])
    batch = _BatchMeta()
    processor._handle_frame(frame, batch, now=1.0)  # type: ignore[attr-defined]

    assert frame.appended
    meta = frame.appended[0]
    assert meta.lines
    assert meta.lines[0]["a"] == pytest.approx(0.4375, abs=1e-6)
