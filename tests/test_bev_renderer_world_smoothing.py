from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from noesis.telemetry.bev import BevRenderer, CalibrationSnapshot, Footpoint


class _FakeWs:
    def __init__(self) -> None:
        self.messages = []

    @staticmethod
    def response_model_timing_since(_started_ns):
        return {"duration_ms": 0.0}

    def broadcast_bev_sync(self, payload, **_kwargs):
        self.messages.append(payload)
        return SimpleNamespace(
            submission_id=len(self.messages),
            message_count=1,
        )


def _calibration(unit_scale: float = 1.0) -> CalibrationSnapshot:
    return CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.6,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=unit_scale,
    )


def _floor_camera_calibration() -> CalibrationSnapshot:
    return CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.6, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )


def _active_floorplan(
    *,
    bounds: dict[str, float] | None = None,
    grid_shape: list[int] | None = None,
    grid_res_m: float = 0.1,
    snapshot_ts_us: int = 100,
    floorplan_ts_us: int = 200,
    ray_to_floorplan_alignment: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "camera_id": "cam0",
        "bounds": bounds
        or {"min_x": -4.0, "max_x": 4.0, "min_z": 0.0, "max_z": 8.0},
        "grid_shape": grid_shape or [80, 80],
        "grid_res_m": grid_res_m,
        "frame": "camera_local_ground_m",
        "units": "meters",
        "snapshot_ts_us": snapshot_ts_us,
        "floorplan_ts_us": floorplan_ts_us,
        "snapshot_id": f"snapshot-{snapshot_ts_us}",
        "snapshot_content_sha256": "a" * 64,
        "calibration_fingerprint": "b" * 64,
        "source": "active_floorplan",
    }
    if ray_to_floorplan_alignment is not None:
        payload["ray_to_floorplan_alignment"] = ray_to_floorplan_alignment
    return payload


def _coverage_envelopes() -> dict[str, object]:
    return {
        "contract": "noesis.bev.coverage_envelopes",
        "contract_version": 1,
        "frame": "camera_local_ground_m",
        "units": "meters",
        "cameras": {
            "cam0": {
                "boundary_tolerance_m": 0.25,
                "regions": [
                    {
                        "id": "main",
                        "polygon_xz_m": [
                            [0.0, 0.0],
                            [4.0, 0.0],
                            [4.0, 4.0],
                            [0.0, 4.0],
                        ],
                    },
                    {
                        "id": "foyer",
                        "polygon_xz_m": [
                            [3.0, 4.0],
                            [6.0, 4.0],
                            [6.0, 8.0],
                            [3.0, 8.0],
                        ],
                    },
                ],
            },
        },
    }


def _coverage_calibration() -> CalibrationSnapshot:
    return CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.eye(3, dtype=np.float64),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )


def _coverage_renderer(
    ws: _FakeWs,
    *,
    trails: bool = False,
) -> BevRenderer:
    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": trails,
            "draw_stride": 1,
            "smooth_tau_s": 0.0,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
        },
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera: _active_floorplan(
            bounds={
                "min_x": 0.0,
                "max_x": 4.0,
                "min_z": 0.0,
                "max_z": 4.0,
            },
            grid_shape=[40, 40],
        ),
        coverage_envelopes_cfg=_coverage_envelopes(),
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return renderer


def test_camera_local_bev_publishes_floorplan_frame_and_prefers_floor_contact_ray() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.004, 0.0, -2.56],
            [0.0, 0.01, -3.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=1.25,
                world_z=2.5,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert payload["frame"] == "camera_local_ground_m"
    assert payload["world_frame"] == "camera_local_ground_m"
    assert payload["frame_mode"] == "camera_local"
    assert point["displaySource"] == "floor_contact_ray"
    assert math.isclose(point["x"], 0.4, abs_tol=1e-3)
    assert math.isclose(point["y"], 3.59, abs_tol=1e-3)


def test_camera_local_bev_replaces_floor_only_world_with_current_floor_contact_ray() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.004, 0.0, -2.56],
            [0.0, 0.01, -3.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=1.25,
                world_z=2.5,
                anchor_source="pose_floor_only",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "floor_contact_ray"
    assert math.isclose(point["x"], 0.4, abs_tol=1e-3)
    assert math.isclose(point["y"], 3.59, abs_tol=1e-3)


def test_camera_local_bev_normalizes_floor_contact_into_floorplan_space() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera: _active_floorplan(
            ray_to_floorplan_alignment={
                "version": 1,
                "quality": "ok",
                "reason": "fit_ok",
                "sample_count": 128,
                "inlier_count": 128,
                "residual_m": {"p50": 0.01, "p95": 0.02},
                "matrix_2x3": [[1.0, 0.0, 0.2], [0.0, 1.0, -1.0]],
            },
        ),
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.004, 0.0, -2.56],
            [0.0, 0.01, -3.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=1.25,
                world_z=2.5,
                anchor_source="pose_floor_only",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert payload["floorplanAlignment"]["applied"] is True
    assert point["displaySource"] == "floor_contact_ray"
    assert math.isclose(point["x"], 0.6, abs_tol=1e-3)
    assert math.isclose(point["y"], 2.59, abs_tol=1e-3)
    assert math.isclose(point["floorplanX"], 0.6, abs_tol=1e-3)
    assert math.isclose(point["floorplanZ"], 2.59, abs_tol=1e-3)


def test_camera_local_bev_uses_live_fused_world_when_no_floor_contact_ray_is_requested() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.004, 0.0, -2.56],
            [0.0, 0.01, -3.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="depth_anchor",
                stable_id=7,
                tracker_id=101,
                world_x=1.25,
                world_z=2.5,
                anchor_source="pose_depth_fused",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "world_to_camera_local"
    assert math.isclose(point["x"], 1.25, abs_tol=1e-6)
    assert math.isclose(point["y"], 2.5, abs_tol=1e-6)


def test_camera_local_bev_prefers_depth_fused_world_for_display() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.004, 0.0, -2.56],
            [0.0, 0.01, -3.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=0.4,
                world_z=3.59,
                anchor_source="pose_depth_fused",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "world_to_camera_local"
    assert math.isclose(point["x"], 0.4, abs_tol=1e-3)
    assert math.isclose(point["y"], 3.59, abs_tol=1e-3)


def test_camera_local_bev_keeps_depth_diagnostic_when_canonical_world_wins(monkeypatch) -> None:
    monkeypatch.setenv("NOESIS_BEV_ALIGNMENT_DEBUG", "1")
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 0.0,
            "max_jump_m": 0.05,
            "alpha": 0.1,
            "beta": 0.0,
        },
        frame="camera_local_ground_m",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.004, 0.0, -2.56],
            [0.0, 0.01, -3.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=0.4,
                world_z=3.59,
                depth_m=6.0,
                depth_source="depth_registered_m",
                anchor_source="pose_depth_fused",
                canonical_world_required=True,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "world_to_camera_local"
    assert math.isclose(point["x"], 0.4, abs_tol=1e-3)
    assert math.isclose(point["y"], 3.59, abs_tol=1e-3)
    selection = point["alignmentDebug"]["displaySelection"]
    assert selection["selected"] == "world_to_camera_local"
    assert selection["reason"] == "canonical_live_world_only"
    assert selection["registeredDepthCandidate"]["depthSource"] == "depth_registered_m"
    assert selection["worldCandidate"]["insideBounds"] is True

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=1.4,
                world_z=3.59,
                depth_m=6.0,
                depth_source="depth_registered_m",
                anchor_source="pose_depth_fused",
                canonical_world_required=True,
            )
        ],
        timestamp_us=1_100_000,
    )
    second_payload = ws.messages[-1]
    second = second_payload["footpoints"][0]
    second_world = second["alignmentDebug"]["displaySelection"]["worldCandidate"]
    assert math.isclose(second["x"], second_world["x"], abs_tol=1e-9)
    assert math.isclose(second["y"], second_world["z"], abs_tol=1e-9)
    assert second_payload["bev_points_smoothed"] is False


def test_camera_local_bev_drops_canonical_track_without_valid_world() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
                depth_source="depth_registered_m",
                canonical_world_required=True,
            )
        ],
        timestamp_us=1_000_000,
    )

    assert ws.messages[-1]["footpoints"] == []


def test_camera_local_bev_rejects_mismatched_canonical_world_revision(monkeypatch) -> None:
    monkeypatch.setenv("NOESIS_BEV_ALIGNMENT_DEBUG", "1")
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    base = _floor_camera_calibration()
    calib = SimpleNamespace(
        **base.__dict__,
        world_frame_id="backend_world_m",
        world_frame_revision="active-revision",
        frame_transform_sha256="a" * 64,
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=0.4,
                world_z=3.59,
                canonical_world_required=True,
                world_frame="backend_world_m",
                world_frame_revision="stale-revision",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    assert payload["footpoints"] == []
    assert payload["droppedFootpoints"][0]["reason"] == "canonical_world_revision_mismatch"


def test_camera_local_bev_prefers_depth_fused_world_when_it_disagrees_with_floor_contact_ray() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.array(  # type: ignore[method-assign]
        [
            [0.004, 0.0, -2.56],
            [0.0, 0.01, -3.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[
            Footpoint(
                u=740.0,
                v=760.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=1.25,
                world_z=2.5,
                anchor_source="pose_depth_fused",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "world_to_camera_local"
    assert math.isclose(point["x"], 1.25, abs_tol=1e-3)
    assert math.isclose(point["y"], 2.5, abs_tol=1e-3)


def test_camera_local_bev_prefers_registered_depth_when_it_agrees_with_floor_contact_ray() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "registered_depth_anchor"
    assert math.isclose(point["x"], 0.6, abs_tol=1e-3)
    assert math.isclose(point["y"], 6.0, abs_tol=1e-3)


def test_camera_local_bev_prefers_registered_depth_when_it_disagrees_with_floor_contact_ray() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=500.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "registered_depth_anchor"
    assert math.isclose(point["x"], 0.6, abs_tol=1e-3)
    assert math.isclose(point["y"], 6.0, abs_tol=1e-3)


def test_camera_local_bev_resets_smoothing_and_trails_on_display_source_switch() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": True,
            "draw_stride": 1,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
            "smooth_tau_s": 0.0,
            "max_points_per_track": 8,
        },
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 0.0,
            "max_jump_m": 0.10,
            "alpha": 1.0,
            "beta": 0.0,
        },
        frame="camera_local_ground_m",
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=700.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_000_000,
    )
    first = ws.messages[-1]["footpoints"][0]

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
            )
        ],
        timestamp_us=1_033_000,
    )
    second = ws.messages[-1]["footpoints"][0]
    second_payload = ws.messages[-1]

    assert first["displaySource"] == "floor_contact_ray"
    assert second["displaySource"] == "registered_depth_anchor"
    assert math.isclose(float(second["x"]), 0.6, abs_tol=1e-3)
    assert math.isclose(float(second["y"]), 6.0, abs_tol=1e-3)
    assert math.hypot(
        float(second["x"]) - float(first["x"]),
        float(second["y"]) - float(first["y"]),
    ) > 0.101
    assert second_payload["trails"] == []

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=742.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
            )
        ],
        timestamp_us=1_066_000,
    )
    resumed_payload = ws.messages[-1]
    assert len(resumed_payload["trails"]) == 1
    assert len(resumed_payload["trails"][0]["points"]) == 2
    assert math.isclose(
        resumed_payload["trails"][0]["points"][0]["x"],
        second["x"],
        abs_tol=1e-6,
    )


def test_camera_local_bev_waits_without_output_until_active_floorplan_exists() -> None:
    ws = _FakeWs()
    floorplan_active = {"enabled": False}

    def floorplan_bounds(_camera_id: str):
        if not floorplan_active["enabled"]:
            return None
        return _active_floorplan()

    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": True,
            "draw_stride": 1,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
            "smooth_tau_s": 0.0,
            "max_points_per_track": 8,
        },
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=floorplan_bounds,
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
            )
        ],
        timestamp_us=1_000_000,
    )
    assert ws.messages == []
    pending = renderer.health_snapshot("cam0")
    assert pending["healthy"] is True
    assert pending["floorplan_authority_pending_cameras"] == ["cam0"]
    assert pending["cameras"]["cam0"]["floorplan_authority_state"] == (
        "startup_pending"
    )

    floorplan_active["enabled"] = True
    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=742.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
            )
        ],
        timestamp_us=1_100_000,
    )
    first_active_point = ws.messages[-1]["footpoints"][0]
    assert renderer.health_snapshot("cam0")["floorplan_authority_ready_cameras"] == [
        "cam0"
    ]

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=744.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
            )
        ],
        timestamp_us=1_200_000,
    )

    payload = ws.messages[-1]
    assert payload["boundsSource"] == "active_floorplan"
    assert len(payload["trails"]) == 1
    assert len(payload["trails"][0]["points"]) == 2
    assert math.isclose(payload["trails"][0]["points"][0]["x"], first_active_point["x"], abs_tol=1e-6)


def test_camera_local_bev_uses_active_floorplan_bounds_for_floor_contact_candidates() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera_id: _active_floorplan(
            bounds={"min_x": -4.0, "max_x": 4.0, "min_z": 0.0, "max_z": 7.0},
            grid_shape=[70, 80],
        ),
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = True

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=500.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                debug={
                    "image_candidates": [
                        {"name": "image_base", "source": "test", "priority": 1, "u": 740.0, "v": 600.0}
                    ]
                },
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert payload["boundsSource"] == "active_floorplan"
    assert math.isclose(payload["xMin"], -4.0, abs_tol=1e-6)
    assert math.isclose(payload["zMax"], 7.0, abs_tol=1e-6)
    assert point["displaySource"] == "floor_contact_ray"
    assert math.isclose(point["x"], 0.666667, abs_tol=1e-3)
    assert math.isclose(point["y"], 6.666667, abs_tol=1e-3)


def test_camera_local_bev_publishes_floorplan_normalized_point_fields() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": True, "draw_stride": 1, "min_dt_s": 0.0, "min_step_px": 0.0, "smooth_tau_s": 0.0},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera_id: _active_floorplan(
            snapshot_ts_us=123,
            floorplan_ts_us=456,
        ),
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = True

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=640.0,
                v=600.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_000_000,
    )
    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=600.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_100_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]
    trail_point = payload["trails"][0]["points"][-1]

    assert payload["boundsSource"] == "active_floorplan"
    assert payload["floorplanCoordinateSpace"] == "floorplan_normalized_v1"
    assert payload["floorplanGridShape"] == [80, 80]
    assert math.isclose(payload["floorplanGridResM"], 0.1, abs_tol=1e-9)
    assert payload["floorplanSnapshotTsUs"] == 123
    assert payload["floorplanTsUs"] == 456
    assert point["floorplanInside"] is True
    assert math.isclose(point["floorplanX"], point["x"], abs_tol=1e-9)
    assert math.isclose(point["floorplanZ"], point["y"], abs_tol=1e-9)
    assert math.isclose(point["normX"], (point["x"] + 4.0) / 8.0, abs_tol=1e-6)
    assert math.isclose(point["normY"], 1.0 - (point["y"] / 8.0), abs_tol=1e-6)
    assert point["gridCol"] == int(math.floor(point["normX"] * 80.0))
    assert point["gridRow"] == int(math.floor(point["normY"] * 80.0))
    assert trail_point["floorplanInside"] is True
    assert math.isclose(trail_point["normX"], point["normX"], abs_tol=1e-6)
    assert math.isclose(trail_point["normY"], point["normY"], abs_tol=1e-6)


def test_camera_local_bev_reports_dropped_floor_contact_normalized_debug(monkeypatch) -> None:
    monkeypatch.setenv("NOESIS_BEV_ALIGNMENT_DEBUG", "1")
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera_id: _active_floorplan(
            bounds={"min_x": -4.0, "max_x": 4.0, "min_z": 0.0, "max_z": 7.0},
            grid_shape=[70, 80],
        ),
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = True

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=500.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    dropped = payload["droppedFootpoints"][0]

    assert payload["footpoints"] == []
    assert payload["alignmentDebug"]["droppedFootpointCount"] == 1
    assert dropped["reason"] == "floor_contact_outside_floorplan"
    assert dropped["floorplanInside"] is False
    assert dropped["gridCell"] == []
    assert dropped["gridRow"] is None
    assert dropped["gridCol"] is None
    assert dropped["normY"] < 0.0


def test_camera_local_bev_drops_floor_contact_when_active_floorplan_bounds_exclude_all_candidates() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera_id: _active_floorplan(
            bounds={"min_x": -4.0, "max_x": 4.0, "min_z": 0.0, "max_z": 7.0},
            grid_shape=[70, 80],
        ),
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = True

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=500.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    assert payload["boundsSource"] == "active_floorplan"
    assert payload["footpoints"] == []
    assert payload["trails"] == []


def test_camera_local_bev_prefers_floor_contact_ray_before_floorplan_depth() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        depth_sampler=lambda camera_id, u, v, ts: {"depth_m": 4.0, "support_count": 9},
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=500.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                world_x=25.0,
                world_z=25.0,
                anchor_source="pose_floor_only",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "floor_contact_ray"
    assert math.isclose(point["x"], 1.142857, abs_tol=1e-3)
    assert math.isclose(point["y"], 11.428571, abs_tol=1e-3)


def test_camera_local_bev_does_not_use_floorplan_depth_when_floor_ray_leaves_visible_footprint() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        depth_sampler=lambda camera_id, u, v, ts: {"depth_m": 3.0, "support_count": 9},
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    renderer.config_per_cam["cam0"].z_range = (0.0, 5.0)

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=500.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    assert payload["footpoints"] == []


def test_camera_local_bev_uses_registered_track_depth_before_any_depth_sampler() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        depth_sampler=lambda camera_id, u, v, ts: {"depth_m": 3.0, "support_count": 9},
    )
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _floor_camera_calibration(),
        footpoints=[
            Footpoint(
                u=740.0,
                v=626.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
                depth_m=6.0,
                anchor_source="pose_depth_fused",
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert point["displaySource"] == "registered_depth_anchor"
    assert math.isclose(point["x"], 0.6, abs_tol=1e-3)
    assert math.isclose(point["y"], 6.0, abs_tol=1e-3)


def test_world_bev_uses_canonical_backend_world_points_without_extra_lag() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 2.0,
            "max_jump_m": 0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "ttl_s": 10.0,
            "reset_after_s": 10.0,
        },
        frame="world",

    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    calib = _calibration()

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=0.0, world_z=0.0)],
        timestamp_us=1_000_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=10.0, world_z=0.0)],
        timestamp_us=1_100_000,
    )

    assert len(ws.messages) >= 2
    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert payload["frame_mode"] == "world"
    assert payload["trail_smoothing_owner"] == "backend"
    assert payload["bev_world_points_smoothed"] is False
    assert math.isclose(point["y"], 0.0, abs_tol=1e-6)
    assert math.isclose(point["x"], 10.0, abs_tol=1e-6)


def test_world_bev_preserves_scene_units_without_second_stage_smoothing() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 2.0,
            "max_jump_m": 0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "ttl_s": 10.0,
            "reset_after_s": 10.0,
        },
        frame="world",

    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    calib = _calibration(unit_scale=0.1)

    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=0.0, world_z=0.0)],
        timestamp_us=1_000_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=10.0, world_z=0.0)],
        timestamp_us=1_100_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert payload["trail_smoothing_owner"] == "backend"
    assert payload["bev_world_points_smoothed"] is False
    assert math.isclose(point["x"], 10.0, abs_tol=1e-6)


def test_world_bev_skips_anchor_hold_points() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 2.0,
            "max_jump_m": 0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "ttl_s": 10.0,
            "reset_after_s": 10.0,
        },
        frame="world",

    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    calib = _calibration()
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=1.0, world_z=2.0, anchor_source="anchor_hold")],
        timestamp_us=1_000_000,
    )

    assert len(ws.messages) == 1
    payload = ws.messages[-1]
    assert payload["trail_smoothing_owner"] == "backend"
    assert payload["bev_world_points_smoothed"] is False
    assert payload["footpoints"] == []
    assert payload["trails"] == []


def test_backend_trails_are_published_from_producer_history() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": True,
            "draw_stride": 1,
            "smooth_tau_s": 0.0,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
            "max_points_per_track": 8,
            "max_segments_per_track": 8,
        },
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 10.0,
            "max_jump_m": 10.0,
            "alpha": 1.0,
            "beta": 0.0,
            "ttl_s": 10.0,
            "reset_after_s": 10.0,
        },
        frame="world",

    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    calib = _calibration()
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=0.0, world_z=0.0)],
        timestamp_us=1_000_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=1.0, world_z=0.0)],
        timestamp_us=1_100_000,
    )

    payload = ws.messages[-1]
    assert isinstance(payload.get("trails"), list)
    assert len(payload["trails"]) == 1
    trail = payload["trails"][0]
    assert trail["stableId"] == 7
    assert trail["trackerId"] == 101
    assert len(trail["points"]) == 2
    assert math.isclose(trail["points"][-1]["x"], 1.0, abs_tol=1e-6)
    assert math.isclose(trail["points"][-1]["y"], 0.0, abs_tol=1e-6)


def test_backend_trails_do_not_apply_second_stage_lag_after_point_smoothing() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": True,
            "draw_stride": 1,
            "smooth_tau_s": 0.25,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
            "max_speed_px_per_s": 600.0,
            "max_points_per_track": 8,
            "max_segments_per_track": 8,
        },
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 100.0,
            "max_jump_m": 100.0,
            "alpha": 1.0,
            "beta": 0.0,
            "ttl_s": 10.0,
            "reset_after_s": 10.0,
        },
        frame="world",

    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    calib = _calibration()
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=0.0, world_z=0.0)],
        timestamp_us=1_000_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=1.0, world_z=0.0)],
        timestamp_us=1_100_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=1.0, world_z=0.0)],
        timestamp_us=1_200_000,
    )

    payload = ws.messages[-1]
    trail = payload["trails"][0]
    assert math.isclose(trail["points"][-1]["x"], 1.0, abs_tol=1e-6)


def test_world_bev_smoothing_resets_when_tracker_changes_under_same_stable_id() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={
            "enabled": True,
            "max_speed_mps": 2.0,
            "max_jump_m": 0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "ttl_s": 10.0,
            "reset_after_s": 10.0,
        },
        frame="world",

    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    calib = _calibration()
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=0.0, world_z=0.0)],
        timestamp_us=1_000_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=202, world_x=10.0, world_z=0.0)],
        timestamp_us=1_100_000,
    )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert math.isclose(point["x"], 10.0, abs_tol=1e-6)


def test_backend_trails_do_not_splice_history_across_tracker_remap() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": True,
            "draw_stride": 1,
            "smooth_tau_s": 0.0,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
            "max_points_per_track": 8,
            "max_segments_per_track": 8,
        },
        smoothing_cfg={
            "enabled": False,
        },
        frame="world",

    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    calib = _calibration()
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=0.0, world_z=0.0)],
        timestamp_us=1_000_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=101, world_x=0.5, world_z=0.0)],
        timestamp_us=1_100_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=202, world_x=10.0, world_z=0.0)],
        timestamp_us=1_200_000,
    )
    renderer.render_and_publish(
        "cam0",
        calib,
        footpoints=[Footpoint(u=640.0, v=360.0, stable_id=7, tracker_id=202, world_x=10.5, world_z=0.0)],
        timestamp_us=1_300_000,
    )

    payload = ws.messages[-1]
    trails = payload["trails"]

    assert len(trails) == 2
    by_tracker = {trail["trackerId"]: trail for trail in trails}
    assert set(by_tracker) == {101, 202}
    assert math.isclose(by_tracker[101]["points"][-1]["x"], 0.5, abs_tol=1e-6)
    assert math.isclose(by_tracker[202]["points"][-1]["x"], 10.5, abs_tol=1e-6)


def test_backend_trail_break_starts_new_segment_after_reacquisition() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": True,
            "draw_stride": 1,
            "smooth_tau_s": 0.0,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
            "max_points_per_track": 8,
            "max_segments_per_track": 8,
        },
        smoothing_cfg={"enabled": False},
        frame="world",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = _calibration()

    for timestamp_us, x, break_required in (
        (1_000_000, 0.0, False),
        (1_100_000, 0.5, False),
        (1_200_000, 10.0, True),
        (1_300_000, 10.5, False),
    ):
        renderer.render_and_publish(
            "cam0",
            calib,
            footpoints=[
                Footpoint(
                    u=640.0,
                    v=360.0,
                    stable_id=7,
                    tracker_id=101,
                    world_x=x,
                    world_z=0.0,
                    trail_break_required=break_required,
                )
            ],
            timestamp_us=timestamp_us,
        )

    trail = ws.messages[-1]["trails"][0]
    assert [point["x"] for point in trail["points"]] == pytest.approx([10.0, 10.5])


def test_backend_trail_segment_id_starts_new_segment_without_break_pulse() -> None:
    ws = _FakeWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={
            "enabled": True,
            "draw_stride": 1,
            "smooth_tau_s": 0.0,
            "min_dt_s": 0.0,
            "min_step_px": 0.0,
            "max_points_per_track": 8,
            "max_segments_per_track": 8,
        },
        smoothing_cfg={"enabled": False},
        frame="world",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.config_per_cam.get("cam0") or renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calib = _calibration()

    for timestamp_us, x, segment_id in (
        (1_000_000, 0.0, 0),
        (1_100_000, 0.5, 0),
        (1_200_000, 10.0, 1),
        (1_300_000, 10.5, 1),
    ):
        renderer.render_and_publish(
            "cam0",
            calib,
            footpoints=[
                Footpoint(
                    u=640.0,
                    v=360.0,
                    stable_id=7,
                    tracker_id=101,
                    world_x=x,
                    world_z=0.0,
                    trail_break_required=False,
                    trail_segment_id=segment_id,
                )
            ],
            timestamp_us=timestamp_us,
        )

    payload = ws.messages[-1]
    trail = payload["trails"][0]
    assert [point["x"] for point in trail["points"]] == pytest.approx([10.0, 10.5])
    assert payload["footpoints"][0]["trailSegmentId"] == 1


def test_camera_local_coverage_union_admits_foyer_beyond_floorplan_raster() -> None:
    ws = _FakeWs()
    renderer = _coverage_renderer(ws, trails=True)
    calib = _coverage_calibration()

    for timestamp_us, u, v in (
        (1_000_000, 420.0, 550.0),
        (1_100_000, 450.0, 600.0),
    ):
        renderer.render_and_publish(
            "cam0",
            calib,
            footpoints=[
                Footpoint(
                    u=u,
                    v=v,
                    method="image_foot",
                    stable_id=7,
                    tracker_id=101,
                )
            ],
            timestamp_us=timestamp_us,
        )

    payload = ws.messages[-1]
    point = payload["footpoints"][0]

    assert payload["boundsSource"] == "active_floorplan_plus_coverage_envelope"
    assert payload["floorplanBounds"] == {
        "min_x": 0.0,
        "max_x": 4.0,
        "min_z": 0.0,
        "max_z": 4.0,
    }
    assert payload["displayBounds"] == {
        "min_x": -0.25,
        "max_x": 6.25,
        "min_z": -0.25,
        "max_z": 8.25,
    }
    assert payload["coverageEnvelope"]["regions"][1]["id"] == "foyer"
    assert point["x"] == pytest.approx(4.5)
    assert point["y"] == pytest.approx(6.0)
    assert point["coverageInside"] is True
    assert point["coverageRegion"] == "foyer"
    assert point["floorplanInside"] is False
    assert point["normX"] > 1.0

    trail = payload["trails"][0]
    assert trail["points"][-1]["coverageInside"] is True
    assert trail["points"][-1]["coverageRegion"] == "foyer"
    assert trail["points"][-1]["floorplanInside"] is False


def test_camera_local_coverage_union_rejects_gap_inside_union_bbox() -> None:
    ws = _FakeWs()
    renderer = _coverage_renderer(ws)
    renderer._alignment_debug_enabled = True

    renderer.render_and_publish(
        "cam0",
        _coverage_calibration(),
        footpoints=[
            Footpoint(
                u=100.0,
                v=600.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_000_000,
    )

    payload = ws.messages[-1]
    assert payload["footpoints"] == []
    assert len(payload["droppedFootpoints"]) == 1
    dropped = payload["droppedFootpoints"][0]
    assert dropped["reason"] == "floor_contact_outside_coverage_envelope"
    assert dropped["coverageInside"] is False
    assert dropped["coverageRegion"] is None


def test_camera_local_coverage_boundary_tolerance_does_not_clamp_position() -> None:
    ws = _FakeWs()
    renderer = _coverage_renderer(ws)

    renderer.render_and_publish(
        "cam0",
        _coverage_calibration(),
        footpoints=[
            Footpoint(
                u=420.0,
                v=200.0,
                method="image_foot",
                stable_id=7,
                tracker_id=101,
            )
        ],
        timestamp_us=1_000_000,
    )

    point = ws.messages[-1]["footpoints"][0]
    assert point["coverageInside"] is True
    assert point["coverageRegion"] == "main"
    assert point["floorplanInside"] is False
    assert point["x"] == pytest.approx(4.2)
    assert point["y"] == pytest.approx(2.0)


def test_camera_local_coverage_config_fails_closed_when_polygon_is_invalid() -> None:
    ws = _FakeWs()
    invalid = _coverage_envelopes()
    invalid["cameras"]["cam0"]["regions"][0]["polygon_xz_m"] = [  # type: ignore[index]
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]

    with pytest.raises(ValueError, match="non-zero area"):
        BevRenderer(
            ws,
            frame="camera_local_ground_m",
            coverage_envelopes_cfg=invalid,
        )
