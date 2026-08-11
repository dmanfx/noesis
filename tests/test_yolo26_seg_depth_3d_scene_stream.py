from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "testpipelines" / "yolo26-seg-depth-3d" / "scene_stream.py"
SPEC = importlib.util.spec_from_file_location("yolo26_seg_depth_3d_scene_stream", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
scene_stream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scene_stream
SPEC.loader.exec_module(scene_stream)


@dataclass(frozen=True)
class _Snapshot:
    camera_id: str
    intrinsics: np.ndarray
    extrinsics_col_major: list[float]
    floor_y: float
    image_size: tuple[int, int]
    unit_scale: float = 1.0


@dataclass(frozen=True)
class _DepthFrame:
    depth_map: np.ndarray
    valid_mask: np.ndarray
    pts_us: int


@dataclass(frozen=True)
class _FrameMeta:
    frame_number: int
    source_id: int


def test_scene_packet_builder_exports_structured_world_grid() -> None:
    builder = scene_stream.ScenePacketBuilder(
        grid_width=32,
        anchor_basis_row_major=[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    )
    snapshot = _Snapshot(
        camera_id="living-room",
        intrinsics=np.array(
            [
                [100.0, 0.0, 1.5],
                [0.0, 100.0, 1.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        extrinsics_col_major=np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        floor_y=0.0,
        image_size=(4, 4),
    )
    depth_frame = _DepthFrame(
        depth_map=np.full((4, 4), 2.0, dtype=np.float32),
        valid_mask=np.ones((4, 4), dtype=bool),
        pts_us=123456,
    )
    rgb = np.zeros((4, 4, 4), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    rgb[..., 3] = 255

    packet = builder.build_frame_packet(
        frame_rgb=rgb,
        depth_frame=depth_frame,
        calibration_snapshot=snapshot,
        frame_meta=_FrameMeta(frame_number=7, source_id=0),
        detections=[{"label": "person", "world_point": [0.0, 0.0, 2.0]}],
    )

    assert packet["type"] == "scene_frame"
    assert packet["camera_id"] == "living-room"
    assert packet["frame_id"] == 7
    assert packet["ts_us"] == 123456
    assert packet["frame_mode"] == "backend_world_m"
    assert packet["coord_space"] == "backend_world_m"
    assert packet["units"] == "meters"
    assert packet["grid_width"] == 32
    assert packet["grid_height"] == 32
    assert packet["detections"] == [{"label": "person", "world_point": [0.0, 0.0, 2.0]}]

    positions = np.frombuffer(
        scene_stream.base64.b64decode(packet["positions_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((32, 32, 3))
    camera_local = np.frombuffer(
        scene_stream.base64.b64decode(packet["camera_local_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((32, 32, 3))
    anchor_local = np.frombuffer(
        scene_stream.base64.b64decode(packet["anchor_local_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((32, 32, 3))
    colors = np.frombuffer(
        scene_stream.base64.b64decode(packet["colors_rgb_u8_b64"]),
        dtype=np.uint8,
    ).reshape((32, 32, 3))
    valid = np.frombuffer(
        scene_stream.base64.b64decode(packet["valid_u8_b64"]),
        dtype=np.uint8,
    ).reshape((32, 32))

    assert np.all(valid == 1)
    assert np.all(colors[..., 0] == 10)
    assert np.all(colors[..., 1] == 20)
    assert np.all(colors[..., 2] == 30)
    assert np.allclose(positions[..., 2], 2.0)
    assert np.allclose(camera_local[..., 2], 2.0)
    assert np.allclose(anchor_local[..., 2], 2.0)
    assert np.allclose(anchor_local[..., 0], -camera_local[..., 0])
    assert np.allclose(anchor_local[..., 1], -camera_local[..., 1])


def test_scene_packet_builder_uses_per_camera_anchor_basis_mapping() -> None:
    builder = scene_stream.ScenePacketBuilder(
        grid_width=32,
        anchor_basis_by_camera_id={
            "family-room": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        },
    )
    snapshot = _Snapshot(
        camera_id="family-room",
        intrinsics=np.array(
            [
                [100.0, 0.0, 1.5],
                [0.0, 100.0, 1.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        extrinsics_col_major=np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        floor_y=0.0,
        image_size=(4, 4),
    )
    depth_frame = _DepthFrame(
        depth_map=np.full((4, 4), 2.0, dtype=np.float32),
        valid_mask=np.ones((4, 4), dtype=bool),
        pts_us=1,
    )
    packet = builder.build_frame_packet(
        frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_frame=depth_frame,
        calibration_snapshot=snapshot,
        frame_meta=_FrameMeta(frame_number=1, source_id=0),
        detections=[],
    )
    camera_local = np.frombuffer(
        scene_stream.base64.b64decode(packet["camera_local_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((32, 32, 3))
    anchor_local = np.frombuffer(
        scene_stream.base64.b64decode(packet["anchor_local_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((32, 32, 3))
    assert np.allclose(anchor_local[..., 0], camera_local[..., 0])
    assert np.allclose(anchor_local[..., 1], -camera_local[..., 2])
    assert np.allclose(anchor_local[..., 2], camera_local[..., 1])


def test_scene_packet_builder_applies_depth_registration_to_exported_points() -> None:
    class _RegistrationManager:
        @staticmethod
        def apply(*, camera_id: str, raw_depth_m: float):
            assert camera_id == "living-room"
            return raw_depth_m * 0.5, "ok", "cam0:abc123"

        pass

    class _Entry:
        registration_id = "cam0:abc123"

    class _Bundle:
        entries = {"living-room": _Entry()}

    _RegistrationManager.bundle = _Bundle()

    builder = scene_stream.ScenePacketBuilder(
        grid_width=32,
        depth_registration_manager=_RegistrationManager(),
    )
    snapshot = _Snapshot(
        camera_id="living-room",
        intrinsics=np.array(
            [
                [100.0, 0.0, 1.5],
                [0.0, 100.0, 1.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        extrinsics_col_major=np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        floor_y=0.0,
        image_size=(4, 4),
    )
    depth_frame = _DepthFrame(
        depth_map=np.full((4, 4), 2.0, dtype=np.float32),
        valid_mask=np.ones((4, 4), dtype=bool),
        pts_us=99,
    )

    packet = builder.build_frame_packet(
        frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_frame=depth_frame,
        calibration_snapshot=snapshot,
        frame_meta=_FrameMeta(frame_number=1, source_id=0),
        detections=[],
    )

    positions = np.frombuffer(
        scene_stream.base64.b64decode(packet["positions_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((32, 32, 3))
    assert packet["depth_registration_id"] == "cam0:abc123"
    assert np.allclose(positions[..., 2], 1.0)


def test_build_scene_config_includes_camera_pose() -> None:
    snapshot = _Snapshot(
        camera_id="living-room",
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics_col_major=np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        floor_y=0.25,
        image_size=(1920, 1080),
        unit_scale=1.0,
    )
    payload = scene_stream.build_scene_config(
        camera_id="living-room",
        display_name="Living Room Camera",
        source_id=0,
        frame_size=(1920, 1080),
        grid_size=(160, 90),
        calibration_snapshot=snapshot,
        alignment_info={
            "matrix_row_major": [1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 1.0],
            "floor_y": 12.0,
            "scene_per_m": 100.0,
            "s_obj_to_m": 0.01,
        },
        menon_scene_transform={
            "available": True,
            "source": "camera_device_similarity",
            "scene_per_m": 90.0,
            "camera_count": 3,
            "matrix_row_major": [90.0, 0.0, 0.0, 2.0, 0.0, 90.0, 0.0, 3.0, 0.0, 0.0, 90.0, 4.0, 0.0, 0.0, 0.0, 1.0],
            "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "position_rmse_scene_units": 1.25,
        },
        menon_anchor_projection={
            "available": True,
            "basis_name": "cv_y_flip_to_device_visual",
            "basis_rotation_row_major": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            "seed_scene_per_m": 90.0,
            "seed_source": "camera_device_similarity",
            "camera_count": 3,
        },
        frontend_reprojection={
            "available": True,
            "pose_source": "calibration_bundle",
            "frame_conversion_applied": "analytic_bundle_ray",
            "image_flip_override": {"u": False, "v": False},
            "camera_pitch_override_deg": 0.0,
            "local_camera_conversion_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "pose_world": {
                "position": [1.0, 2.0, 3.0],
                "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "forward": [0.0, 0.0, 1.0],
            },
            "pose_scene": {
                "position": [100.0, 200.0, 300.0],
                "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "forward": [0.0, 0.0, 1.0],
            },
        },
        device_layer_reference={
            "available": True,
            "device_id": "dev-1",
            "device_name": "Living Room Camera",
            "pose_world": {
                "position": [1.0, 2.0, 3.0],
                "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "forward": [0.0, 0.0, 1.0],
            },
            "pose_scene": {
                "position": [100.0, 200.0, 300.0],
                "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "forward": [0.0, 0.0, 1.0],
            },
        },
        scene_registration_correction={
            "available": True,
            "mode": "translation_scene_delta",
            "source": "calibration_scene_to_device_scene_translation",
            "translation_scene_units": [10.0, 20.0, 30.0],
            "position_residual_scene_units": 37.5,
        },
        target_fps=5.0,
    )
    assert payload["type"] == "scene_config"
    assert payload["camera_id"] == "living-room"
    assert payload["display_name"] == "Living Room Camera"
    assert payload["source_id"] == 0
    assert payload["world_frame"] == "backend_world_m"
    assert payload["coord_space"] == "backend_world_m"
    assert payload["units"] == "meters"
    assert payload["grid_size"] == [160, 90]
    calibration = payload["calibration"]
    assert calibration["floor_y"] == 0.25
    assert calibration["camera_pose_world"]["position"] == [0.0, 0.0, 0.0]
    assert calibration["camera_pose_scene"]["position"] == [2.0, 3.0, 4.0]
    assert payload["alignment"]["floor_y"] == 12.0
    assert payload["alignment"]["scene_per_m"] == 100.0
    assert payload["alignment"]["s_obj_to_m"] == 0.01
    assert payload["menon_scene_transform"]["scene_per_m"] == 90.0
    assert payload["menon_anchor_projection"]["basis_name"] == "cv_y_flip_to_device_visual"
    assert payload["coordinate_systems"]["camera_local"]["bev_ground_frame"] == "camera_local_ground_m"
    assert payload["coordinate_systems"]["menon_anchor_local"]["basis_name"] == "cv_y_flip_to_device_visual"
    assert payload["frontend_reprojection"]["frame_conversion_applied"] == "analytic_bundle_ray"
    assert payload["frontend_reprojection"]["image_flip_override"] == {"u": False, "v": False}
    assert payload["device_layer_reference"]["device_id"] == "dev-1"
    assert payload["device_layer_reference"]["pose_scene"]["position"] == [100.0, 200.0, 300.0]
    assert payload["scene_registration_correction"]["mode"] == "translation_scene_delta"
    assert payload["scene_registration_correction"]["translation_scene_units"] == [10.0, 20.0, 30.0]


def test_build_scene_catalog_sorts_camera_entries_by_source() -> None:
    payload = scene_stream.build_scene_catalog(
        [
            {"camera_id": "family-room", "display_name": "Family Room Camera", "runtime_source_id": 2, "calibration_source_id": 2},
            {"camera_id": "living-room", "display_name": "Living Room Camera", "runtime_source_id": 0, "calibration_source_id": 0},
            {"camera_id": "kitchen", "display_name": "Kitchen Camera", "runtime_source_id": 1, "calibration_source_id": 1},
        ]
    )

    assert payload["type"] == "scene_catalog"
    assert [entry["camera_id"] for entry in payload["cameras"]] == ["living-room", "kitchen", "family-room"]


def test_build_scene_fusion_catalog_entry_appends_synthetic_camera() -> None:
    payload = scene_stream.build_scene_fusion_catalog_entry(
        [
            {"camera_id": "living-room", "runtime_source_id": 0},
            {"camera_id": "kitchen", "runtime_source_id": 3},
        ]
    )

    assert payload["camera_id"] == scene_stream.SCENE_FUSION_CAMERA_ID
    assert payload["display_name"] == scene_stream.SCENE_FUSION_DISPLAY_NAME
    assert payload["source_id"] == 4
    assert payload["calibration_source_id"] == -1


def test_build_scene_fusion_config_uses_scene_space_contract() -> None:
    payload = scene_stream.build_scene_fusion_config(
        source_id=7,
        frame_size=(1920, 1080),
        grid_size=(160, 181),
        alignment_info={
            "matrix_row_major": [1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 1.0],
            "floor_y": 12.0,
            "scene_per_m": 100.0,
            "s_obj_to_m": 0.01,
        },
        menon_scene_transform={
            "available": True,
            "source": "camera_device_similarity",
            "scene_per_m": 100.0,
            "matrix_row_major": [100.0, 0.0, 0.0, 2.0, 0.0, 100.0, 0.0, 3.0, 0.0, 0.0, 100.0, 4.0, 0.0, 0.0, 0.0, 1.0],
        },
        target_fps=4.0,
        input_camera_ids=["living-room", "kitchen"],
    )

    assert payload["camera_id"] == scene_stream.SCENE_FUSION_CAMERA_ID
    assert payload["source_id"] == 7
    assert payload["coord_space"] == "menon_scene"
    assert payload["world_frame"] == "menon_scene"
    assert payload["units"] == "scene_units"
    assert payload["fusion"]["camera_ids"] == ["living-room", "kitchen"]
    assert payload["fusion"]["separator_rows"] == scene_stream.SCENE_FUSION_SEPARATOR_ROWS


def test_build_scene_fusion_frame_transforms_backend_world_packets_into_scene_space() -> None:
    transform = [10.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 10.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    frame_a_positions = np.asarray([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)
    frame_b_positions = np.asarray([[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)
    colors = np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    valid = np.asarray([[1, 1]], dtype=np.uint8)
    payload = scene_stream.build_scene_fusion_frame(
        configs_by_camera={
            "cam-a": {"camera_id": "cam-a", "source_id": 0, "menon_scene_transform": {"available": True, "matrix_row_major": transform}},
            "cam-b": {"camera_id": "cam-b", "source_id": 1, "menon_scene_transform": {"available": True, "matrix_row_major": transform}},
            scene_stream.SCENE_FUSION_CAMERA_ID: {"camera_id": scene_stream.SCENE_FUSION_CAMERA_ID, "source_id": 9},
        },
        frames_by_camera={
            "cam-a": {
                "type": "scene_frame",
                "camera_id": "cam-a",
                "source_id": 0,
                "frame_id": 11,
                "ts_us": 100,
                "frame_mode": "backend_world_m",
                "world_frame": "backend_world_m",
                "coord_space": "backend_world_m",
                "grid_width": 2,
                "grid_height": 1,
                "positions_xyz_f32_b64": scene_stream._b64_array(frame_a_positions),
                "colors_rgb_u8_b64": scene_stream._b64_array(colors),
                "valid_u8_b64": scene_stream._b64_array(valid),
                "detections": [],
            },
            "cam-b": {
                "type": "scene_frame",
                "camera_id": "cam-b",
                "source_id": 1,
                "frame_id": 12,
                "ts_us": 120,
                "frame_mode": "backend_world_m",
                "world_frame": "backend_world_m",
                "coord_space": "backend_world_m",
                "grid_width": 2,
                "grid_height": 1,
                "positions_xyz_f32_b64": scene_stream._b64_array(frame_b_positions),
                "colors_rgb_u8_b64": scene_stream._b64_array(colors),
                "valid_u8_b64": scene_stream._b64_array(valid),
                "detections": [],
            },
        },
        fusion_source_id=9,
        preferred_camera_ids=["cam-a", "cam-b"],
    )

    assert payload is not None
    assert payload["camera_id"] == scene_stream.SCENE_FUSION_CAMERA_ID
    assert payload["coord_space"] == "menon_scene"
    assert payload["source_id"] == 9
    assert payload["merged_camera_ids"] == ["cam-a", "cam-b"]
    assert payload["grid_width"] == 2
    assert payload["grid_height"] == 3
    positions = np.frombuffer(
        scene_stream.base64.b64decode(payload["positions_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((3, 2, 3))
    valid_out = np.frombuffer(
        scene_stream.base64.b64decode(payload["valid_u8_b64"]),
        dtype=np.uint8,
    ).reshape((3, 2))

    assert np.allclose(positions[0, 0], [11.0, 2.0, 3.0])
    assert np.allclose(positions[0, 1], [21.0, 2.0, 3.0])
    assert np.all(valid_out[1] == 0)
    assert np.allclose(positions[2, 0], [1.0, 12.0, 3.0])
    assert np.allclose(positions[2, 1], [1.0, 22.0, 3.0])


def test_build_scene_fusion_frame_applies_per_camera_scene_translation_correction() -> None:
    transform = [10.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 10.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    frame_positions = np.asarray([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)
    colors = np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    valid = np.asarray([[1, 1]], dtype=np.uint8)
    payload = scene_stream.build_scene_fusion_frame(
        configs_by_camera={
            "cam-a": {
                "camera_id": "cam-a",
                "source_id": 0,
                "menon_scene_transform": {"available": True, "matrix_row_major": transform},
                "scene_registration_correction": {
                    "available": True,
                    "mode": "translation_scene_delta",
                    "translation_scene_units": [5.0, -1.0, 2.0],
                },
            },
            scene_stream.SCENE_FUSION_CAMERA_ID: {"camera_id": scene_stream.SCENE_FUSION_CAMERA_ID, "source_id": 9},
        },
        frames_by_camera={
            "cam-a": {
                "type": "scene_frame",
                "camera_id": "cam-a",
                "source_id": 0,
                "frame_id": 11,
                "ts_us": 100,
                "frame_mode": "backend_world_m",
                "world_frame": "backend_world_m",
                "coord_space": "backend_world_m",
                "grid_width": 2,
                "grid_height": 1,
                "positions_xyz_f32_b64": scene_stream._b64_array(frame_positions),
                "colors_rgb_u8_b64": scene_stream._b64_array(colors),
                "valid_u8_b64": scene_stream._b64_array(valid),
                "detections": [],
            },
        },
        fusion_source_id=9,
        preferred_camera_ids=["cam-a"],
    )

    assert payload is not None
    positions = np.frombuffer(
        scene_stream.base64.b64decode(payload["positions_xyz_f32_b64"]),
        dtype=np.float32,
    ).reshape((1, 2, 3))
    assert np.allclose(positions[0, 0], [16.0, 1.0, 5.0])
    assert np.allclose(positions[0, 1], [26.0, 1.0, 5.0])


def test_scene_viewer_server_coalesces_pending_messages_per_camera(tmp_path: Path) -> None:
    static_dir = tmp_path / "viewer"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html></html>")

    server = scene_stream.SceneViewerServer(
        host="127.0.0.1",
        port=0,
        static_dir=static_dir,
        log_level="warning",
    )

    server.publish_catalog({"type": "scene_catalog", "cameras": [{"camera_id": "living-room", "source_id": 0}]})
    server.publish_config({"type": "scene_config", "camera_id": "living-room", "source_id": 0, "value": 1})
    server.publish_config({"type": "scene_config", "camera_id": "living-room", "source_id": 0, "value": 2})
    server.publish_frame({"type": "scene_frame", "camera_id": "living-room", "source_id": 0, "frame_id": 10})
    server.publish_frame({"type": "scene_frame", "camera_id": "living-room", "source_id": 0, "frame_id": 11})
    server.publish_stats({"type": "scene_stats", "camera_id": "living-room", "source_id": 0, "published_frames": 3})
    server.publish_stats({"type": "scene_stats", "camera_id": "living-room", "source_id": 0, "published_frames": 4})

    drained = server._drain_pending_for_broadcast()
    assert [item["type"] for item in drained] == ["scene_catalog", "scene_config", "scene_stats", "scene_frame"]
    assert next(item for item in drained if item["type"] == "scene_config")["value"] == 2
    assert next(item for item in drained if item["type"] == "scene_stats")["published_frames"] == 4
    assert next(item for item in drained if item["type"] == "scene_frame")["frame_id"] == 11

    assert server._drain_pending_for_broadcast() == []
    assert server.has_frame_for_camera("living-room") is True
    assert server.has_frame_for_camera("kitchen") is False


def test_scene_viewer_server_publishes_latest_scene_fusion_frame(tmp_path: Path) -> None:
    static_dir = tmp_path / "viewer"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html></html>")

    server = scene_stream.SceneViewerServer(
        host="127.0.0.1",
        port=0,
        static_dir=static_dir,
        log_level="warning",
    )
    transform = [10.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 10.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    colors = np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    valid = np.asarray([[1, 1]], dtype=np.uint8)

    server.publish_config({"type": "scene_config", "camera_id": "cam-a", "source_id": 0, "menon_scene_transform": {"available": True, "matrix_row_major": transform}})
    server.publish_config({"type": "scene_config", "camera_id": "cam-b", "source_id": 1, "menon_scene_transform": {"available": True, "matrix_row_major": transform}})
    server.publish_config(
        scene_stream.build_scene_fusion_config(
            source_id=9,
            frame_size=(1920, 1080),
            grid_size=(2, 3),
            alignment_info=None,
            menon_scene_transform={"available": True, "matrix_row_major": transform},
            target_fps=4.0,
            input_camera_ids=["cam-a", "cam-b"],
        )
    )
    server.publish_frame(
        {
            "type": "scene_frame",
            "camera_id": "cam-a",
            "source_id": 0,
            "frame_id": 11,
            "ts_us": 100,
            "frame_mode": "backend_world_m",
            "world_frame": "backend_world_m",
            "coord_space": "backend_world_m",
            "grid_width": 2,
            "grid_height": 1,
            "positions_xyz_f32_b64": scene_stream._b64_array(np.asarray([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)),
            "colors_rgb_u8_b64": scene_stream._b64_array(colors),
            "valid_u8_b64": scene_stream._b64_array(valid),
            "detections": [],
        }
    )
    server.publish_frame(
        {
            "type": "scene_frame",
            "camera_id": "cam-b",
            "source_id": 1,
            "frame_id": 12,
            "ts_us": 120,
            "frame_mode": "backend_world_m",
            "world_frame": "backend_world_m",
            "coord_space": "backend_world_m",
            "grid_width": 2,
            "grid_height": 1,
            "positions_xyz_f32_b64": scene_stream._b64_array(np.asarray([[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)),
            "colors_rgb_u8_b64": scene_stream._b64_array(colors),
            "valid_u8_b64": scene_stream._b64_array(valid),
            "detections": [],
        }
    )

    drained = server._drain_pending_for_broadcast()
    fusion_frame = next(
        item
        for item in drained
        if item["type"] == "scene_frame" and item["camera_id"] == scene_stream.SCENE_FUSION_CAMERA_ID
    )
    assert fusion_frame["merged_camera_count"] == 2
    assert fusion_frame["source_id"] == 9
