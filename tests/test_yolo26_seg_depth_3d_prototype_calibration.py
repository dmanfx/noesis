from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "testpipelines" / "yolo26-seg-depth-3d" / "prototype_calibration.py"
SPEC = importlib.util.spec_from_file_location("yolo26_seg_depth_3d_prototype_calibration", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prototype_calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prototype_calibration
SPEC.loader.exec_module(prototype_calibration)


def test_build_device_layer_reference_pose_exports_scene_and_world_variants(tmp_path: Path) -> None:
    virtual_devices_path = tmp_path / "virtual-devices.json"
    virtual_devices_path.write_text(
        json.dumps(
            [
                {
                    "id": "camera-1",
                    "name": "Living Room Camera",
                    "type": "Camera",
                    "position": {"x": 600.0, "y": 230.0, "z": 15.0},
                    "rotationY": 0.0,
                    "rotationX": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = prototype_calibration.build_device_layer_reference_pose(
        camera_id="living-room",
        display_name="Living Room Camera",
        alignment_info={
            "matrix_row_major": np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
            "floor_y": 0.0,
            "scene_per_m": 100.0,
            "s_obj_to_m": 0.01,
        },
        virtual_devices_path=virtual_devices_path,
    )

    assert payload["available"] is True
    assert payload["device_id"] == "camera-1"
    assert payload["device_name"] == "Living Room Camera"
    assert payload["temporary_offset_deg"] == {"yaw": 15.0, "pitch": 0.0, "roll": 5.0}
    assert payload["pose_scene"]["position"] == [600.0, 230.0, 15.0]
    assert np.allclose(payload["pose_world"]["position"], [6.0, 2.3, 0.15], atol=1e-6)
    assert np.allclose(payload["pose_scene"]["forward"], [np.sin(np.deg2rad(15.0)), 0.0, np.cos(np.deg2rad(15.0))], atol=1e-5)
    assert np.allclose(payload["pose_world"]["forward"], payload["pose_scene"]["forward"], atol=1e-5)


def test_build_frontend_reprojection_inputs_exports_canonical_defaults() -> None:
    snapshot = prototype_calibration.CalibrationSnapshot(
        camera_id="living-room",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        extrinsics_col_major=np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        floor_y=0.25,
        image_size=(1920, 1080),
        unit_scale=1.0,
    )

    payload = prototype_calibration.build_frontend_reprojection_inputs(
        camera_id="living-room",
        display_name="Living Room Camera",
        calibration_snapshot=snapshot,
        alignment_info={
            "matrix_row_major": np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
            "floor_y": 0.25,
            "scene_per_m": 100.0,
            "s_obj_to_m": 0.01,
        },
    )

    assert payload["available"] is True
    assert payload["tracking_projection_mode"] == "frontend_reproject"
    assert payload["pose_source"] == "calibration_bundle"
    assert payload["frame_conversion_requested"] == "analytic_bundle_ray"
    assert payload["frame_conversion_applied"] == "analytic_bundle_ray"
    assert payload["image_flip_override"] == {"u": False, "v": False}
    assert payload["camera_pitch_override_deg"] == 0.0
    assert payload["pose_world"]["position"] == [0.0, 0.0, 0.0]
    assert payload["pose_scene"]["position"] == [0.0, 0.0, 0.0]
    assert np.allclose(
        payload["local_camera_conversion_row_major"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    )


def test_build_menon_anchor_projection_exports_fixed_anchor_visual_basis(tmp_path: Path) -> None:
    virtual_devices_path = tmp_path / "virtual-devices.json"
    virtual_devices_path.write_text(
        json.dumps(
            [
                {
                    "id": "camera-1",
                    "name": "Kitchen Camera",
                    "type": "Camera",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotationY": 0.0,
                    "rotationX": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    snapshot = prototype_calibration.CalibrationSnapshot(
        camera_id="kitchen",
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics_col_major=np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        floor_y=0.0,
        image_size=(1920, 1080),
        unit_scale=1.0,
    )
    payload = prototype_calibration.build_menon_anchor_projection(
        camera_id="kitchen",
        display_name="Kitchen Camera",
        calibration_snapshot=snapshot,
        alignment_info={
            "matrix_row_major": np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
            "floor_y": 0.0,
            "scene_per_m": 92.0,
            "s_obj_to_m": 1.0 / 92.0,
        },
        virtual_devices_path=virtual_devices_path,
        scene_transform_info={
            "available": True,
            "source": "camera_device_similarity",
            "scene_per_m": 92.0,
            "camera_count": 3,
            "matrix_row_major": np.diag([92.0, 92.0, 92.0, 1.0]).astype(np.float32).reshape(-1).tolist(),
        }
    )

    assert payload["available"] is True
    assert payload["basis_name"] == "cv_xy_flip_to_device_visual"
    assert np.allclose(
        payload["basis_rotation_row_major"],
        [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    )
    assert payload["anchor_pose_source"] == "calibration_scene_pose"
    assert payload["device_reference_source"] == "device_layer_camera_object"
    assert payload["residual_angle_deg"] == 0.0
    assert payload["seed_scene_per_m"] == 92.0
    assert payload["seed_source"] == "camera_device_similarity"
    assert payload["camera_count"] == 3


def test_build_scene_registration_correction_exports_translation_delta(tmp_path: Path) -> None:
    virtual_devices_path = tmp_path / "virtual-devices.json"
    virtual_devices_path.write_text(
        json.dumps(
            [
                {
                    "id": "camera-1",
                    "name": "Kitchen Camera",
                    "type": "Camera",
                    "position": {"x": 140.0, "y": 200.0, "z": 310.0},
                    "rotationY": 0.0,
                    "rotationX": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    snapshot = prototype_calibration.CalibrationSnapshot(
        camera_id="kitchen",
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics_col_major=np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        floor_y=0.0,
        image_size=(1920, 1080),
        unit_scale=1.0,
    )

    payload = prototype_calibration.build_scene_registration_correction(
        camera_id="kitchen",
        display_name="Kitchen Camera",
        calibration_snapshot=snapshot,
        alignment_info={
            "matrix_row_major": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            "floor_y": 0.0,
            "scene_per_m": 100.0,
            "s_obj_to_m": 0.01,
        },
        virtual_devices_path=virtual_devices_path,
        scene_transform_info={
            "available": True,
            "source": "camera_device_similarity",
            "scene_per_m": 100.0,
            "camera_count": 3,
            "matrix_row_major": np.diag([100.0, 100.0, 100.0, 1.0]).astype(np.float32).reshape(-1).tolist(),
        },
    )

    assert payload["available"] is True
    assert payload["mode"] == "translation_scene_delta"
    assert np.allclose(payload["translation_scene_units"], [140.0, 200.0, 310.0], atol=1e-6)
    assert abs(float(payload["position_residual_scene_units"]) - np.linalg.norm([140.0, 200.0, 310.0])) < 1e-6


def test_resolver_catalog_entries_preserve_runtime_source_order(tmp_path: Path) -> None:
    cameras_path = tmp_path / "cameras.yaml"
    cameras_path.write_text(
        """
cameras:
  "0":
    name: living-room
  "1":
    name: kitchen
""".strip(),
        encoding="utf-8",
    )
    extrinsics_path = tmp_path / "camera_calibration.json"
    extrinsics_path.write_text(
        json.dumps(
            {
                "cameras": {
                    "living-room": {
                        "pose": {
                            "frame": "menon_scene",
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "yaw_pitch_roll_deg": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                        }
                    },
                    "kitchen": {
                        "pose": {
                            "frame": "menon_scene",
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "yaw_pitch_roll_deg": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    alignment_path = tmp_path / "ply_alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "align": {"matrix": np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(), "floor_y": 0.0, "units": {"s_obj_to_m": 1.0}},
                "meta": {"units": {"scene_per_m": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    resolver = prototype_calibration.PrototypeCalibrationResolver.from_source_configs(
        [
            {"sensor_id": "0", "sensor_name": "Living Room Camera", "runtime_source_id": 0},
            {"sensor_id": "1", "sensor_name": "Kitchen Camera", "runtime_source_id": 1},
        ],
        frame_size=(1920, 1080),
        cameras_path=cameras_path,
        extrinsics_path=extrinsics_path,
        alignment_path=alignment_path,
    )

    assert resolver.camera_labels() == {0: "living-room", 1: "kitchen"}
    assert [entry["camera_id"] for entry in resolver.catalog_entries()] == ["living-room", "kitchen"]


def test_resolver_sync_menon_scene_transform_persists_similarity(tmp_path: Path) -> None:
    cameras_path = tmp_path / "cameras.yaml"
    cameras_path.write_text(
        """
intrinsics_models:
  test_model:
    intrinsics:
      fx: 1000.0
      fy: 1000.0
      cx: 960.0
      cy: 540.0

cameras:
  "0":
    name: living-room
    model: test_model
  "1":
    name: kitchen
    model: test_model
""".strip(),
        encoding="utf-8",
    )
    extrinsics_path = tmp_path / "camera_calibration.json"
    extrinsics_path.write_text(
        json.dumps(
            {
                "cameras": {
                    "living-room": {
                        "pose": {
                            "position": [1.0, 2.0, 3.0],
                            "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
                            "rotation_order": "YXZ",
                            "frame": "backend_world_m",
                        }
                    },
                    "kitchen": {
                        "pose": {
                            "position": [4.0, 2.0, 3.0],
                            "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
                            "rotation_order": "YXZ",
                            "frame": "backend_world_m",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    alignment_path = tmp_path / "ply_alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "matrix": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                "floor_y": 0.0,
                "units": {"s_obj_to_m": 1.0},
            }
        ),
        encoding="utf-8",
    )
    virtual_devices_path = tmp_path / "virtual-devices.json"
    virtual_devices_path.write_text(
        json.dumps(
            [
                {
                    "id": "cam-lr",
                    "name": "Living Room Camera",
                    "type": "Camera",
                    "position": {"x": 100.0, "y": 200.0, "z": 300.0},
                    "rotationY": 0.0,
                    "rotationX": 0.0,
                },
                {
                    "id": "cam-kit",
                    "name": "Kitchen Camera",
                    "type": "Camera",
                    "position": {"x": 400.0, "y": 200.0, "z": 300.0},
                    "rotationY": 0.0,
                    "rotationX": 0.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    resolver = prototype_calibration.PrototypeCalibrationResolver.from_source_configs(
        [
            {"sensor_id": "0", "sensor_name": "Living Room Camera", "runtime_source_id": 0},
            {"sensor_id": "1", "sensor_name": "Kitchen Camera", "runtime_source_id": 1},
        ],
        frame_size=(1920, 1080),
        cameras_path=cameras_path,
        extrinsics_path=extrinsics_path,
        alignment_path=alignment_path,
    )
    resolver = prototype_calibration.PrototypeCalibrationResolver(
        bindings=resolver.bindings,
        cameras_path=cameras_path,
        extrinsics_path=extrinsics_path,
        alignment_path=alignment_path,
        menon_virtual_devices_path=virtual_devices_path,
        frame_size=(1920, 1080),
    )

    synced = resolver.sync_menon_scene_transform(persist=True)
    assert synced["available"] is True
    assert synced["persisted"] is True
    assert synced["camera_count"] == 2
    assert np.allclose(synced["matrix_row_major"], np.diag([100.0, 100.0, 100.0, 1.0]).reshape(-1), atol=1e-5)

    raw_align = json.loads(alignment_path.read_text(encoding="utf-8"))
    assert raw_align["scene_similarity"]["camera_count"] == 2
    assert raw_align["scene_similarity"]["source"] == "camera_device_similarity"
