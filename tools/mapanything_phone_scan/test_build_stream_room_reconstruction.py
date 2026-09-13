from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from noesis.calibration.manager import CalibrationSnapshot
from noesis.virtual_twin.store import VirtualTwinStore
from scripts.build_stream_room_reconstruction import (
    CameraCalibration,
    DepthSnapshot,
    RgbFrame,
    _build_frame_clouds,
    _write_revision,
)


def test_live_rgb_provenance_survives_normal_revision_writer(tmp_path: Path) -> None:
    dewarper = tmp_path / "camera_dewarper.ini"
    dewarper.write_text(
        """[property]
output-width=100
output-height=80

[surface0]
width=100
height=80
focal-length=50;50
dst-focal-length=40;40
src-x0=49.5
src-y0=39.5
dst-principal-point=49.5;39.5
distortion=0.01;0.0;0.0;0.0
""",
        encoding="utf-8",
    )
    depth = np.full((40, 50), 2.0, dtype=np.float32)
    snapshot = DepthSnapshot(
        path=tmp_path / "depth_0000.zarr",
        timestamp_us=1,
        depth=depth,
        confidence=np.ones_like(depth),
        mask=np.ones_like(depth, dtype=bool),
    )
    intrinsics = np.asarray(
        [[40.0, 0.0, 24.5], [0.0, 40.0, 19.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    identity = np.eye(4, dtype=np.float64)
    calibration_snapshot = CalibrationSnapshot(
        camera_id="living-room",
        intrinsics=intrinsics,
        extrinsics_col_major=tuple(identity.flatten(order="F")),
        floor_y=0.0,
        image_size=(50, 40),
    )
    camera = CameraCalibration(
        source_id=0,
        camera_id="living-room",
        snapshot=calibration_snapshot,
        world_to_camera=identity,
        camera_to_world=identity,
        intrinsics=intrinsics,
        floor_y=0.0,
    )
    rgb = RgbFrame(
        source_index=37,
        image_bgr=np.full((80, 100, 3), 180, dtype=np.uint8),
        source_uri="pipeline://living-room",
        dewarper_config=str(dewarper),
        transformed=True,
    )
    frames = _build_frame_clouds(
        camera=camera,
        snapshots=(snapshot,),
        rgb_frames=(rgb,),
        min_confidence=0.0,
        pixel_step=1,
        depth_clip_percentile=99.0,
        max_depth_m=10.0,
    )
    assert frames[0].rgb_frame is rgb
    result = _write_revision(
        store=VirtualTwinStore(tmp_path / "virtual_twin"),
        revision_id="producer-r1",
        camera=camera,
        frames=frames,
        points_budget=100_000,
        min_confidence=0.0,
        mesh_pixel_step=4,
        mesh_max_edge_m=1.0,
        mesh_max_depth_delta_m=1.0,
        ceiling_clip={},
        observed_floor_y=0.0,
        floor_estimate={},
        floor_alignment={"status": "disabled"},
        world_correction=None,
        scene_similarity={},
        floorplan_footprint=None,
        update_latest=False,
    )
    metadata = json.loads(
        (Path(result["revision_dir"]) / "room_points_meta.json").read_text(
            encoding="utf-8"
        )
    )
    contract = metadata["static_camera_image"]
    assert contract["source_uri"] == "pipeline://living-room"
    assert contract["source_index"] == 37
    assert contract["image_size"] == [50, 40]
    np.testing.assert_allclose(
        np.asarray(contract["intrinsics"]),
        [[20.0, 0.0, 24.75], [0.0, 20.0, 19.75], [0.0, 0.0, 1.0]],
        atol=1e-8,
    )
    assert contract["rectification"]["status"] == "rectified"
    assert contract["rectification"]["config_sha256"]
