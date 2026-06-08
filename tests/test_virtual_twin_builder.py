from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import zarr

from noesis.calibration.manager import CalibrationSnapshot
from noesis.virtual_twin.builder import VirtualTwinFrameInput, build_virtual_twin_revision, rgb_colorfulness
from noesis.virtual_twin.geometry import PlaneCandidate
from noesis.virtual_twin.store import VirtualTwinStore
from scripts.build_virtual_twin_reconstruction import (
    _latest_mapanything_snapshot,
    _resize_depth_snapshot,
    _valid_depth_coverage,
)


def _read_glb_json(path: Path) -> dict:
    data = Path(path).read_bytes()
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    assert version == 2
    json_len, chunk_type = struct.unpack_from("<I4s", data, 12)
    assert chunk_type == b"JSON"
    return json.loads(data[20 : 20 + json_len].decode("utf-8").rstrip(" "))


def _write_wall_obj(path: Path, centers: list[np.ndarray]) -> None:
    lines: list[str] = []
    vertex_index = 1
    for idx, center in enumerate(centers):
        x, y, z = [float(v) for v in center]
        half = 0.75
        verts = [
            (x - half, y - half, z),
            (x + half, y - half, z),
            (x + half, y + half, z),
            (x - half, y + half, z),
        ]
        lines.append(f"o wall_{idx}")
        lines.append("usemtl wall")
        for vert in verts:
            lines.append(f"v {vert[0]} {vert[1]} {vert[2]}")
        lines.append(f"f {vertex_index} {vertex_index + 1} {vertex_index + 2} {vertex_index + 3}")
        vertex_index += 4
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_depth_zarr(path: Path, *, timestamp_us: int, depth_value: float = 2.5) -> None:
    root = zarr.open_group(str(path), mode="w")
    depth = np.full((3, 4), depth_value, dtype=np.float32)
    conf = np.full((3, 4), 0.9, dtype=np.float32)
    mask = np.ones((3, 4), dtype=np.uint8)
    root.create_dataset("depth_z", shape=depth.shape, chunks=depth.shape, data=depth)
    root.create_dataset("conf", shape=conf.shape, chunks=conf.shape, data=conf)
    root.create_dataset("mask", shape=mask.shape, chunks=mask.shape, data=mask)
    root.attrs.update(camera_id="living-room", timestamp_us=int(timestamp_us))


def test_pipeline_depth_snapshot_loader_uses_latest_persisted_zarr(tmp_path: Path) -> None:
    older = tmp_path / "data" / "depth" / "living-room" / "20260502" / "17" / "100.zarr"
    newer = tmp_path / "data" / "depth" / "living-room" / "20260502" / "17" / "200.zarr"
    older.parent.mkdir(parents=True)
    _write_depth_zarr(older, timestamp_us=100, depth_value=1.0)
    _write_depth_zarr(newer, timestamp_us=200, depth_value=2.0)

    snapshot = _latest_mapanything_snapshot(
        depth_base=tmp_path / "data" / "depth",
        camera_id="living-room",
        min_ts_us=100,
        min_confidence=0.5,
    )

    assert snapshot is not None
    assert snapshot.path == newer
    assert snapshot.timestamp_us == 200
    assert _valid_depth_coverage(snapshot, min_confidence=0.5) == 1.0
    resized = _resize_depth_snapshot(snapshot, (6, 8, 3))
    assert resized.depth.shape == (6, 8)
    assert resized.confidence.shape == (6, 8)
    assert resized.mask.shape == (6, 8)


def test_build_virtual_twin_revision_writes_required_bundle(tmp_path: Path) -> None:
    h, w = 90, 100
    k = np.asarray([[120.0, 0.0, 50.0], [0.0, 120.0, 45.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    depth = np.zeros((h, w), dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    masks: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    regions = [
        (slice(12, 42), slice(8, 30), 3.0),
        (slice(36, 72), slice(38, 62), 4.0),
        (slice(18, 80), slice(70, 94), 5.0),
    ]
    for idx, (rows, cols, z) in enumerate(regions):
        mask = np.zeros((h, w), dtype=bool)
        mask[rows, cols] = True
        yy, xx = np.nonzero(mask)
        depth[mask] = z + (0.015 * np.sin(xx * 0.25)).astype(np.float32)
        valid |= mask
        image[mask] = np.asarray([40 + idx * 60, 180 - idx * 35, 220], dtype=np.uint8)
        x = ((float(np.mean(xx)) - k[0, 2]) * z) / k[0, 0]
        y = ((float(np.mean(yy)) - k[1, 2]) * z) / k[1, 1]
        centers.append(np.asarray([x, y, z], dtype=np.float32))
        masks.append(mask)

    obj_path = tmp_path / "structural.obj"
    _write_wall_obj(obj_path, centers)
    snapshot = CalibrationSnapshot(
        camera_id="living-room",
        intrinsics=k,
        extrinsics_col_major=np.eye(4, dtype=np.float64).flatten(order="F").tolist(),
        floor_y=0.0,
        image_size=(w, h),
    )
    frame = VirtualTwinFrameInput(
        frame_id="living-room_0001",
        camera_id="living-room",
        image_bgr=image,
        map_depth=depth,
        map_confidence=conf,
        map_mask=valid,
        calibration=snapshot,
        plane_candidates=[
            PlaneCandidate(frame_id="living-room_0001", plane_id=f"wall_{idx}", mask=mask, normal=np.asarray([0, 0, 1], dtype=np.float32))
            for idx, mask in enumerate(masks)
        ],
    )

    result = build_virtual_twin_revision(
        frames=[frame],
        menon_obj_path=obj_path,
        store=VirtualTwinStore(tmp_path / "vt"),
        revision_id="synthetic_rev",
        min_zero_planes=3,
        min_registration_correspondences=3,
    )

    rev_dir = Path(result["revision_dir"])
    for name in ("manifest.json", "planes.json", "surfaces.glb", "points.ply", "points.npz", "tracking_alignment.json", "metrics.json"):
        assert (rev_dir / name).exists(), name
    metrics = json.loads((rev_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["accepted_plane_count"] == 3
    assert metrics["gates"]["registration_min_correspondences_pass"] is True
    assert metrics["room_model_leakage_ratio"] is not None
    assert metrics["gates"]["room_model_leakage_pass"] is True
    assert metrics["browser_render_budget"]["artifact_type"] == "textured_model_surface_mesh"
    assert metrics["browser_render_budget"]["served_glb_triangle_count"] > 0
    assert metrics["browser_render_budget"]["texture"]["texture_coverage_ratio"] > 0.0
    assert metrics["browser_render_budget"]["texture"]["texture_tone_map"]["exposure"] > 1.0
    glb = _read_glb_json(rev_dir / "surfaces.glb")
    primitive = glb["meshes"][0]["primitives"][0]
    assert primitive["mode"] == 4
    assert "indices" in primitive
    assert "TEXCOORD_0" in primitive["attributes"]
    assert glb["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert glb["images"][0]["mimeType"] == "image/png"
    points_payload = np.load(rev_dir / "points.npz")
    assert "accepted_scene_mask" in points_payload
    planes_payload = json.loads((rev_dir / "planes.json").read_text(encoding="utf-8"))
    assert planes_payload["schema"] == "noesis.virtual_twin.planes.v2"
    assert planes_payload["normal_fusion"]["status"] == "computed_from_mapanything_depth"
    assert len(planes_payload["planes"]) == 3
    assert all("normal_support" in plane for plane in planes_payload["planes"])
    assert all("depth_support" in plane for plane in planes_payload["planes"])
    assert all("fusion_score" in plane for plane in planes_payload["planes"])
    depth_payload = np.load(rev_dir / "mapanything" / "living-room_0001.npz")
    assert "normals_camera" in depth_payload.files
    assert "normals_valid" in depth_payload.files
    assert depth_payload["normals_camera"].shape == (h, w, 3)


def test_rgb_colorfulness_detects_grayscale_keyframes() -> None:
    gray = np.full((8, 8, 3), 80, dtype=np.uint8)
    color = gray.copy()
    color[:, :, 0] = 20
    color[:, :, 1] = 100
    color[:, :, 2] = 180

    assert rgb_colorfulness(gray) == 0.0
    assert rgb_colorfulness(color) > 100.0
