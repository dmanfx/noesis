from __future__ import annotations

import json
import io
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest
import trimesh

from tools.mapanything_phone_scan import texture_single_room_review_surface as texture
from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity


def _view(index=0, *, depth=2.0, affine=None, image_size=(64, 64)):
    k = np.asarray([[20.0, 0, 31.5], [0, 20.0, 31.5], [0, 0, 1]])
    return texture.PhotoView(
        index, np.full((64, 64), depth, dtype=np.float32), np.ones((64, 64), dtype=bool),
        np.ones((64, 64), dtype=np.float32), k, np.eye(4), k,
        np.asarray([[1.0, 0, 0], [0, 1.0, 0]]) if affine is None else affine,
        Path("unused.png"), image_size, 1.0, {},
    )


def _square():
    return np.asarray([[-.2, -.2, 2.0], [.2, -.2, 2.0], [.2, .2, 2.0], [-.2, .2, 2.0]]), np.asarray([[0, 1, 2], [0, 2, 3]])


def test_triangle_support_requires_all_corners_center_and_unoccluded_depth():
    vertices, faces = _square()
    good, occluded = _view(0), _view(1, depth=1.0)
    candidates, scores = texture.face_candidates(vertices, faces, [good, occluded])
    assert np.all(candidates[:, 0] == 0)
    assert np.all(scores[:, 0] > 0)
    assert not np.any(candidates == 1)
    good.mask[30, 30] = False  # Projected first corner of BOTH triangles.
    candidates, _ = texture.face_candidates(vertices, faces, [good])
    assert np.all(candidates == -1)


def test_out_of_photo_projection_is_rejected_without_uv_clipping():
    vertices, faces = _square()
    outside = _view(0, affine=np.asarray([[1.0, 0, 100], [0, 1.0, 0]]))
    candidates, _ = texture.face_candidates(vertices, faces, [outside])
    assert np.all(candidates == -1)
    scene = trimesh.Scene()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    with pytest.raises(texture.TextureSurfaceError, match="UV clipping"):
        texture.export_partition(mesh, np.zeros(2, dtype=np.int16), {0: outside}, scene, "test", {0: Image.new("RGB", (64, 64))})


def test_adjacency_regularization_coalesces_valid_islands_not_invalid_candidates():
    count = 12
    neighbors = np.full((count, 3), -1, dtype=np.int32)
    for index in range(count):
        if index:
            neighbors[index, 0] = index - 1
        if index < count - 1:
            neighbors[index, 1] = index + 1
    weights = (neighbors >= 0).astype(np.float32)
    candidates = np.tile([0, 1], (count, 1)).astype(np.int16)
    scores = np.tile([1.0, .95], (count, 1)).astype(np.float32)
    candidates[1::2] = [1, 0]
    candidates[-1] = [1, -1]  # View0 cannot cover this final face.
    scores[-1] = [1.0, 0.0]
    labels, report = texture.coherent_labels(candidates, scores, neighbors, weights, minimum_patch_faces=4)
    assert report["after"]["photo_patch_count"] < report["before"]["photo_patch_count"]
    assert labels[-1] == 1
    assert np.all((candidates == labels[:, None]).any(1))
    assert report["colour_blending"] is False


def test_neutral_fallback_keeps_exact_triangle_positions():
    vertices, faces = _square()
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene = trimesh.Scene()
    report = texture.export_partition(source, np.full(2, -1, dtype=np.int16), {}, scene, "surface", {})
    neutral = next(iter(scene.geometry.values()))
    np.testing.assert_array_equal(neutral.vertices[neutral.faces], vertices[faces])
    assert report["neutral_triangle_count"] == 2
    np.testing.assert_array_equal(neutral.visual.vertex_colors, np.tile([136, 142, 146, 255], (4, 1)))


def _json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _fixture(tmp_path, monkeypatch):
    scan, consensus, alignment, mesh_dir = [tmp_path / name for name in ("scan", "consensus", "alignment", "mesh")]
    model_root = tmp_path / "suite" / "mapanything_da3_pose_sparse_depth"
    for path in (scan / "frames", consensus / "raw", alignment, mesh_dir, model_root / "raw"):
        path.mkdir(parents=True)
    frames, model_frames, raw_rows = [], [], []
    k = np.asarray([[20.0, 0, 31.5], [0, 20.0, 31.5], [0, 0, 1]])
    for index in range(2):
        photo = scan / "frames" / f"frame_{index:04d}.png"
        Image.new("RGB", (512, 512), (128, 120, 100)).save(photo)
        sha = texture._sha(photo)
        frames.append({"index": index, "frame_id": prepared_frame_identity(index, sha), "frame": str(photo.relative_to(scan)),
                       "width": 512, "height": 512, "sha256": sha, "timestamp_s": float(index),
                       "quality": {"blur_score": 400, "luminance_p50": 120}})
        raw_path = consensus / "raw" / f"view_{index:04d}.npz"
        fields = dict(depth_z=np.full((64, 64), 2.0), mask=np.ones((64, 64)), confidence=np.ones((64, 64)),
                      intrinsics=k, camera_pose=np.eye(4), model_rgb=np.full((64, 64, 3), 128, dtype=np.uint8))
        np.savez(raw_path, **fields)
        np.savez(model_root / "raw" / raw_path.name, **fields)
        raw_rows.append({"path": str(raw_path), "sha256": texture._sha(raw_path), "size_bytes": raw_path.stat().st_size})
        model_frames.append({"index": index, "phone_frame_index": index, "source_frame": frames[-1]["frame"],
                             "timestamp_s": float(index), "fixed_camera_anchor": False})
    prepared_path = scan / "prepared_frames_manifest.json"
    _json(prepared_path, {"frames": frames})
    _json(model_root.parent / "variant_suite_manifest.json", {"scan_dir": str(scan), "prepared_frames_manifest_sha256": texture._sha(prepared_path)})
    _json(model_root / "variant_manifest.json", {"inputs": {"phone_view_count": 2, "uses_static_reference": False},
          "world_output_policy": "mapanything_depth_and_intrinsics_backprojected_through_da3_pose_carrier", "frames": model_frames})
    _json(consensus / "consensus_manifest.json", {"phone_walk_only": True, "static_camera_data_used": False, "pose_carrier": "da3",
          "inputs": {"mapanything_raw": str(model_root / "raw")}})
    output_manifest = consensus / "scan_outputs_manifest.json"
    _json(output_manifest, {"view_count": 2})
    output_binding = {"path": str(output_manifest), "sha256": texture._sha(output_manifest)}
    camera_binding = {"frame_id": "backend_world_m", "revision": "camera-test"}
    target = {"camera_id": "living-room", "revision_id": "static-test", "coordinate_frame": "backend_world_m_stream_points",
              "world_frame": "backend_world_m", "world_frame_revision": "world-test", "camera_frame_binding": camera_binding}
    transform = {"source_raw_root": str(consensus / "raw"), "source_output_manifest": output_binding,
                 "world_from_mapanything_row_major": np.eye(4).tolist()}
    _json(alignment / "alignment_report.json", {"status": "passed", "quality_gate": {"passed": True}, "target": target,
          "transform": transform, "inputs": {"phone_source": {"raw_root": str(consensus / "raw"), "output_manifest": output_binding, "views": raw_rows}}})
    _json(alignment / "phone_ma_to_noesis_world.json", {**transform, "target_coordinate_frame": target["coordinate_frame"],
          "source_coordinate_frame": "consensus_phone_metric_world_unaligned_to_noesis", "target_binding": target})
    vertices, faces = _square()
    scene = trimesh.Scene(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    mesh_path = mesh_dir / "surface_mesh.glb"
    mesh_path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    input_path = mesh_dir / "surface_input_manifest.json"
    _json(input_path, {"source": {"consensus_manifest": texture._record(consensus / "consensus_manifest.json"),
          "alignment_report": texture._record(alignment / "alignment_report.json"),
          "alignment_transform": texture._record(alignment / "phone_ma_to_noesis_world.json")}})
    _json(mesh_dir / "surface_mesh_manifest.json", {"status": "review_only", "accepted_for_canonical_use": False,
          "coordinate_frame": "backend_world_m_stream_points", "output": {"sha256": texture._sha(mesh_path), "triangle_count": 2},
          "source": {"surfels_manifest": str(input_path), "surfels_manifest_sha256": texture._sha(input_path)}})
    monkeypatch.setattr(texture, "image_match", lambda *_: (np.asarray([[8.0, 0, 0], [0, 8.0, 0]]), {"status": "fixture_exact_resize"}))
    args = texture._parser().parse_args(["--scan-dir", str(scan), "--consensus-dir", str(consensus), "--alignment-dir", str(alignment),
                                         "--mesh-dir", str(mesh_dir), "--output-dir", str(tmp_path / "textured"), "--texture-edge-px", "512"])
    return args


def test_bound_producer_exports_actual_textured_glb_and_exact_geometry(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    report = texture.build(args)
    assert report["output"]["photo_triangle_count"] == 2
    assert report["output"]["exported_triangle_count"] == 2
    loaded = trimesh.load(args.output_dir / "textured_surface_mesh.glb", force="scene", process=False)
    assert all(isinstance(item.visual, trimesh.visual.texture.TextureVisuals) for item in loaded.geometry.values())
    part = next(iter(loaded.geometry.values()))
    vertices, faces = _square()
    np.testing.assert_allclose(part.vertices[part.faces], vertices[faces], atol=1e-7)
    assert report["policy"]["uv_clipping"] is False
    assert report["policy"]["geometry_changed"] is False
    assert (args.output_dir / "face_photo_assignment.npz").is_file()


def test_prepared_digest_mismatch_fails_before_export(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    path = args.scan_dir / "prepared_frames_manifest.json"
    payload = texture._json(path)
    payload["frames"][0]["timestamp_s"] = 100.0
    _json(path, payload)
    with pytest.raises(texture.TextureSurfaceError, match="different prepared manifest"):
        texture.build(args)
    assert not args.output_dir.exists()


def test_texture_quality_exclusion_preserves_all_geometry_as_neutral(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    args.minimum_texture_blur = 999.0
    report = texture.build(args)
    assert report["output"]["neutral_triangle_count"] == 2
    assert report["output"]["photo_triangle_count"] == 0
    assert report["output"]["exported_triangle_count"] == 2
    assert all(view["geometry_preserved_when_texture_excluded"] for view in report["views"])


def _photometric_fixture():
    points = np.asarray([[(5 + 2.5 * (row % 20) - 31.5) / 10,
                          (5 + 5 * (row // 20) - 31.5) / 10, 2.0] for row in range(200)])
    affine = np.asarray([[8.0, 0, 0], [0, 8.0, 0]])
    views = {index: _view(index, affine=affine, image_size=(512, 512)) for index in range(3)}
    first = texture._encoded_rgb(np.asarray([.30, .25, .20]))
    second = texture._encoded_rgb(np.asarray([.24, .20, .16]))
    photos = {0: Image.new("RGB", (512, 512), tuple(first)),
              1: Image.new("RGB", (512, 512), tuple(second)),
              2: Image.new("RGB", (512, 512), tuple(first))}
    candidates = np.tile([0, 1], (len(points), 1))
    return points, candidates, views, photos


def test_measured_photometric_gain_requires_same_surface_overlap_and_anchor():
    points, candidates, views, photos = _photometric_fixture()
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0)
    assert report["status"] == "accepted_measured_gains"
    np.testing.assert_array_equal(gains[0], np.ones(3))
    expected_gain = texture._linear_rgb(np.asarray(photos[0])[0, 0]) / texture._linear_rgb(np.asarray(photos[1])[0, 0])
    np.testing.assert_allclose(gains[1], expected_gain, atol=1e-5)
    np.testing.assert_array_equal(gains[2], np.ones(3))  # Disconnected view is not adjusted.
    assert report["disconnected_views"] == [2]
    assert report["heldout_linear_rgb_error_after_p50"] < report["heldout_linear_rgb_error_before_p50"]
    assert report["reserved_sample_scope"] == "same_scene_same_bound_images_not_independent_accuracy"
    assert all(.67 <= value <= 1.5 for gain in gains.values() for value in gain)


def test_inconsistent_photometric_holdout_retains_exact_original_colours():
    points, candidates, views, photos = _photometric_fixture()
    second = np.asarray(photos[1]).copy()
    _, _, uv = texture.project_photo(points, views[1])
    first_colour = np.asarray(photos[0])[0, 0]
    for x, y in np.rint(uv[::5]).astype(int):
        second[y - 5:y + 6, x - 5:x + 6] = first_colour
    photos[1] = Image.fromarray(second)
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0)
    assert report["status"] == "rejected_holdout_or_clipping_original_photos_retained"
    assert report["heldout_linear_rgb_error_before_p50"] == 0
    assert all(np.array_equal(gain, np.ones(3)) for gain in gains.values())


def test_half_pixel_boundary_is_not_wrapped_for_photometric_neighborhood():
    points, candidates, views, photos = _photometric_fixture()
    for view in views.values():
        view.model_to_photo = np.asarray([[1.0, 0, -31.0], [0, 1.0, 0]])
    boundary = np.tile([0, 0, 2], (200, 1))
    gains, report = texture.measured_photo_gains(boundary, candidates, views, photos, anchor=0)
    assert report["status"] == "insufficient_measured_overlap"
    assert all(np.array_equal(gain, np.ones(3)) for gain in gains.values())


def test_exported_gltf_uv_and_embedded_photo_preserve_top_bottom_orientation():
    vertices, faces = _square()
    view = _view()
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:32] = (255, 20, 10)
    pixels[32:] = (10, 20, 255)
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene = trimesh.Scene()
    texture.export_partition(source, np.zeros(2, dtype=np.int16), {0: view}, scene, "surface", {0: Image.fromarray(pixels)})
    payload = trimesh.exchange.gltf.export_glb(scene, tree_postprocessor=texture._unlit)
    json_length = struct.unpack_from("<I", payload, 12)[0]
    tree = json.loads(payload[20:20 + json_length])
    assert tree["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
    binary = payload[28 + json_length:]
    primitive = tree["meshes"][0]["primitives"][0]
    accessor = tree["accessors"][primitive["attributes"]["TEXCOORD_0"]]
    buffer = tree["bufferViews"][accessor["bufferView"]]
    offset = buffer.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    actual_uv = np.frombuffer(binary, dtype="<f4", count=accessor["count"] * 2, offset=offset).reshape(-1, 2)
    _, _, projected = texture.project_photo(vertices, view)
    np.testing.assert_allclose(actual_uv, projected / 64, atol=1e-7)
    image_buffer = tree["bufferViews"][tree["images"][0]["bufferView"]]
    start = image_buffer.get("byteOffset", 0)
    embedded = np.asarray(Image.open(io.BytesIO(binary[start:start + image_buffer["byteLength"]])))
    xy = np.rint(actual_uv * 64).astype(int)
    np.testing.assert_array_equal(embedded[xy[:, 1], xy[:, 0]], pixels[np.rint(projected[:, 1]).astype(int), np.rint(projected[:, 0]).astype(int)])
    assert not np.array_equal(embedded[xy[0, 1], xy[0, 0]], embedded[xy[2, 1], xy[2, 0]])
    assert "KHR_materials_unlit" in tree["materials"][0]["extensions"]


def test_invalid_photometric_work_bound_fails_before_reading_inputs(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    args.photometric_sample_limit = 0
    with pytest.raises(texture.TextureSurfaceError, match="finite supported bounds"):
        texture.build(args)
    assert not args.output_dir.exists()


def _two_component_photometric_fixture():
    points, candidates, views, photos = _photometric_fixture()
    views.update({index: _view(index, affine=views[0].model_to_photo, image_size=(512, 512)) for index in (3, 4)})
    photos.update({3: photos[1].copy(), 4: photos[0].copy()})
    return np.tile(points, (2, 1)), np.vstack((candidates, np.tile([2, 3], (len(points), 1)))), views, photos


def test_component_normalization_preserves_supplied_anchor_and_other_largest_support_anchor():
    points, candidates, views, photos = _two_component_photometric_fixture()
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0,
                                                face_counts={0: 1, 1: 100, 2: 2, 3: 200, 4: 999})
    assert report["accepted_component_count"] == 2
    assert report["component_count"] == 3
    assert report["attempted_fitted_view_count"] == report["fitted_view_count"] == 2
    assert report["accepted_adjusted_view_count"] == 2
    assert [component["anchor_view"] for component in report["components"]] == [0, 3, 4]
    np.testing.assert_array_equal(gains[0], np.ones(3))  # Explicit anchor wins over face count.
    np.testing.assert_array_equal(gains[3], np.ones(3))
    np.testing.assert_array_equal(gains[4], np.ones(3))  # Unmeasured singleton.
    np.testing.assert_allclose(gains[1], texture._linear_rgb(np.asarray(photos[0])[0, 0]) /
                               texture._linear_rgb(np.asarray(photos[1])[0, 0]), atol=1e-5)
    np.testing.assert_allclose(gains[2], texture._linear_rgb(np.asarray(photos[3])[0, 0]) /
                               texture._linear_rgb(np.asarray(photos[2])[0, 0]), atol=1e-5)
    assert np.all(gains[1] > 1) and np.all(gains[2] < 1)  # Sign follows each component's actual reference.
    assert report["disconnected_components_globally_calibrated"] is False


def test_component_holdout_rejection_does_not_reject_independent_accepted_component():
    points, candidates, views, photos = _two_component_photometric_fixture()
    altered = np.asarray(photos[2]).copy()
    _, _, uv = texture.project_photo(points[200::5], views[2])
    for x, y in np.rint(uv).astype(int):
        altered[y - 5:y + 6, x - 5:x + 6] = np.asarray(photos[3])[0, 0]
    photos[2] = Image.fromarray(altered)
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0, face_counts={3: 100})
    assert report["status"] == "accepted_measured_gains"
    assert report["accepted_component_count"] == report["fitted_view_count"] == 1
    assert report["attempted_fitted_view_count"] == 2
    assert report["components"][1]["status"] == "rejected_holdout_or_clipping_original_photos_retained"
    assert report["components"][1]["heldout_linear_rgb_error_before_p50"] == 0
    np.testing.assert_array_equal(gains[2], np.ones(3))
    np.testing.assert_array_equal(gains[3], np.ones(3))
    assert np.all(gains[1] > 1)


def test_component_clipping_rejection_is_separate_from_other_component():
    points, candidates, views, photos = _two_component_photometric_fixture()
    altered = np.asarray(photos[3]).copy()
    altered[:30] = 245  # Outside measured points; the fitted gain would newly clip >1% of this photo.
    photos[3] = Image.fromarray(altered)
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0, face_counts={2: 100})
    assert report["accepted_component_count"] == 1
    rejected = report["components"][1]
    assert rejected["status"] == "rejected_holdout_or_clipping_original_photos_retained"
    assert rejected["newly_clipped_pixel_fraction"]["3"] > .01
    np.testing.assert_array_equal(gains[3], np.ones(3))
    assert np.all(gains[1] > 1)


def test_component_no_overlap_keeps_all_photos_exact_with_explicit_singletons():
    points, candidates, views, photos = _photometric_fixture()
    candidates[:] = -1
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0)
    assert report["component_count"] == 3
    assert len(report["components"]) == 3
    assert report["status"] == "insufficient_measured_overlap"
    assert report["accepted_component_count"] == report["fitted_view_count"] == 0
    assert all(np.array_equal(gain, np.ones(3)) for gain in gains.values())


def test_official_image_map_preserves_anisotropic_resize_and_crop_offsets():
    da_k = np.asarray([[330., 0, 252], [0, 328, 140], [0, 0, 1]])
    da_to_ma = np.asarray([[1.05, 0, -6.075], [0, 1.05, .025], [0, 0, 1]])
    affine = texture._compose_photo_map((1280, 720), (504, 280), (1280, 720), da_k, da_to_ma @ da_k)
    points = np.asarray([[0., 0., 1], [503., 279., 1], [252., 140., 1]])
    model = points @ da_to_ma.T
    np.testing.assert_allclose(model @ affine.T, points[:, :2] * [1280 / 504, 720 / 280], atol=1e-10)
    assert affine[0, 0] != affine[1, 1]  # A similarity/SIFT map cannot represent this pipeline exactly.


def _official_replay_fixture(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    prepared = texture._json(args.scan_dir / "prepared_frames_manifest.json")
    da_root = tmp_path / "da3"
    (da_root / "raw").mkdir(parents=True)
    k = np.asarray([[20., 0, 31.5], [0, 20., 31.5], [0, 0, 1]])
    saved_da_rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    for index in range(2):
        np.savez(da_root / "raw" / f"view_{index:04d}.npz", model_rgb=saved_da_rgb, intrinsics=k)
    _json(da_root / "run_inputs.json", {"scan": str(args.scan_dir),
          "prepared_manifest": str(args.scan_dir / "prepared_frames_manifest.json"),
          "prepared_sha256": texture._sha(args.scan_dir / "prepared_frames_manifest.json"),
          "view_count": 2, "settings": {"process_res": 504}})
    _json(da_root / "scan_outputs_manifest.json", {"view_count": 2, "frames": [
        {"index": index, "source_frame": row["frame"], "timestamp_s": row["timestamp_s"], "fixed_camera_anchor": False}
        for index, row in enumerate(prepared["frames"])]})
    class FakeTensor:
        def permute(self, *_):
            return self
        def cpu(self):
            return self
        def numpy(self):
            # .25 byte excess avoids precision-dependent truncation in this fake.
            return (saved_da_rgb.astype(float) + .25) / 255 / np.array([.229, .224, .225]) - np.array([.485, .456, .406]) / np.array([.229, .224, .225])
    class FakeProcessor:
        def _process_one(self, *_args, **_kwargs):
            return FakeTensor(), (64, 64), None, None
    def fake_crop(rgb, size, *, intrinsics):
        assert size == (64, 64)
        return Image.fromarray(rgb), intrinsics.copy()
    monkeypatch.setattr(texture, "_official_rgb_replayers", lambda: (FakeProcessor(), fake_crop, {"test_fixture": True}))
    consensus = texture._json(args.consensus_dir / "consensus_manifest.json")
    consensus["inputs"]["da3_raw"] = str(da_root / "raw")
    return args, prepared["frames"], consensus


def test_official_maps_require_exact_rgb_replay_for_every_view(tmp_path, monkeypatch):
    args, frames, consensus = _official_replay_fixture(tmp_path, monkeypatch)
    maps, report = texture.verified_official_photo_maps(args, frames, consensus, Path(consensus["inputs"]["mapanything_raw"]))
    assert report["status"] == "verified_byte_exact_official_preprocessing_all_views"
    assert report["verified_view_count"] == len(maps) == 2
    np.testing.assert_allclose(maps[1], [[8, 0, 0], [0, 8, 0]], atol=1e-10)
    assert report["geometry_intrinsics_changed"] is False
    # One later mismatching model invalidates deterministic maps for ALL views.
    model = Path(consensus["inputs"]["mapanything_raw"]) / "view_0001.npz"
    with np.load(model) as raw:
        fields = {key: raw[key].copy() for key in raw.files}
    fields["model_rgb"][0, 0, 0] = 127
    np.savez(model, **fields)
    maps, report = texture.verified_official_photo_maps(args, frames, consensus, model.parent)
    assert not maps
    assert report["status"] == "fallback_to_measured_image_match"
    assert report["verified_view_count"] == 1
    assert "not byte-exact at view 1" in report["reason"]


def test_official_maps_missing_source_identity_retains_explicit_fallback_diagnostic(tmp_path, monkeypatch):
    args, frames, consensus = _official_replay_fixture(tmp_path, monkeypatch)
    run_path = Path(consensus["inputs"]["da3_raw"]).parent / "run_inputs.json"
    run = texture._json(run_path)
    run["prepared_sha256"] = "mismatch"
    _json(run_path, run)
    maps, report = texture.verified_official_photo_maps(args, frames, consensus, Path(consensus["inputs"]["mapanything_raw"]))
    assert not maps and report["status"] == "fallback_to_measured_image_match"
    assert "binding mismatch" in report["reason"]


def test_bound_view_loader_uses_proven_maps_without_feature_fit_or_predicted_k_substitution(tmp_path, monkeypatch):
    args, _, consensus = _official_replay_fixture(tmp_path, monkeypatch)
    _json(args.consensus_dir / "consensus_manifest.json", consensus)
    model = Path(consensus["inputs"]["mapanything_raw"]) / "view_0001.npz"
    with np.load(model) as raw:
        fields = {key: raw[key].copy() for key in raw.files}
    fields["intrinsics"][0, 0] = 23.0  # Model predicted K is deliberately unlike preprocessed input K.
    np.savez(model, **fields)
    def disallowed_feature_fit(*_):
        raise AssertionError("SIFT must not run when every source image replays exactly")
    monkeypatch.setattr(texture, "image_match", disallowed_feature_fit)
    views, provenance = texture.load_bound_views(args)
    assert provenance["photo_mapping_verification"]["verified_view_count"] == 2
    assert all(view.provenance["image_match"]["feature_fit_used"] is False for view in views)
    np.testing.assert_array_equal(views[1].model_intrinsics, fields["intrinsics"])
    _, _, projected = texture.project_photo(np.asarray([[.2, 0, 2]]), views[1])
    np.testing.assert_allclose(projected, [[(23 * .2 / 2 + 31.5) * 8, 31.5 * 8]])


def test_bound_view_loader_replay_failure_keeps_measured_feature_fit_with_reason(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    views, provenance = texture.load_bound_views(args)
    proof = provenance["photo_mapping_verification"]
    assert proof["status"] == "fallback_to_measured_image_match"
    assert "da3_raw" in proof["reason"]
    assert all(view.provenance["image_match"]["status"] == "fixture_exact_resize" for view in views)


def test_training_anchor_preselection_is_deterministic_bounded_and_training_only():
    edges = [(10, 11, np.full(3, -.10)), (11, 12, np.full(3, -.20)), (12, 13, np.full(3, -.20))]
    support = {index: 100 for index in (10, 11, 12, 13)}
    anchors, report = texture.training_component_anchors([13, 11, 10, 12], edges, 10, support)
    repeated, repeated_report = texture.training_component_anchors([10, 11, 12, 13], list(reversed(edges)), 10, support)
    assert anchors == repeated == [10, 12, 13]
    assert report == repeated_report
    assert len(anchors) <= 3
    assert report["anchor_preselection_uses_reserved_samples"] is False
    assert report["unbounded_training_gains_applied"] is False
    np.testing.assert_allclose(report["training_gain_dynamic_range"], np.exp(np.full(3, .50)))


def test_preferred_anchor_is_retained_without_preselection_when_it_passes(monkeypatch):
    points, candidates, views, photos = _two_component_photometric_fixture()
    def unnecessary_preselection(*_):
        raise AssertionError("passing preferred references must be retained")
    monkeypatch.setattr(texture, "training_component_anchors", unnecessary_preselection)
    _, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0, face_counts={3: 100})
    for component in report["components"]:
        if component["pair_count"]:
            assert len(component["anchor_trials"]) == 1
            assert component["selected_anchor_view"] == component["preferred_anchor_view"]


def test_supplied_anchor_is_never_replaced_even_when_its_fit_is_rejected(monkeypatch):
    points, candidates, views, photos = _photometric_fixture()
    second = np.asarray(photos[1]).copy()
    _, _, uv = texture.project_photo(points[::5], views[1])
    for x, y in np.rint(uv).astype(int):
        second[y - 5:y + 6, x - 5:x + 6] = np.asarray(photos[0])[0, 0]
    photos[1] = Image.fromarray(second)
    def forbidden_preselection(*_):
        raise AssertionError("supplied-anchor component may not try another reference")
    monkeypatch.setattr(texture, "training_component_anchors", forbidden_preselection)
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0, face_counts={1: 999})
    component = next(item for item in report["components"] if item["pair_count"])
    assert component["preferred_anchor_view"] == component["anchor_view"] == 0
    assert component["selected_anchor_view"] is None
    assert len(component["anchor_trials"]) == 1
    assert all(np.array_equal(gain, np.ones(3)) for gain in gains.values())


def _three_view_second_component():
    points, candidates, views, photos = _two_component_photometric_fixture()
    candidates = np.column_stack((candidates, np.full(len(candidates), -1)))
    candidates[200:] = [2, 3, 4]
    photos[4] = Image.new("RGB", (512, 512), tuple(texture._encoded_rgb(np.asarray([.17, .14, .11]))))
    return points, candidates, views, photos


def _fake_fit_result(indices, reference, passed):
    gains = {index: np.ones(3) if index == reference else np.full(3, .9) for index in indices}
    return gains, {"anchor_view": reference,
                   "status": "accepted_measured_gains" if passed else "rejected_holdout_or_clipping_original_photos_retained",
                   "solver_success": True, "residual_checks_passed": passed, "clipping_checks_passed": True,
                   "heldout_linear_rgb_error_before_p50": .03, "heldout_linear_rgb_error_after_p50": .02 if passed else .03,
                   "heldout_linear_rgb_error_before_p80": .05, "heldout_linear_rgb_error_after_p80": .04 if passed else .06,
                   "candidate_gains": {str(index): gain.tolist() for index, gain in gains.items()},
                   "newly_clipped_pixel_fraction": {str(index): 0.0 for index in indices if index != reference}}


def test_first_accepted_preselected_anchor_stops_further_trials(monkeypatch):
    points, candidates, views, photos = _three_view_second_component()
    calls = []
    def fake_fit(indices, pairs, reference, sampled, original_photos):
        calls.append(reference)
        return _fake_fit_result(indices, reference, reference != 2)
    monkeypatch.setattr(texture, "_fit_component_gains", fake_fit)
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0,
                                                face_counts={2: 100, 3: 100, 4: 100})
    component = report["components"][1]
    assert component["anchor_preselection"]["preselected_anchor_views"] == [2, 3, 4]
    assert calls == [0, 2, 3]  # Candidate4 was preselected, never evaluated after candidate3 passed.
    assert component["preferred_anchor_view"] == 2 and component["selected_anchor_view"] == 3
    assert [trial["anchor_view"] for trial in component["anchor_trials"]] == [2, 3]
    np.testing.assert_array_equal(gains[3], np.ones(3))
    assert report["attempted_anchor_fit_count"] == 3


def test_total_anchor_failure_retains_original_pixels_and_logs_every_bounded_trial(monkeypatch):
    points, candidates, views, photos = _three_view_second_component()
    original_pixels = {index: photo.tobytes() for index, photo in photos.items()}
    monkeypatch.setattr(texture, "_fit_component_gains", lambda indices, pairs, reference, sampled, original_photos:
                        _fake_fit_result(indices, reference, False))
    gains, report = texture.measured_photo_gains(points, candidates, views, photos, anchor=0,
                                                face_counts={2: 100, 3: 100, 4: 100})
    component = report["components"][1]
    assert [trial["anchor_view"] for trial in component["anchor_trials"]] == [2, 3, 4]
    assert component["anchor_view"] == component["preferred_anchor_view"] == 2
    assert component["selected_anchor_view"] is None
    assert report["accepted_component_count"] == 0
    assert all(np.array_equal(gain, np.ones(3)) for gain in gains.values())
    assert all(photo.tobytes() == original_pixels[index] for index, photo in photos.items())


def test_reserved_colour_changes_cannot_change_training_anchor_preselection(monkeypatch):
    points, candidates, views, photos = _three_view_second_component()
    monkeypatch.setattr(texture, "_fit_component_gains", lambda indices, pairs, reference, sampled, original_photos:
                        _fake_fit_result(indices, reference, False))
    _, before = texture.measured_photo_gains(points, candidates, views, photos, anchor=0,
                                            face_counts={2: 100, 3: 100, 4: 100})
    changed = np.asarray(photos[3]).copy()
    _, _, reserved_uv = texture.project_photo(points[200::5], views[3])
    for x, y in np.rint(reserved_uv).astype(int):
        changed[y - 5:y + 6, x - 5:x + 6] = texture._encoded_rgb(np.asarray([.20, .18, .15]))
    photos[3] = Image.fromarray(changed)
    _, after = texture.measured_photo_gains(points, candidates, views, photos, anchor=0,
                                           face_counts={2: 100, 3: 100, 4: 100})
    assert before["components"][1]["anchor_preselection"] == after["components"][1]["anchor_preselection"]
    for first, second in zip(before["pairs"], after["pairs"], strict=True):
        assert first["training_log_ratio_median"] == second["training_log_ratio_median"]
    assert after["anchor_preselection_uses_reserved_samples"] is False
    assert after["absolute_colour_calibration_established"] is False
    assert after["disconnected_components_globally_calibrated"] is False
