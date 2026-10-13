"""Focused checks for colour encoding, UV origin, and bound-source rejection."""
import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from tools.room_mesh_export.build_browser_mesh import (
    bake, export_glb, linear_to_srgb, ray_scene, read_source, unwrap, uv_raster,
)


def square():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
    return vertices, faces, uvs


def test_linear_colour_becomes_srgb_texture_once():
    np.testing.assert_allclose(linear_to_srgb(np.array([0, .0031308, .5, 1])),
                               [0, .04044994, .73535698, 1], atol=1e-7)


def test_uv_raster_preserves_top_left_image_origin():
    _, faces, uvs = square()
    face_map, bary = uv_raster(uvs, faces, 16)
    assert (face_map >= 0).all()
    vertices, _, _ = square()
    for y, x in ((0, 0), (0, 15), (15, 0), (15, 15)):
        weights = np.r_[bary[y, x], 1-bary[y, x].sum()]
        point = weights @ vertices[faces[face_map[y, x]]]
        np.testing.assert_allclose(point, [(x+.5)/16, 1-(y+.5)/16, 0], atol=1e-6)


def test_dense_colour_bake_uses_source_triangle_barycentrics():
    vertices, faces, uvs = square()
    colours = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]], np.float32)
    atlas, report = bake(ray_scene(vertices, faces, 1), faces, colours, vertices, faces, uvs, 32, 1)
    pixels = np.asarray(atlas)
    assert pixels[0, 16, 0] > 250 and pixels[0, 16, 2] < 40
    assert pixels[-1, 16, 2] > 250 and pixels[-1, 16, 0] < 40
    assert report["source_projection_distance_m"]["maximum"] < 1e-6
    assert report["image_inpainting"] is False


def test_actual_xatlas_binding_produces_one_finite_layout():
    vertices, faces, _ = square()
    result_vertices, result_faces, uv, mapping, report = unwrap(vertices, faces, 64,
                                                              chart_iterations=1, chart_area=.75)
    assert result_faces.shape == faces.shape
    np.testing.assert_array_equal(result_vertices, vertices[mapping])
    assert np.isfinite(uv).all() and uv.min() >= 0 and uv.max() <= 1
    assert report["chart_count"] >= 1
    assert report["world_space_corners_and_winding_changed"] is False


def test_spatial_partition_keeps_every_original_triangle_corner_and_winding():
    vertices, faces, _ = square()
    _, output_faces, _, mapping, report = unwrap(vertices, faces, 64, cell_m=.25)
    recovered = mapping[output_faces]
    assert sorted(map(tuple, recovered)) == sorted(map(tuple, faces))
    assert report["spatial_patch_count"] == 2


def test_large_uv_alias_is_rejected_instead_of_mixing_surface_colours():
    _, faces, uvs = square()
    faces = np.concatenate((faces, faces))
    with pytest.raises(ValueError, match="excessive ambiguous"):
        uv_raster(uvs, faces, 32)


def test_small_uv_alias_stays_neutral_without_changing_geometry():
    vertices, faces, uvs = square()
    # A distinct world surface occupies just one tiny part of the same UV image.
    extra = np.array([[0, 0, 1], [.01, 0, 1], [0, .01, 1]], np.float32)
    tiny_uv = np.array([[.499, .499], [.505, .499], [.499, .505]], np.float32)
    vertices = np.concatenate((vertices, extra))
    uvs = np.concatenate((uvs, tiny_uv))
    faces = np.concatenate((faces, [[4, 5, 6]])).astype(np.uint32)
    colours = np.tile([0, 0, 1], (len(vertices), 1)).astype(np.float32)
    colours[4:] = [1, 0, 0]
    atlas, report = bake(ray_scene(vertices, faces, 1), faces, colours, vertices, faces, uvs, 256, 1)
    assert report["uv_raster"]["neutral_uv_overlap_texels"] > 0
    np.testing.assert_array_equal(np.asarray(atlas)[128, 128], [128, 128, 128])
    assert report["unobserved_geometry_added"] is False


def test_serialized_glb_embeds_one_jpeg_and_preserves_uvs(tmp_path):
    vertices, faces, uvs = square()
    picture = np.zeros((32, 32, 3), np.uint8)
    picture[:16, :, 0] = 255
    picture[16:, :, 2] = 255
    path = tmp_path / "room.glb"
    export_glb(path, vertices, faces, uvs, np.tile([0, 0, 1], (4, 1)),
               Image.fromarray(picture), {"review_only": True})
    blob = path.read_bytes()
    length = struct.unpack_from("<I", blob, 12)[0]
    document = json.loads(blob[20:20+length])
    binary = blob[28+length:]
    assert document["extensionsUsed"] == ["KHR_materials_unlit"]
    assert len(document["images"]) == len(document["materials"]) == 1
    view = document["bufferViews"][document["images"][0]["bufferView"]]
    loaded = np.asarray(Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"]+view["byteLength"]])))
    assert loaded[4, 4, 0] > 250 and loaded[28, 4, 2] > 250
    uv_view = document["bufferViews"][document["accessors"][2]["bufferView"]]
    decoded = np.frombuffer(binary, "<f4", count=8, offset=uv_view["byteOffset"]).reshape(4, 2)
    np.testing.assert_array_equal(decoded, uvs)


def test_wrong_source_hash_is_rejected_before_decoding(tmp_path):
    path = tmp_path / "different.glb"
    path.write_bytes(b"wrong source")
    manifest = {"output": {"sha256": "0"*64, "size_bytes": path.stat().st_size}}
    with pytest.raises(ValueError, match="does not match"):
        read_source(path, manifest)
