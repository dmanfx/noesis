import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from tools.mapanything_phone_scan import texture_single_room_review_surface as texture
from tools.mapanything_phone_scan.bake_continuous_video_color import build, finalize_bound_bake, robust_video_colors, unlit_vertex_color


def test_no_photo_keeps_explicit_input_interpolation_not_invented_photo():
    fallback = np.asarray([[80, 120, 180], [40, 50, 60]], dtype=np.uint8)
    output, count = robust_video_colors(np.zeros((2, 4, 3), dtype=np.uint8), np.zeros((2, 4)), fallback)
    np.testing.assert_array_equal(output, fallback)
    np.testing.assert_array_equal(count, 0)


def test_one_real_photo_is_exact_and_missing_slots_do_not_darkening():
    samples = np.asarray([[[127, 162, 213], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    output, count = robust_video_colors(samples, np.asarray([[.5, 0, 0]]), np.zeros((1, 3), dtype=np.uint8))
    np.testing.assert_array_equal(output, samples[:, 0])
    assert count[0] == 1


def test_extreme_exposure_outlier_is_downweighted_in_linear_rgb():
    samples = np.asarray([[[120, 120, 120], [121, 121, 121], [122, 122, 122], [245, 245, 245]]], dtype=np.uint8)
    output, count = robust_video_colors(samples, np.ones((1, 4)), np.zeros((1, 3), dtype=np.uint8))
    assert np.all((output >= 120) & (output < 127))
    assert count[0] == 4


def test_negative_or_nonfinite_sample_weights_fail_closed():
    for weights in (np.asarray([[-1., 0]]), np.asarray([[np.nan, 0]])):
        with pytest.raises(ValueError, match="malformed"):
            robust_video_colors(np.zeros((1, 2, 3)), weights, np.zeros((1, 3)))


def test_vertex_color_is_bound_to_a_white_unlit_material_not_default_pbr():
    tree = {"meshes": [{"primitives": [{"attributes": {"COLOR_0": 2}, "indices": 0}]}]}
    unlit_vertex_color(tree)
    assert tree["meshes"][0]["primitives"][0]["material"] == 0
    assert tree["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [1, 1, 1, 1]
    assert tree["materials"][0]["extensions"]["KHR_materials_unlit"] == {}
    assert tree["materials"][0]["doubleSided"] is True
    assert tree["extensionsUsed"] == ["KHR_materials_unlit"]


def test_unlit_vertex_export_rejects_missing_color_attribute():
    with pytest.raises(ValueError, match="COLOR_0"):
        unlit_vertex_color({"meshes": [{"primitives": [{"attributes": {"POSITION": 1}}]}]})


@pytest.mark.parametrize("already_unlit", [False, True])
def test_material_finalization_roundtrip_preserves_baked_geometry_color_and_provenance(tmp_path, already_unlit):
    source = tmp_path / "source"
    source.mkdir()
    mesh = trimesh.creation.box()
    colors = np.tile([55, 24, 132, 255], (len(mesh.vertices), 1)).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    mesh.metadata["vertex_color_encoding"] = "gltf_linear_uint8"
    path = source / "surface_mesh.glb"
    extra = {"tree_postprocessor": unlit_vertex_color} if already_unlit else {}
    path.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), include_normals=True, **extra))
    sidecar = source / "vertex_video_color_sources.npz"
    np.savez_compressed(sidecar, sample_count=np.ones(len(mesh.vertices), dtype=np.uint8))
    manifest = {"status": "review_only", "accepted_for_canonical_use": False, "source": {},
        "output": {"sha256": texture._sha(path)}, "color_bake": {
            "vertex_color_encoding": "gltf_linear_uint8", "color_sources": texture._record(sidecar)}}
    (source / "surface_mesh_manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "output"
    result = finalize_bound_bake(source, output)
    loaded = trimesh.load(output / "surface_mesh.glb", force="scene", process=False)
    direct = next(iter(loaded.geometry.values()))
    np.testing.assert_array_equal(direct.vertices, mesh.vertices)
    np.testing.assert_array_equal(direct.faces, mesh.faces)
    np.testing.assert_array_equal(direct.visual.vertex_attributes["color"], colors)
    assert result["color_bake"]["gltf_unlit_material_bound"] is True
    assert (output / "vertex_video_color_sources.npz").read_bytes() == sidecar.read_bytes()


def test_original_provider_color_gate_cannot_silently_switch_to_same_mesh_visibility(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path / "unused", color_top_k=4, surface_visibility_dir=tmp_path)
    with pytest.raises(ValueError, match="original provider depth gate"):
        build(args)
    assert not args.output_dir.exists()
