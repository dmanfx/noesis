import numpy as np
import trimesh

from tools.mapanything_phone_scan.render_cpu_review_mesh import color_rgba


def test_native_vertex_colors_do_not_require_missing_to_color_method():
    mesh = trimesh.creation.box()
    colors = np.tile([12, 34, 56, 255], (len(mesh.vertices), 1)).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    assert np.array_equal(color_rgba(mesh.visual, len(mesh.vertices)), colors)


def test_texture_neutral_visual_conversion_is_preserved():
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=[.2, .4, .6, 1.]))
    rgba = color_rgba(mesh.visual, len(mesh.vertices))
    assert rgba.shape == (len(mesh.vertices), 4)
    assert np.array_equal(rgba[0], [255, 255, 255, 255])
    assert np.array_equal(mesh.visual.material.baseColorFactor, [51, 102, 153, 255])


def test_explicit_gltf_linear_colors_are_display_encoded_only_in_cpu_consumer():
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = np.tile([55, 55, 55, 255], (len(mesh.vertices), 1))
    rgba = color_rgba(mesh.visual, len(mesh.vertices), encoding="gltf_linear_uint8")
    assert np.array_equal(rgba[0], [128, 128, 128, 255])
    assert np.array_equal(mesh.visual.vertex_colors[0], [55, 55, 55, 255])


def test_unlit_material_plus_color_attribute_keeps_native_vertex_rgb():
    visual = trimesh.visual.texture.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1., 1., 1., 1.]))
    visual.vertex_attributes["color"] = np.tile([55, 55, 55, 255], (8, 1)).astype(np.uint8)
    rgba = color_rgba(visual, 8, encoding="gltf_linear_uint8")
    assert np.array_equal(rgba[0], [128, 128, 128, 255])
