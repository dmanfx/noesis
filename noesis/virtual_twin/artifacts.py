from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


def _align4(payload: bytes) -> bytes:
    pad = (-len(payload)) % 4
    return payload + (b" " * pad)


def _align4_bin(payload: bytes) -> bytes:
    pad = (-len(payload)) % 4
    return payload + (b"\x00" * pad)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_points_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if colors is None:
        rgb = np.full((pts.shape[0], 3), 210, dtype=np.uint8)
    else:
        rgb = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
        if rgb.shape[0] != pts.shape[0]:
            raise ValueError("PLY colors must match point count")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for point, color in zip(pts, rgb):
            handle.write(
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_points_npz(path: Path, points: np.ndarray, colors: np.ndarray | None = None, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"points": np.asarray(points, dtype=np.float32)}
    if colors is not None:
        payload["colors"] = np.asarray(colors, dtype=np.uint8)
    for key, value in arrays.items():
        payload[str(key)] = value
    np.savez_compressed(path, **payload)


def write_points_glb(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if pts.shape[0] == 0:
        pts = np.zeros((1, 3), dtype=np.float32)
    if colors is None:
        rgb = np.full((pts.shape[0], 3), [190, 215, 235], dtype=np.uint8)
    else:
        rgb = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
        if rgb.shape[0] != pts.shape[0]:
            raise ValueError("GLB colors must match point count")

    position_bytes = pts.astype("<f4", copy=False).tobytes(order="C")
    color_bytes = rgb.astype(np.uint8, copy=False).tobytes(order="C")
    position_offset = 0
    color_offset = len(_align4_bin(position_bytes))
    binary_blob = _align4_bin(position_bytes) + _align4_bin(color_bytes)

    finite_pts = pts[np.all(np.isfinite(pts), axis=1)]
    if finite_pts.size:
        min_pos = [float(x) for x in np.min(finite_pts, axis=0)]
        max_pos = [float(x) for x in np.max(finite_pts, axis=0)]
    else:
        min_pos = [0.0, 0.0, 0.0]
        max_pos = [0.0, 0.0, 0.0]

    gltf = {
        "asset": {"version": "2.0", "generator": "Noesis virtual twin"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "virtual_twin_surface_points"}],
        "meshes": [
            {
                "name": "virtual_twin_surface_points",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "mode": 0,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "virtual_twin_points",
                "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]},
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": color_offset, "byteLength": len(color_bytes), "target": 34962},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(pts.shape[0]),
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos,
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5121,
                "count": int(rgb.shape[0]),
                "type": "VEC3",
                "normalized": True,
            },
        ],
    }
    json_chunk = _align4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    bin_chunk = _align4_bin(binary_blob)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        handle.write(json_chunk)
        handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
        handle.write(bin_chunk)


def write_mesh_glb(path: Path, vertices: np.ndarray, indices: np.ndarray, colors: np.ndarray | None = None) -> None:
    verts = np.asarray(vertices, dtype=np.float32).reshape((-1, 3))
    idx = np.asarray(indices, dtype=np.uint32).reshape((-1,))
    if verts.shape[0] == 0 or idx.shape[0] == 0:
        raise ValueError("mesh GLB requires non-empty vertices and indices")
    if int(np.max(idx)) >= int(verts.shape[0]) or int(np.min(idx)) < 0:
        raise ValueError("mesh GLB index buffer references vertices outside the vertex buffer")
    if colors is None:
        rgb = np.full((verts.shape[0], 3), [205, 225, 238], dtype=np.uint8)
    else:
        rgb = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
        if rgb.shape[0] != verts.shape[0]:
            raise ValueError("mesh GLB colors must match vertex count")

    position_bytes = verts.astype("<f4", copy=False).tobytes(order="C")
    color_bytes = rgb.astype(np.uint8, copy=False).tobytes(order="C")
    index_bytes = idx.astype("<u4", copy=False).tobytes(order="C")
    position_offset = 0
    color_offset = len(_align4_bin(position_bytes))
    index_offset = color_offset + len(_align4_bin(color_bytes))
    binary_blob = _align4_bin(position_bytes) + _align4_bin(color_bytes) + _align4_bin(index_bytes)

    finite_verts = verts[np.all(np.isfinite(verts), axis=1)]
    if finite_verts.size:
        min_pos = [float(x) for x in np.min(finite_verts, axis=0)]
        max_pos = [float(x) for x in np.max(finite_verts, axis=0)]
    else:
        min_pos = [0.0, 0.0, 0.0]
        max_pos = [0.0, 0.0, 0.0]

    gltf = {
        "asset": {"version": "2.0", "generator": "Noesis virtual twin"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "virtual_twin_model_surface_mesh"}],
        "meshes": [
            {
                "name": "virtual_twin_model_surface_mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                        "mode": 4,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "virtual_twin_model_surface_material",
                "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 0.78]},
                "alphaMode": "BLEND",
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": color_offset, "byteLength": len(color_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(verts.shape[0]),
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos,
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5121,
                "count": int(rgb.shape[0]),
                "type": "VEC3",
                "normalized": True,
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5125,
                "count": int(idx.shape[0]),
                "type": "SCALAR",
                "min": [int(np.min(idx))],
                "max": [int(np.max(idx))],
            },
        ],
    }
    json_chunk = _align4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    bin_chunk = _align4_bin(binary_blob)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        handle.write(json_chunk)
        handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
        handle.write(bin_chunk)


def _png_bytes_from_texture(texture: np.ndarray) -> bytes:
    tex = np.asarray(texture, dtype=np.uint8)
    if tex.ndim != 3 or tex.shape[2] not in {3, 4}:
        raise ValueError("texture atlas must be an RGB or RGBA uint8 image")
    import cv2

    if tex.shape[2] == 3:
        encoded_input = tex[:, :, ::-1]
    else:
        encoded_input = tex[:, :, [2, 1, 0, 3]]
    ok, encoded = cv2.imencode(".png", encoded_input)
    if not ok:
        raise ValueError("failed to encode texture atlas as PNG")
    return bytes(encoded)


def write_textured_mesh_glb(
    path: Path,
    vertices: np.ndarray,
    indices: np.ndarray,
    texcoords: np.ndarray,
    texture: np.ndarray,
) -> None:
    verts = np.asarray(vertices, dtype=np.float32).reshape((-1, 3))
    idx = np.asarray(indices, dtype=np.uint32).reshape((-1,))
    uvs = np.asarray(texcoords, dtype=np.float32).reshape((-1, 2))
    if verts.shape[0] == 0 or idx.shape[0] == 0:
        raise ValueError("textured mesh GLB requires non-empty vertices and indices")
    if uvs.shape[0] != verts.shape[0]:
        raise ValueError("textured mesh GLB texcoords must match vertex count")
    if int(np.max(idx)) >= int(verts.shape[0]) or int(np.min(idx)) < 0:
        raise ValueError("textured mesh GLB index buffer references vertices outside the vertex buffer")

    png_bytes = _png_bytes_from_texture(texture)
    position_bytes = verts.astype("<f4", copy=False).tobytes(order="C")
    uv_bytes = uvs.astype("<f4", copy=False).tobytes(order="C")
    index_bytes = idx.astype("<u4", copy=False).tobytes(order="C")
    position_offset = 0
    uv_offset = len(_align4_bin(position_bytes))
    index_offset = uv_offset + len(_align4_bin(uv_bytes))
    image_offset = index_offset + len(_align4_bin(index_bytes))
    binary_blob = _align4_bin(position_bytes) + _align4_bin(uv_bytes) + _align4_bin(index_bytes) + _align4_bin(png_bytes)

    finite_verts = verts[np.all(np.isfinite(verts), axis=1)]
    if finite_verts.size:
        min_pos = [float(x) for x in np.min(finite_verts, axis=0)]
        max_pos = [float(x) for x in np.max(finite_verts, axis=0)]
    else:
        min_pos = [0.0, 0.0, 0.0]
        max_pos = [0.0, 0.0, 0.0]

    gltf = {
        "asset": {"version": "2.0", "generator": "Noesis virtual twin"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "virtual_twin_textured_model_surface_mesh"}],
        "meshes": [
            {
                "name": "virtual_twin_textured_model_surface_mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "mode": 4,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "virtual_twin_rgb_surface_material",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "doubleSided": True,
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/png", "name": "virtual_twin_rgb_atlas"}],
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uv_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": image_offset, "byteLength": len(png_bytes)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(verts.shape[0]),
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos,
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(uvs.shape[0]),
                "type": "VEC2",
                "min": [float(np.min(uvs[:, 0])), float(np.min(uvs[:, 1]))],
                "max": [float(np.max(uvs[:, 0])), float(np.max(uvs[:, 1]))],
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5125,
                "count": int(idx.shape[0]),
                "type": "SCALAR",
                "min": [int(np.min(idx))],
                "max": [int(np.max(idx))],
            },
        ],
    }
    json_chunk = _align4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    bin_chunk = _align4_bin(binary_blob)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        handle.write(json_chunk)
        handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
        handle.write(bin_chunk)


def sample_colors_from_image(image_bgr: np.ndarray | None, pixels: np.ndarray, *, default: tuple[int, int, int] = (210, 210, 210)) -> np.ndarray:
    pix = np.asarray(pixels, dtype=np.int32).reshape((-1, 2))
    if image_bgr is None:
        return np.full((pix.shape[0], 3), default, dtype=np.uint8)
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        return np.full((pix.shape[0], 3), default, dtype=np.uint8)
    xs = np.clip(pix[:, 0], 0, image.shape[1] - 1)
    ys = np.clip(pix[:, 1], 0, image.shape[0] - 1)
    bgr = image[ys, xs, :3]
    return bgr[:, ::-1].copy()


__all__ = [
    "sample_colors_from_image",
    "write_json",
    "write_mesh_glb",
    "write_points_glb",
    "write_points_npz",
    "write_points_ply",
    "write_textured_mesh_glb",
]
