"""Direct photo-to-atlas colour sampling using the existing verified witnesses.

Reuse exact frame/ray bindings, fitted gain acceptance and provider visibility.
No new photo, relaxed gate, inpainting or geometry correction is introduced.
"""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import distance_transform_edt

from tools.mapanything_phone_scan import texture_single_room_review_surface as texture
from tools.mapanything_phone_scan.bake_continuous_video_color import robust_video_colors
from tools.room_mesh_export.build_browser_mesh import linear_to_srgb, record, uv_raster


def witness_candidates(ids, weights, bary, top_k=4):
    """Combine duplicate corner witnesses before selecting a bounded photo pool."""
    if (ids.ndim != 3 or ids.shape[1:] != (3, 4) or ids.shape != weights.shape
            or bary.shape != (len(ids), 3) or top_k not in (1, 4)):
        raise ValueError("invalid bounded photo witness shape")
    ids = ids.reshape(-1, 12)
    weighted = (weights * bary[:, :, None]).reshape(-1, 12)
    scores = np.zeros_like(weighted)
    for column in range(12):
        same = ids == ids[:, column:column+1]
        first = ~same[:, :column].any(1)
        scores[:, column] = np.where(first & (ids[:, column] >= 0), (weighted * same).sum(1), 0)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    selected_scores = np.take_along_axis(scores, order, axis=1)
    selected_ids = np.take_along_axis(ids, order, axis=1)
    selected_ids[selected_scores <= 0] = -1
    return selected_ids.astype(np.int16), selected_scores.astype(np.float32)


def bilinear_rgb(photo, uv):
    """Bilinear photographic sampling in linear light, without RGB synthesis."""
    if uv.shape != (len(uv), 2) or not np.isfinite(uv).all():
        raise ValueError("non-finite photo coordinates")
    low = np.floor(uv).astype(np.int32)
    if (np.any(low < 0) or np.any(low[:, 0] + 1 >= photo.shape[1])
            or np.any(low[:, 1] + 1 >= photo.shape[0])):
        raise ValueError("photo sampling outside valid image bounds")
    fraction = uv - low
    colour = np.zeros((len(uv), 3), np.float32)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        weight = (fraction[:, 0] if dx else 1-fraction[:, 0]) * (fraction[:, 1] if dy else 1-fraction[:, 1])
        colour += texture._linear_rgb(photo[low[:, 1]+dy, low[:, 0]+dx]) * weight[:, None]
    return colour


def verified_views(source_manifest):
    source = source_manifest["source"]
    prepared = Path(source["prepared_manifest"]["path"])
    consensus = Path(source["consensus_manifest"]["path"])
    alignment = Path(source["alignment_transform"]["path"])
    photos = Path(source["bound_native_texture_photos"]["manifest"]["path"])
    gains_path = Path(source["photometric_manifest"]["path"])
    for name, path in (("prepared_manifest", prepared), ("consensus_manifest", consensus),
                       ("alignment_transform", alignment), ("photometric_manifest", gains_path)):
        texture._bound_file(source[name], path, name)
    texture._bound_file(source["bound_native_texture_photos"]["manifest"], photos, "native photos")
    args = Namespace(scan_dir=prepared.parent, consensus_dir=consensus.parent, alignment_dir=alignment.parent,
                     texture_photo_dir=photos.parent, max_input_pixels=16_000_000, max_views=256,
                     texture_edge_px=2560, minimum_texture_blur=100., minimum_luminance=20.)
    views, provenance = texture.load_bound_views(args)
    gains_manifest = texture._json(gains_path)
    original_geometry = Path(source["color_source_surface"]["path"])
    texture._bound_file(gains_manifest["source"]["mesh"], original_geometry, "fitted gain source geometry")
    if len(gains_manifest["views"]) != len(views):
        raise ValueError("photometric cohort changed")
    fields = ("frame_id", "source_frame_index", "prepared_photo", "texture_photo", "conditioned_provider_raw",
              "consensus_raw", "provider_intrinsics", "common_intrinsics", "camera_to_world", "model_to_photo_row_major")
    for old, view in zip(gains_manifest["views"], views, strict=True):
        if any(old.get(key) != view.provenance.get(key) for key in fields):
            raise ValueError("photo-to-atlas gain frame/ray binding changed")
    return views, gains_manifest["photometric_normalization"].get("applied_gains", {}), provenance


def bake_photos(source_scene, source_faces, source_colours, source_manifest,
                vertices, faces, uvs, edge, threads, padding_pixels, *, mode="photo-best"):
    views, gains, provenance = verified_views(source_manifest)
    witness_path = Path(source_manifest["color_bake"]["color_sources"]["path"])
    texture._bound_file(source_manifest["color_bake"]["color_sources"], witness_path, "photo witnesses")
    with np.load(witness_path, allow_pickle=False) as raw:
        source_ids, source_weights = raw["source_view_index"].copy(), raw["source_weight"].copy()
    if (source_ids.shape != source_weights.shape or source_ids.shape != (len(source_colours), 4)
            or np.any((source_ids < -1) | (source_ids >= len(views))) or np.any(source_weights < 0)
            or not np.isfinite(source_weights).all()):
        raise ValueError("saved source witnesses exceed the exact bounded cohort")
    face_map, bary_map, ambiguous, raster_report = uv_raster(uvs, faces, edge, diagnostics=True)
    covered = face_map >= 0
    pixels = np.flatnonzero((covered & ~ambiguous).reshape(-1))
    if len(pixels) > 16_777_216:
        raise ValueError("photo atlas exceeds bounded sampling work")
    # Keep four original candidate witnesses even for best-view mode, allowing
    # an alternative only when the per-texel original depth gate rejects a view.
    positions = np.zeros((len(pixels), 3), np.float32)
    candidates = np.full((len(pixels), 4), -1, np.int16)
    weights = np.zeros((len(pixels), 4), np.float32)
    fallback = np.zeros((len(pixels), 3), np.uint8)
    flat_bary, flat_faces = bary_map.reshape(-1, 2), face_map.reshape(-1)
    for start in range(0, len(pixels), 131_072):
        chunk = pixels[start:start+131_072]
        bary = flat_bary[chunk]
        triangle = vertices[faces[flat_faces[chunk]]]
        point = (triangle[:, 0]*bary[:, :1] + triangle[:, 1]*bary[:, 1:]
                 + triangle[:, 2]*(1-bary.sum(1))[:, None]).astype(np.float32)
        hit = source_scene.compute_closest_points(o3d.core.Tensor(point), nthreads=threads)
        dense_faces = source_faces[hit["primitive_ids"].numpy()]
        dense_uv = hit["primitive_uvs"].numpy()
        dense_bary = np.column_stack((1-dense_uv.sum(1), dense_uv))
        ids, score = witness_candidates(source_ids[dense_faces], source_weights[dense_faces], dense_bary)
        end = start + len(chunk)
        positions[start:end] = hit["points"].numpy()
        candidates[start:end], weights[start:end] = ids, score
        colour = (source_colours[dense_faces]*dense_bary[:, :, None]).sum(1)
        fallback[start:end] = np.rint(linear_to_srgb(colour)*255).clip(0, 255).astype(np.uint8)
    del bary_map, flat_bary, flat_faces, source_ids, source_weights
    samples = np.zeros((len(pixels), 4, 3), np.uint8)
    accepted_weights = np.zeros_like(weights)
    candidate_count = int(np.count_nonzero(weights))
    for view in views:
        if view.model_to_photo is None:
            continue
        rows, slots = np.nonzero((candidates == view.index) & (weights > 0))
        if not len(rows):
            continue
        gain = np.asarray(gains.get(str(view.index), [1., 1., 1.]))
        if gain.shape != (3,) or not np.isfinite(gain).all() or np.any((gain < .67) | (gain > 1.5)):
            raise ValueError("gain exceeds existing accepted bounds")
        with Image.open(view.photo_path) as photo:
            if photo.size != view.photo_size:
                raise ValueError("exact bound native photo was unexpectedly resized")
            rgb = np.asarray(photo.convert("RGB"))
        for start in range(0, len(rows), 32_768):
            selected, slot = rows[start:start+32_768], slots[start:start+32_768]
            valid, quality = texture.depth_visible(positions[selected], view)
            chosen, slot = selected[valid], slot[valid]
            if len(chosen):
                _, _, uv = texture.project_photo(positions[chosen], view)
                colour = bilinear_rgb(rgb, uv) * gain
                samples[chosen, slot] = texture._encoded_rgb(colour)
                accepted_weights[chosen, slot] = weights[chosen, slot] * quality[valid]
        print(f"direct photo {view.index + 1}/{len(views)}: {len(rows):,} candidate texels", flush=True)
    valid_count = np.count_nonzero(accepted_weights > 0, axis=1)
    if mode == "photo-best":
        # Preserve original per-vertex camera-quality ranking; do not blend
        # shifted fine photographic edges from different camera viewpoints.
        best = accepted_weights.argmax(1)
        good = valid_count > 0
        colour = fallback.copy()
        colour[good] = samples[np.flatnonzero(good), best[good]]
    elif mode == "photo-blend":
        colour, _ = robust_video_colors(samples, accepted_weights, fallback)
    else:
        raise ValueError("unknown direct photo selection mode")
    image = np.full((edge*edge, 3), 128, np.uint8)
    image[pixels] = colour
    image = image.reshape(edge, edge, 3)
    known = covered & ~ambiguous
    distance, nearest = distance_transform_edt(~known, return_indices=True)
    padding = (~covered) & (distance <= padding_pixels)
    image[padding] = image[nearest[0][padding], nearest[1][padding]]
    report = {"filled_texels": len(pixels), "fraction": len(pixels)/(edge*edge),
        "padding_texels": int(padding.sum()), "uv_raster": raster_report,
        "colour_transfer": "direct_bound_native_photo_linear_bilinear_then_srgb",
        "photo_selection": mode, "photo_visibility": "original_provider_depth_gate_unchanged",
        "direct_photo_texels": int(np.count_nonzero(valid_count)),
        "direct_photo_fraction_of_covered": float(np.mean(valid_count > 0)),
        "existing_vertex_colour_fallback_texels": int(np.count_nonzero(valid_count == 0)),
        "photo_candidate_samples": candidate_count,
        "accepted_photo_samples": int(np.count_nonzero(accepted_weights)),
        "unobserved_geometry_added": False, "image_inpainting": False,
        "world_sample_policy": "nearest_exact_dense_surface_not_mesh_geometry_change",
        "source_witnesses": record(witness_path), "photo_bindings": provenance,
        "photometric_manifest": source_manifest["source"]["photometric_manifest"]}
    return Image.fromarray(image), report
