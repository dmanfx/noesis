# Same-geometry Living Room texture trial

Status: completed offline presentation comparison, 2026-10-07. This follows
the [browser-sized mesh export](README.md), not a new reconstruction or live
scene replacement. The user prioritizes visual appearance over more triangles.

## What changed and what did not

Reuse the existing high-detail export's 400,000 oriented triangles. Verify its
saved simplified geometry against the bound raw GLB, then verify every new UV
layout with a canonical world-space triangle digest. UV seams may duplicate
vertices; world coordinates and winding must not change. Final packaging retains
the original 16-bit position/UV and 8-bit normal quantization settings.

Atlas packing uses two-pixel xatlas gutters instead of eight. With the same
4096-square texture, actual surface samples increase from 20.18% to 46.24% of
the atlas, or 2.29 times as many texels. This is texel utilization, not measured
image-quality improvement or a physical-accuracy claim. Keeping only two pixels
of outside-chart colour dilation exposed hairline seams; restore eight-pixel
outside-chart dilation without changing covered pixels. The narrow-dilation
trial is retained as a diagnostic, not selected as the preview default.

Direct-photo variants reuse the exact existing 130-view, 2560×1440 native-frame
photo cohort. Source hashes, original model-ray preprocessing, camera poses,
source colour witnesses and previously accepted photometric gains are checked.
The existing blur/luminance gates and raw-provider depth visibility remain in
force. No same-mesh visibility substitution is introduced. Select a bounded
four-witness pool per texel from the closest dense source triangle; sample the
photos in linear light and either choose the highest-weight valid photo or use
the existing robust colour blend. Closest-point sampling changes only colour,
never the retained renderer geometry.

In both photo modes, 88.07% of covered texels have an eligible direct photo
sample. The other 11.93% retain explicitly recorded existing source colour;
there is no image inpainting or invented observation. This texel percentage is
not comparable to the earlier per-vertex direct-colour percentage. Tiny UV
ambiguities remain neutral (237 texels, 0.00305%); the existing 0.1% rejection
limit is unchanged. Each variant has one embedded JPEG, unlit material and draw.

## Results and limitations

All variants use the same 400,000-triangle geometry and 4K atlas. Sizes are
decimal MB, not decoded GPU memory.

| Texture variant | Compact GLB | Compatible GLB | Visual review |
| --- | ---: | ---: | --- |
| Original colour / original packing | 8.24 MB | 15.58 MB | Baseline |
| Packing test / narrow gutters | 10.19 MB | 17.53 MB | Hairline seams; diagnostic |
| Original colour / improved packing | 9.20 MB | 16.53 MB | Modest extra detail, original colour blotches |
| Direct photographs / sharpest view | 10.84 MB | 18.17 MB | Stronger view-to-view colour patches; diagnostic |
| Direct photographs / robust blend | 9.69 MB | 17.02 MB | Selected review default; finer detail, residual patches |

Matched fixed/phone-camera screenshots show a modest photographic-detail
improvement, particularly around windows and stonework. Geometry roughness,
colour discontinuities and incomplete photographic coverage remain plainly
visible; this is not photorealistic. The sharpest-view variant also retains the
narrow outside-chart dilation, so it is not a controlled blend-only comparison.
New assets use JPEG quality 85; the retained baseline uses the prior encoder
settings. Neither file sizes nor atlas occupancy isolate perceptual quality.

All five actual Three.js assets passed browser loading and resource-disposal
checks. Menon's existing embedded-image loader accepted the compatible
photo-blend asset without product changes. Chrome 150 on Intel UHD 630 hardware
WebGL reached the approximately 60 fps display cap in three-second orbit tests
for every variant, including the baseline. Decoded geometry plus texture stays
approximately 100.80 MB. Initial loading fetches only the selected photo-blend
asset; the 412-pixel mobile layout has no horizontal overflow. No browser console
errors or warnings were observed. These are bounded local-consumer results,
not a performance or cross-device guarantee.

The source remains review-only and inherits approximately 23.64% inferred
surface area. No physical geometry qualification, tracking/world authority,
RoomWalk utility change, Menon integration, inference rerun or runtime restart
follows from better texture. Additional images were not needed for this trial.

## Reproduce

Use a new external output directory. The baseline must be a completed export
from the original workflow, including `detail/mesh_manifest.json`, bound raw
GLB and saved simplified geometry. Reuse its local Python/npm dependencies.
Source manifests locate the bound photographs; there is no recursive discovery
or automatic replacement photo selection. Run from the repository root:

```bash
export ROOM_TEXTURE_BASE=/path/to/existing-room-mesh-trial
export ROOM_TEXTURE_ROOT=/path/to/new-room-texture-trial
export ROOM_TEXTURE_PYTHON="$PWD/data/mapanything_phone_scan_runtime/venv/bin/python"
export ROOM_TEXTURE_MENON=/path/to/Menon
export ROOM_TEXTURE_SOURCE=/path/to/final_closed_video_color_v2/surface_mesh.glb
export PYTHONPATH="$ROOM_TEXTURE_BASE/python-deps:."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2

nice -n 10 "$ROOM_TEXTURE_PYTHON" tools/room_mesh_export/refine_texture.py \
  --base-dir "$ROOM_TEXTURE_BASE/detail" --output-dir "$ROOM_TEXTURE_ROOT/packed" \
  --atlas-padding 2 --bake-padding 2 --mode vertex

nice -n 10 "$ROOM_TEXTURE_PYTHON" tools/room_mesh_export/refine_texture.py \
  --base-dir "$ROOM_TEXTURE_BASE/detail" --output-dir "$ROOM_TEXTURE_ROOT/packed-padded" \
  --layout-from "$ROOM_TEXTURE_ROOT/packed" --bake-padding 8 --mode vertex

nice -n 10 "$ROOM_TEXTURE_PYTHON" tools/room_mesh_export/refine_texture.py \
  --base-dir "$ROOM_TEXTURE_BASE/detail" --output-dir "$ROOM_TEXTURE_ROOT/photo-best" \
  --layout-from "$ROOM_TEXTURE_ROOT/packed" --bake-padding 2 --mode photo-best

nice -n 10 "$ROOM_TEXTURE_PYTHON" tools/room_mesh_export/refine_texture.py \
  --base-dir "$ROOM_TEXTURE_BASE/detail" --output-dir "$ROOM_TEXTURE_ROOT/photo-blend" \
  --layout-from "$ROOM_TEXTURE_ROOT/packed" --bake-padding 8 --mode photo-blend

node tools/room_mesh_export/package_texture_trial.mjs \
  --trial-root "$ROOM_TEXTURE_ROOT" --baseline-root "$ROOM_TEXTURE_BASE" \
  --gltf-cli "$ROOM_TEXTURE_BASE/js-tools/node_modules/.bin/gltf-transform" \
  --default-asset photo-blend

python3 tools/room_mesh_export/serve_trial.py --trial-root "$ROOM_TEXTURE_ROOT" \
  --three-root "$ROOM_TEXTURE_MENON/node_modules/three" --source-glb "$ROOM_TEXTURE_SOURCE" \
  --menon-loader "$ROOM_TEXTURE_MENON/src/three/BoundedGltfLoader.js" --port 8842
```

The preview is loopback-only at `http://127.0.0.1:8842`, not an installed service.
Use the Texture selector and a shared camera to compare variants. Compact files
are `room-web-meshopt.glb`; compatible files are `room-web-compatible.glb`.

```bash
"$ROOM_TEXTURE_PYTHON" -m pytest -q tools/room_mesh_export/test_browser_mesh.py \
  tools/room_mesh_export/test_texture_refinement.py
node tools/room_mesh_export/validate_browser.mjs \
  --dependency-root "$ROOM_TEXTURE_MENON" --browser-executable /path/to/chrome \
  --output-dir "$ROOM_TEXTURE_ROOT/browser-checks" --url http://127.0.0.1:8842 \
  --compatible-asset photo-blend
```

## Higher-resolution depth: inspected, not run

The retained DA3-BASE run uses `process_res=504` with `upper_bound_resize`,
yielding 504×280 depth grids. Conditioned MapAnything uses 518×294 inputs, and
the fused common grid follows DA3's actual depth shape. A larger output buffer
or interpolation of those saved arrays would not add inferred detail.

A genuine larger-grid experiment needs higher-resolution DA3 inference,
matching conditioned MapAnything preprocessing/prior bindings and rebuilt
consensus. DA3 already exposes the process-resolution control in
`tools/mapanything_phone_scan/da3_inference.py` and `app.py`; the conditioned
runner currently uses default MapAnything preprocessing and does not expose the
matching larger-grid setting. A 756×420 DA3 candidate has 2.25 times the current
pixel count, but its quality and GPU capacity are untested. The
[official DA3 CLI documentation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/CLI.md)
describes the process-resolution control; it does not establish this experiment's
memory safety or reconstruction quality.

The separate DA3Metric TensorRT profile is fixed at batch 3, 518×294. It supplies
scale for the DA3 depth rather than replacing its spatial grid, so increasing
DA3-BASE resolution does not by itself require enlarging that metric engine.
An actual higher-resolution metric profile would require host engine work.
The existing large MapAnything window already has tight GPU headroom; any
larger-grid trial needs a separately authorized exclusive GPU window and bounded
capacity measurement. No service interruption or new inference was performed
for this texture comparison.
