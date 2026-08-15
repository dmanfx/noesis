# Virtual Twin Reconstruction
_Status: first-pass living-room workflow, validated against the active builder on 2026-05-03._

The virtual twin is a Noesis-owned offline reconstruction bundle. Menon does not
solve dense depth projection in the browser for this path; it renders the
versioned GLB/JSON artifacts that Noesis writes under
`data/virtual_twin/revisions/<revision_id>/`.

## What The Builder Uses

The living-room builder combines four evidence sources:

- RGB keyframes captured from the configured DS9.1 camera source.
- DS9.1-persisted MapAnything Zarr snapshots from `data/depth/<camera>/...`.
- ZeroPlane plane masks/normals/depth estimates for the same keyframes.
- Menon structural surfaces parsed from the SweetHome3D OBJ.

The builder triggers `POST /api/v1/depth/refresh?seconds=N` on the DS9.1 runtime
REST API before each keyframe unless `--no-mapanything-refresh` is explicitly
passed. In the split dev setup, the standalone virtual-twin artifact API uses
port `8080` and the DS9.1 runtime depth API uses port `8082`; override the
refresh URL with `--mapanything-refresh-url` when those ports differ. That means
the virtual twin consumes the same MapAnything path that the DS9.1 pipeline
persists to disk; it does not switch to the always-on DAv2 tracking lane as a
geometry source.

## Geometry Flow

For each keyframe, MapAnything depth pixels are back-projected through the
camera intrinsics, then fused with ZeroPlane plane instances. Pixels that belong
to a strong plane are replaced by the fitted plane intersection geometry, while
lower-confidence non-planar samples remain as surfels for diagnostics.

The builder also computes camera-space normals from the same MapAnything depth,
mask, confidence, and calibration evidence. These normals are supporting
evidence only: they score, down-rank, or reject ZeroPlane/MapAnything plane
candidates when dense depth gradients disagree with the fitted plane normal,
but they do not replace the stream-derived plane equation with Menon model
geometry. Each revision persists `normals_camera` and `normals_valid` arrays in
the per-frame `mapanything/*.npz` evidence, and `planes.json` schema v2 adds
per-plane `normal_support`, `depth_support`, and `fusion_score` fields.

The fused points and planes stay in `backend_world_m` until registration. Plane
centroids/normals are matched against Menon structural surfaces, using the
existing Menon scene similarity as the explicit prior. The output correction is
persisted in `tracking_alignment.json`; it is not hidden in browser state or
authored camera edits.

## Surface Artifact

`surfaces.glb` is the browser-facing artifact. It is a model-surface mesh in
Menon scene units, not a floating point cloud. Dense fused points remain in
`points.ply` and `points.npz` for analysis.

The surface-selection pass works by transforming fused MapAnything+ZeroPlane
support into Menon scene space, then assigning each source pixel only to the
front-most Menon structural surface visible at that same camera pixel. The
builder still checks that the fused point is close to that visible surface, but
it no longer lets a point choose an arbitrary nearby wall behind the intended
geometry. This is why the first pass can adhere to the room geometry while
avoiding broad RGB bleed across unrelated structural blocks.

## RGB Texture Bake

The RGB texture atlas is baked in Noesis:

1. Supported Menon model triangles receive UVs into an embedded PNG atlas.
2. Atlas texels are sampled as points on those model triangles.
3. Each texel is projected back through the revision's scene-to-world
   registration and into each saved RGB keyframe.
4. A texel is filled only when the corresponding MapAnything depth/confidence
   gate agrees with the projected model point.
5. Filled texels receive the explicit exposure/gamma/contrast tone map recorded
   in `metrics.browser_render_budget.texture`.

The builder rejects effectively grayscale keyframes by default. If the camera is
in IR/night mode, the job fails closed instead of producing a grayscale artifact
that looks like an RGB reconstruction. `--allow-grayscale-texture` is reserved
for explicit diagnostic bundles.

## Surface Fit Tightening

The first RGB textured revision proved that the right structural surfaces were
selected and that the model-surface GLB renders cleanly in Menon. The remaining
fit work is texture registration: the selected surfaces are right, but the RGB
evidence can land with a small yaw/scale/translation error if the global fit is
oversized or slightly rotated.

The builder now performs two surface-aware tightening passes:

- it rasterizes the Menon structural OBJ from the camera pose into a coarse
  front-most visible-surface map, then uses image-pixel-to-surface
  correspondences to solve a bounded rotation/scale/translation correction;
- it refines the result against dense point-to-structural-surface support before
  the texture bake;
- each atlas texel is colored only by source pixels that both back-project onto
  the same Menon surface and are assigned to that surface in the front-most
  visibility map.

This keeps the useful part of the current method, where depth selects the right
geometry, while reducing the broad-projector effect where pixels from one side
of the room can bleed onto another structural block.

## Operational Build

Typical living-room command:

```bash
python3 scripts/build_virtual_twin_reconstruction.py \
  --camera living-room \
  --model-obj "$NOESIS_MENON_STRUCTURAL_OBJ" \
  --keyframes 4 \
  --frame-stride 30 \
  --max-frames-read 900 \
  --browser-point-budget 150000 \
  --texture-tile-px 96 \
  --min-texture-coverage 0.02
```

The revision catalog remains available through
`/api/v1/virtual-twin/latest` for operator diagnostics. Production rendering is
owned by one promoted `noesis.scene.release` from
`/api/v1/scenes/current/payload`; it never assembles a home from independently
newest camera revisions. Each promoted release closes over the exact manifest,
all declared camera artifacts, the authored OBJ, all referenced MTL and texture
dependencies, and its validation report by release-owned path, byte length, and
SHA-256. `current/payload` exposes role-addressed same-origin URLs for that
complete set. Current-release reads revalidate every byte and fail with conflict
status if an on-disk artifact has changed. Promotion, current metadata, and the
current payload validate the complete cohort. A role-addressed binary request
then verifies only its selected file and serves the verified in-memory snapshot,
so a path replacement cannot change bytes after validation and a 34-request
consumer does not rehash the full scene 34 times.

The release boundary rejects symlinked, hardlinked, non-regular, oversized,
zero-byte, or replaced-during-read inputs. OBJ/MTL parsing is UTF-8 strict and
bounded by file, line, and dependency counts; unknown MTL map options fail
instead of being guessed. A new release-owned directory is assembled off-path
and published with an atomic no-replace rename. Existing directories must match
the exact declared tree and are never overwritten or permission-repaired.

The current unpromoted migration candidate
`home_rgbmesh_20260623T2158_v1` selects the family-room, kitchen, and
living-room revisions captured on 2026-06-23. Its validation report proves 27
camera artifacts plus the authored OBJ, one MTL, and four JPEG textures. It is
intentionally a candidate until the Menon atomic-load consumer and browser
readback gates pass; building a candidate never implies promotion.
