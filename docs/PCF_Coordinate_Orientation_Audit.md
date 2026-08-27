# PCF coordinate and orientation audit

Status: implemented and validated against the deployed Living Room, Family
Room, and Kitchen Scene Priors on 2026-08-15.

This audit covers PCF reconstruction and alignment reviews, Scene Prior build
and runtime serialization, offline floorplan diagnostics, the oai2-fe heatmap
and BEV renderers, and all four depth-drawer 3D representations. It does not
change the canonical DS9.1 runtime graph.

## Result

There is one presentation contract for every room:

- camera-right is positive display X and screen-right;
- height above the floor is positive display Y;
- camera-forward is positive display Z and screen-up;
- a serialized raster has row zero at maximum display Z, rows advance toward
  minimum Z and the camera, and columns advance from minimum to maximum X;
- physical registration accepts only a positive-scale proper Sim(3), and the
  admitted PCF and multi-room paths further require rigid metric scale; and
- the backend-to-camera display matrix is applied only after registration, to
  display positions or raster addressing. It never changes backend geometry,
  pose rotations, registration inputs, or tracking authority.

The deployed OpenCV-camera to camera-ground display matrices have determinant
`-1`. That is the explicit change from camera Y-down to height Y-up while
retaining camera-right and camera-forward. It is a coordinate/presentation
basis, not an admissible metric registration transform. Physical camera poses,
PCF alignments, and cross-room corrections remain proper with determinant
`+1` after scale removal.

## Canonical coordinate chain

| Stage | Contract | Allowed transform and authority |
| --- | --- | --- |
| Raw OpenCV camera | `x` right, `y` down, `z` forward; camera-to-world poses | Proper pose only: orthonormal rotation, determinant `+1`. Authority: model output and calibrated camera pose. |
| PCF local world | First-view-relative DA3/MapAnything metric points and poses | Floor leveling and provider alignment are physical, positive-scale proper transforms. No display flip is allowed here. |
| Accepted room transform | PCF local world to accepted room/backend world | Proper Sim(3); current PCF handoff requires scale `1` within `1e-3`, orthonormal rotation, and determinant `+1` at `tools/mapanything_phone_scan/build_conditioned_scene_prior_bundle.py:182`. |
| `backend_world_m` | Shared metric X/Y/Z, Y-up | Immutable Scene Prior and registration authority. Cross-room corrections are rigid, orthonormal, determinant `+1`; see `tools/mapanything_phone_scan/pcf_multiroom_pose_graph.py:215` and `tools/mapanything_phone_scan/register_pcf_rooms.py:146`. |
| Scene Prior backend grid | Rows increase backend `+Z`; columns increase backend `+X` | Storage addressing only, declared at `noesis/scene_prior_builder.py:724`. No image convention is stored in the NPZ. |
| `camera_local_ground_m` | `+X` camera-right, `+Y` height above floor, `+Z` camera-forward | Presentation/addressing conversion derived only from the calibrated camera. Shared authority is `noesis_core/coordinate_frames.py:31`. Deployed display determinant is `-1`; it must not enter metric registration. |
| Serialized PCF raster | Row zero is max `+Z`; columns are min-to-max `+X` | Raster addressing only, defined at `noesis_core/coordinate_frames.py:21` and implemented at `noesis_core/coordinate_frames.py:133`. |
| Canvas and BEV | Serialized row order is drawn directly | No horizontal/vertical canvas reflection. Metric overlays use `z = max_z - normalized_row * span_z`, for example `oai2-fe/src/components/BevView.tsx:737`. |
| Three.js | Geometry remains in `camera_local_ground_m` | A proper view camera presents `+X` right and `+Z` up. No model or group axis receives a negative scale. Shared camera authority is `oai2-fe/src/lib/cameraGroundPresentation.js:1`. |

The shared Python implementation validates the camera pose, preserves the
recorded OpenCV right/forward axes, transforms positions without touching pose
rotations, and owns raster indexing at
`noesis_core/coordinate_frames.py:59`, `noesis_core/coordinate_frames.py:73`,
`noesis_core/coordinate_frames.py:121`, and
`noesis_core/coordinate_frames.py:133`. Scene Prior, cached scene fusion,
phone diagnostics, alignment previews, and multi-room presentation now all use
that implementation.

## Operation inventory and classification

### Current intentional operations

| Operation | Classification | Disposition and authority |
| --- | --- | --- |
| `np.flip(..., axis=0)` in `noesis/scene_prior_builder.py:1013` | Raster addressing | Retained. The builder's numeric preview rows increase `+Z`; this one named conversion writes a PNG with row zero at max `+Z`. It does not change the backend grid or point geometry. This is the only retained runtime `np.flip`, `flipud`, or `fliplr` in the audited PCF/Scene Prior path. |
| Legacy Scene Prior preview orientation value `row_increases_camera_forward_column_increases_camera_right` | Contract compatibility | Existing immutable revisions remain readable at `noesis_core/contracts/scene_prior.py:80`, but the value described the pre-PNG numeric grid rather than the written preview image. New revisions serialize the canonical row-zero-max-Z value at `noesis/scene_prior_builder.py:1381`. `render_pcf_multiroom_presentation.py:112` accepts the legacy label only to derive the recorded camera basis; it never applies another raster flip. |
| `max_z - row * dz` in Scene Prior serialization (`noesis_core/scene_prior.py:757`, `noesis_core/scene_prior.py:817`, `noesis_core/scene_prior.py:1549`) | Raster addressing | Retained and made the canonical serialized-row rule. Scene fusion uses the same rule at `noesis_core/scene_fusion.py:219`; heightfield vertices use it at `oai2-fe/src/lib/heightfieldModel.js:149`. |
| Pixel-Y inversion in `_write_topdown` at `tools/mapanything_phone_scan/alignment.py:1041` and the raw trajectory preview at `tools/mapanything_phone_scan/inference.py:250` | Raster addressing | Retained. These functions convert mathematical positive Z to image rows after their inputs are in the intended coordinate frame. Alignment/evaluator callers now supply calibrated camera-ground positions. |
| Family-camera display matrix in `render_pcf_multiroom_presentation.py:91` | Coordinate-frame conversion / presentation | Retained and required to have determinant `-1` at line 147. The same matrix is applied in memory to all registered room surfels and both phone paths at lines 414-423; the source NPZ is hashed and never rewritten. |
| Raw-phone review GLB `Rx(pi)` in `inference.py:223` and supplement comparison GLB `Rx(pi)` in `supplement.py:867` | Camera/view presentation | Retained. This is a proper determinant-`+1` OpenCV/phone-local to glTF review authoring rotation. It affects only raw review GLBs. The raw NPZ, poses, PCF alignment, and backend-world artifacts do not consume it. |
| Candidate half-turn at `tools/mapanything_phone_scan/alignment.py:101` | Physical-geometry diagnostic | Retained only as a counterfactual visibility check. If the half-turn explains the cloud, alignment fails and asks for calibration repair at lines 136-152; it never rotates the target or produces a transform. |
| `image_flip` inference at `geometry/depth_source.py:980` | Ray/image-addressing diagnostic | Retained as payload provenance only. `geometry/depth_source.py:1031` states that the already camera-local raster must not consume it. oai2-fe records it in export metadata at `DepthDrawer.tsx:1705` but does not apply it (`DepthDrawer.tsx:1376`); `scripts/dump_floorplan_views.py:827` likewise records it and directly decodes the grid at line 833. |
| DS9.1 ray `_apply_image_flip` helpers in `DS9/noesis/pipelines/hooks.py` | Ray/image addressing | Retained for call-shape compatibility, but both authoritative inference methods return `(False, False)`. The shared BEV presentation helper in `noesis/telemetry/bev.py` does the same. |
| Positive `ctx.scale(dpr, dpr)` and identity `ctx.setTransform(...)` in oai2-fe | Camera/view presentation | Retained. These are device-pixel scaling and canvas-state reset, not reflections. Examples: `renderUtils.ts:315`, `renderUtils.ts:489`, `extrudedFloorplan.ts:375`, and `extrudedFloorplan.ts:437`. There is no negative canvas scale in the audited renderers. |
| `np.transpose(HWC, CHW)` at `tools/mapanything_phone_scan/da3_inference.py:186` and `noesis/mapanything_manual_inference.py:152` | Tensor layout | Retained. These transpose channel/storage order for inference; they do not change image or world orientation. The remaining `.T` uses in the audited geometry code are matrix/vector linear algebra, not image transposition. |
| `geometry.rotateX(-Math.PI / 2)` at `oai2-fe/src/components/Depth3DView.tsx:364` | Coordinate-frame conversion | Retained. It is a proper determinant-`+1` rotation that lays a Three.js XY `PlaneGeometry` onto XZ; it is in the separate, currently unmounted legacy `Depth3DModal`, not one of the four Depth drawer PCF subtabs. |

The other transpose hits under `DS9/scripts/` and model build/export utilities
are likewise explicit HWC/CHW tensor-layout conversions (or their inverse for
an RGB preview). None is called by Scene Prior raster serialization or oai2-fe
presentation, and none was changed.

The remaining syntactic reversal/rotation hits are non-spatial: reverse-bin
searches at `noesis_core/scene_fusion.py:365` and
`noesis_core/scene_prior.py:964`; BGR-to-RGB channel reversal at
`noesis/virtual_twin/artifacts.py:400` and
`noesis/virtual_twin/builder.py:1917`; asymmetric synthetic test-image
generation at `tests/test_ma_service.py:147` and line 172; a vertical plot label
at `evaluate_mapanything_prior_variants.py:413`; and the disclosure-arrow CSS
at `oai2-fe/src/styles/depth-drawer.css:641`. None transforms metric geometry,
raster addressing, or a camera/view presentation.

### Removed inconsistent operations

| Removed operation | Classification | Replacement |
| --- | --- | --- |
| Evaluator `_camera_oriented_bev`, `np.rot90(..., 2)`, and the hard-coded Living Room 180-degree convention | Camera/view presentation | Removed. `evaluate_mapanything_prior_variants.py:616` derives one display matrix from the selected calibrated camera after metric evaluation, and lines 699-783 render every candidate directly in that frame. Absence of the room-specific rotation is guarded at `test_prior_variant_evaluation.py:39`. |
| `rotate_180` in point-preserving heatmap generation | Camera/view presentation | Removed. `_present_camera_ground` changes display positions only (`render_phone_heatmap_diagnostics.py:242`), and `_rasterize` / `_point_splat` use canonical row indices directly. Asymmetric probes are tested at `test_point_preserving_heatmap.py:55` and line 79. |
| Backend-world alignment GLB `Rx(pi)` | Camera/view presentation mixed into a metric artifact | Removed. Already aligned GLBs now export their unchanged `backend_world_m` vertices at `alignment.py:1034`; the viewer owns presentation. The separate alignment PNG is converted after all metrics at `alignment.py:1628`. |
| `group.scale.x = -1` in ScenePriorPointCloud3DView, FusedPointCloud3DView, and CachedHeightfield3DView | Camera/view presentation / mirror | Removed. All three use the same unmirrored camera direction at `ScenePriorPointCloud3DView.tsx:95`, `FusedPointCloud3DView.tsx:134`, and `CachedHeightfield3DView.tsx:109`. The Visible Floor camera uses the same screen basis at `FloorPlane3DView.tsx:55` and line 61. |
| `flipHorizontal` / `flipVertical` and negative canvas scaling in `renderUtils` | Camera/view presentation / mirror | Removed. `RenderLayerOptions` starts at `renderUtils.ts:185` and has no flip fields; the texture renderer draws the serialized raster directly. |
| `dump_floorplan_views --apply-image-flip`, `--no-flip`, and array slicing flips | Camera/view presentation / mirror | Removed. The tool decodes each layer exactly once at `scripts/dump_floorplan_views.py:833`. |

There are no current room-specific/manual orientation transforms. The
multi-room path rejects reflected, scaled, sheared, or visually nudged
registration. Family Room remains the fixed metric gauge and Kitchen receives
only its evidence-derived proper correction. Selecting Family as the review
camera changes only the shared post-registration display basis; it does not add
a Family-only or Kitchen-only geometry transform.

## 2D and 3D presentation authority

- Heatmap and textured/structural floorplan canvases consume the serialized
  row-zero-max-Z grids directly. The camera marker maps Z=`0` through the same
  bounds at `oai2-fe/src/components/DepthDrawer.tsx:754`.
- Inline BEV maps normalized row Y back through `max_z - normY * spanZ` at
  `oai2-fe/src/components/BevView.tsx:737`; no `image_flip` is applied.
- Obstacles use the pure row/column isometric projection at
  `oai2-fe/src/lib/cameraGroundPresentation.js:38`. Decreasing row (farther
  forward) moves upward, and increasing column moves right.
- Heightfield reconstructs row zero at max Z at
  `oai2-fe/src/lib/heightfieldModel.js:149`.
- Point Cloud applies the backend-to-camera display matrix only to its loaded
  in-memory GLB at `ScenePriorPointCloud3DView.tsx:236`, then uses the shared
  Three camera without a model reflection.
- Visible Floor keeps its metric vertices unchanged and places the camera below
  the plane with up=`+Z` at `FloorPlane3DView.tsx:55` and line 61.

The asymmetric JavaScript gates at
`oai2-fe/src/lib/heightfieldModel.test.mjs:53` and
`oai2-fe/src/lib/pcfPresentation.test.mjs:18` check all four current 3D
representations plus the retained fused-cloud component. They fail on either a
left/right mirror or a 180-degree reversal.

## Deployed-room validation

`scripts/validate_pcf_orientation.py:79` loads the current Scene Prior catalog
and calibrated extrinsics, requires proper camera rotations, probes camera,
camera-right, and camera-forward landmarks, validates row-zero-max-Z indexing,
and requires the camera to lie in the bottom 15 percent of each complete PCF
raster.

| Camera | Scene Prior | Raster rows x columns | Camera row fraction from top | Metric E determinant | Display determinant | Axis probes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Living Room | `sceneprior_living-room_20260802T202254Z_737f02e4f303` | 391 x 466 | 0.9437 | +1.000000 | -1.000000 | camera `(0,0)`, right `(1,0)`, forward `(0,1)` |
| Family Room | `sceneprior_family-room_20260811T015847Z_8b80dc69a7c4` | 291 x 517 | 0.9313 | +1.000000 | -1.000000 | camera `(0,0)`, right `(1,0)`, forward `(0,1)` |
| Kitchen | `sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483` | 448 x 698 | 0.8929 | +1.000000 | -1.000000 | camera `(0,0)`, right `(1,0)`, forward `(0,1)` |

The validation command was:

```bash
python3 scripts/validate_pcf_orientation.py \
  --presentation-manifest \
  "$PCF_PRESENTATION_MANIFEST"
```

The supplied reference manifest passed these additional checks:

- its display matrix is exactly the Family Room Scene Prior camera-ground
  matrix and has determinant `-1`;
- camera-right maps to `(1, 0)` and camera-forward maps to `(0, 1)`;
- the source backend NPZ hash and size are identical before and after export,
  no transformed NPZ was written, and all 1,075,656 surfels remained present;
- Family and Kitchen reintegration corrections are proper with determinant
  `+1`; and
- the evidence remains `review_only` with
  `accepted_for_canonical_use: false`. Correct presentation does not upgrade
  failed cross-session admission evidence.

## Verification record and known harness limits

- Python syntax and Ruff checks passed for every changed Python surface.
- 39 focused tests passed: 33 coordinate, Scene Prior, scene-fusion, heatmap,
  evaluator, and legacy-manifest compatibility tests plus six multi-room
  pose-graph/presentation tests. The
  one deselected Scene Prior HTTP test blocks in the current
  shared Starlette TestClient/AnyIO portal before the endpoint runs. An
  unrelated REST boundary test reproduces the same block.
- The three oai2-fe orientation/render test files passed, including asymmetric
  heightfield, obstacle, camera-basis, no-negative-scale, and no-canvas-flip
  assertions; the Vite production build passed.
- The multi-room presentation self-test and the three-room/reference validator
  passed.
- The legacy depth-storage `image_flip` integration test currently blocks in
  Zarr `open_group` during snapshot storage. The pure diagnostic hint helper
  was exercised directly and passed; no orientation code is reached by the
  blocked storage test.
