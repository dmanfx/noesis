# DS8 SV3DT Conventions Matrix (Golden Baseline)

Purpose: establish a single source of truth for coordinate frames, units, and I/O
expectations across DS8 SV3DT modules so diagnostics can verify each contract.

## Golden Baseline (SV3DT only, DS8 only)

- Pipeline: `config/infer_v3dt_baseline.yaml`
- Cameras (intrinsics): `config/cameras_v3dt_baseline.yaml`
- Extrinsics: `config/camera_calibration.json`
- Dewarper: `config/dewarper_v3dt_baseline.txt`
- Tracker config: `config/v3dt/nvtracker_v3dt_baseline.yml`
- camInfo dir: `config/v3dt/caminfo_baseline/`

## Canonical Frames and Units (DS8)

- World frame: Y-up, meters.
- Extrinsics E: world->camera, column-major (Fortran order).
- Intrinsics K: from `config/cameras*.yaml`, scaled to streammux resolution.
- Alignment: `config/ply_alignment.json` matrix is row-major; `floor_y` is in meters;
  `units.s_obj_to_m` is a scale applied to translations (must be 1.0 for meter extrinsics).

Source: `docs/DS8_api_contracts_ws.md`
> Semantics: 4x4 world->camera transform
> Layout: Flattened to 16 floats, column-major (Fortran order)
> World frame: Y-up
> Units: Meters
> Alignment: matrix is row-major; floor_y is meters; units.s_obj_to_m is scale
> pixel_to_world does NOT apply align.matrix; MapAnything pose conditioning applies it.

MapAnything camera frame is OpenCV cam->world with +X right, +Y down, +Z forward.

Source: `noesis/metadata/mapanything_pose.py`
> MapAnything expects per-view camera_poses in OpenCV cam->world convention:
> +X right, +Y down, +Z forward (camera frame), 4x4 T_wc.

## Module-by-Module Conventions

### 1) nvstreammux (stream scaling)

- Input: per-source decoded frames at their native resolution.
- Output: frames scaled to `streammux.width` x `streammux.height`.
- If `enable-padding=1`, aspect ratio is preserved via black borders.

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html
> width: If non-zero, muxer scales input frames to this width.
> enable-padding: Maintains aspect ratio by padding with black borders when scaling input frames.

### 1b) PGIE (YOLO) preproc — aspect ratio (SV3DT critical)

- **Do not enable letterboxing here.** SV3DT projection assumes the streammux output
  is the only scaling. `maintain-aspect-ratio=0` and `symmetric-padding=0` in
  `pipelines/config_infer_primary_yolo11_seg.ini` are required to avoid double
  aspect correction.

### 2) nvdewarper (family-room rectification)

- Distortion coefficients order is radial k0,k1,k2 then tangential k3,k4.
- Distortion is specified as an array of 4/5 floats in the config.

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdewarper.html
> Distortion coefficients for perspective source:
> Three radial coefficients: (k_0,k_1,k_2) and two tangential coefficients: (k_3,k_4).
> distortion=k_0; k_1; k_2; k_3; k_4

### 3) cameras*.yaml (intrinsics)

- Canonical intrinsics source is `config/cameras.yaml` (or preview variant).
- Intrinsics are scaled to streammux resolution when source resolution differs.

Source: `docs/DS8_api_contracts_ws.md`
> Canonical source: config/cameras.yaml -> intrinsics_models
> Streammux scaling: K is scaled to match streammux.width x streammux.height when source resolution differs.

### 4) camera_calibration*.json (extrinsics E)

- E is world->camera, column-major, meters, Y-up world.
- Camera forward is +Z (from DS8 WS contracts).

Source: `docs/DS8_api_contracts_ws.md`
> Semantics: 4x4 world->camera transform
> Layout: column-major
> World frame: Y-up
> Camera forward: +Z
> Units: meters

### 5) ply_alignment.json (alignment)

- Matrix is row-major 4x4; applied by MapAnything pose conditioning.
- pixel_to_world does not apply alignment (Menon applies it client-side).

Source: `docs/DS8_api_contracts_ws.md`
> matrix: 16 floats, row-major (C order)
> MapAnything pose conditioning applies align.matrix as T' = M_align @ T_wc
> pixel_to_world does NOT apply align.matrix

### 6) CamInfo generation (SV3DT)

- camInfo projection matrix uses streammux resolution (not tracker width).
- Default baseline envs for camInfo generation:
  - NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p
  - NOESIS_V3DT_CAMINFO_INVERT_E=0
  - NOESIS_V3DT_CAMINFO_Y_FLIP=1
  - NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy
  - NOESIS_V3DT_CAMINFO_WORLD_SCALE=1 (meters)

Source: `scripts/generate_v3dt_caminfo.py`
> Baseline defaults: MATRIX_TYPE=w2p, INVERT_E=0, Y_FLIP=1, WORLD_AXES=xzy.
> Extrinsics are stored in meters; camInfo defaults to meters unless WORLD_SCALE=100.

Source: `docs/DS8_v3dt_forensics.md`
> export NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p
> export NOESIS_V3DT_CAMINFO_INVERT_E=0
> export NOESIS_V3DT_CAMINFO_Y_FLIP=1
> export NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy

The generator right-multiplies the canonical world-to-camera transform by the
`xzy` axis map. Therefore the locked tracker profile emits coordinates in that
remapped tracker tuple and the producer must apply the same signed permutation
before publishing `backend_world_m`. For this profile, the real tracker output
uses `zLen` as height and its ground endpoint is
`(xCentre, yCentre, zCentre - 0.5*zLen)`. This is a measured property of the
locked camInfo/profile, not a blanket claim about every `NvDsObj3DBbox` layout.

### 7) nvtracker SV3DT (ObjectModelProjection)

- camInfo supports projectionMatrix_3x4 or projectionMatrix_3x4_w2p.
- projectionMatrix_3x4 assumes principal point at (0,0) and SV3DT adds (w/2, h/2).
- projectionMatrix_3x4_w2p assumes pixel origin at top-left and needs no extra shift.
- modelInfo height/radius are defined in world coordinates.
- outputFootLocation controls 3D bbox output; outputVisibility/outputConvexHull gate meta.

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
> projectionMatrix_3x4 assumes principal point at (0,0); SV3DT internally adds (img_width/2, img_height/2).
> projectionMatrix_3x4_w2p is world->pixel with origin at top-left; no further translation required.
> modelInfo height and radius represent the 3D human model in world coordinates.
> outputVisibility/outputFootLocation/outputConvexHull enable meta outputs; outputFootLocation controls 3D bbox output.

### 8) MapAnything pose conditioning (depth)

- E is inverted to cam->world (T_wc) and aligned with ply_alignment matrix.
- Translation is scaled by units.s_obj_to_m; rotation remains orthonormal.

Source: `noesis/metadata/mapanything_pose.py`
> Converts E (world->camera) into camera_poses (cam->world).
> Applies alignment M (row-major) and translation scaling via units.s_obj_to_m.

### 9) nvdsanalytics (ROI/occupancy)

- Analytics uses bottom-center of bbox in pixel coordinates for rules.
- ROI polygons are specified as x;y coordinates in config.

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsanalytics.html
> For all analytics calculations bottom center coordinate of bbox is used:
> (x_left + width/2, y_top + height).
> roi-<label>=x1;y1;... coordinates define polygons.

### 10) v3dt_forensics (offline diagnostics)

- Intrinsics are scaled using base_w=2*cx, base_h=2*cy (assumes principal point at image center).
- Ray check uses bottom-center pixel and floor_y for intersection; reports a Y-flip diagnostic.
- Expected camInfo projection applies NOESIS_V3DT_CAMINFO_WORLD_AXES and optional Y-flip.

Source: `noesis/diagnostics/v3dt_forensics.py`
> _scale_intrinsics uses base_w=2*cx, base_h=2*cy
> ray check uses bottom-center pixel and floor_y
> expected P applies world-axes map and optional Y-flip

## Diagnostics Workorder (extend v3dt_forensics.py)

1) Add a "conventions" snapshot section that prints:
   - streammux scaling (width/height, enable-padding)
   - intrinsics base resolution (from cameras.yaml if present) vs 2*cx assumption
   - dewarper output size and dst K vs camera model K
   - camInfo matrix type and projection principal-point assumptions
2) Add warnings for:
   - enable-padding=1 without pad offsets applied to K
   - camera model resolution missing (cx/cy not at center)
   - camInfo w2p vs 3x4 mismatches relative to tracker config
3) Add a per-camera "expected ray" section that reports:
   - ray intersection t for normal and y-flip
   - whether camInfo matches expected P with and without y-flip
4) Add an "SV3DT meta contract" section with required flags:
   - ObjectModelProjection.outputFootLocation/outputVisibility
   - modelInfo height/radius units vs WORLD_SCALE
5) Extend the report to correlate bbox3d projection errors with:
   - streammux scaling (expected bbox height from fy and zLen)
   - dewarper dst K deviations (family-room rectified K)

## Open Questions (current)

- The installed binding documents the SDK default `NvDsObj3DBbox` as Y-up, while
  the locked `xzy` camInfo profile is empirically Z-up (`zLen` is the 1.85 m
  height). Any new camInfo profile must prove its own axis contract rather than
  inheriting the locked profile's Z-foot rule.
- The calibration snapshot scaling uses 2*cx; if camera models have off-center
  principal points, we need to decide whether to encode base resolution explicitly
  in cameras.yaml or adjust diagnostics to use the explicit resolution field.
