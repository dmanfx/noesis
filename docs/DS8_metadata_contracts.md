# DS8 Metadata Contracts
_Status: current as of 2026-03-16._

This document summarizes the key metadata structures used by the DS8 pipeline, both on-frame (user meta) and in downstream telemetry.

## 1. Intrinsics (Calibration Bundle + optional per-frame meta)

**Primary producer (DS8 runtime):** `noesis/ds8_runtime._CalibrationProvider` via the `calibration-bundle` WebSocket message and `CalibrationSnapshot` for BEV/geometry consumers.

**Optional producer (DeepStream user meta):** `noesis/metadata/intrinsics.py` via `attach_intrinsics` (supported for `pyds.NvDsFrameMeta`).

### Storage

- DS8 canonical path (Service Maker pipelines):
  - Intrinsics are distributed out-of-band via `calibration-bundle` (WS) and stored on the depth subsystem (`DepthStorageManager.calibration_bundle`).
  - BEV/geometry consumers should use `_CalibrationProvider.snapshot(...)` to obtain a `CalibrationSnapshot` (K + E + floor_y + unit scale).
- For DeepStream frames (NvDsFrameMeta), intrinsics are attached as NvDsUserMeta:
  - Meta type: value retrieved via `nvds_get_user_meta_type("NOESIS.INTRINSICS")`.
  - `user_meta_data` holds either an `_IntrinsicsMetaPayload` or a dict-like structure.
- For non-DeepStream frames (e.g., tests or intermediate objects), intrinsics may be attached as:
  - `frame_meta["user_meta"]["intrinsics"]` for mapping-like frames.
  - `frame_meta.noesis_intrinsics` for generic objects.

### Payload

The intrinsics payload (after `as_payload()`) has the shape:

```json
{
  "fx": <float>,
  "fy": <float>,
  "cx": <float>,
  "cy": <float>,
  "k1": <float>,
  "k2": <float>,
  "k3": <float>,
  "height_m": <float>
}
```

The loader accepts either a `cameras` or `sources` mapping at the root of `config/cameras.yaml`, and supports reusable intrinsics under `intrinsics_models` (or `models`) referenced by `model`/`intrinsics_model`. Keys `fx`, `fy`, `cx`, `cy` are required, with optional `K_matrix`/`camera_matrix` fallbacks and distortion coefficients (`k1`/`k2`/`k3` or `distortion_coeffs`).

DS8 code that consumes intrinsics should prefer the runtime `calibration-bundle` / `CalibrationSnapshot`. Per-frame intrinsics user meta may be absent in Service Maker pipelines.

## 1.1 Depth Registration Artifact

**Primary producer:** offline builder `scripts/build_depth_registration.py`.

**Primary consumer:** baseline DS8 startup + fused world estimator (`noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`).

This artifact is separate from `NOESIS.OBJECT_DEPTH`, `DepthResult`, floorplan caches, and MapAnything snapshot storage. It is a DS8-owned config artifact that maps raw DAv2 anchor range into MapAnything-aligned room range on a per-camera basis.

Canonical bundle shape:

```json
{
  "depth_registration_contract_version": 1,
  "cameras": {
    "<camera_id>": {
      "camera_id": "<camera_id>",
      "registration_id": "<camera_id>:<short_id>",
      "created_ts_us": <int>,
      "transform_type": "piecewise_linear_1d",
      "source_space": "dav2_anchor_range_m_raw",
      "target_space": "mapanything_room_range_m",
      "scope": "people_tracking_depth_registration",
      "raw_range_domain_m": [<float lo>, <float hi>],
      "knots_raw_m": [<float>, ...],
      "knots_registered_m": [<float>, ...],
      "generation_tool_version": "<string>",
      "fit_metrics": {
        "mean_abs_error_m": <float>,
        "median_abs_error_m": <float>,
        "p90_abs_error_m": <float>,
        "raw_domain_m": [<float lo>, <float hi>]
      },
      "sample_counts": {
        "input_pairs": <int>
      },
      "provenance": {
        "source_uri": "<string>",
        "frames_per_camera": <int>,
        "sample_pixel_step": <int>,
        "row_start_frac": <float>,
        "max_luma_mad": <float>,
        "max_depth_delta_m": <float>
      },
      "calibration_fingerprint": { "fingerprint_sha256": "<sha256>", ... },
      "dav2_profile": { "fingerprint_sha256": "<sha256>", ... },
      "mapanything_profile": { "fingerprint_sha256": "<sha256>", ... }
    }
  }
}
```

Contract rules:
- Runtime must treat the artifact as read-only and load it before pipeline activation.
- In baseline non-`v3dt` mode, every enabled camera must have a valid entry whose calibration + model fingerprints match the active runtime inputs.
- The artifact corrects only the estimator-local depth observation. It must not overwrite the raw `NOESIS.OBJECT_DEPTH` payload.
- `registration_id`, `created_ts_us`, `generation_tool_version`, `fit_metrics`, `sample_counts`, and `provenance` are all part of the operational contract now and should be preserved when regenerating the bundle.

## 2. Depth Result

**Producer:** `noesis/pipelines/hooks.MapAnythingProcessor` → `DepthStorageManager` + `DepthTelemetryPublisher`.

### In-Memory Schema

As defined in `noesis/metadata/depth_result.py`:

```python
DepthResult(
    source_id: int,
    frame_id: int,
    ts: int,           # epoch seconds
    width: int,
    height: int,
    depth_map_ref: str,
    minmax: Tuple[float, float],
    unit: str = "m",
)
```

### JSON Representation

Via `.to_dict()` / `.to_json()`:

```json
{
  "source_id": <int>,
  "frame_id": <int>,
  "ts": <int>,
  "width": <int>,
  "height": <int>,
  "depth_map_ref": "<string>",
  "minmax": [<float min>, <float max>],
  "unit": "m"
}
```

Consumers (WS clients, REST, or back-end services) must treat `depth_map_ref` as an opaque reference; its structure is defined by the storage subsystem.

## 2.1 Depth Normals (MapAnything)

When the depth RPC (`ma_depth_response`) is used, the payload may include an optional normals map derived from the depth snapshot:

```json
{
  "normals_b64": "<base64 float16/float32>",
  "normals_shape": [<H>, <W>, 3],
  "normals_dtype": "float16" | "float32",
  "normals_space": "camera" | "world",
  "normals_error": "<optional>"
}
```

Notes:
- Normals are computed on-demand and do **not** replace `DepthResult`.
- `normals_space="camera"` uses the OpenCV camera frame (+X right, +Y down, +Z forward).
- When `normals_error` is present, consumers should ignore normals and still use depth.

### UI visualization (oai2-fe DepthDrawer)

When present, the DepthDrawer’s **Normals** tab visualizes normals as an RGB image by mapping each normal component from **[-1, +1] → [0, 255]**:
- **R** = X component
- **G** = Y component
- **B** = Z component

Pixels with invalid or near-zero normals are rendered transparent (typically because depth was invalid/masked there).

## 3. Analytics Object & Frame Meta

**Producers/Consumers:** `noesis/pipelines/hooks._AnalyticsTelemetryProcessor` and exclusion/BEV helpers.

### Object-Level Analytics Meta

Attached as NvDsUserMeta with type `NVIDIA.DSANALYTICSOBJ.USER_META` (or equivalent type id). The payload is cast to NvDsAnalyticsObjInfo where available, and normalized by `_extract_analytics_obj_meta` to:

```json
{
  "dirStatus": "<string|null>",
  "lcStatus": "<string|null>",
  "ocStatus": "<string|null>",
  "roiStatus": "<string|null>",
  "direction_status": "<string|null>",
  "line_crossing_status": "<string|null>",
  "overcrowding_status": "<string|null>",
  "roi_status": "<string|null>"
}
```

Snake_case keys are provided for backwards compatibility with older consumers.

### Frame-Level Analytics Meta

When available (NvDsAnalyticsFrameMeta), frame-level meta is normalized by `_extract_analytics_frame_meta` to include, for example:

```json
{
  "objects_in_roi": { "<roi-id>": <int count>, ... },
  "line_crossing_cumulative": { ... },
  "line_crossing_current": { ... },
  "overcrowding_status": "<string|null>"
}
```

This is used to supplement occupancy information when object-level ROI counts are ambiguous or missing.

## 4. Tracking & Occupancy Derived Structures

**Producer:** `_AnalyticsTelemetryProcessor` and compatibility logic.

### Track Dictionary

Each track emitted via tracking telemetry or internal structures has fields such as:

```json
{
  "camera_id": "<string>",
  "bbox": [<float left>, <float top>, <float width>, <float height>],
  "center": [<float cx>, <float cy>],
  "class_id": <int>,
  "confidence": <float>,
  "tracker_confidence": <float|null>,
  "analytics": { /* see analytics obj meta above */ },
  "zone": "<string|null>",
  "frame_id": <int>,
  "stable_id": <int>,
  "dwell_time": <float|null>,
  "bbox3d": {
    "xCentre": <float>, "yCentre": <float>, "zCentre": <float>,
    "xLen": <float>, "yLen": <float>, "zLen": <float>,
    "xRot": <float>, "yRot": <float>, "zRot": <float>
  },
  "velocity3d": [<float xVel>, <float yVel>, <float zVel>],
  "image_foot": [<float u>, <float v>],
  "image_base": [<float u>, <float v>],
  "world": [<float x>, <float y>, <float z>],
  "world_valid": <bool>,
  "world_quality": "good"|"estimated"|"invalid",
  "world_quality_reason": "<string|null>",
  "world_frame": "<string|null>",
  "world_source": "bbox3d"|"pose_depth_fused"|"pose_floor_only"|"person_anchor_depth_fused"|"person_anchor_floor_only"|"gravity_drop"|"anchor_hold"|null
}
```

These structures are not stored as user meta on frames by default but are the basis for WebSocket tracking telemetry and occupancy calculations.

- `track_id` remains an internal tracker identifier and must not be emitted to clients.
- `stable_id` is global (can be present across multiple cameras) and must be an integer `>= 1` for people tracks.
- Negative/provisional stable IDs are internal-only and must not be emitted to clients.
- `dwell_time` is derived per track by `_AnalyticsTelemetryProcessor` using zone entry timestamps; it is null when no zone is available.
- `bbox3d` and `velocity3d` are attached when `NVDS_OBJ_3D_META` (SV3DT/MV3DT) is present.
- `image_foot` is the active current-frame image anchor chosen by the backend estimator: pose-derived when available, otherwise the person mask/depth anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`. `image_base` is the canonical image reprojection of the filtered world state.
- In baseline (non-`v3dt`) DS8 mode, `world` is produced by the backend fused world estimator in `hooks.py`: the canonical person anchor is pose-derived when available and otherwise comes from `NOESIS.OBJECT_DEPTH.anchor_uv` for class-0 tracks. A concurrent DAv2 range observation from `NOESIS.OBJECT_DEPTH.anchor_depth_m` can refine either current-anchor path as `pose_depth_fused` or `person_anchor_depth_fused`. When DAv2 is unavailable for a frame, the same estimator continues as `pose_floor_only`, `person_anchor_floor_only`, `gravity_drop`, or `anchor_hold`.
- In `v3dt` mode, `world`/`world_source="bbox3d"` continue to come from `NVDS_OBJ_3D_META`.
- `world_frame` may be set to `"camera_local"` until shared global calibration is available.
- `world_quality_reason` is the canonical diagnostic string explaining why the current update was fused, floor-only, guarded, held, or invalid.

### Occupancy State

Occupancy per zone is represented as:

```json
{
  "<zone-id>": <int count>,
  ...
}
```

The occupancy publisher (`occupancy_publisher`) expects `publish_state(room_id, occupied, count, ts_ns)` calls aligned with this representation.

In DS8, `_AnalyticsTelemetryProcessor` derives occupancy counts per sensor from analytics ROI status (zones) on each frame and publishes through `pipeline.occupancy_publisher`, emitting vacate events when a previously occupied zone disappears.

Vacates are emitted immediately when a zone’s count drops to zero; there is no grace window in the current DS8 implementation.

## 5. Custom User Meta Lifecycle (DS8 / DeepStream)

When attaching **custom user meta** (`NvDsUserMeta`) from DS8 hooks or native bridges, follow DeepStream’s copy/release semantics exactly to avoid heap corruption:

- `copy_func(data, user_data)` and `release_func(data, user_data)` are called with `data == NvDsUserMeta*` **(not** `user_meta_data`).
- `copy_func` must **deep‑copy** `user_meta_data` and return the new payload pointer.
- `release_func` must **free only** `user_meta_data` and set it to `NULL`. **Never** free the `NvDsUserMeta*`.

Minimal pattern (JSON payload):

```cpp
static gpointer pose_meta_copy(gpointer data, gpointer) {
  auto* meta = static_cast<NvDsUserMeta*>(data);
  if (!meta || !meta->user_meta_data) return nullptr;
  return g_strdup(static_cast<const gchar*>(meta->user_meta_data));
}

static void pose_meta_release(gpointer data, gpointer) {
  auto* meta = static_cast<NvDsUserMeta*>(data);
  if (!meta) return;
  if (meta->user_meta_data) {
    g_free(meta->user_meta_data);
    meta->user_meta_data = nullptr;
  }
}
```

**DS8 note:** Service Maker Python wrappers do not expose `obj_user_meta_list` or `append` for arbitrary user meta. Use a **native bridge** (e.g., `native/noesis_pose_meta_ext.cpp`) to unwrap `ObjectMetadata` → `NvDsObjectMeta*` and call `nvds_add_user_meta_to_obj`.

## 6. Pose Feature User Meta (Object-Level)

**Producer:** `noesis/pipelines/hooks.PoseFeatureProcessor` (YOLO26 pose SGIE).

**Meta type:** `NOESIS.POSE_FEATURES` (user meta attached to each `NvDsObjectMeta`).

**Payload:** JSON string attached as user meta data. Shape:

```json
{
  "type": "pose_features",
  "version": 1,
  "model": "yolo26-pose",
  "source_id": 0,
  "frame_id": 123,
  "object_id": 456,
  "class_id": 0,
  "bbox": [left, top, width, height],
  "score": 0.87,
  "kpt_mean_conf": 0.62,
  "kpt_min_conf": 0.08,
  "kpt_valid_frac": 0.58,
  "stable_id": 12,
  "ts_us": 1700000000000,
  "features": {
    "height_proxy": 184.2,
    "torso_leg_ratio": 0.47,
    "shoulder_width_norm": 0.26,
    "left_arm_ratio": 0.91,
    "leg_symmetry": 0.07
  }
}
```

**Notes:**

- Features are derived from 2D keypoints in ROI space and are **ratio‑oriented** for scale stability.
- The payload is intended for downstream StableID enhancements; it is **not** emitted on the WebSocket tracking stream.
- StableIDManager may consume this meta as a secondary identity signal with a bounded in‑RAM pose gallery (no disk persistence).
- The baseline fused world estimator also consumes the pose keypoints as the canonical image-anchor authority. DAv2 depth may refine range for that same anchor, but pose meta remains the source of the floor anchor chain.
- See `docs/DS8_pose_stable_id_integration.md` for fusion thresholds, env flags, and memory caps.
- Attached in DS8 via `noesis_pose_meta_ext.attach_pose_features(...)` which calls `nvds_add_user_meta_to_obj` with the lifecycle functions defined above.

## 7. Object Depth User Meta (Object-Level)

**Producer:** DS8 seg+depth prototype and the baseline DS8 runtime depth-tracking lane (YOLO26 seg + DepthAnything V2 metric) attach this payload after full-frame depth is aligned once into canonical DS8 frame coordinates and sampled strictly over the decoded instance mask.

**Meta type:** `NOESIS.OBJECT_DEPTH` (user meta attached to each `NvDsObjectMeta`).

**Payload:** JSON string attached as user meta data. Shape:

```json
{
  "type": "object_depth",
  "version": 2,
  "model": "depth-anything-v2-metric-hypersim-vits",
  "source_id": 0,
  "frame_id": 123,
  "object_id": 456,
  "class_id": 0,
  "bbox": [left, top, width, height],
  "score": 0.87,
  "sampling_mode": "instance_mask",
  "status": "ok",
  "unit": "m",
  "is_metric": true,
  "sample_count": 512,
  "valid_fraction": 0.82,
  "depth_center": 1.2,
  "depth_median": 1.3,
  "depth_mean": 1.31,
  "depth_p10": 0.9,
  "depth_p90": 1.8,
  "depth_min": 0.7,
  "depth_max": 2.0,
  "mask_area_px": 640,
  "stable_id": 12,
  "anchor_uv": [420.5, 541.5],
  "anchor_source": "lower_body_band",
  "anchor_depth_m": 1.25,
  "anchor_sample_count": 72,
  "anchor_valid_fraction": 0.91,
  "world_point": [1.0, 0.0, 3.5],
  "world_point_depth": [1.1, 0.2, 3.6],
  "world_point_floor": [1.0, 0.0, 3.4],
  "projection_method": "depth",
  "spatial_status": "ok",
  "spatial_class": "person",
  "depth_map_ref": "memory://depth/family-room/1700000000000",
  "ts_us": 1700000000000
}
```

**Notes:**

- The payload is attached per object only after full-frame depth has been aligned once into canonical post-mux DS8 frame coordinates and sampled strictly over the decoded instance mask.
- `bbox` and all mask/depth statistics are expressed in that canonical DS8 frame space. Source-native dimensions (`source_frame_width`, `source_frame_height`) are diagnostic-only and must not be used for object-depth sampling.
- There is no bbox fallback in the current prototype path. If the mask is missing, mask decode fails, the aligned depth frame is not ready, or the geometry is inconsistent, the payload is still attached with a non-`"ok"` `status`.
- `status` is mandatory and distinguishes usable samples (`"ok"`) from object-local failures such as `"no_valid_depth"`, `"missing_mask"`, `"mask_decode_failed"`, `"depth_not_ready"`, or `"transform_mismatch"`.
- Version `2` adds optional person-only spatial fields derived from the segmentation mask plus the DS8 calibration bundle. `anchor_uv` is the bottom-of-mask image anchor in canonical frame space, `anchor_depth_m` is the preferred lower-body or torso-core depth sample, and `world_point*` fields are produced by `pixel_to_world(...)` without applying `align.matrix`.
- `anchor_sample_count` and `anchor_valid_fraction` describe the support of the specific lower-body / torso anchor band that produced `anchor_depth_m`. Baseline DS8 tracking weights DAv2 using these anchor-band support fields instead of whole-mask support alone so far-camera people can still contribute depth when the chosen anchor band is well supported.
- `projection_method` is one of `"depth"`, `"floor_guarded"`, `"depth_only"`, or `"floor_only"`. `spatial_status` surfaces whether that world projection is usable (`"ok"`) or why it is absent (`"geometry_unavailable"`, `"anchor_unavailable"`, `"projection_unavailable"`).
- In the baseline DS8 runtime, `NOESIS.OBJECT_DEPTH` is a required observation for the non-`v3dt` fused world estimator. The runtime consumes the raw depth/anchor fields (`status`, `sample_count`, `valid_fraction`, `anchor_source`, `anchor_depth_m`, `anchor_sample_count`, `anchor_valid_fraction`) and computes the canonical per-track `track.world` inside `hooks.py`.
- When a valid depth-registration artifact is loaded, the runtime keeps `anchor_depth_m` raw in `NOESIS.OBJECT_DEPTH`, derives a corrected `depth_registered_m` only inside the fused estimator, and projects that corrected value on the pose anchor ray before writing `track.world`.
- The object-level `world_point`, `world_point_depth`, `world_point_floor`, and `projection_method` fields remain diagnostic-only in baseline runtime mode. They are useful for inspection and parity testing, but they are not the authoritative tracking output once the fused estimator is active.
- Non-person classes keep the raw object-depth payload behavior and omit the spatial fields.
- The runtime uses a bounded internal aligned-depth cache to hand the canonical full-frame depth map from the depth-capture operator to the later object-fusion/overlay operators. That cache is prototype-internal only and is not a public DS8 metadata contract.
- `sampling_mode` is currently fixed to `"instance_mask"`. Consumers must not infer bbox-based semantics from missing numeric fields.
- `unit` and `is_metric` travel with the payload so downstream consumers can distinguish metric meters from relative-depth fallbacks without guessing from the model name.
- `depth_map_ref` is optional and is an opaque pointer to any stored full-frame depth artifact when one exists; consumers must not parse semantics out of the string.
- Attached in DS8 via `noesis_depth_meta_ext.attach_object_depth(...)`; instance masks are copied out of `NvOSD_MaskParams` via the native helper `noesis_depth_meta_ext.extract_object_mask(...)` before sampling.

## 8. Pose Keypoint Visualization (OSD)

Pose keypoints/skeletons are **rendered in the mosaic** using DS8 display metadata inserted just before `nvdsosd`:

- Config gate: `visualization.display_keypoints` (in `config/infer.yaml`).
- Source: YOLO26 pose SGIE tensor meta (`unique_id` from `models.pose.gie_id`).
- Rendering: `PoseKeypointOverlayProcessor` attaches `DisplayMeta` lines/circles at the tiler stage.

This visualization does **not** add new user meta; it is purely an OSD overlay.

## 9. Calibration & Geometry

**Consumers:** `noesis/telemetry/bev.py`, calibration RPCs in `websocket_server.py`, geometry helpers.

- Calibration data (intrinsics + extrinsics + floor plane) must be consistent across:
  - `config/cameras.yaml` / calibration outputs.
  - RPCs like `set_extrinsics`, `set_align`, and `auto_calibrate_pose`.
  - BEV computations based on `CalibrationSnapshot` in `bev.py`.

The baseline non-`v3dt` world estimator depends on this calibration stack in three places that must stay semantically aligned:

- pose-derived image anchors from `NOESIS.POSE_FEATURES` / pose keypoints,
- DAv2 range observations from `NOESIS.OBJECT_DEPTH.anchor_depth_m`,
- and world projection via `pixel_to_world(...)` in `hooks.py`.

Any change to calibration formats or semantics must be reflected here and in the corresponding DS8 docs, because the backend fused estimator is now the canonical owner of `track.world`.
