# DS8 Metadata Contracts
_Status: current as of 2026-02-02._

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
  "world_frame": "<string|null>"
}
```

These structures are not stored as user meta on frames by default but are the basis for WebSocket tracking telemetry and occupancy calculations.

- `track_id` remains an internal tracker identifier and must not be emitted to clients.
- `stable_id` is global (can be present across multiple cameras) and must be an integer `>= 1` for people tracks.
- Negative/provisional stable IDs are internal-only and must not be emitted to clients.
- `dwell_time` is derived per track by `_AnalyticsTelemetryProcessor` using zone entry timestamps; it is null when no zone is available.
- `bbox3d` and `velocity3d` are attached when `NVDS_OBJ_3D_META` (SV3DT/MV3DT) is present.
- `image_foot` is the tracker-provided footpoint in image coordinates; `image_base` is the projected bbox3d base-center.
- `world` is derived from the SV3DT 3D bbox footpoint when available, otherwise from the existing ray-plane intersection.
- `world_frame` may be set to `"camera_local"` until shared global calibration is available.

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
- See `docs/DS8_pose_stable_id_integration.md` for fusion thresholds, env flags, and memory caps.
- Attached in DS8 via `noesis_pose_meta_ext.attach_pose_features(...)` which calls `nvds_add_user_meta_to_obj` with the lifecycle functions defined above.

## 7. Pose Keypoint Visualization (OSD)

Pose keypoints/skeletons are **rendered in the mosaic** using DS8 display metadata inserted just before `nvdsosd`:

- Config gate: `visualization.display_keypoints` (in `config/infer.yaml`).
- Source: YOLO26 pose SGIE tensor meta (`unique_id` from `models.pose.gie_id`).
- Rendering: `PoseKeypointOverlayProcessor` attaches `DisplayMeta` lines/circles at the tiler stage.

This visualization does **not** add new user meta; it is purely an OSD overlay.

## 6. Calibration & Geometry

**Consumers:** `noesis/telemetry/bev.py`, calibration RPCs in `websocket_server.py`, geometry helpers.

- Calibration data (intrinsics + extrinsics + floor plane) must be consistent across:
  - `config/cameras.yaml` / calibration outputs.
  - RPCs like `set_extrinsics`, `set_align`, and `auto_calibrate_pose`.
  - BEV computations based on `CalibrationSnapshot` in `bev.py`.

Any change to calibration formats or semantics must be reflected here and in the corresponding DS8 docs.
