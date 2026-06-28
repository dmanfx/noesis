# DS8 WebSocket API Contracts
_Status: validation-diagnostics addendum current as of 2026-05-27._

The WebSocket server (`websocket_server.WebSocketServer`) is the primary transport for DS8 telemetry, depth retrieval, and WebRTC signaling. All active DS8 telemetry message types are JSON unless a future binary payload explicitly documents otherwise.

## 1. Common Envelope

Most JSON messages have:

```json
{ "type": "<message-type>", ... }
```

BEV JPEG binary payloads are retired. The server still has a generic binary
coalescer for future payload types, but current BEV delivery is metadata-only.

## 2. Stats (`type: stats`)

Emitted ~1 Hz when `stats_callback` is registered (`ds8_runtime._build_stats_callback`).

```json
{
  "type": "stats",
  "payload": {
    "timestamp": <float>,
    "uptime": <float>,
    "stack": "ds8",
    "application": {
      "running": <bool>,
      "cameras_active": <int>,
      "processors_active": <int>
    },
    "pipeline": {
      "stack": "ds8",
      "prepared": <bool>,
      "activated": <bool>,
      "depth_enabled": <bool>,
      "depth_fps": <float>,
      "analytics_reload_count": <int>,
      "mosaic_layout": {
        "mosaic_w": <int optional>,
        "mosaic_h": <int optional>,
        "rows": <int|null>,
        "cols": <int|null>,
        "source_count": <int>,
        "sources": [{"source_id": <int>, "camera_id": "<string>"}],
        "frame_w": <int>,
        "frame_h": <int>,
        "square_seq_grid": <bool>,
        "tile_order": "source-id"
      },
      "latency_ms": {
        "enabled": <bool>,
        "window_sec": <float>,
        "count": <int>,
        "p50": <float|null>,
        "p95": <float|null>,
        "max": <float|null>,
        "last_sample_age_sec": <float|null>,
        "reason": "<string optional>"
      },
      "errors": [<string>, ...]
    },
    "cameras": {
      "<camera_id>": {
        "fps": <float>,
        "frames_processed": <int>,
        "status": "running"|"unknown",
        "latency_ms": { /* same shape as pipeline.latency_ms, optional */ },
        "tracking": {
          "occupancy": {"<zone>": <int>, ...},
          "active_tracks": [ /* same per-track shape as tracking.tracks, plus occupancy-scoped subset */ ],
          "transitions": [ /* line/zone transitions */ ]
        }
      }
    }
  }
}
```

`latency_ms` fields are populated only when `NVDS_ENABLE_LATENCY_MEASUREMENT` is truthy and the DeepStream latency meta library is available.

`depth_enabled` and `depth_fps` still describe the on-demand MapAnything branch. The baseline DAv2 depth-tracking lane used by non-`v3dt` world estimation is separate and always-on when the baseline runtime starts successfully.

Baseline non-`v3dt` startup also requires a prebuilt room-registration artifact (`depth_registration.path` in `config/infer.yaml`, default `config/depth_registration.json`). DS8 loads that artifact before activation and fails fast if any enabled camera is missing a valid DAv2→MapAnything registration entry.

## 3. Mosaic Video (WebRTC)

Mosaic video is delivered via RTSP→WebRTC gateway; WebSocket is **signaling only**.

- `webrtc_offer` (client → server): `{ "type": "webrtc_offer", "sdp": "<offer sdp>" }`
- `webrtc_answer` (server → owner): `{ "type": "webrtc_answer", "sdp": "<answer sdp>" }`
- `webrtc_ice_candidate` (bi-directional): `{ "type": "webrtc_ice_candidate", "candidate": "<candidate>", "sdpMLineIndex": 0 }`
- `webrtc_error` (server → client): `{ "type": "webrtc_error", "error": "<string>" }`

Ownership: the most recent `webrtc_offer` sender is the owner; non-owners sending ICE get `webrtc_error (webrtc_not_owner)`; takeover triggers `webrtc_error (webrtc_taken_over)` to the previous owner.

## 4. BEV Frames

Emitted by `BevRenderer`:

```json
{
  "type": "bev-frame",
  "cameraId": "<camera>",
  "ts": <int microseconds>,
  "w": <int>,
  "h": <int>,
  "mpp": <float>,
  "xMin": <float>, "xMax": <float>,
  "zMin": <float>, "zMax": <float>,
  "overlay": <bool>,
  "footpoints": [
    {
      "x": <float>,
      "y": <float>,
      "method": "<string>",
      "stableId": <int|null>,
      "trackerId": <int|null>,
      "anchorSource": "<string|null>",
      "anchorQuality": "<string|null>",
      "anchorReason": "<string|null>",
      "displaySource": "world"|"world_to_camera_local"|"image_anchor"|"image_depth_anchor"|"floor_contact_ray"
    }
  ],
  "trails": [ {"stableId": <int|null>, "trackerId": <int|null>, "points": [ {"x": <float>, "y": <float>, "t": <int ms>} ]} ],
  "H": [<9 floats>],
  "sampleXZ": [<float x>, <float z>] | null,
  "frame": "backend_world_m"|"camera_local_ground_m",
  "world_frame": "backend_world_m"|"camera_local_ground_m",
  "frame_mode": "world"|"camera_local",
  "units": "meters",
  "s_obj_to_m": <float>,
  "trail_smoothing_owner": "frontend"|"backend"|"none",
  "bev_points_smoothed": <bool>,
  "bev_world_points_smoothed": <bool>
}
```

- `footpoints[].method` is an image-anchor/render provenance string emitted by the backend (`image_base`, `image_foot`, `bbox`, etc.), not the canonical track world estimator source.
- `footpoints[].anchorSource`, `anchorQuality`, and `anchorReason` mirror the backend world estimator diagnostics from tracking telemetry so BEV/Three.js consumers can explain why a point was accepted, guarded, or held.
- `footpoints[].displaySource` declares which coordinate path produced the displayed BEV point. The primary inline floorplan view uses `frame_mode=camera_local` and `frame=camera_local_ground_m`, so displayed points and producer trails are in the same camera-local ground frame as MapAnything floorplan rasters. For tracked people, `world_to_camera_local` is preferred whenever the backend fused world estimator produced a current live observation (`pose_depth_fused`, `person_anchor_depth_fused`, floor-only variants, `gravity_drop`, or `bbox3d`). `floor_contact_ray` is the calibrated image-ground fallback when no current live world observation is available. `image_depth_anchor` remains a legacy/non-person direct image-depth path; static MapAnything/floorplan snapshots are not a live person-depth placement source.
- When `NOESIS_BEV_ALIGNMENT_DEBUG=1` is set, `bev-frame` may include top-level `alignmentDebug`, and each footpoint may include `rawX`, `rawY`, `smoothed`, and `alignmentDebug` with per-candidate image anchors, ray-floor projections, optional static floorplan/MapAnything snapshot samples, snapshot ids, grid cells, and selected-coordinate bounds. These fields are diagnostic-only and are not used to place live tracked people.

- Optional JPEG binary: **Retired**. The framed `bev:<camera>` binary path is no longer produced (meta-only mode is the supported baseline per design decisions and Baselines.md). The general binary coalescer in the WebSocket server is retained for potential future use (e.g., binary depth).
- In world mode (`frame_mode=world`), BEV footpoints remain producer-owned scene coordinates and should be treated as the canonical `track.world` head points emitted by the backend. The BEV renderer must not apply a second world-space low-pass filter to those points.
- Motion smoothing ownership is declared explicitly by `trail_smoothing_owner`. In the current baseline world-mode path the backend owns trail history (`trail_smoothing_owner=backend`) while `bev_world_points_smoothed=false`, because the canonical per-track world estimator in `hooks.py` already owns the only track-position smoothing stage.
- When `trail_smoothing_owner=backend`, `trails` carries the producer trail polylines already used by the BEV renderer, in the declared BEV `frame` with epoch-millisecond sample times. The dashboard should render those directly instead of reconstructing its own history from `footpoints`.
- World-mode BEV omits `anchor_hold` head points from `footpoints`/`trails` so stale held positions do not render as drifting or out-of-bounds trail segments after temporary occlusion.
- Backend world-BEV smoothing and trail history are keyed by tracker-local identity (`trackerId` when present, otherwise `stableId`) to match the nvOSD trail path; `stableId` remains display metadata and may legitimately span multiple tracker histories over time.
- Coordinate note: BEV renders on the ground plane (XZ). `footpoints[].x` is X and `footpoints[].y` is Z in the declared `frame`.

## 5. Depth Telemetry (`type: depth_result`)

Produced by `DepthTelemetryPublisher`:

```json
{
  "type": "depth_result",
  "payload": {
    "source_id": <int>,
    "frame_id": <int>,
    "ts": <int>,
    "width": <int>,
    "height": <int>,
    "depth_map_ref": "<string>",
    "minmax": [<float min>, <float max>],
    "unit": "m"
  }
}
```

## 6. Tracking Telemetry (`type: tracking`)

Produced by `TrackingTelemetryPublisher`; people-only (class_id=0). `track_id` is internal and never exposed.

```json
{
  "type": "tracking",
  "source_id": <int>,
  "camera_id": "<string>",
  "coord_space": "<string>",
  "units": "<string>",
  "world_source": "backend_world_fused",
  "track_id_strategy": "camera_tracker_fallback",
  "calibration_version": "<string>",
  "tracking_contract_version": 3,
  "image_size": [<int width>, <int height>],
  "frame_size": [<int width>, <int height>],
  "tracks": [
    {
      "stable_id": <int>,
      "camera_id": "<string>",
      "bbox": [<float>, <float>, <float>, <float>],
      "center": [<float>, <float>],
      "class_id": 0,
      "confidence": <float|null>,
      "tracker_confidence": <float|null>,
      "analytics": { /* NvDsAnalytics obj meta (camel + snake keys) */ },
      "zone": "<string|null>",
      "frame_id": <int>,
      "dwell_time": <float|null>,
      "bbox3d": { /* when V3DT/SV3DT enabled */ },
      "velocity3d": [<float>, <float>, <float>],
      "visibility": <float|null>,
      "image_foot": [<float>, <float>],
      "image_base": [<float>, <float>],
      "world": [<float>, <float>, <float>],
      "world_valid": <bool>,
      "world_quality": "good"|"estimated"|"invalid",
      "world_quality_reason": "<string|null>",
      "world_frame": "menon_scene"|"camera_local"|null,
      "world_source": "bbox3d"|"pose_depth_fused"|"pose_floor_only"|"person_anchor_depth_fused"|"person_anchor_floor_only"|"gravity_drop"|"anchor_hold"|null,
      "projection_confidence": <float|null>,
      "temporal_confidence": <float|null>,
      "reid_confidence": <float|null>,
      "reid_identity": "<string|null>",
      "appearance_id": "<string|null>",
      "occluded": <bool|null>,
      "occlusion_uncertainty_m": <float|null>,
      "depth_status": "<string|null>",
      "depth_anchor_source": "<string|null>",
      "depth_anchor_m": <float|null>,
      "depth_used_m": <float|null>,
      "depth_registered_m": <float|null>,
      "depth_registration_status": "<string|null>",
      "depth_registration_id": "<string|null>",
      "depth_center_m": <float|null>,
      "depth_median_m": <float|null>,
      "depth_sample_count": <int|null>,
      "depth_valid_fraction": <float|null>,
      "depth_anchor_sample_count": <int|null>,
      "depth_anchor_valid_fraction": <float|null>
    }
  ]
}
```

Tracking telemetry has two world-source scopes:

- Top-level `world_source="backend_world_fused"` advertises that the baseline DS8 runtime owns the canonical world estimator in the backend.
- Per-track `world_source` records which observation path updated that specific track on the current frame.

Baseline non-`v3dt` mode uses one canonical person-anchor estimator: pose-derived image anchor when available, otherwise the person mask/depth image anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`. A concurrent DAv2 range observation from `NOESIS.OBJECT_DEPTH` is fused on that same current-anchor ray when valid. The canonical per-track values are:

- `pose_depth_fused`: pose anchor and DAv2 anchor depth both contributed to the world-state update.
- `pose_floor_only`: pose anchor updated the world-state filter without a usable DAv2 observation on that frame.
- `person_anchor_depth_fused`: the person mask/depth anchor (`anchor_uv`) plus DAv2 anchor depth both contributed to the world-state update on a frame without usable pose.
- `person_anchor_floor_only`: the person mask/depth anchor updated the world-state filter without a usable DAv2 observation on that frame.
- `gravity_drop`: no current admissible person anchor was available, but a stored pose-derived height reference allowed a floor-consistent gravity drop.
- `anchor_hold`: no current valid observation; the estimator is briefly holding the last reliable world state.
- `bbox3d`: `v3dt` mode only.

Depth exposure:

- `depth_used_m` is the DAv2 anchor depth that actually qualified for the fused estimator on that track update (`status="ok"` with sufficient support).
- `depth_anchor_m` is the raw anchor depth carried by `NOESIS.OBJECT_DEPTH`; it may be present even when `depth_used_m` is null.
- `depth_registered_m` is the room-registered DAv2 anchor depth after applying the offline DAv2→MapAnything mapping for that camera; this is the value projected on the current anchor ray when registration is active.
- `depth_registration_status` is `ok` when the runtime used a valid registration mapping on that frame. Other values explain why the estimator stayed on floor-only (`out_of_domain_or_invalid`) or why registration was unavailable.
- `depth_registration_id` identifies the exact per-camera registration artifact entry used by the estimator.
- `depth_status`, `depth_anchor_source`, `depth_sample_count`, and `depth_valid_fraction` are published on both `tracking.tracks[]` and `stats.payload.cameras[*].tracking.active_tracks[]` so the runtime OSD and dashboard can explain whether baseline depth is contributing on a given frame.
- `depth_anchor_sample_count` and `depth_anchor_valid_fraction` surface the support of the actual lower-body / torso anchor band that drove the fused update. These fields are the canonical explanation for why a track landed on `pose_depth_fused` / `person_anchor_depth_fused` versus `pose_floor_only` / `person_anchor_floor_only`; whole-mask support can be lower or noisier without disqualifying a good anchor-band sample.
- The on-screen `z=` label is sourced from the same `depth_used_m` value that the estimator actually projected, not directly from the raw `depth_anchor_m`.

When pose anchoring, gravity-drop, and recent-anchor hold all fail, DS8 leaves `world_valid=false` instead of promoting bbox-bottom floor projection into a synthetic world point.

Validation diagnostics:

- `projection_confidence`, `temporal_confidence`, `reid_confidence`,
  `reid_identity`, `appearance_id`, `occluded`, and
  `occlusion_uncertainty_m` are optional diagnostics for validation and UI
  explanation. Producers may omit them when that evidence is unavailable, but
  consumers must not reinterpret missing values as a pass.
- The Noesis/Menon validation toolbox consumes these fields when present for
  `TRACK.projection_confidence`, `TRACK.occlusion_bridge`,
  `TRACK.identity_continuity`, `TRACK.reid_geometry_consistency`, and
  per-track audit artifacts.
- Cross-space validation traces may refer to this track `world` vector as
  `backend_world_m` when handing it to Menon. Menon must apply its declared
  room/scene alignment exactly once and should expose the transform stages in
  trace/debug evidence for `MENON.transform_audit`.

## 7. Control & RPC Message Types

Handled in `websocket_server.py`:

- `clear_stats` → clears latency samples and broadcasts updated stats.
- `set_vis_toggle` → visualization toggle; server broadcasts `toggle_update`.
- `trail_settings_update` (server → client) → current trail tuning config snapshot, including `enabled`.
- `update_detection_config` / `set_detection_toggle` → broadcast updates to clients.
- `bev-config` / `bev-overlay` → update BEV renderer config; ack via `bev-config-ack` or `bev-overlay-update`.
- `ma_heatmap_ready` → notification only.
- Calibration RPCs: `pixel_to_world` → `pixel_to_world_response`; `set_extrinsics`, `set_align`, `solve_pnp` → corresponding `*_result` messages.
- Depth/MapAnything: `get_ma_depth` / `get_ma_depth_cache` → `ma_depth_response`.
- Floorplan: `get_floorplan` → `floorplan_response`.
- Auto-calibration: `auto_calibrate_pose` → `auto_calibrate_result`.
- Heartbeat: `ping` → `pong`.

### ma_depth_response

Request:

```json
{
  "type": "get_ma_depth",
  "camera": "<camera-id>",
  "request_id": "<optional>",
  "ts_max_us": <optional int>,
  "cache_only": <optional bool>
}
```

- `get_ma_depth_cache` uses the same request shape and response shape, but forces `cache_only=true`.

```json
{
  "type": "ma_depth_response",
  "camera": "<camera-id>",
  "cache_only": <bool>,
  "served_from_cache": <bool>,
  "ts_us": <int>,
  "request_id": "<optional>",
  "ok": <bool>,
  "error": "<optional>",
  "payload": {
    "ts": <int>,
    "depth_b64": "<base64 float32>",
    "conf_b64": "<base64 float32>",
    "mask_b64": "<base64 uint8>",
    "shape": [<H>, <W>],
    "normals_b64": "<base64 float16/float32 optional>",
    "normals_shape": [<H>, <W>, 3],
    "normals_dtype": "float16"|"float32",
    "normals_space": "camera"|"world",
    "normals_error": "<optional>"
  }
}
```

- `cache_only=true` and `get_ma_depth_cache` return only an existing valid cached MapAnything payload and must not enable the MapAnything depth gate. If no valid cached payload is available, the response has `ok:false`, `error:"no_cached_depth"`, and no `payload`.
- Without `cache_only`, a cache miss may enable the on-demand MapAnything branch according to runtime gating.
- Normals are attached when `NOESIS_MAPANYTHING_NORMALS_ENABLE` is truthy; errors are reported via `normals_error` while keeping the depth payload.
- `ma_depth_response` remains the MapAnything full-frame RPC contract. The always-on baseline DAv2 tracking lane does not publish a second full-frame WebSocket depth stream; it influences `track.world` through `NOESIS.OBJECT_DEPTH` and the fused backend estimator instead.

### floorplan_response

Returned from `get_floorplan` (`DepthStorageManager.generate_topdown_floorplan`):

```json
{
  "type": "floorplan_response",
  "request_id": "<id>",
  "camera_id": "<camera>",
  "cache_only": <bool>,
  "served_from_cache": <bool>,
  "ts": <int>,
  "snapshot_ts": <int|null>,
  "frame": "camera_local_ground",
  "orientation": "camera_xz_forward",
  "floorplan_contract_version": <int>,
  "units": "scene",
  "s_obj_to_m": <float>,
  "bounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>},
  "scale_m_per_px": <float>,
  "scale_scene_per_px": <float>,
  "point_count": <int>,
  "density": {"grid_b64": "<base64 float32>", "grid_shape": [<H>,<W>], "value_min": <float>, "value_max": <float>},
  "height": {"grid_b64": "<...>"},
  "height_agl": {"grid_b64": "<...>"},
  "distance": {"grid_b64": "<...>"},
  "obstacle_height": {"grid_b64": "<...>"},
  "walkable": {"grid_b64": "<...>"},
  "clean_floorplan_meta": {"...": "<optional debug metadata>"},
  "grid_res_m": <float>,
  "grid_res_scene": <float>,
  "max_extent_m": <float>,
  "max_extent_scene": <float>,
  "image_flip": {"u": <bool>, "v": <bool>},
  "error": "<string optional>"
}
```

- `obstacle_height` and `walkable` are optional clean layers generated for room floorplan responses when clean-surface estimation is available:
  - `obstacle_height`: float32 meters above an estimated floor plane (floor clamped to 0).
  - `walkable`: float32 {0,1} where 1 is walkable floor and 0 is obstacle/furniture.
- `image_flip` is diagnostic-only for floorplan responses. The serialized floorplan grids are already in their final camera-local X/Z orientation, so clients must not mirror the raster again using this hint.

### auto_calibrate_result

Current DS8 runtime proxies Menon auto-calibration: `{ type: "auto_calibrate_result", ok: <bool>, updated: [<cameraId>], results: [...], error?: <string|null> }`.

## 8. Calibration Data Conventions

Same as prior DS8 revisions:

- **Extrinsics (`E`)**: stored in `config/camera_calibration.json`, world→camera, 4×4 column-major, meters.
- **PoseV1 (`pose`)**: stored scene pose summaries (`position`, `yaw_pitch_roll_deg`, `rotation_order=YXZ`, `frame=menon_scene`) are authoritative scene-camera poses, not raw OpenCV image-camera extrinsics. Converting PoseV1 to `E` applies a fixed local 180 degree roll so the resulting camera basis matches DS8 depth/image math (`+X right`, `+Y down`, `+Z forward`).
- **Intrinsics (`K`)**: from `config/cameras.yaml` (`intrinsics_models` + `cameras` map). Scaled to streammux resolution in `ds8_runtime._CalibrationProvider`.
- **Alignment (`align`)**: `config/ply_alignment.json` with `matrix` (room-model alignment), `floor_y`, and `units.s_obj_to_m`.
- **Calibration bundle (`calibration-bundle`)**: carries the canonical backend-world-meters pose/extrinsics plus the room-alignment metadata Menon needs for its own scene conversion.
- `pixel_to_world_response` returns world-frame meters; Menon applies its room alignment and scene-unit conversion client-side.
