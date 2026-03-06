# DS8 WebSocket API Contracts
_Status: validated against code on 2026-02-22._

The WebSocket server (`websocket_server.WebSocketServer`) is the primary transport for DS8 telemetry, depth retrieval, and WebRTC signaling. All message types are JSON unless noted as binary.

## 1. Common Envelope

Most JSON messages have:

```json
{ "type": "<message-type>", ... }
```

Binary payloads (BEV JPEG) are framed as `[len(header)][header bytes][JPEG bytes]`.

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
          "active_tracks": [ /* diagnostic only */ ],
          "transitions": [ /* line/zone transitions */ ]
        }
      }
    }
  }
}
```

`latency_ms` fields are populated only when `NVDS_ENABLE_LATENCY_MEASUREMENT` is truthy and the DeepStream latency meta library is available.

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
  "footpoints": [ {"x": <float>, "y": <float>, "method": "bbox"|"sv3dt", "stableId": <int|null>, "trackerId": <int|null>} ],
  "H": [<9 floats>],
  "sampleXZ": [<float x>, <float z>] | null,
  "world_frame": "menon_scene"|"camera_local",
  "frame_mode": "world"|"camera_local",
  "units": "scene",
  "s_obj_to_m": <float>,
  "trail_smoothing_owner": "frontend"|"backend"|"none",
  "bev_world_points_smoothed": <bool>
}
```

- Optional JPEG binary: `[len(header)][header="bev:<camera>"][JPEG bytes]` when BEV JPEG output is enabled (`bev.jpeg_enabled` or `NOESIS_BEV_JPEG_ENABLED=1`).
- In world mode (`frame_mode=world`), BEV footpoints are producer-owned scene coordinates and backend motion smoothing is disabled (`trail_smoothing_owner=frontend`, `bev_world_points_smoothed=false`).
- Coordinate note: BEV renders on the ground plane (XZ). `footpoints[].x` is scene/world X, and `footpoints[].y` is scene/world Z.

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
      "world_frame": "menon_scene"|"camera_local"|null,
      "world_source": "bbox3d"|"ray_floor"|null
    }
  ]
}
```

## 7. Control & RPC Message Types

Handled in `websocket_server.py`:

- `clear_stats` → clears latency samples and broadcasts updated stats.
- `set_vis_toggle` → visualization toggle; server broadcasts `toggle_update`.
- `trail_settings_update` (server → client) → current trail tuning config snapshot, including `enabled`.
- `update_detection_config` / `set_detection_toggle` → broadcast updates to clients.
- `bev-config` / `bev-overlay` → update BEV renderer config; ack via `bev-config-ack` or `bev-overlay-update`.
- `ma_heatmap_ready` → notification only.
- Calibration RPCs: `pixel_to_world` → `pixel_to_world_response`; `set_extrinsics`, `set_align`, `solve_pnp` → corresponding `*_result` messages.
- Depth/MapAnything: `get_ma_depth` → `ma_depth_response`.
- Floorplan: `get_floorplan` → `floorplan_response`.
- Auto-calibration: `auto_calibrate_pose` → `auto_calibrate_result`.
- Heartbeat: `ping` → `pong`.

### ma_depth_response

```json
{
  "type": "ma_depth_response",
  "camera": "<camera-id>",
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

- Normals are attached when `NOESIS_MAPANYTHING_NORMALS_ENABLE` is truthy; errors are reported via `normals_error` while keeping the depth payload.

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
  "orientation": "xz",
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

- `obstacle_height` and `walkable` are optional clean layers (currently kitchen-only):
  - `obstacle_height`: float32 meters above an estimated floor plane (floor clamped to 0).
  - `walkable`: float32 {0,1} where 1 is walkable floor and 0 is obstacle/furniture.

### auto_calibrate_result

Current DS8 runtime proxies Menon auto-calibration: `{ type: "auto_calibrate_result", ok: <bool>, updated: [<cameraId>], results: [...], error?: <string|null> }`.

## 8. Calibration Data Conventions

Same as prior DS8 revisions:

- **Extrinsics (`E`)**: stored in `config/camera_calibration.json`, world→camera, 4×4 column-major, meters.
- **Intrinsics (`K`)**: from `config/cameras.yaml` (`intrinsics_models` + `cameras` map). Scaled to streammux resolution in `ds8_runtime._CalibrationProvider`.
- **Alignment (`align`)**: `config/ply_alignment.json` with `matrix` (row-major), `floor_y`, `units.s_obj_to_m`.
- **Calibration bundle (`calibration-bundle`)**: for client-side visualization, extrinsic translations are exposed in native scene units via `scene_per_m = 1 / s_obj_to_m` (rotation unchanged).
- `pixel_to_world_response` returns world-frame meters; Menon applies `align.matrix` client-side.
