# Telemetry Schema (DS8)
_Status: validated against code on 2026-03-16._

Canonical WebSocket payloads live in `docs/DS8_api_contracts_ws.md`. This page summarizes **where** telemetry is produced in the DS8 stack and the exact field sets emitted today. Older telemetry descriptions are archived under `docs/history/`.

## Producers and Message Types

- **Stats** – built in `noesis/ds8_runtime.py:_build_stats_callback` and broadcast once per second when clients are connected.
- **Tracking** – `TrackingTelemetryPublisher.publish` in `noesis/telemetry/publishers.py`, fed by `_AnalyticsTelemetryProcessor` (`noesis/pipelines/hooks.py`).
- **Depth** – `DepthTelemetryPublisher.publish` in `noesis/telemetry/publishers.py`, fed by `MapAnythingProcessor`.
- **BEV** – `BevRenderer._publish` in `noesis/telemetry/bev.py` (JSON always; optional JPEG when enabled).

## Stats Payload (type: `stats`)

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
      "mosaic_layout": { /* may be null */ },
      "latency_ms": { /* present only when NVDS latency is enabled */ },
      "errors": [<string>, ...]
    },
    "cameras": {
      "<camera_id>": {
        "fps": <float>,
        "frames_processed": <int>,
        "status": "running"|"unknown",
        "tracking": {
          "occupancy": {"<zone>": <int>, ...},
          "active_tracks": [ /* same per-track shape as tracking.tracks[] for the current camera */ ],
          "transitions": [ /* line/zone transitions */ ]
        },
        "latency_ms": { /* present when NVDS latency enabled */ }
      }
    }
  }
}
```

Notes:
- `latency_ms` comes from `LatencyCollector` and is populated only when `NVDS_ENABLE_LATENCY_MEASUREMENT` is truthy and the DeepStream latency meta library is available.
- `mosaic_layout` mirrors the current tiler layout: `mosaic_w/h`, optional `rows/cols`, `square_seq_grid`, `frame_w/h`, `source_count`, and `sources` array of `{source_id, camera_id}`.

## Tracking Payload (type: `tracking`)

Emitted once per frame per source. Only **people** tracks (class_id=0) are published; raw tracker IDs remain internal.

```json
{
  "type": "tracking",
  "source_id": <int>,
  "camera_id": "<camera>",
  "coord_space": "<string>",
  "units": "<string>",
  "world_source": "backend_world_fused",
  "track_id_strategy": "camera_tracker_fallback",
  "calibration_version": "<string>",
  "tracking_contract_version": 3,
  "tracks": [
    {
      "stable_id": <int>,
      "camera_id": "<camera>",
      "bbox": [<float left>, <float top>, <float width>, <float height>],
      "center": [<float cx>, <float cy>],
      "class_id": 0,
      "confidence": <float|null>,
      "tracker_confidence": <float|null>,
      "analytics": { /* NvDsAnalytics obj meta, camel + snake case keys */ },
      "zone": "<string|null>",
      "frame_id": <int>,
      "dwell_time": <float|null>,
      "bbox3d": { /* present in V3DT/SV3DT mode */ },
      "velocity3d": [<float x>, <float y>, <float z>],
      "visibility": <float|null>,
      "image_foot": [<float u>, <float v>],
      "image_base": [<float u>, <float v>],
      "world": [<float x>, <float y>, <float z>],
      "world_valid": <bool>,
      "world_quality": "good"|"estimated"|"invalid",
      "world_quality_reason": "<string|null>",
      "world_frame": "menon_scene"|"camera_local"|null,
      "world_source": "bbox3d"|"pose_depth_fused"|"pose_floor_only"|"person_anchor_depth_fused"|"person_anchor_floor_only"|"gravity_drop"|"anchor_hold"|null,
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

- `stable_id` is always present for people tracks; `track_id` is never exposed.
- Top-level `world_source="backend_world_fused"` means the backend owns the canonical baseline world estimator; per-track `world_source` records which observation path updated that track on the current frame.
- In baseline non-`v3dt` mode, `world` is produced by the shared `PersonGroundState` estimator (`noesis/telemetry/person_ground_state.py`): posture-aware pose contact when available, otherwise the person mask/depth anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`, with human CV filtering and stationary lock. Tracks may also carry `motion_mode`, `posture`, `trail_append_allowed`, and `idle_jitter_m`. In `v3dt` mode, `world_source="bbox3d"` continues to come from tracker 3D metadata.
- `depth_used_m` is the DAv2 anchor depth that actually contributed to the fused baseline world update on that frame; `depth_anchor_m` remains the raw anchor carried by `NOESIS.OBJECT_DEPTH`.
- `depth_registered_m` is the room-registered DAv2 anchor depth after applying the offline DAv2->MapAnything mapping for that camera; this is the value the estimator projects when registration is active.
- `depth_registration_status` and `depth_registration_id` make the registration path observable on both tracks and active-tracks without changing the raw `NOESIS.OBJECT_DEPTH` payload semantics.
- `depth_anchor_sample_count` and `depth_anchor_valid_fraction` describe the support of the specific lower-body / torso anchor band that drove the fused update; they are more authoritative than whole-mask support when diagnosing why a far-camera track fused depth or stayed floor-only.
- `stats.payload.cameras[*].tracking.active_tracks[]` mirrors the same depth-registration fields for the current camera, and the runtime OSD `z=` label uses `depth_used_m` / registered depth rather than raw `depth_anchor_m`.

## Depth Payload (type: `depth_result`)

Published per MapAnything inference result via `DepthResult.to_dict()`:

```json
{
  "type": "depth_result",
  "payload": {
    "source_id": <int>,
    "frame_id": <int>,
    "ts": <int epoch_seconds>,
    "width": <int>,
    "height": <int>,
    "depth_map_ref": "<path or opaque ref>",
    "minmax": [<float min>, <float max>],
    "unit": "m"
  }
}
```

- This full-frame `depth_result` contract remains MapAnything-specific. The always-on DAv2 tracking lane does not publish a second full-frame depth message; it contributes through `NOESIS.OBJECT_DEPTH` and the fused tracking world update.
- Baseline non-`v3dt` startup also requires a read-only depth-registration artifact (`config/depth_registration.json` by default, overridable via `depth_registration.path` in `infer.yaml` or `--depth-registration-config`). That artifact is loaded before activation and is not generated automatically by the runtime.

## BEV Payload (type: `bev-frame`)

```json
{
  "type": "bev-frame",
  "cameraId": "<camera>",
  "ts": <int microseconds>,
  "w": <int>,
  "h": <int>,
  "mpp": <float meters_per_px>,
  "xMin": <float>, "xMax": <float>,
  "zMin": <float>, "zMax": <float>,
  "overlay": <bool>,
  "footpoints": [ {"x": <float>, "y": <float>, "method": "bbox"|"image_foot"|"image_base"|"<string>", "stableId": <int|null>, "trackerId": <int|null>, "displaySource": "world"|"world_to_camera_local"|"image_anchor"|"image_depth_anchor"|"registered_depth_anchor"|"floor_contact_ray", "motionMode": "walk"|"idle"|"sit"|"lie"|"unknown"|null, "posture": "standing"|"sitting"|"lying"|"unknown"|null, "trailAppendAllowed": <bool|null>, "idleJitterM": <float|null>} ],
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

- BEV JPEG binary delivery is retired. BEV is metadata-only; `bev.jpeg_enabled`
  and `NOESIS_BEV_JPEG_ENABLED` are ignored by the DS8 runtime.
- Footpoints use `stableId` when available; `trackerId` may be present as a debug/fallback identity key (not stable across restarts).
- The primary inline floorplan BEV uses `frame=camera_local_ground_m`; `footpoints[].x` is local X and `footpoints[].y` is local Z in meters so tracks are drawn in the same coordinate frame as MapAnything floorplan rasters.
- `displaySource=image_depth_anchor` indicates a camera-local X/Z point unprojected from the image anchor and MapAnything-registered fused depth, matching the floorplan depth-unprojection basis. When depth registration rejects a sample, BEV display does not use raw object depth for the floorplan overlay.
- In world mode, BEV head points are the canonical backend `track.world` positions and are not low-pass filtered a second time inside `BevRenderer`.
- `trail_smoothing_owner="backend"` with `bev_world_points_smoothed=false` is valid and expected in the baseline world-mode path: the backend owns trail history, while `PersonGroundState` (human CV filter + stationary lock) owns the only track-position filtering stage.
- When `trailAppendAllowed` is false, producer trails do not grow new path samples (person is idle/sit/lie locked).
- World-mode BEV skips `anchor_hold` head points and trail samples so stale held positions do not render as drifting or out-of-bounds trails during brief occlusions.

## Related Docs

- Contracts and RPCs: `docs/DS8_api_contracts_ws.md`
- Metadata shapes: `docs/DS8_metadata_contracts.md`
- BEV rendering and smoothing: `docs/DS8_Baselines.md` (BEV section)
