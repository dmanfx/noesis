# Telemetry Schema (DS8)
_Status: validated against shared DS8/DS9 contracts on 2026-07-11._

Canonical WebSocket payloads live in `docs/DS8_api_contracts_ws.md`. This page summarizes **where** telemetry is produced in the DS8 stack and the exact field sets emitted today. Older telemetry descriptions are archived under `docs/history/`.

## Producers and Message Types

- **Stats** – built in `noesis/ds8_runtime.py:_build_stats_callback` and broadcast once per second when clients are connected.
- **Tracking** – `TrackingTelemetryPublisher.publish` in `noesis/telemetry/publishers.py`, fed by `_AnalyticsTelemetryProcessor` (`noesis/pipelines/hooks.py`).
- **Dense depth telemetry** – `DepthTelemetryPublisher.publish` in
  `noesis/telemetry/publishers.py`, fed by the gated `MapAnythingProcessor`.
- **Tracking depth evidence** – the always-on baseline DAv2 branch rendezvous
  with object metadata inside `_ObjectDepthFusionProcessor`; it attaches
  `NOESIS.OBJECT_DEPTH` and contributes to strict observations/world state, but
  does not emit a second full-frame depth message.
- **BEV** – `BevRenderer._publish` in `noesis/telemetry/bev.py` (JSON only;
  JPEG delivery is retired).

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
      "bev": {
        "frame": "camera_local_ground_m",
        "health": {
          "contract": "noesis.bev.health",
          "contract_version": 2,
          "healthy": <bool>,
          "config_ready": <bool>,
          "renderer_ready": <bool>,
          "rendering_active": <bool>,
          "configured_camera_count": <int>,
          "active_camera_count": <int>,
          "inactive_camera_count": <int>,
          "failed_camera_count": <int>,
          "cameras": { /* active_ready, inactive_ready, or failed */ }
        }
      },
      "active_floorplan": {
        "contract": "noesis.active_floorplan.health",
        "contract_version": 1,
        "healthy": <bool>,
        "configured_camera_count": <int>,
        "active_camera_count": <int>,
        "missing_cameras": [<camera-id>, ...],
        "rejection_count": <int>,
        "stale_count": <int>,
        "conflict_count": <int>,
        "cameras": { /* exact accepted snapshot identity by camera */ }
      },
      "capture_event_fusion": {
        "contract": "noesis.capture_event_controller_health",
        "contract_version": 1,
        "healthy": <bool>,
        "shared_gate_scope": "process",
        "shared_gate_owned": <bool>,
        "last_fatal_error_code": <string|null>,
        "counters": { /* bounded admission/fusion/failure counters */ },
        "cameras": { /* bounded per-camera state */ }
      },
      "zero_copy_core": {
        "counters": {"<counter-name>": <int>, ...},
        "stage_timings": {"<stage-name>": { /* ... */ }, ...},
        "boundary_serialization_metrics": {"ws": { /* ... */ }, "rest": { /* ... */ }}
      },
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
- `bev.health`, `active_floorplan`, and `capture_event_fusion` are bounded
  fixed-schema health surfaces shared by DS8, protected V3DT, and DS9. BEV v2
  reports configuration/renderer readiness separately from occupied render
  activity: an empty camera is `inactive_ready`, while a real homography or
  publication failure is `failed`. Active-floorplan health still requires an
  exact, calibration-current record for every configured camera; capture-event
  health requires that no request owns the per-camera/process-wide MapAnything
  valve admission and that no fatal barrier error remains.
- `active_floorplan.cameras[*]` binds BEV bounds to the accepted portable
  snapshot reference, write ID, content digest, calibration fingerprint, and
  snapshot/floorplan timestamps. Extrinsics invalidation clears the affected
  camera; alignment invalidation clears the whole registry. Persisted cache
  records with a mismatched calibration fingerprint are not reactivated.
- `zero_copy_core.counters` is the operational truth for the DAv2 bridge. The
  exact-frame path uses `depth_bridge_put_total`,
  `depth_bridge_exact_resolve_total`, `depth_bridge_wait_total`, and
  `depth_bridge_wait_timeout_total`; bounded prior-frame use and misses use
  `depth_bridge_lagged_resolve_total`, `depth_bridge_lagged_age_frames_total`,
  `depth_bridge_lagged_age_us_total`, and `depth_bridge_miss_total`.
  Successful object-meta attachment uses `object_depth_attach_total` plus
  `object_depth_status_total.<status>`; failures use
  `object_depth_attach_failure_total` plus a reason suffix such as
  `.native_rejected`. Failure counters never double as successful status
  counters.

## Tracking Payload (type: `tracking`)

Emitted per source on the tracking publish gate. Non-empty frames are bounded by `NOESIS_TRACKING_PUBLISH_MAX_HZ` / `NOESIS_WS_TRACKING_MAX_HZ`; sustained empty frames use `NOESIS_TRACKING_EMPTY_HEARTBEAT_HZ` (default 2 Hz). Only **people** tracks (class_id=0) are published; raw tracker IDs remain internal.

**Empty frames are first-class:** when a camera has zero people, Noesis still publishes `tracks: []` with an advancing top-level `frame_id` (count transitions to zero always force a publish, followed by the bounded empty heartbeat). Downstream clients such as Menon use empty lists to clear presence immediately and use the frame sequence to distinguish zero occupancy from a stalled producer. An empty source frame also removes that camera's evidence from the canonical world immediately; tracker shadow age already owns brief detector occlusion.

The tracking envelope, its world snapshot, and its world events are one
release-gated cohort. Bounded sender admission freezes the bytes, synchronous
exact-count journal acknowledgement and private world commit happen next, and
only then may client delivery begin. Commit failure aborts the complete cohort
with no partial visibility.

```json
{
  "type": "tracking",
  "source_id": <int>,
  "frame_id": <int>,
  "track_count": <int>,
  "observed_at_us": <int epoch microseconds>,
  "capture_time_status": "estimated",
  "media_pts_ns": <int>,
  "camera_id": "<camera>",
  "coord_space": "<string>",
  "units": "<string>",
  "world_source": "backend_world_fused",
  "track_id_strategy": "camera_tracker_fallback",
  "calibration_version": "<string>",
  "tracking_contract_version": 3,
  "observation_contract": "noesis.observation.person",
  "observation_contract_version": 1,
  "observations": [ /* strict ObservationEnvelope v1 rows */ ],
  "world_snapshot": { /* strict WorldSnapshot v1 */ },
  "world_events": [ /* strict WorldEvent v1 rows */ ],
  "tracks": [
    {
      "stable_id": <int|null>,
      "tracker_id": <int>,
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
      "world_frame": "backend_world_m"|null,
      "world_source": "bbox3d"|"pose_depth_fused"|"pose_floor_only"|"person_anchor_depth_fused"|"person_anchor_floor_only"|"gravity_drop"|"anchor_hold"|null,
      "embedding_present": <bool|null>,
      "embedding_sequence": <int|null>,
      "embedding_model_sha256": "<lowercase sha256|null>",
      "embedding_dimension": <int|null>,
      "pose_present": <bool|null>,
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

- `stable_id` is the numeric compatibility identity and may be null for an
  authoritative open-set unknown; raw tracker IDs remain internal compatibility
  data and are not durable person identities.
- `tracker_id` is a camera/run-local association key used to join the public
  track to its strict observation. It is not a user-facing identity and is not
  stable across restarts.
- `observations` is the canonical product surface. `tracks` is the compatible
  operator/diagnostic view, and every person row used by semantic acceptance
  must associate with one same-frame strict observation.
- `embedding_sequence`, `embedding_model_sha256`, and `embedding_dimension`
  form an all-or-none persisted-provenance triad on both associated surfaces.
  They are stamped only after the exact private identity-evidence row is
  durably appended. They are absent when evidence capture is off, append fails,
  or identity is only continuity-held. Partial triads are invalid, and no raw
  embedding vector is public.
- Top-level `world_source="backend_world_fused"` means the backend owns the canonical baseline world estimator; per-track `world_source` records which observation path updated that track on the current frame.
- In baseline non-`v3dt` mode, `world` is produced by the shared `PersonGroundState` estimator (`noesis/telemetry/person_ground_state.py`): posture-aware pose contact when available, otherwise the person mask/depth anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`, with human CV filtering and stationary lock. Tracks may also carry `motion_mode`, `posture`, `trail_append_allowed`, and `idle_jitter_m`. In `v3dt` mode, `world_source="bbox3d"` continues to come from tracker 3D metadata.
- `depth_used_m` is the DAv2 anchor depth that actually contributed to the fused baseline world update on that frame; `depth_anchor_m` remains the raw anchor carried by `NOESIS.OBJECT_DEPTH`.
- `depth_registered_m` is the room-registered DAv2 anchor depth after applying the offline DAv2->MapAnything mapping for that camera; this is the value the estimator projects when registration is active.
- `depth_registration_status` and `depth_registration_id` make the registration path observable on both tracks and active-tracks without changing the raw `NOESIS.OBJECT_DEPTH` payload semantics.
- `depth_anchor_sample_count` and `depth_anchor_valid_fraction` describe the support of the specific lower-body / torso anchor band that drove the fused update; they are more authoritative than whole-mask support when diagnosing why a far-camera track fused depth or stayed floor-only.
- `stats.payload.cameras[*].tracking.active_tracks[]` mirrors the same depth-registration fields for the current camera, and the runtime OSD `z=` label uses `depth_used_m` / registered depth rather than raw `depth_anchor_m`.
- `observations[].payload.depth_present` is true only for `depth_status="ok"`,
  `depth_registration_status="ok"`, and a finite positive
  `depth_registered_m`. If `depth_used_m` is present it must match the
  registered value. Missing, raw-passthrough, or rejected registration and raw
  anchor depth do not qualify.

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
    "depth_map_ref": "noesis-depth://artifact/<sha256>",
    "minmax": [<float min>, <float max>],
    "unit": "m"
  }
}
```

- This full-frame `depth_result` contract remains MapAnything-specific. The always-on DAv2 tracking lane does not publish a second full-frame depth message; it contributes through `NOESIS.OBJECT_DEPTH` and the fused tracking world update.
- `depth_map_ref` is an opaque artifact identifier, not a filesystem path or
  fetch URL.
- Baseline non-`v3dt` startup also requires a read-only depth-registration artifact (`config/depth_registration.json` by default, overridable via `depth_registration.path` in `infer.yaml` or `--depth-registration-config`). That artifact is loaded before activation and is not generated automatically by the runtime.

Fresh `ma_depth_response` and floorplan RPCs use a separate capture-event
contract. One shared controller owns the process-wide MapAnything valve plus
per-camera admission. It fences the worker and transactional store before and
after a bounded burst, fuses only raw snapshots newer than the baseline, and
reloads the exact fused result by portable reference, write ID, timestamp, and
content/manifest evidence. The current capture mode is depth-only; sealed
fusion evidence records RGB as `not_requested`. Cache-only RPCs return an
existing valid cache entry or `no_cached_depth`/`no_cached_floorplan` without
capture, generation, persistence, or active-floorplan mutation. Integrity
errors remain stable machine codes and never return a stale payload as success.

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
  "boundsSource": "config"|"auto_extents"|"active_floorplan"|"active_floorplan_plus_coverage_envelope",
  "displayBounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>},
  "floorplanCoordinateSpace": "floorplan_normalized_v1",
  "floorplanBounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>} | null,
  "coverageEnvelope": {
    "contract": "noesis.bev.coverage_envelopes",
    "contractVersion": 1,
    "frame": "camera_local_ground_m",
    "units": "meters",
    "cameraId": "<camera>",
    "boundaryToleranceM": <float>,
    "regions": [{"id": "<semantic-region>", "polygonXZ": [[<float x>, <float z>], "..."]}]
  },
  "floorplanGridShape": [<rows>, <cols>] | null,
  "floorplanGridResM": <float|null>,
  "floorplanSnapshotTsUs": <int|null>,
  "floorplanTsUs": <int|null>,
  "overlay": <bool>,
  "footpoints": [ {"x": <float>, "y": <float>, "method": "bbox"|"image_foot"|"image_base"|"<string>", "stableId": <int|null>, "trackerId": <int|null>, "displaySource": "world"|"world_to_camera_local"|"image_anchor"|"image_depth_anchor"|"registered_depth_anchor"|"floor_contact_ray", "motionMode": "walk"|"idle"|"sit"|"lie"|"unknown"|null, "posture": "standing"|"sitting"|"lying"|"unknown"|null, "trailAppendAllowed": <bool|null>, "idleJitterM": <float|null>, "floorplanInside": <bool>, "coverageInside": <bool>, "coverageRegion": "<semantic-region>"|null} ],
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
- The primary inline floorplan BEV uses `frame=camera_local_ground_m`;
  `footpoints[].x` is local X and `footpoints[].y` is local Z in meters so
  tracks are drawn in the same coordinate frame as MapAnything floorplan
  rasters. Its bounds/grid/timestamps come only from the bounded active-floorplan
  registry. The separate strict observations and global world snapshot remain
  canonical `backend_world_m`; camera-local BEV does not redefine world state.
- For cameras with `coverageEnvelope`, semantic polygon union owns point/trail
  admission and `displayBounds` covers both that union and the active floorplan
  raster. `floorplanBounds` remains the raster's exact metric footprint, so
  `coverageInside=true` with `floorplanInside=false` is valid. The dashboard
  renders those points from raw metric coordinates and leaves the area outside
  the raster visibly unknown; it does not stretch the raster, clamp the point,
  or add position smoothing. Cameras without an envelope retain the
  active-floorplan rectangle gate.
- Calibration invalidation removes affected active records. The renderer resets
  camera history when its active-floorplan coordinate signature changes and
  does not reuse a previous homography after a current calibration/homography
  failure.
- `displaySource=registered_depth_anchor` is the active floorplan placement when
  a valid registered DAv2 person anchor is available. `image_depth_anchor` is a
  legacy/non-person direct image-depth path; static MapAnything snapshots are
  not live person-placement evidence. When registration rejects a sample, BEV
  does not promote raw object depth onto the floorplan.
- In world mode, BEV head points are the canonical backend `track.world` positions and are not low-pass filtered a second time inside `BevRenderer`.
- `trail_smoothing_owner="backend"` with `bev_world_points_smoothed=false` is valid and expected in the baseline world-mode path: the backend owns trail history, while `PersonGroundState` (human CV filter + stationary lock) owns the only track-position filtering stage.
- When `trailAppendAllowed` is false, producer trails do not grow new path samples (person is idle/sit/lie locked).
- World-mode BEV skips `anchor_hold` head points and trail samples so stale held positions do not render as drifting or out-of-bounds trails during brief occlusions.

## Related Docs

- Contracts and RPCs: `docs/DS8_api_contracts_ws.md`
- Metadata shapes: `docs/DS8_metadata_contracts.md`
- BEV rendering and smoothing: `docs/DS8_Baselines.md` (BEV section)
