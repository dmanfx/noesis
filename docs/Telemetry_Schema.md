# Telemetry schema
_Status: canonical native DS9.1 summary, updated 2026-08-23._

Canonical WebSocket payloads live in `docs/api_contracts_ws.md`. This page summarizes **where** telemetry is produced in the DS9.1 stack and the exact field sets emitted today. Older telemetry descriptions are archived under `docs/history/`.

## Producers and Message Types

- **Stats** – built in `DS9/noesis/ds9_runtime_core.py:_build_stats_callback` and broadcast once per second when clients are connected.
- **Tracking** – `TrackingTelemetryPublisher.publish` in the DS9.1-owned
  telemetry module, fed by `_AnalyticsTelemetryProcessor` in
  `DS9/noesis/pipelines/hooks.py`.
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
    "stack": "ds9",
    "application": {
      "running": <bool>,
      "cameras_active": <int>,
      "processors_active": <int>
    },
    "pipeline": {
      "stack": "ds9",
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
  fixed-schema health surfaces shared by the DS9.1 baseline and disabled V3DT
  adapter. BEV v2
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
      "world_frame_revision": "<revision|null>",
      "world_source": "bbox3d"|"pose_depth_fused"|"pose_depth_only"|"pose_floor_only"|"person_anchor_depth_fused"|"person_anchor_depth_only"|"person_anchor_floor_only"|"gravity_drop"|"cv_prediction"|"image_motion_prediction"|"anchor_hold"|null,
      "world_filter_prediction": [<float x>, <float y>, <float z>]|null,
      "world_prediction_image_foot": [<float u>, <float v>]|null,
      "world_prediction_provenance": { /* bounded non-authoritative image-motion provenance */ }|null,
      "world_measurement_accepted": <bool|null>,
      "world_rejection_reason": "<string|null>",
      "world_contact_basis": "<string|null>",
      "world_image_motion_supported": <bool|null>,
      "world_image_motion_streak": <int|null>,
      "world_state_continuity": "restored_short_ghost"|null,
      "world_reacquire_count": <int|null>,
      "world_reacquired": <bool|null>,
      "motion_mode": "walk"|"idle"|"sit"|"lie"|"unknown"|null,
      "posture": "standing"|"sitting"|"lying"|"unknown"|null,
      "trail_append_allowed": <bool|null>,
      "trail_break_required": <bool|null>,
      "trail_segment_id": <int|null>,
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
- Top-level `world_source="backend_world_fused"` means the backend owns the canonical baseline world estimator; per-track `world_source` records which observation path updated that track on the current frame. `world_frame_revision` identifies the exact active calibration/Scene Prior revision and must be preserved by every spatial consumer.
- In the active baseline, `world` is produced by the shared
  `PersonGroundState` estimator (`noesis/telemetry/person_ground_state.py`):
  posture-aware pose contact when available, otherwise the person mask/depth
  anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`, with human CV filtering and
  stationary lock. Tracks may also carry `motion_mode`, `posture`,
  `trail_append_allowed`, and `idle_jitter_m`.
- `cv_prediction` is a bounded constant-velocity continuation of the
  canonical filtered world point when the current metric observation is
  missing, stale, or physically rejected. It is not a fresh measurement:
  `world_measurement_accepted` is
  false, `world_filter_prediction` records the displayed prediction, and the
  last-good measurement state is not advanced. Reject-driven prediction uses
  one fixed last-good rejection anchor and is capped at 0.40 seconds; repeated
  rejects do not integrate drift. `anchor_hold` is a retained last-good point
  after a missing or rejected measurement; it may remain visible in BEV with
  `worldAdmission="held"`, but its trail must not grow. Its normal cap is 0.40
  seconds. A seated/lying lifecycle may hold for at most 2.0 seconds only while
  exact-frame bbox evidence remains stationary. A predicted point may extend a
  trail only while `trail_append_allowed=true`.
  `image_motion_prediction` is the stricter projective continuation for a
  materially moving detector box when the current metric/depth anchor is
  missing or rejected. It transports the last physically accepted image foot
  through the current bbox affine change and projects that pixel through the
  active corrected floor plane. It never updates the filter or prediction
  origin; `world_prediction_image_foot` and
  `world_prediction_provenance` expose its non-authoritative basis. If image,
  ray, metric-speed, or TTL gates fail, the producer fails closed instead of
  publishing a frozen point.
- World/filter/lock state is discarded on the first exact processed frame that
  omits a tracker key. A later reuse of the same numeric tracker ID starts a new
  world lifecycle and cannot inherit the prior position or velocity.
- `depth_used_m` is the DAv2 anchor depth that actually contributed to the fused baseline world update on that frame; `depth_anchor_m` remains the raw anchor carried by `NOESIS.OBJECT_DEPTH`.
- `depth_registered_m` is the room-registered DAv2 anchor depth after applying the offline DAv2->MapAnything mapping for that camera; this is the value the estimator projects when registration is active.
- `depth_registration_status` and `depth_registration_id` make the registration path observable on both tracks and active-tracks without changing the raw `NOESIS.OBJECT_DEPTH` payload semantics.
- `depth_anchor_sample_count` and `depth_anchor_valid_fraction` describe the support of the specific lower-body / torso anchor band that drove the fused update; they are more authoritative than whole-mask support when diagnosing why a far-camera track fused depth or stayed floor-only.
- `stats.payload.cameras[*].tracking.active_tracks[]` mirrors the same depth-registration fields for the current camera, and the runtime OSD `depth=` label uses `depth_used_m` / registered depth rather than raw `depth_anchor_m`.
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
- Baseline startup also requires the read-only depth-registration artifact at
  `DS9/config/depth_registration.json`, selected by `depth_registration.path`
  in `DS9/config/infer.yaml` or an explicit command argument. It is loaded
  before sources open and is not generated automatically by the runtime.

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
    "bounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>},
    "regions": [{"id": "<semantic-region>", "polygonXZ": [[<float x>, <float z>], "..."]}]
  },
  "floorplanGridShape": [<rows>, <cols>] | null,
  "floorplanGridResM": <float|null>,
  "floorplanSnapshotTsUs": <int|null>,
  "floorplanTsUs": <int|null>,
  "overlay": <bool>,
  "footpoints": [ {"x": <float>, "y": <float>, "method": "bbox"|"image_foot"|"image_base"|"<string>", "stableId": <int|null>, "trackerId": <int|null>, "trackerLifecycleGeneration": <int|null>, "frameId": <int>, "anchorSource": "<string|null>", "displaySource": "world"|"world_to_camera_local"|"world_floor_fallback_to_camera_local"|"image_anchor"|"image_depth_anchor"|"registered_depth_anchor"|"floor_contact_ray", "canonicalWorld": <bool>, "worldAdmission": "accepted"|"predicted"|"held"|null, "worldFrame": "backend_world_m"|null, "worldFrameRevision": "<revision>"|null, "motionMode": "walk"|"idle"|"sit"|"lie"|"unknown"|null, "posture": "standing"|"sitting"|"lying"|"unknown"|null, "trailAppendAllowed": <bool|null>, "idleJitterM": <float|null>, "floorplanInside": <bool>, "coverageInside": <bool>, "coverageRegion": "<semantic-region>"|null} ],
  "droppedFootpointCount": <int>,
  "droppedFootpoints": [ {"stableId": <int|null>, "trackerId": <int|null>, "trackerLifecycleGeneration": <int|null>, "trailSegmentId": <int|null>, "reason": "<canonical admission reason>", "canonicalWorld": true, "anchorSource": "<string|null>"} ],
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
  and `NOESIS_BEV_JPEG_ENABLED` are ignored by the DS9.1 runtime.
- Footpoints use `stableId` when available; `trackerId` may be present as a debug/fallback identity key (not stable across restarts).
- The primary inline floorplan BEV uses `frame=camera_local_ground_m`;
  `footpoints[].x` is local X and `footpoints[].y` is local Z in meters so
  tracks are drawn in the same coordinate frame as MapAnything floorplan
  rasters. Its bounds/grid/timestamps come only from the bounded active-floorplan
  registry. The separate strict observations and global world snapshot remain
  canonical `backend_world_m`; camera-local BEV does not redefine world state.
- For cameras with `coverageEnvelope`, the semantic polygon union is the
  producer-side admission surface for legacy/noncanonical candidates and
  `displayBounds` covers both that union and the active floorplan raster.
  Canonical world points remain producer-authoritative: `coverageInside=false`
  is diagnostic provenance and is not silently re-applied as a second
  dashboard-only rejection gate. `floorplanBounds` remains the raster's exact
  metric footprint, so a displayed point may have `floorplanInside=false`. The
  dashboard renders admitted points from raw metric coordinates and leaves the
  area outside the raster visibly unknown; it does not stretch the raster,
  clamp the point, or add position smoothing. Without an envelope, the active
  PCF rectangle remains the semantic `floorplanBounds`, while the producer
  publishes the exact fixed, bounded display envelope for canonical live
  points. A finite point inside that envelope is emitted unchanged; points
  beyond it remain explicit bounded drops. The display envelope is a
  presentation/sanity contract only and never makes PCF evidence live tracking
  authority or auto-fits to tracks.
- Calibration invalidation removes affected active records. The renderer resets
  camera history when its active-floorplan coordinate signature changes and
  does not reuse a previous homography after a current calibration/homography
  failure.
- `displaySource=world_to_camera_local` is the only live canonical tracker
  placement path in the active floorplan BEV: it is the revision-checked
  horizontal projection of canonical `track.world`.
  `world_floor_fallback_to_camera_local` is a legacy/noncanonical diagnostic
  path retained for compatibility when a noncanonical world fallback is
  projected into the camera-local view; it must never be used to place an
  active canonical tracked person. Registered depth, floor-contact rays, and
  image anchors are diagnostics only; static MapAnything snapshots are not
  live person-placement evidence. When registration rejects a sample, BEV
  does not promote raw object depth onto the floorplan.
- `camera_local_ground_m` uses horizontal camera-right and camera-forward axes
  projected onto the gravity-aligned floor. Camera pitch and camera height do
  not contribute to local X/Z; the floor point directly below the camera is
  `(0, 0)`. Consumers must not reconstruct this transform from a raw or
  differently revisioned calibration matrix.
- In world mode, BEV head points are the canonical backend `track.world` positions and are not low-pass filtered a second time inside `BevRenderer`.
- `trail_smoothing_owner="backend"` with `bev_world_points_smoothed=false` is valid and expected in the baseline world-mode path: the backend owns trail history, while `PersonGroundState` (human CV filter + stationary lock) owns the only track-position filtering stage.
- When `trailAppendAllowed` is false, producer trails do not grow new path samples (person is idle/sit/lie locked).
- World-mode BEV emits a canonical `anchor_hold` head with
  `worldAdmission="held"` so a visible track does not disappear during a brief
  occlusion or physical rejection; held points never append trail samples.
  Canonical `cv_prediction` heads use `worldAdmission="predicted"` and may
  extend the producer trail only when `trailAppendAllowed=true`. Unplaceable
  canonical tracks beyond the bounded display margin are omitted and reported
  through the bounded dropped-point fields rather than clamped to a raster
  edge; points just outside the PCF remain visible with their exact metric
  coordinates and `floorplanInside=false`.
- The post-tiler OSD trail path joins the canonical analytics row on the exact
  source/frame cohort, maps the declared source-image basis into the configured
  mosaic tile, and resets on tracker lifecycle or basis changes. It does not use
  last-seen joins, bbox-bottom fallback, or mosaic-edge clamping.
- BEV trail history includes `trackerLifecycleGeneration`; numeric tracker-ID
  reuse removes the older generation immediately rather than connecting or
  retaining its path.

## Related Docs

- Contracts and RPCs: `docs/api_contracts_ws.md`
- Metadata shapes: `docs/metadata_contracts.md`
- BEV rendering and smoothing: `DS9/docs/bev_capture_event_integration.md`
