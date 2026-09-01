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
release-gated cohort. Bounded sender admission freezes the bytes, private world
commit happens next, and only then may client delivery begin. Reconstructable
journal persistence receives the same exact payload count through its own
finite asynchronous queue; its failure is surfaced as degraded persistence and
cannot stall or partially mutate canonical tracking/world/BEV. Authority commit
failure still aborts the complete cohort with no partial visibility.

The nested world snapshot's `observed_start_us`/`observed_end_us` describe the
evidence-time extent of its retained entities. They are not stream freshness
clocks: `observed_end_us` may regress when the entity with the freshest retained
evidence disappears. Snapshot `sequence` and `published_at_us` are the
strictly advancing ordering/freshness fields for downstream consumers.

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
      "track_key": "<source:tracker:generation>",
      "tracker_lifecycle_generation": <int>,
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
      "world_quality": "good"|"estimated"|"held"|"invalid",
      "world_quality_reason": "<string|null>",
      "world_frame": "backend_world_m"|null,
      "world_frame_revision": "<revision|null>",
      "world_transform_sha256": "<lowercase sha256|null>",
      "world_quantity": "ground_footprint"|null,
      "world_covariance": [<9 row-major float values>]|null,
      "world_support_state": "floor"|"seat"|"couch"|"unknown"|null,
      "world_posture": "standing"|"sitting"|"lying"|"unknown"|null,
      "world_source": "bbox3d"|"pose_depth_fused"|"pose_depth_only"|"pose_floor_only"|"person_anchor_depth_fused"|"person_anchor_depth_only"|"person_anchor_floor_only"|"gravity_drop"|"cv_prediction"|"image_motion_prediction"|"anchor_hold"|null,
      "world_resolver_confidence": <float 0..1|null>,
      "world_resolver_selected_id": "floor_ray"|"registered_depth"|"pose_scale"|"gravity_reconstruction"|"",
      "world_resolver_fused": <bool|null>,
      "world_resolver_disagreement_m": <float|null>,
      "world_filter_prediction": [<float x>, <float y>, <float z>]|null,
      "world_prediction_image_foot": [<float u>, <float v>]|null,
      "world_prediction_provenance": { /* bounded non-authoritative process provenance */ }|null,
      "world_inferred_raw_observation": [<float x>, <float y>, <float z>]|null,
      "world_inferred_process_observation": [<float x>, <float y>, <float z>]|null,
      "world_measurement_accepted": <bool|null>,
      "world_rejection_reason": "<string|null>",
      "world_contact_basis": "<string|null>",
      "world_image_motion_supported": <bool|null>,
      "world_image_motion_streak": <int|null>,
      "world_state_continuity": "restored_short_ghost"|null,
      "world_reacquire_count": <int|null>,
      "world_reacquired": <bool|null>,
      "world_post_occlusion_support_required": <bool|null>,
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
- `world_valid=true` on a compatible track is stronger than producer
  candidacy: it means the strict `CanonicalWorldService` admitted the exact
  source/tracker/lifecycle/frame observation with a world coordinate in this
  release-gated cohort. Before delivery, a rejected candidate is stripped of
  `world`, `world_source`, filter/process provenance, and inferred-coordinate
  fields, is restamped `world_quality="invalid"` with
  `world_quality_reason="canonical_world_service_rejected"`, and cannot append
  a trail. The publication receipt carries the admitted key set to BEV, so a
  public valid track, BEV current head, nested world snapshot, standalone world
  snapshot, and Menon projection cannot disagree about positional admission.
- `embedding_sequence`, `embedding_model_sha256`, and `embedding_dimension`
  form an all-or-none persisted-provenance triad on both associated surfaces.
  They are stamped only after the exact private identity-evidence row is
  durably appended. They are absent when evidence capture is off, append fails,
  or identity is only continuity-held. Partial triads are invalid, and no raw
  embedding vector is public.
- Top-level `world_source="backend_world_fused"` means the backend owns the canonical baseline world estimator; per-track `world_source` records which observation path updated that track on the current frame. `world_frame_revision` plus `world_transform_sha256` identify the exact active calibration/Scene Prior transform and must be preserved by every spatial consumer.
- In the active baseline, one universal resolver independently preserves the
  current floor-ray, registered-depth, optional pose-scale, and weak gravity
  hypotheses with full covariance. It uses no room/camera strategy, genuinely
  fuses only a mutually compatible contributor set, and retains a
  substantially disagreeing candidate as an alternate instead of averaging
  it. The selected current
  measurement feeds the existing `PersonGroundState`, which remains the sole
  human CV filter, physical gate, stationary lock, source hysteresis, and
  reacquisition authority. See
  [`universal_world_localization.md`](universal_world_localization.md).
- `world` is explicitly a `ground_footprint`. `world_covariance` is present
  only for an accepted current measurement and describes the emitted filtered
  point after displacement inflation. Normal tracking and stats rows retain
  only compact `world_resolver_*` decision scalars. The full candidate tree is
  request-gated BEV presentation evidence and cannot move a downstream dot or
  trail.
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
  A standing lifecycle may also hold the exact last public point for at most
  2.0 seconds when the current row has a strong pose-confirmed upright
  silhouette, torso-motion contact, admitted/plausible floor support, admitted
  observation range, sufficient detector/tracker confidence, and a floor
  candidate within 0.75 m of that output. Because this is a gain-zero presence
  hold, bbox stationarity and a resolved walk/idle label are not required. The
  candidate itself remains rejected and the trail does not grow.
  The producer's typed stationary proof is stamped from the public tracking
  `posture`. The strict service consults `world_posture` only when `posture` is
  absent, so a resolver-diagnostic posture cannot override, suppress, or
  manufacture the longer sitting/lying hold.
  `image_motion_prediction` is the stricter projective continuation for a
  materially moving detector box when the current metric/depth anchor is
  missing or rejected. It transports the last physically accepted image foot
  through the current bbox affine change and projects that pixel through the
  active corrected floor plane. It may advance the physically gated bounded
  process posterior, but it never updates last-good metric state or its fixed
  transport origin; `world_prediction_image_foot` and
  `world_prediction_provenance` expose its non-authoritative basis. If image,
  ray, metric-speed, or TTL gates fail, the producer fails closed instead of
  publishing a frozen point.
- An established, confidently standing lifecycle with explicit lower-body
  occlusion and a learned upright height may use the current gravity
  reconstruction as non-authoritative process evidence. The first eligible
  raw reconstruction is paired with the last exact queue-admitted **metric**
  world output; later rows apply only a recent three-sample XZ medoid's raw
  delta to those two immutable origins. Neither an inferred/held row nor a
  hidden callback may rebase them. Each selected medoid sample and consensus
  span remains within the 0.40-second window. Lifecycle, calibration revision,
  trail segment, range, complete monotonic media time, posterior speed, and
  visible-output gates all remain mandatory. A raw gap
  beyond 0.40 seconds clears the short medoid window but does not rebase either
  immutable origin. The same episode may restart only while the gap and latest
  service output are no older than 1.25 seconds, the transition names an exact
  recent service commit carrying the same service-owned inferred root, and all
  lifecycle/segment/frame/transform/time bindings still match. Otherwise it
  fails closed for that row without renewing the root; any later candidate must
  independently satisfy the ordinary recent-evidence or exact-root restart
  gates.
  Producer root lifetime follows the ordered queue: only a committed metric or
  unrelated projective successor may end it. A metric accepted solely on a
  rate-suppressed callback cannot replace the root still owned by the strict
  service; inferred rows and their bounded descendants retain that root until a
  queue-visible successor, lifecycle/revision change, or expiry.
  `world_inferred_raw_observation`,
  `world_inferred_process_observation`, and
  `world_prediction_provenance.transport="fixed_occlusion_origin_raw_world_delta"`
  expose the exact algebra. This lane never updates accepted metric geometry,
  body height, last-good metric state, or cross-camera measurement authority.
- A proven `image_motion_prediction`, `cv_prediction`, or `anchor_hold` is
  carried into the canonical snapshot as a high-uncertainty held observation
  so tracking, BEV, and Menon receive the same Noesis-owned continuation.
  Image transport, including learned-height occlusion transport, requires
  complete fixed-origin provenance. For learned-height transport, the strict
  service additionally proves `process_observation = trusted_world_origin +
  (raw_consensus - raw_origin)`. The medoid-selected consensus retains its own
  evidence timestamp when it is older than the current cohort. Independently,
  `filter_transition` version 1 binds an exact queue-admitted source output,
  media-time gate, physical filter base, gain, process observation, and
  posterior. The strict service recomputes the innovation update and requires
  that origin to equal either its latest point or one point in its bounded
  same-segment history of exact service commits for that source/lifecycle.
  This accommodates finite producer/consumer queue lag without trusting a
  producer-invented origin; the proposed posterior must also satisfy the human
  speed bound relative to the service's latest committed point. Exact output
  origins are retained for at most the 1.25-second physical filter horizon;
  this validation retention does not extend the 0.40-second CV bridge or make
  a held output fresh metric evidence.
  The producer queue and service each derive an immutable projective root only
  from an admitted `image_motion_prediction`; CV/hold descendants may carry it
  but cannot renew it with their own PTS or recreate it from a source label.
  Service history is keyed by source, camera, tracker lifecycle, world-frame
  revision, world-transform SHA-256, and active calibration-artifact SHA-256,
  so a calibration artifact change invalidates roots and retained origins.
  `world_filter_prediction` must exactly equal the recomputed emitted tracking
  point; the filter may legitimately move less than its process input. CV and
  hold rows require complete `bounded_cv_process` provenance, or a typed
  `bounded_output_hold` from the last published coordinate when final output
  admission rejects the proposed process step. A bounded CV row may bind a
  slightly older exact service commit only while its retained metric anchor is
  still identical to the latest commit and the result is physically bounded
  from that latest point. A recent-projective bridge additionally requires the
  matched commit itself to be an image-motion output inside 0.405 seconds, so
  CV cannot chain. `bounded_output_hold` remains latest-output-only. Their
  `filter_transition` binds the output and retained metric origin by exact
  media PTS, trail segment, tracker lifecycle, and world-registration identity.
  `process_observation` is the actual candidate presented to final output
  admission; transition gain is exactly `1` for an admitted bounded CV step
  and `0` for an output hold. The strict service recomputes that posterior,
  enforces the canonical speed/time bound, and requires exact finite equality
  between `world_filter_prediction` and the emitted `world` point. Rejected
  pre-seeded/bbox3d observations use this same proof rather than a parallel
  display-only hold path.
  A gain-zero `bounded_output_hold` advances the latest visible PTS while
  preserving a separate kinematic coordinate/PTS/segment. Only recovery after
  that exact condition may reduce filter gain along the original transition
  line to remain within 4 m/s of the latest displayed point while the older
  kinematic clock supplies the real elapsed motion budget. The strict service
  verifies both bounds; ordinary metric rows retain strict rejection or
  evidence-backed reanchor behavior and are never generally slewed or clipped.
  All three require `state_integrated=true`, remain non-authoritative, and
  never become fresh metric or cross-camera fusion evidence. When a
  contemporaneous fresh camera observation exists, global fusion uses it and
  retains the held row as rejected source evidence; when only held rows exist,
  it selects the newest one exactly instead of averaging process continuations.
  Global fusion's default velocity cap is the same 4 m/s source/service
  contract, including across identity continuity.
  Snapshot entity lifecycle remains presence-time based. The service also
  refuses incomplete or mixed target-frame revisions and preserves full
  covariance through conservative covariance intersection for fresh evidence.
- World/filter/lock state is discarded on the first exact processed frame that
  omits a tracker key. A later reuse of the same numeric tracker ID starts a new
  world lifecycle and cannot inherit the prior position or velocity.
- Cold weak metric bootstrap normally requires three mutually consistent
  same-family rows. Two consecutive exact current `pose:ankle_pair` contacts
  may seed because both observed feet support the floor contact; single-ankle
  or mixed runs still require three samples unless the independent strong
  torso/silhouette motion proof is current. One intervening weaker
  bbox/non-floor row may preserve the pending ankle-family run but cannot be
  published or become its coordinate; a second intervening row or expired gap
  clears the run.
- `depth_used_m` is the current registered DAv2 anchor range admitted as a
  resolver candidate; `world_source` and compact `world_resolver_*` fields
  describe the normal decision, while request-gated BEV diagnostics expose
  exact contributors. `depth_anchor_m` remains the raw anchor carried by
  `NOESIS.OBJECT_DEPTH`.
- `depth_registered_m` is the room-registered DAv2 anchor depth after applying the offline DAv2->MapAnything mapping for that camera; this is the value the estimator projects when registration is active.
- `depth_registration_status` and `depth_registration_id` make the registration path observable on both tracks and active-tracks without changing the raw `NOESIS.OBJECT_DEPTH` payload semantics.
- `depth_anchor_sample_count` and `depth_anchor_valid_fraction` describe the
  exact depth-band support. Candidate covariance, score, compatibility, and
  rejection fields explain selection more completely than whole-mask support.
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
  "sourceId": <int>,
  "sourceEpoch": <int>,
  "frameId": <int>,
  "observedAtUs": <int epoch microseconds>,
  "trackingPublicationSequence": <int>,
  "trackingOutboundSubmissionId": <int>,
  "cohort": {
    "source_id": <int>,
    "source_epoch": <int>,
    "frame_id": <int>,
    "observed_at_us": <int epoch microseconds>,
    "tracking_publication_sequence": <int>,
    "tracking_outbound_submission_id": <int>
  },
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
  "droppedFootpoints": [ {"stableId": <int|null>, "trackerId": <int|null>, "trackerLifecycleGeneration": <int|null>, "trailSegmentId": <int|null>, "reason": "<admission reason>", "canonicalWorld": <bool>, "trailRetained": <bool|null>, "anchorSource": "<string|null>"} ],
  "trails": [ {"stableId": <int|null>, "trackerId": <int|null>, "trackerLifecycleGeneration": <int|null>, "canonicalWorld": <bool>, "trailSegmentId": <int|null>, "points": [ {"x": <float>, "y": <float>, "t": <int ms>, "floorplanInside": <bool>, "coverageInside": <bool>, "coverageRegion": "<semantic-region>"|null} | {"t": <int ms>, "breakBefore": true} ]} ],
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
- The top-level cohort fields exactly mirror `cohort`, including
  `sourceEpoch == cohort.source_epoch`. For one `sourceId`, a higher epoch is a
  new media timeline and remains admissible when media, observation, frame,
  sequence, and outbound-submission clocks restart. Dashboard admission clears
  all source-local heads, trails, smoothing, sampling phase, and source-time
  state before consuming it. A lower epoch is stale and is rejected.
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
- A trail point with `breakBefore=true` has no coordinates and terminates the
  preceding polyline. A later finite point starts a disconnected segment. The
  marker may be terminal while a recovered live head is not yet eligible for
  trail sampling; consumers must then also suppress any old-tail-to-head
  connector. Bounded earlier history remains visible through a missing-world
  interval or reanchor without a false straight-line bridge. Completed and
  current segments share the configured time and total per-track point bounds.
- `trails[].trailSegmentId` identifies the current segment only. Retained
  completed segments in the combined point list are visual history separated
  by `breakBefore` and do not inherit that top-level identity.
- World-mode BEV emits a canonical `anchor_hold` head with
  `worldAdmission="held"` so a visible track does not disappear during a brief
  occlusion or physical rejection; held points never append trail samples.
  Canonical `cv_prediction` heads use `worldAdmission="predicted"` and may
  extend the producer trail only when `trailAppendAllowed=true`. Unplaceable
  canonical tracks beyond the bounded display margin are omitted and reported
  through the bounded dropped-point fields rather than clamped to a raster
  edge; points just outside the PCF remain visible with their exact metric
  coordinates and `floorplanInside=false`.
- `droppedFootpointCount` is the complete number of omitted footpoints across
  active admission paths in the exact cohort. `droppedFootpoints` is only the
  first 64 reason records for diagnosis; consumers remove any live head absent
  from the exact current `footpoints` cohort and must not treat that bounded
  list as an exhaustive lifecycle signal. Alignment-debug mode may additionally
  count and sample legacy/noncanonical presentation rejects marked
  `canonicalWorld=false`; those records are never live-head authority.
- Dashboard current-head state is independent of trail state. Every finite
  canonical `footpoints` member updates its lifecycle-keyed head even when
  trails are disabled or unsampled; complete-cohort absence removes only that
  head. Producer `trails` may remain visible across the gap, separated by
  `breakBefore`, without recreating a live dot from the retained tail.
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
