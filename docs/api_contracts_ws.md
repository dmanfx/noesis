# DS9.1 WebSocket API contracts
_Status: canonical native-host observation/world/depth/PCF contract, updated 2026-08-25._

The WebSocket server (`noesis.server.websocket.WebSocketServer`) is the primary transport for DS9.1 telemetry, depth retrieval, and WebRTC signaling. All active DS9.1 telemetry message types are JSON unless a future binary payload explicitly documents otherwise.

## Authentication boundary

The runtime WebSocket is an internal native-host transport. Its HTTP upgrade must
carry `Authorization: Bearer <internal gateway token>`; the token is loaded from
`NOESIS_INTERNAL_AUTH_TOKEN_FILE` (owner-only) and is never accepted in a URL or
query string. Browsers obtain a short-lived, one-use ticket from Menon and
connect through the same-origin gateway; they never receive the internal token.

`NOESIS_INTERNAL_AUTH_MODE=disabled` is an explicit loopback-development mode,
not a production fallback. Missing or invalid authentication returns HTTP 401
before a client enters the WebSocket connection set.

The exact authenticated path `/healthz` is reserved for runtime lifecycle
checks. After a successful upgrade it sends one bounded frame and closes:

```json
{"type":"health","contract":"noesis.ws.health","contract_version":1}
```

That v1 frame is the unbound development liveness contract. The canonical
native DS9.1 runtime sends strict `noesis.ws.health` v2 from its explicit health
identity:

```json
{
  "type": "health",
  "contract": "noesis.ws.health",
  "contract_version": 2,
  "deployment_id": "<deployment-id>",
  "selector_sha256": "<64 lowercase hex characters>",
  "state_release_id": "<state-release-id>",
  "runtime_family": "ds9",
  "runtime_variant": "<family-prefixed exact variant>",
  "instance_id": "<producer instance>",
  "run_id": "<producer run>",
  "boot_id": "<kernel boot identity>",
  "software_revision": "<exact Git revision>",
  "generated_at_us": 1,
  "ready": true
}
```

The v2 frame is emitted only when the exact bound producer matches capability
health and both `tracking_observations` and `global_world` are healthy. Before
that point the server closes with code `1013` and reason `health_not_ready`;
it never sends a partial v2 payload or falls back to v1.

Health connections never enter the telemetry client set, acquire a WebRTC
gateway, invoke UI snapshot callbacks, or receive configuration, calibration,
trail, stats, tracking, world, or media payloads. DS9.1 baseline and the disabled V3DT adapter share this
server boundary. They are therefore excluded from telemetry-client capacity.

Authenticated telemetry clients are bounded for the single-home LAN product.
`NOESIS_WS_MAX_TELEMETRY_CLIENTS` defaults to 8 and is hard-clamped to 1–16.
An excess client is closed with code `1013` and reason
`telemetry_capacity_reached` before registration or any initial snapshots. A
disconnected slot is immediately reusable; health checks remain available at
capacity.

## 1. Common Envelope

Most JSON messages have:

```json
{ "type": "<message-type>", ... }
```

BEV JPEG binary payloads are retired. The server still has a generic binary
coalescer for future payload types, but current BEV delivery is metadata-only.

### Sender admission and ordering

Every cross-thread JSON producer is admitted through a finite owned queue. The
server validates response-assembly timing, serializes exactly once with
non-finite numbers forbidden, applies per-message and per-batch byte/count
limits, and retains the immutable encoded bytes before returning a typed sender
receipt. A receipt proves only that the exact bytes entered the owned event-loop
queue; it is not a WebSocket-client delivery acknowledgement. Closed/stopped
lifecycle state, a missing/closed loop, queue saturation, invalid JSON, or a
size violation fail before a receipt exists. Caller mutation after receipt
cannot change the admitted value. In addition to the 256-submission cap, exact
frozen payload bytes share a 256 MiB global in-flight budget; completion,
failure, explicit gate abort, and cancellation release the reservation exactly
once. Admission,
quiescence, and boundary metrics expose payload/current/peak/limit byte counts,
and a valid shutdown receipt requires zero pending bytes.

Canonical tracking/world/event admission is additionally release-gated. The
admission receipt reserves the exact immutable bytes and event-loop ownership,
but the scheduled coroutine waits on a one-shot authority decision. The runtime
commits the exact prepared private fusion state and only then releases the
batch. Commit failure resolves the gate as an explicit abort: no batch member
begins client delivery, aborted submissions/bytes are counted separately, and
the poisoned publisher emits no successor. A gate left unresolved remains an
in-flight byte/future lease and prevents successful shutdown quiescence.
Reconstructable world-journal persistence is a separate bounded asynchronous
worker: its admission confirms only the exact queued payload count. Queue or
durable-write failure degrades persistence health and counters but cannot wait
on, abort, mutate, or poison canonical tracking/world/BEV. The authority-gate
dwell is excluded from the 3 ms WebSocket serialization boundary and remains
observable via the publication transaction tests.

Canonical type routing is explicit and fail-closed:

- `tracking` followed by its optional exact `world_snapshot` and ordered
  `world_event` cohort may use only `admit_broadcast_batch_sync`. The boundary
  validates tracking-first order, exact cohort fields, embedded/separate
  snapshot equality, event count/order/equality, and rejects BEV or
  noncanonical mixing before admission.
- `bev-frame` and `bev-status` may use only `broadcast_bev_sync`. They share
  one typed, bounded sender route. A cohort-bound status carries the same
  top-level tracking identity as the failed BEV attempt, including
  `trackingOutboundSubmissionId`; the dashboard rejects an older or unbound
  status once a newer canonical frame is admitted. A startup/transport status
  without a cohort is only a reset before canonical frame admission.
- Generic sync, batch, targeted, async, and coalescer entry points reject all
  five canonical types, including top-level canonical JSON supplied as raw text
  or UTF-8 binary. Noncanonical raw traffic remains supported. Pre-encoded
  envelopes carry a server-instance owner token and must have exact ASCII byte
  length plus matching encoded/declared type, so caller-constructed frozen JSON
  is not trusted.

The release callback is an in-process authority trust boundary, not a
cryptographic capability. Production call-site tests therefore restrict the
gated API to the byte-identical DS9.1 tracking publishers plus its WebSocket
definition; those publishers alone bind it to `CanonicalWorldService.commit`.
An arbitrary same-process caller could always violate Python object privacy, so
exclusive call sites, exact shape validation, and fail-closed runtime review are
the maintainability enforcement.

Canonical `tracking`, `world_snapshot`, `world_event`, `bev-frame`, and
`bev-status` messages are never latest-only coalesced. One tracking frame plus its exact world
snapshot/events enters the queue as one ordered batch and is delivered in list
order by one coroutine. Batch admission is all-or-none, but transport delivery
is not a multi-frame socket transaction: a later client disconnect can still
interrupt delivery. Generic coalescing, when explicitly configured for a
non-canonical message type, occurs only after sender admission.

## 2. Stats (`type: stats`)

Emitted ~1 Hz when `stats_callback` is registered
(`DS9/noesis/ds9_runtime_core.py`).

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
          "active_cameras": [<camera-id>, ...],
          "inactive_cameras": [<camera-id>, ...],
          "failed_cameras": [<camera-id>, ...],
          "floorplan_authority_ready_camera_count": <int>,
          "floorplan_authority_ready_cameras": [<camera-id>, ...],
          "floorplan_authority_pending_camera_count": <int>,
          "floorplan_authority_pending_cameras": [<camera-id>, ...],
          "cameras": { /* active_ready, inactive_ready, or failed by configured camera */ }
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
        "cameras": { /* bounded per-camera request state */ }
      },
      "scene_prior": {
        "contract": "noesis.scene_prior.health",
        "contract_version": 1,
        "site_id": "<site id>",
        "mode": "shadow",
        "camera_count": <int>,
        "revision_count": <int>,
        "loaded_revision_count": <int>,
        "cameras": { /* exact space/prior binding by camera */ }
      },
      "identity_v2_shadow": {
        "status": "enabled"|"disabled"|"degraded",
        "last_error": <string|null>
      },
      "boundary_cpu_serialization_p99_ms": <float|null>,
      "boundary_cpu_serialization_p99_10s_ms": <float|null>,
      "boundary_cpu_serialization_p99_60s_ms": <float|null>,
      "boundary_cpu_serialization_ws_p99_ms": <float|null>,
      "boundary_cpu_serialization_rest_p99_ms": <float|null>,
      "boundary_serialization_errors_total": <int>,
      "boundary_serialization_ws_errors_total": <int>,
      "boundary_serialization_rest_errors_total": <int>,
      "zero_copy_core": {
        "counters": {"<counter-name>": <int>, ...},
        "stage_timings": {"<stage-name>": { /* bounded timing summary */ }, ...},
        "boundary_serialization_metrics": {"ws": { /* ... */ }, "rest": { /* ... */ }}
      },
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

The boundary p99 is the conservative maximum of every budgeted route and the
true rolling 10- and 60-second windows, not a lifetime pooled percentile. A
WebSocket JSON total includes response construction, cross-thread publisher
handoff, serializer-worker handoff when used, NumPy-leaf conversion, the
single admission-freeze JSON encoding, and local send-task dispatch. Sync
producers carry those pre-encoded immutable bytes through delivery rather than
encoding twice. It excludes blocking provider/domain work, explicitly
configured non-canonical coalescing dwell, and network flow-control time.
Every producer-created JSON payload must carry response-construction timing;
missing timing fails closed and increments the tagged serialization error
counter. Live zero-copy gates require a p99 and an error counter in every stats
sample, and reject either error-counter growth or a nonzero baseline.

The three geometry/capture health blocks are runtime-owned and fixed-schema:

- `bev.health` v2 separates renderer readiness from successful exact-frame
  activity. Every successfully published BEV frame increments `success_count`
  and makes that camera `active_ready`, including an exact frame with
  `footpoints: []` and no trails. `inactive_ready` means that the configured
  camera has not completed its first exact BEV publication; it is not an
  occupancy label. `healthy` requires a valid configured renderer and no
  failed or unexpected camera.
- A camera-local renderer awaiting its first valid active-floorplan record is
  reported with `floorplan_authority_state=startup_pending` and in the pending
  camera fields above. That bounded bootstrap state is not a rendered success
  and cannot satisfy active-floorplan N/N acceptance. After authority has been
  established, loss or invalidation changes the authority state to `lost`,
  fails renderer health, and enters fatal runtime handling.
- `active_floorplan.healthy` requires one accepted, calibration-current record
  for every configured camera. Its camera rows expose the exact portable
  snapshot reference, write ID, content digest, calibration fingerprint, and
  capture/floorplan timestamps used by BEV.
- `capture_event_fusion.healthy` describes an idle, unpoisoned controller. It is
  false while a request owns a per-camera/global admission lease and remains
  false after a fatal barrier error until a later exact capture succeeds.

`depth_enabled` and `depth_fps` still describe the on-demand MapAnything branch.
The baseline DAv2 depth-tracking lane used by world estimation is separate and
always on when the baseline runtime starts successfully.

Native baseline startup also requires a prebuilt room-registration artifact
(`depth_registration.path` in `DS9/config/infer.yaml`, default
`DS9/config/depth_registration.json`). DS9.1 loads that artifact before opening
the sources and fails fast if an enabled camera lacks a valid
DAv2-to-MapAnything registration entry.

The baseline DAv2 capture and later object-fusion operators rendezvous by exact
`(source_id, frame_id, media PTS)`. Live fusion is nonblocking by default
(`NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS=0`, clamped to 0–250 ms): it consumes
the exact sibling when already available, otherwise an admissible prior frame
within the configured depth cadence. An explicit positive wait remains
available for bounded offline diagnostics; future and wrong-source frames are
never accepted. The result is observable under
`pipeline.zero_copy_core.counters`:

- `depth_bridge_put_total`, `depth_bridge_exact_resolve_total`,
  `depth_bridge_wait_total`, and `depth_bridge_wait_timeout_total` describe the
  exact-frame rendezvous.
- `depth_bridge_lagged_resolve_total`, `depth_bridge_lagged_age_frames_total`,
  `depth_bridge_lagged_age_us_total`, and `depth_bridge_miss_total` describe the
  bounded post-timeout result.
- `object_depth_attach_total` and `object_depth_status_total.<status>` count
  payloads actually attached to object metadata. `object_depth_attach_failure_total`
  and `object_depth_attach_failure_total.<reason>` count rejected/unavailable
  native attachment paths. A failed attachment is not reported as an attached
  status and cannot become canonical depth evidence.
- `tracking.publication_worker.pending_total`,
  `pending_high_watermark`, `inflight`, `enqueued_total`, `completed_total`,
  `overflow_total`, and `failures_total` expose the exact canonical worker.
  `identity_v2.shadow.pending_sources`, `pending_sources_high_watermark`,
  `inflight`, `enqueued_total`, `completed_total`, `coalesced_total`, the
  shadow-only drop totals, and `worker_failures_total` expose the optional
  shadow lane. Their stage timings are
  `tracking.publication_worker_item`,
  `tracking.publication_worker_queue_wait`,
  `identity_v2.shadow_process_source_frame`, and
  `identity_v2.shadow_queue_wait`; BEV retains
  `bev.render_and_publish`.

## 3. Mosaic Video (WebRTC)

Mosaic video is delivered through one private H.264 SHM→WebRTC path; WebSocket
is **signaling only**.

- `webrtc_offer` (client → server): `{ "type": "webrtc_offer", "sdp": "<offer sdp>" }`
- `webrtc_answer` (server → owner): `{ "type": "webrtc_answer", "sdp": "<answer sdp>" }`
- `webrtc_ice_candidate` (bi-directional): `{ "type": "webrtc_ice_candidate", "candidate": "<candidate>", "sdpMLineIndex": 0 }`
- `webrtc_error` (server → client): `{ "type": "webrtc_error", "error": "<string>" }`

Ownership: each accepted `webrtc_offer` leases one gateway slot to that client.
The same client may reuse its lease; a different client cannot take it over.
When every slot is leased, the offer receives
`webrtc_error (webrtc_capacity_reached)`. ICE from a client without that exact
lease receives `webrtc_error (webrtc_not_owner)`.

Owner disconnect or session invalidation revokes media immediately: the server
clears owner routing, and `reset_peer` rebuilds ICE, DTLS, and RTP state before a
warm gateway slot may be reused. Reset failure retires that slot. Menon closes
its `RTCPeerConnection` and stops all local media tracks whenever the Noesis
socket disconnects or its authenticated browser session becomes invalid.
The native runtime requires advancing H.264 access units from the private SHM
edge before a warm gateway is useful. It starts a bounded warm set (one by
default) and creates remaining configured slots only on demand. RTSP is disabled
and is not a media or readiness dependency.

Negotiation privacy: offer/answer SDP, ICE candidates, DTLS fingerprints, and
TURN credentials are transient signaling data and are not persisted or written
to runtime logs. Diagnostics expose only bounded connection-state, direction,
payload-type, and packet/frame-count summaries. External STUN is disabled by
default for the LAN deployment; `NOESIS_MOSAIC_WEBRTC_STUN_SERVER` is an explicit
deployment opt-in.

## 4. BEV Frames

Emitted by `BevRenderer`:

```json
{
  "type": "bev-frame",
  "cameraId": "<camera>",
  "ts": <int microseconds>,
  "sourceId": <int>,
  "frameId": <int>,
  "observedAtUs": <int epoch microseconds>,
  "trackingPublicationSequence": <int>,
  "trackingOutboundSubmissionId": <int>,
  "cohort": {
    "source_id": <int>,
    "frame_id": <int>,
    "observed_at_us": <int epoch microseconds>,
    "tracking_publication_sequence": <int>,
    "tracking_outbound_submission_id": <int>
  },
  "w": <int>,
  "h": <int>,
  "mpp": <float>,
  "xMin": <float>, "xMax": <float>,
  "zMin": <float>, "zMax": <float>,
  "boundsSource": "config"|"auto_extents"|"active_floorplan"|"active_floorplan_plus_coverage_envelope",
  "displayBounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>},
  "floorplanCoordinateSpace": "floorplan_normalized_v1",
  "floorplanBounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>},
  "coverageEnvelope": {
    "contract": "noesis.bev.coverage_envelopes",
    "contractVersion": 1,
    "frame": "camera_local_ground_m",
    "units": "meters",
    "cameraId": "<camera>",
    "boundaryToleranceM": <float>,
    "bounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>},
    "regions": [
      {"id": "<semantic-region>", "polygonXZ": [[<float x>, <float z>], "..."]}
    ]
  },
  "floorplanGridShape": [<rows>, <cols>]|null,
  "floorplanGridResM": <float|null>,
  "floorplanSnapshotTsUs": <int|null>,
  "floorplanTsUs": <int|null>,
  "floorplanSnapshotId": "<immutable snapshot id>",
  "floorplanSnapshotContentSha256": "<sha256>",
  "floorplanCalibrationFingerprint": "<sha256>",
  "overlay": <bool>,
  "footpoints": [
    {
      "x": <float>,
      "y": <float>,
      "method": "<string>",
      "stableId": <int|null>,
      "trackerId": <int|null>,
      "trackerLifecycleGeneration": <int|null>,
      "trackKey": "<source:tracker:generation>"|null,
      "frameId": <int>,
      "anchorSource": "<string|null>",
      "anchorQuality": "<string|null>",
      "anchorReason": "<string|null>",
      "displaySource": "world"|"world_to_camera_local"|"world_floor_fallback_to_camera_local"|"image_anchor"|"image_depth_anchor"|"floor_contact_ray"|"registered_depth_anchor",
      "worldAdmission": "accepted"|"predicted"|"held"|null,
      "worldFrame": "backend_world_m"|null,
      "worldFrameRevision": "<revision>"|null,
      "worldTransformSha256": "<sha256>"|null,
      "resolverDiagnostics": {
        "contract": "noesis.world_resolver_diagnostics",
        "version": 1,
        "frameId": <int>,
        "sourceId": <raw source int>,
        "sensorId": <mapped BEV sensor int>,
        "trackKey": "<source:tracker:generation>",
        "trackerLifecycleGeneration": <int>,
        "worldFrame": "backend_world_m",
        "worldFrameRevision": "<revision>",
        "worldTransformSha256": "<sha256>",
        "calibrationRevision": "<revision>",
        "pcfRevision": "<PCF geometry/frame revision|null>",
        "floorplanSnapshotId": "<Scene Prior artifact id|null>",
        "floorplanWorldFrame": "backend_world_m"|null,
        "floorplanWorldFrameRevision": "<revision>"|null,
        "selectedId": "<candidate id|null>",
        "selectedKind": "floor_ray"|"registered_depth"|"pose_scale"|"gravity_reconstruction"|null,
        "decision": "<bounded decision|null>",
        "reason": "<bounded reason|null>",
        "resolved": {"world": {"x": <float>, "z": <float>}, "display": {"x": <float>, "z": <float>}, "covarianceXZ": [[<float>, <float>], [<float>, <float>]]}|null,
        "candidates": [ /* at most four exact-cohort candidate records */ ],
        "disagreement": {"distanceM": <float>, "reason": "<string>"}|null
      }|null,
      "motionMode": "walk"|"idle"|"sit"|"lie"|"unknown"|null,
      "posture": "standing"|"sitting"|"lying"|"unknown"|null,
      "trailAppendAllowed": <bool|null>,
      "idleJitterM": <float|null>,
      "floorplanInside": <bool>,
      "coverageInside": <bool>,
      "coverageRegion": "<semantic-region>"|null
    }
  ],
  "droppedFootpointCount": <int>,
  "droppedFootpoints": [
    {
      "stableId": <int|null>,
      "trackerId": <int|null>,
      "trackerLifecycleGeneration": <int|null>,
      "trailSegmentId": <int|null>,
      "reason": "canonical_world_missing"|"canonical_world_nonfinite"|"canonical_world_outside_admission_surface"|"canonical_world_outside_max_distance"|"canonical_world_outside_display_guard"|"<other canonical admission reason>",
      "canonicalWorld": true,
      "anchorSource": "<string|null>"
    }
  ],
  "trails": [ {"stableId": <int|null>, "trackerId": <int|null>, "trackerLifecycleGeneration": <int|null>, "canonicalWorld": <bool>, "trailSegmentId": <int|null>, "points": [ {"x": <float>, "y": <float>, "t": <int ms>, "floorplanInside": <bool>, "coverageInside": <bool>, "coverageRegion": "<semantic-region>"|null} ]} ],
  "H": [<9 floats>],
  "sampleXZ": [<float x>, <float z>] | null,
  "frame": "backend_world_m"|"camera_local_ground_m",
  "world_frame": "backend_world_m"|"camera_local_ground_m",
  "canonicalWorldFrame": "backend_world_m",
  "canonicalWorldFrameRevision": "<revision>"|null,
  "canonicalWorldTransformSha256": "<sha256>"|null,
  "frame_mode": "world"|"camera_local",
  "units": "meters",
  "s_obj_to_m": <float>,
  "trail_smoothing_owner": "frontend"|"backend"|"none",
  "bev_points_smoothed": <bool>,
  "bev_world_points_smoothed": <bool>
}
```

- `footpoints[].method` is an image-anchor/render provenance string emitted by the backend (`image_base`, `image_foot`, `bbox`, etc.), not the canonical track world estimator source.
- `sourceId`, `frameId`, `observedAtUs`, and
  `trackingPublicationSequence` identify the exact tracking
  publication cohort that produced the BEV frame. Every current footpoint
  repeats that `frameId`. Validators join by camera/source/frame and require
  the tracking and BEV observation times to agree; last-seen identity joins are
  not an acceptable substitute.
- The top-level cohort fields, nested `cohort`, and
  `trackingOutboundSubmissionId` are all required. Dashboard ingestion rejects
  partial, contradictory, duplicate, or older cohorts, including frames that
  arrive after a status/error message.
- When the BEV renderer is active, tracking and BEV share one publication gate.
  Its effective interval is
  `max(selected tracking interval, configured BEV interval)`, where the
  selected tracking interval is the occupied cadence or the empty-frame
  heartbeat cadence for that frame. Tracking publishes first; its typed sender
  receipt commits the gate/lifecycle state, and a `bev-frame` is attempted only
  for that exact receipt. `trackingOutboundSubmissionId` binds the BEV attempt
  to the prior tracking/world batch; an admitted BEV receipt must have a larger
  outbound submission ID. Count
  changes and tracker-lifecycle/key-set changes bypass the interval and force
  the same-frame pair. If tracking publication fails, no BEV is emitted for
  that frame. A renderer that is not configured does not change tracking
  cadence.
- `footpoints[].anchorSource`, `anchorQuality`, and `anchorReason` mirror the backend world estimator diagnostics from tracking telemetry so BEV/Three.js consumers can explain why a point was accepted, guarded, or held.
- `footpoints[].resolverDiagnostics` is presentation-only output from the
  universal resolver. The renderer admits it only when camera, raw source,
  mapped sensor, tracker, tracker-lifecycle generation, exact track key, frame,
  observation time, active world revision, source-to-world transform SHA-256,
  camera-calibration revision, PCF geometry revision, and Scene Prior artifact
  identity match the rendered point. The PCF geometry revision and Scene Prior
  snapshot ID are separate namespaces and are never compared to each other. Candidate
  positions and 2x2 X/Z covariance are transformed with the same world-to-view
  Jacobian as the canonical point. The normal dashboard exposes this through
  the off-by-default `Localization details` toggle. Candidates, uncertainty
  ellipses, disagreement, and the optional `legacy` point cannot change the
  canonical dot or trail. `legacy` is the retired room policy's current-frame
  choice reconstructed from the same bounded candidates; it is not a second
  filtered track and is never localization authority.
  The toggle is a connection-scoped capability request: the server publishes
  rich resolver details only while at least one connected dashboard requests
  them. The server broadcasts the effective state to every dashboard, and
  automatically disables the capability when the last requesting connection
  disconnects. A normal dashboard therefore adds no rich resolver tree or
  serialization cost unless it explicitly opts in.
- `footpoints[].displaySource` declares which coordinate path produced the displayed BEV point. The primary inline floorplan view uses `frame_mode=camera_local` and `frame=camera_local_ground_m`, so displayed points and producer trails are in the same camera-local ground frame as MapAnything floorplan rasters. Every live tracker footpoint is marked canonical-world-required by the DS9 producer: its only admissible production display source is `world_to_camera_local`, projected from the producer-owned filtered `track.world`. `world_floor_fallback_to_camera_local` is a legacy/noncanonical diagnostic path emitted only for compatibility when a noncanonical world fallback is projected into the camera-local view; it is not permitted to move an active canonical tracked-person dot. Registered depth, floor-contact rays, and image anchors may appear in alignment diagnostics, but they cannot move a live canonical tracked-person dot. A live track without a finite canonical world point inside the producer-published `displayBounds` is omitted. The dashboard uses that exact metric display envelope for camera-local cohorts; it does not invent a second margin, auto-fit to tracks, clamp points, or remap raw metric coordinates. The active PCF raster remains the semantic `floorplanBounds`, not a live tracking validity gate: a finite canonical point in the display envelope may be emitted unchanged with `floorplanInside=false`. When `coverageInside=false` is carried on a canonical world point, it remains diagnostic and does not become a second dashboard-only rejection gate; legacy/noncanonical candidates still honor the configured coverage polygon. `canonical_world_outside_display_guard` is a bounded renderer-side presentation safety drop, not a replacement for the producer's world-validity decision. `registered_depth_anchor`, `floor_contact_ray`, `image_depth_anchor`, and `image_anchor` remain legacy/non-tracker or diagnostic source values only.
- `camera_local_ground_m` uses the horizontal projection of the active camera's
  right and forward axes. Camera pitch and camera height must not contribute to
  local X/Z. Consequently, the floor point vertically below the camera is
  exactly `(0, 0)`. Consumers must not substitute the pitched OpenCV camera
  frame or reconstruct this transform from raw, differently revisioned `E`.
- `worldAdmission=predicted` means the exact current cohort did not provide an
  accepted current metric observation because its candidates were missing,
  stale, or physically rejected, so the producer displayed a bounded
  constant-velocity continuation of the canonical world filter. It remains in
  the same revision-bound world frame, but is not a fresh position measurement:
  `world_measurement_accepted=false`, `world_source="cv_prediction"`, and
  `world_filter_prediction` carries the displayed prediction. Prediction does
  not advance the last-good measurement timestamp. Rejected updates predict
  from one fixed last-good rejection anchor and are capped at 0.40 seconds;
  repeated rejects cannot advance that anchor or accumulate unbounded drift.
  Trail growth remains controlled by `trailAppendAllowed`.
- `footpoints[].worldAdmission=held` means the exact current cohort had no
  accepted new position and the producer intentionally retained its last-good
  canonical world point. The head remains visible, but no trail sample is
  appended. The normal hold cap is 0.40 seconds. A trusted seated or lying
  lifecycle may hold for at most 2.0 seconds only while exact-frame detector
  boxes prove stationary continuity; that exception remains
  `world_measurement_accepted=false` and trail-disabled. A genuinely
  unplaceable canonical track is omitted and accounted for by
  `droppedFootpointCount` plus a bounded `droppedFootpoints` reason list; normal
  operation never hides that omission behind alignment-debug mode.
- `floorplanBounds` and the other `floorplan*` fields bind the MapAnything raster
  to the active floorplan registry's metric bounds, grid, resolution, exact
  snapshot identity, and capture timestamps. The registry accepts only
  successful `camera_local_ground_m`/meter floorplans with an exact snapshot
  ID, content SHA, and current calibration fingerprint tuple. Stale,
  conflicting, or malformed responses never replace the active record, and
  the dashboard clears points instead of retaining a dot across a mismatch.
  When a reviewed ray-to-floorplan transform exists, `floorplanAlignment`
  reports its quality/count/residual evidence and whether it was applied.
- A configured `coverageEnvelope` is a separate camera-local geometry contract.
  Its regions are a union of X/Z polygons and its boundary tolerance never
  moves, clips, or smooths a point. For legacy/noncanonical candidates it is
  the producer-side admission surface. Canonical world points remain owned by
  the producer's revision-bound world state: `coverageInside=false` is preserved
  as diagnostic provenance and is not silently applied as a second dashboard
  rejection gate. `displayBounds`/`xMin`/`xMax`/`zMin`/`zMax` cover both the
  raster and the configured envelope, while `floorplanBounds` continues to
  describe only the raster. Consumers use raw metric `x`/`y` coordinates rather
  than remapping out-of-range floorplan normalization. Cameras without a
  configured envelope retain the active-floorplan rectangle as their semantic
  `floorplanBounds`; the producer publishes the exact fixed, bounded display
  envelope for canonical live points. The dashboard consumes that envelope
  verbatim. It is not an alternate world estimator or a PCF extension.
- Camera-local mode with an active-floorplan provider has no config-bounds or
  auto-extents substitute. Before the first valid record, a provider result of
  `None` records `startup_pending` and emits neither a camera-local `bev-frame`
  nor an error `bev-status`; it does not invoke the fatal callback. A malformed
  payload or provider exception fails immediately. Once one valid record has
  made the camera ready, a missing, malformed, or rejected authority is a fatal
  `active_floorplan` failure and the previous bounds are never reused.
- A valid authority plus an exact tracking frame with no people is still a
  normal BEV publication (`footpoints: []`) and a renderer success. Health must
  not reinterpret empty occupancy as absence of a frame attempt.
- The DS9.1 baseline and disabled V3DT adapter scale image anchors, candidate
  points, and bbox coordinates from the track/frame image size into the active
  calibration snapshot's `image_size` before BEV homography, ray-floor, or
  registered-depth unprojection. The disabled V3DT adapter does not retain its historical
  unscaled source-frame path.
- `footpoints[].motionMode`, `posture`, `trailAppendAllowed`, and `idleJitterM` are producer-owned human-pathing diagnostics from `PersonGroundState` (`noesis/telemetry/person_ground_state.py`). When `trailAppendAllowed` is false (stationary / sit / lie lock), backend trails must not grow new path samples for that track; the head may still update in place.
- When `NOESIS_BEV_ALIGNMENT_DEBUG=1` is set, `bev-frame` may include top-level `alignmentDebug`, and each footpoint may include `rawX`, `rawY`, `smoothed`, and `alignmentDebug` with per-candidate image anchors, ray-floor projections, optional static floorplan/MapAnything snapshot samples, live-world candidate diagnostics, snapshot ids, grid cells, and selected-coordinate bounds. These fields are diagnostic-only and are not used to place live tracked people.

- Optional JPEG binary: **Retired**. The framed `bev:<camera>` binary path is no
  longer produced (metadata-only mode is the supported baseline documented in
  `runtime_baseline.md`). The general binary coalescer remains available only
  for explicitly contracted binary payloads.
- In world mode (`frame_mode=world`), BEV footpoints remain producer-owned scene coordinates and should be treated as the canonical `track.world` head points emitted by the backend. The BEV renderer must not apply a second world-space low-pass filter to those points.
- Motion smoothing ownership is declared explicitly by `trail_smoothing_owner`.
  In the current baseline world-mode path the backend owns trail history
  (`trail_smoothing_owner=backend`) while `bev_world_points_smoothed=false`,
  because `DS9/noesis/pipelines/hooks.py` and the shared
  `person_ground_state.py` already own the only track-position filtering stage
  (human CV filter + stationary lock).
- When `trail_smoothing_owner=backend`, `trails` carries the producer trail polylines already used by the BEV renderer, in the declared BEV `frame` with epoch-millisecond sample times. The dashboard should render those directly instead of reconstructing its own history from `footpoints`.
- `trails[].canonicalWorld` is the producer-owned authority marker for the
  retained trail, including frames where the current `footpoints` cohort is
  empty. A `true` marker permits a finite trail sample with
  `coverageInside=false` to remain displayable; coverage remains diagnostic for
  that canonical world trail. Consumers must fail closed for an unmarked
  retained trail with no current canonical footpoint; during the transition,
  legacy payloads may infer authority only from a matching current footpoint.
- The post-tiler OSD trail consumer joins a canonical track only on the exact
  `(source_id, frame_id)` analytics cohort and maps the track's declared
  source-image basis into that source's configured mosaic tile. It must not use
  a last-seen row, bbox-bottom fallback, or edge clamp; tracker lifecycle or
  image-basis changes break the trail segment.
- World-mode BEV emits a canonical `anchor_hold` head point with
  `worldAdmission=held` so a visible track does not disappear merely because
  the current measurement was rejected. Held points never extend trail
  history. A canonical `cv_prediction` head is emitted with
  `worldAdmission=predicted` and may extend history only while
  `trailAppendAllowed=true`. Legacy/non-canonical held anchors remain
  inadmissible.
- Backend world-BEV trail history is keyed by tracker-local identity plus
  `trackerLifecycleGeneration` when available (`stableId` is fallback display
  metadata only). A reused numeric tracker ID removes its older generation
  immediately and cannot inherit or connect that trail.
- Coordinate note: BEV renders on the ground plane (XZ). `footpoints[].x` is X and `footpoints[].y` is Z in the declared `frame`.
- The active-floorplan BEV is a local display surface, so it declares
  `frame=camera_local_ground_m`. It does not change canonical tracking/world
  ownership: strict observations and the global world snapshot remain in
  `backend_world_m`, with any world-to-local projection applied only for this
  display.

### BEV Trail Jitter Regression Checks

When a camera-local floorplan trail looks jumpy or twitchy, first check whether the producer is mixing coordinate spaces or revisions before adding smoothing. `BevRenderer` resets that camera's smoother and trail history when the active floorplan coordinate-space signature changes. Live tracked-person history has one stable source, `world_to_camera_local`; registered depth and floor-ray candidates are diagnostic only and cannot create source-switch trail segments. Canonical live points are not smoothed again in BEV: each displayed point is the direct revision-checked transform of that cohort's filtered `track.world`.

BEV renderer health is fail-closed. A current homography failure publishes a
`bev-status` error and is never hidden with a last-known transform. Homography
or WebSocket publication failure updates `noesis.bev.health` and invokes the
runtime failure callback; a runtime claiming required BEV health must surface
that false state rather than continuing with stale geometry.

Extrinsics updates clear the affected camera's active-floorplan record and
floorplan cache. Shared alignment updates clear all records and in-memory
caches. Persisted floorplan caches are bound to their calibration fingerprint,
so a mismatch is a cache miss, not permission to publish the previous geometry.

Also inspect producer pathing health: `motionMode` / `posture` thrash, `trailAppendAllowed=false` while still growing path history, elevated `idleJitterM` while a person is clearly stationary, and `world_source` flip rate (sticky hysteresis should keep sources stable across brief pose dropouts). Sitting/lying vibration is primarily a contact-geometry + idle-lock problem, not something to “fix” with heavier global EMA.

Use `scripts/bev_alignment_diagnostics.py` against the same native WebSocket
path that reproduces the issue and inspect `trail_segment_speed_mps`,
`top_trail_segments`, source transitions, raw jumps, and display-selection
reason. A cosmetic smoother may hide the symptom; a durable fix explains
whether it came from a floorplan reset, identity/tracker transition, projection
policy change, or bad raw anchor geometry. Historical experiments are recorded
in `plans/archive/ds8/ds8_design_decisions.md`.

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
    "depth_map_ref": "noesis-depth://artifact/<sha256>",
    "minmax": [<float min>, <float max>],
    "unit": "m"
  }
}
```

The reference is an opaque stable identifier for the internal artifact, not a
filesystem path or fetch URL. Public telemetry never reveals the storage root,
camera directory, or backend URI.

Publication is durable-before-visible across the DS9.1 baseline and disabled
V3DT adapter.
The MapAnything worker must receive the exact `WriteHandle` commit receipt
before it records the depth frame, constructs `DepthResult`, or publishes this
message. `NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S` controls that bounded wait: the
default is 30 seconds, input must be finite, and valid values are clamped to
0.1–60 seconds. A timeout or write error poisons the owned worker, signals a
fatal runtime component failure, and publishes no reference. The retired
`NOESIS_DEPTH_STORE_ENABLED` switch is not a supported bypass; there is no
direct-write or `memory://` publication fallback.

## 6. Tracking Telemetry (`type: tracking`)

Produced by `TrackingTelemetryPublisher`; people-only (class_id=0). `track_id` is internal and never exposed.

Empty frames are first-class: when a camera's active person count is zero, Noesis still publishes `type:"tracking"` with `tracks: []` and an advancing top-level `frame_id`. Count transitions to zero publish immediately; sustained emptiness uses the bounded `NOESIS_TRACKING_EMPTY_HEARTBEAT_HZ` gate (default 2 Hz). A tracker-key-set change also publishes immediately even when the count is unchanged. The publisher owns a contiguous per-source `tracking_publication_sequence`; every processed frame updates the tracker lifecycle registry and disappearance emits an exact tombstone. A same-camera numeric tracker ID may reuse its lifecycle generation only when it returns within 350 ms and strict bbox position/scale compatibility proves a short metadata gap. A moved, size-incompatible, expired, evicted, unknown, or post-reconnect return receives a new positive generation. Downstream clients can therefore distinguish rate-limited camera-frame gaps from absence and numeric tracker-ID reuse. On every exact processed frame, the canonical world service removes absent filter state from active authority. Scalar-only quarantine may restore it only across that same compatible 350 ms lifecycle grace; otherwise position, velocity, lock, rejection, and hold state start cold. The exact tombstone still breaks BEV/OSD trail history even when the compatible generation is reused. Tracker shadow age remains the owner of brief detector occlusion.

```json
{
  "type": "tracking",
  "source_id": <int>,
  "frame_id": <int>,
  "captured_at_us": <int epoch microseconds>,
  "track_count": <int>,
  "observed_at_us": <int epoch microseconds>,
  "capture_time_status": "estimated",
  "media_pts_ns": <int>,
  "tracking_continuity_contract": "noesis.tracking.publication-continuity",
  "tracking_continuity_contract_version": 1,
  "tracking_publication_sequence": <int>,
  "cohort": {
    "source_id": <int>,
    "frame_id": <int>,
    "observed_at_us": <int epoch microseconds>,
    "tracking_publication_sequence": <int>
  },
  "tracker_lifecycle_tombstones": [
    {
      "camera_id": "<string>",
      "tracker_id": <int>,
      "tracker_lifecycle_generation": <int>,
      "last_seen_frame_id": <int>,
      "last_seen_observed_at_us": <int>,
      "disappeared_at_frame_id": <int>,
      "disappeared_at_observed_at_us": <int>
    }
  ],
  "camera_id": "<string>",
  "coord_space": "<string>",
  "units": "<string>",
  "world_source": "backend_world_fused",
  "track_id_strategy": "camera_tracker_fallback",
  "calibration_version": "<string>",
  "tracking_contract_version": 3,
  "image_size": [<int width>, <int height>],
  "frame_size": [<int width>, <int height>],
  "observation_contract": "noesis.observation.person",
  "observation_contract_version": 1,
  "observations": [ /* strict ObservationEnvelope v1 objects */ ],
  "world_snapshot": { /* strict WorldSnapshot v1 object */ },
  "world_events": [ /* strict WorldEvent v1 objects */ ],
  "tracks": [
    {
      "stable_id": <int|null>,
      "tracker_id": <int>,
      "tracker_lifecycle_generation": <positive int>,
      "track_key": "<source:tracker:generation>",
      "camera_id": "<string>",
      "bbox": [<float>, <float>, <float>, <float>],
      "center": [<float>, <float>],
      "class_id": 0,
      "confidence": <float|null>,
      "tracker_confidence": <float|null>,
      "analytics": { /* NvDsAnalytics obj meta (camel + snake keys) */ },
      "zone": "<string|null>",
      "zone_source": "nvdsanalytics_roi"|"camera_default"|null,
      "zone_authoritative": <bool>,
      "frame_id": <int>,
      "observed_at_us": <int epoch microseconds>,
      "capture_time_status": "estimated",
      "media_pts_ns": <int|null>,
      "dwell_time": <float|null>,
      "bbox3d": { /* V3DT tracker-tuple diagnostics */ },
      "velocity3d": [<float>, <float>, <float>], /* tracker tuple */
      "visibility": <float|null>,
      "image_foot": [<float>, <float>],
      "image_base": [<float>, <float>],
      "world": [<float>, <float>, <float>],
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
      "world_source": "bbox3d"|"pose_depth_fused"|"person_anchor_depth_fused"|"pose_depth_only"|"person_anchor_depth_only"|"pose_floor_only"|"person_anchor_floor_only"|"gravity_drop"|"cv_prediction"|"image_motion_prediction"|"anchor_hold"|null,
      "world_resolver_confidence": <float 0..1|null>,
      "world_resolver_selected_id": "floor_ray"|"registered_depth"|"pose_scale"|"gravity_reconstruction"|"",
      "world_resolver_fused": <bool|null>,
      "world_resolver_disagreement_m": <float|null>,
      "world_floor_range_m": <float|null>,
      "world_floor_range_limit_m": <float|null>,
      "world_floor_incidence_sin": <float|null>,
      "world_floor_admitted": <bool|null>,
      "world_floor_rejection_reason": "floor_ray_range_exceeded"|"floor_ray_geometry_invalid"|null,
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
      "idle_jitter_m": <float|null>,
      "sticky_world_source": "<string|null>",
      "source_switch_count": <int|null>,
      "scene_prior": {
        "contract": "noesis.scene_prior.track_diagnostic",
        "contract_version": 1,
        "prior_id": "<immutable prior id>",
        "space_id": "<physical space id>",
        "mode": "shadow",
        "coordinate_frame": "backend_world_m",
        "status": "pass"|"warning"|"fail"|"unknown"|"error",
        "inside_extent": <bool>,
        "inside_authored_space": <bool>,
        "evidence_observed": <bool>,
        "evidence_confidence": <float 0..1>,
        "reasons": ["<stable reason>", ...]
      } | null,
      "lower_body_occluded": <bool|null>,
      "lower_body_occlusion_level": "none"|"feet_ankles"|"knees"|"waist_hips"|null,
      "lower_body_occlusion_confidence": <float|null>,
      "lower_body_occlusion_reason": "<string|null>",
      "projection_confidence": <float|null>,
      "temporal_confidence": <float|null>,
      "reid_confidence": <float|null>,
      "reid_required": <float|null>,
      "reid_identity": "<string|null>",
      "appearance_id": "<string|null>",
      "identity_state": "unknown"|"provisional"|"visitor"|"resident"|"handoff"|null,
      "identity_kind": "resident"|"visitor"|"provisional"|null,
      "overlap_permit": <bool|null>,
      "resident_uuid": "<string|null>",
      "display_name": "<string|null>",
      "visitor_generation": <int|null>,
      "identity_observation_key": {
        "run_id": "<string>",
        "camera_id": "<string>",
        "tracker_id": "<string>",
        "frame_id": <int>,
        "observation_id": "obs1:<sha256>"
      } | null, /* authoritative identity-v2 only */
      "identity_v2": {
        "mode": "authoritative",
        "state": "unknown"|"provisional"|"visitor"|"resident",
        "reason": "<string>",
        "subject_id": null,
        "compatibility_sid": <int|null>,
        "display_name": "<string|null>",
        "resident_uuid": "<uuid|null>",
        "visitor_generation": <int|null>,
        "calibrated_confidence": <float|null>,
        "provisional_evidence_count": <int|null>,
        "overlap_permit": <bool|null>,
        "fresh_embedding": <bool|null>,
        "evidence_persistence": "durable"|"queued"|"dropped"|null
      } | null,
      "id_event": "<string|null>",
      "id_reject_reason": "<string|null>",
      "embedding_present": <bool|null>,
      "embedding_sequence": <int|null>,
      "embedding_model_sha256": "<lowercase sha256|null>",
      "embedding_dimension": <int|null>,
      "pose_present": <bool|null>,
      "sid_candidate": <int|null>,
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

For `tracks[].zone`, the shared DS9.1/V3DT resolver treats per-object
`analytics.ocStatus` as the primary household room membership. A unique
`analytics.roiStatus` label is compatibility evidence only when `ocStatus` has
no nonempty membership. Labels are compared exactly: identical duplicates are
allowed, but empty, padded, over-160-character, malformed, or multiple distinct
labels produce no authoritative zone. A camera-derived fallback is stamped
`zone_source="camera_default"` and `zone_authoritative=false`.

Tracking telemetry has two world-source scopes:

- Top-level `world_source="backend_world_fused"` advertises that the baseline DS9.1 runtime owns the canonical world estimator in the backend.
- Per-track `world_source` records which observation path updated that specific track on the current frame.
- Per-track `world_source="cv_prediction"` is a bounded continuation of the
  canonical constant-velocity state when the current metric observation is
  missing, stale, or physically rejected. It is not a fresh measurement and
  does not advance the
  last-good measurement timestamp; `world_measurement_accepted=false` and
  `world_filter_prediction` explain the displayed point. `anchor_hold` remains
  a retained last-good position after a missing or rejected measurement.
  `world_source="image_motion_prediction"` is the stricter projective
  continuation used when the current bbox moved but its metric/depth anchor was
  missing or rejected.  It transports the last physically accepted image foot
  through the current bbox affine change and projects that pixel through the
  active corrected floor plane.  It is non-authoritative, never updates the
  filter or last-good origin, is bounded by image-motion, ray, metric-speed,
  and short-TTL gates, and exposes `world_prediction_image_foot` plus
  `world_prediction_provenance`.  If those gates fail after material bbox
  motion, the track is omitted for that frame rather than frozen at an old
  position.
- `cv_prediction`, `image_motion_prediction`, and `anchor_hold` remain
  tracking/BEV continuity and are never submitted as fresh global-world
  observations. Canonical global observations require a complete
  `world_frame_revision` plus `world_transform_sha256`; mixed target-frame
  revisions are retained as conflict evidence instead of being averaged.

The producer applies the human-speed limit to the proposed published filter
state, not only to its stored velocity. An observation inside the measurement
innovation allowance is quarantined when it would create an unreachable
same-segment step. A coherent relocation must instead be published as a new
`trail_segment_id` with `trail_break_required=true`.

`tracks[].scene_prior` is optional, additive shadow evidence for cameras bound
by the configured site catalog. It evaluates only a valid
`world_frame="backend_world_m"` position and never changes `world`,
`world_valid`, `world_source`, canonical observations, or global world state.
`unknown` means the prior or world observation has insufficient coverage;
`warning`/`fail` are review diagnostics, not tracking rejection. The exact
revision contract is `noesis.scene_prior.revision` v1.

The additive `observations` array is the canonical product input. Each strict
`ObservationEnvelope` carries producer/runtime/run identity, a monotonic
per-source sequence, tracker-local identity, capture/observation/publication
time semantics, media PTS, coordinate frame and units, content-addressed
calibration/model/config provenance, image evidence, and an optional uncertain
world observation. `tracks` remains a compatibility/operator-diagnostic view;
new product consumers validate `observations` instead of inferring missing
semantics from legacy track fields. Authoritative generated definitions live at
`contracts/schema/observation_envelope.schema.json` and
`contracts/typescript/noesis-contracts.ts`.

Each producer-created observation may also carry bounded
`payload.world_diagnostics`. This non-biometric evidence preserves the raw
floor/depth candidates, prefilter measurement and prediction, floor admission,
physical innovation decision, fusion-policy weights, and depth-registration
status used for that exact track frame. When no strict world observation is
admitted, `first_divergence_reason` is always populated by the canonical world
service from the earliest explicit producer rejection it can preserve (physical
measurement rejection, floor admission rejection, producer quality reason,
depth-registration status, or a structural world-envelope failure). These
diagnostics explain a missing world entity; they never make an invalid candidate
admissible and never contain embeddings, identity labels, masks, or images.

`tracker_id` is exposed only as a camera/run-local association and diagnostic
key so a public track can be joined to its strict observation. It is not a
user-visible or durable person identity and must never substitute for resolved
identity fields.

Fresh persisted embedding provenance is an all-or-none triad on both the public
track and its matching `observations[].payload` when authoritative identity-v2
owns that row:

- `embedding_sequence` is the exact sequence of the durably appended private
  identity-evidence record.
- `embedding_model_sha256` fingerprints the active ReID engine bytes.
- `embedding_dimension` is the extracted tensor dimension (currently 256 for
  the reviewed Swin `fc_pred` profile).

The triad is stamped only after the private evidence append returns a complete,
model-matching durable record for every observation in that source frame. Async
authoritative publication may omit the triad while the frame is still being
persisted; `identity_v2.evidence_persistence` is then `queued` or `dropped`.
Default shadow scoring persists its private evidence independently and never
retrofits this triad onto an already admitted canonical tracking row.
Partial triads are contract errors. Raw embedding vectors are never part of
public tracking or observation payloads.

`embedding_present` is exact current-frame server-extraction truth. It is
`true` only when the producer extracted a model-profile embedding for that
exact track/frame; a legacy StableID gallery/cache diagnostic cannot set it.
In authoritative mode, successful identity-v2 validation additionally creates
`identity_observation_key`, and `fresh_embedding` describes its live identity
decision. In shadow mode, those nested decision/linkage fields remain private
and are not joined back into canonical WebSocket tracking. Authoritative
`evidence_persistence` is
`durable` when the triad is present, `queued` while async score-only evidence
is awaiting the writer, and `dropped` when bounded admission discarded it.
Queued/dropped authoritative rows retain live identity truth but are not
durable evidence anchors. In authoritative mode, a missing tensor or
tracker-continuity hold sets `identity_v2.fresh_embedding=false` and clears the
key/triad. In either mode a missing exact tensor reports
`embedding_present=false`.

`observations[].payload.depth_present` is semantic evidence, not a proxy for a
non-null depth field. Registered depth is usable only when both
`depth_status="ok"` and `depth_registration_status="ok"`,
`depth_registered_m` is finite and positive, and a present `depth_used_m` is
also finite, positive, and equal to `depth_registered_m` within relative and
absolute tolerance `1e-6`. A missing `depth_used_m` is allowed; a raw anchor,
rejected registration, or registered/used mismatch makes `depth_present=false`
and cannot become a registered-depth BEV anchor. The BEV producer additionally
enforces its operational `0.05 < depth_registered_m < 50.0` meter guard.

The nested `world_snapshot` is the exact canonical snapshot produced in that
publication cycle. The tracking sequence, lifecycle/tombstone publication
state, health progress, and world authority advance only after the ordered
batch has bounded admission and private fusion commits. A pre-admission failure
discards the prepared world candidate and reuses the same
tracking/observation/event sequence on retry. An authority failure after
admission aborts the unresolved gate before any client delivery and poisons the
publisher. Optional journal queue/write failure is different: it is reported
as degraded reconstructable persistence and cannot block the canonical cohort.
A gate release failure after successful commit also poisons; no successor may
conceal the missing cohort. The same committed snapshot is then emitted
independently:

```json
{
  "type": "world_snapshot",
  "source_id": <int>,
  "frame_id": <int>,
  "observed_at_us": <int epoch microseconds>,
  "tracking_publication_sequence": <int>,
  "cohort": { /* identical to tracking.cohort */ },
  "payload": {
    "contract": "noesis.world.snapshot",
    "contract_version": 1,
    "snapshot_id": "<run-id>:<sequence>",
    "producer": {"runtime": "ds9", "instance_id": "<host>", "run_id": "<uuid>", "software_revision": "<revision>"},
    "sequence": <int>,
    "observed_start_us": <int>,
    "observed_end_us": <int>,
    "published_at_us": <int>,
    "frame": "backend_world_m",
    "units": "meters",
    "entities": [ /* subject, lifecycle, covariance, velocity, source evidence, conflicts */ ]
  }
}
```

Consumers reject unsupported major versions. Contradictory simultaneous camera
positions are retained as rejected evidence with `conflict=true`; they are
never averaged. Empty snapshots are authoritative absence/lifecycle evidence,
not a transport failure. See `contracts/schema/world_snapshot.schema.json`.

Every `entities[].sources[]` row preserves the exact strict `observation_id`,
its source `zone`, and that label's `zone_source`/`zone_authoritative`
provenance. `nvdsanalytics_roi` is the only spatially authoritative zone
source; the name denotes authoritative nvdsanalytics polygon-membership
evidence from primary `ocStatus` or ROI-only compatibility, not a requirement
for a second ROI-filtering configuration. A `camera_default` label remains
available for camera-local occupancy, dwell, and operator diagnostics but
cannot populate `room_id`. `room_id` must equal the sole exact authoritative
non-null zone among accepted contemporaneous source observations. Rejected
positional evidence cannot vote. No accepted authoritative zone requires
`room_id=null`; multiple distinct accepted zones require `room_id=null` and
entity conflict state.

Each transition is also emitted as
`{"type":"world_event","cohort":...,"source_id":...,"frame_id":...,"observed_at_us":...,"tracking_publication_sequence":...,"payload":...}`
inside that same ordered batch.
`noesis.world.event` v1 covers `appeared`, `held`, `resumed`, `lost`,
`conflict_started`, and `conflict_cleared`; it carries the durable subject,
producer sequence/time, world frame, last position, and reason. Events are
derived from canonical snapshot transitions, not browser timers. Observations,
snapshots, and events are retained in the bounded owner-only integrity-chained
world journal configured by `NOESIS_WORLD_JOURNAL_PATH`,
`NOESIS_WORLD_JOURNAL_MAX_RECORDS`, and
`NOESIS_WORLD_JOURNAL_RETENTION_HOURS`. Canonical runtime uses a finite
`AsyncContractJournal` only for reconstructable retention. Its typed admission
receipt proves the exact payload count entered the bounded persistence queue;
it is deliberately not durability or authority proof. The worker retains the
synchronous hash-chained SQLite journal, exposes pending/capacity/failure
health, and drains on orderly shutdown. Queue saturation or later durable
failure degrades persistence without stalling the media or spatial-publication
path. A configured retention limit smaller than one complete current cohort is
rejected before queueing, so pruning cannot split a retained cohort. Fusion
state is service-private; read consumers receive only the cached immutable
last-committed snapshot, so inspection cannot consume a sequence, expire an
entity, clear a source, or bypass the revision/authority boundary.

The active baseline uses one universal measurement resolver followed by the
existing `PersonGroundState` in
`noesis/telemetry/person_ground_state.py`. The analytics hook independently
constructs current-frame floor-ray and registered-depth hypotheses at their own
exact anchors, plus a weak gravity reconstruction only when the existing
upright/occlusion state permits it. Seated or lying hips/torsos are not floor
contacts, and bent-leg ankle extrapolation remains rejected.

Every candidate is independently range/geometry checked and carries full 3x3
covariance derived from ray incidence, pixel/contact uncertainty, depth
support/spread, occupied-person registration residuals, posture, and occlusion.
No room or camera chooses a different strategy. Compatible hypotheses genuinely
contribute through covariance intersection. Statistical compatibility alone is
insufficient: candidates separated by more than the universal 1.25 m maximum
remain primary/alternate rather than being averaged. PCF extent, authored
boundary, observed confidence, and floor elevation are soft evidence only and
never clamp the result. The selected current measurement then enters
PersonGroundState, which remains the sole temporal filter and physical gate.
The exact design is in
[`universal_world_localization.md`](universal_world_localization.md).

- `pose_depth_fused`: compatible pose-floor and registered-depth hypotheses
  both mathematically contributed to the current measurement before the
  PersonGroundState update.
- `pose_floor_only`: pose anchor updated the world-state filter without a usable DAv2 observation on that frame.
- `person_anchor_depth_fused`: compatible person-contact floor and registered
  depth hypotheses both mathematically contributed on a frame without usable
  pose.
- `person_anchor_floor_only`: the person mask/depth anchor updated the world-state filter without a usable DAv2 observation on that frame.
- `gravity_drop`: a stored upright height reference allowed a floor-consistent
  gravity drop. This is both the degraded path when no current person anchor is
  available and the authoritative path while lower-body occlusion is active.
  In the latter case, a syntactically valid bbox/depth bottom at a counter or
  table edge is deliberately demoted before fusion.
- `anchor_hold`: no current valid observation; the estimator is briefly holding the last reliable world state.
- `image_motion_prediction`: the current metric/depth observation was missing
  or rejected, so a last-accepted image foot was transported by the current
  detector-box affine motion and floor-projected for bounded display
  continuity.  This point is not a measurement and must not become a new
  predictor origin.
- `bbox3d`: `v3dt` mode only. `bbox3d` and `velocity3d` preserve the locked
  profile's tracker tuple for diagnostics. They are not public world-space
  vectors. The producer derives the bbox ground endpoint, applies the required
  `xzy` map, and publishes only the resulting Y-up meter position as
  `world_frame="backend_world_m"`; missing/invalid bbox or axis metadata fails
  closed.
- In V3DT mode, `image_foot` is the native tracker ground-foot observation and
  `image_base` is the independently projected opposite cuboid endpoint. This
  distinction is intentional and is covered by the global-world v2 gate.
- This per-camera SV3DT contract makes no MV3DT overlap, time-sync, peer-ID, or
  fused-position claim.

Human pathing fields (producer-owned):

- `motion_mode`: locomotion class from the stationary lock (`walk` / `idle` / `sit` / `lie` / `unknown`).
- `posture`: geometric posture guess (`standing` / `sitting` / `lying` / `unknown`).
- `trail_append_allowed`: when false, BEV and OSD trail history must not grow (person is locked stationary).
- `idle_jitter_m`: residual magnitude while locked; useful for tuning deadzones.
- `sticky_world_source` / `source_switch_count`: hysteresis diagnostics for source thrash.
- `lower_body_occluded`: true when recent upright anatomy plus current
  keypoint/bbox evidence indicates that the visible detector bottom is an
  occluder edge rather than the person's floor contact.
- `lower_body_occlusion_level`: deepest hidden support region:
  `feet_ankles`, `knees`, or `waist_hips`. The producer retains this state until
  direct lower-body evidence is clear for three consecutive updates.
- `lower_body_occlusion_confidence` / `lower_body_occlusion_reason`: bounded
  operator diagnostics explaining the keypoint-loss and scale-collapse evidence.
- During authoritative occlusion gravity-drop, `image_foot` and `image_base`
  both carry the calibrated reconstructed floor-contact pixel. Consumers must
  not reselect bbox bottom or the visible counter/table edge.

Depth exposure:

- `depth_used_m` is the current registered DAv2 anchor range that qualified as
  a resolver hypothesis. It may be present even when another hypothesis is
  selected. `world_source` and the compact `world_resolver_*` fields explain
  the authoritative decision; exact contributors and candidates are available
  only through request-gated BEV `resolverDiagnostics`.
- `depth_anchor_m` is the raw anchor depth carried by `NOESIS.OBJECT_DEPTH`; it may be present even when `depth_used_m` is null.
- `depth_registered_m` is the room-registered DAv2 anchor depth after applying the offline DAv2→MapAnything mapping for that camera; this is the value projected on the current anchor ray when registration is active.
- Registered-depth consumers use the coherence rule above. If
  `depth_used_m` is present but differs from `depth_registered_m` outside the
  `1e-6` relative/absolute tolerance, neither canonical semantic depth nor the
  camera-local registered-depth display path may accept it.
- `depth_registration_status` is `ok` when the runtime used a valid
  registration mapping on that frame. A rejected or stale metric sample is
  never projected, but it cannot suppress an independently valid floor ray in
  any camera.
- `depth_registration_id` identifies the exact per-camera registration artifact entry used by the estimator.
- `depth_status`, `depth_anchor_source`, `depth_sample_count`, and `depth_valid_fraction` are published on both `tracking.tracks[]` and `stats.payload.cameras[*].tracking.active_tracks[]` so the runtime OSD and dashboard can explain whether baseline depth is contributing on a given frame.
- `depth_anchor_sample_count` and `depth_anchor_valid_fraction` surface support
  of the exact lower-body/ankle depth band. Resolver candidate diagnostics and
  covariance are the canonical explanation for whether it was selected,
  fused, retained as an alternate, or rejected; whole-mask support is not a
  substitute.
- The on-screen `depth=` label is optical/registered range sourced from
  `depth_used_m`; it is not canonical world Z. The parser still removes or
  preserves legacy `z=` fragments during mixed-version transitions.
- The strict observation's `depth_present` boolean follows the usable-depth
  rule above. Consumers must not infer it from `depth_anchor_m`,
  `depth_median_m`, or attachment attempts independently.

When pose anchoring, gravity-drop, and recent-anchor hold all fail, DS9.1 leaves `world_valid=false` instead of promoting bbox-bottom floor projection into a synthetic world point.

Validation diagnostics:

- `projection_confidence`, `temporal_confidence`, `reid_confidence`,
  `reid_required`, `reid_identity`, `appearance_id`, `identity_state`,
  `identity_kind`, `overlap_permit`, `resident_uuid`, `display_name`,
  `id_event`, `id_reject_reason`, `embedding_present`, `pose_present`,
  `sid_candidate`, `occluded`, and
  `occlusion_uncertainty_m` are optional diagnostics for validation and UI
  explanation. Producers may omit them when that evidence is unavailable, but
  consumers must not reinterpret missing values as a pass.
- Household identity fields (additive, backward compatible):
  - `identity_state`: lifecycle state (`unknown`, `provisional`, `visitor`, `resident`, `handoff`).
  - `identity_kind`: public kind exposed on the wire (`resident`, `visitor`, `unknown`, `provisional`).
  - `reid_confidence`: best match score used for the current assignment decision.
  - `reid_required`: threshold applied for that decision (when known).
  - `overlap_permit`: `true` when topology grants dual-camera activity for the same SID.
  - `resident_uuid` / `display_name`: enrollment metadata (Phase 3; may be null in Phase 0).
  - `visitor_generation`: non-repeating generation for a recycled compatibility visitor SID.
  - `id_event` / `id_reject_reason`: assignment lifecycle and reject diagnostics.
  - `embedding_present` / `pose_present`: whether ReID/pose evidence was available on the frame.
  - `embedding_sequence` / `embedding_model_sha256` /
    `embedding_dimension`: optional persisted-evidence provenance, present only
    as the complete triad described above.
  - `sid_candidate`: provisional candidate SID before confirmation (when applicable).
- A sub-second metadata gap may retain a settled StableID without a fresh
  embedding only for the same camera and tracker-local ID, within 0.75 seconds,
  and only when strict bbox overlap, center-motion, and area-ratio gates all
  pass. The carried embedding remains internal and the current row reports no
  fresh embedding evidence. This continuity aid cannot claim an ID already
  active in the same frame and is not cross-camera ReID.
- Identity-v2 runs one joint resolver call after a complete source-frame
  primitive batch is detached from SDK metadata. In default `shadow` mode,
  canonical tracking/world/BEV is admitted first and never waits for or accepts
  mutation from that call. A separate bounded worker retains only the newest
  pending scalar snapshot per source, so an older shadow cohort or empty
  heartbeat may be coalesced when scoring/visitor persistence is slower than
  input. Capacity, copy, scoring, and persistence failures affect shadow
  comparison freshness only. Consequently, a shadow diagnostic is not a
  same-frame field promised on the canonical `tracking` row; consumers must not
  wait for it or join it by last-seen state. No frame surface, SDK object,
  borrowed diagnostic row, or embedding is published on WebSocket. A frame
  suppressed by the tracking publication cadence has no shadow work. In
  `authoritative` mode, the call remains synchronous after the one-shot metadata
  walk because its result owns same-frame public identity. V2 then clears and
  replaces all legacy public identity fields: an open-set unknown has null
  `stable_id`, name, resident UUID, and visitor generation; a visitor has a
  compatibility SID and generation but no name; a resident has a compatibility
  SID, durable UUID, and display name. `stable_id` is only the numeric
  compatibility boundary, never a durable subject identity. After a source
  reconnect, the internal identity tracklet key is scoped by `source_epoch` so
  a reused numeric tracker ID starts a new coordinator lifecycle; the public
  diagnostic `tracker_id` remains unchanged.
- Internal resident/visitor subject keys do not become new legacy wire IDs.
  Shadow `subject_id` remains private comparison evidence and is not emitted on
  canonical tracking. Authoritative output uses resident UUID/name or visitor
  SID/generation and leaves legacy `reid_identity` / `appearance_id` clear.
- In authoritative mode, a resolved label may be held briefly across SGIE
  reinference gaps on the exact same tracker-local track. Such rows set
  `identity_v2.fresh_embedding=false`, cannot produce overlap proof or an
  enrollment key, and are cleared immediately when the track disappears. A
  fresh resolver unknown still clears the label; this bounded continuity hold
  never converts rejection into a match. The corresponding shadow decision is
  private cache/evidence only.
- Once a fresh resident/visitor subject is accepted for a camera-local tracker,
  it cannot switch directly to another subject while that tracker state is
  continuous. Alternative subjects are hard-masked. The accepted subject is
  retained only when it independently passes every open-set and exclusivity
  gate; otherwise the fresh row is `unknown`. A different subject becomes
  eligible only after a real fresh-evidence gap expires the complete tracker
  state. The household resident prior cannot rescue a rejection.
- A visible track with neither a fresh embedding nor an eligible continuity
  hold is `provisional`, not an open-set `unknown`. `unknown` is reserved for a
  resolver decision backed by fresh evidence that rejected every admissible
  subject; canonical world fusion preserves that distinction exactly.
- Authoritative identity is resolved after the hook's first one-shot walk over
  transient DeepStream object metadata. That walk forces neutral `#XX` and never
  reuses a legacy SID. An explicit downstream Service Maker
  `BatchMetadataOperator` probe at the tiler sink then receives fresh wrappers
  after whole-frame resolution and joins the bounded exact
  `(camera,frame,tracker)` decision cache. Residents render `#<sid> <name>`,
  visitors render `#<sid>`, and any missing, stale, mismatched, unknown, or
  provisional decision remains `#XX`. No wrapper is retained between passes,
  and legacy SID mask coloring remains disabled.
- `identity_observation_key` exists only when the server extracted a valid
  model-profile embedding for that exact frame. No embedding bytes cross the
  WebSocket boundary. The key can be used for the authenticated two-step v2
  enrollment flow while its bounded server cache remains live.
- Dual-camera sharing is not sticky. An identity-specific overlap permit is
  emitted only when the configured topology edge, contemporaneous world
  distance, time delta, and appearance threshold all pass for that exact pair.
  A later frame must prove the exception again.
- Occupied semantic acceptance is stricter than schema serialization or a
  lifecycle canary. A passing live report must join one same-frame public person
  track to its strict observation and exact private hash-chain evidence row,
  prove the complete embedding triad, pose, usable depth, finite
  `backend_world_m`, no raw vectors, and no pipeline errors. An empty house is
  reported as blocked. The gate is implemented and unit-tested; this document
  does not claim that an occupied live run has passed.
- The Noesis/Menon validation toolbox consumes these fields when present for
  `TRACK.projection_confidence`, `TRACK.occlusion_bridge`,
  `TRACK.identity_continuity`, `TRACK.reid_geometry_consistency`, and
  per-track audit artifacts.
- Cross-space validation traces may refer to this track `world` vector as
  `backend_world_m` when handing it to Menon. Menon must apply its declared
  room/scene alignment exactly once and should expose the transform stages in
  trace/debug evidence for `MENON.transform_audit`.

## 7. Control & RPC Message Types

Handled in `noesis/server/websocket.py`:

- `clear_stats` → clears latency samples and broadcasts updated stats.
- `set_vis_toggle` → visualization toggle; server broadcasts `toggle_update`.
- `trail_settings_update` (server → client) → current trail tuning config snapshot, including `enabled`.
- `bev-config` / `bev-overlay` → update BEV renderer config; ack via `bev-config-ack` or `bev-overlay-update`.
- Calibration RPCs: `pixel_to_world` → `pixel_to_world_response`; `set_extrinsics`, `set_align`, `solve_pnp` → corresponding `*_result` messages.
- Depth/MapAnything: `get_ma_depth` / `get_ma_depth_cache` → `ma_depth_response`.
- Floorplan: `get_floorplan` → `floorplan_response`.
- Auto-calibration: `auto_calibrate_pose` → `auto_calibrate_result`.
- Heartbeat: `ping` → `pong`.

The former `update_detection_config`, `set_detection_toggle`, and
`ma_heatmap_ready` messages are retired. None had a registered DS9.1/V3DT
runtime owner, so acknowledging or rebroadcasting them misrepresented frontend
state as applied. The dashboard no longer emits them and Menon's gateway does
not admit them. Any future live detector control must first define an explicit
runtime callback, application receipt, validation, and rollback contract.

### ma_depth_response

Request:

```json
{
  "type": "get_ma_depth",
  "camera": "<camera-id>",
  "request_id": "<required opaque correlation ID>",
  "ts_max_us": <optional int>,
  "cache_only": <optional bool>
}
```

- `get_ma_depth_cache` uses the same request shape and response shape, but forces `cache_only=true`.
- `request_id` is mandatory for both request types. It must be the canonical
  snake-case field, a non-whitespace string, and between 1 and 128 UTF-8 bytes.
  It is opaque: the server does not normalize it or generate a replacement.
  A missing, alias-only, non-string, empty, whitespace-only, malformed-Unicode,
  or oversized value closes that WebSocket with policy code `1008` and reason
  `invalid_depth_request_id` before provider admission. No uncorrelated JSON
  error is emitted.

```json
{
  "type": "ma_depth_response",
  "camera": "<camera-id>",
  "cache_only": <bool>,
  "served_from_cache": <bool>,
  "ts_us": <int>,
  "request_id": "<exact accepted request ID>",
  "ok": <bool>,
  "error": "<optional>",
  "payload": {
    "contract": "noesis.depth.bulk_snapshot",
    "contract_version": 1,
    "ts": <int>,
    "shape": [<H>, <W>],
    "snapshot_id": "<transactional write ID>",
    "snapshot_ref": "<portable store-relative ref>",
    "content_sha256": "<lowercase exact-snapshot digest>",
    "role": "capture_event_fused",
    "fusion_level": "intra_capture",
    "components": {
      "depth": {
        "component": "depth",
        "dtype": "<f4",
        "shape": [<H>, <W>],
        "byte_count": <int>,
        "sha256": "<lowercase raw-byte digest>",
        "url": "/api/v1/depth/snapshots/<camera>/<snapshot-id>/components/depth?snapshot_ref=<encoded-ref>&content_sha256=<digest>"
      },
      "conf": { "component": "conf", "dtype": "<f4", "shape": [<H>, <W>], "byte_count": <int>, "sha256": "<digest>", "url": "<same exact route>" },
      "mask": { "component": "mask", "dtype": "|u1", "shape": [<H>, <W>], "byte_count": <int>, "sha256": "<digest>", "url": "<same exact route>" },
      "rgb": { "component": "rgb", "dtype": "|u1", "shape": [<H>, <W>, 3], "byte_count": <int>, "sha256": "<digest>", "url": "<optional same exact route>" }
    },
    "normals": {
      "mode": "client_derived_depth_gradient_v1",
      "space": "camera",
      "dtype": "float32"
    },
    "capture_event_evidence_sha256": "<lowercase digest; exact fresh capture>",
    "capture_event": {
      "contract": "noesis.capture_event_controller",
      "contract_version": 1,
      "camera_id": "<canonical camera>",
      "request_kind": "depth",
      "capture_mode": "depth_only",
      "baseline_raw_timestamp_us": <int>,
      "raw_snapshot_count": <int>,
      "fusion_evidence_sha256": "<lowercase digest>",
      "fused_snapshot": {
        "timestamp_us": <int>,
        "snapshot_id": "<write ID>",
        "artifact_ref": "depth-zarr:<portable ref>",
        "content_sha256": "<lowercase digest>",
        "manifest_sha256": "<lowercase digest>",
        "event_id": "capture-event-sha256:<digest>",
        "source_snapshot_ids": ["<write ID>", ...],
        "snapshot_role": "capture_event_fused",
        "fusion_level": "intra_capture"
      }
    }
  }
}
```

Every `ma_depth_response` for an accepted request, including cache misses,
rate limits, capacity rejection, shutdown, timeout, provider errors, and
successful descriptors, carries the exact original `request_id`. A provider
cannot replace it. Every error response also carries `ts_us:0`; locally
generated provider failures use stable snake-case error codes rather than
exception text.

The payload is a compact control-plane descriptor, not a tensor container.
`depth_b64`, `conf_b64`, `mask_b64`, `normals_b64`, and any other inline binary
field are forbidden. `depth`, `conf`, and `mask` are required component rows;
`rgb` is optional. Every row binds an exact little-endian/raw dtype, shape,
byte count, SHA-256 digest, and relative same-origin URL. The browser retrieves
those URLs through the authenticated Menon gateway; it never receives the
owner-only Noesis bearer token or connects to the internal REST port directly.
`capture_event` and `capture_event_evidence_sha256` are present together only
for the successful fresh-capture response; an existing cache descriptor omits
both.
Only `role="capture_event_fused"` with `fusion_level="intra_capture"` is
publicly bulk-readable. Cache selection skips raw commits—even when a raw
commit is newer—and an exact raw-component request fails closed instead of
exposing an intermediate capture.

The descriptor is limited to 8,388,608 pixels, 64 MiB per component, and
128 MiB for the complete snapshot transfer. These limits admit an ordinary
3840x2160 depth/confidence/mask/RGB cohort while rejecting unbounded allocation.
Bulk component transfer, digest verification, and client normal generation are
measured separately from the 3 ms assembled-JSON WebSocket boundary. The
descriptor itself remains inside that JSON boundary.

- `cache_only=true` and `get_ma_depth_cache` return only an existing valid
  cached fused MapAnything payload. A raw-only store is a cache miss and
  returns `ok:false`,
  `error:"no_cached_depth"`, and no `payload`; it does not enter capture
  admission, open the MapAnything valve, select/fuse a cohort, or write storage.
- While authenticated REST `POST /api/v1/depth/refresh` owns its asynchronous
  manual window, polling clients use this cache-only form. It may report a miss
  until a source worker commits a snapshot, but it never competes for the
  process gate or creates a second capture owner.
- A non-cache request uses the one process-wide MapAnything valve and the same
  `CaptureEventController` as floorplan capture. Admission is per canonical
  camera plus one global non-blocking lease because the graph has only one
  valve. Cross-camera and same-camera contention both return
  `capture_event_busy`; the same code is returned when a REST manual window or
  its durable drain owns the gate.
- A successful fresh request is reloaded from the exact committed fused
  snapshot. The storage adapter revalidates its portable reference, write ID,
  timestamp, sequence, content and manifest digests, role, fusion level, and
  source write IDs. No `latest` lookup or previously cached payload can replace
  that result.
- Newly committed snapshots use commit-manifest payload version 2, which binds
  the raw component descriptors and digests used by this transport. Manifest
  version 1 remains readable for existing storage consumers, but it has no
  component manifest and therefore cannot produce a bulk descriptor or serve a
  component. The runtime fails that request with
  `bulk_component_manifest_missing`; it does not synthesize an inline or
  latest-snapshot fallback.
- Current runtime captures are explicitly depth-only. Full sealed fusion
  evidence records `rgb.status="not_requested"`; the public compact evidence
  exposes `capture_mode="depth_only"` and its hash. A future RGB-required call
  without a pipeline-owned timestamped frame fails with
  `rgb_frame_unavailable` rather than opening a second reader.
- `normals` declares a client display policy; no server normal tensor is sent.
  The dashboard worker derives depth-gradient normals after validating the
  exact component bytes. Those normals are not calibrated metric geometry.
  The Depth drawer's visible-floor estimator separately unprojects neighboring
  typed depth samples with the bound camera intrinsics and crosses metric
  tangents; extrinsics then provide its world-horizontal test. Reconstruction
  normals remain server/offline products derived from persisted depth plus
  camera calibration.
- `ma_depth_response` remains the MapAnything full-frame RPC contract. The always-on baseline DAv2 tracking lane does not publish a second full-frame WebSocket depth stream; it influences `track.world` through `NOESIS.OBJECT_DEPTH` and the fused backend estimator instead.

### floorplan_response

Returned from `get_floorplan` (`DepthStorageManager.generate_topdown_floorplan`):

```json
{
  "type": "get_floorplan",
  "camera": "<camera-id>",
  "request_id": "<optional>",
  "max_age_sec": <optional float>,
  "grid_res_m": <optional float>,
  "max_extent_m": <optional float>,
  "cache_only": <optional bool>,
  "scene_prior_only": <optional bool>,
  "snapshot_ref": "<optional exact depth response reference>",
  "snapshot_id": "<optional exact depth response write ID>",
  "snapshot_content_sha256": "<optional bulk descriptor content_sha256>"
}
```

```json
{
  "type": "floorplan_response",
  "request_id": "<id>",
  "camera_id": "<camera>",
  "cache_only": <bool>,
  "scene_prior_only": <bool>,
  "display_source": "pcf"|"static_fallback"|null,
  "served_from_cache": <bool>,
  "ts": <int>,
  "snapshot_ts": <int|null>,
  "snapshot_ref": "<portable store-relative reference>",
  "snapshot_id": "<transactional write ID>",
  "snapshot_content_sha256": "<lowercase digest>",
  "calibration_fingerprint": "<lowercase digest>",
  "frame": "camera_local_ground_m",
  "orientation": "camera_ground_right_forward",
  "floorplan_contract_version": 10,
  "units": "meters",
  "s_obj_to_m": <float>,
  "bounds": {"min_x": <float>, "max_x": <float>, "min_z": <float>, "max_z": <float>},
  "scale_m_per_px": <float>,
  "scale_scene_per_px": <float>,
  "point_count": <int>,
  "density": {"grid_b64": "<base64 float32>", "grid_shape": [<H>,<W>], "value_min": <float>, "value_max": <float>},
  "height": {"grid_b64": "<...>"},
  "height_agl": {"grid_b64": "<...>"},
  "distance": {"grid_b64": "<...>"},
  "observed": {"grid_b64": "<float32 0|1>", "grid_shape": [<H>,<W>]},
  "unknown": {"grid_b64": "<float32 0|1>", "grid_shape": [<H>,<W>]},
  "obstacle_height": {"grid_b64": "<...>"},
  "walkable": {"grid_b64": "<...>"},
  "inferred_walkable": {"grid_b64": "<float32 0|1 optional>"},
  "observation_meta": {
    "observed_definition": "one_or_more_valid_projected_depth_points",
    "unknown_definition": "zero_valid_projected_depth_points_within_grid_bounds",
    "observed_cells": <int>,
    "unknown_cells": <int>,
    "total_cells": <int>
  },
  "clean_floorplan_meta": {"...": "<optional debug metadata>"},
  "grid_res_m": <float>,
  "grid_res_scene": <float>,
  "max_extent_m": <float>,
  "max_extent_scene": <float>,
  "image_flip": {"u": <bool>, "v": <bool>},
  "exact_snapshot_reused": <bool optional>,
  "depth_burst_triggered": <bool optional>,
  "depth_burst_fresh": <bool optional>,
  "capture_event_evidence_sha256": "<lowercase digest; optional exact fresh capture>",
  "capture_event": { /* same compact controller evidence as ma_depth_response; request_kind=floorplan */ },
  "scene_static_height_agl": {"grid_b64": "<base64 float32>", "grid_shape": [<H>,<W>], "value_min": 0, "value_max": <float>},
  "scene_static_observed": {"grid_b64": "<float32 0|1>", "grid_shape": [<H>,<W>]},
  "scene_static_confidence": {"grid_b64": "<float32 0..1>", "grid_shape": [<H>,<W>]},
  "scene_composite_height_agl": {"grid_b64": "<base64 float32>", "grid_shape": [<H>,<W>], "value_min": 0, "value_max": <float>},
  "scene_composite_observed": {"grid_b64": "<float32 0|1>", "grid_shape": [<H>,<W>]},
  "scene_composite_source": {"grid_b64": "<float32 0|1|2>", "grid_shape": [<H>,<W>]},
  "scene_prior_meta": {
    "contract": "noesis.scene_prior.floorplan_composite",
    "contract_version": 1,
    "prior_id": "<immutable prior id>",
    "space_id": "<physical space id>",
    "mode": "shadow",
    "composition_policy": "live_observed_wins_static_fills_live_unknown",
    "live_observed_cells": <int>,
    "static_available_cells": <int>,
    "static_fill_cells": <int>,
    "composite_observed_cells": <int>
  },
  "scene_prior_error": "<explicit shadow-composition error; optional>",
  "scene_prior_diagnostic_height_agl": {"grid_b64": "<base64 float32>", "grid_shape": [<H>,<W>], "value_min": 0, "value_max": <float>},
  "scene_prior_diagnostic_observed": {"grid_b64": "<float32 0|1>", "grid_shape": [<H>,<W>]},
  "scene_prior_diagnostic_unknown": {"grid_b64": "<float32 0|1>", "grid_shape": [<H>,<W>]},
  "scene_prior_diagnostic_reconstruction_extent": {"grid_b64": "<float32 0|1>", "grid_shape": [<H>,<W>]},
  "scene_prior_diagnostic_surface_rgb": {"rgb_b64": "<base64 uint8 RGB>", "rgb_shape": [<H>,<W>,3]},
  "scene_prior_diagnostic_meta": {"source": "<manifest source model>", "derivation": "prior_conditioned_fusion_points_and_grid", "prior_id": "<immutable prior id>"},
  "error": "<string optional>"
}
```

- `scene_prior_only=true` is the read-only canonical PCF presentation request.
  DS9 bypasses static-frame capture, depth/floorplan caches, and MapAnything gate
  control, then derives the complete diagnostic raster family and point/floor
  layers from the immutable Scene Prior revision bound to the camera. Success
  declares `scene_prior_only=true`, `display_source="pcf"`, and the exact
  `scene_prior_meta.prior_id`. A missing or disabled binding is an explicit
  error; it is never replaced with a static-camera result.
- The `scene_prior_diagnostic_*` family includes density, height, height AGL,
  distance, gradient, obstacle height/mask, walkable/observed/unknown masks,
  inferred walkable, structural and surface evidence, authored room footprint,
  complete reconstruction extent, boundaries, measured perimeter, RGB surface
  color, confidence, and floor support. The reconstruction extent frames every
  PCF view; the authored footprint remains semantic room authority and does not
  crop measured evidence. All grids share the response's calibrated
  `camera_local_ground_m` bounds and orientation.
  In `scene_prior_only` mode this family is authoritative, so the response
  omits the redundant `scene_static_*` and `scene_composite_*` raster copies.
- `cache_only=true` returns only a contract- and calibration-valid memory/disk
  floorplan cache entry. On a miss it returns
  `error:"no_cached_floorplan"` before snapshot access, floorplan generation,
  persistence, cache fill, capture admission, or MapAnything gate control.
- A normal request is an explicit fresh capture: the shared controller creates
  exactly one depth-only fused cohort and floorplan generation is pinned to
  that result by `snapshot_ref`, `snapshot_id`, and
  `snapshot_content_sha256`. It never performs a generic latest/current cache
  lookup first. Cache reuse is requested explicitly with `cache_only=true`,
  and exact success atomically publishes independent bounded memory entries for
  its write-ID key and the cache-only `latest` alias. The exact result is never
  persisted over the generic disk cache.
- The three optional snapshot-identity request fields are an all-or-none
  continuation of a successful manual `get_ma_depth` response. This mode
  derives the floorplan from that exact committed fused snapshot without
  opening the depth valve, selecting a second cohort, or consulting `latest`.
  It is incompatible with `cache_only=true`. The response repeats the same
  identity; the depth descriptor's `content_sha256` is sent as
  `snapshot_content_sha256`. The response sets `exact_snapshot_reused=true`,
  `depth_burst_triggered=false`, and `depth_burst_fresh=false`.
- The successful response must repeat the fused snapshot's reference, write ID,
  content digest, and timestamp. Identity/integrity failure is returned as an
  error; the provider never substitutes the previously stale response or a
  newer unrequested `latest` snapshot.
- Only a successful floorplan in `camera_local_ground_m`/meters with bounded
  geometry, a current calibration fingerprint, and an exact snapshot identity
  may update the per-camera active-floorplan registry. Older versions are
  ignored, same-version conflicts fail closed, and rejected payloads never
  become BEV bounds.
- `obstacle_height` and `walkable` are optional clean layers generated for room floorplan responses when clean-surface estimation is available:
  - `obstacle_height`: float32 meters above an estimated floor plane (floor clamped to 0).
  - `walkable`: float32 {0,1} where 1 is walkable floor and 0 is obstacle/furniture.
- Version 8 made observation authority explicit. `observed` and `unknown`
  are exact complementary float32 {0,1} grids. Every rendered metric layer
  must treat `unknown=1` as unavailable rather than displaying its numeric
  fill value as measured floor. When clean walkability exists,
  `inferred_walkable` is present and equals `walkable ∩ unknown`, allowing the
  UI to distinguish inferred interior fill from directly observed floor.
- Version 10 changes the raster basis to calibrated camera-right and
  camera-forward projected onto the gravity-aligned ground plane. It also
  carries calibrated camera-height metric-scale conditioning and asymmetric
  observed bounds; clients must use
  `orientation=camera_ground_right_forward`.
- `image_flip` is diagnostic-only for floorplan responses. The serialized floorplan grids are already in their final camera-local X/Z orientation, so clients must not mirror the raster again using this hint.
- `point_count=0` with a 1x1 zero grid is an explicit no-valid-depth sentinel.
  It preserves RPC shape for UI diagnostics but is not floorplan quality evidence.
  DS9 behavior acceptance requires `point_count>0`, a non-sentinel grid, and a
  snapshot inside the requested age bound for every active camera.
- Scene-prior fields are additive and appear only for a camera with an exact
  catalog binding that enables floorplan layers. They are resampled from
  `backend_world_m` into this response's current calibrated camera-local grid.
  Source code `0` is unavailable, `1` is static prior, and `2` is live. A live
  observed cell always owns the composite value. Static evidence may fill only
  a live unknown cell; it never overwrites or relabels the original live
  `height_agl`, `observed`, `unknown`, `walkable`, or structural layers.
- The oai2-fe Depth drawer uses explicit `scene_prior_only` responses as the
  sole source for its standard Heatmap diagnostics, derived normals,
  confidence histogram/metrics, textured floorplan, and four established 3D
  representations. Opening the drawer requests that PCF payload automatically.
  Its Refresh control still issues a normal fresh static-frame request for
  later comparison work, but validates and discards that response from visible
  drawer state; it cannot replace an admitted PCF response.

The wire payload above is `floorplan_contract_version=10`. The retained
`noesis.ds9.floorplan-live-gate` v4 schema can replay exact-capture evidence, but
ordinary development validates only the affected fresh/cache/PCF path directly;
it does not require a promotion report.

### Depth/floorplan error codes

Depth and floorplan providers return machine-stable codes in `error`; callers
must not parse exception prose. The common transport/admission codes are
`no_provider`, `rate_limited`, `provider_capacity_exceeded`, `timeout`,
`shutting_down`, `camera_required`, `unknown_camera_alias`,
`depth_branch_unavailable`, `capture_event_busy`, and
`capture_event_cancelled`. Cache misses are `no_cached_depth` and
`no_cached_floorplan`.

Capture-barrier and cohort failures retain their precise codes, including
`depth_gate_transition_failed`, `depth_gate_state_mismatch`,
`mapanything_idle_failed`, `mapanything_idle_receipt_invalid`,
`depth_storage_flush_failed`, `depth_storage_frontier_unclean`,
`depth_raw_baseline_failed`, `no_raw_snapshots_for_capture_event`,
`insufficient_raw_observations`, `raw_snapshot_limit_exceeded`,
`raw_snapshot_cohort_too_wide`, `raw_snapshot_scope_mismatch`,
`raw_snapshot_precedes_baseline`, `duplicate_raw_snapshot_evidence`,
`capture_event_fusion_failed`, and `fused_snapshot_validation_failed`.
RGB-required callers may also receive `rgb_frame_unavailable`.

Exact-reload and floorplan integrity failures use
`exact_depth_load_failed`, `exact_depth_identity_mismatch`,
`exact_depth_invalid`, `floorplan_generation_failed`,
`floorplan_snapshot_integrity_failed`, `active_floorplan_contract_failed`, or
`stale_floorplan_version`. Fatal barrier/integrity codes also enter runtime
failure handling. They are never converted into an apparently successful stale
payload.

### auto_calibrate_result

Current DS9.1 runtime proxies Menon auto-calibration: `{ type: "auto_calibrate_result", ok: <bool>, updated: [<cameraId>], results: [...], error?: <string|null> }`.

## 8. Calibration Data Conventions

Same as prior DS9.1 revisions:

- **Extrinsics (`E`)**: stored in `config/camera_calibration.json`, world→camera, 4×4 column-major, meters.
- **PoseV1 (`pose`)**: stored scene pose summaries (`position`, `yaw_pitch_roll_deg`, `rotation_order=YXZ`, `frame=menon_scene`) are authoritative scene-camera poses, not raw OpenCV image-camera extrinsics. Converting PoseV1 to `E` applies a fixed local 180 degree roll so the resulting camera basis matches DS9.1 depth/image math (`+X right`, `+Y down`, `+Z forward`).
- **Intrinsics (`K`)**: from `config/cameras.yaml` (`intrinsics_models` + `cameras` map). Scaled to streammux resolution by the calibration provider in `DS9/noesis/ds9_runtime_core.py`.
- **Alignment (`align`)**: `config/ply_alignment.json` with `matrix` (room-model alignment), `floor_y`, and `units.s_obj_to_m`.
- **Calibration bundle (`calibration-bundle`)**: carries the canonical backend-world-meters pose/extrinsics plus the room-alignment metadata Menon needs for its own scene conversion. When `align.scene_similarity` is present, it includes `s_obj_to_m` and authoritative `world_to_scene_sha256`, computed over canonical JSON `{world_to_scene_col_major, s_obj_to_m}`. Consumers use that digest directly rather than reproducing Python float serialization.
- `pixel_to_world_response` returns world-frame meters; Menon applies its room alignment and scene-unit conversion client-side.
