# DS9.1 metadata contracts
_Status: canonical native-host metadata contract, updated 2026-08-23._

This document summarizes the key metadata structures used by the DS9.1 pipeline, both on-frame (user meta) and in downstream telemetry.

## 1. Intrinsics (Calibration Bundle + optional per-frame meta)

**Primary producer:** the calibration provider in
`DS9/noesis/ds9_runtime_core.py`, via the `calibration-bundle` WebSocket message
and `CalibrationSnapshot` for BEV/geometry consumers.

**Optional producer (DeepStream user meta):** `noesis/metadata/intrinsics.py` via `attach_intrinsics` (supported for `pyds.NvDsFrameMeta`).

### Storage

- DS9.1 canonical path (Service Maker pipelines):
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

DS9.1 code that consumes intrinsics should prefer the runtime `calibration-bundle` / `CalibrationSnapshot`. Per-frame intrinsics user meta may be absent in Service Maker pipelines.

## 1.1 Depth Registration Artifact

**Primary producer:** offline builder `scripts/build_depth_registration.py`.

**Primary consumer:** baseline DS9.1 startup + fused world estimator (`DS9/noesis/ds9_runtime_core.py`, `DS9/noesis/pipelines/hooks.py`).

This artifact is separate from `NOESIS.OBJECT_DEPTH`, `DepthResult`, floorplan caches, and MapAnything snapshot storage. It is a DS9.1-owned config artifact that maps raw DAv2 anchor range into MapAnything-aligned room range on a per-camera basis.

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
- Runtime must treat the artifact as read-only and load it before opening sources.
- In the native baseline, every enabled camera must have a valid entry whose calibration + model fingerprints match the active runtime inputs.
- The artifact corrects only the estimator-local depth observation. It must not overwrite the raw `NOESIS.OBJECT_DEPTH` payload.
- `registration_id`, `created_ts_us`, `generation_tool_version`, `fit_metrics`, `sample_counts`, and `provenance` are all part of the operational contract now and should be preserved when regenerating the bundle.

## 2. Depth Result

**Producer:** `DS9/noesis/pipelines/hooks.MapAnythingProcessor` →
`DepthStorageManager` + `DepthTelemetryPublisher`.

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

`.to_dict()` is the internal persistence representation and may contain a local
storage path. `DepthTelemetryPublisher` uses `.to_public_dict()` instead. The
wire value is a deterministic `noesis-depth://artifact/<sha256>` identifier and
never exposes a filesystem path, camera directory, or backend storage URI.
Consumers must treat it as opaque; it is not itself a fetch URL.

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

**Producers/Consumers:** `DS9/noesis/pipelines/hooks._AnalyticsTelemetryProcessor` and exclusion/BEV helpers.

### Object-Level Analytics Meta

Attached as NvDsUserMeta with type `NVIDIA.DSANALYTICSOBJ.USER_META` (or equivalent type id). The payload is cast to NvDsAnalyticsObjInfo where available, and normalized by `_extract_analytics_obj_meta` to:

```json
{
  "dirStatus": "<string|null>",
  "lcStatus": ["<line-label>", "..."],
  "ocStatus": ["<overcrowding-roi-label>", "..."],
  "roiStatus": ["<roi-filtering-label>", "..."],
  "direction_status": "<string|null>",
  "line_crossing_status": "<string|null>",
  "overcrowding_status": "<string|null>",
  "roi_status": "<string|null>"
}
```

Snake_case keys are provided for backwards compatibility with older consumers.

For household room membership, `ocStatus` is the primary per-object polygon
evidence because the established overcrowding ROIs already carry the exact room
IDs. `roiStatus` is consulted only when there is no nonempty `ocStatus`
membership. The shared DS9.1/V3DT resolver requires one exact unique label;
identical duplicates collapse, while padded, empty, over-160-character,
malformed, or multiple distinct labels fail closed. It never trims,
case-normalizes, truncates, or picks the first label.

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
  "stable_id": <int|null>,
  "tracker_id": <int>,
  "track_key": "<source:tracker:generation>",
  "tracker_lifecycle_generation": <int>,
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
  "world_quality": "good"|"estimated"|"held"|"invalid",
  "world_quality_reason": "<string|null>",
  "world_frame": "<string|null>",
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
  "world_upright_body_candidate": [<float x>, <float y>, <float z>]|null,
  "world_upright_body_height_m": <float|null>,
  "world_upright_body_scatter_m": <float|null>,
  "world_upright_body_anchor_count": <int|null>,
  "world_upright_body_strong_proof": <bool|null>,
  "world_upright_body_reacquire_support": <bool|null>,
  "world_seated_pose_candidate": [<float x>, <float y>, <float z>]|null,
  "world_seated_torso_height_m": <float|null>,
  "world_seated_stationary_lock": <bool|null>,
  "world_floor_candidate": [<float x>, <float y>, <float z>]|null,
  "world_floor_range_m": <float|null>,
  "world_floor_range_limit_m": <float|null>,
  "world_floor_incidence_sin": <float|null>,
  "world_floor_admitted": <bool|null>,
  "world_first_output_ankle_contact_supported": <bool|null>,
  "world_floor_rejection_reason": "floor_ray_range_exceeded"|"floor_ray_geometry_invalid"|null,
  "world_depth_candidate": [<float x>, <float y>, <float z>]|null,
  "world_prefilter_measurement": [<float x>, <float y>, <float z>]|null,
  "world_filter_prediction": [<float x>, <float y>, <float z>]|null,
  "world_prediction_image_foot": [<float u>, <float v>]|null,
  "world_prediction_provenance": { /* bounded non-authoritative image-motion provenance */ }|null,
  "world_measurement_accepted": <bool|null>,
  "world_rejection_reason": "<string|null>",
  "world_innovation_m": <float|null>,
  "world_innovation_limit_m": <float|null>,
  "world_reacquire_count": <int|null>,
  "world_reacquired": <bool|null>,
  "world_contact_basis": "<string|null>",
  "world_image_motion_supported": <bool|null>,
  "world_image_motion_streak": <int|null>,
  "world_state_continuity": "restored_short_ghost"|null,
  "world_fusion_policy_id": "<historical-comparator id|null>",
  "world_floor_weight_scale": <historical-comparator float|null>,
  "world_depth_weight_scale": <historical-comparator float|null>,
  "world_floor_weight_effective": <historical-comparator float|null>,
  "world_depth_weight_effective": <historical-comparator float|null>,
  "motion_mode": "walk"|"idle"|"sit"|"lie"|"unknown"|null,
  "posture": "standing"|"sitting"|"lying"|"unknown"|null,
  "trail_append_allowed": <bool|null>,
  "trail_break_required": <bool|null>,
  "trail_segment_id": <int|null>,
  "idle_jitter_m": <float|null>,
  "source_switch_count": <int|null>,
  "sticky_world_source": "<string|null>",
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
  "projection_confidence": <float|null>,
  "temporal_confidence": <float|null>,
  "reid_confidence": <float|null>,
  "reid_identity": "<string|null>",
  "appearance_id": "<string|null>",
  "embedding_present": <bool|null>,
  "embedding_sequence": <int|null>,
  "embedding_model_sha256": "<lowercase sha256|null>",
  "embedding_dimension": <int|null>,
  "pose_present": <bool|null>,
  "occluded": <bool|null>,
  "occlusion_uncertainty_m": <float|null>
}
```

These structures are not stored as user meta on frames by default but are the basis for WebSocket tracking telemetry and occupancy calculations.

`scene_prior` is the detached summary of the selected point. The universal
resolver may also evaluate the same exact-revision Scene Prior independently
for each current hypothesis. Only extent, authored boundary, observed-space
confidence, and floor elevation are soft likelihood evidence; PCF never clamps
or manufactures a track. A camera without an exact catalog binding omits the
field and candidate PCF evidence.

`world_upright_body_*` and `world_seated_*` are typed exact-cohort diagnostics
for body-to-floor projection; they are not alternate dashboard coordinates.
The upright anchor count is the number of retained anatomical planes after the
bounded outlier check. Strong proof requires five planes spanning head,
shoulders, and hips. `world_upright_body_reacquire_support=true` additionally
means the current five-plane height/scatter, detector confidence, and PCF gates
permit the ordinary physical/reacquisition path. An anchor count of three or
four never grants authority from one row. A cold lifecycle may nevertheless
publish on the second of two compatible exact-current four-plane rows within
`reacquire_max_gap_s`; both rows retain
`world_upright_body_strong_proof=false` and
`world_upright_body_reacquire_support=false`, the first is nonpublishing
corroboration, and the second supplies canonical `world`. Three-plane, mixed,
incompatible, expired, or basis-changing evidence cannot form this proof. For
an established relocation, moderate rows may accumulate same-basis trajectory
evidence while the field remains false, but only a current verified five-plane
row can finalize the reanchor.
`world_first_output_ankle_contact_supported=true` identifies the current final
sample of the bounded first-output ankle proof. Every contributing sample has
already passed tight contact, adequate incidence, detector-semantic, and
immutable-binding gates; the field is corroboration metadata, not a second
coordinate. BEV still renders only the final canonical `world` admitted for
that exact tracking cohort. If that row establishes the first queue-published
metric point, its exact tracking/world/BEV cohort publishes immediately instead
of waiting for the normal cadence.

Zero-person camera frames still produce a tracking publication with
`track_count=0`, `tracks=[]`, an empty observation set, and advancing frame
timing. The nonempty-to-empty transition is immediate; sustained emptiness is
cadenced by `NOESIS_TRACKING_EMPTY_HEARTBEAT_HZ` (2 Hz default). Canonical world
state removes that producer/source evidence while retaining its sequence
watermark so absence cannot be confused with replay or telemetry stall.

- `track_id` remains an internal tracker identifier and must not be emitted to clients.
- `tracker_id` is the bounded camera/run-local association key used by strict
  observation joins. It may be emitted for diagnostics but is not a durable or
  user-visible identity.
- `stable_id` is the numeric compatibility boundary. It is an integer `>= 1`
  for a resolved resident/visitor and may be null for an authoritative
  identity-v2 open-set unknown.
- Negative/provisional stable IDs are internal-only and must not be emitted to clients.
- A sub-second metadata gap may reuse a settled StableID without a fresh
  embedding only for the same camera and tracker-local ID, within 0.75 seconds,
  and only when strict bbox overlap, center-motion, and area-ratio gates all
  pass. The carried embedding remains internal and the current row still
  reports no fresh embedding evidence. This continuity aid cannot claim an ID
  already active in that frame and is not cross-camera ReID.
- `dwell_time` is derived per track by `_AnalyticsTelemetryProcessor` using zone entry timestamps; it is null when no zone is available.
- `bbox3d` and `velocity3d` are attached when `NVDS_OBJ_3D_META` (SV3DT/MV3DT) is present. In the locked V3DT profile they retain the tracker tuple for diagnostics; consumers must not reinterpret them directly as canonical world coordinates.
- In baseline mode, `image_foot` is the active current-frame image anchor chosen by the backend estimator and `image_base` is the canonical image reprojection of the filtered world state. In V3DT mode, `image_foot` is the native tracker ground-foot metadata, while `image_base` is the separately derived projection of the opposite cuboid endpoint. Keeping both prevents a self-referential reprojection check.
- In baseline mode, one universal resolver builds independently valid
  `floor_ray`, `registered_depth`, optional `pose_scale`, and
  `gravity_reconstruction` hypotheses in the exact current track cohort. It
  uses no camera/room strategy. Each hypothesis carries full covariance and
  evidence. Compatible floor/depth hypotheses genuinely contribute through
  covariance intersection only when every contributor is mutually compatible;
  a large statistical or absolute metric
  disagreement is never averaged and is retained as an alternate. A rejected
  floor ray never suppresses valid registered depth, and rejected/stale depth
  never suppresses an independent valid floor ray. The resolver's current
  measurement then feeds the existing `PersonGroundState`, which remains the
  sole physical filter, lock, source-hysteresis, and reacquisition authority.
  `gravity_drop`, `cv_prediction`, `image_motion_prediction`, and `anchor_hold`
  retain their explicitly degraded semantics.
- `world_quantity` is currently always `ground_footprint`. A body root or
  seated pelvis requires a future separate field. `world_covariance` is the
  resolver covariance after enlargement for any accepted
  resolver-to-filter displacement. It is absent for process-only prediction or
  hold rather than carrying stale measurement uncertainty.
- Normal public tracking metadata carries only compact `world_resolver_*`
  decision scalars. The exact candidate tree described in
  [`universal_world_localization.md`](universal_world_localization.md) is
  attached only to request-gated BEV presentation diagnostics. It is bounded
  to four candidates, contains no image, mask, tensor, SDK object, or history,
  and cannot move the BEV dot.
- `cv_prediction` means the current metric observation was missing, stale, or
  physically rejected, so the producer displayed a bounded constant-velocity
  continuation of the canonical filtered world state. It remains in the
  revision-bound `world_frame`, sets `world_measurement_accepted=false`, and
  does not advance the last-good measurement timestamp; `world_filter_prediction`
  records the displayed prediction. Rejected updates predict from one fixed
  last-good rejection anchor for no more than 0.40 seconds, so a reject run
  cannot integrate drift. `anchor_hold` means the producer retained the
  last-good world point after a missing or physically rejected measurement.
  It may remain visible as a canonical BEV head with `worldAdmission="held"`,
  but it must not append a trail sample. Its normal cap is 0.40 seconds; a
  seated/lying lifecycle may extend to 2.0 seconds only with exact-frame
  stationary bbox continuity. Prediction may append only when
  `trail_append_allowed=true`.
- `image_motion_prediction` is a separate, stricter degraded path for a
  materially moving detector box whose current metric/depth anchor is missing
  or rejected. It transports the last physically accepted image foot through
  the current bbox affine change and projects that pixel through the active
  corrected floor plane. It is non-authoritative, does not update filter or
  last-good image state, and exposes `world_prediction_image_foot` plus
  `world_prediction_provenance`; failed image/ray/metric/TTL gates fail closed
  instead of publishing a frozen point.
- `cv_prediction`, `image_motion_prediction`, and `anchor_hold` never become
  fresh global-world fusion evidence. Canonical observations require complete
  `world_frame_revision` and `world_transform_sha256` identity. Mixed target
  revisions are retained as conflict evidence rather than averaged, while
  compatible camera observations retain full correlated covariance through
  conservative covariance intersection.
- World estimator state is lifecycle-scoped, not merely numeric-tracker-ID
  scoped. An exact processed frame removes all world/filter/lock state for
  tracker keys absent from that source before the numeric ID can be reused.
- Impossible innovations are rejected rather than clipped into plausible motion.
  The filter also evaluates its proposed posterior against
  `max_speed_mps * dt`; an observation inside the wider measurement-noise gate
  is still quarantined when it would publish an unreachable same-segment step.
  A bounded run of mutually consistent observations may reacquire the same
  tracker lifecycle; that increments durable `trail_segment_id` and pulses
  `trail_break_required` so trail consumers cannot draw a teleport even if
  they miss the pulse frame. Source hysteresis is committed only after the
  physical gate accepts the observation.
- In `v3dt` mode, `world`/`world_source="bbox3d"` come from the bbox ground endpoint only after the profile's required `xzy` tracker-to-world conversion. The public result is Y-up meters in `world_frame="backend_world_m"`; absent/invalid bbox or axis metadata leaves world invalid instead of selecting a ray-plane fallback.
- The current V3DT contract does not permit `world_frame="camera_local"`. Shared-world SV3DT output does not by itself prove MV3DT overlap fusion or cross-camera ID propagation.
- `world_frame_revision` and `world_transform_sha256` identify the exact
  active calibrated/Scene Prior world edge used for the track. A renderer or
  consumer must reject a missing or mismatched identity rather than mixing the
  point with stale camera/prior geometry.
- `world_quality_reason` is the canonical diagnostic string explaining why the current update was fused, floor-only, guarded, predicted, held, or invalid. Producer-side floor-ray rejection uses the stable reasons `floor_ray_range_exceeded`, `floor_ray_geometry_invalid`, `cold_floor_semantic_confidence_below_minimum`, and `cold_floor_ray_incidence_below_minimum`; the cold reasons mean first metric authority lacked current detector semantic proof or adequately conditioned ray geometry. Tracker confidence and neutral PCF coverage cannot substitute. Downstream renderers must not clamp or reinterpret those rejected candidates as valid world positions.
- `projection_confidence`, `temporal_confidence`, `reid_confidence`,
  `reid_identity`, `appearance_id`, `occluded`, and
  `occlusion_uncertainty_m` are optional validation diagnostics. They are used
  by saved/live telemetry validation and track-audit reports when present; their
  absence means the corresponding validation evidence is incomplete, not
  implicitly passing.
- Identity-v2 does not attach Python objects or embeddings as DeepStream user
  meta. Each hook copies server-extracted SGIE values into a bounded detached
  primitive batch while walking the transient source frame once. Authoritative
  mode invokes the shared coordinator synchronously because it owns same-frame
  public identity. Shadow mode receives a separate deep scalar snapshot only
  after canonical tracking/world/BEV admission. Its isolated worker keeps the
  newest pending cohort per source, so stale shadow cohorts may be coalesced or
  dropped without changing the exact canonical cohort. Shadow results do not
  mutate or delay that cohort and are not promised on its tracking row; they
  remain comparison/evidence and bounded post-resolution OSD-cache work. A
  publication-rate-skipped frame has no shadow work, and an admitted empty
  cohort is eligible but may be superseded before scoring. Copy, capacity,
  scoring, or visitor-persistence failure degrades only shadow work. No raw
  surface, SDK object, borrowed SDK diagnostic mapping, or public embedding crosses either
  asynchronous boundary. After reconnect, the coordinator's internal tracker
  key is source-epoch scoped while the public process-local `tracker_id` stays
  raw. The exact enrollment key is derived from
  run/camera/tracker/frame identity, active engine-byte SHA-256, declared tensor
  layer/dimension, and float32 embedding bytes. Authoritative public metadata
  exposes the key and decision diagnostics, never the embedding. Shadow keys
  and decisions remain in private evidence/cache state and are not joined back
  into an already admitted canonical row.
- Persisted public embedding provenance is the all-or-none
  `embedding_sequence` / `embedding_model_sha256` / `embedding_dimension`
  triad. The sequence links to the exact durably appended private evidence row;
  the fingerprint and dimension must match the active model. The triad is
  omitted when evidence capture is disabled, persistence fails, the mode is
  shadow, or the public identity is only a tracker-continuity hold. Partial
  triads are invalid.
- `embedding_present` describes a valid server extraction for that exact
  track/frame; a legacy gallery/cache diagnostic is not frame evidence. In
  authoritative mode, `identity_observation_key` and
  `identity_v2.fresh_embedding` additionally describe live validation and
  decision truth. Async authoritative rows may omit the durable triad while
  `identity_v2.evidence_persistence` is explicitly `queued` or `dropped`;
  those rows retain live identity truth but cannot serve as durable evidence
  anchors. Synchronous persisted rows use `durable`. Shadow linkage/decisions
  remain private. Missing tensors and continuity holds set the applicable live
  fields false and clear the key/triad.
- In authoritative mode, the adapter may retain a resolved public overlay for
  a bounded tracker-continuity window when SGIE does not emit a fresh tensor on
  an intervening frame. It does not retain that row as overlap/enrollment
  evidence, marks it `fresh_embedding=false`, and removes it on the first
  camera frame where the tracker is absent. Shadow mode may retain the analogous
  decision only in its private evidence/OSD cache.
- An accepted subject is immutable for the lifetime of that camera-local
  tracker state. Other subjects become hard constraints, not score competitors;
  the locked subject still has to pass every open-set and exclusivity gate.
  Contradictory evidence therefore publishes unknown rather than a new subject,
  and only an evidence gap long enough to expire tracker state permits
  reassignment.
- Missing fresh SGIE evidence without a valid tracker-continuity hold is
  represented as `provisional`; `unknown` is reserved for an evidence-backed
  open-set rejection. In authoritative mode the metadata walk writes an
  explicit neutral OSD override (`#XX`) before whole-frame resolution so stale
  legacy StableID labels cannot leak. Detached tracking/world dictionaries are
  resolved in the same frame, but transient DeepStream wrappers are not retained
  or revisited for an unsafe second-pass OSD restamp.
- The active ReID engine path, output layer, and embedding dimension are
  explicit runtime configuration. Startup hashes the actual engine bytes and
  rejects a missing/empty engine, missing layer/dimension, topology mismatch,
  or incompatible persisted model profile instead of inferring a model.
- The strict `noesis.observation.person` envelope mirrors the complete
  embedding-provenance triad and exposes `pose_present` and `depth_present` as
  semantic booleans. `depth_present=true` requires object-depth `status="ok"`,
  `depth_registration_status="ok"`, and a finite positive
  `depth_registered_m`; `depth_used_m`, when present, must match the registered
  value. Missing, raw-passthrough, or rejected registration and raw anchors do
  not qualify.
- The strict envelope's optional `world_diagnostics` object is bounded,
  non-biometric decision evidence. It carries finite raw floor/depth candidates,
  prefilter/prediction points, floor admission and range, physical innovation,
  active fusion weights, and depth-registration status. An image-only/invalid
  world observation emitted by the canonical service always has an explicit
  `first_divergence_reason`; retaining diagnostics does not admit that candidate.
- Canonical `WorldSourceEvidence` preserves the strict `observation_id` and
  source zone. Fusion and the `WorldEntity` contract derive `room_id` only from
  agreeing accepted authoritative analytics source zones. Rejected,
  camera-default, and unprovenanced sources do not vote. One exact accepted
  label requires the matching non-null `room_id`; no label requires a null
  room; and multiple distinct labels require both a null room and visible
  conflict state.
- Noesis/Menon validation traces may serialize the track `world` vector as
  `backend_world_m` when proving world-to-BEV or world-to-Menon agreement. That
  alias is a validation/debug naming convention for the same backend-owned
  meter-space track point, not a separate metadata payload.

### Occupancy State

Occupancy per zone is represented as:

```json
{
  "<zone-id>": <int count>,
  ...
}
```

The occupancy publisher (`occupancy_publisher`) expects `publish_state(room_id, occupied, count, ts_ns)` calls aligned with this representation.

In DS9.1, `_AnalyticsTelemetryProcessor` derives diagnostic occupancy counts per
sensor from each published local track zone and publishes through
`pipeline.occupancy_publisher`, emitting vacate events when a previously
occupied zone disappears. Those local counts may include a `camera_default`
fallback; only exact authoritative analytics evidence can vote for canonical
`room_id`.

Vacates are emitted immediately when a zone’s count drops to zero; there is no grace window in the current DS9.1 implementation.

## 5. Custom User Meta Lifecycle (DS9.1 / DeepStream)

When attaching **custom user meta** (`NvDsUserMeta`) from DS9.1 hooks or native bridges, follow DeepStream’s copy/release semantics exactly to avoid heap corruption:

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

**DS9.1 note:** Service Maker Python wrappers do not expose `obj_user_meta_list` or `append` for arbitrary user meta. Use a **native bridge** (e.g., `DS9/native/noesis_pose_meta_ext.cpp`) to unwrap `ObjectMetadata` → `NvDsObjectMeta*` and call `nvds_add_user_meta_to_obj`.

## 6. Pose Feature User Meta (Object-Level)

**Producer:** `DS9/noesis/pipelines/hooks.PoseFeatureProcessor` (YOLO26 pose SGIE).

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
- See `docs/pose_stable_id_integration.md` for fusion thresholds, env flags, and memory caps.
- Attached in DS9.1 via `noesis_pose_meta_ext.attach_pose_features(...)` which calls `nvds_add_user_meta_to_obj` with the lifecycle functions defined above.

## 7. Object Depth User Meta (Object-Level)

**Producer:** the baseline DS9.1 runtime DAv2 depth-tracking lane attaches this
payload after full-frame depth is aligned once into canonical DS9.1 frame
coordinates. Mask-capable detectors use the decoded person instance mask;
box-only detectors use a bounded lower-person bbox band.

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
  "sampling_mode": "instance_mask" | "pose_capsule_native" | "bbox_diagnostic",
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
  "measurement_frame_id": 14,
  "measurement_ts_us": 1700000000000,
  "measurement_age_us": 0,
  "measurement_cached": false,
  "depth_tensor_frame_id": 14,
  "depth_tensor_ts_us": 1700000000000,
  "depth_tensor_age_frames": 0,
  "depth_tensor_age_us": 0,
  "world_point": [1.0, 0.0, 3.5],
  "world_point_depth": [1.1, 0.2, 3.6],
  "world_point_floor": [1.0, 0.0, 3.4],
  "projection_method": "depth",
  "spatial_status": "ok",
  "spatial_class": "person",
  "depth_map_ref": "noesis-depth://artifact/<sha256>",
  "ts_us": 1700000000000
}
```

**Notes:**

- The payload is attached per object only after full-frame depth has been aligned once into canonical post-mux DS9.1 frame coordinates. `instance_mask` samples decoded person support. `pose_capsule_native` derives a GPU-resident union of at most two compound contacts from attached pose metadata: each contact is a thin lower-leg segment plus a full ankle disk, with separate line and ankle radii. The host diagnostic mask and CUDA sampler use the same pixel-center predicate and read one compact statistic. Only endpoints and the two radii cross the Python/native boundary. If no ankle contact is observed it fails closed before launching depth work. It does not upload a host mask or run a second body-mask sample. A bbox-only core sample is diagnostic and cannot become person-ground authority.
- `bbox` and all mask/band/depth statistics are expressed in that canonical DS9.1 frame space. Source-native dimensions (`source_frame_width`, `source_frame_height`) are diagnostic-only and must not be used for object-depth sampling.
- If neither the native mask nor bounded bbox-band path can produce usable statistics, the aligned depth frame is not ready, or geometry is inconsistent, the payload may still be attached with a non-`"ok"` `status`.
- `status` is mandatory and distinguishes usable samples (`"ok"`) from object-local failures such as `"no_valid_depth"`, `"missing_mask"`, `"mask_decode_failed"`, `"depth_not_ready"`, or `"transform_mismatch"`.
- Version `2` adds optional person-only spatial fields derived from the segmentation mask plus the DS9.1 calibration bundle. `anchor_uv` is the bottom-of-mask image anchor in canonical frame space, `anchor_depth_m` is the preferred lower-body or torso-core depth sample, and `world_point*` fields are produced by `pixel_to_world(...)` without applying `align.matrix`.
- `anchor_sample_count` and `anchor_valid_fraction` describe the support of the specific lower-body / torso anchor band that produced `anchor_depth_m`. Baseline DS9.1 tracking weights DAv2 using these anchor-band support fields instead of whole-mask support alone so far-camera people can still contribute depth when the chosen anchor band is well supported.
- `depth_spread_m` and `anchor_depth_spread_m` are P10-P90 evidence spreads.
  `evidence_quality` and `evidence_reason` state whether support was accepted;
  high-spread, low-support, and bbox-only person evidence is rejected.
- `measurement_frame_id`, `measurement_ts_us`, `measurement_age_us`, and
  `measurement_cached` retain the original sample provenance when a bounded
  prior depth result is reused; cache reuse never rewrites an old observation
  as a current measurement.
- `depth_tensor_frame_id`, `depth_tensor_ts_us`,
  `depth_tensor_age_frames`, and `depth_tensor_age_us` identify the GPU depth
  tensor that was sampled. Tensor age is measured against the object-geometry
  measurement cohort, not against a later cache attachment. A lagged tensor is
  diagnostic-only unless the object geometry from that same tensor frame is
  available; sampling a current bbox or pose against an older tensor must not
  masquerade as exact registered-depth evidence.
- `projection_method` is one of `"depth"`, `"floor_guarded"`, `"depth_only"`, or `"floor_only"`. `spatial_status` surfaces whether that world projection is usable (`"ok"`) or why it is absent (`"geometry_unavailable"`, `"anchor_unavailable"`, `"projection_unavailable"`).
- In the baseline DS9.1 runtime, `NOESIS.OBJECT_DEPTH` is required by the fused
  world estimator. The runtime consumes the raw depth/anchor fields (`status`,
  `sample_count`, `valid_fraction`, `anchor_source`, `anchor_depth_m`,
  `anchor_sample_count`, `anchor_valid_fraction`) and computes canonical
  `track.world` inside `DS9/noesis/pipelines/hooks.py`.
- When a valid depth-registration artifact is loaded, the runtime keeps `anchor_depth_m` raw in `NOESIS.OBJECT_DEPTH`, derives a corrected `depth_registered_m` only inside the fused estimator, and projects that corrected value on the pose anchor ray before writing `track.world`.
- The object-level `world_point`, `world_point_depth`, `world_point_floor`, and `projection_method` fields remain diagnostic-only in baseline runtime mode. They are useful for inspection and parity testing, but they are not the authoritative tracking output once the fused estimator is active.
- Non-person classes keep the raw object-depth payload behavior and omit the spatial fields.
- The runtime uses a bounded internal aligned-depth cache to hand the canonical full-frame depth map from the depth-capture operator to the later object-fusion/overlay operators. That cache is prototype-internal only and is not a public DS9.1 metadata contract.
- The bridge keys frames by exact `(source_id, frame_id, media PTS)`. Object
  fusion is nonblocking by default
  (`NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS=0`, hard-clamped to 0–250 ms): it
  consumes an already-available exact frame or only a same-source, non-future
  prior frame within the configured cadence. A positive wait is available for
  explicit bounded diagnostics. Wrong-PTS, future, over-age, and wrong-source
  frames are misses rather than implicit matches.
- The bounded one-frame bridge fallback remains nonblocking and visible in
  tensor provenance. It does not become a current world-measurement candidate;
  floor geometry and the existing bounded person-state continuation handle the
  current cohort instead.
- `sampling_mode` explicitly identifies instance-mask, pose-capsule, or
  bbox-diagnostic sampling;
  consumers must not infer one mode from missing numeric fields.
- `unit` and `is_metric` travel with the payload so downstream consumers can distinguish metric meters from relative-depth fallbacks without guessing from the model name.
- `depth_map_ref` is optional and is an opaque pointer to any stored full-frame depth artifact when one exists; consumers must not parse semantics out of the string.
- Attached in DS9.1 via `noesis_depth_meta_ext.attach_object_depth(...)`; instance masks are copied out of `NvOSD_MaskParams` via the native helper `noesis_depth_meta_ext.extract_object_mask(...)` before sampling. Native attachment must receive the owning batch metadata and return success. A missing function/extension, native exception, or native rejection means the object does not carry this user meta; it is not silently counted as an attached non-`ok` payload.
- `stats.payload.pipeline.zero_copy_core.counters` makes the bridge observable.
  Exact/lagged/miss behavior uses the `depth_bridge_*` counters; successful
  attachment uses `object_depth_attach_total` and
  `object_depth_status_total.<status>`; failures use
  `object_depth_attach_failure_total` and a reason-suffixed counter. Attachment
  failures also emit a rate-limited warning.

## 8. Pose Keypoint Visualization (OSD)

Pose keypoints/skeletons are **rendered in the mosaic** using DS9.1 display metadata inserted just before `nvdsosd`:

- Config gate: `visualization.display_keypoints` in `DS9/config/infer.yaml`.
- Source: YOLO26 pose SGIE tensor meta (`unique_id` from `models.pose.gie_id`).
- Rendering: `PoseKeypointOverlayProcessor` attaches `DisplayMeta` lines/circles at the tiler stage.

This visualization does **not** add new user meta; it is purely an OSD overlay.

## 9. Calibration & Geometry

**Consumers:** `noesis/telemetry/bev.py`, calibration RPCs in `noesis/server/websocket.py`, geometry helpers.

- Calibration data (intrinsics + extrinsics + floor plane) must be consistent across:
  - `config/cameras.yaml` / calibration outputs.
  - RPCs like `set_extrinsics`, `set_align`, and `auto_calibrate_pose`.
  - BEV computations based on `CalibrationSnapshot` in `bev.py`.

The baseline world estimator depends on this calibration stack in three places
that must stay semantically aligned:

- pose-derived image anchors from `NOESIS.POSE_FEATURES` / pose keypoints,
- DAv2 range observations from `NOESIS.OBJECT_DEPTH.anchor_depth_m`,
- and world projection via `pixel_to_world(...)` in `hooks.py`.

Any change to calibration formats or semantics must be reflected here and in the corresponding DS9.1 docs, because the backend fused estimator is now the canonical owner of `track.world`.
