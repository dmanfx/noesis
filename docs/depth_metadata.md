# Depth Metadata (DS8)
_Status: validated against shared DS8/DS9 depth contracts on 2026-07-11._

DS8 has two distinct depth paths. They share calibration and world-estimation
semantics, but they do not share a public payload or runtime cadence.

## Gated dense MapAnything depth

- `MapAnythingProcessor` produces `DepthResult`, dense stored snapshots,
  `type:"depth_result"`, `ma_depth_response`, and floorplan inputs.
- `depth_map_ref` on public telemetry is an opaque
  `noesis-depth://artifact/<sha256>` identifier. It is not a path or fetch URL.
- `DepthStorageManager` owns the private snapshot layout and optional normals.
  Clients retrieve bounded RPC payloads rather than parsing storage paths.
- The MapAnything valve is on demand. Its `depth_enabled`/`depth_fps` stats do
  not describe the baseline tracking-depth lane.

## Always-on baseline DAv2 tracking depth

- The full-frame DAv2 branch captures aligned device depth and rendezvouses with
  the later object-fusion stage by exact `(source_id, frame_id, media PTS)`.
- Fusion waits for `NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS` (20 ms by default,
  clamped to 0–250 ms). After timeout it may use only a same-source, non-future
  prior frame within the configured depth cadence.
- Person masks, or a bounded lower-person bbox band for box-only detectors, are
  sampled into `NOESIS.OBJECT_DEPTH` object user meta. The raw payload declares
  `sampling_mode` and retains `status`, support, anchor, and metric-depth
  fields; the backend may apply the read-only DAv2→MapAnything registration
  before using depth in canonical world estimation.
- DAv2 does not publish another full-frame `depth_result`. Its public effects
  are the per-track depth diagnostics and strict
  `noesis.observation.person.payload.depth_present`.

`depth_present` is true only when object-depth `status` is `ok` and a finite,
positive usable depth exists. Registration/transform rejection makes it false.
When `depth_registration_status` is `ok`, raw `depth_anchor_m` alone is not
enough; `depth_registered_m` or `depth_used_m` must be usable.

## Failure and health evidence

The stats payload exposes the DAv2 bridge under
`pipeline.zero_copy_core.counters`:

- `depth_bridge_put_total`, `depth_bridge_exact_resolve_total`,
  `depth_bridge_wait_total`, and `depth_bridge_wait_timeout_total`
- `depth_bridge_lagged_resolve_total`, `depth_bridge_lagged_age_frames_total`,
  `depth_bridge_lagged_age_us_total`, and `depth_bridge_miss_total`
- `depth_tracking_device_frames_total` and `object_depth_gpu_roi_copies_total`
- `object_depth_attach_total`, `object_depth_status_total.<status>`,
  `object_depth_attach_failure_total`, and
  `object_depth_attach_failure_total.<reason>`

A native attachment failure means no `NOESIS.OBJECT_DEPTH` meta was attached.
It increments failure counters and a rate-limited warning; it must not increment
the successful attachment/status counters or be reinterpreted as usable depth.

Calibration is distributed via `calibration-bundle` and cached on the storage
manager so MapAnything floorplans, DAv2 registration, and depth-to-world
projections use one authority. See `DS8_metadata_contracts.md` for the full
`DepthResult`, registration, and `NOESIS.OBJECT_DEPTH` schemas.

The older per-object `NOESIS.MDE` user-meta format is archived under `docs/history/` for reference.
