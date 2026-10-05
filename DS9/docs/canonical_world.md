# Canonical world and capability health

The native DS9.1 runtime uses the SDK-neutral
`CanonicalWorldService` from `noesis_core.runtime_world`. DS9 pipeline hooks
adapt frame/object metadata into versioned observations; there is no alternate
legacy world schema or fallback runtime.

## Provenance and time

The world factory fingerprints the effective model/tracker config, referenced
files, runtime config, and per-camera calibration snapshot. Missing calibration
is explicit; it is never replaced with synthetic geometry.

Public tracks include:

- positive `observed_at_us` processing time;
- `capture_time_status` (currently `estimated` because cameras are not
  synchronized);
- nonnegative stream-relative `media_pts_ns`, never represented as Unix time;
- stable identity and visitor generation where applicable.

## Position, support and continuity

The baseline producer resolves one `ground_footprint` in `backend_world_m`.
Current typed hypotheses may reference finite measured support planes bound to
the exact camera, calibration, frame revision and transform. Unknown/different
supports do not fuse into an intermediate elevation. Predictions and holds
retain the committed support and cannot establish a new floor.

Canonical entities publish `support_surface`, full XYZ/covariance,
`position_state` (`measured`, `predicted`, `held`) and
`continuity_segment_id`; source evidence preserves its own position truth.
Lifecycle presence is independent of measurement freshness. An accepted support
or authoritative continuity change breaks trails and resets Menon's delayed
interpolation. Held coordinates stay exact. None of these display consumers
becomes a second position estimator.

BEV transforms canonical XYZ and covariance into its declared basis before
dropping height. Menon applies the target frame's single validated structural
presentation, including linear velocity and A Sigma A^T covariance. Diagnostic
PCF meshes/camera markers cannot replace missing canonical evidence. See
[`universal_world_localization.md`](../../docs/universal_world_localization.md).

## Transactional publication

For each publishable frame:

1. DS9 creates canonical observations.
2. The world service commits the exact snapshot/event cohort and journal count.
3. The publication gate releases ordered `tracking`, `world_snapshot`, and
   `world_event` messages.
4. The BEV publisher attaches the exact committed tracking cohort and publishes
   afterward.

Generic WebSocket paths reject canonical message types. A failed commit or
post-admission authority error produces no partial successor and poisons that
publisher. Shutdown closes and drains the runtime publication gate before
WebSocket egress.

## Capability health

After successful publication, the monitor records progress for
`tracking_observations` and `global_world`. The authenticated endpoint is:

```text
GET /api/v1/health/capabilities
```

Unknown, stale, blocked, and failed remain distinct. A running process or open
port alone is not capability health.

## Focused validation

Run the directly affected observation/world/publication tests and, when runtime
behavior changed, one short authenticated tracking/world/BEV smoke. Do not run
the historical static-prep or release-promotion suites by default.
