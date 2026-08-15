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
