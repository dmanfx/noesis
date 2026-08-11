# DS9 Canonical World And Capability Health

DS9 uses the same SDK-neutral world owner as DS8. The runtime constructs
`CanonicalWorldService` through `noesis_core.runtime_world.create_runtime_world_service`
and passes it to the DS9 copy of `TrackingTelemetryPublisher`. There is no DS8
runtime import and no alternate DS9 world schema.

## Provenance Inputs

The factory receives the already-materialized `pipeline.config`, not the base
template. Its model fingerprint covers the selected `models` and `tracker`
configuration and hashes referenced files that actually exist. Its runtime
configuration fingerprint covers the effective configuration. Calibration is
fingerprinted per source from the active calibration-provider snapshot; an
unavailable snapshot is represented explicitly rather than replaced with a
synthetic calibration.

These are semantic provenance fingerprints for canonical observations. They do
not replace the DS9 build-provenance and checksum requirements in
`DS9/asset_manifest.yaml`.

## Public Track Evidence

Both DS9 public-track paths add:

- `observed_at_us`: positive wall-clock time when the frame was processed;
- `capture_time_status: estimated`: the cameras are not yet clock-synchronized;
- `media_pts_ns`: the nonnegative, stream-relative DeepStream buffer PTS;
- stable identity fields from shared diagnostics, including
  `visitor_generation` when the subject is a visitor.

Media PTS is never presented as Unix time. Visitor generation is part of the
canonical visitor entity ID, preventing a recycled bounded visitor SID from
aliasing an earlier person.

## Publications And Health

Each tracking publication retains the existing `type: tracking` envelope and
adds versioned person observations plus the current `noesis.world.snapshot`.
The snapshot is also broadcast as its own `type: world_snapshot` message.
The complete tracking/snapshot/event cohort is frozen and count/byte-admitted
first, but client delivery remains behind a one-shot gate. The shared world
owner synchronously confirms the exact integrity-journal append count, commits
private fusion authority, and only then releases the batch. Commit failure
aborts with zero delivery and poisons the publisher; accepting an asynchronous
journal queue item is not sufficient authority. Read consumers receive only
the immutable cached committed snapshot and cannot mutate fusion state.
The WebSocket boundary accepts this route only as a tracking-first exact cohort;
generic broadcast paths reject tracking, world snapshot/event, and BEV types.
`bev-frame` has a separate dedicated typed-receipt route after tracking commit.
Because the release callback is in-process rather than cryptographic, a static
regression restricts production gated-API call sites to the byte-identical
DS8/DS9 tracking publishers and the WebSocket definition.

After successful canonical publication, the capability monitor records producer
progress for `tracking_observations` and `global_world`. The shared REST router
exposes the resulting contract at:

```text
GET /api/v1/health/capabilities
```

Unknown, stale, failed, and blocked states remain explicit. A listening port or
running process is not sufficient evidence of capability health.

The DS9 REST app also mounts the shared `/api/v1/scenes` router. Atomic scene
release registration, promotion, rollback, current payload, and history remain
single-owner product state rather than an SDK-specific world-model fork.

## Validation

These checks are safe on the current DS8 host and do not load DeepStream:

```bash
python3 -m unittest \
  DS9.tests.test_world_contract_adapter \
  DS9.tests.test_world_snapshot_runtime -v
python3 DS9/scripts/validate_runtime_ownership.py
bash DS9/scripts/run_static_prep_checks.sh
```

Fresh DS9 runtime evidence remains blocked by the missing DS9-native artifact
set recorded in the manifest. Do not treat the focused contract tests as a
runtime cutover result.
