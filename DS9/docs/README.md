# DS9 Docs Index

This directory is the launch handoff for the DeepStream 9 Noesis app under
`DS9/`.

## Read First

1. `../README.md` - current DS9 launch status, runtime ownership boundary, build
   order, run commands, and latest validation summary.
2. `migration_state.md` - detailed migration state, validated gates, and
   remaining work.
3. `known_blockers.md` - active caveats and resolved blockers.
4. `validation_runbook.md` - commands for the next validation pass.
5. `driver_595_migration.md` - checkpointed host-driver maintenance, DS8
   acceptance, and offline rollback boundary.
6. `driver_580_rollback_rehearsal_2026-07-10.md` - exact non-mutating rollback
   evidence, corrected solver transaction, ordered recovery, and post-reboot
   DS8 acceptance contract.
7. `secondary_docker_staging.md` - isolated DS9 image, external artifact,
   canonical engine-build, capacity, and rollback workflow.
8. `runtime_container_boundary.md` - canonical isolated-container plan/run,
   mount security, profile policy, lifecycle, cleanup, and evidence contract.
9. `runtime_ownership.yaml` - machine-readable shared/adapter/duplicate
   ownership and capability-parity state.
10. `../asset_manifest.yaml` plus `asset_manifest.schema.json` - expected
   DS9-owned assets, compatibility, staging state, and provenance contract.
11. `canonical_world.md` - canonical observations, world snapshots, artifact
    fingerprints, time semantics, identity generations, and capability health.
12. `bev_capture_event_integration.md` - canonical BEV ownership, exact paired
    publication, active-floorplan authority lifecycle, raw-only capture fusion,
    GPU-first RGB boundary, and the v4 live acceptance contract.

## Current Position

DS9 runtime execution is owned by:

- `DS9/noesis/ds9_runtime.py`
- `DS9/noesis/ds9_runtime_core.py`

DS9 executable code must not import or spawn `noesis/ds8_runtime.py`, or import
DS8 preflight helpers. The static prep and ownership checks enforce that
boundary.

DS9 is not yet a hermetic standalone repository. It still consumes parent-repo
shared application context such as `config/cameras.yaml`, common calibration and
geometry helpers, WebSocket server code, and a few root validation clients. That
is an explicit packaging boundary, not a DS8 runtime fallback.

## Active Docs

- `known_blockers.md` - current blocker/caveat list.
- `migration_state.md` - detailed state and historical validation evidence.
- `validation_runbook.md` - operator validation commands.
- `driver_595_migration.md` - exact driver-only migration, DS8 regression, and
  offline 580 rollback runbook.
- `driver_580_rollback_rehearsal_2026-07-10.md` - validated offline cache,
  dependency transaction, recovery order, and the remaining reboot boundary.
- `secondary_docker_staging.md` - fail-closed secondary Docker and canonical
  plus promoted Wholebody49/V3DT engine maintenance workflow.
- `runtime_container_boundary.md` - fail-closed canonical live-canary
  supervisor, immutable/read-write mount split, exclusive ownership, and
  private evidence.
- `../DS9_REBUILD_AND_SMOKE_GATES.md` - ordered V3DT no-GPU plans,
  exclusive-GPU builds, and live bbox3d/world/identity acceptance gates.
- `MapAnything_Depth.md` - MapAnything depth behavior notes.
- `DA3Metric_Large.md` - official DA3Metric-Large export, FP16 engine,
  restart-scoped manual-depth selector, metric scaling, and validation record.
- `MapAnything_Depth_Panel_Quality_Plan.md` - measured inference, fusion,
  floorplan, 2D, and 3D quality program.
- `MapAnything_HR0_Runbook.md` - isolated FP32 `378x672` export, fixture,
  source-inspection, build, and benchmark workflow.
- `canonical_world.md` - canonical world and capability-health adapter contract.
- `bev_capture_event_integration.md` - shared BEV/fusion implementation and v4
  live-promotion acceptance boundary.
- `Static_ROI_Exclusion.md` - ROI exclusion behavior notes.

## Historical Reference

`history/ds8/` contains copied DS8-era contract/reference documents and the
DS8 design-decision ledger. Keep them only as migration evidence until a later
docs pass rewrites the remaining contracts into DS9-native names.

Do not use historical DS8 docs as permission to route DS9 execution through DS8
runtime paths, DS8 engines, or DS8 native extension binaries.
