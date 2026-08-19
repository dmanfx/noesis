# Noesis upgrade history

This is the concise operational record of major runtime and behavior changes.
Detailed work orders, evidence, and superseded diagrams remain in the archives.

## 2026-08-18 — Kitchen and Family Room SV3DT profiles reached per-room parity

- Accepted the Kitchen phone-walk Scene Prior as per-room geometry and retained
  its existing reviewed V3DT camera orientation.
- Added a V3DT-only analytics bundle that removes Kitchen portraits and Family
  Room portrait, TV/reflection, and window detections before tracking while
  leaving baseline analytics untouched.
- Made the V3DT household confirmation threshold profile-controlled and used a
  one-embedding threshold in the Kitchen and Family Room profiles; the
  non-V3DT identity policy is unchanged.
- Revalidated repeated July one-person replays with zero tracks in inactive
  rooms, complete BBox3D/world output, and primary StableID reuse for 24/25
  Kitchen and 46/47 Family Room raw tracklets.
- Applied and visually checked the corrected feet-anchored cuboid in both room
  profiles at multiple image positions.
- Confirmed a short synchronized Kitchen/Family doorway overlap, but kept
  MV3DT disabled because the latest common three-room registration remains
  rejected and review-only.

## 2026-08-15 — Documentation authority reconciliation

- Rebuilt the current index, architecture description, native baseline,
  focused testing guide, decisions, and application diagrams from the live
  native DS9.1 graph.
- Renamed active API/metadata documents to runtime-neutral names and corrected
  the dashboard, Scene Prior/PCF, depth, identity, ROI, media, and integration
  descriptions.
- Moved DS8, DS9.0, container, completed migration, bridge-audit, and inactive
  prototype guidance into explicit non-normative archives.
- Replaced stale active workstream instructions with current native DS9.1
  checklists while retaining the detailed historical worklogs.
- Updated DeepStream development/profiling/MV3DT agent routing so Noesis work
  selects native DS9.1, direct validation, and the current geometry deferral.
- Corrected operator examples to discover and load the service's installed
  `native.env` before using native-root variables.

## 2026-08-15 — Canonical native-host DS9.1 runtime

- Made `DS9/scripts/run_canonical_runtime_host.py` the runtime supervisor.
- Bound DeepStream 9.1.0, CUDA 13.2, TensorRT 10.16.0.72, GStreamer 1.24.2,
  Python 3.12, native plugins/extensions, secrets, and the accepted engine
  realization without Docker.
- Made engine maintenance native-host by default and rejected Docker authority
  in the host path.
- Preserved the accepted three-camera graph and restored package/import
  environment parity.
- Recorded in commit `3f1cae4`.

## 2026-08-15 — PCF BEV tracking authority repair

- Paired canonical committed tracking cohorts with PCF-backed BEV frames so
  dashboard dots and trails appear on Scene Prior floorplans.
- Preserved empty-occupancy behavior and did not let the static PCF source mint
  tracking authority.
- Focused tests and a short live BEV capture verified all three cameras.
- Recorded in commit `59aae04`.

## 2026-08-13 — PCF became the depth-panel presentation source

- The depth drawer now opens against admitted immutable Scene Prior evidence.
- Full reconstruction extent, authored room footprint, diagnostics, and
  floorplan presentation were separated explicitly.
- Dashboard bundle and contract updates are recorded by `60d2226` and
  `5d77c65`.

## 2026-08-12 — DeepStream 9.0 to 9.1 direct upgrade

- Rebuilt the selected runtime image/toolchain authority, seven native
  extensions, seven inference parsers, one TensorRT plugin, three GStreamer
  plugins, and ten selected engine realizations for CUDA 13.2 / TensorRT
  10.16.0.72.
- Rebound DAv2 and MapAnything depth registrations to exact DS9.1 engine/config
  content while retaining accepted calibration knots and fusion policy.
- Restored the baseline WebRTC-only media path and runtime publication gate.
- Removed identity-retention work from the per-frame hot path, restoring the
  live three-camera rate to roughly 30 FPS.
- Core commits: `6e69f95`, `9d5f275`, `5a6c93c`, `b787db8`, `4c87067`, and
  `4c19515`.

## 2026-08-10 to 2026-08-12 — DS9 application parity and hardening

- Ported the shared product surface to the DS9 adapter: source progress,
  analytics, tracking/world publication, identity, pose, ReID, depth, ROI
  exclusion, media, lifecycle, secrets, artifact authority, and direct live
  gates.
- Added deferred MV3DT assets and contracts without enabling the capability.
- Container/candidate mechanics used during this transition are retired and
  archived; they are not the current development workflow.

## Earlier — DS8 application foundation

The DS8 generation introduced the Service Maker pipeline, GPU-first memory
contract, MapAnything/DAv2 depth roles, StableID, canonical telemetry, BEV,
WebRTC mosaic, and many shared services still used by DS9.1. DS8 itself is no
longer executable authority. Its plans and decisions are retained under
`docs/history/ds8/` and `plans/archive/ds8/` to explain the origin of those
contracts.

## Earlier — DS7 and pre-Service-Maker application

GI/GStreamer pad-probe and older runtime material is retained only under
`docs/history/ds7/` and `plans/archive/legacy_prototypes/`.
