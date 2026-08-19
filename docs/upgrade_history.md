# Noesis upgrade history

This is the concise operational record of major runtime and behavior changes.
Detailed work orders, evidence, and superseded diagrams remain in the archives.

## 2026-08-19 — Kitchen/Family MV3DT promoted to explicit opt-in

- Replaced the rejected room transform with independent Kitchen and Family
  Room static-camera anchors and accepted their binding after direct dynamic
  handoff validation.
- Fixed the DS9.1 communicator startup race with one shared, non-threaded MQTT
  connection, so all three streams are online before the first batch.
- Tuned the Kitchen/Family-only object model and association gates for the
  short occluded doorway overlap, including two-frame late reassociation.
- Preserved MV3DT's batch-global native ID through the product StableID layer;
  baseline and SV3DT remain camera-scoped and unchanged.
- Replayed the complete July single-person set, two May multi-person spans, and
  a targeted difficult partial-body interval. All three July doorway episodes
  handed off, visually confirmed same-person multi-view pairs fused, and no
  separate-person or partial-body duplicate track was falsely merged.
- Captured the rendered three-camera mosaic and checked the cuboid anchor at
  four image positions. Its bottom-face centroid remained on the feet/gravity
  point.
- Made `--tracking-mode mv3dt` select the accepted profile without an evaluation
  environment flag through both the DS9 runtime and native-host supervisor.
  MV3DT remains opt-in; omitting the selector still launches baseline.
- Removed strict streammux timestamp synchronization for the non-PTP live RTSP
  set while retaining common batch frame IDs. A native-host live smoke brought
  up all three communicators, WebRTC, REST, and WebSocket, completed 412
  publication callbacks, and delivered 137 encoded mosaic frames with zero
  drops during the bounded active interval.

## 2026-08-19 — Kitchen/Family MV3DT evaluation lane

- Added an explicitly gated, evaluation-only Kitchen/Family MV3DT profile while
  leaving the canonical appliance and per-room SV3DT profiles unchanged.
- Bound the review Kitchen transform into the Family gauge and limited real
  peer exchange to Kitchen and Family Room. A Living self-topic loop avoids a
  DS9.1 empty-peer batch deadlock without creating a cross-camera edge.
- Made recorded replay use complete batches and shared batch frame IDs.
- Materialized MV3DT MQTT and analytics state outside the checkout and baseline
  writable state, restoring the V3DT-specific portrait/reflection exclusions.
- Replayed a complete July single-person cycle and a complete 120-second
  multi-person segment. Both advanced at application rate with complete 3D
  metadata; visual samples confirmed cuboid bases remain under the feet.
- Kept the lane review-only: neither cohort proves a Kitchen/Family StableID
  handoff, and the bound geometry still fails its held-out acceptance gates.
  The remaining input is a synchronized occupied shared-FOV Kitchen/Family
  capture tied to accepted connector geometry.

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

## 2026-08-19 — Live MV3DT complete-batch repair

- Changed only the explicit MV3DT profile to wait for a complete three-camera
  streammux batch; baseline remains unchanged.
- Bound the asset validator and focused tests to the complete-batch invariant
  required by the ordered MV3DT peer-message synchronizer.
- Corrected the prior 137-frame live smoke classification: it proved startup,
  not sustained progress, whereas the synchronized July replay completed 7,110
  source-frame publications and normal EOS.
- A probe-free live run sustained approximately 30 FPS per camera for 148
  seconds after first output, delivered 4,480 encoded mosaic frames without a
  drop, completed 13,443 publication callbacks, and accepted orderly EOS.

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
