# Handoff

## Current status

Menon Wave E is implemented and statically validated. Production person, occupancy, environment, virtual-twin surface, and room-reconstruction presentation now consume Noesis-owned canonical contracts:

- People: strict `noesis.world.snapshot` v1 only.
- Coordinates: explicit backend-world-to-Menon-scene similarity exactly once.
- Scene: strict `noesis.scene.release` v1 current payload only.
- Authored environment: verified release-owned OBJ, MTL, and complete texture inventory only.
- Camera reconstructions: all cameras from the same promoted release cohort only.

Per-camera tracking/range/clamp/smoothing logic remains available as explicitly non-production diagnostics and is never used as fallback. Per-camera occupancy is diagnostic-only; canonical entity `room_id` drives production occupancy.

## Validated evidence

- `npm test` under Node 22 -> PASS (`112/112`, 2 suites).
- `npm run build` under Node 22 -> PASS (`tsc` and Vite; existing chunk-size advisory only).
- `npm audit --omit=dev` -> PASS (`0 vulnerabilities`).
- `node scripts/validate-scene-release-consumer.mjs --release ../Noesis_Devel/data/virtual_twin/releases/home_rgbmesh_20260623T2158_v1.json --virtual-twin-root ../Noesis_Devel/data/virtual_twin` -> PASS:
  - release `home_rgbmesh_20260623T2158_v1`
  - 3 cameras
  - 27 camera artifacts
  - 5 authored dependencies (1 MTL, 4 textures)
  - 34 verified current-release requests
  - no `/virtual-twin/` route
- Focused adversarial tests cover stale/out-of-order/replayed world state, run changes, identity kinds, immediate disappearance, double transforms, cross-origin/role URL drift, mixed scene cohorts, content hash/size failures, atomic commit, and malformed authored dependency graphs.

## Live state and rollout boundary

No live Menon or Noesis process was restarted or modified during this wave. The currently running services therefore predate these contracts. Browser/live claims require the coordinated appliance cutover and promoted scene activation; static validation alone is not evidence that the old live process is isolated or rendering canonical world state.

## Open risks

- Production occupancy is intentionally empty until canonical world entities carry `room_id`; Menon will not fall back to summing per-camera occupancy.
- GLB/OBJ rendering still needs browser/GPU validation after TLS/authenticated cutover even though byte contracts, TypeScript, and production bundling pass.
- Scene promotion and live process restart must remain rollback-safe and coordinated with the gateway/TLS cutover.

## Next three actions

1. Promote the validated scene release through the scene registry after the root execution stream confirms all Noesis and gateway tests.
2. Perform the coordinated restart/cutover, then verify authenticated current-payload/dependency/artifact requests and monotonically progressing world snapshots in the browser.
3. Walk through camera overlaps and disappearances while capturing Tier 4 evidence: one global person, correct identity kind, no double transform, immediate removal, and canonical room occupancy when `room_id` is available.
