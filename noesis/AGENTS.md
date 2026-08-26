# AGENTS.md — Shared Noesis application modules

`noesis/` contains SDK-neutral services and product contracts consumed by the
canonical DS9.1 runtime. It is not an independently runnable DS8 stack.

## Policy precedence

- Root `AGENTS.md` applies.
- A nearer `AGENTS.md` may add narrower rules.
- Historical DS8 descriptions under `docs/history/` are non-normative.
- Work in this subtree belongs on the repository `DS9` branch. Treat
  `feature_DS8` as read-only and verify the branch before committing.

## Scope

- `server/`: REST application services and boundary metrics.
- `telemetry/`: tracking, depth, BEV, latency, and world publication.
- `metadata/`: public metadata schemas and adapters.
- identity, calibration, world, scene, and persistence modules shared by DS9.1.

The executable adapter lives under `DS9/noesis/`. Immutable model-artifact and
persisted replay identifiers may retain DS8-era names for read compatibility;
those labels do not confer runtime authority.

## Rules

1. Preserve SDK-neutral public contracts unless the task explicitly changes
   them; test the producer and direct consumer together.
2. Keep GPU/native work in the DS9.1 adapter or native bridge. Do not add raw
   frame CPU branches or analytics appsinks to shared modules.
3. Shared telemetry, journal, identity, persistence, and integration code must
   not perform unbounded work or durable/network I/O on the media callback.
   Use bounded workers and reuse already-extracted compact metadata rather than
   repeating tensor, NumPy, or JSON conversion.
4. No hidden fallback algorithms, alternate metadata sources, or legacy runtime
   routes.
5. Maintain canonical publication ordering and lifecycle barriers for tracking,
   world, BEV, depth, and WebSocket output.
6. For mirrored files, modify the DS9.1-owned copy required by
   `DS9/docs/runtime_ownership.yaml`; update the shared copy only when the
   contract is intentionally shared.
7. Validate ordinary changes with focused tests and the affected direct
   consumer. Do not invoke appliance staging or promotion.
8. Use portable paths and private-file/environment authorities for runtime
   state and credentials.
