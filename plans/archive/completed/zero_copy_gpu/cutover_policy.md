# Zero-Copy DS8 Cutover Policy

## Intent
Hard-cutover DS8 runtime core to GPU/NVMM-only frame processing. Any attempt to reintroduce CPU frame-copy paths in core data flow is a blocker.

## Core vs Boundary
- Core (must be zero-copy): decode, mux, preprocess, infer, tracking, analytics, tiler, OSD, RTSP/WebRTC pipeline.
- Boundary (allowed controlled CPU serialization): WS/REST payload encoding and depth retrieval payload assembly.

## Enforcement
- Runtime tracks `zero_copy_violations` and emits violation events with location/reason.
- In production mode, any core-path violation raises and fails startup or request path.
- Deprecated migration flags and CPU fallback guards are removed from active runtime code paths.

## SLOs
- Boundary serialization budget: <= 3.0ms p99 per assembled message path.
- Core violations: 0.

## Rollback
- Rollback is code-level only (revert/patch), not environment-flag based.
- Rollback does not permit DS7 fallback.
