# Plan: BEV World-Space Alignment (oai2-fe, noesis)

## Scope and objective
This plan delivers a safe migration for BEV/floorplan visualization so it matches the frontend-tracked Menon world/scene coordinate space.

- Outcome: BEV/floorplan and tracking dots/trails are aligned in world/scene units, with frontend-owned trail smoothing/persistence.

## Summary
Primary source of truth for behavior in this migration is the BEV payload’s frame metadata and BEV track payload validity, not ad-hoc inference.

This migration is implemented as **world-first** and does not ship a dashboard coordinate-space toggle (by design).

## Decision principles
- Default behavior after this work: **world-first** for BEV/tracking alignment.
- Never mix frame assumptions inside the same live render path.
- Favor explicit frame metadata over heuristic branch guessing.
- Legacy camera-local behavior (if needed) is controlled via backend config/env (`NOESIS_BEV_FRAME=camera_local`) and is intentionally not an operator-facing UI toggle.

## Current-state diagnosis captured in this plan
- `oai2-fe/src/App.tsx` already prefers `track.world` when `world_valid` and projects it via camera transform.
- `oai2-fe/src/components/BevView.tsx` consumes `bev-frame.meta.footpoints` and floorplan bounds; it is not yet explicitly tied to frame mode.
- No dashboard control currently exists for world-vs-backend legacy mode.
- Backend publishes frame metadata (`frame`, `world_frame`, `frame_mode`) in BEV-related payloads; floorplan payload still needs strict frame handling to prevent mismatch.

---

## Phase 1 – Align BEV to frontend world space

### 1) Define a frontend coordinate-mode abstraction
- **Files:** `oai2-fe/src/hooks/useWebSocketClient.ts`, `oai2-fe/src/App.tsx`
- Add an internal mode enum (or typed union) such as:
  - `coordSpace: "world" | "camera_local_legacy"`
- Derive mode from message metadata:
  - `world` if `bev-frame.frame === "world"` OR `world_frame === "menon_scene"`
  - otherwise `camera_local_legacy`.

### 2) Centralize trail/world conversion with explicit mode branch
- **File:** `oai2-fe/src/App.tsx`
- Replace scattered conversion logic with a single conversion helper used by both live trail ingestion and rendering model updates.
- In `world` mode:
  - require `track.world_valid`
  - project world point through existing world-to-camera transform and map to BEV plane consistently
- In `camera_local_legacy` mode:
  - use established legacy fallback path (tracked center / existing non-world conversion).
- Preserve mode-change trail reset logic (do a one-time flush when mode transitions) to avoid mixed-frame ghosts.

### 3) Make BEV rendering mode-aware
- **File:** `oai2-fe/src/components/BevView.tsx`
- Add explicit props/state for:
  - `coordSpace`
  - `frame` / `world_frame` / source metadata
- In `world` mode, render BEV points + bounds against world-space interpretation.
- In `camera_local_legacy`, render legacy mapping behavior.
- Keep unknown metadata fallback to legacy to preserve compatibility.

### 4) Address floorplan frame correctness path in backend/frontend handoff
- **Files:** `geometry/depth_source.py`, `noesis/telemetry/bev.py`, optionally `noesis/ds8_runtime.py`
- Ensure floorplan payload includes explicit frame and coordinate meaning.
- Guarantee that when BEV is in world mode, the floorplan input used for scaling/origin is aligned to that same world frame; if only camera-local floorplan exists, add a deterministic transform or explicit compatibility handling.

### 5) Strengthen backend/telemetry consistency
- **Files:** `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`
- Verify `frame` and `world_frame` output in BEV are stable and explicit per frame.
- Ensure no assumptions of implicit frame when `world_valid` is false or absent.
- Keep graceful fallback behavior if world data is missing.

### 6) Add/extend parity safety check
- **File:** `scripts/menon_bev_track_parity_smoke_test.py`
- Add assertion that world-mode stable IDs are near-consistent between:
  - `track.world`
  - `bev-frame.footpoints`
- Gate check on world mode + world validity.

### 7) Deliverables for Phase 1
- One stable world-mode pipeline from websocket payloads to displayed BEV.
- Minimal behavior change for operators (feature remains visually correct in world mode by default).
- Backward-compatible behavior if metadata is missing.

---

## Phase 2 – Dashboard Coordinate Toggle
_Removed: current implementation intentionally does not provide a dashboard coordinate-space toggle. The only supported operator path is world-first rendering in scene units._

If legacy camera-local BEV is needed for debugging, it should be enabled via backend config/env (`NOESIS_BEV_FRAME=camera_local`) and treated as a restart/config-change workflow, not a UI toggle.

---

## Test matrix

1. **World baseline**
   - valid `track.world` stream
   - BEV points/trails stay in-bounds and follow motion without multi-second lag

2. **Missing world data**
   - `world_valid=false` or missing `world`
   - behavior stays stable (no crashes); explicit “missing data” telemetry is surfaced
