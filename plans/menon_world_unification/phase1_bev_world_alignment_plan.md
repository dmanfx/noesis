# BEV World-Space Alignment Plan (Noesis/oai2-fe)

## Goal
Align dashboard BEV/floorplan rendering and tracking trails to frontend world-space coordinates first, then add a controlled fallback path back to the previous backend world/camera-local mode in a follow-up phase.

## Phase 1 — Frontend world-space BEV alignment (implemented)

- [x] Update websocket track schema to accept world coordinates.
  - File: `oai2-fe/src/hooks/useWebSocketClient.ts`
  - Notes:
    - Added `Track.world?: [number, number, number]`
    - Added `Track.world_valid?: boolean`
    - Added `Track.world_frame?: string`

- [x] Update BEV metadata shape to expose coordinate-frame hints.
  - File: `oai2-fe/src/components/BevView.tsx`
  - Notes:
    - Added `BevMeta.frame`, `BevMeta.world_frame`, `BevMeta.frame_mode`
    - Added `BevFrameMode = 'world' | 'camera_local_legacy'`

- [x] Add frontend mode switching state from BEV metadata and switch trail ingestion.
  - File: `oai2-fe/src/App.tsx`
  - Notes:
    - Added per-camera `bevFrameModeByCam`/`bevFrameModeByCamRef`.
    - Added `resolveBevFrameMode` (uses `frame_mode`, `frame`, and `world_frame` hints).
    - In stats processing, world mode now uses `track.world.[x,z]` directly for trail points.
    - Legacy path conversion via camera intrinsics/extrinsics for per-camera local mode is no longer used in BEV path.

- [x] Ensure floorplan display frame-compatibility.
  - File: `oai2-fe/src/components/BevView.tsx`
  - Notes:
    - Floorplan rendering and bounds selection now check frame compatibility against current BEV `coordMode`.
    - Incompatible or missing-frame floorplans are treated as non-applicable and fallback defaults are used.
    - Added optional `frame` support in floorplan response typing via `oai2-fe/src/components/DepthDrawer.tsx`.

- [x] Wire BEV component into selected mode.
  - File: `oai2-fe/src/App.tsx`
  - Notes:
    - Passes `coordMode={bevFrameModeByCam[cam]}` into `BevView`.

## Phase 2 — Optional backend-frame toggle (next)

- [ ] Add explicit UI toggle in dashboard for each camera (or global) to override BEV coordinate interpretation.
- [ ] Add user/session persistence for toggle choice (localStorage + startup restore).
- [ ] Ensure toggle drives both tracking-point ingestion (`App.tsx`) and BEV floorplan compatibility (`BevView.tsx`) consistently.
- [ ] Document fallback behavior when floorplan frame metadata is missing/unknown for either mode.
- [ ] Add lightweight verification script or smoke check to compare:
  - `world` mode uses BEV `track.world` and world-mode floorplan only.
  - `camera_local_legacy` mode preserves previous behavior.

## Validation notes

- Completed checks (Phase 1):
  - UI components compile-path reviewed for new metadata/frame-mode flow.
  - Type additions were propagated where BEV and depth-floorplan payloads are consumed.
- Suggested follow-up checks:
  - Confirm BEV `frame_mode` / `frame` / `world_frame` values in live stream for each camera.
  - Validate floorplan behavior when frame metadata is missing vs. explicit camera-local.
