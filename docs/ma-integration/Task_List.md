# MapAnything Integration Task List

**Version:** 1.1  
**Date:** September 29, 2025  
**Author:** AI Coding Assistant  

This task list is designed for the coding agent to implement the integration systematically. Tasks are atomic, with dependencies, assignees (agent), and effort estimates (hours). Use todo_write tool to track progress: mark in_progress for current, completed on finish. Prioritize Phase 0-1 first. Menon tasks: cd ~/Menon; edits there, test vite dev.

## Overall Guidelines
- Commit per phase; branch `feature/ma-integration` (Noesis); `feature/ma-ma-integration` (Menon if separate).
- Lint/test after each task; fix errors immediately (max 3 loops).
- No user decisions: follow plan exactly.
- Track: Use todo_write with id=phase.task_num, status=pending -> in_progress -> completed.
- Cross-Project: For Menon, use absolute paths; test WS end-to-end.

## Phase 0: Preparation (Effort: 4h) ✅ COMPLETED

**Summary:** All dependencies installed, Apache model weights downloaded and verified, config created, storage directories set up. Committed as `f6a7291`.

1. **Update requirements.txt and install deps** (dep: none, 1h) ✅
   - ✅ Appended: map-anything (git), uniception, huggingface_hub, zarr, numcodecs, fastapi, uvicorn[standard]
   - ✅ Installed manually: All deps including hydra-core, omegaconf, opencv-python-headless, safetensors, trimesh
   - ✅ Verified: `import mapanything` successful
   - ✅ Installed editable: `pip install -e external/map-anything --no-deps`
   **Status: Completed**

2. **Set env and download weights** (dep: 1, 1h) ✅
   - ✅ Set: `MAPANYTHING_WEIGHTS=/home/mayor/.cache/huggingface/hub` (added to .env)
   - ✅ Downloaded: `facebook/map-anything-apache` (1.13GB, snapshot 09928f5)
   - ✅ Computed SHA: `c5606f982ca7e70ca94deb152107cc105b1db5f45316d25a2f84e2ee212be2f5`
   - ✅ Verified: Model reloads from cache in ~20s (includes DINOv2 torch hub loading)
   **Status: Completed**

3. **Create config/mapanything.ini** (dep: none, 1h) ✅
   - ✅ Created: `config/mapanything.ini` with [service], [inference], [performance], [storage] sections
   - ✅ Extended: `config.py` with `load_ma_config()` function (parses INI, returns dict)
   - ✅ Tested: `load_ma_config()['inference']['model_id']` returns `facebook/map-anything-apache`
   **Status: Completed**

4. **Cleanup** (dep: 1-3, 1h) ✅
   - ✅ Created: `data/depth/`, `data/calib/`, `data/.gitkeep`
   - ✅ Committed: `f6a7291 feat: prep ma env - add deps, config, weights, storage dir` (5 files, 66 insertions)
   - ✅ Phase marked complete
   **Status: Completed**

## Phase 0.5: Menon Inspection (Effort: 1h) ✅ COMPLETED
1. **Review Menon codebase** (dep: phase0, 1h) — ✅ Completed
   - cd /home/mayor/Menon; ls src/ (confirm structure: services/WebSocketClient.js, ui/components/SettingsPanel.js, features/occupancy/OccupancyVisualizer.js, three/Overlays.ts).
   - Verified WebSocket handlers (`calibration-bundle`, `pixel_to_world`).
   - Confirmed no existing depth visualization code (clean slate for Phase 4a).
   - Notes recorded in `docs/ma-integration/menon_inspection_notes.md`.

## Phase 1: Microservice (Effort: 12h) ✅ COMPLETED
1. **Create dir and server.py skeleton** (dep: phase0, 2h) — ✅ Completed
   - Scaffolded `services/mapanything_svc/server.py` with FastAPI app and models directory.
   - Verified swagger UI reachable via local uvicorn run.

2. **Implement get_model and auth** (dep: 1, 2h) — ✅ Completed
   - Added secure API key dependency and BF16-capable lazy loader with Apache-only guard.
   - Health endpoint exercised with/without header.

3. **Implement infer_mono endpoint** (dep: 2, 3h) — ✅ Completed
   - Implemented full inference path with retries, AMP, dynamic memory mode, and JSON serialization.
   - Synthetic frame tests return expected depth/conf/mask shapes.

4. **Implement infer_multi endpoint** (dep: 3, 3h) — ✅ Completed
   - Added per-scene async queue, multi-view outputs (poses, intrinsics, scale) and mask handling.
   - Manual curl confirms multi-camera dicts populated.

5. **Add run.sh and integrate startup** (dep: 4, 2h) — ✅ Completed
   - Created launch script (now using `python3 -m uvicorn`) and wired auto-spawn in `main.py` with health polling.
   - Service starts automatically with main application.

6. **Perf tune** (dep: 5, integrated) — ✅ Completed
   - Enabled `PYTORCH_CUDA_ALLOC_CONF`, TF32, and runtime memory-efficiency switching based on VRAM usage.

## Phase 2: Adapters & Core (Effort: 16h) ✅ COMPLETED
1. **Create adapters/mapanything_adapter.py** (dep: phase1, 3h) — ✅ Completed
   - Added RGB conversion, intrinsics scaling, CanvasTexture-friendly payload encoding, and view builders for mono/multi batches.

2. **Create geometry/depth_source.py** (dep: 1, 4h) — ✅ Completed
   - Implemented FastAPI client with exponential backoff, Zarr persistence, depth summary stats, and optional latest-depth loader for RPCs.
   - Integrated scheduling and storage with main application pipeline.

3. **Create geometry/floor.py** (dep: 2, 4h) — ✅ Completed
   - Delivered backprojection, RANSAC plane fitting, and world transform utilities feeding pixel-to-world upgrades.

4. **Zarr store integration** (dep: 3, 3h) — ✅ Completed
   - Depth outputs persisted under `data/depth/<cam>/<date>/<hour>/<ts>.zarr` with Blosc compression and metadata for quick retrieval.

5. **Calib update from multi** (dep: 4, 2h) — ✅ Completed
   - Multi-view responses feed metric scale, pose confidence, and calibration bundle updates with versioned JSON snapshots and WS rebroadcast.

## Phase 3: Contracts & Storage (Effort: 8h) ✅ COMPLETED
1. **MQTT depth_summary** (dep: phase2, 2h) — ✅ Completed
   - Depth summaries published on `noesis/geometry/<room>/<cam>/depth_summary` with median/p10/p90/conf stats.

2. **InfluxDB metrics** (dep: 1, 3h) — ✅ Completed
   - Added `mde.depth.summary`, `mde.scale`, and `mde.pose.error` series for long-term analytics.

3. **Calib JSON persist** (dep: 2, 3h) — ✅ Completed
   - Calibration bundle versioned under `data/calib/` with scale and pose confidence and WS rebroadcast on update.

## Phase 4: UI Hooks (Effort: 20h)
### 4a: Menon Integration (Effort: 10h) ✅ COMPLETED
1. **Extend WebSocketClient.js** — ✅ Completed (MapAnything diagnostics/depth routing, legacy-safe RPC helper, enriched pixel-to-world cache).
2. **Add Toggles to SettingsPanel.js** — ✅ Completed (persistent heatmap/confidence/status switches broadcasting via event bus).
3. **Depth Visuals in OccupancyVisualizer.js** — ✅ Completed (viridis heatmap planes with confidence alpha and per-room scheduling).
4. **Overlays & Coordinate Transform Enhancements** — ✅ Completed (HUD status indicator plus confidence-aware screenToWorld fallback logic).

### 4b: oai2-fe Diagnostics (Effort: 10h) ✅ COMPLETED
1. **WS RPC Enhancements** — ✅ Completed (`get_ma_depth` endpoint, diagnostics broadcasts, pixel-to-world confidence/method fields).
2. **DepthDrawer.tsx** — ✅ Completed (drawer UI with tabs, camera selector, auto-refresh hooks, and styling).
3. **Heatmap Viewer** — ✅ Completed (base64 decoding into canvas viridis heatmap with hover metadata).
4. **Stats & Histogram** — ✅ Completed (summary cards, confidence histogram, and metric bars).
5. **Metrics & Gauges** — ✅ Completed (baseline metrics panel derived from summaries, ready for timeseries wiring).

## Phase 5: Perf & Licensing (Effort: 8h) ✅ COMPLETED
1. **Optimizations** — ✅ Completed (dynamic memory-efficient inference, AMP/TF32 enabled, motion gating for redundant frames).
2. **Licensing** — ✅ Completed (Apache-only guard, weights checksum verification, helper script for SHA validation, documentation updates).

## Phase 6: Tests (Effort: 12h) ✅ COMPLETED
1. **Unit: service** — ✅ Completed (pytest covers mono/multi endpoints with dummy model and skips when httpx unavailable).
2. **Unit: geometry** — ✅ Completed (backprojection/plane fitting assertions ensure numeric stability).
3. **Integration: fusion** — ✅ Completed (multi-view helpers smoke-tested with synthetic data).
4. **UX/Perf** — ✅ Completed (`scripts/perf_test.py` added for latency checks; manual UI verification for Menon and oai2-fe).

## Phase 7: Rollout & Docs (Effort: 8h)
1. **Phase A deploy** (dep: phase6, 2h)
   - Run full pipeline with mono; verify depth in Menon (OccupancyVisualizer plane) and oai2-fe (drawer heatmap).

2. **Phase B** (dep: 1, 2h)
   - Enable multi timer; check calib update (Menon CoordinateTransform re-init).

3. **Phase C optional** (dep: 2, 2h)
   - Add Menon button in SettingsPanel: collect views (from cache), call demo_colmap.py via new /api/colmap endpoint, download zip.

4. **Docs & release** (dep: 3, 2h)
   - Create docs/MIGRATION_GUIDE_MA_INTEGRATION.md: steps to enable, configs, troubleshooting.
   - Update RELEASE_NOTES: vX.Y - MA integration.
   - Menon: Update docs/websocket.md with MA messages; README for toggles.
   - oai2-fe: Add to src/README.md "MA Diagnostics drawer usage".
   - git merge/tag Noesis; commit Menon changes separately.
   - Cross-test: Noesis run; Menon/oai2-fe connect WS; verify flows.

**Tracking:** Start with todo_write merge=false, todos=[{id:'0.1', content:'Update requirements...', status:'pending'}, ... all tasks including 0.5 and expanded 4a/4b]. Update after each completed.
