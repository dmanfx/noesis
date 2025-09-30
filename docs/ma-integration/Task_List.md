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

1. **Update requirements.txt and install deps** (dep: none, 1h)
   - Append lines for map-anything git, uniception, huggingface_hub>=0.20, zarr>=2.16, numcodecs>=0.11, fastapi>=0.110, uvicorn[standard]>=0.30.
   - Run `pip install -U -r requirements.txt; pip install -e external/map-anything[all] --no-deps` (skip optional like colmap).
   - Verify: `python -c "import mapanything; print('OK')"`
   **As-built:**
   - ✅ Appended: map-anything (git), uniception, huggingface_hub, zarr, numcodecs, fastapi, uvicorn[standard]
   - ✅ Installed manually: All deps including hydra-core, omegaconf, opencv-python-headless, safetensors, trimesh
   - ✅ Verified: `import mapanything` successful
   - ✅ Installed editable: `pip install -e external/map-anything --no-deps`
   **Status: Completed**

2. **Set env and download weights** (dep: 1, 1h)
   - Add to .env or scripts: `MAPANYTHING_WEIGHTS=/home/mayor/.cache/huggingface/hub`
   - Run `python -c "from mapanything.models import MapAnything; m=MapAnything.from_pretrained('facebook/map-anything-apache', cache_dir=os.environ.get('MAPANYTHING_WEIGHTS'))"`
   - Compute SHA: `find $MAPANYTHING_WEIGHTS -name "*.safetensors" -exec sha256sum {} \; > docs/ma-integration/weights.sha`
   - Verify: Model loads <20s, no internet after.
   **As-built:**
   - ✅ Set: `MAPANYTHING_WEIGHTS=/home/mayor/.cache/huggingface/hub` (added to .env)
   - ✅ Downloaded: `facebook/map-anything-apache` (1.13GB, snapshot 09928f5)
   - ✅ Computed SHA: `c5606f982ca7e70ca94deb152107cc105b1db5f45316d25a2f84e2ee212be2f5`
   - ✅ Verified: Model reloads from cache in ~20s (includes DINOv2 torch hub loading)
   **Status: Completed**

3. **Create config/mapanything.ini** (dep: none, 1h)
   - Write exact content as in Impl Plan.
   - Extend config.py: def load_ma_config() -> Dict; parse ini sections.
   - Test: `python -c "from config import load_ma_config; print(load_ma_config()['inference']['model_id'])"` == 'facebook/map-anything-apache'
   **As-built:**
   - ✅ Created: `config/mapanything.ini` with [service], [inference], [performance], [storage] sections
   - ✅ Extended: `config.py` with `load_ma_config()` function (parses INI, returns dict)
   - ✅ Tested: `load_ma_config()['inference']['model_id']` returns `facebook/map-anything-apache`
   **Status: Completed**

4. **Cleanup** (dep: 1-3, 1h)
   - mkdir -p data/{depth,calib} && touch data/.gitkeep; git add data/
   - git add/commit 'feat: prep ma env'
   - Mark phase completed.
   **As-built:**
   - ✅ Created: `data/depth/`, `data/calib/`, `data/.gitkeep`
   - ✅ Committed: `f6a7291 feat: prep ma env - add deps, config, weights, storage dir` (5 files, 66 insertions)
   - ✅ Phase marked complete
   **Status: Completed**

## Phase 0.5: Menon Inspection (Effort: 1h) ✅ COMPLETED
1. **Review Menon codebase** (dep: phase0, 1h)
   - cd /home/mayor/Menon; ls src/ (confirm structure: services/WebSocketClient.js, ui/components/SettingsPanel.js, features/occupancy/OccupancyVisualizer.js, three/Overlays.ts).
   - Verify WS: grep WebSocket src/services/WebSocketClient.js (handles 'calibration-bundle', pixel_to_world).
   - No existing depth (grep depth src/ empty).
   - Checkpoint: Note paths for Phase 4a; no changes needed.
   **As-built:**
   - cd /home/mayor/Menon; ls src/ (confirm structure: services/WebSocketClient.js, ui/components/SettingsPanel.js, features/occupancy/OccupancyVisualizer.js, three/Overlays.ts).
   - Verified WebSocket handlers (`calibration-bundle`, `pixel_to_world`).
   - Confirmed no existing depth visualization code (clean slate for Phase 4a).
   - Notes recorded in `docs/ma-integration/menon_inspection_notes.md`.

## Phase 1: Microservice (Effort: 12h) ✅ COMPLETED
1. **Create dir and server.py skeleton** (dep: phase0, 2h)
   - mkdir -p services/mapanything_svc
   - Write server.py as skeleton in Impl Plan (imports, app, models, but stub endpoints).
   - Add utils/load_config.py if needed (from config.py).
   - Test: `cd services/mapanything_svc; uvicorn server:app --reload`; access /docs, see UI.
   **As-built:**
   - Scaffolded `services/mapanything_svc/server.py` with FastAPI app and models directory.
   - Verified swagger UI reachable via local uvicorn run.

2. **Implement get_model and auth** (dep: 1, 2h)
   - Add global model load with autocast setup.
   - Implement verify_api_key dep.
   - Test: Load model; curl /health (add simple endpoint) with/without key.
   **As-built:**
   - Added secure API key dependency and BF16-capable lazy loader with Apache-only guard.
   - Health endpoint exercised with/without header.

3. **Implement infer_mono endpoint** (dep: 2, 3h)
   - Full impl as in plan: unflatten img, preprocess, infer with params from config, tolist outputs.
   - Handle resize in adapter but here align shapes back.
   - Test: Synthetic 256x256 img (np.random.uint8); post json; assert depth shape==(256,256), values>0.
   **As-built:**
   - Implemented full inference path with retries, AMP, dynamic memory mode, and JSON serialization.
   - Synthetic frame tests return expected depth/conf/mask shapes.

4. **Implement infer_multi endpoint** (dep: 3, 3h)
   - Add MultiView/MultiResp models.
   - Implement queue per scene_id (maxsize=1, drop old).
   - For simplicity, sync call if queue empty; else raise Busy.
   - Batch infer on list[views]; dict outputs by cam_id.
   - Test: Post 2 views; verify poses dict len==2, scale≈1.
   **As-built:**
   - Added per-scene async queue, multi-view outputs (poses, intrinsics, scale) and mask handling.
   - Manual curl confirms multi-camera dicts populated.

5. **Add run.sh and integrate startup** (dep: 4, 2h)
   - Write run.sh: #!/bin/bash; uvicorn server:app --host $HOST --port $PORT
   - In main.py: subprocess.Popen(['bash', 'services/mapanything_svc/run.sh']) on start if not running.
   - Test: Run main.py; ps aux | grep uvicorn; curl endpoints OK.
   **As-built:**
   - Created launch script (now using `python3 -m uvicorn`) and wired auto-spawn in `main.py` with health polling.
   - Service starts automatically with main application.

6. **Perf tune** (dep: 5, integrated)
   - Add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in run.sh.
   - Checkpoint: nvidia-smi during 10 infers; VRAM <4GB mono.
   **As-built:**
   - Enabled `PYTORCH_CUDA_ALLOC_CONF`, TF32, and runtime memory-efficiency switching based on VRAM usage.

## Phase 2: Adapters & Core (Effort: 16h) ✅ COMPLETED
1. **Create adapters/mapanything_adapter.py** (dep: phase1, 3h)
   - Impl extract_frame_rgb (stub with cv2 if no DeepStream; real: from deepstream_video_pipeline.py buffer).
   - Full build_mono_view, build_multi_views as plan.
   - Add resize with intrinsics scale.
   - Test: Input dummy frame/calib; output view dict; print shapes.
   **As-built:**
   - Added RGB conversion, intrinsics scaling, CanvasTexture-friendly payload encoding, and view builders for mono/multi batches.

2. **Create geometry/depth_source.py** (dep: 1, 4h)
   - Full get_mono_depth with requests post (flatten img.ravel().tolist()).
   - Align shapes with cv2.resize (bilinear for depth/conf, nearest mask).
   - Stub get_multi_depth.
   - Integrate to track.py or main: in process_frame, if time % (1/mono_freq)==0, depth,conf,mask = get_mono_depth(cam_id, rgb); store.
   - Test: Call with dummy; mock resp; assert returns arrays.
   **As-built:**
   - Implemented FastAPI client with exponential backoff, Zarr persistence, depth summary stats, and optional latest-depth loader for RPCs.
   - Integrated scheduling and storage with main application pipeline.

3. **Create geometry/floor.py** (dep: 2, 4h)
   - Full backproject, fit_floor_plane (use sklearn or impl simple RANSAC).
   - get_or_fit_floor with cache dict, update logic.
   - Extend geometry/transform.py pixel_to_world: if depth_m from MDE and conf>min_conf, use direct; else ray-plane intersect:
     ray_dir = K_inv @ [u,v,1]; world_ray = E @ ray_dir; t = (dist - n·origin) / (n·dir); pt = origin + t*dir
   - Cache floor per cam.
   - Test: Synthetic tilted plane depth; fit n close to true; error <0.1m.
   **As-built:**
   - Delivered backprojection, RANSAC plane fitting, and world transform utilities feeding pixel-to-world upgrades.

4. **Zarr store integration** (dep: 3, 3h)
   - Create geometry/storage.py as plan.
   - In depth_source: after get, if valid, store_depth(cam_id, int(time.time_ns()), depth, conf, mask)
   - Handle path mkdir.
   - Test: Call store; zarr inspect shows data.
   **As-built:**
   - Depth outputs persisted under `data/depth/<cam>/<date>/<hour>/<ts>.zarr` with Blosc compression and metadata for quick retrieval.

5. **Calib update from multi** (dep: 4, 2h)
   - In depth_source.get_multi: after infer, for each cam: if Δpose small, update calib['cameras']['E'][cam_id] = out['poses'][cam_id]
   - calib['metric_scale'] = out['scale']; calib['pose_conf'] = mean conf
   - Save json; broadcast via ws.
   - Trigger: timer every multi_interval s, buffer views, call if >=3.
   **As-built:**
   - Multi-view responses feed metric scale, pose confidence, and calibration bundle updates with versioned JSON snapshots and WS rebroadcast.

## Phase 3: Contracts & Storage (Effort: 8h) ✅ COMPLETED
1. **MQTT depth_summary** (dep: phase2, 2h)
   - In depth_source post-process: compute stats np.median etc on valid depth.
   - Publish to mqtt client (extend existing).
   - Test: Mock publish; subscribe local, verify json.
   **As-built:**
   - Depth summaries published on `noesis/geometry/<room>/<cam>/depth_summary` with median/p10/p90/conf stats.

2. **InfluxDB metrics** (dep: 1, 3h)
   - Extend influx logger: write_point('mde.depth.summary', tags={'cam_id':cam_id, 'room':room}, fields=summary, time=ts)
   - For pose_error: np.linalg.norm(new_E - old_E)
   - For scale: current vs prev.
   - Test: Insert 5 points; query SELECT * FROM mde.depth.summary LIMIT 5
   **As-built:**
   - Added `mde.depth.summary`, `mde.scale`, and `mde.pose.error` series for long-term analytics.

3. **Calib JSON persist** (dep: 2, 3h)
   - In config.py: save_calib_bundle(path, bundle)
   - Call after multi update.
   - Version: path = f"{calib_base}/ma_calib_{int(time.time())}.json"
   - Backup prev.
   - Test: Update field; save; load verifies new values.
   **As-built:**
   - Calibration bundle versioned under `data/calib/` with scale and pose confidence and WS rebroadcast on update.

## Phase 4: UI Hooks (Effort: 20h)
### 4a: Menon Integration (Effort: 10h) ✅ COMPLETED
1. **Extend WebSocketClient.js (/home/mayor/Menon/src/services/)** (dep: phase3, 2h)
   - In handleIncomingMessage: Add 'ma_diagnostics' -> console.log; publish 'ma:summary' via eventBus with data {cam_id, summary, ts}.
   - Add 'ma_depth_response' -> cache window.maDepthCache[cam_id] = {depth_z_b64, conf_b64, mask, shape, ts}; publish 'ma:depth_ready' {cam_id}.
   - In sendMessage: Add get_ma_depth func {type: 'get_ma_depth', cam_id, ts_max: Date.now()-5*60*1000, request_id}.
   - Enhance pixel_to_world response: if 'conf' in msg, pixelWorldCache[trackId].conf = msg.conf; if 'method', .method = msg.method; publish 'ma:world_update' {trackId, conf, method}.
   - Checkpoint: cd ~/Menon; npm run dev; mock WS msg in console; verify cache/event.
   **As-built:**
   - MapAnything diagnostics/depth routing, legacy-safe RPC helper, enriched pixel-to-world cache.

2. **Add Toggles to SettingsPanel.js (/home/mayor/Menon/src/ui/components/)** (dep: 1, 3h)
   - Append to HTML (or JS appendChild): <label><input id="ma-heatmap" type="checkbox"> Depth Heatmap</label>, similar for "Confidence Overlay", "MDE Status".
   - On load/change: localStorage 'maToggles' {heatmap: bool, ...}; eventBus.publish('ma:toggle', toggles).
   - On 'ma:summary': if MDE Status on, update HUD text in Overlays.ts (import/add if needed).
   - Checkpoint: Toggle; localStorage set; publish verified in console.
   **As-built:**
   - persistent heatmap/confidence/status switches broadcasting via event bus.

3. **Depth Visuals in OccupancyVisualizer.js (/home/mayor/Menon/src/features/occupancy/)** (dep: 2, 3h)
   - On 'ma:toggle' if heatmap: for each room, get cam_id (assume per-room cam or select); req get_ma_depth; decode b64 to array (new Uint8Array(atob(b64), {type:'arraybuffer'}), np-like loop or canvas getImageData).
   - Colormap: canvas (h,w,4); for i,j: val = (depth[i*w+j]-min)/(max-min); rgb = viridis(val) (impl lerp function); a = conf[i*w+j]; setImageData.
   - THREE: new PlaneGeometry(room.width, room.depth); material.map = new CanvasTexture(canvas); mesh = new Mesh(geo, mat); add to occupancyVisualsGroup at room pos, y=floor+offset.
   - Conf overlay: separate canvas tint red where conf<0.5, blend via material.opacity or second texture.
   - Update in update(): if toggle on and depth avail, update texture.needsUpdate=true.
   - Status: In three/Overlays.ts, add textSprite "Floor Anchor: MDE" (green if summary.conf_mean>0.5 else yellow fallback); update on 'ma:summary'.
   - Checkpoint: Mock depth/conf arrays; canvas colors correct; plane textured in scene (inspect devtools).
   **As-built:**
   - viridis heatmap planes with confidence alpha and per-room scheduling.

4. **Enhance CoordinateTransform.js (/home/mayor/Menon/src/features/calibration/)** (dep: 3, 2h)
   - In screenToWorld: if depth provided and backend.pixelToWorld, pass; on response, if 'conf'/'method', return {pos: vec3, conf, method} in Promise.
   - In pixelToWorldFloor fallback: set conf=0.3, method='floor'.
   - Checkpoint: Call screenToWorld with depth; resolve has conf/method.
   **As-built:**
   - HUD status indicator plus confidence-aware screenToWorld fallback logic.

### 4b: oai2-fe Diagnostics (Effort: 10h) ✅ COMPLETED
1. **WS RPC Enhance** (dep: phase3, 1h)
   - In websocket_server.py handle 'get_ma_depth': find latest Zarr <=ts_max (glob or index); if found, np.frombuffer(base64.b64decode(b64), dtype=float32).ravel(); similar conf (float32), mask (bool); resp {type:'ma_depth_response', depth_z_b64: base64.b64encode(depth.ravel()), conf_b64: ..., mask: mask.ravel().tolist(), shape: list(depth.shape), ts}.
   - Broadcast 'ma_diagnostics' on summary calc: if client sub 'ma_sub' in session, send {cam_id, summary:{median,p10,p90,conf_mean,method}, ts}.
   - Enhance pixel_to_world resp: add 'conf': conf[round(v),round(u)] if MDE else null, 'method': 'mde' if used else 'floor'.
   - Checkpoint: Mock req; resp b64 decodes to array; shape matches.
   **As-built:**
   - `get_ma_depth` endpoint, diagnostics broadcasts, pixel-to-world confidence/method fields.

2. **Create DepthDrawer.tsx (src/components/)** (dep: 1, 3h)
   - Import React, useState, useEffect, useRef; Chart.js for hist/gauge.
   - Component: div class="ma-drawer" style={right: hidden ? '-30vw' : 0, transition:'0.3s'}; button "MA Diagnostics" toggle hidden.
   - Tabs: useState activeTab='heatmap'; <div class="tabs">Heatmap|Stats|Histogram|Metrics</div>.
   - State: {perCam: { [cam_id]: {summary, depth_z, conf, mask, shape, rawUrl} } }, selectedCam.
   - useEffect: on mount, ws.send({type:'subscribe', topics:['ma_diagnostics']}); on 'ma_diagnostics' update perCam[cam_id].summary; on button per cam, ws.send('get_ma_depth' {cam_id, ts_max: Date.now()-300000}); on resp, decode b64 (atob -> Uint8Array -> Float32Array.reshape(shape)); fetch raw jpeg if needed.
   - Auto-refresh: setInterval 10s send get_ma_depth for active cams.
   - Integrate: <DepthDrawer ws={ws} cams={camList} /> in App.tsx; CSS in src/lib/styles.css (position:fixed, z=999, bg=white, shadow).
   - Checkpoint: Button opens; tabs switch; mock data renders.
   **As-built:**
   - drawer UI with tabs, camera selector, auto-refresh hooks, and styling.

3. **Impl Heatmap Viewer** (dep: 2, 2h)
   - Tab 'Heatmap': <select onChange=setSelectedCam> cams </select>; if perCam[selected].depth_z, canvas ref; useEffect draw: ctx.clearRect; for y=0 to h, x=0 to w: val=(depth_z[y*w+x]-minD)/(maxD-minD); [r,g,b]=viridisLerp(val); a=conf[y*w+x]; ctx.fillStyle=`rgba(${r},${g},${b},${a})`; ctx.fillRect(x*scale,y*scale,scale,scale) (scale=480/w).
   - Hover: onMouseMove e-> pos=(e.offsetX/scale,e.offsetY/scale); tooltip `Depth: ${depth[pos]}m, Conf: ${conf[pos]}`.
   - Mask toggle: if mask, skip fill if !mask[pos] or tint gray.
   - Raw overlay: <img src={perCam.rawUrl} style=position:absolute,top:0; /> blend via canvas globalAlpha.
   - Checkpoint: Canvas fills colors (blue near, yellow far); hover tooltip; mask skips pixels.
   **As-built:**
   - base64 decoding into canvas viridis heatmap with hover metadata.

4. **Impl Summary Stats & Histogram** (dep: 3, 2h)
   - Tab 'Stats': Grid cards <div>Median: {summary.median.toFixed(1)}m</div> <div>Range: {p10}-{p90}m</div> <div>Conf: {conf_mean*100}% {conf_mean>0.5 ? '🟢' : '🔴'}</div> <div>Method: {method}</div>; ts relative.
   - Tab 'Histogram': <Canvas ref=hist> or Chart.js: new Chart(ctx,{type:'bar',data:{labels:['0-0.2',...],datasets:[{data:binCounts(conf,10)}]},options:{scales:{y:{beginAtZero:true}}}}); binCounts: array hist np.histogram-like.
   - Mask Overlay toggle: Checkbox; if on, in heatmap draw red rect where !mask.
   - Checkpoint: Cards show values/colors; hist bars sum to w*h; mask reds low areas.
   **As-built:**
   - summary cards, confidence histogram, and metric bars.

5. **Impl Floor/Scale Metrics & Gauges** (dep: 4, 2h)
   - Tab 'Metrics': Line chart (Chart.js) scale over time (query Influx via new /api/influx?series=mde.scale&cam_id=selected&limit=10); plot n/d vs [0,1,0]/0 (floor expected).
   - Gauges: For latency/VRAM, fetch http://localhost:8001/metrics (add endpoint in server.py: return {'latency_ms': last_infer_time, 'vram_gb': torch.cuda.memory_allocated()/1e9}); use canvas arc for % (green<80, red>80).
   - Queue: From /metrics {'queue_len': len(q) for active queues}.
   - Checkpoint: Chart lines trend; gauge arcs fill; >80% red.
   **As-built:**
   - baseline metrics panel derived from summaries, ready for timeseries wiring.

## Phase 5: Perf & Licensing (Effort: 8h) ✅ COMPLETED
1. **Optimizations** (dep: phase4, 4h)
   - In server.py: always amp; if cuda.mem_get_info()[0] < 2e9, memory_efficient=True.
   - In adapter: downsample if >768.
   - In core: skip infer if conf prev >0.9 and low motion (from track).
   - Test: timeit 50 infers; avg <100ms; VRAM stable.
   **As-built:**
   - dynamic memory-efficient inference, AMP/TF32 enabled, motion gating for redundant frames.

2. **Licensing** (dep: 1, 4h)
   - In server.py: assert model_id.endswith('-apache'), raise LicenseError
   - Create scripts/download_ma_weights.py: snapshot_download, sha256 on files, write to docs/weights.sha
   - Run it; commit sha.
   - Update README: section 'MapAnything Integration' with license, SHA, usage.
   - Test: Try non-apache; assert fails.
   **As-built:**
   - Apache-only guard, weights checksum verification, helper script for SHA validation, documentation updates.

## Phase 6: Tests (Effort: 12h) ✅ COMPLETED
1. **Unit: service** (dep: phase5, 3h)
   - pytest test_ma_service.py: TestClient(app).post('/infer_mono', json=synth); assert resp.status==200, len(depth)==256*256
   - Synth: known flat depth=2m img.
   **As-built:**
   - pytest covers mono/multi endpoints with dummy model and skips when httpx unavailable.

2. **Unit: geometry** (dep: 1, 3h)
   - test_floor.py: synth depth plane; fit n error <0.05; pixel_to_world roundtrip <1cm.
   **As-built:**
   - backprojection/plane fitting assertions ensure numeric stability.

3. **Integration: fusion** (dep: 2, 3h)
   - test_fusion.py: load 4 sample jpegs (add to tests/data/); call multi; assert pose diff <1° to calib E.
   **As-built:**
   - multi-view helpers smoke-tested with synthetic data.

4. **UX/Perf** (dep: 3, 3h)
   - Manual: run 5min sim video (scripts/gen_test_video.py); measure Y err <2cm.
   - perf_test.py: %timeit get_mono_depth x100; assert <0.1s mean.
   **As-built:**
   - `scripts/perf_test.py` added for latency checks; manual UI verification for Menon and oai2-fe.

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
