# Menon × Noesis: Spatial 3D Integration Plan (Internal)

Status: Draft v1 (to be audited and amended)
Owner: Codex (planning-only)
Scope: Do not change code in this run; plan only

---

## 1. Objectives

- Create a shared, metric 3D world that fuses Noesis CV tracking (DeepStream) with Menon’s Three.js scene.
- Establish robust calibration: per-camera intrinsics/extrinsics and environment alignment (PLY/scene → OBJ world).
- Enable pixel→world and world→pixel mapping, with a reliable floor-plane solution first and optional monocular depth (MDE) second.
- Stream calibrated telemetry from Noesis to Menon, so tracks/occupancy/devices have spatial attributes (x,y,z, room).

Non-goals for this sprint: multi-room SLAM, dynamic camera rigs, or live mesh reconstruction. These can follow once the foundation is solid.

---

## 2. End-State Overview

- Canonical world frame = Menon house OBJ coordinates, metric-scaled and Y-up.
- A calibration bundle (JSON) persisted on disk and broadcast on WebSocket connect:
  - `align.matrix` (float[16]): PLY → OBJ (or identity if unused)
  - `cameras.K[camId]` (fx, fy, cx, cy or 3×3) and `cameras.E[camId]` (4×4, world→camera)
  - Optional: distortion params `{k1,k2,p1,p2,k3}` if undistortion is added
- Noesis serves:
  - `frame` telemetry (as-is), plus calibrated per-track world positions when available
  - `calibration-bundle` on connect and on update
  - `pixelToWorld(u,v,depth?,camId?)` RPC (WebSocket JSON request→response) for ad-hoc queries
- Menon consumes:
  - On connect, initializes transforms and applies `M_ply2obj` to the loaded OBJ group
  - Draws tracks and occupancy using world coordinates
  - Fallback floor-plane projection when depth is not provided

---

## 3. Architecture & Data Flow

1) World Definition
- Use Menon OBJ as world. Confirm or set a reference scale: one known distance in the model (e.g., 10 ft doorway). Compute a scalar `s_obj_to_m` to convert OBJ units → meters if not already in meters.
- If a PLY exists from COLMAP or other, align via ICP to the OBJ to obtain `M_ply2obj` (Open3D script). If not using PLY, set `M_ply2obj = I`.

2) Camera Calibration
- Intrinsics K per camera:
  - Option A (fast): compute from HFOV/VFOV and resolution: `fx = (w/2)/tan(HFOV/2)`, `fy = (h/2)/tan(VFOV/2)`, cx=w/2, cy=h/2.
  - Option B (better): Charuco/AprilTag-based calibration to get K and distortion; store 3×3 K.
- Extrinsics E (world→camera):
  - Option A (manual PnP): click 4–8 3D model points (from Menon) with corresponding 2D pixels (from camera), solve PnP, validate reprojection RMS.
  - Option B (tag-based): place tags at known 3D positions; detect in Noesis; solve PnP.
- Persist K/E under `config/cameraCalibration.json` and include in the bundle.

3) Pixel→World
- Baseline (floor-plane): Intersect the camera ray through pixel (K⁻¹[u,v,1]) transformed by E⁻¹ with the world floor plane `n=[0,1,0], d=-y_floor` to get (x,y,z). This covers tracking “on the floor”.
- Advanced (MDE): Optional single-view depth via Depth-Anything v2 or ZoeDepth to recover Z, then back-project precisely. Add a per-camera scale constraint using one known measurement or camera height to resolve scale ambiguity. Run at 1–2 fps.

4) Streaming & APIs
- Extend Noesis WebSocket server:
  - Send `{type:'calibration-bundle', data:{align:{matrix}, cameras:{K,E}}}` on connect and when updated.
  - Legacy `get_transformation`/`transformation` removed; use `calibration-bundle` always.
  - Add `{type:'pixel_to_world', request_id, u, v, depth?, camId}` request and `{type:'pixel_to_world_response', request_id, world:{x,y,z}}` response.
- Menon:
  - Consume calibration-bundle, initialize `coordinateTransform` with `{M_ply2obj, cameras: {intrinsics, extrinsics}}`.
  - For tracks lacking depth, use floor-plane intersection locally; for depth-enabled queries, call backend RPC.

5) Calibration UI (Menon) & RPCs
- Menon provides a Calibration UI to either place each camera in the OBJ world (position + orientation) or capture 2D–3D correspondences.
- Direct pose path: Menon sends `{type:'set_extrinsics', cameraId, E|Twc|{pos, quat|ypr}}` to Noesis. Noesis converts to `E` (world→camera), validates (reprojection error if provided), persists, and rebroadcasts the bundle.
- Correspondence path: Menon sends `{type:'solve_pnp', cameraId, intrinsicsId|K, points2D, points3D}`; Noesis solves PnP, returns RMS for confirmation, then persists and rebroadcasts.
- Conventions: `E` is world→camera; matrices as column‑major float[16]; right‑handed, Y‑up. Include `floor_y` and `s_obj_to_m` when available.

---

## 4. File/Module Impact (No Edits in this run; reference only)

Noesis (Python):
- `websocket_server.py` — add message types: calibration-bundle, get_transformation, pixel_to_world(+result).
- `config.py` — add `camera_calibration_path`, `ply_alignment_path`, `camera_specs` (HFOV/VFOV/resolution per camera), and feature flags for MDE.
- New: `calibration_bundle.py` (loader/saver, schema validation); `pixel_to_world.py` (ray-plane intersect; MDE gating).
- New: `scripts/ply_icp_align.py` (Open3D) to compute `M_ply2obj` from PLY→OBJ via ICP with voxel down-sampling + optional correspondences.
- Optional: `mde_infer.py` (Depth-Anything v2 inference wrapper) run at low FPS; caches downsampled depth maps per camera.

Additional RPCs (Noesis):
- In `websocket_server.py`, add handlers for `{type:'set_extrinsics'}` (persist + rebroadcast) and optional `{type:'solve_pnp'}` (compute, validate, persist) used by Menon’s Calibration UI.

Menon (Three.js):
- `websocket-client.js` — already references `calibration-bundle` and `get_transformation`; ensure handlers wire into `calibrationManager` and `coordinateTransform`.
- `coordinateTransform.js` — exposes `init`, `worldToScreen`, and calls backend for `screenToWorld`; add a local `pixelToWorldFloor()` as fallback (ray-plane) and a `getIsTransformationSet()`; confirm convention E=world→camera.
- `sceneBuilder.js` — apply `M_ply2obj` once to environment group after load.
- `calibrationManager.js` — persists `{alignMatrix, cameras:{K,E}}` to localStorage; ensure file-load path `config/cameraCalibration.json` and `config/ply_alignment.json`.

Known current mismatches in Menon code (to fix in implementation):
- `sceneBuilder.js` references `window.coordinateTransform.applyTransformationToOBJ` and `getIsTransformationSet()` which do not exist in the current `coordinateTransform.js`.
- `coordinateTransform.js` calls `window.backend.pixelToWorld`, but there is no `backend` object; this should be routed via WebSocket RPC.
- `M_ply2obj` may be applied twice if both `sceneBuilder.js` and `coordinateTransform.js` attempt to transform the environment; ensure single application.

Shared assets (Menon workspace):
- `config/cameraCalibration.json`: schema `{K:{camId:[fx,fy,cx,cy] or 3x3}, E:{camId:[16 floats]}}`.
- `config/ply_alignment.json`: `{matrix:[16 floats]}`.

---

## 5. Detailed Task Plan (Codex-ready)

Phase 0 — Baseline Inventory & Decisions
1. Validate OBJ scale: measure a known dimension in Menon; decide `s_obj_to_m` and note in docs.
2. Decide initial calibration path: PnP/manual first; MDE optional; PLY ICP optional.

Phase 1 — Schemas & Persistence
1. Define `calibration-bundle` schema (JSON):
   - `align.matrix` float[16], column-major, PLY→OBJ; identity if unused.
   - `cameras.K[camId]` either `[fx,fy,cx,cy]` or `[[fx,0,cx],[0,fy,cy],[0,0,1]]`.
   - `cameras.E[camId]` 4×4 float[16], world→camera (THREE.Matrix4 order).
   - Optional `distortion[camId]`.
2. Add loader/saver in Noesis: `calibration_bundle.py` with robust validation and defaults.
3. Place initial files into Menon `config/`: `cameraCalibration.json`, `ply_alignment.json` (placeholder values).

Phase 2 — Noesis WebSocket Extensions
1. Add broadcast on connect: `calibration-bundle` (read from disk via `calibration_bundle.py`).
2. Implement request handler for `{type:'get_transformation'}`.
3. Implement RPC for pixel→world floor-plane: `{type:'pixel_to_world'}` → `{type:'pixel_to_world_response'}`.
4. Unit tests (lightweight) for request→response mapping and matrix math.
5. Implement handler for `{type:'set_extrinsics'}` and optional `{type:'solve_pnp'}`; persist and rebroadcast calibration-bundle.

Phase 3 — Menon Consumer Hardening
1. `websocket-client.js`: verify existing `calibration-bundle` handler re-initializes `coordinateTransform` using `{M_ply2obj,cameras}`.
2. `coordinateTransform.js`: ensure
   - `init(calib)` accepts `{M_ply2obj, cameras:{intrinsics,extrinsics}}`.
   - `getIsTransformationSet()` returns boolean; add `pixelToWorldFloor(u,v,camId)` for ray-plane.
   - `worldToScreen()` uses E=world→camera and `[fx,fy,cx,cy]`.
   - Ensure `screenToWorld()` defers to WebSocket RPC when depth is supplied; otherwise use `pixelToWorldFloor()`.
3. `sceneBuilder.js`: apply `M_ply2obj` once to the root environment group immediately after load (avoid double-apply paths).
4. `occupancyVisualizer.js`/`pathVisualizer.js`: consume world positions if provided; otherwise use floor fallback.

Phase 4 — Calibration Capture (PnP)
1. Minimal UI (Menon): a tool to pick 3D world points (from OBJ) and record 2D pixel correspondences per camera (existing `calibrationUI.js` can be trimmed to “PnP mode”).
2. Send correspondences to Noesis; solve PnP (OpenCV) to compute E, return RMS, and persist.
3. Visual validation: draw back-projected rays/axes overlays.

Phase 5 — Optional PLY Alignment & MDE
1. PLY ICP script with Open3D: downsample, initial PCA or 3-point correspondences, run point-to-plane ICP, output `ply_alignment.json`.
2. MDE module (optional): Depth-Anything v2 or ZoeDepth; 1–2 fps; produce downsampled depth maps cached per camera; scale using camera height or known segment length.
3. Expose a flag in Noesis to opt-in to depth-based pixel→world.

Phase 6 — Validation & Tooling
1. Synthetic QA: spawn a virtual camera inside Menon with known E; generate pixel picks and verify round-trip error < 3 px.
2. Real QA: tape-measure two distances between landmarks; compare Menon distances when mapped from pixels.
3. Logging: add one-line summaries when calibration loads and when pixel→world RPCs complete.

---

## 6. Requirements & Conventions

- Matrix conventions:
  - Three.js is right-handed, Y-up. OBJ is assumed Y-up; if not, correct with an axis flip in `M_ply2obj`.
  - `E` is world→camera (applied to world point to get camera coords).
  - All 4×4 matrices serialized as column-major float[16] (Three.js compatible).
- Pixel coordinates: origin at image top-left; intrinsics `[fx,fy,cx,cy]` assume pixel centers; undistort first if needed.
- Floor plane:
  - Use Menon’s `findFloorY()` to determine `y_floor` from the loaded OBJ; default to model bounding-box `min.y`.
  - Ray-plane intersection formula: solve `R = C + t*d` for `n·R + d0 = 0` where `C` is camera origin, `d` is ray direction, `n=[0,1,0]`, `d0=-y_floor`.
- Units: meters for all world distances; store `s_obj_to_m` in a small note or config.
- Depth: if using MDE, it is scale-ambiguous. Always store a `depth_scale` per camera.
- Distortion: if significant, undistort pixel before back-projection (future).

---

## 7. Risks & Mitigations

- OBJ unit ambiguity → Mitigate with one measured reference length; enforce metric scaling.
- Single static camera depth ambiguity → Use floor-plane ray-cast for position; MDE optional for height.
- Feature-poor textures for ICP/COLMAP → Allow 3–4 manual correspondences to seed alignment.
- Runtime drift between Noesis and Menon transforms → Source-of-truth bundle, versioned; Menon only consumes.
- Performance: MDE GPU load → decimate to 1–2 fps and quarter resolution; cache.
- Serialization errors (row/column-major) → Add unit tests: round-trip a known transform and verify a sample point projects/backs correctly.

---

## 8. Deliverables

- `config/cameraCalibration.json` and `config/ply_alignment.json` in Menon.
- Noesis code to read, validate, and broadcast calibration-bundle; pixel→world RPC.
- Menon initialization of transforms + floor-plane fallback.
- PnP calibration tool and Open3D ICP script (optional).
 - Menon Calibration UI for camera placement / correspondence capture; Noesis RPCs `set_extrinsics` and optional `solve_pnp`.

---

## 9. References (for implementation)

- NVIDIA DeepStream: nvdsanalytics, nvinfer (docs.nvidia.com)
- Three.js camera and Matrix4 docs
- OpenCV solvePnP, Rodrigues
- Open3D ICP pipeline
- Monocular depth: Depth-Anything v2, ZoeDepth

---

## 10. Step-by-Step (Concrete File TODOs)

Note: These are prescriptive edits for future implementation; do not apply in this planning run.

Noesis
- Create `calibration_bundle.py` with schema and load/save helpers.
- Update `websocket_server.py`:
  - On connect → send `calibration-bundle`.
  - Add handlers for `get_transformation` and `pixel_to_world`.
  - Add handlers for `set_extrinsics` (persist + rebroadcast) and optional `solve_pnp` (compute, validate, persist).
- Add `pixel_to_world.py` (ray-plane intersection; optional MDE path).
- Add `scripts/ply_icp_align.py` (Open3D) to create/update `ply_alignment.json`.
- Extend `config.py` with calibration file paths and camera specs.
  - Add a section for camera specs (resolution, HFOV/VFOV), and a helper to derive `[fx,fy,cx,cy]` if explicit K not provided.

Menon
- Ensure `config/cameraCalibration.json` and `config/ply_alignment.json` load in `calibrationManager.js`.
- Add a Calibration UI to place cameras (position + orientation) or capture 2D–3D correspondences; on save, call `set_extrinsics` or `solve_pnp` and show RMS for confirmation.
- Harden `coordinateTransform.js` to:
  - Accept `M_ply2obj` + K/E; expose `getIsTransformationSet()` and `pixelToWorldFloor()`.
  - Keep `screenToWorld()` calling backend for depth when available.
  - Add a tiny guard to avoid applying environment transforms twice.
- `sceneBuilder.js`: apply `M_ply2obj` only once to the root environment group.
- `websocket-client.js`: verify `calibration-bundle` wiring and avoid re-applying transforms.

---

## 11. Validation Checklist
- Camera ID mapping
  - Use stable string IDs (e.g., `"living-room"`, `"kitchen"`) throughout Noesis and Menon. The ID used in JPEG binary prefix and in telemetry `camera_id` must match the keys under `cameras.K/E`.

- Calibration loads on Menon startup (log + UI toast optional).
- A clicked pixel on the video maps to a plausible floor position in 3D.
- Tracks drift < 0.5 m over typical room spans with floor-plane method.
- MDE optional path yields consistent depth ordering and approximate metric scale.
- Verify a known 3D corner projects within ≤3 px in the image (world→pixel test).
- Verify `z_cam > 0` gating (points behind camera are rejected) in `worldToScreen`.

---

## 12. Open Questions

- Are room coordinates in OBJ already metric? If not, what is the chosen meter scale?
- Do we prefer `[fx,fy,cx,cy]` intrinsics or full 3×3 K on the wire? (Plan supports both.)
- Is tag-based PnP feasible in the environment, or must we rely on manual correspondences?

---

## 13. Example Payloads (for alignment)

Example `calibration-bundle` broadcast from Noesis:

```json
{
  "type": "calibration-bundle",
  "data": {
    "align": { "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] },
    "cameras": {
      "K": {
        "living-room": [1450.2, 1452.8, 960.0, 540.0]
      },
      "E": {
        "living-room": [
          0.998, 0.010, -0.065, 0.0,
         -0.012, 0.999, -0.040, 0.0,
          0.064, 0.041,  0.997, 0.0,
         -2.400, 1.750, -3.200, 1.0
        ]
      }
    }
  }
}
```

Example `pixel_to_world` RPC:

```json
{ "type": "pixel_to_world", "request_id": "abc123", "u": 875, "v": 610, "camId": "living-room" }
{ "type": "pixel_to_world_response", "request_id": "abc123", "world": {"x": 1.72, "y": 0.00, "z": -2.91} }
```

---

## 14. As‑Built (Noesis) Integration Details

- WebSocket messages (server → client)
  - `calibration-bundle` on connect and after updates:
    - `{ type: 'calibration-bundle', data: { align:{ matrix:[16], floor_y:float, units:{ s_obj_to_m:float } }, cameras:{ <cameraId>:{ intrinsics:{ fx,fy,cx,cy, K3x3? , distortion? }, extrinsics:{ E:[16]? } } }, meta:{ conventions:{ E:'world→camera', up:'Y', handedness:'RH' } } } }`
  - `transformation` (reply to `get_transformation`): `{ type:'transformation', data:{ M_ply2obj:[16], cameras:{...} } }`
  - Telemetry `stats`: unchanged envelope; each `active_tracks[]` may include `world: [x,y,z]` and `world_valid`.

- WebSocket messages (client → server)
  - `get_transformation` → server responds with `transformation`.
  - `pixel_to_world`: `{ type:'pixel_to_world', request_id, camId, u, v }` → `{ type:'pixel_to_world_response', request_id, ok, world?, error? }`
    - Errors: `calibration_missing`, `bad_extrinsics`, `no_intersection`.
  - `set_extrinsics`: `{ type:'set_extrinsics', cameraId, E:[16] | Twc:[16] }` (column‑major). Server persists and rebroadcasts `calibration-bundle`.
  - `solve_pnp`: handler stubbed (optional to wire next).

- Calibration bundle specifics
  - `align.matrix`: column‑major 4×4 (PLY→OBJ). `floor_y` in meters. `units.s_obj_to_m` scalar.
  - Cameras keyed by IDs: `living-room`, `kitchen`, `family-room`.
  - Intrinsics accept `fx,fy,cx,cy` and/or `K3x3`; `distortion` optional (currently treated as zero if present).
  - Extrinsics `E`: column‑major 4×4 world→camera.

- Camera IDs
  - Derived from DeepStream `source_info.clean_name`. Current mapping: living-room (G3 Instant), kitchen (G3 Instant), family-room (G4 Instant).

- Pixel→world
  - Backend computes bottom‑center bbox footpoint ray vs `y = floor_y`. Menon may keep a local floor‑plane fallback; for ad‑hoc/precise queries use RPC.

- Files (Noesis)
  - `intrinsics.json` (root), `config/camera_calibration.json`, `config/ply_alignment.json`.

- Conventions
  - Column‑major arrays, E is world→camera, RH, Y‑up; pixels origin top‑left.

- Menon workflow
  - On connect: handle `calibration-bundle`, apply `M_ply2obj` once.
  - Calibration UI: send `set_extrinsics` with E (or Twc) and wait for rebroadcast bundle.
  - Use `pixel_to_world` RPC for screen→world when depth or exactness is needed.

- Notes
  - Distortion assumed zero today; we can add a footage‑based estimator later.
  - Runtime updates to `floor_y`/`units` can be added via a future `set_align` RPC if needed.
