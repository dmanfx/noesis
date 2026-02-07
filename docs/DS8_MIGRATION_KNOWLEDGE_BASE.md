# DS8 Migration Knowledge Base (for Planner/Debugger Agents)
_Status: current as of 2026-02-02._

This file compresses the key decisions, architecture, and current status of the DS8 migration so new agents can pick up where previous work left off.

## 1. legacy stack vs DS8 Split

- **legacy stack (legacy):** `deepstream_video_pipeline.py` + GI/GStreamer, appsinks, pad probes, INI configs.
- **DS8 (canonical):** everything under `noesis/`:
  - `noesis/ds8_runtime.py` – runtime harness (CLI, WS, REST).
  - `noesis/pipelines/ds8_pipeline.py` – Service Maker `Pipeline` graph from `config/infer.yaml`.
  - `noesis/pipelines/hooks.py` – DS8 `BatchMetadataOperator`/`Probe` hooks (intrinsics, MapAnything, analytics, exclusion).
  - `noesis/server/*` – REST APIs (depth + analytics).
  - `noesis/telemetry/*` – depth, tracking, BEV publishers.
  - `noesis/metadata/*` – intrinsics + depth schemas.
- Rules:
  - Do **not** mix legacy stack-style GI/GStreamer pipeline construction (appsinks, pad probes, ad-hoc Gst graphs) into the DS8 pipeline/hook code under `noesis/pipelines/*` and `noesis/telemetry/*`.
  - Mosaic delivery is an intentional exception: `noesis/mosaic_webrtc_gateway.py` runs a small, separate GStreamer pipeline to bridge RTSP→WebRTC.
  - Do **not** silently route DS8 failures through legacy stack.

## 2. DS8 Pipeline Architecture (High-Level)

Built in `noesis/pipelines/ds8_pipeline.py` from `config/infer.yaml`:

- Sources: `nvmultiurisrcbin` (default ingest+mux) or `nvurisrcbin` → `nvstreammux` when debugging (`NOESIS_DS8_USE_NVURISRCBIN=1`).
- Optional preprocess: `nvdspreprocess` (config `pipelines/config_preproc.ini`).
- PGIE YOLO11: `nvinfer` (`yolo11m.onnx_b3_gpu0_fp16.engine`), `unique-id=1`.
- Main tee: splits into:
  - Analytics path:
    - Optional `analytics_exclude` (`nvdsanalytics` exclude stage).
    - `nvtracker` (NvDCF, `config/nvtracker.yaml`).
    - `nvdsanalytics` (post stage, YAML + INI).
    - `nvmultistreamtiler` → `nvdsosd`.
  - Full-frame MapAnything branch (see §3).
- Sink tee: `sink_tee` branch → mosaic / BEV sinks:
  - Mosaic output: `rtsp_out` (`nvrtspoutsinkbin`) exposes `rtsp://127.0.0.1:8554/<rtsp_path>` for the WebRTC gateway.
  - `bev_sink` reserved for future Flow retriever; current BEV frames are generated from analytics metadata + calibration (no extra appsink branch in DS8).

All linking is done via `ds_pipeline.link(...)` wrapped in `_safe_link`.

## 3. MapAnything Path (Depth)

- DS8 MapAnything is a **full-frame `nvinfer` branch**, not a per-object SGIE.
- Current design (as of latest decisions):
  - Runs as a parallel branch off the main tee.
  - Uses a full-frame TensorRT engine (RGB image input), not per-object crops.
  - Does **not** rely on `input-tensor-from-meta` + 12-channel fused tensors going forward.
  - Engine IO contract (2025-12-01): single input `images` (N,3,294,518) FP16, intrinsics baked in the export wrapper (fx=fy=1000, cx=W/2, cy=H/2), output `depth` (N,1,294,518) FP16. Engine rebuilt with TensorRT 10.13 at `/home/mayor/Noesis_Devel/models/mapanything_depth/1/model.plan` (see `pipelines/config_infer_secondary_mapanything.ini`).
- Hooks:
  - `attach_mapanything_postprocess_hook` attaches `_MapAnythingOperator` via `Probe("mapanything_postprocess", ...)`.
  - `MapAnythingProcessor`:
    - DS8 path: `handle_nvds_tensor_ds8()` uses `tensor_meta.get_layers()` and converts DS8 tensors to numpy:
      - primary path via `np.from_dlpack(tensor)` when the DS8 tensor exposes CPU-backed DLPack;
      - fallback path via `tensor.__dlpack__()` → `torch.utils.dlpack.from_dlpack(...).detach().cpu().numpy()` for GPU-backed tensors.
      - The resulting arrays are passed into `_emit_from_tensors`.
    - legacy stack path: `handle_nvds_tensor()` uses pyds `NvDsInferTensorMeta` helpers.
  - `_emit_from_tensors` selects depth/confidence/mask, stores via `DepthStorageManager`, updates depth FPS, publishes `DepthResult` via `DepthTelemetryPublisher`.

### Engine situation

- A fused 12-channel engine (`ma_model_fp16_b3_fused.plan`) was temporarily used to get DS8 running; it expects tensor-from-meta and a dedicated fused-preprocess stage, which DS8 does not have.
- Decision (2025‑11‑30): **Use the existing legacy stack full-frame MapAnything engine** (RGB image input) for DS8, and remove tensor-from-meta assumptions from MapAnything configs.
- Update (2025‑12‑01): Re-exported MapAnything ONNX as images-only (no intrinsics input) and rebuilt the FP16 TensorRT engine with TensorRT 10.13. DS8 config now points to `/home/mayor/Noesis_Devel/models/mapanything_depth/1/model.plan` with `input-tensor-from-meta=0`, `infer-dims=3;518;518`, `output-tensor-meta=1`.

### Pose-conditioned MapAnything (optional)

Upstream MapAnything supports providing `camera_poses` (and/or calibration/depth) as optional geometric inputs to its `infer()` API. If we re-export/rebuild a MapAnything model/engine that consumes poses, DS8 should source them from the existing calibration files and convert them to MapAnything’s expected convention.

- Translation utility (source of truth): `noesis/metadata/mapanything_pose.py`
- Inputs:
  - `config/camera_calibration.json` per-camera `E`: **world→camera**, 4×4, **column-major**.
  - `config/ply_alignment.json` alignment `matrix` (row-major) + unit scale `units.s_obj_to_m`.
- Output:
  - MapAnything `camera_poses`: **OpenCV cam→world** 4×4 matrices (`T_wc`), with alignment applied as `T' = M @ T_wc` and unit scale applied to translation only.
- Update behavior:
  - `MapAnythingPoseProvider` auto-reloads when the calibration/alignment files change (mtime-based), keeping pose-conditioned inference in sync with `E` updates.

## 4. Metadata & Telemetry (DS8 + legacy stack Compatibility)

- Calibration (single source of truth):
  - **Runtime delivery:** `_CalibrationProvider` inside `noesis/ds8_runtime.py` loads `config/cameras.yaml` + `config/camera_calibration*.json` + `config/ply_alignment.json` and serves `calibration-bundle` over WS; BEV uses `CalibrationSnapshot` from this provider.
  - **Helper (not yet wired):** `CalibrationManager` (`noesis/calibration/manager.py`) exists for future centralized ownership but is not used by the live runtime.
  - **Intrinsics source:** `config/cameras.yaml` → `intrinsics_models` (deprecated: `intrinsics.json`, `config.py`).
  - **Extrinsics source:** `config/camera_calibration.json` → `cameras.<name>.E` (column-major, world→camera).
  - **Alignment source:** `config/ply_alignment.json` (row-major `matrix`, `floor_y`, `units.s_obj_to_m`).
  - **Full conventions:** See `docs/DS8_api_contracts_ws.md` §8 and `plans/DS8/ds8_design_decisions.md`.
- Intrinsics:
  - DS8 canonical: `CalibrationManager` (in `noesis/calibration/manager.py`) loads intrinsics from `config/cameras.yaml` and distributes calibration via `calibration-bundle`; BEV uses `CalibrationSnapshot` from `snapshot(...)`.
  - Optional DeepStream user meta: `noesis/metadata/intrinsics.py` + `attach_intrinsics_hook` can attach intrinsics to `pyds.NvDsFrameMeta`, but DS8 runtime does not attach per-frame intrinsics by default.
- MapAnything tensors:
  - `_MapAnythingOperator`:
    - DS8: iterates `batch_meta.frame_items` and `frame_meta.tensor_items`, calls `handle_nvds_tensor_ds8`.
    - legacy stack: uses `frame_meta.frame_user_meta_list` and `NVDSINFER_TENSOR_OUTPUT_META`.
- Analytics telemetry:
  - `_AnalyticsTelemetryProcessor` and `_AnalyticsTelemetryOperator`:
    - DS8: use `frame_items` and `frame_meta.object_items` + `nvdsanalytics_obj_items`.
    - legacy stack: use `NvDsFrameMeta` → `obj_meta_list` + `NvDsAnalyticsObjInfo` via pyds.
  - Produces tracking dictionaries with `track_id`, `camera_id`, `bbox`, `center`, `class_id`, `confidence`, `tracker_confidence`, `zone`, `dwell_time`, `stable_id`, and analytics fields.
  - Pose feature meta: `PoseFeatureProcessor` attaches `NOESIS.POSE_FEATURES` user meta from
    the YOLO26 pose SGIE. DS8 hooks use `noesis_pose_meta_ext.extract_pose_features(...)` to
    pass ratio features + quality into `StableIDManager` as a **secondary** identity signal.
    Pose galleries are bounded in RAM with TTL pruning (no disk persistence). See
    `docs/DS8_pose_stable_id_integration.md`.
  - Occupancy: derived from zone labels; published via `pipeline.occupancy_publisher.publish_state(...)` including vacate events.
- Exclusion:
  - DS8: primary removal via the `analytics_exclude` component (`nvdsroiexclude` by default) when `analytics.exclude.enable: true` in `config/infer.yaml`. `_ExcludePruneProcessor.handle_frame_ds8` logs would-be removals as a safety net (DS8 object_items are read-only).
  - legacy stack: `_ExcludePruneProcessor.handle_frame` removes objects via `nvds_remove_obj_meta_from_frame` / `obj_meta_list`.

## 5. Depth Gating (BufferOperator)

- Logical gating:
  - `DS8Pipeline` tracks `depth_enabled`, `depth_frame_samples`, timers via `enable_depth(seconds)` and `mark_depth_enabled(boolean)`.
  - MapAnythingProcessor drops tensors when `depth_enabled` is `False`.
- Physical gating:
  - `DepthGateOperator(BufferOperator)` is implemented; its `handle_buffer` returns `bool(pipeline.depth_enabled)`.
  - When `pyservicemaker.BufferOperator` / `Probe` are available, DS8 attaches it as:
    ```python
    from pyservicemaker import Probe
    gate = DepthGateOperator(pipeline)
    probe = Probe("depth_gate", gate)
    pipeline.ds_pipeline.attach(pipeline.depth_gate_attach, probe)
    pipeline.depth_gate_supported = True
    ```
  - `depth_gate_attach` is set to the MapAnything `nvinfer` node name in `ds8_pipeline`. On older DS8 builds where BufferOperator cannot be attached, gating remains logical-only but depth enable/disable and stats still function.

## 6. Mosaic / WebRTC / WS Integration

- Mosaic video is delivered via WebRTC:
  - DS8 pipeline builds RTSP output (`rtsp_out` / `nvrtspoutsinkbin`) and serves `rtsp://127.0.0.1:8554/<rtsp_path>`.
  - `noesis/mosaic_webrtc_gateway.py` consumes RTSP and provides a WebRTC video track to the browser.
  - WebSocket is used for signaling (`webrtc_offer` / `webrtc_answer` / `webrtc_ice_candidate`) and for non-video telemetry (stats, tracking, depth, BEV).
- WebSocket:
  - `websocket_server.py` stays the canonical WS API provider.
  - WebRTC signaling and telemetry contracts are documented in `docs/DS8_api_contracts_ws.md`.
  - BEV frames are generated by `BevRenderer` from DS8 analytics metadata and calibration and use the existing `bev-frame` JSON + binary JPEG framing (`[len(header)][b"bev:<cameraId>"][jpeg]`).

## 7. Engines & Preprocess Summary

- PGIE (YOLO11):
  - Engine: `models/yolo11m.onnx_b3_gpu0_fp16.engine`.
  - Config: `pipelines/config_infer_primary_yolo11*.ini`.
  - Preprocess: `nvdspreprocess` tuned for YOLO (3×H×W).
- MapAnything:
  - Engine: DS8 full-frame MA engine (images-only input, intrinsics baked at export time) at `models/mapanything_depth/1/model.plan` (FP16, batch 1–3, input `images`, output `depth`).
  - Config: `pipelines/config_infer_secondary_mapanything.ini` points to this engine and uses `input-tensor-from-meta=0`, `infer-dims=3;518;518`, `output-tensor-meta=1`.
  - Preprocess: reuses the main full-frame NVMM surfaces from the PGIE path; no dedicated MapAnything preprocess stage in DS8.

## 8. Status Snapshot (Phases)

For up-to-date checklists see `plans/DS8/ds8_master_work_orders.md` and individual `ds8_migration_checklist_*.md` files, but in broad strokes:

- Phases 0–2: implemented (stack split, DS8 graph, `infer.yaml`).
- Phase 3: hooks implemented; DS8 vs legacy stack metadata compatibility in place; on-device validation mostly done (tracking/occupancy/BEV telemetry paths use DS8 batch metadata; exclusion pruning is DS8-read-only with ROI filtering handled in analytics).
- Phase 4: analytics REST aligned and depth REST functional on DS8 host; ROI models map to DS8 analytics YAML. Full live ROI-behavior validation in the running DS8 pipeline is still on the checklist.
- Phase 5: Flow/gating implemented:
  - Depth gating uses `DepthGateOperator(BufferOperator)` attached at the MapAnything node when supported by the DS8 build, with logical gating always available.
  - Mosaic delivery implemented via RTSP→WebRTC gateway (WebSocket is signaling-only). BEV frames are produced by `BevRenderer` from DS8 metadata instead of a separate Flow retriever.
  - Depth tensors from MapAnything flow into `MapAnythingProcessor`, which stores snapshots under `data/depth/...`, updates depth FPS, and publishes `depth_result` WS messages during enabled windows.
- Phase 6: DS8 runtime + WS/REST run end-to-end on GPU host:
  - `noesis/ds8_runtime.py` builds the DS8 pipeline, attaches hooks, starts WebSocket + REST, and wires depth/analytics APIs.
  - WebSocketServer emits stats, tracking, depth_result, BEV messages, and WebRTC signaling for mosaic. UI validation still required for full legacy stack parity (Phase 7).
- Phase 7: full legacy stack vs DS8 parity runs still to be executed and documented.
- Phase 8: legacy stack/adapter decommission decisions not yet made.

## 9. How New Planner/Debugger Agents Should Use This

- Before major changes, always read:
  - `AGENTS.md`, `noesis/AGENTS.md`, `plans/AGENTS.md`.
  - This file: `docs/DS8_MIGRATION_KNOWLEDGE_BASE.md`.
  - `plans/DS8/ds8_master_work_orders.md` and the relevant `ds8_migration_checklist_*.md`.
- Treat this file as a **compressed summary**, not the authority:
  - If you see discrepancies between this and the checklists or code, update this file and the design decisions log accordingly.
