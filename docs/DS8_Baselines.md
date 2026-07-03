# DS8 Baselines & Live Defaults

This page condenses the current DS8 defaults and non-negotiable behaviors so
you don’t have to sift through the historical work orders.

## Identity, Trails, and Telemetry
- `stable_id` is the only user-visible person ID. `track_id` is internal and
  must not be emitted to clients.
- BEV/tracking use `stable_id` as the primary user-visible person ID.
  - BEV `footpoints[]` may additionally include `trackerId` as a debug/fallback
    identity key. UI must not treat `trackerId` as stable across restarts.
- **Pose-assisted StableID (secondary signal):** when pose SGIE is enabled
  (`config/infer.yaml` `models.pose.enable: true`), StableIDManager auto-enables
  pose ratio fusion unless disabled via `NOESIS_REID_POSE_ENABLED=0` or
  `NOESIS_POSE_FEATURES_ENABLED=0`. Memory is bounded via per‑ID galleries and
  TTL pruning. See `DS8_pose_stable_id_integration.md` for thresholds/caps.
- Trail colors are stable-id-first; fallback to camera-namespaced colors to keep
  cross-camera IDs visually distinct.
- Negative or provisional IDs stay internal; never send negative IDs to UI.
- StableIDManager is fed by the ReID SGIE tensors (no torchreid extractor) and
  is enabled by default (`NOESIS_REID_ENABLED=1`). Default embedding interval is
  `0.0s` (every frame); device defaults to `cuda:0`. Disable entirely with
  `NOESIS_REID_ENABLED=0`.

## Output & Delivery
- Mosaic video is delivered via RTSP → WebRTC gateway; WebSocket is signaling
  only for mosaic (no JPEG-over-WS in DS8).
- BEV JPEG binaries are **retired**:
  - `NOESIS_BEV_JPEG_ENABLED` and `bev.jpeg_enabled` are ignored.
  - JSON BEV metadata is the supported delivery path.
- Inline dashboard BEV frame mode defaults to `camera_local_ground_m` so
  footpoints and trails share the same metric X/Z frame as MapAnything
  floorplan rasters.
  - Override with `NOESIS_BEV_FRAME=backend_world_m` only for explicit world-BEV
    validation or Menon-facing traces.
  - For camera-local BEV, tracked people normally publish
    `displaySource=world_to_camera_local` from the live fused tracking world
    state. `displaySource=floor_contact_ray` is the calibrated fallback when no
    current live world observation is available. Static MapAnything/floorplan
    snapshots are not used as live person-depth placement inputs.
  - Producer-owned BEV smoothing is active in the backend; the dashboard should
    render `footpoints` and backend `trails` directly.

## Depth / MapAnything / Baseline World Tracking
- MapAnything is still a full-frame SGIE branch (no tensor-from-meta), but it
  is no longer the only depth-related path in DS8.
- MapAnything lane:
  - gated with `mapanything_valve`
  - REST: `GET /api/v1/depth/refresh?seconds=N` opens the gate for N seconds
  - owns `depth_result` and `ma_depth_response`
  - writes dense snapshots under `data/depth/<camera>/...`
- Baseline non-`v3dt` tracking lane:
  - uses an always-on full-frame DAv2 branch (`models.depth_tracking`)
  - attaches raw per-object depth through `NOESIS.OBJECT_DEPTH`
  - fuses pose anchor + floor observation + registered DAv2 range inside the
    backend world estimator
  - does not publish a second full-frame WebSocket depth stream
- Baseline startup now requires a valid DAv2->MapAnything registration artifact:
  - config key: `depth_registration.path`
  - default artifact: `config/depth_registration.json`
  - DS8 fails fast if any enabled camera is missing a valid entry
- `depth_used_m` and the OSD `z=` label represent the registered DAv2 depth
  that actually participated in the fused world update, not the raw
  `anchor_depth_m`.
- Depth normals (in `ma_depth_response`) are optional; enable/disable with
  `NOESIS_MAPANYTHING_NORMALS_ENABLE=1|0` and choose space/dtype via
  `NOESIS_MAPANYTHING_NORMALS_SPACE` (`camera`|`world`) and
  `NOESIS_MAPANYTHING_NORMALS_DTYPE` (`float16`|`float32`).

## Detector Profiles (PGIE)
- **Default:** YOLO11 detect, PGIE `unique-id=1`, person is `class_id=0`.
  - Runtime switch: omit `--pgie-profile`, set `NOESIS_PGIE_PROFILE=yolo11`,
    or pass `--pgie-profile yolo11`.
  - Size switch: `--size s|m|l` (default `m`).
  - Assets:
    - `models/yolo11s.onnx`,
      `models/engines/yolo11s_b3_fp16.engine`
    - `models/yolo11m.onnx`,
      `models/engines/yolo11m_b3_fp16.engine`
    - `models/yolo11l.onnx`,
      `models/engines/yolo11l_b3_fp16.engine`
  - Runtime materializes size-specific PGIE/preprocess configs from
    `pipelines/config_infer_primary_yolo11.ini` and `pipelines/config_preproc.ini`.
- **Optional:** YOLO11-seg (instance masks), PGIE `unique-id=1`, person is
  `class_id=0`.
  - Switch via `--pgie-profile yolo11_seg --size s|m|l` (default `m`).
  - Assets:
    - `models/yolo11s-seg_cust_fused.onnx`,
      `models/engines/yolo11s-seg_cust_fused.engine`
    - `models/yolo11m-seg_cust.onnx`,
      `models/engines/yolo11m-seg_cust.engine`
    - `models/yolo11l-seg_cust.onnx`,
      `models/engines/yolo11l-seg_cust.engine`
  - Runtime materializes size-specific PGIE/preprocess configs from
    `pipelines/config_infer_primary_yolo11_seg.ini` and `pipelines/config_preproc.ini`.
  - Some YOLO11-seg assets, including the current `l` ONNX/engine, require the
    TensorRT plugin library from `external/DeepStream-Yolo-Seg`; DS8 preflight
    loads it when the selected ONNX contains `EfficientNMSX_TRT` or
    `ROIAlignX_TRT`.
- **Optional:** RF-DETR-seg (n/s/m)
  - Switch via `--pgie-profile rfdetr_seg` plus `--size n|s|m` (default `m`).
  - Engines:
    - `models/engines/rfdetr_seg_n_312_b3_fp16.engine`
    - `models/engines/rfdetr_seg_s_384_b3_fp16.engine`
    - `models/engines/rfdetr_seg_m_432_b3_fp16.engine`
  - Configs: runtime materializes `build/config_infer_primary_rfdetr_seg_<size>.ini`
    from `pipelines/config_infer_primary_rfdetr_seg.template.ini` and uses
    `pipelines/config_preproc_rfdetr_{312|384|432}.ini`
  - Parser: `pipelines/nvdsinfer_rfdetr_seg/libnvdsinfer_rfdetr_seg.so`
  - Person remap: parser maps RF-DETR person class to DS `class_id=0`; keep
    PGIE `unique-id=1` so ReID/SGIE continue to hook correctly.
- **Optional:** YOLO26-seg (n/s/m). Switch via `--pgie-profile yolo26_seg` plus
  `--size n|s|m` (default m). Engines must exist under `models/engines/`.
- **Optional:** DEIMv2 Wholebody49. Switch via `--pgie-profile wholebody49`
  plus `--size s|x` (default `s`).
  - `s`: DINOv3-S instance-mask model,
    `models/engines/deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine`.
  - `x`: DINOv3-X label-only model,
    `models/engines/deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine`.
  - Runtime materializes `build/config_infer_primary_wholebody49_<size>.ini`
    plus `build/config_preproc_wholebody49_<size>_b3.ini`.
  - Parser: `pipelines/nvdsinfer_deimv2_wholebody49/libnvdsinfer_deimv2_wholebody49.so`.
  - The X instance-mask engine is not a promoted runtime asset until it is
    explicitly built and validated.

## Trails & ID Refactor Guardrails
- BEV uses `stable_id` for user-visible identity; `trackerId` may be included as
  debug/fallback identity only.
- BEV JPEG output is retired; use BEV JSON metadata.
- Camera-namespaced fallback colors remain enabled to avoid cross-camera color
  collisions when stable_id is absent.

## V3DT / SV3DT Baseline (2026-01-22 locked)
- Pipeline: `config/infer_v3dt_baseline.yaml`
- Tracker: `config/v3dt/nvtracker_v3dt_baseline.yml`
- CamInfo dir: `config/v3dt/caminfo_baseline/`
- Calibration: `config/archive/calibration_v3dt_baseline.json`
- Per-camera pitch (preview extrinsics): family-room **-16°**, kitchen **-21°**,
  living-room **-15°**
- Model dimensions: height **2.2 m**, radius **0.35 m**
- CamInfo conventions (env defaults):
  - `NOESIS_V3DT_CAMINFO_INVERT_E=0` (E is world→camera)
  - `NOESIS_V3DT_CAMINFO_Y_FLIP=1`
  - `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`
  - `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`
- Known gaps: living-room still slightly shallow (bottom offset ~160px); BEV
  validation pending. See `history/ds8/v3dt/` for detailed runs.
- Optional camInfo regeneration: `NOESIS_V3DT_AUTOGEN_CAMINFO=1` regenerates
  camInfo using the current streammux resolution before starting the pipeline.

## WebRTC / Mosaic
- Canonical mosaic path is RTSP (`nvrtspoutsinkbin`) consumed by
  `noesis/mosaic_webrtc_gateway.py`.
- Ensure RTSP output is enabled when WebRTC is enabled
  (`mosaic_output.mosaic_webrtc_enabled: true` auto-enables RTSP).

## Where to Confirm Details
- Full rationale and history: `DS8_MIGRATION_KNOWLEDGE_BASE.md`,
  `history/ds8/` (mirrored plans and work orders).
- Contracts: `DS8_api_contracts_ws.md`, `DS8_api_contracts_rest.md`,
  `DS8_metadata_contracts.md`.
