# DS8 Design Decisions Log

Use this file to record non-trivial design choices made during the DS8 migration. Each entry should be short and reference any relevant docs or code.

## Template

- **Date:** YYYY-MM-DD
- **Author:** (Codex / human)
- **Area:** (e.g., Pipeline, Telemetry, REST, Metadata)
- **Decision:**
- **Rationale:**
- **References:** (docs URLs, file paths, etc.)

## Entries

- **Date:** 2026-02-01
- **Author:** Codex
- **Area:** ReID / Pose
- **Decision:** Feed YOLO26 pose feature meta into StableIDManager as a gated secondary similarity signal (pose-only fallback when embeddings are missing) and store pose vectors in bounded RAM (deque + TTL + global cap) with no disk persistence; add a native extractor to read pose meta from `NvDsObjectMeta`.
- **Rationale:** Pose ratios can reinforce identity when appearance embeddings are weak or missing, but pose features are high‑volume; bounded in‑memory storage prevents resource growth. Service Maker bindings do not expose object user meta lists, so a native bridge is required to read the pose payload safely.
- **References:** `reid/stable_id_manager.py`, `noesis/pipelines/hooks.py`, `native/noesis_pose_meta_ext.cpp`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-02-01
- **Author:** Codex
- **Area:** Telemetry / Stats
- **Decision:** Add an always-on DS8 latency heartbeat using DeepStream’s built-in latency measurement, sampling at `osd:src` via a Service Maker `Probe` and exporting rolling p50/p95/max to the WebSocket `stats` payload.
- **Rationale:** Keeps a low-overhead, continuously available end-to-end latency signal without introducing DS7 pad probes or CPU branches. Service Maker buffer probes require attaching to an output pad, so `osd:src` is the last stable tap point before the sink tee/RTSP branches. Uses `ctypes` to call `libnvds_meta.so:nvds_measure_buffer_latency` to avoid the `pyds.nvds_measure_buffer_latency` wrapper’s unconditional stdout printing.
- **References:** `noesis/telemetry/latency_metrics.py`, `noesis/pipelines/ds8_pipeline.py`, `noesis/ds8_runtime.py`, `docs/DS8_api_contracts_ws.md`, `/opt/nvidia/deepstream/deepstream/sources/includes/nvds_latency_meta.h`

- **Date:** 2026-02-01
- **Author:** Codex
- **Area:** Metadata / Pose
- **Decision:** Enable host tensor copy for the YOLO26 pose SGIE, default pose tensor decoding to the manual DLPack path (torch off), and share a per-frame pose tensor cache between feature extraction and keypoint overlay so each tensor is consumed once.
- **Rationale:** Pose hooks run twice per object (features + overlay). Consuming the same tensor via torch DLPack was the highest-risk path for DeepStream user-meta corruption and segfaults. Host copy plus a single decode per frame eliminates ownership hazards while preserving pose features and overlays.
- **References:** `pipelines/config_infer_secondary_yolo26_pose.ini`, `noesis/pipelines/hooks.py`, `plans/DS8/ds8_migration_checklist_hooks.md`

- **Date:** 2026-01-31
- **Author:** Codex
- **Area:** Depth / WebSocket / UI
- **Decision:** Compute and attach optional MapAnything normals to `ma_depth_response` payloads on demand, using camera-space normals encoded as float16.
- **Rationale:** Normals are needed for the depth drawer without introducing a streaming telemetry channel; computing on demand keeps bandwidth and compute bounded while reusing existing depth RPC flow.
- **References:** `geometry/depth_source.py`, `noesis/ds8_runtime.py`, `oai2-fe/src/components/DepthDrawer.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-01-30
- **Author:** Codex
- **Area:** Metadata / Pose
- **Decision:** Attach YOLO26 pose features as object‑level user meta (`NOESIS.POSE_FEATURES`) via a small pybind11 helper that uses `nvds_add_user_meta_to_obj`, storing a compact JSON payload.
- **Rationale:** Service Maker Python wrappers do not expose `obj_user_meta_list`/append APIs, so a native shim is the most reliable way to attach per‑object metadata without introducing DS7 pad probes or CPU branches.
- **References:** `noesis/pipelines/hooks.py`, `native/noesis_pose_meta_ext.cpp`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-01-30
- **Author:** Codex
- **Area:** Telemetry / Floorplan
- **Decision:** Apply the same extrinsics‑driven image‑axis flips used by BEV homography to floorplan generation, and tag cached floorplans with `image_flip` for invalidation.
- **Rationale:** BEV flips image axes when the camera frame is mirrored relative to world axes; without applying the same flip in floorplan generation, the top‑down background appears left↔right mirrored while tracks move correctly.
- **References:** `geometry/depth_source.py`, `noesis/telemetry/bev.py`

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Pipeline / PGIE / Segmentation / GPU-first
- **Decision:** Fuse YOLO26-seg models to output a single tensor (`output0`) that already contains per-detection masks (fixed 64x64 flattened) for the top 30 detections, remove the prototype tensor from graph outputs, and enable `disable-output-host-copy=1` so DeepStream keeps outputs in device memory.
- **Rationale:** The unfused graph (`output0` + `output1` proto) triggers a large device→host copy for `output1` and forces CPU-side mask composition, which violates the DS8 GPU-first policy and increases overhead. Fusing mask compose into the TensorRT engine keeps the heavy work on-GPU and allows the custom parser to copy only the minimal slices required to populate DeepStream metadata. Capping to 30 detections bounds ROIAlign workload and yielded a large utilization drop on YOLO26n (reported ~10% GPU in DS8 runs vs ~80–90% when composing 300 masks).
- **References:** `utils/fuse_yolo26_seg_onnx.py`, `pipelines/config_infer_primary_yolo26_seg.template.ini`, `pipelines/nvdsinfer_yolo26_seg/nvdsinfer_yolo26_seg.cpp`, `noesis/ds8_runtime.py`, `models/yolo26{n,s,m}-seg_fused.onnx`, `models/engines/yolo26{n,s,m}-seg_fused_b3_fp16.engine`

- **Date:** 2026-01-28
- **Author:** Codex
- **Area:** Pipeline / PGIE Profiles
- **Decision:** Add a `yolo26_seg` PGIE profile with a CLI-only `--size {n,s,m}` selector (default `m`) that materializes a size-specific PGIE INI under `build/` from a generic template and points the overlay at the chosen b3 engine.
- **Rationale:** Keeps the DS8 config surface clean (single template), avoids maintaining multiple static INIs, and guarantees the engine/parser paths are explicit and validated at runtime without DS7 fallback.
- **References:** `noesis/ds8_runtime.py`, `pipelines/config_infer_primary_yolo26_seg.template.ini`, `pipelines/nvdsinfer_yolo26_seg/`, `docs/DS8_testing_guide.md`

- **Date:** 2026-01-23
- **Author:** Codex
- **Area:** V3DT / Baseline / Naming
- **Decision:** Normalize the SV3DT baseline filenames to stable, short paths:
  `config/infer_v3dt_baseline.yaml`,
  `config/v3dt/nvtracker_v3dt_baseline.yml`,
  `config/v3dt/caminfo_baseline/`,
  `config/cameras_v3dt_baseline.yaml`,
  `config/archive/calibration_v3dt_baseline.json`,
  `config/dewarper_v3dt_baseline.txt`,
  `config/analytics_exclude_baseline.ini`.
- **Rationale:** The shortened baseline names avoid long, brittle filenames and make
  the locked baseline easier to reference across scripts, docs, and checklists.
- **References:** `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`,
  `plans/DS8/v3dt/README.md`, `plans/DS8/v3dt/AGENTS.md`

- **Date:** 2026-01-22
- **Author:** Codex
- **Area:** V3DT / Baseline / Calibration
- **Decision:** Lock the SV3DT baseline on the preview calibration + camInfo set:
  `config/archive/calibration_v3dt_baseline.json`,
  `config/v3dt/caminfo_baseline/`,
  tracker `config/v3dt/nvtracker_v3dt_baseline.yml`,
  and pipeline `config/infer_v3dt_baseline.yaml`.
  CamInfo uses `w2p`, `INVERT_E=0`, `Y_FLIP=1`, `WORLD_AXES=xzy`, `WORLD_SCALE=1`, model height 2.2m.
- **Rationale:** This configuration yields the lowest reprojection errors observed to date across
  family-room and kitchen and materially improves living-room stability; it is the first
  end-to-end set that reproduces tracks without SV3DT collapse.
- **References:** `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`, `plans/DS8/v3dt/iteration_notes_2026-01-22.md`,
  `diagnostics/trackdump_runs/rect774_pitch_m16_h220_k21_l15_live_20260122_225218/v3dt_report_20260122_225453.md`

- **Date:** 2026-01-22
- **Author:** Codex
- **Area:** V3DT / PGIE / Projection
- **Decision:** Keep PGIE aspect-ratio handling disabled for SV3DT runs:
  `maintain-aspect-ratio=0` and `symmetric-padding=0`.
- **Rationale:** When PGIE letterboxing is enabled, SV3DT projection math sees a
  double aspect correction (streammux + PGIE) which produces large reprojection
  errors and unstable 3D fits.
- **References:** `pipelines/config_infer_primary_yolo11_seg.ini`,
  `diagnostics/trackdump_runs/rect774_pitch_m15_h200_20260122_195943/v3dt_report_20260122_200148.md`

- **Date:** 2026-01-22
- **Author:** Codex
- **Area:** V3DT / Alignment / BEV
- **Decision:** Keep alignment scale in meters (`config/ply_alignment.json` `units.s_obj_to_m=1.0`).
- **Rationale:** Non-unit scaling (e.g., 0.010849) shrinks world translations by ~100×,
  corrupting BEV and pixel→world outputs when extrinsics are already in meters.
- **References:** `config/ply_alignment.json`, `noesis/metadata/mapanything_pose.py`,
  `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`

- **Date:** 2026-01-21
- **Author:** Codex
- **Area:** Bidnetpipe / Segmentation / Model Prep
- **Decision:** Bake ADE20K mean/std normalization and softmax into the exported BiSeNetV2 ONNX so nvinfer receives probability maps without requiring per-channel std in `nvinfer` config.
- **Rationale:** `nvinfer` supports scalar `net-scale-factor` and per-channel offsets but not per-channel std. Exporting an ONNX wrapper that applies `/255`, mean/std normalization, and softmax keeps preprocessing faithful to the BiSeNet training pipeline and allows `segmentation-threshold` to operate on probabilities.
- **References:** `Bidnetpipe/model_setup.py`, `Bidnetpipe/DOCS.md`, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html, https://github.com/CoinCheung/BiSeNet

- **Date:** 2026-01-18
- **Author:** Codex
- **Area:** V3DT / Telemetry / BEV
- **Decision:** Prefer `image_base` (bbox3d base-center projection) for BEV/footpoint anchoring, then fall back to `image_foot`, then bbox bottom-center.
- **Rationale:** `image_foot` can land on a cuboid edge when SV3DT foot location is offset; using `image_base` aligns the projected 3D cuboid base with the person center while preserving safe fallbacks when the projection is missing or invalid.
- **References:** `noesis/pipelines/hooks.py`, `docs/DS8_metadata_contracts.md`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-01-15
- **Author:** Codex
- **Area:** V3DT / Calibration / Dewarper
- **Decision:** Keep `nvdewarper` output at the rectified ChArUco K, but apply a **family-room-only intrinsics scale (2.7x)** in SV3DT camInfo to align projected cuboids.
- **Rationale:** Scaling both the dewarper output K and SV3DT camInfo made cuboid projection worse. Scaling camInfo only (with dewarper fixed) yields a near-ideal family-room projection (H ratio ~1.01, W ratio ~1.03). This isolates the correction to SV3DT projection math while preserving the undistorted image geometry.
- **References:** `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`, `config/dewarper_family_room_charuco_rtsp.txt`, `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_family-room.yml`, `diagnostics/v3dt_report_20260115_192145.md`, `plans/DS8/v3dt/iteration_notes_2026-01-15.md`
  _2026-01-22 (Codex): Superseded — current baseline removes the 2.7x camInfo scale hack and uses per-camera pitch preview extrinsics instead._

- **Date:** 2026-01-14
- **Author:** Codex
- **Area:** V3DT / Calibration / camInfo
- **Decision:** Allow `scripts/generate_v3dt_caminfo.py` to use an explicit `resolution` field in `config/cameras.yaml` intrinsics models when scaling K to streammux size, instead of assuming principal point is centered.
- **Rationale:** ChArUco intrinsics can yield a principal point offset from image center. Inferring base resolution from `2*cx/2*cy` mis-scales K when `cx/cy` are off-center, leading to projection drift. Explicit `resolution` keeps scaling consistent with the true calibration frame size.
- **References:** `scripts/generate_v3dt_caminfo.py`, `config/cameras_preview_charuco_fr.yaml`

- **Date:** 2026-01-14
- **Author:** Codex
- **Area:** V3DT / Calibration / Auto Tilt
- **Decision:** Add an optional image-space Y flip to the depth-plane fit used by the tilt preview tool (`--flip-image-y`), to align MapAnything depth (image Y-down) with Y-up world coordinates.
- **Rationale:** Auto-tilt runs were producing correct downward pitch but ~180° roll; flipping the image-space Y axis yields upright roll while preserving negative pitch, indicating the depth plane normals were computed in image Y-down coordinates.
- **References:** `noesis/calibration/tilt_preview.py`, `scripts/auto_tilt_from_depth.py`, `docs/DS8_v3dt_forensics.md`

- **Date:** 2026-01-13
- **Author:** Codex
- **Area:** V3DT / Calibration
- **Decision:** Add a depth-plane **tilt-only** preview tool that aligns the ground-plane normal to +Y while preserving yaw and camera center, writing to a separate preview calibration file.
- **Rationale:** We need a safe way to correct pitch/roll drift (tilt) without overwriting Menon’s calibration or perturbing yaw/translation. A preview file enables side-by-side camInfo comparisons before committing.
- **References:** `noesis/calibration/tilt_preview.py`, `scripts/auto_tilt_from_depth.py`, `docs/DS8_v3dt_forensics.md`, `plans/DS8/v3dt/README.md`

- **Date:** 2026-01-13
- **Author:** Codex
- **Area:** V3DT / Calibration / Telemetry
- **Decision:** Treat SV3DT world as **Z-up** (height = `bbox3d.zLen`) and add a camInfo axis-remap escape hatch (`NOESIS_V3DT_CAMINFO_WORLD_AXES`, e.g., `xzy` to swap Y/Z) to align Y-up calibration with DeepStream’s SV3DT frame.
- **Rationale:** Live telemetry showed `bbox3d.zLen ≈ 1.7m` while `bbox3d.yLen ≈ 0.7m`, matching model height and indicating Z is the vertical axis. DeepStream documentation describes world foot positions as 2D (X,Y) on the ground plane, which implies Z-up. The remap keeps calibration files intact while letting SV3DT receive the expected axis convention.
- **References:** `/tmp/ds_sv3dt_docs/gst-nvtracker.html`, `scripts/generate_v3dt_caminfo.py`, `noesis/pipelines/hooks.py`, `noesis/diagnostics/v3dt_forensics.py`, `plans/DS8/v3dt/status_summary_2026-01-12_meters_caminfo.md`

- **Date:** 2026-01-13
- **Author:** Codex
- **Area:** V3DT / Runtime Defaults
- **Decision:** Set V3DT camInfo defaults to the working baseline: `NOESIS_V3DT_AUTOGEN_CAMINFO=1`, `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p`, `NOESIS_V3DT_CAMINFO_INVERT_E=1`, `NOESIS_V3DT_CAMINFO_Y_FLIP=1`, `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`.
- **Rationale:** This baseline produces bbox3d on all cameras with correct height scale; making it the default reduces setup errors and ensures new agents reproduce the best-known state without manual env tweaking.
- **References:** `noesis/ds8_runtime.py`, `scripts/generate_v3dt_caminfo.py`, `docs/DS8_v3dt_forensics.md`, `plans/DS8/v3dt/README.md`
  _2026-01-22 (Codex): Superseded — locked baseline now uses `INVERT_E=0` with pre-generated camInfo (`NOESIS_V3DT_AUTOGEN_CAMINFO=0`)._

- **Date:** 2026-01-12
- **Author:** Codex
- **Area:** Diagnostics / V3DT
- **Decision:** Add a DS8 V3DT forensics toolkit that captures a calibration snapshot (inputs + derived projection math), logs per-frame tracking telemetry to NDJSON, generates a report (with concrete checks like bbox3d coverage, height sanity, and reprojection error), and renders an offline HTML panel. Persist raw `set_extrinsics` payloads to `logs/calibration_raw/` for audit.
- **Rationale:** SV3DT failures were being debugged by implication; a deterministic snapshot + runtime log + report makes root causes explicit (units, projection mismatch, Y-axis convention, bbox3d coverage, track fragmentation) and keeps raw Menon payloads intact for comparison.
- **References:** `noesis/diagnostics/v3dt_forensics.py`, `noesis/diagnostics/telemetry_log.py`, `scripts/v3dt_forensics.py`, `docs/DS8_v3dt_forensics.md`, `noesis/calibration/manager.py`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / PGIE Profiles
- **Decision:** Implement a runtime `--pgie-profile {yolo11_seg,rfdetr_seg}` “profile overlay” that deep-merges only `preprocess.config-file`, `models.pgie.config-file-path`, and `models.pgie.engine` into the selected base YAML, writes `build/effective_pipeline_<profile>.yaml`, and builds DS8 from that effective config.
- **Rationale:** Avoids maintaining duplicate near-identical pipeline YAMLs and guarantees the PGIE swap does not alter DS8 topology; keeps default behavior YOLO while enabling an explicit RF-DETR startup switch with clear, auditable effective config logging.
- **References:** `noesis/ds8_runtime.py`, `config/infer.yaml`, `config/infer_v3dt_medium.yaml`, `config/infer_v3dt_sv3dt.yaml`, `plans/DS8/ds8_rfdetr_seg_swap_checklist_validated.md`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / RF-DETR Segmentation
- **Decision:** Start DS8 RF-DETR integration with `RFDETRSegPreview` defaults (`resolution=432`, `num_queries=200`, `num_classes=90`) and a people-only DeepStream mapping (`DS classId=0`), cropping query masks to the detected bbox (DeepStream per-object instance mask expectation).
- **Rationale:** Matches upstream `rfdetr==1.3.0` SegPreview defaults (minimizes “guessing”/overrides), preserves existing DS8 person-only + ReID-on-classId=0 behavior, and ensures masks render correctly via nvdsosd by supplying bbox-relative instance masks rather than full-frame masks.
- **References:** `pipelines/config_preproc_rfdetr_432.ini`, `pipelines/config_infer_primary_rfdetr_seg.ini`, `pipelines/nvdsinfer_rfdetr_seg/nvdsinfer_rfdetr_seg.cpp`, `plans/DS8/ds8_rfdetr_seg_swap_checklist_validated.md`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / RF-DETR Parser Scoring + Class IDs
- **Decision:** Score RF-DETR detections using sigmoid on the “person” logit only (`score = sigmoid(logits[person_idx])`) and default `NOESIS_RFDETR_PERSON_CLASS_IDX=1` (COCO category id 1), mapping those detections to DeepStream `classId=0` for DS8 parity.
- **Rationale:** Upstream RF-DETR postprocess uses per-class sigmoid (not softmax) and COCO class indices correspond to COCO category IDs (1..90). Scoring/mapping this way preserves the existing DS8 “person == classId 0” contract required by ReID (`operate-on-class-ids=0`) while avoiding brittle best-class filtering under sigmoid multi-label logits.
- **References:** `pipelines/nvdsinfer_rfdetr_seg/nvdsinfer_rfdetr_seg.cpp`, `/home/mayor/.local/lib/python3.12/site-packages/rfdetr/models/lwdetr.py` (`PostProcess`), `/home/mayor/.local/lib/python3.12/site-packages/rfdetr/util/coco_classes.py`, `plans/DS8/ds8_rfdetr_seg_swap_checklist_validated.md`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / Mosaic RTSP / WebRTC
- **Decision:** Treat `sinks.mosaic_sink` and `sinks.bev_sink` as **semantic placeholders** (for future Flow retrievers) and do not build/link them as real `fakesink` tee branches when RTSP mosaic output is enabled.
- **Rationale:** Building extra fakesink branches off the post-OSD tee can interfere with RTSP mosaic delivery on some DS8 builds, resulting in an RTSP port that accepts TCP but times out on RTSP OPTIONS/SETUP/PLAY and prevents the WebRTC gateway from answering offers. Keeping the RTSP branch as the only terminal output (when enabled) restores RTSP responsiveness and WebRTC video delivery.
- **References:** `noesis/pipelines/ds8_pipeline.py`, `noesis/ds8_runtime.py`, `config/infer_v3dt_medium.yaml`, `scripts/webrtc_gateway_smoke_test.py`

- **Date:** 2026-01-09
- **Author:** Codex
- **Area:** Calibration / WebSocket RPC / SV3DT camInfo / Units
- **Decision:** Persist DS8 camera extrinsics (`config/camera_calibration.json` `E`) in **meters** and auto-coerce cm-like translations to meters in `CalibrationManager.set_extrinsics()` to prevent SV3DT camInfo scaling drift (camInfo remains cm via `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100`).
- **Rationale:** A real-world failure mode was observed where `set_extrinsics` inputs effectively used centimeters (camera center Y ≈ 147–230), but camInfo generation assumes meters and multiplies translation by `WORLD_SCALE=100`. This produced projection translations 100× too large and caused SV3DT track termination/mask flicker under motion. Coercing cm→m at the persistence boundary keeps DS8 calibration conventions stable and prevents accidental reintroduction of this class of bug.
- **References:** `noesis/calibration/manager.py`, `config/camera_calibration.json`, `config/v3dt/camInfo_living-room.yml`, `config/v3dt/camInfo_kitchen.yml`, `config/v3dt/camInfo_family-room.yml`, `noesis/ds8_runtime.py` (`_ds8_set_extrinsics_handler`, `_maybe_sync_v3dt_caminfo`), `scripts/generate_v3dt_caminfo.py`, `plans/DS8/v3dt/ds8_v3dt_hardening_work_order.md` (V3DT-H01)

- **Date:** 2025-12-30
- **Author:** Codex
- **Area:** Tracking / SV3DT / MV3DT / Calibration / Units
- **Decision:** Standardize all V3DT world coordinates on **METERS** (not centimeters). Changed `NOESIS_V3DT_CAMINFO_WORLD_SCALE` default from `100` to `1.0` in both `scripts/generate_v3dt_caminfo.py` and `noesis/ds8_runtime.py`. Regenerated all camInfo files with `modelInfo.height: 1.7`, `radius: 0.35` (meters).
- **Rationale:** Meters is the standard/canonical convention for world coordinates. The documentation already specified meters, but the implementation was using centimeters. Aligning to meters simplifies reasoning, matches documentation, and follows best practices.
- **References:** `scripts/generate_v3dt_caminfo.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `config/v3dt/camInfo_*.yml`, `plans/DS8/v3dt/ds8_v3dt_hardening_work_order.md` (V3DT-H01)
  _2026-01-09 (Codex): Superseded — current SV3DT path keeps calibration/telemetry in meters, but generates camInfo in centimeters (`NOESIS_V3DT_CAMINFO_WORLD_SCALE=100`) for SV3DT stability; see 2026-01-09 entry above._

- **Date:** 2025-12-30
- **Author:** Codex
- **Area:** Tracking / SV3DT / Metadata Bridge / Native Extension
- **Decision:** Add try/catch protection around all meta extractions in `noesis_v3dt_meta_ext` native bridge; confirm `NVDS_OBJ_WORLD_FOOT_LOCATION` cannot be extracted because Service Maker does not expose `ObjectWorldFootLocationUserMetadata` class (only `ObjectImageFootLocationUserMetadata` for image-space foot); world footpoints must be derived from `bbox3d` (yCentre − 0.5·yLen).
- **Rationale:** Grep of `/opt/nvidia/deepstream/deepstream/service-maker/includes/metadata.hpp` shows only: `ObjectVisibilityUserMetadata`, `ObjectImageFootLocationUserMetadata`, `Object3DBBoxUserMetadata`. There is no class for world foot location. The try/catch wrappers ensure individual extraction failures don't crash the pipeline and are logged once. This supersedes the 2025-12-29 entry which incorrectly implied the class existed but crashed.
- **References:** `native/noesis_v3dt_meta_ext.cpp`, `/opt/nvidia/deepstream/deepstream/service-maker/includes/metadata.hpp`, `plans/DS8/v3dt/ds8_v3dt_hardening_work_order.md` (V3DT-H05)

- **Date:** 2025-12-29
- **Author:** Codex
- **Area:** Tracking / SV3DT / Metadata Bridge
- **Decision:** Disable extracting `NVDS_OBJ_WORLD_FOOT_LOCATION` via Service Maker `ObjectImageFootLocationUserMetadata` in the DS8 native bridge (`noesis_v3dt_meta_ext`) because it segfaults in `deepstream::ObjectImageFootLocationUserMetadata::getImageFootLocation()`; rely on `bbox3d`-derived world footpoints instead.
- **Rationale:** Under SV3DT, iterating `NVDS_OBJ_WORLD_FOOT_LOCATION` inside a Service Maker `BatchMetadataOperator` triggered a reproducible SIGSEGV in NVIDIA's Service Maker library. Dropping the field keeps DS8 SV3DT stable while preserving the core 3D outputs (`bbox3d`/`velocity3d`); BEV/world are still computed from the 3D bbox (ground footpoint = yCentre − 0.5·yLen).
- **References:** `native/noesis_v3dt_meta_ext.cpp`, `noesis/pipelines/hooks.py`, `/opt/nvidia/deepstream/deepstream/lib/libnvds_service_maker.so` (crash site), `scripts/sv3dt_meta_smoke_test.py`
  _2025-12-30 (Codex): Superseded by the 2025-12-30 entry above which clarifies that the class doesn't exist in Service Maker._

- **Date:** 2025-12-29
- **Author:** Codex
- **Area:** Tracking / SV3DT / Calibration / camInfo
- **Decision:** Treat `config/camera_calibration.json` `E` as **world→camera** when generating V3DT camInfo projection matrices (`projectionMatrix_3x4_w2p`) and expose an override via `NOESIS_V3DT_CAMINFO_INVERT_E` (default `0`).
- **Rationale:** Menon composes `Twc` (camera→world) then sends `E = inv(Twc)` (world→camera) to Noesis. SV3DT expects `projectionMatrix_3x4_w2p = K @ E[:3,:]` in world→camera form; inverting again produces massive projection errors (degenerate 3D cuboids and SV3DT track termination). The env flag keeps the behavior auditable if a different calibration source provides `Twc` instead.
- **References:** `noesis/ds8_runtime.py`, `scripts/generate_v3dt_caminfo.py`, `plans/DS8/v3dt/README.md`, `plans/DS8/v3dt/oom_killed_infer_v3dt_debug.md`, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
  _2026-01-12 (Codex): Updated default to `NOESIS_V3DT_CAMINFO_INVERT_E=0` after Menon confirmed it sends `E` as world→camera for all cameras._

- **Date:** 2025-12-28
- **Author:** Codex
- **Area:** Tracking / SV3DT
- **Decision:** Tune SV3DT for robustness/accuracy by (1) enabling `TrajectoryManagement.enableReAssoc` and `ReID` (tracker-internal ReID-based re-association), (2) switching to cascaded association matching, (3) raising `minDetectorConfidence` to suppress low-confidence jitter/ghosts, and (4) running BodyPose3DNet periodically (`poseInferenceInterval=4`) instead of first-frame-only height initialization.
- **Rationale:** The observed failure modes (jitter, fragmentation, ghosts) are classic symptoms of low-confidence detections creating tentative targets, overly strict frame-to-frame association, and insufficient re-association after occlusion. NVIDIA’s SV3DT reference config includes both trajectory re-association and a dedicated ReID module; enabling these and tightening lifecycle thresholds is the most direct “heavy but stable” configuration to explore the tracker’s ceiling before optimizing cost.
- **References:** `config/v3dt/nvtracker_sv3dt.yml`, `config/v3dt/nvtracker_sv3dt_sample.yml`, `/tmp/deepstream_reference_apps/deepstream-tracker-3d/configs/config_tracker_NvDCF_accuracy_3D.yml`, `/tmp/deepstream_reference_apps/deepstream-tracker-3d/README.md`

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Telemetry / BEV / UI
- **Decision:** Add a BEV dual-trail mode that publishes `footpoints_alt` alongside primary `footpoints` (typically SV3DT vs bbox) and renders both as light/dark variants of the same hue without duplicating the head dot; expose a `dual_trail_visualization_enabled` toggle via WebSocket and the settings panel.
- **Rationale:** Comparing SV3DT and bbox paths side-by-side helps validate 3D tracking quality without changing external telemetry contracts; using the same hue with light/dark variants keeps identity consistent while avoiding visual clutter. Keeping the dot to a single primary path prevents confusing “double heads” while still showing both trails.
- **References:** `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`, `websocket_server.py`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2025-12-28
- **Author:** Codex
- **Area:** Tracking / Telemetry / V3DT
- **Decision:** Prefer `NVDS_OBJ_WORLD_FOOT_LOCATION` for BEV/world footpoints when SV3DT/MV3DT is enabled; extend the V3DT meta bridge to extract `world_foot` (float[2]) and `visibility` and plumb them through telemetry. Add a DS8 runtime preflight that regenerates `config/v3dt/camInfo_*.yml` from `config/cameras.yaml` + `config/camera_calibration.json` when a V3DT tracker config is selected.
- **Rationale:** DeepStream already computes an explicit ground-plane footpoint (`ptWorldFeet`), which is a better BEV input than inferring footpoint from the 3D bbox centroid (less jitter/out-of-room drift). CamInfo files are static inputs to SV3DT/MV3DT; if calibration changes (e.g., auto-calibrate updates `camera_calibration.json`) but camInfo files are not regenerated, 3D outputs can silently become inconsistent—especially harmful for MV3DT fusion. Auto-sync keeps runs reproducible and avoids “it worked yesterday” drift.
- **References:** `native/noesis_v3dt_meta_ext.cpp`, `noesis/pipelines/hooks.py`, `noesis/ds8_runtime.py`, `/opt/nvidia/deepstream/deepstream-8.0/sources/gst-plugins/gst-nvtracker/nvtracker_proc.cpp` (meta payload types)

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Tracking / Telemetry / BEV
- **Decision:** Expose `NVDS_OBJ_3D_META` (`NvDsObj3DBbox`) to DS8 telemetry hooks via a small native bridge module (`noesis_v3dt_meta_ext`) that uses Service Maker C++ metadata iterators (Python `ObjectMetadata` does not expose `obj_user_meta_list`). Emit additive `bbox3d`/`velocity3d` fields and derive a best-effort `world` footpoint from the 3D bbox; when present, BEV uses SV3DT world coordinates instead of 2D bbox ray-plane intersections. Add an optional `world_frame` label from config (`camera_local` until global calibration is available).
- **Rationale:** SV3DT provides higher-quality 3D estimates and robust footpoints without introducing CPU branches. Because Service Maker Python bindings do not expose arbitrary per-object user meta, a minimal pybind11 extension is the cleanest in-process way to read `NVDS_OBJ_3D_META` without introducing `nvmsgconv` or repurposing preprocess metadata. The world-frame label keeps cross-camera comparisons guarded until shared global extrinsics are provided.
- **References:** `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `native/noesis_v3dt_meta_ext.cpp`, `scripts/build_noesis_v3dt_meta_ext.sh`, `plans/DS8/v3dt/integration_plan.md`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Tracking / Calibration / IDs
- **Decision:** Integrate SV3DT on all DS8 cameras for 3D state estimation and occlusion-robust tracking; enable MV3DT only for true-overlap “vision neighbor” cameras (kitchen ↔ family-room); keep `stable_id` as the only user-visible identity and treat MV3DT’s global tracker ID as an internal `mv3dt_id` hint/constraint; require a shared, metric, Y-up global calibration before enabling MV3DT.
- **Rationale:** SV3DT improves single-camera robustness (priority A). MV3DT is designed for overlapping FoVs and can mis-associate across non-overlap cameras, so restrict its neighbor graph initially. Preserving the stable-id-only external contract avoids UI/WS breakage while still leveraging MV3DT’s strongest capability (cross-camera ID propagation + multi-view fusion) as an internal signal. MV3DT fundamentally depends on a shared global world frame, so calibration must be solved early.
- **References:** `plans/ds8/v3dt/README.md`, `plans/ds8/v3dt/research_notes.md`, `plans/ds8/v3dt/integration_plan.md`, `plans/ds8/v3dt/codex_agent_prompt.md`, `plans/DS8/ds8_id_contract_v2.md`, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Metadata / Calibration
- **Decision:** In DS8, distribute intrinsics via the runtime `calibration-bundle` (and `_CalibrationProvider.snapshot()` for BEV) instead of attaching intrinsics as per-frame user meta; disable `attach_intrinsics_hook` by default in `noesis/ds8_runtime.py`.
- **Rationale:** Service Maker `FrameMetadata` in our current bindings does not expose a supported writable user-meta surface for custom payloads, which caused repeated debug logs and no effective attachment. DS7 semantics already treat intrinsics as calibration data rather than per-frame metadata; DS8 already broadcasts `calibration-bundle` and uses it for BEV/depth consumers.
- **References:** `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/metadata/intrinsics.py`, `docs/DS8_metadata_contracts.md`, `plans/DS8/ds8_migration_checklist_hooks.md`

- **Date:** 2025-12-26
- **Author:** Codex
- **Area:** Depth / BEV / UI
- **Decision:** Render floorplan/BEV canvases with letterboxed “contain” scaling (no stretch) and map overlays/footpoints to the content rect; scale intrinsics to the depth raster with aspect-preserving resize + padding before generating floorplan grids.
- **Rationale:** Fixed-aspect canvases were stretching the floorplan footprint (wide/obtuse look). Depth-derived X/Z needs intrinsics aligned to the actual depth raster resolution (including streammux padding) to avoid horizontal inflation.
- **References:** `oai2-fe/src/lib/renderUtils.ts`, `oai2-fe/src/components/BevView.tsx`, `oai2-fe/src/components/DepthDrawer.tsx`, `geometry/depth_source.py`

- **Date:** 2025-12-24
- **Author:** Codex
- **Area:** Calibration / WebSocket RPC
- **Decision:** Auto-calibrate-all in DS8 derives the camera list from `config/cameras.yaml` labels (source-id order) instead of relying on `auto_calibrate_from_latest_depth` defaults.
- **Rationale:** The calibration script defaults are static (`DEFAULT_CAMERAS`) and can drift from the DS8 runtime configuration. Using the DS8 camera labels ensures the calibrate-all button targets the configured sources without DS7 fallbacks.
- **References:** `noesis/ds8_runtime.py`, `scripts/auto_calibrate_from_depth.py`, `config/cameras.yaml`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2025-12-25
- **Author:** Codex
- **Area:** Telemetry / Occupancy / UI
- **Decision:** When DS8 tracking telemetry cannot infer a per-object zone from nvdsanalytics `roiStatus`, fall back to using the camera/stream name as the track `zone` (and therefore the occupancy “room” key and `dwell_time` basis). Preserve `roiStatus`-derived zones when they exist.
- **Rationale:** The current DS8 analytics config commonly enables `overcrowding`/`direction_detection` but not `roi-filtering`, so `roiStatus` is often absent and the UI showed `Zone: -` and `0.0s` dwell for all tracks. The fallback keeps the dashboard’s Active Tracks + Occupancy sections useful without requiring an immediate analytics config refactor.
- **References:** `noesis/pipelines/hooks.py`, `config/config_nvdsanalytics_post.ini`, `oai2-fe/src/App.tsx`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2025-12-25
- **Author:** Codex
- **Area:** Telemetry / Occupancy
- **Decision:** Add an occupancy grace window so zones are not vacated immediately when object tracks drop out briefly; implement as stable-id last-seen tracking with `NOESIS_OCCUPANCY_GRACE_S` (default 1.0s, clamp 0–10s).
- **Rationale:** Temporary detector/tracker occlusions caused occupancy to flicker (drop to 0 then rebound). A small grace window provides a smoother, more “human” occupancy signal without changing the DeepStream analytics configuration.
- **References:** `noesis/pipelines/hooks.py`, `plans/DS8/ds8_migration_checklist_hooks.md`, `oai2-fe/src/App.tsx`

- **Date:** 2025-12-24
- **Author:** Codex
- **Area:** IDs / Telemetry / OSD
- **Decision:** Make `stable_id` the only user-visible identity across DS8 (mosaic OSD labels, WS/REST tracking payloads, and `bev-frame` footpoints). Keep `track_id` internal-only. Ensure StableIDManager never emits negative/provisional IDs by returning a positive pending stable ID during new-track hysteresis, and add a telemetry-level fallback stable-ID allocator if StableIDManager is unavailable/unhealthy. Stamp mosaic OSD labels inside the analytics telemetry hook (pre-tiler) and format as `"label sid <stable_id> <confidence>"` to keep mosaic/BEV/Active Tracks IDs consistent across all cameras.
- **Rationale:** User-facing tracker IDs caused confusing mismatches and collisions across cameras; tiler-stage metadata can collapse camera context, so stamping per-source before tiler keeps IDs correct. Positive-only stable IDs eliminate `XX` placeholders and prevent leaking raw tracker IDs.
- **References:** `noesis/pipelines/hooks.py`, `reid/stable_id_manager.py`, `oai2-fe/src/App.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** Analytics / ROI Editor
- **Decision:** Implement the ROI editor as a header chip that opens a right-side drawer with a solo-tile zoom view derived from the live WebRTC mosaic, editing exclusion polygons in the exclude stage pixel space (`config_width/config_height`). The UI relies on additive `stats.payload.pipeline.mosaic_layout` metadata for tile mapping and uses REST `/api/v1/analytics/rois` for apply/reload; v1 focuses on polygon editing only (no inverse/class-id controls).
- **Rationale:** Keeps ROI editing aligned with DS8’s canonical exclusion path (`nvdsroiexclude`) and avoids guessing tiler layout or GStreamer internals in the UI. Using the mosaic video for preview ensures the editor stays in sync with the running pipeline without introducing new appsinks or CPU branches.
- **References:** `noesis/ds8_runtime.py` (stats payload + CORS), `noesis/server/analytics_api.py` (ROI REST + INI sync), `oai2-fe/src/components/RoiEditorDrawer.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2025-12-21
- **Author:** Codex
- **Area:** Pipeline / ReID / StableID
- **Decision:** Drive StableID from the dedicated OSNet ReID SGIE (`reid_osnet`, `gie_id=3`, batch=16) and consume embeddings in the telemetry hook (probe attached to the ReID element) using a manual DLPack→cudaMemcpy decode (avoid torch dlpack consumer). Keep StableID people-only and allow the same StableID to be active on multiple cameras (overlapping FoVs).
- **Rationale:** Torch `from_dlpack` consumption of Service Maker tensor outputs triggered a native double-free/segfault in `TensorOutputUserMetadata::getLayers()` during DS8 runs. A manual DLPack decode that copies the embedding to host memory (without transferring ownership) avoids the crash while keeping the ReID path GPU-first. Attaching telemetry at the ReID element ensures tensor meta is still valid at the point of access.
- **References:** `noesis/pipelines/ds8_pipeline.py`, `noesis/pipelines/hooks.py`, `reid/stable_id_manager.py`, `pipelines/config_infer_secondary_reid_osnet.ini`, `config/infer.yaml`, `scripts/reid_stable_id_smoke_test.py`

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** ReID / StableID
- **Decision:** Implement appearance-based StableID in DS8 by adding a dedicated per-object ReID SGIE (OSNet) after the tracker (`tracker → reid_sgie → analytics`) with `output-tensor-meta=1`, then consuming the embedding tensor (`features`) inside `_AnalyticsTelemetryProcessor` to drive `StableIDManager.update(..., embedding=...)`. Gate StableID creation on embeddings (`require_embeddings=true`) to avoid allocator-only IDs when ReID outputs are missing.
- **Rationale:** DS7’s StableID intention was OSNet-based global IDs, but it lacked a robust per-object embedding source in the pipeline. DS8 can keep the entire path GPU-first (crop/resize/infer on GPU), update galleries with multiple embeddings over time, and preserve the “live feel” by running ReID at a controlled cadence (`secondary-reinfer-interval`) with async SGIE output.
- **References:** scripts/export_reid_osnet_to_onnx.py (exports OSNet-IBN MSMT17 with ImageNet normalization in-graph + dynamic batch), pipelines/config_infer_secondary_reid_osnet.ini (SGIE settings: batch-size=16, output-tensor-meta=1), config/infer.yaml (models.reid + reid.* knobs), noesis/pipelines/ds8_pipeline.py (reid SGIE insertion), noesis/pipelines/hooks.py (embedding extraction + OSD label append), reid/stable_id_manager.py (gallery/ghost matching).
  _2025-12-20 (Codex): Superseded by the 2025-12-21 SGIE-driven StableID approach above (manual DLPack decode; no nvtracker ReID meta dependency)._

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** ReID / StableID
- **Decision:** Use NvDeepSORT’s built-in ReID path inside `nvtracker` with an OSNet ONNX+TensorRT engine (normalization/L2 inside ONNX) and `outputReidTensor: 1`, then feed StableIDManager from `NVDS_TRACKER_OBJ_REID_META` via pyds traversal in `_AnalyticsTelemetryProcessor` (`reid.source=tracker`). Keep the standalone ReID SGIE disabled by default.
- **Rationale:** Avoids Service Maker tensor-meta dereference instability and SGIE tensor-meta pool tuning, keeps ReID tightly coupled to tracking/association, and provides a single authoritative embedding source for cross-camera StableID while remaining GPU-first.
- **References:** config/nvtracker.yaml (NvDeepSORT + OSNet ReID settings), config/infer.yaml (`reid.source=tracker`, `models.reid.enable=false`), noesis/pipelines/hooks.py (`_extract_tracker_reid_embedding_pyds`), /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml (reference), /opt/nvidia/deepstream/deepstream/sources/tracker_ReID/README (tracker ReID model config).
  _2025-12-21 (Codex): Superseded by the SGIE-driven StableID + tee-after-analytics approach due to Flow Buffer/pyds bridge instability (SIGSEGV) and difficulty reliably accessing `NVDS_TRACKER_OBJ_REID_META` from Service Maker metadata._

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Telemetry / Occupancy
- **Decision:** Derive DS8 occupancy counts inside `_AnalyticsTelemetryProcessor` from analytics ROI zone labels and publish via `pipeline.occupancy_publisher`, emitting vacate events when a zone disappears; track telemetry carries `stable_id` (nullable) and per-zone `dwell_time` derived from per-track entry timestamps to maintain DS7-compatible schema.
- **Rationale:** Keeps occupancy and dwell semantics aligned with DS7 without adding DS7 pad probes; uses DS8 batch metadata already available in the analytics operator and preserves WS payload shape stability even when REID is disabled.
- **References:** docs/DS8_metadata_contracts.md, noesis/pipelines/hooks.py

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Gating
- **Decision:** Defer DS8 BufferOperator-based depth gating because `pyservicemaker.flow.BufferOperator` is unavailable in the installed DS8 package; keep depth-enabled stats and MapAnything processor-side gating, and surface a warning when gating cannot be attached.
- **Rationale:** Avoid guessing undocumented Flow APIs; gating requires a documented BufferOperator attach point. Current DS8 build lacks the class, so upstream drop control cannot be implemented safely.
- **References:** noesis/pipelines/ds8_pipeline.py (depth_gate_attach/depth_gate_supported), plans/DS8/ds8_migration_checklist_ds8_pipeline.md
  _2025-12-16 (Codex): Superseded by the 2025-12-14 decision to use an upstream GStreamer `valve` gate for physical SGIE compute savings + startup preroll stability._

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Gating
- **Decision:** Keep depth gating logical-only until Service Maker exposes a supported attach path for `BufferOperator`—the class exists, but `Pipeline.attach`/`Node.attach` only accept Probe/Receiver/Feeder, so gating cannot be wired without undocumented calls.
- **Rationale:** API inspection shows BufferOperator cannot be attached via documented Pipeline/Flow interfaces; forcing attachment would require private `_instance` hooks. Depth enable continues to gate MapAnything processing logically and via REST timers until NVIDIA exposes an operator attach interface.
- **References:** /home/mayor/.local/lib/python3.12/site-packages/pyservicemaker/pipeline.py, noesis/pipelines/ds8_pipeline.py, plans/DS8/ds8_migration_checklist_ds8_pipeline.md
  _2025-12-16 (Codex): Superseded by the 2025-12-14 `mapanything_valve` gating approach (canonical for GPU baseline reduction)._

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / MapAnything branch
- **Decision:** Terminate the MapAnything SGIE branch at a dedicated `fakesink` instead of feeding it into the tiler, keeping the branch metadata-only and avoiding multiple consumers on a single OSD pad.
- **Rationale:** MapAnything outputs are consumed via BatchMetadataOperator (tensor meta only); linking to tiler introduced invalid multi-sink linking without a tee and doubled frames in mosaic. A drop sink preserves tensor processing while keeping the mosaic path clean.
- **References:** noesis/pipelines/ds8_pipeline.py (mapanything_fullframe_sink, sink_tee), plans/DS8/ds8_master_work_orders.md

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Mosaic output
- **Decision:** Provide mosaic output as an RTSP stream (`nvrtspoutsinkbin` `rtsp_out`) consumed by the RTSP→WebRTC gateway; WebRTC is the canonical mosaic delivery path.
- **Rationale:** Keeps mosaic delivery off the WebSocket binary path and avoids brittle in-pipeline retrieval mechanisms.
- **References:** noesis/pipelines/ds8_pipeline.py (`rtsp_out`), noesis/mosaic_webrtc_gateway.py, noesis/ds8_runtime.py

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Depth gating
- **Decision:** Depth gating will be implemented via a `DepthGateOperator(BufferOperator)` wrapped in a `Probe("depth_gate", DepthGateOperator(pipeline))` attached at `pipeline.depth_gate_attach` (MapAnything SGIE). `Pipeline.attach` takes the Probe, not the operator directly.
- **Rationale:** Matches DS8 docs (FrameSkipper pattern) and avoids unsupported attachments; current runtime remains logical-only gating until this operator is integrated.
- **References:** pyservicemaker Probe/BufferOperator docs, plans/DS8/ds8_migration_checklist_flow_retrievers.md, plans/DS8/ds8_master_work_orders.md

- **Date:** 2025-11-29
- **Author:** Codex-patcher
- **Area:** Pipeline / MapAnything
- **Decision:** Added a dedicated MapAnything `nvinfer` config (`pipelines/config_infer_secondary_mapanything.ini`) that expects fused 12-channel tensor meta, enables `input-tensor-from-meta`, and pins the engine path; `models.mapanything` now references this config via `config-file-path` with an absolute engine to avoid mis-resolving under `config/`.
- **Rationale:** DS8 `nvinfer` refused to start without a config file; aligning with the fused-input design from `mapanything_preprocess_fused` avoids CPU pre-processing guesses and keeps engine resolution deterministic, even though the current plan file targets an older TensorRT build.
- **References:** pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml, pipelines/mapanything_preprocess_fused/README.md

- **Date:** 2025-11-29
- **Author:** Codex-build
- **Area:** Pipeline / MapAnything Engine
- **Decision:** Rebuilt MapAnything TensorRT engine (`ma_model_fp16_b3_fused.plan`) from ONNX model (`ma_onnx_out_fused/model_fused_sim.onnx`) using TensorRT 10.13.0 to resolve version mismatch with DS8 nvinfer. Engine configured with batch size 3 (min=1, opt=3, max=3), FP16 precision, and input shape `mapanything_fused:3x12x518x518`.
- **Rationale:** The existing engine was built with an older TensorRT version incompatible with TensorRT 10.13.0 used by DS8. Rebuilding ensures compatibility and allows DS8 to load the engine without version errors. Used fixed batch size 3 to match DS8 pipeline configuration (`batch_size: 3` in `config/infer.yaml`).
- **References:** export_ma_onnx/export_to_onnx.py, pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml, /tmp/trtexec_build.log

- **Date:** 2025-11-29
- **Author:** Opus
- **Area:** Hooks / DS8 API Compatibility
- **Decision:** Added DS8 pyservicemaker API compatibility layer to hooks.py operators. DS8 uses `batch_meta.frame_items` (iterable) instead of `batch_meta.frame_meta_list` (linked list), `frame_meta.object_items` instead of `obj_meta_list`, and `tensor_meta.get_layers()` → `np.from_dlpack()` for tensor extraction.
- **Rationale:** DS8 pyservicemaker BatchMetadataOperator receives pyservicemaker.BatchMetadata objects with different API than pyds. Each operator now checks for `frame_items` attribute (DS8 path) before falling back to pyds linked-list iteration (DS7 fallback). MapAnythingProcessor.handle_nvds_tensor_ds8 uses get_layers()+DLPack for tensor conversion.
- **References:** noesis/pipelines/hooks.py (_IntrinsicsOperator, _MapAnythingOperator, _AnalyticsTelemetryOperator, _ExcludePruneOperator)

- **Date:** 2025-11-29
- **Author:** Opus
- **Area:** Pipeline / Validation Results
- **Decision:** DS8 pipeline validation completed with partial success. Pipeline builds/prepares/activates correctly; PGIE and MapAnything SGIE engines load from TensorRT 10.13 plans; tracker and analytics initialize; 3 RTSP sources connect; REST API responds correctly. Remaining issues at the time: (1) MapAnything depth FPS=0 because SGIE expects 12-channel fused tensor from preprocess but current preprocess produces 3x640x640 for PGIE; (2) DS8 object_items is read-only so exclusion prune cannot remove objects.
- **Rationale:** Captures validation state for Phase 7 progress. MapAnything required full-frame SGIE mode and proper gating; exclusion should leverage analytics ROI filtering instead of metadata modification.
- **References:** plans/DS8/ds8_migration_checklist_ds8_pipeline.md, plans/DS8/ds8_migration_checklist_hooks.md, plans/DS8/ds8_migration_checklist_ds8_runtime.md

- **Date:** 2025-11-30
- **Author:** Codex
- **Area:** Pipeline / MapAnything
- **Decision:** Use the existing DS7 full-frame MapAnything TensorRT engine (RGB image input) for the DS8 MapAnything branch instead of the fused 12-channel tensor-from-meta engine. DS8 `nvinfer` for MapAnything should consume images directly (no `input-tensor-from-meta` in its config) and load the same engine file used by the legacy DS7 pipeline, keeping semantics aligned with the external MapAnything service path.
- **Rationale:** The fused 12-channel engine required a dedicated tensor-producing preprocess stage and diverged from the legacy behavior, where MapAnything consumes full-frame BGR images and performs its own fusion internally. Reusing the DS7 full-frame engine simplifies the DS8 pipeline, avoids tensor-from-meta complexity, and keeps the depth path consistent with the existing deployment, while still benefiting from DS8 Service Maker integration.
- **References:** config/infer.yaml (models.mapanything), pipelines/config_infer_secondary_mapanything.ini, deepstream_video_pipeline.py (MapAnything branch), docs/reference/DEPTH_STACK_FLOW_V2.md, docs/reference/Pipeline_Architecture_Diagram.md

- **Date:** 2025-11-30
- **Author:** Codex
- **Area:** Pipeline / MapAnything Engine
- **Decision:** Pin the DS8 MapAnything SGIE to `/home/mayor/Noesis_Devel/models/mapanything_depth/1/model.plan` (max batch 3, FP16, image input 3x518x518 RGB) with `infer-dims=3;518;518`, `model-color-format=0`, and `input-tensor-from-meta=0`, keeping `output-tensor-meta=1` so depth tensors propagate via metadata.
- **Rationale:** Aligns DS8 execution with the DS7 full-frame engine artifact and removes the fused 12-channel tensor-from-meta dependency; preserves tensor meta emission for `MapAnythingProcessor` while ensuring nvinfer consumes GPU surfaces directly.
- **References:** config/infer.yaml, pipelines/config_infer_secondary_mapanything.ini, models/mapanything_depth/config.pbtxt

- **Date:** 2025-12-01
- **Author:** Codex
- **Area:** Pipeline / MapAnything Engine
- **Decision:** Re-exported MapAnything as a single-input ONNX (images only; intrinsics baked with fx=fy=1000, cx=W/2, cy=H/2) and rebuilt the FP16 TensorRT engine with TensorRT 10.13 (`models/mapanything_depth/1/model.plan`, batch 1–3, input `images`, output `depth`). DS8 configs now point at this engine with `input-tensor-from-meta=0` and `infer-dims=3;518;518`.
- **Rationale:** DS8 nvinfer documentation only guarantees a single input tensor; the two-input/fused paths depended on undocumented tensor-from-meta plumbing and mismatched preprocessing, leading to depth FPS=0 and TRT version mismatches. The rebuilt engine is compatible with the installed TensorRT and keeps MapAnything full-frame on-GPU.
- **References:** export_ma_onnx/export_to_onnx.py, ma_onnx_out_clean/model.onnx, models/mapanything_depth/1/model.plan, pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml

- **Date:** 2025-12-01
- **Author:** Codex
- **Area:** Hooks / MapAnything tensor processing
- **Decision:** Convert DS8 `TensorOutputUserMetadata` to numpy via torch DLPack fallback (explicit `__dlpack__(0)`), enforce finite-mask filtering, and fall back to wall-clock timestamps when DS8 FrameMetadata lacks epoch-based `buffer_pts`. Also relax gie_id filtering when necessary by converting tensor_items via `as_tensor_output()`.
- **Rationale:** Numpy cannot consume DS8 GPU tensors directly; torch DLPack conversion produces CPU arrays without guessing pyservicemaker internals. Finite masks avoid NaN min/max warnings, and wall-clock PTS prevents retention logic from pruning snapshots dated at 1970. Ensures depth_result telemetry and snapshot storage run in DS8 runtime.
- **References:** noesis/pipelines/hooks.py (MapAnythingProcessor, _MapAnythingOperator), plans/DS8/ds8_migration_checklist_hooks.md, plans/DS8/ds8_migration_checklist_depth_api.md

- **Date:** 2025-12-02
- **Author:** Codex
- **Area:** Telemetry / BEV
- **Decision:** Generate BEV status + JPEG directly from the analytics telemetry hook using calibration data (cameras.yaml intrinsics + camera_calibration.json extrinsics) and `BevRenderer`, instead of adding a new Flow/appsink retriever. Overlay drawing respects BEV config/overlay toggles, and BEV WS framing stays `len + b"bev:<cameraId>" + jpeg`.
- **Rationale:** Avoids introducing another appsink in DS8 (GPU-first guardrail) while still providing BEV frames to oai2-fe. Reuses existing analytics metadata traversal and calibration bundle to compute footpoints and publish BEV JSON + binary frames without CPU pad probes.
- **References:** noesis/pipelines/hooks.py (_AnalyticsTelemetryProcessor._publish_bev), noesis/telemetry/bev.py, noesis/ds8_runtime.py (_CalibrationProvider wiring, ws_server calibration_getter), plans/DS8/ds8_migration_checklist_flow_retrievers.md

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Mosaic / WebRTC
- **Decision:** Keep the RTSP→WebRTC bridge as a separate GI/GStreamer pipeline (`noesis/mosaic_webrtc_gateway.py`) and answer browser offers by (1) extracting the offered H.264 PT and setting `rtph264pay.pt`, (2) after `set-remote-description` completes, requesting/linking `webrtcbin`’s `sink_%u` pad and forcing its transceiver to `SENDONLY` with H.264 RTP caps, and (3) creating the SDP answer only after the sender is linked and at least one encoded frame has been observed. For reconnects, rebuild the gateway pipeline (new `webrtcbin`) per offer, and use a pad-blocking probe when relinking to avoid `not-linked` stream errors.
- **Rationale:** On this environment’s GStreamer (`webrtcbin` 1.24.2), generating an answer before a sender stream is bound can yield “ICE connected but black” (no RTP/decoded frames). Reusing a single webrtcbin across peer reconnections also leads to connected ICE/DTLS but no media. Rebuilding the gateway per offer plus safe relinking produces consistent `a=sendonly` answers and actual video delivery; validated with `scripts/webrtc_gateway_smoke_test.py` across sequential sessions and PT=96/PT=103.
- **References:** noesis/mosaic_webrtc_gateway.py, websocket_server.py, gst-inspect-1.0 webrtcbin

- **Date:** 2025-12-03
- **Author:** Planner
- **Area:** Depth RPCs (MapAnything / Floorplan)
- **Decision:** Align DS8 depth RPC behavior with DS7’s WebSocket contracts: DS8 runtime exposes `ma_depth_provider` and `floorplan_provider` on `WebSocketServer` that delegate to `geometry.depth_source.DepthStorageManager` to serve `ma_depth_response` and `floorplan_response` payloads identical to DS7’s, but backed by DS8’s on‑GPU MapAnything depth snapshots.
- **Rationale:** The oai2-fe depth drawer is driven via `get_ma_depth`/`ma_depth_response` and `get_floorplan`/`floorplan_response`. Providing DS7-compatible RPC payloads in DS8 removes the last DS7-only dependencies for depth/floorplan visualization.
- **References:** main.py (_ma_depth_provider), geometry/depth_source.py (DepthStorageManager.load_latest_depth, generate_topdown_floorplan), websocket_server.py (get_ma_depth/get_floorplan handlers), noesis/ds8_runtime.py (providers), plans/DS8/ds8_master_work_orders.md, plans/DS8/ds8_migration_checklist_ds8_runtime.md

- **Date:** 2025-12-06
- **Author:** Codex
- **Area:** Mosaic
- **Decision:** Standardize on RTSP→WebRTC gateway for mosaic delivery and remove non-WebRTC mosaic delivery approaches from DS8-facing documentation.
- **Rationale:** A single canonical mosaic path prevents doc drift and matches the deployed UI (WebRTC).
- **References:** noesis/pipelines/ds8_pipeline.py (`rtsp_out`), noesis/ds8_runtime.py (gateway startup), noesis/mosaic_webrtc_gateway.py, websocket_server.py (WebRTC signaling)

- **Date:** 2025-12-07
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC (PREFERRED PATH)
- **Decision:** Implement a decoupled **RTSP-to-WebRTC Media Gateway** (`noesis/mosaic_webrtc_gateway.py`) as the **preferred mosaic output path**. The gateway runs a separate GStreamer pipeline (`rtspsrc → rtph264depay → h264parse → rtph264pay → webrtcbin`) that consumes the DS8 RTSP mosaic output (`rtsp://127.0.0.1:8554/mosaic`) and serves it via WebRTC to browser clients. Signaling (offer/answer/ICE) flows through the existing `WebSocketServer`.
- **Rationale:** Direct `webrtcbin` integration into the DS8 pyservicemaker pipeline proved unreliable due to undocumented pad/transceiver behaviors. The decoupled gateway keeps the DS8 pipeline unchanged (focused on inference/analytics/RTSP output) while providing a clean, standard WebRTC endpoint. This approach:
  - Uses standard GStreamer elements with documented behavior
  - Decouples WebRTC complexity from the DS8 pipeline lifecycle
  - Allows independent debugging of media vs. signaling issues
  - Preserves existing RTSP output as a fallback
  - Successfully negotiates H.264 sendonly with proper SDP (codec and direction)
- **References:** noesis/mosaic_webrtc_gateway.py, noesis/ds8_runtime.py (webrtc_gateway startup), websocket_server.py (register_webrtc_gateway), oai2-fe/src/hooks/useWebRTCClient.ts, plans/ds8_mosaic_webrtc_plan.md

- **Date:** 2025-12-07
- **Author:** Codex
- **Area:** WebRTC / SDP Negotiation
- **Decision:** Configure `webrtcbin` transceivers with explicit **SENDONLY** direction and **H.264** caps after setting remote description (before creating answer). Frontend sets H.264 as preferred codec via `setCodecPreferences()` to ensure codec match. Payload type 103 used to match Chrome's H.264 offer.
- **Rationale:** Initial SDP answers showed `a=inactive` and VP8 codec despite H.264 being sent. The fix required: (1) browser sends H.264-first codec preferences, (2) gateway uses `get-transceiver` signal to access and modify transceiver direction after offer is processed, (3) PT 103 matched between `rtph264pay` and browser offer. This resolved the connection failure (ICE connected but peer connection stayed at "connecting").
- **References:** noesis/mosaic_webrtc_gateway.py (_on_set_remote_description_done), oai2-fe/src/hooks/useWebRTCClient.ts (setCodecPreferences), config/infer.yaml (mosaic_output.mosaic_webrtc_enabled)

- **Date:** 2025-12-09
- **Author:** Codex
- **Area:** Pipeline / Sources & Lifecycle
- **Decision:** For the DS8 path, configure `nvmultiurisrcbin` directly from `infer.yaml` (URI list + streammux settings) but disable its embedded Civetweb REST server by setting `port="0"` and align key behavior flags (`drop-pipeline-eos`, cache/sort/align) with the DS7 `noesis_multiurisrcbin.ini`. In `ds8_runtime`, call `ds8_pipeline.prepare(on_message=_psm_message_cb)` and treat non-success prepare return codes from `pyservicemaker.Pipeline.prepare` as hard failures.
- **Rationale:** DS7 relied on `noesis_multiurisrcbin.ini` and a dynamic REST port allocator to avoid port-9000 conflicts; DS8 initially used nvmultiurisrcbin defaults, which started Civetweb on port 9000 and could abort the process when that port was in use. DS8 does not need the REST management API because URIs are driven by YAML, so disabling it avoids clashes while keeping ingest semantics. Wiring the documented `on_message` callback into `prepare()` and checking its integer return code ensures Service Maker errors and EOS/state transitions are logged and prevent partially initialized graphs from running silently.
- **References:** noesis/pipelines/ds8_pipeline.py (multi_cfg/prepare), noesis/ds8_runtime.py (`_on_pyservicemaker_message`, `main`), pipelines/noesis_multiurisrcbin.ini, gst-inspect-1.0 nvmultiurisrcbin

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC Gateway
- **Decision:** Fix the RTSP→WebRTC gateway to (1) insert a leaky `queue` between `rtph264pay` and `webrtcbin` to avoid pre-negotiation backpressure stalls, (2) derive the H.264 RTP payload type from the browser’s SDP offer and set `rtph264pay.pt` accordingly (no hard-coded PT), and (3) move the gateway FPS probe to the `h264parse` output (encoded frames) instead of the RTP payloader output (packet-rate), avoiding Python/GIL starvation.
- **Rationale:** The previous gateway could stall after a single buffer and/or starve other Python threads (WS server, signal handling) because `webrtcbin` can block until negotiation and the RTP-level probe fires at ~kHz packet rates. Matching the offer’s PT is required by WebRTC SDP semantics, and queue isolation keeps media flowing even when peers are absent.
- **References:** noesis/mosaic_webrtc_gateway.py, websocket_server.py (webrtc_offer routing), plans/ds8_mosaic_webrtc_plan.md

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** Runtime / Shutdown
- **Decision:** Treat WebSocket startup failure as fatal in DS8 runtime, and re-assert `SIGTERM` default handling (terminate) to make `timeout(1)`-based debug runs reliable (avoid orphaned DS8 processes holding ports).
- **Rationale:** A running DS8 pipeline without a WS server breaks WebRTC signaling and UI telemetry, and orphan processes cause misleading “websocket doesn’t connect” regressions via port conflicts. Default SIGTERM termination avoids reliance on Python-level shutdown when GI/GStreamer callbacks can delay signal handling.
- **References:** noesis/ds8_runtime.py

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** Hooks / MapAnything Postprocess
- **Decision:** Make MapAnything postprocess non-blocking by cloning DS8 `Tensor` outputs and processing conversion/storage/telemetry on a background worker thread with a bounded queue (drop-on-full). Also treat non-epoch PTS values as invalid and fall back to `time.time_ns()` to avoid 1970-dated depth snapshots.
- **Rationale:** Inline Python tensor conversion/storage inside the Service Maker metadata operator caused a repeatable pipeline stall (“~1–20 frames then GPU 0% / no frames to UI”). A bounded async worker prevents backpressure from stalling the pipeline, and stronger PTS epoch detection keeps depth timestamps and retention behavior sane.
- **References:** noesis/pipelines/hooks.py (MapAnythingProcessor async queue + PTS fallback), noesis/ds8_runtime.py (NOESIS_MAPANYTHING_POSTPROCESS_ENABLED), .cursor/debug.log (gateway frame counts)

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** WebRTC / Codec Interop
- **Decision:** For the RTSP→WebRTC gateway, enforce H.264 `packetization-mode=1` in webrtcbin transceiver caps and re-inject SPS/PPS periodically via `h264parse config-interval=1` and `rtph264pay config-interval=1`.
- **Rationale:** Browsers often prefer/require `packetization-mode=1` for H.264 WebRTC, and late-join peers can end up “connected but black” if SPS/PPS were only present at RTSP startup. Periodic config insertion plus explicit packetization mode stabilizes playback.
- **References:** noesis/mosaic_webrtc_gateway.py, oai2-fe/src/hooks/useWebRTCClient.ts

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC Gateway

- **Date:** 2025-12-17
- **Author:** Codex
- **Area:** Hooks / Mosaic Overlay
- **Decision:** Implement DS8 mosaic motion trails as a `BatchMetadataOperator` probe attached upstream of `nvdsosd` (at `tiler`), rendering trails via Service Maker display metadata (`BatchMetadata.acquire_display_meta()` + `FrameMetadata.append()` + `pyservicemaker.osd.Line/Text`) with a time-window history PLUS a per-track point cap (`max_points_per_track`) and newest-first rendering to prevent “frozen heads” when display-meta capacity is reached; apply stride sampling + decimation, footpoint EMA smoothing, and age-based alpha fade; restrict to people class (`class_id=0`) and color by StableIDManager `stable_id` when available; expose multi-trail budgets via `max_segments_per_track`, `max_tracks`, and `max_display_metas`; also render the same trail history into per-camera BEV JPEGs in `BevRenderer` so trails are visible in the dashboard’s BEV views.
- **Rationale:** DS8’s canonical mosaic delivery is in-pipeline (tiler → osd → RTSP/WebRTC), so overlays must be rendered before `rtsp_out` without introducing CPU appsink branches or Flow retrievers. Time-window trails behave consistently across FPS; stable-id coloring gives visual continuity across re-tracks/cameras; locking ensures safe runtime toggling via WebSocket control messages.
- **References:** config/infer.yaml, noesis/pipelines/hooks.py, noesis/ds8_runtime.py, docs/reference/Deepstream 8 Structure
- **Decision:** Treat the RTSP→WebRTC bridge as a two-phase link: keep RTSP ingest running into a drain sink until a WebRTC offer arrives; after `set-remote-description` completes, request/link `webrtcbin`’s `sink_%u` pad and only then create an SDP answer.
- **Rationale:** Requesting/linking the sender pad before the offer caused “ICE connected but no video” (no RTP/decoded frames) because `webrtcbin` can answer `a=sendonly` without actually mapping a sender stream to the negotiated session. Deferring the link until after remote description ensures the sender stream is bound to the peer connection and media actually flows. Verified with `scripts/webrtc_gateway_smoke_test.py` for both PT=96 and PT=103 (browser-like PT).
- **References:** noesis/mosaic_webrtc_gateway.py, scripts/webrtc_gateway_smoke_test.py

- **Date:** 2025-12-23
- **Author:** Codex
- **Area:** Telemetry / UI Trails
- **Decision:** When `stable_id` is missing, use a camera-namespaced fallback color key (`(camera_index + 1) * 2^32 + track_id`) for frontend trails/legend and the DS8 mosaic overlay; keep stable_id-based colors global. Disable BEV JPEG binaries by default (meta-only mode), while preserving the existing BEV JPEG framing when enabled.
- **Rationale:** Namespaced fallback colors prevent cross-camera color collisions before stable IDs are available. Disabling BEV JPEGs reduces unnecessary CPU/bandwidth because the frontend uses cached floorplans + canvas overlays; opt-in preserves legacy behavior when needed.
- **References:** plans/DS8/ds8_trails_id_refactor_plan.md, plans/DS8/ds8_trails_id_refactor_work_order.md, oai2-fe/src/lib/camera.ts, oai2-fe/src/lib/trails.ts, noesis/pipelines/hooks.py, noesis/telemetry/bev.py, config/infer.yaml, docs/DS8_api_contracts_ws.md

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** WebRTC / Browser SDP Interop
- **Decision:** When parsing browser SDP offers that advertise multiple H.264 payload types, prefer `packetization-mode=1` and a constrained‑baseline `profile-level-id` (`42e0…`) when selecting the H.264 PT to mirror in the answer and configure `rtph264pay.pt`.
- **Rationale:** Some browser offers include multiple H.264 PTs for different profiles/packetization-modes; selecting the first H.264 PT (often `profile-level-id=42001f`) could trigger `webrtcbin` to generate an `a=inactive` answer for m=video (“ICE connected but no video”). Preferring constrained‑baseline improves interop and eliminates the inactive-answer failure mode for captured Chrome/Chromium-style offers.
- **References:** noesis/mosaic_webrtc_gateway.py (`_extract_h264_payload_type`), scripts/webrtc_gateway_browser_offer_replay_test.py, `.cursor/webrtc_offers/*.sdp`

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Mosaic Output / Encoded Mosaic Sizing
- **Decision:** Keep the encoded mosaic output fixed at 1920×1080 (for RTSP/WebRTC encoder compatibility) and enable `nvmultistreamtiler square-seq-grid=true` by default so the tiler selects a square layout (e.g., 3 sources → 2×2) and avoids stretching streams into tall/skinny tiles. Provide an escape hatch via `NOESIS_MOSAIC_TILER_SQUARE_SEQ_GRID=0` plus `NOESIS_MOSAIC_TILER_COLUMNS` / `NOESIS_MOSAIC_TILER_ROWS` for explicit layouts.
- **Rationale:** DS7 achieved correct per-stream aspect in a 3×1 layout by increasing mosaic width (e.g., 5760×1080), but that can exceed NVENC/V4L2 limits for the DS8 RTSP/WebRTC encoding path. If the output stays 1920×1080, a 1×N tiler layout distorts 16:9 streams (e.g., 1×3 → 640×1080 tiles). Square tiling preserves per-tile 16:9 within the fixed output at the cost of empty tiles.
- **References:** noesis/pipelines/ds8_pipeline.py (tiler config), deepstream_video_pipeline.py (DS7 tiler sizing), DEBUG_PROGRESS.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** WebRTC Gateway / Backpressure
- **Decision:** Move the gateway’s leaky queue to the H.264 access‑unit stage (`h264parse → queue(leaky) → rtph264pay`) and keep the RTP queue non‑leaky (`rtph264pay → queue → webrtcbin`), and set `h264parse/rtph264pay config-interval=-1` (SPS/PPS with every IDR).
- **Rationale:** A leaky, tiny queue placed *after* `rtph264pay` can drop individual RTP packets under negotiation/backpressure, corrupting keyframes. This produces the failure mode “ICE connected, bytesReceived increasing, framesDecoded=0 until much later” because the browser never receives an intact IDR to start decoding. Dropping whole frames before packetization avoids partial-keyframe corruption while still preventing RTSP ingest from stalling.
- **References:** noesis/mosaic_webrtc_gateway.py, docs/DS8_testing_guide.md, DEBUG_PROGRESS.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** Mosaic Output / Runtime Toggles
- **Decision:** Make the NVJPEG/appsink mosaic branch opt-in: `mosaic_output.jpeg_enabled` defaults to `false`, `ds8_pipeline` only constructs the NVJPEG/appsink chain when explicitly enabled, and `ds8_runtime` logs whether the branch was built/attached. RTSP/WebRTC toggles remain independent.
- **Rationale:** The NVJPEG/appsink path is deprecated and was impacting RTSP/WebRTC stability when left on by default. Keeping it disabled unless explicitly requested avoids unintended branch construction while preserving the ability to re-enable for tests.
- **References:** config/infer.yaml (mosaic_output.jpeg_enabled), noesis/pipelines/ds8_pipeline.py (branch gating), noesis/ds8_runtime.py (startup log).
  _2026-01-10 (Codex): Superseded — DS8 no longer supports mosaic NVJPEG/appsink output; `mosaic_output.jpeg_enabled` is removed and mosaic video is RTSP→WebRTC only._

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** Tracking / Stable IDs
- **Decision:** Feed `StableIDManager` with lightweight per-track synthetic patches (bbox-derived colors, bounded 256×256) instead of adding CPU appsink branches; fall back to `stable_id=track_id` when the manager returns `None`, and expose `NOESIS_REID_TEST_MODE=1` to inject a synthetic tracking sample for smoke tests.
- **Rationale:** Stable IDs must stay enabled (NOESIS_REID_ENABLED default on) without violating the GPU-first policy. Synthetic patches provide deterministic input for the manager and allow smoke tests to assert non-null/persistent IDs without adding full-frame CPU crops or backpressure.
- **References:** noesis/pipelines/hooks.py (_reid_crop_from_track, _maybe_assign_stable_id, NOESIS_REID_TEST_MODE), noesis/ds8_runtime.py (_build_stable_id_manager), scripts/reid_stable_id_smoke_test.py.
  _2025-12-20 (Codex): Superseded by ID Contract v2 (`plans/DS8/ds8_id_contract_v2.md`): DS8 must use real person crops/embeddings for global stable IDs; synthetic patches are removed/disabled and stable_id may be null until embeddings are available._

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** Identity / Stable IDs
- **Decision:** Adopt ID Contract v2: treat `stable_id` as the master, global person identity (cross-camera) used for all user-facing features; allow a single `stable_id` to be concurrently active on multiple cameras (overlapping FoVs); compute stable IDs from real person crops/embeddings (no synthetic patches), producing crops in-pipeline via Service Maker `BufferOperator` + DLPack (GPU crop, CPU ROI upload) and emitting `stable_id=null` when ReID is disabled/unavailable.
- **Rationale:** DS8’s current synthetic-patch feed prevents appearance-based matching, so stable IDs cannot be meaningfully global across cameras and cannot reliably survive tracker ID churn. Contracting stable IDs as primary makes UI joins/replay predictable and enables overlap deployments. Cropping on GPU and uploading only the ROI preserves the GPU-first policy without introducing new appsink/CPU branches.
- **References:** plans/DS8/ds8_id_contract_v2.md, reid/stable_id_manager.py, noesis/pipelines/hooks.py, noesis/pipelines/ds8_pipeline.py, docs/DS8_api_contracts_ws.md, docs/DS8_metadata_contracts.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** Depth / Floorplan RPC
- **Decision:** `DepthStorageManager.generate_topdown_floorplan` now returns a minimal zero-density payload (cached + persisted) when no valid points exist and honours `max_age_sec`/cache-only paths; calibration bundles are seeded onto the manager in `ds8_runtime` and reused for floorplan RPCs.
- **Rationale:** UI/WS clients expect a floorplan payload even when depth snapshots have no valid points; previously this returned errors like `no_points` or `missing_calibration`. A zero-grid response keeps RPCs successful, improves cache hit rate, and surfaces stale/missing data via bounds/point_count while avoiding restarts.
- **References:** geometry/depth_source.py (DepthStorageManager caches/fallback), noesis/ds8_runtime.py (_ds8_floorplan_provider, calibration seeding), scripts/floorplan_rpc_smoke_test.py.
- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Analytics / Exclusion
- **Decision:** Treat `config/nvdsanalytics.yaml` as the exclusion source of truth: ROI REST updates rewrite the YAML, regenerate `config/config_nvdsanalytics_exclude.ini`, refresh `pipeline.config` stages, and node.set both `analytics` and `analytics_exclude`; hooks re-read the live analytics YAML when extracting exclusion polygons.
- **Rationale:** The DS8 pipeline had been reading stale INI configs, so ROI REST edits never reached the running nvdsroiexclude element. Syncing YAML→INI and pushing runtime updates keeps exclusion behavior aligned with live ROI edits without a restart.
- **References:** noesis/server/analytics_api.py, noesis/pipelines/hooks.py, noesis/pipelines/ds8_pipeline.py

- **Date:** 2025-12-18
- **Author:** Codex
- **Area:** UI / BEV Visualization
- **Decision:** Draw BEV motion trails client-side in the dashboard `BevView` canvas using `bev-frame.footpoints` (and optional `stableId`) rather than baking trails into a frequently-updated BEV JPEG.
- **Rationale:** The dashboard BEV panels already render the floorplan height map client-side and should remain static/low-refresh; a canvas overlay provides smooth, multi-track trails without increasing JPEG encoding/bandwidth.
- **References:** oai2-fe/src/components/BevView.tsx, noesis/telemetry/bev.py, docs/DS8_api_contracts_ws.md

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Depth / MapAnything telemetry
- **Decision:** Store MapAnything depth snapshots under FE camera labels (with numeric fallback) and seed `DepthStorageManager.calibration_bundle` from `cameras.yaml`/calibration assets; WS depth/floorplan providers map numeric/name camera keys to return DS7-compatible payloads.
- **Rationale:** Depth drawer/floorplan requests are keyed by camera name, but DS8 was storing snapshots under numeric source ids without calibration, producing “no depth”/`missing_calibration` responses. Canonical camera ids plus a populated calibration bundle align DS8 responses with FE expectations.
- **References:** noesis/pipelines/hooks.py (MapAnythingProcessor camera labels), noesis/ds8_runtime.py (_ds8_ma_depth_provider/_ds8_floorplan_provider, calibration bundle wiring), geometry/depth_source.py

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Tracking / Stable IDs
- **Decision:** Enable `StableIDManager` in DS8 runtime (gated by `NOESIS_REID_ENABLED`) using bbox-only updates (no new CPU/appsink crops) and maintain/prune tracks each frame; tracking telemetry carries stable_id/dwell fields matching DS7 schema.
- **Rationale:** DS8 previously omitted stable ids, diverging from DS7 occupancy/telemetry semantics. Instantiating StableIDManager preserves schema parity while keeping the GPU-first guardrail; embeddings remain optional via the existing manager.
- **References:** noesis/ds8_runtime.py (_build_stable_id_manager, pipeline.stable_id_mgr), noesis/pipelines/hooks.py (_maybe_assign_stable_id/_maintain_stable_ids), plans/DS8/ds8_migration_checklist_ds8_pipeline.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** MapAnything / SGIE Gating
- **Decision:** Gate MapAnything SGIE compute with an explicit GStreamer `valve` in the DS8 graph (`main_tee → mapanything_queue → mapanything_valve → mapanything_fullframe`) and “prime” the branch at startup: briefly open the valve during activation, then close it when depth remains disabled (default). The prime window is configurable via `NOESIS_MAPANYTHING_GATE_PRIME_SECONDS` (default 1.0s).
- **Rationale:** The DS7 stack uses valve-based gating so MapAnything only runs on-demand. In DS8, BufferOperator-only gating was not reliably preventing SGIE compute and starting with the valve closed could stall the pipeline at ~1 frame (no WebRTC video; GPU ~0%) because the SGIE branch hadn’t prerolled. Priming allows SGIE pads/events to preroll, then the valve closes to keep GPU utilization near the DS7 baseline until depth is explicitly enabled.
- **References:** noesis/pipelines/ds8_pipeline.py (mapanything_valve, activate prime), noesis/ds8_runtime.py (activation via ds8_pipeline.activate), plans/DS8/ds8_migration_checklist_ds8_pipeline.md
  _2025-12-16 (Codex): Validated end-to-end: DS8 runtime starts with depth OFF by default (`NOESIS_DEPTH_ENABLE_SECONDS=0`), `/api/v1/depth/refresh?seconds=N` toggles the valve deterministically (logs `Depth gate toggled: enabled=… valve.drop=…`), GPU spikes only during the window, and oai2-fe shows depth heatmap + views while WebRTC mosaic stays connected._

- **Date:** 2025-12-16
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC vs NVJPEG
- **Decision:** Treat RTSP→WebRTC gateway as the canonical mosaic output; keep NVJPEG/appsink mosaic deprecated and default OFF, enabling it only explicitly for debugging or legacy JPEG consumers.
- **Rationale:** WebRTC is the primary UI path and avoids the fragility/backpressure risks of always-on NVJPEG/appsink branches. Keeping NVJPEG inert unless requested prevents unintended pipeline side-effects while preserving an escape hatch.
- **References:** config/infer.yaml (mosaic_output.*), noesis/pipelines/ds8_pipeline.py (RTSP branch + NVJPEG branch gating), noesis/mosaic_webrtc_gateway.py, scripts/webrtc_gateway_smoke_test.py
  _2026-01-10 (Codex): Superseded — DS8 no longer supports mosaic NVJPEG/appsink output; mosaic video is RTSP→WebRTC only._

- **Date:** 2025-12-15
- **Author:** Codex
- **Area:** MapAnything / ONNX export + TensorRT engine
- **Decision:** Export MapAnything ONNX via `MapAnything.forward()` (torch-only) and derive `depth/conf/mask` tensors for SGIE output; configure the SGIE to run at `infer-dims=3;294;518` with aspect-preserving padding, and build the TensorRT engine with fixed batch=3 (min=opt=max=3) to match the DS8 deployment’s `batch_size=3`.
- **Rationale:** Exporting via `MapAnything.infer()` included numpy-based postprocess and produced mostly-zero/sparse depth outputs in DS8 (black heatmap). On TensorRT 10.13, building the engine with dynamic batch 1–3 hit a Myelin internal error; fixed batch 3 is stable and matches the streammux batch size used by this repo’s DS8 config. 294×518 preserves 16:9 structure for 1080p sources while aligning with the model’s patch constraints.
- **References:** export_ma_onnx/export_to_onnx.py, pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml, scripts/ma_depth_rpc_smoke_test.py

- **Date:** 2025-12-17
- **Author:** Codex
- **Area:** Pipeline / Mosaic layout
- **Decision:** Default the DS8 tiler to an explicit 3-column by 1-row mosaic so the primary RTSP/WebRTC output matches the one-row view expected by the UI, while keeping the previous square-seq-grid behavior and the `NOESIS_MOSAIC_TILER_COLUMNS/ROWS` overrides available.
- **Rationale:** The UI now consumes one composite stream, so forcing a single row of three tiles avoids the tall skinny tiles from square-auto layouts and keeps the player’s `StreamPanel` from introducing extra padding or distortion; rows still auto-scale (via ceil(source_count/columns)) if more cameras are added, and the telemetry log `ds8_mosaic_tiler_config` now reports the fixed 3×1 grid for debugging.
- **References:** noesis/pipelines/ds8_pipeline.py (tiler config), oai2-fe/src/styles/app.css (stream overlay), plans/DS8/ds8_migration_checklist_ds8_pipeline.md

- **Date:** 2025-12-16
- **Author:** Codex
- **Area:** Telemetry / BEV calibration
- **Decision:** Auto-fit BEV extents now use camera-local ray hits (translated/rotated by camera pose) and intrinsics are scaled to the streammux resolution using camera specs/intrinsics models (with a principal-point fallback) to mirror DS7 behavior.
- **Rationale:** DS8 was mixing camera-local footpoints with world-frame extents and unscaled intrinsics (e.g., 720p models), pushing BEV points into the top-left corner and dropping most of the room coverage. Local extents plus resolution-scaled K keep BEV overlays centered and consistent across all cameras.
- **References:** noesis/telemetry/bev.py, noesis/ds8_runtime.py, config/camera_calibration.json, config/cameras.yaml, intrinsics.json
  _2026-01-27 (Codex): Superseded — BEV now stays in the calibration world frame to align with legacy tracker coordinates when global extrinsics are in use._

- **Date:** 2025-12-16
- **Author:** Codex
- **Area:** Depth / MapAnything alignment
- **Decision:** Undo MapAnything SGIE letterboxing by cropping symmetric padding and resizing depth/conf/mask to the streammux (camera) resolution before persistence, so stored depth snapshots line up with intrinsics and downstream floorplan/BEV consumers.
- **Rationale:** DS8 had been storing 294×518 model-space tensors while floorplan generation assumed full-res camera coordinates, producing an up-left shifted “wedge” and missing coverage. Aligning tensors back to the camera frame restores DS7-style geometry and keeps DepthStorageManager outputs in the same pixel space as intrinsics.
- **References:** noesis/pipelines/hooks.py (`MapAnythingProcessor._align_to_frame`), pipelines/config_infer_secondary_mapanything.ini (maintain-aspect-ratio/symmetric-padding), geometry/depth_source.py (floorplan projection uses intrinsics).

- **Date:** 2025-12-31
- **Author:** Codex
- **Area:** Calibration / Menon WebSocket Compatibility
- **Decision:** Accept Menon calibration identifiers sent as objects (e.g., `{"id","name"}`) by coercing them into a string camera id, and treat Menon legacy `{cmd:"calibrate"}` as a compatibility alias that runs depth-based auto-calibration only when not immediately preceded by `set_extrinsics` (otherwise it acts as an ACK to avoid overwriting Menon-provided extrinsics).
- **Rationale:** Menon frequently sends `{cmd:"calibrate"}` right after publishing `set_extrinsics`; without a guard this would trigger depth auto-calibration and overwrite the extrinsics that Menon is intended to own (home 3D model source-of-truth). Coercing structured camera ids avoids requiring immediate frontend changes while keeping DS8’s calibration storage keyed to `config/cameras.yaml` labels.
- **References:** websocket_server.py (`cmd:"calibrate"` handler + `set_extrinsics` timestamp guard), noesis/ds8_runtime.py (camera id resolution + `unknown_camera` guard), plans/DS8/ds8_migration_checklist_ds8_runtime.md, plans/DS8/ds8_migration_checklist_websocket_server.md

- **Date:** 2026-01-01
- **Author:** Codex
- **Area:** MapAnything / Pose Inputs
- **Decision:** When supplying camera poses to MapAnything, derive MapAnything `camera_poses` as `T_wc = inv(E)` from `config/camera_calibration.json` where `E` is stored as world→camera (column-major), then apply `config/ply_alignment.json` as a post-multiply `T' = M @ T_wc` (row-major), and apply `units.s_obj_to_m` to translation only (unit conversion without scaling rotation).
- **Rationale:** MapAnything expects OpenCV cam→world 4×4 poses; Noesis stores world→camera extrinsics as `E`. Alignment and unit scaling are needed for consistent "world" coordinates across consumers, but scaling must not be baked into the rotation block of a pose matrix. Keeping the conversion in one shared utility prevents convention drift when re-exporting/rebuilding pose-conditioned MapAnything models/engines.
- **References:** noesis/metadata/mapanything_pose.py, config/camera_calibration.json, config/ply_alignment.json, docs/DS8_MIGRATION_KNOWLEDGE_BASE.md

- **Date:** 2026-01-01
- **Author:** Codex
- **Area:** Calibration / Unified Conventions
- **Decision:** Standardize all DS8 calibration on the following conventions:
  - **E (extrinsics):** 4×4 world→camera, **column-major** (Fortran order), Y-up world, +Z forward camera, meters.
  - **K (intrinsics):** Loaded from `config/cameras.yaml` ONLY; `intrinsics.json` and `config.py` are deprecated.
  - **align.matrix:** 4×4 **row-major** (C order); Menon applies it client-side; DS8 `pixel_to_world` does NOT apply it.
  - **`set_extrinsics` inputs:** Accept both `E` (world→camera) and `Twc` (camera→world, inverted before storage).
  - **Streammux scaling:** K is scaled when source intrinsics resolution differs from streammux output.
- **Rationale:** Multiple calibration sources and inconsistent conventions caused K divergence between `calibration-bundle` and runtime snapshots. A single `CalibrationManager` owning all loading/validation/broadcasting eliminates this drift.
- **References:** plans/DS8/ds8_calibration_workflow_unification_work_order.md, docs/DS8_api_contracts_ws.md §8, noesis/calibration/manager.py

- **Date:** 2026-01-02
- **Author:** Codex
- **Area:** Calibration / Frontend Name Resolution
- **Decision:** Implement robust "fuzzy matching" resolution in Menon's `CalibrationManager` to map user-facing device names to canonical backend stream keys. Logic includes:
  1. Exact match.
  2. Case-insensitive match.
  3. Normalization (spaces to underscores/hyphens).
  4. Substring containment (stripping explicit "Camera" suffix/noise).
- **Rationale:** Users tend to name devices descriptively (e.g., "Family Room Camera") while the backend requires strict keys from `camera_calibration.json` (e.g., "family-room"). Enforcing strict naming in the UI is hostile; strict backend keys are necessary for config stability. The resolver bridges this gap transparently, logging the mapping for verification.
- **References:** Menon/src/features/calibration/CalibrationManager.js (`resolveCameraId`), noesis/config/camera_calibration.json

- **Date:** 2026-01-14
- **Author:** Codex
- **Area:** SV3DT / Distortion Handling
- **Decision:** For wide-angle cameras, undistort the video **before** PGIE/tracker using `nvdewarper`, and treat the dewarped output as pinhole (fx/fy/cx/cy preserved, distortion zeroed) for SV3DT camInfo generation and tilt fitting.
- **Rationale:** SV3DT camInfo is pinhole-only; leaving distortion unmodeled causes region-dependent tracking and undersized cuboids. `nvdewarper` is the DeepStream-supported GPU path for correcting lens distortion.
- **References:** `/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream-dewarper-test/README`, `config/dewarper_family_room_charuco_rtsp.txt`, `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml`

- **Date:** 2026-01-27
- **Author:** Codex
- **Area:** Telemetry / BEV / Coordinate frame
- **Decision:** Keep BEV footpoints and auto-fit extents in the calibration world frame (homography X/Z) and remove camera-local yaw/translation rotation. Continue using camera position only for max-distance gating.
- **Rationale:** The camera-local rotation introduced for V3DT caused BEV points to drift relative to legacy tracker world coordinates when global extrinsics are in use. World-frame BEV keeps tracking telemetry and BEV views aligned for non-V3DT runs.
- **References:** noesis/telemetry/bev.py, docs/DS8_metadata_contracts.md, plans/DS8/v3dt/work_order.md
## 2026-01-21: Bidnetpipe uses a custom segmentation parser for floor-only masks

- **Decision:** For the Bidnetpipe DS8 segmentation test pipeline, use a custom
  semantic segmentation parser (`NvDsInferParseCustomBiSeNetFloor`) so the
  `NvDsInferSegmentationMeta` class_map is generated as a binary floor vs
  background mask.
- **Rationale:** `nvinfer`/`nvsegvisual` do not provide a config-driven per-class
  filter for semantic segmentation masks, and host-side `class_map` rewrites were
  not reflected in `nvsegvisual`. A custom parser ensures the metadata itself is
  already filtered before visualization and downstream stats.
- **Docs/refs:** `nvdsinfer_custom_impl.h` for the parser signature; `gst-nvinfer`
  config keys `custom-lib-path` and `parse-segmentation-func-name` (see
  `Bidnetpipe/DOCS.md`).

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Telemetry / BEV / Image-axis alignment
- **Decision:** Auto-infer image-axis flips from extrinsics (right/up vs expected) so BEV footpoints are always camera-local with +X to image-right and +Z forward; update the FE to render BEV X without a hard-coded mirror.
- **Rationale:** V3DT-era calibrations introduced a 180° roll (X/Y sign change) relative to legacy extrinsics, which inverted BEV trails vs the stream. Auto-detecting axis flips keeps BEV aligned with the image/floorplan without requiring manual calibration edits.
- **References:** `noesis/telemetry/bev.py`, `geometry/homography.py`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Depth / Floorplan / Coordinate frame
- **Decision:** Generate floorplan bounds/grids in the same camera-local ground-plane frame as BEV (translate by camera position, rotate by yaw) and invalidate cached floorplans that lack the new `frame` marker.
- **Rationale:** V3DT-era calibrations shifted extrinsics to a shared world frame; floorplans were still generated in raw camera coordinates, causing BEV trails to appear scaled against the background. Aligning floorplan coords to BEV restores consistent overlays for default tracking.
- **References:** `geometry/depth_source.py`, `noesis/telemetry/bev.py`

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Calibration / BEV / Legacy extrinsics
- **Decision:** Use `config/camera_calibration_legacy.json` for default (non‑V3DT) tracking BEV/floorplan calibration, with `NOESIS_CALIBRATION_EXTRINSICS` as an explicit override and `camera_calibration.json` reserved for V3DT runs.
- **Rationale:** The V3DT calibration file introduces pose differences that exaggerate BEV depth for kitchen/living; default tracking is validated against the pre‑V3DT calibration set.
- **References:** `noesis/ds8_runtime.py`, `config/camera_calibration_legacy.json`
