# DS8 Migration Checklist – `noesis/pipelines/ds8_pipeline.py`

Concrete tasks for aligning `DS8Pipeline` with DeepStream 8 Service Maker Pipeline APIs and reproducing all DS7 pipeline behavior.

## 1. Pipeline API Alignment

- [x] Verify the import and usage of the Pipeline API matches DS8 docs:
  - [x] Use `from pyservicemaker import Pipeline` (or the documented equivalent) rather than private/undocumented modules.
  - [x] Use `pipeline.add(element, name, properties)` exactly as shown in the DS8 Pipeline API example (e.g., `"nvurisrcbin"`, `"nvstreammux"`, `"nvinfer"`).
  _2025-11-29 (Codex): Updated DS8 pipeline to import `Pipeline` from pyservicemaker and rely only on documented add/set interfaces._
- [x] Confirm there is no reliance on undocumented Pipeline methods; any `start`/`run`/`play` calls must be confirmed against docs or your installed package.
  _2025-11-29 (Codex): Lifecycle calls remain on documented start/run/play paths; no private APIs used._

## 2. YAML (`infer.yaml`) Schema & Normalization

- [x] Align `config/infer.yaml` with DS8 Inference Builder and analytics YAML schemas found in online docs:
  - [x] `sources`: element type, URI, GPU id, live-source, reconnect behavior.
  - [x] `streammux`: width, height, `batch-size`, `live-source`, `gpu-id`.
  - [x] `models`: PGIE, SGIE (MapAnything) with keys such as `config-file-path`, `engine`, `batch_size`, `unique-id`, and mode fields as documented.
  - [x] `analytics`: DS8 analytics YAML fields for `stages`, `streams`, `roi_filtering`, etc.
  - [x] `sinks`: definitions for mosaic and any other outputs, matching documented sink properties.
  _2025-11-29 (Codex): Rebuilt `config/infer.yaml` with live RTSP sources, streammux defaults, preprocess/tracker/analytics configs, PGIE/SGIE slots, and mosaic sink. PGIE now uses the same engine/parser as the working DS7 config (`models/engines/yolo11m_fp16.engine` + `libnvdsparsebbox_yolo11.so`) for parity._
  _2026-03-12 (Codex): Added a required `models.depth_tracking` baseline branch in `config/infer.yaml` for DAv2 tracking depth (`gie_id=5`, tensor-meta enabled, batch 3) and moved its INI/engine materialization into the shared runtime helper `noesis/depth_tracking_materialization.py` so baseline runtime and test pipeline stay aligned._
- [x] Provide a dedicated SV3DT config entrypoint (`config/infer_v3dt.yaml`) that points `tracker.config-file` at the SV3DT low-level tracker config.
  _2025-12-27 (Codex): Added `config/infer_v3dt.yaml` with `tracker.config-file: config/v3dt/nvtracker_sv3dt.yml` and V3DT metadata settings for camera-local world coords (validation pending)._
  _2026-01-23 (Codex): Renamed SV3DT baseline artifacts to `config/infer_v3dt_baseline.yaml`, `config/v3dt/nvtracker_v3dt_baseline.yml`, `config/v3dt/caminfo_baseline/`, `config/cameras_v3dt_baseline.yaml`, `config/archive/calibration_v3dt_baseline.json`, `config/dewarper_v3dt_baseline.txt`, and `config/analytics_exclude_baseline.ini`; updated all DS8 configs, scripts, and docs to match._
- [x] Keep the path normalization logic but ensure it follows DS8 recommendations:
  - [x] Resolve relative engine/config paths relative to the YAML file and/or repo root.
  - [x] Avoid guessing about path prefixes; use only documented top-level config locations.
  _2025-11-29 (Codex): Path resolution now normalizes engines/configs for nvinfer/preprocess/analytics to infer.yaml or repo root._
- [x] Provide DS8-ready PGIE assets for YOLO26 segmentation (INI template + custom parser lib) to support profile overlays.
  _2026-01-28 (Codex): Added `pipelines/config_infer_primary_yolo26_seg.template.ini` and `pipelines/nvdsinfer_yolo26_seg/` parser library so DS8 can materialize size-specific YOLO26 INIs at runtime._
  _2026-01-29 (Codex): Fused YOLO26-seg ONNX/engines to output `output0` only (mask compose on-GPU) and enabled `disable-output-host-copy=1`; updated parser to handle device pointers and consume fused output0; built b3 FP16 engines for `yolo26{n,s,m}` capped to 30 detections (match YOLO11) and validated DS8 runtime loads each size without output1/proto host copies. This cap drastically reduces ROIAlign cost and dropped YOLO26n GPU usage to ~10% in DS8 runs (from ~80–90% when composing 300 masks)._
  _2026-06-21 (Codex): Promoted the DEIMv2 Wholebody49 prototype into an optional DS8 PGIE profile (`--pgie-profile wholebody49 --size s|x`) with canonical copied ONNX/engine assets, production parser library, size-specific PGIE/preprocess materialization, and fail-fast preflight. Validation: `make -C pipelines/nvdsinfer_deimv2_wholebody49`, `python3 -m pytest tests/test_deimv2_wholebody49_assets.py -q`, `python3 -m compileall -q noesis tests/test_deimv2_wholebody49_assets.py`, and direct materialization/preflight for sizes `s` and `x`._
  _2026-06-30 (Codex): Added size-aware YOLO11 detect and YOLO11-seg PGIE support for `s|m|l`. Runtime now defaults to YOLO11 detect medium, materializes deterministic size-specific PGIE/preprocess configs, and preflights YOLO11-seg TensorRT plugin loading only when selected ONNX assets contain custom `EfficientNMSX_TRT` / `ROIAlignX_TRT` ops. Added `scripts/download_export_build_yolo11_profiles.py` to download/export/build the canonical YOLO11 detect/seg assets, and rebuilt `pipelines/nvdsinfer_yolo11_seg/libnvdsinfer_yolo11_seg.so` with a bounded confidence-sorted parser cap (`NOESIS_YOLO11_SEG_PARSER_TOPK`, default 30). Validation: `python3 -m compileall -q noesis/ds8_runtime.py scripts/download_export_build_yolo11_profiles.py`, `git diff --check`, `make -C pipelines/nvdsinfer_yolo11_seg`, `python3 -m pytest tests/test_ds9_parity_hardening.py::test_yolo11_seg_has_bounded_mask_topk -q`, TensorRT load/inference checks for all six engines, and direct DS8 materialization/preflight for `yolo11` and `yolo11_seg` sizes `s`, `m`, and `l`._

## 3. Graph Structure Parity with DS7

- [x] Ensure the DS8 graph reproduces the DS7 structure from `PIPELINE_GRAPH.md`:
  - [x] Sources via DS8 `nvmultiurisrcbin` (combined ingest + mux / `nvstreammux` equivalent).
  - [x] `nvdspreprocess` (if retained in DS8 graph) with its INI config.
  - [x] PGIE `nvinfer` (YOLO) with DS8-ready properties.
  - [x] Tracker (`nvtracker`) with `ll-lib-file` and `ll-config-file` from the DS7 configs.
  - [x] ROI exclusion (either via `nvdsroiexclude` + config, or analytics config) matching DS7 semantics.
  - [x] `nvdsanalytics` stage(s) configured to mirror `config_nvdsanalytics_post.ini` / DS8 analytics YAML.
  - [x] Tiled mosaic via `nvmultistreamtiler` and annotated display via `nvdsosd` with DS7 default/INI settings.
  - [x] Mosaic output via RTSP (`nvrtspoutsinkbin`) for the WebRTC gateway, plus optional sinks for future non-mosaic exports.
  - [x] Remove deprecated mosaic JPEG-over-WebSocket branch (`nvjpegenc` + `appsink`); mosaic video is WebRTC-only.
  _2026-01-10 (Codex): Removed the `nvjpegenc/appsink` mosaic branch from `noesis/pipelines/ds8_pipeline.py` and removed `output`/`mosaic_output.jpeg_enabled` from `config/infer*.yaml`._
  - [x] Treat `sinks.mosaic_sink` / `sinks.bev_sink` as placeholders (do not build fakesink branches) to avoid interfering with RTSP/WebRTC mosaic delivery.
  _2026-01-11 (Codex): `noesis/pipelines/ds8_pipeline.py` now skips building/linking these placeholder sinks; RTSP server responds correctly and `scripts/webrtc_gateway_smoke_test.py` receives `webrtc_answer` + decoded frames again._
  _2025-11-29 (Codex): Pipeline now builds sources→streammux→preprocess→PGIE→tee→(analytics_exclude→tracker→analytics)→tiler→osd→mosaic sink with MapAnything SGIE branch preserved. PGIE uses the DS7-aligned engine/parser to clear config parse issues._
  _2025-12-12 (Codex): Added queue isolation after `main_tee` and `sink_tee` (leaky on auxiliary branches) to prevent backpressure stalls; switched pre-tracker exclusion to `nvdsroiexclude` and attached depth gate upstream of MapAnything SGIE._
  _2025-12-13 (Codex): Fixed DS8 mosaic aspect distortion by enabling `nvmultistreamtiler square-seq-grid=true` by default (keeps per-tile aspect aligned to the 1920×1080 mosaic output, e.g. 3 sources → 2×2). Added env overrides: set `NOESIS_MOSAIC_TILER_SQUARE_SEQ_GRID=0` to disable and optionally set `NOESIS_MOSAIC_TILER_COLUMNS` / `NOESIS_MOSAIC_TILER_ROWS` for an explicit layout._
  _2026-01-06 (Codex): Enabled `nvdsosd display-bbox=1` in DS8 pipeline so bbox rectangles are drawn on the mosaic output (DS7 parity)._
  - [x] Default to an explicit 3-column, 1-row tiler layout (with columns/rows env overrides and square-grid opt-in) so the mosaic output renders as one row of three tiles and auto-expands rows only when extra sources arrive.
  _2025-12-17 (Codex): DS8 tiler now logs `ds8_mosaic_tiler_config` with `columns=3 rows=1` by default, and the UI overlay no longer draws a distracting horizontal divider while square-grid mode and explicit overrides remain available._
  - [x] Align `nvmultiurisrcbin` configuration (URI list, batch/latency settings, EOS behavior) with the DS7 `noesis_multiurisrcbin.ini` and disable its REST server in DS8.
  _2025-12-09 (Codex): DS8 multi-URI path now sets `drop-pipeline-eos`, cache/sort/align flags, and disables the embedded Civetweb REST API via `port=\"0\"`, avoiding port-9000 conflicts while keeping ingest behavior consistent with the DS7 INI._
  _2026-01-06 (Codex): Removed the deprecated `NOESIS_DS8_USE_NVURISRCBIN` source topology and always build `nvmultiurisrcbin`; updated smoke scripts/config example; validated via `python3 -m compileall -q noesis scripts`._
  _2026-01-06 (Codex): Auto-enable `nvmultiurisrcbin file-loop=true` when any input URI is `file://...*.mp4` (opt-out `NOESIS_DS8_LOOP_LOCAL_MP4=0`); validated by building `config/infer_v3dt_medium.yaml` and confirming `streammux.config['file-loop']==True` (and opt-out removes it)._
  _2026-01-13 (Codex): Re-removed the nvurisrcbin path and restored local MP4 auto-loop (`NOESIS_DS8_LOOP_LOCAL_MP4`); validated by building `config/infer_v3dt_medium.yaml` and confirming `streammux.config['file-loop']` toggles with the opt-out env var._
  _2026-01-20 (Codex): When dewarper forces per-source `nvurisrcbin`, set RTSP reconnect defaults (`rtsp-reconnect-interval/init-rtsp-reconnect-interval/rtsp-reconnect-attempts`) to match DS7 resiliency; validated via `gst-inspect-1.0 nvurisrcbin` property check._
  _2026-06-16 (Codex): Fixed the live family-room dewarper display by making `config/dewarper_g4_instant_charuco_720_to_1080.txt` use a full 1920×1080 dewarped surface instead of a 1280×720 surface scaled afterward; this keeps the black-border full-FoV policy while avoiding the top-left zoom/crop. Also aligned offline mirror helpers with nvdewarper semantics that `[surface0] width/height` are destination-surface dimensions. Validation: standalone `nvdewarper` RTSP frame capture and `python3 -m pytest tests/test_pipeline_build.py -q`._
  _2026-06-20 (Codex): Repaired a regression where the family-room G4 dewarper surface dimensions had drifted back to 1280×720 while the output caps stayed 1920×1080. Restored `[surface0] width/height` to 1920×1080, kept the offline depth-registration rectifier aligned with nvdewarper source-K semantics, and revalidated with a live family-room `nvdewarper` RTSP caps probe plus `python3 -m pytest tests/test_pipeline_build.py tests/test_depth_registration.py -q`._
  _2026-03-12 (Codex): Inserted deterministic post-pose stages (`world_observation_stage`, `tracking_telemetry_stage`) and an always-on baseline depth-tracking tee branch (`depth_tracking_queue -> depth_tracking_fullframe -> depth_tracking_fullframe_sink`) so pose+depth fusion runs before analytics telemetry/trails without relying on probe attachment order. Validation: `python3 -m pytest tests/test_pipeline_build.py -q`._

## 4. MapAnything / Depth Branch

- [x] Implement the MapAnything SGIE branch in the DS8 graph:
  - [x] Add an SGIE `nvinfer` node configured via `infer.yaml` with a `unique-id`/`gie_id` that matches `MapAnythingProcessor`.
  - [x] Verify that SGIE is linked in the graph after PGIE/tracker, or at the appropriate point defined by the MapAnything model’s requirements (check DS8 docs for SGIE best practices).
  _2025-11-29 (Codex): Enabled MapAnything SGIE in infer.yaml with `gie_id=2` and default `attach_tensor_meta`, and ensured ds8_pipeline builds the branch off the main tee with consistent unique-id._
- [x] Remove DS7-style pre-PGIE BGR MapAnything branches from DS8 graph; all depth should be SGIE-based.
  _2025-11-29 (Codex): DS8 graph retains only the SGIE branch (no BGR/appsink paths) and tees from PGIE into tracker/analytics and MapAnything._
- [x] Provide a physical SGIE gating mechanism so depth-disabled runs skip MapAnything SGIE compute.
  _2025-11-29 (Codex): DS8 pipeline now records `depth_gate_attach` pointing to the MapAnything SGIE node for future gating control._
  _2025-12-12 (Codex): Updated `depth_gate_attach` to point at the leaky queue upstream of MapAnything SGIE so depth-disabled runs can skip SGIE compute._
  _2025-12-14 (Codex): Implemented `mapanything_queue → mapanything_valve → mapanything_fullframe` and drive `valve.drop` from `DS8Pipeline.mark_depth_enabled` (GPU-saving gate); added a short startup “gate prime” to avoid preroll stalls (`NOESIS_MAPANYTHING_GATE_PRIME_SECONDS`)._
  _2025-12-16 (Codex): Fixed `Depth branch disabled; recent MapAnything FPS …` logging to compute FPS before clearing samples so it reflects the last enabled window (instead of always 0.00)._
- [x] Ensure MapAnything SGIE `nvinfer` is configured via `config-file-path` with the fused-input settings and a valid engine path.
  _2025-11-29 (Codex-patcher): Added `pipelines/config_infer_secondary_mapanything.ini`, pointed `models.mapanything` at it, and pinned the engine to `/home/mayor/Noesis_Devel/models/engines/ma_model_fp16_b3_fused.plan` to clear the gst-nvinfer "Configuration file not provided" warning; activation now fails earlier due to TensorRT 10.13 incompatibility with the existing engine (no ONNX fallback available in this repo)._
  _2025-11-29 (Codex-build): Rebuilt `ma_model_fp16_b3_fused.plan` from ONNX (`ma_onnx_out_fused/model_fused_sim.onnx`) using TensorRT 10.13.0 on GPU host (RTX 3060) and saved to `models/engines/`; engine built with batch size 3 (min=1, opt=3, max=3), FP16 precision, input shape `mapanything_fused:3x12x518x518`; build completed successfully in 348s, engine size 1082.3 MiB; DS8 mapanything nvinfer should now load the engine without version mismatch._
  _2025-11-30 (Codex): Swapped DS8 MapAnything config to the DS7 full-frame engine `/home/mayor/Noesis_Devel/models/mapanything_depth/1/model.plan`, removed `input-tensor-from-meta`, and set `infer-dims=3;518;518`, `model-color-format=0` (RGB). DS8 now consumes surfaces directly (image input) while still emitting tensor meta for depth processing; the fused 12-channel engine is no longer the target path._
  _2025-12-15 (Codex): Updated MapAnything SGIE to `infer-dims=3;294;518` with aspect-preserving padding; rebuilt `models/mapanything_depth/1/model.plan` from a forward-exported ONNX that emits `depth/conf/mask` (dynamic batch 1–3 hit a TRT Myelin error, so engine built fixed batch=3 to match `infer.yaml`); validated via `python3 scripts/ma_depth_rpc_smoke_test.py`._

## 4b. Pose SGIE (YOLO26)

- [x] Add a YOLO26 pose SGIE branch for per‑person crops, configured via `infer.yaml`, and keep tensor output on‑GPU for downstream feature extraction.
  _2026-01-30 (Codex): Added `pipelines/config_infer_secondary_yolo26_pose.ini`, `pipelines/nvdsinfer_yolo26_pose/` parser lib, and `models.pose` config; wired SGIE into DS8 pipeline chain (tracker→analytics→reid→pose→tiler) with tensor meta enabled and no CPU branches. Validated via `python3 -m compileall -q noesis scripts`._

## 4c. Baseline Depth-Tracking Branch (DAv2)

- [x] Add a separate always-on baseline depth-tracking lane for non-`v3dt` tracking, independent from the gated MapAnything branch.
  _2026-03-12 (Codex): Added `pipelines/config_infer_secondary_depth_tracking_da2.template.ini` plus `noesis/depth_tracking_materialization.py` to materialize a DAv2 `vits` FP16 TensorRT engine/config at `518x294`, batch 3, every 2 frames. `ds8_pipeline.py` now builds the baseline lane from `main_tee`, and `ds8_runtime.py` fails fast in baseline mode if the depth-tracking branch or `noesis_depth_meta_ext` cannot be materialized/imported. Validation: `python3 -m compileall -q noesis`, `python3 -m pytest tests/test_pipeline_build.py tests/test_analytics_telemetry_hook.py -q`._
  _2026-03-15 (Codex): Removed the transient `depth_tracking_capture_stage` and attached baseline depth capture directly to `depth_tracking_fullframe`, keeping the canonical branch `main_tee -> depth_tracking_queue -> depth_tracking_fullframe -> depth_tracking_fullframe_sink`. Baseline startup now also requires the native `noesis_depth_tracking_tensor_ext` bridge so DAv2 extraction never touches the unstable Service Maker Python tensor wrapper path. Follow-up hardening moved the branch to the raw tensor-meta pattern used by NVIDIA's tensor-meta samples (`network-type=100`, `output-tensor-meta=1`, `disable-output-host-copy=1`) so DeepStream no longer runs classifier postprocess on the DAv2 lane and the extractor reads `out_buf_ptrs_dev` only. Validation: `bash ./scripts/build_noesis_depth_tracking_tensor_ext.sh`, `python3 -m pytest tests/test_pipeline_build.py tests/test_depth_tracking_frame_processor.py tests/test_analytics_telemetry_hook.py -q`, `timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s` (stable to timeout, no segfault)._
  _2026-03-16 (Codex): Added a required read-only depth-registration artifact surface (`depth_registration.path` in `config/infer.yaml`, default `config/depth_registration.json`) and wired `ds8_runtime.py` to load/validate it before activation in baseline mode. The runtime now rejects missing or fingerprint-mismatched per-camera registrations instead of silently running raw DAv2 range against the room estimator. Validation: `python3 -m pytest tests/test_depth_registration.py tests/test_pipeline_build.py -q`._

## 5. Analytics Configuration & Exclusion ROIs

- [x] Use `analytics` config from `infer.yaml` and/or DS8 analytics YAML instead of DS7 INI where possible:
  - [x] Ensure stage and stream naming matches `noesis.server.analytics_api` expectations.
  - [x] Confirm `_extract_exclusion_polygons` logic matches the DS8 analytics config schema (stages → streams → roi_filtering → rois).
  _2025-11-29 (Codex): infer.yaml now points analytics/post/exclude to config/ INI files and loads DS8 analytics stages from config/nvdsanalytics.yaml; ds8_pipeline retains parsed stages for ROI/exclusion hooks to consume._
- [x] Ensure `attach_exclude_prune_hook` is wired to the `analytics` component in the DS8 pipeline and uses only documented DeepStream metadata APIs.
  _2025-11-29 (Codex): Exclusion hook attaches BatchMetadataOperator to `analytics`, pruning via nvds_remove_obj_meta_from_frame with YAML polygons._

## 6. Depth Valve / Gating Mechanics

- [x] Gate MapAnything SGIE compute using a GStreamer `valve` (DS7 parity) rather than relying on BufferOperator drops.
  - [x] Keep the gate upstream of SGIE (`main_tee → queue → valve → nvinfer`) so SGIE compute is physically avoided when depth is disabled.
  - [x] Prime the SGIE branch at startup (briefly open the valve, then close) to avoid preroll stalls when starting with the gate closed.
- [x] Update `DS8Pipeline.mark_depth_enabled` and `enable_depth` so they:
  - [x] Toggle `valve.drop` to open/close the SGIE branch.
  - [x] Continue updating `depth_frame_samples`, `depth_last_toggle`, and `depth_enabled` for stats and REST/WS reporting.
  _2025-12-14 (Codex): Added `mapanything_valve` gating + startup prime (log: `MapAnything gate primed; closed valve after …s`), configurable via `NOESIS_MAPANYTHING_GATE_PRIME_SECONDS`._
  _2025-12-15 (Codex): Validated GPU utilization drops to ~15–40% SM with depth disabled and spikes during `curl http://127.0.0.1:8080/api/v1/depth/refresh?seconds=10`, then returns to baseline; WebRTC stays connected (`NOESIS_MOSAIC_WEBRTC_ENABLED=1` + `python3 scripts/webrtc_gateway_smoke_test.py`)._
  _2025-12-16 (Codex): UI validation: oai2-fe depth drawer renders heatmap and all depth views during a depth-refresh window; after timer expiry the gate closes and GPU returns to baseline without affecting WebRTC mosaic playback._
  _2026-03-12 (Codex): Clarified the split depth model ownership: MapAnything valve gating remains on-demand RPC/depth drawer behavior only, while the new baseline DAv2 tracking lane is always-on in non-`v3dt` mode and is not controlled by `depth_enabled` / `enable_depth`._

## 7. Sinks & Flow Integration Points

- [x] Define explicit outputs in the DS8 graph:
  - [x] RTSP mosaic output (`rtsp_out`) for the WebRTC gateway.
  - [x] (If required) additional sinks for future Flow-based tensor/raw-frame retrieval (BEV/per-sensor exports).
  _2025-12-16 (Codex): Canonical mosaic output is RTSP→WebRTC; Flow retrieval remains deferred/optional for non-mosaic exports._

## 8. Error Handling & Diagnostics

- [x] Ensure `DS8Pipeline.errors` is populated with clear, actionable messages whenever a DS8 API call fails:
  - [x] Include context such as component name, property name, and exception details.
- [x] Ensure that `build_pipeline` throws on critical misconfigurations (missing YAML, invalid element names) so the runtime can fail fast.
  _2025-11-29 (Codex): prepare()/activate() now fail fast when pyservicemaker is unavailable or errors are present; enable_depth logs/writes gating limitations; ds8_runtime exits non-zero if preparation/activation fail._
  _2025-12-09 (Codex): `prepare()` now checks the integer return code from `pyservicemaker.Pipeline.prepare(on_message=...)` and records non-success values in `pipeline.errors`, preventing partially initialized graphs from activating silently._

## 9. Validation Steps for `ds8_pipeline`

- [x] With a candidate `infer.yaml`, run a DS8 test deployment and confirm:
  - [x] All components are created successfully (no missing plugins).
    _2025-11-29 (Opus): Validated: 19 components created (sources, streammux, preprocess, PGIE, tee, tracker, analytics_exclude, analytics, MapAnything SGIE, tiler, osd, sink_tee, sinks). All elements linked successfully._
  - [x] The graph topology matches `PIPELINE_GRAPH.md` when inspected via DS8 diagnostics.
    _2025-11-29 (Opus): Linking log confirms DS8 graph: sources→streammux→preprocess→PGIE→main_tee→(analytics_exclude→tracker→analytics→tiler→osd→sink_tee→sinks) + MapAnything branch (main_tee→mapanything_fullframe→fakesink)._
  - [x] PGIE, tracker, and analytics output object counts and classes match those from the DS7 pipeline on a test stream.
    _2025-12-13 (Codex): Tracking telemetry now includes StableIDManager outputs (bbox-only, GPU-first) with maintenance hooks and DS7 schema; counts/classes wiring mirrors DS7 and is parity-ready for Phase-7 runs. Smoke: `python3 scripts/reid_stable_id_smoke_test.py --no-spawn` asserts stable_id presence/persistence on tracking WS stream._
    _2025-12-14 (Codex): Re-ran `python3 scripts/reid_stable_id_smoke_test.py` (runtime-spawned, NOESIS_REID_TEST_MODE=1) and observed non-null stable_id persisting across frames without adding CPU appsink branches._
    _2025-12-20 (Codex): Inserted per-object ReID SGIE after tracker (OSNet embeddings) and plumbed tensor outputs into StableIDManager so StableID is appearance-based (cross-camera gallery/ghost matching). Re-exported OSNet-IBN MSMT17 to dynamic-batch=16 ONNX and built FP16 TensorRT engine at `models/engines/reid_osnet_ibn_msmt17_dyn_b16_fp16.engine`. Validated via `python3 -m compileall -q noesis reid scripts` and `trtexec ... --saveEngine=models/engines/reid_osnet_ibn_msmt17_dyn_b16_fp16.engine`._
    _2025-12-21 (Codex): Enabled the dedicated OSNet ReID SGIE (`models.reid.enable=true`, `gie_id=3`, batch=16) and consumed embeddings via Service Maker `tensor_items` in the telemetry hook using a manual DLPack→cudaMemcpy decode (avoid torch dlpack consumer) to eliminate ReID-related native crashes. Validated via `python3 scripts/reid_stable_id_smoke_test.py --pipeline-config config/infer_smoke_reid.yaml`, `timeout 70s python3 noesis/ds8_runtime.py --pipeline-config config/infer_reid_rtsp_min.yaml --disable-rest`, and a MapAnything depth burst (`--depth-enable-seconds 20` on `config/infer.yaml`) without regressions._
  - [x] Baseline runtime topology exposes deterministic world-observation and tracking-telemetry stages, plus the always-on DAv2 depth-tracking lane required by the fused world estimator.
    _2026-03-12 (Codex): Updated the graph to `... -> yolo26_pose -> world_observation_stage -> tracking_telemetry_stage -> tiler` with sibling tee branches for `depth_tracking_fullframe` and gated `mapanything_fullframe`. Regression coverage in `tests/test_pipeline_build.py` now asserts the new stage order and required depth-tracking components._
    _2026-03-16 (Codex): Revalidated the baseline topology against the real RTSP-built depth-registration artifact; `timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest` loaded `mapanything_fullframe`, `depth_tracking_fullframe`, and entered the main loop with all three live RTSP sources._
  - [x] Depth tensors from MapAnything SGIE are correctly picked up by `MapAnythingProcessor` and stored/published.
    _2025-12-01 (Codex): Rebuilt MapAnything engine from images-only ONNX (images→depth) with TensorRT 10.13 (FP16, batch 1–3), updated SGIE config to use image input (`input-tensor-from-meta=0`, `infer-dims=3;518;518`), and wired the DS8 mapanything branch to `/home/mayor/Noesis_Devel/models/mapanything_depth/1/model.plan`. MapAnythingProcessor now receives tensor meta (layer `depth`) and records depth FPS > 0._
    _2025-12-01 (Codex): DS8 tensor output conversion now uses torch DLPack fallback + finite masking and wall-clock PTS fallback; depth snapshots written under `data/depth/<camera_id>/...` during enabled window._
    _2025-12-15 (Codex): Verified updated MapAnything outputs are non-zero/dense at 294×518 and survive end-to-end storage/WS RPC; validated via `python3 scripts/ma_depth_rpc_smoke_test.py`._
  - [x] Exclusion polygons from analytics config produce the same pruning behavior as DS7.
    _2025-12-13 (Codex): Analytics API hot-reloads now regenerate the nvdsroiexclude INI and node.set the analytics_exclude element; hooks re-read analytics YAML so exclusion polygons applied match DS7 config without restart._
  - [x] Mosaic output is present as an RTSP stream for the WebRTC gateway.
    _2025-12-16 (Codex): Pipeline builds `rtsp_out` when WebRTC is enabled and the gateway consumes `rtsp://127.0.0.1:8554/<rtsp_path>`; validated via `scripts/webrtc_gateway_smoke_test.py`._

## 10. DS8 Utility Prototypes

- [x] Add a DS8 room-layout segmentation utility pipeline under `testpipelines/` that materializes its own semantic-seg model assets, uses `pyservicemaker` Pipeline API + `nvinfer`/`nvsegvisual`, stays GPU-first until mask emission, and writes a reusable per-source mask bundle for downstream Noesis consumers.
  _2026-03-11 (Codex): Added `testpipelines/room_layout_seg/` around a grouped `SegFormer-B5` ADE20K export (`models/room_layout_segformer_b5_ade20k/` ONNX + TRT engine + labels metadata), plus a DS8 probe that writes `layout_manifest.json`, dense class map, preview, FP16 probability bundle, and binary masks. Validated with `python3 -m pytest -q testpipelines/room_layout_seg/tests`, `python3 testpipelines/room_layout_seg/model_setup.py`, `python3 testpipelines/room_layout_seg/main.py --camera livingroomclip --headless --duration 4 --frames-per-package 8 --emit-every-frames 8`, and a short EGL run via `python3 testpipelines/room_layout_seg/main.py --camera livingroomclip --duration 4 --frames-per-package 8 --emit-every-frames 8`._
  _2026-03-24 (Codex): Extended the DS8 `yolo26-seg-depth-3d` utility prototype with prototype-only source-pose compensation metadata (`roll z = 180`) for the living-room stream and applied it in the standalone Three.js viewer to keep the dense RGBD shell upright without touching Menon's runtime logic. Validated with `python3 -m pytest tests/test_yolo26_seg_depth_3d_scene_stream.py tests/test_yolo26_seg_depth_3d_prototype_calibration.py -q`, `python3 -m compileall -q testpipelines/yolo26-seg-depth-3d`, `npm run build`, live backend restart on `127.0.0.1:8773`, and Playwright screenshot/row-trend checks._

- [x] Add a three-room DS8 comparison utility for the official YOLO26 ADE20K semantic Nano, Small, Medium, and Large checkpoints without routing semantic output through the detector PGIE contract.
  _2026-08-07 (Codex): Added `testpipelines/yolo26-sem-ade20k/`, exported exact static-batch-3 ONNX graphs, built RTX 3060/DS8 FP16 TensorRT engines, and added a verified custom parser for the exported `UINT8 [3,640,640]` class map. All four engines loaded through `nvinfer`, processed the canonical Living Room, Kitchen, and Family Room streams, exited with code 0, and wrote per-room class maps/masked frames plus three-room mosaics and JSON summaries. The focused contract suite passed (3 tests), all engine bindings were deserialized and checked, and short `trtexec` runs passed for every size._
  _2026-08-09 (Codex): Extended the snapshot contract to persist the exact aligned RGB frame for every room and model, refreshed Nano/Small/Medium/Large captures across the three canonical cameras, and wired those paired RGB/class-map artifacts into the OAI2 Sem-seg camera selector. Validated with the focused contract suite, four live batch-3 capture runs, the frontend semantic tests/build, and browser checks against the staged and active UI._
  _2026-08-09 follow-up (Codex): The live refresh sampled Living Room at 02:38 in near-total darkness, causing every flavor to hallucinate large outdoor regions and Large to collapse to only `wall` and `sky`. Restored the preserved well-lit Living Room RGB/class-map evidence for all four flavors, retained the fresh Kitchen and Family Room captures, and labeled the camera-specific provenance in the viewer._
  _2026-08-09 manual-capture follow-up (Codex): Added an owner-coordinated Sem-seg refresh action. A button press runs the selected fixed batch-3 engine over all three canonical cameras, publishes only a complete validated capture set, and swaps the viewer to those returned RGB/class-map URLs. Camera/model selection and tab load remain read-only. Focused manager, REST, gateway-policy, frontend-invariant, and production-build checks pass; live runtime capture is the promotion smoke._
  _2026-08-08 (Codex): Added an exact-engine still-image runner for saved evidence frames. Verified the checksummed raw Living Room fixed-camera anchor associated with the 48-frame MapAnything phone scan `20260802-162254-8bcc7dd7`, repeated only that frame across the static batch-3 input, and required all three output maps to match. Nano, Small, Medium, and Large completed successfully and wrote full-resolution masked images, raw class maps, and JSON summaries; Medium/Large pixel agreement rose from 74.8% in the prior live captures to 86.1% on the shared well-lit anchor. Added a no-build browser inspector that switches among all four maps, adjusts overlay opacity, and resolves mouse position back to the exact source pixel, ADE20K class ID, and label; a headless Chrome interaction smoke verified `wall` class 0 and `shelf` class 24 hover results._
  _2026-08-08 (Codex): Replaced the unused `oai2-fe` depth-drawer Stats tab with a display-only **Sem-seg** view backed by the exact saved well-lit Living Room source and Nano/Small/Medium/Large class maps. Added a 150-color stable ADE20K palette, a coverage-sorted legend of classes present, click-again and explicit-clear filter paths, and pixel hover inspection. The display filter dims non-selected pixels only; it launches no inference or depth request. Validated all 49 focused frontend tests, the production Vite build, asset hashes/copy-through, and live Chromium interactions for tab activation, 1920x1080 rendering, 22-class Large legend, wall isolation, and both clear paths with no browser exceptions._
  _2026-08-09 retention follow-up (Codex): Removed Nano and Medium from the model inventory, runtime/API choices, static viewer evidence, and writable capture state. Small and Large remain as the only supported semantic flavors; focused contracts, the frontend build, exact engine deserialization, and a live Small refresh validate the retained lane._
