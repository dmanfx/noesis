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
  _2026-07-10 (Codex): Replaced active DS8/V3DT/DS9 inline RTSP locators with `uri_secret` references backed by strict owner-only state. Runtime materializes values only in memory, strips reference keys before Service Maker properties, and keeps generated config, provenance, and fingerprints public. Focused runtime-secret, pipeline, depth-registration, and DS9 parity tests validate fail-closed behavior._
- [x] Provide a dedicated SV3DT config entrypoint (`config/infer_v3dt.yaml`) that points `tracker.config-file` at the SV3DT low-level tracker config.
  _2025-12-27 (Codex): Added `config/infer_v3dt.yaml` with `tracker.config-file: config/v3dt/nvtracker_sv3dt.yml` and V3DT metadata settings for camera-local world coords (validation pending)._
  _2026-01-23 (Codex): Renamed SV3DT baseline artifacts to `config/infer_v3dt_baseline.yaml`, `config/v3dt/nvtracker_v3dt_baseline.yml`, `config/v3dt/caminfo_baseline/`, `config/cameras_v3dt_baseline.yaml`, `config/archive/calibration_v3dt_baseline.json`, `config/dewarper_v3dt_baseline.txt`, and `config/analytics_exclude_baseline.ini`; updated all DS8 configs, scripts, and docs to match._
  _2026-07-10 (Codex): Ported the locked source/config semantics to a DS9-owned successor profile with unpadded 1920x1080 projection, external artifact-root ownership, strict preflight/runtime materialization, ordered BodyPose3DNet/NvMOT-ReID guarded build plans, and focused CPU tests. No DS8 engine/native binary was reused and no GPU build or live runtime claim was made; DS9 V3DT remains gated on its two engines, runtime-quality evidence, and shared-world calibration._
  _2026-07-11 (Codex): Superseded the camera-local/shared-calibration blocker. The active DS8, protected-reimplementation, and DS9 profiles bind the same separated-camera calibration and byte-identical locked camInfo set, explicitly publish `backend_world_m` with `caminfo_world_axes=xzy`, and share a fail-closed tracker-to-world conversion. The privacy-safe DS9 global-world v2 gate binds exact same-session config/runtime evidence and rejects v1/raw-tuple/camera-local promotion. Focused DS9 V3DT/ownership tests passed (153), both calibration profiles and Menon pose geometry passed, and the rebuilt DS9 bridge/profile provenance validates against ten-engine realization `da833111...a774`. All selected V3DT engines are realized; fresh occupied v2 behavior/resource/shutdown evidence and separate MV3DT overlap/time-sync/fusion proof remain pending._
  _2026-07-12 (Codex): Superseded the unpromoted V3DT semantic-v2 draft with
  lane-neutral `semantic_gate_v3`. The exact-replay report/source/snapshot bundle now
  requires canonical runtime-world roles, registered depth, one-to-one public
  and canonical observations, dual capture/observation bounds, contiguous
  publication sequence, per-frame tracker lifecycle generations and exact
  tombstones, plus strict pre-seal biometric leak detection/projection. Focused
  semantic, runtime-world, live-runner, ownership, tamper, replay, endpoint,
  lifecycle, and privacy tests passed; fresh occupied promotion evidence
  remains pending._
  _2026-07-12 (Codex): Tightened the unpromoted v3 proof with a sealed
  ownership-bound acquisition interval, fixed two-second pre-window capture
  allowance, explicitly unanchored live origins with window-contiguous
  publication/lifecycle values, exact last-presence tombstones, and marker-only
  fail-path privacy evidence._
  _2026-07-12 (Codex): Split processed lifecycle state from successfully
  published presence across DS8, protected V3DT, and DS9. Publication sequence,
  gate timestamps/counts, and tombstone retirement now commit only after the
  tracking receipt; failures retry with the current disappearance frame/time
  and fixed last-published presence. Semantic acquisition covers all tracking
  frames and the canonical publication clock, accepts only first-frame
  unanchored-origin tombstones, and uses shared strict unique-key JSON. Baseline
  ReID now requires the same neutral semantic-v3 proof alongside identity
  evidence; focused lifecycle, semantic, ownership, runner, and parity suites
  pass._
  _2026-07-12 (Codex): Closed the remaining behavior-report trust gap. Identity,
  V3DT world, Wholebody occupied, and Wholebody media now export strict
  owner-private exact-replay validators, and the live runner dispatches all six
  promotion behavior branches only after a successful producer process. The
  identity source contract is v2 so policy and before/after health evidence can
  recompute its claims; v1 is deliberately non-promotable. V3DT rebuilds the
  session config/launcher binding; all inputs reject duplicate/non-finite JSON.
  Focused producer, runner, ownership, registry, and promotion suites pass._
- [x] Keep the path normalization logic but ensure it follows DS8 recommendations:
  - [x] Resolve relative engine/config paths relative to the YAML file and/or repo root.
  - [x] Avoid guessing about path prefixes; use only documented top-level config locations.
  _2025-11-29 (Codex): Path resolution now normalizes engines/configs for nvinfer/preprocess/analytics to infer.yaml or repo root._
  _2026-07-12 (Codex): Removed active checkout identity from DS8 model,
  label, parser, and commented media paths. Canonical baseline and supported
  reimplementation YAMLs now select a tracked reviewed DAv2 source INI rather
  than an ignored generated build file; production still derives its
  engine-only copy under `NOESIS_BUILD_DIR`. The dev console routes launch,
  activity, and saved-profile state through that same build authority and
  derives detect/mask OSD behavior from the reviewed profile contract even
  when no generated profile INI exists. The equivalent active DS9 depth source
  was mirrored. Focused pipeline/dev-console/inference tests passed (153), and
  DS9 static preparation passed without a generated build-tree prerequisite._
- [x] Provide DS8-ready PGIE assets for YOLO26 segmentation (INI template + custom parser lib) to support profile overlays.
  _2026-01-28 (Codex): Added `pipelines/config_infer_primary_yolo26_seg.template.ini` and `pipelines/nvdsinfer_yolo26_seg/` parser library so DS8 can materialize size-specific YOLO26 INIs at runtime._
  _2026-01-29 (Codex): Fused YOLO26-seg ONNX/engines to output `output0` only (mask compose on-GPU) and enabled `disable-output-host-copy=1`; updated parser to handle device pointers and consume fused output0; built b3 FP16 engines for `yolo26{n,s,m}` capped to 30 detections (match YOLO11) and validated DS8 runtime loads each size without output1/proto host copies. This cap drastically reduces ROIAlign cost and dropped YOLO26n GPU usage to ~10% in DS8 runs (from ~80–90% when composing 300 masks)._
  _2026-07-10 (Codex): Added a repo-owned guarded DS8 YOLO26-seg engine maintenance command for `n/s/m`. CPU-only planning now pins and hashes the exact TRT 10.13/CUDA 13.0/GPU, ONNX, parser, and template contract; real mode uses a nonblocking process lock, preserves prior bytes, enforces disk/time/log/GPU-memory bounds, builds to a sibling temporary, requires separate positive candidate deserialization before atomic install, and re-deserializes the final path. Focused tests reject the observed exit-zero/`PASSED` transcript containing Error[6]/Error[4], the deceptive `trtexec --version` banner followed by Model-missing failure, final-path failure, and lock contention. The plan and ten CPU tests passed. Guarded `s` run `20260710T222107305588Z` preserved prior SHA-256 `c65a86ec...ca08`, installed independently verified SHA-256 `2edd2864...b670`, passed candidate and final-path deserialization in 1.076s and 1.075s, stayed within 658 MiB peak GPU memory, and left no compute owner. Runtime-quality validation remains pending._
  _2026-07-10 (Codex): Guarded `n` run `20260710T222623433238Z` preserved prior SHA-256 `39f341a1...18e67`, installed independently verified SHA-256 `ffc2666a...bcb0`, passed candidate and final-path deserialization in 1.075s and 1.078s, stayed within 622 MiB peak GPU memory, and left no compute owner._
  _2026-07-10 (Codex): Guarded `m` run `20260710T223108901671Z` preserved prior SHA-256 `5b7a1869...d11a2`, installed independently verified SHA-256 `2ce7cb72...7ac8`, passed candidate and final-path deserialization in 1.075s and 1.074s, stayed within 904 MiB peak GPU memory, and left no compute owner. All `n/s/m` engines now match the canonical DS8 TensorRT 10.13.3.9 host; runtime-quality validation remains pending._
  _2026-06-21 (Codex): Promoted the DEIMv2 Wholebody49 prototype into an optional DS8 PGIE profile (`--pgie-profile wholebody49 --size s|x`) with canonical copied ONNX/engine assets, production parser library, size-specific PGIE/preprocess materialization, and fail-fast preflight. Validation: `make -C pipelines/nvdsinfer_deimv2_wholebody49`, `python3 -m pytest tests/test_deimv2_wholebody49_assets.py -q`, `python3 -m compileall -q noesis tests/test_deimv2_wholebody49_assets.py`, and direct materialization/preflight for sizes `s` and `x`._
  _2026-07-10 (Codex): Extended the same profile semantics to the DS9 successor without reusing DS8 binaries. The shared materializer now accepts runtime-owned model/ONNX/engine/pipeline/build roots; DS9 owns matching templates, labels, a DS9-header-built dual-mode parser, source staging, CLI/preflight wiring, OSD derivation, manifest records, and guarded TensorRT 10.14 specs. CPU/ONNX/parser/plan gates pass. Runtime-quality parity remains blocked until an exclusive-GPU window builds both DS9 engines and an occupied-scene mask/box quality gate passes._
  _2026-07-11 (Codex): Hardened the active DS8 and DS9 Wholebody49 parser sources as byte-identical fail-closed contracts: exact unique output names/sets, batch-stripped `[1240,6]` and `[1240,80,80]` shapes, normalized-xyxy coordinates, finite tensors, body-class-only output, and leak-free mask rollback. Installed DS8/DS9 `nvdsinfer` `SplitFullDims` sources justify rejecting batch-bearing callback dimensions. A compiled fake-NvDsInfer adversarial harness passes for both copies, including ASan/UBSan; parser binaries remain intentionally unrebuilt until the guarded DS8/DS9 artifact rebuild and manifest rebase._
  _2026-07-11 (Codex): Replaced the DS9 Wholebody workspace-only `trtexec` build with a DS9-owned TensorRT 10.14 C++ builder after two S-mask attempts safely crossed the unchanged 9,000 MiB outer guard. A third guarded attempt proved that 4 GiB TACTIC_DRAM still reached 9,441 MiB across 78 samples before candidate publication and rolled back cleanly. Transaction `20260711T080353252694Z` then rejected the 3 GiB trial before allocation or candidate creation with TensorRT Error Code 3 because `hasSingleBit(poolSize)` was false; it peaked at 51 MiB across two samples and retained the prior realization. The next reviewed contract changes only TACTIC_DRAM to the legal 2 GiB power-of-two value, retains 4 GiB WORKSPACE, exact FP16 batch-three shapes, default optimization/tactic-source/aux-stream policy, private source/ONNX snapshots, exclusive candidate creation, and independent candidate/final `trtexec` loads. Compile-time, Python, independent-validator, and adversarial source-contract checks enforce the power-of-two rule; actual engines and occupied-scene quality remain gated._
  _2026-07-11 (Codex): The legal 2 GiB TACTIC_DRAM S trial `20260711T081443440868Z` still reached 9,441 MiB across 78 samples; the host finalizer proved no candidate, clean rollback, and preserved realization `c183f91a...16dd`. The reviewed successor keeps TACTIC_DRAM at 2 GiB, selects WORKSPACE by the already validated variant (S 2 GiB, X 4 GiB), and leaves optimization, tactic-source, auxiliary-stream, FP16, profile, and output contracts unchanged. Producer/independent proof tests reject swapped or duplicate workspace markers; 134 focused build/rollback/provenance tests, strict pinned-image compilation, and both isolated no-GPU plans pass without authority mutation. Engines and occupied-scene evidence remain gated._
  _2026-07-11 (Codex): Transaction `20260711T083837156365Z` falsified the tighter workspace as a sufficient build-peak control: S still reached 9,436 MiB across 78 samples, with no candidate, an already-absent engine, clean rollback, and realization `cda791ec...dc99` preserved. The next reviewed single-variable search policy retains S/X WORKSPACE 2/4 GiB and shared TACTIC_DRAM 2 GiB, getter-pins builder optimization levels S=0 and X=3, and leaves tactic sources, auxiliary streams, guards, FP16, profile, and outputs unchanged. The 140-test focused build/rollback/provenance set, strict pinned-image compile, S/X no-GPU plans, lint, syntax, docs, and diff checks pass without GPU or authority mutation._
  _2026-07-11 (Codex): The optimization-level-0 S trial `20260711T130350917674Z` reduced peak GPU memory to 506 MiB across 19 samples and exposed an exact final-node tactic-admission failure instead of host pressure. Half-format tactics requested 4,571,136,000 bytes against roughly 1.95 GB budgets; the float tactic requested 9,142,272,000 bytes. TensorRT failed closed with Error Code 10, no candidate, clean rollback, and realization `3a896c8d...fa6918a` preserved. The reviewed successor changes only S WORKSPACE to 8 GiB so the half tactic fits while the float tactic remains excluded; S opt0, X 4 GiB/opt3, shared 2 GiB TACTIC_DRAM, FP16, profiles, outputs, tactic sources, auxiliary streams, and 9,000/11,000 MiB guards are unchanged. Strict pinned compilation, 141 focused producer/validator/adversarial tests, both no-GPU plans, and deterministic source-rebase dry-run pass without authority mutation._
  _2026-07-11 (Codex): Transaction `20260711T131135058479Z` proved the 8 GiB S admission policy generated an engine in 55.7403 seconds. Its 4,880 MiB value came from only 19 coarse host samples and is not an authoritative peak; TensorRT's 4,603 MiB result covers only its allocator. Candidate publication was correctly withheld because an overlong message among 55,191 VERBOSE diagnostics set the bounded logger's sticky truncation flag; the log otherwise had 25 INFO, three WARNING, and zero ERROR/INTERNAL_ERROR messages. No candidate was written, rollback found the engine already absent, and realization `493d1cdb...a95bb0` remained prior. The reviewed successor ignores only VERBOSE before copy/truncation bookkeeping while keeping INFO-and-higher emission, captured truncation fatal, and errors sticky fatal. Both S/X source contracts and exact transcript proof pin that logger policy. Strict pinned compilation, 148 focused tests including an extracted-C++ logger harness, both no-GPU plans, and deterministic rebase dry-run pass without GPU or authority mutation._
  _2026-07-11 (Codex): Transaction `20260711T132034937854Z` then generated a 25,348,956-byte S engine and passed both candidate and installed-path deserialization. Host finalization rejected the run because Wholebody proof lacked its caller-known maintenance-manifest path, removed the no-prior engine, and preserved realization `e87a620d...a501b`. Its 4,846 MiB coarse result conflicts with a roughly 9,472 MiB operator observation, so neither that value nor the earlier 4,880 MiB value clears the unchanged 9,000 MiB guard._
  _2026-07-11 (Codex): Closed both proof gaps without another engine build. Engine maintenance now creates before start and establishes a 25 ms NVML-v2 raw-byte sampler with a 250 ms maximum gap, exact GPU/container/transaction/root/wrapper identity, prompt pidfd notification, and durable JSONL inside the retained engine-finalize cohort. Finalization, reconciliation, and authoritative realization validation independently reconstruct and bind the same evidence; direct reconciliation cannot omit it, Wholebody has no legacy exemption, and only eight frozen exact pre-guard artifact tuples remain compatible. Caller-known maintenance path/digest is likewise mandatory. The 94-test sampler/Wholebody/reconcile suite and 33-test wrapper suite pass; Ruff, compilation, shell syntax, plans, and live read-only compatibility validation are separate recorded gates. A live sampler smoke and no-GPU S maintenance plan also pass without changing realization `e87a620d...a501b`; a fresh exclusive-GPU engine run remains required for authoritative maximum-observed and runtime-quality evidence._
  _2026-07-11 (Codex): Sealed 25 ms evidence then resolved the S build envelope. Transaction `20260711T150649395390Z` reached 9,481 MiB with 8 GiB WORKSPACE and rolled back at the 9,000 MiB guard; after an append-only source rebase, `20260711T151452845569Z` reached 9,476 MiB with 6 GiB WORKSPACE and rolled back at the same guard. Installed TensorRT 10.14 headers and the successful transcript confirm WORKSPACE is tactic-admission memory, not a full-device peak bound. The reviewed contract keeps S optimization level 0, 6 GiB WORKSPACE, and 2 GiB TACTIC_DRAM while setting the outer guard to 10,000 MiB, preserving more than 2 GiB free on the 12 GiB target. X remains level 3, 4 GiB WORKSPACE, 2 GiB TACTIC_DRAM, and 11,000 MiB._
  _2026-07-11 (Codex): Committed transaction `20260711T152338477896Z` realized the mode-0600 S engine at 25,327,772 bytes and SHA-256 `1fb95225...dd6bdf`; its guard evidence contains 2,738 samples, a 27.742316 ms maximum gap, and 9,481 MiB maximum observed. Committed transaction `20260711T152552934749Z` realized the mode-0600 X engine at 108,579,580 bytes and SHA-256 `a5f4322d...138c3`; its evidence contains 12,750 samples, a 30.858598 ms maximum gap, and 1,017 MiB maximum observed. Candidate and installed-path loads passed for both. Canonical, V3DT, and Wholebody49 file/provenance validation pass against final ten-engine realization `99dc3aba...614b`. Runtime mask/box quality and occupied-scene evidence remain pending._
  _2026-07-11 (Codex): Implemented the fail-closed DS9 Wholebody runtime-quality admission contract without launching live work. Each S/X promotion now requires one sealed 300-second session containing independently replayed occupied v2, direct-RTSP/WebRTC decoded-media, and generic resource-soak v2 evidence, with exact container/checkout/realization/primary-engine/config binding, all-source 30-second frame advancement, provisional 2 fps / 2.5-second liveness floors, zero core CPU-copy violations, OOM/leak/PID bounds, and provisional S 10,000 MiB / X 11,000 MiB GPU ceilings. Empty-house evidence fails honestly; the old V3DT-only soak contract is rejected. Focused producer, runner, supervisor, ownership, replay, and adversarial tests passed; fresh live S/X sessions remain pending._
  _2026-06-30 (Codex): Added size-aware YOLO11 detect and YOLO11-seg PGIE support for `s|m|l`. Runtime now defaults to YOLO11 detect medium, materializes deterministic size-specific PGIE/preprocess configs, and preflights YOLO11-seg TensorRT plugin loading only when selected ONNX assets contain custom `EfficientNMSX_TRT` / `ROIAlignX_TRT` ops. Added `scripts/download_export_build_yolo11_profiles.py` to download/export/build the canonical YOLO11 detect/seg assets, and rebuilt `pipelines/nvdsinfer_yolo11_seg/libnvdsinfer_yolo11_seg.so` with a bounded confidence-sorted parser cap (`NOESIS_YOLO11_SEG_PARSER_TOPK`, default 30). Validation: `python3 -m compileall -q noesis/ds8_runtime.py scripts/download_export_build_yolo11_profiles.py`, `git diff --check`, `make -C pipelines/nvdsinfer_yolo11_seg`, `python3 -m pytest tests/test_ds9_parity_hardening.py::test_yolo11_seg_has_bounded_mask_topk -q`, TensorRT load/inference checks for all six engines, and direct DS8 materialization/preflight for `yolo11` and `yolo11_seg` sizes `s`, `m`, and `l`._
  _2026-07-10 (Codex): Superseded runtime ONNX inspection/build fallback with one fail-closed DS8/DS9 inference boundary. Every active PGIE/SGIE and NvMOT config is atomically derived under the runtime build root with the selected nonempty engine and no ONNX/ETLT/UFF/calibration/custom-builder inputs; parser libraries exporting TensorRT builder symbols are rejected. Runtime profile/depth/native-extension materializers never export, compile, invoke `trtexec`, or delete/rebuild an engine. Source-rich configs remain immutable inputs to explicit offline maintenance. YOLO11-seg now loads its required TensorRT plugin from the selected profile contract rather than opening ONNX bytes. Validation: `pytest -q tests/test_inference_runtime_contract.py` (44 passed), canonical root pytest (862 passed, 15 skipped), explicit DS9 pytest (126 passed), DS9 static prep, schema drift, docs consistency, compilation, and whitespace checks._
  _2026-07-12 (Codex): The canonical CPU gate exposed that DS9 static prep inherited its Python import path from the caller after ownership validation began using shared `noesis_core` strict JSON. The launcher now owns an ordered DS9-plus-repository `PYTHONPATH`, and a subprocess regression runs it with no inherited `PYTHONPATH`; direct static prep and the focused regression pass._
  _2026-07-10 (Codex): Accepted production evidence after the engine-only cutover is stored under appliance checkpoint `20260710T203037Z-driver595-postboot/`. V3DT log `ds8-runtime-v3dt-30s-engine-only-orderly-eos.log` passed with startup `8.104s`, 30.0 active seconds, tracking sequence `42→1590`, orderly shutdown `1.467s`, exit `0`, and no forced kill/signature/missing marker. Baseline rerun `ds8-runtime-baseline-30s-engine-only-orderly-eos-rerun.log` passed with startup `10.569s`, sequence `36→895`, orderly shutdown `1.567s`, exit `0`, and the same clean checks. Both logs show engine deserialization/`Load new model` from `build/runtime_inference/...`, empty generated-config source-key scans, the exact EOS/callback/wait/shutdown marker order, and no residual process, listener, or GPU owner._

## 3. Graph Structure Parity with DS7

- [x] Ensure the DS8 graph reproduces the DS7 structure from `PIPELINE_GRAPH.md`:
  - [x] Sources via DS8 `nvmultiurisrcbin` (combined ingest + mux / `nvstreammux` equivalent).
  - [x] `nvdspreprocess` (if retained in DS8 graph) with its INI config.
  - [x] PGIE `nvinfer` (YOLO) with DS8-ready properties.
  - [x] Tracker (`nvtracker`) with `ll-lib-file` and `ll-config-file` from the DS7 configs.
  - [x] ROI exclusion through the repo-owned pre-tracker
    `nvdsroiexclude` element, with the analytics YAML rendered into its exact
    native config and no Python pruning path.
  - [x] `nvdsanalytics` stage(s) configured to mirror `config_nvdsanalytics_post.ini` / DS8 analytics YAML.
  - [x] Tiled mosaic via `nvmultistreamtiler` and annotated display via `nvdsosd` with DS7 default/INI settings.
  - [x] Mosaic output via one NVENC encode and AU-aligned SHM publish for the WebRTC gateways; `nvrtspoutsinkbin` is optional tooling only.
  - [x] Remove deprecated mosaic JPEG-over-WebSocket branch (`nvjpegenc` + `appsink`); mosaic video is WebRTC-only.
  _2026-01-10 (Codex): Removed the `nvjpegenc/appsink` mosaic branch from `noesis/pipelines/ds8_pipeline.py` and removed `output`/`mosaic_output.jpeg_enabled` from `config/infer*.yaml`._
  - [x] Treat `sinks.mosaic_sink` / `sinks.bev_sink` as placeholders (do not build fakesink branches) to avoid interfering with mosaic delivery.
  _2026-01-11 (Codex): `noesis/pipelines/ds8_pipeline.py` now skips building/linking these placeholder sinks; RTSP server responds correctly and `scripts/webrtc_gateway_smoke_test.py` receives `webrtc_answer` + decoded frames again._
  _2026-07-11 (Codex): Brought the protected V3DT runtime onto the same media-readiness and capacity contract as baseline DS8 and DS9: the exact RTSP mount must answer a successful DESCRIBE, one bounded warm gateway starts by default, and remaining slots are created on demand. The production canary now forces and requires exactly one warm slot so a zero-warm inherited environment cannot pass on a generic capacity prefix. Focused V3DT/DS9 RTSP, WebRTC, canary, and lifecycle tests passed._
  _2026-08-08 (Codex): Superseded RTSP readiness with SHM feeder readiness plus a decoded WebRTC media probe. The canonical graph keeps only the raw pre-encode queue leaky, uses bounded non-leaky compressed queues, encodes at 12 Mbps CBR, and emits an IDR every 10 frames (`h264_iframeinterval=10`, `h264_idrinterval=10`)._
  _2026-08-08 (Codex): Validated the new transport end to end with the real signaling server: one client decoded 158 frames and two simultaneous clients each decoded 159 frames over five-second synthetic H.264 SHM runs. The installed NVIDIA encoder independently proved IDRs at input frames 0/10/20/30/40 for the final `10/10` settings; with a natural `100/100` GOP, a repo-owned force-IDR request after frame 14 was acknowledged and made frame 15 the next IDR without any RTSP sink._
  _2026-08-09 (Codex): Promoted `deploy-20260809-mosaic-shm-v2-ds9` after rebasing on the active semseg Small/Large v12 release and carrying its live state forward with online SQLite backups. The running SHM output measured 240 AUs in 9.822 seconds (24.43 fps, 9.82 Mbit/s, 24 keyframes); one live WebRTC peer decoded 307 frames, and two simultaneous peers decoded 221 and 220 frames over 10 seconds. The six-unit appliance cohort stayed active, UDP 5400 and RTSP 8554 were absent, and peer disconnect/rebuild logged no severe media failure. Occupied-person visual inspection remains pending._
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
  _2026-07-10 (Codex): Added the repo-owned zero-copy `noesiseos` control immediately after `streammux` in DS8, V3DT, and DS9. Its property setter enters terminal drop state and schedules downstream EOS asynchronously after Service Maker releases its setter locks; shutdown requires an exact monotonic acknowledgement, the pipeline EOS callback, and `Pipeline.wait()` return before callback-owned state is released. All potentially closed downstream valves use EOS-preserving drop modes. The canonical live YOLO26m gate processed 30 seconds of advancing authenticated world state and exited `0` in 1.567 seconds after SIGTERM, with no forced kill, reconnect loss, native fault, or GStreamer error._
  _2026-07-10 (Codex): Added a fail-closed DS9 successor canary supervisor on the isolated secondary Docker daemon. Canonical readiness is fixed to baseline YOLO26 detect `m`; V3DT is separately fixed to YOLO26-seg `s`; YOLO11-seg is alternate-only. Plan mode is write-free/no-launch and requires an owner-only external engine realization linked to exact tracked manifest/source-contract digests; the tracked manifest remains immutable. It also gates exact image/base/source/engine provenance, exclusive owners, ports, and the shared artifact transaction lock. Authorized run mode alone receives GPU/host network/IPC, with an exact 26 GiB memory ceiling, read-only checkout/artifacts, three individual secret mounts, four private writable roots, actual-inspect verification, lock ownership through confirmed container GPU handoff, ordered EOS/exit-zero cleanup, checkout immutability, and sealed evidence. CPU validation: 28 focused container/secret tests passed, Ruff/compile/bash/static-prep/docs consistency passed; no DS9 container or GPU runtime was launched._
  _2026-07-12 (Codex): Closed a release-audit discrepancy in engine-maintenance plan mode: the plan path now requires pre-existing caller-owned engine/evidence/log directories, exact mode-0700 evidence directories, and the exact mode-0600 transaction lock, opens that lock read-only, and refuses drift without `mkdir` or `chmod`. Recursive mode/size/mtime/ctime tests prove a successful plan changes no artifact metadata; missing-path and unsafe-mode tests prove it does not repair state. The full wrapper suite passed 35 tests, and a real MapAnything runc/no-GPU plan preserved the complete external artifact fingerprint._
  _2026-07-12 (Codex): Completed the canonical DS9 MapAnything FP32 maintenance/finalization transaction. Real inference, candidate/final deserialization, the 9,000 MiB NVML guard, exact source/fixture/build authority, and atomic realization reconciliation all passed; the new 3,883,865,652-byte engine is SHA-256 `eabc1169c7d725ed7cff171ed54c402c4282fdcf23b87a58f4e41a95fe23ecc8` and realization `6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4` validates for canonical, V3DT, and Wholebody49. This is artifact readiness only; live multi-camera MapAnything and floorplan behavior remains separately gated._
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

- [x] Use the durable DS8 analytics YAML as the exclusion source of truth and
  derive the native INI before graph construction:
  - [x] Ensure stage and stream naming matches `noesis.server.analytics_api` expectations.
  - [x] Require exact canonical source coverage and render
    `stages → exclude → streams → roi_filtering → rois` without duplicate or
    ambiguous keys.
  _2025-11-29 (Codex): Historical implementation pointed analytics/post/exclude to INI files, loaded DS8 analytics stages from `config/nvdsanalytics.yaml`, and retained parsed stages for ROI/exclusion hooks._
- [x] Keep the repo-owned `nvdsroiexclude` element immediately before the
  tracker as the sole canonical exclusion implementation.
  - [x] Reject enabled streams without a polygon; allow an empty ROI set only
    for an explicitly disabled stream.
  - [x] Require strict startup parsing, exact active-config SHA-256, monotonic
    reload request/accept/failure sequences, zero reload-error drift, and an
    object-removal counter.
  - [x] Build separate DS8 and DS9 binaries from byte-identical owned source,
    with each build rejecting the wrong DeepStream major.
  _2025-11-29 (Codex): Historical `attach_exclude_prune_hook` attached a Python BatchMetadataOperator after analytics._
  _2026-07-10 (Codex): Retired that Python hook and removed its duplicate
  polygon cache/processor/operator path. Baseline DS8, V3DT, and DS9 now share
  the same pre-tracker native contract, startup synchronization, receipt
  verification, and source-coverage failure behavior. Rebuilt with
  `bash gst-plugins/build_nvdsroiexclude.sh` for DS8 and
  `NOESIS_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 bash DS9/scripts/build_nvdsroiexclude_ds9.sh`
  inside the pinned DS9 environment. Validation: 68 focused
  API/graph/shutdown/gate tests, 12 native plugin/origin tests, and 50 DS9
  supervisor-boundary tests passed._

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
  - [x] AU-aligned H.264 SHM output (`mosaic_h264_shmsink`) for the single feeder and WebRTC gateways.
  - [x] Optional RTSP tooling output (`rtsp_out`) only when explicitly enabled.
  - [x] (If required) additional sinks for future Flow-based tensor/raw-frame retrieval (BEV/per-sensor exports).
  _2025-12-16 (Codex): Canonical mosaic output is RTSP→WebRTC; Flow retrieval remains deferred/optional for non-mosaic exports._
  _2026-08-08 (Codex): The transport-only encoded appsink/appsrc edge replaces RTSP re-consumption and does not expose raw frames or metadata outside the Service Maker graph._

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
    _2026-07-10 (Codex): Superseded DS9's OSNet profile with exact DS8 TAO Swin-Tiny parity. One shared contract now locks source SHA-256, RGB ImageNet preprocessing, direct `256x128` resize, dynamic batch 1..16, `fc_pred/256`, synchronous raw tensor metadata, and parser absence. DS9 owns matching source/config/build/manifest/preflight paths and fails closed until its TensorRT 10.14 engine exists. GPU-free ReID/profile/pipeline tests passed; no engine was built while DS8 owned the GPU._
    _2026-07-10 (Codex): Removed the remaining V3DT OSNet/512 identity split by moving `config/infer_v3dt_reimpl_fast1056_mp4.yaml` to the same Swin engine/config and `fc_pred/256` contract as baseline/DS9, with both runtime contracts pinned to exact Swin identity. Sequential live reports under `/mnt/noesis_storage/checkpoints/20260710T203037Z-driver595-postboot/` passed without gallery deletion, state reset, or wait: V3DT `ds8-runtime-v3dt-30s-swin-shared-final.json` (SHA-256 `2710f73c...d997`, startup `8.56s`, sequence `81→1789`, shutdown `1.617s`, exit `0`) followed immediately by baseline `ds8-runtime-baseline-after-v3dt-swin-final.json` (SHA-256 `58db8a33...0010`, startup `10.565s`, sequence `29→774`, shutdown `1.917s`, exit `0`). Both identity logs report model SHA-256 `7a15727d...f6830` and `fc_pred/256`._
  - [x] Baseline runtime topology exposes deterministic world-observation and tracking-telemetry stages, plus the always-on DAv2 depth-tracking lane required by the fused world estimator.
    _2026-03-12 (Codex): Updated the graph to `... -> yolo26_pose -> world_observation_stage -> tracking_telemetry_stage -> tiler` with sibling tee branches for `depth_tracking_fullframe` and gated `mapanything_fullframe`. Regression coverage in `tests/test_pipeline_build.py` now asserts the new stage order and required depth-tracking components._
    _2026-03-16 (Codex): Revalidated the baseline topology against the real RTSP-built depth-registration artifact; `timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest` loaded `mapanything_fullframe`, `depth_tracking_fullframe`, and entered the main loop with all three live RTSP sources._
  - [x] Depth tensors from MapAnything SGIE are correctly picked up by `MapAnythingProcessor` and stored/published.
    _2025-12-01 (Codex): Rebuilt MapAnything engine from images-only ONNX (images→depth) with TensorRT 10.13 (FP16, batch 1–3), updated SGIE config to use image input (`input-tensor-from-meta=0`, `infer-dims=3;518;518`), and wired the DS8 mapanything branch to `/home/mayor/Noesis_Devel/models/mapanything_depth/1/model.plan`. MapAnythingProcessor now receives tensor meta (layer `depth`) and records depth FPS > 0._
    _2025-12-01 (Codex): DS8 tensor output conversion now uses torch DLPack fallback + finite masking and wall-clock PTS fallback; depth snapshots written under `data/depth/<camera_id>/...` during enabled window._
    _2025-12-15 (Codex): Verified updated MapAnything outputs are non-zero/dense at 294×518 and survive end-to-end storage/WS RPC; validated via `python3 scripts/ma_depth_rpc_smoke_test.py`._
  - [x] Exclusion polygons from analytics config produce the same pruning behavior as DS7.
    _2025-12-13 (Codex): Analytics API hot-reloads now regenerate the nvdsroiexclude INI and node.set the analytics_exclude element; hooks re-read analytics YAML so exclusion polygons applied match DS7 config without restart._
    _2026-07-10 (Codex): Superseded config-file/node-set inference with an
    exact native transaction receipt. API and plugin tests prove the accepted
    hash/sequence, retained prior config after rejection, disabled-empty policy,
    fatal ambiguous commits, and DS8/DS9 plugin-origin parity._
  - [ ] Prove a real occupied stream is excluded and restored in the same
    authenticated runtime with advancing frames.
    _2026-07-10 (Codex): The strict gate is implemented at
    `scripts/roi_reload_smoke_test.py`; unit tests cover mandatory `finally`
    restore and blocked unoccupied scenes. Live occupied evidence remains
    pending and must not be inferred from the historical counter-only smoke._
  - [x] Mosaic output reaches the gateway as complete H.264 AUs over the configured SHM socket.
    _2025-12-16 (Codex): Historical RTSP ingress validation was performed with `scripts/webrtc_gateway_smoke_test.py`._
    _2026-08-08 (Codex): Current validation requires `MosaicH264ShmFeeder ready`, advancing gateway RTP, and at least one decoded frame; RTSP is forced off by the bounded canary._
  - [x] Canonical live shutdown proves native quiescence instead of relying on timeout, forced termination, interpreter bypass, or source-level EOS.
    _2026-07-10 (Codex): `python3 scripts/ds8_runtime_30s_gate.py --duration-s 30 --log-path <owner-state-log>` authenticated against REST and WebSocket health, required RTSP DESCRIBE plus advancing tracking/world sequences (38 to 1017), then required `Orderly pipeline EOS accepted`, expected EOS callback, Service Maker wait return, `Shutdown complete`, exact exit `0`, and no forced kill. Focused lifecycle, bridge, graph, and manifest tests plus the full root suite (`785 passed, 15 skipped`) passed._
    _2026-07-11 (Codex): Repaired defects exposed by the first final DS9
    baseline attempt without changing the shutdown checker: closed validation
    clients now leave the handler once and do not emit false `ERROR` lines;
    WebSocket shutdown must prove both listener and thread quiescence; aggregate
    world observation-window regression no longer poisons monotonic capability
    progress; and executor/publisher queue delay is a separate stage inside the
    unchanged assembled 3 ms serialization CPU budget. Focused cross-runtime tests passed (228); a
    fresh live DS9 pass remains pending._
    _2026-07-11 (Codex): Extended the same fail-closed single-render contract to
    every mounted product surface: 41 JSON routes are measured and the five
    pre-rendered/file routes are explicitly exempt. Static route inventory plus
    representative boundary and endpoint tests passed (27); an undeclared
    successful response now raises and increments a tagged boundary error._
    _2026-07-11 (Codex): Replaced the incomplete no-op test pipeline with an
    explicitly non-native synthetic lifecycle backend: readable properties,
    synthetic analytics receipts, monotonic typed EOS callback, and blocking
    wait now exercise orderly shutdown without claiming DeepStream execution.
    DS8/DS9 selectors are separate, and every stub result is stamped
    `backend=synthetic_stub`, `native_runtime=false`, `promotable=false`.
    Focused lifecycle/graph/boundary tests passed. Three-second DS8/DS9
    isolated-state smokes then passed with 4/4 stats samples,
    `0.724547`/`0.739423 ms` boundary
    p99, zero errors/violations, exact callback/wait/shutdown markers, exit `0`,
    and no forced kill. Native acceptance remains unchanged and still requires
    the real plugin, callback, wait, media, and GPU evidence._
    _2026-07-12 (Codex): Hardened the native DS8/V3DT canary so it can execute
    from an immutable checkpoint without writing into the checkout or protected
    operator state. A mandatory external mode-0700 marked cohort now owns HOME,
    XDG/TMP/CUDA/GStreamer caches, generated build configs, depth/floorplans,
    analytics, calibration/alignment, StableID/identity-v2, world, scene,
    virtual-twin, diagnostics, logs, and reports; runtime argv carries the exact
    external `--storage-base`. Only three pre-existing owner-private secret file
    paths survive a narrow environment boundary, and the bearer is load-only.
    Reuse preserves the sequential baseline/V3DT identity cohort while evidence
    names remain create-once. Focused gate coverage passed 27 tests; the broader
    state/secrets/calibration/pipeline/inference set passed 163, and the separate
    DS9 runtime-config/hygiene/secret set passed 7. Python compilation, Ruff,
    docs consistency, and scoped whitespace checks also passed; no runtime or
    GPU workload was launched. The source/runtime change requires a new sealed
    checkpoint and canonical-graph supersession before final live evidence._

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

## 11. DS9 Successor Ownership Promotion

- [x] Separate static DS8/DS9 contract truth from dynamic DS9 acceptance.
  _2026-07-11 (Codex): The tracked ownership YAML is now selector-free and uses
  status only for implementation/contract truth. Missing realized or live
  evidence remains a validator-computed blocker; dynamic evidence cannot
  override a real tracked `known_gap` or `blocked` status._
- [x] Add an owner-private replayable promotion registry and offline recorder.
  _2026-07-11 (Codex): Added fixed-path canonical hash-chained JSONL plus a
  persisted head, no-follow descriptor CAS, advisory locking, explicit
  supersession/revoke, fsync, current-matrix filtering, and an offline command
  that reconstructs and independently validates selectors before append. No
  runtime, Docker container, model build, or GPU work is started by recording._
- [x] Make occupied behavior evidence independently replayable.
  _2026-07-11 (Codex): Wholebody, floorplan, and V3DT world reports now bind
  minimal timestamped privacy-closed source transcripts; identity and semantic
  evidence retains its sealed transcript/snapshot model. Validation exactly
  recomputes reports, rejects root-session relabeling, and checks the inspected
  runtime window._
  _2026-07-11 (Codex): Upgraded the DS9 floorplan producer and ownership gate
  to v2: serialized exact capture receipts for every reviewed camera, immutable
  snapshot/floorplan digests, explicit depth-only RGB truth, active-floorplan
  and local-BEV N/N health, and a cache-only controller/registry zero-mutation
  proof. Focused producer, replay, runner, and ownership adversarial tests pass;
  fresh native DS9 evidence remains required._
  _2026-07-12 (Codex): Superseded the floorplan behavior gate with v3 readiness semantics. Exact fresh/cache floorplans and active-registry identity still require every configured camera, while BEV renderer health independently reconciles active, inactive-ready empty, and failed cameras; no occupied-camera N/N assumption remains. Focused replay, runner, ownership, health, and camera-local parity tests pass; fresh native DS9 evidence remains required._
  _2026-07-12 (Codex): Hardened DS9 behavior evidence as one immutable
  source-first cohort. Shared publication is create-if-absent with private
  modes, fsync, race-safe linking, bounded bytes, and fail-closed crash
  residue; an incomplete cohort requires a fresh session. Semantic v3 locks raw
  canonical bytes, Wholebody S/X lock the independently derived full source
  inventory, and V3DT locks every launch path plus the inspected ephemeral
  build mount. Appliance-run remains a separate non-promotable lifecycle.
  Focused merged tests passed; no native runtime was started._
  _2026-07-12 (Codex): An independent ownership-boundary audit found that the
  standalone producers rejected object-equivalent JSON reseals while promotion
  still accepted them. Ownership now invokes each producer's canonical encoder
  for every registered report and replay source, including resource-soak raw
  samples, before schema replay. Authentic producer, whitespace/key-order,
  selector, immutable-write, and promotion tests pass; no runtime was started._

## 12. Selector-Driven Appliance Deployment

- [x] Bind DS8 and DS9 to one closed selector, complete runtime snapshot, and
  selected mutable state release without runtime fallback.
  _2026-07-11 (Codex): Added strict canonical selector/state/health contracts,
  cross-language `noesis-runtime-v1` inventory parity, private state and build
  bindings, DS8 selector admission, DS9 sealed supervisor context and pre-exec
  revalidation, selector-bound REST/WS readiness, and appliance-only DS9 mounts
  for the selected build, world, identity, and analytics stores. Noesis acquires
  no nested deployment lease; Menon remains the full-lifetime shared-lease
  owner. Validated with generated-schema check, 23 focused Noesis/DS9 appliance
  tests within a 154-test combined DS8/contract gate, all 101 DS9 supervisor
  boundary tests, and 19 Menon cross-language deployment/release tests; no
  service, container, network, or GPU runtime was started._
  _2026-07-11 (Codex): Adversarial follow-up now re-admits the selector, state
  manifest/baseline, release root, mutable payload/parent inode cohort, and
  build directory immediately before DS8 state use and DS9 container exec.
  DS9 additionally requires the exact canonical owner-owned single-link
  supervisor and rejects duplicate container environment keys or inherited
  nested-lease variables. Focused compile and 130 appliance/supervisor/container
  boundary tests passed without launching a service, container, network, or
  GPU workload._
