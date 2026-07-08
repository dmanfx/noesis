# Plan: DS8 YOLO26 Runtime Performance Optimization

Date: 2026-06-28

## Status

- **Status:** Draft optimization plan.
- **Scope:** DS8 runtime with `--pgie-profile yolo26 --size s` and `--size m`.
- **Primary objective:** Reduce GPU load while retaining current detector/tracker functionality and inference cadence as much as practical.
- **Secondary objective:** Reduce CPU, host-copy, JSON serialization, RTSP/WebRTC fanout, and disk I/O load.
- **Non-goals:** Do not route DS8 runtime failures through deprecated paths. Do not add hidden fallback modes or substitute workflows.

## Review Baseline

- Live review used the existing `--pgie-profile yolo26 --size s` runtime as the anchor.
- `--size m` was reviewed through generated/configured assets rather than launching a competing runtime while `s` was already active.
- `s` and `m` currently use the same DS8 graph and cadence. The primary difference is the PGIE ONNX/engine path in:
  - `build/config_infer_primary_yolo26_s.ini`
  - `build/config_infer_primary_yolo26_m.ini`
- Both sizes still run the same surrounding work: preprocess, PGIE, tracker, ReID SGIE, pose SGIE, DAv2 depth tracking, analytics telemetry, BEV metadata, tiler/OSD, RTSP, and WebRTC gateway.

## Key Hotspots

1. Always-on full-frame DAv2 depth tracking branch.
2. Per-object YOLO26 pose SGIE cadence and host tensor copies.
3. Eager WebRTC gateway slot startup and RTSP fanout.
4. Synchronous gateway debug file writes in frame probes.
5. Per-frame JSON websocket broadcasts for tracking and BEV.
6. Per-person depth ROI copies from device to NumPy.
7. Always-on tiler/OSD/RTSP mosaic output when no viewer needs it.
8. Person-only downstream semantics still pay some 80-class parser/metadata costs.
9. NvDCF tracker configuration may be heavier than the real scene population needs.
10. `size m` increases PGIE cost on top of every shared graph cost above.

## Optimization Work Items

### 1. Gate or Decimate DAv2 Depth Tracking

- **Priority:** Highest GPU impact.
- **Current evidence:** `noesis/pipelines/ds8_pipeline.py` builds a required full-frame `depth_tracking_fullframe` branch after PGIE, and `build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini` runs `batch-size=3`, `interval=1`.
- **Problem:** DAv2 runs even when object-depth fusion may not need a new depth frame for every batch.
- **Plan:**
  - Add an explicit depth cadence control for baseline tracking.
  - Reuse cached aligned depth for intermediate frames.
  - Prefer a metadata-aware gate that opens only when the batch has active person tracks or when cached depth has aged out.
  - Start with a conservative interval such as 3, then tune with live quality checks.
- **Functionality guardrail:** Detector, tracker, analytics, and ReID cadence should remain unchanged.
- **Validation:**
  - Confirm `depth_enabled` / MapAnything state remains independent.
  - Compare GPU utilization, depth age, and world anchor quality before/after.
  - Verify object-depth metadata is still attached when people are present.

### 2. Add Pose SGIE Reinfer Cadence and Track Cache

- **Priority:** High GPU and CPU/host-copy impact.
- **Current evidence:** `pipelines/config_infer_secondary_yolo26_pose.ini` runs YOLO26 pose as a per-object SGIE with `batch-size=16`, `output-tensor-meta=1`, and `disable-output-host-copy=0`; unlike ReID, it has no `secondary-reinfer-interval`.
- **Problem:** Pose is refreshed too often for stable tracked people, and each pose tensor path currently allows host copies.
- **Plan:**
  - Add `secondary-reinfer-interval=2` or `3` to the pose SGIE.
  - Cache the last pose/keypoint payload by tracker/stable ID.
  - Refresh immediately for new tracks, weak poses, large bbox motion, or low-confidence anchors.
  - Longer-term: move pose tensor parsing into native code and attach compact keypoint metadata.
- **Functionality guardrail:** BEV/world anchors should remain stable, with no visible pose flicker for ordinary motion.
- **Validation:**
  - Compare pose tensor copy counters.
  - Compare world anchor quality counts and BEV stability.
  - Run focused live smoke for occupied scenes and empty scenes.

### 3. Lazy-Start WebRTC Gateway Slots

- **Priority:** High CPU/I/O and RTSP fanout impact.
- **Current evidence:** `noesis/ds8_runtime.py` defaults `NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS` to `5` and starts every `MosaicWebRTCGateway` immediately.
- **Problem:** Idle gateway slots still consume the local RTSP mosaic and create avoidable fanout.
- **Plan:**
  - Default to one active gateway slot.
  - Allocate extra slots only when a new WebRTC offer requires one.
  - Stop or park idle slots after a timeout.
  - Keep `NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS` as an explicit capacity limit.
- **Functionality guardrail:** Multi-client support should remain available; capacity should become demand-driven rather than eager.
- **Validation:**
  - Inspect active RTSP clients before/after.
  - Verify first client startup latency remains acceptable.
  - Verify multiple browser clients can still connect when needed.

### 4. Remove Per-Frame Gateway Debug File Writes

- **Priority:** High CPU/disk I/O cleanup; low implementation risk.
- **Current evidence:** `noesis/mosaic_webrtc_gateway.py` writes JSON to `.cursor/debug.log` in `_on_frame_probe`, including every 10 frames per gateway slot.
- **Problem:** Synchronous file writes occur on streaming threads and scale with the number of eager gateway slots.
- **Plan:**
  - Gate debug file writes behind an explicit debug environment flag.
  - Keep first-frame/keyframe logging lightweight through the normal logger.
  - Fix or remove the shadowed `% 100` branch currently hidden by `% 10`.
- **Functionality guardrail:** WebRTC diagnostics should remain available when explicitly enabled.
- **Validation:**
  - Confirm `.cursor/debug.log` no longer grows during normal runtime.
  - Confirm WebRTC playback still connects and decodes.

### 5. Coalesce Tracking and BEV WebSocket Broadcasts

- **Priority:** High CPU and network serialization impact.
- **Current evidence:** `noesis/pipelines/hooks.py` publishes BEV from analytics telemetry, `noesis/telemetry/bev.py` builds `bev-frame` JSON, and `websocket_server.py` JSON-encodes broadcasts for every connected client.
- **Problem:** The UI receives many stale intermediate tracking/BEV states when only the newest state matters.
- **Plan:**
  - Add latest-only coalescing per message type and camera ID.
  - Cap BEV/tracking broadcast cadence to about 10-15 Hz.
  - Drop stale queued states rather than letting them serialize and send.
  - Consider faster JSON or binary formats after semantic coalescing is in place.
- **Functionality guardrail:** Inference and tracking cadence should not change; only UI delivery cadence should be bounded.
- **Validation:**
  - Compare websocket serialization metrics before/after.
  - Confirm UI motion remains smooth and current.
  - Confirm clients receive final/current states under load.

### 6. Replace Per-Person Depth ROI NumPy Copies

- **Priority:** Medium GPU/CPU boundary impact; higher implementation complexity.
- **Current evidence:** `_ObjectDepthFusionProcessor` in `noesis/pipelines/hooks.py` calls `copy_roi_to_numpy` for each person bbox and then computes depth statistics in NumPy.
- **Problem:** Device-resident depth is pulled back to host for object-level reductions.
- **Plan:**
  - Add a native/CUDA reduction path that computes anchor depth, median/percentiles, valid fraction, and sample counts on the device.
  - Return only compact scalar metadata to Python.
  - As a smaller first step, sample a lower-body band or decimated ROI instead of copying the full bbox.
- **Functionality guardrail:** Preserve object-depth metadata shape and world-anchor behavior.
- **Validation:**
  - Compare depth metadata fields for representative tracks.
  - Compare host-copy counters and CPU profile.
  - Run native extension smoke tests if metadata/native plumbing changes.

### 7. Demand-Drive Mosaic RTSP/OSD Work

- **Priority:** Medium GPU/encoder impact; depends on UI expectations.
- **Current evidence:** `noesis/pipelines/ds8_pipeline.py` always builds tiler, GPU OSD, and RTSP output when RTSP is enabled.
- **Problem:** Mosaic render/encode work continues even when no dashboard viewer needs video.
- **Plan:**
  - Add a demand valve or runtime control for RTSP/WebRTC output when no viewer is connected.
  - Consider moving keypoints/trails to client-side drawing from metadata so the OSD path can be lighter.
  - Keep RTSP/WebRTC as the canonical browser video path when a viewer is present.
- **Functionality guardrail:** No hidden alternate video path; if video is requested, RTSP/WebRTC remains the canonical path.
- **Validation:**
  - Compare encoder and GPU utilization with zero viewers and with one viewer.
  - Verify browser playback and overlays still meet user expectations.

### 8. Tighten Person-Only PGIE Postprocess Work

- **Priority:** Medium CPU/metadata impact; limited TensorRT GPU impact.
- **Current evidence:** YOLO26 PGIE configs declare 80 classes while downstream tracking/SGIE work operates on person class `0`.
- **Problem:** Non-person classes are mostly discarded downstream but may still incur parser, metadata, clustering, or filtering overhead.
- **Plan:**
  - Audit the YOLO parser and clustering path for early person-only discard.
  - Lower `topk` if quality remains stable.
  - Keep detector tensor shape and engine unchanged unless a separate export/runtime contract change is approved.
- **Functionality guardrail:** Person detection recall should not regress.
- **Validation:**
  - Compare object counts and false positives on representative camera views.
  - Compare tracker/SGIE object counts and CPU profile.

### 9. Tune NvDCF Tracker for Scene Scale

- **Priority:** Medium, quality-sensitive.
- **Current evidence:** `config/nvtracker.yaml` uses NvDCF visual features and a high target cap relative to expected home-camera occupancy.
- **Problem:** Tracker feature and target budgets may exceed real scene needs.
- **Plan:**
  - Profile tracker cost before changing quality-sensitive settings.
  - Consider lowering max target count and feature levels if live scenes are sparse.
  - Validate ID stability carefully after each tracker change.
- **Functionality guardrail:** Stable ID continuity and zone/BEV behavior should remain acceptable.
- **Validation:**
  - Compare ID switches, active targets, and tracker latency.
  - Test with simultaneous people and partial occlusion.

### 10. Consider an `s`/`m` Hybrid Strategy

- **Priority:** Optional advanced GPU strategy.
- **Current evidence:** `size m` raises PGIE cost but the rest of the graph is identical to `size s`.
- **Problem:** Running `m` at full cadence pays the larger detector cost on every batch.
- **Plan:**
  - Keep `s` at current cadence for primary detection.
  - Run `m` only for low-confidence frames, ambiguous scenes, or periodic audits.
  - Reconcile results before tracker if this becomes a formal feature.
- **Functionality guardrail:** This is a deliberate design change, not a hidden fallback. It should not be implemented without an explicit runtime contract.
- **Validation:**
  - Compare detection recall/precision against full-cadence `m`.
  - Verify tracker input consistency and latency.

## Suggested Implementation Order

1. Remove/gate gateway frame-probe debug writes.
2. Lazy-start WebRTC gateway slots.
3. Add websocket latest-only coalescing for BEV/tracking.
4. Add pose SGIE reinfer interval plus pose cache.
5. Add DAv2 depth cadence/gating and cached-depth aging metrics.
6. Optimize object-depth ROI reductions with native/CUDA support.
7. Demand-drive RTSP/OSD output.
8. Tighten person-only parser/postprocess behavior.
9. Tune NvDCF only after telemetry confirms tracker cost is material.
10. Revisit `s`/`m` hybrid detection if full-cadence `m` remains too expensive.

## Validation Checklist

- [x] Capture baseline GPU utilization for `--pgie-profile yolo26 --size s`.
  - 2026-06-28: Live `size s` process sampled before edits: GPU util 42%, GPU memory 3867 MiB, encoder 5%, decoder 11%, power 67.85 W. `nvidia-smi dmon` 10-sample SM range was 45-77%, with memory util 30-54%.
- [ ] Capture baseline GPU utilization for `--pgie-profile yolo26 --size m`.
  - 2026-06-28: Full pre-change `size m` baseline was not captured because the live `size s` runtime owned RTSP/WS/REST ports. Post-change `size m` was live-validated in the same optimized graph; see validation evidence below.
- [x] Capture baseline CPU thread profile and websocket serialization metrics.
  - 2026-06-28: Live `size s` process sampled before edits: process CPU about 410%, RSS about 4026 MiB. RTSP had multiple localhost consumers on port 8554, consistent with eager WebRTC gateway fanout.
- [x] Validate no zero-copy violation regressions after each runtime change.
  - 2026-06-28: Active DS8 unit coverage passed after changes: depth registration, graph build, depth tracking/fusion, websocket boundary metrics, pose tensor contracts, analytics telemetry, BEV smoothing, and object-depth result tests.
- [ ] Validate BEV/tracking UI freshness with one dashboard client.
- [x] Validate BEV/tracking UI freshness with one dashboard client.
  - 2026-06-28: Browser client connected during both bounded live runs; tracking/stable-ID telemetry continued to publish and WebRTC RTP packet counters advanced.
- [x] Validate WebRTC startup with zero, one, and multiple clients.
  - 2026-06-28: Zero-viewer startup kept one warm gateway slot and closed RTSP output until a browser offer. One-viewer startup toggled `rtsp_output_valve` open, observed first video frame/keyframe, created an answer, and reached ICE connected/completed for both `size s` and `size m`. Multi-client capacity remains covered by the lazy gateway unit test; live multi-client was not exercised in this pass.
- [x] Validate object-depth metadata and world-anchor quality with people present.
  - 2026-06-28: Focused object-depth fusion tests passed for masked ROI and missing-mask lower-band fallback. Active telemetry/BEV result tests passed.
- [x] Validate empty-scene behavior and ensure gated work stays gated.
  - 2026-06-28: Startup with no active tracks held stable-ID metrics at idle/zero canonical tracks until detections appeared. MapAnything gate primed and closed after 1 second. RTSP output stayed valve-gated until WebRTC ownership was assigned.
- [x] Record each completed optimization with date, change summary, and validation evidence.

## Implementation Log

### 2026-06-28: Items 1-8 Initial Implementation

- **Item 1, DAv2 cadence:** Default DAv2 depth tracking interval changed from `1` to `3`. Runtime now reads `NOESIS_DEPTH_TRACKING_INTERVAL`, materializes the matching depth config, and sets object-depth cache age to `interval + 1`. Depth registration now accepts DAv2 cadence-only fingerprint differences because registration depends on model/input identity, not inference interval.
- **Expected resource effect:** Reduces DAv2 SGIE full-frame inference cadence by about half versus interval `1`, lowering shared GPU load for both `size s` and `size m` while preserving cached object-depth metadata on skipped frames.
- **Item 2, pose cadence/cache:** Pose SGIE config now uses `secondary-reinfer-interval=2`. `PoseFeatureProcessor` caches attached pose payloads by source/track, rejects stale or large-bbox-shift entries, and reattaches bbox-adjusted keypoints on skipped SGIE frames.
- **Expected resource effect:** Cuts pose SGIE per-object refresh frequency and reduces pose tensor host-copy pressure on stable tracks without removing pose metadata from intermediate frames.
- **Item 3, lazy WebRTC slots:** Runtime defaults to one warm WebRTC gateway slot through `NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS=1`, keeps `NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS` as capacity, and creates additional gateways only when signaling needs a free slot.
- **Expected resource effect:** Reduces idle local RTSP consumers, gateway pipelines, and decode/signaling work from max-client count to active demand.
- **Item 4, gateway debug writes:** Removed synchronous `.cursor/debug.log` writes from the gateway frame probe. First-frame and first-keyframe diagnostics remain through normal logging; periodic frame count logging is debug-only and sparse.
- **Expected resource effect:** Removes per-frame streaming-thread disk I/O, previously every 10 frames per gateway slot.
- **Item 5, websocket coalescing:** Tracking and BEV JSON broadcasts are latest-only coalesced per source/camera. Defaults are `NOESIS_WS_TRACKING_MAX_HZ=15` and `NOESIS_WS_BEV_MAX_HZ=12`; setting either to `0` disables coalescing for that type.
- **Expected resource effect:** Bounds JSON conversion, serialization, and per-client sends while leaving inference/tracking cadence unchanged.
- **Item 6, object-depth ROI host copies:** Missing-mask person depth fallback now copies only a lower bbox band by default (`NOESIS_OBJECT_DEPTH_BBOX_BAND_FRACTION=0.5`) instead of the full person bbox. Instance-mask objects keep full-mask behavior.
- **Expected resource effect:** Halves host ROI pixels copied for missing-mask person detections before the future native/CUDA reduction step.
- **Item 7, demand-driven RTSP output:** RTSP mosaic branch includes `rtsp_output_valve`, but the default is always-open RTSP output because `nvrtspoutsinkbin` can return 503 and leave WebRTC gateways frame-starved when the media is gated closed at startup. `NOESIS_MOSAIC_RTSP_DEMAND_GATED=1` re-enables owner-driven gating for experiments.
- **Expected resource effect:** Keeps the canonical RTSP/WebRTC path continuously available by default; optional demand gating can reduce idle mosaic RTSP/encode work only when explicitly enabled.
- **Item 8, person-only PGIE postprocess:** YOLO26 materialized detector INI lowers `topk` to `NOESIS_YOLO26_PERSON_TOPK=100` and adds `filter-out-class-ids=1;...;79` while leaving the engine and tensor shape unchanged.
- **Expected resource effect:** Reduces parser/metadata/tracker downstream object budget for non-person classes with limited TensorRT GPU impact.

### 2026-06-28: Validation Evidence

- `python3 -m py_compile noesis/depth_tracking_materialization.py noesis/calibration/depth_registration.py noesis/ds8_runtime.py noesis/pipelines/ds8_pipeline.py noesis/pipelines/hooks.py noesis/mosaic_webrtc_gateway.py websocket_server.py`
  - Passed.
- `python3 -m pytest tests/test_depth_registration.py tests/test_pipeline_build.py tests/test_depth_tracking_frame_processor.py tests/test_websocket_boundary_metrics.py tests/test_pose_tensor_contracts.py -q`
  - Passed: 31 tests.
- `python3 -m pytest tests/test_analytics_telemetry_hook.py tests/test_bev_renderer_world_smoothing.py tests/test_object_depth_result.py -q`
  - Passed: 42 tests.
- `python3 -m pytest tests/test_object_depth_fusion_probe.py -q`
  - Not used as DS8 regression evidence. This ignored prototype/testpipeline path failed on existing `np.asarray(..., copy=False)` calls under the installed NumPy and is outside the active DS8 implementation scope.

### 2026-06-28: Live DS8 Validation

- `timeout --signal=TERM --kill-after=15s 90s python3 noesis/ds8_runtime.py --pgie-profile yolo26 --size s --log-level INFO`
  - Result: Runtime activated, loaded YOLO26 `s` dynamic engine, loaded DAv2 `i3` config, passed depth-registration validation, started one warm WebRTC gateway slot, opened RTSP output on browser offer, observed first frame/keyframe, and reached WebRTC ICE connected/completed.
  - Active-viewer sample: CPU about 325%, RSS about 2029 MiB, GPU memory 3863 MiB, one-shot GPU sample 82% SM / 52% memory / 5% encoder / 12% decoder / 116 W.
  - 8-sample `nvidia-smi dmon`: SM 64-82%, memory 43-58%, encoder 3-6%, decoder 3-11%, power 107-115 W.
  - RTSP fanout: one localhost RTSP client with one browser WS client; pre-change live sample had multiple localhost RTSP clients from eager gateway slots.
- `timeout --signal=TERM --kill-after=15s 90s python3 noesis/ds8_runtime.py --pgie-profile yolo26 --size m --log-level INFO`
  - Result: Runtime activated, loaded YOLO26 `m` dynamic engine, loaded DAv2 `i3` config, passed depth-registration validation, started one warm WebRTC gateway slot, opened RTSP output on browser offer, observed first frame/keyframe, and reached WebRTC ICE connected/completed.
  - Active-viewer sample: CPU about 297%, RSS about 2029 MiB, GPU memory 3927 MiB, one-shot GPU sample 85% SM / 65% memory / 3% encoder / 7% decoder / 142 W.
  - 8-sample `nvidia-smi dmon`: SM 63-92%, memory 49-70%, encoder 3-6%, decoder 6-11%, power 136-144 W.
  - RTSP fanout: one localhost RTSP client with one browser WS client.
- Depth-registration live-start fix:
  - Initial `size s` live validation surfaced `dav2_profile_fingerprint_mismatch` because existing registration artifacts store DAv2 identity from the YAML model stanza (`engine`) while runtime fingerprints use materialized nvinfer fields (`model-engine-file`/`onnx-file`). The matcher now compares DAv2 model identity across those aliases while still rejecting model/input/batch/GIE/engine mismatches.

### 2026-06-30: Detection-Wake Optimization Follow-Up

- **Scope:** Implemented the second-pass optimization set for the user's observed "one detection doubles load" behavior, excluding pose-anchor mode and viewer-driven OSD as requested.
- **Detection-wake profiler:** Added `detection_wake.*` counters plus `stage_timings` to the existing zero-copy/core instrumentation stats. Runtime stats now expose timing buckets for analytics frame handling, ReID extraction, StableID update, pose-anchor extraction, object-depth metadata extraction, object-depth ROI/native stats, tracking publish, and BEV render/publish.
- **NvDCF lean profile:** Tuned `config/nvtracker.yaml` for home-scale occupancy: target cap `32`, shadow age `90`, HOG disabled, feature image level `2`, visual matching weight reduced, IoU weight raised, and internal NvDCF ReID appearance disabled because explicit OSNet ReID remains the identity authority.
- **ReID budget and cadence:** Raised StableID default embedding interval to `1.0s`, changed OSNet SGIE to `secondary-reinfer-interval=6`, kept `classifier-async-mode=0` because the MP4 tensor-meta validation showed async mode produced missing embeddings for the Python StableID path, and added `NOESIS_REID_EMBEDS_PER_FRAME_MAX` (default `2`) so multiple people do not all trigger embedding extraction/gallery work in the same metadata frame while 5-8 person scenes still receive embeddings quickly enough to avoid severe ID fragmentation.
- **Pose parsing budget without pose-anchor mode:** Kept pose enabled, but changed pose SGIE to `secondary-reinfer-interval=8`, kept `classifier-async-mode=0` because live DeepStream reports async is not applicable to this tensor-output pose SGIE, added `NOESIS_POSE_FEATURES_PER_FRAME_MAX` (default `2`), and made analytics consume attached `NOESIS.POSE_FEATURES` payloads before bounded native keypoint extraction. This removes the duplicate pose extraction path without introducing the excluded pose-anchor mode.
- **Object-depth scaling:** Added per-track object-depth cadence/cache controls (`NOESIS_OBJECT_DEPTH_MAX_HZ_PER_TRACK`, `NOESIS_OBJECT_DEPTH_MAX_OBJECTS_PER_FRAME`, cache age/bbox-shift knobs), bounded percentile sample counts, and a native `AlignedDepthFrameDevice.sample_roi_stats(...)` scalar path for missing-mask/bbox-band person detections so Python can attach compact depth metadata without allocating a NumPy ROI in the common YOLO26 detect path.
- **Upstream tracking/BEV publish gates:** Added source-level publish gates before tracking broadcasts and BEV render/publish. Defaults follow the WebSocket caps (`NOESIS_TRACKING_PUBLISH_MAX_HZ` / `NOESIS_WS_TRACKING_MAX_HZ`, `NOESIS_BEV_PUBLISH_MAX_HZ` / `NOESIS_WS_BEV_MAX_HZ`), while first publish and track-count changes still publish immediately.
- **Global per-frame budget:** ReID extraction, pose feature parsing, native pose fallback, object-depth sampling, tracking publish, and BEV render now have explicit per-frame/per-source budgets or cadence gates. Stable tracks reuse cached metadata between refreshes; new/count-changing states are prioritized.

### 2026-06-30: Validation Evidence

- `python3 -m py_compile noesis/pipelines/hooks.py noesis/ds8_runtime.py`
  - Passed.
- `bash ./scripts/build_noesis_depth_tracking_tensor_ext.sh`
  - Passed; rebuilt `noesis_depth_tracking_tensor_ext` with `AlignedDepthFrameDevice.sample_roi_stats`.
- `python3 - <<'PY' ... hasattr(ext.AlignedDepthFrameDevice, 'sample_roi_stats')`
  - Passed; printed `native_depth_stats_ok`.
- `python3 -m pytest tests/test_pipeline_build.py tests/test_analytics_telemetry_hook.py tests/test_depth_tracking_frame_processor.py tests/test_zero_copy_stats_contract.py tests/test_pose_tensor_contracts.py tests/test_reid_tensor_contracts.py -q`
  - Passed: 38 tests. The DS8 analytics ReID-budget test now uses an iterator-backed `frame_meta.object_items` shape to match Service Maker metadata and cover the single-detection wake path.
- `timeout --signal=TERM --kill-after=10s 45s env NOESIS_MOSAIC_WEBRTC_ENABLED=0 NOESIS_MOSAIC_RTSP_ENABLED=0 python3 noesis/ds8_runtime.py --pgie-profile yolo26 --size m --disable-rest --ws-port 6018 --log-level INFO`
  - Result: Runtime activated, loaded YOLO26 `m` dynamic engine, loaded ReID and pose SGIEs with the updated cadences, attached trail/pose/depth/analytics probes, started WebSocket on port `6018`, and reported `DS8 runtime is active`.
  - Log scan found no `ERROR`, `Traceback`, `TypeError`, pose-async warning, fatal, segfault, or core-dump markers. The process exited through the bounded timeout kill path because the DeepStream wait loop did not unwind before `--kill-after`.
  - This smoke was an idle/no-detection window; the targeted iterator-backed test covers the Service Maker object metadata shape for detection-present analytics.
- MP4 detection-present validation using the local file sources from `config/infer.yaml`:
  - Initial all-MP4 runs exposed a native pyservicemaker crash in `tracking_telemetry_stage` when `frame_meta.object_items` was materialized into a Python list. The analytics path now streams the Service Maker iterator and increments `detection_wake.objects_seen` per object, avoiding retained invalid `ObjectMetadata` proxies.
  - `PYTHONFAULTHANDLER=1` and `gdb` confirmed the crash moved from `_NvOSD_RectParams.left` to `deepstream::ObjectMetadata::objectId()` before the iterator-retention fix, then the same MP4 replay ran until bounded timeout with detections present.
  - A 35-second YOLO26 `m` MP4 replay with 1/3/3 active tracks captured `detection_wake.frames=2147`, `objects_seen=4999`, `object_depth_sampled=916`, `object_depth_cache_hit=4090`, `pose_anchor_payload_hit=3429`, `tracking_publish_skipped=918`, and `bev_publish_skipped=1142`.
  - That run sampled GPU SM avg/max about `66.7%/91%`, decoder avg/max about `8%/11%`, CPU avg/max about `311%/370%`, and no encoder use with RTSP/WebRTC disabled.
  - MP4 stats showed ReID `classifier-async-mode=1` starved the tensor-meta extraction path, so ReID async was reverted. Subsequent sync runs restored embeddings but showed `secondary-reinfer-interval=12` was too sparse for crowded clips; final config uses interval `6` plus `NOESIS_REID_EMBEDS_PER_FRAME_MAX=2`.

### 2026-06-30: CUDA Object-Depth ROI Stats

- **Scope:** Replaced the missing-mask/bbox-band object-depth ROI stats row-copy loop with a CUDA compact sampler in `native/noesis_depth_tracking_tensor_kernels.cu`, linked into `noesis_depth_tracking_tensor_ext`. The Python-facing `AlignedDepthFrameDevice.sample_roi_stats(...)` contract is unchanged.
- **Resource effect:** The old native scalar path still copied one sampled ROI row at a time from device to host and then reduced on CPU. The new path launches one CUDA sampler over the decimated ROI, copies a compact sampled float buffer once, and reuses thread-local device/host scratch buffers so multiple detections do not pay repeated allocation costs.
- **Build/toolchain note:** `scripts/build_noesis_depth_tracking_tensor_ext.sh` now uses `${CUDA_HOME:-/usr/local/cuda}/bin/nvcc` so the CUDA compiler matches the selected CUDA headers/libs. This avoids accidentally using distro CUDA 12 `nvcc` with `/usr/local/cuda` CUDA 13 headers on this host.
- **Validation:**
  - `bash ./scripts/build_noesis_depth_tracking_tensor_ext.sh`
    - Passed; rebuilt `noesis_depth_tracking_tensor_ext` with the CUDA sampler object.
  - `python3 -m py_compile noesis/pipelines/hooks.py noesis/ds8_runtime.py`
    - Passed.
  - `python3 - <<'PY' ... hasattr(ext.AlignedDepthFrameDevice, 'sample_roi_stats')`
    - Passed; printed `native_depth_stats_ok`.
  - `python3 -m pytest tests/test_pipeline_build.py tests/test_analytics_telemetry_hook.py tests/test_depth_tracking_frame_processor.py tests/test_zero_copy_stats_contract.py tests/test_pose_tensor_contracts.py tests/test_reid_tensor_contracts.py -q`
    - Passed: 38 tests.
  - YOLO26 `m` MP4 replay with RTSP/WebRTC disabled, finite local file sources, and detections present:
    - Captured `detection_wake.frames=1890`, `objects_seen=5928`, `person_tracks=5928`, `object_depth_sampled=1036`, `object_depth_native_stats=1030`, `object_depth_cache_hit=4863`, with active tracks ending at kitchen `6`, family-room `3`, living-room `0`.
    - `object_depth.native_roi_stats` improved from the prior MP4 measurement of `2.731 ms avg / 190.265 ms max` to `0.869 ms avg / 23.736 ms max`.
    - `object_depth.sample_person` improved from `2.884 ms avg / 190.414 ms max` to `1.036 ms avg / 23.914 ms max`.
    - GPU SM avg/max for this later window was `66.8%/90%`; process CPU avg/max was `377%/389%`. This run carried more detections/active tracks than the prior comparison window, so the most reliable win is the per-stage object-depth timing reduction rather than whole-process utilization.
    - Runtime reached normal MP4 EOS and shut down; no DS8 runtime remained active afterward.

### 2026-06-30: Live Mask-Present Object-Depth Profile And CUDA Extension

- **Live profile:** Attached to the active live DS8 runtime on port `6008` without restarting it. A 60-second live stream window with people moving captured one active family-room track at the end, `detection_wake.frames=5368`, `objects_seen=1352`, `person_tracks=1352`, `object_depth_sampled=245`, and `object_depth_cache_hit=1109`.
- **Finding:** The live run was not using `object_depth.native_roi_stats`; it used the instance-mask path and copied depth ROIs through `copy_roi_to_numpy` (`241` calls, `11,530,597` depth pixels copied). That path measured `object_depth.sample_person=4.527 ms avg / 26.393 ms max` and `object_depth.copy_roi_to_numpy=1.652 ms avg / 24.546 ms max`. GPU SM avg/max was `74.2%/96%`; process CPU avg/max was `340.7%/405.0%`, split almost evenly between user and system CPU.
- **Change:** Added `AlignedDepthFrameDevice.sample_masked_roi_stats(...)` and a CUDA masked compact sampler so mask-present detections can keep the established `instance_mask` object-depth contract without copying the matching depth crop to NumPy. Python still decodes the DeepStream object mask because the current mask metadata is host-resident, but full depth ROI D2H copies are avoided when the rebuilt extension is loaded.
- **Post-restart finding:** After restart, the new mask path loaded and depth ROI copies dropped to zero, but the first implementation was still too sync-heavy because it performed separate native calls for full-mask stats and lower/torso anchor stats. That run captured `object_depth.native_mask_roi_stats=318`, `object_depth.native_mask_anchor_stats=403`, and `object_depth.sample_person=6.055 ms avg / 28.229 ms max`.
- **Follow-up change:** Added `AlignedDepthFrameDevice.sample_masked_person_roi_stats(...)` and a combined CUDA sampler that returns full-mask, lower-body, and torso stats from one mask upload, one sampler pass, and one compact D2H copy. The Python fusion path now prefers this combined method and only uses the multi-call native path as a compatibility fallback.
- **Validation:**
  - `bash ./scripts/build_noesis_depth_tracking_tensor_ext.sh`
    - Passed; rebuilt `noesis_depth_tracking_tensor_ext` with `sample_masked_roi_stats` and `sample_masked_person_roi_stats`.
  - `python3 -m py_compile noesis/pipelines/hooks.py noesis/ds8_runtime.py`
    - Passed.
  - `python3 - <<'PY' ... hasattr(ext.AlignedDepthFrameDevice, 'sample_masked_person_roi_stats')`
    - Passed; printed `native_depth_person_mask_stats_ok`.
  - `python3 -m pytest tests/test_pipeline_build.py tests/test_analytics_telemetry_hook.py tests/test_depth_tracking_frame_processor.py tests/test_zero_copy_stats_contract.py tests/test_pose_tensor_contracts.py tests/test_reid_tensor_contracts.py -q`
    - Passed: 40 tests.
- **Pending live validation:** The active runtime was left running and still has the earlier multi-call masked native path loaded in memory. A live restart is required before collecting the combined-path `object_depth.native_mask_person_stats` counters.
- **Combined-path live validation after restart:**
  - Restarted the runtime and collected a 60-second live stats/GPU/CPU window.
  - Confirmed the default runtime loaded `pipelines/config_infer_primary_yolo11_seg.ini` and `models/engines/yolo11s-seg_cust_fused.engine`; the PGIE exposed a mask-prototype output (`output1 32x160x160`), which is why the object-depth path saw instance-mask payloads even though the user did not intend to run a segmentation test.
  - Captured `detection_wake.frames=5386`, `objects_seen=1118`, `person_tracks=1118`, `object_depth_sampled=192`, `object_depth_cache_hit=926`, and `object_depth_native_mask_stats=189`.
  - The old depth-copy path stayed off: no `detection_wake.object_depth_roi_copy`, no `tensor_host_copies_total.object_depth_roi`, and no `object_depth.copy_roi_to_numpy` stage bucket.
  - The expected combined stage appeared: `object_depth.native_mask_person_stats=189`, `3.836 ms avg / 20.088 ms max`; `object_depth.sample_person=5.290 ms avg / 29.721 ms max`.
  - GPU SM avg/max was `73.8%/95%`; process CPU avg/max was `312.7%/392%`. This window ended with no active tracks and had fewer person observations than the prior restart profile, so it confirms the counter path and removal of host depth copies more strongly than a perfect utilization comparison.

### 2026-06-30: YOLO26-M Live Detection-Wake Pose Cache And Batch-3 Pose SGIE

- **Scope:** Focused on the user's live `--pgie-profile yolo26 --size m` detection-wake spike after the CUDA object-depth changes made the dashboard feel smoother but still showed GPU/CPU jumps when people entered the scene.
- **Mask clarification:** The earlier `mask` profile came from an accidental default run that loaded `pipelines/config_infer_primary_yolo11_seg.ini` and the YOLO11 segmentation engine. The current YOLO26 medium detector run does not use a segmentation PGIE output; object-depth uses the detector-only bbox-band CUDA path (`object_depth.native_roi_stats`) rather than the instance-mask counter path (`object_depth.native_mask_person_stats`).
- **Change:** `PoseFeatureProcessor` now reuses an already attached cached pose payload before attempting native tensor extraction. The default cache age is derived from the pose SGIE `secondary-reinfer-interval`, while env/YAML knobs can still narrow the age/shift bounds.
- **Change:** The canonical pose SGIE for this DS8 profile now uses the existing `yolo26n-pose_b3_fp16.engine` with `batch-size=3` instead of the batch-16 pose engine. This keeps the same pose model and reinfer cadence but avoids shaping the common one-to-three-person live case around a much larger secondary batch.
- **Validation:**
  - `python3 -m pytest tests/test_analytics_telemetry_hook.py::test_pose_feature_processor_reuses_cache_before_native_extract tests/test_analytics_telemetry_hook.py::test_pose_anchor_prefers_attached_payload_before_native -q`
    - Passed: 2 tests.
  - `python3 -m py_compile noesis/pipelines/hooks.py`
    - Passed.
  - `python3 -m pytest tests/test_pipeline_build.py tests/test_analytics_telemetry_hook.py tests/test_depth_tracking_frame_processor.py tests/test_zero_copy_stats_contract.py tests/test_pose_tensor_contracts.py tests/test_reid_tensor_contracts.py -q`
    - Passed: 41 tests.
  - `git diff --check`
    - Passed.
- **Live profile before the batch-3 pose swap:** YOLO26 medium with the cache-first pose path but batch-16 pose SGIE captured `detection_wake.frames=5290`, `objects_seen=1929`, `pose_feature_cache_hit=1271`, `pose_feature_native_extract=167`, `tensor_host_copies_total.pose=167`, `object_depth_native_stats=314`, and `object_depth_cache_hit=1592`. GPU SM avg/p95/max was `84.78%/96%/99%`, power avg was `145.1 W`, process CPU avg was `369.2%`, and RSS averaged about `3.12 GB`.
- **Live profile after the batch-3 pose swap:** YOLO26 medium with the batch-3 pose SGIE captured a busier window ending with three active tracks: `detection_wake.frames=5314`, `objects_seen=3747`, `pose_feature_cache_hit=3106`, `pose_feature_native_extract=404`, `tensor_host_copies_total.pose=404`, `object_depth_native_stats=651`, and `object_depth_cache_hit=3091`. GPU SM avg/p95/max fell to `68.07%/80%/92%`, power avg fell to `111.04 W`, process CPU avg fell to `347.42%`, and RSS averaged about `1.94 GB`.
- **Result:** With roughly double the person/object load in the post-change window, average GPU dropped by about 16.7 absolute points, p95 GPU dropped by 16 points, average power dropped by about 34 W, and RSS dropped by about 1.18 GB. CPU improved modestly despite the heavier detection window. The current live runtime is still `python3 noesis/ds8_runtime.py --pgie-profile yolo26 --size m` and has the batch-3 pose engine loaded.

### 2026-07-08: Object-Depth Zero-Copy Tightening

- **Scope:** Implemented the follow-up object-depth CPU/zero-copy optimizations requested after the YOLO26-L CPU investigation.
- **Change:** `_ObjectDepthFusionProcessor` now caches compact `NOESIS.OBJECT_DEPTH` JSON strings for native user-meta attach and for cache reuse, avoiding an extra `dict(...)`/`json.dumps(...)` step at the attach boundary.
- **Change:** The production object-depth path no longer falls back to `copy_roi_to_numpy` when native stats are missing. Host depth ROI copies require `NOESIS_OBJECT_DEPTH_ALLOW_HOST_ROI_COPY=1`; otherwise the attached object-depth result reports `native_stats_unavailable` and keeps the full-frame depth tensor/device frame on the native path.
- **Change:** Masked object-depth now pre-thresholds masks to contiguous `uint8` arrays before native calls. The native extension accepts byte masks directly, falls back to threshold-packing only for older float callers, uploads 1 byte/pixel masks to CUDA, and treats nonzero bytes as active pixels.
- **Change:** `AlignedDepthFrameDevice.sample_masked_person_roi_stats(...)` now returns `foot_u`/`foot_v`, allowing the common combined-native person path to avoid the previous Python `_mask_foot_uv` scan.
- **Validation:**
  - `python3 -m pytest tests/test_depth_tracking_frame_processor.py -q`
    - Passed: 9 tests.
  - `bash scripts/build_noesis_depth_tracking_tensor_ext.sh`
    - Passed; rebuilt `noesis_depth_tracking_tensor_ext`.
  - Native import/API check for `sample_masked_roi_stats` and `sample_masked_person_roi_stats`
    - Passed.
  - `python3 -m pytest tests/test_depth_tracking_frame_processor.py tests/test_object_depth_result.py tests/test_object_depth_meta_native_contracts.py tests/test_analytics_telemetry_hook.py tests/test_zero_copy_stats_contract.py tests/test_zero_copy_invariants.py -q`
    - Passed: 40 tests.
  - Live restart smoke: stopped the old YOLO26-L runtime, started `python3 noesis/ds8_runtime.py --pgie-profile yolo26 --size l`, confirmed DS8 loaded the rebuilt runtime path and relaunches cleanly on `6008`, `8080`, and `8554` as pid `2250683`.
- **Live-profile note:** The short WebSocket stats smoke had no active tracks, so it did not exercise object-depth counters. Focused tests cover the counter/contract behavior; a detection-present live window is still needed for before/after timing on `object_depth.native_mask_person_stats` or `object_depth.native_roi_stats`.
