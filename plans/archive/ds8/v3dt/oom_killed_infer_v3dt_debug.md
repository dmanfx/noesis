# Debug Notes: DS8 SV3DT OOM (“Killed”) Root Cause + Fix

**Date:** 2025-12-29  
**Repo:** `/home/mayor/Noesis_Devel`  
**Stack:** DS8 canonical (`noesis/`)  

## 1) Summary (what was happening)

Running DS8 with the SV3DT pipeline config:

```bash
python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_sv3dt.yaml
```

starts successfully, loads engines (YOLO, MapAnything, OSNet SGIE, nvtracker ReID + BodyPose3DNet), enters `Main Loop Running...`, then after some runtime the process prints:

```
Killed
```

This is the Linux kernel OOM killer terminating `python3` due to host RAM exhaustion.

Baseline DS8 startup **without** specifying `--pipeline-config` (default config) does **not** show the same behavior.

Key observation: the memory climb happens **even when no WebSocket clients connect**, so this is not purely “WS broadcast backlog”.

**Trigger point (from runtime logs):** RAM begins climbing only after the pipeline is fully constructed and has entered steady-state (after `Main Loop Running...` and the first ROI-exclude/analytics setup messages), then increases roughly linearly until the kernel OOM-kills the process.

**Status:** fixed. Root cause was an upstream DeepStream `gst-nvtracker` host-memory leak triggered by SV3DT/3D projection outputs; see §5.
Additional SV3DT runaway-RSS behavior (after the nvtracker patch) was traced to camInfo generation using the wrong `E` direction from `config/camera_calibration.json`; see §6.1.

## 2) When this issue appeared

This surfaced during the SV3DT bring-up/tuning work while switching the runtime from the default pipeline config to the SV3DT-enabled one (`config/infer_v3dt_sv3dt.yaml`).

The earliest confirmed OOM-kill evidence in kernel logs is from **2025-12-29** (see §3).

## 3) Evidence collected

### 3.1 Kernel log confirms OOM killer (host RAM)

From `journalctl -k`:

```
oom-kill: ... task=python3 ...
Out of memory: Killed process <pid> (python3) ... anon-rss:15591944kB ...
```

System snapshot at time of investigation:

- RAM: ~31 GiB
- Swap: 0
- GPU: RTX 3060 12GB (GPU memory usage was not the limiting factor)

### 3.2 The failing config differs only by tracker config

Diff check between `config/infer.yaml` and `config/infer_v3dt_sv3dt.yaml` shows the only functional change is:

- `tracker.config-file: config/nvtracker.yaml` → `config/v3dt/nvtracker_sv3dt.yml`

So the leak is strongly correlated with SV3DT tracker mode and/or code paths that are only active when `config/v3dt/*` is selected.

## 4) What has been tried (and did NOT fix it)

### 4.1 “Maybe it’s the ResNet tracker-ReID engine build” (not the current root cause)

We verified the tracker-internal ReID model and engine:

- TAO model downloaded: `models/tracker_reid/resnet50_market1501.etlt`
- Tracker engine exists and loads: `models/tracker_reid/resnet50_market1501.etlt_b32_gpu0_fp16.engine`
- `config/v3dt/nvtracker_sv3dt.yml` / `config/v3dt/nvtracker_sv3dt_sample.yml` were updated to reference the engine **in `models/tracker_reid/`**, because nvtracker caches the serialized TRT plan next to the `.etlt`.

In the reported runtime output, nvtracker logs:

```
[NvMultiObjectTracker] Loading TRT Engine for model: ...resnet50_market1501...engine
[NvMultiObjectTracker] Loading Complete!
```

So the OOM-kill is happening after engines are already loaded and the pipeline is running.

### 4.2 WebSocket backlog hypothesis (ruled out by “no clients” reproduction)

Initial hypothesis: telemetry publish could be scheduling unbounded async sends.

Mitigation implemented anyway:

- `websocket_server.py`: added a high-frequency JSON coalescer in `broadcast_sync` so `tracking` (per `source_id`) and `bev-frame` (per `cameraId`) messages don’t create unbounded task backlogs.

Result:
- The OOM behavior still occurs even when **no WS clients connect**, so WS backlog is not sufficient to explain the leak.

### 4.3 SV3DT tracker-stage 3D-meta cache leak hypothesis (did not resolve)

There is an SV3DT-only cache that stores 3D meta right after nvtracker:

- `noesis/pipelines/hooks.py`: `_V3DTObj3DMetaCache` populated by `_V3DTObj3DMetaCacheOperator`.

We identified a real bug: the cache previously pruned only on `get()`. If inline extraction succeeds later, `get()` might not be called, so pruning could be skipped.

Fix implemented:
- Cache now prunes on every `update()` using a monotonic prune queue.
- Added toggles:
  - `NOESIS_V3DT_OBJ3D_CACHE=0` disables the cache.
  - `NOESIS_V3DT_CACHE_DEBUG=1` logs cache size every ~5s.

Result:
- User reports the same linear memory growth **even with `NOESIS_V3DT_OBJ3D_CACHE=0`**, so this cache is not the sole leak.

### 4.4 SV3DT meta extraction disabled (ruled out)

Mitigation implemented:
- `noesis/pipelines/hooks.py`: add `NOESIS_V3DT_META_EXTRACT=0` to skip `noesis_v3dt_meta_ext` extraction.

Result:
- The RSS leak persists even with:
  - `NOESIS_V3DT_OBJ3D_CACHE=0` (cache disabled), and
  - `NOESIS_V3DT_META_EXTRACT=0` (meta extraction disabled).

### 4.5 DS8 runtime warning about swap-less TRT builds (not the current issue)

`noesis/ds8_runtime.py` has a preflight warning function:

- `_warn_if_v3dt_reid_engine_build_may_oom(...)`

This is useful when the tracker ReID engine must be built with no swap, but the current reports show the engine already exists and loads, and the OOM occurs later during runtime.

## 5) Root cause (confirmed)

The leak is in the open-source DeepStream `gst-nvtracker` plugin glue code:

- `/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvtracker/nvtracker_proc.cpp`
  - `NvTrackerProc::updateObjMaskMeta()` allocates `pObjectMeta->mask_params.data` via `g_malloc` and overwrites the pointer for preserved objects across frames without freeing the previous allocation.
  - `NvTrackerProc::updateObjectProjectionMeta()` contains an unconditional debug overlay `if(1)` that acquires `NvDsDisplayMeta` and draws 3D bbox lines every frame when `outputFootLocation` is enabled.

This explains why RSS growth was correlated with SV3DT/3D projection being enabled, and why DS8-side toggles (WS clients, meta extraction, cache) did not affect the leak rate.

## 6) Fix (implemented in this repo)

- Patch: `patches/deepstream/gst-nvtracker_mask_params_leak_fix.patch`
  - Frees the previous `mask_params.data` allocation before overwriting.
  - Disables the always-on 3D bbox debug overlay.
- Build helper: `scripts/build_patched_nvtracker.sh`
- Runtime integration: `noesis/ds8_runtime.py`
  - Auto-builds and auto-loads the patched nvtracker when the selected pipeline uses a V3DT tracker config under `config/v3dt/`.
  - Fails fast for V3DT runs if the patched plugin cannot be built (prevents silently running the leaky upstream plugin).

## 6.1) Follow-on root cause (confirmed): runaway RSS with Noesis camInfo `E` direction

After the nvtracker leak was fixed, SV3DT could still show runaway RSS / eventual OOM-kill when using the repo’s auto-generated per-camera camInfo files (`config/v3dt/camInfo_*.yml`) built from `config/camera_calibration.json`.

**Repro (before this fix):**

- `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_single_caminfo.yaml --duration-s 120`

**Fix (at the time of this incident):**

- The calibration source feeding `camera_calibration.json` was providing the opposite convention (Camera→World), so camInfo generation had to invert it to obtain World→Camera.

**Current canonical convention (Menon, 2026-01):**

- Menon sends extrinsics as `E = inv(Twc)` where `Twc` is Camera→World, so `E` is **World→Camera**.
- SV3DT `projectionMatrix_3x4_w2p` should therefore be built from **World→Camera** without inversion: `P = K @ E[:3,:]`.
- Env override remains available:
  - `NOESIS_V3DT_CAMINFO_INVERT_E=1` only if your stored `E` is Camera→World.

**Validation (after this fix):**

- `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_single_caminfo.yaml --duration-s 60` → PASS
- `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_3cam_sv3dt_real_caminfo.yaml --duration-s 120` → PASS
- `python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_sample_3cam_sv3dt_real_caminfo.yaml` → PASS (`bbox3d` observed)

## 7) Commands used / repro cheatsheet

- Baseline (works):  
  `python3 noesis/ds8_runtime.py`

- V3DT pipeline (SV3DT enabled; patched nvtracker auto-loaded):  
  `python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt.yaml`

- SV3DT RTSP pipeline (patched nvtracker auto-loaded):  
  `python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_sv3dt.yaml`

- Automated regression gate (spawns runtime + monitors RSS):  
  `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt.yaml --duration-s 300`  
  `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sv3dt.yaml --duration-s 300`
  `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_single_caminfo.yaml --duration-s 60`  
  `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_3cam_sv3dt_real_caminfo.yaml --duration-s 120`

- SV3DT meta smoke test (`bbox3d` appears on tracking WS telemetry):  
  `python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_sample.yaml`

## 8) Files touched during this investigation

- `config/v3dt/nvtracker_sv3dt.yml` (SV3DT tuning + tracker ReID + pose)
- `config/v3dt/nvtracker_sv3dt_sample.yml`
- `pipelines/config_infer_secondary_reid_osnet.ini` (OSNet reinfer cadence; unrelated to SV3DT-only diff but part of “heavy mode”)
- `websocket_server.py` (JSON coalescer; ruled out as sole cause)
- `noesis/pipelines/hooks.py` (SV3DT meta cache pruning + toggles; ruled out as sole cause)
- `noesis/ds8_runtime.py` (preflight warning about swap-less engine builds)
- `noesis/ds8_runtime.py` (V3DT camInfo `E` inversion + auto-regeneration)
- `patches/deepstream/gst-nvtracker_mask_params_leak_fix.patch` (upstream nvtracker host-RAM leak fix)
- `scripts/build_patched_nvtracker.sh` (build + override plugin paths)
- `scripts/v3dt_oom_regression_test.py` (headless RSS regression gate)
- `scripts/sv3dt_meta_smoke_test.py` (bbox3d telemetry smoke test)
- `scripts/generate_v3dt_caminfo.py` (camInfo generator; now matches runtime inversion behavior)
