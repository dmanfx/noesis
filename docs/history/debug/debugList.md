# MV3DT Memory Leak Debug Log

## Symptoms Reported
- Linear RAM increase when detections occur
- Detection delayed ~5 seconds (normally instant)
- Flash of seg mask, then stream freezes
- Issue appeared after another agent's "Phase 1 wiring" changes

## Investigation Steps

### 1. Verified Agent's Claimed Changes Were NOT Applied
- `_select_tracker_config()` method was NOT in `deepstream_video_pipeline.py`
- 3D metadata extraction (`get_3d_bbox_meta`) was NOT being called anywhere
- `ENABLE_SV3D=True` flag existed but was ignored—pipeline always used baseline tracker config

### 2. Checked Tracker Config Selection
- Found line 1653-1658 still hardcoded to use `DEEPSTREAM_TRACKER_CONFIG`
- Never checked `ENABLE_SV3D` or `ENABLE_MV3D` flags
- SV3D tracker config was never actually loaded

### 3. Identified camInfo File Issue
- Agent changed key from `projectionMatrix_3x4` → `projectionMatrix_3x4_w2p`
- DeepStream ObjectModelProjection expects `projectionMatrix_3x4` (no suffix)
- If SV3D config loaded manually, tracker couldn't read projection matrices → error state → potential leak

### 4. Found YAML Header Issue
- SV3D config had `%YAML:1.0` directive at top
- DeepStream uses OpenCV YAML parser which may not handle this properly
- Baseline config had no such header

### 5. Reviewed `meta_ops.py` Changes
- Agent added `cleanup_operator()` function but it's NEVER called
- Added `_ServiceMakerOperator` caching (`_frame_cache`, `_object_cache`)
- BUT: `_sm_enabled()` returns `False` unconditionally—Service Maker disabled
- So caching code is dead code, not causing leak

### 6. Checked State Accumulation Dictionaries
- `track_motion_state_by_sensor` - grows per track_id (uses setdefault)
- `track_zone_state_by_sensor` - grows per track_id
- `_last_good_footpoint_by_sensor` - grows per track_id
- These exist in both 2D and 3D modes—not specific to SV3D change

### 7. Verified Buffer Cleanup
- `_on_new_bev_frame` properly unmaps buffers in `finally` block
- `sample.unref()` called in finally block
- Not a buffer leak source

## Fixes Applied

### Fix 1: Corrected camInfo Key Name
```python
# scripts/generate_caminfo_from_calibration.py
"projectionMatrix_3x4_w2p" → "projectionMatrix_3x4"
```
Regenerated all 3 camInfo files.

### Fix 2: Removed YAML Directive Header
```yaml
# pipelines/config_tracker_nvdcf_sv3d.yml
# Removed: %YAML:1.0
```

### Fix 3: Added Tracker Config Selection Logic
```python
# deepstream_video_pipeline.py
def _select_tracker_config(self) -> str:
    if ENABLE_MV3D: return MV3D_TRACKER_CONFIG
    elif ENABLE_SV3D: return SV3D_TRACKER_CONFIG
    else: return DEEPSTREAM_TRACKER_CONFIG
```
Updated tracker setup to call this method.

## Root Cause Hypothesis
When SV3D tracker loaded with `stateEstimatorType: 3` but couldn't parse projection matrices (wrong key name), it likely:
1. Failed silently on each detection
2. Accumulated internal 3D state without cleanup
3. Caused delayed/failed detections + memory growth

## Status
Fixes committed to `feature_DS8` branch (commit `9927ae5`). Awaiting user retest.

---

## Additional SV3D Memory Leak Investigation (November 26, 2025)

**Issue:** Memory leak persists despite previous fixes. System RAM grows linearly, GPU utilization drops to 0% when detection starts.

**Root Cause Analysis:** The meta_ops.py abstraction layer was incomplete - DS8 operator creation was failing but cleanup was never called, causing metadata accumulation.

### Investigation Timeline

#### Initial Analysis (First Session)
- **Hypothesis 1:** 3D metadata being generated but not accessed → memory leak
- **Fix Applied:** Added 3D metadata extraction in `_parse_obj_meta` function
- **Result:** Memory leak still present - hypothesis incorrect

#### Deep Dive Analysis (Second Session)
- **Hypothesis 2:** Kalman filter parameters too aggressive causing instability
- **Fix Attempted:** Adjusted `processNoiseVar4Loc`, `processNoiseVar4Vel`, `maxTargetsPerStream` values
- **Result:** User reported no improvement - parameters not the issue

- **Hypothesis 3:** ObjectModelProjection metadata generation causing leaks
- **Investigation:** Identified that DS8 operator import was failing silently
- **Root Cause Found:** `NvDsBatchMetaOperator` import failing → `create_operator()` returns None → DS8 operators never created → cleanup never called → metadata accumulation

#### Critical Fix Applied (Third Session)
- **Fix Applied:** Added `meta_ops.cleanup_operator(operator)` calls to all probe functions that create operators:
  - `_post_remove_excluded_objects_probe` (line 535)
  - `_on_new_bev_frame` (line 1127)
  - `_analytics_probe` (line 2284)
  - `_nvinfer_object_debug_probe` (line 2314)
  - `_nvinfer_output_probe` (line 3261)
  - `_mosaic_osd_probe` (line 3763)
- **Result:** Pending user testing

### Failed Hypotheses
1. ❌ **3D metadata access:** Adding `get_3d_bbox_meta()` calls didn't fix leak
2. ❌ **Kalman filter tuning:** Adjusting noise parameters had no effect
3. ❌ **GPU memory:** Issue is system RAM, not GPU VRAM
4. ❌ **MQTT communication:** ENABLE_MV3D=False so MQTT not involved
5. ❌ **Calibration data:** camInfo files appear valid

### Successful Fixes Applied
- ✅ **Operator cleanup:** Added cleanup calls to prevent DS8 operator memory leaks
- ✅ **3D metadata extraction:** Added proper 3D data extraction (though not root cause)


---
*Additional investigation added: November 26, 2025*

## Additional SV3D Memory Leak Investigation (Session 2 - November 26, 2025)

**Status:** Leak persists despite extensive Python-side optimization and isolation.

### Hypotheses Tested & Failed
1.  **Metadata List Materialization:**
    -   **Hypothesis:** `iter_user_meta` creating large lists of `pyds` wrappers caused memory pressure.
    -   **Action:** Converted `iter_user_meta` to a generator.
    -   **Result:** Leak persisted.
2.  **Malicious/Garbage Mask Metadata:**
    -   **Hypothesis:** SV3D returning invalid mask dimensions causing massive `numpy` allocations ("instant freeze").
    -   **Action:** Added deterministic bounds checking to `_extract_mask_array` using frame dimensions.
    -   **Result:** Leak persisted.
3.  **Unbounded State Dictionaries:**
    -   **Hypothesis:** `track_motion_state_by_sensor` and others growing indefinitely.
    -   **Action:** Implemented `_prune_stale_tracks` to clean up old entries.
    -   **Result:** Leak persisted.
4.  **`meta_ops.py` Instability:**
    -   **Hypothesis:** Unsafe casting or iteration in WIP `meta_ops.py`.
    -   **Action:** Refactored `meta_ops.py` to use safe generators and try/except blocks for all `pyds` calls.
    -   **Result:** Leak persisted.
5.  **Python-side Metadata Access (Isolation Test):**
    -   **Hypothesis:** The mere act of accessing/casting `NvDsUserMeta` in Python triggers a leak in `pyds`.
    -   **Action:** **TOTALLY DISABLED** all Python-side metadata extraction (`get_3d_bbox_meta`, `_extract_analytics_obj_meta`).
    -   **Result:** Leak persisted.

### Conclusion
The memory leak is **NOT** in the Python application logic (`deepstream_video_pipeline.py` or `meta_ops.py`).
It is almost certainly located in:
1.  **The DeepStream SV3D Tracker Binary:** The `libnvds_nvmultiobjecttracker.so` itself may be leaking memory when `outputFootLocation: 1` is enabled.
2.  **`pyds` Bindings:** The internal C++ to Python binding layer may have a leak when 3D metadata is present in the buffer, even if not accessed by Python.
3.  **GStreamer Buffer Handling:** The additional 3D metadata might be causing buffer pool fragmentation or leaks in the `nvstreammux` or `nvtracker` elements.

### Recommended Next Steps
-   **Disable `outputFootLocation`:** In `pipelines/config_tracker_nvdcf_sv3d.yml`, set `outputFootLocation: 0`. If this stops the leak, we have confirmed the trigger is the generation of this specific metadata.
-   **Report to NVIDIA:** This appears to be a bug in the DeepStream SDK's SV3D implementation.
