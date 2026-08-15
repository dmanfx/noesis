# MV3DT Integration - Required Files List

This document lists the exact files needed by an external LLM to create a surgical plan, blueprint, and work order for MV3DT (Multi-View 3D Tracking) integration into the Noesis DeepStream pipeline.

---

## 1. Tracker + Infer Configs (HIGHEST PRIORITY)

### 1.1 Tracker Low-Level Config
**File:** `pipelines/config_tracker_nvdcf_batch_lowlevel.yml`
- **Purpose:** Current NvDCF tracker low-level configuration that MV3DT will replace/extend
- **Key Info:** Contains BaseConfig, TargetManagement, TrajectoryManagement, DataAssociator, StateEstimator, VisualTracker sections
- **Status:** ✅ Found - This is the file referenced by `DEEPSTREAM_TRACKER_CONFIG` in config.py

**Alternative/Related:**
- `config/nvtracker.yaml` - DS8 canonical tracker config (may be used as reference)
- `config/mv3dt_tracker.yaml` - Placeholder MV3DT config template (already exists)

### 1.2 PGIE/SGIE Infer Configs
**File:** `pipelines/config_infer_primary_yolo11.ini`
- **Purpose:** Primary inference engine (YOLOv11) configuration
- **Key Info:** Contains class IDs (0=person, 1=bicycle, etc.), confidence thresholds, batch size, model paths
- **Status:** ✅ Found

**Related:**
- `config/infer.yaml` - DS8 inference builder config (references the INI above)
- `pipelines/config_infer_primary_yolo11_seg.ini` - Segmentation variant (if used)

---

## 2. Calibration + BEV / Geometry (HIGHEST PRIORITY)

### 2.1 Camera + Calibration Config
**File:** `config/cameras.yaml`
- **Purpose:** Camera intrinsics, models, and per-camera settings
- **Key Info:** Maps source_id → camera name, intrinsics model, height_m
- **Status:** ✅ Found

**Related Calibration Files:**
- `config/camera_calibration.json` - Extrinsics (E matrices) per camera
- `intrinsics.json` - Intrinsics models (K matrices) for different camera types
- `config/cameras.template.yaml` - Template showing expected structure

### 2.2 BEV / Geometry Helpers
**File:** `noesis/telemetry/bev.py`
- **Purpose:** Bird's-eye view rendering and world coordinate projection
- **Key Info:**
  - `CalibrationSnapshot` dataclass (defines world frame structure)
  - `BevRenderer` class (handles homography and world projection)
  - `Footpoint` dataclass (2D image points → 3D world)
- **Status:** ✅ Found

**Related Geometry Files:**
- `geometry/homography.py` - Core homography calculations (`img_to_plane_homography`, `parse_extrinsics`, etc.)
- `geometry/transform.py` - World coordinate transformations (`pixel_to_world`, `build_align_matrix`)

---

## 3. Metadata / Telemetry Plumbing (HIGHEST PRIORITY)

### 3.1 Meta/Telemetry Helpers
**File:** `pipelines/meta_ops.py`
- **Purpose:** DeepStream metadata traversal and manipulation helpers
- **Key Info:**
  - `iter_frames()`, `iter_objects()` - Traverse NvDsBatchMeta
  - `get_source_id()`, `get_object_id()`, `get_class_id()` - Accessors
  - `create_operator()` - DS8 metadata operator creation
- **Status:** ✅ Found

**File:** `deepstream_video_pipeline.py` (or `noesis/pipelines/hooks.py`)
- **Purpose:** Main metadata parsing and track dictionary building
- **Key Info:**
  - `_parse_obj_meta()` method - Extracts detections and builds track dictionaries
  - `_build_track_dict()` - Creates track objects with camera_id, track_id, bbox, etc.
  - Shows where to attach `position3d`, `global_track_id` fields
- **Status:** ✅ Found (in `deepstream_video_pipeline.py` around line 2349)

**Related:**
- `noesis/pipelines/hooks.py` - Pipeline callback hooks (contains `_AnalyticsTelemetryProcessor` with `_build_track_dict()`)

### 3.2 ReID/StableID Integration
**File:** `reid/stable_id_manager.py`
- **Purpose:** Global track ID management across cameras
- **Key Info:**
  - `StableIDManager.update()` - Assigns stable IDs to tracks
  - Shows how `stable_id` is attached to track dictionaries
  - Cross-camera identity matching logic
- **Status:** ✅ Found

**Usage Example:** See `deepstream_video_pipeline.py` lines 2533-2554 for how stable_id is integrated

### 3.3 WebSocket / API Schema for Tracks/BEV
**File:** `oai2-fe/src/hooks/useWebSocketClient.ts`
- **Purpose:** Frontend TypeScript types for track objects and BEV messages
- **Key Info:**
  - `Track` type definition (lines 5-13): track_id, stable_id, camera_id, zone, center, velocity
  - `CamerasStats` type: Shows structure of tracking data sent to frontend
  - BEV message types: `bev-frame`, `bev-status` message formats
- **Status:** ✅ Found

**Backend WebSocket:**
- `websocket_server.py` - WebSocket server implementation (shows message broadcasting)

---

## 4. Deployment / Site Configuration

### 4.1 MQTT Config (if used)
**File:** `config.py` (lines 397-417)
- **Purpose:** MQTT broker configuration for occupancy publishing
- **Key Info:**
  - `IntegrationsSettings` class: MQTT_HOST, MQTT_PORT, BASE_TOPIC, etc.
  - Shows existing MQTT integration pattern
- **Status:** ✅ Found

**Related:**
- `occupancy_publisher.py` - MQTT publisher implementation (shows how MQTT is used)

### 4.2 Main App Config File
**File:** `config.py`
- **Purpose:** Central configuration with feature flags and processing settings
- **Key Info:**
  - `ProcessingSettings.DEEPSTREAM_TRACKER_CONFIG` - Where tracker config path is defined
  - `ProcessingSettings` class - Processing toggles and env overrides
  - Shows where to add `ENABLE_SV3D`, `ENABLE_MV3DT` flags
- **Status:** ✅ Found

---

## 5. Optional but Helpful

### 5.1 Codebase Documentation
**File:** `CODEBASE_DESCRIPTION.md`
- **Purpose:** High-level architecture and code organization overview
- **Status:** ✅ Found

### 5.2 Pipeline Structure
**File:** `noesis/pipelines/ds8_pipeline.py`
- **Purpose:** DS8 pipeline builder (shows how tracker is configured)
- **Key Info:** Lines 280-292 show tracker component configuration
- **Status:** ✅ Found

---

## Summary: Minimum Required Files

For a **surgical** plan (not generic), the external LLM needs at minimum:

### Critical (Must Have):
1. ✅ `pipelines/config_tracker_nvdcf_batch_lowlevel.yml` - Current tracker config
2. ✅ `config/cameras.yaml` - Camera intrinsics/extrinsics mapping
3. ✅ `noesis/telemetry/bev.py` - BEV world frame definition
4. ✅ `geometry/homography.py` - World coordinate projection logic
5. ✅ `pipelines/meta_ops.py` - Metadata traversal helpers
6. ✅ `deepstream_video_pipeline.py` - Track dictionary building (`_parse_obj_meta`)
7. ✅ `config.py` - Main config with feature flags location

### Highly Recommended:
8. ✅ `pipelines/config_infer_primary_yolo11.ini` - Class IDs and thresholds
9. ✅ `config/camera_calibration.json` - Extrinsics data
10. ✅ `intrinsics.json` - Intrinsics models
11. ✅ `oai2-fe/src/hooks/useWebSocketClient.ts` - Frontend schema
12. ✅ `reid/stable_id_manager.py` - Global ID integration pattern

### Nice to Have:
13. ✅ `CODEBASE_DESCRIPTION.md` - Architecture overview
14. ✅ `noesis/pipelines/ds8_pipeline.py` - Pipeline builder pattern
15. ✅ `config/mv3dt_tracker.yaml` - Existing MV3DT template

---

## File Paths Summary (Absolute)

All files are relative to workspace root: `/home/mayor/Noesis_Devel/`

1. `pipelines/config_tracker_nvdcf_batch_lowlevel.yml`
2. `pipelines/config_infer_primary_yolo11.ini`
3. `config/cameras.yaml`
4. `config/camera_calibration.json`
5. `intrinsics.json`
6. `noesis/telemetry/bev.py`
7. `geometry/homography.py`
8. `geometry/transform.py`
9. `pipelines/meta_ops.py`
10. `deepstream_video_pipeline.py`
11. `noesis/pipelines/hooks.py`
12. `reid/stable_id_manager.py`
13. `oai2-fe/src/hooks/useWebSocketClient.ts`
14. `config.py`
15. `CODEBASE_DESCRIPTION.md`
16. `noesis/pipelines/ds8_pipeline.py`
17. `config/mv3dt_tracker.yaml`

---

## Notes for External LLM

1. **Tracker Config:** The current tracker uses NvDCF (`libnvds_nvmultiobjecttracker.so`). MV3DT will likely replace or extend this.

2. **World Frame:** BEV uses a world coordinate system defined by:
   - Camera extrinsics (E matrices) from `camera_calibration.json`
   - Floor plane (`floor_y` from cameras.yaml)
   - Unit scale (from calibration bundle)
   - MV3DT's world frame must match this exactly.

3. **Track Schema:** Current tracks have:
   - `track_id` (per-camera DeepStream ID)
   - `stable_id` (cross-camera global ID from StableIDManager)
   - `camera_id` (string like "living-room")
   - `bbox`, `center`, `velocity`, `zone`
   - MV3DT should add: `position3d`, `global_track_id` (MV3DT's own global ID)

4. **Class IDs:** YOLOv11 uses COCO classes. Person = class_id 0. See `config_infer_primary_yolo11.ini` for thresholds.

5. **Config Pattern:** Feature flags go in `config.py` → `ProcessingSettings` class. Tracker config path is `DEEPSTREAM_TRACKER_CONFIG`.

6. **Metadata Flow:** `meta_ops.py` provides the traversal API. `_parse_obj_meta()` in `deepstream_video_pipeline.py` shows where to attach 3D data.

---

**Status:** ✅ All critical files identified and verified to exist in codebase.
