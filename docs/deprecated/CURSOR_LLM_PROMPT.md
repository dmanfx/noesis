# CURSOR LLM IMPLEMENTATION PROMPT

## 🎯 MISSION: TensorRT FP16 Migration

**CONTEXT:** You are a senior AI engineer implementing TensorRT FP16 optimization for a YOLO inference pipeline. The codebase has 5 critical CPU fallback violations that must be eliminated to achieve GPU-only inference.

**CRITICAL FINDINGS:**
- CPU violations at detection.py lines 375, 752, 753, 766, 937
- No existing TensorRT optimization 
- Mixed device handling causing performance issues
- All models need conversion to TensorRT FP16

---

## 🚨 STRICT CONSTRAINTS (NEVER VIOLATE)

### ❌ FORBIDDEN ACTIONS:
1. **NO `.cpu()` CALLS** - Never add CPU fallbacks, fail fast instead
2. **NO PLACEHOLDER CODE** - Every function must be fully implemented
3. **NO FP32 OPERATIONS** - All tensors must be FP16 in GPU-only mode
4. **NO SILENT FAILURES** - Always raise exceptions for violations
5. **NO DEVICE AUTO-DETECTION** - Respect FORCE_GPU_ONLY setting

### ✅ REQUIRED PATTERNS:
```python
# GPU Enforcement Pattern (use everywhere):
if self.config.models.FORCE_GPU_ONLY:
    if tensor.device.type != 'cuda':
        raise RuntimeError(f"GPU violation: tensor on {tensor.device}")
    if tensor.dtype != torch.float16:
        raise RuntimeError(f"FP16 violation: tensor is {tensor.dtype}")

# Error Handling Pattern:
try:
    # GPU operation
    result = gpu_operation(tensor)
except Exception as e:
    if self.config.models.FORCE_GPU_ONLY:
        raise RuntimeError(f"GPU-only operation failed: {e}")
    # Only fallback if GPU-only mode is disabled
```

---

## 📋 IMPLEMENTATION SEQUENCE

### TASK 1: Configuration (START HERE)
**File:** `config.py`
**Action:** Add TensorRT settings to ModelsSettings class after line 103

```python
# TensorRT Optimization Settings
ENABLE_TENSORRT: bool = True
TENSORRT_FP16: bool = True  
FORCE_GPU_ONLY: bool = True
TENSORRT_WORKSPACE_SIZE: int = 4  # GB
TENSORRT_MAX_BATCH_SIZE: int = 8
TENSORRT_ENGINE_CACHE_DIR: str = "models/engines"

# Engine paths
DETECTION_ENGINE_PATH: str = "models/engines/detection_fp16.trt"
POSE_ENGINE_PATH: str = "models/engines/pose_fp16.trt" 
SEGMENTATION_ENGINE_PATH: str = "models/engines/segmentation_fp16.trt"
REID_ENGINE_PATH: str = "models/engines/reid_fp16.trt"

# Performance settings
TENSORRT_WARMUP_ITERATIONS: int = 10
TENSORRT_OPTIMIZATION_LEVEL: int = 3
```

**Validation:** Run `python -c "from config import AppConfig; print(AppConfig().models.ENABLE_TENSORRT)"`

### TASK 2: Remove CPU Fallbacks (HIGH PRIORITY)
**File:** `detection.py`
**Action:** Fix lines 375, 752, 753, 766 with GPU enforcement

**Line 375 EXACT REPLACEMENT:**
```python
# REPLACE THIS:
masks_data = seg_results[0].masks.data.cpu().numpy()

# WITH THIS:
if self.config.models.FORCE_GPU_ONLY:
    masks_data = seg_results[0].masks.data
    if masks_data.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in segmentation masks")
    if masks_data.dtype != torch.float16:
        masks_data = masks_data.half()
else:
    masks_data = seg_results[0].masks.data.cpu().numpy()
```

**Lines 752-753 EXACT REPLACEMENT:**
```python
# REPLACE THESE:
raw_keypoints_xy = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints.xy is not None else None
raw_keypoints_conf = pose_results.keypoints.conf.cpu().numpy() if pose_results.keypoints.conf is not None else None

# WITH THESE:
if self.config.models.FORCE_GPU_ONLY:
    raw_keypoints_xy = pose_results.keypoints.xy if pose_results.keypoints.xy is not None else None
    raw_keypoints_conf = pose_results.keypoints.conf if pose_results.keypoints.conf is not None else None
    if raw_keypoints_xy is not None and raw_keypoints_xy.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in pose keypoints")
    if raw_keypoints_conf is not None and raw_keypoints_conf.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in pose confidence")
else:
    raw_keypoints_xy = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints.xy is not None else None
    raw_keypoints_conf = pose_results.keypoints.conf.cpu().numpy() if pose_results.keypoints.conf is not None else None
```

**Line 766 EXACT REPLACEMENT:**
```python
# REPLACE THIS:
pose_boxes_xyxy = pose_results.boxes.xyxy.cpu().numpy() if pose_results.boxes is not None else None

# WITH THIS:
if self.config.models.FORCE_GPU_ONLY:
    pose_boxes_xyxy = pose_results.boxes.xyxy if pose_results.boxes is not None else None
    if pose_boxes_xyxy is not None and pose_boxes_xyxy.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in pose boxes")
else:
    pose_boxes_xyxy = pose_results.boxes.xyxy.cpu().numpy() if pose_results.boxes is not None else None
```

**ADD GPU Enforcement to DetectionManager.__init__:**
```python
# ADD after line 180 in DetectionManager.__init__:
if config and hasattr(config.models, 'FORCE_GPU_ONLY') and config.models.FORCE_GPU_ONLY:
    if not torch.cuda.is_available():
        raise RuntimeError("FORCE_GPU_ONLY enabled but CUDA not available")
    self.device = torch.device(config.models.DEVICE)
    self.force_gpu_only = True
    self.logger.info(f"GPU-only mode enforced on device: {self.device}")
else:
    self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    self.force_gpu_only = False
```

### TASK 3: Complete TensorRT Engine Builder
**File:** `tensorrt_builder.py` (exists but incomplete)
**Action:** Implement missing functions

**Complete these functions:**
1. `_export_to_onnx()` - Export PyTorch model to ONNX
2. `_validate_engine()` - Test engine with dummy input
3. `build_yolo_engine()` - Build YOLO-specific engines
4. `build_reid_engine()` - Build ReID-specific engines

### TASK 4: Complete GPU-Only Inference Manager  
**File:** `tensorrt_inference.py` (exists but incomplete)
**Action:** Complete implementation

**Critical Functions:**
1. `GPUOnlyDetectionManager._postprocess_detections()` - GPU-only NMS and filtering
2. `GPUOnlyDetectionManager._extract_reid_features()` - GPU-only feature extraction
3. `GPUOnlyDetectionManager.preprocess_frame()` - GPU-only preprocessing

### TASK 5: Migration Script
**File:** `migrate_to_tensorrt.py` (exists but incomplete)
**Action:** Complete automated migration

---

## 🔍 VALIDATION CHECKLIST

After each task, verify:
- [ ] File imports without errors
- [ ] No `.cpu()` calls added
- [ ] All GPU enforcement checks work
- [ ] FP16 precision maintained
- [ ] Logging shows GPU-only mode active

**Test Command After Each Task:**
```bash
python -c "
from config import AppConfig
from detection import DetectionManager
config = AppConfig()
config.models.FORCE_GPU_ONLY = True
dm = DetectionManager('models/yolo11m.pt', config=config)
print('✅ GPU enforcement working')
"
```

---

## 🎯 SUCCESS CRITERIA

**MUST ACHIEVE:**
- Zero CPU fallbacks in inference pipeline
- All models running TensorRT FP16
- 2x+ performance improvement
- All operations GPU-only when FORCE_GPU_ONLY=True

**FINAL VALIDATION:**
```bash
# This should work without errors:
python migrate_to_tensorrt.py --validate-only
```

---

## 🚀 EXECUTION INSTRUCTIONS

1. **START WITH TASK 1** - Config changes first
2. **VALIDATE EACH STEP** - Test after every change
3. **IMPLEMENT COMPLETELY** - No partial implementations
4. **USE EXACT CODE PROVIDED** - Follow patterns precisely
5. **FAIL FAST** - Raise exceptions for violations
6. **LOG EVERYTHING** - Comprehensive logging for debugging

**BEGIN IMPLEMENTATION NOW WITH TASK 1 (config.py)**

Remember: GPU-only means GPU-only. No exceptions, no fallbacks, no compromises. 