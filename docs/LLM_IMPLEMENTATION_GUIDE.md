# LLM Implementation Guide: TensorRT FP16 Migration

## Implementation Sequence & Precise Tasks

### TASK 1: Configuration Updates (Priority: P0)
**File:** `config.py`
**Objective:** Add TensorRT settings to ModelsSettings class

**Exact Changes Required:**
```python
# ADD these lines to ModelsSettings class after line 103:
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

**Validation:** Verify config loads without errors and new settings accessible via `config.models.ENABLE_TENSORRT`

---

### TASK 2: TensorRT Engine Builder (Priority: P1)
**File:** `tensorrt_builder.py` (already exists, needs completion)
**Objective:** Complete implementation with specific model support

**Critical Functions to Implement:**
1. `_export_to_onnx()` - Convert PyTorch models to ONNX
2. `_build_engine_from_onnx()` - Compile ONNX to TensorRT engine  
3. `_validate_engine()` - Test engine functionality
4. `build_all_engines()` - Process all model types

**Specific Requirements:**
- Input shapes: YOLO (1,3,640,640), ReID (1,3,256,128)
- FP16 precision enforcement
- Engine validation with dummy inputs
- Error handling for build failures

---

### TASK 3: GPU-Only Inference Manager (Priority: P1)  
**File:** `tensorrt_inference.py` (already exists, needs completion)
**Objective:** Complete GPU-only inference with strict enforcement

**Critical Classes to Complete:**
1. `TensorRTInferenceEngine.infer()` - Core inference execution
2. `GPUOnlyDetectionManager._postprocess_detections()` - GPU-only post-processing
3. `GPUOnlyDetectionManager._extract_reid_features()` - GPU-only feature extraction

**Strict Requirements:**
- NO `.cpu()` calls anywhere
- Device validation on every tensor operation
- FP16 precision enforcement
- Immediate failure on GPU violations

---

### TASK 4: Legacy Code CPU Fallback Removal (Priority: P0)
**File:** `detection.py`
**Objective:** Remove ALL CPU fallback operations

**Specific Lines to Fix:**
```python
# Line 375 - REPLACE:
masks_data = seg_results[0].masks.data.cpu().numpy()
# WITH:
if self.config.models.FORCE_GPU_ONLY:
    if seg_results[0].masks.data.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in segmentation")
    masks_data = seg_results[0].masks.data
else:
    masks_data = seg_results[0].masks.data.cpu().numpy()

# Line 752 - REPLACE:
raw_keypoints_xy = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints.xy is not None else None
# WITH:
if self.config.models.FORCE_GPU_ONLY:
    raw_keypoints_xy = pose_results.keypoints.xy if pose_results.keypoints.xy is not None else None
    if raw_keypoints_xy is not None and raw_keypoints_xy.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in pose keypoints")
else:
    raw_keypoints_xy = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints.xy is not None else None

# Line 753 - REPLACE:
raw_keypoints_conf = pose_results.keypoints.conf.cpu().numpy() if pose_results.keypoints.conf is not None else None
# WITH:
if self.config.models.FORCE_GPU_ONLY:
    raw_keypoints_conf = pose_results.keypoints.conf if pose_results.keypoints.conf is not None else None
    if raw_keypoints_conf is not None and raw_keypoints_conf.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in pose confidence")
else:
    raw_keypoints_conf = pose_results.keypoints.conf.cpu().numpy() if pose_results.keypoints.conf is not None else None

# Line 766 - REPLACE:
pose_boxes_xyxy = pose_results.boxes.xyxy.cpu().numpy() if pose_results.boxes is not None else None
# WITH:
if self.config.models.FORCE_GPU_ONLY:
    pose_boxes_xyxy = pose_results.boxes.xyxy if pose_results.boxes is not None else None
    if pose_boxes_xyxy is not None and pose_boxes_xyxy.device.type != 'cuda':
        raise RuntimeError("GPU enforcement violation in pose boxes")
else:
    pose_boxes_xyxy = pose_results.boxes.xyxy.cpu().numpy() if pose_results.boxes is not None else None
```

**Additional Device Enforcement:**
```python
# ADD to DetectionManager.__init__ after line 180:
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

---

### TASK 5: Migration Script (Priority: P2)
**File:** `migrate_to_tensorrt.py` (already exists, needs implementation)
**Objective:** Automated migration with validation

**Required Functions:**
1. `validate_environment()` - Check CUDA/TensorRT availability
2. `build_engines()` - Convert all models to TensorRT
3. `performance_comparison()` - Benchmark PyTorch vs TensorRT
4. `validate_accuracy()` - Ensure output consistency

---

## LLM IMPLEMENTATION PROMPT

**CONTEXT:** You are implementing a TensorRT FP16 migration for a YOLO-based inference pipeline. The codebase has 5 critical CPU fallback violations that must be eliminated.

**OBJECTIVE:** Convert all inference operations to TensorRT FP16 with strict GPU-only enforcement.

**CRITICAL CONSTRAINTS:**
1. **NO CPU FALLBACKS:** Never add `.cpu()` calls - fail fast instead
2. **STRICT GPU ENFORCEMENT:** Every tensor must be validated for GPU placement
3. **FP16 PRECISION:** All operations must use torch.float16
4. **NO PLACEHOLDER CODE:** Implement complete, functional code only
5. **PRESERVE FUNCTIONALITY:** Maintain existing API compatibility

**IMPLEMENTATION ORDER:**
1. **config.py** - Add TensorRT settings (TASK 1)
2. **detection.py** - Fix CPU fallback violations (TASK 4) 
3. **tensorrt_builder.py** - Complete engine builder (TASK 2)
4. **tensorrt_inference.py** - Complete GPU-only manager (TASK 3)
5. **migrate_to_tensorrt.py** - Complete migration script (TASK 5)

**VALIDATION REQUIREMENTS:**
- Each file must load without import errors
- All GPU enforcement checks must work
- TensorRT engines must build successfully
- Performance must improve 2x minimum

**GUARD RAILS:**
- ❌ Do NOT add any `.cpu()` calls
- ❌ Do NOT create placeholder/stub functions  
- ❌ Do NOT ignore device validation
- ❌ Do NOT use FP32 precision
- ✅ DO validate tensor devices before operations
- ✅ DO implement complete error handling
- ✅ DO maintain API compatibility
- ✅ DO add comprehensive logging

**ERROR HANDLING PATTERN:**
```python
# Use this pattern for all GPU operations:
if self.config.models.FORCE_GPU_ONLY:
    if tensor.device.type != 'cuda':
        raise RuntimeError(f"GPU enforcement violation: tensor on {tensor.device}")
    if tensor.dtype != torch.float16:
        raise RuntimeError(f"FP16 enforcement violation: tensor is {tensor.dtype}")
```

**SUCCESS CRITERIA:**
- Zero CPU fallbacks in inference pipeline
- All models running on TensorRT FP16
- 2x+ performance improvement
- All tests passing

Begin implementation with TASK 1 (config.py). Implement each task completely before moving to the next. Validate each step before proceeding. 