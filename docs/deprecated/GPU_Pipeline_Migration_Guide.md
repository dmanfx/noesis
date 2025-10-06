# GPU Pipeline Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the legacy multiprocessing pipeline to the unified GPU pipeline, targeting an 80-85% reduction in CPU usage.

## Migration Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA Compute Capability 6.0+ (GTX 1060 or newer)
- NVDEC hardware video decoding support
- Minimum 8GB GPU memory (12GB+ recommended for 3+ cameras)
- CUDA Toolkit 11.0+ installed

### Software Requirements
- PyTorch 2.0+ with CUDA support
- TensorRT 8.0+ 
- PyCUDA 2022.1+
- FFmpeg with NVDEC support

## Pre-Migration Checklist

- [ ] **Backup current configuration**
  ```bash
  cp config.json config.json.backup
  ```

- [ ] **Verify GPU capabilities**
  ```bash
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  ffmpeg -hwaccels | grep cuda
  ```

- [ ] **Check TensorRT engines exist**
  ```bash
  ls -la models/engines/*.engine
  ```

- [ ] **Test GPU memory availability**
  ```bash
  python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
  ```

## Migration Steps

### Step 1: Update Configuration

#### 1.1 Disable Legacy Pipeline
```python
# In config.py or config.json
"processing": {
    "ENABLE_DECOUPLED_PIPELINE": false,  # MUST be false
    "ENABLE_MULTIPROCESSING": false,     # MUST be false
    "USE_UNIFIED_GPU_PIPELINE": true,    # Enable unified pipeline
    ...
}
```

#### 1.2 Enable GPU-Only Features
```python
"processing": {
    "ENABLE_NVDEC": true,                # Hardware video decoding
    "NVDEC_FALLBACK_TO_CPU": false,      # No CPU fallback
    "ENABLE_GPU_PREPROCESSING": true,     # GPU preprocessing
    "FORCE_GPU_ONLY": true,              # Strict GPU enforcement
    ...
}
```

#### 1.3 Optimize Performance Settings
```python
"processing": {
    "MAX_QUEUE_SIZE": 20,                # Reduced for lower latency
    "NVDEC_BUFFER_SIZE": 5,              # Reduced for memory efficiency
    "GPU_BATCH_SIZE": 2,                 # Smaller batches
    "UNIFIED_PIPELINE_THREADS": 2,       # One per camera
    "PIPELINE_QUEUE_TIMEOUT_MS": 5.0,    # Faster response
    ...
}
```

#### 1.4 TensorRT Configuration
```python
"models": {
    "ENABLE_TENSORRT": true,             # MUST be true
    "TENSORRT_FP16": true,               # FP16 optimization
    "TENSORRT_WORKSPACE_SIZE": 2,        # 2GB workspace
    "FORCE_GPU_ONLY": true,              # No CPU inference
    ...
}
```

### Step 2: Build TensorRT Engines

If engines don't exist, build them:

```python
# build_tensorrt_engines.py
from tensorrt_builder import TensorRTEngineBuilder

builder = TensorRTEngineBuilder(
    workspace_size=2,  # GB
    fp16_mode=True,
    device='cuda:0'
)

# Build detection engine
builder.build_detection_engine(
    onnx_path='models/yolo12m.onnx',
    engine_path='models/engines/detection_fp16.engine'
)

# Build pose engine
builder.build_pose_engine(
    onnx_path='models/yolo11m-pose.onnx',
    engine_path='models/engines/pose_fp16.engine'
)
```

### Step 3: Validate Configuration

Run the validation script:

```python
# validate_config.py
from config import config, validate_unified_pipeline_config

results = validate_unified_pipeline_config(config)

if results['errors']:
    print("❌ Configuration errors found:")
    for error in results['errors']:
        print(f"  - {error}")
    exit(1)
else:
    print("✅ Configuration validated successfully")
```

### Step 4: Test Single Camera

Start with one camera to verify functionality:

```python
# Temporarily reduce cameras in config
"cameras": {
    "RTSP_STREAMS": [
        'rtsp://192.168.3.214:7447/stream1'  # Test with one camera first
    ]
}
```

Run the application:
```bash
python main.py
```

Monitor GPU usage:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Step 5: Scale to Multiple Cameras

Once single camera works:

1. **Add cameras incrementally**:
   ```python
   "RTSP_STREAMS": [
       'rtsp://192.168.3.214:7447/stream1',
       'rtsp://192.168.3.214:7447/stream2'
   ]
   ```

2. **Monitor for NVDEC limits**:
   - RTX 3060: Max 3 concurrent NVDEC streams
   - RTX 3070+: Max 5 concurrent streams

3. **Adjust threads if needed**:
   ```python
   "UNIFIED_PIPELINE_THREADS": 3  # >= number of cameras
   ```

### Step 6: Performance Tuning

#### 6.1 Enable Event-Driven I/O (Optional)
```python
# Integrate event-driven reader
from event_driven_io import integrate_event_driven_reader
integrate_event_driven_reader(pipeline)
```

#### 6.2 GPU Memory Optimization
```python
"processing": {
    "GPU_MEMORY_POOL_SIZE_MB": 500,     # Pre-allocated pool
    "GPU_MEMORY_DEFRAG_INTERVAL": 1000, # Defrag every N frames
    "ENABLE_MEMORY_POOLING": true        # Use memory pooling
}
```

#### 6.3 Thread Optimization
```python
"processing": {
    "USE_THREAD_AFFINITY": true,         # Pin threads to cores
    "THREAD_PRIORITY": "HIGH",           # Thread priority
    "DECODER_THREAD_PRIORITY": "REALTIME" # NVDEC priority
}
```

## Rollback Procedure

If issues occur, rollback is simple:

1. **Restore configuration**:
   ```bash
   cp config.json.backup config.json
   ```

2. **Or manually revert settings**:
   ```python
   "processing": {
       "USE_UNIFIED_GPU_PIPELINE": false,
       "ENABLE_DECOUPLED_PIPELINE": true,  # Re-enable legacy
       "FORCE_GPU_ONLY": false
   }
   ```

3. **Restart application**

## Troubleshooting

### Common Issues

#### 1. "NVDEC initialization failed"
- **Cause**: Hardware limit reached or driver issue
- **Solution**: 
  - Reduce number of cameras
  - Update NVIDIA drivers
  - Check `nvidia-smi dmon` for decoder usage

#### 2. "GPU out of memory"
- **Cause**: Insufficient GPU memory
- **Solution**:
  - Reduce `GPU_BATCH_SIZE`
  - Lower `TENSORRT_WORKSPACE_SIZE`
  - Use smaller models

#### 3. "TensorRT engine not found"
- **Cause**: Engines not built
- **Solution**: Run engine builder script (Step 2)

#### 4. High CPU usage persists
- **Cause**: CPU fallback occurring
- **Check**:
  - Logs for "CPU fallback" warnings
  - `FORCE_GPU_ONLY` is true
  - All NVDEC/TensorRT operations succeed

### Debug Commands

```bash
# Monitor GPU utilization
nvidia-smi dmon -i 0

# Check NVDEC usage
nvidia-smi pmon -i 0

# Profile application
python main.py --profile

# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## Performance Expectations

### Before Migration (Legacy Pipeline)
- CPU Usage: 60-70%
- GPU Decode: Minimal/None
- GPU Compute: 20-30%
- Memory Transfers: High

### After Migration (Unified GPU Pipeline)
- CPU Usage: 5-10% ✅
- GPU Decode: 9-11%
- GPU Compute: 60-70%
- Memory Transfers: Minimal

### Performance Monitoring

```python
# Add to your monitoring script
def check_performance():
    import psutil
    import pynvml
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # GPU usage
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    
    print(f"CPU: {cpu_percent}%")
    print(f"GPU: {gpu_util.gpu}%")
    print(f"GPU Memory: {gpu_util.memory}%")
    
    if cpu_percent > 15:
        print("⚠️ CPU usage higher than expected!")
```

## Best Practices

1. **Start Small**: Test with one camera before scaling
2. **Monitor Continuously**: Keep GPU/CPU monitors running during migration
3. **Check Logs**: Look for warnings about CPU fallbacks
4. **Gradual Rollout**: Migrate one system at a time in production
5. **Document Changes**: Keep notes on configuration adjustments

## Support Resources

- **Documentation**: `/docs/GPU_Pipeline_Optimization_Plan.md`
- **Phase Summaries**: `/docs/Phase_*_Summary.md`
- **Configuration Reference**: `/docs/Configuration_Reference.md`
- **Performance Profiling**: `/docs/Performance_Profiling_Guide.md`

## Conclusion

The unified GPU pipeline migration typically takes 30-60 minutes and results in dramatic CPU usage reduction. The key is ensuring all components stay on GPU without fallbacks. With proper configuration and validation, the system will achieve the target 5-10% CPU usage while maintaining full functionality. 