# DeepStream Integration Guide

This document describes the DeepStream 6.4 integration in the Noesis project, replacing the previous DALI-based video pipeline.

## Overview

The DeepStream pipeline provides end-to-end GPU-accelerated video processing with the following architecture:

```
nvurisrcbin → nvstreammux → nvdspreprocess → appsink → TensorRT → appsrc → nvtracker → nvdsosd → sink
```

### Key Components

- **nvurisrcbin**: Hardware-accelerated video decoding for RTSP/file sources
- **nvstreammux**: Batching multiple streams for efficient processing
- **nvdspreprocess**: GPU-accelerated preprocessing (resize, normalize)
- **TensorRT**: FP16 inference for detection/segmentation/pose models
- **nvtracker**: Optional DeepStream tracker (NvDCF or custom ByteTrack)
- **nvdsosd**: On-screen display for visualization

## Building and Installation

### Prerequisites

1. **Base Docker Image**:
   ```bash
   docker pull nvcr.io/nvidia/deepstream:6.4-gc-triton-devel
   ```

2. **System Requirements**:
   - NVIDIA GPU with Compute Capability >= 7.0
   - CUDA 12.2+
   - TensorRT 8.6+
   - GStreamer 1.16+

### Building the DLPack Converter

The nvbufsurface_to_dlpack module enables zero-copy tensor conversion:

```bash
# Install dependencies
pip install cython numpy

# Set environment
export DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream
export CUDA_PATH=/usr/local/cuda

# Build extension
python setup_nvbufsurface.py build_ext --inplace
```

### Docker Build

```dockerfile
FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel

# Install Python dependencies
RUN pip install torch torchvision pygobject numpy opencv-python

# Copy application
COPY . /app
WORKDIR /app

# Build DLPack converter
RUN python setup_nvbufsurface.py build_ext --inplace

# Copy config files
COPY config_preproc.txt tracker_nvdcf.yml /app/

CMD ["python", "main.py"]
```

## Configuration

### Stream Configuration

In `config.py`, configure RTSP streams with the `enabled` attribute:

```python
RTSP_STREAMS: List[Dict[str, Any]] = [
    {
        "name": "Living Room Camera",
        "url": "rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?",
        "width": 1920,
        "height": 1080,
        "enabled": True
    },
    {
        "name": "Kitchen Camera",
        "url": "rtsp://192.168.3.214:7447/qt3VqVdZpgG1B4Vk?",
        "width": 1920,
        "height": 1080,
        "enabled": False  # Disabled stream
    }
]
```

### DeepStream Settings

```python
# DeepStream Pipeline Configuration
ENABLE_DEEPSTREAM: bool = True
DEEPSTREAM_SOURCE_LATENCY: int = 100  # ms
DEEPSTREAM_MUX_BATCH_SIZE: int = 1   # Auto-calculated from enabled streams
DEEPSTREAM_MUX_SCALE_MODE: int = 2   # 0=stretch, 1=crop, 2=letterbox
DEEPSTREAM_PREPROCESS_CONFIG: str = "config_preproc.txt"
DEEPSTREAM_TRACKER_CONFIG: str = "tracker_nvdcf.yml"
DEEPSTREAM_TRACKER_LIB: str = ""      # Empty = NvDCF, or "libbytetrack_ds.so"
DEEPSTREAM_ENABLE_OSD: bool = True
```

### Preprocessing Configuration (config_preproc.txt)

```ini
[property]
target-unique-ids=src_tensor
tensor-name=src_tensor
tensor-data-type=1  # FP16
batch-size=1
channels=3
width=640
height=640
gpu-id=0
nvbuf-memory-type=1
network-input-format=0  # RGB
mean-0=123.675
mean-1=116.28
mean-2=103.53
std-0=58.395
std-1=57.12
std-2=57.375
```

## Custom Tracker Integration

### ByteTrack Integration

To use ByteTrack instead of NvDCF:

1. Build ByteTrack as a DeepStream plugin:
   ```bash
   git clone https://github.com/your-repo/bytetrack-deepstream
   cd bytetrack-deepstream
   make
   cp libbytetrack_ds.so /opt/nvidia/deepstream/deepstream/lib/
   ```

   **Note**: ByteTrack DLL must be built against DeepStream 6.4 SDK headers and linked with the appropriate CUDA libraries. Ensure the plugin exports the required NvMOT_* symbols and implements the NvMOTContext interface for proper integration.

2. Set in config:
   ```python
   DEEPSTREAM_TRACKER_LIB = "libbytetrack_ds.so"
   ```

### Requirements for Custom Trackers

Custom tracker libraries must:
- Export NvMOT_* symbols as defined in the DeepStream SDK
- Be built against DeepStream 6.4 headers
- Support the NvMOTContext interface

Example tracker interface:
```c
extern "C" {
    NvMOTContext* NvMOT_CreateContext(
        NvMOTConfig *pConfigIn,
        NvMOTConfigResponse *pConfigResponse
    );
    
    NvMOTStatus NvMOT_Process(
        NvMOTContext *pContext,
        NvMOTProcessParams *pParams,
        NvMOTTrackedObjBatch *pTrackedObjBatch
    );
    
    void NvMOT_DestroyContext(NvMOTContext *pContext);
}
```

## Environment Variables

- `DEEPSTREAM_PATH`: Path to DeepStream installation (default: `/opt/nvidia/deepstream/deepstream`)
- `CUDA_PATH`: Path to CUDA installation (default: `/usr/local/cuda`)
- `GST_DEBUG`: GStreamer debug level (e.g., `3` for INFO)
- `CUDA_VISIBLE_DEVICES`: GPU device selection

## Debugging

### Enable GStreamer Debug Output

```bash
export GST_DEBUG=3
export GST_DEBUG_FILE=gst_debug.log
```

### Generate Pipeline Graph

```bash
export GST_DEBUG_DUMP_DOT_DIR=/tmp
# Graphs will be generated in /tmp as .dot files
```

### Common Issues

1. **"Failed to create nvurisrcbin"**
   - Ensure DeepStream is properly installed
   - Check GPU driver compatibility

2. **"nvbufsurface_to_dlpack not found"**
   - Rebuild the Cython extension
   - Verify PYTHONPATH includes the build directory

3. **"Pipeline error: no element X"**
   - Missing GStreamer plugin
   - Run `gst-inspect-1.0 X` to verify

4. **RTSP Connection Issues**
   - Verify network connectivity
   - Check firewall rules
   - Test with `gst-launch-1.0 uridecodebin uri=rtsp://...`

## Performance Tuning

### Batch Size Optimization

The batch size is automatically calculated from enabled streams:
```python
batch_size = sum(1 for s in RTSP_STREAMS if s.get("enabled", True))
```

For optimal performance:
- Keep batch sizes power of 2 (1, 2, 4, 8)
- Match TensorRT engine batch size

### Memory Pool Configuration

```python
GPU_MEMORY_POOL_SIZE_MB: int = 500
ENABLE_MEMORY_POOLING: bool = True
```

### Latency Optimization

For low-latency applications:
```python
DEEPSTREAM_SOURCE_LATENCY: int = 50  # Reduce from 100
batched-push-timeout: 20000  # Reduce from 40000
```

## Migration from DALI

Key differences from the previous DALI implementation:

1. **Pipeline Structure**: DeepStream uses GStreamer elements vs DALI's operator graph
2. **Tensor Format**: DeepStream provides NVMM buffers requiring DLPack conversion
3. **Preprocessing**: Configured via text file instead of code
4. **Multi-stream**: Native batching support vs manual coordination

### Removed Components
- `dali_video_pipeline.py`
- `dali_video_processor.py`
- `advanced_resize_optimizer.py`
- All DALI configuration keys

### New Components
- `deepstream_video_pipeline.py`
- `deepstream_inference_bridge.py`
- `nvbufsurface_to_dlpack.pyx`
- Configuration files: `config_preproc.txt`, `tracker_nvdcf.yml`

## API Reference

### DeepStreamVideoPipeline

```python
pipeline = DeepStreamVideoPipeline(
    sources=[...],  # List of source configs
    config=app_config,
    device_id=0
)

# Start pipeline
success = pipeline.start()

# Read tensor
ret, tensor_data = pipeline.read_gpu_tensor()
# tensor_data = {
#     'tensor': torch.Tensor,  # GPU tensor
#     'source_id': int,        # Camera index
#     'frame_num': int,        # Frame number
#     'timestamp': float       # Timestamp
# }

# Get stats
stats = pipeline.get_stats()

# Stop pipeline
pipeline.stop()
```

### DeepStreamInferenceBridge

```python
bridge = DeepStreamInferenceBridge(config, output_queue)

# Process tensor
buffer = bridge.process_tensor(tensor, metadata)

# Push to pipeline
bridge.push_buffer(buffer)

# Get stats
stats = bridge.get_stats()
```

## Testing

### Smoke Test

```bash
python deepstream_video_pipeline.py --source rtsp://camera_url --duration 10
```

### Integration Test

```bash
python main.py --rtsp rtsp://camera_url --use-unified-pipeline --enable-deepstream
```

### Performance Test

```bash
python performance_benchmark.py --pipeline deepstream --duration 300
``` 