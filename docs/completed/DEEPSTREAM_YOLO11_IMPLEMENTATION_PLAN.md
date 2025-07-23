# DeepStream 7.1 + YOLO11/12 Implementation Plan

## Project Overview

This plan outlines the implementation of a GPU-accelerated RTSP video processing pipeline using DeepStream 7.1 and YOLO11/12 for object detection and tracking. The system will process multiple RTSP streams with zero CPU copies, utilizing TensorRT for optimized inference.

## Current State Analysis

### ✅ What's Working
- NVIDIA RTX 3060 with CUDA 12.9
- Core Python packages (torch, opencv, numpy)
- DeepStream 7.1 files available (.deb + SDK)
- Existing codebase with GPU pipeline architecture

### ❌ What Needs Implementation
- DeepStream 7.1 installation and configuration
- YOLO11/12 model integration
- Custom parser for YOLO11/12
- TensorRT engine optimization
- Multi-stream RTSP processing
- Zero-copy GPU pipeline

## Phase 1: System Setup (Step A)

### 1.1 Environment Setup
- [ ] Run `setup_ubuntu_system.sh` to install all dependencies
- [ ] Install TensorRT from NVIDIA developer portal
- [ ] Verify DeepStream 7.1 installation
- [ ] Test CUDA and GStreamer NVIDIA plugins

### 1.2 Python Environment
- [ ] Create new virtual environment (`venv_new`)
- [ ] Install all required Python packages
- [ ] Verify PyTorch CUDA support
- [ ] Test DeepStream Python bindings

### 1.3 Model Preparation
- [ ] Download YOLO11/12 models
- [ ] Convert models to TensorRT format
- [ ] Create custom parser for YOLO11/12
- [ ] Test model inference

## Phase 2: DeepStream Integration (Step B)

### 2.1 DeepStream Pipeline Architecture

```
RTSP Stream → nvstreammux → nvinfer (YOLO11/12) → nvtracker → nvmultistreamtiler → nvv4l2h264enc → RTSP Output
```

### 2.2 Key Components

#### 2.2.1 Input Stage
- **nvurisrcbin**: RTSP stream ingestion
- **nvv4l2decoder**: Hardware-accelerated decoding
- **nvinferserver**: TensorRT inference with YOLO11/12

#### 2.2.2 Processing Stage
- **nvtracker**: Multi-object tracking (NvDCF)
- **nvmultistreamtiler**: Multi-stream composition
- **nvosd**: On-screen display with bounding boxes

#### 2.2.3 Output Stage
- **nvv4l2h264enc**: Hardware-accelerated encoding
- **nvv4l2h265enc**: HEVC encoding (optional)
- **rtph264pay**: RTP packetization

### 2.3 Custom Parser Implementation

#### 2.3.1 YOLO11/12 Parser Requirements
```cpp
// Custom parser for YOLO11/12 output format
class YOLO11Parser {
    // Parse YOLO11/12 detection outputs
    // Handle different output formats (boxes, masks, poses)
    // Convert to DeepStream metadata format
};
```

#### 2.3.2 Parser Features
- Support for YOLO11/12 detection outputs
- Pose estimation support (if using YOLO11-pose)
- Instance segmentation support
- Confidence threshold filtering
- NMS (Non-Maximum Suppression)

### 2.4 Configuration Files

#### 2.4.1 Primary Inference Config
```txt
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=models/yolo11_detection.engine
labelfile-path=models/coco_labels.txt
batch-size=1
process-mode=1
network-mode=1
num-detected-classes=80
interval=0
gie-unique-id=1
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseYOLO11
custom-lib-path=libnvdsparsebbox_yolo11.so
```

#### 2.4.2 Tracker Config
```yaml
# tracker_NvDCF_perf.yml
tracker-width: 1920
tracker-height: 1080
gpu-id: 0
enable-batch-process: 1
enable-past-frame: 1
input-tensor-from-meta: 1
input-tensor-meta-keys: "detector_bbox"
past-frame: 30
enable-reid: 1
reid-feature-size: 128
reid-history-size: 100
```

## Phase 3: Implementation Steps

### 3.1 DeepStream Installation & Setup
```bash
# Install DeepStream 7.1
sudo apt install -y ./deepstream-7.1_7.1.0-1_amd64.deb

# Extract SDK and copy Python bindings
tar -xvf deepstream_sdk_v7.1.0_x86_64.tbz2
sudo cp -r deepstream_sdk_v7.1.0_x86_64/sources/deepstream_python_apps/bindings /opt/nvidia/deepstream/deepstream-7.1/

# Set environment variables
export DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream-7.1
export LD_LIBRARY_PATH=$DEEPSTREAM_DIR/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=$DEEPSTREAM_DIR/lib/gst-plugins
```

### 3.2 YOLO11/12 Model Preparation
```bash
# Download YOLO11/12 models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt

# Convert to TensorRT
python tensorrt_builder.py --model yolo11m.pt --engine yolo11m_detection.engine --fp16
```

### 3.3 Custom Parser Development
```cpp
// libnvdsparsebbox_yolo11.cpp
#include "nvdsinfer_custom_impl.h"
#include <cstring>
#include <iostream>

extern "C" bool NvDsInferParseYOLO11(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    // YOLO11/12 specific parsing logic
    // Handle different output formats
    // Convert to DeepStream metadata
    return true;
}
```

### 3.4 Python Integration
```python
# deepstream_yolo11_pipeline.py
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DeepStreamYOLO11Pipeline:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.source_bin = None
        self.sink_bin = None
        
    def create_pipeline(self):
        # Create GStreamer pipeline
        # Configure nvinfer with YOLO11/12
        # Set up custom parser
        pass
        
    def start_pipeline(self):
        # Start RTSP processing
        # Handle multiple streams
        pass
```

## Phase 4: Testing & Validation

### 4.1 Unit Tests
- [ ] Test YOLO11/12 model loading
- [ ] Test custom parser functionality
- [ ] Test TensorRT engine inference
- [ ] Test RTSP stream ingestion

### 4.2 Integration Tests
- [ ] Test complete DeepStream pipeline
- [ ] Test multi-stream processing
- [ ] Test GPU memory usage
- [ ] Test performance benchmarks

### 4.3 Performance Validation
- [ ] Measure FPS for single stream
- [ ] Measure FPS for multiple streams
- [ ] Monitor GPU memory usage
- [ ] Validate zero-copy processing

## Phase 5: Optimization

### 5.1 TensorRT Optimization
- [ ] FP16 precision optimization
- [ ] Batch size optimization
- [ ] Model quantization (INT8)
- [ ] Dynamic shape optimization

### 5.2 Pipeline Optimization
- [ ] Buffer pool optimization
- [ ] Threading optimization
- [ ] Memory management
- [ ] Stream synchronization

### 5.3 Multi-Stream Optimization
- [ ] Stream multiplexing
- [ ] Load balancing
- [ ] Resource sharing
- [ ] Scalability testing

## Implementation Timeline

### Week 1: System Setup
- Day 1-2: Run setup script, install dependencies
- Day 3-4: Install TensorRT, verify installations
- Day 5: Test basic DeepStream functionality

### Week 2: Model Integration
- Day 1-2: Download and convert YOLO11/12 models
- Day 3-4: Develop custom parser
- Day 5: Test model inference

### Week 3: Pipeline Development
- Day 1-2: Create DeepStream pipeline
- Day 3-4: Integrate custom parser
- Day 5: Test single stream processing

### Week 4: Multi-Stream & Optimization
- Day 1-2: Implement multi-stream support
- Day 3-4: Performance optimization
- Day 5: Final testing and validation

## Success Criteria

### Performance Targets
- **Single Stream**: 60+ FPS (1080p)
- **Multi-Stream**: 30+ FPS per stream (4 streams)
- **Latency**: <100ms end-to-end
- **GPU Memory**: <4GB for 4 streams

### Functionality Targets
- [ ] Zero CPU copies for video processing
- [ ] YOLO11/12 detection working
- [ ] Multi-object tracking functional
- [ ] RTSP output working
- [ ] Web dashboard accessible

### Quality Targets
- [ ] Detection accuracy >90% mAP
- [ ] Tracking consistency >95%
- [ ] System stability >99.9% uptime
- [ ] Memory leaks <1MB/hour

## Risk Mitigation

### Technical Risks
1. **YOLO11/12 compatibility**: Use proven YOLO11 models, test thoroughly
2. **Custom parser complexity**: Start with simple parser, iterate
3. **Performance issues**: Profile early, optimize incrementally
4. **Memory leaks**: Implement proper cleanup, monitor continuously

### Operational Risks
1. **Dependency conflicts**: Use isolated environment, document versions
2. **Hardware limitations**: Test on target hardware early
3. **Integration complexity**: Break into smaller, testable components

## Next Steps

1. **Immediate**: Run `setup_ubuntu_system.sh`
2. **Next**: Install TensorRT and verify DeepStream
3. **Following**: Download YOLO11/12 models
4. **Then**: Develop custom parser
5. **Finally**: Integrate into existing pipeline

## Resources

- [DeepStream 7.1 Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [YOLO11 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [GStreamer NVIDIA Plugins](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) 