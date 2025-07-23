# DeepStream Video Pipeline: Stream to WebSocket Map

## Overview
This document provides a comprehensive visual mapping of the DeepStream video processing pipeline from RTSP stream input to WebSocket output, showing the current implementation status and data flow.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DEEPSTREAM VIDEO PIPELINE MAP                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           INPUT LAYER                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

RTSP Streams (config.py):
├── Living Room Camera: rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm? (1920x1080) ✅ ENABLED
├── Kitchen Camera: rtsp://192.168.3.214:7447/qt3VqVdZpgG1B4Vk? (1920x1080) ❌ DISABLED  
└── Family Room Camera: rtsp://192.168.3.214:7447/4qWTBhW6b4nLeUFE? (1280x720) ❌ DISABLED

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        DEEPSTREAM PIPELINE LAYER                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

deepstream_video_pipeline.py:
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  SOURCE BIN CREATION                                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvurisrcbin (per source) - DeepStream's unified source element that handles RTSP, file, and camera inputs with automatic demuxing, decoding, and format conversion to GPU memory.                                                                                    │
│ ├── URI: rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?                                                       │
│ ├── GPU-ID: 0                                                                                               │
│ ├── CUDA Memory Type: Device Memory (0)                                                                     │
│ ├── RTSP Properties:                                                                                        │
│ │   ├── drop-on-latency: True                                                                               │
│ │   ├── latency: 50ms (config.processing.DEEPSTREAM_SOURCE_LATENCY)                                         │
│ │   └── protocols: TCP only (4)                                                                             │
│ └── Dynamic Pad Creation:                                                                                   │
│     ├── pad-added signal → cb_newpad()                                                                      │
│     └── child-added signal → decodebin_child_added()                                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STREAM MULTIPLEXING                                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvstreammux - DeepStream's batch multiplexer that combines multiple video streams into a single batch for efficient GPU processing, handling padding, scaling, and memory management.                                                                                                 │
│ ├── batch-size: 1 (auto-calculated from enabled streams)                                                    │
│ ├── width: 1280 (config.processing.DEEPSTREAM_MUX_SCALE_MODE=2)                                            │
│ ├── height: 720 (config.processing.DEEPSTREAM_MUX_SCALE_MODE=2)                                             │
│ ├── batched-push-timeout: 4000000ms                                                                         │
│ ├── gpu-id: 0                                                                                               │
│ ├── nvbuf-memory-type: 0 (CUDA device memory)                                                               │
│ ├── enable-padding: 1                                                                                       │
│ ├── live-source: 1 (for RTSP streams)                                                                       │
│ ├── sync-inputs: 0 (disable input synchronization for better batch performance)                              │
│ └── drop-pipeline-eos: 1 (drop EOS events to prevent pipeline stalls)                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  GPU PREPROCESSING                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvdspreprocess - DeepStream's GPU-accelerated preprocessing element that handles tensor preparation for inference.                                                                                                                          │
│ ├── config-file: config_preproc.txt                                                                         │
│ ├── gpu-id: 0                                                                                               │
│ ├── enable: True                                                                                            │
│ ├── process-on-frame: True                                                                                  │
│ ├── output-frame-meta: True                                                                                 │
│ ├── output-tensor-meta: True                                                                                │
│ ├── network-input-shape: 1;3;640;640                                                                        │
│ ├── processing-width: 640                                                                                   │
│ ├── processing-height: 640                                                                                  │
│ ├── network-color-format: 0 (RGB)                                                                           │
│ ├── tensor-data-type: 0 (FLOAT32)                                                                           │
│ ├── custom-lib-path: /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so            │
│ └── custom-tensor-preparation-function: CustomTensorPreparation                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  INFERENCE & TRACKING                                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvinfer (Primary GIE) - DeepStream's GPU inference engine that runs TensorRT models with custom parsers, providing hardware-accelerated AI inference with metadata extraction.                                                                                                   │
│ ├── config-file-path: config_infer_primary_yolo11.txt                                                       │
│ ├── YOLO-11 Custom Parser: libnvdsparsebbox_yolo11.so                                                       │
│ ├── input-tensor-meta: True                                                                                 │
│ ├── output-tensor-meta: True                                                                                │
│ └── Pad Probe: _nvinfer_src_pad_buffer_probe()                                                              │
│                                                                                                             │
│ nvtracker - DeepStream's object tracking element that maintains object identities across frames using algorithms like NvDCF, providing persistent tracking metadata.                                                                                        │
│ ├── ll-config-file: config_tracker_nvdcf_batch.yml                                                         │
│ ├── ll-lib-file: libnvds_nvmultiobjecttracker.so                                                            │
│ ├── tracker-width: 640                                                                                      │
│ ├── tracker-height: 384                                                                                     │
│ ├── gpu-id: 0                                                                                               │
│ └── tracking-id-reset-mode: 0 (Never reset tracking ID)                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ANALYTICS PROCESSING                                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvdsanalytics - DeepStream's analytics element that provides advanced video analytics capabilities including ROI filtering, line crossing, direction detection, and overcrowding analysis.                                                                                    │
│ ├── config-file: config_nvdsanalytics.txt                                                                   │
│ ├── enable: 1                                                                                               │
│ ├── config-width: 1280                                                                                      │
│ ├── config-height: 720                                                                                      │
│ ├── osd-mode: 2 (all info display)                                                                          │
│ ├── display-font-size: 12                                                                                   │
│ ├── ROI Filtering: Center area (320;180;960;180;960;540;320;540)                                           │
│ ├── Line Crossing: Entry line (100;360;1180;360)                                                            │
│ ├── Direction Detection: North, South, East, West vectors                                                   │
│ ├── Overcrowding: Threshold 5 objects in full frame ROI                                                     │
│ └── Pad Probe: _analytics_probe()                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  VISUALIZATION & OUTPUT                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvvideoconvert - DeepStream's GPU-accelerated video format converter that transforms video formats and color spaces for display and processing, maintaining GPU memory efficiency.                                                                                              │
│ ├── gpu-id: 0                                                                                               │
│ └── Format conversion for display                                                                           │
│                                                                                                             │
│ nvdsosd - DeepStream's on-screen display element that overlays bounding boxes, labels, and tracking information directly on video frames using GPU rendering.                                                                                          │
│ ├── gpu-id: 0                                                                                               │
│ └── On-screen display for visualization                                                                     │
│                                                                                                             │
│ Conditional Output Paths:                                                                                   │
│ ├── Native OSD Mode (USE_NATIVE_DEEPSTREAM_OSD=True):                                                       │
│ │   ├── nvvideoconvert_post → nvjpegenc → appsink_jpeg                                                      │
│ │   └── GPU-encoded JPEG bytes for direct WebSocket streaming                                               │
│ └── Python OSD Mode (USE_NATIVE_DEEPSTREAM_OSD=False):                                                      │
│     ├── appsink (classic)                                                                                   │
│     ├── emit-signals: True                                                                                  │
│     ├── max-buffers: 1                                                                                      │
│     ├── drop: True                                                                                          │
│     ├── sync: False                                                                                         │
│     └── new-sample signal → _on_new_sample()                                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  TENSOR EXTRACTION (Zero-Copy GPU)                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ _on_new_sample() Processing:                                                                                │
│ ├── Batch Metadata: pyds.gst_buffer_get_nvds_batch_meta()                                                  │
│ ├── Tensor Extraction:                                                                                      │
│ │   ├── NvDsPreProcessBatchMeta (type 27) - Custom preprocessing metadata                                  │
│ │   ├── NvDsPreProcessTensorMeta                                                                           │
│ │   ├── CuPy Zero-Copy: cp.cuda.UnownedMemory()                                                            │
│ │   └── PyTorch Tensor: torch.utils.dlpack.from_dlpack()                                                   │
│ ├── Shape Handling:                                                                                         │
│ │   ├── NCHW (order=0): (n, c, h, w)                                                                       │
│ │   └── NHWC (order=1): (n, h, w, c) → permute(0, 3, 1, 2)                                                │
│ ├── Fallback Tensor Extraction:                                                                             │
│ │   ├── NvDsInferTensorMeta (type 12) - Inference metadata                                                 │
│ │   └── Same zero-copy GPU tensor extraction process                                                        │
│ └── Queue Output: tensor_queue.put()                                                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        UNIFIED GPU PIPELINE LAYER                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

main.py (DeepStreamProcessorWrapper):
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE LOOP                                                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ _processing_loop()                                                                                          │
│ ├── Stage 1: DeepStream Read                                                                                │
│ │   ├── deepstream_processor.read_gpu_tensor()                                                             │
│ │   ├── GPU Tensor Validation                                                                               │
│ │   └── Performance Optimization (if enabled)                                                               │
│ │                                                                                                           │
│ ├── Stage 2: DeepStream Preprocessing (Integrated)                                                          │
│ │   ├── No additional preprocessing needed                                                                  │
│ │   └── Use DeepStream preprocessed tensor directly                                                         │
│ │                                                                                                           │
│ ├── Stage 3: TensorRT Inference                                                                             │
│ │   ├── _process_tensor_gpu_only()                                                                          │
│ │   ├── GPUOnlyDetectionManager                                                                             │
│ │   └── TensorRTModelManager (shared)                                                                       │
│ │                                                                                                           │
│ ├── Stage 4: Tracking (Optional)                                                                            │
│ │   ├── TrackingSystem.update()                                                                             │
│ │   └── ByteTrack algorithm                                                                                 │
│ │                                                                                                           │
│ ├── Stage 5: Visualization Conversion                                                                       │
│ │   ├── _tensor_to_numpy_bgr() (ONLY GPU→CPU transfer)                                                     │
│ │   └── OpenCV format for visualization                                                                     │
│ │                                                                                                           │
│ └── Stage 6: Analysis Frame Creation                                                                        │
│     ├── AnalysisFrame object                                                                                │
│     ├── DetectionResult conversion                                                                          │
│     └── Output to analysis_frame_queue                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        MAIN APPLICATION LAYER                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

main.py (ApplicationManager):
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  RESULT PROCESSING                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ _start_result_processing()                                                                                  │
│ ├── process_results() thread                                                                                │
│ ├── analysis_frame_queue.get()                                                                              │
│ └── _process_analysis_frame()                                                                               │
│                                                                                                             │
│ _process_analysis_frame():                                                                                  │
│ ├── Visualization:                                                                                          │
│ │   ├── visualization_manager.annotate_frame()                                                             │
│ │   ├── Detection boxes, tracking boxes, traces                                                            │
│ │   ├── Keypoints, masks, labels                                                                            │
│ │   └── Performance profiling (if enabled)                                                                  │
│ │                                                                                                           │
│ ├── Frame Encoding:                                                                                         │
│ │   ├── cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])                             │
│ │   └── JPEG compression for WebSocket                                                                      │
│ │                                                                                                           │
│ ├── Binary Message Construction:                                                                            │
│ │   ├── camera_id_bytes = analysis_frame.camera_id.encode('utf-8')                                        │
│ │   ├── camera_id_length = len(camera_id_bytes)                                                            │
│ │   └── binary_message = bytes([camera_id_length]) + camera_id_bytes + jpeg_data                           │
│ │                                                                                                           │
│ └── WebSocket Broadcasting:                                                                                 │
│     ├── websocket_server.broadcast_sync(binary_message)                                                    │
│     └── Frame saving (if enabled)                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        WEBSOCKET SERVER LAYER                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

websocket_server.py (WebSocketServer):
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  SERVER MANAGEMENT                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ start()                                                                                                     │
│ ├── websockets.serve(handle_client, host, port)                                                            │
│ ├── _periodic_stats_broadcast() task                                                                        │
│ └── Event loop management                                                                                   │
│                                                                                                             │
│ broadcast_sync()                                                                                            │
│ ├── asyncio.run_coroutine_threadsafe()                                                                      │
│ └── broadcast() coroutine                                                                                   │
│                                                                                                             │
│ broadcast()                                                                                                 │
│ ├── Message type handling:                                                                                  │
│ │   ├── dict → JSON string                                                                                  │
│ │   ├── bytes → Binary message                                                                              │
│ │   └── str → Direct string                                                                                 │
│ ├── asyncio.gather() for all clients                                                                        │
│ └── Error handling per client                                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  CLIENT HANDLING                                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ handle_client()                                                                                             │
│ ├── Client connection tracking                                                                              │
│ ├── Message processing:                                                                                     │
│ │   ├── clear_stats command                                                                                 │
│ │   ├── set_vis_toggle command                                                                              │
│ │   └── JSON message parsing                                                                                │
│ └── Connection lifecycle management                                                                         │
│                                                                                                             │
│ _periodic_stats_broadcast()                                                                                 │
│ ├── stats_callback() invocation                                                                             │
│ ├── JSON stats message construction                                                                         │
│ └── Periodic broadcasting (1 second intervals)                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           OUTPUT LAYER                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

WebSocket Clients:
├── Binary Frame Messages: [camera_id_length][camera_id][jpeg_data]
├── JSON Stats Messages: Performance metrics, system status
└── Toggle Update Messages: Visualization control

## Data Flow Summary

### 1. Stream Input
- **Source**: RTSP streams from IP cameras
- **Format**: H.264/H.265 encoded video
- **Resolution**: 1920x1080 (Living Room), 1280x720 (Family Room)
- **Protocol**: RTSP over TCP with 50ms latency

### 2. DeepStream Processing
- **Decoder**: nvurisrcbin with NVIDIA hardware decoder
- **Memory**: CUDA device memory (zero-copy)
- **Batching**: nvstreammux with batch size 1
- **Preprocessing**: nvdspreprocess with custom library
- **Inference**: YOLO-11 with custom parser
- **Tracking**: NvDCF tracker (configurable)
- **Analytics**: nvdsanalytics with ROI, line crossing, direction detection

### 3. Tensor Extraction
- **Method**: Zero-copy GPU tensor extraction
- **Library**: CuPy + PyTorch DLPack
- **Format**: NCHW or NHWC tensors
- **Memory**: GPU memory only (no CPU transfer)
- **Fallback**: Inference metadata if preprocessing metadata unavailable

### 4. Unified Pipeline
- **Processing**: TensorRT inference on GPU
- **Tracking**: ByteTrack algorithm (configurable)
- **Conversion**: Single GPU→CPU transfer for visualization
- **Output**: AnalysisFrame objects

### 5. WebSocket Streaming
- **Format**: Binary messages with JPEG frames
- **Structure**: [camera_id_length][camera_id][jpeg_data]
- **Stats**: JSON messages with performance metrics
- **Control**: Toggle messages for visualization

## Performance Characteristics

### GPU Memory Usage
- **DeepStream**: ~2-4GB for video decoding and preprocessing
- **TensorRT**: ~1-2GB for inference engines
- **Tracking**: ~100-200MB for tracking state
- **Analytics**: ~50-100MB for analytics processing
- **Total**: ~4-6GB peak usage

### Latency
- **RTSP Input**: 50ms latency setting
- **DeepStream Processing**: ~10-30ms per frame
- **TensorRT Inference**: ~5-15ms per frame
- **Analytics**: ~1-5ms per frame
- **WebSocket Output**: ~1-5ms per frame
- **Total Pipeline**: ~20-50ms end-to-end

### Throughput
- **Input**: 30 FPS per camera
- **Processing**: 10-30 FPS depending on model complexity
- **Output**: 20 FPS max (configurable)
- **Multi-camera**: Linear scaling with GPU memory

## Current Status

### ✅ Implemented Components
1. **DeepStream Pipeline**: Complete with YOLO-11 integration
2. **Zero-Copy Tensor Extraction**: Working with CuPy/PyTorch
3. **Native OSD Mode**: GPU-encoded JPEG streaming
4. **Python OSD Mode**: Frame annotation and processing
5. **WebSocket Server**: Binary frame streaming
6. **Multi-camera Support**: Configurable RTSP streams
7. **Performance Monitoring**: Comprehensive metrics
8. **Error Handling**: Robust pipeline management
9. **Analytics Integration**: ROI filtering, line crossing, direction detection
10. **Configurable Tracking**: Native DeepStream vs. Python ByteTrack

### 🔧 Configuration Options
1. **Camera Sources**: RTSP streams with individual settings
2. **Processing Pipeline**: DeepStream + TensorRT
3. **Visualization**: Configurable display options (Native vs. Python)
4. **Tracking**: Configurable tracking options (Native vs. Python)
5. **Performance**: Profiling and optimization settings
6. **Output**: WebSocket streaming and file saving

### 📊 Monitoring Points
1. **GPU Memory**: Real-time memory usage tracking
2. **Pipeline Performance**: FPS and latency metrics
3. **Error Rates**: Consecutive failure tracking
4. **WebSocket Stats**: Client connections and message rates
5. **System Resources**: CPU and GPU utilization
6. **Analytics Events**: ROI violations, line crossings, overcrowding

## Key Features

### GPU-Only Operation
- **Zero CPU Fallbacks**: All operations must succeed on GPU
- **Memory Efficiency**: Minimal GPU→CPU transfers
- **Performance Optimization**: TensorRT and DeepStream integration

### Real-Time Streaming
- **Low Latency**: <50ms end-to-end processing
- **High Throughput**: 20+ FPS per camera
- **Scalable**: Multi-camera support with linear scaling

### Robust Error Handling
- **Pipeline Recovery**: Automatic restart on failures
- **Resource Management**: GPU memory pool and cleanup
- **Monitoring**: Comprehensive performance tracking

### Flexible Configuration
- **Dynamic Sources**: Enable/disable cameras at runtime
- **Model Selection**: YOLO-11 with custom parser support
- **Visualization Control**: Real-time toggle updates via WebSocket
- **Analytics Configuration**: ROI, line crossing, direction detection

### Advanced Analytics
- **ROI Filtering**: Define regions of interest for object counting
- **Line Crossing**: Detect objects crossing defined lines
- **Direction Detection**: Analyze object movement patterns
- **Overcrowding**: Monitor object density in specified areas

This pipeline represents a complete GPU-accelerated video processing system optimized for real-time multi-camera analysis with minimal latency and maximum throughput, featuring configurable tracking and visualization modes with advanced analytics capabilities. 