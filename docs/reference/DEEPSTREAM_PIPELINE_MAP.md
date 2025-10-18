# DeepStream Video Pipeline: Stream to WebSocket Map

## Overview
This document maps the current DeepStream video processing pipeline from RTSP stream input to WebSocket output, reflecting the implementation in `deepstream_video_pipeline.py`.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DEEPSTREAM VIDEO PIPELINE MAP                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           INPUT LAYER                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

RTSP Streams (`config.py`):
├── Living Room Camera: rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm? (1920x1080) ✅ ENABLED
├── Kitchen Camera: rtsp://192.168.3.214:7447/qt3VqVdZpgG1B4Vk? (1920x1080) ❌ DISABLED
└── Family Room Camera: rtsp://192.168.3.214:7447/4qWTBhW6b4nLeUFE? (1280x720) ❌ DISABLED

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        DEEPSTREAM PIPELINE LAYER                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

`deepstream_video_pipeline.py`:
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MULTI-STREAM INPUT & BATCHING                                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvmultiurisrcbin - DeepStream's unified, high-level source element that internally handles multiple streams (RTSP/file/camera), including decoding, and batching into a single GPU memory stream for downstream processing. This replaces the legacy approach of using individual `nvurisrcbin` and `nvstreammux` elements.                                  │
│ ├── uri-list: rtsp://...,rtsp://... (comma-separated list of source URLs)                                    │
│ ├── sensor-id-list: 0,1,2 (comma-separated list of unique sensor IDs)                                          │
│ ├── max-batch-size: 3 (auto-calculated from enabled streams)                                                │
│ ├── width: 1920 (max source resolution)                                                                       │
│ ├── height: 1080 (max source resolution)                                                                      │
│ ├── batched-push-timeout: -1                                                                                  │
│ ├── live-source: 1 (for RTSP streams)                                                                       │
│ ├── drop-pipeline-eos: 1 (prevents pipeline stalls on stream EOS)                                           │
│ ├── rtsp-reconnect-interval: 30 (seconds)                                                                   │
│ ├── REST API Port: 9000 (for dynamic stream management)                                                     │
│ └── Pad Probe (src): Logs initial batch buffers for debug validation.                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  INFERENCE, MAPANYTHING SGIE & TRACKING                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvinfer (Primary GIE) - DeepStream's GPU inference engine that runs TensorRT models with custom parsers, providing hardware-accelerated AI inference with metadata extraction.                                                                                                   │
│ ├── config-file-path: `pipelines/config_infer_primary_yolo11.ini`                                           │
│ ├── YOLO-11 Custom Parser: `libnvdsparsebbox_yolo11.so`                                                     │
│ ├── input-tensor-meta: True                                                                                 │
│ └── Dynamic config: confidence, IOU, enable flag, and target classes via `custom-lib-props`                 │
│                                                                                                             │
│ MapAnything fused branch (secondary path)                                                                   │
│ ├── nvdspreprocess `mapanything_preprocess`                                                                 │
│ │   • Config: `pipelines/config_preprocess_mapanything_fused.ini`                                           │
│ │   • Custom lib: `pipelines/mapanything_preprocess_fused/libmapanything_preprocess_fused.so`               │
│ │   • Responsibilities: crop PGIE ROIs, normalize RGB, append nine intrinsics channels using intrinsics table│
│ ├── nvinfer `mapanything_sgie_fused` (UID 22)                                                               │
│ │   • Config: `pipelines/config_infer_secondary_mapanything_fused.ini`                                      │
│ │   • Engine: `models/engines/ma_model_fp16_b3_fused.plan`                                                  │
│ │   • Emits tensor meta (`mapanything_fused` → depth/conf tensors) consumed by `_mapanything_depth_probe`   │
│ └── Environment handling: `MA_INTRINSICS_TABLE` default set by pipeline constructor                         │
│                                                                                                             │
│ nvtracker - DeepStream's object tracking element that maintains object identities across frames using algorithms like NvDCF, providing persistent tracking metadata.                                                                                        │
│ ├── ll-config-file: `pipelines/config_tracker_nvdcf_batch.yml`                                              │
│ ├── ll-lib-file: `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`                    │
│ ├── tracker-width: 640                                                                                      │
│ ├── tracker-height: 384                                                                                     │
│ ├── gpu-id: 0                                                                                               │
│ └── tracking-id-reset-mode: 0 (Never reset tracking ID)                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ANALYTICS PROCESSING                                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvdsanalytics (pre-tracker, exclusion) - Removes detections in exclusion ROIs before tracking.              │
│ ├── unique-id: 101                                                                                          │
│ ├── config-file: `pipelines/config_nvdsanalytics_exclude.ini`                                               │
│ └── Pad Probe (src): `_remove_excluded_objects_probe()`                                                     │
│                                                                                                             │
│ nvdsanalytics (post-tracker) - Provides ROI counts, line-crossing, direction, overcrowding metadata.        │
│ ├── unique-id: 201                                                                                          │
│ ├── config-file: `pipelines/config_nvdsanalytics_post.ini`                                                  │
│ └── Pad Probe (src): `_analytics_probe()` extracts telemetry                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  VISUALIZATION & OUTPUT                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nvstreamdemux - De-multiplexes the batched stream into individual streams, one per original source.         │
│ └── Pads: Request pads `src_%u` acquired once and reused (no duplicates). Calibrated to actual `source_id`. │
│                                                                                                             │
│ Per-Stream Visualization & JPEG Branch (Per demux pad):                                                     │
│ ├── queue → nvvideoconvert (pre) → capsfilter (video/x-raw(memory:NVMM), format=RGBA) → nvdsosd →           │
│ │   nvvideoconvert (post) → capsfilter (video/x-raw(memory:NVMM), format=I420) → nvjpegenc → appsink        │
│ ├── OSD Probe (sink): `_per_branch_osd_probe()` draws per-stream overlays using frame_meta.source_id.       │
│ ├── nvjpegenc: GPU JPEG encode; quality = config.visualization.JPEG_QUALITY, preset-level=1                 │
│ └── appsink: emits `new-sample`; callback enqueues JPEG to `jpeg_queues[sensor_id]`                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  METADATA & TELEMETRY EXTRACTION (VIA BUFFER PROBES)                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Instead of a final appsink, metadata is extracted at various points in the pipeline using buffer probes, which allows for inspection without disrupting the primary data flow.                                                                                                    │
│ ├── nvdsanalytics_exclude (src pad): `_remove_excluded_objects_probe()`                                     │
│ │   └── Removes objects detected within defined exclusion zones before they are tracked.                    │
│ ├── nvdsanalytics_post (src pad): `_analytics_probe()`                                                      │
│ │   └── Extracts final object metadata, including tracking IDs and analytics results (ROI, line crossing) for WebSocket telemetry.                                                                                                         │
│ └── nvdsosd (sink pad): `_per_branch_osd_probe()`                                                           │
│     └── Injects custom drawing commands (e.g., for motion trails) into the OSD overlay before rendering.     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        PYTHON APPLICATION LAYER                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

DeepStream integration is encapsulated in `DeepStreamVideoPipeline`, which starts/stops the pipeline and exposes:
- `read_encoded_jpeg(source_id)` to retrieve per-source JPEG bytes
- `get_stats()` for runtime metrics
- Runtime controls: detection config, toggles, and trail visualization

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        MAIN APPLICATION LAYER                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

main.py (ApplicationManager):
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  JPEG FORWARDING (NATIVE DS OSD MODE)                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ _start_jpeg_processing_loop()                                                                               │
│ ├── Iterate sources via `multi_stream_processor.source_info`                                                │
│ ├── For each sensor_id: `read_encoded_jpeg(sensor_id)`                                                     │
│ ├── Prepend 1-byte length + UTF-8 camera id header                                                          │
│ └── websocket_server.broadcast_sync(header + jpeg_bytes)                                                    │
│                                                                                                             │
│ Notes: JPEGs come from per-branch `nvjpegenc` appsinks; Python does not annotate or re-encode frames.        │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

See `docs/reference/WebSocket_API.md` for message types and runtime control endpoints.

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

WebSocket Clients receive:
- JSON stats messages: `{"type":"stats","payload":{...}}`
- JSON telemetry frames: `{"type":"frame","payload":{...}}`
- JPEG frame messages: `{"type":"video_frame","source_id":N,"jpeg_bytes":...}`
- Runtime config broadcasts: detection and visualization updates

## Data Flow Summary

### 1. Stream Input
- **Source**: RTSP streams from IP cameras
- **Format**: H.264/H.265 encoded video
- **Resolution**: 1920x1080 (Living Room), 1280x720 (Family Room)
- **Protocol**: RTSP over TCP with 50ms latency

### 2. DeepStream Processing
- **Input & Batching**: nvmultiurisrcbin handles all sources.
- **Preprocessing**:
  - Primary `nvdspreprocess` driven by `config_preproc.ini`
  - Secondary fused `nvdspreprocess` (MapAnything) with custom library and intrinsics table
- **Inference**:
  - PGIE: YOLO-11 with custom parser
  - SGIE: MapAnything fused TensorRT plan (`ma_model_fp16_b3_fused.plan`)
- **Tracking**: NvDCF tracker
- **Analytics**: nvdsanalytics (exclude + post)
- **Demuxing**: nvstreamdemux separates streams for output.

### 3. Metadata Extraction
- Method: Buffer probes on analytics pads plus the MapAnything SGIE branch.
- `_mapanything_depth_probe`: reads tensor meta (UID 22), computes depth summaries, attaches NVDS user meta.
- Analytics probe: Extracts tracking and analytics data for telemetry.
- OSD probe: Injects drawing commands for trail visualization.

### 4. Python Application
- **Wrapper**: `DeepStreamVideoPipeline` class manages the GStreamer pipeline.
- **Coordination**: Starts, stops, and monitors the pipeline.
- **Output**: Provides methods to get encoded JPEGs from per-source queues.

### 5. WebSocket Streaming
- **Format**: Binary messages with per-source JPEG frames and JSON for telemetry.
- **Structure**: JPEG bytes sent directly.
- **Stats**: JSON messages with performance and tracking metrics.

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
1. DeepStream pipeline with YOLO-11 integration
2. GPU-encoded JPEG per-stream branches with appsink delivery
3. Native OSD mode with custom trail visualization via probe
4. WebSocket server: stats, telemetry, and frame broadcasting
5. Multi-camera support via `nvmultiurisrcbin`
6. Performance monitoring and robust error handling
7. Analytics integration: exclusion + post-tracker analytics
8. Runtime detection config and toggle updates

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
