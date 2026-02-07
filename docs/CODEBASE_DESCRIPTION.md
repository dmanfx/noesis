# Noesis Codebase Description
_Status: current as of 2026-02-02._

## Project Overview

**Noesis** is a high-performance, GPU-accelerated real-time video analytics application built on **NVIDIA DeepStream 8.0 Service Maker**. The system processes multiple RTSP camera streams end-to-end on the GPU, performing object detection, tracking, re-identification, analytics, depth estimation, and visualization with zero CPU fallbacks in the core pipeline.

### Purpose
Real-time multi-camera video analytics pipeline for:
- Object detection and instance segmentation (YOLOv11)
- Multi-object tracking with cross-camera re-identification
- Spatial analytics (ROI filtering, line crossing, occupancy)
- Bird's-eye view (BEV) visualization with trail rendering
- Monocular depth estimation (MapAnything)
- Real-time WebRTC streaming to browser frontends

### Key Features
- **End-to-End GPU Processing**: All operations from video decoding (NVDEC) to AI inference (TensorRT) run on GPU via NVMM surfaces
- **Multi-Stream Support**: Processes multiple RTSP camera streams simultaneously via `nvmultiurisrcbin`
- **YOLOv11 Object Detection**: Custom-parsed YOLOv11 segmentation model for primary inference with instance masks
- **Advanced Tracking**: NVIDIA NvDCF tracker with OSNet-based re-identification for stable cross-camera IDs
- **Pose-assisted StableID**: YOLO26 pose SGIE ratio features can be fused into StableID as a secondary signal (bounded in RAM; no disk persistence)
- **Analytics**: ROI filtering, line crossing, direction detection, overcrowding via `nvdsanalytics`
- **Bird's-Eye View (BEV)**: Real-time top-down visualization with homography-based projection and motion trails
- **MapAnything Integration**: Full-frame depth estimation with valve-gated GPU inference branch
- **Real-time WebRTC Streaming**: H.264 video via RTSP→WebRTC gateway for browser delivery
- **Motion Trails**: GPU-rendered persistent trails behind tracked objects in the mosaic OSD

### Technology Stack
- **Backend**: Python 3.10+, NVIDIA DeepStream 8.0 Service Maker (`pyservicemaker`), TensorRT
- **Frontend**: React + TypeScript, Vite, WebSocket/WebRTC client
- **ML Models**: YOLOv11-seg (Ultralytics), OSNet ReID (torchreid), MapAnything (Meta Research), OPtional:Yolo26_seg/pose, RF-DETR (--pgie-profile)
- **GPU Libraries**: CUDA, cuDNN, TensorRT, `pyds` DeepStream Python bindings
- **Communication**: WebSockets (JSON telemetry + optional BEV JPEG binaries), WebRTC (H.264 video)

---

## Architecture

### High-Level Structure

The system follows a **layered architecture** with clear separation between the GPU pipeline (DeepStream 8 Service Maker) and the application layer:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       React Frontend (oai2-fe)                          │
│  - StreamPanel (WebRTC video), BevPanel, DepthDrawer, ControlsPanel     │
│  - WebSocket client for metadata + WebRTC for video                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        │ WebSocket (ws://host:6008)                     │
        │ JSON telemetry + optional BEV JPEG binaries    │
        │                                                │
        │ WebRTC (via MosaicWebRTCGateway)               │
        │ H.264 video over SRTP                          │
        └───────────────────────┬────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    Python Application Layer                             │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ DS8 Runtime Harness (noesis/ds8_runtime.py)                        │ │
│  │  - Coordinates pipeline lifecycle (build → prepare → activate)     │ │
│  │  - Manages WebSocket server and REST APIs                          │ │
│  │  - Attaches metadata hooks (analytics, ReID, depth)                │ │
│  └───────────────────┬────────────────────────────────────────────────┘ │
│                      │                                                  │
│  ┌───────────────────▼───────────────────┬────────────────────────────┐ │
│  │ WebSocketServer                       │ MosaicWebRTCGateway        │ │
│  │ (websocket_server.py)                 │ (mosaic_webrtc_gateway.py) │ │
│  │  - Client management                  │  - RTSP→WebRTC passthrough │ │
│  │  - Metadata broadcast                 │  - Browser signaling       │ │
│  │  - RPC handlers                       │  - No transcode            │ │
│  └───────────────────────────────────────┴────────────────────────────┘ │
│                      │                                                  │
│  ┌───────────────────▼────────────────────────────────────────────────┐ │
│  │           DS8 Pipeline (noesis/pipelines/ds8_pipeline.py)          │ │
│  │  - pyservicemaker.Pipeline graph construction from YAML            │ │
│  │  - Component wiring (sources → inference → tracking → output)      │ │
│  │  - Valve-based MapAnything gating                                  │ │
│  └───────────────────┬────────────────────────────────────────────────┘ │
│                      │                                                  │
│  ┌───────────────────▼────────────────────────────────────────────────┐ │
│  │        Metadata Hooks (noesis/pipelines/hooks.py)                  │ │
│  │  - BatchMetadataOperator probes for telemetry extraction           │ │
│  │  - Trail overlay rendering via NvDsDisplayMeta                     │ │
│  │  - MapAnything tensor postprocess                                  │ │
│  │  - OSNet ReID embedding extraction for StableIDManager             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│              DeepStream 8 Service Maker Runtime (GPU)                    │
│                                                                          │
│  nvmultiurisrcbin ──► nvstreammux ──► nvdspreprocess ──► nvinfer(PGIE)   │
│                                                              │           │
│                               ┌───────────────────────────── tee         │
│                               │                               │          │
│                               ▼                               ▼          │
│                         nvtracker                    queue → valve       │
│                               │                               │          │
│                               ▼                               ▼          │
│                       nvdsanalytics               nvinfer(MapAnything)   │
│                               │                               │          │
│                               ▼                               ▼          │
│                   nvinfer(ReID SGIE)                      fakesink       │
│                               │                                          │
│                               ▼                                          │
│                     nvmultistreamtiler                                   │
│                               │                                          │
│                               ▼                                          │
│                           nvdsosd                                        │
│                               │                                          │
│                           sink_tee ───────────────────────────────┐      │
│                               │                                   │      │
│                               ▼                                   ▼      │
│                        nvrtspoutsinkbin                      placeholders │
│                           (H.264 RTSP)                      (Flow-reserved)│
└──────────────────────────────────────────────────────────────────────────┘
```

### Modules/Components

#### 1. **DS8 Pipeline Layer** (`noesis/pipelines/ds8_pipeline.py`)
- **DS8Pipeline**: Main pipeline dataclass managing DeepStream 8.0 Service Maker components
- **Component**: Dataclass representing a GStreamer element (name, element type, config, downstream)
- **build_pipeline()**: Constructs pipeline from YAML configuration using `pyservicemaker.Pipeline`
- **Key Pipeline Stages**:
  - **Sources**: `nvmultiurisrcbin` (default) for multi-stream ingest with reconnection
  - **Stream Muxer**: `nvstreammux` batches streams (implicit in nvmultiurisrcbin)
  - **Preprocess**: Optional `nvdspreprocess` for ROI/tensor preparation
  - **Primary Inference**: `nvinfer` with YOLOv11-seg model (instance segmentation)
  - **Tee**: Splits flow to tracker chain and optional MapAnything branch
  - **Tracker**: `nvtracker` with NvDCF multi-object tracking
  - **Analytics**: `nvdsanalytics` for ROI/line crossing events
  - **ReID SGIE**: `nvinfer` with OSNet model for cross-camera re-identification
  - **Tiler**: `nvmultistreamtiler` creates mosaic view
  - **OSD**: `nvdsosd` overlays bounding boxes, masks, labels, and trails
  - **Output Sinks**: RTSP (`nvrtspoutsinkbin`) for mosaic delivery (consumed by the RTSP→WebRTC gateway). Mosaic JPEG/WebSocket output is removed in DS8.
- **Depth Gating**: Valve-based gating for MapAnything branch with automatic priming

#### 2. **Metadata Hooks** (`noesis/pipelines/hooks.py`)
- **BatchMetadataOperator Probes**: Attach to pipeline nodes for per-frame processing
- **Hook Types**:
  - `attach_intrinsics_hook`: Extracts camera intrinsics per frame
  - `attach_mapanything_postprocess_hook`: Decodes depth tensor meta from MapAnything SGIE
  - `attach_pose_feature_hook`: Parses YOLO26 pose SGIE tensors and attaches pose ratio user meta (for StableID)
  - `attach_analytics_telemetry_hook`: Extracts tracks, events, and occupancy data
  - `attach_trail_overlay_hook`: Renders motion trails via NvDsDisplayMeta lines
  - `attach_osd_label_hook`: Customizes OSD text labels (stable ID, confidence)
  - `attach_exclude_prune_hook`: Removes objects outside ROI exclusion zones
  - `attach_analytics_reload_bridge`: Hot-reloads analytics config at runtime
- **TrailOverlayProcessor**: GPU-rendered per-person motion trails with configurable styling
- **MapAnythingProcessor**: Tensor-to-depth conversion with async storage

#### 3. **DS8 Runtime Harness** (`noesis/ds8_runtime.py`)
- **Entry Point**: `main()` function orchestrating the application lifecycle
- **Responsibilities**:
  - Parse CLI arguments and load YAML configuration
  - Build DS8 pipeline from `config/infer.yaml`
  - Attach all metadata hooks (intrinsics, analytics, depth, trails, ReID)
  - Start WebSocket server and REST API
  - Activate pipeline via `pyservicemaker.Pipeline.activate()`
  - Manage WebRTC gateway startup (RTSP→WebRTC passthrough)
  - Handle graceful shutdown on SIGINT/SIGTERM
- **CalibrationProvider**: Provides camera intrinsics/extrinsics for BEV rendering

#### 4. **WebSocket Server** (`websocket_server.py`)
- **WebSocketServer**: Async WebSocket server for real-time client communication
- **Features**:
  - Binary frame broadcasting (optional BEV JPEG frames)
  - JSON metadata streaming (detections, tracks, analytics, BEV points)
  - RPC handlers: calibration, depth requests, floorplan generation, BEV config
  - WebRTC signaling relay for `MosaicWebRTCGateway`
  - Toggle callbacks for trail visualization state
  - Rate limiting and telemetry tracking

#### 5. **WebRTC Gateway** (`noesis/mosaic_webrtc_gateway.py`)
- **MosaicWebRTCGateway**: Ultra-light GStreamer WebRTC gateway
- **Pipeline**: `rtspsrc → rtph264depay → h264parse → rtph264pay → webrtcbin`
- **Zero Transcode**: Passthrough H.264 from RTSP to WebRTC (no GPU load)
- **Signaling**: Uses WebSocketServer for SDP offer/answer and ICE candidate exchange
- **IDR Requests**: Forces keyframes on peer connection for fast startup

#### 6. **Stable ID Manager** (`reid/stable_id_manager.py`)
- **StableIDManager**: Cross-camera re-identification using visual embeddings
- **Features**:
  - Maps `(sensor_id, ds_obj_id)` → `stable_id` for persistent identities
  - Ghost registry for re-association after occlusion/disappearance
  - Identity gallery with recent embeddings per stable ID
  - Optional pose ratio fusion (YOLO26 pose SGIE) via native bridge; bounded pose gallery + TTL pruning
  - Configurable cosine similarity thresholds for matching
  - Multi-zone active support for overlapping camera fields of view
  - EMA smoothing and adaptive penalties for robustness
- **EmbeddingExtractor**: OSNet-based feature extraction (optional; DS8 uses SGIE tensors)

#### 7. **Telemetry & BEV** (`noesis/telemetry/`)
- **bev.py**: Bird's-eye view rendering using homography transforms
  - `BevRenderer`: Renders top-down view from camera detections
  - `HomographyCache`: Caches homography matrices per calibration
  - `BevTrailConfig`: Configurable trail rendering for BEV canvas
  - Projects 2D footpoints to 3D world coordinates
- **publishers.py**: Telemetry publishers
  - `DepthTelemetryPublisher`: Broadcasts depth results over WebSocket
  - `TrackingTelemetryPublisher`: Broadcasts tracking/occupancy telemetry

#### 8. **REST APIs** (`noesis/server/`)
- **analytics_api.py**: Analytics configuration endpoints
  - GET/PUT/DELETE for ROI zones, lines, and exclusion areas
  - Hot-reload support for `nvdsanalytics` config
  - Persists changes to YAML and INI files
- **depth_api.py**: Depth request endpoints
  - Trigger MapAnything depth capture
  - Retrieve cached depth snapshots
  - Floorplan generation from depth data

#### 9. **Geometry & Depth** (`geometry/`)
- **homography.py**: Homography matrix calculations for image→plane projection
- **depth_source.py**: `DepthStorageManager` for async depth snapshot storage
  - Zarr-based storage with LRU eviction
  - Floorplan generation from depth maps
- **depth_publisher.py**: Depth diagnostics publishing

#### 10. **Frontend** (`oai2-fe/src/`)
- **App.tsx**: Main React component orchestrating UI panels
- **Components**:
  - `StreamPanel`: WebRTC video player with detection overlays
  - `BevPanel`: Bird's-eye view visualization with trails
  - `DepthDrawer`: MapAnything depth visualization drawer
  - `ControlsPanel`: User controls (trails, detection toggles, ROI editor)
  - `TopDownDrawer`: Top-down floorplan view
- **Hooks**:
  - `useWebSocketClient`: WebSocket connection and message handling
  - `useWebRTC`: WebRTC peer connection management
- **Libraries**:
  - `lib/trails.ts`: Motion trail rendering logic
  - `lib/calibration.ts`: Camera calibration utilities

---

## Code Organization

### Directory Structure

```
Noesis_Devel/
├── websocket_server.py              # WebSocket server implementation
├── config.py                        # Application configuration
├── requirements.txt                 # Python dependencies
│
├── noesis/                          # Core DS8 application package
│   ├── ds8_runtime.py              # DS8 runtime harness (main entry point)
│   ├── mosaic_webrtc_gateway.py    # RTSP→WebRTC passthrough gateway
│   ├── pipelines/
│   │   ├── ds8_pipeline.py         # Service Maker pipeline builder
│   │   └── hooks.py                # BatchMetadataOperator hooks
│   ├── telemetry/
│   │   ├── bev.py                  # Bird's-eye view rendering
│   │   └── publishers.py           # Telemetry publishers
│   ├── metadata/
│   │   ├── depth_result.py         # Depth result structures
│   │   └── intrinsics.py           # Camera intrinsics loader
│   ├── server/
│   │   ├── analytics_api.py        # Analytics REST endpoints
│   │   └── depth_api.py            # Depth REST endpoints
│   └── config/
│       └── adapters.py             # Configuration adapters
│
├── reid/                            # Re-identification module
│   ├── stable_id_manager.py        # Cross-camera stable ID assignment
│   └── embedding_extractor.py      # OSNet feature extraction
│
├── geometry/                        # Geometry and depth processing
│   ├── homography.py               # Homography calculations
│   ├── depth_source.py             # Depth storage manager
│   ├── depth_publisher.py          # Depth diagnostics
│   ├── floor.py                    # Floor plane estimation
│   └── transform.py                # Coordinate transforms
│
├── config/                          # Configuration files
│   ├── infer.yaml                  # DS8 pipeline configuration
│   ├── cameras.yaml                # Camera stream configurations
│   ├── nvtracker.yaml              # Tracker configuration
│   ├── nvdsanalytics.yaml          # Analytics stages configuration
│   ├── config_nvdsanalytics_post.ini   # Analytics INI
│   ├── config_nvdsanalytics_exclude.ini # ROI exclusion INI
│   └── camera_calibration.json     # Camera extrinsics
│
├── pipelines/                       # DeepStream INI configs
│   ├── config_infer_primary_yolo11_seg.ini
│   ├── config_infer_secondary_reid_osnet.ini
│   ├── config_infer_secondary_mapanything.ini
│   └── config_preproc.ini
│
├── models/                          # ML model files
│   └── engines/                    # TensorRT engine files
│
├── oai2-fe/                         # React frontend
│   ├── src/
│   │   ├── App.tsx                 # Main React component
│   │   ├── components/             # UI components
│   │   ├── hooks/                  # React hooks
│   │   ├── lib/                    # Utility libraries
│   │   └── telemetry/              # Telemetry UI
│   ├── package.json                # Node dependencies
│   └── vite.config.ts              # Vite build config
│
├── docs/                            # Documentation
│   ├── DS8_README_FOR_AGENTS.md    # DS8 orientation
│   ├── DS8_api_contracts_ws.md     # WebSocket contracts
│   ├── DS8_api_contracts_rest.md   # REST contracts
│   ├── DS8_metadata_contracts.md   # Metadata schemas
│   └── DS8_testing_guide.md        # Testing reference
│
├── plans/DS8/                       # DS8 work orders and checklists
│   ├── ds8_master_work_orders.md
│   └── ds8_design_decisions.md
│
├── tests/                           # Test files
└── legacy/                          # Deprecated archived code
```

### Naming Conventions
- **Python**: snake_case for functions/variables, PascalCase for classes
- **TypeScript**: camelCase for functions/variables, PascalCase for components/types
- **Files**: snake_case for Python, PascalCase for React components
- **Config**: kebab-case for YAML keys, UPPER_CASE for environment variables

---

## Configuration

### Pipeline Configuration (`config/infer.yaml`)

The primary DS8 pipeline is defined in `config/infer.yaml`:

```yaml
version: 1
batch_size: 3

streammux:
  width: 1920
  height: 1080
  batch-size: 3
  live-source: 1

sources:
  - element: nvurisrcbin
    uri: rtsp://camera-url
    gpu-id: 0

models:
  pgie:
    config-file-path: pipelines/config_infer_primary_yolo11_seg.ini
    engine: models/engines/yolo11s-seg.engine
  reid:
    enable: true
    config-file-path: pipelines/config_infer_secondary_reid_osnet.ini
  mapanything:
    enable: true
    config-file-path: pipelines/config_infer_secondary_mapanything.ini

tracker:
  config-file: config/nvtracker.yaml

analytics:
  enable: true
  config-file: config/config_nvdsanalytics_post.ini

mosaic_output:
  rtsp_enabled: true
  rtsp_port: 8554
  mosaic_webrtc_enabled: true

visualization:
  trails:
    enabled: true
    window_s: 8.0
    color_key: stable_id
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NOESIS_DS8_PIPELINE_CONFIG` | Pipeline YAML path | `config/infer.yaml` |
| `NOESIS_CAMERAS_CONFIG` | Cameras YAML path | `config/cameras.yaml` |
| `NOESIS_WS_HOST` / `NOESIS_WS_PORT` | WebSocket bind | `0.0.0.0:6008` |
| `NOESIS_REST_HOST` / `NOESIS_REST_PORT` | REST API bind | `0.0.0.0:8080` |
| `NOESIS_MOSAIC_RTSP_ENABLED` | Enable RTSP output | `true` |
| `NOESIS_MOSAIC_WEBRTC_ENABLED` | Enable WebRTC gateway | `true` |
| `NOESIS_REID_ENABLED` | Enable StableIDManager | `true` |
| `NOESIS_BEV_JPEG_ENABLED` | Enable BEV JPEG binaries | `false` |
| `NOESIS_DS8_FPS_PROBE` | Enable FPS debug probes | `false` |

---

## Data Flow

1. **Video Input**: RTSP streams → `nvmultiurisrcbin` → GPU decode (NVDEC)
2. **Batching**: Multiple streams → `nvstreammux` → batched NVMM tensor
3. **Preprocess**: Optional `nvdspreprocess` for ROI/tensor preparation
4. **Primary Inference**: Batched frames → `nvinfer` (YOLOv11-seg) → detections + masks
5. **Tracking**: Detections → `nvtracker` (NvDCF) → tracked objects with IDs
6. **Analytics**: Tracks → `nvdsanalytics` → events (ROI, line crossing, occupancy)
7. **ReID**: Tracked crops → `nvinfer` (OSNet SGIE) → embedding tensors → StableIDManager
8. **Depth Branch** (parallel): Frames → `valve` → `nvinfer` (MapAnything) → depth tensor
9. **Visualization**: Frames + metadata → `nvmultistreamtiler` → `nvdsosd` → overlays + trails
10. **Output**:
    - **RTSP**: `nvrtspoutsinkbin` → H.264 stream at `rtsp://host:8554/mosaic`
    - **WebRTC**: `MosaicWebRTCGateway` consumes RTSP → WebRTC to browser
    - **WebSocket**: telemetry JSON + WebRTC signaling; optional BEV JPEG binaries
11. **Metadata Extraction**: BatchMetadataOperator probes extract `NvDsBatchMeta`
12. **WebSocket Broadcast**: Tracking telemetry (JSON) → frontend
13. **BEV Rendering**: Footpoints + calibration → homography → top-down view

---

## Key Design Patterns

1. **Singleton Pattern**: Pipeline singleton (`_PIPELINE_SINGLETON`) ensures single pipeline instance
2. **Factory Pattern**: `build_pipeline()` creates pipeline from YAML configuration
3. **Probe/Operator Pattern**: `BatchMetadataOperator` + `Probe` for non-invasive metadata extraction
4. **Observer Pattern**: WebSocket server broadcasts to multiple clients; toggle callbacks
5. **Strategy Pattern**: Different depth request strategies (fresh vs cache-first)
6. **Valve Pattern**: GStreamer `valve` element for conditional branch gating
7. **Gateway Pattern**: `MosaicWebRTCGateway` bridges RTSP and WebRTC protocols

---

## Integration Points

1. **DeepStream → Python**: `BatchMetadataOperator` probes extract `NvDsBatchMeta` from pipeline
2. **Python → Frontend**: WebSocket server streams telemetry (JSON) and optional BEV JPEG binaries.
3. **Frontend → Backend**: RPC messages for calibration, depth requests, BEV config
4. **RTSP → WebRTC**: `MosaicWebRTCGateway` passthrough (no transcoding)
5. **Telemetry**: Publishers send tracking/depth data over WebSocket

---

## Performance Characteristics

- **GPU-Only Processing**: Zero CPU fallbacks in core pipeline; all operations on GPU via NVMM
- **Multi-Stream**: Supports 3+ simultaneous RTSP streams with batched inference
- **Real-time**: Sub-100ms latency from frame capture to frontend display
- **Scalable**: Batch processing via `nvmultiurisrcbin`/`nvstreammux` for efficient GPU utilization
- **Memory Efficient**: GPU memory pools, frame coalescing, Zarr-based depth storage
- **WebRTC Passthrough**: No transcode overhead for browser video delivery

---

## Development Workflow

### Running the Application

```bash
# Activate DeepStream environment
source ./activate_deepstream.sh

# Start DS8 runtime
python noesis/ds8_runtime.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --ws-port 6008 \
  --rest-port 8080
```

### Frontend Development

```bash
cd oai2-fe
npm install
npm run dev  # Development server at http://localhost:5173
```

### Key Environment Variables for Development

```bash
export NOESIS_LOG_LEVEL=DEBUG
export NOESIS_DS8_FPS_PROBE=1          # Enable FPS probes
export NOESIS_MOSAIC_RTSP_ENABLED=1    # Enable RTSP output
export NOESIS_MOSAIC_WEBRTC_ENABLED=1  # Enable WebRTC gateway
export NOESIS_BEV_JPEG_ENABLED=0       # Optional BEV JPEG binaries over WebSocket
export NOESIS_REID_ENABLED=1           # Enable ReID
```

---

## Current State

- **Stack**: DeepStream 8.0 Service Maker (`pyservicemaker`)
- **Video Delivery**: RTSP → WebRTC gateway (mosaic); WebSocket carries signaling (plus telemetry/optional BEV JPEG binaries).
- **Tracking**: NvDCF + OSNet ReID for stable cross-camera IDs
- **Depth**: MapAnything full-frame with valve gating
- **Trails**: GPU-rendered via NvDsDisplayMeta on mosaic OSD
- **APIs**: REST (FastAPI) for analytics, WebSocket for telemetry

---

This description provides a comprehensive overview of the Noesis codebase structure, DS8 architecture, and key components.
