# Noesis Codebase Description
_Status: current as of 2026-08-08._

## Project Overview

**Noesis** is a high-performance, GPU-first real-time video analytics application built on **NVIDIA DeepStream 8.0 Service Maker**. The system processes multiple RTSP camera streams end-to-end on the GPU for decode, preprocess, inference, tracking, analytics, tiling, and OSD, with CPU used only at the metadata/serialization edge and for the minimal per-object depth-fusion boundary that DeepStream does not expose as a pure GPU contract.

### Purpose
Real-time multi-camera video analytics pipeline for:
- Runtime-selectable object detection and instance segmentation, with YOLO26 segmentation as the active baseline tracking path
- Multi-object tracking with cross-camera re-identification
- Spatial analytics (ROI filtering, line crossing, occupancy)
- Bird's-eye view (BEV) visualization with trail rendering
- Split monocular depth processing:
  - MapAnything full-frame depth/floorplan/RPC reference lane
  - Depth Anything V2 baseline tracking-depth lane fused into `track.world`
- Real-time WebRTC streaming to browser frontends

### Key Features
- **GPU-First Core Pipeline**: Decode, preprocess, inference, tracking, analytics, tiling, and OSD stay on GPU via NVMM surfaces; CPU is used only for metadata extraction, serialization, and the minimal per-object depth fusion boundary
- **Multi-Stream Support**: Processes multiple RTSP camera streams simultaneously via `nvmultiurisrcbin`
- **Primary Detection Profiles**: Runtime-selectable PGIE path, including YOLO26 segmentation in active baseline tracking work
- **Advanced Tracking**: NVIDIA NvDCF tracker with ReID-based re-association (occlusion recovery) plus Swin-Tiny ReID (TAO ReIdentificationNet Transformer) for stable cross-camera IDs
- **Pose-assisted StableID**: YOLO26 pose SGIE ratio features can be fused into StableID as a secondary signal (bounded in RAM; no disk persistence)
- **Analytics**: ROI filtering, line crossing, direction detection, overcrowding via `nvdsanalytics`
- **Bird's-Eye View (BEV)**: Real-time top-down visualization from canonical backend `track.world`; world-mode no longer runs a second BEV smoother over already-filtered world positions
- **Split Depth Architecture**:
  - MapAnything: gated full-frame GPU depth/RPC/floorplan branch
  - DAv2: always-on baseline tracking depth fused into world estimation
  - Offline DAv2->MapAnything registration artifact aligns room-relative range before projection
- **Real-time WebRTC Streaming**: one GPU H.264 encode, AU-aligned SHM fanout, and one RTP packetization per browser peer
- **Motion Trails**: GPU-rendered persistent trails behind tracked objects in the mosaic OSD

### Technology Stack
- **Backend**: Python 3.10+, NVIDIA DeepStream 8.0 Service Maker (`pyservicemaker`), TensorRT
- **Frontend**: React + TypeScript, Vite, WebSocket/WebRTC client
- **ML Models**: Runtime-selectable PGIE profiles (commonly YOLO26-seg in current baseline work), YOLO26 pose SGIE, Swin-Tiny ReID (TAO ReIdentificationNet Transformer), Depth Anything V2 metric, MapAnything (Meta Research), optional RF-DETR (`--pgie-profile`)
- **GPU Libraries**: CUDA, cuDNN, TensorRT, `pyds` DeepStream Python bindings
- **Communication**: WebSockets (JSON telemetry and WebRTC signaling), WebRTC
  (H.264 video)

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
        │ JSON telemetry + WebRTC signaling              │
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
│  │  - Attaches metadata hooks (analytics, ReID, depth, fused world)   │ │
│  └───────────────────┬────────────────────────────────────────────────┘ │
│                      │                                                  │
│  ┌───────────────────▼───────────────────┬────────────────────────────┐ │
│  │ WebSocketServer                       │ MosaicWebRTCGateway        │ │
│  │ (websocket_server.py)                 │ (mosaic_webrtc_gateway.py) │ │
│  │  - Client management                  │  - H.264 AU→WebRTC delivery│ │
│  │  - Metadata broadcast                 │  - Browser signaling       │ │
│  │  - RPC handlers                       │  - No transcode            │ │
│  └───────────────────────────────────────┴────────────────────────────┘ │
│                      │                                                  │
│  ┌───────────────────▼────────────────────────────────────────────────┐ │
│  │           DS8 Pipeline (noesis/pipelines/ds8_pipeline.py)          │ │
│  │  - pyservicemaker.Pipeline graph construction from YAML            │ │
│  │  - Component wiring (sources → inference → tracking → output)      │ │
│  │  - Valve-based MapAnything gating                                  │ │
│  │  - Always-on DAv2 tracking-depth lane                              │ │
│  └───────────────────┬────────────────────────────────────────────────┘ │
│                      │                                                  │
│  ┌───────────────────▼────────────────────────────────────────────────┐ │
│  │        Metadata Hooks (noesis/pipelines/hooks.py)                  │ │
│  │  - BatchMetadataOperator probes for telemetry extraction           │ │
│  │  - Trail overlay rendering via NvDsDisplayMeta                     │ │
│  │  - MapAnything tensor postprocess                                  │ │
│  │  - DAv2 object-depth fusion + pose+depth world estimation          │ │
│  │  - Swin ReID embedding extraction for StableIDManager             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│              DeepStream 8 Service Maker Runtime (GPU)                    │
│                                                                          │
│  source nodes ──► nvstreammux ──► [nvdspreprocess] ──► nvinfer(PGIE)     │
│                                                              │           │
│                               ┌───────────────────────────── tee ───────┐│
│                               │                    │                    ││
│                               ▼                    ▼                    ▼│
│                    [nvdsroiexclude]         depth_tracking_queue   mapanything_queue
│                               │                    │                    ││
│                               ▼                    ▼                    ▼│
│                         nvtracker         depth_tracking_fullframe     valve│
│                               │                    │                    ││
│                               ▼                    ▼                    ▼│
│                       [nvdsanalytics]   depth_tracking_fullframe_sink mapanything_fullframe
│                               │                                         ││
│                               ▼                                         ▼│
│                   [nvinfer(ReID SGIE)]                 mapanything_fullframe_sink
│                               │                                          │
│                               ▼                                          │
│                   [nvinfer(Pose SGIE)]                                   │
│                               │                                          │
│                               ▼                                          │
│                 world_observation_stage                                  │
│                               │                                          │
│                               ▼                                          │
│                 tracking_telemetry_stage                                 │
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
│                    mosaic_encode_queue                    other sinks   │
│                               │                              / placeholders│
│                               ▼                                          │
│                   nvvideoconvert (NVMM)                                 │
│                               │                                          │
│                               ▼                                          │
│                  noesisforceidr → NVENC                                 │
│                               │                                          │
│                               ▼                                          │
│                 h264parse → shmsink (AUs)                               │
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
  - **Primary Inference**: `nvinfer` with runtime-materialized PGIE profile (commonly YOLO26-seg in current baseline work)
  - **Tee**: Splits flow to tracker chain, always-on DAv2 tracking-depth branch, and gated MapAnything branch
  - **Exclude Stage**: repo-owned `nvdsroiexclude` pruning before tracking;
    durable YAML is rendered to a bounded native INI and exact source coverage
    is required
  - **Tracker**: `nvtracker` with NvDCF multi-object tracking
  - **Analytics**: optional `nvdsanalytics` for ROI/line crossing events
  - **ReID SGIE**: `nvinfer` with Swin-Tiny ReID model for cross-camera re-identification
  - **Pose SGIE**: `nvinfer` with YOLO26 pose for pose-first anchor authority
  - **World Observation**: backend fused pose+depth world estimation
  - **Tracking Telemetry Stage**: canonical `track.world` publishing before BEV/OSD consumers
  - **Tiler**: `nvmultistreamtiler` creates mosaic view
  - **OSD**: `nvdsosd` overlays bounding boxes, masks, labels, and trails
  - **Output Sinks**: `sink_tee` feeds the GPU mosaic encoder and AU-aligned `shmsink`; optional RTSP tooling can tee the already encoded AUs. Mosaic JPEG/WebSocket output is removed in DS8.
- **Depth Ownership**:
  - MapAnything remains valve-gated for RPC/full-frame depth work
  - DAv2 remains always on for baseline tracking
  - a read-only depth-registration artifact aligns DAv2 range into MapAnything/room space before the fused world update

#### 2. **Metadata Hooks** (`noesis/pipelines/hooks.py`)
- **BatchMetadataOperator Probes**: Attach to pipeline nodes for per-frame processing
- **Hook Types**:
  - `attach_intrinsics_hook`: Extracts camera intrinsics per frame
  - `attach_mapanything_postprocess_hook`: Decodes depth tensor meta from MapAnything SGIE
  - `attach_pose_feature_hook`: Parses YOLO26 pose SGIE tensors and attaches pose ratio user meta (for StableID)
  - `attach_analytics_telemetry_hook`: Extracts tracks, events, and occupancy data
  - `attach_trail_overlay_hook`: Renders motion trails via NvDsDisplayMeta lines
  - `attach_osd_label_hook`: Customizes OSD text labels (stable ID, confidence)
  - `attach_analytics_reload_bridge`: Commits analytics updates to the native
    pre-tracker element and requires its exact hash/sequence receipt
- **Exclusion ownership**: `nvdsroiexclude` is the sole pruning path; no Python
  metadata hook or fallback removes exclusion objects
- **TrailOverlayProcessor**: GPU-rendered per-person motion trails with configurable styling
- **MapAnythingProcessor**: Tensor-to-depth conversion with async storage
- **Baseline depth/world path**: consumes `NOESIS.OBJECT_DEPTH`, applies the offline DAv2->MapAnything registration artifact inside the pose-ray depth observation, and writes canonical `track.world`

#### 3. **DS8 Runtime Harness** (`noesis/ds8_runtime.py`)
- **Entry Point**: `main()` function orchestrating the application lifecycle
- **Responsibilities**:
  - Parse CLI arguments and load YAML configuration
  - Build DS8 pipeline from `config/infer.yaml`
  - Attach all metadata hooks (intrinsics, analytics, depth, trails, ReID)
  - Start WebSocket server and REST API
  - Activate pipeline via `pyservicemaker.Pipeline.activate()`
  - Manage SHM feeder readiness and bounded WebRTC gateway startup
  - Handle graceful shutdown on SIGINT/SIGTERM
- **CalibrationProvider**: Provides camera intrinsics/extrinsics for BEV rendering
- **DepthRegistrationManager**: Loads and validates the per-camera DAv2->MapAnything registration artifact before baseline startup

#### 4. **WebSocket Server** (`websocket_server.py`)
- **WebSocketServer**: Async WebSocket server for real-time client communication
- **Features**:
  - JSON metadata streaming (detections, tracks, analytics, BEV points)
  - RPC handlers: calibration, depth requests, floorplan generation, BEV config
  - WebRTC signaling relay for `MosaicWebRTCGateway`
  - Toggle callbacks for trail visualization state
  - Rate limiting and telemetry tracking

#### 5. **WebRTC Gateway** (`noesis/mosaic_webrtc_gateway.py`)
- **MosaicH264ShmFeeder** (`noesis/mosaic_h264_bridge.py`): one
  `shmsrc → h264parse → appsink` reader that fans complete encoded AUs to peers
- **MosaicWebRTCGateway**: Ultra-light GStreamer WebRTC gateway
- **Per-peer pipeline**: `appsrc → rtph264pay → bounded queue → webrtcbin`
- **Zero Transcode**: H.264 remains encoded after NVENC; the SHM, RTP, and
  WebRTC bitstream edge is CPU transport rather than GPU video processing
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
- **EmbeddingExtractor**: torchreid-based feature extraction (optional; DS8 uses SGIE tensors)

#### 7. **Telemetry & BEV** (`noesis/telemetry/`)
- **bev.py**: Bird's-eye view rendering using homography transforms
  - `BevRenderer`: Renders top-down view from camera detections
  - `HomographyCache`: Caches homography matrices per calibration
  - `BevTrailConfig`: Configurable trail rendering for BEV canvas
  - World mode (`menon_scene`): publishes ground-plane footpoints in scene units (XZ); backend smoothing is disabled and the dashboard owns smoothing/jitter control
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
- **depth_publisher.py**: Dormant, explicitly enabled MQTT/Influx depth
  diagnostics publisher with owner-only credential-file loading; no active
  DS8/DS9 constructor call

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
│   ├── depth_tracking_materialization.py # DAv2 depth-tracking asset materialization + native guardrails
│   ├── mosaic_h264_bridge.py       # Single SHM H.264 AU reader/fanout
│   ├── mosaic_glib_context.py      # Shared GLib default-context driver
│   ├── mosaic_webrtc_gateway.py    # Per-peer AU→RTP→WebRTC gateway
│   ├── calibration/
│   │   └── depth_registration.py   # DAv2->MapAnything registration contracts/loader
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
│   └── embedding_extractor.py      # torchreid feature extraction (unused in DS8)
│
├── geometry/                        # Geometry and depth processing
│   ├── homography.py               # Homography calculations
│   ├── depth_source.py             # Depth storage manager
│   ├── depth_publisher.py          # Dormant secure depth diagnostics publisher
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
│   ├── config_infer_secondary_depth_tracking_da2.template.ini
│   ├── config_infer_secondary_reid_swin.ini
│   ├── config_infer_secondary_mapanything.ini
│   └── config_preproc.ini
│
├── models/                          # ML model files
│   └── engines/                    # TensorRT engine files
│
├── native/                          # Native metadata/tensor extraction bridges
│   ├── noesis_depth_meta_ext.cpp   # NOESIS.OBJECT_DEPTH bridge
│   └── noesis_depth_tracking_tensor_ext.cpp # Baseline DAv2 tensor extractor/alignment bridge
│
├── scripts/
│   └── build_depth_registration.py # Offline DAv2->MapAnything registration builder
│
├── services/
│   └── mapanything_svc/
│       └── server.py               # MapAnything service used for RPC/reference depth and registration builds
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

The primary DS8 pipeline is defined in `config/infer.yaml`. The excerpt below is intentionally minimal and shows only the parts that matter to the current baseline topology:

```yaml
version: 1
depth_registration:
  path: config/depth_registration.json
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
    gie_id: 1
  pose:
    enable: true
    config-file-path: pipelines/config_infer_secondary_yolo26_pose.ini
    gie_id: 4
  depth_tracking:
    enable: true
    name: depth_tracking_fullframe
    config-file-path: build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini
    gie_id: 5
  mapanything:
    enable: true
    name: mapanything_fullframe
    config-file-path: pipelines/config_infer_secondary_mapanything.ini
    gie_id: 2

tracker:
  config-file: config/nvtracker.yaml

analytics:
  enable: true
  config-file: config/config_nvdsanalytics_post.ini

mosaic_output:
  rtsp_enabled: false
  rtsp_port: 8554
  mosaic_webrtc_enabled: true
  mosaic_h264_shm_socket: /tmp/noesis-mosaic-h264
  video_bitrate_kbps: 12000
  h264_iframeinterval: 10
  h264_idrinterval: 10
  encoder: nvv4l2h264enc

visualization:
  trails:
    enabled: true
    anchor_mode: floor_plane_gravity_drop
    gap_predict_ttl_s: 0.0
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NOESIS_DS8_PIPELINE_CONFIG` | Pipeline YAML path | `config/infer.yaml` |
| `NOESIS_CAMERAS_CONFIG` | Cameras YAML path | `config/cameras.yaml` |
| `NOESIS_WS_HOST` / `NOESIS_WS_PORT` | WebSocket bind | `0.0.0.0:6008` |
| `NOESIS_REST_HOST` / `NOESIS_REST_PORT` | REST API bind | `0.0.0.0:8080` |
| `NOESIS_MOSAIC_RTSP_ENABLED` | Enable optional RTSP tooling output | config (`false`) |
| `NOESIS_MOSAIC_WEBRTC_ENABLED` | Enable H.264 SHM output and WebRTC gateways | config (`true`) |
| `NOESIS_MOSAIC_H264_SHM` | Override the H.264 SHM socket path | config (`/tmp/noesis-mosaic-h264`) |
| `NOESIS_REID_ENABLED` | Enable StableIDManager | `true` |
| `NOESIS_BEV_JPEG_ENABLED` | Ignored; BEV JPEG binaries are retired | n/a |
| `NOESIS_DS8_FPS_PROBE` | Enable FPS debug probes | `false` |

---

## Data Flow

1. **Video Input**: RTSP streams → `nvmultiurisrcbin` → GPU decode (NVDEC)
2. **Batching**: Multiple streams → `nvstreammux` → batched NVMM tensor
3. **Preprocess**: Optional `nvdspreprocess` for ROI/tensor preparation
4. **Primary Inference**: Batched frames → runtime-selected `nvinfer` PGIE → detections + masks
5. **Tracking**: Detections → `nvtracker` (NvDCF) → tracked objects with IDs
6. **Analytics**: Tracks → `nvdsanalytics` → events (ROI, line crossing, occupancy)
7. **ReID**: Tracked crops → `nvinfer` (Swin ReID SGIE) → embedding tensors → StableIDManager
8. **Pose SGIE**: Tracked persons → `nvinfer` (YOLO26 pose SGIE) → pose keypoints/ratios for pose-first anchoring
9. **Always-On Tracking Depth Branch**: PGIE tee → `depth_tracking_queue` → `nvinfer` (DAv2) → native tensor extraction/alignment → `NOESIS.OBJECT_DEPTH`
10. **Gated Reference Depth Branch**: PGIE tee → `mapanything_queue` → `valve` → `nvinfer` (MapAnything) → full-frame depth / floorplan / RPC path
11. **World Estimation**: Pose anchor + floor observation + registered DAv2 depth observation → fused backend `track.world`
12. **Visualization**: Frames + canonical tracking telemetry → `nvmultistreamtiler` → `nvdsosd` → overlays + trails
13. **Output**:
    - **Encode**: `sink_tee` → leaky raw queue → NVMM conversion → `noesisforceidr` → `nvv4l2h264enc` → `h264parse`
    - **WebRTC**: non-leaky AU queue → `shmsink` → one `MosaicH264ShmFeeder` → bounded per-peer `appsrc` → `rtph264pay` → `webrtcbin`
    - **Optional RTSP tooling**: a non-leaky post-encode queue may tee the same H.264 AUs to `nvrtspoutsinkbin`; it is off by default and is not consumed by WebRTC
    - **WebSocket**: telemetry JSON + WebRTC signaling
14. **Metadata Extraction**: `BatchMetadataOperator` probes extract `NvDsBatchMeta` and publish canonical track/depth telemetry
15. **BEV Rendering**: Backend-owned `track.world` → BEV/Three.js/world-mode consumers without a second world-space smoother

---

## Key Design Patterns

1. **Singleton Pattern**: Pipeline singleton (`_PIPELINE_SINGLETON`) ensures single pipeline instance
2. **Factory Pattern**: `build_pipeline()` creates pipeline from YAML configuration
3. **Probe/Operator Pattern**: `BatchMetadataOperator` + `Probe` for non-invasive metadata extraction
4. **Observer Pattern**: WebSocket server broadcasts to multiple clients; toggle callbacks
5. **Registration Pattern**: Offline DAv2->MapAnything artifact aligns room-relative range before runtime world projection
6. **Valve Pattern**: GStreamer `valve` element for conditional MapAnything branch gating
7. **Gateway Pattern**: one SHM AU feeder isolates the Service Maker graph from bounded per-peer `MosaicWebRTCGateway` lifecycles

---

## Integration Points

1. **DeepStream → Python**: `BatchMetadataOperator` probes extract `NvDsBatchMeta` from the canonical DS8 graph
2. **Native → Python**: `noesis_depth_tracking_tensor_ext` and `noesis_depth_meta_ext` bridge baseline DAv2 tensors and object-depth user meta into Python-visible contracts
3. **Offline Registration Build**: `scripts/build_depth_registration.py` pairs DAv2 and MapAnything depth on matching frames to produce `config/depth_registration.json`
4. **Python → Frontend**: WebSocket server streams telemetry JSON and WebRTC signaling
5. **Frontend → Backend**: RPC messages for calibration, depth requests, floorplan generation, and BEV config
6. **Encoded Mosaic → WebRTC**: SHM AU fanout plus per-peer RTP packetization (no transcoding)

---

## Performance Characteristics

- **GPU-First Processing**: Decode, preprocess, inference, tracking, analytics, tiling, and OSD stay on GPU; CPU is limited to metadata extraction, serialization, and the minimal per-object depth-fusion boundary
- **Multi-Stream**: Supports 3+ simultaneous RTSP streams with batched inference
- **Real-time**: Sub-100ms latency from frame capture to frontend display
- **Scalable**: Batch processing via `nvmultiurisrcbin`/`nvstreammux` for efficient GPU utilization
- **Memory Efficient**: GPU memory pools, frame coalescing, and artifact-based registration keep runtime state bounded
- **WebRTC Passthrough**: No transcode overhead for browser video delivery

---

## Development Workflow

### Running the Application

```bash
# Activate DeepStream environment
source ./activate_deepstream.sh

# Start DS8 runtime
python noesis/ds8_runtime.py \
  --pgie-profile yolo26_seg \
  --size s
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
export NOESIS_MOSAIC_RTSP_ENABLED=0    # Keep optional RTSP tooling off
export NOESIS_MOSAIC_WEBRTC_ENABLED=1  # Enable WebRTC gateway
# NOESIS_BEV_JPEG_ENABLED is ignored; BEV JPEG binaries are retired.
export NOESIS_REID_ENABLED=1           # Enable ReID
```

---

## Current State

- **Stack**: DeepStream 8.0 Service Maker (`pyservicemaker`)
- **Video Delivery**: H.264 AU SHM → WebRTC gateways (mosaic); WebSocket carries
  telemetry JSON and signaling.
- **Tracking**: NvDCF (ReID re-association) + Swin ReID for stable cross-camera IDs
- **Depth**:
  - baseline DAv2 lane is always on and contributes to fused `track.world`
  - MapAnything stays valve-gated for full-frame RPC/floorplan/reference work
  - DAv2 depth is room-registered through the offline `config/depth_registration.json` artifact before the pose-ray world update
- **Trails**: GPU-rendered via NvDsDisplayMeta on mosaic OSD
- **APIs**: REST (FastAPI) for analytics, WebSocket for telemetry

---

This description provides a comprehensive overview of the Noesis codebase structure, DS8 architecture, and key components.
