# High-Level Pipeline Flow

```mermaid
graph TD
  subgraph "Input"
    A[RTSP / File / Camera Sources]
  end

  subgraph "DeepStream Pipeline"
    A --> B[nvmultiurisrcbin\n(batching, decode)]
    B --> C[nvdspreprocess\n(primary)]
    C --> D[nvinfer (YOLO11 PGIE)]
    D --> E[nvdsanalytics (exclude)]
    E --> F[nvdspreprocess\n(MapAnything fused)]
    F --> G[nvinfer (MapAnything SGIE, UID 22)]
    G --> I[nvtracker]
    I --> J[nvdsanalytics (post)]
    J --> H[nvstreamdemux]
  end

  subgraph "Python Application"
    H --> L[Dynamic JPEG Branches\nqueue → nvvideoconvert → caps(RGBA, NVMM) → nvdsosd → nvvideoconvert → caps(I420, NVMM) → nvjpegenc → appsink]
    L --> M[WebSocket Server]
    J -. telemetry .-> N[Analytics Probe\n(_analytics_probe)]
    N --> M
    G -. depth meta .-> O[MapAnything Depth Probe\n(_mapanything_depth_probe)]
    O --> M
  end

  M --> P[Web Clients]
```
