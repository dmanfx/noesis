# High-Level Pipeline Flow

```mermaid
graph TD
  subgraph "Input"
    A[RTSP / File / Camera Sources]
  end

  subgraph "DeepStream Pipeline"
    A --> B[nvmultiurisrcbin\n(batching, decode)]
    B --> C[nvdspreprocess]
    C --> D[nvinfer (YOLO11)]
    D --> E[nvtracker]
    E --> F[nvdsanalytics (post)]
    F --> G[nvdsosd]
    G --> H[nvstreamdemux]
  end

  subgraph "Python Application"
    H --> I[Dynamic JPEG Branches\nqueue → nvvideoconvert → caps(NV12, NVMM) → nvjpegenc → appsink]
    I --> J[WebSocket Server]
    F -. telemetry .-> K[Analytics Probe\n(_analytics_probe)]
    K --> J
  end

  J --> L[Web Clients]
```