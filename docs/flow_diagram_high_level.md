# High-Level Pipeline Flow
_Status: current as of 2026-02-02._

```mermaid
graph TD
  subgraph "Input"
    A[RTSP / File / Camera Sources]
  end

  subgraph "DeepStream Pipeline (DS8)"
    A --> S0[nvurisrcbin (per source)]
    S0 --> MUX[nvstreammux (batching)]
    MUX --> PRE[nvdspreprocess]
    PRE --> PGIE[nvinfer (PGIE YOLO11)]
    PGIE --> TEE[main_tee]
    TEE --> EX[nvdsroiexclude]
    EX --> TRK[nvtracker]
    TRK --> ANA[nvdsanalytics (post)]
    ANA --> REID[nvinfer (ReID SGIE)]
    REID --> POSE[nvinfer (Pose SGIE)]
    POSE --> TILER[nvmultistreamtiler]
    TILER --> OSD[nvdsosd]
    OSD --> OUTTEE[sink_tee]
    OUTTEE --> RTSP[nvrtspoutsinkbin]
  end

  subgraph "MapAnything Branch"
    TEE --> MAQ[mapanything_queue]
    MAQ --> MAV[mapanything_valve]
    MAV --> MASGIE[nvinfer (MapAnything SGIE)]
    MASGIE --> MASINK[fakesink]
  end

  subgraph "Python Application"
    ANA -. telemetry .-> WS[WebSocket Server]
    POSE -. object meta .-> META[NOESIS.POSE_FEATURES user meta]
    TILER -. display meta .-> KP[Pose keypoint overlay]
  end

  WS --> CLIENTS[Web Clients]
```
