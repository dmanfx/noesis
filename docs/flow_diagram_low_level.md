# Low-Level Pipeline Flow (Detailed)
_Status: current as of 2026-02-02._

```mermaid
flowchart LR
  %% Elements
  SRC[nvurisrcbin (per source)]
  MUX[nvstreammux<br/>batching]
  PRE[nvdspreprocess<br/>config: pipelines/config_preproc.ini]
  PGIE[nvinfer (PGIE YOLO11)]
  TEE[main_tee]
  ROIX[nvdsroiexclude<br/>config: config/config_nvdsanalytics_exclude.ini]
  TRK[nvtracker<br/>lib: libnvds_nvmultiobjecttracker.so<br/>conf: config/nvtracker.yaml]
  ANA[nvdsanalytics (post)<br/>config: config/config_nvdsanalytics_post.ini]
  REID[nvinfer (ReID SGIE)]
  POSE[nvinfer (Pose SGIE)]
  TILER[nvmultistreamtiler]
  OSD[nvdsosd]
  OUTTEE[sink_tee]
  RTSPQ[rtsp_queue] --> RTSPV[nvvideoconvert] --> RTSP[nvrtspoutsinkbin]
  BEV[bev_sink (fakesink)]

  %% Flow
  SRC --> MUX --> PRE --> PGIE --> TEE
  TEE --> ROIX --> TRK --> ANA --> REID --> POSE --> TILER --> OSD --> OUTTEE
  OUTTEE --> RTSPQ
  OUTTEE --> BEV

  %% MapAnything branch
  TEE --> MAQ[mapanything_queue] --> MAV[mapanything_valve] --> MASGIE[nvinfer (MapAnything SGIE)] --> MASINK[fakesink]

  %% Hooks / metadata
  ANA -. telemetry hook .-> TELE[_AnalyticsTelemetryProcessor]
  REID -. tensor meta .-> SID[StableIDManager embeddings]
  POSE -. tensor meta .-> PF[PoseFeatureProcessor → NOESIS.POSE_FEATURES (object user meta)]
  TILER -. display meta .-> KP[PoseKeypointOverlayProcessor]
  MASGIE -. tensor meta .-> MA[MapAnythingProcessor (depth/conf/mask)]
```
