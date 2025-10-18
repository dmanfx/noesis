# Low-Level Pipeline Flow (Detailed)

```mermaid
flowchart LR
  %% Elements
  SRC[nvmultiurisrcbin<br/>uri-list, sensor-id-list, batch, width/height, REST:9000]
  PRE[nvdspreprocess<br/>config: pipelines/config_preproc.ini]
  PGIE[nvinfer<br/>config: pipelines/config_infer_primary_yolo11.ini<br/>input-tensor-meta: true]
  EXA[nvdsanalytics_exclude<br/>uid:101<br/>config: pipelines/config_nvdsanalytics_exclude.ini]
  MAP_PRE[nvdspreprocess<br/>config: pipelines/config_preprocess_mapanything_fused.ini
           custom-lib: libmapanything_preprocess_fused.so]
  SGIE[nvinfer<br/>uid:22<br/>config: pipelines/config_infer_secondary_mapanything_fused.ini]
  TRK[nvtracker<br/>lib: libnvds_nvmultiobjecttracker.so<br/>conf: pipelines/config_tracker_nvdcf_batch.yml]
  ANA[nvdsanalytics_post<br/>uid:201<br/>config: pipelines/config_nvdsanalytics_post.ini]
  DMX[nvstreamdemux]

  %% Flow
  SRC --> PRE --> PGIE --> EXA --> MAP_PRE --> SGIE --> TRK --> ANA --> DMX

  %% Probes
  EXA -- src pad probe --> P1[_remove_excluded_objects_probe]
  SGIE -- src pad probe --> P4[_mapanything_depth_probe]
  ANA -- src pad probe --> P2[_analytics_probe]
  %% Debug probe on demux sink
  DMX -- sink pad probe --> P0[demux debug]

  %% Demux signals
  DMX -- pad-added --> S1[_demux_new_pad_cb]
  DMX -- pad-removed --> S2[_demux_pad_removed_cb]

  %% Dynamic branch per pad
  subgraph Per-Stream Branch (per DMX pad)
    Q[queue<br/>leaky=downstream, max=3] --> CVC1[nvvideoconvert (pre)]
    CVC1 --> CAPS1[capsfilter<br/>video/x-raw(memory:NVMM), format=RGBA]
    CAPS1 --> OSD[nvdsosd<br/>process-mode: GPU<br/>display-text: 1]
    OSD -- sink pad probe --> P3[_per_branch_osd_probe(sensor_id)]
    OSD --> CVC2[nvvideoconvert (post)]
    CVC2 --> CAPS2[capsfilter<br/>video/x-raw(memory:NVMM), format=I420]
    CAPS2 --> ENC[nvjpegenc]
    ENC --> SINK[appsink<br/>emit-signals, drop, max-buffers=5]
    SINK -- new-sample --> CB[_on_new_jpeg_sample(sensor_id)]
  end

  %% Branch hookup
  S1 -. create & link .-> Q
  DMX -. link pad to queue .-> Q
  Q -. sync_state_with_parent .-> CVC1
  CVC1 -. sync_state_with_parent .-> CAPS1
  CAPS1 -. sync_state_with_parent .-> OSD
  OSD -. sync_state_with_parent .-> CVC2
  CVC2 -. sync_state_with_parent .-> CAPS2
  CAPS2 -. sync_state_with_parent .-> ENC
  ENC -. sync_state_with_parent .-> SINK

  %% Pad->source mapping
  Q -. one-shot probe (BUFFER) .-> MAP[learn frame_meta.source_id (0-based)]
  MAP -. map to configured .-> M0[source_idx to sensor_id]

  %% Teardown
  S2 -. set NULL & remove .-> Q
  S2 -. clear mapping .-> M0

  %% WebSocket data paths
  P2 -. telemetry .-> WS[WebSocketServer.broadcast_frame]
  P4 -. depth meta .-> WS
  SINK -. JPEG bytes .-> QQ[jpeg_queues[source_id]]
  QQ -. periodic .-> W2[WebSocketServer.broadcast_sync {type:'video_frame'}]

  %% Runtime controls
  subgraph Runtime Config
    R1[update_detection_config]
    R2[update_detection_toggle]
    R3[set_trail_visualization]
  end
  R1 -. set properties .-> PGIE
  R2 -. adjust custom-lib-props .-> PGIE
  R3 -. toggle trails .-> P3

  %% REST sensors
  subgraph Dynamic Sensors
    ADD[add_sensor(sensor_id, uri)]
    REM[remove_sensor(sensor_id)]
  end
  ADD -. POST /stream {add} .-> SRC
  REM -. POST /stream {remove} .-> SRC
```
