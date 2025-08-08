# Low-Level Pipeline Flow (Detailed)

```mermaid
graph LR
  %% Elements
  SRC[nvmultiurisrcbin\nuri-list, sensor-id-list, batch, width/height, REST:9000]
  PRE[nvdspreprocess\nconfig: pipelines/config_preproc.ini]
  PGIE[nvinfer\nconfig: pipelines/config_infer_primary_yolo11.ini\ninput-tensor-meta: true]
  EXA[nvdsanalytics_exclude\nuid:101\nconfig: pipelines/config_nvdsanalytics_exclude.ini]
  TRK[nvtracker\nlib: libnvds_nvmultiobjecttracker.so\nconf: pipelines/config_tracker_nvdcf_batch.yml]
  ANA[nvdsanalytics_post\nuid:201\nconfig: pipelines/config_nvdsanalytics_post.ini]
  DMX[nvstreamdemux]

  %% Flow
  SRC --> PRE --> PGIE --> EXA --> TRK --> ANA --> DMX

  %% Probes
  EXA -- src pad probe --> P1[_remove_excluded_objects_probe]
  ANA -- src pad probe --> P2[_analytics_probe]
  %% OSD probes are now per-branch
  SRC -- src pad probe --> P0[debug trace]

  %% Demux signals
  DMX -- pad-added --> S1[_demux_new_pad_cb]
  DMX -- pad-removed --> S2[_demux_pad_removed_cb]

  %% Dynamic branch per pad
  subgraph Branch[Per-Stream Branch (per DMX pad)]
    Q[queue\nleaky downstream, max 3] --> CVC1[nvvideoconvert (pre)]
    CVC1 --> CAPS1[capsfilter\nvideo/x-raw(memory:NVMM), format=RGBA]
    CAPS1 --> OSD[nvdsosd\nprocess-mode: GPU\ndisplay-text: 1]
    OSD -- sink pad probe --> P3[_osd_sink_pad_buffer_probe(sensor_id)]
    OSD --> CVC2[nvvideoconvert (post)]
    CVC2 --> CAPS2[capsfilter\nvideo/x-raw(memory:NVMM), format=I420]
    CAPS2 --> ENC[nvjpegenc]
    ENC --> SINK[appsink\nemit-signals, drop, max-buffers=5]
    SINK -- new-sample --> CB[_on_new_jpeg_sample(sensor_id)]
  end

  %% Branch hookup
  S1 -. create & link .-> Q
  DMX -. link pad->queue .-> Q
  Q -. sync_state_with_parent .-> CVC
  CVC -. sync_state_with_parent .-> CAPS
  CAPS -. sync_state_with_parent .-> ENC
  ENC -. sync_state_with_parent .-> SINK

  %% Pad->source mapping
  Q -. one-shot probe (BUFFER) .-> MAP[learn frame_meta.source_id (0-based)]
  MAP -. map to configured .-> M0[source_idx→sensor_id]

  %% Teardown
  S2 -. set NULL & remove .-> Q
  S2 -. clear mapping .-> M0

  %% WebSocket data paths
  P2 -. telemetry .-> WS[WebSocketServer.broadcast_frame]
  SINK -. JPEG bytes .-> QQ[jpeg_queues[source_id]]
  QQ -. periodic .-> W2[WebSocketServer.broadcast_sync {type:"video_frame"}]

  %% Runtime controls
  subgraph Runtime[Runtime Config]
    R1[update_detection_config]
    R2[update_detection_toggle]
    R3[set_trail_visualization]
  end
  R1 -. set properties .-> PGIE
  R2 -. adjust custom-lib-props .-> PGIE
  R3 -. toggle trails .-> P3

  %% REST sensors
  subgraph Sensors[Dynamic Sensors]
    ADD[add_sensor(sensor_id, uri)]
    REM[remove_sensor(sensor_id)]
  end
  ADD -. POST /stream {add} .-> SRC
  REM -. POST /stream {remove} .-> SRC
```