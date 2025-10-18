# DeepStream Elements and Probes

This document summarizes the elements created in `deepstream_video_pipeline.py`, their key properties, and the probes/signals attached.

## Elements

- nvmultiurisrcbin (multiurisrc)
  - Properties: `uri-list`, `sensor-id-list`, `max-batch-size`, `width`, `height`, `batched-push-timeout=-1`, `live-source=1`, `drop-pipeline-eos=1`, `rtsp-reconnect-interval=30`, `port`, `ip-address="localhost"`

- nvdspreprocess (preprocess)
  - Properties: `config-file = config.processing.DEEPSTREAM_PREPROCESS_CONFIG`

- nvinfer (nvinfer)
  - Properties: `config-file-path = pipelines/config_infer_primary_yolo11.ini`, `input-tensor-meta=True`
  - Runtime updates (via Python): `confidence-threshold`, `iou-threshold`, `enable`, `custom-lib-props="target-classes:<ids>"`

- nvdspreprocess (mapanything_preprocess)
  - Properties: `config-file = pipelines/config_preprocess_mapanything_fused.ini`
  - Custom library resolves to `pipelines/mapanything_preprocess_fused/libmapanything_preprocess_fused.so`
  - Emits fused tensor meta (`mapanything_fused`, FP16) for the SGIE

- nvinfer (mapanything_sgie_fused)
  - Properties: `config-file-path = pipelines/config_infer_secondary_mapanything_fused.ini`, `input-tensor-meta=True`, `unique-id=22`
  - Loads TensorRT engine `models/engines/ma_model_fp16_b3_fused.plan`
  - Probe: src pad → `_mapanything_depth_probe` (decodes depth/conf tensors, attaches NVDS user meta)

- nvdsanalytics (nvdsanalytics_exclude)
  - Properties: `unique-id=101`, `config-file=pipelines/config_nvdsanalytics_exclude.ini`
  - Probe: src pad → `_remove_excluded_objects_probe()` (drops objects in exclusion ROIs)

- nvtracker (nvtracker)
  - Properties: `ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`, `ll-config-file=pipelines/config_tracker_nvdcf_batch.yml`

- nvdsanalytics (nvdsanalytics_post)
  - Properties: `unique-id=201`, `config-file=pipelines/config_nvdsanalytics_post.ini`
  - Probe: src pad → `_analytics_probe()` (builds tracking/occupancy/transitions telemetry)

- nvstreamdemux (demux)
  - Pads: Request pads `src_%u` acquired once via helper and reused across calibration/branches
  - Calibration: one-shot probe maps pad → `frame_meta.source_id` (0-based) → configured `sensor_id`
  - Probe: sink pad → demux debug probe (logs present source_ids per batch)

- Per-branch nvdsosd (one per stream)
  - Properties: `process-mode=0` (GPU), `display-text=1`
  - Probe: sink pad → `_per_branch_osd_probe(sensor_id)` draws overlays only for that stream

## Per-Stream Branch (per demux pad)

- Chain: `queue → nvvideoconvert (pre) → caps (video/x-raw(memory:NVMM), format=RGBA) → nvdsosd → nvvideoconvert (post) → caps (video/x-raw(memory:NVMM), format=I420) → nvjpegenc → appsink`
- appsink: `emit-signals=True`, `sync=False`, `max-buffers=5`, `drop=True`
- appsink `new-sample` → `_on_new_jpeg_sample(sensor_id)` enqueues JPEG bytes to `jpeg_queues[sensor_id]`
- Branch elements `sync_state_with_parent()` to PLAYING

Tear down: set elements to NULL, remove from pipeline, release the demux request pad, clear mappings.

## Bus handling

- ERROR, EOS, WARNING, INFO, STATE_CHANGED handled in `_on_bus_message()`; errors stop the mainloop.

## Runtime controls (Python API)

- `update_detection_config({confidence_threshold?, iou_threshold?, detection_enabled?, target_classes?})`
- `update_detection_toggle(toggle_name, enabled)` for `detect_people`, `detect_vehicles`, `detect_furniture`
- `set_detection_enabled(enabled)`
- `update_confidence_threshold(value)`
- `update_iou_threshold(value)`
- `update_target_classes([ids])`
- `set_trail_visualization(enabled)` / `toggle_trail_visualization(enabled)`

## Dynamic sensors (nvmultiurisrcbin REST)

- `add_sensor(sensor_id, uri)` → POST `http://localhost:<port>/stream` with `{change:"add", sensorId: str, uri}`
- `remove_sensor(sensor_id)` → POST with `{change:"remove", sensorId: str, uri: ""}`
