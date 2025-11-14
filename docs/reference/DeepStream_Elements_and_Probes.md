# DeepStream Elements and Probes

This document summarizes the elements created in `deepstream_video_pipeline.py`, their key properties, and the probes/signals attached.

## Elements

- nvmultiurisrcbin (multiurisrc)
  - Properties: `uri-list`, `sensor-id-list`, `max-batch-size`, `width`, `height`, `batched-push-timeout=-1`, `live-source=1`, `drop-pipeline-eos=1`, `rtsp-reconnect-interval=30`, `port`, `ip-address="localhost"`

- nvdspreprocess (preprocess)
  - Properties: `config-file = config.processing.DEEPSTREAM_PREPROCESS_CONFIG`

- nvinfer (nvinfer)
  - Properties: `config-file-path = pipelines/config_infer_primary_yolo11.ini`, `input-tensor-meta=False`
  - Runtime updates (via Python): `confidence-threshold`, `iou-threshold`, `enable`, `custom-lib-props="target-classes:<ids>"`

- nvdsroiexclude (pre-tracker ROI exclude)
  - Properties: `config-file=pipelines/config_nvdsanalytics_exclude.ini`, `id-mode=pad-index|source-id`
  - Behavior: In-place metadata transform that removes objects fully inside static ROIs and optionally draws ROI outlines (via display meta). No pad probes required.

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

## Metadata Handling (Operator)

- For any probe that reads or mutates metadata, prefer the shared helpers in `pipelines/meta_ops.py`.
  - Operator-first traversal with safe `pyds` fallbacks keeps code concise and DS8-compatible.
  - See: docs/reference/Metadata_Ops.md

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
