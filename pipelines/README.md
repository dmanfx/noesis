Active configs
- config_infer_primary_yolo11.ini: Primary detector (YOLO‑11), used by nvinfer in deepstream_video_pipeline.py
- config_preproc.ini: nvdspreprocess settings for pre‑inference GPU transforms
- config_nvdsanalytics_exclude.ini, config_nvdsanalytics_post.ini: Analytics stages for zone filtering and post‑tracker analytics
- config_tracker_nvdcf_batch.yml: Tracker config used by nvtracker (includes NvDCF tuning for occlusions)
- config_infer_secondary_depth_anything_v2.ini: Depth Anything V2 metric SGIE that augments detections with per-object metric depth

Example/Reference configs (not wired into the current pipeline)
- config_infer_secondary_classification.ini
- config_tracker_nvdcf_basic.yml
- config_tracker_nvdcf_batch_lowlevel.yml
- dstest1_pgie_config.txt

Depth Anything V2 TensorRT build helper
--------------------------------------

Use ``scripts/prepare_depth_anything_v2_engine.py`` to clone LiheYoung/Depth-Anything-V2,
download the metric ViT-L checkpoint, export to ONNX, and compile the FP16 TensorRT engine
referenced by ``config_infer_secondary_depth_anything_v2.ini``. The script is idempotent
and can skip engine generation (``--skip-trtexec``) if you only need the ONNX artifact.

Note: The primary detector uses a custom parser library. Current deployment points to the system lib path. Rebuild from legacy/deepstream_parser if you want to pin a local parser lib and update config_infer_primary_yolo11.ini accordingly.

ReID/StableID Integration
- OSNet ReID is integrated via `reid/` modules; see `reid/README.md` for details and config flags.
- Tracker config (`config_tracker_nvdcf_batch.yml`) is tuned for partial occlusions:
  - `searchRegionPaddingScale: 5.0`
  - `targetVisibilityThreshold: 0.3`
  - `maxMatchingFrameGap: 40`, `maxTrackingFrameGap: 90`
