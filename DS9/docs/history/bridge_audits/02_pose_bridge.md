# DS9 Pose Bridge and Pose Parser Exposure Audit

Status: review report only. No code or config changes were made.

Scope:
- `native/noesis_pose_meta_ext.cpp`
- `pipelines/nvdsinfer_yolo26_pose/nvdsinfer_yolo26_pose.cpp`
- `pipelines/config_infer_secondary_yolo26_pose.ini`
- `noesis/pipelines/hooks.py`
- `noesis/metadata/pose_features.py`
- `docs/history/ds8/DS8_pose_stable_id_integration.md`
- `docs/history/ds8/DS8_metadata_contracts.md`
- `docs/history/ds8/ds8_design_decisions.md`

Official NVIDIA DS9 evidence used:
- [DeepStream 9.0 Gst-nvinfer](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html)
- [DeepStream 9.0 Gst-nvdspostprocess](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspostprocess.html)
- [DeepStream 9.0 NvDsInferTensorMeta API](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/structNvDsInferTensorMeta.html)
- [DeepStream 9.0 nvdsmeta.h API](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/9_80_2sources_2includes_2nvdsmeta_8h.html)
- [DeepStream 9.0 Service Maker Python advanced features](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_advanced_features.html)
- [DeepStream 9.0 Service Maker C++ ObjectMetadata API](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/classdeepstream_1_1ObjectMetadata.html)
- [DeepStream 9.0 Service Maker C++ TensorOutputUserMetadata API](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/classdeepstream_1_1TensorOutputUserMetadata.html)
- [DeepStream 9.0 Service Maker C++ BatchMetadata API](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/classdeepstream_1_1BatchMetadata.html)

## Questions

### 1. Does DS9 Service Maker Python expose object user-meta append/read equivalent to `obj_user_meta_list` and `nvds_add_user_meta_to_obj`?

Verdict: not proven.

Proven:
- DS9 DeepStream C metadata still exposes `nvds_add_user_meta_to_obj`, `nvds_remove_user_meta_from_object`, and object user-meta clearing APIs in `nvdsmeta.h`.
- DS9 Service Maker C++ `ObjectMetadata` exposes user metadata iteration and `append(const UserMetadata&)`.
- DS9 Service Maker C++ `BatchMetadata` can acquire generic `UserMetadata` from the pool.

Not proven:
- The official DS9 Service Maker Python page describes `BatchMetadata.frame_items`, frame/object/display/user metadata concepts, and `TensorOutputUserMetadata`, but it does not document a Python `ObjectMetadata.append(...)`, object user-meta iterator by meta type, `BatchMetadata.acquire(UserMetadata)`, or a Python equivalent of `nvds_add_user_meta_to_obj`.
- Therefore, there is no NVIDIA-doc-backed proof that DS9 Python alone can attach `NOESIS.POSE_FEATURES` to each object and later read that same object-level user meta without a native bridge.

Repo evidence:
- `native/noesis_pose_meta_ext.cpp` unwraps `deepstream::ObjectMetadata` to `NvDsObjectMeta*`, scans `obj_user_meta_list`, removes existing pose meta, acquires `NvDsUserMeta`, sets copy/release callbacks, and calls `nvds_add_user_meta_to_obj`.
- `noesis/pipelines/hooks.py` imports `noesis_pose_meta_ext`; if `attach_pose_features` is missing, pose meta attachment is skipped.
- `docs/history/ds8/DS8_metadata_contracts.md` and `docs/history/ds8/ds8_design_decisions.md` record that the native bridge exists because Service Maker Python did not expose the required object user-meta surface.

Call: Keep `noesis_pose_meta_ext` for the first DS9 port. Change only after a DS9 runtime probe proves the Python package exposes object-level user-meta append/read with lifecycle-safe custom payloads.

### 2. Does DS9 expose object-level tensor metadata/keypoints enough to eliminate `noesis_pose_meta_ext`?

Verdict: tensor placement is proven, Python elimination is not proven.

Proven:
- DS9 `Gst-nvinfer` can attach raw output tensor data as `NvDsInferTensorMeta`.
- For secondary GIE mode, NVIDIA documents that `NvDsInferTensorMeta` is attached to each `NvDsObjectMeta` object's `obj_user_meta_list`.
- `NvDsInferTensorMeta` carries `unique_id`, output layer info, host output pointers, device output pointers, network info, and aspect-ratio/padding flags.
- DS9 Service Maker C++ has `TensorOutputUserMetadata::getLayers()`.

Not proven:
- NVIDIA's DS9 Service Maker Python docs do not document object-level `TensorOutputUserMetadata` iteration/conversion from `ObjectMetadata`.
- NVIDIA docs do not provide a built-in keypoint decoder for YOLO26 pose output. The current keypoint decode is Noesis model-specific logic: choose `output0`, select the best row, account for ROI, letterbox, normalization, and emit 17 `(x, y, confidence)` keypoints.
- Service Maker C++ docs list `TensorOutputUserMetadata` as iterable within `FrameMetadata` or `RoiMetadata`; that does not prove Python can directly iterate SGIE object tensor meta from `ObjectMetadata`.

Repo evidence:
- `native/noesis_pose_meta_ext.cpp` reads `NVDSINFER_TENSOR_OUTPUT_META` from `obj_user_meta_list`, filters by `gie_id`, decodes the YOLO26 pose tensor row, and converts keypoints from model/ROI space to absolute frame coordinates.
- `noesis/pipelines/hooks.py` uses `noesis_pose_meta_ext.extract_pose_keypoints(...)` in `PoseFeatureProcessor`, pose keypoint overlay, and the fused world-anchor path.
- `PoseFeatureResult` stores `keypoints_roi`, `keypoints_abs`, confidence summaries, ratio features, stable id, and timestamps.
- The metadata contract says pose keypoints are the preferred person image-anchor authority for baseline world estimation and are also a secondary StableID signal.

Call: Keep `noesis_pose_meta_ext`. DS9 proves the tensor exists at the native metadata level, but not that Python exposes enough object-level tensor access plus lifecycle-safe object user-meta attach/read to remove this bridge.

### 3. Does DS9 `nvinfer` support a pure tensor-output SGIE path that removes the no-op `yolo26_pose` parser?

Verdict: proven.

Proven:
- DS9 `Gst-nvinfer` supports `output-tensor-meta=1` and raw tensor metadata.
- DS9 `Gst-nvdspostprocess` documents the parser-disabled tensor path: set `output-tensor-meta=1` in the inference config and set `network-type=100` ("other") to disable parsing in `nvinfer`.
- DS9 `Gst-nvinfer` supports secondary operation on upstream object metadata through `process-mode=2`, `operate-on-gie-id`, and `operate-on-class-ids`.

Repo evidence:
- `pipelines/config_infer_secondary_yolo26_pose.ini` already sets `process-mode=2`, `network-type=100`, `gie-unique-id=4`, `operate-on-gie-id=1`, `operate-on-class-ids=0`, `output-blob-names=output0`, `output-tensor-meta=1`, and `disable-output-host-copy=0`.
- The pose config does not set `custom-lib-path` or `parse-bbox-func-name`.
- `pipelines/nvdsinfer_yolo26_pose/nvdsinfer_yolo26_pose.cpp` is a no-op detector parser that clears the output object list; it exists only to satisfy parser expectations in a detector-style setup.

Call: Remove the no-op `nvdsinfer_yolo26_pose` parser from the DS9 migration target, after confirming no DS9 build file or packaged deployment still references the library. The active DS9 config shape should stay pure tensor-output SGIE: `network-type=100` plus `output-tensor-meta=1`, with Noesis post-processing consuming the tensor metadata.

## Final Calls

| Component | Call | Rationale |
| --- | --- | --- |
| `native/noesis_pose_meta_ext.cpp` | Keep | DS9 docs prove native object tensor/user meta exists, but do not prove Python has equivalent object user-meta append/read and object SGIE tensor access. The bridge still carries Noesis-specific keypoint decode and object JSON meta lifecycle. |
| Pose feature hook usage in `hooks.py` | Keep | Existing hooks depend on `extract_pose_keypoints`, `attach_pose_features`, and `extract_pose_features` for StableID, overlay, and fused world anchoring. |
| `noesis/metadata/pose_features.py` payload | Keep | It is the local contract for pose features, keypoints, quality, stable id, and timestamps. DS9 does not replace this schema. |
| `pipelines/config_infer_secondary_yolo26_pose.ini` tensor mode | Keep | Keep the parser-free DS9 shape: SGIE crop inference, `network-type=100`, `output-tensor-meta=1`, host copy enabled for safe keypoint decode. Later path cleanup should remove absolute machine-local paths, but that is outside this audit. |
| `pipelines/nvdsinfer_yolo26_pose/nvdsinfer_yolo26_pose.cpp` | Remove | DS9 docs prove pure tensor-output inference can disable parsing with `network-type=100`; the current DS9 pose config already uses that path and does not reference the no-op parser. |

## Follow-up Acceptance Gate

Before deleting `noesis_pose_meta_ext`, require a DS9 runtime proof on the target package:
1. Python can iterate `NVDSINFER_TENSOR_OUTPUT_META` from each secondary pose `ObjectMetadata`.
2. Python can allocate, configure copy/release lifecycle, append, replace, and read custom object user meta equivalent to `NOESIS.POSE_FEATURES`.
3. The Python path reproduces `keypoints_abs`, `keypoints_roi`, pose quality, StableID pose features, overlay, and fused world-anchor behavior with no CPU/appsink fallback.

Until all three pass, keep the native pose bridge.
