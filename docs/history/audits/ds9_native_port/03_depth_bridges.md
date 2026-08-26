# DS9 Depth Bridge Audit

Date: 2026-05-10

Scope: object-depth and DAv2 depth tensor bridges in the DS9 migration mirror. This report audits only the DS9 tree and official NVIDIA DeepStream 9.0 documentation for DS9 API claims.

## Files Audited

- [`noesis_depth_meta_ext.cpp`](../../../../DS9/native/noesis_depth_meta_ext.cpp)
- [`noesis_depth_tracking_tensor_ext.cpp`](../../../../DS9/native/noesis_depth_tracking_tensor_ext.cpp)
- [`object_depth.py`](../../../../DS9/noesis/metadata/object_depth.py)
- [`depth_result.py`](../../../../DS9/noesis/metadata/depth_result.py)
- [`hooks.py`](../../../../DS9/noesis/pipelines/hooks.py)
- [`config_infer_secondary_depth_tracking_da2.template.ini`](../../../../DS9/pipelines/config_infer_secondary_depth_tracking_da2.template.ini)
- [`DS8_metadata_contracts.md`](../../runtime/ds8/DS8_metadata_contracts.md)
- [`DS8_testing_guide.md`](../../runtime/ds8/DS8_testing_guide.md)
- [`ds8_design_decisions.md`](../../runtime/ds8/ds8_design_decisions.md)

## NVIDIA DS9 Evidence

- DS9 Service Maker Python Advanced Features documents `BufferOperator` / `BatchMetadata`, `frame_items`, frame/object metadata, and `TensorOutputUserMetadata`, but does not document Python object-level arbitrary user-meta append/read or mask-param accessors:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_service_maker_python_advanced_features.html#leveraging-metadata>
- DS9 Service Maker C++ `ObjectMetadata` documents `iterate(... UserMetadata ..., meta_type)`, `append(UserMetadata)`, `maskParams()`, and `setMaskParams(...)`:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/classdeepstream_1_1ObjectMetadata.html>
- DS9 `pyds.NvDsObjectMeta` documents `mask_params` and `obj_user_meta_list`; DS9 `pyds` methods document `nvds_add_user_meta_to_obj`, `nvds_acquire_user_meta_from_pool`, remove/clear object user-meta, `user_copyfunc`, and `user_releasefunc`:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html>
  <https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/Methods/methodsdoc.html>
- DS9 `NvOSD_MaskParams` still exposes `data`, `size`, `threshold`, `width`, and `height` in both C SDK and `pyds` Python docs:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/struct__NvOSD__MaskParams.html>
  <https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/NvOSD/NvOSD_MaskParams.html>
- DS9 `NvDsInferTensorMeta` still exposes `unique_id`, `num_output_layers`, `output_layers_info`, `out_buf_ptrs_host`, `out_buf_ptrs_dev`, `gpu_id`, `priv_data`, `network_info`, `maintain_aspect_ratio`, and `symmetric_padding`:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/structNvDsInferTensorMeta.html>
- DS9 `pyds.NvDsInferTensorMeta` exposes `out_buf_ptrs_host`, `out_buf_ptrs_dev`, `gpu_id`, and `output_layers_info(...)`, but does not list `network_info`, `maintain_aspect_ratio`, or `symmetric_padding` in the Python wrapper:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/NvDsInfer/NvDsInferTensorMeta.html>
- DS9 `Gst-nvinfer` docs state `output-tensor-meta=1` attaches `NvDsInferTensorMeta` to `frame_user_meta_list` for full-frame mode or `obj_user_meta_list` for secondary/object mode. The docs also list `output-instance-mask`, `maintain-aspect-ratio`, `symmetric-padding`, `scaling-filter`, and `scaling-compute-hw`:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications>
  <https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvinfer.html#tensor-metadata>
- DS9 SDK API docs for `gstnvinfer_property_parser.h` still define `CONFIG_GROUP_INFER_DISABLE_OUTPUT_HOST_COPY` as `"disable-output-host-copy"`:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/9_80_2sources_2gst-plugins_2gst-nvinfer_2gstnvinfer__property__parser_8h.html>
- DS9 `NvBufSurfTransform` exposes GPU-capable surface transforms and ROI/crop flags for `NvBufSurface` image buffers, not a Python-documented helper that resizes raw `NvDsInferTensorMeta::out_buf_ptrs_dev` layers or copies arbitrary tensor ROIs:
  <https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/9_80_2sources_2includes_2nvbufsurftransform_8h.html>

## Bridge 1: `noesis_depth_meta_ext`

What it does:

- Unwraps Service Maker `ObjectMetadata` to `NvDsObjectMeta*`.
- Registers `NOESIS.OBJECT_DEPTH` with `nvds_get_user_meta_type`.
- Replaces old object-depth meta by walking `obj_user_meta_list`.
- Acquires `NvDsUserMeta` from the batch pool, attaches JSON payload via `nvds_add_user_meta_to_obj`, and provides DeepStream copy/release callbacks.
- Reads the JSON payload back from `obj_user_meta_list`.
- Copies `NvOSD_MaskParams` (`data`, `size`, `threshold`, `width`, `height`) into a NumPy array for strict instance-mask sampling.

Local contract:

- `ObjectDepthResult` is a versioned object-level JSON payload with required status/sample/anchor fields.
- The metadata contract says `NOESIS.OBJECT_DEPTH` is required by baseline non-V3DT fused world estimation, must remain raw, and is attached after full-frame depth is aligned once and sampled over the decoded instance mask.
- `hooks.py` requires this bridge to attach object-depth meta, extract existing object-depth meta, and decode instance masks before sampling depth over person masks.

Proven:

- DS9 C++ Service Maker can append and iterate object user metadata and expose mask params.
- DS9 `pyds` can append object user metadata, acquire user meta from the pool, access object user-meta lists, set user-meta copy/release callbacks, and access `NvOSD_MaskParams`.
- The DS9 `NvOSD_MaskParams` C/Python fields used by the bridge are still present.

Not proven:

- Official DS9 Service Maker Python docs do not prove a safe Python-native replacement for arbitrary object user-meta append/read on Service Maker `ObjectMetadata`.
- Official DS9 Service Maker Python docs do not prove a Python-native mask-param accessor on Service Maker `ObjectMetadata`.
- Using `pyds` directly would require a supported conversion from Service Maker Python `ObjectMetadata` to `pyds.NvDsObjectMeta` plus payload ownership/copy/release parity. That conversion is not proven by the audited DS9 docs.

Call: **KEEP for the first DS9 port.**

Change later only if a DS9 smoke proves one of these:

- Service Maker Python exposes documented object-level arbitrary user-meta append/read and mask-param APIs, including lifecycle/copy/release behavior, or
- The bridge is intentionally rewritten as a DS9 C++ Service Maker plugin/operator that owns the object-depth contract directly.

Do not remove this bridge just because DS9 C++ Service Maker or `pyds` exposes lower-level pieces; the current Python Service Maker replacement path is not proven safe enough.

## Bridge 2: `noesis_depth_tracking_tensor_ext`

What it does:

- Unwraps Service Maker `FrameMetadata` to `NvDsFrameMeta*`.
- Walks `frame_user_meta_list` looking for `NVDSINFER_TENSOR_OUTPUT_META` matching the DAv2 `gie_id`.
- Uses `NvDsInferTensorMeta::out_buf_ptrs_dev` as the primary source.
- Selects the depth layer, supports float and half outputs, converts half to float on GPU with NPP, resizes the full-frame depth tensor to canonical frame size on GPU, then keeps the aligned frame in CUDA memory.
- Exposes `copy_roi_to_numpy(...)`, which copies only the object ROI to host for final masked statistics.
- Provides `capture_mapanything_tensor_layers_exact(...)` for the canonical
  MapAnything CPU-edge product. It requires one exact UID, exact
  per-frame `depth/conf/mask` shape, and no manual batch slice because nvinfer
  already offsets attached frame pointers; the former generic fallback/debug
  export has been removed. Because those pointers remain nvinfer-owned, the
  bridge copies them only during the frame-metadata lifetime, releases the GIL
  during D2H, and returns exactly three owned `294x518` float32 arrays
  (`1,827,504` bytes). Alignment/storage/publication then run on the bounded
  runtime-owned worker, which shutdown drains and joins explicitly.

Local contract:

- The DAv2 config template uses full-frame input, `output-tensor-meta=1`, and `disable-output-host-copy=1`.
- The metadata contract says object-depth payloads are attached only after full-frame depth has been aligned once into canonical frame coordinates and sampled strictly over decoded instance masks.
- `hooks.py` fails fast when the bridge is missing for baseline depth tracking, stores the GPU-resident aligned depth frame, and copies only object ROIs before statistics.

Proven:

- DS9 `NvDsInferTensorMeta` still exposes device buffers via `out_buf_ptrs_dev`.
- DS9 `Gst-nvinfer` still documents `output-tensor-meta=1` and the frame/object attachment location for tensor meta.
- DS9 Service Maker C++ `TensorOutputUserMetadata` exposes `getLayers()`, and Service Maker `Tensor` exposes `data()`, `shape()`, `dtype()`, `bits()`, `deviceId()`, and `deviceType()`.
- DS9 has GPU-capable `NvBufSurfTransform` for `NvBufSurface` image buffers.

Not proven:

- Official DS9 Service Maker Python docs do not expose a Python API that gives raw device tensor pointers plus shape/type with enough ownership guarantees to replace the native traversal of `NvDsInferTensorMeta`.
- Official DS9 docs do not expose a Python helper that resizes a raw full-frame depth tensor from `out_buf_ptrs_dev` to canonical frame size while keeping it device-resident.
- Official DS9 docs do not expose a Python helper for arbitrary depth-tensor ROI copy from a GPU-resident aligned tensor. `NvBufSurfTransform` is a surface-transform API, not a documented raw tensor ROI/resize replacement for this bridge.
- DS9 official SDK API docs prove the `disable-output-host-copy` config key still exists; live validation is still required to prove the Noesis DAv2 branch preserves the intended device-only behavior with this specific model and graph.

Call: **KEEP, with a DS9 validation/change item.**

Required DS9 change work:

- Rebuild the bridge against DS9 headers and confirm `NvDsInferTensorMeta` ABI/field access at compile time.
- Validate `disable-output-host-copy=1` under DS9 with a live DAv2 run. The key is documented in the DS9 parser header, but Noesis still needs runtime proof that `out_buf_ptrs_host` remains unused/absent and `out_buf_ptrs_dev` is populated for this branch.
- Keep the current bridge behavior until DS9 proves a Python or C++ Service Maker replacement that preserves full-frame device residency, GPU resize/alignment, and object-only host ROI copies.

## Field Compatibility

`NvDsInferTensorMeta`:

- DS9 C SDK fields needed by the native bridge are present: `unique_id`, `num_output_layers`, `output_layers_info`, `out_buf_ptrs_host`, `out_buf_ptrs_dev`, and `gpu_id`.
- DS9 also includes `network_info`, `maintain_aspect_ratio`, and `symmetric_padding`; those match the current field names and do not require source edits in the audited bridge.
- DS9 Python wrapper docs omit `network_info`, `maintain_aspect_ratio`, and `symmetric_padding`, so any future Python rewrite must not assume those fields are available through `pyds`.

`NvOSD_MaskParams`:

- DS9 C SDK and `pyds` docs still expose `data`, `size`, `threshold`, `width`, and `height`.
- The native object-depth bridge uses those same fields. No source edit is indicated by DS9 field naming.

Call: **NO SOURCE EDIT REQUIRED for these struct field names based on official DS9 docs.** Rebuild against DS9 to catch ABI/header drift, but the documented field surface is compatible with the current native source.

## Final Calls

| Component | Call | Reason |
| --- | --- | --- |
| `noesis_depth_meta_ext` | **KEEP** | DS9 Python Service Maker replacement is not proven for arbitrary object user-meta lifecycle or Service Maker object mask extraction. |
| `noesis_depth_tracking_tensor_ext` | **KEEP + VALIDATE** | DS9 exposes device tensor metadata, but not a documented Python/Service Maker path for raw tensor GPU resize/alignment and ROI copy while preserving full-frame device residency. |
| `NvDsInferTensorMeta` field access | **CHANGE NOT REQUIRED** | DS9 C SDK documents the fields used by the native bridge. |
| `NvOSD_MaskParams` field access | **CHANGE NOT REQUIRED** | DS9 C SDK and `pyds` docs document the fields used by the native bridge. |
| DAv2 `disable-output-host-copy=1` assumption | **KEEP + VALIDATE** | DS9 SDK API docs still define the config key; validate live before claiming Noesis DS9 zero-copy equivalence for DAv2. |

Overall recommendation: **keep both depth native bridges for the first DS9 migration pass, rebuild them against DS9, and treat removal as a later reduction only after DS9 Service Maker Python or a DS9-native C++ operator proves the same metadata lifecycle and GPU-residency behavior.**
