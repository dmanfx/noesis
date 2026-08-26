# DS9 ReID and V3DT Native Metadata Bridge Audit

Date: 2026-05-10

Scope: DS9 migration review only. Live DS8 remains untouched. This audit uses `DS9/` files as local evidence and official NVIDIA DeepStream 9.0 documentation for DS9 capability claims.

## Questions Answered

| Question | Answer | Call |
| --- | --- | --- |
| Does DS9 Service Maker Python expose object-level SGIE tensor metadata reliably enough to remove `noesis_reid_meta_ext`? | Not proven. DS9 `gst-nvinfer` still documents SGIE tensor output as `NvDsInferTensorMeta` on `NvDsObjectMeta.obj_user_meta_list`, and Service Maker Python docs describe `TensorOutputUserMetadata` as frame-level access. The official docs do not prove a Python object-level tensor iterator/converter equivalent to the native bridge. | Keep |
| Does DS9 Service Maker Python expose V3DT visibility/image-foot/3D bbox/world-foot metadata directly enough to remove `noesis_v3dt_meta_ext`? | Not proven. DS9 docs prove the V3DT metadata exists and DS9 Service Maker C++ exposes wrappers for visibility, image-foot, and 3D bbox. They do not prove Python exposes those wrappers directly, and no official `ObjectWorldFootLocationUserMetadata` wrapper was found. | Keep |
| Is tracker-internal ReID metadata more canonical/reliable in DS9 than the current OSNet SGIE path? | Tracker-internal ReID is canonical for nvtracker association/re-association. It is not yet proven more reliable as Noesis StableID input than the current OSNet SGIE path because the DS9 sandbox still wires the dedicated SGIE and does not enable or consume tracker ReID output. | Change candidate, not replacement |

## Local Evidence

### Current OSNet SGIE path

- `DS9/pipelines/config_infer_secondary_reid_osnet.ini` defines a dedicated OSNet SGIE:
  - secondary/object mode: `process-mode=2`
  - person-only input: `operate-on-gie-id=1`, `operate-on-class-ids=0`
  - tensor export: `output-tensor-meta=1`
  - unique id: `gie-unique-id=3`
- `DS9/noesis/pipelines/ds8_pipeline.py` still inserts the ReID SGIE after tracker and before analytics when `models.reid` is enabled.
- `DS9/noesis/pipelines/hooks.py` discovers the ReID SGIE id from config, calls `noesis_reid_meta_ext.extract_reid_embedding(...)`, and records a ReID host-copy counter only after converting the extracted embedding.
- `DS9/native/noesis_reid_meta_ext.cpp` unwraps `deepstream::ObjectMetadata` to `NvDsObjectMeta*`, iterates `obj_user_meta_list`, filters `NVDSINFER_TENSOR_OUTPUT_META`, selects the requested layer, and returns a normalized float vector.

That bridge is not just convenience glue; it is the current object-level SGIE tensor access path.

### Current V3DT path

- `docs/history/runtime/ds8/DS8_metadata_contracts.md` defines the downstream
  tracking contract for `bbox3d`, `velocity3d`, `image_foot`, `image_base`,
  `world`, `world_frame`, and `world_source`; it states that in `v3dt` mode,
  `world` and `world_source="bbox3d"` come from `NVDS_OBJ_3D_META`.
- `DS9/noesis/pipelines/hooks.py` enables V3DT meta extraction only in V3DT mode, calls `noesis_v3dt_meta_ext.extract_obj_3d_meta(...)`, derives `world` from `bbox3d`, and forwards `bbox3d`, `velocity3d`, `visibility`, `image_foot`, `image_base`, and world fields into public tracks.
- `DS9/native/noesis_v3dt_meta_ext.cpp` uses Service Maker C++ object user-meta iteration for `NVDS_OBJ_VISIBILITY`, `NVDS_OBJ_IMAGE_FOOT_LOCATION`, and `NVDS_OBJ_3D_META`. It explicitly avoids `NVDS_OBJ_WORLD_FOOT_LOCATION` because the Service Maker wrapper is absent, deriving world footpoints from bbox instead.
- `docs/history/runtime/ds8/ds8_design_decisions.md` records the same bridge
  rationale: Service Maker Python did not expose arbitrary per-object user meta,
  so the native bridge was used to read `NVDS_OBJ_3D_META`; later notes
  confirmed no Service Maker world-foot wrapper and kept bbox-derived world
  footpoints.

## Official NVIDIA DS9 Evidence

### SGIE tensor metadata

- NVIDIA `gst-nvinfer` docs for DeepStream 9.0 state that raw tensor output is attached as `NvDsInferTensorMeta`; for primary mode it is added to `NvDsFrameMeta.frame_user_meta_list`, and for secondary/object mode it is added to `NvDsObjectMeta.obj_user_meta_list`. Source: [Gst-nvinfer Tensor Metadata](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#tensor-metadata).
- NVIDIA Service Maker Python docs state that `TensorOutputUserMetadata` is attached to frame metadata when `output-tensor-meta` is enabled. They do not describe object-level SGIE tensor access through Python `ObjectMetadata`. Source: [Service Maker Python Advanced Features, Leveraging Metadata](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_advanced_features.html#leveraging-metadata).
- NVIDIA Service Maker C++ API docs for 9.0 show `deepstream::ObjectMetadata::iterate(... UserMetadata ..., meta_type)` and `append(...)`, which is exactly the C++ surface the local bridge uses. Source: [ObjectMetadata API](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/classdeepstream_1_1ObjectMetadata.html).

Conclusion: DS9 proves the metadata exists and C++ can iterate object user meta. It does not prove Service Maker Python has a reliable object-level SGIE tensor API, so removal is not justified.

### V3DT metadata

- NVIDIA `gst-nvtracker` docs state SV3DT can output visibility, foot locations in world plane and 2D image, convex hull, and 3D bbox data to object metadata when configured, and that 3D bbox is stored in `NvDsObj3DBbox` user meta. Source: [Gst-nvtracker, Metadata for 3D Tracking](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#meta-data-for-3d-tracking).
- NVIDIA tracker config docs list `outputVisibility`, `outputFootLocation`, and `outputConvexHull`; `outputFootLocation` controls foot-location object meta and the 3D bbox output. Source: [Gst-nvtracker, Object Re-ID and Object Model Projection parameters](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#configuration-parameters).
- NVIDIA Service Maker C++ API docs for 9.0 expose:
  - `ObjectVisibilityUserMetadata::getVisibility()` at [ObjectVisibilityUserMetadata](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/classdeepstream_1_1ObjectVisibilityUserMetadata.html)
  - `ObjectImageFootLocationUserMetadata::getImageFootLocation()` at [ObjectImageFootLocationUserMetadata](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/classdeepstream_1_1ObjectImageFootLocationUserMetadata.html)
  - `Object3DBBoxUserMetadata::get3DBbox()` at [Object3DBBoxUserMetadata](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/classdeepstream_1_1Object3DBBoxUserMetadata.html)
- No official DS9 Service Maker API page was found for `ObjectWorldFootLocationUserMetadata`.

Conclusion: DS9 proves C++ access to the core V3DT object metadata, but not Python direct access. World-foot direct extraction remains not proven; Noesis should continue deriving world footpoints from `bbox3d` unless a DS9 Python API or tested C++ bridge proves otherwise.

### Tracker-internal ReID

- NVIDIA `gst-nvtracker` docs describe tracker ReID as a TensorRT-backed module for NvDeepSORT and target re-association, with an internal gallery and cosine similarity. Source: [Gst-nvtracker, Target Re-Identification](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#target-re-identification).
- The same docs state `outputReidTensor` outputs ReID features to user meta for downstream usage, and the ReID feature output section says downstream modules can retrieve those features when NvDeepSORT or ReID-based re-association is used. Source: [Gst-nvtracker, Re-ID Feature Output](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#re-id-feature-output).
- NVIDIA Service Maker C++ API docs for 9.0 expose `ObjectReidUserMetadata`, wrapping `NvDsObjReid`, with `featureSize()` and `featureVector()` CPU pointer access. Source: [ObjectReidUserMetadata](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/classdeepstream_1_1ObjectReidUserMetadata.html).

Conclusion: tracker ReID is canonical inside the tracker. It is a credible future DS9 integration target, especially now that DS9 C++ documents `ObjectReidUserMetadata`. It is not proven as the better Noesis StableID source until the DS9 sandbox enables `outputReidTensor`, chooses tracker ReID cadence/model parameters, and proves extraction from the actual Service Maker Python hook point.

## Proven

- DS9 `gst-nvinfer` still places secondary GIE tensor metadata under object user meta, not a guaranteed frame-level Python tensor list.
- DS9 Service Maker Python docs prove frame metadata, object metadata, and frame-level `TensorOutputUserMetadata` concepts, but not object-level SGIE tensor access in Python.
- DS9 Service Maker C++ docs expose object user-meta iteration.
- DS9 `gst-nvtracker` can produce V3DT object metadata for visibility, foot location, convex hull, and 3D bbox when configured.
- DS9 Service Maker C++ docs expose wrappers for object visibility, image foot location, 3D bbox, and tracker ReID feature vectors.

## Not Proven

- A DS9 `pyservicemaker` Python method that directly iterates `ObjectMetadata` user metadata by type and converts object-level SGIE `NvDsInferTensorMeta` into layer tensors.
- A DS9 `pyservicemaker` Python wrapper for `ObjectVisibilityUserMetadata`, `ObjectImageFootLocationUserMetadata`, `Object3DBBoxUserMetadata`, or `ObjectReidUserMetadata`.
- Any official DS9 `ObjectWorldFootLocationUserMetadata` wrapper.
- That tracker ReID output is enabled, complete, synchronized, and available at the current Noesis analytics hook point.
- That tracker-internal ReID output cadence is equivalent to the current OSNet SGIE cadence. NVIDIA documents interval dependence for tracker ReID feature extraction.

## Final Calls

### `noesis_reid_meta_ext`

Keep for the first DS9 migration. The only safe removal path is a DS9 proof that Service Maker Python can access object-level SGIE `NVDSINFER_TENSOR_OUTPUT_META` at the ReID SGIE downstream hook with layer name, unique id, dimensions, host/device pointer behavior, and lifetime all verified. Until then, removing the bridge would remove the current StableID embedding source.

Change candidate: add a DS9 smoke test later that compares native bridge extraction with any discovered Python-native object tensor API on the same SGIE output. If they match over live frames and lifecycle stress, then replacement becomes reasonable.

### `noesis_v3dt_meta_ext`

Keep. DS9 C++ exposes the necessary V3DT wrappers, but direct Python exposure is not proven, and world-foot direct access is still not proven. The current bridge maps the real DS9/DeepStream object metadata into the Noesis telemetry contract without adding a fallback path.

Change candidate: if DS9 Python exposes the exact C++ metadata wrappers, replace the bridge only after proving `visibility`, `image_foot`, `bbox3d`, `velocity3d`, and derived `world` parity against the native bridge.

### Tracker-internal ReID

Do not replace OSNet SGIE with tracker-internal ReID as part of this bridge-removal pass. Treat tracker ReID as a future DS9 experiment, not a proven simplification. It may be more canonical for tracker-owned association, but the current Noesis StableID contract still depends on a controlled appearance embedding source that is already wired as a dedicated SGIE.

Recommended DS9 experiment: enable tracker ReID output in a separate DS9 branch/config, extract `ObjectReidUserMetadata` through C++ first, and compare feature availability, cadence, identity stability, and CPU/GPU cost against the current OSNet SGIE path before changing the canonical migration plan.
