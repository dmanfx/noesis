# DS9 Custom `nvinfer` Parser Audit

Date: 2026-05-10

Scope: `DS9/pipelines/nvdsinfer_*`, `DS9/config/infer.yaml`,
`docs/history/runtime/ds8/DS8_testing_guide.md`,
`docs/history/runtime/ds8/ds8_design_decisions.md`, plus official NVIDIA DS9
documentation. No code or config edits were made.

## Executive Answer

DS9 `gst-nvinfer` does not add built-in parser support that replaces the YOLO26-Seg, YOLO11-Seg, or RF-DETR-Seg custom parsers. Those parsers are still model-output translators from custom tensor layouts into DeepStream object and instance-mask metadata.

The parser ABI used by these files is not changed for the audited entrypoints. DS9 still documents the same custom object parser and custom instance-mask parser signatures in `nvdsinfer_custom_impl.h`; rebuild against DS9 headers, but do not plan a source-level signature rewrite.

The YOLO26 pose no-op parser should be removed from the DS9 runtime/build plan unless DS9 validation proves otherwise. This is not because DS9 added a pose parser; it is because DS9 documents a tensor-output-only path, and the local DS9 pose config already uses it: `network-type=100`, `output-tensor-meta=1`, and no custom parser reference.

`disable-output-host-copy` and device-buffer behavior are still documented in DS9. Keep the device-pointer-aware parser paths for the segmentation parsers that rely on them, but validate each parser under DS9 because the exact pointer location and output dtype remain runtime contracts.

## Official DS9 Evidence

- [Gst-nvinfer, DS9 current docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html): `gst-nvinfer` supports detector, classifier, semantic segmentation, and instance segmentation network types; the DS9-specific new parser-adjacent feature called out in the table is oriented bounding boxes, not YOLO26/YOLO11/RF-DETR mask parsing.
- [Gst-nvinfer parser properties](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html): DS9 still lists `parse-bbox-func-name`, `parse-bbox-instance-mask-func-name`, `parse-segmentation-func-name`, and `custom-lib-path`; instance segmentation parsing remains custom-library driven when not using a built-in MaskRCNN path.
- [Using a Custom Model with DeepStream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html): NVIDIA still describes custom output parsing as the required path when model outputs need custom box/class/mask decoding.
- [DS9 `nvdsinfer_custom_impl.h`](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/sources_2includes_2nvdsinfer__custom__impl_8h.html): `CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE` and `CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE` use the same argument shapes as the audited parser code.
- [Gst-nvdspostprocess, DS9 current docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspostprocess.html): NVIDIA documents the tensor-only nvinfer setup as `output-tensor-meta=1` plus `network-type=100` to disable native parsing.
- [DS9 `NvDsInferTensorMeta`](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/structNvDsInferTensorMeta.html): tensor meta is attached when `output-tensor-meta` is true and exposes both host and device output buffer pointer arrays.
- [DS9 `NvDsInferContextInitParams`](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/struct__NvDsInferContextInitParams.html): `disableOutputHostCopy` remains documented as the flag where nvinfer returns GPU buffers for GPU-side post-processing.

## Local Evidence

- `DS9/config/infer.yaml` selects `pipelines/config_infer_primary_yolo11_seg.ini` as the default PGIE and enables the YOLO26 pose SGIE with tensor meta.
- `DS9/pipelines/config_infer_secondary_yolo26_pose.ini` is already tensor-only: `network-type=100`, `output-tensor-meta=1`, `disable-output-host-copy=0`, no `custom-lib-path`, and no `parse-bbox-func-name`.
- `DS9/pipelines/config_infer_secondary_depth_tracking_da2.template.ini` and `DS9/pipelines/config_infer_secondary_mapanything.ini` also use `network-type=100` with `output-tensor-meta=1`, so tensor-only nvinfer is already an accepted local DS9 config idiom.
- `docs/history/runtime/ds8/DS8_testing_guide.md` records the YOLO26 fused-output
  contract: a single `output0`, `disable-output-host-copy=1`, and bounded top-30
  mask composition.
- `docs/history/runtime/ds8/ds8_design_decisions.md` records why YOLO26-Seg was
  fused: avoid large proto-tensor host copies and keep mask composition on GPU.
- The same design log records RF-DETR parser semantics: people-only mapping to DeepStream `classId=0`, sigmoid person-logit scoring, and bbox-relative masks.

## Parser Calls

| Parser | Current role | DS9 call | Why |
| --- | --- | --- | --- |
| `nvdsinfer_yolo26_seg` | Parses fused `output0` rows into `NvDsInferInstanceMaskInfo`; supports host/device buffers and FP32/FP16. | Keep, rebuild for DS9. | DS9 has no built-in parser for this fused YOLO26 mask layout. The DS8 decision log ties this parser to the GPU-first fused-output contract. |
| `nvdsinfer_yolo11_seg` | Parses fused YOLO11 rows into `NvDsInferInstanceMaskInfo`; assumes host FP32 output. | Keep, but harden or constrain config. | DS9 does not replace the fused YOLO11 mask parser. If DS9 config uses `disable-output-host-copy=1` or FP16 outputs, port the YOLO26-style device/FP16 handling or explicitly keep host FP32 output for this parser. |
| `nvdsinfer_rfdetr_seg` | Parses `dets`, `labels`, and `masks`; maps person detections to DeepStream class `0`; supports device buffers. | Keep, rebuild for DS9. | RF-DETR’s sigmoid scoring, class remap, and bbox-relative mask crop are model-specific. Built-in MaskRCNN/YOLO detector support does not cover this. |
| `nvdsinfer_yolo26_pose` | No-op object detector parser that only clears detections; pose tensors are consumed from tensor meta elsewhere. | Remove from DS9 build/runtime plan, keep source only as historical reference until validation passes. | Local DS9 pose config already uses `network-type=100` and `output-tensor-meta=1`, which is the DS9-documented tensor-only shape. There is no DS9 config reference to this parser. |

## Likely DS9 Prep Edits

1. Build only the three segmentation parser libraries for the first DS9 bring-up: YOLO26-Seg, YOLO11-Seg, and RF-DETR-Seg. Do not include `libnvdsinfer_yolo26_pose.so` unless a DS9 runtime test proves `network-type=100` is rejected for this SGIE.
2. Materialize or add the missing DS9 primary PGIE config files referenced by `DS9/config/infer.yaml`, especially `pipelines/config_infer_primary_yolo11_seg.ini`, with DS9-local parser library paths and no machine-local absolute paths.
3. For each segmentation PGIE config, keep `network-type=3`, `output-instance-mask=1`, `custom-lib-path`, and the matching `parse-bbox-instance-mask-func-name`.
4. For YOLO26-Seg, keep the fused single-output model contract and revalidate `disable-output-host-copy=1` under DS9 with a parser smoke test that confirms device pointers, mask dimensions, and object counts.
5. For YOLO11-Seg, decide explicitly: either leave output host copy enabled and force FP32 host output, or port the YOLO26 device/FP16 branches before enabling `disable-output-host-copy=1`.
6. For RF-DETR-Seg, keep output names `dets`, `labels`, and `masks`; add a DS9 smoke test that checks person-logit scoring, class remap to `0`, and bbox-relative mask size.
7. Update parser comments/build notes from "DeepStream 8" to DS9-neutral wording during the implementation phase, after this report is reviewed.
8. Rebuild against installed DS9 headers only. Treat compile errors in `nvdsinfer_custom_impl.h` macros as the acceptance gate for ABI drift.

## Recommendation

Keep and rebuild the three instance-segmentation parsers for DS9. Remove the YOLO26 pose no-op parser from the DS9 build/runtime path because the DS9 tensor-only nvinfer config already covers that role. Do not remove any segmentation parser based on DS9 built-in support; the NVIDIA docs do not show built-in replacements for these custom output formats.
