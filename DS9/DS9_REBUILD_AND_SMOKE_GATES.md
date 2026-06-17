# DS9 Rebuild And Smoke Gates

This file lists the first rebuild/smoke work to run only after DS9 is installed
or a DS9 root is provided through `DS9_DEEPSTREAM_HOME`.

Do not run these against the live DS8 symlink.

## Prerequisite Gate

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 \
DS9_CUDA_HOME=/usr/local/cuda-13.1 \
  ./DS9/scripts/check_ds9_prereqs.sh
```

The script must report a DeepStream 9 root. If it reports missing DS9 headers,
stop; do not point it at `/opt/nvidia/deepstream/deepstream`.

## Native Bridge Rebuilds

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 \
DS9_CUDA_HOME=/usr/local/cuda-13.1 \
  ./DS9/scripts/build_all_native_ds9.sh
```

Expected outputs go under `DS9/native_extensions/` when using the launch
wrapper `DS9/scripts/build_native_extensions.sh`. The older
`DS9/scripts/build_all_native_ds9.sh` prep wrapper may still write historical
build products under `DS9/artifacts/native/`.

Required bridges:

- `noesis_pose_meta_ext`
- `noesis_depth_meta_ext`
- `noesis_depth_tracking_tensor_ext`
- `noesis_reid_meta_ext`
- `noesis_v3dt_meta_ext`
- `noesis_latency_ext` only if in-process latency remains required

## Parser Rebuilds

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 \
DS9_CUDA_HOME=/usr/local/cuda-13.1 \
  ./DS9/scripts/build_all_parsers_ds9.sh
```

Expected outputs go under each active parser directory in `DS9/pipelines/`.

Active DS9 parsers:

- `libnvdsinfer_yolo26_seg.so`
- `libnvdsinfer_yolo11_seg.so`
- `libnvdsparsebbox_yolo.so`
- `libnvdsinfer_rfdetr.so`
- `libnvdsinfer_rfdetr_seg.so`

The YOLO26 pose no-op parser is archived and must not be rebuilt for the first
DS9 target unless a DS9 runtime test proves tensor-only SGIE config is rejected.

## ROI Exclusion Plugin Rebuild

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 \
  ./DS9/scripts/build_nvdsroiexclude_ds9.sh
```

Expected output:

- `DS9/build/nvdsroiexclude/libgstnvdsroiexclude.so`

Prep validation after build:

```bash
GST_PLUGIN_PATH="$PWD/DS9/build/nvdsroiexclude:${GST_PLUGIN_PATH:-}" \
  gst-inspect-1.0 nvdsroiexclude
```

Do not install the plugin over the live DS8 runtime during prep.

## Runtime Smoke Gates After DS9 Cutover

- Pose: tensor-only YOLO26 pose SGIE emits object-level tensor metadata and
  `noesis_pose_meta_ext` extracts keypoints and attaches `NOESIS.POSE_FEATURES`.
  Passed in the core DS9 parity run through the DS9-native Service Maker/native
  extension path.
- Object depth: `noesis_depth_meta_ext` attaches/extracts `NOESIS.OBJECT_DEPTH`
  and mask extraction still matches `NvOSD_MaskParams`. Passed in the focused
  DS9 bridge smoke with object-depth GPU ROI copies, attaches, and `ok`
  payloads observed.
- DAv2 depth: `noesis_depth_tracking_tensor_ext` sees `out_buf_ptrs_dev` with
  `disable-output-host-copy=1`, aligns full-frame depth on GPU, and copies only
  object ROIs. Passed in the focused DS9 bridge smoke with native device-frame
  captures and zero core-path CPU-copy violations.
- ReID: `noesis_reid_meta_ext` extracts OSNet SGIE embeddings at the current
  StableID hook point. The core ReID stable-ID smoke passed; focused native
  extraction evidence also passed with `tensor_host_copies_total.reid > 0` and
  tracking payloads showing `embedding_present=true`.
- V3DT: `noesis_v3dt_meta_ext` extracts visibility, image foot, and 3D bbox
  metadata with current world-foot derivation. Still blocked for DS9 evidence:
  the DS9 smoke script now refuses to spawn without an explicit DS9 V3DT
  pipeline config, and the available V3DT configs are not DS9-scoped.
- ROI exclusion: `nvdsroiexclude` removes objects before tracker; excluded
  objects do not produce tracks, ReID embeddings, or pose work. Focused host MP4
  ROI pruning and hot restore passed.
- Alternate profiles: YOLO11 detect-only, YOLO26 detect-only `n/s/m/l/x`,
  YOLO26 segmentation `n/s/m`, RF-DETR segmentation `n/s/m`, and RF-DETR
  detect-only `n/s/m` materialization pass. Focused startup smokes have covered
  representative detector/segmentation variants.
- PyDS quarantine: default DS9 run does not require
  `NOESIS_DS9_ALLOW_PYDS_COMPAT=1` or
  `NOESIS_DS9_ALLOW_NATIVE_TENSOR_COMPAT=1`.

Do not treat artifact presence or preflight import success as behavior parity.
Object-depth, depth tensor, ReID, ROI prune/restore, YOLO11 detect-only,
YOLO26 detect-only, YOLO26-seg, RF-DETR-seg, and RF-DETR detect-only now have
focused DS9 evidence. V3DT still needs DS9-native staging and a focused smoke or
an explicit "not in current DS9 target" decision before full option-surface
parity is claimed.
