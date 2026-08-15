# DS9 Archive

This directory preserves DS9-prep evidence that is no longer part of the active
DS9 build/runtime target.

## `nvdsinfer_yolo26_pose/`

Archived on 2026-05-10.

The YOLO26 pose parser is a no-op detector parser that clears detections. It is
not needed for the DS9 pose SGIE target because the active pose config is the
DS9-documented tensor-output shape: `network-type=100` plus
`output-tensor-meta=1`, with no `custom-lib-path` or parser function reference.

Keep the archived source as historical evidence only. Native DS9.1 consumes
pose tensors through metadata and `noesis_pose_meta_ext`; the selected pose
engine/parser contract is recorded in `DS9/asset_manifest.yaml`.
