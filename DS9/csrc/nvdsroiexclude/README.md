# `nvdsroiexclude` DS9 Prep Source

This source reconstructs the Noesis ROI exclusion plugin for the DS9 prep
workspace. It is intentionally source-controlled under `DS9/` because the live
checkout only had build artifacts for the existing plugin.

Behavior:

- GStreamer element name: `nvdsroiexclude`.
- In-place metadata transform.
- Reads DeepStream-style ROI INI groups such as
  `[roi-filtering-stream-0]`.
- Removes `NvDsObjectMeta` before tracker when all four bbox corners are inside
  an enabled ROI polygon.
- Uses the official DeepStream metadata API
  `nvds_remove_obj_meta_from_frame`.
- Optionally emits ROI outline display metadata when `osd-mode != 0`; it does
  not draw on CPU pixels.

Build after DS9 is installed:

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 \
  ./DS9/scripts/build_nvdsroiexclude_ds9.sh
```

The script refuses DS8 headers. Do not install the built plugin over the live
DS8 runtime during prep.
