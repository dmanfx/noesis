# `nvdsroiexclude` native DeepStream 9.1 source

This is the canonical source for Noesis' in-place ROI-exclusion element. It
removes object metadata before tracking when all four bounding-box corners are
inside an enabled ROI polygon. It uses DeepStream metadata APIs and never maps
or copies frame pixels.

Behavior:

- element name: `nvdsroiexclude`;
- input: DeepStream ROI INI groups such as `[roi-filtering-stream-0]`;
- mutation: `nvds_remove_obj_meta_from_frame` before NvDCF;
- optional OSD outline metadata when `osd-mode != 0`.

Build against the installed native DS9.1 SDK:

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.1 \
  ./DS9/scripts/build_nvdsroiexclude_ds9.sh
```

Load-check the affected plugin directly after a rebuild. Do not install or load
the archived root DS8 binary.
