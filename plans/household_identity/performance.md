# Household identity — performance contract

## Hot path

```text
NVMM frame
  -> YOLO26-m / NvDCF / Swin ReID SGIE
  -> NvDsInferTensorMeta
  -> native ReID metadata extraction
  -> one frame-cohort identity resolution
  -> tracking/world/OSD publication
```

Do not add CPU crop extraction, TorchReID inference, an appsink, full-frame CPU
copies, or unbounded embedding work.

## Current budgets

| Concern | Target |
| --- | --- |
| Native embedding extraction | under 0.3 ms per object |
| Identity resolve tick | under 2 ms typical |
| Overlap geometry work | under 0.2 ms |
| End-to-end effect | no material regression from the accepted native three-camera baseline |

The accepted application baseline is documented in `docs/runtime_baseline.md`:
roughly 29.9 FPS/camera live, and 25.45 FPS/camera in the 90.67-second non-July
recorded pressure run. These are comparison anchors, not a requirement to run a
benchmark for every identity edit.

For hot-path changes, measure only the affected counters plus one bounded live
or recorded run. Stop once identity output advances, no errors appear, and the
change does not create a gross FPS/CPU/GPU regression.
