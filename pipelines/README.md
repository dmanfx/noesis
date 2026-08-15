# Legacy pipeline directory

This root directory contains pre-DS9.1 configs and build remnants. It is not the
canonical pipeline/config authority and must not be used to launch, rebuild, or
repair the current application.

Current files live under:

- `DS9/config/` — canonical graph configuration;
- `DS9/pipelines/` — model/preprocess/parser configuration;
- `DS9/native/` and `DS9/gst-plugins/` — native DS9.1 sources;
- external artifact root selected by `NOESIS_DS9_ARTIFACT_ROOT` — accepted
  engines and realization.

Do not load the old YOLO11, OSNet, parser, tracker, or DeepStream 8 binaries
from this directory. Historical rationale is under `docs/history/`.
