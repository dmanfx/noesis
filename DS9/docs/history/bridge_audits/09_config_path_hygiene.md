# DS9 Config Path Hygiene

Date: 2026-05-11

Scope: DS9 prep mirror only. No files in the live DS8 runtime path were changed.

## Completed

- Copied the active runtime config files referenced by `config/infer.yaml` into
  the DS9 prep mirror:
  - `pipelines/config_infer_primary_yolo11_seg.ini`
  - `pipelines/config_infer_primary_yolo26_seg.template.ini`
  - `pipelines/config_infer_primary_rfdetr_seg.ini`
  - `pipelines/config_preproc.ini`
  - `config/config_nvdsanalytics_post.ini`
  - `config/nvtracker.yaml`
  - `config/depth_registration.json`
  - `config/dewarper_g3_instant_charuco_1080.txt`
  - `build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini`
- Converted active DS9 prep configs away from machine-local `/home/mayor/...`
  model paths.
- Converted active DeepStream library references away from the mutable
  `/opt/nvidia/deepstream/deepstream` symlink and toward explicit
  `/opt/nvidia/deepstream/deepstream-9.0` paths where a DeepStream library path
  is required by config.
- Updated stale DS8-only comments in active DS9 prep config/parser files so
  future review is not misled about the intended target.
- Extended `scripts/run_static_prep_checks.sh` so the above config set is now a
  required part of the DS9 prep surface, and so active configs fail on:
  - machine-local `/home/mayor` paths
  - mutable DeepStream symlink library paths

## Validation

Passed:

```bash
./DS9/scripts/run_static_prep_checks.sh
```

Manual hygiene scan:

```bash
rg -n "/home/mayor|/opt/nvidia/deepstream/deepstream/lib|deepstream-8|DS8" \
  DS9/config DS9/pipelines DS9/build -g '!**/archive/**'
```

Result: no active DS9 prep config matches.

## Remaining Blocker

Rebuilds and DS9 import/runtime smoke tests still require a real DS9 install
root plus matching CUDA/toolchain. Until then, the prep scripts intentionally
stop before compiling against the active DS8 symlink or current CUDA root.
