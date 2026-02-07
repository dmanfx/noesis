# DS8 README for Agents

This document orients Codex agents to the DeepStream 8 stack in this repo.

## 1. High-Level Architecture

Canonical runtime path:

- **DS8 (canonical, under `noesis/`):**
  - `noesis/ds8_runtime.py` – main runtime harness (CLI, WS, REST).
  - `noesis/pipelines/ds8_pipeline.py` – builds DS8 Service Maker pipeline from `config/infer.yaml`.
  - `noesis/pipelines/hooks.py` – attaches metadata operators (intrinsics, MapAnything, analytics, exclusion).
  - `noesis/server/depth_api.py` – REST control for depth bursts.
  - `noesis/server/analytics_api.py` – REST API for analytics ROI management.
  - `noesis/telemetry/*` – depth, tracking, BEV publishers.
  - `noesis/metadata/*` – intrinsics and depth result schemas.

Deprecated pre-DS8 runtime paths have been decommissioned.
All runtime work should target the DS8 stack unless the user explicitly requests maintenance of deprecated artifacts.

Current defaults/baselines are summarized in `DS8_Baselines.md`.

## 2. Configuration Map

Key config files:

- `config/infer.yaml`
  - DS8 Service Maker / pipeline config:
    - Sources (URIs, per-source settings).
    - Streammux configuration.
    - Models (PGIE, MapAnything SGIE).
    - Analytics (nvdsanalytics-style stages/streams config).
    - Sinks for mosaic and other outputs.
    - Mosaic output (`mosaic_output`): RTSP output drives the WebRTC gateway when
      `mosaic_webrtc_enabled: true` (mosaic JPEG/WebSocket path removed in DS8).
      Default bitrate: 4000 kbps (H.264).
    - BEV settings (under `bev`):
      - `frame`: BEV coordinate frame (`camera_local` default; set `world` for world-frame BEV).
      - Env override: `NOESIS_BEV_FRAME=world` (takes precedence over config).
  - Calibration/extrinsics selection:
    - Default extrinsics: `config/camera_calibration.json`.
    - Env override: `NOESIS_CALIBRATION_EXTRINSICS=/path/to/file.json`.
- `config/cameras.yaml`
  - Camera intrinsics and optional extrinsics metadata used by `noesis/metadata/intrinsics.py` and BEV.
- V3DT baseline artifacts:
  - `config/infer_v3dt_baseline.yaml` (pipeline)
  - `config/v3dt/nvtracker_v3dt_baseline.yml` (tracker)
  - `config/v3dt/caminfo_baseline/` (camInfo directory)
  - `config/cameras_v3dt_baseline.yaml` (intrinsics)
  - `config/archive/calibration_v3dt_baseline.json` (extrinsics)
  - `config/dewarper_v3dt_baseline.txt` (family-room dewarper)
  - `config/analytics_exclude_baseline.ini` (exclusion ROIs)
- Runtime config files still consumed by DS8 components:
  - `pipelines/config_infer_primary_yolo11*.ini` – nvinfer model config.
  - `pipelines/config_nvdsanalytics_post.ini` – analytics stage config.
  - `pipelines/config_nvdsanalytics_exclude.ini` – exclusion ROI config.
  - `pipelines/config_osd.ini` – OSD behavior config.

When designing DS8 behavior, keep it semantically aligned with current config behavior and DS8 YAML contracts where possible.

## 3. DS8 Docs to Consult

When touching DS8 code or config, cross-check with the online DS8 docs. Primary references:

- **Service Maker for Python (Pipeline / Flow / advanced):**
  - Quick start: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_quick_start.html`
  - Flow APIs intro: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_into_to_flow_api.html`
  - Pipeline APIs intro: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_into_to_pipeline_api.html`
  - Advanced features: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_advanced_features.html`
- **Traditional app migration & plugins:**
  - Migration guide: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_traditional_app_migration.html`
  - Plugin overview: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_plugin.html`
- **Model and calibration tooling:**
  - Inference Builder: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Inference_Builder.html`
  - AutoMagicCalib: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AutoMagicCalib.html`
- **Repo metadata contracts (required for custom user meta):**
  - `docs/DS8_metadata_contracts.md` (includes **NvDsUserMeta copy/release semantics** and pose meta contract)
  - `docs/DS8_pose_stable_id_integration.md` (pose-assisted StableID integration + memory controls)

These URLs may change between DeepStream releases; if a link fails, search the DeepStream dev guide for the document title.

Use API names and config keys only when you can confirm them in the docs or the installed modules.

## 4. Entrypoints & Basic Commands

Typical DS8 runtime invocation (subject to project-specific overrides):

```bash
python3 noesis/ds8_runtime.py --pipeline-config config/infer.yaml --cameras-config config/cameras.yaml
```

This should:

- Build the DS8 pipeline from `config/infer.yaml`.
- Attach metadata hooks.
- Start the WebSocket server on the configured host/port.
- Optionally start REST APIs for depth and analytics when flags are provided.

Always verify CLI options in `noesis/ds8_runtime.py` before relying on them.

When running V3DT baselines, use the tracking-mode flag or env:

```bash
python3 noesis/ds8_runtime.py --tracking-mode v3dt
```

## 5. Planning & Execution Order

For migration work, follow this order:

1. Read `plans/DS8/ds8_master_work_orders.md`.
2. Open the relevant checklist(s) for the phase you are in (e.g., `ds8_migration_checklist_ds8_pipeline.md`).
3. Implement only the items in that phase unless the user directs otherwise.
4. Update the checklist and, when needed, `ds8_design_decisions.md` after completing each task.

## 6. Testing & Validation

See `docs/DS8_testing_guide.md` for concrete test commands and DS8 regression checks.
