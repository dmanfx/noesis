# AGENTS.md - plans/DS8/v3dt (V3DT Planning)

This folder contains DS8 V3DT (SV3DT/MV3DT) planning docs. Use it to guide and record work on the DS8 canonical stack under `noesis/`.

## Core truths (do not violate)

1. **Respect the DS7 vs DS8 split**
   - Do not silently route DS8 failures through the DS7 stack. If DS8 code cannot be made to work, fail loudly in logs/docs and stop.
   - Only modify DS7 (`deepstream_video_pipeline.py` and friends) if the user explicitly asks for DS7 changes.
   - Prefer implementing new functionality in the DS8 stack under `noesis/`.

2. **Follow the DS8 plan and checklists**
   - Before changing DS8-related code, read:
     - `plans/DS8/ds8_master_work_orders.md`
     - The relevant checklist(s) under `plans/DS8/` (e.g., `ds8_migration_checklist_ds8_pipeline.md`, `ds8_migration_checklist_hooks.md`).
   - Treat these documents as the authoritative description of what to do and in what order. If code and docs disagree, surface the discrepancy in a doc update rather than guessing.

3. **No guessing of DeepStream/Service Maker APIs**
   - For any call into DeepStream 8 or Service Maker (e.g., `pyservicemaker`, `pyds`, `nvinfer`, `nvdsanalytics`):
     - Use only methods, properties, and config keys that you can verify in the official docs or the installed Python modules.
     - Do not invent new method names or config keys.
   - DS8 docs to consult (non-exhaustive):
     - `docs/DS8_README_FOR_AGENTS.md`
     - Service Maker for Python docs (Pipeline APIs, Flow APIs, Advanced Features)
     - Inference Builder docs
     - DS8 analytics / nvdsanalytics YAML docs

4. **GPU-first policy for DS8**
   - Keep decode, pre-processing, inference, tracking, analytics, tiling, and OSD on the GPU (NVMM) wherever DeepStream supports it.
   - Convert to CPU only at the very edge when required (e.g., final JPEG encoding if not handled by `nvjpegenc`, BEV composition, or explicit ReID crops).
   - Do not add new appsink/CPU-based branches to the DS8 path; those patterns belong only to legacy DS7 code.

5. **Testing and validation**
   - When you make DS8 changes, use `docs/DS8_testing_guide.md` as your reference for how to validate behavior (unit tests, functional tests, DS7 vs DS8 parity checks).
   - Prefer small, focused validations rather than only end-to-end tests.
   - For interface/contract changes, also consult:
     - `docs/DS8_api_contracts_ws.md`
     - `docs/DS8_api_contracts_rest.md`
     - `docs/DS8_metadata_contracts.md`

6. **Progress documentation**
   - When completing work defined in `plans/DS8/*.md`:
     - Update the relevant checklist item from `[ ]` to `[x]`.
     - Add a short, dated note under that item describing what you changed and how you validated it.
   - Do this often so the plans reflect near real-time progress.
   - Do not add progress notes into code comments unless the user explicitly asks; keep them in the planning docs.

7. **Design decisions**
   - When you make a non-trivial design choice for DS8 (API shape, data schema, etc.), capture it in `plans/DS8/ds8_design_decisions.md` with a short rationale and any doc references you relied on.

## Planning docs in this folder

- Treat `plans/DS8/v3dt/integration_plan.md`, `plans/DS8/v3dt/research_notes.md`, and `plans/DS8/v3dt/work_order.md` as the V3DT-specific source of truth.
- Keep planning docs concise; avoid large code blocks and prefer short pointers to files/lines when needed.
- Do not edit historical archives under `plans/archive/` or `plans/DS8/archive/` unless explicitly requested.

## DS8 implementation guardrails (if you touch code)

- Do not use GI/GStreamer to construct DS8 pipelines under `noesis/` (except the intentional mosaic WebRTC gateway).
- Prefer Service Maker / Flow APIs for pipeline construction and metadata extraction.
- Do not add new pad probes or appsinks in DS8 code; use Service Maker metadata operators instead.
- Favor clear, actionable logs; fail fast in DS8 production paths rather than silently degrading into DS7 behavior.

## V3DT-specific reminders

- DS8-only: SV3DT/MV3DT work is expected to live under `noesis/` and `config/v3dt/`.
- Confirm stream/camera mapping, calibration frame, and MQTT settings early; if shared-world calibration is missing, disable MV3DT and log loudly.
- Keep external contracts stable: `stable_id` remains the user-visible identity in WS/BEV/UI payloads.

## Calibration and camInfo truths (lock these in)

- Tracker sees 1920x1080 (streammux output). camInfo intrinsics must be scaled to 1920x1080.
- Family-room raw is 1280x720, dewarper outputs 1280x720, then streammux scales to 1920x1080 (scale = 1.5).
- Calibration JSON uses E as world-to-camera, column-major.
- World frame is X/Z on floor with Y up; SV3DT samples often assume Z up, so a world_axes=xzy swap is reasonable.
- Alignment scale must stay in meters: `config/ply_alignment.json` `units.s_obj_to_m=1.0`.
- PGIE letterboxing must be disabled (`maintain-aspect-ratio=0`, `symmetric-padding=0`).

## Golden baseline (quarantine everything else)

Use only this minimal set of files to get a clean, reproducible baseline:

- Pipeline: `config/infer_v3dt_baseline.yaml`
- Camera intrinsics: `config/cameras_v3dt_baseline.yaml`
- Calibration extrinsics: `config/archive/calibration_v3dt_baseline.json`
- Dewarper: `config/dewarper_v3dt_baseline.txt`
- Tracker config: `config/v3dt/nvtracker_v3dt_baseline.yml`
- camInfo dir: `config/v3dt/caminfo_baseline/`
- Tracking mode: `v3dt` (`NOESIS_TRACKING_MODE=v3dt` or `--tracking-mode v3dt`)

Baseline expectations:

- `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p`
- `NOESIS_V3DT_CAMINFO_INVERT_E=0`
- `NOESIS_V3DT_CAMINFO_Y_FLIP=1`
- `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`
- camInfo `modelInfo.height=2.2`, `radius=0.35`

Everything else is quarantine material until the baseline works.
See `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md` for shortfalls
and confirmed no-go items.
