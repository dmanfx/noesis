# AGENTS.md - plans/DS8/v3dt (V3DT Planning)

This folder contains active DS8 V3DT planning docs (SV3DT/MV3DT) for work under `noesis/`.

## Policy precedence

- Inherit repo-wide rules from `AGENTS.md` and planning-process rules from `plans/AGENTS.md`.
- This file adds only V3DT-specific constraints for this subtree.
- Within `plans/DS8/v3dt/`, this file overrides parent `AGENTS.md` where guidance conflicts.
- Files under `docs/history/` and `plans/*/archive/` are archival and non-normative unless explicitly requested.

## V3DT-specific operating rules

1. **DS8 only**
   - Do not route DS8 failures through deprecated pre-DS8 paths.
   - Only touch deprecated/pre-DS8 code if the user explicitly asks.

2. **No fallbacks without explicit user approval**
   - Do not introduce or rely on fallback paths, degraded modes, substitute algorithms, alternate calibration flows, or "temporary" V3DT workarounds unless the user explicitly asks for them or agrees after you discuss the blocker.
   - If the primary V3DT path fails, surface the failure clearly and stop instead of masking it with a fallback.

3. **V3DT source of truth**
   - Use `plans/DS8/v3dt/integration_plan.md`, `plans/DS8/v3dt/research_notes.md`, and `plans/DS8/v3dt/work_order.md` as the V3DT implementation spec.
   - Keep planning docs concise; avoid embedding large code blocks where a short pointer is sufficient.

4. **External contract stability**
   - Keep `stable_id` as the user-visible identity in WS/BEV/UI payloads.
   - Treat tracker internal IDs (including MV3DT IDs) as internal implementation details.

5. **Doc-backed DS8 API usage**
   - Use only verified Service Maker / DeepStream APIs and config keys.
   - Verify APIs against `docs/DS8_README_FOR_AGENTS.md`, official DeepStream docs, or installed modules.

6. **DS8 implementation guardrails**
   - Keep DS8 GPU-first behavior; do not add new DS8 appsink/CPU branches.
   - Under `noesis/`, prefer Service Maker / Flow APIs and DS8 metadata operators instead of new GStreamer pad-probe patterns.
   - Fail loudly on DS8 errors; do not silently degrade into deprecated runtime behavior.

7. **Progress discipline**
   - Update relevant checklist items (`[ ]` to `[x]`) with dated validation notes as work completes.
   - Record non-trivial design decisions in `plans/DS8/ds8_design_decisions.md`.

## Locked baseline (quarantine everything else)

Use this baseline set for reproducible V3DT runs:

- Pipeline: `config/infer_v3dt_baseline.yaml`
- Cameras: `config/cameras_v3dt_baseline.yaml`
- Calibration: `config/archive/calibration_v3dt_baseline.json`
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
- Keep `config/ply_alignment.json` `units.s_obj_to_m=1.0`
- Keep PGIE letterboxing disabled (`maintain-aspect-ratio=0`, `symmetric-padding=0`)

See `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md` for the locked baseline notes and no-go list.

## Recovery checklist (if `config/v3dt/` is missing)

Use coded defaults in `noesis/ds8_runtime.py`, then regenerate camInfo in the baseline directory.

Defaults:

- Pipeline: `config/infer_v3dt_baseline.yaml`
- Cameras: `config/cameras_v3dt_baseline.yaml`
- Tracker: `config/v3dt/nvtracker_v3dt_baseline.yml`
- camInfo dir: `config/v3dt/caminfo_baseline/`

Commands (locked baseline):

```bash
python3 scripts/generate_v3dt_caminfo.py \
  --pipeline-config config/infer_v3dt_baseline.yaml \
  --cameras-config config/cameras_v3dt_baseline.yaml \
  --calibration config/archive/calibration_v3dt_baseline.json \
  --output-dir config/v3dt/caminfo_baseline \
  --model-height 2.2 \
  --model-radius 0.35 \
  --target-width 1920 \
  --target-height 1080

python3 scripts/sanity_check_v3dt_calibration.py \
  --pipeline-config config/infer_v3dt_baseline.yaml \
  --cameras-config config/cameras_v3dt_baseline.yaml \
  --calibration config/archive/calibration_v3dt_baseline.json
```

For intentional non-baseline experiments, set a different `--calibration` path explicitly (for example `config/camera_calibration.json`) and record that run as non-baseline.
