# DS8 Tracker/ReID A/B + Pipeline Builder GUI (Execution-Complete Plan)

## Summary
Build a DS8-native experiment framework that can A/B swap tracker backends and ReID models while preserving current downstream contracts (`stable_id`, occupancy, telemetry, OSD/trails), plus a lightweight local GUI for run control.
Plan-mode constraint: file writes are deferred in this mode; target artifact path is `plans/tracker_reid_ab_gui/plan_tracker_reid_ab_gui.md` (inside a dedicated folder under `./plans/{project}/`).

## Scope
- In scope:
- Baseline-mode tracker swapping: DeepStream `nvtracker` profiles and Python tracker profiles.
- ReID model selector for StableID + tracker appearance association.
- Lightweight "pipeline builder" GUI for process/env/profile management.
- Download/build scripts for tracker code and ReID engines.
- Validation gates for accuracy + runtime performance on RTX 3060.
- Out of scope:
- Hot-swapping tracker/ReID mid-run without restart.
- Changing public WS contract to expose raw tracker IDs.
- Reworking Menon/world-frame semantics beyond preserving existing behavior.

## Locked decisions (no implementer ambiguity)
- `--v3dt` always forces the current V3DT tracker path (`config/infer_v3dt_baseline.yaml` + `config/v3dt/nvtracker_v3dt_baseline.yml`) regardless of tracker profile selection.
- Baseline experiments are 2D-first; V3DT remains the protected 3D lane.
- Licensing policy: permissive-by-default, case-by-case exceptions allowed (record each exception in plan notes before enabling runtime use).
- Python tracker integrations are people-only (`class_id=0`) by default to protect FPS and reduce identity cross-class errors.
- ReID embeddings come from DS8 SGIE outputs (not ad-hoc per-tracker extra inference by default).

## Important API / interface additions
- CLI additions in `noesis/ds8_runtime.py`:
- `--tracker-profile`
- `--reid-profile`
- `--variant`
- `--gui` (optional local GUI launcher mode)
- Env additions:
- `NOESIS_TRACKER_PROFILE`
- `NOESIS_REID_PROFILE`
- `NOESIS_VARIANT`
- `NOESIS_GUI_HOST`, `NOESIS_GUI_PORT`, `NOESIS_GUI_TOKEN`
- New config files:
- `config/ab_variants.yaml`
- `config/tracker_profiles.yaml`
- `config/reid_profiles.yaml`
- GUI endpoints (new local API surface):
- `POST /gui/run/start`
- `POST /gui/run/stop`
- `GET /gui/run/status`
- `GET /gui/profiles`
- `POST /gui/presets/save`
- `GET /gui/presets`
- `GET /gui/logs/tail`

## Implementation plan (ordered, atomic)
1. Add planning workspace and governance artifacts.
- Create folder `plans/tracker_reid_ab_gui/`.
- Add plan doc, validation matrix, and runbook templates.
- Register design decisions in `plans/DS8/ds8_design_decisions.md`.

2. Add profile registries and variant schema.
- Implement schema for tracker/reid/pgie combinations with strict validation.
- Add default variants: `baseline_nvdcf_osnet`, `baseline_bytetrack_lightmbn`, `baseline_botsort_lightmbn`, `v3dt_default_locked`.

3. Implement tracker selector in runtime.
- Parse `--tracker-profile`, `--variant`, env equivalents.
- Resolve final profile with precedence: CLI > variant > env > defaults.
- Enforce V3DT guardrail override.

4. Implement `nvtracker` backend profile switching.
- Add low-level tracker YAML profiles under `config/trackers/`.
- Wire profile selection to `tracker.config-file` in effective pipeline YAML.
- Validate supported DS8 backend modes per profile.

5. Add baseline Python tracker path.
- Extend DS8 pipeline builder to allow `tracker.enable=false` in baseline mode.
- Add `PythonTrackAssigner` processor that writes `obj_meta.object_id` before downstream telemetry/stable-id logic.
- Keep current hooks contract unchanged for consumers.

6. Add Python tracker implementations (phased).
- Phase A: `ByteTrack`.
- Phase B: `BoT-SORT` using DS8 ReID embeddings for appearance matching.
- Phase C (optional): `StrongSORT` / `DeepOCSORT` only if they fit compute and dependency constraints.

7. Add ReID profile selector and embedding contract hardening.
- Replace hardcoded embedding assumptions in hooks with profile-driven fields (`embed_dim`, `layer_name`, `gie_id`).
- Ensure StableIDManager and tracker-association both consume the same embedding contract.
- Provide fallback safety behavior when a profile misconfigures output shape.

8. Add tracker/reid acquisition and engine build scripts.
- `scripts/fetch_trackers.sh` for pinned tracker sources.
- `scripts/build_reid_profiles.sh` for ONNX/TRT engine generation and profile registration.
- Record commit SHAs and licenses in an evidence manifest.

9. Implement lightweight pipeline builder GUI.
- New module `noesis/server/pipeline_builder_gui.py` + static page.
- Features: dropdown selectors, free-form env editor, start/stop/restart, status PID/uptime, log tail.
- Save/load named presets under `~/.noesis/presets/`.
- Bind localhost by default; optional token auth.

10. Add observability and comparison outputs.
- Stamp run metadata (`tracker_profile`, `reid_profile`, `variant`, `tracking_mode`) into diagnostics.
- Add summary script for A/B output metrics.

11. Execute staged validation and promote defaults.
- Run short smoke tests per profile, then 5-10 minute benchmark windows.
- Promote only variants that meet correctness and performance gates.
- Keep a rollback table mapping variant -> known-good fallback.

## Tracker shortlist and recommended ReID pairings
- `nvdcf_indoor_occlusion` -> `osnet_x0_5` (default speed/stability) or `lightmbn` (candidate faster appearance model).
- `py_bytetrack` -> `osnet_x0_5` or `lightmbn` (StableID-focused ReID; tracker itself motion/IoU-first).
- `py_botsort` -> `lightmbn` preferred, fallback `osnet_x0_5`.
- `py_strongsort` / `py_deepocsort` (optional lane) -> `osnet_x1_0` or `lightmbn` depending FPS.
- Research-only/offline lane -> `transreid` or CLIP-based ReID profiles (do not make realtime default on 3060).

## Test cases and acceptance scenarios
- Functional contract tests:
- `stable_id` present for people tracks and never replaced by raw tracker IDs in client payloads.
- Occupancy counts and zone transitions remain coherent across profile swaps.
- V3DT lane retains `bbox3d`/world fields with `--v3dt`.
- Integration tests:
- Start/stop lifecycle from GUI does not orphan DS8 runtime processes.
- Variant/env precedence works exactly as specified.
- Misconfigured profiles fail fast with actionable errors.
- Performance gates:
- Maintain dashboard target near 30 FPS in baseline default variant on current 3-camera setup.
- Track latency and dropped-frame indicators per profile.
- Regression gates:
- No crash in `hooks.py` embedding extraction when embedding dims differ by profile.
- No breakage of ReID alias/soft-merge APIs.

## Rollout and rollback
- Stage 1: enable profile framework with `nvtracker`-only variants.
- Stage 2: enable `py_bytetrack` experimental flag.
- Stage 3: enable `py_botsort` with ReID association.
- Stage 4: expose all promoted variants in GUI.
- Rollback: single env or CLI override to `--variant baseline_nvdcf_osnet` or `--v3dt`.

## Assumptions and defaults
- Current canonical runtime entry remains `noesis/ds8_runtime.py`.
- DS8 metadata/public contracts remain authoritative and unchanged.
- Profile downloads/builds are pin-by-commit and documented.
- Case-by-case license exceptions are explicitly documented before runtime enablement.
- Artifact destination when file writing is allowed: `plans/tracker_reid_ab_gui/plan_tracker_reid_ab_gui.md`.

## Source shortlist used for the "cutting-edge" pass
- ByteTrack: https://github.com/FoundationVision/ByteTrack
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
- Deep-OC-SORT: https://github.com/GerardMaggiolino/Deep-OC-SORT
- StrongSORT: https://github.com/dyhBUPT/StrongSORT
- BoxMOT (multi-tracker wrapper): https://github.com/mikel-brostrom/boxmot
- OSNet / Torchreid: https://github.com/KaiyangZhou/deep-person-reid
- LightMBN: https://github.com/jixunbo/LightMBN
- FastReID (deployment tooling): https://github.com/JDAI-CV/fast-reid
- TransReID: https://arxiv.org/abs/2102.04378
