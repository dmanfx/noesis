# Codex Agent Prompt – Implement SV3DT + MV3DT in DS8 Noesis

You are a Codex agent implementing **SV3DT** and **MV3DT** integration into the **DS8 canonical stack** under `noesis/` in `/home/mayor/Noesis_Devel`.

## Current baseline (do not regress)

See `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md` for the locked
SV3DT baseline (configs, pitch values, model height, and confirmed no-gos).
All new work should start from that baseline and keep its constraints intact.

## Hard constraints (must follow)

1. **DS8 only**: Do not modify deprecated pre-DS8 pipeline code unless explicitly requested.
2. **No deprecated-stack fallback**: If DS8 MV3DT/SV3DT cannot be made to work, fail loudly (logs + docs) and stop; do not route through removed runtime paths.
3. **No guessing of DS8/DeepStream APIs**:
   - Only use DeepStream/Service Maker methods and config keys you can verify in:
     - installed Python modules (`pyservicemaker`, `pyds`)
     - official DeepStream docs
     - NVIDIA reference apps (deepstream_reference_apps)
4. **GPU-first policy**: Do not add new appsink/CPU branches to DS8. Decode → inference → tracking → analytics → OSD stay on GPU/NVMM.
5. **Progress discipline**:
   - Before DS8 code changes, read `plans/DS8/ds8_master_work_orders.md` and relevant checklists under `plans/DS8/`.
   - Update checklists as you complete items (checkbox + dated note).
   - Record non-trivial design choices in `plans/DS8/ds8_design_decisions.md`.

## Goal (what “done” means)

Implement the plan in:
- `plans/DS8/v3dt/integration_plan.md`
- `plans/DS8/v3dt/research_notes.md`
- `plans/DS8/v3dt/work_order.md`

Specifically:
- SV3DT runs on all cameras and emits 3D bbox meta (`NVDS_OBJ_3D_META`) for people.
- MV3DT runs only for the overlap pair (kitchen ↔ family-room) via an MQTT vision-neighbor graph.
- DS8 telemetry/BEV uses SV3DT/MV3DT 3D outputs when available.
- External contracts remain stable: `stable_id` is the only user-visible identity (no raw tracker IDs in WS payloads/UI).

## Inputs you must confirm early (blockers)

1. Verify the current stream/camera mapping (do not assume it’s unchanged if configs were edited):
   - source 0 → `living-room` (camera id `0`) `rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?`
   - source 1 → `kitchen` (camera id `1`) `rtsp://192.168.3.214:7447/qt3VqVdZpgG1B4Vk?`
   - source 2 → `family-room` (camera id `2`) `rtsp://192.168.3.214:7447/4qWTBhW6b4nLeUFE?`
   - streammux output is 1920×1080; family-room intrinsics originate at 1280×720 and must be scaled when building camInfo.
2. Confirm global calibration is real:
   - Current `config/camera_calibration.json` appears camera-local (centers x≈0,z≈0).
   - MV3DT requires a shared global world frame (meters, Y-up).
   - If global extrinsics are missing, implement only SV3DT “bbox recovery” mode and explicitly disable MV3DT, with clear logs/docs.
3. Confirm MQTT broker settings for MV3DT:
   - host/port (assume `127.0.0.1:1883` unless user overrides)
   - auth requirements (if any)

## Implementation outline (do in this order)

0. **Pre-calibration staging (Phase 0a; do first if Menon extrinsics aren’t ready yet)**
   - Create the `config/v3dt/` config skeleton (SV3DT + MV3DT configs + MQTT configs + pub/sub graph).
   - Integrate DS8 hook support to parse `NVDS_OBJ_3D_META` (`pyds.NvDsObj3DBbox`) when present and publish additive 3D fields.
   - Provision BodyPose3DNet assets/engine and wire `PoseEstimator` into the tracker config.
   - Keep MV3DT disabled and treat any 3D coordinates as **camera-local** until global calibration is real.
1. **Global calibration (Phase 0)**
   - Implement or integrate a workflow to produce correct shared-world extrinsics in `config/camera_calibration.json` (world→camera, column-major 16 floats).
   - Validate by computing camera centers and checking separation in X/Z.
2. **Generate camInfo YAMLs**
   - Create `config/v3dt/camInfo_*.yml` with `projectionMatrix_3x4` (preferred; matches NVIDIA sample) or `projectionMatrix_3x4_w2p`, and `modelInfo` in units consistent with the chosen camInfo scale (meters or centimeters).
   - Ensure camInfo list order matches source order (pad index).
3. **Create SV3DT tracker config**
   - Create `config/v3dt/nvtracker_sv3dt.yml`:
     - `StateEstimator.stateEstimatorType: 3`
     - `ObjectModelProjection.cameraModelFilepath: [...]`
     - `outputFootLocation: 1` (required for `NVDS_OBJ_3D_META`)
     - `outputVisibility: 1` (even if we can’t read it via pyds yet)
     - add `PoseEstimator` (BodyPose3DNet) with `poseInferenceInterval: -1` initially
4. **Plumb 3D meta into DS8 hooks**
   - In `noesis/pipelines/hooks.py`, extract `NVDS_OBJ_3D_META` and attach:
     - `bbox3d`, `velocity3d`, and a best-effort `world` position.
   - Update BEV to prefer SV3DT/MV3DT world positions when present.
5. **Create MV3DT config and MQTT neighbor graph**
   - Create `config/v3dt/nvtracker_mv3dt.yml` with `MultiViewAssociator` + `Communicator` (MQTT).
     - Do not invent config keys: start from `/tmp/deepstream_reference_apps/deepstream-tracker-3d-multi-view/config_templates/config_tracker.yml` and adapt paths/values.
   - Create `config/v3dt/pub_sub_info_config_0.yml`:
     - kitchen subscribes to family-room; family-room subscribes to kitchen; living-room subscribes to none.
   - Create `config/v3dt/mqtt_proto_adaptor.txt` in the DeepStream message-broker adaptor format.
   - Set `nvstreammux.sync-inputs=1` (and other time-sync properties only if verified via `gst-inspect-1.0 nvstreammux`).
6. **Identity integration**
   - Keep `stable_id` as master identity (WS/BEV/UI).
   - Add internal `mv3dt_id` and optionally use it to strengthen StableID assignment (people-only).
7. **Docs + checklists + validation**
   - Update DS8 checklists and add dated notes.
   - Update contracts docs for any additive fields.
   - Validate using `docs/DS8_testing_guide.md` patterns; add a small, focused smoke test if none exists for 3D meta extraction.

## Reference material you may rely on (verified sources)

- DeepStream nvtracker docs:
  - https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
- DeepStream MV3DT docs:
  - https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html
- NVIDIA reference apps (config ground truth):
  - `deepstream-tracker-3d` and `deepstream-tracker-3d-multi-view` under https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps
  - Local copy used for this plan:
    - `/tmp/deepstream_reference_apps/deepstream-tracker-3d-multi-view/config_templates/config_tracker.yml`
- Installed DeepStream sources:
  - `/opt/nvidia/deepstream/deepstream-8.0/sources/gst-plugins/gst-nvtracker/nvtracker_proc.cpp` (meta attachment)
  - `/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream-test5/deepstream_test5_app_main.c` (3D meta extraction pattern)
