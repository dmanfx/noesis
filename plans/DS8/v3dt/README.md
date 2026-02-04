# DS8 V3DT (SV3DT + MV3DT) Integration – Entry Point

This folder contains DS8-focused planning docs for integrating **SV3DT** (Single‑View 3D Tracking) and **MV3DT** (Multi‑View 3D Tracking) into the canonical DeepStream 8 stack under `noesis/`.

**Audience:** future implementation agent (Codex) + project owner review.

## What you asked for (site context)

- Cameras:
  - **Kitchen** and **Family‑room** overlap (facing each other) → MV3DT target pair.
  - **Living‑room** is to the right of Kitchen and appears adjacent / non‑overlap → handled via StableID + geometry gating, not MV3DT neighbor graph initially.
- Priorities (your order): **A** single‑camera robustness → **B** overlap global IDs → **C** high‑quality 3D outputs → **D** non‑overlap handoff.
- World frame intent: **meters**, **Y‑up**, BEV should be accurate and consistent.
- Time sync: likely NTP; willing to set `nvstreammux.sync-inputs=1`.
- Hardware: single machine + single GPU.
- MQTT: local Mosquitto is acceptable (and already in use for HomeSeer integration).
- Targets: **people**; dogs only if low complexity.
- Tracking should use **pose** (enable BodyPose3DNet inside the tracker).

## Documents

- `plans/DS8/v3dt/research_notes.md` – SV3DT/MV3DT deep dive (how it works, configs, outputs/meta, Python access).
- `plans/DS8/v3dt/integration_plan.md` – concrete DS8 integration plan (phases, files, acceptance criteria, validation).
- `plans/DS8/v3dt/work_order.md` – implementation task breakdown with acceptance criteria (DS8-specific).
- `plans/DS8/v3dt/enhancements_plan.md` – follow-on ideas once MV3DT is stable (occupancy dedupe, drift detection, etc.).
- `plans/DS8/v3dt/codex_agent_prompt.md` – copy/paste prompt to drive an implementation agent.

## Key recommendation (high-level)

- Use **SV3DT on all cameras** (per-camera robustness + 3D state estimator).
- Use **MV3DT only for the true-overlap pair** (**kitchen ↔ family‑room**) at first (vision neighbors must overlap).
- Keep **`stable_id` (StableIDManager)** as the **only** user-visible identity (per `plans/DS8/ds8_id_contract_v2.md`), and treat MV3DT’s global tracker ID as an **internal hint/constraint**, not the public ID.
  - Note: StableID can now also consume YOLO26 pose SGIE ratio features as a secondary signal (see `docs/DS8_pose_stable_id_integration.md`); this is distinct from the tracker’s BodyPose3DNet PoseEstimator.

## Current locked baseline (2026-01-22)

See the full write-up in `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`.

Quick reference:

- Pipeline: `config/infer_v3dt_baseline.yaml`
- Tracker: `config/v3dt/nvtracker_v3dt_baseline.yml`
- CamInfo dir: `config/v3dt/caminfo_baseline/`
- Calibration (baseline): `config/archive/calibration_v3dt_baseline.json`
- Cameras config: `config/cameras_v3dt_baseline.yaml`
- Use tracking mode `v3dt` (`--tracking-mode v3dt` or `NOESIS_TRACKING_MODE=v3dt`) with these configs.

Confirmed no-go items (see full list in the status summary):

- Do not enable PGIE aspect ratio / symmetric padding.
- Do not change `config/ply_alignment.json` `units.s_obj_to_m` away from `1.0`.
- Do not use model height >= 2.4m (family-room regression).

## Default-baseline recovery (if config/v3dt is missing)

The coded defaults are the source of truth for v3dt runs. When `config/v3dt/`
is deleted or stale, regenerate camInfo using the default baseline inputs
defined in `noesis/ds8_runtime.py`.

**Defaults (coded in `noesis/ds8_runtime.py`):**

- v3dt pipeline: `config/infer_v3dt_baseline.yaml`
- v3dt cameras: `config/cameras_v3dt_baseline.yaml`
- v3dt tracker: `config/v3dt/nvtracker_v3dt_baseline.yml`
- camInfo dir: `config/v3dt/caminfo_baseline/`

**Recovery steps:**

```bash
python3 scripts/generate_v3dt_caminfo.py \
  --pipeline-config config/infer_v3dt_baseline.yaml \
  --cameras-config config/cameras_v3dt_baseline.yaml \
  --calibration config/camera_calibration.json \
  --output-dir config/v3dt/caminfo_baseline \
  --model-height 2.2 \
  --model-radius 0.35 \
  --target-width 1920 \
  --target-height 1080

python3 scripts/sanity_check_v3dt_calibration.py \
  --pipeline-config config/infer_v3dt_baseline.yaml \
  --cameras-config config/cameras_v3dt_baseline.yaml \
  --calibration config/camera_calibration.json
```

## Critical dependency (must be solved early)

Your current `config/camera_calibration.json` extrinsics appear to be **camera-local** (camera centers all near x≈0,z≈0). **MV3DT requires a shared global world frame** across cameras. The plan treats “global calibration” as Phase 0.

## What we can do before Menon extrinsics

We can still get most of the integration “mechanics” ready without a shared-world calibration:

- Add DS8 hook support to extract SV3DT/MV3DT 3D bbox meta (`NVDS_OBJ_3D_META` / `NvDsObj3DBbox`) and publish additive 3D fields.
  - Note: Service Maker Python `ObjectMetadata` does not expose `obj_user_meta_list`, so DS8 requires the native bridge module `noesis_v3dt_meta_ext` (build: `scripts/build_noesis_v3dt_meta_ext.sh`).
- Prepare SV3DT/MV3DT tracker configs + MQTT neighbor graph configs (but keep MV3DT disabled until global calibration exists).
- Provision BodyPose3DNet assets/engine and enable pose in the tracker config (helps SV3DT robustness even in single-camera mode).
- Gate `nvstreammux.sync-inputs=1` behind a config toggle for later MV3DT experiments (only useful once timestamps + overlap are validated).

We should *not* enable MV3DT fusion/ID propagation or interpret 3D positions as a shared house/world frame until Menon (or another tool) provides global extrinsics.

## Quick validation (SV3DT meta plumbing)

To validate that DS8 is extracting `NVDS_OBJ_3D_META` and publishing `bbox3d` in WS tracking telemetry:

- `python3 scripts/sv3dt_meta_smoke_test.py`
  - Defaults to `config/infer_v3dt_sample.yaml` (offline Retail02 clip) when present.
  - For live RTSP, run `python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_sv3dt.yaml` and ensure a person is visible.
    - Note: SV3DT requires a patched DeepStream `nvtracker` plugin to avoid an upstream host-RAM leak; `noesis/ds8_runtime.py` auto-builds/auto-loads it for V3DT tracker configs (see `plans/DS8/v3dt/oom_killed_infer_v3dt_debug.md`).
- Note: `noesis/ds8_runtime.py` only auto-regenerates `config/v3dt/camInfo_*.yml` when `NOESIS_V3DT_AUTOGEN_CAMINFO=1` (default is `0`).
  - For the locked baseline, keep `NOESIS_V3DT_AUTOGEN_CAMINFO=0` and use the pre-generated camInfo dir in the tracker config (see summary above).

## Unit Convention and Calibration

**Canonical unit: METERS** (as of 2025-12-30, V3DT-H01)

### Calibration inputs

| File | Description | Units |
|------|-------------|-------|
| `config/camera_calibration.json` | Per-camera `E` matrix (world→camera, column-major 4×4) | Meters |
| `config/cameras.yaml` | Intrinsics models (`fx`, `fy`, `cx`, `cy`) | Pixels at native resolution |
| `config/ply_alignment.json` | Floor alignment (`floor_y`, `s_obj_to_m`) | Meters |

### camInfo generation

The `config/v3dt/camInfo_*.yml` files are auto-generated from calibration by `noesis/ds8_runtime.py` at startup (when a V3DT tracker config is selected).

**Runtime expectation for SV3DT in this repo:**

Set these in the environment that launches `noesis/ds8_runtime.py` (systemd `Environment=...`, docker `environment:`, or a shell export before running).

```bash
export NOESIS_V3DT_AUTOGEN_CAMINFO=1
export NOESIS_V3DT_CAMINFO_WORLD_SCALE=1     # generate camInfo in meters (default)
export NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p
export NOESIS_V3DT_CAMINFO_INVERT_E=0
export NOESIS_V3DT_CAMINFO_Y_FLIP=1
export NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy
```

**Locked baseline (2026-01-22, all cameras tracking):**

```bash
export NOESIS_V3DT_AUTOGEN_CAMINFO=0
export NOESIS_V3DT_CAMINFO_WORLD_SCALE=1
export NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p
export NOESIS_V3DT_CAMINFO_INVERT_E=0
export NOESIS_V3DT_CAMINFO_Y_FLIP=1
export NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy
```

This baseline restores tracking for all three cameras; see `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md` for the latest forensics results and shortfalls.

**Key environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NOESIS_V3DT_AUTOGEN_CAMINFO` | `0` | Regenerate `config/v3dt/camInfo_*.yml` at DS8 startup |
| `NOESIS_V3DT_CAMINFO_WORLD_SCALE` | `1.0` | Unit multiplier for camInfo world units (1.0 = meters, 100.0 = centimeters). If you use `100.0`, you must retune SV3DT world-space noise/thresholds accordingly. |
| `NOESIS_V3DT_CAMINFO_MATRIX_TYPE` | `w2p` | `w2p` writes `projectionMatrix_3x4_w2p`; `3x4` writes `projectionMatrix_3x4` (DeepStream adds `(w/2,h/2)` internally) |
| `NOESIS_V3DT_CAMINFO_INVERT_E` | `0` | Invert `E` matrix before computing projection (set to `1` only if your stored `E` is camera→world / `Twc`) |
| `NOESIS_V3DT_CAMINFO_Y_FLIP` | `1` | Flip image Y axis when building projection (escape hatch for convention mismatches) |
| `NOESIS_V3DT_CAMINFO_WORLD_AXES` | `xzy` | Remap world axes when generating camInfo (swaps Y/Z to align Y‑up calibration with SV3DT’s Z‑up frame) |
| `NOESIS_V3DT_MODEL_HEIGHT_M` | n/a | Model height is stored in camInfo (`modelInfo.height`); set via `scripts/generate_v3dt_caminfo.py --model-height` |
| `NOESIS_V3DT_MODEL_RADIUS_M` | n/a | Model radius is stored in camInfo (`modelInfo.radius`); set via `scripts/generate_v3dt_caminfo.py --model-radius` |

**Manual regeneration:**

```bash
python3 scripts/generate_v3dt_caminfo.py --pipeline-config \
  config/infer_v3dt_baseline.yaml
```

### Output units

In the current DS8 V3DT setup, **camInfo is generated in meters** by default (`NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`), and SV3DT world-space outputs are treated as meters end-to-end.

Note: older troubleshooting notes reference a centimeter-scale camInfo workflow (`NOESIS_V3DT_CAMINFO_WORLD_SCALE=100`) plus “scale back to meters” behavior. That conversion path is not currently implemented as a general hook in this repo; prefer meters-native camInfo + meter-tuned tracker configs.

| Output | Location | Units |
|--------|----------|-------|
| `bbox3d.{xCentre,yCentre,zCentre}` | WS tracking telemetry | Meters (SV3DT world; Z‑up) |
| `bbox3d.{xLen,yLen,zLen}` | WS tracking telemetry | Meters (`zLen` is height) |
| `velocity3d` | WS tracking telemetry | Meters/second |
| `world` footpoint | WS tracking telemetry | Meters (SV3DT world if `world_source=bbox3d`) |
| BEV footpoints | `bev-frame` WS payload | Meters |

### Validation

```bash
# Verify calibration meters + camInfo meters (ratio should be ~1 for each camera)
python3 - <<'PY'
import json, math
from pathlib import Path
import numpy as np, yaml

target_w, target_h = 1920, 1080  # streammux output resolution
cams = yaml.safe_load(Path("config/cameras.yaml").read_text())
models = cams["intrinsics_models"]
cal = json.loads(Path("config/camera_calibration.json").read_text())["cameras"]

def scale_intrinsics(fx, fy, cx, cy, w, h):
    base_w = 2.0 * float(cx) if cx else 0.0
    base_h = 2.0 * float(cy) if cy else 0.0
    scale_x = float(w) / base_w if base_w else 1.0
    scale_y = float(h) / base_h if base_h else 1.0
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y

print("camera ratio_check (camInfo_units / meters): expected ~ 1")
for cam_id in sorted(cams["cameras"], key=lambda k: int(k)):
    name = cams["cameras"][cam_id]["name"]
    model = cams["cameras"][cam_id]["model"]
    intr = models[model]["intrinsics"]
    fx, fy, cx, cy = map(float, (intr["fx"], intr["fy"], intr["cx"], intr["cy"]))
    fx, fy, cx, cy = scale_intrinsics(fx, fy, cx, cy, target_w, target_h)
    caminfo = yaml.safe_load(Path(f"config/v3dt/camInfo_{name}.yml").read_text())
    if "projectionMatrix_3x4" in caminfo:
        # DeepStream shifts by (w/2,h/2) internally; camInfo stores the zero-centered form.
        K = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]], dtype=float)
        P = np.array(caminfo["projectionMatrix_3x4"], dtype=float).reshape((3, 4))
    else:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        P = np.array(caminfo["projectionMatrix_3x4_w2p"], dtype=float).reshape((3, 4))
    RT = np.linalg.inv(K) @ P
    t_caminfo = RT[:, 3]

    E = np.array(cal[name]["E"], dtype=float).reshape((4, 4), order="F")
    t = E[:3, 3]

    ratios = []
    for i in range(3):
        ratios.append(abs(t_caminfo[i] / t[i]) if abs(t[i]) > 1e-9 else float("nan"))
    print(f"- {name}: ratios_abs(x,y,z)={[round(r,6) if math.isfinite(r) else r for r in ratios]}")
PY

# Terminated track dumps (helps diagnose motion fragmentation / rapid ID churn)
# Enabled via `config/v3dt/nvtracker_sv3dt.yml`:
#   TargetManagement.outputTerminatedTracks: 1
#   TargetManagement.terminatedTrackFilename: /tmp/noesis_track_dump_
# Interpretation: lots of short-lived terminated tracklets while a person is continuously visible
# indicates fragmentation (tracks constantly being killed + re-created under motion/occlusion).
ls -1 /tmp/noesis_track_dump_*.txt 2>/dev/null | tail -n 5 || true

# Check bbox3d.yLen (adult heights should be ~1.4–2.1m)
python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_sv3dt.yaml

# Run OOM regression test (should complete without memory growth)
python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sv3dt.yaml --duration-s 300
```

### V3DT Forensics Toolkit

Use the forensics tool when calibration/3D tracking issues need explicit math + telemetry:

```bash
python3 scripts/v3dt_forensics.py snapshot --pipeline-config build/effective_pipeline_yolo11_seg.yaml
export NOESIS_V3DT_DIAG_LOG=1
export NOESIS_V3DT_DIAG_DIR=diagnostics
python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium.yaml
python3 scripts/v3dt_forensics.py analyze --log diagnostics/v3dt_frames_<session>.ndjson \
  --snapshot diagnostics/v3dt_snapshot_<timestamp>.json \
  --scale-sweep auto
python3 scripts/v3dt_forensics.py panel --snapshot diagnostics/v3dt_snapshot_<timestamp>.json \
  --report diagnostics/v3dt_report_<timestamp>.json
```

See `docs/DS8_v3dt_forensics.md` for the full workflow.

### Tilt-only preview (depth plane)

When yaw/translation are trusted but tilt still looks off, generate a **preview**
calibration file that adjusts pitch/roll from the latest MapAnything depth snapshot:

```bash
python3 scripts/auto_tilt_from_depth.py \
  --camera living-room \
  --camera family-room \
  --output config/camera_calibration_preview.json
```

If a camera improves only with the opposite plane-normal sign, override it:

```bash
python3 scripts/auto_tilt_from_depth.py \
  --camera living-room \
  --camera family-room \
  --normal-sign family-room=-1 \
  --output config/camera_calibration_preview.json
```

Test it by generating camInfo from the preview calibration:

```bash
python3 scripts/generate_v3dt_caminfo.py \
  --pipeline-config config/infer_v3dt_medium.yaml \
  --calibration config/camera_calibration_preview.json \
  --output-dir config/v3dt_preview
```

Then point the tracker camInfo list to `config/v3dt_preview/` (or copy the
preview camInfo files into `config/v3dt/` for a one-off run).

## Hardening work order

See `plans/DS8/v3dt/ds8_v3dt_hardening_work_order.md` for the task breakdown covering:
- Unit convention standardization (V3DT-H01 ✓)
- MV3DT config alignment (V3DT-H02 ✓)
- Time-matching tolerance (V3DT-H03 ✓)
- Native extension robustness (V3DT-H05 ✓)

## Note on legacy docs

There are older, DS7-oriented MV3DT notes under `plans/mv3dt/`. This folder (`plans/DS8/v3dt/`) is the canonical DS8 plan; any useful legacy content has been rewritten and integrated here.
