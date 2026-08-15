# Noesis Dev Console — Pipeline Control Utility (v3, post-Codex review 1)

> Compatibility decisions are derived **only** from the materialized effective pipeline YAML plus referenced DeepStream INI/YAML files. Profile names are hints, not truth.

## Problem Statement

1. PGIE swaps partially automated via `--pgie-profile` / `--size` on `noesis/ds8_runtime.py`.
2. Everything else requires hand-editing YAML or agent-driven changes.
3. Process control is manual CLI + Ctrl+C.
4. Metadata paths are not co-validated when swapping models (OSD, object-depth, mask tint, SGIE contracts).

---

## Validation Severity Model

| Severity | Launch behavior | UI |
|----------|-----------------|-----|
| **BLOCK** | Start disabled | Red, actionable fix text |
| **WARN** | Start allowed with acknowledgment | Amber, lists downstream degradation |
| **INFO** | Start allowed | Gray note |

---

## Critical Gap: Metadata Paths vs Model Type

`ds8_pipeline.py` hardcodes OSD: `display-mask=1`, `display-bbox=0`. Detect PGIE (`network-type=0`) populates `rect_params` only → mosaic shows little useful output.

### Consumer compatibility matrix

| Consumer | Requires seg masks | On detect-only today |
|----------|-------------------|----------------------|
| nvdsosd `display-mask` | `mask_params` | **Broken** |
| nvdsosd `display-bbox` | — | Works if enabled |
| `_apply_instance_mask_color_ds8` | `mask_params` | Silent no-op |
| `NOESIS.OBJECT_DEPTH` | `mask_params` (no bbox fallback) | `missing_mask` every person |
| Baseline world tracking | Indirect via object depth | Degraded; pose anchor may carry |
| Trails | Optional mask peak | Works via `rect_params` |
| ReID/Pose/Tracker SGIEs | PGIE boxes gie_id=1 | Works if contracts hold |

### Baseline + detect PGIE (precise severity)

- **WARN (not BLOCK):** `NOESIS.OBJECT_DEPTH` attaches `status=missing_mask`; `depth_used_m` / registered object-depth anchors unavailable.
- World output may rely on pose-floor or hold behavior when pose SGIE valid.
- **BLOCK** only for validation tiers that explicitly require depth-fused object anchors (user-selectable "strict baseline" launch mode).

### OSD YAML schema (DeepStream prop names)

```yaml
osd:
  process-mode: 0
  display-mask: 1
  display-bbox: 0
  display-text: 1
```

Console derives OSD from **mask availability** (parsed INI: `network-type`, `output-instance-mask`, parser func). `network-type=3` alone is insufficient if `output-instance-mask=0`.

### Source editing constraint (v1)

Changing sources is restart-required and **blocked in v1** unless all of the following match:
- `batch_size`, `streammux.batch-size`
- PGIE / DAv2 / MapAnything engine batch
- V3DT camInfo order (if v3dt)
- Camera/dewarper mapping vs `cameras.yaml`
- `depth_registration.json` active-camera fingerprints

---

## Missing Metadata / Compatibility Cases (full inventory)

| Area | Console behavior |
|------|------------------|
| PGIE INI | Parse `network-type`, parser func, `output-instance-mask`, `output-blob-names`, `cluster-mode` |
| Preprocess | `tensor-name`, `input-tensor-from-meta`; YOLO26 detect requires `images` |
| OSD | Derive mask/bbox from mask availability |
| Object depth | Warn on missing masks; no bbox fallback |
| SGIEs | Validate `operate-on-gie-id=1`, class 0, gie IDs 2-5 |
| ReID | SGIE `features` layer, dim 512, native ext — separate from tracker `reidFeatureSize` |
| Tracker | Block tracking-mode vs YAML family mismatch; verify camInfo paths |
| Dewarper | Reuse runtime intrinsics sync validation |
| Mosaic | RTSP/WebRTC env at build time — restart required |
| Native extensions | Gate rebuild/smoke when metadata plumbing touched |

---

## Process Lifecycle

- **Stop:** SIGINT to DS8 runtime **process group**, wait for pipeline/WS/REST/RTSP cleanup, escalate after timeout. SIGTERM is NOT the graceful path for `ds8_runtime.py`.
- **Ports:** Console `127.0.0.1:9090`. Child runtime launch spec defaults `--ws-host 127.0.0.1 --rest-host 127.0.0.1`. Non-local binds require explicit operator opt-in and appear in launch preview.
- **Port conflicts:** REST/WS port conflicts are preflight BLOCK/WARN results. Console surfaces actual WS/REST/RTSP endpoints after launch; never silently assumes an unrelated process is the intended runtime.
- **Security:** Local-only bind by default; document that console controls subprocesses, env, file writes, hot APIs.

## Materialization Artifact Isolation

- Split **dry-run preview** vs **committed launch** artifacts.
- Committed artifacts live under `build/dev_console/<launch_id>/` (immutable while process running).
- Record: base preset, profile, size, tracking_mode, env, generated INIs, effective YAML.
- Console preview must not overwrite shared `build/effective_pipeline_<profile>.yaml` used by external launches.

## Preset Typing

Presets typed: `canonical`, `smoke`, `experimental`, `archival`.

Each preset declares: runtime entrypoint, tracking_mode, cameras config, expected source type (rtsp/file), required artifacts.

Only `canonical` + `runnable` presets count toward "all presets pass preflight" gate.

Explicit inventory:
- canonical: `infer.yaml`, `infer_v3dt_baseline.yaml`
- experimental: `infer_v3dt_reimpl_*.yaml` (7 variants)
- smoke: `infer_smoke_reid.yaml`, `infer_reid_rtsp_min.yaml`
- archival: `bisenetpipe_sources.yaml` (if present)

## Runtime Proxy Allowlist (v1)

**Allowed:** WS stats/tracking subscribe, REST depth refresh, WS trail toggle, WS bev-config/overlay, WebRTC signaling.

**Excluded from proxy (dedicated editor only):** calibration writes, ROI persistence, ReID alias mutations, canonical config promotion — each requires validation, confirmation, backup, diff, rollback.

## Strict Baseline Launch Mode

Explicit preflight option `strict_baseline=true`: BLOCK launch when detect PGIE or object-depth would be `missing_mask` for any enabled camera. Default off (WARN only).

---

## Package Layout

```
noesis/dev_console/
  __main__.py
  server.py
  supervisor.py          # SIGINT process-group lifecycle
  launch_spec.py
  manifest.py
  manifest_data.yaml
  metadata_compat.py     # parses effective YAML + INIs
  presets.py + presets/  # beside overlays, not in config/
  validator.py           # calls noesis/ds8_preflight.py
  config_editor.py
  runtime_proxy.py
  overlays/              # generated dev overlays
  static/
noesis/ds8_preflight.py  # shared with ds8_runtime.py
```

---

## UI Tabs

1. Process — start/stop/restart, log tail, health, launch preview (argv/env/effective YAML)
2. Presets — all `config/infer*.yaml` variants (explicit inventory)
3. Models — PGIE/ReID/Pose/Tracker + metadata impact panel
4. Sources — URI/dewarper (with batch coupling blocks)
5. Knobs — env + YAML, restart badges
6. Runtime — WS stats, depth burst, trails, BEV
7. Preflight — BLOCK/WARN/INFO checklist

---

## Implementation Phases (Autonomous)

### Phase 0 — Contract / design gate

- Read `plans/DS8/ds8_master_work_orders.md` + relevant checklists
- Add design decision to `plans/DS8/ds8_design_decisions.md` for console schema/API
- Define validation severity model and docs/tests list

**Gate:** design doc written; no code yet.

### Phase 1 — Shared materialization + preflight core

- Extract `noesis/ds8_preflight.py` from `ds8_runtime.py`
- `metadata_compat.py` parses effective YAML + PGIE/preprocess INIs
- `launch_spec.py` + materialization preview
- Runtime refactored to call shared preflight (no duplicate validators)

**Gate:** unit tests for INI parsing, preflight parity; `pytest` + review artifact in `plans/noesis_dev_console/phase1_review.md`

### Phase 2 — Process supervisor

- SIGINT process-group stop, log tail, port discovery
- Presets for all infer*.yaml files
- Stub smoke labeled **offline plumbing only** (not metadata/GPU acceptance)

**Gate:** supervisor tests; stub smoke; review artifact `phase2_review.md`

### Phase 3 — OSD refactor + metadata compatibility

- YAML `osd:` block in `ds8_pipeline.py`
- Auto-derive OSD from parsed `network-type`
- Dependency warnings panel with BLOCK/WARN/INFO

**Gate:** detect→bbox OSD, seg→mask OSD; baseline+detect WARN; `pytest` + `phase3_review.md`

### Phase 4 — Model/source editors

- Structured editors with overlay YAML output
- Source-count edit blocks unless batch artifacts coherent
- Path normalization (no new `/home/...` hardcoding; existing absolute paths flagged WARN on promotion)
- Manifest includes deprecated env aliases (e.g. `NOESIS_REID_RESER_SID_POOL`) marked deprecated

**Gate:** overlay round-trip tests; `phase4_review.md`

### Phase 5 — Knobs catalog

- Full `manifest_data.yaml` for all `NOESIS_*` from `ds8_runtime.py` + `hooks.py`
- Preset diff view

**Gate:** manifest coverage test; `phase5_review.md`

### Phase 6 — Runtime proxy

- WS stats, REST depth burst, trail/BEV hot controls
- Documented endpoint proxy only

**Gate:** integration smoke when GPU available; `phase6_review.md`

### Phase 7 — Polish

- Explicit promote-to-canonical (backup, diff, rollback)
- Preset import/export
- `docs/DS8_testing_guide.md` section
- `./scripts/check_agents_docs_consistency.py`

**Gate:** all presets pass preflight; final review artifact; Codex audit pass

---

## Key Refactors

| Refactor | Phase |
|----------|-------|
| `noesis/ds8_preflight.py` | 1 |
| YAML `osd:` block | 3 |
| `launch_spec.py` | 1 |
| Path normalization | 4 |

---

## Success Criteria

1. Preset switch <30s without agent prompting
2. Detect↔seg auto-adjusts OSD from parsed INI, not profile name
3. Baseline+detect shows explicit object-depth WARN
4. All `NOESIS_REID_*` discoverable
5. Reproducible phase review artifacts (not opaque agent-only)
6. SIGINT graceful shutdown
7. Local-only default bind

## Out of Scope (v1)

- Hot PGIE swap without restart
- DEIMv2 graph slot
- Bbox-fallback object-depth
- Source count changes without full batch artifact coherence