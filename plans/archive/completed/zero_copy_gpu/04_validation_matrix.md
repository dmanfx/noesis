# Zero-Copy GPU Hard-Cutover Validation Matrix

## 1. Cutover Decision Rule (Blocking)

Hard cutover is **PASS** only if all blocking gates below pass:

| Gate | Metric | Threshold | Source |
|---|---|---:|---|
| A1 | `core_zero_copy_violations` | `== 0` | runtime stats |
| A2 | `boundary_cpu_serialization_p99_ms` | `<= 3.0 ms` | runtime stats |
| A3 | unit + smoke suite | all pass | `pytest` + smoke scripts |
| A4 | runtime soak gate | pass | soak scripts |

Any gate failure is a release blocker for hard cutover.

## 2. Execution Status (2026-02-08)

Blocking assets listed in this matrix are implemented and exercised in-repo:

- New integration assets present:
  - `tests/test_zero_copy_stats_contract.py`
  - `tests/test_zero_copy_boundary_budget.py`
  - `scripts/zero_copy_stats_smoke_test.py`
- Runtime/perf gate assets present:
  - `scripts/zero_copy_runtime_gate.py`
  - `scripts/zero_copy_perf_gate.py`
- Tensor-path contract tests present:
  - `tests/test_reid_tensor_contracts.py`
  - `tests/test_pose_tensor_contracts.py`
  - `tests/test_pose_meta_native_contracts.py`
- Gate run snapshot:
  - `python3 scripts/zero_copy_smoke_test.py` -> pass.
  - `python3 scripts/zero_copy_runtime_gate.py --duration-s 120` -> pass.
  - `python3 scripts/zero_copy_perf_gate.py capture-baseline ...` -> pass.
  - `python3 scripts/zero_copy_perf_gate.py compare ...` -> pass.

## 3. Standard Test Environment

- Runtime entrypoint: `noesis/ds8_runtime.py`
- Primary config: `config/infer.yaml`
- Camera config: `config/cameras.yaml`
- Disable optional output noise during validation unless explicitly testing it:
  - `NOESIS_MOSAIC_RTSP_ENABLED=0`
  - `NOESIS_MOSAIC_WEBRTC_ENABLED=0`
- Enable latency metrics when running runtime/perf gates:
  - `NVDS_ENABLE_LATENCY_MEASUREMENT=1`

Artifacts to keep per run (required for sign-off):

- `logs/runtime_smoke/*` (from `tests/run_smoke.sh`)
- `logs/zero_copy/*` (new zero-copy smoke/runtime/soak scripts)
- `plans/zero_copy_gpu/perf_baseline.md` updated with baseline/candidate summaries

## 4. Unit Validation Matrix

### 4.1 Existing unit coverage to run as part of cutover

| ID | Status | Asset | Command | Pass criteria | Fail criteria |
|---|---|---|---|---|---|
| U-E1 | Existing | `tests/test_pipeline_build.py` | `python3 -m pytest tests/test_pipeline_build.py -q` | All tests pass; DS8 graph builds; depth gate lifecycle assertions pass | Any assertion failure or exception |
| U-E2 | Existing | `tests/test_mapanything_postprocess.py` | `python3 -m pytest tests/test_mapanything_postprocess.py -q` | Depth result emits; storage round-trip valid; shape/minmax checks pass | Missing result/publish, malformed payload, failed assertions |
| U-E3 | Existing | `tests/test_latency_metrics.py` | `python3 -m pytest tests/test_latency_metrics.py -q` | Percentile math and env-disabled behavior pass | Any mismatch in window/percentile or collector behavior |
| U-E4 | Existing | `tests/test_analytics_telemetry_hook.py` | `python3 -m pytest tests/test_analytics_telemetry_hook.py -q` | Tracking payload, occupancy grace window behavior pass | Tracking payload or occupancy assertions fail |
| U-E5 | Existing | `tests/test_exclude_prune_hook.py` | `python3 -m pytest tests/test_exclude_prune_hook.py -q` | ROI prune behavior deterministic and correct | ROI prune regression |
| U-E6 | Existing | `tests/test_mapanything_preprocess_fused.py` | `python3 -m pytest tests/test_mapanything_preprocess_fused.py -q` | Fused preprocess packers produce expected tensors | Build or numeric mismatch failure |

### 4.2 New unit tests required for hard cutover

Create `tests/test_zero_copy_invariants.py` with the exact cases below:

| ID | Status | New test case | Pass criteria | Fail criteria |
|---|---|---|---|---|
| U-N1 | New | `test_ds8_graph_enforces_gpu_osd_mode` | Built graph has `components["osd"].config["process-mode"] == 0` | Any non-zero process-mode |
| U-N2 | New | `test_ds8_graph_core_caps_are_nvmm` | Core video caps in DS8 chain are `video/x-raw(memory:NVMM)` where caps are explicit | Any core caps fallback to CPU memory |
| U-N3 | New | `test_no_core_appsink_or_cpu_frame_branch` | Core chain contains no `appsink`/CPU frame extraction path | Presence of CPU frame branch in core path |
| U-N4 | New | `test_stats_contract_exposes_zero_copy_metrics` | Stats payload includes `core_zero_copy_violations` and `boundary_cpu_serialization_p99_ms` keys | Missing keys or non-numeric values |
| U-N5 | New | `test_prod_fail_fast_on_core_violation` | In hard-fail mode, first core violation raises/fails immediately | Runtime continues after core violation |

Run target:

```bash
python3 -m pytest tests/test_zero_copy_invariants.py -q
```

## 5. Integration Validation Matrix

### 5.1 Existing integration checks reused

| ID | Status | Asset | Command | Pass criteria | Fail criteria |
|---|---|---|---|---|---|
| I-E1 | Existing | DS8 runtime smoke harness | `tests/run_smoke.sh` | Exit code 0; runtime checks marked PASS (no FAIL rows) | Any FAIL in summary or non-zero exit |
| I-E2 | Existing | Depth RPC smoke | `python3 scripts/ma_depth_rpc_smoke_test.py --pipeline-config config/infer.yaml --cameras-config config/cameras.yaml` | `[PASS]`; cache-first + fresh depth response valid; fresh `ts_us` increases | Timeout, malformed payload, stale fresh response |
| I-E3 | Existing | Stable ID smoke | `python3 scripts/reid_stable_id_smoke_test.py --pipeline-config config/infer_smoke_reid.yaml --cameras-config config/cameras.yaml` | `[PASS]`; non-null stable IDs persist across frames | No stable IDs or no persistence |
| I-E4 | Existing | WebRTC smoke (when RTSP/WebRTC enabled) | `python3 scripts/webrtc_gateway_smoke_test.py --ws ws://127.0.0.1:6008 --duration 5 --pt 103` | RTP packets and decoded frames are both `> 0` | No RTP or no decoded frames |

### 5.2 New integration checks required

| ID | Status | New asset | Required behavior | Pass criteria | Fail criteria |
|---|---|---|---|---|---|
| I-N1 | New | `tests/test_zero_copy_stats_contract.py` | Validate WS `stats` schema for zero-copy metrics | All expected metric fields present and typed correctly | Missing/invalid fields |
| I-N2 | New | `tests/test_zero_copy_boundary_budget.py` | Validate boundary serialization histogram/percentile accounting | Computed p99 is deterministic and in expected units (ms) | p99 accounting mismatch |
| I-N3 | New | `scripts/zero_copy_stats_smoke_test.py` | Spawn runtime, connect WS, read rolling stats for 60s | `core_zero_copy_violations==0` and `boundary_cpu_serialization_p99_ms<=3.0` for full window | Any violation or budget breach |
| I-N4 | New | Pose alignment live check (`--pgie-profile yolo26_seg --size n`) | Validate keypoints align with detection/instance mask across cameras after ROI remap | Keypoints remain anchored to person/mask under movement in all active cameras | Systematic per-camera offset or bbox-size-coupled drift |

## 6. Smoke Gate Matrix

### 6.1 New smoke orchestrator required by manifest

Implement `scripts/zero_copy_smoke_test.py` (blocking gate script in `manifest.yaml`).

Required behavior:

1. Start DS8 runtime with `config/infer.yaml`.
2. Trigger boundary traffic (`/api/v1/depth/refresh` and WS depth request path).
3. Sample WS `stats` once per second for at least 90 seconds.
4. Emit final JSON summary and exit code.

Required pass/fail thresholds:

- `core_zero_copy_violations`: must stay `0` for the whole run.
- `boundary_cpu_serialization_p99_ms`: must be `<= 3.0` at end of run and max sampled value.
- `pipeline.errors`: must remain empty.
- Runtime process must stay alive for full test window.

Command:

```bash
python3 scripts/zero_copy_smoke_test.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 90 \
  --stats-ws ws://127.0.0.1:6008
```

## 7. Runtime Gate Matrix (Pre-Soak)

Add `scripts/zero_copy_runtime_gate.py` for a medium-duration blocking gate (default 20 minutes).

| ID | Status | Command | Pass criteria | Fail criteria |
|---|---|---|---|---|
| R-N1 | New | `python3 scripts/zero_copy_runtime_gate.py --pipeline-config config/infer.yaml --cameras-config config/cameras.yaml --duration-s 1200` | Process survives full window; `core_zero_copy_violations==0`; `boundary_cpu_serialization_p99_ms<=3.0`; no pipeline errors | Crash/restart, non-zero core violations, p99 budget breach, pipeline errors |
| R-N2 | New | same script with `--depth-burst-interval-s 15` | Repeated depth bursts do not create core-copy violations; boundary p99 still in budget | Any violation during burst traffic |

## 8. Soak Gate Matrix

### 8.1 Existing soak-style script to reuse

`scripts/v3dt_oom_regression_test.py` can be used immediately for memory stability until dedicated zero-copy soak script lands.

Recommended command for interim gate:

```bash
python3 scripts/v3dt_oom_regression_test.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 7200 \
  --sample-s 1.0 \
  --max-rss-mib 8192 \
  --max-delta-mib 1024
```

Interim pass criteria:

- Runtime remains alive for 2 hours.
- RSS delta `< 1024 MiB`.
- No OOM/runaway condition triggered by script.

### 8.2 New dedicated soak script required

Add `scripts/zero_copy_soak_test.py` (6h release gate):

- Poll WS stats every 1s.
- Record `/proc/<pid>/status` RSS every 5s.
- Trigger boundary traffic every 10s.
- Output minute-window aggregates to `logs/zero_copy/soak_*.jsonl`.

Blocking thresholds:

- `core_zero_copy_violations == 0` at all times.
- Per-minute `boundary_cpu_serialization_p99_ms <= 3.0`.
- Runtime uptime uninterrupted for full duration.
- RSS growth `<= 1536 MiB` over 6h.

## 9. Performance Gate Matrix

### 9.1 Baseline capture (must happen before cutover flip)

Add `scripts/zero_copy_perf_gate.py` with two modes:

- `capture-baseline`
- `compare`

Capture command:

```bash
python3 scripts/zero_copy_perf_gate.py capture-baseline \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 900 \
  --out logs/zero_copy/perf_baseline.json
```

Candidate command:

```bash
python3 scripts/zero_copy_perf_gate.py compare \
  --baseline logs/zero_copy/perf_baseline.json \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 900 \
  --out logs/zero_copy/perf_candidate.json
```

Blocking perf thresholds:

| Metric | Threshold |
|---|---:|
| `core_zero_copy_violations` | `== 0` |
| `boundary_cpu_serialization_p99_ms` | `<= 3.0 ms` |
| camera throughput (`fps_p50`) | `>= 95%` of baseline |
| latency (`pipeline.latency_ms.p95`) | `<= 110%` of baseline |
| GPU memory used (`nvidia-smi` p95) | `<= 110%` of baseline |

If baseline artifacts are missing, perf gate is **not passable**.

## 10. Full Execution Order for Release Candidate

Run in this order:

1. Unit existing suite (`U-E*`)
2. New unit zero-copy invariants (`U-N*`)
3. Existing integration suite (`I-E*`)
4. New integration zero-copy checks (`I-N*`)
5. Zero-copy smoke (`scripts/zero_copy_smoke_test.py`)
6. Runtime gate (`scripts/zero_copy_runtime_gate.py`)
7. Soak gate (`scripts/zero_copy_soak_test.py`, or interim OOM gate until implemented)
8. Perf compare gate (`scripts/zero_copy_perf_gate.py compare`)

Cutover is approved only when all blocking gates pass with archived artifacts.

## 11. Pose and ReID Native Path Closure (Completed 2026-02-08)
- Native pose decode is active via `noesis_pose_meta_ext.extract_pose_keypoints(...)`.
- Native ReID embedding decode is active via `noesis_reid_meta_ext.extract_reid_embedding(...)`.
- Post-work pose alignment patch shipped:
  - payload includes `keypoints_roi` + `keypoints_abs`;
  - overlay remaps using source-bbox -> current-bbox scaling.
- Validation outcomes:
  - runtime gate: pass;
  - soak gate: pass (`max_zero_copy_violations=0`);
  - live visual check confirms pose alignment with detections/masks.

## 12. StableID GPU Similarity Validation (Current)

### 12.1 Unit parity checks (blocking)
| ID | New test | Command | Pass criteria | Fail criteria |
|---|---|---|---|---|
| SID-U1 | `tests/test_stable_id_gpu_parity.py::test_cpu_gpu_assignment_timeline_parity` | `python3 -m pytest tests/test_stable_id_gpu_parity.py -q` | CPU and GPU backend produce identical `stable_id` assignment timeline on fixed replay | Any divergence in assignment sequence |
| SID-U2 | `tests/test_stable_id_gpu_parity.py::test_similarity_numeric_tolerance` | same | Cosine similarity/top-k match within tolerance vs CPU reference | Rank/order mismatch beyond tolerance |

### 12.2 Integration checks (blocking)
| ID | New check | Command | Pass criteria | Fail criteria |
|---|---|---|---|---|
| SID-I1 | Runtime backend stats | `NOESIS_STABLEID_GPU_ENABLED=1 python3 scripts/ds8_runtime_30s_gate.py` | `stableid_backend_mode` reported as `gpu` when CUDA is available | Backend remains CPU without explicit override |
| SID-I2 | Existing stable-id smoke on GPU backend | `NOESIS_STABLEID_GPU_ENABLED=1 python3 scripts/reid_stable_id_smoke_test.py --pipeline-config config/infer_smoke_reid.yaml --cameras-config config/cameras.yaml` | Stable IDs remain non-null and persistent | Missing/unstable IDs |

### 12.3 Runtime and soak gates (blocking)
1. After each StableID GPU implementation step, run:
```bash
NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/ds8_runtime_30s_gate.py
```
Pass criteria:
- process alive for full 30s;
- no fatal signatures in log;
- no new core zero-copy violations.

2. Post-integration soak gate:
```bash
NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/zero_copy_soak_test.py --ws ws://127.0.0.1:6582 --duration 30 --startup-timeout 12 --max-violations 0
```
Pass criteria:
- `max_zero_copy_violations == 0`;
- backend stats indicate StableID GPU mode is active.

### 11.4 Performance gates (blocking)
| Metric | Threshold |
|---|---:|
| `stableid_gpu_match_p95_ms` | <= CPU-baseline p95 |
| `stableid_backend_mode` | `gpu` |
| `core_zero_copy_violations` | 0 |

## 12. MapAnything Async-Only Hardening (Completed 2026-02-08)
- Removed fallback processing of `converted_items[0]` when no matching `gie_id` is present.
- Removed sync-debug path; async worker path is now the only production behavior.
- Added counters:
  - `tensor_gie_mismatch_drops_total.mapanything`
  - `tensor_host_copies_total.mapanything`
- Validation completed:
  - `pytest -q tests/test_mapanything_operator_contracts.py tests/test_mapanything_postprocess.py tests/test_zero_copy_invariants.py` -> pass.
  - `NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/ds8_runtime_30s_gate.py` -> pass.
  - `NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/zero_copy_soak_test.py --ws ws://127.0.0.1:6582 --duration 30 --startup-timeout 12 --max-violations 0` -> pass.
