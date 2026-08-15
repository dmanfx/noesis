# Zero-Copy Baseline Metrics

Status: finalized baseline + candidate perf comparison captured (2026-02-08)

## Unit and smoke checks
- `pytest -q tests/test_pipeline_build.py tests/test_analytics_telemetry_hook.py tests/test_depth_api.py tests/test_zero_copy_invariants.py`
  - Result: pass.
- `python3 scripts/zero_copy_smoke_test.py`
  - Result: pass (`max_zero_copy_violations=0`, `max_boundary_p99_ms=0.765237`).
- `python3 scripts/zero_copy_stats_smoke_test.py --duration-s 60`
  - Result: pass.

## Runtime gate checks
- `NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/ds8_runtime_30s_gate.py`
  - Result: pass (`duration_s~30s`, no fatal signatures).
- `NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/zero_copy_soak_test.py --ws ws://127.0.0.1:6582 --duration 30 --startup-timeout 12 --max-violations 0`
  - Result: pass (`max_zero_copy_violations=0`).
- `NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/ds8_runtime_30s_gate.py`
  - Result: pass (`stableid_backend_mode=gpu`).
- `NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/zero_copy_soak_test.py --ws ws://127.0.0.1:6582 --duration 30 --startup-timeout 12 --max-violations 0`
  - Result: pass (`max_zero_copy_violations=0`).

## Perf gate artifacts
- Baseline capture:
  - Command: `python3 scripts/zero_copy_perf_gate.py capture-baseline --duration-s 90 --out logs/zero_copy/perf_baseline.json`
  - Result: pass (`ok=true`).
- Candidate compare:
  - Command: `python3 scripts/zero_copy_perf_gate.py compare --baseline logs/zero_copy/perf_baseline.json --duration-s 90 --out logs/zero_copy/perf_candidate.json`
  - Result: pass (`ok=true`).
- Captured metrics (`logs/zero_copy/perf_candidate.json`):
  - `core_zero_copy_violations=0`
  - `boundary_cpu_serialization_p99_ms=0.603463`
  - `fps_p50_ratio=1.0068497007495107`
  - `gpu_memory_used_p95_mib_ratio=1.0011675423234092`
  - latency ratio check treated as non-blocking fallback in this environment because DS8 stats did not expose native `pipeline.latency_ms.p95`; both runs used the same fallback source (`boundary_cpu_serialization_p95_ms`), recorded in `metric_source_policy`.

## Implementation notes
- Pose extraction/overlay runs via native metadata (`noesis_pose_meta_ext`).
- ReID extraction runs via native metadata (`noesis_reid_meta_ext`).
- StableID similarity math runs on GPU in hard-cutover mode.
- MapAnything postprocess is async-only in production.
- Deprecated migration toggles, debug CPU fallback paths, and temporary mode/fallback stats were removed.
- WS runtime robustness fix applied: imported `websockets` in `websocket_server.py` to avoid exception-path `NameError` during client disconnect handling.
