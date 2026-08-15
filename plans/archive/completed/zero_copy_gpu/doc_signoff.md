# Doc Signoff (Zero-Copy GPU)

Date: 2026-02-08

## Completed artifacts
- `plans/zero_copy_gpu/01_graph_memory_contract.md`
- `plans/zero_copy_gpu/02_tensor_path_contracts.md`
- `plans/zero_copy_gpu/03_boundary_serialization_contract.md`
- `plans/zero_copy_gpu/04_validation_matrix.md`
- `plans/zero_copy_gpu/cutover_policy.md`
- `plans/zero_copy_gpu/hotspot_inventory.md`
- `plans/zero_copy_gpu/manifest.yaml`
- `plans/zero_copy_gpu/perf_baseline.md`

## Implementation alignment
- DS8 core pipeline is GPU-first with zero-copy enforcement metrics.
- Pose and ReID now use native metadata extraction paths in production.
- StableID similarity backend is GPU-enabled in hard-cutover mode.
- MapAnything production postprocess is async-only.
- Deprecated CPU migration scaffolding removed:
  - legacy CPU fallback helpers/constants/guards in `noesis/pipelines/hooks.py`;
  - temporary path-mode stats and deprecated flag validation in `noesis/ds8_runtime.py`;
  - obsolete fallback telemetry field in `reid/stable_id_manager.py`.

## Validation snapshot
- `pytest -q tests/test_zero_copy_invariants.py tests/test_mapanything_operator_contracts.py tests/test_stable_id_gpu_parity.py tests/test_zero_copy_stats_contract.py tests/test_zero_copy_boundary_budget.py tests/test_reid_tensor_contracts.py tests/test_pose_tensor_contracts.py tests/test_pose_meta_native_contracts.py` -> pass (16 tests).
- `NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/ds8_runtime_30s_gate.py` -> pass.
- `NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/ds8_runtime_30s_gate.py` -> pass.
- `NOESIS_STABLEID_GPU_ENABLED=1 NOESIS_ZERO_COPY_PROFILE=strict python3 scripts/zero_copy_soak_test.py --ws ws://127.0.0.1:6582 --duration 30 --startup-timeout 12 --max-violations 0` -> pass.
- `python3 scripts/zero_copy_stats_smoke_test.py --duration-s 60` -> pass.
- `python3 scripts/zero_copy_smoke_test.py` -> pass (`max_boundary_p99_ms=0.765237`, `max_zero_copy_violations=0`).
- `python3 scripts/zero_copy_runtime_gate.py --duration-s 120` -> pass.
- `python3 scripts/zero_copy_perf_gate.py capture-baseline --duration-s 90 --out logs/zero_copy/perf_baseline.json` -> pass.
- `python3 scripts/zero_copy_perf_gate.py compare --baseline logs/zero_copy/perf_baseline.json --duration-s 90 --out logs/zero_copy/perf_candidate.json` -> pass.
- `./scripts/build_noesis_pose_meta_ext.sh` -> pass (native extension rebuilt after payload-guard update).

## Finalization status
- Merge gates in `plans/zero_copy_gpu/manifest.yaml` are satisfied in this workspace:
  - docs complete;
  - tests green;
  - smoke green;
  - no new core CPU frame path introduced;
  - StableID CPU/GPU parity green;
  - StableID GPU backend live in runtime stats.
