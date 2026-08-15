# Session Notes

- 2026-02-10: Begin execution of strict pose-first unification plan. Priority order: docs scaffold, Menon pose export, Noesis pose loader, strict runtime gate, baseline alignment, tests.
- 2026-02-10: Implemented strict PoseV1 authority on Menon save path (`set_extrinsics` pose-only, removed calibrate fallback trigger).
- 2026-02-10: Implemented Noesis strict default (`NOESIS_CALIBRATION_POSE_ONLY=1`), pose-required runtime handler, floor-geometry startup checks, and baseline `menon_scene` world defaults for tracking/BEV.
- 2026-02-10: Added/updated validation artifacts (`tests/test_menon_pose_extrinsics.py`, `tests/test_calibration_manager.py`, `scripts/menon_pose_calibration_smoke_test.py`, `scripts/calibration_rpc_smoke_test.py`).
- 2026-02-10 12:58: Immediate execution plan for closeout pass: run strict pose smoke, calibration RPC smoke, BEV/track parity smoke, then close open checklist/gate/doc items.
- 2026-02-10 13:02: Active checklist item `Validate BEV and track outputs against Menon expectations` completed; parity PASS with `p95_err_m=0.013097595290442646` and all remaining phase gates closed.
- 2026-02-10 13:09: Patched `scripts/check_agents_docs_consistency.py` to stop failing on archive/history-only references; checker now PASS and `V-010` closed.
- 2026-02-10 16:06: Immediate execution plan for root-fix pass: create OBJ-unit calibration artifact, lock baseline runtime to that artifact (no implicit fallback), remove Menon-side world scaling, then rerun pose/docs/syntax validations.
- 2026-02-10 16:09: Active checklist items in Phase 1.6 completed (`camera_calibration_menon_obj.json` lock, baseline no-fallback default, no client-side world scaling) and validations passed (`py_compile`, `node --check`, pose smoke, docs consistency check).
- 2026-07-09 23:20: Immediate execution plan: record the shared Spatial OS work order, repair truth/safety regressions, add versioned observation/world contracts, then cut Menon from browser-owned fusion to backend global entities with Tier 1-4 evidence.
- 2026-07-10 09:10: Active checklist item is canonical Menon consumption. Immediate plan: implement strict bounded `noesis.world.snapshot` state, route rendering through authoritative entities with one renderer-boundary transform, add promoted scene-release loading with atomic swaps, and validate adversarial cases plus the Node 22 suite/build.
- 2026-07-10 09:42: Canonical world/occupancy and coherent scene consumption are complete in code. Final evidence: 112/112 Node tests, TypeScript/Vite build PASS, production audit 0 vulnerabilities, and exact real-release consumer PASS for 3 cameras/27 camera artifacts/5 authored dependencies; live processes intentionally remain untouched pending coordinated cutover.
