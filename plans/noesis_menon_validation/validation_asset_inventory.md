# Validation Asset Inventory

Status: current as of 2026-05-27.

This inventory maps existing validation assets into the layered Noesis/Menon
toolbox so future agents can reuse what is already present instead of inventing
parallel checks.

## New shared toolbox

| Asset | Tier | Primary use |
|---|---|---|
| `noesis/validation/` | Tier 1-4 | Shared report schema, validators, visual artifact helpers, telemetry parsing, Menon trace parsing, and Menon browser snapshot conversion. |
| `scripts/noesis_validation_runner.py` | Tier 2 | Fixture/offline report generation for transforms, anchors, camera checks, tracking, BEV, scene, Menon placement, and visual overlays. |
| `scripts/noesis_validation_telemetry_report.py` | Tier 3 | Saved/live DS8 WebSocket telemetry validation and track-audit generation. |
| `scripts/noesis_validation_menon_trace_report.py` | Tier 4 | Saved Menon placement-trace validation for Noesis-world to Menon-scene agreement. |
| `scripts/noesis_validation_capture_menon_trace.py` | Tier 4 | Playwright-based Menon browser debug capture converted into the shared Menon trace contract. |
| `scripts/noesis_validation_regression_runner.py` | Tier 2-4 | Registry-driven regression suite with expected-threshold checks. |
| `tests/test_validation_toolbox.py` | Tier 1-2 | Focused tests for common reports, validators, fixtures, telemetry, Menon traces, browser snapshot conversion, and regression runner behavior. |

## Existing scripts to reuse

| Asset | Tier | Primary use |
|---|---|---|
| `scripts/menon_bev_track_parity_smoke_test.py` | Tier 4 | Menon/BEV parity smoke for Noesis track placement. |
| `scripts/validate_virtual_twin_tracking.py` | Tier 4 | Virtual-twin tracking validation. |
| `scripts/validate_virtual_twin_visual_alignment.py` | Tier 4 | Virtual-twin visual alignment validation. |
| `scripts/menon_pose_calibration_smoke_test.py` | Tier 4 | Menon pose/extrinsics calibration smoke. |
| `scripts/sanity_check_v3dt_calibration.py` | Tier 3 | V3DT calibration sanity checks. |
| `scripts/v3dt_forensics.py` | Tier 3 | V3DT snapshot, analysis, and panel generation. |
| `scripts/sv3dt_meta_smoke_test.py` | Tier 3 | SV3DT/MV3DT metadata smoke. |
| `scripts/ma_depth_rpc_smoke_test.py` | Tier 3 | MapAnything depth RPC contract smoke. |
| `scripts/floorplan_rpc_smoke_test.py` | Tier 3 | Floorplan RPC contract smoke. |
| `scripts/auto_calibrate_rpc_smoke_test.py` | Tier 3 | Auto-calibration RPC contract smoke. |
| `scripts/calibration_rpc_smoke_test.py` | Tier 3 | Calibration RPC contract smoke. |
| `scripts/reid_stable_id_smoke_test.py` | Tier 3 | ReID/StableID runtime smoke. |
| `scripts/roi_reload_smoke_test.py` | Tier 3 | Analytics ROI reload smoke. |
| `scripts/webrtc_gateway_smoke_test.py` | Tier 3 | WebRTC gateway smoke. |
| `scripts/zero_copy_smoke_test.py` | Tier 3 | Zero-copy smoke. |
| `scripts/zero_copy_stats_smoke_test.py` | Tier 3 | Zero-copy stats smoke. |

## Existing pytest coverage to reuse

| Asset | Tier | Primary use |
|---|---|---|
| `tests/test_calibration_manager.py` | Tier 1-2 | Calibration manager behavior. |
| `tests/test_menon_pose_extrinsics.py` | Tier 1-2 | Menon pose/extrinsics math. |
| `tests/test_bev_renderer_world_smoothing.py` | Tier 1-2 | BEV renderer world smoothing. |
| `tests/test_bev_motion_smoothing.py` | Tier 1-2 | BEV motion smoothing. |
| `tests/test_depth_registration.py` | Tier 1-2 | Depth registration behavior. |
| `tests/test_depth_normals.py` | Tier 1-2 | Depth normal calculations. |
| `tests/test_depth_tracking_frame_processor.py` | Tier 1-2 | Depth tracking frame processing. |
| `tests/test_virtual_twin_builder.py` | Tier 1-2 | Virtual-twin artifact building. |
| `tests/test_virtual_twin_geometry.py` | Tier 1-2 | Virtual-twin geometry helpers. |
| `tests/test_virtual_twin_registration.py` | Tier 1-2 | Virtual-twin registration. |
| `tests/test_virtual_twin_store_api.py` | Tier 1-2 | Virtual-twin artifact API/store. |
| `tests/test_stable_id_manager.py` | Tier 1-2 | StableID manager behavior. |
| `tests/test_stable_id_manager_pose.py` | Tier 1-2 | Pose-assisted StableID behavior. |
| `tests/test_stable_id_gpu_parity.py` | Tier 1-2 | StableID GPU parity checks. |
| `tests/test_v3dt_forensics_snapshot.py` | Tier 1-2 | V3DT forensic snapshot. |
| `tests/test_v3dt_forensics_analysis.py` | Tier 1-2 | V3DT forensic analysis. |
| `tests/test_v3dt_forensics_logger.py` | Tier 1-2 | V3DT forensic logger. |
| `tests/test_v3dt_forensics_panel.py` | Tier 1-2 | V3DT forensic panel. |
| `tests/test_object_depth_result.py` | Tier 1-2 | Object-depth result contracts. |
| `tests/test_object_depth_fusion_probe.py` | Tier 1-2 | Object-depth fusion probe. |
| `tests/test_object_depth_meta_native_contracts.py` | Tier 1-2 | Object-depth native metadata contracts. |
| `tests/test_analytics_telemetry_hook.py` | Tier 1-2 | Analytics telemetry hook contract. |

## Use Rules

- Prefer the shared toolbox report schema for new validators so outputs compose
  into the regression runner.
- Keep existing smoke scripts as focused runtime acceptance gates; do not fold
  them into the fixture runner unless they can emit the shared report format
  without changing their canonical runtime behavior.
- If a script validates a DS8 runtime behavior, run it against
  `noesis/ds8_runtime.py` or mark the evidence blocked.
