# Validation catalog

| Capability | Preferred direct check |
| --- | --- |
| Fixture/report contracts | `tests/test_validation_toolbox.py` selected test(s) |
| Registered fixture regression | `scripts/noesis_validation_regression_runner.py` |
| Saved/live tracking and BEV | `scripts/noesis_validation_telemetry_report.py` |
| Menon scene placement | `scripts/noesis_validation_menon_trace_report.py` |
| Browser trace capture | `scripts/noesis_validation_capture_menon_trace.py` |
| Authored-scene geometry | `scripts/noesis_validation_authored_scene.py` |
| Track geometry acceptance | `scripts/noesis_validation_track_geometry_acceptance.py` |
| Pose synchronization | `scripts/menon_pose_sync_report.py` |

Read each script's `--help` and supply the exact current input. Do not infer
that an old command line, port, or fixture validates a newly changed contract.
