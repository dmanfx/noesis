# WO-2A/B synchronized capture and conventional VIO

Status: implementation complete for the bounded import, native estimator
bridge, and public-data execution proof. Phone-specific capture and metric
calibration remain an explicit evidence dependency for the Fold 8 Ultra.
VI3 is deferred.

## WO-2A capture and import

The default in-page recorder captures camera video and available browser motion
samples as a raw observation bundle, paired with the selected static room
camera. Existing-video upload remains a separate RGB-only input. The current
capture workflow is described in the
[phone-scan README](../../tools/mapanything_phone_scan/README.md).
Browser callback receipt times and unverified Generic Sensor clock domains do
not establish acquisition timing or a calibrated camera/IMU offset, and cannot
admit metric VIO.

For calibrated metric-VIO input, use the native Android
[OpenCamera-Sensors recorder](https://github.com/prime-slam/OpenCamera-Sensors),
with synchronized video/IMU recording enabled, audio disabled, and OIS/EIS and
video stabilization disabled. The recorder's reported Camera2 timestamp source
and sensor IDs are retained; a Fold 8 Ultra model string does not prove a
shared realtime clock or camera-to-IMU calibration.

`tools/mapanything_phone_scan/capture.py` defines the bounded
`noesis.phone_capture.v1` import. It accepts a manifest or derives one from an
OpenCamera-Sensors export containing the video, `*_timestamps.csv`, separate
`*_accel.csv`, and `*_gyro.csv` streams. The normalized report preserves raw
streams and records device identity, sensor identity, clock domains and offset,
units, axes, camera intrinsics/distortion, camera-to-IMU transform, encoded
resolution, crop, orientation, and stabilization. Accel and gyro remain
separate timestamped streams; the estimator materializer records its linear
interpolation onto a union timestamp grid and retains samples bracketing the
camera interval.

ZIP/TAR import rejects traversal, absolute or non-normalized paths, duplicate
members, symlinks and other links, nonregular members, and archives exceeding
the configured file, member, byte, IMU-row, video-frame, or IMU-gap limits.
Video dimensions are probed from the encoded stream. Camera timestamp CSV
values already converted to integer nanoseconds are kept exact; only ffprobe
container seconds are converted to nanoseconds. Separate IMU metric coverage
uses the intersection of accel and gyro ranges after the measured offset.
Missing calibration is retained as null evidence but cannot set
`metric_vio_allowed`; unknown K is never replaced with an identity matrix.
Distortion with more than four coefficients is retained for review and blocks
metric OpenVINS admission rather than being truncated.

The service exposes `POST /api/scans/sensor-bundle` alongside the unchanged
video endpoint. It publishes import, coverage, calibration, timestamp, and
metric-admission state, prepares exact source frame timestamps, and offers
`POST /api/scans/{scan_id}/initiate-vio`. The page exposes browser camera/motion
capture, existing-video upload, and native sensor-bundle import as separate
inputs. For native synchronized bundles, prepared frames carry their exact
`capture_time_ns` and `source_frame_index`; the VIO adapter matches output rows
by exact timestamp and then binds prepared-frame hashes.

Before a phone recording's first metric-VIO run, record the native settings
above, verify the timestamp source in the export, calibrate intrinsics/distortion
at the encoded resolution, measure `T_imu_camera` in that same encoded camera axis, and
estimate the camera/IMU clock offset from a motion sequence. Record all values
in the manifest. Until those checks exist for the phone recording, the bundle
can support RGB reconstruction and raw sensor review but is blocked from
metric VIO.

## WO-2B native estimator

The selected conventional estimator is the ROS-free OpenVINS source at pinned
commit `69488123ed9362dd44b6f28e7f4680abbff1442b` from
<https://github.com/rpng/open_vins>. It was built in the configured large
reconstruction storage with ROS and ArUco disabled and the repository-owned
`tools/mapanything_phone_scan/native/openvins_phone_runner.cpp` bridge. The
bridge source/build is `tools/mapanything_phone_scan/native/` and the exact
portable build target is `openvins_phone_runner` (the bridge is an external
CMake target and is never copied into an upstream checkout). Dependencies,
source, and build artifacts are kept outside the repository. No Docker, neural
weight change, or double-integration fallback is used.

The reproducible storage-only build is:

```sh
REPO_ROOT=$(git rev-parse --show-toplevel)
UPGRADE_ROOT=<configured-reconstruction-storage>
OVROOT="$UPGRADE_ROOT/open_vins"
LIB_BUILD="$UPGRADE_ROOT/open_vins-build-20260904"
BRIDGE_BUILD="$UPGRADE_ROOT/openvins-phone-bridge-build-20260904"
DEPROOT="$UPGRADE_ROOT/deps/root"
git clone https://github.com/rpng/open_vins.git "$OVROOT"
git -C "$OVROOT" checkout 69488123ed9362dd44b6f28e7f4680abbff1442b
git -C "$OVROOT" apply \
  "$REPO_ROOT/tools/mapanything_phone_scan/native/openvins_storage_compat.patch"
cmake -S "$OVROOT/ov_msckf" -B "$LIB_BUILD" -DENABLE_ROS=OFF \
  -DENABLE_ARUCO_TAGS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$DEPROOT/usr;$DEPROOT/usr/lib/x86_64-linux-gnu" \
  -DCMAKE_LIBRARY_PATH="$DEPROOT/usr/lib/x86_64-linux-gnu" \
  -DCMAKE_INCLUDE_PATH="$DEPROOT/usr/include;$DEPROOT/usr/include/eigen3"
cmake --build "$LIB_BUILD" --target ov_msckf_lib -j2
cmake -S "$REPO_ROOT/tools/mapanything_phone_scan/native" -B "$BRIDGE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DOPENVINS_ROOT="$OVROOT" \
  -DOPENVINS_BUILD="$LIB_BUILD" -DDEPENDENCY_ROOT="$DEPROOT"
cmake --build "$BRIDGE_BUILD" --target openvins_phone_runner -j2
```

The pinned checkout is patched by
`tools/mapanything_phone_scan/native/openvins_storage_compat.patch`; its
`use_mask=true` path fails explicitly when optional image-codec support is not
available. The checked-in
`tools/mapanything_phone_scan/native/openvins_phone_base_config.yaml` is the
portable service tuning baseline. The built bridge carries a scoped DT_RPATH,
so the phone Python service receives only `NOESIS_PHONE_SCAN_VIO_EXECUTABLE`
and `NOESIS_PHONE_SCAN_VIO_CONFIG` and does not inherit a process-wide
`LD_LIBRARY_PATH`. The exact direct-run and service drop-in examples are in
`tools/mapanything_phone_scan/native/README.md`.

The bridge materializes imported video densely into an ASL layout, generates
OpenVINS camera/IMU YAML from the imported calibration, sets monocular mode
(`use_stereo: false`, `max_cameras: 1`), feeds IMU through the calibrated
CAM-to-IMU time shift plus a following bracket sample, and emits only a state
whose native timestamp matches the camera row. Its output is validated for
initialization, finite values, segment/reset identity, explicit frame/time
identity, and finite symmetric positive-semidefinite covariance. Materializer
output directories are unique and stale result files are removed before each
run; stdout/stderr are streamed to bounded log files.

The result contract is `noesis.phone_capture.vio_result.v1`:

```json
{
  "frame": {
    "source": "camera",
    "target": "vio_world",
    "pose_convention": "T_vio_world_camera",
    "camera_axes": "x_right_y_down_z_forward",
    "world_axes": "z_up_gravity_up",
    "pose_origin": "camera_optical_center",
    "velocity_origin": "imu_center",
    "gravity_frame": "vio_world",
    "gravity_semantics": "physical_world_acceleration"
  },
  "scale": {"mode": "metric", "source": "imu_camera_calibration"},
  "poses": [{
    "capture_time_ns": 0,
    "prepared_frame_id": "...",
    "source_frame_index": 0,
    "T_vio_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "velocity_mps": [0, 0, 0],
    "gravity_mps2": [0, 0, -9.81],
    "gyro_bias_rads": [0, 0, 0],
    "accel_bias_mps2": [0, 0, 0],
    "covariance": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
  }],
  "covariance_frame": "camera_pose_tangent_se3_row_major",
  "covariance_tangent_frame": "vio_world_rotation_additive_position"
}
```

The covariance is camera optical-center pose covariance with world-frame
rotation error and additive camera position; it is not labeled as a generic
left-SE(3) tangent. OpenVINS' estimator gravity compensation vector is
`[0,0,+9.81]` in its z-up world; the portable result reports physical gravity
as `[0,0,-9.81]` and keeps the compensation value separately. Camera velocity
is explicitly IMU-center velocity. VIO retains its relative gauge; it does not
establish the Noesis house origin or global yaw. WO-3 consumes consecutive
same-segment rows as relative constraints and does not import an absolute VIO
world pose as a house calibration.

## Execution evidence

The public synchronized EuRoC MH_01_easy recording was used with the first
500 monocular camera frames and 36,820 materialized IMU rows. The estimator
received only camera/IMU data and shipped calibration; EuRoC ground truth was
loaded by the separate evaluator after estimation. The selected camera span
is `28.049999872 s`, which is the recorded timestamp interval and is not
treated as a nominal 25-second clip. OpenVINS initialized at source frame 187,
retained frames 187 through 499 with no gaps or reset segments, and emitted
313 initialized states. Initialization delay was `10.9 s`, retained interval
was `17.149999872 s`, and tracking/retained coverage was `313/500 = 0.626`.

The native direct result is recorded under
`$NOESIS_RECONSTRUCTION_WORK_ROOT/benchmarks/MH_01_easy_mono_subset/openvins_result_snapshot2.json`.
The benchmark report is
`$NOESIS_RECONSTRUCTION_WORK_ROOT/benchmarks/MH_01_easy_mono_subset/openvins_benchmark_report_snapshot2.json`.
`tools/mapanything_phone_scan/evaluate_openvins_benchmark.py` converts the
ground-truth body trajectory to the calibrated camera optical center, matches
timestamps (maximum absolute match error 256 ns), and computes best-fit SE(3)
ATE. It reports best-fit Sim(3) scale only as a diagnostic:

| Measure | Result |
| --- | ---: |
| Camera-origin SE(3) ATE RMSE | 0.0690 m |
| Camera-origin SE(3) ATE median / max | 0.0596 / 0.1320 m |
| Sim(3) scale diagnostic | 1.02150 |
| Sim(3) ATE RMSE (diagnostic) | 0.0418 m |
| Initialized states / input frames | 313 / 500 |
| Reset segments / retained source gaps | 0 / 0 |

The generated imported-recording path was also run through the actual native
estimator, rather than only checking YAML or an executable path. The run
materialized 500 dense PNGs and 36,820 IMU rows, emitted 313 native poses, and
the application adapter selected 12 exact prepared-frame poses with IDs from
`prepared:smoke-0200` through `prepared:smoke-0475`. Its artifact is under
`$NOESIS_RECONSTRUCTION_WORK_ROOT/benchmarks/MH_01_easy_mono_subset/materializer_smoke/run8/vio_result.json`.

The complete HTTP producer-to-consumer path was exercised with a calibrated
public EuRoC MH_01 sensor bundle. `POST /api/scans/sensor-bundle` admitted the
bundle with `metric_vio_allowed=true`; normal frame preparation produced 39
service-generated prepared frames, and `POST /api/scans/{scan_id}/initiate-vio`
ran the configured native OpenVINS bridge to produce 30 camera poses. The
saved result was then passed to the trajectory consumer using the exact
service-produced image hashes and frame IDs: `_load_prepared_frames` loaded
39 rows and `_validate_vio_constraints` produced 29 consecutive
`verified_vio_relative` edges with zero skipped gap/reset pairs. No prepared
IDs were manually substituted. The scan state, calibrated import report,
prepared manifest and selected image bytes, native input metadata/logs, saved
result, and direct-consumer evidence are preserved under
`$NOESIS_RECONSTRUCTION_WORK_ROOT/benchmarks/MH_01_easy_mono_subset/http_public_euroc_e2e_20260905/`.
This is a public EuRoC benchmark execution proof and is not Fold 8 Ultra
capture or household calibration evidence.

The endpoint producer in
`tools/mapanything_phone_scan/solve_pcf_connector_multianchor.py` now accepts
only `noesis.pcf.connector_frame_binding_input.v3`. It reads the source and
target frame identities from the exact SHA-256 addressed manifests, derives
the moving/fixed baseline revisions from those manifest bytes, and requires
each nonidentity endpoint map to an existing accepted registration report with
an exact endpoint, matrix, transform digest, and manifest digest. These
intermediate reconstruction-frame edges do not mint physical calibration
provenance; the final WO-1 binding still requires its actual calibration and
world-alignment artifacts. The emitted composition is
`target_from_fixed_baseline @ graph_edge @ moving_baseline_from_source`.
Without proven endpoint conversion it emits an honest review-only graph edge.
`tools/mapanything_phone_scan/test_solve_pcf_connector_multianchor.py` runs
the actual graph solver with noncommuting endpoint rotations, composes the
result, and feeds that actual producer report into
`build_accepted_room_to_home_binding`; it passed `4 passed`. The same test
also rejects stale manifest bytes and wrong/arbitrary baseline provenance and
covers a manifest-owned identity edge without a conversion artifact. It
passed `5 passed`.

Focused validation passes:

```text
python3 -m pytest -q \
  tools/mapanything_phone_scan/test_capture_vio.py \
  tools/mapanything_phone_scan/test_trajectory_refinement.py \
  tools/mapanything_phone_scan/test_solve_pcf_connector_multianchor.py
20 passed
```

The tests cover safe archive members, exact integer-nanosecond import and
video duration, separate sensor units and endpoint intersection, missing and
higher-order calibration admission, exact VFR/nonzero-PTS frame identity,
precomputed stream interpolation, and covariance rejection. The actual phone
recording, Fold 8 Ultra Camera2 timestamp capability, phone calibration, and
metric reconstruction accuracy remain unverified until the user supplies a
native recording and calibration evidence.
