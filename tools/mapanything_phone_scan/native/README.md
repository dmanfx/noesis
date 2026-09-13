# Native OpenVINS phone bridge

`openvins_phone_runner.cpp` is the ROS-free bridge used by the phone-scan VIO
adapter. It feeds OpenVINS `ImuData` and `CameraData` measurements directly and
exports camera optical-center poses, state covariance, initialization quality,
segment/reset identity, and the explicit camera/world frame contract expected
by `vio.py`. It does not integrate IMU data itself and has no fallback
estimator.

The selected estimator source is the official OpenVINS repository at commit
`69488123ed9362dd44b6f28e7f4680abbff1442b`:
<https://github.com/rpng/open_vins>. Keep that checkout, build, and dependencies
under the configured reconstruction upgrade storage directory. The runner
accepts an ASL/EuRoC `mav0` directory (`cam0/data.csv`, `cam0/data/*.png`, and
`imu0/data.csv`) for the benchmark proof. The app adapter materializes the same
bounded layout from an imported phone bundle after calibration and frame
preparation.

The bridge is a repository-owned CMake target in this directory. It links the
already-built upstream `ov_msckf_lib` from an explicitly supplied checkout and
does not copy files into the upstream tree. The upstream checkout needs the
small, checked-in `openvins_storage_compat.patch` to select the installed
OpenCV components and avoid an optional image-codec dependency in its mask
loader. Keep the phone base configuration's `use_mask: false`; the bridge's
explicit `--camera-mask` path loads and validates its own static grayscale mask
and supplies it through each `CameraData` measurement. Five-coefficient
Brown-Conrady inputs use this path after full-image rectification. The adapter
hashes the mask, and the result must confirm that the native bridge applied it.
A missing, wrong-size or nonbinary mask fails explicitly. No custom target is
injected into OpenVINS.

From a fresh pinned checkout, build the upstream library and then this bridge
in the bounded storage directory (replace `<upgrade-root>` with the configured
large storage directory):

```sh
REPO_ROOT=$(git rev-parse --show-toplevel)
OVROOT=<upgrade-root>/open_vins
LIB_BUILD=<upgrade-root>/open_vins-build-20260904
BRIDGE_BUILD=<upgrade-root>/openvins-phone-bridge-build-20260904
DEPROOT=<upgrade-root>/deps/root
git clone https://github.com/rpng/open_vins.git "$OVROOT"
git -C "$OVROOT" checkout 69488123ed9362dd44b6f28e7f4680abbff1442b
git -C "$OVROOT" apply \
  "$REPO_ROOT/tools/mapanything_phone_scan/native/openvins_storage_compat.patch"
cmake -S "$OVROOT/ov_msckf" -B "$LIB_BUILD" \
  -DENABLE_ROS=OFF -DENABLE_ARUCO_TAGS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$DEPROOT/usr;$DEPROOT/usr/lib/x86_64-linux-gnu" \
  -DCMAKE_LIBRARY_PATH="$DEPROOT/usr/lib/x86_64-linux-gnu" \
  -DCMAKE_INCLUDE_PATH="$DEPROOT/usr/include;$DEPROOT/usr/include/eigen3"
cmake --build "$LIB_BUILD" --target ov_msckf_lib -j2
cmake -S "$REPO_ROOT/tools/mapanything_phone_scan/native" -B "$BRIDGE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DOPENVINS_ROOT="$OVROOT" \
  -DOPENVINS_BUILD="$LIB_BUILD" -DDEPENDENCY_ROOT="$DEPROOT"
cmake --build "$BRIDGE_BUILD" --target openvins_phone_runner -j2
```

The checked-in `openvins_phone_base_config.yaml` is the portable tuning
baseline. `vio.py` copies it into each materialized capture and replaces the
relative camera and IMU calibration paths with measured per-capture files; a
benchmark dataset YAML is not a service default. For a user service, the
operator can apply a drop-in with these values after substituting local paths:

```ini
[Service]
Environment=NOESIS_PHONE_SCAN_VIO_EXECUTABLE=<upgrade-root>/openvins-phone-bridge-build-20260904/openvins_phone_runner
Environment=NOESIS_PHONE_SCAN_VIO_CONFIG=<repo-root>/tools/mapanything_phone_scan/native/openvins_phone_base_config.yaml
```

The bridge executable carries a scoped DT_RPATH for its OpenVINS and dependency
libraries, so this service drop-in does not set a process-wide
`LD_LIBRARY_PATH`. The exact path in a concrete installation is the bridge
build directory produced by the commands above.

For a direct run on an ASL/EuRoC layout:

```sh
"<upgrade-root>/openvins-phone-bridge-build-20260904/openvins_phone_runner" \
  --capture-dir "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/mav0" \
  --config "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/mav0/estimator_config.yaml" \
  --output "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/openvins_result.json" \
  --capture-id benchmark-mh01 --camera-sensor-id cam0 --time-domain euroc_ns
```

The accepted baseline runs in monocular plus IMU mode (`use_stereo: false`,
`max_cameras: 1`). Evaluate it with ground truth held out from estimation:

```sh
python3 tools/mapanything_phone_scan/evaluate_openvins_benchmark.py \
  --result "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/openvins_result.json" \
  --ground-truth "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/mav0/state_groundtruth_estimate0/data.csv" \
  --sensor-yaml "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/mav0/cam0/sensor.yaml" \
  --camera-data "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/mav0/cam0/data.csv" \
  --subset-frames 500 \
  --output "<upgrade-root>/benchmarks/MH_01_easy_mono_subset/openvins_benchmark_report.json"
```

The evaluator converts the EuRoC body trajectory to the camera optical center,
aligns ATE by SE(3), and reports a best-fit Sim(3) scale separately. It records
the retained interval, initialized/unemitted frames, frame gaps, reset
segments, and integer nanosecond matching error. The shipped run retained 313
of the first 500 dense monocular frames (0.626 camera coverage), initialized
after 10.9 s, and retained 17.149999872 s of the 28.049999872 s input span.
Camera-origin SE(3)-aligned ATE was 0.069021 m RMSE, 0.059565 m median, and
0.132038 m maximum; the diagnostic Sim(3) scale was 1.021498 and its RMSE was
0.041802 m. The exact result and evaluator artifacts are
`<upgrade-root>/benchmarks/MH_01_easy_mono_subset/openvins_result_snapshot2.json`
and
`<upgrade-root>/benchmarks/MH_01_easy_mono_subset/openvins_benchmark_report_snapshot2.json`.
This public EuRoC result validates estimator execution and the bridge contract;
it is not evidence about the Fold 8 Ultra or a household scene.

The estimator configuration must supply camera intrinsics, a supported exact
distortion model, the camera-to-IMU transform, IMU noise, and measured time
offset. The importer keeps raw captures when any of those are missing but
blocks metric VIO admission. The bridge feeds the dense camera stream and
separate accel/gyro streams through the calibrated offset, retains bracketing
IMU samples, and emits only states whose timestamp matches the exact camera
row. It carries source frame identity, camera-origin pose, IMU-center velocity,
physical gravity semantics, and the declared
`vio_world_rotation_additive_position` covariance convention. OpenVINS gravity
is reported in its z-up world frame; Noesis y-up conversion belongs at the
downstream presentation boundary.

For Android, use the [RoomWalk companion](../android_companion/README.md) and
retain its exact Camera2/encoder timing rows and separate native IMU streams.
The [capture and calibration workflow](../README.md#native-calibration-recordings)
keeps optional sensor diagnostics separate from normal room walks. A model name
or matching encoded dimensions does not prove calibration transfer.
Raw capture remains importable while any measurement is missing; metric VIO
remains blocked. Browser callback times never establish acquisition timestamps.

## Provisional online-calibration experiment

`materialize_openvins_calibration_input(..., prior=prior)` and
`run_openvins_calibration(...)` provide a separate offline experiment on retained
native captures. A `noesis.phone_capture.vio_calibration_prior.v1` input must
declare provisional camera geometry, IMU noise priors, initial extrinsics/time
offset, uncertainty and provenance for the exact capture ID. The wrapper keeps
the original import report unchanged. Its optional analysis resize preserves
exact camera times and transforms K with the pixel-center convention; complete
D5 rectification and the native invalid-pixel mask still apply.

This path enables bounded online extrinsic/time refinement, reads updated
calibration when feeding IMU and converting camera poses, and exports calibration
covariance and per-frame history. Dynamic initialization does not enable the
pinned initializer's incomplete calibration-return path. Bounds are experiment
limits, not accuracy or admission thresholds. Its result schema is
`noesis.phone_capture.vio_calibration_result.v1` and always declares
`accepted_for_metric_vio: false`. It cannot substitute for the app's admitted
metric-VIO result.

The retained 623-frame phone walk completed this experiment on 2026-09-10 and
emitted 521 states. Its translation drift was grossly inconsistent with the
short room walk, so the result is rejected for metric use. Execution and
covariance-export checks do not make that trajectory useful. No new phone
recording or board session is required by this experiment; it is not a
prerequisite for the native video/IMU/static capture workflow.
