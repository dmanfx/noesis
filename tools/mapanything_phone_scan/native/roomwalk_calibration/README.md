# RoomWalk ChArUco camera–IMU solver

This is a standalone CPU consumer of Basalt's pinned spline optimizer, not a
replacement VIO estimator. It consumes explicit ChArUco 3D coordinates and
indexed, full-D5-undistorted 2D corners from `roomwalk_calibration.py`. Its
AprilGrid-named upstream methods do not convert the target: the point table
is exactly the ChArUco table. Independent accelerometer and gyroscope times
are passed separately without nearest-neighbor pairing or ROS bag conversion.

## Build and test

From the repository root, choose an existing large-storage parent:

```sh
python3 tools/mapanything_phone_scan/native/roomwalk_calibration/build.py \
  --storage-parent "$CALIBRATION_STORAGE_PARENT"
```

The builder makes a unique directory, downloads hash-pinned source archives,
builds only `roomwalk_calibrate_imu` with one compile job, and prints its path.
It needs the host's CMake, C++17 compiler, and TBB development package. It uses
Eigen 5.0.1 and does not link to the separately published Basalt binary ABI.
No global installation, ROS, Pangolin, camera service, GPU, or OpenVINS rebuild
is involved. The build receipt records archive and adapter hashes; building
alone does not claim functional inference or calibration validation.

```sh
export NOESIS_PHONE_SCAN_CALIBRATION_SOLVER="$CALIBRATION_BUILD_ROOT/build/roomwalk_calibrate_imu"
OPENBLAS_NUM_THREADS=1 python3 -B -m pytest -q -p no:cacheprovider \
  tools/mapanything_phone_scan/test_roomwalk_calibration.py
```

Native tests run only when this explicit executable is configured. They fit
actual synthetic observations with independent sensor clocks, nonidentity
rotation/translation and both offset signs. A further test fits nonidentity
accelerometer/gyro correction matrices, exercises Python preparation, and
checks a temporally independent camera/gyro holdout. A 12 ms rolling-shutter
fixture exercises point-specific spline timestamps, a separate visual-only
holdout, and rejection of deliberately wrong translation/time estimates.
No ground truth transform
or timing value is supplied to the optimizer as an observation.

For retained standalone evidence, use `synthetic.py --executable ... --output
<new-directory> --offset-ns -23000000 --skew-ns 12000000`. The builder accepts
`--archive-cache <prior-build-root>` to reverify and re-extract only its seven
pinned archives without downloading again. Extracted source trees are not reused.
Synthetic accuracy validates numerical
execution, not phone calibration or surveyed room accuracy.

## Worker contract

`run_calibration(capture_dir, request, output_dir, *, settings, progress,
cancelled)` reads an already imported native capture and creates a new,
separate output directory. `progress` receives `(fraction, message)`;
`cancelled` returns a boolean. The returned report and `report.json` use
`roomwalk.calibration_report.v1`. Every artifact has an output-relative path,
byte count and SHA-256. Failed and cancelled jobs retain their report/evidence.
The function never alters the capture or selects a calibration for runtime use.

Server camera/motion workers cap OpenCV at two threads. Camera jobs keep a
20-minute budget; dense 90-second motion jobs use 25 minutes beneath the parent's
30-minute deadline. Native encoded pixels, 10 Hz motion sampling and the 900-view
cap are unchanged. A short test proves capture timing, not motion-fit quality.

The request uses `roomwalk.calibration_request.v1`, mode `camera` or `imu`, and
the explicit board definition. Defaults are 10×14 squares, 0.018 m square,
0.0132 m marker, `DICT_4X4_1000`, marker IDs 300 through 369 and legacy false.
Different dimensions/IDs are validated; contradictory supplied corner/marker
coordinates are rejected. `camera_calibration_id` and `noise_calibration_id`
are opaque IDs resolved by the server, never client filesystem paths.
`board_geometry_confirmed` is an explicit boolean user attestation of measured
physical dimensions/flatness, not a metrological certificate. Its absence blocks
metric lever-arm qualification but not dimensionless camera intrinsic fitting.

Server settings may be `CalibrationSettings` or a dictionary:

```python
{
    "camera_calibration_dir": prior_camera_artifact_directory_or_none,
    "noise_calibration_dir": prior_noise_artifact_directory_or_none,
    "solver_path": native_executable_or_none,
}
```

The camera directory contains a completed camera `report.json` and its hashed
`camera_result.json`. The backend verifies that hash, all five coefficients,
native dimensions and actual lens/focus/crop binding. A missing/unqualified
camera can be fitted for diagnostic IMU use only when the request explicitly
sets `allow_provisional_camera: true`. An incompatible existing reference is
rejected even in provisional mode.

Qualified camera fits additionally export `camera_profile.json`, with
`intrinsics {fx,fy,cx,cy}`, five `distortion` coefficients in OpenCV order,
`distortion_model: brown_conrady`, native `resolution_px`, and the exact `binding`.
`camera_result_sha256` hashes canonical JSON content; the report separately hashes
each artifact's file bytes. `camera_profile(result)` reproduces this export.
`camera_binding_for_capture(capture_dir)` rechecks every original native
association/optical setting in an imported walk, with a 20,000-frame/ten-minute
bound and no video decode, hash or calibration-duration constraint. It returns
the same binding shape; require both bindings qualified and their hashes equal.
Malformed or unverified timing raises `CalibrationError` rather than producing
a matching binding. This API never changes the capture or activates a profile.

An optional noise directory contains `noise_result.json`, listed with
`path`, `sha256`, and `bytes` in `report.json.artifacts`. It must carry
`imu_noise_calibrated`, a four-coefficient `noise` object, `unit_conventions`,
and `binding.device` / `binding.sensors` matching native metadata. The default
short-session method instead declares `noise_model_usable: true` only after its
measured densities, heldout checks and conservative drift priors reproduce the
retained evidence. It keeps `imu_noise_calibrated: false`. This usable model may
weight the fit but needs the separate fixed OpenVINS profile check before use on
short walks; it does not weaken extrinsic/timing qualification. Comparison
excludes per-session sensor sample counts/timestamps. Density units are
`rad/s/sqrt(Hz)` and `m/s^2/sqrt(Hz)`; random-walk units are
`rad/s^2/sqrt(Hz)` and `m/s^3/sqrt(Hz)`. Unknown units or mismatched sensors do
not qualify noise. Qualified densities are converted using each stream's own
observed cadence; otherwise explicit provisional residual weights are retained.

## Numerical and authority boundaries

- Camera fitting uses the original encoded pixel geometry and estimates all
  five Brown–Conrady coefficients. Two-second blocked holdouts and separate
  per-view pose-fit/scoring corners do not enter the returned camera fit.
  The conservative camera policy is versioned and reported, including native
  pixel residuals, coverage, tilt span and reverse-fit focal stability.
  A null distortion mode can qualify only as a distinct encoded-output model
  when both request/result capability keys are explicitly unavailable, no
  selectable modes are advertised, and the locked-focus software fingerprint
  is retained. Its binding records unsupported control and never assumes OFF.
  Missing output from an available control remains a rejection. This optical
  binding alone cannot qualify rolling-row geometry; the separate mapper must
  recheck the original capability, routing, geometry and per-frame evidence.
- IMU fitting holds the camera fixed, initializes rotation/time from camera
  rotation increments and gyro evidence, then uses Basalt's visual/inertial
  spline residuals. Bootstrap camera poses are removed as residuals after
  spline initialization. Final-quarter camera and IMU samples are held out;
  a separate visual-only spline receives only withheld corners/initial PnP poses,
  never IMU data or the training trajectory. Its raw-D5 pixel residuals, rotation
  increments, triangular-integrated acceleration and lever-arm design rank
  qualify or reject the returned calibration without refitting it. The
  independent time diagnostic must agree within 1 ms and show angular excitation.
- Each corner is a separate upstream measurement at its exposure midpoint.
  Camera2 skew spans the full active array, so the mapper accounts for centered
  per-stream aspect cropping. It requires recorded physical-sensor geometry and
  matching physical-result identity, default sensor pixel mode, equal active and
  pre-correction arrays, and no zoom/stabilization/rotation. Distortion mode must
  be reported OFF, or the narrow unsupported-control contract below must pass.
  It does not infer a logical-camera-to-physical-sensor mapping. Missing evidence
  produces an explicit diagnostic-only sensor-timestamp approximation and blocks
  extrinsic/time qualification. The current logical-camera RoomWalk recording
  does not establish that mapping merely by reporting an active physical ID.
  The companion now retains nested physical capture results when the device
  actually supplies them. Their own physical ID, exposure controls and exact
  sensor timestamp must match the corresponding logical exposure before use;
  a producer-supplied boolean is not sufficient.
- For a directly pinned physical output, Camera2's unsupported-control active-
  array contract is checked separately. Both request/result keys must explicitly
  be unavailable, the retained mode-list value null or empty, every physical
  result explicitly null for that mode, and physical output/characteristics/
  focus IDs and software fingerprints must agree. Active/pre-correction arrays
  must be equal, integer and zero-origin; all existing crop/pixel-mode/exposure/
  stabilization checks remain. A missing mode-list field is rejected. The result
  records `camera2_unsupported_control_same_physical_array_v1`, keeps the raw mode
  null and `assumed_off: false`, and never admits a logical-camera inference.
  This follows Android's [active-array contract](https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#SENSOR_INFO_ACTIVE_ARRAY_SIZE)
  and AOSP's [unsupported-mode detection](https://android.googlesource.com/platform/frameworks/av/+/dd833ba601e642d72da2200a2fd9c3fa1cb69d2d/services/camera/libcameraservice/device3/DistortionMapper.cpp).
  Independent motion holdouts still qualify or reject the fitted timing.
- Stationary/video sensor identity comparison uses exact IEEE float32 values
  for Android `Sensor.getMaximumRange()` and `getResolution()`. The two JSON
  producers emit different decimal spellings of the same native float. No
  numeric tolerance is used: even a one-ULP change is rejected. All other sensor
  identity fields, device/software binding, raw noise data and hashes stay exact.
- The correction equation is `corrected = matrix * raw - bias`. Full matrices,
  biases and original parameter vectors are exported. The solver's one-Hz
  weight normalization is a computational convention, not sensor cadence or
  an IMU noise estimate. Actual timestamps and rates remain separately recorded.
- `T_imu_camera` maps camera to IMU coordinates. Basalt uses
  `t_imu = t_camera + cam_time_offset_ns`; Noesis's
  `imu_to_camera_offset_ns` is its negative. Camera optical axes are unchanged
  by corner undistortion. Here `t_camera` is the point's exposure midpoint in the
  Camera2 sensor clock, not an uncorrected encoded-frame callback timestamp.
- Visual-only fitting has an unobservable internal body/extrinsic split. Its
  observable convergence criterion is three successive maximum pixel changes
  below 1e-5 px and per-point objective changes below 1e-12, or the upstream state
  step criterion. This does not relax the separate native-pixel accuracy gates.
- Time/lever-arm bounds and convergence do not alone establish accuracy.
  `camera_imu_extrinsics_calibrated` and `time_offset_calibrated` are data-driven
  heldout-evidence flags. Explicit residual weights remain provisional until a
  separate stationary-noise result qualifies. Statistical covariance is `null`;
  translation residual sensitivity is labeled as such, not as covariance.
- Every output remains review-only with `accepted_for_metric_vio: false`.
  Camera quality may qualify intrinsics for a matching capture; it supplies
  neither qualified camera–IMU extrinsics nor measured noise by itself.
  Downstream IMU-correction and row-time consumers remain separate admission
  gates even when the calibration worker's heldout checks pass.

The source archives carry their upstream licenses. Basalt and basalt-headers
are pinned to `6d8637b9d68ea18a1a63c1baa72818779e156932` and
`aa441ba3e51050c47ba1902537792a2e4db7e43d` respectively. The adapter adds input
validation, resource bounds and serialization around those optimizer templates.

## Separate short-profile consumer check

`motion_profile.py` assembles hash-bound camera, camera–IMU and noise results.
It runs OpenVINS with fixed calibration, never online refinement, and compares
the camera-origin trajectory with the independent visual-only target holdout.
A single first-pose rigid anchor resolves coordinate origins and is excluded
from scoring. Scale is diagnosed but never fitted/applied. Versioned application
limits require at least 5 seconds/20 poses, 80% heldout coverage, gaps no larger
than 0.2 seconds, 0.15 m motion extent, position RMSE/p95 below 0.10/0.20 m,
orientation RMSE/p95 below 3/6 degrees and scale error below 5%. These are
short-profile acceptance limits, not surveyed room-accuracy claims.

Only a passing check can be explicitly selected for future matching native
walk imports, bounded to five minutes. The selected profile retains the measured
motion envelope, native executable/config hashes and consumer version. Reuse
revalidates physical exposure timing, optical/sensor identities and original
frame mapping. The OpenVINS consumer applies the actual correction matrices,
conservative corrected noise and centre-exposure timing; its VIO-only full-D5
analysis can be scaled to 1280 pixels without altering source/reconstruction
images or cadence. A centre-timed global-shutter approximation is declared;
rolling-shutter compensation is not claimed. Source calibration reports stay
review-only. This downstream capability never admits Noesis world/PCF geometry.
