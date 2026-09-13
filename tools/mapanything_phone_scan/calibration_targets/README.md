# Native camera and IMU calibration target

This target is for the offline Basalt 0.1.7 camera/IMU calibration consumer.
OpenVINS remains the phone VIO estimator, and MapAnything + DA3 remain the
reconstruction providers. Recording a target does not admit a calibration.
This is an optional solver reference. A fresh RoomWalk capture does not require
another board session or a long stationary recording; native video, IMU streams
and paired static evidence are captured and retained together.

The target uses AprilTag 36h11 IDs 0–35, six columns and six rows, with IDs 0–5
across the bottom. Each outer black tag edge is 25.4 mm, and the white gap is
7.62 mm (`tagSpacing: 0.3`). Basalt's pinned detector uses a two-cell black
border; the ordinary one-cell marker default does not match it. The JSON beside
the PDF supplies its four geometry parameters. Confirm physical dimensions on
the actual printed sheet and keep it flat; digital PDF geometry is not evidence
of printer scale or board flatness.

The generator writes vector cells and a 100 mm measurement line on US Letter.
The generator rotates each OpenCV marker by 180 degrees, as established by
the direct pinned-detector check: p0 bottom-left, then bottom-right, top-right,
top-left. Native detection of the rendered final PDF accepted all 36 tags and
144 corners. Changing marker orientation without changing the target's corner
convention can produce inconsistent geometry even when IDs decode correctly.

The official [Basalt calibration guide](https://github.com/VladyslavUsenko/basalt/blob/6d8637b9d68ea18a1a63c1baa72818779e156932/doc/Calibration.md)
describes the solver. The isolated 0.1.7 binaries support `--no-gui`. This pinned
version's EuRoC reader expects two cameras; the phone uses a **monocular ROS1 bag**
with one `sensor_msgs/Image` topic and one `sensor_msgs/Imu` topic. Compressed
image messages are unsupported in this version, while bag-level compression
preserves raw `mono8` image messages. Keep encoded geometry and exact integer
camera timestamps. Derive combined IMU rows with explicit bounded interpolation
and retain the original independent streams and conversion provenance.

Fit native camera intrinsics/distortion on the encoded image geometry and
validate them on withheld target views. A five-coefficient Brown-Conrady model
can be fully rectified to unchanged K and optical axes before the camera/IMU
solver uses a pinhole camera model. Do not truncate the distortion vector or
silently substitute a different camera model. The original stock-app ChArUco
profile is retained but has no automatic native-capture binding.

Basalt's headless camera/IMU path holds camera intrinsics fixed, first converges
extrinsics, then refines time offset and IMU scale/misalignment. Preserve the
entire solver result, including those IMU correction matrices; they may not be
discarded when converting to the OpenVINS sensor model. Its convention is
`t_imu = t_camera + cam_time_offset_ns`, so the Noesis
`imu_to_camera_offset_ns` has the opposite sign. Validate the convention and
residual alignment on withheld movement before metric admission. Rolling-shutter
readout remains a separate modeling and validation concern.

Long stationary recordings feed Allan-deviation analysis using their measured
sample cadence. Review time gaps, stationarity, temperature settling, each
axis's curve and the observable noise regions before fitting coefficients.
The installed AllanTools library's synthetic white-noise test proves library
execution only; no generic or synthetic coefficients are assigned to the phone.
