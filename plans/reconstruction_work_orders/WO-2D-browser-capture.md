# WO-2D: Record camera and IMU inside Room Walk

Status: implementation and deployment complete; deployed review evidence is
complete, 2026-09-05. The Fold 8 Ultra's own field capture remains unverified.

The user wants the existing Room Walk capture button to preserve camera video,
accelerometer, gyroscope, and available timing information directly from Android
Chrome, without requiring a separate recorder or calibration before capture.
The earlier native-bundle importer did not complete this browser workflow.

## Work orders

1. Browser capture: replace the default external-camera file input with an
   in-page preview and recorder. Verify both sensor streams before enabling
   recording; show their live counts. Retain device-frame units, sensor or event
   timestamp provenance, video-frame metadata, camera settings, interruptions,
   and missing information explicitly. Preserve a downloadable capture and
   retry after upload errors. Keep work bounded and stop cleanly on interruption.
2. Persistence: accept the browser producer's TAR through the existing sensor
   bundle endpoint using the separate `noesis.phone_capture.browser.v1` schema.
   Preserve original video and sensor observations, expose saved counts and
   timing diagnostics, and run normal adaptive RGB preparation. Browser callback
   times must never become fabricated native acquisition timestamps.
3. Secure access: use the existing appliance certificate infrastructure to serve
   the capture page through a secure LAN origin. Preserve normal existing
   consumers where practical; document any actual phone certificate-trust step.
4. Review: check producer-to-importer compatibility with a real encoded video,
   saved sensor rows, and prepared RGB views; check missing sensors, interrupted
   capture, retry, malformed input, and native-import compatibility. Restart the
   affected service and verify its direct HTTP/HTTPS consumers. Actual Fold
   sensor delivery must be demonstrated on that device before claiming it.

## Deployment and evidence

The affected service was restarted and remains active. Both the retained HTTP
health endpoint and the hostname-verified HTTPS health endpoint are healthy.
Chrome loads `https://TauntonMainframe.local:8789`, and **Record phone walk**
shows the live sensor panel. On the desktop review surface, no rear camera is
available, so the recording start control remains disabled; this is an
environment limitation rather than evidence of Fold capture.

The deployed service imported `root12s-browser-smoke.tar` using the browser
producer schema and created scan `20260905-175814-c45afff2`, named
**Browser capture verification (simulated IMU)**. The scan reached `ready` with
15 adaptive RGB views from 48 candidates. The persisted capture retained the
browser camera video, two accelerometer rows, and two gyroscope rows. The
capture remains explicitly non-metric: `metric_vio_allowed=false`, native VIO
compatibility is false, and browser callback timing is not promoted to native
acquisition time. The encoded video contains 360 frames at 640x480 with a
validated ffprobe duration of 11.967 seconds.

The narrow prepared-video duration fallback was then exercised by importing the
same producer fixture as scan `20260905-180446-7df21485`, named **Browser
duration verification (simulated IMU)**. It reached `ready` with 15 adaptive
views from 48 candidates. The persisted prepared-state probe and prepared-frame
manifest both report `duration_s=11.967` and
`duration_source=capture_import.video.encoded_duration_s`, so the earlier `0s`
display is corrected; Chrome's direct consumer view displays `12s` rounded from
the same duration. Both sensor counts remain visible in that view. All prepared rows retain null `capture_time_ns` and
`source_frame_index`; their derived candidate-period timestamps do not become
fabricated native acquisition timestamps. A live `initiate-vio` request returns
HTTP 409 because this browser capture remains uncalibrated and is not native
VIO-compatible.

The deployed asset, manifest, sensor-sample, and CA checks all returned HTTP
200. The import and prepared-state snapshots are retained in
`$NOESIS_RECONSTRUCTION_WORK_ROOT/browser_capture_smoke_20260905/`:

- `deployed-import.json`
- `deployed-prepared-state.json`

The corrected rerun's import, prepared-state, and prepared-frame-manifest
snapshots are retained alongside the original evidence:

- `deployed-duration-import.json`
- `deployed-duration-prepared-state.json`
- `deployed-duration-prepared-manifest.json`

The two simulated scans were bounded review fixtures. After checking their exact
names and `ready` state, both were deleted from the normal Room Walk list via
the API with HTTP 204; their source TAR and evidence files and reports remain
retained.

The narrow duration regression passed with 2 focused tests. Latest focused
validation passed with 12 Python tests and 5 JavaScript tests. The documentation
consistency check passed with 89 checks, and `git diff --check` passed. These
results establish the deployed producer, importer, persistence, and preparation
path using simulated browser IMU data; they do not establish real Fold 8 Ultra
sensor delivery, phone calibration, camera/IMU clock alignment, or metric VIO.

## Boundaries

- Calibration is not required to record and retain browser camera and IMU data.
- The first implementation does not assert calibrated metric VIO from browser
  data. Camera/IMU timing and calibration remain unverified until measured.
- Keep existing MapAnything/DA3 selection and reconstruction behavior.
- No changes to live tracking calibration, no runtime promotion ceremony, no
  broad regression suite, no commits or changes to unrelated dirty work.
- GPT-5.6-Luna agents implement the three bounded lanes; the parent reviews
  integration, requests corrections, and reports evidence and limitations.
