# RoomWalk for Android

A single Android workspace for native recording, optional calibration and the
RoomWalk reconstruction service. The current native integration is RoomWalk
0.4.2, versionCode 23; it packages the same web UI and 3D viewer as the browser
application. Native recording still uses Camera2, a hardware
MediaCodec surface encoder, MediaMuxer, and Android sensors; it never substitutes
WebView camera capture. Native 8K availability is
checked on the phone; this application does not assume that a stock-camera 8K
mode is available to Camera2.

## Current user flow

The workspace has three primary tabs:

- **Capture** chooses the purpose and records the phone evidence. **Reconstruction**
  means many overlapping room viewpoints; pairing a Noesis camera is optional,
  and adding views to an existing reconstruction is an explicit retained
  supplement. **Path refinement** means the same walker holds the phone against
  their own torso with elbows tucked and stable, then moves their whole body
  with the phone; it requires a paired Noesis camera and a completed reference
  reconstruction.
- **Library** retains the original video, raw capture, selected reconstruction
  views, supplements, qualified VIO/dense trajectory artifacts, paired
  references and review exports. Legacy captures are not implicitly relabelled
  and their raw evidence is unchanged.
- **Setup** contains the optional advanced camera/lens and camera–IMU
  calibration workflow; its board, focus, timing and holdout gates remain in
  force for that protocol, not as the primary normal-walk path.

Ordinary reconstruction and path-refinement capture do not require a saved
calibration selection or calibration-specific focus lock. RGB reconstruction
works without calibration. Native 8K/device capability and timing checks still
apply. Saved timing proofs distinguish automatic normal-walk mode from
locked-focus board mode: if only a pinned calibration check is retained, the
phone may need one 10-second automatic-mode timing check, not a new full
calibration. These bounded checks establish recording-path availability;
they do not make calibration a prerequisite for reconstruction.

For path processing, choose the entry marked **Noesis PCF** when using a retained
fused room. The native intent retains its exact revision, camera, manifest and
frame-binding digests. A changed or incompatible reference cannot silently fall
back to the original raw reconstruction. These fields survive saved captures,
packaging and upload. Reconstruction mode remains unchanged.

Choose DA3 before running the visual model, align the walk to its paired static
camera, then select **Review path**. A selected Noesis PCF room uses its already
verified registration; an ordinary raw reference needs its own matching world
alignment. The existing visual-revisit engine tests corrections
against a reserved late-walk interval and preserves the original path when a
candidate fails. Qualified VIO runs automatically when its capture already
admits it; otherwise the raw sensors remain usable for gyro diagnostics, not
uncalibrated position integration. Scale-changing candidates require fresh
independent static registration.

Library displays original/reference phone paths, an explicit Noesis lifecycle
selector, a time slider and JSON/CSV exports. No person is selected automatically.
Clock and registration limitations remain visible; camera-center versus ground
separation is not anatomical-position error or certified 10 cm accuracy. The
retained September 12 comparison is available without relabelling its older
carry protocol. Dense VIO and original raw recordings stay separate and retained.

RoomWalk 0.4.2/code 23 uses RoomWalk server 1.13.0 or newer. Install it
over the existing app with the same signing key to retain captures and settings.
The packaged app and intent boundaries were exercised on an emulator; native
8K recording and path accuracy still require the physical phone.

The following versioned notes preserve the optional board-calibration protocol.
In that protocol, version 0.3.4 bound locked-focus preview and recording
outputs to the measured physical lens using Android's
[physical output routing](https://developer.android.com/reference/android/hardware/camera2/params/OutputConfiguration#setPhysicalCameraId(java.lang.String)).
Manual focus alone does not prevent a logical camera from switching lenses.
The pinned stream uses that physical camera's actual capture-result timestamps
and controls, retaining separate logical-camera diagnostics. Missing physical
metadata or an unsupported 8K configuration stops explicitly; there is no
automatic lens or resolution substitution. Pinned recordings identify the
physical output camera separately from the logical device, so older logical-
camera calibration and timing proof cannot silently qualify the new path.

For full calibration takes, focus sharply at the intended working distance
before locking. Vary the phone's position and angle while keeping the board
sharp; nearer/farther movement is optional and must remain within that sharp
range. Do not refocus during the take. Published 0.3.5 prompts state this during
camera coverage and all moving phases of the 90-second motion sequence.
Recording settings, physical-lens routing and focus/test bindings are unchanged
from 0.3.4; installing 0.3.5 does not itself require repeating a matching test or
calibration. Install over the existing app to retain its settings and captures.

In the optional board-calibration workflow, version 0.3.6 requires an explicit
qualified camera reference and measured-board confirmation before the guided
motion capture or processing action. It preserves
the user's board confirmation with its unchanged saved definition, offers
**Check phone** when reopening leaves the camera list unloaded, and labels
connection failures as stale status while retrying with bounded backoff.
Processing failures lead to the retained take's settings, not an automatic
request to record again. The current native focus metadata is checked against a
loaded camera reference before opening motion capture; a retained native guard
also blocks Record if focus is unlocked or changed inside that viewer. These
preflights do not replace the backend's exact captured-focus binding. Native
acquisition and physical routing are unchanged; an app update cannot repair a
take recorded at a different focus.

For that optional locked-focus board mode, **run a new ten-second test when
upgrading from 0.3.3 or earlier**, keeping focus locked and moving gently within
the sharp range. Fixed focus can blur outside that range; it must not silently
change lenses. Ordinary reconstruction/path-refinement mode uses its automatic-
mode timing proof when needed and does not require a new full calibration. A
stopped preview offers **Restart
preview** in the same viewer. Preview and ongoing camera/encoder acquisition
also detect five seconds without frames. Early recording stops show the actual
duration and reason; partial originals remain in Library, not marked as completed
calibration takes. Device-specific physical-lens 8K support still requires the
phone test; emulator output/metadata routing checks are not that proof.

The camera-close/save handoff repair is retained: the exact recording stays
pending until its ZIP is ready, then offers its upload action. **Check/reuse**
explains missing processed results and lets you resume existing full calibration
takes from the phone or server. Failed result loading is visible and retryable;
ten-second timing tests cannot substitute for full calibration takes. Selecting
a retained recording restores its own calibration settings, not the previous
take's context. Nothing uploads or processes merely by choosing a recording.

Calibration recording, upload, processing and the next step remain in one
guided view. A five-second settling countdown precedes stationary
acquisition; native elapsed time and actual sensor sample counts appear at the
top without scrolling. Extra settings, measurements and history are collapsed.
Actionable failures remain visible, and passing results offer the next step.
The 0.3.1 native refresh acknowledgement repair is retained. Install over the
existing app with the same signing key to retain captures and settings.

## One workspace

- **Capture:** phone/camera checks, optional server and paired-room selection,
  native landscape preview, mode-appropriate timing checks, record/stop,
  stationary IMU recording and background upload controls. Focus lock is a
  board-calibration control, not a normal-walk prerequisite.
- **Library:** retained phone artifacts with export/share/repackage/retry, and
  the existing server scan list, file imports, provider reconstruction, 3D
  inspection, additional videos, independent static-camera alignment and PCF
  review workflows. Model computation remains on the Noesis host.
- **Setup (optional calibration):** editable ChArUco definition, separate Camera/lens and
  Camera–IMU modes, native recording, imported capture selection, CPU processing,
  progress/cancellation, qualification results and downloadable evidence.

The packaged UI opens without server connectivity. Native capture and local
artifact actions remain available subject to their normal gates; paired room
recording, uploads, calibration computation and reconstruction need the server.
Reconstruction pairing is optional. Path refinement requires its paired room
camera and completed reference reconstruction. Calibration board takes
intentionally do not start a room-camera companion.

### Optional advanced calibration workflow

This section preserves the original calibration controls and qualification gates
for users who explicitly choose Setup. It is not required to record ordinary
reconstruction coverage or to make RGB reconstruction available.

If setup was interrupted, open **Check/reuse → Continue camera setup** (or the
other missing stage). Choose the existing full take, then follow **Upload
recording → Process recording**; already uploaded takes go directly to
processing. Refresh recordings before making a replacement. An ordinary room
walk is not a calibration take and does not need to be repeated to recover setup.

1. Place the board flat and **leave it fixed** throughout both board takes.
   Move the phone, not the board. Check the phone, choose its camera, focus on
   the board and **Lock focus** in
   the native preview. Run the same ten-second timing test if that locked mode
   is not yet qualified. Camera and Camera–IMU modes reuse the normal recorder,
   encoded resolution, exact Camera2 association and raw IMU streams.
2. In **Calibration**, check the board definition. Defaults match the retained
   target: 10×14 squares, 0.018 m square, 0.0132 m marker, `DICT_4X4_1000`, IDs
   300–369, legacy pattern off. Physical size/flatness confirmation is an
   explicit user attestation, not an inferred measurement.
3. Record the guided **60-second stationary sensor** take. Set the phone on a
   stable surface during the five-second countdown, then leave it untouched.
   The countdown is not part of the raw recording. This opens neither camera
   nor encoder. When saved, follow **Upload recording → Process recording** in
   the same view. There is no need to find the take in Library.
   The short model measures white noise and checks held-out stationarity; its
   conservative drift terms are labelled model priors, not measured long-term
   random walk. Three-hour stationary recording is not a room-walk prerequisite.
4. Record the guided **60-second Camera/lens** take. Move the phone so the fixed
   target visits varied image positions and angles within its sharp range; keep
   the locked focus unchanged. The preview stops at 60 seconds.
   If **Record** is disabled, the preview explains whether focus must be locked
   or the ten-second timing test must pass for that exact focus/mode. Let the
   test finish saving, then reopen the camera without changing focus. Extra test
   controls collapse once the required check has passed. Upload and process the
   full take using the same guided view. A completed job is not necessarily qualified:
   native geometry stability, blocked heldout reprojection and view coverage
   are reported separately.
5. If the user explicitly selects a qualified camera result for **future
   matching native imports**, that choice is snapshotted on each import. Actual
   device/lens/focus/crop must match before the existing full-five-coefficient
   rectification path applies it. Earlier scans are not rewritten; mismatches
   remain available for uncalibrated RGB reconstruction and display the reason.
6. For **Camera–IMU**, select that camera result and the short sensor result.
   Record the guided **90-second** take: an eight-second still start, rotations
   about all axes, translations, mixed movement, then a still finish. Keep the
   board fixed and visible. The timed guide stops automatically. Upload and process. The
   pinned CPU spline solver estimates the camera-to-IMU transform, signed timing
   offset and IMU scale/misalignment/bias corrections. Missing prerequisites or
   failed holdouts remain explicit; no identity transform, zero offset or
   guessed noise is treated as calibrated evidence.
7. **Check motion profile** runs the fixed OpenVINS consumer on the retained
   camera–IMU take and compares it with independently withheld visual-only board
   motion. No calibration or scale is refitted to pass. Only a passing profile
   exposes **Use for future matching short walks**. Failed jobs keep their inputs
   and explain which step needs attention; recording longer cannot repair a
   missing backend or unsupported physical-camera metadata.
8. Return to **Capture** for ordinary one-to-five-minute walks. Use the
   automatic-mode timing proof when the phone has only retained a locked-focus
   board-mode check; no new full calibration is needed. Keep the native 8K,
   device-availability and timing checks satisfied, then stay still for the
   first eight seconds, move slowly, revisit the starting area and finish still.
   Reconstruction pairing remains optional; path refinement uses its required
   paired room camera and completed reference reconstruction.

Calibration artifacts retain source hashes, units, optical binding, transform
direction, time-offset sign and rejected evidence. Calibration jobs alone do not
admit metric VIO. An explicitly selected, passing motion profile is rechecked
against each future native import and its actual consumer; raw imports and old
scans are never rewritten. It does not change live Noesis world/tracking calibration. Independent
static-room alignment remains a separate downstream authority. See the
[CPU solver contract](../native/roomwalk_calibration/README.md) for build,
configuration, validation policy and remaining admission gates.

### Android web boundary

Only the packaged app document on the configured HTTPS origin may issue bounded
native actions. There is no `addJavascriptInterface`, arbitrary-origin bridge,
cleartext mixed content or TLS bypass. File selection uses Android's document
picker; same-origin exports stream through verified HTTPS into a user-selected
document. Backend and static assets must be from compatible versions. Rebuild
the APK when the shared UI changes; restart only the RoomWalk service when
deploying changed backend routes, with operational approval.

## Install and use

When the verified parent handoff publishes the RoomWalk 0.4.0 build, use its
**Install RoomWalk Android companion** link to open the download page in the
phone's regular browser. Tap **Prepare APK in browser**, then **Save APK to
phone**. The page fetches the APK over the current HTTPS connection, checks its
exact release size and SHA-256, and offers the verified bytes for a local
browser save. This section documents the handoff procedure only; it does not
claim that the service or APK publication has completed. Keep the tab open until
the browser finishes saving; normal download and installation checks still
apply. Update the page's filename, byte count, and SHA-256 when a verified APK
is published.

1. Install the signed APK on the Android phone. Android may ask to allow the
   selected browser or file manager to install this app.
2. Open **RoomWalk**, allow camera access, and tap **Check phone**. Choose a
   supported rear camera. The app requires Android 13 or newer for recording;
   older supported installations can produce a capability report.
3. **Check connection** loads the available static room cameras. For path
   refinement, select the paired camera and completed reference reconstruction;
   reconstruction may leave the room camera unselected. Tap **Capture** to open
   the full-screen viewer. Setup and saved-capture screens use portrait; the
   viewer changes to landscape. Compose the shot using the live preview, then
   run the **10-second automatic-mode test** first when the phone needs that
   timing proof, starting and ending
   with the phone still.
   Inspect the saved timing result. If the companion has not found its 8K mode,
   tap **Upload phone report**. This sends the latest capability report to the
   configured RoomWalk server; a stock-camera 8K recording remains evidence
   that the phone can record that mode. A failed standard Camera2 size check
   alone does not establish a hardware limitation.
4. In the viewer, use **Record** and **Stop & save**. A successful retained
   short test qualifies the same mode for a full walk. Stopping returns to the
   portrait main screen while the saved capture is packaged. Keep the app in the
   foreground during recording; leaving it stops capture. The screen stays awake in the viewer.
5. **Upload** sends the saved ZIP to the configured RoomWalk HTTPS origin.
   A foreground-service notification tracks the transfer, which continues when
   the app is backgrounded or the screen is off. The main screen shows progress,
   throughput and estimated time remaining, with cancellation and explicit retry.
   **Transferred to RoomWalk** means the server has durably saved the archive;
   video and timing validation then continue on the server. Open RoomWalk to
   follow that processing or see validation errors.
   **Export** saves a copy through Android's file picker; **Share** grants
   read access to the selected artifact. **Saved captures** reopens retained
   recordings and capability reports. **Package again** retries packaging after
   freeing space or reopening a completed take. Upload never deletes the phone's copy.
6. Open RoomWalk to prepare the uploaded phone video and motion evidence and
   reconstruct with MapAnything + DA3. The exact static recording, tracking
   stream, and provenance remain attached to the scan as independent evidence.
   Phone and static acquisition clocks stay explicit; HTTP clock observations
   do not establish hardware synchronization or metric VIO admission.

The retained 0.1.7 paired-capture implementation is used when a room-camera
companion is selected, especially for path refinement; it is not the normal
reconstruction requirement. **Record** first starts the selected room-camera
recording and tracking observer, waits for both to report recording, then starts
native phone video and IMU. Every ten seconds a bounded worker renews the static
lease. Stopping or backgrounding finalizes both sides; native frame callbacks
perform no companion-service requests. If the room stream is lost, the phone
stops and retains its partial evidence. Starting uses one retained request ID,
so a lost response can be retried without creating a second session. The exact
phone capture, camera, static session, server origin, clock observations, and
failure state are retained in `companion_capture.json` and the capture manifest.
**Finalize paired capture** retries the saved session.

Version 0.1.10 routes the appliance origin `TauntonMainframe.local` directly to
the fixed LAN address `192.168.3.126`; it never performs DNS for that origin.
The logical hostname remains the HTTPS certificate/SNI and HTTP Host authority.
Other HTTPS origins use Android's public `DnsResolver` with an IPv4-only,
cancellable one-second limit; they do not wait for AAAA results or attempt IPv6
connections. The complete LAN health check has a three-second deadline with no
delayed retry and at most two concurrent checks with no queue. TCP setup is
capped at 750 ms.
TLS verifies the original hostname and trusted certificate chain during the
handshake, and HTTP Host and TLS SNI retain that hostname. The resolved address
does not replace the saved server origin. Host-specific TLS factories are cached
to preserve connection reuse; an untrusted certificate or mismatched hostname
still fails before sending HTTP.

Paired capture checks the connection before its initial clock probes, including
after a long preview. Clock and heartbeat budgets remain three and five seconds;
the server keeps idle connections for 30 seconds to cover the ten-second
heartbeat cadence. Upload TLS setup is capped at three seconds, then its
180-second body/response read timeout is restored. These connection bounds also
apply to background uploads.

### Background transfers and server import

Version 0.1.11 moves saved-archive uploads into a single Android `dataSync`
foreground service. A bounded partial CPU wake lock covers the active transfer;
the display can turn off. Notification permission is requested when first
uploading, but declining it does not prevent the transfer. Cancellation closes
the upload socket off the UI thread. Progress, terminal errors and the receipt
are saved locally. An interrupted process is reported as interrupted; retry is
explicit and uses the retained ZIP. Uploads do not start a camera or change
recording resolution, encoding, frame timing, IMU acquisition or focus settings.

The native service sends `Prefer: respond-async` on the existing
`POST /api/scans/sensor-bundle` route. The server can return HTTP 202 after
receiving, hashing and durably storing the complete archive and checking the
bounded manifest and requested capture/session/camera identities. Its
`upload_receipt` has schema `noesis.phone_capture.upload_receipt.v1`, status
`stored`, `size_bytes`, `sha256`, `capture_id` and the optional
`companion_session_id` / `companion_camera_id`. The phone verifies those values
against the actual streamed bytes and selected capture before marking transfer
complete. This receipt does not assert completed import, acquisition timing,
metric calibration or final companion association.

Full archive, video and timing validation continue in a bounded server worker;
RoomWalk reports `importing_capture` or `import_failed` until import succeeds.
Final companion association and frame preparation still require the existing
checks. The original archive is retained on the server, including when import
fails or is interrupted. Exact retries refer to the same scan; another archive
cannot replace a paired capture. Existing clients without the preference keep
their synchronous HTTP 201/200 contract.

Paired uploads send all three identities to RoomWalk and require the returned
scan to confirm that same association. The first successfully packaged paired
ZIP is retained unchanged: repeated Upload or Package again uses identical
bytes, allowing the service to return the original scan after a lost receipt.
If finalization remains unconfirmed, the app retains the phone files and shows
the room state requiring attention. Older unpaired captures remain uploadable.

The normal fresh-walk workflow requires no separate stationary noise recording
or printed calibration board. Optional **Advanced diagnostics** contains a
one-minute IMU-only recording, configurable from one to five minutes. This
collects diagnostic motion data; it does not assert calibrated noise, camera/IMU
extrinsics, or metric VIO. Camera focus controls remain available in the viewer.

Version 0.1.1 adds diagnostic reports covering the default and maximum-resolution
stream maps, discovered physical cameras, and advertised 8K video profiles.
Those additional modes are reported for diagnosis; the current recording path
continues to require its exact advertised 8K/30 configuration and timing checks.
Install updates over the existing app using the same signing key to retain
captures and settings.

Version 0.1.2 additionally queries the camera driver's support for exact 8K
encoder surfaces when the static size list omits 8K. It checks the first rear
camera that passes encoder, clock and FPS prerequisites. The finite query set
uses standard output use cases and vendor IDs actually advertised by that
camera, with and without a preview. The encoder is configured for its surface
but is not started, and no camera session or recording is opened. A positive
query is configuration-support evidence only; capture quality and acquisition
timing still require a recorded test. Results are included in the same explicit
phone-report upload. Vendor modes are not enabled for recording by this probe.

Version 0.1.3 permits the **10-second test** after a positive standard recording
query, even when the static size list omits 8K. Every attempt rechecks the camera,
encoder, preview and sensor prerequisites, then queries the actual recording
surfaces before starting the encoder or opening the camera. Capture reuses that
same configuration, including the 30 FPS session parameters and standard output
use cases. The recorder itself enforces the short-test duration. A positive
query alone does not establish native image detail or acquisition timing.

Version 0.1.4 enables full walks when a retained, completed ten-second test
matches the current OS build, camera, encoder, bitrate, preview and standard
output use cases. The test must have no capture failures or dropped metadata,
complete camera/encoder timestamp associations and IMU coverage, and 8–12 seconds
of frames averaging 29–31 FPS around the requested 30 FPS clock. This recognizes
startup latency and fractional camera clocks; it does not lower the recording
request. The app checks at most 64 recent capture-result files of at most 512 KiB
each. Every full walk revalidates that evidence and queries its actual surfaces
again. Updating over the existing installation preserves a qualifying test;
**Check phone** finds it. A missing or mismatched test leaves only the short test
available for a mode absent from the static size list.

The viewfinder uses the camera's actual preview dimensions and sensor orientation,
corrects TextureView scaling and display rotation (including 180-degree flips),
and fits the complete image. This display transform does not rotate, crop or
rescale the encoded video or change its coordinate frame. Longer-take reliability
and native resolving detail are separate from the recorded short-test timing check.

Version 0.1.5 replaces the embedded preview panel with **Capture → full-screen
live preview → Record → Stop & save**. The viewer has a compact control strip
beside the image; setup, server, upload and saved-capture menus remain on the main
screen. The complete camera image fits the remaining viewport without stretching
or cropping. Preview opens before recording and produces no video, IMU files or
capture session directory. A separate preview controller must confirm
`CameraDevice.onClosed` before the native recorder can open the selected camera.
Failed or timed-out closure prevents the recording attempt. The recorder still
owns the exact 8K encoder configuration, sensor streams and timestamp checks.
Closing the viewer or pressing Back while recording stops and saves the take;
backgrounding closes an idle preview or stops an active recording. Closing an
idle preview does not create a capture.

Historical 0.1.6/0.1.7 notes below preserve standalone IMU acquisition and the
optional board-mode focus control. IMU-only acquisition is optional under
**Advanced diagnostics** and records the same native accelerometer and gyroscope selected by video capture;
no camera, preview or encoder is opened. Keep the phone still and the app in the
foreground for this diagnostic. The screen stays awake and progress reports
elapsed time, sample counts and stored bytes. **Stop & save IMU**, Back, or
backgrounding retains an explicitly partial take. Reaching the selected duration
completes acquisition without claiming sensor-noise calibration. Existing longer
recordings remain available through Saved captures.

A finite 8,192-record worker queue preserves independent nanosecond sensor
timestamps, raw XYZ, available sensor bias estimates, accuracy and reception
timestamps. Bias estimates are not subtracted. Each stream is limited to four
million rows and the combined source bundle to 512 MiB. Queue overflow, clock
regression, missing/stalled sensor streams, row/byte limits, or low free storage
stop acquisition and retain its evidence. Sensor identity, actual sample counts,
cadence, stop reason, drop count and exact CSV hashes are recorded in
`imu_capture_manifest.json` (`noesis.phone_imu_calibration.v1`). A forced process
termination can leave the initial `recording` manifest and flushed raw CSV;
that state cannot be packaged as completed evidence.

The IMU ZIP contains exactly `imu_capture_manifest.json`, `accel.csv`, and
`gyro.csv`. **Upload** explicitly sends it over verified HTTPS to
`POST /api/phone-calibration/imu-bundle` with `application/zip` and `X-File-Name`.
A successful `stored` receipt must match the capture ID before the app reports
success. Partial recordings can be retained by the receiver; noise fitting and
review are separate work. **Export**, **Share**, **Saved captures**, and
**Package again** also support finalized IMU takes. Uploads preserve local files.

In the optional board-calibration mode, the viewer provides **Lock focus** and
**Unlock focus**. Locking requires a fresh, settled autofocus result with an actual focus distance and
manual-distance support. It requests AF OFF at that measured distance and saves
the setting only after a capture result confirms AF mode, stationary lens,
distance (within 0.01 diopters or 1%, whichever is larger), and active physical
camera. Logical cameras must report the active physical camera. The persistent
setting binds the logical and physical camera, OS fingerprint, and fixed 8K/30,
OIS-off/EIS-off recording mode. Preview reopen, exact session queries and capture
apply the same setting. Missing or mismatched confirmation prevents capture or
stops it with retained evidence; unavailable manual focus is not replaced with
a guessed distance. **Unlock focus** explicitly removes the setting and restores
automatic focus. A changed focus setting needs a matching short timing test to
qualify a mode that was not statically advertised. Focus metadata and recorded
confirmation counts are included in diagnostics and capture results. The
viewfinder's control strip scrolls on smaller screens.

Ordinary reconstruction and path-refinement modes retain automatic focus unless
the user separately selects the board-calibration protocol. Their saved timing
proof is checked for that automatic mode; a locked-focus board proof is not
silently reused. After a completed test, use **Upload** to send its video and raw timing bundle.
If capture fails, **Upload phone report** sends that attempt's diagnostic result
with the capability report. Original local files are retained in both cases.

The explicit **Upload phone report** action posts JSON to
`POST /api/phone-diagnostics`. The server accepts at most 512 KiB, retains the
exact report under `.phone-diagnostics/<sha256>.json` in its configured storage
root, deduplicates identical reports, and caps storage at 128 reports. Receipt
schema is `noesis.phone_capture.android_diagnostic_receipt.v1`. This endpoint
does not create a scan or start reconstruction. Reports remain available for
local export and sharing.

The installer can carry the appliance's public CA certificate and default HTTPS
origin. HTTPS retains hostname and certificate verification. No private TLS key
or APK signing secret is included. A stock build without the appliance CA uses
standard Android HTTPS trust.

## Capture and authority boundaries

- Records exact 7680×4320 output with a supported hardware encoder; acquisition
  uses an explicit camera selection. It requests disabled OIS/EIS and preserves
  actual camera-result metadata rather than assuming the settings took effect.
- Android 13's `OutputConfiguration.TIMESTAMP_BASE_SENSOR` selects the Camera2
  timestamp base. The camera must report `SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME`
  for the common native clock route. Encoder PTS are matched to
  `SENSOR_TIMESTAMP / 1000` with integer truncation, without a fitted offset.
- Every encoded frame must have one unique matching sensor timestamp before
  the app exports the acquisition timestamp table. Original encoder PTS,
  capture results, sensor readings, API/camera identity and failures are retained.
  RoomWalk checks those rows and the encoded video independently on import.
- Accelerometer and gyroscope streams retain independent native timestamps,
  device axes, units and sensor identity. Bounded queues keep persistence work
  off acquisition callbacks; overflow invalidates capture rather than hiding
  dropped metadata. The encoder and recording sizes are bounded.
- Video streams to phone storage. A take stops at 10 minutes or 6 GiB of video,
  or when the storage reserve is reached. Packaging creates another local copy;
  enough free storage for both video and ZIP is required. RoomWalk's server
  upload limit is 8 GiB. Keep additional server space for extracted data and
  reconstruction artifacts.
- Native timestamp correspondence is separate from measured camera/IMU time
  offset, lens calibration, camera-to-IMU extrinsics and IMU noise. Those values
  remain unknown. The app does not transfer a stock-camera 8K calibration or
  admit metric VIO. Missing timing association leaves raw RGB review available.

Capture files live in the app's external files directory under `captures/`:
`camera.mp4`, `accel.csv`, `gyro.csv`, `encoder_pts.csv`,
`camera_results.jsonl`, `capture_result.json`, `capabilities.json`,
`companion_capture.json`, and the
native `capture_manifest.json`. `timestamps.csv` is present only after exact
frame association succeeds. Archives use `noesis.phone_capture.v1` with the
`noesis.phone_capture.android.v1` evidence extension. Abrupt process termination
may leave raw files or an unfinished MP4; a completed archive is not asserted
until packaging succeeds. Export important captures before uninstalling.

## Build

This app uses Android framework APIs and has no third-party runtime dependencies.
The small build driver uses an installed JDK and Android SDK directly; no Gradle
or Android Studio installation is required.

Prerequisites: JDK 21, Android platform 36, and Android build tools 36.0.0.
Use a persistent private signing keystore with alias `roomwalk`; retain it for
future updates. Keep the keystore and password file outside the repository.

```bash
python3 tools/mapanything_phone_scan/android_companion/build.py \
  --sdk "$ANDROID_HOME" --java-home "$JAVA_HOME" \
  --keystore "$ROOMWALK_KEYSTORE" --password-file "$ROOMWALK_PASSWORD_FILE" \
  --server-url "$ROOMWALK_SERVER_URL" --ca-certificate "$ROOMWALK_PUBLIC_CA" \
  --output "$ROOMWALK_OUTPUT/roomwalk-companion-0.1.10.apk"
```

The driver compiles resources/Java, runs D8, aligns and signs the APK, verifies
its signature/alignment, and writes its SHA-256 sidecar. It deletes only its own
`android_companion/build/` directory before building. Local generated files and
signing artifacts are ignored. An Android emulator can exercise installation,
permissions, UI, and unsupported hardware behavior; actual 8K encoding and
camera/IMU timing must be checked on the target phone.

Primary API references:
[Camera2 output timestamp base](https://developer.android.com/reference/android/hardware/camera2/params/OutputConfiguration#TIMESTAMP_BASE_SENSOR),
[Camera timestamp source](https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#SENSOR_INFO_TIMESTAMP_SOURCE),
[sensor timestamps](https://developer.android.com/reference/android/hardware/SensorEvent#timestamp).
