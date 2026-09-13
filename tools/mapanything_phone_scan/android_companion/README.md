# RoomWalk Android companion

A native camera/IMU recorder with background LAN uploads for the RoomWalk reconstruction service.
It uses Camera2, a hardware MediaCodec surface encoder, MediaMuxer, and Android
sensors. It does not embed the browser recorder. Native 8K availability is
checked on the phone; this application does not assume that a stock-camera 8K
mode is available to Camera2.

## Install and use

Use RoomWalk’s **Install RoomWalk Android companion** link to open the
download page in the phone’s regular browser. Tap **Prepare APK in browser**,
then **Save APK to phone**. The page fetches the APK over the current HTTPS
connection, checks its exact release size and SHA-256, and offers the verified
bytes for a local browser save. This avoids a second network transfer by the
download manager. Keep the tab open until the browser finishes saving; normal
download and installation checks still apply. A direct download link remains
available. Update the page's filename, byte count, and SHA-256 when publishing
a different APK.

1. Install the signed APK on the Android phone. Android may ask to allow the
   selected browser or file manager to install this app.
2. Open **RoomWalk**, allow camera access, and tap **Check phone**. Choose a
   supported rear camera. The app requires Android 13 or newer for recording;
   older supported installations can produce a capability report.
3. **Check connection** loads the available static room cameras. Select the
   camera for this walk, then tap **Capture** to open the full-screen viewer.
   Setup and saved-capture screens use portrait; the viewer changes to landscape.
   Compose the shot using
   the live preview, then run the **10-second test** first, starting and ending
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

Version 0.1.7 makes paired capture the normal recording path. **Record** first
starts the selected room-camera recording and tracking observer, waits for both
to report recording, then starts native phone video and IMU. Every ten seconds a
bounded worker renews the static lease. Stopping or backgrounding finalizes both
sides; native frame callbacks perform no companion-service requests. If the
room stream is lost, the phone stops and retains its partial evidence. Starting
uses one retained request ID, so a lost response can be retried without creating
a second session. The exact phone capture, camera, static session, server origin,
clock observations, and failure state are retained in `companion_capture.json`
and the capture manifest. **Finalize paired capture** retries the saved session.

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

Version 0.1.6 introduced standalone IMU acquisition and persistent focus lock.
In 0.1.7, IMU-only acquisition is optional under **Advanced diagnostics** and
records the same native accelerometer and gyroscope selected by video capture;
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

The viewer additionally provides **Lock focus** and **Unlock focus**. Locking
requires a fresh, settled autofocus result with an actual focus distance and
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

After a completed test, use **Upload** to send its video and raw timing bundle.
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
