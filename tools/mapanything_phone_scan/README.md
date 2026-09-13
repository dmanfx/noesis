# Phone-Walk Room Reconstruction

This LAN-only tool records a phone walk, prepares an adaptive set of views, and
runs either MapAnything or Depth Anything 3 (DA3). It also contains an offline
consensus workflow that combines both model outputs into a cleaner shared room
reconstruction.

The intended long-term room workflow is:

1. Capture a slow, overlapping phone walk to build the room reconstruction.
2. Run the same prepared views through MapAnything and DA3.
3. Build the consistency-gated consensus reconstruction from both outputs.
4. Register that reconstruction to the room's calibrated static-camera data.
5. Use the saved room transform to place live detections and tracks into the
   reconstructed room coordinate system.

The phone walk supplies coverage and room geometry. Static-camera runs supply
the fixed Noesis alignment and validation reference. A static frame is not
silently mixed into phone-only inference or fusion.

This tool saves review candidates. It does not automatically publish them into
the live native DS9.1 depth, tracking, floorplan, or virtual-twin contracts.

The [canonical end-to-end Prior-Conditioned Fusion (PCF) operating and handoff
runbook](../../docs/PCF_Workflow.md) carries the process from capture through
the sealed evidence bundle, immutable Scene Prior, runtime binding, and
dashboard verification. The fusion decision, validation basis, and algorithm
detail are in
[`Phone_Walk_Fusion_Reconstruction.md`](../../docs/Phone_Walk_Fusion_Reconstruction.md).

## Find the relevant workflow

| Task | Section |
| --- | --- |
| Start the browser/service and inspect configuration | [Service](#start-the-service), [configuration](#configuration) |
| Import measured phone calibration | [Phone-camera calibration](#measured-phone-camera-calibration) |
| Capture/upload a walk and pair static evidence | [Capture workflow](#capture-and-single-provider-workflow), [paired static capture](#paired-static-camera-capture) |
| Browser motion recording versus native metric-VIO prerequisites | [Browser camera/IMU](#browser-camera--imu-capture), [synchronized capture](#synchronized-camera--imu-capture), [native timing preflight](#native-timing-preflight-before-a-new-room-walk) |
| Extend a retained walk or inspect provider inputs | [Add video](#adding-another-video-to-an-existing-walk), [provider semantics](#provider-semantics) |
| Fuse and diagnose phone-only geometry | [Consensus fusion](#consensus-fusion-workflow), [diagnostic views](#heatmap-style-diagnostic-views) |
| Fit against independent static-camera evidence | [Alignment findings](#static-camera-alignment-findings), [reusable fitting process](../../docs/room_reconstruction_fitting.md) |
| Generate a PCF candidate or perform an authorized handoff | [Conditioned workflow](#validated-landscape-da3-conditioned-mapanything-workflow), [PCF phase boundaries](../../docs/PCF_Workflow.md#phase-boundaries-and-stopping-points) |
| Direct integration and focused checks | [Tracking contract](#detection-and-tracking-integration-contract), [artifacts](#important-artifacts), [validation commands](#validation-commands) |

Historical validation sections name their tested captures. They do not establish
quality, calibration or runtime admission for a new walk.

## Validated reference capture

The consensus result reviewed on 2026-08-09 used this scan:

```text
data/mapanything_phone_scans/20260801-112036-2d9225dd
```

It is a **portrait** capture:

```text
48 prepared views
1080 x 1920 pixels per prepared frame
```

A second stored capture is landscape and was validated on 2026-08-09 with the
same provider, fusion, static-world alignment, and diagnostic rules:

```text
data/mapanything_phone_scans/20260802-162254-8bcc7dd7
48 prepared views
1920 x 1080 pixels per prepared frame
```

Within a walk, consistent orientation, slow motion, repeated overlap, and
deliberate revisits matter more than rotating the phone midway through the
recording.

## Start the service

The deployed user service is:

```bash
systemctl --user status noesis-phone-scan.service
```

It is enabled at login and listens on port 8788. To run the launcher directly
from the repository root:

```bash
./tools/mapanything_phone_scan/run.sh
```

On first start, the launcher prepares an isolated Python environment under
`data/mapanything_phone_scan_runtime/`. It reuses the machine's CUDA-enabled
PyTorch installation and pins the official model dependencies used by this
tool.

The launcher keeps the existing HTTP listener on port 8788 for known local
consumers and adds a secure listener on port 8789. It prints both URLs; the
secure Room Walk URL is:

```text
https://TauntonMainframe.local:8789
```

`GET /api/health` reports this value as `secure_capture_url` and reports the
CA bootstrap route as `ca_certificate_url` so a client can discover the exact
current endpoints without guessing a hostname or port.

The appliance certificate currently has the DNS SAN
`TauntonMainframe.local` and does not authorize `192.168.3.126` as a direct
HTTPS hostname. The native RoomWalk companion routes this hostname directly to
`192.168.3.126` while retaining the hostname for certificate/SNI verification;
browser access should continue using the hostname above. The HTTPS listener reuses the Menon appliance certificate and key;
override their locations with `NOESIS_PHONE_SCAN_TLS_CERT_FILE` and
`NOESIS_PHONE_SCAN_TLS_KEY_FILE`, and its public CA certificate defaults to
the same Menon TLS directory as `menon-local-ca-cert.pem` (override with
`NOESIS_PHONE_SCAN_TLS_CA_CERT_FILE`). Change the secure port with
`NOESIS_PHONE_SCAN_HTTPS_PORT`. If the certificate or key is missing or does
not match, the launcher fails closed instead of starting an untrusted HTTPS
endpoint.

Android Chrome must trust the existing Menon local CA before it will treat this
as a trusted secure origin. If that CA is not already installed on the phone,
open the bounded public download route from the same LAN over the retained HTTP
listener, replacing `<LAN-IP>` with the appliance IPv4 address:

```text
http://<LAN-IP>:8788/api/browser-capture/ca-certificate
```

The route serves only the configured CA certificate, never the private key or
an arbitrary file; the download is named `Noesis-Room-Walk-CA.crt`. Compare
its SHA-256 fingerprint with the value printed by the launcher on the trusted
appliance console before installing it. Then use Android's **Install a
certificate** flow and select it as a CA certificate. The exact Settings label
varies by Android release. Never transfer or install
`menon-local-ca-key.pem`. After trust is installed, reopen the hostname URL and
confirm Chrome shows a valid connection; a certificate bypass or warning page
does not satisfy this requirement.

The HTTP URL remains available for existing consumers and is not a substitute
for the trusted browser origin:

```text
http://192.168.x.x:8788
```

Connect the phone to the same trusted LAN, open the HTTPS URL in Chrome, and
permit the local-network, camera, and motion/sensor prompts. The service has
no user authentication; do not forward either port through the router.

Walks with a paired static recording reconstruct that recording as their
alignment target. The recorded camera is selected automatically. Static depth
is inferred separately from the phone views, using the saved camera
rectification and calibration-frame binding. A failed or incomplete paired
reference stops alignment; it cannot silently select older room geometry.
The reference artifacts and exact captured world revision remain attached to
the walk, and the original recordings are preserved.

For walks without a paired recording, alignment targets come from one validated
Noesis scene release. The default for this installation is
`data/virtual_twin/releases/home_rgbmesh_20260623T2158_v1.json`; another home
must set `NOESIS_PHONE_SCAN_ALIGNMENT_RELEASE` to its own validated release
manifest before starting the service. The browser lists only cameras whose
release revision, backend-world metadata, RGB keyframe, and calibration row are
present.

## Measured phone-camera calibration

Import the camera handoff once, preserving its original archive and all declared
source files:

```bash
python3 -m tools.mapanything_phone_scan.phone_calibration \
  --bundle "$PHONE_CALIBRATION_BUNDLE" \
  --output-root "$PHONE_CALIBRATION_ROOT"
```

The command returns `profile.json`. It verifies every declared file size and
SHA-256, checks JSON/NPZ agreement, and retains the five OpenCV pinhole
coefficients `[k1, k2, p1, p2, k3]` without truncation. Configure the returned
profile with `NOESIS_PHONE_SCAN_CAMERA_CALIBRATION`. Its recorder association
defaults to `NOESIS_PHONE_SCAN_CALIBRATED_CAPTURE_MODE=unbound`, which registers
the evidence without applying it to images. Once the recording path is known
to match, select `uploaded_video`, `browser`, or `native_sensor_bundle` for that
setting and reload the phone service while it is idle.

The source video must have the calibration's native encoded dimensions and
orientation, and must use the same physical lens, zoom, focus, crop, and
stabilization settings. A matching aspect ratio or phone model is insufficient.
This implementation permits the preparation stage's own pure resize; it does
not assume that a different native recording resolution is the same camera
projection. A mode, resolution, or rotation mismatch leaves the existing RGB
path available and records why the profile was not applied. The capture page,
prepared-view card, and `/api/health` expose the profile/application status.
Existing saved preparations are not modified.

For a matched new recording, selected images are rectified once before final
prepared hashes, thumbnails, and contact sheets are written. Every view carries
its exact rectified K, original D5, source and prepared hashes, original image
identity, resize transform, and profile hash. The full imported profile is
retained with the preparation. Both providers use the same rectified images.
Provider output masks exclude rays outside the original distorted image.

MapAnything uses its installed multimodal preprocessor so image crops/resizes
and supplied K move together, and its network consumes the calibrated rays.
DA3-BASE does not condition its pose/depth network on K alone. Its installed
input processor transforms measured K onto the exact output grid; the phone
adapter uses that K for DA3Metric focal conversion and 3D backprojection while
retaining the network-estimated K separately. Reports explicitly distinguish
measured geometry from network conditioning. Windowed outputs preserve the
calibration lineage and diagnostics. Consensus requires matching per-view
calibration and preparation lineage from both providers, verifies the retained
profile hash, and carries the profile's evidence and capture-binding limits
into its manifests and raw views. Its common rays retain the full-D5 border
validity check. Fusion does not admit a sensor-projection hypothesis as measured
capture calibration. Existing overlap, fusion, and static-fit
checks remain in force; no dummy camera poses are introduced.

The 2026-09-07 `roomwalk_phone_video` handoff describes raw **7680×4320** frames.
Its source reports 0.813029 px mean training error and 1.242761 px mean held-out
error over 14 pose groups. The owner subsequently confirmed a Fold 8 Ultra,
the phone's default camera app, **8K at 1×**, and the default rear camera for
that mode. The installed profile is therefore bound to `uploaded_video`.
Record a landscape walk with those same settings and upload the original
video. Keep focus and stabilization settings consistent with the calibration
recording; their explicit values were not reported.

The complete user-provided camera listing and recording-mode confirmation are
retained as `recording_mode_binding.json` beside the imported profile, leaving
the original bundle and measured K/D unchanged. Camera 0 and labelled physical
Camera 5 share the same reported sensor calibration and 6.25 mm focal length;
their association with the main recording camera is an inference, not a
verified physical-camera API binding. These sensor metadata values do not
replace the calibration measured on the encoded 8K video. The sidecar records
the evidence; the service setting and native-image checks enforce the runtime
binding, with the operator responsible for matching lens and capture settings.

The retained 4K uploads and 1080×1920 browser walk do not qualify for automatic
reuse. These camera intrinsics supply neither camera-to-IMU calibration/timing
nor a room/world pose; metric-VIO admission and static-camera/world authority
remain separate.

## Capture and single-provider workflow

1. Select **New Walk** to clear the current review and enter fresh-capture mode.
   That mode survives a browser refresh; it does not delete or overwrite an
   earlier walk.
2. Give the walk a room name, then record it or choose an existing video. The
   upload is automatically saved under
   `data/mapanything_phone_scans/<scan-id>/phone_walk.mp4`.
   Saved walks appear in the **Walks** list. Press and hold a walk to rename it;
   the new name is saved immediately without renaming or moving its asset
   directory.
3. The server samples existing encoded frames near four views per second;
   missing recording intervals are not filled with synthetic repeated images.
   Before selection, it collapses adjacent runs of identical or strongly
   supported near-identical images to one representative. Comparisons stay
   anchored to the start of each run so slow camera movement remains visible.
   It keeps frames that add viewpoint information, enforces temporal coverage, and
   inserts bridge views when adjacent selected frames lack reliable visual
   overlap. The default 256 selected-view limit is an emergency ceiling, not a
   requested frame count; preparation reports if it ever constrains a walk.
4. Selection combines relative sharpness, exposure, feature coverage, visual
   motion, and overlap. The manifest records every selected timestamp, reason,
   quality score, adjacent-view connectivity measurement, and any warnings.
   It also records dropped repeat counts, original candidate indices, and repeat
   time spans. A stationary or frozen section may leave a longer timestamp gap;
   coverage and bridge repair cannot reinsert those repeats. Visual connectivity
   and reconstruction alignment checks still apply across the gap. Browser
   `encoded_pts` values describe the saved video, not verified camera acquisition
   time. Preview freezes alone do not establish that the saved video froze.
5. Once `prepared_frames_manifest.json` is durable, choose **MapAnything** or
   **DA3** and select **Run reconstruction**.
6. The selected provider reconstructs every prepared frame. MapAnything uses
   one joint pass through 80 views. Larger adaptive sets are divided into
   80-view windows with 24 exact duplicate views between
   neighboring windows. Duplicate camera poses initialize a proper Sim(3),
   pixel-corresponding duplicate 3D surfaces robustly refine scale and
   translation, and both camera and dense-surface gates must pass before a
   window enters the saved reconstruction. DA3 uses the same fail-closed
   mechanism with 48-view windows and 16 exact duplicate views. Unique views
   from every accepted
   window are retained; review points are confidence weighted and averaged in
   3.5 cm voxels.
7. The tool saves a review GLB, camera trajectory, RGB, depth, confidence or
   validity data, masks, poses, intrinsics, metric scale, and raw NPZ arrays.
8. Select **Align to selected camera** for a paired walk; its recorded camera is
   selected automatically. The tool first builds or
   verifies its saved static reference, then aligns the phone reconstruction.
   Static-reference progress and artifact links appear in the alignment card.
   Without a paired recording, choose the static camera physically installed
   in the scanned room and select **Align to selected camera**.
   The target is saved with the walk before
   alignment starts. The tool estimates the phone floor, preserves gravity and
   metric scale, and registers room structure to that camera's validated room
   reconstruction. For an ordinary phone-only walk, shared RGB landmarks across
   multiple phone views and the fixed-camera keyframe choose the room pose by
   depth-backed PnP consensus; vertical geometry then performs only a bounded
   refinement. Refinement and candidate scoring use the same visible-source
   domain as validation. Source-fit metrics are evaluated over surfaces the single static
   camera could actually observe, while points hidden behind its measured depth
   remain recorded as global diagnostics instead of counting as alignment
   failures. Foreground disagreements are still scored. The tool also verifies
   that the selected camera faces its own target cloud and never rotates or
   rewrites the authoritative cloud or global Noesis calibration.
   The [reusable fitting procedure](../../docs/room_reconstruction_fitting.md)
   records controlled experiments, reference checks, and diagnostic replay commands.
9. A weak or ambiguous registration fails its quality gate. After a passed DA3
   alignment, select **Generate PCF review** to run the canonical
   `da3_pose_sparse_depth` conditioned MapAnything stage, DA3-carried
   consistency fusion, and static-world evaluation over the same prepared
   views. The app auto-saves the PCF GLB, raw evidence, collaboration image,
   diagnostic layers, 2.5 cm point layers, evaluation metrics, and run log.
   On GPU-constrained installations,
   `NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=1` makes this button explicitly pause
   the native Noesis/Menon appliance and restore it when the PCF job exits. The
   runtime lease and any restoration error are saved with the job.
10. A completed browser PCF remains a review candidate. The button never seals
    a room-scan bundle, builds or binds a Scene Prior, or changes live Noesis.

## Paired static-camera capture

The native RoomWalk companion is the selected phone route. Version 0.1.11 pairs
its Camera2 video and both IMU streams with the chosen static camera. In the app,
select the room camera, open **Capture**, and use **Record**. Static encoded
video and canonical tracking must be ready before the phone recording starts.
**Stop and save** closes both recordings; **Upload** sends the
phone bundle with the exact saved static-session reference. Interrupted uploads
can retry the same saved archive. Keep the app in the foreground during capture.
Setup and saved-capture screens use portrait; Capture opens in landscape and
returns to portrait when closed. Upload uses a foreground-service notification
and continues with the phone screen off. Its transfer receipt confirms durable
server storage and the archive hash; full video/timing validation continues in
the server's bounded background import worker. Open RoomWalk to follow that
processing. Recordings and exact paired identities remain retained on retry.
No stationary calibration session or target-board recording is required to make
a room walk. The optional short sensor check is a diagnostic, not a prerequisite.

Version 0.1.10 routes `TauntonMainframe.local` directly to the fixed
`192.168.3.126` LAN address, skipping DNS for the appliance origin. The
certificate, TLS/SNI and HTTP Host still use the saved RoomWalk hostname. Other
HTTPS origins retain IPv4-only DNS lookup. The initial health check keeps its
three-second deadline, and subsequent camera-list and capture requests reuse the
verified connection. Install the update over the existing app to retain saved
captures.

The browser recorder also supports paired capture. Existing-video uploads remain
available separately. Its sequence is:

1. Select **Record phone walk**, allow camera and motion access, and choose the
   physical **Static room camera** installed in the room. The browser lists
   only cameras resolved through the active native DS9.1 source configuration.
2. Select **Start recording**. The phone recorder starts only after the
   selected camera's original encoded video is arriving and its canonical
   tracking observer is ready. A camera that is merely configured, or a
   tracking status that is stale or partial, does not pass the start gate.
3. Walk with the phone while the page sends bounded clock probes and heartbeats.
   **Mark event** is optional and records a user label plus the server receipt
   clock for a visible moment such as a doorway or turn.
4. Select **Stop and save** when the walk is complete. The page finalizes both sides of
   the session and exposes saved links for the static Matroska video, packet
   timing, tracking/world records, calibration/runtime provenance, clock
   exchanges, and session metadata. Keep the tab open until saving finishes.
   If upload fails, retry or download the phone bundle from that tab. Closing
   the tab can lose its unsaved phone data; the server retains static evidence
   and expires the abandoned recording lease.

There is one active paired session at a time. Each session is limited to 900
seconds and the server lease is renewed for at most 45 seconds per heartbeat;
an abandoned tab therefore cannot hold the lane indefinitely. The static
recording is evidence for later alignment, calibration, or validation and is
kept separate from phone-only view selection and provider fusion. It does not
change live tracking or establish a common acquisition clock. Browser callback
times remain estimated/unverified, and a phone camera trajectory must not be
treated as a tracked person's body or foot trajectory without separate identity,
offset, scale, and timing evidence. See
[`WO-2E-static-paired-capture.md`](../../plans/reconstruction_work_orders/WO-2E-static-paired-capture.md)
for the capture contract and failure behavior.

For reconstruction review, prefer original static observations just before or
after the walker leaves the field of view; simultaneous occupancy is not a
requirement. The offline `prepare_paired_static_reference` API also accepts an
explicit `selection_manifest`: six sorted decoded frame indices with exact PTS,
the source-video hash and camera identity, and either a visually reviewed clear
assertion or a hashed full-resolution exclusion mask in
`post_dewarper_streammux_pixels`. A missing detection is not a clear-frame
assertion. The selected keyframe is one actual observation, not an invented
background. Exclusions are applied before static depth fusion and to RGB
alignment features, and are saved with the reference. They withhold evidence;
they do not fill occluded room structure. Ordinary calls without this option
retain the existing six-fraction sampling policy.

## Browser camera + IMU capture

The default **New Walk** flow records camera and browser motion-sensor samples
in the page and uploads the raw observation bundle over the HTTPS origin. This
keeps the camera frames and available sensor samples together for review and
later processing. Browser callback arrival times are not acquisition times;
they never admit metric VIO, establish a camera-to-IMU transform, or create a
Noesis world origin. If the browser does not expose the required sensor API,
the RGB video remains usable as an uncalibrated reconstruction input.

New in-page recordings request exact rear-camera 7680×4320 or its portrait
orientation with `resizeMode: none`. Unsupported modes stop setup without
falling back to HD/4K. Readiness requires a progressing preview and at least two
fresh readings from each sensor. Generic Sensor clock reversals, stale samples,
and timestamps ahead of callback receipt fail capture while retaining raw
observations. Camera settings are checked again at start and during recording;
the importer rejects strict-8K bundles whose encoded dimensions differ from
the negotiated settings. Legacy bundles remain importable as raw evidence.

The recorder requests 128 Mbit/s, retaining the 512 MiB browser byte limit;
this permits roughly 30 seconds, with actual capacity depending on encoding
and manifest size. The page displays this limit before recording. These checks
do not prove encoder throughput, physical lens/crop equivalence, acquisition
timing, or calibration transfer. Preview rows identify preview/recording phase
and remain separate from encoded-frame identity. RoomWalk explicitly reports
synchronization as unverified. For synchronized camera/IMU acquisition, use
the native bundle route below; native 8K availability still needs device testing.

Generic Sensor `timestamp` values retain an unverified clock domain; callback
receipt times separately identify the browser performance clock. The importer
does not infer a common origin from the API name. Even when an older bundle
declares a common clock, sensor times later than receipt make callback-lag
statistics unavailable. Raw values are preserved. An affine fit between these
clocks is an arrival-time diagnostic, not a calibrated camera/IMU offset.

## Synchronized camera + IMU capture

The native [RoomWalk Android companion](android_companion/README.md) records
Camera2 video and native sensor streams, exports acquisition timing evidence,
and uploads directly to this service. Its 8K mode must pass phone capability
and recording checks; it does not assume stock-camera calibration applies.


The native companion is the selected capture route for this phone. It retains
separate accelerometer and gyroscope acquisition timestamps and the original
Camera2/encoder association rows. Native acquisition does not supply camera
intrinsics, camera-to-IMU extrinsics, measured timing offset, or IMU noise by
itself. The earlier OpenCamera-Sensors bundle format remains importable as
external evidence; installing another recorder is unnecessary for the native
companion workflow.

Select **Import camera + IMU bundle** in the browser to upload the archive.
The server keeps the raw streams, probes encoded dimensions, preserves exact
integer-nanosecond frame timestamps, checks archive paths and bounded sizes,
and shows calibration and metric-admission status. Missing calibration keeps
the RGB workflow available while metric VIO remains blocked. Prepared views
retain exact source frame times and IDs. When the report admits metric VIO,
**Run OpenVINS** materializes the dense encoded camera stream and separate IMU
streams, applies the measured time offset, and binds native camera-origin poses
back to exact prepared frames. The result contract is
`noesis.phone_capture.vio_result.v1`; its camera axes are OpenCV
`x_right_y_down_z_forward`, its estimator world is z-up/gravity-up, and its
covariance convention is declared in the result. OpenVINS is a conventional
fixed estimator; VI3 remains deferred.

Native captures also use IMU motion during ordinary reconstruction preparation,
without requiring metric VIO. For each exact Camera2 frame, the processor retains
angular-speed and acceleration magnitudes in a 50 ms timing neighborhood around
the exposure. The magnitude is invariant to sensor orientation; acceleration
still includes gravity and sensor bias. A bounded soft preference for lower
rotation during exposure supplements visual quality and overlap repair. Missing
clock evidence, a sample gap, or missing exposure leaves the visual score
unchanged and records why motion was unavailable. This is a view-selection
heuristic, not an IMU position estimate or measured camera/IMU timing correction.
`prepared_frames_manifest.json` retains the per-frame values, original score,
applied penalty and hashes of the raw evidence. MapAnything and DA3 consume the
resulting same prepared phone views. Static frames remain independent alignment
and validation evidence. The review page links the phone video, IMU samples,
frame timestamps and paired static artifacts together.

For an explicit offline motion-quality experiment, create a separate prepared
revision without changing the original scan or ordinary soft-selection default:

```bash
python3 -m tools.mapanything_phone_scan.prepare_motion_revision \
  --scan-dir "$PHONE_SCAN_DIR" --output-dir "$EXPERIMENT_ROOT/motion_revision" \
  --maximum-exposure-rotation-deg 1.0
```

The threshold is an experimental exposure-blur heuristic, not calibrated
orientation. The tool rechecks native motion evidence and RGB hashes, copies
only retained prepared images, preserves parent frame IDs and every rejected
row, and invalidates adjacency measurements across a newly removed gap. Both
providers must use the exact resulting view identities. Reconstruction and
alignment gates remain unchanged. On the September 12 Living Room capture,
excluding four severe-motion views retained 127 views and cleared the original
last-window registration failure; a reserved late-walk gyro comparison also
improved, but not every reserved interval improved. Do not generalize this
single-capture result to metric VIO or surveyed room accuracy.

The explicit browser `input type=file` video upload remains the RGB-only flow
for an existing video or a browser capture that cannot collect sensor samples;
the in-page camera + IMU flow above is the default raw-observation route.
Browser callback arrival timestamps are not acquisition times and are never
used to admit metric VIO. Native build, configuration, exact
run commands, and the public EuRoC monocular execution/evaluation evidence are
recorded in [`native/README.md`](native/README.md) and
[`plans/reconstruction_work_orders/WO-2.md`](../../plans/reconstruction_work_orders/WO-2.md).

### Native calibration recordings

Normal room walks always retain both IMU streams. Companion 0.1.7 places its
separate sensor-only recorder under advanced diagnostics, defaults to one minute,
and caps it at five minutes. It opens neither camera nor encoder. Completion,
early stop, backgrounding and resource limits retain the acquired samples and
stop reason. These short recordings can check sensor behavior; they do not
measure long-term bias random walk. The former three-hour default is withdrawn
from the room-walk workflow.

The archive contains exactly `imu_capture_manifest.json`, `accel.csv`, and
`gyro.csv`. It retains native integer timestamps, SI units, Android device axes,
sensor identity, raw values and separate vendor bias estimates. It does not
subtract those estimates, interpolate streams, or create video-frame times.
`POST /api/phone-calibration/imu-bundle` accepts a bounded ZIP and retains it
under `<scan-storage>/.imu-calibration/<capture-id>/`, independently of scans.
The receiver verifies each stream's counts, hashes, sizes, timestamp counters,
and declared units before saving the original ZIP, CSVs, manifest and receipt.
A partial recording remains evidence. Uploading never marks IMU noise calibrated.

If an offline camera/IMU calibration recording is explicitly needed, select the same lens, 8K mode,
zoom and orientation as room walks. In the live viewer, focus on the calibration
target, then use **Lock focus** to retain the actual reported lens setting.
The explicit lock is camera-bound and persists across preview and recording;
keep it for subsequent room walks using that calibration. Unsupported manual
focus or an unavailable lens reading fails the lock instead of guessing a
setting. **Unlock focus** returns to automatic focus and ends that matching
calibration configuration. Keep stabilization off and verify actual capture
results, not just the requested setting.

The [calibration target and solver notes](calibration_targets/README.md) describe
an optional offline measurement path. In that target-based path, camera-to-IMU
rotation/translation and time offset use target-visible movement
about all three axes and translation. Retain separate withheld motion for
validation. A camera's rolling-shutter readout remains recorded evidence; a
shared clock does not establish zero offset. Do not transfer stock-camera
intrinsics solely from matching dimensions or turn on metric VIO from a
plausible trajectory. The imported Android report recomputes full admission
only after exact camera/encoder timing, actual OIS/EIS settings, required
calibration, geometry and IMU coverage have all been checked.

The OpenVINS input adapter handles a five-coefficient Brown-Conrady camera by
rectifying every dense encoded image with all five supplied coefficients,
preserving the supplied K and encoded dimensions. It carries a hashed mask for
invalid border pixels into the native bridge and requires confirmation that
those pixels were excluded. It never truncates the fifth coefficient. Models
outside the explicit supported pinhole/fisheye contracts remain rejected.

## Adding another video to an existing walk

After the first reconstruction completes, select **Add Video** to record another
pass or **Use saved video** to upload one already on the phone. The upload and
its prepared frames are immediately auto-saved below
`<scan-id>/supplements/<addition-id>/`; the original video, prepared views, and
reconstruction outputs are never overwritten.

An addition uses the provider selected for the original reconstruction. Before
using the GPU, the service searches the current reconstruction's retained RGB
views for strong overlap with the new video. It copies up to eight exact prior
views into the new inference batch as bridge views and reserves the remaining
view budget for the new pass. Static-camera images are not inserted into this
phone-to-phone bridge batch.

Publishing an added pass requires both checks below:

1. The duplicate bridge camera poses must agree on one stable metric Sim(3)
   transform from the new joint reconstruction into the original phone frame.
2. Independently matched RGB landmarks from the new views must solve against
   prior reconstructed 3D points with a camera pose consistent with that
   transform.

If either gate fails, the base reconstruction remains current and all uploaded
video and prepared frames remain available for review or retry. If both pass,
the service creates an immutable revision containing the joint inference,
append-to-base transform, source comparison, camera path, confidence-weighted
4 cm surfels, raw provenance counts, report, and manifest. New-only voxels need
support from at least two views; overlapping evidence is confidence weighted,
with the original reconstruction retaining greater authority. Later additions
may bridge through retained frames from any accepted earlier pass.

If the base walk is already registered to Noesis, the saved phone-to-Noesis
transform is composed into a Noesis-world derivative of the merged revision.
If Noesis registration happens later, that derivative is generated when the
alignment completes. Deleting an addition is allowed only from the newest end
of a dependent revision chain.

The browser runs one base provider per scan. A completed, quality-gated DA3
alignment exposes the optional PCF action, which runs conditioned MapAnything
and the selected dual-provider fusion without replacing the base DA3 output.
PCF currently uses the original prepared walk only. If an added-video revision
is active, the app refuses PCF rather than silently omitting those added views.

## Provider semantics

### Prepared image orientation

Inspect the prepared contact sheet before diagnosing cross-window registration
failures. Encoded dimensions alone do not establish whether the scene is
upright. The Living Room capture `20260906-050646-107f244c` required a reviewed,
lossless 90-degree counterclockwise correction because its saved pixels were
sideways without display-rotation metadata. Its prepared manifest retains the
original frame hashes/identities and pixel transforms; the raw capture is
unchanged. All 256 views subsequently passed the existing five-window run.
This correction is specific to that saved preparation, not automatic handling
for future uploads. Repreparing that original video requires preserving the
correction. It does not establish IMU calibration or Noesis-world alignment.

### MapAnything

- Official Apache-licensed `facebook/map-anything-apache` model.
- Jointly predicts multi-view poses, depth, intrinsics, metric scaling, and
  learned confidence.
- The installed RTX 3060 capacity check passes 80 joint views at 10.56 GB peak
  allocated memory; 96 views OOM with DS9.1 stopped. The 80-view default is a
  measured per-window GPU limit, not a capture limit. Use
  `benchmark_mapanything_capacity.py` before increasing it on another GPU.
- Native DS9.1 must be paused while this offline reconstruction owns the GPU;
  the deployed browser action exposes and records its configured automatic
  pause/restore lease rather than competing for memory or silently changing
  runtime authority.

### DA3

- Official Apache-licensed `depth-anything/DA3-BASE` supplies joint any-view
  geometry, poses, intrinsics, and learned multiview confidence.
- DA3Metric-Large supplies metric depth through the validated FP16 TensorRT
  engine for the RTX 3060.
- That DA3Metric engine is coupled to the installed TensorRT version and must
  be rebuilt after a TensorRT upgrade. MapAnything and consensus fusion do not
  use serialized TensorRT plans in this utility.
- The DA3Metric non-sky output is a validity mask, not a confidence map.
- The integration uses DA3 Base confidence for weighting and the DA3Metric
  non-sky mask for validity. It does not invent confidence values.
- Adaptive sequences above 48 views use 16-view-overlap DA3 windows. Exact
  duplicate RGB-D views gate each fixed-scale window registration, and every
  unique accepted view is retained for the conditioned MapAnything and PCF
  stages.

## Consensus fusion workflow

The authoritative implementation is
`tools/mapanything_phone_scan/build_consensus_fusion.py`.

It does not average raw point clouds. The fusion performs every stage below:

1. **Shared pose graph**
   - Aligns the DA3 trajectory to MapAnything with a Sim(3) estimate.
   - Uses relative-pose observations from both models at one-, two-, and
     four-frame spans.
   - Admits only geometrically verified revisit constraints; ending a walk
     does not create a loop edge or force endpoint position/orientation.
   - An explicitly passed retained-consensus trajectory report can replace the
     initial graph result. Fusion then recomputes consistency, depth selection,
     evidence weights, world points, and distinct-view surfel support.
2. **Common camera rays**
   - Converts the 518x294 MapAnything depth and 504x280 DA3 depth to a shared
     504x280 angular grid derived from both providers' geometry intrinsics,
     including supplied calibration when present.
   - This is geometric reprojection, not image-sized depth averaging.
   - Raw RGB keeps the MapAnything reference image projected onto those rays.
     Averaging images warped through different inferred intrinsics would
     duplicate edges and weaken the static-camera visual anchor. The manifest
     records this RGB source; both providers still contribute depth evidence.
3. **Separate confidence ranking**
   - Converts each provider's raw confidence distribution to its own empirical
     percentile rank score; these are uncalibrated scores, not probabilities.
   - Combines that rank with temporal multiview reprojection consistency
     and a depth-boundary penalty.
4. **Agreement fusion**
   - When depths agree within approximately 10-15 cm, selects a weighted median
     surface so an artificial midpoint wall is not created.
5. **Moderate disagreement selection**
   - Chooses the provider with stronger multiview reprojection consistency.
6. **Large-disagreement rejection and hole filling**
   - Rejects major conflicts as uncertain.
   - Accepts a single-provider hole fill only when confidence, multiview
     consistency, and neighboring support clear their thresholds.
7. **Surfel fusion**
   - Integrates accepted depths into confidence-weighted 4 cm surfels.
   - Requires support from at least two distinct views for accepted surfels;
     multiple pixels from one view do not satisfy that requirement.
   - Preserves the provider-selection and disagreement arrays in every raw
     fused view for diagnostics.

The 4 cm surfel GLB is the conservative review artifact. It is not the
highest-density representation available for downstream point models; the
accepted full-resolution evidence remains in the consensus `raw/` directory.

### Reusable command

Once both raw output roots exist for matching prepared views:

```bash
python3 tools/mapanything_phone_scan/build_consensus_fusion.py \
  data/mapanything_phone_scans/<scan-id> \
  --mapanything-raw data/mapanything_phone_scans/<scan-id>/<ma-output>/raw \
  --da3-raw data/mapanything_phone_scans/<scan-id>/<da3-output>/raw \
  --output-dir data/mapanything_phone_scans/<scan-id>/consensus_fusion
```

The input roots must contain matching view counts from the same prepared phone
frames. The builder fails rather than substituting another model or input set.

### Point-preserving 2 cm export for PTv3 and Roomform

To reuse the accepted consensus evidence at PTv3's 2 cm sampling scale without
rerunning either depth model:

```bash
python3 testpipelines/roomform/build_point_preserving_fusion.py \
  data/mapanything_phone_scans/<scan-id>/<consensus-output>/raw \
  data/mapanything_phone_scans/<scan-id>/<consensus-output>/point_preserving_fusion_2cm
```

This export does not reintroduce points rejected by consensus. It removes the
review GLB's every-other-pixel sampling, performs confidence-weighted 2 cm
aggregation, and retains voxels with at least two accepted samples and 0.80
accumulated weight. It writes GLB, PLY, NPZ, and JSON report artifacts. The NPZ
also carries accumulated weights, sample counts, and distinct-view counts.

Use the matching consensus `camera_solution.npz` when supplying this cloud to
Roomform so its point evidence and scanner stations stay in the same frame.

On the validated landscape living-room capture, 3,206,909 accepted samples
produced 417,845 RGB points. Relative to the 89,499-point 4 cm review cloud,
nearest-neighbor spacing improved from 30.5 mm to 15.8 mm, the fraction of
points isolated beyond 5 cm fell from 1.31% to 0.10%, and local roughness fell
from 19.3 mm to 9.5 mm. The raw horizontal DA3 cloud remains slightly denser,
while this fusion retains the cross-model conflict rejection.

For the validated portrait capture, the preserved input and output roots are:

```text
MapAnything: data/mapanything_phone_scans/20260801-112036-2d9225dd/outputs/raw
DA3:         data/mapanything_phone_scans/20260801-112036-2d9225dd/da3_toggle_phone_only_20260808/raw
Consensus:   data/mapanything_phone_scans/20260801-112036-2d9225dd/consensus_fusion_20260809
```

## Heatmap-style diagnostic views

The renderer applies the same floor estimation, camera-ground presentation,
rasterization, height, density, obstacle, walkable, and edge calculations to
MapAnything, DA3, and consensus outputs. It derives +X camera-right and +Z
camera-forward from the first phone view (or the selected static camera for
backend-world evaluation), writes row zero at maximum +Z, and never applies a
room-specific rotate or mirror.

Phone-only diagnostic leveling examines at most eight supported planes instead
of assuming the largest plane is the floor. Candidates must be within 35 degrees
of the mean camera-up direction, have at least 1.5% support, and imply a median
camera height of 0.7–2.2 m. The lowest qualifying plane is used; its candidates
and rejection reasons are saved in the diagnostic manifest. Camera-up here is
a visual heuristic, not calibrated IMU gravity. Missing floor evidence stops
the floor diagnostic. These display transforms do not alter raw reconstruction
or static-world alignment. Trajectory previews retain equal X/Z scale and the
complete camera path; an unlevelled model-axis projection is not a floorplan.

For reusable paths:

```bash
python3 tools/mapanything_phone_scan/render_phone_heatmap_diagnostics.py \
  data/mapanything_phone_scans/<scan-id> \
  --mapanything-raw data/mapanything_phone_scans/<scan-id>/<ma-output>/raw \
  --da3-raw data/mapanything_phone_scans/<scan-id>/<da3-output>/raw \
  --consensus-raw data/mapanything_phone_scans/<scan-id>/consensus_fusion/raw
```

The standard expanded-section label is **Diagnostic layers**. In addition to
the common Heatmap layout, consensus writes
`consensus_collaboration_diagnostics.png`, which shows:

- normalized per-model depth;
- fused depth;
- absolute disagreement;
- calibrated reliability;
- multiview consistency;
- selected provider;
- accepted versus rejected regions;
- original and optimized camera trajectories.

## Validated portrait results

The 2026-08-09 portrait fusion produced:

```text
93,939 confidence-weighted surfels
68.95% fused valid-pixel coverage
44.59% direct agreement among mutually valid model pixels
30.89% moderate disagreements resolved by multiview consistency
24.52% large disagreements rejected
8.21% total pixels filled by a validated single-model observation
```

Historical internal phone-view consistency (both parities participated in
inference; this is not independent physical accuracy):

| Metric | MapAnything | DA3 | Consensus |
| --- | ---: | ---: | ---: |
| Multiview reprojection median | 5.98 cm | 5.35 cm | **4.08 cm** |
| Internal even-to-odd median | 11.90 cm | 9.15 cm | **8.31 cm** |
| Internal even-to-odd p80 | 36.88 cm | 28.77 cm | **24.90 cm** |
| Internal odd-frame coverage | 35.1% | 38.1% | 32.7% |

That historical soft loop constraint reduced the joint trajectory's first-to-last position
gap from 74.3 cm to 55.1 cm. A stronger constraint reached 24 cm but degraded
internal depth consistency, so it was rejected. Current fusion has removed
the unconditional endpoint edge and requires verified revisit evidence.

## Static-camera alignment findings

Static-camera data was held out of the phone-only fusion and used afterward for
Noesis registration and validation.

The consensus reconstruction passed every existing alignment gate:

```text
Phone-cloud source overlap within 30 cm: 95.93%
Phone-cloud source median residual:      10.19 cm
Vertical plane median residual:           8.35 cm
Fixed-camera target coverage:            51.98%
Fixed-camera depth disagreement median:  31.45 cm
```

This is a mixed result. Consensus is best on internal multiview consistency and
held-out phone views, and it produces the cleanest floorplan representation.
MapAnything alone remains better against the current fixed-camera depth
comparison, whose earlier median disagreement was approximately 21.5 cm.
Therefore:

- keep consensus as the preferred phone-walk room reconstruction candidate;
- keep the static reconstruction as the independent alignment authority;
- retain all provider and disagreement diagnostics;
- do not claim consensus is more accurate on every static-visible surface;
- do not promote a room transform unless the Noesis alignment gate passes.

## Validated landscape DA3-conditioned MapAnything workflow

MapAnything accepts per-view depth, pose, and calibration inputs. The validated
landscape workflow uses this support to condition MapAnything with DA3 rather
than trying to inject an unordered DA3 point cloud.

The prior builder derives a sparse metric-depth input from DA3 only. It keeps
pixels that pass temporal reprojection and depth-boundary checks, then samples
10% of those reliable pixels with a fixed seed. For the validated walk, the
prior retained 6.45% of all pixels. This avoids presenting dense correlated
DA3 errors to MapAnything as ground truth.

Four controlled variants are produced:

| Variant | Purpose | Result |
| --- | --- | --- |
| DA3 pose | Test trajectory conditioning alone | Poor geometry; model-to-carrier pose median 22.21 cm |
| Sparse DA3 depth | Test metric surface conditioning alone | Strong phone consistency, but still needs trajectory registration |
| DA3 pose + sparse depth | Joint trajectory and surface conditioning | Best unfused variant; model-to-carrier pose median 1.99 cm |
| DA3 pose + sparse depth + static view | Test a calibrated static view inside the joint batch | Slightly worse than pose + depth without the static view |

The best operational result is a second consistency-gated fusion of DA3 with
the pose-plus-depth MapAnything output, using the validated DA3 trajectory as
the pose carrier. It is saved as `prior_conditioned_consensus_da3_carrier`.
The pose-plus-depth output remains the unfused comparison/control artifact.

### Landscape validation results

Every candidate below used the same static-world crop, 5 cm diagnostics, 2.5 cm
non-averaged point splat, held-out phone-frame test, and fixed-camera test.

| Candidate | Internal median | Held-out median | Static target median | Target within 30 cm | Fixed-camera depth delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| MapAnything image-only | 10.12 cm | 12.99 cm | 25.96 cm | 53.0% | 49.07 cm |
| DA3 | 5.38 cm | 8.26 cm | 14.76 cm | 65.5% | 33.51 cm |
| Earlier image-only MA + DA3 fusion | 3.57 cm | 5.31 cm | 21.59 cm | 57.1% | 29.63 cm |
| MapAnything + DA3 pose | 11.81 cm | 22.32 cm | 13.88 cm | 72.5% | 77.23 cm |
| MapAnything + sparse DA3 depth | 4.36 cm | 5.82 cm | 16.97 cm | 62.7% | 31.61 cm |
| MapAnything + DA3 pose/depth | 4.60 cm | 7.48 cm | **13.03 cm** | **65.1%** | 26.84 cm |
| MapAnything + DA3 pose/depth/static | 4.82 cm | 7.72 cm | 14.03 cm | 64.7% | 27.15 cm |
| **Prior-conditioned MA + DA3 fusion** | 4.31 cm | 6.58 cm | 13.78 cm | 62.8% | **23.73 cm** |

Compared with the earlier image-only consensus, the prior-conditioned fusion
increased valid fused pixels from 51.28% to 88.68%, increased agreement among
mutually valid pixels from 32.13% to 89.27%, and reduced large disagreements
from 42.97% to 3.23%. The two models are no longer statistically independent
after conditioning, so those agreement numbers are interpreted together with
the independently held-out phone and static-camera improvements.

The calibrated static reconstruction remains the world-frame authority and an
independent validation target. Adding its RGB/depth as a 49th joint inference
view did not improve this walk. Live detections and tracks must continue to use
the exact per-camera calibration; the phone reconstruction supplies the room
surface and floorplan, not a replacement for calibrated track projection.

### Family Room qualification

The same workflow was repeated for landscape scan
`20260810-215847-571c6efe`. Its admitted immutable lineage resolved an earlier
Family Room target-frame mismatch upstream. Current alignment validates the
calibrated camera against the authoritative cloud and refuses a backwards
target; it does not rotate a working target copy or add a Family-only display
correction.

| Candidate | Internal median | Held-out median | Static source median | Static target median | Source within 30 cm | Target within 30 cm | Fixed-camera depth delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA3 | 3.69 cm | 4.66 cm | 10.99 cm | 7.15 cm | 83.2% | 87.0% | 12.65 cm |
| MapAnything + DA3 pose/depth | 4.41 cm | 5.53 cm | 12.96 cm | 9.29 cm | 81.3% | 77.8% | 13.84 cm |
| **Prior-conditioned MA + DA3 fusion** | **3.35 cm** | **4.40 cm** | 12.25 cm | 9.37 cm | 82.7% | 78.4% | 12.94 cm |

DA3 alone remained slightly closer to the fixed-camera target, while the
prior-conditioned fusion retained the best internal and held-out phone-view
consistency and the cleaner review geometry. The selected scene-prior source is
therefore `prior_conditioned_consensus_da3_carrier`; DA3 and the unfused
pose/depth result remain required comparison evidence.

This qualifies a room-local static scene prior only. It does **not** establish
cross-camera overlap or authorize AMC/MV3DT calibration. The current topology
has no Living Room/Family Room overlap; the only prospective multi-view edge is
Kitchen/Family Room, and that edge remains disabled until the Kitchen geometry
is corrected and synchronized occupied overlap evidence passes review.

### Reusable commands

For the routine path, run the selected pose-plus-depth variant on matching
prepared phone views and DA3 raw data. Omit `--variants` when qualifying all
four variants:

```bash
  data/mapanything_phone_scan_runtime/venv/bin/python \
  tools/mapanything_phone_scan/run_mapanything_prior_variants.py \
  data/mapanything_phone_scans/<scan-id> \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --world-from-da3 \
    data/mapanything_phone_scans/<scan-id>/alignment/phone_ma_to_noesis_world.json \
  --target-revision data/virtual_twin/revisions/<approved-room-revision> \
  --calibration config/camera_calibration.json \
  --camera <camera-id> \
  --variants da3_pose_sparse_depth \
  --output-root <large-storage-root>/<scan-id>/da3_prior_suite
```

Fuse the best unfused variant with DA3 while preserving the validated DA3 pose
carrier:

```bash
python3 tools/mapanything_phone_scan/build_consensus_fusion.py \
  data/mapanything_phone_scans/<scan-id> \
  --mapanything-raw <suite-root>/mapanything_da3_pose_sparse_depth/raw \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --pose-carrier da3 \
  --output-dir <suite-root>/prior_conditioned_consensus_da3_carrier
```

Evaluate the selected result and regenerate the common diagnostic layouts:

```bash
python3 tools/mapanything_phone_scan/evaluate_mapanything_prior_variants.py \
  data/mapanything_phone_scans/<scan-id> \
  --suite-root <suite-root> \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --prior-consensus-raw \
    <suite-root>/prior_conditioned_consensus_da3_carrier/raw \
  --variants da3_pose_sparse_depth \
  --world-from-da3 \
    data/mapanything_phone_scans/<scan-id>/alignment/phone_ma_to_noesis_world.json \
  --target-revision data/virtual_twin/revisions/<approved-room-revision> \
  --calibration config/camera_calibration.json \
  --camera <camera-id> \
  --output-dir <suite-root>/evaluation_static_world
```

The suite manifest records model/package identity, prior thresholds and seed,
input hashes, pose disagreement, output coordinate frames, and the selected
review artifacts. Pose-conditioned raw NPZ files preserve the model-predicted
pose separately as `model_camera_pose`; their `world_points` use predicted
MapAnything depth and intrinsics backprojected through the supplied world pose
carrier.

### Hand off an approved PCF candidate to Noesis

Do not treat a successful evaluation directory as a deployed room. Continue at
**Seal the approved reconstruction** in the
[canonical PCF runbook](../../docs/PCF_Workflow.md). That runbook uses
`build_conditioned_scene_prior_bundle.py` to enforce the alignment and
static-visible admission gates, then uses `scripts/build_scene_prior.py` to
derive and bind the immutable 2.5 cm Scene Prior. This explicit bridge is the
only documented PCF-to-runtime path.

### Align a whole-home PCF review assembly to an authored model

Use `solve_pcf_model_structural_alignment.py` only for a Menon review overlay;
it does not publish calibration or canonical world state. The admitted static
camera supplies a bounded yaw/correspondence seed. The final uniform planar
similarity (yaw, X/Z, and XZ scale) comes from the four dominant Family Room
vertical wall planes and the reviewed Family Room floor polygon in the authored
OBJ. It deliberately does not use whole-cloud ICP or the editable camera-device
position as final geometry authority.

Run it from the repository root so the package imports resolve:

```bash
python3 -m tools.mapanything_phone_scan.solve_pcf_model_structural_alignment \
  --review-manifest "$PCF_REVIEW_MANIFEST" \
  --surfels-npz "$PCF_SURFELS_NPZ" \
  --surfels-manifest "$PCF_SURFELS_MANIFEST" \
  --authored-scene "$MENON_AUTHORED_SCENE_OBJ" \
  --room-group-map config/authored_scene_room_groups.json \
  --seed-yaw-deg "$PCF_CAMERA_SEED_YAW_DEG" \
  --output "$PCF_STRUCTURAL_ALIGNMENT_REPORT"
```

The generated `noesis.pcf.menon_structural_alignment` report is review-only and
fail-closed. Its assembly, GLB, surfel, authored-scene, room-group, and camera-
anchor digests must remain bound, all structural gates must pass, and Menon must
apply the resulting transform once to the complete assembly. `Structural fit`
restores this report; `Camera seed` is only a diagnostic comparison. Internal
room-to-room registration uncertainty remains unchanged by this global model
alignment.

### Derive a surface mesh for Menon review

`build_pcf_review_surface_mesh.py` converts the retained multi-room PCF surfels
into a colored cutaway surface for owner review. It reconstructs each room owner
independently with screened Poisson, trims triangles that are not supported by
nearby measurements, and removes only small disconnected components. Rebuilding
rooms separately prevents an uncertain cross-room registration from inventing
surfaces across a doorway or room join. The default 4 cm voxel size, 10 cm
support limit, and 1.85 m ceiling cutaway favor a readable interior without
claiming unobserved closure.

```bash
python3 tools/mapanything_phone_scan/build_pcf_review_surface_mesh.py \
  --surfels-npz "$PCF_SURFELS_NPZ" \
  --surfels-manifest "$PCF_SURFELS_MANIFEST" \
  --output-glb "$PCF_MESH_GLB" \
  --output-manifest "$PCF_MESH_MANIFEST"
```

To expose that derived GLB through the same bounded Noesis-to-Menon review
route, create a new immutable contract-v4 assembly from the active contract-v3
point assembly:

```bash
python3 tools/mapanything_phone_scan/publish_pcf_review_surface_mesh.py \
  --pcf-storage-root "$NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT" \
  --current-descriptor "$NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT/review-assemblies/current.json" \
  --mesh-glb "$PCF_MESH_GLB" \
  --mesh-manifest "$PCF_MESH_MANIFEST" \
  --source-structural-alignment "$PCF_STRUCTURAL_ALIGNMENT_REPORT" \
  --structural-output "$MENON_PCF_STRUCTURAL_ALIGNMENT_CONFIG" \
  --assembly-id "$PCF_MESH_ASSEMBLY_ID" \
  --activate
```

The publisher verifies the point-assembly, surfel, mesh, and structural-report
digest chain before changing the active review selector. It preserves the
source point assembly, copies the already reviewed transform without changing
its yaw, translation, or scale, and binds Menon to the new mesh digest. Both the
mesh and point variants remain presentation-only; neither changes calibration,
room registration, canonical world state, or tracking authority.

## Detection and tracking integration contract

The intended detection/tracking use is not to rebuild the room continuously.
Instead:

1. Build and approve a room reconstruction from a phone walk.
2. Align it to the calibrated static-camera Noesis world.
3. Save the accepted world transform and its source artifact identities.
4. Apply the existing calibrated camera-to-world projection to detections and
   tracks.
5. Render those world-space tracks against the approved room reconstruction.

Producer and consumer coordinate frames, units, camera calibration identity,
room revision, and transform provenance must match. A visually plausible but
unvalidated transform is not sufficient for live tracking.

## Offline walk consistency checks

The [trajectory motion reviewer](trajectory_motion_review.py) compares saved
provider camera poses with native gyro and accelerometer evidence. Run it in
the Room Walk environment with an explicit split reserved for heldout review:

```bash
python3 -m tools.mapanything_phone_scan.trajectory_motion_review \
  --scan-dir "$SCAN_DIR" \
  --source-output-manifest "$PROVIDER_OUTPUT/scan_outputs_manifest.json" \
  --output-dir "$REVIEW_DIR/motion" --heldout-start-s 42
```

Use `--allow-partial` only to review an explicitly incomplete provider subset;
the report retains omitted views and excludes unsupported window/gap edges.
It verifies prepared RGB identities and native timestamps, units and axes.
Optional `--fit-rotation-candidate` fits training intervals only and reports
heldout SO(3) error and excitation limits. It does not admit camera/IMU
calibration, integrate inertial position or turn camera speed into person speed.
Single-window DA3 outputs now declare their camera-to-world convention and
unaligned metric frame, matching the windowed provider contract.

The [handset reference reviewer](trajectory_reference_review.py) checks an
aligned camera trajectory against independently annotated device centers:

```bash
python3 -m tools.mapanything_phone_scan.trajectory_reference_review \
  --scan "$SCAN_DIR" --provider-manifest "$PROVIDER_OUTPUT/scan_outputs_manifest.json" \
  --alignment-dir "$ALIGNMENT_DIR" --annotation-csv "$ANNOTATIONS_CSV" \
  --annotation-binding "$ANNOTATION_BINDING_JSON" \
  --output-dir "$REVIEW_DIR/handset" --heldout-start-s 42 --timing-allowance-s 0.1
```

Choose the split and timing allowance for the actual take; the allowance must
match its saved timing evidence. The module docstring defines the annotation
CSV and binding sidecar, including actual image hashes, pixel axes, calibration,
camera/world revisions, rectification evidence and exact recorded-frame timing.
Annotate before viewing reconstructed projections. `--allow-partial` preserves
an explicit provider subset. `--fit-translation-candidate` is optional and uses
only training annotations. Heldout pixels, original comparable geometry and
nearest references, plus changed visibility coverage, evaluate that trial.
Its successful execution never replaces alignment or certifies surveyed accuracy.

The [depth registration reviewer](../../noesis/calibration/depth_registration_review.py)
reports the runtime mapping's usable domain and local slope, alongside occupied
trace status counts:

```bash
python3 -m noesis.calibration.depth_registration_review \
  --registration DS9/config/depth_registration.json --camera living-room \
  --tracking "$TRACKING_TRACE" --tracker-id 1534 \
  --output "$REVIEW_DIR/depth_registration.json"
```

The trace may be a saved tracking CSV or companion NDJSON. Optional `--samples`
accepts the explicitly bound independent same-anchor sample schema documented
in the module. Capture/time holdout, source/target range coverage and residuals
are checked separately. Exit 1 means a written but unqualified review, and exit
2 means invalid input. Supplied reference attestations are not external-content
verification. Passing this offline review never installs a mapping. Existing
fused person coordinates and phone optical centers cannot qualify depth labels.

## Important artifacts

The curated selected landscape evidence is retained at:

```text
docs/evidence/phone_walk_fusion/20260809-landscape-living-room/
```

For the validated portrait run:

```text
consensus_fusion_20260809/consensus_manifest.json
consensus_fusion_20260809/scan_outputs_manifest.json
consensus_fusion_20260809/camera_solution.npz
consensus_fusion_20260809/surfel_points.npz
consensus_fusion_20260809/consensus_surfel_reconstruction.glb
consensus_fusion_20260809/consensus_collaboration_diagnostics.png
consensus_fusion_20260809/noesis_alignment_validation/alignment_report.json
phone_only_heatmap_diagnostics/consensus_fusion/phone_only_heatmap_diagnostics.png
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_2cm.glb
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_2cm.ply
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_2cm.npz
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_report.json
```

## Configuration

Common optional environment variables:

```text
NOESIS_PHONE_SCAN_PORT=8788
NOESIS_PHONE_SCAN_STORAGE_ROOT=data/mapanything_phone_scans
NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT=/large-storage-root/noesis-phone-pcf
NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=0
NOESIS_PHONE_SCAN_RUNTIME_ROOT=data/mapanything_phone_scan_runtime
NOESIS_PHONE_SCAN_CANDIDATE_FPS=4
NOESIS_PHONE_SCAN_MAX_CANDIDATE_FRAMES=1200
NOESIS_PHONE_SCAN_MAX_SELECTED_FRAMES=256
NOESIS_PHONE_SCAN_CANDIDATE_EDGE_PX=1280
NOESIS_PHONE_SCAN_FEATURE_EDGE_PX=640
NOESIS_PHONE_SCAN_MIN_KEYFRAME_INTERVAL_S=0.40
NOESIS_PHONE_SCAN_MAX_KEYFRAME_INTERVAL_S=1.25
NOESIS_PHONE_SCAN_MAX_EDGE_PX=1920
NOESIS_PHONE_SCAN_MA_MAX_JOINT_VIEWS=80
NOESIS_PHONE_SCAN_MA_WINDOW_OVERLAP_VIEWS=24
NOESIS_PHONE_SCAN_POINT_BUDGET=600000
NOESIS_PHONE_SCAN_MA_DEVICE=cuda:0
NOESIS_PHONE_SCAN_DA3_DEVICE=cuda:0
NOESIS_PHONE_SCAN_DA3_PROCESS_RES=504
NOESIS_PHONE_SCAN_DA3_REF_VIEW=middle
NOESIS_PHONE_SCAN_DA3_MAX_JOINT_VIEWS=48
NOESIS_PHONE_SCAN_DA3_WINDOW_OVERLAP_VIEWS=16
NOESIS_PHONE_SCAN_DA3_ENGINE=data/ds9_artifacts/models/engines/da3metric_large_294x518_b3_fp16_trt10.16.engine
NOESIS_PHONE_SCAN_LOCAL_FILES_ONLY=1
NOESIS_PHONE_SCAN_STATIC_ANCHOR=0
NOESIS_PHONE_SCAN_ALIGNMENT_CAMERA_ID=living-room
NOESIS_PHONE_SCAN_ALIGNMENT_RELEASE=data/virtual_twin/releases/home_rgbmesh_20260623T2158_v1.json
NOESIS_PHONE_SCAN_ALIGNMENT_CALIBRATION=config/camera_calibration.json
```

PCF retains conditioned MapAnything raw arrays, fused raw arrays, and static-
world review artifacts. Budget roughly 1.5-2 GB for a 180-210-view room and
place `NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT` on durable large storage. Deleting a
walk from the app also deletes that walk's separately stored PCF runs.
Set `NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=1` when the conditioned MapAnything
window cannot coexist with the native appliance on the installed GPU. The app
stops `menon-appliance.target` only if it was active before PCF and restores it
on both success and failure. An interrupted phone-tool process records the
lease before stopping the target so its next startup can recover the appliance.

The app fails clearly if FFmpeg, CUDA, cached official models, the validated
TensorRT engine, or the local Three.js dependency is unavailable. It does not
select a degraded inference path automatically.

## Validation commands

```bash
pytest -q \
  tools/mapanything_phone_scan/test_phone_scan.py \
  tools/mapanything_phone_scan/test_consensus_fusion.py \
  tools/mapanything_phone_scan/test_prior_variants.py \
  tools/mapanything_phone_scan/test_prior_variant_evaluation.py \
  tools/mapanything_phone_scan/test_point_preserving_heatmap.py

python3 -m py_compile \
  tools/mapanything_phone_scan/build_consensus_fusion.py \
  tools/mapanything_phone_scan/render_phone_heatmap_diagnostics.py \
  tools/mapanything_phone_scan/run_mapanything_prior_variants.py \
  tools/mapanything_phone_scan/evaluate_mapanything_prior_variants.py
```

The consensus tests cover common-ray reprojection, source selection,
large-disagreement rejection, single-model fill validation, and loop-constrained
pose optimization.
