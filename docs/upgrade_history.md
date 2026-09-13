# Noesis upgrade history

This is the concise operational record of major runtime and behavior changes.
Detailed work orders, evidence, and superseded diagrams remain in the archives.

## 2026-09-13 — Walk-derived identity repair and independent trajectory review

- The native baseline now reserves tracking lifecycle identity before StableID
  assignment and retains a bounded private confirmed binding for eligible
  returns within 0.35 seconds of source-media time. Public disappearance and
  tombstones remain immediate. Registry deletion/recycling, epoch/generation
  changes, competing claims and contradictory appearance invalidate retention.
  Compact scalar diagnostics make the owner and retention decisions observable.
- Compact or wide partial-body boxes no longer imply sitting/lying without
  positive body evidence. Missing or malformed world calibration binding is
  distinguished from a real digest conflict; rejected geometry stays unavailable.
- Added reusable native-IMU motion review, heldout handset reprojection and
  independent-structure checks, and occupied depth-registration qualification
  review. They preserve input identity, partial coverage and rejected evidence.
  Single-window DA3 now declares its pose convention and unaligned metric frame.
- Focused runtime/contracts and phone-review tests passed. The occupied native
  replay observed a 67 ms same-generation return retaining its ID after an
  immediate public tombstone, and WebRTC decoded 245 frames. This verifies the
  changed behavior, not an overall identity-accuracy percentage. Live restoration
  produced healthy world/BEV on all three cameras and 195 exact channel cohorts.
- Broader depth-adapter testing retained 14 failures reproduced from pre-task
  source; they are not counted as a passing suite. The bounded replay needed
  forced termination after its shutdown deadline; its temporary databases were
  isolated and live service restoration succeeded.
- The recorded handset/structure conflict and depth-domain plateau remain
  explicit. No floor shift, extrinsic correction, replacement depth mapping,
  metric VIO admission or new live Scene Prior was inferred from the walk.

## 2026-09-12 — Motion-selected PCF review and explicit static exclusions

- Added an opt-in native exposure-motion revision tool, preserving original
  recordings, rejected views, RGB hashes and parent prepared identities. It
  invalidates adjacency across removed frames; ordinary preparation defaults
  and all reconstruction/alignment gates remain unchanged.
- The paired static-reference builder accepts exact recorded-frame selection
  with hashed post-dewarper exclusions. Depth fusion and RGB alignment respect
  the masks, and the keyframe remains an actual recorded observation.
- On the new Living Room capture, excluding four severely blurred views
  retained 127 views and cleared the original final-window registration failure.
  Selected PCF produced 228,229 retained surfels. Against the older 48-view PCF,
  common-reference coverage improved, while common-cell static holdout median
  disagreement worsened slightly and large-error tails improved. This remains
  offline review evidence, not metric VIO or surveyed accuracy.
- Fixed diagnostic path rendering for walks outside the static-camera crop:
  only drawn segments are clipped, without moving cameras or changing geometry.
  Native GPU inference stages restored their original service state; no live
  Scene Prior or calibration binding was changed.

## 2026-09-12 — Background RoomWalk uploads and portrait setup

- Companion 0.1.11 uses portrait for setup and saved captures, changes to
  landscape for the camera viewer, and returns to portrait on close. Rotation
  preserves the Activity and camera selection. Recording still stops on leaving
  the app; uploads continue independently with the screen off.
- A single Android foreground upload service retains progress, rate, receipt
  and failure state. It supports cancellation and explicit retry of the same
  archive, releases its bounded wake lock when finished, and leaves a terminal
  notification. Notification permission denial does not prevent transfer.
- Native uploads can receive a durable, hash-checked HTTP 202 storage receipt
  before full 8K decode and import. One server worker performs those existing
  checks, with at most two pending jobs and a bounded 30-minute decode budget.
  Original archives and failed/interrupted import states remain retained;
  final pairing and preparation still require successful validation.
- Focused server tests passed, including exact retry, queue bounds, restart,
  persistence failure and native acquisition-timestamp preservation. Android
  checks exercised screen-off progress, Activity/process loss, cancel/retry,
  receipt integrity and both orientation transitions. The real HTTPS Android
  upload to an isolated instance of the application completed with a stored
  receipt, followed by verified native import of all six fixture frames.
- Transport measurements distinguish host loopback and Android emulator
  networking from the physical LAN. The host NIC is 1 Gbps; no phone Wi-Fi
  throughput claim is made. Socket tuning showed no emulator improvement and
  was not retained. Real uploads now retain server-observed receive timing and
  throughput separately from import.

## 2026-09-11 — Fixed RoomWalk appliance address

- Companion 0.1.10 routes `TauntonMainframe.local` directly to the fixed LAN
  address `192.168.3.126` and skips DNS for that origin. The logical hostname
  remains the TLS/SNI and HTTP Host authority, so the existing certificate
  validation is preserved. Other HTTPS origins retain their IPv4-only lookup.

## 2026-09-10 — IPv4-only RoomWalk LAN connections

- Companion 0.1.9 replaces the longer setup allowance with an IPv4-only route:
  a cancellable one-second DNS A query, 750 ms TCP setup and a three-second
  complete health-check deadline. The original hostname remains the verified
  TLS/SNI identity, HTTP Host and saved capture origin.
- Android validation resolved the real hostname in 3–5 ms. The complete cold
  health check took 517 ms, loaded all three cameras in 67 ms, and reused the
  connection after eleven seconds idle. A wire-level HTTPS fixture confirmed
  IPv4, the original Host/SNI and rejection of a wrong certificate hostname or
  untrusted certificate before any HTTP request.
- A bounded Android fixture started the live Living Room recorder, renewed its
  lease and stopped cleanly over IPv4. It retained five clock probes, 363
  encoded packet records and 23 tracking records, with no packet-record drops
  or partial recorder/tracking output. No phone video was recorded by this
  connection fixture.

## 2026-09-10 — RoomWalk connection timeout repair

- Companion 0.1.8 caps individual TCP address attempts at one second and reuses
  its verified TLS configuration. Initial connection and pre-recording checks
  get a bounded setup budget, including one retry for a transient hostname
  lookup failure. Upload streaming and certificate/hostname verification stay
  on the existing HTTPS path.
- The phone service retains idle connections for 30 seconds, spanning the
  native capture's ten-second heartbeat. Clock and heartbeat operation limits
  remain short; setup no longer repeatedly consumes their budgets.
- The Android cold/idle connection check passed eight assertions: initial
  health completed in 7.53 seconds, all three room cameras loaded in 56 ms, and
  a direct clock request after eleven seconds idle completed in 2 ms. Six
  launcher tests and the host's same-socket idle-connection smoke passed.
- The updated Android coordinator started one live Living Room fixture,
  renewed its lease, and stopped encoded video and tracking cleanly. Its
  five retained clock probes included the real heartbeat; the fixture did not
  record phone video or claim simultaneous physical-phone acquisition.

## 2026-09-10 — Native paired room walks without a separate calibration session

- Companion 0.1.7 starts the selected static room recording and tracking
  observer before native phone video/IMU, renews a bounded lease, and stops both
  captures. Exact saved session/camera/phone references survive interrupted
  finalization, packaging and duplicate uploads. Normal native stops map to the
  server's clean-stop reason. Sensor-only diagnostics are optional, default to
  one minute and are capped at five; the former three-hour default is withdrawn.
- Preparation uses native angular-motion evidence as a bounded view-quality
  preference and retains per-frame motion, timestamps and source hashes. The
  existing 623-frame walk supplied motion for all 84 candidates, retaining 38
  views with all adjacent overlap checks passing. Static imagery remains an
  independent alignment source, separate from MapAnything + DA3 phone fusion.
- Live static capture retained 421 decoded 1080p H.264 frames and 28 canonical
  tracking records with no reported drops or partial status. Its calibration
  and rectification bindings passed the direct static-reference consumer.
  The APK packaged and uploaded a retained physical-phone fixture; duplicate
  upload returned the same scan and exact static-session/archive association.
  The combined review page shows video, IMU and static evidence together.
- The provisional online OpenVINS calibration path executed on the saved phone
  walk and exported calibration/covariance history, but its position estimates
  drifted grossly. It remains rejected for metric use and is not a fresh-walk
  prerequisite. No new board or long phone recording is required for capture.

## 2026-09-10 — Native phone calibration acquisition and VIO adapter completion

- Companion 0.1.6 adds a separate three-hour stationary IMU recorder with
  bounded storage, retained partial recordings, export/upload controls, and
  explicit persistent focus locking from the preview's measured lens setting.
- A dedicated service endpoint verifies and retains IMU archives without
  creating scans or assigning calibration status. Full Android VIO admission
  is recomputed after exact timing and actual stabilization checks, fixing
  an admission flag that previously remained false after verification.
- The OpenVINS adapter rectifies all five Brown-Conrady coefficients without
  changing K or encoded dimensions, and the rebuilt bridge confirms its
  required invalid-pixel mask. Seventy-two focused importer/consumer/receiver tests
  passed. The native public-data mask smoke emitted 313 initialized poses.
- The existing physical 623-frame recording still passes exact association
  and OIS/EIS-off verification. These software checks do not establish the
  missing camera/IMU extrinsics, time offset, native intrinsics or sensor noise;
  physical calibration recordings and their validation remain required.

## 2026-09-09 — Native phone reconstruction trial and diagnostic corrections

- The 20.79-second native bathroom recording retained 623 exact camera/encoder
  associations and 39 prepared views. MapAnything, DA3, and consensus completed
  on those same views. No static recording or calibrated IMU pose was used.
- Corrected phone diagnostic leveling after this recording exposed a dominant
  wall being selected as the floor. A bounded plane search now checks upright
  orientation, support, and camera height; missing floor evidence fails explicitly.
  Candidate diagnostics remain separate from reconstruction and alignment authority.
- Trajectory previews preserve equal axis scale, include every camera position,
  and label the model X/Z projection without assuming MapAnything or a level floor.

## 2026-09-09 — Physical 8K capture validated and RoomWalk viewfinder corrected

- Companion 0.1.5 opens a full-screen live viewer from **Capture**, with Record,
  short-test and stop/save controls in a narrow strip beside the image. Setup
  and transfer menus stay on the main screen. The preview closes with explicit
  camera-owner acknowledgement before the unchanged native 8K/IMU recorder
  starts; Back and backgrounding close an idle preview or save an active take.
- The exported SM-F976U short test decoded to 300 HEVC frames at 7680×4320
  over 10.03 seconds. Independent import verified all 300 camera/encoder timestamp
  associations and continuous accelerometer/gyroscope coverage. The selected
  physical camera stayed fixed and reported OIS/EIS off. One startup frame
  interval was 66.64 ms; subsequent intervals were approximately 33.32 ms.
- Companion 0.1.4 corrects viewfinder rotation and aspect using actual camera
  metadata, handles 180-degree display changes, and brings the full viewfinder
  into view when capture starts. Encoded image coordinates are unchanged.
- Full walks may use a retained successful short test for the same current
  camera/encoder configuration, with fresh prerequisite and surface-query checks.
  Longer recording reliability, native resolving detail and camera/IMU calibration
  are not established by the short test.

## 2026-09-08 — Installable RoomWalk Android companion

- Version 0.1.3 enables a bounded short recording test when the static stream
  list omits 8K but a standard session query succeeds. It freshly queries the
  actual recording surfaces, uses the same configuration for capture, enforces
  the ten-second limit in the recorder and retains rejected-attempt diagnostics.
  The target phone accepted standard 8K/30 encoder/preview configurations;
  actual capture, resolving detail and timestamp association still need testing.
- Version 0.1.2 adds direct driver queries for exact 8K encoder configurations
  when the static stream map omits that size. It queries only standard and
  advertised vendor use cases, retains each answer in the phone report, and
  does not open a recording or enable a vendor mode. Driver acceptance still
  requires a recorded test before capture or timing can be established.
- Version 0.1.1 distinguishes failure to find the companion's standard 8K mode
  from the phone's proven stock-camera recording capability. Expanded camera
  reports and an explicit HTTPS phone-report upload expose the device's modes
  for diagnosis without changing recording or timing acceptance. The installer
  fetches and verifies APK bytes in the browser before offering a local save.
- Added a signed native Android APK with Camera2/MediaCodec 8K capability
  checks, independent IMU streams, exact camera/encoder timing evidence,
  bounded storage capture, local exports and RoomWalk HTTPS upload. The
  first version includes a short timing test and retained capture recovery.
- The native importer verifies original frame associations and MP4 timing.
  Failed association remains raw RGB evidence with no acquisition-time claim;
  missing calibration continues to block metric VIO.
- SDK 36 compilation, signature/alignment verification, Android 16 installation
  and launch, the permission/unsupported-8K UI, ten timestamp-association
  checks and 45 directly affected importer/consumer tests passed. The installed
  APK also packaged two recorded fixtures that imported and prepared RGB with
  verified/unverified timing kept separate. Actual phone
  8K acquisition and throughput are not established by these software checks.

## 2026-09-07 — Correct browser phone capture resolution and readiness

- Replaced the 1080p request/4K cap with exact unscaled rear-camera 8K checks,
  rejecting unsupported modes without a resolution downgrade. Import validates
  the encoded dimensions of new strict-8K bundles while preserving legacy input.
- Setup requires preview progress and both sensor streams; invalid Generic
  Sensor clocks and changed camera geometry stop capture. Preview phases and
  requested/reported bitrate are preserved with the raw observations.
- The page states that acquisition synchronization is unverified and shows
  the roughly 30-second capacity at the requested bitrate. These implementation
  changes do not establish phone hardware support or synchronized 8K+IMU capture.

## 2026-09-07 — Retained IMU walk replay with reported camera intrinsics

- Reprocessed all 256 views of `20260906-050646-107f244c` in a separate Camera 2
  sensor-intrinsics candidate. The browser's physical lens, crop, and distortion
  processing remain unverified; this is not a transfer of the measured 8K
  calibration. Original source fingerprints and the installed 8K upload profile
  are unchanged.
- Both providers pass their window-registration checks. Consensus passes all
  15 static-fit checks with 406,847 supported surface points. Valid pixels rise
  from 64.6% to 82.4%, but the gated wall median changes from 7.90 to 8.29 cm
  and internal depth median from 6.77 to 8.22 cm. The all-comparable wall median
  improves from 11.35 to 10.96 cm; the denser result is not a uniform quality win.
- Standalone DA3 remains rejected: its wall median improves from 12.08 to
  10.91 cm, while source overlap falls below its gate. Failed evidence is retained.
- Consensus now validates provider frame IDs, hashes, timestamps, calibration
  lineage, and profile fingerprints. Fused raw views retain D5 and common-ray
  border validity; manifests preserve the profile's authority limits. Focused
  calibration/fusion tests and the recorded producer/consumer replay exercise
  this path. Metric VIO and live-world admission remain unavailable.
- The separate `imu_walk_calibrated_reprocess_20260907` evidence directory
  contains source verification, projection assumptions, provider/fusion outputs,
  passed and rejected static fits, comparison figures, and an aligned consensus
  GLB with its exact transform and source bindings. The existing walk stays intact.

## 2026-09-07 — Phone calibration import and calibrated reconstruction geometry

- Imported the supplied `roomwalk_phone_video` 8K handoff after checking all
  12 source-file sizes and SHA-256 values and JSON/NPZ numerical agreement.
  Preserved the original bundle, quality reports, observations, and D5 values.
- Added capture-mode and native-image binding, one-time preparation
  rectification, per-view calibrated K/source provenance, and visible profile
  status. Unbound or mismatched captures retain the existing RGB workflow with
  explicit reasons for not applying calibration.
- MapAnything consumes supplied intrinsics with its coupled preprocessor.
  DA3 uses measured intrinsics for metric focal scaling and backprojection,
  retaining network K and explicitly reporting that its pose/depth network is
  not conditioned on K alone. Both window paths preserve this distinction.
- The real handoff passes a bounded synthetic 8K video preparation test,
  producing calibrated 1280×720 selected views with exact profile hashes.
  This validates the import/preparation path; no compatible household video
  was reprocessed. Existing 4K uploads and the portrait browser recording do
  not match the supplied native calibration.
- The owner confirmed the phone camera app's default rear camera at 8K and
  1× on the reported Fold 8 Ultra. Bound the installed profile to matching
  original landscape video uploads. Preserved the supplied six-camera metadata
  listing and confirmation in a sidecar alongside the unchanged measured
  profile. Camera 0/physical Camera 5 is a metadata-based identity inference;
  no sensor K/D was substituted for the measured video calibration.
- Camera/IMU calibration, timing, metric-VIO admission, and live-world authority
  remain unchanged. See the [phone reconstruction guide](../tools/mapanything_phone_scan/README.md#measured-phone-camera-calibration)
  for profile import and capture matching.
- Thirty-nine focused calibration, preparation, API, provider, and window tests
  pass. Installed MapAnything/DA3 preprocessing was exercised with the real
  imported profile and synthetic images; model-forward/metric tests use CPU
  substitutes and do not establish household reconstruction quality. The
  reloaded service is healthy and its browser displays the imported 8K profile
  as available for matching uploaded videos. Direct configuration checks admit
  the matching 8K upload mode and retain rejection of browser/4K recordings.

## 2026-09-06 — Browser clock reporting preserves unverified timing

- Generic Sensor capture now records sensor and receipt clock provenance
  separately. Import validation suppresses callback-lag statistics when an
  older bundle's declared common clock contradicts its timestamps.
- Replayed the retained Living Room manifest through the corrected importer:
  all 16,045 accelerometer and 16,053 gyroscope samples retain their numeric
  values. The roughly 16.4-day origin difference is not a camera/IMU offset.
  Of 6,736 video callbacks, 1,834 precede the recorder start call; callback
  indices do not establish encoded-frame identity.
- Ten Python and five browser-capture tests pass. Added a short native timing
  preflight procedure. Camera/IMU timing is still uncalibrated and metric VIO
  remains blocked; the original capture and import report are unchanged.

## 2026-09-06 — Retained-walk trajectory, fusion, surface, and pose checks completed

- Verified-loop refinement now uses the RGB projection stored with each raw
  depth grid. It preserves source identity and temporal holdouts. The consensus
  rebuild checks the passed pose-only solution and recomputes depth admission,
  geometry, evidence weights, and distinct-view surfel support.
- On the same 256-view Living Room capture, consensus trajectory holdout p80
  improves from 0.110095 m to 0.098891 m. The rebuilt static fit passes all
  15 checks at 0.085710 m, but the baseline is better at 0.079048 m and also
  has lower internal depth error. Baseline remains the preferred room review.
  Standalone DA3 refinement and an alternate window-overlap experiment fail
  their existing gates and remain separate diagnostics.
- Added a source-bound single-room adapter for the existing measured-surface
  builder. Baseline and refined full-height/cutaway meshes were produced with
  unchanged ray/support gates. Withheld single-view points remain separate.
  The current Menon whole-home contract still needs accepted shared-frame
  geometry and exact scene/calibration/camera bindings.
- Complete static-camera localization uses retained RGB on the depth grid and
  writes insufficient-view failure evidence. Its aligned replay yields only
  three qualifying fitted views against the unchanged minimum of four; no
  aggregate pose or calibration replacement is admitted. Original
  Kitchen/Family registration and the separate Living/Kitchen bridge also
  remain rejected for cross-room use.
- Forty-three focused tests pass across trajectory, fusion, measured surfaces,
  and static localization. The retained-data producer/consumer paths, rejected
  candidates, source hashes, comparison figures, and repeatable commands are
  recorded in [Room reconstruction fitting](room_reconstruction_fitting.md).
  The reloaded phone service passes health and retains the existing passing
  alignment. Independent room measurements and new connector observations
  remain outstanding; intrinsic and metric-VIO work was explicitly deferred.

## 2026-09-06 — Room fitting uses visible structure and a reusable replay procedure

- Phone-to-static refinement now uses the same camera visibility and occlusion
  policy as validation. Foreground disagreements stay eligible. Scale, gravity,
  pose bounds, and quality thresholds are unchanged. Reports keep global and
  visible metrics separate and expose untrimmed residuals beside the existing
  robust wall statistic.
- The real paired Living Room walk now passes all 15 alignment checks in the
  browser service: wall residual improves from 0.122825 m to 0.085703 m.
  Error also improves on withheld image regions and on unchanged baseline
  support. Kitchen's matched replay passes at 0.054205 m, down from 0.081716 m.
  Family Room's old reference fails its camera-orientation preflight.
- Added a diagnostic CLI that preserves failed fits, exact input identities,
  and optional bounded snapshots outside saved scan state. Twenty-seven focused
  tests pass, including visible/hidden surfaces, rejected diagnostics, source
  binding, single-projection RGB landmarks, and agreement between saved reports
  and returned summaries. The
  live service serves all eight Living Room alignment artifacts.
- Independent DA3 reconstruction completes the identical 256 prepared views,
  but its static fit fails at 0.120824 m. Static floor/height checks also reveal
  unresolved metric provenance: configured height is 2.60 m while the captured
  transform implies 2.086880 m. No numerical calibration correction or live
  world promotion was applied.
- Consensus retains one reference RGB projection on its common camera rays.
  The previous average of differently warped images blurred landmarks; an
  RGB-only experiment restored 31 consensus views / 1,940 inliers and all
  15 fit checks at 0.079614 m, with the same fused geometry and unchanged
  thresholds. The normal rebuilt producer-to-fitter path then passed all
  15 checks at 0.079048 m with 30 consensus views / 1,907 inliers. All non-RGB
  arrays across 256 views, camera solutions, and non-color surfel data match
  the original fusion exactly. Provider-disagreement diagnostics remain
  available; the corrected consensus is a separate saved review result.
- The reusable sequence, experiment decisions, metric-measurement needs, and
  per-room replay commands are recorded in
  [Room reconstruction fitting](room_reconstruction_fitting.md).

## 2026-09-06 — Room Walk alignment consumes its paired static recording

- Paired walks now reconstruct their finalized static recording as an
  independent alignment reference, with a locked camera selection, visible
  build status, and artifact links. Invalid pairing, calibration, rectification,
  or artifact provenance fails closed. Unpaired walks retain saved references.
- Six observations use decoded recording timestamps and captured calibration
  rays. MapAnything metric range is converted to calibrated Z depth, then fused
  by the existing static agreement rule. Original model depths and estimated
  intrinsics remain diagnostic artifacts. The captured frame binding is applied
  once, and downstream PCF resolves the same verified reference.
- Eighteen focused tests passed, including the installed MapAnything
  preprocessing/range contract, pairing rejection, artifact tampering, cache
  reuse without inference, and downstream alignment/PCF routing. Syntax and
  documentation checks passed; the restarted service and browser expose the
  new workflow.
- The real `Livingroom_imu` recording produced 99,198 static points from six
  observations over 211 seconds, with 14 hashed artifacts. Alignment consumed
  that reference and passed visual matching with 29 consensus phone views,
  but its visible vertical residual was 0.122825 m against the unchanged
  0.10 m gate. The reference remains reviewable; no accepted alignment or live
  world promotion was produced, and the phone reconstruction is preserved.

## 2026-09-06 — Living Room phone reconstruction recovered with upright views

- Recovered `Livingroom_imu` (`20260906-050646-107f244c`) after reproducing its
  cross-window registration failure. The recorded pixels were sideways without
  encoded display-rotation metadata. A capture-specific, lossless 90-degree
  counterclockwise correction retained the same 256 selected observations,
  timestamps, model pixel budget, 80-view limit, and 24-view overlap.
- Saved original prepared-frame identities/hashes and the explicit pixel
  transform in the corrected preparation manifest. Preserved the raw video,
  IMU/static companion data, and previous failed attempts. This was a reviewed
  data correction, not an automatic rotation rule for later uploads.
- All five windows completed and all four camera/surface registration gates
  passed unchanged. The first join's pose-position p80 improved from 1.996 m
  to 0.085 m and orientation p80 from 61.1 to 1.89 degrees. The run saved
  332,867 review points in 395.6 seconds.
- Verified all 1,287 manifest-listed file sizes, finite/proper camera poses,
  fourteen raw views around window boundaries, seven served artifacts, and the
  Room Walk 3D view. This remains a phone reconstruction awaiting independent
  Noesis-world alignment; passing internal checks is not surveyed accuracy.

## 2026-09-06 — Room Walk removes repeated views and preserves recording gaps

- Replaced FPS-filter frame synthesis with bounded selection of existing
  encoded frames. Ordinary/browser preparation retains encoded PTS, including
  nonzero origins; native acquisition timestamp and source-frame mapping remain
  unchanged.
- Added fixed-anchor adjacent-repeat removal before adaptive selection, with
  exact decoded-content or conservative feature-supported similarity evidence.
  Coverage, endpoints, and bridge repair cannot reinsert repeated views. The
  manifest records removed candidate identities, repeat spans, and counts;
  reconstruction geometry gates remain unchanged.
- Focused preparation, browser-import, native VFR, and API tests passed. A live
  synthetic upload with three four-second holds reduced 48 candidates to three
  distinct views, removed 45 repeats, retained timestamps 0/4/8 seconds, and
  served all five checked artifacts. The synthetic scan was then removed.
- A recorded 208.908-second Living Room walk retained 256 selected views and
  produced zero repeat removals among 836 candidates. Its failed 54–80 second
  overlap had distinct images and passing visual connectivity; the maximum
  encoded timestamp gap was 113 ms. This does not establish a recording freeze
  as the cause of its existing cross-window alignment failure, and that failed
  run was preserved.

## 2026-09-05 — Room Walk gets a trusted LAN HTTPS origin

- Kept the existing phone-scan HTTP listener on port 8788 for known consumers
  and added a second Uvicorn listener on port 8789 using the existing Menon
  appliance certificate and key.
- Served both listeners from one imported FastAPI app under one explicit
  lifespan, with startup recovery before HTTPS exposure and coordinated drains
  before service shutdown. Missing or mismatched TLS material fails closed.
- Documented the trusted URL `https://TauntonMainframe.local:8789`, the current
  DNS-only certificate SAN, Android CA installation requirement, and the rule
  that a browser certificate bypass is not trust evidence.
- The in-page browser camera + IMU recorder is the default raw-observation path;
  browser callback timing remains non-metric. The calibrated native sensor
  bundle remains the metric-VIO path, and the explicit browser file input stays
  available for RGB-only uploads.

## 2026-09-05 — Room Walk can retain a bounded static-camera companion

- Added an optional paired session that starts only after the selected physical
  static camera's original encoded video and canonical tracking stream are both
  ready. It saves the source-preserving Matroska recording, packet timing,
  receive-ordered tracking/world cohorts, calibration/runtime provenance,
  bounded clock exchanges, and optional user markers beside the phone bundle.
- Kept the lane single-owner and bounded to 900 seconds per session with a
  renewable 45-second lease. Start, heartbeat, stop, finalization, and phone
  association are retry-safe; a failed phone upload preserves finalized static
  evidence, and the static companion remains separate from phone-only provider
  fusion.
- A bounded Living Room smoke retained H.264 1920x1080 at 30 fps in Matroska
  for 13.091766 seconds (391 decoded frames and 392 packet records with zero
  metadata drops), 26 contiguous tracking publications and 26 world snapshots,
  11 clock probes, 1 marker, and all four REST/calibration snapshots while the
  same Noesis run stayed ready. Browser start retry, heartbeat, and stop passed.
- Synthetic phone-fixture checks reached ready with 15 views; duplicate/manual
  TAR retry returned 200, an idempotency conflict returned 409 while preserving
  the scan, and a failed upload preserved static evidence. All 10 artifact links
  returned 200, and Chrome showed paired links with healthy stopped statuses.
  These checks do not verify Fold 8 Ultra timing or pose accuracy, do not supply
  simultaneous phone ground truth, and retain phone-camera versus body/feet
  identity as unresolved evidence.

## 2026-09-04 — Reconstruction frame, evidence, and review boundaries unified

- Added revision-bound Noesis-to-Menon transport that keeps metric frame,
  calibration provenance, artifact revision, target coordinate revision, and
  authored scene presentation identity distinct. Serialized GlobalWorldFusion
  snapshots with different accepted source edge hashes can render through one
  validated target-owned mapping; stale or tampered mappings fail at bundle
  ingress. Existing v1 bindings and the active catalog remain unchanged.
- Added bounded synchronized sensor import, conventional OpenVINS relative
  constraints, and depth-backed withheld trajectory refinement. The Fold 8
  Ultra's own capture/calibration and metric phone evidence remain pending;
  VI3 remains deferred.
- Larger inertial scale changes now rerun the existing fixed-scale world
  alignment on the actual refined depth/pose/cloud carrier before conditioning.
  Public EuRoC HTTP capture-to-VIO-to-trajectory integration and a recorded
  rescaled alignment/PCF consumer check passed. Validated static-camera
  replacement supports exact backups and regeneration of direct dependents;
  retained household inputs did not justify changing manual extrinsics.
- Recorded correlated MapAnything/DA3 confidence and distinct-view support,
  then carried observed, uncertain, and unknown geometry into review-only
  multi-room reintegration and separately rendered surface classes. The
  current connector joins remain review-only and cannot place live people.
- See the implementation records for [`WO-1`](../plans/reconstruction_work_orders/WO-1.md),
  [`WO-2`](../plans/reconstruction_work_orders/WO-2.md),
  [`WO-3`](../plans/reconstruction_work_orders/WO-3.md),
  [`WO-4`](../plans/reconstruction_work_orders/WO-4.md), and
  [`WO-5`](../plans/reconstruction_work_orders/WO-5.md).

## 2026-08-31 — Living Room BEV uses its full PCF-derived camera pose

- Added an opt-in full-pose mode to the static-camera PCF localizer and bound
  the Living Room Scene Prior to its admitted result. The estimate uses 17
  consistent early/late phone-walk views and 1,799 PnP inliers; no measured
  mount height, room-specific depth curve, or dashboard correction is used.
- Replaced the incorrect 1.934 m low-envelope camera result with the PCF pose
  and a 2.058 m camera-to-floor distance. The fused PCF floor fit has 0.518
  degrees residual tilt and 0.032 m p90 residual across 759 spatial cells.
- Activated `sceneprior_living-room_20260802T202254Z_9380ed8099bb`. Direct
  replay of the recorded Living trace places all 337 sampled track points
  inside the PCF extent and restores continuous forward motion through both
  former plateau intervals; the deepest sample reaches 7.82 m camera-forward
  at PCF world `[7.48, 0.00, 8.48]`.
- Kitchen and Family Room bindings are unchanged, and this work changes only
  Noesis BEV/world localization; Menon presentation was not modified.

## 2026-08-30 — Published-cadence ground continuity and lifecycle-correct BEV trails

- Added typed seated and upright body-to-floor projection so a detected person
  remains localizable when furniture hides every ankle/floor contact. The
  upright solver jointly estimates height and footprint from calibrated head,
  shoulder, and hip planes without using detector-box bottom; the seated lane
  carries broad uncertainty and stationary hold instead of suppressing the dot.
- Made strong upright proof require five retained planes spanning three
  anatomical height bands, while preserving complete side profiles whose
  apparent left/right width collapses under perspective. One three- or
  four-plane solve remains nonpublishing, while two compatible exact-current
  four-plane/two-band solves may establish a cold lifecycle within the bounded
  reacquisition interval; only the second current row supplies the coordinate.
  Three-plane, mixed, incompatible, expired, and basis-changing evidence cannot
  form that proof. Established relocations still require a current verified
  five-plane finalizer, preventing a low-residual wrong-range solve from
  breaking a trusted lifecycle. Current body-plane and learned-height gravity
  estimates are no longer co-fused from the same body evidence.
- Kept independently proven, queue-rooted image-motion continuity visible
  while a divergent body-plane solve remains in the physical reacquisition
  gate. The projective transition preserves but cannot advance the pending
  metric consensus, avoiding a display hole without loosening relocation.
- Made motion and stationarity follow exact published cohorts instead of raw
  adjacent frame numbers, and admitted only typed weak floor evidence through
  bounded same-basis consensus. A separate strong, short-lived torso/silhouette
  motion proof may shorten cold observed-ankle bootstrap without renewing
  itself or promoting the torso point to metric authority.
- Made two consecutive, mutually consistent current ankle-pair floor contacts
  sufficient for cold bootstrap while keeping single-ankle and mixed runs on
  the normal three-sample/motion-corroborated path. One intervening weaker
  bbox/non-floor row now consumes a bounded unavailable-row allowance instead
  of replacing the stronger pending ankle candidate; a second row or expired
  gap still clears it.
- Prevented a cold mirrored ankle sequence from overriding a strong
  revision-bound Scene Prior contradiction by repetition alone. A bounded
  two-sample exact-ankle exception requires tight plausible contact, adequate
  incidence, current detector semantic confidence, and one immutable binding on
  every sample. Independent current registered person-floor depth may still
  establish the lifecycle, and an already metric-established person may
  override an imperfect prior through the unchanged physical filter.
- Required current detector semantic confidence before a cold selected
  `floor_ray` can establish metric output and rejected shallow incidence below
  `0.20` before first metric authority; NvDCF confidence cannot turn static
  furniture into a person, and neutral PCF coverage cannot admit an unstable
  mirror ray. When any lifecycle earns its first accepted metric point, its
  exact tracking/world/BEV cohort now publishes immediately instead of losing
  the proof-bearing callback to normal cadence.
- Added separate torso image-motion and torso-range roles. Accepted confident
  non-lying rows, including standing rows, may arm and later consume a fixed
  translation-only torso-to-foot origin when current floor evidence is missing,
  under silhouette, ground-edge articulation, direction, magnitude, lifecycle,
  TTL, ray, and physical gates. The complete pose-compatible bundle is retained
  independently, so later bbox-only exact rows cannot erase or mismatch it. A
  missing anatomy row preserves only recent real motion observations inside the
  existing bounded gap. Stale depth cannot erase independent exact-current pose
  consensus.
- Added a camera-agnostic learned-height continuity lane for an established
  standing lifecycle with explicit lower-body occlusion. It bias-aligns the
  current gravity reconstruction by applying a recent three-sample raw XZ
  medoid's displacement between immutable raw and queue-visible metric world
  origins,
  preserving the selected sample's own evidence time when the medoid chooses
  an older in-window row,
  then subjects that process-only point to range, media-time, lifecycle,
  revision, segment, and physical-output gates. Strict world ingestion replays
  the origin/delta algebra before carrying the exact tracking/BEV coordinate to
  Menon as held evidence; it cannot train metric state or cross-camera fusion.
  A raw gap beyond 0.40 seconds now clears only the short medoid window. A
  restart row with that longer gap is eligible through 1.25 seconds only when
  an exact recent service commit carries the same service-owned immutable root
  and every lifecycle/segment/frame/transform/time binding still matches; a
  rejected restart cannot renew the root.
- Bound both learned-height and bbox-affine image motion to one versioned
  filter-transition proof. The strict world service now verifies the exact last
  committed source/lifecycle origin, complete media cadence, fixed human
  speed/jump limits, prediction/hold base, filter gain, process input, and
  recomputed posterior. Missing media PTS fails the producer lane closed, held
  rows cannot replace the retained metric origin, and a longer raw gap cannot
  renew that origin or bypass the service-owned restart gate.
- Made projective lineage structural instead of label-authorized. Only an
  admitted image-motion output establishes the immutable queue/service root;
  bounded CV/hold descendants can inherit but never renew it. Strict-service
  continuity keys now include the active calibration-artifact SHA-256 in
  addition to lifecycle, frame revision, and transform digest, so a changed
  artifact retires old service roots and history. Global fusion now defaults to
  the same 4 m/s velocity cap enforced by the producer and strict service.
- Made occlusion exit require both three consecutive evidence rows and 0.20
  seconds of source media time. A pending moving false sit/lie transition now
  survives a single image-motion dropout, so callback wall time cannot consume
  physical evidence budget or cause a one-frame continuity collapse.
- Bound typed stationary-hold evidence to `PersonGroundState`'s public tracking
  posture. Strict service admission now gives `posture` precedence and treats
  `world_posture` only as a fallback diagnostic when the public field is absent,
  preventing resolver `unknown` from suppressing a proven seated/lying hold or
  a resolver-only label from granting one.
- Prevented a post-occlusion bbox-only floor ray from relocating an established
  track. It may still provide physically in-gate pose-dropout continuity, but a
  distant same-lifecycle reanchor now requires observed ankle support,
  registered lower-body depth contact, or a floor ray with high-confidence
  exact-current detector-pose support, and begins a new trail segment
  only after the normal bounded consensus.
- Made observed floor support an explicit resolver authority tier. Non-floor
  torso, gravity, seat, couch, and unknown-support hypotheses remain
  diagnostic, cannot participate in mixed-support covariance intersection,
  and cannot steer either metric or bounded process state.
- Allowed an established lifecycle to reanchor after three tight same-family
  observed ankle-floor samples even when current image-motion evidence is
  absent. Cold, bbox, gravity, leg-extension, and other inferred bases remain
  motion-gated; one unavailable row is tolerated, while a second or an expired
  gap clears the candidate.
- Kept accepted-foot projective continuation non-renewing, preserved pending
  metric reacquisition consensus across it, and limited a recent projective
  posterior to one explicitly budgeted, non-chainable visible CV bridge row.
  Rate-suppressed callbacks can advance its bounded process but cannot consume
  its token, which is independent of mutable rejection diagnostics. Proven image
  transport now updates the high-uncertainty held canonical snapshot as the
  exact same coordinate shown by tracking/BEV and consumed by Menon; it remains
  ineligible as fresh metric or cross-camera fusion evidence.
- Carried strictly proven state-integrated CV predictions and anchor holds into
  that same held snapshot boundary only when the bounded process posterior
  exactly equals the tracking coordinate and its versioned transition binds
  the prior queue-visible output, retained metric origin, media PTS, visible
  segment, lifecycle, and registration. Rejected bbox3d/pre-seeded observations
  now use that same output gate and proof instead of emitting an unprovable
  tracking/BEV-only hold. Fresh camera evidence excludes held rows from
  covariance intersection; held-only cohorts select the newest exact point and
  do not train the global velocity baseline.
- Restamped a final output-speed rejection as an exact
  `bounded_output_hold` at the last published coordinate, so its strict
  observation no longer describes the rejected process candidate while the
  tracking row and snapshot display the held point. Gain-zero holds now advance
  only the visible clock while preserving a separate motion-bearing kinematic
  clock. On the first recovery after that exact condition, projective and
  metric filters may reduce gain along the original transition line to respect
  the 4 m/s latest-visible bound; ordinary observations retain strict
  quarantine rather than receiving a general slew or coordinate clip. The
  world service independently verifies both visible and kinematic limits.
- Rejected pose contacts materially below their detector silhouette, normalized
  the guarded bbox-floor minimum height across calibration resolutions, and
  prevented predictions or holds from consuming a hidden metric segment break
  by admitting them against the last ordered tracking/BEV queue watermark;
  rate-suppressed internal rows cannot become visible continuity authority, and
  public trail fields cannot leak an uncommitted internal segment break.
- Restored lifecycle-keyed BEV heads and trails: a missing canonical point
  removes the live head while retaining prior history during the gap, return
  starts a fresh visible segment, and current heads no longer depend on trail
  enablement or a second trail sample. Dashboard head state is now separate
  from trail state and reconciled against the complete exact `footpoints`
  cohort, so the capped dropped-reason sample cannot leave stale heads and a
  retained producer trail cannot recreate one.
- Bound dashboard BEV admission and visual history to exact
  `(sourceId, sourceEpoch)` timelines. A higher same-source epoch now admits a
  replay/reconnect clock rewind only after clearing local heads, trails,
  smoothing, sampling phase, and source-time state; an older epoch is rejected,
  and top-level/nested epoch disagreement fails closed.
- Corrected world-snapshot consumer freshness semantics: retained-entity
  `observed_end_us` may regress when the freshest entity disappears, while
  snapshot `sequence` and `published_at_us` remain the monotonic stream
  ordering authority.
- Matched tracker-lifecycle reappearance to the existing 0.75-second ground
  quarantine. A same-camera numeric tracker ID now retains its generation
  across a sub-750 ms dropout only when normalized bbox position and scale
  remain compatible; incompatible, expired, evicted, and source-epoch returns
  still start cold. This preserves learned height and image-motion authority
  across real short metadata gaps without keying kinematics by StableID or
  carrying state across an unproven reuse.
- Bound public tracking and BEV position validity to the exact world rows
  admitted by `CanonicalWorldService`. A service-rejected producer candidate
  is cleared before transport with
  `world_quality_reason=canonical_world_service_rejected`; the paired BEV head
  is removed through the typed receipt, and the producer's visible world
  watermark advances only after the service commit. This closes the prior case
  where tracking/BEV could draw a candidate that Menon correctly refused.
- Deferred service proof/output-cache advancement until global fusion exposes
  exact source evidence. Position, registration, and velocity-gated candidates
  now leave the prior source root unchanged and cannot authorize a later
  process row; only the existing non-authoritative held-continuation exception
  remains usable as a private source-local root. Both publisher mirrors admit
  public tracking/BEV coordinates only from `source.accepted=true` evidence.
- Kept one immutable learned-height episode root across exact coherent-torso
  process rows as well as explicit lower-body occlusion. A transient change in
  the occlusion label no longer re-roots every standing row and creates a dot
  hole; loss of both current proof forms, a non-upright transition, lifecycle
  change, segment change, or expiry still ends the episode.
- Made the ordered queue the authority that ends that inferred root. A metric
  accepted only on a rate-suppressed callback can no longer erase producer
  lineage that the strict service has not seen; inferred rows and bounded
  descendants retain the root until a queue-visible successor or expiry.
- Added a typed two-second, gain-zero upright-presence hold for a strong current
  standing row with pose, torso contact, admitted/plausible floor geometry,
  admitted range, human silhouette, confidence, and a floor candidate within
  0.75 m of the exact public output. It intentionally does not require bbox
  stationarity or a preclassified motion mode, because neither proves a
  non-moving output; the candidate stays rejected and the trail stays fixed.
- Retained a hard-capped, same-segment history of exact service-committed
  outputs for bbox-affine image-motion validation. An ordered worker may now
  accept a proof formed from a slightly older real commit while independently
  speed-gating its posterior from the latest commit. This restores valid dots
  lost to producer/consumer queue lag without admitting invented origins,
  cross-segment continuity, or stale-origin teleports.
- Separated exact-origin validation retention from permission to generate a
  continuation. Any real service commit remains matchable for at most the
  1.25-second physical-filter horizon, including a recent CV output, while the
  CV bridge itself remains non-chainable and limited to 0.40 seconds. Origins
  are expired before validating the next cohort, closing the previous one-row
  stale-cache admission at a long media gap.
- Applied the same lag tolerance narrowly to bounded CV proofs: an earlier
  exact commit is eligible only with the unchanged latest metric anchor, a
  latest-output speed gate, and the original 0.405-second bridge horizon.
  Recent-projective CV still requires an image-motion origin and cannot chain;
  output holds remain latest-only so a delayed hold cannot move a dot backward.

## 2026-08-28 — Guarded missing-contact continuity and target-frame dashboard projection

- Added one camera-agnostic detector-bottom hypothesis for tracked person rows
  that have no usable pose/depth ground contact. Admission requires a bounded
  confident upright/tall-narrow box; seated/lying evidence remains excluded,
  the candidate cannot train body height, and the universal resolver, PCF
  evidence, and `PersonGroundState` retain final authority.
- Made the admission signal honor either current detector confidence or NvDCF
  tracker confidence, so DeepStream's detector-confidence sentinel on
  tracker-generated frames no longer makes a visible tracked box disappear
  from candidate generation between detector observations.
- Corrected dashboard world-to-camera-local projection to transform raw camera
  center and axes through the exact revision-bound calibration-to-target edge
  and horizontalize them before projection. This removes the pitch/height Z
  offset and preserves the accepted Family Room map lock without changing
  Kitchen raw extrinsics.
- In a 24-second paced three-MP4 application run, the guarded bbox hypothesis
  supplied accepted canonical continuity rows in all three rooms (219 Family,
  29 Kitchen, and 10 Living in that sample) with no pipeline/runtime error and
  no new image, surface, or tensor copy. Focused resolver/ground tests and
  dashboard geometry/admission tests passed.

## 2026-08-27 — Family Room camera-to-PCF yaw map lock

- Replaced the temporary dashboard yaw comparison markers with an authoritative
  revision-bound Family Room camera-to-PCF map lock. The selected static
  PCF/video fit applies `+18.25` target-world degrees around the camera optical
  center, changing the active heading from `158.39` to `176.64` degrees while keeping
  camera position and floor elevation unchanged.
- Rebuilt and bound
  `sceneprior_family-room_20260811T015847Z_8b80dc69a7c4`; its manifest carries
  the map-lock evidence fingerprint and its corrected frame transform is
  consumed by tracking/world localization and PCF presentation together.
- Preserved the raw shared calibration bundle, image-space depth-registration
  fingerprints, and the Kitchen/Living Scene Prior bindings. Focused contract,
  calibration, and orientation checks passed before the runtime restart.

## 2026-08-25 — Universal uncertainty-aware person localization

- Replaced the active Family/Kitchen/Living room-specific measurement policy
  with one camera-agnostic resolver. Floor-ray, registered-depth, and eligible
  gravity hypotheses now retain their own evidence, full covariance, anatomical
  anchor, posture/occlusion state, revision identity, and PCF diagnostics until
  the resolver makes one current-frame decision.
- Made compatible measurements contribute through covariance intersection;
  mutually incompatible candidates remain explicit alternates instead of being
  averaged. `PersonGroundState` remains the only temporal filter and separately
  owns physical admission, stationary lock, prediction, reacquisition, and
  trail continuity.
- Carried full covariance and exact calibration/world-transform provenance into
  canonical world fusion. Mixed revisions fail closed, and held/predicted rows
  remain display continuity rather than fresh global evidence.
- Kept BEV as a pure projection of the canonical ground footprint and added a
  request-gated dashboard `Localization details` overlay for candidates,
  covariance, disagreement, selected source, legacy comparison, and exact PCF
  evidence. The normal disabled state adds no rich diagnostic payload.
- Paced three-MP4 validation exercised occupied Living, Kitchen, and Family
  scenes with exact tracking/BEV cohort matching, no duplicate lifecycle rows,
  no continuous step above the 4 m/s physical contract, no canonical queue
  overflow, no pipeline error, and zero host-copy violations. Resolver work
  averaged approximately 0.04 ms per evaluated track; the 20 FPS Living source
  and 30 FPS Kitchen/Family sources retained their recorded pacing.
- Confirmed that a remaining Family-raster exterior sample represented a person
  visible in the adjacent kitchen at the same recorded timestamp. It stays an
  honest predicted cross-room coordinate with explicit outside-PCF evidence;
  it is not snapped into Family space or silently dropped.

## 2026-08-23 — Revision-exact PCF tracking and trail projection

- Bound Living Room and Kitchen, as well as Family Room, to their exact active
  Scene Prior revisions. Living Room now consumes its recorded floor-leveling
  edge; Kitchen records its identity edge explicitly.
- Corrected BEV world projection to use horizontal camera-right/forward axes,
  removing the camera-pitch/height offset from displayed dots.
- Kept valid canonical holds visible with explicit provenance, stopped them
  from extending trails, and added bounded `cv_prediction` continuation for a
  missing, stale, or physically rejected current metric observation without
  advancing last-good measurement state. Exposed bounded drop reasons for
  genuinely unplaceable tracks.
- Made the post-tiler OSD trail join exact-frame analytics, honor the track's
  declared image basis, map into the configured source tile, and break on
  lifecycle/basis changes. Missing or invalid floor anchors no longer fall back
  or clamp to the bottom of the mosaic.
- Removed stale cached depth from current-frame position authority and made
  idle exit/reacquisition depend on coherent image motion as well as world
  evidence, retaining the physical outlier gate.
- Removed canonical filter/lock/velocity state from active authority on the
  first exact tracker-row absence. A scalar-only quarantine restores it and
  reuses the lifecycle generation only for a same-camera/tracker return within
  750 ms whose bbox position and scale remain compatible; all other reused IDs
  start cold with a new generation. The exact tombstone still breaks BEV/OSD
  trails across either case. Bounded
  reject-driven prediction to a fixed last-good anchor for 0.40 seconds
  instead of integrating drift indefinitely.
- Kept seated/lying people visible for at most 2.0 seconds only when exact-frame
  detector boxes prove stationarity; these holds remain non-measurements and
  never append trails. Added strict same-camera, same-tracker, sub-second bbox
  continuity so a brief no-embedding gap does not replace a settled StableID
- Added bounded `image_motion_prediction`: when a detector box moves while the
  current metric/depth anchor is missing or rejected, transport the last
  accepted image foot through the box affine change and project it through the
  active corrected floor. The prediction is non-authoritative, does not move
  filter state, and fails closed after an implausible bbox/ray/world-speed
  change; BEV marks it as predicted and exposes its provenance.
  with a provisional label.
- Reduced pose-contact depth work to one GPU-resident union of at most two
  compound contacts and one compact statistics read. Each contact uses a thin
  lower-leg segment plus a full ankle disk; host and CUDA paths share the same
  pixel-center predicate, so furniture beside the leg cannot enter through a
  wider native-only capsule. Only endpoints and the two radii cross the native
  boundary; frames without observed ankles fail closed before launching depth
  work, and no full-frame or host-mask branch was added.

## 2026-08-23 — Canonical Family Room ground tracking and dashboard mapping

- Bound the raw Family camera calibration to the active leveled room revision
  without modifying raw `E`; live floor rays now use the same floor/world basis
  as the Scene Prior.
- Removed the dashboard BEV's independent registered-depth/floor-ray position
  selection and second smoothing pass. Live dots are exact transforms of the
  canonical filtered world point.
- Replaced seated hip-floor projection with observed ankle/person support and
  made furniture-contaminated or bbox-only depth fail closed as position
  evidence while retaining diagnostics and original measurement age.
- Bound frontend ingestion to exact tracking cohorts and exact floorplan
  snapshot/content/calibration identity, clearing dots on errors, mismatches,
  or transport loss.
- Renamed the OSD optical-range fragment from `z=` to `depth=`.

## 2026-08-23 — Occupied tracking performance recovery

- Codified the recovered behavior as canonical hot-path invariants for future
  pipeline/native work: GPU/NVMM ownership, nonblocking bounded optional work,
  exact ordered publication cohorts, pooled reuse, occupancy-amplification
  limits, capability preservation, and separate source/encode/WebRTC proof.
- Moved the 3840x720 OSD surface to GPU mode, eliminating one full-frame
  device-to-host and host-to-device transfer per mosaic frame.
- Made the secondary DAv2 branch latest-frame-only and readiness-query-only;
  aligned frames use bounded reusable device storage, while compact ROI
  readbacks use private CUDA streams and pinned host buffers.
- Moved identity evidence and gallery autosaves off the media callback and
  prewarmed the production StableID CUDA similarity shape at startup.
- Isolated diagnostic Identity-v2 shadow resolution and visitor persistence
  from the exact tracking/BEV publication worker after paced replay exposed
  database-dominated tail latency and canonical queue overflow. The shadow
  lane now retains only the newest pending compact snapshot per source and may
  degrade its own freshness; canonical rows never wait for or accept mutation
  from it. Authoritative identity remains synchronous, reconnects still start a
  source-epoch-scoped identity tracklet, and runtime stats expose both workers'
  backlog, completion/failure totals, and stage timings.
- Set explicit eight-surface streammux and tiler pools so short bounded
  metadata work cannot exhaust the SDK defaults.
- Replaced the world journal's per-publication 10,000-row retention scan with
  direct sequence-boundary pruning and indexed age-boundary lookup while
  preserving its hash chain, bounded retention, WAL, and full durability.
- Made the live object-depth rendezvous nonblocking by default; already-ready
  exact frames and bounded prior frames retain their existing provenance and
  an explicit diagnostic wait remains available.
- Increased only the Family Room RTSP jitter buffer from 100 ms to 250 ms after
  a matched live ingest test measured a 197.9 ms Family gap at 100 ms and no
  gaps above 100 ms at 250 ms; Living Room and Kitchen remain at 100 ms.
- In a 45-second occupied live run with up to three Family Room tracks, all
  sources finished at 29.6–29.8 FPS and the encoded mosaic sustained 30.79 FPS.
  Its largest AU gap was 112.6 ms, with no gap above 200 ms, no source stall,
  no slow-peer AU drop, no pipeline error, and zero CPU-copy violations.
  Tracking publication fell from about 19.9 ms to 6.0 ms average.
- A full three-room replay with the 82.7-second Family Room motion clip then
  sustained 30.10 FPS. Encoded-AU p99 was 80 ms, the maximum gap was 144 ms,
  no gap exceeded 150 or 250 ms, all sources remained healthy at about 30 FPS,
  and WebRTC decoded 908 frames in 30 seconds without an error.

## 2026-08-19 — Kitchen/Family MV3DT promoted to explicit opt-in

- Replaced the rejected room transform with independent Kitchen and Family
  Room static-camera anchors and accepted their binding after direct dynamic
  handoff validation.
- Fixed the DS9.1 communicator startup race with one shared, non-threaded MQTT
  connection, so all three streams are online before the first batch.
- Tuned the Kitchen/Family-only object model and association gates for the
  short occluded doorway overlap, including two-frame late reassociation.
- Preserved MV3DT's batch-global native ID through the product StableID layer;
  baseline and SV3DT remain camera-scoped and unchanged.
- Replayed the complete July single-person set, two May multi-person spans, and
  a targeted difficult partial-body interval. All three July doorway episodes
  handed off, visually confirmed same-person multi-view pairs fused, and no
  separate-person or partial-body duplicate track was falsely merged.
- Captured the rendered three-camera mosaic and checked the cuboid anchor at
  four image positions. Its bottom-face centroid remained on the feet/gravity
  point.
- Made `--tracking-mode mv3dt` select the accepted profile without an evaluation
  environment flag through both the DS9 runtime and native-host supervisor.
  MV3DT remains opt-in; omitting the selector still launches baseline.
- Removed strict streammux timestamp synchronization for the non-PTP live RTSP
  set while retaining common batch frame IDs. A native-host live smoke brought
  up all three communicators, WebRTC, REST, and WebSocket, completed 412
  publication callbacks, and delivered 137 encoded mosaic frames with zero
  drops during the bounded active interval.

## 2026-08-19 — Kitchen/Family MV3DT evaluation lane

- Added an explicitly gated, evaluation-only Kitchen/Family MV3DT profile while
  leaving the canonical appliance and per-room SV3DT profiles unchanged.
- Bound the review Kitchen transform into the Family gauge and limited real
  peer exchange to Kitchen and Family Room. A Living self-topic loop avoids a
  DS9.1 empty-peer batch deadlock without creating a cross-camera edge.
- Made recorded replay use complete batches and shared batch frame IDs.
- Materialized MV3DT MQTT and analytics state outside the checkout and baseline
  writable state, restoring the V3DT-specific portrait/reflection exclusions.
- Replayed a complete July single-person cycle and a complete 120-second
  multi-person segment. Both advanced at application rate with complete 3D
  metadata; visual samples confirmed cuboid bases remain under the feet.
- Kept the lane review-only: neither cohort proves a Kitchen/Family StableID
  handoff, and the bound geometry still fails its held-out acceptance gates.
  The remaining input is a synchronized occupied shared-FOV Kitchen/Family
  capture tied to accepted connector geometry.

## 2026-08-18 — Kitchen and Family Room SV3DT profiles reached per-room parity

- Accepted the Kitchen phone-walk Scene Prior as per-room geometry and retained
  its existing reviewed V3DT camera orientation.
- Added a V3DT-only analytics bundle that removes Kitchen portraits and Family
  Room portrait, TV/reflection, and window detections before tracking while
  leaving baseline analytics untouched.
- Made the V3DT household confirmation threshold profile-controlled and used a
  one-embedding threshold in the Kitchen and Family Room profiles; the
  non-V3DT identity policy is unchanged.
- Revalidated repeated July one-person replays with zero tracks in inactive
  rooms, complete BBox3D/world output, and primary StableID reuse for 24/25
  Kitchen and 46/47 Family Room raw tracklets.
- Applied and visually checked the corrected feet-anchored cuboid in both room
  profiles at multiple image positions.
- Confirmed a short synchronized Kitchen/Family doorway overlap, but kept
  MV3DT disabled because the latest common three-room registration remains
  rejected and review-only.

## 2026-08-16 — PCF became an optional Room Walk browser stage

- Added a persisted **Generate PCF review** action after a passed DA3-to-static-
  camera alignment.
- The action runs the selected DA3-pose-plus-sparse-depth conditioned
  MapAnything variant, DA3-carried consistency fusion, and static-world
  evaluation over the exact adaptive prepared-view set.
- Added browser review for the PCF GLB, collaboration diagnostics, 5 cm layers,
  2.5 cm point-preserving layers, fixed-camera evidence, metrics, manifests,
  and run log.
- Kept the action review-only and fail-closed: it does not publish a Scene
  Prior and refuses MapAnything base walks, failed alignments, camera/revision
  mismatches, and active provider-specific added-video revisions.
- Added configurable large-storage placement through
  `NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT`; deleting a walk deletes its separately
  stored PCF runs as well.
- Added an explicit, persisted GPU runtime lease for constrained hosts. When
  configured, PCF pauses an active `menon-appliance.target`, restores it after
  success or failure, and recovers that restoration after a phone-tool restart.

## 2026-08-15 — Phone-walk frame selection became adaptive

- Replaced uniform 48-view sampling with dense quality, motion, coverage, and
  feature-connectivity selection plus a 256-view emergency ceiling.
- Added measured 80-view MapAnything windows with 24 exact overlap views,
  duplicate-pose Sim(3), robust duplicate-surface refinement, and fail-closed
  camera/surface gates.
- Added 48-view DA3 windows with 16 exact overlap views and the same fail-closed
  duplicate-pose and dense-surface registration, so all adaptive views feed the
  DA3-prior MapAnything and PCF stages.
- On the stored Living Room walk, increased 48 to 182 retained views, improved
  connected adjacent pairs from 83% to 100%, and reduced median matched-3D
  residual from 8.3 cm to 4.6 cm while recovering 98.3% of baseline points
  within 20 cm.
- The independent calibrated Living Room alignment passed every gate at 7.2 cm
  median vertical-plane residual, 89.5% vertical source overlap within 30 cm,
  8.2 cm full-cloud source median, and 97.8% full-cloud source overlap within
  30 cm.
- Confirmed adaptive selection retained 184 Family Room and 206 Kitchen views
  with 100% adjacent connectivity and without reaching the 256-view ceiling.
- Rebuilt matched PCF candidates from 182 Living Room, 184 Family Room, and 206
  Kitchen views and generated 48-versus-adaptive diagnostics using the saved
  baseline crop, 5 cm grids, 2.5 cm point layers, and DA3-carrier pose
  correspondence rather than applying the world transform twice.

## 2026-08-15 — PCF and Scene Prior orientation contract unified

- Centralized calibrated camera-ground conversion and row-zero-max-Z raster
  addressing for Scene Prior, cached scene fusion, and offline PCF diagnostics.
- Corrected new Scene Prior preview manifests to describe the written
  row-zero-max-Z PNG while retaining read compatibility for immutable revisions
  carrying the older pre-PNG numeric-grid label.
- Removed the evaluator's Living Room 180-degree convention, heatmap
  post-rotation, three negative Three.js model scales, optional canvas/raster
  flips, and the aligned-backend GLB half-turn.
- Kept the backend-to-camera determinant-`-1` basis presentation-only while
  strengthening proper metric-transform gates for PCF and multi-room inputs.
- Added asymmetric left/right/forward tests and validated the deployed Living
  Room, Family Room, and Kitchen rasters plus the review-only Family/Kitchen
  presentation without changing backend geometry.
- Detailed audit: [`PCF_Coordinate_Orientation_Audit.md`](PCF_Coordinate_Orientation_Audit.md).

## 2026-08-15 — Living Room SV3DT cuboid anchor correction

- Preserved SV3DT image-foot and 3D object metadata while removing only the
  tracker-generated red foot dot and displaced blue debug cuboid.
- Added a Living Room-only replacement cuboid with its bottom-face centroid
  fixed to the instance-mask person base and bbox-bottom fallback.
- Validated the exact centroid invariant in focused tests and inspected a
  34-second July Living Room replay at standing, walking, and seated positions.
- Kept the baseline configuration and tracking/world/StableID contracts
  unchanged.

## 2026-08-15 — PCF became a reproducible first-class capability

- Defined PCF consistently as Prior-Conditioned Fusion and recorded
  `prior_conditioned_consensus_da3_carrier` as the selected candidate.
- Added one canonical capture-to-runtime runbook with verified commands for
  conditioned inference, fusion, common evaluation, machine and human quality
  gates, evidence sealing, 2.5 cm Scene Prior construction, catalog binding,
  native runtime loading, and dashboard verification.
- Preserved the phone-only reconstruction boundary, independent static-camera
  alignment/validation authority, calibrated live-track authority, and the
  separation between measured extent and authored semantic membership.
- Documented artifact retention, safe retry/cleanup rules, a handoff prompt,
  and the active Living Room, Family Room, and Kitchen PCF inventory.

## 2026-08-15 — Documentation authority reconciliation

- Rebuilt the current index, architecture description, native baseline,
  focused testing guide, decisions, and application diagrams from the live
  native DS9.1 graph.
- Renamed active API/metadata documents to runtime-neutral names and corrected
  the dashboard, Scene Prior/PCF, depth, identity, ROI, media, and integration
  descriptions.
- Moved DS8, DS9.0, container, completed migration, bridge-audit, and inactive
  prototype guidance into explicit non-normative archives.
- Replaced stale active workstream instructions with current native DS9.1
  checklists while retaining the detailed historical worklogs.
- Updated DeepStream development/profiling/MV3DT agent routing so Noesis work
  selects native DS9.1, direct validation, and the current geometry deferral.
- Corrected operator examples to discover and load the service's installed
  `native.env` before using native-root variables.

## 2026-08-15 — Canonical native-host DS9.1 runtime

- Made `DS9/scripts/run_canonical_runtime_host.py` the runtime supervisor.
- Bound DeepStream 9.1.0, CUDA 13.2, TensorRT 10.16.0.72, GStreamer 1.24.2,
  Python 3.12, native plugins/extensions, secrets, and the accepted engine
  realization without Docker.
- Made engine maintenance native-host by default and rejected Docker authority
  in the host path.
- Preserved the accepted three-camera graph and restored package/import
  environment parity.
- Recorded in commit `3f1cae4`.

## 2026-08-19 — Live MV3DT complete-batch repair

- Changed only the explicit MV3DT profile to wait for a complete three-camera
  streammux batch; baseline remains unchanged.
- Bound the asset validator and focused tests to the complete-batch invariant
  required by the ordered MV3DT peer-message synchronizer.
- Corrected the prior 137-frame live smoke classification: it proved startup,
  not sustained progress, whereas the synchronized July replay completed 7,110
  source-frame publications and normal EOS.
- A probe-free live run sustained approximately 30 FPS per camera for 148
  seconds after first output, delivered 4,480 encoded mosaic frames without a
  drop, completed 13,443 publication callbacks, and accepted orderly EOS.

## 2026-08-15 — PCF BEV tracking authority repair

- Paired canonical committed tracking cohorts with PCF-backed BEV frames so
  dashboard dots and trails appear on Scene Prior floorplans.
- Preserved empty-occupancy behavior and did not let the static PCF source mint
  tracking authority.
- Focused tests and a short live BEV capture verified all three cameras.
- Recorded in commit `59aae04`.

## 2026-08-15 — Reproducible PCF multi-room registration lab path

- Added exhaustive learned cross-walk matching, bidirectional RGB-D PnP, a
  floor-locked planar pose graph, complete-view and temporal holdouts, bounded
  local-overlap refinement, and provenance-preserving 2.5 cm reintegration.
- Produced a review-only continuous Kitchen/Family Room reconstruction from
  the two accepted PCF walks while keeping Family Room as the fixed gauge.
- Re-exported that review through the recorded Family camera-ground basis after
  a raw backend-X/Z plot made Kitchen appear reversed. The exporter applies one
  shared presentation transform to both rooms, proves the fused NPZ remains
  unchanged, and does not introduce a Kitchen geometry flip.
- Rejected canonical admission because held-out planar prediction remained
  above gate and one temporal holdout lacked independent Kitchen support. The
  next input is a short Kitchen/Family connector walk, not another full-room
  capture.
- Detailed workflow: [`PCF_Multiroom_Registration.md`](PCF_Multiroom_Registration.md).

## 2026-08-13 — PCF became the depth-panel presentation source

- The depth drawer now opens against admitted immutable Scene Prior evidence.
- Full reconstruction extent, authored room footprint, diagnostics, and
  floorplan presentation were separated explicitly.
- Dashboard bundle and contract updates are recorded by `60d2226` and
  `5d77c65`.

## 2026-08-12 — DeepStream 9.0 to 9.1 direct upgrade

- Rebuilt the selected runtime image/toolchain authority, seven native
  extensions, seven inference parsers, one TensorRT plugin, three GStreamer
  plugins, and ten selected engine realizations for CUDA 13.2 / TensorRT
  10.16.0.72.
- Rebound DAv2 and MapAnything depth registrations to exact DS9.1 engine/config
  content while retaining accepted calibration knots and fusion policy.
- Restored the baseline WebRTC-only media path and runtime publication gate.
- Removed identity-retention work from the per-frame hot path, restoring the
  live three-camera rate to roughly 30 FPS.
- Core commits: `6e69f95`, `9d5f275`, `5a6c93c`, `b787db8`, `4c87067`, and
  `4c19515`.

## 2026-08-10 to 2026-08-12 — DS9 application parity and hardening

- Ported the shared product surface to the DS9 adapter: source progress,
  analytics, tracking/world publication, identity, pose, ReID, depth, ROI
  exclusion, media, lifecycle, secrets, artifact authority, and direct live
  gates.
- Added deferred MV3DT assets and contracts without enabling the capability.
- Container/candidate mechanics used during this transition are retired and
  archived; they are not the current development workflow.

## Earlier — DS8 application foundation

The DS8 generation introduced the Service Maker pipeline, GPU-first memory
contract, MapAnything/DAv2 depth roles, StableID, canonical telemetry, BEV,
WebRTC mosaic, and many shared services still used by DS9.1. DS8 itself is no
longer executable authority. Its plans and decisions are retained under
`docs/history/runtime/ds8/` and `plans/archive/ds8/` to explain the origin of those
contracts.

## Earlier — DS7 and pre-Service-Maker application

GI/GStreamer pad-probe and older local-only prototype material is retained in
the checksummed supplemental archive indexed by
`archive/manifests/supplemental_history_20260826.json`.
