# Noesis upgrade history

This is the concise operational record of major runtime and behavior changes.
Detailed work orders, evidence, and superseded diagrams remain in the archives.

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
