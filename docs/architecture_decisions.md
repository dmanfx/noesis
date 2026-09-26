# Architecture decisions

This is the current decision ledger. Historical DS8 and container-era ledgers
remain in the archives for rationale and chronology.

## Find the relevant decision

| Concern | Decisions |
| --- | --- |
| Native runtime, direct validation and performance boundaries | [ADR-001](#adr-001--one-native-deepstream-91-runtime), [ADR-002](#adr-002--direct-application-validation-is-the-default), [ADR-004](#adr-004--gpu-first-graph-with-explicit-cpu-boundaries), [ADR-021](#adr-021--optional-depth-and-persistence-cannot-stall-media) |
| Canonical publications, spatial authority and identity isolation | [ADR-006](#adr-006--canonical-publication-is-transactional), [ADR-020](#adr-020--one-revision-bound-ground-state-feeds-every-spatial-view), [ADR-023](#adr-023--shadow-identity-is-isolated-from-canonical-publication), [ADR-024](#adr-024--one-universal-uncertainty-aware-person-localization-resolver) |
| Baseline tracking and explicit MV3DT | [ADR-003](#adr-003--baseline-three-camera-tracking-remains-canonical), [ADR-018](#adr-018--kitchenfamily-mv3dt-is-a-ready-explicit-runtime-option), [ADR-019](#adr-019--live-mv3dt-waits-for-complete-camera-batches) |
| PCF candidates, alignment and presentation | [ADR-007](#adr-007--prior-conditioned-fusion-is-the-canonical-room-source), [ADR-015](#adr-015--the-room-walk-browser-may-build-review-only-pcf-candidates), [ADR-026](#adr-026--static-camera-pcf-map-lock-is-a-revision-bound-frame-edge), [ADR-028](#adr-028--reconstruction-evidence-and-presentation-remain-revision-bound), [ADR-043](#adr-043--path-capture-binds-the-selected-retained-pcf-version), [raster pairing](#2026-09-23--pcf-raster-format-is-versioned-independently-and-served-as-a-compatible-pair) |
| Capture, paired static references and calibrated reconstruction | [ADR-030](#adr-030--room-walk-may-retain-one-bounded-static-camera-companion-session), [ADR-033](#adr-033--paired-static-recordings-supply-independent-alignment-references), [ADR-034](#adr-034--fit-only-static-camera-visible-structure-and-retain-replay-diagnostics), [ADR-036](#adr-036--retained-trajectory-refinement-binds-rgb-rays-and-rebuilds-fusion-support), [ADR-037](#adr-037--phone-intrinsics-bind-the-capture-projection-and-provider-geometry), [ADR-041](#adr-041--roomwalk-040-separates-reconstruction-and-path-refinement-capture-intents), [ADR-042](#adr-042--metric-vio-scale-candidates-require-fresh-static-registration) |
| Agent scope, instruction hierarchy and workflow completion | [Task scope](#2026-09-05--scope-agent-workflows-to-the-authorized-application-task), [local policy](#2026-09-06--keep-agent-policy-local-to-implementation-responsibilities), [evidence and phase boundaries](#2026-09-07--operational-guidance-follows-evidence-and-authorized-phases) |

Accepted decisions define intended authority. Superseded entries retain their
status; dated tests and runtime observations belong with their evidence and do
not imply that every permitted runtime option is selected.

## 2026-09-23 — PCF raster format is versioned independently and served as a compatible pair

The `floorplan_contract_version` owns geometry and orientation, not the binary
packing of presentation rasters. The Scene Prior diagnostic metadata therefore
declares `raster_encoding_version=1`, and each compact layer declares its own
`grid_encoding` alongside the existing typed base64 prefix. Existing unprefixed
float32 grids stay readable. Continuous PCF display grids may use binary16;
exact masks use packed bits and small categorical grids use uint8. This format
cannot quantize retained PCF evidence or canonical tracking/world measurements.

The producer, built OAI² frontend, and any Menon grid consumer must be checked
as one wire-compatible set. Unsupported raster versions or encodings are a
visible compatibility failure, never an apparently empty room. A Vite source
smoke does not establish compatibility of the immutable bundle served by Menon;
production promotion uses a reviewed built artifact and tests it through the
actual gateway route before switching selectors.

## 2026-09-17 — Optional calibration motion prerequisites fail before detection and stale status recovers

The optional RoomWalk calibration/motion protocol requires an explicit qualified
camera reference and user-confirmed printed board dimensions. The isolated
worker repeats these checks before reading video and checks exact captured
lens/focus/crop binding before native target detection. A completed camera job
alone is insufficient; this protocol must not silently estimate substitute
intrinsics. Ordinary reconstruction and path-refinement capture do not require
this saved calibration reference.

The shared UI requires those explicit choices before guided calibration motion
capture or processing. It compares available native focus metadata with the
loaded camera reference without a tolerance (float32 normalization only matches
Camera2's native type). Native Record additionally checks that the viewer-entry
focus lock has not changed for that board protocol. These convenience preflights
do not replace immutable recorded metadata or the backend's exact
binding/admission checks. Unknown offline reference metadata cannot establish a
match.

The user's board confirmation is retained only with its saved definition;
editing dimensions clears it and a retained take restores its own attestation.
Processing failures lead to that saved take's settings rather than implicitly
requesting recapture. Status-fetch failures visibly mark progress stale and
retry, one request at a time, with 8–30-second backoff while the panel is open.
No retry creates another calibration job or repeats an upload.

## 2026-09-17 — Motion preflight distinguishes unsupported controls and native sensor precision

RoomWalk's point-time mapper may use Camera2's unsupported-distortion-control
active-array contract only for a directly pinned physical output with explicit
negative request/result capabilities, a retained null/empty mode list, equal
zero-origin active/pre-correction arrays, matching physical/software identities
and unchanged per-frame controls. This is separate evidence from camera-fitting
qualification. Missing capabilities or logical-camera inference still fail;
raw null values are not replaced by OFF. The mathematical row-time equation,
independent motion holdouts and downstream admission checks are unchanged.
The [solver contract](../tools/mapanything_phone_scan/native/roomwalk_calibration/README.md)
records the Android/AOSP references and required evidence.

Android sensor range/resolution identity uses exact float32 equality, matching
the native getter types. Different JSON decimal renderings of the same native
float must not invalidate stationary/video reuse. This is not a tolerance:
different sensor IDs, software revisions, units, or even one float32 ULP still
reject reuse. Original noise coefficients, provenance and artifact hashes remain.

Offline camera/motion job children explicitly bound OpenCV to two threads.
Dense 90-second motion detection retains native pixels, the 10 Hz target cadence
and 900-view cap, with a 25-minute worker budget below the parent's 30-minute
deadline. The camera-fit budget remains 20 minutes. No appliance restart,
quality/cadence reduction, implicit profile selection or runtime admission is
part of this repair.

## 2026-09-17 — Native camera corner refinement is image-scale and geometry bounded

RoomWalk's offline ChArUco detector localizes chess corners from four opposing
checker half-edges. Their midlines locate the intended intersection when white
gutters or softened edges bias a single gradient-intersection estimate. Local
board orientation and support size come from nearby detected corner geometry,
not a camera model. Support is bounded to 64 pixels inside a 193 × 193 native-
pixel patch; marker polygons and image borders are masked with gradient-support
clearance. This replaces the initial image-resolution-scaled subpixel policy,
which improved but did not qualify the retained September 16 phone take.

No fitted camera residual, heldout membership or calibration error threshold
selects the image measurement. Original points, support size, edge spread,
opposing gaps, displacement and failure reasons remain in the observations
artifact. An unsafe, weak or nonconvergent corner rejects its whole view rather
than silently removing individual high-error points. Known-geometry synthetic
tests cover gutters, blur, rotation, shear, noise and detector-to-hull plumbing.

Separate temporal and pose/scoring-corner holdouts, native-pixel acceptance
limits, original capture retention and review-only authority remain unchanged.
Reprocessing creates a separate job; a successful synthetic test or completed
job does not establish qualified phone calibration or metric VIO.

Optional board-calibration guidance keeps fixed focus but explicitly limits
angle/translation changes to the target's sharp range. Moving closer/farther is
not a requirement when it defocuses the target; changing focus mid-take would
change the bound camera configuration. Native prompt changes require a
subsequent APK build; editing guidance does not update an installed phone
application. This focus rule does not gate ordinary automatic-mode walks.

## ADR-001 — One native DeepStream 9.1 runtime

**Accepted:** 2026-08-15

The only supported DeepStream application is DS9.1 on the host through
`DS9/scripts/run_canonical_runtime_host.py`. DS8, DS9.0, and Docker are not
fallbacks. Inert legacy entrypoints remain only until the recorded destructive
cleanup is accepted. The supervisor verifies exact platform, Python, secrets,
artifact, and port authority before launch.

**Why:** Native parity restored the accepted graph and performance without a
second execution environment. One authority removes container/host drift and
the operational cost of candidate promotion mechanics.

## ADR-002 — Direct application validation is the default

**Accepted:** 2026-08-12; reaffirmed 2026-08-15

Normal work ends after focused tests and one practical direct smoke of the
changed capability. Immutable releases, selectors, state clones, evidence
sealing, broad suites, and promotion ceremony require explicit need.

**Why:** Validation must serve application behavior rather than displace it.

## ADR-003 — Baseline three-camera tracking remains canonical

**Accepted:** 2026-08-12; MV3DT option updated 2026-08-19

YOLO26-m + NvDCF baseline tracking remains the default. Kitchen/Family MV3DT is
an explicit opt-in under ADR-018; AMC remains disabled. Living Room/Family Room
do not overlap and Kitchen/Living Room are adjacency only.

**Why:** Cross-camera 3D tracking requires correct shared geometry and occupied,
synchronized overlap evidence. Enabling it earlier would create false spatial
authority.

## ADR-004 — GPU-first graph with explicit CPU boundaries

**Accepted:** DS8 era; carried forward and revalidated on DS9.1

Decode, preprocessing, inference, tracking, analytics, tiling, and OSD remain in
NVMM/GPU memory. CPU conversion is limited to declared serialization,
persistence, reconstruction, and display boundaries and is measured.

## ADR-005 — DAv2 and MapAnything have different duties

**Accepted:** 2026-08-10; rebound to DS9.1 assets 2026-08-12

DAv2 is always-on object/tracking depth. MapAnything is a gated full-frame
manual capture/reconstruction lane. Hardened registration binds both model
profiles to the exact engine and config bytes; it may not be weakened when an
engine changes.

## ADR-006 — Canonical publication is transactional

**Accepted:** 2026-08-10; lifecycle wiring corrected 2026-08-12

Tracking, its world snapshot/events, and the associated BEV publication form an
ordered committed cohort. A shared publication gate covers depth and analytics
callbacks. The exact callback that establishes a lifecycle's first accepted
metric coordinate bypasses the normal scalar cadence so proof and first visible
dot cannot be separated. Shutdown closes and drains that gate before WebSocket
egress.

**Why:** Dashboard state must never outrun committed world authority, and native
callbacks must not race transport teardown.

## ADR-007 — Prior-Conditioned Fusion is the canonical room source

**Accepted:** 2026-08-13; lifecycle and authority clarified 2026-08-15

PCF means **Prior-Conditioned Fusion**: DA3 world poses and sparse reliable
metric depth condition MapAnything, then the two surface estimates are
consistency-gated with DA3 retained as pose carrier. Its selected artifact name
is `prior_conditioned_consensus_da3_carrier`. A calibrated static-camera room
revision aligns and independently validates the phone-only reconstruction; its
RGB/depth is not inserted into the phone inference batch.

An approved PCF candidate is sealed as a room-scan bundle and converted into an
immutable, camera-bound Scene Prior before Noesis may load it. For that admitted
Scene Prior, the depth drawer and BEV floorplan use the PCF reconstruction.
Full measured reconstruction extent is distinct from authored semantic room
membership. Live tracking authority is paired into the PCF BEV payload. PCF
does not create or replace tracks, manufacture measurements, or clamp points.
Under ADR-024, revision-matched reliable PCF fields may act only as soft
likelihood evidence when the universal resolver adjudicates independently
generated person hypotheses. The canonical lifecycle and gates are in
[`PCF_Workflow.md`](PCF_Workflow.md).

**Why:** Phone coverage produces the clearest room geometry observed so far;
DA3 supplies the stable metric pose carrier, sparse DA3 depth constrains
MapAnything without treating every correlated pixel as truth, and final
consistency gating rejects unsupported conflicts. Keeping the static camera
independent preserves a meaningful registration check and the existing
calibrated tracking authority. Room-layout models, learned point refiners,
meshes, and Gaussian splats remain optional derivatives unless they demonstrate
new measured capability; they are not PCF admission dependencies or replacement
geometry authority.

## ADR-008 — Menon owns the browser boundary

**Accepted:** 2026-08-10; preserved on native host

Noesis REST and WebSocket ports remain loopback-only and require a private
internal bearer. Menon owns browser authentication, one-use WebSocket tickets,
dashboard delivery, and media forwarding. RTSP is disabled; the mosaic uses one
H.264 SHM/WebRTC path.

## ADR-009 — Stable identity is product identity

**Accepted:** DS8 era; active in DS9.1

`stable_id` is the user-visible identity. Tracker IDs are transient diagnostic
identifiers. Resident/visitor, exclusivity, generation, and persistence rules
must remain explicit and fail closed.

## ADR-010 — Useful history is archived, not normative

**Accepted:** 2026-08-15; archive layout consolidated 2026-08-26

Superseded DS7/DS8, DS9.0, container, migration, experiment, and validation
diaries are retained under the single `docs/history/` documentation archive and
`plans/archive/` work-record archive. Active indexes never route implementation
work through them.

## ADR-011 — SV3DT metadata and cuboid presentation are separate

**Accepted:** 2026-08-15 for the Living Room optimized profile

SV3DT keeps `outputFootLocation` enabled because image-foot and 3D object user
metadata are tracking evidence. The tracker-generated red foot dot and blue
projected cuboid are debug presentation only; a V3DT-only probe removes those
exact display primitives without removing their user metadata. Noesis draws a
replacement cuboid whose bottom-face centroid is the person's instance-mask
base, with tracked bbox bottom-center as a bounded fallback.

The correction is visual only: it does not change 3D world coordinates,
StableID, tracker association, or the baseline pipeline. Other room profiles
must enable and validate the presentation independently.

**Why:** NVIDIA couples useful projection metadata to a debug cuboid whose
image projection can be visibly displaced from the tracked person. Product
presentation must remain person-anchored without corrupting tracking authority.

## ADR-012 — Multi-room PCF joins are observation-gated planar pose graphs

**Accepted:** 2026-08-15 for the lab workflow; no Kitchen/Family canonical join
accepted yet

Two independently accepted PCF rooms may share a home-world transform only
after original-RGB overlap discovery, bidirectional depth-backed PnP, a fixed
metric scale, an accepted-floor Y lock, a view-balanced planar pose graph, and
complete-view plus temporal-segment holdouts. One accepted room remains the
immutable world gauge. Whole-room ICP, nearest-neighbor initialization,
reflection, scale fitting, and visually nudged transforms are not admission
paths.

If the evidence graph fails, a short doorway connector walk adds overlap; the
system does not repeat full-room capture or promote the best-looking rejected
transform. A rejected transform may drive a clearly marked review-only
reintegrated artifact, with no Scene Prior, tracking, or dashboard authority.
Human review uses one recorded reference-camera ground basis applied to all
registered rooms after reintegration. Backend-world X/Z plots are not an
orientation contract. Camera-display handedness, raster row inversion, and 3D
view transforms must not be fed back into registration or applied to only one
room. In particular, Kitchen has no pre-registration flip; its accepted proper
Sim(3) is the complete local-to-backend transform.
The reproducible workflow and gates are in
[`PCF_Multiroom_Registration.md`](PCF_Multiroom_Registration.md).

**Why:** Cross-room surfaces can produce a convincing but wrong join when a
patterned floor or one textured view dominates. Independent phone-view and
temporal holdouts measure whether the placement generalizes, while the common
accepted floor removes a weak PnP degree of freedom. A short connector is the
minimum household action that supplies missing evidence without rebuilding
accepted rooms.

## ADR-013 — PCF presentation uses one camera-ground coordinate contract

**Accepted:** 2026-08-15

PCF and Scene Prior metric geometry remains in proper local/world frames through
all registration and serialization authority. A calibrated camera-ground
display frame is derived only after registration: positive X is camera-right,
positive Y is height above floor, and positive Z is camera-forward. For the
deployed OpenCV cameras its backend-to-display linear determinant is `-1`
because camera Y-down becomes height Y-up; that matrix is presentation-only and
may transform display positions, never registration geometry or pose
rotations.

Serialized display rasters use row zero at maximum camera-forward Z and columns
from minimum to maximum camera-right X. oai2-fe draws that row order directly.
New Scene Prior revision manifests use the same literal orientation value;
already-deployed immutable manifests with the older numeric-grid label remain
read-compatible but never trigger another display flip.
Three.js preserves model geometry and obtains camera-right screen-right plus
camera-forward screen-up through camera placement, not negative model scale.
Room-specific 180-degree rotations, consumer-applied `image_flip`, and manual
presentation nudges are prohibited. The implementation and complete retained-
operation inventory are in
[`PCF_Coordinate_Orientation_Audit.md`](PCF_Coordinate_Orientation_Audit.md).

**Why:** A display reflection can be visually plausible while reversing
asymmetric landmarks or contaminating metric registration. Separating proper
physical transforms, the explicit camera-basis conversion, raster addressing,
and viewer-camera presentation makes each operation testable and prevents a
Living Room convention from becoming hidden geometry authority.

## ADR-014 — Phone walks use adaptive views and gated provider windows

**Accepted:** 2026-08-15

Phone-walk preparation analyzes a dense candidate stream and selects views from
relative image quality, viewpoint change, temporal coverage, and adjacent-view
feature connectivity. The 256-view configuration is an emergency ceiling, not
a requested count. Selection records candidate count, selected timestamps,
reasons, quality, connectivity, and whether that ceiling constrained the walk.

MapAnything remains joint only within a measured 80-view GPU window. Longer
selected sequences use 24 exact duplicate views between neighboring windows.
Duplicate camera poses initialize a proper Sim(3); pixel-corresponding 3D from
the duplicate images robustly refines scale and translation while preserving
the pose-derived rotation. Both camera and dense-surface gates must pass before
a window enters the reconstruction. Unique views are retained in one base
phone frame and their review points are confidence-weighted in 3.5 cm voxels.
DA3 uses the same exact-overlap, fail-closed registration pattern with a
48-view joint window and 16 duplicate views. The resulting full adaptive DA3
trajectory—not a 48-view truncation—carries the sparse metric priors and final
PCF reconstruction.
This does not mix in a static-camera frame or change static reconstruction and
calibration authority.

**Why:** Uniform 48-view sampling left multi-second gaps and weak or broken
adjacent overlap in all three stored room walks. The RTX 3060 runs 80 views but
OOMs at 96 even with native DS9.1 stopped, so one larger monolithic pass is not
a valid implementation. Exact-overlap windows preserve the full adaptive walk
while making both GPU capacity and cross-window geometric consistency explicit
and fail-closed.

## ADR-015 — The Room Walk browser may build review-only PCF candidates

**Accepted:** 2026-08-16

After a DA3 base reconstruction has passed registration to the explicitly
selected static camera and revision, the Room Walk browser may run the selected
Prior-Conditioned Fusion path as an optional final stage. The application runs
DA3-pose-plus-sparse-depth conditioned MapAnything over the same immutable
prepared views, fuses it against DA3 with the DA3 pose carrier, and evaluates
the result in the independent static-camera world. PCF jobs are persisted
separately on configurable large storage and expose their GLB, raw evidence,
diagnostics, metrics, provenance, and log in the browser.

This browser boundary ends at a review candidate. It does not seal a room-scan
bundle, build or bind a Scene Prior, or change live tracking/world authority.
On GPU-constrained hosts, an explicit deployment setting may give the job a
recorded resource lease that pauses the native appliance only when it was
already active and restores it on success, failure, or interrupted-tool
recovery. The current action also refuses provider-specific
added-video revisions because silently omitting or pretending to absorb those
views would violate the exact shared-view PCF contract.

**Why:** PCF is the selected room-reconstruction method, but requiring an
expert to manually reproduce three canonical commands after every accepted
walk made the routine path error-prone. A persisted, fail-closed browser job
can automate those same commands without widening the publication boundary or
weakening camera/revision identity checks.

## ADR-016 — Whole-home PCF is a release-bound Menon review sidecar

**Accepted:** 2026-08-16 for review; the current three-room registration remains
rejected for canonical use

Noesis owns the metric PCF assembly and publishes its immutable point artifact,
provenance, registration status, uncertainty, and complete binding to the
current authored scene. The binding records the scene release, authored-model,
calibration, runtime-configuration, and runtime world-alignment digests plus the
exact live `world_to_scene_col_major` similarity. It also binds the calibrated
Family Room device-reference camera pose and that camera's admitted Scene Prior
reference pose inside the final PCF. The anchor keeps metric scale, gravity,
floor height, calibrated pitch, and calibrated roll fixed. Sparse mutual RGB
matches and PCF-depth-backed PnP remain an uncertainty diagnostic only; their
free translation is not admitted. A static image supplies registration
evidence only; its points are not inserted into the phone reconstruction.

Menon is a read-only presentation consumer: it verifies those values against
its loaded cohort and applies one floor-locked planar camera correction to the
entire assembly. The correction maps the admitted Family camera X/Z and heading
to the device-reference X/Z and heading while applying zero vertical
translation, then composes with `world_to_scene`. Menon may expose the result
only as an owner-visible toggleable sidecar. It does not anchor a PCF corner or
bounding box, recenter, reflect, ICP-align, transform rooms independently,
visually nudge, or write the PCF into its authored model.

The review descriptor also carries the inferred static-camera center for every
room in the common assembly frame. Menon may render these centers as yellow
diagnostic spheres, but must place them in the same transform group as the PCF
geometry. A marker may never drive an additional per-room adjustment. Its
purpose is to make room-registration disagreement with authored camera devices
visible instead of hiding that disagreement behind the primary Family anchor.
The Menon review renderer may apply a non-authoritative ceiling cutaway above
1.85 m to the loaded point copy so the rooms remain inspectable from above.
That presentation filter never changes the immutable GLB or Noesis geometry.

Any missing or mismatched binding, changed artifact bytes, or unavailable
current release fails closed. A rejected multi-room registration remains
visibly labeled review-only with its measured uncertainty; successful visual
projection does not grant tracking, world, Scene Prior, deployment, or
canonical geometry authority.

**Why:** The authored model and PCF already share the Noesis world only through
an explicit deployed scene transform. Reusing that authority gives a
reproducible overlay without inventing a second alignment in Menon, while exact
release and byte binding prevents a convincing but stale or misregistered
artifact from silently entering the digital home.

## ADR-017 — Kitchen/Family MV3DT remains an isolated evaluation lane (superseded)

**Accepted:** 2026-08-19 for recorded-input evaluation only

The canonical appliance remains non-MV3DT. A separate Kitchen/Family review
profile may run only with `NOESIS_MV3DT_EVALUATION=1`; the runtime rejects that
profile without the explicit flag. It binds the current review-only
Kitchen-to-Family transform, keeps Family Room as the fixed gauge, and gives
only Kitchen and Family Room cross-camera MQTT edges. Living Room retains its
local geometry and uses a self-topic synchronization loop because the ordered
DS9.1 communicator blocks a batched tracker when a stream has an empty peer
entry. The self-loop is not a vision-neighbor edge and exposes no other
camera's measurements or IDs to Living Room.

Recorded cohorts use complete batches, the shared tracker batch counter for
frame IDs, and profile-owned analytics state materialized outside both Git and
the baseline writable state. This prevents appliance state from silently
replacing the V3DT portrait/reflection exclusions while preserving the
non-V3DT runtime unchanged.

The profile is not promotable. Both the July single-person cohort and the
multi-person stress segment advance normally after the communicator fix, and
visual samples place the corrected cuboid base under the tracked feet. The
single-person exclusions reduce coincident Kitchen/Family observations from
81 frames to 12, but neither cohort produces a verified cross-room StableID
handoff. The bound static transform is still rejected by its held-out geometry
gates, and the recordings do not provide an accepted same-person shared-FOV
correspondence set. Canonical MV3DT therefore remains blocked on accepted
Kitchen geometry plus a synchronized occupied Kitchen/Family overlap capture.

**Why:** This preserves a fast, directly testable two-room implementation
without granting tracking authority to rejected geometry or changing the
proven baseline/SV3DT lane.

## ADR-018 — Kitchen/Family MV3DT is a ready explicit runtime option

**Accepted:** 2026-08-19; supersedes ADR-017 for current operation

`--tracking-mode mv3dt` selects the accepted Kitchen/Family Room profile. It is
never the implicit default, so baseline and SV3DT behavior remain unchanged.
Kitchen and Family Room are the only peer edge. Living Room remains local-only;
its MQTT self-topic satisfies the DS9.1 ordered communicator without exposing
peer measurements or IDs.

The native host supervisor exposes the same explicit selector on both `check`
and `run`; omitting it remains baseline. The live cameras do not publish one
shared PTP/NTP timestamp domain, so streammux uses complete current-frame
batches with `sync-inputs: 0`, and `useBatchNumForFrameId: 1` gives every camera
in each batch the common MV3DT frame ID. The recorded lane retains its infinite
complete-batch timeout; the live lane retains the normal finite timeout.

The accepted geometry uses independent Kitchen and Family Room static-camera
anchors in the Family Room gauge. Shared MQTT connection startup prevents a
camera communicator from joining only at end-of-stream. The 1.7 m object model,
two-frame probation and common-frame gate, 0.18 peer score, 4.75 m peer fusion
safety radius, and 0.05 peer-visibility floor are confined to this MV3DT
profile. Product identity uses MV3DT's batch-global tracker ID as one StableID
manager key and unions present IDs across cameras; baseline and SV3DT retain
their camera-scoped lifecycle.

Acceptance evidence is direct application behavior: all three July
Kitchen/Family doorway episodes adopted a shared native ID, including late
reassociation after the peer view disappeared. The multi-person recording
produced shared IDs on visually confirmed same-person pairs without merging
the other person or persistent partial-body duplicate tracks. A captured
rendered mosaic confirmed that the replacement cuboid bottom-face centroid
stays on the person-mask foot/gravity point at near, far, doorway, and full-body
positions.

A native-host live-camera smoke loaded the accepted profile, connected all
three communicators, started WebRTC/REST/WebSocket, completed 412 publication
callbacks without rejection, and delivered 137 encoded mosaic frames without
drops during the bounded active interval.

**Why:** The prior blocker was not missing code; it was incorrect camera
anchors, a communicator startup race, an over-strict match threshold, and
camera-scoped handling of a batch-global MV3DT ID. Those failures are now
corrected and exercised while the normal tracking lanes remain isolated.

## ADR-019 — Live MV3DT waits for complete camera batches

**Accepted:** 2026-08-19; supersedes the live batching and live-smoke claims in
ADR-018

The opt-in live MV3DT profile uses `streammux.batched-push-timeout: -1` while
retaining `live-source: 1` and `sync-inputs: 0`. The ordered MV3DT peer-message
synchronizer and `useBatchNumForFrameId: 1` require every tracker input buffer
to contain all three camera frames. The installed `nvstreammux` documents that
a finite timeout pushes a partial batch; live RTSP jitter reproduced a tracker
wedge after 137-139 complete output batches, followed by graph-wide
backpressure and failed orderly EOS. The earlier 137-frame live smoke therefore
was startup evidence, not sustained-liveness evidence.

The synchronized July file replay completed 7,110 source-frame publications
and normal finite-source EOS with complete batches. Live MV3DT now applies the
same completeness invariant. A probe-free live run then sustained all three
sources at approximately 30 FPS for 148 seconds after first output, delivered
4,480 encoded mosaic frames with zero drops, completed 13,443 publication
callbacks, and accepted orderly EOS. An unavailable camera remains a
fail-closed condition handled by the existing per-source progress watchdog and
supervisor-owned restart; the runtime does not feed partial batches to MV3DT
or silently fall back to baseline/SV3DT. Baseline configuration and graph
construction are unchanged.

**Why:** Queue isolation cannot repair a peer synchronizer that received an
invalid partial cohort. Enforcing complete tracker cohorts removes the trigger
while preserving the accepted batch-global identity and geometry contracts.

## ADR-020 — One revision-bound ground state feeds every spatial view

**Accepted:** 2026-08-23

Baseline person grounding is estimated once in the camera's active
`backend_world_m` revision. Every active room prior has an explicit revision
binding: identity is still an explicit edge, while a leveled prior uses its
recorded rigid `Wc`. Raw `E` remains calibration evidence; live floor rays use
the active `E @ inv(Wc)` view and its leveled floor. Frame, transform,
calibration, alignment, and Scene Prior revisions must agree or the observation
fails closed.

Seated or lying hips and torsos are never projected as floor contacts. Visible
ankles or supported person pixels may contribute; the guarded `bbox_bottom`
floor ray defined by ADR-024 may also contribute when its explicit evidence
gates pass. Cached-as-current and contaminated depth remain diagnostics. BEV,
the dashboard, OSD trails, and later 3D clients consume the filtered world
state and may only apply a revision-checked view transform. A 2D BEV transform
uses horizontal camera-right and camera-forward axes, never the pitched camera
frame. They cannot reselect depth, cast a new floor ray, or smooth the canonical
point a second time.

A current canonical hold stays visible with explicit held provenance and does
not extend its trail. Its normal cap is 0.40 seconds; only a trusted
seated/lying lifecycle with stationary bbox continuity across consecutive
published observations may hold for up to 2.0 seconds. A published observation
remains exact to its tracking cohort, but continuity does not require adjacent
raw source-frame numbers: it requires a strictly increasing source frame ID and
a positive bounded media-time gap. A missing, stale, or non-monotonic
observation resets the motion or stationarity proof. Coherent image motion also
requires the current anatomical/contact point and detector silhouette to move
in the same direction on one stable image basis.
The producer stamps the longer hold's typed stationary evidence from
`PersonGroundState.posture`. At strict ingestion, the public tracking
`posture` is authoritative; `world_posture` is only a resolver-diagnostic
fallback when that field is absent. An `unknown` resolver posture therefore
cannot suppress a proven sitting/lying hold, and a resolver-only sit/lie label
cannot grant one over a contradictory public posture.

When a current depth sample is unavailable or stale, a bounded canonical
constant-velocity prediction may remain visible with predicted provenance.
Reject-driven prediction remains anchored to the last accepted state, is
capped at 0.40 seconds, does not advance last-good state, and may extend history
only when `trail_append_allowed=true`. Exact tracker-row absence removes the
state from active authority and places it in a 0.75-second quarantine. It is
restored only for the same camera/tracker key when the returning bbox also
matches the prior image position and scale; otherwise the new lifecycle starts
cold. BEV/OSD trail history remains generation-keyed and is never restored.

A present tracker row with no current canonical world point is a different
boundary: BEV removes that lifecycle's live head immediately but retains its
already accepted trail while the point is missing. A returning point begins a
fresh visible segment, so no line crosses the authority gap. Revision,
transform, lifecycle, and display-guard failures clear the affected history
instead. Backend and frontend key both heads and trails by tracker lifecycle,
not by a reusable display label; a valid current head is rendered even when
trails are disabled or have fewer than two samples. The dashboard stores
current heads separately from trail samples and reconciles them against each
complete `footpoints` cohort, so a diagnostic `droppedFootpoints` cap cannot
leave a stale dot behind and a retained backend trail cannot masquerade as a
live head. OSD joins the canonical track on the exact source/frame cohort,
maps the declared source-image basis
into the configured mosaic tile, and breaks history on tracker lifecycle or
coordinate-basis changes. It never connects a missing/out-of-bounds canonical
anchor to bbox bottom or a clamped mosaic edge.

The physical gate applies to the proposed published posterior as well as the
raw measurement innovation. Measurement-noise slack may admit evidence for
filtering, but it may not create a continuous step faster than
`max_speed_mps`. Such a proposal is quarantined and uses the bounded prediction
path; only evidence-backed reacquisition may relocate the head, and that
relocation always increments the trail segment. Only the accepted current
metric observation may consume that segment break; a prediction or hold still
has to satisfy the visible output-speed gate and cannot use a hidden filter
segment change to publish an instantaneous relocation. Final admission compares
every alternate source with the last lifecycle/revision point admitted to the
ordered tracking/BEV publication queue. Rate-suppressed callback rows may update
internal filter state but cannot replace that visible watermark. Rejected
alternates restore its coordinate and segment while advancing only media time.
An exact gain-zero `bounded_output_hold` advances that visible publication
clock but preserves a separate motion-bearing kinematic coordinate, media PTS,
and segment. Only after such a hold may the next otherwise admissible metric or
projective recovery reduce its filter gain along the original
prediction-to-measurement line so the emitted point remains within 4 m/s of
the latest displayed point. The kinematic clock still supplies the real
elapsed motion budget, and the strict service independently checks both clocks.
Ordinary adjacent observations do not receive this slew treatment: they retain
the existing reject/quarantine or evidence-backed segment-break behavior, and
no coordinate is post-hoc clipped.
Public trail fields are bound to that admitted watermark as well: a nearby
prediction may preserve a newer hidden metric segment internally, but it
publishes the prior visible segment with no break until a visible metric row
legitimately consumes the relocation.
The strict world service is also the final positional admission boundary for
the compatible tracking row and its BEV head. Before any bytes are released,
the tracking publisher normalizes `world_valid` against the exact set of
source/tracker/lifecycle/frame observations that the service committed with a
world coordinate. A producer candidate rejected there is cleared to
`world_valid=false`, loses its coordinate and process provenance, cannot append
a trail, and reports `canonical_world_service_rejected`. The typed publication
receipt binds that accepted key set to the paired BEV render, and the producer's
queue-visible world watermark advances only after the commit and only for those
accepted keys. Tracking, BEV, the nested/standalone world snapshot, and Menon
therefore cannot expose different answers for the same cohort.
Fusion rejection also cannot become hidden future authority. The service
advances its source-local output/proof cache only for exact source evidence
accepted into the resulting snapshot; position, registration, and velocity
rejections leave the prior root unchanged. The one explicit exception is a
`non_authoritative_held_continuation`, which may remain a source-local process
origin while fresh evidence from another camera owns the fused entity. Public
tracking and BEV admission still require `source.accepted=true`, so that
exception cannot mint a competing displayed coordinate.
Every canonical CV/hold row carries a versioned output-transition proof that
binds this queue-visible watermark and the independently retained last metric
watermark to exact media PTS, lifecycle, segment, and registration. The strict
world service recomputes the gain-one bounded process step or gain-zero output
hold and rejects a jointly rewritten posterior. Ordered-worker lag may bind a
bounded CV row to an earlier exact commit only if its metric anchor still
equals the latest metric anchor and the posterior is also speed-gated from the
latest output. A recent-projective bridge still requires an image-motion origin
inside 0.405 seconds and cannot chain from CV; an output hold is latest-only.
The service output key includes source, camera, tracker ID, lifecycle
generation, world-frame revision, world-transform SHA-256, and the active
calibration-artifact SHA-256. A calibration artifact change therefore retires
the old output history and every continuity root even if human-readable
revision labels happen to match. Global fusion uses the same 4 m/s default as
the producer and strict service, so identity continuity cannot reintroduce a
faster published velocity after source-local admission.
Pre-seeded/bbox3d measurements
enter this same gate after physical rejection; they do not own a parallel
continuation contract.
Current observed-ankle floor
candidates may accumulate while published-cadence image-motion consensus is
still warming, but they cannot relocate an established state until that
independent motion proof is current.

BEV visual continuity is scoped to the exact `(sourceId, sourceEpoch)` media
timeline. The top-level epoch and nested cohort epoch must match. A higher
epoch for the same source admits a legitimate replay/reconnect clock rewind,
but the dashboard clears its source-local heads, trails, smoothing, sampling
phase, and source-time clock before consuming the first new-epoch cohort. A
lower epoch is stale and cannot restore an earlier display timeline.

World snapshot observation extents are entity-set metadata rather than stream
watermarks. Removing the entity with the freshest retained evidence may lower
`observed_end_us` in the next valid snapshot. Snapshot `sequence` and
`published_at_us` remain the monotonic publication authority used by Menon and
other consumers for freshness.

**Why:** A second display estimator and two different floor revisions produced
posture- and range-dependent metre-scale placement errors. One explicit state
and one immutable frame edge make every view numerically comparable and keep
uncertain evidence visible without granting it position authority.

## ADR-021 — Optional depth and persistence cannot stall media

**Accepted:** 2026-08-23

The mosaic path uses GPU `nvdsosd`; mapping the 3840x720 RGBA surface through
the host is forbidden. Full-frame DAv2 remains a secondary observation branch:
its queue is latest-frame-only, device readiness is query-only, and its private
CUDA work cannot back-pressure tracking, OSD, or NVENC. Compact per-person ROI
results use private streams and reusable pinned host buffers.

Periodic identity evidence and gallery persistence run through bounded writers
outside the media callback. StableID prewarms the production-shaped CUDA
similarity operation during startup so lazy Torch/CUDA initialization cannot
land on the first occupied frame. Mux and tiler pools are explicitly sized for
the bounded metadata queues rather than relying on SDK defaults.

Native tensor/surface extraction is single-owner per frame and shared by its
consumers. Work amplified by detections, tracks, cameras, clients, or retained
evidence has explicit bounds. Performance changes preserve the selected models,
resolution, inference cadence, tracker, and outputs unless an explicit matched
quality tradeoff is approved. Config flags alone do not establish zero-copy;
the native consumer and source/encode/delivery layers must be measured. The
operational rules are centralized in
[`performance_invariants.md`](performance_invariants.md).

**Why:** The observed corruption and freezes were buffer starvation and
synchronization effects, not insufficient detector throughput. Optional
evidence and durable I/O must degrade their own freshness without delaying the
authoritative video/tracking path.

## ADR-022 — Image-motion prediction is display-only continuity

**Accepted:** 2026-08-23

When a current person ground measurement is missing or physically rejected, the
canonical producer may transport the last physically accepted image foot from
an exact current torso-motion reference, with detector-box affine motion as the
shorter fallback, then project that transported pixel through the active
corrected floor plane. The torso reference is current image anatomy only; it is
never itself projected as a floor or depth hypothesis. A physically accepted,
confident non-lying metric row, including a standing row, may arm the accepted
foot and torso reference. A later confident non-lying row may consume that
reference only when current floor evidence is missing or rejected and every
projective gate below passes. The complete pose-compatible origin is stored as
an independent immutable foot/world/bbox/torso bundle; a newer bbox-only metric
row may refresh the short bbox origin without erasing or mixing that bundle.

The transport is non-integrating in image space. Every result is recomputed
from the last physically accepted foot, bbox, torso reference, timestamp, and
lifecycle; a predicted row never becomes the next transport origin. The anatomy
path applies torso translation to the accepted foot and never scales that
offset from a detector box shortened by occlusion. Both torso references and
the accepted and transported feet must remain within bounded silhouette
envelopes, each foot must remain below its torso reference, and torso/bbox
displacements must agree in direction and bounded magnitude. The ground-motion
comparison uses detector-box bottom-center, not box center: a torso/box-center
rise while the bottom edge remains planted is articulation, not foot
translation. That hard rejection also blocks the bbox-affine fallback for the
row. Lifecycle, basis, TTL, image speed, silhouette scale, ray, range,
world-delta, and metric-speed gates bound the resulting
`image_motion_prediction`.

If policy allows it, the floor-projected weak observation must still pass
`PersonGroundState`'s physical CV and posterior-speed admission. It may advance
the bounded canonical process origin and append an explicitly predicted trail
point, but it never advances `last_good_world`, accepted image geometry, or
fresh global-measurement authority. When its fixed-origin provenance is
complete and `state_integrated=true`, the global service carries the exact point
into the canonical snapshot as a high-uncertainty held coordinate. This is an
exact BEV/Menon parity rule, not permission to train cross-camera fusion. It also
neither increments nor clears a
pending same-basis metric reacquisition consensus; that candidate position,
timestamp, basis, and count are restored around projective evaluation.

Both bbox-affine and learned-height continuations publish one shared version-1
filter-transition proof captured at the branch that chose the posterior. The
proof names the exact queue-admitted source output, complete media-time basis,
bounded prediction/hold base, gain, and process observation. The canonical
world service retains its latest point plus a bounded same-segment history of
its own exact commits per camera/source/tracker/lifecycle/revision. It verifies
the named origin against that service-owned set, recomputes the posterior
algebra, and independently enforces the fixed human speed bound from its latest
point. This absorbs finite lag between the media estimator and ordered
publication worker without allowing a stale origin to jump past current state.
The service retains any exact committed output for this validation purpose for
at most the 1.25-second filter horizon; it does not require that output itself
to be image-derived or metrically fresh. This cache rule is intentionally
separate from the 0.40-second, non-chainable CV-generation budget.
The ordered producer watermark and the strict service each also own an
immutable projective-episode root: only a committed
`image_motion_prediction` may establish it, bounded CV/hold descendants may
carry it, and neither descendant may renew it with its own publication PTS.
An ordinary metric/other output or expiry terminates the lane. Producer labels
can request the bridge but cannot create or resurrect that root.
Mutating both `world` and
`world_filter_prediction` therefore cannot turn an unrelated coordinate into a
valid held observation. Learned-height alignment additionally retains the last
metric queue output separately from later held publications; a raw-evidence gap
beyond 0.40 seconds clears only its short medoid window. The original raw and
trusted-world roots remain immutable for that occlusion episode. A later raw
window may resume from them only within 1.25 seconds and only when the exact
inferred root and a recent service-committed output carrying that same root
match; lifecycle, segment, frame, transform, calibration-bound output key, and
monotonic evidence time must still agree. A root mismatch, expired output, or
gap beyond 1.25 seconds rejects that restart row. It cannot renew the root; any
later candidate must independently satisfy the ordinary recent-evidence or
exact-root restart gates.
An inferred root ends only when the ordered publication queue exposes a
committed metric or unrelated projective successor. A metric candidate accepted
on a rate-suppressed callback is not public service authority and therefore
cannot erase the producer root while the service still owns it. Inferred-image
rows and their bounded CV/hold descendants retain the same root until that
queue-visible successor, lifecycle/revision change, or expiry ends the episode.

After
accepted-foot transport becomes unavailable, exactly one short-TTL
`cv_prediction` may bridge from an immediately preceding projective process
posterior. Successful projective integration stamps a dedicated one-visible-row
token. Rate-suppressed callbacks may advance only its dedicated bounded process
timestamp; they cannot spend the token. Enqueuing the bridge in the canonical
tracking/BEV cohort clears it, and a different metric mutation cannot
impersonate a bridge advance. The visible CV continuation therefore cannot
chain. Stationary
`anchor_hold` remains the only fallback for a seated/lying box proven
stationary, and failed transport after material bbox motion fails closed.
Strict state-integrated `cv_prediction` and `anchor_hold` rows follow the same
held snapshot boundary when their `bounded_cv_process` posterior exactly equals
the emitted world coordinate. A final output-speed rejection is restamped as a
typed `bounded_output_hold` at the exact last published coordinate instead of
retaining diagnostics for the rejected candidate. Global fusion never averages a held process row
with fresh camera evidence and never averages multiple held rows: fresh metric
evidence wins, otherwise the newest held point is selected exactly. Held rows
also do not update the independent global velocity baseline. Entity lifecycle
remains based on observation age, so a current held row is still `present`.

A current pose-confirmed standing person can extend that exact gain-zero hold
to a fixed two-second metric horizon when detector and tracker confidence, a
tall/narrow silhouette, torso-motion contact, floor admission, floor-contact
plausibility, range, lifecycle, and floor-candidate distance from the public
output all pass. Bbox stationarity and a preclassified walk/idle label are not
authority for this lane: the output does not move, and requiring either created
a one-frame dot hole during ordinary walking. The candidate is never promoted;
the service independently validates the typed exact-frame evidence and holds
only the last public coordinate.

**Why:** The tracker can move coherently while DAv2/pose contact is temporarily
unavailable. Reusing a fixed world point makes the BEV lie about position;
integrating a rejected CV state compounds error. A bounded projective bridge
keeps all displays on the same PCF world while preserving honest authority and
bounded failure behavior. Recomputing the bounded bridge from accepted image
geometry also prevents repeated predictions from accumulating pixel drift.

For an established confidently standing lifecycle whose lower body is
explicitly occluded, the learned upright height may also produce an
exact-current gravity reconstruction as process-only evidence. Its absolute
camera-space bias is removed without a camera-specific correction: the first
eligible raw point is paired with the last queue-admitted metric world point,
and later rows add only the raw XZ displacement from a recent three-sample
medoid. If that medoid selects an older in-window observation, its original
timestamp and media PTS remain explicit; it is not mislabeled as current. The
selected sample, consensus span, and consecutive raw-evidence gap remain within
0.40 seconds during ordinary continuation. A longer gap starts a fresh medoid
window without changing either origin; the service-owned 1.25-second restart
gate above decides whether that same episode may resume. Both origins are
immutable for the occlusion episode. Lifecycle,
target revision, trail segment, range, monotonic media time, occlusion evidence,
and physical posterior admission must remain exact; inferred rows never renew
accepted geometry, body height, the metric anchor, or either origin. The global
service independently replays the algebra and admits the result only as the
same high-uncertainty held point already published by tracking and BEV.

Occlusion exit is evidence-time bounded, not callback-rate bounded. The
producer requires both three consecutive exit rows and 0.20 seconds of
monotonic source media time. A moving apparent sit/lie transition remains
pending through a single missing image-motion callback and clears only after
that same combined budget, preventing processing stalls or bursty callbacks
from changing physical semantics.

Clearing the occlusion label does not authorize a distant detector-bottom
relocation. A reanchor requires observed ankle support, registered lower-body
depth contact, or a floor ray paired with a high-confidence exact-current
detector pose on the same track. Without one of those supports, bbox-only floor
rays may continue only inside the existing physical gate; they cannot
accumulate a same-lifecycle reanchor consensus. This keeps ordinary
pose-dropout continuity while preventing a displaced association from earning
a new world origin.

## ADR-023 — Shadow identity is isolated from canonical publication

**Accepted:** 2026-08-23

Identity-v2 authoritative mode remains synchronous after the source metadata
walk because its decision owns the same-frame public identity. Shadow mode is
diagnostic: the media callback copies at most 64 compact primitives with a
bounded embedding dimension only after the exact tracking/world/BEV cohort is
admitted. It gives a separate deep scalar snapshot to one bounded shadow
worker. That worker retains at most the newest pending item per source and at
most 64 pending sources; replacing a stale same-source item, rejecting an
excess source, or becoming unavailable affects shadow freshness only. It does
not retain frame surfaces, SDK objects, diagnostic rows, or public embeddings.

Canonical rows never wait for shadow scoring and are never mutated by its
result. A rate-gated camera frame has no shadow work. An admitted cohort,
including an empty heartbeat, is eligible for shadow work but may be superseded
by a newer pending cohort from the same source. A source reconnect scopes the
coordinator's tracker key by `source_epoch` while leaving the public
process-local numeric tracker ID unchanged. Copy, binding, mode, resolver, or
visitor-persistence failure degrades and stops only the shadow lane; it does
not fail the canonical publication worker or stop media. Authoritative
identity retains its synchronous fail-closed behavior.

The runtime exposes current/high-water pending counts, in-flight state,
completed/coalesced/drop/failure totals, and per-call shadow and canonical
publication timings in the existing core-path stats. The canonical worker
remains exact per-source FIFO with no coalescing or silent drops.

**Why:** A paced replay against the populated visitor store measured shadow
processing at about 22.4 source-frames/s, including roughly 89 ms p95 tails,
while canonical publication required about 29.6 source-frames/s. Visitor
observation/session SQLite mutations dominated that tail. Sharing the exact
publication worker therefore accumulated backlog until its finite queue
failed. Shadow comparison and visitor persistence are reconstructable evidence;
tracking/world/BEV are the product authority and must remain independent.

## ADR-024 — One universal uncertainty-aware person localization resolver

**Accepted:** 2026-08-25

The canonical baseline no longer selects a floor-only, depth-only, or nominally
fused algorithm by camera or room. It constructs a bounded exact-cohort set of
independent `floor_ray`, `registered_depth`, optional `pose_scale`, and
`gravity_reconstruction` hypotheses and resolves them with one camera-agnostic
algorithm. Camera-specific calibration and registration residuals enter as
measurement evidence and covariance; camera, room, site, and home identity
never select estimator behavior. The historical
`world_measurement_fusion_policy.json` remains non-authoritative, DS9-only
comparison material. It contains one canonical depth-registration binding per
camera rather than runtime-lane alternatives. The native baseline validates
and loads it into a separate
diagnostic-only field, then consults it only while a dashboard requests
`Localization details`. It reconstructs the retired current-frame selection
from the resolver's already-built compact candidates; it cannot feed
`track.world`, PersonGroundState, or any canonical consumer.

Every candidate carries a finite 3x3 PSD covariance derived from its geometry
and evidence. A mutually compatible contributor set combines through
covariance intersection. Compatibility requires both statistical agreement
and a bounded absolute metric separation against every contributor, not only
the primary. Substantially disagreeing candidates are not averaged: the best
supported candidate remains primary, one incompatible alternate is retained,
and output covariance is inflated. CV, image-projective, and anchor-hold
continuations are never measurements or fresh global-fusion evidence.

`PersonGroundState` remains the only temporal filter, physical gate,
stationary lock, reacquisition, and trail-state authority. The resolver feeds
one current `ground_footprint` measurement into that existing state; it does
not introduce another tracker or smoother. Public covariance is attached only
after the current measurement is physically accepted and is enlarged by any
resolver-to-filter displacement. Body root and semantic support surfaces are
separate future quantities, not aliases of ground footprint.

Resolver quality is not a blanket admission switch. A weak result may enter
bounded temporal consensus only when its selected current hypothesis is typed
as a floor-supported `floor_ray` or `registered_depth` observation and passes
the minimum confidence, support, and source-reliability gates. Observed ankle
contacts and registered person-mask floor support qualify directly; a weak
bbox or extrapolated-leg ray additionally requires a trusted lifecycle or
independent coherent image motion. Gravity reconstruction, body-height
estimates, and non-floor torso support cannot use this path. A cold weak state
normally requires three mutually consistent observations on one contact basis
within the bounded reacquisition gap unless one of the typed exact-ankle or
upright-body exceptions below supplies stronger authority. A single generic
weak detection never creates canonical world authority.

Temporal repetition alone does not override a strong revision-bound Scene Prior
contradiction for a cold observed-ankle ray: a mirror can preserve convincing
ankle motion without a person occupying that floor point. The narrow exception
requires two bounded, mutually consistent exact `pose:ankle_pair` samples under
one immutable lifecycle/revision/calibration binding. Every sample must have
admitted plausible contact, contact-gap ratio at most `0.05`, bbox/ankle range
disagreement at most `0.50 m`, ray-incidence sine at least `0.20`, current
detector semantic confidence, and no current no-ground contradiction. The
second current sample supplies the coordinate. Independent current registered
person-floor depth may also establish the lifecycle. Once the same lifecycle
owns accepted metric truth, observed ankles may override a sparse or stale
prior through the ordinary physical filter; the prior still never clamps or
supplies the coordinate.

Two bounded, mutually consistent current `pose:ankle_pair` floor contacts are a
narrower cold-start authority because both observed feet identify the
support line. Single-ankle or mixed single/pair runs retain the normal
three-sample rule unless the separate strong torso/silhouette motion proof is
current. One intervening lower-authority bbox/non-floor row may not replace a
pending ankle-family candidate; it consumes the one unavailable-row allowance
and a second intervening row or expired gap clears the run. The point published
on success is always the current ankle observation, never the saved candidate
or weaker intervening hypothesis.

One narrower cold-start rule covers a moving person whose ankles appear only
intermittently. It uses a separate, stronger bootstrap-motion proof rather than
ordinary coherent motion: three published observations of the complete
shoulder/hip torso point and detector silhouette must exceed the stronger
displacement threshold and agree in both direction and relative magnitude.
That proof is basis-bound and short-lived. A later merely coherent sample may
consume its remaining lifetime but does not renew its timestamp. A stale gap or
basis change clears it, its TTL expires independently, and an incoherent current
sample cannot consume it. While that proof is current, two mutually consistent
observed-ankle floor samples may seed `ground_footprint`. Only those ankle
samples seed metric state; the torso point supplies image motion provenance and
never becomes a metric or floor measurement. A wrong motion basis, stationary
or slowly drifting silhouette, or non-ankle weak source retains the normal
consensus requirement. One unavailable anatomy row contributes no motion
sample, but it also does not erase recent real observations inside the same
bounded reacquisition gap; an expired gap, malformed time, basis conflict, or
actual incoherent observation still clears the proof.

A tracker row may remain visually valid while neither pose nor object depth
contains a current floor contact. In that bounded case the detector silhouette
may contribute one low-confidence `bbox_bottom` floor-ray hypothesis, but only
for a sufficiently confident, tall/narrow person box with no seated or lying
evidence. DeepStream detector confidence and NvDCF tracker confidence are
alternate candidate-construction and continuity signals: the detector field's
documented `-0.1` tracker-frame sentinel is not treated as a low-confidence
detection. Before a lifecycle owns a queue-published metric point, however, a
selected `floor_ray` requires current detector semantic confidence for final
admission and ray-incidence sine of at least `0.20`; NvDCF confidence cannot
establish that the box is a person, and neutral/unobserved PCF coverage cannot
make shallow floor geometry authoritative.
This hypothesis cannot establish learned height, bypass range or PCF evidence,
or skip `PersonGroundState` physical admission. It is a continuity measurement
candidate, not a dashboard-only fabricated dot. Its minimum resolved height is
expressed as a calibration-raster fraction (48/1080), not a camera-resolution-
specific pixel constant. A pose floor contact materially below the detector
silhouette is rejected before floor projection and cannot contaminate the
upright reference.

Pose torso evidence has two deliberately separate, non-floor roles. The
image-only motion point requires both shoulders and both hips with bounded
in-box geometry and corroborates detector-silhouette motion only. The metric
range path samples one trimmed torso-core capsule through a bounded native
statistic. Torso range may become a registered-depth hypothesis only for an
already trusted lifecycle with exact-current depth, sufficient person
confidence, and seated, lying, or active lower-body-occlusion context; ordinary
upright/dropout rows remain excluded. A lying body projection is typed as
`couch`, not as visible floor contact. Torso range is never verified
ground contact, cannot seed the weak floor-consensus path or learned height,
and cannot suppress an independently eligible `bbox_bottom` floor ray. Floor
support remains the semantic authority tier for standing and unknown posture.
For explicit sitting or lying posture, a typed exact-current body footprint
(`seat` or `couch`) ranks ahead of an ankle/floor ray because a visible ankle
may belong to a tucked leg, furniture edge, or overlapping person rather than
the non-upright person's support location. Untyped body/torso range still ranks
behind floor support. Covariance intersection remains limited to equal support
states, so body and foot geometry are never averaged into a plausible-looking
compromise.

Hidden physical contact does not make a visible person unlocalizable. A typed
`pose_scale` ground-footprint lane projects exact-current body observations
through calibrated anatomical height planes. The upright solve uses nose,
shoulders, and hips, estimates one common body height and X/Z footprint, and
never uses the furniture-clipped detector bottom. One-frame strong proof
requires five retained planes spanning head, shoulder, and hip bands; a
four-plane/two-band solve remains weak even with a small residual because it can
agree at the wrong range. Complete side profiles are valid despite apparent
left/right width collapse, while a three-joint partial torso retains the width
guard needed for safe midpoint inference. Valid moderate three/four-plane or
noisier rows may advance the bounded same-basis reacquisition trajectory. One
current five-plane strong proof may seed a cold lifecycle immediately; one
three- or four-plane row cannot. As a cold-start-only exception, the second of
two compatible exact-current four-plane/two-band solves may establish the first
coordinate when both share the lifecycle/body-plane basis, have a positive gap
no greater than `reacquire_max_gap_s`, and pass the ordinary physical-step
bound. The first solve is retained corroboration only, and unavailable rows do
not renew its timestamp. Three-plane, mixed, incompatible, expired, and
basis-changing evidence cannot form this proof. Once a lifecycle owns
queue-published metric truth, moderate body rows may accumulate relocation
evidence, but only a current verified five-plane row can satisfy body
reacquisition support and finalize the reanchor.
When that gate quarantines a body solve, an independently valid queue-rooted
image-motion continuation may keep the existing dot visible; the projective
integrator preserves but never increments the pending metric consensus.

The seated form projects an exact torso observation to the support footprint
only from lifecycle-local geometry: exact-current registered range measures the
same pose-torso anchor. The prior standing footprint is never used to derive a
seated plane because that circular construction would reproduce the old
standing coordinate by definition. Population body heights, fixed
seated-height fractions, and room/camera offsets are forbidden as canonical
position authority. Until registered depth establishes the physical reference,
the pose-only seated candidate is unavailable. It is typed as seat support
rather than visible foot contact and ranks ahead of simultaneous ankle-floor
evidence. The lying form uses only an exact-current registered body projection,
typed as couch support. Both may use
stationary hold when current geometry is temporarily absent and do not
disappear merely because ankles are hidden. Trusted non-upright body updates
use a tight 0.10 m lock deadzone and a minimum 0.30 posterior gain so an old
foot-derived coordinate converges to the body footprint without retaining the
broader seated-jitter deadzone. Existing physical innovation, output-speed,
and three-sample reanchor gates remain authoritative. Exact registered body
rows and pose rows projected from their measured plane share that one
reacquisition family, so intermittent DAv2 cadence cannot strand a seated dot
at its prior standing footprint. Upright body
projection and learned-height gravity reconstruction are alternative uses of
the same current body evidence and are never co-fused on one row. Neither rule
depends on camera or room identity.

Exact-cohort evidence remains independent through resolution. A stale or
cached depth branch is unavailable for metric authority, but in universal
resolver mode it cannot clear an independently selected exact-current pose
floor observation or that observation's same-basis reacquisition consensus.
Only the final post-resolver absence or rejection may mark the current metric
state unavailable.

Revision-matched PCF contributes soft extent, authored-boundary,
observed-confidence, and floor-height evidence. It never clamps or snaps a
track. A strong authored-boundary or measured-extent contradiction keeps the
candidate unchanged for diagnostics but makes it weak before temporal
admission; two agreeing candidates cannot reinforce the same contradiction
into an accepted fusion. Unlabeled obstacle/furniture evidence is diagnostic
only. BEV, OSD, the dashboard, canonical world service, and later Menon views
consume the same filtered revision-bound world point. The normal dashboard's
off-by-default `Localization details` overlay may draw exact-cohort candidates,
uncertainty, and disagreement, but cannot affect the dot or trail.

Every canonical spatial handoff preserves tracker lifecycle, target-frame
revision, and source-to-world transform SHA-256. BEV fails closed on any
mismatch. The global service refuses incomplete or mixed target revisions and
uses full-matrix covariance intersection for compatible fresh cameras. It never
promotes `cv_prediction`, `image_motion_prediction`, or `anchor_hold` into a
fresh global measurement. Strictly proven state-integrated continuations may
replace a single-camera held canonical coordinate for exact downstream display
parity, but cannot steer cross-camera fusion or its velocity baseline.

The exact contract and practical tests are specified in
[`universal_world_localization.md`](universal_world_localization.md).

**Why:** The prior path calculated several useful geometric signals but
collapsed them through different per-room policies before the canonical world
layer could compare their support or uncertainty. Some outputs labeled
"fused" numerically used only depth X/Z. Preserving typed hypotheses and
honest covariance connects the existing camera-local human estimator to the
existing canonical world service without duplicating PersonGroundState or
making a depth model the organizing authority.

## ADR-025 — Reconstructable world persistence is isolated from spatial authority

**Accepted:** 2026-08-25

The exact tracking, world-snapshot/event, and BEV cohort remains ordered and
release-gated on its in-memory canonical world commit. The world journal is
reconstructable retention, not localization authority. Runtime therefore
admits the exact already-validated journal payload tuple to one finite
`AsyncContractJournal`; its receipt proves the queued payload count, not
durability. The worker batches writes into the existing private,
integrity-chained synchronous SQLite journal and exposes pending, capacity,
worker, and failure health.

Queue saturation or durable-write failure degrades only persistence and cannot
wait on, mutate, abort, or poison the media or canonical spatial-publication
path. An authority commit or WebSocket release-gate failure remains fail-closed
for the whole exact cohort. Orderly shutdown drains the persistence worker and
then closes the durable journal.

**Why:** Paced occupied replay showed that durable SQLite work on the exact
publication worker creates backlog even after query-level optimization. The
journal can be reconstructed from canonical publications; making live tracking
wait for disk violates the accepted hot-path boundary without improving the
world estimate.

## ADR-026 — Static-camera PCF map lock is a revision-bound frame edge

**Accepted:** 2026-08-27

A residual found by matching a calibrated static-camera recording to its PCF
belongs at the camera-to-PCF frame boundary. It does not rewrite the shared raw
camera-calibration bundle, rotate only the dashboard raster, or introduce a
room-selected localization algorithm. The Scene Prior builder may compose one
explicit, content-hashed yaw residual around the camera optical center in the
target world frame. The resulting immutable prior records the base and corrected
camera axes, pivot, residual, and evidence fingerprint; its frame-transform
digest is consumed by the same world estimator, PCF rasterizer, BEV, and later
3D consumers.

The current Family Room target-world correction is `+18.25` degrees. The raw
image/BEV edge-fit residual is `-18.25` degrees and is sign-inverted when
expressed as target-world yaw, so the visible PCF center moves counter-clockwise
from the uncorrected camera center. The correction is bound to the exact
camera-calibration digest, static reconstruction revision/metadata digest, and
recorded-clip evidence in
`config/scene_prior_camera_map_lock_family_room.json`. Kitchen and Living Room
bindings remain byte-for-byte unchanged. A stale calibration or reconstruction
causes the builder to fail closed instead of carrying the residual forward.

Dashboard camera-local presentation composes that same revision-bound edge:
camera center/right/forward are transformed by
`target_from_calibration_col_major` after inverting raw camera-from-calibration
`E`, then right/forward are projected onto the target-world ground plane. Raw
pitched camera-space Z is not a floor-display coordinate because camera height
and pitch introduce a false forward offset. Missing or invalid frame bindings
fail closed. Kitchen requires no residual map lock; its active optical center
is approximately 0.10 m inside the nearest authored floor boundary, so that
small visible wall offset is expected geometry rather than an extrinsics error.

**Why:** The Family PCF/video comparison exposed an azimuth error after floor
leveling was already correct. Rotating around the camera center corrects the
ray-to-map relationship without moving the camera or floor, preserves the raw
multi-camera/depth identities, and makes the correction reproducible rather
than a presentation-only offset.

## ADR-027 — A full static-camera PCF anchor may define a Scene Prior frame edge

**Accepted:** 2026-08-31

When a revision-matched static camera can be localized repeatedly against an
accepted metric PCF walk, the Scene Prior builder may use that evidence as the
complete camera-to-PCF pose. The anchor must retain metric scale and gravity,
use independent early and late walk views, reject inconsistent per-view PnP
poses, and verify a repeatable PCF floor. The resulting frame edge is derived
as `camera_to_pcf @ camera_from_calibration`; it is consumed unchanged by the
world estimator and BEV rather than being recreated as a dashboard offset.

This is an optional evidence path, not a room-selected localization algorithm.
Camera intrinsics remain sourced from the canonical calibration bundle. The
anchor does not preserve an old measured mount height, add a camera-specific
depth curve, or modify PCF geometry. A Scene Prior may use either the existing
yaw-only map lock or a full PCF pose anchor, never both. Bindings without a
full-pose anchor retain their existing behavior.

The Living Room anchor uses 17 consistent views spanning both ends of its
48-view walk and 1,799 PnP inliers. Its PCF floor fit has 759 independent cells,
0.518 degrees residual tilt, and 0.032 m p90 residual. The resulting camera
height over the PCF floor is 2.058 m; it replaces the incorrect 1.934 m
low-envelope result without using physical room measurements.

**Why:** A full camera pose and a metric PCF floor jointly determine where
image rays meet the room. Preserving a known-bad legacy height or correcting
only yaw can make the projected forward coordinate asymptote before the person
reaches the foyer. The revision-bound SE(3) edge fixes that geometry at its
source while remaining reproducible for future homes.

## ADR-028 — Reconstruction evidence and presentation remain revision-bound

**Accepted:** 2026-09-04

The reconstruction lanes keep physical metric-frame identity, source
calibration and world-alignment provenance, scene-prior artifact revision,
target-coordinate revision, and authored world-to-scene presentation identity
as separate contracts. A validated target-owned presentation is applied once
at the Noesis-to-Menon boundary, so a fused entity may carry several accepted
source-edge hashes while rendering through one target mapping. Menon validates
the mapping digest against its matrix and frame/scene identities at bundle
ingress; a revision, source hash, or presentation mismatch fails closed. The
existing v1 binding path remains compatible and the current catalog is not
relabelled to claim a new common frame.

Phone reconstruction evidence is bounded and provenance-preserving. Sensor
capture retains exact camera and IMU timing/calibration fields; conventional
VIO supplies relative same-segment constraints and does not establish the
Noesis house origin. Trajectory refinement uses explicit visual and
depth-backed constraints with withheld temporal checks. Conditioned
MapAnything and DA3 evidence is treated as correlated, so agreement does not
receive an independence bonus; single-view support is retained as a separate
review class.

A physical scale change must reach depth, poses, world points and relative
visual constraints together. Larger changes rerun the existing fixed-scale
world alignment on the actual refined carrier before PCF conditioning; a
matrix or passed report for earlier geometry cannot authorize reuse. Candidate
artifacts remain at stable paths, with acceptance controlling consumption
instead of a separate promotion lifecycle.

Multi-room PCF assembly remains review-only until its registration and metric
frame evidence is accepted. X/Z ownership is only an overlap candidate;
compatibility requires recorded camera pose, full intrinsics, retained mask,
and depth-ray support in three dimensions. Complementary heights are retained,
contradictory or unsupported surfaces are marked uncertain or unknown and
rendered in separate review classes. Rejected connector joins, manual
extrinsics, authored geometry, and a static model's apparent scale cannot
promote a reconstruction or place live people.

The implementation and bounded evidence are recorded in
[`WO-1`](../plans/reconstruction_work_orders/WO-1.md),
[`WO-2`](../plans/reconstruction_work_orders/WO-2.md),
[`WO-3`](../plans/reconstruction_work_orders/WO-3.md),
[`WO-4`](../plans/reconstruction_work_orders/WO-4.md), and
[`WO-5`](../plans/reconstruction_work_orders/WO-5.md).

**Why:** The five work orders share one authority boundary but produce
different kinds of evidence. Keeping those identities and evidence classes
explicit prevents an accepted rendering mapping from becoming an unvalidated
registration, and prevents review geometry or relative phone motion from
silently becoming live-world authority.

## ADR-029 — Room Walk uses the Menon-trusted HTTPS origin

**Accepted:** 2026-09-05

The phone-scan launcher keeps its existing LAN HTTP listener on port 8788 for
known consumers and serves the same FastAPI app through a second HTTPS listener
on port 8789. The HTTPS listener reuses the existing Menon appliance
certificate and private key, with paths configurable through
`NOESIS_PHONE_SCAN_TLS_CERT_FILE` and `NOESIS_PHONE_SCAN_TLS_KEY_FILE`. The
two Uvicorn servers share one imported app instance and run inside one explicit
FastAPI lifespan owned by the launcher. Both listener configs disable their own
lifespan callbacks, so startup recovery and shutdown do not create duplicate
`PhoneScanService` executors or state-recovery passes. The HTTPS listener is
exposed only after shared startup recovery completes, and both listeners drain
before shared shutdown runs.

The current certificate is valid for the DNS name
`TauntonMainframe.local` and has no LAN-IP SAN. Browser Room Walk access
therefore uses `https://TauntonMainframe.local:8789`; opening the raw
`192.168.3.126` address does not satisfy certificate validation. The native
Android companion routes that hostname directly to the fixed `192.168.3.126`
LAN address while retaining the hostname for TLS/SNI and HTTP Host. Android Chrome must trust the existing
Menon local CA certificate before this is a trusted secure origin. When the CA
is not already trusted, the bounded `GET /api/browser-capture/ca-certificate`
route on HTTP 8788 serves only the exact configured public CA certificate for
bootstrap; the launcher prints its DER SHA-256 fingerprint for out-of-band
verification. A warning page or certificate bypass is not accepted as trust
evidence, and the CA private key is never exposed through the phone-scan static
or asset roots.

The browser's in-page camera + IMU recording is the default path for collecting
raw, uncalibrated observations. Browser callback arrival timestamps are not
acquisition timestamps and cannot admit metric VIO or establish the Noesis
world frame. The calibrated native camera + IMU bundle remains the metric-VIO
path; HTTPS transport changes origin security only and does not change spatial
authority.

**Timing diagnostic correction, 2026-09-06:** Generic Sensor timestamps do not
inherit a verified browser clock origin from their API name. Receipt time has
separate provenance. A declared common origin contradicted by sensor timestamps
after receipt disables callback-lag statistics while retaining the raw samples.
The retained Living Room walk exhibits this contradiction. Receipt-time clock
fits cannot supply camera/IMU acquisition calibration; use the native
[timing preflight](../tools/mapanything_phone_scan/README.md#native-timing-preflight-before-a-new-room-walk)
before collecting calibration motion or another metric-VIO candidate.

**Why:** Android Chrome requires a secure, trusted origin for camera and motion
sensor access. Reusing the already trusted appliance certificate avoids a new
certificate authority and preserves the existing HTTP consumer while keeping
one phone-scan service state and one recovery/shutdown lifecycle.

## ADR-030 — Room Walk may retain one bounded static-camera companion session

**Accepted:** 2026-09-05

The Room Walk browser may pair one selected physical static camera with one
phone capture for evidence. The companion resolves that camera through the
active native DS9.1 source configuration, records the original encoded stream,
and observes the selected camera's canonical tracking/world publications. The
phone side is allowed to start only after both the encoded-video and tracking
ready gates pass. Start, heartbeat, optional marker, stop, finalization, and
phone-bundle association are retry-safe; one session is active at a time, its
duration is capped at 900 seconds, and its renewable lease is capped at 45
seconds.

The session persists source and calibration/runtime provenance, exact tracking
and world publication identities, packet timing, bounded browser/server clock
exchanges, and optional user markers. It retains partial and failed evidence,
and a failed phone upload does not invalidate finalized static evidence. The
browser exposes links to the saved artifacts after stop. The original encoded
video is remuxed without a decoded CPU branch, inference, or re-encoding. The
companion does not become a second perception authority or add work to canonical
media callbacks.

The paired static data remains separate from phone-only frame selection and
provider fusion. Browser callback timing is estimated/unverified: clock probes
are evidence of request/receipt timing, not synchronized acquisition time. A
phone camera trajectory is not a person's body or foot trajectory, so paired
capture does not establish calibration ground truth or change live tracking
calibration without separate identity, camera-to-body offset, scale, and timing
evidence.

**Why:** A bounded companion session preserves the fixed-camera video and
canonical tracking context needed for later comparison while keeping source
authority, timing uncertainty, and phone-only reconstruction responsibilities
explicit.


## 2026-09-05 — Scope agent workflows to the authorized application task

**Decision:** Existing-runtime diagnosis preserves selected pipeline quality and
outputs and uses bounded measurements. New graph generation and new detector
imports have separate skill routes; existing-engine and non-detector maintenance
follow the native host contract. Generator questions resolve checkout and host
facts first. Import reports are conditional; vendor encoder fallbacks are not
an execution route. AMC and SOP skill selection is explicit-only, without lifting
AMC deferment or the native-runtime boundary.

Audits remain read-only until implementation is requested. Independent
investigations may use up to two subagents, enforced by project Codex config;
shared-file edits and runtime operations have one owner. This is optional
investigation parallelism, not an independent release-review requirement.

**Why:** Separate task modes prevent general vendor workflows from widening a
bounded repair or overriding the canonical application contract. Focused docs
checks now include the detector importer, generator, and requirement reference.

## ADR-031 — Phone frame selection preserves gaps and removes adjacent repeats

**Accepted:** 2026-09-06

Phone-only preparation samples existing encoded frames instead of using an FPS
filter that manufactures repeated images during recording gaps. Native capture
bundles retain their explicit acquisition timestamp and source-frame mapping;
browser and ordinary video PTS remain encoded-media timing only.

A bounded pass collapses adjacent repeated-image runs before adaptive selection.
Exact decoded-content equality is sufficient evidence; near-identical images
also require spatial feature support and very small displacement. Each comparison
uses the fixed run anchor, preserving cumulative slow motion. A representative
retains its original timestamp and candidate identity. Preparation reports the
removed candidate identities, counts, and repeat spans.

Temporal coverage, endpoint selection, and connectivity repair cannot force
repeats back into the selected set. Real timestamp gaps may remain after a pause.
They do not authorize fabricated camera poses, interpolation across missing
observations, or weaker camera/dense-surface registration gates. A preview stall
is not proof of a recording stall, and repeated-image evidence alone does not
establish the cause of a reconstruction failure.

**Why:** Redundant images consume bounded joint-inference capacity without adding
viewpoint evidence. Preserving recording gaps and distinct views makes brief
pauses tolerable while retaining the existing reconstruction authority checks.

## ADR-032 — Reviewed image orientation corrections retain source provenance

**Accepted:** 2026-09-06

A capture-specific prepared-image rotation must retain the original encoded
recording, selected observation indices, and timestamps. Corrected frames record
their original prepared-frame identity/hash, original dimensions, explicit
source-to-prepared pixel transform, and new content hash/identity. Rotating RGB
pixels does not verify camera acquisition timing, rotate raw IMU measurements,
or establish a camera-to-IMU or Noesis-world frame binding.

For `20260906-050646-107f244c`, visual inspection confirmed sideways pixels with
no encoded display-rotation metadata. A lossless 90-degree counterclockwise
correction was applied to the same 256 selected observations. The matched rerun
passed all existing camera and dense-surface window gates at the same model
pixel budget. Prior failed preparations and outputs remain available.

This is a reviewed correction of that capture; no automatic orientation rule
was added. Portrait dimensions alone are not evidence that pixels are sideways.
Repreparing from the original recording must account for the recorded correction
before inference. Native calibration or pose/depth priors require their own
consistent frame transformations if a future workflow rotates their RGB views.

**Why:** Correcting the observed input orientation recovered inference without
discarding observations or weakening registration checks, while explicit
provenance keeps the correction separate from calibration and world authority.

## ADR-033 — Paired static recordings supply independent alignment references

**Accepted:** 2026-09-06

A Room Walk with an associated, successfully finalized static recording uses
that recording to build its alignment reference. Its camera and phone-archive
association must match. An invalid paired source or failed reference build
stops the operation. Unpaired walks retain the validated saved-camera route.

Static reconstruction consumes bounded observations from the recorded camera,
the captured rectification, and the calibration bundle. It does not consume
phone RGB, depth, or inferred poses. Same-camera depth estimates are fused
using the existing static depth-agreement rule. Rectified camera rays enter the
captured target frame through `target_from_calibration * inverse(E)` exactly
once. No additional floor, yaw, or scale correction is inferred from the phone
walk, and tracking observations are not calibration authority.

MapAnything receives the known rectified intrinsics as conditioning, but its
output rays and recovered intrinsics remain predictions. The builder places
its metric range (`depth_along_ray`) on the captured calibrated rays, converts
that range to Z depth for static fusion, and retains the predicted intrinsics
and original model depths as diagnostics. Learned intrinsics cannot replace
the captured calibration.

The static reference has its own content-bound revision and artifact manifest.
It preserves the source recording identity, sampled observations, image
transforms, calibration and frame-binding hashes, and model settings. The
alignment report retains the captured world-frame revision. Reuse verifies
the reference artifacts; downstream PCF resolves the same verified reference.
Neither reconstruction nor a passed alignment promotes geometry into live
Noesis tracking or rewrites global calibration.

Static inference shares the bounded phone/PCF inference worker lock. Failure
preserves the original capture and phone reconstruction. Completed static
reference artifacts remain reviewable even when the unchanged alignment
quality gates reject the geometric fit.

**Why:** A contemporaneous static recording provides an independent reference
for the same room state, while explicit source and frame provenance prevents
older geometry or live tracking estimates from silently entering the fit.

## ADR-034 — Fit only static-camera-visible structure and retain replay diagnostics

**Accepted:** 2026-09-06

Phone-to-static refinement and candidate scoring use the same fixed-camera
visibility and occlusion model as their validation. Visibility is recomputed
at each bounded optimizer iteration. Foreground disagreements remain eligible;
points behind the camera's nearest supported depth do not pull the fit toward
unobserved geometry. Gravity, fixed scale, transform bounds, and existing
quality thresholds remain unchanged.

The global point-cloud and structure metrics remain separate diagnostics. The
report identifies the existing robust wall-gate statistic and additionally
records untrimmed residuals across comparable points. Relative fit admission
does not establish surveyed metric accuracy or remove a reference-model bias.

The explicit diagnostic replay command writes outside the saved scan, retains
failure reasons and optional bounded intermediate arrays, and never updates
scan state or live-world authority. The room-specific source and target
identities remain in each report. The working procedure and experiment ledger
are in [Room reconstruction fitting](room_reconstruction_fitting.md).

**Why:** On matched Living Room inputs, the previous all-structure refinement
moved an already plausible RGB anchor to a rejected wall fit. Visibility-aware
refinement reduced the gated residual from 0.122825 m to 0.085703 m without
relaxing a gate; spatial holdout checks also improved over baseline. The same
implementation reduced the matched Kitchen result from 0.081716 m to
0.054205 m. Image-cell balancing was tested and rejected. Family Room's old
reference/calibration mismatch remains a separate preflight failure.

## ADR-035 — Consensus preserves one RGB projection for visual anchoring

**Accepted:** 2026-09-06

Consensus raw views retain the MapAnything reference RGB image remapped onto
the existing common camera rays. They do not average RGB images warped using
different inferred intrinsics. The fusion manifest records the RGB projection
source and policy. Depth selection, uncertainty, camera poses, metric scale,
and fusion thresholds remain unchanged.

**Why:** Averaging two different projections of one recorded image duplicates
edges and weakens the landmarks used by static-camera PnP. On the matched
256-view Living Room consensus, replacing only RGB restored a passed anchor
with 31 consensus views and 1,940 inliers; the blended images supported only
two solved views and 21 inliers. Better visual anchoring does not itself admit
the room fit: the usual independent static-reference gates still apply.

## ADR-036 — Retained trajectory refinement binds RGB rays and rebuilds fusion support

**Accepted:** 2026-09-06

Visual revisit verification matches the RGB stored on the retained depth and
intrinsics grid. Prepared frames provide capture identity; their resized pixels
cannot replace a cropped or remapped raw projection. Refinement preserves the
input coordinate-frame label and source hashes, requires withheld temporal
constraints before admission, and rebuilds camera poses and world points
without forcing endpoint closure.

The joint consensus builder accepts an explicitly passed pose-only refinement
of the same capture and original consensus. It verifies source and solution
hashes, original poses, camera rays, scale, and origin gauge, then recomputes
multiview consistency, depth selection, evidence weights, raw geometry, and
actual distinct-view surfel support. A previous surfel cloud cannot be reused
merely by changing its pose metadata. A fresh fixed-scale static alignment is
required before the rebuilt result is used as a room review surface.

Complete static-camera localization uses the same retained-RGB/depth projection
contract. Insufficient independent-view support remains a hard rejection and
now writes bounded per-view diagnostic evidence before raising the failure.
Its fitted poses and comparison calibration must share the explicitly bound
world frame. A pose request or a saved failure report cannot admit calibration.

**Why:** The 256-view Living Room consensus has a materially different RGB
projection from a resized prepared frame. Using the retained projection finds
two admitted revisit constraints and eight withheld constraints; their p80
translation residual improves from 0.110095 to 0.098891 m. The static fit is a
separate check and slightly worsens before fusion is rebuilt. This is a measured
tradeoff, not evidence of surveyed accuracy or permission to relax either gate.


## ADR-037 — Phone intrinsics bind the capture projection and provider geometry

**Accepted:** 2026-09-07

A measured phone calibration is retained with its original archive, verified
source hashes, native image coordinates, and all five OpenCV distortion
coefficients. Applying it requires an explicitly associated recorder mode and
matching native encoded resolution/orientation. Preparation rectifies selected
images once and binds their final hashes to the measured output K, source K/D,
resize transform, and calibration profile hash. Existing saved walks retain
their original preparations and results.

MapAnything consumes calibrated rays through its own coupled image/intrinsics
preprocessor. DA3-BASE's installed network does not consume K without extrinsic
inputs, so its inferred poses/depth remain model estimates. Its exact input
processor places measured K on the output grid for metric focal conversion and
backprojection; network-estimated K remains separate diagnostic evidence.
Both providers reject invalid rectification-border rays, preserve calibration
through window assembly, and retain existing reconstruction acceptance gates.

Consensus also checks the prepared profile fingerprint and both providers'
per-view calibration and source-processing lineage before fusion. It preserves
that lineage and the profile's evidence/capture-binding limits in the fused raw
views and manifests, and recomputes D5 border validity on the common rays.
Reported sensor metadata applied through an assumed recording crop remains an
explicit review hypothesis; fusion cannot upgrade it to measured browser
calibration or metric VIO.

**Why:** The supplied 8K handoff has a nonzero fifth distortion coefficient.
Truncating it or treating its native K as the camera matrix for the 4K/browser
recordings would change the projection without evidence. Camera intrinsics
alone cannot establish synchronized IMU data, camera/IMU extrinsics, surveyed
room scale, or a static/world camera pose. The operational import and use steps
are in the [phone reconstruction guide](../tools/mapanything_phone_scan/README.md#measured-phone-camera-calibration).

## 2026-09-06 — Keep agent policy local to implementation responsibilities

**Decision:** Root agent guidance owns repository invariants, task scope, branch
rules, and completion criteria. Subtree files inherit those rules and add local
implementation constraints and task-specific reading. Core world contracts,
identity, and phone-walk reconstruction now have scoped entrypoints beside their
implementation; the root also routes cross-boundary edits to those entrypoints.
Identity plan guidance retains acceptance work without automatically loading an
entire workstream checklist for an ordinary edit.

DS9 smoke requirements use the root's practicality and documentation exceptions.
Shared publication guidance names the exact tracking/world/BEV cohort separately
from optional depth/diagnostic freshness. Artifact guidance distinguishes the
checked-in manifest from realization under the selected external artifact root.

**Why:** Repeated policy wording had diverged, and specialized rules under a
sibling planning directory were not a reliable entrypoint for implementation
work. Relative Markdown links and docs-check coverage now include all ten active
repository instruction files. These changes preserve native runtime, evidence,
identity, geometry, and authority-admission requirements.

## 2026-09-07 — Operational guidance follows evidence and authorized phases

Generator delivery reports actual failed/skipped checks instead of formatting
an incomplete graph as validated. New detector imports derive engine profiles
from the consumer contract, keep capacity experiments bounded and optional, and
validate detections against known positive/negative fixture expectations rather
than an assumed occupancy fraction. Utility environments require only the
packages their selected phase uses.

PCF candidate generation/evaluation ends with review-only evidence. Sealing,
authored Scene Prior binding and activation retain their existing gates in the
separately authorized later phases. Profiling starts with existing-pipeline
diagnosis; new-graph construction has its own reference and preserves selected
quality, ordered publication and required outputs.

Indexed documentation distinguishes checked-out wiring, configured conditional
identity permits, accepted contracts and dated observations. Section navigation
routes to existing definitions without duplicating them in instruction files.
These are workflow/documentation changes, not runtime activation or new model,
geometry, identity or quality authority.

## 2026-09-07 — Require explicit browser 8K geometry and preserve timing limits

New browser phone captures require exact unscaled rear-camera 8K negotiation,
returned-settings checks, and an encoded-dimension check on import. The UI
reports the bounded recording duration and unverified synchronization. Invalid
Generic Sensor clocks stop capture instead of being corrected using callback
arrival offsets; raw observations remain evidence. Preview callback phases do
not become encoded-frame IDs. Legacy imports retain their existing raw-evidence
semantics. Matching dimensions do not transfer native-camera calibration or
admit metric VIO. Synchronized acquisition continues through the native bundle
boundary, with device-specific 8K and timing validation still required.

## 2026-09-08 — Native RoomWalk Android capture companion

RoomWalk keeps its reconstruction service and web review interface. The native
[Android companion](../tools/mapanything_phone_scan/android_companion/README.md)
records Camera2 video through a hardware MediaCodec surface and preserves
separate Android IMU streams. It requires explicit 8K/30 capability checks and
Android's SENSOR output timestamp base with a REALTIME camera clock. Actual
encoder PTS must match Camera2 sensor timestamps exactly at encoder precision;
raw acquisition, encoder and container timing remain separate evidence.

A static Camera2 size-list omission is not sufficient to reject a mode when
the driver supports an exact configuration query. Positive standard 8K/30
queries may enable a bounded recording test. Each attempt revalidates the
prerequisites and queries the actual encoder/preview surfaces before opening
the camera, then creates the session from that same configuration and request
parameters. Vendor use cases remain diagnostic-only. Query acceptance alone
does not enable full walks for a newly found mode or establish native image
detail, capture cadence or timestamp association; those require recorded evidence.

A completed local ten-second test can qualify the matching OS build, camera,
encoder/bitrate, preview and standard use cases for full walks. Qualification
requires exact complete frame associations, IMU coverage, no capture failures
or dropped metadata, and a sufficient recorded span at nominal 30 FPS, as defined
in the companion's capture preflight. Each new take rechecks the saved evidence
and queries its actual surfaces. This permits longer recording without granting
camera calibration, metric VIO or native resolving-detail authority. Viewfinder
rotation and aspect correction affect presentation only; encoded coordinates stay
unchanged.

The native companion's Capture action opens a full-screen viewer with a live
preview before recording, following the web capture sequence. Its separate
preview-only camera owner writes no capture artifacts and starts no IMU or
encoder work. Recording begins only after that owner confirms camera closure;
the canonical native recorder then opens its independently checked 8K session.
Setup and transfer menus stay outside the viewer, and Back/close during a take
uses the existing stop-and-save path. This UI transition does not change video
coordinates, camera/IMU timing semantics or calibration admission.

The service validates those original rows, decoded frame counts and MP4 timing
before labeling camera acquisition timestamps verified. A missing or failed
association leaves RGB preparation available with no acquisition timestamp
claim. Unknown lens calibration, distortion, camera/IMU extrinsics, measured
time offset and IMU noise remain unknown; metric VIO stays blocked. Existing
stock-camera calibration is not transferred by matching 8K dimensions.

Captures stream to phone storage with bounded workers and size/duration limits.
Backgrounding stops capture; export and explicit HTTPS upload retain local
recordings. The app preserves certificate and hostname checks, packages no
private keys, and does not start the static-camera runtime. This first companion
build uses independently selected static reconstruction for later alignment;
it does not orchestrate a paired static recording. APK installation/emulator
checks and importer tests establish software behavior. Physical-phone 8K
performance, firmware timing and camera calibration remain separate checks.

## 2026-09-10 — Complete native phone calibration inputs without changing authority

Stationary IMU acquisition is separate from 8K video capture. The companion
uses the same native sensor-selection policy for both paths and retains raw
accelerometer/gyroscope values, separate clocks, sensor identity and vendor bias
estimates. Duration, queue, row and storage limits produce explicit partial
recordings. A dedicated bounded receiver verifies the archive and retains it
outside room scans; acquisition checks and upload success do not constitute an
IMU noise fit or metric-VIO admission.

An explicit focus lock uses the actual reported lens setting and applies it
consistently to preview and video for the selected camera. Calibration remains
bound to the matching native lens, crop, zoom, focus, encoded geometry and
stabilization evidence. The existing stock-app camera profile is not admitted
for native capture by matching dimensions alone. Camera/IMU extrinsics, measured
time offset and noise still require physical calibration and withheld validation.

The fresh-walk workflow below supersedes a long stationary acquisition or new
target session as a capture prerequisite. This paragraph describes metric-VIO
admission, not permission to record and reconstruct a normal walk.

Five-coefficient Brown-Conrady calibration is supported by rectifying dense
OpenVINS input with the full coefficient vector and unchanged K and dimensions.
Invalid border pixels are excluded with a hashed static mask supplied to the
native `CameraData` consumer; the adapter requires the bridge's confirmation.
Unsupported models remain rejected. Android full-calibration admission is
recomputed after its exact encoded-frame timestamp associations and actual
OIS/EIS evidence are verified. This corrects the stale pre-verification
admission flag without relaxing any missing-calibration or timing requirement.

## 2026-09-10 — One native room walk retains phone video, IMU and static evidence

The native companion coordinates the existing paired-capture service: select
the room camera, wait for original encoded static video and canonical tracking,
then acquire Camera2 video and both native IMU streams. Bounded heartbeats and
clock exchanges accompany the take. Stop closes both captures; the saved phone
archive carries an immutable session/camera/capture reference so retries cannot
silently attach another static recording. A failed session remains retained and
explicitly incomplete. Host correlation probes do not establish cross-device
acquisition synchronization.

The companion uses IPv4 exclusively for its LAN HTTPS connections. The configured
`TauntonMainframe.local` appliance origin routes directly to the fixed
`192.168.3.126` address without DNS; other origins use a cancellable DNS
A-record query within one second. IPv6 addresses are never attempted. The
original hostname remains the TLS certificate/SNI and HTTP Host authority and
the saved server origin. The trusted TLS factory completes a verified handshake
before Android can replace the hostname with the numeric route. Host-specific
bindings remain bounded and include the trust-factory identity. An IP route
cannot admit a certificate for another hostname.

The complete initial health check has a three-second deadline and establishes
the route before camera inventory or pre-recording clock probes. TCP setup is
limited to 750 milliseconds. The phone service keeps idle HTTP connections for
30 seconds, covering the ten-second paired heartbeat cadence. Clock probes and
capture leases retain their existing authority and bounds; LAN reachability
does not require a long setup allowance or serial IPv6 failures.

Ordinary frame preparation now consumes the native IMU magnitudes as a bounded
soft motion preference alongside visual quality and overlap. Exact frame IDs,
exposure times, raw-evidence hashes, motion values and score adjustments remain
in the prepared manifest. This use requires verified native acquisition clocks,
units, axes and covered timing neighborhoods, but needs neither camera/IMU
extrinsics nor gravity subtraction. It creates no metric pose authority.
MapAnything + DA3 continue to use the same phone-view set; the paired static
recording is reconstructed independently for alignment and validation.

Long stationary sensor acquisition and a new target board are not normal capture
prerequisites. Short sensor diagnostics are optional. Metric VIO retains its
calibration and timing gates; provisional server-side online-calibration
experiments remain distinct from admitted poses and do not block a fresh walk.

## 2026-09-12 — Native LAN transfer is separate from capture and import

RoomWalk's main Android interface uses portrait orientation. The camera viewer
requests landscape before opening its preview and returns to portrait after
closing. The Activity handles these configuration changes so rotation does not
destroy the camera coordinator or selected capture. Recording remains a
foreground-only operation; leaving it still stops and saves the take.

Saved-archive uploads instead belong to a single `dataSync` foreground service,
with a bounded partial CPU wake lock, finite cancellation/watchdog work and
durable progress, failure and receipt state. The user can turn off the screen or
leave the Activity. An interrupted transfer requires an explicit retry of the
same retained ZIP. TLS still verifies the original hostname; the configured
static IPv4 route and capture quality/timing contracts remain unchanged.

Native uploads may opt into `Prefer: respond-async` on the existing sensor-bundle
endpoint. HTTP 202 confirms only that the server has durably stored the complete
archive, calculated its SHA-256 and checked the bounded manifest and requested
capture/session/camera identities. The client checks the receipt against the
actual streamed bytes. It does not interpret this transport receipt as final
companion association, decoded-frame validation or metric admission.

One server worker performs full import, with at most two pending imports. It
retains the archive and failure status if import fails or is interrupted. Exact
archive retries resolve to the same scan; a different archive cannot replace a
paired capture. The existing decoded-frame, timestamp, calibration and identity
checks still run before final association and frame preparation. The background
native decode budget is bounded separately from the LAN request so a valid 8K
walk is not constrained by the synchronous client's two-minute probe allowance.
Clients without the preference retain the synchronous response contract.

## 2026-09-12 — Native motion and occupied static evidence remain bounded review inputs

Native gyro magnitude can support an explicitly selected exposure-blur revision
without supplying calibrated inertial poses. The review selector preserves the
original capture, rejected rows, parent prepared identities and RGB hashes;
changed adjacency cannot inherit a removed frame's visual edge. Both DA3 and
conditioned MapAnything consume the same retained view identities, with the
existing registration and consistency gates unchanged. The ordinary preparation
default remains a bounded soft motion preference.

Static-reference selection may name six original decoded observations and
per-frame foreground exclusions bound to source-video hash, exact PTS, selected
camera and post-dewarper pixels. A reviewed absence of a person is distinct from
an absent detection. An explicit keyframe remains one actual recorded image;
the builder masks excluded depth before temporal fusion and the alignment
consumer masks excluded RGB features. Foreground/reflection exclusions and
unobserved regions remain visible provenance, not inpainted reconstruction.
Separate predeclared observations can evaluate the candidate but cannot be
silently reused to fit it.

A self-fitted gyro axis transform or effective time shift is diagnostic only.
It does not satisfy camera/IMU calibration, measured hardware timing or metric
VIO admission. Static learned depth remains independent reconstruction evidence,
not surveyed geometric truth. Candidate generation does not bind a Scene Prior,
change camera calibration or publish a phone trajectory as a person's path.

## 2026-09-13 — Private identity continuity and independent walk review

The tracking lifecycle owner reserves source epoch and generation before
StableID assignment. The baseline manager may retain at most 256 confirmed
private bindings across compatible owner-admitted returns within 0.35 seconds
of the last observed source-media PTS. Disappearance still removes active
presence immediately and publishes its exact tombstone. Owner expiry/reset,
registry incarnation changes, competing active claims and contradictory fresh
appearance invalidate this continuity. General appearance matching owns other
returns. Source-media absence is never measured with the host processing clock;
MV3DT's distinct batch-global key is unchanged.

Cold compact or wide detector boxes no longer imply sitting/lying without
positive body geometry. Unknown posture cannot grant floor-contact or longer
hold authority. World-service diagnostics distinguish missing and invalid
calibration bindings from actual digest conflicts while preserving unavailable
geometry and the producer's explicit rejection.

Reusable offline [motion review](../tools/mapanything_phone_scan/trajectory_motion_review.py)
checks exact prepared/native frame identities, same-window gyro/visual rotation
and heldout activity. [Depth registration review](../noesis/calibration/depth_registration_review.py)
separates occupied usability from independently qualified same-anchor labels,
and reports heldout range/time coverage, residuals and observable mapping slope.
Neither command admits calibration or supplies person coordinates. A camera
optical center is not a person's ground point or torso range label.

The [handset reference reviewer](../tools/mapanything_phone_scan/trajectory_reference_review.py)
adds explicitly bound blind image annotations and heldout reprojection checks.
A translation-only trial fits training annotations and is evaluated against
both heldout pixels and fixed original comparable geometry/references, retaining
visibility losses. Improving image agreement by losing structural support
cannot pass this review, and a passing diagnostic still admits no calibration.

The September 12 walk exposes disagreement between handset reprojections and
learned static structure. A translation that helps handset pixels worsens wall
residuals and coverage; a smooth static floor above Y=0 does not establish a
surveyed height. No numerical floor/extrinsic correction or replacement depth
mapping follows from those observations. Qualified independent controls and
heldout agreement remain prerequisites for those changes. Partial provider
outputs, failed windows and missing coverage remain explicit review evidence.


## ADR-038 — Orderly EOS includes looping source-owned terminal sinks

Date: 2026-09-13. Status: accepted for the native DS9.1 graph.

DS9.1 `nvurisrcbin` creates an internal fakesink for looping local files.
That makes the source bin participate in GStreamer EOS aggregation even
though its sink is upstream of the post-mux orderly-EOS bridge. An occupied
hybrid replay confirmed every normal output sink posted EOS while `source_0`
did not, causing the runtime watchdog to terminate the process.

The graph declares the exact internal sink paths to the repository-owned
`noesiseos` bridge. Its bounded worker resolves all targets before sending
standard EOS, guards late buffers at their upstream pads, and acknowledges
only when the main path and all declared sinks accept their events. Missing
or invalid targets fail closed. The runtime retains its independent pipeline
EOS, wait-thread completion and callback-resource drain requirements. Normal
file looping, replay pacing, models and output cadence are unchanged. See
[the native bridge contract](../DS9/gst-plugins/noesiseos/README.md).

External replay controllers must allow the configured native shutdown grace
plus a supervisor margin and distinguish orderly exit from watchdog or forced
termination. Observation completion alone is not a successful replay shutdown.

## ADR-039 — RoomWalk shares one Android workspace with evidence-gated calibration

Date: 2026-09-14. Status: accepted for the RoomWalk application; not live world
or metric-VIO admission.

The Android APK packages the browser workspace and 3D viewer. A bounded,
main-document/same-origin native action channel preserves Camera2/MediaCodec
capture, exact acquisition timestamp association, focus/timing gates, paired
static recording, retained phone artifacts and background uploads. The WebView
does not replace the recorder with browser camera APIs. Computation stays on the
existing Noesis host; an offline shell is not an offline reconstruction engine.

Camera/lens and camera–IMU calibration use that same native recorder and an
explicit ChArUco board definition. One bounded CPU child processes each job,
retaining original evidence, source hashes, independent checks and rejected fits.
Stationary IMU noise is measured separately from moving-board recordings.
Execution completion, camera quality, inertial quality and downstream admission
are distinct states; null covariance and missing prerequisites stay explicit.

An advanced user may optionally select qualified camera intrinsics for future
matching native imports. Each such import snapshots the choice and verifies its
actual device, physical lens, locked focus, crop, stabilization and native image
geometry before applying the existing full-D5 rectification. Mismatches cannot
fall through to a different calibration; old scans are not rewritten. Ordinary
reconstruction and path-refinement capture do not require this saved selection
or locked-focus binding. This does not activate camera–IMU corrections, admit
metric VIO, publish PCF or alter Noesis tracking/world frames.
Independent static-camera alignment remains authoritative for those tasks.

See the [Android workflow](../tools/mapanything_phone_scan/android_companion/README.md)
and [CPU calibration contract](../tools/mapanything_phone_scan/native/roomwalk_calibration/README.md).

## ADR-040 — Short RoomWalk motion profiles require fixed-consumer evidence

Date: 2026-09-15. Status: accepted for the offline RoomWalk application; physical
phone calibration and live world admission remain distinct.

The locked-focus and fixed-board behavior in this decision applies to the
optional calibration/profile protocol. It does not make a paired room camera or
locked focus a normal reconstruction prerequisite.

The practical setup uses 60 seconds of stationary sensors, 60 seconds of
camera/lens coverage and 90 seconds of camera–IMU motion. The board stays fixed;
the phone moves. Native timed prompts and automatic board-take stops guide
acquisition. A short recording measures white noise, not long-term random walk;
drift coefficients remain explicitly conservative model priors. Full Allan
characterization is optional and separate, not a three-hour room-walk gate.

The guided UI keeps each take's recording, upload, processing and next action
in place, binding upload receipts and jobs to that exact take. A five-second
settling countdown precedes stationary acquisition; native elapsed time and
actual sample counts remain visible without touching the phone. Optional
settings and measurements are collapsed, while actionable failures stay visible.
Navigation and completed acquisition never imply qualification. The exact-mode
focus/timing prerequisites and all processing acceptance gates remain intact.

Recovery identifies each retained take by its capture ID and original calibration
intent; short timing tests are not full calibration takes. Native camera closure
must keep finalization busy, and the UI waits for that exact capture's packaged
artifact before offering upload. Missing processed prerequisites and failed
result loading are explicit states, not blank disabled selectors. Choosing an
existing phone/server take only restores its next action; upload, processing and
profile selection remain explicit, with unchanged backend qualification gates.

For board-calibration recording, locked-focus native capture pins both preview
and encoder outputs to the measured physical camera. Focus distance on an
unpinned logical stream does not prevent automatic lens switching. The selected
physical capture result supplies
the output's actual sensor timestamp and controls; logical results are retained
separately and cannot substitute for missing physical evidence. Manifest camera
identity names the physical output while recorder metadata separately names the
logical device. This distinguishes the projection from older logical-output
calibration. The preflight also binds the output-routing policy, so the new path
requires its own exact-timing test. Rejected physical 8K configurations do not
fall back to another lens or resolution. Preview recovery waits for confirmed
camera release before reopening in place. Five seconds without preview frames
or ongoing camera/encoder progress triggers an explicit bounded stop rather than
leaving a frozen display. Interrupted acquisitions remain visible partial
evidence rather than completed calibration takes.

Existing camera, measured transform, signed offset, physical scale and independent
holdout requirements remain intact. A separate job runs the fixed OpenVINS
consumer with actual sensor corrections, then compares its camera trajectory
with withheld visual-only target motion. It never refits calibration or scale.
The versioned check is a bounded application capability, not surveyed accuracy.

Only a passing profile may be explicitly selected for future matching native
walk imports, with a five-minute scope. Reuse binds optical/sensor identities,
the checked motion envelope and native executable/configuration hashes. Derived
reports do not overwrite raw captures, and later selections do not rewrite old
scans. Ordinary VIO also needs direct coverage/gap/reset and motion sanity checks.
Failed checks retain evidence and RGB reconstruction with actionable explanations.

VIO-only image analysis may scale to 1280 pixels using explicit pixel-centre
intrinsic/full-D5 mapping; source 8K recordings, reconstruction images, dense
cadence and prepared-frame identities remain unchanged. Source `capture_time_ns`
and centre-exposure `pose_time_ns` are separate. This is an exposure-centred
global-shutter approximation within its tested motion envelope, not rolling-
shutter compensation. Physical capture results are used only when their own ID,
controls and timestamp bind the exact recorded exposure.

MapAnything plus DA3 consensus and independent static-camera alignment remain
the reconstruction workflow. Nothing here publishes PCF, changes live Noesis
tracking/world calibration or converts a camera trajectory into a person track.
See the [guided workflow](../tools/mapanything_phone_scan/android_companion/README.md)
and [profile policy](../tools/mapanything_phone_scan/native/roomwalk_calibration/README.md).

## ADR-041 — RoomWalk 0.4.0 separates reconstruction and path-refinement capture intents

Date: 2026-09-19. Status: accepted for the current RoomWalk application
contract. Application validation and delivery are recorded in the upgrade history.
The capture contract remains current; the original export-only path processing
below is superseded by the executable review and registration rules in ADR-042.

RoomWalk presents two explicit capture purposes. **Reconstruction** builds room
geometry from many overlapping viewpoints. Its Noesis-camera pairing is
optional, and adding views to an existing completed reconstruction is an
explicit supplement. **Path refinement** is a separate same-walker protocol:
the walker holds the phone against their own torso with elbows tucked and stable
and moves their whole body with the phone. It requires a paired Noesis camera
and a completed reference reconstruction. These purposes are exposed through
the Capture, Library and Setup workspace tabs.

Ordinary reconstruction and path-refinement capture bypass saved
calibration-specific focus as a prerequisite. RGB reconstruction remains usable
without calibration. Native 8K/device capability and timing checks remain. A
saved locked-focus board-mode timing proof does not substitute for the
automatic-mode proof used by an ordinary walk; when that proof is absent, one
10-second automatic-mode timing check may be required, but no new full
calibration is implied. Device-availability checks are bounded evidence about
the recording path. The original board calibration controls, focus binding,
camera/IMU prerequisites and independent holdouts remain available under
optional Setup for users who select that protocol.

The native 0.4.0 integration is versionCode 21. Installing the verified build
over the existing app with the same signing key retains captures and settings.
Older captures are not implicitly relabelled. Their raw files, prepared
identities and original calibration bindings remain unchanged unless the user
explicitly starts a new review or supplement.

The retained-capture supplement endpoint
`POST /api/scans/{scan_id}/supplements/from-scan?source_scan_id=<id>` hard-links
the source video and safe raw-capture members into a new supplement directory.
This is an explicit additive revision, not a rewrite of the source evidence.
The path-review endpoint
`POST /api/scans/{scan_id}/path-review?target_scan_id=<id>` requires the
completed reference reconstruction and exports original camera poses, an IMU
rotation-only diagnostic when eligible, and retained paired references. It does
not correct inertial position, establish body ground truth, or certify the
desired 10 cm result. The user considers twenty centimetres noticeable;
neither physical-position error bound has been demonstrated.

Existing qualified VIO remains available. Its newly retained dense camera
trajectory is a separate artifact from the reconstruction's selected views and
does not become a person trajectory. The retained
`trajectory_motion_review` on
`data/mapanything_phone_scans/20260912-210621-2f755956`, using an existing
external modern127-pose DA3 artifact, found 123 supported rotation intervals,
with 0.2925° median rotation error and 0.9899 median `rho`. This supports sensor
usability for rotation diagnostics, not positional accuracy or path/body
alignment.

## ADR-042 — Metric-VIO scale candidates require fresh static registration

Date: 2026-09-21. Status: accepted for the current RoomWalk path-review
implementation; review-only and not live-world or person-position admission.

RoomWalk 0.4.1/server 1.12.0 connects path mode to executable refinement and a
paired Noesis comparison, replacing the export-only behavior described in
ADR-041. The original prepared identities and raw captures remain unchanged.
An ordered partial DA3 output retains its original prepared-index subset rather
than reindexing or rewriting the parent scan. The existing visual-revisit engine
uses a bounded 256-view/2-GiB input budget and a final-30-percent temporal holdout;
rejected corrections remain diagnostics rather than replacing the source path.

Phone paths and the selected reconstruction must independently register to the
same explicit backend-world revision and transform binding. Comparison never
fits registration to the Noesis people it evaluates. Native HTTP probes support
an explicitly approximate callback-clock join; an explicit retained light-cue
timeline must match the original archive and exact recorded cohort. Missing
world positions, held/predicted outputs, revision mismatches and path gaps are
excluded from current-measurement comparisons. Library presents selectable
lifecycle tracks and a time slider without assigning the phone carrier by
proximity. Horizontal phone-optical-center/ground-footprint separation is a
diagnostic, not body-position accuracy.

When the path-review API worker sees `capture.metric_vio_allowed` and no
completed VIO result, it runs the existing qualified VIO worker before invoking
the existing trajectory-refinement engine. The adapter passes an optional
`revalidate_scaled_carrier` callback through the path-review layers; it does not
replace the refinement engine or relax its temporal holdout and deformation
gates. The [path-review adapter](../tools/mapanything_phone_scan/path_review.py)
and [trajectory refinement](../tools/mapanything_phone_scan/trajectory_refinement.py)
retain the original capture and write review artifacts to a new output
directory.

If qualified VIO produces a non-unit metric scale change, the candidate raw
geometry, poses and output manifest are materialized together. After the
temporal holdout passes, the API callback runs the existing
`run_noesis_alignment` registration again over that materialized candidate,
using a fresh output directory and the saved target-world settings. The
revalidation must pass its static-registration quality gate and produce a
valid unit-scale world transform. A missing callback, missing target evidence,
or failed/malformed registration rejects the scaled candidate; the original
registration is not reused as a fallback.

Path comparison validates the exact candidate manifest and its registration's
world binding, then applies the candidate registration transform to the
candidate poses. It does not apply the original walk transform (`W`) to scaled
candidate geometry. The offline CLI has no configured revalidation callback,
so it explicitly rejects non-unit scale candidates rather than silently
conditioning them with the old registration. Existing retained no-VIO or
unit-scale review paths are unchanged.

This remains a review diagnostic: it does not establish phone-to-person ground
truth, a 10 cm accuracy result, hardware synchronization, or live Noesis-world
publication. See [path comparison](../tools/mapanything_phone_scan/path_comparison.py)
for the candidate-manifest and world-binding checks.

## ADR-043 — Path capture binds the selected retained PCF version

Status: accepted, 2026-09-21.

A source phone scan and its later fused reconstruction are distinct artifacts.
Selecting a room whose configured Noesis Scene Prior comes from PCF must not
silently select that scan's original provider output or failed raw alignment.
RoomWalk reads the bounded configured catalog and marks the exact available
PCF choice. The optional `walk_intent.target_reference` records the prior ID,
camera, manifest digest and frame-binding digest at capture time. Native saved
metadata and upload preserve it without changing legacy intent or normal
reconstruction behavior.

The [reference resolver](../tools/mapanything_phone_scan/path_reference.py)
checks the existing catalog, manifest, quality, source-scan identity and point
artifact. Path review verifies this selected room's calibration/world edge
against the walk's independent static-camera registration. It consumes the
already registered PCF artifact directly, not fabricated provider poses or the
original scan's unrelated registration. Missing, changed, ambiguous or
incompatible references fail explicitly; the raw reconstruction is never a
fallback. Review retains the exact selected manifest, points and binding.

This is read-only reference selection, not inference, a new room admission,
live catalog mutation, or promotion. The new walk still needs its own passed
static-camera registration and retained timing evidence; choosing PCF does not
certify body position or improve a path merely by changing its label.
