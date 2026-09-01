# Architecture decisions

This is the current decision ledger. Historical DS8 and container-era ledgers
remain in the archives for rationale and chronology.

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
confidence, and seated or active lower-body-occlusion context; lying and
ordinary upright/dropout rows remain excluded. Torso range is never verified
ground contact, cannot seed the weak floor-consensus path or learned height,
and cannot suppress an independently eligible `bbox_bottom` floor ray. Floor
support is a semantic authority tier, not a confidence-score hint: when any
valid floor-supported hypothesis exists, it ranks ahead of non-floor body,
seat, couch, or unknown support. Covariance intersection is allowed only among
equal support states, and the hook independently requires every selected and
contributing metric hypothesis to be floor-supported. A precise torso/body
range therefore remains diagnostic even when it is spatially compatible with
the selected floor point; it cannot pull the metric result, steer the bounded
process posterior, or replace the accepted image origin.

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
with broad height uncertainty. It is typed as seat support rather than visible
foot contact, may use stationary hold when current geometry is temporarily
absent, and does not disappear merely because ankles are hidden. Upright body
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
