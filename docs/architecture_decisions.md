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

**Accepted:** 2026-08-12

YOLO26-m + NvDCF baseline tracking is active. MV3DT and AMC remain disabled.
Only Kitchen/Family Room is a prospective overlap edge; Living Room/Family
Room do not overlap and Kitchen/Living Room are adjacency only.

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
callbacks. Shutdown closes and drains that gate before WebSocket egress.

**Why:** Dashboard state must never outrun committed world authority, and native
callbacks must not race transport teardown.

## ADR-007 — PCF is the canonical depth-panel floorplan source

**Accepted:** 2026-08-13

For an admitted camera-bound Scene Prior, the depth drawer and BEV floorplan use
the immutable PCF revision. Full measured reconstruction extent is distinct
from authored semantic room membership. Live tracking authority is paired into
the PCF BEV payload; PCF does not create or replace tracks.

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

**Accepted:** 2026-08-15

Superseded DS7/DS8, DS9.0, container, migration, experiment, and validation
diaries are retained under `docs/history/`, `DS9/docs/history/`, and
`plans/archive/`. Active indexes never route implementation work through them.
