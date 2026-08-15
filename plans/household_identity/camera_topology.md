# Household identity — camera topology

This topology governs only StableID exclusivity and handoff. It does not assert
MV3DT geometry and does not enable AMC.

## Current source map

The order is verified against `DS9/config/infer.yaml` and recorded in
`config/camera_topology.yaml`:

| Source ID | Camera |
| --- | --- |
| 0 | Living Room |
| 1 | Kitchen |
| 2 | Family Room |

## Geometry boundary

- Kitchen ↔ Family Room: configured identity-overlap candidate.
- Living Room ↔ Family Room: no overlap.
- Kitchen ↔ Living Room: close adjacency, not overlap.

The Kitchen ↔ Family Room permit is not unconditional. The manager grants it
only when the pair is enabled and timestamps, world distance, and appearance
meet the configured bounds. Missing or invalid world evidence denies the
permit; `overlap_allow_appearance_only` is false.

Handoff after a track has left one camera is not dual-camera activity and uses
normal continuity/gallery policy.

## Practical checks

1. One person genuinely co-visible in Kitchen/Family Room keeps one identity.
2. Different people in those rooms never share an identity.
3. Kitchen-to-Family handoff remains continuous after release.
4. Living/Family and Kitchen/Living never receive a simultaneous overlap
   permit.

Do not broaden the overlap graph until current geometry and synchronized video
prove the physical relationship.
