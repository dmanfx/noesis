# Pose metadata and StableID routing

_Status: checked-out native DS9.1 routing, verified 2026-09-07._

This note describes the identity calls in the current source. The configured
process mode and admitted authority artifacts determine which lane owns live
public identity; inspect them before making a runtime claim.

## Current DS9 hook integration

The [`_AnalyticsTelemetryProcessor`](../DS9/noesis/pipelines/hooks.py) has two
identity routes:

- When identity-v2 is not authoritative, `_maybe_assign_stable_id()` calls
  `StableIDManager.update()` with the sensor/tracker identity, box, timestamp,
  zone, frame placeholder, and ReID embedding. It does not pass the optional
  `pose_features` or `pose_quality` arguments.
- When identity-v2 is authoritative, that legacy assignment call is bypassed.
  The hook collects `IdentityFramePrimitive` rows and calls
  `IdentityV2Service.process_source_frame()` once for the complete source-frame
  cohort. The primitives carry the embedding, box/confidence data, and world
  evidence alongside the public and diagnostic rows.

The hook separately extracts pose keypoints for anchor processing and the
`pose_present` diagnostic. A present pose or a persistent label does not prove
that optional pose matching in `StableIDManager` contributed to the decision.

## Identity-v2 mode and missing embeddings

[`noesis/identity_v2_service.py`](../noesis/identity_v2_service.py) reads
`NOESIS_IDENTITY_V2_MODE`; its source default is `shadow`. Authoritative startup
requires the matching authority-cutover artifact in addition to the scorer's
calibration. The source default is not evidence of the running process's mode.

Without a fresh embedding, the service may retain an already accepted overlay
within its bounded tracker-continuity hold and marks it
`tracker_continuity_hold_no_fresh_embedding`. Without an eligible hold, the row
becomes provisional with `server_embedding_unavailable`; authoritative mode
clears its public identity. Neither path is pose-only matching or a fallback to
legacy identity output.

## Optional manager capabilities

[`StableIDManager.update()`](../reid/stable_id_manager.py) still accepts optional
pose features and quality. Its manager-level implementation includes quality
gating, pose/appearance blending, and pose-only ghost/gallery matching for
callers that supply those inputs. These capabilities are not wired by the
current DS9 hook call described above. Changing a manager pose threshold does
not add the missing call inputs.

[`test_stable_id_manager_pose.py`](../tests/test_stable_id_manager_pose.py)
exercises those optional manager capabilities in isolation. It does not prove
the DS9 hook uses them or authorize an identity-v2 authority cutover.

## Scope and validation

Follow [identity guidance](../reid/AGENTS.md) and the
[household identity contracts](../plans/household_identity/contracts.md) for
assignment changes. Preserve selected SGIE/native extraction, exact
camera/frame/tracker joins, and the separate authority-acceptance gates. Do not
add pose-only assignment or another extraction path merely to match older
descriptions of this integration.

Documentation-only corrections use the [root documentation checks](../AGENTS.md#files-docs-and-commits).
An authorized integration change needs focused producer/consumer validation
selected from the affected contracts; it does not require rebuilding unchanged
native artifacts or running the application for a documentation correction.
