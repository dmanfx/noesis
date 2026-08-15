# Menon camera reprojection acceptance

Use this check only when camera calibration, backend-world coordinates, or
Menon's world-to-scene transform changes.

Require an exact camera ID, calibrated dewarped frame size, K/E or pose
authority, backend-world point, authored transform, and visible Menon result.
Compare fit and untouched holdout evidence separately. Reject missing frame
metadata, reflected/degenerate transforms, stale run/sequence data, or a point
that is visually corrected only by client-side clamping.

Prefer the existing authored-scene, track-geometry, telemetry, and Menon-trace
tools from `validation_catalog.md`. One representative camera/track check plus
the directly affected holdout is sufficient unless the calibration change
itself spans all cameras.
