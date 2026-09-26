# AGENTS.md — Phone-walk reconstruction

## Policy precedence

[Root policy](../../AGENTS.md) applies. This tool produces reconstruction review
candidates and has separate capture, fusion, alignment, and admission boundaries.

## Read by change

Use the relevant [README sections](README.md): provider semantics and consensus
fusion for reconstruction; camera/IMU capture for timing and calibration;
detection/tracking integration for spatial authority; configuration for execution.
For requested PCF execution or handoff, read only the applicable stages of the
[PCF runbook](../../docs/PCF_Workflow.md). Runtime admission additionally uses the
[Scene Prior contract](../../docs/scene_prior_v1.md).

## Reconstruction and evidence rules

1. Preserve the selected MapAnything + DA3 consistency-gated workflow and matching
   prepared phone-view identities. Keep provider depth, pose, confidence, and
   validity-mask semantics distinct as defined in the README.
2. Static-camera reconstruction supplies independent Noesis alignment and
   validation authority. Do not silently mix static frames into phone-only
   inference/fusion. An explicitly selected paired workflow must preserve its
   separate source provenance and validation boundary.
3. Reconstruction, PCF results, meshes, and structural overlays remain review
   candidates until an explicitly authorized runtime handoff admits them. A
   plausible display or passing evaluation does not publish calibration, world
   state, identity, overlap, or a Scene Prior.
4. Preserve recordings, prepared-frame order/IDs, hashes, model identity,
   camera/calibration identity, coordinate frames, units, transforms, and target
   revisions. Failed alignment or revision mismatch remains review-only; do not
   substitute scans, cameras, transforms, or weakened acceptance thresholds.
5. Metric VIO requires an admitted capture report with acquisition timestamps,
   camera/IMU calibration, units/axes, and measured timing offset. Browser callback
   times cannot admit it. Missing calibration leaves RGB reconstruction available;
   camera trajectories do not become person trajectories without separate evidence.

## Execution and validation

- An authorized conditioned MapAnything run that cannot coexist with the native
  appliance uses the documented `NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=1` lease:
  record before stopping a previously active `menon-appliance.target`, restore on
  success/failure, and recover interrupted leases. Ordinary edits do not trigger
  that execution condition.
- Preserve input evidence and use new output directories for changed experiments.
  Validate the changed capture/provider/alignment producer and direct consumer
  with focused checks. Documentation-only changes use the root docs checks.
- Sealed evidence and Scene Prior binding stages belong to their authorized
  admission/handoff task. They do not expand an ordinary repair into an appliance
  release or runtime restart workflow.
