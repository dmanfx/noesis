# DS9.1 documentation index

Status: native-host baseline, 2026-08-15.

## Current operational docs

1. [`../README.md`](../README.md): runtime ownership, versions, lane, and start
   point.
2. [`../PIPELINE_GRAPH.md`](../PIPELINE_GRAPH.md): exact graph and data flow.
3. [`runtime_host_boundary.md`](runtime_host_boundary.md): native environment,
   preflight, lifecycle, readiness, and security boundary.
4. [`validation_runbook.md`](validation_runbook.md): focused direct checks.
5. [`deepstream_9_1_agent_skills.md`](deepstream_9_1_agent_skills.md): required
   skill routing and repository pin overrides.
6. [`../DS9_REBUILD_AND_SMOKE_GATES.md`](../DS9_REBUILD_AND_SMOKE_GATES.md):
   affected-artifact rebuild and smoke guidance.

## Current product/model docs

- [`canonical_world.md`](canonical_world.md): observation and world authority.
- [`bev_capture_event_integration.md`](bev_capture_event_integration.md): BEV,
  active floorplan, and exact capture publication.
- [`MapAnything_Depth.md`](MapAnything_Depth.md): selected full-frame manual
  depth lane.
- [`DA3Metric_Large.md`](DA3Metric_Large.md): optional manual depth profile,
  not selected by default.
- [`Static_ROI_Exclusion.md`](Static_ROI_Exclusion.md): native exclusion plugin
  and transactional reload behavior.
- [`asset_manifest.schema.json`](asset_manifest.schema.json) and
  [`runtime_ownership.yaml`](runtime_ownership.yaml): machine-readable asset and
  source ownership contracts.

## Capability state

The baseline lane is YOLO26-m + NvDCF with Swin ReID, YOLO26 pose, always-on
DAv2 tracking depth, gated MapAnything depth, WebRTC media, and JSON BEV. MV3DT
and AMC are disabled pending the documented Kitchen geometry/overlap gate.

## History

[`history/README.md`](history/README.md) indexes the DS8 migration copies,
DS9.0/container deployment material, completed 9.1 upgrade plans, bridge audits,
and model experiments. Those files are evidence only and must not be used as
current run/build instructions.
