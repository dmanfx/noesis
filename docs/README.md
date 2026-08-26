# Noesis documentation

Status: current for the native DeepStream 9.1 application, 2026-08-26.

This directory separates current specifications from historical implementation
records. If an archived document conflicts with a current document, the current
document and live source/config win.

## Start here

| Need | Current document |
| --- | --- |
| System architecture and ownership | [`CODEBASE_DESCRIPTION.md`](CODEBASE_DESCRIPTION.md) |
| Installed runtime, models, versions, ports, and performance | [`runtime_baseline.md`](runtime_baseline.md) |
| Pipeline hot-path and performance invariants | [`performance_invariants.md`](performance_invariants.md) |
| Universal person world localization | [`universal_world_localization.md`](universal_world_localization.md) |
| Exact DeepStream graph | [`../DS9/PIPELINE_GRAPH.md`](../DS9/PIPELINE_GRAPH.md) |
| Native runtime operation | [`../DS9/docs/runtime_host_boundary.md`](../DS9/docs/runtime_host_boundary.md) |
| Focused validation | [`testing_guide.md`](testing_guide.md) |
| Run or reproduce Prior-Conditioned Fusion (PCF) | [`PCF_Workflow.md`](PCF_Workflow.md) |
| Join overlapping accepted PCF rooms | [`PCF_Multiroom_Registration.md`](PCF_Multiroom_Registration.md) |
| PCF coordinate/orientation contract and flip audit | [`PCF_Coordinate_Orientation_Audit.md`](PCF_Coordinate_Orientation_Audit.md) |
| WebSocket contract | [`api_contracts_ws.md`](api_contracts_ws.md) |
| REST contract | [`api_contracts_rest.md`](api_contracts_rest.md) |
| Metadata contracts | [`metadata_contracts.md`](metadata_contracts.md) |
| Pose-assisted StableID behavior | [`pose_stable_id_integration.md`](pose_stable_id_integration.md) |
| Architecture decisions | [`architecture_decisions.md`](architecture_decisions.md) |
| Upgrade/change history | [`upgrade_history.md`](upgrade_history.md) |
| Agent skill routing | [`../DS9/docs/deepstream_9_1_agent_skills.md`](../DS9/docs/deepstream_9_1_agent_skills.md) |

## Product and data guides

- [`Telemetry_Schema.md`](Telemetry_Schema.md) and
  [`telemetry_contract.md`](telemetry_contract.md): telemetry shapes and depth
  publication boundary.
- [`universal_world_localization.md`](universal_world_localization.md):
  camera-agnostic floor/depth hypothesis resolution, covariance, conservative
  PCF evidence, PersonGroundState boundary, and dashboard diagnostics.
- [`depth_metadata.md`](depth_metadata.md) and
  [`DEPTH_STACK_FLOW_V2.md`](DEPTH_STACK_FLOW_V2.md): object and full-frame
  depth semantics.
- [`PCF_Workflow.md`](PCF_Workflow.md): canonical end-to-end
  Prior-Conditioned Fusion runbook, from phone capture through Scene Prior
  activation and dashboard verification.
- [`PCF_Multiroom_Registration.md`](PCF_Multiroom_Registration.md):
  cross-session registration, pose-graph gates, connector capture, and
  provenance-preserving multi-room reintegration.
- [`PCF_Coordinate_Orientation_Audit.md`](PCF_Coordinate_Orientation_Audit.md):
  canonical coordinate chain, flip inventory, line authority, asymmetric
  orientation gates, and current three-room validation.
- [`Phone_Walk_Fusion_Reconstruction.md`](Phone_Walk_Fusion_Reconstruction.md):
  PCF algorithm, selection evidence, and quality interpretation.
- [`scene_prior_v1.md`](scene_prior_v1.md): immutable runtime artifact and
  catalog contract built from approved PCF evidence.
- [`Virtual_Twin_Reconstruction.md`](Virtual_Twin_Reconstruction.md):
  independent static-camera room-revision workflow.
- [`charuco_intrinsics_calibration.md`](charuco_intrinsics_calibration.md):
  camera intrinsics workflow.
- [`roi_editor.md`](roi_editor.md) and
  [`Static_ROI_Exclusion.md`](Static_ROI_Exclusion.md): ROI editing and native
  exclusion behavior.
- [`Runtime_Secrets.md`](Runtime_Secrets.md): private runtime inputs.
- [`Occupancy_Publishing.md`](Occupancy_Publishing.md) and
  [`Integrations_Playbook.md`](Integrations_Playbook.md): integrations.

## Diagrams and operations

- [`flow_diagram_high_level.md`](flow_diagram_high_level.md) and
  [`flow_diagram_low_level.md`](flow_diagram_low_level.md): current application
  and publication flows.
- [`WebSocket_API.md`](WebSocket_API.md): concise operator-facing transport
  reference; field-level authority remains
  [`api_contracts_ws.md`](api_contracts_ws.md).
- [`Cache_Clearing.md`](Cache_Clearing.md): targeted GStreamer registry and
  TensorRT realization repair.
- [`CONVENTIONS.md`](CONVENTIONS.md): repository commit, pull-request, and
  release-note conventions.

## Runtime-specific DS9.1 docs

Use [`../DS9/docs/README.md`](../DS9/docs/README.md) for model, native bridge,
canonical-world, BEV/capture, maintenance, and operator references.

## Active work records

- [`../plans/ds91_native_host_only_migration.md`](../plans/ds91_native_host_only_migration.md):
  accepted native state and remaining destructive legacy/Docker cleanup.
- [`../plans/household_identity/README.md`](../plans/household_identity/README.md):
  current identity contracts and uncompleted real-evidence authority gates.
- [`../plans/noesis_menon_validation/README.md`](../plans/noesis_menon_validation/README.md):
  stable fixtures and direct Noesis-to-Menon validation tiers.

## Historical material

DeepStream 7/8, DeepStream 9.0, container-era DS9.1, completed upgrade plans,
bridge audits, and experiment reports live under [`history/`](history/README.md),
[`../plans/archive/`](../plans/archive/README.md). They explain prior decisions
but are not instructions for the current app.

## Documentation change log

- **2026-08-26:** Consolidated runtime history under one root archive, moved
  completed DS9 native-port work orders into the plans archive, retired active
  DS8 visuals and nested archive policy files, and aligned current MV3DT status
  with the accepted Kitchen/Family Room explicit opt-in.
- **2026-08-25:** Replaced the active room-specific baseline localization
  policy with one universal, uncertainty-aware world measurement resolver;
  documented typed hypotheses, real compatible-source fusion, conservative
  PCF evidence, exact-cohort diagnostics, and the normal-dashboard
  `Localization details` toggle.
- **2026-08-23:** Made the occupied-scene performance recovery durable through
  one canonical hot-path policy covering GPU/NVMM ownership, bounded optional
  work, exact publication cohorts, pooled reuse, occupancy-amplification caps,
  capability preservation, and source/encode/WebRTC measurement.
- **2026-08-15:** Established one PCF/Scene Prior camera-ground orientation
  contract, removed room-specific and duplicated display flips, and validated
  Living Room, Family Room, Kitchen, and the review-only Family/Kitchen
  presentation evidence without mutating backend geometry.
- **2026-08-15:** Made Prior-Conditioned Fusion a first-class documented
  capability with one canonical capture-to-runtime runbook, explicit authority
  and admission gates, the previously missing bundle-to-Scene-Prior handoff,
  retention rules, and the current three-room inventory.
- **2026-08-15:** Rebased the documentation on the native-host DS9.1 runtime;
  replaced DS8/container entrypoints, refreshed architecture and pipeline
  diagrams, renamed live contracts, added runtime/testing/decision/history
  guides, and moved superseded material into explicit archives.
- **2026-08-15:** Recorded the native runtime, dependency parity correction,
  restored performance baseline, and canonical PCF BEV tracking authority.
