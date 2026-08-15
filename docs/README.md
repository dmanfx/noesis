# Noesis documentation

Status: current for the native DeepStream 9.1 application, 2026-08-15.

This directory separates current specifications from historical implementation
records. If an archived document conflicts with a current document, the current
document and live source/config win.

## Start here

| Need | Current document |
| --- | --- |
| System architecture and ownership | [`CODEBASE_DESCRIPTION.md`](CODEBASE_DESCRIPTION.md) |
| Installed runtime, models, versions, ports, and performance | [`runtime_baseline.md`](runtime_baseline.md) |
| Exact DeepStream graph | [`../DS9/PIPELINE_GRAPH.md`](../DS9/PIPELINE_GRAPH.md) |
| Native runtime operation | [`../DS9/docs/runtime_host_boundary.md`](../DS9/docs/runtime_host_boundary.md) |
| Focused validation | [`testing_guide.md`](testing_guide.md) |
| WebSocket contract | [`api_contracts_ws.md`](api_contracts_ws.md) |
| REST contract | [`api_contracts_rest.md`](api_contracts_rest.md) |
| Metadata contracts | [`metadata_contracts.md`](metadata_contracts.md) |
| Architecture decisions | [`architecture_decisions.md`](architecture_decisions.md) |
| Upgrade/change history | [`upgrade_history.md`](upgrade_history.md) |
| Agent skill routing | [`../DS9/docs/deepstream_9_1_agent_skills.md`](../DS9/docs/deepstream_9_1_agent_skills.md) |

## Product and data guides

- [`Telemetry_Schema.md`](Telemetry_Schema.md) and
  [`telemetry_contract.md`](telemetry_contract.md): telemetry shapes and depth
  publication boundary.
- [`depth_metadata.md`](depth_metadata.md) and
  [`DEPTH_STACK_FLOW_V2.md`](DEPTH_STACK_FLOW_V2.md): object and full-frame
  depth semantics.
- [`scene_prior_v1.md`](scene_prior_v1.md): immutable Scene Prior/PCF evidence.
- [`Phone_Walk_Fusion_Reconstruction.md`](Phone_Walk_Fusion_Reconstruction.md)
  and [`Virtual_Twin_Reconstruction.md`](Virtual_Twin_Reconstruction.md):
  offline reconstruction workflows.
- [`charuco_intrinsics_calibration.md`](charuco_intrinsics_calibration.md):
  camera intrinsics workflow.
- [`roi_editor.md`](roi_editor.md) and
  [`Static_ROI_Exclusion.md`](Static_ROI_Exclusion.md): ROI editing and native
  exclusion behavior.
- [`Runtime_Secrets.md`](Runtime_Secrets.md): private runtime inputs.
- [`Occupancy_Publishing.md`](Occupancy_Publishing.md) and
  [`Integrations_Playbook.md`](Integrations_Playbook.md): integrations.

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
[`../DS9/docs/history/`](../DS9/docs/history/README.md), and
[`../plans/archive/`](../plans/archive/README.md). They explain prior decisions
but are not instructions for the current app.

## Documentation change log

- **2026-08-15:** Rebased the documentation on the native-host DS9.1 runtime;
  replaced DS8/container entrypoints, refreshed architecture and pipeline
  diagrams, renamed live contracts, added runtime/testing/decision/history
  guides, and moved superseded material into explicit archives.
- **2026-08-15:** Recorded the native runtime, dependency parity correction,
  restored performance baseline, and canonical PCF BEV tracking authority.
