# Validation Report Examples

Status: current as of 2026-05-27.

These examples show how future agents should read the shared validation report
shape. The exact check IDs will vary by fixture, telemetry capture, or Menon
trace, but status, confidence, failure category, evidence, and suggested
diagnostics should stay consistent.

## Passing Report

Use this as the target shape for a fixture or runtime window where the relevant
evidence is present and agrees.

```json
{
  "summary": {
    "status": "pass",
    "level": "verified",
    "failure_count": 0,
    "warning_count": 0,
    "blocked_count": 0
  },
  "checks": [
    {
      "id": "TRACK.detection_world_projection",
      "status": "pass",
      "failure_type": null,
      "metric": {
        "p95_foot_agreement_m": 0.02,
        "min_projected_bbox_iou": 0.87
      },
      "detail": "Detection footpoints, floor contact, room bounds, bbox reprojection, aspect, and height assumptions agree."
    }
  ]
}
```

Agent response pattern:

- Treat the validated layer as usable for the scoped fixture or runtime window.
- Still name the evidence mode, such as fixture-only, saved telemetry, live DS8,
  saved Menon trace, or live Menon browser capture.

## Warning Report

Warnings mean the output may be usable, but an independent check is outside the
good band or partially contradicted.

```json
{
  "summary": {
    "status": "warning",
    "level": "tentative",
    "failure_count": 0,
    "warning_count": 1,
    "blocked_count": 0
  },
  "checks": [
    {
      "id": "TRACK.projection_confidence",
      "status": "warning",
      "failure_type": "projection_failure",
      "metric": {
        "score": 0.61,
        "component_counts": {
          "reprojection": 1,
          "temporal_smoothness": 1,
          "semantic_validity": 1
        }
      },
      "suggested_next_diagnostic": "Inspect which projection confidence component collapsed: footpoint, floor, ray, room bounds, or person height."
    }
  ]
}
```

Agent response pattern:

- Do not call the feature fully validated.
- State which component weakened the result and what diagnostic should run next.
- Avoid masking the warning with a fallback path.

## Failing Report

Failures mean a required threshold was violated.

```json
{
  "summary": {
    "status": "fail",
    "level": "invalid",
    "failure_count": 1,
    "warning_count": 0,
    "blocked_count": 0
  },
  "checks": [
    {
      "id": "SCENE.menon_assets",
      "status": "fail",
      "failure_type": "scene_failure",
      "metric": {
        "non_walkable_floor_count": 1,
        "doorway_blocked_count": 1,
        "missing_overlay_layer_count": 2
      },
      "suggested_next_diagnostic": "Inspect Menon asset scale/origin, collision mesh export, floor walkability, doorway clearance, camera marker pose, and debug overlay layer rendering."
    }
  ]
}
```

Agent response pattern:

- Stop treating the affected layer as valid.
- Categorize the failure and route the next fix to the likely subsystem.
- Preserve the failing evidence instead of replacing it with a substitute route.

## Blocked Report

Blocked means the required evidence was unavailable. It is not a pass.

```json
{
  "summary": {
    "status": "blocked",
    "level": "tentative",
    "failure_count": 0,
    "warning_count": 0,
    "blocked_count": 1
  },
  "checks": [
    {
      "id": "MENON.checkout.required",
      "status": "blocked",
      "failure_type": "infrastructure_failure",
      "detail": "Menon checkout validation was required, but no Menon root was provided.",
      "suggested_next_diagnostic": "Provide --menon-root or MENON_ROOT and rerun the Menon trace report."
    }
  ]
}
```

Agent response pattern:

- Say exactly which evidence is missing.
- Do not downgrade to Noesis-only validation when Menon evidence is required.
- Rerun with the missing fixture, telemetry capture, DS8 runtime, or Menon root.

## Minimal Commands

```bash
python3 scripts/noesis_validation_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --fixture-id minimal_validation_fixture
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --require-menon-root
```
