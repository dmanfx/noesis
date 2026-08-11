from __future__ import annotations

from copy import deepcopy

from noesis.validation.menon_production_geometry import (
    validate_menon_production_geometry,
)


IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def _production_geometry() -> dict[str, object]:
    endpoint = [1.0, 0.05, 2.0]
    return {
        "contract": "menon.production-track-geometry-debug",
        "contractVersion": 1,
        "bounded": True,
        "validationTruncated": False,
        "worldToSceneMatrix": IDENTITY,
        "validations": [
            {
                "contract": "menon.track.scene.validation",
                "contractVersion": 1,
                "status": "pass",
                "structuralStatus": "pass",
                "physicalStatus": "pass",
                "identity": {"entityId": "entity-1", "pathKey": "entity:entity-1"},
                "checks": [
                    {"code": "canonical_frame_contract", "status": "pass"},
                    {"code": "production_visual_present", "status": "pass"},
                    {
                        "code": "world_to_scene_exact",
                        "status": "pass",
                        "expectedScenePoint": [1.0, 0.0, 2.0],
                        "actualScenePoint": [1.0, 0.0, 2.0],
                        "toleranceScene": 1e-6,
                    },
                    {
                        "code": "visual_root_identity",
                        "status": "pass",
                        "maxElementError": 0.0,
                        "toleranceScene": 1e-6,
                    },
                    {
                        "code": "rendered_trail_endpoint_exact",
                        "status": "pass",
                        "expectedEndpoint": endpoint,
                        "actualEndpoint": endpoint,
                        "toleranceScene": 1e-6,
                    },
                    {
                        "code": "rendered_trail_world_endpoint_exact",
                        "status": "pass",
                        "expectedEndpoint": endpoint,
                        "actualEndpoint": endpoint,
                        "toleranceScene": 1e-6,
                    },
                    {
                        "code": "decorator_input_endpoint_exact",
                        "status": "pass",
                        "expectedEndpoint": endpoint,
                        "actualEndpoint": endpoint,
                        "toleranceScene": 1e-6,
                    },
                    {
                        "code": "decorator_world_anchor_alignment",
                        "status": "pass",
                        "trailEndpointWorld": endpoint,
                        "decoratorAnchorWorld": endpoint,
                        "toleranceScene": 1e-6,
                    },
                ],
            }
        ],
        "stageVisibility": {
            "bounded": True,
            "truncated": False,
            "entries": [
                {
                    "key": "entity:entity-1",
                    "canonicalWorld": {"entityPresent": True},
                    "presentation": {"entityPresent": True, "rejectionReason": None},
                    "renderer": {
                        "visualPresent": True,
                        "markerObjectPresent": True,
                        "markerEndpointPresent": True,
                        "markerVisible": True,
                        "rejectionReason": None,
                    },
                    "trail": {
                        "geometryPresent": True,
                        "movementSegmentPresent": False,
                        "currentPointPresent": True,
                        "rejectionReason": "trail_has_no_movement_segment_yet",
                    },
                }
            ],
        },
    }


def test_production_geometry_independently_accepts_exact_renderer_endpoints() -> None:
    checks = validate_menon_production_geometry(
        _production_geometry(),
        expected_world_to_scene=IDENTITY,
    )

    assert {check.status.value for check in checks} == {"pass"}


def test_production_geometry_recomputes_endpoint_error_instead_of_trusting_status() -> None:
    payload = deepcopy(_production_geometry())
    validation = payload["validations"][0]
    endpoint_check = next(
        check
        for check in validation["checks"]
        if check["code"] == "rendered_trail_endpoint_exact"
    )
    endpoint_check["actualEndpoint"] = [2.0, 0.05, 2.0]

    checks = validate_menon_production_geometry(payload, expected_world_to_scene=IDENTITY)
    renderer = next(check for check in checks if check.id == "MENON.production_renderer_geometry")

    assert renderer.status.value == "fail"
    assert renderer.metric["failure_count"] == 1
    assert renderer.metric["failures"][0]["numeric_failures"] == [
        "rendered_trail_endpoint_exact"
    ]


def test_production_geometry_keeps_physical_alignment_separate_from_renderer() -> None:
    payload = deepcopy(_production_geometry())
    payload["validations"][0]["physicalStatus"] = "fail"
    payload["validations"][0]["status"] = "fail"

    checks = validate_menon_production_geometry(payload, expected_world_to_scene=IDENTITY)
    renderer = next(check for check in checks if check.id == "MENON.production_renderer_geometry")
    physical = next(check for check in checks if check.id == "MENON.production_physical_placement")

    assert renderer.status.value == "pass"
    assert physical.status.value == "fail"
