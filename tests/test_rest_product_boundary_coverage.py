from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.responses import Response

from noesis.server import (
    analytics_api,
    boundary_metrics,
    depth_api,
    health_api,
    reid_api,
    reid_v2_api,
    scene_api,
    virtual_twin_api,
)


def test_successful_measured_route_fails_closed_without_boundary_context() -> None:
    app = FastAPI()
    app.router.route_class = boundary_metrics.BoundaryMetricsRoute

    @app.get("/unmarked")
    def unmarked() -> dict[str, bool]:
        return {"ok": True}

    boundary_metrics.reset_boundary_serialization_metrics()
    with pytest.raises(RuntimeError, match="did not declare boundary measurement"):
        TestClient(app).get("/unmarked")

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["count"] == 0
    assert metrics["errors_total"] == 1
    assert metrics["errors"] == {
        "rest|/unmarked|unmarked_response|missing_boundary_context|RuntimeError": 1
    }


def test_pre_rendered_response_requires_an_explicit_exemption() -> None:
    app = FastAPI()
    app.router.route_class = boundary_metrics.BoundaryMetricsRoute

    @app.get("/artifact", response_class=Response)
    def artifact(request: Request) -> Response:
        response = Response(b"artifact", media_type="application/octet-stream")
        boundary_metrics.mark_rest_response_exempt(
            request,
            reason="pre_rendered_response",
        )
        return response

    boundary_metrics.reset_boundary_serialization_metrics()
    response = TestClient(app).get("/artifact")

    assert response.status_code == 200
    assert response.content == b"artifact"
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["count"] == 0
    assert metrics["errors_total"] == 0


def test_every_product_route_declares_measurement_or_explicit_exemption() -> None:
    routers = (
        analytics_api.app.router,
        depth_api.app.router,
        reid_api.app.router,
        reid_v2_api.router,
        virtual_twin_api.app.router,
        health_api.router,
        scene_api.router,
    )
    product_routes = [
        route
        for router in routers
        for route in router.routes
        if isinstance(route, APIRoute)
    ]

    # Forty-one framework-rendered JSON routes are measured. Four verified scene
    # artifacts, one virtual-twin FileResponse, and one exact depth-component
    # response are outside that JSON budget.
    assert len(product_routes) == 47
    measured = 0
    exempt = 0
    for route in product_routes:
        assert isinstance(route, boundary_metrics.BoundaryMetricsRoute), route.path
        source = inspect.getsource(route.endpoint)
        if "mark_rest_response_exempt(" in source:
            exempt += 1
            assert "mark_rest_response(" not in source
        else:
            measured += 1
            assert "measure_rest_response_model(" in source, route.path
            assert "mark_rest_response(" in source, route.path

    assert measured == 41
    assert exempt == 6
