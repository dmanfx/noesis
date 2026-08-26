from __future__ import annotations

import json
import time

import fastapi.routing
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_serializer
import pytest

from noesis.server import boundary_metrics


class WireResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    public_name: str = Field(alias="publicName")
    optional_value: str | None = Field(default=None, alias="optionalValue")


class InternalWireResponse(WireResponse):
    internal_only: str


class ExplodingWireResponse(BaseModel):
    value: str

    @field_serializer("value")
    def explode_during_wire_render(self, _value: str) -> str:
        raise RuntimeError("intentional render failure")


def _build_probe_app() -> FastAPI:
    child = FastAPI()
    child.router.route_class = boundary_metrics.BoundaryMetricsRoute

    @child.get(
        "/probe",
        response_model=WireResponse,
        response_model_exclude_none=True,
        status_code=201,
    )
    def probe(request: Request, response: Response) -> InternalWireResponse:
        model_start_ns = time.perf_counter_ns()
        payload = InternalWireResponse(
            public_name="resident",
            optional_value=None,
            internal_only="must-not-cross-wire",
        )
        model_ms = (time.perf_counter_ns() - model_start_ns) / 1_000_000.0
        response.headers["x-contract-header"] = "preserved"
        boundary_metrics.mark_rest_response(
            request,
            "/probe:get",
            "WireResponse",
            model_duration_ms=model_ms,
        )
        return payload

    # Production mounts these API routers into the DS9.1 runtime app. This
    # exercises that exact include-router response path.
    parent = FastAPI()
    parent.include_router(child.router)
    return parent


def test_rest_boundary_measures_the_single_actual_wire_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_probe_app()
    operation = app.openapi()["paths"]["/probe"]["get"]
    assert "parameters" not in operation
    assert operation["responses"] == {
        "201": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/WireResponse"}
                }
            },
        }
    }

    serialize_calls = 0
    json_render_calls = 0
    original_serialize = fastapi.routing.serialize_response
    original_dumps = json.dumps

    async def counted_serialize(*args, **kwargs):
        nonlocal serialize_calls
        serialize_calls += 1
        return await original_serialize(*args, **kwargs)

    def counted_dumps(*args, **kwargs):
        nonlocal json_render_calls
        json_render_calls += 1
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(fastapi.routing, "serialize_response", counted_serialize)
    monkeypatch.setattr(json, "dumps", counted_dumps)
    boundary_metrics.reset_boundary_serialization_metrics()

    response = TestClient(app).get("/probe")

    expected_body = b'{"publicName":"resident"}'
    assert response.status_code == 201
    assert response.content == expected_body
    assert response.headers["x-contract-header"] == "preserved"
    assert response.headers["content-type"] == "application/json"
    assert int(response.headers["content-length"]) == len(expected_body)
    assert serialize_calls == 1
    # FastAPI 0.135 uses Pydantic's direct dump_json path here. Any json.dumps
    # call would therefore be the forbidden surrogate render this test guards.
    assert json_render_calls == 0
    assert not hasattr(boundary_metrics, "json")

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["count"] == 1
    assert metrics["boundary_serialization_errors_total"] == 0
    assert metrics["boundary_serialization_errors"] == {}
    assert metrics["last_payload_bytes"] == len(expected_body)
    assert metrics["windows"]["10s"]["last_payload_bytes"] == len(expected_body)
    total_key = "rest|/probe:get|WireResponse|total|ok"
    render_key = "rest|/probe:get|WireResponse|json_encode|ok"
    route_key = "rest|/probe:get|WireResponse|ok"
    assert metrics["stages"][total_key]["last_payload_bytes"] == len(expected_body)
    assert metrics["stages"][total_key]["budgeted"] is True
    assert metrics["stages"][render_key]["last_payload_bytes"] == len(expected_body)
    assert metrics["routes"][route_key]["count"] == 1
    assert (
        metrics["routes"][route_key]["p99_ms"] == metrics["stages"][total_key]["p99_ms"]
    )

    compact = boundary_metrics.get_boundary_serialization_metrics_compact()
    assert "routes" not in compact
    assert "stages" not in compact
    assert compact["p99_10s_ms"] == metrics["p99_10s_ms"]
    assert compact["p99_60s_ms"] == metrics["p99_60s_ms"]
    assert compact["max_path_p99_ms"] == metrics["max_path_p99_ms"]
    assert compact["boundary_serialization_errors_total"] == 0


def test_rest_boundary_rejects_duplicate_marks() -> None:
    boundary_metrics.reset_boundary_serialization_metrics()
    app = FastAPI()
    app.router.route_class = boundary_metrics.BoundaryMetricsRoute

    @app.get("/duplicate")
    def duplicate(request: Request) -> dict[str, bool]:
        boundary_metrics.mark_rest_response(
            request,
            "/duplicate:get",
            "DictResponse",
            model_duration_ms=0.0,
        )
        boundary_metrics.mark_rest_response(
            request,
            "/duplicate:get",
            "DictResponse",
            model_duration_ms=0.0,
        )
        return {"ok": True}

    with pytest.raises(RuntimeError, match="marked more than once"):
        TestClient(app).get("/duplicate")
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["errors_total"] == 1
    assert metrics["errors"] == {
        "rest|/duplicate:get|DictResponse|local_response|RuntimeError": 1
    }


def test_rest_boundary_counts_only_local_model_validation_and_render_errors() -> None:
    app = FastAPI()
    app.router.route_class = boundary_metrics.BoundaryMetricsRoute

    @app.get("/model-error")
    def model_error() -> dict[str, bool]:
        with boundary_metrics.measure_rest_response_model(
            "/model-error:get", "WireResponse"
        ):
            raise ValueError("intentional model failure")

    @app.get("/validation-error", response_model=WireResponse)
    def validation_error(request: Request) -> dict[str, object]:
        boundary_metrics.mark_rest_response(
            request,
            "/validation-error:get",
            "WireResponse",
            model_duration_ms=0.0,
        )
        return {"optionalValue": None}

    @app.get("/render-error", response_model=ExplodingWireResponse)
    def render_error(request: Request) -> ExplodingWireResponse:
        response = ExplodingWireResponse(value="boom")
        boundary_metrics.mark_rest_response(
            request,
            "/render-error:get",
            "ExplodingWireResponse",
            model_duration_ms=0.0,
        )
        return response

    @app.get("/domain-error")
    def domain_error() -> dict[str, bool]:
        raise RuntimeError("domain failure before the response boundary")

    client = TestClient(app)
    boundary_metrics.reset_boundary_serialization_metrics()

    with pytest.raises(ValueError, match="intentional model failure"):
        client.get("/model-error")
    with pytest.raises(ResponseValidationError):
        client.get("/validation-error")
    with pytest.raises(Exception, match="intentional render failure"):
        client.get("/render-error")
    with pytest.raises(RuntimeError, match="domain failure"):
        client.get("/domain-error")

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["errors_total"] == 3
    assert metrics["boundary_serialization_errors_total"] == 3
    assert metrics["boundary_serialization_errors"] == metrics["errors"]
    assert (
        metrics["errors"][
            "rest|/model-error:get|WireResponse|response_model|ValueError"
        ]
        == 1
    )
    assert (
        metrics["errors"][
            "rest|/validation-error:get|WireResponse|response_model|ResponseValidationError"
        ]
        == 1
    )
    render_errors = {
        key: count
        for key, count in metrics["errors"].items()
        if key.startswith("rest|/render-error:get|ExplodingWireResponse|json_encode|")
    }
    assert sum(render_errors.values()) == 1
    assert not any("domain-error" in key for key in metrics["errors"])
    assert metrics["count"] == 0


def test_rest_boundary_error_cardinality_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(boundary_metrics, "_MAX_ERROR_BUCKETS", 3)
    boundary_metrics.reset_boundary_serialization_metrics()

    for index in range(6):
        with pytest.raises(ValueError):
            with boundary_metrics.measure_rest_response_model(
                f"/model-error-{index}", "WireResponse"
            ):
                raise ValueError(f"failure-{index}")

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["errors_total"] == 6
    assert len(metrics["errors"]) == 3
    assert (
        metrics["errors"]["rest|__overflow__|__overflow__|__overflow__|__overflow__"]
        == 4
    )
