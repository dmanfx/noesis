"""Registered scene-prior route, actual verified inputs and private HTTP auth."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from noesis.server import scene_prior_api
from noesis.server import view_coverage as api
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis_core.contracts.view_coverage import ViewAnalysis, ViewAnalysisContext
from noesis_core.view_coverage import ViewCoverageService

_spec = importlib.util.spec_from_file_location("noesis_view_coverage_fixture", Path(__file__).with_name("test_view_coverage.py"))
_fixture = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fixture)
PATH = "/api/v1/scene-priors/cameras/camera-a/view-analysis"
TOKEN = "synthetic-view-analysis-private-" + "v" * 48
HEADERS = {"Authorization": "Bearer " + TOKEN}


@pytest.fixture
def journey(tmp_path):
    source = _fixture.fixture(tmp_path)
    calibration = SimpleNamespace(snapshot=lambda _source, _camera: source.snapshot)
    app = FastAPI(); app.include_router(scene_prior_api.app.router)
    app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    app.state.scene_prior_runtime_provider = lambda: (source.priors, calibration, {0: "camera-a"})
    with TestClient(app) as client:
        yield SimpleNamespace(source=source, app=app, client=client, calibration=calibration)
    assert api._workers._value == 2


def test_registered_current_context_get_does_no_analysis_then_post_retains_exact_provenance(journey, monkeypatch):
    f = journey
    with monkeypatch.context() as patch:
        patch.setattr(ViewCoverageService, "analyze", lambda *_: pytest.fail("GET cannot analyze or capture points"))
        response = f.client.get(PATH, headers=HEADERS)
    assert response.status_code == 200 and response.headers["cache-control"] == "private, no-store"
    context = ViewAnalysisContext.model_validate(response.json())
    assert context.binding == f.source.context.binding
    body = {**context.binding.model_dump(mode="json"), "points": [{"id":"original-point", "position_m":[.5,1.,1.5]}]}
    result = f.client.post(PATH, json=body, headers=HEADERS)
    assert result.status_code == 200
    value = ViewAnalysis.model_validate(result.json())
    assert value.results[0].projection.pixel == (320.,340.)
    assert value.results[0].projection.optical_visibility == "unknown"
    assert value.provenance.model_dump(exclude={"distortion_applied"}) == context.provenance.model_dump(exclude={"distortion_applied"})
    assert context.provenance.distortion_applied is False
    assert value.provenance.distortion_applied is any(row.projection.distortion_applied for row in value.results)
    assert value.camera_position_m == context.camera_position_m
    assert len(result.content) <= 256 * 1024


def test_private_auth_strict_json_query_camera_and_body_bounds(journey):
    f = journey; body = f.source.request().model_dump(mode="json")
    assert f.client.get(PATH).status_code == 401
    assert f.client.post(PATH,json=body).status_code == 401
    assert f.client.get(PATH+"?guess=1",headers=HEADERS).status_code == 422
    assert f.client.request("GET",PATH,content=b"x",headers=HEADERS).status_code == 422
    assert f.client.post(PATH,content=json.dumps(body),headers={**HEADERS,"Content-Type":"text/plain"}).status_code == 415
    assert f.client.post(PATH,content='{"points":[],"points":[]}',headers={**HEADERS,"Content-Type":"application/json"}).status_code == 422
    assert f.client.post(PATH,content=b" "*(64*1024+1),headers={**HEADERS,"Content-Type":"application/json"}).status_code == 413
    assert f.client.post(PATH,json={**body,"room_id":"guessed"},headers=HEADERS).status_code == 422
    assert f.client.get(PATH.replace("camera-a","camera-b"),headers=HEADERS).status_code == 404
    wrong={**body,"world_frame_revision":"stale-frame"}
    assert f.client.post(PATH,json=wrong,headers=HEADERS).status_code == 409


def test_runtime_owner_replacement_during_work_refuses_without_cached_result(journey, monkeypatch):
    f=journey; original=ViewCoverageService._ray
    def replaced(*args):
        result=original(*args)
        f.app.state.scene_prior_runtime_provider=lambda:(f.source.priors,f.calibration,{0:"camera-a"})
        return result
    monkeypatch.setattr(ViewCoverageService,"_ray",replaced)
    response=f.client.post(PATH,json=f.source.request().model_dump(mode="json"),headers=HEADERS)
    assert response.status_code==409 and response.json()=={"detail":"view_analysis_owner_retired"}


def test_owner_replacement_after_serialization_refuses_both_context_and_analysis_delivery(journey, monkeypatch):
    f=journey; original=api.JSONResponse
    def replaced(*args,**kwargs):
        response=original(*args,**kwargs)
        f.app.state.scene_prior_runtime_provider=lambda:(f.source.priors,f.calibration,{0:"camera-a"})
        return response
    monkeypatch.setattr(api,"JSONResponse",replaced)
    for method in ("get","post"):
        options={"headers":HEADERS}
        if method=="post": options["json"]=f.source.request().model_dump(mode="json")
        response=getattr(f.client,method)(PATH,**options)
        assert response.status_code==409 and response.json()=={"detail":"view_analysis_owner_retired"}


def test_two_worker_admission_is_finite_and_releases_on_invalid_request(journey):
    f=journey
    assert api._workers.acquire(blocking=False) and api._workers.acquire(blocking=False)
    try:
        response=f.client.get(PATH,headers=HEADERS)
        assert response.status_code==503 and response.json()=={"detail":"view_analysis_busy"}
    finally:
        api._workers.release(); api._workers.release()
    assert f.client.post(PATH,json={},headers=HEADERS).status_code==422
    assert f.client.get(PATH,headers=HEADERS).status_code==200
