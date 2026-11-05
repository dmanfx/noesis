import importlib.util
from pathlib import Path
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
from noesis.server import scene_prior_api
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis_core.contracts.room_coverage import RoomCoverageResult

spec = importlib.util.spec_from_file_location('coverage_api_fixture', Path(__file__).with_name('test_view_coverage.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

def test_authenticated_domain_map_and_strict_original_binding(tmp_path):
    f=module.fixture(tmp_path); app=FastAPI();app.include_router(scene_prior_api.app.router)
    token='synthetic-domain-coverage-'+ 'q'*48
    app.add_middleware(InternalBearerAuthMiddleware, token=token)
    calibration=SimpleNamespace(snapshot=lambda *_:f.snapshot)
    app.state.scene_prior_runtime_provider=lambda:(f.priors,calibration,{0:'camera-a'})
    path='/api/v1/scene-priors/cameras/camera-a/domain-coverage'
    body={**f.context.binding.model_dump(mode='json'),'height_above_floor_m':1.}
    with TestClient(app) as client:
        assert client.post(path,json=body).status_code==401
        headers={'Authorization':'Bearer '+token}
        response=client.post(path,json=body,headers=headers)
        assert response.status_code==200, response.text
        assert len(RoomCoverageResult.model_validate(response.json()).cells)==4
        assert response.headers['cache-control']=='private, no-store'
        assert client.post(path+'?ignored=true',json=body,headers=headers).status_code==422
        assert client.post(path,json={**body,'height_above_floor_m':'1'},headers=headers).status_code==422
        assert client.post(path,json={**body,'frame_transform_sha256':'f'*64},headers=headers).status_code==409
