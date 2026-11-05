"""Registered private HTTP with real scene selector/semantic/anchor SQLite."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from noesis.server.scene_api import router
from noesis.server import semantic_shape as api
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis_core.semantic_shape_store import SemanticShapeStore,REPLAY_US
from noesis_core.scene_store import SceneReleaseConflict

spec=importlib.util.spec_from_file_location('semantic_shape_store_fixture',Path(__file__).with_name('test_semantic_shape_store.py'))
source_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(source_module)
CURRENT='/api/v1/scenes/current/semantic-shapes';LOOKUP='/api/v1/scenes/semantic-shapes/lookup'
TOKEN='synthetic-shape-internal-'+'s'*48
HEADERS={'Authorization':'Bearer '+TOKEN}

@pytest.fixture
def journey(tmp_path,monkeypatch):
    source=source_module.make_source(tmp_path,monkeypatch)
    app=FastAPI();app.include_router(router);app.add_middleware(InternalBearerAuthMiddleware,token=TOKEN)
    with TestClient(app) as client: yield SimpleNamespace(source=source,client=client)
    assert api._workers._value==2


def test_registered_owner_bearer_exact_save_readback_and_lost_reply_original_recovery(journey):
    f=journey;before=source_module.unchanged(f.source.path)
    response=f.client.get(CURRENT,headers=HEADERS)
    assert response.status_code==200 and response.headers['cache-control']=='private, no-store'
    assert response.json()==f.source.snapshot.model_dump(mode='json') and source_module.unchanged(f.source.path)==before
    result=f.client.put(CURRENT,json=f.source.body,headers=HEADERS)
    assert result.status_code==200,result.text
    saved=result.json();assert saved['review_states'][0]['status']=='reviewed' and saved['replayed'] is False
    assert f.client.get(CURRENT,headers=HEADERS).json()['record']==saved['record']
    before=source_module.unchanged(f.source.path)
    assert f.client.post(LOOKUP,json=f.source.body,headers=HEADERS).json()=={**saved,'replayed':True}
    assert f.client.put(CURRENT,json=f.source.body,headers=HEADERS).json()=={**saved,'replayed':True}
    assert source_module.unchanged(f.source.path)==before
    forged=copy.deepcopy(f.source.body);forged['actor_id']='different-owner'
    assert f.client.post(LOOKUP,json=forged,headers=HEADERS).status_code==404
    assert source_module.unchanged(f.source.path)==before


def test_auth_strict_bodies_query_forged_coordinates_and_worker_bound(journey):
    f=journey;before=source_module.unchanged(f.source.path)
    assert f.client.get(CURRENT).status_code==401
    assert f.client.put(CURRENT,json=f.source.body).status_code==401
    assert f.client.post(LOOKUP,json=f.source.body).status_code==401
    assert f.client.get(CURRENT+'?guess=1',headers=HEADERS).status_code==422
    assert f.client.request('GET',CURRENT,content=b'x',headers=HEADERS).status_code==422
    assert f.client.put(CURRENT,content='{}',headers={**HEADERS,'Content-Type':'text/plain'}).status_code==415
    assert f.client.put(CURRENT,content='{"shapes":[],"shapes":[]}',headers={**HEADERS,'Content-Type':'application/json'}).status_code==422
    assert f.client.put(CURRENT,content=b' '*(256*1024+1),headers={**HEADERS,'Content-Type':'application/json'}).status_code==413
    forged=copy.deepcopy(f.source.body);forged['shapes'][0]['point']=[9,9,9]
    assert f.client.put(CURRENT,json=forged,headers=HEADERS).status_code==422
    assert api._workers.acquire(blocking=False) and api._workers.acquire(blocking=False)
    try: assert f.client.get(CURRENT,headers=HEADERS).status_code==503
    finally: api._workers.release();api._workers.release()
    assert source_module.unchanged(f.source.path)==before


def test_current_source_change_refuses_write_but_expired_original_lookup_never_revalidates_source(journey,monkeypatch):
    f=journey;saved=f.client.put(CURRENT,json=f.source.body,headers=HEADERS).json()
    changed=copy.deepcopy(f.source.anchor_command);changed.update(operation_id=source_module.uid(),expected_revision=f.source.snapshot.anchor_revision)
    changed['anchors'][0]['geometry']['point']=[.2,0.,0.];f.source.anchor_store.replace(changed)
    stale=copy.deepcopy(f.source.body);stale.update(operation_id=source_module.uid(),expected_revision=saved['record']['revision'])
    before=source_module.unchanged(f.source.path)
    assert f.client.put(CURRENT,json=stale,headers=HEADERS).status_code==409
    assert f.client.post(LOOKUP,json=f.source.body,headers=HEADERS).json()=={**saved,'replayed':True}
    late=saved['record']['updated_at_us']+REPLAY_US
    monkeypatch.setattr(api,'_read_store',lambda:SemanticShapeStore(database_path=f.source.path,now_us=lambda:late))
    assert f.client.post(LOOKUP,json=f.source.body,headers=HEADERS).status_code==410
    assert source_module.unchanged(f.source.path)==before


def test_canonical_scene_transaction_allows_exactly_one_cas_winner(journey):
    f=journey;commands=[{**copy.deepcopy(f.source.body),'operation_id':source_module.uid()} for _ in range(2)]
    def write(command):
        try:return f.source.store.replace(command).record.revision
        except SceneReleaseConflict:return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as workers: values=list(workers.map(write,commands))
    assert values.count('conflict')==1 and len(set(values))==2
