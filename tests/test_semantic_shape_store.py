"""Real promoted OBJ + reviewed semantic/anchor SQLite -> separate shapes."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from noesis.server.scene_api import _store
from noesis_core.contracts.semantic_anchor import AnchorWrite
from noesis_core.contracts.semantic_space import SpaceWrite
from noesis_core.contracts.semantic_shape import ShapeWrite
from noesis_core.scene_store import SceneReleaseConflict
from noesis_core.semantic_anchor_store import SemanticAnchorStore
from noesis_core.semantic_space_store import SemanticSpaceStore
from noesis_core.semantic_shape_store import SemanticShapeStore, ShapeLookupMissing, ShapeLookupExpired, REPLAY_US

def uid(): return str(uuid.uuid4())
def unchanged(path): return hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_mtime_ns

def make_source(tmp_path,monkeypatch,*,kind='surface',points=None):
    spec=importlib.util.spec_from_file_location('semantic_shape_scene_fixture',Path(__file__).with_name('test_scene_api.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    client,release=module._client(tmp_path,monkeypatch)
    authored=tmp_path/'virtual-twin'/release['authored_scene_path']
    authored.write_bytes(b'mtllib home.mtl\no synthetic-home\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    release['authored_scene']['sha256']=hashlib.sha256(authored.read_bytes()).hexdigest();release['authored_scene_size_bytes']=authored.stat().st_size
    with client:
        assert client.post('/api/v1/scenes/releases',json=release).status_code==201
        assert client.post('/api/v1/scenes/releases/release-1/promote',json={'actor_id':'fixture','expected_current_release_id':None}).status_code==200
    scenes=_store();semantic=SemanticSpaceStore(scenes);anchor_store=SemanticAnchorStore(scenes);store=SemanticShapeStore(scenes)
    binding=semantic.read().binding.model_dump(mode='json');semantic_id=uid()
    semantic_command={'actor_id':'owner-1','operation_id':uid(),'expected_revision':None,'expected_binding':binding,
        'objects':[{'id':semantic_id,'kind':kind,'label':'Declared object','description':'Synthetic owner declaration','retired':False,'reviewed_binding':binding,'relations':[]}]}
    semantic_result=semantic.replace(SpaceWrite.model_validate(semantic_command));source=anchor_store.read()
    points=points if points is not None else ([(0.,0.,0.),(1.,0.,0.)] if kind=='door' else [(0.,0.,0.),(1.,0.,0.),(1.,0.,1.),(0.,0.,1.)])
    anchor_command={'actor_id':'owner-1','operation_id':uid(),'expected_revision':None,'expected_binding':binding,'expected_artifact':source.artifact.model_dump(mode='json'),
        'expected_semantic_revision':semantic_result.record.revision,'anchors':[{'id':uid(),'semantic_id':semantic_id,'semantic_revision':semantic_result.record.revision,
        'binding':binding,'artifact':source.artifact.model_dump(mode='json'),'declaration':'owner_declared','retired':False,
        'geometry':{'space':'authored_scene_geometry_local','units':'artifact_units','point':point,'normal':None}} for point in points]}
    anchor_result=anchor_store.replace(AnchorWrite.model_validate(anchor_command));snapshot=store.read()
    body={'actor_id':'owner-1','operation_id':uid(),'expected_revision':None,'expected_binding':binding,'expected_artifact':source.artifact.model_dump(mode='json'),
        'expected_semantic_revision':semantic_result.record.revision,'expected_anchor_revision':anchor_result.record.revision,
        'shapes':[{'id':uid(),'semantic_id':semantic_id,'kind':kind,'retired':False,'anchor_ids':[anchor.id for anchor in anchor_result.record.anchors]}]}
    return SimpleNamespace(store=store,scenes=scenes,semantic=semantic,anchor_store=anchor_store,body=body,snapshot=snapshot,release=release,
        semantic_command=semantic_command,anchor_command=anchor_command,path=store.database_path,authored=authored)

@pytest.mark.parametrize('kind',['door','surface'])
def test_saved_exact_segment_polygon_and_independent_readback_preserve_old_registries(tmp_path,monkeypatch,kind):
    f=make_source(tmp_path,monkeypatch,kind=kind);old_semantic=f.semantic.read();old_anchors=f.anchor_store.read();old_scene=f.scenes.current()
    assert f.snapshot.status=='unknown' and len(f.snapshot.point_choices)==len(f.body['shapes'][0]['anchor_ids'])
    result=f.store.replace(f.body);shape=result.record.shapes[0]
    assert shape.anchor_ids==tuple(f.body['shapes'][0]['anchor_ids'])
    assert tuple(vertex.point for vertex in shape.vertices)==tuple(point.geometry.point for point in f.snapshot.point_choices)
    assert shape.units=='artifact_units' and result.review_states[0].status=='reviewed'
    assert SemanticShapeStore(database_path=f.path).read().record==result.record
    assert f.semantic.read()==old_semantic and f.anchor_store.read()==old_anchors and f.scenes.current()==old_scene
    with sqlite3.connect(f.path) as connection: assert connection.execute('SELECT count(*) FROM promotion_history').fetchone()[0]==1

@pytest.mark.parametrize('points',[
    [(0.,0.,0.),(1.,0.,1.),(1.,0.,0.),(0.,0.,1.)],
    [(0.,0.,0.),(1.,0.,0.),(1.,.1,1.),(0.,0.,1.)],
    [(0.,0.,0.),(1.,0.,0.),(2.,0.,0.)],
    [(0.,0.,0.),(1.,0.,0.),(0.,0.,0.)],
])
def test_invalid_geometry_refuses_without_writing_shape_state(tmp_path,monkeypatch,points):
    f=make_source(tmp_path,monkeypatch,points=points);before=unchanged(f.path)
    with pytest.raises(SceneReleaseConflict):f.store.replace(f.body)
    assert unchanged(f.path)==before
    assert f.store.read().status=='unknown'


def test_forged_points_all_revision_cas_and_stable_identity_retirement(tmp_path,monkeypatch):
    f=make_source(tmp_path,monkeypatch);before=unchanged(f.path)
    forged=copy.deepcopy(f.body);forged['shapes'][0]['vertices']=[{'anchor_id':uid(),'point':[9.,9.,9.]}]
    with pytest.raises(ValidationError):f.store.replace(forged)
    forged=copy.deepcopy(f.body);forged['shapes'][0]['anchor_ids'][0]=uid()
    with pytest.raises(SceneReleaseConflict):f.store.replace(forged)
    for field in ('expected_revision','expected_semantic_revision','expected_anchor_revision'):
        wrong=copy.deepcopy(f.body);wrong[field]=uid()
        with pytest.raises(SceneReleaseConflict):f.store.replace(wrong)
    wrong=copy.deepcopy(f.body);wrong['expected_binding']['config_sha256']='0'*64
    with pytest.raises(SceneReleaseConflict):f.store.replace(wrong)
    wrong=copy.deepcopy(f.body);wrong['expected_artifact']['sha256']='0'*64
    with pytest.raises(SceneReleaseConflict):f.store.replace(wrong)
    assert unchanged(f.path)==before
    saved=f.store.replace(f.body)
    retirement=copy.deepcopy(f.body);retirement.update(operation_id=uid(),expected_revision=saved.record.revision);retirement['shapes'][0]['retired']=True
    retired=f.store.replace(retirement)
    revived=copy.deepcopy(retirement);revived.update(operation_id=uid(),expected_revision=retired.record.revision);revived['shapes'][0]['retired']=False
    with pytest.raises(SceneReleaseConflict):f.store.replace(revived)
    assert retired.record.shapes[0]==saved.record.shapes[0].model_copy(update={'retired':True})


def test_stale_anchor_source_can_retire_exact_old_shape_and_original_lookup_is_independent(tmp_path,monkeypatch):
    f=make_source(tmp_path,monkeypatch);saved=f.store.replace(f.body)
    edit=copy.deepcopy(f.anchor_command);edit.update(operation_id=uid(),expected_revision=f.snapshot.anchor_revision);edit['anchors'][0]['geometry']['point']=[.1,0.,0.]
    new_anchors=f.anchor_store.replace(edit);current=f.store.read();assert current.review_states[0].status=='stale'
    retirement=copy.deepcopy(f.body);retirement.update(operation_id=uid(),expected_revision=saved.record.revision,expected_anchor_revision=new_anchors.record.revision);retirement['shapes'][0]['retired']=True
    final=f.store.replace(retirement);assert final.record.shapes[0]==saved.record.shapes[0].model_copy(update={'retired':True})
    before=unchanged(f.path);assert f.store.lookup(f.body)==saved.model_copy(update={'replayed':True});assert unchanged(f.path)==before
    other=copy.deepcopy(f.body);other['actor_id']='other-owner'
    with pytest.raises(ShapeLookupMissing):f.store.lookup(other)
    wrong=copy.deepcopy(f.body);wrong['shapes'][0]['anchor_ids'].reverse()
    with pytest.raises(SceneReleaseConflict):f.store.lookup(wrong)
    expired=SemanticShapeStore(database_path=f.path,now_us=lambda:saved.record.updated_at_us+REPLAY_US)
    with pytest.raises(ShapeLookupExpired):expired.lookup(f.body)
    assert unchanged(f.path)==before


def test_missing_input_read_and_lookup_never_create_database(tmp_path):
    path=tmp_path/'absent'/'scene.sqlite3';store=SemanticShapeStore(database_path=path)
    with pytest.raises(SceneReleaseConflict):store.read()
    assert not path.parent.exists()
