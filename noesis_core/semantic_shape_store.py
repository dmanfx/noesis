"""Separate reviewed shapes, sharing the real scene/semantic/anchor SQLite CAS."""
from __future__ import annotations
import hashlib
import time
import uuid
from pathlib import Path
import numpy as np
from .contracts.semantic_shape import ShapeInput, ShapeRecord, ShapeResult, ShapeReview, ShapeSnapshot, ShapeWrite, SemanticShape, ShapeVertex
from .semantic_anchor_store import SemanticAnchorStore
from .semantic_space_store import canonical
from .scene_store import SceneReleaseConflict, SceneReleaseStoreError

MAX_BYTES=256*1024
REPLAY_US=24*60*60*1_000_000
MAX_OPERATIONS=128
MAX_STORED_BYTES=32*1024*1024

class ShapeLookupMissing(SceneReleaseStoreError): pass
class ShapeLookupExpired(SceneReleaseStoreError): pass

def validate_geometry(kind, points):
    """Numerical conditioning only; no physical-unit or surface interpretation."""
    values=np.asarray(points,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise SceneReleaseConflict('shape coordinates are invalid')
    centered=values-values.mean(axis=0)
    scale=float(np.max(np.linalg.norm(centered,axis=1)))
    floor=128*np.finfo(float).eps*max(float(np.max(np.abs(values))),scale,np.finfo(float).tiny)
    if scale<=floor: raise SceneReleaseConflict('shape points are coincident or ill-conditioned')
    normalized=centered/scale
    for i in range(len(values)):
        for j in range(i):
            if np.linalg.norm(values[i]-values[j])<=max(floor,scale*1e-9):
                raise SceneReleaseConflict('shape requires distinct saved coordinates')
    if kind=='door': return
    _,singular,basis=np.linalg.svd(normalized,full_matrices=False)
    if singular[1]<=singular[0]*1e-8: raise SceneReleaseConflict('shape polygon is degenerate or ill-conditioned')
    if singular[2]>singular[0]*1e-8: raise SceneReleaseConflict('shape polygon is not planar')
    plane=normalized@basis[:2].T
    area=abs(sum(np.cross(plane[(i+1)%len(plane)],plane[i]) for i in range(len(plane))))*.5
    if area<=1e-8: raise SceneReleaseConflict('shape polygon has degenerate area')
    def cross(a,b,c): return float(np.cross(b-a,c-a))
    def on(a,b,p): return abs(cross(a,b,p))<=1e-10 and np.all(p>=np.minimum(a,b)-1e-10) and np.all(p<=np.maximum(a,b)+1e-10)
    def intersects(a,b,c,d):
        x,y,z,w=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
        return (x*y<0 and z*w<0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)
    n=len(plane)
    for i in range(n):
        for j in range(i+1,n):
            if j==i+1 or i==0 and j==n-1: continue
            if intersects(plane[i],plane[(i+1)%n],plane[j],plane[(j+1)%n]):
                raise SceneReleaseConflict('shape polygon self-intersects')

class SemanticShapeStore:
    def __init__(self, scenes=None, *, database_path:Path|None=None, now_us=None):
        self.scenes=scenes
        self.anchors=SemanticAnchorStore(scenes,database_path=database_path)
        self.semantic=self.anchors.semantic
        self.database_path=self.semantic.database_path
        self.now_us=now_us or (lambda:time.time_ns()//1000)

    def _source(self,connection):
        binding,artifact,semantic,release=self.anchors._source(connection)
        anchors=self.anchors._record(connection)
        reviews={row.id:row.status for row in self.anchors._reviews(anchors,binding,artifact,semantic)}
        choices=tuple(anchor for anchor in anchors.anchors if reviews[anchor.id]=='reviewed')
        return binding,artifact,semantic,anchors,choices,release

    def _record(self,connection):
        if not self.semantic._has_table(connection,'semantic_shape_state'):
            return ShapeRecord(revision=None,updated_at_us=None,actor_id=None,shapes=())
        row=connection.execute('SELECT payload_json FROM semantic_shape_state WHERE singleton=1').fetchone()
        if row is None: return ShapeRecord(revision=None,updated_at_us=None,actor_id=None,shapes=())
        if not isinstance(row[0],str) or len(row[0].encode())>MAX_BYTES: raise SceneReleaseStoreError('stored shape registry exceeds byte limit')
        record=ShapeRecord.model_validate_json(row[0])
        if canonical(record.model_dump(mode='json'))!=row[0]: raise SceneReleaseStoreError('stored shape registry is not canonical')
        return record

    @staticmethod
    def _reviews(record,binding,artifact,semantic,anchors,choices):
        objects={item.id:item for item in semantic.objects}; points={item.id:item for item in choices}
        rows=[]
        for shape in record.shapes:
            item=objects.get(shape.semantic_id)
            status=('retired' if shape.retired else 'missing' if item is None or any(key not in points for key in shape.anchor_ids) else
                'stale' if item.retired or item.reviewed_binding!=binding or item.kind!=shape.kind or shape.binding!=binding or shape.artifact!=artifact
                or shape.semantic_revision!=semantic.revision or shape.anchor_revision!=anchors.revision
                or any(points[vertex.anchor_id].semantic_id!=shape.semantic_id or points[vertex.anchor_id].geometry.point!=vertex.point for vertex in shape.vertices) else 'reviewed')
            rows.append(ShapeReview(id=shape.id,status=status))
        return tuple(rows)

    @staticmethod
    def _command(command):
        value=ShapeWrite.model_validate(command.model_dump(mode='json') if isinstance(command,ShapeWrite) else command)
        if len(canonical(value.model_dump(mode='json')).encode())>MAX_BYTES: raise SceneReleaseStoreError('shape command exceeds byte limit')
        return value

    @staticmethod
    def _fingerprint(command): return hashlib.sha256(canonical(command.model_dump(mode='json')).encode()).hexdigest()

    def _lookup(self,connection,command,now):
        if not self.semantic._has_table(connection,'semantic_shape_operations'): return None
        row=connection.execute('SELECT * FROM semantic_shape_operations WHERE actor_id=? AND operation_id=?',(command.actor_id,command.operation_id)).fetchone()
        if row is None: return None
        if row['fingerprint']!=self._fingerprint(command): raise SceneReleaseConflict('original shape operation fingerprint differs')
        if type(row['expires_at_us']) is not int or row['expires_at_us']<1: raise SceneReleaseStoreError('stored shape operation expiry invalid')
        if row['expires_at_us']<=now: raise ShapeLookupExpired('original shape operation retention expired')
        raw=row['result_json']
        if not isinstance(raw,str) or len(raw.encode())>MAX_BYTES: raise SceneReleaseStoreError('stored shape operation exceeds byte limit')
        result=ShapeResult.model_validate_json(raw)
        inputs=tuple(ShapeInput(**{key:getattr(shape,key) for key in ShapeInput.model_fields}) for shape in result.record.shapes)
        reviews=tuple(ShapeReview(id=shape.id,status='retired' if shape.retired else 'reviewed') for shape in result.record.shapes)
        if (canonical(result.model_dump(mode='json'))!=raw or result.replayed or result.operation_id!=command.operation_id
            or result.binding!=command.expected_binding or result.artifact!=command.expected_artifact
            or result.semantic_revision!=command.expected_semantic_revision or result.anchor_revision!=command.expected_anchor_revision
            or result.record.actor_id!=command.actor_id or result.record.revision is None or inputs!=command.shapes or result.review_states!=reviews):
            raise SceneReleaseStoreError('stored shape operation does not match original')
        for shape in result.record.shapes:
            if not shape.retired and (shape.binding!=result.binding or shape.artifact!=result.artifact or shape.semantic_revision!=result.semantic_revision or shape.anchor_revision!=result.anchor_revision):
                raise SceneReleaseStoreError('stored shape source does not match original')
        return result.model_copy(update={'replayed':True})

    def read(self):
        with self.semantic._read_connection() as connection:
            if connection is None: raise SceneReleaseConflict('no scene release is promoted')
            connection.execute('BEGIN')
            binding,artifact,semantic,anchors,choices,_=self._source(connection); record=self._record(connection)
            result=ShapeSnapshot(binding=binding,artifact=artifact,semantic_revision=semantic.revision,anchor_revision=anchors.revision,
                point_choices=choices,record=record,status='unknown' if record.revision is None else 'available',
                review_states=self._reviews(record,binding,artifact,semantic,anchors,choices))
            if len(canonical(result.model_dump(mode='json')).encode())>MAX_BYTES: raise SceneReleaseStoreError('shape response exceeds byte limit')
            connection.execute('COMMIT'); return result

    def lookup(self,command):
        command=self._command(command)
        with self.semantic._read_connection() as connection: result=self._lookup(connection,command,self.now_us())
        if result is None: raise ShapeLookupMissing('original shape operation unavailable; no write was attempted')
        return result

    def replace(self,command):
        command=self._command(command)
        if self.scenes is None: raise SceneReleaseStoreError('shape mutation requires canonical scene store')
        now=self.now_us()
        with self.scenes._connection() as connection:
            connection.execute('PRAGMA busy_timeout=2000'); connection.execute('BEGIN IMMEDIATE')
            original=self._lookup(connection,command,now)
            if original is not None: connection.execute('COMMIT'); return original
            binding,artifact,semantic,anchors,choices,release=self._source(connection); previous=self._record(connection)
            if (binding!=command.expected_binding or artifact!=command.expected_artifact or semantic.revision!=command.expected_semantic_revision
                or anchors.revision!=command.expected_anchor_revision or previous.revision!=command.expected_revision):
                raise SceneReleaseConflict('shape scene/artifact/semantic/anchor/registry revision changed; reload before reviewing')
            old_by_id={shape.id:shape for shape in previous.shapes}; by_id={shape.id:shape for shape in command.shapes}
            for old in previous.shapes:
                shape=by_id.get(old.id)
                if shape is None or shape.semantic_id!=old.semantic_id or shape.kind!=old.kind or old.retired and not shape.retired:
                    raise SceneReleaseConflict('stable shape IDs retain semantic identity, kind and retirement')
            objects={item.id:item for item in semantic.objects}; points={item.id:item for item in choices}; saved=[]
            for shape in command.shapes:
                if shape.retired:
                    old=old_by_id.get(shape.id)
                    if old is None or shape.anchor_ids!=old.anchor_ids: raise SceneReleaseConflict('retirement preserves exact existing shape')
                    saved.append(old.model_copy(update={'retired':True})); continue
                item=objects.get(shape.semantic_id)
                if item is None or item.retired or item.reviewed_binding!=binding or item.kind!=shape.kind or semantic.revision is None or anchors.revision is None:
                    raise SceneReleaseConflict('shape requires current reviewed active semantic object of same kind')
                selected=[points.get(key) for key in shape.anchor_ids]
                if any(point is None or point.semantic_id!=shape.semantic_id for point in selected):
                    raise SceneReleaseConflict('shape requires exact current saved reviewed points for same semantic ID')
                vertices=tuple(ShapeVertex(anchor_id=point.id,point=point.geometry.point) for point in selected)
                validate_geometry(shape.kind,[vertex.point for vertex in vertices])
                saved.append(SemanticShape(**shape.model_dump(),declaration='owner_declared',space='authored_scene_geometry_local',units='artifact_units',
                    binding=binding,artifact=artifact,semantic_revision=semantic.revision,anchor_revision=anchors.revision,vertices=vertices))
            if any(not shape.retired for shape in saved): self.scenes.read_authored_scene(release)
            record=ShapeRecord(revision=str(uuid.uuid4()),updated_at_us=now,actor_id=command.actor_id,shapes=tuple(saved))
            result=ShapeResult(operation_id=command.operation_id,binding=binding,artifact=artifact,semantic_revision=semantic.revision,anchor_revision=anchors.revision,
                record=record,replayed=False,review_states=self._reviews(record,binding,artifact,semantic,anchors,choices))
            payload,raw=canonical(record.model_dump(mode='json')),canonical(result.model_dump(mode='json'))
            if len(raw.encode())>MAX_BYTES: raise SceneReleaseStoreError('shape response exceeds byte limit')
            connection.execute('CREATE TABLE IF NOT EXISTS semantic_shape_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload_json TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS semantic_shape_operations (actor_id TEXT NOT NULL, operation_id TEXT NOT NULL, fingerprint TEXT NOT NULL, expires_at_us INTEGER NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY(actor_id,operation_id))')
            connection.execute('DELETE FROM semantic_shape_operations WHERE expires_at_us<=?',(now,))
            count,size=connection.execute('SELECT count(*),coalesce(sum(length(CAST(result_json AS BLOB))),0) FROM semantic_shape_operations').fetchone()
            if count>=MAX_OPERATIONS or size+len(raw.encode())+len(payload.encode())>MAX_STORED_BYTES: raise SceneReleaseConflict('shape operation retention full')
            connection.execute('INSERT OR REPLACE INTO semantic_shape_state VALUES(1,?)',(payload,))
            connection.execute('INSERT INTO semantic_shape_operations VALUES(?,?,?,?,?)',(command.actor_id,command.operation_id,self._fingerprint(command),now+REPLAY_US,raw))
            if self._record(connection)!=record or self._lookup(connection,command,now)!=result.model_copy(update={'replayed':True}): raise SceneReleaseStoreError('shape exact write readback failed')
            connection.execute('COMMIT')
        # Independent read-only original readback, never a later current snapshot.
        if self.lookup(command)!=result.model_copy(update={'replayed':True}): raise SceneReleaseStoreError('shape independent original readback failed')
        return result
