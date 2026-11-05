"""Bounded scalar continuity and off-callback owner selection persistence."""
from __future__ import annotations
import hashlib,json,secrets,sqlite3,threading,time,uuid
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import quote
from .contracts.object_tracks import *
from .private_paths import prepare_private_writable_file,validate_private_file

class ObjectTrackError(RuntimeError): pass
class ObjectTrackConflict(ObjectTrackError): pass
class ObjectTrackMissing(ObjectTrackError): pass


def canonical(value):
    data=value.model_dump(mode='json') if hasattr(value,'model_dump') else value
    raw=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    if len(raw.encode())>MAX_BYTES: raise ObjectTrackError('object_track_byte_limit')
    return raw

def key_text(key): return canonical(key)

class ObjectTrackService:
    def __init__(self,run_id,database_path,*,monotonic=time.monotonic,clock_us=lambda:time.time_ns()//1000):
        self.run_id=run_id;self.path=Path(database_path);self._mono=monotonic;self._clock=clock_us
        self._lock=threading.RLock();self._closed=False;self._tracks={};self._sources={};self._drop_generation=0;self._reported_drop=0;self._untracked=0;self._last_persist=-float('inf')
        self.spatial_sampler=None;self._sampled={};self._active={};self._terminated=set();self._spatial={};self._generation=0
        path=prepare_private_writable_file(self.path,label='object track selections')
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS object_track_selections(actor TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS object_track_reviews(actor TEXT,operation TEXT,token TEXT,deadline INTEGER,payload TEXT,PRIMARY KEY(actor,operation))')
            db.execute('CREATE TABLE IF NOT EXISTS object_track_operations(actor TEXT,operation TEXT,fingerprint TEXT,deadline INTEGER,payload TEXT,PRIMARY KEY(actor,operation))')
        validate_private_file(path,label='object track selections')
        with self._db() as db:self._selected_keys=frozenset(key_text(item.key) for actor,payload in db.execute('SELECT actor,payload FROM object_track_selections').fetchall() for item in ObjectSelectionRecord.model_validate_json(payload).selections if not item.retired)
    @contextmanager
    def _db(self):
        connection=sqlite3.connect(self.path,timeout=2)
        try:
            connection.execute('PRAGMA synchronous=FULL')
            with connection:yield connection
        finally:connection.close()
    def _now(self):
        now=self._clock()
        if type(now) is not int or not 0<now<=2**53-1:raise ObjectTrackError('object_track_clock_invalid')
        return now
    def _record(self,db,actor):
        row=db.execute('SELECT payload FROM object_track_selections WHERE actor=?',(actor,)).fetchone()
        record=ObjectSelectionRecord.model_validate_json(row[0]) if row else ObjectSelectionRecord(revision=None,updated_at_us=None,selections=())
        return record.model_copy(update={'selections':tuple(self._latest(item) for item in record.selections)})
    def _latest(self,item):
        live=self._tracks.get(key_text(item.key))
        if live and not item.retired and live[1].observed_at_us>=item.last_seen.observed_at_us:
            item=item.model_copy(update={'last_seen':live[1]})
        state=self._state(item.key,item.last_seen,item.state)
        return item.model_copy(update={'state':state,'spatial':self._spatial.get(key_text(item.key)) or item.spatial,'spatial_reason':'measured_surface' if self._spatial.get(key_text(item.key)) or item.spatial else item.last_seen.spatial_reason,
            'termination_reason':'native_termination_observed' if state=='terminated' else 'native_termination_available' if self._sources.get(item.key.camera_id,{}).get('termination_available') else item.termination_reason})
    def _write(self,db,actor,record):
        db.execute('INSERT OR REPLACE INTO object_track_selections VALUES (?,?)',(actor,canonical(record)))
    def _capacity(self,db):
        for table in ('object_track_reviews','object_track_operations'):
            db.execute('DELETE FROM '+table+' WHERE deadline<=?',(self._now(),))
            if db.execute('SELECT count(*) FROM '+table).fetchone()[0]>=MAX_OPERATIONS:raise ObjectTrackConflict('object_track_retention_full')
        if db.execute('SELECT count(*) FROM object_track_selections').fetchone()[0]>64:raise ObjectTrackConflict('object_track_actor_limit')
        size=sum(db.execute('SELECT coalesce(sum(length(CAST(payload AS BLOB))),0) FROM '+table).fetchone()[0] for table in ('object_track_selections','object_track_reviews','object_track_operations'))
        if size>32*1024*1024:raise ObjectTrackConflict('object_track_storage_limit')
    def _state(self,key,last_seen,prior=None):
        if self._closed:return 'runtime_unavailable'
        if key.run_id!=self.run_id:return 'run_replaced'
        if key_text(key) in self._terminated or prior=='terminated':return 'terminated'
        source=self._sources.get(key.camera_id)
        if source is None or self._mono()-source['accepted']>=5:return 'source_lost'
        if (key.source_id,key.source_epoch)!=(source['source_id'],source['epoch']):return 'source_replaced'
        if self._reported_drop>source['drop'] or self._active.get((key.camera_id,key.native_id), (None,None))[0]!=key_text(key):return 'continuity_gap'
        live=self._tracks.get(key_text(key))
        return 'observed' if live and live[1].frame_id==source['frame'] else 'not_observed'
    def note_drop(self,generation):
        # Callback-safe scalar invalidation: no lock, allocation, DB or wait.
        self._reported_drop=generation
    def ingest(self,capture):
        """Called by ObjectObservationService's existing bounded worker only."""
        with self._lock:
            if self._closed:return
            old=self._sources.get(capture.camera_id);at=self._mono()
            if old and capture.source_id==old['source_id'] and (capture.source_epoch,capture.frame_id)<=(old['epoch'],old['frame']):return
            segment=0 if old is None else old['segment']
            if old and (capture.generation!=old['drop'] or capture.source_epoch!=old['epoch'] or capture.source_id!=old['source_id'] or capture.accepted-old['accepted']>=5 or capture.frame_id!=old['frame']+1):segment+=1
            self._drop_generation=capture.generation;self._untracked=capture.untracked
            if capture.camera_id not in self._sources and len(self._sources)>=16:
                oldest=min(self._sources,key=lambda name:self._sources[name]['accepted']);del self._sources[oldest]
            self._sources[capture.camera_id]={'epoch':capture.source_epoch,'source_id':capture.source_id,'frame':capture.frame_id,'accepted':capture.accepted,'segment':segment,'drop':capture.generation,'termination_available':getattr(capture,'termination_available',False)}
            observed,captured,_status,pts=capture.temporal
            changed={};sample_budget=4
            for ordinal,native,class_id,label,confidence,tracker,bbox in capture.track_rows:
                active=self._active.get((capture.camera_id,native))
                if active is None or active[1]!=(capture.source_id,capture.source_epoch,segment) or active[0] in self._terminated:
                    self._generation+=1
                    key=NativeObjectKey(run_id=self.run_id,camera_id=capture.camera_id,source_id=capture.source_id,source_epoch=capture.source_epoch,continuity_generation=self._generation,native_id=native)
                    self._active[(capture.camera_id,native)]=(key_text(key),(capture.source_id,capture.source_epoch,segment))
                else:key=NativeObjectKey.model_validate_json(active[0])
                spatial=None;reason='nonperson_spatial_measurement_not_admitted'
                # Sampling is off the callback and limited to explicitly saved
                # tracks. Selection itself never enables inference or polling.
                selected=key_text(key) in getattr(self,'_selected_keys',())
                if self.spatial_sampler is not None and selected:
                    reason='sampling_budget'
                    if sample_budget and at-self._sampled.get(key_text(key),-float('inf'))>=1:
                        sample_budget-=1;self._sampled[key_text(key)]=at
                        try:spatial,reason=self.spatial_sampler.sample(capture,bbox)
                        except Exception:reason='sampling_failed'
                room=getattr(capture,'rooms',{}).get(native)
                prefix=':'.join(quote(str(v),safe='') for v in (self.run_id,capture.camera_id,capture.source_id,capture.source_epoch,capture.frame_id))
                sample=ObjectTrackSample(observation_id=f'{prefix}:{ordinal}',frame_id=capture.frame_id,observed_at_us=observed,captured_at_us=captured,media_pts_ns=pts,
                    image_width=capture.width,image_height=capture.height,source_image_width=capture.source_width,source_image_height=capture.source_height,
                    class_id=class_id,label=label,detector_confidence=confidence,tracker_confidence=tracker,evidence='tracker_only' if confidence is None else 'detector_and_tracker',
                    bbox=dict(zip(('left','top','width','height'),bbox)),room_id=room,room_source='nvdsanalytics_roi' if room else None,spatial=spatial,spatial_reason=reason)
                text=key_text(key);self._tracks[text]=(key,sample,capture.accepted);changed[text]=sample
                if spatial is not None:self._spatial[text]=spatial
            for native,class_id in getattr(capture,'terminations',()):
                active=self._active.get((capture.camera_id,native))
                if active and active[1]==(capture.source_id,capture.source_epoch,segment):self._terminated.add(active[0])
            self._tracks={k:v for k,v in self._tracks.items() if at-v[2]<5}
            while len(self._active)>MAX_TRACKS:self._active.pop(next(iter(self._active)))
            while len(self._terminated)>MAX_TRACKS:self._terminated.pop()
            self._sampled={k:v for k,v in self._sampled.items() if at-v<10}
            self._spatial={k:v for k,v in self._spatial.items() if k in self._tracks}
            while len(self._tracks)>MAX_TRACKS:self._tracks.pop(next(iter(self._tracks)))
            # Batch durable latest samples at most once per second. Reads join
            # the exact current scalar cache; no SQLite I/O enters the callback.
            if at-self._last_persist<1 and not getattr(capture,'terminations',()):return
            self._last_persist=at
            with self._db() as db:
                for actor,payload in db.execute('SELECT actor,payload FROM object_track_selections').fetchall():
                    record=ObjectSelectionRecord.model_validate_json(payload)
                    rows=tuple(self._latest(item) for item in record.selections)
                    if rows!=record.selections:self._write(db,actor,record.model_copy(update={'selections':rows}))
    def snapshot(self):
        with self._lock:
            rows=[];used=2048;omitted=0
            for key,sample,_ in self._tracks.values():
                state=self._state(key,sample)
                row=ObjectTrack(key=key,last_seen=sample,state=state,spatial=self._spatial.get(key_text(key)),spatial_reason='measured_surface' if key_text(key) in self._spatial else sample.spatial_reason,termination_reason='native_termination_observed' if state=='terminated' else 'native_termination_available' if self._sources.get(key.camera_id,{}).get('termination_available') else 'native_nonperson_termination_not_exposed');size=len(row.model_dump_json().encode())
                if used+size>MAX_BYTES:omitted+=1;continue
                rows.append(row);used+=size
            reason='closed' if self._closed else 'queue_dropped' if self._reported_drop>self._drop_generation else 'native_tracker_evidence' if rows else 'native_id_unavailable' if self._untracked else 'no_input'
            return ObjectTracksSnapshot(run_id=self.run_id,status='unknown' if self._closed or not rows or reason=='queue_dropped' else 'partial' if omitted or any(t.state!='observed' for t in rows) else 'available',reason=reason,generated_at_us=self._now(),dropped_frames=self._reported_drop,untracked_count=self._untracked,omitted_count=omitted,tracks=tuple(rows))
    def read(self,actor):
        with self._lock,self._db() as db:
            record=self._record(db,actor)
            record=record.model_copy(update={'selections':tuple(item.model_copy(update={'state':self._state(item.key,item.last_seen,item.state)}) for item in record.selections)})
            return ObjectSelectionsSnapshot(record=record)
    def prepare(self,command):
        command=ObjectSelectionCommand.model_validate(command)
        with self._lock,self._db() as db:
            db.execute('BEGIN IMMEDIATE');self._capacity(db);now=self._now();record=self._record(db,command.actor_id)
            if self._closed or record.revision!=command.expected_revision:raise ObjectTrackConflict('object_track_revision_or_owner_changed')
            old=next((item for item in record.selections if item.id==command.selection.id),None)
            if command.selection.observation_id is None:
                if old is None:raise ObjectTrackMissing('object_track_selection_missing')
                item=old.model_copy(update={'label':command.selection.label,'retired':command.selection.retired})
            else:
                if command.selection.retired:raise ObjectTrackConflict('object_track_new_retired_selection')
                match=next(((key,sample) for key,sample,_ in self._tracks.values() if sample.observation_id==command.selection.observation_id and self._state(key,sample)=='observed'),None)
                if match is None:raise ObjectTrackConflict('object_track_original_not_current')
                item=ObjectSelection(id=command.selection.id,label=command.selection.label,retired=False,key=match[0],original=match[1],last_seen=match[1],state='observed')
            if old and old.retired and not item.retired:raise ObjectTrackConflict('object_track_selection_retired')
            if not old and len(record.selections)>=MAX_SELECTIONS:raise ObjectTrackConflict('object_track_selection_limit')
            review=ObjectSelectionReview(review_token=secrets.token_urlsafe(32),command=command,selection=item,expires_at_us=now+120000000,remaining_ms=120000)
            db.execute('INSERT OR REPLACE INTO object_track_reviews VALUES (?,?,?,?,?)',(command.actor_id,command.operation_id,review.review_token,review.expires_at_us,canonical(review)))
            return review.model_copy(update={'remaining_ms':max(1,(review.expires_at_us-self._now())//1000)})
    def _lookup(self,db,command):
        row=db.execute('SELECT fingerprint,deadline,payload FROM object_track_operations WHERE actor=? AND operation=?',(command.actor_id,command.operation_id)).fetchone()
        if not row:raise ObjectTrackMissing('object_track_original_missing')
        if row[0]!=hashlib.sha256(canonical(command).encode()).hexdigest():raise ObjectTrackConflict('object_track_operation_conflict')
        if row[1]<=self._now():raise ObjectTrackMissing('object_track_original_expired')
        return ObjectSelectionResult.model_validate_json(row[2]).model_copy(update={'replayed':True})
    def lookup(self,command):
        command=ObjectSelectionCommand.model_validate(command)
        with self._db() as db:return self._lookup(db,command)
    def confirm(self,confirmation):
        confirmation=ObjectSelectionConfirm.model_validate(confirmation);command=confirmation.command
        with self._lock,self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            try:return self._lookup(db,command)
            except ObjectTrackMissing:pass
            self._capacity(db);record=self._record(db,command.actor_id)
            row=db.execute('SELECT token,deadline,payload FROM object_track_reviews WHERE actor=? AND operation=?',(command.actor_id,command.operation_id)).fetchone()
            if not row or row[0]!=confirmation.review_token or row[1]<=self._now():raise ObjectTrackConflict('object_track_review_expired')
            review=ObjectSelectionReview.model_validate_json(row[2])
            if canonical(review.command)!=canonical(command) or record.revision!=command.expected_revision:raise ObjectTrackConflict('object_track_review_changed')
            item=review.selection
            if self._closed or item.key.run_id!=self.run_id:raise ObjectTrackConflict('object_track_owner_retired')
            if command.selection.observation_id is not None:
                state=self._state(item.key,item.original)
                if state not in ('observed','not_observed'):raise ObjectTrackConflict('object_track_continuity_retired')
                live=self._tracks.get(key_text(item.key))
                if live:item=item.model_copy(update={'last_seen':live[1],'state':state})
            rows=tuple([v for v in record.selections if v.id!=item.id]+[item]);record=ObjectSelectionRecord(revision=str(uuid.uuid4()),updated_at_us=self._now(),selections=rows)
            self._write(db,command.actor_id,record)
            self._selected_keys=frozenset(key_text(item.key) for actor,payload in db.execute('SELECT actor,payload FROM object_track_selections').fetchall() for item in ObjectSelectionRecord.model_validate_json(payload).selections if not item.retired)
            result=ObjectSelectionResult(operation_id=command.operation_id,record=record,replayed=False)
            db.execute('DELETE FROM object_track_reviews WHERE actor=? AND operation=?',(command.actor_id,command.operation_id))
            db.execute('INSERT INTO object_track_operations VALUES (?,?,?,?,?)',(command.actor_id,command.operation_id,hashlib.sha256(canonical(command).encode()).hexdigest(),self._now()+REPLAY_US,canonical(result)))
            self._capacity(db)
        # Independent connection reads the exact stored original after COMMIT.
        original=self.lookup(command)
        return original.model_copy(update={'replayed':False})
    def close(self):
        with self._lock:
            if self._closed:return
            with self._db() as db:
                for actor,payload in db.execute('SELECT actor,payload FROM object_track_selections').fetchall():
                    record=ObjectSelectionRecord.model_validate_json(payload)
                    self._write(db,actor,record.model_copy(update={'selections':tuple(self._latest(item) for item in record.selections)}))
            self._closed=True


def install_object_track_owner(transaction,pipeline,app,run_id,database_path):
    service=transaction.acquire('object_tracks',lambda:ObjectTrackService(run_id,database_path),lambda value:value.close())
    def bind():
        pipeline.object_track_service=service
        if app is not None:app.state.object_track_service=service
    def clear():
        if getattr(pipeline,'object_track_service',None) is service:del pipeline.object_track_service
        if app is not None and getattr(app.state,'object_track_service',None) is service:del app.state.object_track_service
    transaction.bind('object_track_binding',bind,clear)
    return service
