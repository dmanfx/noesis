from __future__ import annotations

import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest


def work(hooks, source=0, frame=1):
    return hooks._TrackingPublicationWork(source_id=source, camera_id=f'camera-{source}',
        frame_id=frame, observed_at_us=1_000_000+frame, now_ts=100+frame/30,
        timestamp_us=2_000_000+frame, tracks=[], footpoints=[], temporal_contract={})


@pytest.mark.parametrize('value', [None, '0', 'true', 'False'])
def test_sampler_default_off_is_literal_opt_in(monkeypatch, caplog, value):
    from noesis.pipelines import hooks
    if value is None:
        monkeypatch.delenv(hooks._PublicationWorkerDiagnostics.ENV, raising=False)
    else:
        monkeypatch.setenv(hooks._PublicationWorkerDiagnostics.ENV, value)
    worker = hooks._TrackingPublicationWorker(SimpleNamespace())
    worker.shutdown(timeout_s=2)
    assert worker.diagnostics is None and worker.diagnostics_summary is None
    assert 'diagnostic summary' not in caplog.text


def test_actual_worker_sampling_preserves_fifo_and_logs_once_after_close(monkeypatch, caplog):
    from noesis.pipelines import hooks
    monkeypatch.setenv(hooks._PublicationWorkerDiagnostics.ENV, '1')
    caplog.set_level(logging.INFO, logger=hooks.__name__)
    started, release = threading.Event(), threading.Event()
    events = []
    def publish(row):
        private_local_must_not_be_recorded = 'SECRET_LOCAL_PAYLOAD'
        events.append((row.source_id, row.frame_id))
        if len(events) == 1:
            started.set(); assert release.wait(2)
        assert private_local_must_not_be_recorded
    worker = hooks._TrackingPublicationWorker(SimpleNamespace(_publish_tracking_work=publish),
        world_close=lambda: events.append('closed'))
    assert worker.enqueue(work(hooks)) and started.wait(2)
    assert worker.enqueue(work(hooks, frame=2))
    assert worker.enqueue(work(hooks, source=1))
    assert worker.enqueue(work(hooks, frame=3))
    time.sleep(.08)
    worker.shutdown(wait=False)
    assert worker.diagnostics_summary is None
    assert 'diagnostic summary' not in caplog.text
    release.set(); worker.shutdown(timeout_s=2)
    assert events == [(0,1),(0,2),(1,1),(0,3),'closed']
    summary = worker.diagnostics_summary
    assert 0 < summary['samples'] <= 750
    assert summary['error'] is None
    assert not worker.diagnostics._thread.is_alive()
    assert 'SECRET_LOCAL_PAYLOAD' not in json.dumps(summary)
    assert any(loc[1] == 'publish' for row in summary['stacks'] for loc in row['locations'])
    assert caplog.text.count('Publication worker diagnostic summary:') == 1


def test_production_bounds_and_exact_location_payload(monkeypatch):
    from noesis.pipelines import hooks
    cls = hooks._PublicationWorkerDiagnostics
    assert (cls.MAX_SAMPLES,cls.MAX_STACKS,cls.MAX_DEPTH,cls.DURATION_S,cls.PERIOD_S) == (750,256,24,15.,.02)
    diagnostic = cls(); diagnostic._started = 0.
    waits = []
    diagnostic._stop = SimpleNamespace(wait=lambda period: waits.append(period) or False, set=lambda:None)
    monkeypatch.setattr(hooks, 'time', SimpleNamespace(monotonic=lambda:0.))
    counter = iter(range(1000))
    monkeypatch.setattr(diagnostic, '_capture', lambda ident: tuple(
        ('/'+'x'*500, 'exact_function', next(counter)) for _ in range(1)))
    diagnostic._sample(123)
    assert diagnostic._samples == 750 and len(waits) == 750
    assert all(period >= .02 for period in waits)
    assert len(diagnostic._stacks) <= 256 and diagnostic._omitted > 0
    assert diagnostic._bytes <= 96*1024
    summary = diagnostic.close()
    assert len(json.dumps(summary,sort_keys=True,ensure_ascii=True).encode('ascii')) <= 96*1024
    assert all(len(row['locations']) <= 24 for row in summary['stacks'])


def test_real_capture_has_only_exact_scalar_locations():
    from noesis.pipelines import hooks
    diagnostic = hooks._PublicationWorkerDiagnostics()
    def recurse(depth):
        return recurse(depth-1) if depth else diagnostic._capture(threading.get_ident())
    stack = recurse(40)
    assert len(stack) == 24
    assert all(type(path) is str and type(name) is str and type(line) is int for path,name,line in stack)
    assert any(path == __file__ and name == 'recurse' for path,name,line in stack)


def test_capture_exception_releases_all_thread_frame_references(monkeypatch):
    from noesis.pipelines import hooks
    class Frame:
        f_code=SimpleNamespace(co_filename='exact.py',co_name='exact')
        f_lineno=1
        @property
        def f_back(self): raise ValueError('synthetic frame walk failure')
    frames={1:Frame(),2:Frame()}
    monkeypatch.setattr(hooks,'sys',SimpleNamespace(_current_frames=lambda:frames))
    with pytest.raises(ValueError,match='frame walk'):
        hooks._PublicationWorkerDiagnostics()._capture(1)
    assert frames == {}


def test_final_json_bound_includes_worst_legal_stacks_headers_and_stages():
    from noesis.pipelines import hooks
    diagnostic=hooks._PublicationWorkerDiagnostics()
    diagnostic._started=1.; diagnostic._ended=16.; diagnostic._samples=750
    # Exact legal24-deep/512filename/128function locations. Accounting fits
    # under96KiB but headers/stages push the FINAL representation over it.
    depth=24
    long='\u2603'*512
    for count,length,offset in ((375,24,0),(374,6,24)):
        stack=tuple((long,'f'*128,offset+line) for line in range(length))
        diagnostic._stacks[stack]=count
    diagnostic._stacks[(('\u2603'*255,'f'*128,100),)]=1
    diagnostic._bytes=sum(len(json.dumps(stack,ensure_ascii=True).encode('ascii'))+64
                          for stack in diagnostic._stacks)
    assert diagnostic._bytes <= diagnostic.MAX_BYTES
    diagnostic._stages={name:[750,10**12,10**9] for name in
        ('quality_ingress_stage','identity_producer_prepare','tracking_publish_world_commit')}
    summary=diagnostic.close()
    encoded=json.dumps(summary,sort_keys=True,ensure_ascii=True).encode('ascii')
    assert len(encoded) <= 96*1024
    assert summary['stacks'] == []
    assert summary['stacks_omitted_reason'] == 'final_json_byte_bound'
    assert summary['samples'] == summary['omitted_samples'] == 750
    assert summary['error'] is None and len(summary['stages_ns']) == 3


def test_close_timeout_and_error_summaries_are_bounded():
    from noesis.pipelines import hooks
    diagnostic=hooks._PublicationWorkerDiagnostics()
    joins=[]
    diagnostic._thread=SimpleNamespace(join=lambda timeout:joins.append(timeout),is_alive=lambda:True)
    summary=diagnostic.close()
    assert joins == [.5] and summary['sample_counts_available'] is False
    assert summary['error'] == 'sampler_not_quiesced'
    assert len(json.dumps(summary).encode()) <= 96*1024
    diagnostic=hooks._PublicationWorkerDiagnostics()
    diagnostic._error='X'*(96*1024)
    summary=diagnostic.close()
    assert summary['error'] == 'summary_byte_bound'
    assert len(json.dumps(summary).encode()) <= 96*1024


def test_duration_and_idle_start_wait_are_finite(monkeypatch):
    from noesis.pipelines import hooks
    cls=hooks._PublicationWorkerDiagnostics
    d=cls(); d._started=0.
    monkeypatch.setattr(hooks,'time',SimpleNamespace(monotonic=lambda:15.))
    d._sample(1)
    assert d._samples == 0
    d=cls(); d._created=0.
    d._stop=SimpleNamespace(wait=lambda period:False)
    monkeypatch.setattr(hooks,'time',SimpleNamespace(monotonic=lambda:60.))
    d._sample(1)
    assert d._samples == 0 and d._error == 'no_work_before_start_bound'


def test_sampler_start_failure_does_not_change_worker_publication(monkeypatch):
    from noesis.pipelines import hooks
    monkeypatch.setenv(hooks._PublicationWorkerDiagnostics.ENV,'1')
    original=threading.Thread.start
    def start(thread):
        if thread.name == 'NoesisPublicationDiagnostics':
            raise RuntimeError('optional thread unavailable')
        return original(thread)
    monkeypatch.setattr(threading.Thread,'start',start)
    events=[]
    worker=hooks._TrackingPublicationWorker(SimpleNamespace(
        _publish_tracking_work=lambda row:events.append(row.frame_id)))
    assert worker.enqueue(work(hooks))
    worker.shutdown(timeout_s=2)
    assert events == [1] and worker.terminal_failure is None
    assert worker.diagnostics_summary['error'] == 'RuntimeError'


def test_optional_stage_counter_failure_cannot_replace_real_refusal(monkeypatch):
    from noesis.pipelines import hooks
    diagnostic=hooks._PublicationWorkerDiagnostics()
    def fail(*args): raise ValueError('optional counter failure')
    monkeypatch.setattr(diagnostic,'record_stage',fail)
    refusal=RuntimeError('real canonical refusal')
    with pytest.raises(RuntimeError) as caught:
        with diagnostic.stage('quality_ingress_stage',0):
            raise refusal
    assert caught.value is refusal and diagnostic._error == 'ValueError'


def test_sampler_error_cannot_change_canonical_worker_result(monkeypatch):
    from noesis.pipelines import hooks
    monkeypatch.setenv(hooks._PublicationWorkerDiagnostics.ENV, '1')
    def fail(*args): raise ValueError('not exported')
    monkeypatch.setattr(hooks._PublicationWorkerDiagnostics, '_capture', fail)
    published=[]
    def publish(row): published.append(row.frame_id); time.sleep(.07)
    worker=hooks._TrackingPublicationWorker(SimpleNamespace(_publish_tracking_work=publish))
    assert worker.enqueue(work(hooks))
    worker.shutdown(timeout_s=2)
    assert published == [1] and worker.terminal_failure is None
    assert worker.diagnostics_summary['error'] == 'ValueError'
    assert not worker.diagnostics._thread.is_alive()


def test_terminal_publication_failure_still_closes_sampler_once(monkeypatch, caplog):
    from noesis.pipelines import hooks
    monkeypatch.setenv(hooks._PublicationWorkerDiagnostics.ENV, '1')
    caplog.set_level(logging.INFO, logger=hooks.__name__)
    failure=RuntimeError('canonical original failure')
    def publish(row): raise failure
    worker=hooks._TrackingPublicationWorker(SimpleNamespace(_publish_tracking_work=publish))
    assert worker.enqueue(work(hooks))
    worker.shutdown(timeout_s=2)
    assert worker.terminal_failure is failure
    assert not worker.diagnostics._thread.is_alive()
    assert caplog.text.count('Publication worker diagnostic summary:') == 1


@pytest.mark.parametrize('fail_quality',[False,True])
def test_actual_quality_and_publisher_world_stage_boundaries(monkeypatch,fail_quality):
    from noesis.pipelines import hooks
    monkeypatch.setenv(hooks._PublicationWorkerDiagnostics.ENV, '1')
    events=[]
    class Publisher:
        def publish(self, source, tracks, *, frame_metadata):
            events.append('publisher_world_transaction')
            return hooks.TrackingPublicationReceipt(source_id=source,
                frame_id=frame_metadata['frame_id'],observed_at_us=frame_metadata['observed_at_us'],
                tracking_publication_sequence=1,outbound_submission_id=1,outbound_message_count=1)
    failure=RuntimeError('actual quality refusal')
    def quality_stage(*args,**kwargs):
        events.append('quality_stage')
        if fail_quality: raise failure
    ingress=SimpleNamespace(stage=quality_stage)
    processor=hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={},ds_pipeline=object(),xc3d_world_quality_ingress=ingress),
        tracking_pub=Publisher(),camera_labels={0:'camera-0'},sensor_id_map={})
    worker=hooks._TrackingPublicationWorker(processor)
    processor._tracking_publication_worker=worker
    row=work(hooks)
    from dataclasses import replace
    assert worker.enqueue(replace(row,world_quality_work=object()))
    worker.shutdown(timeout_s=2)
    assert worker.terminal_failure is (failure if fail_quality else None)
    assert events == (['quality_stage'] if fail_quality else ['quality_stage','publisher_world_transaction'])
    stages=worker.diagnostics_summary['stages_ns']
    assert set(stages) == ({'quality_ingress_stage'} if fail_quality else
                          {'quality_ingress_stage','tracking_publish_world_commit'})
    assert all(v['count'] == 1 and 0 <= v['max'] <= v['total'] for v in stages.values())
