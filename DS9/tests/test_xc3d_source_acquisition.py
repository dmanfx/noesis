"""SYNTHETIC scalar cohorts; no physical/acquisition clock qualification."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from collections import deque
import threading
import copy
import json
import pytest
from noesis.pipelines.source_acquisition import (
    SourceAcquisitionState, AcquisitionUnavailable, ScopedFrameReceipts,
    PrivateAssociationWork, current_frame_key, ordered_current_frames,
)


def retained_swin_case():
    from dataclasses import replace
    from noesis_core.tracking_continuity import TrackingLifecycleRegistry
    from noesis.pipelines.source_acquisition import OriginalSwinCache,OriginalSwinExtraction,freeze_owner_retention
    from reid.mv3dt_person_association import AppearanceProfile,MemberKey,ObservationStamp,SourceObservation,HouseholdDescriptor
    state=SourceAcquisitionState('run','graph','source_pts')
    registry=TrackingLifecycleRegistry(reappearance_grace_s=.43)
    cache=OriginalSwinCache(run_id='run',graph_id='graph')
    profile=AppearanceProfile('model','layer',256,'fingerprint')
    def current(frame):
        original=state.observe_native(row(frame))
        life=registry.identity_observation(1,9,frame_id=frame,observed_at_us=frame*33333,media_pts_ns=original.pts_ns)
        tracks=[dict(tracker_id=9,tracker_lifecycle_generation=life.generation)]
        c=registry.update_frame(source_id=1,camera_id='kitchen',frame_id=frame,observed_at_us=frame*33333,
            media_pts_ns=original.pts_ns,tracks=tracks)
        retention=freeze_owner_retention(c,registry.identity_retention_keys(1,media_pts_ns=c.media_pts_ns,max_gap_ns=430000000),430000000,original)
        private=PrivateAssociationWork(1,frame,original.pts_ns,0,original,None,owner_retention=retention)
        result=state.consume_owner_work(private,0,tracks)
        cache.bind_owner(original,0,frame,result['members'],owner_retention=retention)
        return original,life
    birth,_=current(1)
    f=state.observe_native(row(2))
    extraction=OriginalSwinExtraction(f,9,profile,(1.,)+(0.,)*255,100.066)
    assert cache.record(extraction) and cache.freeze_exact(f,9) is extraction
    n,life=current(3)
    member=MemberKey('run','kitchen',1,0,9,life.generation)
    stamp=ObservationStamp(member,2,'actual-original-F',f.pts_ns/1e9,'source-media',f.pts_ns/1e9,'source-media',1)
    observation=SourceObservation(stamp,HouseholdDescriptor('unknown'))
    owner=cache.owners[1][(0,9,life.generation)]
    return cache,extraction,observation,n,owner,profile


@pytest.mark.parametrize('fault',[None,'beforebirth','reuse','reset','profile','run','clock','buffer','expired_source',
    'expired_monotonic','future','duplicate','capacity'])
def test_original_F_frozen_cache_binding_and_refusals(fault):
    from dataclasses import replace
    from noesis.pipelines.source_acquisition import OriginalSwinExtraction
    cache,e,o,n,owner,profile=retained_swin_case()
    now=100.1
    if fault=='beforebirth':o=replace(o,stamp=replace(o.stamp,lifecycle_started_frame=3))
    elif fault=='reuse':o=replace(o,stamp=replace(o.stamp,member=replace(o.stamp.member,lifecycle_generation=2)))
    elif fault=='reset':cache.owners[1].clear()
    elif fault=='profile':profile=replace(profile,fingerprint='wrong')
    elif fault=='run':n=replace(n,run_id='wrong')
    elif fault=='clock':n=replace(n,source_clock_epoch=1)
    elif fault=='buffer':e=replace(e,original=replace(e.original,buffer_scope=999))
    elif fault=='expired_source':n=replace(n,pts_ns=n.pts_ns+400000000)
    elif fault=='expired_monotonic':now=101.
    elif fault=='future':o=replace(o,stamp=replace(o.stamp,frame_id=9))
    elif fault=='capacity':
        # Genuine recorded copies remain bounded, without unbounded retained vectors.
        for i in range(300):
            original=replace(e.original,buffer_scope=1000+i)
            value=OriginalSwinExtraction(original,9,profile,e.vector,100.066)
            cache.record(value);cache.freeze_exact(original,9)
        assert len(cache.frozen_extractions[1])==256 and cache.counters['frozen_capacity']>0
        cache.finish_frozen(n)
        assert not cache.frozen_extractions[1]
        return
    if fault not in (None,'duplicate'):
        with pytest.raises(AcquisitionUnavailable):cache.bind_retained_original(e,o,n,owner,profile,now)
        return
    assert cache.bind_retained_original(e,o,n,owner,profile,now)
    assert cache.members[1][owner.member][0] is e and cache.members[1][owner.member][3].stamp is o.stamp
    if fault=='duplicate':assert cache.bind_retained_original(e,o,n,owner,profile,now) is False
    summary=cache.producer_summary()
    assert len(summary['examples'])<=64 and 'vector' not in repr(summary)


def r10_clock_callback(native, output):
    """Compile ONLY the actual nested callback; never execute its SDK/source setup."""
    path = Path(__file__).resolve().parents[1]/"scripts/probe_source_acquisition.py"
    tree = ast.parse(path.read_text())
    execute = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="execute_probe")
    callback = next(node for node in execute.body if isinstance(node,ast.ClassDef) and node.name=="Frames")
    factory = ast.parse("def factory(native,frames):\n    drops=0\n    return None\n").body[0]
    factory.body[1:]=[callback,*ast.parse("return Frames(),lambda:drops").body]
    module = ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[]))
    namespace={"BatchMetadataOperator":object,"ScopedFrameReceipts":ScopedFrameReceipts,
        "ordered_current_frames":ordered_current_frames,"queue":__import__("queue")}
    exec(compile(module,str(path),"exec"),namespace)
    return namespace["factory"](native,output)


@pytest.mark.parametrize("spec", [((1,10),(1,11),(2,7)),((1,10),(1,11)),((2,7),)])
def test_r10_clock_actual_callback_enqueues_owned_reordered_repeated_partial_keys_only(spec):
    import queue
    rows=[row(number,source,slot) for slot,(source,number) in enumerate(spec)]
    batch=SimpleNamespace(frame_items=[frame(receipt) for receipt in reversed(rows)])
    calls=[]
    def typed(frame_meta):
        assert type(frame_meta) is not tuple  # native FrameMetadata API never sees the returned key
        calls.append(1)
        return current_frame_key(frame_meta)
    native=SimpleNamespace(stamp_source_acquisition=lambda _batch:None,extract_frame_identity=typed,
        extract_source_acquisition=lambda _batch:{"scope":1,"original_buffer_scope":1,"rows":rows})
    output=queue.Queue(maxsize=256)
    callback,drops=r10_clock_callback(native,output)
    callback.handle_metadata(batch)
    copied=[output.get_nowait() for _ in spec]
    assert drops()==0 and len(calls)==len(spec)
    assert [item["original_frame"] for item in copied if item["source_id"]==1]==sorted(number for source,number in spec if source==1)
    assert all(type(item) is dict and all(not hasattr(value,"frame_items") for value in item.values()) for item in copied)


@pytest.mark.parametrize("fault",["key_shape","wrong_pts","duplicate_slot","foreign_scope"])
def test_r10_clock_actual_callback_rejects_malformed_or_mismatched_owned_identity(fault):
    import queue
    receipt=row(10)
    batch=SimpleNamespace(frame_items=[frame(receipt)])
    payload={"scope":1,"original_buffer_scope":1,"rows":[dict(receipt)]}
    if fault=="wrong_pts":payload["rows"][0]["post_tracker_pts_ns"]+=1
    elif fault=="duplicate_slot":payload["rows"].append(dict(receipt))
    elif fault=="foreign_scope":payload["rows"][0]["original_buffer_scope"]=2
    native=SimpleNamespace(stamp_source_acquisition=lambda _batch:None,
        extract_frame_identity=(lambda _frame:(0,1)) if fault=="key_shape" else current_frame_key,
        extract_source_acquisition=lambda _batch:payload)
    output=queue.Queue(maxsize=256)
    callback,drops=r10_clock_callback(native,output)
    callback.handle_metadata(batch)
    assert drops()==1 and output.empty()


def row(frame=1, source=1, slot=0, scope=1):
    return {"source_id": source, "original_source_id": source, "original_frame": frame,
            "slot": slot, "original_batch_id": slot, "scope": scope, "original_buffer_scope": scope,
            "sdk_source_id": source + 100, "original_sdk_source_id": source + 100,
            "original_pts_ns": frame * 33333333, "original_ntp_ns": 1000000000 + frame * 33333333,
            "post_tracker_ntp_ns": 1000000000 + frame * 33333333,
            "post_tracker_frame": frame + 20000, "post_tracker_pts_ns": frame * 33333333,
            "receipt_valid": True}


def test_original_owner_birth_survives_ten_thousand_renewals_and_frame_ring():
    state = SourceAcquisitionState("synthetic-run", "synthetic-graph", "source_sr")
    first = state.observe_native(row())
    for frame in range(1, 10001):
        cohort = first if frame == 1 else state.observe_native(row(frame))
        result = state.owner_frame(1, 3, cohort, [{"id": 9, "tracker_lifecycle_generation": 4}])
        assert result["members"][0][1] == first
        assert result["status"] == "accepted_identity_dto_unavailable"
        assert result["sdk_birth"] is None
    assert len(state.frames[1]) == 256
    assert result["members"][0][1].original_frame == 1
    assert state.summary()["comparable"] is False


@pytest.mark.parametrize("fault", ["receipt", "pts_join", "source_join", "duplicate", "regression", "negative", "ntp_missing"])
def test_invalid_exact_scalar_join_withdraws_members_and_never_last_seen(fault):
    state = SourceAcquisitionState("r", "g")
    first = state.observe_native(row())
    state.owner_frame(1, 1, first, [{"id": 9, "tracker_lifecycle_generation": 1}])
    next_row = row(2)
    if fault == "receipt": next_row["receipt_valid"] = False
    elif fault == "pts_join": next_row["post_tracker_pts_ns"] = 1
    elif fault == "source_join": next_row["original_source_id"] = 2
    elif fault in {"duplicate", "regression"}: next_row = row(1)
    elif fault == "negative": next_row["original_pts_ns"] = -1
    else:
        next_row["original_ntp_ns"] = 0
        next_row["post_tracker_ntp_ns"] = 0
        state.observe_native(next_row)
        assert not state.mapping(1)["comparable"]
        return
    with pytest.raises(AcquisitionUnavailable): state.observe_native(next_row)
    assert not state.members[1]
    assert not state.frames[1]


@pytest.mark.parametrize("boundary", ["empty", "termination", "raw_id_change", "generation", "outage"])
def test_actual_owner_boundaries_remove_old_member_without_native_alias(boundary):
    state = SourceAcquisitionState("r", "g")
    cohort = state.observe_native(row())
    state.owner_frame(1, 1, cohort, [{"id": 9, "tracker_lifecycle_generation": 1}])
    if boundary == "outage":
        state.reset(1, "source_outage")
    else:
        tracks = [] if boundary == "empty" else [{"id": 10 if boundary == "raw_id_change" else 9,
                       "tracker_lifecycle_generation": 2 if boundary == "generation" else 1}]
        state.owner_frame(1, 1, cohort, tracks, [9] if boundary == "termination" else [])
    assert (1, 9, 1) not in state.members[1]


def test_synthetic_sr_mapping_is_bounded_and_host_time_is_not_common_clock():
    state = SourceAcquisitionState("r", "g", "source_sr")
    for index in range(40):
        state.observe_sr(1, 1, 15, index * 90000, 10**12 + index * 10**9, 10**12 + index * 10**9 + 5000)
    mapping = state.mapping(1)
    assert mapping["valid"] and mapping["rtp_rate_hz"] == 90000
    assert mapping["residual_max_ns"] == 0
    assert mapping["comparable"] is False and mapping["host_arrival_is_latency_only"]
    assert len(state.srs[1]) == 32
    state.observe_sr(1, 2, 20, 1, 12**12, 13**12)
    assert state.epochs[1] == 1 and not state.mapping(1)["valid"]
    with pytest.raises(AcquisitionUnavailable, match="stale_sr_session"):
        state.observe_sr(1, 1, 15, 1, 14**12, 15**12)


def test_host_or_pts_and_sr_absence_are_typed_unavailable():
    for origin in ("host_receive", "source_pts", "source_sr"):
        state = SourceAcquisitionState("r", "g", origin)
        state.observe_native(row())
        assert state.summary()["comparable"] is False
        assert state.mapping(1)["reason"] == "sr_mapping_unavailable"


def test_owner_birth_survives_withdrawal_but_does_not_authorize_a_member():
    state = SourceAcquisitionState("r", "g")
    first = state.observe_native(row())
    state.owner_frame(1, 1, first, [{"id": 9, "tracker_lifecycle_generation": 1}])
    empty = state.observe_native(row(2))
    denied = state.owner_frame(1, 1, empty, [])
    assert denied["members"] == ()
    returned = state.observe_native(row(3))
    renewal = state.owner_frame(1, 1, returned, [{"id": 9, "tracker_lifecycle_generation": 1}])
    assert renewal["members"][0][1] == first
    state.owner_frame(1, 1, returned, [], terminated=[9])
    assert not state.births[1]


def test_actual_native_scalar_to_analytics_hook_boundary_without_sdk(monkeypatch):
    hooks, operator, processor, native = operator_fixture(monkeypatch, [row()])
    batch = SimpleNamespace(frame_items=[frame(row())])
    operator.handle_metadata(batch)
    evidence = processor.handle_servicemaker_frame.call_args.kwargs["xc3d_private_evidence"]
    assert evidence.original.original_frame == 1
    assert native.extract_source_acquisition.call_count == 1
    native.extract_source_acquisition.side_effect = RuntimeError("synthetic private extraction failure")
    operator.handle_metadata(batch)
    assert processor.handle_servicemaker_frame.call_count == 2
    assert processor.handle_servicemaker_frame.call_args.kwargs["xc3d_private_evidence"].original is None
    assert not processor.pipeline.xc3d_source_acquisition.frames[1]
    processor.tracking_pub._report_failure.assert_not_called()
    assert all(call.args[0] != "tracking.native_lifecycle_extract_failed_total"
               for call in hooks._increment_core_counter.call_args_list)


def frame(receipt):
    return SimpleNamespace(batch_id=receipt["slot"], pad_index=receipt["source_id"],
        source_id=receipt["sdk_source_id"], frame_number=receipt["post_tracker_frame"],
        buf_pts=receipt["post_tracker_pts_ns"], ntp_timestamp=receipt["post_tracker_ntp_ns"])


def lifecycle(receipt, *, shadow=(), terminated=(), barrier=False):
    result = {key: receipt[key] for key in ("scope", "slot", "source_id", "sdk_source_id",
        "post_tracker_frame", "post_tracker_pts_ns", "post_tracker_ntp_ns")}
    result.update(frame_id=receipt["post_tracker_frame"], media_pts_ns=receipt["post_tracker_pts_ns"],
        shadow_available=True, terminated_available=barrier,
        shadow=[{"class_id": 0, "tracker_id": raw, "tracker_state": 2,
                 "tracker_frame_id": receipt["post_tracker_frame"], "age": 5} for raw in shadow],
        terminated=[{"class_id": 0, "tracker_id": raw} for raw in terminated],
        termination_boundary="complete_buffer_source_barrier" if barrier else "not_at_buffer_barrier")
    return result


def operator_fixture(monkeypatch, rows):
    import sys
    monkeypatch.setitem(sys.modules, "pyservicemaker", None)
    from noesis.pipelines import hooks
    # Suppress only the SDK base constructor. Exercise the actual owned
    # operator methods/validator/queue, with SDK imports deliberately blocked.
    monkeypatch.setattr(hooks._BatchMetadataOperatorBase, "__init__", lambda self: None)
    native = SimpleNamespace(extract_frame_identity=current_frame_key, extract_tracker_lifecycle=Mock(return_value={
        "scope": 1, "rows": [lifecycle(receipt, barrier=True) for receipt in rows]}),
        extract_source_acquisition=Mock(return_value={"scope": 1, "original_buffer_scope": 1,
                                                      "rows": copy.deepcopy(rows)}))
    monkeypatch.setattr(hooks, "noesis_analytics_meta_ext", native)
    monkeypatch.setattr(hooks, "_increment_core_counter", Mock())
    path = Path(__file__).resolve()
    pipeline = SimpleNamespace(config={"xc3d_candidate": {"contract": "noesis.xc3d.m960.candidate.v1"}},
        world_service=SimpleNamespace(producer=SimpleNamespace(run_id="synthetic")), yaml_path=path)
    processor = SimpleNamespace(pipeline=pipeline, tracking_pub=Mock(), handle_servicemaker_frame=Mock())
    return hooks, hooks._AnalyticsTelemetryOperator(processor), processor, native


@pytest.mark.parametrize("spec", [((1,10),(1,11),(2,7)), ((1,10),(1,11),(1,12)),
    ((1,10),), ((1,10),(2,7)), ((1,10),(1,11))])
@pytest.mark.parametrize("private_exception", [False, True])
def test_repeated_unequal_and_partial_batches_publish_every_exact_cohort(monkeypatch, spec, private_exception):
    rows = [row(number, source, slot) for slot, (source, number) in enumerate(spec)]
    hooks, operator, processor, native = operator_fixture(monkeypatch, rows)
    if private_exception:
        native.extract_source_acquisition.side_effect = RuntimeError("synthetic private extraction failure")
    # Enumeration is deliberately reversed; exact frame/PTS, not list position
    # or source cardinality, controls lookup and within-source publication order.
    batch = SimpleNamespace(frame_items=[frame(receipt) for receipt in reversed(rows)])
    operator.handle_metadata(batch)
    calls = processor.handle_servicemaker_frame.call_args_list
    assert len(calls) == len(spec)
    for source in set(source for source, _number in spec):
        current = [call for call in calls if call.args[0].pad_index == source]
        assert [call.args[0].frame_number - 20000 for call in current] == sorted(number for src, number in spec if src == source)
        for call in current:
            key = current_frame_key(call.args[0])
            original = call.kwargs["xc3d_private_evidence"].original
            if private_exception:
                assert original is None and call.kwargs["xc3d_private_evidence"].denial
                continue
            assert (original.slot, original.source_id, original.sdk_source_id, original.pts_ns, original.ntp_ns) == (
                key[0], key[1], key[2], key[4], key[5])
    processor.tracking_pub._report_failure.assert_not_called()


@pytest.mark.parametrize("fault", ["missing", "null", "duplicate", "duplicate_owned", "slot", "source", "sdk_source", "pts", "ntp", "scope", "original_scope", "foreign_original_scope", "malformed_bounds", "null_batch", "exception"])
def test_private_row_denial_keeps_all_canonical_cohorts_and_safe_siblings(monkeypatch, fault):
    rows = [row(10, 1, 0), row(7, 2, 1)]
    hooks, operator, processor, native = operator_fixture(monkeypatch, rows)
    payload = copy.deepcopy(native.extract_source_acquisition.return_value)
    bad = payload["rows"][0]
    if fault == "missing": payload["rows"].pop(0)
    elif fault == "null": bad.update(receipt_valid=False, private_error="null SDK receipt")
    elif fault == "duplicate": payload["rows"].append(copy.deepcopy(bad))
    elif fault == "duplicate_owned": bad.update(receipt_valid=False, private_error="duplicate owned receipt")
    elif fault == "slot": bad["slot"] = 2
    elif fault == "source": bad["source_id"] = 0
    elif fault == "sdk_source": bad["sdk_source_id"] = 999
    elif fault == "pts": bad["post_tracker_pts_ns"] += 1
    elif fault == "ntp": bad["post_tracker_ntp_ns"] += 1
    elif fault == "scope": bad["scope"] = 2
    elif fault == "original_scope": bad["original_buffer_scope"] = 0
    elif fault == "foreign_original_scope": bad["original_buffer_scope"] = 99
    elif fault == "malformed_bounds": payload["rows"] *= 2
    elif fault == "null_batch": payload = None
    else: native.extract_source_acquisition.side_effect = ValueError("synthetic private exception")
    native.extract_source_acquisition.return_value = payload
    operator.handle_metadata(SimpleNamespace(frame_items=[frame(receipt) for receipt in rows]))
    calls = processor.handle_servicemaker_frame.call_args_list
    assert len(calls) == 2
    by_source = {call.args[0].pad_index: call.kwargs["xc3d_private_evidence"] for call in calls}
    assert by_source[1].original is None and by_source[1].denial
    if fault not in {"malformed_bounds", "null_batch", "exception"}:
        assert by_source[2].original.original_frame == 7
    processor.tracking_pub._report_failure.assert_not_called()
    assert all(call.args[0] != "tracking.native_lifecycle_extract_failed_total"
               for call in hooks._increment_core_counter.call_args_list)


def test_equal_slots_different_buffers_never_cross_callback_scope():
    first, second = SimpleNamespace(), SimpleNamespace()
    receipts = ScopedFrameReceipts(first, {"scope": 1, "original_buffer_scope": 1, "rows": [row()]})
    assert receipts.lookup(first, frame(row()))["original_frame"] == 1
    with pytest.raises(AcquisitionUnavailable, match="current_buffer_scope"):
        receipts.lookup(second, frame(row()))
    next_receipts = ScopedFrameReceipts(second, {"scope": 2, "original_buffer_scope": 2, "rows": [row(2, scope=2)]})
    assert next_receipts.lookup(second, frame(row(2, scope=2)))["original_buffer_scope"] == 2


@pytest.mark.parametrize("field,value", [("batch_id",3),("batch_id",-1),("pad_index",3),
    ("frame_number",-1),("buf_pts",(1<<64)-1),("source_id",None),("ntp_timestamp",None)])
def test_malformed_current_frame_bounds_are_denied(field, value):
    current = frame(row())
    setattr(current, field, value)
    with pytest.raises(AcquisitionUnavailable): current_frame_key(current)


def test_source_clock_reset_is_scoped_and_detached_older_receipt_is_immutable():
    state = SourceAcquisitionState("r", "g")
    kitchen, family = state.observe_native(row(10)), state.observe_native(row(7, source=2))
    state.reset(1, "synthetic_source_reconnect")
    assert kitchen.source_clock_epoch == 0 and kitchen.original_frame == 10
    assert state.epochs == [0,1,0] and list(state.frames[2]) == [family]
    assert not state.frames[1]


def test_intrabuffer_epoch_or_clock_conflict_is_not_sorted_away():
    rows = [row(10, slot=0), row(11, slot=1)]
    rows[1]["post_tracker_frame"] = 1  # Reset counter with strictly later PTS.
    with pytest.raises(AcquisitionUnavailable, match="intrabuffer_epoch"):
        ordered_current_frames([frame(receipt) for receipt in rows])
    with pytest.raises(AcquisitionUnavailable, match="duplicate_current_slot"):
        ordered_current_frames([frame(row()), frame(row(2))])


def test_no_ntp_is_legitimate_but_never_a_comparable_clock(monkeypatch):
    receipt = row()
    receipt.update(original_ntp_ns=0, post_tracker_ntp_ns=0)
    _hooks, operator, processor, _native = operator_fixture(monkeypatch, [receipt])
    operator.handle_metadata(SimpleNamespace(frame_items=[frame(receipt)]))
    original = processor.handle_servicemaker_frame.call_args.kwargs["xc3d_private_evidence"].original
    assert original.ntp_ns == 0
    assert processor.pipeline.xc3d_source_acquisition.summary()["comparable"] is False


def test_unexpected_private_validation_exception_does_not_skip_any_canonical_frame(monkeypatch):
    rows = [row(10, slot=0), row(11, slot=1), row(7, source=2, slot=2)]
    _hooks, operator, processor, _native = operator_fixture(monkeypatch, rows)
    processor.pipeline.xc3d_source_acquisition.observe_native = Mock(side_effect=TypeError("synthetic private failure"))
    operator.handle_metadata(SimpleNamespace(frame_items=[frame(receipt) for receipt in rows]))
    assert processor.handle_servicemaker_frame.call_count == 3
    for call in processor.handle_servicemaker_frame.call_args_list:
        assert call.kwargs["xc3d_private_evidence"].denial == "private_frame_validation_failed"
    processor.tracking_pub._report_failure.assert_not_called()


def test_default_baseline_adapter_does_not_extract_private_acquisition(monkeypatch):
    rows = [row(10, slot=0), row(11, slot=1)]
    hooks, _operator, processor, native = operator_fixture(monkeypatch, rows)
    pipeline = SimpleNamespace(config={})
    processor.pipeline = pipeline
    operator = hooks._AnalyticsTelemetryOperator(processor)
    operator.handle_metadata(SimpleNamespace(frame_items=[frame(receipt) for receipt in rows]))
    assert processor.handle_servicemaker_frame.call_count == 2
    assert all("xc3d_private_evidence" not in call.kwargs for call in processor.handle_servicemaker_frame.call_args_list)
    native.extract_source_acquisition.assert_not_called()


def test_repeated_shadow_rows_and_termination_only_at_proven_buffer_barrier(monkeypatch):
    rows = [row(10, slot=0), row(11, slot=1), row(12, slot=2)]
    hooks, operator, processor, native = operator_fixture(monkeypatch, rows)
    lifecycles = [lifecycle(receipt, shadow=[9], terminated=[9] if index == 2 else (), barrier=index == 2)
                  for index, receipt in enumerate(rows)]
    native.extract_tracker_lifecycle.return_value["rows"] = lifecycles
    operator.handle_metadata(SimpleNamespace(frame_items=[frame(receipt) for receipt in reversed(rows)]))
    calls = processor.handle_servicemaker_frame.call_args_list
    assert len(calls) == 3
    for index, call in enumerate(calls):
        evidence = call.kwargs["tracker_lifecycle"]
        alive, removed = hooks._tracker_lifecycle_ids_for_frame(evidence, source_id=1,
            frame_id=rows[index]["post_tracker_frame"], media_pts_ns=rows[index]["post_tracker_pts_ns"])
        assert (alive, removed) == (((9,), ()) if index < 2 else ((), (9,)))
        assert call.kwargs["xc3d_private_evidence"].terminated == (() if index < 2 else (9,))


@pytest.mark.parametrize("fault", ["bad_shadow_frame", "early_termination", "unavailable", "duplicate_id", "scope", "extract"])
def test_invalid_canonical_lifecycle_uses_real_failure_gate(monkeypatch, fault):
    rows = [row(10, slot=0), row(11, slot=1)]
    _hooks, operator, processor, native = operator_fixture(monkeypatch, rows)
    bad = native.extract_tracker_lifecycle.return_value["rows"][0]
    if fault == "bad_shadow_frame":
        bad.update(shadow=[{"class_id": 0, "tracker_id": 9, "tracker_state": 2, "tracker_frame_id": 0}])
    elif fault == "early_termination":
        bad.update(terminated=[{"class_id": 0, "tracker_id": 9}], termination_boundary="not_at_buffer_barrier")
    elif fault == "unavailable": bad["lifecycle_unavailable"] = "unassignable repeated shadow sample"
    elif fault == "duplicate_id": bad["terminated"] = [{"class_id": 0, "tracker_id": 9}] * 2
    elif fault == "scope": bad["scope"] = 100
    else: native.extract_tracker_lifecycle.side_effect = ValueError("synthetic canonical corruption")
    operator.handle_metadata(SimpleNamespace(frame_items=[frame(receipt) for receipt in rows]))
    processor.handle_servicemaker_frame.assert_not_called()
    processor.tracking_pub._report_failure.assert_called_once()
    native.extract_source_acquisition.assert_not_called()


@pytest.mark.parametrize("operator_name", ["_DepthTrackingFrameOperator", "_ObjectDepthFusionOperator"])
def test_both_actual_depth_operators_bind_each_repeated_frame_and_deny_only_bad_row(monkeypatch, operator_name):
    rows = [row(10, slot=0), row(11, slot=1), row(7, source=2, slot=2)]
    hooks, _operator, _processor, native = operator_fixture(monkeypatch, rows)
    native.extract_source_acquisition.return_value["rows"][1]["receipt_valid"] = False
    depth_processor = SimpleNamespace(xc3d_exact_acquisition=True, handle_servicemaker_frame=Mock())
    operator = getattr(hooks, operator_name)(depth_processor)
    batch = SimpleNamespace(frame_items=[frame(receipt) for receipt in reversed(rows)])
    operator.handle_metadata(batch)
    calls = depth_processor.handle_servicemaker_frame.call_args_list
    assert len(calls) == 3
    for call in calls:
        current = call.args[-1]
        receipt = call.kwargs["source_receipt"]
        assert receipt["slot"] == current.batch_id
        original = hooks._xc3d_depth_source_frame(receipt, current.source_id, current.buf_pts // 1000)
        assert original == (None if current.batch_id == 1 else rows[current.batch_id]["original_frame"])


@pytest.mark.parametrize("private_diag_exception", [False, True])
def test_r10_actual_fifo_publisher_does_not_apply_newer_media_reset_to_older_queued_work(monkeypatch, private_diag_exception):
    from noesis_core.tracking_continuity import TrackingLifecycleRegistry
    from noesis.pipelines.source_acquisition import freeze_owner_retention
    hooks, _operator, _processor, _native = operator_fixture(monkeypatch, [row(10)])
    if private_diag_exception:
        real_diagnostic = hooks.private_owner_diagnostic
        def failing_diagnostic(evidence, epoch, result):
            if evidence.canonical_frame == 20010:
                raise TypeError("synthetic private serialization failure")
            return real_diagnostic(evidence, epoch, result)
        monkeypatch.setattr(hooks, "private_owner_diagnostic", failing_diagnostic)
    media, owner = SourceAcquisitionState("r", "g"), SourceAcquisitionState("r", "g")
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    published = []
    pipeline = SimpleNamespace(xc3d_private_owner_state=owner, xc3d_private_owner_diagnostics=deque(maxlen=64))
    identity_api = SimpleNamespace(reserve=Mock(), commit=Mock())
    pipeline.identity_association_service = identity_api  # Unaccepted: must NOT be called.
    def publish(source, tracks, *, frame_metadata):
        number = frame_metadata["frame_id"]
        if number == 20010:
            entered.set()
            assert release.wait(2)
        published.append(number)
        return hooks.TrackingPublicationReceipt(source, number, frame_metadata["observed_at_us"], len(published), 1, 1)
    lifecycle_owner = TrackingLifecycleRegistry(reappearance_grace_s=.43)
    processor = SimpleNamespace(pipeline=pipeline, _tracking_lifecycle=lifecycle_owner,
        _stamp_footpoint_lifecycles=Mock(), publication_gate=hooks.RuntimePublicationGate(),
        tracking_pub=SimpleNamespace(publish=publish), _commit_enqueued_world_output_watermarks=Mock(),
        _last_tracking_publish_ts_by_sensor={}, _last_tracking_count_by_sensor={}, bev_renderer=None)
    def perform(work):
        hooks._AnalyticsTelemetryProcessor._publish_tracking_work(processor, work)
        if work.frame_id == 20013: done.set()
    processor._publish_tracking_work = perform
    def work(number, source_epoch=0, terminated=(), denial=False):
        receipt = row(number)
        original = None if denial else media.observe_native(receipt)
        if lifecycle_owner.source_epoch(1) != source_epoch:
            assert lifecycle_owner.reset_source(1) == source_epoch
        tracks = [] if terminated else [{"tracker_id":9,"bbox":[10.,10.,30.,60.]}]
        continuity = lifecycle_owner.update_frame(source_id=1,camera_id="kitchen",frame_id=number+20000,
            observed_at_us=number*1000,media_pts_ns=receipt["original_pts_ns"],tracks=tracks,
            terminated_tracker_ids=terminated)
        retention = None if original is None else freeze_owner_retention(continuity,
            lifecycle_owner.identity_retention_keys(1,media_pts_ns=continuity.media_pts_ns,
                max_gap_ns=lifecycle_owner._reappearance_grace_ns),lifecycle_owner._reappearance_grace_ns,original)
        evidence = PrivateAssociationWork(1, number+20000, receipt["original_pts_ns"], media.epochs[1],
            original, "missing owned receipt" if denial else None, terminated,retention)
        return hooks._TrackingPublicationWork(1, "kitchen", number+20000, number*1000, float(number), number*1000,
            tracks, [], {"media_pts_ns": receipt["original_pts_ns"]},
            continuity=continuity, source_epoch=source_epoch, xc3d_private_evidence=evidence)
    worker = hooks._TrackingPublicationWorker(processor)
    try:
        assert worker.enqueue(work(10)) and entered.wait(2)
        assert worker.enqueue(work(11, terminated=(9,)))
        media.reset(1, "newer_media_reset")
        assert not owner.members[1]  # Older publisher is still blocked, not mutated by media.
        assert worker.enqueue(work(12, source_epoch=1, denial=True))
        assert worker.enqueue(work(13, source_epoch=1))
        release.set()
        assert done.wait(2)
        assert worker.terminal_failure is None
        assert published == [20010,20011,20012,20013]
        diagnostic = list(pipeline.xc3d_private_owner_diagnostics)
        if private_diag_exception:
            assert diagnostic[0][3]["status"] == "acquisition_unavailable"
            assert diagnostic[0][3]["members"] == []
        else:
            assert diagnostic[0][3]["members"][0]["original_birth"]["original_frame"] == 10
        assert diagnostic[1][3]["members"] == []  # Termination wins; no renewal/resurrection.
        assert diagnostic[2][3]["status"] == "acquisition_unavailable"
        assert diagnostic[2][3]["members"] == []
        assert diagnostic[3][3]["members"][0]["original_birth"]["original_frame"] == 13
        serialized = json.loads(json.dumps(diagnostic))
        if not private_diag_exception:
            assert serialized[0][3]["current_work"]["original"]["slot"] == 0
            assert serialized[0][3]["current_work"]["clock_epoch"] == 0
        assert serialized[3][3]["current_work"]["original"]["sdk_source_id"] == 101
        assert serialized[3][3]["current_work"]["clock_epoch"] == 1
        identity_api.reserve.assert_not_called()
        identity_api.commit.assert_not_called()
    finally:
        release.set()
        worker.shutdown(timeout_s=2)


def test_native_code_uses_verified_getters_and_owned_copy_release():
    path = Path(__file__).resolve().parents[1] / "native/noesis_analytics_meta_ext.cpp"
    text = path.read_text()
    for token in ("frame.bufferPTS()", "frame.ntpTimestamp()", "frame.frameNum()", "frame.sourceId()",
                  "batch.acquire(meta)", "source_clock_copy", "source_clock_release", "owner.append(meta)",
                  '"rtpsession"', '"sync_src"', "source.getGObject()"):
        assert token in text
    assert '"NOESIS.XC3D.ORIGINAL_BUFFER_SCOPE.v1"' in text
    assert "slots[frame.batchId()]" in text
    assert "stored->buffer_scope" in text and "stored->source_id == frame.sourceId()" in text
    assert "frames[index].frame_id == int(sample.frameNum)" in text
    assert '"complete_buffer_source_barrier"' in text
    assert "records[py::int_(current.pad_index)]" not in text


def producer_case():
    """SYNTHETIC physical admission, actual service/kernel and original operations."""
    from noesis.pipelines.source_acquisition import (OrderedIdentityProducer, OrderedProducerWork,
                                                    OriginalAssociationOperation)
    from noesis.identity_association_service import MeasurementReference
    from tests.test_identity_association_service import admitted
    from tests.test_mv3dt_person_association import observation, proof, event, now
    actor = OrderedIdentityProducer("run-a", admitted())
    old = observation(raw=7, time=10.)
    adopted = observation(raw=9, frame=2, time=10.1, media=5.1, started=1)
    peer = observation("family-room", 8, frame=2, time=10.1, media=5.1)
    left_b = observation(raw=17, frame=2, time=10.1, media=5.1)
    right_b = observation("family-room", 18, frame=2, time=10.1, media=5.1)
    adoption = event(old, adopted, peer, token="synthetic-native-adoption", time=10.1)
    adopted_proof = proof(adopted, peer, "synthetic-adopted-pair")
    ordinary = proof(left_b, right_b, "synthetic-ordinary-pair")
    operations = (OriginalAssociationOperation("transition", adopted, adopted_proof, adoption),
                  OriginalAssociationOperation("link", proof=ordinary))
    references = tuple(MeasurementReference.from_evidence(measured) for pair in (adopted_proof, ordinary)
                       for measured in (pair.left, pair.right))
    work = OrderedProducerWork("run-a", 1, 0, 2, now(10.1), (old, peer, left_b, right_b), operations, references)
    return actor, work, (old, adopted, peer, left_b, right_b), (adopted_proof, ordinary), adoption


def test_actual_ordered_producer_two_people_adoption_and_link_final_revision_reservations():
    actor, work, _observations, _proofs, event = producer_case()
    prepared = actor.prepare(work)
    assert prepared.status == "ready" and len(prepared.contributions) == 2
    dtos = [dto for dto, _cohort in prepared.contributions]
    assert len({dto.physical_key for dto in dtos}) == 2
    assert len({dto.association_revision for dto in dtos}) == 1
    adopted = next(dto for dto in dtos if dto.native_transition is not None)
    assert adopted.native_transition is event
    assert adopted.proof.receipt.event_id == event.event_id
    reservations = [actor.service.prepare(dto, cohort, work.at).reservation
                    for dto, cohort in prepared.contributions]
    assert all(reservations)
    # Direct SERVICE callback staging only, NOT a guessed World transaction.
    private = []
    for reservation in reservations:
        result = actor.service.commit(reservation, work.at, lambda dto, cohort: private.append(dto.physical_key))
        assert result.disposition == "committed"
    assert len(private) == 2 and len(set(private)) == 2
    assert all(actor.service.commit(reservation, work.at, lambda *_: None).disposition == "denied"
               for reservation in reservations)


def test_ordered_producer_a_only_never_reissues_old_b_dto_or_renews_b():
    from noesis.pipelines.source_acquisition import OrderedProducerWork, OriginalAssociationOperation
    from noesis.identity_association_service import MeasurementReference
    from tests.test_mv3dt_person_association import observation, proof, now
    actor, work, observations, _proofs, _event = producer_case()
    original = actor.prepare(work)
    prior_b, prior_b_cohort = next((dto, cohort) for dto, cohort in original.contributions if dto.native_transition is None)
    left = observation(raw=9, frame=3, time=10.2, media=5.2, started=1)
    right = observation("family-room", 8, frame=3, time=10.2, media=5.2)
    pair = proof(left, right, "synthetic-a-only-update")
    fresh = actor.prepare(OrderedProducerWork("run-a", 1, 0, 3, now(10.2), (left, right),
        (OriginalAssociationOperation("link", proof=pair),),
        tuple(MeasurementReference.from_evidence(item) for item in (pair.left, pair.right))))
    assert len(fresh.contributions) == 1
    assert actor.service.prepare(prior_b, prior_b_cohort, now(10.2)).disposition == "denied"
    assert prior_b.physical_key not in fresh.retired_keys
    assert observations[3].stamp.member not in fresh.withdrawn_members
    assert prior_b.members[0].observation.acquired_at_s == 10.1
    # Retention of B's original World evidence is World-owned; no re-add/renewal here.


@pytest.mark.parametrize("negative", ["expired", "revoked", "reset"])
def test_original_adoption_producer_replay_cannot_restore_invalid_authority(negative):
    from dataclasses import replace
    from noesis.pipelines.source_acquisition import OrderedProducerWork, OrderedAssociationIntegrityError
    from tests.test_mv3dt_person_association import now, removal, SOURCES
    actor, work, observations, _proofs, _event = producer_case()
    prepared = actor.prepare(work)
    reservations = [actor.service.prepare(dto, cohort, work.at).reservation for dto, cohort in prepared.contributions]
    mutations = ()
    if negative == "revoked": mutations = (removal(observations[1], kind="termination", time=10.2),)
    if negative == "reset": mutations = (replace(SOURCES[1], source_epoch=1),)
    replay = OrderedProducerWork("run-a", 1, 0, 3, now(10.46 if negative == "expired" else 10.2),
                                 mutations, work.operations, work.measurements)
    with pytest.raises(OrderedAssociationIntegrityError): actor.prepare(replay)
    assert all(actor.service.commit(reservation, replay.at, lambda *_: pytest.fail("stale callback")).disposition == "denied"
               for reservation in reservations)


def test_actual_producer_older_pending_then_ordered_reset_withdraws_before_replacement():
    from dataclasses import replace
    from noesis.pipelines.source_acquisition import OrderedProducerWork, OriginalAssociationOperation, OrderedAssociationIntegrityError
    from noesis.identity_association_service import MeasurementReference, policy_identity
    from tests.test_mv3dt_person_association import SOURCES, observation, proof, now, removal
    actor, older, observations, _proofs, _event = producer_case()
    # Newer detached work does NOT mutate service or revoke these reservations.
    prepared = actor.prepare(older)
    reservation = actor.service.prepare(*prepared.contributions[1], older.at).reservation
    newer_binding = replace(SOURCES[1], source_epoch=1)
    admission = replace(actor.admission, policy=replace(actor.admission.policy, sources=(SOURCES[0],newer_binding,SOURCES[2])))
    admission = replace(admission, policy_identity=policy_identity(admission.policy))
    replacement = observation(raw=9, frame=1, time=10.2, media=5.2, epoch=1)
    replacement_ref = MeasurementReference.from_evidence(proof(replacement, observations[2], "synthetic-local-reference").left)
    newer = OrderedProducerWork("run-a", 1, 1, 1, now(10.2),
        (removal(observations[1],kind="termination",time=10.2),newer_binding,admission,replacement),
        (OriginalAssociationOperation("local", observation=replacement),),(replacement_ref,))
    assert actor.service.commit(reservation, older.at, lambda *_: "older exact").disposition == "committed"
    updated = actor.prepare(newer)
    assert updated.contributions[0][0].members[0].observation.member.source_epoch == 1
    assert observations[1].stamp.member in updated.withdrawn_members
    assert observations[3].stamp.member in updated.withdrawn_members  # Other old Kitchen member reset.
    assert all(member.source_id != 2 for member in updated.withdrawn_members)  # Family survives.
    with pytest.raises(OrderedAssociationIntegrityError, match="delayed_or_replayed"):
        actor.prepare(replace(older, frame_id=4))


def test_unqualified_native_producer_is_connected_before_actual_canonical_publication(monkeypatch):
    from noesis.pipelines.source_acquisition import OrderedIdentityProducer
    from noesis.identity_association_service import ProducerAdmission
    hooks, _operator, _processor, _native = operator_fixture(monkeypatch, [row()])
    actor = OrderedIdentityProducer("r", ProducerAdmission("unavailable", "synthetic-negative-source", "real-inputs-null"))
    pipeline = SimpleNamespace(xc3d_ordered_identity_producer=actor)
    evidence = PrivateAssociationWork(1,20001,33333333,0,None,"private unavailable")
    producer_work = actor.native_unavailable_work(1,0,20001,evidence)
    seen = []
    actual_prepare = actor.prepare
    def prepare(work):
        seen.append("producer")
        return actual_prepare(work)
    monkeypatch.setattr(actor, "prepare", prepare)
    def publish(source, tracks, *, frame_metadata):
        seen.append("canonical-publication")
        return hooks.TrackingPublicationReceipt(source,20001,1000,1,1,1)
    lifecycle = SimpleNamespace(prepare_publication=lambda value:value, source_epoch=lambda _source:0, mark_published=Mock())
    processor = SimpleNamespace(pipeline=pipeline, _tracking_lifecycle=lifecycle,
        _stamp_footpoint_lifecycles=Mock(),publication_gate=hooks.RuntimePublicationGate(),
        tracking_pub=SimpleNamespace(publish=publish),_commit_enqueued_world_output_watermarks=Mock(),
        _last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    continuity=hooks.TrackingContinuityUpdate(1,20001,1000,False,(),(),camera_id="kitchen",media_pts_ns=33333333)
    work=hooks._TrackingPublicationWork(1,"kitchen",20001,1000,1.,1000,[],[],{"media_pts_ns":33333333},
        continuity=continuity,identity_producer_work=producer_work)
    hooks._AnalyticsTelemetryProcessor._publish_tracking_work(processor,work)
    assert seen == ["producer","canonical-publication"]
    assert actor.service.stats().current_dtos == 0


def test_real_FIFO_admitted_producer_denies_missing_World_API_before_publication(monkeypatch):
    hooks,_operator,_processor,_native=operator_fixture(monkeypatch,[row()])
    actor,producer_work,_observations,_proofs,_event=producer_case()
    pipeline=SimpleNamespace(xc3d_ordered_identity_producer=actor)
    lifecycle=SimpleNamespace(prepare_publication=lambda value:value)
    publisher=Mock(side_effect=AssertionError("no partial public transaction before World API"))
    processor=SimpleNamespace(pipeline=pipeline,_tracking_lifecycle=lifecycle,
        _stamp_footpoint_lifecycles=Mock(),publication_gate=hooks.RuntimePublicationGate(),
        tracking_pub=SimpleNamespace(publish=publisher))
    owned_diagnostic=[]
    def perform(work):
        try:hooks._AnalyticsTelemetryProcessor._publish_tracking_work(processor,work)
        except Exception:
            owned_diagnostic.append(actor.service.stats().current_dtos)
            raise
    processor._publish_tracking_work=perform
    continuity=hooks.TrackingContinuityUpdate(1,2,1000,True,(),(),camera_id="kitchen",media_pts_ns=5100000000)
    work=hooks._TrackingPublicationWork(1,"kitchen",2,1000,10.1,1000,[],[],{"media_pts_ns":5100000000},
        continuity=continuity,identity_producer_work=producer_work)
    failed=threading.Event()
    worker=hooks._TrackingPublicationWorker(processor,failure_callback=lambda _error:failed.set())
    try:
        assert worker.enqueue(work) and failed.wait(2)
        assert "accepted_world_transaction_api_unavailable" in str(worker.terminal_failure)
        publisher.assert_not_called()
        assert owned_diagnostic==[2]
    finally:worker.shutdown(timeout_s=2)


def test_real_FIFO_actual_service_older_work_finishes_before_detached_newer_reset():
    from dataclasses import replace
    from noesis.pipelines import hooks
    from noesis.pipelines.source_acquisition import OrderedProducerWork
    from tests.test_mv3dt_person_association import SOURCES,now
    from noesis.identity_association_service import policy_identity
    actor,older,observations,_proofs,_event=producer_case()
    entered,release,done=threading.Event(),threading.Event(),threading.Event()
    effects=[]
    owned_reservations=[]
    def perform(work):
        prepared=actor.prepare(work.identity_producer_work)
        if work.source_epoch==0:
            # Actual service reservation pending while newer media evidence is
            # constructed and queued. Synthetic consumer staging, NOT World API.
            reservations=[actor.service.prepare(dto,cohort,older.at).reservation for dto,cohort in prepared.contributions]
            owned_reservations.append(tuple(reservations))
            entered.set();assert release.wait(2)
            staged=[]
            for reservation in reservations:
                assert actor.service.commit(reservation,older.at,lambda dto,_cohort:staged.append(dto.physical_key)).disposition=="committed"
            effects.append(("older-private-transaction",tuple(staged)))
        else:
            effects.append(("ordered-reset",prepared.withdrawn_members));done.set()
    processor=SimpleNamespace(_publish_tracking_work=perform)
    worker=hooks._TrackingPublicationWorker(processor)
    def packet(frame_id,epoch,identity_work):
        return hooks._TrackingPublicationWork(1,"kitchen",frame_id,1000,10.1,1000,[],[],{},
            source_epoch=epoch,identity_producer_work=identity_work)
    try:
        assert worker.enqueue(packet(2,0,older)) and entered.wait(2)
        binding=replace(SOURCES[1],source_epoch=1)
        admission=replace(actor.admission,policy=replace(actor.admission.policy,sources=(SOURCES[0],binding,SOURCES[2])))
        admission=replace(admission,policy_identity=policy_identity(admission.policy))
        newer=OrderedProducerWork("run-a",1,1,1,now(10.2),(binding,admission),denial="fresh-proof-unavailable")
        assert len(owned_reservations[0])==2  # Detached media never calls service diagnostics/mutation.
        assert worker.enqueue(packet(1,1,newer))
        release.set();assert done.wait(2)
        assert worker.terminal_failure is None
        assert effects[0][0]=="older-private-transaction" and len(set(effects[0][1]))==2
        assert effects[1][0]=="ordered-reset"
        assert observations[1].stamp.member in effects[1][1]
        assert all(member.source_id!=2 for member in effects[1][1])
    finally:release.set();worker.shutdown(timeout_s=2)


@pytest.mark.parametrize("barrier", ["termination", "source-reset"])
def test_real_FIFO_no_fresh_proof_withdrawal_denies_missing_World_API(monkeypatch, barrier):
    from dataclasses import replace
    from noesis.pipelines.source_acquisition import OrderedProducerWork
    from tests.test_mv3dt_person_association import SOURCES, now, removal
    hooks, _operator, _processor, _native = operator_fixture(monkeypatch, [row()])
    actor, older, observations, _proofs, _event = producer_case()
    mutation = (removal(observations[1], kind="termination", time=10.2) if barrier == "termination"
                else replace(SOURCES[1], source_epoch=1))
    producer_work = OrderedProducerWork("run-a", 1, 1 if barrier == "source-reset" else 0,
        1 if barrier == "source-reset" else 3, now(10.2), (mutation,), denial="fresh-proof-unavailable")
    publisher = Mock(side_effect=AssertionError("withdrawal must reach accepted World transaction"))
    processor = SimpleNamespace(pipeline=SimpleNamespace(xc3d_ordered_identity_producer=actor),
        _tracking_lifecycle=SimpleNamespace(prepare_publication=lambda value: value),
        _stamp_footpoint_lifecycles=Mock(), publication_gate=hooks.RuntimePublicationGate(),
        tracking_pub=SimpleNamespace(publish=publisher))
    prepared = []
    actual_prepare = actor.prepare
    def prepare(work):
        result = actual_prepare(work)
        prepared.append(result)
        return result
    monkeypatch.setattr(actor, "prepare", prepare)
    def perform(work):
        # All real service mutations, including this synthetic precondition,
        # stay on the actual FIFO worker; no media-side mutation is admitted.
        actor.prepare(older)
        hooks._AnalyticsTelemetryProcessor._publish_tracking_work(processor, work)
    processor._publish_tracking_work = perform
    frame = producer_work.frame_id
    continuity = hooks.TrackingContinuityUpdate(1, frame, 1000, True, (), (),
        camera_id="kitchen", media_pts_ns=5200000000)
    work = hooks._TrackingPublicationWork(1, "kitchen", frame, 1000, 10.2, 1000, [], [],
        {"media_pts_ns": 5200000000}, continuity=continuity,
        source_epoch=producer_work.source_epoch, identity_producer_work=producer_work)
    failed = threading.Event()
    worker = hooks._TrackingPublicationWorker(processor, failure_callback=lambda _error: failed.set())
    try:
        assert worker.enqueue(work) and failed.wait(2)
        assert "accepted_world_transaction_api_unavailable" in str(worker.terminal_failure)
        assert prepared[-1].status == "unavailable" and not prepared[-1].contributions
        assert observations[1].stamp.member in prepared[-1].withdrawn_members
        assert all(member.source_id != 2 for member in prepared[-1].withdrawn_members)
        publisher.assert_not_called()
    finally:
        worker.shutdown(timeout_s=2)


def test_producer_work_rejects_borrowed_SDK_object_before_queue():
    from noesis.pipelines.source_acquisition import OrderedProducerWork
    with pytest.raises(TypeError,match="unsupported/SDK/embedding"):
        OrderedProducerWork("r",1,0,1,None,mutations=(SimpleNamespace(surface="borrowed"),))


def clock_sdk_r4_fixture(monkeypatch, *, fault=None):
    """Actual script, FAKE public SDK/adapter; no SDK imports/start/source effects.

    Native casting/registration is exercised by root's direct changed-ELF check,
    not claimed by this deliberately SDK-neutral call/order/lifetime fixture.
    """
    import importlib.util
    import sys
    from noesis.pipelines.source_acquisition import CAMERAS
    path = Path(__file__).resolve().parents[1]/"scripts/probe_source_acquisition.py"
    spec = importlib.util.spec_from_file_location("clock_sdk_r4_owned_test",path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    events, nodes, observers_alive = [], {}, set()
    class CustomObject:
        def __init__(self,name,target): self.name,self.target=name,target
    class PythonProbe:
        def __init__(self,name,handler): self.name,self.handler=name,handler
    class Node:
        def __init__(self,name): self.name,self.objects=name,{}
        def find(self,name):
            events.append(("find",self.name,name))
            if fault=="missing_anchor" and self.name=="mux": return None
            if fault=="node_instead_of_anchor" and self.name=="mux": return self
            if fault=="python_probe_instead_of_anchor" and self.name=="mux":
                return PythonProbe(name,None)
            return self.objects[name]
    class FakePipeline:
        def __init__(self,name): events.append(("pipeline",name));self.stopped=False
        def add(self,factory,name,properties):
            events.append(("add",factory,name));nodes[name]=Node(name);return self
        def attach(self,target,what,name="",tips=""):
            if isinstance(what,str):
                assert what=="noesis_xc3d_sr_anchor_v1" and tips=="src"
                assert target=="mux" and name=="xc3d_sr_pipeline_anchor"
                assert all(f"source{i}" in nodes for i in range(3)) and "mux" in nodes
                events.append(("attach",target,what,name,tips))
                nodes[target].objects[name]=CustomObject(name,nodes[target])
            else:
                assert target=="mux" and type(what) is PythonProbe
                events.append(("scalar_probe",target,what.name,tips))
            return self
        def __getitem__(self,name): return nodes[name]
        def link(self,*args): events.append(("link",args));return self
        def start(self):
            assert observers_alive=={0,1,2}
            events.append(("start",))
            raise RuntimeError("synthetic_stop_before_any_source_or_diagnostic_loop")
        def stop(self):
            self.stopped=True
            events.append(("stop",))
            assert not observers_alive or nodes["mux"].objects.get("xc3d_sr_pipeline_anchor") is not None
            if fault=="stop_failed": raise RuntimeError("synthetic_stop_failed")
        def wait(self):
            assert self.stopped and observers_alive=={0,1,2}
            events.append(("wait",))
            if fault=="wait_failed": raise RuntimeError("synthetic_wait_failed")
    class Observer:
        def __init__(self,anchor,source):
            events.append(("observer_attempt",source,type(anchor).__name__,id(anchor)))
            if type(anchor) is not CustomObject:
                raise TypeError("synthetic_actual_CustomObject_required")
            assert anchor.name=="xc3d_sr_pipeline_anchor"
            assert anchor.target is nodes["mux"] and all(f"source{i}" in nodes for i in range(3))
            if fault=="observer_failed" and source==1: raise RuntimeError("synthetic_observer_failed")
            self.source=source;observers_alive.add(source)
            events.append(("observer",source))
        def __del__(self):
            if not hasattr(self,"source"): return
            assert any(event[0]=="stop" for event in events)
            events.append(("destroy_observer",self.source))
            observers_alive.discard(self.source)
    def register():
        events.append(("register",))
        if fault=="foreign_collision": raise RuntimeError("synthetic_foreign_factory_key")
        if fault=="registration_exception": raise RuntimeError("synthetic_registration_failed")
        return fault!="registration_false"
    native=SimpleNamespace(register_sender_report_anchor_factory=register,SenderReportObserver=Observer)
    fake_assets=SimpleNamespace(candidate_analytics_pin=lambda *_args:None,
        preflight_candidate_analytics=lambda *_args,**_kwargs:native)
    fake_secrets=SimpleNamespace(load_camera_uri_registry=lambda *_args:{camera:"synthetic-no-source" for camera in CAMERAS})
    monkeypatch.setitem(sys.modules,"noesis.v3dt_assets",fake_assets)
    monkeypatch.setitem(sys.modules,"noesis_core.runtime_secrets",fake_secrets)
    monkeypatch.setitem(sys.modules,"pyservicemaker",SimpleNamespace(Pipeline=FakePipeline,
        BatchMetadataOperator=object,Probe=PythonProbe))
    monkeypatch.setenv("NOESIS_XC3D_ROOT_OPERATION_LEASE","synthetic-sdk-neutral-fixture-only")
    monkeypatch.setenv("NOESIS_CAMERA_SECRETS_FILE","synthetic-never-read")
    return script,events,observers_alive


def test_clock_sdk_r4_actual_script_one_static_mux_anchor_all_sources_before_observers_and_start(monkeypatch):
    script,events,alive=clock_sdk_r4_fixture(monkeypatch)
    with pytest.raises(RuntimeError,match="synthetic_stop_before_any_source"):
        script.execute_probe({})
    assert not alive
    register=events.index(("register",));start=events.index(("start",));stop=events.index(("stop",));wait=events.index(("wait",))
    assert start<stop<wait
    attach=("attach","mux","noesis_xc3d_sr_anchor_v1","xc3d_sr_pipeline_anchor","src")
    find=("find","mux","xc3d_sr_pipeline_anchor")
    assert len([event for event in events if event[0]=="attach"])==1
    assert len([event for event in events if event[0]=="find"])==1
    assert register<events.index(("add","nvstreammux","mux"))<events.index(attach)<events.index(find)
    for source in range(3):
        assert events.index(("add","nvurisrcbin",f"source{source}"))<events.index(attach)
        observer=("observer",source)
        assert register<events.index(attach)<events.index(find)<events.index(observer)<start<stop
        assert events.index(("destroy_observer",source))>wait
    attempts=[event for event in events if event[0]=="observer_attempt"]
    assert [event[2] for event in attempts]==["CustomObject"]*3 and len({event[3] for event in attempts})==1


@pytest.mark.parametrize("fault,reason",[
    ("registration_false","anchor_registration_failed"),
    ("registration_exception","synthetic_registration_failed"),
    ("foreign_collision","synthetic_foreign_factory_key"),
])
def test_clock_sdk_r3_setup_failure_never_starts_and_cleans_attached_observers(monkeypatch,fault,reason):
    script,events,alive=clock_sdk_r4_fixture(monkeypatch,fault=fault)
    with pytest.raises((RuntimeError,TypeError),match=reason): script.execute_probe({})
    assert ("start",) not in events and ("stop",) in events and not alive
    stop=events.index(("stop",))
    assert all(index>stop for index,event in enumerate(events) if event[0]=="destroy_observer")
    if fault.startswith("registration") or fault=="foreign_collision":
        assert not any(event[0] in {"add","attach","observer"} for event in events)


@pytest.mark.parametrize("fault,reason",[
    ("missing_anchor","pipeline_anchor_missing"),
    ("observer_failed","synthetic_observer_failed"),
    ("node_instead_of_anchor","synthetic_actual_CustomObject_required"),
    ("python_probe_instead_of_anchor","synthetic_actual_CustomObject_required"),
])
def test_clock_sdk_r4_shared_anchor_or_partial_observer_failure_never_starts_and_disconnects_after_stop(monkeypatch,fault,reason):
    script,events,alive=clock_sdk_r4_fixture(monkeypatch,fault=fault)
    with pytest.raises((RuntimeError,TypeError),match=reason): script.execute_probe({})
    assert ("start",) not in events and not alive
    stop=events.index(("stop",))
    assert all(i>stop for i,event in enumerate(events) if event[0]=="destroy_observer")
    if fault=="observer_failed":
        assert ("observer",0) in events and ("observer",1) not in events


def test_clock_sdk_r4_cleanup_failure_is_not_swallowed_and_releases_shared_anchor_observers_after_stop(monkeypatch):
    script,events,alive=clock_sdk_r4_fixture(monkeypatch,fault="stop_failed")
    with pytest.raises(RuntimeError,match="synthetic_stop_failed") as failure: script.execute_probe({})
    assert failure.value.__context__ and "synthetic_stop_before_any_source" in str(failure.value.__context__)
    assert not alive and events.index(("stop",))<min(i for i,event in enumerate(events) if event[0]=="destroy_observer")


def test_clock_sdk_r5_wait_failure_is_not_success_or_observer_release_before_wait(monkeypatch):
    script,events,alive=clock_sdk_r4_fixture(monkeypatch,fault="wait_failed")
    with pytest.raises(RuntimeError,match="synthetic_wait_failed"):
        script.execute_probe({})
    assert not alive
    assert events.index(("stop",))<events.index(("wait",))<min(
        i for i,event in enumerate(events) if event[0]=="destroy_observer")


def test_clock_sdk_r3_native_PUBLIC_factory_noop_static_contract():
    """Source-backed API/static contract ONLY; not a compiled/native conversion test."""
    path=Path(__file__).resolve().parents[1]/"native/noesis_analytics_meta_ext.cpp"
    text=path.read_text()
    handler=text[text.index("class SenderReportAnchorHandler"):text.index("class SenderReportAnchorFactory")]
    assert "return deepstream::probeReturn::Probe_Ok;" in handler
    assert "handleBuffer(deepstream::BufferProbe&, const deepstream::Buffer&)" in handler
    assert "py::" not in handler and "getGObject" not in handler and "extract" not in handler
    factory=text[text.index("class SenderReportAnchorFactory"):text.index("// Root-only source diagnostic.")]
    assert "deepstream::BufferProbe::type()" in factory
    assert "new deepstream::BufferProbe(name, kSenderReportAnchorFactory, new SenderReportAnchorHandler())" in factory
    assert factory.count("new SenderReportAnchorFactory()") == 1
    assert "std::lock_guard<std::mutex> lock(sender_report_factory_mutex)" in factory
    assert "existing != sender_report_factory" in factory and "if (existing) throw" in factory
    assert "if (!common.addCustomFactory" in factory and "sender_report_factory_registered = true" in factory
    assert "delete sender_report_factory" not in factory and "common.unregister" not in factory
    assert "py::init<deepstream::CustomObject&, unsigned int>" in text


def test_clock_sdk_r4_native_PUBLIC_mux_parent_scoped_source_checks_and_RAII_before_hooks():
    """Installed public source/API static checks; real no-start adapter is ROOT-only."""
    path=Path(__file__).resolve().parents[1]/"native/noesis_analytics_meta_ext.cpp"
    text=path.read_text()
    constructor=text[text.index("SenderReportObserver(deepstream::CustomObject&"):text.index("~SenderReportObserver()")]
    checks=("source_id >= 3","dynamic_cast<deepstream::BufferProbe*>","probe->getTarget()",
        "mux->getGObject()","gst_object_get_parent(mux_object)","GST_IS_PIPELINE(pipeline.get())",
        "gst_bin_get_by_name(GST_BIN(pipeline.get()), source_name.c_str())",
        "gst_object_get_parent(object.get())","source_parent.get() != pipeline.get()",
        '!= "nvurisrcbin"',"std::make_shared<SenderReportState>","state_->watch_source")
    indices=[constructor.index(token) for token in checks]
    assert indices==sorted(indices)
    assert 'anchor.getName() != "xc3d_sr_pipeline_anchor"' in constructor and 'mux->getName() != "mux"' in constructor
    assert "mux->getProbe(anchor.getName()) != probe" in constructor and '!= "nvstreammux"' in constructor
    assert "!name || source_name != name.get()" in constructor and "GST_IS_BIN(object.get())" in constructor
    assert 'const std::string source_name = "source" + std::to_string(source_id)' in constructor
    assert "std::unique_ptr<GstObject, GstObjectUnref>" in text and "gst_object_unref(object)" in text
    assert "OwnedGstObject pipeline(" in constructor and "OwnedGstObject object(" in constructor
    assert "OwnedGstObject source_parent(" in constructor
    assert "std::unique_ptr<gchar, decltype(&g_free)> name" in constructor
    assert "gst_bin_get_by_name_recurse_up" not in constructor and "GST_OBJECT_PARENT" not in constructor
    assert "gst_object_ref(source)" in text  # only the checked source reaches watch_source
    assert "gst_element_set_state" not in constructor and "pipeline.start" not in constructor
    script=path.parents[1]/"scripts/probe_source_acquisition.py"
    source=script.read_text()
    assert "_instance" not in source and "unwrap" not in source
    assert "receipts.lookup_key(batch_meta, key)" in source
    assert "plain native mux.src BufferProbe -> Node.find CustomObject" in source
    assert 'pipeline.attach("mux", "noesis_xc3d_sr_anchor_v1", anchor_name, tips="src")' in source
    assert 'anchor = pipeline["mux"].find(anchor_name)' in source
    assert 'f"xc3d_sr_anchor_{source}"' not in source and 'tips="vsrc_0"' not in source


def test_clock_sdk_r5_late_pad_shared_lifetime_and_duplicate_safe_static_contract():
    """Source checks complement, not replace, the real no-network native harness."""
    text=(Path(__file__).resolve().parents[1]/"native/noesis_analytics_meta_ext.cpp").read_text()
    state=text[text.index("class SenderReportState"):text.index("class SenderReportObserver {")]
    attach=state[state.index("void attach_session("):state.index("void watch_source(")]
    assert attach.index('"pad-added"')<attach.index('gst_element_get_static_pad(element, "sync_src")')
    assert "connection.pad && connection.object == GST_OBJECT(pad)" in state
    assert "connections_mutex_" in state and "connections_.size() >= 32" in state
    assert "new Context(shared_from_this())" in state and "release_signal" in state and "release_probe" in state
    assert "closing_.exchange(true)" in state and "connections.swap(connections_)" in state
    assert "static_cast<SenderReportObserver*>(opaque)" not in state
