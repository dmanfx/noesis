"""Exact-original priority contract; synthetic CPU inputs, no grants.

These use the actual enqueue/gate and existing native lineage fixture.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from noesis.pipelines import hooks

_path = Path(__file__).with_name("test_xc3d_projective_continuity.py")
_spec = importlib.util.spec_from_file_location("depth_priority_lineage_examples", _path)
_examples = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _examples
_spec.loader.exec_module(_examples)
i138_image_native = _examples.i138_image_native


@pytest.fixture
def publication(i138_image_native, monkeypatch):
    n = i138_image_native
    n.pipeline.config = {
        "xc3d_world_quality": {"contract": "noesis.xc3d.world_quality.v1", "enabled": True},
        "xc3d_image_regions": {"enabled": True},
    }
    n.pipeline.ds_pipeline = object()  # Actual enqueue branch; never starts a thread.
    p = hooks._AnalyticsTelemetryProcessor(pipeline=n.pipeline, tracking_pub=n.f.publisher,
        camera_labels={1: "kitchen", 2: "family-room"}, sensor_id_map={1: 1, 2: 2})
    p._tracking_lifecycle = n.f.registries[2]
    p._tracking_publish_interval_s = 1 / 12
    p.bev_renderer = None
    queued = []
    worker = SimpleNamespace(enqueue=lambda work: queued.append(work) or True)
    monkeypatch.setattr(p, "_ensure_tracking_publication_worker", lambda: worker)
    monkeypatch.setattr(p, "_canonical_publication_backlogged", lambda: False)
    return SimpleNamespace(n=n, p=p, queued=queued, monkeypatch=monkeypatch, identity={})


def packet(c, frame, *, source=2, cached=False, empty=False, terminated=()):
    if empty:
        # One real registry/source-original callback, not a second same-frame event.
        c.n.frame[0] = frame
        update = c.n.f.registries[source].update_frame
        with c.monkeypatch.context() as patch:
            patch.setattr(c.n.f.registries[source], "update_frame", lambda **kwargs:
                update(**{**kwargs, "terminated_tracker_ids": terminated}))
            w, tracks = _examples.lineage_packet(c.n, source, frame, empty=True)
    else:
        w, tracks = c.n.packet(source, frame, raw=3 if source == 2 else 6, terminated=terminated)
    if cached:
        from noesis.pipelines.world_quality_adapter import capture_quality_input, freeze_quality_work
        evidence = []
        for r in w.rows:
            d = replace(r.evidence.depth, measurement_cached=True,
                measurement_frame_id=415, measurement_ts_us=415 * 33_333_333 // 1000)
            evidence.append(capture_quality_input(tracks[0], r.evidence.lifecycle,
                r.evidence.measurements, r.evidence.resolution, d, r.evidence.body))
        w = freeze_quality_work(w.private, w.continuity, tuple(evidence))
    return w, tracks


def enqueue(c, value, now, *, private=None, force=False):
    from noesis.pipelines.source_acquisition import OrderedProducerWork
    w, tracks = value
    c.n.now[0] = now
    # Exact native-unavailable producer packet: no invented metric/identity grant.
    source = w.continuity.source_id
    identity = OrderedProducerWork(w.private.original.run_id, source, w.continuity.source_epoch,
        w.continuity.frame_id, None, denial="native_metric_clock_camera_epoch_original_swin_unavailable")
    c.identity[w.continuity.frame_id] = identity
    temporal = {"observed_at_us": w.continuity.observed_at_us,
        "media_pts_ns": w.continuity.media_pts_ns}
    points = [hooks.Footpoint(tracker_id=t["tracker_id"],
        tracker_lifecycle_generation=t["tracker_lifecycle_generation"], frame_id=w.continuity.frame_id,
        world_x=t["world"][0], world_y=t["world"][1], world_z=t["world"][2]) for t in tracks]
    result = c.p._enqueue_tracking_publication(sensor_id=source, camera_id=w.continuity.camera_id,
        frame_meta=SimpleNamespace(frame_number=w.continuity.frame_id, buf_pts=w.private.media_pts_ns),
        tracks=tracks, footpoints=points, now_ts=now, temporal_contract=temporal,
        continuity=w.continuity, force=force,
        identity_producer_work=identity,
        xc3d_private_evidence=w.private if private is None else private,
        world_quality_inputs=tuple(r.evidence for r in w.rows))
    return result, temporal, points


def seed_and_suppress(c, *, missing_receipt=False):
    first = packet(c, 414)
    assert enqueue(c, first, 100.)[0]
    current = packet(c, 415)
    assert current[0].denial is None and current[0].rows[0].depth_current
    assert current[0].rows[0].current_measured
    private = replace(current[0].private, owner_retention=None) if missing_receipt else None
    suppressed, temporal, points = enqueue(c, current, 100.02, private=private)
    assert not suppressed and [w.frame_id for w in c.queued] == [414]
    return current, temporal, points


def test_i146_due_slot_preserves_complete_suppressed_current_original(publication):
    c = publication
    (original, tracks), temporal, points = seed_and_suppress(c)
    expected_point = tuple(tracks[0]["world"])
    # Mutating the callback's containers must not change the retained packet.
    tracks[0]["world"][0] = 999.
    points[0].world_x = 999.
    temporal["observed_at_us"] = 999
    assert enqueue(c, packet(c, 416, cached=True), 100.10)[0]
    assert [w.frame_id for w in c.queued] == [414, 415]
    selected = c.queued[-1]
    assert selected.xc3d_private_evidence.original is original.private.original
    assert selected.continuity is original.continuity
    assert selected.identity_producer_work is c.identity[415]
    assert selected.identity_producer_work.frame_id == 415
    assert selected.observed_at_us == original.continuity.observed_at_us
    assert selected.timestamp_us == original.private.media_pts_ns // 1000
    assert tuple(selected.tracks[0]["world"]) == expected_point
    assert selected.footpoints[0].world_x == expected_point[0]
    assert selected.world_quality_work.rows[0].current_measured
    assert selected.world_quality_work.rows[0].original is original.private.original
    assert selected.world_quality_work.ingress_monotonic_s == 100.02
    assert selected.xc3d_private_evidence.owner_retention is original.private.owner_retention
    assert all(w.private.original.original_frame < 415 for w in selected.historical_quality_work)
    assert c.p._last_tracking_enqueue_ts_by_sensor[2] == 100.10


def test_i146_pending_original_never_ingests_future_or_duplicate_history(publication):
    c = publication
    seed_and_suppress(c)
    assert not enqueue(c, packet(c, 416, cached=True), 100.04)[0]
    assert enqueue(c, packet(c, 417, cached=True), 100.10)[0]
    assert [w.frame_id for w in c.queued] == [414, 415]
    selected = c.queued[-1]
    assert all(w.private.original.original_frame < 415 for w in selected.historical_quality_work)
    assert enqueue(c, packet(c, 418, cached=True), 100.20)[0]
    later = c.queued[-1]
    assert later.frame_id > 415
    assert any(w.private.original.original_frame == 416 for w in later.historical_quality_work)


def current_floor(value):
    """Validated exact-current floor estimate, not a fabricated owner grant."""
    from noesis.pipelines.world_quality_adapter import capture_quality_input, freeze_quality_work
    from noesis_core.contracts.world_measurement import (
        WorldMeasurementHypothesis, WorldMeasurementSet, ResolvedGroundMeasurement,
    )
    work, tracks = value
    row, = work.rows
    previous = row.evidence.resolution
    candidate = WorldMeasurementHypothesis(candidate_id="synthetic-current-floor",
        cohort=previous.cohort, kind="floor_ray", position=previous.position,
        covariance=previous.covariance, anchor="pose_ankle_floor", posture="standing",
        support_state="floor", pixel_uncertainty_px=1.)
    measurements = WorldMeasurementSet(cohort=previous.cohort, hypotheses=(candidate,))
    resolution = ResolvedGroundMeasurement(cohort=previous.cohort, status="measured",
        position=candidate.position, covariance=candidate.covariance,
        selected_candidate_id=candidate.candidate_id, selected_kind=candidate.kind,
        contributor_ids=(candidate.candidate_id,), quality="estimated", reason="synthetic-current-floor")
    tracks[0]["world_source"] = "pose_floor_only"
    # Cached depth is not current evidence; the independent current floor is.
    depth = replace(row.evidence.depth, measurement_cached=True,
        measurement_frame_id=415, measurement_ts_us=415 * 33_333_333 // 1000)
    evidence = capture_quality_input(tracks[0], row.evidence.lifecycle,
        measurements, resolution, depth, row.evidence.body)
    return freeze_quality_work(work.private, work.continuity, (evidence,)), tracks


def two_member_packet(c, frame, *, cached=(), floors=(2,)):
    """Two actual native births/owner rows; synthetic measurements, no grants."""
    from noesis.pipelines.world_quality_adapter import capture_quality_input, freeze_quality_work
    c.n.frame[0] = frame
    work, tracks = _examples.lineage_packet(c.n, 2, frame, raws=(3, 2))
    inputs = []
    for row, track in zip(work.rows, tracks):
        raw = row.evidence.lifecycle.tracker_id
        if raw in floors:
            floor, _ = current_floor((replace(work, rows=(row,)), [track]))
            evidence = floor.rows[0].evidence
        elif raw in cached:
            depth = replace(row.evidence.depth, measurement_cached=True,
                measurement_frame_id=415, measurement_ts_us=415 * 33_333_333 // 1000)
            evidence = capture_quality_input(track, row.evidence.lifecycle,
                row.evidence.measurements, row.evidence.resolution, depth, row.evidence.body)
        else:
            evidence = row.evidence
        inputs.append(evidence)
    frozen = freeze_quality_work(work.private, work.continuity, tuple(inputs))
    assert frozen.denial is None
    assert len(frozen.private.owner_retention.keys) == len(frozen.rows) == 2
    return frozen, tracks


@pytest.mark.parametrize("other_kind", ["floor", "depth"])
def test_i153_other_members_current_point_cannot_supersede_pending_depth(publication, other_kind):
    c = publication
    floors = (2,) if other_kind == "floor" else ()
    assert enqueue(c, two_member_packet(c, 414, floors=floors), 100.)[0]
    pending = two_member_packet(c, 415, floors=floors)
    assert not enqueue(c, pending, 100.02)[0]
    current = two_member_packet(c, 416, floors=floors, cached=(3,))
    cap, seat = current[0].rows
    assert not cap.current_measured and seat.current_measured
    assert enqueue(c, current, 100.10)[0]
    assert [w.frame_id for w in c.queued] == [414, 415]
    selected = c.queued[-1]
    assert selected.xc3d_private_evidence.original is pending[0].private.original
    assert selected.continuity is pending[0].continuity
    assert selected.identity_producer_work is c.identity[415]
    assert selected.world_quality_work.ingress_monotonic_s == 100.02
    assert selected.observed_at_us == pending[0].continuity.observed_at_us
    assert all(r.original is pending[0].private.original for r in selected.world_quality_work.rows)
    assert all(w.continuity.frame_id < 415 for w in selected.historical_quality_work)
    assert enqueue(c, two_member_packet(c, 417, floors=floors), 100.20)[0]
    assert [w.continuity.frame_id for w in c.queued[-1].historical_quality_work] == [416]


@pytest.mark.parametrize("same_member_kind", ["floor", "depth"])
def test_i153_same_member_current_successor_supersedes_pending_depth(publication, same_member_kind):
    c = publication
    assert enqueue(c, two_member_packet(c, 414), 100.)[0]
    assert not enqueue(c, two_member_packet(c, 415), 100.02)[0]
    current = two_member_packet(c, 416, floors=(2, 3) if same_member_kind == "floor" else (2,))
    assert all(r.denial is None and r.current_measured for r in current[0].rows)
    assert enqueue(c, current, 100.10)[0]
    assert [w.frame_id for w in c.queued] == [414, 416]
    assert c.queued[-1].xc3d_private_evidence.original is current[0].private.original
    assert [w.continuity.frame_id for w in c.queued[-1].historical_quality_work] == [415]


def test_i153_existing_forced_current_dispatch_wins_over_pending_depth(publication):
    c = publication
    assert enqueue(c, two_member_packet(c, 414), 100.)[0]
    assert not enqueue(c, two_member_packet(c, 415), 100.02)[0]
    current = two_member_packet(c, 416, cached=(3,))
    assert not current[0].rows[0].current_measured and current[0].rows[1].current_measured
    # The real caller's existing first-metric/lifecycle force must not be
    # hidden by priority for a different member, even before the normal slot.
    assert enqueue(c, current, 100.04, force=True)[0]
    assert [w.frame_id for w in c.queued] == [414, 416]
    assert c.queued[-1].xc3d_private_evidence.original is current[0].private.original
    assert [w.continuity.frame_id for w in c.queued[-1].historical_quality_work] == [415]
    assert not c.p._depth_priority_pending


@pytest.mark.parametrize("all_refreshed", [False, True])
@pytest.mark.parametrize("intervening_suppressed", [False, True])
def test_i153_all_pending_depth_members_must_be_refreshed(publication, all_refreshed, intervening_suppressed):
    c = publication
    assert enqueue(c, two_member_packet(c, 414, floors=()), 100.)[0]
    pending = two_member_packet(c, 415, floors=())
    assert all(r.depth_current and r.current_measured for r in pending[0].rows)
    assert not enqueue(c, pending, 100.02)[0]
    current = two_member_packet(c, 416, floors=(3,) if all_refreshed else (),
        cached=() if all_refreshed else (2,))
    if intervening_suppressed:
        assert not enqueue(c, current, 100.04)[0]
        assert enqueue(c, two_member_packet(c, 417, floors=(), cached=(3, 2)), 100.10)[0]
    else:
        assert enqueue(c, current, 100.10)[0]
    expected = 416 if all_refreshed else 415
    assert [w.frame_id for w in c.queued] == [414, expected]
    selected = c.queued[-1]
    original = current[0].private.original if all_refreshed else pending[0].private.original
    assert selected.xc3d_private_evidence.original is original
    assert all(r.original is original for r in selected.world_quality_work.rows)
    assert all(w.continuity.frame_id < expected for w in selected.historical_quality_work)


@pytest.mark.parametrize("kind", ["registered_depth", "floor_ray"])
def test_i148_current_measured_successor_wins_and_superseded_history_is_ordered(publication, kind):
    c = publication
    seed_and_suppress(c)
    assert not enqueue(c, packet(c, 416, cached=True), 100.04)[0]
    successor = packet(c, 417)
    if kind == "floor_ray":
        successor = current_floor(successor)
    row, = successor[0].rows
    assert successor[0].denial is None and row.denial is None and row.current_measured
    assert row.evidence.resolution.selected_kind == kind
    assert row.depth_current is (kind == "registered_depth")
    assert enqueue(c, successor, 100.10)[0]
    assert [w.frame_id for w in c.queued] == [414, 417]
    selected = c.queued[-1]
    assert selected.xc3d_private_evidence.original is successor[0].private.original
    assert selected.world_quality_work.rows[0].current_measured
    assert selected.identity_producer_work is c.identity[417]
    history = selected.historical_quality_work
    assert [(w.continuity.source_id, w.continuity.frame_id) for w in history] == [(2, 415), (2, 416)]
    # Optional retention intentionally filters absent body-depth/Swin rows;
    # it must preserve the original lifecycle/barrier, not synthesize a vote.
    assert all(w.rows == () for w in history)
    assert all(w.private.owner_retention is not None for w in history)
    assert len({w.private.original for w in history}) == len(history)
    assert all(w.continuity.frame_id < selected.frame_id for w in history)


def test_i148_other_source_cannot_deliver_pending_camera_future_history(publication):
    c = publication
    (pending, _), _, _ = seed_and_suppress(c)
    assert not enqueue(c, packet(c, 416, cached=True), 100.04)[0]
    other = packet(c, 300, source=1)
    assert enqueue(c, other, 100.05)[0]
    kitchen = c.queued[-1]
    assert (kitchen.source_id, kitchen.frame_id) == (1, 300)
    assert not any(w.continuity.source_id == 2 and w.continuity.frame_id >= 415
        for w in kitchen.historical_quality_work)
    assert enqueue(c, packet(c, 417, cached=True), 100.10)[0]
    selected = c.queued[-1]
    assert (selected.source_id, selected.frame_id) == (2, 415)
    assert selected.xc3d_private_evidence.original is pending.private.original
    assert all(w.continuity.source_id != 2 or w.continuity.frame_id < 415
        for w in selected.historical_quality_work)
    assert enqueue(c, packet(c, 418, cached=True), 100.20)[0]
    later = c.queued[-1]
    family_history = [w for w in later.historical_quality_work if w.continuity.source_id == 2]
    assert [w.continuity.frame_id for w in family_history] == [416, 417]
    assert len({w.private.original for w in family_history}) == 2
    assert [w.frame_id for w in c.queued] == [414, 300, 415, 418]


@pytest.mark.parametrize("fault", ["missing-receipt", "epoch-reset", "absence", "termination", "stop", "reorder", "default-off"])
def test_i146_invalid_or_superseded_pending_never_reappears(publication, fault):
    c = publication
    if fault == "default-off": c.n.pipeline.config["xc3d_image_regions"]["enabled"] = False
    seed_and_suppress(c, missing_receipt=fault == "missing-receipt")
    if fault == "epoch-reset": c.n.f.registries[2].reset_source(2)
    if fault == "stop": c.n.pipeline._xc3d_applied_source_stop_mask = 1 << 2
    if fault == "reorder":
        from noesis.pipelines.source_acquisition import AcquisitionUnavailable
        with pytest.raises(AcquisitionUnavailable, match="frame_clock_duplicate_or_regression"):
            packet(c, 413)
        assert [w.frame_id for w in c.queued] == [414]
        return  # Rejected source input cannot manufacture an enqueue/barrier.
    value = packet(c, 416,
        cached=fault not in {"absence", "termination"},
        empty=fault in {"absence", "termination"},
        terminated=(3,) if fault == "termination" else ())
    enqueue(c, value, 100.10)
    assert c.queued[-1].frame_id != 415
    assert all(w.frame_id != 415 for w in c.queued)
