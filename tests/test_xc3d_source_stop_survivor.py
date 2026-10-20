"""Explicit peer EOS must not erase an independently current native endpoint.

Synthetic raster/geometry; public adapter, typed native receipts, actual owner,
World, publisher and Node. This is not a reproduction of R104's private state.
"""

from importlib.util import module_from_spec, spec_from_file_location
from dataclasses import replace
from pathlib import Path
import sys
import pytest


def _examples():
    name = "xc3d_source_stop_peer_examples"
    if name not in sys.modules:
        path = Path(__file__).parents[1] / "DS9/tests/test_xc3d_continuous_peer_gap.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_current_family_keeps_existing_key_after_explicit_kitchen_source_stop(tmp_path, monkeypatch):
    m = _examples()
    peer = m._PeerGap(tmp_path, monkeypatch)
    try:
        seed = peer.publish(peer.packet(1, 2), peer.packet(2, 2))
        original, = seed.entities
        assert {s.camera_id for s in m._accepted(original)} == {"family-room"}
        before = peer.publish(peer.packet(2, 3))
        previous, = before.entities
        own_original, = m._accepted(previous)
        current, tracks = peer.packet(2, 4)
        stop = peer.f.ingress.stop_image_region_source(
            1, graph_id=current.private.original.graph_id)
        assert stop.reason == "image_region_source_stopped"
        result = peer.publish((current, tracks))
        family = next(d for d in peer.memberships()
                      if any(v.observation.member.source_id == 2 for v in d.members))
        assert family.physical_key.serial == int(original.entity_id.rsplit(":", 1)[1])
        assert all(e.eligible for e in family.eligibility), family.eligibility
        entity, = result.entities
        assert entity.entity_id == original.entity_id
        assert {s.camera_id for s in m._accepted(entity)} == {"family-room"}
        assert m._accepted(entity)[0].observation_id != own_original.observation_id
        assert entity.observed_at_us > previous.observed_at_us
        assert all(not e.eligible for d in peer.memberships()
                   for e in d.eligibility if e.observation.member.source_id == 1)
        actual = peer.examples.actual_node_wire(peer.f, peer.f.wire, "explicit-peer-source-stop")
        assert all(row["accepted"] for row in actual["diagnostics"])
        assert all(len(row["people"]) == 1 for row in actual["snapshots"])
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


def test_stopped_kitchen_selected_original_is_not_borrowed_by_family(tmp_path, monkeypatch):
    m = _examples()
    peer = m._PeerGap(tmp_path, monkeypatch, selected_source=1)
    try:
        seed = peer.publish(peer.packet(1, 2), peer.packet(2, 2))
        old, = seed.entities
        stopped_original, = m._accepted(old)
        assert stopped_original.camera_id == "kitchen"
        peer.publish(peer.packet(2, 3))
        current, tracks = peer.packet(2, 4)
        peer.f.ingress.stop_image_region_source(1, graph_id=current.private.original.graph_id)
        snapshot = peer.publish((current, tracks))
        entity, = snapshot.entities
        assert {s.camera_id for s in m._accepted(entity)} == {"family-room"}
        assert all(s.observation_id != stopped_original.observation_id for s in m._accepted(entity))
        assert entity.observed_at_us > old.observed_at_us
        actual = peer.examples.actual_node_wire(peer.f, peer.f.wire, "stopped-selected-original")
        assert all(row["accepted"] for row in actual["diagnostics"])
        assert max(len(row["people"]) for row in actual["snapshots"]) == 1
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


def test_surviving_native_receipts_never_renew_an_old_position(tmp_path, monkeypatch):
    m = _examples()
    peer = m._PeerGap(tmp_path, monkeypatch)
    try:
        first = peer.publish(peer.packet(1, 2), peer.packet(2, 2))
        old, = first.entities
        original, = m._accepted(old)
        for frame in range(4, 86, 8):
            current, tracks = peer.packet(2, frame, valid=False)
            if frame == 4:
                peer.f.ingress.stop_image_region_source(1, graph_id=current.private.original.graph_id)
            snapshot = peer.publish((current, tracks))
            assert len(snapshot.entities) <= 1
            for held in snapshot.entities:
                assert (held.position, held.covariance, held.observed_at_us, held.stale_after_us) == (
                    old.position, old.covariance, old.observed_at_us, old.stale_after_us)
                assert tuple(s.observation_id for s in m._accepted(held)) == (original.observation_id,)
        assert snapshot.published_at_us > old.stale_after_us
        assert snapshot.entities == ()
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


def _owner_stop_case(fault=None):
    from tests import test_mv3dt_person_association as m

    owner = m.PersonAssociationLedger("run-a", m.spatial_policy())
    selected, p, rows, c, d = m.local_occupancy_fixture(1, extra=False)
    for row in rows:
        owner.observe_spatial(row, m.SpatialLifecycleReceipt("native:" + row.stamp.observation_id,
            row.stamp, "ACTIVE"), quarantine_unresolved_birth=True)
    assert owner.join_estimated_image_regions(p, projective_policy=selected, at_interval=c,
        delivery_context=d, live_track_continuity=True, semantic_joint_max_mean_px=5.,
        unresolved_birth_quarantine=True).disposition == "accepted"
    kitchen, old_family = rows
    origin = owner._members[old_family.stamp.member].person_key
    birth = owner._members[old_family.stamp.member].image_region_birth_seed
    _, p, updated, _, d = m.local_occupancy_fixture(2, extra=False)
    family = updated[1]
    if fault == "reuse":
        family = replace(family, stamp=replace(family.stamp, lifecycle_started_frame=2))
    elif fault == "epoch":
        binding = replace(m.SOURCES[2], source_epoch=1)
        owner.reset_source(binding)
        family = replace(family, stamp=replace(family.stamp,
            member=replace(family.stamp.member, source_epoch=1)))
    owner.observe_spatial(family, m.SpatialLifecycleReceipt("current-family", family.stamp, "ACTIVE"),
        quarantine_unresolved_birth=True)
    c = m.spatial_context((kitchen, family), sources=tuple(owner._bindings.values()))
    cap = p.owner_availability[1]
    original = p.originals[1]
    if fault in {"reuse", "epoch"}:
        cap = replace(cap, observation=family.stamp,
            retained_members=(family.stamp.member,), active_members=(family.stamp.member,))
        original = replace(original, stamp=family.stamp)
    if fault == "failed_caps": cap = replace(cap, availability_status="failed")
    elif fault == "foreign_graph": cap = replace(cap, graph_id="foreign-graph")
    elif fault == "unretained": cap = replace(cap, retained_members=(), active_members=())
    elif fault == "shadow": cap = replace(cap, shadow_members=(family.stamp.member,))
    p = replace(p, cohort=(kitchen.stamp, family.stamp), originals=() if fault == "missing_body" else (original,),
        owner_availability=() if fault == "missing_caps" else (cap,))
    d = replace(d, ingresses=(m.CurrentVisualIngress(kitchen.stamp, 100.),
        replace(d.ingresses[1], observation=family.stamp)))
    if fault == "stale": d = replace(d, evaluation_monotonic_s=d.evaluation_monotonic_s + .36)
    if fault == "peer_failed_not_stopped":
        p = replace(p, owner_availability=(replace(m.image_owner_capsule(kitchen, 100.),
            availability_status="failed"), cap))
    else:
        assert owner.stop_image_region_source(m.SOURCES[1], graph_id="graph-a").reason == "image_region_source_stopped"
    if fault == "own_stop": owner.stop_image_region_source(m.SOURCES[2], graph_id="graph-a")
    result = owner.join_estimated_image_regions(p, projective_policy=selected, at_interval=c,
        delivery_context=d, live_track_continuity=True, semantic_joint_max_mean_px=5.,
        unresolved_birth_quarantine=True)
    return owner, family, kitchen, origin, birth, c, result


def test_public_stopped_peer_preserves_prior_birth_seed_not_new_birth():
    owner, family, kitchen, origin, birth, c, result = _owner_stop_case()
    current = owner._members[family.stamp.member]
    assert result.corrected_members == (kitchen.stamp.member,)
    assert current.person_key == origin and current.image_region_birth_seed == birth
    assert current.image_region_birth_established and not current.image_region_birth_quarantined
    assert owner.image_region_error(family.stamp, c) is None
    assert owner.image_region_error(kitchen.stamp, c) is not None


@pytest.mark.parametrize("fault", ["missing_body", "missing_caps", "failed_caps", "foreign_graph",
    "unretained", "shadow", "stale", "own_stop", "reuse", "epoch", "peer_failed_not_stopped"])
def test_stopped_peer_does_not_authorize_unqualified_current_endpoint(fault):
    owner, family, _kitchen, _origin, _birth, c, _result = _owner_stop_case(fault)
    assert owner.image_region_error(family.stamp, c) is not None
