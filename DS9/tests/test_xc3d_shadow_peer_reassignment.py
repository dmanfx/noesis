"""Typed producer→owner→World→Node regression; all geometry is synthetic."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import time

import pytest


def _examples():
    name = "shadow_peer_quality_examples"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("test_xc3d_world_quality_adapter.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _ShadowReassignment:
    def __init__(self, tmp_path, monkeypatch):
        from noesis.identity_association_service import IdentityAssociationService
        from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
        from noesis.pipelines.source_acquisition import OriginalSwinCache
        from noesis_core.source_progress import DecodedSourceProgressMonitor, SourceProgressPolicy
        from reid.mv3dt_person_association import ProjectiveContinuityPolicy, ProjectiveCameraBinding

        self.m = _examples()
        self.observation_map = {}
        self.frame = 1
        epoch = time.time_ns()
        monkeypatch.setattr(time, "time_ns", lambda: epoch + self.frame * 33_333_333)
        monkeypatch.setattr(time, "time", lambda: time.time_ns() / 1e9)
        monkeypatch.setattr(time, "monotonic", lambda: 100. + self.frame / 30.)
        selected = {}

        def factory(run, spatial):
            p = ((100., 0., 0., 0.), (0., 100., 0., 0.), (0., 0., 1., 0.))
            ex = self.m.world_examples()
            selected["policy"] = ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(
                b, "shadow-reassignment", "raster:" + b.camera_id, (1920, 1080),
                "backend_world_m", ex.frame_binding(b.source_id).target_frame.revision,
                ex.frame_binding(b.source_id).source_camera_calibration_sha256, p,
                ((100., 0., 0., -10. if b.source_id == 1 else 10.), p[1], p[2]),
            ) for b in spatial.sources))
            return IdentityAssociationService.for_estimated_image_regions(
                run, spatial, selected["policy"], live_track_continuity=True,
                semantic_joint_max_mean_px=5., unresolved_birth_quarantine=True)

        self.f = self.m.NativeQualityFixture(tmp_path, identity_factory=factory)
        self.f.ingress.projective_bridge = ImageRegionQualityBridge(selected["policy"], OriginalSwinCache())
        self.monitor = DecodedSourceProgressMonitor(self.f.labels,
            policy=SourceProgressPolicy(poll_interval_seconds=60.), clock=time.monotonic)
        self.monitor.start(lambda failure: None)

    def packet(self, source, frame, raws=(9,), *, shadow=(), no_body=(), no_measurement=(), false_old=False):
        from noesis.pipelines.source_acquisition import freeze_owner_availability, freeze_owner_retention
        from noesis.pipelines.world_quality_adapter import capture_body_support

        self.frame = frame
        # The final native receipt, not this test's allocation call, owns which
        # rows are measured versus genuinely retained SHADOW in this cohort.
        work, rows = self.f.packet(source, frame, raws=tuple(dict.fromkeys((*raws, *shadow))),
                                   scale=4. if source == 1 else 1.)
        tracks, inputs = [], []
        for row, track in zip(work.rows, rows):
            raw = track["tracker_id"]
            if raw in shadow:
                continue
            u = 50. if source == 1 else 40. if raw == 9 else 10.
            dy = 20. if source == 2 and raw == 9 and false_old else 0.
            track["bbox"] = [u - 8., 15. + dy, 16., 40.]
            joints = [(u, 30. + dy, .1)] * 17
            for i, point in {0: (u, 23., .9), 1: (u - 1., 21., .9), 3: (u - 3., 21., .9),
                             5: (u - 2., 30., .9), 6: (u + 2., 30., .9),
                             11: (u - 2., 42., .9), 12: (u + 2., 42., .9)}.items():
                joints[i] = (point[0], point[1] + dy, point[2])
            if raw in no_measurement:
                track.update(world=None, world_valid=False, world_measurement_accepted=False,
                             world_quality="invalid", world_source=None)
            body = None if raw in no_body else capture_body_support(
                track, row.evidence.lifecycle, joints, pose_is_current=True)
            inputs.append(replace(self.m.capture_quality_input(track, row.evidence.lifecycle,
                row.evidence.measurements, row.evidence.resolution, row.evidence.depth), body=body))
            tracks.append(track)
        registry = self.f.registries[source]
        continuity = registry.update_frame(source_id=source, camera_id=self.f.labels[source], frame_id=frame,
            observed_at_us=work.continuity.observed_at_us, media_pts_ns=work.continuity.media_pts_ns,
            tracks=tracks, shadow_tracker_ids=shadow)
        retention = freeze_owner_retention(continuity, registry.identity_retention_keys(source,
            media_pts_ns=work.private.media_pts_ns, max_gap_ns=430_000_000), 430_000_000, work.private.original)
        private = replace(work.private, owner_retention=retention)
        mono = time.monotonic()
        self.monitor.record_progress(source, now=mono)
        availability = freeze_owner_availability(private, continuity, self.monitor, ingress_monotonic_s=mono)
        assert availability is not None and availability.owner is retention
        return replace(self.m.freeze_quality_work(private, continuity, tuple(inputs)),
                       ingress_monotonic_s=mono, owner_availability=availability), tracks

    def publish(self, *packets):
        snapshot = self.f.publish(list(packets))
        for dto in self.memberships():
            for member in dto.members:
                stamp = member.observation
                self.observation_map[stamp.observation_id] = {
                    "source_id": stamp.member.source_id, "tracker_id": stamp.member.raw_id,
                    "frame_id": stamp.frame_id}
        return snapshot

    def memberships(self):
        return self.f.identity.spatial_memberships(at_interval=self.f.ingress._context)

    def establish_then_reassign(self, *, positive=True):
        from noesis.pipelines.source_acquisition import owner_retention_members

        first = self.publish(self.packet(1, 1), self.packet(2, 1))
        old, = first.entities
        assert next(s.camera_id for s in old.sources if s.accepted) == "family-room"
        joined, = self.memberships()
        assert len(joined.members) == 2
        second = self.publish(self.packet(2, 2, no_body=(9,), no_measurement=(9,)))
        held, = second.entities
        assert (held.position, held.observed_at_us, held.stale_after_us) == (
            old.position, old.observed_at_us, old.stale_after_us)
        family = self.packet(2, 3, raws=(10,) if positive else (), shadow=(9,))
        assert family[0].continuity.terminated_tracker_ids == ()
        assert family[0].owner_availability.owner is family[0].private.owner_retention
        retained, measured = owner_retention_members(family[0].private.owner_retention,
            family[0].private.original, family[0].continuity.source_epoch, 3)
        old_member = next(m.observation.member for m in joined.members
                          if m.observation.member.source_id == 2)
        old_native = (old_member.source_epoch, old_member.raw_id, old_member.lifecycle_generation)
        assert old_native in retained and old_native not in measured
        split = self.publish(self.packet(1, 3), family)
        rows = self.memberships()
        old_family = next(dto for dto in rows if any(
            m.observation.member.source_id == 2 and m.observation.member.raw_id == 9 for m in dto.members))
        assert next(m.observation.frame_id for m in old_family.members
                    if m.observation.member == old_member) == 2
        if positive:
            assert old_family.physical_key != joined.physical_key
            current_pair = next(dto for dto in rows if len(dto.members) == 2)
            assert current_pair.physical_key == joined.physical_key
            assert current_pair.association_basis == "estimated_semantic_joint_correspondence"
            assert {(m.observation.member.source_id, m.observation.member.raw_id)
                    for m in current_pair.members} == {(1, 9), (2, 10)}
        return old, split

    def close(self):
        assert self.monitor.stop()["quiesced"]
        self.f.world.close()


def _own_originals(peer, snapshot):
    return [(entity, peer.observation_map[s.observation_id]) for entity in snapshot.entities
            for s in entity.sources if s.accepted]


def test_shadow_peer_positive_reassignment_allows_two_own_current_world_node_originals(tmp_path, monkeypatch):
    peer = _ShadowReassignment(tmp_path, monkeypatch)
    try:
        old, _ = peer.establish_then_reassign()
        returned = peer.publish(peer.packet(1, 4), peer.packet(2, 4, (9, 10)))
        assert len(returned.entities) == 2
        eligibility = {(e.observation.member.source_id, e.observation.member.raw_id): e
                       for dto in peer.memberships() for e in dto.eligibility}
        assert eligibility[(2, 9)].eligible and eligibility[(2, 10)].eligible
        assert not eligibility[(1, 9)].eligible  # Ambiguous K never relocates either body.
        assert eligibility[(2, 9)].observation.lifecycle_started_frame == 1
        own = _own_originals(peer, returned)
        assert {(o["source_id"], o["tracker_id"], o["frame_id"]) for _, o in own} == {(2, 9, 4), (2, 10, 4)}
        assert {(e.position.x, e.position.z) for e, _ in own} == {(1., 2.), (4., 2.)}
        assert len({s.observation_id for e in returned.entities for s in e.sources if s.accepted}) == 2
        assert all(e.observed_at_us > old.observed_at_us and
                   e.stale_after_us == e.observed_at_us + 2_000_000 for e in returned.entities)
        assert old.stale_after_us == old.observed_at_us + 2_000_000
        actual = peer.f.examples.actual_node_wire(peer.f, peer.f.wire, "shadow-positive-reassignment")
        assert all(d["accepted"] for d in actual["diagnostics"])
        assert len(actual["snapshots"][-1]["people"]) == 2
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


@pytest.mark.parametrize("positive", [False, True])
def test_no_measurement_or_no_positive_reassignment_never_borrows_expired_original(tmp_path, monkeypatch, positive):
    peer = _ShadowReassignment(tmp_path, monkeypatch)
    try:
        old, _ = peer.establish_then_reassign(positive=positive)
        raws = (9, 10) if positive else (9,)
        current = peer.publish(peer.packet(1, 4), peer.packet(2, 4, raws,
            false_old=not positive, no_measurement=(9,) if positive else ()))
        for frame in range(9, 70, 5):
            current = peer.publish(peer.packet(1, frame), peer.packet(2, frame, raws,
                false_old=not positive, no_measurement=(9,) if positive else ()))
            assert len(current.entities) <= 1
        assert current.published_at_us > old.stale_after_us
        assert all(o["tracker_id"] != 9 or o["source_id"] != 2 for _, o in _own_originals(peer, current))
        assert all(e.observed_at_us > old.observed_at_us for e in current.entities)
        actual = peer.f.examples.actual_node_wire(peer.f, peer.f.wire,
            "shadow-no-new-authority-" + str(positive))
        assert all(d["accepted"] for d in actual["diagnostics"])
        assert max(len(s["people"]) for s in actual["snapshots"]) <= 1
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()
