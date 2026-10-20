"""Public continuous-camera custody across a genuinely gapped image peer.

Synthetic calibrated rasters/coordinates; real native-owner receipts, image
association, ordered quality ingress, World, publisher, and Node consumer.
"""

from dataclasses import replace
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

def _quality_examples():
    name = "xc3d_continuous_peer_gap_quality_examples"
    if name not in sys.modules:
        path = Path(__file__).with_name("test_xc3d_world_quality_adapter.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class _PeerGap:
    def __init__(self, tmp_path, monkeypatch, *, selected_source=2):
        from noesis.identity_association_service import IdentityAssociationService
        from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
        from noesis.pipelines.source_acquisition import OriginalSwinCache
        from noesis_core.source_progress import DecodedSourceProgressMonitor, SourceProgressPolicy
        from reid.mv3dt_person_association import ProjectiveContinuityPolicy, ProjectiveCameraBinding

        self.q = _quality_examples()
        self.examples = self.q.world_examples()
        # Unlike the baseline fixture's common metric target, the actual image
        # lane permits two independently revision-bound local target charts.
        # Give K a truthful, incompatible synthetic chart with the SAME floor;
        # no transform or coordinate is inferred between these target revisions.
        base = self.examples.metric()
        physical = tuple(self.examples.ArtifactFingerprint(role=role, sha256=digest * 64)
                         for role, digest in (("camera_calibration_physical", "c"),
                                              ("world_alignment_physical", "d")))
        revision = self.examples.metric_frame_revision_sha256(
            "backend_world_m", (0., 1., 0.), 0., tuple(p.sha256 for p in physical))
        frame_ref = type(base.frame)(frame_id="backend_world_m", revision=revision)
        kitchen_target = type(base)(
            contract=base.contract, contract_version=base.contract_version,
            frame=frame_ref, units=base.units, accepted=True, physical_provenance=physical,
            floor_plane=type(base.floor_plane)(frame=frame_ref, normal=(0., 1., 0.), offset_m=0.),
        )
        binding = self.examples.frame_binding
        self.examples.frame_binding = lambda source, target=None: binding(
            source, target=target or (kitchen_target if source == 1 else base))
        monkeypatch.setattr(self.q, "world_examples", lambda: self.examples)
        original_presentation = self.examples.synthetic_presentation

        def presentation(actor):
            bundle = original_presentation(actor)
            data = bundle["data"]
            prototype = next(iter(data["world_frame_presentations"].values()))
            presentations = {}
            for source, edge in actor.frames.items():
                target = edge.target_frame.model_dump(mode="json")
                row = data["cameras"]["frame_bindings"][actor.labels[source]]
                row["world_frame"] = target
                row["metric_frame"]["frame"] = target
                projected = deepcopy(prototype)
                projected["world_frame"] = target
                projected["source_transform_sha256s"] = [edge.target_from_source_sha256]
                presentations[target["revision"]] = projected
            data["world_frame_presentations"] = presentations
            return bundle

        self.examples.synthetic_presentation = presentation
        self.at = 2
        self.epoch = time.time_ns()
        self.selected_source = selected_source
        monkeypatch.setattr(time, "time_ns", lambda: self.epoch + (self.at - 2) * 31_250_000)
        monkeypatch.setattr(time, "time", lambda: time.time_ns() / 1e9)
        monkeypatch.setattr(time, "monotonic", lambda: 100. + (self.at - 2) / 32.)
        selected = {}

        def factory(run, spatial):
            p = ((100., 0., 0., 0.), (0., 100., 0., 0.), (0., 0., 1., 0.))
            policy = ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(
                b, "synthetic-continuous-peer-gap-prior", "raster:" + b.camera_id,
                (1920, 1080), "backend_world_m",
                self.examples.frame_binding(b.source_id).target_frame.revision,
                self.examples.frame_binding(b.source_id).source_camera_calibration_sha256,
                p, ((100., 0., 0., -10. if b.source_id == 1 else 10.), p[1], p[2]),
            ) for b in spatial.sources))
            selected["policy"] = policy
            return IdentityAssociationService.for_estimated_image_regions(
                run, spatial, policy, live_track_continuity=True,
                semantic_joint_max_mean_px=33.724887199080946,
                unresolved_birth_quarantine=True,
            )

        self.f = self.q.NativeQualityFixture(tmp_path, identity_factory=factory)
        self.f.ingress.projective_bridge = ImageRegionQualityBridge(
            selected["policy"], OriginalSwinCache())
        self.monitor = DecodedSourceProgressMonitor(
            self.f.labels, policy=SourceProgressPolicy(poll_interval_seconds=60.),
            clock=time.monotonic,
        )
        self.monitor.start(lambda failure: None)

    def packet(self, source, frame, *, pose=True, valid=True, x=1.):
        from noesis.pipelines.source_acquisition import freeze_owner_availability, freeze_owner_retention
        from noesis.pipelines.world_quality_adapter import capture_body_support

        self.at = frame
        raw = 0 if source == 1 else 3
        work, tracks = self.f.packet(
            source, frame, raws=(raw,), x=x - (raw - 9) * 3.,
            scale=1. if source == self.selected_source else 4.,
            households={raw: {"identity_kind": "visitor", "stable_id": 1004 if source == 1 else 1000,
                              "visitor_generation": 1}},
        )
        inputs = []
        for row, track in zip(work.rows, tracks):
            u = 50. if source == 1 else 40.
            track["bbox"] = [u - 10., 20., 20., 40.]
            joints = [(u, 30., .9)] * 17
            joints[5], joints[6] = (u - 2., 30., .9), (u + 2., 30., .9)
            joints[11], joints[12] = (u - 2., 42., .9), (u + 2., 42., .9)
            if not valid:
                track.update(world=None, world_valid=False, world_measurement_accepted=False,
                             world_quality="invalid", world_source=None)
            body = capture_body_support(track, row.evidence.lifecycle, joints,
                                        pose_is_current=True) if pose else None
            inputs.append(replace(self.q.capture_quality_input(
                track, row.evidence.lifecycle, row.evidence.measurements,
                row.evidence.resolution, row.evidence.depth), body=body))
        registry = self.f.registries[source]
        continuity = registry.update_frame(
            source_id=source, camera_id=self.f.labels[source], frame_id=frame,
            observed_at_us=work.continuity.observed_at_us,
            media_pts_ns=work.continuity.media_pts_ns, tracks=tracks,
        )
        retention = freeze_owner_retention(
            continuity, registry.identity_retention_keys(
                source, media_pts_ns=work.private.media_pts_ns, max_gap_ns=430_000_000),
            430_000_000, work.private.original,
        )
        private = replace(work.private, owner_retention=retention)
        mono = time.monotonic()
        self.monitor.record_progress(source, now=mono)
        availability = freeze_owner_availability(
            private, continuity, self.monitor, ingress_monotonic_s=mono)
        assert availability is not None and availability.owner is retention
        return replace(self.q.freeze_quality_work(private, continuity, tuple(inputs)),
                       ingress_monotonic_s=mono, owner_availability=availability), tracks

    def publish(self, *packets):
        return self.f.publish(list(packets))

    def memberships(self):
        return self.f.identity.spatial_memberships(at_interval=self.f.ingress._context)

    def close(self):
        assert self.monitor.stop()["quiesced"]
        self.f.world.close()


def _accepted(entity):
    return tuple(source for source in entity.sources if source.accepted)


def _gap(peer):
    """F remains current; K media/delivery gap is .533333328/.500s."""
    first = peer.publish(peer.packet(1, 2), peer.packet(2, 2))
    old, = first.entities
    joined, = peer.memberships()
    assert len(joined.members) == 2
    assert joined.association_basis == "estimated_semantic_joint_correspondence"
    for frame in (6, 10, 14, 17):
        peer.publish(peer.packet(2, frame))
    before = peer.f.world.current_snapshot()
    continuous = next(dto for dto in peer.memberships()
                      if any(m.observation.member.source_id == 2 for m in dto.members))
    family = next(m.observation for m in continuous.members if m.observation.member.source_id == 2)
    result = peer.publish(peer.packet(1, 18, pose=False, valid=False), peer.packet(2, 18))
    return SimpleNamespace(first=first, old=old, joined=joined, before=before,
                           family=family, result=result)


def test_continuous_family_keeps_own_original_and_current_authority_through_peer_gap(tmp_path, monkeypatch):
    peer = _PeerGap(tmp_path, monkeypatch)
    try:
        gap = _gap(peer)
        assert {s.camera_id for s in _accepted(gap.old)} == {"family-room"}
        assert gap.old.attribution_quarantine is not None
        assert {c.household_stable_id for c in gap.old.attribution_quarantine.owners} == {1000, 1004}
        kitchen_alternative = next(s for s in gap.old.sources if s.camera_id == "kitchen")
        assert kitchen_alternative.rejection_reason == "registration_identity_conflict"
        assert time.monotonic() == 100.5
        family_dto = next(dto for dto in peer.memberships()
                          if any(m.observation.member.source_id == 2 for m in dto.members))
        family_eligibility = next(e for e in family_dto.eligibility
                                  if e.observation.member.source_id == 2)
        assert family_dto.physical_key == gap.joined.physical_key
        assert family_eligibility.eligible, family_eligibility.reason
        kitchen_dto = next(dto for dto in peer.memberships()
                           if any(m.observation.member.source_id == 1 for m in dto.members))
        kitchen_current = next(m.observation for m in kitchen_dto.members
                               if m.observation.member.source_id == 1)
        kitchen_prior = next(m.observation for m in gap.joined.members
                             if m.observation.member.source_id == 1)
        assert abs(kitchen_current.media_time_s - kitchen_prior.media_time_s - .533333328) < 1e-8
        assert not next(e for e in kitchen_dto.eligibility
                        if e.observation == kitchen_current).eligible
        current, = gap.result.entities
        assert current.entity_id == gap.old.entity_id
        assert {s.camera_id for s in _accepted(current)} == {"family-room"}
        assert current.observed_at_us > gap.old.observed_at_us
        # Cross the initial immutable point's deadline with genuinely new F
        # originals. The absent/UNKNOWN K peer never supplies a second head.
        for frame in range(26, 83, 8):
            snapshot = peer.publish(peer.packet(1, frame, pose=False, valid=False), peer.packet(2, frame))
            entity, = snapshot.entities
            assert entity.entity_id == gap.old.entity_id
            assert {s.camera_id for s in _accepted(entity)} == {"family-room"}
            assert entity.observed_at_us == snapshot.published_at_us
        assert snapshot.published_at_us > gap.old.stale_after_us
        rejoined = peer.publish(peer.packet(1, 84), peer.packet(2, 84))
        assert len(rejoined.entities) == 1
        joined_again, = peer.memberships()
        assert len(joined_again.members) == 2
        assert joined_again.physical_key == gap.joined.physical_key
        actual = peer.examples.actual_node_wire(peer.f, peer.f.wire, "continuous-family-peer-gap")
        assert all(row["accepted"] for row in actual["diagnostics"])
        assert all(len(row["people"]) == 1 for row in actual["snapshots"])
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


def test_kitchen_selected_original_is_not_borrowed_by_continuous_family(tmp_path, monkeypatch):
    peer = _PeerGap(tmp_path, monkeypatch, selected_source=1)
    try:
        gap = _gap(peer)
        kitchen_original, = _accepted(gap.old)
        assert kitchen_original.camera_id == "kitchen"
        assert gap.old.attribution_quarantine is not None
        for entity in gap.result.entities:
            assert all(source.observation_id != kitchen_original.observation_id
                       for source in _accepted(entity))
        for frame in range(26, 83, 8):
            snapshot = peer.publish(peer.packet(1, frame, pose=False, valid=False), peer.packet(2, frame))
            assert len(snapshot.entities) == 1
            assert {s.camera_id for s in _accepted(snapshot.entities[0])} == {"family-room"}
        assert snapshot.published_at_us > gap.old.stale_after_us
        actual = peer.examples.actual_node_wire(peer.f, peer.f.wire, "selected-kitchen-peer-gap")
        assert all(row["accepted"] for row in actual["diagnostics"])
        assert max(len(row["people"]) for row in actual["snapshots"]) == 1
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


def test_current_native_receipts_do_not_renew_a_position_without_new_measurement(tmp_path, monkeypatch):
    peer = _PeerGap(tmp_path, monkeypatch)
    try:
        first = peer.publish(peer.packet(1, 2), peer.packet(2, 2))
        original, = first.entities
        source, = _accepted(original)
        for frame in (6, 10, 14, 17):
            peer.publish(peer.packet(2, frame, valid=False))
        peer.publish(peer.packet(1, 18, pose=False, valid=False),
                     peer.packet(2, 18, valid=False))
        for frame in range(26, 83, 8):
            snapshot = peer.publish(peer.packet(1, frame, pose=False, valid=False),
                                    peer.packet(2, frame, valid=False))
            assert len(snapshot.entities) <= 1
            for held in snapshot.entities:
                assert (held.entity_id, held.position, held.covariance,
                        held.observed_at_us, held.stale_after_us) == (
                            original.entity_id, original.position, original.covariance,
                            original.observed_at_us, original.stale_after_us)
                assert tuple(s.observation_id for s in _accepted(held)) == (source.observation_id,)
        assert snapshot.published_at_us > original.stale_after_us
        assert snapshot.entities == ()
        actual = peer.examples.actual_node_wire(peer.f, peer.f.wire, "peer-gap-no-position-renewal")
        assert all(row["accepted"] for row in actual["diagnostics"])
        assert len(actual["snapshots"][-1]["people"]) == 0
        assert max(len(row["people"]) for row in actual["snapshots"]) == 1
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


def test_both_gapped_peers_do_not_create_continuous_family_authority(tmp_path, monkeypatch):
    peer = _PeerGap(tmp_path, monkeypatch)
    try:
        first = peer.publish(peer.packet(1, 2), peer.packet(2, 2))
        old, = first.entities
        # No intermediate F receipt: both source delivery and image histories
        # are genuinely stale. A current native return alone is not a rejoin.
        returned_family = peer.packet(2, 18)
        result = peer.publish(peer.packet(1, 18, pose=False, valid=False), returned_family)
        family = next(dto for dto in peer.memberships()
                      if any(m.observation.member.source_id == 2 for m in dto.members))
        assert not next(e for e in family.eligibility
                        if e.observation.member.source_id == 2).eligible
        for entity in result.entities:
            assert (entity.observed_at_us, entity.stale_after_us) == (old.observed_at_us, old.stale_after_us)
            assert all(s.frame_id != 18 for s in _accepted(entity))
        rejoined = peer.publish(peer.packet(1, 20), peer.packet(2, 20))
        assert len(rejoined.entities) == 1
        assert rejoined.entities[0].observed_at_us > old.observed_at_us
        actual = peer.examples.actual_node_wire(peer.f, peer.f.wire, "both-peers-gapped-no-authority")
        assert all(row["accepted"] for row in actual["diagnostics"])
        assert max(len(row["people"]) for row in actual["snapshots"]) == 1
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()
