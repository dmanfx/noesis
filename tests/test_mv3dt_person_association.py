from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest

from reid.mv3dt_person_association import (
    AcquisitionTime,
    AppearanceEvidence,
    AppearanceProfile,
    AssociationPolicy,
    HouseholdDescriptor,
    LifecycleRetention,
    LinkProof,
    MeasurementEvidence,
    MemberKey,
    MemberRemoval,
    NativeTransition,
    ObservationStamp,
    PersonAssociationLedger,
    PhysicalPersonKey,
    SourceAbsence,
    SourceBinding,
    SourceObservation,
    appearance_extraction_identity,
    SpatialAssociationPolicy,
    SpatialClockContext,
    SpatialClockInterval,
    SpatialCurrentPersonProof,
    SpatialLifecycleReceipt,
    SpatialPersonCandidate,
    CurrentVisualProof,
    CurrentVisualIngress,
    CurrentVisualDeliveryContext,
    ProjectiveCameraBinding, ProjectiveContinuityPolicy, ProjectiveMeasurementBinding,
    ProjectiveMotionSample, ProjectiveMotionPair, ProjectiveContinuityProof,
)


SOURCES = (
    SourceBinding("living-room", 0, 0, "cal-l", "transform-l", "capture-clock", "synchronized-media"),
    SourceBinding("kitchen", 1, 0, "cal-k", "transform-k", "capture-clock", "synchronized-media"),
    SourceBinding("family-room", 2, 0, "cal-f", "transform-f", "capture-clock", "synchronized-media"),
)


def _fundamental_exact_result(call, left, right):
    """Compare numeric bytes (including signed zero) or direct-call exceptions."""
    try:
        result = call(left, right)
    except Exception as exc:
        return ('error', type(exc), str(exc))
    return ('value', None if result is None else tuple(tuple(
        (type(v), v.hex() if type(v) is float else v) for v in row) for row in result))


def _fundamental_configured_pairs():
    """Literal three-camera config and accepted task prior, not overlap admission."""
    import json
    from pathlib import Path
    import numpy as np
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'config/cameras.yaml').read_text())
    calibration = json.loads((root / 'config/camera_calibration.json').read_text())['cameras']
    projections = {}
    freeze = lambda m: tuple(tuple(float(v) for v in row) for row in m)
    for camera in config['cameras'].values():
        k = config['intrinsics_models'][camera['model']]['intrinsics']
        K = np.array(((k['fx'], 0., k['cx']), (0., k['fy'], k['cy']), (0., 0., 1.)))
        E = np.asarray(calibration[camera['name']]['E']).reshape(4, 4, order='F')
        projections[camera['name']] = freeze(K @ E[:3])
    pairs = [(projections[a], projections[b]) for a, b in permutations(projections, 2)]
    prior = json.loads((root / 'DS9/config/xc3d_image_regions_prior_r1.json').read_text())
    for camera in prior['cameras']:
        K = np.asarray(camera['intrinsics'])
        E = np.asarray(camera['world_extrinsics_col_major']).reshape(4, 4, order='F')
        pairs.append((freeze(K @ E[:3]), freeze(camera['candidate_peer_projection'])))
    return pairs


def test_region_fundamental_cache_configured_three_camera_and_task_prior_exact_parity(monkeypatch):
    cached = PersonAssociationLedger._region_fundamental_cached
    original = PersonAssociationLedger._region_fundamental_uncached
    cached.cache_clear()
    calls = []
    def count(left, right):
        calls.append((left, right))
        return original(left, right)
    monkeypatch.setattr(PersonAssociationLedger, '_region_fundamental_uncached', staticmethod(count))
    pairs = _fundamental_configured_pairs()
    assert len(pairs) == 8
    for left, right in pairs:
        expected = _fundamental_exact_result(original, left, right)
        assert _fundamental_exact_result(PersonAssociationLedger._region_fundamental, left, right) == expected
        cold = len(calls)
        # Equal newly allocated literal tuples reuse VALUES, not object identity.
        equal = tuple(tuple(float(v) for v in row) for row in left)
        assert _fundamental_exact_result(PersonAssociationLedger._region_fundamental, equal, right) == expected
        assert len(calls) == cold
    assert len(calls) == 8 and cached.cache_info().currsize == 8
    cached.cache_clear()


def test_region_fundamental_cache_changed_reverse_and_bounded_eviction():
    cached = PersonAssociationLedger._region_fundamental_cached
    cached.cache_clear()
    left, right = _fundamental_configured_pairs()[-1]
    call = PersonAssociationLedger._region_fundamental
    uncached = PersonAssociationLedger._region_fundamental_uncached
    for a, b in ((left, right), (right, left)):
        assert _fundamental_exact_result(call, a, b) == _fundamental_exact_result(uncached, a, b)
    for i in range(20):
        changed = (left[0][:-1] + (left[0][-1] + i + 1.,),) + left[1:]
        assert _fundamental_exact_result(call, changed, right) == _fundamental_exact_result(uncached, changed, right)
    assert cached.cache_info().currsize == cached.cache_info().maxsize == 16
    misses = cached.cache_info().misses
    call(left, right)
    assert cached.cache_info().misses == misses + 1
    cached.cache_clear()


@pytest.mark.parametrize('kind', ['mutable', 'short', 'nan', 'inf', 'bool', 'huge', 'subclass'])
def test_region_fundamental_cache_direct_unaccepted_inputs_preserve_behavior(kind):
    cached = PersonAssociationLedger._region_fundamental_cached
    cached.cache_clear()
    left, right = _fundamental_configured_pairs()[-1]
    if kind == 'mutable':
        left = [list(row) for row in left]
    elif kind == 'short':
        left = left[:2]
    else:
        class Scalar(float):
            pass
        value = {'nan': float('nan'), 'inf': float('inf'), 'bool': True,
                 'huge': 10**1000, 'subclass': Scalar(1.)}[kind]
        left = ((value,) + left[0][1:],) + left[1:]
    call = PersonAssociationLedger._region_fundamental
    uncached = PersonAssociationLedger._region_fundamental_uncached
    assert _fundamental_exact_result(call, left, right) == _fundamental_exact_result(uncached, left, right)
    if kind == 'mutable':
        left[0][3] += 123.
        assert _fundamental_exact_result(call, left, right) == _fundamental_exact_result(uncached, left, right)
    assert cached.cache_info().currsize == cached.cache_info().misses == 0


def test_region_fundamental_cache_signed_zero_scalar_type_and_degenerate_none():
    cached = PersonAssociationLedger._region_fundamental_cached
    cached.cache_clear()
    left = ((100., 0., 0., 0.), (0., 100., 0., 0.), (0., 0., 1., 0.))
    right = (left[0][:-1] + (100.,),) + left[1:]
    variants = [left, ((100., -0., 0., 0.),) + left[1:], ((100, 0., 0., 0.),) + left[1:]]
    for value in variants:
        assert _fundamental_exact_result(PersonAssociationLedger._region_fundamental, value, right) == _fundamental_exact_result(
            PersonAssociationLedger._region_fundamental_uncached, value, right)
    assert cached.cache_info().misses == 3
    for degenerate in (left, ((0.,) * 4,) * 3):
        assert PersonAssociationLedger._region_fundamental(degenerate, degenerate) is None
        before = cached.cache_info().misses
        assert PersonAssociationLedger._region_fundamental(degenerate, degenerate) is None
        assert cached.cache_info().misses == before
    cached.cache_clear()


@pytest.mark.parametrize('case', ['late_birth', 'closed', 'compacted', 'retired_generation'])
def test_observation_refusal_diagnostic_actual_branch_detached_finite_and_no_authority(case):
    import json
    owner = PersonAssociationLedger('run-a', policy())
    a = observation(raw=7, frame=10, time=10., media=5., started=1,
                    generation=2 if case == 'retired_generation' else 1)
    assert owner.observe(a).disposition == 'observed'
    assert owner.withdraw(MemberRemoval('real-removal', a.stamp, 'termination', 20, 11.,
                                       a.stamp.clock_domain, 5.1)).disposition == 'removed'
    if case == 'compacted':
        assert owner.observe(observation(raw=9, frame=21, time=11.1, media=5.2, started=21)).disposition == 'observed'
    candidate = observation(raw=8 if case == 'late_birth' else 7, frame=22, time=11.2,
                            media=5.3, started=15 if case == 'late_birth' else 1,
                            generation=1)
    before = (owner.stats(), owner._authority_revision, dict(owner._lifecycle_frontiers))
    result = owner.observe(candidate)
    expected = {'late_birth': 'lifecycle_frontier', 'closed': 'closed_lifecycle',
                'compacted': 'lifecycle_frontier', 'retired_generation': 'retired_lifecycle_generation'}[case]
    assert (result.disposition, result.reason) == ('unresolved', expected)
    assert before == (owner.stats(), owner._authority_revision, dict(owner._lifecycle_frontiers))
    note = owner.observation_refusal_diagnostic()
    assert note['stamp']['member']['raw_id'] == candidate.stamp.member.raw_id
    assert note['stamp']['lifecycle_started_frame'] == candidate.stamp.lifecycle_started_frame
    assert note['source_frontier'] == 20 and note['current_stamp'] is None
    assert (note['closed_kind'] == 'termination') == (case == 'closed')
    assert (note['compacted_closed_total'] > 0) == (case == 'compacted')
    assert len(json.dumps(note, allow_nan=False).encode()) <= 8192
    note['stamp']['member']['raw_id'] = 999
    assert owner.observation_refusal_diagnostic()['stamp']['member']['raw_id'] != 999
    owner.reset_run('run-new')
    assert owner.observation_refusal_diagnostic() is None


def test_observation_refusal_diagnostic_capture_failure_cannot_mask_refusal(monkeypatch):
    import reid.mv3dt_person_association as module
    owner = PersonAssociationLedger('run-a', policy())
    a = observation(frame=10, started=1)
    owner.observe(a)
    owner.withdraw(MemberRemoval('end', a.stamp, 'termination', 20, 11., a.stamp.clock_domain, 5.1))
    monkeypatch.setattr(module.json, 'dumps', lambda *a, **k: (_ for _ in ()).throw(ValueError('observer only')))
    assert owner.observe(observation(raw=8, frame=21, time=11.1, media=5.2, started=15)).reason == 'lifecycle_frontier'
    assert owner.observation_refusal_diagnostic() is None


def policy(**changes: object) -> AssociationPolicy:
    return replace(AssociationPolicy(SOURCES, "room-metric", "revision-1", "selected-swin",
                                     AppearanceProfile("selected-swin", "synthetic-layer", 512, "synthetic-fingerprint"),
                                     (("kitchen", "family-room"),)), **changes)


def observation(camera: str = "kitchen", raw: int = 7, *, frame: int = 1,
                time: float = 10.0, media: float | None = 5.0, epoch: int = 0,
                generation: int = 1, run: str = "run-a",
                started: int = 1,
                household: HouseholdDescriptor = HouseholdDescriptor()) -> SourceObservation:
    source = {"living-room": 0, "kitchen": 1, "family-room": 2}[camera]
    key = MemberKey(run, camera, source, epoch, raw, generation)
    stamp = ObservationStamp(key, frame, f"{camera}-{raw}-{generation}-{frame}", time,
                             "capture-clock", media, "synchronized-media", started)
    return SourceObservation(stamp, household)


def now(time: float = 10.05) -> AcquisitionTime:
    return AcquisitionTime(time, "capture-clock", "synchronized-media")


def test_household_owner_view_is_read_only_original_current_and_owned_grace():
    owner = PersonAssociationLedger("run-a", policy())
    house = HouseholdDescriptor("resident", "original-resident", 9)
    left, right = observation(household=house), observation("family-room", 8, household=house)
    first, second = owner.observe(left), owner.observe(right)
    rows = owner.household_claim_owners()
    assert {k for k, _ in rows} == {first.person_key, second.person_key}
    assert {o for _, o in rows} == {left, right}
    joined = owner.link(proof(left, right), now())
    receipt = LifecycleRetention("original-owned-grace", left.stamp, left.stamp.member, "inactive", .35)
    owner.withdraw(removal(left, retention=receipt))
    before, revision = owner.stats(), owner.authority_revision
    rows = owner.household_claim_owners()
    assert owner.stats() == before and owner.authority_revision == revision
    assert {k for k, _ in rows} == {joined.person_key}
    assert {o for _, o in rows} == {left, right}
    assert all(o.household is house for _, o in rows)
    assert next(o for _, o in rows if o.stamp == left.stamp).stamp is left.stamp
    owner.expire_retention(1, 5.35)  # Exact original deadline still belongs to owner.
    assert len(owner.household_claim_owners()) == 2
    owner.expire_retention(1, 5.351)
    assert owner.household_claim_owners() == ((joined.person_key, right),)
    owner.withdraw(removal(right, kind="termination", token="right-terminated"))
    assert owner.household_claim_owners() == ()


def spatial_policy(**changes):
    return replace(SpatialAssociationPolicy(SOURCES, policy().appearance_profile,
                                           (("kitchen", "family-room"),)), **changes)


def spatial_context(rows, *, evaluation=None, qualified=False, sources=SOURCES):
    intervals = []
    for source in sorted({r.stamp.member.source_id for r in rows}):
        stamps = [r.stamp for r in rows if r.stamp.member.source_id == source]
        # One actual acquisition cohort per source. Older B membership is NOT
        # folded into A's current acquisition uncertainty or renewed by it.
        high = max(s.acquired_at_s for s in stamps)
        low = high
        evaluated = high if evaluation is None else evaluation
        intervals.append(SpatialClockInterval(next(b for b in sources if b.source_id == source),
                                             low, high, evaluated, evaluated))
    offsets = tuple((i.binding.source_id, 0., 0.) for i in intervals)
    valid = tuple((i.binding.source_id, 0., 1000.) for i in intervals)
    return SpatialClockContext(tuple(intervals), "synthetic-comparability" if qualified else None,
                               "synthetic-common" if qualified else None, offsets if qualified else (),
                               valid if qualified else (), "synthetic" if qualified else None)


def spatial_person_proof(rows, *, selected=None, similarity=.9, distance=.4, token="spatial-proof"):
    kitchens = [r for r in rows if r.stamp.member.camera_id == "kitchen"]
    families = [r for r in rows if r.stamp.member.camera_id == "family-room"]
    candidates = []
    for left in kitchens:
        for right in families:
            pair = proof(left, right)
            same = (left.stamp.member.raw_id, right.stamp.member.raw_id) in {(7, 8), (17, 18)}
            score = similarity if same else .2
            bindings = SOURCES[left.stamp.member.source_id], SOURCES[right.stamp.member.source_id]
            candidates.append(SpatialPersonCandidate(left.stamp, right.stamp, pair.left_appearance,
                              pair.right_appearance, score, "copy:" + left.stamp.observation_id,
                              "copy:" + right.stamp.observation_id, distance if same else 3.,
                              "synthetic-geometry:" + left.stamp.observation_id + right.stamp.observation_id,
                              "synthetic-qualified-frame", "synthetic-qualified-revision",
                              tuple(b.calibration_revision for b in bindings), tuple(b.transform_id for b in bindings),
                              ("independent:" + left.stamp.observation_id, "independent:" + right.stamp.observation_id)))
    selected = selected or tuple((c.left.member, c.right.member) for c in candidates if c.similarity >= .6)
    return SpatialCurrentPersonProof(token, tuple(r.stamp for r in rows), tuple(candidates), selected, "synthetic")


def spatial_kernel_pair():
    owner = PersonAssociationLedger("run-a", spatial_policy())
    rows = (observation(), observation("family-room", 8))
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("lifecycle:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    return owner, rows, spatial_person_proof(rows), spatial_context(rows, qualified=True)


def test_spatial_kernel_one_ledger_local_then_supported_two_member_and_default_strict():
    owner, rows, candidate, context = spatial_kernel_pair()
    assert not hasattr(owner.policy, "target_frame") and not hasattr(owner.policy, "world_revision")
    first = owner._members[rows[0].stamp.member].person_key
    assert owner.join_spatial(candidate, at_interval=context).disposition == "accepted"
    assert len(owner._persons) == 1 and owner._members[rows[1].stamp.member].person_key == first
    assert owner.link(proof(*rows), now()).reason == "common_metric_policy_required"
    strict, _, _, _, _, pair = setup_pair()
    assert strict.link(pair, now()).disposition == "accepted"
    assert strict.join_spatial(candidate, at_interval=context).reason == "spatial_policy_required"


@pytest.mark.parametrize("defect", ["naked-boolean", "clock", "incomplete", "tie", "crossing", "geometry",
                                     "separation", "copy", "profile", "retimestamp", "household"])
def test_spatial_kernel_competing_or_unqualified_new_joins_never_merge(defect):
    owner, rows, candidate, context = spatial_kernel_pair()
    if defect in {"tie", "crossing", "incomplete"}:
        rows += (observation(raw=17), observation("family-room", 18))
        for row in rows[2:]:
            owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        candidate = spatial_person_proof(rows)
        context = spatial_context(rows, qualified=True)
        if defect == "incomplete":
            candidate = replace(candidate, candidates=candidate.candidates[:1])
        else:
            candidate = replace(candidate, candidates=tuple(replace(c, similarity=.9, separation_m=.2) for c in candidate.candidates))
    elif defect == "naked-boolean":
        candidate = True
    elif defect == "clock":
        context = spatial_context(rows)
    else:
        c = candidate.candidates[0]
        if defect == "geometry":
            c = replace(c, geometry_receipt_id=None)
        elif defect == "separation":
            c = replace(c, separation_m=1.251)
        elif defect == "copy":
            c = replace(c, right_copy_id=c.left_copy_id)
        elif defect == "profile":
            c = replace(c, left_appearance=replace(c.left_appearance, profile=replace(policy().appearance_profile, fingerprint="wrong")))
        elif defect == "retimestamp":
            c = replace(c, left_appearance=replace(c.left_appearance, acquired_at_s=10.01))
        else:
            resident = replace(rows[0], household=HouseholdDescriptor("resident", "resident-a", 9))
            visitor = replace(rows[1], household=HouseholdDescriptor("visitor", "visitor-b", 10, 4))
            owner = PersonAssociationLedger("run-a", spatial_policy())
            for row in (resident, visitor):
                owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        candidate = replace(candidate, candidates=(c,))
    before = owner.stats().active_persons
    result = owner.join_spatial(candidate, at_interval=context)
    assert result.disposition == "association_unresolved", result
    assert owner.stats().active_persons == before


def test_spatial_kernel_complete_two_person_batch_and_post_frontier_stale_proof():
    owner, rows, _, _ = spatial_kernel_pair()
    rows += (observation(raw=17), observation("family-room", 18))
    for row in rows[2:]:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    original = spatial_person_proof(rows)
    assert owner.join_spatial(original, at_interval=spatial_context(rows, qualified=True)).disposition == "accepted"
    assert owner.stats().active_persons == 2
    newer = observation(frame=2, time=10.1, media=5.1)
    owner.observe_spatial(newer, SpatialLifecycleReceipt("next-native", newer.stamp, "ACTIVE"))
    owner.compact_history()
    assert owner.join_spatial(original, at_interval=spatial_context(rows, evaluation=10.1, qualified=True)).disposition != "accepted"
    assert owner.join_spatial(replace(original, proof_id="renamed-old"), at_interval=spatial_context(rows, evaluation=10.1, qualified=True)).disposition != "accepted"


@pytest.mark.parametrize("defect", ["none", "missing-receipt", "uncertainty", "expired-mapping"])
def test_spatial_kernel_distinct_source_clock_offsets_need_qualified_worst_case_proof(defect):
    sources = tuple(replace(b, clock_domain="family-own-clock") if b.source_id == 2 else b for b in SOURCES)
    owner = PersonAssociationLedger("run-a", spatial_policy(sources=sources))
    left = observation()
    right = observation("family-room", 8, time=100., media=50.)
    right = replace(right, stamp=replace(right.stamp, clock_domain="family-own-clock"))
    rows = left, right
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    context = spatial_context(rows, sources=sources, qualified=True)
    context = replace(context, offsets=((1, 0., 0.), (2, -90., -90.)))
    if defect == "missing-receipt":
        context = replace(context, comparability_receipt_id=None)
    elif defect == "uncertainty":
        context = replace(context, offsets=((1, -.2, .2), (2, -90., -90.)))
    elif defect == "expired-mapping":
        context = replace(context, valid_ranges=((1, 0., 11.), (2, 0., 99.)))
    result = owner.join_spatial(spatial_person_proof(rows), at_interval=context)
    assert (result.disposition == "accepted") is (defect == "none")
    assert owner.stats().active_persons == (1 if defect == "none" else 2)


def visual_proof(rows, *, scores=None, originals=None, token="visual"):
    spatial = spatial_person_proof(rows)
    original_by_member = {r.stamp.member: r for r in (originals or rows)}
    candidates = []
    for candidate in spatial.candidates:
        left, right = original_by_member[candidate.left.member], original_by_member[candidate.right.member]
        original = proof(left, right)
        score = (scores or {}).get((candidate.left.member.raw_id, candidate.right.member.raw_id), candidate.similarity)
        candidates.append(replace(candidate, left_appearance=original.left_appearance,
                                  right_appearance=original.right_appearance, similarity=score,
                                  left_copy_id="original-copy:" + left.stamp.observation_id,
                                  right_copy_id="original-copy:" + right.stamp.observation_id,
                                  separation_m=None, geometry_receipt_id=None, geometry_frame=None,
                                  geometry_revision=None, calibration_revisions=(), transform_ids=(), independence_ids=()))
    viable = [(c.left.member, c.right.member) for c in candidates if c.similarity >= .60]
    selected = tuple(p for p in viable if all(sum(k in e for e in viable) == 1 for k in p))
    return CurrentVisualProof(token, spatial.cohort, tuple(candidates), selected)


def visual_delivery(rows, *, sequence=1, evaluation=100., ingress=None, originals=None, original_ingress=None):
    by_member = {r.stamp.member: r for r in (originals or rows)}
    ingress = evaluation if ingress is None else ingress
    original_ingress = ingress if original_ingress is None else original_ingress
    return CurrentVisualDeliveryContext("synthetic-one-ordered-actor", sequence, evaluation,
        tuple(CurrentVisualIngress(r.stamp, ingress, by_member[r.stamp.member].stamp, original_ingress) for r in rows))


def visual_kernel(rows=None):
    rows = rows or (observation(), observation("family-room", 8))
    owner = PersonAssociationLedger("run-a", spatial_policy())
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    return owner, rows


def test_current_visual_kernel_positive_is_explicit_estimate_without_common_metric_or_household_permit():
    owner, rows = visual_kernel()
    result = owner.join_current_visual(visual_proof(rows), at_interval=spatial_context(rows),
                                       delivery_context=visual_delivery(rows))
    assert result.disposition == "accepted" and result.reason == "current_visual"
    assert owner.stats().active_persons == 1
    assert all(m.association_basis == "current_visual" for m in owner._members.values())
    assert owner._links == {} and not hasattr(owner.policy, "target_frame")
    assert all(m.observation.household.state == "unknown" for m in owner._members.values())


@pytest.mark.parametrize("defect", ["common", "geometry", "profile", "copied", "retimestamp", "stale-ingress", "actor", "missing-competitor", "incomplete"])
def test_current_visual_kernel_rejects_invented_or_missing_current_evidence(defect):
    owner, rows = visual_kernel()
    candidate = visual_proof(rows)
    context, delivery = spatial_context(rows), visual_delivery(rows)
    if defect == "common":
        context = spatial_context(rows, qualified=True)
    elif defect == "stale-ingress":
        delivery = visual_delivery(rows, evaluation=100.351, ingress=100.)
    elif defect == "actor":
        assert owner.join_current_visual(candidate, at_interval=context, delivery_context=delivery).disposition == "accepted"
        delivery = replace(delivery, actor_id="different-actor", sequence=2)
    elif defect in {"missing-competitor", "incomplete"}:
        added = observation(raw=17)
        owner.observe_spatial(added, SpatialLifecycleReceipt("native-competitor", added.stamp, "ACTIVE"))
        rows += (added,)
        context, delivery = spatial_context(rows), visual_delivery(rows)
        if defect == "incomplete":
            full = visual_proof(rows)
            candidate = replace(full, candidates=full.candidates[:1])
    else:
        c = candidate.candidates[0]
        if defect == "geometry":
            c = replace(c, separation_m=0.)
        elif defect == "profile":
            c = replace(c, left_appearance=replace(c.left_appearance, model="not-selected-swin"))
        elif defect == "copied":
            c = replace(c, right_copy_id=c.left_copy_id)
        else:
            c = replace(c, left_appearance=replace(c.left_appearance, acquired_at_s=10.01))
        candidate = replace(candidate, candidates=(c,))
    before = owner.stats().active_persons
    result = owner.join_current_visual(candidate, at_interval=context, delivery_context=delivery)
    assert result.disposition == "association_unresolved", result
    assert owner.stats().active_persons == before


def test_current_visual_kernel_tie_missing_appearance_and_fresh_crossing_split_truthfully():
    owner, rows = visual_kernel()
    first = owner.join_current_visual(visual_proof(rows), at_interval=spatial_context(rows), delivery_context=visual_delivery(rows))
    key = first.person_key
    next_rows = (observation(frame=2, time=10.1, media=5.1), observation("family-room", 8, frame=2, time=10.1, media=5.1))
    for row in next_rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("next:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    full = visual_proof(next_rows, scores={(7, 8): .2})
    missing = replace(full, candidates=(replace(full.candidates[0], left_appearance=None),))
    result = owner.join_current_visual(missing, at_interval=spatial_context(next_rows),
                                       delivery_context=visual_delivery(next_rows, sequence=2, evaluation=100.1))
    assert result.disposition == "association_unresolved" and owner.stats().active_persons == 1
    result = owner.join_current_visual(full, at_interval=spatial_context(next_rows),
                                       delivery_context=visual_delivery(next_rows, sequence=3, evaluation=100.1))
    assert owner.stats().active_persons == 2 and len(result.corrected_members) == 1
    assert key in owner._persons and {m.observation for m in owner._members.values()} == set(next_rows)
    assert owner._closed == {} and owner._lifecycle_frontiers == {}
    assert owner.stats().allocated_serial == 3


def test_current_visual_kernel_original_cache_ingress_and_source_clock_age_never_refresh():
    owner, originals = visual_kernel()
    initial = visual_proof(originals)
    owner.join_current_visual(initial, at_interval=spatial_context(originals), delivery_context=visual_delivery(originals))
    rows = (observation(frame=10, time=10.3, media=5.3), observation("family-room", 8, frame=10, time=10.3, media=5.3))
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    cached = visual_proof(rows, originals=originals)
    delivery = visual_delivery(rows, sequence=2, evaluation=100.3, originals=originals, original_ingress=100.)
    assert owner.join_current_visual(cached, at_interval=spatial_context(rows), delivery_context=delivery).disposition == "accepted"
    assert all(a.stamp.frame_id == 1 and a.acquired_at_s == 10. for a in owner._appearance_origins.values())
    retimed = replace(delivery, sequence=3, ingresses=tuple(replace(i, appearance_ingress_monotonic_s=100.3) for i in delivery.ingresses))
    assert owner.join_current_visual(cached, at_interval=spatial_context(rows), delivery_context=retimed).reason == "current_visual_appearance_ingress_retimestamp"
    stale = replace(delivery, sequence=4, evaluation_monotonic_s=100.36)
    assert owner.join_current_visual(cached, at_interval=spatial_context(rows), delivery_context=stale).disposition == "association_unresolved"
    assert owner.stats().active_persons == 1  # Missing/expired proof is NOT contradiction.


def test_current_visual_kernel_ingress_and_proof_replay_frontiers():
    owner, rows = visual_kernel()
    original = visual_proof(rows)
    delivery = visual_delivery(rows)
    accepted = owner.join_current_visual(original, at_interval=spatial_context(rows), delivery_context=delivery)
    revision = owner.authority_revision
    assert owner.join_current_visual(original, at_interval=spatial_context(rows), delivery_context=delivery) == accepted
    assert owner.authority_revision == revision
    retimed = visual_delivery(rows, sequence=2, evaluation=100.1, ingress=100.1)
    assert owner.join_current_visual(original, at_interval=spatial_context(rows), delivery_context=retimed).reason == "current_visual_ingress_retimestamp"
    assert owner.join_current_visual(replace(original, proof_id="renamed"), at_interval=spatial_context(rows), delivery_context=delivery).reason == "current_visual_sequence_reused"


def test_current_visual_current_older_competitor_cannot_vanish_when_its_source_advances():
    older = observation(raw=17, time=9.9, media=4.9)
    current = observation(frame=2, time=10., media=5.)
    family = observation("family-room", 8, frame=2)
    owner, rows = visual_kernel((older, current, family))
    context = spatial_context(rows)
    omitted = visual_proof((current, family))
    result = owner.join_current_visual(omitted, at_interval=context, delivery_context=visual_delivery(rows))
    assert result.reason == "current_visual_complete_cohort_required"
    full = visual_proof(rows, scores={(17, 8): .9})
    result = owner.join_current_visual(full, at_interval=context,
                                       delivery_context=visual_delivery(rows, sequence=2))
    assert result.disposition == "observed" and owner.stats().active_persons == 3


def proof(left: SourceObservation, right: SourceObservation, token: str = "pair-1", *,
          similarity: float = 0.90, distance: float = 0.40) -> LinkProof:
    def measured(obs: SourceObservation, x: float) -> MeasurementEvidence:
        binding = SOURCES[obs.stamp.member.source_id]
        return MeasurementEvidence(obs.stamp, obs.stamp.member,
                                   "measurement-" + obs.stamp.observation_id,
                                   "room-metric", "revision-1", binding.calibration_revision,
                                   binding.transform_id, x, 0.0,
                                   independence_id="source-" + obs.stamp.observation_id)

    def appearance(obs: SourceObservation) -> AppearanceEvidence:
        selected = policy().appearance_profile
        return AppearanceEvidence(obs.stamp, "appearance-" + obs.stamp.observation_id,
                                  "selected-swin", obs.stamp.acquired_at_s, obs.stamp.clock_domain,
                                  appearance_extraction_identity(obs.stamp, selected), selected)

    return LinkProof(token, measured(left, 0.0), measured(right, distance),
                     appearance(left), appearance(right), similarity)


def setup_pair(*, owner: PersonAssociationLedger | None = None,
               left: SourceObservation | None = None,
               right: SourceObservation | None = None):
    owner = owner or PersonAssociationLedger("run-a", policy())
    left = left or observation()
    right = right or observation("family-room", 8)
    first = owner.observe(left)
    second = owner.observe(right)
    pair = proof(left, right)
    return owner, left, right, first, second, pair


def joined_pair():
    owner, left, right, first, _, pair = setup_pair()
    accepted = owner.link(pair, now())
    assert accepted.disposition == "accepted"
    assert accepted.person_key == first.person_key
    return owner, left, right, accepted


def removal(obs: SourceObservation, *, kind: str = "absence", token: str = "absence-1",
            frame: int = 2, time: float = 10.1, media: float | None = 5.1,
            retention: LifecycleRetention | None = None) -> MemberRemoval:
    return MemberRemoval(token, obs.stamp, kind, frame, time, obs.stamp.clock_domain,
                         media, retention)


def event(old: SourceObservation, new: SourceObservation, peer: SourceObservation | None,
          *, kind: str = "adoption", token: str = "event-1", time: float = 10.1) -> NativeTransition:
    return NativeTransition(token, kind, old.stamp, new.stamp,
                            peer.stamp if peer is not None else None, time, "capture-clock")


def test_fresh_overlap_opaque_key_immutable_outputs_and_order_invariance() -> None:
    decisions = []
    left, right = observation(), observation("family-room", 7)
    for inputs in permutations((left, right)):
        owner = PersonAssociationLedger("run-a", policy())
        for obs in inputs:
            owner.observe(obs)
        result = owner.link(proof(*inputs), now())
        assert result.disposition == "accepted"
        assert result.person_key == PhysicalPersonKey("run-a", 1)
        assert result.person_key.serial != left.stamp.member.raw_id
        assert len(result.members) == 2
        assert result.link.members == tuple(view.observation for view in result.members)
        assert all(view.household.state == "unknown" for view in result.members)
        with pytest.raises(FrozenInstanceError):
            result.person_key.serial = 999  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            result.members[0].observation.frame_id = 999  # type: ignore[misc]
        decisions.append(result)
    assert decisions[0] == decisions[1]


def test_duplicate_proof_pair_order_and_clock_do_not_renew_receipt() -> None:
    owner, left, right, _, _, pair = setup_pair()
    accepted = owner.link(pair, now())
    swapped = LinkProof(pair.proof_id, pair.right, pair.left, pair.right_appearance,
                        pair.left_appearance, pair.similarity)
    assert owner.link(swapped, now(10.2)) == accepted
    assert owner.link(pair, now(10.3)) == accepted
    assert owner.stats().active_persons == 1
    assert owner.snapshot(now(10.36))[0].link is None
    assert owner.snapshot(now(10.36))[0].state == "unresolved"
    assert owner.link(pair, now(10.36)).reason == "stale_evidence"
    assert accepted.link.admitted_at == now()


def test_new_observation_needs_new_exact_proof_not_sticky_membership() -> None:
    owner, left, right, accepted = joined_pair()
    updated = observation(frame=2, time=10.1, media=5.1)
    pending = owner.observe(updated)
    assert pending.person_key == accepted.person_key
    assert pending.disposition == "unresolved"
    assert pending.link is None
    assert owner.snapshot(now(10.1))[0].state == "unresolved"
    assert owner.link(proof(left, right, "old-stamp"), now(10.1)).reason == "noncurrent_observation"
    renewed = owner.link(proof(updated, right, "current-pair"), now(10.1))
    assert renewed.disposition == "accepted"
    assert renewed.person_key == accepted.person_key
    acquired = {s.member.camera_id: s.acquired_at_s for s in renewed.link.members}
    assert acquired == {"kitchen": 10.1, "family-room": 10.0}


@pytest.mark.parametrize("distance,similarity,reason", [
    (2.0, .95, "metric_separation"), (0.4, .59, "appearance_denied"),
    (0.4, float("nan"), "appearance_denied"), (0.4, float("inf"), "appearance_denied"),
])
def test_distinct_people_crossing_similar_clothes_and_partial_body_deny(
        distance: float, similarity: float, reason: str) -> None:
    owner, left, right, first, second, _ = setup_pair(right=observation("family-room", 7))
    result = owner.link(proof(left, right, distance=distance, similarity=similarity), now())
    assert result.reason == reason
    assert first.person_key != second.person_key
    assert len(owner.snapshot(now())) == 2


@pytest.mark.parametrize("left_house,right_house", [
    (HouseholdDescriptor("resident", "resident-a", 1), HouseholdDescriptor("resident", "resident-b", 2)),
    (HouseholdDescriptor("visitor", "visit-1", 11, 1), HouseholdDescriptor("visitor", "visit-2", 12, 2)),
    (HouseholdDescriptor("resident", "same-text", 1), HouseholdDescriptor("visitor", "same-text", 1, 1)),
    (HouseholdDescriptor("visitor", "visit", 11, 1), HouseholdDescriptor("visitor", "visit", 11, 2)),
])
def test_incompatible_household_descriptors_cannot_join_or_native_adopt(left_house, right_house) -> None:
    owner, left, right, _, _, pair = setup_pair(
        left=observation(household=left_house), right=observation("family-room", 7, household=right_house))
    assert owner.link(pair, now()).reason == "identity_conflict"
    native = event(left, left, right, time=10.02)
    assert owner.transition(native, left, now(), pair).reason == "identity_conflict"
    assert owner.stats().active_persons == 2


@pytest.mark.parametrize("state", ["unknown", "provisional", "resident", "visitor"])
def test_household_descriptor_never_promoted_or_aliased(state: str) -> None:
    descriptor = HouseholdDescriptor(state)
    if state == "resident":
        descriptor = HouseholdDescriptor(state, "enrolled-uuid", 5)
    elif state == "visitor":
        descriptor = HouseholdDescriptor(state, "run-session-slot", 9, 4)
    owner, left, right, _, _, pair = setup_pair(
        left=observation(household=descriptor), right=observation("family-room", 8, household=descriptor))
    result = owner.link(pair, now())
    assert result.disposition == "accepted"
    assert tuple(m.household for m in result.members) == (descriptor, descriptor)
    assert all(m.household is descriptor for m in result.members)
    assert not hasattr(result.person_key, "stable_id")
    assert not hasattr(result, "world_position")


@pytest.mark.parametrize("constraint", ["rejected", "unresolved"])
def test_household_owner_constraint_is_not_overridden(constraint: str) -> None:
    owner, left, right, _, _, pair = setup_pair(left=observation(household=HouseholdDescriptor(
        "unknown", constraint=constraint)))
    assert owner.link(pair, now()).reason == "household_constraint"


@pytest.mark.parametrize("field,value,reason", [
    ("origin", "predicted", "independent_measurement_required"),
    ("origin", "held", "independent_measurement_required"),
    ("origin", "peer-copy", "independent_measurement_required"),
    ("units", "cm", "independent_measurement_required"),
    ("target_frame", "other-room", "measurement_binding"),
    ("world_revision", "revision-2", "measurement_binding"),
    ("calibration_revision", "other-calibration", "measurement_binding"),
    ("transform_id", "other-transform", "measurement_binding"),
    ("x_m", float("nan"), "nonfinite_measurement"),
    ("y_m", float("inf"), "nonfinite_measurement"),
    ("independence_id", "", "independent_measurement_required"),
])
def test_measurement_provenance_negative(field: str, value: object, reason: str) -> None:
    owner, _, _, _, _, pair = setup_pair()
    assert owner.link(replace(pair, left=replace(pair.left, **{field: value})), now()).reason == reason
    assert owner.stats().active_persons == 2


@pytest.mark.parametrize("camera", ["kitchen", "family-room"])
def test_living_never_receives_overlap_admission(camera: str) -> None:
    owner, left, right, _, _, pair = setup_pair(left=observation("living-room"), right=observation(camera, 8))
    assert owner.link(pair, now()).reason == "overlap_denied"


def test_same_camera_and_competing_member_denials() -> None:
    owner, left, right, accepted = joined_pair()
    competitor = observation(raw=10)
    owner.observe(competitor)
    assert owner.link(proof(competitor, left, "same-camera"), now()).reason == "same_camera_collision"
    result = owner.link(proof(competitor, right, "competitor"), now())
    assert result.reason == "same_camera_collision"
    assert owner.stats().active_persons == 2
    assert owner.snapshot(now())[0].person_key == accepted.person_key


def test_empty_enabled_overlap_and_correlated_measurements_deny() -> None:
    owner, _, _, _, _, pair = setup_pair(owner=PersonAssociationLedger("run-a", policy(overlap_pairs=())))
    assert owner.link(pair, now()).reason == "overlap_denied"
    owner, _, _, _, _, pair = setup_pair()
    correlated = replace(pair, right=replace(pair.right, independence_id=pair.left.independence_id))
    assert owner.link(correlated, now()).reason == "correlated_measurements"
    copied = replace(pair, right=replace(pair.right, owner=pair.left.owner))
    assert owner.link(copied, now()).reason == "independent_measurement_required"


@pytest.mark.parametrize("field,value,reason", [
    ("model", "unselected-model", "appearance_binding"),
    ("acquired_at_s", 9.0, "appearance_observation"),
    ("acquired_at_s", 10.2, "appearance_clock"),
    ("clock_domain", "estimated-host-clock", "appearance_clock"),
    ("evidence_id", "", "appearance_binding"),
])
def test_missing_stale_wrong_model_and_host_only_appearance(field, value, reason) -> None:
    owner, _, _, _, _, pair = setup_pair()
    result = owner.link(replace(pair, left_appearance=replace(pair.left_appearance, **{field: value})), now())
    assert result.reason == reason


def test_missing_appearance_future_time_and_unavailable_clock_domain_deny() -> None:
    owner, _, _, _, _, pair = setup_pair()
    assert owner.link(replace(pair, left_appearance=None), now()).reason == "appearance_missing"
    assert owner.link(pair, now(9.9)).reason == "future_evidence"
    assert owner.link(pair, AcquisitionTime(10.05, "estimated-host-clock", "host-estimated")).reason == "clock_domain"
    assert owner.link(pair, AcquisitionTime(float("nan"), "capture-clock", "synchronized-media")).reason == "clock_domain"


@pytest.mark.parametrize("field,value,reason", [
    ("source_id", 0, "source_binding"), ("camera_id", "living-room", "source_binding"),
    ("source_epoch", 1, "source_binding"), ("run_id", "old-run", "source_binding"),
    ("lifecycle_generation", 0, "malformed_observation"), ("raw_id", -1, "malformed_observation"),
    ("raw_id", True, "malformed_observation"),
    ("raw_id", 2**64 - 1, "malformed_observation"),
])
def test_exact_member_binding_rejects_malformed_and_unbound_inputs(field, value, reason) -> None:
    owner = PersonAssociationLedger("run-a", policy())
    obs = observation()
    changed = replace(obs, stamp=replace(obs.stamp, member=replace(obs.stamp.member, **{field: value})))
    assert owner.observe(changed).reason == reason
    assert owner.stats().active_members == 0


def test_out_of_order_and_contradictory_observation_leave_exact_current_state() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    newer = observation(frame=3, time=10.2, media=5.2)
    first = owner.observe(newer)
    assert owner.observe(observation(frame=2, time=10.1, media=5.1)).reason == "reordered_observation"
    assert owner.observe(replace(newer, household=HouseholdDescriptor("provisional"))).reason == "reordered_observation"
    assert owner.observe(observation(raw=8, frame=3, time=10.21, media=5.2)).reason == "contradictory_cohort"
    assert owner.observe(observation(frame=4, time=10.3, media=5.0)).reason == "reordered_media_time"
    assert owner.snapshot(now(10.2))[0].members == first.members


def test_late_adoption_changes_raw_member_and_preserves_older_physical_key() -> None:
    owner, old, peer, first, _, _ = setup_pair()
    new = observation(raw=8, frame=2, time=10.1, media=5.1)
    adoption = event(old, new, peer)
    joined = owner.transition(adoption, new, now(10.1), proof(new, peer))
    assert joined.disposition == "accepted"
    assert joined.person_key == first.person_key
    assert len(owner.snapshot(now(10.1))) == 1
    assert {m.observation.member for m in joined.members} == {new.stamp.member, peer.stamp.member}
    assert old.stamp.member not in {m.observation.member for m in joined.members}
    assert owner.observe(observation(frame=3, time=10.2, media=5.2)).reason == "closed_lifecycle"
    before = owner.stats()
    assert owner.transition(adoption, new, now(10.2), proof(new, peer)) == joined
    assert owner.stats() == before


def test_handoff_releases_one_source_keeps_key_and_requires_fresh_return_proof() -> None:
    owner, left, right, accepted = joined_pair()
    result = owner.withdraw(removal(left))
    assert result.person_key == accepted.person_key
    assert result.link is None
    assert len(result.members) == 1
    assert result.members[0].observation == right.stamp
    new_left = observation(raw=9, frame=3, time=10.2, media=5.2, generation=2, started=3)
    local = owner.observe(new_left)
    assert local.person_key != accepted.person_key
    joined = owner.link(proof(new_left, right, "return"), now(10.2))
    assert joined.person_key == accepted.person_key


def test_correction_splits_offender_only_and_old_adoption_does_not_resurrect() -> None:
    owner, old, peer, first, _, _ = setup_pair()
    adopted = observation(raw=8, frame=2, time=10.1, media=5.1)
    adoption = event(old, adopted, peer)
    old_proof = proof(adopted, peer)
    accepted = owner.transition(adoption, adopted, now(10.1), old_proof)
    corrected = observation(raw=19, frame=3, time=10.2, media=5.2)
    correction = event(adopted, corrected, peer, kind="correction", token="correction", time=10.2)
    split = owner.transition(correction, corrected, now(10.2))
    assert split.reason == "correction_detached"
    assert split.person_key != accepted.person_key
    views = owner.snapshot(now(10.2))
    assert len(views) == 2
    assert views[0].person_key == first.person_key
    assert views[0].members[0].observation == peer.stamp
    assert all(v.link is None for v in views)
    assert owner.transition(adoption, adopted, now(10.2), old_proof).disposition == "unresolved"
    assert owner.stats().active_persons == 2
    assert owner.link(old_proof, now(10.2)).reason == "noncurrent_observation"
    assert owner.transition(correction, corrected, now(10.25)) == split


def test_correction_can_detach_without_raw_id_change() -> None:
    owner, left, right, accepted = joined_pair()
    split = owner.transition(event(left, left, right, kind="correction", time=10.05), left, now())
    assert split.reason == "correction_detached"
    assert split.person_key != accepted.person_key
    assert owner.snapshot(now())[0].members[0].observation == right.stamp


@pytest.mark.parametrize("change,reason", [
    ({"occurred_at_s": 9.0}, "stale_event"),
    ({"occurred_at_s": 10.5}, "future_event"),
    ({"clock_domain": "estimated-host-time"}, "clock_domain"),
    ({"kind": "raw-id-flip"}, "malformed_event"),
])
def test_native_candidate_event_cannot_bypass_exact_lineage_or_time(change, reason) -> None:
    owner, left, right, _, _, pair = setup_pair()
    native = replace(event(left, left, right, time=10.02), **change)
    assert owner.transition(native, left, now(), pair).reason == reason
    assert owner.stats().active_persons == 2


def test_unknown_lineage_epoch_change_and_missing_proof_are_unresolved() -> None:
    owner, left, right, _, _, _ = setup_pair()
    new = observation(raw=8, frame=2, time=10.1, media=5.1)
    assert owner.transition(event(left, new, right), new, now(10.1)).reason == "fresh_pair_proof_required"
    assert owner.transition(event(observation(raw=99), new, right), new,
                            now(10.1), proof(new, right)).reason == "noncurrent_event_lineage"
    changed = observation(raw=8, frame=2, time=10.1, media=5.1, generation=2)
    assert owner.transition(event(left, changed, right), changed,
                            now(10.1), proof(changed, right)).reason == "event_lineage"
    assert owner.stats().active_persons == 2


@pytest.mark.parametrize("kind", ["absence", "termination"])
def test_removal_wins_against_late_adoption_and_proof(kind) -> None:
    owner, left, right, _, _, pair = setup_pair()
    owner.withdraw(removal(left, kind=kind))
    assert owner.link(pair, now(10.1)).reason == "noncurrent_observation"
    candidate = event(left, left, right, time=10.05)
    assert owner.transition(candidate, left, now(10.1), pair).reason == "noncurrent_event_lineage"
    assert len(owner.snapshot(now(10.1))) == 1


def test_source_empty_lost_peer_and_all_source_absence_never_create_presence() -> None:
    owner, left, right, accepted = joined_pair()
    empty = SourceAbsence("empty-k", "kitchen", 1, 0, 2, 10.1, "capture-clock")
    assert owner.clear_source(empty).disposition == "removed"
    view = owner.snapshot(now(10.1))[0]
    assert view.person_key == accepted.person_key
    assert view.members[0].observation == right.stamp
    assert view.link is None
    assert owner.clear_source(replace(empty, receipt_id="empty-f", camera_id="family-room", source_id=2)).disposition == "removed"
    assert owner.snapshot(now(10.1)) == ()
    assert owner.stats().active_members == 0
    assert owner.observe(observation(frame=3, time=10.2, media=5.2)).reason == "closed_lifecycle"


def test_source_epoch_resets_old_members_proofs_receipts_and_run_restart_clears_all() -> None:
    owner, left, right, accepted = joined_pair()
    binding = replace(SOURCES[1], source_epoch=1, calibration_revision="new-cal", transform_id="new-transform")
    assert owner.reset_source(binding).disposition == "reset"
    assert owner.stats().receipts == owner.stats().retired_keys == 0
    assert owner.snapshot(now())[0].members[0].observation == right.stamp
    assert owner.observe(left).reason == "source_binding"
    new = observation(epoch=1, frame=1)
    local = owner.observe(new)
    assert local.person_key != accepted.person_key
    new_proof = proof(new, right, "new-era")
    new_proof = replace(new_proof, left=replace(new_proof.left, calibration_revision="new-cal", transform_id="new-transform"))
    assert owner.link(new_proof, now()).disposition == "accepted"
    assert owner.reset_run("run-b").disposition == "reset"
    assert owner.stats().active_persons == owner.stats().active_members == owner.stats().closed_members == 0
    assert owner.stats().receipts == owner.stats().retired_keys == 0
    assert owner.observe(new).reason == "source_binding"
    restarted = owner.observe(observation(epoch=1, run="run-b"))
    assert restarted.person_key == PhysicalPersonKey("run-b", 1)


def test_same_numeric_id_new_generation_has_no_old_household_lock() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    resident = observation(household=HouseholdDescriptor("resident", "resident-a", 1))
    old = owner.observe(resident)
    assert owner.observe(observation(generation=2)).reason == "lifecycle_reset_required"
    owner.withdraw(removal(resident, kind="termination"))
    new = observation(generation=2, frame=3, time=10.2, media=5.2, started=3)
    result = owner.observe(new)
    assert result.person_key != old.person_key
    assert result.members[0].household == HouseholdDescriptor()


def test_valid_short_return_preserves_key_and_original_clocks_not_presence() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    obs = observation()
    first = owner.observe(obs)
    receipt = LifecycleRetention("owner-inactive", obs.stamp, obs.stamp.member, "inactive", .35)
    removed = owner.withdraw(removal(obs, retention=receipt))
    assert removed.members == ()
    assert owner.snapshot(now(10.1)) == ()
    assert owner.stats().retained_members == 1
    returned = observation(frame=3, time=10.2, media=5.2)
    result = owner.observe(returned)
    assert result.person_key == first.person_key
    assert removed.consumed_tokens == ("removal:absence-1",)
    assert receipt.last_observation == obs.stamp
    assert obs.stamp.media_time_s == 5.0


@pytest.mark.parametrize("media,grace,lifecycle,owner_matches", [
    (None, .35, "inactive", True), (4.9, .35, "inactive", True),
    (5.5, .35, "inactive", True), (5.1, 2.0, "inactive", True),
    (5.1, .35, "tentative", True), (5.1, .35, "inactive", False),
])
def test_invalid_short_return_has_no_host_time_fallback(media, grace, lifecycle, owner_matches) -> None:
    owner = PersonAssociationLedger("run-a", policy())
    obs = observation()
    owner.observe(obs)
    owner_key = obs.stamp.member if owner_matches else replace(obs.stamp.member, raw_id=88)
    receipt = LifecycleRetention("retention", obs.stamp, owner_key, lifecycle, grace)
    owner.withdraw(removal(obs, media=media, retention=receipt))
    assert owner.stats().retained_members == 0
    result = owner.observe(observation(frame=3, time=10.2, media=max(5.2, media or 0)))
    assert result.reason == "closed_lifecycle"


def test_missing_original_media_expiry_and_termination_cancel_grace() -> None:
    for terminate in (False, True):
        owner = PersonAssociationLedger("run-a", policy())
        obs = observation()
        owner.observe(obs)
        retention = LifecycleRetention("retention", obs.stamp, obs.stamp.member, "inactive", .35)
        owner.withdraw(removal(obs, retention=retention))
        if terminate:
            owner.withdraw(removal(obs, kind="termination", token="termination", frame=3, time=10.2, media=5.2))
        else:
            owner.expire_retention(1, 5.36)
        assert owner.stats().retained_members == 0
        assert owner.observe(observation(frame=4, time=10.3, media=5.3)).reason == "closed_lifecycle"
    owner = PersonAssociationLedger("run-a", policy())
    missing = observation(media=None)
    owner.observe(missing)
    retention = LifecycleRetention("retention", missing.stamp, missing.stamp.member, "inactive", .35)
    owner.withdraw(removal(missing, retention=retention))
    assert owner.stats().retained_members == 0


def test_receipt_token_conflict_cannot_rewrite_current_state() -> None:
    owner, _, _, _, _, pair = setup_pair()
    accepted = owner.link(pair, now())
    result = owner.link(replace(pair, similarity=.91), now())
    assert result.reason == "token_conflict"
    assert owner.snapshot(now())[0].link == accepted.link


def test_capacity_does_not_merge_people_and_every_registry_remains_bounded() -> None:
    owner = PersonAssociationLedger("run-a", policy(max_entities=2, max_history=3))
    owner.observe(observation())
    owner.observe(observation("family-room", 8))
    assert owner.observe(observation("living-room", 9)).reason == "entity_capacity"
    assert owner.stats().active_persons == 2
    owner, left, right, _, _, pair = setup_pair(owner=owner)
    assert owner.link(pair, now()).disposition == "accepted"
    # Obsolete detail compacts safely even under a deliberately small budget.
    for frame in range(2, 7):
        left = observation(frame=frame, time=10 + frame / 100, media=5 + frame / 100)
        right = observation("family-room", 8, frame=frame, time=10 + frame / 100, media=5 + frame / 100)
        owner.observe(left)
        owner.observe(right)
        result = owner.link(proof(left, right, f"pair-{frame}"), now(10 + frame / 100))
    assert result.disposition == "accepted"
    stats = owner.stats()
    assert stats.active_persons <= 2 and stats.active_members <= 4
    assert stats.receipts <= 4 and stats.retired_keys <= 4 and stats.closed_members <= 4
    assert stats.receipts + stats.retired_keys <= 3
    # Normal disappearance stays usable; no reset is necessary as history cleanup.
    result = owner.withdraw(removal(left, frame=7, time=10.1, media=5.1))
    assert result.disposition == "removed"
    assert owner.stats().active_members == 1
    assert owner.stats().blocked_sources == 0
    assert owner.observe(observation(raw=19, frame=8, time=10.2, media=5.2, started=8)).disposition == "observed"
    assert owner.reset_source(replace(SOURCES[1], source_epoch=1)).disposition == "reset"
    assert owner.stats().blocked_sources == 0


@pytest.mark.parametrize("changes", [
    {"max_entities": 65}, {"max_history": 129}, {"max_age_s": .36},
    {"max_distance_m": 1.26}, {"min_similarity": .59},
    {"sources": list(SOURCES)}, {"sources": (SOURCES[0], SOURCES[0])},
    {"overlap_pairs": (("living-room", "family-room"),)},
    {"appearance_model": ""}, {"target_frame": ""},
])
def test_invalid_configuration_cannot_weaken_boundaries(changes) -> None:
    with pytest.raises(ValueError):
        policy(**changes)


def test_ledger_retains_only_compact_source_household_and_receipt_scalars() -> None:
    owner, _, _, accepted = joined_pair()
    assert set(owner._members[next(iter(owner._members))].__slots__) == {"observation", "person_key", "appearance_after_frame"}
    assert set(accepted.members[0].__slots__) == {"observation", "household"}
    assert not hasattr(accepted, "embedding")
    assert not hasattr(accepted, "measurement")
    assert not hasattr(accepted, "similarity")


def test_same_source_observations_cannot_be_renamed_into_new_confirmation() -> None:
    owner, left, right, accepted = joined_pair()
    renamed = proof(left, right, "renamed-proof")
    renamed = replace(renamed, left=replace(renamed.left, evidence_id="renamed-measurement"),
                      left_appearance=replace(renamed.left_appearance, evidence_id="renamed-appearance"))
    before = owner.stats()
    assert owner.link(renamed, now(10.2)).reason == "appearance_origin_conflict"
    assert owner.stats() == before
    assert owner.snapshot(now(10.2))[0].link == accepted.link


def test_direct_owner_receipt_consumer_checks_currentness_without_clock_renewal() -> None:
    owner, left, _, accepted = joined_pair()
    before = owner.stats()
    assert owner.validate_receipt(accepted.link, now(10.2)).disposition == "accepted"
    assert owner.validate_receipt(accepted.link, now(10.4)).reason == "stale_evidence"
    assert owner.stats() == before
    assert accepted.link.admitted_at == now()
    owner.observe(observation(frame=2, time=10.1, media=5.1))
    assert owner.validate_receipt(accepted.link, now(10.1)).reason == "receipt_invalidated"
    # Forging a changed target/key does not match the owner-generated receipt.
    changed = replace(accepted.link, person_key=PhysicalPersonKey("other-run", 1))
    assert owner.validate_receipt(changed, now()).reason == "receipt_invalidated"


def test_already_observed_replacement_keeps_oldest_allocation_and_retires_duplicate() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    replacement = observation(raw=8)
    first = owner.observe(replacement)
    old = observation()
    second = owner.observe(old)
    peer = observation("family-room", 8)
    third = owner.observe(peer)
    result = owner.transition(event(old, replacement, peer, time=10.02), replacement,
                              now(), proof(replacement, peer))
    assert result.disposition == "accepted"
    assert result.person_key == first.person_key
    assert set(result.retired_keys) == {second.person_key, third.person_key}
    assert owner.stats().active_persons == 1
    assert owner.stats().active_members == 2
    assert owner.stats().retired_keys == 2


def test_correction_small_budget_compacts_detail_and_withdraws_offender() -> None:
    owner, left, right, _, _, pair = setup_pair(owner=PersonAssociationLedger(
        "run-a", policy(max_entities=2, max_history=3)))
    accepted = owner.link(pair, now())
    assert owner.stats().receipts == 2
    result = owner.transition(event(left, left, right, kind="correction", time=10.05), left, now())
    assert result.reason == "correction_detached"
    views = owner.snapshot(now())
    assert len(views) == 2
    assert views[0].person_key == accepted.person_key
    assert views[0].members[0].observation == right.stamp
    assert views[0].link is None
    assert owner.stats().blocked_sources == 0


def test_termination_after_correction_prevents_old_member_or_receipt_return() -> None:
    owner, left, right, accepted = joined_pair()
    corrected = owner.transition(event(left, left, right, kind="correction", time=10.05), left, now())
    owner.withdraw(removal(left, kind="termination"))
    assert owner.observe(observation(frame=3, time=10.2, media=5.2)).reason == "closed_lifecycle"
    assert owner.validate_receipt(accepted.link, now(10.1)).reason == "receipt_invalidated"
    assert owner.snapshot(now(10.1))[0].person_key == accepted.person_key
    assert corrected.person_key != accepted.person_key


def test_unbound_clock_domains_and_expired_local_source_are_unresolved() -> None:
    bindings = tuple(replace(b, clock_domain="other-capture-clock") if b.source_id == 2 else b for b in SOURCES)
    owner = PersonAssociationLedger("run-a", policy(sources=bindings))
    left = observation()
    right = observation("family-room", 8)
    right = replace(right, stamp=replace(right.stamp, clock_domain="other-capture-clock"))
    owner.observe(left)
    owner.observe(right)
    assert owner.link(proof(left, right), now()).reason == "clock_domain"
    assert owner.snapshot(now(11.0))[0].state == "unresolved"


def test_low_generation_replay_and_stale_removal_cannot_rewind_source_state() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    current = observation(generation=3)
    owner.observe(current)
    owner.withdraw(removal(current, kind="termination"))
    assert owner.observe(observation(generation=2, frame=3, time=10.2, media=5.2)).reason == "retired_lifecycle_generation"
    newer = observation(raw=99, generation=4, frame=5, time=10.4, media=5.4, started=5)
    owner.observe(newer)
    assert owner.clear_source(SourceAbsence("old-empty", "kitchen", 1, 0, 4, 10.3,
                                           "capture-clock")).reason == "reordered_source_absence"
    assert owner.withdraw(removal(newer, frame=4, time=10.3, media=5.3)).reason == "removal_lineage"
    assert owner.stats().active_members == 1


def test_stale_old_native_lineage_cannot_extend_unowned_presence() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    old = observation()
    owner.observe(old)
    peer = observation("family-room", 8, frame=11, time=11.0, media=6.0)
    owner.observe(peer)
    replacement = observation(raw=8, frame=11, time=11.0, media=6.0)
    result = owner.transition(event(old, replacement, peer, time=11.0), replacement,
                              now(11.0), proof(replacement, peer))
    assert result.reason == "stale_event_lineage"
    assert owner.stats().active_persons == 2


@pytest.mark.parametrize("bad", [
    SourceObservation(replace(observation().stamp, clock_domain=[])),
    SourceObservation(replace(observation().stamp, acquired_at_s=float("nan"))),
    SourceObservation(observation().stamp, HouseholdDescriptor(state=[])),
    SourceObservation(observation().stamp, HouseholdDescriptor("provisional", "invented-sid", 5)),
    None,
])
def test_malformed_inputs_return_typed_denial_without_allocations(bad) -> None:
    owner = PersonAssociationLedger("run-a", policy())
    assert owner.observe(bad).disposition == "rejected"
    assert owner.stats().active_members == 0


def test_matching_clock_name_cannot_admit_estimated_host_time() -> None:
    owner, left, _, _, _, pair = setup_pair()
    host = replace(left, stamp=replace(left.stamp, clock_basis="host-estimated"))
    assert owner.observe(host).reason == "clock_domain"
    assert owner.link(pair, AcquisitionTime(10.05, "capture-clock", "host-estimated")).reason == "clock_domain"
    with pytest.raises(ValueError):
        policy(sources=tuple(replace(b, clock_basis="host-estimated") for b in SOURCES))


def test_cached_appearance_cannot_be_retagged_as_current_source_observation() -> None:
    owner, _, _, _, _, pair = setup_pair()
    cached = replace(pair, left_appearance=replace(pair.left_appearance, acquired_at_s=9.99))
    assert owner.link(cached, now()).reason == "appearance_observation"


def test_nonfinite_huge_integer_geometry_returns_typed_denial() -> None:
    owner, _, _, _, _, pair = setup_pair()
    assert owner.link(replace(pair, left=replace(pair.left, x_m=2**10000)), now()).reason == "nonfinite_measurement"
    overflow = replace(pair, left=replace(pair.left, x_m=2**1023),
                       right=replace(pair.right, x_m=-(2**1023)))
    assert owner.link(overflow, now()).reason == "nonfinite_measurement"


def test_10000_30hz_renewals_preserve_key_and_all_registry_bounds() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    first_key = None
    previous = None
    first_proof = None
    for frame in range(1, 10001):
        time = 10 + frame / 30
        media = 5 + frame / 30
        left = observation(frame=frame, time=time, media=media)
        right = observation("family-room", 8, frame=frame, time=time, media=media)
        assert owner.observe(left).disposition in {"observed", "unresolved"}
        assert owner.observe(right).disposition in {"observed", "unresolved"}
        if previous is not None:
            assert owner.validate_receipt(previous, now(time)).disposition != "accepted"
        pair = proof(left, right, f"endurance-{frame}")
        result = owner.link(pair, now(time))
        assert result.disposition == "accepted", (frame, result.reason, owner.stats())
        first_key = first_key or result.person_key
        assert result.person_key == first_key
        assert owner.validate_receipt(result.link, now(time)).disposition == "accepted"
        stats = owner.stats()
        assert stats.active_persons == 1 and stats.active_members == 2
        assert stats.receipts + stats.retired_keys <= owner.policy.max_history
        assert stats.closed_members <= owner.policy.max_history
        assert len(owner._bindings) <= 3 and len(owner._cursors) <= 3
        assert len(owner._closed) <= owner.policy.max_history
        assert len(owner._persons) <= owner.policy.max_entities
        assert len(owner._members) <= 2 * owner.policy.max_entities
        assert len(owner._links) <= owner.policy.max_entities and len(owner._blocked) <= 3
        assert len(owner._lifecycle_frontiers) <= 3
        assert stats.retired_serial_frontier <= stats.allocated_serial
        if frame == 1:
            first_proof = pair
        previous = result.link
    assert owner.link(first_proof, now(time)).disposition != "accepted"
    assert owner.link(replace(first_proof, proof_id="renamed-old"), now(time)).disposition != "accepted"
    print(f"kernel_endurance: ticks=10000 key={first_key} final={owner.stats()}")


def test_300_adoption_correction_retirement_removal_cycles_and_fresh_ttl_replays() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    anchor_key = None
    previous_removal = None
    for cycle in range(300):
        frame = 4 * cycle + 1
        time = 10 + frame / 30
        left = observation(frame=frame, time=time, media=frame / 30)
        right = observation("family-room", 1000 + cycle * 3, frame=frame,
                            time=time, media=frame / 30, generation=cycle + 1, started=frame)
        owner.observe(left)
        owner.observe(right)
        joined = owner.link(proof(left, right, f"cycle-{cycle}"), now(time))
        assert joined.disposition == "accepted"
        anchor_key = anchor_key or joined.person_key
        assert joined.person_key == anchor_key
        if previous_removal is not None:
            assert owner.withdraw(previous_removal).disposition != "removed"
        adopted = replace(right, stamp=replace(right.stamp,
                          member=replace(right.stamp.member, raw_id=right.stamp.member.raw_id + 1),
                          frame_id=frame + 1, observation_id=f"adopt-{cycle}",
                          acquired_at_s=time + .01, media_time_s=frame / 30 + .01))
        adoption = event(right, adopted, left, token=f"adopt-{cycle}", time=time + .01)
        adopted_proof = proof(adopted, left, f"adopt-proof-{cycle}")
        result = owner.transition(adoption, adopted, now(time + .01), adopted_proof)
        assert result.disposition == "accepted" and result.person_key == anchor_key
        correction = event(adopted, adopted, left, kind="correction", token=f"correct-{cycle}", time=time + .02)
        split = owner.transition(correction, adopted, now(time + .02))
        assert split.reason == "correction_detached" and split.person_key != anchor_key
        assert owner.transition(correction, adopted, now(time + .02)) == split
        fresh_left = observation(frame=frame + 2, time=time + .03, media=frame / 30 + .03)
        fresh_right = replace(adopted, stamp=replace(adopted.stamp, frame_id=frame + 2,
                              observation_id=f"fresh-{cycle}", acquired_at_s=time + .03,
                              media_time_s=frame / 30 + .03))
        owner.observe(fresh_left)
        owner.observe(fresh_right)
        owner.compact_history()
        assert owner.transition(adoption, adopted, now(time + .03), adopted_proof).disposition != "accepted"
        assert owner.transition(correction, adopted, now(time + .03)).disposition == "unresolved"
        assert owner.link(adopted_proof, now(time + .03)).disposition != "accepted"
        assert owner.link(replace(adopted_proof, proof_id=f"renamed-{cycle}"), now(time + .03)).disposition != "accepted"
        again = owner.link(proof(fresh_left, fresh_right, f"again-{cycle}"), now(time + .03))
        assert again.person_key == anchor_key
        ended = removal(fresh_right, kind="termination", token=f"end-{cycle}", frame=frame + 3,
                        time=time + .04, media=frame / 30 + .04)
        assert owner.withdraw(ended).disposition == "removed"
        previous_removal = ended
        stats = owner.stats()
        assert stats.active_persons == stats.active_members == 1
        assert stats.receipts + stats.retired_keys <= owner.policy.max_history
        assert stats.closed_members <= owner.policy.max_history
        assert stats.lifecycle_frontiers <= 3 and len(owner._cursors) <= 3
    stats = owner.stats()
    assert stats.compacted_receipts > 128 and stats.compacted_closed > 128
    assert stats.compacted_retired > 128 and stats.allocated_serial > 128
    assert anchor_key.serial == 1
    print(f"kernel_churn: cycles=300 key={anchor_key} final={stats}")


def test_150_media_grace_removals_expire_without_reset_or_capacity_failure() -> None:
    owner = PersonAssociationLedger("run-a", policy(max_entities=1, max_history=2))
    for cycle in range(150):
        frame = cycle * 3 + 1
        obs = observation(raw=100 + cycle, generation=cycle + 1, started=frame,
                          frame=frame, time=10 + cycle, media=5 + cycle)
        result = owner.observe(obs)
        assert result.disposition == "observed", (cycle, result)
        retention = LifecycleRetention(f"grace-{cycle}", obs.stamp, obs.stamp.member, "inactive", .35)
        ended = removal(obs, token=f"absent-{cycle}", frame=frame + 1,
                        time=10 + cycle + .1, media=5 + cycle + .1, retention=retention)
        assert owner.withdraw(ended).disposition == "removed"
        assert owner.snapshot(now(10 + cycle + .1)) == ()
        assert owner.stats().retained_members == 1
        assert owner.stats().closed_members <= 2 and owner.stats().receipts <= 2
    assert owner.stats().allocated_serial == 150


def test_compaction_protects_current_grace_and_immutable_birth_provenance() -> None:
    owner = PersonAssociationLedger("run-a", policy())
    original = observation()
    first = owner.observe(original)
    grace = LifecycleRetention("grace", original.stamp, original.stamp.member, "inactive", .35)
    owner.withdraw(removal(original, retention=grace))
    newcomer = observation(raw=99, frame=3, generation=2, started=3, time=10.15, media=5.15)
    owner.observe(newcomer)
    owner.compact_history()
    assert owner.stats().retained_members == 1
    returned = observation(frame=4, time=10.2, media=5.2)
    assert owner.observe(returned).person_key == first.person_key
    retagged = observation(frame=5, time=10.25, media=5.25, started=5)
    assert owner.observe(retagged).reason == "lifecycle_binding"


def cached_proof(left, right, original_left, original_right, token="cached"):
    current = proof(left, right, token)
    original = proof(original_left, original_right, "original")
    return replace(current, left_appearance=original.left_appearance,
                   right_appearance=original.right_appearance)


@pytest.mark.parametrize("age", [0.0, .333, .35])
def test_honest_original_appearance_older_frame_accepts_without_timestamp_renewal(age) -> None:
    original_left, original_right = observation(), observation("family-room", 8)
    left = observation(frame=12, time=10 + age, media=5 + age)
    right = observation("family-room", 8, frame=12, time=10 + age, media=5 + age)
    owner, _, _, _, _, _ = setup_pair(left=left, right=right)
    pair = cached_proof(left, right, original_left, original_right)
    result = owner.link(pair, now(10 + age))
    assert result.disposition == "accepted"
    assert result.link.expires_at_s == 10.35
    assert all(a.stamp.frame_id == 1 and a.acquired_at_s == 10 for a in result.link.appearances)
    assert owner.validate_receipt(result.link, now(10.351)).disposition != "accepted"


@pytest.mark.parametrize("field,value", [("raw_id", 99), ("source_epoch", 1),
                                         ("lifecycle_generation", 2), ("run_id", "other-run")])
def test_original_appearance_cannot_alias_member_epoch_generation_or_run(field, value) -> None:
    owner, left, right, _, _, pair = setup_pair()
    appearance = replace(pair.left_appearance, stamp=replace(pair.left_appearance.stamp,
                         member=replace(pair.left_appearance.stamp.member, **{field: value})))
    assert owner.link(replace(pair, left_appearance=appearance), now()).disposition == "rejected"


@pytest.mark.parametrize("field,value", [("layer_id", "wrong-layer"), ("dimensions", 1024),
                                         ("fingerprint", "wrong-fingerprint")])
def test_original_appearance_semantic_profile_is_exact(field, value) -> None:
    owner, _, _, _, _, pair = setup_pair()
    altered = replace(pair.left_appearance, profile=replace(pair.left_appearance.profile, **{field: value}))
    assert owner.link(replace(pair, left_appearance=altered), now()).reason == "appearance_binding"


@pytest.mark.parametrize("rename", [False, True])
def test_cached_original_extraction_id_cannot_be_retimed_or_renamed(rename) -> None:
    owner, old_left, old_right, accepted = joined_pair()
    fresh_left = observation(frame=2, time=10.1, media=5.1)
    fresh_right = observation("family-room", 8, frame=2, time=10.1, media=5.1)
    owner.observe(fresh_left)
    owner.observe(fresh_right)
    original = proof(old_left, old_right).left_appearance
    retagged = replace(original, stamp=fresh_left.stamp, acquired_at_s=10.1,
                       evidence_id="renamed" if rename else original.evidence_id)
    pair = cached_proof(fresh_left, fresh_right, old_left, old_right, "retag")
    assert owner.link(replace(pair, left_appearance=retagged), now(10.1)).reason == "appearance_origin_conflict"


def test_actual_absence_and_same_key_correction_clear_original_appearance_lineage() -> None:
    owner, left, right, _ = joined_pair()
    grace = LifecycleRetention("grace", left.stamp, left.stamp.member, "inactive", .35)
    owner.withdraw(removal(left, retention=grace))
    returned = observation(frame=3, time=10.2, media=5.2)
    owner.observe(returned)
    assert owner.link(cached_proof(returned, right, left, right), now(10.2)).reason == "appearance_lineage"
    owner, left, right, _ = joined_pair()
    owner.transition(event(left, left, right, kind="correction", time=10.05), left, now())
    assert owner.link(proof(left, right, "after-correction"), now()).reason == "appearance_lineage"


def test_original_origin_identity_still_denies_retimestamp_after_many_compactions() -> None:
    owner, old_left, old_right, original = joined_pair()
    old_appearance = proof(old_left, old_right).left_appearance
    for frame in range(2, 150):
        time = 10 + frame / 30
        left = observation(frame=frame, time=time, media=5 + frame / 30)
        right = observation("family-room", 8, frame=frame, time=time, media=5 + frame / 30)
        owner.observe(left)
        owner.observe(right)
        assert owner.link(proof(left, right, f"fresh-{frame}"), now(time)).disposition == "accepted"
    pair = proof(left, right, "renamed-cache")
    forged = replace(old_appearance, stamp=left.stamp, acquired_at_s=left.stamp.acquired_at_s,
                     evidence_id="renamed-after-compaction")
    assert owner.link(replace(pair, left_appearance=forged), now(time)).reason == "appearance_origin_conflict"


def projective_fixture(*, start=1, stationary=False, period=.05):
    """Synthetic original source-local camera points; NOT admitted calibration."""
    matrix = ((100., 0., 0., 0.), (0., 100., 0., 0.), (0., 0., 1., 0.))
    bindings = tuple(ProjectiveCameraBinding(b, "synthetic-prior-with-scale", "raster:" + b.camera_id,
        (640, 480), "original-ground:" + b.camera_id, "own-world-rev:" + b.camera_id,
        "camera-cal-sha:" + b.camera_id, matrix, matrix) for b in SOURCES if b.camera_id != "living-room")
    selected = ProjectiveContinuityPolicy(bindings)
    sides, rows = [], []
    for camera, raw in (("kitchen", 7), ("family-room", 8)):
        binding = next(c for c in bindings if c.source.camera_id == camera)
        samples = []
        for index in range(3):
            frame = start + index
            row = observation(camera, raw, frame=frame, time=10 + (frame - 1) * period, media=5 + (frame - 1) * period)
            u, v = 30. + (0 if stationary else frame * 10), 100.
            measurement = ProjectiveMeasurementBinding("original-measurement:" + row.stamp.observation_id,
                binding.target_frame, binding.world_revision, binding.measurement_calibration_sha256,
                binding.source.transform_id, "original-pose-hypothesis:" + row.stamp.observation_id)
            samples.append(ProjectiveMotionSample(row.stamp, 100 + (frame - 1) * period,
                "body:" + row.stamp.observation_id, binding.raster_id, binding.image_size,
                (u - 5, v - 20, 10., 25.), (u / 100, v / 100, 1.), measurement, (u, v), 1.))
        sides.append(tuple(samples))
        rows.append(row)
    rows = tuple(rows)
    proof_record = ProjectiveContinuityProof(visual_proof(rows, token=f"projective-{start}"),
                                             (ProjectiveMotionPair(*sides),))
    delivery = visual_delivery(rows, sequence=start, evaluation=100 + (start + 1) * period)
    return selected, proof_record, rows, spatial_context(rows), delivery


def projective_kernel_fixture(**kwargs):
    selected, candidate, rows, context, delivery = projective_fixture(**kwargs)
    owner = PersonAssociationLedger("run-a", spatial_policy())
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    return owner, selected, candidate, rows, context, delivery


def projective_history_fixture(*, observe_history=True, deliver_history=True, stationary=False,
                               skip_observe=(), skip_delivery=(), inactive_at=None):
    """Three real 5Hz originals, with public native progression and FIFO delivery."""
    selected, candidate, rows, context, delivery = projective_fixture(period=.2, stationary=stationary)
    owner = PersonAssociationLedger("run-a", spatial_policy())
    pair = candidate.pairs[0]
    for index in range(3):
        current = tuple(SourceObservation(side[index].stamp) for side in (pair.left, pair.right))
        if (observe_history or index == 2) and index not in skip_observe:
            for row in current:
                owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp,
                    "INACTIVE" if index == inactive_at else "ACTIVE"))
        if deliver_history and index < 2 and index not in skip_delivery:
            # Missing 3D history cannot join, but records genuine delivery of the
            # current public observations; it is not an invented historical receipt.
            owner.join_projective_continuity(ProjectiveContinuityProof(visual_proof(current), ()),
                projective_policy=selected, at_interval=spatial_context(current),
                delivery_context=visual_delivery(current, sequence=index + 1, evaluation=100 + index * .2))
    return owner, selected, candidate, rows, context, replace(delivery, sequence=3)


def test_projective_r14_five_hz_history_is_not_current_measurement_freshness():
    owner, selected, candidate, rows, context, delivery = projective_history_fixture()
    originals = repr(candidate)
    result = owner.join_projective_continuity(candidate, projective_policy=selected,
        at_interval=context, delivery_context=delivery)
    assert result.disposition == "accepted", result.reason
    assert owner.stats().active_persons == 1
    assert repr(candidate) == originals  # No original time/frame is rewritten.
    assert owner.spatial_member_error(candidate.pairs[0].left[0].stamp, context) is not None
    assert owner.spatial_member_error(rows[0].stamp, context) is None
    assert owner.projective_continuity_error(rows[0].stamp, context) is None
    assert owner.spatial_member_error(rows[0].stamp, spatial_context(rows, evaluation=10.8)) is not None


@pytest.mark.parametrize("fault", ["no-native-history", "no-delivery-history", "native-gap", "delivery-gap",
    "inactive", "endpoint-age", "appearance-age", "source-gap", "media-gap", "prequeue-gap", "anatomy",
    "cached-depth", "generation", "birth", "source-reset", "run-reset", "stationary"])
def test_projective_r14_history_never_bypasses_current_or_continuity_barriers(fault):
    options = {"observe_history": fault != "no-native-history",
               "deliver_history": fault != "no-delivery-history",
               "skip_observe": (1,) if fault == "native-gap" else (),
               "skip_delivery": (1,) if fault == "delivery-gap" else (),
               "inactive_at": 1 if fault == "inactive" else None,
               "stationary": fault == "stationary"}
    owner, selected, candidate, rows, context, delivery = projective_history_fixture(**options)
    pair = candidate.pairs[0]
    first, middle, endpoint = pair.left
    if fault == "endpoint-age":
        delivery = replace(delivery, evaluation_monotonic_s=100.8)
        context = spatial_context(rows, evaluation=10.8)
    elif fault == "appearance-age":
        edge = candidate.visual.candidates[0]
        profile = edge.left_appearance.profile
        identity = appearance_extraction_identity(first.stamp, profile)
        original = replace(edge.left_appearance, stamp=first.stamp, acquired_at_s=first.stamp.acquired_at_s,
                           evidence_id=identity, extraction_id=identity)
        candidate = replace(candidate, visual=replace(candidate.visual,
                            candidates=(replace(edge, left_appearance=original),)))
    elif fault in {"source-gap", "media-gap", "prequeue-gap"}:
        if fault == "source-gap":
            middle = replace(middle, stamp=replace(middle.stamp, acquired_at_s=10.36))
        elif fault == "media-gap":
            middle = replace(middle, stamp=replace(middle.stamp, media_time_s=5.36))
        else:
            middle = replace(middle, ingress_monotonic_s=100.36)
        pair = replace(pair, left=(first, middle, endpoint))
    elif fault == "anatomy":
        pair = replace(pair, left=(replace(first, support_basis="pose_torso_support"), middle, endpoint))
    elif fault == "cached-depth":
        pair = replace(pair, left=(first, first, endpoint))
    elif fault in {"generation", "birth"}:
        old = replace(first.stamp, **({"member": replace(first.stamp.member, lifecycle_generation=2)}
                      if fault == "generation" else {"lifecycle_started_frame": 0}))
        pair = replace(pair, left=(replace(first, stamp=old), middle, endpoint))
    elif fault == "source-reset":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    elif fault == "run-reset":
        owner.reset_run("another-run")
    candidate = replace(candidate, pairs=(pair,))
    result = owner.join_projective_continuity(candidate, projective_policy=selected,
        at_interval=context, delivery_context=delivery)
    assert result.disposition != "accepted", (fault, result)
    assert not any(m.association_basis == "prior_projective_continuity" for m in owner._members.values())


def test_projective_r14_prior_change_cannot_reuse_a_prechange_history():
    owner, selected, candidate, rows, context, delivery = projective_history_fixture()
    assert owner.join_projective_continuity(candidate, projective_policy=selected,
        at_interval=context, delivery_context=delivery).disposition == "accepted"
    changed = replace(selected, cameras=tuple(replace(c, prior_id="different-prior") for c in selected.cameras))
    result = owner.join_projective_continuity(candidate, projective_policy=changed,
        at_interval=context, delivery_context=replace(delivery, sequence=4))
    assert result.disposition != "accepted"


def test_projective_r14_returning_same_generation_cannot_reuse_preabsence_anchors():
    owner, selected, candidate, rows, context, delivery = projective_history_fixture()
    old = rows[0]
    retained = LifecycleRetention("r14-real-grace", old.stamp, old.stamp.member, "inactive", .35)
    assert owner.withdraw(removal(old, frame=4, time=10.45, media=5.45, retention=retained)).disposition == "removed"
    returned = observation(frame=5, time=10.5, media=5.5)
    owner.observe_spatial(returned, SpatialLifecycleReceipt("actual-return", returned.stamp, "ACTIVE"))
    pair = candidate.pairs[0]
    last = pair.left[-1]
    current_sample = replace(last, stamp=returned.stamp, ingress_monotonic_s=100.5,
        body_evidence_id="actual-return-body", measurement=replace(last.measurement,
        evidence_id="actual-return-measurement", measured_stage_id="actual-return-stage"))
    current = returned, rows[1]
    candidate = replace(candidate, visual=visual_proof(current),
                        pairs=(replace(pair, left=pair.left[:2] + (current_sample,)),))
    delivery = replace(visual_delivery(current, sequence=4, evaluation=100.5),
        ingresses=(CurrentVisualIngress(returned.stamp, 100.5), CurrentVisualIngress(rows[1].stamp, 100.4)))
    result = owner.join_projective_continuity(candidate, projective_policy=selected,
        at_interval=spatial_context(current), delivery_context=delivery)
    assert result.disposition == "association_unresolved" and owner.stats().active_persons == 2


def test_projective_r14_inspection_of_extended_history_is_read_only():
    owner, selected, candidate, rows, context, delivery = projective_history_fixture()
    # An inspection cannot create the delivery continuity needed for admission.
    before = repr(vars(owner))
    report = owner.inspect_projective_continuity(candidate, projective_policy=selected,
        at_interval=context, delivery_context=delivery)
    assert report.error is None and all(supported for _, _, supported in report.pairs)
    assert repr(vars(owner)) == before


def projective_fourth_history_fixture():
    owner, selected, candidate, rows, context, delivery = projective_history_fixture()
    assert owner.join_projective_continuity(candidate, projective_policy=selected,
        at_interval=context, delivery_context=delivery).disposition == "accepted"
    selected, fourth, rows, context, delivery = projective_fixture(start=2, period=.2)
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("fourth:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    delivery = replace(delivery, sequence=4)
    assert owner.join_projective_continuity(fourth, projective_policy=selected,
        at_interval=context, delivery_context=delivery).disposition == "accepted"
    return owner, selected, fourth, rows, context, delivery


def test_projective_r15_fourth_five_hz_grant_uses_current_endpoint_not_oldest_anchor():
    owner, selected, candidate, rows, context, delivery = projective_fourth_history_fixture()
    frozen_originals = repr(candidate)
    assert owner.stats().active_persons == 1
    for row in rows:
        assert owner.projective_continuity_error(row.stamp, context) is None
    for side in (candidate.pairs[0].left, candidate.pairs[0].right):
        assert owner.projective_continuity_error(side[0].stamp, context) == "projective_noncurrent_member"
    assert repr(candidate) == frozen_originals


@pytest.mark.parametrize("fault", ["source-age", "arrival-age", "absence", "source-reset", "run-reset", "inactive", "prior"])
def test_projective_r15_current_grant_still_refuses_expiry_and_barriers(fault):
    owner, selected, candidate, rows, context, delivery = projective_fourth_history_fixture()
    current = rows[0].stamp
    if fault == "source-age":
        context = spatial_context(rows, evaluation=10.96)
    elif fault == "arrival-age":
        # A later real actor clock may advance on a denied observation; it
        # cannot retime the accepted endpoint or its historical originals.
        denied = owner.join_current_visual(replace(candidate.visual, selected_pairs=()),
            at_interval=context, delivery_context=replace(delivery, sequence=5, evaluation_monotonic_s=100.96))
        assert denied.disposition != "accepted"
        assert owner._members[current.member].association_basis == "prior_projective_continuity"
    elif fault == "absence":
        owner.withdraw(removal(rows[0], frame=5, time=10.65, media=5.65))
    elif fault == "source-reset":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    elif fault == "run-reset":
        owner.reset_run("new-run")
    elif fault == "inactive":
        owner.observe_spatial(rows[0], SpatialLifecycleReceipt("now-inactive", current, "INACTIVE"))
    else:
        changed = replace(selected, cameras=tuple(replace(c, prior_id="new-prior") for c in selected.cameras))
        assert owner.join_projective_continuity(candidate, projective_policy=changed,
            at_interval=context, delivery_context=replace(delivery, sequence=5)).disposition != "accepted"
    assert owner.projective_continuity_error(current, context) is not None


def anatomical_fixture(*, start=1, qualified=True, right_offset=0.):
    """Explicit SYNTHETIC static-qualified scalar evidence, NOT native calibration."""
    from hashlib import sha256
    from reid.mv3dt_person_association import (AnatomicalConsensusPolicy, StaticAnatomyQualification,
        AnatomicalDepthSupport, AnatomicalPairSupport, AnatomicalConsensusProof)
    base, moving, rows, context, delivery = projective_fixture(start=start)
    qualification = StaticAnatomyQualification('synthetic-static-only',sha256(repr(base).encode()).hexdigest(), 'a'*64)
    selected = AnatomicalConsensusPolicy(base,qualification if qualified else None)
    sides=[]
    for index, source in enumerate((moving.pairs[0].left[-1],moving.pairs[0].right[-1])):
        supports=[]
        for anatomy,v in (('pose_ankle_support',160.),('pose_torso_support',100.)):
            u=60.+(right_offset if index else 0.)
            m=replace(source.measurement,evidence_id=source.measurement.evidence_id+anatomy,
                      measured_stage_id=source.measurement.measured_stage_id+anatomy)
            original=replace(source,bbox_xywh=(u-10.,80.,20.,100.),original_point=(u/100,v/100,1.),
                measurement=m,body_evidence_id=source.body_evidence_id+anatomy,observed_anchor_uv=(u,v),
                pixel_uncertainty_px=None,observed_support_radius_px=None,support_basis=anatomy)
            supports.append(AnatomicalDepthSupport(original,((u-2,v,u+2,v,2.,2.),),(.99,1.01),'camera_z_m'))
        sides.append(tuple(supports))
    visual=replace(moving.visual,selected_pairs=())
    candidate=AnatomicalConsensusProof(visual,(AnatomicalPairSupport(rows[0].stamp,rows[1].stamp,*sides),))
    return selected,candidate,rows,context,delivery


def test_image_regions_floor_diagnostic_captures_actual_delivery_gap_and_seed_guard():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    live_image_step(owner,1,mode='fresh',wall=100.)
    result,rows,context,delivery=live_image_step(owner,2,mode='missing',wall=100.4)
    record=owner.image_region_decision_diagnostics()['transitions'][-1]
    member=next(m for m in record['members'] if m['member']['source_id']==1)
    note=member['floor_advance_note']
    assert note['cause']=='delivery_ingress_gap'
    assert note['previous_original_stamp']['frame_id']==1 and note['current_original_stamp']['frame_id']==2
    assert note['previous_ingress_monotonic_s']==100. and note['current_ingress_monotonic_s']==100.4
    assert note['ingress_gap_s']==pytest.approx(.4)
    assert any(isinstance(m.get('paired_native_seed'),list) and m['paired_native_seed'][0]['frame_id']==1 for m in record['members'])
    failures=[v['failure'] for m in record['members'] for v in m['live_native_checks'].values()]
    assert 'seed_before_floor' in failures


def test_image_regions_floor_diagnostic_shadow_is_not_a_new_original_or_ingress():
    owner=PersonAssociationLedger('run-a',spatial_policy());_,rows,_,_=live_image_step(owner,1,mode='fresh')
    update=spatial_shadow_update(rows[0],frame=2,time=10.033366,media=5.033366)
    owner.update_spatial_lifecycle(update,retain_image_membership=True)
    note=owner._members[rows[0].stamp.member].floor_advance_note
    assert note.cause=='retained_unmeasured' and note.native_event_frame==2
    assert note.previous_original_stamp==note.current_original_stamp==rows[0].stamp
    assert note.current_ingress_monotonic_s is None


def test_image_regions_floor_diagnostic_failure_parity_detached_and_byte_bound():
    import json
    owners=[PersonAssociationLedger('run-a',spatial_policy()) for _ in range(2)]
    owners[1]._image_diag_check=lambda *_:(_ for _ in ()).throw(RuntimeError('observer'))
    results=[]
    for owner in owners:
        live_image_step(owner,1,mode='fresh',wall=100.)
        results.append(live_image_step(owner,2,mode='missing',wall=100.4)[0])
    assert results[0]==results[1] and owners[0].authority_revision==owners[1].authority_revision
    view=owners[0].image_region_decision_diagnostics()
    record=view['transitions'][-1];record['members'][0]['floor_advance_note']['cause']='tampered'
    assert owners[0].image_region_decision_diagnostics()['transitions'][-1]['members'][0]['floor_advance_note']['cause']!='tampered'
    owners[0]._image_decision_records=owners[0]._image_decision_records[-1:]*8
    view=owners[0].image_region_decision_diagnostics();names=('floor_advance_note','paired_native_seed','owner_availability','live_native_checks','custody_check')
    size=sum(len(json.dumps({k:m[k] for k in names if k in m},separators=(',',':'),allow_nan=False).encode()) for r in view['transitions'] for m in r['members'])
    assert size<=8192


def image_owner_capsule(row, ingress, *, graph='graph-a'):
    from reid.mv3dt_person_association import ImageRegionOwnerAvailability
    s=row.stamp;k=s.member
    return ImageRegionOwnerAvailability(s,graph,s.frame_id,s.media_time_s,1000000+s.frame_id,
        ingress,350000000,(k,),(k,),(),(),ingress,0.,20.,'running')


def test_image_regions_latest_completed_unresolved_exports_actual_first_join_cues():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=two_cue_image_region_fixture(competitor='missing')
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert result.reason=='image_region_unavailable_or_ambiguous' and len(owner._image_decision_records)==0
    view=owner.image_region_decision_diagnostics();record=view['transitions'][-1]
    assert record['record_kind']=='latest_evaluation' and view['last_evaluated_sequence']==1
    assert record['outcome']['reason']==result.reason and record['authority_revision_after']==owner.authority_revision
    assert record['pre_components']==record['post_components'] and len(record['pairs'])==2
    assert record['pairs'][0]['state'] is True and len(record['pairs'][0]['cues'])==2
    assert record['pairs'][1]['state'] is None and not record['pairs'][0]['supported']
    record['pairs'][0]['cues'][0]['forward']=False
    assert owner.image_region_decision_diagnostics()['transitions'][-1]['pairs'][0]['cues'][0]['forward'] is True
    owner._image_diag_begin(replace(p,proof_id='unfinished'),c,replace(d,sequence=2))
    assert owner.image_region_decision_diagnostics()['last_evaluated_sequence']==1
    owner._image_decision_pending=None
    owner.reset_run('run-b')
    assert owner.image_region_decision_diagnostics()['transitions']==[]
    assert owner.image_region_decision_diagnostics()['last_evaluated_sequence'] is None


def test_image_regions_latest_completed_history_deduplicated_and_combined_bounds():
    import json
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in range(1,13):
        selected,p,rows,c,d=image_vote_competition_fixture(frame,crowded=frame>1)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    view=owner.image_region_decision_diagnostics()
    assert len(view['transitions'])==8 and view['transitions'][-1]['record_kind']=='transition_and_latest_evaluation'
    history=tuple(owner._image_decision_records)
    owner.join_estimated_image_regions(replace(p,proof_id='same-topology-later-evaluation'),projective_policy=selected,
        at_interval=c,delivery_context=replace(d,sequence=13),live_track_continuity=True)
    assert tuple(owner._image_decision_records)==history
    view=owner.image_region_decision_diagnostics()
    assert len(view['transitions'])==8 and view['transition_history_omitted']==1
    assert view['transitions'][-1]['sequence']==13 and view['transitions'][-1]['record_kind']=='latest_evaluation'
    assert len({t['sequence'] for t in view['transitions']})==8
    names=('floor_advance_note','paired_native_seed','owner_availability','live_native_checks','custody_check','lineage')
    assert sum(len(json.dumps({n:m[n] for n in names if n in m},separators=(',',':'),allow_nan=False).encode()) for t in view['transitions'] for m in t['members'])<=8192
    assert any(m.get('continuity_detail_status')=='byte_bound_omitted' for t in view['transitions'] for m in t['members'])


def test_image_regions_latest_completed_observer_failure_cannot_change_decision():
    owners=[PersonAssociationLedger('run-a',spatial_policy()) for _ in range(2)]
    owners[1]._image_diag_finish=lambda *_:(_ for _ in ()).throw(RuntimeError('observer-only'))
    results=[]
    for owner in owners:
        selected,p,rows,c,d=two_cue_image_region_fixture(competitor='missing')
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        results.append(owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True))
    assert results[0]==results[1] and owners[0].authority_revision==owners[1].authority_revision
    assert owners[0].snapshot(now(10.))==owners[1].snapshot(now(10.))


def r27_stale_original_fixture(owner, case, *, native_gap=False, before_current=None):
    """Literal R27 frame/age shape, synthetic geometry and real owner events."""
    seed,ff,kf,fa,ka=(233,240,241,.3502184683457017,.2985446653328836) if case==340 else (
        272,274,276,.35743669094517827,.26054974598810077)
    selected,p,old,c,d=two_cue_image_region_fixture(frame=seed,competitor='incompatible')
    old=tuple(replace(r,stamp=replace(r.stamp,acquired_at_s=10.,media_time_s=5.),
        household=HouseholdDescriptor('visitor','claim-'+str(i),1000+i,1)) for i,r in enumerate(old))
    d=visual_delivery(old,sequence=seed,evaluation=100.)
    p=replace(p,cohort=tuple(r.stamp for r in old),originals=tuple(replace(o,stamp=r.stamp,ingress_monotonic_s=100.) for o,r in zip(p.originals,old)),
        owner_availability=tuple(image_owner_capsule(r,100.) for r in old))
    kernel=isinstance(owner,PersonAssociationLedger)
    def observe(rows):
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),
            **({'retain_image_membership':True} if kernel else {}))
    observe(old)
    owner.join_estimated_image_regions(p,at_interval=spatial_context(old),delivery_context=d,
        **({'projective_policy':selected,'live_track_continuity':True} if kernel else {}))
    ledger=owner if kernel else owner.ledger; key=ledger._members[old[0].stamp.member].person_key
    if before_current:before_current(old,spatial_context(old))
    if case==397:
        owner.update_spatial_lifecycle(spatial_shadow_update(old[1],frame=273,time=10.001,media=5.001),
            **({'retain_image_membership':True} if kernel else {}))
    _,p,rows,_,_=two_cue_image_region_fixture(frame=ff,competitor='incompatible')
    rows=tuple(replace(r,household=old[i].household,stamp=replace(r.stamp,frame_id=kf if i==0 else ff,
        acquired_at_s=10.+((kf if i==0 else ff)-seed)*.001+(1. if native_gap else 0.),
        media_time_s=5.+((kf if i==0 else ff)-seed)*.001+(1. if native_gap else 0.))) for i,r in enumerate(rows))
    observe(rows);evaluation=100.4
    ingress=(evaluation-ka,evaluation-fa,evaluation-fa)
    originals=tuple(replace(o,stamp=r.stamp,ingress_monotonic_s=mono) for o,r,mono in zip(p.originals,rows,ingress)
        if not (case==340 and r==rows[0] or case==397 and r==rows[1]))
    p=replace(p,cohort=tuple(r.stamp for r in rows),originals=originals,
        owner_availability=tuple(image_owner_capsule(r,mono) for r,mono in zip(rows,ingress)))
    d=visual_delivery(rows,sequence=case,evaluation=evaluation)
    d=replace(d,ingresses=tuple(replace(row,ingress_monotonic_s=mono) for row,mono in zip(d.ingresses,ingress)))
    return key,selected,p,rows,spatial_context(rows),d


@pytest.mark.parametrize('case',[340,397])
def test_image_regions_r27_stale_originals_keep_old_custody_not_stale_votes(case):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    key,selected,p,rows,c,d=r27_stale_original_fixture(owner,case)
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert len(owner._persons)==2,result
    assert {owner._members[r.stamp.member].person_key for r in rows[:2]}=={key}
    assert owner.image_region_error(rows[0].stamp,c) is None
    assert owner.current_visual_liveness_error(rows[1].stamp)=='current_visual_stale_member_delivery'
    if case==397:assert owner._members[rows[1].stamp.member].projective_native_floor==273


@pytest.mark.parametrize('defect',['nan','prior','raster','alias','bbox','anatomy','ingress'])
def test_image_regions_r27_stale_original_integrity_still_binding(defect):
    owner=PersonAssociationLedger('run-a',spatial_policy());_,selected,p,rows,c,d=r27_stale_original_fixture(owner,397)
    o=p.originals[-1];patch=o.patches[0]
    if defect=='nan':o=replace(o,patches=(replace(patch,anchor_uv=(float('nan'),0.)),))
    elif defect=='prior':o=replace(o,prior_id='different-prior')
    elif defect=='raster':o=replace(o,raster_id='foreign-raster')
    elif defect=='alias':o=replace(o,body_evidence_id=p.originals[0].body_evidence_id)
    elif defect=='bbox':o=replace(o,bbox_xywh=(-1.,0.,1.,1.))
    elif defect=='anatomy':o=replace(o,patches=(replace(patch,anatomy='fabricated'),))
    else:o=replace(o,ingress_monotonic_s=o.ingress_monotonic_s+.01)
    p=replace(p,originals=p.originals[:-1]+(o,))
    r=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert r.reason!='estimated_live_track_continuity'


@pytest.mark.parametrize('barrier',['missing','expired','shadow','not_retained','terminated','recovering',
    'stop','close','reset','absence','native_gap','birth','appearance','rejoin','unseeded','default_false'])
def test_image_regions_r27_dormant_real_custody_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,selected,p,rows,c,d=r27_stale_original_fixture(owner,397,native_gap=barrier=='native_gap')
    cap=p.owner_availability[1];k=rows[1].stamp.member
    changes={'expired':{'last_progress_age_s':20.},'shadow':{'shadow_members':(k,)},
        'not_retained':{'retained_members':(),'active_members':()},
        'terminated':{'retained_members':(),'active_members':(),'terminated_members':(k,)},
        'recovering':{'availability_status':'recovering'}}
    if barrier=='missing':p=replace(p,owner_availability=p.owner_availability[:1])
    elif barrier in changes:p=replace(p,owner_availability=(p.owner_availability[0],replace(cap,**changes[barrier]))+p.owner_availability[2:])
    elif barrier=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    elif barrier=='close':owner.close_image_region_run()
    elif barrier=='reset':owner.reset_source(replace(SOURCES[2],source_epoch=1))
    elif barrier=='absence':owner.withdraw(removal(rows[1],frame=274,time=rows[1].stamp.acquired_at_s,media=rows[1].stamp.media_time_s))
    elif barrier in {'birth','appearance','rejoin','unseeded'}:
        m=owner._members[k]; changes={'birth':{'image_region_native_seed':tuple(replace(s,lifecycle_started_frame=2) for s in m.image_region_native_seed)},
            'appearance':{'appearance_after_frame':272},'rejoin':{'spatial_join_required':True},'unseeded':{'image_region_native_seed':(),'image_region_front_confirmed':False}}
        owner._members[k]=replace(m,**changes[barrier])
    r=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=barrier!='default_false')
    assert r.reason!='estimated_live_track_continuity'


def test_image_regions_r27_public_both_stale_expiry_claims_and_queued_grant():
    from noesis.identity_association_service import IdentityAssociationService
    from tests.test_identity_association_service import spatial_reference
    selected,*_=two_cue_image_region_fixture()
    service=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    queued=[]
    def before(old,c):
        dto=next(x for x in service.spatial_memberships(at_interval=c) if len(x.members)==2)
        queued.append(service.prepare_spatial_physical_member(dto,spatial_reference(old[0],c),at_interval=c).reservation)
        assert queued[-1] is not None
    key,_,p,rows,c,d=r27_stale_original_fixture(service,397,before_current=before)
    service.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    pair=next(x for x in service.spatial_memberships(at_interval=c) if x.physical_key==key)
    assert pair.household_conflict.stable_ids==(1000,1001)
    assert service.prepare_spatial_member(pair,spatial_reference(rows[0],c),at_interval=c).reason=='settled_household_claim_conflict'
    # Queue-stale visual support is not a new association/pose grant. This
    # same-source CURRENT original may still be consumed exactly once for
    # position under actual native custody, retaining strict household denial.
    delayed=service.prepare_spatial_physical_member(pair,spatial_reference(rows[1],c),at_interval=c).reservation
    assert delayed is not None
    assert service.commit_spatial_physical_member(delayed,at_interval=c,consumer=lambda *_:'original-F-position').disposition=='committed'
    assert service.prepare_spatial_physical_member(pair,spatial_reference(rows[1],c),at_interval=c).reason=='spatial_member_measurement_consumed'
    intent=service.prepare_spatial_physical_member(pair,spatial_reference(rows[0],c),at_interval=c).reservation
    assert intent is not None and service.commit_spatial_physical_member(intent,at_interval=c,consumer=lambda *_:'fresh-K').disposition=='committed'
    assert service.commit_spatial_physical_member(queued[0],at_interval=c,consumer=lambda *_:pytest.fail('old revision')).disposition=='denied'
    service.join_estimated_image_regions(replace(p,proof_id='both-stale'),at_interval=c,
        delivery_context=replace(d,sequence=398,evaluation_monotonic_s=100.6))
    pair=next(x for x in service.spatial_memberships(at_interval=c) if x.physical_key==key)
    assert all(e.eligible for e in pair.eligibility)
    assert all(service.ledger.current_visual_liveness_error(r.stamp)=='current_visual_stale_member_delivery' for r in rows[:2])
    # Neither position can be renewed from its already consumed reference.
    assert all(service.prepare_spatial_physical_member(pair,spatial_reference(r,c),at_interval=c).disposition=='denied' for r in rows[:2])
    service.join_estimated_image_regions(replace(p,proof_id='source-expired'),at_interval=c,
        delivery_context=replace(d,sequence=399,evaluation_monotonic_s=121.))
    assert not any(e.eligible for x in service.spatial_memberships(at_interval=c) for e in x.eligibility)


@pytest.mark.parametrize('barrier',['self_false','alternative_true'])
def test_image_regions_r27_dormant_never_erases_fresh_geometric_boundary(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy());_,selected,p,rows,c,d=r27_stale_original_fixture(owner,397)
    _,other,_,_,_=two_cue_image_region_fixture(frame=274,competitor='supported')
    index=1 if barrier=='self_false' else 2
    patches=other.originals[index].patches
    if barrier=='self_false':
        def move(q):
            a,b,e,f,r,s=q.capsule
            return replace(q,anchor_uv=(q.anchor_uv[0],q.anchor_uv[1]+50),capsule=(a,b+50,e,f+50,r,s))
        patches=tuple(move(q) for q in patches)
    fresh=replace(other.originals[index],stamp=rows[index].stamp,ingress_monotonic_s=100.3,patches=patches)
    p=replace(p,originals=tuple(o for o in p.originals if o.stamp.member!=rows[index].stamp.member)+(fresh,),
        owner_availability=tuple(image_owner_capsule(r,100.3) if i==index else cap for i,(r,cap) in enumerate(zip(rows,p.owner_availability))))
    d=replace(d,ingresses=tuple(replace(v,ingress_monotonic_s=100.3) if i==index else v for i,v in enumerate(d.ingresses)))
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert owner._members[rows[0].stamp.member].person_key!=owner._members[rows[1].stamp.member].person_key


def test_image_regions_r27_actual_current_return_same_key_original_seed_unchanged():
    owner=PersonAssociationLedger('run-a',spatial_policy());key,selected,p,rows,c,d=r27_stale_original_fixture(owner,340)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    old=owner._members[rows[1].stamp.member].image_region_original
    _,fresh,_,_,_=two_cue_image_region_fixture(frame=242)
    returned=replace(rows[1],stamp=replace(rows[1].stamp,frame_id=242,observation_id='actual-F-return',
        acquired_at_s=10.009,media_time_s=5.009))
    owner.observe_spatial(returned,SpatialLifecycleReceipt('native:return',returned.stamp,'ACTIVE'),retain_image_membership=True)
    rows=(rows[0],returned,rows[2]);c=spatial_context(rows)
    caps=list(p.owner_availability);caps[1]=image_owner_capsule(returned,100.39)
    caps[2]=replace(caps[2],source_frame_id=242,source_media_time_s=5.009,source_observed_at_us=1000242,
        availability_checked_monotonic_s=100.39,owner_ingress_monotonic_s=100.39)
    p=replace(p,proof_id='actual-current-return',cohort=tuple(r.stamp for r in rows),
        originals=(replace(fresh.originals[1],stamp=returned.stamp,ingress_monotonic_s=100.39),),owner_availability=tuple(caps))
    d=replace(d,sequence=341,evaluation_monotonic_s=100.45,ingresses=tuple(
        replace(v,observation=returned.stamp,ingress_monotonic_s=100.39) if i==1 else v for i,v in enumerate(d.ingresses)))
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert owner._members[returned.stamp.member].person_key==key,result
    assert owner._members[returned.stamp.member].image_region_original is old
    assert owner.image_region_error(returned.stamp,c) is None


def test_image_regions_r27_capsule_lineage_diagnostic_detached_and_same_byte_bound():
    import json
    from copy import deepcopy
    owner=PersonAssociationLedger('run-a',spatial_policy());_,selected,p,rows,c,d=r27_stale_original_fixture(owner,397)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    # Force a real stop transition so the already-paired current receipt is observed.
    owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,
        delivery_context=replace(d,sequence=398),live_track_continuity=True)
    owner._image_decision_records=[deepcopy(owner._image_decision_records[-1]) for _ in range(8)]
    view=owner.image_region_decision_diagnostics()
    names=('floor_advance_note','paired_native_seed','owner_availability','live_native_checks','custody_check','lineage')
    assert sum(len(json.dumps({n:m[n] for n in names if n in m},separators=(',',':'),allow_nan=False).encode()) for t in view['transitions'] for m in t['members'])<=8192
    candidates=[m for m in view['transitions'][-1]['members'] if m.get('owner_availability',{}).get('present')]
    assert candidates, (view['capture_errors'],[(t['sequence'],[(m['member']['raw_id'],m.get('owner_availability'),m.get('lineage')) for m in t['members']]) for t in view['transitions'][-1:]])
    row=candidates[0]
    assert all(type(row['owner_availability'][n]) is bool for n in ('retained','active','shadow','terminated'))
    assert row['lineage']['before']['native_state']=='ACTIVE'
    row['owner_availability']['active']=False
    assert any(m.get('owner_availability',{}).get('active') for m in owner.image_region_decision_diagnostics()['transitions'][-1]['members'])


def dormant_image_fixture(owner, *, houses=None, before_update=None):
    selected,seed,old,context,delivery=two_cue_image_region_fixture()
    if houses is not None:old=tuple(replace(r,household=h) for r,h in zip(old,houses))
    caps=tuple(image_owner_capsule(r,100.) for r in old)
    for r in old:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(replace(seed,owner_availability=caps),
        at_interval=context,delivery_context=delivery,**({'projective_policy':selected,'live_track_continuity':True}
        if isinstance(owner,PersonAssociationLedger) else {}))
    if before_update:before_update(old,context)
    _,fresh,rows,_,_=two_cue_image_region_fixture(frame=2)
    rows=(replace(rows[0],household=old[0].household),old[1]);owner.observe_spatial(rows[0],SpatialLifecycleReceipt('native:new',rows[0].stamp,'ACTIVE'))
    delivery=visual_delivery(rows,sequence=2,evaluation=100.352077130228281)
    delivery=replace(delivery,ingresses=(replace(delivery.ingresses[0],ingress_monotonic_s=100.1),
        replace(delivery.ingresses[1],ingress_monotonic_s=100.)))
    fresh=replace(fresh,cohort=tuple(r.stamp for r in rows),originals=(replace(fresh.originals[0],ingress_monotonic_s=100.1),),
        owner_availability=(image_owner_capsule(rows[0],100.1),caps[1]))
    return selected,fresh,rows,spatial_context(rows),delivery


def test_image_regions_dormant_seq328_same_key_stale_member_never_votes():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,proof,rows,context,delivery=dormant_image_fixture(owner)
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
        delivery_context=delivery,live_track_continuity=True)
    assert len(owner._persons)==1,result
    assert owner.image_region_error(rows[0].stamp,context) is None
    assert owner.current_visual_liveness_error(rows[1].stamp)=='current_visual_stale_member_delivery'


@pytest.mark.parametrize('change',[{'stall_seconds':.1},{'availability_status':'recovering'},
    {'availability_status':'starting'},{'availability_status':'stopped'}])
def test_image_regions_dormant_expired_or_unavailable_source_cannot_hold(change):
    owner=PersonAssociationLedger('run-a',spatial_policy());selected,p,rows,c,d=dormant_image_fixture(owner)
    # Same-frame changed receipts cannot renew or replace ORIGINAL availability.
    p=replace(p,owner_availability=(p.owner_availability[0],replace(p.owner_availability[1],**change)))
    r=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert r.disposition!='accepted'
    assert owner.image_region_error(rows[0].stamp,c) is not None


@pytest.mark.parametrize('change',[{'graph_id':'future-graph'},{'source_frame_id':99},
    {'source_ingress_monotonic_s':100.01},{'availability_checked_monotonic_s':100.01},
    {'owner_ingress_monotonic_s':float('nan')}])
def test_image_regions_dormant_malformed_future_or_retimed_owner_refused(change):
    owner=PersonAssociationLedger('run-a',spatial_policy());selected,p,rows,c,d=dormant_image_fixture(owner)
    p=replace(p,owner_availability=(p.owner_availability[0],replace(p.owner_availability[1],**change)))
    assert owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True).disposition=='association_unresolved'


def test_image_regions_dormant_missing_bound_and_default_false_still_split():
    for live,caps in [(True,False),(False,True)]:
        owner=PersonAssociationLedger('run-a',spatial_policy());selected,p,rows,c,d=dormant_image_fixture(owner)
        if not caps:
            p=replace(p,owner_availability=())
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=live)
        assert len(owner._persons)==2


def test_image_regions_dormant_real_source_stop_and_run_close_revoke():
    for stop in [True,False]:
        owner=PersonAssociationLedger('run-a',spatial_policy());selected,p,rows,c,d=dormant_image_fixture(owner)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
        if stop:owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
        else:owner.close_image_region_run()
        assert owner.image_region_error(rows[0].stamp,c) is not None


def test_image_regions_dormant_cap_expires_without_any_new_receipt_or_vote():
    owner=PersonAssociationLedger('run-a',spatial_policy());selected,p,rows,c,d=dormant_image_fixture(owner)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    original=owner._members[rows[1].stamp.member].image_region_owner_availability
    owner._visual_evaluation=120.
    assert owner.image_region_error(rows[0].stamp,c) is not None
    assert owner._members[rows[1].stamp.member].image_region_owner_availability==original


@pytest.mark.parametrize('barrier',['stop','run_close'])
def test_image_regions_actual_stop_cannot_reintroduce_owner_cached_original(barrier):
    from hashlib import sha256
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=two_cue_image_region_fixture()
    for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    member=owner._members[rows[1].stamp.member]
    original=member.image_region_original;fingerprint=sha256(repr(selected).encode()).hexdigest()
    assert owner._image_region_retained(member,c,fingerprint)
    if barrier=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    else:owner.close_image_region_run()
    assert owner._members[rows[1].stamp.member].image_region_original is original
    assert not owner._image_region_retained(member,c,fingerprint)
    if barrier=='stop':
        result=owner.join_estimated_image_regions(replace(p,proof_id='post-stop-no-private-body',originals=(p.originals[0],)),
            projective_policy=selected,at_interval=c,delivery_context=replace(d,sequence=2,evaluation_monotonic_s=100.05),
            live_track_continuity=True)
        assert result.reason!='estimated_live_track_continuity'
        diagnostic=owner.image_region_decision_diagnostics()['transitions'][-1]
        stopped=next(m for m in diagnostic['members'] if m['member']['source_id']==2)
        assert stopped['input_status']=='absent' and stopped['effective_status']=='unavailable'
        assert owner.image_region_error(rows[1].stamp,c)=='image_region_current_support_required'


def test_image_regions_stopped_queued_current_body_is_not_join_geometry():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=two_cue_image_region_fixture()
    for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert result.disposition!='accepted' and len(owner._persons)==2
    assert owner.image_region_error(rows[1].stamp,c)=='image_region_current_support_required'


@pytest.mark.parametrize('barrier',['native_gap','delivery_gap','birth','inactive','absence','reset','alternate'])
def test_image_regions_dormant_bound_cannot_erase_actual_membership_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy());selected,p,rows,c,d=dormant_image_fixture(owner)
    if barrier=='native_gap':owner._members[rows[0].stamp.member]=replace(owner._members[rows[0].stamp.member],projective_native_floor=2)
    elif barrier=='delivery_gap':owner._members[rows[0].stamp.member]=replace(owner._members[rows[0].stamp.member],projective_delivery_floor=2)
    elif barrier=='birth':owner._members[rows[0].stamp.member]=replace(owner._members[rows[0].stamp.member],image_region_native_seed=())
    elif barrier=='inactive':
        m=owner._members[rows[0].stamp.member];owner._members[rows[0].stamp.member]=replace(m,spatial_lifecycle=replace(m.spatial_lifecycle,native_state='INACTIVE'))
    elif barrier=='absence':owner.withdraw(removal(rows[1]))
    elif barrier=='reset':owner.reset_source(replace(SOURCES[2],source_epoch=1))
    elif barrier=='alternate':
        _,extra,all_rows,_,_=two_cue_image_region_fixture(frame=2,competitor='supported')
        other=all_rows[2];owner.observe_spatial(other,SpatialLifecycleReceipt('new-other',other.stamp,'ACTIVE'))
        p=replace(p,cohort=p.cohort+(other.stamp,),originals=p.originals+(replace(extra.originals[2],ingress_monotonic_s=100.1),))
        d=replace(d,ingresses=d.ingresses+(CurrentVisualIngress(other.stamp,100.1),));c=spatial_context(rows+(other,))
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    if barrier=='delivery_gap':
        # A measurement-delivery frontier alone is no longer person death.
        assert result.reason=='estimated_live_track_continuity'
        assert owner.image_region_error(rows[0].stamp,c) is None
        assert owner.current_visual_liveness_error(rows[1].stamp)=='current_visual_stale_member_delivery'
    else:assert result.reason!='estimated_live_track_continuity'


def image_region_fixture(*, competitor=None, frame=1):
    """Synthetic consistent image epipolar geometry, NOT occupied calibration."""
    from reid.mv3dt_person_association import (ObservedImageRegion, ImageRegionOriginal,
        EstimatedImageRegionProof)
    selected,_,_,_,_=projective_fixture()
    p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.))
    q=((100.,0.,0.,100.),(0.,100.,0.,0.),(0.,0.,1.,0.))
    selected=replace(selected,cameras=tuple(replace(c,own_projection=p if c.source.camera_id=='kitchen' else q,
        peer_projection=q if c.source.camera_id=='kitchen' else p) for c in selected.cameras))
    rows=[observation('kitchen',7,frame=frame,time=10+(frame-1)*.05,media=5+(frame-1)*.05),
          observation('family-room',8,frame=frame,time=10+(frame-1)*.05,media=5+(frame-1)*.05)]
    if competitor is not None:
        rows.append(observation('family-room',9,frame=frame,time=rows[0].stamp.acquired_at_s,media=rows[0].stamp.media_time_s))
    rows=tuple(rows); delivery=visual_delivery(rows,sequence=frame,evaluation=100+(frame-1)*.05)
    originals=[]
    for i,row in enumerate(rows):
        c=next(c for c in selected.cameras if c.source.camera_id==row.stamp.member.camera_id)
        if i==2 and competitor=='missing': continue
        y=150. if i==2 and competitor=='incompatible' else 100.
        region=ObservedImageRegion('pose_torso_support',(62.,y),(60.,y,64.,y,2.,2.))
        originals.append(ImageRegionOriginal(row.stamp,delivery.evaluation_monotonic_s,'body:'+row.stamp.observation_id,
            c.raster_id,c.prior_id,c.image_size,(40.,50.,60.,140.),(region,)))
    proof_record=EstimatedImageRegionProof('synthetic-image-regions:'+str(frame),tuple(r.stamp for r in rows),tuple(originals))
    return selected,proof_record,rows,spatial_context(rows),delivery


def test_unresolved_birth_pending_then_true_join_and_known_pose_gap():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2,3):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None)
        if frame==1:
            rows=rows[1:];p=replace(p,cohort=tuple(r.stamp for r in rows),originals=p.originals[1:],owner_availability=p.owner_availability[1:])
            c=spatial_context(rows);d=visual_delivery(rows,sequence=frame,evaluation=100.)
        for r in rows:
            owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        if frame==2:
            assert owner.image_region_error(rows[0].stamp,c)=='image_region_ambiguous_member'
            p=replace(p,originals=p.originals[1:])
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
        if frame==1:
            key=owner._members[rows[0].stamp.member].person_key
            assert owner.image_region_error(rows[0].stamp,c) is None
        if frame==2:
            assert len(owner._persons)==2
            assert owner.image_region_error(rows[0].stamp,c)=='image_region_ambiguous_member'
            assert owner.image_region_error(rows[1].stamp,c) is None
        if frame==3:
            assert len(owner._persons)==1 and owner._members[rows[0].stamp.member].person_key==key
            assert all(owner.image_region_error(r.stamp,c) is None for r in rows)


@pytest.mark.parametrize('mode',['excluded','unknown','default'])
def test_unresolved_birth_complete_exclusions_vs_unknown_and_default(mode):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(competitor=None)
    p=replace(p,originals=(p.originals[0],replace(p.originals[1],semantic_joints=tuple(
        (i,x,y+20,z) for i,x,y,z in p.originals[1].semantic_joints))))
    if mode=='unknown':p=replace(p,originals=p.originals[:1])
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=mode!='default')
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=mode!='default')
    assert len(owner._persons)==2
    assert all((owner.image_region_error(r.stamp,c) is None)==(mode!='unknown') for r in rows)


def test_unresolved_birth_established_independent_survives_pose_gap_but_not_new_native_gap():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2,10):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None)
        rows=rows[:1];p=replace(p,cohort=(rows[0].stamp,),originals=p.originals[:1],owner_availability=p.owner_availability[:1])
        c=spatial_context(rows);d=visual_delivery(rows,sequence=frame,evaluation=100+(frame-1)*.05)
        r=rows[0];owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        if frame==1:
            owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
                live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
        else:
            assert owner._members[r.stamp.member].image_region_birth_quarantined==(frame==10)


@pytest.mark.parametrize('barrier',['stale','alias','malformed','stop'])
def test_unresolved_birth_invalid_or_unavailable_original_cannot_classify_independent(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(competitor=None)
    p=replace(p,originals=(p.originals[0],replace(p.originals[1],semantic_joints=tuple(
        (i,x,y+20,z) for i,x,y,z in p.originals[1].semantic_joints))))
    if barrier=='stale':d=replace(d,evaluation_monotonic_s=100.36)
    if barrier=='alias':p=replace(p,originals=(replace(p.originals[0],raster_id='other'),)+p.originals[1:])
    if barrier=='malformed':p=replace(p,originals=(replace(p.originals[0],semantic_joints=((True,1.,2.,.9),)*4),)+p.originals[1:])
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    if barrier=='stop':owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert owner._members[rows[0].stamp.member].image_region_birth_quarantined


def semantic_joint_fixture(frame=1, *, competitor='wrong',mixed=False):
    selected,p,rows,c,d=two_cue_image_region_fixture(frame=frame,competitor='supported' if competitor else None)
    originals=[]
    for n,o in enumerate(p.originals):
        dy=20. if n==2 and competitor=='wrong' else 0.
        joints=tuple((i,60.+(i%2)*4.+(2. if n else 0.),(80. if i in (5,6) else 100.)+dy,.9) for i in (5,6,11,12))
        patches=o.patches
        if mixed and n==1:
            patches=tuple(replace(q,anchor_uv=(62.,120.),capsule=(60.,120.,64.,120.,2.,2.))
                if q.anatomy=='pose_shoulder_support' else q for q in patches)
        originals.append(replace(o,semantic_joints=joints,patches=patches))
    return selected,replace(p,originals=tuple(originals),owner_availability=tuple(image_owner_capsule(r,d.evaluation_monotonic_s) for r in rows)),rows,c,d


@pytest.mark.parametrize('mode',['exclude','below','edge','nonoverlap','duplicate','reordered','foreign_id',
    'nonfinite','weak','out_of_raster','bool_id','no_native','stale','stamp','stop','epoch'])
def test_i159_partial_semantic_negative_only_birth(mode):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(competitor=None)
    a,b=p.originals
    a=replace(a,semantic_joints=a.semantic_joints[:2])
    dy=20. if mode=='exclude' else 10. if mode=='edge' else 0.
    b=replace(b,semantic_joints=tuple((i,x,y+dy,v) for i,x,y,v in b.semantic_joints))
    if mode=='nonoverlap':b=replace(b,semantic_joints=b.semantic_joints[2:])
    elif mode=='duplicate':a=replace(a,semantic_joints=(a.semantic_joints[0],)*2)
    elif mode=='reordered':a=replace(a,semantic_joints=a.semantic_joints[::-1])
    elif mode=='foreign_id':a=replace(a,semantic_joints=((7,60.,80.,.9),))
    elif mode=='nonfinite':a=replace(a,semantic_joints=((5,float('nan'),80.,.9),))
    elif mode=='weak':a=replace(a,semantic_joints=((5,60.,80.,.34),))
    elif mode=='out_of_raster':a=replace(a,semantic_joints=((5,-1.,80.,.9),))
    elif mode=='bool_id':a=replace(a,semantic_joints=((True,60.,80.,.9),))
    elif mode=='stale':d=replace(d,evaluation_monotonic_s=d.evaluation_monotonic_s+.36)
    elif mode=='stamp':a=replace(a,stamp=replace(a.stamp,frame_id=a.stamp.frame_id+1))
    elif mode=='epoch':a=replace(a,stamp=replace(a.stamp,member=replace(a.stamp.member,source_epoch=1)))
    p=replace(p,originals=(a,b))
    for r in rows:owner.observe_spatial(r,None if mode=='no_native' else SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    if mode=='stop':owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    if mode=='no_native':
        assert not owner._persons  # Public observation itself requires native authority.
        return
    assert len(owner._persons)==2  # Partial evidence NEVER joins, even zero residual.
    assert all(not owner._members[r.stamp.member].image_region_birth_quarantined for r in rows)==(mode in {'exclude','edge'})
    if mode in {'exclude','edge','below','nonoverlap'}:
        pair=owner.image_region_decision_diagnostics()['transitions'][-1]['pairs'][0]
        assert pair['semantic_mean_px'] is None
        assert pair['semantic_partial_lower_bound_px']==(None if mode=='nonoverlap' else dy/2)
        assert pair['semantic_state'] is (False if mode in {'exclude','edge'} else None)


def test_i159_literal_r93_shoulders_denominator_preserving_bound():
    import json
    from pathlib import Path
    import numpy as np
    prior=json.loads((Path(__file__).resolve().parents[1]/'DS9/config/xc3d_image_regions_prior_r1.json').read_text())
    cameras=[]
    for c in prior['cameras']:
        own=np.asarray(c['intrinsics'])@np.asarray(c['world_extrinsics_col_major']).reshape(4,4,order='F')[:3]
        cameras.append(tuple(tuple(float(v) for v in row) for row in own))
    fs=[PersonAssociationLedger._region_fundamental(own,tuple(tuple(row) for row in c['candidate_peer_projection']))
        for own,c in zip(cameras,prior['cameras'])]
    # Literal K703/15g8 shoulders; hips .0314/.0415 are NOT supplied or filled.
    k=((5,710.5352172851562,507.9140625,.8740234375),(6,686.1103515625,508.96875,.93994140625))
    peers=(
        ((5,486.5584716796875,388.9585876464844,.99072265625),(6,542.0723266601562,382.84063720703125,.99658203125)),
        ((5,365.29461669921875,487.59063720703125,.96630859375),(6,392.3983459472656,483.53985595703125,.9765625)),
        ((5,486.8636474609375,394.6429748535156,.8388671875),(6,463.1363525390625,397.3617248535156,.97802734375)))
    bounds=[PersonAssociationLedger._semantic_joint_lower_bound(*fs,k,f) for f in peers]
    assert bounds==pytest.approx((71.24956014861202,2.42924057138215,58.78864192801231),abs=1e-9)
    assert [v>=33.724887199080946 for v in bounds]==[True,False,True]
    assert all(PersonAssociationLedger._semantic_joint_mean(*fs,k,f) is None for f in peers)


def test_i159_every_partial_sum_is_lower_than_unchanged_complete_mean():
    from itertools import combinations
    selected,p,*_=semantic_joint_fixture(competitor=None)
    a,b=selected.cameras
    fs=(PersonAssociationLedger._region_fundamental(a.own_projection,a.peer_projection),
        PersonAssociationLedger._region_fundamental(b.own_projection,b.peer_projection))
    left=p.originals[0].semantic_joints
    right=tuple((i,x,y+offset,c) for (i,x,y,c),offset in zip(p.originals[1].semantic_joints,(10.,20.,30.,40.)))
    full=PersonAssociationLedger._semantic_joint_mean(*fs,left,right)
    assert full==25.
    for count in (1,2,3):
        for indices in combinations(range(4),count):
            part=tuple(left[i] for i in indices)
            lower=PersonAssociationLedger._semantic_joint_lower_bound(*fs,part,right)
            assert 0<=lower<=full
            assert PersonAssociationLedger._semantic_joint_mean(*fs,part,right) is None


def local_occupancy_fixture(frame, *, extra=True):
    selected,p,rows,c,d=semantic_joint_fixture(frame,competitor='supported' if extra else None)
    originals=[]
    for n,o in enumerate(p.originals):
        dx=30. if n==2 else 0.
        joints=tuple((i,x+dx,y,v) for i,x,y,v in o.semantic_joints)
        center=sum(j[1] for j in joints[:2])/2
        heads=((0,center,65.,.9),(1,center+1,63.,.9),(3,center+3,63.,.9))
        patches=tuple(replace(v,anchor_uv=(v.anchor_uv[0]+dx,v.anchor_uv[1]),
            capsule=tuple(x+dx if i in (0,2) else x for i,x in enumerate(v.capsule))) for v in o.patches)
        originals.append(replace(o,semantic_joints=joints,head_joints=heads,patches=patches))
    return selected,replace(p,originals=tuple(originals)),rows,c,d


def asymmetric_image_gap_fixture(*, gapped_source=1, aligned_final=False, reverse=False):
    """Public native observations: source/media .533866667s, ingress .500050994s.

    The other endpoint advances genuinely within both unchanged .35s bounds.
    No private birth, seed, rejoin or quarantine flag is written by the fixture.
    """
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(1,competitor=None)
    def observe(items):
        for r in (items[::-1] if reverse else items):owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),
            quarantine_unresolved_birth=True,retain_image_membership=True,image_source_local_reactivation=True)
    def join(proof,context,delivery):
        return owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
            delivery_context=delivery,live_track_continuity=True,semantic_joint_max_mean_px=5.,
            unresolved_birth_quarantine=True)
    observe(rows);assert join(p,c,d).disposition=='accepted'
    origin=owner._members[rows[0].stamp.member].person_key
    _,template,new,_,_=semantic_joint_fixture(2,competitor=None)
    middle_s=.3 if aligned_final else .1
    middle=tuple(r if r.stamp.member.source_id==gapped_source else replace(n,
        stamp=replace(n.stamp,acquired_at_s=10.+middle_s,media_time_s=5.+middle_s)) for r,n in zip(rows,new))
    observe(tuple(r for r in middle if r.stamp.member.source_id!=gapped_source))
    mono=tuple(100. if r.stamp.member.source_id==gapped_source else 100.+middle_s if aligned_final else 100.2 for r in middle)
    p=replace(template,cohort=tuple(r.stamp for r in middle),originals=tuple(p.originals[i]
        if r.stamp.member.source_id==gapped_source else replace(o,stamp=r.stamp,ingress_monotonic_s=m)
        for i,(o,r,m) in enumerate(zip(template.originals,middle,mono))),
        owner_availability=tuple(image_owner_capsule(r,m) for r,m in zip(middle,mono)))
    d=replace(visual_delivery(middle,sequence=2,evaluation=max(mono)),
        ingresses=tuple(CurrentVisualIngress(r.stamp,m) for r,m in zip(middle,mono)))
    assert join(p,spatial_context(middle),d).disposition=='accepted'
    _,template,new,_,_=semantic_joint_fixture(3,competitor=None)
    rows=tuple(replace(r,stamp=replace(r.stamp,frame_id=17 if aligned_final or r.stamp.member.source_id==gapped_source else 3,
        acquired_at_s=10.533866667 if aligned_final or r.stamp.member.source_id==gapped_source else 10.136666666,
        media_time_s=5.533866667 if aligned_final or r.stamp.member.source_id==gapped_source else 5.136666666)) for r in new)
    observe(rows)
    mono=tuple(100.500050994 if r.stamp.member.source_id==gapped_source else 100.360557379 for r in rows)
    p=replace(template,cohort=tuple(r.stamp for r in rows),originals=tuple(replace(o,stamp=r.stamp,ingress_monotonic_s=m)
        for o,r,m in zip(template.originals,rows,mono)),owner_availability=tuple(image_owner_capsule(r,m) for r,m in zip(rows,mono)))
    d=replace(visual_delivery(rows,sequence=3,evaluation=100.509850063),
        ingresses=tuple(CurrentVisualIngress(r.stamp,m) for r,m in zip(rows,mono)))
    return owner,origin,selected,p,rows,spatial_context(rows),d


@pytest.mark.parametrize('gapped_source',[1,2])
@pytest.mark.parametrize('reverse',[False,True])
def test_asymmetric_image_gap_retains_continuous_local_birth_and_key(gapped_source,reverse):
    owner,origin,selected,p,rows,c,d=asymmetric_image_gap_fixture(gapped_source=gapped_source,reverse=reverse)
    current=next(r for r in rows if r.stamp.member.source_id!=gapped_source)
    gap=next(r for r in rows if r.stamp.member.source_id==gapped_source)
    assert owner._members[current.stamp.member].image_region_birth_established
    assert not owner._members[gap.stamp.member].image_region_birth_established
    if reverse:p=replace(p,cohort=p.cohort[::-1],originals=p.originals[::-1],owner_availability=p.owner_availability[::-1])
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    continuing=owner._members[current.stamp.member];detached=owner._members[gap.stamp.member]
    assert continuing.person_key==origin and continuing.image_region_birth_established
    assert owner.spatial_member_error(current.stamp,c) is None
    assert owner.image_region_error(current.stamp,c) is None
    assert detached.person_key!=origin and detached.image_region_birth_quarantined
    assert owner.image_region_error(gap.stamp,c)=='image_region_ambiguous_member'
    assert len(owner._persons)==2 and not owner.estimated_split_custodies(at_interval=c)
    original=continuing.observation
    for frame in range(4,13):
        delta=(frame-3)*.25
        fresh=replace(current,stamp=replace(current.stamp,frame_id=frame,observation_id='continuous:'+str(frame),
            acquired_at_s=current.stamp.acquired_at_s+delta,media_time_s=current.stamp.media_time_s+delta))
        owner.observe_spatial(fresh,SpatialLifecycleReceipt('native:'+str(frame),fresh.stamp,'ACTIVE'),
            quarantine_unresolved_birth=True)
        context=spatial_context((gap,fresh));mono=100.360557379+delta
        proof=replace(p,proof_id='continuous:'+str(frame),cohort=(gap.stamp,fresh.stamp),
            originals=tuple(replace(o,stamp=fresh.stamp,ingress_monotonic_s=mono) if o.stamp.member==fresh.stamp.member else o for o in p.originals),
            owner_availability=tuple(image_owner_capsule(fresh,mono) if cap.observation.member==fresh.stamp.member else cap for cap in p.owner_availability))
        delivery=replace(d,sequence=frame,evaluation_monotonic_s=mono+.01,
            ingresses=tuple(CurrentVisualIngress(fresh.stamp,mono) if v.observation.member==fresh.stamp.member else v for v in d.ingresses))
        owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
        assert owner._members[fresh.stamp.member].person_key==origin
        assert owner.image_region_error(fresh.stamp,context) is None
        assert owner.image_region_error(gap.stamp,context) is not None
    assert fresh.stamp.acquired_at_s>original.stamp.acquired_at_s+2.  # New own observations, not renewed old evidence.


@pytest.mark.parametrize('barrier',['stale','inactive','stop','missing_caps','shadow_caps','not_retained',
    'expired_caps','foreign_graph','malformed_original','stale_original','both_gap','false','default'])
def test_asymmetric_image_gap_never_promotes_unqualified_survivor(barrier):
    owner,origin,selected,p,rows,c,d=asymmetric_image_gap_fixture()
    current=rows[1];k=current.stamp.member
    if barrier=='stale':d=replace(d,evaluation_monotonic_s=d.evaluation_monotonic_s+.36)
    elif barrier=='inactive':owner.observe_spatial(current,SpatialLifecycleReceipt('inactive',current.stamp,'INACTIVE'),quarantine_unresolved_birth=True)
    elif barrier=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    elif barrier in {'missing_caps','shadow_caps','not_retained','expired_caps','foreign_graph'}:
        cap=p.owner_availability[1]
        if barrier=='missing_caps':p=replace(p,owner_availability=p.owner_availability[:1])
        else:
            changes={'shadow_caps':{'shadow_members':(k,)},'not_retained':{'active_members':(),'retained_members':()},
                'expired_caps':{'last_progress_age_s':21.},'foreign_graph':{'graph_id':'foreign'}}[barrier]
            p=replace(p,owner_availability=(p.owner_availability[0],replace(cap,**changes)))
    elif barrier in {'malformed_original','stale_original','false'}:
        o=p.originals[1]
        if barrier=='malformed_original':o=replace(o,semantic_joints=(o.semantic_joints[0],)*2)
        elif barrier=='stale_original':o=replace(o,ingress_monotonic_s=o.ingress_monotonic_s-.36)
        else:o=replace(o,patches=split_birth_fixture(3)[1].originals[1].patches)
        p=replace(p,originals=(p.originals[0],o))
    elif barrier=='both_gap':
        fresh=replace(current,stamp=replace(current.stamp,frame_id=18,observation_id='second-native-gap',
            acquired_at_s=10.733866667,media_time_s=5.733866667))
        owner.observe_spatial(fresh,SpatialLifecycleReceipt('gap',fresh.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        current=fresh;rows=(rows[0],fresh);c=spatial_context(rows);mono=100.8
        p=replace(p,cohort=tuple(r.stamp for r in rows),originals=(p.originals[0],replace(p.originals[1],
            stamp=fresh.stamp,ingress_monotonic_s=mono,semantic_joints=p.originals[1].semantic_joints[:2])),
            owner_availability=(p.owner_availability[0],image_owner_capsule(fresh,mono)))
        d=replace(d,evaluation_monotonic_s=mono+.01,ingresses=(d.ingresses[0],CurrentVisualIngress(fresh.stamp,mono)))
        assert not owner._members[k].image_region_birth_established
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=barrier!='default')
    if barrier=='default':
        assert owner._members[k].person_key!=origin  # Exact legacy detach choice, not a new default policy.
    else:assert owner.image_region_error(current.stamp,c) is not None
    if barrier=='false':
        assert owner._members[k].person_key!=origin
        assert owner.image_region_decision_diagnostics()['transitions'][-1]['pairs'][0]['state'] is False


def test_asymmetric_image_gap_selected_current_alternative_keeps_strict_correction():
    owner,origin,selected,p,rows,c,d=asymmetric_image_gap_fixture(aligned_final=True)
    false=split_birth_fixture(3)[1].originals[1]
    old=p.originals[1]
    new=observation('family-room',9,frame=17,time=rows[1].stamp.acquired_at_s,
        media=rows[1].stamp.media_time_s,started=17)
    owner.observe_spatial(new,SpatialLifecycleReceipt('new-positive',new.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    p=replace(p,cohort=p.cohort+(new.stamp,),originals=(p.originals[0],replace(old,
        semantic_joints=false.semantic_joints,patches=false.patches),replace(old,stamp=new.stamp,body_evidence_id='independent-current-body')),
        owner_availability=p.owner_availability+(image_owner_capsule(new,old.ingress_monotonic_s),))
    c=spatial_context(rows+(new,));d=replace(d,ingresses=d.ingresses+(CurrentVisualIngress(new.stamp,old.ingress_monotonic_s),))
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert result.disposition=='accepted'
    assert owner._members[rows[1].stamp.member].person_key!=origin
    assert owner._members[rows[0].stamp.member].person_key==owner._members[new.stamp.member].person_key
    assert len(owner._persons)==2


@pytest.mark.parametrize('barrier',[None,'weak_head','same_head','same_torso','duplicate','clipped',
    'malformed_head','duplicate_head_id','stale','missing_caps','shadow','stop','epoch','unowned_peer','duplicate_pending'])
@pytest.mark.parametrize('reverse',[False,True])
def test_local_occupancy_only_current_separate_body_and_occupied_nonvoting_peer(barrier,reverse):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=local_occupancy_fixture(1,extra=False)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert len(owner._persons)==1
    selected,p,rows,c,d=local_occupancy_fixture(2)
    a,b,x=p.originals
    if barrier=='weak_head':x=replace(x,head_joints=())
    elif barrier=='same_head':x=replace(x,head_joints=b.head_joints)
    elif barrier=='same_torso':x=replace(x,semantic_joints=b.semantic_joints)
    elif barrier=='duplicate':x=replace(x,head_joints=b.head_joints,semantic_joints=b.semantic_joints)
    elif barrier=='clipped':x=replace(x,head_joints=((0,x.bbox_xywh[0],65.,.9),)+x.head_joints[1:])
    elif barrier=='malformed_head':x=replace(x,head_joints=((0,float('nan'),65.,.9),))
    elif barrier=='duplicate_head_id':x=replace(x,head_joints=(x.head_joints[0],)*3)
    elif barrier=='stale':d=replace(d,evaluation_monotonic_s=d.evaluation_monotonic_s+.36)
    elif barrier=='epoch':x=replace(x,stamp=replace(x.stamp,member=replace(x.stamp.member,source_epoch=1)))
    p=replace(p,originals=(a,b,x))
    if barrier in {'unowned_peer','duplicate_pending'}:
        new=observation('kitchen' if barrier=='unowned_peer' else 'family-room',99,frame=2,time=rows[0].stamp.acquired_at_s,media=rows[0].stamp.media_time_s)
        rows=rows+(new,);c=spatial_context(rows)
        d=replace(d,ingresses=d.ingresses+(CurrentVisualIngress(new.stamp,d.evaluation_monotonic_s),))
        p=replace(p,cohort=p.cohort+(new.stamp,),originals=p.originals+(replace(a if barrier=='unowned_peer' else x,stamp=new.stamp,body_evidence_id='new-body'),),
            owner_availability=p.owner_availability+(image_owner_capsule(new,d.evaluation_monotonic_s),))
    if barrier=='missing_caps':p=replace(p,owner_availability=())
    elif barrier=='shadow':p=replace(p,owner_availability=p.owner_availability[:2]+(replace(p.owner_availability[2],shadow_members=(rows[2].stamp.member,)),))
    if reverse:
        p=replace(p,cohort=p.cohort[::-1],originals=p.originals[::-1],owner_availability=p.owner_availability[::-1])
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    if barrier=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    m=owner._members[rows[2].stamp.member]
    assert m.image_region_birth_quarantined is (barrier is not None)
    if barrier=='duplicate_pending':assert owner._members[rows[-1].stamp.member].image_region_birth_quarantined
    assert all(not {rows[0].stamp.member,rows[2].stamp.member}<=set(group) for group in owner._persons.values())
    if barrier is None:
        assert len(owner._persons)==2
        assert owner.image_region_error(rows[0].stamp,c)=='image_region_ambiguous_member'
        assert owner.image_region_error(rows[1].stamp,c) is None
        assert owner.image_region_error(rows[2].stamp,c) is None


def test_local_occupancy_literal_r95_head_and_full_torso_separation():
    from reid.mv3dt_person_association import ImageRegionOriginal
    _selected,p,*_=semantic_joint_fixture(competitor=None)
    data=(
        (((0,480.1549377441406,372.30389404296875,.935546875),(1,481.99530029296875,368.11212158203125,.9384765625),(3,493.4976501464844,367.42303466796875,.94677734375)),
         ((5,516.9624633789062,388.09490966796875,.99609375),(6,488.091552734375,391.1957092285156,.98486328125),(11,510.9812316894531,447.1820373535156,.99755859375),(12,491.7723083496094,445.91876220703125,.99365234375)),(475.,354.,61.5,184.)),
        (((0,410.4651184082031,361.6664123535156,.9365234375),(1,412.43023681640625,356.0121154785156,.974609375),(3,426.4883728027344,354.5796813964844,.96826171875)),
         ((5,452.0348815917969,381.87109375,.990234375),(6,431.6278991699219,386.017578125,.947265625),(11,454.,460.57891845703125,.98779296875),(12,440.5465087890625,460.27734375,.96435546875)),(400.5,335.,81.5,241.5)))
    bodies=tuple(replace(p.originals[0],head_joints=h,semantic_joints=t,bbox_xywh=b) for h,t,b in data)
    assert PersonAssociationLedger._image_observed_supports_separate(*bodies)
    assert PersonAssociationLedger._image_observed_supports_separate(*bodies[::-1])
    assert not PersonAssociationLedger._image_observed_supports_separate(bodies[0],bodies[0])


def test_local_occupancy_genuine_shadow_switch_return_and_unique_rematch():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2,3,4):
        selected,p,rows,c,d=local_occupancy_fixture(frame,extra=frame!=1)
        if frame==2:
            event=spatial_shadow_update(old,frame=frame,time=rows[0].stamp.acquired_at_s,
                media=rows[0].stamp.media_time_s)
            assert owner.update_spatial_lifecycle(event,retain_image_membership=True).reason=='retained_unmeasured'
            cap=replace(old_cap,source_frame_id=frame,source_media_time_s=event.at_media_time_s,
                source_observed_at_us=1000002,shadow_members=(old.stamp.member,),
                availability_checked_monotonic_s=d.evaluation_monotonic_s,
                owner_ingress_monotonic_s=d.evaluation_monotonic_s)
            rows=(rows[0],rows[2])
            p=replace(p,cohort=tuple(r.stamp for r in rows),originals=(p.originals[0],p.originals[2]),
                owner_availability=(p.owner_availability[0],cap,p.owner_availability[2]))
            d=replace(d,ingresses=(d.ingresses[0],d.ingresses[2]))
        if frame==4:
            o=p.originals[2]
            p=replace(p,originals=p.originals[:2]+(replace(o,
                semantic_joints=tuple((i,x,y+20,v) for i,x,y,v in o.semantic_joints)),))
        for r in rows:
            owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),
                image_source_local_reactivation=True,retain_image_membership=True,quarantine_unresolved_birth=True)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
        k=rows[0].stamp.member
        if frame==1:
            old=rows[1];old_cap=p.owner_availability[1];origin=owner._members[k].person_key
        else:
            returned=owner._members[old.stamp.member]
            assert len(owner._persons)==2
            # Real selected alternative moved the peer, not a forged native
            # end or private clearing of an outstanding negative-birth fact.
            assert returned.image_region_split_birth is None
            assert returned.image_region_rejected_birth_peer is None
            if frame==2:
                assert returned.person_key==origin and returned.spatial_join_required
                assert not returned.image_region_participant
                assert owner._members[k].person_key!=origin
            elif frame==3:
                assert returned.person_key==origin and not returned.image_region_birth_quarantined
                assert returned.image_region_vote_blocked_stamp is None and not returned.image_region_vote_pending
                assert owner.image_region_error(returned.observation.stamp,c) is None
                assert owner.image_region_error(rows[0].stamp,c)=='image_region_ambiguous_member'
            else:
                assert owner._members[k].person_key==returned.person_key==origin
                assert all(owner.image_region_error(r.stamp,c) is None for r in rows)


def joint_appearance_fixture(frame=1):
    from reid.mv3dt_person_association import ImageRegionAppearance
    selected,p,rows,c,d=semantic_joint_fixture(frame,competitor='supported')
    profile=replace(spatial_policy().appearance_profile,dimensions=256)
    appearances=[]
    for i,r in enumerate(rows):
        token=appearance_extraction_identity(r.stamp,profile)
        a=AppearanceEvidence(r.stamp,token,profile.model_id,r.stamp.acquired_at_s,r.stamp.clock_domain,token,profile)
        vector=tuple(float(j==(1 if i==2 else 0)) for j in range(256))
        appearances.append(ImageRegionAppearance(a,d.evaluation_monotonic_s,vector))
    return spatial_policy(appearance_profile=profile),selected,replace(p,appearances=tuple(appearances)),rows,c,d


@pytest.mark.parametrize('barrier',[None,'absent','missing','stale','profile','nonfinite','duplicate','tie','zero','wrong_stamp','unknown_body','false_body','no_caps','shadow','epoch','overflow'])
def test_i135_exact_current_joint_appearance_corroboration(barrier):
    policy,selected,p,rows,c,d=joint_appearance_fixture()
    owner=PersonAssociationLedger('run-a',policy)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    if barrier=='absent':p=replace(p,appearances=())
    elif barrier=='missing':p=replace(p,appearances=p.appearances[:2])
    elif barrier=='stale':p=replace(p,appearances=(replace(p.appearances[0],ingress_monotonic_s=d.evaluation_monotonic_s-.36),)+p.appearances[1:])
    elif barrier=='profile':p=replace(p,appearances=(replace(p.appearances[0],appearance=replace(p.appearances[0].appearance,profile=replace(policy.appearance_profile,fingerprint='foreign'))),)+p.appearances[1:])
    elif barrier=='nonfinite':p=replace(p,appearances=(replace(p.appearances[0],vector=(float('nan'),)+p.appearances[0].vector[1:]),)+p.appearances[1:])
    elif barrier=='duplicate':p=replace(p,appearances=p.appearances+(p.appearances[0],))
    elif barrier=='tie':p=replace(p,appearances=p.appearances[:2]+(replace(p.appearances[2],vector=p.appearances[0].vector),))
    elif barrier=='zero':p=replace(p,appearances=(replace(p.appearances[0],vector=(0.,)*256),)+p.appearances[1:])
    elif barrier=='wrong_stamp':p=replace(p,appearances=(replace(p.appearances[0],appearance=replace(p.appearances[0].appearance,stamp=replace(rows[0].stamp,observation_id='foreign'))),)+p.appearances[1:])
    elif barrier=='unknown_body':p=replace(p,originals=p.originals[:2])
    elif barrier=='false_body':p=replace(p,originals=tuple(replace(o,semantic_joints=tuple((i,x,y+100.,v) for i,x,y,v in o.semantic_joints)) if n==1 else o for n,o in enumerate(p.originals)))
    elif barrier=='no_caps':p=replace(p,owner_availability=())
    elif barrier=='shadow':p=replace(p,owner_availability=(replace(p.owner_availability[0],shadow_members=(rows[0].stamp.member,)),)+p.owner_availability[1:])
    elif barrier=='epoch':p=replace(p,appearances=(replace(p.appearances[0],appearance=replace(p.appearances[0].appearance,stamp=replace(rows[0].stamp,member=replace(rows[0].stamp.member,source_epoch=1)))),)+p.appearances[1:])
    elif barrier=='overflow':p=replace(p,appearances=p.appearances*22)
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    joined=owner._members[rows[0].stamp.member].person_key==owner._members[rows[1].stamp.member].person_key
    expected=barrier in (None,'unknown_body')
    assert joined is expected,(barrier,result)
    if barrier!='false_body':assert len(owner._persons)==(2 if expected else 3)


def split_birth_fixture(frame=2):
    selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None)
    o=p.originals[1]
    o=replace(o,patches=tuple(replace(q,anchor_uv=(62.,140.),capsule=(60.,140.,64.,140.,2.,2.)) for q in o.patches),
        semantic_joints=tuple((j[0],j[1],j[2]+40.,j[3]) for j in o.semantic_joints))
    return selected,replace(p,originals=(p.originals[0],o)),rows,c,d


@pytest.mark.parametrize('mode',['occluded','absent_competitor','missing_feature','appearance_tie',
    'unknown_chosen','false_chosen','malformed_supplied','stale_supplied','native_shadow','expired_owner'])
def test_i137_featured_occluded_competitor_remains_unknown_not_global_veto(mode):
    policy,selected,p,rows,c,d=joint_appearance_fixture()
    p=replace(p,originals=p.originals[:2]+(replace(p.originals[2],semantic_joints=()),))
    if mode=='absent_competitor':p=replace(p,originals=p.originals[:2])
    elif mode=='missing_feature':p=replace(p,appearances=p.appearances[:2])
    elif mode=='appearance_tie':p=replace(p,appearances=p.appearances[:2]+(replace(p.appearances[2],vector=p.appearances[0].vector),))
    elif mode=='unknown_chosen':p=replace(p,originals=(replace(p.originals[0],semantic_joints=()),)+p.originals[1:])
    elif mode=='false_chosen':p=replace(p,originals=(replace(p.originals[0],semantic_joints=tuple((i,x,y+100.,v) for i,x,y,v in p.originals[0].semantic_joints)),)+p.originals[1:])
    elif mode=='malformed_supplied':p=replace(p,originals=p.originals[:2]+(replace(p.originals[2],bbox_xywh=(-1.,20.,40.,150.)),))
    elif mode=='stale_supplied':p=replace(p,originals=p.originals[:2]+(replace(p.originals[2],ingress_monotonic_s=p.originals[2].ingress_monotonic_s-.4),))
    elif mode=='native_shadow':p=replace(p,owner_availability=p.owner_availability[:2]+(replace(p.owner_availability[2],shadow_members=(rows[2].stamp.member,)),))
    elif mode=='expired_owner':p=replace(p,owner_availability=p.owner_availability[:2]+(replace(p.owner_availability[2],last_progress_age_s=20.),))
    owner=PersonAssociationLedger('run-a',policy)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    expected=mode in ('occluded','absent_competitor')
    assert (owner._members[rows[0].stamp.member].person_key==owner._members[rows[1].stamp.member].person_key) is expected
    record=owner.image_region_decision_diagnostics()['transitions'][-1]
    other=next(r for r in record['pairs'] if r['right']['raw_id']==rows[2].stamp.member.raw_id)
    if expected:assert other['semantic_state'] is None
    assert not other['appearance_corroborated']


@pytest.mark.parametrize('barrier',[None,'wall','source_age','birth','epoch','profile','floor','missing','shadow','same_stamp_changed'])
def test_i139_qualified_original_feature_on_current_body(barrier):
    policy,selected,p,rows,c,d=joint_appearance_fixture(3)
    _,_,old,_,_,_=joint_appearance_fixture(2)
    p=replace(p,appearances=old.appearances)
    owner=PersonAssociationLedger('run-a',policy)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    first=p.appearances[0]
    if barrier=='wall':p=replace(p,appearances=(replace(first,ingress_monotonic_s=d.evaluation_monotonic_s-.36),)+p.appearances[1:])
    elif barrier in ('source_age','birth','epoch','same_stamp_changed'):
        stamp=(replace(first.appearance.stamp,acquired_at_s=rows[0].stamp.acquired_at_s-.36) if barrier=='source_age' else
            replace(first.appearance.stamp,lifecycle_started_frame=2) if barrier=='birth' else
            replace(first.appearance.stamp,member=replace(first.appearance.stamp.member,source_epoch=1)) if barrier=='epoch' else
            replace(rows[0].stamp,observation_id='different-current-original'))
        token=appearance_extraction_identity(stamp,policy.appearance_profile)
        a=replace(first.appearance,stamp=stamp,acquired_at_s=stamp.acquired_at_s,evidence_id=token,extraction_id=token)
        p=replace(p,appearances=(replace(first,appearance=a),)+p.appearances[1:])
    elif barrier=='profile':p=replace(p,appearances=(replace(first,appearance=replace(first.appearance,profile=replace(policy.appearance_profile,fingerprint='foreign'))),)+p.appearances[1:])
    elif barrier=='floor':
        # Real correction/absence controls are covered by the public lifecycle
        # cases; here the original is literally before its immutable birth.
        stamp=replace(first.appearance.stamp,frame_id=0)
        p=replace(p,appearances=(replace(first,appearance=replace(first.appearance,stamp=stamp)),)+p.appearances[1:])
    elif barrier=='missing':p=replace(p,appearances=p.appearances[:2])
    elif barrier=='shadow':p=replace(p,owner_availability=(replace(p.owner_availability[0],shadow_members=(rows[0].stamp.member,)),)+p.owner_availability[1:])
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert (owner._members[rows[0].stamp.member].person_key==owner._members[rows[1].stamp.member].person_key) is (barrier is None)
    if barrier is None:
        for value in p.appearances:
            assert owner._appearance_origins[value.appearance.stamp.member]==value.appearance
            assert owner._visual_ingresses[value.appearance.stamp.member].appearance_ingress_monotonic_s==value.ingress_monotonic_s


def queued_image_position_fixture(owner, *, paired=True):
    for frame in (1,2):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None)
        if not paired:
            rows=rows[1:];p=replace(p,cohort=tuple(r.stamp for r in rows),
                originals=p.originals[1:],owner_availability=p.owner_availability[1:])
            c=spatial_context(rows)
            d=replace(d,ingresses=tuple(i for i in d.ingresses if i.observation in p.cohort))
        if frame==2:d=replace(d,evaluation_monotonic_s=d.evaluation_monotonic_s+.5)
        for r in rows:
            owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        if isinstance(owner,PersonAssociationLedger):
            owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
                live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
        else:owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    return selected,p,rows,c,d


@pytest.mark.parametrize('paired',[False,True])
@pytest.mark.parametrize('barrier',['none','default','missing_caps','expired_caps','stop','rejoin','birth',
    'birth_quarantine','split_pending','vote','conflict','missing_ingress','shadow','native_gap','not_dormant'])
def test_position_only_delivery_custody_never_changes_association_freshness(paired,barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=queued_image_position_fixture(owner,paired=paired)
    stamp=rows[0].stamp;key=stamp.member;m=owner._members[key]
    before_seed=m.image_region_native_seed;before_original=m.image_region_original
    if barrier=='missing_caps':owner._members[key]=replace(m,image_region_owner_availability=None)
    elif barrier=='expired_caps':owner._members[key]=replace(m,image_region_owner_availability=replace(m.image_region_owner_availability,last_progress_age_s=20.))
    elif barrier=='stop':owner.stop_image_region_source(next(b for b in SOURCES if b.source_id==key.source_id),graph_id='graph-a')
    elif barrier=='rejoin':owner._members[key]=replace(m,spatial_join_required=True)
    elif barrier=='birth':
        changed=replace(stamp,lifecycle_started_frame=stamp.frame_id)
        owner.observe_spatial(replace(rows[0],stamp=changed),SpatialLifecycleReceipt('actual-new-birth',changed,'ACTIVE'))
        stamp=changed
    elif barrier=='birth_quarantine':owner._members[key]=replace(m,image_region_birth_quarantined=True)
    elif barrier=='split_pending':owner._members[key]=replace(m,image_region_rejected_birth_peer=stamp)
    elif barrier=='vote':owner._members[key]=replace(m,image_region_vote_blocked_stamp=stamp)
    elif barrier=='conflict':owner._members[key]=replace(m,image_region_conflict_custody=True)
    elif barrier=='missing_ingress':owner._visual_ingresses.pop(key)
    elif barrier=='shadow':owner.update_spatial_lifecycle(spatial_shadow_update(rows[0],frame=3,time=10.1,media=5.1),retain_image_membership=True)
    elif barrier=='native_gap':
        changed=replace(stamp,frame_id=20,acquired_at_s=11.,media_time_s=6.,observation_id='gap-current')
        owner.observe_spatial(replace(rows[0],stamp=changed),SpatialLifecycleReceipt('actual-gap',changed,'ACTIVE'))
        stamp=changed;c=spatial_context((replace(rows[0],stamp=changed),)+rows[1:])
    elif barrier=='not_dormant' and paired:owner._members[key]=replace(m,image_region_dormant_custody=False)
    result=owner.image_region_position_delivery_error(stamp,c,live_track_continuity=barrier!='default')
    assert (result is None)==(barrier=='none' or barrier=='not_dormant' and not paired)
    if barrier=='none':
        assert owner.current_visual_liveness_error(stamp)=='current_visual_stale_member_delivery'
        assert owner._members[key].image_region_native_seed==before_seed
        assert owner._members[key].image_region_original==before_original


def test_position_only_delay_never_bootstraps_new_semantic_join():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(competitor=None)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('unseeded',r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,
        delivery_context=replace(d,evaluation_monotonic_s=d.evaluation_monotonic_s+.5),
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert result.disposition!='accepted' and len(owner._persons)==2
    assert all(owner.image_region_position_delivery_error(r.stamp,c,live_track_continuity=True) for r in rows)


def test_split_birth_same_lineage_false_does_not_mint_additional_person():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for fixture in (semantic_joint_fixture,split_birth_fixture):
        selected,p,rows,c,d=fixture(competitor=None) if fixture is semantic_joint_fixture else fixture()
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert len(owner._persons)==2  # FALSE remains a physical-component split.
    assert owner.image_region_error(rows[0].stamp,c) is None
    assert owner.image_region_error(rows[1].stamp,c)=='image_region_ambiguous_member'


def test_split_birth_genuine_unique_other_person_releases_negative_provenance():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None) if frame==1 else split_birth_fixture(frame)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    # A genuine CURRENT third member is uniquely the peer's positive partner.
    selected,p,rows,c,d=semantic_joint_fixture(3,competitor='wrong')
    false=split_birth_fixture(3)[1].originals[1]
    true=replace(p.originals[2],patches=p.originals[1].patches,semantic_joints=p.originals[1].semantic_joints)
    p=replace(p,originals=(p.originals[0],false,true))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    before=owner.authority_revision
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert owner._members[rows[0].stamp.member].person_key==owner._members[rows[2].stamp.member].person_key
    f=owner._members[rows[1].stamp.member]
    assert f.image_region_split_birth is None and f.image_region_rejected_birth_peer is None
    assert f.image_region_birth_established and owner.image_region_error(rows[1].stamp,c) is None
    assert owner.authority_revision>before


def split_birth_shadow_fixture():
    """R48 shape: genuine FALSE split, then retained/unmeasured origin peer."""
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None) if frame==1 else split_birth_fixture(frame)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    old=rows[0];origin=owner._members[old.stamp.member].person_key
    selected,p,rows,c,d=split_birth_fixture(3)
    current=rows[1]
    update=spatial_shadow_update(old,frame=3,time=rows[0].stamp.acquired_at_s,
        media=rows[0].stamp.media_time_s,token='native-retained-K288')
    assert owner.update_spatial_lifecycle(update,retain_image_membership=True).disposition=='observed'
    owner.observe_spatial(current,SpatialLifecycleReceipt('native-F288',current.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    cap=replace(image_owner_capsule(old,100.05),source_frame_id=3,
        source_media_time_s=rows[0].stamp.media_time_s,source_observed_at_us=1000003,
        availability_checked_monotonic_s=d.evaluation_monotonic_s,
        owner_ingress_monotonic_s=d.evaluation_monotonic_s,
        shadow_members=(old.stamp.member,))
    p=replace(p,cohort=(current.stamp,),originals=p.originals[1:],
        owner_availability=(cap,p.owner_availability[1]))
    d=replace(d,ingresses=d.ingresses[1:])
    return owner,origin,old,current,selected,p,c,d


def estimated_split_custody_fixture(*, live=True, birth=True):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None) if frame==1 else split_birth_fixture(frame)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=birth)
        result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=live,semantic_joint_max_mean_px=5. if live else None,
            unresolved_birth_quarantine=birth if live else False)
    return owner,selected,p,rows,c,d,result


def i142_ordered_shadow_custody_fixture():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(1,competitor=None)
    for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    event=spatial_shadow_update(rows[1],frame=2,time=rows[1].stamp.acquired_at_s+.05,
        media=rows[1].stamp.media_time_s+.05)
    assert not owner.update_spatial_lifecycle(event,retain_image_membership=True).corrected_members
    family=replace(p.owner_availability[1],source_frame_id=2,source_media_time_s=event.at_media_time_s,
        source_observed_at_us=1000002,shadow_members=(rows[1].stamp.member,),
        availability_checked_monotonic_s=100.05,owner_ingress_monotonic_s=100.05)
    c=replace(c,intervals=tuple(replace(i,acquisition_min_s=event.at_acquired_at_s,
        acquisition_max_s=event.at_acquired_at_s,evaluation_min_s=event.at_acquired_at_s,
        evaluation_max_s=event.at_acquired_at_s) if i.binding.source_id==2 else i for i in c.intervals))
    proof=replace(p,cohort=(rows[0].stamp,),originals=p.originals[:1],owner_availability=(p.owner_availability[0],family))
    d=replace(d,sequence=2,evaluation_monotonic_s=100.05,ingresses=d.ingresses[:1])
    owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    return owner,rows,c


@pytest.mark.parametrize('fault',[None,'missing','graph','not_retained','not_active','terminated','not_shadow',
    'expired','media_expired','birth','epoch','event','floor','stop','closed'])
def test_i142_shadow_custody_is_old_original_only_and_barrier_bound(fault):
    owner,rows,c=i142_ordered_shadow_custody_fixture();key=rows[1].stamp.member
    m=owner._members[key];cap=m.image_region_owner_availability
    if fault=='missing':m=replace(m,image_region_owner_availability=None)
    elif fault in {'graph','not_retained','not_active','terminated','not_shadow','expired','media_expired'}:
        change={'graph':{'graph_id':'foreign'},'not_retained':{'retained_members':()},
            'not_active':{'active_members':()},'terminated':{'terminated_members':(key,)},
            'not_shadow':{'shadow_members':()},'expired':{'stall_seconds':.001,'last_progress_age_s':1.},
            'media_expired':{'source_media_time_s':cap.observation.media_time_s+1.}}[fault]
        m=replace(m,image_region_owner_availability=replace(cap,**change))
    elif fault in {'birth','epoch'}:
        stamp=replace(m.observation.stamp,**({'lifecycle_started_frame':99} if fault=='birth' else
            {'member':replace(key,source_epoch=1)}))
        m=replace(m,observation=replace(m.observation,stamp=stamp))
    elif fault=='event':m=replace(m,spatial_lifecycle_update=None)
    elif fault=='floor':m=replace(m,projective_native_floor=m.projective_native_floor+1)
    elif fault=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    elif fault=='closed':owner.close_image_region_run()
    owner._members[key]=m
    current=owner._members[rows[0].stamp.member];before=owner.authority_revision
    assert owner._image_split_shadow_custody(current,c,current.image_region_policy) is (fault is None)
    assert not owner._image_pair_custody(current,c,current.image_region_policy,fresh_delivery=False)
    assert owner.image_region_error(rows[1].stamp,c) is not None
    assert owner.authority_revision==before  # The helper is a pure creation predicate.


@pytest.mark.parametrize('fault',[None,'duplicate','same_frame','backward','graph','epoch','birth','participant'])
def test_i144_only_forward_nonparticipant_gap_is_member_local(fault):
    from reid.mv3dt_person_association import EstimatedImageRegionProof
    owner,rows,c=i142_ordered_shadow_custody_fixture()
    key=rows[1].stamp.member;member=owner._members[key];previous=member.image_region_owner_availability
    event=spatial_shadow_update(rows[1],frame=3,time=rows[1].stamp.acquired_at_s+.1,
        media=rows[1].stamp.media_time_s+.1,token='later-native-shadow')
    if fault!='same_frame':owner.update_spatial_lifecycle(event,retain_image_membership=True)
    cap=replace(previous,source_frame_id=3,source_media_time_s=event.at_media_time_s,
        source_observed_at_us=1000003,owner_ingress_monotonic_s=100.5,
        availability_checked_monotonic_s=100.5)
    if fault=='same_frame':
        # Actual current cursor and last accepted owner frame are unchanged.
        cap=replace(previous,availability_status='recovering')
    elif fault=='backward':cap=replace(cap,source_frame_id=1)
    elif fault=='graph':cap=replace(cap,graph_id='foreign')
    elif fault=='epoch':cap=replace(cap,observation=replace(cap.observation,member=replace(key,source_epoch=1)))
    elif fault=='birth':cap=replace(cap,observation=replace(cap.observation,lifecycle_started_frame=99))
    elif fault=='participant':
        # Negative branch control only: a current participant must NEVER get
        # the member-local exception, even with the identical forward gap.
        owner._members[key]=replace(owner._members[key],image_region_participant=True)
    proof=EstimatedImageRegionProof('i144-only-forward',owner.image_region_active_cohort(),(),
        (cap,cap) if fault=='duplicate' else (cap,))
    before=(tuple(owner._members.items()),owner.authority_revision)
    result=owner._image_owner_capsules(proof,CurrentVisualDeliveryContext('i144',3,100.5,()))
    if fault is None:
        assert result is None
        ended=owner._members[key]
        assert ended.image_region_owner_availability is None
        assert ended.image_region_native_seed==() and not ended.image_region_front_confirmed
        assert owner.authority_revision>before[1]
    else:
        assert result in {'image_region_owner_availability_binding','image_region_owner_availability_reordered'}
        assert (tuple(owner._members.items()),owner.authority_revision)==before


def test_i124_refusal_is_actual_same_decision_detached_and_not_authority():
    owner,selected,p,rows,c,d,result=estimated_split_custody_fixture()
    view,=owner.estimated_split_custodies(at_interval=c)
    before=owner._authority_revision
    m=owner._members[view.prior_seed[0].member]
    owner._members[view.prior_seed[0].member]=replace(m,image_region_owner_availability=None)
    assert owner.estimated_split_custodies(at_interval=c)==()
    diagnostic=owner.estimated_split_custody_diagnostic(view,at_interval=c,observed_monotonic_s=121.)
    assert diagnostic['filter_reason']=='split_custody_owner_unavailable'
    assert diagnostic['fixed_deadline_mono_s']==120.05 and diagnostic['expired_now']
    assert diagnostic['same_decision'] and owner._authority_revision==before
    diagnostic['member_guards'][0][0]=-999
    assert owner.estimated_split_custody_diagnostic(view,at_interval=c,observed_monotonic_s=100.2)['member_guards'][0][0]!=-999
    assert owner.estimated_split_custody_diagnostic(None,at_interval=c,observed_monotonic_s=100.)=={'available':False}
    assert owner.estimated_split_custody_diagnostic(view,at_interval=c,observed_monotonic_s=float('nan'))=={'available':False}


def test_estimated_split_custody_immutable_episode_does_not_change_false_or_membership():
    owner,selected,p,rows,c,d,result=estimated_split_custody_fixture()
    view,=owner.estimated_split_custodies(at_interval=c)
    assert view.origin_key in view.successor_keys and len(owner._persons)==2
    assert view.split_stamps==tuple(r.stamp for r in rows)
    assert all(s.frame_id==1 for s in view.prior_seed)
    assert view.native_owner_receipts==p.owner_availability
    assert view.owner_deadline_monotonic_s==120.05
    assert owner.image_region_decision_diagnostics()['transitions'][-1]['pairs'][0]['semantic_state'] is False
    assert owner.image_region_error(rows[1].stamp,c)=='image_region_ambiguous_member'
    with pytest.raises(AttributeError):view.episode_id='forged'
    selected,p,rows,c,d=split_birth_fixture(3)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('fresh:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert owner.estimated_split_custodies(at_interval=c)==(view,)  # No new episode/deadline.


def test_i121_native_destination_is_exact_split_mapping_and_diagnostic_is_detached():
    owner,selected,p,rows,c,d,result=estimated_split_custody_fixture()
    view,=owner.estimated_split_custodies(at_interval=c)
    assert view.native_destinations==tuple((s,owner._members[s.member].person_key) for s in view.split_stamps)
    before=(tuple(owner._members.items()),owner.authority_revision)
    diagnostic=owner.image_region_decision_diagnostics()
    note=diagnostic['transitions'][-1]['split_custody_creation']
    assert note['failure']=='issued' and note['native_custody'] and note['births_established']
    assert note['owner_deadline_monotonic_s']==view.owner_deadline_monotonic_s
    note['members'].clear()
    assert len(owner.image_region_decision_diagnostics()['transitions'][-1]['split_custody_creation']['members'])==2
    assert (tuple(owner._members.items()),owner.authority_revision)==before


def test_i121_missing_native_authority_refuses_episode_without_changing_false():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None) if frame==1 else split_birth_fixture(frame)
        if frame==2:p=replace(p,owner_availability=())
        for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert not owner.estimated_split_custodies(at_interval=c)
    record=owner.image_region_decision_diagnostics()['transitions'][-1]
    assert record['pairs'][0]['semantic_state'] is False
    assert record['split_custody_creation']['failure']=='native_custody'
    assert all(not m['owner_present'] for m in record['split_custody_creation']['members'])


def test_r116_rejoin_receipt_preserves_original_episode_and_deadline():
    owner,selected,_,_,c,_,_=estimated_split_custody_fixture()
    prior,=owner.estimated_split_custodies(at_interval=c)
    selected,p,rows,c,d=semantic_joint_fixture(3,competitor=None)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('rejoin:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    view,=owner.estimated_split_custodies(at_interval=c)
    assert result.disposition=='accepted' and len(owner._persons)==1
    assert view.rejoined_key==prior.origin_key and view.rejoined_stamps==tuple(r.stamp for r in rows)
    assert view.rejoin_revision<=owner.authority_revision
    assert replace(view,rejoined_key=None,rejoined_stamps=(),rejoin_revision=None)==prior
    assert all(m.image_region_split_custody is view for m in owner._members.values())
    assert all(owner.spatial_member_error(r.stamp,c) is None for r in rows)


@pytest.mark.parametrize('barrier',['missing_caps','expired_caps','shadow_caps','not_retained','native_gap','ambiguous','false','stop'])
def test_r116_rejoin_never_certifies_unavailable_or_unselected_same_pair(barrier):
    owner,selected,_,_,c,_,_=estimated_split_custody_fixture()
    prior,=owner.estimated_split_custodies(at_interval=c)
    if barrier=='stop':owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
    selected,p,rows,c,d=(split_birth_fixture(3) if barrier=='false' else
        semantic_joint_fixture(20 if barrier=='native_gap' else 3,competitor='missing' if barrier=='ambiguous' else None))
    if barrier=='ambiguous':
        p=replace(p,originals=p.originals[:2])
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('current:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    if barrier=='missing_caps':p=replace(p,owner_availability=())
    elif barrier=='expired_caps':p=replace(p,owner_availability=tuple(replace(x,last_progress_age_s=20.) for x in p.owner_availability))
    elif barrier=='shadow_caps':p=replace(p,owner_availability=(replace(p.owner_availability[0],shadow_members=(rows[0].stamp.member,)),)+p.owner_availability[1:])
    elif barrier=='not_retained':p=replace(p,owner_availability=tuple(replace(x,retained_members=()) for x in p.owner_availability))
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert not any(v.rejoined_key is not None for v in owner.estimated_split_custodies(at_interval=c))
    assert prior.owner_deadline_monotonic_s==120.05


@pytest.mark.parametrize('barrier',['false','native_gap','stop','absence','reset','appearance'])
def test_r116_existing_reunion_receipt_revokes_instead_of_renewing(barrier):
    owner,selected,_,_,c,_,_=estimated_split_custody_fixture()
    for frame in (3,4):
        if frame==4 and barrier=='stop':owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
        if frame==4 and barrier=='absence':owner.withdraw(removal(rows[0],frame=4,time=10.15,media=5.15))
        if frame==4 and barrier=='reset':
            owner.reset_source(replace(SOURCES[1],source_epoch=1))
            assert not any(m.image_region_split_custody for m in owner._members.values())
            return
        selected,p,rows,c,d=(split_birth_fixture(frame) if frame==4 and barrier=='false' else
            semantic_joint_fixture(20 if frame==4 and barrier=='native_gap' else frame,competitor=None))
        if frame==4 and barrier=='absence':
            assert not any(m.image_region_split_custody for m in owner._members.values())
            return
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        if frame==4 and barrier=='appearance':
            # Existing appearance barrier authority, never a diagnostic input.
            k=rows[0].stamp.member
            owner._members[k]=replace(owner._members[k],appearance_after_frame=rows[0].stamp.frame_id)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
        if frame==3:
            prior,=owner.estimated_split_custodies(at_interval=c)
            assert prior.rejoined_key is not None
    assert owner.estimated_split_custodies(at_interval=c)==()


@pytest.mark.parametrize('barrier',['missing_owner','malformed_owner','expired_owner','not_retained','termination','absence','reset','new_birth','stop','run_closed','native_gap','policy','alternative','rejoin'])
def test_estimated_split_custody_native_owner_barriers_are_not_position_permissions(barrier):
    owner,selected,p,rows,c,d,_=estimated_split_custody_fixture()
    view,=owner.estimated_split_custodies(at_interval=c)
    if barrier in {'termination','absence'}:
        owner.withdraw(removal(rows[0],kind='termination' if barrier=='termination' else 'absence',frame=3,time=10.1,media=5.1))
    elif barrier=='reset':owner.reset_source(replace(SOURCES[1],source_epoch=1))
    elif barrier=='new_birth':
        old=rows[0]
        owner.withdraw(removal(old,kind='termination',frame=3,time=10.1,media=5.1))
        born=observation('kitchen',old.stamp.member.raw_id,generation=2,frame=4,time=10.15,media=5.15,started=4)
        owner.observe_spatial(born,SpatialLifecycleReceipt('new-owner',born.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    elif barrier=='stop':owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
    elif barrier=='run_closed':owner.close_image_region_run()
    else:
        selected,p,rows,c,d=semantic_joint_fixture(20 if barrier=='native_gap' else 3,competitor='wrong' if barrier=='alternative' else None) if barrier in {'alternative','rejoin'} else split_birth_fixture(20 if barrier=='native_gap' else 3)
        if barrier=='alternative':
            false=split_birth_fixture(3)[1].originals[1]
            p=replace(p,originals=(p.originals[0],false,replace(p.originals[2],patches=p.originals[1].patches,semantic_joints=p.originals[1].semantic_joints)))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('next:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        if barrier=='missing_owner':p=replace(p,owner_availability=p.owner_availability[1:])
        elif barrier=='malformed_owner':p=replace(p,owner_availability=(replace(p.owner_availability[0],graph_id='foreign'),)+p.owner_availability[1:])
        elif barrier=='expired_owner':p=replace(p,owner_availability=tuple(replace(cap,last_progress_age_s=20.) for cap in p.owner_availability))
        elif barrier=='not_retained':p=replace(p,owner_availability=tuple(replace(cap,retained_members=(),active_members=()) for cap in p.owner_availability))
        elif barrier=='policy':selected=replace(selected,cameras=tuple(replace(cam,prior_id='changed') for cam in selected.cameras))
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert view not in owner.estimated_split_custodies(at_interval=c)


def test_estimated_split_custody_original_deadline_not_renewed_by_new_receipts():
    owner,selected,p,rows,c,d,_=estimated_split_custody_fixture()
    view,=owner.estimated_split_custodies(at_interval=c)
    selected,p,rows,c,d=split_birth_fixture(3)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('next:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    d=replace(d,evaluation_monotonic_s=view.owner_deadline_monotonic_s+.001)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert owner.estimated_split_custodies(at_interval=c)==()
    assert all(m.image_region_split_custody is None for m in owner._members.values())


def test_estimated_split_custody_finite_inputs_never_publish_overflowed_owner_deadline():
    _,p,rows,c,d=semantic_joint_fixture(competitor=None)
    caps=p.owner_availability
    assert PersonAssociationLedger._image_split_deadline(caps)==120.
    # Deliberate scalar fault injection, not a reachable normal host clock or
    # a claim of an admitted production cohort. The resulting typed view must
    # NEVER carry infinity even though each input scalar is finite.
    overflow=replace(caps[0],availability_checked_monotonic_s=1e308,stall_seconds=1e308)
    assert PersonAssociationLedger._image_split_deadline((overflow,caps[1])) is None


def test_estimated_split_custody_cap179_unknown_self_unsupported_true_is_not_reassignment():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(competitor='wrong')
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('seed',r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    third=rows[2].stamp.member;third_key=owner._members[third].person_key
    selected,p,rows,c,d=semantic_joint_fixture(2,competitor='supported')
    # Literal UNKNOWN cannot uniquely support the possible current alternative.
    # NEW hold-only policy preserves old-point custody, NOT membership or votes.
    p=replace(p,originals=(p.originals[0],p.originals[2]))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('current',r.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    view,=owner.estimated_split_custodies(at_interval=c)
    assert len(owner._persons)==3 and third_key not in view.successor_keys
    assert owner._members[third].person_key==third_key and owner._members[third].image_region_birth_established
    pairs=owner.image_region_decision_diagnostics()['transitions'][-1]['pairs']
    assert any(p['semantic_state'] is True and not p['supported'] and not p['unique'] for p in pairs)


@pytest.mark.parametrize('live,birth',[(False,False),(True,False)])
def test_estimated_split_custody_not_issued_in_default_or_nonbirth_mode(live,birth):
    owner,_,_,_,c,_,_=estimated_split_custody_fixture(live=live,birth=birth)
    assert owner.estimated_split_custodies(at_interval=c)==()


def test_split_birth_retained_singleton_shadow_is_not_native_owner_end():
    owner,origin,old,current,selected,p,c,d=split_birth_shadow_fixture()
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    assert result.disposition=='observed',result
    peer=owner._members[old.stamp.member];member=owner._members[current.stamp.member]
    assert peer.person_key==origin and peer.spatial_join_required and not peer.image_region_participant
    assert peer.spatial_lifecycle.native_state=='ACTIVE' and peer.observation.stamp==old.stamp
    assert len(owner._persons)==2  # FALSE remains split, no remerge or point grant.
    assert not member.image_region_birth_established
    assert owner.image_region_error(current.stamp,c)=='image_region_ambiguous_member'
    assert owner.spatial_member_error(old.stamp,c) is not None


@pytest.mark.parametrize('barrier',['missing_owner','expired_owner','not_retained','terminated','stop','absence','shadow_gap'])
def test_split_birth_shadow_custody_releases_only_real_owner_barriers(barrier):
    owner,origin,old,current,selected,p,c,d=split_birth_shadow_fixture()
    cap=p.owner_availability[0]
    if barrier=='missing_owner':p=replace(p,owner_availability=p.owner_availability[1:])
    elif barrier=='expired_owner':p=replace(p,owner_availability=(replace(cap,last_progress_age_s=20.),)+p.owner_availability[1:])
    elif barrier=='not_retained':p=replace(p,owner_availability=(replace(cap,retained_members=(),active_members=(),shadow_members=()),)+p.owner_availability[1:])
    elif barrier in {'terminated','absence'}:
        assert owner.withdraw(removal(old,kind='termination' if barrier=='terminated' else 'absence',frame=4,time=10.15,media=5.15)).disposition=='removed'
        p=replace(p,owner_availability=p.owner_availability[1:])
    elif barrier=='stop':
        owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
        p=replace(p,owner_availability=p.owner_availability[1:])
    elif barrier=='shadow_gap':
        cap=replace(cap,owner_grace_ns=10000000)
        p=replace(p,owner_availability=(cap,)+p.owner_availability[1:])
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    m=owner._members[current.stamp.member]
    assert m.image_region_rejected_birth_peer is None
    assert m.image_region_birth_established  # Fresh no-peer proof, not timer release alone.


def test_split_birth_shadow_fresh_return_keeps_negative_fact_without_remerge():
    owner,origin,old,current,selected,p,c,d=split_birth_shadow_fixture()
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    rejected=owner._members[current.stamp.member].image_region_rejected_birth_peer
    selected,p,rows,c,d=split_birth_fixture(4)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('actual-return:'+r.stamp.observation_id,r.stamp,'ACTIVE'),
        image_source_local_reactivation=True,quarantine_unresolved_birth=True)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    m=owner._members[rows[1].stamp.member]
    assert m.image_region_rejected_birth_peer==rejected and not m.image_region_birth_established
    assert owner._members[rows[0].stamp.member].person_key==origin and len(owner._persons)==2


@pytest.mark.parametrize('barrier',['reset','new_birth','run_closed','native_gap_then_shadow'])
def test_split_birth_shadow_negative_fact_never_crosses_real_lineage_barrier(barrier):
    owner,origin,old,current,selected,p,c,d=split_birth_shadow_fixture()
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    key=current.stamp.member;before=owner.authority_revision
    if barrier=='reset':
        owner.reset_source(replace(SOURCES[1],source_epoch=1))
        assert owner._members[key].image_region_rejected_birth_peer is None
    elif barrier=='new_birth':
        owner.withdraw(removal(old,kind='termination',frame=4,time=10.15,media=5.15))
        _,_,rows,_,_=split_birth_fixture(5)
        born=replace(rows[0],stamp=replace(rows[0].stamp,
            member=replace(old.stamp.member,lifecycle_generation=old.stamp.member.lifecycle_generation+1),
            lifecycle_started_frame=5))
        assert owner.observe_spatial(born,SpatialLifecycleReceipt('real-new-birth',born.stamp,'ACTIVE'),
            quarantine_unresolved_birth=True).disposition=='observed'
        assert owner._members[key].image_region_rejected_birth_peer is None
    elif barrier=='run_closed':
        owner.close_image_region_run()
        assert owner._image_rejected_birth_custody(owner._members[key],owner._members[old.stamp.member]) is None
    else:
        accepted_floor=owner._members[key].image_region_rejected_birth_native_floor
        _,_,rows,_,_=split_birth_fixture(20)
        returned=rows[0]
        owner.observe_spatial(returned,SpatialLifecycleReceipt('real-native-gap',returned.stamp,'ACTIVE'),
            image_source_local_reactivation=True,quarantine_unresolved_birth=True)
        owner.update_spatial_lifecycle(spatial_shadow_update(returned,frame=21,time=11.,media=6.),retain_image_membership=True)
        assert owner._members[old.stamp.member].projective_native_floor==21
        assert owner._members[key].image_region_rejected_birth_native_floor==accepted_floor
        assert owner._image_rejected_birth_custody(owner._members[key],owner._members[old.stamp.member]) is None
    assert owner.authority_revision>before


@pytest.mark.parametrize('end',['renewed_false','true','deadline','stop','native_gap','peer_absence','default_off'])
def test_split_birth_bounded_custody_and_real_independent_controls(end):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    pending=None
    for frame in (1,2,3):
        fixture=semantic_joint_fixture if frame==1 or end=='true' and frame==3 else split_birth_fixture
        selected,p,rows,c,d=fixture(frame,competitor=None) if fixture is semantic_joint_fixture else fixture(frame)
        if frame==3 and end=='native_gap':
            selected,p,rows,c,d=split_birth_fixture(20)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),quarantine_unresolved_birth=end!='default_off')
        if frame==3 and end=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
        if frame==3 and end=='peer_absence':
            assert owner.withdraw(removal(rows[0],frame=3,time=10.1,media=5.1)).disposition=='removed'
            p=replace(p,cohort=(rows[1].stamp,),originals=(p.originals[1],),owner_availability=(p.owner_availability[1],))
        if frame==3 and end=='deadline':
            # No new timer: the ORIGINAL real availability deadline is fixed.
            p=replace(p,owner_availability=tuple(replace(cap,stall_seconds=.01) for cap in p.owner_availability))
            owner._visual_evaluation=pending.expires_monotonic_s+.01
            d=replace(d,evaluation_monotonic_s=pending.expires_monotonic_s+.01)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=end!='default_off')
        m=owner._members[rows[1].stamp.member]
        if frame==2:pending=m.image_region_split_birth
    if end=='true':
        assert len(owner._persons)==1 and m.image_region_split_birth is None
    elif end=='renewed_false':
        assert m.image_region_split_birth==pending and m.image_region_birth_quarantined
    elif end=='stop':
        assert owner.image_region_error(rows[1].stamp,c)=='image_region_current_support_required'
    elif end in {'native_gap','peer_absence','default_off'}:
        assert m.image_region_split_birth is None
    else:
        assert m.image_region_split_birth is None and not m.image_region_birth_established


def test_semantic_joint_causal_full_regions_ambiguous_points_unique_and_literal_shoulder_false():
    selected,p,rows,c,d=semantic_joint_fixture(mixed=True)
    strict=PersonAssociationLedger('run-a',spatial_policy());owner=PersonAssociationLedger('run-a',spatial_policy())
    for ledger in (strict,owner):
        for r in rows:ledger.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    strict.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert strict._members[rows[0].stamp.member].person_key==strict._members[rows[2].stamp.member].person_key
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.)
    assert result.reason=='estimated_semantic_joint_correspondence' and len(owner._persons)==2
    diagnostic=owner.image_region_decision_diagnostics()['transitions'][-1]['pairs'][0]
    assert diagnostic['state'] is False and diagnostic['semantic_state'] is True
    assert diagnostic['semantic_mean_px']==pytest.approx(0.)


@pytest.mark.parametrize('barrier',['unknown','tie','all_wrong','stale','malformed','alias','torso_false','gap','stop','reset'])
def test_semantic_joint_new_join_strict_negatives(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=semantic_joint_fixture(9 if barrier=='gap' else 1,competitor='tie' if barrier=='tie' else 'wrong')
    if barrier=='unknown':p=replace(p,originals=p.originals[:2]+(replace(p.originals[2],semantic_joints=()),))
    elif barrier=='all_wrong':
        p=replace(p,originals=(p.originals[0],)+tuple(replace(o,semantic_joints=tuple((j[0],j[1],j[2]+20,j[3]) for j in o.semantic_joints)) for o in p.originals[1:]))
    elif barrier=='malformed':p=replace(p,originals=(replace(p.originals[0],semantic_joints=((True,60.,80.,.9),)*4),)+p.originals[1:])
    elif barrier=='alias':p=replace(p,originals=(replace(p.originals[0],raster_id='bad'),)+p.originals[1:])
    elif barrier=='torso_false':
        o=p.originals[1];p=replace(p,originals=(p.originals[0],replace(o,patches=tuple(replace(q,anchor_uv=(62.,160.),capsule=(60.,160.,64.,160.,2.,2.)) if q.anatomy=='pose_torso_support' else q for q in o.patches)))+p.originals[2:])
    elif barrier=='stale':d=replace(d,evaluation_monotonic_s=d.evaluation_monotonic_s+.36)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    if barrier=='stop':owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
    elif barrier=='reset':owner.reset_source(replace(SOURCES[1],source_epoch=1))
    if barrier=='gap':
        # A first join is never allowed by a manufactured existing custody flag.
        p=replace(p,originals=())
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,semantic_joint_max_mean_px=5.)
    assert result.disposition!='accepted'


def test_semantic_joint_seeded_missing_raw_uses_actual_caps_without_confirmation_renewal():
    owner=PersonAssociationLedger('run-a',spatial_policy());key=None;seed=None
    for frame in (1,2,3):
        selected,p,rows,c,d=semantic_joint_fixture(frame)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        if frame>1:p=replace(p,originals=p.originals[1:])
        result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.)
        assert result.disposition=='accepted',result
        current=owner._members[rows[0].stamp.member]
        if frame==1:key=current.person_key;seed=current.image_region_original
        assert current.person_key==key and current.image_region_original==seed
    owner.withdraw(removal(rows[1],frame=3,time=rows[1].stamp.acquired_at_s,media=rows[1].stamp.media_time_s))
    assert len(owner._persons[key])==1


def contradictory_cue_fixture(frame=1, *, mixed=False, competitor='incompatible'):
    selected,p,rows,c,d=two_cue_image_region_fixture(frame=frame,competitor=competitor)
    if mixed:
        original=p.originals[1]
        original=replace(original,patches=tuple(replace(q,anchor_uv=(62.,120.),
            capsule=(60.,120.,64.,120.,2.,2.)) if q.anatomy=='pose_shoulder_support' else q
            for q in original.patches))
        p=replace(p,originals=(p.originals[0],original)+p.originals[2:])
    return selected,replace(p,owner_availability=tuple(image_owner_capsule(r,d.evaluation_monotonic_s) for r in rows)),rows,c,d


def test_image_regions_contradictory_cue_causal_seeded_pair_keeps_key_but_no_votes():
    owner=PersonAssociationLedger('run-a',spatial_policy());key=None
    for frame in (1,2):
        selected,p,rows,c,d=contradictory_cue_fixture(frame,mixed=frame==2)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,contradictory_cue_quarantine=True)
        if frame==1:key=owner._members[rows[0].stamp.member].person_key
    assert owner._members[rows[0].stamp.member].person_key==owner._members[rows[1].stamp.member].person_key==key
    assert result.reason=='estimated_image_region_conflict_custody'
    assert all(owner.image_region_error(r.stamp,c)=='image_region_contradictory_cue_member' for r in rows[:2])


@pytest.mark.parametrize('barrier',['default','unseeded','all_false','torso_false','alternative_true',
    'invalid','stale','missing_caps','expired_caps','shadow_caps','not_active','not_retained',
    'terminated','native_gap','birth','appearance','rejoin','stop','run_close','reset','directional'])
def test_image_regions_contradictory_cue_keeps_strict_barriers(barrier,monkeypatch):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,p,rows,c,d=contradictory_cue_fixture()
    if barrier!='unseeded':
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    selected,p,rows,c,d=contradictory_cue_fixture(9 if barrier=='native_gap' else 2,mixed=True,
        competitor='supported' if barrier=='alternative_true' else 'incompatible')
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    if barrier in {'all_false','torso_false'}:
        o=p.originals[1]
        patches=tuple(replace(q,anchor_uv=(62.,120. if q.anatomy=='pose_torso_support' else 80.),
            capsule=(60.,120. if q.anatomy=='pose_torso_support' else 80.,64.,120. if q.anatomy=='pose_torso_support' else 80.,2.,2.))
            if barrier=='torso_false' or q.anatomy=='pose_torso_support' else q for q in o.patches)
        p=replace(p,originals=(p.originals[0],replace(o,patches=patches))+p.originals[2:])
    elif barrier=='invalid':p=replace(p,originals=(replace(p.originals[0],raster_id='wrong'),)+p.originals[1:])
    elif barrier=='stale':
        p=replace(p,originals=tuple(replace(o,ingress_monotonic_s=o.ingress_monotonic_s-.36) for o in p.originals))
        d=replace(d,ingresses=tuple(replace(i,ingress_monotonic_s=i.ingress_monotonic_s-.36) for i in d.ingresses))
    elif barrier=='missing_caps':p=replace(p,owner_availability=p.owner_availability[:1])
    elif barrier in {'expired_caps','shadow_caps','not_active','not_retained','terminated'}:
        cap=p.owner_availability[1];k=rows[1].stamp.member
        changes={'expired_caps':{'last_progress_age_s':20.},'shadow_caps':{'shadow_members':(k,)},
            'not_active':{'active_members':()},'not_retained':{'retained_members':(),'active_members':()},
            'terminated':{'terminated_members':(k,)}}
        p=replace(p,owner_availability=(p.owner_availability[0],replace(cap,**changes[barrier]))+p.owner_availability[2:])
    elif barrier in {'birth','appearance','rejoin'}:
        k=rows[0].stamp.member;m=owner._members[k]
        changes={'birth':{'image_region_native_seed':tuple(replace(s,lifecycle_started_frame=2) for s in m.image_region_native_seed)},
            'appearance':{'appearance_after_frame':1},'rejoin':{'spatial_join_required':True}}
        owner._members[k]=replace(m,**changes[barrier])
    elif barrier=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    elif barrier=='run_close':owner.close_image_region_run()
    elif barrier=='reset':owner.reset_source(replace(SOURCES[2],source_epoch=1))
    elif barrier=='directional':
        original=owner._region_feasible
        monkeypatch.setattr(owner,'_region_feasible',lambda f,a,b,*,cameras:
            True if a[1]==120. and cameras[0][0][3]==100. else original(f,a,b,cameras=cameras))
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,contradictory_cue_quarantine=barrier!='default')
    assert not any(m.image_region_conflict_custody for m in owner._members.values())


@pytest.mark.parametrize('value',[None,0,1,'true',()])
def test_image_regions_contradictory_cue_requires_exact_boolean(value):
    owner=PersonAssociationLedger('run-a',spatial_policy());selected,p,_,c,d=contradictory_cue_fixture()
    assert owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=True,contradictory_cue_quarantine=value).reason=='image_region_contradictory_cue_opt_in_required'


def test_image_regions_contradictory_cue_diagnostic_failure_cannot_change_policy(monkeypatch):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    monkeypatch.setattr(owner,'_image_diag_begin',lambda *_:None)
    for frame in (1,2):
        selected,p,rows,c,d=contradictory_cue_fixture(frame,mixed=frame==2)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,contradictory_cue_quarantine=True)
    assert result.reason=='estimated_image_region_conflict_custody'


@pytest.mark.parametrize('next_event',['partial','absence','termination','stop','run_close'])
def test_image_regions_contradictory_cue_cannot_recover_by_partial_or_survive_real_barrier(next_event):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,p,rows,c,d=contradictory_cue_fixture(frame,mixed=frame==2)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,contradictory_cue_quarantine=True)
    key=owner._members[rows[0].stamp.member].person_key
    if next_event in {'absence','termination'}:
        owner.withdraw(replace(removal(rows[1],frame=2,time=rows[1].stamp.acquired_at_s,media=rows[1].stamp.media_time_s),
            kind=next_event))
        assert owner._members[rows[0].stamp.member].person_key==key
        assert not owner._members[rows[0].stamp.member].image_region_conflict_custody
    elif next_event in {'stop','run_close'}:
        if next_event=='stop':owner.stop_image_region_source(SOURCES[1],graph_id='graph-a')
        else:owner.close_image_region_run()
        assert owner.image_region_error(rows[0].stamp,c)=='image_region_current_support_required'
    else:
        selected,p,rows,c,d=contradictory_cue_fixture(3)
        p=replace(p,originals=tuple(replace(o,patches=o.patches[:1]) for o in p.originals))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,contradictory_cue_quarantine=True)
        assert owner._members[rows[0].stamp.member].person_key!=owner._members[rows[1].stamp.member].person_key


def two_cue_image_region_fixture(**kwargs):
    """Two genuine named synthetic regions; no duplicate anatomy witness."""
    selected,proof,rows,context,delivery=image_region_fixture(**kwargs)
    originals=[]
    for o in proof.originals:
        torso=o.patches[0];a,b,c,d,r,s=torso.capsule
        shoulder=replace(torso,anatomy='pose_shoulder_support',
            anchor_uv=(torso.anchor_uv[0],torso.anchor_uv[1]-20),capsule=(a,b-20,c,d-20,r,s))
        originals.append(replace(o,patches=(torso,shoulder)))
    return selected,replace(proof,originals=tuple(originals)),rows,context,delivery


def test_image_regions_stationary_positive_needs_no_appearance_xyz_depth_or_history():
    selected,candidate,rows,context,delivery=image_region_fixture()
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.disposition=='accepted',result
    assert len({m.person_key for m in owner._members.values()})==1
    assert not hasattr(candidate,'candidates') and not hasattr(candidate.originals[0],'original_point')
    assert owner.image_region_error(rows[0].stamp,context) is None


@pytest.mark.parametrize('competitor,heads',[('incompatible',2),('supported',3),('missing',3)])
def test_image_regions_complete_graph_unknown_and_ambiguity_do_not_vanish(competitor,heads):
    selected,candidate,rows,context,delivery=image_region_fixture(competitor=competitor)
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert len(owner._persons)==heads,result
    if competitor!='incompatible': assert result.disposition=='observed'


@pytest.mark.parametrize('fault',['source_age','delivery_age','raster','prior','anatomy','capsule','anchor','empty','retag','epoch','birth','cohort','degenerate','contradiction'])
def test_image_regions_missing_stale_invalid_and_all_named_cues_are_conservative(fault):
    from reid.mv3dt_person_association import ObservedImageRegion
    selected,candidate,rows,context,delivery=image_region_fixture()
    o=candidate.originals[0]
    if fault=='source_age': context=replace(context,intervals=tuple(replace(i,evaluation_min_s=i.evaluation_min_s+.36,evaluation_max_s=i.evaluation_max_s+.36) for i in context.intervals))
    elif fault=='delivery_age': delivery=replace(delivery,evaluation_monotonic_s=delivery.evaluation_monotonic_s+.36)
    elif fault=='raster': o=replace(o,raster_id='wrong')
    elif fault=='prior': o=replace(o,prior_id='wrong')
    elif fault=='anatomy': o=replace(o,patches=(replace(o.patches[0],anatomy='box-bottom'),))
    elif fault=='capsule': o=replace(o,patches=(replace(o.patches[0],capsule=(60.,100.,64.,100.,-1.,2.)),))
    elif fault=='anchor': o=replace(o,patches=(replace(o.patches[0],anchor_uv=(500.,300.)),))
    elif fault=='empty': o=replace(o,patches=())
    elif fault=='retag': o=replace(o,ingress_monotonic_s=o.ingress_monotonic_s-.1)
    elif fault=='epoch': o=replace(o,stamp=replace(o.stamp,member=replace(o.stamp.member,source_epoch=8)))
    elif fault=='birth': o=replace(o,stamp=replace(o.stamp,lifecycle_started_frame=99))
    elif fault=='cohort': candidate=replace(candidate,cohort=candidate.cohort[:1])
    elif fault=='degenerate': selected=replace(selected,cameras=tuple(replace(c,peer_projection=c.own_projection) for c in selected.cameras))
    elif fault=='contradiction':
        a=ObservedImageRegion('pose_shoulder_support',(62.,80.),(60.,80.,64.,80.,2.,2.))
        b=replace(a,anchor_uv=(62.,120.),capsule=(60.,120.,64.,120.,2.,2.))
        o=replace(o,patches=o.patches+(a,))
        candidate=replace(candidate,originals=(o,replace(candidate.originals[1],patches=candidate.originals[1].patches+(b,))))
    if fault!='contradiction': candidate=replace(candidate,originals=(o,)+candidate.originals[1:])
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.disposition!='accepted',result
    assert len(owner._persons)==2


def test_image_regions_fresh_update_requires_actor_then_bounded_confirmed_continuation():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        if frame==2:
            assert owner.image_region_error(rows[0].stamp,context)=='image_region_current_support_required'
            candidate=replace(candidate,originals=())
        result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.disposition=='accepted' and len(owner._persons)==1
    assert result.reason=='estimated_image_region_continuation'
    assert all(m.image_region_original.stamp.frame_id==1 for m in owner._members.values())


def test_image_regions_rectangular_bbox_ambiguity_is_not_named_region_support():
    selected,candidate,rows,context,delivery=image_region_fixture(competitor='supported')
    # Actual R5 broad-box ambiguity: no named-region input can turn two coarse
    # possible bodies into a unique edge. No occupied numeric labels are inputs.
    candidate=replace(candidate,originals=tuple(replace(o,patches=()) for o in candidate.originals))
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery).disposition=='observed'
    assert len(owner._persons)==3


@pytest.mark.parametrize('change',['original_id_reuse','prior_change','same_stamp_retag','same_source_collision'])
def test_image_regions_original_lineage_and_prior_cannot_retag_support(change):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,first,rows,context,delivery=image_region_fixture()
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(first,projective_policy=selected,at_interval=context,delivery_context=delivery).disposition=='accepted'
    selected,candidate,rows,context,delivery=image_region_fixture(frame=1 if change=='same_stamp_retag' else 2)
    if change=='same_stamp_retag':
        delivery=replace(delivery,sequence=2)
        candidate=replace(candidate,originals=(replace(candidate.originals[0],body_evidence_id='renamed'),)+candidate.originals[1:])
    else:
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    if change=='original_id_reuse': candidate=replace(candidate,originals=(replace(candidate.originals[0],body_evidence_id=first.originals[0].body_evidence_id),)+candidate.originals[1:])
    if change=='prior_change': selected=replace(selected,cameras=tuple(replace(c,prior_id='new-unbound-prior') for c in selected.cameras))
    if change=='same_source_collision': candidate=replace(candidate,originals=(candidate.originals[0],candidate.originals[0]))
    result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.disposition!='accepted'
    assert all(owner.image_region_error(m.observation.stamp,context) is not None or len(owner._persons[m.person_key])==1 for m in owner._members.values())


def test_image_regions_sustained_renewals_preserve_physical_key_and_bounded_storage():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    physical=None
    for frame in range(1,1001):
        selected,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
        assert result.disposition=='accepted',result
        if physical is None: physical=result.person_key
        assert result.person_key==physical
    assert len(owner._members)==2 and len(owner._persons)==1
    assert len(owner._visual_ingresses)==2 and len(owner._receipts)==0
    assert sum(m.image_region_original is not None for m in owner._members.values())==2


def test_image_regions_exact_projective_camera_minors_work_for_nonrectified_pair():
    from reid.mv3dt_person_association import ObservedImageRegion
    p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.))
    q=((0.,-100.,0.,100.),(100.,0.,0.,50.),(0.,0.,1.,0.))
    f=PersonAssociationLedger._region_fundamental(p,q)
    reverse=PersonAssociationLedger._region_fundamental(q,p)
    for point in ((1.,2.,4.,1.),(2.,1.,3.,1.),(3.,2.,6.,1.)):
        def image(matrix):
            r=tuple(sum(a*b for a,b in zip(row,point)) for row in matrix)
            return r[0]/r[2],r[1]/r[2]
        a,b=image(p),image(q)
        ca=(*a,*a,1.,1.); cb=(*b,*b,1.,1.)
        assert PersonAssociationLedger._region_feasible(f,ca,cb,cameras=(p,q)) is True
        assert PersonAssociationLedger._region_feasible(reverse,cb,ca,cameras=(q,p)) is True
    rank2=(p[0],p[1],p[1])
    assert PersonAssociationLedger._region_fundamental(rank2,q) is None


def test_image_regions_25_pixel_conflict_is_not_forced_and_curved_remainder_unknown():
    selected,candidate,rows,context,delivery=image_region_fixture()
    b=candidate.originals[1]; p=b.patches[0]
    b=replace(b,patches=(replace(p,anchor_uv=(62.,125.),capsule=(60.,125.,64.,125.,2.,2.)),))
    candidate=replace(candidate,originals=(candidate.originals[0],b))
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery).disposition=='observed'
    assert len(owner._persons)==2


def test_image_regions_low_cosine_causal_boundary_does_not_change_legacy_anatomy():
    selected,candidate,rows,context,delivery=anatomical_fixture()
    old=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: old.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    candidate=replace(candidate,visual=replace(candidate.visual,candidates=tuple(replace(c,similarity=.53898) for c in candidate.visual.candidates)))
    assert old.join_anatomical_consensus(candidate,anatomical_policy=selected,at_interval=context,delivery_context=delivery).disposition=='observed'
    assert len(old._persons)==2
    # Separate image entry has no appearance/cosine field to lower or fabricate.
    selected,candidate,rows,context,delivery=image_region_fixture()
    new=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: new.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert new.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery).disposition=='accepted'


@pytest.mark.parametrize('case',['front','behind-both','behind-one','infinite','degenerate'])
def test_image_regions_internal_actual_witness_cheirality(case):
    p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.))
    q=((100.,0.,0.,100.),(0.,100.,0.,0.),(0.,0.,1.,0.))
    a,b=(60.,100.),(80.,100.)
    if case=='behind-both': a,b=b,a
    if case=='behind-one': q=tuple(tuple(-v for v in row) for row in q)
    if case=='infinite': b=a
    if case=='degenerate': q=p
    assert PersonAssociationLedger._region_front_witness(p,q,a,b) is (case=='front')
    f=PersonAssociationLedger._region_fundamental(p,q)
    if f is not None and case in {'front','behind-both','behind-one'}:
        ca=(*a,*a,1.,1.); cb=(*b,*b,1.,1.)
        assert PersonAssociationLedger._region_feasible(f,ca,cb,cameras=(p,q)) is (True if case=='front' else None)


@pytest.mark.parametrize('expiry',['source','wall'])
def test_image_regions_continuation_separate_original_ages_expire_without_renewal(expiry):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    confirmed=None
    for frame in range(1,10):
        selected,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        wall=100+(frame-1)*(.0125 if expiry=='source' else .1)
        delivery=replace(delivery,evaluation_monotonic_s=wall,
            ingresses=tuple(CurrentVisualIngress(r.stamp,wall) for r in rows))
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        if frame==1:
            candidate=replace(candidate,originals=tuple(replace(o,ingress_monotonic_s=wall) for o in candidate.originals))
        else: candidate=replace(candidate,originals=())
        result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
        if frame==1:
            confirmed=tuple(m.image_region_original for m in owner._members.values())
        if frame<=3:
            assert result.disposition=='accepted',result
            assert tuple(m.image_region_original for m in owner._members.values())==confirmed
    assert result.disposition=='observed' and len(owner._persons)==2


def test_image_regions_alternating_pose_gaps_do_not_walk_confirmation_forward():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in range(1,10):
        selected,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        if frame>1: candidate=replace(candidate,originals=(candidate.originals[frame%2],))
        result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
        if frame<=3:
            assert result.disposition=='accepted',result
            assert all(m.image_region_original.stamp.frame_id==1 for m in owner._members.values())
    assert len(owner._persons)==2


@pytest.mark.parametrize('barrier',['unknown_competitor','contradiction','invalid_current','native_gap','birth','inactive','prior'])
def test_image_regions_confirmed_continuation_cannot_ignore_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,candidate,rows,context,delivery=image_region_fixture()
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery).disposition=='accepted'
    selected,candidate,rows,context,delivery=image_region_fixture(frame=9 if barrier=='native_gap' else 2,
        competitor='missing' if barrier=='unknown_competitor' else None)
    if barrier=='birth':
        rows=tuple(replace(r,stamp=replace(r.stamp,lifecycle_started_frame=2)) for r in rows)
        candidate=replace(candidate,cohort=tuple(r.stamp for r in rows))
        delivery=replace(delivery,ingresses=tuple(CurrentVisualIngress(r.stamp,delivery.evaluation_monotonic_s) for r in rows))
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'INACTIVE' if barrier=='inactive' else 'ACTIVE'))
    if barrier in {'contradiction','invalid_current'}:
        b=candidate.originals[1]; p=b.patches[0]
        b=replace(b,patches=(replace(p,anchor_uv=(62.,150.),capsule=(60.,150.,64.,150.,2.,2.)),)) if barrier=='contradiction' else replace(b,raster_id='wrong')
        candidate=replace(candidate,originals=(b,))
    else: candidate=replace(candidate,originals=())
    if barrier=='prior': selected=replace(selected,cameras=tuple(replace(c,prior_id='changed') for c in selected.cameras))
    result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.disposition!='accepted',result
    assert len(owner._persons)!=1 or all(owner.image_region_error(m.observation.stamp,context) for m in owner._members.values())


def test_image_regions_inactive_return_within_age_cannot_resurrect_confirmation():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2,3):
        selected,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'INACTIVE' if frame==2 else 'ACTIVE'))
        if frame>1: candidate=replace(candidate,originals=())
        result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
        if frame==1: assert result.disposition=='accepted'
    assert result.disposition=='observed' and len(owner._persons)==2


def test_image_regions_r19_unchecked_seed_cannot_enable_retention():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,candidate,rows,context,delivery=image_region_fixture()
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    # Explicit pre-upgrade fault injection: R19 epipolar-only seed had no
    # front-qualified confirmation scalar. It cannot bootstrap retention.
    for key,m in tuple(owner._members.items()): owner._members[key]=replace(m,image_region_front_confirmed=False)
    selected,candidate,rows,context,delivery=image_region_fixture(frame=2)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(replace(candidate,originals=()),projective_policy=selected,at_interval=context,delivery_context=delivery).disposition=='observed'
    assert len(owner._persons)==2


@pytest.mark.parametrize('case',['behind-both','behind-one'])
def test_image_regions_public_ledger_never_accepts_epipolar_only_behind_region(case):
    selected,candidate,rows,context,delivery=image_region_fixture()
    if case=='behind-both':
        o=candidate.originals[0]; p=o.patches[0]
        o=replace(o,patches=(replace(p,anchor_uv=(82.,100.),capsule=(80.,100.,84.,100.,2.,2.)),))
        candidate=replace(candidate,originals=(o,)+candidate.originals[1:])
    else:
        left,right=selected.cameras
        minus=tuple(tuple(-v for v in row) for row in left.peer_projection)
        selected=replace(selected,cameras=(replace(left,peer_projection=minus),replace(right,own_projection=minus)))
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.disposition=='observed' and len(owner._persons)==2


def recorded_image_shoulder_geometry():
    """Literal R7 target10 + frozen task prior from WorldR18; NOT metric truth."""
    k=(705.9833984375,436.0776672363281,705.9833984375,436.0776672363281,1.5798086872648676,1.5798086872648676)
    f=(547.74853515625,596.6199035644531,547.74853515625,596.6199035644531,24.271050437584606,24.271050437584606)
    kp=((-532.4041636823156,-327.605386229604,914.1001344489305,8022.378568520152),(31.365515388043132,-783.4257693864575,298.4229446963692,1282.7502260612378),(0.09758572731851303,-0.35836794954530016,0.9284661752387182,-0.7184407653953737))
    kq=((952.4091396010563,-290.3426322367182,-551.8237736194833,-10861.481541765612),(48.46193948445271,-823.8984100695418,-276.46483996167865,3437.1890022204607),(0.30064027645302954,-0.26254382545248556,-0.8548463677613963,4.116007560942405))
    fp=((809.0277251157554,-306.73526077576,-841.0237239486139,-1436.8989525179168),(-29.874185675556447,-873.5846723093254,-295.14267715274934,5582.528709380555),(0.0562450367276905,-0.2789092109348931,-0.9586689459345056,11.599428576634772))
    fq=((-273.66894428103427,-346.97029481195284,1087.4419430473297,-1239.820515066338),(121.12870085361494,-829.9188363047602,295.42666704677396,-480.3990936676146),(0.37232740779934714,-0.37837957944203415,0.9174424091144663,-6.261268053316554))
    return ((k,f,kp,kq),(f,k,fp,fq))


@pytest.mark.parametrize('direction',[0,1])
def test_image_regions_recorded10_curved_disc_front_witness_without_padding(direction):
    a,b,p,q=recorded_image_shoulder_geometry()[direction]
    original=(a,b,p,q)
    f=PersonAssociationLedger._region_fundamental(p,q)
    assert PersonAssociationLedger._region_feasible(f,a,b,cameras=(p,q)) is True
    assert (a,b,p,q)==original


@pytest.mark.parametrize('direction',[0,1])
def test_image_regions_recorded10_every_accepted_gradient_witness_is_inside_literal_discs(direction,monkeypatch):
    from math import hypot
    a,b,p,q=recorded_image_shoulder_geometry()[direction]
    original=PersonAssociationLedger._region_front_witness
    checked=[]
    def contained(left,right,uv,other):
        accepted=original(left,right,uv,other)
        if accepted:
            assert hypot(uv[0]-a[0],uv[1]-a[1])<=a[4]
            assert hypot(other[0]-b[0],other[1]-b[1])<=b[4]
            checked.append((uv,other))
        return accepted
    monkeypatch.setattr(PersonAssociationLedger,'_region_front_witness',staticmethod(contained))
    assert PersonAssociationLedger._region_feasible(PersonAssociationLedger._region_fundamental(p,q),a,b,cameras=(p,q)) is True
    assert checked


def test_image_regions_recorded10_distinct_seated_child_full_shoulder_is_still_incompatible():
    from math import hypot
    a,_,p,q=recorded_image_shoulder_geometry()[0]
    # Literal R7 F3:g2/frame300 shoulder joints; full untrimmed observed circle,
    # NOT the 5px depth ROI. Source-reviewed distinct seated child, not a label input.
    left=(911.6687622070312,684.5426025390625)
    right=(869.9124755859375,667.451171875)
    center=tuple((x+y)/2 for x,y in zip(left,right))
    radius=hypot(left[0]-right[0],left[1]-right[1])/2
    b=center+center+(radius,radius)
    assert PersonAssociationLedger._region_feasible(PersonAssociationLedger._region_fundamental(p,q),a,b,cameras=(p,q)) is False


def image_diagnostic_observe(owner,frame,*,mode='fresh',wall_step=.05,competitor=None,live=False):
    selected,candidate,rows,context,delivery=image_region_fixture(frame=frame,competitor=competitor)
    mono=100+(frame-1)*wall_step
    delivery=replace(delivery,evaluation_monotonic_s=mono,ingresses=tuple(CurrentVisualIngress(r.stamp,mono) for r in rows))
    candidate=replace(candidate,originals=tuple(replace(o,ingress_monotonic_s=mono) for o in candidate.originals))
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    if mode=='missing':candidate=replace(candidate,originals=())
    elif mode=='invalid':candidate=replace(candidate,originals=(replace(candidate.originals[0],raster_id='wrong'),)+candidate.originals[1:])
    elif mode=='contradiction':
        o=candidate.originals[1];p=o.patches[0]
        candidate=replace(candidate,originals=(candidate.originals[0],replace(o,patches=(replace(p,anchor_uv=(62.,150.),capsule=(60.,150.,64.,150.,2.,2.)),))))
    return owner.join_estimated_image_regions(candidate,projective_policy=selected,at_interval=context,delivery_context=delivery,
        live_track_continuity=live)


def live_image_step(owner,frame,*,mode='missing',competitor=None,wall=None,native_state='ACTIVE',prior_change=False,houses=None,live=True,born_change=False,current_lag=0.):
    """R13 source/actual ingress clocks are independent; no original is retimed."""
    selected,candidate,rows,_,_=image_region_fixture(frame=frame,competitor=competitor)
    rows=tuple(replace(r,stamp=replace(r.stamp,acquired_at_s=10+(frame-1)*.033366,
        media_time_s=5+(frame-1)*.033366)) for r in rows)
    if houses is not None: rows=tuple(replace(r,household=h) for r,h in zip(rows,houses))
    if born_change: rows=(replace(rows[0],stamp=replace(rows[0].stamp,lifecycle_started_frame=frame)),)+rows[1:]
    wall=100+(frame-1)*.081476 if wall is None else wall
    delivery=visual_delivery(rows,sequence=frame,evaluation=wall)
    if current_lag: delivery=replace(delivery,ingresses=tuple(replace(i,ingress_monotonic_s=wall-current_lag) for i in delivery.ingresses))
    if frame==6:
        delivery=replace(delivery,ingresses=tuple(replace(i,ingress_monotonic_s=wall-age)
            for i,age in zip(delivery.ingresses,(.02164,.04904,0.))))
    originals=tuple(replace(o,stamp=r.stamp,ingress_monotonic_s=delivery.ingresses[i].ingress_monotonic_s)
        for i,(o,r) in enumerate(zip(candidate.originals,rows)))
    if mode=='missing': originals=()
    elif mode=='invalid': originals=(replace(originals[0],raster_id='wrong'),)+originals[1:]
    elif mode=='contradiction':
        o=originals[1];p=o.patches[0]
        originals=(originals[0],replace(o,patches=(replace(p,anchor_uv=(62.,150.),capsule=(60.,150.,64.,150.,2.,2.)),)))+originals[2:]
    if prior_change: selected=replace(selected,cameras=tuple(replace(c,prior_id=c.prior_id+'changed') for c in selected.cameras))
    candidate=replace(candidate,cohort=tuple(r.stamp for r in rows),originals=originals)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,native_state))
    context=spatial_context(rows)
    result=owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery,
        **({'live_track_continuity':live,'projective_policy':selected} if isinstance(owner,PersonAssociationLedger) else {}))
    return result,rows,context,delivery


def test_image_regions_live_native_r13_wall_expired_pose_current_tracks_hold_immutable_seed():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    result,rows,context,_=live_image_step(owner,1,mode='fresh')
    key=result.person_key
    seed=tuple(owner._members[r.stamp.member].image_region_original for r in rows)
    for frame in range(2,21):
        result,rows,context,delivery=live_image_step(owner,frame)
        assert result.reason=='estimated_live_track_continuity',result
        assert result.person_key==key and len(owner._persons)==1
        assert all(owner.image_region_error(r.stamp,context) is None for r in rows)
        assert tuple(owner._members[r.stamp.member].image_region_original for r in rows)==seed
        assert all(owner._members[r.stamp.member].visual_confirmed_frame==1 for r in rows)
        if frame==6:
            assert delivery.evaluation_monotonic_s-seed[1].ingress_monotonic_s==pytest.approx(.40738)
            assert rows[0].stamp.acquired_at_s-seed[0].stamp.acquired_at_s==pytest.approx(.16683)


@pytest.mark.parametrize('mode',['missing','invalid'])
def test_image_regions_live_never_bootstraps_without_front_seed(mode):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    result,_,_,_=live_image_step(owner,1,mode=mode)
    assert result.disposition=='observed' and len(owner._persons)==2


@pytest.mark.parametrize('barrier',['invalid','contradiction','new_unknown','alternate','delivery_gap','native_gap','prior','inactive','absence','reset','birth','stale_delivery'])
def test_image_regions_live_revokes_actual_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,rows,_,_=live_image_step(owner,1,mode='fresh')
    if barrier=='absence': owner.withdraw(removal(rows[0]))
    elif barrier=='reset': owner.reset_source(replace(SOURCES[1],source_epoch=1))
    frame=20 if barrier=='native_gap' else 2
    result,_,_,_=live_image_step(owner,frame,
        mode=barrier if barrier in {'invalid','contradiction'} else 'fresh' if barrier=='alternate' else 'missing',
        competitor='missing' if barrier=='new_unknown' else 'supported' if barrier=='alternate' else None,
        wall=101 if barrier=='delivery_gap' else None,native_state='INACTIVE' if barrier=='inactive' else 'ACTIVE',
        prior_change=barrier=='prior',born_change=barrier=='birth',current_lag=.36 if barrier=='stale_delivery' else 0.)
    assert result.reason!='estimated_live_track_continuity'
    assert not any(m.association_basis=='estimated_live_track_continuity' for m in owner._members.values())


def test_image_regions_live_known_competitor_pose_hole_is_not_new_unknown_competitor():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    first,_,_,_=live_image_step(owner,1,mode='fresh',competitor='incompatible')
    for frame in range(2,9):
        result,_,_,_=live_image_step(owner,frame,competitor='incompatible')
        assert result.reason=='estimated_live_track_continuity'
        assert result.person_key==first.person_key and len(owner._persons)==2


def test_image_regions_live_default_false_reproduces_r13_old_pose_wall_expiry_split():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in range(1,7): result,_,_,_=live_image_step(owner,frame,mode='fresh' if frame==1 else 'missing',live=False)
    assert len(owner._persons)==2 and result.reason=='image_region_unavailable_or_ambiguous'


@pytest.mark.parametrize('value',[1,0,'true',None])
def test_image_regions_live_requires_exact_boolean_without_mutation(value):
    selected,proof,rows,context,delivery=image_region_fixture()
    owner=PersonAssociationLedger('run-a',spatial_policy())
    revision=owner.authority_revision
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
        delivery_context=delivery,live_track_continuity=value)
    assert result.reason=='image_region_live_track_boolean_required' and owner.authority_revision==revision


def spatial_shadow_update(last, *, frame=9, time=10.4, media=5.4, token='actual-retained-unmeasured'):
    from reid.mv3dt_person_association import SpatialLifecycleUpdate
    return SpatialLifecycleUpdate(token,last.stamp,frame,time,last.stamp.clock_domain,media)


def test_image_regions_lifecycle_update_preserves_original_native_truth_and_literal_roster():
    selected,proof,rows,context,delivery=image_region_fixture(competitor='incompatible')
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery)
    before=owner._members[rows[2].stamp.member]
    update=spatial_shadow_update(rows[2])
    revision=owner.authority_revision
    result=owner.update_spatial_lifecycle(update)
    assert result.disposition=='observed' and result.reason=='retained_unmeasured'
    shadow=owner._members[rows[2].stamp.member]
    assert shadow.observation is before.observation and shadow.spatial_lifecycle is before.spatial_lifecycle
    assert shadow.spatial_lifecycle.native_state=='ACTIVE' and not shadow.image_region_participant
    # The other OLD ACTIVE stamps are still literal participants, not age-filtered.
    assert owner.image_region_active_cohort()==tuple(sorted((r.stamp for r in rows[:2]),key=lambda s:s.member))
    assert owner.authority_revision==revision+1
    assert owner.update_spatial_lifecycle(update).reason=='retained_unmeasured'
    assert owner.authority_revision==revision+1
    with pytest.raises(FrozenInstanceError): update.at_frame_id=10
    assert owner.observe_spatial(rows[2],before.spatial_lifecycle).reason=='spatial_lifecycle_reactivation_requires_new_observation'


@pytest.mark.parametrize('mutation',['old_frame','old_time','old_media','clock','classification','last_stamp','receipt_alias','replay','epoch'])
def test_image_regions_lifecycle_update_rejects_stale_changed_receipts_atomically(mutation):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,_,rows,_,_=image_region_fixture()
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    update=spatial_shadow_update(rows[1]); owner.update_spatial_lifecycle(update)
    bad=replace(update,receipt_id='later',at_frame_id=10,at_acquired_at_s=10.45,at_media_time_s=5.45)
    if mutation=='old_frame': bad=replace(bad,at_frame_id=8)
    elif mutation=='old_time': bad=replace(bad,at_acquired_at_s=10.3)
    elif mutation=='old_media': bad=replace(bad,at_media_time_s=5.3)
    elif mutation=='clock': bad=replace(bad,clock_domain='invented-clock')
    elif mutation=='classification': bad=replace(bad,classification='INACTIVE')
    elif mutation=='last_stamp': bad=replace(bad,last_observation=replace(bad.last_observation,observation_id='invented'))
    elif mutation=='receipt_alias': bad=replace(bad,receipt_id=update.receipt_id)
    elif mutation=='replay': bad=replace(update,receipt_id='changed-same-frame')
    else: bad=replace(bad,last_observation=replace(bad.last_observation,member=replace(bad.last_observation.member,source_epoch=1)))
    revision=owner.authority_revision; before=owner.snapshot(now(10.4))
    assert owner.update_spatial_lifecycle(bad).disposition=='association_unresolved'
    assert owner.authority_revision==revision and owner.snapshot(now(10.4))==before


def test_image_regions_lifecycle_split_capacity_is_atomic():
    owner=PersonAssociationLedger('run-a',replace(spatial_policy(),max_entities=2,max_history=4))
    live_image_step(owner,1,mode='fresh')
    _,_,rows,_,_=image_region_fixture(frame=2,competitor='incompatible')
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    revision=owner.authority_revision; before=owner.snapshot(now(10.05))
    update=spatial_shadow_update(rows[1],frame=2,time=10.05,media=5.05)
    assert owner.update_spatial_lifecycle(update).reason=='spatial_lifecycle_split_capacity'
    assert owner.authority_revision==revision and owner.snapshot(now(10.05))==before


@pytest.mark.parametrize('mutation',['bool_frame','nan_time','missing_media','oversized_id','wrong_type'])
def test_image_regions_lifecycle_update_malformed_fields_never_mutate(mutation):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,_,rows,_,_=image_region_fixture()
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    update=spatial_shadow_update(rows[1])
    if mutation=='bool_frame': update=replace(update,at_frame_id=True)
    elif mutation=='nan_time': update=replace(update,at_acquired_at_s=float('nan'))
    elif mutation=='missing_media': update=replace(update,at_media_time_s=None)
    elif mutation=='oversized_id': update=replace(update,receipt_id='x'*257)
    else: update=object()
    revision=owner.authority_revision
    assert owner.update_spatial_lifecycle(update).reason=='spatial_lifecycle_update_binding'
    assert owner.authority_revision==revision and len(owner.image_region_active_cohort())==2


@pytest.mark.parametrize('opt_in',[False,True])
def test_image_regions_source_local_reactivation_is_explicit_and_keeps_seed_revoked(opt_in):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,_,first,_,_=image_region_fixture()
    original=first[1]
    owner.observe_spatial(original,SpatialLifecycleReceipt('first-real',original.stamp,'ACTIVE'))
    key=owner._members[original.stamp.member].person_key
    owner.update_spatial_lifecycle(spatial_shadow_update(original,frame=2,time=10.05,media=5.05))
    _,_,rows,_,_=image_region_fixture(frame=3); current=rows[1]
    result=owner.observe_spatial(current,SpatialLifecycleReceipt('later-real',current.stamp,'ACTIVE'),
        image_source_local_reactivation=opt_in)
    assert result.reason=='local_observation'
    member=owner._members[current.stamp.member]
    assert member.person_key==key and member.association_basis=='source_local'
    assert member.spatial_join_required is (not opt_in)
    assert not member.image_region_front_confirmed and member.image_region_native_seed==()
    assert member.spatial_lifecycle_update is None and member.image_region_participant
    assert owner.spatial_member_error(current.stamp,spatial_context((current,)))==(
        None if opt_in else 'current_rejoin_proof_required')


@pytest.mark.parametrize('barrier',['same_event','source_rewind','birth','epoch','inactive'])
def test_image_regions_source_local_reactivation_does_not_bypass_native_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,_,rows,_,_=image_region_fixture(); old=rows[1]
    owner.observe_spatial(old,SpatialLifecycleReceipt('first-real',old.stamp,'ACTIVE'))
    owner.update_spatial_lifecycle(spatial_shadow_update(old,frame=2,time=10.05,media=5.05))
    _,_,rows,_,_=image_region_fixture(frame=3); row=rows[1]
    if barrier=='same_event': row=replace(row,stamp=replace(row.stamp,frame_id=2))
    elif barrier=='source_rewind': row=replace(row,stamp=replace(row.stamp,acquired_at_s=10.))
    elif barrier=='birth': row=replace(row,stamp=replace(row.stamp,lifecycle_started_frame=2))
    elif barrier=='epoch': row=replace(row,stamp=replace(row.stamp,member=replace(row.stamp.member,source_epoch=1)))
    result=owner.observe_spatial(row,SpatialLifecycleReceipt('real-current',row.stamp,'INACTIVE' if barrier=='inactive' else 'ACTIVE'),
        image_source_local_reactivation=True)
    assert result.disposition!='observed' if barrier!='inactive' else True
    member=owner._members[old.stamp.member]
    assert member.spatial_join_required and not member.image_region_front_confirmed and not member.image_region_native_seed


@pytest.mark.parametrize('bad',[1,None,'true'])
def test_image_regions_source_local_opt_in_rejects_nonboolean_without_observation(bad):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,_,rows,_,_=image_region_fixture(); row=rows[1]
    revision=owner.authority_revision
    assert owner.observe_spatial(row,SpatialLifecycleReceipt('native',row.stamp,'ACTIVE'),
        image_source_local_reactivation=bad).reason=='image_source_local_boolean_required'
    assert owner.authority_revision==revision and not owner.image_region_active_cohort()


def test_image_regions_r17_true_pair_loses_uniqueness_when_known_current_competitor_pose_is_unknown():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,proof,rows,context,delivery=image_region_fixture(competitor='incompatible')
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
        delivery_context=delivery).disposition=='accepted'
    selected,proof,rows,context,delivery=image_region_fixture(frame=2,competitor='missing')
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
        delivery_context=delivery)
    assert result.reason=='image_region_unavailable_or_ambiguous' and result.corrected_members
    transition=owner.image_region_decision_diagnostics()['transitions'][-1]
    true_pair=next(p for p in transition['pairs'] if p['right']['raw_id']==8)
    competitor=next(p for p in transition['pairs'] if p['right']['raw_id']==9)
    assert true_pair['state'] is True and not true_pair['unique'] and not true_pair['supported']
    assert competitor['state'] is None and competitor['possible']
    assert all(v is True for c in true_pair['cues'] for v in (c['forward'],c['reverse']))


@pytest.mark.parametrize('barrier',['unseeded','self_false','alternative_true','self_missing','one_cue','delivery_gap','native_gap','prior','shadow','birth','epoch','inactive','stale_delivery'])
def test_image_regions_current_pair_hypothesis_never_mints_or_ignores_positive_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,proof,old,context,delivery=two_cue_image_region_fixture()
    if barrier!='unseeded':
        for r in old: owner.observe_spatial(r,SpatialLifecycleReceipt('seed:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
            delivery_context=delivery,live_track_continuity=True)
    if barrier=='shadow':owner.update_spatial_lifecycle(spatial_shadow_update(old[1],frame=2,time=10.05,media=5.05))
    elif barrier=='epoch':owner.reset_source(replace(SOURCES[1],source_epoch=1))
    frame=10 if barrier=='native_gap' else 3 if barrier=='shadow' else 2
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,
        competitor='supported' if barrier=='alternative_true' else 'missing')
    if barrier=='self_false':
        o=proof.originals[1]
        def moved(p):
            a,b,c,d,r,s=p.capsule
            return replace(p,anchor_uv=(p.anchor_uv[0],p.anchor_uv[1]+50),capsule=(a,b+50,c,d+50,r,s))
        proof=replace(proof,originals=(proof.originals[0],replace(o,patches=tuple(moved(p) for p in o.patches))))
    elif barrier=='self_missing':proof=replace(proof,originals=proof.originals[1:])
    elif barrier=='one_cue':proof=replace(proof,originals=tuple(replace(o,patches=o.patches[:1]) for o in proof.originals))
    elif barrier=='delivery_gap':
        delivery=replace(delivery,evaluation_monotonic_s=101.,ingresses=tuple(replace(i,ingress_monotonic_s=101.) for i in delivery.ingresses))
        proof=replace(proof,originals=tuple(replace(o,ingress_monotonic_s=101.) for o in proof.originals))
    elif barrier=='prior':selected=replace(selected,cameras=tuple(replace(c,prior_id=c.prior_id+'changed') for c in selected.cameras))
    elif barrier=='birth':
        row=replace(rows[0],stamp=replace(rows[0].stamp,lifecycle_started_frame=2))
        rows=(row,)+rows[1:];proof=replace(proof,cohort=tuple(r.stamp for r in rows))
    elif barrier=='stale_delivery':
        delivery=replace(delivery,ingresses=tuple(replace(i,ingress_monotonic_s=delivery.evaluation_monotonic_s-.352311) for i in delivery.ingresses))
        proof=replace(proof,originals=tuple(replace(o,ingress_monotonic_s=delivery.evaluation_monotonic_s-.352311) for o in proof.originals))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('real:'+r.stamp.observation_id,r.stamp,'INACTIVE' if barrier=='inactive' and r==rows[0] else 'ACTIVE'))
    if barrier=='inactive':proof=replace(proof,cohort=owner.image_region_active_cohort(),originals=tuple(o for o in proof.originals if o.stamp.member!=rows[0].stamp.member))
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
        delivery_context=delivery,live_track_continuity=True)
    if barrier=='alternative_true':
        assert result.reason=='estimated_live_track_continuity' and not result.corrected_members
        assert owner.image_region_error(rows[0].stamp,context)=='image_region_ambiguous_member'
        assert owner.image_region_error(rows[1].stamp,context) is None
    else:
        assert result.reason!='estimated_live_track_continuity'
        assert not any(m.image_region_pair_hypothesis for m in owner._members.values())


def test_image_regions_current_pair_hypothesis_later_missing_self_keeps_native_pair_not_pose_grant():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    key=None; seed=None
    for frame in range(1,12):
        selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor=None if frame==1 else 'missing')
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,
            delivery_context=delivery,live_track_continuity=True)
        if frame==1:
            key=result.person_key;seed=tuple(owner._members[r.stamp.member].image_region_original for r in rows)
        else:
            assert result.reason=='estimated_live_track_continuity' and result.person_key==key
            assert all(owner._members[r.stamp.member].image_region_pair_hypothesis for r in rows[:2])
            assert tuple(owner._members[r.stamp.member].image_region_original for r in rows[:2])==seed
            assert all(owner.image_region_error(r.stamp,context) is None for r in rows[:2])
    # Explicit R35 followthrough: after a fresh two-cue hypothesis, missing
    # pose uses uninterrupted PAIR native continuity, not an old pose as new.
    for frame in (12,13):
        selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor='missing')
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(replace(proof,originals=()),projective_policy=selected,
            at_interval=context,delivery_context=delivery,live_track_continuity=True)
        assert result.reason=='estimated_live_track_continuity' and not result.corrected_members
        assert result.person_key==key
        assert all(owner._members[r.stamp.member].image_region_pair_hypothesis for r in rows[:2])
        assert tuple(owner._members[r.stamp.member].image_region_original for r in rows[:2])==seed


@pytest.mark.parametrize('barrier',['self_false','alternative_true','native_gap','inactive','absence','prior'])
def test_image_regions_shadow_membership_retention_revokes_on_real_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,proof,old,context,delivery=two_cue_image_region_fixture()
    for r in old:owner.observe_spatial(r,SpatialLifecycleReceipt('seed:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=True)
    seed=owner._members[old[0].stamp.member].person_key
    result=owner.update_spatial_lifecycle(spatial_shadow_update(old[0],frame=2,time=10.05,media=5.05),retain_image_membership=True)
    assert not result.corrected_members and all(m.image_region_shadow_membership for m in owner._members.values())
    if barrier=='absence':owner.withdraw(removal(old[0],frame=3,time=10.1,media=5.1))
    frame=12 if barrier=='native_gap' else 3
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor='supported' if barrier=='alternative_true' else None)
    if barrier=='self_false':
        o=proof.originals[1]
        def moved(p):
            a,b,c,d,r,s=p.capsule
            return replace(p,anchor_uv=(p.anchor_uv[0],p.anchor_uv[1]+50),capsule=(a,b+50,c,d+50,r,s))
        proof=replace(proof,originals=(proof.originals[0],replace(o,patches=tuple(moved(p) for p in o.patches))))
    if barrier=='prior':selected=replace(selected,cameras=tuple(replace(c,prior_id=c.prior_id+'changed') for c in selected.cameras))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('real:'+r.stamp.observation_id,r.stamp,
        'INACTIVE' if barrier=='inactive' and r==rows[0] else 'ACTIVE'),retain_image_membership=True)
    if barrier=='inactive':proof=replace(proof,cohort=owner.image_region_active_cohort(),originals=proof.originals[1:])
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=True)
    assert not any(m.image_region_shadow_membership and owner.image_region_error(m.observation.stamp,context) is None
        for m in owner._members.values())


@pytest.mark.parametrize('value',[1,None,'true'])
def test_image_regions_shadow_retention_exact_boolean_only(value):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,_,rows,_,_=image_region_fixture()
    assert owner.observe_spatial(rows[0],SpatialLifecycleReceipt('native',rows[0].stamp,'ACTIVE'),
        retain_image_membership=value).reason=='image_region_live_track_boolean_required'
    assert owner.update_spatial_lifecycle(spatial_shadow_update(rows[0]),retain_image_membership=value).reason=='image_region_live_track_boolean_required'


@pytest.mark.parametrize('barrier',['stale_delivery','delivery_gap','native_gap'])
def test_image_regions_established_pair_missing_self_keeps_all_native_freshness_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor=None if frame==1 else 'missing')
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=True)
    assert all(owner._members[r.stamp.member].image_region_pair_hypothesis for r in rows[:2])
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=12 if barrier=='native_gap' else 3,competitor='missing')
    if barrier=='delivery_gap':delivery=replace(delivery,evaluation_monotonic_s=101.,ingresses=tuple(replace(i,ingress_monotonic_s=101.) for i in delivery.ingresses))
    if barrier=='stale_delivery':delivery=replace(delivery,ingresses=tuple(replace(i,ingress_monotonic_s=delivery.evaluation_monotonic_s-.352311) for i in delivery.ingresses))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('later:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(replace(proof,originals=()),projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=True)
    assert result.reason!='estimated_live_track_continuity'


@pytest.mark.parametrize('partial',['shoulder','torso','shoulder_unknown'])
def test_image_regions_established_pair_present_partial_is_not_stricter_than_missing(partial):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in (1,2):
        selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor=None if frame==1 else 'missing')
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=True)
    key=result.person_key
    originals=tuple(owner._members[r.stamp.member].image_region_original for r in rows[:2])
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=3,competitor='missing')
    anatomy='pose_torso_support' if partial=='torso' else 'pose_shoulder_support'
    proof=replace(proof,originals=tuple(replace(o,patches=tuple(p for p in o.patches if p.anatomy==anatomy)) for o in proof.originals))
    if partial=='shoulder_unknown':proof=replace(proof,originals=proof.originals[:1])
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('next:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=True)
    assert result.reason=='estimated_live_track_continuity' and result.person_key==key and not result.corrected_members
    assert tuple(owner._members[r.stamp.member].image_region_original for r in rows[:2])==originals


@pytest.mark.parametrize('barrier',['unseeded','default_false','self_false','alternative_true','invalid','native_gap','delivery_gap','reset'])
def test_image_regions_present_partial_cannot_mint_or_bypass_real_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    if barrier!='unseeded':
        for frame in (1,2):
            selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor=None if frame==1 else 'missing')
            for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
            owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,
                live_track_continuity=barrier!='default_false')
    if barrier=='reset':owner.reset_source(replace(SOURCES[1],source_epoch=1))
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=12 if barrier=='native_gap' else 3,
        competitor='supported' if barrier=='alternative_true' else 'missing')
    proof=replace(proof,originals=tuple(replace(o,patches=tuple(p for p in o.patches if p.anatomy=='pose_shoulder_support')) for o in proof.originals))
    if barrier=='self_false':
        o=proof.originals[1];p=o.patches[0];a,b,c,d,r,s=p.capsule
        proof=replace(proof,originals=(proof.originals[0],replace(o,patches=(replace(p,anchor_uv=(p.anchor_uv[0],p.anchor_uv[1]+50),capsule=(a,b+50,c,d+50,r,s)),))))
    if barrier=='invalid':proof=replace(proof,originals=(replace(proof.originals[0],body_evidence_id=''),)+proof.originals[1:])
    if barrier=='delivery_gap':
        delivery=replace(delivery,evaluation_monotonic_s=101.,ingresses=tuple(replace(i,ingress_monotonic_s=101.) for i in delivery.ingresses))
        proof=replace(proof,originals=tuple(replace(o,ingress_monotonic_s=101.) for o in proof.originals))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('later:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,
        live_track_continuity=barrier!='default_false')
    if barrier=='alternative_true':
        assert result.reason=='estimated_live_track_continuity'
        assert owner.image_region_error(rows[0].stamp,context)=='image_region_ambiguous_member'
    else:assert result.reason!='estimated_live_track_continuity'


def image_vote_competition_fixture(frame=1,*,crowded=False):
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,
        competitor='supported' if crowded else 'incompatible')
    extra=replace(rows[0],stamp=replace(rows[0].stamp,member=replace(rows[0].stamp.member,raw_id=10),
        observation_id='extra-kitchen:'+rows[0].stamp.observation_id))
    original=replace(proof.originals[0],stamp=extra.stamp,body_evidence_id='extra-body:'+extra.stamp.observation_id,
        patches=proof.originals[2].patches)
    rows=rows+(extra,)
    return selected,replace(proof,cohort=tuple(r.stamp for r in rows),originals=proof.originals+(original,)),rows,spatial_context(rows),visual_delivery(rows,sequence=frame,evaluation=100+(frame-1)*.05)


def floor_crossed_vote_quarantine_fixture(owner, *, both=False, native_gap=False, houses=False, before_current=None):
    """Real owner mutations: unique121 -> shadow131/return132 -> delivery219."""
    def step(frame, wall, *, bodies=True, crowded=False):
        selected,p,rows,_,_= (image_vote_competition_fixture(frame,crowded=crowded) if both
            else two_cue_image_region_fixture(frame=frame,competitor='supported' if crowded else 'incompatible'))
        rows=tuple(replace(r,stamp=replace(r.stamp,acquired_at_s=10+(frame-121)*.001,
            media_time_s=5+(frame-121)*.001)) for r in rows)
        if native_gap and frame==245:
            rows=tuple(replace(r,stamp=replace(r.stamp,acquired_at_s=11.,media_time_s=6.)) for r in rows)
        if houses:
            rows=tuple(replace(r,household=HouseholdDescriptor('visitor','claim-'+str(i),1000+i,1)) for i,r in enumerate(rows))
        p=replace(p,cohort=tuple(r.stamp for r in rows),originals=tuple(
            replace(o,stamp=rows[i].stamp,ingress_monotonic_s=wall) for i,o in enumerate(p.originals)) if bodies else (),
            owner_availability=tuple(image_owner_capsule(r,wall) for r in rows))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'),
            **({'retain_image_membership':True} if isinstance(owner,PersonAssociationLedger) else {}))
        return selected,p,rows,spatial_context(rows),visual_delivery(rows,sequence=frame,evaluation=wall)
    def join(p,c,d):
        return owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d,
            **({'projective_policy':selected,'live_track_continuity':True} if isinstance(owner,PersonAssociationLedger) else {}))
    ledger=owner if isinstance(owner,PersonAssociationLedger) else owner.ledger
    selected,p,old,c,d=step(121,100.)
    assert join(p,c,d).disposition=='accepted'
    key=ledger._members[old[0].stamp.member].person_key
    for shadow in (tuple(r for r in old if r.stamp.member.camera_id=='kitchen') if both else (old[0],)):
        owner.update_spatial_lifecycle(spatial_shadow_update(shadow,frame=131,time=10.010,media=5.010),
            **({'retain_image_membership':True} if isinstance(owner,PersonAssociationLedger) else {}))
    for frame,wall in ((132,100.1),(219,100.5)):
        # Native time is uninterrupted even though original delivery crosses .35.
        selected,p,rows,c,d=step(frame,wall,bodies=False)
        result=join(p,c,d)
        assert ledger._members[old[0].stamp.member].person_key==key,result
    if before_current:before_current(rows,c)
    selected,p,rows,c,d=step(245,100.6,crowded=True)
    return key,selected,p,rows,c,d


def test_image_regions_floor_crossed_quarantine_keeps_seed_key_and_only_unambiguous_vote():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    key,selected,p,rows,c,d=floor_crossed_vote_quarantine_fixture(owner)
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert len(owner._persons)==2,result
    assert {owner._members[r.stamp.member].person_key for r in rows[:2]}=={key}
    assert owner.image_region_error(rows[0].stamp,c)=='image_region_ambiguous_member'
    assert owner.image_region_error(rows[1].stamp,c) is None
    assert all(owner._members[r.stamp.member].image_region_original.stamp.frame_id==121 for r in rows[:2])
    assert owner._members[rows[0].stamp.member].projective_native_floor==131
    assert owner._members[rows[1].stamp.member].projective_delivery_floor==219


@pytest.mark.parametrize('barrier',['missing_caps','expired','not_retained','not_active','shadow','terminated',
    'recovering','stop','run_close','reset','absence','birth','appearance','native_gap','rejoin','unseeded','default_false'])
def test_image_regions_floor_crossed_quarantine_custody_barriers(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    key,selected,p,rows,c,d=floor_crossed_vote_quarantine_fixture(owner,native_gap=barrier=='native_gap')
    if barrier=='missing_caps':p=replace(p,owner_availability=p.owner_availability[:1])
    elif barrier in {'expired','not_retained','not_active','shadow','terminated','recovering'}:
        cap=p.owner_availability[1];k=rows[1].stamp.member
        changes={'expired':{'last_progress_age_s':20.},'not_retained':{'retained_members':(),'active_members':()},
            'not_active':{'active_members':()},'shadow':{'shadow_members':(k,)},
            'terminated':{'retained_members':(),'active_members':(),'terminated_members':(k,)},
            'recovering':{'availability_status':'recovering'}}
        p=replace(p,owner_availability=(p.owner_availability[0],replace(cap,**changes[barrier]))+p.owner_availability[2:])
    elif barrier=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    elif barrier=='run_close':owner.close_image_region_run()
    elif barrier=='reset':owner.reset_source(replace(SOURCES[2],source_epoch=1))
    elif barrier=='absence':owner.withdraw(removal(rows[1],frame=245,time=rows[1].stamp.acquired_at_s,media=rows[1].stamp.media_time_s))
    elif barrier in {'birth','appearance','rejoin','unseeded'}:
        k=rows[0].stamp.member;m=owner._members[k]
        changes={'birth':{'image_region_native_seed':tuple(replace(s,lifecycle_started_frame=2) for s in m.image_region_native_seed)},
            'appearance':{'appearance_after_frame':121},'rejoin':{'spatial_join_required':True},
            'unseeded':{'image_region_native_seed':(),'image_region_front_confirmed':False}}
        owner._members[k]=replace(m,**changes[barrier])
    result=owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
        live_track_continuity=barrier!='default_false')
    assert not any(m.image_region_vote_blocked_stamp is not None for m in owner._members.values()),result


def test_image_regions_floor_crossed_quarantine_both_ambiguous_no_votes():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    _,selected,p,rows,c,d=floor_crossed_vote_quarantine_fixture(owner,both=True)
    owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,live_track_continuity=True)
    assert len(owner._persons)==2
    assert all(owner.image_region_error(r.stamp,c)=='image_region_ambiguous_member' for r in rows)


def test_image_regions_floor_crossed_quarantine_public_current_grant_and_queued_revision():
    from noesis.identity_association_service import IdentityAssociationService
    from tests.test_identity_association_service import spatial_reference
    selected,*_=two_cue_image_region_fixture()
    service=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    queued=[]
    def before(rows,c):
        dto=next(x for x in service.spatial_memberships(at_interval=c) if len(x.members)==2)
        queued.append(service.prepare_spatial_physical_member(dto,spatial_reference(rows[1],c),at_interval=c).reservation)
        assert queued[-1] is not None
    key,_,p,rows,c,d=floor_crossed_vote_quarantine_fixture(service,houses=True,before_current=before)
    service.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    dtos=service.spatial_memberships(at_interval=c);assert len(dtos)==2
    dto=next(x for x in dtos if x.physical_key==key)
    assert dto.household_conflict.stable_ids==(1000,1001)
    assert service.prepare_spatial_member(dto,spatial_reference(rows[1],c),at_interval=c).reason=='settled_household_claim_conflict'
    assert service.prepare_spatial_physical_member(dto,spatial_reference(rows[0],c),at_interval=c).reason=='image_region_ambiguous_member'
    current=service.prepare_spatial_physical_member(dto,spatial_reference(rows[1],c),at_interval=c).reservation
    assert current is not None
    assert service.commit_spatial_physical_member(current,at_interval=c,consumer=lambda *_:'fresh-F').disposition=='committed'
    assert service.commit_spatial_physical_member(queued[0],at_interval=c,consumer=lambda *_:pytest.fail('old revision')).disposition=='denied'
    # Use an unconsumed reference on the fresh distinct person to check that
    # even an otherwise identical ordered classification invalidates callbacks.
    distinct=next(x for x in service.spatial_memberships(at_interval=c) if x.physical_key!=key)
    intent=service.prepare_spatial_member(distinct,spatial_reference(rows[2],c),at_interval=c).reservation
    assert intent is not None
    revision=service.ledger.authority_revision
    service.join_estimated_image_regions(replace(p,proof_id='reclassified-current-quarantine'),at_interval=c,
        delivery_context=replace(d,sequence=246))
    assert service.ledger.authority_revision>revision
    assert service.commit_spatial_member(intent,at_interval=c,consumer=lambda *_:pytest.fail('stale classification')).disposition=='denied'
    service.ledger._visual_evaluation=120.6
    assert service.ledger.image_region_error(rows[1].stamp,c)=='image_region_current_support_required'


def test_image_regions_vote_quarantine_both_ambiguous_keep_custody_zero_votes_and_bounded_diagnostic():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    keys=None
    for frame in range(1,13):
        selected,proof,rows,context,delivery=image_vote_competition_fixture(frame,crowded=frame>1)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=True)
        current=set(owner._persons)
        if keys is None:keys=current;assert len(keys)==2
        else:
            assert current==keys and not result.corrected_members and not result.retired_keys
            assert all(owner.image_region_error(r.stamp,context)=='image_region_ambiguous_member' for r in rows)
            assert all(owner._members[r.stamp.member].image_region_vote_blocked_stamp==r.stamp for r in rows)
            assert all(owner._members[r.stamp.member].image_region_original.stamp.frame_id==1 for r in rows)
    view=owner.image_region_decision_diagnostics()
    assert len(view['transitions'])==8 and view['evicted']>0
    record=view['transitions'][-1]
    assert all(m['vote_block_after']['frame_id']==12 and not m['vote_pending_after'] for m in record['members'])
    assert all(p['state'] is True and not p['unique'] for p in record['pairs'])
    record['members'][0]['vote_block_after']['frame_id']=9999
    assert owner.image_region_decision_diagnostics()['transitions'][-1]['members'][0]['vote_block_after']['frame_id']==12


@pytest.mark.parametrize('barrier',['unseeded','default_false','self_false','native_gap','delivery_gap','stale_delivery','reset','absence','inactive'])
def test_image_regions_vote_quarantine_does_not_bypass_original_membership_or_freshness(barrier):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    selected,proof,old,context,delivery=two_cue_image_region_fixture(competitor='incompatible')
    if barrier!='unseeded':
        for r in old:owner.observe_spatial(r,SpatialLifecycleReceipt('seed:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=barrier!='default_false')
    if barrier=='reset':owner.reset_source(replace(SOURCES[1],source_epoch=1))
    if barrier=='absence':owner.withdraw(removal(old[0],frame=2,time=10.05,media=5.05))
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=12 if barrier=='native_gap' else 3,competitor='supported')
    if barrier=='self_false':
        o=proof.originals[1]
        def moved(p):
            a,b,c,d,r,s=p.capsule
            return replace(p,anchor_uv=(p.anchor_uv[0],p.anchor_uv[1]+50),capsule=(a,b+50,c,d+50,r,s))
        proof=replace(proof,originals=(proof.originals[0],replace(o,patches=tuple(moved(p) for p in o.patches)),proof.originals[2]))
    if barrier=='delivery_gap':
        delivery=replace(delivery,evaluation_monotonic_s=101.,ingresses=tuple(replace(i,ingress_monotonic_s=101.) for i in delivery.ingresses))
        proof=replace(proof,originals=tuple(replace(o,ingress_monotonic_s=101.) for o in proof.originals))
    if barrier=='stale_delivery':
        delivery=replace(delivery,ingresses=tuple(replace(i,ingress_monotonic_s=delivery.evaluation_monotonic_s-.352311) for i in delivery.ingresses))
        proof=replace(proof,originals=tuple(replace(o,ingress_monotonic_s=delivery.evaluation_monotonic_s-.352311) for o in proof.originals))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('current:'+r.stamp.observation_id,r.stamp,
        'INACTIVE' if barrier=='inactive' and r==rows[0] else 'ACTIVE'))
    if barrier=='inactive':proof=replace(proof,cohort=owner.image_region_active_cohort(),originals=proof.originals[1:])
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery,live_track_continuity=barrier!='default_false')
    assert not any(m.image_region_vote_blocked_stamp is not None for m in owner._members.values())
    if barrier!='inactive':assert result.reason!='estimated_live_track_continuity'


def test_image_regions_diagnostic_captures_actual_wall_expiry_split_not_media_assumption():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in range(1,6):
        image_diagnostic_observe(owner,frame,mode='fresh' if frame==1 else 'missing',wall_step=.1)
    view=owner.image_region_decision_diagnostics()
    record=view['transitions'][-1]
    assert len(record['pre_components'])==1 and len(record['post_components'])==2
    assert record['outcome']['reason']=='image_region_unavailable_or_ambiguous'
    assert all(r['input_status']=='absent' and r['effective_status']=='unavailable' for r in record['members'])
    assert all(r['confirmed_stamp']['frame_id']==1 and r['confirmed_ingress_monotonic_s']==100. for r in record['members'])
    assert all(r['confirmed_wall_age_s']>.35 and r['confirmed_source_age_s']<.35 for r in record['members'])
    assert record['pairs'][0]['state'] is None and record['pairs'][0]['possible'] is True and record['pairs'][0]['supported'] is False


@pytest.mark.parametrize('mode',['contradiction','invalid'])
def test_image_regions_diagnostic_distinguishes_actual_invalid_and_numeric_cues(mode):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1)
    image_diagnostic_observe(owner,2,mode=mode)
    record=owner.image_region_decision_diagnostics()['transitions'][-1]
    if mode=='contradiction':
        assert record['pairs'][0]['state'] is False
        assert record['pairs'][0]['cues'][0]=={'anatomy':'pose_torso_support','forward':False,'reverse':False}
        assert all(r['input_status']=='current_valid' for r in record['members'])
    else:
        assert record['pairs'][0]['state'] is None and record['pairs'][0]['cues']==[]
        assert any(r['input_status']=='current_invalid' for r in record['members'])


def test_image_regions_diagnostic_full_unknown_competitor_uniqueness():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1)
    image_diagnostic_observe(owner,2,mode='missing',competitor='missing')
    record=owner.image_region_decision_diagnostics()['transitions'][-1]
    assert len(record['members'])==3 and len(record['pairs'])==2
    assert any(p['state'] is True and p['unique'] is False for p in record['pairs'])
    assert any(p['state'] is None and p['possible'] is True for p in record['pairs'])


def test_image_regions_diagnostic_bound_detachment_json_and_no_decision_feedback():
    import json
    owners=(PersonAssociationLedger('run-a',spatial_policy()),PersonAssociationLedger('run-a',spatial_policy()))
    owners[1]._image_diag_begin=lambda *args:None
    for frame in range(1,25):
        a=image_diagnostic_observe(owners[0],frame,mode='fresh' if frame%2 else 'contradiction')
        b=image_diagnostic_observe(owners[1],frame,mode='fresh' if frame%2 else 'contradiction')
        assert a==b and owners[0].authority_revision==owners[1].authority_revision
    view=owners[0].image_region_decision_diagnostics()
    assert len(view['transitions'])==8 and view['evicted']==16 and view['capture_errors']==0
    assert all(len(r['members'])<=64 and len(r['pairs'])<=128 for r in view['transitions'])
    json.dumps(view,allow_nan=False)
    view['transitions'][0]['members'][0]['current_stamp']['frame_id']=99999
    assert owners[0].image_region_decision_diagnostics()['transitions'][0]['members'][0]['current_stamp']['frame_id']!=99999
    owners[0].reset_run('next-run')
    assert owners[0].image_region_decision_diagnostics()['transitions']==[]


def test_image_regions_diagnostic_capture_failure_never_changes_decision():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    def unavailable(*args):raise RuntimeError('diagnostic only failure')
    owner._image_diag_begin=unavailable
    result=image_diagnostic_observe(owner,1)
    assert result.disposition=='accepted' and len(owner._persons)==1
    assert owner.image_region_decision_diagnostics()['capture_errors']==1


def test_image_regions_encoded_snapshot_is_exact_join_immutable_across_next_join_reset_and_mutation(monkeypatch):
    from reid.mv3dt_person_association import ImageRegionDecisionSnapshot
    import reid.mv3dt_person_association as module
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1)
    expected=owner.image_region_decision_diagnostics()
    with monkeypatch.context() as m:
        m.setattr(module.json,'loads',lambda *a,**k:pytest.fail('decode at snapshot capture'))
        frozen=owner.image_region_decision_snapshot()
    assert type(frozen) is ImageRegionDecisionSnapshot
    assert type(frozen.envelope_json) is str and type(frozen.record_json) is tuple
    assert all(type(v) is str for v in frozen.record_json)
    assert frozen.materialize()==expected and len(frozen.record_json)<=8
    with pytest.raises(FrozenInstanceError):frozen.envelope_json='caller mutation'
    changed=frozen.materialize();changed['transitions'].clear()
    image_diagnostic_observe(owner,2,mode='contradiction')
    owner.close_image_region_run()
    assert frozen.materialize()==expected
    owner.reset_run('another-run')
    assert owner.image_region_decision_diagnostics()['transitions']==[]
    assert frozen.materialize()==expected  # No late owner re-fetch or live record.


def test_image_regions_encoded_snapshot_rejects_unissued_and_mutable_injected_values():
    from reid.mv3dt_person_association import ImageRegionDecisionSnapshot
    with pytest.raises(ValueError,match='unissued'):
        ImageRegionDecisionSnapshot._freeze('{}',(),(),object())
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1)
    valid=owner.image_region_decision_snapshot()
    forged=object.__new__(ImageRegionDecisionSnapshot)
    for name in ('envelope_json','record_json','record_kinds','_seal'):
        object.__setattr__(forged,name,getattr(valid,name))
    object.__setattr__(forged,'record_json',list(valid.record_json))
    with pytest.raises(ValueError,match='immutable'):forged.check_immutable()


def test_image_regions_encoded_snapshot_keeps_same_eight_kinds_and_detail_budget():
    import json
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for frame in range(1,13):
        image_diagnostic_observe(owner,frame,mode='fresh' if frame%2 else 'contradiction')
    frozen=owner.image_region_decision_snapshot()
    view=frozen.materialize()
    assert view==owner.image_region_decision_diagnostics()
    assert len(view['transitions'])==len(frozen.record_json)==8
    assert view['last_evaluated_sequence']==12
    assert all(len(s)<=1048576 for s in frozen.record_json)
    names=('floor_advance_note','paired_native_seed','owner_availability','live_native_checks','custody_check','lineage')
    assert sum(len(json.dumps({k:m[k] for k in names if k in m},separators=(',',':'),allow_nan=False).encode())
               for r in view['transitions'] for m in r['members'])<=8192


def test_image_regions_json_snapshot_cache_detached_exact_types_and_bounded_lifetime():
    import json
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1)
    original=owner._image_decision_last_completed
    original['scalar_probe']={'signed_zero':-0.0,'maximum':2**64-1,'unicode':'\u00e9\U0001f600',
        'types':[None,False,True,0,1.25,'literal']}
    revision=owner.authority_revision
    first=owner.image_region_decision_diagnostics()
    encoded=owner._image_decision_json[0][1]
    first['transitions'][0]['scalar_probe']['types'][0]='not owner data'
    second=owner.image_region_decision_diagnostics()
    probe=second['transitions'][0]['scalar_probe']
    assert probe==original['scalar_probe'] and type(probe['types']) is list
    assert json.dumps(probe['signed_zero'])=='-0.0' and type(probe['maximum']) is int
    assert owner._image_decision_json[0][1] is encoded
    assert owner.authority_revision==revision
    for frame in range(2,13):
        image_diagnostic_observe(owner,frame,mode='fresh' if frame%2 else 'contradiction')
        view=owner.image_region_decision_diagnostics()
        assert len(owner._image_decision_json)==len(view['transitions'])<=8
    assert all(r is not original for r,_ in owner._image_decision_json)
    owner.reset_run('new-run')
    assert owner._image_decision_json==() and owner.image_region_decision_diagnostics()['transitions']==[]


@pytest.mark.parametrize('fault',['nan','infinity','tuple','object','key','cycle','nodes','bytes'])
def test_image_regions_json_snapshot_malformed_optional_only_no_authority_or_live_reference(fault):
    owner=PersonAssociationLedger('run-a',spatial_policy())
    result=image_diagnostic_observe(owner,1)
    record=owner._image_decision_last_completed
    if fault=='nan':value=float('nan')
    elif fault=='infinity':value=float('inf')
    elif fault=='tuple':value=(1,2)
    elif fault=='object':value=object()
    elif fault=='key':value={1:'not a JSON string key'}
    elif fault=='cycle':value=[];value.append(value)
    elif fault=='nodes':value=[None]*65537
    else:value='x'*1048577
    record['malformed_observer_probe']=value
    before=(owner.authority_revision,tuple(owner._members),tuple(owner._persons))
    with pytest.raises(ValueError,match='image_region_diagnostic_json_'):
        owner.image_region_decision_diagnostics()
    assert result.disposition=='accepted' and before==(owner.authority_revision,tuple(owner._members),tuple(owner._persons))
    assert owner._image_decision_json==()
    del record['malformed_observer_probe']
    assert owner.image_region_decision_diagnostics()['transitions'][-1]['outcome']['disposition']=='accepted'


def test_image_regions_diagnostic_current_false_witness_exact_detached_input_not_confirmed_pose():
    import json
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1);image_diagnostic_observe(owner,2,mode='contradiction')
    record=owner.image_region_decision_diagnostics()['transitions'][-1]
    assert record['pairs'][0]['state'] is False
    for row in record['members']:
        original=row['current_input_original']
        assert row['input_is_current'] is True and row['input_validation_passed'] is True
        assert original['stamp']['frame_id']==2 and row['confirmed_stamp']['frame_id']==1
        assert original['body_evidence_id'].startswith('body:')
        assert original['prior_id'] and original['raster_id']
        assert len(original['image_size'])==2 and len(original['bbox_xywh'])==4
        assert original['patches'][0]['anatomy']=='pose_torso_support'
        assert original['patches'][0]['capsule']==([60.,150.,64.,150.,2.,2.] if row['member']['camera_id']=='family-room' else [60.,100.,64.,100.,2.,2.])
    json.dumps(record,allow_nan=False)
    record['members'][0]['current_input_original']['patches'][0]['capsule'][0]=999999.
    assert owner.image_region_decision_diagnostics()['transitions'][-1]['members'][0]['current_input_original']['patches'][0]['capsule'][0]==60.


def test_image_regions_diagnostic_absent_current_never_exports_retained_original_as_current():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1);image_diagnostic_observe(owner,2,mode='missing',competitor='missing')
    record=owner.image_region_decision_diagnostics()['transitions'][-1]
    assert any(m['effective_status']=='retained' for m in record['members'])
    assert all(m['current_input_original'] is None and m['input_is_current'] is False
        and m['input_validation_passed'] is None and m['input_status']=='absent' for m in record['members'])


@pytest.mark.parametrize('fault',['nan','infinite_bbox','many_patches','patch_iterator','wrong_raster','historical_stamp'])
def test_image_regions_diagnostic_input_copy_finite_bounded_and_no_decision_feedback(fault):
    import json
    class NotIterable:
        def __iter__(self):raise AssertionError('malformed diagnostic field iterated')
    owners=(PersonAssociationLedger('run-a',spatial_policy()),PersonAssociationLedger('run-a',spatial_policy()))
    owners[1]._image_diag_begin=lambda *args:None
    for owner in owners:image_diagnostic_observe(owner,1)
    selected,proof,rows,context,delivery=image_region_fixture(frame=2)
    original=proof.originals[0];patch=original.patches[0]
    if fault=='nan':original=replace(original,patches=(replace(patch,capsule=(float('nan'),)+patch.capsule[1:]),))
    elif fault=='infinite_bbox':original=replace(original,bbox_xywh=(float('inf'),)+original.bbox_xywh[1:])
    elif fault=='many_patches':original=replace(original,patches=(patch,)*1000)
    elif fault=='patch_iterator':original=replace(original,patches=NotIterable())
    elif fault=='wrong_raster':original=replace(original,raster_id='literal-wrong-raster')
    elif fault=='historical_stamp':original=replace(original,stamp=owners[0]._members[rows[0].stamp.member].observation.stamp)
    proof=replace(proof,originals=(original,)+proof.originals[1:])
    results=[]
    for owner in owners:
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('current:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        results.append(owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery))
    assert results[0]==results[1] and owners[0].authority_revision==owners[1].authority_revision
    view=owners[0].image_region_decision_diagnostics();json.dumps(view,allow_nan=False)
    record=view['transitions'][-1]
    member=next(m for m in record['members'] if m['member']['camera_id']=='kitchen')
    if fault=='wrong_raster':
        assert member['input_is_current'] is True and member['input_validation_passed'] is False
        assert member['current_input_original']['raster_id']=='literal-wrong-raster'
    else:assert member['current_input_original'] is None


@pytest.mark.parametrize('fault',['too_many_originals','original_iterator','too_many_cohort'])
def test_image_regions_diagnostic_does_not_inspect_oversized_or_untyped_input(fault):
    class NotIterable:
        def __iter__(self):raise AssertionError('untyped originals iterated')
    owner=PersonAssociationLedger('run-a',spatial_policy())
    image_diagnostic_observe(owner,1)
    selected,proof,rows,context,delivery=image_region_fixture(frame=2)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('current:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    prior=owner.image_region_decision_diagnostics()
    if fault=='too_many_originals':proof=replace(proof,originals=(proof.originals[0],)*65)
    elif fault=='original_iterator':proof=replace(proof,originals=NotIterable())
    else:proof=replace(proof,cohort=(proof.cohort[0],)*129)
    result=owner.join_estimated_image_regions(proof,projective_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.disposition=='association_unresolved'
    assert owner.image_region_decision_diagnostics()==prior


def test_image_regions_diagnostic_current_input_observer_failure_has_no_authority_effect():
    owner=PersonAssociationLedger('run-a',spatial_policy())
    def unavailable(*args):raise RuntimeError('optional observer copy unavailable')
    owner._image_diag_input_original=unavailable
    result=image_diagnostic_observe(owner,1)
    assert result.disposition=='accepted' and len(owner._persons)==1
    view=owner.image_region_decision_diagnostics()
    assert view['capture_errors']==2
    assert all(m['current_input_original'] is None and m['input_validation_passed'] is True for m in view['transitions'][-1]['members'])


@pytest.mark.parametrize('offset,expected',[(0.,1),(100.,2)])
def test_anatomical_stationary_numeric_support_is_connected_to_real_ledger(offset,expected):
    selected,candidate,rows,context,delivery=anatomical_fixture(right_offset=offset)
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native',row.stamp,'ACTIVE'))
    result=owner.join_anatomical_consensus(candidate,anatomical_policy=selected,at_interval=context,delivery_context=delivery)
    assert owner.stats().active_persons==expected
    assert result.disposition==('accepted' if expected==1 else 'observed')
    if expected==1:
        assert all(m.association_basis=='current_registered_anatomical_consensus' for m in owner._members.values())


@pytest.mark.parametrize('fault,reason',[
    ('qualification','anatomical_static_pair_qualification_unavailable'),
    ('interval','anatomical_registered_interval_unavailable'),
    ('ankle-mean','anatomical_representative_outside_sampled_support'),
    ('age','anatomical_original_binding_or_age'),('raster','anatomical_original_binding_or_age'),
    ('scope','anatomical_static_pair_qualification_binding'),
])
def test_anatomical_missing_or_misbound_evidence_never_becomes_support(fault,reason):
    selected,candidate,rows,context,delivery=anatomical_fixture(qualified=fault!='qualification')
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native',row.stamp,'ACTIVE'))
    pair=candidate.pairs[0]; a=pair.left_originals[0]
    if fault=='interval':a=replace(a,registered_depth_interval_m=None,depth_convention=None)
    elif fault=='ankle-mean':a=replace(a,capsules=((50.,160.,50.,162.,1.,1.),(70.,160.,70.,162.,1.,1.)))
    elif fault=='age':a=replace(a,original=replace(a.original,ingress_monotonic_s=99.))
    elif fault=='raster':a=replace(a,original=replace(a.original,raster_id='wrong'))
    elif fault=='scope':selected=replace(selected,qualification=replace(selected.qualification,scope='public_metric'))
    candidate=replace(candidate,pairs=(replace(pair,left_originals=(a,pair.left_originals[1])),))
    result=owner.join_anatomical_consensus(candidate,anatomical_policy=selected,at_interval=context,delivery_context=delivery)
    assert result.reason==reason and owner.stats().active_persons==2


def test_anatomical_full_competitor_graph_unknown_blocks_but_evaluated_incompatibility_does_not():
    selected,candidate,rows,context,delivery=anatomical_fixture()
    competitor=replace(rows[0],stamp=replace(rows[0].stamp,member=replace(rows[0].stamp.member,raw_id=17),observation_id='other-original'))
    rows=rows+(competitor,)
    owner=PersonAssociationLedger('run-a',spatial_policy())
    for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native',row.stamp,'ACTIVE'))
    candidate=replace(candidate,visual=replace(visual_proof(rows,scores={(7,8):.91,(17,8):.91}),selected_pairs=()))
    result=owner.join_anatomical_consensus(candidate,anatomical_policy=selected,at_interval=spatial_context(rows),
                                         delivery_context=visual_delivery(rows,sequence=1,evaluation=100.1))
    assert result.disposition=='observed' and owner.stats().active_persons==3
    # Unknown was NOT omitted: the otherwise supported pair is still blocked.
    assert len(owner._persons)==3

@pytest.mark.parametrize("defect", ["proof", "policy", "context", "pair", "missing-member", "capacity"])
def test_projective_inspection_invalid_input_is_read_only_and_never_raises_keyerror(defect):
    owner, selected, candidate, rows, context, delivery = projective_kernel_fixture()
    if defect == "proof":
        candidate = None
    elif defect == "policy":
        selected = None
    elif defect == "context":
        context = None
    elif defect == "pair":
        candidate = replace(candidate, pairs=(None,))
    elif defect == "missing-member":
        owner.withdraw(removal(rows[1], frame=4, time=10.15, media=5.15))
    else:
        candidate = replace(candidate, pairs=candidate.pairs * 22)
    before = repr(vars(owner)), owner.stats()
    report = owner.inspect_projective_continuity(candidate, projective_policy=selected,
        at_interval=context, delivery_context=delivery)
    assert report.error is not None and report.pairs == ()
    assert (repr(vars(owner)), owner.stats()) == before


def test_projective_numeric_gate_positive_and_no_cue_point91_stays_two_without_oracle_labels():
    owner, selected, candidate, rows, context, delivery = projective_kernel_fixture()
    missing = owner.join_projective_continuity(replace(candidate, pairs=()), projective_policy=selected,
                                               at_interval=context, delivery_context=delivery)
    assert missing.disposition == "association_unresolved" and owner.stats().active_persons == 2
    delivery = replace(delivery, sequence=2)
    result = owner.join_projective_continuity(candidate, projective_policy=selected, at_interval=context, delivery_context=delivery)
    assert result.disposition == "accepted" and owner.stats().active_persons == 1
    assert all(m.association_basis == "prior_projective_continuity" for m in owner._members.values())
    assert owner.join_projective_continuity(candidate, projective_policy=selected, at_interval=context, delivery_context=delivery) is result
    assert owner.stats().receipts == owner.stats().retired_keys == 0 and len(owner._projective_frontiers) == 2
    assert all(c.measurement_calibration_sha256 != c.source.calibration_revision for c in selected.cameras)


@pytest.mark.parametrize("defect", ["stationary", "jitter", "radius-none", "radius-zero", "radius-nan", "bbox", "raster", "image",
    "point", "negative-depth", "own-scale", "one-way", "calibration", "transform", "world", "stage", "origin", "binding",
    "prequeue", "own-age", "epoch", "birth", "order", "short", "reversed"])
def test_projective_numeric_gate_rejects_unsupported_or_original_binding_defects(defect):
    owner, selected, candidate, rows, context, delivery = projective_kernel_fixture(stationary=defect == "stationary")
    pair = candidate.pairs[0]
    sample = pair.left[0]
    if defect in {"own-scale", "one-way"}:
        c = selected.cameras[0]
        matrix = ((200., 0., 0., 0.),) + c.own_projection[1:]
        c = replace(c, **{"own_projection" if defect == "own-scale" else "peer_projection": matrix})
        selected = replace(selected, cameras=(c, selected.cameras[1]))
    elif defect == "short":
        pair = replace(pair, left=pair.left[:2])
    elif defect == "reversed":
        pair = replace(pair, right=tuple(reversed(pair.right)))
    elif defect == "jitter":
        def jitter(side):
            return tuple(replace(s, observed_anchor_uv=(40 + i * .1, 100.), original_point=((40 + i * .1) / 100, 1., 1.)) for i, s in enumerate(side))
        pair = replace(pair, left=jitter(pair.left), right=jitter(pair.right))
    elif defect != "stationary":
        if defect.startswith("radius"):
            sample = replace(sample, pixel_uncertainty_px={"radius-none": None, "radius-zero": 0., "radius-nan": float("nan")}[defect])
        elif defect in {"bbox", "raster", "image", "point", "negative-depth", "prequeue"}:
            field, value = {"bbox": ("bbox_xywh", (0., 0., 1., 1.)), "raster": ("raster_id", "wrong"),
                "image": ("image_size", (320, 240)), "point": ("original_point", (500., 1., 1.)),
                "negative-depth": ("original_point", (.4, 1., -1.)), "prequeue": ("ingress_monotonic_s", 99.)}[defect]
            sample = replace(sample, **{field: value})
        elif defect in {"epoch", "birth", "order", "own-age"}:
            stamp = sample.stamp
            stamp = replace(stamp, **({"member": replace(stamp.member, source_epoch=1)} if defect == "epoch" else
                {"lifecycle_started_frame": 0} if defect == "birth" else {"media_time_s": 5.2} if defect == "order" else {"acquired_at_s": 9.}))
            sample = replace(sample, stamp=stamp)
        else:
            field, value = {"calibration": ("calibration_revision", "source-frame-calibration-is-NOT-camera-SHA"), "transform": ("transform_id", "wrong"),
                "world": ("world_revision", "wrong"), "stage": ("measured_stage_id", ""), "origin": ("origin", "held"),
                "binding": ("target_frame", "wrong")}[defect]
            sample = replace(sample, measurement=replace(sample.measurement, **{field: value}))
        pair = replace(pair, left=(sample,) + pair.left[1:])
    candidate = replace(candidate, pairs=(pair,))
    result = owner.join_projective_continuity(candidate, projective_policy=selected, at_interval=context, delivery_context=delivery)
    assert result.disposition == "association_unresolved", (defect, result)
    assert owner.stats().active_persons == 2 and owner.stats().receipts == 0


def test_projective_full_appearance_competitor_without_cue_still_blocks_uniqueness():
    owner, selected, candidate, rows, context, delivery = projective_kernel_fixture()
    competitor = observation(raw=17, frame=3, time=10.1, media=5.1)
    owner.observe_spatial(competitor, SpatialLifecycleReceipt("real-competing-native", competitor.stamp, "ACTIVE"))
    rows += (competitor,)
    candidate = replace(candidate, visual=replace(visual_proof(rows, scores={(7, 8): .91, (17, 8): .91}),
                                                selected_pairs=((rows[0].stamp.member, rows[1].stamp.member),)))
    result = owner.join_projective_continuity(candidate, projective_policy=selected, at_interval=spatial_context(rows),
        delivery_context=visual_delivery(rows, sequence=1, evaluation=100.1))
    assert result.disposition == "association_unresolved" and owner.stats().active_persons == 3
