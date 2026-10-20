"""R45 F8/frame181: single-room nonmembership is not measured exclusion."""
import pytest
from pydantic import ValidationError
from noesis_core.contracts.world_measurement import WorldMeasurementHypothesis, WorldMeasurementSet
from noesis_core.world.resolver import UniversalWorldMeasurementResolver, WorldMeasurementResolverConfig


def literal_f181():
    # Frozen original_body_inspection_only F181 registered_depth, not a new fit.
    return WorldMeasurementHypothesis.model_validate(dict(
        anchor='person_mask_floor',candidate_id='registered_depth',kind='registered_depth',method=None,
        cohort=dict(calibration_revision='9b34e3ea8a3808475584a55383afcff96373a9e0845d33dc026a6659c975be93',
            camera_id='family-room',frame_id=181,observed_at_us=1792130859101301,
            pcf_revision='vt_family_room_stream_rgbmesh_20260623T215850_645134993',source_id=2,
            track_key='2:8:5',tracker_id=8,tracker_lifecycle_generation=5,world_frame='backend_world_m',
            world_revision='sceneprior_family-room_20260811T015847Z_8b80dc69a7c4',
            world_transform_sha256='5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217'),
        confidence=1.,correlation_group='registered_depth:2:8',
        covariance=dict(values=[.14169753571313717,0.,.27748330738378607,0.,.020000000000000018,0.,
            .2774833073837861,0.,.5446035506504942]),depth_registration_sigma_m=.7382162590832481,
        depth_support_fraction=1.,floor_depth_disagreement_m=.46965140338330674,motion_consistency=.75,
        occlusion=0.,pcf=dict(boundary_signed_distance_m=-2.711665,evidence_observed=False,
            extent_outside_distance_m=0.,floor_height_m=0.,inside_authored_space=False,inside_extent=True,
            observed_confidence=0.,obstacle_clearance_m=1.007782,
            prior_id='sceneprior_family-room_20260811T015847Z_8b80dc69a7c4',
            reasons=['outside_authored_space'],revision_id='vt_family_room_stream_rgbmesh_20260623T215850_645134993',
            status='fail'),position=dict(x=11.915390086333183,y=0.,z=5.652447960459152),
        posture='standing',posture_compatibility=1.,source_reliability=1.,support_score=1.,
        support_state='floor',valid=True))


def resolver(enabled=True):
    return UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=enabled))


def resolve(r,h):
    return r.resolve(WorldMeasurementSet(cohort=h.cohort,hypotheses=(h,)))


def test_r90_literal_current_depth_room_nonmembership_is_soft_only_task_enabled():
    h=literal_f181();off=resolver(False);on=resolver()
    before=resolve(off,h);after=resolve(on,h)
    assert off._pcf_strong_conflict(h) and not on._pcf_strong_conflict(h)
    assert before.quality=='weak' and after.quality in {'good','estimated'}
    assert before.pcf_score==after.pcf_score==on._pcf_score(h.position,h.pcf)
    assert before.pcf_score < .01 and after.confidence > before.confidence
    assert after.position==before.position==h.position
    assert after.covariance==before.covariance==h.covariance
    assert after.cohort==h.cohort and after.contributor_ids==('registered_depth',)
    assert after.reason=='selected_single_valid_hypothesis'


@pytest.mark.parametrize('change',[
    dict(evidence_observed=True),dict(observed_confidence=.001),dict(inside_extent=False),
    dict(reasons=('outside_authored_space','outside_prior_extent')),
    dict(reasons=('outside_prior_extent',)),dict(reasons=()),
    dict(inside_extent=None),dict(evidence_observed=None),
])
def test_r90_observed_extent_missing_or_mixed_evidence_never_exempt(change):
    h=literal_f181();h=h.model_copy(update={'pcf':h.pcf.model_copy(update=change)})
    assert resolver()._pcf_strong_conflict(h)
    assert resolve(resolver(),h).quality=='weak'


@pytest.mark.parametrize('kind',['floor_ray','pose_scale','gravity_reconstruction'])
def test_r90_unrelated_kinds_still_strong_in_same_task(kind):
    h=literal_f181().model_copy(update={'kind':kind,'anchor':'lower_body_contact'})
    assert resolver()._pcf_strong_conflict(h)
    assert resolve(resolver(),h).quality=='weak'


def test_r90_invalid_cached_original_cannot_become_measurement():
    h=literal_f181().model_copy(update={'valid':False,'rejection_reason':'depth_measurement_not_current'})
    assert resolve(resolver(),h).status=='rejected'


def test_r90_exact_cohort_and_prior_revision_still_required():
    h=literal_f181()
    with pytest.raises(ValidationError):
        WorldMeasurementSet(cohort=h.cohort.model_copy(update={'frame_id':182}),hypotheses=(h,))
    raw=h.model_dump();raw['pcf']['revision_id']='foreign'
    with pytest.raises(ValidationError):WorldMeasurementHypothesis.model_validate(raw)


def test_r90_existing_column_task_signature_remains_soft_without_fusion():
    h=literal_f181().model_copy(update={'candidate_id':'registered_body_column','method':'registered_body_column',
        'anchor':'registered_body_column','support_state':'unknown'})
    assert not resolver()._pcf_strong_conflict(h)
    assert resolve(resolver(False),h).status=='rejected'


def test_r90_missing_method_cannot_relabel_column_as_ordinary_registered_depth():
    h=literal_f181().model_copy(update={'anchor':'registered_body_column','method':None})
    assert resolver()._pcf_strong_conflict(h)
    assert resolve(resolver(),h).quality=='weak'


def test_r90_actual_body_projection_ordinary_anchor_has_same_exact_signature():
    h=literal_f181().model_copy(update={'anchor':'person_body_projection','posture':'sitting','support_state':'seat'})
    assert not resolver()._pcf_strong_conflict(h)
    assert resolve(resolver(),h).pcf_score==resolve(resolver(False),h).pcf_score


def analogous_fused_pair():
    """R45 F181 geometry; floor eligibility varied to model R46 fusion.

    The captured R45 floor was invalid ordinary_priority_over_body_column.
    R46 164/170/176 prove valid floor contribution, but omit full hypotheses.
    ONLY this fixture's status is varied; it is not a literal R46 point copy.
    """
    depth=literal_f181()
    raw=depth.model_dump()
    raw.update(candidate_id='floor_ray',kind='floor_ray',anchor='pose_ankle_floor',confidence=.95,
        correlation_group='floor_ray:2:8',support_score=.95,ray_incidence_sin=.21934552305419106,
        pixel_uncertainty_px=2.,depth_registration_sigma_m=None,depth_support_fraction=None,
        floor_depth_disagreement_m=None,valid=True,rejection_reason=None,
        position=dict(x=12.129193598746001,y=0.,z=6.070611204635217),
        covariance=dict(values=[.008593613740656196,0.,.014343793384976369,0.,.04122617907783236,0.,
            .01434379338497637,0.,.030203951185678056]))
    raw['pcf'].update(boundary_signed_distance_m=-2.307054,obstacle_clearance_m=.618971)
    return depth,WorldMeasurementHypothesis.model_validate(raw)


def test_r93_analogous_registered_range_is_not_downgraded_by_semantic_floor_fusion():
    depth,floor=analogous_fused_pair();r=resolver()
    own=resolve(r,depth);floor_own=resolve(r,floor)
    result=r.resolve(WorldMeasurementSet(cohort=depth.cohort,hypotheses=(floor,depth)))
    assert result.selected_candidate_id==depth.candidate_id
    assert not result.fused and result.contributor_ids==(depth.candidate_id,)
    assert result.position==own.position and result.covariance==own.covariance
    assert result.quality==own.quality=='estimated'
    assert result.confidence==own.confidence and result.pcf_score==own.pcf_score
    assert result.alternate_candidate_id==floor.candidate_id
    diagnostic=next(d for d in result.diagnostics if d.candidate_id==floor.candidate_id)
    assert diagnostic.retained_as_alternate and not diagnostic.compatible_with_selected
    assert 'strong_pcf_boundary_conflict' in diagnostic.rejection_reason
    assert r._score(floor,None,None).pcf_strong_conflict
    assert floor_own.quality=='weak' and diagnostic.score==floor_own.confidence


@pytest.mark.parametrize('change',[
    {'evidence_observed':True}, {'observed_confidence':.1},
    {'inside_extent':False,'extent_outside_distance_m':3.,'reasons':('outside_prior_extent',)},
    {'reasons':('outside_authored_space','other_conflict')}, {'prior_id':'foreign-prior'},
])
def test_r93_floor_real_or_mixed_exclusion_still_contributes_shared_conflict(change):
    depth,floor=analogous_fused_pair()
    floor=floor.model_copy(update={'pcf':floor.pcf.model_copy(update=change)})
    result=resolver().resolve(WorldMeasurementSet(cohort=depth.cohort,hypotheses=(depth,floor)))
    assert result.fused and result.quality=='weak'
    assert set(result.contributor_ids)=={'registered_depth','floor_ray'}
    assert result.reason=='selected_hypothesis_strongly_conflicts_with_pcf'


def test_r93_default_shared_mirror_policy_unchanged():
    depth,floor=analogous_fused_pair()
    result=resolver(False).resolve(WorldMeasurementSet(cohort=depth.cohort,hypotheses=(depth,floor)))
    assert result.fused and result.quality=='weak'
    assert result.reason=='selected_hypothesis_strongly_conflicts_with_pcf'


def test_r93_observed_registered_range_still_has_strong_shared_guard():
    depth,floor=analogous_fused_pair()
    depth=depth.model_copy(update={'pcf':depth.pcf.model_copy(update={'evidence_observed':True})})
    result=resolver().resolve(WorldMeasurementSet(cohort=depth.cohort,hypotheses=(depth,floor)))
    assert result.fused and result.quality=='weak'
    assert result.reason=='selected_hypothesis_strongly_conflicts_with_pcf'


def test_r93_incompatible_floor_retains_real_disagreement_and_covariance():
    depth,floor=analogous_fused_pair()
    floor=floor.model_copy(update={'position':floor.position.model_copy(update={'x':depth.position.x+4.})})
    result=resolver().resolve(WorldMeasurementSet(cohort=depth.cohort,hypotheses=(depth,floor)))
    assert not result.fused and result.alternate_candidate_id==floor.candidate_id
    assert result.covariance!=depth.covariance
    assert next(d for d in result.diagnostics if d.candidate_id=='floor_ray').rejection_reason=='not_mutually_compatible_with_contributors'


def test_r93_floor_selected_does_not_gain_registered_range_authority():
    depth,floor=analogous_fused_pair()
    depth=depth.model_copy(update={'source_reliability':.01})
    result=resolver().resolve(WorldMeasurementSet(cohort=depth.cohort,hypotheses=(depth,floor)))
    assert result.selected_candidate_id=='floor_ray' and result.quality=='weak'
    assert all('semantic_room_noncontributor' not in (d.rejection_reason or '') for d in result.diagnostics)


def test_r93_foreign_pcf_revision_never_reaches_fusion():
    depth,floor=analogous_fused_pair();raw=floor.model_dump();raw['pcf']['revision_id']='foreign'
    with pytest.raises(ValidationError): WorldMeasurementHypothesis.model_validate(raw)


def test_r93_explained_floor_cannot_hide_a_real_strong_contributor():
    depth,floor=analogous_fused_pair()
    observed=depth.model_copy(update={'candidate_id':'observed-depth','correlation_group':'independent-depth',
        'pcf':depth.pcf.model_copy(update={'evidence_observed':True})})
    result=resolver().resolve(WorldMeasurementSet(cohort=depth.cohort,hypotheses=(depth,floor,observed)))
    assert result.fused and result.quality=='weak'
    assert set(result.contributor_ids)=={'registered_depth','observed-depth'}
    assert result.reason=='selected_hypothesis_strongly_conflicts_with_pcf'
    assert next(d for d in result.diagnostics if d.candidate_id=='observed-depth').rejection_reason=='strong_pcf_boundary_conflict'
