import pytest
from noesis_core.contracts.world_measurement import WorldMeasurementHypothesis, WorldMeasurementCohort, WorldPriorEvidence, WorldMeasurementSet
from noesis_core.world.resolver import UniversalWorldMeasurementResolver, WorldMeasurementResolverConfig


def column(**changes):
    data=dict(candidate_id='registered_body_column',method='registered_body_column',kind='registered_depth',
        cohort=WorldMeasurementCohort(track_key='2:4:4',tracker_lifecycle_generation=4,camera_id='family-room',
            source_id=2,tracker_id=4,frame_id=36,observed_at_us=1200000,world_revision='bound',calibration_revision='cal',pcf_revision='bound'),
        position=dict(x=11.667815751851432,y=0.,z=5.144617491142958),
        covariance=dict(values=(.1479894229094094,0.,.29063390671350425,0.,.02,0.,.29063390671350425,0.,.5721112158510026)),
        anchor='registered_body_column',posture='unknown',support_state='unknown',
        pcf=WorldPriorEvidence(prior_id='prior',revision_id='bound',status='fail',inside_extent=True,
            inside_authored_space=False,evidence_observed=False,observed_confidence=0.,boundary_signed_distance_m=-3.229745,
            reasons=('outside_authored_space',)))
    data.update(changes)
    return WorldMeasurementHypothesis(**data)


def test_r65_exact_unobserved_room_only_is_soft_only_for_explicit_method():
    c=column();strict=UniversalWorldMeasurementResolver()
    task=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    assert strict._pcf_strong_conflict(c)
    assert not task._pcf_strong_conflict(c)
    assert task._pcf_score(c.position,c.pcf)==strict._pcf_score(c.position,c.pcf)<.01


@pytest.mark.parametrize('changes',[dict(evidence_observed=True),dict(observed_confidence=.1),
    dict(inside_extent=False),dict(reasons=('outside_authored_space','other')),dict(reasons=())])
def test_r65_observed_grid_and_unknown_reasons_remain_strict(changes):
    c=column();c=c.model_copy(update={'pcf':c.pcf.model_copy(update=changes)})
    task=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    assert task._pcf_strong_conflict(c)


def test_r65_ordinary_method_and_bad_semantics_cannot_use_exception():
    task=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    assert task._pcf_strong_conflict(column(method=None))
    for changes in (dict(support_state='floor'),dict(anchor='person_body_projection')):
        with pytest.raises(ValueError):column(**changes)


@pytest.mark.parametrize('posture',['standing','sitting','lying','unknown'])
def test_r68_column_preserves_actual_posture_without_contact_claim(posture):
    c=column(posture=posture)
    assert c.posture==posture and c.support_state=='unknown' and c.support_surface is None


def test_r65_default_resolver_does_not_activate_new_method_without_opt_in():
    c=column();rows=WorldMeasurementSet(cohort=c.cohort,hypotheses=(c,))
    assert UniversalWorldMeasurementResolver().resolve(rows).status=='rejected'
    task=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    result=task.resolve(rows)
    assert result.status=='measured' and result.posture==result.support_state=='unknown'
    assert result.reason!='selected_hypothesis_strongly_conflicts_with_pcf'
    assert c.pcf.boundary_signed_distance_m==-3.229745 and c.position.x==11.667815751851432
