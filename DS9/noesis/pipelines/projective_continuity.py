"""Private bounded ORIGINAL evidence for estimated cross-view correspondence.

The existing presentation similarities are projective PRIOR hypotheses, not
admitted common metric calibration. This module never publishes coordinates,
changes household identity, creates a measurement grant or extracts SDK data.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass, replace
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from math import hypot, isfinite, sqrt
import time
import os
import json

from reid.mv3dt_person_association import (
    AcquisitionTime, CurrentVisualDeliveryContext, CurrentVisualIngress,
    CurrentVisualProof, ProjectiveCameraBinding, ProjectiveContinuityPolicy,
    ProjectiveContinuityProof, ProjectiveMeasurementBinding, ProjectiveMotionPair,
    ProjectiveMotionSample, SpatialPersonCandidate,
    PersonAssociationLedger,
    AnatomicalConsensusProof, AnatomicalDepthSupport, AnatomicalPairSupport, ObservationStamp,
    ObservedImageRegion, ImageRegionOriginal, ImageRegionNativeBox, EstimatedImageRegionProof, ImageRegionOwnerAvailability, MemberKey,
    ImageRegionDecisionSnapshot,
    ImageRegionAppearance, AppearanceEvidence, appearance_extraction_identity,
)

_radius = PersonAssociationLedger._support_radius


def image_regions_enabled(config):
    value = config.get('xc3d_image_regions')
    if value is None:
        return False
    if (type(value) is not dict
            or set(value) - {'live_track_continuity','contradictory_cue_quarantine','semantic_joint_max_mean_px','unresolved_birth_quarantine'} != {'contract','enabled','prior_path','prior_sha256'}
            or value['contract'] != 'noesis.xc3d.estimated_image_regions.task.v1'
            or type(value['enabled']) is not bool
            or type(value.get('live_track_continuity',False)) is not bool
            or type(value.get('contradictory_cue_quarantine',False)) is not bool
            or (value.get('contradictory_cue_quarantine',False)
                and not value.get('live_track_continuity',False))):
        raise ValueError('invalid explicit image-region task configuration')
    threshold=value.get('semantic_joint_max_mean_px')
    if threshold is not None and (type(threshold) not in (int,float) or not isfinite(threshold)
            or threshold<=0 or not value.get('live_track_continuity',False)):
        raise ValueError('invalid explicit image-region task configuration')
    if type(value.get('unresolved_birth_quarantine',False)) is not bool or value.get('unresolved_birth_quarantine',False) and (
            threshold is None or not value.get('live_track_continuity',False)):
        raise ValueError('invalid explicit image-region task configuration')
    if value['enabled']:
        from noesis.v3dt_assets import xc3d_world_quality_enabled
        if not xc3d_world_quality_enabled(config) or projective_continuity_enabled(config):
            raise ValueError('image-region task requires quality and exclusive mode')
    return value['enabled']


def runtime_image_region_policy(config, bindings, provider, repo_root):
    """Bounded startup-only task prior; never changes provider/world geometry."""
    import json
    from pathlib import Path
    import numpy as np
    if not image_regions_enabled(config):
        raise ValueError('image-region task not explicitly enabled')
    setting = config['xc3d_image_regions']
    path = Path(setting['prior_path'])
    if not path.is_absolute():
        path = Path(repo_root) / path
    with path.open('rb') as handle:
        raw = handle.read(32769)
    if len(raw)>32768 or sha256(raw).hexdigest()!=setting['prior_sha256']:
        raise ValueError('image-region task prior size/hash mismatch')
    artifact = json.loads(raw)
    if (set(artifact) != {'contract','estimated_only','source_result_sha256','cameras'}
            or artifact['contract'] != 'noesis.xc3d.estimated_image_prior.v1'
            or artifact['estimated_only'] is not True or len(artifact['cameras']) != 2):
        raise ValueError('invalid estimated image prior artifact')
    base = runtime_projective_policy(bindings, provider)
    expected = {v['camera_id']:v for v in artifact['cameras']}
    if set(expected) != {'kitchen','family-room'}:
        raise ValueError('exact two task prior sources required')
    presentations = provider.world_frame_presentations()
    result = []
    for camera in base.cameras:
        row = expected[camera.source.camera_id]
        snapshot = provider.snapshot(camera.source.source_id,camera.source.camera_id)
        presentation = presentations[camera.world_revision]
        exact = {'source_id':camera.source.source_id,'image_size':list(camera.image_size),
            'source_calibration_revision':camera.source.calibration_revision,
            'transform_id':camera.source.transform_id,'world_revision':camera.world_revision,
            'measurement_calibration_sha256':camera.measurement_calibration_sha256,
            'presentation_sha256':presentation['world_to_scene_sha256'],
            'authored_scene_sha256':presentation['authored_scene_sha256']}
        if any(row.get(k)!=v for k,v in exact.items()):
            raise ValueError('image-region task source revision/raster mismatch')
        for name,actual in (('intrinsics',snapshot.intrinsics),
                ('world_extrinsics_col_major',snapshot.world_extrinsics_col_major),
                ('presentation_col_major',presentation['world_to_scene_col_major'])):
            a,b=np.asarray(row[name],float),np.asarray(actual,float)
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.allclose(a,b,rtol=0,atol=1e-10):
                raise ValueError('image-region task K/E/B/P numeric mismatch')
        peer=np.asarray(row['candidate_peer_projection'],float)
        if peer.shape!=(3,4) or not np.isfinite(peer).all():
            raise ValueError('invalid task peer projection')
        result.append(replace(camera,prior_id='estimated-task:'+sha256(raw).hexdigest(),
            peer_projection=tuple(tuple(float(v) for v in line) for line in peer)))
    return ProjectiveContinuityPolicy(tuple(result))


def full_observed_image_regions(body, *, joint_threshold=.35):
    """Full joint-derived anatomical enclosures, NOT depth ROIs or uncertainty.

    Shoulder region is the minimal disc enclosing the observed shoulder pair.
    Torso is the full shoulder-mid/hip-mid segment with radius equal to half
    the larger observed bilateral span. No trims, 5px minimum, sigma, missing
    joint interpolation or projection-dependent enlargement is used.
    """
    if body is None:
        return ()
    points = body.keypoints
    def point(i):
        u,v,c=points[i]
        return (u,v) if c>=joint_threshold and all(isfinite(x) for x in (u,v,c)) else None
    left,right=point(5),point(6)
    if left is None or right is None:
        return ()
    width=hypot(left[0]-right[0],left[1]-right[1])
    if width<=0:
        return ()
    shoulder=tuple((a+b)/2 for a,b in zip(left,right))
    regions=[ObservedImageRegion('pose_shoulder_support',shoulder,shoulder+shoulder+(width/2,width/2))]
    a,b=point(11),point(12)
    if a is not None and b is not None:
        hip=tuple((x+y)/2 for x,y in zip(a,b))
        radius=max(width,hypot(a[0]-b[0],a[1]-b[1]))/2
        anchor=tuple((x+y)/2 for x,y in zip(shoulder,hip))
        regions.insert(0,ObservedImageRegion('pose_torso_support',anchor,shoulder+hip+(radius,radius)))
    return tuple(regions)


class _OriginalBodyDiagnostic:
    """Optional owned-scalar inspection only; never read by association decisions."""

    def __init__(self, targets):
        self.targets = tuple(targets)
        if (len(self.targets) > 4 or len(set(self.targets)) != len(self.targets)
                or any(type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 86400
                       for v in self.targets)):
            raise ValueError('invalid original body diagnostic targets')
        self.records = {}
        self.errors = 0
        self.sampling_lookup = None
        self._sampling_rows = {}  # Same <=64 retained inspection records, owned scalars only.
        self.admission_records = {}
        self.admission_omitted = 0

    def observe_admission(self, plans, stamps, availability, result, originals, *, appearance_info=None):
        """Freeze only already-computed native/join scalars at existing targets."""
        if not self.targets:return
        try:
            capsules={v.observation.member for v in availability}
            participants={s.member for s in stamps}
            for work,_binding,_watermark,_retained,_tracks,_observations in plans:
                media=work.private.media_pts_ns/1e9
                for index,target in enumerate(self.targets):
                    distance=abs(media-target);key=(work.continuity.source_id,index)
                    if distance>.100000001:continue
                    old=self.admission_records.get(key)
                    if old is not None and distance>=old['distance_s']:continue
                    if old is None and len(self.admission_records)>=8:
                        self.admission_omitted+=1;continue
                    owner=work.private.owner_retention
                    rows=[]
                    if owner is not None:
                        active={(v[0],v[1]) for v in owner.active}
                        shadow={(v[0],v[1]) for v in owner.shadow}
                        for entry in owner.keys[:64]:
                            raw,generation=entry[2:4]
                            member=MemberKey(work.private.original.run_id,work.continuity.camera_id,
                                work.continuity.source_id,owner.source_epoch,raw,generation)
                            known=originals.get(member)
                            rows.append(dict(raw=raw,generation=generation,retained=True,
                                birth=known.stamp.lifecycle_started_frame if known else None,
                                original_frame=known.stamp.frame_id if known else None,
                                active=(raw,generation) in active,shadow=(raw,generation) in shadow,
                                terminated=raw in owner.terminated,participant=member in participants,
                                capsule_included=member in capsules))
                    self.admission_records[key]={'source':work.continuity.source_id,
                        'epoch':work.continuity.source_epoch,'frame':work.continuity.frame_id,
                        'pts_ns':work.private.media_pts_ns,'target_media_s':target,'distance_s':distance,
                        'owner_present':owner is not None,'native_members':rows,
                        'owner_epoch':owner.source_epoch if owner else None,
                        'owner_frame':owner.frame_id if owner else None,
                        'native_member_count':len(owner.keys) if owner else None,
                        'native_members_omitted':max(0,len(owner.keys)-len(rows)) if owner else None,
                        'native_active':[list(v[:2]) for v in owner.active[:64]] if owner else None,
                        'native_active_omitted':max(0,len(owner.active)-64) if owner else None,
                        'native_shadow':[list(v[:2]) for v in owner.shadow[:64]] if owner else None,
                        'native_shadow_omitted':max(0,len(owner.shadow)-64) if owner else None,
                        'native_terminated':list(owner.terminated[:64]) if owner else None,
                        'native_terminated_omitted':max(0,len(owner.terminated)-64) if owner else None,
                        'availability_present':work.owner_availability is not None,
                        'availability_status':work.owner_availability.status if work.owner_availability else None,
                        'join_disposition':result.disposition,'join_reason':result.reason,
                        'withdrawn':[(m.source_id,m.source_epoch,m.raw_id,m.lifecycle_generation)
                            for m in result.withdrawn_members[:128]],
                        'withdrawn_omitted':max(0,len(result.withdrawn_members)-128),
                        'retired':[k.serial for k in result.retired_keys[:64]],
                        'retired_omitted':max(0,len(result.retired_keys)-64),
                        'appearance':appearance_info}
        except Exception:
            try:self.errors+=1
            except Exception:pass

    def admission_summary(self):
        rows=deepcopy(list(self.admission_records.values()));omitted=self.admission_omitted
        while True:
            value={'limit':8,'byte_limit':55*1024,'omitted_records':omitted,'records':rows}
            # Match the existing final runtime summary's default JSON encoder,
            # including spaces/escaping; the bound is on delivered bytes.
            encoded=json.dumps(value,allow_nan=False)
            if len(encoded.encode())<=55*1024:return json.loads(encoded)
            rows.pop();omitted+=1

    def _sampling_snapshot(self,row):
        if not callable(self.sampling_lookup):
            return {'available':False,'reason':'sampling_diagnostic_not_bound'}
        try:
            value=self.sampling_lookup(row)
            if (type(value) is not dict or len(value)>20 or type(value.get('available')) is not bool
                    or any(type(k) is not str or len(k)>64 for k in value)):
                raise ValueError('unbounded sampling diagnostic')
            return {k:self._owned(v) for k,v in value.items()}
        except Exception:
            return {'available':False,'reason':'sampling_diagnostic_unavailable'}

    @classmethod
    def from_environment(cls):
        # Construction happens outside frame callbacks. Empty means no work.
        value = os.environ.get('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S', '')
        if len(value) > 128:
            raise ValueError('original body diagnostic targets too long')
        return cls(tuple(float(v.strip()) for v in value.split(',')) if value else ())

    @staticmethod
    def _owned(value, depth=0):
        """Only existing compact contract values; no arrays, tensors or SDK objects."""
        if value is None or type(value) in (bool, int):
            return value
        if isinstance(value, str) and len(value) <= 256:
            return str(value)
        if type(value) is float and isfinite(value):
            return value
        if depth >= 10:
            raise ValueError('diagnostic nesting bound')
        if type(value) in (tuple, list) and len(value) <= 17:
            return tuple(_OriginalBodyDiagnostic._owned(v, depth+1) for v in value)
        names = (tuple(f.name for f in fields(value)) if is_dataclass(value)
                 else tuple(type(value).model_fields) if hasattr(type(value), 'model_fields') else ())
        if names and len(names) <= 32:
            return {n:_OriginalBodyDiagnostic._owned(getattr(value,n),depth+1) for n in names}
        raise ValueError('diagnostic noncompact or unowned value')

    def observe(self, row, stamp, camera, ingress_monotonic_s):
        if not self.targets:
            return
        try:
            original = row.original
            if original.pts_ns <= 0:
                return
            media = original.pts_ns / 1e9
            matches = [(i,t) for i,t in enumerate(self.targets) if abs(media-t) <= .100000001]
            if not matches:
                return
            life = row.evidence.lifecycle
            # Diagnostic preference only. Never infer a current original from
            # cached track scalars, and never change the actual selected row.
            receipt=row.evidence.body_depth;body=row.evidence.body
            preferred=False
            if row.denial is None and row.depth_current is True and receipt is not None and body is not None:
                try:
                    from noesis.pipelines.world_quality_adapter import OriginalBodyDepth
                    cohort,patch=receipt.cohort,receipt.patch
                    preferred=(type(receipt) is OriginalBodyDepth and body.lifecycle==life
                        and (cohort.source_id,cohort.camera_id,cohort.tracker_id,cohort.tracker_lifecycle_generation,cohort.frame_id)==
                            (original.source_id,original.camera_id,life.tracker_id,life.generation,original.original_frame)
                        and (stamp.member.run_id,stamp.member.source_id,stamp.member.camera_id,stamp.member.source_epoch,
                            stamp.member.raw_id,stamp.member.lifecycle_generation,stamp.frame_id)==
                            (original.run_id,original.source_id,original.camera_id,original.source_clock_epoch,
                             life.tracker_id,life.generation,original.original_frame)
                        and stamp.media_time_s is not None and abs(stamp.media_time_s-media)<=1e-8
                        and (patch.measurement_frame_id,patch.measurement_ts_us)==(original.original_frame,original.pts_ns//1000)
                        and patch.image_size==body.image_size==camera.image_size and receipt.image_flips==(False,False)
                        and receipt.registration_status=='ok' and bool(receipt.registration_id)
                        and all(type(v) in (int,float) and isfinite(v) and v>0 for v in (receipt.raw_depth_m,receipt.registered_depth_m))
                        and len(receipt.original_point)==3 and all(isfinite(v) for v in receipt.original_point))
                except (ValueError,TypeError,AttributeError,OverflowError):
                    preferred=False
            prefix = (original.run_id,original.graph_id,original.source_id,
                      original.source_clock_epoch,life.tracker_id,life.generation)
            eligible = [(prefix+(i,),t) for i,t in matches
                        if (prefix+(i,) in self.records or len(self.records) < 64)
                        and (prefix+(i,) not in self.records or
                             (preferred, -abs(media-t)) >
                             (self.records[prefix+(i,)]['selection_current_registered_original'],
                              -self.records[prefix+(i,)]['distance_s']))]
            if not eligible:
                return
            evidence = row.evidence
            measurements = evidence.measurements
            payload = {
                'original':self._owned(original), 'stamp':self._owned(stamp),
                'lifecycle':self._owned(life), 'ingress_monotonic_s':ingress_monotonic_s,
                'pixel_coordinates':'original post-dewarper source raster; x right, y down; confidence score',
                'point_coordinates':'receipt original_point in cohort world_frame/revision; not a physical grant',
                'raster_id':camera.raster_id, 'camera_binding':self._owned(camera.source),
                'image_size':self._owned(camera.image_size),
                'body':self._owned(evidence.body),
                'pose_profile':None,  # Not present in owned OriginalBodySupport; never infer it.
                'measurement_cohort':self._owned(measurements.cohort) if measurements else None,
                'hypotheses':self._owned(measurements.hypotheses) if measurements else None,
                'body_depth':self._owned(evidence.body_depth),
                'anatomy_depths':self._owned(evidence.anatomy_depths),
                'depth_current':row.depth_current, 'current_measured':row.current_measured,
                'denial':row.denial,
                'sampling_disposition':self._sampling_snapshot(row),
                'selection_policy':'prefer_current_registered_body_depth_within_existing_window_else_nearest_original',
                'selection_current_registered_original':preferred,
                'selection_status':'current_registered_original_within_window' if preferred else 'closest_available_original_within_window',
            }
            for key,target in eligible:
                self.records[key] = dict(payload,target_media_s=target,distance_s=abs(media-target))
                self._sampling_rows[key]=row
        except (ValueError, TypeError, AttributeError, OverflowError):
            # Optional malformed inspection cannot reject a canonical cohort.
            self.errors += 1

    def summary(self):
        records=deepcopy(tuple(self.records.values()))
        for key,record in zip(self.records,records):
            latest=self._sampling_snapshot(self._sampling_rows[key])
            if latest.get('available') is True:
                record['sampling_disposition']=latest
                record['sampling_disposition_stage']='final_exact_original_lookup'
            else:
                record['sampling_disposition_stage']='captured_at_original_inspection'
                record['sampling_disposition_final_available']=False
        result={'enabled':bool(self.targets),'targets_s':self.targets,'limit':64,
                'errors':self.errors,'records':records}
        if self.admission_records or self.admission_omitted:
            result['target_admission']=self.admission_summary()
        return result


def projective_continuity_enabled(config):
    value = config.get('xc3d_projective_continuity')
    if value is None:
        return False
    from noesis.v3dt_assets import xc3d_world_quality_enabled
    if (type(value) is not dict or set(value) != {'contract','enabled'}
            or value.get('contract') != 'noesis.xc3d.projective_continuity.v1'
            or type(value.get('enabled')) is not bool):
        raise ValueError('invalid explicit projective-continuity contract')
    if value['enabled'] and not xc3d_world_quality_enabled(config):
        raise ValueError('projective continuity requires explicit world-quality baseline mode')
    return value['enabled']


def runtime_projective_policy(bindings, provider):
    """Freeze literal current K/(E*inverse(B))/P once outside frame callbacks.

    Do not remove the relative presentation scale or apply B twice. Every edge
    retains its source frame/revision and exact raster. Shared authored scene
    binding supplies only the explicitly named estimated image-support prior.
    """
    import numpy as np
    from noesis_core.contracts.scene_prior import ScenePriorWorldToScenePresentation
    sources = sorted((b for b in bindings if b.camera_id in {'kitchen','family-room'}),
                     key=lambda b: b.source_id)
    if len(sources) != 2:
        raise ValueError('projective exact Kitchen/Family bindings unavailable')
    presentations = provider.world_frame_presentations()
    rows = []
    for source in sources:
        snapshot = provider.snapshot(source.source_id, source.camera_id)
        contract = getattr(snapshot, 'frame_contract', None)
        if (contract is None or contract.source_frame.revision != source.calibration_revision
                or contract.transform_sha256 != source.transform_id):
            raise ValueError('projective original frame binding mismatch')
        presentation = ScenePriorWorldToScenePresentation.model_validate(
            presentations.get(contract.target_frame.revision))
        if (not presentation.accepted or source.transform_id not in presentation.source_transform_sha256s
                or presentation.world_frame.revision != contract.target_frame.revision
                or presentation.authored_scene_sha256 is None):
            raise ValueError('projective explicit authored-scene prior unavailable')
        size = tuple(snapshot.image_size)
        K = np.asarray(snapshot.intrinsics, dtype=float)
        E = np.asarray(snapshot.world_extrinsics_col_major, dtype=float).reshape(4,4,order='F')
        P = np.asarray(presentation.world_to_scene_col_major, dtype=float).reshape(4,4,order='F')
        if (K.shape != (3,3) or not all(np.isfinite(m).all() for m in (K,E,P))
                or len(size) != 2 or any(type(v) is not int or not 8 < v <= 32768 for v in size)
                or K[0,0] <= 0 or K[1,1] <= 0 or not np.allclose(K[2], (0,0,1))):
            raise ValueError('projective literal original pinhole raster unavailable')
        raster = sha256(repr((source, size, tuple(K.reshape(-1)),
                             snapshot.camera_calibration_sha256)).encode()).hexdigest()
        rows.append((source,snapshot,contract,presentation,size,K,E,P,raster))
    if rows[0][3].authored_scene_sha256 != rows[1][3].authored_scene_sha256:
        raise ValueError('projective priors name different authored scenes')
    prior_id = 'review-prior:' + sha256(repr(tuple((r[0],r[3].world_to_scene_sha256,r[8])
                                                  for r in rows)).encode()).hexdigest()
    result = []
    freeze = lambda matrix: tuple(tuple(float(v) for v in row) for row in matrix)
    for i, row in enumerate(rows):
        source,snapshot,contract,presentation,size,K,E,P,raster = row
        peer = rows[1-i]
        own_projection = K @ E[:3]
        peer_projection = peer[5] @ (peer[6] @ np.linalg.inv(peer[7]) @ P)[:3]
        result.append(ProjectiveCameraBinding(source,prior_id,raster,size,
            contract.target_frame.frame_id,contract.target_frame.revision,
            snapshot.camera_calibration_sha256,freeze(own_projection),freeze(peer_projection)))
    return ProjectiveContinuityPolicy(tuple(result))


def sample_from_quality(row, stamp, camera, ingress_monotonic_s, *, joint_threshold, diagnostics=None):
    """Original association cues do not depend on the selected World output."""
    body = _original_body_depth(row, stamp, camera, ingress_monotonic_s,
        joint_threshold=joint_threshold, diagnostics=diagnostics)
    if body is not None:
        return body
    sample = _original_ankle_hypothesis(row,stamp,camera,ingress_monotonic_s,
        joint_threshold=joint_threshold,diagnostics=diagnostics)
    if sample is not None and not row.current_measured and diagnostics is not None:
        diagnostics['original_floor_independent_of_selected_World'] += 1
    return sample


def _original_body_depth(row, stamp, camera, ingress_monotonic_s, *, joint_threshold, diagnostics=None):
    """Genuine current unflattened body point and SAME sampled patch enclosure.

    This provides only estimated correspondence evidence. It never changes the
    selected/held/absent World row or creates uncertainty, a grant, metric/time
    authority, or hidden anatomy. All old numeric/appearance gates still apply.
    """
    def unavailable(reason):
        if diagnostics is not None:
            diagnostics['body_depth:' + reason] += 1
        return None
    receipt, body, measurements, depth = (row.evidence.body_depth, row.evidence.body,
                                         row.evidence.measurements, row.evidence.depth)
    if receipt is None:
        if diagnostics is not None:
            diagnostics['body_depth:absent_with_native_support' if depth is not None and depth.native_pose_support
                        else 'body_depth:absent_without_native_support'] += 1
            diagnostics['body_depth:exact_current_rows'] += row.depth_current
        return unavailable('receipt_absent')
    if row.denial is not None or not row.depth_current or body is None or measurements is None:
        return unavailable('integrity_currentness_or_body')
    cohort, patch = measurements.cohort, receipt.patch
    if (receipt.cohort != cohort or body.lifecycle != row.evidence.lifecycle
            or body.image_size != camera.image_size or patch.image_size != camera.image_size
            or receipt.image_flips != (False, False)
            or (cohort.source_id,cohort.camera_id,cohort.tracker_id,cohort.tracker_lifecycle_generation,
                cohort.frame_id,cohort.world_frame,cohort.world_revision,cohort.calibration_revision,
                cohort.world_transform_sha256) != (
                stamp.member.source_id,stamp.member.camera_id,stamp.member.raw_id,
                stamp.member.lifecycle_generation,stamp.frame_id,camera.target_frame,camera.world_revision,
                camera.measurement_calibration_sha256,camera.source.transform_id)
            or (patch.measurement_frame_id,patch.measurement_ts_us) !=
                (row.original.original_frame,row.original.pts_ns//1000)
            or patch not in depth.native_pose_support):
        return unavailable('original_binding')
    required = (5,6,11,12) if patch.anchor_source == 'pose_torso_support' else (5,6)
    if any(body.keypoints[i][2] < joint_threshold for i in required):
        return unavailable('current_anatomy')
    accepted = [s for s in depth.pose_depth_samples if s.anchor_source == patch.anchor_source
                and s.anchor_uv == patch.anchor_uv and s.evidence_reason is None
                and s.evidence_quality != 'rejected']
    point = receipt.original_point
    if (len(accepted) != 1 or accepted[0].depth_median != receipt.raw_depth_m
            or receipt.registration_status not in {'ok','raw_passthrough'}
            or receipt.registration_status == 'ok' and not receipt.registration_id
            or len(point) != 3 or not all(isfinite(v) for v in point)
            or not all(isfinite(v) and v > 0 for v in (receipt.raw_depth_m,receipt.registered_depth_m))
            or not isinstance(ingress_monotonic_s,(int,float)) or not isfinite(ingress_monotonic_s)
            or ingress_monotonic_s < 0):
        return unavailable('registration_or_range')
    radius = patch.enclosing_radius_px()
    identity = sha256(repr((row.original, body, receipt)).encode()).hexdigest()
    binding = ProjectiveMeasurementBinding('original-body-depth:'+identity,
        cohort.world_frame,cohort.world_revision,cohort.calibration_revision,
        cohort.world_transform_sha256,'original-registered-body:'+identity)
    sample = ProjectiveMotionSample(stamp,float(ingress_monotonic_s),'original-body:'+identity,
        camera.raster_id,body.image_size,body.bbox_xywh,point,binding,patch.anchor_uv,None,
        patch.anchor_source,radius)
    if not PersonAssociationLedger._body_contains(sample, patch.anchor_uv):
        return unavailable('body_patch_outside_detection')
    if diagnostics is not None:
        diagnostics['body_depth:valid:' + patch.anchor_source] += 1
    return sample


def _original_ankle_hypothesis(row, stamp, camera, ingress_monotonic_s, *, joint_threshold, diagnostics=None):
    """Use the ORIGINAL observed-ankle floor hypothesis and its actual radius.

    The selected World representative can still use a different, better depth
    measurement. This original hypothesis is association evidence only. Missing
    current visible ankles or the existing hypothesis' pixel uncertainty refuses
    this cue; a detector bottom/filtered prediction is not substituted.
    """
    def unavailable(reason):
        if diagnostics is not None:
            diagnostics[reason] += 1
        return None

    body = row.evidence.body
    measurements = row.evidence.measurements
    if row.denial is not None:
        return unavailable('sample_original_integrity_denial')
    if body is None or measurements is None:
        return unavailable('sample_body_or_measurement_unavailable')
    if (type(ingress_monotonic_s) not in (int,float) or not isfinite(ingress_monotonic_s)
            or ingress_monotonic_s < 0 or body.image_size != camera.image_size
            or body.lifecycle != row.evidence.lifecycle):
        return unavailable('sample_original_body_or_ingress_binding')
    cohort = measurements.cohort
    if (cohort.source_id,cohort.camera_id,cohort.tracker_id,cohort.tracker_lifecycle_generation,
            cohort.frame_id,cohort.world_frame,cohort.world_revision,cohort.calibration_revision,
            cohort.world_transform_sha256) != (
            stamp.member.source_id,stamp.member.camera_id,stamp.member.raw_id,
            stamp.member.lifecycle_generation,stamp.frame_id,camera.target_frame,
            camera.world_revision,camera.measurement_calibration_sha256,camera.source.transform_id):
        return unavailable('sample_original_measurement_binding')
    candidates = [h for h in measurements.hypotheses if h.cohort == cohort and h.valid and h.kind == 'floor_ray'
        and h.anchor in {'pose_ankle_floor','pose_single_ankle_floor','pose_ankle_support'}
        and h.posture in {'standing','unknown'} and h.support_state == 'floor'
        and h.pixel_uncertainty_px is not None and h.pixel_uncertainty_px > 0]
    ankles = [body.keypoints[i] for i in (15,16) if body.keypoints[i][2] >= joint_threshold]
    if len(candidates) != 1:
        return unavailable('sample_original_ankle_floor_hypothesis_unavailable')
    if not ankles:
        return unavailable('sample_current_ankles_unavailable')
    hypothesis, = candidates
    # The native paired-floor producer uses the mean of exactly two observed
    # ankles; its single-floor producer uses exactly one. Depth support can
    # name a different UV not copied here, so do not guess that contact.
    required_ankles = {'pose_ankle_floor':2,'pose_single_ankle_floor':1}.get(hypothesis.anchor)
    if required_ankles is None or len(ankles) != required_ankles:
        return unavailable('sample_original_ankle_contact_unavailable')
    u = sum(v[0] for v in ankles)/len(ankles)
    v = sum(v[1] for v in ankles)/len(ankles)
    x,y,w,h = body.bbox_xywh
    if not x <= u <= x+w or not y <= v <= y+h:
        return unavailable('sample_current_ankles_outside_body')
    identity = sha256(repr((row.original,body,hypothesis)).encode()).hexdigest()
    measurement = ProjectiveMeasurementBinding('original-floor:'+identity,
        cohort.world_frame,cohort.world_revision,cohort.calibration_revision,
        cohort.world_transform_sha256,'original-floor-hypothesis:'+identity)
    return ProjectiveMotionSample(stamp,float(ingress_monotonic_s),'original-body:'+identity,
        camera.raster_id,body.image_size,body.bbox_xywh,
        (hypothesis.position.x,hypothesis.position.y,hypothesis.position.z),measurement,
        (u,v),float(hypothesis.pixel_uncertainty_px))


def _copy_image_region_decisions(value):
    """Bounded JSON-only copy of the owner's optional diagnostic, not authority."""
    if (type(value) is not dict
            or value.get('contract') != 'noesis.identity.image_region_decisions.v1'
            or value.get('limit') != 8 or type(value.get('evicted')) is not int
            or value['evicted'] < 0 or type(value.get('transitions')) is not list
            or len(value['transitions']) > 8):
        raise ValueError('invalid owner diagnostic envelope')
    for record in value['transitions']:
        if type(record) is not dict:
            raise ValueError('invalid owner transition')
        for key, limit in (('members',64),('pairs',128),('pre_components',64),('post_components',64)):
            if type(record.get(key)) is not list or len(record[key]) > limit:
                raise ValueError('owner transition bound exceeded')
    budget = [200000, 2 * 1024 * 1024]
    def copy(node, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 16:
            raise ValueError('owner diagnostic work bound exceeded')
        if node is None or type(node) in (bool,int):
            return node
        if type(node) is float and isfinite(node):
            return node
        if type(node) is str:
            budget[1] -= len(node)
            if len(node) > 4096 or budget[1] < 0:
                raise ValueError('owner diagnostic text bound exceeded')
            return node
        if type(node) is list and len(node) <= 128:
            return [copy(item,depth+1) for item in node]
        if type(node) is dict and len(node) <= 64 and all(type(k) is str for k in node):
            return {copy(k,depth+1):copy(v,depth+1) for k,v in node.items()}
        raise ValueError('owner diagnostic is not bounded JSON')
    return copy(value)


class ImageRegionQualityBridge:
    """Current owned image geometry -> the single public physical service."""

    def __init__(self, policy, cache, *, joint_threshold=.35):
        if type(policy) is not ProjectiveContinuityPolicy or joint_threshold != .35:
            raise ValueError('exact image-region policy and unchanged joint threshold required')
        self.policy,self._cache=policy,cache
        self.cameras={c.source.source_id:c for c in policy.cameras}
        self.ingresses,self.image_originals={},{}
        self.counts=Counter();self.sequence=0
        self.actor_id='image-region-quality:'+policy.cameras[0].prior_id
        self._body_diagnostic=_OriginalBodyDiagnostic.from_environment()
        self._image_region_decisions = {}
        self.owner_availability = {}  # <=2 SAME original prequeue source snapshots.
        self.appearances = {}  # <=64 exact-current immutable queued feature rows.
        self.native_boxes = {}  # <=64 exact CURRENT native extents; negative-only.
        self.stopped_sources = set()  # Actual terminal native events only, <=3/run.

    def stop_source(self, source_id):
        if type(source_id) is not int or source_id not in (0,1,2):
            raise ValueError('image_region_stop_source_binding')
        if source_id not in self.stopped_sources:
            self.stopped_sources.add(source_id)
            self.counts['source_stops'] += 1
        self.owner_availability.pop(source_id,None)
        self.image_originals={m:v for m,v in self.image_originals.items() if m.source_id!=source_id}
        self.appearances={m:v for m,v in self.appearances.items() if m.source_id!=source_id}
        self.native_boxes={m:v for m,v in self.native_boxes.items() if m.source_id!=source_id}

    @property
    def cache(self):
        # Selected original features are bound only by the successful FIFO
        # owner. Reuse retains their original stamp/ingress, never last-seen ID.
        return self._cache() if callable(self._cache) else self._cache

    def ingest_historical_quality(self, *args):
        return None  # R19 strict CURRENT mode; no copied/renewed old evidence.

    def step(self, plans, originals, context, identity):
        if self._body_diagnostic.targets:
            try:identity.ledger.arm_image_region_support_diagnostic()
            except Exception:pass
        now=time.monotonic();self.sequence+=1
        times={i.binding.source_id:i.evaluation_max_s for i in context.intervals}
        current={m:o for m,o in originals.items()
                 if 0<=times.get(m.source_id,-1)-o.stamp.acquired_at_s<=.35}
        self.ingresses={m:v for m,v in self.ingresses.items()
            if m in originals and originals[m].stamp==v[0]}
        self.appearances={m:v for m,v in self.appearances.items() if m in current
            and v.appearance.stamp.member==current[m].stamp.member
            and v.appearance.stamp.lifecycle_started_frame==current[m].stamp.lifecycle_started_frame
            and v.appearance.stamp.frame_id<=current[m].stamp.frame_id
            and 0<=current[m].stamp.acquired_at_s-v.appearance.acquired_at_s<=.35
            and 0<=now-v.ingress_monotonic_s<=.35}
        self.image_originals={m:v for m,v in self.image_originals.items() if m in current
            and v.stamp==current[m].stamp and 0<=now-v.ingress_monotonic_s<=.35}
        self.native_boxes={m:v for m,v in self.native_boxes.items() if m in current
            and v.stamp==current[m].stamp and 0<=now-v.ingress_monotonic_s<=.35}
        box_overflow=False
        for work,_binding,_watermark,_retained,_tracks,observations in plans:
            if work.continuity.source_id in self.stopped_sources:
                # Preserve actual delivery custody for queued canonical rows so
                # the explicit STOP reason, not a forged missing receipt, denies
                # their vote. This supplies no body/join/availability evidence.
                for observation in observations:
                    self.ingresses[observation.stamp.member]=(observation.stamp,work.ingress_monotonic_s)
                continue  # Queued public rows remain; never fresh private join evidence.
            if work.continuity.source_id in self.cameras:
                self.owner_availability[work.continuity.source_id] = getattr(work,'owner_availability',None)
            camera=self.cameras.get(work.continuity.source_id)
            rows={(v.evidence.lifecycle.tracker_id,v.evidence.lifecycle.generation):v for v in work.rows}
            track_rows=(_tracks if type(_tracks) in (tuple,list) and len(_tracks)<=128
                and all(type(t) is dict for t in _tracks) else ())
            tracks={(t.get('tracker_id'),t.get('tracker_lifecycle_generation')):t for t in track_rows}
            if len(tracks)!=len(track_rows):tracks={}  # No ambiguous native extent.
            for observation in observations:
                stamp=observation.stamp;mono=work.ingress_monotonic_s
                self.ingresses[stamp.member]=(stamp,mono)
                self.image_originals.pop(stamp.member,None)
                self.appearances.pop(stamp.member,None)
                self.native_boxes.pop(stamp.member,None)
                if camera is None:continue
                track=tracks.get((stamp.member.raw_id,stamp.member.lifecycle_generation))
                if track is not None and not box_overflow:
                    box=track.get('bbox');size=track.get('image_size')
                    if (type(box) in (list,tuple) and len(box)==4
                            and all(type(v) in (int,float) and isfinite(v) for v in box)
                            and type(size) in (tuple,list) and tuple(size)==camera.image_size
                            and track.get('frame_id')==stamp.frame_id and track.get('source_epoch')==stamp.member.source_epoch):
                        self.native_boxes[stamp.member]=ImageRegionNativeBox(stamp,mono,camera.raster_id,
                            camera.prior_id,camera.image_size,tuple(float(v) for v in box))
                        if len(self.native_boxes)>64:
                            self.native_boxes.clear();box_overflow=True
                from noesis.pipelines.source_acquisition import OriginalSwinExtraction,OriginalSwinCache,CachedSwinUse
                values=work.swin_extractions
                typed=(type(values) is tuple and len(values)<=64
                    and all(type(x) is OriginalSwinExtraction for x in values))
                features=tuple(x for x in values if x.raw_id==stamp.member.raw_id) if typed else ()
                profile=identity.ledger.policy.appearance_profile
                if typed and len(features)==1 and len(self.appearances)<64:
                    x=features[0];profile=identity.ledger.policy.appearance_profile
                    if (x.original==work.private.original and x.profile==profile
                            and type(x.vector) is tuple and len(x.vector)==profile.dimensions<=256
                            and all(type(v) is float and isfinite(v) for v in x.vector) and any(x.vector)
                            and type(x.ingress_monotonic_s) in (int,float) and isfinite(x.ingress_monotonic_s)
                            and 0<=x.ingress_monotonic_s<=mono<=now and now-x.ingress_monotonic_s<=.35):
                        token=appearance_extraction_identity(stamp,profile)
                        a=AppearanceEvidence(stamp,token,profile.model_id,stamp.acquired_at_s,stamp.clock_domain,token,profile)
                        self.appearances[stamp.member]=ImageRegionAppearance(a,x.ingress_monotonic_s,x.vector)
                elif typed and not features and len(self.appearances)<64:
                    # Only genuine absence can reuse a qualified ORIGINAL F.
                    # Invalid/duplicate/stale supplied F must never be hidden.
                    try:cache=self.cache
                    except Exception:cache=None  # Optional selected evidence, no alternate source.
                    original=work.private.original
                    if (type(cache) is OriginalSwinCache
                            and (cache.run_id,cache.graph_id)==(original.run_id,original.graph_id)):
                        try:use=cache.use(observation,AcquisitionTime(stamp.acquired_at_s,stamp.clock_domain,stamp.clock_basis))
                        except Exception:use=None
                        if type(use) is CachedSwinUse and type(use.appearance) is AppearanceEvidence:
                            a=use.appearance;ingress=use.appearance_ingress_monotonic_s
                            if (a.profile==profile and a.model==profile.model_id and a.stamp.member==stamp.member
                                    and a.stamp.lifecycle_started_frame==stamp.lifecycle_started_frame
                                    and a.stamp.frame_id<=stamp.frame_id
                                    and type(use.vector) is tuple and len(use.vector)==profile.dimensions<=256
                                    and all(type(v) is float and isfinite(v) for v in use.vector) and any(use.vector)
                                    and type(ingress) in (int,float) and isfinite(ingress)
                                    and 0<=ingress<=mono<=now and now-ingress<=.35):
                                self.appearances[stamp.member]=ImageRegionAppearance(a,ingress,use.vector)
                row=rows.get((stamp.member.raw_id,stamp.member.lifecycle_generation))
                if row is None:continue
                self._body_diagnostic.observe(row,stamp,camera,mono)
                body=row.evidence.body
                if (row.denial is not None or body is None or body.lifecycle!=row.evidence.lifecycle
                        or body.image_size!=camera.image_size
                        or row.original.original_frame!=stamp.frame_id
                        or abs(row.original.pts_ns/1e9-stamp.media_time_s)>1e-8):
                    self.counts['body_original_unavailable']+=1;continue
                patches=full_observed_image_regions(body)
                if not patches:
                    self.counts['observed_region_unavailable']+=1;continue
                if len(self.image_originals)>=64:
                    self.counts['region_capacity']+=1;continue
                token=sha256(repr((row.original,body,patches,camera.prior_id)).encode()).hexdigest()
                joints=tuple((i,float(body.keypoints[i][0]),float(body.keypoints[i][1]),float(body.keypoints[i][2]))
                    for i in (5,6,11,12) if len(body.keypoints[i])==3 and all(isfinite(v) for v in body.keypoints[i])
                        and .35<=body.keypoints[i][2]<=1. and 0<=body.keypoints[i][0]<body.image_size[0]
                        and 0<=body.keypoints[i][1]<body.image_size[1]) if len(body.keypoints)==17 else ()
                self.image_originals[stamp.member]=ImageRegionOriginal(stamp,mono,'image-body:'+token,
                    camera.raster_id,camera.prior_id,body.image_size,body.bbox_xywh,patches,joints,
                    tuple((i,float(body.keypoints[i][0]),float(body.keypoints[i][1]),float(body.keypoints[i][2]))
                        for i in range(5) if len(body.keypoints[i])==3 and all(isfinite(v) for v in body.keypoints[i])
                            and .35<=body.keypoints[i][2]<=1. and 0<=body.keypoints[i][0]<body.image_size[0]
                            and 0<=body.keypoints[i][1]<body.image_size[1]))
                self.counts['current_region_originals']+=1
        # Participant truth is owner-classified lifecycle, not body freshness.
        # Stale unclassified ACTIVE members remain UNKNOWN competitors; an
        # actual retained-unmeasured receipt is the only new exclusion basis.
        stamps=identity.image_region_active_cohort()
        if (type(stamps) is not tuple or len(stamps)>128
                or any(type(s) is not ObservationStamp or s.member.source_id not in self.cameras
                    or s.member not in originals or originals[s.member].stamp!=s for s in stamps)
                or len({s.member for s in stamps})!=len(stamps)):
            raise ValueError('image_region_owner_cohort_binding')
        delivery=tuple(CurrentVisualIngress(stamp,mono) for member,(stamp,mono) in self.ingresses.items()
            if member in current and current[member].stamp==stamp)
        availability=[]
        # Optional custody has a tighter bound than the full participant cohort.
        # Over capacity clears ALL custody inputs, never truncates competitors.
        # Custody includes exact retained originals, not just current geometry
        # participants. A native SHADOW supplies neither body nor a fresh vote.
        for member,observation in originals.items() if len(stamps)<=64 else ():
            if member.source_id not in self.cameras:continue
            stamp=observation.stamp
            frozen=self.owner_availability.get(stamp.member.source_id)
            ingress=self.ingresses.get(stamp.member)
            if frozen is None or ingress is None or ingress[0]!=stamp:continue
            owner=frozen.owner; original=frozen.original; member=stamp.member
            # Termination carries raw IDs only: withhold every generation of
            # THAT raw ID, not unrelated exact retained members of the source.
            if (member.raw_id in owner.terminated or original.run_id!=member.run_id or owner.source_epoch!=member.source_epoch
                    or original.camera_id!=member.camera_id or owner.frame_id<stamp.frame_id
                    or len(owner.keys)>64
                    or (member.source_id,member.source_epoch,member.raw_id,member.lifecycle_generation) not in owner.keys):continue
            key=lambda raw,generation:MemberKey(member.run_id,member.camera_id,member.source_id,
                member.source_epoch,raw,generation)
            retained=tuple(key(row[2],row[3]) for row in owner.keys)
            active=tuple(key(row[0],row[1]) for row in owner.active)
            shadow=tuple(key(row[0],row[1]) for row in owner.shadow)
            availability.append(ImageRegionOwnerAvailability(stamp,original.graph_id,owner.frame_id,
                owner.media_pts_ns/1e9,owner.observed_at_us,ingress[1],owner.owner_grace_ns,
                retained,active,shadow,(),frozen.checked_monotonic_s,frozen.last_progress_age_s,
                frozen.stall_seconds,frozen.status,frozen.ingress_monotonic_s))
            if len(availability)>64:
                availability.clear()  # Whole optional proof fails closed, no truncation.
                break
        proof=EstimatedImageRegionProof('image-regions:'+str(self.sequence),stamps,
            tuple(self.image_originals[s.member] for s in stamps if s.member in self.image_originals),tuple(availability),
            tuple(self.appearances[s.member] for s in stamps if s.member in self.appearances) if len(stamps)<=64 else (),
            tuple(self.native_boxes[s.member] for s in stamps if s.member in self.native_boxes)
                if len(stamps)<=64 and not box_overflow else ())
        self.counts['owner_availability_capsules'] += len(availability)
        result=identity.join_estimated_image_regions(proof,at_interval=context,
            delivery_context=CurrentVisualDeliveryContext(self.actor_id,self.sequence,now,delivery))
        try:
            if any(abs(work.private.media_pts_ns/1e9-target)<=.100000001
                    for work,*_ in plans for target in self._body_diagnostic.targets):
                known={v.appearance.stamp.member:v for v in proof.appearances}
                pairs=[];pair_count=0
                for a in stamps:
                    for b in stamps:
                        if a.member.camera_id!='kitchen' or b.member.camera_id!='family-room':continue
                        pair_count+=1
                        if len(pairs)>=16:continue
                        x,y=known.get(a.member),known.get(b.member);score=None
                        if x is not None and y is not None:
                            norm=sum(v*v for v in x.vector)*sum(v*v for v in y.vector)
                            if isfinite(norm) and norm>0:
                                value=sum(v*w for v,w in zip(x.vector,y.vector))/sqrt(norm)
                                if isfinite(value):score=max(-1.,min(1.,value))
                        pairs.append({'left':[a.member.source_id,a.member.raw_id,a.member.lifecycle_generation,a.frame_id],
                            'right':[b.member.source_id,b.member.raw_id,b.member.lifecycle_generation,b.frame_id],
                            'cosine':score,'threshold_pass':score>=identity.ledger.policy.min_similarity if score is not None else None})
                info={'complete':len(known)==len(stamps)<=64,'participants':[
                    {'source':s.member.source_id,'raw':s.member.raw_id,'generation':s.member.lifecycle_generation,
                     'frame':s.frame_id,'exact_current_selected_compatible':s.member in known and known[s.member].appearance.stamp==s,
                     'qualified_original_selected_compatible':s.member in known,
                     'feature_original_frame':known[s.member].appearance.stamp.frame_id if s.member in known else None,
                     'feature_original_media_s':known[s.member].appearance.stamp.media_time_s if s.member in known else None,
                     'feature_original_ingress_monotonic_s':known[s.member].ingress_monotonic_s if s.member in known else None,
                     'feature_source_age_s':s.acquired_at_s-known[s.member].appearance.acquired_at_s if s.member in known else None,
                     'feature_availability_age_s':now-known[s.member].ingress_monotonic_s if s.member in known else None,
                     'observation_id':s.observation_id,'birth':s.lifecycle_started_frame,'media_s':s.media_time_s,
                     'current_body_original':next(({'frame':o.stamp.frame_id,'media_s':o.stamp.media_time_s,
                         'ingress_monotonic_s':o.ingress_monotonic_s,'semantic_joints':o.semantic_joints,
                         'head_joints':o.head_joints}
                         for o in proof.originals if o.stamp==s),None),
                     'reason':'available' if s.member in known else 'exact_current_selected_feature_unavailable_or_incompatible'} for s in stamps[:64]],
                    'participants_omitted':max(0,len(stamps)-64),'pairs':pairs,'pairs_omitted':max(0,pair_count-len(pairs)),
                    'join_disposition':result.disposition,'join_reason':result.reason}
                # Fill only at the NEXT real post-join membership issuance,
                # never by re-running eligibility from this observer. The
                # existing target row owns the finite scalar sink throughout.
                sources={work.continuity.source_id for work,*_ in plans}
                try:
                    info['group_support']=identity.ledger.arm_image_region_support_diagnostic(
                        tuple(s for s in stamps if s.member.source_id in sources))
                except Exception:
                    info['group_support']={'available':False,'reason':'diagnostic_unavailable'}
                self._body_diagnostic.observe_admission(plans,stamps,availability,result,originals,appearance_info=info)
        except Exception:pass  # Optional observation cannot alter the completed join.
        # Optional diagnostics cannot change the completed join or native liveness.
        try:
            accessor = getattr(identity,'image_region_decision_diagnostics',None)
            if not callable(accessor):
                self._image_region_decisions = {'available':False,'reason':'owner_accessor_unavailable'}
            else:
                from noesis.identity_association_service import IdentityAssociationService
                snapshot = getattr(identity, 'image_region_decision_snapshot', None)
                value = snapshot() if type(identity) is IdentityAssociationService and callable(snapshot) else accessor()
                if type(value) is ImageRegionDecisionSnapshot:
                    value.check_immutable()
                    self._image_region_decisions = value
                else:
                    self._image_region_decisions = _copy_image_region_decisions(value)
        except Exception:
            self._image_region_decisions = {'available':False,'reason':'owner_diagnostic_unavailable'}
        key='result:'+result.reason
        self.counts[key if key in self.counts or len(self.counts)<64 else 'other_reason']+=1
        return result

    def diagnostics(self):
        try:
            decisions = (_copy_image_region_decisions(self._image_region_decisions.materialize())
                if type(self._image_region_decisions) is ImageRegionDecisionSnapshot
                else deepcopy(self._image_region_decisions))
        except Exception:
            decisions = {'available':False,'reason':'owner_diagnostic_unavailable'}
        return {'basis':'estimated_current_image_regions','counts':dict(self.counts),
                'current_region_originals':len(self.image_originals),
                'image_region_decisions':decisions,
                'original_body_inspection_only':self._body_diagnostic.summary()}


class ProjectiveQualityBridge:
    """One FIFO actor; <=128 decision +128 diagnostic originals, no person ledger."""

    def __init__(self, policy, cache, *, joint_threshold):
        from noesis.pipelines.source_acquisition import OriginalSwinCache
        if (type(policy) is not ProjectiveContinuityPolicy or not (type(cache) is OriginalSwinCache or callable(cache))
                or type(joint_threshold) not in (int,float) or not isfinite(joint_threshold)
                or not 0 < joint_threshold <= 1):
            raise ValueError('projective actual policy/cache/pose threshold required')
        self.policy,self._cache,self.joint_threshold = policy,cache,joint_threshold
        self.cameras = {c.source.source_id:c for c in policy.cameras}
        self.history,self.ingresses = {},{}
        self.anatomical_originals = {}  # <=64 members /128 original anatomy supports.
        self.sequence = 0
        self.counts = Counter()
        self.numeric_counts = Counter()
        self.numeric_examples = {}  # <=16 aggregate diagnostic examples, no decisions.
        self.hypothesis_history = {}  # Separate <=128 diagnostic originals; never selected.
        self.hypothesis_counts = Counter()
        self.hypothesis_numeric_counts = Counter()
        self.hypothesis_numeric_examples = {}
        self.hypothesis_origins = {}  # <=8 scalar examples of representative coupling.
        self.cue_history = {}  # <=128 total diagnostic originals, <=16/member/basis.
        self.pose_history = {}  # <=128 current-pose stamps, NO synthetic 3D point.
        self.history_diagnosis = Counter()
        self.history_gap_examples = {}  # <=16 original-gap examples, <=6 scalars/side.
        self.last_cue_original = {}  # <=128 diagnostics-only original endpoints.
        self.cue_numeric_counts = {basis:Counter() for basis in (
            'observed_ankle_floor','pose_torso_support','pose_shoulder_support')}
        self.cue_numeric_examples = {basis:{} for basis in self.cue_numeric_counts}
        self.actor_id = 'projective-quality:' + policy.cameras[0].prior_id
        self._body_diagnostic = _OriginalBodyDiagnostic.from_environment()

    def _retain_anatomical_row(self, row, stamp, camera, mono):
        values = getattr(row.evidence, 'anatomy_depths', ())
        body = row.evidence.body
        if not values or body is None or row.denial is not None or not row.depth_current:
            return
        if (len(values) != 2 or tuple(a.sample.anchor_source for a in values) !=
                ('pose_ankle_support', 'pose_torso_support')):
            self.counts['anatomical_original_tuple_unavailable'] += 1
            return
        supports = []
        for item in values:
            receipt, sample = item.receipt, item.sample
            patch, cohort = receipt.patch, receipt.cohort
            if (body.lifecycle != row.evidence.lifecycle or body.image_size != camera.image_size
                    or patch.image_size != camera.image_size or receipt.image_flips != (False, False)
                    or receipt.registration_status not in {'ok','raw_passthrough'}
                    or receipt.registration_status == 'ok' and not receipt.registration_id
                    or receipt.registration_status == 'raw_passthrough' and item.registered_depth_interval_m is not None
                    or (patch.measurement_frame_id, patch.measurement_ts_us) !=
                        (row.original.original_frame, row.original.pts_ns // 1000)
                    or (cohort.source_id,cohort.camera_id,cohort.tracker_id,cohort.tracker_lifecycle_generation,
                        cohort.frame_id,cohort.world_frame,cohort.world_revision,cohort.calibration_revision,
                        cohort.world_transform_sha256) != (stamp.member.source_id,stamp.member.camera_id,
                        stamp.member.raw_id,stamp.member.lifecycle_generation,row.original.original_frame,
                        camera.target_frame,camera.world_revision,camera.measurement_calibration_sha256,camera.source.transform_id)):
                self.counts['anatomical_original_binding_unavailable'] += 1
                return
            token = sha256(repr((row.original,body,item)).encode()).hexdigest()
            measurement = ProjectiveMeasurementBinding('original-anatomy:'+token,cohort.world_frame,
                cohort.world_revision,cohort.calibration_revision,cohort.world_transform_sha256,'original-anatomy-stage:'+token)
            original = ProjectiveMotionSample(stamp,mono,'original-anatomy-body:'+token,camera.raster_id,
                body.image_size,body.bbox_xywh,receipt.original_point,measurement,patch.anchor_uv,None,
                sample.anchor_source,None)
            supports.append(AnatomicalDepthSupport(original,tuple(tuple(c) for c in patch.capsules),
                item.registered_depth_interval_m,item.depth_convention))
        previous = self.anatomical_originals.get(stamp.member)
        frozen = tuple(supports)
        if previous is not None and (stamp.frame_id < previous[0].original.stamp.frame_id
                or stamp.frame_id == previous[0].original.stamp.frame_id and frozen != previous):
            self.counts['anatomical_original_retimestamp_or_rewind'] += 1
            return
        if stamp.member not in self.anatomical_originals and len(self.anatomical_originals) >= 64:
            self.counts['anatomical_original_capacity'] += 1
            return
        self.anatomical_originals[stamp.member] = frozen

    def ingest_historical_quality(self, works, current_originals, at_interval, evaluation_monotonic_s):
        """Retain ORIGINAL F only; never observe a member, ingress N, or issue a grant."""
        from noesis.pipelines.world_quality_adapter import QualityWork
        if (type(works) is not tuple or len(works) > 128 or sum(len(w.rows) for w in works if type(w) is QualityWork) > 128
                or type(evaluation_monotonic_s) not in (int,float) or not isfinite(evaluation_monotonic_s)):
            return 'anatomical_historical_capacity_or_type'
        times = {i.binding.source_id:i.evaluation_max_s for i in at_interval.intervals}
        for work in works:
            if type(work) is not QualityWork or work.denial is not None:
                continue
            original = work.private.original
            mono = work.ingress_monotonic_s
            camera = self.cameras.get(original.source_id)
            if (camera is None or type(mono) not in (int,float) or not 0 <= evaluation_monotonic_s-mono <= .35
                    or not 0 <= times.get(original.source_id,-1)-original.pts_ns/1e9 <= .35):
                continue
            for row in work.rows:
                life = row.evidence.lifecycle
                candidates = [o for k,o in current_originals.items() if (k.run_id,k.camera_id,k.source_id,
                    k.source_epoch,k.raw_id,k.lifecycle_generation) == (original.run_id,original.camera_id,
                    original.source_id,life.source_epoch,life.tracker_id,life.generation)]
                if len(candidates) != 1:
                    continue
                current = candidates[0].stamp
                if (not current.lifecycle_started_frame <= life.frame_id <= current.frame_id
                        or current.clock_domain != camera.source.clock_domain
                        or current.member.source_epoch != camera.source.source_epoch
                        or work.private.clock_epoch != original.source_clock_epoch
                        or current.media_time_s is None or original.pts_ns / 1e9 > current.media_time_s):
                    continue
                token = sha256(repr((original,current.member,current.lifecycle_started_frame)).encode()).hexdigest()
                stamp = ObservationStamp(current.member,life.frame_id,'native-quality:'+token,
                    original.pts_ns/1e9,current.clock_domain,original.pts_ns/1e9,current.clock_basis,current.lifecycle_started_frame)
                self._body_diagnostic.observe(row,stamp,camera,mono)
                self._retain_anatomical_row(row,stamp,camera,mono)
                sample = sample_from_quality(row,stamp,camera,mono,joint_threshold=self.joint_threshold)
                if sample is not None:
                    values = self.history.setdefault(stamp.member,[])
                    if not values or sample.stamp.frame_id > values[-1].stamp.frame_id:
                        if sum(map(len,self.history.values())) < 128:
                            values.append(sample)
                            del values[:-16]
        return None

    def _numeric_example(self, reason, left, right, a=(), b=(), *, _examples=None, **details):
        examples = self.numeric_examples if _examples is None else _examples
        if reason not in examples and len(examples) >= 16:
            return
        examples[reason] = {'endpoint_frames':(left.frame_id,right.frame_id),
            'original_frames':tuple(tuple(s.stamp.frame_id for s in side) for side in (a,b)),
            **details}

    @staticmethod
    def _original_details(histories, endpoints, delivery):
        """At most6 real original rows/side, source clocks never compared."""
        return tuple({'current_frame':endpoint.frame_id,'current_media_s':endpoint.media_time_s,
            'current_own_source_s':endpoint.acquired_at_s,
            'samples':tuple({'frame':s.stamp.frame_id,'media_s':s.stamp.media_time_s,
                'own_source_s':s.stamp.acquired_at_s,'prequeue_monotonic_s':s.ingress_monotonic_s,
                'availability_age_s':delivery.evaluation_monotonic_s-s.ingress_monotonic_s,
                'own_source_gap_s':endpoint.acquired_at_s-s.stamp.acquired_at_s,
                'anatomy':s.support_basis} for s in side[-6:])}
            for side,endpoint in zip(histories,endpoints))

    def _prune_originals(self, values, current, times, now, label):
        result={}
        for member,history in values.items():
            chains={}
            for sample in history:
                endpoint = current[member].stamp if member in current else None
                reason = ('lifecycle_or_source_unavailable' if member not in current else
                    'original_lifecycle_binding' if (sample.stamp.member != member
                        or sample.stamp.lifecycle_started_frame != endpoint.lifecycle_started_frame
                        or sample.stamp.clock_domain != endpoint.clock_domain
                        or sample.stamp.clock_basis != endpoint.clock_basis) else
                    'prequeue_age' if not 0 <= now-sample.ingress_monotonic_s <= .70 else
                    'own_source_age' if not 0 <= times.get(member.source_id,-1)-sample.stamp.acquired_at_s <= .70 else
                    'media_age' if sample.stamp.media_time_s is None or endpoint.media_time_s is None
                        or not 0 <= endpoint.media_time_s-sample.stamp.media_time_s <= .70 else None)
                if reason:
                    self.history_diagnosis[label+':pruned:'+member.camera_id+':'+reason] += 1
                    chains.pop(sample.support_basis,None)
                    continue
                chain=chains.setdefault(sample.support_basis,[])
                if chain:
                    previous=chain[-1]
                    if (not previous.stamp.frame_id < sample.stamp.frame_id
                            or not 0 < sample.stamp.acquired_at_s-previous.stamp.acquired_at_s <= .35
                            or not 0 < sample.stamp.media_time_s-previous.stamp.media_time_s <= .35
                            or not 0 < sample.ingress_monotonic_s-previous.ingress_monotonic_s <= .35):
                        self.history_diagnosis[label+':pruned:'+member.camera_id+':continuity_gap'] += len(chain)
                        chain.clear()
                chain.append(sample)
            # Only previously admitted originals may span two continuity gaps.
            # A fresh endpoint must still be reachable within the original TTL;
            # fresh ingestion and the kernel's observed native/delivery floors
            # remain separate and unchanged. Never reobserve/restamp a sample.
            kept=[s for chain in chains.values() if chain
                and 0 <= now-chain[-1].ingress_monotonic_s <= .35
                and 0 <= times.get(member.source_id,-1)-chain[-1].stamp.acquired_at_s <= .35
                and 0 <= current[member].stamp.media_time_s-chain[-1].stamp.media_time_s <= .35
                for s in chain]
            kept.sort(key=lambda s:s.stamp.frame_id)
            if kept:result[member]=kept
        return result

    def _record_original_choices(self,row,stamp,camera,mono,now,chosen):
        """Read-only diagnosis independent of selected measurement/cadence.

        Keep actual floor AND body alternatives so priority cannot conceal a
        complete witness. These histories never feed a real decision or grant.
        Current 2D pose stamps are counted separately, without constructing a
        missing range, support radius, 3D point or motion-acceptance assertion.
        """
        camera_id=stamp.member.camera_id
        body=_original_body_depth(row,stamp,camera,mono,joint_threshold=self.joint_threshold)
        floor=_original_ankle_hypothesis(row,stamp,camera,mono,joint_threshold=self.joint_threshold)
        cues=tuple(s for s in (floor,body) if s is not None)
        counter=self.history_diagnosis
        counter[camera_id+':current_rows']+=1
        counter[camera_id+':exact_current_depth_rows']+=row.depth_current
        counter[camera_id+':current_2d_pose_rows']+=row.evidence.body is not None
        counter[camera_id+':chosen:'+(chosen.support_basis if chosen is not None else 'unavailable')]+=1
        counter[camera_id+':both_floor_and_body_available']+=len(cues)==2
        counter[camera_id+':floor_discarded_by_body_priority']+=floor is not None and body is not None
        counter[camera_id+':selected_World_not_current']+=not row.current_measured
        depth=row.evidence.depth
        if depth is not None:
            counter[camera_id+':depth_cached']+=depth.measurement_cached is True
            counter[camera_id+':depth_measurement_original_frame_mismatch']+=depth.measurement_frame_id!=row.original.original_frame
            counter[camera_id+':depth_tensor_original_frame_mismatch']+=depth.depth_tensor_frame_id!=row.original.original_frame
        if row.evidence.body is not None:
            visible=tuple(basis for basis,indices in (('observed_ankle_floor',(15,16)),
                ('pose_torso_support',(5,6,11,12)),('pose_shoulder_support',(5,6)))
                if all(row.evidence.body.keypoints[i][2]>=self.joint_threshold for i in indices))
            history=self.pose_history.setdefault(stamp.member,[])
            if sum(map(len,self.pose_history.values()))<128:
                history.append((stamp,mono,visible,row.evidence.body.image_patches))
                del history[:-16]
            else:counter['pose_diagnostic_capacity']+=1
            for patch in row.evidence.body.image_patches:
                counter[camera_id+':current_2d_patch:'+patch.anatomy]+=1
        for sample in cues:
            history=self.cue_history.setdefault(stamp.member,[])
            key_original=(stamp.member,sample.support_basis)
            previous=self.last_cue_original.get(key_original)
            prior=previous[0] if previous is not None else None
            counter[camera_id+':'+sample.support_basis+':originals']+=1
            if prior is not None:
                gap=sample.stamp.acquired_at_s-prior.stamp.acquired_at_s
                bucket='within_window' if 0<gap<=.35 else 'outside_window'
                counter[camera_id+':'+sample.support_basis+':gap_'+bucket]+=1
                key=camera_id+':'+sample.support_basis+':'+bucket
                if key not in self.history_gap_examples and len(self.history_gap_examples)<16:
                    self.history_gap_examples[key]={'frames':(prior.stamp.frame_id,sample.stamp.frame_id),
                        'original_source_frames':(previous[1],row.original.original_frame),
                        'media_s':(prior.stamp.media_time_s,sample.stamp.media_time_s),
                        'own_source_s':(prior.stamp.acquired_at_s,sample.stamp.acquired_at_s),
                        'prequeue_monotonic_s':(prior.ingress_monotonic_s,sample.ingress_monotonic_s),
                        'availability_age_s':now-sample.ingress_monotonic_s}
            if key_original in self.last_cue_original or len(self.last_cue_original)<128:
                self.last_cue_original[key_original]=(sample,row.original.original_frame)
            if sum(map(len,self.cue_history.values()))<128:
                history.append(sample)
                same=[s for s in history if s.support_basis==sample.support_basis]
                if len(same)>16:history.remove(same[0])
            else:counter['cue_diagnostic_capacity']+=1

    def _inspect_original_motion(self, identity, context, delivery, visual, kitchens, families,
                                 appearances, coverage, *, _history=None, _counts=None, _examples=None):
        """Observe the existing numeric validator, NEVER a person decision.

        All current cross-view competitors are counted, including missing cues.
        A bounded diagnostic capacity skip cannot remove a decision competitor.
        No result feeds selected pairs, joining, reservations or World quality.
        """
        from reid.mv3dt_person_association import PersonAssociationLedger
        project = PersonAssociationLedger._project_pixel
        contains = PersonAssociationLedger._body_contains
        history = self.history if _history is None else _history
        counts = self.numeric_counts if _counts is None else _counts
        examples = self.numeric_examples if _examples is None else _examples
        count = len(kitchens)*len(families)
        counts['current_pair_slots'] += count
        if count > 21:
            counts['diagnostic_pair_capacity_cohorts'] += 1
            return
        for left in kitchens:
            for right in families:
                uses = tuple(appearances.get(s.member) for s in (left,right))
                appearance_info = {'coverage':tuple(coverage.get(s.member,'ingress_unavailable') for s in (left,right))}
                if all(uses):
                    x,y = (use.vector for use in uses)
                    cosine = max(-1.,min(1.,sum(a*b for a,b in zip(x,y))/sqrt(sum(a*a for a in x)*sum(b*b for b in y))))
                    appearance_info.update(cosine=cosine,
                        original_source_age_s=tuple(next(i.evaluation_max_s for i in context.intervals
                            if i.binding.source_id == s.member.source_id)-use.appearance.acquired_at_s for s,use in zip((left,right),uses)),
                        original_availability_age_s=tuple(delivery.evaluation_monotonic_s-use.appearance_ingress_monotonic_s for use in uses))
                    counts['raw_pair_similarity_pass' if cosine >= identity.ledger.policy.min_similarity else 'raw_pair_similarity_fail'] += 1
                    ingress = {i.observation.member:i for i in delivery.ingresses}
                    invalid_order = any(not 0 <= use.appearance_ingress_monotonic_s <= ingress[s.member].ingress_monotonic_s
                        for s,use in zip((left,right),uses))
                    expired = any(age > identity.ledger.policy.max_age_s for age in appearance_info['original_availability_age_s'])
                    counts['appearance_origin_invalid_order_pairs'] += invalid_order
                    counts['appearance_original_monotonic_expired_pairs'] += expired
                else:
                    counts['pair_missing_original_appearance'] += 1
                histories = tuple(history.get(s.member,()) for s in (left,right))
                current = tuple(bool(h and h[-1].stamp == s) for h,s in zip(histories,(left,right)))
                if not all(current):
                    counts['current_body_endpoint_unavailable'] += 1
                    counts['still_valid_older_three_originals_but_not_current_endpoint'] += all(len(h)>=3 for h in histories)
                    self._numeric_example('current_body_endpoint_unavailable',left,right,_examples=examples,
                        history_counts=tuple(map(len,histories)),current_endpoints=current,
                        original_details=self._original_details(histories,(left,right),delivery),
                        appearance=appearance_info)
                    continue
                a,b = (self._trajectory(s.member,_history=history) for s in (left,right))
                if a is None or b is None:
                    counts['simultaneous_three_originals_unavailable'] += 1
                    self._numeric_example('simultaneous_three_originals_unavailable',left,right,_examples=examples,
                        history_counts=tuple(map(len,histories)),
                        same_anatomy_counts=tuple(sum(s.support_basis==h[-1].support_basis for s in h) if h else 0 for h in histories),
                        original_details=self._original_details(histories,(left,right),delivery),appearance=appearance_info)
                    continue
                report = identity.inspect_projective_continuity(
                    ProjectiveContinuityProof(visual,(ProjectiveMotionPair(a,b),)),
                    at_interval=context,delivery_context=delivery)
                counts['complete_original_witnesses'] += 1
                reason = report.error or ('numeric_supported' if report.pairs and report.pairs[0][2]
                                          else 'numeric_contradiction')
                counts[reason] += 1
                # Descriptive residuals use the SAME pure projection/support
                # primitives. They do not replace the validator's answer.
                detail = Counter()
                own_errors,peer_errors = [],[]
                for side,peer,camera in ((a,b,self.cameras[left.member.source_id]),
                                         (b,a,self.cameras[right.member.source_id])):
                    own = [project(camera.own_projection,s.original_point) for s in side]
                    mapped = [project(camera.peer_projection,s.original_point) for s in side]
                    for s,p,o,q in zip(side,peer,own,mapped):
                        detail['own_support_misses'] += not contains(s,o)
                        detail['peer_support_misses'] += not contains(p,q)
                        detail['own_cheirality_or_nonfinite'] += o is None
                        detail['peer_cheirality_or_nonfinite'] += q is None
                        if o is not None:
                            own_errors.append(hypot(o[0]-s.observed_anchor_uv[0],o[1]-s.observed_anchor_uv[1]))
                        if q is not None:
                            peer_errors.append(hypot(q[0]-p.observed_anchor_uv[0],q[1]-p.observed_anchor_uv[1]))
                    for i in (1,2):
                        observed = tuple(side[i].observed_anchor_uv[j]-side[i-1].observed_anchor_uv[j] for j in (0,1))
                        detail['observed_degenerate_steps'] += hypot(*observed) <= _radius(side[i])+_radius(side[i-1])
                        if mapped[i] is not None and mapped[i-1] is not None:
                            motion = tuple(mapped[i][j]-mapped[i-1][j] for j in (0,1))
                            target = tuple(peer[i].observed_anchor_uv[j]-peer[i-1].observed_anchor_uv[j] for j in (0,1))
                            detail['projected_degenerate_steps'] += hypot(*motion) <= _radius(peer[i])+_radius(peer[i-1])
                            detail['projected_direction_mismatches'] += sum(x*y for x,y in zip(motion,target)) <= 0
                for name,value in detail.items():
                    counts[name] += value
                self._numeric_example(reason,left,right,a,b,_examples=examples,**dict(detail),appearance=appearance_info,
                    own_residual_px_range=(min(own_errors),max(own_errors)) if own_errors else None,
                    peer_residual_px_range=(min(peer_errors),max(peer_errors)) if peer_errors else None,
                    original_support_basis=tuple(sorted({s.support_basis for s in a+b})),
                    original_radius_px_range=(min(_radius(s) for s in a+b),max(_radius(s) for s in a+b)),
                    availability_age_s=tuple(tuple(delivery.evaluation_monotonic_s-s.ingress_monotonic_s for s in side) for side in (a,b)))

    @property
    def cache(self):
        from noesis.pipelines.source_acquisition import OriginalSwinCache
        result = self._cache() if callable(self._cache) else self._cache
        if type(result) is not OriginalSwinCache:
            raise ValueError('projective original selected Swin cache unavailable')
        return result

    def _trajectory(self, member, *, latest=False, _history=None):
        history = (self.history if _history is None else _history).get(member,())
        if history:
            history = [s for s in history if s.support_basis == history[-1].support_basis]
        if len(history) < 3:
            return None
        if latest:
            return tuple(history[-3:])
        last = history[-1]
        # At most16 retained originals/member; choose the widest actual support
        # span, then one middle original. Never synthesize/interpolate a sample.
        candidates = []
        for a,b in combinations(history[:-1],2):
            if all(0 < q.stamp.acquired_at_s-p.stamp.acquired_at_s <= .35
                   and 0 < q.stamp.media_time_s-p.stamp.media_time_s <= .35
                   and 0 < q.ingress_monotonic_s-p.ingress_monotonic_s <= .35
                   and hypot(q.observed_anchor_uv[0]-p.observed_anchor_uv[0],
                         q.observed_anchor_uv[1]-p.observed_anchor_uv[1]) >
                   _radius(p)+_radius(q) for p,q in ((a,b),(b,last))):
                candidates.append((a,b,last))
        return min(candidates,key=lambda rows:(rows[0].stamp.frame_id,
            abs(rows[1].ingress_monotonic_s-(rows[0].ingress_monotonic_s+last.ingress_monotonic_s)/2))) if candidates else tuple(history[-3:])

    @staticmethod
    def _image_motion_available(history,basis,endpoint):
        """Observed 2D disjoint-support motion ONLY; no inter-camera join proof."""
        rows=[(row,next(p for p in row[3] if p.anatomy==basis)) for row in history
              if any(p.anatomy==basis for p in row[3])]
        if len(rows)<3 or rows[-1][0][0]!=endpoint:
            return False,False
        last=rows[-1][1]
        moving=any(all(hypot(q.anchor_uv[0]-p.anchor_uv[0],q.anchor_uv[1]-p.anchor_uv[1])>
            p.support_radius_px+q.support_radius_px for p,q in ((a[1],b[1]),(b[1],last)))
            for a,b in combinations(rows[:-1],2))
        return True,moving

    def step(self, plans, originals, context, identity):
        now = time.monotonic()
        self.sequence += 1
        # The actual source callback's rows, not retained/grace last-seen state,
        # fence evidence across absence/reappearance BEFORE any historical use.
        for work,_binding,_watermark,_retained,_tracks,observations in plans:
            source = work.continuity.source_id
            present = {o.stamp.member for o in observations}
            reborn = {(r.evidence.lifecycle.tracker_id,r.evidence.lifecycle.generation) for r in work.rows
                      if r.evidence.lifecycle.disposition in {'new','reappeared'}}
            for store in (self.anatomical_originals,self.history,self.hypothesis_history,self.cue_history,self.pose_history):
                for key in tuple(store):
                    if key.source_id == source and (key not in present or (key.raw_id,key.lifecycle_generation) in reborn):
                        del store[key]
        times = {i.binding.source_id:i.evaluation_max_s for i in context.intervals}
        current = {m:o for m,o in originals.items()
            if 0 <= times.get(m.source_id,-1)-o.stamp.acquired_at_s <= .35}
        self.anatomical_originals = {k:v for k,v in self.anatomical_originals.items() if k in current
            and all(0 <= now-s.original.ingress_monotonic_s <= .35
                    and 0 <= times.get(k.source_id,-1)-s.original.stamp.acquired_at_s <= .35 for s in v)}
        self.ingresses = {m:v for m,v in self.ingresses.items() if m in current}
        self.history=self._prune_originals(self.history,current,times,now,'decision')
        self.hypothesis_history=self._prune_originals(self.hypothesis_history,current,times,now,'hypothesis')
        self.cue_history=self._prune_originals(self.cue_history,current,times,now,'all_cues')
        self.pose_history={m:[p for p in values if 0<=now-p[1]<=.35 and
            0<=times.get(m.source_id,-1)-p[0].acquired_at_s<=.35] for m,values in self.pose_history.items() if m in current}
        self.last_cue_original={key:s for key,s in self.last_cue_original.items() if key[0] in current}
        for work,_binding,_watermark,_retained,_tracks,observations in plans:
            camera = self.cameras.get(work.continuity.source_id)
            for observation in observations:
                self.ingresses[observation.stamp.member] = (observation.stamp,work.ingress_monotonic_s)
            if camera is None:
                continue
            by_member = {(r.evidence.lifecycle.tracker_id,r.evidence.lifecycle.generation):r for r in work.rows}
            for observation in observations:
                stamp = observation.stamp
                mono = work.ingress_monotonic_s
                self.ingresses[stamp.member] = (stamp,mono)
                row = by_member[(stamp.member.raw_id,stamp.member.lifecycle_generation)]
                self._body_diagnostic.observe(row,stamp,camera,mono)
                self._retain_anatomical_row(row,stamp,camera,mono)
                sample = sample_from_quality(row,stamp,camera,mono,joint_threshold=self.joint_threshold,
                    diagnostics=self.counts)
                self._record_original_choices(row,stamp,camera,mono,now,sample)
                # Diagnose original hypothesis provenance independently of a
                # held/absent/different selected World representative. This
                # separate ring is NEVER an input to the actual join proof.
                original = sample or _original_ankle_hypothesis(row,stamp,camera,mono,
                    joint_threshold=self.joint_threshold,diagnostics=self.hypothesis_counts)
                if original is not None:
                    self.hypothesis_counts[camera.source.camera_id+':valid_original'] += 1
                    if not row.current_measured:
                        self.hypothesis_counts[camera.source.camera_id+':solely_representative_gate'] += 1
                        fields = dict(row.evidence.world)
                        resolution = row.evidence.resolution
                        key = (camera.source.camera_id,getattr(resolution,'status',None),fields.get('world_source'))
                        if key in self.hypothesis_origins or len(self.hypothesis_origins) < 8:
                            self.hypothesis_origins[key] = {'camera':camera.source.camera_id,'frame':stamp.frame_id,
                                'resolver_status':getattr(resolution,'status',None),
                                'resolver_kind':getattr(resolution,'selected_kind',None),
                                'world_source':fields.get('world_source'),
                                'world_valid':fields.get('world_valid'),
                                'world_measurement_accepted':fields.get('world_measurement_accepted')}
                    values = self.hypothesis_history.setdefault(stamp.member,[])
                    if not values or values[-1].stamp != stamp:
                        if sum(map(len,self.hypothesis_history.values())) < 128:
                            values.append(original)
                            del values[:-16]
                        else:
                            self.hypothesis_counts['diagnostic_sample_capacity'] += 1
                self.counts['current_body_rows' if row.evidence.body is not None else 'missing_body_rows'] += 1
                if sample is None:
                    self.counts['unsupported_original_sample'] += 1
                    continue
                history = self.history.setdefault(stamp.member,[])
                if not history or history[-1].stamp != stamp:
                    if sum(len(v) for v in self.history.values()) >= 128:
                        self.counts['sample_capacity'] += 1
                    else:
                        history.append(sample)
                        del history[:-16]
                        self.counts['original_samples'] += 1
        self.history = {m:v for m,v in self.history.items() if v}
        self.hypothesis_history = {m:v for m,v in self.hypothesis_history.items() if v}
        stamps = tuple(sorted((o.stamp for m,o in current.items() if m.source_id in self.cameras
            and (m not in self.ingresses or type(self.ingresses[m][1]) not in (int,float)
                 or now-self.ingresses[m][1] <= .35)),key=lambda s:s.member))
        appearances,delivery,coverage = {},[],{}
        missing_ingress = False
        for stamp in (o.stamp for o in current.values()):
            row = self.ingresses.get(stamp.member)
            if row is None or row[0] != stamp or type(row[1]) not in (int,float) or not isfinite(row[1]) or not 0 <= row[1] <= now:
                missing_ingress = True
                continue
            cache = self.cache
            keys = ('use_owner_extraction_unavailable','use_original_appearance_stamp_unavailable',
                    'use_binding_unavailable','use_original_source_time_expired')
            before = tuple(cache.counters[k] for k in keys)
            use = cache.use(originals[stamp.member],AcquisitionTime(stamp.acquired_at_s,
                stamp.clock_domain,stamp.clock_basis))
            coverage[stamp.member] = next((k for k,v in zip(keys,before) if cache.counters[k] > v),'available')
            if use is not None and use.appearance_ingress_monotonic_s is not None:
                appearances[stamp.member] = use
            delivery.append(CurrentVisualIngress(stamp,row[1],
                use.appearance.stamp if use is not None else None,
                use.appearance_ingress_monotonic_s if use is not None else None))
        candidates = []
        kitchens = [s for s in stamps if s.member.camera_id=='kitchen']
        families = [s for s in stamps if s.member.camera_id=='family-room']
        if kitchens and families:
            self.counts['overlap_cohorts'] += 1
            self.counts['overlap_missing_appearance_rows'] += sum(s.member not in appearances for s in stamps)
        if len(kitchens)*len(families) <= 128 and all(s.member in appearances for s in stamps) and not missing_ingress:
            for left in kitchens:
                for right in families:
                    a,b = appearances[left.member],appearances[right.member]
                    aa,bb = sum(v*v for v in a.vector),sum(v*v for v in b.vector)
                    cosine = max(-1.,min(1.,sum(x*y for x,y in zip(a.vector,b.vector))/sqrt(aa*bb)))
                    candidates.append(SpatialPersonCandidate(left,right,a.appearance,b.appearance,cosine,
                        a.appearance.extraction_id,b.appearance.extraction_id))
        viable = [(c.left.member,c.right.member) for c in candidates if c.similarity >= identity.ledger.policy.min_similarity]
        selected = tuple(p for p in viable if all(sum(m in q for q in viable)==1 for m in p))
        self.counts['candidate_edges'] += len(candidates)
        self.counts['appearance_supported_edges'] += len(selected)
        if identity._anatomical_policy is not None:
            if len(kitchens)*len(families)>32:
                self.counts['anatomical_pair_capacity'] += 1
            pairs = tuple(AnatomicalPairSupport(left,right,self.anatomical_originals[left.member],
                self.anatomical_originals[right.member]) for left in kitchens for right in families
                if len(kitchens)*len(families)<=32 and left.member in self.anatomical_originals and right.member in self.anatomical_originals)
            visual = CurrentVisualProof('anatomical-cohort:'+str(self.sequence),stamps,tuple(candidates),())
            proof = AnatomicalConsensusProof(visual,pairs)
            delivery_context = CurrentVisualDeliveryContext(self.actor_id,self.sequence,now,tuple(delivery))
            # The kernel selects the complete tri-state graph; producer makes no
            # shadow person decision and unavailable competitors stay visible.
            result = identity.join_anatomical_consensus(proof,at_interval=context,delivery_context=delivery_context)
            self.counts['result:'+result.reason] += 1
            return result
        # Read-only producer hint, NOT a renewal permission. The kernel alone
        # validates the exact seed/prior/native continuity and original lag.
        shared = {frozenset(m.observation.member for m in dto.members)
            for dto in identity.spatial_memberships(at_interval=context)
            if dto.association_basis == 'prior_projective_continuity' and len(dto.members)==2}
        pairs = []
        for left,right in selected:
            latest = frozenset((left,right)) in shared
            a,b = self._trajectory(left,latest=latest),self._trajectory(right,latest=latest)
            if a is not None and b is not None:
                pairs.append(ProjectiveMotionPair(a,b))
        proof = ProjectiveContinuityProof(CurrentVisualProof('projective-cohort:'+str(self.sequence),stamps,
            tuple(candidates),selected),tuple(pairs))
        delivery_context = CurrentVisualDeliveryContext(self.actor_id,self.sequence,now,tuple(delivery))
        self._inspect_original_motion(identity,context,delivery_context,proof.visual,kitchens,families,
            appearances,coverage)
        self._inspect_original_motion(identity,context,delivery_context,proof.visual,kitchens,families,
            appearances,coverage,_history=self.hypothesis_history,_counts=self.hypothesis_numeric_counts,
            _examples=self.hypothesis_numeric_examples)
        # Three bounded same-anatomy inspections; NEVER choose a decision from
        # diagnostic support. All actual competitors remain in each visual graph.
        for basis in self.cue_numeric_counts:
            histories={m:[s for s in values if s.support_basis==basis] for m,values in self.cue_history.items()}
            self._inspect_original_motion(identity,context,delivery_context,proof.visual,kitchens,families,
                appearances,coverage,_history=histories,_counts=self.cue_numeric_counts[basis],
                _examples=self.cue_numeric_examples[basis])
        if len(kitchens)*len(families)<=21:
            for left in kitchens:
                for right in families:
                    for basis in self.cue_numeric_counts:
                        poses=tuple([p for p in self.pose_history.get(s.member,()) if basis in p[2]] for s in (left,right))
                        if all(len(p)>=3 and p[-1][0]==s for p,s in zip(poses,(left,right))):
                            self.history_diagnosis['paired_current_2d_three_originals:'+basis]+=1
                        support=tuple(self._image_motion_available(self.pose_history.get(s.member,()),basis,s)
                                      for s in (left,right))
                        if all(s[0] for s in support):
                            self.history_diagnosis['paired_current_2d_patch_three_originals:'+basis]+=1
                            if all(s[1] for s in support):
                                self.history_diagnosis['paired_current_2d_patch_disjoint_motion:'+basis]+=1
        result = identity.join_projective_continuity(proof,at_interval=context,
            delivery_context=delivery_context)
        key = result.reason if len(self.counts) < 64 or result.reason in self.counts else 'other_reason'
        self.counts[key] += 1
        self.counts['accepted' if result.disposition == 'accepted' else 'not_accepted'] += 1
        return result

    def diagnostics(self):
        return {'basis':'prior_projective_continuity','history_samples':sum(map(len,self.history.values())),
                'original_body_inspection_only':self._body_diagnostic.summary(),
                'history_members':len(self.history),'ingress_members':len(self.ingresses),'counts':dict(self.counts),
                'numeric_inspection_only':{'counts':dict(self.numeric_counts),'examples':dict(self.numeric_examples)},
                'original_hypothesis_inspection_only':{'provenance_counts':dict(self.hypothesis_counts),
                    'counts':dict(self.hypothesis_numeric_counts),'examples':dict(self.hypothesis_numeric_examples),
                    'representative_coupling_examples':tuple(self.hypothesis_origins.values()),
                    'history_samples':sum(map(len,self.hypothesis_history.values()))},
                'original_swin_counts':{k:self.cache.counters[k] for k in (
                    'fresh_extractions','fifo_owned_holdovers','unavailable_or_expired',
                    'media_lock_unavailable','pending_capacity','owner_withdrawals',
                    'use_owner_extraction_unavailable','use_original_appearance_stamp_unavailable',
                    'use_binding_unavailable','use_original_source_time_expired')},
                'history_diagnosis_only':{'counts':dict(self.history_diagnosis),
                    'gap_examples':self.history_gap_examples,
                    'all_cue_samples':sum(map(len,self.cue_history.values())),
                    'pose_stamps':sum(map(len,self.pose_history.values())),
                    'same_anatomy_inspection':{basis:{'counts':dict(counts),'examples':self.cue_numeric_examples[basis]}
                        for basis,counts in self.cue_numeric_counts.items()}}}
