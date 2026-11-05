"""Synthetic verified prior + actual immutable calibration, without runtime IO."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from pydantic import ValidationError
from noesis.calibration.manager import CalibrationSnapshot
from noesis.virtual_twin.artifacts import write_points_glb
from noesis_core.contracts.scene_prior import ScenePriorRevision, ScenePriorCatalog, ScenePriorFrameBinding
from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256
from noesis_core.scene_prior import ScenePriorSet
from noesis_core.contracts.view_coverage import ViewAnalysisRequest, MAX_TOTAL_STEPS
from noesis_core.view_coverage import ViewCoverageService, ViewCoverageError, ViewCoverageRetired


def fixture(root, *, observed=1, count=8, height=.2):
    root.mkdir(parents=True, exist_ok=True)
    shape = (2, 2)
    arrays = {name: np.ones(shape, dtype=np.uint8) for name in ("authored_walkable", "observed", "floor_supported", "walkable_candidate")}
    arrays.update(observed=np.full(shape, observed, dtype=np.uint8), obstacle_mask=np.zeros(shape, dtype=np.uint8),
        evidence_confidence=np.full(shape, .8, dtype=np.float32), floor_height_m=np.zeros(shape, dtype=np.float32),
        height_agl_p95_m=np.full(shape, height, dtype=np.float32), boundary_signed_distance_m=np.ones(shape, dtype=np.float32),
        obstacle_signed_clearance_m=np.ones(shape, dtype=np.float32), point_count=np.full(shape, count, dtype=np.uint32),
        floor_support_count=np.full(shape, 3, dtype=np.uint32), obstacle_support_count=np.zeros(shape, dtype=np.uint32))
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
    glb = root / "points.glb"; write_points_glb(glb, np.array([[.5, 0., .5], [.5, .2, .5]], dtype=np.float32))
    payloads = {"grid_npz": ("grid.npz", buffer.getvalue()), "metrics": ("metrics.json", b"{}"), "preview": ("preview.png", b"png"), "points_glb": ("points.glb", glb.read_bytes())}
    fp = lambda role, letter: {"role": role, "sha256": letter * 64}
    bounds = dict(min_x=0, max_x=2, min_z=0, max_z=2)
    prior_id = "synthetic-view-prior"
    T = np.eye(4); T[0, 3] = .25; T[2, 3] = .25
    source, target = RevisionedFrame("backend_world_m", "source-calibration-v1"), RevisionedFrame("backend_world_m", prior_id)
    values = tuple(float(value) for value in T.flatten(order="F"))
    frame = ScenePriorFrameBinding.model_validate(dict(contract="noesis.scene_prior.frame_binding", contract_version=1,
        source_frame=dict(frame_id=source.frame_id, revision=source.revision), target_frame=dict(frame_id=target.frame_id, revision=target.revision),
        source_camera_calibration_sha256="a"*64, source_world_alignment_sha256="b"*64, target_revision_id="target-v1",
        target_revision_metadata_sha256="c"*64, target_from_source_col_major=values,
        target_from_source_sha256=revisioned_transform_sha256(source, target, values),
        source_floor_plane=dict(frame=dict(frame_id=source.frame_id, revision=source.revision), normal=[0.,1.,0.], offset_m=0.),
        target_floor_plane=dict(frame=dict(frame_id=target.frame_id, revision=target.revision), normal=[0.,1.,0.], offset_m=0.)))
    manifest = ScenePriorRevision.model_validate(dict(contract="noesis.scene_prior.revision", contract_version=1, prior_id=prior_id,
        site_id="synthetic-site", space_id="synthetic-space", created_at_us=1, created_by="fixture", intended_use="shadow",
        source=dict(source_type="mapanything_multiview_room_walk", bundle_id="fixture-bundle", bundle_schema="noesis.reference.room_scan_bundle.v1", bundle_manifest_sha256="1"*64, reference_sha256="2"*64, capture_id="fixture", captured_at_us=1, model="synthetic"),
        alignment=fp("alignment","3"), authored_scene=fp("authored_scene","4"), world_to_scene=fp("world_to_scene","b"),
        semantic_binding=dict(room_labels=["Synthetic"], authored_groups=["synthetic"], room_group_map=fp("room_map","5")),
        grid=dict(coordinate_frame="backend_world_m", units="meters", orientation="row_increases_positive_z_column_increases_positive_x", bounds=bounds, resolution_m=1., rows=2, columns=2),
        preview=dict(coordinate_frame="camera_local_ground_m", units="meters", orientation="row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x", reference_camera_id="camera-a", camera_calibration=fp("camera_calibration","a"), target_revision_metadata=fp("metadata","c"), camera_position_world_m=[.5,2.,.5], camera_right_world_xz=[-1.,0.], camera_forward_world_xz=[0.,1.], bounds=bounds,resolution_m=1.,rows=2,columns=2),
        derivation=dict(algorithm="noesis_scene_prior_2_5d_v1",floor_y_m=0.,floor_support_band_m=.12,obstacle_min_height_m=.18,obstacle_max_height_m=2.2,obstacle_min_support=3,max_source_height_m=3.2),
        quality=dict(passed=True,source_point_count=8,selected_point_count=8,authored_cell_count=4,observed_cell_count=4,floor_supported_cell_count=4,obstacle_cell_count=0,authored_observed_fraction=1.,authored_floor_supported_fraction=1.,alignment_status="passed"),
        artifacts=[dict(role=role,relative_path=name,sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data)) for role,(name,data) in payloads.items()]))
    for name, data in payloads.values(): (root / name).write_bytes(data)
    raw = manifest.model_dump_json().encode(); (root / "manifest.json").write_bytes(raw)
    catalog = ScenePriorCatalog.model_validate(dict(contract="noesis.scene_prior.catalog",contract_version=1,site_id="synthetic-site",
        revisions=[dict(prior_id=prior_id,space_id="synthetic-space",manifest_path="manifest.json",manifest_sha256=hashlib.sha256(raw).hexdigest(),manifest_size_bytes=len(raw))],
        camera_bindings=[dict(camera_id="camera-a",space_id="synthetic-space",prior_id=prior_id,mode="shadow",include_floorplan_layers=True,frame_binding=frame.model_dump(mode="json"))]))
    (root / "catalog.json").write_text(catalog.model_dump_json())
    priors = ScenePriorSet.load(root / "catalog.json")
    K = np.array([[100.,0.,320.],[0.,100.,240.],[0.,0.,1.]]); K.setflags(write=False)
    E = np.eye(4); E[0,0] = E[1,1] = -1.; E[:3,3] = [.25,2.,-.25]
    snapshot = CalibrationSnapshot(camera_id="camera-a",intrinsics=K,extrinsics_col_major=tuple(E.flatten(order="F")),floor_y=0.,image_size=(640,480),unit_scale=1.,frame_contract=frame.frame_transform(),camera_calibration_sha256="a"*64,world_alignment_sha256="b"*64,scene_prior_id=prior_id,target_revision_id="target-v1")
    service = ViewCoverageService(priors,snapshot,"camera-a")
    context = service.context()
    def request(points=None, **changes):
        return ViewAnalysisRequest.model_validate({**context.binding.model_dump(mode="json"),"points":points or [{"id":"point-a","position_m":[.5,1.,1.5]}],**changes})
    return SimpleNamespace(priors=priors,snapshot=snapshot,service=service,context=context,request=request)


def test_exact_accepted_transform_linear_projection_and_clear_sampling_are_separate_qualified_evidence(tmp_path):
    f=fixture(tmp_path); result=f.service.analyze(f.request())
    assert result.camera_position_m == (.5,2.,.5)
    row=result.results[0]; assert row.projection.state=="in" and row.projection.pixel==(320.,340.)
    assert row.projection.optical_visibility=="unknown" and row.projection.basis=="mathematical_linear_k_only"
    assert row.static_prior.state=="no_sampled_static_obstruction" and row.static_prior.reason=="sampled_prior_only"
    assert result.provenance.frame_transform_sha256==f.context.binding.frame_transform_sha256
    assert result.provenance.authored_scene_sha256==f.context.provenance.authored_scene_sha256==f.priors.revision_for_camera("camera-a").manifest.authored_scene.sha256
    assert result.provenance.projection_metadata_status=="undeclared" and not result.provenance.distortion_applied
    assert "do_not_establish_physical_visibility" in result.qualification
    other=f.service.analyze(f.request([{"id":"behind","position_m":[.5,1.,-.5]},{"id":"outside","position_m":[100.,1.,1.5]}]))
    assert [row.projection.reason for row in other.results]==["behind_camera","outside_image"]


@pytest.mark.parametrize("changes,expected",[({"height":3.},"potential_static_obstruction"),({"observed":0},"unknown"),({"count":1},"unknown"),({"height":float("nan")},"unknown")])
def test_original_height_envelope_unobserved_sparse_and_missing_height(tmp_path,changes,expected):
    f=fixture(tmp_path,**changes); result=f.service.analyze(f.request()); assert result.results[0].static_prior.state==expected
    assert result.results[0].projection.optical_visibility=="unknown"


def test_outside_grid_vertical_step_bounds_and_strict_request_limits(tmp_path):
    f=fixture(tmp_path)
    result=f.service.analyze(f.request([{"id":"outside","position_m":[5.,5.,5.]},{"id":"vertical","position_m":[.5,1.,.5]},{"id":"far","position_m":[10000.,1.,10000.]}]))
    assert [row.static_prior.reason for row in result.results]==["outside_grid","vertical_ray_not_resolved_by_2_5d","step_bound"]
    assert result.total_samples<=MAX_TOTAL_STEPS
    with pytest.raises(ValidationError): f.request([{"id":str(index),"position_m":[0.,0.,0.]} for index in range(65)])
    with pytest.raises(ValidationError): f.request(world_frame="menon_scene")
    with pytest.raises(ViewCoverageError,match="binding_mismatch"): f.service.analyze(f.request(frame_transform_sha256="f"*64))
    with pytest.raises(ViewCoverageError): ViewCoverageService(f.priors,replace(f.snapshot,unit_scale=100.),"camera-a").context()
    with pytest.raises(ViewCoverageRetired): ViewCoverageService(f.priors,f.snapshot,"camera-a",is_current=lambda:False).analyze(f.request())


def test_original_projection_metadata_is_applied_only_for_exact_supported_declared_order(tmp_path):
    import cv2
    f=fixture(tmp_path)
    coefficients=(-.04,-.07,.06,-.02)
    snapshot=replace(f.snapshot,projection_metadata_status="declared",projection_model="opencv_fisheye",
        distortion_model="fisheye",distortion_coefficients=coefficients,distortion_coefficient_names=None)
    service=ViewCoverageService(f.priors,snapshot,"camera-a")
    result=service.analyze(f.request())
    expected,_=cv2.fisheye.projectPoints(np.array([[[0.,1.,1.]]]),np.zeros(3),np.zeros(3),snapshot.intrinsics,np.array(coefficients))
    assert result.version==2 and service.context().version==1
    assert service.context().provenance.distortion_applied is False
    assert result.provenance.distortion_applied is True
    assert result.provenance.distortion_coefficients==coefficients
    assert result.provenance.projection_model=="opencv_fisheye"
    assert result.results[0].projection.basis=="declared_opencv_fisheye"
    assert result.results[0].projection.image_domain=="declared_distorted_calibration_image"
    np.testing.assert_allclose(result.results[0].projection.pixel,expected[0,0],atol=1e-10,rtol=0)
    assert result.results[0].projection.pixel!=(320.,340.)
    assert result.results[0].projection.optical_visibility=="unknown"
    for changes,reason in (({"projection_model":"unsupported_declared_model"},"unsupported_projection_model"),
            ({"distortion_coefficient_names":("k2","k1","k3","k4")},"unsupported_coefficient_order"),
            ({"projection_metadata_status":"invalid","projection_model":None,"distortion_coefficients":None},"invalid_projection_metadata")):
        row=ViewCoverageService(f.priors,replace(snapshot,**changes),"camera-a").analyze(f.request()).results[0]
        assert row.projection.state=="unknown" and row.projection.reason==reason
        assert row.projection.basis=="unknown" and row.projection.pixel is None
        assert row.projection.distortion_applied is False and row.projection.optical_visibility=="unknown"
