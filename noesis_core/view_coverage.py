"""Bounded, request-only declared projection and 2.5D height-envelope sampling."""
from __future__ import annotations

import math
import time
import numpy as np

from noesis.calibration.manager import CalibrationSnapshot
from .scene_prior import ScenePriorSet, ScenePriorError
from .contracts.view_coverage import (
    ViewAnalysisRequest, ViewAnalysis, ViewAnalysisProvenance, PointViewAnalysis, ViewAnalysisContext, ViewAnalysisBinding,
    ViewAnalysisResultProvenance, StaticPriorAnalysis, MAX_RAY_STEPS, MAX_TOTAL_STEPS,
)


from .view_projection import project_camera_point


class ViewCoverageError(ValueError):
    pass


class ViewCoverageRetired(ViewCoverageError):
    pass


class ViewCoverageService:
    def __init__(self, priors: ScenePriorSet, calibration: CalibrationSnapshot, camera_id: str, *, is_current=lambda: True, deadline=None):
        self.priors, self.calibration, self.camera_id = priors, calibration, camera_id
        self.is_current, self.deadline = is_current, deadline

    def _check(self):
        if self.is_current() is not True or self.deadline is not None and time.monotonic() >= self.deadline:
            raise ViewCoverageRetired("view_analysis_owner_retired")

    def _inputs(self, request=None):
        self._check()
        priors, calibration, camera_id = self.priors, self.calibration, self.camera_id
        if not isinstance(priors, ScenePriorSet) or not isinstance(calibration, CalibrationSnapshot) or calibration.camera_id != camera_id:
            raise ViewCoverageError("view_analysis_calibration_unavailable")
        revision, binding = priors.revision_for_camera(camera_id), priors.frame_binding(camera_id)
        contract = calibration.frame_contract
        if revision is None or binding is None or contract is None:
            raise ViewCoverageError("view_analysis_frame_binding_unavailable")
        expected = binding.frame_transform()
        if (contract != expected or contract.target_frame.frame_id != "backend_world_m"
            or calibration.camera_calibration_sha256 != binding.source_camera_calibration_sha256
            or calibration.world_alignment_sha256 != binding.source_world_alignment_sha256
            or calibration.scene_prior_id != (binding.artifact_revision_id or binding.target_frame.revision)
            or calibration.target_revision_id != binding.target_revision_id or calibration.unit_scale != 1.0):
            raise ViewCoverageError("view_analysis_calibration_binding_mismatch")
        if request is not None and (request.prior_id != revision.manifest.prior_id or request.world_frame_revision != contract.target_frame.revision
            or request.camera_calibration_sha256 != calibration.camera_calibration_sha256
            or request.world_alignment_sha256 != calibration.world_alignment_sha256
            or request.frame_transform_sha256 != contract.transform_sha256):
            raise ViewCoverageError("view_analysis_request_binding_mismatch")
        K = np.asarray(calibration.intrinsics, dtype=np.float64)
        size = calibration.image_size
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.array_equal(K[2], [0., 0., 1.]) or len(size) != 2
            or any(type(value) is not int or not 1 <= value <= 32768 for value in size)):
            raise ViewCoverageError("view_analysis_linear_calibration_invalid")
        # The existing frame owner composes camera-from-source E with ONE
        # inverse accepted source->target edge. Never apply presentation align.
        E = np.asarray(contract.camera_from_target_col_major(calibration.extrinsics_col_major), dtype=np.float64).reshape(4, 4, order="F")
        origin = np.linalg.inv(E)[:3, 3]
        metadata = priors.camera_view_metadata(camera_id, extrinsics_col_major=calibration.extrinsics_col_major)
        if (metadata["source_coordinate_frame_revision"] != contract.target_frame.revision
            or metadata["frame_transform_sha256"] != contract.transform_sha256
            or not np.allclose(origin, metadata["camera_position_world_m"], atol=1e-9, rtol=0)):
            raise ViewCoverageError("view_analysis_camera_metadata_mismatch")
        manifest = revision.manifest
        if (binding.presentation is not None and binding.presentation.authored_scene_sha256 is not None
            and binding.presentation.authored_scene_sha256 != manifest.authored_scene.sha256):
            raise ViewCoverageError("view_analysis_authored_scene_binding_mismatch")
        entry = next((entry for entry in priors.catalog.revisions if entry.prior_id == manifest.prior_id), None)
        grid_artifact = next((artifact for artifact in manifest.artifacts if artifact.role == "grid_npz"), None)
        if entry is None or grid_artifact is None:
            raise ViewCoverageError("view_analysis_prior_provenance_unavailable")
        provenance = ViewAnalysisProvenance(prior_id=manifest.prior_id, authored_scene_sha256=manifest.authored_scene.sha256, prior_manifest_sha256=entry.manifest_sha256,
            grid_sha256=grid_artifact.sha256, camera_calibration_sha256=calibration.camera_calibration_sha256,
            world_alignment_sha256=calibration.world_alignment_sha256, frame_transform_sha256=contract.transform_sha256,
            source_frame_revision=contract.source_frame.revision, world_frame="backend_world_m", world_frame_revision=contract.target_frame.revision,
            target_revision_id=binding.target_revision_id, image_size=size, unit_scale=1.0, floor_y_m=manifest.derivation.floor_y_m,
            grid_resolution_m=manifest.grid.resolution_m, mode=priors.binding(camera_id).mode, quality=manifest.quality,
            projection_metadata_status=getattr(calibration, "projection_metadata_status", "undeclared"),
            projection_model=getattr(calibration, "projection_model", None), distortion_model=getattr(calibration, "distortion_model", None),
            distortion_coefficients=getattr(calibration, "distortion_coefficients", None),
            distortion_coefficient_names=getattr(calibration, "distortion_coefficient_names", None))
        self._check()
        return E, K, origin, revision, provenance

    def context(self) -> ViewAnalysisContext:
        _E, _K, origin, _revision, provenance = self._inputs()
        binding = ViewAnalysisBinding(**{name: getattr(provenance, name) for name in ViewAnalysisBinding.model_fields})
        return ViewAnalysisContext(camera_id=self.camera_id, binding=binding, provenance=provenance, camera_position_m=tuple(float(value) for value in origin))

    def analyze(self, request: ViewAnalysisRequest) -> ViewAnalysis:
        request = ViewAnalysisRequest.model_validate(request)
        E, K, origin, revision, provenance = self._inputs(request)
        size = self.calibration.image_size
        results, total = [], 0
        for point in request.points:
            self._check()
            target = np.asarray(point.position_m, dtype=np.float64)
            projected = E @ np.r_[target, 1.]
            projection = project_camera_point(projected[:3], K, size, self.calibration)
            static = self._ray(revision, origin, target, MAX_TOTAL_STEPS - total)
            total += static.sample_count
            results.append(PointViewAnalysis(id=point.id, position_m=point.position_m, projection=projection, static_prior=static))
        self._check()
        result_provenance = ViewAnalysisResultProvenance(**{**provenance.model_dump(),
            "distortion_applied": any(row.projection.distortion_applied for row in results)})
        return ViewAnalysis(camera_id=self.camera_id, provenance=result_provenance, camera_position_m=tuple(float(value) for value in origin), results=tuple(results), total_samples=total)

    def _ray(self, revision, origin, target, budget):
        def unknown(reason, count=0):
            return StaticPriorAnalysis(state="unknown", reason=reason, sample_count=count)
        if not revision.manifest.quality.passed:
            return unknown("prior_quality_not_passed")
        distance = float(np.linalg.norm((target - origin)[[0, 2]]))
        if distance < 1e-12:
            return unknown("vertical_ray_not_resolved_by_2_5d")
        needed = distance / (float(revision.manifest.grid.resolution_m) * .5)
        if not math.isfinite(needed) or needed > MAX_RAY_STEPS or math.ceil(needed) > budget:
            return unknown("step_bound")
        count = max(1, math.ceil(needed))
        fractions = (np.arange(count, dtype=np.float64) + .5) / count
        samples = origin[None, :] + fractions[:, None] * (target - origin)[None, :]
        values = revision.sample(samples[:, 0], samples[:, 2])
        missing, potential = None, False
        for index, sample in enumerate(samples):
            if index % 64 == 0:
                self._check()
            if not bool(values["inside_extent"][index]):
                missing = missing or "outside_grid"; continue
            if not bool(values["observed"][index]):
                missing = missing or "unobserved_cell"; continue
            if int(values["point_count"][index]) < revision.manifest.derivation.obstacle_min_support:
                missing = missing or "sparse_cell"; continue
            floor, height = float(values["floor_height_m"][index]), float(values["height_agl_p95_m"][index])
            if not math.isfinite(floor) or not math.isfinite(height) or height < 0:
                missing = missing or "invalid_height"; continue
            if sample[1] < floor:
                missing = missing or "ray_below_sampled_floor"; continue
            if height >= revision.manifest.derivation.obstacle_min_height_m and floor + height >= sample[1]:
                potential = True
        if potential:
            return StaticPriorAnalysis(state="potential_static_obstruction", reason="sampled_height_envelope_intersects_ray", sample_count=count)
        if missing:
            return unknown(missing, count)
        return StaticPriorAnalysis(state="no_sampled_static_obstruction", reason="sampled_prior_only", sample_count=count)
