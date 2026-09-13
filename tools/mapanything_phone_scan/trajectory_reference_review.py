"""Offline heldout handset reference review; never publishes alignment or calibration.

An explicit annotation binding joins manually inspected pixels to their camera,
rectification, recorded-frame timing, and saved alignment artifacts. It is an
input provenance declaration, not proof that visual device centers are surveyed
optical centers. Run with ``python -m tools.mapanything_phone_scan.trajectory_reference_review``.

The annotation CSV uses frame_id, phone_pts_s, u_px, v_px, uncertainty_px,
uncertainty_semantics, target=visible_phone_device_center, visibility=visible,
source_image, image_width_px, image_height_px, and image_sha256. Make the
annotations without seeing the reconstructed projection and record the method.

The JSON sidecar uses schema=noesis.phone_scan.handset_annotation_binding.v1,
camera_id, revision_id, world_frame, world_frame_revision, companion_session_id,
annotation_csv_sha256, independent_of_reconstruction_and_projection=true,
pixel_frame=rectified_static_image_px,
pixel_axes=origin_top_left_u_right_v_down, intrinsics (3x3), image_size_px,
and image_root. Each of diagnostic_inputs, aligned_camera_solution, calibration,
temporal_alignment, frame_timing_map, rectification_producer,
rectification_config, and annotation_procedure supplies {path, sha256}.
Paths resolve relative to the sidecar. Bind the actual captured calibration,
rectification evidence and exact-frame timing map; never infer them from image
filenames. The light-alignment equation, allowance and source/camera/session
must agree with the recorded timeline. Hashes verify retained contents, not
the physical correctness of a supplied annotation or reference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from PIL import Image

from .alignment import (_apply_yaw_parameters, _yaw_transform, _fixed_camera_comparable_mask,
                        _fixed_camera_visible_cloud_metrics, _fixed_camera_visible_structure_metrics)
from .trajectory_motion_review import (_csv, _evidence, _integer, _json, _load_provider,
                                       _read, _validate_poses, TrajectoryMotionReviewError)

SCHEMA = "noesis.phone_scan.trajectory_reference_review.v1"
BINDING_SCHEMA = "noesis.phone_scan.handset_annotation_binding.v1"
Error = TrajectoryMotionReviewError


@dataclass(frozen=True)
class TrajectoryReferenceReviewSettings:
    heldout_start_s: float
    timing_allowance_s: float
    allow_partial: bool = False
    max_pose_gap_s: float = 0.85
    fit_translation_candidate: bool = False
    max_translation_axis_m: float = 0.5

    def validate(self):
        if not math.isfinite(self.heldout_start_s) or self.heldout_start_s < 0:
            raise Error("heldout start must be finite and nonnegative")
        if not math.isfinite(self.timing_allowance_s) or not 0 <= self.timing_allowance_s <= 1:
            raise Error("timing allowance must be explicit and within 0–1 seconds")
        if not 0.01 <= self.max_pose_gap_s <= 2:
            raise Error("maximum pose gap must be within 0.01–2 seconds")
        if not 0.01 <= self.max_translation_axis_m <= 1:
            raise Error("translation bound must be within 0.01–1 metre per axis")


def _artifact(binding, key, base):
    value = binding.get(key, {})
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise Error(f"binding needs {key} artifact path and SHA-256")
    path = (base / value["path"]).resolve()
    raw = _read(path)
    if hashlib.sha256(raw).hexdigest() != value.get("sha256"):
        raise Error(f"{key} artifact hash mismatch")
    return path, raw


def _npz(raw):
    # Bound expanded archive size too, before NumPy allocates arrays.
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(row.file_size for row in archive.infolist()) > 128 * 1024 * 1024:
            raise Error("expanded diagnostic archive exceeds 128 MiB")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _matrix(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise Error(f"{label} needs a finite {shape} matrix")
    return result


def _load_alignment(source, directory, binding, base):
    report, _ = _json(directory / "alignment_report.json")
    # Historical diagnostic reports retain NaN for unsupported optional metrics.
    # Validate every consumed numeric field below rather than consuming those metrics.
    diagnostic = json.loads(_read(directory / "diagnostic_report.json"))
    transform, _ = _json(directory / "phone_ma_to_noesis_world.json")
    target = report.get("target", {})
    provider_hash = source["evidence"]["provider_manifest"]["sha256"]
    if (report.get("status") != "passed"
            or report.get("inputs", {}).get("phone_source", {}).get("output_manifest", {}).get("sha256") != provider_hash
            or diagnostic.get("input_hashes", {}).get("phone_output_manifest_sha256") != provider_hash
            or transform.get("source_output_manifest", {}).get("sha256") != provider_hash):
        raise Error("alignment must be passed and bound to this exact provider manifest")
    for key in ("camera_id", "world_frame", "world_frame_revision", "revision_id", "companion_session_id"):
        if not target.get(key) or binding.get(key) != target[key]:
            raise Error(f"annotation/alignment {key} binding mismatch")
    tb = transform.get("target_binding", {})
    if (target.get("coordinate_frame") != "backend_world_m_stream_points"
            or diagnostic.get("camera_id") != target["camera_id"]
            or diagnostic.get("target_revision_id") != target["revision_id"]
            or any(tb.get(k) != target.get(k) for k in
                   ("world_frame", "world_frame_revision", "camera_frame_binding", "companion_session_id"))):
        raise Error("saved alignment target frame/revision mismatch")
    evidence = {name: _evidence(directory / name, _read(directory / name)) for name in
                ("alignment_report.json", "diagnostic_report.json", "phone_ma_to_noesis_world.json")}
    arrays = {}
    for key, name in (("diagnostic_inputs", "diagnostic_inputs.npz"), ("aligned_camera_solution", "aligned_camera_solution.npz")):
        path, raw = _artifact(binding, key, base)
        if path != (directory / name).resolve():
            raise Error(f"{key} is not in the selected alignment directory")
        evidence[key] = _evidence(path, raw)
        arrays[key] = _npz(raw)
    di = arrays["diagnostic_inputs"]
    C = _validate_poses([di.get("target_camera_from_world")], 1)[0]
    K = _matrix(di.get("intrinsics"), (3, 3), "intrinsics")
    calibration_path, raw = _artifact(binding, "calibration", base)
    calibration = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != diagnostic["input_hashes"].get("camera_calibration_sha256"):
        raise Error("calibration does not match the alignment diagnostic")
    frame = calibration.get("frame_binding", {})
    if (calibration.get("E_semantics") != "camera_from_calibration_frame_raw"
            or frame != target.get("camera_frame_binding")
            or frame.get("world_frame") != {"frame_id": target["world_frame"], "revision": target["world_frame_revision"]}):
        raise Error("camera calibration/world binding mismatch")
    cc = calibration.get("cameras", {}).get(target["camera_id"], {})
    E = np.asarray(cc.get("E"), dtype=float).reshape(4, 4, order="F")
    T = np.asarray(frame.get("target_from_calibration_col_major"), dtype=float).reshape(4, 4, order="F")
    _validate_poses([E, T], 2)
    fx, fy, cx, cy = cc["K"]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise Error("calibration focal lengths must be finite and positive")
    expected_K = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    if not np.allclose(C, E @ np.linalg.inv(T), atol=1e-7, rtol=0) or not np.allclose(K, expected_K, atol=1e-7, rtol=0):
        raise Error("diagnostic projection disagrees with bound calibrated camera")
    if (binding.get("pixel_frame") != "rectified_static_image_px"
            or binding.get("pixel_axes") != "origin_top_left_u_right_v_down"
            or not np.allclose(_matrix(binding.get("intrinsics"), (3, 3), "annotation intrinsics"), K, atol=1e-7, rtol=0)):
        raise Error("annotation pixel domain/intrinsics mismatch")
    world = _validate_poses([transform.get("world_from_mapanything_row_major")], 1)[0]
    if (transform.get("scale") != 1.0
            or transform.get("source_coordinate_frame") != source["manifest"]["coordinate_frame"]
            or transform.get("target_coordinate_frame") != target["coordinate_frame"]):
        raise Error("reference review requires a metric rigid alignment")
    poses = _validate_poses(arrays["aligned_camera_solution"].get("camera_poses"), len(source["poses"]))
    if not np.allclose(poses, world @ source["poses"], atol=2e-5, rtol=0):
        raise Error("aligned poses disagree with the selected raw provider and saved transform")
    for key in ("source_structure", "target_structure", "target_normals", "target_points", "leveled_registration"):
        value = np.asarray(di.get(key), dtype=float)
        if value.ndim != 2 or value.shape[1] != 3 or not 1 <= len(value) <= 1_000_000 or not np.isfinite(value).all():
            raise Error(f"invalid saved {key}")
    if di["target_structure"].shape != di["target_normals"].shape:
        raise Error("target point/normal counts disagree")
    grid = di.get("target_depth_grid")
    if (not isinstance(grid, np.ndarray) or grid.ndim != 2 or not np.isfinite(grid).any()
            or np.isnan(grid).any() or np.any(grid <= 0)
            or list(grid.shape[::-1] * np.array([8, 8])) != binding.get("image_size_px")):
        raise Error("target visibility domain/image dimensions mismatch")
    parameters = _matrix(di.get("best_parameters"), (3,), "saved alignment parameters")
    floor = _validate_poses([di.get("floor_transform")], 1)[0]
    if not np.allclose(_yaw_transform(parameters) @ floor, world, atol=1e-7, rtol=0):
        raise Error("diagnostic geometry transform disagrees with saved aligned poses")
    if not np.allclose(np.linalg.norm(di["target_normals"], axis=1), 1, atol=1e-4, rtol=0):
        raise Error("target normals must be unit length")
    evidence["calibration"] = _evidence(calibration_path, raw)
    return poses, di, target, evidence


def _load_annotations(scan, path, binding, base, target, settings):
    rows, raw = _csv(path, {"frame_id", "phone_pts_s", "u_px", "v_px", "uncertainty_px", "uncertainty_semantics",
                            "target", "visibility", "source_image", "image_width_px", "image_height_px", "image_sha256"})
    if len(rows) > 1000 or hashlib.sha256(raw).hexdigest() != binding.get("annotation_csv_sha256"):
        raise Error("annotation CSV hash mismatch or more than 1000 annotations")
    if binding.get("independent_of_reconstruction_and_projection") is not True:
        raise Error("blind annotation independence must be explicitly declared")
    tp, tr = _artifact(binding, "temporal_alignment", base)
    temporal = json.loads(tr)
    if any(temporal.get(k) != v for k, v in (("scan_id", scan.name), ("camera_id", target["camera_id"]), ("companion_session_id", target["companion_session_id"]))):
        raise Error("timing scan/camera/session binding mismatch")
    timing = temporal.get("light_alignment", {})
    if (timing.get("equation") != "static_mkv_pts_s = phone_mp4_pts_s + offset_s"
            or timing.get("clock_rate") != 1.0
            or settings.timing_allowance_s != timing.get("local_review_allowance_s")):
        raise Error("timing equation/rate/explicit engineering allowance mismatch")
    offset = float(timing["offset_s"])
    if not math.isfinite(offset):
        raise Error("nonfinite timing offset")
    mp, mr = _artifact(binding, "frame_timing_map", base)
    mapped, _ = _csv(mp, {"ds9_frame_id", "derived_static_frame_index", "static_decoded_pts_s", "estimated_phone_pts_s", "recorded_static_frame_exists", "mapping_status"})
    by_frame = {}
    for row in mapped:
        if row["recorded_static_frame_exists"] != "True" or row["mapping_status"] != "exact_cohort_to_unique_differential_pts_pair":
            continue
        fid = _integer(row["ds9_frame_id"], "static frame id")
        signature = (int(row["derived_static_frame_index"]), float(row["static_decoded_pts_s"]), float(row["estimated_phone_pts_s"]))
        if fid in by_frame and by_frame[fid] != signature:
            raise Error("ambiguous static frame timing correspondence")
        by_frame[fid] = signature
    # Preserve exact rectification producer/config evidence; filenames alone are not domain proof.
    evidence = {"annotations": _evidence(path, raw), "temporal_alignment": _evidence(tp, tr), "frame_timing_map": _evidence(mp, mr)}
    for key in ("rectification_producer", "rectification_config", "annotation_procedure"):
        p, r = _artifact(binding, key, base)
        evidence[key] = _evidence(p, r)
    image_root = (base / binding["image_root"]).resolve()
    ids, last_time = set(), -math.inf
    for row in rows:
        fid = _integer(row["frame_id"], "annotation frame id")
        time, u, v, tolerance = [float(row[k]) for k in ("phone_pts_s", "u_px", "v_px", "uncertainty_px")]
        width, height = [_integer(row[k], "image dimension") for k in ("image_width_px", "image_height_px")]
        if (fid in ids or not np.isfinite([time, u, v, tolerance]).all() or time <= last_time
                or not 0 <= u < width or not 0 <= v < height or not 0 < tolerance <= 200
                or [width, height] != binding["image_size_px"]
                or row["target"] != "visible_phone_device_center" or row["visibility"] != "visible"
                or not row["uncertainty_semantics"].strip()):
            raise Error("annotation identity/order/visible pixel/tolerance is invalid")
        ids.add(fid); last_time = time
        match = by_frame.get(fid)
        if (match is None or abs(match[2] - time) > 1e-8 or abs(match[1] - offset - time) > 1e-8
                or match[0] != fid - temporal["static_ds9_mapping"]["static_decoded_frame_index_equals_ds9_frame_id_minus"]):
            raise Error("annotation recorded frame/time identity mismatch")
        image_path = (image_root / row["source_image"]).resolve()
        if not image_path.is_relative_to(image_root):
            raise Error("annotation image escapes image root")
        image_raw = _read(image_path)
        if hashlib.sha256(image_raw).hexdigest() != row["image_sha256"]:
            raise Error("annotation source image hash mismatch")
        with Image.open(io.BytesIO(image_raw)) as image:
            if image.size != (width, height):
                raise Error("annotation actual image dimensions mismatch")
        row.update(frame_id=fid, phone_pts_s=time, u_px=u, v_px=v, uncertainty_px=tolerance)
    return rows, evidence, {"allowance_meaning": timing.get("allowance_meaning"), "full_walk_application": timing.get("full_walk_application")}


def interpolate_positions(times, positions, query, selected, max_gap_s, windows=None):
    """Return NaN plus a reason for unsupported times; never clamp or bridge omissions."""
    times, query = np.asarray(times), np.asarray(query)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise Error("provider pose times must be finite and strictly increasing")
    result = np.full((len(query), 3), np.nan)
    reasons, spans = [], []
    for i, value in enumerate(query):
        j = int(np.searchsorted(times, value))
        if j < len(times) and times[j] == value:
            result[i] = positions[j]; reasons.append("supported_exact_pose"); spans.append(0.0); continue
        if j == 0 or j == len(times):
            reasons.append("outside_pose_time_range"); spans.append(None); continue
        span = float(times[j] - times[j - 1]); spans.append(span)
        if selected[j] != selected[j - 1] + 1:
            reasons.append("omitted_prepared_view_gap"); continue
        if windows is not None and windows[j] != windows[j - 1]:
            reasons.append("provider_window_transition"); continue
        if span > max_gap_s:
            reasons.append("pose_gap_exceeds_limit"); continue
        result[i] = positions[j - 1] + (value - times[j - 1]) / span * (positions[j] - positions[j - 1])
        reasons.append("supported_interpolation")
    return result, reasons, spans


def project(points, camera_from_world, intrinsics):
    camera = np.asarray(points) @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.05)
    uv = np.full((len(camera), 2), np.nan)
    homogeneous = camera[valid] @ intrinsics.T
    uv[valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    return uv, valid


def _metric(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0, "median": None, "p80": None, "rms": None, "max": None}
    return {"count": len(values), "median": float(np.median(values)), "p80": float(np.quantile(values, .8)),
            "rms": float(np.sqrt(np.mean(values ** 2))), "max": float(np.max(values))}


def fit_translation(points, uv, tolerance, train, C, K, bound):
    # Only these copies enter the optimizer; heldout values cannot influence it.
    p, observed, scale = points[train].copy(), uv[train].copy(), tolerance[train].copy()
    if len(p) < 3:
        raise Error("translation diagnostic requires at least three supported training annotations")
    def residual(delta):
        camera = (p + delta) @ C[:3, :3].T + C[:3, 3]
        projected = camera @ K.T
        predicted = projected[:, :2] / np.maximum(projected[:, 2, None], .05)
        return ((predicted - observed) / scale[:, None]).ravel()
    fit = least_squares(residual, np.zeros(3), bounds=(-bound, bound), max_nfev=100, loss="linear")
    return fit.x, bool(fit.success), int(fit.nfev)


def assess_geometry(di, shift):
    C, K, grid = [di[k] for k in ("target_camera_from_world", "intrinsics", "target_depth_grid")]
    result, conflicts = {}, []
    for label, source, target in (("structure", _apply_yaw_parameters(di["source_structure"], di["best_parameters"]), di["target_structure"]),
                                  ("full_cloud", _apply_yaw_parameters(di["leveled_registration"], di["best_parameters"]), di["target_points"])):
        baseline_mask, before_coverage = _fixed_camera_comparable_mask(source, grid, C, K, cell_px=8, occlusion_tolerance_m=.30)
        candidate_mask, after_coverage = _fixed_camera_comparable_mask(source + shift, grid, C, K, cell_px=8, occlusion_tolerance_m=.30)
        selected = source[baseline_mask]
        if not len(selected):
            raise Error("independent geometry has no comparable baseline support")
        _, nearest = cKDTree(target).query(selected)
        before = selected - target[nearest]; after = selected + shift - target[nearest]
        if label == "structure":
            normals = di["target_normals"][nearest]
            before_error = np.abs(np.sum(before * normals, axis=1)); after_error = np.abs(np.sum(after * normals, axis=1))
            metric = lambda s: _fixed_camera_visible_structure_metrics(s, target, di["target_normals"], grid, C, K)
        else:
            before_error = np.linalg.norm(before, axis=1); after_error = np.linalg.norm(after, axis=1)
            metric = lambda s: _fixed_camera_visible_cloud_metrics(s, target, grid, C, K)
        before_stats, after_stats = _metric(before_error), _metric(after_error)
        retained = int(np.count_nonzero(baseline_mask & candidate_mask))
        coverage_ok = retained == int(np.count_nonzero(baseline_mask))
        residual_ok = all(after_stats[k] <= before_stats[k] + 1e-8 for k in ("median", "p80", "rms"))
        if not coverage_ok: conflicts.append(label + "_coverage_loss")
        if not residual_ok: conflicts.append(label + "_fixed_domain_residual_regression")
        result[label] = {"original_visibility": before_coverage, "candidate_visibility": after_coverage,
                         "retained_original_comparable_count": retained, "coverage_nonregression": coverage_ok,
                         "fixed_original_points_and_references": {"original": before_stats, "candidate": after_stats},
                         "residual_nonregression": residual_ok,
                         "dynamic_visible_original": metric(source), "dynamic_visible_candidate": metric(source + shift)}
    return result, conflicts


def review_trajectory_reference(scan, provider_manifest, alignment_dir, annotation_csv, annotation_binding, output_dir, settings):
    settings.validate()
    scan, provider_manifest, alignment_dir, annotation_csv, annotation_binding, output_dir = map(Path, (scan, provider_manifest, alignment_dir, annotation_csv, annotation_binding, output_dir))
    if output_dir.exists():
        raise Error("use a new output directory to preserve previous evidence")
    binding, binding_raw = _json(annotation_binding)
    if binding.get("schema") != BINDING_SCHEMA:
        raise Error("unsupported annotation binding schema")
    source = _load_provider(scan, provider_manifest, settings.allow_partial)
    poses, di, target, evidence = _load_alignment(source, alignment_dir, binding, annotation_binding.parent)
    rows, annotations_evidence, timing_notes = _load_annotations(scan, annotation_csv, binding, annotation_binding.parent, target, settings)
    times = np.asarray([r["timestamp_s"] for r in source["manifest"]["frames"]])
    query = np.asarray([r["phone_pts_s"] for r in rows]); observed = np.asarray([[r["u_px"], r["v_px"]] for r in rows]); tolerance = np.asarray([r["uncertainty_px"] for r in rows])
    C, K = di["target_camera_from_world"], di["intrinsics"]
    points, reasons, spans = interpolate_positions(times, poses[:, :3, 3], query, source["selected"], settings.max_pose_gap_s, source["windows"])
    predicted, supported = project(points, C, K)
    train = supported & (query < settings.heldout_start_s); heldout = supported & ~train
    if not train.any() or not heldout.any():
        raise Error("review requires supported annotations in both train and heldout splits")
    errors = np.linalg.norm(predicted - observed, axis=1)
    timing_rows, timing_errors = [], []
    for offset in np.linspace(-settings.timing_allowance_s, settings.timing_allowance_s, 21):
        p, _, _ = interpolate_positions(times, poses[:, :3, 3], query + offset, source["selected"], settings.max_pose_gap_s, source["windows"])
        uv, valid = project(p, C, K); error = np.linalg.norm(uv - observed, axis=1)
        timing_errors.append(error)
        timing_rows.append({"offset_s": float(offset), "supported_count": int(valid.sum()), "error_px": _metric(error[valid])})
    timing_errors = np.asarray(timing_errors)
    common = supported & np.isfinite(timing_errors).all(axis=0)
    result = {"schema": SCHEMA, "status": "offline_reference_review", "runtime_admission_ready": False,
              "scope": source["scope"], "target": target, "settings": asdict(settings),
              "boundaries": ["Handset device center versus reconstructed optical center; visual tolerances include unresolved lens offset and are not statistical confidence intervals.",
                             "Heldout means excluded from this translation fit; these frames may participate in the original reconstruction.",
                             "Annotation binding is a retained provenance declaration, not surveyed ground truth.",
                             "No timing, calibration, world state, or saved alignment is changed."],
              "evidence": {**source["evidence"], **evidence, **annotations_evidence, "annotation_binding": _evidence(annotation_binding, binding_raw)},
              "original": {"supported_count": int(supported.sum()), "error_px": _metric(errors[supported]),
                           "train_error_px": _metric(errors[train]), "heldout_error_px": _metric(errors[heldout]),
                           "outside_visual_tolerance_count": int(np.count_nonzero(errors[supported] > tolerance[supported]))},
              "timing_sensitivity": {"source_limitations": timing_notes, "offsets": timing_rows, "common_supported_count": int(common.sum()),
                                     "common_domain_global_median_min_max_px": [float(np.min(np.median(timing_errors[:, common], axis=1))), float(np.max(np.median(timing_errors[:, common], axis=1)))] if common.any() else None}}
    candidate_errors = np.full(len(rows), np.nan)
    if settings.fit_translation_candidate:
        shift, converged, steps = fit_translation(points, observed, tolerance, train, C, K, settings.max_translation_axis_m)
        trial, valid = project(points + shift, C, K)
        candidate_errors = np.linalg.norm(trial - observed, axis=1)
        geometry, conflicts = assess_geometry(di, shift)
        retained = valid[supported].all()
        if not retained: conflicts.append("handset_projection_support_loss")
        if not converged: conflicts.append("translation_optimizer_not_converged")
        improved = retained and _metric(candidate_errors[heldout])["rms"] < _metric(errors[heldout])["rms"]
        if not improved: conflicts.append("heldout_pixel_rms_not_improved")
        result["translation_candidate"] = {"world_translation_m": shift.tolist(), "optimizer_converged": converged, "optimizer_evaluations": steps,
            "training_frame_ids": [rows[i]["frame_id"] for i in np.flatnonzero(train)], "heldout_frame_ids": [rows[i]["frame_id"] for i in np.flatnonzero(heldout)],
            "train_error_px": _metric(candidate_errors[train]) if valid[train].all() else None,
            "heldout_error_px": _metric(candidate_errors[heldout]) if valid[heldout].all() else None,
            "geometry": geometry, "conflicts": conflicts, "reference_nonregression_passed": not conflicts,
            "ready_to_replace_alignment": False, "status": "conflicting_references" if conflicts else "diagnostic_only_not_admitted"}
    result["annotations"] = []
    for i, row in enumerate(rows):
        reason = reasons[i] if supported[i] or not np.isfinite(points[i]).all() else "behind_or_too_close_to_camera"
        result["annotations"].append({**row, "support": reason, "supported": bool(supported[i]), "pose_span_s": spans[i],
            "split": "train" if train[i] else "heldout" if heldout[i] else "excluded",
            "original_projected_uv": predicted[i].tolist() if supported[i] else None,
            "original_error_px": float(errors[i]) if supported[i] else None,
            "candidate_error_px": float(candidate_errors[i]) if np.isfinite(candidate_errors[i]) else None})
    def clean(value):
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    result = clean(result)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "trajectory_reference_review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scan", "provider-manifest", "alignment-dir", "annotation-csv", "annotation-binding", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--heldout-start-s", required=True, type=float)
    parser.add_argument("--timing-allowance-s", required=True, type=float)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-pose-gap-s", type=float, default=.85)
    parser.add_argument("--fit-translation-candidate", action="store_true")
    parser.add_argument("--max-translation-axis-m", type=float, default=.5)
    args = parser.parse_args(argv)
    try:
        settings = TrajectoryReferenceReviewSettings(args.heldout_start_s, args.timing_allowance_s, args.allow_partial, args.max_pose_gap_s, args.fit_translation_candidate, args.max_translation_axis_m)
        result = review_trajectory_reference(args.scan, args.provider_manifest, args.alignment_dir, args.annotation_csv, args.annotation_binding, args.output_dir, settings)
    except (Error, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"trajectory reference review failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "original": result["original"], "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
