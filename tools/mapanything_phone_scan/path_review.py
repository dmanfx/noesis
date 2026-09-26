"""Refine and compare a retained phone path without calling it body ground truth.

Existing visual/VIO constraints produce a separately gated trajectory candidate.
Independent room alignment and recorded timing join it to paired Noesis tracks.
Original geometry, calibration, tracking state and capture evidence never change.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .trajectory_motion_review import (
    TrajectoryMotionReviewError,
    TrajectoryMotionReviewSettings,
    _load_provider,
    review_trajectory_motion,
)
from .walk_intent import effective_walk_intent, validate_target_reference
from .path_comparison import compare_review_path
from .path_refinement import refine_review_path


PATH_REVIEW_SCHEMA = "noesis.phone_scan.path_review.v1"
MAX_JSON_BYTES = 16 * 1024 * 1024


def bound_artifact(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected a retained scan-relative artifact")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("The retained artifact is missing or outside its scan")
    return path


def _manifest_binding(root: Path, state: dict[str, Any], explicit_manifest: Path | None = None) -> dict[str, Any]:
    outputs = state.get("outputs") or {}
    path = explicit_manifest.resolve() if explicit_manifest is not None else bound_artifact(root, (outputs.get("artifacts") or {}).get("manifest"))
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Reconstruction manifest exceeds the review size limit")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    return {"scan_id": state["id"], "manifest": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "manifest_selection": "explicit_offline_artifact" if explicit_manifest is not None else "saved_scan_output",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "provider": manifest.get("provider"), "model_id": manifest.get("model_id"),
            "coordinate_frame": manifest.get("coordinate_frame"),
            "pose_convention": manifest.get("pose_convention")}


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_path_review(scan_dir: Path, state: dict[str, Any], target_dir: Path,
                      target_state: dict[str, Any], output_dir: Path, *,
                      companion_dir: Path | None = None,
                      source_output_manifest: Path | None = None,
                      source_alignment_dir: Path | None = None,
                      target_output_manifest: Path | None = None,
                      target_alignment_dir: Path | None = None,
                      temporal_alignment_path: Path | None = None,
                      target_reference: dict[str, Any] | None = None,
                      revalidate_scaled_carrier=None) -> dict[str, Any]:
    """Refine supported poses and compare independently registered paired paths."""
    scan, target, output = scan_dir.resolve(), target_dir.resolve(), output_dir.resolve()
    if output.exists() or output == scan or output == target:
        raise ValueError("Use a new path-review directory; retained evidence is never overwritten")
    intent = effective_walk_intent(state)
    if intent and intent["mode"] == "path_refinement" and intent["target_scan_id"] != target_state["id"]:
        raise ValueError("Path capture and reference reconstruction disagree")
    selected_reference = (intent or {}).get("target_reference")
    if selected_reference is not None and target_reference is None:
        raise ValueError("Resolve the recorded PCF reference before review; raw output is not a substitute")
    if target_reference is not None:
        selection = validate_target_reference(target_reference["selection"])
        if target_reference["source_scan_id"] != target_state["id"]:
            raise ValueError("The selected PCF belongs to a different reference reconstruction")
        if selected_reference is not None and selected_reference != selection:
            raise ValueError("The selected PCF differs from the recorder's retained reference")
        if target_output_manifest is not None or target_alignment_dir is not None:
            raise ValueError("PCF and raw-provider reference overrides cannot be combined")
    source_binding = _manifest_binding(scan, state, source_output_manifest)
    if target_reference is not None:
        target_binding = {"scan_id": target_state["id"], "manifest_selection": "scene_prior_pcf",
                          "selection": selection, "label": target_reference["label"],
                          "manifest_sha256": selection["manifest_sha256"],
                          "source": target_reference["manifest"]["source"],
                          "coordinate_frame": target_reference["frame_binding"]["target_frame"],
                          "evidence": target_reference["evidence"]}
    else:
        target_binding = source_binding if target_state["id"] == state["id"] else _manifest_binding(target, target_state, target_output_manifest)
    manifest_path = source_output_manifest.resolve() if source_output_manifest is not None else bound_artifact(scan, source_binding["manifest"])
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"report": "path_review.json"}
    if target_reference is not None:
        # Snapshot the exact selected existing artifacts, not new scene geometry.
        # Recheck bytes at copy time so a concurrent change cannot be published
        # under the digest verified by the resolver.
        for role, filename in (("manifest", "reference_pcf_manifest.json"), ("points", "reference_pcf_points.glb")):
            asset = target_reference["assets"][role]
            path = Path(asset["path"])
            if path.is_symlink() or not 0 < asset["size_bytes"] <= 64 * 1024 * 1024 or path.stat().st_size != asset["size_bytes"]:
                raise ValueError("Selected PCF artifact changed before review")
            with path.open("rb") as stream:
                raw = stream.read(asset["size_bytes"] + 1)
            if len(raw) != asset["size_bytes"] or hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                raise ValueError("Selected PCF artifact changed before review")
            (output / filename).write_bytes(raw)
            artifacts[f"reference_pcf_{role}"] = filename
        _write(output / "reference_pcf_binding.json", {"selection": selection, "frame_binding": target_reference["frame_binding"],
                                                       "evidence": target_reference["evidence"]})
        artifacts["reference_pcf_binding"] = "reference_pcf_binding.json"
        target_binding["artifacts"] = {k: v for k, v in artifacts.items() if k.startswith("reference_pcf_")}
    limitations = [
        "A phone optical-center path is not a person's ground footprint; close-body instructions do not measure the offset.",
        "Internal agreement with a reconstruction is not independent physical-position validation.",
        "Phone, reference room and Noesis coordinate frames remain separate until their explicit registration is validated.",
        "HTTP clock probes do not establish phone/static-camera acquisition synchronization.",
        "The 0.10 m value is the requested error target, not achieved or measured accuracy.",
    ]
    visual: dict[str, Any] = {"status": "unavailable", "trajectory_origin": "camera_optical_center"}
    motion: dict[str, Any] = {"status": "unavailable", "position_refinement": False}
    refinement: dict[str, Any] = {"status": "needs_evidence", "position_refined": False}
    comparison: dict[str, Any] = {"status": "needs_evidence"}
    try:
        source = _load_provider(scan, manifest_path, allow_partial=True)
        prepared = source["prepared"]["frames"]
        rows = []
        for output_index, prepared_index in enumerate(source["selected"]):
            frame = prepared[prepared_index]
            timestamp = frame.get("capture_time_ns")
            rows.append({"provider_index": output_index, "prepared_frame_id": frame["frame_id"],
                         "timestamp_s": frame["timestamp_s"],
                         "capture_time_ns": str(timestamp) if type(timestamp) is int else None,
                         "source_frame_index": frame.get("source_frame_index"),
                         "window_index": int(source["windows"][output_index]),
                         "camera_to_world": source["poses"][output_index].tolist()})
        trajectory = {"schema": "noesis.phone_scan.review_camera_trajectory.v1",
                      "source": source_binding, "origin": "camera_optical_center",
                      "units": "provider_metric_meters", "acquisition_ns_encoding": "decimal_string",
                      "scope": source["scope"], "poses": rows, "interpolated": False,
                      "body_ground_reference": False, "accuracy_qualified": False}
        _write(output / "camera_trajectory.json", trajectory)
        artifacts["trajectory"] = "camera_trajectory.json"
        visual = {"status": "available", "trajectory_origin": "camera_optical_center",
                  "pose_count": len(rows), "scope": source["scope"], "evidence": source["evidence"],
                  "coordinate_frame": source_binding["coordinate_frame"], "interpolated": False}
        capture = state.get("capture") or {}
        if capture.get("capture_kind") == "android_camera_imu":
            # This is a fixed diagnostic split, not fitted calibration or an
            # accuracy holdout. The original reviewer keeps its output isolation.
            first, last = float(rows[0]["timestamp_s"]), float(rows[-1]["timestamp_s"])
            settings = TrajectoryMotionReviewSettings(heldout_start_s=first + 0.7 * (last - first), allow_partial=True)
            try:
                with tempfile.TemporaryDirectory(prefix="roomwalk-path-review-") as temporary:
                    diagnostic = Path(temporary) / "motion"
                    result = review_trajectory_motion(scan, manifest_path, diagnostic, settings)
                    for filename in ("motion_review.json", "intervals.csv"):
                        shutil.copyfile(diagnostic / filename, output / filename)
                    artifacts.update(motion="motion_review.json", motion_intervals="intervals.csv")
                motion = {"status": result["status"], "rotation": result["rotation"],
                          "speed_activity": result["speed_activity"], "timing": result["timing"],
                          "imu": result["imu"], "position_refinement": False,
                          "split_semantics": "70_percent_temporal_diagnostic_split_not_accuracy_validation"}
            except (ValueError, OSError, KeyError, TypeError) as exc:
                motion = {"status": "needs_evidence", "reason": str(exc)[:2000], "position_refinement": False}
        else:
            motion["reason"] = "Native acquisition-timed camera and IMU evidence is required for inertial consistency review"
        refined = refine_review_path(scan, source, state, output / "refinement",
                                     revalidate_scaled_carrier=revalidate_scaled_carrier)
        refinement_report = refined["report"]
        refinement = {"status": refined["status"], "position_refined": refined["accepted_camera_poses"] is not None,
                      "reason": refinement_report.get("rejection_reason"),
                      "accepted_visual_constraint_count": refinement_report.get("accepted_visual_constraint_count", 0),
                      "accepted_vio_constraint_count": refinement_report.get("accepted_vio_constraint_count", 0),
                      "vio_constraint_admission": refinement_report.get("vio_constraint_admission"),
                      "pose_changes": refinement_report.get("accepted_camera_pose_changes")}
        artifacts["refinement"] = "refinement/path_refinement_report.json"
        materialized = (refinement_report.get("refinement") or {}).get("materialized") or {}
        revalidation = (refinement_report.get("refinement") or {}).get("world_alignment_revalidation") or {}
        refined_alignment = Path(revalidation["transform_path"]).parent if refined["accepted_camera_poses"] is not None and revalidation.get("transform_path") else None
        try:
            compared = compare_review_path(scan, source, state, target, target_state, target_binding, output,
                                           companion_dir, refined_poses=refined["accepted_camera_poses"],
                                           source_alignment_dir=source_alignment_dir, target_alignment_dir=target_alignment_dir,
                                           temporal_alignment_path=temporal_alignment_path,
                                           refined_alignment_dir=refined_alignment,
                                           refined_output_manifest=Path(materialized["source_manifest"]) if refined_alignment else None,
                                           target_reference=target_reference)
            artifacts.update(compared["artifacts"])
            comparison = {key: compared.get(key) for key in ("status", "reason", "registration", "timing", "excluded_observations", "metric_semantics")}
            comparison.update(track_count=len(compared["tracks"]), matched_count=sum(t["matched_count"] for t in compared["tracks"]))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            comparison = {"status": "needs_alignment", "reason": str(exc)[:2000],
                          "action": "Align both this walk and the selected room reconstruction to the same Noesis world, then rerun path review."}
    except (TrajectoryMotionReviewError, OSError, KeyError, TypeError) as exc:
        visual["status"] = "needs_evidence"
        visual["reason"] = str(exc)[:2000]
        motion["reason"] = "The retained visual path must have explicit frame and pose provenance first"
        limitations.append("Legacy or incomplete pose artifacts are retained unchanged; missing semantics are not guessed.")
        refinement["reason"] = comparison["reason"] = visual["reason"]

    companion = state.get("companion_capture") or {}
    vio = state.get("vio") or {}
    report = {
        "schema": PATH_REVIEW_SCHEMA, "review_only": True, "status": comparison["status"],
        "scan_id": state["id"], "capture_intent": intent,
        "capture_intent_source": state.get("walk_intent_source", "legacy_unspecified"),
        "source": source_binding, "reference_reconstruction": target_binding,
        "visual_path": visual, "imu_consistency": motion, "path_refinement": refinement, "path_comparison": comparison,
        "sensor_refined_path": {"status": "retained" if vio.get("status") == "complete" else "not_available",
                                "artifact": (vio.get("results") or {}).get("artifact"),
                                "dense_artifact": (vio.get("results") or {}).get("dense_artifact"),
                                "coordinate_frame": "vio_world" if vio.get("status") == "complete" else None,
                                "aligned_to_reference_room": False},
        "paired_noesis": {"session_id": companion.get("session_id"), "camera_id": companion.get("camera_id"),
                          "artifacts": deepcopy(companion.get("artifacts") or {}),
                          "clocks_joined": comparison.get("timing", {}).get("status") in {"callback_clock_review", "light_cue_review"},
                          "synchronization_verified": False, "person_identity_verified": False},
        "body_ground_reference": {"status": "not_established", "carry_protocol_declared": (intent or {}).get("carry_protocol"),
                                  "phone_to_body_offset_measured": False},
        "accuracy": {"target_m": 0.1, "noticeable_error_m": 0.2, "measured_position_error_m": None,
                     "qualified": False, "reason": "Needs independent placement, body offset, identity, timing and room registration checks"},
        "alignment_artifacts": deepcopy(((state.get("alignment") or {}).get("results") or {}).get("artifacts") or {}),
        "raw_evidence": deepcopy(state.get("capture") or {}),
        "raw_streams_modified": False, "promotes_live_world": False,
        "artifacts": artifacts, "limitations": limitations,
    }
    _write(output / "path_review.json", report)
    return report


def main() -> int:
    """Explicit offline reuse of retained processing, without editing scan state."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--target-scan-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-output-manifest", type=Path)
    parser.add_argument("--source-alignment-dir", type=Path)
    parser.add_argument("--target-output-manifest", type=Path)
    parser.add_argument("--target-alignment-dir", type=Path)
    parser.add_argument("--companion-dir", type=Path)
    parser.add_argument("--temporal-alignment", type=Path)
    args = parser.parse_args()
    state = json.loads((args.scan_dir / "scan_state.json").read_text())
    target_dir = args.target_scan_dir or args.scan_dir
    target = json.loads((target_dir / "scan_state.json").read_text())
    result = build_path_review(args.scan_dir, state, target_dir, target, args.output_dir,
                              companion_dir=args.companion_dir, source_output_manifest=args.source_output_manifest,
                              source_alignment_dir=args.source_alignment_dir, target_output_manifest=args.target_output_manifest,
                              target_alignment_dir=args.target_alignment_dir, temporal_alignment_path=args.temporal_alignment)
    print(json.dumps({key: result[key] for key in ("status", "path_refinement", "path_comparison")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
