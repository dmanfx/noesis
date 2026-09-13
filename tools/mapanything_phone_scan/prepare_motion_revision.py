"""Create an explicit IMU-exposure-filtered review input, preserving its parent.

This opt-in selector uses rotation magnitude, not a calibrated inertial pose.
The excluded frames remain in the source recording and in the revision report.
No default preparation policy, saved scan pointer, or live-world binding changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from .imu_motion import load_native_motion
from .prepared_frame_identity import prepared_frame_identity


def select_motion_subset(
    frames: list[dict[str, Any]], maximum_exposure_rotation_deg: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter only available native exposure scores; preserve unknown evidence."""
    if not math.isfinite(maximum_exposure_rotation_deg) or not 0.5 <= maximum_exposure_rotation_deg <= 10:
        raise ValueError("experimental exposure-rotation limit must be between 0.5 and 10 degrees")
    retained, rejected = [], []
    for parent_index, original in enumerate(frames):
        row = copy.deepcopy(original)
        if row.get("index") != parent_index or row.get("frame_id") != prepared_frame_identity(parent_index, row.get("sha256", "")):
            raise ValueError("parent prepared frame identity is inconsistent")
        motion = row.get("imu_motion") or {}
        if motion.get("status") == "available":
            rotation = float(motion["rotation_during_exposure_estimate_rad"])
            if not math.isfinite(rotation) or rotation < 0:
                raise ValueError("native exposure rotation must be finite and nonnegative")
            if math.degrees(rotation) > maximum_exposure_rotation_deg:
                rejected.append({"original_frame": row, "reason": "exposure_rotation_exceeds_explicit_review_limit"})
                continue
        row["parent_prepared_index"] = parent_index
        row["parent_prepared_frame_id"] = row["frame_id"]
        row["index"] = len(retained)
        row["frame_id"] = prepared_frame_identity(row["index"], row["sha256"])
        quality = row.setdefault("quality", {})
        old_edge = quality.pop("edge_from_previous", None)
        quality["parent_edge_from_previous"] = old_edge
        # A removed view changes adjacency; never relabel its measured edge.
        same_predecessor = bool(retained and retained[-1]["parent_prepared_index"] == parent_index - 1)
        quality["edge_from_previous"] = old_edge if same_predecessor else None
        if retained and not same_predecessor:
            quality["motion_revision_gap_requires_reconstruction_validation"] = True
        retained.append(row)
    if len(retained) < 2:
        raise ValueError("exposure-rotation limit leaves fewer than two views")
    return retained, rejected


def prepare_motion_revision(source: Path, output: Path, maximum_exposure_rotation_deg: float) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("use a new review directory outside the source scan")
    manifest_path = source / "prepared_frames_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    capture_report = json.loads((source / "capture" / "capture_import.json").read_text())
    motion, summary = load_native_motion(source / "capture", capture_report)
    if motion is None:
        raise ValueError(f"native exposure-motion evidence unavailable: {summary.get('reason')}")
    for row in manifest["frames"]:
        image = (source / row["frame"]).resolve()
        if not image.is_relative_to(source) or hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("parent RGB path or image hash does not match its prepared manifest")
        row["imu_motion"] = motion.frame(row.get("capture_time_ns"))
    retained, rejected = select_motion_subset(manifest["frames"], maximum_exposure_rotation_deg)
    experiment = {
        "schema": "noesis.phone_scan.motion_selection_revision.v1", "review_only": True,
        "metric_pose_authority": False, "camera_imu_offset_measured": False,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "maximum_exposure_rotation_deg": maximum_exposure_rotation_deg,
        "policy": "explicit_native_exposure_rotation_exclusion", "motion_evidence": summary,
        "retained_count": len(retained), "rejected_count": len(rejected),
        "unknown_motion_retained_count": sum(r["imu_motion"].get("status") != "available" for r in retained),
        "rejected": rejected,
    }
    state = json.loads((source / "scan_state.json").read_text())
    # A derivative is not a newly associated companion capture. The captured
    # static reference must be built from the original scan/session association.
    revision_state = {"id": output.name, "parent_scan_id": state.get("id"),
                      "status": "ready", "provider": None, "outputs": None,
                      "active_revision": None, "alignment": None, "review_only": True}
    manifest["frames"] = retained
    manifest["parent_preparation"] = manifest.pop("preparation", None)
    manifest["preparation"] = {"strategy": "explicit_native_exposure_motion_subset_v1",
                               "frame_count": len(retained),
                               "parent_prepared_count": len(retained) + len(rejected),
                               "maximum_exposure_rotation_deg": maximum_exposure_rotation_deg,
                               "rejected_prepared_count": len(rejected)}
    manifest["motion_selection_revision"] = experiment
    output.mkdir(parents=True, exist_ok=False)
    # Direct consumers intentionally reject RGB symlinks escaping the scan.
    # Copy only these bounded selected images, never the large source video.
    for row in retained:
        for field in ("frame", "thumbnail"):
            relative = row.get(field)
            if relative and (source / relative).is_file():
                destination = output / relative
                if not destination.resolve().is_relative_to(output):
                    raise ValueError("prepared artifact path escapes the review directory")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, destination)
    (output / "capture").symlink_to(source / "capture", target_is_directory=True)
    (output / "prepared_frames_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output / "scan_state.json").write_text(json.dumps(revision_state, indent=2)+"\n")
    return experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-exposure-rotation-deg", type=float, required=True)
    args = parser.parse_args()
    result = prepare_motion_revision(args.scan_dir, args.output_dir, args.maximum_exposure_rotation_deg)
    print(json.dumps({k: v for k, v in result.items() if k != "rejected"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
