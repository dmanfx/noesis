#!/usr/bin/env python3
"""Freeze explicit retained Living179 records; never run reconstruction.

This bootstrap utility refuses existing destinations. It hashes small named
metadata/camera files and prepared images, and records size/mtime for the
explicit 537 provider arrays. Existing whole-view-set hashes are retained;
large raw arrays and the original video are not recursively rehashed/copied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def freeze(repo: Path, storage: Path, destination: Path) -> dict:
    base = storage / "pcf_august_more_views_20261009"
    scan = base / "scan_adaptive_1920"
    lane = base / "attempt_64_32"
    static = repo / "data/mapanything_phone_scans/20260912-210621-2f755956/paired_static_reference/paired_static_reference_c681919be7e2a303"
    da3 = lane / "da3"
    ma = lane / "pcf/mapanything_da3_pose_sparse_depth"
    pcf = lane / "pcf/prior_conditioned_consensus_da3_carrier"
    read = lambda p: json.loads(p.read_text())

    def portable(value):
        if isinstance(value, str):
            return value.replace(str(repo), "${NOESIS_REPO_ROOT}").replace(str(storage), "${NOESIS_STORAGE_ROOT}")
        if isinstance(value, dict):
            return {key: portable(v) for key, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        return value

    metadata = read(lane / "lane_manifest.json")
    prepared_path = scan / "prepared_frames_manifest.json"
    prepared = read(prepared_path)
    if (metadata["status"] != "complete" or not metadata["review_only"]
            or metadata["accepted_for_canonical_use"] or len(prepared["frames"]) != 179
            or digest(prepared_path) != metadata["prepared_manifest"]["sha256"]
            or digest(static / "calibration.json") != metadata["calibration"]["sha256"]):
        raise ValueError("Retained review lane identity/status differs from the reviewed baseline")
    evidence = []

    def record(path: Path, purpose: str, hash_content: bool = True):
        stat = path.stat()
        item = {"path": portable(str(path)), "purpose": purpose, "size_bytes": stat.st_size}
        item.update({"sha256": digest(path)} if hash_content else {"mtime_ns": stat.st_mtime_ns})
        evidence.append(item)

    named = [
        (prepared_path, "exact prepared cohort"),
        (lane / "lane_manifest.json", "completed reviewed candidate, not canonical admission"),
        (base / "STATUS.md", "retained review and limitations"),
        (base / "comparison/REPORT.md", "matched comparison interpretation"),
        (base / "comparison/metrics.json", "matched comparison metrics"),
        (da3 / "scan_outputs_manifest.json", "DA3 identity and per-view camera outputs"),
        (ma / "variant_manifest.json", "conditioned MapAnything identity and priors"),
        (pcf / "consensus_manifest.json", "selected fused PCF provenance"),
        (pcf / "scan_outputs_manifest.json", "selected fused output identity"),
        (lane / "evaluation/evaluation_metrics.json", "comparison and pose-transfer transform"),
        (repo / "config/camera_calibration.json", "current bound calibration identity, read-only guard"),
        (repo / "config/ply_alignment.json", "current bound world alignment identity, read-only guard"),
    ]
    for folder in ("alignment_da3", "alignment_consensus"):
        for name in ("alignment_report.json", "phone_ma_to_noesis_world.json"):
            named.append((lane / folder / name, folder + " retained registration"))
    for name in ("manifest.json", "room_points_meta.json", "calibration.json", "keyframe.png", "keyframe_valid_mask.png"):
        named.append((static / name, "independent fixed-camera evidence"))
    for provider in (da3, ma, pcf):
        named.append((provider / "camera_solution.npz", "exact provider camera solution"))
    for path, purpose in named:
        record(path, purpose)
    for path, purpose in [(pcf / "surfel_points.npz", "accepted multiview-supported PCF surfels"),
                          (static / "room_points.npz", "independent model-derived static reconstruction"),
                          (scan / "phone_walk.mp4", "original video; retained SHA below, no re-encoding")]:
        record(path, purpose, hash_content=False)
    for index in range(179):
        for provider in (da3, ma, pcf):
            record(provider / "raw" / f"view_{index:04d}.npz", "provider raw identity; stat guard plus retained manifest identity", False)
    frames = []
    for source in prepared["frames"]:
        path = scan / source["frame"]
        if digest(path) != source["sha256"]:
            raise ValueError(f"Prepared image changed: {source['frame_id']}")
        frames.append({key: source[key] for key in ("index", "frame_id", "sha256", "timestamp_s", "timestamp_source", "width", "height")}
                     | {"path": portable(str(path)), "original_image_path": portable(str(path)),
                        "source_video_path": portable(str(scan / "phone_walk.mp4")),
                        "selection_reason": source.get("selection_reason")})
    held = [29, 59, 89, 119, 149, 177]
    eligible = [index for index in range(179) if index not in held]
    medium = [eligible[round(i * (len(eligible) - 1) / 47)] for i in range(48)]
    sparse = [medium[round(i * 47 / 11)] for i in range(12)]
    ids = lambda indices: [frames[index]["frame_id"] for index in indices]
    da3_transform = read(lane / "alignment_da3/phone_ma_to_noesis_world.json")
    consensus_transform = read(lane / "alignment_consensus/phone_ma_to_noesis_world.json")
    evaluation = read(lane / "evaluation/evaluation_metrics.json")
    alignment = evaluation["candidates"]["prior_conditioned_consensus"]["alignment"]
    consensus = read(pcf / "consensus_manifest.json")
    baseline = {
        "schema": "noesis.atlas_bench.baseline.v1", "id": "living179-reviewed-20261009",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_issue": "https://github.com/dmanfx/noesis/issues/37",
        "selection": {"status": "completed retained offline review candidate",
                      "review_only": True, "accepted_for_canonical_use": False,
                      "reason": "current complete 179-view source with reviewed coverage comparison and unchanged static alignment gates",
                      "accuracy_winner_claimed": False,
                      "original_scan_id": "20260802-162254-8bcc7dd7",
                      "preparation_revision": "pcf_august_more_views_20261009/scan_adaptive_1920",
                      "lane_revision": "attempt_64_32", "camera_id": "living-room"},
        "prepared_manifest_path": portable(str(prepared_path)),
        "prepared_manifest_sha256": metadata["prepared_manifest"]["sha256"],
        "original_video_retained_sha256": prepared["parent_capture"]["original_video_sha256"],
        "provider_paths": {"da3_raw": portable(str(da3 / "raw")),
                           "conditioned_mapanything": portable(str(ma)), "pcf_fused": portable(str(pcf)),
                           "pcf_geometry": portable(str(pcf / "surfel_points.npz"))},
        "provider_semantics": {"pcf_method": "prior_conditioned_consensus_da3_carrier",
                               "phone_walk_only": consensus["phone_walk_only"],
                               "static_camera_data_used": consensus["static_camera_data_used"],
                               "conditioned_sources_are_independent": False,
                               "surfel_count": consensus["surfel_fusion"]["surfel_count"],
                               "da3_confidence": "DA3 Base learned multiview confidence; Metric non-sky is validity only"},
        "coordinates": {"distance_unit": "m", "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
                        "depth": "optical_axis_z", "da3_frame": da3_transform["source_coordinate_frame"],
                        "conditioned_mapanything_frame": "backend_world_m pose-carried",
                        "pcf_frame": consensus_transform["source_coordinate_frame"],
                        "backend_frame": da3_transform["target_coordinate_frame"],
                        "atlas_frame": "separate estimated gravity-aligned gauge; metric accuracy and registration unverified"},
        "transforms": {
            "da3_to_backend": portable(da3_transform),
            "pcf_to_backend_direct_alignment": portable(consensus_transform),
            "pcf_to_backend_evaluation_pose_transfer": portable(alignment),
            "conditioned_mapanything_reference_transforms": portable(read(ma / "variant_manifest.json")["reference_transforms"]),
            "rule": "Keep direct fit and evaluation pose-transfer distinct; never apply DA3 transform directly to fused PCF gauge"},
        "validation": {"static_revision": static.name, "static_root": portable(str(static)),
                       "world_revision": da3_transform["target_binding"]["world_frame_revision"],
                       "camera_frame_binding": da3_transform["target_binding"]["camera_frame_binding"],
                       "current_config_is_guarded_not_modified": True,
                       "independent_of_phone_inference": True,
                       "surveyed_truth": False, "static_participated_in_alignment": True,
                       "historical_internal_even_to_odd_independent": False},
        "cohorts": {"smoke": ids([0, 2]), "sparse": ids(sparse), "medium": ids(medium), "full": ids(eligible)},
        "frames": frames, "evidence": evidence,
        "limitations": [
            "Review candidate, not a canonical admission or physical accuracy winner",
            "Atlas holdouts were already seen by retained PCF/DA3; no independent PCF holdout claim",
            "September model-derived static reference postdates August phone walk and participated in alignment",
            "Current source comparison changes preparation and inference context as well as frame count",
            "IMU unavailable; camera-derived up is not calibrated IMU gravity",
            "Raw arrays use size/mtime guards plus retained manifest and view-set identity; no new full-array hashing",
        ],
    }
    holdouts = {"schema": "noesis.atlas_bench.holdouts.v1", "baseline_id": baseline["id"],
                "frame_ids": ids(held), "indices": held,
                "frames": [frames[index] for index in held],
                "policy": "Never upload as Atlas input/context/appearance anchor, including derived A05/A06/Cxx",
                "independent_of_retained_pcf": False,
                "static_reference": {"root": portable(str(static)), "role": "registration/validation only; never phone inference input"}}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "baseline.json").write_text(json.dumps(portable(baseline), indent=2) + "\n")
    (destination / "holdouts.json").write_text(json.dumps(portable(holdouts), indent=2) + "\n")
    lock = {"baseline_sha256": digest(destination / "baseline.json"),
            "holdouts_sha256": digest(destination / "holdouts.json")}
    (destination / "baseline.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return {"destination": str(destination), "evidence_files": len(evidence), "frames": len(frames), **lock}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(freeze(arguments.repo, arguments.storage_root, arguments.destination), indent=2))
