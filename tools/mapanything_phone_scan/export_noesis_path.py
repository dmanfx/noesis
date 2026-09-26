"""Export measured Noesis ground paths from a paired RoomWalk capture.

This is an offline view of retained canonical DS9 messages, not a replay or a
new world producer. No person is selected as the phone carrier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .path_comparison import _tracks


FIELDS = (
    "camera_id", "source_id", "track_key", "source_epoch", "tracker_id", "lifecycle_generation",
    "stable_id", "observer_sequence", "tracking_publication_sequence",
    "frame_id", "observed_at_us", "media_pts_ns", "world_x_m", "world_y_m",
    "world_z_m", "world_frame_revision", "world_transform_sha256",
    "world_source", "trail_segment_id", "break_before",
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cohorts(path: Path, camera_id: str, source_id: int) -> dict:
    tracking: dict[int, tuple[int, int, int]] = {}
    world: dict[int, tuple[int, int, int]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            message = json.loads(line)["message"]
            kind = message.get("type")
            if kind not in {"tracking", "world_snapshot"}:
                continue
            cohort = message.get("cohort") or {}
            sequence = cohort.get("tracking_publication_sequence")
            identity = (cohort.get("source_id"), cohort.get("frame_id"), cohort.get("observed_at_us"))
            if type(sequence) is not int or identity[0] != source_id:
                raise ValueError("Invalid retained tracking/world cohort identity")
            if kind == "tracking" and message.get("camera_id") != camera_id:
                raise ValueError("Retained tracking camera differs from paired camera")
            destination = tracking if kind == "tracking" else world
            if sequence in destination:
                raise ValueError("Duplicate retained tracking/world cohort")
            destination[sequence] = identity
    sequences = list(tracking)
    if (not sequences or sequences != list(range(sequences[0], sequences[-1] + 1))
            or any(tracking.get(sequence) != identity for sequence, identity in world.items())):
        raise ValueError("Retained tracking/world cohorts are discontinuous or mismatched")
    missing_world = [sequence for sequence in sequences if sequence not in world]
    if missing_world not in ([], [sequences[-1]]):
        raise ValueError("An interior tracking cohort has no retained world snapshot")
    return {"tracking_publications": len(tracking), "world_snapshots": len(world),
            "missing_terminal_world_snapshot": bool(missing_world)}


def export(scan_dir: Path, companion_dir: Path, output_dir: Path) -> dict:
    scan_dir, companion_dir, output_dir = (path.resolve() for path in (scan_dir, companion_dir, output_dir))
    session_path = companion_dir / "session.json"
    alignment_path = scan_dir / "alignment" / "alignment_report.json"
    video_path = companion_dir / "static_camera.mkv"
    tracking_path = companion_dir / "tracking.ndjson"
    timing_path = companion_dir / "packet_timing.jsonl"
    session, alignment = _json(session_path), _json(alignment_path)
    target = alignment.get("target") or {}
    camera_id = target.get("camera_id")
    observer = (session.get("tracking") or {}).get("observer") or {}
    if (session.get("status") != "complete" or session.get("session_id") != target.get("companion_session_id")
            or (session.get("phone") or {}).get("scan_id") != scan_dir.name
            or (session.get("camera") or {}).get("authority", {}).get("camera_id") != camera_id
            or alignment.get("status") != "passed"
            or (session.get("tracking") or {}).get("partial") is not False
            or observer.get("partial") is not False):
        raise ValueError("Paired capture, scan, camera, or passed alignment binding is incomplete")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing export: {output_dir}")
    for path in (video_path, tracking_path, timing_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked retained capture artifact: {path}")

    source_id = session["camera"]["authority"]["source_id"]
    cohort_status = _cohorts(tracking_path, camera_id, source_id)
    groups, rejected, raw_evidence = _tracks(tracking_path, camera_id, target)
    rows = []
    per_lifecycle = []
    for key, group in groups.items():
        previous_segment = None
        for index, row in enumerate(group["rows"]):
            segment = row["trail_segment_id"]
            rows.append({
                "camera_id": camera_id,
                "source_id": row["source_id"],
                "track_key": key,
                "source_epoch": group["source_epoch"],
                "tracker_id": group["tracker_id"],
                "lifecycle_generation": group["lifecycle_generation"],
                "stable_id": row["stable_id"],
                "observer_sequence": row["observer_sequence"],
                "tracking_publication_sequence": row["publication_sequence"],
                "frame_id": row["frame_id"],
                "observed_at_us": row["time_ns"] // 1000,
                "media_pts_ns": row["media_pts_ns"],
                "world_x_m": row["world"][0],
                "world_y_m": row["world"][1],
                "world_z_m": row["world"][2],
                "world_frame_revision": target["world_frame_revision"],
                "world_transform_sha256": target["camera_frame_binding"]["transform_sha256"],
                "world_source": row["world_source"],
                "trail_segment_id": segment,
                "break_before": index == 0 or row["trail_break_required"] or segment != previous_segment,
            })
            previous_segment = segment
        per_lifecycle.append({
            "track_key": key, "stable_ids": group["stable_ids"],
            "measured_rows": len(group["rows"]),
            "invalid_world_rows": group["invalid_world_count"],
        })
    if not rows:
        raise ValueError("No revision-bound measured ground footprints in this capture")
    rows.sort(key=lambda row: (row["observed_at_us"], row["observer_sequence"], row["track_key"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "measured_ground_paths.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "noesis.phone_scan.offline_noesis_path.v1",
        "status": "ready",
        "source": "canonical_live_ds9_observer_during_paired_static_recording",
        "replayed": False,
        "scan_id": scan_dir.name,
        "companion_session_id": session["session_id"],
        "camera_id": camera_id,
        "source_id": source_id,
        "world_frame": target["world_frame"],
        "world_frame_revision": target["world_frame_revision"],
        "world_transform_sha256": target["camera_frame_binding"]["transform_sha256"],
        "quantity": "person_ground_footprint",
        "timing": "DS9 callback observation time; not verified camera acquisition time",
        "phone_carrier_identified": False,
        "bev_frames_saved": False,
        "measured_rows": len(rows),
        "cohorts": cohort_status,
        "lifecycles": per_lifecycle,
        "rejected_track_rows": rejected,
        "artifacts": {
            "csv": {"path": str(csv_path), "sha256": _hash(csv_path)},
            "raw_tracking": raw_evidence,
            "static_video": {"path": str(video_path), "sha256": _hash(video_path)},
            "packet_timing": {"path": str(timing_path), "sha256": _hash(timing_path)},
            "session": {"path": str(session_path), "sha256": _hash(session_path)},
            "alignment": {"path": str(alignment_path), "sha256": _hash(alignment_path)},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--companion-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = export(args.scan_dir, args.companion_dir, args.output_dir)
    print(json.dumps({"manifest": str(args.output_dir.resolve() / "manifest.json"),
                      "measured_rows": report["measured_rows"], "lifecycles": report["lifecycles"]}, indent=2))


if __name__ == "__main__":
    main()
