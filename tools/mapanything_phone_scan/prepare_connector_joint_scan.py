#!/usr/bin/env python3
"""Materialize a prepared-view manifest for a saved RoomWalk addition.

The RoomWalk addition already contains one MapAnything joint reconstruction
whose leading views are exact bridge images and whose remaining views are the
new adaptive keyframes.  This command records that exact ordered RGB carrier
without copying or rewriting source images, so DA3 and PCF consume precisely
the same view sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ConnectorPreparedScanError(RuntimeError):
    """Raised when a saved addition cannot define one immutable view carrier."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConnectorPreparedScanError(f"{path} is not a JSON object")
    return value


def prepare(
    *,
    base_scan_dir: Path,
    revision_manifest_path: Path,
    joint_mapanything_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_scan_dir = base_scan_dir.resolve()
    revision_manifest_path = revision_manifest_path.resolve()
    joint_mapanything_manifest_path = joint_mapanything_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ConnectorPreparedScanError(f"output directory exists: {output_dir}")
    revision = _load_json(revision_manifest_path)
    joint = _load_json(joint_mapanything_manifest_path)
    frames = joint.get("frames")
    joint_record = revision.get("joint_inference")
    bridge_record = revision.get("bridge")
    if not isinstance(frames, list) or not isinstance(joint_record, dict):
        raise ConnectorPreparedScanError("revision lacks its joint view sequence")
    if not isinstance(bridge_record, dict):
        raise ConnectorPreparedScanError("revision lacks bridge provenance")
    expected_count = int(joint_record.get("view_count") or 0)
    bridge_count = int(joint_record.get("bridge_view_count") or 0)
    added_count = int(joint_record.get("added_view_count") or 0)
    if len(frames) != expected_count or expected_count != bridge_count + added_count:
        raise ConnectorPreparedScanError("joint view counts are inconsistent")
    if bridge_count < 2 or added_count < 2:
        raise ConnectorPreparedScanError("connector needs bridge and new views")

    prepared_frames: list[dict[str, Any]] = []
    for index, row in enumerate(frames):
        if not isinstance(row, dict) or int(row.get("index", -1)) != index:
            raise ConnectorPreparedScanError(f"joint view {index} is malformed")
        source_value = str(row.get("source_frame") or "")
        source = Path(source_value)
        source = source if source.is_absolute() else base_scan_dir / source
        source = source.resolve()
        try:
            source.relative_to(base_scan_dir)
        except ValueError as exc:
            raise ConnectorPreparedScanError(
                f"joint view {index} leaves the base scan root"
            ) from exc
        if not source.is_file():
            raise ConnectorPreparedScanError(f"joint view is missing: {source}")
        is_bridge = index < bridge_count
        expected_fragment = "bridge_inputs" if is_bridge else "/frames/"
        if expected_fragment not in source.as_posix():
            raise ConnectorPreparedScanError(
                f"joint view {index} violates bridge/new ordering"
            )
        prepared_frames.append(
            {
                "index": index,
                "frame": str(source),
                "timestamp_s": row.get("timestamp_s"),
                "fixed_camera_anchor": False,
                "joint_view_index": index,
                "source_kind": "living_bridge" if is_bridge else "connector_walk",
                "sha256": _sha256(source),
            }
        )

    output_dir.mkdir(parents=True)
    manifest = {
        "schema": "noesis.pcf.connector.prepared_views.v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "frame_count": len(prepared_frames),
        "bridge_view_count": bridge_count,
        "added_view_count": added_count,
        "base_scan_id": revision.get("base_scan_id"),
        "supplement_id": revision.get("supplement_id"),
        "revision_id": revision.get("revision_id"),
        "source_joint_mapanything_manifest": str(joint_mapanything_manifest_path),
        "source_joint_mapanything_manifest_sha256": _sha256(
            joint_mapanything_manifest_path
        ),
        "source_revision_manifest": str(revision_manifest_path),
        "source_revision_manifest_sha256": _sha256(revision_manifest_path),
        "frames": prepared_frames,
    }
    manifest_path = output_dir / "prepared_frames_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scan-dir", type=Path, required=True)
    parser.add_argument("--revision-manifest", type=Path, required=True)
    parser.add_argument("--joint-mapanything-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = prepare(
        base_scan_dir=args.base_scan_dir,
        revision_manifest_path=args.revision_manifest,
        joint_mapanything_manifest_path=args.joint_mapanything_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "revision_id": result["revision_id"],
                "frame_count": result["frame_count"],
                "bridge_view_count": result["bridge_view_count"],
                "added_view_count": result["added_view_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
