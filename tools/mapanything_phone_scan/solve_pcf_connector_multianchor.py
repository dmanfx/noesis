#!/usr/bin/env python3
"""Solve one connector RoomWalk against two already joined PCF room anchors.

The input registration reports contain depth-backed PnP observations from the
same connector carrier into each accepted room.  This command expresses both
observation sets in the existing fixed-room world, then solves one floor-locked
planar pose graph with a smooth per-view correction field.  It does not use ICP
or nearest-neighbour point-cloud fitting.

Rejected solutions remain useful only as explicitly review-only candidates;
the report never promotes a failed graph to canonical geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan import pcf_multiroom_pose_graph as graph  # noqa: E402
from tools.mapanything_phone_scan.register_pcf_rooms import (  # noqa: E402
    _manifest_transform,
)
from noesis_core.coordinate_frames import (  # noqa: E402
    RevisionedFrame,
    revisioned_transform_sha256,
)


class ConnectorMultiAnchorError(RuntimeError):
    """Raised when the two anchor reports cannot form one connector graph."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConnectorMultiAnchorError(f"{path} does not contain a JSON object")
    return value


def _content_identity_digest(
    *, manifest_sha256: str, frame: dict[str, str]
) -> str:
    """Bind a declared frame token to the exact manifest that names it."""
    payload = {
        "coordinate_frame": frame["coordinate_frame"],
        "frame_id": frame["frame_id"],
        "manifest_sha256": manifest_sha256,
        "revision": frame["revision"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest_frame(
    path: Path, *, digest: str, label: str
) -> dict[str, Any]:
    """Read a manifest-owned frame identity and derive its content token."""
    manifest = _load_json(path)
    value = manifest.get("frame_identity")
    if not isinstance(value, dict):
        raise ConnectorMultiAnchorError(
            f"{label} manifest has no content-bound frame_identity"
        )
    fields = ("frame_id", "revision", "coordinate_frame")
    if any(not isinstance(value.get(key), str) or not value[key] for key in fields):
        raise ConnectorMultiAnchorError(
            f"{label} manifest frame_identity is incomplete"
        )
    frame = {key: str(value[key]) for key in fields}
    if frame["coordinate_frame"] != "backend_world_m":
        raise ConnectorMultiAnchorError(
            f"{label} manifest frame_identity uses unsupported coordinate frame"
        )
    try:
        RevisionedFrame(frame["frame_id"], frame["revision"])
    except ValueError as exc:
        raise ConnectorMultiAnchorError(
            f"{label} manifest frame_identity has an invalid frame token"
        ) from exc
    return {
        **frame,
        "manifest_sha256": digest,
        "content_identity_sha256": _content_identity_digest(
            manifest_sha256=digest, frame=frame
        ),
        "baseline_frame_identity": manifest.get("baseline_frame_identity"),
    }


def _manifest_baseline_frame(
    manifest_frame: dict[str, Any], *, role: str
) -> dict[str, str]:
    """Derive a deterministic baseline identity from a manifest's bytes."""
    frame = manifest_frame.get("frame", manifest_frame)
    if not isinstance(frame, dict):
        raise ConnectorMultiAnchorError("manifest baseline has no frame identity")
    declared = manifest_frame.get("baseline_frame_identity")
    if isinstance(declared, dict) and declared.get("role") == role:
        fields = ("frame_id", "revision", "coordinate_frame")
        if any(not isinstance(declared.get(key), str) or not declared[key] for key in fields):
            raise ConnectorMultiAnchorError(
                f"manifest baseline {role} identity is incomplete"
            )
        baseline = {key: str(declared[key]) for key in fields}
        if baseline["coordinate_frame"] != "backend_world_m":
            raise ConnectorMultiAnchorError(
                f"manifest baseline {role} uses unsupported coordinate frame"
            )
        try:
            RevisionedFrame(baseline["frame_id"], baseline["revision"])
        except ValueError as exc:
            raise ConnectorMultiAnchorError(
                f"manifest baseline {role} identity is invalid"
            ) from exc
        return baseline
    manifest_sha256 = str(
        manifest_frame.get("manifest_sha256") or manifest_frame.get("sha256") or ""
    )
    if len(manifest_sha256) != 64:
        raise ConnectorMultiAnchorError("manifest baseline is missing its digest")
    revision = hashlib.sha256(
        json.dumps(
            {
                "coordinate_frame": frame["coordinate_frame"],
                "frame_id": frame["frame_id"],
                "manifest_sha256": manifest_sha256,
                "role": role,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "frame_id": frame["frame_id"],
        "revision": revision,
        "coordinate_frame": frame["coordinate_frame"],
    }


def _candidate_transform(report: dict[str, Any], label: str) -> np.ndarray:
    pose_graph = report.get("pose_graph")
    candidate = pose_graph.get("candidate") if isinstance(pose_graph, dict) else None
    value = (
        candidate.get("global_transform_moving_to_fixed_row_major")
        if isinstance(candidate, dict)
        else report.get("global_transform_moving_to_fixed_row_major")
    )
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ConnectorMultiAnchorError(f"{label} has no finite candidate transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ConnectorMultiAnchorError(f"{label} candidate has an invalid last row")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=2e-3):
        raise ConnectorMultiAnchorError(f"{label} candidate is not rigid")
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=2e-3):
        raise ConnectorMultiAnchorError(f"{label} candidate is reflected")
    return transform


def _load_frame_binding_input(path: Path | None) -> dict[str, Any] | None:
    """Validate explicit endpoint provenance for a canonical edge.

    The graph estimates a moving-baseline -> fixed-baseline transform.  A
    caller may bind it to another pair of frames only with immutable source
    and target manifest digests plus an accepted, explicit composition.  A
    caller-supplied final matrix is intentionally not accepted as provenance.
    """
    if path is None:
        return None
    payload = _load_json(path)
    if payload.get("schema") != "noesis.pcf.connector_frame_binding_input.v3":
        raise ConnectorMultiAnchorError(
            "frame binding input schema is unsupported"
        )

    def _frame(value: Any, label: str) -> dict[str, str]:
        if not isinstance(value, dict):
            raise ConnectorMultiAnchorError(f"frame binding input {label} is missing")
        fields = ("frame_id", "revision", "coordinate_frame")
        if any(not isinstance(value.get(key), str) or not value[key] for key in fields):
            raise ConnectorMultiAnchorError(
                f"frame binding input {label} lacks exact frame identity"
            )
        try:
            RevisionedFrame(str(value["frame_id"]), str(value["revision"]))
        except ValueError as exc:
            raise ConnectorMultiAnchorError(
                f"frame binding input {label} has an invalid frame token"
            ) from exc
        return {key: str(value[key]) for key in fields}

    def _plane(value: Any, label: str, frame: dict[str, str]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ConnectorMultiAnchorError(f"frame binding input {label} is missing")
        try:
            normal = np.asarray(value["normal"], dtype=np.float64)
            offset = float(value["offset_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConnectorMultiAnchorError(f"frame binding input {label} is malformed") from exc
        if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset):
            raise ConnectorMultiAnchorError(f"frame binding input {label} is not finite")
        if not np.isclose(float(np.linalg.norm(normal)), 1.0, atol=1e-6):
            raise ConnectorMultiAnchorError(f"frame binding input {label} normal is not unit length")
        return {
            "frame_id": frame["frame_id"],
            "revision": frame["revision"],
            "normal": normal.astype(float).tolist(),
            "offset_m": offset,
        }

    def _manifest(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ConnectorMultiAnchorError(f"frame binding input {label} is missing")
        manifest_path = Path(str(value.get("path") or "")).expanduser().resolve()
        digest = str(value.get("sha256") or "").lower()
        if not manifest_path.is_file() or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ConnectorMultiAnchorError(
                f"frame binding input {label} lacks an exact manifest path/digest"
            )
        actual_digest = _sha256(manifest_path)
        if actual_digest != digest:
            raise ConnectorMultiAnchorError(
                f"frame binding input {label} digest does not match its manifest"
            )
        identity = _frame(value.get("frame"), f"{label}.frame")
        manifest_identity = _manifest_frame(
            manifest_path, digest=digest, label=label
        )
        if identity != {
            key: manifest_identity[key]
            for key in ("frame_id", "revision", "coordinate_frame")
        }:
            raise ConnectorMultiAnchorError(
                f"{label}.frame does not match the frame_identity in its manifest"
            )
        return {
            "path": str(manifest_path),
            "sha256": digest,
            "frame": identity,
            "content_identity_sha256": manifest_identity["content_identity_sha256"],
            "baseline_frame_identity": manifest_identity.get("baseline_frame_identity"),
        }

    def _rigid_col_major(value: Any, label: str) -> list[float]:
        values = np.asarray(value, dtype=np.float64)
        if values.size != 16 or not np.isfinite(values).all():
            raise ConnectorMultiAnchorError(f"{label} transform is malformed")
        transform = values.reshape((4, 4), order="F")
        rotation = transform[:3, :3]
        if (
            not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4)
            or not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=2e-4)
        ):
            raise ConnectorMultiAnchorError(
                f"{label} transform is not a proper rigid transform"
            )
        return transform.flatten(order="F").astype(float).tolist()

    def _accepted_component(
        value: Any,
        *,
        label: str,
        source_frame: dict[str, str],
        target_frame: dict[str, str],
        manifest_sha256: str,
    ) -> dict[str, Any]:
        """Load an existing accepted registration report as endpoint evidence."""
        if source_frame == target_frame and value is None:
            identity = np.eye(4, dtype=np.float64).flatten(order="F").tolist()
            return {
                "status": "derived_identity_from_manifest_baseline",
                "source_frame": source_frame,
                "target_frame": target_frame,
                "manifest_sha256": manifest_sha256,
                "target_from_source_col_major": identity,
            }
        if not isinstance(value, dict):
            return {
                "status": "unproven_endpoint_conversion",
                "source_frame": source_frame,
                "target_frame": target_frame,
                "manifest_sha256": manifest_sha256,
            }
        report_path = Path(
            str(value.get("acceptance_report_path") or "")
        ).expanduser().resolve()
        if not value.get("acceptance_report_path"):
            return {
                "status": "unproven_endpoint_conversion",
                "source_frame": source_frame,
                "target_frame": target_frame,
                "manifest_sha256": manifest_sha256,
            }
        if not report_path.is_file():
            raise ConnectorMultiAnchorError(
                f"{label} must name an existing accepted registration report"
            )
        if str(value.get("manifest_sha256") or "").lower() != manifest_sha256:
            raise ConnectorMultiAnchorError(
                f"{label} is not tied to the exact baseline manifest"
            )
        try:
            evidence_report = _load_json(report_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ConnectorMultiAnchorError(
                f"{label} is not a readable accepted registration report"
            ) from exc
        pose_graph = evidence_report.get("pose_graph")
        holdout = evidence_report.get("holdout")
        if (
            evidence_report.get("schema")
            != "noesis.pcf.connector_multianchor_pose_graph.v1"
            or evidence_report.get("status") != "passed"
            or evidence_report.get("accepted_for_canonical_use") is not True
            or evidence_report.get("disposition")
            != "validated_cross_session_registration"
            or not isinstance(pose_graph, dict)
            or pose_graph.get("accepted") is not True
            or pose_graph.get("reason_codes") not in ([], ())
            or not isinstance(holdout, dict)
            or holdout.get("source") != "pose_graph"
            or holdout.get("status") != "passed"
        ):
            raise ConnectorMultiAnchorError(
                f"{label} registration report is not accepted by the existing holdout gate"
            )
        exact = evidence_report.get("bound_transform")
        if not isinstance(exact, dict):
            raise ConnectorMultiAnchorError(
                f"{label} accepted report lacks bound_transform evidence"
            )
        def _exact_frame(value: Any) -> dict[str, str]:
            frame = _frame(value, f"{label}.bound_transform.frame")
            return frame
        if (
            _exact_frame(exact.get("source_frame")) != source_frame
            or _exact_frame(exact.get("target_frame")) != target_frame
        ):
            raise ConnectorMultiAnchorError(
                f"{label} accepted evidence endpoints do not match the manifest-derived baselines"
            )
        matrix = _rigid_col_major(
            exact.get("target_from_source_col_major"),
            f"{label}.accepted_report",
        )
        expected_digest = revisioned_transform_sha256(
            RevisionedFrame(source_frame["frame_id"], source_frame["revision"]),
            RevisionedFrame(target_frame["frame_id"], target_frame["revision"]),
            tuple(matrix),
        )
        if exact.get("target_from_source_sha256") != expected_digest:
            raise ConnectorMultiAnchorError(
                f"{label} accepted report transform digest is invalid"
            )
        if value.get("target_from_source_col_major") is not None:
            declared = _rigid_col_major(
                value["target_from_source_col_major"], f"{label}.declared"
            )
            if not np.allclose(
                np.asarray(declared), np.asarray(matrix), atol=1e-9, rtol=0.0
            ):
                raise ConnectorMultiAnchorError(
                    f"{label} matrix does not match accepted binding evidence"
                )
        return {
            "status": "accepted_frame_binding_evidence",
            "acceptance_report_path": str(report_path),
            "acceptance_report_sha256": _sha256(report_path),
            "source_frame": source_frame,
            "target_frame": target_frame,
            "manifest_sha256": manifest_sha256,
            "target_from_source_col_major": matrix,
            "target_from_source_sha256": expected_digest,
        }

    source_frame = _frame(payload.get("source_frame"), "source_frame")
    target_frame = _frame(payload.get("target_frame"), "target_frame")
    source_manifest = _manifest(payload.get("source_manifest"), "source_manifest")
    target_manifest = _manifest(payload.get("target_manifest"), "target_manifest")
    source_manifest_frame = source_manifest["frame"]
    target_manifest_frame = target_manifest["frame"]
    if source_frame != source_manifest_frame:
        raise ConnectorMultiAnchorError(
            "source_frame must be derived from the moving manifest identity"
        )
    if target_frame != target_manifest_frame:
        raise ConnectorMultiAnchorError(
            "target_frame must be derived from the target manifest identity"
        )
    composition = payload.get("composition")
    if not isinstance(composition, dict):
        raise ConnectorMultiAnchorError("frame binding input composition is missing")
    if (
        composition.get("method")
        != "target_from_fixed_baseline @ graph_edge @ moving_baseline_from_source"
        or composition.get("source_frame") != source_frame
        or composition.get("target_frame") != target_frame
        or composition.get("source_manifest_sha256") != source_manifest["sha256"]
        or composition.get("target_manifest_sha256") != target_manifest["sha256"]
    ):
        raise ConnectorMultiAnchorError(
            "frame binding input lacks an exact manifest-backed composition"
        )
    moving_baseline_frame = _manifest_baseline_frame(
        source_manifest, role="moving_baseline"
    )
    fixed_baseline_frame = _manifest_baseline_frame(
        target_manifest, role="fixed_baseline"
    )
    moving_component = _accepted_component(
        composition.get("moving_baseline_from_source"),
        label="moving_baseline_from_source",
        source_frame=source_frame,
        target_frame=moving_baseline_frame,
        manifest_sha256=source_manifest["sha256"],
    )
    target_component = _accepted_component(
        composition.get("target_from_fixed_baseline"),
        label="target_from_fixed_baseline",
        source_frame=fixed_baseline_frame,
        target_frame=target_frame,
        manifest_sha256=target_manifest["sha256"],
    )
    canonical_eligible = all(
        component["status"]
        in {"accepted_frame_binding_evidence", "derived_identity_from_manifest_baseline"}
        for component in (moving_component, target_component)
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "source_frame": source_frame,
        "target_frame": target_frame,
        "canonical_eligible": canonical_eligible,
        "source_manifest": source_manifest,
        "target_manifest": target_manifest,
        "composition": {
            "status": "accepted" if canonical_eligible else "review_only",
            "accepted_for_canonical_use": canonical_eligible,
            "method": composition["method"],
            "source_frame": source_frame,
            "target_frame": target_frame,
            "source_manifest_sha256": source_manifest["sha256"],
            "target_manifest_sha256": target_manifest["sha256"],
            "moving_baseline_frame": moving_baseline_frame,
            "fixed_baseline_frame": fixed_baseline_frame,
            "moving_baseline_from_source": moving_component,
            "target_from_fixed_baseline": target_component,
        },
        "source_floor_plane": _plane(
            payload.get("source_floor_plane"), "source_floor_plane", source_frame
        ),
        "target_floor_plane": _plane(
            payload.get("target_floor_plane"), "target_floor_plane", target_frame
        ),
    }


def _holdout_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    """Select a complete leave-one-view evaluation owned by the pose graph."""
    leaveout = candidate.get("leave_one_whole_moving_view_out")
    folds = leaveout.get("folds") if isinstance(leaveout, dict) else None
    passed = bool(isinstance(folds, list) and folds and all(
        isinstance(fold, dict) and fold.get("passed") is True for fold in folds
    ))
    return {
        "source": "pose_graph",
        "evaluation": "leave_one_whole_moving_view_out",
        "status": "passed" if passed else "failed_or_unavailable",
        "fold_count": len(folds) if isinstance(folds, list) else 0,
        "passed_fold_count": sum(
            1 for fold in folds
            if isinstance(fold, dict) and fold.get("passed") is True
        ) if isinstance(folds, list) else 0,
    }


def _room_poses(room: dict[str, Any], output_from_room: np.ndarray) -> dict[int, np.ndarray]:
    raw_root = Path(str(room["pcf_raw_root"]))
    world_manifest = Path(str(room["world_manifest"]))
    raw_paths = sorted(raw_root.glob("view_*.npz"))
    if not raw_paths:
        raise ConnectorMultiAnchorError(f"no PCF views in {raw_root}")
    world_from_local = _manifest_transform(world_manifest)
    poses: dict[int, np.ndarray] = {}
    for index, path in enumerate(raw_paths):
        with np.load(path, allow_pickle=False) as row:
            if "camera_pose" not in row.files:
                raise ConnectorMultiAnchorError(f"{path} has no camera_pose")
            local_pose = np.asarray(row["camera_pose"], dtype=np.float64)
        if local_pose.shape != (4, 4) or not np.isfinite(local_pose).all():
            raise ConnectorMultiAnchorError(f"malformed camera pose in {path}")
        poses[index] = output_from_room @ world_from_local @ local_pose
    return poses


def _observations(
    report: dict[str, Any],
    *,
    output_from_anchor: np.ndarray,
    fixed_view_offset: int,
    anchor_name: str,
) -> list[graph.PnPTransformObservation]:
    rows = report.get("admitted_pnp")
    if not isinstance(rows, list) or not rows:
        raise ConnectorMultiAnchorError(f"{anchor_name} report has no admitted PnP")
    result: list[graph.PnPTransformObservation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConnectorMultiAnchorError(
                f"{anchor_name} PnP row {index} is not an object"
            )
        transform = np.asarray(
            row.get("transform_moving_to_fixed_row_major"), dtype=np.float64
        )
        if transform.shape != (4, 4):
            raise ConnectorMultiAnchorError(
                f"{anchor_name} PnP row {index} has no transform"
            )
        result.append(
            graph.PnPTransformObservation(
                moving_view=int(row["moving_view"]),
                fixed_view=fixed_view_offset + int(row["fixed_view"]),
                direction=f"{anchor_name}:{row['direction']}",
                transform_moving_to_fixed=output_from_anchor @ transform,
                inlier_count=int(row["inlier_count"]),
                reprojection_median_px=float(row["reprojection_median_px"]),
                reprojection_p80_px=float(row["reprojection_p80_px"]),
            )
        )
    return result


def solve(
    *,
    kitchen_report_path: Path,
    living_report_path: Path,
    kitchen_family_manifest_path: Path,
    output_dir: Path,
    frame_binding_input_path: Path | None = None,
) -> dict[str, Any]:
    frame_binding = _load_frame_binding_input(frame_binding_input_path)
    kitchen_report = _load_json(kitchen_report_path)
    living_report = _load_json(living_report_path)
    kitchen_family = _load_json(kitchen_family_manifest_path)
    moving_kitchen = kitchen_report.get("moving_room")
    moving_living = living_report.get("moving_room")
    if not isinstance(moving_kitchen, dict) or not isinstance(moving_living, dict):
        raise ConnectorMultiAnchorError("registration reports lack moving-room records")
    identity_fields = ("pcf_raw_root", "world_manifest", "scan_id", "prior_id")
    mismatches = [
        field
        for field in identity_fields
        if moving_kitchen.get(field) != moving_living.get(field)
    ]
    if mismatches:
        raise ConnectorMultiAnchorError(
            f"anchor reports do not use the same connector carrier: {mismatches}"
        )
    moving_room = moving_kitchen
    kitchen_room = kitchen_report.get("fixed_room")
    living_room = living_report.get("fixed_room")
    if not isinstance(kitchen_room, dict) or not isinstance(living_room, dict):
        raise ConnectorMultiAnchorError("registration reports lack fixed-room records")

    if frame_binding is not None:
        expected_manifests = (
            ("source_manifest", moving_room.get("world_manifest"), "moving connector"),
            ("target_manifest", kitchen_family_manifest_path, "fixed family"),
        )
        for key, expected_path_value, label in expected_manifests:
            expected_path = Path(str(expected_path_value or "")).expanduser().resolve()
            declared = frame_binding[key]
            if declared["path"] != str(expected_path):
                raise ConnectorMultiAnchorError(
                    f"{key} is not the exact {label} manifest used by the graph"
                )
            if declared["sha256"] != _sha256(expected_path):
                raise ConnectorMultiAnchorError(
                    f"{key} digest does not match the graph input manifest"
                )

    moving_manifest = kitchen_family.get("moving_room")
    if not isinstance(moving_manifest, dict):
        raise ConnectorMultiAnchorError("Kitchen/Family manifest lacks moving_room")
    kitchen_to_family = np.asarray(
        moving_manifest.get("global_world_correction_row_major"), dtype=np.float64
    )
    if kitchen_to_family.shape != (4, 4):
        raise ConnectorMultiAnchorError("Kitchen/Family manifest lacks its transform")

    connector_to_kitchen = _candidate_transform(kitchen_report, "Kitchen anchor")
    connector_to_living = _candidate_transform(living_report, "Living anchor")
    living_to_family = (
        kitchen_to_family @ connector_to_kitchen @ np.linalg.inv(connector_to_living)
    )

    moving_poses = _room_poses(moving_room, np.eye(4, dtype=np.float64))
    kitchen_poses = _room_poses(kitchen_room, kitchen_to_family)
    living_offset = 1_000
    living_poses = {
        living_offset + index: pose
        for index, pose in _room_poses(living_room, living_to_family).items()
    }
    fixed_poses = {**kitchen_poses, **living_poses}
    observations = [
        *_observations(
            kitchen_report,
            output_from_anchor=kitchen_to_family,
            fixed_view_offset=0,
            anchor_name="kitchen",
        ),
        *_observations(
            living_report,
            output_from_anchor=living_to_family,
            fixed_view_offset=living_offset,
            anchor_name="living",
        ),
    ]

    result = graph.solve_cross_session_pose_graph(
        observations,
        moving_poses,
        fixed_poses,
        vertical_translation_m=0.0,
    )
    candidate = result.report.get("candidate")
    if not isinstance(candidate, dict):
        raise ConnectorMultiAnchorError(
            "combined graph had insufficient support and produced no candidate"
        )
    holdout = _holdout_evidence(candidate)
    bound_transform: dict[str, Any] | None = None
    if (
        result.accepted
        and frame_binding is not None
        and frame_binding["canonical_eligible"]
        and holdout["status"] == "passed"
    ):
        source = RevisionedFrame(
            frame_binding["source_frame"]["frame_id"],
            frame_binding["source_frame"]["revision"],
        )
        target = RevisionedFrame(
            frame_binding["target_frame"]["frame_id"],
            frame_binding["target_frame"]["revision"],
        )
        # Compose the graph edge with the explicitly accepted endpoint maps.
        # Each matrix maps column-vector points in the named source frame to
        # the next frame in the expression below.
        graph_edge = np.asarray(
            candidate["global_transform_moving_to_fixed_row_major"],
            dtype=np.float64,
        )
        moving_baseline_from_source = np.asarray(
            frame_binding["composition"]["moving_baseline_from_source"][
                "target_from_source_col_major"
            ],
            dtype=np.float64,
        ).reshape((4, 4), order="F")
        target_from_fixed_baseline = np.asarray(
            frame_binding["composition"]["target_from_fixed_baseline"][
                "target_from_source_col_major"
            ],
            dtype=np.float64,
        ).reshape((4, 4), order="F")
        composed = (
            target_from_fixed_baseline
            @ graph_edge
            @ moving_baseline_from_source
        )
        composed_col_major = composed.flatten(order="F").astype(float).tolist()
        bound_transform = {
            "source_frame": dict(frame_binding["source_frame"]),
            "target_frame": dict(frame_binding["target_frame"]),
            "target_from_source_col_major": composed_col_major,
            "target_from_source_sha256": revisioned_transform_sha256(
                source, target, composed_col_major
            ),
            "source_floor_plane": dict(frame_binding["source_floor_plane"]),
            "target_floor_plane": dict(frame_binding["target_floor_plane"]),
            "composition": dict(frame_binding["composition"]),
            "source_manifest": dict(frame_binding["source_manifest"]),
            "target_manifest": dict(frame_binding["target_manifest"]),
            "graph_edge_moving_baseline_to_fixed_baseline_row_major": graph_edge.tolist(),
        }
    canonical_accepted = bool(
        result.accepted and holdout["status"] == "passed" and bound_transform is not None
    )
    report_reason_codes = list(result.reason_codes)
    if result.accepted and holdout["status"] != "passed":
        report_reason_codes.append("canonical_holdout_unavailable")
    if result.accepted and frame_binding is None:
        report_reason_codes.append("canonical_frame_binding_missing")
    if (
        result.accepted
        and frame_binding is not None
        and not frame_binding["canonical_eligible"]
    ):
        report_reason_codes.append("canonical_endpoint_conversion_unproven")
    report_reason_codes = sorted(set(report_reason_codes))
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "noesis.pcf.connector_multianchor_pose_graph.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed" if canonical_accepted else "rejected",
        "accepted_for_canonical_use": canonical_accepted,
        "disposition": (
            "validated_cross_session_registration"
            if canonical_accepted
            else "review_only_rejected_registration_candidate"
        ),
        "method": (
            "two_endpoint_depth_pnp_floor_locked_planar_smooth_per_view_pose_graph"
        ),
        "whole_cloud_icp_used": False,
        "nearest_neighbor_surface_fitting_used": False,
        "fixed_world": "family_accepted_backend_world_m",
        "reason_codes": report_reason_codes,
        "pose_graph": result.report,
        "candidate": candidate,
        "holdout": holdout,
        "frame_binding_input": (
            {
                "path": frame_binding["path"],
                "sha256": frame_binding["sha256"],
                "source_frame": frame_binding["source_frame"],
                "target_frame": frame_binding["target_frame"],
                "source_manifest": frame_binding["source_manifest"],
                "target_manifest": frame_binding["target_manifest"],
                "composition": frame_binding["composition"],
            }
            if frame_binding is not None
            else None
        ),
        "derived_endpoint_transforms": {
            "connector_to_kitchen_row_major": connector_to_kitchen.tolist(),
            "connector_to_living_row_major": connector_to_living.tolist(),
            "kitchen_to_family_row_major": kitchen_to_family.tolist(),
            "living_to_family_row_major": living_to_family.tolist(),
        },
        "support_input": {
            "connector_view_count": len(moving_poses),
            "kitchen_fixed_view_count": len(kitchen_poses),
            "living_fixed_view_count": len(living_poses),
            "kitchen_pnp_direction_count": len(kitchen_report["admitted_pnp"]),
            "living_pnp_direction_count": len(living_report["admitted_pnp"]),
            "combined_pnp_direction_count": len(observations),
            "living_fixed_view_id_offset": living_offset,
        },
        "sources": {
            "kitchen_registration_report": {
                "path": str(kitchen_report_path),
                "sha256": _sha256(kitchen_report_path),
                "status": kitchen_report.get("status"),
            },
            "living_registration_report": {
                "path": str(living_report_path),
                "sha256": _sha256(living_report_path),
                "status": living_report.get("status"),
            },
            "kitchen_family_manifest": {
                "path": str(kitchen_family_manifest_path),
                "sha256": _sha256(kitchen_family_manifest_path),
                "status": kitchen_family.get("status"),
            },
            "connector": moving_room,
            "kitchen": kitchen_room,
            "living": living_room,
        },
    }
    if bound_transform is not None:
        report["bound_transform"] = bound_transform
    output_path = output_dir / "registration_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitchen-report", type=Path, required=True)
    parser.add_argument("--living-report", type=Path, required=True)
    parser.add_argument("--kitchen-family-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frame-binding-input",
        type=Path,
        help=(
            "explicit validated source/target frame identities, endpoint "
            "transform, and floor planes for canonical binding"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = solve(
        kitchen_report_path=args.kitchen_report,
        living_report_path=args.living_report,
        kitchen_family_manifest_path=args.kitchen_family_manifest,
        output_dir=args.output_dir,
        frame_binding_input_path=args.frame_binding_input,
    )
    metrics = report["candidate"].get("training_observation_error", {})
    print(
        json.dumps(
            {
                "status": report["status"],
                "reason_codes": report["reason_codes"],
                "training_translation_p80_m": metrics.get("translation_p80_m"),
                "training_yaw_p80_deg": metrics.get("yaw_p80_deg"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
