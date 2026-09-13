from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.mapanything_phone_scan import solve_pcf_connector_multianchor as producer
from tools.mapanything_phone_scan.solve_pcf_connector_multianchor import ConnectorMultiAnchorError
from noesis.scene_prior_builder import build_accepted_room_to_home_binding
from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.scene_prior import metric_frame_revision_sha256
from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256


_TARGET_REVISION = metric_frame_revision_sha256(
    "backend_world_m", (0.0, 1.0, 0.0), 0.0, ("c" * 64, "d" * 64)
)


def _write_world_manifest(
    path: Path,
    *,
    family: bool = False,
    frame_revision: str = "connector-r1",
) -> None:
    payload = {
        "alignment": {
            "rotation_row_major": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "frame_identity": {
            "frame_id": "backend_world_m",
            "revision": frame_revision,
            "coordinate_frame": "backend_world_m",
        },
    }
    if family:
        payload = {
            "moving_room": {"global_world_correction_row_major": np.eye(4).tolist()},
            "frame_identity": {
                "frame_id": "backend_world_m",
                "revision": _TARGET_REVISION,
                "coordinate_frame": "backend_world_m",
            },
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _accepted_component(
    *,
    report_path: Path,
    source_frame: dict[str, str],
    target_frame: dict[str, str],
    matrix: np.ndarray,
    manifest_sha256: str,
) -> dict[str, object]:
    matrix_values = matrix.flatten(order="F").astype(float).tolist()
    edge_sha = revisioned_transform_sha256(
        RevisionedFrame(source_frame["frame_id"], source_frame["revision"]),
        RevisionedFrame(target_frame["frame_id"], target_frame["revision"]),
        tuple(matrix_values),
    )
    acceptance_report = {
        "schema": "noesis.pcf.connector_multianchor_pose_graph.v1",
        "status": "passed",
        "accepted_for_canonical_use": True,
        "disposition": "validated_cross_session_registration",
        "pose_graph": {"accepted": True, "reason_codes": []},
        "holdout": {"source": "pose_graph", "status": "passed"},
        "bound_transform": {
            "source_frame": source_frame,
            "target_frame": target_frame,
            "target_from_source_col_major": matrix_values,
            "target_from_source_sha256": edge_sha,
            "source_floor_plane": {
                "normal": [0.0, 1.0, 0.0],
                "offset_m": 0.0,
            },
            "target_floor_plane": {
                "normal": [0.0, 1.0, 0.0],
                "offset_m": 0.0,
            },
        },
    }
    report_path.write_text(
        json.dumps(acceptance_report, indent=2), encoding="utf-8"
    )
    return {
        "acceptance_report_path": str(report_path),
        "manifest_sha256": manifest_sha256,
        "target_from_source_col_major": matrix_values,
    }


def _frame_binding(
    path: Path,
    source_manifest: Path,
    target_manifest: Path,
    moving_component: dict[str, object],
    target_component: dict[str, object],
) -> None:
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    target_digest = hashlib.sha256(target_manifest.read_bytes()).hexdigest()
    source_frame = {
        "frame_id": "backend_world_m",
        "revision": "connector-r1",
        "coordinate_frame": "backend_world_m",
    }
    target_frame = {
        "frame_id": "backend_world_m",
        "revision": _TARGET_REVISION,
        "coordinate_frame": "backend_world_m",
    }
    path.write_text(
        json.dumps(
            {
                "schema": "noesis.pcf.connector_frame_binding_input.v3",
                "source_frame": source_frame,
                "target_frame": target_frame,
                "source_manifest": {
                    "path": str(source_manifest),
                    "sha256": source_digest,
                    "frame": source_frame,
                },
                "target_manifest": {
                    "path": str(target_manifest),
                    "sha256": target_digest,
                    "frame": target_frame,
                },
                "composition": {
                    "method": "target_from_fixed_baseline @ graph_edge @ moving_baseline_from_source",
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "source_manifest_sha256": source_digest,
                    "target_manifest_sha256": target_digest,
                    "moving_baseline_from_source": moving_component,
                    "target_from_fixed_baseline": target_component,
                },
                "source_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
                "target_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
            }
        ),
        encoding="utf-8",
    )


def _report(path: Path, fixed_name: str, moving_manifest: Path, fixed_manifest: Path) -> None:
    identity = np.eye(4, dtype=np.float64).tolist()
    observations = []
    for view in range(5):
        for direction in ("forward", "reverse"):
            observations.append(
                {
                    "moving_view": view,
                    "fixed_view": view,
                    "direction": direction,
                    "transform_moving_to_fixed_row_major": identity,
                    "inlier_count": 30,
                    "reprojection_median_px": 1.0,
                    "reprojection_p80_px": 2.0,
                }
            )
    payload = {
        "moving_room": {
            "pcf_raw_root": str(moving_manifest.parent / "connector-raw"),
            "world_manifest": str(moving_manifest),
            "world_manifest_sha256": hashlib.sha256(moving_manifest.read_bytes()).hexdigest(),
            "scan_id": "connector",
            "prior_id": "connector-prior",
        },
        "fixed_room": {
            "name": fixed_name,
            "pcf_raw_root": str(fixed_manifest.parent / f"{fixed_name}-raw"),
            "world_manifest": str(fixed_manifest),
        },
        "pose_graph": {"candidate": {"global_transform_moving_to_fixed_row_major": identity}},
        "admitted_pnp": observations,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_views(root: Path, raw_name: str) -> None:
    raw = root / raw_name
    raw.mkdir()
    for index, position in enumerate(((0.0, 0.0, 0.0), (0.0, 0.0, 0.6), (0.5, 0.0, 0.6), (0.5, 0.0, 1.2), (1.0, 0.0, 1.2))):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = position
        np.savez(raw / f"view_{index:04d}.npz", camera_pose=pose)


def test_producer_emits_exact_bound_transform_only_with_supplied_frames(
    tmp_path: Path, monkeypatch
) -> None:
    kitchen_report = tmp_path / "kitchen.json"
    living_report = tmp_path / "living.json"
    family_manifest = tmp_path / "family.json"
    frame_binding = tmp_path / "frame_binding.json"
    moving_manifest = tmp_path / "connector-world.json"
    kitchen_manifest = tmp_path / "kitchen-world.json"
    _write_world_manifest(moving_manifest)
    _write_world_manifest(kitchen_manifest)
    _write_views(tmp_path, "connector-raw")
    _write_views(tmp_path, "kitchen-raw")
    _write_views(tmp_path, "living-raw")
    _report(kitchen_report, "kitchen", moving_manifest, kitchen_manifest)
    _report(living_report, "living", moving_manifest, tmp_path / "living-world.json")
    _write_world_manifest(tmp_path / "living-world.json")
    family_manifest.write_text(
        json.dumps(
            {
                "moving_room": {
                    "global_world_correction_row_major": np.eye(4).tolist()
                },
                "frame_identity": {
                    "frame_id": "backend_world_m",
                    "revision": _TARGET_REVISION,
                    "coordinate_frame": "backend_world_m",
                },
            }
        ),
        encoding="utf-8",
    )
    source_digest = hashlib.sha256(moving_manifest.read_bytes()).hexdigest()
    target_digest = hashlib.sha256(family_manifest.read_bytes()).hexdigest()
    moving_baseline = producer._manifest_baseline_frame(
        {"frame_id": "backend_world_m", "revision": "connector-r1", "coordinate_frame": "backend_world_m", "manifest_sha256": source_digest},
        role="moving_baseline",
    )
    fixed_baseline = producer._manifest_baseline_frame(
        {"frame_id": "backend_world_m", "revision": _TARGET_REVISION, "coordinate_frame": "backend_world_m", "manifest_sha256": target_digest},
        role="fixed_baseline",
    )
    moving_matrix = np.eye(4, dtype=np.float64)
    moving_matrix[:3, :3] = _rotation_y(31.0)
    moving_matrix[:3, 3] = [1.0, 0.0, 0.0]
    target_matrix = np.eye(4, dtype=np.float64)
    target_matrix[:3, :3] = _rotation_y(-47.0)
    target_matrix[:3, 3] = [0.0, 0.0, 2.0]
    moving_component = _accepted_component(
        report_path=tmp_path / "moving-acceptance.json",
        source_frame={"frame_id": "backend_world_m", "revision": "connector-r1", "coordinate_frame": "backend_world_m"},
        target_frame=moving_baseline,
        matrix=moving_matrix,
        manifest_sha256=source_digest,
    )
    target_component = _accepted_component(
        report_path=tmp_path / "target-acceptance.json",
        source_frame=fixed_baseline,
        target_frame={"frame_id": "backend_world_m", "revision": _TARGET_REVISION, "coordinate_frame": "backend_world_m"},
        matrix=target_matrix,
        manifest_sha256=target_digest,
    )
    _frame_binding(
        frame_binding,
        moving_manifest,
        family_manifest,
        moving_component,
        target_component,
    )

    report = producer.solve(
        kitchen_report_path=kitchen_report,
        living_report_path=living_report,
        kitchen_family_manifest_path=family_manifest,
        output_dir=tmp_path / "output",
        frame_binding_input_path=frame_binding,
    )
    assert report["accepted_for_canonical_use"] is True
    assert report["disposition"] == "validated_cross_session_registration"
    bound = report["bound_transform"]
    assert bound["source_frame"]["frame_id"] == "backend_world_m"
    assert bound["target_frame"]["revision"] == _TARGET_REVISION
    assert len(bound["target_from_source_col_major"]) == 16
    assert report["holdout"]["status"] == "passed"
    composed = np.asarray(bound["target_from_source_col_major"], dtype=np.float64).reshape((4, 4), order="F")
    np.testing.assert_allclose(
        composed,
        target_matrix @ np.eye(4) @ moving_matrix,
        atol=1e-9,
    )

    # Feed the actual producer report into WO-1's acceptance helper.  The
    # builder must accept the composed edge and its producer-owned digest.
    binding = build_accepted_room_to_home_binding(
        artifact_revision_id="connector-artifact-r1",
        source_frame_revision="connector-r1",
        target_coordinate_revision=_TARGET_REVISION,
        target_from_source_col_major=bound["target_from_source_col_major"],
        source_floor_normal=(0.0, 1.0, 0.0),
        source_floor_offset_m=0.0,
        target_floor_normal=(0.0, 1.0, 0.0),
        target_floor_offset_m=0.0,
        source_camera_calibration_sha256="a" * 64,
        source_world_alignment_sha256="b" * 64,
        source_camera_calibration_physical_sha256="c" * 64,
        source_world_alignment_physical_sha256="d" * 64,
        target_revision_id="family-r1",
        target_revision_metadata_sha256="e" * 64,
        metric_frame_provenance=(
            ArtifactFingerprint(role="camera_calibration_physical", sha256="c" * 64),
            ArtifactFingerprint(role="world_alignment_physical", sha256="d" * 64),
        ),
        acceptance_report=report,
    )
    assert binding.target_from_source_col_major == tuple(bound["target_from_source_col_major"])


def test_producer_keeps_accepted_graph_review_only_without_frame_binding(
    tmp_path: Path, monkeypatch
) -> None:
    kitchen_report = tmp_path / "kitchen.json"
    living_report = tmp_path / "living.json"
    family_manifest = tmp_path / "family.json"
    moving_manifest = tmp_path / "connector-world.json"
    kitchen_manifest = tmp_path / "kitchen-world.json"
    _write_world_manifest(moving_manifest)
    _write_world_manifest(kitchen_manifest)
    _write_views(tmp_path, "connector-raw")
    _write_views(tmp_path, "kitchen-raw")
    _write_views(tmp_path, "living-raw")
    _report(kitchen_report, "kitchen", moving_manifest, kitchen_manifest)
    _report(living_report, "living", moving_manifest, tmp_path / "living-world.json")
    _write_world_manifest(tmp_path / "living-world.json")
    family_manifest.write_text(
        json.dumps(
            {
                "moving_room": {
                    "global_world_correction_row_major": np.eye(4).tolist()
                },
                "frame_identity": {
                    "frame_id": "backend_world_m",
                    "revision": _TARGET_REVISION,
                    "coordinate_frame": "backend_world_m",
                },
            }
        ),
        encoding="utf-8",
    )
    report = producer.solve(
        kitchen_report_path=kitchen_report,
        living_report_path=living_report,
        kitchen_family_manifest_path=family_manifest,
        output_dir=tmp_path / "output",
    )
    assert report["accepted_for_canonical_use"] is False
    assert "bound_transform" not in report
    assert "canonical_frame_binding_missing" in report["reason_codes"]


def test_binding_rejects_arbitrary_final_matrix_without_accepted_composition(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source.json"
    target_manifest = tmp_path / "target.json"
    _write_world_manifest(source_manifest, frame_revision="connector-r1")
    _write_world_manifest(target_manifest, frame_revision=_TARGET_REVISION)
    binding_path = tmp_path / "binding.json"
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    target_digest = hashlib.sha256(target_manifest.read_bytes()).hexdigest()
    binding_path.write_text(
        json.dumps(
            {
                "schema": "noesis.pcf.connector_frame_binding_input.v3",
                "source_frame": {
                    "frame_id": "backend_world_m",
                    "revision": "connector-r1",
                    "coordinate_frame": "backend_world_m",
                },
                "target_frame": {
                    "frame_id": "backend_world_m",
                    "revision": _TARGET_REVISION,
                    "coordinate_frame": "backend_world_m",
                },
                "source_manifest": {
                    "path": str(source_manifest),
                    "sha256": source_digest,
                    "frame": {
                        "frame_id": "backend_world_m",
                        "revision": "connector-r1",
                        "coordinate_frame": "backend_world_m",
                    },
                },
                "target_manifest": {
                    "path": str(target_manifest),
                    "sha256": target_digest,
                    "frame": {
                        "frame_id": "backend_world_m",
                        "revision": _TARGET_REVISION,
                        "coordinate_frame": "backend_world_m",
                    },
                },
                "composition": {
                    "method": "target_from_fixed_baseline @ graph_edge @ moving_baseline_from_source",
                    "source_frame": {
                        "frame_id": "backend_world_m",
                        "revision": "connector-r1",
                        "coordinate_frame": "backend_world_m",
                    },
                    "target_frame": {
                        "frame_id": "backend_world_m",
                        "revision": _TARGET_REVISION,
                        "coordinate_frame": "backend_world_m",
                    },
                    "source_manifest_sha256": source_digest,
                    "target_manifest_sha256": target_digest,
                    "target_from_source_col_major": np.eye(4).reshape(-1, order="F").tolist(),
                },
                "source_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
                "target_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["composition"].pop("moving_baseline_from_source", None)
    payload["composition"]["target_from_fixed_baseline"] = {
        "manifest_sha256": target_digest,
    }
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    normalized = producer._load_frame_binding_input(binding_path)
    assert normalized is not None
    assert normalized["canonical_eligible"] is False
    assert normalized["composition"]["status"] == "review_only"


def test_binding_rejects_stale_manifest_or_wrong_manifest_frame_identity(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source.json"
    target_manifest = tmp_path / "target.json"
    _write_world_manifest(source_manifest, frame_revision="connector-r1")
    _write_world_manifest(target_manifest, frame_revision=_TARGET_REVISION)
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    target_digest = hashlib.sha256(target_manifest.read_bytes()).hexdigest()
    payload = {
        "schema": "noesis.pcf.connector_frame_binding_input.v3",
        "source_frame": {
            "frame_id": "backend_world_m",
            "revision": "connector-r1",
            "coordinate_frame": "backend_world_m",
        },
        "target_frame": {
            "frame_id": "backend_world_m",
            "revision": _TARGET_REVISION,
            "coordinate_frame": "backend_world_m",
        },
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": source_digest,
            "frame": {
                "frame_id": "backend_world_m",
                "revision": "connector-r1",
                "coordinate_frame": "backend_world_m",
            },
        },
        "target_manifest": {
            "path": str(target_manifest),
            "sha256": target_digest,
            "frame": {
                "frame_id": "backend_world_m",
                "revision": _TARGET_REVISION,
                "coordinate_frame": "backend_world_m",
            },
        },
        "composition": {
            "method": "target_from_fixed_baseline @ graph_edge @ moving_baseline_from_source",
            "source_frame": {
                "frame_id": "backend_world_m",
                "revision": "connector-r1",
                "coordinate_frame": "backend_world_m",
            },
            "target_frame": {
                "frame_id": "backend_world_m",
                "revision": _TARGET_REVISION,
                "coordinate_frame": "backend_world_m",
            },
            "source_manifest_sha256": source_digest,
            "target_manifest_sha256": target_digest,
        },
        "source_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
        "target_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    source_manifest.write_text(
        source_manifest.read_text(encoding="utf-8") + "\nchanged\n",
        encoding="utf-8",
    )
    with pytest.raises(ConnectorMultiAnchorError, match="digest"):
        producer._load_frame_binding_input(binding_path)


def test_same_manifest_owned_baselines_derive_identity_without_conversion(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source.json"
    target_manifest = tmp_path / "target.json"
    baseline = {
        "frame_id": "backend_world_m",
        "revision": "already-baseline-r1",
        "coordinate_frame": "backend_world_m",
    }
    _write_world_manifest(source_manifest, frame_revision="already-baseline-r1")
    _write_world_manifest(target_manifest, frame_revision="already-baseline-r1")
    for path, role in (
        (source_manifest, "moving_baseline"),
        (target_manifest, "fixed_baseline"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["baseline_frame_identity"] = {"role": role, **baseline}
        path.write_text(json.dumps(payload), encoding="utf-8")
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    target_digest = hashlib.sha256(target_manifest.read_bytes()).hexdigest()
    binding_path = tmp_path / "identity-binding.json"
    frame = dict(baseline)
    binding_path.write_text(
        json.dumps(
            {
                "schema": "noesis.pcf.connector_frame_binding_input.v3",
                "source_frame": frame,
                "target_frame": frame,
                "source_manifest": {
                    "path": str(source_manifest),
                    "sha256": source_digest,
                    "frame": frame,
                },
                "target_manifest": {
                    "path": str(target_manifest),
                    "sha256": target_digest,
                    "frame": frame,
                },
                "composition": {
                    "method": "target_from_fixed_baseline @ graph_edge @ moving_baseline_from_source",
                    "source_frame": frame,
                    "target_frame": frame,
                    "source_manifest_sha256": source_digest,
                    "target_manifest_sha256": target_digest,
                },
                "source_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
                "target_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
            }
        ),
        encoding="utf-8",
    )
    normalized = producer._load_frame_binding_input(binding_path)
    assert normalized is not None
    assert normalized["canonical_eligible"] is True
    assert (
        normalized["composition"]["moving_baseline_from_source"]["status"]
        == "derived_identity_from_manifest_baseline"
    )
    assert (
        normalized["composition"]["target_from_fixed_baseline"]["status"]
        == "derived_identity_from_manifest_baseline"
    )
