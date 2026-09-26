from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256
from tools.mapanything_phone_scan.path_reference import PcfPathReferences


SCAN_ID = "20260802-162254-8bcc7dd7"
PRIOR_ID = "sceneprior_living-room_20260802T202254Z_test"
CAMERA_ID = "living-room"


def _sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _artifact(role: str, value: str) -> dict[str, Any]:
    return {
        "producer": "fixture",
        "role": role,
        "sha256": _sha(value),
        "version": "fixture-v1",
    }


def _frame_binding(
    *,
    prior_id: str = PRIOR_ID,
    camera_sha: str | None = None,
    world_sha: str | None = None,
    target_meta_sha: str | None = None,
) -> dict[str, Any]:
    source_frame = {"frame_id": "backend_world_m", "revision": "world-r1"}
    target_frame = {"frame_id": "backend_world_m", "revision": prior_id}
    matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    transform_sha = revisioned_transform_sha256(
        RevisionedFrame(**source_frame), RevisionedFrame(**target_frame), matrix
    )
    floor = {"frame": source_frame, "normal": [0.0, 1.0, 0.0], "offset_m": 0.0}
    target_floor = {"frame": target_frame, "normal": [0.0, 1.0, 0.0], "offset_m": 0.0}
    return {
        "contract": "noesis.scene_prior.frame_binding",
        "contract_version": 1,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "source_camera_calibration_sha256": camera_sha or _sha("camera"),
        "source_world_alignment_sha256": world_sha or _sha("world-alignment"),
        "target_revision_id": "target-r1",
        "target_revision_metadata_sha256": target_meta_sha or _sha("target-meta"),
        "target_from_source_col_major": matrix,
        "target_from_source_sha256": transform_sha,
        "source_floor_plane": floor,
        "target_floor_plane": target_floor,
    }


def _manifest(
    *,
    prior_id: str = PRIOR_ID,
    camera_id: str = CAMERA_ID,
    scan_id: str = SCAN_ID,
    frame_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    camera_sha = _sha("camera")
    world_sha = _sha("world-alignment")
    target_meta_sha = _sha("target-meta")
    return {
        "contract": "noesis.scene_prior.revision",
        "contract_version": 1,
        "prior_id": prior_id,
        "site_id": "fixture-site",
        "space_id": "living-room",
        "created_at_us": 1,
        "created_by": "fixture",
        "intended_use": "shadow",
        "source": {
            "source_type": "conditioned_multimodel_room_walk",
            "bundle_id": "living-room-conditioned-bundle-v1",
            "bundle_schema": "noesis.reference.room_scan_bundle.v1",
            "bundle_manifest_sha256": _sha("bundle-manifest"),
            "reference_sha256": _sha("reference"),
            "capture_id": scan_id,
            "captured_at_us": 1,
            "model": "facebook/map-anything + depth-anything/DA3Metric-Large + prior_conditioned_consensus_da3_carrier",
        },
        "alignment": _artifact("mapanything_alignment_report", "alignment"),
        "authored_scene": _artifact("authored_scene", "scene"),
        "world_to_scene": {
            **_artifact("world_to_scene", "world-alignment"),
            "version": "backend_world_to_authored_scene_similarity_v1",
        },
        "semantic_binding": {
            "room_labels": ["Living Room"],
            "authored_groups": ["room_fixture"],
            "room_group_map": _artifact("authored_scene_room_group_map", "groups"),
        },
        "grid": {
            "coordinate_frame": "backend_world_m",
            "units": "meters",
            "orientation": "row_increases_positive_z_column_increases_positive_x",
            "bounds": {"min_x": 0.0, "max_x": 1.0, "min_z": 0.0, "max_z": 1.0},
            "resolution_m": 1.0,
            "rows": 1,
            "columns": 1,
        },
        "preview": {
            "coordinate_frame": "camera_local_ground_m",
            "units": "meters",
            "orientation": "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x",
            "reference_camera_id": camera_id,
            "camera_calibration": {
                **_artifact("reference_camera_calibration", "camera"),
                "version": "camera_from_backend_col_major_v1",
            },
            "target_revision_metadata": {
                **_artifact("reference_target_revision_metadata", "target-meta"),
                "version": "noesis.room_reconstruction.stream_points.v4",
            },
            "camera_position_world_m": [0.0, 1.0, 0.0],
            "camera_right_world_xz": [1.0, 0.0],
            "camera_forward_world_xz": [0.0, 1.0],
            "bounds": {"min_x": 0.0, "max_x": 1.0, "min_z": 0.0, "max_z": 1.0},
            "resolution_m": 1.0,
            "rows": 1,
            "columns": 1,
        },
        "derivation": {
            "algorithm": "noesis_scene_prior_2_5d_full_evidence_v2",
            "floor_y_m": 0.0,
            "floor_support_band_m": 0.1,
            "obstacle_min_height_m": 0.1,
            "obstacle_max_height_m": 2.0,
            "obstacle_min_support": 1,
            "max_source_height_m": 3.0,
        },
        "quality": {
            "passed": True,
            "source_point_count": 1,
            "selected_point_count": 1,
            "authored_cell_count": 1,
            "observed_cell_count": 1,
            "floor_supported_cell_count": 1,
            "obstacle_cell_count": 0,
            "authored_observed_fraction": 1.0,
            "authored_floor_supported_fraction": 1.0,
            "alignment_status": "passed",
        },
        "artifacts": [
            {"role": "grid_npz", "relative_path": "grid.npz", "sha256": _sha("grid"), "size_bytes": 4},
            {"role": "metrics", "relative_path": "metrics.json", "sha256": _sha("metrics"), "size_bytes": 7},
            {"role": "preview", "relative_path": "preview.png", "sha256": _sha("preview"), "size_bytes": 7},
            {"role": "points_glb", "relative_path": "points.glb", "sha256": _sha(b"glTF"), "size_bytes": 4},
        ],
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "scene_priors"
    revision_root = root / "revisions" / PRIOR_ID
    revision_root.mkdir(parents=True)
    frame_binding = _frame_binding()
    manifest = _manifest(frame_binding=frame_binding)
    manifest_path = revision_root / "manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(manifest_bytes)
    points_path = revision_root / "points.glb"
    points_path.write_bytes(b"glTF")
    catalog = {
        "contract": "noesis.scene_prior.catalog",
        "contract_version": 1,
        "site_id": "fixture-site",
        "revisions": [
            {
                "prior_id": PRIOR_ID,
                "space_id": "living-room",
                "manifest_path": f"revisions/{PRIOR_ID}/manifest.json",
                "manifest_sha256": _sha(manifest_bytes),
                "manifest_size_bytes": len(manifest_bytes),
            }
        ],
        "camera_bindings": [
            {
                "camera_id": CAMERA_ID,
                "space_id": "living-room",
                "prior_id": PRIOR_ID,
                "mode": "shadow",
                "include_floorplan_layers": True,
                "frame_binding": frame_binding,
            }
        ],
    }
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return {
        "root": root,
        "catalog_path": catalog_path,
        "manifest_path": manifest_path,
        "points_path": points_path,
        "catalog": catalog,
        "manifest": manifest,
        "frame_binding": frame_binding,
    }


def _rewrite_manifest(fixture: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> None:
    manifest = deepcopy(fixture["manifest"])
    mutate(manifest)
    path = fixture["manifest_path"]
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(body)
    entry = fixture["catalog"]["revisions"][0]
    entry["manifest_sha256"] = _sha(body)
    entry["manifest_size_bytes"] = len(body)
    fixture["catalog_path"].write_text(
        json.dumps(fixture["catalog"], sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def test_no_catalog_has_no_matching_reference() -> None:
    assert PcfPathReferences(None).for_scan(SCAN_ID) is None


def test_available_descriptor_and_resolved_shape_are_exact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    resolver = PcfPathReferences(fixture["catalog_path"])

    descriptor = resolver.for_scan(SCAN_ID)
    assert descriptor is not None
    assert descriptor["status"] == "available"
    assert descriptor["label"] == "Noesis PCF · Living Room"
    assert descriptor["source_scan_id"] == SCAN_ID
    assert descriptor["selection"]["kind"] == "scene_prior_pcf"
    assert descriptor["selection"]["prior_id"] == PRIOR_ID
    assert descriptor["selection"]["camera_id"] == CAMERA_ID
    assert descriptor["selection"]["manifest_sha256"] == fixture["catalog"]["revisions"][0]["manifest_sha256"]
    assert descriptor["selection"]["frame_binding_sha256"] == _sha(
        json.dumps(fixture["frame_binding"], sort_keys=True, separators=(",", ":"))
    )
    assert descriptor["selection"]["frame_binding_sha256"] != fixture["frame_binding"]["target_from_source_sha256"]

    resolved = resolver.resolve(SCAN_ID, descriptor["selection"])
    assert resolved["selection"] == descriptor["selection"]
    assert resolved["label"] == descriptor["label"]
    assert resolved["source_scan_id"] == SCAN_ID
    assert resolved["manifest"] == fixture["manifest"]
    assert resolved["frame_binding"] == fixture["frame_binding"]
    for key in ("catalog", "manifest", "points"):
        assert set(resolved["evidence"][key]) == {"path", "sha256", "size_bytes"}
    assert resolved["assets"]["manifest"] == resolved["evidence"]["manifest"]
    assert resolved["assets"]["points"] == resolved["evidence"]["points"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda manifest: manifest["quality"].update(passed=False), "quality gate"),
        (lambda manifest: manifest["quality"].update(alignment_status="failed"), "alignment status"),
        (lambda manifest: manifest["source"].update(model="mapanything-only"), "source model"),
        (lambda manifest: manifest.update(site_id="another-site"), "site_id"),
        (
            lambda manifest: manifest["preview"]["camera_calibration"].update(sha256=_sha("wrong-camera")),
            "calibration digest",
        ),
    ],
)
def test_matching_invalid_candidate_is_unavailable(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], reason: str
) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_manifest(fixture, mutate)
    result = PcfPathReferences(fixture["catalog_path"]).for_scan(SCAN_ID)
    assert result is not None
    assert result["status"] == "unavailable"
    assert reason in result["reason"]


def test_points_tamper_is_unavailable_and_resolve_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["points_path"].write_bytes(b"tampered")
    resolver = PcfPathReferences(fixture["catalog_path"])
    result = resolver.for_scan(SCAN_ID)
    assert result == {
        "status": "unavailable",
        "reason": "camera living-room: points_glb fingerprint or size does not match the manifest",
    }
    with pytest.raises(ValueError, match="points_glb fingerprint"):
        resolver.resolve(SCAN_ID, {
            "kind": "scene_prior_pcf",
            "prior_id": PRIOR_ID,
            "manifest_sha256": fixture["catalog"]["revisions"][0]["manifest_sha256"],
            "camera_id": CAMERA_ID,
            "frame_binding_sha256": _sha(
                json.dumps(fixture["frame_binding"], sort_keys=True, separators=(",", ":"))
            ),
        })


def test_exact_selection_and_source_identity_are_required(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    resolver = PcfPathReferences(fixture["catalog_path"])
    descriptor = resolver.for_scan(SCAN_ID)
    assert descriptor is not None
    with pytest.raises(ValueError, match="exact active PCF reference"):
        wrong = dict(descriptor["selection"])
        wrong["manifest_sha256"] = "0" * 64
        resolver.resolve(SCAN_ID, wrong)
    with pytest.raises(ValueError, match="source capture_id"):
        resolver.resolve("20260802-162255-8bcc7dd7", descriptor["selection"])
    assert resolver.for_scan("20260802-162255-8bcc7dd7") is None


def test_symlink_points_asset_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["points_path"].unlink()
    fixture["points_path"].symlink_to(fixture["root"] / "outside.glb")
    (fixture["root"] / "outside.glb").write_bytes(b"glTF")
    result = PcfPathReferences(fixture["catalog_path"]).for_scan(SCAN_ID)
    assert result is not None
    assert result["status"] == "unavailable"
    assert "points_glb" in result["reason"]
