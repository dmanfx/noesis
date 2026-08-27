from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from noesis.scene_prior_builder import (
    _CameraMapLockInput,
    _authored_grid,
    _camera_local_raster_to_preview_image,
    _camera_local_preview_arrays,
    _camera_preview_frame,
    _derive_arrays,
    _deterministic_npz,
)
from noesis.pipelines.hooks import _AnalyticsTelemetryProcessor
from noesis.server import scene_prior_api
from noesis.virtual_twin.artifacts import write_points_glb
from noesis_core.coordinate_frames import CAMERA_LOCAL_RASTER_ORIENTATION
from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.scene_prior import (
    ScenePriorArtifact,
    ScenePriorBounds,
    ScenePriorCameraBinding,
    ScenePriorCatalog,
    ScenePriorCatalogEntry,
    ScenePriorDerivation,
    ScenePriorGrid,
    ScenePriorPreview,
    ScenePriorQuality,
    ScenePriorRevision,
    ScenePriorSemanticBinding,
    ScenePriorSource,
)
from noesis_core.scene_prior import (
    LoadedScenePrior,
    ScenePriorError,
    ScenePriorSet,
    _calibrated_raster_geometry,
    _decode_float32_layer,
    _derive_preview_diagnostics,
    _encoded_mask_layer,
    _encoded_uint8_layer,
)


def _json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _artifact(role: str, path: str, payload: bytes) -> ScenePriorArtifact:
    return ScenePriorArtifact(
        role=role,
        relative_path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _write_prior(root: Path) -> tuple[Path, dict[str, bytes]]:
    shape = (2, 2)
    arrays = {
        "authored_walkable": np.ones(shape, dtype=np.uint8),
        "observed": np.ones(shape, dtype=np.uint8),
        "evidence_confidence": np.full(shape, 0.8, dtype=np.float32),
        "floor_supported": np.ones(shape, dtype=np.uint8),
        "obstacle_mask": np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
        "walkable_candidate": np.asarray([[1, 0], [1, 1]], dtype=np.uint8),
        "floor_height_m": np.zeros(shape, dtype=np.float32),
        "height_agl_p95_m": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "boundary_signed_distance_m": np.ones(shape, dtype=np.float32),
        "obstacle_signed_clearance_m": np.asarray(
            [[1.0, -1.0], [1.0, 1.0]], dtype=np.float32
        ),
        "point_count": np.full(shape, 8, dtype=np.uint32),
        "floor_support_count": np.full(shape, 3, dtype=np.uint32),
        "obstacle_support_count": np.asarray([[0, 4], [0, 0]], dtype=np.uint32),
    }
    point_positions = np.asarray(
        [
            (column + 0.5, height, row + 0.5)
            for row in range(2)
            for column in range(2)
            for height in (0.0, 0.05, 0.2, 0.25, 0.5, 0.55, 1.0, 1.05)
        ],
        dtype=np.float32,
    )
    point_colors = np.tile(
        np.asarray([[64, 128, 192]], dtype=np.uint8),
        (point_positions.shape[0], 1),
    )
    fixture_glb = root / "fixture-points.glb"
    write_points_glb(fixture_glb, point_positions, point_colors)
    points_glb = fixture_glb.read_bytes()
    fixture_glb.unlink()
    artifact_payloads = {
        "grid.npz": _deterministic_npz(arrays),
        "metrics.json": b"{}\n",
        "preview.png": b"png",
        "points.glb": points_glb,
    }
    revision = ScenePriorRevision(
        contract="noesis.scene_prior.revision",
        contract_version=1,
        prior_id="sceneprior_room-a_20260801T000000Z_123456789abc",
        site_id="site-a",
        space_id="room-a",
        created_at_us=1,
        created_by="test",
        intended_use="shadow",
        source=ScenePriorSource(
            source_type="mapanything_multiview_room_walk",
            bundle_id="bundle-a",
            bundle_schema="noesis.reference.room_scan_bundle.v1",
            bundle_manifest_sha256="1" * 64,
            reference_sha256="2" * 64,
            capture_id="capture-a",
            captured_at_us=1,
            model="map-anything",
        ),
        alignment=ArtifactFingerprint(role="alignment", sha256="3" * 64),
        authored_scene=ArtifactFingerprint(role="authored_scene", sha256="4" * 64),
        world_to_scene=ArtifactFingerprint(role="world_to_scene", sha256="5" * 64),
        semantic_binding=ScenePriorSemanticBinding(
            room_labels=("Room A",),
            authored_groups=("room_1",),
            room_group_map=ArtifactFingerprint(role="room_map", sha256="6" * 64),
        ),
        grid=ScenePriorGrid(
            coordinate_frame="backend_world_m",
            units="meters",
            orientation="row_increases_positive_z_column_increases_positive_x",
            bounds=ScenePriorBounds(min_x=0, max_x=2, min_z=0, max_z=2),
            resolution_m=1,
            rows=2,
            columns=2,
        ),
        preview=ScenePriorPreview(
            coordinate_frame="camera_local_ground_m",
            units="meters",
            orientation=CAMERA_LOCAL_RASTER_ORIENTATION,
            reference_camera_id="camera-a",
            camera_calibration=ArtifactFingerprint(
                role="camera_calibration",
                sha256="7" * 64,
            ),
            target_revision_metadata=ArtifactFingerprint(
                role="target_revision_metadata",
                sha256="8" * 64,
            ),
            camera_position_world_m=(0, 1, 0),
            camera_right_world_xz=(1, 0),
            camera_forward_world_xz=(0, 1),
            bounds=ScenePriorBounds(min_x=0, max_x=2, min_z=0, max_z=2),
            resolution_m=1,
            rows=2,
            columns=2,
        ),
        derivation=ScenePriorDerivation(
            algorithm="noesis_scene_prior_2_5d_v1",
            floor_y_m=0,
            floor_support_band_m=0.12,
            obstacle_min_height_m=0.18,
            obstacle_max_height_m=2.2,
            obstacle_min_support=3,
            max_source_height_m=3.2,
        ),
        quality=ScenePriorQuality(
            passed=True,
            source_point_count=32,
            selected_point_count=32,
            authored_cell_count=4,
            observed_cell_count=4,
            floor_supported_cell_count=4,
            obstacle_cell_count=1,
            authored_observed_fraction=1,
            authored_floor_supported_fraction=1,
            alignment_status="passed",
        ),
        artifacts=(
            _artifact("grid_npz", "grid.npz", artifact_payloads["grid.npz"]),
            _artifact("metrics", "metrics.json", artifact_payloads["metrics.json"]),
            _artifact("preview", "preview.png", artifact_payloads["preview.png"]),
            _artifact("points_glb", "points.glb", artifact_payloads["points.glb"]),
        ),
    )
    revision_dir = root / "revisions" / revision.prior_id
    revision_dir.mkdir(parents=True)
    for name, payload in artifact_payloads.items():
        (revision_dir / name).write_bytes(payload)
    manifest_bytes = _json_bytes(revision)
    (revision_dir / "manifest.json").write_bytes(manifest_bytes)
    entry = ScenePriorCatalogEntry(
        prior_id=revision.prior_id,
        space_id=revision.space_id,
        manifest_path=f"revisions/{revision.prior_id}/manifest.json",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_size_bytes=len(manifest_bytes),
    )
    catalog = ScenePriorCatalog(
        contract="noesis.scene_prior.catalog",
        contract_version=1,
        site_id="site-a",
        revisions=(entry,),
        camera_bindings=(
            ScenePriorCameraBinding(
                camera_id="camera-a",
                space_id="room-a",
                prior_id=revision.prior_id,
                mode="shadow",
                include_floorplan_layers=True,
            ),
        ),
    )
    catalog_path = root / "catalog.json"
    catalog_path.write_bytes(_json_bytes(catalog))
    return catalog_path, artifact_payloads


def _layer(values: np.ndarray) -> dict[str, object]:
    grid = np.asarray(values, dtype=np.float32)
    return {
        "grid_b64": base64.b64encode(grid.tobytes()).decode(),
        "grid_shape": list(grid.shape),
        "value_min": 0,
        "value_max": float(np.nanmax(grid)),
    }


def test_scene_prior_preview_reads_legacy_numeric_grid_orientation(
    tmp_path: Path,
) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    revision = ScenePriorSet.load(catalog_path).revision_for_camera("camera-a")
    assert revision is not None and revision.manifest.preview is not None
    payload = revision.manifest.preview.model_dump(mode="json")
    payload["orientation"] = (
        "row_increases_camera_forward_column_increases_camera_right"
    )

    legacy = ScenePriorPreview.model_validate(payload)

    assert legacy.orientation == payload["orientation"]


def _decode(layer: dict[str, object]) -> np.ndarray:
    shape = tuple(layer["grid_shape"])
    return _decode_float32_layer({"layer": layer}, "layer", expected_shape=shape)


def test_compact_mask_and_uint8_layers_round_trip_without_geometry_loss() -> None:
    mask = np.asarray([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]], dtype=np.float32)
    encoded_mask = _encoded_mask_layer(mask)
    assert str(encoded_mask["grid_b64"]).startswith(f"bit:{mask.size}:")
    np.testing.assert_array_equal(_decode(encoded_mask), mask)

    source = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.float32)
    encoded_source = _encoded_uint8_layer(source, value_min=0.0, value_max=2.0)
    assert str(encoded_source["grid_b64"]).startswith("u8:")
    np.testing.assert_array_equal(_decode(encoded_source), source)


def test_scene_prior_load_evaluate_and_live_wins_composition(tmp_path: Path) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    priors = ScenePriorSet.load(catalog_path)

    diagnostic = priors.evaluate("camera-a", (1.5, 0.0, 0.5))
    assert diagnostic is not None
    assert diagnostic["status"] == "warning"
    assert diagnostic["reasons"] == ["inside_static_obstacle_candidate"]
    assert diagnostic["extent_outside_distance_m"] == 0.0

    outside = priors.evaluate("camera-a", (3.5, 0.0, 2.5))
    assert outside is not None
    assert outside["inside_extent"] is False
    assert outside["extent_outside_distance_m"] == pytest.approx(
        (1.5**2 + 0.5**2) ** 0.5,
        abs=1e-6,
    )

    live_height = np.asarray([[10.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    live_observed = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    result = priors.compose_floorplan(
        "camera-a",
        {
            "frame": "camera_local_ground_m",
            "units": "meters",
            "bounds": {"min_x": 0, "max_x": 2, "min_z": 0, "max_z": 2},
            "height_agl": _layer(live_height),
            "observed": _layer(live_observed),
        },
        extrinsics_col_major=np.eye(4).flatten(order="F").tolist(),
        floor_y_m=0,
    )
    composite = _decode(result["scene_composite_height_agl"])
    static = _decode(result["scene_static_height_agl"])
    source = _decode(result["scene_composite_source"])
    # Live floorplan row zero is the far/forward edge.  The static prior is
    # sampled into that display orientation rather than vertically mirrored.
    np.testing.assert_allclose(static, [[3, 4], [1, 2]])
    np.testing.assert_allclose(composite, [[10, 4], [1, 2]])
    np.testing.assert_allclose(source, [[2, 1], [1, 1]])
    assert result["scene_prior_meta"]["composition_policy"] == (
        "live_observed_wins_static_fills_live_unknown"
    )
    assert result["scene_prior_meta"]["static_fill_cells"] == 3


def test_scene_prior_static_floorplan_survives_live_cache_miss(tmp_path: Path) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    priors = ScenePriorSet.load(catalog_path)

    result = priors.compose_static_floorplan(
        "camera-a",
        {
            "type": "floorplan_response",
            "request_id": "cache-only",
            "camera_id": "camera-a",
            "error": "no_cached_floorplan",
        },
    )

    assert "error" not in result
    assert result["live_floorplan_error"] == "no_cached_floorplan"
    assert result["frame"] == "camera_local_ground_m"
    assert result["scene_prior_meta"]["status"] == "static_only"
    assert result["scene_prior_only"] is False
    assert "scene_prior_diagnostic_height_agl" not in result
    assert result["scene_prior_meta"]["live_observed_cells"] == 0
    assert result["scene_prior_meta"]["static_available_cells"] == 4
    np.testing.assert_allclose(
        _decode(result["scene_static_height_agl"]),
        [[3.0, 4.0], [1.0, 2.0]],
    )
    np.testing.assert_allclose(
        _decode(result["scene_composite_source"]),
        np.ones((2, 2), dtype=np.float32),
    )
    assert result["scene_prior_meta"]["raster_orientation"] == (
        "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x"
    )


def test_scene_prior_only_floorplan_is_canonical_pcf_presentation(tmp_path: Path) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    priors = ScenePriorSet.load(catalog_path)

    result = priors.compose_static_floorplan(
        "camera-a",
        {
            "type": "floorplan_response",
            "request_id": "pcf-only",
            "camera_id": "camera-a",
            "cache_only": True,
            "scene_prior_only": True,
        },
        extrinsics_col_major=np.eye(4).flatten(order="F").tolist(),
    )

    assert result["scene_prior_only"] is True
    assert result["display_source"] == "pcf"
    assert result["served_from_cache"] is True
    assert result["scene_prior_meta"]["status"] == "pcf"
    assert result["scene_prior_meta"]["camera_geometry_source"] == (
        "current_calibrated_extrinsics"
    )
    assert result["scene_prior_meta"]["composition_policy"] == "pcf_only"
    assert result["scene_prior_meta"]["revision_manifest_path"] == (
        "revisions/sceneprior_room-a_20260801T000000Z_123456789abc/manifest.json"
    )
    assert len(result["scene_prior_meta"]["revision_manifest_sha256"]) == 64
    assert result["scene_prior_diagnostic_meta"]["source"] == "map-anything"
    assert result["scene_prior_diagnostic_meta"]["derivation"] == (
        "prior_conditioned_fusion_points_and_grid"
    )
    np.testing.assert_allclose(
        _decode(result["scene_prior_diagnostic_height_agl"]),
        [[3.0, 4.0], [1.0, 2.0]],
    )
    assert result["scene_prior_diagnostic_surface_rgb"]["rgb_shape"] == [2, 2, 3]
    assert "scene_static_height_agl" not in result
    assert "scene_composite_height_agl" not in result


def test_scene_prior_only_floorplan_uses_current_camera_calibration(tmp_path: Path) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    priors = ScenePriorSet.load(catalog_path)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[0, 3] = -0.25

    result = priors.compose_static_floorplan(
        "camera-a",
        {
            "camera_id": "camera-a",
            "cache_only": True,
            "scene_prior_only": True,
        },
        extrinsics_col_major=world_to_camera.flatten(order="F").tolist(),
    )

    assert result["bounds"] == {
        "min_x": -1.0,
        "max_x": 2.0,
        "min_z": 0.0,
        "max_z": 2.0,
    }
    assert result["scene_prior_diagnostic_meta"]["camera_position_world_m"] == [
        0.25,
        0.0,
        0.0,
    ]
    assert result["scene_prior_diagnostic_height_agl"]["grid_shape"] == [2, 3]


def test_scene_prior_camera_view_exposes_verified_camera_local_cloud(
    tmp_path: Path,
) -> None:
    catalog_path, artifacts = _write_prior(tmp_path)
    priors = ScenePriorSet.load(catalog_path)

    metadata = priors.camera_view_metadata(
        "camera-a",
        extrinsics_col_major=np.eye(4).flatten(order="F").tolist(),
    )
    assert metadata["source_coordinate_frame"] == "backend_world_m"
    assert metadata["target_coordinate_frame"] == "camera_local_ground_m"
    assert metadata["orientation"] == {
        "screen_right": "camera_right_positive_x",
        "screen_up": "camera_forward_positive_z",
        "vertical": "height_above_floor_positive_y",
    }
    np.testing.assert_allclose(
        metadata["world_to_camera_local_row_major"],
        np.eye(4),
    )
    artifact, payload = priors.artifact_for_camera("camera-a", "points_glb")
    assert artifact.sha256 == hashlib.sha256(artifacts["points.glb"]).hexdigest()
    assert payload == artifacts["points.glb"]


def test_scene_prior_api_serves_view_contract_and_verified_glb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, artifacts = _write_prior(tmp_path)
    priors = ScenePriorSet.load(catalog_path)
    snapshot = type(
        "CalibrationSnapshot",
        (),
        {"extrinsics_col_major": np.eye(4).flatten(order="F").tolist()},
    )()
    calibration = type(
        "CalibrationProvider",
        (),
        {"snapshot": lambda _self, _source_id, _camera_id: snapshot},
    )()
    monkeypatch.setattr(
        scene_prior_api.app.state,
        "scene_prior_runtime_provider",
        lambda: (priors, calibration, {0: "camera-a"}),
        raising=False,
    )

    client = TestClient(scene_prior_api.app)
    metadata_response = client.get("/api/v1/scene-priors/cameras/camera-a")
    assert metadata_response.status_code == 200
    metadata = metadata_response.json()
    assert metadata["prior_id"].startswith("sceneprior_room-a_")
    assert metadata["artifact_urls"]["points_glb"].endswith(
        "/camera-a/artifacts/points_glb"
    )

    artifact_response = client.get(metadata["artifact_urls"]["points_glb"])
    assert artifact_response.status_code == 200
    assert artifact_response.content == artifacts["points.glb"]
    assert artifact_response.headers["content-type"] == "model/gltf-binary"
    assert artifact_response.headers["x-noesis-artifact-sha256"] == hashlib.sha256(
        artifacts["points.glb"]
    ).hexdigest()


def test_tracking_hook_adds_shadow_diagnostic_without_mutating_world(
    tmp_path: Path,
) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    priors = ScenePriorSet.load(catalog_path)
    processor = type("Processor", (), {"scene_priors": priors})()
    track = {
        "world": [0.5, 0.0, 0.5],
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_source": "pose_depth_fused",
    }
    world_before = list(track["world"])

    _AnalyticsTelemetryProcessor._apply_scene_prior_shadow(
        processor,
        "camera-a",
        track,
    )

    assert track["world"] == world_before
    assert track["world_source"] == "pose_depth_fused"
    assert track["scene_prior"]["mode"] == "shadow"
    assert track["scene_prior"]["status"] == "pass"


def test_scene_prior_rejects_tampered_immutable_artifact(tmp_path: Path) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    grid_path = next((tmp_path / "revisions").glob("*/grid.npz"))
    grid_path.write_bytes(grid_path.read_bytes() + b"tamper")
    with pytest.raises(ScenePriorError, match="exceeds|size mismatch|fingerprint"):
        ScenePriorSet.load(catalog_path)


def test_scene_prior_catalog_rejects_cross_space_camera_binding(tmp_path: Path) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    payload = json.loads(catalog_path.read_text())
    payload["camera_bindings"][0]["space_id"] = "different-room"
    with pytest.raises(ValidationError, match="space does not match"):
        ScenePriorCatalog.model_validate(payload)


def test_scene_prior_npz_is_deterministic() -> None:
    arrays = {
        "b": np.asarray([2], dtype=np.uint8),
        "a": np.asarray([1], dtype=np.uint8),
    }
    assert _deterministic_npz(arrays) == _deterministic_npz(
        dict(reversed(list(arrays.items())))
    )


def test_scene_prior_preview_follows_reference_camera_right_and_forward() -> None:
    grid = ScenePriorGrid(
        coordinate_frame="backend_world_m",
        units="meters",
        orientation="row_increases_positive_z_column_increases_positive_x",
        bounds=ScenePriorBounds(min_x=0, max_x=3, min_z=0, max_z=2),
        resolution_m=1,
        rows=2,
        columns=3,
    )
    obstacle = np.asarray([[1, 0, 0], [0, 0, 1]], dtype=np.uint8)
    arrays = {
        "authored_walkable": np.ones((2, 3), dtype=np.uint8),
        "observed": np.ones((2, 3), dtype=np.uint8),
        "floor_supported": np.ones((2, 3), dtype=np.uint8),
        "obstacle_mask": obstacle,
    }

    transformed, bounds = _camera_local_preview_arrays(
        arrays,
        grid=grid,
        camera_position_world_m=(0, 1, 0),
        camera_right_world_xz=(-1, 0),
        camera_forward_world_xz=(0, 1),
    )

    np.testing.assert_array_equal(
        transformed["obstacle_mask"],
        np.asarray([[0, 0, 1], [1, 0, 0]], dtype=bool),
    )
    assert bounds == ScenePriorBounds(min_x=-3, max_x=0, min_z=0, max_z=2)


def test_scene_prior_preview_image_puts_forward_at_top_without_mirroring_x() -> None:
    # Numeric preview rows increase with camera-forward +Z. PNG rows increase
    # downward, so this is the one intentional row-address conversion.
    numeric = np.asarray(
        [
            [[255, 0, 0], [0, 0, 255]],   # near: left red, right blue
            [[0, 255, 0], [255, 255, 0]], # far: left green, right yellow
        ],
        dtype=np.uint8,
    )

    image = _camera_local_raster_to_preview_image(numeric)

    np.testing.assert_array_equal(image[0, 0], [0, 255, 0])
    np.testing.assert_array_equal(image[0, 1], [255, 255, 0])
    np.testing.assert_array_equal(image[1, 0], [255, 0, 0])
    np.testing.assert_array_equal(image[1, 1], [0, 0, 255])


def test_full_evidence_derivation_retains_points_outside_authored_room() -> None:
    triangle = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.repeat(
        np.asarray(
            [[0.25, 0.0, 0.25], [2.25, 0.0, 0.25]],
            dtype=np.float64,
        ),
        60,
        axis=0,
    )
    grid, authored = _authored_grid(
        [triangle],
        1.0,
        evidence_points=points,
    )
    derivation = ScenePriorDerivation(
        algorithm="noesis_scene_prior_2_5d_full_evidence_v2",
        floor_y_m=0,
        floor_support_band_m=0.12,
        obstacle_min_height_m=0.18,
        obstacle_max_height_m=2.2,
        obstacle_min_support=3,
        max_source_height_m=3.2,
    )

    arrays, selected = _derive_arrays(
        points,
        grid=grid,
        authored=authored,
        derivation=derivation,
    )

    assert grid.bounds.max_x == 3.0
    assert int(np.count_nonzero(selected)) == 120
    assert arrays["authored_walkable"][0, 2] == 0
    assert arrays["observed"][0, 2] == 1
    assert arrays["floor_supported"][0, 2] == 1
    assert arrays["walkable_candidate"][0, 2] == 1


def test_full_evidence_diagnostics_frame_reconstruction_not_authored_box(
    tmp_path: Path,
) -> None:
    catalog_path, _ = _write_prior(tmp_path)
    base = ScenePriorSet.load(catalog_path).revision_for_camera("camera-a")
    assert base is not None
    shape = (2, 3)
    arrays = {
        "authored_walkable": np.asarray([[1, 1, 0], [1, 1, 0]], dtype=np.uint8),
        "observed": np.ones(shape, dtype=np.uint8),
        "evidence_confidence": np.full(shape, 0.8, dtype=np.float32),
        "floor_supported": np.ones(shape, dtype=np.uint8),
        "obstacle_mask": np.zeros(shape, dtype=np.uint8),
        "walkable_candidate": np.ones(shape, dtype=np.uint8),
        "floor_height_m": np.zeros(shape, dtype=np.float32),
        "height_agl_p95_m": np.full(shape, 0.5, dtype=np.float32),
        "boundary_signed_distance_m": np.asarray(
            [[1.0, 1.0, -1.0], [1.0, 1.0, -1.0]], dtype=np.float32
        ),
        "obstacle_signed_clearance_m": np.ones(shape, dtype=np.float32),
        "point_count": np.full(shape, 8, dtype=np.uint32),
        "floor_support_count": np.full(shape, 3, dtype=np.uint32),
        "obstacle_support_count": np.zeros(shape, dtype=np.uint32),
    }
    grid = ScenePriorGrid(
        coordinate_frame="backend_world_m",
        units="meters",
        orientation="row_increases_positive_z_column_increases_positive_x",
        bounds=ScenePriorBounds(min_x=0, max_x=3, min_z=0, max_z=2),
        resolution_m=1,
        rows=2,
        columns=3,
    )
    preview = base.manifest.preview.model_copy(
        update={
            "bounds": ScenePriorBounds(min_x=0, max_x=3, min_z=0, max_z=2),
            "columns": 3,
        }
    )
    derivation = base.manifest.derivation.model_copy(
        update={"algorithm": "noesis_scene_prior_2_5d_full_evidence_v2"}
    )
    points = np.asarray(
        [
            (column + 0.5, height, row + 0.5)
            for row in range(2)
            for column in range(3)
            for height in (0.0, 0.5, 1.0)
        ],
        dtype=np.float32,
    )
    revision = LoadedScenePrior(
        manifest=base.manifest.model_copy(
            update={"grid": grid, "preview": preview, "derivation": derivation}
        ),
        root=base.root,
        arrays=arrays,
        points_world_m=points,
        colors_rgb_u8=np.full((points.shape[0], 3), 128, dtype=np.uint8),
    )

    geometry = _calibrated_raster_geometry(
        revision,
        np.eye(4).flatten(order="F").tolist(),
    )
    diagnostics = _derive_preview_diagnostics(revision, geometry)

    assert geometry.bounds_payload() == {
        "min_x": 0.0,
        "max_x": 3.0,
        "min_z": 0.0,
        "max_z": 2.0,
    }
    assert int(np.count_nonzero(diagnostics["room_footprint"])) == 4
    assert int(np.count_nonzero(diagnostics["reconstruction_extent"])) == 6
    assert int(np.count_nonzero(diagnostics["observed"])) == 6


def test_scene_prior_preview_frame_composes_floor_corrected_camera_pose() -> None:
    world_correction = np.asarray(
        [
            [0, 0, 1, 3],
            [0, 1, 0, 2],
            [-1, 0, 0, 4],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    frame = _camera_preview_frame(
        {
            "reference_camera_id": "camera-a",
            "camera_calibration_row": {
                "E": np.eye(4, dtype=np.float64).flatten(order="F").tolist(),
            },
            "camera_calibration_bytes": b"camera-calibration",
            "target_revision_metadata": {
                "schema": "target.v1",
                "revision_id": "vt_camera-a_exact",
                "floor_alignment": {
                    "world_correction_col_major": world_correction.flatten(
                        order="F"
                    ).tolist(),
                    "source_floor_normal": [0.0, 1.0, 0.0],
                    "source_floor_offset": 2.0,
                    "target_floor_y": 0.0,
                },
            },
            "target_revision_metadata_bytes": b"target-metadata",
        }
    )

    np.testing.assert_allclose(frame.camera_position_world_m, (3, 2, 4))
    np.testing.assert_allclose(frame.camera_right_world_xz, (0, -1), atol=1e-12)
    np.testing.assert_allclose(frame.camera_forward_world_xz, (1, 0), atol=1e-12)
    assert frame.target_revision_id == "vt_camera-a_exact"
    assert frame.source_floor_offset_m == pytest.approx(2.0)


def test_scene_prior_preview_frame_applies_yaw_map_lock_about_camera_center() -> None:
    evidence = ArtifactFingerprint(
        role="camera_to_pcf_map_lock",
        sha256="a" * 64,
        version="noesis.scene_prior.camera_map_lock.input.v1",
        producer="test",
    )
    frame = _camera_preview_frame(
        {
            "reference_camera_id": "family-room",
            "camera_calibration_row": {
                "E": np.eye(4, dtype=np.float64).flatten(order="F").tolist(),
            },
            "camera_calibration_bytes": b"camera-calibration",
            "target_revision_metadata": {
                "schema": "target.v1",
                "revision_id": "vt_family-room_exact",
                "floor_alignment": {
                    "world_correction_col_major": np.eye(4, dtype=np.float64)
                    .flatten(order="F")
                    .tolist(),
                    "target_floor_y": 0.0,
                },
            },
            "target_revision_metadata_bytes": b"target-metadata",
        },
        camera_map_lock=_CameraMapLockInput(
            evidence=evidence,
            yaw_correction_deg=18.25,
        ),
    )

    radians = np.deg2rad(18.25)
    np.testing.assert_allclose(frame.camera_position_world_m, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(
        frame.camera_forward_world_xz,
        (np.sin(radians), np.cos(radians)),
        atol=1e-12,
    )
    assert frame.camera_map_lock is not None
    assert frame.camera_map_lock.evidence == evidence
    assert frame.camera_map_lock.yaw_correction_deg == pytest.approx(18.25)
    assert frame.source_floor_normal == pytest.approx((0.0, 1.0, 0.0))
    assert frame.source_floor_offset_m == pytest.approx(0.0)


def test_scene_prior_preview_frame_derives_reference_locked_source_floor() -> None:
    frame = _camera_preview_frame(
        {
            "reference_camera_id": "camera-a",
            "camera_calibration_row": {
                "E": np.eye(4, dtype=np.float64).flatten(order="F").tolist(),
            },
            "camera_calibration_bytes": b"camera-calibration",
            "target_revision_metadata": {
                "schema": "target.v1",
                "revision_id": "vt_camera-a_reference",
                "floor_alignment": {
                    "status": "reference_locked",
                    "world_correction_col_major": np.eye(4, dtype=np.float64)
                    .flatten(order="F")
                    .tolist(),
                    "target_floor_y": 0.0,
                },
            },
            "target_revision_metadata_bytes": b"target-metadata",
        }
    )

    assert frame.source_floor_normal == pytest.approx((0.0, 1.0, 0.0))
    assert frame.source_floor_offset_m == pytest.approx(0.0)
