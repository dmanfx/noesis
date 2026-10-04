"""Focused measured scene-presentation registration checks.

The current-scene integration fixture is intentionally skipped on checkouts
without the reviewed local OBJ and calibration capture. It is not a synthetic
substitute for the authored floor geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noesis.calibration.manager import (
    CalibrationManager, CalibrationValidationError, load_camera_labels,
)
from noesis.calibration.presentation_registration import (
    PresentationRegistrationError, _canonical_sha256, build_registration,
    derive_entry, load_registration,
)
from noesis_core.contracts.scene_prior import ScenePriorFrameBinding
from noesis_core.contracts.support_surface import SupportSurfaceSet


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/virtual_twin/derived-authored-floor-20261001/TanglewoodManor_Layout_noGarage.obj"
ARTIFACT = ROOT / "data/virtual_twin/derived-authored-floor-20261001/presentation-registration.json"
CAPTURE = ROOT.parent / "Menon/artifacts/menon-integration-20261001/before"
PARENT = CAPTURE / "authored-scene.obj"
BUNDLE = CAPTURE / "calibration-bundle.json"
SUPPORT = ROOT / "DS9/config/support_surfaces.json"
REQUIRED = (MODEL, ARTIFACT, PARENT, BUNDLE, SUPPORT)
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED),
    reason="reviewed current authored scene and calibration capture are unavailable",
)


def _write_artifact(path: Path, payload: dict) -> None:
    payload["registration_sha256"] = _canonical_sha256({
        key: value for key, value in payload.items() if key != "registration_sha256"
    })
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manager() -> CalibrationManager:
    catalog = json.loads((ROOT / "data/scene_priors/catalog.json").read_text())
    bindings = {
        row["camera_id"]: ScenePriorFrameBinding.model_validate(row["frame_binding"])
        for row in catalog["camera_bindings"]
    }
    support = SupportSurfaceSet.model_validate(json.loads(SUPPORT.read_text()))
    manager = CalibrationManager(
        cameras_yaml_path=ROOT / "config/cameras.yaml",
        camera_calibration_json_path=ROOT / "config/camera_calibration.json",
        ply_alignment_json_path=ROOT / "config/ply_alignment.json",
        scene_prior_frame_bindings=bindings,
        support_surfaces={support.camera_id: support},
        scene_presentation_registration_path=ARTIFACT,
        presentation_authored_obj_path=MODEL,
    )
    manager.set_camera_labels(load_camera_labels(ROOT / "config/cameras.yaml"))
    return manager


def test_current_authored_floor_registration_rebuilds_and_binds() -> None:
    built = build_registration(
        bundle_path=BUNDLE,
        authored_obj_path=MODEL,
        parent_authored_obj_path=PARENT,
        room_group_map_path=ROOT / "config/authored_scene_room_groups.json",
        support_measurements_path=ROOT / "config/calibration/living-room-support-measurements.json",
        support_surfaces_path=SUPPORT,
        upper_room="Foyer",
        lower_rooms=("Living Room", "Kitchen", "Family Room"),
    )
    assert built == load_registration(ARTIFACT, authored_obj_path=MODEL)
    assert built["foyer_top_scene_y"] == pytest.approx(
        built["foyer_rise_m"] * next(iter(built["presentations"].values()))["constraints"]["scene_per_m"]
    )
    manager = _manager()
    presentations = manager.calibration_bundle()["world_frame_presentations"]
    assert len(presentations) == 3
    assert {row["registration_sha256"] for row in presentations.values()} == {
        built["registration_sha256"]
    }
    assert {row["authored_scene_sha256"] for row in presentations.values()} == {
        built["authored_scene_sha256"]
    }


def test_artifact_loader_rejects_wrong_obj_unknown_fields_and_target_key(tmp_path: Path) -> None:
    with pytest.raises(PresentationRegistrationError, match="authored OBJ"):
        load_registration(ARTIFACT, authored_obj_path=PARENT)
    payload = json.loads(ARTIFACT.read_text())
    payload["unexpected"] = True
    path = tmp_path / "unknown.json"
    _write_artifact(path, payload)
    with pytest.raises(PresentationRegistrationError, match="unknown fields"):
        load_registration(path, authored_obj_path=MODEL)
    payload.pop("unexpected")
    revision, entry = next(iter(payload["presentations"].items()))
    payload["presentations"]["wrong-revision"] = payload["presentations"].pop(revision)
    _write_artifact(path, payload)
    with pytest.raises(PresentationRegistrationError, match="revision key"):
        load_registration(path, authored_obj_path=MODEL)


def test_active_source_or_measured_plane_change_rejects_registration() -> None:
    manager = _manager()
    manager._align["scene_similarity"]["scene_per_m"] *= 1.01
    with pytest.raises(CalibrationValidationError, match="cannot be reproduced"):
        manager.world_frame_presentations()

    manager = _manager()
    original = manager._support_surfaces["living-room"]
    payload = original.model_dump(mode="json")
    payload["surfaces"][0]["offset_m"] -= 0.01
    manager._support_surfaces["living-room"] = SupportSurfaceSet.model_validate(payload)
    with pytest.raises(CalibrationValidationError, match="support plane is missing or changed"):
        manager.world_frame_presentations()


def test_old_camera_marker_y_is_only_a_finite_diagnostic() -> None:
    bundle = json.loads(BUNDLE.read_text())
    similarity = bundle["align"]["scene_similarity"]
    binding = bundle["cameras"]["frame_bindings"]["living-room"]
    anchor = next(row for row in similarity["correspondences"] if row["camera_id"] == "living-room")
    changed_anchor = dict(anchor)
    changed_anchor["scene_position"] = list(anchor["scene_position"])
    changed_anchor["scene_position"][1] += 1000.0
    entry = derive_entry(
        camera_id="living-room", binding=binding, similarity=similarity,
        correspondence=changed_anchor,
        calibration_revision=binding["camera_calibration_sha256"],
    )
    assert entry["constraints"]["camera_marker_height_residual_m"] > 1.0
    assert entry["constraints"]["lower_floor_scene_y"] == 0.0
