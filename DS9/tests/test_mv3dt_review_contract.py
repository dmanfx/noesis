from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from noesis.v3dt_assets import validate_v3dt_assets


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"
PIPELINE_PATH = DS9_ROOT / "config" / "infer_mv3dt_kitchen_family_review.yaml"
PROFILE_ROOT = DS9_ROOT / "config" / "v3dt" / "kitchen_family_review"


def test_review_profile_is_isolated_and_explicit() -> None:
    canonical = yaml.safe_load(
        (DS9_ROOT / "config" / "infer_mv3dt.yaml").read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(PIPELINE_PATH.read_text(encoding="utf-8"))

    assert canonical["v3dt"]["activation_state"] == "ready_opt_in"
    assert canonical["v3dt"]["geometry_authority"] == "accepted"
    assert candidate["v3dt"] == {
        "profile": "mv3dt",
        "activation_state": "evaluation_only",
        "evaluation_scope": "kitchen-family",
        "geometry_authority": "review_only",
        "geometry_binding": (
            "DS9/config/v3dt/kitchen_family_review/geometry_binding.json"
        ),
        "world_frame": "backend_world_m",
        "caminfo_world_axes": "xzy",
        "camera_order": ["living-room", "kitchen", "family-room"],
        "reid_track_grace_s": 0.5,
        "household_confirm_embeddings": 1,
    }
    assert candidate["tracker"]["config-file"] == canonical["tracker"]["config-file"]
    assert candidate["v3dt"]["geometry_binding"] != canonical["v3dt"]["geometry_binding"]
    assert canonical["streammux"]["live-source"] == 1
    assert canonical["streammux"]["batched-push-timeout"] == -1
    assert candidate["streammux"]["sync-inputs"] == 0
    assert candidate["streammux"]["batched-push-timeout"] == -1
    tracker = yaml.safe_load(
        (PROFILE_ROOT / "nvtracker_mv3dt.yaml").read_text(encoding="utf-8")
    )
    assert tracker["BaseConfig"]["useBatchNumForFrameId"] == 1
    assert tracker["TargetManagement"]["probationAge"] == 2
    assert tracker["MultiViewAssociator"]["enableMsgSync"] == 1
    assert tracker["MultiViewAssociator"]["minCommonFrames4MatchScore"] == 2
    assert tracker["MultiViewAssociator"]["maxPeerToPredDistance4Fusion"] == 4.75
    assert tracker["MultiViewAssociator"]["minPeerVisibility4Fusion"] == 0.05
    assert tracker["MultiViewAssociator"]["minPeerTrackletMatchScore"] == 0.18
    for camera_model in tracker["ObjectModelProjection"]["cameraModelFilepath"]:
        model = yaml.safe_load((REPO_ROOT / camera_model).read_text(encoding="utf-8"))
        assert model["modelInfo"] == {"height": 1.7, "radius": 0.35}
    mqtt_template = (DS9_ROOT / "config/v3dt/config_mqtt.template.txt").read_text(
        encoding="utf-8"
    )
    assert "share-connection = 1" in mqtt_template
    assert "set-threaded = 0" in mqtt_template


def test_review_profile_validates_only_as_evaluation_mv3dt() -> None:
    bundle = validate_v3dt_assets(
        PIPELINE_PATH,
        cameras_config=DS9_ROOT / "config" / "cameras_v3dt.yaml",
        require_engines=False,
        require_sources=False,
        expected_profile="mv3dt",
        expected_activation_state="evaluation_only",
    )

    assert bundle.tracker_config == PROFILE_ROOT / "nvtracker_mv3dt.yaml"
    assert [path.name for path in bundle.camera_models] == [
        "camInfo_living-room.yml",
        "camInfo_kitchen.yml",
        "camInfo_family-room.yml",
    ]


def test_canonical_mv3dt_profile_is_ready_but_still_explicit_opt_in() -> None:
    canonical_path = DS9_ROOT / "config" / "infer_mv3dt.yaml"
    bundle = validate_v3dt_assets(
        canonical_path,
        cameras_config=DS9_ROOT / "config" / "cameras_v3dt.yaml",
        require_engines=False,
        require_sources=False,
        expected_profile="mv3dt",
        expected_activation_state="ready_opt_in",
    )
    config = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    binding = json.loads(
        (PROFILE_ROOT / "geometry_binding_accepted.json").read_text(encoding="utf-8")
    )

    assert bundle.tracker_config == PROFILE_ROOT / "nvtracker_mv3dt.yaml"
    assert config["v3dt"]["activation_state"] == "ready_opt_in"
    assert binding["accepted"] is True
    assert binding["canonical_use"] is True
    assert binding["status"] == "accepted"
    np.testing.assert_array_equal(
        np.asarray(
            binding["moving_correction_in_backend_world_row_major"],
            dtype=np.float64,
        ),
        np.eye(4, dtype=np.float64),
    )
    assert binding["yaw_deg"] == 0.0
    assert binding["calibration_config"].endswith(
        "kitchen_family_review/camera_calibration.json"
    )
    assert "source_report_storage_relative" not in binding


def test_only_kitchen_and_family_are_peers() -> None:
    pub_sub = yaml.safe_load(
        (PROFILE_ROOT / "pub_sub_info_config_0.yml").read_text(encoding="utf-8")
    )
    living, kitchen, family = pub_sub["pubBrokerTopicStr"]

    assert pub_sub["subPeerBrokerTopicStrs"] == [[living], [family], [kitchen]]
    assert pub_sub["subPeerBrokerTopicStrs"][0] == [living]
    assert living not in {
        topic
        for subscriptions in pub_sub["subPeerBrokerTopicStrs"][1:]
        for topic in subscriptions
    }


def test_kitchen_calibration_is_composed_into_family_gauge() -> None:
    source = json.loads(
        (
            DS9_ROOT
            / "config/v3dt/living_family_phone_optimized/camera_calibration.json"
        ).read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (PROFILE_ROOT / "camera_calibration.json").read_text(encoding="utf-8")
    )
    binding = json.loads(
        (PROFILE_ROOT / "geometry_binding.json").read_text(encoding="utf-8")
    )

    transform = np.asarray(
        binding["moving_correction_in_backend_world_row_major"], dtype=np.float64
    )
    source_e = np.asarray(
        source["cameras"]["kitchen"]["E"], dtype=np.float64
    ).reshape((4, 4), order="F")
    candidate_e = np.asarray(
        candidate["cameras"]["kitchen"]["E"], dtype=np.float64
    ).reshape((4, 4), order="F")

    np.testing.assert_allclose(
        candidate_e,
        source_e @ np.linalg.inv(transform),
        atol=1e-12,
    )
    assert candidate["cameras"]["living-room"] == source["cameras"]["living-room"]
    assert candidate["cameras"]["kitchen"]["pose"]["source"] == (
        "kitchen_phone_walk_static_camera_localization_2026-08-19"
    )
    assert candidate["cameras"]["family-room"]["pose"]["source"] == (
        "family_scene_prior_static_camera_anchor_v3_2026-08-17"
    )
    assert binding["static_camera_anchors"]["kitchen"]["status"] == (
        "passed_review_anchor"
    )
    assert binding["static_camera_anchors"]["family-room"]["status"] == (
        "passed_review_anchor"
    )
