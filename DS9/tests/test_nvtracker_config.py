from pathlib import Path

import yaml


CONFIG = Path(__file__).resolve().parents[1] / "config" / "nvtracker.yaml"


def test_nvdcf_uses_native_cascaded_matching_without_another_tracker() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["DataAssociator"] == {
        "dataAssociatorType": 0,
        "associationMatcherType": 1,
        "checkClassMatch": 1,
        "minMatchingScore4Overall": 0.4,
        "minMatchingScore4SizeSimilarity": 0.35,
        "minMatchingScore4Iou": 0.2,
        "minMatchingScore4VisualSimilarity": 0.42,
        "matchingScoreWeight4VisualSimilarity": 0.20,
        "matchingScoreWeight4SizeSimilarity": 0.35,
        "matchingScoreWeight4Iou": 0.45,
        "tentativeDetectorConfidence": 0.2,
        "minMatchingScore4TentativeIou": 0.5,
    }
    assert config["VisualTracker"]["visualTrackerType"] == 1  # Existing NvDCF.
    assert config["StateEstimator"]["stateEstimatorType"] == 1
    assert config["AppearanceModel"]["useReidFeature"] == 0
    assert config["TrajectoryManagement"] == {"useUniqueID": 0}
    assert set(config) == {
        "BaseConfig", "TargetManagement", "TrajectoryManagement", "DataAssociator",
        "StateEstimator", "VisualTracker", "AppearanceModel", "ShadowTracker", "Misc",
    }


def test_nvdcf_baseline_preserves_native_lifetime_and_metadata_limits() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["BaseConfig"] == {"minDetectorConfidence": 0.0}
    assert config["TargetManagement"] == {
        "enableBboxUnClipping": 1,
        "preserveStreamUpdateOrder": 0,
        "maxTargetsPerStream": 32,
        "minIouDiff4NewTarget": 0.55,
        "minTrackerConfidence": 0.2,
        "probationAge": 2,
        "maxShadowTrackingAge": 90,
        "earlyTerminationAge": 6,
        "outputShadowTracks": 1,
        "outputTerminatedTracks": 1,
    }


def test_nvdcf_baseline_preserves_selected_state_and_visual_features() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["StateEstimator"] == {
        "stateEstimatorType": 1,
        "processNoiseVar4Loc": 2.0,
        "processNoiseVar4Size": 1.0,
        "processNoiseVar4Vel": 0.1,
        "measurementNoiseVar4Detector": 4.0,
        "measurementNoiseVar4Tracker": 16.0,
    }
    assert config["VisualTracker"] == {
        "visualTrackerType": 1,
        "useColorNames": 1,
        "useHog": 0,
        "featureImgSizeLevel": 2,
        "featureFocusOffsetFactor_y": -0.1,
        "filterLr": 0.075,
        "filterChannelWeightsLr": 0.015,
        "gaussianSigma": 0.75,
    }
    assert config["AppearanceModel"] == {
        "useReidFeature": 0, "reidFeatureSize": 128, "reidHistorySize": 64,
    }
    assert config["ShadowTracker"] == {
        "targetVisibilityThreshold": 0.15,
        "minDetectorConfidence": 0.0,
        "minTargetBboxSize": 10,
    }
    assert config["Misc"] == {"locUncertaintyThreshold": 50.0}
