from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from noesis.pipelines import hooks
from noesis.server import analytics_api


REPO_ROOT = Path(__file__).resolve().parents[2]


class _StableIDManager:
    def __init__(self) -> None:
        self.present_calls: list[tuple[int, list[int], float]] = []
        self.prune_calls: list[float] = []

    def remove_missing_tracks(
        self, sensor_id: int, present_track_ids: list[int], ts: float
    ) -> None:
        self.present_calls.append(
            (int(sensor_id), sorted(int(value) for value in present_track_ids), float(ts))
        )

    def prune_ghosts(self, ts: float) -> None:
        self.prune_calls.append(float(ts))


def _processor(*, tracking_mode: str, grace_s: float) -> tuple[object, _StableIDManager]:
    manager = _StableIDManager()
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.pipeline = SimpleNamespace(stable_id_mgr=manager)
    processor._stable_id_enabled = True
    processor._tracking_mode = tracking_mode
    processor._v3dt_reid_track_grace_s = grace_s
    processor._v3dt_reid_last_seen_by_track = {}
    return processor, manager


def test_living_room_profile_enables_bounded_v3dt_reid_grace() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "DS9/config/infer_v3dt_living_room_optimized.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["v3dt"]["reid_track_grace_s"] == 0.5


def test_room_profiles_keep_bounded_grace_and_resolvable_camera_models() -> None:
    expected_analytics = (
        "DS9/config/v3dt/household_tracking_exclusions/nvdsanalytics.yaml"
    )
    expected_exclude = (
        "DS9/config/v3dt/household_tracking_exclusions/"
        "config_nvdsanalytics_exclude.ini"
    )
    for pipeline_name in (
        "infer_v3dt_living_family_phone_optimized.yaml",
        "infer_v3dt_living_kitchen_tracking_candidate.yaml",
    ):
        config = yaml.safe_load(
            (REPO_ROOT / "DS9/config" / pipeline_name).read_text(encoding="utf-8")
        )
        tracker_path = REPO_ROOT / config["tracker"]["config-file"]
        tracker = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))

        assert config["v3dt"]["reid_track_grace_s"] == 0.5
        assert config["v3dt"]["household_confirm_embeddings"] == 1
        assert config["analytics"]["stages_config"] == expected_analytics
        assert config["analytics"]["exclude"]["config-file"] == expected_exclude
        assert config["visualization"]["v3dt_cuboid"]["enabled"] is True
        assert len(tracker["ObjectModelProjection"]["cameraModelFilepath"]) == 3
        for relative_path in tracker["ObjectModelProjection"][
            "cameraModelFilepath"
        ]:
            assert (REPO_ROOT / relative_path).is_file()


def test_v3dt_exclusions_are_isolated_from_baseline_analytics() -> None:
    baseline = yaml.safe_load(
        (REPO_ROOT / "DS9/config/infer.yaml").read_text(encoding="utf-8")
    )
    living = yaml.safe_load(
        (
            REPO_ROOT / "DS9/config/infer_v3dt_living_room_optimized.yaml"
        ).read_text(encoding="utf-8")
    )

    assert baseline["analytics"]["stages_config"] == "config/nvdsanalytics.yaml"
    assert living["analytics"]["stages_config"].startswith("DS9/config/v3dt/")
    assert living["analytics"]["stages_config"] != baseline["analytics"]["stages_config"]


def test_v3dt_exclusion_yaml_and_runtime_ini_are_equivalent() -> None:
    profile_dir = REPO_ROOT / "DS9/config/v3dt/household_tracking_exclusions"
    stages = yaml.safe_load(
        (profile_dir / "nvdsanalytics.yaml").read_text(encoding="utf-8")
    )["analytics"]["stages"]
    expected = analytics_api._render_exclude_ini(stages["exclude"])
    actual = (profile_dir / "config_nvdsanalytics_exclude.ini").read_text(
        encoding="utf-8"
    )

    assert actual == expected


def test_v3dt_grace_retains_a_raw_track_only_for_the_configured_gap() -> None:
    processor, manager = _processor(tracking_mode="v3dt", grace_s=0.5)

    processor._maintain_stable_ids(0, [7], 10.0)
    processor._maintain_stable_ids(0, [], 10.1)
    processor._maintain_stable_ids(0, [], 10.6)

    assert [present for _sensor, present, _ts in manager.present_calls] == [
        [7],
        [7],
        [],
    ]


def test_non_v3dt_lifecycle_remains_immediate() -> None:
    processor, manager = _processor(tracking_mode="baseline", grace_s=0.5)

    processor._maintain_stable_ids(0, [7], 10.0)
    processor._maintain_stable_ids(0, [], 10.1)

    assert [present for _sensor, present, _ts in manager.present_calls] == [[7], []]


def test_v3dt_grace_does_not_cross_sensor_boundaries() -> None:
    processor, manager = _processor(tracking_mode="v3dt", grace_s=0.5)

    processor._maintain_stable_ids(0, [7], 10.0)
    processor._maintain_stable_ids(1, [], 10.1)

    assert manager.present_calls[-1][0:2] == (1, [])
