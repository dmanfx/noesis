from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOM_IDS = {
    "0": "LivingRoom",
    "1": "Kitchen",
    "2": "FamilyRoom",
}


def _post_stage(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload["analytics"]["stages"]["post"]


def _post_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    with path.open("r", encoding="utf-8") as stream:
        parser.read_file(stream)
    return parser


def _flatten(points: list[list[int]]) -> str:
    return ";".join(str(value) for point in points for value in point)


def test_overcrowding_is_the_single_room_designation_in_ds8_and_ds9() -> None:
    ds8_post = _post_stage(REPO_ROOT / "config" / "nvdsanalytics.yaml")
    ds9_post = _post_stage(REPO_ROOT / "DS9" / "config" / "nvdsanalytics.yaml")
    assert ds9_post == ds8_post

    ds8_ini_path = REPO_ROOT / "config" / "config_nvdsanalytics_post.ini"
    ds9_ini_path = REPO_ROOT / "DS9" / "config" / "config_nvdsanalytics_post.ini"
    assert ds9_ini_path.read_text(encoding="utf-8") == ds8_ini_path.read_text(
        encoding="utf-8"
    )
    ini = _post_ini(ds8_ini_path)

    assert set(ds8_post["streams"]) == set(EXPECTED_ROOM_IDS)
    for stream_id, room_id in EXPECTED_ROOM_IDS.items():
        stream = ds8_post["streams"][stream_id]
        assert "roi_filtering" not in stream

        overcrowding = stream["overcrowding"]
        assert overcrowding["enable"] is True
        assert overcrowding["class_ids"] == [0]
        assert overcrowding["object_threshold"] == 1
        assert overcrowding["roi"]["id"] == room_id

        section_name = f"overcrowding-stream-{stream_id}"
        section = ini[section_name]
        roi_key = f"roi-{room_id}"
        assert section["enable"] == "1"
        assert section["class-id"] == "0"
        assert section["object-threshold"] == "1"
        assert section[roi_key] == _flatten(overcrowding["roi"]["points_px"])
        assert f"roi-filtering-stream-{stream_id}" not in ini
