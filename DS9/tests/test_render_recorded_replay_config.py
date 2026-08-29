from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "DS9" / "scripts" / "render_recorded_replay_config.py"


def _module():
    spec = importlib.util.spec_from_file_location("render_recorded_replay_config", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _videos(tmp_path: Path) -> dict[str, Path]:
    videos = {}
    for camera_id in ("living-room", "kitchen", "family-room"):
        video = tmp_path / f"{camera_id}.mp4"
        video.write_bytes(b"video")
        videos[camera_id] = video
    return videos


def test_render_keeps_pipeline_quality_and_only_rebinds_sources(tmp_path: Path) -> None:
    module = _module()
    baseline = yaml.safe_load((ROOT / "DS9" / "config" / "infer.yaml").read_text(encoding="utf-8"))
    videos = _videos(tmp_path)
    baseline_path = ROOT / "DS9" / "config" / "infer.yaml"
    rendered = module.render_replay_config(
        baseline,
        videos,
        baseline_path=baseline_path,
    )

    assert rendered["models"] == baseline["models"]
    assert rendered["tracker"] == baseline["tracker"]
    assert rendered["canonical_world"] == baseline["canonical_world"]
    assert rendered["streammux"]["live-source"] == 0
    assert rendered["recorded_replay"] == {"realtime": True, "preserve_frames": True}
    for index, camera_id in enumerate(module.CANONICAL_CAMERA_ORDER):
        assert "uri_secret" not in rendered["sources"][index]
        assert rendered["sources"][index]["uri"] == videos[camera_id].resolve().as_uri()
    assert rendered["scene_priors"]["path"] == str((ROOT / "data" / "scene_priors" / "catalog.json").resolve())
    assert rendered["analytics"]["config-file"] == str(
        (ROOT / "DS9" / "config" / "config_nvdsanalytics_post.ini").resolve()
    )


def test_render_requires_one_video_for_every_canonical_camera(tmp_path: Path) -> None:
    module = _module()
    baseline = yaml.safe_load((ROOT / "DS9" / "config" / "infer.yaml").read_text(encoding="utf-8"))
    videos = _videos(tmp_path)
    videos.pop("family-room")
    with pytest.raises(module.ReplayConfigError, match="missing cameras: family-room"):
        module.render_replay_config(baseline, videos)
