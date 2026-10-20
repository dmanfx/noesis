from __future__ import annotations

import importlib.util
import copy
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


def test_finite_replay_explicitly_disables_inherited_loop_without_mutating_baseline(tmp_path: Path) -> None:
    module = _module()
    baseline = {
        "sources": [
            {"uri_secret": "living-room", "file-loop": True, "latency": 123},
            {"uri_secret": "kitchen", "gpu-id": 0},
            {"uri_secret": "family-room", "file-loop": False, "dewarper": {"enable": True}},
        ],
        "streammux": {"live-source": 1, "batch-size": 3, "width": 1920},
        "models": {"selected": "unchanged"},
        "tracker": {"quality": "unchanged"},
    }
    before = copy.deepcopy(baseline)
    videos = _videos(tmp_path)
    rendered = module.render_replay_config(baseline, videos)
    assert all(source.get("file-loop") is False for source in rendered["sources"])
    assert baseline == before
    expected = copy.deepcopy(before)
    for source, camera_id in zip(expected["sources"], module.CANONICAL_CAMERA_ORDER):
        source.pop("uri_secret")
        source.update(camera_id=camera_id, uri=videos[camera_id].resolve().as_uri())
        source["file-loop"] = False
    expected["streammux"]["live-source"] = 0
    expected["recorded_replay"] = {"realtime": True, "preserve_frames": True}
    assert rendered == expected


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


def test_render_resolves_support_surface_path_from_ds9_config_scope(tmp_path: Path) -> None:
    module = _module()
    ds9_root = tmp_path / "checkout" / "DS9"
    baseline_path = ds9_root / "config" / "infer.yaml"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text("", encoding="utf-8")
    baseline = {
        "sources": [{}, {}, {}],
        "streammux": {},
        "support_surfaces": {"path": "config/support_surfaces.json"},
    }

    rendered = module.render_replay_config(
        baseline,
        _videos(tmp_path),
        baseline_path=baseline_path,
    )

    assert rendered["support_surfaces"]["path"] == str(
        (ds9_root / "config" / "support_surfaces.json").resolve()
    )
