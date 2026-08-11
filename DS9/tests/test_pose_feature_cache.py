from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def test_ds9_pose_feature_processor_constructs_and_reuses_track_cache() -> None:
    """Exercise the DS9-owned processor in an import-isolated interpreter."""

    script = textwrap.dedent(
        """
        from pathlib import Path
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        hooks_path = Path(hooks.__file__).resolve()
        expected_root = (Path.cwd() / "DS9").resolve()
        assert hooks_path.is_relative_to(expected_root), hooks_path

        pipeline = SimpleNamespace(
            stable_id_mgr=SimpleNamespace(active_tracks={}),
            frame_size=(1920, 1080),
            config={},
        )
        processor = hooks.PoseFeatureProcessor(pipeline=pipeline, gie_id=4)
        assert processor._pose_cache == {}

        bbox = [32.0, 40.0, 120.0, 240.0]
        assert processor._cached_pose_payload(0, 17, bbox, 1, 1000, None) is None

        processor._cache_pose_payload(
            0,
            17,
            bbox,
            1,
            {"score": 0.93, "keypoints_roi": []},
        )
        reused = processor._cached_pose_payload(0, 17, bbox, 2, 2000, None)
        assert reused is not None
        assert reused["pose_cache_reused"] is True
        assert reused["pose_cache_age_frames"] == 1
        assert reused["object_id"] == 17

        second = hooks.PoseFeatureProcessor(pipeline=pipeline, gie_id=4)
        assert second._pose_cache == {}
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(DS9_ROOT), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
