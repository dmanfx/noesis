from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def test_ds9_public_metadata_attach_paths_receive_owning_batch() -> None:
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        assert Path(hooks.__file__).resolve().is_relative_to(
            (Path.cwd() / "DS9").resolve()
        )

        frame_meta = SimpleNamespace()
        batch_meta = SimpleNamespace(frame_items=[frame_meta])
        operator_calls = []

        class Processor:
            def handle_servicemaker_frame(self, batch, frame):
                operator_calls.append((batch, frame))

        hooks._PoseFeatureOperator(Processor()).handle_metadata(batch_meta)
        hooks._ObjectDepthFusionOperator(Processor()).handle_metadata(batch_meta)
        assert operator_calls == [
            (batch_meta, frame_meta),
            (batch_meta, frame_meta),
        ]

        pipeline = SimpleNamespace(
            stable_id_mgr=SimpleNamespace(active_tracks={}),
            frame_size=(1920, 1080),
            config={},
        )
        pose = hooks.PoseFeatureProcessor(pipeline=pipeline, gie_id=4)
        pose_obj = SimpleNamespace()
        pose_calls = []

        def attach_pose(batch, obj, payload_json, replace_existing):
            pose_calls.append((batch, obj, json.loads(payload_json), replace_existing))
            return True

        assert pose._attach_pose_payload(
            batch_meta,
            attach_pose,
            pose_obj,
            {"score": 0.9},
        )
        assert pose_calls == [
            (batch_meta, pose_obj, {"score": 0.9}, True),
        ]

        depth = hooks._ObjectDepthFusionProcessor(
            depth_store=hooks._AlignedDepthFrameStore(),
            fallback_frame_size=(1920, 1080),
            depth_model_name="depth-anything-v2-metric-hypersim-vits",
            depth_unit="m",
            depth_is_metric=True,
            depth_every_n_frames=2,
        )
        depth_obj = SimpleNamespace()
        depth_calls = []

        def attach_depth(batch, obj, payload_json, replace_existing):
            depth_calls.append((batch, obj, json.loads(payload_json), replace_existing))
            return True

        hooks.noesis_depth_meta_ext = SimpleNamespace(
            attach_object_depth=attach_depth,
        )
        assert depth._attach_object_depth_payload(
            batch_meta,
            depth_obj,
            {"status": "ok", "depth_median": 2.5},
        )
        assert depth_calls == [
            (
                batch_meta,
                depth_obj,
                {"status": "ok", "depth_median": 2.5},
                True,
            ),
        ]
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
